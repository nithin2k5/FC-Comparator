import time

import cv2
import numpy as np
import pytest

from fc_comparator.align import Aligner
from fc_comparator.config import AlignmentConfig
from fc_comparator.synthetic import render_board

REF_COLS = [["round", "fork_left", "small", "round"]] * 4
# Same layout as the reference but a different part mixed in: alignment must not depend on the clips.
TEST_COLS = [["round", "fork_left", "small", "round"], ["fork_left", "fork_left", "small", "fork_left"]] * 2


def fiducial_error(aligned: np.ndarray, reference: np.ndarray) -> float:
    """Mean abs difference over the static board regions (fiducials, connectors)."""
    boxes = [(40, 40, 90, 90), (1150, 40, 90, 90), (215, 95, 90, 90), (995, 95, 90, 90), (40, 830, 90, 90)]
    diffs = [
        np.abs(aligned[y:y + h, x:x + w].astype(int) - reference[y:y + h, x:x + w].astype(int)).mean()
        for x, y, w, h in boxes
    ]
    return float(np.mean(diffs))


@pytest.fixture(scope="module")
def reference():
    return render_board(REF_COLS, seed=0)


@pytest.mark.parametrize(
    "shift,angle",
    [((20, 0), 0), ((0, -20), 0), ((-20, 15), 0), ((14, 14), 1.5), ((-25, -10), -2.0)],
)
def test_realigns_shifted_board(reference, shift, angle):
    moved = render_board(TEST_COLS, shift=shift, angle=angle, seed=5)
    aligner = Aligner(AlignmentConfig(), reference)
    aligned, info = aligner.align(moved)
    assert info.ok and info.applied, info.message
    assert info.shift_px == pytest.approx(np.hypot(*shift), abs=4)
    before = fiducial_error(moved, reference)
    after = fiducial_error(aligned, reference)
    assert after < 8 < before, (before, after)


def test_unshifted_board_is_nearly_identity(reference):
    aligned, info = Aligner(AlignmentConfig(), reference).align(render_board(TEST_COLS, seed=9))
    assert info.ok and info.shift_px < 2


def test_rejects_large_shift(reference):
    cfg = AlignmentConfig(max_shift_px=30)
    moved = render_board(TEST_COLS, shift=(60, 0), seed=5)
    img, info = Aligner(cfg, reference).align(moved)
    assert not info.ok and "shift" in info.message
    assert img is moved


def test_fails_on_blank_image(reference):
    blank = np.full_like(reference, 240)
    _, info = Aligner(AlignmentConfig(), reference).align(blank)
    assert not info.ok


def test_size_mismatch_and_no_reference(reference):
    _, info = Aligner(AlignmentConfig()).align(reference)
    assert not info.ok and "reference" in info.message
    small = cv2.resize(reference, (640, 480))
    _, info = Aligner(AlignmentConfig(), reference).align(small)
    assert not info.ok and "size" in info.message


def test_alignment_is_fast(reference):
    aligner = Aligner(AlignmentConfig(), reference)
    moved = render_board(TEST_COLS, shift=(20, 20), seed=5)
    aligner.align(moved)
    t = time.perf_counter()
    for _ in range(5):
        aligner.align(moved)
    assert (time.perf_counter() - t) / 5 < 0.3
