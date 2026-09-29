"""Generate the synthetic sample set used by the demo config and the end-to-end test.

    python scripts/make_samples.py            # writes samples/

Outputs
  samples/reference.jpg          reference board for ROI teaching + alignment
  samples/boards/*.jpg           test boards
  samples/manifest.yaml          part number + expected NG positions per board
  samples/dataset/<class>/*.jpg  teach crops (template set / YOLO training data)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fc_comparator.capture.sources import write_image  # noqa: E402
from fc_comparator.classify import crop_roi  # noqa: E402
from fc_comparator.classify.dataset import save_crop  # noqa: E402
from fc_comparator.synthetic import layout_rois, render_board  # noqa: E402

P001 = ["round", "fork_left", "small", "round"]  # master: round, fork, small, round (forks face left)
P002 = ["fork_right", "fork_right", "small", "fork_right"]  # master: fork, fork, small, fork (forks face right)


def with_(cols, cable, row, label):
    cols = [list(c) for c in cols]
    cols[cable - 1][row - 1] = label
    return cols


# name, part, columns, render kwargs, expected NG [(cable, row)], note
BOARDS = [
    ("board_P001_ok", "P001", [P001] * 4, {}, [], "all correct"),
    ("board_P002_ok", "P002", [P002] * 4, {}, [], "all correct"),
    ("board_P001_mixed", "P001", [P001, P001, P002, P001], {}, [(3, 1), (3, 4)], "cable 3 is a P002 harness"),
    ("board_P002_mixed", "P002", [P001, P002, P002, P002], {}, [(1, 1), (1, 4)], "cable 1 is a P001 harness"),
    ("board_P001_missing", "P001", with_([P001] * 4, 2, 3, "missing"), {}, [(2, 3)], "clip missing on cable 2 row 3"),
    ("board_P001_shifted_ok", "P001", [P001] * 4, {"shift": (20, -15), "angle": 1.0}, [], "board shifted 20/-15 px, 1 deg"),
    (
        "board_P001_shifted_mixed",
        "P001",
        [P001, P002, P001, P001],
        {"shift": (-18, 20)},
        [(2, 1), (2, 4)],
        "shifted -18/20 px, cable 2 is P002",
    ),
    (
        "board_P002_wrong_clip",
        "P002",
        with_(with_([P002] * 4, 4, 2, "round"), 3, 3, "fork_right"),
        {},
        [(4, 2), (3, 3)],
        "round instead of fork at C4R2, fork instead of small at C3R3",
    ),
]

TEACH_SEEDS = range(500, 512)


def teach_columns(rng):
    labels = ["round", "fork_left", "fork_right", "small", "missing"]
    return [[str(rng.choice(labels)) for _ in range(4)] for _ in range(4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "samples")
    args = ap.parse_args()
    out: Path = args.out
    for sub in ("boards", "dataset"):
        shutil.rmtree(out / sub, ignore_errors=True)

    write_image(out / "reference.jpg", render_board([P001] * 4, seed=1), quality=92)

    manifest = []
    for i, (name, part, cols, kw, expected, note) in enumerate(BOARDS):
        img = render_board(cols, seed=100 + i, **kw)
        write_image(out / "boards" / f"{name}.jpg", img, quality=92)
        manifest.append(
            {"image": f"boards/{name}.jpg", "part": part, "expected_ng": [list(p) for p in expected], "note": note}
        )

    # An obstructed position (a gloved finger over the clip): must be "uncertain", never OK.
    img = render_board([P001] * 4, seed=150)
    cv2.ellipse(img, (790, 480), (70, 45), 20, 0, 360, (150, 175, 205), -1, cv2.LINE_AA)
    cv2.ellipse(img, (790, 480), (70, 45), 20, 0, 360, (110, 130, 160), 3, cv2.LINE_AA)
    write_image(out / "boards" / "board_P001_obstructed.jpg", img, quality=92)
    manifest.append(
        {"image": "boards/board_P001_obstructed.jpg", "part": "P001", "expected_ng": [[3, 2]], "note": "unknown object at C3R2"}
    )

    (out / "manifest.yaml").write_text(
        "# Expected results for the sample boards (used by tests/test_e2e.py)\n"
        + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )

    # Teach dataset: labeled crops from boards with known content.
    rng = np.random.default_rng(7)
    rois = layout_rois()
    n = 0
    for seed in TEACH_SEEDS:
        cols = teach_columns(rng)
        img = render_board(cols, seed=seed, shift=tuple(rng.uniform(-2, 2, 2)))
        for roi in rois:
            label = cols[roi.cable - 1][roi.row - 1]
            path = save_crop(out / "dataset", label, crop_roi(img, roi), f"s{seed}c{roi.cable}r{roi.row}")
            # store as JPEG to keep the repository small
            jpg = path.with_suffix(".jpg")
            write_image(jpg, cv2.imread(str(path)), quality=92)
            path.unlink()
            n += 1
    print(f"Wrote {len(manifest)} boards and {n} teach crops to {out}")


if __name__ == "__main__":
    main()
