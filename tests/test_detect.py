import time

import cv2
import numpy as np
import pytest

from fc_comparator.annotations import AnnotationStore
from fc_comparator.config import AppConfig, TemplateDetectorConfig
from fc_comparator.detect import TemplateDetector, create_detector, nms
from fc_comparator.models import Box, Taxonomy
from fc_comparator.synthetic import render_marked_board

from .conftest import random_columns


@pytest.fixture(scope="module")
def detector(marked_store):
    return TemplateDetector(TemplateDetectorConfig(), Taxonomy(), marked_store)


def match(dets, gt, iou=0.5):
    """(correct, wrong_label, missed, false_positives)"""
    used, correct, wrong, missed = set(), 0, 0, 0
    for g in gt:
        cand = [(d.iou(g), j) for j, d in enumerate(dets) if j not in used]
        best = max(cand, default=(0.0, -1))
        if best[0] >= iou:
            used.add(best[1])
            correct += dets[best[1]].label == g.label
            wrong += dets[best[1]].label != g.label
        else:
            missed += 1
    return correct, wrong, missed, len(dets) - len(used)


def test_learns_from_a_few_marked_images(detector):
    assert set(detector.labels) == {"round", "fork_left", "fork_right", "small"}
    assert detector.negatives  # automatic "not a clip" patches


def test_detects_every_clip_on_unseen_boards(detector):
    rng = np.random.default_rng(7)
    totals = np.zeros(4, int)
    for i in range(6):
        img, gt = render_marked_board(random_columns(rng), seed=900 + i,
                                      shift=tuple(rng.uniform(-25, 25, 2)), angle=float(rng.uniform(-2, 2)))
        dets = detector.detect(img)
        totals += match(dets, gt)
        for d in dets:
            assert d.confidence > 0.6, d
    correct, wrong, missed, fp = totals
    assert (wrong, missed, fp) == (0, 0, 0) and correct > 60


def test_fork_orientation(detector):
    img, gt = render_marked_board([["fork_left"] * 4, ["fork_right"] * 4], seed=42)
    dets = detector.detect(img)
    assert match(dets, gt) == (8, 0, 0, 0)


def test_missing_clip_and_obstruction_are_not_confident_clips(detector):
    cols = [["round", "fork_left", "small", "round"] for _ in range(4)]
    cols[1][2] = "missing"
    img, gt = render_marked_board(cols, seed=77)
    cv2.ellipse(img, (780, 470), (70, 45), 20, 0, 360, (150, 175, 205), -1, cv2.LINE_AA)  # finger over C3R2
    dets = detector.detect(img)
    at = lambda x, y: [d for d in dets if abs(d.center[0] - x) < 40 and abs(d.center[1] - y) < 40]  # noqa: E731
    assert at(520, 640) == []  # nothing where the clip is missing
    assert all(d.confidence < 0.6 for d in at(780, 470))  # never a confident clip under the finger


def test_speed(detector):
    img = render_marked_board([["round", "fork_left", "small", "round"]] * 4, seed=3)[0]
    detector.detect(img)
    t = time.perf_counter()
    detector.detect(img)
    assert time.perf_counter() - t < 0.5


def test_empty_store(tmp_path):
    det = TemplateDetector(TemplateDetectorConfig(), Taxonomy(), AnnotationStore(tmp_path))
    assert not det.ready and det.detect(np.zeros((100, 100, 3), np.uint8)) == []


def test_nms_keeps_the_best_box():
    boxes = [Box("round", 0, 0, 10, 10, 0.6), Box("small", 1, 1, 10, 10, 0.9), Box("round", 50, 50, 10, 10, 0.5)]
    assert [b.confidence for b in nms(boxes)] == [0.9, 0.5]


def test_factory_falls_back_to_templates(tmp_path, marked_store):
    cfg = AppConfig(base_dir=tmp_path)
    cfg.detector.model_path = "missing.pt"
    cfg.annotations.dir = str(marked_store.root)
    assert create_detector(cfg).name == "template"
    cfg.detector.backend = "yolo"
    with pytest.raises(FileNotFoundError):
        create_detector(cfg)
    cfg.detector.backend = "bogus"
    with pytest.raises(ValueError):
        create_detector(cfg)
