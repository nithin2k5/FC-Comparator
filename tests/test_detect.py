import time

import cv2
import numpy as np
import pytest

from fc_comparator.config import AppConfig, TemplateDetectorConfig
from fc_comparator.core.models import Box, Taxonomy
from fc_comparator.vision.dataset import AnnotationStore
from fc_comparator.vision.detect import TemplateDetector, create_detector, nms
from fc_comparator.vision.synthetic import render_marked_board

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


def test_missing_clip_and_covered_clip(detector):
    cols = [["round", "fork_left", "small", "round"] for _ in range(4)]
    cols[1][2] = "missing"
    img, gt = render_marked_board(cols, seed=77)
    cv2.ellipse(img, (780, 470), (70, 45), 20, 0, 360, (150, 175, 205), -1, cv2.LINE_AA)  # finger over C3R2
    dets = detector.detect(img)
    near = lambda x, y, r: [d for d in dets if abs(d.center[0] - x) < r and abs(d.center[1] - y) < r]  # noqa: E731
    assert near(520, 640, 40) == []  # nothing where the clip is missing
    # The covered fork must not be "seen": whatever the finger looks like, it is not a confident
    # fork_left there (the station then reports C3R2 as missing or wrong - never OK; see test_e2e).
    assert not [d for d in near(760, 470, 60) if d.label == "fork_left" and d.confidence >= 0.6]


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


def test_auto_never_switches_to_an_undertrained_model(tmp_path, marked_store):
    import json

    cfg = AppConfig(base_dir=tmp_path)
    cfg.dataset.dir = str(marked_store.root)
    model = tmp_path / "models" / "clip_detector.pt"
    model.parent.mkdir()
    model.write_bytes(b"not really a model")
    report = model.with_name("clip_detector_report.json")
    report.write_text(json.dumps({"accuracy": 0.42}))
    det = create_detector(cfg, marked_store)
    assert det.name == "template" and "42%" in det.note and "train longer" in det.note
    report.write_text(json.dumps({"accuracy": 0.99}))  # good report but unreadable weights
    det = create_detector(cfg, marked_store)
    assert det.name == "template" and "could not be loaded" in det.note


def test_factory_falls_back_to_templates(tmp_path, marked_store):
    cfg = AppConfig(base_dir=tmp_path)
    cfg.detector.model_path = "missing.pt"
    cfg.dataset.dir = str(marked_store.root)
    assert create_detector(cfg).name == "template"
    cfg.detector.backend = "yolo"
    with pytest.raises(FileNotFoundError):
        create_detector(cfg)
    cfg.detector.backend = "bogus"
    with pytest.raises(ValueError):
        create_detector(cfg)
