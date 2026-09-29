import time

import cv2
import numpy as np
import pytest

from fc_comparator.classify import TemplateClassifier, create_classifier, crop_roi
from fc_comparator.classify.dataset import dataset_counts, dataset_warnings, harvest_board, save_crop
from fc_comparator.config import AppConfig, TemplateConfig
from fc_comparator.models import Roi
from fc_comparator.synthetic import layout_rois, render_board

from .conftest import random_columns

ALL = ["round", "fork_left", "fork_right", "small", "missing"]


def crops_of(img, rois):
    return [crop_roi(img, r) for r in rois]


def test_dataset_layout(dataset_dir):
    counts = dataset_counts(dataset_dir)
    assert set(counts) == {"round", "fork_left", "small", "missing"}
    assert sum(counts.values()) == 8 * 16
    assert dataset_warnings(dataset_dir) == []


def test_template_classifier_accuracy_on_unseen_boards(dataset_dir):
    clf = TemplateClassifier(TemplateConfig(), dataset_dir)
    assert set(clf.labels) == set(ALL)  # fork_right derived by mirroring
    rng = np.random.default_rng(7)
    rois = layout_rois()
    total = correct = 0
    for b in range(4):
        cols = random_columns(rng, ALL)
        img = render_board(cols, seed=900 + b)
        for roi, res in zip(rois, clf.classify(crops_of(img, rois))):
            truth = cols[roi.cable - 1][roi.row - 1]
            total += 1
            correct += res.label == truth
            if res.label == truth:
                assert res.confidence > 0.6, (truth, res)
    assert correct == total


def test_fork_orientation_is_distinguished(dataset_dir):
    clf = TemplateClassifier(TemplateConfig(), dataset_dir)
    img = render_board([["fork_left"] * 4, ["fork_right"] * 4], seed=42)
    res = clf.classify(crops_of(img, layout_rois(2, 4)))
    assert [r.label for r in res] == ["fork_left"] * 4 + ["fork_right"] * 4


def test_unknown_object_gets_low_confidence(dataset_dir):
    clf = TemplateClassifier(TemplateConfig(), dataset_dir)
    rng = np.random.default_rng(0)
    weird = [
        rng.integers(0, 255, (120, 120, 3), dtype=np.uint8),  # noise
        np.full((120, 120, 3), 30, np.uint8),  # something covering the camera
    ]
    star = np.full((120, 120, 3), 246, np.uint8)
    cv2.drawMarker(star, (60, 60), (20, 20, 20), cv2.MARKER_TILTED_CROSS, 90, 12)
    weird.append(star)
    for res in clf.classify(weird):
        assert res.confidence < 0.6, res


def test_few_templates_ambiguity_becomes_low_confidence():
    """A fork that looks round-ish must not be confidently called round."""
    cfg = TemplateConfig()
    clf = TemplateClassifier(cfg)
    base = render_board([["round", "fork_left"]], seed=1, noise=0)
    rois = layout_rois(1, 2)
    clf.add("round", crop_roi(base, rois[0]))
    clf.add("fork_left", crop_roi(base, rois[1]))
    # 50/50 blend of a round and a fork clip: genuinely ambiguous
    blend = cv2.addWeighted(crop_roi(base, rois[0]), 0.5, crop_roi(base, rois[1]), 0.5, 0)
    res = clf.classify([blend])[0]
    assert res.confidence < 0.6


def test_empty_classifier_is_uncertain():
    clf = TemplateClassifier(TemplateConfig())
    assert not clf.ready
    assert clf.classify([np.zeros((50, 50, 3), np.uint8)])[0].label == "uncertain"


def test_vectorized_scores_equal_opencv_matchtemplate(dataset_dir):
    clf = TemplateClassifier(TemplateConfig(max_templates_per_class=5), dataset_dir)
    crop = crop_roi(render_board([["fork_right"]], seed=11), layout_rois(1, 1)[0])
    s, m = clf.cfg.size, clf.cfg.search_margin
    probe = clf._prep(crop, s + 2 * m)
    for label, temps in clf.templates.items():
        vals = sorted((float(cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED).max()) for t in temps), reverse=True)
        assert clf.class_scores(crop)[label] == pytest.approx(np.mean(vals[:3]), abs=1e-3)


def test_template_speed(dataset_dir):
    clf = TemplateClassifier(TemplateConfig(), dataset_dir)
    crops = crops_of(render_board([["round", "fork_left", "small", "missing"]] * 4, seed=3), layout_rois())
    clf.classify(crops)
    t = time.perf_counter()
    clf.classify(crops)
    assert time.perf_counter() - t < 0.5


def test_crop_roi_clamps():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop_roi(img, Roi(1, 1, 90, 90, 20, 20)).shape == (10, 10, 3)
    assert crop_roi(img, Roi(1, 1, 10, 10, 20, 20), padding=0.5).shape == (40, 40, 3)
    assert crop_roi(img, Roi(1, 1, 200, 200, 20, 20)).shape == (20, 20, 3)


def test_harvest_requires_fork_orientation(tmp_path):
    img = render_board([["fork_left"] * 4], seed=1)
    with pytest.raises(ValueError):
        harvest_board(img, layout_rois(1, 4), ["fork"] * 4, tmp_path, fork_orientation=None)
    paths = harvest_board(img, layout_rois(1, 4), ["fork", "fork", "fork", "fork"], tmp_path, fork_orientation="left")
    assert len(paths) == 4 and all(p.parent.name == "fork_left" for p in paths)
    with pytest.raises(ValueError):
        save_crop(tmp_path, "banana", img)


def test_factory_falls_back_to_templates(tmp_path, dataset_dir):
    cfg = AppConfig(base_dir=tmp_path)
    cfg.classifier.model_path = "does_not_exist.pt"
    cfg.classifier.dataset_dir = str(dataset_dir)
    clf = create_classifier(cfg)
    assert clf.name == "template" and clf.ready
    cfg.classifier.backend = "yolo"
    with pytest.raises(FileNotFoundError):
        create_classifier(cfg)
