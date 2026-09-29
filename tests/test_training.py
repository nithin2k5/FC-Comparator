import numpy as np
import pytest

from fc_comparator.annotations import AnnotationStore
from fc_comparator.config import AppConfig, TemplateDetectorConfig
from fc_comparator.detect import TemplateDetector
from fc_comparator.models import Box, Taxonomy
from fc_comparator.training import evaluate, export_yolo_dataset, samples_from_store, split_records

from .conftest import build_store


def test_export_yolo_dataset_with_mirroring(tmp_path, marked_store):
    ds = export_yolo_dataset(marked_store, Taxonomy(), tmp_path / "ds", val_split=0.25, seed=1)
    assert (ds.train_images, ds.val_images) == (3, 1)
    for split in ("train", "val"):
        imgs = sorted(p.stem for p in (ds.root / "images" / split).iterdir())
        labels = sorted(p.stem for p in (ds.root / "labels" / split).iterdir())
        assert imgs == labels
        originals = [s for s in imgs if not s.endswith("_mirror")]
        assert all(f"{s}_mirror" in imgs for s in originals)  # mirrors stay in their own split
    # mirrored copies swap fork_left/fork_right counts
    total = {c: v["train"] + v["val"] for c, v in ds.boxes.items()}
    assert total["fork_left"] == total["fork_right"]
    line = next(p for p in (ds.root / "labels" / "train").iterdir()).read_text().splitlines()[0].split()
    assert len(line) == 5 and 0 <= int(line[0]) < 4 and all(0 <= float(v) <= 1 for v in line[1:])
    text = ds.data_yaml.read_text()
    assert "names" in text and "fork_left" in text


def test_export_skips_unknown_labels_and_needs_marks(tmp_path):
    store = AnnotationStore(tmp_path / "a")
    with pytest.raises(ValueError, match="No marked images"):
        export_yolo_dataset(store, Taxonomy(), tmp_path / "ds")
    iid, _ = store.add_image(np.full((100, 100, 3), 200, np.uint8))
    store.set_boxes(iid, [Box("round", 10, 10, 20, 20), Box("clamp", 50, 50, 20, 20)])
    ds = export_yolo_dataset(store, Taxonomy(), tmp_path / "ds")
    assert ds.skipped_labels == {"clamp": 1} and ds.train_images == ds.val_images == 1


def test_split_is_reproducible(marked_store):
    a = split_records(marked_store.records(), 0.5, 3)
    b = split_records(marked_store.records(), 0.5, 3)
    assert [r.id for r in a[1]] == [r.id for r in b[1]] and len(a[1]) == 2


def test_evaluate_confusion_matrix(marked_store, tmp_path):
    det = TemplateDetector(TemplateDetectorConfig(), Taxonomy(), marked_store)
    unseen = build_store(tmp_path / "unseen", images=2, seed=700)
    res = evaluate(det, samples_from_store(unseen), Taxonomy().classes, min_confidence=0.6)
    n = len(res.classes)
    assert res.confusion.shape == (n + 1, n + 1)
    assert res.accuracy == 1.0 and res.missed_clips == 0 and res.false_clips == 0
    assert all(p == 1.0 and r == 1.0 for c, (p, r) in res.precision_recall().items() if res.confusion[res.classes.index(c)].sum())
    assert "accuracy: 100.0%" in res.as_text()


class _Fixed:
    name = "fixed"

    def __init__(self, dets):
        self.dets = dets

    def detect(self, _img):
        return self.dets


def test_evaluate_counts_errors():
    truth = [Box("round", 0, 0, 10, 10), Box("small", 50, 0, 10, 10), Box("round", 100, 0, 10, 10)]
    dets = [Box("round", 0, 0, 10, 10, 0.9), Box("fork_left", 50, 0, 10, 10, 0.8), Box("small", 200, 200, 10, 10, 0.7)]
    res = evaluate(_Fixed(dets), [(np.zeros((10, 10, 3), np.uint8), truth)], Taxonomy().classes)
    c = Taxonomy().classes
    cm = res.confusion
    assert cm[c.index("round"), c.index("round")] == 1
    assert cm[c.index("small"), c.index("fork_left")] == 1  # wrong class
    assert cm[c.index("round"), -1] == 1  # missed clip
    assert cm[-1, c.index("small")] == 1  # false clip
    assert res.accuracy == pytest.approx(1 / 3)


@pytest.mark.slow
def test_train_detector_end_to_end(tmp_path, marked_store):
    pytest.importorskip("ultralytics")
    from fc_comparator.detect.yolo import YoloDetector
    from fc_comparator.training import train_detector

    cfg = AppConfig(base_dir=tmp_path)
    cfg.detector.model_path = "models/det.pt"
    cfg.training.workdir = "runs"
    cfg.training.batch = 4
    msgs = []
    res = train_detector(cfg, marked_store, msgs.append, epochs=1, imgsz=320, base_model="yolo11n.yaml")
    assert res.model_path == tmp_path / "models" / "det.pt" and res.model_path.is_file()
    assert res.report_path.is_file() and (tmp_path / "models" / "det_confusion.png").is_file()
    assert any(m.startswith("epoch 1/1") for m in msgs)
    det = YoloDetector(res.model_path, imgsz=320)
    assert set(det.labels) == set(Taxonomy().classes)
    out = det.detect(np.full((320, 320, 3), 240, np.uint8))
    assert isinstance(out, list)
