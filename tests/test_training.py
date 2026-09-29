"""Training script + YOLO classifier integration (short from-scratch run, no downloads)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ultralytics")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_classifier  # noqa: E402

from fc_comparator.classify.yolo import YoloClassifier  # noqa: E402
from fc_comparator.classify.dataset import save_crop  # noqa: E402


def test_split_mirrors_forks_within_split(tmp_path):
    src = tmp_path / "src"
    rng = np.random.default_rng(0)
    for label in ("round", "fork_left"):
        for i in range(10):
            save_crop(src, label, rng.integers(0, 255, (40, 40, 3), dtype=np.uint8), f"{label}{i}")
    counts = train_classifier.split_dataset(src, tmp_path / "split", 0.2, 0, mirror_forks=True)
    assert counts["fork_right"] == counts["fork_left"] == {"train": 8, "val": 2}
    val_right = {p.name for p in (tmp_path / "split" / "val" / "fork_right").iterdir()}
    val_left = {p.name for p in (tmp_path / "split" / "val" / "fork_left").iterdir()}
    assert {n.replace("_mirror", "") for n in val_right} == val_left  # mirrors stay in the same split


@pytest.mark.slow
def test_train_script_produces_usable_model(tmp_path):
    out = tmp_path / "models" / "clip.pt"
    rc = train_classifier.main([
        str(ROOT / "samples" / "dataset"), "--from-scratch", "--epochs", "2", "--imgsz", "64",
        "--batch", "32", "--out", str(out), "--workdir", str(tmp_path / "run"),
    ])
    assert rc == 0 and out.is_file()
    metrics = json.loads((tmp_path / "models" / "clip_metrics.json").read_text())
    assert set(metrics["classes"]) == {"round", "fork_left", "fork_right", "small", "missing"}
    assert np.array(metrics["confusion"]).sum() > 0
    assert (tmp_path / "models" / "clip_confusion.png").is_file()

    clf = YoloClassifier(out, imgsz=64)
    res = clf.classify([np.full((120, 120, 3), 200, np.uint8)] * 16)
    assert len(res) == 16
    assert all(r.label in clf.labels and 0 <= r.confidence <= 1 for r in res)
    assert sum(res[0].scores.values()) == pytest.approx(1.0, abs=1e-3)
