"""Train the YOLO clip detector from the marked images, and evaluate detectors.

Used by the Training tab and by ``scripts/train_detector.py``.

Pipeline
  1. export the marked images to a YOLO data set (images/ + labels/, data.yaml),
     split per image into train/val (reproducible with a seed);
  2. mirror augmentation: every image containing a mirror class (fork_left /
     fork_right) is also added flipped with the labels swapped, inside its own
     split so validation stays honest. Ultralytics' random horizontal flip is
     disabled because it would keep the wrong orientation label;
  3. train (YOLO11n by default), reporting progress per epoch; can be stopped;
  4. evaluate the best weights on the validation images: detection-level
     confusion matrix (with a "background" row/column for missed and false
     clips), accuracy, per-class precision and recall.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..config import AppConfig
from ..core.models import Box, Taxonomy
from .camera import read_image, write_image
from .dataset import AnnotationStore, ImageRecord
from .detect.base import Detector

log = logging.getLogger(__name__)

BACKGROUND = "background"


# ---------------------------------------------------------------------------
# Data set export
# ---------------------------------------------------------------------------
@dataclass
class DatasetSummary:
    root: Path
    data_yaml: Path
    train_images: int
    val_images: int
    boxes: dict[str, dict[str, int]]  # class -> {"train": n, "val": n}
    skipped_labels: dict[str, int] = field(default_factory=dict)
    val_ids: list[str] = field(default_factory=list)


def split_records(records: list[ImageRecord], val_split: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    recs = sorted(records, key=lambda r: r.id)
    random.Random(seed).shuffle(recs)
    if len(recs) < 2:
        return recs, list(recs)  # tiny data set: validate on the training image
    n_val = min(len(recs) - 1, max(1, round(len(recs) * val_split)))
    return recs[n_val:], recs[:n_val]


def _yolo_lines(boxes: list[Box], w: int, h: int, index: dict[str, int]) -> list[str]:
    lines = []
    for b in boxes:
        cx, cy = (b.x + b.w / 2) / w, (b.y + b.h / 2) / h
        lines.append(f"{index[b.label]} {cx:.6f} {cy:.6f} {b.w / w:.6f} {b.h / h:.6f}")
    return lines


def export_yolo_dataset(
    store: AnnotationStore, taxonomy: Taxonomy, out_dir: Path, val_split: float = 0.2, seed: int = 0, mirror: bool = True
) -> DatasetSummary:
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    index = {name: i for i, name in enumerate(taxonomy.classes)}
    marked = [r for r in store.records() if r.boxes]
    if not marked:
        raise ValueError("No marked images yet - upload images and mark the clips first")
    train, val = split_records(marked, val_split, seed)
    counts: dict[str, dict[str, int]] = {c: {"train": 0, "val": 0} for c in taxonomy.classes}
    skipped: dict[str, int] = {}
    for rec in marked:
        for b in rec.boxes:
            if b.label not in index:
                skipped[b.label] = skipped.get(b.label, 0) + 1

    for split, recs in (("train", train), ("val", val)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            boxes = [b for b in rec.boxes if b.label in index]
            img = store.load_image(rec.id)
            variants = [(rec.id, img, boxes)]
            if mirror and any(taxonomy.mirror_of(b.label) != b.label for b in boxes):
                W = img.shape[1]
                flipped = [Box(taxonomy.mirror_of(b.label), W - b.x - b.w, b.y, b.w, b.h) for b in boxes]
                variants.append((f"{rec.id}_mirror", cv2.flip(img, 1), flipped))
            for name, im, bx in variants:
                write_image(out_dir / "images" / split / f"{name}.jpg", im, quality=95)
                (out_dir / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(_yolo_lines(bx, im.shape[1], im.shape[0], index)), encoding="utf-8"
                )
                for b in bx:
                    counts[b.label][split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                        "names": {i: n for n, i in index.items()}}, sort_keys=False),
        encoding="utf-8",
    )
    return DatasetSummary(out_dir, data_yaml, len(train), len(val), counts, skipped, [r.id for r in val])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    classes: list[str]  # + BACKGROUND as the last row/column of the matrix
    confusion: np.ndarray  # rows = truth, cols = prediction
    images: int

    @property
    def names(self) -> list[str]:
        return self.classes + [BACKGROUND]

    @property
    def accuracy(self) -> float:
        """Share of marked clips found with the right class."""
        n = len(self.classes)
        total = self.confusion[:n, :].sum()
        return float(np.trace(self.confusion[:n, :n]) / total) if total else 0.0

    def precision_recall(self) -> dict[str, tuple[float, float]]:
        out = {}
        for i, c in enumerate(self.classes):
            tp = self.confusion[i, i]
            pred = self.confusion[:, i].sum()
            true = self.confusion[i, :].sum()
            out[c] = (float(tp / pred) if pred else 0.0, float(tp / true) if true else 0.0)
        return out

    @property
    def false_clips(self) -> int:
        return int(self.confusion[-1, :-1].sum())

    @property
    def missed_clips(self) -> int:
        return int(self.confusion[:-1, -1].sum())

    def as_text(self) -> str:
        names = self.names
        w = max(11, max(len(n) for n in names) + 1)
        lines = [
            f"Images: {self.images}   accuracy: {self.accuracy:.1%}   missed clips: {self.missed_clips}   "
            f"false clips: {self.false_clips}",
            "",
            "true \\ pred".ljust(w) + "".join(n[: w - 1].rjust(w) for n in names),
        ]
        for i, n in enumerate(names):
            lines.append(n.ljust(w) + "".join(str(v).rjust(w) for v in self.confusion[i]))
        lines.append("")
        for c, (p, r) in self.precision_recall().items():
            lines.append(f"{c:<{w}} precision {p:6.1%}   recall {r:6.1%}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"classes": self.classes, "confusion": self.confusion.tolist(), "images": self.images,
                "accuracy": self.accuracy, "precision_recall": self.precision_recall()}


def evaluate(detector: Detector, samples: list[tuple[np.ndarray, list[Box]]], classes: list[str],
             iou: float = 0.5, min_confidence: float = 0.0) -> EvalResult:
    """Match detections to marked boxes (IoU, class-agnostic, best confidence first)."""
    idx = {c: i for i, c in enumerate(classes)}
    bg = len(classes)
    cm = np.zeros((bg + 1, bg + 1), int)
    for img, truth in samples:
        dets = sorted((d for d in detector.detect(img) if d.confidence >= min_confidence), key=lambda d: -d.confidence)
        used: set[int] = set()
        for d in dets:
            cand = [(d.iou(t), k) for k, t in enumerate(truth) if k not in used and t.label in idx]
            best = max(cand, default=(0.0, -1))
            p = idx.get(d.label, bg)
            if best[0] >= iou:
                used.add(best[1])
                cm[idx[truth[best[1]].label], p] += 1
            else:
                cm[bg, p] += 1  # a clip where none was marked
        for k, t in enumerate(truth):
            if k not in used and t.label in idx:
                cm[idx[t.label], bg] += 1  # a marked clip that was not found
    return EvalResult(list(classes), cm, len(samples))


def confusion_png(result: EvalResult, path: Path) -> Path:
    names = result.names
    cell, margin = 88, 150
    n = len(names)
    img = np.full((margin + n * cell + 20, margin + n * cell + 20, 3), 255, np.uint8)
    norm = result.confusion / np.maximum(result.confusion.sum(axis=1, keepdims=True), 1)
    for i in range(n):
        for j in range(n):
            v = float(norm[i, j])
            color = (int(255 - 190 * v), 255, int(255 - 190 * v)) if i == j else (int(255 - 190 * v), int(255 - 110 * v), 255)
            x, y = margin + j * cell, margin + i * cell
            cv2.rectangle(img, (x, y), (x + cell, y + cell), color, -1)
            cv2.rectangle(img, (x, y), (x + cell, y + cell), (190, 190, 190), 1)
            cv2.putText(img, str(result.confusion[i, j]), (x + 22, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, names[i][:12], (6, margin + i * cell + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, names[i][:10], (margin + i * cell + 4, margin - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "true (rows) vs detected (cols)", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return write_image(path, img)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    model_path: Path
    evaluation: EvalResult
    dataset: DatasetSummary
    epochs_run: int
    seconds: float
    stopped: bool = False
    report_path: Path | None = None


def train_detector(
    cfg: AppConfig,
    store: AnnotationStore,
    progress: Callable[[str], None] | None = None,
    stop: threading.Event | None = None,
    *,
    epochs: int | None = None,
    imgsz: int | None = None,
    base_model: str | None = None,
    seed: int = 0,
) -> TrainResult:
    """Export, train, evaluate and install the model at ``cfg.detector.model_path``."""
    from ultralytics import YOLO

    say = progress or (lambda msg: log.info(msg))
    t = cfg.training
    epochs = epochs or t.epochs
    imgsz = imgsz or t.imgsz
    workdir = cfg.resolve(t.workdir)
    started = time.time()

    ds = export_yolo_dataset(store, cfg.taxonomy, workdir / "dataset", t.val_split, seed, t.mirror)
    say(f"Data set: {ds.train_images} training / {ds.val_images} validation images "
        f"(+ mirrored copies), boxes {ds.boxes}")
    if ds.skipped_labels:
        say(f"Skipped boxes with unknown clip types: {ds.skipped_labels}")

    base = base_model or t.base_model
    if cfg.resolve(base).is_file():  # a local copy (offline stations); else Ultralytics downloads it by name
        base = str(cfg.resolve(base))
    try:
        model = YOLO(base, task="detect")
    except Exception as exc:
        fallback = Path(base).with_suffix(".yaml").name
        say(f"Could not load {base} ({exc}); training from scratch ({fallback})")
        model = YOLO(fallback, task="detect")

    state = {"epochs": 0, "stopped": False}

    def on_epoch(trainer) -> None:
        state["epochs"] = trainer.epoch + 1
        m = trainer.metrics or {}
        map50 = m.get("metrics/mAP50(B)")
        loss = _total_loss(getattr(trainer, "tloss", None))
        say(f"epoch {trainer.epoch + 1}/{trainer.epochs}  loss {loss:.3f}"
            + (f"  mAP50 {map50:.3f}" if map50 is not None else ""))
        if stop is not None and stop.is_set():
            state["stopped"] = True
            trainer.stop = True

    def on_batch(trainer) -> None:
        if stop is not None and stop.is_set():
            state["stopped"] = True
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", on_epoch)
    model.add_callback("on_train_batch_end", on_batch)
    has_mirror = any(cfg.taxonomy.mirror_of(c) != c for c in cfg.taxonomy.classes)
    say(f"Training {base} for up to {epochs} epochs at {imgsz}px on {cfg.detector.device} ...")
    model.train(
        data=str(ds.data_yaml), epochs=epochs, imgsz=imgsz, batch=t.batch, device=cfg.detector.device,
        patience=t.patience, seed=seed, project=str(workdir), name="train", exist_ok=True,
        fliplr=0.0 if has_mirror else 0.5, flipud=0.0, plots=False, verbose=False, workers=0,
    )

    weights = workdir / "train" / "weights"
    best = weights / "best.pt" if (weights / "best.pt").is_file() else weights / "last.pt"
    target = cfg.resolve(cfg.detector.model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    say(f"Model saved to {target}")

    result, report = evaluate_model(cfg, store, target, ds, imgsz, state["epochs"], base)
    say("Validation:\n" + result.as_text())
    return TrainResult(target, result, ds, state["epochs"], time.time() - started, state["stopped"], report)


def evaluate_model(cfg: AppConfig, store: AnnotationStore, model_path: Path, ds: DatasetSummary, imgsz: int,
                   epochs: int, base: str) -> tuple[EvalResult, Path]:
    """Score a trained model on the validation images; write ``*_report.json`` and ``*_confusion.png`` next to it."""
    from .detect.yolo import YoloDetector

    det = YoloDetector(model_path, imgsz, cfg.detector.device, cfg.detector.min_score)
    samples = [(store.load_image(i), store.get(i).boxes) for i in ds.val_ids]
    result = evaluate(det, samples, cfg.taxonomy.classes, min_confidence=cfg.detector.confidence_threshold)
    report = model_path.with_name(model_path.stem + "_report.json")
    report.write_text(json.dumps({**result.to_dict(), "confidence_threshold": cfg.detector.confidence_threshold,
                                  "epochs": epochs, "imgsz": imgsz, "base": Path(base).name,
                                  "val_images": ds.val_ids,
                                  "dataset": {"train": ds.train_images, "val": ds.val_images, "boxes": ds.boxes}},
                                 indent=2), encoding="utf-8")
    confusion_png(result, model_path.with_name(model_path.stem + "_confusion.png"))
    return result, report


def _total_loss(tloss) -> float:
    """Sum of the running training losses (a tensor or a dict, depending on the Ultralytics version)."""
    try:
        if isinstance(tloss, dict):
            return float(sum(float(v) for v in tloss.values()))
        return float(tloss.sum())
    except Exception:
        return float("nan")


def samples_from_store(store: AnnotationStore, ids: list[str] | None = None) -> list[tuple[np.ndarray, list[Box]]]:
    recs = [store.get(i) for i in ids] if ids else [r for r in store.records() if r.boxes]
    return [(read_image(store.image_path(r.id)), r.boxes) for r in recs]
