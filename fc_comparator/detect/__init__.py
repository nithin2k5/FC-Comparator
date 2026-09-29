"""Clip detectors: YOLO (primary) and template matching (from the first marked images)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..annotations import AnnotationStore
from ..config import AppConfig
from .base import Detector, nms
from .template import TemplateDetector

log = logging.getLogger(__name__)

__all__ = ["Detector", "TemplateDetector", "create_detector", "model_report", "nms"]


def model_report(model_path: Path) -> dict | None:
    """The evaluation report written next to a model by training (None if absent/unreadable)."""
    p = Path(model_path).with_name(Path(model_path).stem + "_report.json")
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def create_detector(cfg: AppConfig, store: AnnotationStore | None = None) -> Detector:
    """Build the configured detector.

    ``auto`` uses the YOLO model when the weights exist, Ultralytics imports and
    the model's validation accuracy reaches ``min_model_accuracy``; otherwise it
    uses template matching on the marked images. ``yolo``/``template`` force one.
    """
    d = cfg.detector
    if d.backend not in ("auto", "yolo", "template"):
        raise ValueError(f"Unknown detector.backend {d.backend!r} (auto | yolo | template)")
    model_path = cfg.resolve(d.model_path)
    store = store or AnnotationStore(cfg.resolve(cfg.annotations.dir))
    template: TemplateDetector | None = None
    note = ""

    if d.backend in ("auto", "yolo"):
        if not model_path.is_file():
            if d.backend == "yolo":
                raise FileNotFoundError(f"YOLO model not found: {model_path}")
            note = "no trained model yet"
        else:
            report = model_report(model_path)
            acc = report.get("accuracy") if report else None
            if d.backend == "auto" and acc is not None and acc < d.min_model_accuracy:
                template = TemplateDetector(d.template, cfg.taxonomy, store)
                if template.ready:
                    note = (f"trained model reached only {acc:.0%} validation accuracy "
                            f"(needs {d.min_model_accuracy:.0%}) - train longer or mark more images")
                    log.warning("Not using %s: %s", model_path, note)
                    template.note = note
                    return template
            try:
                from .yolo import YoloDetector

                det = YoloDetector(model_path, d.imgsz, d.device, d.min_score)
            except Exception as exc:
                if d.backend == "yolo":
                    raise
                note = f"model could not be loaded ({exc})"
                log.warning("YOLO model unavailable: %s; using template matching", exc)
            else:
                unknown = [c for c in det.labels if c not in cfg.taxonomy.classes]
                if unknown:
                    log.warning("Model classes %s are not in taxonomy.classes; retrain after changing clip types", unknown)
                det.note = f"trained model, validation accuracy {acc:.0%}" if acc is not None else "trained model"
                log.info("Using YOLO detector %s (classes %s)", model_path, det.labels)
                return det

    det = template or TemplateDetector(d.template, cfg.taxonomy, store)
    det.note = note or "template matching on the marked images"
    if not det.ready:
        det.note = "no marked clips yet - upload and mark images in Training data"
        log.warning(det.note)
    return det
