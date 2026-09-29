"""Clip detectors: YOLO (primary) and template matching (from the first marked images)."""

from __future__ import annotations

import logging

from ..annotations import AnnotationStore
from ..config import AppConfig
from .base import Detector, nms
from .template import TemplateDetector

log = logging.getLogger(__name__)

__all__ = ["Detector", "TemplateDetector", "create_detector", "nms"]


def create_detector(cfg: AppConfig, store: AnnotationStore | None = None) -> Detector:
    """Build the configured detector.

    ``auto`` uses the YOLO model when the weights exist and Ultralytics imports,
    otherwise template matching on the marked images.
    """
    d = cfg.detector
    model_path = cfg.resolve(d.model_path)
    if d.backend not in ("auto", "yolo", "template"):
        raise ValueError(f"Unknown detector.backend {d.backend!r} (auto | yolo | template)")
    if d.backend in ("auto", "yolo"):
        if model_path.is_file():
            try:
                from .yolo import YoloDetector

                det = YoloDetector(model_path, d.imgsz, d.device, d.min_score)
                unknown = [c for c in det.labels if c not in cfg.taxonomy.classes]
                if unknown:
                    log.warning("Model classes %s are not in taxonomy.classes; retrain after changing clip types", unknown)
                log.info("Using YOLO detector %s (classes %s)", model_path, det.labels)
                return det
            except Exception as exc:
                if d.backend == "yolo":
                    raise
                log.warning("YOLO model unavailable (%s); falling back to template matching", exc)
        elif d.backend == "yolo":
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
    store = store or AnnotationStore(cfg.resolve(cfg.annotations.dir))
    det = TemplateDetector(d.template, cfg.taxonomy, store)
    if not det.ready:
        log.warning("No marked clips yet - upload and mark images in Training data")
    return det
