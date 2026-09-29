"""Clip classifiers: YOLO (primary) and template matching (fallback)."""

from __future__ import annotations

import logging

from ..config import AppConfig
from .base import Classifier, crop_roi
from .template import TemplateClassifier

log = logging.getLogger(__name__)

__all__ = ["Classifier", "TemplateClassifier", "create_classifier", "crop_roi"]


def create_classifier(cfg: AppConfig) -> Classifier:
    """Build the configured classifier.

    ``auto`` uses the YOLO model when the weights file exists and Ultralytics
    imports, otherwise falls back to template matching on the dataset folder.
    """
    c = cfg.classifier
    model_path = cfg.resolve(c.model_path)
    if c.backend not in ("auto", "yolo", "template"):
        raise ValueError(f"Unknown classifier.backend {c.backend!r} (auto | yolo | template)")

    if c.backend in ("auto", "yolo"):
        if model_path.is_file():
            try:
                from .yolo import YoloClassifier

                clf = YoloClassifier(model_path, c.yolo.imgsz, c.yolo.device)
                log.info("Using YOLO classifier %s (classes %s)", model_path, clf.labels)
                return clf
            except Exception as exc:
                if c.backend == "yolo":
                    raise
                log.warning("YOLO model unavailable (%s); falling back to template matching", exc)
        elif c.backend == "yolo":
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

    clf = TemplateClassifier(c.template, cfg.resolve(c.dataset_dir))
    if not clf.ready:
        log.warning("No templates in %s - teach clip samples in Setup mode", cfg.resolve(c.dataset_dir))
    return clf
