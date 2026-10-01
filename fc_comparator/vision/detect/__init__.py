"""Object detectors: a part's trained YOLO model, or template matching on its labelled images."""

from __future__ import annotations

import logging

from ...config import AppConfig
from ..parts import Part
from .base import Detector, nms
from .template import TemplateDetector

log = logging.getLogger(__name__)

__all__ = ["Detector", "TemplateDetector", "create_detector", "nms"]


def create_detector(cfg: AppConfig, part: Part, version: str | None = None) -> Detector:
    """The detector a part is inspected with.

    ``yolo``: the part's active model (or ``version``). ``template``: template matching
    on the part's labelled images. Raises when the part has no usable detector.
    """
    d = cfg.detector
    if d.backend == "template":
        det = TemplateDetector(d.template, part.taxonomy(), part.store)
        det.note = "template matching on the labelled images"
        if not det.ready:
            raise RuntimeError(f"{part.code}: no labelled images")
        return det
    if d.backend != "yolo":
        raise ValueError(f"Unknown detector.backend {d.backend!r} (yolo | template)")
    info = part.model(version) if version else part.active()
    if info is None:
        raise RuntimeError(f"{part.code}: no active model - train one in Model Setup")
    from .yolo import YoloDetector

    det = YoloDetector(info.path, d.imgsz, d.device, d.min_score)
    det.version = info.version
    acc = f", {info.accuracy:.1%}" if info.accuracy is not None else ""
    det.note = f"model {info.version}{acc}"
    log.info("Loaded %s model %s (labels %s)", part.code, info.version, det.labels)
    return det
