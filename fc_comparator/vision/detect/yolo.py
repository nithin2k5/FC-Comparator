"""Ultralytics YOLO detector (YOLO11 / YOLOv8) trained on the marked images."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...core.models import Box
from .base import Detector

log = logging.getLogger(__name__)


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, model_path: str | Path, imgsz: int = 960, device: str = "cpu", min_score: float = 0.25):
        from ultralytics import YOLO  # imported lazily: heavy, optional

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path), task="detect")
        self.imgsz = imgsz
        self.device = device
        self.min_score = min_score
        self._names: dict[int, str] = dict(self.model.names)
        # Warm-up so the first real inspection is not slowed by lazy initialisation.
        self.detect(np.full((imgsz, imgsz, 3), 240, np.uint8))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._names.values())

    def detect(self, image: np.ndarray) -> list[Box]:
        r = self.model.predict(
            image, imgsz=self.imgsz, conf=self.min_score, iou=0.5, agnostic_nms=True, device=self.device, verbose=False
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        return [
            Box(self._names[int(c)], float(x0), float(y0), float(x1 - x0), float(y1 - y0), float(p))
            for (x0, y0, x1, y1), c, p in zip(xyxy, cls, conf)
        ]
