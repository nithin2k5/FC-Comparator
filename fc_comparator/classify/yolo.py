"""Ultralytics YOLO classification model (YOLO11-cls / YOLOv8-cls) applied per ROI crop.

A per-ROI classifier is used rather than a whole-board detector: the ROIs are
fixed by the fixture, a crop classifier needs far less labeled data (one
folder per class, no boxes to draw), and all 16 crops go through the network
in a single batch, which keeps inspection well under a second on a Pi 5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models import CLASSIFIER_LABELS, Classification
from .base import Classifier

log = logging.getLogger(__name__)


class YoloClassifier(Classifier):
    name = "yolo"

    def __init__(self, model_path: str | Path, imgsz: int = 128, device: str = "cpu"):
        from ultralytics import YOLO  # imported lazily: heavy, optional

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path), task="classify")
        self.imgsz = imgsz
        self.device = device
        self._names: dict[int, str] = dict(self.model.names)
        unknown = [n for n in self._names.values() if n not in CLASSIFIER_LABELS]
        if unknown:
            log.warning("Model %s has classes %s that are not valid clip labels", model_path, unknown)
        # Warm-up so the first real inspection is not slowed by lazy initialisation.
        self.classify([np.full((imgsz, imgsz, 3), 255, np.uint8)])

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._names.values())

    def classify(self, crops: list[np.ndarray]) -> list[Classification]:
        if not crops:
            return []
        results = self.model.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False)
        out = []
        for r in results:
            probs = r.probs.data.cpu().numpy().astype(float)
            top = int(np.argmax(probs))
            out.append(
                Classification(self._names[top], float(probs[top]), {self._names[i]: float(p) for i, p in enumerate(probs)})
            )
        return out
