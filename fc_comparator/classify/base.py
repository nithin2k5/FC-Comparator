"""Classifier interface and ROI cropping."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models import Classification, Roi


class Classifier(ABC):
    name: str = "classifier"

    @property
    @abstractmethod
    def labels(self) -> tuple[str, ...]:
        """Labels this classifier can output."""

    @property
    def ready(self) -> bool:
        return bool(self.labels)

    @abstractmethod
    def classify(self, crops: list[np.ndarray]) -> list[Classification]:
        """Classify BGR crops; one Classification per crop, same order."""


def crop_roi(image: np.ndarray, roi: Roi, padding: float = 0.0) -> np.ndarray:
    """Crop an ROI (optionally padded by a fraction of its size), clamped to the image."""
    px, py = int(round(roi.w * padding)), int(round(roi.h * padding))
    h, w = image.shape[:2]
    x0, y0 = max(0, roi.x - px), max(0, roi.y - py)
    x1, y1 = min(w, roi.x + roi.w + px), min(h, roi.y + roi.h + py)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(1, roi.h), max(1, roi.w), 3), np.uint8)
    return image[y0:y1, x0:x1].copy()
