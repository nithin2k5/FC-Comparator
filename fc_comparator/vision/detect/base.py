"""Detector interface: find every clip in a board image."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.models import Box


class Detector(ABC):
    name: str = "detector"
    note: str = ""  # why this detector was chosen (shown in the UI)

    @property
    @abstractmethod
    def labels(self) -> tuple[str, ...]:
        """Clip classes this detector can report."""

    @property
    def ready(self) -> bool:
        return bool(self.labels)

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Box]:
        """Return clip boxes (label, confidence) in image pixels."""


def nms(boxes: list[Box], iou: float = 0.5) -> list[Box]:
    """Class-agnostic non-maximum suppression (one clip, one box)."""
    kept: list[Box] = []
    for b in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if all(b.iou(k) < iou for k in kept):
            kept.append(b)
    return kept
