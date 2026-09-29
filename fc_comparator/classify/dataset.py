"""Labeled crop dataset on disk: ``<dataset_dir>/<label>/<name>.png``.

The same folder feeds the template-matching fallback and YOLO training.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from ..capture.sources import IMAGE_EXTENSIONS, write_image
from ..models import CLASSIFIER_LABELS, FORK, Roi
from .base import crop_roi


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def dataset_counts(dataset_dir: Path) -> dict[str, int]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return {}
    return {d.name: len(list_images(d)) for d in sorted(dataset_dir.iterdir()) if d.is_dir()}


def dataset_warnings(dataset_dir: Path) -> list[str]:
    counts = dataset_counts(dataset_dir)
    warnings = []
    unknown = [k for k in counts if k not in CLASSIFIER_LABELS]
    if unknown:
        warnings.append(f"Unknown class folders {unknown}; valid: {list(CLASSIFIER_LABELS)}")
    if counts.get(FORK) and (counts.get("fork_left") or counts.get("fork_right")):
        warnings.append("Both 'fork' and oriented fork folders exist; label forks as fork_left/fork_right only")
    for label, n in counts.items():
        if 0 < n < 10:
            warnings.append(f"Class {label!r} has only {n} samples; collect at least 20-30")
    return warnings


def save_crop(dataset_dir: Path, label: str, crop: np.ndarray, stem: str = "") -> Path:
    if label not in CLASSIFIER_LABELS:
        raise ValueError(f"Invalid label {label!r}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    name = f"{stem + '_' if stem else ''}{stamp}.png"
    return write_image(Path(dataset_dir) / label / name, crop)


def harvest_board(
    image: np.ndarray,
    rois: Iterable[Roi],
    pattern: list[str],
    dataset_dir: Path,
    *,
    fork_orientation: str | None = "left",
    stem: str = "",
    padding: float = 0.0,
    only: set[tuple[int, int]] | None = None,
) -> list[Path]:
    """Save every ROI crop of a *known-good* board, labeled from its master pattern.

    ``fork_orientation`` turns pattern entries ``fork`` into ``fork_left`` /
    ``fork_right`` (the teacher states which way the forks face on this board).
    ``only`` limits harvesting to specific (cable, row) positions.
    """
    saved = []
    for roi in rois:
        if only is not None and roi.key not in only:
            continue
        label = pattern[roi.row - 1]
        if label == FORK:
            if fork_orientation not in ("left", "right"):
                raise ValueError("Specify fork_orientation 'left' or 'right' for fork positions")
            label = f"fork_{fork_orientation}"
        crop = crop_roi(image, roi, padding)
        saved.append(save_crop(dataset_dir, label, crop, f"{stem}c{roi.cable}r{roi.row}"))
    return saved
