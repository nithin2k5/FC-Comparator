"""Evidence images: every NG image, plus every Nth OK image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.models import InspectionReport
from ...vision.camera import write_image


class ImageSaver:
    def __init__(self, image_dir: str | Path, save_ok_every_n: int = 20, quality: int = 90):
        self.image_dir = Path(image_dir)
        self.save_ok_every_n = save_ok_every_n
        self.quality = quality
        self._ok_count = 0

    def should_save(self, report: InspectionReport) -> bool:
        if not report.ok:
            return True
        if self.save_ok_every_n <= 0:
            return False
        self._ok_count += 1
        return self._ok_count % self.save_ok_every_n == 1 or self.save_ok_every_n == 1

    def path_for(self, report: InspectionReport, inspection_id: int | None, suffix: str = "") -> Path:
        ts = report.timestamp
        safe_pn = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in report.part_number) or "unknown"
        name = f"{ts:%H%M%S}_{inspection_id or 0:06d}_{safe_pn}_{report.verdict.value}{suffix}.jpg"
        return self.image_dir / f"{ts:%Y-%m-%d}" / name

    def save(self, image: np.ndarray, report: InspectionReport, inspection_id: int | None, suffix: str = "") -> Path:
        return write_image(self.path_for(report, inspection_id, suffix), image, self.quality)
