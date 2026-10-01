"""Inspector: board image -> detect objects -> place them on the master's layout (cable, row)
-> confidence threshold -> compare with the master -> report."""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ..config import AppConfig
from ..core.compare import compare
from ..core.layout import match_layout
from ..core.models import (
    MISSING,
    UNCERTAIN,
    Box,
    Classification,
    InspectionReport,
    PartNumber,
    Taxonomy,
)
from ..vision.detect import Detector

log = logging.getLogger(__name__)


class Inspector:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def inspect(
        self,
        image: np.ndarray,
        part_number: str,
        master: PartNumber | None,
        detector: Detector | None,
        taxonomy: Taxonomy | None = None,
        operator_id: str = "",
        not_ready: str = "",
    ) -> tuple[InspectionReport, list[Box]]:
        """Return (report, all detections). ``not_ready`` explains why the part cannot be inspected."""
        t0 = time.perf_counter()
        cfg = self.cfg
        report = InspectionReport(part_number, "master", [], operator_id=operator_id,
                                  detector=_detector_name(detector))
        if not_ready or master is None or master.layout is None or detector is None:
            report.error = f"{part_number}: not set up ({not_ready or 'no master or model'})"
            return self._done(report, t0), []
        taxonomy = taxonomy or Taxonomy.from_labels(sorted(master.labels()))
        try:
            dets = [d for d in detector.detect(image) if d.confidence >= cfg.detector.min_score]
        except Exception as exc:  # a broken model must produce NG, never OK
            log.exception("Detector failed")
            report.error = f"detector error: {exc}"
            return self._done(report, t0), []

        # 1. which detection sits at which (cable, row)
        h, w = image.shape[:2]
        expected_labels = {(c, r): master.expected(c, r) for c in range(1, master.cables + 1)
                           for r in range(1, master.rows + 1)}
        pl = match_layout(dets, master.layout, cfg.layout, (w, h), expected_labels, taxonomy)
        s = w / master.layout.image_size[0] if master.layout.image_size[0] else 1.0
        clip_w, clip_h = master.layout.clip_size[0] * s, master.layout.clip_size[1] * s
        report.placement = pl.info
        if not pl.info.ok:
            report.error = f"placement failed: {pl.info.message}"

        # 2. what is at each position, with the confidence threshold applied
        thr = cfg.detector.confidence_threshold
        found: dict[tuple[int, int], Classification] = {}
        boxes: dict[tuple[int, int], tuple[float, float, float, float] | None] = {}
        for key, j in pl.assignment.items():
            if j is None:
                found[key] = Classification(MISSING, 0.0)
                cx, cy = pl.expected.get(key, (0.0, 0.0))
                boxes[key] = (cx - clip_w / 2, cy - clip_h / 2, clip_w, clip_h)
            else:
                d = dets[j]
                found[key] = (Classification(d.label, d.confidence) if d.confidence >= thr
                              else Classification(UNCERTAIN, d.confidence, guess=d.label))
                boxes[key] = (d.x, d.y, d.w, d.h)

        # 3. compare with the master
        report.positions = [
            dataclasses.replace(
                p, box=boxes.get(p.key),
                reason=(f"low confidence (looks like {found[p.key].guess} {p.confidence:.0%})"
                        if p.found == UNCERTAIN and found[p.key].guess else p.reason),
            )
            for p in compare(found, master, taxonomy)
        ]

        # 4. confident objects where the master has none are errors too (wrong part / extra object),
        #    counted only inside the board area (connectors and marks around it are irrelevant)
        xs = [c[0] for c in pl.expected.values()]
        ys = [c[1] for c in pl.expected.values()]
        area = (min(xs) - clip_w, min(ys) - clip_h, max(xs) + clip_w, max(ys) + clip_h)

        def inside(b: Box) -> bool:
            return area[0] <= b.center[0] <= area[2] and area[1] <= b.center[1] <= area[3]

        report.extras = [dets[j] for j in pl.extras if dets[j].confidence >= thr and inside(dets[j])]
        n_pos = master.cables * master.rows
        n_wrong = sum(not p.ok and p.found in taxonomy.classes for p in report.positions)
        if not report.error and (len(report.extras) >= max(2, 0.25 * n_pos) or n_wrong >= 0.75 * n_pos):
            report.error = (f"board does not match {master.code} ({n_wrong}/{n_pos} positions wrong, "
                            f"{len(report.extras)} unexpected objects) - wrong part?")
        return self._done(report, t0), dets

    @staticmethod
    def _done(report: InspectionReport, t0: float) -> InspectionReport:
        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report


def _detector_name(detector: Detector | None) -> str:
    if detector is None:
        return ""
    return f"{detector.name} {getattr(detector, 'version', '')}".strip()
