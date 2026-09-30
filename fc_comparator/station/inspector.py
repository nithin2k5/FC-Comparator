"""Inspector: board image -> detect clips -> place them on the part's layout (cable, row)
-> confidence threshold -> compare with the master -> report."""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ..config import AppConfig
from ..core.compare import compare
from ..core.layout import LayoutError, infer_grid, match_layout
from ..core.models import (
    MISSING,
    UNCERTAIN,
    Box,
    Classification,
    InspectionReport,
    PartNumber,
    PlacementInfo,
)
from ..vision.dataset import AnnotationStore
from ..vision.detect import Detector, create_detector

log = logging.getLogger(__name__)


class Inspector:
    def __init__(self, cfg: AppConfig, detector: Detector):
        self.cfg = cfg
        self.detector = detector

    @classmethod
    def from_config(cls, cfg: AppConfig, store: AnnotationStore | None = None) -> "Inspector":
        return cls(cfg, create_detector(cfg, store))

    def detect(self, image: np.ndarray) -> list[Box]:
        return [d for d in self.detector.detect(image) if d.confidence >= self.cfg.detector.min_score]

    def inspect(self, image: np.ndarray, part_number: str, operator_id: str = "") -> tuple[InspectionReport, list[Box]]:
        """Return (report, all detections)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        report = InspectionReport(part_number, "master", [], operator_id=operator_id, detector=self.detector.name)
        part = cfg.parts.get(part_number)
        if part is None:
            report.error = f"unknown part number {part_number!r}"
            return self._done(report, t0), []
        if not self.detector.ready:
            report.error = "no trained detector - train the model on the Training tab first"
            return self._done(report, t0), []
        try:
            dets = self.detect(image)
        except Exception as exc:  # a broken model must produce NG, never OK
            log.exception("Detector failed")
            report.error = f"detector error: {exc}"
            return self._done(report, t0), []

        # 1. which detection sits at which (cable, row)
        h, w = image.shape[:2]
        assignment, expected, extras_idx, placement, clip_w, clip_h = self._place(dets, part, (w, h))
        report.placement = placement
        if not placement.ok:
            report.error = f"placement failed: {placement.message}"

        # 2. what is at each position, with the confidence threshold applied
        thr = cfg.detector.confidence_threshold
        found: dict[tuple[int, int], Classification] = {}
        boxes: dict[tuple[int, int], tuple[float, float, float, float] | None] = {}
        for key, j in assignment.items():
            if j is None:
                found[key] = Classification(MISSING, 0.0)
                cx, cy = expected.get(key, (0.0, 0.0))
                boxes[key] = (cx - clip_w / 2, cy - clip_h / 2, clip_w, clip_h) if clip_w else None
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
            for p in compare(found, part, cfg.taxonomy)
        ]

        # 4. confident clips where the master has none are errors too (wrong part / extra clip),
        #    counted only inside the clip area (connectors and board marks around it are irrelevant)
        if expected and clip_w:
            xs = [c[0] for c in expected.values()]
            ys = [c[1] for c in expected.values()]
            area = (min(xs) - clip_w, min(ys) - clip_h, max(xs) + clip_w, max(ys) + clip_h)
            inside = lambda b: area[0] <= b.center[0] <= area[2] and area[1] <= b.center[1] <= area[3]  # noqa: E731
        else:
            inside = lambda b: True  # noqa: E731
        report.extras = [dets[j] for j in extras_idx if dets[j].confidence >= thr and inside(dets[j])]
        n_pos = part.cables * part.rows
        n_wrong = sum(not p.ok and p.found in cfg.taxonomy.classes for p in report.positions)
        if not report.error and (len(report.extras) >= max(2, 0.25 * n_pos) or n_wrong >= 0.75 * n_pos):
            report.error = (f"board does not match {part.code} ({n_wrong}/{n_pos} positions wrong, "
                            f"{len(report.extras)} unexpected clips) - wrong part?")
        return self._done(report, t0), dets

    def _place(self, dets: list[Box], part: PartNumber, size: tuple[int, int]):
        """(assignment, expected centres, extra detection indices, placement info, clip w, clip h)."""
        if part.layout is not None:
            expected_labels = {(c, r): part.expected(c, r) for c in range(1, part.cables + 1)
                               for r in range(1, part.rows + 1)}
            pl = match_layout(dets, part.layout, self.cfg.layout, size, expected_labels, self.cfg.taxonomy)
            s = size[0] / part.layout.image_size[0] if part.layout.image_size[0] else 1.0
            return (pl.assignment, pl.expected, pl.extras, pl.info,
                    part.layout.clip_size[0] * s, part.layout.clip_size[1] * s)

        # A part without a master photo: group the detections into its cables x rows grid.
        placement = PlacementInfo(expected=part.cables * part.rows)
        assignment: dict[tuple[int, int], int | None] = {
            (c, r): None for c in range(1, part.cables + 1) for r in range(1, part.rows + 1)}
        expected: dict[tuple[int, int], tuple[float, float]] = {}
        extras: list[int] = []
        try:
            grid = infer_grid(dets, cables=part.cables, rows=part.rows)
        except LayoutError as exc:
            placement.ok, placement.message = False, f"{exc} (create the part from a photo of a good board)"
            return assignment, expected, extras, placement, 0.0, 0.0
        for key, center in grid.centers.items():
            idx = sorted(grid.cells.get(key, []), key=lambda j: -dets[j].confidence)
            assignment[key] = idx[0] if idx else None
            extras += idx[1:]
            expected[key] = center
        placement.matched = sum(v is not None for v in assignment.values())
        placement.message = f"grid {grid.cables}x{grid.rows} from detections (no master photo)"
        return assignment, expected, extras, placement, grid.clip_size[0], grid.clip_size[1]

    @staticmethod
    def _done(report: InspectionReport, t0: float) -> InspectionReport:
        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report
