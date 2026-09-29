"""Inspection pipeline.

``Inspector``  image -> detect clips -> place them on the part's layout (cable, row)
               -> confidence threshold -> compare with the master -> report
``Station``    capture + Inspector + alert + NG lock + logging (what the UI and CLI drive)

The alert is fired as soon as the verdict is known, before annotation and
storage, so the tower light reacts well within the 1 s budget.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .alert import AlertController, StationLock, create_alert
from .annotate import annotate
from .annotations import AnnotationStore
from .capture import FrameSource, create_source
from .compare import compare
from .config import AppConfig
from .detect import Detector, create_detector
from .layout import LayoutError, infer_grid, match_layout
from .models import MISSING, UNCERTAIN, Box, Classification, InspectionReport, PlacementInfo
from .storage import ImageSaver, InspectionStore

log = logging.getLogger(__name__)


class StationLocked(RuntimeError):
    pass


class Inspector:
    def __init__(self, cfg: AppConfig, detector: Detector):
        self.cfg = cfg
        self.detector = detector

    @classmethod
    def from_config(cls, cfg: AppConfig, store: AnnotationStore | None = None) -> "Inspector":
        return cls(cfg, create_detector(cfg, store))

    def inspect(self, image: np.ndarray, part_number: str, operator_id: str = "") -> tuple[InspectionReport, list[Box]]:
        """Return (report, all detections)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        report = InspectionReport(part_number, cfg.compare.mode, [], operator_id=operator_id, detector=self.detector.name)
        part = cfg.parts.get(part_number)
        if part is None:
            report.error = f"unknown part number {part_number!r}"
            return self._done(report, t0), []
        if not self.detector.ready:
            report.error = "no trained detector - mark clips in Training data first"
            return self._done(report, t0), []

        try:
            dets = [d for d in self.detector.detect(image) if d.confidence >= cfg.detector.min_score]
        except Exception as exc:  # a broken model must produce NG, never OK
            log.exception("Detector failed")
            report.error = f"detector error: {exc}"
            return self._done(report, t0), []

        # 1. which detection sits at which (cable, row)
        h, w = image.shape[:2]
        extras_idx: list[int] = []
        if part.layout is not None:
            expected_labels = {(c, r): part.pattern[r - 1] for c in range(1, part.cables + 1) for r in range(1, part.rows + 1)}
            pl = match_layout(dets, part.layout, cfg.layout, (w, h), expected_labels, cfg.taxonomy)
            assignment, expected, extras_idx, placement = pl.assignment, pl.expected, pl.extras, pl.info
            s = w / part.layout.image_size[0] if part.layout.image_size[0] else 1.0
            clip_w, clip_h = part.layout.clip_size[0] * s, part.layout.clip_size[1] * s
        else:
            placement = PlacementInfo(expected=part.cables * part.rows)
            try:
                grid = infer_grid(dets, cables=part.cables, rows=part.rows)
            except LayoutError as exc:
                placement.ok, placement.message = False, f"{exc} (create a master from a marked image for this part)"
                grid = None
            assignment, expected = {}, {}
            if grid is not None:
                for key, center in grid.centers.items():
                    idx = sorted(grid.cells.get(key, []), key=lambda j: -dets[j].confidence)
                    assignment[key] = idx[0] if idx else None
                    extras_idx += idx[1:]
                    expected[key] = center
                clip_w, clip_h = grid.clip_size
                placement.matched = sum(v is not None for v in assignment.values())
                placement.message = f"grid {grid.cables}x{grid.rows} from detections (no master layout)"
            else:
                assignment = {(c, r): None for c in range(1, part.cables + 1) for r in range(1, part.rows + 1)}
                clip_w = clip_h = 0.0
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
                found[key] = (
                    Classification(d.label, d.confidence) if d.confidence >= thr
                    else Classification(UNCERTAIN, d.confidence, guess=d.label)
                )
                boxes[key] = (d.x, d.y, d.w, d.h)

        # 3. compare with the master
        positions = compare(found, part, part.cables, part.rows, report.mode, cfg.taxonomy)
        report.positions = [
            dataclasses.replace(
                p,
                box=boxes.get(p.key),
                reason=(f"low confidence (looks like {found[p.key].guess} {p.confidence:.0%})"
                        if p.found == UNCERTAIN and found[p.key].guess else p.reason),
            )
            for p in positions
        ]
        # 4. confident clips where the master has none are errors too (wrong part / extra clip),
        #    counted only inside the clip area (connectors and board marks around it are irrelevant)
        if expected and clip_w:
            xs = [c[0] for c in expected.values()]
            ys = [c[1] for c in expected.values()]
            mx, my = clip_w * 1.0, clip_h * 1.0
            area = (min(xs) - mx, min(ys) - my, max(xs) + mx, max(ys) + my)
            inside = lambda b: area[0] <= b.center[0] <= area[2] and area[1] <= b.center[1] <= area[3]  # noqa: E731
        else:
            inside = lambda b: True  # noqa: E731
        report.extras = [dets[j] for j in extras_idx if dets[j].confidence >= thr and inside(dets[j])]
        n_pos = part.cables * part.rows
        if not report.error and len(report.extras) >= max(2, 0.25 * n_pos):
            report.error = f"board does not match {part.code}'s layout ({len(report.extras)} unexpected clips) - wrong part?"
        return self._done(report, t0), dets

    @staticmethod
    def _done(report: InspectionReport, t0: float) -> InspectionReport:
        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report


@dataclass
class StationResult:
    report: InspectionReport
    annotated: np.ndarray
    inspection_id: int
    image_path: str
    alert_latency_ms: float  # capture finished -> alert outputs set
    detections: list[Box]


class Station:
    """Everything an inspection station does, independent of the UI."""

    def __init__(
        self,
        cfg: AppConfig,
        source: FrameSource | None = None,
        inspector: Inspector | None = None,
        alert: AlertController | None = None,
        store: InspectionStore | None = None,
        annotations: AnnotationStore | None = None,
    ):
        self.cfg = cfg
        self.annotations = annotations or AnnotationStore(cfg.resolve(cfg.annotations.dir))
        self.source = source or create_source(cfg)
        self.inspector = inspector or Inspector.from_config(cfg, self.annotations)
        self.alert = alert or AlertController(create_alert(cfg.alert), cfg.alert)
        self.store = store or InspectionStore(cfg.resolve(cfg.storage.database))
        self.images: ImageSaver | None = ImageSaver(cfg.resolve(cfg.storage.image_dir), cfg.storage.save_ok_every_n)
        self.lock = StationLock(cfg.security.supervisor_pin, cfg.security.lock_on_ng)
        self._busy = threading.Lock()

    def open(self) -> None:
        try:
            self.source.open()
        except Exception as exc:
            log.error("Frame source failed to open: %s", exc)
        self.alert.idle()

    def close(self) -> None:
        self.alert.close()
        self.source.close()
        self.store.close()

    def reload(self) -> None:
        """Rebuild the detector after new markings, a new model or changed settings."""
        self.annotations.reload()
        self.inspector = Inspector.from_config(self.cfg, self.annotations)
        self.images = ImageSaver(self.cfg.resolve(self.cfg.storage.image_dir), self.cfg.storage.save_ok_every_n)
        self.lock.pin_hash = self.cfg.security.supervisor_pin
        self.lock.enabled = self.cfg.security.lock_on_ng

    def rebuild_io(self) -> list[str]:
        """Recreate frame source and alert backend after settings changed. Returns problems."""
        problems = []
        self.source.close()
        self.source = create_source(self.cfg)
        try:
            self.source.open()
        except Exception as exc:
            problems.append(f"camera: {exc}")
        try:
            backend = create_alert(self.cfg.alert)
        except Exception as exc:
            from .alert import ConsoleAlert

            problems.append(f"alert backend {self.cfg.alert.backend}: {exc} (using console)")
            backend = ConsoleAlert()
        old, self.alert = self.alert, AlertController(backend, self.cfg.alert)
        old.close()
        self.alert.idle()
        return problems

    def can_inspect(self, part_number: str) -> tuple[bool, str]:
        st = self.lock.state
        if st.locked and part_number != st.part_number:
            return False, f"Station locked after NG on {st.part_number}: re-inspect {st.part_number} or supervisor PIN"
        return True, ""

    def inspect(self, part_number: str, operator_id: str = "", image: np.ndarray | None = None) -> StationResult:
        ok, why = self.can_inspect(part_number)
        if not ok:
            raise StationLocked(why)
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("Inspection already running")
        try:
            frame = image if image is not None else self.source.capture()
            t_capture = time.perf_counter()
            report, dets = self.inspector.inspect(frame, part_number, operator_id)

            # Alert first: the tower light must not wait for disk I/O.
            self.alert.ok() if report.ok else self.alert.ng()
            alert_ms = (time.perf_counter() - t_capture) * 1000

            annotated = annotate(frame, report)
            iid = self.store.save_report(report, station=self.cfg.station.name)
            image_path = ""
            if self.images is not None and self.images.should_save(report):
                try:
                    image_path = str(self.images.save(annotated, report, iid))
                    if self.cfg.storage.save_raw_images:
                        self.images.save(frame, report, iid, suffix="_raw")
                    self.store.set_image_path(iid, image_path)
                except Exception as exc:  # full disk etc. must not lose the DB record
                    log.error("Saving evidence image failed: %s", exc)

            if report.ok:
                if self.lock.release_by_pass(part_number):
                    self.store.log_event("unlock_pass", operator_id, f"re-inspection passed ({part_number})", iid)
            else:
                reason = report.error or f"{len(report.mismatches) + len(report.extras)} NG finding(s)"
                self.lock.lock(reason, part_number, iid)
            log.info("Inspection %d %s %s in %.0f ms (alert after %.0f ms)",
                     iid, part_number, report.verdict.value, report.duration_ms, alert_ms)
            return StationResult(report, annotated, iid, image_path, alert_ms, dets)
        finally:
            self._busy.release()

    def acknowledge(self, pin: str, supervisor_id: str = "") -> bool:
        st = self.lock.state
        if not self.lock.acknowledge(pin):
            self.store.log_event("ack_failed", supervisor_id, "wrong PIN", st.inspection_id)
            return False
        self.store.log_event("ack", supervisor_id, f"NG acknowledged: {st.reason} ({st.part_number})", st.inspection_id)
        self.alert.idle()
        return True
