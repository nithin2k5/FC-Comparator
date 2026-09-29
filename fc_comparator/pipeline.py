"""Inspection pipeline.

``Inspector``  image -> align -> crop ROIs -> classify -> threshold -> compare -> report
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
from pathlib import Path

import numpy as np

from .align import Aligner
from .alert import AlertController, StationLock, create_alert
from .annotate import annotate
from .capture import FrameSource, create_source
from .capture.sources import read_image, write_image
from .classify import Classifier, create_classifier, crop_roi
from .compare import compare
from .config import AppConfig
from .models import UNCERTAIN, AlignmentInfo, Classification, InspectionReport
from .storage import ImageSaver, InspectionStore

log = logging.getLogger(__name__)


class StationLocked(RuntimeError):
    pass


class Inspector:
    def __init__(self, cfg: AppConfig, classifier: Classifier, aligner: Aligner | None = None):
        self.cfg = cfg
        self.classifier = classifier
        self.aligner = aligner

    @classmethod
    def from_config(cls, cfg: AppConfig, classifier: Classifier | None = None) -> "Inspector":
        aligner = None
        if cfg.alignment.enabled:
            ref_path = cfg.resolve(cfg.reference_image)
            if ref_path.is_file():
                aligner = Aligner(cfg.alignment, read_image(ref_path))
            else:
                log.warning("Alignment enabled but reference image %s is missing", ref_path)
        return cls(cfg, classifier or create_classifier(cfg), aligner)

    def inspect(self, image: np.ndarray, part_number: str, operator_id: str = "") -> tuple[InspectionReport, np.ndarray]:
        """Return (report, image in reference coordinates)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        mode = cfg.compare.mode
        report = InspectionReport(part_number, mode, [], operator_id=operator_id, classifier=self.classifier.name)
        part = cfg.parts.get(part_number)
        if part is None and mode in ("master", "both"):
            report.error = f"unknown part number {part_number!r}"
            return self._done(report, t0), image

        # 1. align
        work = image
        if cfg.alignment.enabled:
            if self.aligner is None:
                report.alignment = AlignmentInfo(applied=False, ok=False, message="no reference image")
            else:
                work, report.alignment = self.aligner.align(image)
            if not report.alignment.ok and cfg.alignment.fail_as_ng:
                report.error = f"alignment failed: {report.alignment.message}"
        else:
            report.alignment = AlignmentInfo(applied=False, ok=True, message="disabled")

        # 2. classify every ROI in one batch
        rois = [r for r in cfg.rois if r.cable <= cfg.station.cables and r.row <= cfg.station.rows]
        crops = [crop_roi(work, r, cfg.classifier.roi_padding) for r in rois]
        try:
            results = self.classifier.classify(crops)
        except Exception as exc:  # a broken model must produce NG, never OK
            log.exception("Classifier failed")
            report.error = f"classifier error: {exc}"
            results = [Classification(UNCERTAIN, 0.0) for _ in crops]

        # 3. confidence threshold: below it the position is "uncertain" (= NG)
        thr = cfg.classifier.confidence_threshold
        found = {}
        for roi, res in zip(rois, results):
            if res.label != UNCERTAIN and res.confidence < thr:
                res = Classification(UNCERTAIN, res.confidence, res.scores, guess=res.label)
            found[roi.key] = res

        # 4. compare
        positions = compare(found, part, cfg.station.cables, cfg.station.rows, mode)
        report.positions = [
            dataclasses.replace(p, reason=f"low confidence (looks like {found[p.key].guess} {p.confidence:.0%})")
            if p.found == UNCERTAIN and p.key in found and found[p.key].guess
            else p
            for p in positions
        ]
        return self._done(report, t0), work

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


class Station:
    """Everything an inspection station does, independent of the UI."""

    def __init__(
        self,
        cfg: AppConfig,
        source: FrameSource | None = None,
        inspector: Inspector | None = None,
        alert: AlertController | None = None,
        store: InspectionStore | None = None,
    ):
        self.cfg = cfg
        self.source = source or create_source(cfg)
        self.inspector = inspector or Inspector.from_config(cfg)
        self.alert = alert or AlertController(create_alert(cfg.alert), cfg.alert)
        self.store = store or InspectionStore(cfg.resolve(cfg.storage.database))
        self.images: ImageSaver | None = ImageSaver(cfg.resolve(cfg.storage.image_dir), cfg.storage.save_ok_every_n)
        self.lock =StationLock(cfg.security.supervisor_pin, cfg.security.lock_on_ng)
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
        """Rebuild classifier/aligner after setup changes (new ROIs, templates, model...)."""
        self.inspector = Inspector.from_config(self.cfg)
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
            report, aligned = self.inspector.inspect(frame, part_number, operator_id)

            # Alert first: the tower light must not wait for disk I/O.
            self.alert.ok() if report.ok else self.alert.ng()
            alert_ms = (time.perf_counter() - t_capture) * 1000

            annotated = annotate(aligned, report, self.cfg.rois)
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
                reason = report.error or f"{len(report.mismatches)} NG position(s)"
                self.lock.lock(reason, part_number, iid)
            log.info(
                "Inspection %d %s %s in %.0f ms (alert after %.0f ms)",
                iid, part_number, report.verdict.value, report.duration_ms, alert_ms,
            )
            return StationResult(report, annotated, iid, image_path, alert_ms)
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

    def save_reference(self, image: np.ndarray) -> Path:
        path = self.cfg.resolve(self.cfg.reference_image)
        write_image(path, image)
        return path
