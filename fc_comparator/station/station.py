"""Station: capture + Inspector + alert + NG lock + logging (what the UI and CLI drive).

The alert is fired as soon as the verdict is known, before annotation and
storage, so the tower light reacts well within the 1 s budget.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from ..core.models import Box, InspectionReport
from ..vision.camera import FrameSource, create_source
from ..vision.dataset import AnnotationStore
from ..vision.drawing import annotate
from .alerts import AlertController, create_alert
from .inspector import Inspector
from .lock import StationLock
from .storage import ImageSaver, InspectionStore

log = logging.getLogger(__name__)


class StationLocked(RuntimeError):
    pass


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
        dataset: AnnotationStore | None = None,
    ):
        self.cfg = cfg
        self.dataset = dataset or AnnotationStore(cfg.resolve(cfg.dataset.dir))
        self.source = source or create_source(cfg)
        self.inspector = inspector or Inspector.from_config(cfg, self.dataset)
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
        self.dataset.reload()
        self.inspector = Inspector.from_config(self.cfg, self.dataset)
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
            from .alerts import ConsoleAlert

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
