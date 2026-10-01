"""Station: capture + Inspector + alert + NG lock + logging (what the UI and CLI drive).

Every part number is inspected with its own active model and master. The
alert is fired as soon as the verdict is known, before annotation and
storage, so the tower light reacts well within the 1 s budget.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from ..core.models import Box, InspectionReport
from ..vision.camera import FrameSource, create_source
from ..vision.detect import Detector, create_detector
from ..vision.drawing import annotate
from ..vision.parts import Part, PartRepository
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
        alert: AlertController | None = None,
        store: InspectionStore | None = None,
        parts: PartRepository | None = None,
        detector_factory: Callable[[AppConfig, Part], Detector] = create_detector,
    ):
        self.cfg = cfg
        self.parts = parts or PartRepository(cfg.resolve(cfg.storage.parts_dir))
        self.source = source or create_source(cfg)
        self.inspector = Inspector(cfg)
        self.detector_factory = detector_factory
        self.alert = alert or AlertController(create_alert(cfg.alert), cfg.alert)
        self.store = store or InspectionStore(cfg.resolve(cfg.storage.database))
        self.images: ImageSaver | None = ImageSaver(cfg.resolve(cfg.storage.image_dir), cfg.storage.save_ok_every_n)
        self.lock = StationLock(cfg.security.supervisor_pin, cfg.security.lock_on_ng)
        self._busy = threading.Lock()
        self._detectors: dict[str, tuple[tuple, Detector]] = {}
        self._det_lock = threading.Lock()

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
        """Forget loaded detectors and apply changed settings."""
        with self._det_lock:
            self._detectors.clear()
        self.inspector = Inspector(self.cfg)
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

    # -- parts ---------------------------------------------------------------------
    def setup_problems(self, code: str) -> list[str]:
        """Why ``code`` cannot be inspected (empty = ready)."""
        part = self.parts.get(code)
        if part is None:
            return ["unknown part number"]
        return part.setup_problems(self.cfg.detector.backend)

    def _detector_key(self, part: Part) -> tuple:
        if self.cfg.detector.backend == "template":
            index = part.dir / "index.json"
            return ("template", index.stat().st_mtime_ns if index.is_file() else 0)
        return ("yolo", part.active_model)

    def detector_for(self, code: str) -> Detector:
        """The part's detector, loaded once and reused until its model changes."""
        part = self.parts.get(code)
        if part is None:
            raise KeyError(f"unknown part number {code!r}")
        key = self._detector_key(part)
        with self._det_lock:
            cached = self._detectors.get(code)
            if cached is not None and cached[0] == key:
                return cached[1]
            det = self.detector_factory(self.cfg, part)
            self._detectors[code] = (key, det)
            return det

    def prepare(self, code: str) -> list[str]:
        """Load the part's model ahead of the first inspection. Returns the setup problems."""
        problems = self.setup_problems(code)
        if not problems:
            try:
                self.detector_for(code)
            except Exception as exc:
                log.exception("Loading the detector of %s failed", code)
                problems = [f"model could not be loaded ({exc})"]
        return problems

    def log_model_change(self, kind: str, actor: str, part: str, detail: str = "") -> int:
        return self.store.log_event(kind, actor, detail, part_number=part)

    # -- inspection ---------------------------------------------------------------------
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
            # The part's model is loaded before the capture (normally it is already, see prepare()).
            part = self.parts.get(part_number)
            problems = self.setup_problems(part_number)
            detector = None
            if not problems:
                try:
                    detector = self.detector_for(part_number)
                except Exception as exc:
                    log.exception("Loading the detector of %s failed", part_number)
                    problems = [f"model could not be loaded ({exc})"]
            frame = image if image is not None else self.source.capture()
            t_capture = time.perf_counter()
            report, dets = self.inspector.inspect(
                frame, part_number, part.master if part else None, detector,
                part.taxonomy() if part else None, operator_id, "; ".join(problems))

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
                    self.store.log_event("unlock_pass", operator_id, f"re-inspection passed ({part_number})", iid,
                                         part_number=part_number)
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
            self.store.log_event("ack_failed", supervisor_id, "wrong PIN", st.inspection_id, part_number=st.part_number)
            return False
        self.store.log_event("ack", supervisor_id, f"NG acknowledged: {st.reason} ({st.part_number})", st.inspection_id,
                             part_number=st.part_number)
        self.alert.idle()
        return True
