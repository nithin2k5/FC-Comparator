"""Inspect screen: live preview, part selection/scan, INSPECT, OK/NG banner, mismatch list, NG lock."""

from __future__ import annotations

import logging
from datetime import date

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..barcode import parse_scan
from ..capture import FileSource
from ..capture.sources import read_image
from ..pipeline import Station, StationResult
from .widgets import (
    IDLE_COLOR,
    NG_COLOR,
    OK_COLOR,
    WARN_COLOR,
    Banner,
    ImageView,
    PinDialog,
    Task,
    big_button,
    error_box,
)

log = logging.getLogger(__name__)


class InspectScreen(QWidget):
    trigger_requested = Signal()  # thread-safe entry point (GPIO pedal callback runs on another thread)
    lock_changed = Signal()  # lock listeners run on the inspection worker thread

    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self.station = station
        self.cfg = station.cfg
        self._busy = False
        self._showing_result = False
        self._tasks: set[Task] = set()
        self._file_image = None  # image loaded via "Load image" while in camera mode

        # -- left: image ------------------------------------------------------
        self.view = ImageView(placeholder="Waiting for camera...")
        self.live_btn = big_button("Live view")
        self.live_btn.clicked.connect(self.show_live)
        self.load_btn = big_button("Load image...")
        self.load_btn.clicked.connect(self.load_image)
        img_btns = QHBoxLayout()
        img_btns.addWidget(self.live_btn)
        img_btns.addWidget(self.load_btn)
        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        left.addLayout(img_btns)

        # -- right: controls ----------------------------------------------------
        self.banner = Banner()
        self.scan = QLineEdit()
        self.scan.setPlaceholderText("Scan barcode (part number or OP:operator)")
        self.scan.returnPressed.connect(self._on_scan)
        self.part = QComboBox()
        self.part.currentTextChanged.connect(self._on_part_changed)
        self.operator = QLineEdit()
        self.operator.setPlaceholderText("Operator ID")
        self.pattern_lbl = QLabel()
        self.pattern_lbl.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Scan", self.scan)
        form.addRow("Part number", self.part)
        form.addRow("Master pattern", self.pattern_lbl)
        form.addRow("Operator", self.operator)

        self.inspect_btn = big_button(f"INSPECT  [{self.cfg.trigger.key}]", "primary")
        self.inspect_btn.clicked.connect(self.inspect)
        self.ack_btn = big_button("Supervisor acknowledge (PIN)", "danger")
        self.ack_btn.clicked.connect(self.acknowledge)
        self.ack_btn.hide()

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Cable", "Row", "Expected", "Found", "Conf."])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.stats_lbl = QLabel()
        self.timing_lbl = QLabel()
        self.timing_lbl.setStyleSheet("color: #718096;")

        right = QVBoxLayout()
        right.addWidget(self.banner)
        right.addLayout(form)
        right.addWidget(self.inspect_btn)
        right.addWidget(self.ack_btn)
        right.addWidget(QLabel("Mismatches"))
        right.addWidget(self.table, 1)
        right.addWidget(self.stats_lbl)
        right.addWidget(self.timing_lbl)
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setMaximumWidth(620)

        lay = QHBoxLayout(self)
        lay.addLayout(left, 1)
        lay.addWidget(right_w)

        self.trigger_requested.connect(self.inspect)
        self.lock_changed.connect(self._update_lock_ui)
        self.station.lock.subscribe(lambda _state: self.lock_changed.emit())

        self.preview = QTimer(self)
        self.preview.timeout.connect(self._refresh_preview)
        self.preview.start(max(30, int(1000 / max(1, self.cfg.ui.preview_fps))))

        self.refresh_parts()
        self._update_lock_ui()
        self._update_stats()

    # -- parts / scanning ------------------------------------------------------
    def refresh_parts(self) -> None:
        current = self.part.currentText()
        self.part.blockSignals(True)
        self.part.clear()
        self.part.addItems(sorted(self.cfg.parts))
        if current in self.cfg.parts:
            self.part.setCurrentText(current)
        self.part.blockSignals(False)
        self._on_part_changed(self.part.currentText())

    def _on_part_changed(self, code: str) -> None:
        pn = self.cfg.parts.get(code)
        if pn is None:
            self.pattern_lbl.setText("-" if self.cfg.compare.mode != "cross" else "(cross-cable mode)")
            return
        rows = "  ".join(f"R{i + 1}: {p}" for i, p in enumerate(pn.pattern))
        self.pattern_lbl.setText(f"{rows}\n{pn.description}" if pn.description else rows)

    def _on_scan(self) -> None:
        text = self.scan.text()
        self.scan.clear()
        s = parse_scan(text, self.cfg.barcode, set(self.cfg.parts))
        if s.kind == "operator":
            self.operator.setText(s.value)
        elif s.kind == "part":
            if self.station.lock.locked and s.value != self.station.lock.state.part_number:
                self.banner.show_state("LOCKED", NG_COLOR, "Resolve the NG before changing part")
                return
            self.part.setCurrentText(s.value)
            self.banner.show_state("READY", IDLE_COLOR, s.value)
        else:
            self.banner.show_state("UNKNOWN", WARN_COLOR, f"Barcode not recognised: {text.strip()[:30]}")

    def focus_scan(self) -> None:
        self.scan.setFocus()

    # -- preview -------------------------------------------------------------
    def _refresh_preview(self) -> None:
        if self._showing_result or not self.isVisible():
            return
        frame = self._file_image if self._file_image is not None else self.station.source.latest()
        if frame is not None:
            self.view.set_image(frame)
        err = getattr(self.station.source, "error", "")
        if err:
            self.timing_lbl.setText(f"Camera: {err}")

    def show_live(self) -> None:
        self._showing_result = False
        self._file_image = None
        self._refresh_preview()

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load board image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            error_box(self, str(exc))
            return
        if isinstance(self.station.source, FileSource):
            self.station.source.set_image(img)
        else:
            self._file_image = img  # camera mode: inspect this file once instead of a live frame
        self._showing_result = False
        self.view.set_image(img)

    # -- inspection ---------------------------------------------------------
    def inspect(self) -> None:
        if self._busy:
            return
        part = self.part.currentText()
        if not part and self.cfg.compare.mode != "cross":
            self.banner.show_state("NO PART", WARN_COLOR, "Scan or select a part number")
            return
        ok, why = self.station.can_inspect(part)
        if not ok:
            self.banner.show_state("LOCKED", NG_COLOR, why)
            return
        self._busy = True
        self.inspect_btn.setEnabled(False)
        self.banner.show_state("INSPECTING...", IDLE_COLOR)
        image = self._file_image
        operator = self.operator.text().strip()

        task = Task(lambda: self.station.inspect(part, operator, image=image), self._on_result, self._on_error)
        self._tasks.add(task)
        task.signals.done.connect(lambda _r, t=task: self._tasks.discard(t))
        task.signals.failed.connect(lambda _e, t=task: self._tasks.discard(t))
        QThreadPool.globalInstance().start(task)

    def _on_result(self, res: StationResult) -> None:
        self._busy = False
        self.inspect_btn.setEnabled(True)
        self._file_image = None
        rep = res.report
        self._showing_result = True
        self.view.set_image(res.annotated)
        if rep.ok:
            self.banner.show_state("OK", OK_COLOR, f"{rep.part_number}  all {len(rep.positions)} positions correct")
        else:
            detail = rep.error or f"{len(rep.mismatches)} position(s) wrong"
            self.banner.show_state("NG", NG_COLOR, detail)
        self.table.setRowCount(0)
        for p in rep.mismatches:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate([p.cable, p.row, p.expected, p.found, f"{p.confidence:.0%}"]):
                item = QTableWidgetItem(str(v))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QColor(NG_COLOR))
                item.setToolTip(p.reason)
                self.table.setItem(r, c, item)
        alert_err = self.station.alert.last_error
        self.timing_lbl.setText(
            f"#{res.inspection_id}  inspection {rep.duration_ms:.0f} ms, alert after {res.alert_latency_ms:.0f} ms, "
            f"{rep.classifier}, {rep.alignment.message}" + (f"  ALERT FAULT: {alert_err}" if alert_err else "")
        )
        self._update_lock_ui()
        self._update_stats()
        self.focus_scan()

    def _on_error(self, message: str) -> None:
        self._busy = False
        self.inspect_btn.setEnabled(True)
        if "StationLocked" in message:
            self.banner.show_state("LOCKED", NG_COLOR, message.split(":", 1)[-1].strip())
            return
        log.error("Inspection failed: %s", message)
        self.station.alert.ng()  # a failed inspection is never silently OK
        self.banner.show_state("ERROR", NG_COLOR, message[:80])

    # -- lock ------------------------------------------------------------------
    def _update_lock_ui(self) -> None:
        st = self.station.lock.state
        self.ack_btn.setVisible(st.locked)
        self.part.setEnabled(not st.locked)
        if st.locked:
            self.inspect_btn.setText(f"RE-INSPECT {st.part_number}  [{self.cfg.trigger.key}]")
        else:
            self.inspect_btn.setText(f"INSPECT  [{self.cfg.trigger.key}]")
        win = self.window()
        if hasattr(win, "set_navigation_locked"):
            win.set_navigation_locked(st.locked)

    def acknowledge(self) -> None:
        pin = PinDialog.ask(self, "Supervisor PIN", numeric_only=True)
        if pin is None:
            return
        supervisor = self.operator.text().strip()
        if self.station.acknowledge(pin, supervisor):
            self.banner.show_state("ACKNOWLEDGED", WARN_COLOR, "Station unlocked by supervisor")
        else:
            self.banner.show_state("WRONG PIN", NG_COLOR, "Station remains locked")
        self._update_lock_ui()

    def _update_stats(self) -> None:
        try:
            rep = self.station.store.daily_report(date.today())
        except Exception:
            return
        self.stats_lbl.setText(f"Today: {rep.total} inspected, {rep.ok} OK, {rep.ng} NG ({rep.ng_rate:.1%})")

    def stop(self) -> None:
        self.preview.stop()
