"""History / reports screen: browse inspections, view evidence, export, daily report."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..capture.sources import read_image
from ..pipeline import Station
from ..storage import export_csv, export_excel
from .widgets import NG_COLOR, ImageView, big_button, error_box, info_box

COLUMNS = ["id", "ts", "part_number", "operator_id", "result", "ng_count", "duration_ms"]
HEADERS = ["#", "Time", "Part", "Operator", "Result", "NG pos.", "ms"]


class HistoryScreen(QWidget):
    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self.station = station
        self.store = station.store
        today = QDate.currentDate()
        self.d_from = QDateEdit(today)
        self.d_to = QDateEdit(today)
        for d in (self.d_from, self.d_to):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
        self.part = QComboBox()
        self.result = QComboBox()
        self.result.addItems(["All", "OK", "NG"])
        b_refresh = big_button("Refresh")
        b_refresh.clicked.connect(self.refresh)
        b_csv = big_button("Export CSV")
        b_csv.clicked.connect(lambda: self.export("csv"))
        b_xlsx = big_button("Export Excel")
        b_xlsx.clicked.connect(lambda: self.export("xlsx"))
        filters = QHBoxLayout()
        for lbl, w in (("From", self.d_from), ("To", self.d_to), ("Part", self.part), ("Result", self.result)):
            filters.addWidget(QLabel(lbl))
            filters.addWidget(w)
        for b in (b_refresh, b_csv, b_xlsx):
            filters.addWidget(b)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_select)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.image = ImageView(placeholder="No saved image for this inspection")
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setStyleSheet("font-family: Consolas, monospace; font-size: 15px;")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.image)
        right.addWidget(self.detail)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        split.addWidget(right)
        split.setSizes([700, 600])

        lay = QVBoxLayout(self)
        lay.addLayout(filters)
        lay.addWidget(split, 3)
        lay.addWidget(QLabel("Daily report (for the 'To' date)"))
        lay.addWidget(self.report, 1)

    def _range(self) -> tuple[datetime, datetime]:
        start = datetime.combine(self.d_from.date().toPython(), time.min)
        end = datetime.combine(self.d_to.date().toPython(), time.min) + timedelta(days=1)
        return start, end

    def _filters(self) -> dict:
        f = {}
        if self.part.currentText() not in ("", "All"):
            f["part_number"] = self.part.currentText()
        if self.result.currentText() != "All":
            f["result"] = self.result.currentText()
        return f

    def refresh(self) -> None:
        current_part = self.part.currentText()
        self.part.blockSignals(True)
        self.part.clear()
        self.part.addItems(["All", *sorted(self.station.cfg.parts)])
        if current_part:
            self.part.setCurrentText(current_part)
        self.part.blockSignals(False)

        start, end = self._range()
        rows = self.store.inspections(start, end, limit=2000, **self._filters())
        self.table.setRowCount(0)
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, key in enumerate(COLUMNS):
                v = row[key]
                if key == "ts":
                    v = str(v).replace("T", " ")[:19]
                elif key == "duration_ms":
                    v = f"{v:.0f}"
                item = QTableWidgetItem(str(v))
                if row["result"] == "NG":
                    item.setForeground(QColor(NG_COLOR))
                self.table.setItem(r, c, item)
        self.report.setPlainText(self.store.daily_report(self.d_to.date().toPython()).as_text())

    def _on_select(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        iid = int(self.table.item(sel[0].row(), 0).text())
        row = self.store.get_inspection(iid)
        if row is None:
            return
        lines = [
            f"Inspection #{iid}  {row['result']}  part {row['part_number']}  operator {row['operator_id'] or '-'}",
            f"{row['ts']}  {row['duration_ms']:.0f} ms  classifier {row['classifier']}  alignment: {row['align_message']}",
        ]
        if row["error"]:
            lines.append(f"Error: {row['error']}")
        for p in self.store.positions(iid):
            mark = "OK" if p["ok"] else "NG"
            lines.append(
                f"  {mark} C{p['cable']}R{p['row']}  expected {p['expected']:<10} found {p['found']:<10} "
                f"{p['confidence']:.0%}  {p['reason']}"
            )
        self.detail.setPlainText("\n".join(lines))
        path = row["image_path"]
        if path and Path(path).is_file():
            try:
                self.image.set_image(read_image(path))
            except Exception:
                self.image.clear()
        else:
            self.image.clear()
            self.image.setText("No saved image for this inspection")

    def export(self, kind: str) -> None:
        start, end = self._range()
        default = f"inspections_{start:%Y%m%d}_{(end - timedelta(days=1)):%Y%m%d}.{kind}"
        filt = "Excel (*.xlsx)" if kind == "xlsx" else "CSV (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "Export", default, filt)
        if not path:
            return
        try:
            if kind == "xlsx":
                out = export_excel(self.store, path, start, end, **self._filters())
                info_box(self, f"Exported to {out}")
            else:
                a, b = export_csv(self.store, path, start, end, **self._filters())
                info_box(self, f"Exported to\n{a}\n{b}")
        except Exception as exc:
            error_box(self, f"Export failed: {exc}")
