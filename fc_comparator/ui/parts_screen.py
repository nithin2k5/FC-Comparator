"""Part numbers screen: create, edit and delete part numbers and their master patterns."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import save_config
from ..models import EXPECTED_LABELS, PartNumber
from ..pipeline import Station
from .widgets import big_button, confirm, error_box


class PartsScreen(QWidget):
    config_changed = Signal()

    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self.station = station
        self.cfg = station.cfg

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Part number", "Master pattern (row 1 → n)", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_select)

        editor = QGroupBox("Edit part number")
        form = QFormLayout(editor)
        self.code = QLineEdit()
        self.desc = QLineEdit()
        form.addRow("Part number", self.code)
        form.addRow("Description", self.desc)
        self.row_combos: list[QComboBox] = []
        for r in range(self.cfg.station.rows):
            cb = QComboBox()
            cb.addItems(EXPECTED_LABELS)
            cb.setToolTip("fork = either orientation; fork_left / fork_right = orientation is checked")
            self.row_combos.append(cb)
            form.addRow(f"Row {r + 1}", cb)
        btns = QHBoxLayout()
        b_new = big_button("New")
        b_new.clicked.connect(self.new)
        b_save = big_button("Save")
        b_save.clicked.connect(self.save)
        b_del = big_button("Delete", "danger")
        b_del.clicked.connect(self.delete)
        for b in (b_new, b_save, b_del):
            btns.addWidget(b)
        form.addRow(btns)
        editor.setMaximumWidth(520)

        left = QVBoxLayout()
        left.addWidget(self.table)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 1)
        lay.addWidget(editor)
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
        for code in sorted(self.cfg.parts):
            pn = self.cfg.parts[code]
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate([code, ", ".join(pn.pattern), pn.description]):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        pn = self.cfg.parts.get(self.table.item(rows[0].row(), 0).text())
        if pn is None:
            return
        self.code.setText(pn.code)
        self.desc.setText(pn.description)
        for cb, label in zip(self.row_combos, pn.pattern):
            cb.setCurrentText(label)

    def new(self) -> None:
        self.table.clearSelection()
        self.code.clear()
        self.desc.clear()
        for cb in self.row_combos:
            cb.setCurrentIndex(0)
        self.code.setFocus()

    def save(self) -> None:
        code = self.code.text().strip()
        pn = PartNumber(code, [cb.currentText() for cb in self.row_combos], self.desc.text().strip())
        try:
            pn.validate(self.cfg.station.rows)
        except ValueError as exc:
            error_box(self, str(exc))
            return
        if code in self.cfg.parts and self.cfg.parts[code].pattern != pn.pattern:
            if not confirm(self, f"Change the master pattern of {code}?"):
                return
        old = self.cfg.parts.get(code)
        self.cfg.parts[code] = pn
        save_config(self.cfg)
        self.station.store.log_event(
            "part_saved", "", f"{code}: {old.pattern if old else 'new'} -> {pn.pattern}"
        )
        self.refresh()
        self.config_changed.emit()

    def delete(self) -> None:
        code = self.code.text().strip()
        if code not in self.cfg.parts:
            return
        if self.station.lock.locked and self.station.lock.state.part_number == code:
            error_box(self, "This part is locked after an NG; resolve that first.")
            return
        if not confirm(self, f"Delete part number {code}? History is kept."):
            return
        del self.cfg.parts[code]
        save_config(self.cfg)
        self.station.store.log_event("part_deleted", "", code)
        self.new()
        self.refresh()
        self.config_changed.emit()
