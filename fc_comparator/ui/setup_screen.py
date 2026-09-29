"""Setup / teach screen: reference image, ROI drawing, labeled crops, alignment test."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..capture.sources import read_image
from ..classify import crop_roi
from ..classify.dataset import dataset_counts, dataset_warnings, harvest_board, save_crop
from ..config import save_config
from ..models import CLASSIFIER_LABELS, FORK, Roi
from ..pipeline import Station
from .widgets import big_button, bgr_to_qimage, confirm, error_box, info_box

log = logging.getLogger(__name__)


class RoiCanvas(QWidget):
    """Displays an image with ROI boxes; drag to draw a box, tap to select one."""

    roi_drawn = Signal(int, int, int, int)  # x, y, w, h in image pixels
    roi_clicked = Signal(object)  # (cable, row) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(False)
        self.image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self.rois: dict[tuple[int, int], Roi] = {}
        self.selected: tuple[int, int] | None = None
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None

    def set_image(self, img: np.ndarray) -> None:
        self.image = img
        self._pixmap = QPixmap.fromImage(bgr_to_qimage(img))
        self.update()

    def _geometry(self) -> tuple[float, float, float]:
        ih, iw = self.image.shape[:2]
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def _to_image(self, p: QPointF) -> tuple[float, float]:
        s, ox, oy = self._geometry()
        ih, iw = self.image.shape[:2]
        return min(max((p.x() - ox) / s, 0), iw - 1), min(max((p.y() - oy) / s, 0), ih - 1)

    def paintEvent(self, _event):  # noqa: N802
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#1a202c"))
        if self.image is None:
            qp.setPen(QColor("#a0aec0"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Capture or load an image")
            return
        s, ox, oy = self._geometry()
        ih, iw = self.image.shape[:2]
        qp.drawPixmap(QRectF(ox, oy, iw * s, ih * s), self._pixmap, QRectF(0, 0, iw, ih))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        qp.setFont(font)
        for key, r in self.rois.items():
            sel = key == self.selected
            qp.setPen(QPen(QColor("#f6e05e" if sel else "#00d1ff"), 3 if sel else 2))
            rect = QRectF(ox + r.x * s, oy + r.y * s, r.w * s, r.h * s)
            qp.drawRect(rect)
            qp.drawText(rect.topLeft() + QPointF(4, 16), f"C{r.cable}R{r.row}")
        if self._drag_start is not None and self._drag_now is not None:
            qp.setPen(QPen(QColor("#f6e05e"), 2, Qt.PenStyle.DashLine))
            qp.drawRect(QRectF(self._drag_start, self._drag_now).normalized())

    def mousePressEvent(self, e):  # noqa: N802
        if self.image is not None and e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._drag_now = e.position()

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag_start is not None:
            self._drag_now = e.position()
            self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._drag_start is None:
            return
        a, b = self._to_image(self._drag_start), self._to_image(e.position())
        self._drag_start = self._drag_now = None
        x0, y0 = int(min(a[0], b[0])), int(min(a[1], b[1]))
        w, h = int(abs(a[0] - b[0])), int(abs(a[1] - b[1]))
        if w >= 8 and h >= 8:
            self.roi_drawn.emit(x0, y0, w, h)
        else:  # a tap: select the ROI under the finger
            hit = next((k for k, r in self.rois.items() if r.x <= a[0] <= r.x + r.w and r.y <= a[1] <= r.y + r.h), None)
            self.roi_clicked.emit(hit)
        self.update()


class SetupScreen(QWidget):
    config_changed = Signal()

    def __init__(self, station: Station, parent=None):
        super().__init__(parent)
        self.station = station
        self.cfg = station.cfg
        self.canvas = RoiCanvas()
        self.canvas.rois = self.cfg.roi_map()
        self.canvas.roi_drawn.connect(self._on_drawn)
        self.canvas.roi_clicked.connect(self._on_clicked)
        self._dirty = False

        # image group
        g_img = QGroupBox("1. Image")
        v = QVBoxLayout(g_img)
        b_cap = big_button("Capture from camera")
        b_cap.clicked.connect(self.capture)
        b_load = big_button("Load image file...")
        b_load.clicked.connect(self.load)
        b_ref = big_button("Save as reference image")
        b_ref.clicked.connect(self.save_reference)
        self.align_chk = QCheckBox("Align images to reference (ORB + homography)")
        self.align_chk.setChecked(self.cfg.alignment.enabled)
        self.align_chk.toggled.connect(self._toggle_align)
        self.img_info = QLabel()
        self.img_info.setWordWrap(True)
        for w in (b_cap, b_load, b_ref, self.align_chk, self.img_info):
            v.addWidget(w)

        # ROI group
        g_roi = QGroupBox("2. Inspection positions (ROIs)")
        v = QVBoxLayout(g_roi)
        self.target = QComboBox()
        self._fill_targets()
        self.target.currentIndexChanged.connect(self._on_target)
        v.addWidget(QLabel("Drag a box on the image for the selected position:"))
        v.addWidget(self.target)
        row = QHBoxLayout()
        b_grid = big_button("Auto grid")
        b_grid.setToolTip("Draw the first (C1R1) and last position, then generate all others evenly spaced")
        b_grid.clicked.connect(self.auto_grid)
        b_del = big_button("Delete selected")
        b_del.clicked.connect(self.delete_selected)
        row.addWidget(b_grid)
        row.addWidget(b_del)
        v.addLayout(row)
        self.b_save_roi = big_button("Save ROIs")
        self.b_save_roi.clicked.connect(self.save_rois)
        v.addWidget(self.b_save_roi)

        # teach group
        g_teach = QGroupBox("3. Teach clip samples")
        v = QVBoxLayout(g_teach)
        self.label_combo = QComboBox()
        self.label_combo.addItems([lbl for lbl in CLASSIFIER_LABELS if lbl != FORK])
        b_crop = big_button("Save crop of selected position as label")
        b_crop.clicked.connect(self.save_selected_crop)
        v.addWidget(self.label_combo)
        v.addWidget(b_crop)
        v.addWidget(QLabel("Known-good board of part:"))
        hl = QHBoxLayout()
        self.harvest_part = QComboBox()
        self.fork_dir = QComboBox()
        self.fork_dir.addItems(["forks face left", "forks face right"])
        hl.addWidget(self.harvest_part)
        hl.addWidget(self.fork_dir)
        v.addLayout(hl)
        b_harvest = big_button("Save all positions from this board")
        b_harvest.clicked.connect(self.harvest)
        v.addWidget(b_harvest)
        self.ds_info = QLabel()
        self.ds_info.setWordWrap(True)
        v.addWidget(self.ds_info)
        b_reload = big_button("Reload classifier with new samples")
        b_reload.clicked.connect(self.reload_classifier)
        v.addWidget(b_reload)

        panel = QWidget()
        pv = QVBoxLayout(panel)
        for g in (g_img, g_roi, g_teach):
            pv.addWidget(g)
        pv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(520)

        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(scroll)

        self.refresh()

    # -- helpers -----------------------------------------------------------
    def refresh(self) -> None:
        self.harvest_part.clear()
        self.harvest_part.addItems(sorted(self.cfg.parts))
        self._update_dataset_info()
        ref = self.cfg.resolve(self.cfg.reference_image)
        if self.canvas.image is None and ref.is_file():
            self.canvas.set_image(read_image(ref))
            self.img_info.setText(f"Showing reference image {ref.name}")

    def _fill_targets(self) -> None:
        self.target.clear()
        for c in range(1, self.cfg.station.cables + 1):
            for r in range(1, self.cfg.station.rows + 1):
                have = "✓" if (c, r) in self.canvas.rois else " "
                self.target.addItem(f"{have} Cable {c}  Row {r}", (c, r))

    def _target_key(self) -> tuple[int, int] | None:
        return self.target.currentData()

    def _on_target(self, _i: int) -> None:
        self.canvas.selected = self._target_key()
        self.canvas.update()

    def _on_drawn(self, x: int, y: int, w: int, h: int) -> None:
        key = self._target_key()
        if key is None:
            return
        self.canvas.rois[key] = Roi(key[0], key[1], x, y, w, h)
        self._dirty = True
        idx = self.target.currentIndex()
        self._fill_targets()
        self.target.setCurrentIndex(min(idx + 1, self.target.count() - 1))  # advance to the next position
        self.canvas.update()

    def _on_clicked(self, key) -> None:
        if key is None:
            return
        for i in range(self.target.count()):
            if self.target.itemData(i) == key:
                self.target.setCurrentIndex(i)
                break

    def _set_teach_image(self, img: np.ndarray, source: str) -> None:
        """Show an image for teaching; aligned to the reference so the ROIs fit."""
        aligner = self.station.inspector.aligner
        msg = source
        if self.cfg.alignment.enabled and aligner is not None:
            img, info = aligner.align(img)
            msg += f" - alignment: {info.message} ({info.shift_px:.1f}px)"
        self.canvas.set_image(img)
        self.img_info.setText(msg)

    def _update_dataset_info(self) -> None:
        ds = self.cfg.resolve(self.cfg.classifier.dataset_dir)
        counts = dataset_counts(ds)
        text = "Samples: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "none yet")
        warns = dataset_warnings(ds)
        if warns:
            text += "\n⚠ " + "\n⚠ ".join(warns)
        text += f"\nClassifier in use: {self.station.inspector.classifier.name}"
        self.ds_info.setText(text)

    # -- actions -----------------------------------------------------------
    def capture(self) -> None:
        try:
            img = self.station.source.capture(timeout=3)
        except Exception as exc:
            error_box(self, f"Capture failed: {exc}")
            return
        self._set_teach_image(img, "Captured from camera")

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            try:
                self._set_teach_image(read_image(path), f"Loaded {path}")
            except Exception as exc:
                error_box(self, str(exc))

    def save_reference(self) -> None:
        if self.canvas.image is None:
            return
        if not confirm(self, "Use the displayed image as the new reference?\nROIs are defined on the reference image."):
            return
        path = self.station.save_reference(self.canvas.image)
        self.station.reload()
        self.img_info.setText(f"Reference saved to {path}")
        self.config_changed.emit()

    def _toggle_align(self, on: bool) -> None:
        self.cfg.alignment.enabled = on
        save_config(self.cfg)
        self.station.reload()
        self.config_changed.emit()

    def auto_grid(self) -> None:
        n, m = self.cfg.station.cables, self.cfg.station.rows
        first, last = self.canvas.rois.get((1, 1)), self.canvas.rois.get((n, m))
        if first is None or last is None:
            error_box(self, f"Draw C1R1 and C{n}R{m} first.")
            return
        dx = (last.x - first.x) / (n - 1) if n > 1 else 0
        dy = (last.y - first.y) / (m - 1) if m > 1 else 0
        for c in range(1, n + 1):
            for r in range(1, m + 1):
                self.canvas.rois[(c, r)] = Roi(c, r, round(first.x + (c - 1) * dx), round(first.y + (r - 1) * dy), first.w, first.h)
        self._dirty = True
        self._fill_targets()
        self.canvas.update()

    def delete_selected(self) -> None:
        key = self._target_key()
        if key in self.canvas.rois:
            del self.canvas.rois[key]
            self._dirty = True
            self._fill_targets()
            self.canvas.update()

    def save_rois(self) -> None:
        rois = sorted(self.canvas.rois.values(), key=lambda r: (r.cable, r.row))
        old = self.cfg.rois
        self.cfg.rois = rois
        problems = [p for p in self.cfg.validate() if "ROI" in p or "positions have no ROI" in p]
        if problems and not confirm(self, "Problems:\n" + "\n".join(problems) + "\n\nSave anyway?"):
            self.cfg.rois = old
            return
        save_config(self.cfg)
        self.station.reload()
        self.station.store.log_event("setup", "", f"ROIs saved ({len(rois)})")
        self._dirty = False
        info_box(self, f"Saved {len(rois)} ROIs.")
        self.config_changed.emit()

    def save_selected_crop(self) -> None:
        key = self._target_key()
        roi = self.canvas.rois.get(key) if key else None
        if self.canvas.image is None or roi is None:
            error_box(self, "Select a position that has an ROI (and an image).")
            return
        label = self.label_combo.currentText()
        crop = crop_roi(self.canvas.image, roi, self.cfg.classifier.roi_padding)
        path = save_crop(self.cfg.resolve(self.cfg.classifier.dataset_dir), label, crop, f"c{roi.cable}r{roi.row}")
        self.img_info.setText(f"Saved {label} sample: {path.name}")
        self._update_dataset_info()

    def harvest(self) -> None:
        part = self.cfg.parts.get(self.harvest_part.currentText())
        if self.canvas.image is None or part is None:
            error_box(self, "Load a board image and pick its part number.")
            return
        orient = "left" if self.fork_dir.currentIndex() == 0 else "right"
        if not confirm(self, f"Is every clip on this board correct for {part.code}?\nAll positions will be saved as samples."):
            return
        rois = list(self.canvas.rois.values())
        paths = harvest_board(
            self.canvas.image, rois, part.pattern, self.cfg.resolve(self.cfg.classifier.dataset_dir),
            fork_orientation=orient, stem=part.code, padding=self.cfg.classifier.roi_padding,
        )
        self.station.store.log_event("teach", "", f"harvested {len(paths)} crops from {part.code}")
        self.img_info.setText(f"Saved {len(paths)} samples from {part.code}")
        self._update_dataset_info()

    def reload_classifier(self) -> None:
        try:
            self.station.reload()
        except Exception as exc:
            error_box(self, f"Reload failed: {exc}")
            return
        self._update_dataset_info()
        info_box(self, f"Classifier reloaded: {self.station.inspector.classifier.name}")

    @property
    def dirty(self) -> bool:
        return self._dirty
