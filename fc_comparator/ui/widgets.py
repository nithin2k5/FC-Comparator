"""Reusable touch-friendly widgets."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

OK_COLOR = "#1e9e3a"
NG_COLOR = "#d62828"
IDLE_COLOR = "#4a5568"
WARN_COLOR = "#d98e04"

STYLE = """
QWidget { font-size: 17px; }
QPushButton { min-height: 56px; padding: 6px 14px; border-radius: 8px; background: #e2e8f0; border: 1px solid #a0aec0; }
QPushButton:pressed { background: #cbd5e0; }
QPushButton:checked { background: #2b6cb0; color: white; }
QPushButton:disabled { color: #a0aec0; }
QPushButton#primary { background: #2b6cb0; color: white; font-size: 26px; font-weight: bold; min-height: 90px; }
QPushButton#danger { background: #c53030; color: white; font-weight: bold; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit { min-height: 44px; font-size: 18px; }
QTableWidget { font-size: 16px; }
QHeaderView::section { padding: 6px; font-weight: bold; }
"""


def bgr_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QLabel):
    """Shows a BGR image scaled to fit, keeping the aspect ratio."""

    def __init__(self, parent=None, placeholder: str = "No image"):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1a202c; color: #a0aec0; font-size: 22px;")
        self._pixmap: QPixmap | None = None

    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            return
        self._pixmap = QPixmap.fromImage(bgr_to_qimage(img))
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._rescale()


class Banner(QLabel):
    """Large OK / NG / status banner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(110)
        f = QFont()
        f.setPointSize(40)
        f.setBold(True)
        self.setFont(f)
        self.show_state("READY", IDLE_COLOR)

    def show_state(self, text: str, color: str, detail: str = "") -> None:
        self.setText(text if not detail else f"{text}\n{detail}")
        size = 40 if not detail else 30
        self.setStyleSheet(f"background: {color}; color: white; border-radius: 10px; font-size: {size}pt; font-weight: bold;")


class PinDialog(QDialog):
    """Touch keypad for PINs/passwords (a physical keyboard works too)."""

    def __init__(self, title: str, parent=None, numeric_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(title))
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setMinimumHeight(56)
        self.edit.returnPressed.connect(self.accept)
        lay.addWidget(self.edit)
        grid = QGridLayout()
        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "⌫", "0", "OK"]
        for i, k in enumerate(keys):
            b = QPushButton(k)
            b.setMinimumSize(90, 70)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, key=k: self._key(key))
            grid.addWidget(b, i // 3, i % 3)
        lay.addLayout(grid)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        lay.addWidget(cancel)
        if numeric_only:
            from PySide6.QtGui import QIntValidator

            self.edit.setValidator(QIntValidator(0, 99999999))

    def _key(self, key: str) -> None:
        if key == "OK":
            self.accept()
        elif key == "⌫":
            self.edit.backspace()
        else:
            self.edit.insert(key)

    @staticmethod
    def ask(parent, title: str, numeric_only: bool = False) -> str | None:
        d = PinDialog(title, parent, numeric_only)
        d.edit.setFocus()
        return d.edit.text() if d.exec() == QDialog.DialogCode.Accepted else None


def big_button(text: str, object_name: str = "", checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    if object_name:
        b.setObjectName(object_name)
    b.setCheckable(checkable)
    return b


def error_box(parent: QWidget, text: str, title: str = "Error") -> None:
    QMessageBox.critical(parent, title, text)


def info_box(parent: QWidget, text: str, title: str = "Information") -> None:
    QMessageBox.information(parent, title, text)


def confirm(parent: QWidget, text: str, title: str = "Confirm") -> bool:
    return QMessageBox.question(parent, title, text) == QMessageBox.StandardButton.Yes


class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Run ``fn`` on the thread pool; results come back on the GUI thread via signals."""

    def __init__(self, fn: Callable[[], object], on_done: Callable[[object], None], on_error: Callable[[str], None]):
        super().__init__()
        self.fn = fn
        self.signals = _TaskSignals()
        self.signals.done.connect(on_done)
        self.signals.failed.connect(on_error)

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # delivered to the GUI, never lost
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.done.emit(result)
