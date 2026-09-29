"""UI smoke test (offscreen). Skipped when PySide6 is not installed."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fc_comparator.alert import AlertController, ConsoleAlert  # noqa: E402
from fc_comparator.capture import FileSource  # noqa: E402
from fc_comparator.config import load_config  # noqa: E402
from fc_comparator.pipeline import Station  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_main_window_inspection_flow(app, tmp_path):
    from fc_comparator.ui.main_window import MainWindow

    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg.storage.database = str(tmp_path / "db.sqlite")
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.path = tmp_path / "config.yaml"  # never overwrite the sample config
    station = Station(cfg, source=FileSource(ROOT / "samples" / "boards" / "board_P001_mixed.jpg"),
                      alert=AlertController(ConsoleAlert(), cfg.alert))
    station.open()
    win = MainWindow(station)
    win.show()
    screen = win.inspect
    screen.part.setCurrentText("P001")
    screen.scan.setText("OP:77")
    screen.scan.returnPressed.emit()
    assert screen.operator.text() == "77"

    screen.inspect()
    QThreadPool.globalInstance().waitForDone(10000)
    app.processEvents()
    app.processEvents()
    assert "NG" in screen.banner.text()
    assert screen.table.rowCount() == 2
    assert station.lock.locked and screen.ack_btn.isVisibleTo(win)
    assert not win.nav_buttons["settings"].isEnabled()

    for key in ("history",):
        win.go(key)
    win.history.refresh()
    assert win.history.table.rowCount() == 1
    win.close()
    station.close()
