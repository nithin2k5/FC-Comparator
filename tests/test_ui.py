"""tkinter UI tests: drive the real widgets (skipped where no display is available)."""

import time
import tkinter as tk
from pathlib import Path

import pytest

from fc_comparator.alert import AlertController, ConsoleAlert
from fc_comparator.capture import FileSource
from fc_comparator.capture.sources import read_image
from fc_comparator.config import load_config
from fc_comparator.pipeline import Station

ROOT = Path(__file__).resolve().parents[1]
BOARDS = ROOT / "samples" / "boards"

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def tk_root():
    # One interpreter per module; creation is retried because loading Tk's library
    # files occasionally fails transiently on Windows (e.g. while a scanner holds them).
    error = None
    for _ in range(3):
        try:
            r = tk.Tk()
            break
        except tk.TclError as exc:
            error = exc
            time.sleep(0.5)
    else:
        pytest.skip(f"no usable Tk display: {error}")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def root(tk_root):
    top = tk.Toplevel(tk_root)
    yield top
    top.destroy()


@pytest.fixture
def dialogs(monkeypatch):
    """Replace modal dialogs with recorders so tests never block."""
    from fc_comparator.ui import main_window, inspect_screen, parts_screen, settings_screen, setup_screen

    shown = {"error": [], "info": [], "pins": []}
    for mod in (main_window, inspect_screen, parts_screen, settings_screen, setup_screen):
        if hasattr(mod, "error_box"):
            monkeypatch.setattr(mod, "error_box", lambda _p, text, *a: shown["error"].append(text))
        if hasattr(mod, "info_box"):
            monkeypatch.setattr(mod, "info_box", lambda _p, text, *a: shown["info"].append(text))
        if hasattr(mod, "confirm"):
            monkeypatch.setattr(mod, "confirm", lambda *_a: True)

    class FakePin:
        @staticmethod
        def ask(_parent, _title, numeric_only=False):
            return shown["pins"].pop(0) if shown["pins"] else None

    for mod in (main_window, inspect_screen, settings_screen):
        monkeypatch.setattr(mod, "PinDialog", FakePin)
    return shown


@pytest.fixture
def app(root, tmp_path, dialogs):
    from fc_comparator.ui.main_window import MainWindow
    from fc_comparator.ui.widgets import setup_style

    cfg = load_config(ROOT / "config" / "config.yaml")
    # The config is saved to tmp (never overwrite the sample), so pin relative paths first.
    cfg.reference_image = str(cfg.resolve(cfg.reference_image))
    cfg.camera.file_path = str(cfg.resolve(cfg.camera.file_path))
    cfg.classifier.dataset_dir = str(cfg.resolve(cfg.classifier.dataset_dir))
    cfg.classifier.model_path = str(cfg.resolve(cfg.classifier.model_path))
    cfg.storage.database = str(tmp_path / "db.sqlite")
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.path = tmp_path / "config.yaml"
    setup_style(root)
    station = Station(cfg, source=FileSource(BOARDS / "board_P001_ok.jpg"), alert=AlertController(ConsoleAlert(), cfg.alert))
    station.open()
    win = MainWindow(root, station)
    win.pack(fill="both", expand=True)
    root.update()
    yield win
    win.shutdown()
    station.close()


def wait_idle(root, screen, timeout=15.0):
    end = time.monotonic() + timeout
    while screen.busy and time.monotonic() < end:
        root.update()
        time.sleep(0.01)
    for _ in range(5):
        root.update()
    assert not screen.busy, "inspection did not finish"


def test_inspection_lock_and_acknowledge(root, app, dialogs):
    screen = app.inspect
    screen.scan_var.set("OP:77")
    screen.on_scan()
    assert screen.operator_var.get() == "77"
    screen.scan_var.set("P001")
    screen.on_scan()
    assert screen.part_var.get() == "P001"

    screen.inspect()
    wait_idle(root, screen)
    assert screen.banner.text.startswith("OK")

    app.station.source.load(BOARDS / "board_P001_mixed.jpg")
    app.pedal_pressed()  # foot pedal / F9 path
    wait_idle(root, screen)
    assert screen.banner.text.startswith("NG")
    rows = [screen.table.item(i, "values") for i in screen.table.get_children()]
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [("3", "1", "round", "fork_right"), ("3", "4", "round", "fork_right")]
    assert app.station.lock.locked and screen.ack_btn.winfo_manager() == "pack"
    assert app.nav_buttons["settings"].instate(["disabled"])
    assert not app.nav_buttons["history"].instate(["disabled"])
    assert "RE-INSPECT P001" in screen.inspect_btn.cget("text")

    screen.scan_var.set("P002")  # cannot switch part while locked
    screen.on_scan()
    assert screen.part_var.get() == "P001" and screen.banner.text.startswith("LOCKED")

    dialogs["pins"] += ["0000"]
    screen.acknowledge()
    assert app.station.lock.locked and screen.banner.text.startswith("WRONG PIN")
    dialogs["pins"] += ["1234"]
    screen.acknowledge()
    root.update()
    assert not app.station.lock.locked and screen.ack_btn.winfo_manager() == ""
    assert not app.nav_buttons["settings"].instate(["disabled"])

    app.go("history")
    assert len(app.history.table.get_children()) == 2
    first = app.history.table.get_children()[0]
    app.history.table.selection_set(first)
    root.update()
    assert "NG" in app.history.detail.get("1.0", "end")


def test_password_gate_parts_and_settings(root, app, dialogs):
    assert not app.go("parts")  # dialog cancelled
    dialogs["pins"] += ["9999"]
    assert not app.go("parts") and dialogs["error"][-1] == "Wrong password."
    dialogs["pins"] += ["5678"]
    assert app.go("parts") and app.current == "parts"

    parts = app.parts
    parts.new()
    parts.code_var.set("P003")
    parts.desc_var.set("test part")
    for var, label in zip(parts.row_vars, ["small", "fork_left", "round", "fork"]):
        var.set(label)
    parts.save()
    assert app.cfg.parts["P003"].pattern == ["small", "fork_left", "round", "fork"]
    assert load_config(app.cfg.path).parts["P003"].description == "test part"
    assert "P003" in app.inspect.part.cget("values")

    assert app.go("settings")  # already in setup mode: no password asked again
    app.settings.v["classifier.confidence_threshold"].set(0.7)
    app.settings.v["compare.mode"].set("both")
    app.settings.save()
    assert dialogs["info"][-1] == "Settings saved and applied."
    saved = load_config(app.cfg.path)
    assert saved.classifier.confidence_threshold == pytest.approx(0.7) and saved.compare.mode == "both"

    app.leave_admin()
    assert app.current == "inspect" and not app.admin


def test_setup_screen_roi_editing(root, app, dialogs):
    dialogs["pins"] += ["5678"]
    assert app.go("setup")
    setup = app.setup
    setup.canvas.set_image(read_image(BOARDS / "board_P001_ok.jpg"))
    setup.canvas.rois.clear()
    setup._fill_targets(select=0)
    setup._on_drawn(200, 240, 120, 120)  # C1R1
    assert setup.target_key() == (1, 2)  # auto-advanced
    setup._fill_targets(select=15)
    setup._on_drawn(980, 750, 120, 120)  # C4R4
    setup.auto_grid()
    assert len(setup.canvas.rois) == 16
    assert setup.canvas.rois[(3, 2)].x == 720 and setup.canvas.rois[(3, 2)].y == 410
    setup.save_rois()
    assert dialogs["info"][-1] == "Saved 16 ROIs."
    assert len(load_config(app.cfg.path).rois) == 16
