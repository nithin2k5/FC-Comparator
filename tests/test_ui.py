"""UI tests: drive the real CustomTkinter window (skipped where no display is available).

Modal dialogs are replaced by recorders so the tests never block.
"""

import shutil
import time
import tkinter as tk
from pathlib import Path

import pytest

ctk = pytest.importorskip("customtkinter")

from fc_comparator.alert import AlertController, ConsoleAlert  # noqa: E402
from fc_comparator.annotations import AnnotationStore  # noqa: E402
from fc_comparator.capture import FileSource  # noqa: E402
from fc_comparator.capture.sources import write_image  # noqa: E402
from fc_comparator.config import load_config, save_config  # noqa: E402
from fc_comparator.pipeline import Inspector, Station  # noqa: E402
from fc_comparator.synthetic import Geometry, render_marked_board  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOARDS = ROOT / "samples" / "boards"
pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def tk_root():
    error = None
    for _ in range(3):  # loading Tk's library files occasionally fails transiently on Windows
        try:
            r = ctk.CTk()
            break
        except tk.TclError as exc:
            error = exc
            time.sleep(0.5)
    else:
        pytest.skip(f"no usable Tk display: {error}")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture(scope="module")
def shared_detector():
    return Inspector.from_config(load_config(ROOT / "config" / "config.yaml")).detector


@pytest.fixture
def dialogs(monkeypatch):
    """Record dialogs; answer yes; hand out queued PINs and form answers."""
    from fc_comparator.ui import main_window, widgets
    from fc_comparator.ui.pages import data, history, inspect, parts, settings, train

    shown = {"error": [], "info": [], "pins": [], "fields": [], "yes": True}
    for mod in (main_window, widgets, data, history, inspect, parts, settings, train):
        for name, fn in (("show_error", lambda _p, text, *a, **k: shown["error"].append(text)),
                         ("show_info", lambda _p, text, *a, **k: shown["info"].append(text)),
                         ("ask_yes_no", lambda *_a, **_k: shown["yes"]),
                         ("ask_fields", lambda *_a, **_k: shown["fields"].pop(0) if shown["fields"] else None)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn)

    class FakePin:
        @staticmethod
        def ask(_parent, _title, numeric_only=True):
            return shown["pins"].pop(0) if shown["pins"] else None

    for mod in (main_window, inspect, settings):
        monkeypatch.setattr(mod, "PinDialog", FakePin)
    return shown


@pytest.fixture
def app(tk_root, tmp_path, dialogs, shared_detector):
    from fc_comparator.ui.main_window import MainWindow

    shutil.copytree(ROOT / "samples" / "annotations", tmp_path / "annotations")
    cfg = load_config(ROOT / "config" / "config.yaml")
    cfg.annotations.dir = str(tmp_path / "annotations")
    cfg.camera.file_path = str(cfg.resolve(cfg.camera.file_path))
    cfg.detector.model_path = str(tmp_path / "models" / "none.pt")
    cfg.storage.database = str(tmp_path / "db.sqlite")
    cfg.storage.image_dir = str(tmp_path / "images")
    save_config(cfg, tmp_path / "config.yaml")
    top = ctk.CTkToplevel(tk_root)
    top.geometry("1500x950")
    station = Station(cfg, source=FileSource(BOARDS / "board_P001_ok.jpg"),
                      alert=AlertController(ConsoleAlert(), cfg.alert),
                      inspector=Inspector(cfg, shared_detector))
    station.reload = lambda: None  # keep the shared detector (rebuilding takes seconds)
    station.open()
    win = MainWindow(top, station)
    win.pack(fill="both", expand=True)
    top.update()
    yield win
    win.shutdown()
    station.close()
    top.destroy()


def settle(root, until=lambda: True, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        if until():
            break
        time.sleep(0.01)
    for _ in range(5):
        root.update()


def test_inspect_flow_lock_and_history(app, dialogs):
    page = app.inspect
    page.scan.insert(0, "OP:77")
    page.on_scan()
    assert page.operator.get() == "77"
    page.scan.insert(0, "P002")
    page.on_scan()
    assert page.part_code == "P002" and "R1 fork_right" in [w.cget("text").strip() for w in page.chips.winfo_children()]

    page.set_part("P001")
    page.inspect()
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "OK" and page.finding_texts == ["No findings."]

    app.station.source.load(BOARDS / "board_P001_mixed.jpg")
    app.pedal_pressed()  # foot pedal / F9
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "NG"
    assert any("C3" in t and "found fork_right, expected round" in t for t in page.finding_texts)
    assert app.station.lock.locked and page.ack_btn.winfo_manager() == "pack"
    assert app.nav_buttons["settings"].cget("state") == "disabled"

    page.scan.insert(0, "P002")  # cannot switch part while locked
    page.on_scan()
    assert page.part_code == "P001" and page.banner.text == "LOCKED"

    dialogs["pins"] += ["0000"]
    page.acknowledge()
    assert app.station.lock.locked and page.banner.text == "WRONG PIN"
    dialogs["pins"] += ["1234"]
    page.acknowledge()
    settle(app.root)
    assert not app.station.lock.locked and page.ack_btn.winfo_manager() == ""

    assert app.go("history")
    app.pages["history"].refresh()
    tree = app.pages["history"].table.tree
    assert len(tree.get_children()) == 2
    tree.selection_set(tree.get_children()[0])
    settle(app.root)
    assert "unexpected" not in app.pages["history"].detail.get("1.0", "end")
    assert "expected round, found fork_right" in app.pages["history"].detail.get("1.0", "end")


def test_mark_new_images_and_create_a_part(app, dialogs, tmp_path):
    dialogs["pins"] += ["5678"]
    assert app.go("data") and app.admin
    page = app.pages["data"]

    # upload a good board of a new part with another layout
    geo = Geometry(cable_x0=300, cable_dx=220, row_y0=280, row_dy=200)
    pattern = ["fork_right", "small", "round"]
    img, boxes = render_marked_board([pattern] * 5, seed=5, geometry=geo)
    photo = write_image(tmp_path / "upload" / "p004_good.jpg", img)
    page._import([photo])
    settle(app.root, lambda: dialogs["info"])
    assert "Added 1 image" in dialogs["info"][-1]
    image_id = next(r.id for r in page.store.records() if r.source_name == "p004_good.jpg")
    page.table.select(image_id)
    settle(app.root, lambda: page.image_id == image_id)
    assert page.canvas.image is not None and page.canvas.boxes == []

    # mark every clip through the canvas' own mouse handling
    canvas = page.canvas

    class Ev:
        def __init__(self, x, y):
            self.x, self.y = x, y

    for b in boxes:
        page._choose_class(b.label)
        (ax, ay), (bx, by) = canvas.to_canvas(b.x, b.y), canvas.to_canvas(b.x + b.w, b.y + b.h)
        canvas._press(Ev(ax, ay))
        canvas._motion(Ev(bx, by))
        canvas._release(Ev(bx, by))
    saved = AnnotationStore(tmp_path / "annotations").get(image_id).boxes  # auto-saved to disk
    assert len(saved) == 15 and [b.label for b in saved] == [b.label for b in boxes]
    assert abs(saved[0].x - boxes[0].x) < 3

    # undo / delete / keyboard class change
    canvas._select(0)
    canvas._digit(4)  # key 4 -> small
    assert canvas.boxes[0].label == "small"
    canvas.undo()
    assert canvas.boxes[0].label == "fork_right"
    canvas._select(0)
    canvas.delete_selected()
    assert len(canvas.boxes) == 14
    canvas.undo()
    assert len(canvas.boxes) == 15

    # make it the master of a new part
    dialogs["fields"].append(["P004", "Harness D"])
    page.make_master()
    assert dialogs["info"][-1].startswith("P004 saved")
    part = app.cfg.parts["P004"]
    assert (part.cables, part.rows, part.pattern, part.master_image) == (5, 3, pattern, image_id)
    assert load_config(tmp_path / "config.yaml").parts["P004"].layout is not None
    assert page.store.get(image_id).good


def test_password_gate_parts_and_settings(app, dialogs):
    assert not app.go("parts")  # dialog cancelled
    dialogs["pins"] += ["9999"]
    assert not app.go("parts") and "not correct" in dialogs["error"][-1]
    dialogs["pins"] += ["5678"]
    assert app.go("parts") and app.current == "parts"

    parts = app.pages["parts"]
    parts.search.insert(0, "harness c")
    parts.refresh_list()
    assert parts.table.tree.get_children() == ("P003",)
    parts.load("P003")
    assert len(parts.row_menus) == 5 and parts.preview._img is not None  # master layout preview
    parts.desc.delete(0, "end")
    parts.desc.insert(0, "Harness C rev 2")
    parts.save()
    assert app.cfg.parts["P003"].description == "Harness C rev 2" and app.cfg.parts["P003"].layout is not None

    assert app.go("settings")
    s = app.pages["settings"]
    s.v["detector.confidence_threshold"][0].set("0.7")
    s._add_class_row("")
    s.class_rows[-1][1].insert(0, "clamp")
    s.save()
    saved = load_config(app.cfg.path)
    assert saved.detector.confidence_threshold == pytest.approx(0.7)
    assert saved.taxonomy.classes[-1] == "clamp"
    assert "Clip types changed" in dialogs["info"][-1]

    # renaming a clip type updates markings and part patterns
    entry = next(e for name, e, _r in s.class_rows if name == "small")
    entry.delete(0, "end")
    entry.insert(0, "fir_tree")
    s.save()
    assert "fir_tree" in app.cfg.taxonomy.classes and "small" not in app.cfg.taxonomy.classes
    assert app.cfg.parts["P001"].pattern[2] == "fir_tree"
    assert "fir_tree" in app.station.annotations.stats()["per_class"]

    # a clip type still in use cannot be removed
    in_use = next(r for r in s.class_rows if r[0] == "round")
    s._remove_class_row(in_use)
    assert "still used" in dialogs["error"][-1]

    app.leave_admin()
    assert app.current == "inspect" and not app.admin
