"""UI tests: drive the real tkinter window (skipped where no display is available).

Message boxes, login and input dialogs are replaced by recorders so the tests never block.
"""

import json
import shutil
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

from fc_comparator.config import load_config, save_config
from fc_comparator.station import Station
from fc_comparator.station.alerts import AlertController, ConsoleAlert
from fc_comparator.station.security import verify_secret
from fc_comparator.vision.camera import FileSource, write_image
from fc_comparator.vision.detect import create_detector
from fc_comparator.vision.parts import ModelInfo, PartRepository
from fc_comparator.vision.synthetic import Geometry, render_marked_board

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BOARDS = FIXTURES / "boards"
pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def tk_root():
    from fc_comparator.ui import style

    error = None
    for _ in range(3):  # loading Tk's library files occasionally fails transiently on Windows
        try:
            r = tk.Tk()
            break
        except tk.TclError as exc:
            error = exc
            time.sleep(0.5)
    else:
        pytest.skip(f"no usable Tk display: {error}")
    style.apply(r)
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture(scope="module")
def shared_factory():
    """The sample parts' template detectors are built once (seconds each); new parts get their own."""
    cfg = load_config(FIXTURES / "config.yaml")
    repo = PartRepository(cfg.resolve(cfg.storage.parts_dir))
    cache = {code: create_detector(cfg, repo.get(code)) for code in repo.codes()}

    def factory(app_cfg, part):
        return cache[part.code] if part.code in cache else create_detector(app_cfg, part)

    return factory


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes; answer yes; hand out queued answers for input and login dialogs."""
    from fc_comparator.ui import main_window, widgets
    from fc_comparator.ui.pages import history, inspect, settings, setup, setup_images, setup_master, setup_train

    shown = {"error": [], "info": [], "answers": [], "logins": [], "yes": True}
    for mod in (main_window, widgets, history, inspect, settings, setup, setup_images, setup_master, setup_train):
        for name, fn in (("show_error", lambda _p, text, *a, **k: shown["error"].append(text)),
                         ("show_info", lambda _p, text, *a, **k: shown["info"].append(text)),
                         ("ask_yes_no", lambda *_a, **_k: shown["yes"]),
                         ("ask_string", lambda *_a, **_k: shown["answers"].pop(0) if shown["answers"] else None),
                         ("ask_login", lambda *_a, **_k: shown["logins"].pop(0) if shown["logins"] else None)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn)
    return shown


@pytest.fixture
def app(tk_root, tmp_path, dialogs, shared_factory):
    from fc_comparator.ui.main_window import MainWindow

    shutil.copytree(FIXTURES / "parts", tmp_path / "parts")
    cfg = load_config(FIXTURES / "config.yaml")
    cfg.storage.parts_dir = str(tmp_path / "parts")
    cfg.camera.file_path = str(cfg.resolve(cfg.camera.file_path))
    cfg.storage.database = str(tmp_path / "db.sqlite")
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.training.workdir = str(tmp_path / "runs")
    save_config(cfg, tmp_path / "config.yaml")
    top = tk.Toplevel(tk_root)
    top.geometry("1500x950")
    station = Station(cfg, source=FileSource(BOARDS / "board_P001_ok.jpg"),
                      alert=AlertController(ConsoleAlert(), cfg.alert), detector_factory=shared_factory)
    station.open()
    win = MainWindow(top, station)
    win.pack(fill="both", expand=True)
    top.update()
    settle(top, lambda: win.inspect._loading == "")
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


class Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


def draw(canvas, box):
    (ax, ay), (bx, by) = canvas.to_canvas(box.x, box.y), canvas.to_canvas(box.x + box.w, box.y + box.h)
    canvas._press(Ev(ax, ay))
    canvas._motion(Ev(bx, by))
    canvas._release(Ev(bx, by))


def kinds(app):
    return [e["kind"] for e in app.station.store.events()]


def login(app, dialogs, user="nice", password="nice1234"):
    dialogs["logins"].append((user, password))
    return app.go("setup")


# ---------------------------------------------------------------------------
def test_opens_on_inspect_without_login(app):
    assert app.current == "inspect" and app.session_user is None
    titles = [app.tabs.tab(t, "text").strip() for t in app.tabs.tabs()]
    assert titles == ["Inspect", "History", "Model Setup"]
    assert app.inspect.part_code == "P001" and app.inspect.ready and app.inspect.banner.text == "READY"


def test_model_setup_needs_the_login_and_ends(app, dialogs):
    assert not app.go("setup") and app.current == "inspect"  # login cancelled
    assert not login(app, dialogs, "nice", "wrong") and "Wrong user name" in dialogs["error"][-1]
    assert app.current == "inspect" and kinds(app) == ["setup_login_denied"]

    assert login(app, dialogs) and app.session_user == "nice" and "Model Setup: nice" in app.session_label.cget("text")
    assert app.go("history")  # leaving the tab ends the session
    assert app.session_user is None and kinds(app)[-1] == "setup_logout"
    assert not app.go("setup")  # ... so the login is asked again (cancelled here)

    assert login(app, dialogs)
    app._last_input -= app.cfg.auth.setup_timeout_min * 60 + 1  # 10 minutes without input
    app._check_timeout()
    assert app.session_user is None and app.current == "inspect"
    assert "no input for 10 min" in app.station.store.events()[-1]["detail"]


def test_inspect_flow_lock_and_history(app, dialogs):
    page = app.inspect
    page.part.set("OP:77")
    page.on_scan()
    assert page.operator.get() == "77" and page.part_code == "P001"
    page.part.set("P002")
    page.on_scan()
    settle(app.root, lambda: page._loading == "")
    assert page.part_code == "P002" and "Row 1: fork_right" in page.part_info.cget("text")

    page.set_part("P001")
    settle(app.root, lambda: page._loading == "")
    page.inspect()
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "OK" and len(page.finding_texts) == 16

    app.station.source.load(BOARDS / "board_P001_mixed.jpg")
    app.pedal_pressed()  # foot pedal / F9
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "NG"
    assert any(t.startswith("C3 R1 round fork_right") for t in page.finding_texts)
    assert app.station.lock.locked and page.ack_btn.winfo_manager() == "grid"
    assert str(app.tabs.tab(app.pages["setup"], "state")) == "disabled"
    assert not app.go("setup") and "locked" in dialogs["error"][-1]

    page.part.set("P002")  # cannot switch part while locked
    page.on_scan()
    assert page.part_code == "P001" and page.banner.text == "LOCKED"

    dialogs["answers"] += ["0000"]
    page.acknowledge()
    assert app.station.lock.locked and page.banner.text == "WRONG PIN"
    dialogs["answers"] += ["1234"]
    page.acknowledge()
    settle(app.root)
    assert not app.station.lock.locked and page.ack_btn.winfo_manager() == ""
    assert str(app.tabs.tab(app.pages["setup"], "state")) == "normal"

    assert app.go("history")
    history = app.pages["history"]
    tree = history.table.tree
    assert len(tree.get_children()) == 2
    tree.selection_set(tree.get_children()[0])
    settle(app.root)
    text = history.detail.get("1.0", "end")
    assert "expected round, found fork_right" in text and "unexpected" not in text


def test_part_that_is_not_set_up_cannot_be_inspected(app):
    app.station.parts.create("PX", "nothing yet")
    page = app.inspect
    page.on_parts_changed()
    page.set_part("PX")
    settle(app.root, lambda: page._loading == "")
    assert page.banner.text == "NOT SET UP" and "no master" in page.banner.detail.cget("text")
    page.inspect()
    settle(app.root)
    assert page.banner.text == "NOT SET UP" and app.station.store.inspections() == []


def test_new_part_images_labels_and_master(app, dialogs, tmp_path):
    """Model Setup: create P004, add and label an image, rename a label, set the master, inspect it."""
    assert login(app, dialogs)
    setup = app.pages["setup"]
    geo = Geometry(cable_x0=300, cable_dx=220, row_y0=280, row_dy=200)
    columns = [["fork_right", "small", "round"]] * 4 + [["fork_right", "round", "round"]]  # cable 5 differs

    dialogs["answers"] += ["Harness D"]
    part = setup.open_code("P004")
    assert part is not None and part.description == "Harness D" and setup.part is part
    assert "P004" in setup.code.cget("values") and "NOT SET UP" in setup.info.cget("text")

    # b. add an image
    img, boxes = render_marked_board(columns, seed=9, geometry=geo)
    photo = write_image(tmp_path / "new" / "board_new.jpg", img)
    images = setup.images
    assert images.add_paths([photo]) == 1 and "Added 1 image(s) to P004" in dialogs["info"][-1]
    image_id = part.store.records()[0].id
    settle(app.root, lambda: images.current == image_id)
    canvas = images.canvas
    assert canvas.image is not None and canvas.boxes == []

    # c. label: the first box of a new label asks for its name
    for b in boxes:
        if b.label in part.labels:
            images.label.set(b.label)
            canvas.set_class(b.label)
        else:
            dialogs["answers"].append(b.label)
            canvas.current_class = ""
        draw(canvas, b)
    saved = part.store.get(image_id).boxes  # saved automatically
    assert [b.label for b in saved] == [b.label for b in boxes] and abs(saved[0].x - boxes[0].x) < 3
    assert part.labels == ["fork_right", "small", "round"]
    assert images.table.tree.item(image_id, "values")[1] == "15"
    canvas._select(0)
    canvas._digit(2)  # key 2 -> second label
    assert canvas.boxes[0].label == "small"
    canvas.undo()
    assert canvas.boxes[0].label == "fork_right"

    manager = images.manage_labels()  # rename on every image
    manager.table.select("small")
    dialogs["answers"].append("tiny")
    manager.rename()
    assert part.labels == ["fork_right", "tiny", "round"] and kinds(app)[-1] == "labels_changed"
    manager.destroy()

    # f. master: one object was not found, the box is drawn by hand
    mimg, mboxes = render_marked_board(columns, seed=5, geometry=geo)
    mboxes = [type(b)("tiny" if b.label == "small" else b.label, b.x, b.y, b.w, b.h) for b in mboxes]
    editor = setup.master_panel.open_editor(mimg, mboxes[:-1])
    settle(app.root)
    assert "cable 5 row 3 has no" in editor.status.cget("text")
    assert str(editor.save_btn.cget("state")) == "disabled"
    editor.label.set("round")
    editor.canvas.set_class("round")
    draw(editor.canvas, mboxes[-1])
    settle(app.root)
    assert "5 cables x 3 rows" in editor.status.cget("text")
    master = editor.save()
    assert master is not None and "Saved the master of P004" in dialogs["info"][-1]
    again = PartRepository(tmp_path / "parts").get("P004")
    assert again.master.pattern == [["fork_right"] * 5, ["tiny"] * 4 + ["round"], ["round"] * 5]
    assert again.store.get(again.master.master_image).good and len(again.store) == 2  # photo kept with its boxes
    assert "READY for inspection" in setup.info.cget("text")
    assert [(e["kind"], e["part_number"]) for e in app.station.store.events() if e["part_number"]] == [
        ("part_created", "P004"), ("labels_changed", "P004"), ("master_saved", "P004")]

    # inspect a board of the new part with an object missing
    test_cols = [list(c) for c in columns]
    test_cols[2][1] = "missing"
    app.station.source.set_image(render_marked_board(test_cols, seed=6, geometry=geo, shift=(15, -10))[0])
    assert app.go("inspect") and app.session_user is None
    app.inspect.set_part("P004")
    settle(app.root, lambda: app.inspect._loading == "", timeout=30)
    assert app.inspect.ready
    app.inspect.inspect()
    settle(app.root, lambda: not app.inspect.busy)
    assert app.inspect.banner.text == "NG"
    assert any(t.startswith("C3 R2 tiny missing") for t in app.inspect.finding_texts)


def test_train_use_and_roll_back_models(app, dialogs, monkeypatch):
    assert login(app, dialogs)
    setup = app.pages["setup"]
    part = setup.open_code("P001")
    train = setup.train
    setup.sections.select(train)
    settle(app.root)
    assert "Ready: 7 labelled images" in train.readiness.cget("text")

    from fc_comparator.vision import training

    def fake_train(accuracy, pr=None):
        def run(cfg, part, progress, stop, epochs):
            progress("epoch 3/5  loss 1.000")
            version = part.next_version()
            part.models_dir.mkdir(parents=True, exist_ok=True)
            (part.models_dir / f"{version}.pt").write_bytes(b"w")
            (part.models_dir / f"{version}.json").write_text(json.dumps(
                {"accuracy": accuracy, "trained": "2026-10-01T12:00:00", "precision_recall": pr or {}}))
            return SimpleNamespace(version=version, model=part.model(version), epochs_run=5, seconds=60.0,
                                   evaluation=SimpleNamespace(accuracy=accuracy))
        return run

    monkeypatch.setattr(training, "train_detector", fake_train(0.97))
    train.epochs.set(5)
    train.train()
    settle(app.root, lambda: not train.training)
    assert "epoch 3/5" in train.log.get("1.0", "end") and part.active_model == ""  # not used yet
    assert str(train.use_btn.cget("state")) == "normal" and "reached the 95%" in train.result_label.cget("text")
    assert train.use_new() and part.active_model == "v001"
    assert "v001 (97.0%" in setup.info.cget("text")

    monkeypatch.setattr(training, "train_detector", fake_train(0.6, {"round": [1.0, 1.0], "small": [0.8, 0.4]}))
    train.train()
    settle(app.root, lambda: not train.training)
    text = train.result_label.cget("text")
    assert "below the 95%" in text and "previous model stays in use" in text and "small: finds 40%" in text
    assert str(train.use_btn.cget("state")) == "disabled" and part.active_model == "v001"
    train.versions.select("v002")
    assert not train.use_selected() and "cannot be used" in dialogs["error"][-1]

    monkeypatch.setattr(training, "train_detector", fake_train(0.99))
    train.train()
    settle(app.root, lambda: not train.training)
    assert train.use_new() and part.active_model == "v003"
    train.versions.select("v001")  # roll back
    assert train.use_selected() and part.active_model == "v001"
    states = {iid: train.versions.tree.item(iid, "values")[4] for iid in train.versions.tree.get_children()}
    assert states == {"v003": "usable", "v002": "below 95%", "v001": "in use"}
    model_events = [(e["kind"], e["detail"]) for e in app.station.store.events() if e["kind"].startswith("model")]
    assert model_events == [("model_trained", "v001: 97.0%"), ("model_activated", "- -> v001"),
                            ("model_trained", "v002: 60.0%"), ("model_trained", "v003: 99.0%"),
                            ("model_activated", "v001 -> v003"), ("model_rollback", "v003 -> v001")]
    assert isinstance(part.active(), ModelInfo)


def test_settings_save_and_security(app, dialogs, tmp_path):
    assert login(app, dialogs)
    s = app.pages["setup"].settings
    s.vars["detector.confidence_threshold"][0].set("0.7")
    s.vars["station.name"][0].set("Line 2")
    s.vars["auth.setup_timeout_min"][0].set("5")
    assert s.save()
    saved = load_config(tmp_path / "config.yaml")
    assert saved.detector.confidence_threshold == pytest.approx(0.7) and saved.station.name == "Line 2"
    assert saved.auth.setup_timeout_min == 5

    s.vars["detector.confidence_threshold"][0].set("abc")
    assert not s.save() and "not a valid value" in dialogs["error"][-1]
    s.vars["detector.confidence_threshold"][0].set("0.1")  # below min_score
    assert not s.save() and "min_score" in dialogs["error"][-1]
    assert app.cfg.detector.confidence_threshold == pytest.approx(0.7)  # rejected values are not kept

    dialogs["answers"] += ["operator", "abcd1234", "abcd1234"]
    s.change_login()
    auth = load_config(tmp_path / "config.yaml").auth
    assert auth.user == "operator" and verify_secret("abcd1234", auth.password)
    dialogs["answers"] += ["operator", "abcd1234", "other"]
    s.change_login()
    assert "did not match" in dialogs["error"][-1]
    dialogs["answers"] += ["9876", "9876"]
    s.change_pin()
    assert verify_secret("9876", app.station.lock.pin_hash)
