"""UI tests: drive the real tkinter window (skipped where no display is available).

Message boxes and input dialogs are replaced by recorders so the tests never block.
"""

import shutil
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import pytest

from fc_comparator.config import AppConfig, load_config, save_config
from fc_comparator.station import Inspector, Station
from fc_comparator.station.alerts import AlertController, ConsoleAlert
from fc_comparator.station.security import verify_secret
from fc_comparator.vision.camera import FileSource, write_image
from fc_comparator.vision.dataset import AnnotationStore
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
def shared_detector():
    return Inspector.from_config(load_config(FIXTURES / "config.yaml")).detector


@pytest.fixture
def dialogs(monkeypatch):
    """Record message boxes; answer yes; hand out queued answers for input dialogs."""
    from fc_comparator.ui import main_window, widgets
    from fc_comparator.ui.pages import history, inspect, parts, settings, training

    shown = {"error": [], "info": [], "answers": [], "yes": True}
    for mod in (main_window, widgets, history, inspect, parts, settings, training):
        for name, fn in (("show_error", lambda _p, text, *a, **k: shown["error"].append(text)),
                         ("show_info", lambda _p, text, *a, **k: shown["info"].append(text)),
                         ("ask_yes_no", lambda *_a, **_k: shown["yes"]),
                         ("ask_string", lambda *_a, **_k: shown["answers"].pop(0) if shown["answers"] else None)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, fn)
    return shown


@pytest.fixture
def app(tk_root, tmp_path, dialogs, shared_detector):
    from fc_comparator.ui.main_window import MainWindow

    shutil.copytree(FIXTURES / "annotations", tmp_path / "annotations")
    cfg = load_config(FIXTURES / "config.yaml")
    cfg.dataset.dir = str(tmp_path / "annotations")
    cfg.camera.file_path = str(cfg.resolve(cfg.camera.file_path))
    cfg.detector.model_path = str(tmp_path / "models" / "none.pt")
    cfg.storage.database = str(tmp_path / "db.sqlite")
    cfg.storage.image_dir = str(tmp_path / "images")
    save_config(cfg, tmp_path / "config.yaml")
    top = tk.Toplevel(tk_root)
    top.geometry("1500x950")
    station = Station(cfg, source=FileSource(BOARDS / "board_P001_ok.jpg"),
                      alert=AlertController(ConsoleAlert(), cfg.alert),
                      inspector=Inspector(cfg, shared_detector))
    station.reload = lambda: None  # keep the shared detector (rebuilding takes seconds)
    station.open()
    logged_out = []
    win = MainWindow(top, station, user="nice", on_logout=lambda: logged_out.append(True))
    win.pack(fill="both", expand=True)
    win.logged_out = logged_out
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


class Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


def draw(canvas, box):
    (ax, ay), (bx, by) = canvas.to_canvas(box.x, box.y), canvas.to_canvas(box.x + box.w, box.y + box.h)
    canvas._press(Ev(ax, ay))
    canvas._motion(Ev(bx, by))
    canvas._release(Ev(bx, by))


def test_login_page(tk_root):
    from fc_comparator.ui.login import LoginPage

    events, logins = [], []
    page = LoginPage(tk_root, load_config(ROOT / "config" / "config.yaml"), logins.append,
                     lambda kind, user, _detail: events.append((kind, user)))
    for user, password in (("nice", "wrong"), ("other", "nice1234"), ("", "")):
        page.reset()
        page.user.insert(0, user)
        page.password.insert(0, password)
        assert not page.submit() and "Wrong" in page.error.cget("text") and page.password.get() == ""
    page.reset()
    page.user.insert(0, "nice")
    page.password.insert(0, "nice1234")
    assert page.submit() and logins == ["nice"]
    assert events == [("login_denied", "nice"), ("login_denied", "other"), ("login_denied", ""), ("login", "nice")]
    assert LoginPage(tk_root, AppConfig(), logins.append).check("nice", "nice1234")  # built-in default
    page.destroy()


def test_logout(app):
    app.logout()
    assert app.logged_out and app.station.store.events()[-1]["kind"] == "logout"


def test_inspect_flow_lock_and_history(app, dialogs):
    page = app.inspect
    page.part.set("OP:77")
    page.on_scan()
    assert page.operator.get() == "77"
    page.part.set("P002")
    page.on_scan()
    assert page.part_code == "P002" and "Row 1: fork_right" in page.part_info.cget("text")

    page.set_part("P001")
    page.inspect()
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "OK" and len(page.finding_texts) == 16

    app.station.source.load(BOARDS / "board_P001_mixed.jpg")
    app.pedal_pressed()  # foot pedal / F9
    settle(app.root, lambda: not page.busy)
    assert page.banner.text == "NG"
    assert any(t.startswith("C3 R1 round fork_right") for t in page.finding_texts)
    assert app.station.lock.locked and page.ack_btn.winfo_manager() == "grid"
    assert str(app.tabs.tab(app.pages["settings"], "state")) == "disabled"
    assert not app.go("parts") and "locked" in dialogs["error"][-1]

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
    assert str(app.tabs.tab(app.pages["settings"], "state")) == "normal"

    assert app.go("history")
    history = app.pages["history"]
    tree = history.table.tree
    assert len(tree.get_children()) == 2
    tree.selection_set(tree.get_children()[0])
    settle(app.root)
    text = history.detail.get("1.0", "end")
    assert "expected round, found fork_right" in text and "unexpected" not in text


def test_new_part_from_a_good_board(app, dialogs, tmp_path):
    """Parts tab: the clips of a good board of a new layout become part P004's master."""
    assert app.go("parts")
    page = app.pages["parts"]
    geo = Geometry(cable_x0=300, cable_dx=220, row_y0=280, row_dy=200)
    columns = [["fork_right", "small", "round"]] * 4 + [["fork_right", "round", "round"]]  # cable 5 differs
    img, boxes = render_marked_board(columns, seed=5, geometry=geo)

    editor = page.open_editor(img, boxes[:-1], None)  # one clip not found yet
    settle(app.root)
    assert "cable 5 row 3 has no clip" in editor.status.cget("text")
    assert str(editor.save_btn.cget("state")) == "disabled"
    editor.clip.set("round")
    editor.canvas.set_class("round")
    draw(editor.canvas, boxes[-1])  # the operator adds the missing box
    settle(app.root)
    assert "5 cables x 3 rows" in editor.status.cget("text")
    editor.code.insert(0, "P004")
    editor.desc.insert(0, "Harness D")
    part = editor.save()
    assert part is not None and "Saved P004" in dialogs["info"][-1]

    saved = load_config(tmp_path / "config.yaml").parts["P004"]
    assert saved.pattern == [["fork_right"] * 5, ["small"] * 4 + ["round"], ["round"] * 5]
    assert saved.description == "Harness D" and saved.layout is not None
    rec = AnnotationStore(tmp_path / "annotations").get(saved.master_image)  # photo kept in the data set
    assert rec.good and rec.part == "P004" and len(rec.boxes) == 15
    assert page.table.selected() == "P004"
    assert app.inspect.part.cget("values")[-1] == "P004"

    # editing keeps the same master photo (no duplicate in the data set)
    n_images = len(page.station.dataset)
    page.edit_selected()
    editor = next(w for w in page.winfo_children() if type(w).__name__ == "PartEditor")
    settle(app.root)
    assert editor.code.get() == "P004" and len(editor.canvas.boxes) == 15
    assert editor.save() is not None
    assert len(page.station.dataset) == n_images
    assert app.cfg.parts["P004"].master_image == saved.master_image

    # inspect a board of the new part with a clip missing
    test_cols = [list(c) for c in columns]
    test_cols[2][1] = "missing"
    app.inspect.set_part("P004")
    app.station.source.set_image(render_marked_board(test_cols, seed=6, geometry=geo, shift=(15, -10))[0])
    app.go("inspect")
    app.inspect.inspect()
    settle(app.root, lambda: not app.inspect.busy)
    assert app.inspect.banner.text == "NG"
    assert any(t.startswith("C3 R2 small missing") for t in app.inspect.finding_texts)


def test_training_tab_marking_and_training(app, dialogs, tmp_path, monkeypatch):
    assert app.go("training")
    page = app.pages["training"]
    img, boxes = render_marked_board([["fork_right", "small", "round"]] * 3, seed=9,
                                     geometry=Geometry(cable_x0=300, cable_dx=250, row_y0=280, row_dy=220))
    photo = write_image(tmp_path / "new" / "board_new.jpg", img)
    page._add([photo])
    assert "Added 1 image" in dialogs["info"][-1]
    image_id = next(r.id for r in page.store.records() if r.source_name == "board_new.jpg")
    page.table.select(image_id)
    settle(app.root, lambda: page.current == image_id)
    canvas = page.canvas
    assert canvas.image is not None and canvas.boxes == []

    for b in boxes:
        page.clip.set(b.label)
        canvas.set_class(b.label)
        draw(canvas, b)
    saved = AnnotationStore(tmp_path / "annotations").get(image_id).boxes  # saved automatically
    assert [b.label for b in saved] == [b.label for b in boxes] and abs(saved[0].x - boxes[0].x) < 3
    assert page.table.tree.item(image_id, "values")[1] == "9"

    canvas._select(0)
    canvas._digit(2)  # key 2 -> second clip type
    assert canvas.boxes[0].label == app.cfg.taxonomy.classes[1]
    canvas.undo()
    assert canvas.boxes[0].label == "fork_right"
    canvas._select(0)
    canvas.delete_selected()
    assert len(canvas.boxes) == 8
    canvas.undo()
    assert len(canvas.boxes) == 9

    # training: progress lines update the bar; the result is reported
    from fc_comparator.vision import training

    def fake_train(cfg, store, progress, stop, epochs):
        progress("epoch 3/5  loss 1.000")
        return SimpleNamespace(evaluation=SimpleNamespace(accuracy=0.97), epochs_run=5, seconds=60.0)

    monkeypatch.setattr(training, "train_detector", fake_train)
    page.epochs.set(5)
    page.train()
    settle(app.root, lambda: not page.training)
    assert "97.0% and is now in use" in dialogs["info"][-1]
    assert "epoch 3/5" in page.log.get("1.0", "end")

    monkeypatch.setattr(training, "train_detector", lambda *a, **k: SimpleNamespace(
        evaluation=SimpleNamespace(accuracy=0.5), epochs_run=5, seconds=60.0))
    page.train()
    settle(app.root, lambda: not page.training)
    assert "keeps the previous detector" in dialogs["info"][-1]

    # an image that is a part's master cannot be deleted
    page.table.select(app.cfg.parts["P001"].master_image)
    settle(app.root)
    page.delete_image()
    assert "master photo of P001" in dialogs["error"][-1]


def test_settings_save_and_security(app, dialogs, tmp_path):
    assert app.go("settings")
    s = app.pages["settings"]
    s.vars["detector.confidence_threshold"][0].set("0.7")
    s.vars["station.name"][0].set("Line 2")
    assert s.save()
    saved = load_config(tmp_path / "config.yaml")
    assert saved.detector.confidence_threshold == pytest.approx(0.7) and saved.station.name == "Line 2"

    s.vars["detector.confidence_threshold"][0].set("abc")
    assert not s.save() and "not a valid value" in dialogs["error"][-1]
    s.vars["detector.confidence_threshold"][0].set("0.1")  # below min_score
    assert not s.save() and "min_score" in dialogs["error"][-1]
    assert app.cfg.detector.confidence_threshold == pytest.approx(0.7)  # rejected values are not kept

    dialogs["answers"] += ["operator", "abcd1234", "abcd1234"]
    s.change_login()
    sec = load_config(tmp_path / "config.yaml").security
    assert sec.login_user == "operator" and verify_secret("abcd1234", sec.login_password)
    dialogs["answers"] += ["operator", "abcd1234", "other"]
    s.change_login()
    assert "did not match" in dialogs["error"][-1]
    dialogs["answers"] += ["9876", "9876"]
    s.change_pin()
    assert verify_secret("9876", app.station.lock.pin_hash)
