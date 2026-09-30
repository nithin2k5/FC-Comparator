"""End-to-end: sample images -> Station (detect, place, compare, alert, lock, log).

Acceptance criteria covered:
* every wrong position is flagged and a correct board passes (manifest boards)
* missing clips and low-confidence detections are never OK
* boards shifted by ~20 px are inspected correctly
* the alert fires within 1 s of NG and every result is logged
* a new part number is set up from a marked good image (numerous parts workflow)
"""

from pathlib import Path

import pytest
import yaml

from fc_comparator.alert import AlertController, ConsoleAlert, Outputs
from fc_comparator.annotations import AnnotationStore
from fc_comparator.capture import FileSource
from fc_comparator.capture.sources import read_image, write_image
from fc_comparator.cli import main as cli_main
from fc_comparator.config import load_config, save_config
from fc_comparator.models import Box, PartNumber
from fc_comparator.pipeline import Inspector, Station, StationLocked
from fc_comparator.synthetic import Geometry, render_marked_board

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CONFIG = ROOT / "config" / "config.yaml"
MANIFEST = yaml.safe_load((SAMPLES / "manifest.yaml").read_text(encoding="utf-8"))
P001 = ["round", "fork_left", "small", "round"]


def tmp_config(tmp_path: Path):
    """The sample config, saved into tmp with absolute paths (safe to modify)."""
    cfg = load_config(CONFIG)
    for obj, attr in ((cfg.annotations, "dir"), (cfg.camera, "file_path"), (cfg.detector, "model_path"),
                      (cfg.training, "workdir")):
        setattr(obj, attr, str(cfg.resolve(getattr(obj, attr))))
    cfg.storage.database = str(tmp_path / "inspections.db")
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.storage.save_ok_every_n = 2
    save_config(cfg, tmp_path / "config.yaml")
    return cfg


@pytest.fixture(scope="module")
def inspector():
    return Inspector.from_config(load_config(CONFIG))


@pytest.fixture
def station(tmp_path, inspector):
    cfg = tmp_config(tmp_path)
    backend = ConsoleAlert()
    st = Station(cfg, source=FileSource(), inspector=Inspector(cfg, inspector.detector),
                 alert=AlertController(backend, cfg.alert))
    st.backend = backend
    st.open()
    yield st
    st.close()


@pytest.mark.parametrize("case", MANIFEST, ids=[Path(m["image"]).stem for m in MANIFEST])
def test_sample_boards(station, case):
    station.lock.enabled = False
    res = station.inspect(case["part"], "OP1", image=read_image(SAMPLES / case["image"]))
    rep = res.report
    if case["error"]:
        assert rep.error and not rep.ok, case["note"]
    else:
        assert not rep.error, rep.error
        assert sorted([p.cable, p.row] for p in rep.mismatches) == sorted(case["expected_ng"]), (case["note"], rep.mismatches)
        assert len(rep.extras) == case["extras"], case["note"]
        assert rep.ok == (not case["expected_ng"] and not case["extras"])
        part = station.cfg.parts[case["part"]]
        assert len(rep.positions) == part.cables * part.rows
    for p in rep.mismatches:  # every finding carries the full detail
        assert p.expected and p.found and 0.0 <= p.confidence <= 1.0 and p.reason
    assert rep.duration_ms < 1000 and res.alert_latency_ms < 1000
    assert station.backend.history[-1] == (Outputs(green=True) if rep.ok else Outputs(red=True, buzzer=True))
    row = station.store.inspections(limit=1)[0]
    assert row["id"] == res.inspection_id and row["result"] == rep.verdict.value
    assert len(station.store.positions(res.inspection_id)) == len(rep.positions)


@pytest.mark.parametrize("shift", [(20, 0), (0, 20), (-20, -20), (20, -20)])
def test_shifted_boards(station, shift):
    station.lock.enabled = False
    mixed = [P001, P001, ["fork_right", "fork_right", "small", "fork_right"], P001]
    img = render_marked_board(mixed, shift=shift, seed=300)[0]
    rep = station.inspect("P001", image=img).report
    assert rep.placement.ok and rep.placement.shift_px == pytest.approx((shift[0] ** 2 + shift[1] ** 2) ** 0.5, abs=8)
    assert sorted(p.key for p in rep.mismatches) == [(3, 1), (3, 2), (3, 4)]


def test_missing_and_uncertain_are_never_ok(station):
    station.lock.enabled = False
    cols = [list(P001) for _ in range(4)]
    cols[0][3] = "missing"
    rep = station.inspect("P001", image=render_marked_board(cols, seed=61)[0]).report
    by = {p.key: p for p in rep.positions}
    assert by[(1, 4)].found == "missing" and not by[(1, 4)].ok

    # every clip found, right class, but below the confidence threshold -> uncertain -> NG
    img, boxes = render_marked_board([P001] * 4, seed=62)

    class Unsure:
        name, labels, ready = "stub", ("round",), True

        def detect(self, _img):
            return [Box(b.label, b.x, b.y, b.w, b.h, 0.45) for b in boxes]

    station.inspector = Inspector(station.cfg, Unsure())
    rep = station.inspect("P001", image=img).report
    assert not rep.ok and all(p.found == "uncertain" and not p.ok for p in rep.positions)
    assert rep.positions[0].reason == "low confidence (looks like round 45%)"


def test_part_without_master_layout_uses_grid(station):
    station.lock.enabled = False
    station.cfg.parts["PX"] = PartNumber("PX", list(P001), cables=4)  # typed pattern, no master image
    rep = station.inspect("PX", image=read_image(SAMPLES / "boards" / "board_P001_missing.jpg")).report
    assert not rep.error and sorted(p.key for p in rep.mismatches) == [(2, 3)]
    assert "grid" in rep.placement.message
    rep = station.inspect("PX", image=read_image(SAMPLES / "boards" / "board_P003_ok.jpg")).report
    assert rep.error and "cable column" in rep.error  # 3 cables found, 4 expected


def test_unknown_part_is_ng(station):
    station.lock.enabled = False
    rep = station.inspect("P999", image=read_image(SAMPLES / "boards" / "board_P001_ok.jpg")).report
    assert not rep.ok and "unknown part" in rep.error


def test_ng_locks_station_until_pass_or_supervisor(station):
    ok_img = read_image(SAMPLES / "boards" / "board_P001_ok.jpg")
    ng_img = read_image(SAMPLES / "boards" / "board_P001_mixed.jpg")
    ng = station.inspect("P001", "OP1", image=ng_img)
    assert station.lock.locked and station.lock.state.inspection_id == ng.inspection_id
    with pytest.raises(StationLocked):
        station.inspect("P002", image=ok_img)
    station.inspect("P001", image=ng_img)
    assert station.lock.locked
    station.inspect("P001", "OP1", image=ok_img)
    assert not station.lock.locked
    station.inspect("P001", image=ng_img)
    assert not station.acknowledge("9999", "SUP1") and station.acknowledge("1234", "SUP1")
    assert [e["kind"] for e in station.store.events()] == ["unlock_pass", "ack_failed", "ack"]


def test_logging_and_evidence_images(station):
    station.lock.enabled = False
    ok_img = read_image(SAMPLES / "boards" / "board_P001_ok.jpg")
    ng_img = read_image(SAMPLES / "boards" / "board_P001_extra_clip.jpg")
    results = [station.inspect("P001", image=im) for im in (ok_img, ng_img, ok_img, ok_img, ng_img)]
    assert len(station.store.inspections()) == 5
    for r in results:
        if not r.report.ok:
            assert Path(r.image_path).is_file()
            assert station.store.get_inspection(r.inspection_id)["extras_count"] == 1
            assert station.store.extras(r.inspection_id)[0]["label"] == "small"
    assert len([r for r in results if r.report.ok and r.image_path]) == 2  # every 2nd OK image
    assert station.store.daily_report().ng == 2


def test_new_part_from_marked_image_via_cli(tmp_path, capsys):
    """Numerous parts: upload a good board of a new layout, mark it, make it a master, inspect."""
    cfg = tmp_config(tmp_path)
    cfg_path = str(tmp_path / "config.yaml")
    geo = Geometry(cable_x0=300, cable_dx=220, row_y0=280, row_dy=200)
    pattern = ["fork_right", "small", "round"]
    img, boxes = render_marked_board([pattern] * 5, seed=5, geometry=geo)
    photo = write_image(tmp_path / "upload" / "p004_good.jpg", img)

    assert cli_main(["-c", cfg_path, "add-images", str(photo.parent), "--part", "P004", "--good"]) == 0
    store = AnnotationStore(cfg.resolve(cfg.annotations.dir))
    image_id = next(r.id for r in store.records() if r.source_name == "p004_good.jpg")
    store.set_boxes(image_id, boxes)  # what the operator does in "Setup"
    try:
        assert cli_main(["-c", cfg_path, "make-master", "--image-id", image_id, "--part", "P004"]) == 0
        saved = load_config(cfg_path).parts["P004"]
        assert (saved.cables, saved.rows, saved.pattern) == (5, 3, pattern)

        test_board = write_image(tmp_path / "p004_test.jpg",
                                 render_marked_board([pattern, pattern, ["fork_right", "missing", "round"], pattern, pattern],
                                                     seed=6, geometry=geo, shift=(15, -10))[0])
        rc = cli_main(["-c", cfg_path, "inspect", "--image", str(test_board), "--part", "P004", "--no-store"])
        out = capsys.readouterr().out
        assert rc == 1 and "NG cable 3 row 2: expected small, found missing" in out
    finally:
        store.delete(image_id)  # keep the shared sample store unchanged


def test_cli_inspect_exit_codes(tmp_path, capsys):
    ok = cli_main(["-c", str(CONFIG), "inspect", "--image", str(SAMPLES / "boards" / "board_P001_ok.jpg"),
                   "--part", "P001", "--no-store"])
    out = tmp_path / "annotated.jpg"
    ng = cli_main(["-c", str(CONFIG), "inspect", "--image", str(SAMPLES / "boards" / "board_P001_extra_clip.jpg"),
                   "--part", "P001", "--no-store", "--out", str(out)])
    text = capsys.readouterr().out
    assert (ok, ng) == (0, 1) and "NG unexpected small" in text and out.is_file()
