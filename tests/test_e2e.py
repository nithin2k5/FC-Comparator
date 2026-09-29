"""End-to-end: sample images -> Station (align, classify, compare, alert, lock, log).

Acceptance criteria covered:
* every mismatched position is flagged and a correct board passes (manifest boards)
* missing clips and low-confidence detections are never OK
* boards shifted by ~20 px are inspected correctly (alignment)
* the alert fires within 1 s of NG and every result is logged
"""

from pathlib import Path

import cv2
import pytest
import yaml

from fc_comparator.alert import AlertController, ConsoleAlert, Outputs
from fc_comparator.capture import FileSource
from fc_comparator.capture.sources import read_image
from fc_comparator.cli import main as cli_main
from fc_comparator.config import load_config
from fc_comparator.models import PartNumber
from fc_comparator.pipeline import Inspector, Station, StationLocked
from fc_comparator.synthetic import render_board

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CONFIG = ROOT / "config" / "config.yaml"
MANIFEST = yaml.safe_load((SAMPLES / "manifest.yaml").read_text(encoding="utf-8"))
P001 = ["round", "fork_left", "small", "round"]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(CONFIG)
    cfg.storage.database = str(tmp_path / "inspections.db")
    cfg.storage.image_dir = str(tmp_path / "images")
    cfg.storage.save_ok_every_n = 2
    return cfg


@pytest.fixture(scope="module")
def inspector():
    return Inspector.from_config(load_config(CONFIG))


@pytest.fixture
def station(cfg, inspector):
    backend = ConsoleAlert()
    shared = Inspector(cfg, inspector.classifier, inspector.aligner)  # reuse loaded templates, own config
    st = Station(cfg, source=FileSource(), inspector=shared, alert=AlertController(backend, cfg.alert))
    st.backend = backend
    st.open()
    yield st
    st.close()


@pytest.mark.parametrize("case", MANIFEST, ids=[Path(m["image"]).stem for m in MANIFEST])
def test_sample_boards_report_exact_ng_positions(station, case):
    img = read_image(SAMPLES / case["image"])
    station.lock.enabled = False
    res = station.inspect(case["part"], "OP1", image=img)
    rep = res.report
    got = sorted([p.cable, p.row] for p in rep.mismatches)
    assert got == sorted(case["expected_ng"]), (case["note"], rep.mismatches)
    assert rep.ok == (not case["expected_ng"])
    assert len(rep.positions) == 16
    for p in rep.mismatches:  # every mismatch carries the full detail
        assert p.expected and p.found and 0.0 <= p.confidence <= 1.0 and p.reason
    # Inspection budget (well under 1 s on a PC; the 1 s target is for a Pi 5)
    assert rep.duration_ms < 1000
    assert res.alert_latency_ms < 1000
    assert station.backend.history[-1] == (Outputs(green=True) if rep.ok else Outputs(red=True, buzzer=True))
    row = station.store.inspections(limit=1)[0]
    assert row["id"] == res.inspection_id and row["result"] == rep.verdict.value
    assert len(station.store.positions(res.inspection_id)) == 16


def test_missing_and_low_confidence_never_ok(station):
    station.lock.enabled = False
    img = render_board([P001, P001, [*P001[:2], "missing", P001[3]], P001], seed=77)
    cv2.circle(img, (520, 640), 55, (200, 200, 90), -1)  # foreign object on C2R3
    rep = station.inspect("P001", image=img).report
    by = {p.key: p for p in rep.positions}
    assert not by[(3, 3)].ok and by[(3, 3)].found == "missing"
    assert not by[(2, 3)].ok and by[(2, 3)].found in ("uncertain", "missing")
    assert rep.verdict.value == "NG"


@pytest.mark.parametrize("shift", [(20, 0), (0, 20), (-20, -20), (20, -20)])
def test_shifted_boards_with_and_without_alignment(station, cfg, shift):
    station.lock.enabled = False
    mixed = [P001, P001, ["fork_right", "fork_right", "small", "fork_right"], P001]
    img = render_board(mixed, shift=shift, seed=300)
    rep = station.inspect("P001", image=img).report
    assert rep.alignment.applied and rep.alignment.ok
    assert sorted(p.key for p in rep.mismatches) == [(3, 1), (3, 4)]


def test_alignment_failure_is_ng(station):
    station.lock.enabled = False
    blank = read_image(SAMPLES / "reference.jpg")
    blank[:] = 240
    rep = station.inspect("P001", image=blank).report
    assert not rep.ok and rep.error.startswith("alignment failed")


def test_unknown_part_is_ng(station):
    station.lock.enabled = False
    rep = station.inspect("P999", image=read_image(SAMPLES / "boards" / "board_P001_ok.jpg")).report
    assert not rep.ok and "unknown part" in rep.error


def test_fork_orientation_required_by_master(station, cfg):
    station.lock.enabled = False
    cfg.parts["P003"] = PartNumber("P003", ["round", "fork_right", "small", "round"])
    rep = station.inspect("P003", image=read_image(SAMPLES / "boards" / "board_P001_ok.jpg")).report
    # P001 boards have left-facing forks: every row-2 position is wrong for P003
    assert sorted(p.key for p in rep.mismatches) == [(1, 2), (2, 2), (3, 2), (4, 2)]
    assert all("orientation" in p.reason for p in rep.mismatches)


def test_ng_locks_station_until_pass_or_supervisor(station):
    ok_img = read_image(SAMPLES / "boards" / "board_P001_ok.jpg")
    ng_img = read_image(SAMPLES / "boards" / "board_P001_mixed.jpg")

    ng = station.inspect("P001", "OP1", image=ng_img)
    assert station.lock.locked and station.lock.state.inspection_id == ng.inspection_id
    with pytest.raises(StationLocked):
        station.inspect("P002", image=ok_img)  # cannot move on to another part
    assert station.inspect("P001", image=ng_img).report.verdict.value == "NG"  # still wrong -> still locked
    assert station.lock.locked
    station.inspect("P001", "OP1", image=ok_img)  # fixed and re-inspected -> released
    assert not station.lock.locked

    station.inspect("P001", image=ng_img)
    assert station.lock.locked
    assert not station.acknowledge("9999", "SUP1")
    assert station.acknowledge("1234", "SUP1")
    assert not station.lock.locked
    kinds = [e["kind"] for e in station.store.events()]
    assert kinds == ["unlock_pass", "ack_failed", "ack"]


def test_every_result_logged_and_ng_images_saved(station):
    station.lock.enabled = False
    ok_img = read_image(SAMPLES / "boards" / "board_P001_ok.jpg")
    ng_img = read_image(SAMPLES / "boards" / "board_P001_missing.jpg")
    results = [station.inspect("P001", image=im) for im in (ok_img, ng_img, ok_img, ok_img, ng_img)]
    rows = station.store.inspections()
    assert len(rows) == 5
    for r in results:
        if not r.report.ok:
            assert r.image_path and Path(r.image_path).is_file()
    ok_saved = [r for r in results if r.report.ok and r.image_path]
    assert len(ok_saved) == 2  # save_ok_every_n = 2 -> 1st and 3rd OK
    rep = station.store.daily_report()
    assert (rep.total, rep.ng) == (5, 2) and rep.failing_positions[0] == (2, 3, 2)


def test_cli_inspect_exit_codes(tmp_path, capsys):
    ok = cli_main(["-c", str(CONFIG), "inspect", "--image", str(SAMPLES / "boards" / "board_P001_ok.jpg"),
                   "--part", "P001", "--no-store"])
    out = tmp_path / "annotated.jpg"
    ng = cli_main(["-c", str(CONFIG), "inspect", "--image", str(SAMPLES / "boards" / "board_P001_mixed.jpg"),
                   "--part", "P001", "--no-store", "--out", str(out)])
    text = capsys.readouterr().out
    assert (ok, ng) == (0, 1)
    assert "NG cable 3 row 1: expected round, found fork_right" in text
    assert out.is_file()
