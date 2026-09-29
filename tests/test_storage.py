import csv
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from fc_comparator.models import AlignmentInfo, InspectionReport, PositionResult
from fc_comparator.storage import ImageSaver, InspectionStore, export_csv, export_excel


def report(ok_positions: int, bad: list[tuple[int, int, str, str]], ts: datetime, part="P001") -> InspectionReport:
    positions = []
    bad_map = {(c, r): (e, f) for c, r, e, f in bad}
    for c in range(1, 5):
        for r in range(1, 5):
            if (c, r) in bad_map:
                e, f = bad_map[(c, r)]
                positions.append(PositionResult(c, r, e, f, 0.8, False, f"{f} != {e}"))
            else:
                positions.append(PositionResult(c, r, "round", "round", 0.97, True))
    return InspectionReport(part, "master", positions, AlignmentInfo(True, True, 80, 12.5, "aligned"), ts, 123.4, "OP7", "template")


@pytest.fixture
def store(tmp_path):
    s = InspectionStore(tmp_path / "db" / "inspections.db")
    yield s
    s.close()


def test_save_and_query(store):
    t0 = datetime(2026, 9, 28, 8, 0)
    ok_id = store.save_report(report(16, [], t0), station="S1")
    ng_id = store.save_report(report(15, [(2, 3, "small", "fork_left")], t0 + timedelta(minutes=1)), image_path="x.jpg")
    rows = store.inspections()
    assert [r["id"] for r in rows] == [ng_id, ok_id]  # newest first
    ng = rows[0]
    assert ng["result"] == "NG" and ng["ng_count"] == 1 and ng["operator_id"] == "OP7"
    assert ng["aligned"] == 1 and ng["align_shift"] == 12.5 and ng["image_path"] == "x.jpg"
    pos = store.positions(ng_id)
    assert len(pos) == 16
    bad = [p for p in pos if not p["ok"]]
    assert bad == [{"inspection_id": ng_id, "cable": 2, "row": 3, "expected": "small", "found": "fork_left",
                    "confidence": 0.8, "ok": 0, "reason": "fork_left != small"}]
    assert store.inspections(result="OK")[0]["id"] == ok_id
    assert store.inspections(part_number="P999") == []
    assert store.get_inspection(ok_id)["result"] == "OK"
    assert store.get_inspection(9999) is None
    store.set_image_path(ok_id, "ok.jpg")
    assert store.inspections(result="OK")[0]["image_path"] == "ok.jpg"


def test_daily_report(store):
    day = date(2026, 9, 28)
    base = datetime.combine(day, datetime.min.time()) + timedelta(hours=9)
    store.save_report(report(16, [], base))
    store.save_report(report(15, [(1, 2, "fork", "round")], base + timedelta(minutes=1)))
    store.save_report(report(14, [(1, 2, "fork", "round"), (4, 4, "round", "missing")], base + timedelta(minutes=2), "P002"))
    store.save_report(report(16, [], base + timedelta(days=1)))  # other day
    store.log_event("ack", "SUP", "acknowledged NG")
    rep = store.daily_report(day)
    assert (rep.total, rep.ok, rep.ng) == (3, 1, 2)
    assert rep.ng_rate == pytest.approx(2 / 3)
    assert rep.failing_positions[0] == (1, 2, 2)
    assert (1, 2, 2) in rep.failing_positions and (4, 4, 1) in rep.failing_positions
    assert rep.failure_kinds[0] == ("fork", "round", 2)
    assert rep.by_part == {"P001": (2, 1), "P002": (1, 1)}
    assert "NG: 2" in rep.as_text()


def test_export_csv(store, tmp_path):
    t = datetime(2026, 9, 28, 10, 0)
    store.save_report(report(16, [], t))
    store.save_report(report(15, [(3, 1, "fork", "round")], t + timedelta(seconds=5)))
    main, pos = export_csv(store, tmp_path / "out" / "history.csv")
    with open(main, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["result"] for r in rows] == ["OK", "NG"]  # oldest first
    assert rows[1]["mismatches"] == "C3R1:round!=fork(0.80)"
    with open(pos, encoding="utf-8-sig") as fh:
        assert len(list(csv.DictReader(fh))) == 32


def test_export_excel(store, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    store.save_report(report(15, [(3, 1, "fork", "round")], datetime(2026, 9, 28, 10, 0)))
    path = export_excel(store, tmp_path / "history.xlsx")
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Inspections", "Positions", "Daily report"]
    assert wb["Daily report"]["D2"].value == 1


def test_image_saver(tmp_path):
    saver = ImageSaver(tmp_path, save_ok_every_n=3)
    ts = datetime(2026, 9, 28, 10, 11, 12)
    ok, ng = report(16, [], ts), report(15, [(1, 1, "round", "small")], ts, part="P/0 01")
    assert saver.should_save(ng)
    assert [saver.should_save(ok) for _ in range(6)] == [True, False, False, True, False, False]
    p = saver.save(np.zeros((10, 10, 3), np.uint8), ng, 42)
    assert p.exists() and p.parent.name == "2026-09-28" and p.name == "101112_000042_P_0_01_NG.jpg"
    assert not ImageSaver(tmp_path, save_ok_every_n=0).should_save(ok)
