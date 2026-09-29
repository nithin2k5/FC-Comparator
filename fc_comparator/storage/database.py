"""SQLite traceability store: every inspection, every position, every acknowledgement."""

from __future__ import annotations

import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ..models import InspectionReport

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    station       TEXT NOT NULL DEFAULT '',
    operator_id   TEXT NOT NULL DEFAULT '',
    part_number   TEXT NOT NULL,
    mode          TEXT NOT NULL,
    result        TEXT NOT NULL CHECK (result IN ('OK', 'NG')),
    ng_count      INTEGER NOT NULL,
    duration_ms   REAL NOT NULL,
    classifier    TEXT NOT NULL DEFAULT '',
    aligned       INTEGER NOT NULL DEFAULT 0,
    align_shift   REAL NOT NULL DEFAULT 0,
    align_message TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    image_path    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_inspections_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS ix_inspections_part ON inspections(part_number, ts);

CREATE TABLE IF NOT EXISTS positions (
    inspection_id INTEGER NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    cable         INTEGER NOT NULL,
    row           INTEGER NOT NULL,
    expected      TEXT NOT NULL,
    found         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    ok            INTEGER NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (inspection_id, cable, row)
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    actor         TEXT NOT NULL DEFAULT '',
    detail        TEXT NOT NULL DEFAULT '',
    inspection_id INTEGER REFERENCES inspections(id)
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
"""


@dataclass
class DailyReport:
    day: date
    total: int = 0
    ok: int = 0
    ng: int = 0
    by_part: dict[str, tuple[int, int]] = field(default_factory=dict)  # part -> (total, ng)
    failing_positions: list[tuple[int, int, int]] = field(default_factory=list)  # (cable, row, count)
    failure_kinds: list[tuple[str, str, int]] = field(default_factory=list)  # (expected, found, count)
    acknowledgements: int = 0

    @property
    def ng_rate(self) -> float:
        return self.ng / self.total if self.total else 0.0

    def as_text(self) -> str:
        lines = [
            f"Daily report {self.day.isoformat()}",
            f"  Inspected: {self.total}   OK: {self.ok}   NG: {self.ng}   NG rate: {self.ng_rate:.1%}",
            f"  Supervisor acknowledgements: {self.acknowledgements}",
        ]
        if self.by_part:
            lines.append("  By part number:")
            lines += [f"    {p:<12} {t:>5} inspected {n:>4} NG" for p, (t, n) in sorted(self.by_part.items())]
        if self.failing_positions:
            lines.append("  Most frequent failing positions:")
            lines += [f"    cable {c} row {r}: {n}" for c, r, n in self.failing_positions]
        if self.failure_kinds:
            lines.append("  Most frequent failures (found instead of expected):")
            lines += [f"    {f} instead of {e}: {n}" for e, f, n in self.failure_kinds]
        return "\n".join(lines)


class InspectionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
            self._db.executescript(SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writes ------------------------------------------------------------
    def save_report(self, report: InspectionReport, station: str = "", image_path: str = "") -> int:
        a = report.alignment
        with self._lock:
            cur = self._db.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute(
                    """INSERT INTO inspections (ts, station, operator_id, part_number, mode, result, ng_count,
                       duration_ms, classifier, aligned, align_shift, align_message, error, image_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        report.timestamp.isoformat(timespec="milliseconds"),
                        station,
                        report.operator_id,
                        report.part_number,
                        report.mode,
                        report.verdict.value,
                        len(report.mismatches),
                        report.duration_ms,
                        report.classifier,
                        int(a.applied),
                        a.shift_px,
                        a.message,
                        report.error,
                        image_path,
                    ),
                )
                iid = cur.lastrowid
                cur.executemany(
                    "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?)",
                    [(iid, p.cable, p.row, p.expected, p.found, p.confidence, int(p.ok), p.reason) for p in report.positions],
                )
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
        return iid

    def set_image_path(self, inspection_id: int, image_path: str) -> None:
        with self._lock:
            self._db.execute("UPDATE inspections SET image_path = ? WHERE id = ?", (image_path, inspection_id))

    def log_event(self, kind: str, actor: str = "", detail: str = "", inspection_id: int | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts, kind, actor, detail, inspection_id) VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(timespec="milliseconds"), kind, actor, detail, inspection_id),
            )
            return cur.lastrowid

    # -- reads -------------------------------------------------------------
    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def inspections(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        part_number: str | None = None,
        result: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Inspections newest first; ``end`` is exclusive."""
        where, args = [], []
        if start:
            where.append("ts >= ?")
            args.append(start.isoformat())
        if end:
            where.append("ts < ?")
            args.append(end.isoformat())
        if part_number:
            where.append("part_number = ?")
            args.append(part_number)
        if result:
            where.append("result = ?")
            args.append(result)
        sql = "SELECT * FROM inspections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self._all(sql, tuple(args))

    def positions(self, inspection_ids: int | list[int]) -> list[dict[str, Any]]:
        ids = [inspection_ids] if isinstance(inspection_ids, int) else list(inspection_ids)
        if not ids:
            return []
        out = []
        for i in range(0, len(ids), 500):  # SQLite parameter limit
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            out += self._all(
                f"SELECT * FROM positions WHERE inspection_id IN ({marks}) ORDER BY inspection_id, cable, row", tuple(chunk)
            )
        return out

    def events(self, start: datetime | None = None, end: datetime | None = None) -> list[dict[str, Any]]:
        start = start or datetime.min
        end = end or datetime.max
        return self._all("SELECT * FROM events WHERE ts >= ? AND ts < ? ORDER BY ts", (start.isoformat(), end.isoformat()))

    def daily_report(self, day: date | None = None, top: int = 5) -> DailyReport:
        day = day or date.today()
        start = datetime.combine(day, time.min)
        end = start + timedelta(days=1)
        rows = self.inspections(start, end)
        rep = DailyReport(day)
        rep.total = len(rows)
        rep.ng = sum(r["result"] == "NG" for r in rows)
        rep.ok = rep.total - rep.ng
        parts: dict[str, list[int]] = {}
        for r in rows:
            t = parts.setdefault(r["part_number"], [0, 0])
            t[0] += 1
            t[1] += r["result"] == "NG"
        rep.by_part = {k: (v[0], v[1]) for k, v in parts.items()}
        bad = [p for p in self.positions([r["id"] for r in rows]) if not p["ok"]]
        rep.failing_positions = [(c, r, n) for (c, r), n in Counter((p["cable"], p["row"]) for p in bad).most_common(top)]
        rep.failure_kinds = [(e, f, n) for (e, f), n in Counter((p["expected"], p["found"]) for p in bad).most_common(top)]
        rep.acknowledgements = sum(e["kind"] == "ack" for e in self.events(start, end))
        return rep
