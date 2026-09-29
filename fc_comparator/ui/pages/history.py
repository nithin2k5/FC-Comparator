"""History & reports page: browse inspections, view evidence, daily report, export."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ...capture.sources import read_image
from ...pipeline import Station
from ...storage import export_csv, export_excel
from ..widgets import Card, ImageView, Table, button, muted, show_error, show_info


class HistoryPage(ctk.CTkFrame):
    title = "History & reports"

    def __init__(self, master, app, station: Station):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.store = station.store
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        bar = Card(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        b = bar.body
        today = date.today().isoformat()
        self.date_from = ctk.CTkEntry(b, width=120, height=36)
        self.date_to = ctk.CTkEntry(b, width=120, height=36)
        for e in (self.date_from, self.date_to):
            e.insert(0, today)
        self.part = ctk.CTkEntry(b, width=130, height=36, placeholder_text="any part")
        self.result = ctk.CTkSegmentedButton(b, values=["All", "OK", "NG"], height=36, command=lambda _v: self.refresh())
        self.result.set("All")
        for label, w in (("From", self.date_from), ("To", self.date_to), ("Part", self.part), ("Result", self.result)):
            muted(b, label, 13).pack(side="left", padx=(0, 6))
            w.pack(side="left", padx=(0, 16))
        button(b, "Show", self.refresh, "primary", width=90).pack(side="left")
        button(b, "Export Excel", lambda: self.export("xlsx"), "secondary", width=120).pack(side="right")
        button(b, "Export CSV", lambda: self.export("csv"), "secondary", width=110).pack(side="right", padx=8)

        self.table = Table(self, [("id", "#", 55), ("ts", "Time", 150), ("part", "Part", 90), ("op", "Operator", 90),
                                  ("result", "Result", 60), ("ng", "Findings", 70), ("ms", "ms", 55)])
        self.table.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        self.table.on_select(self.show_detail)

        detail = Card(self, title="Inspection")
        detail.grid(row=1, column=1, sticky="nsew")
        detail.body.grid_columnconfigure(0, weight=1)
        detail.body.grid_rowconfigure(0, weight=3)
        detail.body.grid_rowconfigure(1, weight=2)
        self.image = ImageView(detail.body, placeholder="Select an inspection")
        self.image.grid(row=0, column=0, sticky="nsew")
        self.detail = ctk.CTkTextbox(detail.body, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.detail.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        report = Card(self, title="Daily report", subtitle="For the 'To' date")
        report.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.report = ctk.CTkTextbox(report.body, height=150, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.report.pack(fill="x")

    def _range(self) -> tuple[datetime, datetime]:
        start = datetime.combine(date.fromisoformat(self.date_from.get().strip()), time.min)
        end = datetime.combine(date.fromisoformat(self.date_to.get().strip()), time.min) + timedelta(days=1)
        return start, end

    def _filters(self) -> dict:
        f = {}
        if self.part.get().strip():
            f["part_number"] = self.part.get().strip()
        if self.result.get() != "All":
            f["result"] = self.result.get()
        return f

    @staticmethod
    def _set_text(box: ctk.CTkTextbox, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def refresh(self) -> None:
        try:
            start, end = self._range()
        except ValueError:
            show_error(self, "Dates must be written as YYYY-MM-DD.")
            return
        rows = []
        for r in self.store.inspections(start, end, limit=3000, **self._filters()):
            findings = r["ng_count"] + r.get("extras_count", 0)
            rows.append((str(r["id"]), (r["id"], str(r["ts"]).replace("T", " ")[:19], r["part_number"], r["operator_id"] or "-",
                                        r["result"], findings or "-", f"{r['duration_ms']:.0f}"),
                         ("ng",) if r["result"] == "NG" else ()))
        self.table.set_rows(rows)
        self._set_text(self.report, self.store.daily_report((end - timedelta(days=1)).date()).as_text())

    def show_detail(self, iid: str | None) -> None:
        row = self.store.get_inspection(int(iid)) if iid else None
        if row is None:
            return
        lines = [f"#{row['id']}  {row['result']}  {row['part_number']}  operator {row['operator_id'] or '-'}",
                 f"{row['ts']}  {row['duration_ms']:.0f} ms  detector {row['detector']}",
                 f"placement: {row['placement_message']}"]
        if row["error"]:
            lines.append(f"error: {row['error']}")
        for p in self.store.positions(row["id"]):
            if not p["ok"]:
                lines.append(f"  NG C{p['cable']}R{p['row']}: expected {p['expected']}, found {p['found']} "
                             f"({p['confidence']:.0%})  {p['reason']}")
        for e in self.store.extras(row["id"]):
            lines.append(f"  NG unexpected {e['label']} ({e['confidence']:.0%}) at x={e['x'] + e['w'] / 2:.0f} "
                         f"y={e['y'] + e['h'] / 2:.0f}")
        if len(lines) == 3 and not row["error"]:
            lines.append("  all positions OK")
        self._set_text(self.detail, "\n".join(lines))
        path = row["image_path"]
        if path and Path(path).is_file():
            try:
                self.image.set_image(read_image(path))
                return
            except Exception:
                pass
        self.image.clear("No image saved for this inspection\n(NG images are always saved; OK images every Nth)")

    def export(self, kind: str) -> None:
        try:
            start, end = self._range()
        except ValueError:
            show_error(self, "Dates must be written as YYYY-MM-DD.")
            return
        default = f"inspections_{start:%Y%m%d}_{(end - timedelta(days=1)):%Y%m%d}.{kind}"
        path = filedialog.asksaveasfilename(parent=self, title="Export", initialfile=default, defaultextension=f".{kind}",
                                            filetypes=[("Excel", "*.xlsx")] if kind == "xlsx" else [("CSV", "*.csv")])
        if not path:
            return
        try:
            if kind == "xlsx":
                show_info(self, f"Saved {export_excel(self.store, path, start, end, **self._filters())}", "Exported")
            else:
                a, b = export_csv(self.store, path, start, end, **self._filters())
                show_info(self, f"Saved\n{a}\n{b}", "Exported")
        except Exception as exc:
            show_error(self, f"Export failed: {exc}")
