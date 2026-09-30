"""History: every inspection with its findings and saved image, the daily report, CSV / Excel export."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from tkinter import filedialog, ttk

from ...station.storage import export_csv, export_excel
from ...vision.camera import read_image
from ..widgets import ImageView, Table, card, set_text, show_error, show_info, text_box


class HistoryPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.store = app.station.store
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        today = date.today().isoformat()
        ttk.Label(bar, text="From").pack(side="left")
        self.date_from = ttk.Entry(bar, width=11)
        self.date_from.insert(0, today)
        self.date_from.pack(side="left", padx=(4, 10))
        ttk.Label(bar, text="To").pack(side="left")
        self.date_to = ttk.Entry(bar, width=11)
        self.date_to.insert(0, today)
        self.date_to.pack(side="left", padx=(4, 10))
        ttk.Label(bar, text="Part").pack(side="left")
        self.part = ttk.Entry(bar, width=12)
        self.part.pack(side="left", padx=(4, 10))
        ttk.Label(bar, text="Result").pack(side="left")
        self.result = ttk.Combobox(bar, values=["All", "OK", "NG"], state="readonly", width=5)
        self.result.set("All")
        self.result.pack(side="left", padx=(4, 10))
        ttk.Button(bar, text="Show", style="Accent.TButton", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="Export Excel...", command=lambda: self.export("xlsx")).pack(side="right")
        ttk.Button(bar, text="Export CSV...", command=lambda: self.export("csv")).pack(side="right", padx=6)

        self.table = Table(self, [("id", "#", 50), ("ts", "Time", 140), ("part", "Part", 90), ("op", "Operator", 80),
                                  ("result", "Result", 60), ("findings", "Findings", 70), ("ms", "ms", 50)],
                           height=20, on_select=self.show_detail)
        self.table.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.image = ImageView(right, placeholder="Select an inspection")
        self.image.grid(row=0, column=0, sticky="nsew")
        self.detail = text_box(right, height=10)
        self.detail.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        rep = card(self, "Daily report (for the 'To' date)")
        rep.panel.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.report = text_box(rep, height=7)
        self.report.pack(fill="x")

    def on_show(self) -> None:
        self.refresh()

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

    def refresh(self) -> None:
        try:
            start, end = self._range()
        except ValueError:
            show_error(self, "Dates must be written as YYYY-MM-DD.")
            return
        rows = []
        for r in self.store.inspections(start, end, limit=3000, **self._filters()):
            findings = r["ng_count"] + r.get("extras_count", 0)
            rows.append((str(r["id"]), (r["id"], str(r["ts"]).replace("T", " ")[:19], r["part_number"],
                                        r["operator_id"] or "-", r["result"], findings or "-",
                                        f"{r['duration_ms']:.0f}"),
                         ("ng",) if r["result"] == "NG" else ()))
        self.table.set_rows(rows)
        set_text(self.report, self.store.daily_report((end - timedelta(days=1)).date()).as_text())

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
                lines.append(f"  NG cable {p['cable']} row {p['row']}: expected {p['expected']}, found {p['found']} "
                             f"({p['confidence']:.0%})  {p['reason']}")
        for e in self.store.extras(row["id"]):
            lines.append(f"  NG unexpected {e['label']} ({e['confidence']:.0%})")
        if len(lines) == 3 and not row["error"]:
            lines.append("  all positions OK")
        set_text(self.detail, "\n".join(lines))
        path = row["image_path"]
        if path and Path(path).is_file():
            try:
                self.image.set_image(read_image(path))
                return
            except Exception:
                pass
        self.image.clear("No image saved for this inspection\n(NG images are always saved, OK images every Nth)")

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
