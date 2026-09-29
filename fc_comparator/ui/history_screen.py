"""History / reports screen: browse inspections, view evidence, export, daily report."""

from __future__ import annotations

import tkinter as tk
from datetime import date, datetime, time, timedelta
from pathlib import Path
from tkinter import filedialog, ttk

from ..capture.sources import read_image
from ..pipeline import Station
from ..storage import export_csv, export_excel
from .widgets import NG_COLOR, ImageView, error_box, info_box

COLUMNS = ("id", "ts", "part_number", "operator_id", "result", "ng_count", "duration_ms")
HEADERS = ("#", "Time", "Part", "Operator", "Result", "NG pos.", "ms")
WIDTHS = (60, 170, 100, 100, 70, 70, 60)


class HistoryScreen(ttk.Frame):
    def __init__(self, master, station: Station):
        super().__init__(master)
        self.station = station
        self.store = station.store
        today = date.today().isoformat()

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 6))
        self.from_var, self.to_var = tk.StringVar(value=today), tk.StringVar(value=today)
        self.part_var, self.result_var = tk.StringVar(value="All"), tk.StringVar(value="All")
        self.part = ttk.Combobox(bar, textvariable=self.part_var, state="readonly", width=10)
        widgets = (
            ("From", ttk.Entry(bar, textvariable=self.from_var, width=11)),
            ("To", ttk.Entry(bar, textvariable=self.to_var, width=11)),
            ("Part", self.part),
            ("Result", ttk.Combobox(bar, textvariable=self.result_var, state="readonly", width=5, values=["All", "OK", "NG"])),
        )
        for label, w in widgets:
            ttk.Label(bar, text=label).pack(side="left", padx=(8, 4))
            w.pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left", padx=(12, 0))

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)
        table_frame = ttk.Frame(panes)
        self.table = ttk.Treeview(table_frame, columns=COLUMNS, show="headings", selectmode="browse")
        for c, h, w in zip(COLUMNS, HEADERS, WIDTHS):
            self.table.heading(c, text=h)
            self.table.column(c, width=w, anchor="center")
        self.table.tag_configure("ng", foreground=NG_COLOR)
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        panes.add(table_frame, weight=1)

        detail_pane = ttk.PanedWindow(panes, orient="vertical")
        self.image = ImageView(detail_pane, placeholder="No saved image for this inspection")
        self.detail = tk.Text(detail_pane, height=10, wrap="none", font=("Consolas", 11))
        detail_pane.add(self.image, weight=3)
        detail_pane.add(self.detail, weight=2)
        panes.add(detail_pane, weight=1)

        report_bar = ttk.Frame(self)
        report_bar.pack(fill="x", pady=(6, 2))
        ttk.Label(report_bar, text="Daily report (for the 'To' date)").pack(side="left")
        ttk.Button(report_bar, text="Export Excel", command=lambda: self.export("xlsx")).pack(side="right")
        ttk.Button(report_bar, text="Export CSV", command=lambda: self.export("csv")).pack(side="right", padx=(0, 4))
        self.report = tk.Text(self, height=9, wrap="none", font=("Consolas", 11))
        self.report.pack(fill="x")

    def _range(self) -> tuple[datetime, datetime]:
        start = datetime.combine(date.fromisoformat(self.from_var.get().strip()), time.min)
        end = datetime.combine(date.fromisoformat(self.to_var.get().strip()), time.min) + timedelta(days=1)
        return start, end

    def _filters(self) -> dict:
        f = {}
        if self.part_var.get() not in ("", "All"):
            f["part_number"] = self.part_var.get()
        if self.result_var.get() != "All":
            f["result"] = self.result_var.get()
        return f

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def refresh(self) -> None:
        self.part.configure(values=["All", *sorted(self.station.cfg.parts)])
        try:
            start, end = self._range()
        except ValueError:
            error_box(self, "Dates must be YYYY-MM-DD.")
            return
        rows = self.store.inspections(start, end, limit=2000, **self._filters())
        self.table.delete(*self.table.get_children())
        for row in rows:
            values = []
            for key in COLUMNS:
                v = row[key]
                if key == "ts":
                    v = str(v).replace("T", " ")[:19]
                elif key == "duration_ms":
                    v = f"{v:.0f}"
                values.append(v)
            self.table.insert("", "end", iid=str(row["id"]), values=values, tags=("ng",) if row["result"] == "NG" else ())
        self._set_text(self.report, self.store.daily_report((end - timedelta(days=1)).date()).as_text())

    def _on_select(self) -> None:
        sel = self.table.selection()
        row = self.store.get_inspection(int(sel[0])) if sel else None
        if row is None:
            return
        lines = [
            f"Inspection #{row['id']}  {row['result']}  part {row['part_number']}  operator {row['operator_id'] or '-'}",
            f"{row['ts']}  {row['duration_ms']:.0f} ms  classifier {row['classifier']}  alignment: {row['align_message']}",
        ]
        if row["error"]:
            lines.append(f"Error: {row['error']}")
        for p in self.store.positions(row["id"]):
            lines.append(
                f"  {'OK' if p['ok'] else 'NG'} C{p['cable']}R{p['row']}  expected {p['expected']:<10} "
                f"found {p['found']:<10} {p['confidence']:.0%}  {p['reason']}"
            )
        self._set_text(self.detail, "\n".join(lines))
        path = row["image_path"]
        if path and Path(path).is_file():
            try:
                self.image.set_image(read_image(path))
                return
            except Exception:
                pass
        self.image.clear("No saved image for this inspection")

    def export(self, kind: str) -> None:
        try:
            start, end = self._range()
        except ValueError:
            error_box(self, "Dates must be YYYY-MM-DD.")
            return
        default = f"inspections_{start:%Y%m%d}_{(end - timedelta(days=1)):%Y%m%d}.{kind}"
        types = [("Excel", "*.xlsx")] if kind == "xlsx" else [("CSV", "*.csv")]
        path = filedialog.asksaveasfilename(parent=self, title="Export", initialfile=default,
                                            defaultextension=f".{kind}", filetypes=types)
        if not path:
            return
        try:
            if kind == "xlsx":
                info_box(self, f"Exported to {export_excel(self.store, path, start, end, **self._filters())}")
            else:
                a, b = export_csv(self.store, path, start, end, **self._filters())
                info_box(self, f"Exported to\n{a}\n{b}")
        except Exception as exc:
            error_box(self, f"Export failed: {exc}")
