"""Inspect screen: live preview, part selection/scan, INSPECT, OK/NG banner, mismatch list, NG lock."""

from __future__ import annotations

import logging
import tkinter as tk
from datetime import date
from tkinter import filedialog, ttk

from ..barcode import parse_scan
from ..capture import FileSource
from ..capture.sources import read_image
from ..pipeline import Station, StationLocked, StationResult
from .widgets import IDLE_COLOR, NG_COLOR, OK_COLOR, WARN_COLOR, Banner, Dispatcher, ImageView, PinDialog, error_box

log = logging.getLogger(__name__)


class InspectScreen(ttk.Frame):
    def __init__(self, master, station: Station, dispatcher: Dispatcher, app=None):
        super().__init__(master)
        self.station = station
        self.cfg = station.cfg
        self.dispatcher = dispatcher
        self.app = app  # MainWindow (navigation locking)
        self.busy = False
        self._showing_result = False
        self._file_image = None  # image loaded via "Load image" while in camera mode

        # -- left: image ------------------------------------------------------
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.view = ImageView(left, placeholder="Waiting for camera...")
        self.view.pack(fill="both", expand=True)
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Live view", command=self.show_live).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btns, text="Load image...", command=self.load_image).pack(side="left", fill="x", expand=True)

        # -- right: controls ----------------------------------------------------
        right = ttk.Frame(self, width=560)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.banner = Banner(right)
        self.banner.pack(fill="x")

        form = ttk.Frame(right)
        form.pack(fill="x", pady=8)
        form.columnconfigure(1, weight=1)
        self.scan_var = tk.StringVar()
        self.scan = ttk.Entry(form, textvariable=self.scan_var)
        self.scan.bind("<Return>", lambda _e: self.on_scan())
        self.part_var = tk.StringVar()
        self.part = ttk.Combobox(form, textvariable=self.part_var, state="readonly")
        self.part.bind("<<ComboboxSelected>>", lambda _e: self._on_part_changed())
        self.pattern_lbl = ttk.Label(form, wraplength=380, justify="left")
        self.operator_var = tk.StringVar()
        self.operator = ttk.Entry(form, textvariable=self.operator_var)
        for r, (label, w) in enumerate(
            (("Scan", self.scan), ("Part number", self.part), ("Master pattern", self.pattern_lbl), ("Operator", self.operator))
        ):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=4)
            w.grid(row=r, column=1, sticky="ew", pady=4)

        self.inspect_btn = ttk.Button(right, style="Primary.TButton", command=self.inspect)
        self.inspect_btn.pack(fill="x", pady=(4, 6))
        ack_slot = ttk.Frame(right)  # keeps the ack button's place in the layout while hidden
        ack_slot.pack(fill="x")
        self.ack_btn = ttk.Button(ack_slot, text="Supervisor acknowledge (PIN)", style="Danger.TButton", command=self.acknowledge)

        ttk.Label(right, text="Mismatches").pack(anchor="w")
        cols = ("cable", "row", "expected", "found", "conf")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=6, selectmode="none")
        for c, w in zip(cols, (70, 60, 130, 130, 80)):
            self.table.heading(c, text=c.capitalize() if c != "conf" else "Conf.")
            self.table.column(c, width=w, anchor="center")
        self.table.tag_configure("ng", foreground=NG_COLOR)
        self.table.pack(fill="both", expand=True)
        self.stats_lbl = ttk.Label(right)
        self.stats_lbl.pack(anchor="w", pady=(6, 0))
        self.timing_lbl = ttk.Label(right, style="Muted.TLabel", wraplength=540, justify="left")
        self.timing_lbl.pack(anchor="w")

        self.station.lock.subscribe(lambda _s: self.dispatcher.call(self.update_lock_ui))
        self._preview_ms = max(30, int(1000 / max(1, self.cfg.ui.preview_fps)))
        self._preview_job = self.after(self._preview_ms, self._preview_tick)

        self.refresh_parts()
        self.update_lock_ui()
        self._update_stats()

    # -- parts / scanning ------------------------------------------------------
    def refresh_parts(self) -> None:
        current = self.part_var.get()
        codes = sorted(self.cfg.parts)
        self.part.configure(values=codes)
        if current in self.cfg.parts:
            self.part_var.set(current)
        elif codes:
            self.part_var.set(codes[0])
        else:
            self.part_var.set("")
        self._on_part_changed()

    def _on_part_changed(self) -> None:
        pn = self.cfg.parts.get(self.part_var.get())
        if pn is None:
            self.pattern_lbl.configure(text="-" if self.cfg.compare.mode != "cross" else "(cross-cable mode)")
            return
        rows = "  ".join(f"R{i + 1}: {p}" for i, p in enumerate(pn.pattern))
        self.pattern_lbl.configure(text=f"{rows}\n{pn.description}" if pn.description else rows)

    def on_scan(self) -> None:
        text = self.scan_var.get()
        self.scan_var.set("")
        s = parse_scan(text, self.cfg.barcode, set(self.cfg.parts))
        if s.kind == "operator":
            self.operator_var.set(s.value)
        elif s.kind == "part":
            if self.station.lock.locked and s.value != self.station.lock.state.part_number:
                self.banner.show_state("LOCKED", NG_COLOR, "Resolve the NG before changing part")
                return
            self.part_var.set(s.value)
            self._on_part_changed()
            self.banner.show_state("READY", IDLE_COLOR, s.value)
        else:
            self.banner.show_state("UNKNOWN", WARN_COLOR, f"Barcode not recognised: {text.strip()[:30]}")

    def focus_scan(self) -> None:
        self.scan.focus_set()

    # -- preview -------------------------------------------------------------
    def _preview_tick(self) -> None:
        try:
            if not self._showing_result and self.winfo_ismapped():
                frame = self._file_image if self._file_image is not None else self.station.source.latest()
                if frame is not None:
                    self.view.set_image(frame)
                err = getattr(self.station.source, "error", "")
                if err:
                    self.timing_lbl.configure(text=f"Camera: {err}")
        finally:
            self._preview_job = self.after(self._preview_ms, self._preview_tick)

    def show_live(self) -> None:
        self._showing_result = False
        self._file_image = None

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Load board image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            error_box(self, str(exc))
            return
        if isinstance(self.station.source, FileSource):
            self.station.source.set_image(img)
        else:
            self._file_image = img  # camera mode: inspect this file once instead of a live frame
        self._showing_result = False
        self.view.set_image(img)

    # -- inspection ---------------------------------------------------------
    def inspect(self) -> None:
        if self.busy:
            return
        part = self.part_var.get()
        if not part and self.cfg.compare.mode != "cross":
            self.banner.show_state("NO PART", WARN_COLOR, "Scan or select a part number")
            return
        ok, why = self.station.can_inspect(part)
        if not ok:
            self.banner.show_state("LOCKED", NG_COLOR, why)
            return
        self.busy = True
        self.inspect_btn.state(["disabled"])
        self.banner.show_state("INSPECTING...", IDLE_COLOR)
        image, operator = self._file_image, self.operator_var.get().strip()
        self.dispatcher.run_task(lambda: self.station.inspect(part, operator, image=image), self._on_result, self._on_error)

    def _on_result(self, res: StationResult) -> None:
        self.busy = False
        self.inspect_btn.state(["!disabled"])
        self._file_image = None
        rep = res.report
        self._showing_result = True
        self.view.set_image(res.annotated)
        if rep.ok:
            self.banner.show_state("OK", OK_COLOR, f"{rep.part_number}: all {len(rep.positions)} positions correct")
        else:
            self.banner.show_state("NG", NG_COLOR, rep.error or f"{len(rep.mismatches)} position(s) wrong")
        self.table.delete(*self.table.get_children())
        for p in rep.mismatches:
            self.table.insert("", "end", values=(p.cable, p.row, p.expected, p.found, f"{p.confidence:.0%}"), tags=("ng",))
        alert_err = self.station.alert.last_error
        self.timing_lbl.configure(
            text=f"#{res.inspection_id}  inspection {rep.duration_ms:.0f} ms, alert after {res.alert_latency_ms:.0f} ms, "
            f"{rep.classifier}, {rep.alignment.message}" + (f"\nALERT FAULT: {alert_err}" if alert_err else "")
        )
        self.update_lock_ui()
        self._update_stats()
        self.focus_scan()

    def _on_error(self, exc: BaseException) -> None:
        self.busy = False
        self.inspect_btn.state(["!disabled"])
        if isinstance(exc, StationLocked):
            self.banner.show_state("LOCKED", NG_COLOR, str(exc))
            return
        log.error("Inspection failed: %s", exc)
        self.station.alert.ng()  # a failed inspection is never silently OK
        self.banner.show_state("ERROR", NG_COLOR, str(exc)[:80])

    # -- lock ------------------------------------------------------------------
    def update_lock_ui(self) -> None:
        st = self.station.lock.state
        key = self.cfg.trigger.key
        if st.locked:
            self.ack_btn.pack(fill="x", pady=(0, 6))
            self.part.state(["disabled"])
            self.inspect_btn.configure(text=f"RE-INSPECT {st.part_number}  [{key}]")
        else:
            self.ack_btn.pack_forget()
            self.part.state(["!disabled", "readonly"])
            self.inspect_btn.configure(text=f"INSPECT  [{key}]")
        if self.app is not None:
            self.app.set_navigation_locked(st.locked)

    def acknowledge(self) -> None:
        pin = PinDialog.ask(self, "Supervisor PIN", numeric_only=True)
        if pin is None:
            return
        if self.station.acknowledge(pin, self.operator_var.get().strip()):
            self.banner.show_state("ACKNOWLEDGED", WARN_COLOR, "Station unlocked by supervisor")
        else:
            self.banner.show_state("WRONG PIN", NG_COLOR, "Station remains locked")
        self.update_lock_ui()

    def _update_stats(self) -> None:
        try:
            rep = self.station.store.daily_report(date.today())
        except Exception:
            return
        self.stats_lbl.configure(text=f"Today: {rep.total} inspected, {rep.ok} OK, {rep.ng} NG ({rep.ng_rate:.1%})")

    def stop(self) -> None:
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
