"""Inspect page: live preview, part selection (scan / search), INSPECT, verdict, findings, NG lock."""

from __future__ import annotations

import logging
from datetime import date
from tkinter import filedialog

import customtkinter as ctk

from ...barcode import parse_scan
from ...capture import FileSource
from ...capture.sources import read_image
from ...models import MISSING
from ...pipeline import Station, StationLocked, StationResult
from .. import theme
from ..widgets import Banner, Card, Dispatcher, ImageView, PartPicker, PinDialog, button, chip, muted, show_error

log = logging.getLogger(__name__)


class InspectPage(ctk.CTkFrame):
    title = "Inspect"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.dispatcher = dispatcher
        self.busy = False
        self._showing_result = False
        self._file_image = None  # an image loaded from disk while in camera mode
        self._last_annotated = None
        self.part_code = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- left: image ------------------------------------------------------
        left = Card(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.body.grid_columnconfigure(0, weight=1)
        left.body.grid_rowconfigure(0, weight=1)
        self.view = ImageView(left.body, placeholder="Waiting for camera ...")
        self.view.grid(row=0, column=0, sticky="nsew")
        bar = ctk.CTkFrame(left.body, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.view_mode = ctk.CTkSegmentedButton(bar, values=["Live", "Last result"], command=self._on_view_mode,
                                                height=38, font=theme.font(13, "bold"))
        self.view_mode.set("Live")
        self.view_mode.pack(side="left")
        button(bar, "Load image ...", self.load_image, "secondary", width=150).pack(side="right")
        self.timing = muted(bar, "")
        self.timing.pack(side="left", padx=16)

        # -- right: controls ----------------------------------------------------
        right = ctk.CTkFrame(self, fg_color="transparent", width=440)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(4, weight=1)

        self.banner = Banner(right)
        self.banner.grid(row=0, column=0, sticky="ew")

        part_card = Card(right)
        part_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = part_card.body
        b.grid_columnconfigure(0, weight=1)
        muted(b, "Scan a barcode, or type a part number and press Enter", 12).grid(row=0, column=0, columnspan=2, sticky="w")
        self.scan = ctk.CTkEntry(b, placeholder_text="P001, OP:1234 ...", height=44, font=theme.font(15))
        self.scan.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.scan.bind("<Return>", lambda _e: self.on_scan())
        self.part_label = ctk.CTkLabel(b, text="No part selected", font=theme.font(26, "bold"), text_color=theme.TEXT,
                                       anchor="w")
        self.part_label.grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.pick_btn = button(b, "Choose ...", self.choose_part, "secondary", width=110)
        self.pick_btn.grid(row=2, column=1, sticky="e", pady=(12, 0))
        self.part_desc = muted(b, "", 13)
        self.part_desc.grid(row=3, column=0, columnspan=2, sticky="w")
        self.chips = ctk.CTkFrame(b, fg_color="transparent")
        self.chips.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        op = ctk.CTkFrame(b, fg_color="transparent")
        op.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        muted(op, "Operator", 13).pack(side="left")
        self.operator = ctk.CTkEntry(op, placeholder_text="ID or scan OP:badge", height=36, font=theme.font(14))
        self.operator.pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.inspect_btn = button(right, f"INSPECT   [{self.cfg.trigger.key}]", self.inspect, "primary",
                                  height=84, corner_radius=14, font=theme.font(26, "bold"))
        self.inspect_btn.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.ack_slot = ctk.CTkFrame(right, fg_color="transparent", height=0)
        self.ack_slot.grid(row=3, column=0, sticky="ew")
        self.ack_btn = button(self.ack_slot, "Supervisor acknowledge (PIN)", self.acknowledge, "danger", height=50)

        findings = Card(right, title="Findings")
        findings.grid(row=4, column=0, sticky="nsew", pady=(14, 0))
        findings.body.grid_columnconfigure(0, weight=1)
        findings.body.grid_rowconfigure(0, weight=1)
        self.findings = ctk.CTkScrollableFrame(findings.body, fg_color="transparent")
        self.findings.grid(row=0, column=0, sticky="nsew")
        self.stats = muted(right, "", 13)
        self.stats.grid(row=5, column=0, sticky="w", pady=(10, 0))

        self.station.lock.subscribe(lambda _s: self.dispatcher.call(self.update_lock_ui))
        self._preview_ms = max(30, int(1000 / max(1, self.cfg.ui.preview_fps)))
        self._preview_job = self.after(self._preview_ms, self._preview_tick)
        self._set_findings([])
        codes = sorted(self.cfg.parts)
        self.set_part(codes[0] if codes else "")
        self.update_lock_ui()
        self._update_stats()

    # -- part selection ----------------------------------------------------------
    def refresh(self) -> None:
        if self.part_code not in self.cfg.parts:
            codes = sorted(self.cfg.parts)
            self.set_part(codes[0] if codes else "")
        else:
            self.set_part(self.part_code)
        self._update_stats()

    def set_part(self, code: str) -> None:
        self.part_code = code
        for w in self.chips.winfo_children():
            w.destroy()
        pn = self.cfg.parts.get(code)
        if pn is None:
            self.part_label.configure(text="No part selected")
            self.part_desc.configure(text="Scan a barcode or choose a part number")
            return
        self.part_label.configure(text=code)
        layout = "master layout" if pn.layout else "no master image yet"
        self.part_desc.configure(text=f"{pn.description or 'No description'}  ·  {pn.cables} cables x {pn.rows} clips  ·  {layout}")
        for i, label in enumerate(pn.pattern):
            color = theme.class_color(self.cfg.taxonomy.classes, label) if label in self.cfg.taxonomy.classes else "#475569"
            chip(self.chips, f"R{i + 1} {label}", color).pack(side="left", padx=(0, 6))

    def choose_part(self) -> None:
        code = PartPicker(self, self.cfg.parts).run()
        if code:
            self._change_part(code)

    def _change_part(self, code: str) -> None:
        st = self.station.lock.state
        if st.locked and code != st.part_number:
            self.banner.show("LOCKED", theme.NG, "Resolve the NG before changing the part")
            return
        self.set_part(code)
        self.banner.show("READY", theme.IDLE, f"{code} selected")

    def on_scan(self) -> None:
        text = self.scan.get()
        self.scan.delete(0, "end")
        s = parse_scan(text, self.cfg.barcode, set(self.cfg.parts))
        if s.kind == "operator":
            self.operator.delete(0, "end")
            self.operator.insert(0, s.value)
        elif s.kind == "part":
            self._change_part(s.value)
        else:
            self.banner.show("UNKNOWN", theme.WARN, f"Not a known part number: {text.strip()[:30]}")

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
                    self.timing.configure(text=f"Camera: {err}")
        finally:
            self._preview_job = self.after(self._preview_ms, self._preview_tick)

    def _on_view_mode(self, value: str) -> None:
        if value == "Live":
            self._showing_result = False
            self._file_image = None
        elif self._last_annotated is not None:
            self._showing_result = True
            self.view.set_image(self._last_annotated)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Load board image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            show_error(self, str(exc))
            return
        if isinstance(self.station.source, FileSource):
            self.station.source.set_image(img)
        else:
            self._file_image = img  # camera mode: the next inspection uses this file
        self._showing_result = False
        self.view_mode.set("Live")
        self.view.set_image(img)

    # -- inspection ---------------------------------------------------------
    def inspect(self) -> None:
        if self.busy:
            return
        part = self.part_code
        if not part:
            self.banner.show("NO PART", theme.WARN, "Scan or choose a part number first")
            return
        ok, why = self.station.can_inspect(part)
        if not ok:
            self.banner.show("LOCKED", theme.NG, why)
            return
        self.busy = True
        self.inspect_btn.configure(state="disabled", text="Inspecting ...")
        image, operator = self._file_image, self.operator.get().strip()
        self.dispatcher.run_task(lambda: self.station.inspect(part, operator, image=image), self._on_result, self._on_error)

    def _on_result(self, res: StationResult) -> None:
        self.busy = False
        self._file_image = None
        rep = res.report
        self._last_annotated = res.annotated
        self._showing_result = True
        self.view_mode.set("Last result")
        self.view.set_image(res.annotated)
        if rep.ok:
            self.banner.show("OK", theme.OK, f"{rep.part_number}: all {len(rep.positions)} clips correct")
        else:
            n = len(rep.mismatches) + len(rep.extras)
            self.banner.show("NG", theme.NG, rep.error or f"{n} finding(s) - see the list below")
        self._set_findings(self._finding_rows(rep))
        err = self.station.alert.last_error
        self.timing.configure(text=f"#{res.inspection_id}  {rep.duration_ms:.0f} ms  ·  alert {res.alert_latency_ms:.0f} ms"
                                   f"  ·  {rep.detector}" + (f"  ·  ALERT FAULT: {err}" if err else ""))
        self.update_lock_ui()
        self._update_stats()
        self.focus_scan()

    def _on_error(self, exc: BaseException) -> None:
        self.busy = False
        self.update_lock_ui()
        if isinstance(exc, StationLocked):
            self.banner.show("LOCKED", theme.NG, str(exc))
            return
        self.station.alert.ng()  # a failed inspection is never silently OK
        self.banner.show("ERROR", theme.NG, str(exc)[:120])

    @staticmethod
    def _finding_rows(rep) -> list[tuple[str, str, str, str]]:
        """(color, place, text, confidence)"""
        rows = []
        if rep.error:
            rows.append((theme.NG, "Board", rep.error, ""))
        for p in rep.mismatches:
            if p.found == MISSING:
                text = f"missing - expected {p.expected}"
                conf = ""
            elif p.found == "uncertain":
                text = f"unsure - {p.reason}"
                conf = f"{p.confidence:.0%}"
            else:
                text = f"found {p.found}, expected {p.expected}"
                conf = f"{p.confidence:.0%}"
            rows.append((theme.NG, f"C{p.cable} · R{p.row}", text, conf))
        for b in rep.extras:
            rows.append((theme.WARN, "Extra", f"unexpected {b.label} at x={b.center[0]:.0f}, y={b.center[1]:.0f}",
                         f"{b.confidence:.0%}"))
        return rows

    def _set_findings(self, rows: list[tuple[str, str, str, str]]) -> None:
        for w in self.findings.winfo_children():
            w.destroy()
        if not rows:
            muted(self.findings, "No findings.", 13).pack(anchor="w", pady=4)
            return
        for color, place, text, conf in rows:
            r = ctk.CTkFrame(self.findings, fg_color=theme.SURFACE_2, corner_radius=10)
            r.pack(fill="x", pady=3)
            ctk.CTkFrame(r, width=6, height=1, fg_color=color, corner_radius=3).pack(side="left", fill="y", padx=(0, 10), pady=6)
            ctk.CTkLabel(r, text=place, font=theme.font(14, "bold"), text_color=theme.TEXT, width=70, anchor="w").pack(side="left")
            ctk.CTkLabel(r, text=text, font=theme.font(13), text_color=theme.TEXT, anchor="w", justify="left",
                         wraplength=250).pack(side="left", fill="x", expand=True, pady=8)
            if conf:
                muted(r, conf, 13).pack(side="right", padx=10)

    @property
    def finding_texts(self) -> list[str]:
        """Plain text of the findings list (used by tests)."""
        out = []
        for row in self.findings.winfo_children():
            if isinstance(row, ctk.CTkLabel):
                out.append(str(row.cget("text")))
                continue
            labels = [w.cget("text") for w in row.winfo_children() if isinstance(w, ctk.CTkLabel)]
            out.append(" ".join(str(t) for t in labels))
        return out

    # -- lock ------------------------------------------------------------------
    def update_lock_ui(self) -> None:
        st = self.station.lock.state
        key = self.cfg.trigger.key
        if st.locked:
            self.ack_btn.pack(fill="x", pady=(10, 0))
            self.pick_btn.configure(state="disabled")
            self.inspect_btn.configure(state="normal", text=f"RE-INSPECT {st.part_number}   [{key}]")
        else:
            self.ack_btn.pack_forget()
            self.pick_btn.configure(state="normal")
            self.inspect_btn.configure(state="normal" if not self.busy else "disabled", text=f"INSPECT   [{key}]")
        self.app.set_navigation_locked(st.locked)

    def acknowledge(self) -> None:
        pin = PinDialog.ask(self, "Supervisor PIN")
        if pin is None:
            return
        if self.station.acknowledge(pin, self.operator.get().strip()):
            self.banner.show("RELEASED", theme.WARN, "Station unlocked by supervisor")
        else:
            self.banner.show("WRONG PIN", theme.NG, "Station remains locked")
        self.update_lock_ui()

    def _update_stats(self) -> None:
        try:
            rep = self.station.store.daily_report(date.today())
        except Exception:
            return
        self.stats.configure(text=f"Today: {rep.total} inspected  ·  {rep.ok} OK  ·  {rep.ng} NG  ({rep.ng_rate:.1%})")

    def stop(self) -> None:
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
            self._preview_job = None
