"""Inspect: scan/choose the part number, press INSPECT (or the foot pedal), read OK / NG.

No login. A part number is inspected with its own active model and master; a part
without them shows NOT SET UP and cannot be inspected.
"""

from __future__ import annotations

import tkinter as tk
from datetime import date
from tkinter import filedialog, ttk

from ...core.models import MISSING, UNCERTAIN
from ...station import StationLocked, StationResult
from ...station.barcode import parse_scan
from ...vision.camera import FileSource, read_image
from .. import style
from ..widgets import Banner, ImageView, Table, ask_string, show_error


class InspectPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.station = app.station
        self.cfg = app.cfg
        self.dispatcher = app.dispatcher
        self.busy = False
        self._file_image = None  # an image file loaded for the next inspection (camera mode)
        self._showing_result = False
        self._last_annotated = None
        self.problems: list[str] = ["no part selected"]  # why the selected part cannot be inspected
        self._loading = ""  # part whose model is being loaded
        self._selected = ""  # the part chosen last (restored after an operator badge scan)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.view = ImageView(self, placeholder="Camera image appears here")
        self.view.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        side = ttk.Frame(self, width=400)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(6, weight=1)

        form = ttk.Frame(side)
        form.grid(row=0, column=0, sticky="ew")
        form.grid_columnconfigure(1, weight=1)
        ttk.Label(form, text="Part number").grid(row=0, column=0, sticky="w", pady=3)
        self.part = ttk.Combobox(form, font=style.font(13))
        self.part.grid(row=0, column=1, sticky="ew", pady=3)
        self.part.bind("<Return>", lambda _e: self.on_scan())
        self.part.bind("<<ComboboxSelected>>", lambda _e: self._change_part(self.part.get()))
        ttk.Label(form, text="Operator").grid(row=1, column=0, sticky="w", pady=3, padx=(0, 8))
        self.operator = ttk.Entry(form, font=style.font(12))
        self.operator.grid(row=1, column=1, sticky="ew", pady=3)
        self.part_info = ttk.Label(side, text="", style="Muted.TLabel", wraplength=390, justify="left")
        self.part_info.grid(row=1, column=0, sticky="w", pady=(2, 8))

        self.inspect_btn = ttk.Button(side, text="INSPECT", style="Big.TButton", command=self.inspect)
        self.inspect_btn.grid(row=2, column=0, sticky="ew")
        row = ttk.Frame(side)
        row.grid(row=3, column=0, sticky="ew", pady=(6, 8))
        ttk.Button(row, text="Inspect an image file...", command=self.load_image).pack(side="left")
        ttk.Button(row, text="Live view", command=self.live_view).pack(side="left", padx=6)

        self.banner = Banner(side)
        self.banner.grid(row=4, column=0, sticky="ew")
        self.ack_btn = ttk.Button(side, text="Supervisor unlock (PIN)", style="Danger.TButton", command=self.acknowledge)
        self.timing = ttk.Label(side, text="", style="Muted.TLabel")
        self.timing.grid(row=5, column=0, sticky="w", pady=(4, 4))

        self.findings = Table(side, [("pos", "Position", 70), ("expected", "Expected", 110),
                                     ("found", "Found", 110), ("conf", "Conf.", 50)], height=8)
        self.findings.grid(row=6, column=0, sticky="nsew")
        self.stats = ttk.Label(side, text="", style="Muted.TLabel")
        self.stats.grid(row=8, column=0, sticky="w", pady=(6, 0))

        self.station.lock.subscribe(lambda _s: self.dispatcher.call(self.update_lock_ui))
        self._preview_ms = max(40, int(1000 / max(1, self.cfg.ui.preview_fps)))
        self._preview_job = self.after(self._preview_ms, self._preview_tick)
        self.on_config_changed()
        self.update_lock_ui()

    # -- part number -------------------------------------------------------------
    @property
    def part_code(self) -> str:
        return self.part.get().strip()

    @property
    def parts(self):
        return self.station.parts

    @property
    def ready(self) -> bool:
        return not self.problems

    def on_config_changed(self) -> None:
        self.on_parts_changed()
        self._update_stats()

    def on_parts_changed(self, code: str | None = None) -> None:
        """Part numbers, models or masters changed in Model Setup: refresh, reload the current part."""
        codes = self.parts.codes()
        self.part.configure(values=codes)
        current = self.part_code
        if current not in codes:
            current = codes[0] if codes else ""
        if code is None or code == current or current != self.part_code:
            self.set_part(current)

    def set_part(self, code: str) -> None:
        """Select a part and load its model in the background (READY / NOT SET UP)."""
        self.part.set(code)
        self._selected = code
        part = self.parts.get(code) if code else None
        if part is None:
            self.problems = ["unknown part number" if code else "no part selected"]
            self._loading = ""
            self.part_info.configure(text="Scan a barcode or choose a part number." if not code else
                                     f"{code} is not a known part number.")
            return
        lines = [part.description or "No description"]
        active = part.active()
        if active is not None:
            lines.append(f"Model {active.summary()}")
        if part.master is not None:
            pn = part.master
            lines.append(f"Master {pn.cables} cables x {pn.rows} rows")
            lines += [f"Row {r}: " + ", ".join(row) for r, row in enumerate(pn.pattern, 1)]
        self.part_info.configure(text="\n".join(lines))
        self.problems = ["loading"]
        self._loading = code
        if not self.busy and not self._showing_result:
            self.banner.show("LOADING", style.IDLE, f"Loading the model of {code}")
        self.dispatcher.run_task(lambda: self.station.prepare(code), lambda p: self._prepared(code, p),
                                 lambda e: self._prepared(code, [str(e)]))

    def _prepared(self, code: str, problems: list[str]) -> None:
        if code != self.part_code:
            return  # another part was chosen meanwhile
        self._loading = ""
        self.problems = list(problems)
        if self.busy or self._showing_result:
            return
        if problems:
            self.banner.show("NOT SET UP", style.WARN, f"{code}: " + "; ".join(problems) + "\n(see Model Setup)")
        else:
            self.banner.show("READY", style.IDLE, f"{code} selected")

    def _change_part(self, code: str) -> None:
        st = self.station.lock.state
        if st.locked and code != st.part_number:
            self.part.set(st.part_number)
            self.banner.show("LOCKED", style.NG, "Resolve the NG before changing the part")
            return
        self._showing_result = False
        self.set_part(code)

    def on_scan(self) -> None:
        """Enter in the part box: a barcode scanner types the code followed by Enter."""
        text = self.part.get()
        s = parse_scan(text, self.cfg.barcode, set(self.parts.codes()))
        st = self.station.lock.state
        if s.kind == "operator":
            self.operator.delete(0, "end")
            self.operator.insert(0, s.value)
            self.part.set(st.part_number if st.locked else self._selected)
        elif s.kind == "part":
            self._change_part(s.value)
        else:
            self.part.set(st.part_number if st.locked else self._selected)
            self.banner.show("UNKNOWN", style.WARN, f"Not a known part number: {text.strip()[:30]}")

    def focus_scan(self) -> None:
        self.part.focus_set()

    # -- camera preview ------------------------------------------------------------
    def _preview_tick(self) -> None:
        try:
            if not self._showing_result and self.winfo_ismapped():
                frame = self._file_image if self._file_image is not None else self.station.source.latest()
                if frame is not None:
                    self.view.set_image(frame)
                err = getattr(self.station.source, "error", "")
                if err and not self.busy:
                    self.timing.configure(text=f"Camera: {err}")
        except tk.TclError:
            return
        self._preview_job = self.after(self._preview_ms, self._preview_tick)

    def live_view(self) -> None:
        self._showing_result = False
        self._file_image = None

    def load_image(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Inspect an image file",
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
        self.view.set_image(img)
        self.inspect()

    # -- inspection ------------------------------------------------------------------
    def inspect(self) -> None:
        if self.busy:
            return
        part = self.part_code
        if part not in self.parts:
            self.banner.show("NO PART", style.WARN, "Scan or choose a part number first")
            return
        ok, why = self.station.can_inspect(part)
        if not ok:
            self.banner.show("LOCKED", style.NG, why)
            return
        if self._loading == part:
            self.banner.show("LOADING", style.IDLE, f"The model of {part} is still loading")
            return
        if self.problems:
            self.banner.show("NOT SET UP", style.WARN, f"{part}: " + "; ".join(self.problems) + "\n(see Model Setup)")
            return
        self.busy = True
        self.inspect_btn.configure(state="disabled", text="Inspecting ...")
        self.banner.show("...", style.IDLE, f"Inspecting {part}")
        image, operator = self._file_image, self.operator.get().strip()
        self.dispatcher.run_task(lambda: self.station.inspect(part, operator, image=image), self._on_result,
                                 self._on_error)

    def _on_result(self, res: StationResult) -> None:
        self.busy = False
        self._file_image = None
        rep = res.report
        self._last_annotated = res.annotated
        self._showing_result = True
        self.view.set_image(res.annotated)
        if rep.ok:
            self.banner.show("OK", style.OK, f"{rep.part_number}: all {len(rep.positions)} positions correct")
        else:
            n = len(rep.mismatches) + len(rep.extras)
            self.banner.show("NG", style.NG, rep.error or f"{n} wrong position(s) - see the list")
        self._show_findings(rep)
        err = self.station.alert.last_error
        self.timing.configure(text=f"#{res.inspection_id}  ·  {rep.duration_ms:.0f} ms  ·  {rep.detector}"
                                   + (f"  ·  ALERT FAULT: {err}" if err else ""))
        self.update_lock_ui()
        self._update_stats()
        self.focus_scan()

    def _on_error(self, exc: BaseException) -> None:
        self.busy = False
        self.update_lock_ui()
        if isinstance(exc, StationLocked):
            self.banner.show("LOCKED", style.NG, str(exc))
            return
        self.station.alert.ng()  # a failed inspection is never silently OK
        self.banner.show("ERROR", style.NG, str(exc)[:160])

    def _show_findings(self, rep) -> None:
        rows = []
        if rep.error:
            rows.append(("err", ("Board", "-", rep.error[:60], ""), ("ng",)))
        for p in rep.positions:
            if p.found == MISSING:
                found, conf = "missing", ""
            elif p.found == UNCERTAIN:
                found, conf = "unsure", f"{p.confidence:.0%}"
            else:
                found, conf = p.found, f"{p.confidence:.0%}"
            rows.append((f"{p.cable}-{p.row}", (f"C{p.cable} R{p.row}", p.expected, found, conf),
                         ("ok",) if p.ok else ("ng",)))
        rows.sort(key=lambda r: "ok" in r[2])  # problems first
        for i, b in enumerate(rep.extras):
            rows.insert(0, (f"extra{i}", ("Extra", "-", f"unexpected {b.label}", f"{b.confidence:.0%}"), ("ng",)))
        self.findings.set_rows(rows, keep_selection=False)

    @property
    def finding_texts(self) -> list[str]:
        """Plain text of the findings table (used by tests)."""
        t = self.findings.tree
        return [" ".join(str(v) for v in t.item(i, "values")) for i in t.get_children()]

    # -- NG lock -------------------------------------------------------------------------
    def update_lock_ui(self) -> None:
        st = self.station.lock.state
        key = self.cfg.trigger.key
        if st.locked:
            self.ack_btn.grid(row=7, column=0, sticky="ew", pady=(8, 0))
            self.inspect_btn.configure(state="normal", text=f"RE-INSPECT {st.part_number}   [{key}]")
        else:
            self.ack_btn.grid_remove()
            self.inspect_btn.configure(state="disabled" if self.busy else "normal", text=f"INSPECT   [{key}]")
        self.app.set_locked(st.locked)

    def acknowledge(self) -> None:
        pin = ask_string(self, "Supervisor unlock", "Supervisor PIN:", secret=True)
        if pin is None:
            return
        if self.station.acknowledge(pin, self.operator.get().strip()):
            self.banner.show("RELEASED", style.WARN, "Station unlocked by supervisor")
        else:
            self.banner.show("WRONG PIN", style.NG, "Station remains locked")
        self.update_lock_ui()

    def _update_stats(self) -> None:
        try:
            rep = self.station.store.daily_report(date.today())
        except Exception:
            return
        self.stats.configure(text=f"Today: {rep.total} inspected  ·  {rep.ok} OK  ·  {rep.ng} NG  ({rep.ng_rate:.1%})")

    def on_show(self) -> None:
        self.on_parts_changed()
        self.focus_scan()

    def stop(self) -> None:
        if self._preview_job is not None:
            try:
                self.after_cancel(self._preview_job)
            except tk.TclError:
                pass
            self._preview_job = None
