"""Model Setup (login required): everything about creating or changing a part's model.

a. select the part number (pick one, or type / scan a new one)
b. add images, c. label the objects       -> "Images & labels"
d. train, e. put a model into use          -> "Train & models"
f. set the master from a known-good board  -> "Master"
g. station settings                        -> "Settings"
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ...vision.parts import Part, check_code
from .. import style
from ..widgets import ask_string, ask_yes_no, show_error


class ModelSetupPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.station = app.station
        self.cfg = app.cfg
        self.dispatcher = app.dispatcher
        self.part: Part | None = None
        self.windows: list[tk.Toplevel] = []  # editors and dialogs closed when the session ends

        # -- a. part number ------------------------------------------------------
        bar = ttk.Frame(self, style="Card.TFrame", padding=10)
        bar.pack(fill="x")
        ttk.Label(bar, text="Part number", style="Card.TLabel", font=style.font(11, "bold")).pack(side="left")
        self.code = ttk.Combobox(bar, font=style.font(13), width=22)
        self.code.pack(side="left", padx=8)
        self.code.bind("<Return>", lambda _e: self.open_code(self.code.get()))
        self.code.bind("<<ComboboxSelected>>", lambda _e: self.open_code(self.code.get()))
        ttk.Button(bar, text="Open / create", style="Accent.TButton",
                   command=lambda: self.open_code(self.code.get())).pack(side="left")
        ttk.Button(bar, text="Description...", command=self.edit_description).pack(side="left", padx=(8, 0))
        self.delete_btn = ttk.Button(bar, text="Delete part", command=self.delete_part)
        self.delete_btn.pack(side="right")
        self.info = ttk.Label(self, text="", justify="left", font=style.font(10))
        self.info.pack(fill="x", pady=(6, 6))

        # -- b-g ---------------------------------------------------------------------
        from .settings import SettingsPanel
        from .setup_images import ImagesPanel
        from .setup_master import MasterPanel
        from .setup_train import TrainPanel

        self.sections = ttk.Notebook(self)
        self.sections.pack(fill="both", expand=True)
        self.images = ImagesPanel(self.sections, self)
        self.train = TrainPanel(self.sections, self)
        self.master_panel = MasterPanel(self.sections, self)
        self.settings = SettingsPanel(self.sections, self)
        for panel, title in ((self.images, "  Images & labels  "), (self.train, "  Train & models  "),
                             (self.master_panel, "  Master  "), (self.settings, "  Settings  ")):
            self.sections.add(panel, text=title)
        self.panels = (self.images, self.train, self.master_panel)
        self.sections.bind("<<NotebookTabChanged>>", lambda _e: self._section_changed())
        self.refresh_codes()
        self.set_part(None)

    # -- session -------------------------------------------------------------------
    @property
    def user(self) -> str:
        return self.app.session_user or ""

    def on_show(self) -> None:
        self.refresh_codes()
        if self.part is not None and self.part.code not in self.station.parts:
            self.set_part(None)
        self.refresh()
        self.code.focus_set()

    def on_logout(self) -> None:
        for w in list(self.windows):
            try:
                w.destroy()
            except tk.TclError:
                pass
        self.windows.clear()

    def track(self, window: tk.Toplevel) -> tk.Toplevel:
        self.windows = [w for w in self.windows if w.winfo_exists()] + [window]
        return window

    def log(self, kind: str, detail: str = "") -> None:
        """Every change of a part's model or master is logged with the part number."""
        self.station.log_model_change(kind, self.user, self.part.code if self.part else "", detail)

    # -- a. part number ------------------------------------------------------------------
    def refresh_codes(self) -> None:
        self.code.configure(values=self.station.parts.codes())

    def open_code(self, text: str) -> Part | None:
        text = text.strip()
        if not text:
            return None
        if self.train.training:
            show_error(self, "Training is running. Wait for it or cancel it before changing the part.")
            self.code.set(self.part.code if self.part else "")
            return None
        part = self.station.parts.get(text)
        if part is None:
            try:
                code = check_code(text)
            except ValueError as exc:
                show_error(self, str(exc))
                return None
            if not ask_yes_no(self, f"{code} is a new part number. Create it?", "New part number"):
                self.code.set(self.part.code if self.part else "")
                return None
            desc = ask_string(self, "New part number", f"Description of {code} (optional):") or ""
            part = self.station.parts.create(code, desc)
            self.set_part(part)
            self.log("part_created", desc)
            self.refresh_codes()
            self.app.parts_changed(code)
        else:
            self.set_part(part)
        return part

    def set_part(self, part: Part | None) -> None:
        self.part = part
        self.code.set(part.code if part else "")
        for panel in self.panels:
            panel.set_part(part)
        self.delete_btn.configure(state="normal" if part else "disabled")
        self.refresh()

    def refresh(self) -> None:
        """The part's status line: model, accuracy, when trained, labelled images, master, readiness."""
        part = self.part
        if part is None:
            self.info.configure(text="Choose a part number, or type / scan a new one and press Enter.",
                                foreground=style.MUTED)
            return
        part.store.reload()
        active = part.active()
        model = f"Model {active.summary()}" if active else "No model in use yet"
        master = (f"Master {part.master.cables} cables x {part.master.rows} rows" if part.master
                  else "No master yet")
        stats = part.store.stats()
        problems = part.setup_problems(self.cfg.detector.backend)
        state = "READY for inspection" if not problems else "NOT SET UP: " + ", ".join(problems)
        desc = f"  ·  {part.description}" if part.description else ""
        self.info.configure(
            text=f"{part.code}{desc}\n{model}  ·  {stats['marked']} of {stats['images']} images labelled  ·  "
                 f"{len(part.labels)} labels  ·  {master}  ·  {state}",
            foreground=style.OK if not problems else style.TEXT)

    def changed(self) -> None:
        """The part's model, master or labels changed: update this page and the Inspect tab."""
        self.refresh()
        for panel in self.panels:
            panel.refresh()
        self.app.parts_changed(self.part.code if self.part else None)

    def edit_description(self) -> None:
        if self.part is None:
            return
        desc = ask_string(self, "Description", f"Description of {self.part.code}:", self.part.description)
        if desc is None:
            return
        self.part.description = desc.strip()
        if self.part.master is not None:
            self.part.master.description = self.part.description
        self.part.save()
        self.changed()

    def delete_part(self) -> None:
        part = self.part
        if part is None:
            return
        if self.train.training:
            show_error(self, "Training is running for this part.")
            return
        if not ask_yes_no(self, f"Delete part number {part.code} with all its images, labels, models and master?"):
            return
        self.log("part_deleted", f"{len(part.store)} images, {len(part.models())} models")
        self.station.parts.delete(part.code)
        self.set_part(None)
        self.refresh_codes()
        self.app.parts_changed(part.code)

    def _section_changed(self) -> None:
        widget = self.nametowidget(self.sections.select())
        if hasattr(widget, "on_show"):
            widget.on_show()

    def on_config_changed(self) -> None:
        self.refresh()

    def stop(self) -> None:
        self.train.stop()
