"""Part numbers screen: create, edit and delete part numbers and their master patterns."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..config import save_config
from ..models import EXPECTED_LABELS, PartNumber
from ..pipeline import Station
from .widgets import confirm, error_box


class PartsScreen(ttk.Frame):
    def __init__(self, master, station: Station, on_config_changed: Callable[[], None]):
        super().__init__(master)
        self.station = station
        self.cfg = station.cfg
        self.on_config_changed = on_config_changed

        cols = ("code", "pattern", "description")
        self.table = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        for c, text, w in zip(cols, ("Part number", "Master pattern (row 1 → n)", "Description"), (140, 380, 300)):
            self.table.heading(c, text=text)
            self.table.column(c, width=w, anchor="w")
        self.table.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        editor = ttk.LabelFrame(self, text="Edit part number", padding=10)
        editor.pack(side="right", fill="y")  # packed before the table so it keeps its full width
        self.table.pack(side="left", fill="both", expand=True, padx=(0, 8))
        editor.columnconfigure(1, weight=1)
        self.code_var, self.desc_var = tk.StringVar(), tk.StringVar()
        ttk.Label(editor, text="Part number").grid(row=0, column=0, sticky="w", pady=4)
        self.code = ttk.Entry(editor, textvariable=self.code_var, width=24)
        self.code.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(editor, text="Description").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(editor, textvariable=self.desc_var).grid(row=1, column=1, sticky="ew", pady=4)
        self.row_vars: list[tk.StringVar] = []
        for r in range(self.cfg.station.rows):
            var = tk.StringVar(value=EXPECTED_LABELS[0])
            ttk.Label(editor, text=f"Row {r + 1}").grid(row=2 + r, column=0, sticky="w", pady=4)
            ttk.Combobox(editor, textvariable=var, values=list(EXPECTED_LABELS), state="readonly").grid(
                row=2 + r, column=1, sticky="ew", pady=4
            )
            self.row_vars.append(var)
        n = 2 + self.cfg.station.rows
        ttk.Label(editor, text="fork = either orientation; fork_left / fork_right = orientation is checked",
                  style="Muted.TLabel", wraplength=360).grid(row=n, column=0, columnspan=2, sticky="w", pady=6)
        btns = ttk.Frame(editor)
        btns.grid(row=n + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="New", command=self.new).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btns, text="Save", command=self.save).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btns, text="Delete", style="Danger.TButton", command=self.delete).pack(side="left", fill="x", expand=True)
        self.refresh()

    def refresh(self) -> None:
        self.table.delete(*self.table.get_children())
        for code in sorted(self.cfg.parts):
            pn = self.cfg.parts[code]
            self.table.insert("", "end", iid=code, values=(code, ", ".join(pn.pattern), pn.description))

    def _on_select(self) -> None:
        sel = self.table.selection()
        pn = self.cfg.parts.get(sel[0]) if sel else None
        if pn is None:
            return
        self.code_var.set(pn.code)
        self.desc_var.set(pn.description)
        for var, label in zip(self.row_vars, pn.pattern):
            var.set(label)

    def new(self) -> None:
        self.table.selection_remove(*self.table.selection())
        self.code_var.set("")
        self.desc_var.set("")
        for var in self.row_vars:
            var.set(EXPECTED_LABELS[0])
        self.code.focus_set()

    def save(self) -> None:
        code = self.code_var.get().strip()
        pn = PartNumber(code, [v.get() for v in self.row_vars], self.desc_var.get().strip())
        try:
            pn.validate(self.cfg.station.rows)
        except ValueError as exc:
            error_box(self, str(exc))
            return
        old = self.cfg.parts.get(code)
        if old is not None and old.pattern != pn.pattern and not confirm(self, f"Change the master pattern of {code}?"):
            return
        self.cfg.parts[code] = pn
        save_config(self.cfg)
        self.station.store.log_event("part_saved", "", f"{code}: {old.pattern if old else 'new'} -> {pn.pattern}")
        self.refresh()
        self.table.selection_set(code)
        self.on_config_changed()

    def delete(self) -> None:
        code = self.code_var.get().strip()
        if code not in self.cfg.parts:
            return
        if self.station.lock.locked and self.station.lock.state.part_number == code:
            error_box(self, "This part is locked after an NG; resolve that first.")
            return
        if not confirm(self, f"Delete part number {code}? History is kept."):
            return
        del self.cfg.parts[code]
        save_config(self.cfg)
        self.station.store.log_event("part_deleted", "", code)
        self.new()
        self.refresh()
        self.on_config_changed()
