"""Part numbers page: search, edit and delete parts; preview each part's master layout."""

from __future__ import annotations

import customtkinter as ctk
import cv2

from ...config import save_config
from ...models import PartNumber
from ...pipeline import Station
from .. import theme
from ..widgets import Card, Dispatcher, ImageView, Table, ask_yes_no, button, muted, show_error


class PartsPage(ctk.CTkFrame):
    title = "Part numbers"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.code: str | None = None
        self.row_menus: list[ctk.CTkOptionMenu] = []

        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        left = Card(self, title="All parts", subtitle="Tip: create parts from a marked good board in Training data.")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        lb = left.body
        lb.grid_columnconfigure(0, weight=1)
        lb.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(lb, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.search = ctk.CTkEntry(top, placeholder_text="Search part number or description", height=38)
        self.search.pack(side="left", fill="x", expand=True)
        self.search.bind("<KeyRelease>", lambda _e: self.refresh_list())
        button(top, "+ New", self.new, "primary", width=90).pack(side="left", padx=(8, 0))
        self.table = Table(lb, [("code", "Part", 110), ("desc", "Description", 200), ("size", "Layout", 70),
                                ("master", "Master image", 100)])
        self.table.grid(row=1, column=0, sticky="nsew")
        self.table.on_select(self.load)

        right = Card(self, title="Edit part")
        right.grid(row=0, column=1, sticky="nsew")
        rb = right.body
        rb.grid_columnconfigure(1, weight=1)
        rb.grid_rowconfigure(5, weight=1)
        muted(rb, "Part number", 13).grid(row=0, column=0, sticky="w", pady=4)
        self.code_entry = ctk.CTkEntry(rb, height=38, font=theme.font(15, "bold"))
        self.code_entry.grid(row=0, column=1, sticky="ew", pady=4)
        muted(rb, "Description", 13).grid(row=1, column=0, sticky="w", pady=4)
        self.desc = ctk.CTkEntry(rb, height=38)
        self.desc.grid(row=1, column=1, sticky="ew", pady=4)
        muted(rb, "Cables on the board", 13).grid(row=2, column=0, sticky="w", pady=4, padx=(0, 12))
        self.cables = ctk.CTkEntry(rb, height=38, width=80)
        self.cables.grid(row=2, column=1, sticky="w", pady=4)
        muted(rb, "Clip per row\n(top to bottom)", 13).grid(row=3, column=0, sticky="nw", pady=4)
        rows_box = ctk.CTkFrame(rb, fg_color="transparent")
        rows_box.grid(row=3, column=1, sticky="ew", pady=4)
        self.rows_frame = ctk.CTkFrame(rows_box, fg_color="transparent")
        self.rows_frame.pack(fill="x")
        rb_btns = ctk.CTkFrame(rows_box, fg_color="transparent")
        rb_btns.pack(anchor="w", pady=(4, 0))
        button(rb_btns, "+ row", self.add_row, "secondary", width=70, height=32).pack(side="left")
        button(rb_btns, "− row", self.remove_row, "secondary", width=70, height=32).pack(side="left", padx=6)
        muted(rb_btns, "'fork' accepts either direction; fork_left / fork_right check it.", 11).pack(side="left", padx=6)
        self.master_info = muted(rb, "", 12, wraplength=520)
        self.master_info.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4))
        self.preview = ImageView(rb, placeholder="No master image - the grid is found automatically at inspection.\n"
                                                 "For best results create the part from a marked good board.")
        self.preview.grid(row=5, column=0, columnspan=2, sticky="nsew")
        actions = ctk.CTkFrame(rb, fg_color="transparent")
        actions.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        button(actions, "Save", self.save, "primary", width=120).pack(side="left")
        button(actions, "Delete", self.delete, "danger", width=120).pack(side="right")
        self.new()

    # -- list ------------------------------------------------------------------
    def refresh(self) -> None:
        self.refresh_list()
        if self.code and self.code in self.cfg.parts:
            self.load(self.code)

    def refresh_list(self) -> None:
        q = self.search.get().strip().lower()
        store = self.station.annotations
        rows = []
        for code in sorted(self.cfg.parts):
            p = self.cfg.parts[code]
            if q and q not in code.lower() and q not in p.description.lower():
                continue
            master = ""
            if p.master_image:
                master = store.get(p.master_image).source_name if p.master_image in store else "(deleted)"
            rows.append((code, (code, p.description, f"{p.cables}x{p.rows}", master or "-"), ()))
        self.table.set_rows(rows)

    # -- editor ----------------------------------------------------------------
    def _set_rows(self, pattern: list[str]) -> None:
        for w in self.rows_frame.winfo_children():
            w.destroy()
        self.row_menus = []
        for label in pattern:
            self._append_row(label)

    def _append_row(self, label: str) -> None:
        values = self.cfg.taxonomy.expected_labels()
        r = ctk.CTkFrame(self.rows_frame, fg_color="transparent")
        r.pack(fill="x", pady=2)
        muted(r, f"R{len(self.row_menus) + 1}", 13).pack(side="left", padx=(0, 8))
        m = ctk.CTkOptionMenu(r, values=values, height=34, width=200)
        m.set(label if label in values else values[0])
        m.pack(side="left")
        self.row_menus.append(m)

    def add_row(self) -> None:
        self._append_row(self.cfg.taxonomy.expected_labels()[0])

    def remove_row(self) -> None:
        if len(self.row_menus) > 1:
            self.row_menus.pop().master.destroy()

    def new(self) -> None:
        self.code = None
        self.code_entry.delete(0, "end")
        self.desc.delete(0, "end")
        self.cables.delete(0, "end")
        self.cables.insert(0, "4")
        self._set_rows([self.cfg.taxonomy.expected_labels()[0]] * 4)
        self.master_info.configure(text="New part. Or: Training data > mark a good board > Create part from this image.")
        self.preview.clear("No master image yet")
        self.code_entry.focus_set()

    def load(self, code: str | None) -> None:
        p = self.cfg.parts.get(code or "")
        if p is None:
            return
        self.code = code
        for entry, value in ((self.code_entry, p.code), (self.desc, p.description), (self.cables, str(p.cables))):
            entry.delete(0, "end")
            entry.insert(0, value)
        self._set_rows(p.pattern)
        store = self.station.annotations
        if p.layout is not None and p.master_image in store:
            self.master_info.configure(text=f"Master: {store.get(p.master_image).source_name}  ·  "
                                            f"{p.layout.cables} x {p.layout.rows} positions learned from the image")
            self.preview.set_image(self._layout_preview(p))
        elif p.layout is not None:
            self.master_info.configure(text="Master layout stored (its image was deleted from Training data).")
            self.preview.clear("Master image not available")
        else:
            self.master_info.configure(text="No master layout: clips are grouped into cables and rows automatically.")
            self.preview.clear("No master image")

    def _layout_preview(self, p: PartNumber):
        img = self.station.annotations.load_image(p.master_image).copy()
        cw, ch = p.layout.clip_size
        for (c, r), (x, y) in p.layout.positions.items():
            color = theme.hex_to_bgr(theme.class_color(self.cfg.taxonomy.classes, p.pattern[r - 1])
                                     if p.pattern[r - 1] in self.cfg.taxonomy.classes else "#64748b")
            cv2.rectangle(img, (int(x - cw / 2), int(y - ch / 2)), (int(x + cw / 2), int(y + ch / 2)), color, 3)
            cv2.putText(img, f"C{c}R{r}", (int(x - cw / 2), int(y - ch / 2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)
        return img

    def save(self) -> None:
        code = self.code_entry.get().strip()
        try:
            cables = int(self.cables.get())
        except ValueError:
            show_error(self, "Cables must be a whole number.")
            return
        old = self.cfg.parts.get(self.code or code)
        part = PartNumber(code, [m.get() for m in self.row_menus], self.desc.get().strip(), cables)
        if old is not None and old.layout is not None:
            if (old.cables, old.rows) == (cables, part.rows):
                part.layout, part.master_image = old.layout, old.master_image
            elif not ask_yes_no(self, "The number of cables or rows changed, so the master layout no longer fits and "
                                      "will be removed. Create a new master from a marked image afterwards.\n\nContinue?"):
                return
        try:
            part.validate(self.cfg.taxonomy)
        except ValueError as exc:
            show_error(self, str(exc))
            return
        if self.code and code != self.code:  # renamed
            if code in self.cfg.parts and not ask_yes_no(self, f"{code} exists already. Replace it?"):
                return
            self.cfg.parts.pop(self.code, None)
        elif old is not None and old.pattern != part.pattern and not ask_yes_no(self, f"Change the master pattern of {code}?"):
            return
        self.cfg.parts[code] = part
        save_config(self.cfg)
        self.station.store.log_event("part_saved", "", f"{code}: {old.pattern if old else 'new'} -> {part.pattern}")
        self.code = code
        self.refresh_list()
        self.table.select(code)
        self.app.on_config_changed()

    def delete(self) -> None:
        code = self.code
        if not code or code not in self.cfg.parts:
            return
        if self.station.lock.locked and self.station.lock.state.part_number == code:
            show_error(self, "This part is locked after an NG; resolve that first.")
            return
        if not ask_yes_no(self, f"Delete part number {code}? Its inspection history is kept.", yes="Delete", danger=True):
            return
        del self.cfg.parts[code]
        save_config(self.cfg)
        self.station.store.log_event("part_deleted", "", code)
        self.new()
        self.refresh_list()
        self.app.on_config_changed()
