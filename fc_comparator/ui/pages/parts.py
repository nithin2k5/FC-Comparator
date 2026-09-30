"""Parts: each part number's master - the clip expected at every position - made from a photo of a good board.

New part: choose a photo (or capture one), the detector finds the clips, check/fix the
boxes, enter the part number, Save. The pattern and the clip positions are read from
the boxes; the photo is kept in the training data set as the part's master image.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from ...config import save_config
from ...core.layout import LayoutError, master_from_boxes
from ...core.models import Box, PartNumber
from ...core.parts import part_from_boxes
from ...vision.camera import read_image
from ...vision.drawing import draw_boxes
from .. import style
from ..box_editor import AnnotationCanvas
from ..widgets import ImageView, Table, ask_yes_no, card, center_window, show_error, show_info


def pattern_rows(pattern: list[list[str]]) -> list[tuple[str, list[str], tuple]]:
    return [(str(r), [f"Row {r}"] + list(row), ()) for r, row in enumerate(pattern, 1)]


class PartsPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.station = app.station
        self.cfg = app.cfg
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = card(self, "Part numbers", "Each part's master says which clip belongs at every position.")
        left.panel.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self.table = Table(left, [("code", "Part number", 120), ("desc", "Description", 150), ("size", "Board", 70)],
                           height=18, on_select=lambda _i: self.show_selected())
        self.table.pack(fill="both", expand=True)
        buttons = ttk.Frame(left, style="Card.TFrame")
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="New from photo...", style="Accent.TButton", command=self.new_from_file).pack(fill="x")
        ttk.Button(buttons, text="New from camera", command=self.new_from_camera).pack(fill="x", pady=4)
        row = ttk.Frame(buttons, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Edit", command=self.edit_selected).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Delete", command=self.delete_selected).pack(side="left", expand=True, fill="x", padx=(4, 0))

        right = card(self, "Master")
        right.panel.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.info = ttk.Label(right, text="", style="CardMuted.TLabel")
        self.info.grid(row=0, column=0, sticky="w")
        self.view = ImageView(right, placeholder="Select a part")
        self.view.grid(row=1, column=0, sticky="nsew", pady=6)
        self.pattern = Table(right, [("row", "", 60)], height=5)
        self.pattern.grid(row=2, column=0, sticky="ew")
        self.refresh()

    # -- list ------------------------------------------------------------------------
    def refresh(self, select: str | None = None) -> None:
        rows = [(code, (code, pn.description, f"{pn.cables} x {pn.rows}"), ())
                for code, pn in sorted(self.cfg.parts.items())]
        self.table.set_rows(rows)
        if select:
            self.table.select(select)
        elif not self.table.selected() and rows:
            self.table.select(rows[0][0])
        self.show_selected()

    def on_show(self) -> None:
        self.refresh()

    def show_selected(self) -> None:
        code = self.table.selected()
        pn = self.cfg.parts.get(code or "")
        if pn is None:
            self.info.configure(text="No part numbers yet - make one from a photo of a good board.")
            self.view.clear("Select a part")
            self.pattern.set_rows([])
            return
        cols = ["row"] + [f"c{c}" for c in range(1, pn.cables + 1)]
        t = self.pattern.tree
        t.configure(columns=cols)
        t.heading("row", text="")
        t.column("row", width=60, stretch=False)
        for c in range(1, pn.cables + 1):
            t.heading(f"c{c}", text=f"Cable {c}")
            t.column(f"c{c}", width=120)
        self.pattern.set_rows(pattern_rows(pn.pattern), keep_selection=False)
        self.info.configure(text=f"{pn.code}  ·  {pn.description or 'no description'}  ·  "
                                 f"{pn.cables} cables x {pn.rows} rows"
                                 + ("" if pn.layout else "  ·  no master photo (positions found per image)"))
        rec_img = self._master_image(pn)
        if rec_img is None:
            self.view.clear("No master photo for this part")
        else:
            img, boxes = rec_img
            self.view.set_image(draw_boxes(img, boxes))

    def _master_image(self, pn: PartNumber) -> tuple[np.ndarray, list[Box]] | None:
        store = self.station.dataset
        if not pn.master_image or pn.master_image not in store:
            return None
        try:
            return store.load_image(pn.master_image), store.get(pn.master_image).boxes
        except Exception:
            return None

    # -- create / edit / delete -------------------------------------------------------
    def new_from_file(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Photo of a good board",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            show_error(self, str(exc))
            return
        self.open_editor(img, None, None, source=path)

    def new_from_camera(self) -> None:
        try:
            img = self.station.source.capture(timeout=5)
        except Exception as exc:
            show_error(self, f"Camera: {exc}")
            return
        self.open_editor(img, None, None)

    def edit_selected(self) -> None:
        pn = self.cfg.parts.get(self.table.selected() or "")
        if pn is None:
            return
        rec = self._master_image(pn)
        if rec is None:
            show_error(self, "This part has no master photo. Make it again from a photo of a good board.")
            return
        self.open_editor(rec[0], rec[1], pn)

    def open_editor(self, image: np.ndarray, boxes: list[Box] | None, part: PartNumber | None,
                    source: str | None = None) -> "PartEditor":
        """``source``: the photo's file, so a photo already in the data set is not added twice."""
        return PartEditor(self, image, boxes, part, source)

    def delete_selected(self) -> None:
        code = self.table.selected()
        if not code or code not in self.cfg.parts:
            return
        if not ask_yes_no(self, f"Delete part number {code}?"):
            return
        del self.cfg.parts[code]
        save_config(self.cfg)
        self.station.store.log_event("part_deleted", self.app.user, code)
        self.refresh()
        self.app.on_config_changed()

    def save_part(self, part: PartNumber, image: np.ndarray, boxes: list[Box], replaced: str | None,
                  source: str | None = None) -> None:
        """Store the master photo (with its boxes) in the data set and the part in the config."""
        store = self.station.dataset
        old = self.cfg.parts.get(replaced or "")
        if old is not None and old.master_image in store:  # editing: same photo, corrected boxes
            image_id = old.master_image
        else:
            image_id, _new = store.add_image(source or image, part=part.code, good=True,
                                             name="" if source else f"master {part.code}")
        store.set_boxes(image_id, boxes)
        store.update(image_id, part=part.code, good=True)
        part.master_image = image_id
        if replaced and replaced != part.code:
            self.cfg.parts.pop(replaced, None)
        self.cfg.parts[part.code] = part
        save_config(self.cfg)
        self.station.store.log_event("part_saved", self.app.user, part.code)
        self.refresh(select=part.code)
        self.app.on_config_changed()


class PartEditor(tk.Toplevel):
    """Check the clips found on a good board, then save it as a part's master."""

    def __init__(self, page: PartsPage, image: np.ndarray, boxes: list[Box] | None, part: PartNumber | None,
                 source: str | None = None):
        super().__init__(page)
        self.page = page
        self.cfg = page.cfg
        self.image = image
        self.part = part
        self.source = source
        self.title("Edit part" if part else "New part")
        self.minsize(1000, 700)
        top = page.winfo_toplevel()
        center_window(self, page, min(1280, max(1000, top.winfo_width() - 80)),
                      min(860, max(700, top.winfo_height() - 80)))
        self.configure(bg=style.BG)
        self.transient(page.winfo_toplevel())

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Part number").pack(side="left")
        self.code = ttk.Entry(top, width=18, font=style.font(12))
        self.code.pack(side="left", padx=(6, 16))
        ttk.Label(top, text="Description").pack(side="left")
        self.desc = ttk.Entry(top, width=40, font=style.font(12))
        self.desc.pack(side="left", padx=6)
        if part:
            self.code.insert(0, part.code)
            self.desc.insert(0, part.description)
        ttk.Button(top, text="Cancel", command=self.destroy).pack(side="right")
        self.save_btn = ttk.Button(top, text="Save part", style="Accent.TButton", command=self.save)
        self.save_btn.pack(side="right", padx=6)

        tools = ttk.Frame(self, padding=(10, 0))
        tools.pack(fill="x")
        ttk.Label(tools, text="Clip type").pack(side="left")
        self.clip = ttk.Combobox(tools, values=self.cfg.taxonomy.classes, state="readonly", width=20)
        self.clip.pack(side="left", padx=6)
        self.clip.bind("<<ComboboxSelected>>", lambda _e: self.canvas.set_class(self.clip.get()))
        ttk.Button(tools, text="Delete box", command=lambda: self.canvas.delete_selected()).pack(side="left", padx=4)
        ttk.Button(tools, text="Undo", command=lambda: self.canvas.undo()).pack(side="left")
        ttk.Button(tools, text="Find clips again", command=self.detect).pack(side="left", padx=4)
        ttk.Label(tools, text="Drag on the image to add a box · click a box to select · wheel to zoom",
                  style="Muted.TLabel").pack(side="left", padx=12)

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.canvas = AnnotationCanvas(body, self.cfg.taxonomy.classes, on_change=lambda _b: self.update_pattern(),
                                       on_select=self._on_select)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        side = ttk.Frame(body, width=380)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="Pattern", style="Title.TLabel").pack(anchor="w")
        self.status = ttk.Label(side, text="", wraplength=370, justify="left")
        self.status.pack(anchor="w", pady=(4, 8))
        self.pattern_text = tk.Text(side, font=style.MONO, height=20, relief="solid", borderwidth=1, wrap="none")
        self.pattern_text.pack(fill="both", expand=True)

        self.clip.set(self.cfg.taxonomy.classes[0])
        self.canvas.set_class(self.clip.get())
        self.after(50, lambda: self._start(boxes))

    def _start(self, boxes: list[Box] | None) -> None:
        self.canvas.set_image(self.image, boxes or [])
        if boxes is None:
            self.detect()
        else:
            self.update_pattern()
        self.code.focus_set()

    def detect(self) -> None:
        self.status.configure(text="Finding the clips ...", foreground=style.MUTED)
        insp = self.page.station.inspector
        thr = self.cfg.detector.confidence_threshold

        def done(dets):
            if not self.winfo_exists():
                return
            self.canvas.set_boxes([Box(d.label, d.x, d.y, d.w, d.h) for d in dets if d.confidence >= thr])
            self.update_pattern()

        def failed(exc):
            if self.winfo_exists():
                self.status.configure(text=f"Detector failed: {exc}. Mark the clips by hand.", foreground=style.NG)

        if not insp.detector.ready:
            failed("no trained detector")
            return
        self.page.app.dispatcher.run_task(lambda: insp.detect(self.image), done, failed)

    def _on_select(self, box: Box | None) -> None:
        if box is not None:
            self.clip.set(box.label)

    def boxes(self) -> list[Box]:
        return list(self.canvas.boxes)

    def update_pattern(self) -> list[list[str]] | None:
        boxes = self.boxes()
        h, w = self.image.shape[:2]
        self.pattern_text.configure(state="normal")
        self.pattern_text.delete("1.0", "end")
        try:
            pattern, layout = master_from_boxes(boxes, (w, h))
        except LayoutError as exc:
            self.status.configure(text=f"{len(boxes)} clip(s) - {exc}", foreground=style.NG)
            self.pattern_text.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            return None
        self.status.configure(text=f"{len(boxes)} clips  ·  {layout.cables} cables x {layout.rows} rows",
                              foreground=style.OK)
        for r, row in enumerate(pattern, 1):
            self.pattern_text.insert("end", f"Row {r}\n")
            for c, label in enumerate(row, 1):
                self.pattern_text.insert("end", f"   cable {c}: {label}\n")
        self.pattern_text.configure(state="disabled")
        self.save_btn.configure(state="normal")
        return pattern

    def save(self) -> PartNumber | None:
        code = self.code.get().strip()
        if not code:
            show_error(self, "Enter the part number.")
            return None
        replaced = self.part.code if self.part else None
        if code in self.cfg.parts and code != replaced and not ask_yes_no(self, f"{code} exists. Replace it?"):
            return None
        h, w = self.image.shape[:2]
        try:
            part = part_from_boxes(code, self.boxes(), (w, h), self.cfg.taxonomy, self.desc.get())
        except (LayoutError, ValueError) as exc:
            show_error(self, str(exc))
            return None
        self.page.save_part(part, self.image, self.boxes(), replaced, self.source)
        show_info(self, f"Saved {code}: {part.cables} cables x {part.rows} rows.", "Part saved")
        self.destroy()
        return part
