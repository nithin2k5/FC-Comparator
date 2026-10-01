"""Model Setup - the selected part's master.

f. Photograph a known-good board. The part's model finds its objects (or draw them by
   hand); they are sorted into cables and rows. The master keeps the pattern - which
   label belongs at each (cable, row) - and the layout - where each position sits in
   the image. "Save master" stays disabled until every position has exactly one object.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from ...core.layout import LayoutError, master_from_boxes
from ...core.models import Box
from ...core.parts import part_from_boxes
from ...vision.camera import read_image
from ...vision.drawing import draw_boxes
from .. import style
from ..box_editor import AnnotationCanvas
from ..widgets import ImageView, Table, card, center_window, show_error, show_info


def pattern_rows(pattern: list[list[str]]) -> list[tuple[str, list[str], tuple]]:
    return [(str(r), [f"Row {r}"] + list(row), ()) for r, row in enumerate(pattern, 1)]


class MasterPanel(ttk.Frame):
    def __init__(self, master, page):
        super().__init__(master, padding=8)
        self.page = page
        self.station = page.station
        self.cfg = page.cfg
        self.part = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        body = card(self, "Master", "The board every inspection of this part is compared with.")
        body.panel.grid(row=0, column=0, sticky="nsew")
        body.grid_rowconfigure(2, weight=1)
        body.grid_columnconfigure(0, weight=1)
        row = ttk.Frame(body, style="Card.TFrame")
        row.grid(row=0, column=0, sticky="ew")
        self.buttons = [
            ttk.Button(row, text="New master from photo...", style="Accent.TButton", command=self.new_from_file),
            ttk.Button(row, text="New master from camera", command=self.new_from_camera),
            ttk.Button(row, text="Edit master", command=self.edit),
        ]
        for i, b in enumerate(self.buttons):
            b.pack(side="left", padx=(0 if i == 0 else 6, 0))
        self.info = ttk.Label(body, text="", style="CardMuted.TLabel")
        self.info.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.view = ImageView(body, placeholder="No master yet")
        self.view.grid(row=2, column=0, sticky="nsew", pady=6)
        self.pattern = Table(body, [("row", "", 60)], height=5)
        self.pattern.grid(row=3, column=0, sticky="ew")
        self.set_part(None)

    def set_part(self, part) -> None:
        self.part = part
        for b in self.buttons:
            b.configure(state="normal" if part else "disabled")
        self.refresh()

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        part = self.part
        pn = part.master if part else None
        if pn is None:
            self.info.configure(text="Choose a part number first." if part is None else
                                "No master yet - photograph a known-good board of this part.")
            self.view.clear("No master yet")
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
        self.info.configure(text=f"{pn.cables} cables x {pn.rows} rows  ·  {pn.cables * pn.rows} positions")
        rec = self._master_image()
        if rec is None:
            self.view.clear("The master photo is missing")
        else:
            self.view.set_image(draw_boxes(*rec))

    def _master_image(self) -> tuple[np.ndarray, list[Box]] | None:
        part = self.part
        if part is None or part.master is None or part.master.master_image not in part.store:
            return None
        try:
            return part.store.load_image(part.master.master_image), part.store.get(part.master.master_image).boxes
        except Exception:
            return None

    # -- new / edit -------------------------------------------------------------------
    def new_from_file(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Photo of a known-good board",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if not path:
            return
        try:
            img = read_image(path)
        except Exception as exc:
            show_error(self, str(exc))
            return
        self.open_editor(img, None, source=path)

    def new_from_camera(self) -> None:
        try:
            img = self.station.source.capture(timeout=5)
        except Exception as exc:
            show_error(self, f"Camera: {exc}")
            return
        self.open_editor(img, None)

    def edit(self) -> None:
        rec = self._master_image()
        if rec is None:
            show_error(self, "This part has no master photo yet.")
            return
        self.open_editor(rec[0], rec[1], image_id=self.part.master.master_image)

    def open_editor(self, image: np.ndarray, boxes: list[Box] | None, source: str | None = None,
                    image_id: str | None = None) -> "MasterEditor":
        return self.page.track(MasterEditor(self, image, boxes, source, image_id))

    def save_master(self, image: np.ndarray, boxes: list[Box], source: str | None, image_id: str | None):
        """Store the photo (with its boxes) among the part's images and make it the master."""
        part = self.part
        h, w = image.shape[:2]
        master = part_from_boxes(part.code, boxes, (w, h), None, part.description)
        if image_id is None or image_id not in part.store:
            image_id, _new = part.store.add_image(source or image, good=True,
                                                  name="" if source else f"master {part.code}")
        part.store.set_boxes(image_id, boxes)
        part.store.update(image_id, good=True)
        master.master_image = image_id
        part.set_master(master)
        self.page.log("master_saved", f"{master.cables} cables x {master.rows} rows")
        self.page.changed()
        return master


class MasterEditor(tk.Toplevel):
    """Check the objects found on a good board, then save them as the part's master."""

    def __init__(self, panel: MasterPanel, image: np.ndarray, boxes: list[Box] | None, source: str | None,
                 image_id: str | None):
        super().__init__(panel)
        self.panel = panel
        self.part = panel.part
        self.cfg = panel.cfg
        self.image = image
        self.source = source
        self.image_id = image_id
        self.title(f"Master of {self.part.code}")
        self.minsize(1000, 700)
        top = panel.winfo_toplevel()
        center_window(self, panel, min(1280, max(1000, top.winfo_width() - 80)),
                      min(860, max(700, top.winfo_height() - 80)))
        self.configure(bg=style.BG)
        self.transient(panel.winfo_toplevel())

        bar = ttk.Frame(self, padding=10)
        bar.pack(fill="x")
        ttk.Label(bar, text=f"Master of {self.part.code}", style="Title.TLabel").pack(side="left")
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right")
        self.save_btn = ttk.Button(bar, text="Save master", style="Accent.TButton", command=self.save)
        self.save_btn.pack(side="right", padx=6)

        tools = ttk.Frame(self, padding=(10, 0))
        tools.pack(fill="x")
        ttk.Label(tools, text="Label").pack(side="left")
        labels = self.part.labels
        self.label = ttk.Combobox(tools, values=labels, state="readonly", width=20)
        self.label.pack(side="left", padx=6)
        self.label.bind("<<ComboboxSelected>>", lambda _e: self.canvas.set_class(self.label.get()))
        ttk.Button(tools, text="Delete box", command=lambda: self.canvas.delete_selected()).pack(side="left", padx=4)
        ttk.Button(tools, text="Undo", command=lambda: self.canvas.undo()).pack(side="left")
        ttk.Button(tools, text="Find objects again", command=self.detect).pack(side="left", padx=4)
        ttk.Label(tools, text="Drag on the image to add a box · click a box to select · wheel scrolls, Ctrl+wheel zooms, right-drag pans",
                  style="Muted.TLabel").pack(side="left", padx=12)

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.canvas = AnnotationCanvas(body, labels, on_change=lambda _b: self.update_pattern(),
                                       on_select=self._on_select, on_need_label=self._new_label)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        side = ttk.Frame(body, width=380)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="Pattern", style="Title.TLabel").pack(anchor="w")
        self.status = ttk.Label(side, text="", wraplength=370, justify="left")
        self.status.pack(anchor="w", pady=(4, 8))
        self.pattern_text = tk.Text(side, font=style.MONO, height=20, relief="solid", borderwidth=1, wrap="none")
        self.pattern_text.pack(fill="both", expand=True)

        if labels:
            self.label.set(labels[0])
            self.canvas.set_class(labels[0])
        self.after(50, lambda: self._start(boxes))

    def _start(self, boxes: list[Box] | None) -> None:
        self.canvas.set_image(self.image, boxes or [])
        if boxes is None:
            self.detect()
        else:
            self.update_pattern()

    def detect(self) -> None:
        part, station = self.part, self.panel.station
        if self.cfg.detector.backend == "yolo" and part.active() is None:
            self.status.configure(text="This part has no model in use yet - draw the boxes by hand.",
                                  foreground=style.WARN)
            self.update_pattern(keep_status=True)
            return
        self.status.configure(text="Finding the objects ...", foreground=style.MUTED)
        thr = self.cfg.detector.confidence_threshold

        def done(dets):
            if not self.winfo_exists():
                return
            self.canvas.set_boxes([Box(d.label, d.x, d.y, d.w, d.h) for d in dets if d.confidence >= thr])
            self.update_pattern()

        def failed(exc):
            if self.winfo_exists():
                self.status.configure(text=f"Finding objects failed: {exc}. Draw the boxes by hand.",
                                      foreground=style.NG)

        self.panel.page.dispatcher.run_task(lambda: station.detector_for(part.code).detect(self.image), done, failed)

    def _on_select(self, box: Box | None) -> None:
        if box is not None:
            self.label.set(box.label)

    def _new_label(self) -> str | None:
        name = self.panel.page.images.new_label()
        if name:
            self.label.configure(values=self.part.labels)
            self.canvas.set_classes(self.part.labels)
            self.label.set(name)
        return name

    def boxes(self) -> list[Box]:
        return list(self.canvas.boxes)

    def update_pattern(self, keep_status: bool = False) -> list[list[str]] | None:
        boxes = self.boxes()
        h, w = self.image.shape[:2]
        self.pattern_text.configure(state="normal")
        self.pattern_text.delete("1.0", "end")
        try:
            pattern, layout = master_from_boxes(boxes, (w, h))
        except LayoutError as exc:
            if not keep_status:
                self.status.configure(text=f"{len(boxes)} object(s) - {exc}", foreground=style.NG)
            self.pattern_text.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            return None
        self.status.configure(text=f"{len(boxes)} objects  ·  {layout.cables} cables x {layout.rows} rows",
                              foreground=style.OK)
        for r, row in enumerate(pattern, 1):
            self.pattern_text.insert("end", f"Row {r}\n")
            for c, label in enumerate(row, 1):
                self.pattern_text.insert("end", f"   cable {c}: {label}\n")
        self.pattern_text.configure(state="disabled")
        self.save_btn.configure(state="normal")
        return pattern

    def save(self):
        try:
            master = self.panel.save_master(self.image, self.boxes(), self.source, self.image_id)
        except (LayoutError, ValueError) as exc:
            show_error(self, str(exc))
            return None
        show_info(self, f"Saved the master of {self.part.code}: {master.cables} cables x {master.rows} rows.",
                  "Master saved")
        self.destroy()
        return master
