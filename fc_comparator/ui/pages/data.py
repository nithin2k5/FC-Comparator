"""Training data page: upload images, mark the clips on them, create part masters from good boards."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ...capture.sources import IMAGE_EXTENSIONS
from ...config import save_config
from ...layout import LayoutError
from ...models import Box
from ...parts import part_from_marked_image
from ...pipeline import Station
from .. import theme
from ..annotator import AnnotationCanvas
from ..widgets import Card, Dispatcher, Table, ask_fields, ask_yes_no, button, muted, show_error, show_info

log = logging.getLogger(__name__)

FILTERS = ["All", "To mark", "Marked", "Good"]


class TrainingDataPage(ctk.CTkFrame):
    title = "Training data"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.store = station.annotations
        self.dispatcher = dispatcher
        self.image_id: str | None = None
        self.dirty = False  # markings changed since the detector was built
        self._class_buttons: dict[str, ctk.CTkButton] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- left: image list -----------------------------------------------------
        left = Card(self, title="Images", subtitle="Upload photos of boards, then mark every clip on them.")
        left.configure(width=330)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        left.grid_propagate(False)
        lb = left.body
        lb.grid_columnconfigure((0, 1), weight=1)
        lb.grid_rowconfigure(3, weight=1)
        button(lb, "+ Images", self.add_files, "primary").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        button(lb, "+ Folder", self.add_folder, "secondary").grid(row=0, column=1, sticky="ew", padx=(4, 0))
        button(lb, "Capture from camera", self.capture, "secondary").grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.filter = ctk.CTkSegmentedButton(lb, values=FILTERS, command=lambda _v: self.refresh_list(), height=34)
        self.filter.set("All")
        self.filter.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        self.table = Table(lb, [("name", "Image", 150), ("clips", "Clips", 50), ("part", "Part", 70)])
        self.table.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.table.on_select(self.open_image)
        self.totals = muted(lb, "", 12, wraplength=290)
        self.totals.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # -- centre: canvas -----------------------------------------------------------
        centre = Card(self)
        centre.grid(row=0, column=1, sticky="nsew")
        cb = centre.body
        cb.grid_columnconfigure(0, weight=1)
        cb.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(cb, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.image_title = ctk.CTkLabel(top, text="No image selected", font=theme.font(16, "bold"), text_color=theme.TEXT)
        self.image_title.pack(side="left")
        button(top, "→", lambda: self.step(1), "secondary", width=44).pack(side="right")
        button(top, "←", lambda: self.step(-1), "secondary", width=44).pack(side="right", padx=6)
        button(top, "Fit", lambda: self.canvas.fit(), "secondary", width=60).pack(side="right")
        self.canvas = AnnotationCanvas(cb, self.cfg.taxonomy.classes, on_change=self._on_boxes_changed,
                                       on_select=self._on_box_selected, on_navigate=self.step)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.hint = muted(cb, "Drag to draw a box  ·  click to select, drag to move, handles to resize  ·  "
                              "1-9 clip type  ·  Del remove  ·  Ctrl+Z undo  ·  wheel zoom, right-drag pan  "
                              "·  ← → images", 11)
        self.hint.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        cb.bind("<Configure>", lambda e: self.hint.configure(wraplength=max(200, e.width - 20)))

        # -- right: tools -------------------------------------------------------------
        right = ctk.CTkScrollableFrame(self, fg_color="transparent", width=300)
        right.grid(row=0, column=2, sticky="nse", padx=(14, 0))
        self.palette = Card(right, title="Clip type", subtitle="New boxes get this type. Keys 1-9.")
        self.palette.pack(fill="x")
        self._build_palette()

        info = Card(right, title="This image")
        info.pack(fill="x", pady=(12, 0))
        ib = info.body
        muted(ib, "Part number on this board (optional)").pack(anchor="w")
        self.part_entry = ctk.CTkComboBox(ib, values=sorted(self.cfg.parts), height=36, command=lambda _v: self._save_meta())
        self.part_entry.set("")
        self.part_entry.pack(fill="x", pady=(2, 8))
        self.part_entry.bind("<FocusOut>", lambda _e: self._save_meta())
        self.good = ctk.CTkSwitch(ib, text="Known-good board", command=self._save_meta)
        self.good.pack(anchor="w")
        self.counts = muted(ib, "", 12, wraplength=260)
        self.counts.pack(anchor="w", pady=(10, 0))

        master_card = Card(right, title="Part master", subtitle="A completely marked, known-good board defines a part: "
                                                                "clip per row and where each clip sits.")
        master_card.pack(fill="x", pady=(12, 0))
        button(master_card.body, "Create / update part from this image", self.make_master, "primary").pack(fill="x")

        tools = Card(right, title="Tools")
        tools.pack(fill="x", pady=(12, 0))
        tb = tools.body
        button(tb, "Auto-mark with detector", self.auto_mark, "secondary").pack(fill="x")
        muted(tb, "Pre-marks the clips the current detector finds; check every box, then correct.", 11,
              wraplength=260).pack(anchor="w", pady=(2, 8))
        button(tb, "Clear all clips", self.clear_boxes, "secondary").pack(fill="x")
        button(tb, "Delete image", self.delete_image, "danger").pack(fill="x", pady=(8, 0))

        self.refresh_list()

    # -- palette ---------------------------------------------------------------
    def _build_palette(self) -> None:
        for w in self.palette.body.winfo_children():
            w.destroy()
        self._class_buttons = {}
        classes = self.cfg.taxonomy.classes
        for i, label in enumerate(classes):
            color = theme.class_color(classes, label)
            b = ctk.CTkButton(self.palette.body, text=f"{i + 1}   {label}", anchor="w", height=38, corner_radius=10,
                              fg_color=theme.SURFACE_2, hover_color=theme.BORDER, text_color=theme.TEXT,
                              border_width=2, border_color=color, font=theme.font(14, "bold"),
                              command=lambda lb=label: self._choose_class(lb))
            b.pack(fill="x", pady=3)
            self._class_buttons[label] = b
        self.canvas.set_classes(classes)
        self._highlight_class()

    def _choose_class(self, label: str) -> None:
        self.canvas.set_class(label)
        self._highlight_class()
        self.canvas.focus_set()

    def _highlight_class(self) -> None:
        classes = self.cfg.taxonomy.classes
        for label, b in self._class_buttons.items():
            active = label == self.canvas.current_class
            color = theme.class_color(classes, label)
            b.configure(fg_color=color if active else theme.SURFACE_2, text_color="#ffffff" if active else theme.TEXT)

    # -- list ------------------------------------------------------------------
    def refresh(self) -> None:
        if list(self._class_buttons) != self.cfg.taxonomy.classes:
            self._build_palette()
        self.part_entry.configure(values=sorted(self.cfg.parts))
        self.refresh_list()

    def refresh_list(self) -> None:
        f = self.filter.get()
        rows = []
        for r in self.store.records():
            if (f == "To mark" and r.marked) or (f == "Marked" and not r.marked) or (f == "Good" and not r.good):
                continue
            name = r.source_name or r.id
            rows.append((r.id, (("✓ " if r.good else "") + name, len(r.boxes) or "-", r.part), () if r.marked else ("muted",)))
        self.table.set_rows(rows)
        st = self.store.stats()
        per = ", ".join(f"{k} {v}" for k, v in st["per_class"].items()) or "none yet"
        self.totals.configure(text=f"{st['images']} images  ·  {st['marked']} marked  ·  {st['good']} good\n"
                                   f"Clips: {per}")

    def step(self, delta: int) -> None:
        ids = list(self.table.tree.get_children())
        if not ids:
            return
        i = ids.index(self.image_id) + delta if self.image_id in ids else 0
        self.table.select(ids[max(0, min(len(ids) - 1, i))])

    # -- image -----------------------------------------------------------------
    def open_image(self, image_id: str | None) -> None:
        if not image_id or image_id == self.image_id:
            return
        self._save_meta()
        try:
            img = self.store.load_image(image_id)
        except Exception as exc:
            show_error(self, f"Cannot open this image: {exc}")
            return
        rec = self.store.get(image_id)
        self.image_id = image_id
        self.canvas.set_image(img, rec.boxes)
        self.image_title.configure(text=f"{rec.source_name or rec.id}   {rec.width}x{rec.height}")
        self.part_entry.set(rec.part)
        self.good.select() if rec.good else self.good.deselect()
        self._update_counts(rec.boxes)
        self.canvas.focus_set()

    def _on_boxes_changed(self, boxes: list[Box]) -> None:
        if self.image_id is None:
            return
        self.store.set_boxes(self.image_id, boxes)
        self.dirty = True
        self._update_counts(boxes)
        rec = self.store.get(self.image_id)
        if self.table.tree.exists(self.image_id):
            self.table.tree.item(self.image_id, values=(("✓ " if rec.good else "") + (rec.source_name or rec.id),
                                                        len(rec.boxes) or "-", rec.part),
                                 tags=() if rec.marked else ("muted",))

    def _on_box_selected(self, box: Box | None) -> None:
        self._highlight_class()

    def _update_counts(self, boxes: list[Box]) -> None:
        c = Counter(b.label for b in boxes)
        self.counts.configure(text=f"{len(boxes)} clip(s) marked" + (": " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) if c else ""))

    def _save_meta(self) -> None:
        if self.image_id is None or self.image_id not in self.store:
            return
        rec = self.store.get(self.image_id)
        part, good = self.part_entry.get().strip(), bool(self.good.get())
        if (part, good) != (rec.part, rec.good):
            self.store.update(self.image_id, part=part, good=good)
            self.refresh_list()

    # -- upload ----------------------------------------------------------------
    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="Add board images",
                                            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if paths:
            self._import([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Add all images in a folder")
        if folder:
            self._import(sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))

    def _import(self, paths: list[Path]) -> None:
        part = self.part_entry.get().strip() if self.image_id is None else ""
        self.totals.configure(text=f"Adding {len(paths)} image(s) ...")

        def work():
            added = dup = bad = 0
            for p in paths:
                try:
                    _, new = self.store.add_image(p, part=part)
                    added += new
                    dup += not new
                except Exception as exc:
                    log.warning("Skipping %s: %s", p, exc)
                    bad += 1
            return added, dup, bad

        def done(res):
            added, dup, bad = res
            self.filter.set("To mark")
            self.refresh_list()
            msg = f"Added {added} image(s)."
            if dup:
                msg += f" {dup} were already in the data set."
            if bad:
                msg += f" {bad} could not be read."
            show_info(self, msg + "\n\nSelect an image and mark every clip on it.", "Images added")
            self.step(0)

        self.dispatcher.run_task(work, done, lambda e: show_error(self, str(e)))

    def capture(self) -> None:
        try:
            img = self.station.source.capture(timeout=3)
        except Exception as exc:
            show_error(self, f"Capture failed: {exc}")
            return
        image_id, _ = self.store.add_image(img, name="camera capture")
        self.refresh_list()
        self.table.select(image_id)

    # -- tools -----------------------------------------------------------------
    def auto_mark(self) -> None:
        if self.canvas.image is None:
            return
        det = self.station.inspector.detector
        if not det.ready:
            show_error(self, "Mark a few images by hand first - the detector learns from them.", "Nothing to learn from yet")
            return
        if self.canvas.boxes and not ask_yes_no(self, "Replace the clips marked on this image with the detector's result?"):
            return
        img = self.canvas.image

        def done(dets):
            self.canvas.set_boxes([d for d in dets if d.confidence >= self.cfg.detector.min_score])
            show_info(self, f"{len(self.canvas.boxes)} clip(s) pre-marked. Check every box: fix the type (1-9), "
                            "move/resize, delete wrong ones and draw any that were missed.", "Please review")

        self.dispatcher.run_task(lambda: det.detect(img), done, lambda e: show_error(self, str(e)))

    def clear_boxes(self) -> None:
        if self.canvas.boxes and ask_yes_no(self, "Remove all clips marked on this image? (Ctrl+Z undoes it)"):
            self.canvas.set_boxes([])

    def delete_image(self) -> None:
        if self.image_id is None:
            return
        used = [c for c, p in self.cfg.parts.items() if p.master_image == self.image_id]
        warn = f"\n\nIt is the master image of {', '.join(used)} (their patterns are kept)." if used else ""
        if not ask_yes_no(self, f"Delete this image and its markings?{warn}", yes="Delete", danger=True):
            return
        self.store.delete(self.image_id)
        self.image_id = None
        self.dirty = True
        self.canvas.set_image(None, [])
        self.image_title.configure(text="No image selected")
        self.refresh_list()

    def make_master(self) -> None:
        if self.image_id is None:
            return
        rec = self.store.get(self.image_id)
        code0 = rec.part or ""
        desc0 = self.cfg.parts[code0].description if code0 in self.cfg.parts else ""
        answer = ask_fields(self, "Part from this image", [("Part number", code0), ("Description", desc0)], "Preview")
        if not answer:
            return
        code, desc = answer
        try:
            part, notes = part_from_marked_image(self.store, self.image_id, code, self.cfg.taxonomy, desc)
        except (LayoutError, ValueError) as exc:
            show_error(self, f"{exc}\n\nMark every clip on a complete good board and try again.", "Cannot create the master")
            return
        # show which clip became which position
        captions = {}
        for i, b in enumerate(self.canvas.boxes):
            key = min(part.layout.positions, key=lambda k: (part.layout.positions[k][0] - b.center[0]) ** 2
                                                          + (part.layout.positions[k][1] - b.center[1]) ** 2)
            captions[i] = f"C{key[0]} R{key[1]}"
        self.canvas.captions = captions
        self.canvas.redraw()
        rows = "\n".join(f"  Row {r + 1}: {label}" for r, label in enumerate(part.pattern))
        exists = code in self.cfg.parts
        text = (f"{code}: {part.cables} cables x {part.rows} clips\n{rows}"
                + ("\n\nNotes:\n" + "\n".join(notes) if notes else "")
                + (f"\n\n{code} already exists and will be replaced." if exists else ""))
        ok = ask_yes_no(self, text, "Save this part?", yes="Replace" if exists else "Save")
        self.canvas.captions = {}
        self.canvas.redraw()
        if not ok:
            return
        self.cfg.parts[code] = part
        save_config(self.cfg)
        self.store.update(self.image_id, part=code, good=True)
        self.good.select()
        self.part_entry.set(code)
        self.station.store.log_event("part_master", "", f"{code} from image {self.image_id}: {part.pattern}")
        self.refresh_list()
        self.app.on_config_changed()
        show_info(self, f"{code} saved. Operators can now inspect it.", "Part saved")

    # -- lifecycle -------------------------------------------------------------
    def on_hide(self) -> None:
        self._save_meta()
        if self.dirty and self.station.inspector.detector.name == "template":
            self.dirty = False
            self.app.rebuild_detector()  # the template detector learns from the markings
