"""Setup page: one flow per part number.

1. choose (or create) the part number
2. add images of it: files, a folder, or frames from the live camera
3. annotate every clip; one completely marked good board is the part's reference
4. train the model and use it for detection
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ...capture.sources import IMAGE_EXTENSIONS, CameraSource
from ...config import save_config
from ...layout import LayoutError
from ...models import Box
from ...parts import part_from_marked_image
from ...pipeline import Station
from ...training import train_detector
from .. import theme
from ..annotator import AnnotationCanvas
from ..widgets import (CameraDialog, Card, Dispatcher, Table, ask_fields, ask_yes_no, button, muted, show_error,
                       show_info)

log = logging.getLogger(__name__)

MIN_PER_CLASS = 20  # marked clips per type before training is worthwhile


class SetupPage(ctk.CTkFrame):
    title = "Setup"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.store = station.annotations
        self.dispatcher = dispatcher
        self.code = ""  # the part being set up
        self.descriptions: dict[str, str] = {}  # descriptions of new parts that have no reference board yet
        self.image_id: str | None = None
        self.dirty = False  # markings changed since the detector was built
        self.training = False
        self._stop = threading.Event()
        self._t0 = 0.0
        self._class_buttons: dict[str, ctk.CTkButton] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- left: 1 part, 2 images ---------------------------------------------------
        left = ctk.CTkFrame(self, fg_color="transparent", width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        part_card = Card(left, title="1   Part number")
        part_card.grid(row=0, column=0, sticky="ew")
        pb = part_card.body
        pb.grid_columnconfigure(0, weight=1)
        self.part_box = ctk.CTkComboBox(pb, values=[], height=38, font=theme.font(15, "bold"),
                                        command=self.select_part)
        self.part_box.set("")
        self.part_box.grid(row=0, column=0, sticky="ew")
        self.part_box.bind("<Return>", lambda _e: self.select_part(self.part_box.get()))
        button(pb, "+ New", self.new_part, "primary", width=80).grid(row=0, column=1, padx=(8, 0))
        self.part_info = muted(pb, "", 12, wraplength=300)
        self.part_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        img_card = Card(left, title="2   Images")
        img_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        ib = img_card.body
        ib.grid_columnconfigure((0, 1), weight=1)
        ib.grid_rowconfigure(3, weight=1)
        self.upload_btns = [
            button(ib, "+ Images", self.add_files, "secondary"),
            button(ib, "+ Folder", self.add_folder, "secondary"),
            button(ib, "Capture from camera", self.open_camera, "secondary"),
        ]
        self.upload_btns[0].grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.upload_btns[1].grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.upload_btns[2].grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.scope = ctk.CTkSegmentedButton(ib, values=["This part", "All parts"], height=32,
                                            command=lambda _v: self.refresh_list())
        self.scope.set("This part")
        self.scope.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        self.table = Table(ib, [("name", "Image", 170), ("clips", "Clips", 50), ("part", "Part", 70)])
        self.table.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.table.on_select(self.open_image)
        self.totals = muted(ib, "", 12, wraplength=300)
        self.totals.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # -- centre: 3 annotate -------------------------------------------------------
        centre = Card(self, title="3   Annotate")
        centre.grid(row=0, column=1, sticky="nsew")
        cb = centre.body
        cb.grid_columnconfigure(0, weight=1)
        cb.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(cb, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.image_title = ctk.CTkLabel(top, text="No image selected", font=theme.font(15, "bold"), text_color=theme.TEXT)
        self.image_title.pack(side="left")
        button(top, "→", lambda: self.step(1), "secondary", width=44).pack(side="right")
        button(top, "←", lambda: self.step(-1), "secondary", width=44).pack(side="right", padx=6)
        button(top, "Fit", lambda: self.canvas.fit(), "secondary", width=60).pack(side="right")
        self.canvas = AnnotationCanvas(cb, self.cfg.taxonomy.classes, on_change=self._on_boxes_changed,
                                       on_select=lambda _b: self._highlight_class(), on_navigate=self.step)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.hint = muted(cb, "Mark every clip: drag a box  ·  1-9 clip type  ·  click to select, drag to move  ·  "
                              "Del remove  ·  Ctrl+Z undo  ·  wheel zoom, right-drag pan  ·  ← → images", 11)
        self.hint.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        cb.bind("<Configure>", lambda e: self.hint.configure(wraplength=max(200, e.width - 60)))
        tools = ctk.CTkFrame(cb, fg_color="transparent")
        tools.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        tools.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="tools")  # shrink together on small screens
        self.ref_btn = button(tools, "★ Set reference", self.make_reference, "primary", width=60)
        self.ref_btn.grid(row=0, column=0, sticky="ew")
        button(tools, "Auto-mark", self.auto_mark, "secondary", width=60).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        button(tools, "Clear", self.clear_boxes, "secondary", width=60).grid(row=0, column=2, sticky="ew", padx=(8, 0))
        button(tools, "Delete image", self.delete_image, "danger", width=60).grid(row=0, column=3, sticky="ew", padx=(8, 0))

        # -- right: clip types, 4 train ---------------------------------------------
        right = ctk.CTkScrollableFrame(self, fg_color="transparent", width=300)
        right.grid(row=0, column=2, sticky="nse", padx=(14, 0))
        self.palette = Card(right, title="Clip type", subtitle="New boxes get this type. Keys 1-9.")
        self.palette.pack(fill="x")
        self._build_palette()

        train = Card(right, title="4   Train & detect", subtitle="One model for all parts.")
        train.pack(fill="x", pady=(12, 0))
        tb = train.body
        tb.grid_columnconfigure(1, weight=1)
        self.data_note = muted(tb, "", 12, wraplength=260)
        self.data_note.grid(row=0, column=0, columnspan=2, sticky="w")
        muted(tb, "Epochs", 13).grid(row=1, column=0, sticky="w", pady=(10, 0), padx=(0, 10))
        self.epochs = ctk.CTkEntry(tb, height=34, width=90)
        self.epochs.insert(0, str(self.cfg.training.epochs))
        self.epochs.grid(row=1, column=1, sticky="w", pady=(10, 0))
        row = ctk.CTkFrame(tb, fg_color="transparent")
        row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.train_btn = button(row, "Train model", self.start_training, "primary", width=150)
        self.train_btn.pack(side="left")
        self.stop_btn = button(row, "Stop", self.stop_training, "danger", width=80)  # shown only while training
        self.progress = ctk.CTkProgressBar(tb, height=10)
        self.progress.set(0)
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        self.train_status = muted(tb, "", 12, wraplength=260)
        self.train_status.grid(row=4, column=0, columnspan=2, sticky="w")
        self.det_label = ctk.CTkLabel(tb, text="", font=theme.font(13, "bold"), text_color=theme.TEXT, anchor="w",
                                      justify="left", wraplength=260)
        self.det_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        self._update_part_info()

    # -- palette ---------------------------------------------------------------
    def _build_palette(self) -> None:
        for w in self.palette.body.winfo_children():
            w.destroy()
        self._class_buttons = {}
        classes = self.cfg.taxonomy.classes
        for i, label in enumerate(classes):
            b = ctk.CTkButton(self.palette.body, text=f"{i + 1}   {label}", anchor="w", height=36, corner_radius=10,
                              fg_color=theme.SURFACE_2, hover_color=theme.BORDER, text_color=theme.TEXT,
                              border_width=2, border_color=theme.class_color(classes, label),
                              font=theme.font(14, "bold"), command=lambda lb=label: self._choose_class(lb))
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
            b.configure(fg_color=theme.class_color(classes, label) if active else theme.SURFACE_2,
                        text_color="#ffffff" if active else theme.TEXT)

    # -- 1 part number -----------------------------------------------------------
    def _known_parts(self) -> list[str]:
        """Parts in the config plus new parts that so far only have images."""
        return sorted(set(self.cfg.parts) | {r.part for r in self.store.records() if r.part} | set(self.descriptions))

    def select_part(self, code: str) -> None:
        code = code.strip()
        if code != self.code:
            self.code = code
            self.image_id = None
            self.canvas.set_image(None, [])
            self.image_title.configure(text="No image selected")
        self.part_box.set(code)
        self.part_box.configure(values=self._known_parts())
        self._update_part_info()
        self.refresh_list()
        if self.image_id is None:
            self.step(0)

    def new_part(self) -> None:
        answer = ask_fields(self, "New part number", [("Part number", ""), ("Description", "")], "Create")
        if not answer or not answer[0]:
            return
        code, desc = answer
        if code in self._known_parts():
            show_info(self, f"{code} exists already and is now selected.", "Part exists")
        else:
            self.descriptions[code] = desc
        self.scope.set("This part")
        self.select_part(code)

    def _update_part_info(self) -> None:
        state = "normal" if self.code else "disabled"
        for b in self.upload_btns:
            b.configure(state=state)
        if not self.code:
            self.part_info.configure(text="Choose a part number, or press + New.", text_color=theme.MUTED)
            return
        p = self.cfg.parts.get(self.code)
        if p is None:
            self.part_info.configure(text="Not ready for inspection yet: annotate a complete good board of this "
                                          "part and press '★ Set reference'.", text_color=theme.WARN)
            return
        desc = p.description or self.descriptions.get(self.code, "")
        ref = "reference board set" if p.layout is not None else "no reference board (pattern only)"
        self.part_info.configure(text=f"{desc + '  ·  ' if desc else ''}{p.cables} cables x {p.rows} clips  ·  {ref}\n"
                                      f"Clip per row: {', '.join(p.pattern)}", text_color=theme.MUTED)

    # -- 2 images ------------------------------------------------------------------
    def refresh(self) -> None:
        if list(self._class_buttons) != self.cfg.taxonomy.classes:
            self._build_palette()
        if not self.code:
            codes = self._known_parts()
            current = self.app.inspect.part_code
            if current in codes:
                self.code = current
            elif codes:
                self.code = codes[0]
        self.select_part(self.code)
        self._refresh_training()

    def _visible(self) -> list:
        recs = self.store.records()
        if self.scope.get() == "This part":
            recs = [r for r in recs if r.part == self.code]
        return recs

    def _row(self, rec) -> tuple:
        ref = self.cfg.parts.get(rec.part)
        star = "★ " if ref is not None and ref.master_image == rec.id else ""
        return (star + (rec.source_name or rec.id), len(rec.boxes) or "-", rec.part), () if rec.marked else ("muted",)

    def refresh_list(self) -> None:
        recs = self._visible()
        self.table.set_rows([(r.id, *self._row(r)) for r in recs])
        marked = sum(r.marked for r in recs)
        what = f"{self.code}: " if self.scope.get() == "This part" and self.code else "All parts: "
        self.totals.configure(text=f"{what}{len(recs)} image(s), {marked} annotated"
                                   + (f", {len(recs) - marked} still to annotate (grey)" if len(recs) > marked else ""))

    def step(self, delta: int) -> None:
        ids = list(self.table.tree.get_children())
        if not ids:
            return
        i = ids.index(self.image_id) + delta if self.image_id in ids else 0
        self.table.select(ids[max(0, min(len(ids) - 1, i))])

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title=f"Add images of {self.code}",
                                            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if paths:
            self._import([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title=f"Add all images in a folder to {self.code}")
        if folder:
            self._import(sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))

    def _import(self, paths: list[Path]) -> None:
        part = self.code
        self.totals.configure(text=f"Adding {len(paths)} image(s) ...")

        def work():
            added, dup, bad = [], 0, 0
            for p in paths:
                try:
                    image_id, new = self.store.add_image(p, part=part)
                    if new:
                        added.append(image_id)
                    else:
                        dup += 1
                except Exception as exc:
                    log.warning("Skipping %s: %s", p, exc)
                    bad += 1
            return added, dup, bad

        def done(res):
            added, dup, bad = res
            self.refresh_list()
            msg = f"Added {len(added)} image(s) to {part}."
            if dup:
                msg += f" {dup} were already in the data set."
            if bad:
                msg += f" {bad} could not be read."
            show_info(self, msg + "\n\nNow annotate every clip on them.", "Images added")
            if added:
                self.table.select(added[0])

        self.dispatcher.run_task(work, done, lambda e: show_error(self, str(e)))

    def open_camera(self) -> None:
        source = self.station.source
        if source.is_live:
            self._show_camera(source)
            return
        # the station inspects image files (camera.source: file): use the camera just for capturing
        cam = CameraSource(self.cfg.camera)
        self.totals.configure(text=f"Opening camera {self.cfg.camera.index} ...")

        def opened(_r):
            try:
                self._show_camera(cam)
            finally:
                cam.close()

        def failed(exc):
            cam.close()
            self.refresh_list()
            show_error(self, f"{exc}\n\nCheck the camera index and backend in Settings > Station & camera.",
                       "Camera not available")

        self.dispatcher.run_task(cam.open, opened, failed)

    def _show_camera(self, source) -> None:
        added: list[str] = []
        part = self.code

        def keep(frame) -> None:
            image_id, new = self.store.add_image(frame, part=part, name=f"camera {datetime.now():%Y-%m-%d %H:%M:%S}")
            if not new:
                raise ValueError("this frame is identical to an image already added")
            added.append(image_id)

        CameraDialog(self, source, keep, f"Capture images of {part}").run()
        self.refresh_list()
        if added:
            self.table.select(added[0])

    # -- 3 annotate ---------------------------------------------------------------
    def open_image(self, image_id: str | None) -> None:
        if not image_id or image_id == self.image_id:
            return
        try:
            img = self.store.load_image(image_id)
        except Exception as exc:
            show_error(self, f"Cannot open this image: {exc}")
            return
        rec = self.store.get(image_id)
        self.image_id = image_id
        self.canvas.set_image(img, rec.boxes)
        self._update_title(rec)
        self.canvas.focus_set()

    def _update_title(self, rec) -> None:
        n = len(rec.boxes)
        self.image_title.configure(text=f"{rec.source_name or rec.id}   ·   "
                                        + (f"{n} clip(s) marked" if n else "no clips marked yet"))

    def _on_boxes_changed(self, boxes: list[Box]) -> None:
        if self.image_id is None:
            return
        self.store.set_boxes(self.image_id, boxes)
        self.dirty = True
        rec = self.store.get(self.image_id)
        self._update_title(rec)
        if self.table.tree.exists(self.image_id):
            values, tags = self._row(rec)
            self.table.tree.item(self.image_id, values=values, tags=tags)

    def auto_mark(self) -> None:
        if self.canvas.image is None:
            return
        det = self.station.inspector.detector
        if not det.ready:
            show_error(self, "Annotate a few images by hand first; auto-mark uses what it learned from them.",
                       "Nothing learned yet")
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
        warn = f"\n\nIt is the reference board of {', '.join(used)} (their patterns are kept)." if used else ""
        if not ask_yes_no(self, f"Delete this image and its markings?{warn}", yes="Delete", danger=True):
            return
        self.store.delete(self.image_id)
        self.image_id = None
        self.dirty = True
        self.canvas.set_image(None, [])
        self.image_title.configure(text="No image selected")
        self.refresh_list()
        self.step(0)

    def make_reference(self) -> None:
        """This completely marked good board defines the part: clip per row and where each clip sits."""
        if self.image_id is None or not self.code:
            return
        code = self.code
        old = self.cfg.parts.get(code)
        desc0 = old.description if old is not None else self.descriptions.get(code, "")
        answer = ask_fields(self, f"Reference board of {code}", [("Description", desc0)], "Preview")
        if answer is None:
            return
        try:
            part, notes = part_from_marked_image(self.store, self.image_id, code, self.cfg.taxonomy, answer[0])
        except (LayoutError, ValueError) as exc:
            show_error(self, f"{exc}\n\nMark every clip on a complete good board and try again.",
                       "Cannot use this image")
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
        text = (f"{code}: {part.cables} cables x {part.rows} clips\n{rows}"
                + ("\n\nNotes:\n" + "\n".join(notes) if notes else "")
                + ("\n\nThis replaces the current reference of " + code + "." if old is not None else ""))
        ok = ask_yes_no(self, text, "Use this board as the reference?", yes="Save")
        self.canvas.captions = {}
        self.canvas.redraw()
        if not ok:
            return
        self.cfg.parts[code] = part
        save_config(self.cfg)
        self.descriptions.pop(code, None)
        self.store.update(self.image_id, part=code, good=True)
        self.station.store.log_event("part_master", "", f"{code} from image {self.image_id}: {part.pattern}")
        self._update_part_info()
        self.refresh_list()
        self.app.on_config_changed()
        show_info(self, f"{code} is ready for inspection.", "Reference saved")

    # -- 4 train & detect ---------------------------------------------------------
    def _refresh_training(self) -> None:
        st = self.store.stats()
        weak = []
        for label in self.cfg.taxonomy.classes:
            n = st["per_class"].get(label, 0)
            twin = self.cfg.taxonomy.mirror_of(label)
            if twin != label and self.cfg.training.mirror:
                n += st["per_class"].get(twin, 0)
            if n < MIN_PER_CLASS:
                weak.append(label)
        per = ", ".join(f"{k} {v}" for k, v in st["per_class"].items()) or "none yet"
        note = f"Uses all {st['marked']} annotated image(s). Clips: {per}."
        if weak:
            note += f"\nAnnotate more of: {', '.join(weak)} ({MIN_PER_CLASS}+ each)."
        self.data_note.configure(text=note)
        self.train_btn.configure(state="normal" if st["marked"] and not self.training else "disabled")
        self._show_detector()

    def _show_detector(self) -> None:
        det = self.station.inspector.detector
        name = {"yolo": "trained model", "template": "template matching"}.get(det.name, det.name)
        self.det_label.configure(text=f"Detecting with: {name}" if det.ready else "Detector not ready",
                                 text_color=theme.TEXT if det.ready else theme.NG)
        if not self.training:
            self.train_status.configure(text=det.note)

    def _say(self, msg: str) -> None:
        self.dispatcher.call(self._progress, msg)

    def _progress(self, msg: str) -> None:
        log.info(msg)
        if msg.startswith("epoch "):
            try:
                done, total = msg.split()[1].split("/")
                frac = int(done) / int(total)
                self.progress.set(frac)
                elapsed = time.time() - self._t0
                eta = elapsed / max(frac, 1e-6) - elapsed
                self.train_status.configure(text=f"Epoch {done}/{total}  ·  about {eta / 60:.0f} min left")
            except (ValueError, IndexError):
                pass
        else:
            self.train_status.configure(text=msg.splitlines()[0])

    def start_training(self) -> None:
        try:
            epochs = int(self.epochs.get())
            if not 1 <= epochs <= 1000:
                raise ValueError
        except ValueError:
            show_error(self, "Epochs must be a number between 1 and 1000.")
            return
        self.training = True
        self._stop.clear()
        self._t0 = time.time()
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.stop_btn.pack(side="left", padx=8)
        self.progress.set(0)
        self.train_status.configure(text="Preparing data set ...")
        self.dispatcher.run_task(
            lambda: train_detector(self.cfg, self.store, self._say, self._stop, epochs=epochs),
            self._trained, self._training_failed)

    def stop_training(self) -> None:
        self._stop.set()
        self.train_status.configure(text="Stopping after this step ...")
        self.stop_btn.configure(state="disabled")

    def _end_training(self) -> None:
        self.training = False
        self.stop_btn.pack_forget()
        self._refresh_training()

    def _trained(self, res) -> None:
        ev = res.evaluation
        self.progress.set(1.0)
        self._end_training()
        summary = (f"Validation accuracy {ev.accuracy:.1%}  ·  {ev.missed_clips} missed  ·  "
                   f"{ev.false_clips} false clip(s) on {ev.images} image(s), {res.epochs_run} epochs"
                   + (" (stopped early)" if res.stopped else "") + ".")
        need = self.cfg.detector.min_model_accuracy
        if ev.accuracy >= need:
            backend = "auto" if self.cfg.detector.backend == "template" else self.cfg.detector.backend
            msg = "The station now detects with the new model."
        elif ask_yes_no(self, f"{summary}\n\nThat is below the {need:.0%} recommended; some clips may be reported as "
                              "uncertain. Annotate more images or train longer for a better model.\n\n"
                              "Use this model for detection anyway?", "Model below target", yes="Use model"):
            backend = "yolo"
            msg = "The station now detects with the new model."
        else:
            backend = "auto"
            msg = "The station keeps template matching."
        if backend != self.cfg.detector.backend:
            self.cfg.detector.backend = backend
            save_config(self.cfg)
        self.station.store.log_event("model_trained", "", f"accuracy {ev.accuracy:.3f}, backend {backend}")
        self.app.rebuild_detector(then=self._refresh_training)
        show_info(self, f"{summary}\n\n{msg}", "Training finished")

    def _training_failed(self, exc: BaseException) -> None:
        self._end_training()
        self.train_status.configure(text=f"Training failed: {exc}")
        show_error(self, f"Training failed: {exc}\n\nOffline? Set Settings > Detection > Base model to "
                         "'from scratch', or copy the pretrained .pt file next to the program.")

    # -- lifecycle -------------------------------------------------------------
    def on_hide(self) -> None:
        if self.code in self.cfg.parts and not self.station.lock.locked:
            self.app.inspect.set_part(self.code)  # inspect what was just set up
        if self.dirty and self.station.inspector.detector.name == "template":
            self.dirty = False
            self.app.rebuild_detector()  # template matching learns from the markings

    def stop(self) -> None:
        self._stop.set()
