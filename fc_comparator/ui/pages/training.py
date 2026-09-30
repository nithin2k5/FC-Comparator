"""Training: the images the clip detector learns from, the boxes marked on them, and training the model.

1. Add images (files, a folder or camera captures).
2. Mark every clip on each image (draw a box, choose its clip type). "Find clips"
   pre-marks an image with the current model; only corrections are needed.
3. Train. The new model is used once its validation accuracy reaches the target.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ...core.models import Box
from ...vision.camera import IMAGE_EXTENSIONS
from ...vision.detect import model_report
from .. import style
from ..box_editor import AnnotationCanvas
from ..widgets import Table, ask_yes_no, card, set_text, show_error, show_info, text_box


class TrainingPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.station = app.station
        self.cfg = app.cfg
        self.dispatcher = app.dispatcher
        self.current: str | None = None
        self._stop: threading.Event | None = None
        self.training = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- images --------------------------------------------------------------
        left = card(self, "Training images")
        left.panel.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        self.table = Table(left, [("name", "Image", 150), ("clips", "Clips", 50), ("part", "Part", 70)],
                           height=20, on_select=lambda iid: self.open_image(iid))
        self.table.pack(fill="both", expand=True)
        b = ttk.Frame(left, style="Card.TFrame")
        b.pack(fill="x", pady=(8, 0))
        ttk.Button(b, text="Add images...", command=self.add_files).pack(fill="x")
        ttk.Button(b, text="Add folder...", command=self.add_folder).pack(fill="x", pady=4)
        ttk.Button(b, text="Capture from camera", command=self.add_capture).pack(fill="x")
        ttk.Button(b, text="Delete image", command=self.delete_image).pack(fill="x", pady=(4, 0))

        # -- marking -------------------------------------------------------------
        mid = ttk.Frame(self)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.grid_rowconfigure(1, weight=1)
        mid.grid_columnconfigure(0, weight=1)
        tools = ttk.Frame(mid)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(tools, text="Clip type").pack(side="left")
        self.clip = ttk.Combobox(tools, values=self.cfg.taxonomy.classes, state="readonly", width=18)
        self.clip.pack(side="left", padx=6)
        self.clip.bind("<<ComboboxSelected>>", lambda _e: self.canvas.set_class(self.clip.get()))
        ttk.Button(tools, text="Find clips", command=self.auto_mark).pack(side="left", padx=(6, 0))
        ttk.Button(tools, text="Delete box", command=lambda: self.canvas.delete_selected()).pack(side="left", padx=4)
        ttk.Button(tools, text="Undo", command=lambda: self.canvas.undo()).pack(side="left")
        ttk.Button(tools, text="Next >", command=lambda: self.step(1)).pack(side="right")
        ttk.Button(tools, text="< Previous", command=lambda: self.step(-1)).pack(side="right", padx=4)
        self.canvas = AnnotationCanvas(mid, self.cfg.taxonomy.classes, on_change=self._boxes_changed,
                                       on_select=self._on_select, on_navigate=self.step)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        ttk.Label(mid, text="Drag to draw a box  ·  click a box to select, drag to move, handles to resize  ·  "
                            "Delete removes  ·  1-9 set the clip type  ·  wheel zooms, right-drag pans  ·  "
                            "changes are saved automatically", style="Muted.TLabel").grid(row=2, column=0, sticky="w")

        # -- train ----------------------------------------------------------------
        right = card(self, "Train the model")
        right.panel.grid(row=0, column=2, sticky="nse", padx=(8, 0))
        right.panel.configure(width=360)
        self.stats = ttk.Label(right, text="", style="Card.TLabel", justify="left", font=style.MONO)
        self.stats.pack(anchor="w")
        row = ttk.Frame(right, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 4))
        ttk.Label(row, text="Epochs", style="Card.TLabel").pack(side="left")
        self.epochs = tk.IntVar(value=self.cfg.training.epochs)
        ttk.Spinbox(row, from_=5, to=500, increment=10, textvariable=self.epochs, width=6).pack(side="left", padx=6)
        self.train_btn = ttk.Button(row, text="Train", style="Accent.TButton", command=self.train)
        self.train_btn.pack(side="left", padx=(6, 0))
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(right, maximum=100)
        self.progress.pack(fill="x", pady=4)
        self.log = text_box(right, height=18, width=46)
        self.log.pack(fill="both", expand=True)
        self.model_info = ttk.Label(right, text="", style="CardMuted.TLabel", wraplength=340, justify="left")
        self.model_info.pack(anchor="w", pady=(6, 0))

        self.clip.set(self.cfg.taxonomy.classes[0])
        self.canvas.set_class(self.clip.get())
        self.refresh()

    # -- list -----------------------------------------------------------------------
    @property
    def store(self):
        return self.station.dataset

    def refresh(self) -> None:
        rows = [(r.id, (r.source_name or r.id, len(r.boxes) or "-", r.part or ""), () if r.boxes else ("muted",))
                for r in self.store.records()]
        self.table.set_rows(rows)
        s = self.store.stats()
        per = "\n".join(f"  {k:<18}{v:>4}" for k, v in s["per_class"].items()) or "  none yet"
        self.stats.configure(text=f"Images  {s['images']}   marked  {s['marked']}\nClips   {s['boxes']}\n{per}")
        self._show_model_info()
        if self.current is None and rows:
            self.table.select(rows[0][0])

    def on_show(self) -> None:
        self.store.reload()
        self.refresh()

    def _show_model_info(self) -> None:
        path = self.cfg.resolve(self.cfg.detector.model_path)
        rep = model_report(path)
        det = self.station.inspector.detector
        if rep is None:
            text = "No trained model yet." if not path.is_file() else f"Model {path.name} (no report)."
        else:
            text = (f"Current model: {rep.get('accuracy', 0):.1%} validation accuracy on {rep.get('images', '?')} "
                    f"images ({rep.get('epochs', '?')} epochs).")
        self.model_info.configure(text=f"{text}\nIn use: {det.name} - {det.note}")

    # -- images ---------------------------------------------------------------------
    def open_image(self, image_id: str | None) -> None:
        if not image_id or image_id not in self.store:
            return
        self.current = image_id
        try:
            img = self.store.load_image(image_id)
        except Exception as exc:
            show_error(self, f"Cannot read the image: {exc}")
            return
        self.canvas.set_image(img, self.store.get(image_id).boxes)

    def step(self, d: int) -> None:
        ids = list(self.table.tree.get_children())
        if not ids:
            return
        i = ids.index(self.current) + d if self.current in ids else 0
        self.table.select(ids[max(0, min(len(ids) - 1, i))])

    def _add(self, paths: list[Path]) -> None:
        added = dup = 0
        errors = []
        for p in paths:
            try:
                _id, new = self.store.add_image(p)
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")
                continue
            added += new
            dup += not new
        self.refresh()
        msg = f"Added {added} image(s)." + (f" {dup} were already in the set." if dup else "")
        if errors:
            msg += "\n\nSkipped:\n" + "\n".join(errors[:10])
        show_info(self, msg, "Images added")

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="Add training images",
                                            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if paths:
            self._add([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Folder with training images")
        if folder:
            self._add(sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))

    def add_capture(self) -> None:
        try:
            img = self.station.source.capture(timeout=5)
        except Exception as exc:
            show_error(self, f"Camera: {exc}")
            return
        image_id, _new = self.store.add_image(img, name="camera")
        self.refresh()
        self.table.select(image_id)

    def delete_image(self) -> None:
        if not self.current or self.current not in self.store:
            return
        used = [code for code, pn in self.cfg.parts.items() if pn.master_image == self.current]
        if used:
            show_error(self, f"This image is the master photo of {', '.join(used)}. Delete or remake that part first.")
            return
        if not ask_yes_no(self, "Delete this image and its marked clips from the training set?"):
            return
        self.store.delete(self.current)
        self.current = None
        self.canvas.set_image(None, [])
        self.refresh()

    # -- marking ------------------------------------------------------------------------
    def _boxes_changed(self, boxes: list[Box]) -> None:
        if self.current and self.current in self.store:
            self.store.set_boxes(self.current, boxes)
            n = len(boxes)
            if self.table.tree.exists(self.current):
                vals = list(self.table.tree.item(self.current, "values"))
                vals[1] = n or "-"
                self.table.tree.item(self.current, values=vals, tags=() if n else ("muted",))

    def _on_select(self, box: Box | None) -> None:
        if box is not None:
            self.clip.set(box.label)

    def auto_mark(self) -> None:
        """Pre-mark the image with the current detector (keeps nothing that was marked before)."""
        if self.canvas.image is None or not self.current:
            return
        insp = self.station.inspector
        if not insp.detector.ready:
            show_error(self, "There is no trained model yet. Mark a few images by hand and train first.")
            return
        if self.canvas.boxes and not ask_yes_no(self, "Replace the boxes on this image with the clips the model finds?"):
            return
        img, thr = self.canvas.image, self.cfg.detector.confidence_threshold
        image_id = self.current

        def done(dets):
            if self.current == image_id:
                self.canvas.set_boxes([Box(d.label, d.x, d.y, d.w, d.h) for d in dets if d.confidence >= thr])

        self.dispatcher.run_task(lambda: insp.detect(img), done, lambda e: show_error(self, f"Detector failed: {e}"))

    # -- training -------------------------------------------------------------------------
    def train(self) -> None:
        if self.training:
            return
        s = self.store.stats()
        if s["marked"] < 2:
            show_error(self, "Mark the clips on at least 2 images first (more is better: 20+).")
            return
        try:
            epochs = int(self.epochs.get())
        except (tk.TclError, ValueError):
            show_error(self, "Epochs must be a number.")
            return
        from ...vision.training import train_detector

        self.training = True
        self._stop = threading.Event()
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0, maximum=epochs)
        set_text(self.log, f"Training on {s['marked']} images, up to {epochs} epochs. This takes a while on a CPU.\n")

        def progress(msg: str) -> None:
            self.dispatcher.call(self._progress, msg)

        stop = self._stop
        self.dispatcher.run_task(lambda: train_detector(self.cfg, self.store, progress, stop, epochs=epochs),
                                 self._trained, self._train_failed)

    def _progress(self, msg: str) -> None:
        if msg.startswith("epoch "):
            try:
                self.progress.configure(value=int(msg.split()[1].split("/")[0]))
            except (ValueError, IndexError):
                pass
        set_text(self.log, msg + "\n", append=True)

    def _finish(self) -> None:
        self.training = False
        self._stop = None
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _trained(self, res) -> None:
        self._finish()
        acc = res.evaluation.accuracy
        target = self.cfg.detector.min_model_accuracy
        set_text(self.log, f"\nDone: {res.epochs_run} epochs in {res.seconds / 60:.0f} min, "
                           f"validation accuracy {acc:.1%}.\n", append=True)
        self.app.rebuild_detector(then=self.refresh)
        if acc < target:
            show_info(self, f"The model reached {acc:.1%} (target {target:.0%}). The station keeps the previous "
                            "detector. Mark more images or train longer.", "Training finished")
        else:
            show_info(self, f"The new model reached {acc:.1%} and is now in use.", "Training finished")

    def _train_failed(self, exc: BaseException) -> None:
        self._finish()
        set_text(self.log, f"\nTraining failed: {exc}\n", append=True)
        show_error(self, f"Training failed: {exc}")

    def stop_training(self) -> None:
        if self._stop is not None:
            self._stop.set()
            set_text(self.log, "Stopping after the current batch ...\n", append=True)

    def stop(self) -> None:
        self.stop_training()
