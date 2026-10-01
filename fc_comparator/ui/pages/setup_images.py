"""Model Setup - images & labels of the selected part.

b. Add images: files, a folder, or frames captured from the live camera preview.
   They are stored under the part (``<parts_dir>/<code>/images``).
c. Label objects: draw a box around each object the station should check and give it
   a label. Label names are the part's own (``red_clip``, ``fork_left``, ``tape`` ...);
   anything left unboxed is ignored. Once a model exists, "Find objects" pre-marks an
   image so only corrections are needed. Labels can be renamed, merged or deleted on
   every image of the part at once.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ...core.models import Box, check_label
from ...vision.camera import IMAGE_EXTENSIONS
from .. import style
from ..box_editor import AnnotationCanvas
from ..widgets import ImageView, Table, ask_string, ask_yes_no, card, center_window, show_error, show_info


class ImagesPanel(ttk.Frame):
    def __init__(self, master, page):
        super().__init__(master, padding=8)
        self.page = page
        self.station = page.station
        self.cfg = page.cfg
        self.part = None
        self.current: str | None = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- b. images -------------------------------------------------------------
        left = card(self, "Images")
        left.panel.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        self.table = Table(left, [("name", "Image", 160), ("boxes", "Boxes", 50)], height=20,
                           on_select=lambda iid: self.open_image(iid))
        self.table.pack(fill="both", expand=True)
        b = ttk.Frame(left, style="Card.TFrame")
        b.pack(fill="x", pady=(8, 0))
        self.buttons = [
            ttk.Button(b, text="Add images...", command=self.add_files),
            ttk.Button(b, text="Add folder...", command=self.add_folder),
            ttk.Button(b, text="Capture from camera...", command=self.open_camera),
            ttk.Button(b, text="Delete image", command=self.delete_image),
        ]
        for i, btn in enumerate(self.buttons):
            btn.pack(fill="x", pady=(0 if i == 0 else 4, 0))

        # -- c. labelling --------------------------------------------------------------
        mid = ttk.Frame(self)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.grid_rowconfigure(1, weight=1)
        mid.grid_columnconfigure(0, weight=1)
        tools = ttk.Frame(mid)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(tools, text="Label").pack(side="left")
        self.label = ttk.Combobox(tools, values=[], state="readonly", width=18)
        self.label.pack(side="left", padx=6)
        self.label.bind("<<ComboboxSelected>>", lambda _e: self.canvas.set_class(self.label.get()))
        ttk.Button(tools, text="New label...", command=self.new_label).pack(side="left")
        ttk.Button(tools, text="Labels...", command=self.manage_labels).pack(side="left", padx=4)
        ttk.Button(tools, text="Find objects", command=self.find_objects).pack(side="left", padx=(8, 0))
        ttk.Button(tools, text="Delete box", command=lambda: self.canvas.delete_selected()).pack(side="left", padx=4)
        ttk.Button(tools, text="Undo", command=lambda: self.canvas.undo()).pack(side="left")
        ttk.Button(tools, text="Next >", command=lambda: self.step(1)).pack(side="right")
        ttk.Button(tools, text="< Previous", command=lambda: self.step(-1)).pack(side="right", padx=4)
        self.canvas = AnnotationCanvas(mid, [], on_change=self._boxes_changed, on_select=self._on_select,
                                       on_navigate=self.step, on_need_label=self.new_label)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        ttk.Label(mid, text="Drag to draw a box around an object, then pick its label  ·  leave anything you don't "
                            "care about unboxed  ·  click a box to select, drag to move, handles to resize  ·  "
                            "Delete removes  ·  1-9 pick a label  ·  wheel scrolls (Shift: sideways), Ctrl+wheel zooms, right-drag pans  ·  "
                            "saved automatically", style="Muted.TLabel", wraplength=900,
                  justify="left").grid(row=2, column=0, sticky="w")

        # -- label counts -----------------------------------------------------------------
        right = card(self, "Labels of this part")
        right.panel.grid(row=0, column=2, sticky="nse", padx=(8, 0))
        self.counts = Table(right, [("label", "Label", 140), ("n", "Boxes", 60)], height=12,
                            on_select=self._count_selected)
        self.counts.pack(fill="both", expand=True)
        self.summary = ttk.Label(right, text="", style="CardMuted.TLabel", wraplength=220, justify="left")
        self.summary.pack(anchor="w", pady=(8, 0))
        self.set_part(None)

    # -- part ---------------------------------------------------------------------
    @property
    def store(self):
        return self.part.store

    def set_part(self, part) -> None:
        self.part = part
        self.current = None
        self.canvas.set_image(None, [], placeholder="Add images, then select one to label its objects"
                              if part else "Choose a part number first")
        state = "normal" if part else "disabled"
        for btn in self.buttons:
            btn.configure(state=state)
        self.refresh()

    def refresh(self) -> None:
        if self.part is None:
            self.table.set_rows([])
            self.counts.set_rows([])
            self.label.configure(values=[])
            self.label.set("")
            self.canvas.set_classes([])
            self.summary.configure(text="")
            return
        rows = [(r.id, (r.source_name or r.id, len(r.boxes) or "-"), () if r.boxes else ("muted",))
                for r in self.store.records()]
        self.table.set_rows(rows)
        self._refresh_labels()
        if self.current is None and rows:
            self.table.select(rows[0][0])

    def _refresh_labels(self) -> None:
        labels = self.part.labels
        counts = self.part.store.label_counts()
        self.counts.set_rows([(lbl, (lbl, counts[lbl]), () if counts[lbl] else ("muted",)) for lbl in labels])
        self.label.configure(values=labels)
        self.canvas.set_classes(labels)
        if self.label.get() not in labels:
            self.label.set(labels[0] if labels else "")
        if self.canvas.selected is None:  # with a box selected, set_class would relabel it
            self.canvas.set_class(self.label.get())
        s = self.part.store.stats()
        self.summary.configure(text=f"{s['images']} images, {s['marked']} labelled, {s['boxes']} boxes.\n"
                                    "Box every object the station should check, on every image.")

    def on_show(self) -> None:
        if self.part is not None:
            self.refresh()

    # -- images ---------------------------------------------------------------------
    def open_image(self, image_id: str | None) -> None:
        if self.part is None or not image_id or image_id not in self.store:
            return
        self.current = image_id
        try:
            img = self.store.load_image(image_id)
        except Exception as exc:
            show_error(self, f"Cannot read the image: {exc}")
            return
        self.canvas.set_image(img, self.store.get(image_id).boxes)
        if self.label.get():
            self.canvas.set_class(self.label.get())

    def step(self, d: int) -> None:
        ids = list(self.table.tree.get_children())
        if not ids:
            return
        i = ids.index(self.current) + d if self.current in ids else 0
        self.table.select(ids[max(0, min(len(ids) - 1, i))])

    def add_paths(self, paths: list[Path]) -> int:
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
        self.page.refresh()
        msg = f"Added {added} image(s) to {self.part.code}." + (f" {dup} were already there." if dup else "")
        if errors:
            msg += "\n\nSkipped:\n" + "\n".join(errors[:10])
        show_info(self, msg, "Images added")
        return added

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title=f"Add images to {self.part.code}",
                                            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if paths:
            self.add_paths([Path(p) for p in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title=f"Folder with images of {self.part.code}")
        if folder:
            self.add_paths(sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))

    def open_camera(self) -> "CameraDialog":
        return self.page.track(CameraDialog(self))

    def add_frame(self, frame) -> str:
        image_id, _new = self.store.add_image(frame, name="camera")
        self.refresh()
        self.page.refresh()
        self.table.select(image_id)
        return image_id

    def delete_image(self) -> None:
        if self.part is None or not self.current or self.current not in self.store:
            return
        master = self.part.master
        if master is not None and master.master_image == self.current:
            show_error(self, "This image is the part's master photo. Set a new master first.")
            return
        if not ask_yes_no(self, "Delete this image and its boxes?"):
            return
        self.store.delete(self.current)
        self.current = None
        self.canvas.set_image(None, [])
        self.refresh()
        self.page.refresh()

    # -- labelling --------------------------------------------------------------------
    def _boxes_changed(self, boxes: list[Box]) -> None:
        if self.part is None or not self.current or self.current not in self.store:
            return
        self.store.set_boxes(self.current, boxes)
        n = len(boxes)
        if self.table.tree.exists(self.current):
            self.table.tree.item(self.current, values=(self.table.tree.item(self.current, "values")[0], n or "-"),
                                 tags=() if n else ("muted",))
        self._refresh_labels()

    def _on_select(self, box: Box | None) -> None:
        if box is not None:
            self.label.set(box.label)

    def _count_selected(self, label: str | None) -> None:
        if label and self.part is not None and label in self.part.labels:
            self.label.set(label)
            self.canvas.set_class(label)

    def new_label(self) -> str | None:
        """Ask for a new label name (also called when a box is drawn while the part has no labels)."""
        if self.part is None:
            return None
        name = ask_string(self, "New label", "Name of the new label (e.g. red_clip, fork_left, tape):")
        if not name:
            return None
        try:
            name = self.store.add_label(name)
        except ValueError as exc:
            show_error(self, str(exc))
            return None
        self._refresh_labels()
        self.label.set(name)
        self.canvas.current_class = name
        return name

    def manage_labels(self) -> "LabelManager | None":
        if self.part is None:
            return None
        return self.page.track(LabelManager(self))

    def labels_changed(self, detail: str) -> None:
        self.page.log("labels_changed", detail)
        if self.current:
            self.open_image(self.current)
        self.refresh()
        self.page.changed()
        if self.part.active() is not None:
            show_info(self, "The model in use was trained with the old label names. Train again so it learns "
                            "the new ones.", "Labels changed")

    def find_objects(self) -> None:
        """Pre-mark the image with the part's model (replaces the boxes on it)."""
        if self.part is None or self.canvas.image is None or not self.current:
            return
        if self.cfg.detector.backend == "yolo" and self.part.active() is None:
            show_error(self, "This part has no model in use yet. Label a few images by hand and train first.")
            return
        if self.canvas.boxes and not ask_yes_no(self, "Replace the boxes on this image with the objects the model "
                                                      "finds?"):
            return
        img, thr, code, image_id = self.canvas.image, self.cfg.detector.confidence_threshold, self.part.code, self.current

        def run():
            return self.station.detector_for(code).detect(img)

        def done(dets):
            if self.current == image_id:
                self.canvas.set_boxes([Box(d.label, d.x, d.y, d.w, d.h) for d in dets if d.confidence >= thr])

        self.page.dispatcher.run_task(run, done, lambda e: show_error(self, f"Finding objects failed: {e}"))


class LabelManager(tk.Toplevel):
    """Add, rename, merge or delete a part's labels (applied to every image of the part)."""

    def __init__(self, panel: ImagesPanel):
        super().__init__(panel)
        self.panel = panel
        self.part = panel.part
        self.withdraw()
        self.title(f"Labels of {self.part.code}")
        self.configure(bg=style.BG)
        self.transient(panel.winfo_toplevel())
        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Renaming to an existing label merges the two. Changes apply to every image of "
                             f"{self.part.code} and to its master.", style="Muted.TLabel", wraplength=380,
                  justify="left").pack(anchor="w", pady=(0, 8))
        self.table = Table(body, [("label", "Label", 200), ("n", "Boxes", 70)], height=12)
        self.table.pack(fill="both", expand=True)
        row = ttk.Frame(body)
        row.pack(fill="x", pady=(10, 0))
        ttk.Button(row, text="Add...", command=self.add).pack(side="left")
        ttk.Button(row, text="Rename / merge...", command=self.rename).pack(side="left", padx=6)
        ttk.Button(row, text="Delete", command=self.delete).pack(side="left")
        ttk.Button(row, text="Close", command=self.destroy).pack(side="right")
        self.refresh()
        center_window(self, panel, 440, 460)
        self.deiconify()

    def refresh(self) -> None:
        counts = self.part.store.label_counts()
        self.table.set_rows([(lbl, (lbl, n), ()) for lbl, n in counts.items()])

    def add(self) -> None:
        if self.panel.new_label():
            self.refresh()

    def rename(self) -> None:
        old = self.table.selected()
        if not old:
            show_error(self, "Select a label first.")
            return
        new = ask_string(self, "Rename label", f"New name for '{old}' (an existing name merges the two):", old)
        if not new or new.strip() == old:
            return
        try:
            new = check_label(new)
        except ValueError as exc:
            show_error(self, str(exc))
            return
        merge = new in self.part.labels
        if merge and not ask_yes_no(self, f"Merge '{old}' into '{new}' on every image?"):
            return
        n = self.part.rename_label(old, new)
        self.refresh()
        self.panel.labels_changed(f"{'merged' if merge else 'renamed'} {old} -> {new} ({n} boxes)")

    def delete(self) -> None:
        name = self.table.selected()
        if not name:
            show_error(self, "Select a label first.")
            return
        n = self.part.store.label_counts().get(name, 0)
        if not ask_yes_no(self, f"Delete '{name}' and its {n} box(es) on every image?"):
            return
        try:
            self.part.delete_label(name)
        except ValueError as exc:
            show_error(self, str(exc))
            return
        self.refresh()
        self.panel.labels_changed(f"deleted {name} ({n} boxes)")


class CameraDialog(tk.Toplevel):
    """Live camera preview; every "Capture" adds the frame to the part's images."""

    def __init__(self, panel: ImagesPanel):
        super().__init__(panel)
        self.panel = panel
        self.source = panel.station.source
        self.withdraw()
        self.title(f"Capture images of {panel.part.code}")
        self.configure(bg=style.BG)
        self.transient(panel.winfo_toplevel())
        self.view = ImageView(self, placeholder="Waiting for the camera ...")
        self.view.pack(fill="both", expand=True, padx=10, pady=(10, 6))
        row = ttk.Frame(self, padding=(10, 0, 10, 10))
        row.pack(fill="x")
        ttk.Button(row, text="Capture", style="Accent.TButton", command=self.capture).pack(side="left")
        self.status = ttk.Label(row, text="", style="Muted.TLabel")
        self.status.pack(side="left", padx=10)
        ttk.Button(row, text="Close", command=self.destroy).pack(side="right")
        self.captured = 0
        self._job = None
        center_window(self, panel, 900, 680)
        self.deiconify()
        self._tick()

    def _tick(self) -> None:
        try:
            frame = self.source.latest()
            if frame is not None:
                self.view.set_image(frame)
            err = getattr(self.source, "error", "")
            if err:
                self.status.configure(text=f"Camera: {err}")
            self._job = self.after(80, self._tick)
        except tk.TclError:
            pass

    def capture(self) -> str | None:
        try:
            frame = self.source.capture(timeout=5)
        except Exception as exc:
            show_error(self, f"Camera: {exc}")
            return None
        image_id = self.panel.add_frame(frame)
        self.captured += 1
        self.status.configure(text=f"{self.captured} image(s) captured")
        return image_id

    def destroy(self) -> None:
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        super().destroy()
