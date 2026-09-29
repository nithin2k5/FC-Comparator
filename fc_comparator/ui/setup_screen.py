"""Setup / teach screen: reference image, ROI drawing, labeled crops, alignment."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk

import numpy as np

from ..capture.sources import read_image
from ..classify import crop_roi
from ..classify.dataset import dataset_counts, dataset_warnings, harvest_board, save_crop
from ..config import save_config
from ..models import CLASSIFIER_LABELS, FORK, Roi
from ..pipeline import Station
from .widgets import DARK_BG, ScrollFrame, confirm, error_box, fit, info_box, to_photo

log = logging.getLogger(__name__)


class RoiCanvas(tk.Canvas):
    """Displays an image with ROI boxes; drag to draw a box, tap to select one."""

    def __init__(self, master, on_drawn: Callable[[int, int, int, int], None], on_clicked: Callable[[object], None]):
        super().__init__(master, bg=DARK_BG, highlightthickness=0, width=640, height=480)
        self.on_drawn = on_drawn
        self.on_clicked = on_clicked
        self.image: np.ndarray | None = None
        self.rois: dict[tuple[int, int], Roi] = {}
        self.selected: tuple[int, int] | None = None
        self._photo = None
        self._geom = (1.0, 0.0, 0.0)  # scale, offset x, offset y
        self._start: tuple[float, float] | None = None
        self._rubber = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._release)

    def set_image(self, img: np.ndarray) -> None:
        self.image = img
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        if self.image is None:
            self.create_text(w / 2, h / 2, text="Capture or load an image", fill="#a0aec0", font=("", 18))
            return
        small, s = fit(self.image, w, h)
        ox, oy = (w - small.shape[1]) / 2, (h - small.shape[0]) / 2
        self._geom = (s, ox, oy)
        self._photo = to_photo(small)
        self.create_image(ox, oy, image=self._photo, anchor="nw")
        for key, r in self.rois.items():
            sel = key == self.selected
            color = "#f6e05e" if sel else "#00d1ff"
            x0, y0 = ox + r.x * s, oy + r.y * s
            self.create_rectangle(x0, y0, x0 + r.w * s, y0 + r.h * s, outline=color, width=3 if sel else 2)
            self.create_text(x0 + 4, y0 + 4, text=f"C{r.cable}R{r.row}", anchor="nw", fill=color, font=("", 11, "bold"))

    def _to_image(self, x: float, y: float) -> tuple[float, float]:
        s, ox, oy = self._geom
        ih, iw = self.image.shape[:2]
        return min(max((x - ox) / s, 0), iw - 1), min(max((y - oy) / s, 0), ih - 1)

    def _press(self, e) -> None:
        if self.image is not None:
            self._start = (e.x, e.y)
            self._rubber = self.create_rectangle(e.x, e.y, e.x, e.y, outline="#f6e05e", dash=(6, 4), width=2)

    def _move(self, e) -> None:
        if self._start is not None and self._rubber is not None:
            self.coords(self._rubber, *self._start, e.x, e.y)

    def _release(self, e) -> None:
        if self._start is None:
            return
        a, b = self._to_image(*self._start), self._to_image(e.x, e.y)
        self._start = None
        x0, y0 = int(min(a[0], b[0])), int(min(a[1], b[1]))
        w, h = int(abs(a[0] - b[0])), int(abs(a[1] - b[1]))
        if w >= 8 and h >= 8:
            self.on_drawn(x0, y0, w, h)
        else:  # a tap: select the ROI under the finger
            hit = next((k for k, r in self.rois.items() if r.x <= a[0] <= r.x + r.w and r.y <= a[1] <= r.y + r.h), None)
            self.on_clicked(hit)
        self.redraw()


class SetupScreen(ttk.Frame):
    def __init__(self, master, station: Station, on_config_changed: Callable[[], None]):
        super().__init__(master)
        self.station = station
        self.cfg = station.cfg
        self.on_config_changed = on_config_changed
        self.dirty = False

        self.canvas = RoiCanvas(self, self._on_drawn, self._on_clicked)
        self.canvas.rois = self.cfg.roi_map()
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 8))

        panel = ScrollFrame(self, width=470)
        panel.pack(side="right", fill="y")
        body = panel.body

        # 1. image
        g = ttk.LabelFrame(body, text="1. Image", padding=8)
        g.pack(fill="x", pady=(0, 8))
        ttk.Button(g, text="Capture from camera", command=self.capture).pack(fill="x", pady=2)
        ttk.Button(g, text="Load image file...", command=self.load).pack(fill="x", pady=2)
        ttk.Button(g, text="Save as reference image", command=self.save_reference).pack(fill="x", pady=2)
        self.align_var = tk.BooleanVar(value=self.cfg.alignment.enabled)
        ttk.Checkbutton(g, text="Align images to the reference", variable=self.align_var,
                        command=self._toggle_align).pack(anchor="w", pady=4)
        self.img_info = ttk.Label(g, wraplength=420, justify="left")
        self.img_info.pack(fill="x")

        # 2. ROIs
        g = ttk.LabelFrame(body, text="2. Inspection positions (ROIs)", padding=8)
        g.pack(fill="x", pady=(0, 8))
        ttk.Label(g, text="Drag a box on the image for the selected position:", wraplength=420).pack(anchor="w")
        self.target_var = tk.StringVar()
        self.target = ttk.Combobox(g, textvariable=self.target_var, state="readonly")
        self.target.bind("<<ComboboxSelected>>", lambda _e: self._on_target())
        self.target.pack(fill="x", pady=4)
        row = ttk.Frame(g)
        row.pack(fill="x")
        ttk.Button(row, text="Auto grid", command=self.auto_grid).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Delete selected", command=self.delete_selected).pack(side="left", fill="x", expand=True)
        ttk.Label(g, text="Auto grid: draw C1R1 and the last position, the rest are spaced evenly.",
                  style="Muted.TLabel", wraplength=420).pack(anchor="w")
        ttk.Button(g, text="Save ROIs", command=self.save_rois).pack(fill="x", pady=(6, 0))

        # 3. teach
        g = ttk.LabelFrame(body, text="3. Teach clip samples", padding=8)
        g.pack(fill="x", pady=(0, 8))
        self.label_var = tk.StringVar(value="round")
        ttk.Combobox(g, textvariable=self.label_var, state="readonly",
                     values=[lbl for lbl in CLASSIFIER_LABELS if lbl != FORK]).pack(fill="x", pady=2)
        ttk.Button(g, text="Save crop of selected position as label", command=self.save_selected_crop).pack(fill="x", pady=2)
        ttk.Label(g, text="Known-good board of part:").pack(anchor="w", pady=(8, 0))
        hl = ttk.Frame(g)
        hl.pack(fill="x")
        self.harvest_var = tk.StringVar()
        self.harvest_part = ttk.Combobox(hl, textvariable=self.harvest_var, state="readonly", width=10)
        self.harvest_part.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.fork_var = tk.StringVar(value="forks face left")
        ttk.Combobox(hl, textvariable=self.fork_var, state="readonly", width=16,
                     values=["forks face left", "forks face right"]).pack(side="left")
        ttk.Button(g, text="Save all positions from this board", command=self.harvest).pack(fill="x", pady=4)
        self.ds_info = ttk.Label(g, wraplength=420, justify="left")
        self.ds_info.pack(fill="x")
        ttk.Button(g, text="Reload classifier with new samples", command=self.reload_classifier).pack(fill="x", pady=(6, 0))

        self._fill_targets()
        self.refresh()

    # -- helpers -----------------------------------------------------------
    def refresh(self) -> None:
        self.harvest_part.configure(values=sorted(self.cfg.parts))
        if self.harvest_var.get() not in self.cfg.parts:
            self.harvest_var.set(next(iter(sorted(self.cfg.parts)), ""))
        self._update_dataset_info()
        ref = self.cfg.resolve(self.cfg.reference_image)
        if self.canvas.image is None and ref.is_file():
            self.canvas.set_image(read_image(ref))
            self.img_info.configure(text=f"Showing reference image {ref.name}")

    def _keys(self) -> list[tuple[int, int]]:
        return [(c, r) for c in range(1, self.cfg.station.cables + 1) for r in range(1, self.cfg.station.rows + 1)]

    def _fill_targets(self, select: int | None = None) -> None:
        keys = self._keys()
        values = [f"{'✓' if k in self.canvas.rois else '  '} Cable {k[0]}  Row {k[1]}" for k in keys]
        self.target.configure(values=values)
        idx = select if select is not None else (self.target.current() if self.target.current() >= 0 else 0)
        self.target.current(min(idx, len(values) - 1))
        self._on_target()

    def target_key(self) -> tuple[int, int] | None:
        i = self.target.current()
        return self._keys()[i] if i >= 0 else None

    def _on_target(self) -> None:
        self.canvas.selected = self.target_key()
        self.canvas.redraw()

    def _on_drawn(self, x: int, y: int, w: int, h: int) -> None:
        key = self.target_key()
        if key is None:
            return
        self.canvas.rois[key] = Roi(key[0], key[1], x, y, w, h)
        self.dirty = True
        self._fill_targets(select=self.target.current() + 1)  # advance to the next position

    def _on_clicked(self, key) -> None:
        if key is not None:
            self._fill_targets(select=self._keys().index(key))

    def _set_teach_image(self, img: np.ndarray, source: str) -> None:
        """Show an image for teaching, aligned to the reference so the ROIs fit."""
        aligner = self.station.inspector.aligner
        msg = source
        if self.cfg.alignment.enabled and aligner is not None:
            img, info = aligner.align(img)
            msg += f" - alignment: {info.message} ({info.shift_px:.1f}px)"
        self.canvas.set_image(img)
        self.img_info.configure(text=msg)

    def _update_dataset_info(self) -> None:
        ds = self.cfg.resolve(self.cfg.classifier.dataset_dir)
        counts = dataset_counts(ds)
        text = "Samples: " + (", ".join(f"{k} {v}" for k, v in counts.items()) or "none yet")
        warns = dataset_warnings(ds)
        if warns:
            text += "\n⚠ " + "\n⚠ ".join(warns)
        text += f"\nClassifier in use: {self.station.inspector.classifier.name}"
        self.ds_info.configure(text=text)

    # -- actions -----------------------------------------------------------
    def capture(self) -> None:
        try:
            img = self.station.source.capture(timeout=3)
        except Exception as exc:
            error_box(self, f"Capture failed: {exc}")
            return
        self._set_teach_image(img, "Captured from camera")

    def load(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Load image",
                                          filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if path:
            try:
                self._set_teach_image(read_image(path), f"Loaded {path}")
            except Exception as exc:
                error_box(self, str(exc))

    def save_reference(self) -> None:
        if self.canvas.image is None:
            return
        if not confirm(self, "Use the displayed image as the new reference?\nROIs are defined on the reference image."):
            return
        path = self.station.save_reference(self.canvas.image)
        self.station.reload()
        self.img_info.configure(text=f"Reference saved to {path}")
        self.on_config_changed()

    def _toggle_align(self) -> None:
        self.cfg.alignment.enabled = self.align_var.get()
        save_config(self.cfg)
        self.station.reload()
        self.on_config_changed()

    def auto_grid(self) -> None:
        n, m = self.cfg.station.cables, self.cfg.station.rows
        first, last = self.canvas.rois.get((1, 1)), self.canvas.rois.get((n, m))
        if first is None or last is None:
            error_box(self, f"Draw C1R1 and C{n}R{m} first.")
            return
        dx = (last.x - first.x) / (n - 1) if n > 1 else 0
        dy = (last.y - first.y) / (m - 1) if m > 1 else 0
        for c in range(1, n + 1):
            for r in range(1, m + 1):
                self.canvas.rois[(c, r)] = Roi(c, r, round(first.x + (c - 1) * dx), round(first.y + (r - 1) * dy), first.w, first.h)
        self.dirty = True
        self._fill_targets()

    def delete_selected(self) -> None:
        key = self.target_key()
        if key in self.canvas.rois:
            del self.canvas.rois[key]
            self.dirty = True
            self._fill_targets()

    def save_rois(self) -> None:
        rois = sorted(self.canvas.rois.values(), key=lambda r: (r.cable, r.row))
        old = self.cfg.rois
        self.cfg.rois = rois
        problems = [p for p in self.cfg.validate() if "ROI" in p]
        if problems and not confirm(self, "Problems:\n" + "\n".join(problems) + "\n\nSave anyway?"):
            self.cfg.rois = old
            return
        save_config(self.cfg)
        self.station.reload()
        self.station.store.log_event("setup", "", f"ROIs saved ({len(rois)})")
        self.dirty = False
        info_box(self, f"Saved {len(rois)} ROIs.")
        self.on_config_changed()

    def save_selected_crop(self) -> None:
        key = self.target_key()
        roi = self.canvas.rois.get(key) if key else None
        if self.canvas.image is None or roi is None:
            error_box(self, "Select a position that has an ROI (and an image).")
            return
        label = self.label_var.get()
        crop = crop_roi(self.canvas.image, roi, self.cfg.classifier.roi_padding)
        path = save_crop(self.cfg.resolve(self.cfg.classifier.dataset_dir), label, crop, f"c{roi.cable}r{roi.row}")
        self.img_info.configure(text=f"Saved {label} sample: {path.name}")
        self._update_dataset_info()

    def harvest(self) -> None:
        part = self.cfg.parts.get(self.harvest_var.get())
        if self.canvas.image is None or part is None:
            error_box(self, "Load a board image and pick its part number.")
            return
        orient = "left" if self.fork_var.get().endswith("left") else "right"
        if not confirm(self, f"Is every clip on this board correct for {part.code}?\nAll positions will be saved as samples."):
            return
        paths = harvest_board(
            self.canvas.image, list(self.canvas.rois.values()), part.pattern,
            self.cfg.resolve(self.cfg.classifier.dataset_dir),
            fork_orientation=orient, stem=part.code, padding=self.cfg.classifier.roi_padding,
        )
        self.station.store.log_event("teach", "", f"harvested {len(paths)} crops from {part.code}")
        self.img_info.configure(text=f"Saved {len(paths)} samples from {part.code}")
        self._update_dataset_info()

    def reload_classifier(self) -> None:
        try:
            self.station.reload()
        except Exception as exc:
            error_box(self, f"Reload failed: {exc}")
            return
        self._update_dataset_info()
        info_box(self, f"Classifier reloaded: {self.station.inspector.classifier.name}")
