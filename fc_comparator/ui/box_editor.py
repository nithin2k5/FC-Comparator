"""Box editor: draw, select, move, resize and relabel clip boxes on an image (a tk.Canvas).

Mouse
    left-drag on empty area   draw a new box with the current clip type
    click a box               select it; drag to move, drag a handle to resize
    wheel                     zoom around the cursor
    right- or middle-drag     pan
Keys (canvas focused)
    1..9       clip type of the selected box / of new boxes
    Delete     remove the selected box
    Ctrl+Z     undo
    arrows     nudge the selected box (Shift = 10 px); without a selection:
               Left/Right go to the previous/next image
    F          fit image to window
    Esc        deselect
"""

from __future__ import annotations

import dataclasses
import math
import tkinter as tk
from collections.abc import Callable

import cv2
import numpy as np

from ..core.models import Box
from . import style
from .widgets import to_photo

HANDLE = 6  # half size of a resize handle in screen pixels
MIN_BOX = 4  # smallest box in image pixels
HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


class AnnotationCanvas(tk.Canvas):
    def __init__(
        self,
        master,
        classes: list[str],
        on_change: Callable[[list[Box]], None] | None = None,
        on_select: Callable[[Box | None], None] | None = None,
        on_navigate: Callable[[int], None] | None = None,
    ):
        super().__init__(master, bg=style.CANVAS_BG, highlightthickness=0, cursor="crosshair", takefocus=1)
        self.classes = list(classes)
        self.current_class = self.classes[0] if self.classes else ""
        self.on_change = on_change
        self.on_select = on_select
        self.on_navigate = on_navigate
        self.readonly = False
        self.image: np.ndarray | None = None
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.captions: dict[int, str] = {}  # extra text per box index (e.g. "C1 R2" when previewing a master)
        self.scale = 1.0
        self.ox = self.oy = 0.0  # image coordinates at the canvas origin
        self._photo = None
        self._undo: list[list[Box]] = []
        self._drag: dict | None = None
        self._pan: tuple | None = None
        self._fitted = True  # keep the image fitted on resize until the user zooms or pans
        self._placeholder = "Add images, then select one to mark its clips"

        self.bind("<Configure>", lambda _e: self.fit() if self._fitted else self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        for b in ("2", "3"):
            self.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.bind(f"<B{b}-Motion>", self._pan_move)
        self.bind("<MouseWheel>", lambda e: self._zoom(e, 1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda e: self._zoom(e, 1))
        self.bind("<Button-5>", lambda e: self._zoom(e, -1))
        self.bind("<Enter>", lambda _e: self.focus_set())
        self.bind("<Delete>", lambda _e: self.delete_selected())
        self.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.bind("<Escape>", lambda _e: self._select(None))
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-Z>", lambda _e: self.undo())
        self.bind("<Key-f>", lambda _e: self.fit())
        for key, (dx, dy) in {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}.items():
            self.bind(f"<{key}>", lambda _e, d=(dx, dy): self._arrow(d, 1))
            self.bind(f"<Shift-{key}>", lambda _e, d=(dx, dy): self._arrow(d, 10))
        for i in range(1, 10):
            self.bind(f"<Key-{i}>", lambda _e, n=i: self._digit(n))

    # -- data ------------------------------------------------------------------
    def set_classes(self, classes: list[str]) -> None:
        self.classes = list(classes)
        if self.current_class not in self.classes:
            self.current_class = self.classes[0] if self.classes else ""
        self.redraw()

    def set_image(self, image: np.ndarray | None, boxes: list[Box], placeholder: str = "") -> None:
        self.image = image
        self.boxes = [dataclasses.replace(b) for b in boxes]
        self.selected = None
        self.captions = {}
        self._undo = []
        if placeholder:
            self._placeholder = placeholder
        self.fit()
        if self.on_select:
            self.on_select(None)

    def set_class(self, label: str) -> None:
        self.current_class = label
        if self.selected is not None and not self.readonly and self.boxes[self.selected].label != label:
            self._push_undo()
            self.boxes[self.selected].label = label
            self._changed()
        self.redraw()

    def set_boxes(self, boxes: list[Box], undoable: bool = True) -> None:
        if undoable:
            self._push_undo()
        self.boxes = [dataclasses.replace(b) for b in boxes]
        self.selected = None
        self._changed()

    def delete_selected(self) -> None:
        if self.selected is None or self.readonly:
            return
        self._push_undo()
        del self.boxes[self.selected]
        self.selected = None
        self._changed()
        if self.on_select:
            self.on_select(None)

    def undo(self) -> None:
        if self._undo and not self.readonly:
            self.boxes = self._undo.pop()
            self.selected = None
            self._changed()

    # -- geometry ----------------------------------------------------------
    def fit(self) -> None:
        if self.image is None:
            self.redraw()
            return
        self._fitted = True
        ih, iw = self.image.shape[:2]
        w, h = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        self.scale = min(w / iw, h / ih) * 0.98
        self.ox = -(w / self.scale - iw) / 2
        self.oy = -(h / self.scale - ih) / 2
        self.redraw()

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def to_image(self, cx: float, cy: float) -> tuple[float, float]:
        return cx / self.scale + self.ox, cy / self.scale + self.oy

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        ih, iw = self.image.shape[:2]
        return min(max(x, 0.0), float(iw)), min(max(y, 0.0), float(ih))

    # -- drawing -------------------------------------------------------------
    def redraw(self) -> None:
        self.delete("all")
        w, h = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        if self.image is None:
            self.create_text(w / 2, h / 2, text=self._placeholder, fill="#94a3b8", font=("Segoe UI", 14))
            return
        ih, iw = self.image.shape[:2]
        x0, y0 = max(0, int(math.floor(self.ox))), max(0, int(math.floor(self.oy)))
        x1 = min(iw, int(math.ceil(self.ox + w / self.scale)))
        y1 = min(ih, int(math.ceil(self.oy + h / self.scale)))
        if x1 > x0 and y1 > y0:
            crop = self.image[y0:y1, x0:x1]
            tw, th = max(1, int(round((x1 - x0) * self.scale))), max(1, int(round((y1 - y0) * self.scale)))
            interp = cv2.INTER_AREA if self.scale < 1 else (cv2.INTER_NEAREST if self.scale > 3 else cv2.INTER_LINEAR)
            self._photo, _k = to_photo(cv2.resize(crop, (tw, th), interpolation=interp), tw, th)
            cx, cy = self.to_canvas(x0, y0)
            self.create_image(cx, cy, image=self._photo, anchor="nw")

        for i, b in enumerate(self.boxes):
            color = style.class_color(self.classes, b.label)
            sel = i == self.selected
            ax, ay = self.to_canvas(b.x, b.y)
            bx, by = self.to_canvas(b.x + b.w, b.y + b.h)
            self.create_rectangle(ax, ay, bx, by, outline=color, width=3 if sel else 2)
            if sel:
                self.create_rectangle(ax, ay, bx, by, outline="#ffffff", width=1, dash=(3, 3))
            caption = b.label + (f"  {self.captions[i]}" if i in self.captions else "")
            if b.confidence < 1.0:
                caption += f" {b.confidence:.0%}"
            tid = self.create_text(ax + 4, ay - 3, text=caption, anchor="sw", fill="#ffffff", font=("Segoe UI", 9, "bold"))
            tb = self.bbox(tid)
            if tb:
                rid = self.create_rectangle(tb[0] - 3, tb[1] - 1, tb[2] + 3, tb[3] + 1, fill=color, outline=color)
                self.tag_lower(rid, tid)
            if sel and not self.readonly:
                for hx, hy in self._handle_points(b).values():
                    self.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill="#ffffff", outline=color, width=2)

    def _handle_points(self, b: Box) -> dict[str, tuple[float, float]]:
        ax, ay = self.to_canvas(b.x, b.y)
        bx, by = self.to_canvas(b.x + b.w, b.y + b.h)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        return {"nw": (ax, ay), "n": (mx, ay), "ne": (bx, ay), "e": (bx, my),
                "se": (bx, by), "s": (mx, by), "sw": (ax, by), "w": (ax, my)}

    # -- hit testing -----------------------------------------------------------
    def _handle_at(self, cx: float, cy: float) -> str | None:
        if self.selected is None:
            return None
        for name, (hx, hy) in self._handle_points(self.boxes[self.selected]).items():
            if abs(cx - hx) <= HANDLE + 2 and abs(cy - hy) <= HANDLE + 2:
                return name
        return None

    def _box_at(self, cx: float, cy: float) -> int | None:
        x, y = self.to_image(cx, cy)
        hits = [i for i, b in enumerate(self.boxes) if b.x <= x <= b.x + b.w and b.y <= y <= b.y + b.h]
        return min(hits, key=lambda i: self.boxes[i].w * self.boxes[i].h) if hits else None  # smallest on top

    # -- mouse -------------------------------------------------------------
    def _press(self, e) -> None:
        self.focus_set()
        if self.image is None:
            return
        handle = self._handle_at(e.x, e.y) if not self.readonly else None
        if handle:
            self._drag = {"mode": "resize", "handle": handle, "orig": dataclasses.replace(self.boxes[self.selected]),
                          "start": self.to_image(e.x, e.y), "snapshot": self._snapshot()}
            return
        hit = self._box_at(e.x, e.y)
        if hit is not None:
            self._select(hit)
            if not self.readonly:
                self._drag = {"mode": "move", "orig": dataclasses.replace(self.boxes[hit]),
                              "start": self.to_image(e.x, e.y), "snapshot": self._snapshot()}
            return
        self._select(None)
        if not self.readonly and self.current_class:
            self._drag = {"mode": "draw", "start": self._clamp(*self.to_image(e.x, e.y)), "rubber": None}

    def _motion(self, e) -> None:
        d = self._drag
        if d is None or self.image is None:
            return
        x, y = self._clamp(*self.to_image(e.x, e.y))
        if d["mode"] == "draw":
            if d["rubber"] is not None:
                self.delete(d["rubber"])
            ax, ay = self.to_canvas(*d["start"])
            bx, by = self.to_canvas(x, y)
            d["rubber"] = self.create_rectangle(ax, ay, bx, by, outline=style.class_color(self.classes, self.current_class),
                                                width=2, dash=(6, 4))
            return
        o: Box = d["orig"]
        sx, sy = d["start"]
        b = self.boxes[self.selected]
        ih, iw = self.image.shape[:2]
        if d["mode"] == "move":
            b.x = min(max(o.x + x - sx, 0.0), iw - o.w)
            b.y = min(max(o.y + y - sy, 0.0), ih - o.h)
        else:
            x0, y0, x1, y1 = o.x, o.y, o.x + o.w, o.y + o.h
            hdl = d["handle"]
            if "w" in hdl:
                x0 = min(x, x1 - MIN_BOX)
            if "e" in hdl:
                x1 = max(x, x0 + MIN_BOX)
            if "n" in hdl:
                y0 = min(y, y1 - MIN_BOX)
            if "s" in hdl:
                y1 = max(y, y0 + MIN_BOX)
            b.x, b.y, b.w, b.h = x0, y0, x1 - x0, y1 - y0
        self.redraw()

    def _release(self, e) -> None:
        d, self._drag = self._drag, None
        if d is None or self.image is None:
            return
        if d["mode"] == "draw":
            (ax, ay), (bx, by) = d["start"], self._clamp(*self.to_image(e.x, e.y))
            w, h = abs(bx - ax), abs(by - ay)
            if w >= MIN_BOX and h >= MIN_BOX and w * self.scale >= 6 and h * self.scale >= 6:
                self._push_undo()
                self.boxes.append(Box(self.current_class, min(ax, bx), min(ay, by), w, h))
                # Not left selected: choosing the type for the *next* box must not relabel this one.
                self.selected = None
                self._changed()
                if self.on_select:
                    self.on_select(None)
            else:
                self.redraw()
        elif self.selected is not None and self.boxes[self.selected] != d["orig"]:
            self._undo.append(d["snapshot"])
            self._changed()

    def _pan_start(self, e) -> None:
        self._pan = (e.x, e.y, self.ox, self.oy)
        self._fitted = False
        self.configure(cursor="fleur")

    def _pan_move(self, e) -> None:
        if self._pan is None:
            return
        x, y, ox, oy = self._pan
        self.ox = ox - (e.x - x) / self.scale
        self.oy = oy - (e.y - y) / self.scale
        self.redraw()

    def _zoom(self, e, steps: int) -> None:
        if self.image is None:
            return
        self._fitted = False
        ih, iw = self.image.shape[:2]
        fit = min(max(self.winfo_width(), 50) / iw, max(self.winfo_height(), 50) / ih)
        ix, iy = self.to_image(e.x, e.y)
        self.scale = min(max(self.scale * (1.2 ** steps), fit * 0.5), 12.0)
        self.ox, self.oy = ix - e.x / self.scale, iy - e.y / self.scale
        self.configure(cursor="crosshair")
        self.redraw()

    # -- keys ----------------------------------------------------------------
    def _digit(self, n: int) -> None:
        if n <= len(self.classes):
            self.set_class(self.classes[n - 1])
            if self.on_select:
                self.on_select(self.boxes[self.selected] if self.selected is not None else None)

    def _arrow(self, d: tuple[int, int], step: int) -> None:
        if self.selected is None:
            if d[1] == 0 and self.on_navigate:
                self.on_navigate(d[0])
            return
        if self.readonly:
            return
        self._push_undo()
        b = self.boxes[self.selected]
        ih, iw = self.image.shape[:2]
        b.x = min(max(b.x + d[0] * step, 0.0), iw - b.w)
        b.y = min(max(b.y + d[1] * step, 0.0), ih - b.h)
        self._changed()

    # -- internals -------------------------------------------------------------
    def _select(self, idx: int | None) -> None:
        self.selected = idx
        if idx is not None:
            self.current_class = self.boxes[idx].label if self.boxes[idx].label in self.classes else self.current_class
        self.redraw()
        if self.on_select:
            self.on_select(self.boxes[idx] if idx is not None else None)

    def _snapshot(self) -> list[Box]:
        return [dataclasses.replace(b) for b in self.boxes]

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        del self._undo[:-100]

    def _changed(self) -> None:
        self.redraw()
        if self.on_change:
            self.on_change([dataclasses.replace(b) for b in self.boxes])
