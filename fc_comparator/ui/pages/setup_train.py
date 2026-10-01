"""Model Setup - train the selected part's model and choose which version is in use.

d. Train in the background (progress bar, Cancel). 20 % of the labelled images are
   held back to score the model; labels with left/right get mirrored copies.
   Needs ``training.min_images`` labelled images and every label
   ``training.min_per_label`` times.
e. A new model that scores at least ``detector.min_model_accuracy`` can be put into
   use ("Use this model"). Otherwise the previous model stays active and the labels
   it struggles with are listed. Every version is kept, so an earlier one can be
   put back into use.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from .. import style
from ..widgets import Table, card, set_text, show_error, show_info, text_box


class TrainPanel(ttk.Frame):
    def __init__(self, master, page):
        super().__init__(master, padding=8)
        self.page = page
        self.station = page.station
        self.cfg = page.cfg
        self.part = None
        self.training = False
        self._stop: threading.Event | None = None
        self.result = None  # ModelInfo of the last training run (this session)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- d. train ------------------------------------------------------------------
        left = card(self, "Train")
        left.panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.readiness = ttk.Label(left, text="", style="Card.TLabel", justify="left", wraplength=520)
        self.readiness.pack(anchor="w")
        row = ttk.Frame(left, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 4))
        ttk.Label(row, text="Epochs", style="Card.TLabel").pack(side="left")
        self.epochs = tk.IntVar(value=self.cfg.training.epochs)
        ttk.Spinbox(row, from_=5, to=500, increment=10, textvariable=self.epochs, width=6).pack(side="left", padx=6)
        self.train_btn = ttk.Button(row, text="Train", style="Accent.TButton", command=self.train)
        self.train_btn.pack(side="left", padx=(6, 0))
        self.cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(left, maximum=100)
        self.progress.pack(fill="x", pady=4)
        self.log = text_box(left, height=18)
        self.log.pack(fill="both", expand=True)

        # -- e. result and versions ---------------------------------------------------------
        right = card(self, "Result and model versions")
        right.panel.grid(row=0, column=1, sticky="nsew")
        self.result_label = ttk.Label(right, text="", style="Card.TLabel", justify="left", wraplength=520)
        self.result_label.pack(anchor="w")
        self.use_btn = ttk.Button(right, text="Use this model", style="Accent.TButton", command=self.use_new,
                                  state="disabled")
        self.use_btn.pack(anchor="w", pady=(8, 12))
        self.versions = Table(right, [("version", "Version", 70), ("trained", "Trained", 130),
                                      ("acc", "Accuracy", 75), ("images", "Images", 60), ("state", "", 110)],
                              height=10)
        self.versions.pack(fill="both", expand=True)
        row = ttk.Frame(right, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        self.switch_btn = ttk.Button(row, text="Use selected version", command=self.use_selected)
        self.switch_btn.pack(side="left")
        ttk.Label(row, text=f"Versions below {self.cfg.detector.min_model_accuracy:.0%} cannot be used.",
                  style="CardMuted.TLabel").pack(side="left", padx=10)
        self.set_part(None)

    # -- part ---------------------------------------------------------------------
    def set_part(self, part) -> None:
        self.part = part
        if not self.training:
            self.result = None
            set_text(self.log, "")
            self.progress.configure(value=0)
        self.refresh()

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        part = self.part
        target = self.cfg.detector.min_model_accuracy
        if part is None:
            self.readiness.configure(text="Choose a part number first.", foreground=style.MUTED)
            self.versions.set_rows([])
            self.result_label.configure(text="")
            for b in (self.train_btn, self.use_btn, self.switch_btn):
                b.configure(state="disabled")
            return
        from ...vision.training import training_problems

        problems = training_problems(self.cfg, part)
        counts = part.store.label_counts()
        if problems:
            self.readiness.configure(text="Not ready to train:\n" + "\n".join(f"  · {p}" for p in problems),
                                     foreground=style.NG)
        else:
            self.readiness.configure(
                text=f"Ready: {part.labelled_images()} labelled images; "
                     + ", ".join(f"{k} {v}" for k, v in counts.items()), foreground=style.OK)
        self.train_btn.configure(state="disabled" if problems or self.training else "normal")

        rows = []
        for m in reversed(part.models()):
            acc = f"{m.accuracy:.1%}" if m.accuracy is not None else "-"
            state = ("in use" if m.version == part.active_model else
                     "usable" if m.meets(target) else f"below {target:.0%}")
            rows.append((m.version, (m.version, m.trained.replace("T", " ")[:16], acc, m.images, state),
                         ("ok",) if m.version == part.active_model else () if m.meets(target) else ("muted",)))
        self.versions.set_rows(rows)
        self.switch_btn.configure(state="normal" if rows else "disabled")
        self._show_result()

    def _show_result(self) -> None:
        part, m = self.part, self.result
        target = self.cfg.detector.min_model_accuracy
        if m is None or part is None or part.model(m.version) is None:
            active = part.active() if part else None
            self.result_label.configure(text=f"In use: {active.summary()}" if active else "No model in use yet.",
                                        foreground=style.TEXT)
            self.use_btn.configure(state="disabled")
            return
        if m.version == part.active_model:
            self.result_label.configure(text=f"{m.summary()} is in use.", foreground=style.OK)
            self.use_btn.configure(state="disabled")
        elif m.meets(target):
            self.result_label.configure(text=f"New model {m.summary()} reached the {target:.0%} needed.",
                                        foreground=style.OK)
            self.use_btn.configure(state="normal")
        else:
            weak = m.weak_labels(target)
            lines = [f"New model {m.summary()} is below the {target:.0%} needed. "
                     + ("The previous model stays in use." if part.active() else "")]
            if weak:
                lines.append("It struggles with:")
                lines += [f"  · {lbl}: finds {r:.0%} of them, {p:.0%} of its finds are right" for lbl, p, r in weak]
            lines.append("Label more images of these, check the boxes, and train again.")
            self.result_label.configure(text="\n".join(lines), foreground=style.NG)
            self.use_btn.configure(state="disabled")

    # -- d. training --------------------------------------------------------------------
    def train(self) -> None:
        if self.training or self.part is None:
            return
        from ...vision import training

        problems = training.training_problems(self.cfg, self.part)
        if problems:
            show_error(self, "Not ready to train:\n" + "\n".join(problems))
            return
        try:
            epochs = int(self.epochs.get())
        except (tk.TclError, ValueError):
            show_error(self, "Epochs must be a number.")
            return
        part, user = self.part, self.page.user
        self.training = True
        self.result = None
        self._stop = threading.Event()
        self.train_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.configure(value=0, maximum=epochs)
        set_text(self.log, f"Training {part.code} on {part.labelled_images()} labelled images, up to {epochs} "
                           "epochs. This takes a while on a CPU.\n")

        def progress(msg: str) -> None:
            self.page.dispatcher.call(self._progress, msg)

        stop = self._stop
        self.page.dispatcher.run_task(
            lambda: training.train_detector(self.cfg, part, progress, stop, epochs=epochs),
            lambda res: self._trained(part, user, res), self._train_failed)

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
        self.cancel_btn.configure(state="disabled")

    def _trained(self, part, user: str, res) -> None:
        self._finish()
        self.station.log_model_change("model_trained", user, part.code,
                                      f"{res.version}: {res.evaluation.accuracy:.1%}")
        set_text(self.log, f"\nDone: {res.version}, {res.epochs_run} epochs in {res.seconds / 60:.0f} min, "
                           f"accuracy {res.evaluation.accuracy:.1%} on the held-back images.\n", append=True)
        if part is self.part:
            self.result = res.model
        self.refresh()
        self.page.refresh()

    def _train_failed(self, exc: BaseException) -> None:
        from ...vision.training import TrainingCancelled

        self._finish()
        self.refresh()
        if isinstance(exc, TrainingCancelled):
            set_text(self.log, "\nTraining cancelled - no new model was saved.\n", append=True)
            return
        set_text(self.log, f"\nTraining failed: {exc}\n", append=True)
        show_error(self, f"Training failed: {exc}")

    def cancel(self) -> None:
        if self._stop is not None:
            self._stop.set()
            set_text(self.log, "Cancelling after the current batch ...\n", append=True)

    def stop(self) -> None:
        self.cancel()

    # -- e. which model is in use ---------------------------------------------------------
    def use_new(self) -> bool:
        return self.result is not None and self.activate(self.result.version)

    def use_selected(self) -> bool:
        version = self.versions.selected()
        if not version:
            show_error(self, "Select a version in the list first.")
            return False
        return self.activate(version)

    def activate(self, version: str) -> bool:
        part = self.part
        info = part.model(version) if part else None
        if info is None:
            return False
        target = self.cfg.detector.min_model_accuracy
        if not info.meets(target):
            show_error(self, f"{info.summary()} is below the {target:.0%} needed and cannot be used.")
            return False
        previous = part.active_model
        if previous == version:
            return True
        part.set_active(version)
        kind = "model_rollback" if previous and version < previous else "model_activated"
        self.page.log(kind, f"{previous or '-'} -> {version}")
        self.page.changed()
        show_info(self, f"{part.code} now uses {info.summary()}.", "Model in use")
        return True
