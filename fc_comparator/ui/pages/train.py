"""Train page: data readiness, YOLO training with live progress, results and detector status."""

from __future__ import annotations

import threading
import time

import customtkinter as ctk

from ...capture.sources import read_image
from ...detect import model_report
from ...pipeline import Station
from ...training import evaluate, samples_from_store, train_detector
from .. import theme
from ..widgets import Card, Dispatcher, ImageView, button, muted, show_error, show_info

TARGET_PER_CLASS = 50  # marked clips per type for a solid model; fewer works, confidence grows with more
BASE_MODELS = {"YOLO11 nano (fast)": "yolo11n.pt", "YOLO11 small (more accurate)": "yolo11s.pt",
               "YOLOv8 nano": "yolov8n.pt", "YOLO11 nano from scratch (offline)": "yolo11n.yaml"}


class TrainPage(ctk.CTkFrame):
    title = "Train model"

    def __init__(self, master, app, station: Station, dispatcher: Dispatcher):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.station = station
        self.cfg = station.cfg
        self.dispatcher = dispatcher
        self._stop = threading.Event()
        self.running = False
        self._t0 = 0.0

        self.grid_columnconfigure((0, 1), weight=1, uniform="cols")
        self.grid_rowconfigure(1, weight=1)

        # data readiness
        self.data_card = Card(self, title="Training data", subtitle="Marked clips per type. More, and more varied, "
                                                                     "images give a more confident model.")
        self.data_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.data_rows = ctk.CTkFrame(self.data_card.body, fg_color="transparent")
        self.data_rows.pack(fill="x")
        self.data_note = muted(self.data_card.body, "", 12, wraplength=460)
        self.data_note.pack(anchor="w", pady=(8, 0))

        # training controls
        ctl = Card(self, title="Train the detector", subtitle="Runs in the background; the station keeps working. "
                                                              "CPU training takes minutes to hours.")
        ctl.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        cb = ctl.body
        cb.grid_columnconfigure(1, weight=1)
        muted(cb, "Base model", 13).grid(row=0, column=0, sticky="w", pady=4)
        self.base = ctk.CTkOptionMenu(cb, values=list(BASE_MODELS), height=36)
        self.base.set(next((k for k, v in BASE_MODELS.items() if v == self.cfg.training.base_model), list(BASE_MODELS)[0]))
        self.base.grid(row=0, column=1, sticky="ew", pady=4)
        muted(cb, "Epochs", 13).grid(row=1, column=0, sticky="w", pady=4)
        self.epochs = ctk.CTkEntry(cb, height=36)
        self.epochs.insert(0, str(self.cfg.training.epochs))
        self.epochs.grid(row=1, column=1, sticky="ew", pady=4)
        muted(cb, "Image size", 13).grid(row=2, column=0, sticky="w", pady=4)
        self.imgsz = ctk.CTkOptionMenu(cb, values=["640", "960", "1280"], height=36)
        self.imgsz.set(str(self.cfg.training.imgsz))
        self.imgsz.grid(row=2, column=1, sticky="ew", pady=4)
        row = ctk.CTkFrame(cb, fg_color="transparent")
        row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.start_btn = button(row, "Start training", self.start, "primary", width=170)
        self.start_btn.pack(side="left")
        self.stop_btn = button(row, "Stop", self.stop_training, "danger", width=90)  # shown only while training
        self.progress = ctk.CTkProgressBar(cb, height=12)
        self.progress.set(0)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        self.status = muted(cb, "Idle", 13)
        self.status.grid(row=5, column=0, columnspan=2, sticky="w")

        # log + results
        log_card = Card(self, title="Training log")
        log_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(16, 0))
        self.log = ctk.CTkTextbox(log_card.body, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.log.pack(fill="both", expand=True)

        res = Card(self, title="Current detector")
        res.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(16, 0))
        rb = res.body
        rb.grid_columnconfigure(0, weight=1)
        rb.grid_rowconfigure(2, weight=1)
        self.det_label = ctk.CTkLabel(rb, text="", font=theme.font(16, "bold"), text_color=theme.TEXT, anchor="w")
        self.det_label.grid(row=0, column=0, sticky="w")
        self.det_note = muted(rb, "", 12, wraplength=460)
        self.det_note.grid(row=1, column=0, sticky="w")
        self.matrix = ImageView(rb, placeholder="The confusion matrix appears here after training")
        self.matrix.grid(row=2, column=0, sticky="nsew", pady=10)
        button(rb, "Evaluate on marked images", self.evaluate_now, "secondary").grid(row=3, column=0, sticky="w")

    # -- refresh ---------------------------------------------------------------
    def refresh(self) -> None:
        for w in self.data_rows.winfo_children():
            w.destroy()
        st = self.station.annotations.stats()
        classes = self.cfg.taxonomy.classes
        weak = []
        for label in classes:
            n = st["per_class"].get(label, 0)
            twin = self.cfg.taxonomy.mirror_of(label)
            effective = n + (st["per_class"].get(twin, 0) if twin != label and self.cfg.training.mirror else 0)
            r = ctk.CTkFrame(self.data_rows, fg_color="transparent")
            r.pack(fill="x", pady=3)
            ctk.CTkLabel(r, text=label, width=110, anchor="w", font=theme.font(13, "bold"), text_color=theme.TEXT).pack(side="left")
            bar = ctk.CTkProgressBar(r, height=10, progress_color=theme.class_color(classes, label))
            bar.set(min(1.0, effective / TARGET_PER_CLASS))
            bar.pack(side="left", fill="x", expand=True, padx=8)
            muted(r, f"{n}" + (f" (+{effective - n} mirrored)" if effective > n else ""), 12).pack(side="left")
            if effective < 20:
                weak.append(label)
        unmarked = st["images"] - st["marked"]
        note = f"{st['marked']} marked image(s), {st['boxes']} clips."
        if unmarked:
            note += f" {unmarked} image(s) still to mark."
        if weak:
            note += f" Mark more of: {', '.join(weak)} (at least 20 each, {TARGET_PER_CLASS}+ recommended)."
        self.data_note.configure(text=note)
        self.start_btn.configure(state="normal" if st["marked"] and not self.running else "disabled")
        self._show_detector()

    def _show_detector(self) -> None:
        det = self.station.inspector.detector
        name = {"yolo": "Trained YOLO model", "template": "Template matching"}.get(det.name, det.name)
        self.det_label.configure(text=name)
        path = self.cfg.resolve(self.cfg.detector.model_path)
        rep = model_report(path)
        text = det.note
        if rep:
            cm = rep.get("confusion") or []
            missed = sum(r[-1] for r in cm[:-1]) if cm else 0
            false = sum(cm[-1][:-1]) if cm else 0
            text += (f"\nLast training: validation accuracy {rep['accuracy']:.1%} on {rep.get('images', '?')} image(s), "
                     f"{missed} missed, {false} false clip(s), {rep.get('epochs', '?')} epochs.")
            png = path.with_name(path.stem + "_confusion.png")
            if png.is_file():
                self.matrix.set_image(read_image(png))
        self.det_note.configure(text=text)

    # -- training ------------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.dispatcher.call(self._append, msg)

    def _append(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        if msg.startswith("epoch "):
            try:
                done, total = msg.split()[1].split("/")
                frac = int(done) / int(total)
                self.progress.set(frac)
                elapsed = time.time() - self._t0
                eta = elapsed / max(frac, 1e-6) - elapsed
                self.status.configure(text=f"Epoch {done}/{total}  ·  about {eta / 60:.0f} min left")
            except (ValueError, IndexError):
                pass

    def start(self) -> None:
        try:
            epochs = int(self.epochs.get())
            if not 1 <= epochs <= 1000:
                raise ValueError
        except ValueError:
            show_error(self, "Epochs must be a number between 1 and 1000.")
            return
        imgsz, base = int(self.imgsz.get()), BASE_MODELS[self.base.get()]
        self.running = True
        self._stop.clear()
        self._t0 = time.time()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.stop_btn.pack(side="left", padx=8)
        self.progress.set(0)
        self.status.configure(text="Preparing data set ...")
        self.log.delete("1.0", "end")
        store = self.station.annotations
        self.dispatcher.run_task(
            lambda: train_detector(self.cfg, store, self._say, self._stop, epochs=epochs, imgsz=imgsz, base_model=base),
            self._done, self._failed)

    def stop_training(self) -> None:
        self._stop.set()
        self.status.configure(text="Stopping after this step ...")
        self.stop_btn.configure(state="disabled")

    def _finish(self) -> None:
        self.running = False
        self.stop_btn.pack_forget()
        self.refresh()

    def _done(self, res) -> None:
        ev = res.evaluation
        self.progress.set(1.0)
        self.status.configure(text=f"Finished in {res.seconds / 60:.1f} min ({res.epochs_run} epochs)"
                                   + (" - stopped early" if res.stopped else ""))
        self._finish()
        ok = ev.accuracy >= self.cfg.detector.min_model_accuracy
        use = ("The station switches to the new model now." if ok else
               f"Below the {self.cfg.detector.min_model_accuracy:.0%} needed, so the station keeps template matching. "
               "Mark more images or train longer.")
        self.app.rebuild_detector(then=self._show_detector)
        show_info(self, f"Validation accuracy {ev.accuracy:.1%}  ·  {ev.missed_clips} missed  ·  "
                        f"{ev.false_clips} false clip(s) on {ev.images} image(s).\n\n{use}", "Training finished")

    def _failed(self, exc: BaseException) -> None:
        self.status.configure(text="Training failed")
        self._finish()
        self._append(f"ERROR: {exc}")
        show_error(self, f"Training failed: {exc}\n\nOffline? Choose 'from scratch' as base model, or copy the "
                         "pretrained .pt file next to the program.")

    def evaluate_now(self) -> None:
        det = self.station.inspector.detector
        store = self.station.annotations
        self.status.configure(text="Evaluating ...")

        def done(res):
            self.status.configure(text="Idle")
            self._append(f"Evaluation of {det.name} on all marked images:\n{res.as_text()}")
            extra = " (template matching is scored on its own examples - train a model for an independent check)" \
                if det.name == "template" else ""
            show_info(self, f"Accuracy {res.accuracy:.1%}, {res.missed_clips} missed, {res.false_clips} false clip(s) "
                            f"on {res.images} marked image(s){extra}.", "Evaluation")

        self.dispatcher.run_task(
            lambda: evaluate(det, samples_from_store(store), self.cfg.taxonomy.classes,
                             min_confidence=self.cfg.detector.confidence_threshold),
            done, lambda e: show_error(self, str(e)))
