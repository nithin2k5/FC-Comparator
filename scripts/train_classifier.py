"""Train the YOLO clip classifier from a folder of labeled crops.

    python scripts/train_classifier.py data/dataset --epochs 60
    python scripts/train_classifier.py samples/dataset --epochs 30 --out models/clip_classifier.pt

Input: one sub-folder per clip class (round, fork_left, fork_right, small, missing).

Steps
  1. stratified train/val split (per class, reproducible with --seed)
  2. optional fork mirroring: every fork_left crop also becomes a fork_right crop
     (flipped) and vice versa, inside the same split so validation stays honest
  3. train an Ultralytics YOLO classification model (YOLO11n-cls by default).
     Horizontal-flip augmentation is DISABLED: it would turn fork_left into
     fork_right while keeping the label.
  4. evaluate the best weights on the validation split: accuracy + confusion
     matrix (printed, and saved as CSV and PNG next to the model)
  5. copy the best weights to --out (the path the station config points at)

Offline use: pretrained weights (yolo11n-cls.pt) are downloaded on first use.
On an offline station copy the .pt file next to this script beforehand, or use
--from-scratch (random init; needs more data/epochs).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fc_comparator.capture.sources import read_image, write_image  # noqa: E402
from fc_comparator.classify.dataset import dataset_counts, dataset_warnings, list_images  # noqa: E402
from fc_comparator.models import CLASSIFIER_LABELS, mirror_label  # noqa: E402


def split_dataset(src: Path, dst: Path, val_split: float, seed: int, mirror_forks: bool) -> dict[str, dict[str, int]]:
    shutil.rmtree(dst, ignore_errors=True)
    rng = random.Random(seed)
    classes = [d.name for d in sorted(src.iterdir()) if d.is_dir() and list_images(d)]
    if mirror_forks:
        for c in list(classes):
            m = mirror_label(c)
            if m != c and m not in classes:
                classes.append(m)
    counts: dict[str, dict[str, int]] = {c: {"train": 0, "val": 0} for c in classes}

    def put(split: str, label: str, img: np.ndarray, name: str) -> None:
        write_image(dst / split / label / name, img)
        counts[label][split] += 1

    for c in classes:
        files = list_images(src / c)
        if not files:
            continue
        if len(files) < 2:
            raise SystemExit(f"Class {c!r} has only {len(files)} image(s); need at least 2 (ideally 30+)")
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * val_split)))
        for split, subset in (("val", files[:n_val]), ("train", files[n_val:])):
            for f in subset:
                img = read_image(f)
                put(split, c, img, f"{f.stem}.png")
                if mirror_forks and mirror_label(c) != c:
                    put(split, mirror_label(c), cv2.flip(img, 1), f"{f.stem}_mirror.png")
    empty = [c for c, v in counts.items() if v["train"] == 0 or v["val"] == 0]
    if empty:
        raise SystemExit(f"Classes without train or val images: {empty}")
    return counts


def evaluate(model_path: Path, val_dir: Path, imgsz: int, device: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="classify")
    names = [model.names[i] for i in sorted(model.names)]
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names)), int)
    for cls_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        files = list_images(cls_dir)
        for i in range(0, len(files), 64):
            batch = [read_image(f) for f in files[i:i + 64]]
            for r in model.predict(batch, imgsz=imgsz, device=device, verbose=False):
                cm[idx[cls_dir.name], int(r.probs.top1)] += 1
    acc = float(np.trace(cm) / max(1, cm.sum()))
    return acc, names, cm


def format_confusion(names: list[str], cm: np.ndarray) -> str:
    w = max(10, max(len(n) for n in names) + 1)
    lines = ["true \\ pred".ljust(w) + "".join(n[:w - 1].rjust(w) for n in names) + "   recall"]
    for i, n in enumerate(names):
        total = cm[i].sum()
        recall = cm[i, i] / total if total else 0.0
        lines.append(n.ljust(w) + "".join(str(v).rjust(w) for v in cm[i]) + f"   {recall:6.1%}")
    return "\n".join(lines)


def confusion_png(names: list[str], cm: np.ndarray, path: Path) -> None:
    cell, margin = 90, 150
    n = len(names)
    img = np.full((margin + n * cell + 20, margin + n * cell + 20, 3), 255, np.uint8)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            color = (int(255 - 200 * v), int(255 - 120 * v), 255) if i != j else (int(255 - 200 * v), 255, int(255 - 200 * v))
            x, y = margin + j * cell, margin + i * cell
            cv2.rectangle(img, (x, y), (x + cell, y + cell), color, -1)
            cv2.rectangle(img, (x, y), (x + cell, y + cell), (180, 180, 180), 1)
            cv2.putText(img, str(cm[i, j]), (x + 25, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, names[i][:12], (5, margin + i * cell + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, names[i][:10], (margin + i * cell + 5, margin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "true (rows) vs predicted (cols)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    write_image(path, img)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="folder with one sub-folder of crops per class")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "clip_classifier.pt")
    ap.add_argument("--model", default="yolo11n-cls.pt", help="base weights (yolo11n-cls.pt, yolov8n-cls.pt, ...)")
    ap.add_argument("--from-scratch", action="store_true", help="random init (no pretrained download)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, 0 (first GPU), mps ...")
    ap.add_argument("--no-mirror-forks", dest="mirror_forks", action="store_false")
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs" / "clip_classifier")
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    if not args.data.is_dir():
        raise SystemExit(f"Dataset folder not found: {args.data}")
    counts = dataset_counts(args.data)
    print(f"Dataset {args.data}: {counts}")
    unknown = [c for c in counts if c not in CLASSIFIER_LABELS]
    if unknown:
        raise SystemExit(f"Unknown class folders {unknown}; valid names: {list(CLASSIFIER_LABELS)}")
    for w in dataset_warnings(args.data):
        print(f"WARNING: {w}")

    split_dir = args.workdir / "data"
    split = split_dataset(args.data, split_dir, args.val_split, args.seed, args.mirror_forks)
    print("Split (train/val):", {c: f"{v['train']}/{v['val']}" for c, v in split.items()})

    base = args.model
    if args.from_scratch:
        base = base.replace(".pt", ".yaml")
    elif not Path(base).is_file():
        print(f"Base weights {base} not found locally - Ultralytics will try to download them.")
    try:
        model = YOLO(base, task="classify")
    except Exception as exc:
        if args.from_scratch:
            raise
        fallback = base.replace(".pt", ".yaml")
        print(f"Could not load {base} ({exc}); training from scratch with {fallback}")
        model = YOLO(fallback, task="classify")

    t = time.time()
    model.train(
        data=str(split_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(args.workdir),
        name="train",
        exist_ok=True,
        fliplr=0.0,  # orientation matters: never mirror during training
        flipud=0.0,
        erasing=0.1,  # large random erasing would make clips look "missing"
        plots=False,
        verbose=False,
        workers=0,
    )
    print(f"Training took {time.time() - t:.0f} s")

    best = args.workdir / "train" / "weights" / "best.pt"
    if not best.is_file():
        best = args.workdir / "train" / "weights" / "last.pt"
    acc, names, cm = evaluate(best, split_dir / "val", args.imgsz, args.device)
    print(f"\nValidation accuracy: {acc:.2%}  ({int(np.trace(cm))}/{int(cm.sum())})\n")
    print(format_confusion(names, cm))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    stem = args.out.with_suffix("")
    np.savetxt(f"{stem}_confusion.csv", cm, fmt="%d", delimiter=",", header=",".join(names), comments="")
    confusion_png(names, cm, Path(f"{stem}_confusion.png"))
    Path(f"{stem}_metrics.json").write_text(
        json.dumps({"accuracy": acc, "classes": names, "confusion": cm.tolist(), "split": split,
                    "epochs": args.epochs, "imgsz": args.imgsz, "base": base}, indent=2),
        encoding="utf-8",
    )
    print(f"\nModel saved to {args.out}")
    print(f"Set classifier.model_path to it (backend auto/yolo) and restart the station.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
