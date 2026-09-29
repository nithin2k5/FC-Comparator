"""Train the YOLO clip detector from the images marked in the station.

    python scripts/train_detector.py                    # uses config/config.yaml
    python scripts/train_detector.py --epochs 80 --imgsz 1280
    python scripts/train_detector.py --evaluate-only    # score the current model / templates

The marked images come from the annotation store (``annotations.dir`` in the
config). The trained weights are written to ``detector.model_path`` together
with ``*_report.json`` (accuracy, per-class precision/recall, confusion matrix)
and ``*_confusion.png``. The Train screen in the app does the same.

Offline: pretrained weights (yolo11n.pt) are downloaded on first use. On an
offline station copy the .pt file into the working directory beforehand or pass
``--base yolo11n.yaml`` to train from scratch (needs more images/epochs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fc_comparator.annotations import AnnotationStore  # noqa: E402
from fc_comparator.config import load_config  # noqa: E402
from fc_comparator.detect import create_detector  # noqa: E402
from fc_comparator.training import evaluate, samples_from_store, train_detector  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--base", help="base weights, e.g. yolo11n.pt, yolov8n.pt or yolo11n.yaml (from scratch)")
    ap.add_argument("--device", help="cpu, 0 (first GPU), mps ...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluate-only", action="store_true", help="evaluate the current detector on all marked images")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.device:
        cfg.detector.device = args.device
    store = AnnotationStore(cfg.resolve(cfg.annotations.dir))
    print(f"Annotation store {store.root}: {store.stats()}")

    if args.evaluate_only:
        det = create_detector(cfg, store)
        result = evaluate(det, samples_from_store(store), cfg.taxonomy.classes,
                          min_confidence=cfg.detector.confidence_threshold)
        print(f"Detector: {det.name}\n{result.as_text()}")
        return 0

    res = train_detector(cfg, store, print, epochs=args.epochs, imgsz=args.imgsz, base_model=args.base, seed=args.seed)
    print(f"\nDone in {res.seconds:.0f} s ({res.epochs_run} epochs). Model: {res.model_path}")
    print(f"Report: {res.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
