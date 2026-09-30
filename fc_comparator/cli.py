"""Command line entry point (``python main.py ...`` or ``python -m fc_comparator ...``).

    main.py                                  start the station UI
    main.py inspect --image board.jpg --part P001 [--out annotated.jpg]
    main.py add-images photos/ --part P004 [--good]     upload images to mark in the UI
    main.py make-master --image-id ID --part P004       part pattern + layout from a marked good board
    main.py train [--epochs 80]              train the YOLO detector on the marked images
    main.py evaluate                         score the current detector on the marked images
    main.py report [--day 2026-09-29]
    main.py export --out history.xlsx [--from 2026-09-01 --to 2026-09-30]
    main.py set-secret --pin | --setup
    main.py check                            validate config, marked data, detector, camera
"""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _cfg(args):
    from .config import load_config

    return load_config(args.config)


def _store(cfg):
    from .annotations import AnnotationStore

    return AnnotationStore(cfg.resolve(cfg.annotations.dir))


def cmd_gui(args) -> int:
    from .ui.app import run

    return run(args.config)


def cmd_inspect(args) -> int:
    from .alert import AlertController, NullAlert
    from .capture import FileSource
    from .capture.sources import write_image
    from .pipeline import Station
    from .storage import InspectionStore

    cfg = _cfg(args)
    alert = None if args.alert else AlertController(NullAlert(), cfg.alert)
    store = InspectionStore(":memory:") if args.no_store else None
    station = Station(cfg, source=FileSource(args.image), alert=alert, store=store)
    if args.no_store:
        station.images = None
    station.lock.enabled = False  # headless: nothing to lock
    try:
        station.open()
        res = station.inspect(args.part, args.operator)
    finally:
        station.close()
    rep = res.report
    print(f"{rep.verdict.value}  part={rep.part_number}  {rep.duration_ms:.0f} ms  detector={rep.detector}")
    print(f"placement: {rep.placement.message}")
    if rep.error:
        print(f"error: {rep.error}")
    for p in rep.mismatches:
        conf = "" if p.found == "missing" else f" ({p.confidence:.0%})"
        print(f"  NG cable {p.cable} row {p.row}: expected {p.expected}, found {p.found}{conf} - {p.reason}")
    for b in rep.extras:
        print(f"  NG unexpected {b.label} ({b.confidence:.0%}) at x={b.center[0]:.0f} y={b.center[1]:.0f}")
    if args.verbose:
        for p in rep.positions:
            print(f"  {'OK' if p.ok else 'NG'} C{p.cable}R{p.row} {p.found:<10} {p.confidence:.0%}")
    if args.out:
        print(f"annotated image: {write_image(args.out, res.annotated)}")
    return 0 if rep.ok else 1


def cmd_add_images(args) -> int:
    from .capture.sources import IMAGE_EXTENSIONS

    cfg = _cfg(args)
    store = _store(cfg)
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS) if p.is_dir() else [p]
    added = dup = 0
    for f in files:
        try:
            _, new = store.add_image(f, part=args.part or "", good=args.good)
        except Exception as exc:
            print(f"skipped {f}: {exc}")
            continue
        added += new
        dup += not new
    print(f"Added {added} image(s), {dup} duplicate(s) skipped. Store: {store.stats()}")
    print("Annotate the clips on them in the app: Setup.")
    return 0


def cmd_make_master(args) -> int:
    from .config import save_config
    from .layout import LayoutError
    from .parts import part_from_marked_image

    cfg = _cfg(args)
    store = _store(cfg)
    if args.image_id not in store:
        print(f"No image {args.image_id} in {store.root}", file=sys.stderr)
        return 2
    try:
        part, notes = part_from_marked_image(store, args.image_id, args.part, cfg.taxonomy, args.description or "")
    except (LayoutError, ValueError) as exc:
        print(f"Cannot create master: {exc}", file=sys.stderr)
        return 2
    existed = args.part in cfg.parts
    cfg.parts[part.code] = part
    save_config(cfg)
    store.update(args.image_id, part=part.code, good=True)
    for n in notes:
        print(f"note: {n}")
    print(f"{'Updated' if existed else 'Created'} {part.code}: {part.cables} cables x {part.rows} rows, pattern {part.pattern}")
    return 0


def cmd_train(args) -> int:
    from .training import train_detector

    cfg = _cfg(args)
    if args.device:
        cfg.detector.device = args.device
    res = train_detector(cfg, _store(cfg), print, epochs=args.epochs, imgsz=args.imgsz, base_model=args.base)
    print(f"Done in {res.seconds:.0f} s ({res.epochs_run} epochs). Model: {res.model_path}")
    return 0


def cmd_evaluate(args) -> int:
    from .detect import create_detector
    from .training import evaluate, samples_from_store

    cfg = _cfg(args)
    store = _store(cfg)
    det = create_detector(cfg, store)
    res = evaluate(det, samples_from_store(store), cfg.taxonomy.classes, min_confidence=cfg.detector.confidence_threshold)
    print(f"Detector: {det.name} on {res.images} marked image(s) (template detector: these are its own examples)")
    print(res.as_text())
    return 0


def cmd_report(args) -> int:
    from .storage import InspectionStore

    cfg = _cfg(args)
    store = InspectionStore(cfg.resolve(cfg.storage.database))
    day = date.fromisoformat(args.day) if args.day else date.today()
    print(store.daily_report(day).as_text())
    return 0


def cmd_export(args) -> int:
    from .storage import InspectionStore, export_csv, export_excel

    cfg = _cfg(args)
    store = InspectionStore(cfg.resolve(cfg.storage.database))
    start = datetime.fromisoformat(args.date_from) if args.date_from else None
    end = datetime.fromisoformat(args.date_to) + timedelta(days=1) if args.date_to else None
    out = Path(args.out)
    if out.suffix.lower() == ".xlsx":
        print(f"Wrote {export_excel(store, out, start, end)}")
    else:
        a, b = export_csv(store, out, start, end)
        print(f"Wrote {a} and {b}")
    return 0


def cmd_set_secret(args) -> int:
    from .config import save_config
    from .security import hash_secret

    cfg = _cfg(args)
    what = "supervisor PIN" if args.pin else "setup password"
    first = getpass.getpass(f"New {what}: ")
    if len(first) < 4 or getpass.getpass("Repeat: ") != first:
        print("Too short or not matching; unchanged.", file=sys.stderr)
        return 2
    if args.pin:
        cfg.security.supervisor_pin = hash_secret(first)
    else:
        cfg.security.setup_password = hash_secret(first)
    save_config(cfg)
    print(f"{what} updated in {cfg.path}")
    return 0


def cmd_check(args) -> int:
    from .capture import create_source
    from .detect import create_detector

    cfg = _cfg(args)
    problems = cfg.validate()
    for p in problems:
        print(f"CONFIG  {p}")
    store = _store(cfg)
    print(f"marked images ({store.root}): {store.stats()}")
    for code, part in sorted(cfg.parts.items()):
        print(f"part {code}: {part.cables}x{part.rows} {'layout from master' if part.layout else 'NO master layout'}")
    det = create_detector(cfg, store)
    print(f"detector: {det.name}, classes {det.labels}")
    if not det.ready:
        problems.append("detector not ready")
    src = create_source(cfg)
    try:
        src.open()
        frame = src.capture(timeout=5)
        print(f"camera/source: OK, frame {frame.shape[1]}x{frame.shape[0]}")
    except Exception as exc:
        print(f"camera/source: FAILED - {exc}")
        problems.append(str(exc))
    finally:
        src.close()
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fc_comparator", description="Wire harness clip inspection station")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="config YAML (default: %(default)s)")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("gui", help="start the station UI (default)")

    p = sub.add_parser("inspect", help="inspect one image file (exit code 0=OK, 1=NG)")
    p.add_argument("--image", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--operator", default="")
    p.add_argument("--out", help="write the annotated image here")
    p.add_argument("--alert", action="store_true", help="drive the configured alert backend")
    p.add_argument("--no-store", action="store_true", help="do not log to the database")
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("add-images", help="upload image files/folders for marking")
    p.add_argument("paths", nargs="+")
    p.add_argument("--part", help="part number shown on these boards")
    p.add_argument("--good", action="store_true", help="the boards are known good (candidates for masters)")

    p = sub.add_parser("make-master", help="create/update a part from a completely marked good board")
    p.add_argument("--image-id", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--description")

    p = sub.add_parser("train", help="train the YOLO detector on the marked images")
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--base", help="yolo11n.pt (default), yolov8n.pt, or yolo11n.yaml to train from scratch")
    p.add_argument("--device", help="cpu, 0 (first GPU), mps ...")

    sub.add_parser("evaluate", help="score the current detector on the marked images")

    p = sub.add_parser("report", help="print the daily report")
    p.add_argument("--day", help="YYYY-MM-DD (default today)")

    p = sub.add_parser("export", help="export history to .csv or .xlsx")
    p.add_argument("--out", required=True)
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")

    p = sub.add_parser("set-secret", help="change the supervisor PIN or setup password")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pin", action="store_true")
    g.add_argument("--setup", action="store_true")

    sub.add_parser("check", help="validate config, marked data, detector and camera")

    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = {
        None: cmd_gui, "gui": cmd_gui, "inspect": cmd_inspect, "add-images": cmd_add_images,
        "make-master": cmd_make_master, "train": cmd_train, "evaluate": cmd_evaluate, "report": cmd_report,
        "export": cmd_export, "set-secret": cmd_set_secret, "check": cmd_check,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
