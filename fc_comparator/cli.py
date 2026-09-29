"""Command line entry point.

    python -m fc_comparator                         # start the station UI
    python -m fc_comparator inspect --image board.jpg --part P001
    python -m fc_comparator report [--day 2026-09-29]
    python -m fc_comparator export --out history.xlsx [--from 2026-09-01 --to 2026-09-30]
    python -m fc_comparator harvest --image good_board.jpg --part P002 --forks right
    python -m fc_comparator set-secret --pin | --setup
    python -m fc_comparator check                   # validate config, camera, classifier
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


def cmd_gui(args) -> int:
    from .ui.app import run

    return run(args.config)


def cmd_inspect(args) -> int:
    from .alert import AlertController, NullAlert
    from .capture import FileSource
    from .capture.sources import write_image
    from .pipeline import Station

    cfg = _cfg(args)
    alert = None if args.alert else AlertController(NullAlert(), cfg.alert)
    store = None
    if args.no_store:
        from .storage import InspectionStore

        store = InspectionStore(":memory:")
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
    print(f"{rep.verdict.value}  part={rep.part_number}  {rep.duration_ms:.0f} ms  classifier={rep.classifier}")
    print(f"alignment: {rep.alignment.message} (shift {rep.alignment.shift_px:.1f}px, {rep.alignment.inliers} inliers)")
    if rep.error:
        print(f"error: {rep.error}")
    for p in rep.mismatches:
        print(f"  NG cable {p.cable} row {p.row}: expected {p.expected}, found {p.found} ({p.confidence:.0%}) - {p.reason}")
    if args.verbose:
        for p in rep.positions:
            print(f"  {'OK' if p.ok else 'NG'} C{p.cable}R{p.row} {p.found:<10} {p.confidence:.0%}")
    if args.out:
        print(f"annotated image: {write_image(args.out, res.annotated)}")
    return 0 if rep.ok else 1


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


def cmd_harvest(args) -> int:
    from .capture.sources import read_image
    from .classify.dataset import dataset_counts, harvest_board
    from .pipeline import Inspector

    cfg = _cfg(args)
    part = cfg.parts.get(args.part)
    if part is None:
        print(f"Unknown part number {args.part}", file=sys.stderr)
        return 2
    img = read_image(args.image)
    inspector = Inspector.from_config(cfg)
    if inspector.aligner is not None:
        img, info = inspector.aligner.align(img)
        print(f"alignment: {info.message}")
    ds = cfg.resolve(cfg.classifier.dataset_dir)
    paths = harvest_board(img, cfg.rois, part.pattern, ds, fork_orientation=args.forks, stem=args.part,
                          padding=cfg.classifier.roi_padding)
    print(f"Saved {len(paths)} crops to {ds}: {dataset_counts(ds)}")
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
    from .classify import create_classifier
    from .classify.dataset import dataset_counts, dataset_warnings

    cfg = _cfg(args)
    problems = cfg.validate()
    for p in problems:
        print(f"CONFIG  {p}")
    ref = cfg.resolve(cfg.reference_image)
    print(f"reference image: {ref} {'OK' if ref.is_file() else 'MISSING'}")
    ds = cfg.resolve(cfg.classifier.dataset_dir)
    print(f"dataset {ds}: {dataset_counts(ds)}")
    for w in dataset_warnings(ds):
        print(f"DATASET {w}")
    clf = create_classifier(cfg)
    print(f"classifier: {clf.name}, labels {clf.labels}")
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

    sub.add_parser("gui", help="start the touchscreen UI (default)")

    p = sub.add_parser("inspect", help="inspect one image file headlessly (exit code 0=OK, 1=NG)")
    p.add_argument("--image", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--operator", default="")
    p.add_argument("--out", help="write the annotated image here")
    p.add_argument("--alert", action="store_true", help="drive the configured alert backend")
    p.add_argument("--no-store", action="store_true", help="do not log to the database")
    p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("report", help="print the daily report")
    p.add_argument("--day", help="YYYY-MM-DD (default today)")

    p = sub.add_parser("export", help="export history to .csv or .xlsx")
    p.add_argument("--out", required=True)
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")

    p = sub.add_parser("harvest", help="add all crops of a known-good board to the dataset")
    p.add_argument("--image", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--forks", choices=["left", "right"], help="which way the forks face on this board")

    p = sub.add_parser("set-secret", help="change the supervisor PIN or setup password")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pin", action="store_true")
    g.add_argument("--setup", action="store_true")

    sub.add_parser("check", help="validate config, dataset, classifier and camera")

    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = {
        None: cmd_gui, "gui": cmd_gui, "inspect": cmd_inspect, "report": cmd_report, "export": cmd_export,
        "harvest": cmd_harvest, "set-secret": cmd_set_secret, "check": cmd_check,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
