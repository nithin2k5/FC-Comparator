"""Command line entry point (``python main.py ...`` or ``python -m fc_comparator ...``).

    main.py                                       start the station UI
    main.py inspect --image board.jpg --part P001 [--out annotated.jpg]
    main.py add-images --part P001 photos/        add images to a part (creates the part if new)
    main.py label --part P001 [--add N] [--rename OLD NEW] [--delete N] [--find]
    main.py train --part P001 [--epochs 80]       train the part's next model version
    main.py use-model --part P001 [--version v002]
    main.py set-master --part P001 --image good.jpg
    main.py report [--day 2026-09-29]
    main.py export --out history.xlsx [--from 2026-09-01 --to 2026-09-30]
    main.py set-secret --pin | --login [--user NAME]
    main.py check                                 validate config, parts, camera

The commands that change a part's model ask for the model login (or read it from the
FCC_USER / FCC_PASSWORD environment variables).
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


class LoginFailed(Exception):
    pass


def _cfg(args):
    from .config import load_config

    return load_config(args.config)


def _repo(cfg):
    from .vision.parts import PartRepository

    return PartRepository(cfg.resolve(cfg.storage.parts_dir))


def _db(cfg):
    from .station.storage import InspectionStore

    return InspectionStore(cfg.resolve(cfg.storage.database))


def _login(cfg) -> str:
    """Ask for the model login. Returns the user name; raises LoginFailed."""
    from .station.security import verify_secret

    user = os.environ.get("FCC_USER") or input("Model login - user: ").strip()
    password = os.environ.get("FCC_PASSWORD") or getpass.getpass("Password: ")
    if user != cfg.auth.user or not verify_secret(password, cfg.auth.password):
        raise LoginFailed("Wrong user name or password.")
    return user


def _part(repo, code: str):
    part = repo.get(code)
    if part is None:
        raise LookupError(f"No part number {code} (add images to it first: add-images --part {code} ...)")
    return part


def cmd_gui(args) -> int:
    from .ui.app import run

    return run(args.config)


def cmd_inspect(args) -> int:
    from .station import Station
    from .station.alerts import AlertController, NullAlert
    from .station.storage import InspectionStore
    from .vision.camera import FileSource, write_image

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
    from .vision.camera import IMAGE_EXTENSIONS
    from .vision.dataset import AnnotationStore

    cfg = _cfg(args)
    user = _login(cfg)
    repo = _repo(cfg)
    part = repo.get(args.part)
    if part is None:
        part = repo.create(args.part, args.description or "")
        _db(cfg).log_event("part_created", user, "", part_number=part.code)
        print(f"Created part {part.code}")
    added = dup = labelled = 0
    for p in map(Path, args.paths):
        if p.is_dir() and (p / "index.json").is_file():  # a labelled image folder: keep its boxes
            src = AnnotationStore(p)
            for rec in src.records():
                image_id, new = part.store.add_image(src.image_path(rec.id), name=rec.source_name)
                added += new
                dup += not new
                if new and rec.boxes:
                    part.store.set_boxes(image_id, rec.boxes)
                    labelled += 1
            continue
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS) if p.is_dir() else [p]
        for f in files:
            try:
                _, new = part.store.add_image(f)
            except Exception as exc:
                print(f"skipped {f}: {exc}")
                continue
            added += new
            dup += not new
    print(f"{part.code}: added {added} image(s) ({labelled} with their labels), {dup} duplicate(s) skipped.")
    print(f"Labels: {part.store.label_counts() or 'none yet'}")
    return 0


def cmd_label(args) -> int:
    from .core.models import Box

    cfg = _cfg(args)
    user = _login(cfg) if (args.add or args.rename or args.delete or args.find) else ""
    repo = _repo(cfg)
    part = _part(repo, args.part)
    changed = []
    for name in args.add or []:
        part.store.add_label(name)
        changed.append(f"added {name}")
    if args.rename:
        old, new = args.rename
        n = part.rename_label(old, new)
        changed.append(f"renamed {old} -> {new} ({n} boxes)")
    for name in args.delete or []:
        n = part.delete_label(name)
        changed.append(f"deleted {name} ({n} boxes)")
    if changed:
        _db(cfg).log_event("labels_changed", user, "; ".join(changed), part_number=part.code)
        print("\n".join(changed))
    if args.find:
        from .vision.detect import create_detector

        det = create_detector(cfg, part)
        thr = cfg.detector.confidence_threshold
        n = 0
        for rec in part.store.records():
            if rec.boxes:
                continue
            boxes = [Box(d.label, d.x, d.y, d.w, d.h) for d in det.detect(part.store.load_image(rec.id))
                     if d.confidence >= thr]
            part.store.set_boxes(rec.id, boxes)
            n += 1
        print(f"Pre-marked {n} unlabelled image(s) with {det.note}; check them in Model Setup.")
    stats = part.store.stats()
    print(f"{part.code}: {stats['images']} images, {stats['marked']} labelled")
    for label, k in part.store.label_counts().items():
        print(f"  {label:<20}{k:>5}")
    return 0


def cmd_train(args) -> int:
    from .vision.training import train_detector

    cfg = _cfg(args)
    user = _login(cfg)
    part = _part(_repo(cfg), args.part)
    if args.device:
        cfg.detector.device = args.device
    res = train_detector(cfg, part, print, epochs=args.epochs, imgsz=args.imgsz, base_model=args.base)
    _db(cfg).log_event("model_trained", user, f"{res.version}: {res.evaluation.accuracy:.1%}", part_number=part.code)
    target = cfg.detector.min_model_accuracy
    print(f"Done in {res.seconds:.0f} s ({res.epochs_run} epochs): {res.model.summary()}")
    if res.model.meets(target):
        print(f"Put it into use with: use-model --part {part.code} --version {res.version}")
    else:
        print(f"Below the {target:.0%} needed - the current model stays in use. Weak labels:")
        for label, p, r in res.model.weak_labels(target):
            print(f"  {label}: precision {p:.0%}, recall {r:.0%}")
    return 0


def cmd_use_model(args) -> int:
    cfg = _cfg(args)
    user = _login(cfg)
    part = _part(_repo(cfg), args.part)
    models = part.models()
    if not models:
        print(f"{part.code} has no trained model yet.", file=sys.stderr)
        return 2
    info = part.model(args.version) if args.version else models[-1]
    if info is None:
        print(f"{part.code} has no model {args.version}. Versions: {[m.version for m in models]}", file=sys.stderr)
        return 2
    target = cfg.detector.min_model_accuracy
    if not info.meets(target):
        print(f"{info.summary()} is below the {target:.0%} needed; not used.", file=sys.stderr)
        return 2
    previous = part.active_model
    part.set_active(info.version)
    kind = "model_rollback" if previous and info.version < previous else "model_activated"
    _db(cfg).log_event(kind, user, f"{previous or '-'} -> {info.version}", part_number=part.code)
    print(f"{part.code} now uses {info.summary()}")
    return 0


def cmd_set_master(args) -> int:
    from .core.layout import LayoutError
    from .core.models import Box
    from .core.parts import part_from_boxes
    from .vision.camera import read_image
    from .vision.detect import create_detector

    cfg = _cfg(args)
    user = _login(cfg)
    part = _part(_repo(cfg), args.part)
    if args.image_id:
        if args.image_id not in part.store:
            print(f"{part.code} has no image {args.image_id}", file=sys.stderr)
            return 2
        rec = part.store.get(args.image_id)
        boxes, size, image_id = rec.boxes, (rec.width, rec.height), rec.id
    else:
        img = read_image(args.image)
        thr = cfg.detector.confidence_threshold
        boxes = [Box(d.label, d.x, d.y, d.w, d.h) for d in create_detector(cfg, part).detect(img)
                 if d.confidence >= thr]
        size, image_id = (img.shape[1], img.shape[0]), ""
    try:
        master = part_from_boxes(part.code, boxes, size, part.taxonomy(), part.description)
    except (LayoutError, ValueError) as exc:
        print(f"Cannot set the master: {exc}", file=sys.stderr)
        return 2
    if not image_id:
        image_id, _ = part.store.add_image(args.image, good=True)
        part.store.set_boxes(image_id, boxes)
    master.master_image = image_id
    part.set_master(master)
    _db(cfg).log_event("master_saved", user, f"{master.cables} cables x {master.rows} rows", part_number=part.code)
    print(f"Master of {part.code}: {master.cables} cables x {master.rows} rows")
    for r, row in enumerate(master.pattern, 1):
        print(f"  row {r}: {', '.join(row)}")
    return 0


def cmd_report(args) -> int:
    cfg = _cfg(args)
    day = date.fromisoformat(args.day) if args.day else date.today()
    print(_db(cfg).daily_report(day).as_text())
    return 0


def cmd_export(args) -> int:
    from .station.storage import export_csv, export_excel

    cfg = _cfg(args)
    store = _db(cfg)
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
    """Run on the station PC itself; it is the way back in when the login is forgotten."""
    from .config import save_config
    from .station.security import hash_secret

    cfg = _cfg(args)
    what = "supervisor PIN" if args.pin else "model login password"
    first = getpass.getpass(f"New {what}: ")
    if len(first) < 4 or getpass.getpass("Repeat: ") != first:
        print("Too short or not matching; unchanged.", file=sys.stderr)
        return 2
    if args.pin:
        cfg.security.supervisor_pin = hash_secret(first)
    else:
        cfg.auth.password = hash_secret(first)
        if args.user:
            cfg.auth.user = args.user
    save_config(cfg)
    print(f"{what} updated in {cfg.path}")
    return 0


def cmd_check(args) -> int:
    from .vision.camera import create_source

    cfg = _cfg(args)
    problems = cfg.validate()
    for p in problems:
        print(f"CONFIG  {p}")
    repo = _repo(cfg)
    print(f"parts ({repo.root}): {len(repo.codes())}")
    for code in repo.codes():
        part = repo.get(code)
        missing = part.setup_problems(cfg.detector.backend)
        active = part.active()
        model = active.summary() if active else "no active model"
        master = f"{part.master.cables}x{part.master.rows}" if part.master else "no master"
        print(f"  {code:<14} {'READY' if not missing else 'NOT SET UP'}  {model}; master {master}; "
              f"{part.labelled_images()} labelled images")
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
    ap = argparse.ArgumentParser(prog="fc_comparator", description="Wire harness inspection station")
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

    p = sub.add_parser("add-images", help="add image files/folders to a part (creates the part if new); "
                                          "a folder with an index.json keeps its labels")
    p.add_argument("--part", required=True)
    p.add_argument("--description", help="description of a new part")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("label", help="list, add, rename/merge or delete a part's labels; pre-mark new images")
    p.add_argument("--part", required=True)
    p.add_argument("--add", action="append", metavar="NAME")
    p.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help="NEW may be an existing label (merge)")
    p.add_argument("--delete", action="append", metavar="NAME")
    p.add_argument("--find", action="store_true", help="pre-mark the unlabelled images with the active model")

    p = sub.add_parser("train", help="train the part's next model version")
    p.add_argument("--part", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--base", help="yolo11n.pt (default), yolov8n.pt, or yolo11n.yaml to train from scratch")
    p.add_argument("--device", help="cpu, 0 (first GPU), mps ...")

    p = sub.add_parser("use-model", help="make a trained version the part's active model (also to roll back)")
    p.add_argument("--part", required=True)
    p.add_argument("--version", help="default: the newest")

    p = sub.add_parser("set-master", help="set a part's master from a photo of a known-good board")
    p.add_argument("--part", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="photo of a good board; the objects are found by the part's model")
    g.add_argument("--image-id", help="one of the part's completely labelled images")

    p = sub.add_parser("report", help="print the daily report")
    p.add_argument("--day", help="YYYY-MM-DD (default today)")

    p = sub.add_parser("export", help="export history to .csv or .xlsx")
    p.add_argument("--out", required=True)
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")

    p = sub.add_parser("set-secret", help="change the supervisor PIN or the model login")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pin", action="store_true")
    g.add_argument("--login", action="store_true")
    p.add_argument("--user", help="with --login: also change the user name")

    sub.add_parser("check", help="validate config, parts and camera")

    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = {
        None: cmd_gui, "gui": cmd_gui, "inspect": cmd_inspect, "add-images": cmd_add_images, "label": cmd_label,
        "train": cmd_train, "use-model": cmd_use_model, "set-master": cmd_set_master, "report": cmd_report,
        "export": cmd_export, "set-secret": cmd_set_secret, "check": cmd_check,
    }
    try:
        return handlers[args.cmd](args)
    except LoginFailed as exc:
        print(exc, file=sys.stderr)
        return 3
    except (LookupError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
