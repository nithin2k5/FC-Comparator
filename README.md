# FC-Comparator

Vision inspection station for wire harness boards. An overhead camera photographs the harness cables laid out on the
board. The station finds every plastic clip with a trained YOLO model, works out which cable and row each clip belongs
to, and compares every position with the master of the scanned part number. When anything is wrong it shows **NG**,
switches the tower light to red, sounds the buzzer, locks the station and logs the result.

The UI is plain **tkinter** (part of Python). After logging in there are five tabs:

```
 Training ─ images of boards + the clips marked on them ─► Train ─► clip detector (YOLO)
 Parts    ─ photo of a good board ─► clips found ─► check ─► Save ─► master of part number
 Inspect  ─ scan part number ─► INSPECT / foot pedal ─► OK / NG ─► tower light, buzzer, lock, log
 History  ─ every inspection, its findings and image; daily report; CSV / Excel export
 Settings ─ camera, detection, alerts, login and supervisor PIN
```

## Contents

1. [Installation and first start](#1-installation-and-first-start)
2. [Project layout](#2-project-layout)
3. [Clip types and the training data](#3-clip-types-and-the-training-data)
4. [Training tab](#4-training-tab)
5. [Parts tab](#5-parts-tab)
6. [Inspect tab](#6-inspect-tab)
7. [History, reports and export](#7-history-reports-and-export)
8. [Settings and configuration](#8-settings-and-configuration)
9. [Camera setup](#9-camera-setup)
10. [Wiring the tower light, buzzer and foot pedal](#10-wiring-the-tower-light-buzzer-and-foot-pedal)
11. [Command line](#11-command-line)
12. [Tests](#12-tests)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Installation and first start

Requires Python 3.11 or newer. Everything runs offline once installed.

```bash
git clone https://github.com/nithin2k5/FC-Comparator.git
cd FC-Comparator
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

On Linux / Raspberry Pi OS also install Tk: `sudo apt install python3-tk`. On the Pi, add your user to the `gpio`,
`video` and (for USB relays) `dialout` groups.

**Login:** username **nice**, password **nice1234**. The supervisor PIN (to release an NG lock) is **1234**. Change both
under Settings → Security before production.

**Offline stations.** Put `yolo11n.pt` (the pretrained base for training) into `models/`; otherwise Ultralytics
downloads it the first time you train. Trained weights are `models/clip_detector.pt`.

---

## 2. Project layout

```
main.py                    entry point:  python main.py [command]
config/config.yaml         the station's settings, clip types and part numbers
dataset/                   training data: images/ + index.json (the clips marked on every image)
models/                    clip_detector.pt (+ _report.json, _confusion.png), yolo11n.pt   [not in git]
data/                      inspections.db and evidence images                               [not in git]
fc_comparator/
  config.py                typed YAML configuration
  cli.py                   command line
  core/                    pure logic, no I/O
    models.py              clip types (taxonomy), Box, Layout, PartNumber, inspection results
    layout.py              cables x rows grid, master from a good board, fitting a master onto a new photo
    compare.py             found vs expected at every position
    parts.py               part number from the clips of a good board
  vision/                  images and clip detection
    camera.py              OpenCV camera / image file sources
    dataset.py             training images + marked boxes (dataset/)
    detect/                YOLO detector, template-matching fallback, detector selection
    training.py            YOLO data-set export (with mirroring), training, evaluation
    drawing.py             OK/NG boxes drawn on the result image
    synthetic.py           synthetic boards for the tests
  station/                 the inspection station
    inspector.py           image -> detect -> place -> compare -> report
    station.py             capture + inspector + alert + NG lock + logging
    lock.py security.py barcode.py
    alerts/                tower light / buzzer: GPIO, USB relay, Modbus TCP, console
    storage/               SQLite history, evidence images, CSV / Excel export
  ui/                      tkinter UI
    app.py login.py main_window.py style.py widgets.py box_editor.py
    pages/                 inspect.py parts.py training.py history.py settings.py
scripts/make_test_fixtures.py   regenerates tests/fixtures (synthetic boards)
tests/                     unit, end-to-end and UI tests; tests/fixtures = synthetic boards + their config
```

---

## 3. Clip types and the training data

The FCC boards carry four cables with four clip rows each. Five kinds of clip occur, and the forks can face either way,
so the detector learns eight classes:

| class | clip |
|---|---|
| `round` | black round push clip (some with side ears) |
| `small` | small black arrow push clip |
| `fork_left` / `fork_right` | black U-bracket with push pin; the side its open end faces |
| `grey_fork_left` / `grey_fork_right` | the same bracket in grey plastic (black pin) |
| `metal_fork_left` / `metal_fork_right` | silver metal bracket with a square pin holder |

A part's master may also say just `fork`, `grey_fork` or `metal_fork`: either direction is then accepted.
Clip types are listed in `config.yaml` (`taxonomy`); a new type needs marked examples and a new training run.

**The data set** (`dataset/`) holds the 20 FCC board photos (`board_01` … `board_20`, 1204×1600). Every clip on every
photo is marked — 320 clips: round 147, small 44, black fork 90, metal fork 22, grey fork 17. In these photos every
black and metal fork faces right and every grey fork left; training adds a mirrored copy of each photo with left/right
swapped, so both directions are learned. YOLO's own random flip is switched off because it would keep the wrong
direction label.

The flat black strip below the connector of cable 2 on most photos belongs to the fixture and is intentionally not
marked.

---

## 4. Training tab

* **Add images…**, **Add folder…** or **Capture from camera** add photos to the data set (duplicates are skipped).
* Select an image and mark **every** clip: choose the clip type (or press 1–9), drag a box around the clip. Click a box
  to select it, drag to move, drag the handles to resize, Delete removes it, Ctrl+Z undoes. Mouse wheel zooms,
  right-drag pans, ← / → go to the previous / next image. Everything is saved automatically.
* **Find clips** pre-marks an image with the current model — only corrections are needed. This makes adding new
  photos fast once a model exists.
* **Train** trains in the background (progress and log on the right; **Stop** ends it early). 20 % of the images are
  held back for validation. The new model is used when its validation accuracy (clips found with the right type at the
  confidence threshold) reaches `detector.min_model_accuracy` (95 %); otherwise the station keeps the previous detector.
  The report is written next to the model (`clip_detector_report.json`, `clip_detector_confusion.png`).

Unmarked clips teach the model "this is not a clip", so mark images completely. More photos — shifted boards, other
lighting, both fork directions, boards with missing clips (mark only what is there) — make the model more robust.

**Current model:** MODEL_RESULTS

Command line: `python main.py train --epochs 120` and `python main.py evaluate`.

---

## 5. Parts tab

A part number's **master** says which clip belongs at every position (cable × row) and where the positions are.

1. **New from photo…** (or **New from camera**): take a photo of a known-good board of the part.
2. The model finds the clips and the editor shows them, with the resulting pattern on the right
   ("4 cables x 4 rows", row by row, cable by cable). Fix any box the same way as on the Training tab.
   **Save part** stays disabled until every position has exactly one clip.
3. Enter the part number (the text on its barcode) and a description, **Save part**.

The photo and its boxes are also added to the training data. **Edit** reopens a part's master photo; **Delete** removes
the part number.

Command line: `python main.py add-part --image good_board.jpg --part P123 --description "..."`.

---

## 6. Inspect tab

1. **Part number.** Scan the barcode into the part box (the scanner types the code + Enter), or pick from the list.
   Scanning an operator badge `OP:1234` fills the operator. The barcode format is `barcode.part_pattern`.
2. **INSPECT**, the trigger key (F9 — most USB foot pedals can send a key) or the GPIO foot pedal. **Inspect an image
   file…** runs the same inspection on a saved photo.
3. **Result.** The big banner shows **OK** or **NG**; the photo shows green boxes for correct clips, red boxes with
   `found ≠ expected` for wrong clips, dashed red for missing clips and orange for unexpected clips. The table lists
   every position, problems first.

**Never OK:** a missing clip, a clip found with less than `detector.confidence_threshold` confidence (*unsure*), a clip
where the master has none, a board that does not fit the part's layout (wrong part / board out of view), an unknown
part number, or a detector error.

**NG lock.** After an NG the station locks: the part can't be changed and the Parts, Training and Settings tabs are
disabled until a re-inspection of the same part passes or a supervisor presses **Supervisor unlock (PIN)**. Both are
logged, wrong PINs too.

---

## 7. History, reports and export

* Every inspection is stored in SQLite (`storage.database`): time, station, operator, part, result, duration, detector,
  placement, and every position with expected / found clip and confidence.
* The annotated image of every NG is saved (`storage.image_dir/YYYY-MM-DD/…`), plus every Nth OK image
  (`storage.save_ok_every_n`).
* **History tab:** filter by date, part and result; view the image and findings; export CSV or Excel; daily report
  (totals, NG rate, per part, most frequent failing positions, supervisor unlocks).
* Command line: `python main.py report --day 2026-09-30`, `python main.py export --out history.xlsx --from … --to …`.

---

## 8. Settings and configuration

The Settings tab edits the everyday settings; everything lives in `config/config.yaml` (written atomically; relative
paths are resolved from the config file's folder).

| key | meaning |
|---|---|
| `camera.*` | `source` (`camera` or `file`), camera number, resolution, `file_path` for the file source |
| `detector.backend` | `auto` (trained model if accurate enough, else template matching), `yolo`, `template` |
| `detector.confidence_threshold` | below this a clip is *unsure* (NG); default 0.6 |
| `detector.min_model_accuracy` | validation accuracy a trained model needs before `auto` uses it; 0.95 |
| `layout.*` | match tolerance (clip sizes), max board shift (px) and rotation (deg) |
| `dataset.dir` | training images and marks |
| `training.*` | base model, epochs, image size, batch, validation split, mirroring |
| `taxonomy.*` | clip types, groups (`fork`), mirror pairs |
| `alert.*` | tower light backend and its hardware settings (§10) |
| `security.*` | login user + password hash, supervisor PIN hash, lock after NG |
| `storage.*`, `barcode.*`, `trigger.*`, `ui.*` | as named |
| `parts.<code>` | `description`, `pattern` (rows top to bottom, each a list of clips per cable), `layout`, `master_image` |

---

## 9. Camera setup

1. Mount the camera rigidly overhead, lens parallel to the board. A fixture with end stops keeps shifts small; the
   layout fit accepts up to `layout.max_shift_px` (200 px) and `layout.max_rotation_deg` (8°).
2. Diffuse, even light (LED panels at 45° or a ring light); avoid glare on the metal forks and changing daylight.
3. Clips should be at least ~30 px across in the image.
4. Turn autofocus off and keep focus / exposure fixed; use the same camera settings for training photos and inspection.
5. Run `python main.py check`, then watch the live image on the Inspect tab.

---

## 10. Wiring the tower light, buzzer and foot pedal

Select the backend in Settings → Alerts (`alert.backend`). The station drives three outputs: **green lamp**, **red
lamp** and **buzzer**.

| Event | green | red | buzzer |
|---|---|---|---|
| OK | on | off | off |
| NG | off | on | on for `buzzer_seconds` (`pulse`), or until unlocked (`until_ack`) |
| unlocked / idle | off | off | off |

**Settings → Alerts → Test tower light** cycles green, then red with buzzer, then off.

### a) Raspberry Pi GPIO (`gpio`)

Pi GPIO pins are 3.3 V and about 16 mA. **Never drive a lamp or buzzer directly.** Use an opto-isolated relay module or a
transistor/MOSFET driver board. A typical 24 V DC tower light with a common wire:

```
 Raspberry Pi (BCM)            3-ch relay module (opto-isolated)          24 V tower light
 ------------------            ------------------------------------       ----------------
 5V   (pin 2)  ─────────────── VCC / JD-VCC
 GND  (pin 6)  ─────────────── GND
 GPIO17 (pin 11) ───────────── IN1  ── relay 1: COM ── +24V, NO ──────────── green
 GPIO27 (pin 13) ───────────── IN2  ── relay 2: COM ── +24V, NO ──────────── red
 GPIO22 (pin 15) ───────────── IN3  ── relay 3: COM ── +24V, NO ──────────── buzzer
                                                          24 V PSU  0V ───── common (black)
 GPIO26 (pin 37) ─── foot pedal (NO contact) ─── GND (pin 39)     # trigger.gpio_pin: 26
```

* Most relay modules are **active-low** (the relay clicks when the input goes to 0 V): set `alert.gpio.active_high:
  false` and check with **Test tower light**.
* The pedal input uses the Pi's internal pull-up with 50 ms debounce.
* Keep the 24 V wiring physically separate from the Pi, and fuse the 24 V supply.

### b) USB relay board (`usb_relay`)

CH340-based "LCUS" serial relay boards with 1, 2, 4 or 8 channels (protocol `A0 <ch> <state> <sum>`). The board appears
as `COMx` on Windows or `/dev/ttyUSB0` on Linux. Set the port and channel numbers, and wire each relay's COM/NO contact
as above.

### c) Modbus TCP PLC (`modbus`)

The station writes three coils with function 05 (Write Single Coil) on `host:port`, unit `unit_id`: `green_coil`,
`red_coil` and `buzzer_coil` (0-based). Lamps are switched off before the new one is switched on, so green and red never
light together. The connection is kept open and reconnects automatically.

### d) Console / none

`console` logs every output change (development). `none` does nothing.

Hardware errors never crash an inspection or turn an NG into an OK; they appear as **ALERT FAULT** on the Inspect tab and
in the log.

---

## 11. Command line

```
python main.py [-c config.yaml] [gui]                  start the station UI (default)
python main.py inspect --image F --part P [--out A.jpg] [--no-store] [-v]   exit code 0 = OK, 1 = NG
python main.py add-images PATH... [--part P] [--good]  add photos to the training data set
python main.py add-part --image F --part P [--description D]      part master from a photo of a good board
python main.py add-part --image-id ID --part P         part master from a marked data-set image
python main.py train [--epochs N] [--imgsz N] [--base yolo11n.pt] [--device cpu|0]
python main.py evaluate                                score the current detector on the marked images
python main.py report [--day YYYY-MM-DD]
python main.py export --out F.csv|F.xlsx [--from D] [--to D]
python main.py set-secret --pin | --login [--user NAME]
python main.py check                                   validate config, data set, detector and camera
```

---

## 12. Tests

```bash
pytest                   # everything (about 3 minutes)
pytest -m "not slow"     # skip the short YOLO training test
pytest -m "not ui"       # skip the UI tests (headless machines)
```

The tests use synthetic boards (`tests/fixtures`, regenerate with `python scripts/make_test_fixtures.py`) so they run
without the trained model.

* `test_e2e.py` runs every fixture board through the full station and checks the exact findings; also shifted boards,
  missing / unsure clips, parts without a master photo, NG lock, logging, evidence images, CLI and a new part set up
  from a marked image.
* `test_ui.py` drives the real tkinter window: login / logout, scanning, inspection, NG lock and supervisor unlock,
  history, a new part from a good board (fixing a missed clip in the editor), marking boxes with the mouse, training,
  settings, login and PIN changes.
* `test_compare.py`, `test_layout.py`, `test_detect.py`, `test_training.py`, `test_annotations.py`, `test_alert.py`
  (fake GPIO, serial port and PLC), `test_storage.py`, `test_config.py`, `test_capture.py`, `test_barcode.py`.

---

## 13. Troubleshooting

| symptom | fix |
|---|---|
| "placement failed: only N/M clips fit" | wrong part on the board, board outside the view, or shifted beyond `layout.max_shift_px` |
| "board does not match … wrong part?" | the clips found belong to another part number |
| many *unsure* positions | add and mark more photos of that clip type, train again; check focus and lighting |
| two clip types are confused | mark more examples of both; see `models/clip_detector_confusion.png` |
| "Save part" stays disabled | a position has no box or two boxes; the pattern panel says which |
| status bar says template matching | no trained model yet, or it is below `detector.min_model_accuracy` — train (longer) |
| `ALERT FAULT` on screen | check the relay port / PLC address; Settings → Alerts → Test tower light |
| forgot the login password | `python main.py set-secret --login` on the station PC |
