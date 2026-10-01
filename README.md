# FC-Comparator

Vision inspection station for wire harness boards. An overhead camera photographs the harness laid out on the board.
Every part number has its own trained model, which finds the objects you labelled for that part (clips, forks, tape
...). The station works out which cable and row each object belongs to and compares every position with the part's
master. When anything is wrong it shows **NG**, switches the tower light to red, sounds the buzzer, locks the station
and logs the result.

The UI is plain **tkinter** (part of Python). It opens straight to the Inspect tab, with no login. There are three tabs:

```
 Inspect      scan part number ─► INSPECT / foot pedal ─► OK / NG ─► tower light, buzzer, lock, log     (no login)
 History      every inspection, its findings and image; daily report; CSV / Excel export                (no login)
 Model Setup  per part number: images ─► labels ─► train ─► use model ─► master; station settings    (model login)
```

## Contents

1. [Installation and first start](#1-installation-and-first-start)
2. [Project layout](#2-project-layout)
3. [Model Setup](#3-model-setup)
4. [Inspect tab](#4-inspect-tab)
5. [History, reports and export](#5-history-reports-and-export)
6. [Settings and configuration](#6-settings-and-configuration)
7. [Camera setup](#7-camera-setup)
8. [Wiring the tower light, buzzer and foot pedal](#8-wiring-the-tower-light-buzzer-and-foot-pedal)
9. [Command line](#9-command-line)
10. [Tests](#10-tests)
11. [Troubleshooting](#11-troubleshooting)

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

**Model login** (Model Setup and the commands that change a model): user **nice**, password **nice1234**. The
supervisor PIN (to release an NG lock) is **1234**. Change both under Model Setup → Settings before production.

**Offline stations.** Put `yolo11n.pt` (the pretrained base for training) into `models/`; otherwise Ultralytics
downloads it the first time you train.

**The FCC photos in `dataset/`** (20 boards, labelled) can become a part's starting images, labels included:
`python main.py add-images --part <code> dataset`.

---

## 2. Project layout

```
main.py                    entry point:  python main.py [command]
config/config.yaml         the station's settings (camera, detection, alerts, login, PIN, storage)
data/                      [not in git]
  parts/<code>/            one folder per part number:
    part.json                description, active model, master (pattern + layout)
    index.json               the part's labels and the boxes on every image
    images/                  the part's images
    models/v001.pt ...       every trained version + v001.json (its score) + v001_confusion.png
  inspections.db           inspection history and event log
  images/                  evidence images
dataset/                   the 20 FCC photos with their boxes (import into a part with add-images)
models/                    yolo11n.pt, the pretrained base for training                         [not in git]
fc_comparator/
  config.py                typed YAML configuration
  cli.py                   command line
  core/                    pure logic, no I/O
    models.py              labels (+ left/right twins), Box, Layout, master (PartNumber), inspection results
    layout.py              cables x rows grid, master from a good board, fitting a master onto a new photo
    compare.py             found vs expected at every position
    parts.py               master from the objects of a good board
  vision/                  images and detection
    camera.py              OpenCV camera / image file sources
    parts.py               part folders: images, labels, model versions, master, readiness
    dataset.py             a part's images + labelled boxes + label set
    detect/                the part's YOLO model, template matching (trials, tests)
    training.py            YOLO data-set export (with mirroring), training, scoring
    drawing.py             OK/NG boxes drawn on the result image
    synthetic.py           synthetic boards for the tests
  station/                 the inspection station
    inspector.py           image -> detect -> place -> compare -> report
    station.py             part's model + capture + inspector + alert + NG lock + logging
    lock.py security.py barcode.py
    alerts/                tower light / buzzer: GPIO, USB relay, Modbus TCP, console
    storage/               SQLite history + event log, evidence images, CSV / Excel export
  ui/                      tkinter UI
    app.py main_window.py style.py widgets.py box_editor.py
    pages/                 inspect.py history.py setup.py (+ setup_images, setup_train, setup_master, settings)
scripts/make_test_fixtures.py   regenerates tests/fixtures (synthetic parts and boards)
tests/                     unit, end-to-end and UI tests
```

---

## 3. Model Setup

Model Setup is the only protected area. Opening the tab asks for the model login. The session ends when you leave
the tab, or after 10 minutes without input (`auth.setup_timeout_min`). While the station is NG-locked, Model Setup
cannot be opened. Everything about creating or changing a model happens here, one part number at a time.

**a. Select the part number.** Pick one from the list. To add a part number, press **New part number...** and type
or scan its code, then give an optional description (typing / scanning a new code into the list and pressing Enter
works too). Codes use letters, digits, `.`, `_` and `-`. The line below shows the part's model in use, its accuracy, when it was trained, how
many images are labelled, the master, and whether the part is ready for inspection.

**b. Add images** (Images & labels): **Add images...**, **Add folder...**, or **Capture from camera...** (live
preview, every Capture adds a frame). Images are stored under the part, in `data/parts/<code>/images`.

**c. Label objects** (Images & labels): draw a box around every object the station should check and give it a label.
You create the label names yourself (`red_clip`, `fork_left`, `tape` ...); there is no fixed list, and each part keeps
its own labels. The first box drawn without a label asks for a new label name. Leave anything you don't care about
unboxed - it is ignored. Click a box to select it, drag to move, drag the handles to resize, Delete removes it, 1-9
picks a label, Ctrl+Z undoes; wheel zooms, right-drag pans, ← / → go to the previous / next image. Everything is saved
automatically. Once a model exists, **Find objects** pre-marks an image with it, so you only correct mistakes.
**Labels...** renames, merges (rename to an existing name) or deletes a label on every image of the part (and in its
master).

**d. Train** (Train & models): runs in the background with a progress bar and **Cancel**. 20 % of the labelled images
are held back to score the model. If a label name contains `left` / `right`, mirrored copies of the images are added
with those labels swapped, so both sides are learned. Training needs at least 5 labelled images
(`training.min_images`) and every label at least 3 times (`training.min_per_label`); the panel says what is missing.

**e. Use the model.** If the new model scores at least `detector.min_model_accuracy` (95 %) on the held-back images,
**Use this model** makes it the part's active model. If it scores lower, the previous model stays active and the panel
lists the labels it struggles with. Every version is kept: select one and **Use selected version** to switch back.

**f. Set the master** (Master): photograph a known-good board (**New master from photo...** / **from camera**). The
part's model finds its objects (or draw them by hand); they are sorted into cables and rows. The master keeps the
pattern - which label belongs at each (cable, row) - and the layout - where each position sits in the image. **Save
master** stays disabled until every position has exactly one object. The photo is added to the part's images.

**g. Settings**: camera, detection thresholds, tower-light hardware, the model login and the supervisor PIN (§6).

Every model change is logged with the time and the part number: part created / deleted, labels changed, training,
activating a model, rolling back, and saving a master.

---

## 4. Inspect tab

No login.

1. **Part number.** Scan the barcode into the part box (the scanner types the code + Enter), or pick from the list.
   The station loads that part's active model and master. If it has neither, the screen says **NOT SET UP** and
   inspection is blocked. Scanning an operator badge `OP:1234` sets the operator name on the results (it is not a
   login). The barcode format is `barcode.part_pattern`.
2. **INSPECT**, the trigger key (F9 - most USB foot pedals can send a key) or the GPIO foot pedal. **Inspect an image
   file...** runs the same inspection on a saved photo.
3. **What happens:** the part's model finds every labelled object; the master layout is fitted onto the photo
   (shift, rotation and scale within `layout.max_shift_px` and `layout.max_rotation_deg`); each position is paired with
   at most one object; objects below `detector.confidence_threshold` (0.6) count as *unsure*; every position is
   compared with the master's label.
4. **Result.** The big banner shows **OK** or **NG**; the photo shows green boxes for correct objects, red boxes with
   `found ≠ expected` for wrong ones, dashed red for missing ones and orange for unexpected ones. The table lists every
   position, problems first. The tower light is set before anything is written to disk.

**Never OK:** a missing, unsure or wrong object (a left/right twin is reported as *wrong orientation*), an unexpected
object inside the board area, a board that does not fit the part's layout (wrong part / board out of view), a part that
is not set up, or a detector error.

**NG lock.** After an NG the station locks: the part can't be changed and Model Setup is closed until a re-inspection
of the same part passes or a supervisor presses **Supervisor unlock (PIN)**. Both are logged, wrong PINs too.

---

## 5. History, reports and export

No login.

* Every inspection is stored in SQLite (`storage.database`): time, station, operator, part, result, duration, model
  version, placement, and every position with expected / found label and confidence.
* The annotated image of every NG is saved (`storage.image_dir/YYYY-MM-DD/…`), plus every Nth OK image
  (`storage.save_ok_every_n`).
* **History tab:** filter by date, part and result; view the image and findings; export CSV or Excel; daily report
  (totals, NG rate, per part, most frequent failing positions, supervisor unlocks, model changes).
* Command line: `python main.py report --day 2026-09-30`, `python main.py export --out history.xlsx --from … --to …`.

---

## 6. Settings and configuration

Model Setup → Settings edits the everyday settings; everything lives in `config/config.yaml` (written atomically;
relative paths are resolved from the config file's folder). Part numbers are not in the config file - they live in
their folders under `storage.parts_dir`.

| key | meaning |
|---|---|
| `camera.*` | `source` (`camera` or `file`), camera number, resolution, `file_path` for the file source |
| `detector.backend` | `yolo` (each part's active model) or `template` (template matching on the part's labelled images - trials and tests) |
| `detector.confidence_threshold` | below this an object is *unsure* (NG); default 0.6 |
| `detector.min_model_accuracy` | score on the held-back images a model needs before it can be used; 0.95 |
| `layout.*` | match tolerance (object sizes), max board shift (px) and rotation (deg) |
| `training.*` | base model, epochs, image size, batch, held-back share, mirroring, `min_images`, `min_per_label` |
| `auth.*` | model login: `user`, `password` (hash), `setup_timeout_min` |
| `security.*` | supervisor PIN (hash), lock after NG |
| `alert.*` | tower light backend and its hardware settings (§8) |
| `storage.*` | `parts_dir`, `database`, `image_dir`, how many OK images to keep |
| `barcode.*`, `trigger.*`, `ui.*` | as named |

---

## 7. Camera setup

1. Mount the camera rigidly overhead, lens parallel to the board. A fixture with end stops keeps shifts small; the
   layout fit accepts up to `layout.max_shift_px` (200 px) and `layout.max_rotation_deg` (8°).
2. Diffuse, even light (LED panels at 45° or a ring light); avoid glare on the metal forks and changing daylight.
3. Objects should be at least ~30 px across in the image.
4. Turn autofocus off and keep focus / exposure fixed; use the same camera settings for training photos and inspection.
5. Run `python main.py check`, then watch the live image on the Inspect tab.

---

## 8. Wiring the tower light, buzzer and foot pedal

Select the backend in Model Setup → Settings → Alerts (`alert.backend`). The station drives three outputs: **green lamp**, **red
lamp** and **buzzer**.

| Event | green | red | buzzer |
|---|---|---|---|
| OK | on | off | off |
| NG | off | on | on for `buzzer_seconds` (`pulse`), or until unlocked (`until_ack`) |
| unlocked / idle | off | off | off |

**Model Setup → Settings → Alerts → Test tower light** cycles green, then red with buzzer, then off.

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

## 9. Command line

Every UI action also has a command-line form. The commands that change a model ask for the model login (or read
`FCC_USER` / `FCC_PASSWORD` from the environment).

```
python main.py [-c config.yaml] [gui]                       start the station UI (default)
python main.py inspect --image F --part P [--out A.jpg] [--no-store] [-v]   exit code 0 = OK, 1 = NG
python main.py add-images --part P [--description D] PATH...  add images (creates the part if new); a folder
                                                             with an index.json (e.g. dataset/) keeps its labels
python main.py label --part P [--add N] [--rename OLD NEW] [--delete N] [--find]
                                                             list / change labels; --find pre-marks new images
python main.py train --part P [--epochs N] [--imgsz N] [--base yolo11n.pt] [--device cpu|0]
python main.py use-model --part P [--version v002]          put a version into use (default: newest); also rollback
python main.py set-master --part P --image F | --image-id ID
python main.py report [--day YYYY-MM-DD]
python main.py export --out F.csv|F.xlsx [--from D] [--to D]
python main.py set-secret --pin | --login [--user NAME]     on the station PC; no login needed
python main.py check                                         validate config, parts and camera
```

---

## 10. Tests

```bash
pytest                   # everything (about 3 minutes)
pytest -m "not slow"     # skip the short YOLO training test
pytest -m "not ui"       # skip the UI tests (headless machines)
```

The tests use synthetic parts (`tests/fixtures`, regenerate with `python scripts/make_test_fixtures.py`) with
template matching, so they run without trained models.

* `test_e2e.py` runs every fixture board through the full station and checks the exact findings; also shifted boards,
  missing / unsure objects, parts that are not set up, NG lock, logging, evidence images, and the CLI (new part,
  labels, master, model login, use-model and rollback).
* `test_ui.py` drives the real tkinter window: no login on Inspect, Model Setup login / leaving / timeout, scanning,
  inspection, NG lock and supervisor unlock, history, a new part (images, labels drawn with the mouse, rename, master
  with a missed object fixed in the editor), training, using and rolling back models, settings, login and PIN changes.
* `test_parts.py`, `test_compare.py`, `test_layout.py`, `test_detect.py`, `test_training.py`, `test_annotations.py`,
  `test_alert.py` (fake GPIO, serial port and PLC), `test_storage.py`, `test_config.py`, `test_capture.py`,
  `test_barcode.py`.

---

## 11. Troubleshooting

| symptom | fix |
|---|---|
| Inspect says NOT SET UP | the part has no model in use or no master: Model Setup → Train & models / Master |
| "placement failed: only N/M objects fit" | wrong part on the board, board outside the view, or shifted beyond `layout.max_shift_px` |
| "board does not match … wrong part?" | the objects found belong to another part number |
| many *unsure* positions | label more photos of that part, train again; check focus and lighting |
| two labels are confused | label more examples of both; see the version's `_confusion.png` in `data/parts/<code>/models` |
| Train stays disabled | the panel lists what is missing (labelled images, labels used fewer than 3 times) |
| "Use this model" stays disabled | the model scored below `detector.min_model_accuracy`; the weak labels are listed |
| "Save master" stays disabled | a position has no box or two boxes; the pattern panel says which |
| Model Setup tab is greyed out | the station is NG-locked: re-inspect or supervisor PIN |
| `ALERT FAULT` on screen | check the relay port / PLC address; Model Setup → Settings → Test tower light |
| forgot the model login | `python main.py set-secret --login` on the station PC |
