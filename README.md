# FC-Comparator

Vision inspection station for wire harness boards. An overhead camera photographs the harness cables laid out on a white
board. The station finds every plastic clip, works out which cable and row each clip belongs to, and compares the result
with the master of the scanned part number. When anything is wrong it raises an alert on screen, on a tower light and on a
buzzer, and every result is logged.

The station is **trained by marking images**. You upload photos of boards and draw a box around each clip, choosing its
type. The station learns the clip types from those marks. A marked known-good board becomes a part's master in one click,
which defines both its clip pattern and where the clips sit. That makes it practical to handle many part numbers with
different layouts.

```
 upload photos ─► mark clips (box + type) ─► good board = part master ─► train (optional YOLO)
                                                                                │
 scan P/N ─► capture ─► detect clips ─► place on the part's layout ─► compare ─► OK / NG ─► light, buzzer, lock, log
```

## Contents

1. [How it works](#1-how-it-works)
2. [Installation](#2-installation)
3. [Quick start with the sample data](#3-quick-start-with-the-sample-data)
4. [Camera setup](#4-camera-setup)
5. [Marking images](#5-marking-images)
6. [Adding a part number](#6-adding-a-part-number)
7. [Training the model](#7-training-the-model)
8. [Operating the station](#8-operating-the-station)
9. [Wiring the tower light, buzzer and foot pedal](#9-wiring-the-tower-light-buzzer-and-foot-pedal)
10. [Logging, reports and export](#10-logging-reports-and-export)
11. [Configuration](#11-configuration)
12. [Command line](#12-command-line)
13. [Tests](#13-tests)
14. [Performance](#14-performance)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. How it works

**Clip types.** The detector learns a configurable list of clip types (Settings → Clip types). The defaults are `round`,
`fork_left`, `fork_right` and `small`. A fork is marked with the direction its open side faces. Two extra kinds of setting
extend the list:

* **Groups** let one expected label accept several types. For example `fork = fork_left, fork_right` means a pattern
  that says `fork` doesn't care about the direction.
* **Mirror pairs** say that two types are left/right mirror images, so every marked `fork_left` also teaches
  `fork_right`.

**Detectors.**

* **Template matching** is used from the first few marked images. Your marked boxes are the templates. The station also
  samples and *mines* "not a clip" patches from your images, such as bare cable, connectors and board marks. These
  compete as a background class, so bare cable isn't mistaken for a clip. With 4 marked boards it found every clip on
  unseen, shifted boards with no false positives.
* **YOLO (Ultralytics YOLO11)** is trained on the marked images (§7). It is faster and more robust on real photos.
  `detector.backend: auto` switches to the trained model only once its validation accuracy reaches
  `detector.min_model_accuracy` (95%). An under-trained model therefore never silently replaces working template
  matching; the reason is shown in the sidebar.

**Automatic layout.** A part's master image records where each clip sits (cable, row). At inspection the detected clips
are fitted onto that layout:

* A translation vote handles board shifts.
* A rotation/scale fit uses only clips of the expected type, so wrong clips can't pull it off.
* Optimal assignment pairs clips with positions.
* Positions with no clip are **missing**. Confident clips where the master has none are **unexpected**.
* A board that doesn't fit the layout is reported as a **likely wrong part**.

Parts that have only a typed pattern and no master image still work: clips are grouped into columns and rows
automatically.

**Never OK:** a missing clip, a clip below `detector.confidence_threshold` (reported as *uncertain*, with the detector's
guess), an unexpected clip in the clip area, a board that can't be placed, an unknown part number, or a detector error.

**Project layout**

```
fc_comparator/
  models.py        Taxonomy (clip types), Box, Layout, PartNumber, PositionResult, InspectionReport
  config.py        typed YAML config (every setting, clip types and part numbers in one file)
  annotations.py   store of uploaded images and the boxes marked on them
  layout.py        grid inference, master from a marked board, layout fitting
  parts.py         part number from a marked good image
  detect/          template-matching detector, YOLO detector, auto selection with the accuracy guard
  training.py      YOLO data-set export (with fork mirroring), training, evaluation + confusion matrix
  compare/         master (A), cross-cable majority (B), both
  pipeline.py      Inspector (image → report) and Station (capture, alert, lock, logging)
  capture/ alert/ storage/ annotate.py barcode.py security.py synthetic.py cli.py
  ui/              CustomTkinter station UI (pages/, marking canvas in annotator.py)
scripts/           train_detector.py, make_samples.py
config/config.yaml sample configuration with three parts
samples/           marked sample images, test boards + manifest (synthetic)
tests/             unit, end-to-end and UI tests
```

---

## 2. Installation

Requires Python 3.11 or newer. Everything runs offline once installed.

### Windows / Linux PC

```bash
git clone https://github.com/nithin2k5/FC-Comparator.git
cd FC-Comparator
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit)

```bash
sudo apt install -y python3-venv python3-tk python3-opencv libgl1
python3 -m venv --system-site-packages .venv      # reuses the system gpiozero / lgpio
source .venv/bin/activate
pip install -r requirements.txt
```

On the Pi, add your user to the `gpio`, `video` and (for USB relays) `dialout` groups. Training a YOLO model on a Pi is
very slow. Train on a PC and copy `models/clip_detector.pt` together with its `_report.json`.

### Offline stations

On a connected machine run `pip download -r requirements.txt -d wheels`. Copy the `wheels/` folder across and run
`pip install --no-index --find-links wheels -r requirements.txt`. For training with pretrained weights, also copy
`yolo11n.pt` into the program folder (§7).

---

## 3. Quick start with the sample data

The sample config uses the `file` source and synthetic data in `samples/`:

* 9 marked images, including a master for each of P001, P002 and P003
* 14 test boards, listed with their expected results in `samples/manifest.yaml`

```bash
python main.py                     # start the station UI
python main.py check               # validate config, marked data, detector and camera
python main.py inspect --image samples/boards/board_P001_mixed.jpg --part P001 --out annotated.jpg
```

```
NG  part=P001  357 ms  detector=template
placement: matched 16/16 (shift 2px, rotation 0.1 deg)
  NG cable 3 row 1: expected round, found fork_right (100%) - fork_right != round
  NG cable 3 row 2: expected fork_left, found fork_right (100%) - wrong orientation: fork_right != fork_left
  NG cable 3 row 4: expected round, found fork_right (100%) - fork_right != round
```

In the UI, use **Load image…** on the Inspect page to try any board from `samples/boards/`.

Default secrets are setup password **5678** and supervisor PIN **1234**. Change both before production, under Settings →
Security.

`python main.py …` and `python -m fc_comparator …` are equivalent.

---

## 4. Camera setup

1. **Mounting.** Mount the camera rigidly overhead, with the lens parallel to the board. A board fixture (end stops) keeps
   shifts small. The layout fit accepts up to `layout.max_shift_px` (200 px) and `layout.max_rotation_deg` (8°).
2. **Lighting.** Use diffuse, even light, such as LED panels at 45° or a ring light. Avoid glare on the clips and
   changing daylight.
3. **Resolution.** Clips should be at least about 40 px across in the image. 1920×1080 over a 60 cm board is plenty.
4. **Fixed settings.** Turn autofocus off and fix focus and exposure (Settings → Station & camera, or `camera.*` in the
   config). Keep the same camera settings for the images you mark and the boards you inspect.
5. Run `python main.py check` and watch the live preview on the Inspect page.

**Inspect** always uses a frame captured after the button press, never a stale buffered one. A disconnected camera
reconnects automatically and the fault is shown on screen.

---

## 5. Marking images

Open **Training data** in the sidebar (setup password).

1. **Upload** with **+ Images** (files), **+ Folder** (a whole folder), or **Capture from camera**. Duplicate images are
   skipped automatically.
2. **Select an image** in the list. Filter by *To mark*, *Marked* or *Good*.
3. **Mark every clip.** Pick the clip type in the palette (or press **1–9**), then drag a box around the clip. Mark
   *every* clip on the image: unmarked clips teach the detector "this is not a clip".

| action | how |
|---|---|
| draw a box | drag on an empty area |
| select / move / resize | click a box, drag it, drag its handles |
| change a box's type | select it, press 1–9 or click the type |
| delete / undo | Del / Ctrl+Z |
| zoom / pan | mouse wheel / right-drag, **Fit** button or F |
| previous / next image | ← / → (with no box selected) |

4. **Auto-mark with detector** pre-marks an image with the current detector. Check every box, fix the type, move or
   delete boxes, and draw any clips it missed. This makes marking many images much faster.
5. Markings **save automatically**. The template detector relearns when you leave the page.

**What to mark:** boards of every part, both fork directions, boards shifted around on the fixture, and some with
missing clips (mark only the clips that are there). About 20 clips per type is enough to start; aim for 50 or more per type
before training YOLO. The Train page shows your progress.

---

## 6. Adding a part number

1. Upload a photo of a **known-good** board of the new part and mark every clip (§5).
2. Click **Create / update part from this image**, then enter the part number and a description.
3. The station works out the cables and rows, shows which clip became which position (C1 R1 …), and lists the clip per
   row. Confirm to save.

The part now has a **pattern**, the clip per row (the same on every cable), and a **layout**, where each clip sits.
Operators can inspect it immediately. If the cables of a good board differ only in fork direction, that row becomes
`fork`, meaning either direction.

**Part numbers** (sidebar) lists all parts with search. Use it to edit descriptions or patterns, see a preview of each
part's master layout, or create a part by hand (pattern only, no master image). Command-line equivalent:

```bash
python main.py add-images photos/P004/ --part P004 --good     # then mark them in the UI
python main.py make-master --image-id <id> --part P004 --description "Harness D"
```

---

## 7. Training the model

Open **Train model** (setup password). The page shows the marked clips per type against the recommended amount.

1. Choose the base model (YOLO11 nano by default), the number of epochs (100) and the image size (960; use 1280 for
   small clips). Press **Start training**. Training runs in the background with progress and an estimated time left, and
   can be stopped at any time. Meanwhile the station keeps inspecting with the current detector.
2. Training exports the marked images as a YOLO data set with a per-image train/validation split. Every image with forks
   is also added mirrored, with `fork_left` and `fork_right` swapped. YOLO's own random flip is turned off, because it
   would keep the wrong direction label.
3. When training finishes, the page reports validation **accuracy**, missed and false clips, per-type precision and
   recall, and a **confusion matrix**. The results are saved next to the model as `clip_detector_report.json` and
   `clip_detector_confusion.png`.
4. If accuracy reaches `detector.min_model_accuracy` (95%), the station switches to the model. Otherwise it keeps
   template matching and tells you to mark more images or train longer.

Command line (same result):

```bash
python scripts/train_detector.py --epochs 100 --imgsz 960
python main.py evaluate            # score the current detector on the marked images
```

**Offline.** The pretrained `yolo11n.pt` is downloaded on first use. On offline stations, copy it into the program
folder beforehand. Otherwise choose *from scratch*, which needs more images and epochs.

**What to expect.** On the synthetic samples (16 marked boards, CPU, 640 px):

* After 30 epochs, the model found the right clips but with low confidence. Median confidence was 0.41, so most clips
  would have been "uncertain", and the accuracy guard correctly kept template matching.
* After 100 epochs (26 minutes), it found **every clip on 6 unseen boards with no misses and no false clips** at the
  station's 0.6 confidence threshold.

Real photos vary more, so mark more images.

---

## 8. Operating the station

**Inspect page**

1. **Choose the part.** Scan the part-number barcode, or type a part number and press Enter, or press **Choose…** to
   search the part list. The part's pattern is shown as coloured chips. Scanning a badge `OP:1234` sets the operator.
   The barcode format is configurable (`barcode.part_pattern`, named group `part`).
2. **Inspect.** Press **INSPECT**, the configured key (default F9; most USB foot pedals can send a key), or the GPIO foot
   pedal.
3. **Read the result.** A large **OK** or **NG** banner appears with the annotated image:
   * **green** boxes: correct positions
   * **red** boxes with `found ≠ expected`: wrong clips
   * **dashed red** boxes: missing clips
   * **orange** boxes: unexpected clips

   The **Findings** list spells each problem out, for example "C3 · R2: found fork_right, expected fork_left".

**Comparison modes** (`compare.mode`):

* `master` (default) compares against the part's pattern.
* `cross` requires all cables in a row to match each other: the majority is the reference, and without a strict
  majority the whole row is flagged.
* `both` requires a position to pass both checks.

**NG lock.** After an NG the station locks. The part can't be changed and the setup pages are disabled until either a
re-inspection of the same part passes, or a supervisor presses **Supervisor acknowledge** and enters the PIN. Both are
logged; wrong PINs too.

---

## 9. Wiring the tower light, buzzer and foot pedal

Select the backend in Settings → Alerts (`alert.backend`). The station drives three outputs: **green lamp**, **red
lamp** and **buzzer**.

| Event | green | red | buzzer |
|---|---|---|---|
| OK | on | off | off |
| NG | off | on | on for `buzzer_seconds` (`pulse`), or until acknowledged (`until_ack`) |
| acknowledged / idle | off | off | off |

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

* Most relay modules are **active-low**, meaning the relay clicks when the input goes to 0 V. For those, turn off
  *Active high*, and check with **Test tower light**.
* The pedal input uses the Pi's internal pull-up with 50 ms debounce.
* Keep the 24 V wiring physically separate from the Pi, and fuse the 24 V supply.

### b) USB relay board (`usb_relay`)

This backend supports CH340-based "LCUS" serial relay boards with 1, 2, 4 or 8 channels (protocol
`A0 <ch> <state> <sum>`). The board appears as `COMx` on Windows or `/dev/ttyUSB0` on Linux. Set the port and channel
numbers, and wire each relay's COM/NO contact as above.

### c) Modbus TCP PLC (`modbus`)

The station writes three coils with function 05 (Write Single Coil) on `host:port`, unit `unit_id`: `green_coil`,
`red_coil` and `buzzer_coil` (0-based addresses). The PLC program maps these coils to its outputs, or uses them for
interlocks such as stopping a conveyor while red is set. Lamps are switched off before the new one is switched on, so
green and red never light together. The connection is kept open and reconnects automatically.

### d) Console / none

`console` logs every output change, for development. `none` does nothing.

Hardware errors never crash an inspection or turn an NG into an OK. They appear as **ALERT FAULT** on the Inspect page
and in the log.

---

## 10. Logging, reports and export

* **Every inspection** is written to SQLite (`storage.database`). Each record holds:
  * timestamp, station, operator, part number, mode, result, duration and detector
  * placement details (shift, message)
  * every position with expected and found clip, confidence and box
  * every unexpected clip

  Older databases are migrated automatically.
* **Evidence images.** The annotated image of every NG is saved as
  `storage.image_dir/YYYY-MM-DD/HHMMSS_<id>_<part>_NG.jpg`, plus every Nth OK image (`storage.save_ok_every_n`).
* **History & reports page.**
  * Filter by date, part and result.
  * View the stored image and all findings of an inspection.
  * Export to CSV or Excel.
  * See the daily report: totals, NG rate, results per part, most frequent failing positions and failure kinds, and the
    number of supervisor acknowledgements.
* **Command line:** `python main.py report --day 2026-09-29`, `python main.py export --out history.xlsx --from … --to …`.

---

## 11. Configuration

All settings live in `config/config.yaml` and are edited from the Settings page. The file is written atomically, and
relative paths are resolved from the config file's folder.

| key | meaning |
|---|---|
| `taxonomy.classes / groups / mirror` | clip types, groups such as `fork`, and mirror pairs |
| `detector.backend` | `auto` (trained model if accurate enough, else templates), `yolo`, `template` |
| `detector.confidence_threshold` | below this a clip is *uncertain* (NG); default 0.6 |
| `detector.min_model_accuracy` | validation accuracy a trained model needs before `auto` uses it; 0.95 |
| `detector.template.*` | template scale, templates per class, negatives, minimum correlation |
| `layout.*` | match tolerance (clip sizes), max shift (px) and rotation (deg) |
| `annotations.dir` | where uploaded images and marks are stored |
| `training.*` | base model, epochs, image size, batch, validation split, mirroring |
| `compare.mode` | `master`, `cross`, `both` |
| `camera.*`, `alert.*`, `security.*`, `storage.*`, `barcode.*`, `trigger.*`, `ui.*` | as named |
| `parts.<code>` | `description`, `cables`, `pattern` (clip per row) and, from a master image, `layout` + `master_image` |

---

## 12. Command line

```
python main.py [-c config.yaml] [gui]                  start the station UI (default)
python main.py inspect --image F --part P [--out A.jpg] [--no-store] [-v]   exit code 0 = OK, 1 = NG
python main.py add-images PATH... [--part P] [--good]  upload images for marking
python main.py make-master --image-id ID --part P      part pattern + layout from a marked good board
python main.py train [--epochs N] [--imgsz N] [--base yolo11n.pt]
python main.py evaluate                                score the current detector on the marked images
python main.py report [--day YYYY-MM-DD]
python main.py export --out F.csv|F.xlsx [--from D] [--to D]
python main.py set-secret --pin | --setup
python main.py check
python scripts/make_samples.py                         regenerate samples/ and the sample parts
```

---

## 13. Tests

```bash
pytest                   # everything (about 3 minutes)
pytest -m "not slow"     # skip the YOLO training test
pytest -m "not ui"       # skip the UI tests (e.g. on a headless machine)
```

* `test_e2e.py` runs every sample board through the full station and checks the exact findings. It also covers:
  * shifted boards, missing and uncertain clips, and parts without a master layout
  * the NG lock, logging and evidence images, and CLI exit codes
  * a complete **new part set up from a marked image**
* `test_layout.py` covers grid inference, masters from marked boards, and layout fitting under shift/rotation with
  missing, extra and wrong clips.
* `test_detect.py` covers template detection accuracy on unseen boards, fork direction, missing and covered clips, speed,
  and the accuracy guard against under-trained models.
* `test_training.py` covers data-set export with mirroring, evaluation and the confusion matrix, and a short real YOLO
  training run.
* `test_ui.py` drives the real window: scan, inspect, NG lock, marking boxes with the mouse, creating a part from a
  marked image, the part editor, and settings including renaming clip types.
* Also: `test_compare.py`, `test_annotations.py`, `test_alert.py` (fake GPIO, serial port and PLC), `test_storage.py`
  (including the database migration), `test_config.py`, `test_capture.py` and `test_barcode.py`.

---

## 14. Performance

Measured on an Intel i5-13420H laptop, CPU only, 1280×960 boards:

| step | time |
|---|---|
| template detector, whole board (≈40 templates + 20 negatives at 0.35 scale) | ~200–300 ms |
| trained YOLO11n detector, whole board at 640 px | ~85 ms |
| layout fit + compare + annotate | < 20 ms |
| template detector rebuild after marking changes (runs in the background) | ~4 s |
| YOLO training, 16 images, 100 epochs, 640 px | ~26 min |

The alert fires as soon as the verdict is known, before the image is annotated or saved. A Raspberry Pi 5 has not been
measured. Expect roughly 3–4× the PC inference times, which is still within the 1 s budget with a trained model. If it
runs tight, lower `detector.imgsz` or export the model to NCNN/ONNX.

---

## 15. Troubleshooting

| symptom | fix |
|---|---|
| "placement failed: only N/M clips fit" | wrong part on the board, board outside the view, or shifted beyond `layout.max_shift_px` |
| "board does not match … wrong part?" | the clips found belong to another part number |
| many *uncertain* positions | mark more images of that clip type and retrain; check focus and lighting |
| a clip type is confused with another | mark more examples of both; check the confusion matrix on the Train page |
| sidebar says the model reached only N% accuracy | mark more images or train longer; the station keeps template matching meanwhile |
| false clips on connectors or cable | make sure marked images are *completely* marked; add images showing those areas |
| `ALERT FAULT` on screen | check the relay port / PLC address; use Settings → Alerts → Test tower light |
| forgot the setup password | `python main.py set-secret --setup` on the station PC |
