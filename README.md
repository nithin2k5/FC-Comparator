# FC-Comparator

Vision inspection station for wire harness boards. An overhead camera photographs four harness cables on a white
board. The station identifies the clip at each of the 4 × 4 positions, compares every position with the master pattern
of the scanned part number, and raises an alert (screen, tower light, buzzer) as soon as a position is wrong. Every
result is logged for traceability.

```
 scan / select P/N ─► capture ─► align to reference ─► crop 16 ROIs ─► classify (YOLO | templates)
        ─► confidence threshold (low ⇒ "uncertain") ─► compare (master | cross-cable | both)
        ─► OK / NG ─► tower light + buzzer ─► lock on NG ─► SQLite + evidence image
```

## Contents

1. [Project structure and data model](#1-project-structure-and-data-model)
2. [Installation](#2-installation)
3. [Quick start (no camera needed)](#3-quick-start-no-camera-needed)
4. [Camera setup](#4-camera-setup)
5. [Teach workflow](#5-teach-workflow)
6. [Training the YOLO model](#6-training-the-yolo-model)
7. [Operating the station](#7-operating-the-station)
8. [Wiring the tower light, buzzer and foot pedal](#8-wiring-the-tower-light-buzzer-and-foot-pedal)
9. [Logging, reports and export](#9-logging-reports-and-export)
10. [Configuration reference](#10-configuration-reference)
11. [Command line](#11-command-line)
12. [Tests](#12-tests)
13. [Performance](#13-performance)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project structure and data model

```
fc_comparator/
  models.py          labels, Roi, PartNumber, Classification, PositionResult, InspectionReport
  config.py          typed YAML config (one file holds every setting, ROIs and part numbers)
  security.py        PBKDF2-hashed setup password and supervisor PIN
  capture/           CameraSource (OpenCV VideoCapture + grab thread), FileSource
  align/             ORB + RANSAC homography alignment to the reference image, with sanity limits
  classify/          YoloClassifier (primary), TemplateClassifier (fallback), dataset helpers
  compare/           master (A), cross-cable majority (B), both
  alert/             Alert interface; console, gpio, usb_relay, modbus backends; AlertController; StationLock
  storage/           SQLite store, evidence images, CSV/Excel export, daily report
  pipeline.py        Inspector (image → report) and Station (capture, alert, lock, logging)
  annotate.py        green/red boxes with "found ≠ expected"
  barcode.py         keyboard-wedge scan parsing (part number or operator badge)
  synthetic.py       synthetic board renderer (samples, tests, demos)
  cli.py             inspect / report / export / harvest / set-secret / check / gui
  ui/                tkinter touchscreen app (standard library; Pillow for fast preview)
scripts/
  train_classifier.py   train the YOLO classifier and report accuracy with a confusion matrix
  make_samples.py       regenerate samples/
config/config.yaml    sample configuration with two part numbers (P001, P002)
samples/              reference image, test boards + manifest, teach dataset
tests/                unit tests and end-to-end tests
```

### Labels

| Label | Meaning | Allowed in master pattern |
|---|---|---|
| `round` | small round/button clip | yes |
| `fork` | U-shaped fork bracket, **either** orientation | yes |
| `fork_left` / `fork_right` | fork whose open side faces left / right | yes (orientation is then checked) |
| `small` | small arrow / fir-tree clip | yes |
| `missing` | no clip at the position | **no**, never OK |
| `uncertain` | confidence below threshold | **no**, never OK |

The classifier outputs `round`, `fork_left`, `fork_right`, `small` and `missing`. If a master pattern says `fork`, both
orientations pass. If it says `fork_left`, only a left-facing fork passes.

### Main records

* **Roi**: `cable, row, x, y, w, h` in reference-image pixels.
* **PartNumber**: `code, description, pattern[rows]`, e.g. `P002 = [fork, fork, small, fork]`.
* **Classification**: `label, confidence, scores{label: p}, guess` (`guess` holds the best label when the result was
  downgraded to `uncertain`).
* **PositionResult**: `cable, row, expected, found, confidence, ok, reason`.
* **InspectionReport**: `part_number, mode, positions[], alignment, timestamp, duration_ms, operator_id, classifier,
  error`, plus `verdict` (OK / NG).

### SQLite schema

| table | columns |
|---|---|
| `inspections` | id, ts, station, operator_id, part_number, mode, result, ng_count, duration_ms, classifier, aligned, align_shift, align_message, error, image_path |
| `positions` | inspection_id, cable, row, expected, found, confidence, ok, reason |
| `events` | id, ts, kind (`ack`, `ack_failed`, `unlock_pass`, `setup_login`, `part_saved`, ...), actor, detail, inspection_id |

---

## 2. Installation

Requires Python 3.11 or newer. Everything runs offline once installed. The UI uses tkinter, which ships with Python on
Windows and macOS; on Debian, Ubuntu and Raspberry Pi OS install it with `sudo apt install python3-tk`.

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

`gpiozero` ships with Raspberry Pi OS. On the Pi, add yourself to the `gpio` and `video` groups.

### Offline stations

Install on a connected machine with `pip download -r requirements.txt -d wheels`, copy the `wheels/` folder, then run
`pip install --no-index --find-links wheels -r requirements.txt`. If you want to train with pretrained weights, copy
`yolo11n-cls.pt` too (see §6).

---

## 3. Quick start (no camera needed)

The sample config uses the `file` source and the synthetic sample boards in `samples/`:

```bash
python -m fc_comparator check                                    # validate config, dataset, source
python -m fc_comparator inspect --image samples/boards/board_P001_mixed.jpg --part P001 --out annotated.jpg
python -m fc_comparator                                          # start the UI
```

```
NG  part=P001  112 ms  classifier=template
alignment: aligned (shift 0.2px, 512 inliers)
  NG cable 3 row 1: expected round, found fork_right (100%) - fork_right != round
  NG cable 3 row 4: expected round, found fork_right (100%) - fork_right != round
```

In the UI, press **Load image…** on the Inspect screen and pick any file from `samples/boards/`.
`samples/manifest.yaml` lists the expected NG positions for each board.

Default secrets: setup password **5678**, supervisor PIN **1234**. Change both before production (Settings screen, or
`python -m fc_comparator set-secret --setup` / `--pin`).

---

## 4. Camera setup

1. **Mounting.** Mount the camera rigidly overhead, with the lens parallel to the board. Use a fixed board fixture
   (end stops or locating pins) so boards land in the same place within about ±20 px. Alignment corrects the rest.
2. **Lighting.** Use diffuse, even light, such as two LED panels at 45° or a ring light. Avoid glare on the plastic
   clips, and shield the station from sunlight and changing room light.
3. **Fiducials (recommended).** Print or stick a few high-contrast marks, such as small black-and-white checker
   squares, into the board corners. A plain white board gives ORB few features, and fiducials make alignment robust.
4. **Resolution.** Each ROI should be at least about 100 × 100 px. 1920×1080 over a 60 cm board is plenty.
5. **Fixed settings.** Turn autofocus off and set a fixed focus and exposure, so every image looks the same:

   ```yaml
   camera:
     source: camera
     index: 0              # try 1, 2 ... if you have several cameras
     backend: auto         # Windows: dshow (default for auto) or msmf; Linux: v4l2
     width: 1920
     height: 1080
     autofocus: false
     focus: 30             # driver specific (Logitech: 0..255 in steps of 5)
     exposure: -6          # Windows DirectShow: log2 seconds; V4L2: absolute units
   ```

   On Linux, `v4l2-ctl -d /dev/video0 --list-ctrls` shows the valid ranges.
6. Run `python -m fc_comparator check` and watch the live preview on the Inspect screen.

The camera is read by a background thread. **Inspect** always uses a frame captured *after* the button was pressed,
never a stale buffered one. If the camera is unplugged, the source keeps reconnecting and the fault shows on screen.

---

## 5. Teach workflow

Everything below happens in **Setup / Teach**, which requires the setup password. Changes are saved to the config file
immediately and logged in the `events` table.

1. **Reference image.** Place a correctly assembled board, press **Capture from camera**, then **Save as reference
   image**. All ROIs are defined on this image, and every new image is aligned to it.
2. **ROIs.** Select *Cable 1 Row 1* and drag a box around that clip position. Make the box a bit larger than the clip,
   so the clip stays inside when a board is slightly off. The selector advances to the next position automatically.
   Shortcut: draw only C1R1 and the last position, then press **Auto grid** to place the rest evenly. Tap a box to
   select it; **Delete selected** removes it. Press **Save ROIs** when done.
3. **Clip samples.** Build the labeled dataset used by template matching and by YOLO training:
   * **Whole board**: put a known-good board of a part number on the station, capture it, choose the part and the
     direction its forks face, then press **Save all positions from this board**. This saves 16 labeled crops per
     press. Repeat with several boards, part numbers and small position changes.
   * **Single position**: tap a box, pick a label (for example `missing` after removing a clip), then press **Save
     crop of selected position as label**.
   * Aim for **30 or more samples per class**. Include every clip type, both fork orientations and empty positions.
     `fork_right` samples are generated from `fork_left` samples (and vice versa) by mirroring.
4. Press **Reload classifier with new samples**.
5. **Part numbers.** On the **Part numbers** screen (password protected), create, edit and delete part numbers and
   choose the expected clip for each row. Use `fork_left` / `fork_right` only when orientation matters.

A prototype that used template matching with 4 templates per class called a fork "round". This build guards against
that in four ways:

* up to 40 templates per class, with each class scored on its best 3 matches
* a softmax confidence that drops when two classes score close together
* a confidence threshold that turns such positions into `uncertain` (NG)
* the YOLO classifier, which is the default once a model is trained

The command line can harvest samples too:
`python -m fc_comparator harvest --image good_P002.jpg --part P002 --forks right`

---

## 6. Training the YOLO model

The station classifies each ROI crop with an Ultralytics YOLO **classification** model (YOLO11n-cls by default;
YOLOv8n-cls works too). Per-ROI classification is used instead of whole-board detection for three reasons:

* The fixture already fixes where the clips are.
* Labeling is just sorting crops into folders; no boxes need drawing.
* All 16 crops go through the network as one batch.

```bash
python scripts/train_classifier.py data/dataset --epochs 60
# or, to try it on the bundled sample dataset:
python scripts/train_classifier.py samples/dataset --epochs 20
```

Input is one sub-folder of crops per class (`round/ fork_left/ fork_right/ small/ missing/`), exactly what Setup mode
writes. The script:

1. makes a reproducible, stratified 80/20 train/val split (`--val-split`, `--seed`)
2. mirrors forks, so each `fork_left` crop also becomes a flipped `fork_right` crop (and vice versa), inside the same
   split
3. trains with **horizontal flip augmentation disabled**, because a flipped fork would keep the wrong orientation
   label
4. evaluates the best weights on the validation set and prints accuracy and a confusion matrix:

```
Validation accuracy: 100.00%  (52/52)

true \ pred  fork_left fork_right    missing      round      small   recall
fork_left           15          0          0          0          0   100.0%
fork_right           0         15          0          0          0   100.0%
missing              0          0          8          0          0   100.0%
round                0          0          0          7          0   100.0%
small                0          0          0          0          7   100.0%
```

5. writes `models/clip_classifier.pt` plus `_confusion.csv`, `_confusion.png` and `_metrics.json`.

With `classifier.backend: auto`, the station uses the model as soon as `classifier.model_path` exists (Settings → Save,
or restart). Otherwise it falls back to template matching.

Useful options: `--model yolov8n-cls.pt`, `--imgsz 128`, `--device 0` (GPU), `--from-scratch` (no pretrained download,
needs more data), `--no-mirror-forks`.

**Offline:** pretrained weights (`yolo11n-cls.pt`) are downloaded on first use. On an offline machine, copy the file
into the working directory beforehand or use `--from-scratch`. Training can also run on a PC; copy the resulting `.pt`
to the station.

Retrain whenever a new clip type or part appears, and check that the confusion matrix has no off-diagonal entries
between classes that matter.

---

## 7. Operating the station

**Inspect screen**

1. Scan the part-number barcode. The scanner types into the *Scan* field and ends with Enter. Alternatively, pick the
   part from the list. Scanning a badge `OP:1234` sets the operator ID. The part-number regex is configurable
   (`barcode.part_pattern`, named group `part`).
2. Press **INSPECT**, the configured key (`trigger.key`, default F9; most USB foot pedals can send a key), or the GPIO
   foot pedal.
3. A large **OK** (green) or **NG** (red) banner appears, together with the annotated image. Green boxes are correct
   positions. Red boxes carry `found ≠ expected` and a confidence. A table lists every mismatch (cable, row, expected,
   found, confidence).

**Comparison modes** (`compare.mode`)

* `master` (A, default): every cable's row *r* must match `pattern[r]`.
* `cross` (B): all cables in the same row must match each other. The majority is the reference and odd cables are
  flagged. Without a strict majority (e.g. 2 vs 2) the whole row is flagged. `missing` and `uncertain` can never be
  the majority.
* `both`: a position must pass both checks.

**What always makes NG:** a missing clip; a detection below `classifier.confidence_threshold` (reported as
`uncertain`, with the classifier's guess in the reason); an alignment failure (when `alignment.fail_as_ng`); an unknown
part number; a classifier or capture error.

**NG lock.** After an NG the station locks. The part cannot be changed, and Setup, Part numbers and Settings are
disabled. The lock clears in one of two ways:

* The board is corrected and a **re-inspection of the same part passes** (logged as `unlock_pass`).
* A supervisor presses **Supervisor acknowledge** and enters the PIN (logged as `ack`; wrong PINs are logged as
  `ack_failed`).

---

## 8. Wiring the tower light, buzzer and foot pedal

Select the backend in Settings (`alert.backend`). The station drives three outputs: **green lamp**, **red lamp** and
**buzzer**.

| Event | green | red | buzzer |
|---|---|---|---|
| OK | on | off | off |
| NG | off | on | on for `buzzer_seconds` (`pulse`), or until acknowledged (`until_ack`) |
| acknowledged / idle | off | off | off |

**Settings → Test tower light** cycles green, then red with buzzer, then off.

### a) Raspberry Pi GPIO (`gpio`)

Pi GPIO pins are 3.3 V and about 16 mA. **Never drive a lamp or buzzer directly.** Use an opto-isolated relay module or
a transistor/MOSFET driver board. Typical 24 V DC tower light with a common wire:

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

* Most relay modules are **active-low**, meaning the relay clicks when the input goes to 0 V. For those, set
  `alert.gpio.active_high: false`. Check with **Test tower light**.
* The pedal input uses the Pi's internal pull-up with 50 ms debounce.
* Keep the 24 V wiring physically separate from the Pi, and fuse the 24 V supply.

### b) USB relay board (`usb_relay`)

This backend supports CH340-based "LCUS" serial relay boards with 1, 2, 4 or 8 channels. The protocol is
`A0 <ch> <state> <sum>`. The board shows up as `COMx` on Windows or `/dev/ttyUSB0` on Linux; on Linux, add your user to
the `dialout` group. Set `alert.usb_relay.port` and the channel numbers, then wire each relay's COM/NO contact like the
GPIO example above. A board that doesn't accept the command is reported as an alert fault on screen, and the inspection
itself continues.

### c) Modbus TCP PLC (`modbus`)

The station writes three coils with function 05 (Write Single Coil) on `alert.modbus.host:port`, unit `unit_id`:
`green_coil`, `red_coil` and `buzzer_coil` (0-based addresses; coil 0 is often shown as 00001 in PLC tools). The PLC
program maps these coils to its outputs. It can also use them for interlocks, for example stopping a conveyor while the
red coil is set. Lamps are switched off before the new one is switched on, so green and red are never lit together. The
TCP connection is kept open and reconnects automatically. `timeout` bounds the time spent on a dead PLC.

### d) Console / none

`console` logs every output change, for development. `none` does nothing.

Hardware errors never crash an inspection or turn an NG into an OK. They appear as **ALERT FAULT** on the Inspect
screen and in the log.

---

## 9. Logging, reports and export

* **Every inspection** is written to SQLite (`storage.database`): timestamp, station, operator, part number, mode,
  result, duration, classifier, alignment details, and all 16 position results with confidences.
* **Evidence images:** the annotated image of every NG is saved as
  `storage.image_dir/YYYY-MM-DD/HHMMSS_<id>_<part>_NG.jpg`, plus every Nth OK image (`storage.save_ok_every_n`). Set
  `storage.save_raw_images: true` to keep the unannotated frame too.
* **History / Reports screen:** filter by date, part and result; view the stored image and every position of an
  inspection; export to CSV or Excel. It also shows the daily report: total inspected, OK, NG, NG rate, results by
  part number, most frequent failing positions, most frequent failure kinds (found instead of expected) and the number
  of supervisor acknowledgements.
* **Command line:**

  ```bash
  python -m fc_comparator report --day 2026-09-29
  python -m fc_comparator export --out history.xlsx --from 2026-09-01 --to 2026-09-30
  python -m fc_comparator export --out history.csv          # also writes history_positions.csv
  ```

---

## 10. Configuration reference

The whole configuration lives in `config/config.yaml`. The Setup and Settings screens edit it. The file is written
atomically, so a power cut cannot corrupt it. Relative paths are resolved from the config file's folder. Main keys:

| key | meaning |
|---|---|
| `station.cables`, `station.rows` | layout (default 4 × 4); patterns must have `rows` entries |
| `camera.*` | source (`camera`/`file`), index, backend, resolution, focus, exposure |
| `reference_image`, `rois` | reference image and one ROI per (cable, row) |
| `alignment.enabled`, `max_shift_px`, `max_rotation_deg`, `min_inliers`, `fail_as_ng` | board alignment |
| `classifier.backend` | `auto` (YOLO if model exists, else templates), `yolo`, `template` |
| `classifier.confidence_threshold` | below this a position is `uncertain` (NG); default 0.6 |
| `classifier.template.*` | template size, search margin, max templates per class, softmax temperature, min match score |
| `compare.mode` | `master`, `cross`, `both` |
| `alert.*` | backend, buzzer mode/duration, GPIO pins, USB relay port/channels, Modbus host/coils |
| `security.*` | hashed setup password and supervisor PIN, `lock_on_ng` |
| `storage.*` | database path, image folder, OK image sampling |
| `barcode.*` | part-number regex, operator badge prefix |
| `trigger.*` | inspect key (foot pedal), optional GPIO pedal pin |
| `parts` | part numbers: `description`, `pattern` |

---

## 11. Command line

```
python -m fc_comparator [-c config.yaml] [gui]                 start the UI (default)
python -m fc_comparator inspect --image F --part P [--out A.jpg] [--alert] [--no-store] [-v]
                                                              exit code 0 = OK, 1 = NG
python -m fc_comparator report [--day YYYY-MM-DD]
python -m fc_comparator export --out F.csv|F.xlsx [--from D] [--to D]
python -m fc_comparator harvest --image F --part P [--forks left|right]
python -m fc_comparator set-secret --pin | --setup
python -m fc_comparator check                                 validate config, dataset, classifier, camera
python scripts/train_classifier.py DATASET [--epochs N] [--out models/clip_classifier.pt]
python scripts/make_samples.py                                regenerate samples/
```

---

## 12. Tests

```bash
pytest                   # everything (about 1 minute)
pytest -m "not slow"     # skip the YOLO training test
```

* `test_compare.py`: comparison logic: master, cross-cable majority, both; missing/uncertain never OK; fork
  orientation; configurable layouts.
* `test_e2e.py`: runs the sample boards from `samples/manifest.yaml` through the full station and checks that exactly
  the expected NG positions are reported. It also covers shifted boards (±20 px), missing and low-confidence
  positions, alignment failure, the NG lock and acknowledgement, logging, evidence images, CLI exit codes, and the
  < 1 s inspection and alert latency.
* `test_align.py`, `test_classify.py`, `test_alert.py` (fake GPIO devices, fake serial port and a fake Modbus PLC),
  `test_storage.py`, `test_capture.py`, `test_config.py`, `test_barcode.py`, `test_training.py` and `test_ui.py`
  (drives the real tkinter window; skipped without a display).

---

## 13. Performance

Measured on an Intel i5-13420H laptop (CPU only), 1280×960 board, 16 positions:

| stage | time |
|---|---|
| alignment (ORB on a 1000 px-wide copy + homography + warp) | ~70 ms |
| template classifier (~250 templates incl. mirrored forks, vectorized NCC) | ~70–110 ms |
| YOLO11n-cls, 16 crops in one batch at 128 px | ~200–250 ms |
| **whole inspection** | **~100–160 ms (templates) / ~300 ms (YOLO)** |

The alert fires as soon as the verdict is known, before annotation and disk I/O. The Raspberry Pi 5 has not been
measured here. Expect roughly 3–4× the PC times, which still fits the 1 s budget. If it runs tight, lower
`alignment.work_width` or `classifier.yolo.imgsz` (96), or export the model to NCNN/ONNX.

---

## 14. Troubleshooting

| symptom | fix |
|---|---|
| `alignment failed: only N feature matches` | add fiducials to the board; check lighting; raise `max_shift_px` if the fixture really allows more |
| many `uncertain` positions | teach more samples of that class (§5), retrain YOLO (§6), check focus and lighting; lower the threshold only if the confusion matrix is clean |
| fork orientation wrong | label forks as `fork_left`/`fork_right` (never plain `fork`); keep `fliplr=0` when training (the script does this) |
| camera opens but picture is black or slow on Windows | try `camera.backend: msmf` or `dshow`; set the exposure explicitly |
| `ALERT FAULT` on screen | check the relay port / PLC address; use Settings → Test tower light |
| forgot setup password | edit `config.yaml` on the PC: `python -m fc_comparator set-secret --setup` |
