# Mixed Reality Sensory Mirror

Prototype for RQ2: how do different visual interaction modalities (proxemic,
object recognition, intentional facial gesture) contribute to sensory
experience for people with intellectual disabilities. See the implementation
plan for design rationale, thematic grounding, and evaluation.

## Architecture

```
[Camera] -> [FastAPI backend: OpenCV ArUco (-> fine-tuned YOLO11s) + MediaPipe] -> WebSocket -> [p5.js frontend]
                                                                |
                                              [Researcher control panel: live config]
```

Object recognition is mid-migration: the backend still runs the ArUco
detector today, but it's being replaced by a YOLO11s model fine-tuned on
real photos of the 5 sensory items — see "Object recognition dataset &
model" below for what's done and what's left.

Each modality (proxemic, object, gesture) is an independent backend module
publishing its own event stream over the same WebSocket, so they can be
evaluated separately — but only one runs at a time, selected live via the
control panel's `active_modality` setting, enforced by a single shared
camera loop rather than by convention. Content mappings live in
`backend/config/content_config.json`, not in code, so the researcher can
reconfigure responses live without redeploying.

## Structure

```
backend/
  main.py                    FastAPI app, WebSocket broadcast, shared camera loop
  modules/
    proxemic.py               Phase 1 — MediaPipe Pose, implemented
    gesture.py                 Phase 3 — MediaPipe Face Mesh, implemented
    object_recognition.py     Phase 2 — ArUco, implemented (mid-migration to YOLO11s)
  scripts/
    generate_aruco_markers.py generates printable marker PNGs (DICT_4X4_50)
    prepare_dataset.py        builds the YOLO training set from dataset/raw/ — see below
  dataset/                    object recognition training data — see below (gitignored)
  assets/
    markers/                  marker_0.png .. marker_5.png, printable
  config/
    content_config.json       modality -> visual/audio response mapping
    config_manager.py         load/save + change logging for hot-reload
  logs/
    config_changes.log        timestamped log of every config edit
frontend/
  index.html, style.css       p5.js client shell
  core.js                     shared p5 lifecycle, WebSocket wiring, message routing
  modules/
    geometry.js                shared drawing helpers
    audio.js                   procedural Web Audio payoffs (no audio assets yet)
    proxemic.js                Phase 1 rendering, implemented
    gesture.js                 Phase 3 rendering, implemented
    object.js                  Phase 2 rendering, implemented (ArUco marker_id -> visual)
control_panel/
  index.html, style.css, app.js   Phase 4 — live modality switcher, all three
                                    modalities selectable (proxemic/gesture/object)
```

## Build order (do not skip ahead)

1. **Proxemic interaction** — implemented, backend and frontend.
2. **Intentional facial gesture** — implemented, backend and frontend.
3. **Object recognition** — implemented and running on ArUco markers today
   (`marker_0`-`marker_5` in `content_config.json`). Mid-migration to a
   YOLO11s model fine-tuned on real photos of the 5 sensory items instead of
   printed tags — dataset built, fully hand-annotated, and split; training
   and the backend/frontend swap are the remaining steps. See "Object
   recognition dataset & model" below.
4. **Researcher control panel** — built for all three modalities, live
   switching and change logging working for each.

## Running the backend

Requires **Python 3.12** — mediapipe's legacy `solutions` API (used here for
Pose/Face Mesh) has no Windows wheel with that API for Python 3.13 yet; 3.13
only gets mediapipe 1.0's newer Tasks API.

```bash
cd backend
py -3.12 -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/python -m uvicorn main:app --reload
```

The camera is picked up by device name (`CAMERA_NAME` in `main.py`, default
`"Brio 100"`) rather than a hardcoded index, since Windows can silently
reorder DirectShow indices as camera-like devices come and go.

Then open:
- **http://localhost:8000** — the mirror itself. The p5.js frontend connects
  over `/ws` and renders whichever modality is active: the three proxemic
  states (`close_still`, `active_movement`, `retreat`) with smoothly
  interpolating visuals, or the two held facial gestures (`held_smile`,
  `mouth_open`) with a hold-progress buildup before they trigger.
- **http://localhost:8000/control** — the researcher control panel, for
  switching the active modality live and seeing every change logged.

Edit `backend/config/content_config.json` while the server is running to see
the color/visual mapping hot-reload without a restart.

## Object recognition markers (current — being replaced)

`backend/scripts/generate_aruco_markers.py` generates printable `DICT_4X4_50`
marker PNGs (with a white quiet-zone border) into `backend/assets/markers/`.
The IDs currently assigned in `content_config.json`'s `objects` section:

| Marker | Item |
|---|---|
| `marker_0` | Black cat plush (placeholder, used only for early testing) |
| `marker_1` | Pop It squishy ball — magenta/pink |
| `marker_2` | Pop It squishy ball — cyan/pastel multi |
| `marker_3` | Pop It squishy ball — mint/red/blue |
| `marker_4` | Noomi Squeezibo fabric squishy (purple cat) |
| `marker_5` | Hand caterpillar fidget |

Run from `backend/`: `.venv/Scripts/python.exe scripts/generate_aruco_markers.py --ids 0 1 2 3 4 5`

This whole approach — a printed tag stuck to each item — is being replaced by
direct camera-based recognition; see the next section. Once that lands, this
section, `generate_aruco_markers.py`, and `assets/markers/` go away.

## Object recognition dataset & model (in progress)

Replacing ArUco with a YOLO11s model fine-tuned on real photos of the 5
items, so the mirror recognises them directly rather than requiring a
printed marker on each one. This also retires `marker_0` (the black cat
placeholder) — no dataset for it, and no real device will use it.

**Status: dataset done (1,686/1,686 images hand-annotated); training and the
backend/frontend swap not started yet.** `backend/modules/object_recognition.py`
still runs the ArUco detector described above until that lands.

### Dataset layout (`backend/dataset/`, gitignored — regenerate, don't commit)

```
backend/dataset/
  raw/<class>/*.jpg + *.txt   source photos + hand-drawn YOLO labels (LabelImg), untouched by the pipeline
  trainval/train/images,labels/   72% of images (grouped by capture burst — see below)
  trainval/val/images,labels/     18%
  test/images,labels/             10%, held out — never used for training/tuning
  data.yaml                   ultralytics dataset config
  split_manifest.json         per-image split/bbox/source record, for audit
  _bbox_preview/<class>/      every image with its box drawn on, for review
```

Class id (fixed by list order in `backend/scripts/prepare_dataset.py`,
matches `data.yaml` and will become `marker_0`..`marker_4` once the backend
swap lands):

| class id | folder (`dataset/raw/<class>/`) | item |
|---|---|---|
| 0 | `pink_pop_it` | Pop It squishy ball — magenta/pink |
| 1 | `blue_green_pop_it` | Pop It squishy ball — cyan/pastel multi |
| 2 | `blue_red_pop_it` | Pop It squishy ball — blue/white/red |
| 3 | `noomi_squeezibo_cat` | Noomi Squeezibo fabric squishy (purple cat) |
| 4 | `caterpillar` | Hand caterpillar fidget |

### Annotating (`LabelImg`)

Boxes are hand-drawn with `LabelImg` (`pip install labelImg` into
`backend/.venv` — not in `requirements.txt`, since it's a one-time authoring
tool, not a runtime dependency of the app). From `backend/`:

```
.venv\Scripts\labelImg.exe dataset\raw\<class> dataset\raw\<class>\classes.txt dataset\raw\<class>
```

A `classes.txt` already sits in each `raw/<class>/` folder (same fixed class
order as above) so ids stay consistent across folders. One-time setup in the
app: toggle the format button (left panel) from "PascalVOC" to **"YOLO"**,
and enable **View → Auto Save mode**. Then per image: `w` draw a box, pick
the class (remembers the last-used one), `d` next image — labels save as
`<image>.txt` next to the source photo.

> **Known issue:** this LabelImg release (1.8.6, 2022) predates the PyQt5
> version it currently pulls in on install, which is stricter about
> int-vs-float arguments to a handful of Qt calls (zoom, scroll, box drawing,
> label text) and crashes on them as installed. Fixed locally by patching
> `labelImg.py`/`libs/canvas.py`/`libs/shape.py` in this venv's
> `site-packages` to `int()`-cast those values — if the venv is recreated,
> `pip install labelImg` will need those same call sites patched again (also
> install `setuptools` first, since Python 3.12 dropped stdlib `distutils`,
> which this LabelImg version still imports directly).

### Building the training set (`prepare_dataset.py`)

```
cd backend
.venv/Scripts/python.exe scripts/prepare_dataset.py
```

Safe to re-run anytime (fully regenerates `trainval/`, `test/`, and the
preview images from whatever's currently in `raw/`). For any photo with a
matching hand-drawn `<image>.txt`, that box is trusted outright; any photo
without one yet falls back to an auto-generated box (OpenCV GrabCut, seeded
from a center-inset rectangle) — meant as a bootstrap, not a substitute for
review. `_bbox_preview/<class>/` colour-codes every box by source (green =
manual, blue = GrabCut, orange = GrabCut fallback) so quality can be checked
before spending GPU time training.

Splitting is stratified by class *and* grouped by capture burst: each
class's raw photos come from 5-10 distinct shooting sessions, with frames
within a burst seconds apart and near-duplicates of each other. Splitting
whole bursts (rather than individual photos) into one side of train/val/test
avoids leaking near-twin frames across the split, at the cost of the actual
per-class ratios drifting a bit from the nominal 72/18/10 target (uneven,
small numbers of bursts per class) — logged per class when the script runs.

### Training (not yet built)

Next: fine-tune `yolo11s.pt` (COCO-pretrained) on `dataset/data.yaml` via
`ultralytics`, evaluate on the held-out `test` split, and swap
`backend/modules/object_recognition.py` to load the result instead of
running `cv2.aruco`.

## Key design commitments

- No emotion classification. Facial input (Phase 3) is intentional,
  user-initiated expression — a fixed vocabulary of held, exaggerated
  expressions — as opposed to inferred internal state.
- No auto-personalisation. The researcher reconfigures content live via the
  control panel (Phase 4) in response to observed engagement.
- Each modality stays fully separable, and exactly one is active at a time —
  selected live via the control panel, not run concurrently.
