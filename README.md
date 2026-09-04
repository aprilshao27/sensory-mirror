# Mixed Reality Sensory Mirror

Prototype for RQ2: how do different visual interaction modalities (proxemic,
object recognition, intentional facial gesture) contribute to sensory
experience for people with intellectual disabilities. See the implementation
plan for design rationale, thematic grounding, and evaluation.

## Architecture

```
[Camera] -> [FastAPI backend: fine-tuned YOLO11s + MediaPipe] -> WebSocket -> [p5.js frontend]
                                                                |
                                              [Researcher control panel: live config]
```

Object recognition runs on a YOLO11s model fine-tuned on real photos of the
5 sensory items, recognising them directly from the camera feed rather than
requiring a printed marker on each one — see "Object recognition dataset &
model" below for the dataset, training, and results.

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
    object_recognition.py     Phase 2 — fine-tuned YOLO11s, implemented
  scripts/
    prepare_dataset.py        builds the YOLO training set from dataset/raw/ — see below
    train_object_model.py     fine-tunes YOLO11s on it, evaluates, publishes weights
  dataset/                    object recognition training data — see below (gitignored)
  models/
    object_recognition.pt     published fine-tuned weights (gitignored, regenerate via the scripts above)
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
    object.js                  Phase 2 rendering, implemented (marker_id -> visual)
control_panel/
  index.html, style.css, app.js   Phase 4 — live modality switcher, all three
                                    modalities selectable (proxemic/gesture/object)
```

## Build order (do not skip ahead)

1. **Proxemic interaction** — implemented, backend and frontend.
2. **Intentional facial gesture** — implemented, backend and frontend.
3. **Object recognition** — implemented, backend and frontend, running on a
   YOLO11s model fine-tuned on real photos of the 5 sensory items
   (`marker_0`-`marker_4` in `content_config.json`) — no printed tags. See
   "Object recognition dataset & model" below for the dataset, training, and
   results.
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

`main.py`'s shared camera loop instantiates all three detectors up front,
regardless of which modality is currently active — so `backend/models/
object_recognition.pt` must exist for the **backend to start at all**, not
just for object recognition to work. On a fresh clone (that file is
gitignored, see below) run `prepare_dataset.py` then `train_object_model.py`
first.

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

## Object recognition dataset & model

`backend/modules/object_recognition.py` runs a YOLO11s model fine-tuned on
real photos of the 5 sensory items, so the mirror recognises them directly
from the camera feed — no printed marker on each item (an earlier ArUco-based
version of this module, and the physical markers it needed, have been
retired). `marker_0` from that earlier version — a black cat plush used only
as a placeholder while the real items were on order — is retired too; there
was never a dataset for it, and no real device uses it.

**Status: done.** Dataset built and 100% hand-annotated (1,686/1,686 images),
YOLO11s fine-tuned and evaluated, weights published to
`backend/models/object_recognition.pt` and loaded by the runtime detector.

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
matches `data.yaml` and `content_config.json`'s `marker_0`..`marker_4` keys
1:1 — the runtime detector publishes the model's class id unchanged, no
translation layer):

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

### Training (`train_object_model.py`)

```
cd backend
.venv/Scripts/python.exe scripts/train_object_model.py
```

Fine-tunes `yolo11s.pt` (COCO-pretrained) on `dataset/data.yaml` — all
layers, not a frozen backbone — for up to 100 epochs at `imgsz=640` with
early stopping (`patience=30`), everything else left at Ultralytics' own
tuned defaults (augmentation, optimizer, batch sizing) rather than
hand-picked. Then evaluates on the held-out `test` split (never touched
during training) and copies the best checkpoint to
`backend/models/object_recognition.pt`, the stable path the runtime loads.

`ultralytics` is in `requirements.txt`, but a plain `pip install -r
requirements.txt` pulls CPU-only `torch` from PyPI. For GPU training, install
the CUDA build explicitly afterwards, matching your driver's supported CUDA
version (`nvidia-smi`) — e.g. for CUDA 13.x:
```
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps torch torchvision
```

**Result** (RTX 4060 laptop GPU, 77 epochs to early-stop): test-set mAP50 =
**0.963**, mAP50-95 = **0.888**, ~8.5ms/image inference — comfortably
real-time for the live camera loop.

| Class | mAP50 | Precision | Recall |
|---|---|---|---|
| pink_pop_it | 0.995 | 0.984 | 1.00 |
| blue_green_pop_it | 0.995 | 0.983 | 1.00 |
| blue_red_pop_it | 0.995 | 1.00 | 0.99 |
| noomi_squeezibo_cat | 0.995 | 0.960 | 1.00 |
| caterpillar | 0.835 | 0.901 | 0.784 |

Caterpillar is the one weak spot — a thin, floppy, beaded item that coils
into much more varied silhouettes than the other four (round pop-its, a
compact plush), so it's a genuinely harder shape to localize consistently
rather than a labeling issue. Precision is still solid (few false
positives); the risk is more the occasional missed detection or loose box.
Being tracked as a candidate for more training data if it proves flaky in
practice, rather than pre-emptively fixed against an offline metric.

At inference, `ObjectRecognitionDetector` also keeps only the
highest-confidence detection per class per frame — Ultralytics' own
class-aware NMS occasionally leaves two overlapping boxes standing for one
physical item (just under its IoU threshold), which would otherwise jitter
that item's tracked position between the two boxes every frame.

## Key design commitments

- No emotion classification. Facial input (Phase 3) is intentional,
  user-initiated expression — a fixed vocabulary of held, exaggerated
  expressions — as opposed to inferred internal state.
- No auto-personalisation. The researcher reconfigures content live via the
  control panel (Phase 4) in response to observed engagement.
- Each modality stays fully separable, and exactly one is active at a time —
  selected live via the control panel, not run concurrently.
