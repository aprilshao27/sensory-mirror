# Mixed Reality Sensory Mirror

Prototype for RQ2: how do different visual interaction modalities (proxemic,
object recognition, intentional facial gesture) contribute to sensory
experience for people with intellectual disabilities. See the implementation
plan for design rationale, thematic grounding, and evaluation.

## Architecture

```
[Camera] -> [FastAPI backend: OpenCV ArUco + MediaPipe] -> WebSocket -> [p5.js frontend]
                                                                |
                                              [Researcher control panel: live config]
```

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
    object_recognition.py     Phase 2 — ArUco, stub (markers ready, detector not built)
  scripts/
    generate_aruco_markers.py generates printable marker PNGs (DICT_4X4_50)
  assets/
    markers/                  marker_0.png .. marker_3.png, printable
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
    object.js                  Phase 2 rendering, stub
control_panel/
  index.html, style.css, app.js   Phase 4 — live modality switcher (proxemic/gesture);
                                    object recognition option present but disabled
```

## Build order (do not skip ahead)

1. **Proxemic interaction** — implemented, backend and frontend.
2. **Intentional facial gesture** — implemented, backend and frontend.
3. **Object recognition (ArUco)** — physical markers chosen, printed, and
   configured (`marker_0`-`marker_3` in `content_config.json`); the
   `ObjectRecognitionDetector` itself and its frontend rendering are still
   stubs.
4. **Researcher control panel** — built for the two working modalities;
   object recognition's radio stays disabled until Phase 2 lands.

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

## Object recognition markers

`backend/scripts/generate_aruco_markers.py` generates printable `DICT_4X4_50`
marker PNGs (with a white quiet-zone border) into `backend/assets/markers/`.
The four IDs currently assigned in `content_config.json`'s `objects` section:

| Marker | Item |
|---|---|
| `marker_0` | Weighted cat plush (stand-in for testing while other items are on order) |
| `marker_1` | Pop It squishy ball fidget |
| `marker_2` | Noomi Squeezibo fabric squishy |
| `marker_3` | Hand caterpillar fidget |

Run from `backend/`: `.venv/Scripts/python.exe scripts/generate_aruco_markers.py --ids 0 1 2 3`

## Key design commitments

- No emotion classification. Facial input (Phase 3) is intentional,
  user-initiated expression — a fixed vocabulary of held, exaggerated
  expressions — as opposed to inferred internal state.
- No auto-personalisation. The researcher reconfigures content live via the
  control panel (Phase 4) in response to observed engagement.
- Each modality stays fully separable, and exactly one is active at a time —
  selected live via the control panel, not run concurrently.
