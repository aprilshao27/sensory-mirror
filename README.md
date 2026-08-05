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
toggled and evaluated separately. Content mappings live in
`backend/config/content_config.json`, not in code, so the researcher can
reconfigure responses live without redeploying.

## Structure

```
backend/
  main.py                    FastAPI app, WebSocket broadcast, camera loop
  modules/
    proxemic.py               Phase 1 — MediaPipe Pose, implemented
    object_recognition.py     Phase 2 — ArUco, stub
    gesture.py                Phase 3 — Face Mesh, stub
  config/
    content_config.json       modality -> visual/audio response mapping
    config_manager.py         load/save + change logging for hot-reload
  logs/
    config_changes.log        timestamped log of every config edit
frontend/
  index.html, style.css, sketch.js   p5.js client, placeholder visuals
control_panel/
  index.html                  Phase 4 — not yet built
```

## Build order (do not skip ahead)

1. **Proxemic interaction** (implemented) — validate end-to-end with a real
   webcam before touching anything else.
2. **Object recognition (ArUco)** — second, decoupled module.
3. **Intentional facial gesture** — third module, cut first if time is short.
4. **Researcher control panel** — build last, against a working modality.

## Running Phase 1

Requires **Python 3.12** — mediapipe's legacy `solutions` API (used here for
Pose/Face Mesh) has no Windows wheel with that API for Python 3.13 yet; 3.13
only gets mediapipe 1.0's newer Tasks API.

```bash
cd backend
py -3.12 -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
./.venv/Scripts/python -m uvicorn main:app --reload
```

Then open http://localhost:8000 — the p5.js frontend connects over `/ws` and
renders the three proxemic states (`close_still`, `active_movement`,
`retreat`) with placeholder visuals that interpolate smoothly, no hard
switching.

Edit `backend/config/content_config.json` while the server is running to see
the color/visual mapping hot-reload without a restart.

## Key design commitments

- No emotion classification. Facial input (Phase 3) is intentional,
  user-initiated expression as opposed to inferred internal state.
- No auto-personalisation. The researcher reconfigures content live via the
  control panel (Phase 4) in response to observed engagement.
- Each modality stays fully separable and toggleable.
