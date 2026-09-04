import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pygrabber.dshow_graph import FilterGraph
from starlette.websockets import WebSocketState

from config.config_manager import ConfigManager
from modules.gesture import GestureDetector
from modules.object_recognition import ObjectRecognitionDetector
from modules.proxemic import ProxemicDetector

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CONTROL_PANEL_DIR = Path(__file__).parent.parent / "control_panel"

# The mirror's camera, identified by name rather than a hardcoded DirectShow
# index — Windows inserts/removes camera-like devices (e.g. the Windows
# Studio Effects virtual camera) into the enumeration over time, which
# silently shifts every index after it. A name lookup survives that; a
# hardcoded index doesn't.
CAMERA_NAME = "Brio 100"


def _find_camera_index(name: str, fallback: int = 0) -> int:
    try:
        return FilterGraph().get_input_devices().index(name)
    except (ValueError, OSError):
        return fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(camera_loop())
    asyncio.create_task(config_watch_loop())
    yield


app = FastAPI(title="Sensory Mirror Backend", lifespan=lifespan)
config_manager = ConfigManager()


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        for ws in list(self.active):
            if ws.application_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_text(payload)
                except Exception:
                    self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_text(json.dumps({"type": "config_update", "config": config_manager.config}))
    try:
        while True:
            await ws.receive_text()  # client doesn't send anything meaningful yet; keeps the socket open
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/api/config")
async def get_config():
    """Used by the control panel to load current modality/content state on open."""
    return config_manager.config


@app.post("/api/config")
async def update_config(new_config: dict, note: str = ""):
    """Used by the Phase 4 control panel to edit content mappings live."""
    config_manager.save(new_config, note=note)
    await manager.broadcast({"type": "config_update", "config": config_manager.config})
    return {"status": "ok", "config": config_manager.config}


async def camera_loop():
    """Single shared camera capture feeding whichever one modality is
    currently active. Exactly one of "proxemic" / "gesture" / "object" (or
    None) is active at a time — config_manager.config["active_modality"] —
    enforced by construction rather than by convention, since three
    independent on/off flags could otherwise end up with more than one true
    at once (a hand-edited config file, two control panel tabs racing each
    other, etc). DSHOW also won't let two independent VideoCapture handles
    share one physical camera, so a single shared loop is required either way.
    """
    proxemic = ProxemicDetector()
    gesture = GestureDetector()
    object_recognition = ObjectRecognitionDetector()
    gesture_was_active = False
    cap = cv2.VideoCapture(_find_camera_index(CAMERA_NAME), cv2.CAP_DSHOW)
    try:
        while True:
            active = config_manager.config["active_modality"]
            if active not in ("proxemic", "gesture", "object"):
                await asyncio.sleep(0.2)
                continue

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # mirror horizontally: raw capture isn't a mirror by default
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if active == "proxemic":
                result = proxemic.process(frame_rgb)
                if result is not None:
                    await manager.broadcast(
                        {
                            "modality": "proxemic",
                            "state": result["state"],
                            "movement": result["movement"],
                            "landmarks": result["landmarks"],
                            "mask": result["mask"],
                            "timestamp": time.time(),
                        }
                    )
                # else: either a transient tracking dropout still within the
                # detector's grace period, or a confirmed absence it already
                # reported — skip broadcasting so the frontend keeps easing
                # toward whatever state it last received instead of flickering.

            gesture_active = active == "gesture"
            if gesture_active and not gesture_was_active:
                # Discard any hold timer left over from before this was last
                # active — otherwise the next frame reads its stale elapsed
                # time as an already-complete hold and false-triggers.
                gesture.reset()
            gesture_was_active = gesture_active

            if gesture_active:
                result = gesture.process(frame_rgb)
                if result is not None:
                    await manager.broadcast(
                        {
                            "modality": "gesture",
                            "gesture_type": result["gesture_type"],
                            "hold_progress": result["hold_progress"],
                            "triggered": result["triggered"],
                            "width_ratio": result["width_ratio"],
                            "open_ratio": result["open_ratio"],
                            "face_landmarks": result["face_landmarks"],
                            "timestamp": time.time(),
                        }
                    )

            if active == "object":
                result = object_recognition.process(frame_rgb)
                await manager.broadcast(
                    {
                        "modality": "object",
                        "markers": result["markers"],
                        "timestamp": time.time(),
                    }
                )

            await asyncio.sleep(0)  # yield to event loop
    finally:
        cap.release()
        proxemic.close()
        gesture.close()
        object_recognition.close()


async def config_watch_loop():
    while True:
        await asyncio.sleep(1)
        if config_manager.has_changed():
            config_manager.reload()
            await manager.broadcast({"type": "config_update", "config": config_manager.config})


app.mount("/control", StaticFiles(directory=CONTROL_PANEL_DIR, html=True), name="control")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
