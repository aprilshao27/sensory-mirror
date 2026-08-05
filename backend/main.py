import asyncio
import json
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from config.config_manager import ConfigManager
from modules.proxemic import ProxemicDetector

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
CONTROL_PANEL_DIR = Path(__file__).parent.parent / "control_panel"

app = FastAPI(title="Sensory Mirror Backend")
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


@app.post("/api/config")
async def update_config(new_config: dict, note: str = ""):
    """Used by the Phase 4 control panel to edit content mappings live."""
    config_manager.save(new_config, note=note)
    await manager.broadcast({"type": "config_update", "config": config_manager.config})
    return {"status": "ok", "config": config_manager.config}


async def proxemic_loop():
    detector = ProxemicDetector()
    cap = cv2.VideoCapture(0)
    try:
        while True:
            if not config_manager.config["modalities"]["proxemic"]:
                await asyncio.sleep(0.2)
                continue

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(frame_rgb)
            if result is not None:
                await manager.broadcast(
                    {
                        "modality": "proxemic",
                        "state": result["state"],
                        "timestamp": time.time(),
                    }
                )
            await asyncio.sleep(0)  # yield to event loop
    finally:
        cap.release()
        detector.close()


async def config_watch_loop():
    while True:
        await asyncio.sleep(1)
        if config_manager.has_changed():
            config_manager.reload()
            await manager.broadcast({"type": "config_update", "config": config_manager.config})


@app.on_event("startup")
async def startup():
    asyncio.create_task(proxemic_loop())
    asyncio.create_task(config_watch_loop())


app.mount("/control", StaticFiles(directory=CONTROL_PANEL_DIR, html=True), name="control")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
