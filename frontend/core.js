// Shared p5 lifecycle, WebSocket connection, and config state. Modality-
// specific rendering lives in modules/*.js (loaded before this file); this
// file only wires the canvas and incoming messages to those modules.
//
// Object recognition isn't routed below — backend/modules/object_recognition.py
// is still an unimplemented stub. Add routing here once it lands, mirroring
// backend/main.py's camera_loop (which likewise skips it for now).

let ws;
let config = null;

function setup() {
  createCanvas(windowWidth, windowHeight);
  colorMode(RGB);
  connectSocket();
}

function connectSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "config_update") {
      config = msg.config;
    } else if (msg.modality === "proxemic") {
      Proxemic.handleMessage(msg);
    } else if (msg.modality === "gesture") {
      Gesture.handleMessage(msg);
    }
  };

  ws.onclose = () => setTimeout(connectSocket, 1000);
}

function draw() {
  background(0);
  Proxemic.update();
  Gesture.update();
  if (!config) return;
  Proxemic.draw(config.proxemic);
  Gesture.draw(config.gestures);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
