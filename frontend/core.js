// Shared p5 lifecycle, WebSocket connection, and config state. Modality-
// specific rendering lives in modules/*.js (loaded before this file); this
// file only wires the canvas and incoming messages to those modules.
//
// Only proxemic is routed below — object/gesture messages don't exist yet
// since backend/modules/object_recognition.py and gesture.py are still
// unimplemented stubs. Add routing here once each lands, mirroring
// backend/main.py's lifespan (which likewise only runs proxemic_loop()).

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
    }
  };

  ws.onclose = () => setTimeout(connectSocket, 1000);
}

function draw() {
  background(0);
  Proxemic.update();
  if (!config) return;
  Proxemic.draw(config.proxemic);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
