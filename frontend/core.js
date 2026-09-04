// Shared p5 lifecycle, WebSocket connection, and config state. Modality-
// specific rendering lives in modules/*.js (loaded before this file); this
// file only wires the canvas and incoming messages to those modules.
//
// keyPressed() forwards key events to ObjectModule's mouse-driven dev
// simulator (see modules/object.js) — delete it once that block is removed.

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
    } else if (msg.modality === "object") {
      ObjectModule.handleMessage(msg);
    }
  };

  ws.onclose = () => setTimeout(connectSocket, 1000);
}

function draw() {
  background(0);
  Proxemic.update();
  Gesture.update();
  ObjectModule.update(config?.objects);
  if (!config) return;
  Proxemic.draw(config.proxemic);
  Gesture.draw(config.gestures, config.active_modality);
  ObjectModule.draw(config.objects, config.ambient_wave);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function keyPressed() {
  ObjectModule.handleDevKey(key);
}
