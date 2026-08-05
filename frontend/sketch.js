// Phase 1 — Proxemic interaction: placeholder visuals.
// Renders a smooth interpolation between three continuous states published
// by the backend over WebSocket. No hard switching — every state's visual
// carries a "weight" that eases toward 1 (active) or 0 (inactive) each frame,
// so transitions read as continuous rather than as a state machine.

let ws;
let config = null;
let targetState = "retreat";
let stateWeights = { close_still: 0, active_movement: 0, retreat: 1 };
const SMOOTHING = 0.05;

let particles = [];

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
      targetState = msg.state;
    }
    // object/gesture messages will be handled once Phase 2/3 modules exist.
  };

  ws.onclose = () => setTimeout(connectSocket, 1000);
}

function draw() {
  background(0);

  for (const state of Object.keys(stateWeights)) {
    const target = state === targetState ? 1 : 0;
    stateWeights[state] += (target - stateWeights[state]) * SMOOTHING;
  }

  if (!config) return;

  drawRetreatFade();
  drawCloseStillPulse();
  drawActiveMovementTrail();
}

function drawRetreatFade() {
  const w = stateWeights.retreat;
  const c = colorFromConfig("retreat");
  noStroke();
  fill(red(c), green(c), blue(c), 120 * w);
  rect(0, 0, width, height);
}

function drawCloseStillPulse() {
  const w = stateWeights.close_still;
  if (w < 0.01) return;
  const c = colorFromConfig("close_still");
  const pulse = 1 + 0.08 * sin(frameCount * 0.05);
  const r = min(width, height) * 0.25 * w * pulse;
  noStroke();
  fill(red(c), green(c), blue(c), 200 * w);
  circle(width / 2, height / 2, r * 2);
}

function drawActiveMovementTrail() {
  const w = stateWeights.active_movement;
  const c = colorFromConfig("active_movement");

  if (w > 0.05 && random() < w) {
    particles.push({
      x: width / 2 + random(-40, 40),
      y: height / 2 + random(-40, 40),
      vx: random(-2, 2),
      vy: random(-2, 2),
      life: 60,
    });
  }

  noStroke();
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.life -= 1;
    const alpha = (p.life / 60) * 220 * w;
    fill(red(c), green(c), blue(c), alpha);
    circle(p.x, p.y, 10);
    if (p.life <= 0) particles.splice(i, 1);
  }
}

function colorFromConfig(state) {
  const hex = config.proxemic[state]?.color ?? "#ffffff";
  return color(hex);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
