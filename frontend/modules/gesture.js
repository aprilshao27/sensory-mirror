// Phase 3 — Intentional facial gesture rendering (not yet implemented).
// Will render per-gesture visuals (see config.gestures in
// content_config.json) in response to {"modality": "gesture", "gesture_type",
// "hold_progress", "timestamp"} messages from backend/modules/gesture.py.
// Not yet wired into core.js's message routing or draw loop.

const Gesture = {
  handleMessage() {
    throw new Error("Phase 3 — not yet implemented");
  },
  draw() {
    throw new Error("Phase 3 — not yet implemented");
  },
};
