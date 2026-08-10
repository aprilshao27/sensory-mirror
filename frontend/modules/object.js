// Phase 2 — Object recognition rendering (not yet implemented).
// Will render per-marker visuals (see config.objects[marker_id] in
// content_config.json) in response to {"modality": "object", "marker_id",
// "timestamp"} messages from backend/modules/object_recognition.py.
// Not yet wired into core.js's message routing or draw loop.

const ObjectModule = {
  handleMessage() {
    throw new Error("Phase 2 — not yet implemented");
  },
  draw() {
    throw new Error("Phase 2 — not yet implemented");
  },
};
