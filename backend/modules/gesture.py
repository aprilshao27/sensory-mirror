"""Phase 3 — Intentional facial gesture module (not yet implemented).

Will use MediaPipe Face Mesh to recognise exactly two pre-defined expression
patterns (held smile, mouth-open/surprised), each requiring a ~1.5-2s hold
before triggering. This module recognises presence/absence of fixed patterns
only — no emotion classification or inference from expression.
Publishes {"modality": "gesture", "gesture_type", "hold_progress", "timestamp"}.
"""


class GestureDetector:
    def __init__(self):
        raise NotImplementedError("Phase 3 — build last; cut first if time runs short.")

    def process(self, frame_rgb) -> dict | None:
        raise NotImplementedError
