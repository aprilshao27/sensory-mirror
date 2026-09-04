"""Phase 3 — Intentional facial gesture module.

MediaPipe Face Mesh detects two fixed, exaggerated expression patterns —
held smile and mouth-open/surprised — from landmark-distance ratios. This is
pattern matching against a small fixed vocabulary, not emotion inference:
each gesture must be held continuously for GESTURE_HOLD_SECONDS before it
"triggers". hold_progress is published every frame while a candidate gesture
is active so the frontend can build its visual/audio payoff continuously
through the hold instead of snapping it in at the end.

Publishes continuously whenever a face is tracked (mirroring proxemic's
always-on state stream), including width_ratio/open_ratio — the raw values
being compared against the thresholds below — so a live session can watch
how close an expression gets without editing code and guessing blind.
"""

import time

import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections as _fmc

# Tune against real camera/lighting during Phase 3 validation, same as
# proxemic's thresholds. Starting points below are set from rough
# anthropometric ratios (mouth width and inner-lip gap relative to
# interocular distance), not a live camera — watch width_ratio/open_ratio in
# an actual session (browser devtools' WS frames, or the console log this
# module can be pointed at) and adjust these if they're still off.
GESTURE_HOLD_SECONDS = 1.5

# MediaPipe Face Mesh landmark indices (468-point topology).
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14


def _unique_indices(connections) -> list[int]:
    """Flattens a FACEMESH_* connection set (pairs of landmark indices
    forming its outline edges) into the distinct point indices it touches.
    Order doesn't matter — the frontend builds a convex hull from these, not
    an edge-ordered path — so this doesn't need to trace the topology.
    """
    return sorted({idx for edge in connections for idx in edge})


# Point sets for the frontend's per-feature soft underlay (eyes/nose/mouth
# glow, styled after proxemic's body-silhouette underlay). Reuses the same
# 468-point face mesh already computed for gesture classification above, so
# this is just a slice of the existing landmarks, not extra detection work.
FACE_FEATURE_INDICES = {
    "left_eye": _unique_indices(_fmc.FACEMESH_LEFT_EYE),
    "right_eye": _unique_indices(_fmc.FACEMESH_RIGHT_EYE),
    "nose": _unique_indices(_fmc.FACEMESH_NOSE),
    "mouth": _unique_indices(_fmc.FACEMESH_LIPS),
}

# Ratios are normalised against interocular distance so they hold steady
# across camera distance, and measured in true pixel space (landmark.x/.y
# scaled by actual frame width/height before computing distance) rather than
# raw normalised coordinates directly — mixing normalised-x and normalised-y
# without that scaling silently distorts the ratio on any non-square camera
# frame, since each axis is normalised against a different dimension.
#
# A resting mouth is roughly half the interocular width; a wide, deliberate
# grin stretches that toward ~0.8-0.9. Thresholds sit between "casual
# expression" and "maximum exaggeration" on purpose — "intentional input",
# per the design brief, not naturalistic smiling or talking.
SMILE_WIDTH_RATIO_THRESHOLD = 0.62  # mouth-corner width / interocular distance

# The anthropometric estimate for this one (0.30) turned out well above what
# a real face/camera setup can actually produce — live testing topped out
# around 0.2 for a maximally wide-open mouth. Set with margin below that
# measured ceiling instead, so it's reliably reachable without straining.
MOUTH_OPEN_RATIO_THRESHOLD = 0.15  # inner-lip gap / interocular distance

# Keeps the two gestures mutually exclusive at the boundary — a wide grin
# with a slightly parted mouth shouldn't leak into mouth_open, and a
# surprised "O" mouth shouldn't leak into held_smile.
SMILE_MAX_OPEN_RATIO = 0.22
MOUTH_OPEN_MAX_WIDTH_RATIO = 0.75

GESTURE_HELD_SMILE = "held_smile"
GESTURE_MOUTH_OPEN = "mouth_open"

# A single frame with no detected face is usually a brief tracking hiccup
# (motion blur, a fast head turn) rather than the person looking away — same
# rationale as proxemic's missed-frame grace period. Held within this many
# consecutive missed frames (~0.15s at 30fps), an in-progress hold keeps
# counting instead of resetting to zero.
MISSED_FRAMES_GRACE = 5


class GestureDetector:
    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._missed_frames = 0
        self._active_gesture = None
        self._hold_start = None
        self._triggered = False  # fires once per hold; clears when the gesture releases

    def process(self, frame_rgb) -> dict | None:
        """Returns {"gesture_type", "hold_progress", "triggered",
        "width_ratio", "open_ratio", "face_landmarks"} for the current frame,
        or None if the frame should be skipped (a transient tracking dropout
        still within the grace period). gesture_type/ratios/face_landmarks
        are None when no face is tracked; gesture_type is otherwise None
        whenever the tracked face isn't currently forming either gesture.
        """
        h, w = frame_rgb.shape[:2]
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            self._missed_frames += 1
            if self._missed_frames < MISSED_FRAMES_GRACE:
                return None
            return self.reset()

        self._missed_frames = 0
        landmarks = results.multi_face_landmarks[0].landmark
        gesture, width_ratio, open_ratio = self._classify(landmarks, w, h)

        if gesture != self._active_gesture:
            self._active_gesture = gesture
            self._hold_start = time.monotonic() if gesture else None
            self._triggered = False

        hold_progress = 0.0
        just_triggered = False
        if gesture is not None:
            elapsed = time.monotonic() - self._hold_start
            hold_progress = min(elapsed / GESTURE_HOLD_SECONDS, 1.0)
            just_triggered = hold_progress >= 1.0 and not self._triggered
            if just_triggered:
                self._triggered = True

        face_landmarks = {
            feature: [{"x": landmarks[i].x, "y": landmarks[i].y} for i in indices]
            for feature, indices in FACE_FEATURE_INDICES.items()
        }

        return {
            "gesture_type": gesture,
            "hold_progress": hold_progress,
            "triggered": just_triggered,
            "width_ratio": round(width_ratio, 3),
            "open_ratio": round(open_ratio, 3),
            "face_landmarks": face_landmarks,
        }

    def reset(self) -> dict:
        """Clears any in-progress hold and reports a confirmed no-face
        frame. Also called on the rising edge of the module being
        re-enabled (see main.py) — otherwise a stale _hold_start from
        before it was last active makes the very next frame read as an
        already-complete hold and fire a false trigger.
        """
        self._missed_frames = 0
        self._active_gesture = None
        self._hold_start = None
        self._triggered = False
        return {
            "gesture_type": None,
            "hold_progress": 0.0,
            "triggered": False,
            "width_ratio": None,
            "open_ratio": None,
            "face_landmarks": None,
        }

    def _classify(self, landmarks, frame_w: int, frame_h: int) -> tuple[str | None, float, float]:
        def dist(a, b):
            dx = (a.x - b.x) * frame_w
            dy = (a.y - b.y) * frame_h
            return (dx * dx + dy * dy) ** 0.5

        interocular = dist(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER])
        if interocular < 1e-6:
            return None, 0.0, 0.0

        width_ratio = dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]) / interocular
        open_ratio = dist(landmarks[MOUTH_INNER_TOP], landmarks[MOUTH_INNER_BOTTOM]) / interocular

        if width_ratio > SMILE_WIDTH_RATIO_THRESHOLD and open_ratio < SMILE_MAX_OPEN_RATIO:
            return GESTURE_HELD_SMILE, width_ratio, open_ratio
        if open_ratio > MOUTH_OPEN_RATIO_THRESHOLD and width_ratio < MOUTH_OPEN_MAX_WIDTH_RATIO:
            return GESTURE_MOUTH_OPEN, width_ratio, open_ratio
        return None, width_ratio, open_ratio

    def close(self):
        self._mesh.close()
