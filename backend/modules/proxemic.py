"""Phase 1 — Proxemic/body interaction module.

Classifies the person in frame into one of three continuous states using
MediaPipe Pose: bounding-box size (distance-to-mirror) and frame-to-frame
landmark movement magnitude. Publishes every frame, not as discrete events,
so this module tracks a relationship.
"""

import mediapipe as mp

# Tune against your actual camera distance/lighting during Phase 1 validation.
CLOSE_BBOX_THRESHOLD = 0.55  # normalised bbox height/width fraction of frame
MOVEMENT_THRESHOLD = 0.015  # mean landmark delta per frame

STATE_CLOSE_STILL = "close_still"
STATE_ACTIVE_MOVEMENT = "active_movement"
STATE_RETREAT = "retreat"


class ProxemicDetector:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_landmarks = None

    def process(self, frame_rgb) -> dict | None:
        """Returns {"state", "bbox_size", "movement"} for the current frame,
        or None if no person is detected.
        """
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            self._prev_landmarks = None
            return None

        landmarks = results.pose_landmarks.landmark
        bbox_size = self._bbox_size(landmarks)
        movement = self._movement_magnitude(landmarks)
        self._prev_landmarks = landmarks

        state = self._classify(bbox_size, movement)
        return {"state": state, "bbox_size": bbox_size, "movement": movement}

    def _bbox_size(self, landmarks) -> float:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _movement_magnitude(self, landmarks) -> float:
        if self._prev_landmarks is None:
            return 0.0
        deltas = [
            ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
            for a, b in zip(landmarks, self._prev_landmarks)
        ]
        return sum(deltas) / len(deltas)

    def _classify(self, bbox_size: float, movement: float) -> str:
        if movement > MOVEMENT_THRESHOLD:
            return STATE_ACTIVE_MOVEMENT
        if bbox_size > CLOSE_BBOX_THRESHOLD:
            return STATE_CLOSE_STILL
        return STATE_RETREAT

    def close(self):
        self._pose.close()
