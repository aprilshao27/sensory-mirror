"""Phase 1 — Proxemic/body interaction module.

Classifies the person in frame into one of three continuous states using
MediaPipe Pose: bounding-box size (distance-to-mirror) and frame-to-frame
landmark movement magnitude. Publishes every frame, not as discrete events,
so this module tracks a relationship.

Segmentation (enable_segmentation=True) additionally produces a soft person
mask, downsampled and sent to the frontend so it can render a translucent
silhouette underlay. Mask quality drops as the person moves away from the
camera, so the mask is blurred more heavily (and at lower resolution) the
farther away they are - this degrades the silhouette toward "loose and soft"
"""

import base64
import cv2
import mediapipe as mp
import numpy as np

# Tune against your actual camera distance/lighting during Phase 1 validation.
# Hysteresis band, not a single cutoff (same rationale as the movement
# thresholds below): entering close_still requires clearing the higher bar;
# once close, it takes dropping below the lower bar to read as "retreat".
# Narrowing the gap between the two makes retreat trigger sooner (closer to
# the mirror); widening it delays retreat until the person is farther back.
CLOSE_BBOX_THRESHOLD = 0.7  # normalised bbox height/width fraction of frame
RETREAT_BBOX_THRESHOLD = 0.5  # below this, the person reads as "far away"

# Hysteresis band, not a single cutoff: landmark jitter alone (no real motion)
# produces small nonzero deltas every frame, so a single threshold flickers
# in and out of active_movement while the person stands still. Entering
# requires clearing the higher bar; once in, it takes dropping below the
# lower bar to leave.
MOVEMENT_ENTER_THRESHOLD = 0.015
MOVEMENT_EXIT_THRESHOLD = 0.008
# Light touch on purpose: with hysteresis already guarding the enter/exit
# boundary and proximity normalisation handling the close-range case, heavy
# smoothing here just damps real movement (needing several frames to build
# up to the same effect) rather than filtering noise. This is only enough
# to knock down an isolated single-frame jitter spike.
MOVEMENT_SMOOTHING = 0.7  # EMA alpha applied to movement before classification

# Landmark movement is measured in frame-normalised coordinates, so the same
# physical sway covers a larger fraction of the frame the closer someone is
# to the camera (perspective, not more actual motion). Movement is only ever
# damped down for bbox sizes past this point, not amplified for anyone
# farther back. Deliberately a fixed value, not tied to CLOSE_BBOX_THRESHOLD
# — that threshold is a visual/state tuning knob and can be set far higher
# than the distance where perspective actually starts amplifying jitter, so
# linking them let close_still tuning silently disable this damping.
NORMALIZATION_REFERENCE_BBOX = 0.35

# MediaPipe Pose indices for nose, shoulders, hips — landmarks that stay in
# frame at normal interaction distance. Extremities (wrists, feet) are
# excluded: MediaPipe still emits a guessed position for them even when
# they're outside the visible frame or occluded, and those guesses are
# unreliable enough to blow bbox_size past the expected [0, 1] range.
BBOX_LANDMARK_INDICES = (0, 11, 12, 23, 24)

STATE_CLOSE_STILL = "close_still"
STATE_ACTIVE_MOVEMENT = "active_movement"
STATE_RETREAT = "retreat"

# A single frame with no detected pose is usually MediaPipe dropping tracking
# for a moment (motion blur from fast movement, a brief occlusion) rather
# than the person actually leaving. Require this many consecutive missed
# frames (~0.25s at 30fps) before reporting retreat, so a normal mid-movement
# dropout doesn't flicker the starfield in over whatever state is active.
MISSED_FRAMES_RETREAT_THRESHOLD = 8

# Mask resolution: higher keeps the silhouette edge defined once it's scaled
# up to fill the canvas; still small enough to keep the WS payload light.
MASK_MAX_DIM = 128


class ProxemicDetector:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_landmarks = None
        self._smoothed_movement = 0.0
        self._is_moving = False
        self._is_retreating = False
        self._missed_frames = 0

    def process(self, frame_rgb) -> dict | None:
        """Returns {"state", "bbox_size", "movement", "landmarks", "mask"}
        for the current frame, or None if the frame should be skipped (either
        a transient tracking dropout within the grace period, or a confirmed
        absence already reported and unchanged). "mask" is a small,
        degradation-aware grayscale segmentation mask package for frontend
        rendering; "landmarks" are normalised (x, y) pose points.
        """
        results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            self._missed_frames += 1
            self._prev_landmarks = None
            self._smoothed_movement = 0.0
            self._is_moving = False
            if self._missed_frames < MISSED_FRAMES_RETREAT_THRESHOLD:
                return None
            self._is_retreating = True
            return {
                "state": STATE_RETREAT,
                "bbox_size": 0.0,
                "movement": 0.0,
                "landmarks": [],
                "mask": None,
            }

        self._missed_frames = 0
        landmarks = results.pose_landmarks.landmark
        bbox_size = self._bbox_size(landmarks)
        raw_movement = self._movement_magnitude(landmarks)
        proximity_scale = max(bbox_size / NORMALIZATION_REFERENCE_BBOX, 1.0)
        raw_movement /= proximity_scale
        self._smoothed_movement += (raw_movement - self._smoothed_movement) * MOVEMENT_SMOOTHING
        movement = self._smoothed_movement
        self._prev_landmarks = landmarks

        state = self._classify(bbox_size, movement)
        mask = None
        if results.segmentation_mask is not None:
            mask = self._package_mask(results.segmentation_mask, bbox_size)

        return {
            "state": state,
            "bbox_size": bbox_size,
            "movement": movement,
            "landmarks": [{"x": lm.x, "y": lm.y} for lm in landmarks],
            "mask": mask,
        }

    def _bbox_size(self, landmarks) -> float:
        points = [landmarks[i] for i in BBOX_LANDMARK_INDICES]
        xs = [lm.x for lm in points]
        ys = [lm.y for lm in points]
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
        if movement > MOVEMENT_ENTER_THRESHOLD:
            self._is_moving = True
        elif movement < MOVEMENT_EXIT_THRESHOLD:
            self._is_moving = False

        if bbox_size > CLOSE_BBOX_THRESHOLD:
            self._is_retreating = False
        elif bbox_size < RETREAT_BBOX_THRESHOLD:
            self._is_retreating = True

        if self._is_moving:
            return STATE_ACTIVE_MOVEMENT
        if self._is_retreating:
            return STATE_RETREAT
        return STATE_CLOSE_STILL

    def _package_mask(self, mask, bbox_size: float) -> dict:
        h, w = mask.shape
        scale = MASK_MAX_DIM / max(h, w)
        out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))

        # closeness: 1.0 near the mirror, falling toward 0 as the person
        # shrinks past the "close" threshold and approaches "retreat". Drives
        # both the mask blur below (segmentation gets noisier at distance)
        # and the frontend's continuous shrink/fade of the silhouette.
        closeness = np.clip(
            (bbox_size - RETREAT_BBOX_THRESHOLD)
            / (CLOSE_BBOX_THRESHOLD - RETREAT_BBOX_THRESHOLD),
            0.0,
            1.0,
        )

        small = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # More blur (and, implicitly, less detail from the low resolution)
        # the farther away the person is — this is the "graceful degrade to
        # loose/blurred" behaviour rather than a crisp cutout at any distance.
        kernel = int(3 + round((1.0 - closeness) * 10)) | 1  # odd, 3..13
        blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)

        mask_u8 = np.clip(blurred * 255, 0, 255).astype(np.uint8)
        return {
            "width": out_w,
            "height": out_h,
            "closeness": float(closeness),
            "data": base64.b64encode(mask_u8.tobytes()).decode("ascii"),
        }

    def close(self):
        self._pose.close()
