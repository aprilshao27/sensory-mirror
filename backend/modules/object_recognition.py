"""Phase 2 — Object recognition module.

Runs cv2.aruco.ArucoDetector (DICT_4X4_50, matching
scripts/generate_aruco_markers.py) per frame and publishes every detected
marker's position continuously, not as a discrete "seen once" event —
marker_5's segmented_light_wave visual (see frontend/modules/object.js)
builds its trailing chain from a recorded trajectory, so it needs a
continuous stream the same way proxemic streams landmarks every frame
rather than firing on state changes.

Detection is dumb and complete: every marker id present in frame is
reported regardless of what content_config.json currently maps it to. This
keeps the module decoupled from the frontend's per-marker visual
assignment, same rationale as staying decoupled from ProxemicDetector.
"""

import math

import cv2

DICTIONARY = cv2.aruco.DICT_4X4_50


class ObjectRecognitionDetector:
    def __init__(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corner refinement noticeably improves bit-sampling accuracy on
        # small/noisy markers (e.g. a marker held up on a phone screen, where
        # moire between the camera sensor and the screen's own pixel grid
        # corrupts a few cells) at a modest per-frame cost.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # DICT_4X4_50's default error-correction budget (0.6, i.e. up to 60% of
        # the max correctable bit errors) is lenient enough to "recover" a noisy
        # read into a different, wrong-but-valid marker id rather than rejecting
        # it — worse here than a brief missed frame, since a wrong id just
        # renders nothing (see frontend/modules/object.js, which ignores ids not
        # in content_config.json) while masking that a real marker was in frame.
        params.errorCorrectionRate = 0.3
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def process(self, frame_rgb) -> dict:
        """Returns {"markers": [{"marker_id", "x", "y", "angle"}, ...]} for
        every marker detected in the current frame. An empty list (not
        None) when nothing is detected — the frontend's own per-marker
        presence timeout governs fade-out, not message cadence, so there's
        no need to distinguish "no markers this frame" from "skip this
        frame" the way proxemic/gesture's grace periods do.

        x/y are the marker centre, normalised [0,1] against frame width/
        height (same convention as gesture.py's face_landmarks). angle is
        the heading in radians of the marker's top edge (top-left ->
        top-right corner), for visuals that want direction of travel later;
        nothing currently consumes it.
        """
        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        markers = []
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids):
                pts = marker_corners[0]  # 4x2: top-left, top-right, bottom-right, bottom-left
                cx, cy = pts.mean(axis=0)
                heading = pts[1] - pts[0]
                angle = math.atan2(float(heading[1]), float(heading[0]))
                markers.append(
                    {
                        "marker_id": int(marker_id[0]),
                        "x": round(float(cx) / w, 4),
                        "y": round(float(cy) / h, 4),
                        "angle": round(angle, 4),
                    }
                )

        return {"markers": markers}

    def close(self):
        pass
