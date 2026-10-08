"""Phase 2 — Object recognition module.

Runs a YOLO11s model, fine-tuned on real photos of the 5 sensory items (see
backend/dataset/ and scripts/prepare_dataset.py / train_object_model.py),
per frame and publishes every detected item's position continuously, not as
a discrete "seen once" event — marker_4's segmented_light_wave visual (see
frontend/modules/object.js) builds its trailing chain from a recorded
trajectory, so it needs a continuous stream the same way proxemic streams
landmarks every frame rather than firing on state changes.

Detection is dumb and complete: every class detected in frame is reported
regardless of what content_config.json currently maps it to. This keeps the
module decoupled from the frontend's per-item visual assignment, same
rationale as staying decoupled from ProxemicDetector.

Class ids come straight from the model (0-4, fixed by dataset/data.yaml's
class order) and are published as "marker_id" unchanged — that id already
matches content_config.json's marker_0..marker_4 keys, so no translation
layer is needed between the two. There's no rotation angle (a bounding box
detector doesn't produce one, unlike ArUco's corner geometry this replaced)
— every consumer already treats "angle" as optional and degrades to
translation-only reactivity without it.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "object_recognition.pt"
# Deliberately higher than the ~0.001 confidence sweep mAP is measured across
# — this is a live per-frame feed, not an offline metric, so a threshold that
# only lets through fairly confident detections matters more for avoiding
# flickery, low-confidence boxes than for squeezing out extra recall.
CONFIDENCE_THRESHOLD = 0.5


class ObjectRecognitionDetector:
    def __init__(self):
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"{WEIGHTS_PATH} not found — run scripts/prepare_dataset.py then "
                "scripts/train_object_model.py to produce it (see README.md)"
            )
        self._model = YOLO(str(WEIGHTS_PATH))

    def process(self, frame_rgb) -> dict:
        """Returns {"markers": [{"marker_id", "x", "y"}, ...]} for every item
        detected in the current frame. An empty list (not None) when nothing
        is detected — the frontend's own per-marker presence timeout governs
        fade-out, not message cadence, so there's no need to distinguish "no
        items this frame" from "skip this frame".

        x/y are the detected box's centre, normalised [0,1] against frame
        width/height (same convention as gesture.py's face_landmarks).
        """
        h, w = frame_rgb.shape[:2]
        # ultralytics expects the same BGR convention cv2.imread produces —
        # which is what both the training images and every other cv2 call in
        # this codebase use — so undo main.py's earlier RGB conversion here
        # rather than let a channel-order mismatch between training and
        # inference quietly hurt accuracy.
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # Keep only the highest-confidence box per class. Ultralytics' own
        # NMS is class-aware and IoU-threshold based, which occasionally
        # leaves two overlapping boxes for the same physical item standing
        # (e.g. a tight box around the body and a looser one including more
        # of it, at just under the IoU cutoff) — harmless for a generic
        # object detector, but here every class IS one specific physical
        # item, so two simultaneous positions for the same marker_id would
        # just jitter that item's tracked position between them. "One
        # detection per class" encodes that physical reality directly,
        # rather than trying to tune NMS thresholds to approximate it.
        best_per_class = {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                best_per_class[cls_id] = (conf, box)

        markers = []
        for cls_id, (_, box) in best_per_class.items():
            cx, cy = box.xywh[0][:2].tolist()
            markers.append(
                {
                    "marker_id": cls_id,
                    "x": round(cx / w, 4),
                    "y": round(cy / h, 4),
                }
            )

        return {"markers": markers}

    def close(self):
        pass
