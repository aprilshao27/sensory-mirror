"""Phase 2 — Object recognition module (not yet implemented).

Will run cv2.aruco.ArucoDetector (DICT_4X4_50) per frame and publish a
discrete event on detection: {"modality": "object", "marker_id": X,
"timestamp": ...}. Must stay fully decoupled from ProxemicDetector so each
modality can be toggled and evaluated independently.
"""


class ObjectRecognitionDetector:
    def __init__(self):
        raise NotImplementedError("Phase 2 — build after proxemic is stable end-to-end.")

    def process(self, frame_bgr) -> dict | None:
        raise NotImplementedError
