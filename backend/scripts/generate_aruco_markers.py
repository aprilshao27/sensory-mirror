"""Generate printable ArUco marker images for the object-recognition modality.

Dictionary must match ObjectRecognitionDetector (backend/modules/object_recognition.py),
and output filenames must match the "marker_<id>" keys in
backend/config/content_config.json.

Run from the backend/ directory:
    python scripts/generate_aruco_markers.py --ids 7 12 --size 800
"""

import argparse
import os

import cv2

DICTIONARY = cv2.aruco.DICT_4X4_50
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "markers")


def generate(ids, size_px, quiet_zone_px, out_dir):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    os.makedirs(out_dir, exist_ok=True)

    for marker_id in ids:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
        padded = cv2.copyMakeBorder(
            marker_img,
            quiet_zone_px, quiet_zone_px, quiet_zone_px, quiet_zone_px,
            cv2.BORDER_CONSTANT, value=255,
        )
        out_path = os.path.join(out_dir, f"marker_{marker_id}.png")
        cv2.imwrite(out_path, padded)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, nargs="+", required=True, help="marker IDs (0-49 for DICT_4X4_50)")
    parser.add_argument("--size", type=int, default=800, help="marker size in pixels, excluding quiet zone")
    parser.add_argument("--quiet-zone", type=int, default=100, help="white border padding in pixels")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    args = parser.parse_args()

    generate(args.ids, args.size, args.quiet_zone, args.out)
