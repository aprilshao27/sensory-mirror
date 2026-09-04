"""Turns backend/dataset/raw/<class>/*.jpg (one folder per item) into a
YOLO-ready detection dataset.

Every raw photo is a single item, so this needs exactly one bounding box per
image. Preferred source: hand-drawn boxes from LabelImg (pip-installed into
this venv — run `labelImg.exe backend/dataset/raw/<class>` from backend/,
YOLO save format, save into the same folder as the images — a matching
classes.txt already sits in each raw/<class>/ dir so class ids stay
consistent across folders). Where no manual `<image>.txt` exists yet, this
falls back to auto-generating one via GrabCut seeded from a center-inset
rectangle — approximate, not hand-verified, meant to be replaced by manual
labels over time rather than trusted as-is. Re-run anytime; already-manually-
labeled images are always picked up automatically.

This writes an annotated preview copy of EVERY image (see
_bbox_preview/<class>/, colour-coded by source — see PREVIEW_COLORS) so every
box, manual or auto, can be checked by eye before spending GPU time training.
This script only prepares the dataset — it deliberately does not chain into
training, so review can happen first.

Output layout under backend/dataset/ (raw/ untouched, everything else here
regenerated on every run):

    trainval/train/images/*.jpg  trainval/train/labels/*.txt
    trainval/val/images/*.jpg    trainval/val/labels/*.txt
    test/images/*.jpg            test/labels/*.txt
    data.yaml                    ultralytics dataset config
    split_manifest.json          per-image split/bbox/source record
    _bbox_preview/<class>/*.jpg  every image, box drawn on, for review

Split: test = 10% (held out, never used for training/tuning), remaining 90%
split 80/20 train/val — stratified per class AND grouped by capture burst.
Each class's raw photos come from ~5-6 distinct shooting sessions (phone/
webcam timestamps in the filenames), with consecutive frames within a burst
seconds apart and near-duplicates of each other. A plain per-image random
split would let near-twin frames land on both sides of a split, leaking
train-time information into "held out" val/test images and making their
metrics read better than true generalization. So splitting groups whole
bursts (by filename timestamp, gapped >5min apart) into one split each,
never dividing a burst — see group_into_sessions/assign_sessions_to_splits.

Run from backend/: .venv/Scripts/python.exe scripts/prepare_dataset.py
"""

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

DATASET_DIR = Path(__file__).parent.parent / "dataset"
RAW_DIR = DATASET_DIR / "raw"

# Folder name (under raw/) == class slug used in data.yaml / labels / preview
# dirs. List order fixes the class id, which content_config.json's
# marker_0..marker_4 keys and the runtime detector both assume matches this.
CLASSES = [
    "pink_pop_it",
    "blue_green_pop_it",
    "blue_red_pop_it",
    "noomi_squeezibo_cat",
    "caterpillar",
]

SEED = 0
TEST_FRACTION = 0.10
VAL_FRACTION_OF_TRAINVAL = 0.20
MAX_SIDE = 1280  # long-side resize cap: well above the 640 training imgsz
GRABCUT_SIDE = 480  # GrabCut runs on a further-downscaled copy — segmentation
# doesn't need full resolution, and GrabCut's cost scales with pixel count;
# 1280->480 is roughly a 7x pixel reduction, the difference between ~11s and
# ~1-2s per image at 1,686 images. The resulting box is scaled back up to
# MAX_SIDE-image coordinates before being saved.
GRABCUT_ITERATIONS = 3
CENTER_MARGIN = 0.06  # GrabCut seed rect inset from each edge, as a fraction
BOX_PADDING = 0.03  # padding added around the segmented box, as a fraction
MIN_BOX_AREA_FRACTION = 0.02  # below this, treat GrabCut as having failed
JPEG_QUALITY = 92

# BGR, since previews are written with cv2.imwrite straight from cv2 buffers.
PREVIEW_COLORS = {
    "manual": (0, 255, 0),  # green — a hand-drawn LabelImg box, trust this
    "grabcut": (255, 160, 0),  # blue — GrabCut found a plausible foreground box
    "grabcut_fallback": (0, 140, 255),  # orange — GrabCut failed, used the seed rect
}


def resize_long_side(img, max_side):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def grabcut_box(img):
    """Returns (x, y, w, h) in FULL-RES `img` pixel coordinates, plus whether
    GrabCut's own result was used (False means the center-seed rectangle was
    returned as a fallback). GrabCut itself runs on a downscaled copy (see
    GRABCUT_SIDE) for speed; the box is scaled back up before returning.
    """
    h, w = img.shape[:2]
    small = resize_long_side(img, GRABCUT_SIDE)
    sh, sw = small.shape[:2]
    scale_x, scale_y = w / sw, h / sh

    mx, my = round(sw * CENTER_MARGIN), round(sh * CENTER_MARGIN)
    seed_rect = (mx, my, sw - 2 * mx, sh - 2 * my)

    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, seed_rect, bgd_model, fgd_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        box, used = seed_rect, False
    else:
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            box, used = seed_rect, False
        else:
            largest = max(contours, key=cv2.contourArea)
            candidate = cv2.boundingRect(largest)  # x, y, w, h
            if candidate[2] * candidate[3] < MIN_BOX_AREA_FRACTION * sw * sh:
                box, used = seed_rect, False
            else:
                box, used = candidate, True

    bx, by, bw, bh = [round(v * s) for v, s in zip(box, (scale_x, scale_y, scale_x, scale_y))]
    pad_x, pad_y = round(bw * BOX_PADDING), round(bh * BOX_PADDING)
    x0 = max(0, bx - pad_x)
    y0 = max(0, by - pad_y)
    x1 = min(w, bx + bw + pad_x)
    y1 = min(h, by + bh + pad_y)
    return (x0, y0, x1 - x0, y1 - y0), used


def read_manual_box(label_path, img_w, img_h):
    """Reads a LabelImg-saved YOLO label back into a pixel box (x, y, w, h)
    against an image of size (img_w, img_h) — safe to call with the resized
    image's dimensions rather than the original raw photo's, since YOLO's
    normalized coordinates are resolution-independent. Only the first line
    is used (one object per photo); the class id embedded in the file is
    ignored in favor of the caller's own class_id — every image in a given
    raw/<class>/ folder belongs to that one class by construction, so
    trusting the folder over a hand-picked label list sidesteps any mismatch
    if a stale/reordered classes.txt was active during labeling.
    """
    line = label_path.read_text().strip().splitlines()[0]
    _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
    w, h = bw * img_w, bh * img_h
    x, y = cx * img_w - w / 2, cy * img_h - h / 2
    return round(x), round(y), round(w), round(h)


def to_yolo_line(class_id, box, img_w, img_h):
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


TIMESTAMP_PATTERNS = [
    # 20260903_054803482_iOS.jpg -> date, HHMMSSfff
    (re.compile(r"^(\d{8})_(\d{2})(\d{2})(\d{2})(\d{3})_iOS$"),
     lambda m: datetime.strptime(m.group(1), "%Y%m%d").replace(
         hour=int(m.group(2)), minute=int(m.group(3)), second=int(m.group(4)),
         microsecond=int(m.group(5)) * 1000)),
    # WIN_20260903_14_48_49_Pro.jpg -> date, HH, MM, SS (optionally suffixed
    # "WIN_..._Pro (2).jpg" — Windows' own de-duplication suffix for a second
    # same-second capture, still the same burst as its unsuffixed sibling)
    (re.compile(r"^WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro(?: \(\d+\))?$"),
     lambda m: datetime.strptime(m.group(1), "%Y%m%d").replace(
         hour=int(m.group(2)), minute=int(m.group(3)), second=int(m.group(4)))),
]
SESSION_GAP = timedelta(minutes=5)  # comfortably above intra-burst gaps (seconds)
# and comfortably below inter-session gaps (hours/days) observed in this dataset.


def parse_timestamp(path):
    for pattern, build in TIMESTAMP_PATTERNS:
        m = pattern.match(path.stem)
        if m:
            return build(m)
    return None  # unrecognized filename — treated as its own singleton session


def group_into_sessions(paths):
    """Clusters paths into capture bursts by filename timestamp, so a whole
    burst (near-duplicate consecutive frames) can be kept together on one
    side of the split — see module docstring. Returns a list of path-lists.
    """
    dated = sorted(((parse_timestamp(p), p) for p in paths), key=lambda x: (x[0] is None, x[0]))
    sessions = []
    current = []
    prev_t = None
    for t, p in dated:
        if t is None:
            if current:
                sessions.append(current)
                current = []
            sessions.append([p])  # unparseable filename: isolated, own session
            prev_t = None
            continue
        if current and prev_t is not None and (t - prev_t) > SESSION_GAP:
            sessions.append(current)
            current = []
        current.append(p)
        prev_t = t
    if current:
        sessions.append(current)
    return sessions


def assign_sessions_to_splits(paths, seed):
    """Assigns whole bursts (never splitting one) to train/val/test, greedily
    giving each next burst to whichever split is currently furthest below its
    target share. With only a handful of (unevenly sized) bursts per class,
    actual per-class ratios will drift from the nominal 72/18/10 — an
    accepted tradeoff for not leaking near-duplicate frames across splits.
    Returns ({path: split_name}, num_sessions, actual_counts, target_counts).
    """
    sessions = group_into_sessions(paths)
    rng = random.Random(seed)
    rng.shuffle(sessions)  # breaks ties between equal-size sessions differently per class
    sessions.sort(key=len, reverse=True)  # place large bursts first — finer-grained sessions fill gaps after

    n = len(paths)
    n_test = round(n * TEST_FRACTION)
    n_val = round((n - n_test) * VAL_FRACTION_OF_TRAINVAL)
    targets = {"train": n - n_test - n_val, "val": n_val, "test": n_test}
    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for session in sessions:
        best = max(targets, key=lambda k: targets[k] - counts[k])
        for p in session:
            assignment[p] = best
        counts[best] += len(session)
    return assignment, len(sessions), counts, targets


def main():
    for name in ("trainval/train/images", "trainval/train/labels", "trainval/val/images",
                 "trainval/val/labels", "test/images", "test/labels"):
        d = DATASET_DIR / name
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    preview_root = DATASET_DIR / "_bbox_preview"
    if preview_root.exists():
        for cls_dir in preview_root.iterdir():
            for f in cls_dir.iterdir():
                f.unlink()
            cls_dir.rmdir()
    preview_root.mkdir(exist_ok=True)

    manifest = []
    source_counts = {"manual": 0, "grabcut": 0, "grabcut_fallback": 0}

    for class_id, slug in enumerate(CLASSES):
        src_dir = RAW_DIR / slug
        paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        split_of, n_sessions, actual_counts, target_counts = assign_sessions_to_splits(paths, seed=SEED + class_id)

        preview_dir = preview_root / slug
        preview_dir.mkdir(exist_ok=True)

        for i, src_path in enumerate(paths):
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"skip (unreadable): {src_path}")
                continue
            img = resize_long_side(img, MAX_SIDE)
            h, w = img.shape[:2]

            manual_label_path = src_path.with_suffix(".txt")
            if manual_label_path.exists():
                box = read_manual_box(manual_label_path, w, h)
                source = "manual"
            else:
                box, used_grabcut = grabcut_box(img)
                source = "grabcut" if used_grabcut else "grabcut_fallback"
            source_counts[source] += 1

            split = split_of[src_path]
            split_dir = DATASET_DIR / "test" if split == "test" else DATASET_DIR / "trainval" / split
            out_name = f"{slug}_{i:04d}"
            cv2.imwrite(str(split_dir / "images" / f"{out_name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            (split_dir / "labels" / f"{out_name}.txt").write_text(to_yolo_line(class_id, box, w, h) + "\n")

            manifest.append({
                "raw_path": str(src_path.relative_to(DATASET_DIR)),
                "class": slug,
                "split": split,
                "box_xywh": list(box),
                "image_size": [w, h],
                "source": source,
            })

            preview_img = img.copy()
            x, y, bw, bh = box
            cv2.rectangle(preview_img, (x, y), (x + bw, y + bh), PREVIEW_COLORS[source], 3)
            cv2.imwrite(str(preview_dir / f"{out_name}.jpg"), preview_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        print(
            f"{slug}: {len(paths)} images across {n_sessions} capture bursts -> "
            f"train={actual_counts['train']} (target {target_counts['train']}) "
            f"val={actual_counts['val']} (target {target_counts['val']}) "
            f"test={actual_counts['test']} (target {target_counts['test']})"
        )

    (DATASET_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    data_yaml = DATASET_DIR / "data.yaml"
    names_block = "\n".join(f"  {cid}: {slug}" for cid, slug in enumerate(CLASSES))
    data_yaml.write_text(
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        f"train: trainval/train/images\n"
        f"val: trainval/val/images\n"
        f"test: test/images\n"
        f"names:\n{names_block}\n"
    )

    total = sum(source_counts.values())
    print(f"\nwrote {DATASET_DIR / 'data.yaml'}")
    print(
        f"box sources: {source_counts['manual']} manual (green), "
        f"{source_counts['grabcut']} auto/GrabCut (blue), "
        f"{source_counts['grabcut_fallback']} auto/fallback (orange) — of {total} total"
    )
    print(f"review every box in backend/dataset/_bbox_preview/<class>/ before training")


if __name__ == "__main__":
    main()
