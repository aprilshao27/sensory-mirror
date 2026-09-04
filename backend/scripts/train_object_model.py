"""Fine-tunes YOLO11s on the object-recognition dataset built by
prepare_dataset.py, evaluates on the held-out test split, and publishes the
result to the stable path the runtime loads.

Hyperparameters are left at Ultralytics' own tuned defaults (augmentation,
optimizer, batch sizing) wherever there's no dataset-specific reason to
override them — see README.md's "Object recognition dataset & model"
section for the rationale behind what IS set below (epochs/imgsz/patience)
and why full fine-tuning (not a frozen backbone) was chosen.

Run from backend/: .venv/Scripts/python.exe scripts/train_object_model.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

BACKEND_DIR = Path(__file__).parent.parent
DATA_YAML = BACKEND_DIR / "dataset" / "data.yaml"
RUNS_DIR = BACKEND_DIR / "runs"
RUN_NAME = "object_recognition"
MODELS_DIR = BACKEND_DIR / "models"
PUBLISHED_WEIGHTS = MODELS_DIR / "object_recognition.pt"

EPOCHS = 100
IMGSZ = 640
PATIENCE = 30
SEED = 0


def main():
    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found — run scripts/prepare_dataset.py first")

    model = YOLO("yolo11s.pt")  # COCO-pretrained
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        patience=PATIENCE,
        seed=SEED,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\n{'=' * 60}\nEvaluating on held-out TEST split (never seen during training)\n{'=' * 60}")
    test_model = YOLO(str(best_weights))
    metrics = test_model.val(
        data=str(DATA_YAML),
        split="test",
        project=str(RUNS_DIR),
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
    )
    print(f"\ntest mAP50: {metrics.box.map50:.4f}   test mAP50-95: {metrics.box.map:.4f}")
    for i, name in metrics.names.items():
        print(f"  {name}: mAP50={metrics.box.ap50[i]:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best_weights, PUBLISHED_WEIGHTS)
    print(f"\npublished {PUBLISHED_WEIGHTS}")


if __name__ == "__main__":
    main()
