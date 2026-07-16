"""
STEP 2: Train Face Recognition Model (LBPH)
=============================================
Trains an OpenCV LBPH (Local Binary Patterns Histograms) face
recognizer on the images collected in Step 1.

LBPH works by:
  1. Dividing the face into a grid of small cells.
  2. Computing a Local Binary Pattern histogram for each cell.
  3. Concatenating histograms into a single feature vector.
  4. At inference, comparing the feature vector against stored
     training vectors using Chi-Square distance.

Outputs:
  models/face_recognizer.yml   – trained LBPH model
  models/label_map.json        – integer label → student info mapping

Usage:
    python 2_train_face_recognizer.py
"""

import cv2
import numpy as np
import json
import os
import sys

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODELS_DIR   = "models"
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognizer.yml")
LABEL_MAP_PATH  = os.path.join(MODELS_DIR, "label_map.json")

# LBPH hyper-parameters
LBPH_RADIUS    = 1   # Radius of the circular LBP neighbourhood
LBPH_NEIGHBORS = 8   # Number of sample points
LBPH_GRID_X    = 8   # Grid columns
LBPH_GRID_Y    = 8   # Grid rows


def load_dataset(dataset_dir: str):
    """
    Walk the dataset directory, load every image and assign integer labels.

    Directory naming convention:  <student_id>_<student_name>
    e.g.  dataset/S001_Alice/  →  id="S001", name="Alice"

    Returns:
        faces     : list of grayscale numpy arrays (100×100)
        labels    : np.ndarray of integer labels
        label_map : dict  { int_label → {"id": ..., "name": ...} }
    """
    faces     = []
    labels    = []
    label_map = {}
    label_id  = 0

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset folder '{dataset_dir}' not found.")
        print("        Run collect_student_data.py first.")
        sys.exit(1)

    student_folders = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not student_folders:
        print("[ERROR] No student folders found inside 'dataset/'.")
        sys.exit(1)

    for folder_name in student_folders:
        folder_path = os.path.join(dataset_dir, folder_name)

        # Parse folder name: "S001_Alice" → id="S001", name="Alice"
        parts        = folder_name.split("_", 1)
        student_id   = parts[0] if len(parts) >= 1 else folder_name
        student_name = parts[1] if len(parts) == 2 else folder_name

        label_map[label_id] = {"id": student_id, "name": student_name}
        img_count = 0

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, fname)
            img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))
            faces.append(img)
            labels.append(label_id)
            img_count += 1

        print(f"  [+] {student_name:20s} (ID: {student_id})  —  {img_count} images  →  label {label_id}")
        label_id += 1

    return faces, np.array(labels), label_map


def train_and_save():
    print("=" * 55)
    print("       ATTENDANCE SYSTEM — Face Recognizer Training")
    print("=" * 55)
    print("\n[INFO] Loading images from dataset/...")

    faces, labels, label_map = load_dataset(DATASET_DIR)
    total_images   = len(faces)
    total_students = len(label_map)

    if total_images == 0:
        print("[ERROR] No images loaded. Check your dataset.")
        sys.exit(1)

    print(f"\n[INFO] Dataset summary:")
    print(f"       Students : {total_students}")
    print(f"       Images   : {total_images}")

    # ── Train LBPH recognizer ──────────────────────────────────────
    print("\n[INFO] Training LBPH Face Recognizer ...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )
    recognizer.train(faces, labels)
    print("[INFO] Training complete.")

    # ── Save model and label map ───────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    recognizer.save(RECOGNIZER_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        # JSON keys must be strings
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print(f"\n[SAVED] Face model  →  {RECOGNIZER_PATH}")
    print(f"[SAVED] Label map   →  {LABEL_MAP_PATH}")
    print("\n[DONE]  Next step   →  train_emotion_model.py")


if __name__ == "__main__":
    train_and_save()
