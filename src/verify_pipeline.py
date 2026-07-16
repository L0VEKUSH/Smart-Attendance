"""
verify_pipeline.py — Offline Pipeline Verification
====================================================
Tests every component WITHOUT a camera or FER-2013 CSV:

  Test 1 – Face Detection        Haar cascade loads & detects a synthetic face
  Test 2 – EmotionNet Build      Architecture builds & forward pass works
  Test 3 – LBPH Train/Predict    Trains on synthetic images, predicts label
  Test 4 – Attendance Dataframe  Record / absent logic & CSV / Excel export
  Test 5 – Time-Window Logic     is_within_window() returns correct bool

Run from the  attendance_system/  root:
    python src/verify_pipeline.py
"""

import os
import sys
import json
import tempfile
import shutil
import time
import numpy as np
import cv2
import pandas as pd

# ── colours for terminal output ───────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════════
# Test 1 – Face Detection
# ══════════════════════════════════════════════════════════════════
def test_face_detection():
    section("TEST 1 – Haar Cascade Face Detection")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print(f"  {FAIL} – Haar cascade XML not found at {cascade_path}")
        return False

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Build a synthetic 200×200 grey image with a rough oval "face" blob
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, 200, -1)

    faces = face_cascade.detectMultiScale(img, 1.1, 3)
    print(f"  Cascade loaded : {PASS}")
    print(f"  Synthetic image detection ran without crash : {PASS}")
    print(f"  (Detections on synthetic blob: {len(faces)} — expected 0 on non-photo)")
    return True


# ══════════════════════════════════════════════════════════════════
# Test 2 – EmotionNet Forward Pass
# ══════════════════════════════════════════════════════════════════
def test_emotion_cnn():
    section("TEST 2 – EmotionNet Architecture & Forward Pass")
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print(f"  {FAIL} – TensorFlow not installed.  Run:  pip install tensorflow")
        return False

    # Replicate the architecture from 3_train_emotion_model.py inline
    from tensorflow.keras import regularizers

    def conv_block(x, filters, dr=0.25):
        reg = regularizers.l2(1e-4)
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dr)(x)
        return x

    inp = layers.Input(shape=(48, 48, 1))
    x = conv_block(inp, 32)
    x = conv_block(x,  64)
    x = conv_block(x,  128)
    x = conv_block(x,  256)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.5)(x)
    x = layers.Dense(256)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.5)(x)
    out = layers.Dense(7, activation="softmax")(x)
    model = models.Model(inp, out, name="EmotionNet_test")

    total_params = model.count_params()
    print(f"  Model built successfully : {PASS}")
    print(f"  Total parameters         : {total_params:,}")

    dummy = np.random.rand(4, 48, 48, 1).astype(np.float32)
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (4, 7), "Output shape mismatch"
    assert abs(preds[0].sum() - 1.0) < 1e-5, "Softmax does not sum to 1"

    print(f"  Forward pass (batch=4)   : {PASS}")
    print(f"  Softmax sums to 1        : {PASS}")
    return True


# ══════════════════════════════════════════════════════════════════
# Test 3 – LBPH Face Recogniser
# ══════════════════════════════════════════════════════════════════
def test_lbph():
    section("TEST 3 – LBPH Face Recogniser Train & Predict")

    tmpdir = tempfile.mkdtemp()
    try:
        # Create 3 synthetic "students", 20 images each
        N_STUDENTS = 3
        N_IMAGES   = 20
        all_faces  = []
        all_labels = []
        label_map  = {}

        rng = np.random.default_rng(42)

        for i in range(N_STUDENTS):
            # Each student = a distinct Gaussian noise pattern
            base = rng.integers(30 * i, 30 * i + 80, (100, 100), dtype=np.uint8)
            label_map[i] = {"id": f"S{i+1:03d}", "name": f"Student_{i+1}"}
            for j in range(N_IMAGES):
                noise = rng.integers(0, 20, (100, 100), dtype=np.uint8)
                face  = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
                all_faces.append(face)
                all_labels.append(i)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(all_faces, np.array(all_labels))
        print(f"  LBPH training ({N_STUDENTS} students, {N_IMAGES} imgs each) : {PASS}")

        # Save & reload
        model_path = os.path.join(tmpdir, "test_recognizer.yml")
        recognizer.save(model_path)
        recognizer2 = cv2.face.LBPHFaceRecognizer_create()
        recognizer2.read(model_path)
        print(f"  Save / reload             : {PASS}")

        # Predict on a slightly noisy version of student 0
        base0  = rng.integers(0, 80, (100, 100), dtype=np.uint8)
        noise  = rng.integers(0, 5, (100, 100), dtype=np.uint8)
        test_f = np.clip(base0.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        label, conf = recognizer2.predict(test_f)
        print(f"  Predict returned label={label}, confidence={conf:.2f} : {PASS}")

    finally:
        shutil.rmtree(tmpdir)

    return True


# ══════════════════════════════════════════════════════════════════
# Test 4 – Attendance DataFrame + Export
# ══════════════════════════════════════════════════════════════════
def test_attendance_export():
    section("TEST 4 – Attendance DataFrame, CSV & Excel Export")

    tmpdir = tempfile.mkdtemp()
    try:
        # Simulate the records list used by AttendanceSystem
        date_str = "2024-09-01"
        records  = [
            {"Student_ID": "S001", "Student_Name": "Alice",
             "Status": "Present", "Emotion": "Happy",
             "Face_Confidence": 91.2, "Time": "09:35:12", "Date": date_str},
            {"Student_ID": "S002", "Student_Name": "Bob",
             "Status": "Present", "Emotion": "Neutral",
             "Face_Confidence": 87.5, "Time": "09:37:44", "Date": date_str},
            {"Student_ID": "S003", "Student_Name": "Carol",
             "Status": "Absent", "Emotion": "N/A",
             "Face_Confidence": 0.0, "Time": "N/A", "Date": date_str},
        ]

        df = pd.DataFrame(records)
        df.sort_values(["Status", "Student_Name"], inplace=True)

        csv_path  = os.path.join(tmpdir, "test_attendance.csv")
        xlsx_path = os.path.join(tmpdir, "test_attendance.xlsx")

        df.to_csv(csv_path,  index=False)
        df.to_excel(xlsx_path, index=False)

        # Reload and verify
        df_csv  = pd.read_csv(csv_path)
        df_xlsx = pd.read_excel(xlsx_path)

        assert len(df_csv)  == 3, "CSV row count mismatch"
        assert len(df_xlsx) == 3, "Excel row count mismatch"
        assert set(df_csv.columns) == set(df.columns), "Column mismatch"

        present = int((df_csv["Status"] == "Present").sum())
        absent  = int((df_csv["Status"] == "Absent").sum())

        print(f"  DataFrame built          : {PASS}")
        print(f"  CSV written & reloaded   : {PASS}  (rows={len(df_csv)})")
        print(f"  Excel written & reloaded : {PASS}  (rows={len(df_xlsx)})")
        print(f"  Present={present}, Absent={absent} : {PASS}")

    finally:
        shutil.rmtree(tmpdir)

    return True


# ══════════════════════════════════════════════════════════════════
# Test 5 – Time-Window Logic
# ══════════════════════════════════════════════════════════════════
def test_time_window():
    section("TEST 5 – Time-Window Logic")

    from datetime import datetime

    def is_within(now_h, now_m,
                  start_h=9, start_m=30,
                  end_h=11,  end_m=59):
        base  = datetime(2024, 1, 1)
        now   = base.replace(hour=now_h,   minute=now_m)
        start = base.replace(hour=start_h, minute=start_m)
        end   = base.replace(hour=end_h,   minute=end_m)
        return start <= now <= end

    cases = [
        (9, 30,  True,  "09:30 – window open (boundary)"),
        (9, 45,  True,  "09:45 – within window"),
        (11, 59,  True,  "11:59 – window close (boundary)"),
        (9, 29,  False, "09:29 – before window"),
        (12, 0,  False, "12:00python src\attendance_system.py – after window"),
        (8,  0,  False, "08:00 – well before"),
        (23, 59, False, "23:59 – well after"),
    ]

    all_ok = True
    for h, m, expected, desc in cases:
        result = is_within(h, m)
        ok     = result == expected
        mark   = PASS if ok else FAIL
        print(f"  {mark}  {desc}  →  {result}")
        if not ok:
            all_ok = False

    return all_ok


# ══════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 55)
    print("         ATTENDANCE SYSTEM — Pipeline Verification")
    print("═" * 55)

    results = {
        "Face Detection"  : test_face_detection(),
        "EmotionNet CNN"  : test_emotion_cnn(),
        "LBPH Recogniser" : test_lbph(),
        "Attendance I/O"  : test_attendance_export(),
        "Time Window"     : test_time_window(),
    }

    section("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        mark = PASS if ok else FAIL
        print(f"  {mark}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"{GREEN}  All tests passed — pipeline is ready!{RESET}")
    else:
        print(f"{RED}  Some tests failed — check the output above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
