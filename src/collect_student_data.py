"""
STEP 1: Collect Student Face Data
===================================
Run this script for EACH student to capture their face images.
These images form the training dataset for the face recognizer.

Usage:
    python 1_collect_student_data.py

The script will prompt for student name and ID, then open
the webcam to capture face samples.
"""

import cv2
import os
import sys
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NUM_SAMPLES      = 150       # Face images to capture per student
CAPTURE_DELAY    = 0.05      # Seconds between captures (avoid duplicates)
FACE_SIZE        = (100, 100)  # Saved image dimensions
DATASET_DIR      = "dataset"


def collect_student_faces(student_name: str, student_id: str) -> int:
    """
    Open webcam and capture NUM_SAMPLES grayscale face images for a student.

    Returns:
        Number of images actually saved.
    """
    save_dir = os.path.join(DATASET_DIR, f"{student_id}_{student_name}")
    os.makedirs(save_dir, exist_ok=True)

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        sys.exit(1)

    print(f"\n[INFO] Camera opened.  Collecting data for: {student_name}  (ID: {student_id})")
    print("[INFO] Look at the camera. Press 'q' to stop early.\n")

    count        = 0
    last_capture = 0.0

    while count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame – retrying...")
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            now = time.time()
            if now - last_capture >= CAPTURE_DELAY:
                face_roi  = gray[y : y + h, x : x + w]
                face_roi  = cv2.resize(face_roi, FACE_SIZE)

                # Light augmentation: also save horizontally flipped copy
                img_path  = os.path.join(save_dir, f"{count:04d}.jpg")
                flip_path = os.path.join(save_dir, f"{count:04d}_flip.jpg")
                cv2.imwrite(img_path,  face_roi)
                cv2.imwrite(flip_path, cv2.flip(face_roi, 1))

                count      += 1
                last_capture = now

            # Visual feedback
            color = (0, 255, 0) if count < NUM_SAMPLES else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f"Captured: {count}/{NUM_SAMPLES}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        # HUD overlay
        cv2.putText(
            frame,
            f"Student: {student_name}  |  ID: {student_id}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )
        progress_pct = int(count / NUM_SAMPLES * 100)
        cv2.putText(
            frame,
            f"Progress: {progress_pct}%  [q = quit early]",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        cv2.imshow("Face Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} images to '{save_dir}'")
    return count


def main():
    print("=" * 55)
    print("       ATTENDANCE SYSTEM — Student Data Collector")
    print("=" * 55)

    while True:
        student_name = input("\nEnter student full name  : ").strip()
        student_id   = input("Enter student ID / roll# : ").strip()

        if not student_name or not student_id:
            print("[ERROR] Name and ID cannot be empty.")
            continue

        n = collect_student_faces(student_name, student_id)

        if n < 30:
            print(f"[WARNING] Only {n} images collected. Aim for ≥ 100 for accuracy.")

        another = input("\nAdd another student? (y/n): ").strip().lower()
        if another != "y":
            break

    print("\n[DONE] Data collection complete.")
    print("       Next step → run  train_face_recognizer.py")


if __name__ == "__main__":
    main()
