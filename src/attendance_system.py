

import cv2
import numpy as np
import pandas as pd
import os
import json
import sys
import time
import argparse
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# Configuration — edit here if needed
# ─────────────────────────────────────────────
WINDOW_START_H  = 9
WINDOW_START_M  = 0
WINDOW_END_H    = 17
WINDOW_END_M    = 0

# LBPH confidence threshold: lower distance = better match.
# Increase to be more lenient (more false positives),
# decrease to be stricter (more unknowns).
LBPH_THRESHOLD  = 85

# Minimum seconds before the same student can be re-logged
RELOG_COOLDOWN  = 3.0

# Minimum face size accepted by the detector (pixels)
MIN_FACE_PX     = 80

# Paths
FACE_MODEL_PATH    = os.path.join("models", "face_recognizer.yml")
EMOTION_MODEL_PATH = os.path.join("models", "emotion_model.h5")
LABEL_MAP_PATH     = os.path.join("models", "label_map.json")
EMOTION_LABELS_PATH = os.path.join("models", "emotion_labels.json")
OUTPUT_DIR         = "attendance_records"

# Default emotion labels (overridden if emotion_labels.json exists)
DEFAULT_EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


# ══════════════════════════════════════════════════════════════════
# AttendanceSystem class
# ══════════════════════════════════════════════════════════════════
class AttendanceSystem:

    def __init__(self, demo_mode: bool = False):
        self.demo_mode     = demo_mode          # bypass time check
        self.face_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer    = None
        self.emotion_model = None
        self.label_map     = {}                 # int → {id, name}
        self.emotion_labels = DEFAULT_EMOTION_LABELS

        # Attendance state
        self.records: list[dict] = []           # rows for the DataFrame
        self.last_logged: dict   = {}           # student_id → timestamp

        # Stats
        self.frames_processed = 0

    # ──────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────
    def load_models(self):
        """Load LBPH face recogniser, EmotionNet CNN, and label mappings."""
        missing = []
        for path in [FACE_MODEL_PATH, EMOTION_MODEL_PATH, LABEL_MAP_PATH]:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            print("\n[ERROR] Required model files not found:")
            for p in missing:
                print(f"        ✗  {p}")
            print("\n  → Complete Steps 1–3 first, then run this script.")
            sys.exit(1)

        print("[INFO] Loading face recognizer ...")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(FACE_MODEL_PATH)

        print("[INFO] Loading emotion model ...")
        self.emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)

        with open(LABEL_MAP_PATH, "r") as f:
            raw = json.load(f)
            self.label_map = {int(k): v for k, v in raw.items()}

        if os.path.exists(EMOTION_LABELS_PATH):
            with open(EMOTION_LABELS_PATH, "r") as f:
                self.emotion_labels = json.load(f)

        n = len(self.label_map)
        names = [v["name"] for v in self.label_map.values()]
        print(f"[INFO] Models loaded.  Registered students ({n}): {', '.join(names)}")

    # ──────────────────────────────────────────────────────────────
    # Time window helpers
    # ──────────────────────────────────────────────────────────────
    def _window_times(self):
        now   = datetime.now()
        start = now.replace(hour=WINDOW_START_H, minute=WINDOW_START_M,
                            second=0, microsecond=0)
        end   = now.replace(hour=WINDOW_END_H,   minute=WINDOW_END_M,
                            second=0, microsecond=0)
        return start, end

    def is_within_window(self) -> bool:
        if self.demo_mode:
            return True
        start, end = self._window_times()
        return start <= datetime.now() <= end

    def wait_for_window(self):
        """Block until the attendance window opens (skip if demo)."""
        if self.demo_mode:
            print("[DEMO] Time-window restriction disabled.\n")
            return

        start, end = self._window_times()
        now = datetime.now()

        if now > end:
            print("[ERROR] Today's attendance window (09:30–10:00) has already passed.")
            sys.exit(0)

        if now < start:
            wait_secs = int((start - now).total_seconds())
            print(f"[INFO] Attendance window opens at 09:30 AM.")
            print(f"[INFO] Waiting {wait_secs // 60}m {wait_secs % 60}s ...")
            time.sleep(wait_secs)

    # ──────────────────────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────────────────────
    def detect_faces(self, gray_frame):
        """Return list of (x, y, w, h) bounding boxes."""
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
        )

    def recognise_face(self, face_gray_100):
        """
        Run LBPH recogniser.
        Returns (student_info_dict, confidence) or (None, confidence) if unknown.
        """
        label, confidence = self.recognizer.predict(face_gray_100)
        if confidence < LBPH_THRESHOLD:
            return self.label_map.get(label, None), confidence
        return None, confidence

    def detect_emotion(self, face_gray):
        """
        Run EmotionNet CNN on a grayscale face ROI.
        Returns (emotion_label, probability).
        """
        face = cv2.resize(face_gray, (48, 48)).astype(np.float32) / 255.0
        face = face.reshape(1, 48, 48, 1)
        probs       = self.emotion_model.predict(face, verbose=0)[0]
        idx         = int(np.argmax(probs))
        emotion     = self.emotion_labels[idx] if idx < len(self.emotion_labels) else "Unknown"
        probability = float(probs[idx])
        return emotion, probability

    # ──────────────────────────────────────────────────────────────
    # Attendance recording
    # ──────────────────────────────────────────────────────────────
    def log_presence(self, student_info: dict, emotion: str, confidence: float):
        """
        Record a student as Present (at most once per RELOG_COOLDOWN seconds).
        """
        sid = student_info["id"]
        now = time.time()

        if sid in self.last_logged and (now - self.last_logged[sid]) < RELOG_COOLDOWN:
            return  # too soon since last log

        self.last_logged[sid] = now

        # Only write the FIRST detection as 'Present'
        already_present = any(r["Student_ID"] == sid and r["Status"] == "Present"
                               for r in self.records)
        if already_present:
            return

        ts = datetime.now()
        record = {
            "Student_ID"      : sid,
            "Student_Name"    : student_info["name"],
            "Status"          : "Present",
            "Emotion"         : emotion,
            "Face_Confidence" : round(100.0 - confidence, 2),
            "Time"            : ts.strftime("%H:%M:%S"),
            "Date"            : ts.strftime("%Y-%m-%d"),
        }
        self.records.append(record)
        print(f"  ✓  Present  │ {student_info['name']:20s} │ {emotion:10s} │ "
              f"conf {100 - confidence:.1f}%  │ {ts.strftime('%H:%M:%S')}")

    def finalise_absentees(self):
        """Add Absent rows for every registered student not already Present."""
        present_ids = {r["Student_ID"] for r in self.records}
        date_str    = datetime.now().strftime("%Y-%m-%d")

        for info in self.label_map.values():
            if info["id"] not in present_ids:
                self.records.append({
                    "Student_ID"      : info["id"],
                    "Student_Name"    : info["name"],
                    "Status"          : "Absent",
                    "Emotion"         : "N/A",
                    "Face_Confidence" : 0.0,
                    "Time"            : "N/A",
                    "Date"            : date_str,
                })
                print(f"  ✗  Absent   │ {info['name']}")

    # ──────────────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────────────
    def save_attendance(self):
        """Write attendance to CSV and Excel, print summary."""
        if not self.records:
            print("\n[WARNING] No attendance data recorded.")
            return

        df = pd.DataFrame(self.records)
        df.sort_values(["Status", "Student_Name"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp     = datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_path  = os.path.join(OUTPUT_DIR, f"attendance_{stamp}.csv")
        xlsx_path = os.path.join(OUTPUT_DIR, f"attendance_{stamp}.xlsx")

        df.to_csv(csv_path,  index=False)
        df.to_excel(xlsx_path, index=False)

        present_n = int((df["Status"] == "Present").sum())
        absent_n  = int((df["Status"] == "Absent").sum())
        total_n   = present_n + absent_n

        print("\n" + "─" * 55)
        print(f"  ATTENDANCE SAVED")
        print(f"  CSV   →  {csv_path}")
        print(f"  Excel →  {xlsx_path}")
        print("─" * 55)
        print(f"  Total Students : {total_n}")
        print(f"  Present        : {present_n}  ({present_n/total_n*100:.0f}%)" if total_n else "")
        print(f"  Absent         : {absent_n}")
        print("─" * 55)

        print("\n  Detailed Report:")
        print(df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────
    # HUD drawing
    # ──────────────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        """Overlay time, window status, and present count onto the frame."""
        now            = datetime.now()
        _, end         = self._window_times()
        remaining_sec  = max(0, int((end - now).total_seconds()))
        rm, rs         = divmod(remaining_sec, 60)

        mode_str = "DEMO MODE" if self.demo_mode else f"Ends: {WINDOW_END_H:02d}:{WINDOW_END_M:02d}"

        present_count = sum(1 for r in self.records if r["Status"] == "Present")

        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame,
                    f"ATTENDANCE SYSTEM  |  {now.strftime('%H:%M:%S')}  |  {mode_str}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(frame,
                    f"Remaining: {rm}m {rs:02d}s  |  Present: {present_count}/{len(self.label_map)}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "═" * 55)
        print("         SMART ATTENDANCE SYSTEM")
        print(f"         Window: {WINDOW_START_H:02d}:{WINDOW_START_M:02d}  –  "
              f"{WINDOW_END_H:02d}:{WINDOW_END_M:02d}")
        print("═" * 55)

        self.wait_for_window()
        self.load_models()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            sys.exit(1)

        print("\n[INFO] Camera active. Press  'q'  to stop early.")
        print("\n  Status  │  Name                │  Emotion     │  Conf   │  Time")
        print("  " + "─" * 53)

        try:
            while self.is_within_window():
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Frame grab failed. Retrying ...")
                    continue

                self.frames_processed += 1
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detect_faces(gray)

                for (x, y, w, h) in faces:
                    face_gray     = gray[y: y + h, x: x + w]
                    face_100      = cv2.resize(face_gray, (100, 100))

                    student_info, confidence = self.recognise_face(face_100)

                    if student_info:
                        emotion, _ = self.detect_emotion(face_gray)
                        self.log_presence(student_info, emotion, confidence)

                        label_txt = f"{student_info['name']} | {emotion}"
                        conf_txt  = f"{100 - confidence:.1f}%"
                        box_color = (0, 220, 0)
                    else:
                        label_txt = "Unknown"
                        conf_txt  = ""
                        box_color = (0, 0, 220)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(frame, label_txt,
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
                    if conf_txt:
                        cv2.putText(frame, conf_txt,
                                    (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (200, 200, 200), 1)

                self._draw_hud(frame)
                cv2.imshow("Attendance System", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Stopped manually.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"\n[INFO] Frames processed : {self.frames_processed}")
        print("[INFO] Finalising attendance ...")
        self.finalise_absentees()
        self.save_attendance()


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Attendance System")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run outside the normal time window (for testing)",
    )
    args = parser.parse_args()

    system = AttendanceSystem(demo_mode=args.demo)
    system.run()
