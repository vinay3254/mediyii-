"""
drowsiness.py — Eye Aspect Ratio (EAR) based drowsiness detection.

Clinical basis:
  The EAR (Eye Aspect Ratio) measures how open the eye is using 6 facial
  landmarks per eye. When EAR falls below 0.25 and stays there for ≥20
  consecutive frames (~0.67s at 30 fps), the eye is closing — indicating
  drowsiness. PERCLOS (% time eyes closed > 80%) is the NHTSA standard
  metric for fatigue: >15% PERCLOS = significantly fatigued.

Reference:
  Soukupová & Čech (2016). "Real-Time Eye Blink Detection using Facial
  Landmarks." CVWW 2016.

Use cases: ICU staff monitoring, long-haul truck drivers, night-shift workers.
"""

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    _mp_available = True
except ImportError:
    _mp_available = False

# ── MediaPipe Face Mesh eye landmark indices (468-landmark model) ─────────────
# CLINICAL NOTE: These 6 points per eye define the EAR formula:
#   p1=outer corner, p2=upper outer, p3=upper inner,
#   p4=inner corner, p5=lower inner, p6=lower outer
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD  = 0.25   # below this = eye closing
CONSEC_FRAMES  = 20     # frames of closure before drowsy alert
PERCLOS_THRESH = 15.0   # % eyes-closed => fatigued (NHTSA standard)

FEATURE_NAMES = ['mean_ear', 'min_ear', 'perclos', 'blink_rate']


def eye_aspect_ratio(landmarks, eye_indices: list,
                     frame_w: int, frame_h: int) -> float:
    """
    Compute EAR from 6 landmark indices.

    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

    CLINICAL NOTE: EAR ≈ 0.30 when eyes fully open; < 0.25 = closing.
    Blink: EAR drops briefly; prolonged closure = microsleep.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h]))

    p1, p2, p3, p4, p5, p6 = pts
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / \
          (2.0 * np.linalg.norm(p1 - p4) + 1e-9)
    return float(ear)


def analyze_drowsiness_video(video_path: str) -> dict:
    """
    Analyse a video for signs of drowsiness using EAR and PERCLOS.

    Returns dict with mean_ear, min_ear, perclos, blink_rate, drowsy_flag,
    ear_history, annotated_frame.
    """
    if not _mp_available:
        raise RuntimeError("[drowsiness] mediapipe is required.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)  # CRITICAL: read actual FPS
    if fps <= 0 or fps > 240:
        fps = 30.0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ear_history = []
    consec = 0
    closed_frames = 0
    blink_count = 0
    drowsy_events = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm_list = results.multi_face_landmarks[0].landmark
            left_ear  = eye_aspect_ratio(lm_list, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lm_list, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            ear_history.append(ear)

            if ear < EAR_THRESHOLD:
                consec += 1
                closed_frames += 1
                if consec == 1:
                    blink_count += 1   # start of a new closure event
                if consec >= CONSEC_FRAMES:
                    drowsy_events += 1
            else:
                consec = 0

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = lm_list[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 200), -1)

            # EAR bar
            bar_h = int(ear * 200)
            bar_color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 200, 0)
            cv2.rectangle(frame, (w - 20, h - bar_h - 5),
                          (w - 5, h - 5), bar_color, -1)

            alert = consec >= CONSEC_FRAMES
            text_color = (0, 0, 255) if alert else (0, 200, 0)
            cv2.putText(frame, f'EAR: {ear:.3f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            if alert:
                cv2.putText(frame, '!!! DROWSY ALERT !!!',
                            (w // 2 - 140, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        last_annotated = frame.copy()

    cap.release()
    face_mesh.close()

    if not ear_history:
        raise ValueError("[drowsiness] No face detected in video.")

    total_frames = len(ear_history)
    # CLINICAL NOTE: PERCLOS = % frames where EAR < EAR_THRESHOLD.
    # NHTSA standard: PERCLOS > 15% indicates significant fatigue.
    perclos = closed_frames / total_frames * 100.0
    mean_ear = float(np.mean(ear_history))
    min_ear  = float(np.min(ear_history))
    duration_min = total_frames / fps / 60.0
    blink_rate = float(blink_count / duration_min) if duration_min > 0 else 0.0

    drowsy_flag = perclos > PERCLOS_THRESH or drowsy_events > 0

    return {
        'mean_ear': round(mean_ear, 4),
        'min_ear': round(min_ear, 4),
        'perclos': round(perclos, 2),
        'blink_rate': round(blink_rate, 1),
        'drowsy_flag': drowsy_flag,
        'drowsy_events': drowsy_events,
        'blink_count': blink_count,
        'fps': fps,
        'total_frames': total_frames,
        'ear_history': np.array(ear_history),
        'annotated_frame': last_annotated,
    }


def analyze_webcam_drowsiness(duration_sec: int = 60) -> dict:
    """Real-time webcam drowsiness monitor. Shows live EAR on screen."""
    import time

    if not _mp_available:
        raise RuntimeError("[drowsiness] mediapipe required.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                 refine_landmarks=True)
    ear_history = []
    consec = 0; closed_frames = 0; blink_count = 0
    start = time.time()
    last_frame = None

    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lm_list = results.multi_face_landmarks[0].landmark
            ear = (eye_aspect_ratio(lm_list, LEFT_EYE, w, h) +
                   eye_aspect_ratio(lm_list, RIGHT_EYE, w, h)) / 2.0
            ear_history.append(ear)
            if ear < EAR_THRESHOLD:
                consec += 1; closed_frames += 1
                if consec == 1: blink_count += 1
            else:
                consec = 0
            color = (0, 0, 255) if consec >= CONSEC_FRAMES else (0, 200, 0)
            cv2.putText(frame, f'EAR:{ear:.3f}  {"DROWSY!" if consec>=CONSEC_FRAMES else "Alert"}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Drowsiness Monitor — Q to quit', frame)
        last_frame = frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); face_mesh.close(); cv2.destroyAllWindows()

    total = len(ear_history)
    perclos = closed_frames / total * 100.0 if total else 0.0
    return {
        'mean_ear': float(np.mean(ear_history)) if ear_history else 0.0,
        'min_ear': float(np.min(ear_history)) if ear_history else 0.0,
        'perclos': round(perclos, 2),
        'blink_rate': blink_count / (total / fps / 60.0) if total else 0.0,
        'drowsy_flag': perclos > PERCLOS_THRESH,
        'fps': fps,
        'total_frames': total,
        'ear_history': np.array(ear_history),
        'annotated_frame': last_frame,
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
