"""
respiratory_rate.py — Breathing rate via shoulder oscillation (MediaPipe Pose).

Clinical basis:
  Respiration causes rhythmic superior-inferior movement of the shoulders and
  chest wall. MediaPipe Pose detects shoulder landmarks (lm[11], lm[12]) whose
  Y-coordinate oscillates at the breathing frequency. A bandpass filter isolates
  the respiratory band (0.1-0.8 Hz = 6-48 breaths/min), and peak detection
  counts breath cycles.

Normal adult respiratory rate: 12-20 breaths/min at rest.
Tachypnoea (>20) may indicate fever, anxiety, or respiratory distress.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import mediapipe as mp
    _mp_available = True
except ImportError:
    _mp_available = False
    print("[resp_rate] mediapipe not available.")

FEATURE_NAMES = ['respiratory_rate_bpm', 'peak_count', 'signal_amplitude']


def _bandpass_resp(signal: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 0.1-0.8 Hz for respiratory signal."""
    # CLINICAL NOTE: 0.1-0.8 Hz = 6-48 breaths/min covers normal and abnormal RR.
    nyq = fs / 2.0
    b, a = butter(4, [max(0.1 / nyq, 1e-4), min(0.8 / nyq, 0.9999)], btype='band')
    return filtfilt(b, a, signal)


def extract_respiratory_signal(video_path: str) -> dict:
    """
    Measure respiratory rate from shoulder vertical movement in a video.

    Parameters
    ----------
    video_path : str

    Returns
    -------
    dict with respiratory_rate_bpm, signal arrays, annotated_frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)  # CRITICAL: always read actual FPS
    if fps <= 0 or fps > 240:
        fps = 30.0

    if not _mp_available:
        cap.release()
        raise RuntimeError("[resp_rate] mediapipe required.")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    shoulder_y = []
    timestamps = []
    frame_idx = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # CLINICAL NOTE: Average both shoulders to reduce single-side artifacts.
            y_avg = (lm[11].y + lm[12].y) / 2.0
            shoulder_y.append(y_avg)
            timestamps.append(frame_idx / fps)

            # Draw shoulder line on frame
            lx = int(lm[11].x * w); ly = int(lm[11].y * h)
            rx = int(lm[12].x * w); ry = int(lm[12].y * h)
            cv2.circle(frame, (lx, ly), 8, (255, 100, 0), -1)
            cv2.circle(frame, (rx, ry), 8, (0, 100, 255), -1)
            cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 200), 2)

        last_annotated = frame.copy()
        frame_idx += 1

    cap.release()
    pose.close()

    if len(shoulder_y) < int(fps * 5):
        raise ValueError(f"[resp_rate] Need ≥5s of signal; got {len(shoulder_y)/fps:.1f}s.")

    signal_arr = np.array(shoulder_y, dtype=np.float64)
    ts_arr = np.array(timestamps, dtype=np.float64)

    # Normalise
    signal_norm = (signal_arr - signal_arr.mean()) / (signal_arr.std() + 1e-9)

    # Bandpass 0.1-0.8 Hz
    filtered = _bandpass_resp(signal_norm, fps)

    # Peak detection — minimum 1.5s between breaths
    min_dist = int(fps * 1.5)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=0.2)

    duration_sec = len(signal_arr) / fps
    duration_min = duration_sec / 60.0
    rr = float(len(peaks) / duration_min) if duration_min > 0 else 0.0
    amplitude = float(filtered.max() - filtered.min())

    normal_rr = 12.0 <= rr <= 20.0

    if last_annotated is not None:
        color = (0, 200, 0) if normal_rr else (0, 0, 255)
        cv2.putText(last_annotated, f'RR: {rr:.0f} br/min',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(last_annotated, 'Normal: 12-20' if normal_rr else 'ABNORMAL RR',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return {
        'respiratory_rate_bpm': round(rr, 1),
        'peak_count': len(peaks),
        'duration_sec': round(duration_sec, 1),
        'signal_amplitude': round(amplitude, 4),
        'normal_range_flag': normal_rr,
        'fps': fps,
        'signal': signal_arr,
        'filtered': filtered,
        'timestamps': ts_arr,
        'peaks_indices': peaks,
        'annotated_frame': last_annotated,
    }


def process_webcam_rr(duration_sec: int = 30) -> dict:
    """Record from webcam for duration_sec seconds, return RR dict."""
    import tempfile, os, time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tmp = tempfile.mktemp(suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp, fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))
    print(f"[resp_rate] Recording {duration_sec}s...")
    start = time.time()
    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
    cap.release(); out.release()
    result = extract_respiratory_signal(tmp)
    os.remove(tmp)
    return result


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
