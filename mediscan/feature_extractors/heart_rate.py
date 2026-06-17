"""
heart_rate.py — Remote photoplethysmography (rPPG) heart rate from video.

Technique: rPPG (remote PPG / imaging PPG)
  Each heartbeat pumps blood through skin capillaries, causing subtle (~0.5-1%)
  changes in green-channel reflectance from the forehead. These oscillations
  at the heart rate frequency can be extracted by:
    1. Track forehead ROI per frame via MediaPipe FaceDetection
    2. Collect mean green channel value per frame
    3. Bandpass filter 0.7-3.0 Hz (42-180 BPM)
    4. FFT → dominant frequency × 60 = BPM

Reference:
  Verkruysse et al. (2008). Remote plethysmographic imaging using ambient light.
  Optics Express, 16(26), 21434-21445.
  de Haan & Jeanne (2013). Robust pulse rate from chrominance-based rPPG.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import mediapipe as mp
    _mp_available = True
except ImportError:
    _mp_available = False
    print("[heart_rate] mediapipe not installed — face detection disabled, using full-frame green channel.")

FEATURE_NAMES = ['bpm_fft', 'dominant_freq_hz', 'signal_quality']


def _bandpass(signal: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """4th-order Butterworth bandpass filter."""
    # CLINICAL NOTE: 0.7-3.0 Hz = 42-180 BPM covers resting + exercise HR range.
    nyq = fs / 2.0
    lo_n, hi_n = lo / nyq, hi / nyq
    lo_n = np.clip(lo_n, 1e-4, 0.9999)
    hi_n = np.clip(hi_n, 1e-4, 0.9999)
    b, a = butter(4, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, signal)


def extract_rppg_signal(video_path: str) -> dict:
    """
    Extract heart rate (BPM) from a face video using rPPG.

    Parameters
    ----------
    video_path : str — path to video file.

    Returns
    -------
    dict with bpm_fft, bpm_peaks, raw_signal, filtered_signal, etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[heart_rate] Cannot open video: {video_path}")

    # CRITICAL: Always read actual FPS — never hardcode 30.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        print(f"[heart_rate] WARNING: Unusual FPS={fps:.1f}. Defaulting to 30.")
        fps = 30.0

    face_detector = None
    if _mp_available:
        face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5)

    green_signal = []
    timestamps = []
    frame_idx = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        h, w = frame.shape[:2]
        forehead = None

        if face_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)
            if results.detections:
                det = results.detections[0]
                bbox = det.location_data.relative_bounding_box
                fx = max(0, int(bbox.xmin * w))
                fy = max(0, int(bbox.ymin * h))
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                # CLINICAL NOTE: Forehead = top 25% of face bounding box.
                # Forehead has thin skin and minimal movement artifact.
                forehead = frame[fy: fy + fh // 4, fx: fx + fw]

                # Draw annotations on last frame
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.rectangle(frame,
                              (fx, fy), (fx + fw, fy + fh // 4),
                              (0, 128, 255), 2)
                cv2.putText(frame, 'Forehead ROI',
                            (fx, max(fy - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 128, 255), 1)
        else:
            # Fallback: centre third of frame as forehead proxy
            y1, y2 = h // 6, h // 3
            x1, x2 = w // 4, 3 * w // 4
            forehead = frame[y1:y2, x1:x2]

        if forehead is not None and forehead.size > 0:
            # CLINICAL NOTE: Green channel captures oxy-Hb absorption at ~540 nm.
            green_signal.append(float(forehead[:, :, 1].mean()))
            timestamps.append(ts)

        last_annotated = frame.copy()
        frame_idx += 1

    cap.release()
    if face_detector is not None:
        face_detector.close()

    if len(green_signal) < int(fps * 5):
        raise ValueError(f"[heart_rate] Need ≥5 s of signal; got {len(green_signal)/fps:.1f}s.")

    signal_arr = np.array(green_signal, dtype=np.float64)
    ts_arr = np.array(timestamps, dtype=np.float64)

    # ── Normalise ────────────────────────────────────────────────────────────
    signal_norm = (signal_arr - signal_arr.mean()) / (signal_arr.std() + 1e-9)

    # ── Bandpass 0.7-3.0 Hz ──────────────────────────────────────────────────
    filtered = _bandpass(signal_norm, lo=0.7, hi=3.0, fs=fps)

    # ── FFT-based BPM ────────────────────────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    dominant_freq = float(freqs[mask][np.argmax(fft_mag[mask])])
    bpm_fft = dominant_freq * 60.0

    # ── Peak-count BPM ───────────────────────────────────────────────────────
    peaks, _ = find_peaks(filtered, distance=fps * 0.5)
    duration_min = len(signal_arr) / fps / 60.0
    bpm_peaks = float(len(peaks) / duration_min) if duration_min > 0 else 0.0

    signal_quality = float(filtered.std())

    if last_annotated is not None:
        cv2.putText(last_annotated, f'BPM: {bpm_fft:.0f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    return {
        'bpm_fft': round(bpm_fft, 1),
        'bpm_peaks': round(bpm_peaks, 1),
        'dominant_freq_hz': round(dominant_freq, 3),
        'signal_quality': round(signal_quality, 4),
        'fps': fps,
        'frame_count': frame_idx,
        'raw_signal': signal_arr,
        'filtered_signal': filtered,
        'timestamps': ts_arr,
        'fft_freqs': freqs,
        'fft_magnitude': fft_mag,
        'annotated_frame': last_annotated,
    }


def process_webcam_rppg(duration_sec: int = 30) -> dict:
    """Record from default webcam for duration_sec seconds, return same dict."""
    import tempfile, os, time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[heart_rate] Cannot open webcam.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        cap.set(cv2.CAP_PROP_FPS, 30)

    tmp = tempfile.mktemp(suffix='.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(tmp, fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))

    print(f"[heart_rate] Recording {duration_sec}s from webcam...")
    start = time.time()
    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            elapsed = time.time() - start
            if int(elapsed) % 5 == 0 and elapsed % 5 < 0.1:
                print(f"  {int(elapsed)}s / {duration_sec}s")

    cap.release()
    out.release()
    result = extract_rppg_signal(tmp)
    os.remove(tmp)
    return result


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
