"""
tremor.py
---------
Tremor frequency analysis for Parkinson's screening.

Clinical background
-------------------
Parkinson's disease is characterised by a *resting* tremor oscillating at
4–6 Hz.  Essential tremor — the most common movement disorder — produces an
*action / postural* tremor at 6–12 Hz.  Physiological tremor (present in all
healthy individuals) sits above 10 Hz and is of very low amplitude.

This module captures the 3-D trajectory of the index-finger tip (MediaPipe
landmark 8) from a short video clip and decomposes the signal with the Fast
Fourier Transform to identify which frequency band carries the most energy.

References
----------
- Deuschl G et al. (1998) Consensus statement of the Movement Disorder
  Society on tremor. Mov Disord 13(S3):2-23.
- Elble RJ (2003) Characteristics of physiologic tremor in young and
  elderly adults. Clin Neurophysiol 114(4):624-635.
"""

import cv2
import numpy as np
from scipy import signal as scipy_signal  # imported but available for future filtering
import mediapipe as mp

# ---------------------------------------------------------------------------
# Public feature list – consumed by the ML pipeline
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'dominant_freq_x',
    'power_4_6_x',
    'power_6_12_x',
    'parkinsonian_ratio_x',
    'rms_jitter',
]


def analyze_tremor_video(video_path: str) -> dict:
    """
    Analyse a video of an outstretched / resting hand and return quantitative
    tremor metrics.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to an MP4 / AVI video file.

    Returns
    -------
    dict
        Keys
        ----
        dominant_freq_x       : float  - Hz, X-axis dominant oscillation
        dominant_freq_y       : float  - Hz, Y-axis dominant oscillation
        power_4_6_x           : float  - summed FFT magnitude in 4-6 Hz band (X)
        power_6_12_x          : float  - summed FFT magnitude in 6-12 Hz band (X)
        parkinsonian_ratio_x  : float  - fraction of total power in 4-6 Hz (X)
        essential_ratio_x     : float  - fraction of total power in 6-12 Hz (X)
        jitter_x              : float  - frame-to-frame std of X displacements
        jitter_y              : float  - frame-to-frame std of Y displacements
        rms_jitter            : float  - sqrt(jitter_x**2 + jitter_y**2)
        tremor_label          : str    - clinical classification string
        fps                   : float  - actual video frame rate
        frame_count           : int    - total frames processed
        tip_x_arr             : np.ndarray - raw normalised X positions
        tip_y_arr             : np.ndarray - raw normalised Y positions
        freqs                 : np.ndarray - FFT frequency bins (Hz)
        fft_mag_x             : np.ndarray - FFT magnitude array for X axis
        annotated_frame       : np.ndarray - last BGR frame with hand landmarks
    """

    # ------------------------------------------------------------------
    # 1. Open video and read native frame rate
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback to 30 fps if the container does not report a valid rate
        fps = 30.0

    # ------------------------------------------------------------------
    # 2. Initialise MediaPipe Hands
    #    max_num_hands=1 limits tracking to a single hand for efficiency.
    #    min_detection_confidence=0.5 balances recall vs. false positives.
    # ------------------------------------------------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tip_x_list = []   # normalised [0,1] X coordinate of index fingertip
    tip_y_list = []   # normalised [0,1] Y coordinate of index fingertip
    timestamps = []   # seconds elapsed since video start
    frame_idx = 0
    annotated_frame = None  # will hold the last processed frame

    # ------------------------------------------------------------------
    # 3. Frame-by-frame landmark extraction
    # ------------------------------------------------------------------
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]  # first (and only) hand

            # Landmark 8 = INDEX_FINGER_TIP
            # Coordinates are normalised to [0,1] relative to frame size
            tip_x_list.append(hand_lm.landmark[8].x)
            tip_y_list.append(hand_lm.landmark[8].y)
            timestamps.append(frame_idx / fps)

            # Draw skeleton for visual feedback
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
            )
            annotated_frame = frame_bgr.copy()

        frame_idx += 1

    cap.release()
    hands_detector.close()

    frame_count = frame_idx

    # ------------------------------------------------------------------
    # 4. Guard: need at least 8 samples for a meaningful FFT
    # ------------------------------------------------------------------
    if len(tip_x_list) < 8:
        return {
            'dominant_freq_x': 0.0,
            'dominant_freq_y': 0.0,
            'power_4_6_x': 0.0,
            'power_6_12_x': 0.0,
            'parkinsonian_ratio_x': 0.0,
            'essential_ratio_x': 0.0,
            'jitter_x': 0.0,
            'jitter_y': 0.0,
            'rms_jitter': 0.0,
            'tremor_label': 'Insufficient data',
            'fps': fps,
            'frame_count': frame_count,
            'tip_x_arr': np.array(tip_x_list),
            'tip_y_arr': np.array(tip_y_list),
            'freqs': np.array([]),
            'fft_mag_x': np.array([]),
            'annotated_frame': annotated_frame,
        }

    tip_x_arr = np.array(tip_x_list)
    tip_y_arr = np.array(tip_y_list)
    n = len(tip_x_arr)

    # ------------------------------------------------------------------
    # 5. FFT analysis for X axis
    # ------------------------------------------------------------------
    # Remove DC (mean) so the zero-frequency bin does not dominate
    sig_x = tip_x_arr - tip_x_arr.mean()

    fft_mag_x = np.abs(np.fft.rfft(sig_x))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)  # frequency resolution = fps/n Hz

    # Skip DC component (index 0) when looking for the dominant frequency
    dominant_freq_x = freqs[np.argmax(fft_mag_x[1:]) + 1]

    # CLINICAL NOTE: Resting tremor at 4-6 Hz is characteristic of Parkinson's disease
    power_4_6_x = fft_mag_x[(freqs >= 4) & (freqs <= 6)].sum()

    # CLINICAL NOTE: Action/postural tremor at 6-12 Hz suggests essential tremor
    power_6_12_x = fft_mag_x[(freqs >= 6) & (freqs <= 12)].sum()

    power_total_x = fft_mag_x.sum() + 1e-9
    parkinsonian_ratio_x = power_4_6_x / power_total_x
    essential_ratio_x = power_6_12_x / power_total_x

    # ------------------------------------------------------------------
    # 6. FFT analysis for Y axis
    # ------------------------------------------------------------------
    sig_y = tip_y_arr - tip_y_arr.mean()
    fft_mag_y = np.abs(np.fft.rfft(sig_y))
    freqs_y = np.fft.rfftfreq(n, d=1.0 / fps)

    dominant_freq_y = freqs_y[np.argmax(fft_mag_y[1:]) + 1]

    power_4_6_y = fft_mag_y[(freqs_y >= 4) & (freqs_y <= 6)].sum()
    power_6_12_y = fft_mag_y[(freqs_y >= 6) & (freqs_y <= 12)].sum()
    power_total_y = fft_mag_y.sum() + 1e-9
    parkinsonian_ratio_y = power_4_6_y / power_total_y
    essential_ratio_y = power_6_12_y / power_total_y

    # ------------------------------------------------------------------
    # 7. Jitter metrics (frame-to-frame displacement variability)
    #    High jitter with low dominant frequency = coarse tremor
    #    np.diff computes successive differences: x[i+1] - x[i]
    # ------------------------------------------------------------------
    jitter_x = float(np.std(np.diff(tip_x_arr)))   # frame-to-frame jitter X
    jitter_y = float(np.std(np.diff(tip_y_arr)))   # frame-to-frame jitter Y
    rms_jitter = float(np.sqrt(jitter_x ** 2 + jitter_y ** 2))

    # ------------------------------------------------------------------
    # 8. Clinical classification
    #    Priority order: Parkinsonian > Essential > Normal > Physiological
    # ------------------------------------------------------------------
    if 4.0 <= dominant_freq_x <= 6.0 and parkinsonian_ratio_x > 0.3:
        tremor_label = 'Parkinsonian'
    elif 6.0 <= dominant_freq_x <= 12.0 and essential_ratio_x > 0.3:
        tremor_label = 'Essential Tremor'
    elif rms_jitter < 0.005:
        tremor_label = 'Normal (minimal tremor)'
    else:
        tremor_label = 'Physiological tremor'

    # ------------------------------------------------------------------
    # 9. Return results
    # ------------------------------------------------------------------
    return {
        'dominant_freq_x': float(dominant_freq_x),
        'dominant_freq_y': float(dominant_freq_y),
        'power_4_6_x': float(power_4_6_x),
        'power_6_12_x': float(power_6_12_x),
        'parkinsonian_ratio_x': float(parkinsonian_ratio_x),
        'essential_ratio_x': float(essential_ratio_x),
        'jitter_x': jitter_x,
        'jitter_y': jitter_y,
        'rms_jitter': rms_jitter,
        'tremor_label': tremor_label,
        'fps': fps,
        'frame_count': frame_count,
        'tip_x_arr': tip_x_arr,
        'tip_y_arr': tip_y_arr,
        'freqs': freqs,
        'fft_mag_x': fft_mag_x,
        'annotated_frame': annotated_frame,
    }
