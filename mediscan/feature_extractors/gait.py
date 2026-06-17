"""
gait.py
-------
Gait symmetry analysis using the Robinson (1987) symmetry index.

Clinical background
-------------------
Gait asymmetry is a sensitive early marker for a wide range of neurological
and orthopaedic conditions, including stroke, hip osteoarthritis, and
Parkinson's disease.  The symmetry index (SI) quantifies how much the left
and right limb amplitudes diverge from perfect bilateral symmetry.

Robinson symmetry index (Robinson 1987)
   SI = |L - R| / ((L + R) / 2) × 100

Interpretation
   SI < 10%  : normal / symmetric gait
   10-20%    : compensatory gait (e.g. mild hip OA, early stroke)
   > 20%     : significant impairment

Reference
---------
- Robinson RO, Herzog W, Nigg BM (1987) Use of force platform variables to
  quantify the effects of chiropractic manipulation on gait symmetry.
  J Manipulative Physiol Ther 10(4):172-176.
- Perry J, Burnfield JM (2010) Gait Analysis: Normal and Pathological Function.
  SLACK Incorporated.
"""

import cv2
import numpy as np
from scipy import signal as scipy_signal
import mediapipe as mp

# ---------------------------------------------------------------------------
# Public feature list – consumed by the ML pipeline
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'symmetry_index',
    'L_amp',
    'R_amp',
    'cadence_steps_per_min',
]


# ---------------------------------------------------------------------------
# Symmetry index helper
# ---------------------------------------------------------------------------

def compute_symmetry_index(L_amp: float, R_amp: float) -> float:
    """
    Compute the Robinson (1987) symmetry index.

    Parameters
    ----------
    L_amp : float
        Vertical oscillation amplitude of the left knee landmark.
    R_amp : float
        Vertical oscillation amplitude of the right knee landmark.

    Returns
    -------
    float
        Symmetry index in percent.

    Clinical Notes
    --------------
    SI < 10%  -> normal gait; symmetric bilateral loading
    SI 10-20% -> compensatory gait; early impairment on one side
    SI > 20%  -> significant impairment; clinically relevant asymmetry
    """
    # CLINICAL NOTE: SI < 10% = normal gait; >10% = compensatory gait; >20% = significant impairment
    SI = abs(L_amp - R_amp) / ((L_amp + R_amp) / 2.0 + 1e-9) * 100.0
    return float(SI)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_gait_video(video_path: str) -> dict:
    """
    Analyse a walking video and extract gait symmetry and cadence metrics.

    The function tracks the vertical (Y-axis) position of both knees
    (MediaPipe Pose landmarks 25 = LEFT_KNEE, 26 = RIGHT_KNEE) and ankles
    (27 = LEFT_ANKLE, 28 = RIGHT_ANKLE) across all frames.  Knee vertical
    oscillation encodes the step cycle: each complete oscillation from trough
    to trough represents one gait cycle (two steps).

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    dict
        Keys
        ----
        symmetry_index        : float  - Robinson SI in percent
        L_amp                 : float  - left-knee oscillation amplitude (norm units)
        R_amp                 : float  - right-knee oscillation amplitude (norm units)
        gait_flag             : bool   - True if SI > 10 (abnormal symmetry)
        cadence_steps_per_min : float  - estimated cadence
        duration_sec          : float  - video duration in seconds
        fps                   : float  - actual video frame rate
        left_knee_signal      : np.ndarray - normalised Y positions, left knee
        right_knee_signal     : np.ndarray - normalised Y positions, right knee
        timestamps            : np.ndarray - time in seconds for each sample
        annotated_frame       : np.ndarray - last BGR frame with pose overlay
    """

    # ------------------------------------------------------------------
    # 1. Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # ------------------------------------------------------------------
    # 2. Initialise MediaPipe Pose
    #    Uses the full-body model; min_detection_confidence=0.5 keeps
    #    false-positive rate low while maintaining good sensitivity.
    # ------------------------------------------------------------------
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    left_knee_y = []    # landmark 25 Y (normalised)
    right_knee_y = []   # landmark 26 Y (normalised)
    left_ankle_y = []   # landmark 27 Y (for step-count cross-check)
    right_ankle_y = []  # landmark 28 Y
    timestamps = []
    frame_idx = 0
    annotated_frame = None

    # ------------------------------------------------------------------
    # 3. Frame loop
    # ------------------------------------------------------------------
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(frame_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # CLINICAL NOTE: Knee vertical oscillation encodes step cycle
            # timing and amplitude
            left_knee_y.append(lm[25].y)
            right_knee_y.append(lm[26].y)
            left_ankle_y.append(lm[27].y)
            right_ankle_y.append(lm[28].y)
            timestamps.append(frame_idx / fps)

            # Draw full pose skeleton
            mp_drawing.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # Highlight left knee in BLUE (BGR: 255,0,0)
            h, w = frame_bgr.shape[:2]
            lk_x = int(lm[25].x * w)
            lk_y = int(lm[25].y * h)
            cv2.circle(frame_bgr, (lk_x, lk_y), 10, (255, 0, 0), -1)

            # Highlight right knee in RED (BGR: 0,0,255)
            rk_x = int(lm[26].x * w)
            rk_y = int(lm[26].y * h)
            cv2.circle(frame_bgr, (rk_x, rk_y), 10, (0, 0, 255), -1)

            annotated_frame = frame_bgr.copy()

        frame_idx += 1

    cap.release()
    pose_detector.close()

    # ------------------------------------------------------------------
    # 4. Guard against empty detection
    # ------------------------------------------------------------------
    if len(left_knee_y) < 4:
        return {
            'symmetry_index': 0.0,
            'L_amp': 0.0,
            'R_amp': 0.0,
            'gait_flag': False,
            'cadence_steps_per_min': 0.0,
            'duration_sec': frame_idx / fps,
            'fps': fps,
            'left_knee_signal': np.array(left_knee_y),
            'right_knee_signal': np.array(right_knee_y),
            'timestamps': np.array(timestamps),
            'annotated_frame': annotated_frame,
        }

    left_knee_signal = np.array(left_knee_y)
    right_knee_signal = np.array(right_knee_y)
    timestamps_arr = np.array(timestamps)
    duration_sec = float(timestamps_arr[-1]) if len(timestamps_arr) > 0 else frame_idx / fps

    # ------------------------------------------------------------------
    # 5. Amplitude computation
    #    Amplitude = peak-to-trough excursion of the knee Y coordinate.
    #    In MediaPipe normalised coords Y increases downward, so the signal
    #    oscillates symmetrically with walking.
    # ------------------------------------------------------------------
    L_amp = float(left_knee_signal.max() - left_knee_signal.min())
    R_amp = float(right_knee_signal.max() - right_knee_signal.min())

    symmetry_index = compute_symmetry_index(L_amp, R_amp)

    # ------------------------------------------------------------------
    # 6. Cadence estimation via peak detection
    #    Each peak in the left-knee Y signal corresponds to the maximum
    #    knee flexion of the LEFT leg (one step).  Two such events
    #    (left + right) = one full gait cycle = 2 steps.
    #    Therefore: steps = 2 * number_of_peaks_in_left_signal.
    #
    # CLINICAL NOTE: Normal cadence is 100-120 steps/min for healthy adults
    # ------------------------------------------------------------------
    # Minimum peak distance = fps / 3 (at most 3 steps per second = 180/min)
    min_peak_distance = max(1, int(fps / 3))

    peaks, _ = scipy_signal.find_peaks(
        left_knee_signal,
        distance=min_peak_distance,
        prominence=0.005,   # ignore micro-oscillations < 0.5% of frame height
    )

    n_steps = len(peaks) * 2   # multiply by 2 to count both left and right steps
    duration_min = duration_sec / 60.0 + 1e-9
    cadence_steps_per_min = float(n_steps / duration_min)

    # ------------------------------------------------------------------
    # 7. Gait flag
    # ------------------------------------------------------------------
    gait_flag = symmetry_index > 10.0

    # ------------------------------------------------------------------
    # 8. Return results
    # ------------------------------------------------------------------
    return {
        'symmetry_index': symmetry_index,
        'L_amp': L_amp,
        'R_amp': R_amp,
        'gait_flag': gait_flag,
        'cadence_steps_per_min': cadence_steps_per_min,
        'duration_sec': duration_sec,
        'fps': fps,
        'left_knee_signal': left_knee_signal,
        'right_knee_signal': right_knee_signal,
        'timestamps': timestamps_arr,
        'annotated_frame': annotated_frame,
    }
