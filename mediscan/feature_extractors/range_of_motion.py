"""
range_of_motion.py — Joint angle measurement for physical therapy ROM tracking.

Clinical basis:
  Range of Motion (ROM) is a key physiotherapy metric. Full ROM means a joint
  can move through its expected angular range. Post-surgery or injury patients
  progressively recover ROM. We track peak joint angles across video frames
  using MediaPipe Pose landmarks and compare to clinical target angles.

  joint_angle uses the law of cosines on vectors from joint to adjacent segments.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _mp_available = True
except ImportError:
    _mp_available = False

FEATURE_NAMES = [
    'shoulder_flexion',
    'knee_flexion',
    'elbow_extension',
    'hip_flexion',
    'overall_rom_score',
]

TARGET_ANGLES = {
    'shoulder_flexion': 180.0,   # full forward flexion
    'knee_flexion':     135.0,   # typical max knee bend
    'elbow_extension':  180.0,   # full extension
    'hip_flexion':       90.0,   # standing forward flex
}


def joint_angle(a, b, c) -> float:
    """
    Compute the angle at joint b, formed by segments b→a and b→c.

    Parameters: a, b, c are mediapipe landmark objects (.x, .y normalised 0-1).
    Returns angle in degrees.

    CLINICAL NOTE: Peak ROM (max across frames) reflects true range; instantaneous
    values are noisy due to pose estimation jitter.
    """
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 0.0
    cos_a = np.dot(v1, v2) / norm
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def rom_score(measured: float, target: float) -> float:
    """ROM completion score 0-100%. min(100, measured/target*100)."""
    return min(100.0, (measured / (target + 1e-9)) * 100.0)


def _draw_angle(frame, lm, idx_a, idx_b, idx_c, angle: float,
                score: float, label: str, frame_w: int, frame_h: int):
    """Draw joint connection lines and angle label on frame."""
    def pt(i):
        return (int(lm[i].x * frame_w), int(lm[i].y * frame_h))

    color = (0, 200, 0) if score >= 80 else (0, 165, 255) if score >= 50 else (0, 0, 255)
    cv2.line(frame, pt(idx_a), pt(idx_b), color, 2)
    cv2.line(frame, pt(idx_b), pt(idx_c), color, 2)
    cv2.circle(frame, pt(idx_b), 6, color, -1)
    jx, jy = pt(idx_b)
    cv2.putText(frame, f'{label}:{angle:.0f}°',
                (jx + 8, jy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def analyze_rom_video(video_path: str) -> dict:
    """
    Measure peak joint angles in a physiotherapy exercise video.

    Returns angles dict, rom_scores dict, overall_score, annotated_frame.
    """
    if not _mp_available:
        raise RuntimeError("[rom] mediapipe required.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)  # CRITICAL: always read actual FPS
    if fps <= 0:
        fps = 30.0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Track maximum angle per joint across all frames
    max_angles = {k: 0.0 for k in TARGET_ANGLES}
    angle_history = {k: [] for k in TARGET_ANGLES}
    last_annotated = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # ── Joint angles (left side as primary) ──────────────────────────
            # CLINICAL NOTE: We use left-side landmarks as default; for right-side
            # exercises swap lm[11]↔lm[12], lm[13]↔lm[14], etc.

            # Shoulder flexion: hip (23) → shoulder (11) → elbow (13)
            sh = joint_angle(lm[23], lm[11], lm[13])

            # Elbow extension: shoulder (11) → elbow (13) → wrist (15)
            el = joint_angle(lm[11], lm[13], lm[15])

            # Knee flexion: hip (23) → knee (25) → ankle (27)
            kn = joint_angle(lm[23], lm[25], lm[27])

            # Hip flexion: shoulder (11) → hip (23) → knee (25)
            hi = joint_angle(lm[11], lm[23], lm[25])

            angles_frame = {
                'shoulder_flexion': sh,
                'elbow_extension': el,
                'knee_flexion': kn,
                'hip_flexion': hi,
            }

            for k, v in angles_frame.items():
                angle_history[k].append(v)
                if v > max_angles[k]:
                    max_angles[k] = v

            # Draw joints on frame
            scores_frame = {k: rom_score(angles_frame[k], TARGET_ANGLES[k])
                            for k in TARGET_ANGLES}
            _draw_angle(frame, lm, 23, 11, 13, sh, scores_frame['shoulder_flexion'], 'Sh', w, h)
            _draw_angle(frame, lm, 11, 13, 15, el, scores_frame['elbow_extension'],  'El', w, h)
            _draw_angle(frame, lm, 23, 25, 27, kn, scores_frame['knee_flexion'],     'Kn', w, h)
            _draw_angle(frame, lm, 11, 23, 25, hi, scores_frame['hip_flexion'],      'Hip', w, h)

        last_annotated = frame.copy()
        frame_idx += 1

    cap.release()
    pose.close()

    if all(v == 0.0 for v in max_angles.values()):
        raise ValueError("[rom] No pose detected in video.")

    scores = {k: rom_score(max_angles[k], TARGET_ANGLES[k]) for k in TARGET_ANGLES}
    overall = float(np.mean(list(scores.values())))

    # Annotate last frame with summary
    if last_annotated is not None:
        y = 25
        for k, sc in scores.items():
            color = (0, 200, 0) if sc >= 80 else (0, 165, 255) if sc >= 50 else (0, 0, 255)
            cv2.putText(last_annotated, f'{k}: {max_angles[k]:.0f}° ({sc:.0f}%)',
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
        cv2.putText(last_annotated, f'Overall ROM: {overall:.0f}%',
                    (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 0) if overall >= 80 else (0, 0, 255), 2)

    result = {**max_angles, 'overall_rom_score': overall}
    result.update({
        'rom_scores': scores,
        'angle_history': angle_history,
        'fps': fps,
        'frame_count': frame_idx,
        'annotated_frame': last_annotated,
    })
    return result


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
