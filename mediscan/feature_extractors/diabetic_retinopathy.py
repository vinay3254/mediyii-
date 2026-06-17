"""
diabetic_retinopathy.py — Microaneurysm & hemorrhage detector for DR grading.

Clinical basis:
  Diabetic Retinopathy (DR) is graded 0-4 by the ETDRS (Early Treatment
  Diabetic Retinopathy Study) scale based on lesion count:
    Grade 0 — No DR (0 microaneurysms)
    Grade 1 — Mild NPDR (<5 microaneurysms)
    Grade 2 — Moderate NPDR (5-14 MA)
    Grade 3 — Severe NPDR (15-29 MA)
    Grade 4 — Proliferative DR (≥30 MA or neovascularisation)

Pipeline: Green channel → CLAHE → Black-hat morphology → SimpleBlobDetector
Dataset: Kaggle DR Detection — kaggle.com/competitions/diabetic-retinopathy-detection
         DRIVE — grand-challenge.org/databases/drive/
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    'ma_count',
    'he_count',
    'lesion_density',
    'green_channel_mean',
    'clahe_mean',
]

DR_GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'Proliferative DR',
}


def grade_dr(ma_count: int) -> int:
    """Map microaneurysm blob count → ETDRS DR grade 0-4."""
    if ma_count == 0:   return 0
    if ma_count < 5:    return 1
    if ma_count < 15:   return 2
    if ma_count < 30:   return 3
    return 4


def _make_blob_detector(min_area: float, max_area: float,
                        min_circ: float = 0.5) -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circ
    params.filterByColor = True
    params.blobColor = 0          # dark blobs on bright background
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


def detect_dr_lesions(img_bgr: np.ndarray) -> dict:
    """
    Detect microaneurysms and hemorrhages in a retinal fundus image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR fundus image.

    Returns
    -------
    dict with feature values + 'annotated_img' + 'tophat_img'.
    """
    # ── 1. Pre-process ───────────────────────────────────────────────────────
    h0, w0 = img_bgr.shape[:2]
    scale = 512 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]
    total_pixels = h * w

    # ── 2. Green channel extraction ──────────────────────────────────────────
    # CLINICAL NOTE: The green channel provides the best contrast for retinal
    # vasculature because haemoglobin has high absorption near 540 nm (green).
    green = img[:, :, 1]
    green_channel_mean = float(green.mean())

    # ── 3. CLAHE — local contrast enhancement ────────────────────────────────
    # CLINICAL NOTE: Contrast Limited Adaptive Histogram Equalisation compensates
    # for the uneven illumination typical of fundus photos (brighter at centre).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    clahe_mean = float(enhanced.mean())

    # ── 4. Black-hat morphology — isolate dark spots ─────────────────────────
    # CLINICAL NOTE: Black-hat = (closing - original). It isolates structures
    # that are smaller than the structuring element and darker than surroundings.
    # Microaneurysms (5-200 µm) are perfectly captured with a 15px ellipse kernel.
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_bh)

    # Normalise tophat for blob detection
    tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

    # ── 5. Blob detection — microaneurysms (small, circular, dark) ───────────
    ma_detector = _make_blob_detector(min_area=5, max_area=200, min_circ=0.5)
    ma_keypoints = ma_detector.detect(tophat_norm)
    ma_count = len(ma_keypoints)

    # ── 6. Blob detection — hemorrhages (larger, less circular) ──────────────
    # CLINICAL NOTE: Flame-shaped haemorrhages are 200-2000 µm, less circular.
    he_detector = _make_blob_detector(min_area=200, max_area=2000, min_circ=0.3)
    he_keypoints = he_detector.detect(tophat_norm)
    he_count = len(he_keypoints)

    lesion_density = float((ma_count + he_count) / total_pixels * 10000)  # per 10k px

    # ── 7. DR grading ────────────────────────────────────────────────────────
    dr_grade = grade_dr(ma_count)

    # ── 8. Build annotated image ─────────────────────────────────────────────
    annotated = img.copy()
    # Yellow circles = microaneurysms
    for kp in ma_keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(4, int(kp.size / 2))
        cv2.circle(annotated, (cx, cy), r, (0, 255, 255), 1)

    # Red circles = hemorrhages
    for kp in he_keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(6, int(kp.size / 2))
        cv2.circle(annotated, (cx, cy), r, (0, 0, 255), 2)

    grade_colors = {0: (0, 200, 0), 1: (0, 200, 180),
                    2: (0, 165, 255), 3: (0, 80, 255), 4: (0, 0, 255)}
    cv2.putText(annotated,
                f'Grade {dr_grade}: {DR_GRADE_LABELS[dr_grade]}  MA:{ma_count} HE:{he_count}',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                grade_colors[dr_grade], 2)
    cv2.putText(annotated, 'Yellow=MA  Red=Hemorrhage',
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Tophat as RGB for display
    tophat_rgb = cv2.cvtColor(tophat_norm, cv2.COLOR_GRAY2BGR)

    return {
        'ma_count': ma_count,
        'he_count': he_count,
        'lesion_density': lesion_density,
        'green_channel_mean': green_channel_mean,
        'clahe_mean': clahe_mean,
        'dr_grade': dr_grade,
        'dr_grade_label': DR_GRADE_LABELS[dr_grade],
        'annotated_img': annotated,
        'tophat_img': tophat_rgb,
        'enhanced_img': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
