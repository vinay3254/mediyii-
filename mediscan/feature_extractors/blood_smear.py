"""
blood_smear.py — RBC/WBC/platelet segmentation from peripheral blood smear images.

Clinical basis:
  A peripheral blood smear (PBS) stained with Giemsa or Wright stain shows:
    RBCs  — pink/orange biconcave discs (H:0-20° in HSV, area 200-5000px²)
    WBCs  — purple/blue nucleated cells (H:120-160° in HSV, area 500-20000px²)
    Platelets — tiny pale purple fragments (area 20-200px²)

  Abnormal RBC:WBC ratio, unusual morphology, or blast cells indicate pathology.

Dataset: BCCD (Blood Cell Count and Detection) — Kaggle
  kaggle.com/datasets/draaslan/blood-cell-count-and-detection
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    'rbc_count', 'wbc_count', 'platelet_count', 'wbc_rbc_ratio', 'platelet_to_rbc_ratio',
]


def count_blood_cells(img_bgr: np.ndarray) -> dict:
    """
    Segment and count RBCs, WBCs, and platelets from a blood smear image.

    Parameters
    ----------
    img_bgr : np.ndarray — BGR microscope image of Giemsa-stained blood smear.

    Returns
    -------
    dict with counts, ratios, annotated_img, and intermediate masks.
    """
    # ── Pre-process ──────────────────────────────────────────────────────────
    h0, w0 = img_bgr.shape[:2]
    scale = 640 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    morph_kernel = np.ones((5, 5), np.uint8)

    # ── RBC Detection (pink/orange hue) ──────────────────────────────────────
    # CLINICAL NOTE: Giemsa-stained RBCs appear pink to salmon-orange.
    # Hue 0-20° covers red-to-orange; also include upper red wrap (160-180°).
    rbc_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]),
                                  np.array([20, 255, 255]))
    rbc_mask2 = cv2.inRange(hsv, np.array([160, 50, 50]),
                                  np.array([180, 255, 255]))
    rbc_mask = cv2.bitwise_or(rbc_mask1, rbc_mask2)
    rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_OPEN, morph_kernel)
    rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_CLOSE, morph_kernel)

    rbc_cnts, _ = cv2.findContours(rbc_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # CLINICAL NOTE: RBCs are 6-8 µm; at 40x magnification ~200-5000px² in image
    rbc_filtered = [c for c in rbc_cnts
                    if 200 < cv2.contourArea(c) < 5000]
    rbc_count = len(rbc_filtered)
    rbc_mean_area = (float(np.mean([cv2.contourArea(c) for c in rbc_filtered]))
                     if rbc_filtered else 0.0)

    # ── WBC Detection (purple/violet nuclei) ─────────────────────────────────
    # CLINICAL NOTE: WBC nuclei stain purple/violet with Giemsa (H:120-160°).
    # WBCs are 10-15 µm; much larger than RBCs.
    wbc_mask = cv2.inRange(hsv, np.array([120, 50, 50]),
                                 np.array([160, 255, 255]))
    wbc_mask = cv2.morphologyEx(wbc_mask, cv2.MORPH_OPEN, morph_kernel)
    wbc_mask = cv2.morphologyEx(wbc_mask, cv2.MORPH_CLOSE, morph_kernel)

    wbc_cnts, _ = cv2.findContours(wbc_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # CLINICAL NOTE: WBC area 500-20000px² at typical magnification
    wbc_filtered = [c for c in wbc_cnts
                    if 500 < cv2.contourArea(c) < 20000]
    wbc_count = len(wbc_filtered)
    wbc_mean_area = (float(np.mean([cv2.contourArea(c) for c in wbc_filtered]))
                     if wbc_filtered else 0.0)

    # ── Platelet Detection (tiny pale fragments) ──────────────────────────────
    # CLINICAL NOTE: Platelets (thrombocytes) are 2-3 µm; appear as tiny pale
    # purple/pink dots. Lower saturation and smaller area than RBCs.
    plt_mask = cv2.inRange(hsv, np.array([0, 20, 30]),
                                np.array([30, 150, 200]))
    plt_mask = cv2.morphologyEx(plt_mask, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))
    plt_cnts, _ = cv2.findContours(plt_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    plt_filtered = [c for c in plt_cnts
                    if 20 < cv2.contourArea(c) < 200]
    platelet_count = len(plt_filtered)

    # ── Clinical ratios ───────────────────────────────────────────────────────
    # Normal RBC:WBC ratio is ~500:1 (500 RBCs per WBC)
    wbc_rbc_ratio = wbc_count / (rbc_count + 1e-9)
    platelet_to_rbc_ratio = platelet_count / (rbc_count + 1e-9)
    # CLINICAL NOTE: Elevated WBC (leukocytosis) >10,500/µL suggests infection/inflammation
    high_wbc_flag = wbc_count > 15  # relative to field count

    # ── Build annotated image ─────────────────────────────────────────────────
    annotated = img.copy()
    # RBCs — green outlines
    cv2.drawContours(annotated, rbc_filtered, -1, (0, 220, 0), 1)
    # WBCs — blue outlines with label
    for c in wbc_filtered:
        cv2.drawContours(annotated, [c], -1, (255, 80, 0), 2)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(annotated, 'WBC', (cx - 15, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)
    # Platelets — yellow dots
    for c in plt_filtered:
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

    # Count overlay
    h_ann = annotated.shape[0]
    cv2.rectangle(annotated, (0, h_ann - 28), (annotated.shape[1], h_ann),
                  (10, 10, 10), -1)
    cv2.putText(annotated,
                f'RBC:{rbc_count}  WBC:{wbc_count}  PLT:{platelet_count}  '
                f'Ratio:{wbc_rbc_ratio:.4f}  {"HIGH WBC!" if high_wbc_flag else "Normal"}',
                (5, h_ann - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 200) if high_wbc_flag else (0, 200, 0), 1)

    return {
        'rbc_count': rbc_count,
        'wbc_count': wbc_count,
        'platelet_count': platelet_count,
        'wbc_rbc_ratio': round(wbc_rbc_ratio, 5),
        'platelet_to_rbc_ratio': round(platelet_to_rbc_ratio, 5),
        'rbc_mean_area': round(rbc_mean_area, 1),
        'wbc_mean_area': round(wbc_mean_area, 1),
        'high_wbc_flag': high_wbc_flag,
        'annotated_img': annotated,
        'rbc_mask': rbc_mask,
        'wbc_mask': wbc_mask,
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
