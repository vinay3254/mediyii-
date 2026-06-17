"""
anemia.py — Conjunctival pallor-based anemia screening.

Clinical basis:
  Anemia (low haemoglobin) reduces the redness of the conjunctiva (inner eyelid).
  The conjunctiva colour can be assessed non-invasively using a smartphone camera.
  We measure pallor using the LAB colour space:
    High L* (lightness) + Low a* (red-green axis) = pale = anaemic

Research:
  - Mannino et al. (2016). "Non-invasive haemoglobin measurement using a
    smartphone camera." PLOS ONE.
  - Eyenuk / HemaApp projects (UW, 2016). Annals of Internal Medicine.

Calibration: Same white-card approach as jaundice.py — mandatory for accuracy.
"""

import cv2
import numpy as np

FEATURE_NAMES = ['l_mean', 'a_mean', 'b_mean', 'pallor_index', 'redness_ratio', 'redness_std']
PALLOR_THRESHOLD = 6.5   # pallor_index above this → flag possible anaemia


def extract_anemia_features(img_bgr: np.ndarray,
                             roi_coords: tuple | None = None,
                             ref_coords: tuple | None = None) -> dict:
    """
    Measure conjunctival pallor to screen for anaemia.

    Parameters
    ----------
    img_bgr    : BGR image, ideally a close-up of the inner lower eyelid.
    roi_coords : (x, y, w, h) of the conjunctiva region (inner eyelid).
                 If None, uses centre 30% of image as proxy.
    ref_coords : (x, y, w, h) of a neutral white calibration patch.

    Returns
    -------
    dict with feature values + 'annotated_img'.
    """
    h0, w0 = img_bgr.shape[:2]
    scale = 512 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    calibrated = ref_coords is not None

    # ── Illumination calibration ─────────────────────────────────────────────
    if calibrated:
        rx, ry, rw, rh = ref_coords
        rx2 = int(rx * scale); ry2 = int(ry * scale)
        rw2 = max(1, int(rw * scale)); rh2 = max(1, int(rh * scale))
        patch = lab[ry2: ry2 + rh2, rx2: rx2 + rw2].astype(np.float32)
        ref_mean = patch.mean(axis=(0, 1))
        correction = (128.0 - ref_mean).reshape(1, 1, 3)
        lab = np.clip(lab.astype(np.float32) + correction, 0, 255).astype(np.uint8)

    # ── ROI selection ────────────────────────────────────────────────────────
    if roi_coords is not None:
        rx, ry, rw2, rh2 = roi_coords
        x1 = int(rx * scale); y1 = int(ry * scale)
        x2 = int((rx + rw2) * scale); y2 = int((ry + rh2) * scale)
    else:
        # Use centre 30% as conjunctiva proxy
        bw, bh = int(w * 0.3), int(h * 0.3)
        x1 = (w - bw) // 2; y1 = (h - bh) // 2
        x2 = x1 + bw; y2 = y1 + bh

    roi_lab = lab[y1:y2, x1:x2]
    roi_bgr = img[y1:y2, x1:x2]

    if roi_lab.size == 0:
        raise ValueError("[anemia] Empty ROI — check coordinates.")

    # ── Feature extraction ───────────────────────────────────────────────────
    # CLINICAL NOTE: Anaemic conjunctiva is pale (high L*) and less red (low a*).
    l_mean = float(roi_lab[:, :, 0].mean())   # Lightness
    a_mean = float(roi_lab[:, :, 1].mean())   # Red-Green axis; lower = paler
    b_mean = float(roi_lab[:, :, 2].mean())   # Yellow-Blue axis

    # CLINICAL NOTE: Pallor Index = L* / a*. High L + low a = pale = anaemic.
    # Threshold 6.5 derived from studies correlating with Hb < 11 g/dL.
    pallor_index = l_mean / (a_mean + 1e-9)

    # RGB redness ratio — complementary feature
    r_mean = float(roi_bgr[:, :, 2].mean())   # BGR so index 2 = Red
    g_mean = float(roi_bgr[:, :, 1].mean())
    # CLINICAL NOTE: Lower redness_ratio = less haemoglobin saturation visible.
    redness_ratio = r_mean / (g_mean + 1e-9)

    # Compute standard deviation of pixel-wise red-to-green ratio to represent vascular heterogeneity
    pixel_r = roi_bgr[:, :, 2].astype(float)
    pixel_g = roi_bgr[:, :, 1].astype(float)
    redness_map = pixel_r / (pixel_g + 1e-9)
    redness_std = float(redness_map.std())

    anemia_flag = pallor_index > PALLOR_THRESHOLD

    # ── Annotated image ──────────────────────────────────────────────────────
    annotated = img.copy()
    box_color = (0, 0, 255) if anemia_flag else (0, 200, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(annotated,
                f"Pallor={pallor_index:.2f} {'ANAEMIA?' if anemia_flag else 'Normal'}",
                (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
    cv2.putText(annotated, f"L*={l_mean:.0f}  a*={a_mean:.0f}  redness={redness_ratio:.2f}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return {
        'l_mean': l_mean,
        'a_mean': a_mean,
        'b_mean': b_mean,
        'pallor_index': pallor_index,
        'redness_ratio': redness_ratio,
        'redness_std': redness_std,
        'anemia_flag': anemia_flag,
        'calibrated': calibrated,
        'annotated_img': annotated,
        'roi_coords_scaled': (x1, y1, x2 - x1, y2 - y1),
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
