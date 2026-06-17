"""
jaundice.py — Sclera/skin yellowness estimation via LAB b* channel.

Clinical basis:
  Jaundice (icterus) results from elevated bilirubin, which absorbs blue light
  and causes a yellow tint visible in sclera, skin, and nailbeds. In CIELAB
  colour space, the b* channel encodes the blue-yellow axis. Elevated b* in the
  sclera or skin ROI indicates bilirubin-induced yellowing.

Research basis: BiliCam — Smartphone-Based Newborn Jaundice Screening
  Hernandez et al., 2016. Ubiquitous Computing (UbiComp).
  University of Washington BiliScreen project.

Calibration note:
  Camera response and ambient lighting dramatically shift LAB values.
  Always include a neutral reference patch (white card) in frame.
  Threshold b* > 145 (0-255 scaled LAB) is approximate; calibrate per device.
"""

import cv2
import numpy as np

FEATURE_NAMES = ['b_star_mean', 'b_star_std', 'l_mean', 'a_mean', 'b_star_to_a_star_ratio']
JAUNDICE_THRESHOLD = 145   # b* (0-255 LAB scale) above this → flag


def normalize_illumination(lab_img: np.ndarray,
                           ref_coords: tuple | None) -> np.ndarray:
    """
    Shift LAB channels so reference patch reads (128, 128, 128) = neutral.

    Parameters
    ----------
    lab_img     : LAB image (uint8, channels 0-255).
    ref_coords  : (x, y, w, h) of a white/neutral patch in the image.
                  If None, returns image unchanged.
    """
    if ref_coords is None:
        print("[jaundice] WARNING: No calibration patch — results may vary per device.")
        return lab_img

    x, y, rw, rh = ref_coords
    patch = lab_img[y: y + rh, x: x + rw].astype(np.float32)
    ref_mean = patch.mean(axis=(0, 1))          # [L_mean, A_mean, B_mean]

    # CALIBRATION NOTE: We shift so reference patch = neutral (128,128,128)
    correction = (128.0 - ref_mean).reshape(1, 1, 3)
    out = lab_img.astype(np.float32) + correction
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_jaundice_features(img_bgr: np.ndarray,
                               roi_coords: tuple | None = None,
                               ref_coords: tuple | None = None) -> dict:
    """
    Estimate jaundice risk from a photo of sclera or skin.

    Parameters
    ----------
    img_bgr    : BGR image.
    roi_coords : (x, y, w, h) of sclera/skin region of interest.
                 If None, uses the centre 20% of the image.
    ref_coords : (x, y, w, h) of a neutral white calibration patch.

    Returns
    -------
    dict with feature values + 'annotated_img' + 'b_channel_img'.
    """
    h0, w0 = img_bgr.shape[:2]
    scale = 512 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]

    # ── Colour space conversion ──────────────────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ── Illumination normalisation ───────────────────────────────────────────
    calibrated = ref_coords is not None
    if calibrated:
        # Scale ref_coords to resized image
        rx, ry, rw, rh = ref_coords
        rx2 = int(rx * scale); ry2 = int(ry * scale)
        rw2 = int(rw * scale); rh2 = int(rh * scale)
        lab = normalize_illumination(lab, (rx2, ry2, rw2, rh2))

    # ── ROI selection ────────────────────────────────────────────────────────
    if roi_coords is not None:
        rx, ry, rw, rh = roi_coords
        x1 = int(rx * scale); y1 = int(ry * scale)
        x2 = int((rx + rw) * scale); y2 = int((ry + rh) * scale)
    else:
        # Default: central 20% of image
        mx, my = w // 2, h // 2
        bw, bh = w // 5, h // 5
        x1, y1, x2, y2 = mx - bw // 2, my - bh // 2, mx + bw // 2, my + bh // 2

    roi = lab[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"[jaundice] Empty ROI: check coordinates. img shape={img.shape}")

    # ── Feature extraction ───────────────────────────────────────────────────
    # CLINICAL NOTE: LAB b* channel (index 2) encodes blue-yellow axis.
    # Higher b* = more yellow = higher bilirubin. Normal sclera b*~120-135;
    # jaundiced sclera b*>145 (BiliCam threshold, device-dependent).
    b_star = roi[:, :, 2].astype(float)
    b_star_mean = float(b_star.mean())
    b_star_std  = float(b_star.std())
    l_mean = float(roi[:, :, 0].mean())
    a_mean = float(roi[:, :, 1].mean())
    b_star_to_a_star_ratio = b_star_mean / (a_mean + 1e-9)

    jaundice_flag = b_star_mean > JAUNDICE_THRESHOLD

    # ── Annotated image ──────────────────────────────────────────────────────
    annotated = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    box_color = (0, 0, 255) if jaundice_flag else (0, 200, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
    label = f"b*={b_star_mean:.1f} {'JAUNDICED' if jaundice_flag else 'Normal'}"
    cv2.putText(annotated, label, (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    if calibrated:
        cv2.putText(annotated, 'CAL', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

    # b* channel as grayscale for visualisation
    b_channel = lab[:, :, 2]
    b_channel_img = cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR)

    return {
        'b_star_mean': b_star_mean,
        'b_star_std': b_star_std,
        'l_mean': l_mean,
        'a_mean': a_mean,
        'b_star_to_a_star_ratio': b_star_to_a_star_ratio,
        'jaundice_flag': jaundice_flag,
        'calibrated': calibrated,
        'annotated_img': annotated,
        'b_channel_img': b_channel_img,
        'roi_coords_scaled': (x1, y1, x2 - x1, y2 - y1),
    }


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
