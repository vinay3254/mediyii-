"""
melanoma.py — ABCDE feature extractor for skin lesion melanoma classification.

Clinical basis: The ABCDE rule is a widely used mnemonic for melanoma detection:
  A = Asymmetry     (benign moles are symmetric)
  B = Border        (malignant lesions have irregular, notched borders)
  C = Color         (multiple colors or uneven pigmentation = suspicious)
  D = Diameter      (>6mm is a warning sign)
  E = Texture       (GLCM captures textural heterogeneity)

Dataset: ISIC Archive — https://www.isic-archive.com
Reference: Nachbar et al. (1994) ABCD rule of dermatoscopy.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ── Canonical feature order ──────────────────────────────────────────────────
FEATURE_NAMES = [
    'asymmetry',
    'border_score',
    'solidity',
    'color_std_L',
    'color_std_A',
    'color_std_B',
    'diameter',
    'glcm_contrast',
    'glcm_homogeneity',
]


def _resize_keep_aspect(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Resize so the longest side equals max_side, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_lesion_features(img_bgr: np.ndarray) -> dict | None:
    """
    Extract ABCDE clinical features from a skin lesion image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image (as returned by cv2.imread).

    Returns
    -------
    dict with keys matching FEATURE_NAMES, plus 'annotated_img'.
    Returns None if no lesion contour is found.
    """
    # ── 1. Pre-process ───────────────────────────────────────────────────────
    img = _resize_keep_aspect(img_bgr, 512)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # ── 2. Segment lesion via Otsu on HSV saturation channel ─────────────────
    # CLINICAL NOTE: Melanocytic lesions have higher colour saturation than
    # surrounding skin, making saturation-channel Otsu a robust segmentation.
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup — close small holes inside lesion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[melanoma] WARNING: No contour found — try a cleaner image.")
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 100:
        print("[melanoma] WARNING: Contour area too small — check segmentation.")
        return None

    # ── A — Asymmetry ────────────────────────────────────────────────────────
    # CLINICAL NOTE: Melanomas are typically asymmetric; benign moles are
    # symmetric. We compare top half vs mirrored bottom half of the lesion mask.
    x, y, bw, bh = cv2.boundingRect(cnt)
    roi_mask = mask[y:y + bh, x:x + bw]
    top_half = roi_mask[: bh // 2, :]
    bot_half = cv2.flip(roi_mask[bh // 2:, :], 0)
    min_h = min(top_half.shape[0], bot_half.shape[0])
    diff = cv2.absdiff(top_half[:min_h], bot_half[:min_h])
    asymmetry = float(diff.sum()) / (bw * bh * 255 + 1e-9)

    # ── B — Border irregularity ───────────────────────────────────────────────
    # CLINICAL NOTE: Perimeter² / (4π × area) equals 1.0 for a perfect circle.
    # Values > 1.5 suggest notched, irregular borders characteristic of melanoma.
    perimeter = cv2.arcLength(cnt, True)
    border_score = float(perimeter ** 2) / (4 * np.pi * area + 1e-9)

    # ── Solidity ─────────────────────────────────────────────────────────────
    # CLINICAL NOTE: Solidity (area / convex hull area) measures boundary shape.
    # Benign moles are usually compact and convex (solidity close to 1.0).
    # Melanomas often have irregular/concave borders, lowering solidity.
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / (hull_area + 1e-9)

    # ── C — Color variance ────────────────────────────────────────────────────
    # CLINICAL NOTE: Melanomas often contain multiple pigment types (brown, black,
    # red, white). High std in LAB channels quantifies this heterogeneity.
    lesion_pixels_lab = lab[mask > 0]  # all pixels inside mask
    if lesion_pixels_lab.shape[0] == 0:
        color_std = np.array([0.0, 0.0, 0.0])
    else:
        color_std = lesion_pixels_lab.std(axis=0).astype(float)  # [L_std, A_std, B_std]

    # ── D — Diameter ──────────────────────────────────────────────────────────
    # CLINICAL NOTE: Equivalent diameter assumes a circle of the same area.
    # > 60px at 512px resolution roughly corresponds to >6mm in clinical photos.
    diameter = float(np.sqrt(4 * area / np.pi))

    # ── E — Texture (GLCM) ───────────────────────────────────────────────────
    # CLINICAL NOTE: Malignant lesions show higher textural heterogeneity.
    # GLCM contrast measures local intensity variation; homogeneity inversely.
    gray_roi = gray[y:y + bh, x:x + bw]
    gray_roi = cv2.resize(gray_roi, (128, 128))  # normalise size for GLCM
    glcm = graycomatrix(gray_roi, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])

    # ── Build annotated image ────────────────────────────────────────────────
    annotated = img.copy()
    cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)          # lesion contour — green
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)  # bounding box — red
    # Asymmetry axes (blue)
    mid_x, mid_y = x + bw // 2, y + bh // 2
    cv2.line(annotated, (mid_x, y), (mid_x, y + bh), (255, 0, 0), 1)
    cv2.line(annotated, (x, mid_y), (x + bw, mid_y), (255, 0, 0), 1)

    risk_level = _risk_label(asymmetry, border_score, color_std[1])
    color_map = {'High': (0, 0, 255), 'Medium': (0, 165, 255), 'Low': (0, 200, 0)}
    cv2.putText(annotated, f'Risk: {risk_level}  A={asymmetry:.3f}  B={border_score:.2f}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[risk_level], 2)
    cv2.putText(annotated, f'D={diameter:.0f}px  Contrast={glcm_contrast:.0f}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    feat = {
        'asymmetry': asymmetry,
        'border_score': border_score,
        'solidity': solidity,
        'color_std_L': float(color_std[0]),
        'color_std_A': float(color_std[1]),
        'color_std_B': float(color_std[2]),
        'diameter': diameter,
        'glcm_contrast': glcm_contrast,
        'glcm_homogeneity': glcm_homogeneity,
        # extras not in ML feature vector but useful for display
        'annotated_img': annotated,
        'mask': mask,
        'contour': cnt,
        'risk_label': risk_level,
    }
    return feat


def _risk_label(asymmetry: float, border: float, color_a_std: float) -> str:
    """Simple rule-based risk label mirroring the decision tree logic."""
    score = 0
    if asymmetry > 0.45: score += 2
    elif asymmetry > 0.25: score += 1
    if border > 2.5: score += 2
    elif border > 1.5: score += 1
    if color_a_std > 18: score += 1
    if score >= 4: return 'High'
    if score >= 2: return 'Medium'
    return 'Low'


def features_to_array(feat_dict: dict) -> np.ndarray:
    """Return a flat numpy array in canonical FEATURE_NAMES order."""
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
