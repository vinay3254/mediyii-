"""
surgical_counter.py — Surgical instrument counter for RSI prevention.

Clinical basis:
  Retained Surgical Items (RSI) occur in ~1 in 5,500-18,760 operations
  and are a "never event" in modern surgical safety frameworks. Current
  prevention relies on manual counts; automated CV counting provides a
  safety net. We classify instrument shapes by contour geometry (sponges
  are circular, forceps are elongated, etc.) and maintain an inventory.

Reference: Gawande et al. (2003). Risk factors for retained instruments
  after surgery. NEJM 348:229-235.
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    'total_count',
    'large_round_count',
    'medium_elongated_count',
    'small_round_count',
    'irregular_count',
]

# Reference counts set at start of procedure via update_reference()
REFERENCE_COUNTS: dict = {
    'large_round': 0,
    'medium_elongated': 0,
    'small_round': 0,
    'irregular': 0,
}


def _classify_shape(area: float, circularity: float,
                    aspect_ratio: float) -> str:
    """
    Classify instrument contour by morphological features.

    CLINICAL NOTE:
      large_round   → surgical sponges (laparotomy pads), pledgets
      medium_elongated → scalpels, scissors, haemostats, forceps, retractors
      small_round   → needles, small sponges, bullet tips
      irregular     → complex instruments, partial occlusions
    """
    if circularity > 0.70 and area > 3000:
        return 'large_round'
    if circularity > 0.65 and area <= 3000:
        return 'small_round'
    if aspect_ratio > 3.0 or aspect_ratio < 0.33:
        return 'medium_elongated'
    return 'irregular'


def segment_instruments(img_bgr: np.ndarray,
                        min_area: int = 500) -> dict:
    """
    Detect and classify surgical instruments in a tray image.

    Parameters
    ----------
    img_bgr  : BGR tray image (overhead view, uniform background).
    min_area : Minimum contour area in pixels to consider.

    Returns
    -------
    dict with inventory, total_count, contour_features, mismatch, annotated_img.
    """
    # ── Pre-process ──────────────────────────────────────────────────────────
    h0, w0 = img_bgr.shape[:2]
    scale = 640 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]
    total_px = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold (inverted — instruments darker than bright tray)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close small gaps within instruments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # ── Classify each contour ────────────────────────────────────────────────
    inventory = {k: 0 for k in REFERENCE_COUNTS}
    contour_features = []

    TYPE_COLORS = {
        'large_round':      (0, 255, 255),   # yellow
        'medium_elongated': (255, 100, 0),   # blue
        'small_round':      (0, 255, 0),     # green
        'irregular':        (0, 128, 255),   # orange
    }

    annotated = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 0.3 * total_px:
            continue   # skip noise and background blob

        perimeter = cv2.arcLength(cnt, True) + 1e-9
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / (bh + 1e-9)
        extent = area / ((bw * bh) + 1e-9)

        shape_type = _classify_shape(area, circularity, aspect_ratio)
        inventory[shape_type] += 1

        contour_features.append({
            'shape_type': shape_type,
            'area': area,
            'circularity': round(circularity, 3),
            'aspect_ratio': round(aspect_ratio, 3),
            'extent': round(extent, 3),
            'bbox': (x, y, bw, bh),
        })

        color = TYPE_COLORS[shape_type]
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 1)
        cv2.putText(annotated, f'{shape_type[:4]}',
                    (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    total_count = sum(inventory.values())

    # ── Mismatch detection ───────────────────────────────────────────────────
    mismatch_details = check_mismatch(inventory)
    mismatch = len(mismatch_details) > 0

    # Summary overlay
    status_color = (0, 0, 255) if mismatch else (0, 200, 0)
    cv2.rectangle(annotated, (0, h - 30), (w, h), (20, 20, 20), -1)
    summary = f"Total:{total_count}  " + "  ".join(
        f"{k[:4]}:{v}" for k, v in inventory.items())
    cv2.putText(annotated, summary, (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
    if mismatch:
        cv2.putText(annotated, '!! COUNT MISMATCH !!',
                    (w // 2 - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    return {
        'total_count': total_count,
        'large_round_count': inventory['large_round'],
        'medium_elongated_count': inventory['medium_elongated'],
        'small_round_count': inventory['small_round'],
        'irregular_count': inventory['irregular'],
        'inventory': inventory,
        'contour_features': contour_features,
        'mismatch': mismatch,
        'mismatch_details': mismatch_details,
        'annotated_img': annotated,
    }


def update_reference(inventory: dict):
    """Record expected counts at the start of a procedure."""
    global REFERENCE_COUNTS
    REFERENCE_COUNTS.update(inventory)
    print(f"[surgical] Reference counts set: {REFERENCE_COUNTS}")


def check_mismatch(current: dict) -> list:
    """Return list of mismatch strings where current ≠ reference."""
    mismatches = []
    for k, ref_v in REFERENCE_COUNTS.items():
        cur_v = current.get(k, 0)
        if ref_v > 0 and cur_v != ref_v:
            mismatches.append(
                f"{k}: expected {ref_v}, found {cur_v} "
                f"({'MISSING' if cur_v < ref_v else 'EXTRA'})"
            )
    return mismatches


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
