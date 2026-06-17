"""
pill_identifier.py — Pill shape + colour feature extraction with KNN matching.

Clinical basis:
  Medication errors are a leading cause of preventable harm. A pill identifier
  lets pharmacists/nurses verify tablets by photographing them. We extract:
    - Shape: aspect ratio, circularity, convexity (round vs oval vs capsule)
    - Colour: dominant RGB via k-means (k=3) on pixel array
  Then we match against a reference database with Euclidean distance / KNN.
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    'aspect_ratio', 'circularity', 'convexity',
    'dominant_r', 'dominant_g', 'dominant_b',
]

PILL_REFERENCE_DB = [
    {'name': 'Aspirin 81mg',       'aspect_ratio': 1.0,  'circularity': 0.95, 'convexity': 0.98, 'dominant_r': 240, 'dominant_g': 240, 'dominant_b': 240},
    {'name': 'Aspirin 325mg',      'aspect_ratio': 1.0,  'circularity': 0.93, 'convexity': 0.97, 'dominant_r': 245, 'dominant_g': 245, 'dominant_b': 245},
    {'name': 'Metformin 500mg',    'aspect_ratio': 1.9,  'circularity': 0.65, 'convexity': 0.94, 'dominant_r': 245, 'dominant_g': 245, 'dominant_b': 248},
    {'name': 'Metformin 1000mg',   'aspect_ratio': 2.2,  'circularity': 0.60, 'convexity': 0.93, 'dominant_r': 248, 'dominant_g': 248, 'dominant_b': 250},
    {'name': 'Lisinopril 10mg',    'aspect_ratio': 1.05, 'circularity': 0.92, 'convexity': 0.96, 'dominant_r': 220, 'dominant_g': 180, 'dominant_b': 190},
    {'name': 'Lisinopril 20mg',    'aspect_ratio': 1.08, 'circularity': 0.90, 'convexity': 0.95, 'dominant_r': 210, 'dominant_g': 170, 'dominant_b': 180},
    {'name': 'Atorvastatin 20mg',  'aspect_ratio': 1.7,  'circularity': 0.68, 'convexity': 0.95, 'dominant_r': 245, 'dominant_g': 235, 'dominant_b': 200},
    {'name': 'Amlodipine 5mg',     'aspect_ratio': 1.0,  'circularity': 0.90, 'convexity': 0.97, 'dominant_r': 245, 'dominant_g': 245, 'dominant_b': 245},
    {'name': 'Omeprazole 20mg',    'aspect_ratio': 2.4,  'circularity': 0.55, 'convexity': 0.90, 'dominant_r': 148, 'dominant_g': 48,  'dominant_b': 185},
    {'name': 'Metoprolol 50mg',    'aspect_ratio': 1.6,  'circularity': 0.70, 'convexity': 0.94, 'dominant_r': 245, 'dominant_g': 230, 'dominant_b': 200},
    {'name': 'Amoxicillin 500mg',  'aspect_ratio': 2.5,  'circularity': 0.52, 'convexity': 0.89, 'dominant_r': 255, 'dominant_g': 200, 'dominant_b': 50},
    {'name': 'Ibuprofen 200mg',    'aspect_ratio': 1.0,  'circularity': 0.91, 'convexity': 0.96, 'dominant_r': 210, 'dominant_g': 80,  'dominant_b': 80},
    {'name': 'Paracetamol 500mg',  'aspect_ratio': 1.8,  'circularity': 0.66, 'convexity': 0.93, 'dominant_r': 248, 'dominant_g': 248, 'dominant_b': 248},
]


def extract_pill_features(img_bgr: np.ndarray) -> dict:
    """
    Extract morphological and colour features from a pill photo.

    Best results: white/grey background, single pill centred, good lighting.
    """
    h0, w0 = img_bgr.shape[:2]
    scale = 512 / max(h0, w0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold — pill vs background
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("[pill] No contour found — use plain background.")

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # CLINICAL NOTE: Capsules AR>2.0; oval tablets 1.5-2.5; round ~1.0
    aspect_ratio = float(max(bw, bh)) / (min(bw, bh) + 1e-9)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True) + 1e-9
    # CLINICAL NOTE: Round pills ~0.9+; ovals ~0.6-0.8; capsules ~0.5
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-9
    convexity = area / hull_area

    # ── Dominant colour via k-means (k=3) ────────────────────────────────────
    roi = img[y: y + bh, x: x + bw]
    pixels = roi.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    dominant_idx = np.bincount(labels.flatten()).argmax()
    dom_color = centers[dominant_idx]  # BGR
    dominant_r = float(dom_color[2])
    dominant_g = float(dom_color[1])
    dominant_b = float(dom_color[0])

    # ── Annotated image ──────────────────────────────────────────────────────
    annotated = img.copy()
    cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 128, 255), 1)
    # Colour swatch
    swatch = np.full((40, 40, 3), [dom_color[0], dom_color[1], dom_color[2]],
                     dtype=np.uint8)
    annotated[5:45, 5:45] = swatch
    cv2.putText(annotated, f'AR:{aspect_ratio:.2f} Circ:{circularity:.2f}',
                (8, annotated.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1)

    feat = {
        'aspect_ratio': round(aspect_ratio, 4),
        'circularity': round(circularity, 4),
        'convexity': round(convexity, 4),
        'dominant_r': round(dominant_r, 1),
        'dominant_g': round(dominant_g, 1),
        'dominant_b': round(dominant_b, 1),
        'annotated_img': annotated,
        'contour': cnt,
        'bbox': (x, y, bw, bh),
    }
    return feat


def match_pill(features: dict, top_k: int = 3) -> list:
    """
    Match extracted features to PILL_REFERENCE_DB via Euclidean distance.

    Returns list of top_k dicts: {name, distance, confidence}.
    """
    # Normalise features for fair comparison
    f_ar   = features['aspect_ratio'] / 3.0
    f_circ = features['circularity']
    f_conv = features['convexity']
    f_r    = features['dominant_r'] / 255.0
    f_g    = features['dominant_g'] / 255.0
    f_b    = features['dominant_b'] / 255.0
    query  = np.array([f_ar, f_circ, f_conv, f_r, f_g, f_b])

    dists = []
    for pill in PILL_REFERENCE_DB:
        ref = np.array([
            pill['aspect_ratio'] / 3.0,
            pill['circularity'],
            pill['convexity'],
            pill['dominant_r'] / 255.0,
            pill['dominant_g'] / 255.0,
            pill['dominant_b'] / 255.0,
        ])
        d = float(np.linalg.norm(query - ref))
        dists.append({'name': pill['name'], 'distance': d})

    dists.sort(key=lambda x: x['distance'])
    max_d = dists[-1]['distance'] + 1e-9

    results = []
    for entry in dists[:top_k]:
        confidence = max(0.0, 1.0 - entry['distance'] / max_d)
        results.append({
            'name': entry['name'],
            'distance': round(entry['distance'], 4),
            'confidence': round(confidence * 100, 1),
        })
    return results


def features_to_array(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)
