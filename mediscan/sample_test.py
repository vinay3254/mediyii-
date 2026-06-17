"""
sample_test.py — Verify the MediScan pipeline with sample images/videos.

USAGE:
  1. Place sample images in the paths below (or update paths to your files).
  2. Run: python sample_test.py
  3. Annotated outputs are saved to output/ folder.

Fill in the three paths per tool before running.
"""

import os, sys, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.makedirs("output", exist_ok=True)

SEPARATOR = "=" * 65


def save_annotated(img, name: str):
    """Save annotated result image to output/."""
    if img is not None:
        path = f"output/{name}.jpg"
        cv2.imwrite(path, img)
        print(f"  → Saved: {path}")


def run_melanoma(image_paths: list):
    from feature_extractors.melanoma import extract_lesion_features, features_to_array
    print(f"\n{SEPARATOR}")
    print("TOOL 1: MELANOMA CLASSIFIER")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue
        img = cv2.imread(path)
        feats = extract_lesion_features(img)
        if feats is None:
            print(f"  [FAIL] No lesion detected in {path}")
            continue
        print(f"\n  Image: {path}")
        print(f"  Features: {', '.join(f'{k}={v:.4f}' for k,v in feats.items() if isinstance(v, float))}")
        print(f"  Risk Label: {feats['risk_label']}")
        arr = features_to_array(feats)
        print(f"  Feature vector: {arr.round(4)}")
        save_annotated(feats['annotated_img'], f"melanoma_{os.path.basename(path)}")


def run_diabetic_retinopathy(image_paths: list):
    from feature_extractors.diabetic_retinopathy import detect_dr_lesions
    print(f"\n{SEPARATOR}")
    print("TOOL 2: DIABETIC RETINOPATHY")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        res = detect_dr_lesions(img)
        print(f"\n  Image: {path}")
        print(f"  MA count: {res['ma_count']}, HE count: {res['he_count']}")
        print(f"  DR Grade: {res['dr_grade']} — {res['dr_grade_label']}")
        print(f"  Lesion density: {res['lesion_density']:.4f} per 10kpx")
        save_annotated(res['annotated_img'], f"dr_{os.path.basename(path)}")
        save_annotated(res['tophat_img'], f"dr_tophat_{os.path.basename(path)}")


def run_jaundice(image_paths: list):
    from feature_extractors.jaundice import extract_jaundice_features, JAUNDICE_THRESHOLD
    print(f"\n{SEPARATOR}")
    print("TOOL 3: JAUNDICE SCREENING")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        res = extract_jaundice_features(img)
        print(f"\n  Image: {path}")
        print(f"  b* Mean: {res['b_star_mean']:.1f} (threshold {JAUNDICE_THRESHOLD})")
        print(f"  L* Mean: {res['l_mean']:.1f}, a* Mean: {res['a_mean']:.1f}")
        flag = "JAUNDICED ⚠️" if res['jaundice_flag'] else "Normal ✅"
        print(f"  Prediction: {flag}")
        save_annotated(res['annotated_img'], f"jaundice_{os.path.basename(path)}")


def run_anemia(image_paths: list):
    from feature_extractors.anemia import extract_anemia_features, PALLOR_THRESHOLD
    print(f"\n{SEPARATOR}")
    print("TOOL 4: ANEMIA DETECTOR")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        res = extract_anemia_features(img)
        print(f"\n  Image: {path}")
        print(f"  Pallor Index: {res['pallor_index']:.3f} (threshold {PALLOR_THRESHOLD})")
        print(f"  L*={res['l_mean']:.1f}, a*={res['a_mean']:.1f}, redness={res['redness_ratio']:.2f}")
        flag = "ANEMIA? ⚠️" if res['anemia_flag'] else "Normal ✅"
        print(f"  Prediction: {flag}")
        save_annotated(res['annotated_img'], f"anemia_{os.path.basename(path)}")


def run_blood_smear(image_paths: list):
    from feature_extractors.blood_smear import count_blood_cells
    print(f"\n{SEPARATOR}")
    print("TOOL 13: BLOOD SMEAR COUNTER")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        res = count_blood_cells(img)
        print(f"\n  Image: {path}")
        print(f"  RBC: {res['rbc_count']}, WBC: {res['wbc_count']}, Platelets: {res['platelet_count']}")
        print(f"  WBC:RBC ratio: {res['wbc_rbc_ratio']:.5f}")
        flag = "HIGH WBC ⚠️" if res['high_wbc_flag'] else "Normal ✅"
        print(f"  Assessment: {flag}")
        save_annotated(res['annotated_img'], f"blood_{os.path.basename(path)}")


def run_pill_identifier(image_paths: list):
    from feature_extractors.pill_identifier import extract_pill_features, match_pill
    print(f"\n{SEPARATOR}")
    print("TOOL 12: PILL IDENTIFIER")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        feats = extract_pill_features(img)
        matches = match_pill(feats)
        print(f"\n  Image: {path}")
        print(f"  Shape: AR={feats['aspect_ratio']:.2f}, Circ={feats['circularity']:.2f}, Convex={feats['convexity']:.2f}")
        print(f"  Dominant colour: R={feats['dominant_r']:.0f} G={feats['dominant_g']:.0f} B={feats['dominant_b']:.0f}")
        print("  Top matches:")
        for m in matches:
            print(f"    {m['name']:25s}  confidence={m['confidence']:.1f}%  dist={m['distance']:.4f}")
        save_annotated(feats['annotated_img'], f"pill_{os.path.basename(path)}")


def run_surgical_counter(image_paths: list):
    from feature_extractors.surgical_counter import segment_instruments
    print(f"\n{SEPARATOR}")
    print("TOOL 11: SURGICAL INSTRUMENT COUNTER")
    print(SEPARATOR)
    for path in image_paths:
        if not os.path.exists(path):
            print(f"  [SKIP] {path}")
            continue
        img = cv2.imread(path)
        res = segment_instruments(img)
        print(f"\n  Image: {path}")
        print(f"  Total instruments: {res['total_count']}")
        print(f"  Inventory: {res['inventory']}")
        print(f"  Mismatch: {'YES ⚠️' if res['mismatch'] else 'No ✅'}")
        save_annotated(res['annotated_img'], f"surgical_{os.path.basename(path)}")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MediScan AI — Sample Test Runner")
    print("  Update the file paths below, then re-run.")
    print("=" * 65)

    # ── FILL IN YOUR SAMPLE IMAGE PATHS HERE ────────────────────────────────
    MELANOMA_SAMPLES = [
        "data/sample_mole_1.jpg",   # benign mole
        "data/sample_mole_2.jpg",   # atypical mole
        "data/sample_mole_3.jpg",   # melanoma-like lesion
    ]
    DR_SAMPLES = [
        "data/fundus_normal.jpg",
        "data/fundus_mild_dr.jpg",
        "data/fundus_severe_dr.jpg",
    ]
    JAUNDICE_SAMPLES = [
        "data/sclera_normal.jpg",
        "data/sclera_yellow.jpg",
        "data/skin_normal.jpg",
    ]
    ANEMIA_SAMPLES = [
        "data/conjunctiva_normal.jpg",
        "data/conjunctiva_pale.jpg",
        "data/conjunctiva_ref.jpg",
    ]
    BLOOD_SAMPLES = [
        "data/blood_smear_1.jpg",
        "data/blood_smear_2.jpg",
        "data/blood_smear_3.jpg",
    ]
    PILL_SAMPLES = [
        "data/pill_round.jpg",
        "data/pill_oval.jpg",
        "data/pill_capsule.jpg",
    ]
    SURGICAL_SAMPLES = [
        "data/tray_before.jpg",
        "data/tray_after.jpg",
        "data/tray_mismatch.jpg",
    ]
    # ─────────────────────────────────────────────────────────────────────────

    run_melanoma(MELANOMA_SAMPLES)
    run_diabetic_retinopathy(DR_SAMPLES)
    run_jaundice(JAUNDICE_SAMPLES)
    run_anemia(ANEMIA_SAMPLES)
    run_blood_smear(BLOOD_SAMPLES)
    run_pill_identifier(PILL_SAMPLES)
    run_surgical_counter(SURGICAL_SAMPLES)

    print(f"\n{SEPARATOR}")
    print("✅ Sample test complete. Check output/ folder for annotated images.")
    print(SEPARATOR)
