"""
calibration.py -- Color calibration for jaundice and anemia screening tools.

Physical reference to use in real demo:
  - A plain white printer paper OR a ColorChecker card held in frame
  - White card should be 10-15% of frame area, in the same lighting as the subject
  - DO NOT use a phone screen as reference -- it emits its own light
  - Ideal: Macbeth ColorChecker Classic, patch D-18 (white) or A-1 (neutral gray)

Calibration corrects for:
  - Ambient light color temperature (warm/cool)
  - Phone camera auto-white-balance drift
  - Different camera sensor spectral responses
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Save plots without a display
import matplotlib.pyplot as plt
import json
import os


# ---------------------------------------------------------------------------
# Interactive reference patch selection
# ---------------------------------------------------------------------------

def select_reference_patch_interactive(img_bgr: np.ndarray) -> tuple:
    """
    Let the operator draw a rectangle over the white/neutral calibration
    reference (e.g., white paper, ColorChecker patch) using OpenCV's
    built-in ROI selector.

    Clinical rationale
    ------------------
    The reference patch must be in the *same image frame* as the patient's
    skin or sclera so that it shares identical lighting conditions.
    Selecting the patch interactively is safer than hardcoding coordinates
    because camera framing varies between operators and sessions.

    Parameters
    ----------
    img_bgr : np.ndarray
        Raw BGR image loaded with cv2.imread().

    Returns
    -------
    (x, y, w, h) : tuple of ints
        Top-left corner (x, y) and dimensions (w, h) of the selected patch.
    """
    print(
        "\n[CALIBRATION] Instructions:\n"
        "  Draw a rectangle over the white/neutral reference patch.\n"
        "  Press ENTER or SPACE to confirm selection.\n"
        "  Press C to cancel and exit.\n"
    )

    # cv2.selectROI returns (x, y, w, h); fromCenter=False means drag from
    # top-left corner which is more intuitive for clinical operators.
    roi = cv2.selectROI(
        windowName="Select Reference Patch -- ENTER to confirm, C to cancel",
        img=img_bgr,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    if w == 0 or h == 0:
        raise ValueError(
            "Reference patch selection cancelled or zero-size. "
            "Please re-run and draw a non-empty rectangle."
        )

    print(f"[CALIBRATION] Patch selected: x={x}, y={y}, w={w}, h={h}")
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def normalize_to_reference(
    img_bgr: np.ndarray,
    ref_coords: tuple
) -> tuple:
    """
    Apply white-balance correction so the reference patch reads as
    neutral gray (L=128, A=128, B=128) in CIE LAB color space.

    CALIBRATION NOTE
    ----------------
    We work in CIE LAB (L*a*b*) rather than RGB because:
      - The L channel is perceptual lightness (independent of hue).
      - The A channel encodes green-red axis -- critical for anemia
        (pallor/redness) and jaundice (yellowness appears as +b*).
      - The B channel encodes blue-yellow axis -- elevated b* is the
        primary jaundice biomarker (scleral icterus).

    The shift formula:
        correction = 128 - mean_of_reference_patch
    ensures the reference reads as mid-gray (128,128,128) after correction,
    which is equivalent to assuming the reference is a Lambertian reflector
    under the scene illuminant.

    Parameters
    ----------
    img_bgr   : np.ndarray   Raw BGR image from camera.
    ref_coords: tuple        (x, y, w, h) of the calibration reference patch.

    Returns
    -------
    normalized_lab_img : np.ndarray (uint8, LAB color space)
    correction_dict    : dict  {'L': float, 'A': float, 'B': float}
    """
    # Convert the full image to CIE LAB.
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    x, y, w, h = ref_coords

    # Extract the reference patch from the LAB image.
    ref_patch = img_lab[y: y + h, x: x + w]

    if ref_patch.size == 0:
        raise ValueError(
            f"Reference patch at ({x},{y},{w},{h}) is empty. "
            "Check that coordinates are within image bounds."
        )

    ref_mean_L = float(ref_patch[:, :, 0].mean())
    ref_mean_A = float(ref_patch[:, :, 1].mean())
    ref_mean_B = float(ref_patch[:, :, 2].mean())

    # Compute per-channel additive corrections.
    # CALIBRATION NOTE: We shift all channels so the reference patch reads
    # (128, 128, 128) = neutral gray in LAB.  128 is the OpenCV mid-point
    # for 8-bit LAB encoding (L: 0-255 maps to L*: 0-100; A,B: 0-255 map
    # to a*,b*: -128 to +127).
    correction = {
        'L': 128.0 - ref_mean_L,
        'A': 128.0 - ref_mean_A,
        'B': 128.0 - ref_mean_B
    }

    print(
        f"[CALIBRATION] Reference patch LAB means: "
        f"L={ref_mean_L:.2f}, A={ref_mean_A:.2f}, B={ref_mean_B:.2f}"
    )
    print(
        f"[CALIBRATION] Corrections applied: "
        f"dL={correction['L']:+.2f}, dA={correction['A']:+.2f}, "
        f"dB={correction['B']:+.2f}"
    )

    # Apply corrections channel-wise using float arithmetic to avoid
    # integer wrap-around artifacts (critical for subtle color shifts).
    img_lab_float = img_bgr.copy()  # shape placeholder
    img_lab_float = img_lab.astype(np.float64)

    img_lab_float[:, :, 0] += correction['L']
    img_lab_float[:, :, 1] += correction['A']
    img_lab_float[:, :, 2] += correction['B']

    # Clip to valid 8-bit range and convert back to uint8.
    img_lab_corrected = np.clip(img_lab_float, 0, 255).astype(np.uint8)

    return img_lab_corrected, correction


# ---------------------------------------------------------------------------
# Calibration persistence
# ---------------------------------------------------------------------------

def save_calibration(correction: dict, path: str = 'calibration.json'):
    """
    Persist the current session's calibration correction to a JSON file.

    By saving calibration, all images captured in the same session
    (same device, same room lighting) can be consistently corrected
    without re-selecting the reference patch each time.

    Parameters
    ----------
    correction : dict   Keys 'L', 'A', 'B' with float correction values.
    path       : str    Destination file path.
    """
    json.dump(correction, open(path, 'w'), indent=2)
    print(f"[CALIBRATION] Correction saved to '{path}'")
    print(f"  L_correction={correction['L']:+.4f}")
    print(f"  A_correction={correction['A']:+.4f}")
    print(f"  B_correction={correction['B']:+.4f}")


def load_calibration(path: str = 'calibration.json') -> dict:
    """
    Load a previously saved calibration correction from JSON.

    Parameters
    ----------
    path : str   Path to the calibration JSON file.

    Returns
    -------
    correction : dict   {'L': float, 'A': float, 'B': float}

    Raises
    ------
    FileNotFoundError if the calibration file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Calibration file not found: '{path}'\n"
            "Run select_reference_patch_interactive() + normalize_to_reference() "
            "+ save_calibration() first."
        )
    correction = json.load(open(path))
    print(f"[CALIBRATION] Loaded correction from '{path}': {correction}")
    return correction


# ---------------------------------------------------------------------------
# Apply saved calibration
# ---------------------------------------------------------------------------

def apply_saved_calibration(
    img_bgr: np.ndarray,
    calibration_path: str = 'calibration.json'
) -> np.ndarray:
    """
    Apply a previously computed calibration correction to a new image.

    Use this for every image captured after the initial calibration step
    within the same lighting session.  Loading the correction from disk
    is intentionally cheap so this function can be called per-frame in a
    live camera loop without performance overhead.

    Parameters
    ----------
    img_bgr          : np.ndarray   BGR image to correct.
    calibration_path : str          Path to the saved calibration JSON.

    Returns
    -------
    normalized_lab_img : np.ndarray (uint8, LAB color space)
    """
    correction = load_calibration(calibration_path)

    # Convert to CIE LAB, apply stored shifts, clip, return.
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    img_lab[:, :, 0] += correction['L']
    img_lab[:, :, 1] += correction['A']
    img_lab[:, :, 2] += correction['B']

    normalized_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    return normalized_lab


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def visualize_calibration(img_bgr: np.ndarray, ref_coords: tuple):
    """
    Create and save a side-by-side diagnostic plot comparing the LAB b*
    channel before and after calibration.

    Clinical rationale
    ------------------
    The b* (blue-yellow) channel is the primary biomarker for jaundice:
    - Normal sclera: b* ~ 10-15
    - Jaundiced sclera: b* > 20 (OpenCV uint8 scale ~= actual b* + 128)
    Plotting b* before/after correction lets the clinician verify that the
    calibration is removing *illuminant bias* without crushing real signal.

    Parameters
    ----------
    img_bgr    : np.ndarray   Original BGR image.
    ref_coords : tuple        (x, y, w, h) of the calibration reference patch.
    """
    os.makedirs('output', exist_ok=True)

    # Original b* channel
    img_lab_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b_orig = img_lab_orig[:, :, 2].astype(np.float32)
    mean_b_orig = float(b_orig.mean())

    # Calibrated b* channel
    img_lab_cal, correction = normalize_to_reference(img_bgr, ref_coords)
    b_cal = img_lab_cal[:, :, 2].astype(np.float32)
    mean_b_cal = float(b_cal.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(b_orig, cmap='RdYlBu_r', vmin=100, vmax=180)
    axes[0].set_title(
        f"Original LAB b* channel\nMean b* = {mean_b_orig:.2f}",
        fontsize=12
    )
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Draw the reference patch rectangle on the original panel
    x, y, w, h = ref_coords
    rect = plt.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='lime', facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].text(
        x, y - 5, 'Ref patch', color='lime', fontsize=9,
        fontweight='bold'
    )

    im1 = axes[1].imshow(b_cal, cmap='RdYlBu_r', vmin=100, vmax=180)
    axes[1].set_title(
        f"Calibrated LAB b* channel\nMean b* = {mean_b_cal:.2f}  "
        f"(shift={correction['B']:+.2f})",
        fontsize=12
    )
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        'Color Calibration Verification -- b* (blue-yellow) Channel',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    out_path = os.path.join('output', 'calibration_check.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[CALIBRATION] Verification plot saved -> '{out_path}'")
    print(
        f"[CALIBRATION] b* mean before: {mean_b_orig:.2f}  |  "
        f"b* mean after: {mean_b_cal:.2f}"
    )


# ---------------------------------------------------------------------------
# Demo setup instructions
# ---------------------------------------------------------------------------

def calibration_demo_instructions() -> str:
    """
    Return a step-by-step string guide for setting up physical calibration
    during a hackathon demo or clinical field test.

    The instructions are written for non-technical operators (nurses, demo
    judges) who may not be familiar with color science.

    Returns
    -------
    str : Multi-line instruction string ready to print or display in a GUI.
    """
    instructions = """
============================================================
   MEDISCAN COLOR CALIBRATION — DEMO SETUP INSTRUCTIONS
============================================================

PURPOSE
-------
Color calibration removes bias caused by:
  * Warm/cool room lighting (fluorescent vs. LED vs. sunlight)
  * Phone camera auto-white-balance variations
  * Different phone camera sensors

You ONLY need to calibrate ONCE per lighting session.
If lighting changes (e.g. you move rooms), recalibrate.

WHAT YOU NEED
-------------
Option A (best): Macbeth ColorChecker Classic card
  -> Use patch D-18 (white) or any neutral gray patch.
Option B (acceptable): Plain white printer paper (80 g/m2)
  -> Cut a 5 x 5 cm piece.
Option C (last resort): Any matte white surface that is
  NOT a screen (screens emit light and will fool the sensor).

STEP-BY-STEP SETUP
------------------
1. Place the white/neutral reference card in the same scene
   as the patient's skin or eye.  It should occupy 10-15% of
   the frame — visible in the corner without blocking the ROI.

2. Take a photo with BOTH the reference card AND the patient
   area (sclera / fingernail / palm) clearly visible.

3. Load the image in Python:
       import cv2
       from calibration import (
           select_reference_patch_interactive,
           normalize_to_reference,
           save_calibration,
           visualize_calibration
       )
       img = cv2.imread('patient_photo.jpg')

4. Interactively mark the reference card region:
       ref_coords = select_reference_patch_interactive(img)
   -> A window opens. Draw a rectangle over the white card.
   -> Press ENTER to confirm.

5. Apply and save the calibration for this session:
       calibrated_img, correction = normalize_to_reference(
           img, ref_coords
       )
       save_calibration(correction, 'calibration.json')

6. VERIFY the calibration quality (optional but recommended):
       visualize_calibration(img, ref_coords)
   -> Opens output/calibration_check.png
   -> b* mean after calibration should be ~128 for white card.

7. For ALL subsequent images in this session, apply:
       from calibration import apply_saved_calibration
       calibrated = apply_saved_calibration(
           new_img, 'calibration.json'
       )
   Then pass `calibrated` to extract_jaundice_features() or
   extract_anemia_features() instead of the raw BGR image.

COMMON MISTAKES TO AVOID
-------------------------
  X  Do NOT use a phone screen, monitor, or tablet as reference.
  X  Do NOT recalibrate between patients if lighting is unchanged.
  X  Do NOT hold the reference card at an angle -- keep it flat,
     facing the camera, in the same plane as the skin ROI.
  X  Do NOT crease or dirty the reference card.

============================================================
"""
    return instructions


# ---------------------------------------------------------------------------
# Quick self-test / demo entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Print the demo instructions when the module is run directly.
    # This lets operators quick-reference the setup guide from the terminal.
    print(calibration_demo_instructions())
