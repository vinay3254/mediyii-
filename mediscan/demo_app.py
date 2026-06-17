"""
demo_app.py — Unified Streamlit demo for all 13 MediScan CV tools.

Run:  streamlit run demo_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io, os, sys, traceback

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI — Medical CV Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0a0f1e; }
    .stApp { background: #0a0f1e; color: #f0f4ff; }
    .tool-header { font-size: 1.4rem; font-weight: 800; margin-bottom: 4px; }
    .risk-high   { color: #ef4444; font-weight: 700; font-size: 1.2rem; }
    .risk-medium { color: #f59e0b; font-weight: 700; font-size: 1.2rem; }
    .risk-low    { color: #10b981; font-weight: 700; font-size: 1.2rem; }
    .feature-box { background: rgba(255,255,255,0.05); border-radius: 8px;
                   padding: 10px; margin: 4px 0; }
    div[data-testid="stSidebar"] { background: #060c1a; }
</style>
""", unsafe_allow_html=True)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_img) -> np.ndarray:
    arr = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def show_feature_bar(label: str, value: float, max_val: float = 1.0,
                     color: str = '#3d8bff'):
    pct = min(100, value / (max_val + 1e-9) * 100)
    st.markdown(f"""
    <div class="feature-box">
      <small style="color:#8b9cc8">{label}</small><br>
      <strong>{value:.4f}</strong>
      <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;margin-top:4px;">
        <div style="width:{pct:.0f}%;height:6px;border-radius:4px;background:{color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def risk_badge(label: str):
    cls = 'risk-high' if label == 'High' else 'risk-medium' if label == 'Medium' else 'risk-low'
    icon = '⚠️' if label == 'High' else '⚡' if label == 'Medium' else '✅'
    st.markdown(f'<p class="{cls}">{icon} Risk Level: {label}</p>', unsafe_allow_html=True)


def plot_signal(signal: np.ndarray, title: str, ylabel: str,
                color: str = '#3d8bff', peaks=None) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1525')
    ax.plot(signal, color=color, linewidth=1.2, alpha=0.9)
    if peaks is not None and len(peaks) > 0:
        ax.plot(peaks, signal[peaks], 'o', color='#f59e0b', markersize=4)
    ax.set_title(title, color='#8b9cc8', fontsize=10)
    ax.set_ylabel(ylabel, color='#8b9cc8', fontsize=8)
    ax.tick_params(colors='#4a5580')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2240')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#0a0f1e')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Tool selector
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stethoscope.png", width=60)
    st.markdown("## 🏥 MediScan AI")
    st.markdown("*13 explainable medical CV tools*")
    st.markdown("---")

    CATEGORIES = {
        "🩺 Screening & Diagnostics": [
            "1. Melanoma Classifier",
            "2. Diabetic Retinopathy",
            "3. Jaundice Screening",
            "4. Anemia Detector",
        ],
        "💓 Vitals & Monitoring": [
            "5. Heart Rate (rPPG)",
            "6. Respiratory Rate",
            "7. Drowsiness Monitor",
        ],
        "🦾 Movement & Rehab": [
            "8. Range of Motion",
            "9. Tremor Analysis",
            "10. Gait Analysis",
        ],
        "🏥 Hospital Workflow": [
            "11. Surgical Counter",
            "12. Pill Identifier",
            "13. Blood Smear Counter",
        ],
    }

    tool_options = []
    for tools in CATEGORIES.values():
        tool_options.extend(tools)

    selected = st.selectbox("Select Tool", tool_options, index=0)
    st.markdown("---")
    st.markdown("**Stack:** OpenCV · MediaPipe · scikit-learn · scipy")
    st.markdown("**No GPU required** — classical CV + interpretable ML")
    st.caption("⚠️ Research demo only. Not a clinical device.")


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. MELANOMA ───────────────────────────────────────────────────────────────
if selected.startswith("1."):
    st.markdown('<p class="tool-header">🔬 Skin Lesion Melanoma Classifier</p>',
                unsafe_allow_html=True)
    st.markdown("Extracts **ABCDE clinical features** (asymmetry, border, colour, diameter, texture) "
                "via OpenCV contour analysis + GLCM, fed into a Decision Tree.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload mole/lesion photo", type=['jpg','png','jpeg'],
                                     key='mel')
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Input Image", use_column_width=True)

    with col2:
        if uploaded:
            with st.spinner("Extracting ABCDE features..."):
                try:
                    from feature_extractors.melanoma import (
                        extract_lesion_features, FEATURE_NAMES)
                    img_bgr = pil_to_bgr(pil_img)
                    feats = extract_lesion_features(img_bgr)
                    if feats is None:
                        st.error("No lesion detected — try a cleaner image.")
                    else:
                        st.image(bgr_to_rgb(feats['annotated_img']),
                                 caption="Annotated (Green=contour, Red=bbox, Blue=asymmetry axes)",
                                 use_column_width=True)
                        risk_badge(feats['risk_label'])

                        st.markdown("**Extracted ABCDE Features:**")
                        show_feature_bar("A — Asymmetry", feats['asymmetry'], 1.0, '#ef4444')
                        show_feature_bar("B — Border Score", feats['border_score'], 5.0, '#f59e0b')
                        show_feature_bar("B — Solidity (Convexity)", feats['solidity'], 1.0, '#eab308')
                        show_feature_bar("C — Colour Std L*", feats['color_std_L'], 50, '#a855f7')
                        show_feature_bar("C — Colour Std a*", feats['color_std_A'], 50, '#ec4899')
                        show_feature_bar("D — Diameter (px)", feats['diameter'], 150, '#3d8bff')
                        show_feature_bar("E — GLCM Contrast", feats['glcm_contrast'], 500, '#10b981')
                        show_feature_bar("E — GLCM Homogeneity", feats['glcm_homogeneity'], 1.0, '#14b8a6')

                        with st.expander("🌳 Decision Tree Rules (sklearn export_text)"):
                            st.code("""
|--- asymmetry <= 0.450
|   |--- border_score <= 2.000
|   |   |--- class: BENIGN
|   |--- border_score > 2.000
|   |   |--- color_std_A <= 18.0
|   |   |   |--- class: SUSPICIOUS
|   |   |--- color_std_A > 18.0
|   |   |   |--- class: MELANOMA
|--- asymmetry > 0.450
|   |--- diameter <= 60.0
|   |   |--- class: SUSPICIOUS
|   |--- diameter > 60.0
|   |   |--- class: MELANOMA
                            """, language='text')
                        with st.expander("ℹ️ Clinical Interpretation"):
                            st.markdown("""
| Feature | Clinical Meaning |
|---------|-----------------|
| **Asymmetry > 0.45** | Melanomas are asymmetric; benign moles symmetric |
| **Border > 2.0** | Irregular, notched border = malignant sign |
| **Solidity < 0.90** | Low solidity (notched, non-convex boundaries) = suspicious |
| **Colour Std > 18** | Multiple pigment shades (brown/black/red/white) |
| **Diameter > 60px** | >6mm in clinical photos = ABCDE criterion D |
| **GLCM Contrast** | Textural heterogeneity of malignant tissue |
""")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 2. DIABETIC RETINOPATHY ───────────────────────────────────────────────────
elif selected.startswith("2."):
    st.markdown('<p class="tool-header">👁 Diabetic Retinopathy Detector</p>',
                unsafe_allow_html=True)
    st.markdown("Detects **microaneurysms** and **hemorrhages** in fundus images via "
                "CLAHE + black-hat morphology + SimpleBlobDetector → ETDRS Grade 0–4.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload retinal fundus image", type=['jpg','png','jpeg'], key='dr')
        if uploaded:
            st.image(Image.open(uploaded), caption="Input Fundus", use_column_width=True)

    with col2:
        if uploaded:
            with st.spinner("Running blob detection pipeline..."):
                try:
                    from feature_extractors.diabetic_retinopathy import detect_dr_lesions
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    res = detect_dr_lesions(img_bgr)

                    grade = res['dr_grade']
                    grade_colors = {0:'#10b981',1:'#14b8a6',2:'#f59e0b',3:'#f97316',4:'#ef4444'}
                    st.markdown(f"<h3 style='color:{grade_colors[grade]}'>Grade {grade}: {res['dr_grade_label']}</h3>",
                                unsafe_allow_html=True)

                    tab1, tab2, tab3 = st.tabs(["Annotated", "Black-Hat", "CLAHE"])
                    with tab1:
                        st.image(bgr_to_rgb(res['annotated_img']),
                                 caption="Yellow=Microaneurysms  Red=Hemorrhages",
                                 use_column_width=True)
                    with tab2:
                        st.image(bgr_to_rgb(res['tophat_img']),
                                 caption="Black-hat output — dark lesions isolated",
                                 use_column_width=True)
                    with tab3:
                        st.image(bgr_to_rgb(res['enhanced_img']),
                                 caption="CLAHE enhanced green channel",
                                 use_column_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Microaneurysms", res['ma_count'])
                    c2.metric("Hemorrhages", res['he_count'])
                    c3.metric("Lesion Density", f"{res['lesion_density']:.2f}/10kpx")

                    with st.expander("📊 DR Grading Scale"):
                        st.markdown("""
| Grade | Label | MA Count | Action |
|-------|-------|----------|--------|
| 0 | No DR | 0 | Annual screening |
| 1 | Mild NPDR | 1–4 | 6-12 month follow-up |
| 2 | Moderate NPDR | 5–14 | 3-6 month follow-up |
| 3 | Severe NPDR | 15–29 | Urgent referral |
| 4 | Proliferative DR | ≥30 | Immediate treatment |
""")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 3. JAUNDICE ───────────────────────────────────────────────────────────────
elif selected.startswith("3."):
    st.markdown('<p class="tool-header">🟡 Jaundice Screening Tool</p>',
                unsafe_allow_html=True)
    st.markdown("Estimates jaundice risk by measuring **LAB b* channel** (yellow axis) "
                "in sclera or skin ROI. Based on BiliCam (UW 2015) neonatal screening research.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload sclera or skin photo", type=['jpg','png','jpeg'], key='jnd')
        scenario = st.selectbox("Demo Scenario", ["Normal (b*~125)", "Borderline (b*~140)", "Jaundiced (b*~160)"])
        has_ref  = st.checkbox("I have a white calibration patch in the photo")
        if uploaded:
            st.image(Image.open(uploaded), caption="Input", use_column_width=True)

    with col2:
        if uploaded:
            with st.spinner("Analysing LAB colour space..."):
                try:
                    from feature_extractors.jaundice import extract_jaundice_features
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    res = extract_jaundice_features(img_bgr)

                    flagged = res['jaundice_flag']
                    if flagged:
                        st.error(f"⚠️ JAUNDICE FLAGGED — b*={res['b_star_mean']:.1f} (threshold 145)")
                    else:
                        st.success(f"✅ Normal — b*={res['b_star_mean']:.1f} (threshold 145)")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(bgr_to_rgb(res['annotated_img']),
                                 caption="Annotated ROI", use_column_width=True)
                    with col_b:
                        st.image(bgr_to_rgb(res['b_channel_img']),
                                 caption="b* channel (brighter = more yellow)",
                                 use_column_width=True)

                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("b* Mean", f"{res['b_star_mean']:.1f}")
                    c2.metric("b* Std", f"{res['b_star_std']:.1f}")
                    c3.metric("L* Mean", f"{res['l_mean']:.1f}")
                    c4.metric("b*/a* Ratio", f"{res['b_star_to_a_star_ratio']:.2f}")
                    c5.metric("Calibrated", "Yes" if res['calibrated'] else "No ⚠️")

                    with st.expander("🧪 Calibration Instructions"):
                        st.markdown("""
**For accurate results:**
1. Print a plain white A4 sheet or use a Macbeth ColorChecker white patch
2. Place it in the SAME lighting as the patient's eye/skin
3. Take photo with the white card visible in one corner
4. Use the ROI selector to mark the white card region
5. The tool normalises all LAB values relative to that neutral reference

**Why?** Different phone cameras and lighting conditions shift LAB b* by 10–30 units,
making uncalibrated readings unreliable across devices.
""")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 4. ANEMIA ─────────────────────────────────────────────────────────────────
elif selected.startswith("4."):
    st.markdown('<p class="tool-header">🩸 Anemia Detector (Conjunctival Pallor)</p>',
                unsafe_allow_html=True)
    st.markdown("Measures **conjunctival pallor** from inner eyelid photo. "
                "High L* + Low a* = Pallor Index → anaemia screen. Published PLOS ONE 2016.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload inner eyelid close-up", type=['jpg','png','jpeg'], key='anm')
        if uploaded:
            st.image(Image.open(uploaded), caption="Input", use_column_width=True)
    with col2:
        if uploaded:
            with st.spinner("Measuring conjunctival colour..."):
                try:
                    from feature_extractors.anemia import extract_anemia_features, PALLOR_THRESHOLD
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    res = extract_anemia_features(img_bgr)
                    flagged = res['anemia_flag']
                    if flagged:
                        st.error(f"⚠️ Possible Anaemia — Pallor Index={res['pallor_index']:.2f} (threshold {PALLOR_THRESHOLD})")
                    else:
                        st.success(f"✅ Normal — Pallor Index={res['pallor_index']:.2f}")
                    st.image(bgr_to_rgb(res['annotated_img']), caption="Annotated ROI", use_column_width=True)
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("Pallor Index", f"{res['pallor_index']:.2f}")
                    c2.metric("L* (lightness)", f"{res['l_mean']:.1f}")
                    c3.metric("a* (redness)", f"{res['a_mean']:.1f}")
                    c4.metric("Redness Ratio", f"{res['redness_ratio']:.2f}")
                    c5.metric("Redness Std", f"{res['redness_std']:.2f}")
                    with st.expander("ℹ️ Clinical Basis"):
                        st.markdown("""
**Pallor Index = L* / a***
- High L* (pale, bright) + Low a* (less red) = anaemic conjunctiva
- Threshold 6.5 correlates with Hb < 11 g/dL
- Reference: Mannino et al. (2016). PLOS ONE. HemaApp (UW 2016).
- **Limitation:** Requires controlled lighting and calibration reference for clinical use.
""")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── 5. HEART RATE ─────────────────────────────────────────────────────────────
elif selected.startswith("5."):
    st.markdown('<p class="tool-header">❤️ Remote Heart Rate Monitor (rPPG)</p>',
                unsafe_allow_html=True)
    st.markdown("Extracts **BPM from webcam video** by bandpass-filtering green-channel "
                "intensity changes in the forehead ROI (Verkruysse 2008 technique).")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload face video (≥10s)", type=['mp4','avi','mov','mkv'], key='hr')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Processing rPPG signal (may take 30–60s for long videos)..."):
                try:
                    import tempfile
                    from feature_extractors.heart_rate import extract_rppg_signal
                    tmp = tempfile.mktemp(suffix=os.path.splitext(uploaded.name)[1])
                    with open(tmp, 'wb') as f:
                        f.write(uploaded.read())
                    res = extract_rppg_signal(tmp)
                    os.remove(tmp)

                    st.metric("Heart Rate (FFT)", f"{res['bpm_fft']:.0f} BPM")
                    st.metric("Heart Rate (Peaks)", f"{res['bpm_peaks']:.0f} BPM")
                    st.metric("Dominant Frequency", f"{res['dominant_freq_hz']:.3f} Hz")

                    # PPG waveform plot
                    sig_img = plot_signal(res['filtered_signal'], "Filtered PPG Signal",
                                          "Amplitude", '#ec4899')
                    st.image(sig_img, caption="Bandpass-filtered green channel (0.7–3.0 Hz)")

                    # FFT spectrum
                    fig, ax = plt.subplots(figsize=(8,2.5), facecolor='#0a0f1e')
                    ax.set_facecolor('#0d1525')
                    mask = (res['fft_freqs'] >= 0.5) & (res['fft_freqs'] <= 4.0)
                    ax.plot(res['fft_freqs'][mask], res['fft_magnitude'][mask],
                            color='#3d8bff', linewidth=1.5)
                    ax.axvline(res['dominant_freq_hz'], color='#ef4444', linestyle='--',
                               label=f"BPM={res['bpm_fft']:.0f}")
                    ax.set_title("FFT Spectrum", color='#8b9cc8', fontsize=10)
                    ax.set_xlabel("Frequency (Hz)", color='#8b9cc8', fontsize=8)
                    ax.tick_params(colors='#4a5580')
                    ax.legend(facecolor='#0d1525', labelcolor='white', fontsize=8)
                    for sp in ax.spines.values(): sp.set_edgecolor('#1a2240')
                    plt.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, facecolor='#0a0f1e')
                    plt.close(fig); buf.seek(0)
                    st.image(buf.read(), caption="FFT — Peak = dominant heart frequency")

                    if res['annotated_frame'] is not None:
                        st.image(bgr_to_rgb(res['annotated_frame']),
                                 caption="Face + Forehead ROI detection",
                                 use_column_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 6. RESPIRATORY RATE ───────────────────────────────────────────────────────
elif selected.startswith("6."):
    st.markdown('<p class="tool-header">🫁 Respiratory Rate Tracker</p>',
                unsafe_allow_html=True)
    st.markdown("Tracks **shoulder vertical oscillation** via MediaPipe Pose. "
                "Bandpass 0.1–0.8 Hz → peak detection → breaths/min.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload video (patient visible, ≥15s)", type=['mp4','avi','mov'], key='rr')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Tracking shoulder landmarks..."):
                try:
                    import tempfile
                    from feature_extractors.respiratory_rate import extract_respiratory_signal
                    tmp = tempfile.mktemp(suffix='.mp4')
                    with open(tmp, 'wb') as f: f.write(uploaded.read())
                    res = extract_respiratory_signal(tmp)
                    os.remove(tmp)

                    rr = res['respiratory_rate_bpm']
                    color = "normal" if res['normal_range_flag'] else "high"
                    if res['normal_range_flag']:
                        st.success(f"✅ Respiratory Rate: {rr:.0f} breaths/min (Normal: 12–20)")
                    else:
                        st.error(f"⚠️ Respiratory Rate: {rr:.0f} breaths/min (Normal: 12–20)")

                    c1,c2,c3 = st.columns(3)
                    c1.metric("RR", f"{rr:.0f} br/min")
                    c2.metric("Peaks Detected", res['peak_count'])
                    c3.metric("Duration", f"{res['duration_sec']:.0f}s")

                    img_sig = plot_signal(res['filtered'], "Filtered Shoulder Signal (0.1–0.8 Hz)",
                                          "Y position", '#14b8a6', res['peaks_indices'])
                    st.image(img_sig, caption="Orange dots = detected breath peaks")
                    if res['annotated_frame'] is not None:
                        st.image(bgr_to_rgb(res['annotated_frame']),
                                 caption="Shoulder landmarks", use_column_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 7. DROWSINESS ────────────────────────────────────────────────────────────
elif selected.startswith("7."):
    st.markdown('<p class="tool-header">😴 Drowsiness / Fatigue Monitor</p>',
                unsafe_allow_html=True)
    st.markdown("Computes **Eye Aspect Ratio (EAR)** and **PERCLOS** from facial landmarks. "
                "EAR < 0.25 for ≥20 frames → alert. PERCLOS > 15% → fatigued.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload face video", type=['mp4','avi','mov'], key='drw')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Analysing eye closure patterns..."):
                try:
                    import tempfile
                    from feature_extractors.drowsiness import analyze_drowsiness_video
                    tmp = tempfile.mktemp(suffix='.mp4')
                    with open(tmp, 'wb') as f: f.write(uploaded.read())
                    res = analyze_drowsiness_video(tmp)
                    os.remove(tmp)

                    if res['drowsy_flag']:
                        st.error(f"⚠️ DROWSINESS DETECTED — PERCLOS={res['perclos']:.1f}%")
                    else:
                        st.success(f"✅ Alert — PERCLOS={res['perclos']:.1f}%")

                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Mean EAR", f"{res['mean_ear']:.3f}")
                    c2.metric("Min EAR", f"{res['min_ear']:.3f}")
                    c3.metric("PERCLOS", f"{res['perclos']:.1f}%")
                    c4.metric("Blink Rate", f"{res['blink_rate']:.0f}/min")

                    img_ear = plot_signal(res['ear_history'], "EAR Over Time",
                                          "EAR", '#6366f1')
                    st.image(img_ear, caption="Red line at 0.25 = drowsiness threshold")

                    if res['annotated_frame'] is not None:
                        st.image(bgr_to_rgb(res['annotated_frame']),
                                 caption="Eye landmarks (MediaPipe Face Mesh)",
                                 use_column_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 8. RANGE OF MOTION ────────────────────────────────────────────────────────
elif selected.startswith("8."):
    st.markdown('<p class="tool-header">🏋️ Physical Therapy ROM Tracker</p>',
                unsafe_allow_html=True)
    st.markdown("Measures **peak joint angles** from exercise video using MediaPipe Pose. "
                "Scores completion vs clinical target angles.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload PT exercise video", type=['mp4','avi','mov'], key='rom')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Detecting pose landmarks..."):
                try:
                    import tempfile
                    from feature_extractors.range_of_motion import analyze_rom_video, TARGET_ANGLES
                    tmp = tempfile.mktemp(suffix='.mp4')
                    with open(tmp, 'wb') as f: f.write(uploaded.read())
                    res = analyze_rom_video(tmp)
                    os.remove(tmp)

                    overall = res['overall_rom_score']
                    if overall >= 80:
                        st.success(f"✅ Overall ROM Score: {overall:.0f}%")
                    elif overall >= 50:
                        st.warning(f"⚡ Overall ROM Score: {overall:.0f}%")
                    else:
                        st.error(f"⚠️ Overall ROM Score: {overall:.0f}%")

                    for joint, target in TARGET_ANGLES.items():
                        measured = res[joint]
                        score = res['rom_scores'][joint]
                        color = "normal" if score >= 80 else "inverse"
                        st.progress(int(score), text=f"{joint.replace('_',' ').title()}: {measured:.0f}° / {target:.0f}° target ({score:.0f}%)")

                    if res['annotated_frame'] is not None:
                        st.image(bgr_to_rgb(res['annotated_frame']),
                                 caption="Joint angles overlaid on pose", use_column_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 9. TREMOR ────────────────────────────────────────────────────────────────
elif selected.startswith("9."):
    st.markdown('<p class="tool-header">✋ Tremor Analysis — Parkinson\'s Screening</p>',
                unsafe_allow_html=True)
    st.markdown("Tracks **index fingertip jitter** via FFT. Parkinsonian tremor = 4–6 Hz; "
                "essential tremor = 6–12 Hz. Compare band powers for classification.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload hand video (10–30s, palm toward camera)",
                                     type=['mp4','avi','mov'], key='trem')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Running FFT tremor analysis..."):
                try:
                    import tempfile
                    from feature_extractors.tremor import analyze_tremor_video
                    tmp = tempfile.mktemp(suffix='.mp4')
                    with open(tmp, 'wb') as f: f.write(uploaded.read())
                    res = analyze_tremor_video(tmp)
                    os.remove(tmp)

                    label = res['tremor_label']
                    if 'Parkinson' in label:
                        st.error(f"⚠️ {label}")
                    elif 'Essential' in label:
                        st.warning(f"⚡ {label}")
                    else:
                        st.success(f"✅ {label}")

                    c1,c2,c3 = st.columns(3)
                    c1.metric("Dominant Freq", f"{res['dominant_freq_x']:.1f} Hz")
                    c2.metric("Parkinson Band Power", f"{res['power_4_6_x']:.3f}")
                    c3.metric("RMS Jitter", f"{res['rms_jitter']:.5f}")

                    # FFT spectrum
                    fig, ax = plt.subplots(figsize=(8,2.5), facecolor='#0a0f1e')
                    ax.set_facecolor('#0d1525')
                    mask = res['freqs'] <= 20
                    ax.plot(res['freqs'][mask], res['fft_mag_x'][mask],
                            color='#a855f7', linewidth=1.5)
                    ax.axvspan(4, 6, alpha=0.15, color='#f59e0b', label='Parkinson 4-6Hz')
                    ax.axvspan(6, 12, alpha=0.08, color='#6366f1', label='Essential 6-12Hz')
                    ax.axvline(res['dominant_freq_x'], color='#ef4444', linestyle='--',
                               label=f"Peak={res['dominant_freq_x']:.1f}Hz")
                    ax.set_title("FFT Spectrum — Tremor Frequency Analysis", color='#8b9cc8', fontsize=10)
                    ax.set_xlabel("Frequency (Hz)", color='#8b9cc8', fontsize=8)
                    ax.tick_params(colors='#4a5580')
                    ax.legend(facecolor='#0d1525', labelcolor='white', fontsize=7)
                    for sp in ax.spines.values(): sp.set_edgecolor('#1a2240')
                    plt.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, facecolor='#0a0f1e')
                    plt.close(fig); buf.seek(0)
                    st.image(buf.read(), caption="Orange=Parkinson band  Purple=Essential tremor band")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 10. GAIT ─────────────────────────────────────────────────────────────────
elif selected.startswith("10."):
    st.markdown('<p class="tool-header">🚶 Gait Analysis — Symmetry Detector</p>',
                unsafe_allow_html=True)
    st.markdown("Compares **left vs right knee oscillation** amplitude. "
                "Symmetry Index < 10% = normal; > 10% = compensatory gait.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload side-view walking video", type=['mp4','avi','mov'], key='gait')
        if uploaded:
            st.video(uploaded)
    with col2:
        if uploaded:
            with st.spinner("Analysing gait symmetry..."):
                try:
                    import tempfile
                    from feature_extractors.gait import analyze_gait_video
                    tmp = tempfile.mktemp(suffix='.mp4')
                    with open(tmp, 'wb') as f: f.write(uploaded.read())
                    res = analyze_gait_video(tmp)
                    os.remove(tmp)

                    si = res['symmetry_index']
                    if si < 10:
                        st.success(f"✅ Normal Gait — Symmetry Index: {si:.1f}%")
                    elif si < 20:
                        st.warning(f"⚡ Asymmetric Gait — SI: {si:.1f}%")
                    else:
                        st.error(f"⚠️ Significant Asymmetry — SI: {si:.1f}%")

                    c1,c2,c3 = st.columns(3)
                    c1.metric("Symmetry Index", f"{si:.1f}%")
                    c2.metric("Cadence", f"{res['cadence_steps_per_min']:.0f} steps/min")
                    c3.metric("Duration", f"{res['duration_sec']:.0f}s")

                    fig, ax = plt.subplots(figsize=(8,2.5), facecolor='#0a0f1e')
                    ax.set_facecolor('#0d1525')
                    ax.plot(res['left_knee_signal'], color='#3d8bff', linewidth=1.2, label='Left Knee')
                    ax.plot(res['right_knee_signal'], color='#a855f7', linewidth=1.2, label='Right Knee', alpha=0.8)
                    ax.set_title("Knee Vertical Oscillation", color='#8b9cc8', fontsize=10)
                    ax.legend(facecolor='#0d1525', labelcolor='white', fontsize=8)
                    ax.tick_params(colors='#4a5580')
                    for sp in ax.spines.values(): sp.set_edgecolor('#1a2240')
                    plt.tight_layout()
                    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, facecolor='#0a0f1e')
                    plt.close(fig); buf.seek(0)
                    st.image(buf.read(), caption="Amplitude difference between sides = asymmetry")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 11. SURGICAL COUNTER ──────────────────────────────────────────────────────
elif selected.startswith("11."):
    st.markdown('<p class="tool-header">⚕️ Surgical Instrument Counter</p>',
                unsafe_allow_html=True)
    st.markdown("Classifies instruments by **contour morphology** (round=sponge, elongated=forceps). "
                "Flags count mismatches vs pre-procedure reference to prevent RSI.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload tray image (overhead view)", type=['jpg','png','jpeg'], key='surg')
        if uploaded:
            st.image(Image.open(uploaded), caption="Surgical Tray", use_column_width=True)
    with col2:
        if uploaded:
            with st.spinner("Detecting instruments..."):
                try:
                    from feature_extractors.surgical_counter import segment_instruments
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    res = segment_instruments(img_bgr)

                    if res['mismatch']:
                        st.error("⚠️ COUNT MISMATCH — Possible retained item!")
                        for m in res['mismatch_details']:
                            st.markdown(f"- {m}")
                    else:
                        st.success(f"✅ All items accounted for — Total: {res['total_count']}")

                    st.image(bgr_to_rgb(res['annotated_img']),
                             caption="Yellow=sponge  Blue=elongated  Green=small",
                             use_column_width=True)

                    inv = res['inventory']
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("Total", res['total_count'])
                    c2.metric("Large Round", inv['large_round'])
                    c3.metric("Elongated", inv['medium_elongated'])
                    c4.metric("Small Round", inv['small_round'])
                    c5.metric("Irregular", inv['irregular'])
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 12. PILL IDENTIFIER ───────────────────────────────────────────────────────
elif selected.startswith("12."):
    st.markdown('<p class="tool-header">💊 Pill Identifier</p>',
                unsafe_allow_html=True)
    st.markdown("Extracts **shape + dominant colour** features, matches to reference database "
                "via Euclidean KNN. Use plain white background for best accuracy.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload pill photo (white background)", type=['jpg','png','jpeg'], key='pill')
        if uploaded:
            st.image(Image.open(uploaded), caption="Input Pill", use_column_width=True)
    with col2:
        if uploaded:
            with st.spinner("Extracting features and matching..."):
                try:
                    from feature_extractors.pill_identifier import extract_pill_features, match_pill
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    feats = extract_pill_features(img_bgr)
                    matches = match_pill(feats)

                    st.image(bgr_to_rgb(feats['annotated_img']),
                             caption="Contour + dominant colour swatch",
                             use_column_width=True)

                    st.markdown("**Top 3 Matches:**")
                    for i, m in enumerate(matches):
                        icon = "🥇" if i==0 else "🥈" if i==1 else "🥉"
                        st.markdown(f"{icon} **{m['name']}** — Confidence: {m['confidence']:.0f}%")
                        st.progress(int(m['confidence']), text="")

                    c1,c2,c3 = st.columns(3)
                    c1.metric("Aspect Ratio", f"{feats['aspect_ratio']:.2f}")
                    c2.metric("Circularity", f"{feats['circularity']:.2f}")
                    c3.metric("Convexity", f"{feats['convexity']:.2f}")

                    dom_r = int(feats['dominant_r'])
                    dom_g = int(feats['dominant_g'])
                    dom_b = int(feats['dominant_b'])
                    st.markdown(f"**Dominant Colour:** RGB({dom_r}, {dom_g}, {dom_b})")
                    st.markdown(
                        f'<div style="width:50px;height:30px;background:rgb({dom_r},{dom_g},{dom_b});'
                        f'border-radius:4px;border:1px solid #333"></div>',
                        unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── 13. BLOOD SMEAR ───────────────────────────────────────────────────────────
elif selected.startswith("13."):
    st.markdown('<p class="tool-header">🔴 Blood Smear Cell Counter</p>',
                unsafe_allow_html=True)
    st.markdown("Segments **RBCs, WBCs, and platelets** from Giemsa-stained blood smear "
                "via HSV range masking + contour area filtering. Dataset: BCCD (Kaggle).")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload blood smear microscope image", type=['jpg','png','jpeg'], key='blood')
        if uploaded:
            st.image(Image.open(uploaded), caption="Input Smear", use_column_width=True)
    with col2:
        if uploaded:
            with st.spinner("Segmenting cells..."):
                try:
                    from feature_extractors.blood_smear import count_blood_cells
                    img_bgr = pil_to_bgr(Image.open(uploaded))
                    res = count_blood_cells(img_bgr)

                    if res['high_wbc_flag']:
                        st.error(f"⚠️ Elevated WBC count — possible leukocytosis")
                    else:
                        st.success(f"✅ Cell counts within expected field range")

                    tab1, tab2, tab3 = st.tabs(["Annotated", "RBC Mask", "WBC Mask"])
                    with tab1:
                        st.image(bgr_to_rgb(res['annotated_img']),
                                 caption="Green=RBCs  Blue=WBCs  Yellow=Platelets",
                                 use_column_width=True)
                    with tab2:
                        st.image(bgr_to_rgb(cv2.cvtColor(res['rbc_mask'], cv2.COLOR_GRAY2BGR)),
                                 caption="RBC HSV mask (H:0-20° + 160-180°)",
                                 use_column_width=True)
                    with tab3:
                        st.image(bgr_to_rgb(cv2.cvtColor(res['wbc_mask'], cv2.COLOR_GRAY2BGR)),
                                 caption="WBC HSV mask (H:120-160°)",
                                 use_column_width=True)

                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("RBC Count", res['rbc_count'])
                    c2.metric("WBC Count", res['wbc_count'])
                    c3.metric("Platelets", res['platelet_count'])
                    c4.metric("WBC:RBC Ratio", f"{res['wbc_rbc_ratio']:.4f}")

                    with st.expander("📋 Normal Reference Ranges"):
                        st.markdown("""
| Cell | Field Count | Clinical Reference |
|------|-------------|-------------------|
| RBC  | 30-80 per field | 4.2–5.4 million/µL |
| WBC  | 2-12 per field | 4,500–10,500/µL |
| Platelet | 8-25 per field | 150,000–400,000/µL |
| WBC:RBC ratio | ~1:500 | Leukocytosis >10,500/µL |
""")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#4a5580;font-size:0.8rem'>"
    "MediScan AI — Classical CV + Interpretable ML · "
    "OpenCV · MediaPipe · scikit-learn · For research & demo use only"
    "</center>", unsafe_allow_html=True)
