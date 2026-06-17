# MediScan AI — Medical Computer Vision Platform

MediScan AI is a complete, runnable medical computer vision platform containing **13 explainable diagnostic and monitoring tools** built with **OpenCV, MediaPipe, and scikit-learn**. By avoiding deep neural networks, the platform runs instantly on standard CPUs, requires no GPU, and offers fully transparent, explainable feature extraction and clinical reasoning models (e.g. Decision Trees, Logistic Regression).

---

## 🚀 Key Features & The 13 Tools

The platform is structured into four main clinical categories:

### 1. Screening & Diagnostics
*   **Skin Lesion (Melanoma) Classifier**: Extracts clinical **ABCDE** features: Asymmetry (Otsu saturation channel difference), Border (circularity metric), Color (LAB channel variance), Diameter (pixel equivalent), and Texture (GLCM contrast/homogeneity).
*   **Diabetic Retinopathy Detector**: Performs green channel enhancement (CLAHE), black-hat morphology, and blob detection to count microaneurysms and hemorrhages, producing an ETDRS-like grading scale.
*   **Jaundice Screening Tool**: Normalizes illumination to a neutral white/gray reference card (converting BGR to LAB space) and measures the yellow axis ($b^*$) mean of the skin or sclera.
*   **Anemia Detector**: Analyzes conjunctival pallor using normalized LAB Lightness ($L^*$) and Redness ($a^*$) metrics to compute a Pallor Index.

### 2. Vitals & Monitoring (Without Hardware)
*   **Remote Heart-Rate (rPPG) Monitor**: Detects face landmarks, isolates the forehead region of interest (ROI), extracts the green channel photoplethysmogram (PPG) signal, filters it (0.7-3.0 Hz bandpass), and calculates BPM via FFT and peak detection.
*   **Respiratory Rate Monitor**: Tracks vertical shoulder movements using MediaPipe Pose, applies a 0.1-0.8 Hz bandpass filter, and calculates respiratory rate from oscillatory peaks.
*   **Drowsiness (EAR) Detector**: Computes the Eye Aspect Ratio (EAR) from 6-point eye landmark geometry to track blinks, closure rate (PERCLOS), and trigger fatigue warnings.

### 3. Movement & Rehab
*   **Tremor Frequency Analyzer**: Performs FFT on 2D hand landmark coordinates (MediaPipe Hands) to classify resting tremors (Parkinsonian 4-6 Hz), postural/action tremors (Essential 6-12 Hz), or normal physiological movement.
*   **Gait Symmetry Analyzer**: Compares left vs. right knee vertical oscillation amplitude during walking to compute the Robinson Symmetry Index (SI) and estimate steps per minute (cadence).
*   **Range of Motion (ROM) Tracker**: Measures real-time joint angles (Shoulder Flexion, Knee Flexion, Elbow Extension, Hip Flexion) using trigonometry on MediaPipe Pose landmarks, scoring against target mobility ranges.

### 4. Clinical & Hospital Tools
*   **Surgical Instrument Counter**: Prevents Retained Surgical Items (RSI) using Otsu thresholding and morphological closure to detect, classify (sponges vs. scissors/forceps), and verify instrument counts against a starting reference.
*   **Pill Identifier**: Performs shape contour analysis (aspect ratio, circularity, convexity) and K-Means dominant color extraction to match pills against a reference database using K-Nearest Neighbors (KNN).
*   **Peripheral Blood Smear Cell Counter**: Segments and counts Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets from microscopic smear images using color-based HSV thresholding and morphology.

---

## 📂 Project Structure

```
├── index.html                   # Interactive web frontend explaining the science/architecture
├── app.js                       # Frontend logic for interactive formulas and science visualization
├── styles.css                   # Sleek dark-mode styling for the frontend
├── README.md                    # This documentation file
└── mediscan/
    ├── requirements.txt         # Required Python libraries
    ├── demo_app.py              # Main interactive Streamlit application containing all 13 tools
    ├── model.py                 # ML training script for Decision Trees and Logistic Regression
    ├── calibration.py           # Color calibration utility for jaundice and anemia screening
    ├── sample_test.py           # Pipeline validation script using sample inputs
    ├── data/                    # Folder for training and testing images
    ├── models/                  # Saved .pkl classifiers and feature importance plots
    ├── output/                  # Processed and annotated output media
    └── feature_extractors/      # Python modules for each tool's feature extraction
        ├── __init__.py
        ├── melanoma.py
        ├── diabetic_retinopathy.py
        ├── jaundice.py
        ├── anemia.py
        ├── heart_rate.py
        ├── respiratory_rate.py
        ├── drowsiness.py
        ├── range_of_motion.py
        ├── tremor.py
        ├── gait.py
        ├── surgical_counter.py
        ├── pill_identifier.py
        └── blood_smear.py
```

---

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r mediscan/requirements.txt
```

*Requirements include: `opencv-python`, `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `mediapipe`, `streamlit`, `matplotlib`, `Pillow`, `pandas`, `plotly`.*

### 2. Run the Interactive Streamlit Demo App
```bash
cd mediscan
streamlit run demo_app.py
```
This launches a browser interface where you can upload images, use your live webcam, view real-time signal processing charts (PPG wave, FFT spectrums), and explore model explainability trees.

### 3. Validate Features / Run Tests
To quickly run the pipeline on your test images:
```bash
python sample_test.py
```

---

## 🩺 Clinical Alignment & References
- **Melanoma**: ABCDE criteria matching international dermatological standards.
- **Heart Rate**: Remote photoplethysmography (rPPG) based on *Verkruysse et al. (2008)*.
- **Drowsiness**: Eye Aspect Ratio (EAR) based on *Soukupová & Čech (2016)*.
- **Gait Symmetry**: Robinson Symmetry Index (SI) *Robinson (1987)*.
- **Calibration**: LAB space illuminant correction to neutral gray reference *BiliCam (2015)*.
