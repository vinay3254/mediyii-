"""
model.py — Unified model trainer for all 13 MediScan CV tools.
Train DecisionTree and LogisticRegression on extracted features.
Prints export_text() decision tree for judge explainability.
Saves models with joblib to models/ directory.
"""

import os
import sys
import glob
import argparse
import importlib

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend so plots save without a display
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# TOOL_CONFIG
# ---------------------------------------------------------------------------
# Each entry maps a clinical tool name to the metadata needed to:
#   1. Dynamically import the correct feature extractor.
#   2. Know what feature names exist (for tree printing and importance plots).
#   3. Know the human-readable label map (class index -> class name).
#
# Clinical rationale for each tool's feature set is documented in the
# corresponding feature_extractors/<tool>.py module.
# ---------------------------------------------------------------------------
TOOL_CONFIG = {
    'melanoma': {
        'feature_names': [
            'asymmetry', 'border_score', 'color_std_L', 'color_std_A',
            'color_std_B', 'diameter', 'glcm_contrast', 'glcm_homogeneity'
        ],
        'label_map': {0: 'benign', 1: 'malignant'},
        'extractor_module': 'feature_extractors.melanoma',
        'extractor_fn': 'extract_lesion_features',
        'input_type': 'image'
    },
    'diabetic_retinopathy': {
        'feature_names': [
            'ma_count', 'he_count', 'lesion_density',
            'green_channel_mean', 'clahe_mean'
        ],
        'label_map': {
            0: 'no_dr', 1: 'mild', 2: 'moderate',
            3: 'severe', 4: 'proliferative'
        },
        'extractor_module': 'feature_extractors.diabetic_retinopathy',
        'extractor_fn': 'detect_dr_lesions',
        'input_type': 'image'
    },
    'jaundice': {
        'feature_names': ['b_star_mean', 'b_star_std', 'l_mean', 'a_mean'],
        'label_map': {0: 'normal', 1: 'jaundiced'},
        'extractor_module': 'feature_extractors.jaundice',
        'extractor_fn': 'extract_jaundice_features',
        'input_type': 'image'
    },
    'anemia': {
        'feature_names': [
            'l_mean', 'a_mean', 'b_mean', 'pallor_index', 'redness_ratio'
        ],
        'label_map': {0: 'normal', 1: 'anemic'},
        'extractor_module': 'feature_extractors.anemia',
        'extractor_fn': 'extract_anemia_features',
        'input_type': 'image'
    },
    'blood_smear': {
        'feature_names': [
            'rbc_count', 'wbc_count', 'platelet_count', 'wbc_rbc_ratio'
        ],
        'label_map': {0: 'normal', 1: 'abnormal'},
        'extractor_module': 'feature_extractors.blood_smear',
        'extractor_fn': 'count_blood_cells',
        'input_type': 'image'
    },
    'pill_identifier': {
        'feature_names': [
            'aspect_ratio', 'circularity', 'convexity',
            'dominant_r', 'dominant_g', 'dominant_b'
        ],
        # label_map is empty because pill classes are dynamic — populated from
        # the pill database at dataset-load time.
        'label_map': {},
        'extractor_module': 'feature_extractors.pill_identifier',
        'extractor_fn': 'extract_pill_features',
        'input_type': 'image'
    },
}

# Supported image extensions when scanning dataset folders.
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset_from_folder(folder: str, tool: str) -> tuple:
    """
    Load a labeled image dataset from a directory tree and extract features.

    Expected folder layout:
        folder/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
            ...

    Each class sub-folder name becomes the string label for that subset.
    Features are extracted by dynamically importing the extractor configured
    in TOOL_CONFIG[tool].

    Parameters
    ----------
    folder : str
        Root directory of the dataset.
    tool : str
        Key into TOOL_CONFIG identifying the clinical tool.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  -- integer class indices
    class_names : list[str]             -- ordered class name strings
    """
    if tool not in TOOL_CONFIG:
        raise ValueError(
            f"Unknown tool '{tool}'. Must be one of {list(TOOL_CONFIG.keys())}"
        )

    cfg = TOOL_CONFIG[tool]

    # Dynamically import the feature extractor so model.py stays tool-agnostic.
    print(f"[{tool}] Importing extractor: {cfg['extractor_module']}.{cfg['extractor_fn']}")
    try:
        mod = importlib.import_module(cfg['extractor_module'])
        extractor_fn = getattr(mod, cfg['extractor_fn'])
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ImportError(
            f"Could not load extractor for '{tool}': {exc}\n"
            f"Make sure {cfg['extractor_module']}.py exists and defines "
            f"{cfg['extractor_fn']}()."
        ) from exc

    # Discover class sub-folders in alphabetical order so label indices are
    # deterministic across runs — critical for reproducible model evaluation.
    class_dirs = sorted(
        [d for d in os.scandir(folder) if d.is_dir()],
        key=lambda e: e.name
    )

    if not class_dirs:
        raise FileNotFoundError(f"No class sub-folders found in '{folder}'")

    class_names = [d.name for d in class_dirs]

    X_rows = []
    y_rows = []

    for class_idx, class_entry in enumerate(class_dirs):
        class_name = class_entry.name
        class_path = class_entry.path

        # Collect all images in this class folder.
        image_paths = []
        for ext_pattern in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(class_path, ext_pattern)))

        if not image_paths:
            print(
                f"  [WARNING] No images found in '{class_path}' "
                f"-- skipping class '{class_name}'."
            )
            continue

        print(
            f"  [LOAD] class='{class_name}' (label={class_idx})  "
            f"images={len(image_paths)}"
        )

        loaded_count = 0
        skipped_count = 0

        for img_path in image_paths:
            try:
                # Call the tool-specific extractor.  It receives the file path
                # and must return a 1-D array/list of floats or None on failure.
                feature_vec = extractor_fn(img_path)

                if feature_vec is None:
                    # Extractor signals that the image is unusable (e.g. blurry,
                    # no ROI found). Skip rather than crashing the whole run.
                    skipped_count += 1
                    continue

                feature_arr = np.array(feature_vec, dtype=np.float32)
                X_rows.append(feature_arr)
                y_rows.append(class_idx)
                loaded_count += 1

            except Exception as exc:
                # Per-image errors must never abort the entire dataset load.
                print(f"    [ERROR] {os.path.basename(img_path)}: {exc}")
                skipped_count += 1

        print(f"    -> loaded={loaded_count}, skipped={skipped_count}")

    if not X_rows:
        raise RuntimeError(
            f"No valid feature vectors extracted for tool='{tool}' from "
            f"folder='{folder}'. Check that images exist and the extractor "
            "is working correctly."
        )

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int32)

    print(
        f"\n[{tool}] Dataset summary: X={X.shape}, y={y.shape}, "
        f"classes={class_names}\n"
    )
    return X, y, class_names


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    tool_name: str,
    class_names: list
):
    """
    Train a DecisionTreeClassifier and LogisticRegression on the supplied
    feature matrix, evaluate on a held-out test set, and persist both models.

    Clinical explainability rationale
    ----------------------------------
    Decision trees are preferred in clinical settings because:
      - export_text() produces a human-readable rule set that a clinician or
        judge can audit without ML expertise.
      - Feature importances show *which* biomarkers drive predictions.
    Logistic Regression provides a probabilistic complement: its coefficients
    are monotone and directly interpretable as odds-ratio contributors.

    Parameters
    ----------
    X            : Feature matrix (n_samples, n_features)
    y            : Integer label array (n_samples,)
    feature_names: Names matching columns of X (for printing and plots)
    tool_name    : Identifier used for file names and headers
    class_names  : Human-readable class strings, indexed by y values

    Returns
    -------
    dt_clf   : Trained DecisionTreeClassifier
    lr_clf   : Trained LogisticRegression
    test_acc : Accuracy on held-out test set (Decision Tree)
    """
    os.makedirs('models', exist_ok=True)

    # ---- Train / test split ------------------------------------------------
    # Stratify so class proportions are preserved in both splits -- essential
    # for imbalanced clinical datasets (e.g. rare positive cases).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train samples: {len(X_train)}   Test samples: {len(X_test)}")

    # ---- Decision Tree -----------------------------------------------------
    # max_depth=5 prevents overfitting on small clinical datasets while still
    # capturing multi-level diagnostic rules (e.g., if diameter > 6mm AND
    # asymmetry > 0.3 => malignant).
    # class_weight='balanced' compensates for class imbalance automatically.
    dt_clf = DecisionTreeClassifier(
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    dt_acc = float(np.mean(dt_preds == y_test))

    # ---- Logistic Regression -----------------------------------------------
    # max_iter=1000 ensures convergence on high-dimensional or correlated
    # feature sets (e.g., GLCM texture + color statistics together).
    lr_clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr_clf.fit(X_train, y_train)
    lr_preds = lr_clf.predict(X_test)
    lr_acc = float(np.mean(lr_preds == y_test))

    # ---- Human-readable output ---------------------------------------------
    separator = '=' * 60
    print(f"\n{separator}")
    print(f"  TOOL: {tool_name.upper()}")
    print(separator)

    # Print the full decision tree as plain text -- this is the key artifact
    # for hackathon judges and clinicians to audit the model logic.
    print("\n[DECISION TREE -- Human-readable rules]")
    tree_text = export_text(dt_clf, feature_names=feature_names)
    print(tree_text)

    # Classification reports include per-class precision, recall, F1 -- the
    # metrics clinicians care about (e.g., recall = sensitivity).
    print("[Decision Tree] Classification Report:")
    print(
        classification_report(
            y_test, dt_preds, target_names=class_names, zero_division=0
        )
    )

    print("[Decision Tree] Confusion Matrix:")
    print(confusion_matrix(y_test, dt_preds))
    print(f"[Decision Tree] Test Accuracy: {dt_acc:.4f}\n")

    print("[Logistic Regression] Classification Report:")
    print(
        classification_report(
            y_test, lr_preds, target_names=class_names, zero_division=0
        )
    )

    print("[Logistic Regression] Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))
    print(f"[Logistic Regression] Test Accuracy: {lr_acc:.4f}\n")

    # ---- Feature importance plot (Decision Tree) ---------------------------
    # Gini-based feature importances show which biomarkers the tree relies on
    # most.  Saving as PNG makes it easy to drop into reports/slides.
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 1.2), 5))
    importances = dt_clf.feature_importances_
    x_pos = np.arange(len(feature_names))
    ax.bar(x_pos, importances, color='steelblue', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Gini Importance')
    ax.set_title(f'{tool_name} -- Decision Tree Feature Importances')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    importance_path = os.path.join('models', f'{tool_name}_importance.png')
    plt.savefig(importance_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Feature importance saved -> {importance_path}")

    # ---- Logistic Regression coefficient plot ------------------------------
    # For binary classifiers, coef_ has shape (1, n_features).
    # For multi-class (OvR), shape is (n_classes, n_features) -- plot class 0
    # coefficients as a representative example; full coefficients are in the
    # saved .pkl and can be inspected post-training.
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 1.2), 5))
    coef_row = lr_clf.coef_[0]  # first row -- binary or OvR class-0
    colors = ['tomato' if c < 0 else 'seagreen' for c in coef_row]
    x_pos = np.arange(len(feature_names))
    ax.bar(x_pos, coef_row, color=colors, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coefficient value')
    ax.set_title(
        f'{tool_name} -- Logistic Regression Coefficients\n'
        f'({"binary" if lr_clf.coef_.shape[0] == 1 else "OvR class-0"})'
    )
    ax.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    coef_path = os.path.join('models', f'{tool_name}_lr_coef.png')
    plt.savefig(coef_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] LR coefficients saved -> {coef_path}")

    # ---- Persist models ----------------------------------------------------
    dt_path = os.path.join('models', f'{tool_name}_dt.pkl')
    lr_path = os.path.join('models', f'{tool_name}_lr.pkl')
    joblib.dump(dt_clf, dt_path)
    joblib.dump(lr_clf, lr_path)
    print(f"[SAVE] {dt_path}")
    print(f"[SAVE] {lr_path}")
    print(separator + '\n')

    return dt_clf, lr_clf, dt_acc


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_pretrained(tool_name: str, model_type: str = 'dt'):
    """
    Load and return a previously trained model from disk.

    Parameters
    ----------
    tool_name  : Key from TOOL_CONFIG (e.g. 'melanoma').
    model_type : 'dt' for DecisionTree or 'lr' for LogisticRegression.

    Returns
    -------
    Loaded sklearn estimator object.

    Raises
    ------
    FileNotFoundError with an actionable message if the .pkl does not exist.
    """
    if model_type not in ('dt', 'lr'):
        raise ValueError(
            f"model_type must be 'dt' or 'lr', got '{model_type}'."
        )

    model_path = os.path.join('models', f'{tool_name}_{model_type}.pkl')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found: '{model_path}'\n"
            f"To train it, run:\n"
            f"    python model.py --tool {tool_name} "
            f"--data_dir <path/to/dataset>"
        )

    clf = joblib.load(model_path)
    print(
        f"[LOAD] Loaded {model_type.upper()} model for '{tool_name}' "
        f"from '{model_path}'"
    )
    return clf


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Command-line interface for the unified MediScan model trainer.

    Usage examples
    --------------
    Train a single tool:
        python model.py --tool melanoma --data_dir data/melanoma

    Train all tools (expects data_dir to contain sub-folders named after tools):
        python model.py --tool all --data_dir data/

    Load and inspect a saved model type:
        python model.py --tool jaundice --data_dir data/jaundice --model_type lr
    """
    all_tool_choices = list(TOOL_CONFIG.keys()) + ['all']

    parser = argparse.ArgumentParser(
        description=(
            'MediScan unified model trainer -- trains Decision Tree '
            '+ Logistic Regression.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--tool',
        choices=all_tool_choices,
        required=True,
        help=(
            f"Clinical tool to train. Choices: {all_tool_choices}. "
            "Use 'all' to train every tool whose data sub-folder exists "
            "under data_dir."
        )
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help=(
            "Path to dataset root. For a single tool: "
            "folder/class_name/images. "
            "For --tool all: data_dir/<tool_name>/class_name/images."
        )
    )
    parser.add_argument(
        '--model_type',
        choices=['dt', 'lr'],
        default='dt',
        help=(
            "Which model type to highlight (default: dt). "
            "Does not affect training; both are always saved."
        )
    )

    args = parser.parse_args()

    # Ensure models/ output directory exists before any training starts.
    os.makedirs('models', exist_ok=True)

    # Build the list of (tool_name, data_folder) pairs to train.
    if args.tool == 'all':
        # Discover tools by looking for sub-directories in data_dir whose
        # names match known tool keys.
        tools_to_train = []
        for tool_key in TOOL_CONFIG:
            candidate = os.path.join(args.data_dir, tool_key)
            if os.path.isdir(candidate):
                tools_to_train.append((tool_key, candidate))
            else:
                print(
                    f"[SKIP] '{candidate}' not found -- "
                    f"skipping tool '{tool_key}'."
                )
        if not tools_to_train:
            print(
                f"[ERROR] No tool data folders found under '{args.data_dir}'. "
                "Create sub-folders named after tool keys to use --tool all."
            )
            sys.exit(1)
    else:
        tools_to_train = [(args.tool, args.data_dir)]

    summary_rows = []

    for tool_name, data_folder in tools_to_train:
        print(f"\n{'#' * 60}")
        print(f"# Training: {tool_name}")
        print(f"# Data dir: {data_folder}")
        print(f"{'#' * 60}\n")

        cfg = TOOL_CONFIG[tool_name]
        feature_names = cfg['feature_names']

        try:
            X, y, class_names = load_dataset_from_folder(
                data_folder, tool_name
            )

            # Guard: need at least 2 classes to train a classifier.
            if len(np.unique(y)) < 2:
                print(
                    f"[ERROR] Only one class found for '{tool_name}' "
                    "-- need >=2. Skipping."
                )
                continue

            # Guard: need enough samples for stratified split.
            min_class_count = int(np.bincount(y).min())
            if min_class_count < 2:
                print(
                    f"[ERROR] At least one class in '{tool_name}' has only "
                    f"{min_class_count} sample(s). Need >=2 per class for "
                    "stratified split. Skipping."
                )
                continue

            dt_clf, lr_clf, test_acc = train_models(
                X, y, feature_names, tool_name, class_names
            )
            summary_rows.append((tool_name, len(X), test_acc, 'OK'))

        except Exception as exc:
            print(f"[FAILED] Tool '{tool_name}': {exc}")
            summary_rows.append((tool_name, 0, 0.0, f'FAILED: {exc}'))

    # ---- Final summary table -----------------------------------------------
    print('\n' + '=' * 60)
    print('  TRAINING SUMMARY')
    print('=' * 60)
    print(f"{'Tool':<25} {'Samples':>8} {'DT Acc':>8} {'Status'}")
    print('-' * 60)
    for tool_name, n_samples, acc, status in summary_rows:
        print(f"{tool_name:<25} {n_samples:>8} {acc:>8.4f}   {status}")
    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
