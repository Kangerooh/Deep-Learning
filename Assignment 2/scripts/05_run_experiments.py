"""
05_run_experiments.py

Orchestrates the full train -> evaluate pipeline for both models
(1D-CNN baseline + EEGNet) on both regimes (intra / cross), and writes
all artefacts to results/:

  - <tag>.keras                 trained model
  - <tag>_history.json          training/validation curves (per epoch)
  - <tag>_curves.png            loss + accuracy curves
  - <tag>_on_<test>_report.csv  per-class precision/recall/f1
  - <tag>_on_<test>_cm.png      confusion matrix
  - summary.csv                 one row per (model, test split) with test accuracy
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from data_processing_utilities import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ["rest", "story_math", "working_memory", "motor"]

VAL_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 8
RANDOM_STATE = 42


# --- pull the model builders from 03 (numeric module name -> import by path) ---
def _load_builders():
    spec = importlib.util.spec_from_file_location(
        "train_model", SCRIPTS_DIR / "03_train_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # safe: 03's main() is behind __main__ guard
    return mod.build_cnn1d, mod.build_eegnet


BUILD_CNN1D, BUILD_EEGNET = _load_builders()


# Experiment matrix: which checkpoint trains on what, and where it's tested.
EXPERIMENTS = [
    {
        "tag": "cnn1d_intra",
        "model_type": "baseline",
        "layout": "cnn1d",
        "train_split": "intra_train",
        "test_splits": ["intra_test"],
    },
    {
        "tag": "cnn1d_cross",
        "model_type": "baseline",
        "layout": "cnn1d",
        "train_split": "cross_train",
        "test_splits": ["cross_test1", "cross_test2", "cross_test3"],
    },
    {
        "tag": "eegnet_intra",
        "model_type": "eegnet",
        "layout": "eegnet",
        "train_split": "intra_train",
        "test_splits": ["intra_test"],
    },
    {
        "tag": "eegnet_cross",
        "model_type": "eegnet",
        "layout": "eegnet",
        "train_split": "cross_train",
        "test_splits": ["cross_test1", "cross_test2", "cross_test3"],
    },
]


def build_model(model_type, X, meta):
    if model_type == "eegnet":
        return BUILD_EEGNET(meta)
    return BUILD_CNN1D(input_shape=X.shape[1:])


def train_one(exp):
    print(f"\n=== TRAIN {exp['tag']} ({exp['model_type']}) on {exp['train_split']} ===")
    X, y, meta = load_split(exp["train_split"], layout=exp["layout"])

    # Stratified hold-out so val set has all 4 classes (npz is ordered by file!)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_STATE
    )

    model = build_model(exp["model_type"], X, meta)
    es = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(RESULTS_DIR / exp["tag"], save_format="tf")
    with open(RESULTS_DIR / f"{exp['tag']}_history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    plot_curves(history.history, exp["tag"])
    return model


def plot_curves(hist, tag):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(hist["loss"], label="train")
    ax_loss.plot(hist["val_loss"], label="val")
    ax_loss.set_title(f"{tag} - loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_acc.plot(hist["accuracy"], label="train")
    ax_acc.plot(hist["val_accuracy"], label="val")
    ax_acc.set_title(f"{tag} - accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{tag}_curves.png", dpi=150)
    plt.close(fig)


def evaluate_one(model, exp, test_split, summary_rows):
    X_test, y_test, _ = load_split(test_split, layout=exp["layout"])
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"  {exp['tag']} on {test_split}: acc = {acc * 100:.2f}%")

    # per-class report -> csv
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / f"{exp['tag']}_on_{test_split}_report.csv"
    )

    # confusion matrix -> png
    plot_confusion(y_test, y_pred, f"{exp['tag']}_on_{test_split}")

    summary_rows.append(
        {
            "model": exp["tag"],
            "train_split": exp["train_split"],
            "test_split": test_split,
            "test_accuracy_pct": round(acc * 100, 2),
        }
    )


def plot_confusion(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(tag)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{tag}_cm.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for exp in EXPERIMENTS:
        model = train_one(exp)
        for test_split in exp["test_splits"]:
            evaluate_one(model, exp, test_split, summary_rows)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()