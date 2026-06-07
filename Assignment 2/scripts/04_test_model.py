from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from data_processing_utilities import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_BASELINE_ACC = 25.0
PLOT_FILE = "performance_comparison.png"

EVAL_SPLITS = [
    ("Intra-Subject", "intra_test"),
    ("Cross-Subject (test1)", "cross_test1"),
    ("Cross-Subject (test2)", "cross_test2"),
    ("Cross-Subject (test3)", "cross_test3"),
]

MODELS = [
    {
        "label": "1D CNN",
        "intra_checkpoint": "cnn1d_intra",
        "cross_checkpoint": "cnn1d_cross",
        "layout": "cnn1d",
        "color": "#4B8BBE",
    },
    {
        "label": "EEGNet",
        "intra_checkpoint": "eegnet_intra",
        "cross_checkpoint": "eegnet_cross",
        "layout": "eegnet",
        "color": "#3CB371",
    },
]


def evaluate_split(checkpoint_name, test_split, layout):
    model_path = RESULTS_DIR / checkpoint_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    X_test, y_test, _ = load_split(test_split, layout=layout)
    model = load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Results for {checkpoint_name} on {test_split}:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy * 100


def evaluate_all_models():
    """Returns {model_label: [acc per EVAL_SPLITS category]}"""
    all_results = {}

    for model_config in MODELS:
        print(f"--- Evaluating {model_config['label']} ---")
        accuracies = []

        for _, test_split in EVAL_SPLITS:
            checkpoint = (
                model_config["intra_checkpoint"]
                if test_split == "intra_test"
                else model_config["cross_checkpoint"]
            )
            accuracies.append(
                evaluate_split(checkpoint, test_split, layout=model_config["layout"])
            )

        all_results[model_config["label"]] = accuracies

    return all_results


def generate_grouped_plot(all_results):
    categories = [name for name, _ in EVAL_SPLITS]
    x = np.arange(len(categories))
    n_models = len(MODELS)
    width = 0.8 / n_models

    plt.figure(figsize=(11, 6))

    for i, model_config in enumerate(MODELS):
        offset = (i - (n_models - 1) / 2) * width
        accuracies = all_results[model_config["label"]]
        bars = plt.bar(
            x + offset,
            accuracies,
            width=width,
            label=model_config["label"],
            color=model_config["color"],
        )

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.5,
                f"{yval:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.axhline(
        y=RANDOM_BASELINE_ACC,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Random ({RANDOM_BASELINE_ACC:.0f}%)",
    )

    plt.xticks(x, categories)
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(
        "Classification Performance: 1D CNN vs EEGNet",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = RESULTS_DIR / PLOT_FILE
    plt.savefig(chart_path, dpi=300)
    print(f"Saved plot to: {chart_path}")
    plt.show()


def main():
    results = evaluate_all_models()
    generate_grouped_plot(results)


if __name__ == "__main__":
    main()
