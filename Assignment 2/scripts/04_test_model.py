from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

def evaluate_split(model_name, test_split):
    data_path = PROCESSED_DATA_DIR / f"{test_split}.npz"
    X_test, y_test = np.load(data_path)["X"], np.load(data_path)["y"]
    
    model_path = RESULTS_DIR / f"{model_name}.keras"
    model = load_model(model_path)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Results for {model_name} on {test_split}:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy * 100

def generate_performance_plot(results_dict):
    categories = list(results_dict.keys())
    accuracies = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#4B8BBE', '#FF0000', '#ffd343', '#3CB371']
    bars = plt.bar(categories, accuracies, color=colors, width=0.55)
    
    # baseline
    plt.axhline(y=25.0, color='black', linestyle='--', linewidth=1.5, label='Random Baseline (25.00%)')
    plt.text(
    x=-0.4, 
    y=26.5, 
    s="Random Baseline (25.00%)", 
    color='black', 
    fontsize=10, 
    fontweight='bold',
    va='bottom'
)
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Classification Performance: Intra-Subject vs. Cross-Subject', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save graph to results
    chart_path = RESULTS_DIR / "performance_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.show()

def main():
    performance_metrics = {}
    
    print("--- Evaluating Intra-Subject Baseline ---")
    performance_metrics['Intra-Subject'] = evaluate_split("cnn1d_intra", "intra_test")
    
    print("--- Evaluating Cross-Subject Generalization ---")
    cross_splits = ["cross_test1", "cross_test2", "cross_test3"] 
    for split in cross_splits:
        label = f"Cross-Subject ({split[-5:]})" # e.g., Cross-Subject (test1)
        performance_metrics[label] = evaluate_split("cnn1d_cross", split)
        
    print("--- Generating Evaluation Metrics Chart ---")
    generate_performance_plot(performance_metrics)

if __name__ == "__main__":
    main()