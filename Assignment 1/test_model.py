import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from train_model import load_data, recursive_predict, persistence_baseline
import os



if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Final GRU settings based on window-size tuning
    best_window = 510
    epochs = 50
    
    """
    1. LOAD TRAINING DATA (FOR SCALER)
    """
    print('loading Xtrain...')
    train_series = load_data("Xtrain.mat", "Xtrain")

    scaler = MinMaxScaler() 
    scaled_data = scaler.fit_transform( 
        train_series.reshape(-1, 1)
        )
    
    """
    2. LOAD TRAINED MODEL
    """
    model_path = f"results/final_GRU_window{best_window}_epochs{epochs}.keras"
    print("Loading trained model...")
    model = load_model(model_path)

    """
    3. RECURSIVE PREDICTIONS (200 STEPS)
    """
    print("Making recursive 200 step prediction")
    future_preds_scaled = recursive_predict(
        model=model,
        scaled_data=scaled_data,
        window_size=best_window,
        steps=200
    )
    future_preds_original = scaler.inverse_transform(future_preds_scaled)

    """
    4. LOAD TEST DATA
    """
    print("Loading Xtest...")
    test_series = load_data("Xtest.mat","Xtest").reshape(-1,1)
   
    """
    5. BASELINE PREDICTIONS --> assumes the future will stay the same as the past 
    """
    last_train_value = train_series[-1]
    baseline_preds = persistence_baseline(
        last_value = last_train_value,
        steps=len(test_series)
    )
    baseline_mse = mean_squared_error(test_series, baseline_preds)
    baseline_mae = mean_absolute_error(test_series, baseline_preds)

    print("\nBaseline results")
    print("Baseline MSE:", baseline_mse)
    print("Baseline MAE:", baseline_mae)

    """
    6. EVALUATOIN
    """
    mse = mean_squared_error(test_series, future_preds_original)
    mae = mean_absolute_error(test_series, future_preds_original)
    print("\nTest results")
    print("MSE:", mse)
    print("MAE:", mae)

    """
    7. SAVE RESULTS
    """
    pred_path = "results/test_predictions.csv"
    np.savetxt(
        pred_path,
        future_preds_original,
        delimiter=","
    )
    print("Saved predictions to:", pred_path)

    """
    8. PLOT RESULTS
    """
    plt.figure(figsize=(10, 4))

    plt.plot(test_series, label="True Xtest")
    plt.plot(future_preds_original, label="Recursive predictions")
    plt.plot(baseline_preds, label="Persistence baseline")
    plt.title("Test vs Predictions (GRU Recursive Forecast)")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plot_path = "results/test_plot.png"

    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", plot_path)
