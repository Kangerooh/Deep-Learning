from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def load_data(data_file, key):
    """
    Loads .mat file and extracts the dataset as flattened 1D array
    """
    data = loadmat(data_file)
    #Xtrain = data["Xtrain"]
    series = data[key].flatten()
    
    return series

def scale_data(series):
    """ 
    Scale data to [0,1] for neural network training
    """
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1))
    return series_scaled
    

# print(series[:30])
# print(series.shape)

# plot data to visualize patterns
# plt.plot(series)
# plt.title("Laser Time Series")
# plt.xlabel("Time step")
# plt.ylabel("Value")
# plt.show()

def create_windows(data, window_size):
    """
    Creates a sliding window
    X = past k values
    y = next value
    """
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y)

def split_data(X,y): 
    """
    Reshape to format of samples, timestamp, features
    Splits into 80% training and 20% validation

    """
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X)) # 80/20 split

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]
    
    return X_train, X_val, y_train, y_val


def build_LSTM_model(X_train):
    """
    Builds and compiles a two layer LSTM network
    Uses dropout regulatization for time series prediction
    """
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True,
            input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def build_GRU_model(X_train):
    """
    Builds and compiles a two layer GRU ntework 
    Uses dropout regulatization for time series prediction
    """
    model = Sequential()
    model.add(GRU(units=128, return_sequences=True,
            input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(GRU(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def training_with_cross_validation(k, X_train, y_train, error_type):
    """
    Does time series cross-validation using TimeSeriesSplit
    Trains a new GRU model for each fold
    Calculates either MSE or MAE 
    """
    time_folds = TimeSeriesSplit(n_splits=k)
    errors_per_fold = [] 

    for fold, (train_index, val_index) in enumerate(time_folds.split(X_train)):
        print(f'Fold {fold + 1}')
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = build_GRU_model(X_train_fold) #reset model every fold
        model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
        
        # raw continuous predictions 
        val_predictions = model.predict(X_val_fold)
        
        # use MSE or MAE as error type
        if error_type == "MSE":
            mse = mean_squared_error(y_val_fold, val_predictions)
            errors_per_fold.append(mse)
            print(f'MSE for fold {fold + 1}: {mse:.5f}')
        elif error_type == "MAE":
            mae = mean_absolute_error(y_val_fold, val_predictions)
            errors_per_fold.append(mae)
            print(f'MAE for fold {fold + 1}: {mae:.5f}')
        
    return errors_per_fold

def visualize(error_dict, title):
    """
    Plots validation error against window sizes
    Highlights minimum error value 
    """
    x = sorted(error_dict.keys())
    y = [error_dict[i] for i in x]

    # to find the minimum value and its index
    min_error = min(y)
    min_x = x[y.index(min_error)]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', label='Error score')
    
    plt.ylim(bottom=0)  # bottom y axis is set to 0
    plt.xlim(left=0)    # same for x axis

    # a red dotted line for the minimum value and add this value to the right side
    plt.axhline(y=min_error, color='r', linestyle='--', linewidth=1, alpha=0.7)
    plt.text(1.01, min_error, f'Min Error: {min_error:.5f}', 
             color='red', va='center', fontweight='bold',
             transform=plt.gca().get_yaxis_transform())

    # vertical lines to have clearer graph
    plt.axvline(x=min_x, color='gray', linestyle=':', alpha=0.5)
    plt.text(min_x, min_error, f'  Size: {min_x}', color='blue', va='bottom')

    plt.xlabel("Window Size")
    plt.ylabel("Error Score")
    plt.title(f"Error Score vs Window Size: {title}")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    #save the plot
    plt.savefig(f'results/{title}.png', bbox_inches='tight')
    plt.show()


def train_final_gru_model(scaled_data, best_window, epochs=50, batch_size=32):
    """
    Creates training/validation datasets using the best window size
    Trains the final GRU model
    Evaluates its performance with MSE and MAE
    """
    # Create windows with best window size
    X, y = create_windows(scaled_data, best_window)

    # Split data chronologically
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Build and train final GRU model
    model = build_GRU_model(X_train)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Validation predictions
    val_pred = model.predict(X_val)

    # Validation errors
    mse = mean_squared_error(y_val, val_pred)
    mae = mean_absolute_error(y_val, val_pred)

    print("Final GRU model")
    print("Best window size:", best_window)
    print("Validation MSE:", mse)
    print("Validation MAE:", mae)

    return model, history, mse, mae


def recursive_predict(model, scaled_data, window_size, steps=200):
    """
    Predict future values recursively:
    each prediction is fed back as input for the next prediction.
    """
    input_window = scaled_data[-window_size:].copy()
    predictions = []

    for _ in range(steps):
        X_input = input_window.reshape(1, window_size, 1)

        pred = model.predict(X_input, verbose=0)
        pred_value = pred[0, 0]

        predictions.append(pred_value)

        # Remove oldest value, add prediction
        input_window = np.append(input_window[1:], [[pred_value]], axis=0)

    return np.array(predictions).reshape(-1, 1)

def persistence_baseline(last_value, steps=200):
    """
    Naive baseline forecast
    Predicts the last observed value repeatedly
    """
    return np.full((steps, 1), last_value)

if __name__ == "__main__":
    """
    Loads and scales data
    Trains final GRU model
    Evaluates performances
    Saves trained model
    Generates recursive future predictions
    Saves prediction reuslts and plots them
    """
    os.makedirs("results", exist_ok=True)

    # Final GRU settings based on window-size tuning
    best_window = 510
    epochs = 50
    batch_size = 32

    print('loading data:')
    series = load_data("Xtrain.mat", "Xtrain")


    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.reshape(-1, 1))

    X, y = create_windows(scaled_data, best_window)
    X_train, X_val, y_train, y_val = split_data(X, y)

    model = build_GRU_model(X_train)

    print("Training GRU model")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Predictions
    val_pred = model.predict(X_val)

    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    # Convert validation predictions and true values back to original scale
    val_pred_original = scaler.inverse_transform(val_pred)
    y_val_original = scaler.inverse_transform(y_val)

    # Errors on original scale
    val_mse_original = mean_squared_error(y_val_original, val_pred_original)
    val_mae_original = mean_absolute_error(y_val_original, val_pred_original)

    print("\nFinal GRU results on scaled validation data")
    print("Best window size:", best_window)
    print("epochs:", epochs)
    print("Validation MSE:", val_mse)
    print("Validation MAE:", val_mae)

    # On orignal data
    print("\nFinal GRU results on original validation data")
    print("Validation MSE original:", val_mse_original)
    print("Validation MAE original:", val_mae_original)

    model_path = f"results/final_GRU_window{best_window}_epochs{epochs}.keras"
    model.save(model_path)
    print("Saved model to:", model_path)

    print("Making recursive 200 step prediction")
    future_preds_scaled = recursive_predict(
        model=model,
        scaled_data=scaled_data,
        window_size=best_window,
        steps=200
    )

    future_preds_original = scaler.inverse_transform(future_preds_scaled)

    pred_path = f"results/final_GRU_window{best_window}_recursive_200.csv"
    np.savetxt(pred_path, future_preds_original, delimiter=",")
    print("Saved recursive predictions to:", pred_path)

    plt.figure(figsize=(10, 4))
    plt.plot(future_preds_original, label="Recursive GRU predictions")
    plt.title(f"Recursive Prediction: GRU, window={best_window}")
    plt.xlabel("Future time step")
    plt.ylabel("Predicted value")
    plt.legend()
    plt.grid(True)

    plot_path = f"results/final_GRU_window{best_window}_recursive_200.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", plot_path)