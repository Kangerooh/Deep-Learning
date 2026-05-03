import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#from sklearn.model_selection import train_test_split #kan dit gebruikt worden voor crossvalidation?
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


def load_data(data_file):
    data = loadmat(data_file)
    #Xtrain = data["Xtrain"]
    series = data["Xtrain"].flatten()
    
    return series

def scale_data(series):
    # Scale data to [0,1] for neural network training
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
    #TODO: cross validation!
    
    # reshape to format of samples, timestamp, features
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training and validations sets. 
    split = int(0.8 * len(X)) # 80/20 split

    X_train = X[:split]
    X_val = X[split:]

    y_train = y[:split]
    y_val = y[split:]
    
    return X_train, X_val, y_train, y_val


def build_LSTM_model(X_train):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True,
            input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def training_with_cross_validation(k, model, X_train, y_train):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors_per_fold = [] # Rename for clarity

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f'Fold {fold + 1}')
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
        
        # Get raw continuous predictions (do NOT use argmax)
        val_predictions = model.predict(X_val_fold)
        
        # Use MSE instead of LSE_score
        mse = mean_squared_error(y_val_fold, val_predictions)
        errors_per_fold.append(mse)
        print(f'MSE for fold {fold + 1}: {mse:.5f}')
        
    return errors_per_fold

def visualize(LSE_dict):
    x = sorted(LSE_dict.keys())
    y = [LSE_dict[i] for i in x]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Window Size")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error vs Window Size")
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_step_size", type=int, default=50,
                        help="Step size for window sizes (default: 50)") #when running it asks us the step size we want
    args = parser.parse_args()

    
    
    np.random.seed(0)
    data = load_data("Xtrain.mat")
    scaled_data = scale_data(data)
    
    #X_train, X_val, y_train, y_val = split_data(X,y)

    LSE_dict = {}
    
    window_step_size = args.window_step_size
    window_size = list(range(window_step_size, 990, window_step_size))
    
    for i in tqdm(window_size, desc="Testing Window Sizes"):
        X_train, y_train = create_windows(scaled_data, i)

        
        k = 10  # Number of folds
        model = build_LSTM_model(X_train)

        LSE_per_fold = training_with_cross_validation(k, model, X_train, y_train)
        average_LSE = np.mean(LSE_per_fold)
        LSE_dict[i] = average_LSE
    
    visualize(LSE_dict)
    print("test")
    
    
    
    #Xtest, ytest = readfiles()
    
    
    
    # # example
    # print("Example input window:", X[0].flatten())
    # print("Target value:", y[0])

    # # checks
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)
    # print("Train shape:", X_train.shape, y_train.shape)
    # print("Val shape:", X_val.shape, y_val.shape)