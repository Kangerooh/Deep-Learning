import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#from sklearn.model_selection import train_test_split #kan dit gebruikt worden voor crossvalidation?
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.model_selection import TimeSeriesSplit
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import csv


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

def build_GRU_model(X_train):
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
    
    #safe the plot
    plt.savefig(f'results/{title}.png', bbox_inches='tight')
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

    error_dict = {}
    
    window_step_size = args.window_step_size
    window_size = list(range(window_step_size, 990, window_step_size))
    
    for i in tqdm(window_size, desc="Testing Window Sizes"):
        X_train, y_train = create_windows(scaled_data, i)

        
        k = 10  # number of folds
        # model = build_LSTM_model(X_train)

        # LSE_per_fold = training_with_cross_validation(k, model, X_train, y_train)
        error_per_fold = training_with_cross_validation(k, X_train, y_train, "MSE")
        average_error = np.mean(error_per_fold)
        error_dict[i] = average_error
    
    visualize(error_dict, "GRU")
    
    #safe error dictionary
    with open('results/LSTM.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in error_dict.items():
            writer.writerow([key, value])

    
    
    
    #Xtest, ytest = readfiles()
    
    
    
    # # example
    # print("Example input window:", X[0].flatten())
    # print("Target value:", y[0])

    # # checks
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)
    # print("Train shape:", X_train.shape, y_train.shape)
    # print("Val shape:", X_val.shape, y_val.shape)