# imports
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
data = loadmat("Xtrain.mat")
Xtrain = data["Xtrain"]
series = data["Xtrain"].flatten()
# print(series[:30])
# print(series.shape)

# hoi allemaal, ik test even git :D

# plot data to visualize patterns
# plt.plot(series)
# plt.title("Laser Time Series")
# plt.xlabel("Time step")
# plt.ylabel("Value")
# plt.show()

# Scale data to [0,1] for neural network training
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1))



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


# set window size
window_size = 10
X, y = create_windows(series_scaled, window_size)

# reshape to format of samples, timestamp, features
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and validations sets. 
split = int(0.8 * len(X)) # 80/20 split

X_train = X[:split]
X_val = X[split:]

y_train = y[:split]
y_val = y[split:]


if __name__ == "__main__":
    # example
    print("Example input window:", X[0].flatten())
    print("Target value:", y[0])

    # checks
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)