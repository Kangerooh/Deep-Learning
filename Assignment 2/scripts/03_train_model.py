import numpy as np

data = np.load("data/processed/intra_train.npz")

X = data["X"]
y = data["y"]

print(X.shape)
print(y.shape)
print(X.mean(), X.std())
print(np.unique(y, return_counts=True))