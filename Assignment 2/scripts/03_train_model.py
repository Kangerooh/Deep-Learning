from pathlib import Path

import numpy as np
"""
First idea for the model is to use EEGNet. This is a model that is specifically designed for EEG data.
It should be a good fit for the data that we want to predict and process.
https://arxiv.org/abs/1611.08024
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "processed" / "intra_train.npz"

data = np.load(data_path)
X = data["X"]
y = data["y"]

print(X.shape)
print(y.shape)
print(X.mean(), X.std())
print(np.unique(y, return_counts=True))