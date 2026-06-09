"""
Hyperparamter search for EEGNET.
This file performs a grid search 
Load data --> split data --> create all combis 
of hyperparameters (learnign rate, dropout rate, F1)--> build, train, evaluate each model
--> save all results--> print best combi

hyperparameters being tested:
LR : 0.001, 0.0005
Dropout: 0.3, 0.5
F1: 8, 16
total number of experiments = 2 x 2 x 2 = 8

"""

from pathlib import Path
import json
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from data_processing_utilities import load_split
from EEGnet import EEGNet

# paths 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# load the data, only intra train
X, y, meta = load_split("intra_train", layout="eegnet")

# split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# building the model, with hyperparameters learning rate, dropout and F1
def build_model(lr, dropout, F1):
    model = EEGNet(
        nb_classes=4,
        Chans=meta["n_chans"],
        Samples=meta["n_timesteps"],
        kernLength=meta["eegnet_kern_length"],
        poolSize1=meta["eegnet_pool_size1"],
        poolSize2=meta["eegnet_pool_size2"],
        sepKernelLength=meta["eegnet_sep_kernel_length"],
        dropoutRate=dropout,
        F1=F1,
        D=2,
        F2=F1 * 2,
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# hyperparameter grid 
grid = list(itertools.product(
    [1e-3, 5e-4], # learning rate
    [0.3, 0.5], # dropout
    [8, 16],  # F1
))

results = []

# train
for i, (lr, dropout, F1) in enumerate(grid):

    print(f"\n=== RUN {i+1}/{len(grid)} ===") # shows nice bar for progression :)
    print(f"lr={lr}, dropout={dropout}, F1={F1}")

    model = build_model(lr, dropout, F1) # built eegnet using settings

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),  
        epochs=25,
        batch_size=32,
        verbose=1,
    )

    best_val_acc = float(np.max(history.history["val_accuracy"]))

    results.append({
        "lr": lr,
        "dropout": dropout,
        "F1": F1,
        "best_val_accuracy": best_val_acc
    })


# save results
with open(RESULTS_DIR / "hyperparameter_results.json", "w") as f:
    json.dump(results, f, indent=2)

#pritn best config
best = max(results, key=lambda x: x["best_val_accuracy"])

print("\n====================")
print("BEST CONFIG:")
print(best)
print("====================")