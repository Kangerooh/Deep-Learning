from pathlib import Path

import numpy as np
from keras.layers import Input
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    MaxPooling1D,
)
from keras.models import Sequential
from keras.optimizers import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

N_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3


def load_npz(split_name):
    path = PROCESSED_DATA_DIR / f"{split_name}.npz"
    data = np.load(path)
    return data["X"], data["y"]


# build a 1D CNN baselinemodel
def build_cnn1d(input_shape, n_classes=N_CLASSES):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(32, kernel_size=7, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.5),
            Dense(n_classes, activation="softmax"),
        ],
        name="cnn1d_baseline",
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_and_save(split_name, model_output_name):
    print(f"\n--- Starting Training for: {split_name} ---")
    X_train, y_train = load_npz(split_name)
    
    model = build_cnn1d(input_shape=X_train.shape[1:])
    
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / f"{model_output_name}.keras"
    model.save(model_path)
    print(f"Saved model to: {model_path}")

def main():
    # Experiment 1: Intra-subject Training
    train_and_save(split_name="intra_train", model_output_name="cnn1d_intra")
    
    # Experiment 2: Cross-subject Training
    train_and_save(split_name="cross_train", model_output_name="cnn1d_cross")



if __name__ == "__main__":
    main()
