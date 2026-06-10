from pathlib import Path



from tensorflow.keras.layers import (

    BatchNormalization,

    Conv1D,

    Dense,

    Dropout,

    GlobalAveragePooling1D,

    Input,

    MaxPooling1D,

)

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam



from data_processing_utilities import load_split

from EEGnet import EEGNet



PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"



N_CLASSES = 4

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-3


def get_compile_kwargs(lr=LEARNING_RATE):
    return{
        "optimizer": Adam(learning_rate=LEARNING_RATE),

        "loss": "sparse_categorical_crossentropy",

        "metrics": ["accuracy"],
    }






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

    model.compile(**get_compile_kwargs())

    return model





def build_eegnet(meta, n_classes=N_CLASSES):

    model = EEGNet(

        nb_classes=n_classes,

        Chans=meta["n_chans"],

        Samples=meta["n_timesteps"],

        kernLength=meta["eegnet_kern_length"],

        poolSize1=meta["eegnet_pool_size1"],

        poolSize2=meta["eegnet_pool_size2"],

        sepKernelLength=meta["eegnet_sep_kernel_length"],

    )

    model.compile(**get_compile_kwargs())

    return model
# tuned hyperparameters 
TUNED_PARAMS = {
    "lr": 0.001,
    "dropout": 0.3,
    "F1": 16,
}

def build_eegnet_tuned(meta, n_classes=N_CLASSES):
    model = EEGNet(

        nb_classes=n_classes,

        Chans=meta["n_chans"],

        Samples=meta["n_timesteps"],

        kernLength=meta["eegnet_kern_length"],

        poolSize1=meta["eegnet_pool_size1"],

        poolSize2=meta["eegnet_pool_size2"],

        sepKernelLength=meta["eegnet_sep_kernel_length"],

        dropoutRate=TUNED_PARAMS["dropout"],

        F1=TUNED_PARAMS["F1"],

        D=2,
    
        F2=TUNED_PARAMS["F1"]*2,
    )
    model.compile(
        optimizer=Adam(
            learning_rate=TUNED_PARAMS["lr"]
        ),
        loss = "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model, model_output_name):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RESULTS_DIR / model_output_name
    model.save(model_path, save_format="tf")

    print(f"Saved model to: {model_path}")





def train_split(split_name, model_output_name, model_type):

    layout = "eegnet" if model_type == "eegnet" else "cnn1d"

    print(f"\n--- {model_type} training: {split_name} ---")



    X, y, meta = load_split(split_name, layout=layout)

    if model_type == "eegnet":

        print(

            f"  Chans={meta['n_chans']}, Samples={meta['n_timesteps']}, "

            f"kernLength={meta['eegnet_kern_length']}, "

            f"pool=({meta['eegnet_pool_size1']}, {meta['eegnet_pool_size2']}), "

            f"sepKernel={meta['eegnet_sep_kernel_length']}, X shape={X.shape}"

        )

        model = build_eegnet(meta)

    else:

        model = build_cnn1d(input_shape=X.shape[1:])



    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    save_model(model, model_output_name)





def run_experiments(model_choice):

    if model_choice in ("baseline", "all"):

        train_split("intra_train", "cnn1d_intra", "baseline")

        train_split("cross_train", "cnn1d_cross", "baseline")



    if model_choice in ("eegnet", "all"):

        train_split("intra_train", "eegnet_intra", "eegnet")

        train_split("cross_train", "eegnet_cross", "eegnet")





def ask_model_choice():

    print("Which model do you want to train?")

    print("  1 = baseline (1D CNN)")

    print("  2 = eegnet")

    print("  3 = all")



    options = {"1": "baseline", "2": "eegnet", "3": "all"}

    while True:

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice in options:

            return options[choice]

        print("Invalid choice. Please enter 1, 2, or 3.")





def main():

    run_experiments(ask_model_choice())





if __name__ == "__main__":

    main()


