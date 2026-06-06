# everyone should download their data locally, dont push because of the size (include in gitignore)
# 1. Read .h5 files
# 2. Extract label from filename
# 3. Downsample by factor 10
# 4. Z-score normalize per sensor per file
# 5. Split into 2-second windows with 1-second overlap
# 6. Save processed .npz files

from pathlib import Path
from collections import Counter

import h5py
import numpy as np


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


SAMPLE_RATE = 2034
DOWNSAMPLE_FACTOR = 10

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0

# EEGNet reference design assumes 128 Hz; scale temporal kernels/pooling to match.
EEGNET_REFERENCE_RATE = 128.0
EEGNET_REF_POOL_SIZE1 = 4
EEGNET_REF_POOL_SIZE2 = 8
EEGNET_REF_SEP_KERNEL = 16

# make really small due to MEG values being ~10^-12
EPS = 1e-20


# Labels
LABEL_MAP = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3,
}

LABEL_NAMES = {
    0: "rest",
    1: "story_math",
    2: "working_memory",
    3: "motor",
}


# Dataset folders to preprocess

DATASETS = {
    "intra_train": RAW_DATA_DIR / "Intra" / "train",
    "intra_test": RAW_DATA_DIR / "Intra" / "test",
    "cross_train": RAW_DATA_DIR / "Cross" / "train",
    "cross_test1": RAW_DATA_DIR / "Cross" / "test1",
    "cross_test2": RAW_DATA_DIR / "Cross" / "test2",
    "cross_test3": RAW_DATA_DIR / "Cross" / "test3",
}



def get_label(filename):
    """
    Extract class label from filename.
    """
    name = Path(filename).stem

    if name.startswith("rest"):
        return LABEL_MAP["rest"]

    if name.startswith("task_story_math"):
        return LABEL_MAP["task_story_math"]

    if name.startswith("task_working_memory"):
        return LABEL_MAP["task_working_memory"]

    if name.startswith("task_motor"):
        return LABEL_MAP["task_motor"]

    raise ValueError(f"Unknown label for file: {filename}")


def read_h5_file(file_path):
    """
    Read one .h5 file.

    Each .h5 file should contain exactly one dataset.
    """
    with h5py.File(file_path, "r") as f:
        dataset_name = list(f.keys())[0]
        matrix = f[dataset_name][()]

    return matrix.astype(np.float32)


def downsample(matrix, factor):
    """
    Downsample the time dimension.

    Input shape:
        sensors x time

    Output shape:
        sensors x reduced_time
    """
    return matrix[:, ::factor]


def zscore_per_sensor(matrix, eps=EPS):
    """
    Z-score normalize each sensor separately over time

    Bc MEG values are very small, eps must also be very small.
    """
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)

    std_safe = np.where(std < eps, 1.0, std)

    return (matrix - mean) / std_safe


def make_windows(matrix, label):
    """
    Split one recording into overlapping windows.

    Input:
        matrix shape: sensors x time

    Output:
        X shape: n_windows x window_timesteps x sensors
        y shape: n_windows
    """
    effective_sample_rate = SAMPLE_RATE / DOWNSAMPLE_FACTOR

    window_size = int(WINDOW_SECONDS * effective_sample_rate)
    stride_size = int(STRIDE_SECONDS * effective_sample_rate)

    n_sensors, n_timesteps = matrix.shape

    X_windows = []
    y_windows = []

    for start in range(0, n_timesteps - window_size + 1, stride_size):
        end = start + window_size

        window = matrix[:, start:end]

        # Convert from sensors x time to time x sensors
        # is better output for models later on
        window = window.T

        X_windows.append(window)
        y_windows.append(label)

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int64)

    return X_windows, y_windows


def preprocess_folder(dataset_name, input_folder):
    """
    Preprocess all .h5 files in one folder and save them as one .npz file.
    """
    input_folder = Path(input_folder)
    output_file = PROCESSED_DATA_DIR / f"{dataset_name}.npz"

    files = sorted(input_folder.glob("*.h5"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {input_folder}")

    print("\n" + "=" * 70)
    print(f"Preprocessing: {dataset_name}")
    print(f"Input folder: {input_folder}")
    print(f"Number of files: {len(files)}")

    all_X = []
    all_y = []
    all_source_files = []

    original_label_counts = Counter()
    window_label_counts = Counter()

    for file_path in files:
        label = get_label(file_path.name)
        original_label_counts[label] += 1

        matrix = read_h5_file(file_path)

        # downsample time dimension
        matrix = downsample(matrix, DOWNSAMPLE_FACTOR)

        # normalize per sensor, per file
        matrix = zscore_per_sensor(matrix)

        # make overlapping windows
        X, y = make_windows(matrix, label)

        all_X.append(X)
        all_y.append(y)
        all_source_files.extend([file_path.name] * len(y))

        window_label_counts[label] += len(y)

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    source_files = np.array(all_source_files)

    n_windows, n_timesteps, n_chans = X_final.shape
    effective_sample_rate = SAMPLE_RATE / DOWNSAMPLE_FACTOR
    rate_scale = effective_sample_rate / EEGNET_REFERENCE_RATE
    eegnet_kern_length = int(effective_sample_rate / 2)
    eegnet_pool_size1 = max(1, round(EEGNET_REF_POOL_SIZE1 * rate_scale))
    eegnet_pool_size2 = max(1, round(EEGNET_REF_POOL_SIZE2 * rate_scale))
    eegnet_sep_kernel_length = max(1, round(EEGNET_REF_SEP_KERNEL * rate_scale))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        X=X_final,
        y=y_final,
        source_files=source_files,
        sample_rate=SAMPLE_RATE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        effective_sample_rate=effective_sample_rate,
        window_seconds=WINDOW_SECONDS,
        stride_seconds=STRIDE_SECONDS,
        # layout metadata for EEGNet (X stays time x channels; reshape at load time)
        x_layout="samples_time_channels",
        n_timesteps=n_timesteps,
        n_chans=n_chans,
        eegnet_input_shape=np.array([n_chans, n_timesteps, 1], dtype=np.int64),
        eegnet_kern_length=eegnet_kern_length,
        eegnet_pool_size1=eegnet_pool_size1,
        eegnet_pool_size2=eegnet_pool_size2,
        eegnet_sep_kernel_length=eegnet_sep_kernel_length,
    )

    print(f"Saved to: {output_file}")
    print(f"X shape (baseline): {X_final.shape}  -> (n_windows, timesteps, channels)")
    print(
        f"EEGNet layout:      ({n_windows}, {n_chans}, {n_timesteps}, 1)  "
        f"[Chans={n_chans}, Samples={n_timesteps}, kernLength={eegnet_kern_length}, "
        f"pool=({eegnet_pool_size1}, {eegnet_pool_size2}), sepKernel={eegnet_sep_kernel_length}]"
    )
    print(f"y shape: {y_final.shape}")

    print("\nOriginal file counts:")
    for label_id, count in sorted(original_label_counts.items()):
        print(f"  {LABEL_NAMES[label_id]}: {count}")

    print("\nWindow counts:")
    for label_id, count in sorted(window_label_counts.items()):
        print(f"  {LABEL_NAMES[label_id]}: {count}")



def main():
    print("MEG preprocessing")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data dir: {RAW_DATA_DIR}")
    print(f"Processed data dir: {PROCESSED_DATA_DIR}")

    print("\nSettings:")
    print(f"  Original sample rate: {SAMPLE_RATE} Hz")
    print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  Effective sample rate: {SAMPLE_RATE / DOWNSAMPLE_FACTOR:.1f} Hz")
    print(f"  Window size: {WINDOW_SECONDS} seconds")
    print(f"  Stride: {STRIDE_SECONDS} seconds")

    for dataset_name, folder in DATASETS.items():
        preprocess_folder(dataset_name, folder)

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()