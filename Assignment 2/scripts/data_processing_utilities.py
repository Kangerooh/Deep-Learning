"""Load processed splits. On disk, X is (n_windows, timesteps, channels)."""

from pathlib import Path

import numpy as np

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_split(split_name, layout="cnn1d"):
    """
    layout "cnn1d":  X is (N, timesteps, channels)
    layout "eegnet": X is (N, channels, timesteps, 1)

    Returns X, y, and meta with n_chans, n_timesteps, eegnet_kern_length.
    """
    if layout not in ("cnn1d", "eegnet"):
        raise ValueError(f'layout must be "cnn1d" or "eegnet", got {layout!r}')

    path = PROCESSED_DATA_DIR / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 02_preprocessing.py first.")

    data = np.load(path)
    X, y = data["X"], data["y"]
    _, n_timesteps, n_chans = X.shape

    if layout == "eegnet":
        X = np.transpose(X, (0, 2, 1))[..., np.newaxis]

    kern_length = int(data["eegnet_kern_length"]) if "eegnet_kern_length" in data else n_timesteps // 2
    meta = {
        "n_chans": n_chans,
        "n_timesteps": n_timesteps,
        "eegnet_kern_length": kern_length,
    }
    return X, y, meta
