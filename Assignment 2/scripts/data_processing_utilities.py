"""Load processed splits. On disk, X is (n_windows, timesteps, channels)."""

from pathlib import Path

import numpy as np

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def compute_eegnet_params(effective_sample_rate):
    """Scale EEGNet temporal sizes from the 128 Hz reference design."""
    rate_scale = effective_sample_rate / 128.0
    return {
        "eegnet_kern_length": int(effective_sample_rate / 2),
        "eegnet_pool_size1": max(1, round(4 * rate_scale)),
        "eegnet_pool_size2": max(1, round(8 * rate_scale)),
        "eegnet_sep_kernel_length": max(1, round(16 * rate_scale)),
    }


def load_split(split_name, layout="cnn1d"):
    """
    layout "cnn1d":  X is (N, timesteps, channels)
    layout "eegnet": X is (N, channels, timesteps, 1)
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

    groups = data["groups"] if "groups" in data.files else None

    meta = {"n_chans": n_chans, "n_timesteps": n_timesteps, "groups": groups}
    if layout == "eegnet":
        meta.update(compute_eegnet_params(float(data["effective_sample_rate"])))

    return X, y, meta
