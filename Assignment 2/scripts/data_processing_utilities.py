"""Load processed splits. On disk, X is (n_windows, timesteps, channels)."""

from pathlib import Path

import numpy as np

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

EEGNET_REFERENCE_RATE = 128.0
EEGNET_REF_POOL_SIZE1 = 4
EEGNET_REF_POOL_SIZE2 = 8
EEGNET_REF_SEP_KERNEL = 16


def _eegnet_params_from_npz(data, n_timesteps):
    """Read EEGNet hyperparameters from .npz, with fallback for older files."""
    effective_rate = (
        float(data["effective_sample_rate"])
        if "effective_sample_rate" in data
        else EEGNET_REFERENCE_RATE
    )
    rate_scale = effective_rate / EEGNET_REFERENCE_RATE

    kern_length = (
        int(data["eegnet_kern_length"])
        if "eegnet_kern_length" in data
        else n_timesteps // 2
    )
    pool_size1 = (
        int(data["eegnet_pool_size1"])
        if "eegnet_pool_size1" in data
        else max(1, round(EEGNET_REF_POOL_SIZE1 * rate_scale))
    )
    pool_size2 = (
        int(data["eegnet_pool_size2"])
        if "eegnet_pool_size2" in data
        else max(1, round(EEGNET_REF_POOL_SIZE2 * rate_scale))
    )
    sep_kernel_length = (
        int(data["eegnet_sep_kernel_length"])
        if "eegnet_sep_kernel_length" in data
        else max(1, round(EEGNET_REF_SEP_KERNEL * rate_scale))
    )

    return {
        "eegnet_kern_length": kern_length,
        "eegnet_pool_size1": pool_size1,
        "eegnet_pool_size2": pool_size2,
        "eegnet_sep_kernel_length": sep_kernel_length,
    }


def load_split(split_name, layout="cnn1d"):
    """
    layout "cnn1d":  X is (N, timesteps, channels)
    layout "eegnet": X is (N, channels, timesteps, 1)

    Returns X, y, and meta with EEGNet shape/kernel parameters.
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

    meta = {
        "n_chans": n_chans,
        "n_timesteps": n_timesteps,
        **_eegnet_params_from_npz(data, n_timesteps),
    }
    return X, y, meta
