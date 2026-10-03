"""Loading, normalising and pooling SFT data (H1 + L1).

Everything downstream works from two arrays per sample:
    x    : complex64 (2, 360, 5760)   SFT coefficients placed on the 1800 s time grid
    mask : float32   (2, 5760)        1 where an SFT exists, 0 where there is a gap
Detector order is always (H1, L1).
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from constants import CNN_TIME_BINS, GPS_T0, N_FREQ, N_TIME, T_SFT


# --------------------------------------------------------------------------
# Raw data -> fixed grid
# --------------------------------------------------------------------------
def place_on_grid(h: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put SFT columns at their nearest 1800 s slot. h: (360, n), t: (n,) GPS seconds."""
    x = np.zeros((N_FREQ, N_TIME), dtype=np.complex64)
    m = np.zeros(N_TIME, dtype=np.float32)
    slots = np.rint((np.asarray(t, dtype=np.float64) - GPS_T0) / T_SFT).astype(int)
    ok = (slots >= 0) & (slots < N_TIME)
    x[:, slots[ok]] = h[:, ok]
    m[slots[ok]] = 1.0
    return x, m


def load_hdf5_sample(path: str, file_id: str | None = None) -> dict:
    """Load one competition HDF5 file -> {'x', 'mask', 'freq'}."""
    import h5py

    file_id = file_id or os.path.splitext(os.path.basename(path))[0]
    xs, ms = [], []
    with h5py.File(path, "r") as f:
        g = f[file_id]
        freq = g["frequency_Hz"][:]
        for det in ("H1", "L1"):
            h = g[det]["SFTs"][:] * 1e22          # same amplitude rescale as Koda
            t = g[det]["timestamps_GPS"][:]
            x, m = place_on_grid(h, t)
            xs.append(x)
            ms.append(m)
    return {"x": np.stack(xs), "mask": np.stack(ms), "freq": freq}


# --------------------------------------------------------------------------
# Normalisation: turn |SFT|^2 into "unit-mean noise" power
# --------------------------------------------------------------------------
def normalize_power(x: np.ndarray, mask: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """|x|^2 divided by a robust per-time-column scale, then a per-frequency-row scale.

    After this, pure Gaussian noise has power ~ Exponential(1) (mean 1, variance 1), which is
    what the physics statistic z = sum(P - 1) / sqrt(N) assumes. Missing slots are set to 0.
    """
    p = (np.abs(x) ** 2).astype(np.float32)
    present = mask[:, None, :] > 0
    p = np.where(present, p, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # median of Exp(1) is ln 2 * mean
        col = np.nanmedian(p, axis=1, keepdims=True) / np.log(2.0)     # (2, 1, N)
        col = np.where(np.isfinite(col) & (col > 0), col, 1.0)
        p = p / col
        row = np.nanmedian(p, axis=2, keepdims=True) / np.log(2.0)     # (2, F, 1)
        row = np.where(np.isfinite(row) & (row > 0), row, 1.0)
        p = p / row

    p = np.where(present, np.minimum(p, clip), 0.0)
    return p.astype(np.float32)


def pool_time(p: np.ndarray, mask: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum power and count valid slots in n_out equal time chunks.

    Returns sums (2, F, n_out) and counts (2, n_out).
    """
    d, f, n = p.shape
    k = n // n_out
    assert k * n_out == n, "n_out must divide N_TIME"
    sums = p.reshape(d, f, n_out, k).sum(-1)
    counts = mask.reshape(d, n_out, k).sum(-1)
    return sums.astype(np.float32), counts.astype(np.float32)


def pooled_mean(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mask-aware mean power; chunks with no data get the noise mean (1.0)."""
    c = counts[:, None, :]
    return np.where(c > 0, sums / np.maximum(c, 1.0), 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# CNN input image (4 channels), after the 12th-place G2Net solution
# --------------------------------------------------------------------------
def build_cnn_image(psds: np.ndarray) -> np.ndarray:
    """psds: (2, 360, T) -> (4, 360, T): [det0, det1, product, exp(product)].

    Logic follows `_create_input_image_from_psds` from the 12th-place repo
    (G2Net-Detecting-Continuous-Gravitational-Waves/src/data.py).
    """
    psds = psds.astype(np.float32).copy()
    inlier = psds < 10.0
    vsum = np.where(inlier, psds, 0.0).sum(1, keepdims=True)
    vcnt = inlier.astype(np.float32).sum(1, keepdims=True)
    psds = psds - vsum / (vcnt + 1e-10)

    inlier = psds < 10.0
    stds = np.array([psds[0, inlier[0]].std(), psds[1, inlier[1]].std()], dtype=np.float32)
    stds = np.where(stds > 0, stds, 1.0)
    psds = np.where(inlier, psds, 0.0) / stds[:, None, None]

    multiplied = (psds / 10.0 + 1.0).prod(0)
    multiplied_exp = np.exp((multiplied / 30.0) ** 2)
    multiplied = (multiplied - multiplied.mean()) / (multiplied.std() + 1e-10)
    multiplied_exp = (multiplied_exp - multiplied_exp.mean()) / (multiplied_exp.std() + 1e-10)
    return np.stack((*psds, multiplied, multiplied_exp)).astype(np.float32)


def sample_to_cnn_image(x: np.ndarray, mask: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    if p is None:
        p = normalize_power(x, mask)
    s, c = pool_time(p, mask, CNN_TIME_BINS)
    return build_cnn_image(pooled_mean(s, c))
