"""Inject simulated CW-like lines into REAL noise SFTs (labelled-noise files from the competition).

This is the training-data recipe both reference solutions rely on: the competition's labelled
train set is tiny, so positives are simulated and added to genuine detector noise (real gaps,
real spectral lines, real non-Gaussianity). The injected line is the same simplified model as
`data/synthetic.py` (linear spin-down + orbital Doppler + crude antenna modulation), NOT a full
PyFstat/LAL waveform. Swap `inject_line` for a PyFstat-based injector when you want fidelity.
"""
from __future__ import annotations

import numpy as np

from constants import FDOT_MAX_BINS, N_FREQ
from data.synthetic import inject_line


def noise_sigma(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust per-detector complex-noise std (median of |x|^2 = sigma^2 ln 2 for Gaussian noise)."""
    out = []
    for d in range(2):
        pm = np.abs(x[d][:, mask[d] > 0]) ** 2
        out.append(np.sqrt(np.median(pm) / np.log(2.0)) if pm.size else 1.0)
    return np.asarray(out, dtype=np.float64)


def make_injected_sample(noise: dict, rng: np.random.Generator, positive: bool, amp_range=(0.15, 0.8)) -> dict:
    """noise: {'x','mask','freq'} from `load_hdf5_sample`. Returns the same sample dict as synthetic."""
    x = noise["x"].copy()
    mask = noise["mask"]
    f_center = float(np.mean(noise["freq"]))
    amp, meta = 0.0, {}
    if positive:
        amp = float(rng.uniform(*amp_range))
        meta = dict(
            f0_bin=float(rng.uniform(40, N_FREQ - 40)),
            fdot_bins=float(rng.uniform(-0.8, 0.8) * FDOT_MAX_BINS),
            alpha=float(rng.uniform(0, 2 * np.pi)),
            delta=float(np.arcsin(rng.uniform(-1, 1))),
        )
        inject_line(x, rng, amp, f_center=f_center, scale=noise_sigma(x, mask), **meta)
        x *= mask[:, None, :]                      # no signal where the detector had no data
    return {"x": x, "mask": mask, "f_center": f_center, "label": int(positive), "strength": amp, "meta": meta}
