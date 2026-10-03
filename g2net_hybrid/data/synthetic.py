"""Synthetic H1/L1 SFT generator for development, tests and ablations.

NOT a substitute for the competition's realistic data: noise is white Gaussian, gaps are
random, and the antenna response is a crude daily modulation. Use it to validate the pipeline,
then switch the cache builder to --source kaggle.
"""
from __future__ import annotations

import numpy as np

from constants import GPS_T0, N_FREQ, N_TIME, T_SFT, FDOT_MAX_BINS
from physics.doppler import orbital_offsets_bins


def random_mask(rng: np.random.Generator, coverage: float) -> np.ndarray:
    m = np.ones(N_TIME, dtype=np.float32)
    target = int((1.0 - coverage) * N_TIME)
    while (m == 0).sum() < target:
        length = int(rng.integers(5, 150))
        start = int(rng.integers(0, N_TIME - length))
        m[start:start + length] = 0
    return m


def inject_line(x, rng, amp, f0_bin, fdot_bins, alpha, delta, f_center, scale=(1.0, 1.0)):
    """Add a drifting, Doppler-modulated line to both detectors (in place).

    `amp` is relative to the per-detector noise std `scale` (1.0 for unit-variance synthetic noise)."""
    t_gps = GPS_T0 + T_SFT * np.arange(N_TIME)
    traj = f0_bin + fdot_bins * (np.arange(N_TIME) / N_TIME) + orbital_offsets_bins(alpha, delta, f_center, t_gps)
    day = 2 * np.pi * t_gps / 86164.0905
    cols = np.arange(N_TIME)
    for det in range(2):
        a = amp * scale[det] * (0.4 + 0.6 * np.abs(np.cos(day + rng.uniform(0, 2 * np.pi))))   # crude antenna modulation
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, N_TIME))
        base = np.floor(traj).astype(int)
        for dk in (-1, 0, 1, 2):                      # sinc leakage into neighbouring bins
            k = base + dk
            w = np.sinc(k - traj)
            ok = (k >= 0) & (k < N_FREQ)
            x[det, k[ok], cols[ok]] += (a * w * phase)[ok]
    return traj


def make_sample(rng: np.random.Generator, positive: bool, amp_range=(0.15, 0.8)) -> dict:
    f_center = float(rng.uniform(50.0, 500.0))
    x = (rng.standard_normal((2, N_FREQ, N_TIME)) + 1j * rng.standard_normal((2, N_FREQ, N_TIME))) / np.sqrt(2)
    x = x.astype(np.complex64)
    mask = np.stack([random_mask(rng, rng.uniform(0.55, 0.95)) for _ in range(2)])

    amp = 0.0
    meta = {}
    if positive:
        amp = float(rng.uniform(*amp_range))
        meta = dict(
            f0_bin=float(rng.uniform(40, N_FREQ - 40)),
            fdot_bins=float(rng.uniform(-0.8, 0.8) * FDOT_MAX_BINS),
            alpha=float(rng.uniform(0, 2 * np.pi)),
            delta=float(np.arcsin(rng.uniform(-1, 1))),
        )
        inject_line(x, rng, amp, f_center=f_center, **meta)

    x *= mask[:, None, :]
    return {"x": x, "mask": mask, "f_center": f_center, "label": int(positive), "strength": amp, "meta": meta}
