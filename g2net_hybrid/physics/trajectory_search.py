"""Power-sum trajectory search (the physics branch core).

For every candidate trajectory   bin(t) = f0 + fdot * (t / T) + doppler(t)   we sum the excess
power (P - 1) the trajectory collects from both detectors and normalise by sqrt(#samples):

        z(f0, fdot, sky) = sum_{d,t} (P_d[bin(t), t] - 1) / sqrt(N)

For pure noise (P ~ Exp(1)) z has zero mean and unit variance, so z is a significance.
This is a simplified incoherent version of the idea in Koda's power_sum code: it uses linear
spin-down + an optional orbital-Doppler template bank, but no amplitude weighting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_fdot_grid(max_bins: float, n: int) -> np.ndarray:
    return np.linspace(-max_bins, max_bins, n)


def _sheared_sum(excess: np.ndarray, counts: np.ndarray, shifts: np.ndarray):
    """For every start row f0: sum excess[f0 + shifts[t], t] over t (out-of-band samples dropped).

    excess: (F, T) contiguous, counts: (T,), shifts: (T,) int.  Returns (num[F], n[F]).
    """
    f, t = excess.shape
    idx = np.arange(f)[:, None] + shifts[None, :]
    valid = (idx >= 0) & (idx < f)
    flat = np.clip(idx, 0, f - 1) * t + np.arange(t)[None, :]
    num = np.where(valid, excess.ravel()[flat], 0.0).sum(1)
    n = (valid * counts[None, :]).sum(1)
    return num, n


@dataclass
class SearchResult:
    z_map: np.ndarray          # (n_fdot, F) combined significance for the best template (NaN = invalid)
    zH_map: np.ndarray         # per-detector maps for the best template
    zL_map: np.ndarray
    best_template: int
    template_max: np.ndarray   # (K,) max z for every template
    fdots: np.ndarray
    n_total: float


def search(
    sums: np.ndarray,
    counts: np.ndarray,
    fdots: np.ndarray,
    offsets: np.ndarray | None = None,
    min_frac: float = 0.5,
) -> SearchResult:
    """sums (2, F, T), counts (2, T), fdots in bins/full-span, offsets (K, T) or (K, 2, T) in bins."""
    d, f, t = sums.shape
    excess = np.ascontiguousarray(sums - counts[:, None, :])
    t_rel = (np.arange(t) + 0.5) / t
    n_total = float(counts.sum())

    if offsets is None:
        offsets = np.zeros((1, t))
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim == 2:
        offsets = np.repeat(offsets[:, None, :], d, axis=1)
    n_tmpl = offsets.shape[0]

    template_max = np.full(n_tmpl, -np.inf)
    best = None
    for k in range(n_tmpl):
        num = np.zeros((d, len(fdots), f))
        cnt = np.zeros((d, len(fdots), f))
        for i, fd in enumerate(fdots):
            for det in range(d):
                shifts = np.rint(fd * t_rel + offsets[k, det]).astype(np.int64)
                num[det, i], cnt[det, i] = _sheared_sum(excess[det], counts[det], shifts)
        ntot = cnt.sum(0)
        z = np.where(ntot >= min_frac * n_total, num.sum(0) / np.sqrt(np.maximum(ntot, 1.0)), np.nan)
        if np.isfinite(z).any():
            template_max[k] = np.nanmax(z)
        if best is None or template_max[k] > template_max[best[0]]:
            best = (k, z, num, cnt)

    k, z, num, cnt = best
    with np.errstate(invalid="ignore", divide="ignore"):
        zh = np.where(cnt[0] > 0, num[0] / np.sqrt(np.maximum(cnt[0], 1.0)), np.nan)
        zl = np.where(cnt[1] > 0, num[1] / np.sqrt(np.maximum(cnt[1], 1.0)), np.nan)
    return SearchResult(z, zh, zl, int(k), template_max, np.asarray(fdots), n_total)
