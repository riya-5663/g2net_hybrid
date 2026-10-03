"""Physics feature extractor: SFT summaries -> fixed-length vector for the fusion MLP."""
from __future__ import annotations

import numpy as np

from constants import FDOT_MAX_BINS, N_FREQ, PHYS_TIME_CHUNK
from physics.doppler import template_bank
from physics.trajectory_search import make_fdot_grid, search

FEATURE_NAMES = [
    "z_max",            # best combined significance over (f0, fdot, sky)
    "z_H", "z_L",       # per-detector significance along that best trajectory
    "z_min_det",        # min(z_H, z_L): both detectors must agree
    "zH_max", "zL_max", # best single-detector significance anywhere
    "map_mean", "map_std",
    "z_max_norm",       # (z_max - map_mean) / map_std
    "z_second",         # best peak >= `exclude` bins away in f0 (look-elsewhere context)
    "z_gap",            # z_max - z_second
    "f0_frac", "fdot_frac",
    "det_df0", "det_dfdot",   # H1/L1 best-peak disagreement
    "tmpl_gap",         # z_max - median over templates of per-template max
    "cov_H", "cov_L",   # fraction of time slots with data
]
N_FEATURES = len(FEATURE_NAMES)


def _nanargmax2d(a: np.ndarray):
    if not np.isfinite(a).any():
        return 0.0, 0, 0
    i, j = np.unravel_index(np.nanargmax(a), a.shape)
    return float(a[i, j]), int(i), int(j)


class PhysicsFeatureExtractor:
    def __init__(self, n_fdot: int = 61, fdot_max: float = FDOT_MAX_BINS, n_templates: int = 1,
                 time_chunk: int = PHYS_TIME_CHUNK, exclude: int = 8):
        self.fdots = make_fdot_grid(fdot_max, n_fdot)
        self.fdot_max = fdot_max
        self.n_templates = n_templates
        self.time_chunk = time_chunk
        self.exclude = exclude

    def extract(self, sums: np.ndarray, counts: np.ndarray, f_center_hz: float) -> np.ndarray:
        """sums (2, 360, T'), counts (2, T') from `pool_time` on normalised power."""
        offsets = template_bank(self.n_templates, f_center_hz, self.time_chunk)
        r = search(sums, counts, self.fdots, offsets)

        zmax, i, j = _nanargmax2d(r.z_map)
        zh_best, zl_best = r.zH_map[i, j], r.zL_map[i, j]
        zh_best = 0.0 if not np.isfinite(zh_best) else float(zh_best)
        zl_best = 0.0 if not np.isfinite(zl_best) else float(zl_best)

        finite = r.z_map[np.isfinite(r.z_map)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std()) if finite.size else 1.0

        with np.errstate(all="ignore"):
            row_max = np.nanmax(np.where(np.isfinite(r.z_map), r.z_map, -np.inf), axis=0)   # (F,)
        lo, hi = max(0, j - self.exclude), min(N_FREQ, j + self.exclude + 1)
        rest = np.concatenate([row_max[:lo], row_max[hi:]])
        rest = rest[np.isfinite(rest)]
        z_second = float(rest.max()) if rest.size else 0.0

        zh_max, ih, jh = _nanargmax2d(r.zH_map)
        zl_max, il, jl = _nanargmax2d(r.zL_map)

        tm = r.template_max[np.isfinite(r.template_max)]
        tmpl_gap = float(zmax - np.median(tm)) if tm.size > 1 else 0.0

        n_slots = counts.shape[1] * self.time_chunk
        feats = np.array([
            zmax, zh_best, zl_best, min(zh_best, zl_best), zh_max, zl_max,
            mean, std, (zmax - mean) / (std + 1e-6),
            z_second, zmax - z_second,
            j / N_FREQ, self.fdots[i] / self.fdot_max,
            abs(jh - jl) / N_FREQ, abs(self.fdots[ih] - self.fdots[il]) / (2 * self.fdot_max),
            tmpl_gap,
            counts[0].sum() / n_slots, counts[1].sum() / n_slots,
        ], dtype=np.float32)
        return feats
