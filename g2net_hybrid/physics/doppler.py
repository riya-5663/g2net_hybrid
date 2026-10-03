"""Orbital Doppler templates (sky-position dependent frequency drift, in SFT bins).

Dominant effect: Earth's orbital motion, v/c ~ 9.9e-5, i.e. up to ~ +-18 bins at 100 Hz.
Earth's rotation (~1.5e-6 v/c, < 0.3 bin at 100 Hz) is ignored, so this is a *circular-orbit,
orbit-only approximation*. It is the same for H1 and L1, so one offset curve serves both.

This is intentionally simple. Validate it against LALSuite/PyFstat (e.g. compare with
`pyfstat` generated trajectories) before trusting it for physics claims.
"""
from __future__ import annotations

import numpy as np

from constants import GPS_T0, N_TIME, T_SFT

GPS_AT_J2000 = 630763213.0          # GPS seconds at 2000-01-01 12:00 UTC
V_ORB_OVER_C = 29.78e3 / 299_792_458.0
OBLIQUITY = np.deg2rad(23.4393)


def earth_velocity_ecliptic(t_gps: np.ndarray) -> np.ndarray:
    """Earth's heliocentric velocity / c in ecliptic coordinates, shape (T, 3)."""
    d = (np.asarray(t_gps, dtype=np.float64) - GPS_AT_J2000) / 86400.0
    sun_lon = np.deg2rad(280.460 + 0.9856474 * d)          # mean longitude of the Sun
    theta = sun_lon + np.pi                                  # Earth as seen from the Sun
    v = np.stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=-1)
    return V_ORB_OVER_C * v


def sky_unit_vector_ecliptic(alpha: float, delta: float) -> np.ndarray:
    """Unit vector towards (alpha=RA, delta=Dec) [rad], in ecliptic coordinates."""
    x = np.cos(delta) * np.cos(alpha)
    y = np.cos(delta) * np.sin(alpha)
    z = np.sin(delta)
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.array([x, y * ce + z * se, -y * se + z * ce])


def orbital_offsets_bins(alpha, delta, f_center_hz, t_gps) -> np.ndarray:
    """Doppler drift in SFT bins relative to the first time in t_gps."""
    v = earth_velocity_ecliptic(t_gps)
    n = sky_unit_vector_ecliptic(alpha, delta)
    shift_hz = f_center_hz * (v @ n)
    return (shift_hz - shift_hz[0]) * T_SFT


def fibonacci_sky(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n roughly uniform points on the sphere -> (alpha, delta)."""
    i = np.arange(n) + 0.5
    delta = np.arcsin(1.0 - 2.0 * i / n)
    alpha = np.mod(np.pi * (1.0 + 5.0 ** 0.5) * i, 2 * np.pi)
    return alpha, delta


def template_bank(n_templates: int, f_center_hz: float, time_chunk: int) -> np.ndarray:
    """Offsets in bins at chunk centres, shape (n_templates, N_TIME // time_chunk).

    Template 0 is always the zero-Doppler template (baseline). Remaining templates are
    spread over the sky. ~4000 templates are needed for <1-bin accuracy at ~100 Hz.
    """
    n_chunks = N_TIME // time_chunk
    centres = (np.arange(n_chunks) + 0.5) * time_chunk - 0.5          # slot index of chunk centre
    t_gps = GPS_T0 + T_SFT * np.concatenate([[0.0], centres])         # prepend t0 for the reference
    bank = np.zeros((n_templates, n_chunks))
    if n_templates > 1:
        alpha, delta = fibonacci_sky(n_templates - 1)
        for k in range(n_templates - 1):
            off = orbital_offsets_bins(alpha[k], delta[k], f_center_hz, t_gps)
            bank[k + 1] = off[1:]
    return bank


def offsets_for_sky(alpha: float, delta: float, f_center_hz: float, time_chunk: int) -> np.ndarray:
    """Single-template offsets at chunk centres (used for tests / oracle search)."""
    n_chunks = N_TIME // time_chunk
    centres = (np.arange(n_chunks) + 0.5) * time_chunk - 0.5
    t_gps = GPS_T0 + T_SFT * np.concatenate([[0.0], centres])
    return orbital_offsets_bins(alpha, delta, f_center_hz, t_gps)[1:]
