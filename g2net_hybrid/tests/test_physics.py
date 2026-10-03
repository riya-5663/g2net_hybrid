"""Sanity tests for the physics branch (numpy only). Run:  python -m tests.test_physics"""
import numpy as np

from constants import PHYS_TIME_BINS, PHYS_TIME_CHUNK
from data.sft import normalize_power, pool_time
from data.synthetic import make_sample
from physics.doppler import offsets_for_sky
from physics.trajectory_search import make_fdot_grid, search


def _prep(s):
    p = normalize_power(s["x"], s["mask"])
    return pool_time(p, s["mask"], PHYS_TIME_BINS)


def test_noise_is_unit_variance():
    s = make_sample(np.random.default_rng(0), False)
    p = normalize_power(s["x"], s["mask"])
    valid = s["mask"][:, None, :].astype(bool).repeat(p.shape[1], 1)
    assert abs(p[valid].mean() - 1.0) < 0.05


def test_oracle_search_recovers_injection():
    rng = np.random.default_rng(3)
    fd = make_fdot_grid(120, 61)
    hits = 0
    for _ in range(4):
        s = make_sample(rng, True, (0.6, 0.8))
        m = s["meta"]
        sums, counts = _prep(s)
        off = offsets_for_sky(m["alpha"], m["delta"], s["f_center"], PHYS_TIME_CHUNK)
        r = search(sums, counts, fd, off[None])
        i, j = np.unravel_index(np.nanargmax(r.z_map), r.z_map.shape)
        hits += abs(j - m["f0_bin"]) <= 3 and abs(fd[i] - m["fdot_bins"]) <= 4
    assert hits >= 3, hits


def test_true_sky_beats_wrong_sky():
    rng = np.random.default_rng(5)
    s = make_sample(rng, True, (0.6, 0.8))
    m = s["meta"]
    sums, counts = _prep(s)
    fd = make_fdot_grid(120, 61)
    good = offsets_for_sky(m["alpha"], m["delta"], s["f_center"], PHYS_TIME_CHUNK)
    bad = offsets_for_sky((m["alpha"] + 2.0) % (2 * np.pi), -m["delta"], s["f_center"], PHYS_TIME_CHUNK)
    r = search(sums, counts, fd, np.stack([good, bad]))
    assert r.best_template == 0 and r.template_max[0] > r.template_max[1]


def test_signal_beats_noise():
    rng = np.random.default_rng(7)
    fd = make_fdot_grid(120, 61)
    noise_z = [np.nanmax(search(*_prep(make_sample(rng, False)), fd).z_map) for _ in range(5)]
    s = make_sample(rng, True, (0.7, 0.8))
    m = s["meta"]
    off = offsets_for_sky(m["alpha"], m["delta"], s["f_center"], PHYS_TIME_CHUNK)
    sig_z = np.nanmax(search(*_prep(s), fd, off[None]).z_map)
    assert sig_z > max(noise_z) + 2, (sig_z, noise_z)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
