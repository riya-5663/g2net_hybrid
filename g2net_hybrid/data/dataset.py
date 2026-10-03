"""Sample -> (CNN image, physics features), plus on-disk caches and the torch Dataset."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from constants import CNN_TIME_BINS, N_FREQ, PHYS_TIME_BINS
from data.sft import build_cnn_image, normalize_power, pool_time, pooled_mean
from physics.features import N_FEATURES, PhysicsFeatureExtractor


def sample_to_arrays(x, mask, f_center, extractor: PhysicsFeatureExtractor):
    """One shared normalisation feeds both branches."""
    p = normalize_power(x, mask)
    s, c = pool_time(p, mask, CNN_TIME_BINS)
    image = build_cnn_image(pooled_mean(s, c))
    sp, cp = pool_time(p, mask, PHYS_TIME_BINS)
    feats = extractor.extract(sp, cp, f_center)
    return image.astype(np.float16), feats


# ---- workers (top-level so they can be pickled) -----------------------------------------
def _synthetic_worker(args):
    seed, positive, amp_range, ext_kwargs = args
    from data.synthetic import make_sample
    s = make_sample(np.random.default_rng(seed), positive, amp_range)
    img, feats = sample_to_arrays(s["x"], s["mask"], s["f_center"], PhysicsFeatureExtractor(**ext_kwargs))
    return img, feats, s["label"], s["strength"]


def _kaggle_worker(args):
    path, label, ext_kwargs = args
    from data.sft import load_hdf5_sample
    d = load_hdf5_sample(path)
    img, feats = sample_to_arrays(d["x"], d["mask"], float(d["freq"].mean()), PhysicsFeatureExtractor(**ext_kwargs))
    return img, feats, int(label), -1.0       # real data has no injection strength


def _injected_worker(args):
    path, seed, positive, amp_range, ext_kwargs = args
    from data.injection import make_injected_sample
    from data.sft import load_hdf5_sample
    s = make_injected_sample(load_hdf5_sample(path), np.random.default_rng(seed), positive, amp_range)
    img, feats = sample_to_arrays(s["x"], s["mask"], s["f_center"], PhysicsFeatureExtractor(**ext_kwargs))
    return img, feats, s["label"], s["strength"]


def _run(worker, jobs, n_jobs, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    n = len(jobs)
    images = np.lib.format.open_memmap(f"{out_dir}/{name}_images.npy", "w+", np.float16, (n, 4, N_FREQ, CNN_TIME_BINS))
    feats = np.zeros((n, N_FEATURES), np.float32)
    labels = np.zeros(n, np.int64)
    strengths = np.zeros(n, np.float32)

    results = map(worker, jobs) if n_jobs <= 1 else ProcessPoolExecutor(n_jobs).map(worker, jobs, chunksize=4)
    try:
        from tqdm import tqdm
        results = tqdm(results, total=n, desc=name)
    except ImportError:
        pass
    for i, (img, f, y, s) in enumerate(results):
        images[i], feats[i], labels[i], strengths[i] = img, f, y, s
    images.flush()
    np.save(f"{out_dir}/{name}_feats.npy", feats)
    np.save(f"{out_dir}/{name}_labels.npy", labels)
    np.save(f"{out_dir}/{name}_strengths.npy", strengths)


def build_synthetic_cache(n, seed, out_dir, name, ext_kwargs=None, amp_range=(0.15, 0.8), n_jobs=1):
    ext_kwargs = ext_kwargs or {}
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % 2).astype(bool)         # balanced
    seeds = rng.integers(0, 2**31 - 1, size=n)
    jobs = [(int(seeds[i]), bool(labels[i]), amp_range, ext_kwargs) for i in range(n)]
    _run(_synthetic_worker, jobs, n_jobs, out_dir, name)


def build_kaggle_cache(input_dir, split, out_dir, name, ext_kwargs=None, limit=None, n_jobs=1):
    """Uses {input_dir}/train_labels.csv and {input_dir}/{split}/{id}.hdf5 (labels == -1 are dropped)."""
    import pandas as pd
    ext_kwargs = ext_kwargs or {}
    df = pd.read_csv(f"{input_dir}/train_labels.csv")
    df = df[df.target >= 0]
    if limit:
        df = df.iloc[:limit]
    jobs = [(f"{input_dir}/{split}/{r.id}.hdf5", r.target, ext_kwargs) for r in df.itertuples()]
    _run(_kaggle_worker, jobs, n_jobs, out_dir, name)


def build_injected_cache(noise_paths, n, seed, out_dir, name, ext_kwargs=None, amp_range=(0.15, 0.8), n_jobs=1):
    """n samples: random noise file each, 50% get an injected signal. Strength is stored for bin analysis."""
    ext_kwargs = ext_kwargs or {}
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % 2).astype(bool)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    picks = rng.integers(0, len(noise_paths), size=n)
    jobs = [(noise_paths[picks[i]], int(seeds[i]), bool(labels[i]), amp_range, ext_kwargs) for i in range(n)]
    _run(_injected_worker, jobs, n_jobs, out_dir, name)


# ---- torch dataset --------------------------------------------------------------------
def load_cache(out_dir, name):
    return dict(
        images=np.load(f"{out_dir}/{name}_images.npy", mmap_mode="r"),
        feats=np.load(f"{out_dir}/{name}_feats.npy"),
        labels=np.load(f"{out_dir}/{name}_labels.npy"),
        strengths=np.load(f"{out_dir}/{name}_strengths.npy"),
    )


def make_torch_dataset(cache: dict, augment: bool = False):
    import torch
    from torch.utils.data import Dataset

    class CachedDataset(Dataset):
        def __len__(self):
            return len(cache["labels"])

        def __getitem__(self, i):
            img = torch.from_numpy(np.asarray(cache["images"][i], dtype=np.float32))
            if augment:                                    # flips are label-preserving for this task
                if np.random.rand() < 0.5:
                    img = img.flip(2)
                if np.random.rand() < 0.5:
                    img = img.flip(1)
            return {
                "image": img,
                "phys": torch.from_numpy(cache["feats"][i]),
                "label": torch.tensor(float(cache["labels"][i])),
            }

    return CachedDataset()
