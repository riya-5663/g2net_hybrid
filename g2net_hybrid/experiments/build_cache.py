"""Build image + physics-feature caches.

  # pipeline check, no competition data needed
  python -m experiments.build_cache --source synthetic --n-train 2000 --n-val 500 --n-jobs 8

  # RECOMMENDED for real runs: simulated signals injected into REAL noise files (target == 0),
  # split by noise FILE so train/val never share a noise realisation  -> cache/train, cache/val
  python -m experiments.build_cache --source injected --input-dir /kaggle/input/<comp> \
         --n-train 8000 --n-val 2000 --n-jobs 4

  # honest real-data check: every labelled competition file (target 0/1)  -> cache/real
  python -m experiments.build_cache --source real --input-dir /kaggle/input/<comp> --n-jobs 4
"""
import argparse
import json
import os

import numpy as np

from data.dataset import build_injected_cache, build_kaggle_cache, build_synthetic_cache, build_real_holdout_cache

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "injected", "real"], default="synthetic")
ap.add_argument("--out", default="cache")
ap.add_argument("--n-train", type=int, default=2000)
ap.add_argument("--n-val", type=int, default=500)
ap.add_argument("--input-dir", help="competition root: train_labels.csv, train/*.hdf5, test/*.hdf5")
ap.add_argument("--val-noise-frac", type=float, default=0.25, help="injected: fraction of noise files reserved for val")
ap.add_argument("--limit", type=int, default=None, help="real: only first N labelled files (use ~20 for a smoke test)")
ap.add_argument("--n-templates", type=int, default=1, help="sky templates (1 = zero Doppler only)")
ap.add_argument("--n-fdot", type=int, default=61)
ap.add_argument("--n-jobs", type=int, default=1)
a = ap.parse_args()

ext = dict(n_templates=a.n_templates, n_fdot=a.n_fdot)
os.makedirs(a.out, exist_ok=True)
meta_path = f"{a.out}/meta.json"                      # test-time features MUST use the same settings
if os.path.exists(meta_path):
    old = json.load(open(meta_path))["ext"]
    assert old == ext, f"cache dir already built with {old}, got {ext}; use a different --out"
json.dump({"ext": ext}, open(meta_path, "w"))

if a.source == "synthetic":
    build_synthetic_cache(a.n_train, 0, a.out, "train", ext, n_jobs=a.n_jobs)
    build_synthetic_cache(a.n_val, 1, a.out, "val", ext, n_jobs=a.n_jobs)
elif a.source == "injected":
    import pandas as pd
    df = pd.read_csv(f"{a.input_dir}/train_labels.csv")
    ids = df[df.target == 0].id.tolist()
    assert len(ids) >= 4, "need labelled-noise files (target == 0) to inject into"
    rng = np.random.default_rng(123)
    rng.shuffle(ids)

# --------------------------------------------------
# Split the REAL NOISE files into:
#   70% train
#   15% validation
#   15% held-out real test
# --------------------------------------------------

    n_test_files = max(1, int(len(ids) * 0.15))
    n_val_files = max(1, int(len(ids) * 0.15))

    test_ids = ids[:n_test_files]

    val_start = n_test_files
    val_end = n_test_files + n_val_files

    val_ids = ids[val_start:val_end]
    train_ids = ids[val_end:]

    paths = lambda L: [
        f"{a.input_dir}/train/{i}.hdf5"
        for i in L
    ]

    print(
        f"{len(train_ids)} train noise files, "
        f"{len(val_ids)} val noise files, "
        f"{len(test_ids)} held-out test noise files"
    )

# --------------------------------------------------
# Injected training set
# --------------------------------------------------

    build_injected_cache(paths(train_ids),a.n_train,0,a.out,"train",ext,n_jobs=a.n_jobs,)

# --------------------------------------------------
# Injected validation set
# --------------------------------------------------

    build_injected_cache(paths(val_ids),a.n_val,1,a.out,"val",ext,n_jobs=a.n_jobs,)

# --------------------------------------------------
# REAL held-out evaluation
# --------------------------------------------------

    build_real_holdout_cache(a.input_dir,paths(test_ids),a.out,"real",ext,n_jobs=a.n_jobs,)
else:
    build_kaggle_cache(a.input_dir, "train", a.out, "real", ext, limit=a.limit, n_jobs=a.n_jobs)
