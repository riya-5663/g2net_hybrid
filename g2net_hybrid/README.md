# g2net_hybrid

Late fusion of a CNN branch (4-channel H1/L1 image, after the 12th-place solution) and a physics
branch (trajectory power-sum search, after Koda's 1st-place idea) -> MLP -> P(CW).

Run everything from this folder (`g2net_hybrid/`) with `python -m ...`.

```
constants.py                  grid sizes (360 x 5760), chunking, fdot range
data/sft.py                   HDF5 loader, grid placement, power normalisation, pooling, 4-channel image
data/synthetic.py             synthetic H1/L1 generator (noise + drifting, Doppler-modulated line)
data/dataset.py               sample -> (image, physics features); caches; torch Dataset
physics/power_score.py        your original trajectory_power_score (unchanged logic)
physics/doppler.py            orbital-Doppler template bank (circular-orbit approximation)
physics/trajectory_search.py  sheared power-sum search over (f0, fdot, sky template)
physics/features.py           18 physics features (z_max, per-detector z, H1/L1 agreement, ...)
models/cnn_branch.py          timm ConvNeXt -> embedding
models/fusion.py              HybridModel, mode = cnn | physics | hybrid
training/loop.py              train / predict / AUC (overall + per injection-strength bin)
experiments/build_cache.py    build caches (synthetic or kaggle)
experiments/run_ablation.py   experiments A (CNN), B (physics), C (hybrid) + delta AUC
tests/test_physics.py         physics-branch sanity tests
```

## Is the code generating data?
Three sources, chosen with `build_cache --source`:
* `synthetic` - fully fake (white Gaussian noise + a simple injected line). Pipeline check only.
* `injected` - **recommended for training**: simulated lines added to REAL noise files
  (`target == 0` in `train_labels.csv`), split by noise file between train and val.
  The injected line is still the simplified model in `data/synthetic.py`, not a PyFstat/LAL waveform.
* `real` - the genuine labelled competition files, no simulation. Use this as the honest check
  (`run_ablation --real-name real`).

## Run order
```bash
pip install -r requirements.txt
python -m tests.test_physics
python -m experiments.build_cache --source real --input-dir $COMP --limit 20      # smoke test on real files
python -m experiments.build_cache --source injected --input-dir $COMP --n-train 8000 --n-val 2000 --n-jobs 4
python -m experiments.build_cache --source real --input-dir $COMP --n-jobs 4
python -m experiments.run_ablation --cache cache --real-name real --epochs 10   # A, B, C + checkpoints/
python -m experiments.predict_test --ckpt checkpoints/hybrid.pt --input-dir $COMP --out sub.csv --n-jobs 4
```
`predict_test` is resumable and takes `--start/--end` so the test set can be split across Kaggle sessions.

## Caveats
* Physics features are computed once and cached (about 0.1 s/sample with 1 template, ~2 s with 33).
  The same `--n-templates/--n-fdot` are stored in `cache/meta.json` and reused at test time.
* `doppler.py` is orbit-only and circular: validate it against PyFstat/LALSuite before relying on it.
* The real labelled set is small; injected-noise validation can be optimistic if few noise files exist.
* The torch parts (CNN, fusion, training, predict_test) were not executed in the authoring sandbox.
