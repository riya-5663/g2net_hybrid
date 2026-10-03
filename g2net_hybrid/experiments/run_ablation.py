"""Experiments A (CNN only), B (physics only), C (hybrid) on the same cached data.

  python -m experiments.run_ablation --cache cache --epochs 10 --modes cnn physics hybrid

Curriculum / warm start (e.g. start from a bright-signal CNN):
  python -m experiments.run_ablation --cache cache_mid --modes cnn --pretrained \
      --init-ckpt ckpt_bright/cnn.pt --lr 5e-5 --epochs 15 --real-name real --save-dir ckpt_mid
"""
import argparse
import json
import os

import torch

from data.dataset import load_cache, make_torch_dataset
from models.fusion import HybridModel
from physics.features import N_FEATURES
from training.loop import auc_report, fit, predict

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--modes", nargs="+", default=["cnn", "physics", "hybrid"])
ap.add_argument("--model", default="convnext_tiny")
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--num-workers", type=int, default=2)
ap.add_argument("--real-name", default=None, help="optional extra cache (e.g. 'real') to evaluate each best model on")
ap.add_argument("--save-dir", default="checkpoints")
ap.add_argument("--init-ckpt", default=None, help="start from this checkpoint (e.g. a bright-signal CNN)")
a = ap.parse_args()

train, val = load_cache(a.cache, "train"), load_cache(a.cache, "val")
train_ds, val_ds = make_torch_dataset(train, augment=True), make_torch_dataset(val)

meta = json.load(open(f"{a.cache}/meta.json")) if os.path.exists(f"{a.cache}/meta.json") else {"ext": {}}
real = load_cache(a.cache, a.real_name) if a.real_name else None
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.save_dir, exist_ok=True)

init_state = None
if a.init_ckpt:
    init_state = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)["state"]

results = {}
for mode in a.modes:
    print(f"\n=== mode: {mode} ===")
    model = HybridModel(N_FEATURES, mode, a.model, a.pretrained)
    if init_state is not None:
        # strict=False so a CNN-only checkpoint can initialise the CNN part of a hybrid.
        # Must happen BEFORE set_phys_stats: the checkpoint also holds the feature-normalisation
        # buffers, and the stats of the current training set must overwrite them.
        res = model.load_state_dict(init_state, strict=False)
        print(f"init from {a.init_ckpt}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    model.set_phys_stats(train["feats"])
    results[mode] = fit(model, train_ds, val_ds, val["strengths"], a.epochs, a.lr, a.batch_size,
                        num_workers=a.num_workers)
    torch.save({"state": model.state_dict(), "mode": mode, "model": a.model, "ext": meta["ext"]},
               f"{a.save_dir}/{mode}.pt")
    if real is not None:
        p = predict(model, make_torch_dataset(real), device, num_workers=a.num_workers)
        results[mode]["real_auc"] = auc_report(real["labels"], p, real["strengths"])["auc"]

print("\n=== summary ===")
print(json.dumps(results, indent=2))
if {"cnn", "hybrid"} <= results.keys():
    print(f"delta AUC (hybrid - cnn): {results['hybrid']['auc'] - results['cnn']['auc']:+.4f}")
if {"physics", "hybrid"} <= results.keys():
    print(f"delta AUC (hybrid - physics): {results['hybrid']['auc'] - results['physics']['auc']:+.4f}")