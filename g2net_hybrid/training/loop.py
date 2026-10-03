"""Plain PyTorch train / predict / evaluate helpers (no Lightning dependency)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def _loss(out: dict, y: torch.Tensor, aux_weight: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(out["logit"], y)
    if "logit_cnn" in out:                         # hybrid: deep supervision of both branches
        loss = loss + aux_weight * (F.binary_cross_entropy_with_logits(out["logit_cnn"], y)
                                    + F.binary_cross_entropy_with_logits(out["logit_phys"], y))
    return loss


@torch.no_grad()
def predict(model, dataset, device, batch_size=64, num_workers=2) -> np.ndarray:
    model.eval()
    probs = []
    for b in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        out = model(b["image"].to(device), b["phys"].to(device))
        probs.append(out["logit"].sigmoid().cpu().numpy())
    return np.concatenate(probs)


def auc_report(y, p, strength, bins=(0.0, 0.3, 0.45, 0.6, 10.0)) -> dict:
    """Overall AUC plus AUC of each injection-strength bin vs all negatives (weak-signal analysis)."""
    rep = {"auc": float(roc_auc_score(y, p))}
    neg = y == 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (y == 1) & (strength > lo) & (strength <= hi)
        if sel.sum() >= 5:
            m = neg | sel
            rep[f"auc[{lo:.2f},{hi:.2f}]"] = float(roc_auc_score(y[m], p[m]))
    return rep


def fit(model, train_ds, val_ds, val_strengths, epochs=10, lr=3e-4, batch_size=32, aux_weight=0.3,
        num_workers=2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.3)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    y_val = np.array([int(val_ds[i]["label"]) for i in range(len(val_ds))])
    best = (-1.0, None, None)
    for ep in range(epochs):
        model.train()
        run = 0.0
        for b in loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(b["image"].to(device), b["phys"].to(device))
                loss = _loss(out, b["label"].to(device), aux_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        p = predict(model, val_ds, device, num_workers=num_workers)
        rep = auc_report(y_val, p, val_strengths)
        print(f"epoch {ep + 1}/{epochs} loss {run / len(loader):.4f} | " + " ".join(f"{k}={v:.3f}" for k, v in rep.items()))
        if rep["auc"] > best[0]:
            best = (rep["auc"], rep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[1]
