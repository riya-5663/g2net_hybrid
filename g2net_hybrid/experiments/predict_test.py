"""Stream the competition test set through the trained model -> submission CSV.

This is the only stage that reads the full ~200 GB. It is resumable and can be split across
Kaggle sessions with --start/--end (then concatenate the partial CSVs).

  python -m experiments.predict_test --ckpt checkpoints/hybrid.pt \
      --input-dir /kaggle/input/<comp> --out /kaggle/working/sub_part0.csv --start 0 --end 1500 --n-jobs 4
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from constants import N_FREQ  # noqa: F401  (kept for shape sanity in debugging)
from data.dataset import sample_to_arrays
from models.fusion import HybridModel
from physics.features import N_FEATURES, PhysicsFeatureExtractor


def _worker(args):
    path, ext = args
    try:
        from data.sft import load_hdf5_sample
        d = load_hdf5_sample(path)
        return sample_to_arrays(d["x"], d["mask"], float(d["freq"].mean()), PhysicsFeatureExtractor(**ext))
    except Exception as e:                      # odd/short files: report, score neutrally
        print(f"[warn] {os.path.basename(path)}: {e!r}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--fallback", type=float, default=0.0, help="score for unreadable files")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = HybridModel(N_FEATURES, ck["mode"], ck["model"], pretrained=False)
    model.load_state_dict(ck["state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    ext = ck["ext"]                              # same physics settings as in training

    ids = pd.read_csv(f"{a.input_dir}/sample_submission.csv").id.tolist()[a.start:a.end]
    done = set(pd.read_csv(a.out).id) if os.path.exists(a.out) else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} ids in range, {len(done)} already done, {len(todo)} to go")
    if not todo:
        return
    if not os.path.exists(a.out):
        pd.DataFrame({"id": [], "target": []}).to_csv(a.out, index=False)

    jobs = [(f"{a.input_dir}/test/{i}.hdf5", ext) for i in todo]
    ex = ProcessPoolExecutor(a.n_jobs) if a.n_jobs > 1 else None
    results = ex.map(_worker, jobs, chunksize=2) if ex else map(_worker, jobs)

    buf = []

    def flush():
        if not buf:
            return
        ok = [b for b in buf if b[1] is not None]
        probs = {}
        if ok:
            img = torch.from_numpy(np.stack([b[1][0] for b in ok]).astype(np.float32)).to(device)
            phys = torch.from_numpy(np.stack([b[1][1] for b in ok])).to(device)
            with torch.no_grad():
                p = model(img, phys)["logit"].sigmoid().cpu().numpy()
            probs = {b[0]: float(v) for b, v in zip(ok, p)}
        pd.DataFrame({"id": [b[0] for b in buf], "target": [probs.get(b[0], a.fallback) for b in buf]}) \
            .to_csv(a.out, mode="a", header=False, index=False)
        buf.clear()

    try:
        from tqdm import tqdm
        results = tqdm(results, total=len(todo))
    except ImportError:
        pass
    for i, r in zip(todo, results):
        buf.append((i, r))
        if len(buf) >= a.batch_size:
            flush()
    flush()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
