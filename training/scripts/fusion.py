import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator
from slop.tta import FiveCrop384, crop_logits

FIT_SEED = 7
EVAL_SEED = 0


def features(root, degrade_seed, max_per_source, models, batch=8):
    ds = SlopDataset(root, transform=FiveCrop384(),
                     degrade=WebRealistic(seed=degrade_seed), max_per_source=max_per_source)
    loader = DataLoader(ds, batch_size=batch, num_workers=10, pin_memory=True)
    zs = []
    labels = None
    for m in models:
        z, labels = crop_logits(m, loader)
        zs.append(z)
    sources = [s.source for s in ds.samples]
    return np.stack(zs, axis=1), labels, sources


def fit_lr(Z, y, l2=1e-3, iters=50):
    """Class-balanced logistic regression via Newton-IRLS. Returns [bias, w1, w2...]."""
    X = np.concatenate([np.ones((len(Z), 1)), Z], axis=1)
    w = np.zeros(X.shape[1])
    sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (sw * (p - y)) + l2 * np.r_[0, w[1:]]
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--fit-per-source", type=int, default=1200)
    args = ap.parse_args()

    tuned = HFDetector.from_checkpoint(args.ckpt).cuda()
    stock = HFDetector().cuda()
    models = [tuned, stock]

    print("== fit features (train distribution, degrade seed 7) ==")
    Zf, yf, _ = features("../data/train", FIT_SEED, args.fit_per_source, models)
    w = fit_lr(Zf, yf)
    print("LR weights [bias, z_tuned, z_stock]:", np.round(w, 6).tolist())

    for name, root in (("cf_eval", "../data/cf_eval"), ("extra_eval", "../data/extra_eval")):
        print(f"== {name} ==")
        Z, y, sources = features(root, EVAL_SEED, None, models)
        z = w[0] + Z @ w[1:]
        probs = 1 / (1 + np.exp(-z))
        res = Evaluator(threshold=0.5).score(probs, y, sources)
        print(f"[fused @LR boundary=0.5] {res.summary().splitlines()[0]}")
        for i, mname in enumerate(("tuned", "stock")):
            solo = 1 / (1 + np.exp(-Z[:, i]))
            r = Evaluator(threshold=0.65).score(solo, y)
            print(f"  [{mname} 5crop raw @0.65] bacc={r.balanced_acc:.4f} best@{r.best_threshold:.2f}={r.best_balanced_acc:.4f}")

    spec = {"fusion": {"bias": float(w[0]), "wTuned": float(w[1]), "wStock": float(w[2])}}
    Path("fusion_weights.json").write_text(json.dumps(spec, indent=2))
    print("saved fusion_weights.json")


if __name__ == "__main__":
    main()
