import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CFPreprocess
from slop.data import SlopDataset
from slop.detector import SNAPSHOT
from slop.eval import Evaluator
from torch.utils.data import DataLoader

ds = SlopDataset("../data/bench", transform=CFPreprocess())
loader = DataLoader(ds, batch_size=16, num_workers=8)
labels = np.array([s.label for s in ds.samples])
print(f"{len(ds)} bench images")

paths = (
    [Path(p) for p in sys.argv[1:]]
    if len(sys.argv) > 1
    else [SNAPSHOT / "onnx" / n for n in ["model.onnx", "model_int8.onnx", "model_q4.onnx"]]
)
for path in paths:
    sess = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs, t_total, n = [], 0.0, 0
    for x, _ in loader:
        x = x.numpy()
        t0 = time.perf_counter()
        out = sess.run(None, {"pixel_values": x})[0].squeeze(-1)
        t_total += time.perf_counter() - t0
        n += len(x)
        probs.append(1 / (1 + np.exp(-out)))
    probs = np.concatenate(probs)
    res = Evaluator(threshold=0.65).score(probs, labels)
    size = path.stat().st_size / 1e6
    print(
        f"{path.name:18s} {size:6.1f}MB  bacc@0.65={res.balanced_acc:.4f} "
        f"(TPR {res.tpr:.3f}/TNR {res.tnr:.3f})  AUC={res.auc:.4f} "
        f"best@{res.best_threshold:.2f}={res.best_balanced_acc:.4f}  {1000 * t_total / n:.0f}ms/img"
    )
