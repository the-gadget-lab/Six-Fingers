import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CFPreprocess
from slop.data import SlopDataset, WebRealistic
from slop.eval import Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../model/dist/model.onnx")
    ap.add_argument("--root", default="../data/cf_eval")
    ap.add_argument("--degrade", action="store_true", default=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ds = SlopDataset(args.root, transform=CFPreprocess(),
                     degrade=WebRealistic(seed=0) if args.degrade else None)
    print(f"{len(ds)} images from {args.root}, model={args.model}")
    sess = rt.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    probs, labels = [], []
    for x, y in tqdm(DataLoader(ds, batch_size=args.batch, num_workers=10)):
        out = sess.run(None, {inp: x.numpy()})[0].squeeze(-1)
        probs.append(1 / (1 + np.exp(-out)))
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    res = Evaluator(threshold=0.65).score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
