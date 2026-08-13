import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CFPreprocess
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/cf_eval")
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--max-per-source", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    ds = SlopDataset(
        args.root, transform=CFPreprocess(),
        degrade=WebRealistic(seed=0) if args.degrade else None,
        max_per_source=args.max_per_source or None,
    )
    print(f"{len(ds)} images, degrade={args.degrade}, ckpt={args.ckpt or 'base'}, bias={args.bias}")

    model = (
        HFDetector.from_checkpoint(args.ckpt, logit_bias=args.bias)
        if args.ckpt
        else HFDetector(logit_bias=args.bias)
    ).cuda().eval()
    ev = Evaluator(threshold=0.65)
    probs, labels = ev.collect(model, ds, batch_size=args.batch, workers=8)
    res = ev.score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
