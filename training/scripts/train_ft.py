import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CFPreprocess
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.train import TrainConfig, Trainer


class TrainAugment:
    """CF-style train pipeline: resize shortest 440, random crop 384, hflip, CLIP norm."""

    def __init__(self, resize=440, crop=384):
        self.prep = CFPreprocess(resize, crop)
        self.crop = crop

    def __call__(self, img):
        from torchvision.transforms import functional as F

        img = img.convert("RGB")
        img = F.resize(img, self.prep.resize, interpolation=F.InterpolationMode.BICUBIC)
        top = random.randint(0, max(0, img.height - self.crop))
        left = random.randint(0, max(0, img.width - self.crop))
        img = F.crop(img, top, left, self.crop, self.crop)
        if random.random() < 0.5:
            img = F.hflip(img)
        t = F.pil_to_tensor(img).float() / 255
        return (t - self.prep.mean) / self.prep.std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-per-source", type=int, default=80000)
    args = ap.parse_args()

    train_ds = SlopDataset(
        "../data/train", transform=TrainAugment(),
        degrade=WebRealistic(jpeg_range=(40, 98), scale_range=(0.25, 1.0)),
        max_per_source=args.max_per_source,
    )
    val_ds = SlopDataset("../data/cf_eval", transform=CFPreprocess(), degrade=WebRealistic(seed=0))
    n_fake = sum(s.label for s in train_ds.samples)
    print(f"train: {len(train_ds)} ({n_fake} fake / {len(train_ds) - n_fake} real), val: {len(val_ds)}")

    model = HFDetector()
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch)
    best = Trainer(model, model.head, train_ds, val_ds, cfg).run()
    print(f"best degraded balanced_acc@0.65: {best:.4f}")


if __name__ == "__main__":
    main()
