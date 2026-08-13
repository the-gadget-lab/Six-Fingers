"""Replicates the RealGuard extension inference pipeline (engine.js analyze()).

Pipeline: decode RGB -> shortest edge 440 Pillow-bilinear (dims clamped >= 384,
Math.round) -> floor center-crop 384 -> /255 -> ImageNet norm -> CF ViT-S fp32
-> logit -> pCal = sigmoid(0.5*logit + 2.9) (Platt calibration from models.json)
-> threshold 0.65.

Frequency branch: computed but fusion is disabled in engine.js (`pFused = pCal`),
so it never affects the score — skipped here.
Metadata branch (fuseSignals): only ever RAISES the score when AI generator
markers exist in the file metadata. Our benchmark images are re-extracted /
re-encoded and carry no such markers, so it is a no-op — skipped here.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

CAL_A, CAL_B = 0.5, 2.9  # models.json "calibration"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RealGuardPreprocess:
    """engine.js preprocess(): their JS bilinear replicates Pillow's exactly,
    so PIL Image.resize(BILINEAR) is the reference implementation."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        w, h = img.size
        scale = 440 / min(w, h)
        # Math.round (half up), clamped to >= inputSize, as in engine.js
        rw = max(384, int(w * scale + 0.5))
        rh = max(384, int(h * scale + 0.5))
        img = img.resize((rw, rh), Image.BILINEAR)
        cx, cy = (rw - 384) // 2, (rh - 384) // 2  # Math.floor center crop
        img = img.crop((cx, cy, cx + 384, cy + 384))
        t = F.pil_to_tensor(img).float() / 255
        return (t - IMAGENET_MEAN) / IMAGENET_STD


def main():
    ds = SlopDataset("../data/cf_eval", transform=RealGuardPreprocess(),
                     degrade=WebRealistic(seed=0))
    print(f"{len(ds)} images, RealGuard pipeline: bilinear 440 -> crop 384 -> "
          f"ImageNet norm, fp32 CF ViT-S, pCal = sigmoid({CAL_A}*logit + {CAL_B})")
    model = HFDetector().cuda().eval()
    loader = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True)

    probs, labels = [], []
    with torch.no_grad():
        from tqdm import tqdm

        for x, y in tqdm(loader):
            logit = model(x.cuda(non_blocking=True))  # fp32, no autocast
            probs.append(torch.sigmoid(CAL_A * logit + CAL_B).float().cpu().numpy())
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    ev = Evaluator(threshold=0.65)
    res = ev.score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
