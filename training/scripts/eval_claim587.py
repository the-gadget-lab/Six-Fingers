"""Replicates the RajeshRk18 ai-image-detector extension pipeline (claim 587).

Two-model fp32 ensemble, exactly as src/lib/constants.js + pipeline.js:
  1. CommunityForensics ViT-S: shortest-edge-440 geometry, center + 4 corner
     384px crops (the TTA condition `max_dim*scale > 416` is always true since
     min_dim*scale == 440), sigmoid logit, crops averaged in logit space.
  2. Organika/sdxl-detector Swin: squash whole image to 224x224, imagenet norm,
     z = logit[artificial] - logit[human].
  Fusion: sigmoid(3.299226 + 0.61576*z1 + 0.216296*z2 + logit(0.65)), AI >= 0.65.

Skipped as provably no-op on this benchmark: metadata branch (0 AI-marker hits
in metadata regions of all 6,523 files; WebRealistic re-encode strips metadata
anyway) and alpha-cutout floor (RGB inputs, and it only lifts scores <0.35 to
0.35 -- cannot flip a 0.65 decision). fp32 pytorch weights stand in for the
fp32 onnx exports (verified identical to 1e-6 on Organika onnx).
Approximation: canvas drawImage high-quality smoothing -> PIL bicubic.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CLIP_MEAN, CLIP_STD
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator
from torch.utils.data import DataLoader

SDXL_SNAPSHOT = next(
    Path(__file__).parents[2].joinpath(
        "data/hf/hub/models--Organika--sdxl-detector/snapshots").glob("*"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FUSION_BIAS, FUSION_W = 3.299226, (0.61576, 0.216296)
THRESHOLD = 0.65


def normalize(img, mean, std):
    t = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1)) / 255
    return (t - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


class Claim587Transform:
    """Returns (cf_crops [5,3,384,384], sdxl [3,224,224]) per image."""

    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        # CF: crop boxes in original coords, each resized straight to 384
        # (mirrors canvas drawImage(sx, sy, srcCrop, srcCrop -> 384x384))
        src = 384 * min(w, h) / 440
        boxes = [((w - src) / 2, (h - src) / 2), (0, 0), (w - src, 0),
                 (0, h - src), (w - src, h - src)]
        cf = torch.stack([
            normalize(img.resize((384, 384), Image.BICUBIC,
                                 box=(sx, sy, sx + src, sy + src)),
                      CLIP_MEAN, CLIP_STD)
            for sx, sy in boxes])
        sdxl = normalize(img.resize((224, 224), Image.BICUBIC),
                         IMAGENET_MEAN, IMAGENET_STD)
        return cf, sdxl


def main():
    ds = SlopDataset("../data/cf_eval", transform=Claim587Transform(),
                     degrade=WebRealistic(seed=0))
    print(f"{len(ds)} images, CF ViT-S fp32 (5-crop TTA) + Organika Swin fp32, "
          f"logistic fusion @ {THRESHOLD}")

    cf = HFDetector().cuda().eval()
    from transformers import AutoModelForImageClassification
    sdxl = AutoModelForImageClassification.from_pretrained(SDXL_SNAPSHOT).cuda().eval()

    loader = DataLoader(ds, batch_size=10, num_workers=8, pin_memory=True)
    probs, labels = [], []
    logit_thr = math.log(THRESHOLD / (1 - THRESHOLD))
    with torch.no_grad():
        from tqdm import tqdm
        for (xc, xs), y in tqdm(loader):
            b, n, c, hh, ww = xc.shape
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z1 = cf(xc.view(b * n, c, hh, ww).cuda(non_blocking=True))
                z1 = z1.float().view(b, n).mean(dim=1)
                lo = sdxl(pixel_values=xs.cuda(non_blocking=True)).logits.float()
            z2 = lo[:, 0] - lo[:, 1]  # artificial minus human
            z = FUSION_BIAS + FUSION_W[0] * z1 + FUSION_W[1] * z2
            probs.append(torch.sigmoid(z + logit_thr).cpu().numpy())
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    res = Evaluator(threshold=THRESHOLD).score(probs, labels,
                                               [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
