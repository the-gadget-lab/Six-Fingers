"""Benchmark of the Proofmark extension (proofmark-webwild-v3, Q8 ONNX).

Replicates their offscreen.js pipeline: ViTImageProcessor preprocessing
(shortest edge 440 bicubic, center crop 384, /255, ImageNet norm),
sigmoid over the single logit, pixel-statistics fusion in logit space,
then their monotonic display calibration mapping raw threshold
0.020887 -> 0.65. Metadata/watermark fusion is skipped: benchmark images
are re-encoded so no generator metadata survives (floor-only rule anyway).
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.data import SlopDataset, WebRealistic
from slop.eval import Evaluator
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

MODEL = Path(
    "/tmp/claude-1000/-home-lokman-projects-ai-slop-detect/a23a9609-2bd2-45e5-bd0e-aacdda9f49d7"
    "/scratchpad/Dino-ImageGen-Ext/public/models/Proofmark/proofmark-webwild-v3/onnx/model_quantized.onnx")
RAW_THRESHOLD = 0.020887045509173325
DISPLAY_THRESHOLD = 0.65
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pixel_adjustment(img: Image.Image) -> float:
    """Port of computePixelSignals: 256px thumbnail, every-other-pixel residual stats."""
    scale = min(1.0, 256 / max(img.size))
    w = max(8, round(img.width * scale))
    h = max(8, round(img.height * scale))
    # ponytail: PIL bilinear stands in for canvas drawImage resampling
    a = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float64)
    lum = a @ np.array([0.2126, 0.7152, 0.0722])

    ys, xs = np.arange(1, h - 1, 2), np.arange(1, w - 1, 2)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    c = lum[Y, X]
    left, right = lum[Y, X - 1], lum[Y, X + 1]
    up, down = lum[Y - 1, X], lum[Y + 1, X]
    residual = np.abs(c - (left + right + up + down) / 4)
    n = c.size

    residual_mean = residual.mean()
    residual_var = (residual**2).mean() - residual_mean**2
    edge_density = (np.abs(right - left) + np.abs(down - up)).mean() / 510
    rgb = a[Y, X]
    saturation = ((rgb.max(-1) - rgb.min(-1) > 150) & (rgb.max(-1) > 210)).mean()
    hist = np.bincount(np.minimum(31, (c / 8).astype(int)).ravel(), minlength=32)
    p = hist[hist > 0] / n
    entropy = -(p * np.log2(p)).sum()

    adj = 0.0
    if residual_mean < 4.2 and edge_density > 0.055:
        adj += 0.025
    if residual_var > 230 and edge_density < 0.045:
        adj += 0.02
    if saturation > 0.13 and entropy > 4.4:
        adj += 0.015
    if residual_mean > 13 and edge_density > 0.12:
        adj -= 0.02
    return float(np.clip(adj, -0.04, 0.06))


class ProofmarkPreprocess:
    def __call__(self, img: Image.Image):
        img = img.convert("RGB")
        adj = pixel_adjustment(img)
        r = F.resize(img, 440, interpolation=F.InterpolationMode.BICUBIC)
        r = F.center_crop(r, 384)
        t = F.pil_to_tensor(r).float() / 255
        return (t - IMAGENET_MEAN) / IMAGENET_STD, torch.tensor(adj)


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    ds = SlopDataset("../data/cf_eval", transform=ProofmarkPreprocess(),
                     degrade=WebRealistic(seed=0))
    print(f"{len(ds)} images, Proofmark webwild-v3 Q8 ONNX (CPU)")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    loader = DataLoader(ds, batch_size=32, num_workers=8)

    probs, labels = [], []
    from tqdm import tqdm
    for (x, adj), y in tqdm(loader):
        logits = sess.run(["logits"], {"pixel_values": x.numpy()})[0][:, 0]
        m = np.clip(sigmoid(logits), 0.001, 0.999)
        fused = np.clip(sigmoid(logit(m) + adj.numpy() * 4), 0.001, 0.999)
        score = np.clip(sigmoid(logit(fused) - logit(RAW_THRESHOLD) + logit(DISPLAY_THRESHOLD)),
                        0.001, 0.999)
        probs.append(score)
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    ev = Evaluator(threshold=0.65)
    res = ev.score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
