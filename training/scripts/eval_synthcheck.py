"""Replicates SynthCheck's inference pipeline (Community Forensics ViT-S/16 INT8 ONNX).

Pipeline per src/inference/detector.ts + preprocess.ts + calibration.ts:
  1. Center-crop (224/256)*short_edge source pixels, resize directly to 224x224
     (single resample, canvas drawImage with high-quality smoothing).
  2. ImageNet mean/std normalize, CHW float32.
  3. ONNX 'pixel_values' -> 'logits' (single logit) -> sigmoid.
  4. Platt calibration: sigmoid(clamp(logit(p) + 3.563478187572664, -40, 40)).
  5. Classify likely-ai at calibrated prob >= 0.65 (raw prob ~0.05).

Approximations vs the extension: PIL bicubic instead of canvas resampling,
crop box rounded to integer pixels, ORT CPU EP instead of wasm.
"""

import math
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

MODEL = Path("/tmp/claude-1000/-home-lokman-projects-ai-slop-detect/a23a9609-2bd2-45e5-bd0e-aacdda9f49d7/scratchpad/synthcheck/weights/community-forensics-int8.onnx")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INTERCEPT = 3.563478187572664


class SynthCheckPreprocess:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        crop = (224 / 256) * min(w, h)
        x, y = (w - crop) / 2, (h - crop) / 2
        img = img.resize((224, 224), Image.BICUBIC, box=(x, y, x + crop, y + crop))
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def calibrate(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    z = np.clip(np.log(p / (1 - p)) + INTERCEPT, -40, 40)
    return 1 / (1 + np.exp(-z))


def main():
    ds = SlopDataset("../data/cf_eval", transform=SynthCheckPreprocess(), degrade=WebRealistic(seed=0))
    print(f"{len(ds)} images, SynthCheck pipeline (commfor-224 INT8 ONNX, center crop, "
          f"Platt intercept {INTERCEPT:.4f}, threshold 0.65)")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    loader = DataLoader(ds, batch_size=32, num_workers=8)

    probs, labels = [], []
    from tqdm import tqdm
    for x, y in tqdm(loader):
        logits = sess.run(["logits"], {"pixel_values": x.numpy()})[0][:, 0]
        raw = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        probs.append(calibrate(raw))
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    ev = Evaluator(threshold=0.65)
    res = ev.score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
