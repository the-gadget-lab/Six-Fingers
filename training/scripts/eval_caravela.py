import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.cf import CFPreprocess
from slop.data import SlopDataset, WebRealistic
from slop.detector import HFDetector
from slop.eval import Evaluator
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class CaravelaCrops:
    """CaravelaLabs TTA: short-edge 440 -> center+4corner 384 crops, short-edge 512 -> center crop.
    Each crop then passes through CF preprocessing (resize 384->440, center-crop 384)."""

    def __init__(self):
        self.post = CFPreprocess()

    def __call__(self, img):
        img = img.convert("RGB")
        crops = []
        for short_edge, center_only in ((440, False), (512, True)):
            scale = short_edge / min(img.size)
            rw, rh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
            r = img.resize((rw, rh), Image.BILINEAR)
            s = 384
            origins = [((rw - s) // 2, (rh - s) // 2)]
            if not center_only:
                origins += [(0, 0), (max(0, rw - s), 0), (0, max(0, rh - s)),
                            (max(0, rw - s), max(0, rh - s))]
            for x, y in origins:
                crop = r.crop((x, y, min(x + s, rw), min(y + s, rh)))
                if crop.size != (s, s):
                    canvas = Image.new("RGB", (s, s))
                    canvas.paste(crop, (0, 0))
                    crop = canvas
                crops.append(self.post(crop))
        return torch.stack(crops)


from PIL import Image  # noqa: E402


def main():
    ds = SlopDataset("../data/cf_eval", transform=CaravelaCrops(), degrade=WebRealistic(seed=0))
    print(f"{len(ds)} images x 6 crops, max-aggregation, stock CF ViT-S fp32")
    model = HFDetector().cuda().eval()
    loader = DataLoader(ds, batch_size=10, num_workers=8, pin_memory=True)

    probs, labels = [], []
    with torch.no_grad():
        from tqdm import tqdm

        for x, y in tqdm(loader):
            b, n, c, h, w = x.shape
            p = model.predict_proba(x.view(b * n, c, h, w).cuda(non_blocking=True))
            probs.append(p.view(b, n).max(dim=1).values.float().cpu().numpy())
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)

    ev = Evaluator(threshold=0.65)
    res = ev.score(probs, labels, [s.source for s in ds.samples])
    print(res.summary())


if __name__ == "__main__":
    main()
