import torch
from PIL import Image

from .cf import CFPreprocess


class FiveCrop384:
    """Resize shortest edge to 440, take center + 4 corner 384-crops. Returns [5,3,384,384]."""

    def __init__(self, resize=440, crop=384):
        self.resize = resize
        self.crop = crop
        self.post = CFPreprocess(resize, crop)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        from torchvision.transforms import functional as F

        img = F.resize(img.convert("RGB"), self.resize, interpolation=F.InterpolationMode.BICUBIC)
        w, h = img.size
        s = self.crop
        origins = [((w - s) // 2, (h - s) // 2), (0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
        t = F.pil_to_tensor(img).float() / 255
        t = (t - self.post.mean) / self.post.std
        return torch.stack([t[:, y : y + s, x : x + s] for x, y in origins])


@torch.no_grad()
def crop_logits(model, loader, device="cuda"):
    """Mean logit over the 5 crops for every image. Returns (logits[N], labels[N])."""
    import numpy as np
    from tqdm import tqdm

    zs, ys = [], []
    model.eval().to(device)
    for x, y in tqdm(loader, leave=False):
        b, n, c, h, w = x.shape
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x.view(b * n, c, h, w).to(device, non_blocking=True))
        zs.append(z.view(b, n).mean(dim=1).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)
