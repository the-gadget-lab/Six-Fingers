import torch
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CFPreprocess:
    """Community Forensics eval preprocessing: shortest edge 440 bicubic, center crop 384, CLIP norm."""

    def __init__(self, resize=440, crop=384):
        self.resize = resize
        self.crop = crop
        self.mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        from torchvision.transforms import functional as F

        img = img.convert("RGB")
        img = F.resize(img, self.resize, interpolation=F.InterpolationMode.BICUBIC)
        img = F.center_crop(img, self.crop)
        t = F.pil_to_tensor(img).float() / 255
        return (t - self.mean) / self.std
