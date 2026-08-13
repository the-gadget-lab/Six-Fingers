import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

REAL, FAKE = 0, 1
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: int
    source: str


class WebRealistic:
    """Simulates what happens to images on the web: resize + JPEG requantization."""

    def __init__(self, jpeg_range=(55, 96), scale_range=(0.4, 1.0), p=0.85, seed=None):
        self.jpeg_range = jpeg_range
        self.scale_range = scale_range
        self.p = p
        self.rng = random.Random(seed) if seed is not None else random

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.rng.random() < self.p:
            scale = self.rng.uniform(*self.scale_range)
            if scale < 0.99:
                w, h = img.size
                img = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.BICUBIC)
        if self.rng.random() < self.p:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=self.rng.randint(*self.jpeg_range))
            buf.seek(0)
            img = Image.open(buf)
            img.load()
        return img.convert("RGB")


class SlopDataset(Dataset):
    """Directory layout: root/{real,fake}/<source>/*.jpg — label from top dir, source from subdir."""

    def __init__(self, root, transform, degrade: WebRealistic | None = None,
                 max_per_source: int | None = None, seed: int = 0):
        self.transform = transform
        self.degrade = degrade
        self.samples: list[Sample] = []
        rng = random.Random(seed)
        for label_name, label in (("real", REAL), ("fake", FAKE)):
            base = Path(root) / label_name
            if not base.is_dir():
                continue
            for src_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
                if max_per_source and len(files) > max_per_source:
                    files = rng.sample(files, max_per_source)
                self.samples += [Sample(f, label, src_dir.name) for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        if self.degrade:
            img = self.degrade(img)
        return self.transform(img), s.label

    def by_source(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, s in enumerate(self.samples):
            out.setdefault(s.source, []).append(i)
        return out
