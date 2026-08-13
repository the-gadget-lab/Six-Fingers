import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image

from slop.data import IMG_EXTS, WebRealistic

SRC = Path("../data/cf_eval")
DST = Path("../data/bench")
PER_SOURCE = 12


def main():
    degrade = WebRealistic(seed=1)
    rng = random.Random(1)
    n = 0
    for label_dir in sorted(SRC.iterdir()):
        for src_dir in sorted(label_dir.iterdir()):
            files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            for f in rng.sample(files, min(PER_SOURCE, len(files))):
                img = degrade(Image.open(f).convert("RGB"))
                out = DST / label_dir.name / src_dir.name / (f.stem + ".jpg")
                out.parent.mkdir(parents=True, exist_ok=True)
                if f.suffix.lower() in (".jpg", ".jpeg"):
                    img.save(out, "JPEG", quality=92)
                else:
                    img.save(out, "JPEG", quality=96)
                n += 1
    print(f"{n} bench images -> {DST}")


if __name__ == "__main__":
    main()
