import io
import os
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

OUT = Path("../data/train/fake/cf_small")
PER_SHARD = 800
MAX_SIDE = 512


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:60]


def save_row(row, out_path: Path):
    img = Image.open(io.BytesIO(row.image_data)).convert("RGB")
    if min(img.size) > MAX_SIDE:
        s = MAX_SIDE / min(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    img.save(out_path, "JPEG", quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(0, 188, 2):
        marker = OUT / f".done_{i}"
        if marker.exists():
            continue
        f = f"data/HFCF_small_{i}.parquet"
        p = hf_hub_download("OwensLab/CommunityForensics-Small", f, repo_type="dataset")
        df = pd.read_parquet(p)
        step = max(1, len(df) // PER_SHARD)
        picked = df.iloc[::step][:PER_SHARD]
        for j, row in picked.iterrows():
            name = f"s{i}_{j}_{slug(row.model_name)}.jpg"
            try:
                save_row(row, OUT / name)
            except Exception as e:
                print(f"skip {name}: {e}", flush=True)
        os.remove(p)
        marker.touch()
        print(f"shard {i}: {len(picked)} saved", flush=True)


if __name__ == "__main__":
    main()
