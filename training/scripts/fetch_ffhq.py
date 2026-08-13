import io
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

OUT = Path("../data/train/real/ffhq")
TARGET = 12000

split_csv = hf_hub_download(
    "OwensLab/CommunityForensics-Eval", "other/ffhq_commfor_train_split.csv", repo_type="dataset"
)
train_names = set(pd.read_csv(split_csv).Filename)
print(f"{len(train_names)} FFHQ train-split names")

shards = sorted(
    f for f in HfApi().list_repo_files("Ryan-sjtu/ffhq512-caption", repo_type="dataset")
    if f.endswith(".parquet")
)
OUT.mkdir(parents=True, exist_ok=True)
saved = 0
for shard in shards[:: max(1, len(shards) // 12)]:
    p = hf_hub_download("Ryan-sjtu/ffhq512-caption", shard, repo_type="dataset")
    df = pd.read_parquet(p)
    for _, row in df.iterrows():
        name = row.image["path"]
        if name not in train_names:
            continue
        img = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
        img.save(OUT / (Path(name).stem + ".jpg"), "JPEG", quality=95)
        saved += 1
    print(f"{shard}: total {saved}", flush=True)
    if saved >= TARGET:
        break
print(f"DONE ffhq {saved}")
