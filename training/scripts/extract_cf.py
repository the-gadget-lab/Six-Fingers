import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

shard_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    Path("../data/hf/hub/datasets--OwensLab--CommunityForensics-Eval/snapshots").glob("*/data")
)
out = Path("../data/cf_eval")


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:80]


counts: dict[str, int] = {}
for shard in tqdm(sorted(shard_dir.glob("*.parquet"))):
    df = pd.read_parquet(shard)
    for _, row in df.iterrows():
        label = "fake" if row.label == 1 else "real"
        source = slug(row.model_name) if row.label == 1 else f"real_{slug(row.real_source)}"
        ext = (row.format or "PNG").lower().replace("jpeg", "jpg")
        d = out / label / source
        d.mkdir(parents=True, exist_ok=True)
        name = f"{shard.stem.split('-')[1]}_{slug(row.image_name)}"
        if not name.lower().endswith(f".{ext}"):
            name += f".{ext}"
        (d / name).write_bytes(row.image_data)
        counts[f"{label}/{source}"] = counts.get(f"{label}/{source}", 0) + 1

for k in sorted(counts):
    print(f"{counts[k]:6d}  {k}")
print(f"total {sum(counts.values())}")
