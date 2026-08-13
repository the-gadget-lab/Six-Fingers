import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slop.data import IMG_EXTS

EXTRA = Path("../data/extra")
TRAIN = Path("../data/train")
EVAL = Path("../data/extra_eval")
EVAL_FRACTION = 5


def main():
    seen: set[str] = set()
    rows = []
    dupes = 0
    for label_dir in (EXTRA / "fake", EXTRA / "real"):
        for src_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            for f in sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
                if digest in seen:
                    dupes += 1
                    f.unlink()
                    continue
                seen.add(digest)
                is_eval = int(digest[:8], 16) % EVAL_FRACTION == 0
                dest_root = EVAL if is_eval else TRAIN
                dest = dest_root / label_dir.name / f"x_{src_dir.name}" / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.is_symlink():
                    dest.symlink_to(f.resolve())
                rows.append({
                    "path": str(f.relative_to(EXTRA)), "label": label_dir.name,
                    "source": src_dir.name, "split": "eval" if is_eval else "train",
                })

    with open(EXTRA / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "split"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} images ({dupes} duplicates removed)")
    from collections import Counter

    for (label, source, split), n in sorted(Counter((r["label"], r["source"], r["split"]) for r in rows).items()):
        print(f"  {label}/{source} [{split}]: {n}")


if __name__ == "__main__":
    main()
