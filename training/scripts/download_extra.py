"""Assemble modern-AI fake + web-real image pool under data/extra/.

Usage: uv run python scripts/download_extra.py <job> [limit]
Jobs: oip, midjourney, openfake, dalle3, unsplash, flickr, finalize
Each job writes images to data/extra/{fake,real}/<source>/ and appends a
per-job manifest chunk; `finalize` dedupes globally and writes manifest.csv.
"""
import csv
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

EXTRA = Path("/home/lokman/projects/ai-slop-detect/data/extra")
MANIFESTS = EXTRA / "manifests"
MIN_SIDE = 200

EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


class Sink:
    """Saves raw image bytes as-is, with hash dedupe + min-side filter."""

    def __init__(self, job):
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        self.mf = open(MANIFESTS / f"{job}.csv", "a", newline="")
        self.w = csv.writer(self.mf)
        self.seen = set()
        self.counts = {}

    def save(self, data, label, source, detail=""):
        if not data:
            return False
        h = hashlib.sha256(data).hexdigest()
        if h in self.seen:
            return False
        try:
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            w, hgt = im.size
        except Exception:
            return False
        if min(w, hgt) < MIN_SIDE or fmt not in EXT:
            return False
        self.seen.add(h)
        d = EXTRA / label / source
        d.mkdir(parents=True, exist_ok=True)
        p = d / (h[:20] + EXT[fmt])
        p.write_bytes(data)
        self.w.writerow([str(p), label, source, detail])
        self.counts[source] = self.counts.get(source, 0) + 1
        return True

    def flush(self):
        self.mf.flush()


def img_bytes(row_val):
    """Extract raw bytes from a datasets Image(decode=False) value."""
    if isinstance(row_val, dict) and row_val.get("bytes"):
        return row_val["bytes"]
    return None


def undecoded(ds, cols):
    import datasets as hfds

    for c in cols:
        ds = ds.cast_column(c, hfds.Image(decode=False))
    return ds


def job_oip(limit=1600):
    """open-image-preferences-v1: flux-dev + sd3.5 columns, one pass."""
    from datasets import load_dataset

    cols = ["image_quality_dev", "image_simplified_dev",
            "image_quality_sd", "image_simplified_sd"]
    ds = undecoded(load_dataset("data-is-better-together/open-image-preferences-v1",
                                split="cleaned", streaming=True), cols)
    s = Sink("oip")
    for row in ds:
        for c in cols:
            src = "flux-dev" if c.endswith("_dev") else "sd35"
            detail = "flux.1-dev" if src == "flux-dev" else "sd-3.5-large"
            s.save(img_bytes(row[c]), "fake", src, detail)
        if s.counts.get("flux-dev", 0) >= limit and s.counts.get("sd35", 0) >= limit:
            break
        if sum(s.counts.values()) % 200 < 4:
            s.flush()
            print(s.counts, flush=True)
    s.flush()
    print("DONE oip", s.counts)


def job_midjourney(limit=1500):
    from datasets import load_dataset

    ds = undecoded(load_dataset("brivangl/midjourney-v6-llava",
                                split="train", streaming=True), ["image"])
    s = Sink("midjourney")
    for row in ds:
        s.save(img_bytes(row["image"]), "fake", "midjourney-v6", "midjourney-v6")
        n = s.counts.get("midjourney-v6", 0)
        if n >= limit:
            break
        if n % 200 == 0:
            s.flush()
            print(s.counts, flush=True)
    s.flush()
    print("DONE midjourney", s.counts)


def job_openfake_real(limit=1200):
    """Stream OpenFake core/train, keep only real rows (laion + pexels).

    ponytail: fake models in OpenFake are ~80 generators shuffled at ~1%% each;
    pulling 1200 of one would cost ~100GB of transfer, so fakes come from
    other datasets and this pass only harvests the ~50%% real rows.
    """
    from datasets import load_dataset

    ds = undecoded(load_dataset("ComplexDataLab/OpenFake", "core",
                                split="train", streaming=True), ["image"])
    s = Sink("openfake")
    src = {"laion": "laion-photos", "pexels": "pexels"}
    for i, row in enumerate(ds):
        if row["label"] == "real":
            name = src.get(row["model"], "laion-photos")
            if s.counts.get(name, 0) < limit:
                s.save(img_bytes(row["image"]), "real", name, row["model"])
        if i % 500 == 0:
            s.flush()
            print(i, s.counts, flush=True)
        if all(s.counts.get(v, 0) >= limit for v in src.values()):
            break
    s.flush()
    print("DONE openfake", s.counts)


def job_rapidata(repo, model_prefixes, source, detail_fallback, limit=1400):
    """Rapidata t2i preference sets: matchup rows image1/image2 + model1/model2."""
    from datasets import load_dataset

    ds = undecoded(load_dataset(repo, split="train", streaming=True),
                   ["image1", "image2"])
    s = Sink(source)
    for row in ds:
        for k in ("1", "2"):
            m = (row.get(f"model{k}") or "").lower()
            if any(m.startswith(p) for p in model_prefixes):
                s.save(img_bytes(row[f"image{k}"]), "fake", source, m or detail_fallback)
        n = s.counts.get(source, 0)
        if n >= limit:
            break
        if n % 200 < 2:
            s.flush()
    s.flush()
    print(f"DONE {source}", s.counts)


def job_gptimgeval():
    """Unzip cached GenEval.zip (GPT-4o generations) into fake/gpt4o."""
    import zipfile
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Yejy53/GPT-ImgEval", "GenEval.zip", repo_type="dataset")
    s = Sink("gptimgeval")
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                s.save(z.read(n), "fake", "gpt4o", "gpt-4o")
    s.flush()
    print("DONE gptimgeval", s.counts)


def job_dalle3(limit=1400):
    """Stream ProGamerGov dalle3 tar over HTTP, stop early."""
    url = ("https://huggingface.co/datasets/ProGamerGov/"
           "synthetic-dataset-1m-dalle3-high-quality-captions/resolve/main/data/data-000000.tar")
    s = Sink("dalle3")
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as resp:
        with tarfile.open(fileobj=resp, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                data = tf.extractfile(member).read()
                s.save(data, "fake", "dalle3", "dall-e-3")
                n = s.counts.get("dalle3", 0)
                if n >= limit:
                    break
                if n % 200 == 0:
                    s.flush()
                    print(s.counts, flush=True)
    s.flush()
    print("DONE dalle3", s.counts)


def fetch(url, timeout=30, max_bytes=15_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research dataset collection)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            return None
        return r.read(max_bytes + 1)[:max_bytes]


def parallel_fetch_save(urls_details, s, label, source, limit, workers=8):
    it = iter(urls_details)
    with ThreadPoolExecutor(workers) as ex:
        futs = {}
        def submit_more():
            while len(futs) < workers * 2:
                try:
                    url, detail = next(it)
                except StopIteration:
                    return
                futs[ex.submit(fetch, url)] = detail
        submit_more()
        while futs and s.counts.get(source, 0) < limit:
            for f in as_completed(list(futs)):
                detail = futs.pop(f)
                try:
                    data = f.result()
                except Exception:
                    data = None
                if data:
                    s.save(data, label, source, detail)
                n = s.counts.get(source, 0)
                if n and n % 100 == 0:
                    s.flush()
                    print(s.counts, flush=True)
                if n >= limit:
                    return
                submit_more()
                break


def job_unsplash(limit=1500):
    import zipfile
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "unsplash-lite.zip"
    if not tmp.exists():
        print("downloading unsplash lite tsv zip...", flush=True)
        urllib.request.urlretrieve("https://unsplash.com/data/lite/latest", tmp)
    urls = []
    with zipfile.ZipFile(tmp) as z:
        name = [n for n in z.namelist() if n.startswith("photos.tsv")][0]
        import csv as _csv
        rows = _csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8"), delimiter="\t")
        for r in rows:
            u = r.get("photo_image_url")
            if u:
                urls.append((u + "?w=1200&fm=jpg&q=85", "unsplash-lite"))
    print(len(urls), "unsplash urls", flush=True)
    s = Sink("unsplash")
    parallel_fetch_save(urls, s, "real", "unsplash", limit)
    s.flush()
    print("DONE unsplash", s.counts)


def job_flickr(limit=1000):
    from datasets import load_dataset

    ds = load_dataset("Chr0my/public_flickr_photos_license_1", split="train", streaming=True)
    def gen():
        for row in ds:
            yield row["url"], "flickr-cc-by-nc-sa"
    s = Sink("flickr")
    # ponytail: many urls are _o originals; fetch() caps at 15MB
    parallel_fetch_save(gen(), s, "real", "flickr", limit, workers=14)
    s.flush()
    print("DONE flickr", s.counts)


def finalize():
    """Global content-hash dedupe across all sources + write manifest.csv."""
    rows = []
    for mf in sorted(MANIFESTS.glob("*.csv")):
        with open(mf) as f:
            rows.extend(csv.reader(f))
    seen, keep = set(), []
    for path, label, source, detail in rows:
        p = Path(path)
        if not p.exists():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen:
            p.unlink()
            continue
        seen.add(h)
        keep.append([path, label, source, detail])
    with open(EXTRA / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "generator_detail"])
        w.writerows(keep)
    # summary
    from collections import defaultdict
    cnt, byt = defaultdict(int), defaultdict(int)
    for path, label, source, _ in keep:
        cnt[(label, source)] += 1
        byt[(label, source)] += Path(path).stat().st_size
    print(f"{'label':6s} {'source':18s} {'count':>7s} {'MB':>9s}")
    for k in sorted(cnt):
        print(f"{k[0]:6s} {k[1]:18s} {cnt[k]:7d} {byt[k]/1e6:9.1f}")
    print(f"TOTAL {sum(cnt.values())} images, {sum(byt.values())/1e9:.2f} GB")


def main():
    job = sys.argv[1]
    if job == "oip":
        job_oip()
    elif job == "midjourney":
        job_midjourney()
    elif job == "openfake":
        job_openfake_real(int(sys.argv[2]) if len(sys.argv) > 2 else 1200)
    elif job == "imagen":
        job_rapidata("Rapidata/Imagen4_t2i_human_preference",
                     ("imagen",), "imagen4", "imagen-4", limit=1400)
    elif job == "gpt4o-rapidata":
        job_rapidata("Rapidata/OpenAI-4o_t2i_human_preference",
                     ("openai", "gpt", "4o"), "gpt4o", "gpt-4o", limit=1000)
    elif job == "gptimgeval":
        job_gptimgeval()
    elif job == "dalle3":
        job_dalle3()
    elif job == "unsplash":
        job_unsplash()
    elif job == "flickr":
        job_flickr()
    elif job == "finalize":
        finalize()
    else:
        raise SystemExit(f"unknown job {job}")


if __name__ == "__main__":
    main()
