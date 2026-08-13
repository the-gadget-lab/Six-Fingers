# Slop Detect: Local AI Image Detector for Chrome

A Manifest V3 Chrome extension that detects AI-generated images **entirely
in your browser**. No cloud APIs, no local servers, no image ever leaves
your device. Submission for [poidh bounty #323](https://poidh.xyz/arbitrum/bounty/323).

- **Model**: [Community Forensics ViT-S/384](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
  (CVPR 2025, MIT), fine-tuned for web-realistic conditions (JPEG
  recompression, downscaling) on 2024-25 generators (FLUX, SD3.5,
  Midjourney v6, DALL-E 3, GPT-4o, Imagen 4), calibrated so the optimal
  operating point sits at the required 65% confidence threshold.
- **Inference**: ONNX Runtime Web (WASM + SIMD), int8-quantized ViT-S
  (~23 MB) running in an offscreen document. The model ships inside the
  extension package, so it is fully offline from the moment it is installed.
- **UI**: every `<img>` on the page (≥96 px) gets a confidence badge;
  images scoring ≥65% are flagged red as AI.

## Install (from source)

Requires Node.js ≥ 20.

```bash
cd extension
npm ci
npm run build
```

The build fetches the model weights once from
[Loke-60000/slop-detect-vit-s-onnx](https://huggingface.co/Loke-60000/slop-detect-vit-s-onnx)
(23MB, SHA-256 verified) and bundles them into the extension package.
The installed extension performs no downloads and runs fully offline.
CI builds both targets on every push (`.github/workflows/build.yml`).

Then open `chrome://extensions`, enable Developer mode, click
"Load unpacked", and select `extension/dist/`. That's it: visit any
page with images.

## Architecture

```
content script (content.ts)
  finds <img> elements, queues them, renders badges
    │ chrome.runtime message
service worker (background.ts)
  routes score requests, caches by URL, owns the offscreen document
    │ chrome.runtime message
offscreen document (offscreen.ts)
  fetches image bytes (extension-privileged, CORS-proof),
  preprocess: resize shortest edge → 440, center-crop 384, CLIP norm,
  ONNX Runtime Web session (WASM+SIMD, single thread) → sigmoid(logit)
```

All components are plain TypeScript classes bundled by esbuild
(`extension/build.mjs`). The ONNX runtime and model weights are packaged
with the extension; the manifest CSP allows only `'self'` +
`'wasm-unsafe-eval'`, and the extension performs no network requests
other than fetching the images already displayed by the page.

## Reproducing the model

The full training/evaluation pipeline is in `training/` (uv project):

```bash
cd training
uv sync
uv run python scripts/fetch_cf_small.py     # fake training images (CF-Small subsample)
uv run python scripts/extract_cf.py         # held-out eval set (CF-Eval subsample)
uv run python scripts/train_ft.py           # fine-tune ViT-S on web-degraded data
uv run python scripts/export.py --ckpt checkpoints/best.pt   # calibrate + ONNX + int8
```

Training data: [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small)
(fakes, train split), COCO train2017 (reals), plus 2024-25 generator
samples listed in `data/extra/LICENSES.md`. Evaluation:
[CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)
subsample (21 held-out generators incl. FLUX, Firefly 3, Ideogram v2,
Imagen 3, Midjourney v6.1) with simulated web degradation.

## Benchmarks (held-out, degraded, 65% threshold)

| stage | balanced accuracy |
|---|---|
| stock CF ViT-S int8, through the extension E2E | 78.5% |
| **fine-tuned + calibrated int8, through the extension E2E** | **85.3%** |

Full tables and methodology: [docs/RESULTS.md](docs/RESULTS.md).

The E2E benchmark drives the actual built extension in headless Chrome:

```bash
cd training && uv run python scripts/make_bench.py
cd ../extension && node e2e/bench.mjs
```

## License

MIT, see [LICENSE](LICENSE).
