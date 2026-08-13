# Benchmark results

All numbers are balanced accuracy at the required **65% confidence
threshold**, on held-out data: a 6,523-image subsample of
[CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval)
(21 generators disjoint from training, including FLUX dev/schnell,
Firefly 2/3, Ideogram v1/v2, Imagen 3, Midjourney v5.2/v6.1 and DALL-E 2/3,
plus reals from RAISE, LAION, COCO-val, FFHQ-test, ImageNet), under two
web-degradation regimes.

| model @0.65 | mild degradation¹ (6,523 img, fp32) | harsh degradation² (312 img, int8, through the extension in Chrome) |
|---|---|---|
| stock CF ViT-S, uncalibrated | 75.4% | 78.5% |
| stock CF ViT-S, calibrated | 84.3% | 82.6% |
| **fine-tuned + calibrated (shipped)** | 82.5% | **85.3%** (TPR 80.6% / TNR 90.0%) |

¹ random downscale to 40–100% + JPEG q55–96 (`WebRealistic`, seed 0)
² same degradation followed by JPEG re-encode (seed 1), which approximates
images that platforms have recompressed; scored end-to-end via
`extension/e2e/bench.mjs` driving the built extension in headless Chrome.

## Why the fine-tuned model ships

The two candidates are close overall, but they fail differently: under
harsher recompression the stock model's real-image accuracy collapses
(TNR 89.9% → 78.3%) while the fine-tuned model, trained under heavy
JPEG/resize augmentation on 177k images, holds TNR at 90% and stays
above 85% balanced accuracy. "Web-realistic" evaluation favors the model
that is robust to compression, and false-flagging real photos is the most
damaging failure mode in practice.

## Fine-tune recipe (reproducible via `training/run_train.sh`)

- Base: Community Forensics ViT-S/384 (CVPR 2025, MIT)
- Data: 81.5k fakes (CF-Small train subsample across ~2,400 generators +
  FLUX/SD3.5/MJ-v6/DALL-E-3/GPT-4o/Imagen-4 samples) + 96k reals
  (COCO train2017, FFHQ train split, Unsplash/Pexels/Flickr/LAION photos)
- Per-sample random degradation: downscale 25–100%, JPEG q40–98
- 10-epoch cap, early stopping on held-out balanced accuracy (stopped at
  epoch 1 by patience 2), AdamW, OneCycle 1e-5 (head ×10), bf16
- Calibration: single logit bias (+Platt shift) fitted on the validation
  sweep so the balanced-accuracy-optimal operating point sits exactly at
  the required 0.65 displayed confidence; baked into the ONNX graph
- Quantization: dynamic int8 (23MB). Measured int8 ≥ fp32 on both regimes;
  int4 rejected (−6pts, +46% latency vs int8)

## Performance

~450ms per image in-browser (WASM + SIMD, single thread, Ryzen-class
desktop CPU); ~2.2 images/s sustained page scan with concurrency 4.
