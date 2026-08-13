#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/home/lokman/projects/ai-slop-detect/data/hf

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for CF-Small shards (max 30 min)"
for _ in $(seq 180); do
  n=$(ls ../data/train/fake/cf_small/.done_* 2>/dev/null | wc -l)
  [ "$n" -ge 94 ] && break
  sleep 10
done
log "cf_small shards: $(ls ../data/train/fake/cf_small/.done_* 2>/dev/null | wc -l)/94"

log "waiting for extra downloads to finish (max 30 min)"
for _ in $(seq 180); do
  pgrep -f download_extra > /dev/null || break
  sleep 10
done
pgrep -f download_extra > /dev/null && log "extra downloads still running; proceeding with current files"

log "staging extra data (dedupe + split + manifest)"
uv run python scripts/stage_extra.py

log "fine-tuning (early stop on val plateau, patience 2)"
uv run python scripts/train_ft.py --epochs 10 --batch 48 --max-per-source 80000

log "calibrating + exporting ONNX (fp32 -> int8)"
uv run python scripts/export.py --ckpt checkpoints/best.pt

log "quantization comparison on tuned model"
uv run python scripts/quant_compare.py ../model/dist/model_fp32.onnx ../model/dist/model.onnx || true

log "PIPELINE_DONE"
