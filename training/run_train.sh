#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/home/lokman/projects/ai-slop-detect/data/hf

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "staging extra data"
uv run python scripts/stage_extra.py

log "fine-tuning (early stop on val plateau, patience 2)"
uv run python scripts/train_ft.py --epochs 10 --batch 48 --max-per-source 80000

log "calibrating + exporting ONNX (fp32 -> int8)"
uv run python scripts/export.py --ckpt checkpoints/best.pt

log "quantization comparison on tuned model"
uv run python scripts/quant_compare.py ../model/dist/model_fp32.onnx ../model/dist/model.onnx || true

log "PIPELINE_DONE"
