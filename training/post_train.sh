#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
until grep -q "PIPELINE_DONE" pipeline2.log 2>/dev/null; do sleep 60; done
cd ../extension
npm run build
node e2e/smoke.mjs
uv run --project ../training python ../training/scripts/make_bench.py || true
node e2e/bench.mjs
echo POST_TRAIN_DONE
