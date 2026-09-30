#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 HUSRCF
# Developer-only smoke; not a performance benchmark or promotion gate.
set -euo pipefail
cd "$(dirname "$0")/.."
result_dir="${1:?usage: bash scripts/test-packed-mq4-smoke.sh NEW_RESULT_DIRECTORY [GPU]}"
gpu_index="${2:-1}"
model_dir="${HIPFIRE_MODELS_DIR:-${HOME}/.hipfire/models}"
if [[ -e "$result_dir" ]]; then
    echo "Refusing to overwrite existing results: $result_dir" >&2
    exit 2
fi
mkdir -p "$result_dir"
result_dir="$(realpath "$result_dir")"
source scripts/gpu-lock.sh
gpu_acquire packed-mq4-smoke
trap gpu_release EXIT
sha256sum target/release/daemon target/release/hipfire \
    benchmarks/prompts/adaptive_kv_long_prefill.txt > "$result_dir/identities.sha256"
git rev-parse HEAD > "$result_dir/base-commit.txt"
git diff --binary > "$result_dir/tracked-source.diff"
sha256sum crates/rdna-compute/src/packed_mq4.rs \
    kernels/src/gemm_mq4_packed.gfx1100.hip >> "$result_dir/identities.sha256"
for format in mq4 mq4v2; do
    if [[ "$format" == mq4 ]]; then
        model="$model_dir/qwen3.6-27b.mq4"
    else
        model="$model_dir/qwen3.8-27b.mq4-xt"
    fi
    sha256sum "$model" >> "$result_dir/identities.sha256"
    smoke_home="$(mktemp -d "/tmp/hipfire-packed-${format}.XXXXXX")"
    HIP_VISIBLE_DEVICES="$gpu_index" HIPFIRE_GFX1100_PACKED_MQ4_PREFILL=1 \
    HIPFIRE_PREFILL_CHUNK_ROWS=512 timeout 900s python3 scripts/serve_harness.py \
        --model "$model" --speculation off --thinking off --max-think-tokens 1 \
        --sampling greedy --kv q8 --max-seq 4096 --max-tokens 1024 --mode battery \
        --prompt-file benchmarks/prompts/adaptive_kv_long_prefill.txt \
        --home "$smoke_home" --port 11793 --serve-warm-timeout-secs 600 \
        --serve-log "$result_dir/$format-serve.log" --out "$result_dir/$format.json" \
        > "$result_dir/$format-harness.log" 2>&1
done
# Raw outputs are never reformatted or edited after this point.
sha256sum "$result_dir/mq4.json" "$result_dir/mq4v2.json" \
    "$result_dir/mq4-serve.log" "$result_dir/mq4v2-serve.log" \
    "$result_dir/mq4-harness.log" "$result_dir/mq4v2-harness.log" \
    > "$result_dir/raw.sha256"
