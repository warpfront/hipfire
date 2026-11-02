#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Native-vs-optional-CK packed-KV prefill A/B on the production Qwen bench path.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${SIDECAR:-$PWD/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so}"
EXE="${EXE:-$PWD/target/release/examples/bench_qwen35_mq4}"
GPU_ID="${GPU_ID:-0}"
KV_MODE="${KV_MODE:-q8}"
PREFILL="${PREFILL:-8192}"
RUNS="${RUNS:-5}"
WORKSPACE_BYTES="${WORKSPACE_BYTES:-536870912}"
SLEEP_SECS="${SLEEP_SECS:-10}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/ck_${KV_MODE}_prefill_ab_$(date +%Y%m%d_%H%M%S)}"

[[ -f "$MODEL" ]] || { echo "missing MODEL=$MODEL" >&2; exit 2; }
[[ -f "$SIDECAR" ]] || { echo "missing SIDECAR=$SIDECAR" >&2; exit 2; }
if [[ "${BUILD:-1}" == "1" ]]; then
    cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 \
        --features deltanet,flash-attn-ck
fi
mkdir -p "$RESULT_DIR"
printf 'mode\tprefill_tok_s\n' >"$RESULT_DIR/summary.tsv"

run_mode() {
    local mode="$1"
    local log="$RESULT_DIR/${mode}.log"
    if [[ "$mode" == "ck" ]]; then
        export HIPFIRE_FLASH_ATTN_CK_LIB="$SIDECAR"
        export HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES="$WORKSPACE_BYTES"
    else
        unset HIPFIRE_FLASH_ATTN_CK_LIB HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES
    fi
    HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_KV_MODE="$KV_MODE" \
        "$EXE" "$MODEL" --prefill "$PREFILL" --prefill-runs "$RUNS" --warmup 2 --gen 1 \
        2>&1 | tee "$log"
    awk -v mode="$mode" '/run[[:space:]]+[0-9]+:/ { print mode "\t" $4 }' "$log" \
        >>"$RESULT_DIR/summary.tsv"
    awk -F= '/PREFILL_NEXT_TOKEN/ { print $2 }' "$log" | tail -1 \
        >"$RESULT_DIR/${mode}.next_token"
}

{
    echo "git_head=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "sidecar=$SIDECAR"
    echo "prefill=$PREFILL"
    echo "kv_mode=$KV_MODE"
    echo "runs=$RUNS"
    echo "workspace_bytes=$WORKSPACE_BYTES"
    rocm-smi --showproductname --showmeminfo vram
} >"$RESULT_DIR/meta.txt" 2>&1

run_mode native
sleep "$SLEEP_SECS"
run_mode ck

python3 - "$RESULT_DIR/summary.tsv" "$RESULT_DIR" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
groups = {mode: [float(r["prefill_tok_s"]) for r in rows if r["mode"] == mode]
          for mode in ("native", "ck")}
for mode, values in groups.items():
    print(f"{mode}: median={statistics.median(values):.3f} tok/s n={len(values)}")
if all(groups.values()):
    print(f"delta_ck_vs_native={(statistics.median(groups['ck']) / statistics.median(groups['native']) - 1) * 100:.2f}%")
native_token = open(f"{sys.argv[2]}/native.next_token").read().strip()
ck_token = open(f"{sys.argv[2]}/ck.next_token").read().strip()
print(f"prefill_next_token_native={native_token} ck={ck_token}")
if not native_token or native_token != ck_token:
    raise SystemExit("native/CK prefill next-token mismatch")
PY

echo "results: $RESULT_DIR"
