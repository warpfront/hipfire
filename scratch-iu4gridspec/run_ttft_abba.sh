#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 gfx1100|gfx1151 HIP_VISIBLE_DEVICES LABEL" >&2
  exit 2
fi
arch="$1"
device="$2"
label="$3"
root=/home/kaden/hipfire-iu4gridspec
scratch="$root/scratch-iu4gridspec"
cli="$root/target/release/hipfire"
daemon="$root/target/release/daemon"
model=/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq
prompt="$root/benchmarks/prompts/ttft_5900.txt"
check="$scratch/check_e2e_json.py"
key=kernel.gfx11_iu4_gridspec
export HIP_VISIBLE_DEVICES="$device"
export HIPFIRE_DAEMON_BIN="$daemon"

run_ttft() {
  local arm="$1" value="$2"
  "$cli" config set "$key" "$value" > "$scratch/e2e-${label}-ttft-${arm}.config"
  "$cli" bench "$model" --ttft --prompt-file "$prompt" --spec off \
    --runs 8 --warmups 2 --json \
    > "$scratch/e2e-${label}-ttft-${arm}.json" \
    2> "$scratch/e2e-${label}-ttft-${arm}.stderr"
  "$check" "$scratch/e2e-${label}-ttft-${arm}.json" "$arch" 5909
}

run_ttft A1 false
run_ttft B1 true
run_ttft B2 true
run_ttft A2 false
"$cli" config set "$key" false > "$scratch/e2e-${label}-final.config"
