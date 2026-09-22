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
eval_bin="$root/target/release/examples/eval_hipfire"
model=/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq
key=kernel.gfx11_iu4_gridspec
export HIP_VISIBLE_DEVICES="$device"

run_eval() {
  local ref_name="$1" ref_path="$2" arm="$3" value="$4"
  local stem="quality-${label}-${ref_name}-${arm}"
  "$cli" config set "$key" "$value" > "$scratch/${stem}.config"
  "$eval_bin" --model "$model" --ref "$ref_path" \
    --output "$scratch/${stem}.kldseq" --max-chunks 24 \
    > "$scratch/${stem}.stdout" 2> "$scratch/${stem}.stderr"
  "$scratch/check_quality_log.py" "$scratch/${stem}.stderr" "$arch"
}

for ref_name in wt2 ag; do
  ref_path="/home/kaden/kldrefs/qwen3.8-27b.ref_${ref_name}.bin"
  run_eval "$ref_name" "$ref_path" off false
  run_eval "$ref_name" "$ref_path" on true
  cmp "$scratch/quality-${label}-${ref_name}-off.kldseq" \
      "$scratch/quality-${label}-${ref_name}-on.kldseq"
  sha256sum "$scratch/quality-${label}-${ref_name}-off.kldseq" \
            "$scratch/quality-${label}-${ref_name}-on.kldseq" \
      > "$scratch/quality-${label}-${ref_name}.sha256"
done
"$cli" config set "$key" false > "$scratch/quality-${label}-final.config"
