#!/usr/bin/env bash
# Mq4e8Combo arms (f)+(g): C1 artifact quality gates.
# (f) C1 on the FP8 route: HIPFIRE_* cleared, then HOME/ROCR/MODELS_DIR/KERNEL_CACHE
#     + HIPFIRE_IU4_PREFILL=0 HIPFIRE_IU4_XMASTER=0 HIPFIRE_GRAPH=0
#     (mirrors Mq4e8Study/weight_eval_results.jsonl env; binary forces NORMALIZE_PROMPT=0).
#     WT2-24 + AG-24. Bar 0.0744 WT2-24; fp8 baseline 0.048659.
# (g) C1 on the iu4 route: same pins as the combo sweep with HIPFIRE_LLOYD_GFX12=1,
#     HIPFIRE_IU4_PREFILL=1, HIPFIRE_IU4_XMASTER=0 (XMASTER-off analog of arms d/e).
#     WT2-24 + AG-24.
set -u
OUT=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Combo
BIN=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/examples/eval_hipfire
C1=/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c1.hfq
REF_WT2=/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin
REF_AG=/home/kaden/kldrefs/qwen3.8-27b.ref_ag.bin

run_fp8() {
  local tag="$1" ref="$2" chunks="$3"
  local log="$OUT/combo_${tag}.log" bin="$OUT/combo_${tag}.bin"
  local model_md5 ref_md5 bin_md5
  model_md5=$(md5sum "$C1" | cut -d' ' -f1)
  ref_md5=$(md5sum "$ref" | cut -d' ' -f1)
  bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$C1 model_md5=$model_md5 ref=$ref ref_md5=$ref_md5 binary=$BIN binary_md5=$bin_md5 chunks=$chunks route=fp8 HIPFIRE_IU4_PREFILL=0 HIPFIRE_IU4_XMASTER=0"
    echo "COMMAND HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_IU4_PREFILL=0 HIPFIRE_IU4_XMASTER=0 HIPFIRE_GRAPH=0 $BIN --model $C1 --ref $ref --output $bin --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks $chunks"
  } > "$log"
  (
    for v in $(env | cut -d= -f1 | grep '^HIPFIRE_' || true); do unset "$v"; done
    export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
    export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
    export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
    export HIPFIRE_IU4_PREFILL=0 HIPFIRE_IU4_XMASTER=0 HIPFIRE_GRAPH=0
    "$BIN" --model "$C1" --ref "$ref" --output "$bin" \
      --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks "$chunks" >> "$log" 2>&1
  )
  local rc=$?
  local kld="NONE" outmd5="NONE"
  kld=$(grep -o 'slice-mean KLD = [0-9.eE+-]*' "$log" | tail -n 1 || true)
  if [ -f "$bin" ]; then outmd5=$(md5sum "$bin" | cut -d' ' -f1); fi
  echo "EXIT_CODE $rc" >> "$log"
  echo "OUTBIN_MD5 $outmd5" >> "$log"
  echo "ROW tag=$tag chunks=$chunks rc=$rc $kld outbin_md5=$outmd5 model_md5=$model_md5"
}

run_iu4() {
  local tag="$1" ref="$2" chunks="$3" xmaster="$4"
  local log="$OUT/combo_${tag}.log" bin="$OUT/combo_${tag}.bin"
  local model_md5 ref_md5 bin_md5
  model_md5=$(md5sum "$C1" | cut -d' ' -f1)
  ref_md5=$(md5sum "$ref" | cut -d' ' -f1)
  bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$C1 model_md5=$model_md5 ref=$ref ref_md5=$ref_md5 binary=$BIN binary_md5=$bin_md5 chunks=$chunks HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster"
    echo "COMMAND HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster $BIN --model $C1 --ref $ref --output $bin --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks $chunks"
  } > "$log"
  (
    for v in $(env | cut -d= -f1 | grep '^HIPFIRE_' || true); do unset "$v"; done
    export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
    export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
    export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
    export HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
    export HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER="$xmaster"
    "$BIN" --model "$C1" --ref "$ref" --output "$bin" \
      --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks "$chunks" >> "$log" 2>&1
  )
  local rc=$?
  local kld="NONE" outmd5="NONE"
  kld=$(grep -o 'slice-mean KLD = [0-9.eE+-]*' "$log" | tail -n 1 || true)
  if [ -f "$bin" ]; then outmd5=$(md5sum "$bin" | cut -d' ' -f1); fi
  echo "EXIT_CODE $rc" >> "$log"
  echo "OUTBIN_MD5 $outmd5" >> "$log"
  echo "ROW tag=$tag chunks=$chunks xmaster=$xmaster rc=$rc $kld outbin_md5=$outmd5 model_md5=$model_md5"
}

# (f) C1 fp8 FIRST
run_fp8 "f_c1_fp8_wt2_24" "$REF_WT2" 24
run_fp8 "f_c1_fp8_ag_24"  "$REF_AG"  24
# (g) C1 iu4, XMASTER off
run_iu4 "g_c1_iu4_wt2_24" "$REF_WT2" 24 0
run_iu4 "g_c1_iu4_ag_24"  "$REF_AG"  24 0
echo SWEEP_FG_DONE
