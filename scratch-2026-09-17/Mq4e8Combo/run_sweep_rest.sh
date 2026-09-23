#!/usr/bin/env bash
# Mq4e8Combo remaining combo arms after C1 prioritization.
# (b)-AG rerun (WT2 done: 0.068969), (c) C4+XMASTER x2, (d) C2-alone x2, (e) C4-alone x2.
set -u
OUT=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Combo
BIN=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/examples/eval_hipfire
C2=/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c2.hfq
C4=/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c4.hfq
REF_WT2=/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin
REF_AG=/home/kaden/kldrefs/qwen3.8-27b.ref_ag.bin

run_one() {
  local tag="$1" model="$2" ref="$3" chunks="$4" xmaster="$5"
  local log="$OUT/combo_${tag}.log" bin="$OUT/combo_${tag}.bin"
  local model_md5 ref_md5 bin_md5
  model_md5=$(md5sum "$model" | cut -d' ' -f1)
  ref_md5=$(md5sum "$ref" | cut -d' ' -f1)
  bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$model model_md5=$model_md5 ref=$ref ref_md5=$ref_md5 binary=$BIN binary_md5=$bin_md5 chunks=$chunks HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster"
    echo "COMMAND HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster $BIN --model $model --ref $ref --output $bin --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks $chunks"
  } > "$log"
  (
    for v in $(env | cut -d= -f1 | grep '^HIPFIRE_' || true); do unset "$v"; done
    export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
    export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
    export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
    export HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
    export HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER="$xmaster"
    "$BIN" --model "$model" --ref "$ref" --output "$bin" \
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

run_one "b_c2_xm_ag_24"   "$C2" "$REF_AG"  24 1
run_one "c_c4_xm_wt2_24"  "$C4" "$REF_WT2" 24 1
run_one "c_c4_xm_ag_24"   "$C4" "$REF_AG"  24 1
run_one "d_c2_iu4_wt2_24" "$C2" "$REF_WT2" 24 0
run_one "d_c2_iu4_ag_24"  "$C2" "$REF_AG"  24 0
run_one "e_c4_iu4_wt2_24" "$C4" "$REF_WT2" 24 0
run_one "e_c4_iu4_ag_24"  "$C4" "$REF_AG"  24 0
echo SWEEP_REST_DONE
