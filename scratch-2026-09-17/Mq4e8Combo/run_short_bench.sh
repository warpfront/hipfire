#!/usr/bin/env bash
# Mq4e8Combo best-arm follow-ups. Best iu4-route arm under 0.0744 is (d) C2 alone
# on iu4 (WT2-24 0.061234): 1-chunk + 2-chunk WT2 rows (XMASTER=0, arm config),
# then hipfire bench matrix on the C2 artifact with XMASTER=1 (per spec) and
# XMASTER=0 (to report the bench effect of the extra quant pass).
set -u
OUT=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Combo
BIN=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/examples/eval_hipfire
HIPBIN=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/hipfire
C2=/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c2.hfq
REF_WT2=/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin

run_one() {
  local tag="$1" chunks="$2"
  local log="$OUT/combo_${tag}.log" bin="$OUT/combo_${tag}.bin"
  local model_md5 ref_md5 bin_md5
  model_md5=$(md5sum "$C2" | cut -d' ' -f1)
  ref_md5=$(md5sum "$REF_WT2" | cut -d' ' -f1)
  bin_md5=$(md5sum "$BIN" | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$C2 model_md5=$model_md5 ref=$REF_WT2 ref_md5=$ref_md5 binary=$BIN binary_md5=$bin_md5 chunks=$chunks HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=0"
    echo "COMMAND HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=0 $BIN --model $C2 --ref $REF_WT2 --output $bin --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks $chunks"
  } > "$log"
  (
    for v in $(env | cut -d= -f1 | grep '^HIPFIRE_' || true); do unset "$v"; done
    export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
    export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
    export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
    export HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
    export HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=0
    "$BIN" --model "$C2" --ref "$REF_WT2" --output "$bin" \
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

run_bench() {
  local tag="$1" xmaster="$2"
  local log="$OUT/combo_${tag}.log"
  local model_md5 hip_md5
  model_md5=$(md5sum "$C2" | cut -d' ' -f1)
  hip_md5=$(md5sum "$HIPBIN" | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$C2 model_md5=$model_md5 hipfire_binary=$HIPBIN hipfire_md5=$hip_md5 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster"
    echo "COMMAND HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster $HIPBIN bench $C2 --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode q8"
  } > "$log"
  (
    for v in $(env | cut -d= -f1 | grep '^HIPFIRE_' || true); do unset "$v"; done
    export HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
    export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
    export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels
    export HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1
    export HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER="$xmaster"
    "$HIPBIN" bench "$C2" --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 \
      --spec off --runs 3 --warmups 1 --kv-mode q8 >> "$log" 2>&1
  )
  local rc=$?
  echo "EXIT_CODE $rc" >> "$log"
  echo "ROW tag=$tag xmaster=$xmaster rc=$rc"
}

run_one "d_c2_iu4_wt2_1" 1
run_one "d_c2_iu4_wt2_2" 2
run_bench "bench_c2_xm1" 1
run_bench "bench_c2_xm0" 0
echo SHORT_BENCH_DONE
