#!/usr/bin/env bash
# Mq4e8Combo bench rows on the C2 artifact (best iu4-route arm under 0.0744 is
# (d) C2-alone; bench runs the artifact with XMASTER=1 per spec, then XMASTER=0
# to report the bench effect of the extra quant pass).
set -u
OUT=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Combo
HIPBIN=/home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/hipfire
C2=/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c2.hfq

run_bench() {
  local tag="$1" xmaster="$2"
  local log="$OUT/combo_${tag}.log"
  local model_md5 hip_md5 daemon_md5
  model_md5=$(md5sum "$C2" | cut -d' ' -f1)
  hip_md5=$(md5sum "$HIPBIN" | cut -d' ' -f1)
  daemon_md5=$(md5sum /home/kaden/ClaudeCode/warpfront/wt-mq4e8/target/release/daemon | cut -d' ' -f1)
  {
    echo "PROVENANCE tag=$tag model=$C2 model_md5=$model_md5 hipfire_binary=$HIPBIN hipfire_md5=$hip_md5 daemon_md5=$daemon_md5 HIPFIRE_IU4_PREFILL=1 HIPFIRE_IU4_XMASTER=$xmaster"
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

run_bench "bench_c2_xm1" 1
run_bench "bench_c2_xm0" 0
echo BENCH_DONE
