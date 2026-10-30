#!/usr/bin/env bash
set -euo pipefail
cd /home/husrcf/Code/hipfire-fn-expert-lru-beta-20261025
source scripts/gpu-lock.sh
export GPU_LOCK_TIMEOUT=60
gpu_acquire "flash-next-expert-cache-S0"
trap gpu_release EXIT
timeout --signal=TERM --kill-after=10s 120s /home/husrcf/wk/hipfire-psr/target/release/examples/fn_expert_table_indirection --execute --hip-graph
