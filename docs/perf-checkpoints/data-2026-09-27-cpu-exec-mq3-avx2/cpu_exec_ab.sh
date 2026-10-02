#!/usr/bin/env bash
# A/B driver for the qt49 AVX2 kernel, kept beside the JSON it produced.
# Scratch during the measurement; committed here so the numbers have a recipe.
# Pre-change pair: /tmp/cpu_exec_pin/{hipfire,daemon}
# Post-change pair: target/release/{hipfire,daemon}
set -u
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Point MODEL at the local qwen3.8-27b.mq3-xt artifact.
MODEL="${MODEL:?set MODEL to the local qwen3.8-27b.mq3-xt path}"
PROMPT=benchmarks/prompts/gpu_offload_probe.txt
OUT=/tmp/cpu_exec_ab
mkdir -p "$OUT"
COMMON=(--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots
        --workload stateless --prompt-file "$PROMPT" --json)

wait_for_no_daemon() {
  for _ in $(seq 40); do
    pgrep -f 'release/daemon' >/dev/null || pgrep -f 'cpu_exec_pin/daemon' >/dev/null || return 0
    sleep 0.5
  done
  echo "WARN: daemon still alive" | tee -a "$OUT/run.log"
}

run_arm() { # tag cli daemon exec
  local tag=$1 cli=$2 daemon=$3 exec=$4
  pkill -f 'release/daemon' 2>/dev/null
  pkill -f 'cpu_exec_pin/daemon' 2>/dev/null
  wait_for_no_daemon
  printf '=== %s start %s load=%s bin=%s\n' "$tag" "$(date +%T)" \
    "$(cut -d' ' -f1 /proc/loadavg)" "$(md5sum "$daemon" | cut -c1-8)" | tee -a "$OUT/run.log"
  env HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_EXEC="$exec" \
      HIPFIRE_CPU_EXEC_TRACE=1 HIPFIRE_DAEMON_BIN="$daemon" \
      "$cli" bench "$MODEL" "${COMMON[@]}" > "$OUT/$tag.json" 2> "$OUT/$tag.err"
  local rc=$?
  printf '=== %s done %s rc=%s load=%s\n' "$tag" "$(date +%T)" "$rc" \
    "$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/run.log"
  rg -o '"decode_tok_s": ?[0-9.]+' "$OUT/$tag.json" | head -3
}

run_arm old_pcie /tmp/cpu_exec_pin/hipfire   /tmp/cpu_exec_pin/daemon   pcie
run_arm old_cpu  /tmp/cpu_exec_pin/hipfire   /tmp/cpu_exec_pin/daemon   cpu
run_arm new_pcie ./target/release/hipfire    "$REPO_ROOT/target/release/daemon" pcie
run_arm new_cpu  ./target/release/hipfire    "$REPO_ROOT/target/release/daemon" cpu
echo "ALL DONE $(date +%T)" | tee -a "$OUT/run.log"
