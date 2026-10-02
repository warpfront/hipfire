#!/usr/bin/env bash
# Interleaved 2x2 A/B: {old,new} binaries x {pcie,cpu} exec, 3 rounds.
# old = /tmp/cpu_exec_pin (pre-change tree), new = target/release (HEAD + qt49 AVX2).
# Scratch during the measurement; committed here so the numbers have a recipe.
set -u
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Point MODEL at the local qwen3.8-27b.mq3-xt artifact.
MODEL="${MODEL:?set MODEL to the local qwen3.8-27b.mq3-xt path}"
PROMPT=benchmarks/prompts/gpu_offload_probe.txt
OUT=/tmp/cpu_exec_ab2
mkdir -p "$OUT"
COMMON=(--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots
        --workload stateless --prompt-file "$PROMPT" --json)

wait_for_no_daemon() {
  for _ in $(seq 40); do
    if ! pgrep -f 'release/daemon' >/dev/null && ! pgrep -f 'cpu_exec_pin/daemon' >/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  echo "WARN: daemon still alive" | tee -a "$OUT/run.log"
}

run_arm() { # tag cli daemon exec
  local tag=$1 cli=$2 daemon=$3 exec=$4
  pkill -f 'release/daemon' 2>/dev/null
  pkill -f 'cpu_exec_pin/daemon' 2>/dev/null
  wait_for_no_daemon
  printf '=== %s start %s load=%s\n' "$tag" "$(date +%T)" "$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/run.log"
  env HIPFIRE_GPU_LAYER_BUDGET=56 HIPFIRE_OFFLOAD_EXEC="$exec" \
      HIPFIRE_DAEMON_BIN="$daemon" \
      "$cli" bench "$MODEL" "${COMMON[@]}" > "$OUT/$tag.json" 2> "$OUT/$tag.err"
  local rc=$?
  printf '=== %s done %s rc=%s load=%s\n' "$tag" "$(date +%T)" "$rc" \
    "$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/run.log"
  python3 -c "
import json,sys
d=json.load(open('$OUT/$tag.json'))
print('    $tag decode median %.1f tok/s | prefill %.1f | vram_free %s MB | samples %s' % (
  d['decode_tok_s']['median'], d['prefill_tok_s']['median'], d['gpu']['vram_free_mb'],
  [round(s['decode_tok_s'],2) for s in d['samples']]))
" 2>> "$OUT/run.log" || echo "    $tag: no json" | tee -a "$OUT/run.log"
}

NEW="$REPO_ROOT/target/release/daemon"
for r in 1 2 3; do
  run_arm new_cpu_$r  ./target/release/hipfire $NEW cpu
  run_arm new_pcie_$r ./target/release/hipfire $NEW pcie
  run_arm old_cpu_$r  /tmp/cpu_exec_pin/hipfire /tmp/cpu_exec_pin/daemon cpu
  run_arm old_pcie_$r /tmp/cpu_exec_pin/hipfire /tmp/cpu_exec_pin/daemon pcie
done
echo "ALL DONE $(date +%T)" | tee -a "$OUT/run.log"
