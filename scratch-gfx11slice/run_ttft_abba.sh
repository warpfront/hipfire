#!/usr/bin/env bash
set -euo pipefail
root=/home/kaden/hipfire-gfx11slice
out="$root/scratch-gfx11slice"
cli="$root/target/release/hipfire"
export HIP_VISIBLE_DEVICES=0 HIPFIRE_DAEMON_BIN="$root/target/release/daemon" HIPFIRE_HOME="$out/home-vmm-ttft"
mkdir -p "$HIPFIRE_HOME"
"$cli" config set kernel.gfx11_iu4_gridspec true > "$out/ttft-gridspec.config"
"$cli" config set kernel.gfx11_iu4_shape true > "$out/ttft-shape.config"
"$cli" config set kernel.gfx11_iu4_symfold true > "$out/ttft-symfold.config"
"$cli" config set kernel.gfx11_a4_candidates 2 > "$out/ttft-a4.config"
rocm-smi --showpids > "$out/vmm-ttft-pids-pre.txt"
for label in A1 B1 B2 A2; do
  if [[ "$label" == B* ]]; then flag=true; else flag=false; fi
  "$cli" config set kernel.gfx11_lean_pbs "$flag" > "$out/vmm-ttft-$label.lean.config"
  "$cli" bench /home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq \
    --ttft --prompt-file "$root/benchmarks/prompts/ttft_511.txt" \
    --runs 4 --warmups 2 --spec off --kv-mode q8 --kv-backend vmm --json \
    > "$out/vmm-ttft-$label.json" 2> "$out/vmm-ttft-$label.stderr"
done
rocm-smi --showpids > "$out/vmm-ttft-pids-post.txt"
