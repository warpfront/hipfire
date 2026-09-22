#!/usr/bin/env bash
set -euo pipefail
root=/home/kaden/hipfire-gfx11slice
out="$root/scratch-gfx11slice"
export HIP_VISIBLE_DEVICES=0 HIPFIRE_DAEMON_BIN="$root/target/release/daemon" HIPFIRE_HOME="$out/home-vmm-matrix"
cli="$root/target/release/hipfire"
model=/home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq
mkdir -p "$HIPFIRE_HOME"
"$cli" config set kernel.gfx11_iu4_gridspec true > "$out/matrix-gridspec.config"
"$cli" config set kernel.gfx11_iu4_shape true > "$out/matrix-shape.config"
"$cli" config set kernel.gfx11_iu4_symfold true > "$out/matrix-symfold.config"
"$cli" config set kernel.gfx11_a4_candidates 2 > "$out/matrix-a4.config"
rocm-smi --showpids > "$out/vmm-matrix-pids-pre.txt"
for label in A1 B1 B2 A2; do
  if [[ "$label" == B* ]]; then flag=true; else flag=false; fi
  "$cli" config set kernel.gfx11_lean_pbs "$flag" > "$out/vmm-matrix-$label.lean.config"
  "$cli" bench "$model" --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode q8 --kv-backend vmm --json \
    > "$out/vmm-matrix-$label.json" 2> "$out/vmm-matrix-$label.stderr"
done
rocm-smi --showpids > "$out/vmm-matrix-pids-post.txt"
