#!/usr/bin/env bash
set -euo pipefail
root=/home/kaden/hipfire-gfx11slice
out="$root/scratch-gfx11slice"
export HIP_VISIBLE_DEVICES=0 HIPFIRE_HOME="$out/home-wt2" HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0
mkdir -p "$HIPFIRE_HOME"
cli="$root/target/release/hipfire"
"$cli" config set kernel.gfx11_iu4_gridspec true > "$out/wt2-gridspec.config"
"$cli" config set kernel.gfx11_iu4_shape true > "$out/wt2-shape.config"
"$cli" config set kernel.gfx11_iu4_symfold true > "$out/wt2-symfold.config"
"$cli" config set kernel.gfx11_a4_candidates 2 > "$out/wt2-a4.config"
rocm-smi --showpids > "$out/wt2-pids-pre.txt"
for arm in A B; do
  if [[ "$arm" == B ]]; then lean=true; else lean=false; fi
  "$cli" config set kernel.gfx11_lean_pbs "$lean" > "$out/wt2-$arm.lean.config"
  "$root/target/release/examples/eval_hipfire" \
    --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq \
    --ref /home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin \
    --kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks 24 \
    --output "$out/wt2-$arm.kldseq" > "$out/wt2-$arm.stdout" 2> "$out/wt2-$arm.stderr"
done
rocm-smi --showpids > "$out/wt2-pids-post.txt"
