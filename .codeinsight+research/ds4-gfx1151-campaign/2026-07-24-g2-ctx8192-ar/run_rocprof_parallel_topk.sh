#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
profile_gen=${HIPFIRE_DEEPSEEK4_PROFILE_GEN_TOKENS:-2}
out=/home/kaden/ds4-gfx1151-evidence/2026-07-24-g2-ctx8192-ar/rocprof-parallel-topk-gen${profile_gen}
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-topk-parallel
prompt=benchmarks/prompts/deepseek4_mq2r_prose_ctx8192.txt

cd "$repo"
install -d "$out"
date -Iseconds > "$out/start.iso"
env -i \
    HOME=/home/kaden \
    PATH=/opt/rocm/bin:/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    ROCR_VISIBLE_DEVICES=1 \
    HIPFIRE_KERNEL_CACHE="$cache" \
    HIPFIRE_DEEPSEEK4_MODEL="$model" \
    HIPFIRE_DEEPSEEK4_GEN_TOKENS="$profile_gen" \
    HIPFIRE_DEEPSEEK4_TEMP=0 \
    HIPFIRE_DEEPSEEK4_TOP_K=0 \
    HIPFIRE_DEEPSEEK4_SEED=424242 \
    HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
    HIPFIRE_DEEPSEEK4_SPEC_DECODE=0 \
    HIPFIRE_DEEPSEEK4_GRAPH=0 \
    HIPFIRE_REPLAY_BACKEND=hip \
    scripts/rocprof-wrap.sh "$out" -- "$binary" \
    < "$prompt" > "$out/stdout.txt" 2> "$out/stderr.txt"
date -Iseconds > "$out/end.iso"
md5sum "$binary" "$prompt" "$out/stdout.txt" "$out/stderr.txt" \
    "$out/trace_kernel_trace.csv" "$out/trace_kernel_stats.csv" > "$out/md5.txt"
rg "GPU dev|MQ2R P3|Config:|Generation:|\\[prompt:|\\[stats\\]" \
    "$out/stderr.txt" > "$out/summary.txt"
rg -q "\\[prompt: 8192 tokens \\(pos 0 → 8192\\)\\]" "$out/stderr.txt"
if rg -q "falling back|replay failed|illegal|FAULT|panic|Error:" "$out/stderr.txt"; then
    printf 'rocprof candidate emitted a failure marker\n' >&2
    exit 5
fi
printf 'G2 ctx8192 rocprof capture PASS: %s\n' "$(tail -n 1 "$out/summary.txt")"
