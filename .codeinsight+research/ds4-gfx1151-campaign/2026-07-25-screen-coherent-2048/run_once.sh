#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-screen-coherent2-2048-top6-tg128

cd "$repo"
install -d "$out"
printf 'SCREENING ONLY: n=1, TG=128; never acceptance evidence\n' > "$out/classification.txt"
md5sum "$binary" "$prompt" > "$out/identity.txt"

env -i \
    HOME=/home/kaden \
    PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    ROCR_VISIBLE_DEVICES=1 \
    HIPFIRE_KERNEL_CACHE="$cache" \
    HIPFIRE_DEEPSEEK4_MODEL="$model" \
    HIPFIRE_DEEPSEEK4_GEN_TOKENS=128 \
    HIPFIRE_DEEPSEEK4_BENCH_PROMPT_TOKENS=2048 \
    HIPFIRE_DEEPSEEK4_BENCH_TILE_PROMPT=1 \
    HIPFIRE_DEEPSEEK4_BENCH_STDIN_ALL=1 \
    HIPFIRE_DEEPSEEK4_BENCH_EXPERTS_PER_TOK=6 \
    HIPFIRE_DEEPSEEK4_TEMP=0 \
    HIPFIRE_DEEPSEEK4_TOP_K=0 \
    HIPFIRE_DEEPSEEK4_SEED=424242 \
    HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
    HIPFIRE_DEEPSEEK4_SPEC_DECODE=0 \
    HIPFIRE_DEEPSEEK4_GRAPH=0 \
    HIPFIRE_REPLAY_BACKEND=redline \
    HIPFIRE_REPLAY_TRANSPORT=pm4 \
    HIPFIRE_REPLAY_PM4_QUEUES=1 \
    HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
    HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
    HIPFIRE_REPLAY_PM4_STATEFUL=static \
    "$binary" < "$prompt" > "$out/stdout.txt" 2> "$out/stderr.txt"

rg -q "\\[prompt: 2048 tokens \\(pos 0 → 2048\\)\\]" "$out/stderr.txt"
rg -q "experts_per_tok=6" "$out/stderr.txt"
rg "Config:|\\[prompt:|retained route ready|\\[stats\\]" "$out/stderr.txt"
printf '%s\n' '--- decoded output ---'
cat "$out/stdout.txt"
