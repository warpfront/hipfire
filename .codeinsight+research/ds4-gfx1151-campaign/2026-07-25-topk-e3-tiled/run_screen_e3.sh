#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
reference=/home/kaden/ds4-gfx1151-evidence/2026-07-25-topk-e2-bounded-acceptance-abba/01-A/stdout.txt
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-topk-e3-tiled-screen

cd "$repo"
install -d "$out"
md5sum "$binary" "$prompt" > "$out/identity.txt"
sha256sum "$binary" >> "$out/identity.txt"

for attempt in $(seq 1 90); do
    if rocm-smi --showpids 2>&1 | rg -q "No KFD PIDs currently running"; then
        break
    fi
    if [[ "$attempt" == 90 ]]; then
        printf 'GPU remained busy\n' >&2
        exit 3
    fi
    sleep 2
done

rocm-smi --showclocks --showtemp --showuse --showpids > "$out/topology.before.txt"
env -i \
    HOME=/home/kaden \
    PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
    ROCR_VISIBLE_DEVICES=1 \
    HIPFIRE_KERNEL_CACHE="$cache" \
    HIPFIRE_DEEPSEEK4_MODEL="$model" \
    HIPFIRE_DEEPSEEK4_GEN_TOKENS=510 \
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
    HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BOUNDED=1 \
    HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BLOCK1024=1 \
    HIPFIRE_REPLAY_BACKEND=redline \
    HIPFIRE_REPLAY_TRANSPORT=pm4 \
    HIPFIRE_REPLAY_PM4_QUEUES=1 \
    HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
    HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
    HIPFIRE_REPLAY_PM4_STATEFUL=static \
    "$binary" < "$prompt" > "$out/stdout.txt" 2> "$out/stderr.txt"
rocm-smi --showclocks --showtemp --showuse --showpids > "$out/topology.after.txt"

cmp "$reference" "$out/stdout.txt"
rg -q "\\[prompt: 2048 tokens \\(pos 0 → 2048\\)\\]" "$out/stderr.txt"
rg -q "experts_per_tok=6" "$out/stderr.txt"
rg -q "retained route ready: capture=ReplayCaptureSummary \\{ launch_count: 2320, unique_kernel_count: 32, sequence_hash: 6907000390144547465 \\}" "$out/stderr.txt"
if rg -q "falling back|replay failed|illegal|FAULT|panic" "$out/stderr.txt"; then
    printf 'failure/fallback marker in screen stderr\n' >&2
    exit 5
fi
rg "GPU dev|MQ2R P3|Config:|Generation:|\\[prompt:|retained route ready|PM4 wait audit|\\[stats\\]" \
    "$out/stderr.txt" > "$out/summary.txt"
printf 'E3 screen output/route parity PASS: %s\n' "$(tail -n 1 "$out/summary.txt")"
