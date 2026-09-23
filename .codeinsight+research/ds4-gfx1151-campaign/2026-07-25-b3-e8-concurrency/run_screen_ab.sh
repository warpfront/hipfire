#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-b3-e8-concurrency-screen-ab

cd "$repo"
install -d "$out"
{
    printf 'classification=two-process 2048/510 screen; not acceptance\n'
    printf 'A=single retained PM4 queue; B=two retained PM4 queues with native gfx11 phase synchronization\n'
    printf 'both_arms=gfx1151 E8 temporal raw-buffer cpol0; Radiowave cache contracts; top-k 6\n'
    printf 'target=resource-independent E8 U4 sibling concurrency, including 148 adjacent pairs\n'
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
} > "$out/identity.txt"

wait_gpu_clean() {
    local attempt
    for attempt in $(seq 1 90); do
        if rocm-smi --showpids 2>&1 | rg -q "No KFD PIDs currently running"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

for arm in A B; do
    dir="$out/$arm"
    queues=1
    native=0
    if [[ "$arm" == B ]]; then
        queues=2
        native=1
    fi
    install -d "$dir"
    wait_gpu_clean
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.before.txt"
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
        HIPFIRE_GFX1151_E8_BUFFER=1 \
        HIPFIRE_REPLAY_BACKEND=redline \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES="$queues" \
        HIPFIRE_REPLAY_PM4_NATIVE_PHASES="$native" \
        HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH=2 \
        HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS=0 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE=1 \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    rg "Radiowave code-object contracts|PM4 wait audit|PM4 phase plan|retained route ready|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    printf 'completed %s: %s\n' "$arm" "$(tail -n 1 "$dir/summary.txt")"
done

cmp "$out/A/stdout.txt" "$out/B/stdout.txt"
for stderr in "$out"/{A,B}/stderr.txt; do
    rg -q "Radiowave code-object contracts: certified_artifacts=32/32 vmem_symbols=12 vmem_launches=1283" \
        "$stderr"
    rg -q "boundaries=2319 covered=2319" "$stderr"
    rg -q "launch_count: 2320, unique_kernel_count: 32, sequence_hash: 13913946932716306919" \
        "$stderr"
    if rg -q "falling back|replay failed|illegal|FAULT|panic" "$stderr"; then
        printf 'failure/fallback marker in %s\n' "$stderr" >&2
        exit 5
    fi
done
rg -q "PM4 phase plan architecture=gfx1151 queues=2" "$out/B/stderr.txt"
printf 'B3 screen output/resource/tape parity PASS\n'
rg "\\[stats\\]" "$out"/*/stderr.txt
