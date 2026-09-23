#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/prose_river_short.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-short-route-v2-guard

cd "$repo"
install -d "$out"
{
    printf 'classification=short-context regression guard; not a competitive claim\n'
    printf 'fixture=13 prompt tokens / 128 output / batch 1 / temperature 0 / prose / top-k 6\n'
    printf 'artifact_sha256=392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511\n'
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
} > "$out/identity.txt"

wait_gpu_clean() {
    local attempt
    local kfd_status
    for attempt in $(seq 1 90); do
        kfd_status=$(rocm-smi --showpids 2>&1)
        if rg -q "No KFD PIDs currently running" <<< "$kfd_status"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

for index in 1 2 3; do
    run=$(printf '%02d' "$index")
    dir="$out/$run"
    install -d "$dir"
    wait_gpu_clean
    date -Iseconds > "$dir/start.iso"
    rocm-smi --showproductname --showbus --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.before.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.before.txt"
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_GEN_TOKENS=128 \
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
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    date -Iseconds > "$dir/end.iso"
    md5sum "$binary" "$prompt" "$dir/stdout.txt" "$dir/stderr.txt" > "$dir/md5.txt"
    rg "GPU dev|MQ2R P3|Config:|Generation:|\\[prompt:|retained route ready|PM4 wait audit|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'completed %s: %s\n' "$run" "$(tail -n 1 "$dir/summary.txt")"
done

for output in "$out"/{02,03}/stdout.txt; do
    cmp "$out/01/stdout.txt" "$output"
done
for stderr in "$out"/*/stderr.txt; do
    rg -q "\\[prompt: 13 tokens \\(pos 0 → 13\\)\\]" "$stderr"
    rg -q "experts_per_tok=6" "$stderr"
    rg -q "retained route ready: capture=ReplayCaptureSummary \\{ launch_count: 2320, unique_kernel_count: 32, sequence_hash: 6907000390144547465 \\}" "$stderr"
    if rg -q "falling back|replay failed|illegal|FAULT|panic" "$stderr"; then
        printf 'failure/fallback marker in %s\n' "$stderr" >&2
        exit 5
    fi
done
printf 'short route-v2 guard output/route parity PASS\n'
rg "\\[stats\\]" "$out"/*/stderr.txt
