#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-radiowave-c2-acceptance-abba

cd "$repo"
install -d "$out"
{
    printf 'classification=authoritative 2048/510 acceptance\n'
    printf 'fixture=2048 prompt / 510 generated / batch 1 / temperature 0 / prose / top-k 6\n'
    printf 'order=ABBAAB; A=full same-agent dependency acquire; B=hash-bound Radiowave cache class\n'
    printf 'both_arms=gfx1151 E8 temporal raw-buffer cpol0\n'
    printf 'artifact_sha256=392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511\n'
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

run_arm() {
    local index=$1
    local arm=$2
    local vmem=0
    local dir="$out/$(printf '%02d' "$index")-$arm"
    if [[ "$arm" == B ]]; then
        vmem=1
    fi
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
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE="$vmem" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    date -Iseconds > "$dir/end.iso"
    md5sum "$binary" "$prompt" "$dir/stdout.txt" "$dir/stderr.txt" > "$dir/md5.txt"
    rg "GPU dev|MQ2R P3|Config:|Generation:|\\[prompt:|Radiowave code-object contracts|retained route ready|PM4 wait audit|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'completed %02d-%s: %s\n' "$index" "$arm" "$(tail -n 1 "$dir/summary.txt")"
}

index=0
for arm in A B B A A B; do
    index=$((index + 1))
    run_arm "$index" "$arm"
done

reference="$out/01-A/stdout.txt"
for output in "$out"/*/stdout.txt; do
    cmp "$reference" "$output"
done
for dir in "$out"/*-*; do
    stderr="$dir/stderr.txt"
    rg -q "\\[prompt: 2048 tokens \\(pos 0 → 2048\\)\\]" "$stderr"
    rg -q "experts_per_tok=6" "$stderr"
    rg -q "boundaries=2319 covered=2319" "$stderr"
    rg -q "launch_count: 2320, unique_kernel_count: 32, sequence_hash: 13913946932716306919" \
        "$stderr"
    if [[ "$dir" == *-A ]]; then
        if rg -q "Radiowave code-object contracts" "$stderr"; then
            printf 'unexpected Radiowave cache classification in %s\n' "$stderr" >&2
            exit 4
        fi
    else
        rg -q "Radiowave code-object contracts: certified_artifacts=32/32 vmem_symbols=12 vmem_launches=1283" \
            "$stderr"
    fi
    if rg -q "falling back|replay failed|illegal|FAULT|panic" "$stderr"; then
        printf 'failure/fallback marker in %s\n' "$stderr" >&2
        exit 5
    fi
done

printf 'C2 acceptance output/resource/route parity PASS\n'
rg "\\[stats\\]" "$out"/*/stderr.txt
