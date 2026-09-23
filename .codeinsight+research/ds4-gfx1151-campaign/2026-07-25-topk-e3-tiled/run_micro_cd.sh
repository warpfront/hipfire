#!/usr/bin/env bash
set -euo pipefail

binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/test_indexer_top_k_buf
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
out=/home/kaden/ds4-gfx1151-evidence/2026-07-25-topk-e4-unrolled-micro
order=(C D D C C D)
iters=1000

install -d "$out"
md5sum "$binary" > "$out/identity.txt"
{
    printf 'C=E3 tiled k-bounded bitonic block1024\n'
    printf 'D=E4 KPAD/wave32 templated and unrolled block1024\n'
    printf 'iters=%s\n' "$iters"
} >> "$out/identity.txt"

wait_gpu_clean() {
    local attempt
    local kfd_status
    for attempt in $(seq 1 60); do
        kfd_status=$(rocm-smi --showpids 2>&1)
        if rg -q "No KFD PIDs currently running" <<< "$kfd_status"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

for index in "${!order[@]}"; do
    arm="${order[$index]}"
    run=$(printf '%02d-%s' "$((index + 1))" "$arm")
    dir="$out/$run"
    install -d "$dir"
    wait_gpu_clean
    unrolled=0
    if [[ "$arm" == D ]]; then
        unrolled=1
    fi
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BOUNDED=1 \
        HIPFIRE_DEEPSEEK4_INDEXER_TOPK_UNROLLED="$unrolled" \
        HIPFIRE_DEEPSEEK4_INDEXER_TOPK_BLOCK1024=1 \
        "$binary" "$iters" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    rg -q "PASS n=2048 k=1 exact=true" "$dir/stdout.txt"
    printf 'completed %s\n' "$run"
done

rg "PASS" "$out"/*/stdout.txt
