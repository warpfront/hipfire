#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/bench_mq2_compute_ceiling
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-g3ad-compute-ceiling
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-ad-mq2-compute-ceiling/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"
{
    printf 'classification=three-fresh-process exact-43-layer MQ2 gate+down microkernel ceiling; not acceptance\n'
    printf 'order_per_process=ABBA; 10 full 43-layer passes per block\n'
    printf 'A=shipping gate K4 plus down K8-all\n'
    printf 'B=same expert/codebook/index/activation traffic with lookup and coordinate MACs stripped\n'
    printf 'output=deliberately invalid in B\n'
    printf 'top_k=6\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    sha256sum "$binary" "$model"
    stat --printf='model_inode=%i model_size=%s model_mtime=%Y\n' "$model"
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

for run in 1 2 3; do
    dir="$out/run${run}"
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
        HIPFIRE_DEEPSEEK4_MQ2_DOWN_K8ALL=1 \
        HIPFIRE_BENCH_PASSES=10 \
        "$binary" > "$dir/stdout.jsonl" 2> "$dir/stderr.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    test "$(wc -l < "$dir/stdout.jsonl")" -eq 4
    printf 'run%s ' "$run"
    tr '\n' ' ' < "$dir/stdout.jsonl"
    printf '\n'
done

printf '%s\n' "$out"
