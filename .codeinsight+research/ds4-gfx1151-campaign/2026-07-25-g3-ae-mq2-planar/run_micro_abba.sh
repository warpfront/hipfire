#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/bench_mq2_planar_route
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-g3ae-planar
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-ae-mq2-planar/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"
{
    printf 'classification=three-fresh-process exact-weight 43-layer MQ2 layout screen; not acceptance\n'
    printf 'A=shipping AoS MQ2-Lloyd gate K4 plus down K8-all\n'
    printf 'B=lossless same-byte per-row codebook plane plus aligned index plane\n'
    printf 'top_k=6 routed_weight_bytes_per_pass=1826095104 passes_per_block=10\n'
    printf 'admission_floor_ms_from_37.55=0.751\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    sha256sum "$binary" "$model"
} > "$out/identity.txt"

for process in 1 2 3; do
    process_dir="$out/process-${process}"
    install -d "$process_dir"
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_LOAD_MTP=0 \
        HIPFIRE_REPLAY_BACKEND=off \
        HIPFIRE_DEEPSEEK4_MQ2_DOWN_K8ALL=1 \
        "$binary" \
        > "$process_dir/stdout.txt" \
        2> "$process_dir/stderr.txt"
    rg -q 'correctness gate_byte_exact=true up_byte_exact=true down_byte_exact=true' \
        "$process_dir/stdout.txt"
    test "$(rg -c '^block=' "$process_dir/stdout.txt")" -eq 4
    printf 'completed process-%s\n' "$process"
done

printf '%s\n' "$out"
