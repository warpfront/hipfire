#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-g3z-A
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-af-pm4-resource-limits/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"

cache_hash() {
    rg --files "$cache" | sort | xargs sha256sum | sha256sum | cut -d' ' -f1
}

{
    printf 'classification=two-process 2048/510 retained-PM4 sizing screen; not acceptance\n'
    printf 'fixture=2048 prompt / 510 generated / batch 1 / temp 0 / prose / top-k 6\n'
    printf 'A=shipping legacy COMPUTE_RESOURCE_LIMITS=0\n'
    printf 'B=RADV SIMD_DEST_CNTL on workgroups with wave count divisible by four\n'
    printf 'admission_floor=2 percent before six-process acceptance ABBA\n'
    printf 'artifact_sha256=392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    printf 'kernel_cache_hash=%s\n' "$(cache_hash)"
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
    stat --printf='model_inode=%i model_size=%s model_mtime=%Y\n' "$model"
    uname -a
} > "$out/identity.txt"
{
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    printf 'kernel_cache_hash=%s\n' "$(cache_hash)"
    sha256sum "$binary"
    stat --printf='model_inode=%i model_size=%s model_mtime=%Y\n' "$model"
} > "$out/session.fingerprint.txt"

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
    local index="$1"
    local arm="$2"
    local policy=legacy
    if [[ "$arm" == B ]]; then
        policy=radv
    fi

    local dir="$out/run${index}-${arm}"
    install -d "$dir"
    wait_gpu_clean
    date -Iseconds > "$dir/start.iso"
    {
        printf 'session_id=%s\n' "$session_id"
        printf 'boot_id='
        cat /proc/sys/kernel/random/boot_id
        printf 'kernel_cache_hash=%s\n' "$(cache_hash)"
        sha256sum "$binary"
        stat --printf='model_inode=%i model_size=%s model_mtime=%Y\n' "$model"
    } > "$dir/fingerprint.txt"
    cmp "$out/session.fingerprint.txt" "$dir/fingerprint.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
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
        HIPFIRE_DEEPSEEK4_TOPK_GATHER_TILED=1 \
        HIPFIRE_REPLAY_BACKEND=redline \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE=1 \
        HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS="$policy" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"

    date -Iseconds > "$dir/end.iso"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'kernel_cache_hash=%s\n' "$(cache_hash)" > "$dir/cache.after.txt"
    rg "gfx1151 PM4 resource-limits|Radiowave code-object contracts|Radiowave argument effects|PM4 wait audit|retained route ready|\[stats\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"

    rg -q "Radiowave code-object contracts: certified_artifacts=32/32" \
        "$dir/stderr.txt"
    rg -q "Radiowave argument effects: certified_launches=2320 fallback_launches=0 unknown_launches=0" \
        "$dir/stderr.txt"
    rg -q "boundaries=2319 covered=2319" "$dir/stderr.txt"
    rg -q "launch_count: 2320, unique_kernel_count: 32, sequence_hash: 4393256763546932634" \
        "$dir/stderr.txt"
    if rg -q "falling back|replay failed|illegal|FAULT|panic|JIT compile failed" \
        "$dir/stderr.txt"; then
        printf 'failure/fallback marker in %s\n' "$dir/stderr.txt" >&2
        exit 5
    fi
    printf 'completed run%s-%s: %s\n' \
        "$index" "$arm" "$(tail -n 1 "$dir/summary.txt")"
}

run_arm 1 A
run_arm 2 B

cmp "$out/run1-A/stdout.txt" "$out/run2-B/stdout.txt"
printf 'PM4 resource-limit screen output/resource/tape gates PASS\n'
rg "gfx1151 PM4 resource-limits|retained route ready|\[stats\]" "$out"/run*-*/stderr.txt
printf '%s\n' "$out"
