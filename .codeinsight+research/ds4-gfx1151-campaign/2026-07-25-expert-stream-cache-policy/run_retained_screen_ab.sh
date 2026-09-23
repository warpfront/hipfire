#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-expert-cpol-screen
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-expert-stream-cache-retained-screen/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"
{
    printf 'classification=2048/510 retained-PM4 two-process screen; not acceptance\n'
    printf 'A=shipping pointer MQ2 expert loads; B=raw-buffer temporal cpol0\n'
    printf 'both_arms=gfx1151 E8 temporal raw-buffer cpol0; tiled gather; top-k 6\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
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

run_arm() {
    local index="$1"
    local arm="$2"
    local -a cpol_env=()
    if [[ "$arm" == B ]]; then
        cpol_env=("HIPFIRE_DEEPSEEK4_MQ2_EXPERT_WEIGHT_CPOL=0")
    fi
    local dir="$out/run${index}-${arm}"
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
        HIPFIRE_DEEPSEEK4_TOPK_GATHER_TILED=1 \
        HIPFIRE_REPLAY_BACKEND=redline \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE=1 \
        "${cpol_env[@]}" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    rg "Radiowave code-object contracts|PM4 wait audit|retained route ready|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    rg -q "Radiowave code-object contracts: certified_artifacts=32/32" "$dir/stderr.txt"
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
printf 'retained screen output/resource/tape gates PASS\n'
rg "\\[stats\\]" "$out"/run*-*/stderr.txt
