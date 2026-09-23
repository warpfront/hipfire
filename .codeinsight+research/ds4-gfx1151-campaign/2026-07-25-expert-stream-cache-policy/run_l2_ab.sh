#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-expert-cpol-screen
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-expert-stream-cache-l2/${session_id}"
kernel_regex='gemv_(mq2g256_lloyd_moe_(gate_up|down)|mfp4g32_e8_soa_u4_buffer)'

cd "$repo"
test ! -e "$out"
install -d "$out"
{
    printf 'classification=2048/8 L2 mechanism diagnostic; not acceptance\n'
    printf 'A=shipping pointer MQ2 expert loads; B=raw-buffer temporal cpol0\n'
    printf 'counter=L2CacheHit; filtered to MQ2 expert and dense E8 U4 kernels\n'
    printf 'kernel_regex=%s\n' "$kernel_regex"
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
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
    local arm="$1"
    local -a cpol_env=()
    if [[ "$arm" == B ]]; then
        cpol_env=("HIPFIRE_DEEPSEEK4_MQ2_EXPERT_WEIGHT_CPOL=0")
    fi
    local dir="$out/$arm"
    install -d "$dir"
    wait_gpu_clean
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_GEN_TOKENS=8 \
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
        HIPFIRE_REPLAY_BACKEND=off \
        "${cpol_env[@]}" \
        rocprofv3 \
            --pmc L2CacheHit \
            --kernel-include-regex "$kernel_regex" \
            --output-format csv \
            --output-directory "$dir" \
            --output-file counters \
            -- "$binary" \
            < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    rg -q "\\[prompt: 2048 tokens \\(pos 0 → 2048\\)\\]" "$dir/stderr.txt"
    if rg -q "falling back|illegal|FAULT|panic|JIT compile failed" "$dir/stderr.txt"; then
        printf 'failure marker in %s\n' "$dir/stderr.txt" >&2
        exit 5
    fi
    printf 'completed l2-%s: %s\n' "$arm" \
        "$(rg '\\[stats\\]' "$dir/stderr.txt" | tail -n 1)"
}

run_arm A
run_arm B
cmp "$out/A/stdout.txt" "$out/B/stdout.txt"
printf 'L2 output gate PASS\n'
printf '%s\n' "$out"
