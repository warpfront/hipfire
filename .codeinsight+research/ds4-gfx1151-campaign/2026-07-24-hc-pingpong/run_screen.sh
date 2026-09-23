#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd
plan=/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723/candidates/p3-all-layers-gptq-head
out=/home/kaden/ds4-gfx1151-evidence/2026-07-24-hc-pingpong
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
prompt=benchmarks/prompts/sweep/code.txt
overlay_sha=bc3644259952efca32074085ed69bcbeb0fad12fb23972fe5694ac9bf13323b8
order=(A B B A)

cd "$repo"
install -d "$out"
printf '%s  %s\n' "$overlay_sha" "$plan/overlay.hfq" > "$out/overlay.sha256"

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

run_one() {
    local run=$1
    local arm=$2
    local max_gen=$3
    local pingpong=0
    local dir="$out/$run"
    if [[ "$arm" == B ]]; then
        pingpong=1
    fi

    install -d "$dir"
    if [[ -s "$dir/summary.txt" ]]; then
        printf 'skipping completed %s: %s\n' "$run" "$(tail -n 1 "$dir/summary.txt")"
        return
    fi
    if ! wait_gpu_clean; then
        printf 'refusing %s: another KFD process is active\n' "$run" >&2
        rocm-smi --showpids >&2
        exit 3
    fi

    rocm-smi --showproductname --showbus --showmeminfo vram --showuse --showpids \
        > "$dir/topology.before.txt"
    date -Iseconds > "$dir/start.iso"
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        HIP_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_REAP_PLAN="$plan" \
        HIPFIRE_REQUIRE_REAP_OVERLAY=1 \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_GEN_TOKENS="$max_gen" \
        HIPFIRE_DEEPSEEK4_TEMP=0 \
        HIPFIRE_DEEPSEEK4_TOP_K=0 \
        HIPFIRE_DEEPSEEK4_SEED=424242 \
        HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
        HIPFIRE_DEEPSEEK4_SPEC_DECODE=0 \
        HIPFIRE_DEEPSEEK4_GRAPH=1 \
        HIPFIRE_DEEPSEEK4_E8_WO_GROUPED=1 \
        HIPFIRE_DEEPSEEK4_E8_U4=1 \
        HIPFIRE_DEEPSEEK4_E8_PREFILL_B2=1 \
        HIPFIRE_DEEPSEEK4_E8_PREFILL_B4=1 \
        HIPFIRE_DEEPSEEK4_FFN_OVERLAP=1 \
        HIPFIRE_DEEPSEEK4_MQ2_PERM=1 \
        HIPFIRE_DEEPSEEK4_HC_PINGPONG="$pingpong" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    date -Iseconds > "$dir/end.iso"

    rg -q "reap: overlay ACTIVE — 554 tensor" "$dir/stderr.txt"
    rg -q "\\[DeepSeek V4 hipGraph\\] captured forward" "$dir/stderr.txt"
    rg -q "\\[stats\\]" "$dir/stderr.txt"
    md5sum "$binary" "$prompt" "$dir/stdout.txt" "$dir/stderr.txt" > "$dir/md5.txt"
    rg "\\[stats\\]|GPU dev|Config:|Generation:|\\[prompt:|overlay ACTIVE|captured forward" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    rocm-smi --showuse --showpids > "$dir/topology.after.txt"
    printf 'completed %s pingpong=%s: %s\n' \
        "$run" "$pingpong" "$(rg '\\[stats\\]' "$dir/summary.txt" | tail -n 1)"
}

run_one prewarm-A A 8
run_one prewarm-B B 8
if ! cmp -s "$out/prewarm-A/stdout.txt" "$out/prewarm-B/stdout.txt"; then
    printf 'refusing timed samples: ping-pong output differs from baseline\n' >&2
    exit 4
fi

for index in "${!order[@]}"; do
    arm=${order[$index]}
    run=$(printf 'code-%02d-%s' "$((index + 1))" "$arm")
    run_one "$run" "$arm" 64
done
