#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd
plan=/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723/candidates/p3-all-layers-gptq-head
out=/home/kaden/ds4-gfx1151-evidence/2026-07-24-redline-stage-a
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
prompt=benchmarks/prompts/sweep/code.txt

cd "$repo"
install -d "$out"
sha256sum "$plan/overlay.hfq" > "$out/overlay.sha256"

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

run_arm() {
    local arm=$1
    local backend=$2
    local dir="$out/$arm"
    install -d "$dir"
    wait_gpu_clean
    date -Iseconds > "$dir/start.iso"
    rocm-smi --showproductname --showbus --showmeminfo vram --showuse --showpids \
        > "$dir/topology.before.txt"
    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_REAP_PLAN="$plan" \
        HIPFIRE_REQUIRE_REAP_OVERLAY=1 \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_GEN_TOKENS=16 \
        HIPFIRE_DEEPSEEK4_TEMP=0 \
        HIPFIRE_DEEPSEEK4_TOP_K=0 \
        HIPFIRE_DEEPSEEK4_SEED=424242 \
        HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
        HIPFIRE_DEEPSEEK4_SPEC_DECODE=0 \
        HIPFIRE_DEEPSEEK4_GRAPH=0 \
        HIPFIRE_DEEPSEEK4_E8_WO_GROUPED=1 \
        HIPFIRE_DEEPSEEK4_E8_U4=1 \
        HIPFIRE_DEEPSEEK4_E8_PREFILL_B2=1 \
        HIPFIRE_DEEPSEEK4_E8_PREFILL_B4=1 \
        HIPFIRE_DEEPSEEK4_FFN_OVERLAP=0 \
        HIPFIRE_DEEPSEEK4_MQ2_PERM=1 \
        HIPFIRE_DEEPSEEK4_HC_PINGPONG=1 \
        HIPFIRE_REPLAY_BACKEND="$backend" \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    date -Iseconds > "$dir/end.iso"
    md5sum "$binary" "$prompt" "$dir/stdout.txt" "$dir/stderr.txt" > "$dir/md5.txt"
    rg "GPU dev|overlay ACTIVE|redline|\\[stats\\]" "$dir/stderr.txt" \
        > "$dir/summary.txt"
}

run_arm hip hip
run_arm pm4 redline

cmp "$out/hip/stdout.txt" "$out/pm4/stdout.txt"
rg -q "retained route ready" "$out/pm4/stderr.txt"
if rg -q "falling back|replay failed|illegal|FAULT|panic" "$out/pm4/stderr.txt"; then
    printf 'retained route emitted a failure/fallback marker\n' >&2
    exit 5
fi

printf 'stage-a parity PASS\n'
rg "\\[stats\\]|retained route ready|PM4 wait audit" \
    "$out/hip/stderr.txt" "$out/pm4/stderr.txt"
