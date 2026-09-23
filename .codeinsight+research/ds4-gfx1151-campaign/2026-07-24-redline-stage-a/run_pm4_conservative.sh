#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd
plan=/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723/candidates/p3-all-layers-gptq-head
out=/home/kaden/ds4-gfx1151-evidence/2026-07-24-redline-pm4-conservative
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
prompt=benchmarks/prompts/sweep/code.txt

cd "$repo"
install -d "$out"
sha256sum "$plan/overlay.hfq" > "$out/overlay.sha256"

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
    HIPFIRE_REPLAY_BACKEND=redline \
    HIPFIRE_REPLAY_TRANSPORT=pm4 \
    HIPFIRE_REPLAY_PM4_QUEUES=1 \
    HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
    HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=conservative \
    HIPFIRE_REPLAY_PM4_STATEFUL=legacy \
    "$binary" < "$prompt" > "$out/stdout.txt" 2> "$out/stderr.txt"

md5sum "$binary" "$prompt" "$out/stdout.txt" "$out/stderr.txt" > "$out/md5.txt"
rg "GPU dev|overlay ACTIVE|redline|\[stats\]" "$out/stderr.txt" > "$out/summary.txt"
cmp /home/kaden/ds4-gfx1151-evidence/2026-07-24-redline-stage-a/hip/stdout.txt \
    "$out/stdout.txt"
printf 'PM4 conservative parity PASS\n'
cat "$out/summary.txt"
