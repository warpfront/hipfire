#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-g3ac-f16-dot2
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-ac-mq2-f16-dot2/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"
{
    printf 'classification=2048/8 ordinary-HIP kernel-trace diagnostic; not acceptance\n'
    printf 'A=shipping MQ2 F32 activation route\n'
    printf 'B=topk-fused gate F16 pack plus F16 FWHT down producer and native dot2 consumers\n'
    printf 'both_arms=gfx1151 E8 temporal raw-buffer cpol0; tiled gather; top-k 6\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
} > "$out/identity.txt"

run_arm() {
    local arm="$1"
    local -a candidate_env=()
    if [[ "$arm" == B ]]; then
        candidate_env=("HIPFIRE_DEEPSEEK4_MQ2_F16_DOT2=1")
    fi
    local dir="$out/$arm"
    install -d "$dir"
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
        "${candidate_env[@]}" \
        rocprofv3 \
            --kernel-trace \
            --stats \
            --output-format csv \
            --output-directory "$dir" \
            --output-file trace \
            -- "$binary" \
            < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    python3 - "$dir/stderr.txt" <<'PY'
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
required = ["[prompt: 2048 tokens (pos 0 → 2048)]", "[stats]"]
missing = [item for item in required if item not in text]
bad = [item for item in ["falling back", "illegal", "FAULT", "panic", "JIT compile failed"] if item in text]
if missing or bad:
    raise SystemExit(f"trace gate failed: missing={missing} bad={bad}")
PY
    printf 'completed trace-%s\n' "$arm"
}

run_arm A
run_arm B
if cmp -s "$out/A/stdout.txt" "$out/B/stdout.txt"; then
    printf 'decoded_output_byte_identical=1\n' | tee "$out/output-gate.txt"
else
    printf 'decoded_output_byte_identical=0\n' | tee "$out/output-gate.txt"
fi
printf '%s\n' "$out"
