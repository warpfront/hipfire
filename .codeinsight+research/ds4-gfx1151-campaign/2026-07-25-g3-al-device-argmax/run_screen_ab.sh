#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat-g3al-parallel
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache_a=/home/kaden/.hipfire_kernels/ds4-mq2r-g3z-A
cache_b=/home/kaden/.hipfire_kernels/ds4-mq2r-g3al-parallel
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-al-device-argmax/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"

cache_hash() {
    rg --files "$1" | sort | xargs sha256sum | sha256sum | cut -d' ' -f1
}

{
    printf 'classification=one-fresh-process-per-arm 2048/128 retained-PM4 sizing screen; not acceptance\n'
    printf 'fixture=2048 prompt / 128 generated / batch 1 / temp 0 / prose / top-k 6\n'
    printf 'A=full 129280-float logits D2H plus host argmax\n'
    printf 'B=exact two-stage GPU argmax retained in the PM4 tape plus four-byte token-id D2H\n'
    printf 'admission_floor=2 percent before six-process 2048/510 acceptance ABBA\n'
    printf 'artifact_sha256=392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511\n'
    printf 'session_id=%s\n' "$session_id"
    printf 'boot_id='
    cat /proc/sys/kernel/random/boot_id
    printf 'cache_A_sha256=%s\n' "$(cache_hash "$cache_a")"
    printf 'cache_B_sha256=%s\n' "$(cache_hash "$cache_b")"
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
    local arm="$1"
    local cache="$cache_a"
    local -a candidate_env=()
    local certified_artifacts=32
    local certified_launches=2320
    local covered_boundaries=2319
    local unique_symbols=32
    local route_hash=4393256763546932634
    if [[ "$arm" == B ]]; then
        cache="$cache_b"
        candidate_env=("HIPFIRE_DS4_AR_DEVICE_ARGMAX=1")
        certified_artifacts=33
        certified_launches=2322
        covered_boundaries=2321
        unique_symbols=34
        route_hash=15558081130791565125
    fi
    local dir="$out/$arm"
    install -d "$dir"
    wait_gpu_clean
    date -Iseconds > "$dir/start.iso"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.before.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.before.txt"

    env -i \
        HOME=/home/kaden \
        PATH=/home/kaden/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
        ROCR_VISIBLE_DEVICES=1 \
        HIPFIRE_KERNEL_CACHE="$cache" \
        HIPFIRE_DEEPSEEK4_MODEL="$model" \
        HIPFIRE_DEEPSEEK4_GEN_TOKENS=128 \
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
        "${candidate_env[@]}" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"

    date -Iseconds > "$dir/end.iso"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'kernel_cache_hash=%s\n' "$(cache_hash "$cache")" > "$dir/cache.after.txt"
    rg "Radiowave code-object contracts|Radiowave argument effects|PM4 wait audit|retained route ready|\[stats\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"

    rg -q "Radiowave code-object contracts: certified_artifacts=${certified_artifacts}/${certified_artifacts}" \
        "$dir/stderr.txt"
    rg -q "Radiowave argument effects: certified_launches=${certified_launches} fallback_launches=0 unknown_launches=0" \
        "$dir/stderr.txt"
    rg -q "boundaries=${covered_boundaries} covered=${covered_boundaries}" "$dir/stderr.txt"
    rg -q "launch_count: ${certified_launches}, unique_kernel_count: ${unique_symbols}, sequence_hash: ${route_hash}" \
        "$dir/stderr.txt"
    if rg -q "falling back|replay failed|illegal|FAULT|panic|JIT compile failed" \
        "$dir/stderr.txt"; then
        printf 'failure/fallback marker in %s\n' "$dir/stderr.txt" >&2
        exit 5
    fi
    printf 'completed %s: %s\n' "$arm" "$(tail -n 1 "$dir/summary.txt")"
}

run_arm A
run_arm B
cmp "$out/A/stdout.txt" "$out/B/stdout.txt"
printf 'decoded_output_byte_identical=1\n' | tee "$out/output-gate.txt"
rg "\[stats\]" "$out"/{A,B}/stderr.txt
printf '%s\n' "$out"
