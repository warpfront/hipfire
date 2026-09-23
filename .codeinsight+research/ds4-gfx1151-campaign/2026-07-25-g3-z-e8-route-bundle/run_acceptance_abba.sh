#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
model=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
prompt=benchmarks/prompts/adaptive_kv_long_prefill.txt
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-g3z-B
session_id="$(date -u +%Y%m%dT%H%M%SZ)-$(cut -c1-8 /proc/sys/kernel/random/boot_id)"
out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-g3-z-e8-route-bundle/${session_id}"

cd "$repo"
test ! -e "$out"
install -d "$out"

cache_hash() {
    rg --files "$cache" | sort | xargs sha256sum | sha256sum | cut -d' ' -f1
}

{
    printf 'classification=authoritative 2048/510 acceptance; fresh-process ABBAAB; n=3/arm\n'
    printf 'fixture=2048 prompt / 510 generated / batch 1 / temp 0 / prose / top-k 6\n'
    printf 'A=shipping route v3; B=105 same-input fused grids plus grouped-E8 U4\n'
    printf 'projection_ms=0.772 projection_route_percent=2.055\n'
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
    local pair=0
    local grouped=0
    local launches=2320
    local symbols=32
    local boundaries=2319
    local hash=4393256763546932634
    local dwords='5774[67]'
    local certified=32
    if [[ "$arm" == B ]]; then
        pair=1
        grouped=1
        launches=2215
        symbols=33
        boundaries=2214
        hash=12445125801925253362
        dwords='5703[56]'
        certified=33
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
        HIPFIRE_GFX1151_E8_PAIR="$pair" \
        HIPFIRE_GFX1151_GROUPED_E8_U4="$grouped" \
        HIPFIRE_REPLAY_BACKEND=redline \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE=1 \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"

    date -Iseconds > "$dir/end.iso"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids \
        > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'kernel_cache_hash=%s\n' "$(cache_hash)" > "$dir/cache.after.txt"
    rg "Radiowave code-object contracts|Radiowave argument effects|PM4 wait audit|retained route ready|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"

    rg -q "Radiowave code-object contracts: certified_artifacts=${certified}/${certified}" \
        "$dir/stderr.txt"
    rg -q "Radiowave argument effects: certified_launches=${launches} fallback_launches=0 unknown_launches=0" \
        "$dir/stderr.txt"
    rg -q "boundaries=${boundaries} covered=${boundaries}" "$dir/stderr.txt"
    rg -q "launch_count: ${launches}, unique_kernel_count: ${symbols}, sequence_hash: ${hash}" \
        "$dir/stderr.txt"
    rg -q "command_dwords: Some\\(${dwords}\\)" "$dir/stderr.txt"
    if rg -q "falling back|replay failed|illegal|FAULT|panic|JIT compile failed" \
        "$dir/stderr.txt"; then
        printf 'failure/fallback marker in %s\n' "$dir/stderr.txt" >&2
        exit 5
    fi
    printf 'completed run%s-%s: %s\n' \
        "$index" "$arm" "$(tail -n 1 "$dir/summary.txt")"
}

index=0
for arm in A B B A A B; do
    index=$((index + 1))
    run_arm "$index" "$arm"
done

reference="$out/run1-A/stdout.txt"
for output in "$out"/run*-*/stdout.txt; do
    cmp "$reference" "$output"
done
printf 'bundle acceptance output/resource/tape gates PASS\n'
rg "\\[stats\\]" "$out"/run*-*/stderr.txt
