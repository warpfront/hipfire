#!/usr/bin/env bash
set -euo pipefail

repo=/home/kaden/hipfire-ds4-gfx1151-opt
binary=/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_chat
standalone=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
base=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd
plan=/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723/candidates/p3-all-layers-gptq-head
cache=/home/kaden/.hipfire_kernels/ds4-mq2r-perm
prompt=${HIPFIRE_BENCH_PROMPT:-benchmarks/prompts/deepseek4_mq2r_prose_ctx8192.txt}
fixture_tag=${HIPFIRE_BENCH_FIXTURE_TAG:-legacy-prefix}
tile_prompt=${HIPFIRE_BENCH_TILE_PROMPT:-0}
experts_per_tok=${HIPFIRE_BENCH_EXPERTS_PER_TOK:-6}
if [[ "$fixture_tag" == legacy-prefix && "$experts_per_tok" == 6 ]]; then
    out=/home/kaden/ds4-gfx1151-evidence/2026-07-24-g1-comparison-2048-510-abba
    route_sequence_hash=6907000390144547465
elif [[ "$fixture_tag" == legacy-prefix ]]; then
    out="/home/kaden/ds4-gfx1151-evidence/2026-07-24-g1-comparison-2048-510-top${experts_per_tok}-abba"
    route_sequence_hash=2541189520118612547
else
    out="/home/kaden/ds4-gfx1151-evidence/2026-07-25-comparison-${fixture_tag}-2048-510-top${experts_per_tok}-abba"
    if [[ "$experts_per_tok" == 6 ]]; then
        route_sequence_hash=6907000390144547465
    else
        route_sequence_hash=2541189520118612547
    fi
fi
order=(A B B A A B)

cd "$repo"
install -d "$out"

{
    printf 'fixture_tag=%s\n' "$fixture_tag"
    printf 'fixture=exactly 2048 tokenizer tokens derived from %s; tile=%s; 510 generated; batch 1; temp 0; prose\n' "$prompt" "$tile_prompt"
    printf 'A=standalone MQ2R on local NVMe\n'
    printf 'B=MQ2-Lloyd base plus P3 overlay on NAS\n'
    printf 'num_experts_per_tok=%s\n' "$experts_per_tok"
    if [[ "$experts_per_tok" == 6 ]]; then
        printf 'quality_config=checkpoint/shipping default\n'
    else
        printf 'quality_config=benchmark-only override; shipping default unchanged at 6\n'
    fi
    printf 'standalone_sha256=392325b5a8cd284c8f305f23f74f178007a14b88173babeb3f4784ec4fc0e511\n'
    printf 'base_sha256=ab8a8900f6792199975e7e2b15854bb869a023ca3fddf62981c1b9cd600e9b96\n'
    printf 'overlay_sha256=bc3644259952efca32074085ed69bcbeb0fad12fb23972fe5694ac9bf13323b8\n'
    stat -c 'standalone path=%n size=%s device=%D' "$standalone"
    stat -f -c 'standalone filesystem=%T' "$standalone"
    findmnt -T "$standalone"
    stat -c 'base path=%n size=%s device=%D' "$base"
    stat -f -c 'base filesystem=%T' "$base"
    findmnt -T "$base"
    stat -c 'overlay path=%n size=%s device=%D' "$plan/overlay.hfq"
    stat -f -c 'overlay filesystem=%T' "$plan/overlay.hfq"
    findmnt -T "$plan/overlay.hfq"
    md5sum "$binary" "$prompt"
    sha256sum "$binary"
    (
        cd "$cache/gfx1151"
        rg --files -g '*.hsaco' | sort | xargs sha256sum
    ) | sha256sum
} > "$out/identity.txt"

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

run_one() {
    local index=$1
    local arm=$2
    local model
    local label
    local -a artifact_env
    if [[ "$arm" == A ]]; then
        model=$standalone
        label=standalone-local
        artifact_env=()
    else
        model=$base
        label=overlay-nas
        artifact_env=(
            "HIPFIRE_REAP_PLAN=$plan"
            "HIPFIRE_REQUIRE_REAP_OVERLAY=1"
            "HIPFIRE_DEEPSEEK4_E8_WO_GROUPED=1"
            "HIPFIRE_DEEPSEEK4_E8_U4=1"
            "HIPFIRE_DEEPSEEK4_E8_PREFILL_B2=1"
            "HIPFIRE_DEEPSEEK4_E8_PREFILL_B4=1"
            "HIPFIRE_DEEPSEEK4_FFN_OVERLAP=0"
            "HIPFIRE_DEEPSEEK4_MQ2_PERM=1"
            "HIPFIRE_DEEPSEEK4_HC_PINGPONG=1"
            "HIPFIRE_DEEPSEEK4_HC_CONTROL_VEC4=1"
            "HIPFIRE_DEEPSEEK4_HC_FINALIZE_FUSED=1"
            "HIPFIRE_DEEPSEEK4_HC_CONTROL_FINALIZE_FUSED=1"
            "HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=0"
            "HIPFIRE_DEEPSEEK4_ROPE_WIDE=1"
            "HIPFIRE_DEEPSEEK4_ATTN_SCOREGRID=1"
            "HIPFIRE_DEEPSEEK4_RMS_PLAIN_NOX=1"
            "HIPFIRE_DEEPSEEK4_MQ2_DOWN_K8ALL=1"
            "HIPFIRE_DEEPSEEK4_INDEXER_TOPK_PARALLEL=1"
        )
    fi
    local run
    run=$(printf '%02d-%s-%s' "$index" "$arm" "$label")
    local dir="$out/$run"

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

    date -Iseconds > "$dir/start.iso"
    rocm-smi --showproductname --showbus --showmeminfo vram --showclocks --showpower --showtemp --showuse --showpids \
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
        HIPFIRE_DEEPSEEK4_BENCH_TILE_PROMPT="$tile_prompt" \
        HIPFIRE_DEEPSEEK4_BENCH_STDIN_ALL=1 \
        HIPFIRE_DEEPSEEK4_BENCH_EXPERTS_PER_TOK="$experts_per_tok" \
        HIPFIRE_DEEPSEEK4_TEMP=0 \
        HIPFIRE_DEEPSEEK4_TOP_K=0 \
        HIPFIRE_DEEPSEEK4_SEED=424242 \
        HIPFIRE_DEEPSEEK4_CHAT_RAW=1 \
        HIPFIRE_DEEPSEEK4_SPEC_DECODE=0 \
        HIPFIRE_DEEPSEEK4_GRAPH=0 \
        HIPFIRE_REPLAY_BACKEND=redline \
        HIPFIRE_REPLAY_TRANSPORT=pm4 \
        HIPFIRE_REPLAY_PM4_QUEUES=1 \
        HIPFIRE_REPLAY_PM4_WAIT_POLICY=resource \
        HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY=required-only \
        HIPFIRE_REPLAY_PM4_STATEFUL=static \
        "${artifact_env[@]}" \
        "$binary" < "$prompt" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    date -Iseconds > "$dir/end.iso"
    md5sum "$binary" "$prompt" "$dir/stdout.txt" "$dir/stderr.txt" > "$dir/md5.txt"
    (
        cd "$cache/gfx1151"
        rg --files -g '*.hsaco' | sort | xargs sha256sum
    ) | sha256sum >> "$dir/md5.txt"
    rg "GPU dev|MQ2R P3|overlay ACTIVE|Config:|Generation:|\\[prompt:|retained route ready|PM4 wait audit|\\[stats\\]" \
        "$dir/stderr.txt" > "$dir/summary.txt"
    rocm-smi --showclocks --showpower --showtemp --showuse --showpids > "$dir/topology.after.txt"
    amd-smi metric --gpu 3 > "$dir/amd-smi.after.txt"
    printf 'completed %s: %s\n' "$run" "$(tail -n 1 "$dir/summary.txt")"
}

for index in "${!order[@]}"; do
    run_one "$((index + 1))" "${order[$index]}"
done

reference="$out/01-A-standalone-local/stdout.txt"
for output in "$out"/*/stdout.txt; do
    cmp "$reference" "$output"
done

for stderr in "$out"/*/stderr.txt; do
    rg -q "\\[prompt: 2048 tokens \\(pos 0 → 2048\\)\\]" "$stderr"
    rg -q "experts_per_tok=${experts_per_tok}" "$stderr"
    rg -q "retained route ready: capture=ReplayCaptureSummary \\{ launch_count: 2320, unique_kernel_count: 32, sequence_hash: ${route_sequence_hash} \\}" "$stderr"
    if rg -q "falling back|replay failed|illegal|FAULT|panic" "$stderr"; then
        printf 'failure/fallback marker in %s\n' "$stderr" >&2
        exit 5
    fi
done

printf '2048/510 standalone-local versus NAS-overlay output/route parity PASS\n'
rg "\\[stats\\]" "$out"/*/stderr.txt
