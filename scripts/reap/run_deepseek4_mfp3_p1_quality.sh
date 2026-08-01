#!/usr/bin/env bash
# Run one target-only ctx=1024 quality bucket for the DeepSeek-V4
# MQ2R-derived MFP3 P1 GPTQ surgery.
#
# Usage:
#   run_deepseek4_mfp3_p1_quality.sh \
#     <wq-b|wq-b-wo-a|wq-b-wo-a-wo-b|attn-all|attn-plus-shared-gate-up|p1-full> \
#     <wikitext|code>
#
# Candidate artifacts and logits are staged on hipx's local NVMe. The immutable
# MQ2-Lloyd and frozen-P3 references stay on k9lin for the paired KLD pass.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
RECIPE_ROOT=${RECIPE_ROOT:-"$CAMPAIGN/candidates/mfp3-p1-gptq-v1"}
REMOTE_HOST=${REMOTE_HOST:-hipx}
REMOTE_REPO=${REMOTE_REPO:-/home/kaden/hipfire-ds4-gfx1151-opt}
REMOTE_BIN=${REMOTE_BIN:-/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_perplexity}
REMOTE_BASE=${REMOTE_BASE:-/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r}
REMOTE_ROOT=${REMOTE_ROOT:-/home/kaden/ds4-mfp3-p1-gptq-v1}
HIP_DEVICE=${HIP_DEVICE:-1}
CTX=${CTX:-1024}
WARMUP=${WARMUP:-8}
STAGE=${1:?usage: $0 <stage> <wikitext|code>}
CORPUS_TAG=${2:?usage: $0 <stage> <wikitext|code>}
KLD_SOURCE="$ROOT/scripts/reap/kld_compare.rs"
KLD_BIN="$ROOT/target/release/deepseek4_kld_compare"

case "$STAGE" in
    wq-b | wq-b-wo-a | wq-b-wo-a-wo-b | attn-all | attn-plus-shared-gate-up)
        CANDIDATE="$RECIPE_ROOT/stages/$STAGE"
        ;;
    p1-full)
        CANDIDATE="$RECIPE_ROOT/p1-full"
        ;;
    attr-*)
        CANDIDATE="$RECIPE_ROOT/attribution/$STAGE"
        ;;
    *)
        echo "unknown MFP3 P1 stage: $STAGE" >&2
        exit 2
        ;;
esac

case "$CORPUS_TAG" in
    wikitext)
        CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
        BASELINE="$CAMPAIGN/results/baseline-wikitext-ctx$CTX.logits"
        P3_REFERENCE="$CAMPAIGN/results/p3-gptq-head-wikitext-ctx$CTX.logits"
        CURRENT_P3_REFERENCE="$RECIPE_ROOT/controls/p3-current-wikitext-ctx$CTX.logits"
        ;;
    code)
        CORPUS=benchmarks/prompts/longcode_pflash.jsonl
        BASELINE="$CAMPAIGN/results/baseline-code-ctx$CTX.logits"
        P3_REFERENCE="$CAMPAIGN/results/p3-gptq-head-code-ctx$CTX.logits"
        CURRENT_P3_REFERENCE="$RECIPE_ROOT/controls/p3-current-code-ctx$CTX.logits"
        ;;
    *)
        echo "corpus must be wikitext or code" >&2
        exit 2
        ;;
esac
if [[ -s "$CURRENT_P3_REFERENCE" ]]; then
    P3_REFERENCE="$CURRENT_P3_REFERENCE"
fi

for path in \
    "$CANDIDATE/overlay.hfq" \
    "$CANDIDATE/reap_plan.json" \
    "$BASELINE" \
    "$P3_REFERENCE"; do
    if [[ ! -s "$path" ]]; then
        echo "missing quality input: $path" >&2
        exit 2
    fi
done
if [[ ! -x "$KLD_BIN" || "$KLD_SOURCE" -nt "$KLD_BIN" ]]; then
    rustc -O "$KLD_SOURCE" -o "$KLD_BIN"
fi

RESULTS="$RECIPE_ROOT/results"
REMOTE_CANDIDATE="$REMOTE_ROOT/stages/$STAGE"
REMOTE_RESULTS="$REMOTE_ROOT/quality/$STAGE"
mkdir -p "$RESULTS"
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "mkdir -p $(printf '%q' "$REMOTE_CANDIDATE") $(printf '%q' "$REMOTE_RESULTS")"
rsync -a --partial \
    "$CANDIDATE/reap_plan.json" \
    "$CANDIDATE/overlay.hfq" \
    "$REMOTE_HOST:$REMOTE_CANDIDATE/"

tag="$STAGE-$CORPUS_TAG-ctx$CTX"
log="$RESULTS/$tag.log"
logits="$RESULTS/$tag.logits"
kld_baseline="$RESULTS/$tag-vs-mq2lloyd.kld.txt"
kld_p3="$RESULTS/$tag-vs-p3.kld.txt"
remote_log="$REMOTE_RESULTS/$tag.log"
remote_logits="$REMOTE_RESULTS/$tag.logits"

overlay_sha=$(sha256sum "$CANDIDATE/overlay.hfq" | cut -d' ' -f1)
plan_sha=$(sha256sum "$CANDIDATE/reap_plan.json" | cut -d' ' -f1)
corpus_md5=$(md5sum "$ROOT/$CORPUS" | cut -d' ' -f1)
printf 'stage=%s corpus=%s ctx=%s warmup=%s corpus_md5=%s overlay_sha256=%s plan_sha256=%s\n' \
    "$STAGE" "$CORPUS_TAG" "$CTX" "$WARMUP" "$corpus_md5" "$overlay_sha" "$plan_sha"

remote=(
    env
    -u HIPFIRE_DSPARK_DRAFT
    -u HIPFIRE_DFLASH_DRAFT
    -u HIPFIRE_DS4_DENSE_ACT_DIR
    -u HIPFIRE_DEEPSEEK4_REAP_KEEPMAP
    "ROCR_VISIBLE_DEVICES=$HIP_DEVICE"
    HIPFIRE_DEEPSEEK4_LOAD_MTP=0
    "HIPFIRE_REAP_PLAN=$REMOTE_CANDIDATE"
    HIPFIRE_REQUIRE_REAP_OVERLAY=1
    "$REMOTE_BIN"
    "$REMOTE_BASE"
    "$CORPUS"
    --ctx "$CTX"
    --warmup "$WARMUP"
    --dump-logits "$remote_logits"
)
printf -v remote_command '%q ' "${remote[@]}"
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd $(printf '%q' "$REMOTE_REPO") && set -o pipefail && \
     $remote_command 2>&1 | tee $(printf '%q' "$remote_log")"
scp -q "$REMOTE_HOST:$remote_log" "$log"
scp -q "$REMOTE_HOST:$remote_logits" "$logits"

if ! rg -q "Model .*overlay ACTIVE" "$log" ||
    ! rg -q "MQ2R-derived MFP3 P1 GPTQ surgery verified" "$log"; then
    echo "quality run did not prove the admitted overlay was active" >&2
    exit 1
fi

"$KLD_BIN" "$BASELINE" "$logits" | tee "$kld_baseline.partial"
mv "$kld_baseline.partial" "$kld_baseline"
"$KLD_BIN" "$P3_REFERENCE" "$logits" | tee "$kld_p3.partial"
mv "$kld_p3.partial" "$kld_p3"
sha256sum "$log" "$logits" "$kld_baseline" "$kld_p3"
