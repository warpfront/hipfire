#!/usr/bin/env bash
# Close out one target-only ctx=1024 quality corpus on the standalone
# DeepSeek-V4 MQ2R-derived MFP3 P1 GPTQ candidate.
#
# Usage:
#   run_deepseek4_mfp3_p1_standalone_quality.sh <wikitext|code>
#
# The 82 GB candidate must already be baked on hipx local NVMe. This harness
# deliberately refuses every runtime overlay and disables MTP/DSpark so the
# result certifies the standalone artifact path.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
RECIPE_ROOT=${RECIPE_ROOT:-"$CAMPAIGN/candidates/mfp3-p1-gptq-v1"}
REMOTE_HOST=${REMOTE_HOST:-hipx}
REMOTE_REPO=${REMOTE_REPO:-/home/kaden/hipfire-ds4-gfx1151-opt}
REMOTE_BIN=${REMOTE_BIN:-/home/kaden/target-ds4-gfx1151-opt/release/examples/deepseek4_perplexity}
REMOTE_MODEL=${REMOTE_MODEL:-/home/kaden/ds4-mfp3-p1-gptq-v1/standalone/deepseek-v4-flash-mfp3p1.mq2r}
REMOTE_RESULTS=${REMOTE_RESULTS:-/home/kaden/ds4-mfp3-p1-gptq-v1/quality/standalone-170}
HIP_DEVICE=${HIP_DEVICE:-1}
CTX=${CTX:-1024}
WARMUP=${WARMUP:-8}
CORPUS_TAG=${1:?usage: $0 <wikitext|code>}
KLD_SOURCE="$ROOT/scripts/reap/kld_compare.rs"
KLD_BIN="$ROOT/target/release/deepseek4_kld_compare"

case "$CORPUS_TAG" in
    wikitext)
        CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
        BASELINE="$CAMPAIGN/results/baseline-wikitext-ctx$CTX.logits"
        P3_REFERENCE="$RECIPE_ROOT/controls/p3-current-wikitext-ctx$CTX.logits"
        ;;
    code)
        CORPUS=benchmarks/prompts/longcode_pflash.jsonl
        BASELINE="$CAMPAIGN/results/baseline-code-ctx$CTX.logits"
        P3_REFERENCE="$RECIPE_ROOT/controls/p3-current-code-ctx$CTX.logits"
        ;;
    *)
        echo "corpus must be wikitext or code" >&2
        exit 2
        ;;
esac

for path in "$BASELINE" "$P3_REFERENCE"; do
    if [[ ! -s "$path" ]]; then
        echo "missing quality reference: $path" >&2
        exit 2
    fi
done
if [[ ! -x "$KLD_BIN" || "$KLD_SOURCE" -nt "$KLD_BIN" ]]; then
    rustc -O "$KLD_SOURCE" -o "$KLD_BIN"
fi
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "test -s $(printf '%q' "$REMOTE_MODEL") && \
     mkdir -p $(printf '%q' "$REMOTE_RESULTS")"

RESULTS="$RECIPE_ROOT/results"
mkdir -p "$RESULTS"
tag="standalone-170-$CORPUS_TAG-ctx$CTX"
log="$RESULTS/$tag.log"
logits="$RESULTS/$tag.logits"
kld_baseline="$RESULTS/$tag-vs-mq2lloyd.kld.txt"
kld_p3="$RESULTS/$tag-vs-p3.kld.txt"
remote_log="$REMOTE_RESULTS/$tag.log"
remote_logits="$REMOTE_RESULTS/$tag.logits"
corpus_md5=$(md5sum "$ROOT/$CORPUS" | cut -d' ' -f1)
printf 'stage=standalone-170 corpus=%s ctx=%s warmup=%s corpus_md5=%s model=%s\n' \
    "$CORPUS_TAG" "$CTX" "$WARMUP" "$corpus_md5" "$REMOTE_MODEL"

remote=(
    env
    -u HIPFIRE_DSPARK_DRAFT
    -u HIPFIRE_DFLASH_DRAFT
    -u HIPFIRE_DS4_DENSE_ACT_DIR
    -u HIPFIRE_DEEPSEEK4_REAP_KEEPMAP
    -u HIPFIRE_REAP_PLAN
    -u HIPFIRE_REAP_OVERLAY
    -u HIPFIRE_REQUIRE_REAP_OVERLAY
    "ROCR_VISIBLE_DEVICES=$HIP_DEVICE"
    HIPFIRE_DEEPSEEK4_LOAD_MTP=0
    "$REMOTE_BIN"
    "$REMOTE_MODEL"
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

if ! rg -q "MQ2R-derived MFP3 P1 GPTQ surgery verified" "$log"; then
    echo "quality run did not prove the standalone derived recipe was active" >&2
    exit 1
fi
if rg -q "overlay ACTIVE" "$log"; then
    echo "standalone quality run unexpectedly attached an overlay" >&2
    exit 1
fi

"$KLD_BIN" "$BASELINE" "$logits" | tee "$kld_baseline.partial"
mv "$kld_baseline.partial" "$kld_baseline"
"$KLD_BIN" "$P3_REFERENCE" "$logits" | tee "$kld_p3.partial"
mv "$kld_p3.partial" "$kld_p3"
sha256sum "$log" "$logits" "$kld_baseline" "$kld_p3"
