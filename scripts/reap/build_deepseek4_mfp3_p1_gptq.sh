#!/usr/bin/env bash
# Build source-derived DeepSeek-V4 P1 matrices as MFP3G32E8+GPTQ.
#
# Usage:
#   build_deepseek4_mfp3_p1_gptq.sh layer <0..42>
#   build_deepseek4_mfp3_p1_gptq.sh all
#
# The full build quantizes seven P1 tensors per layer once, merges all 301
# records, then emits cumulative class subsets for paired quality attribution.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
RECIPE_ROOT=${RECIPE_ROOT:-"$CAMPAIGN/candidates/mfp3-p1-gptq-v1"}
SOURCE_DIR=${SOURCE_DIR:-"$CAMPAIGN/source"}
HESSIAN_DIR=${HESSIAN_DIR:-"$CAMPAIGN/calibration/mfp3-p1-gptq-v1/hessians"}
QUANT_BIN=${QUANT_BIN:-"$ROOT/target/release/hipfire-quantize"}
PLAN_SOURCE="$ROOT/scripts/reap/deepseek4_mfp3_p1_plan.rs"
PLAN_BIN="$ROOT/target/release/deepseek4_mfp3_p1_plan"
MERGE_SOURCE="$ROOT/scripts/reap/hfq_overlay_merge.rs"
MERGE_BIN="$ROOT/target/release/hfq_overlay_merge"
SUBSET_SOURCE="$ROOT/scripts/reap/hfq_overlay_subset.rs"
SUBSET_BIN="$ROOT/target/release/hfq_overlay_subset"
MODE=${1:?usage: $0 <layer N|all>}
LAYER=${2:-}

if [[ ! -x "$QUANT_BIN" ]]; then
    cargo build --release -p hipfire-quantize --bin hipfire-quantize
fi
if [[ ! -x "$PLAN_BIN" || "$PLAN_SOURCE" -nt "$PLAN_BIN" ]]; then
    rustc -O "$PLAN_SOURCE" -o "$PLAN_BIN"
fi
if [[ ! -x "$MERGE_BIN" || "$MERGE_SOURCE" -nt "$MERGE_BIN" ]]; then
    rustc -O "$MERGE_SOURCE" -o "$MERGE_BIN"
fi
if [[ ! -x "$SUBSET_BIN" || "$SUBSET_SOURCE" -nt "$SUBSET_BIN" ]]; then
    rustc -O "$SUBSET_SOURCE" -o "$SUBSET_BIN"
fi

metadata_files=(
    config.json
    tokenizer.json
    tokenizer_config.json
    generation_config.json
    model.safetensors.index.json
)
for metadata in "${metadata_files[@]}"; do
    if [[ ! -s "$SOURCE_DIR/$metadata" ]]; then
        echo "missing pinned source metadata: $SOURCE_DIR/$metadata" >&2
        exit 2
    fi
done
if [[ ! -d "$HESSIAN_DIR" ]]; then
    echo "missing Hessian directory: $HESSIAN_DIR" >&2
    exit 2
fi

build_layer() {
    local layer=$1
    local layer_tag shard_tag shard candidate input_dir name
    printf -v layer_tag '%02d' "$layer"
    printf -v shard_tag '%05d' "$((layer + 2))"
    shard="model-$shard_tag-of-00046.safetensors"
    candidate="$RECIPE_ROOT/layers/layer-$layer_tag"
    input_dir="$candidate/source"
    if [[ ! -s "$SOURCE_DIR/$shard" ]]; then
        echo "missing pinned source shard: $SOURCE_DIR/$shard" >&2
        exit 2
    fi
    for suffix in \
        attn.wq_a.weight attn.wq_b.weight attn.wo_a.weight attn.wo_b.weight \
        ffn.shared_experts.w1.weight ffn.shared_experts.w2.weight \
        ffn.shared_experts.w3.weight; do
        name="layers.$layer.$suffix"
        if [[ ! -s "$HESSIAN_DIR/$name.hblk" ]]; then
            echo "missing Hessian: $HESSIAN_DIR/$name.hblk" >&2
            exit 2
        fi
    done
    mkdir -p "$candidate" "$input_dir"
    "$PLAN_BIN" p1-full "$candidate/reap_plan.json" "$layer"
    for metadata in "${metadata_files[@]}"; do
        ln -sfn "$SOURCE_DIR/$metadata" "$input_dir/$metadata"
    done
    ln -sfn "$SOURCE_DIR/$shard" "$input_dir/$shard"
    if [[ ! -s "$candidate/overlay.hfq" ]]; then
        HIPFIRE_E8_HESSIAN_DIR="$HESSIAN_DIR" "$QUANT_BIN" \
            --input "$input_dir" \
            --output "$candidate/unused.hfq" \
            --reap-overlay "$candidate" \
            --reap-out "$candidate/overlay.hfq" \
            --reap-arch deepseek4
    fi
    sha256sum "$SOURCE_DIR/$shard" "$candidate/reap_plan.json" "$candidate/overlay.hfq"
}

make_stage() {
    local stage=$1
    shift
    local stage_dir="$RECIPE_ROOT/stages/$stage"
    mkdir -p "$stage_dir"
    "$PLAN_BIN" "$stage" "$stage_dir/reap_plan.json"
    if [[ ! -s "$stage_dir/overlay.hfq" ]]; then
        local selectors=()
        local suffix
        for suffix in "$@"; do
            selectors+=(--suffix "$suffix")
        done
        "$SUBSET_BIN" \
            "$stage_dir/overlay.hfq" \
            "$RECIPE_ROOT/p1-full/overlay.hfq" \
            "${selectors[@]}"
    fi
    sha256sum "$stage_dir/reap_plan.json" "$stage_dir/overlay.hfq"
}

case "$MODE" in
    layer)
        if ! [[ "$LAYER" =~ ^[0-9]+$ ]] || ((LAYER < 0 || LAYER > 42)); then
            echo "layer must be an integer in 0..42, got '$LAYER'" >&2
            exit 2
        fi
        build_layer "$LAYER"
        ;;
    all)
        if [[ -n "$LAYER" ]]; then
            echo "all takes no layer argument" >&2
            exit 2
        fi
        overlays=()
        for layer in $(seq 0 42); do
            build_layer "$layer"
            printf -v layer_tag '%02d' "$layer"
            overlays+=("$RECIPE_ROOT/layers/layer-$layer_tag/overlay.hfq")
        done
        mkdir -p "$RECIPE_ROOT/p1-full"
        "$PLAN_BIN" p1-full "$RECIPE_ROOT/p1-full/reap_plan.json"
        if [[ ! -s "$RECIPE_ROOT/p1-full/overlay.hfq" ]]; then
            "$MERGE_BIN" "$RECIPE_ROOT/p1-full/overlay.hfq" "${overlays[@]}"
        fi
        make_stage wq-b .attn.wq_b.weight
        make_stage wq-b-wo-a .attn.wq_b.weight .attn.wo_a.weight
        make_stage wq-b-wo-a-wo-b \
            .attn.wq_b.weight .attn.wo_a.weight .attn.wo_b.weight
        make_stage attn-all \
            .attn.wq_a.weight .attn.wq_b.weight \
            .attn.wo_a.weight .attn.wo_b.weight
        make_stage attn-plus-shared-gate-up \
            .attn.wq_a.weight .attn.wq_b.weight \
            .attn.wo_a.weight .attn.wo_b.weight \
            .ffn.shared_experts.w1.weight .ffn.shared_experts.w3.weight
        sha256sum \
            "$RECIPE_ROOT/p1-full/reap_plan.json" \
            "$RECIPE_ROOT/p1-full/overlay.hfq"
        ;;
    *)
        echo "mode must be 'layer <0..42>' or 'all'" >&2
        exit 2
        ;;
esac
