#!/usr/bin/env bash
# Build the two-corpus block Hessians used by the DeepSeek-V4 MFP3 P1 recipe.
#
# The activation directories must each contain the 258 files emitted by:
#   deepseek4_perplexity ... --ctx 1024 --dump-dense-acts <directory>
#
# shared_experts.w1 and w3 consume the same activation, so w3 is hard-linked
# from the computed w1 Hessian rather than repeating the O(K*256*rows) work.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
CALIBRATION_ROOT=${CALIBRATION_ROOT:-"$CAMPAIGN/calibration/mfp3-p1-gptq-v1"}
ACTS_WIKITEXT=${ACTS_WIKITEXT:-"$CALIBRATION_ROOT/acts/wikitext-ctx1024"}
ACTS_CODE=${ACTS_CODE:-"$CALIBRATION_ROOT/acts/code-ctx1024"}
OUT_DIR=${OUT_DIR:-"$CALIBRATION_ROOT/hessians"}
COLLECTOR=${COLLECTOR:-"$ROOT/target/release/collect_e8_hessian"}

if [[ ! -x "$COLLECTOR" ]]; then
    cargo build --release -p hipfire-quantize --bin collect_e8_hessian
fi
for directory in "$ACTS_WIKITEXT" "$ACTS_CODE"; do
    if [[ ! -d "$directory" ]]; then
        echo "missing activation directory: $directory" >&2
        exit 2
    fi
    count=$(rg --files "$directory" | wc -l)
    if [[ "$count" -ne 258 ]]; then
        echo "$directory: expected 258 activation files, found $count" >&2
        exit 2
    fi
done
mkdir -p "$OUT_DIR"

validate_acts() {
    local path=$1
    local rows k bytes expected
    read -r rows k < <(od -An -tu4 -N8 "$path")
    if [[ -z "$rows" || -z "$k" || "$rows" -eq 0 || "$k" -eq 0 || $((k % 256)) -ne 0 ]]; then
        echo "$path: invalid activation header rows=$rows K=$k" >&2
        exit 1
    fi
    bytes=$(stat -c %s "$path")
    expected=$((8 + rows * k * 4))
    if [[ "$bytes" -ne "$expected" ]]; then
        echo "$path: activation size $bytes != expected $expected" >&2
        exit 1
    fi
    printf '%s\n' "$k"
}

hessian_complete() {
    local path=$1
    local k=$2
    local bytes magic blocks stored_k expected
    [[ -s "$path" ]] || return 1
    read -r magic < <(od -An -tx4 -N4 "$path")
    read -r blocks stored_k < <(od -An -tu4 -j4 -N8 "$path")
    bytes=$(stat -c %s "$path")
    expected=$((12 + (k / 256) * 256 * 256 * 4))
    [[ "$magic" == "45384831" && "$blocks" -eq $((k / 256)) &&
        "$stored_k" -eq "$k" && "$bytes" -eq "$expected" ]]
}

mapfile -t act_files < <(rg --files "$ACTS_WIKITEXT" | sort)
completed=0
for wikitext_path in "${act_files[@]}"; do
    filename=${wikitext_path##*/}
    code_path="$ACTS_CODE/$filename"
    if [[ ! -f "$code_path" ]]; then
        echo "missing code activation peer: $code_path" >&2
        exit 1
    fi
    name=${filename%.acts}
    k_wikitext=$(validate_acts "$wikitext_path")
    k_code=$(validate_acts "$code_path")
    if [[ "$k_wikitext" -ne "$k_code" ]]; then
        echo "$name: corpus K mismatch $k_wikitext != $k_code" >&2
        exit 1
    fi
    hessian="$OUT_DIR/$name.hblk"
    if ! hessian_complete "$hessian" "$k_wikitext"; then
        "$COLLECTOR" \
            --acts "$wikitext_path" \
            --acts "$code_path" \
            --name "$name" \
            --out-dir "$OUT_DIR"
    fi
    completed=$((completed + 1))

    if [[ "$name" == *.ffn.shared_experts.w1.weight ]]; then
        w3_name=${name%.w1.weight}.w3.weight
        w3_hessian="$OUT_DIR/$w3_name.hblk"
        if ! hessian_complete "$w3_hessian" "$k_wikitext"; then
            if [[ -e "$w3_hessian" ]]; then
                echo "$w3_hessian exists but is incomplete; refusing to replace it" >&2
                exit 1
            fi
            ln "$hessian" "$w3_hessian"
        fi
    fi
done

hessian_count=$(rg --files "$OUT_DIR" | rg '\.hblk$' | wc -l)
if [[ "$completed" -ne 258 || "$hessian_count" -ne 301 ]]; then
    echo "incomplete Hessian set: acts=$completed hblk=$hessian_count (expected 258/301)" >&2
    exit 1
fi
sha256sum "$OUT_DIR"/*.hblk >"$OUT_DIR/sha256sums.txt"
printf 'DeepSeek4 MFP3 P1 Hessians complete: %s activation keys -> %s tensor Hessians\n' \
    "$completed" "$hessian_count"
