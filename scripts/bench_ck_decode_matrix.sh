#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${SIDECAR:-$ROOT/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so}"
BENCH="${BENCH:-$ROOT/target/release/examples/bench_qwen35_mq4}"
GPU_ID="${GPU_ID:-0}"
PREFILLS="${PREFILLS:-512 2048 8192}"
GEN_TOKENS="${GEN_TOKENS:-4096}"
RUNS="${RUNS:-5}"
TRIM_EACH_SIDE="${TRIM_EACH_SIDE:-1}"
WARMUP_TOKENS="${WARMUP_TOKENS:-8}"
COOLDOWN="${COOLDOWN:-10}"
WORKSPACE_BYTES="${WORKSPACE_BYTES:-536870912}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/ck-decode-matrix/$RUN_ID}"

for file in "$MODEL" "$SIDECAR" "$BENCH"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done
[[ "$RUNS" -gt $((2 * TRIM_EACH_SIDE)) ]] || {
    echo "RUNS must exceed 2 * TRIM_EACH_SIDE" >&2
    exit 2
}
if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing non-empty OUT_ROOT: $OUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUT_ROOT/raw"

{
    echo "git_head=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "prefills=$PREFILLS"
    echo "gen_tokens=$GEN_TOKENS"
    echo "runs=$RUNS"
    echo "trim_each_side=$TRIM_EACH_SIDE"
    sha256sum "$MODEL" "$SIDECAR" "$BENCH"
    rocm-smi --showproductname --showmeminfo vram
} >"$OUT_ROOT/meta.txt" 2>&1

for prefill in $PREFILLS; do
    for mode in native ck; do
        for ((run = 1; run <= RUNS; run++)); do
            log="$OUT_ROOT/raw/pp${prefill}_tg${GEN_TOKENS}_${mode}_run${run}.log"
            echo "[$mode] PP${prefill}/TG${GEN_TOKENS} run ${run}/${RUNS}"
            if [[ "$mode" == "ck" ]]; then
                env HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_KV_MODE=asym3 \
                    HIPFIRE_FLASH_ATTN_CK_LIB="$SIDECAR" \
                    HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES="$WORKSPACE_BYTES" \
                    "$BENCH" "$MODEL" --prefill "$prefill" --prefill-runs 1 \
                    --warmup "$WARMUP_TOKENS" --gen "$GEN_TOKENS" 2>&1 | tee "$log"
            else
                env -u HIPFIRE_FLASH_ATTN_CK_LIB -u HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES \
                    -u HIPFIRE_FLASH_PREFILL HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_KV_MODE=asym3 \
                    "$BENCH" "$MODEL" --prefill "$prefill" --prefill-runs 1 \
                    --warmup "$WARMUP_TOKENS" --gen "$GEN_TOKENS" 2>&1 | tee "$log"
            fi
            [[ "$run" -eq "$RUNS" ]] || sleep "$COOLDOWN"
        done
    done
done

python3 - "$OUT_ROOT" "$TRIM_EACH_SIDE" <<'PY'
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
trim = int(sys.argv[2])
pattern = re.compile(r"SUMMARY\s+gen_tok_s=([0-9.]+).*?prefill_tok_s=([0-9.]+)")
groups = {}
for path in sorted((root / "raw").glob("*.log")):
    match_name = re.fullmatch(r"pp(\d+)_tg(\d+)_(native|ck)_run(\d+)\.log", path.name)
    if not match_name:
        continue
    text = path.read_text(errors="replace")
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"expected one SUMMARY in {path}, found {len(matches)}")
    pp, tg, mode, run = match_name.groups()
    decode, prefill = map(float, matches[0])
    groups.setdefault((int(pp), int(tg), mode), []).append(
        {"run": int(run), "decode_tok_s": decode, "prefill_tok_s": prefill}
    )

rows = []
for (pp, tg, mode), samples in sorted(groups.items()):
    samples.sort(key=lambda sample: sample["run"])
    decode = sorted(sample["decode_tok_s"] for sample in samples)
    prefill = sorted(sample["prefill_tok_s"] for sample in samples)
    trimmed_decode = decode[trim:len(decode) - trim]
    trimmed_prefill = prefill[trim:len(prefill) - trim]
    rows.append({
        "prefill_tokens": pp,
        "generated_tokens": tg,
        "mode": mode,
        "samples": samples,
        "decode_median_tok_s": statistics.median(decode),
        "decode_trimmed_mean_tok_s": statistics.fmean(trimmed_decode),
        "prefill_median_tok_s": statistics.median(prefill),
        "prefill_trimmed_mean_tok_s": statistics.fmean(trimmed_prefill),
    })

output = {"trim_each_side": trim, "rows": rows}
(root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
for row in rows:
    print(
        f"PP{row['prefill_tokens']}/TG{row['generated_tokens']} {row['mode']}: "
        f"decode median={row['decode_median_tok_s']:.3f} tok/s, "
        f"trimmed mean={row['decode_trimmed_mean_tok_s']:.3f} tok/s"
    )
PY
