#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# dflash_branch_bench.sh — comprehensive local 7900 XTX bench for 0.1.7-alpha
# release notes. Runs through each available target + its DFlash draft,
# measuring AR speed, DFlash τ, and
# multi-turn coherence. Outputs a structured markdown report.

set -uo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

HOME_MODELS="$HOME/.hipfire/models"
BRANCH_MODELS="$WORKDIR/models"
OUT_DIR="$WORKDIR/.bench_0_1_7_alpha"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/report.md"

SHORT_PROMPT="The James Madison wrote Federalist No. 10 arguing that a large republic would curb the effects of factions better than a small one. Explain in detail how this argument works and whether it still applies today."

CODE_PROMPT="def fibonacci(n: int) -> int:
    \"\"\"Compute nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Explain why this is O(2^n) and how memoization fixes it."

MATH_PROMPT="Find all integer solutions (x, y) to the equation x^2 + 2xy + y^2 = 169. Show your reasoning step by step."

# ── Reporting helpers ──────────────────────────────────────────────
init_report() {
    {
        echo "# hipfire dflash → 0.1.7-alpha local bench report"
        echo
        echo "Platform: $(grep 'Marketing Name' /sys/class/drm/card*/device/rocm-* 2>/dev/null | head -1 || echo 'gfx1100 7900 XTX')"
        echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
        echo "Date: $(date -u +%FT%TZ)"
        echo
    } > "$REPORT"
}

section() { echo; echo "## $1"; echo; }
add() { echo "$*" >> "$REPORT"; }

# ── Runner ─────────────────────────────────────────────────────────
run_and_capture() {
    local label="$1"; shift
    local out="$OUT_DIR/${label}.log"
    echo "[bench] $label" >&2
    "$@" > "$out" 2>&1 || echo "[bench] $label non-zero exit" >&2
    echo "$out"
}

extract_tok_s() { grep -Eo '\([0-9]+\.[0-9]+ tok/s\)' "$1" | tail -1 || echo "?"; }
extract_tau()   { grep -Eo 'τ=[0-9]+\.[0-9]+' "$1" | tail -1 || echo "?"; }
extract_accepted()  { grep -Eo 'accepted: [0-9]+' "$1" | tail -1 || echo "?"; }

# ── Config ─────────────────────────────────────────────────────────
declare -A MODELS=(
    [4b]="$HOME_MODELS/qwen3.5-4b.mq4"
    [9b]="$HOME_MODELS/qwen3.5-9b.mq4"
    [27b]="$HOME_MODELS/qwen3.5-27b.mq4"
)
declare -A DRAFTS=(
    [4b]="$BRANCH_MODELS/qwen35-4b-dflash-mq4.hfq"
    [9b]="$BRANCH_MODELS/qwen35-9b-dflash-mq4.hfq"
    [27b]="$HOME_MODELS/qwen35-27b-dflash-mq4.hfq"
)
# ── Run ────────────────────────────────────────────────────────────
init_report

section "1. AR speedbench (short prompt, 200-tok decode)"
add '| model | ctx | decode tok/s |'
add '|-------|-----|--------------|'
for M in 4b 9b 27b; do
    log=$(run_and_capture "ar_${M}" \
        ./target/release/examples/bench_qwen35_mq4 "${MODELS[$M]}" --max 200 --ctx 4096)
    ts=$(extract_tok_s "$log")
    add "| $M | 4096 | $ts |"
done

section "2. DFlash τ + tok/s on 3 prompt types (200-tok decode, ctx=4K)"
add '| model | prompt | tok/s | τ | accepted |'
add '|-------|--------|-------|---|----------|'
for M in 4b 9b 27b; do
    [ -f "${DRAFTS[$M]}" ] || { add "| $M | — | — | — | no draft |"; continue; }
    for P in short code math; do
        case $P in
            short) PROMPT="$SHORT_PROMPT";;
            code)  PROMPT="$CODE_PROMPT";;
            math)  PROMPT="$MATH_PROMPT";;
        esac
        log=$(run_and_capture "dflash_${M}_${P}" \
            ./target/release/examples/dflash_spec_demo \
                --target "${MODELS[$M]}" --draft "${DRAFTS[$M]}" \
                --prompt "$PROMPT" --max 200 --ctx 4096 --no-chatml)
        ts=$(extract_tok_s "$log")
        tau=$(extract_tau "$log")
        acc=$(extract_accepted "$log")
        add "| $M | $P | $ts | $tau | $acc |"
    done
done

section "3. Multi-turn coherence (9B MQ4, 3-turn ChatML)"
MULTITURN_PROMPT="USER: My name is Kaden and I'm working on Rust GPU inference.
ASSISTANT: Nice to meet you Kaden! Working on Rust GPU inference is fascinating.
USER: What GPU do I have?
ASSISTANT: You mentioned you're working on Rust GPU inference, but you haven't told me which GPU.
USER: It's a 7900 XTX. Can you remember my name from earlier?"
log=$(run_and_capture "multiturn_9b" \
    ./target/release/examples/dflash_spec_demo \
        --target "${MODELS[9b]}" --draft "${DRAFTS[9b]}" \
        --prompt "$MULTITURN_PROMPT" --max 80 --ctx 2048)
add '```'
tail -30 "$log" >> "$REPORT"
add '```'

section "Summary"
add 'See each section above. Per-run logs under `'"$OUT_DIR"'`.'

echo
echo "[bench] Done. Report: $REPORT"
