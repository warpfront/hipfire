#!/usr/bin/env bash
# lifecycle: deprecated since 0.4.0, removal 0.5.0 — retired coherence-gate battery; historical reproduction only (docs/VALIDATION.md)
echo "warning: coherence-gate-ornith15.sh is deprecated and will be removed in 0.5.0; retired coherence-gate battery, not acceptance evidence (docs/VALIDATION.md)" >&2

# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery — Ornith 1.5 35B-A3B (arch_id 6), plain AR decode.
#
# Prompts use /no_think: Ornith is a thinking model and the thinking-tail
# attractor is a separate axis from quantization damage.
#
# Hard-fail (exit 1):
#   - daemon non-zero exit / panic / zero tokens / {"type":"error"}
#   - model absent                       (a skipped gate is a failed gate)
#   - max_token_frequency > 0.50 in the first 128 emitted tokens
#   - unique_token_ratio  < 0.15 in the first 128 emitted tokens
#   - max_token_frequency > 0.50 in the last  128 emitted tokens
#   - unique_token_ratio  < 0.30 in the last  128 emitted tokens
#
# Unlike coherence-gate-qwen35-dspark.sh, this gate does NOT check for any
# speculator engagement: Task 5 validates plain autoregressive decode on the
# freshly quantized qt44 trunk, and there is no sidecar attached yet. A later
# task (MTP sidecar) adds its own engagement check.
#
# Deliberate divergence from coherence-gate-qwen35-dspark.sh: a missing model
# is exit 1, not exit 0. A gate that skips proves nothing and reads as green.
#
# Exit codes: 0 clean (inspect report for fluency) · 1 hard error · 2 build/env error
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
MODEL="${HIPFIRE_ORNITH15_MODEL:-$MODELS_DIR/ornith-1.5-35b-a3b.mq4}"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-ornith15-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-420}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

if [ ! -x "$EXE" ]; then
    echo "coherence-gate-ornith15: building daemon..."
    if ! cargo build --release -p hipfire-daemon >&2; then
        echo "coherence-gate-ornith15: build failed" >&2
        exit 2
    fi
fi

if [ ! -f "$MODEL" ]; then
    echo "coherence-gate-ornith15: model absent at $MODEL — FAIL" >&2
    exit 1
fi

# ── GPU lock (best-effort: don't block on a missing/unreadable helper) ────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-ornith15" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

echo "# Coherence battery — Ornith 1.5 35B-A3B (qt44 mq4)" > "$OUT"
echo >> "$OUT"
echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" >> "$OUT"
echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" >> "$OUT"
echo "- date:   $(date -Iseconds)" >> "$OUT"
echo "- model:  \`$MODEL\`" >> "$OUT"
echo "- host: gfx1151 (scalar per-token path; qt44 batched prefill is gfx1200/1201-only)" >> "$OUT"
echo >> "$OUT"

./scripts/_coherence_runner.py \
    --exe "$EXE" --model "$MODEL" --out "$OUT" --timeout "$CASE_TIMEOUT" \
    --genre code     --prompt '/no_think Write a Python function that merges two sorted lists.' \
    --genre reason   --prompt '/no_think A train leaves at 09:15 and arrives at 13:40. How long is the journey? Show your working.' \
    --genre factual  --prompt '/no_think What is the capital of Australia, and why is it not Sydney?' \
    --genre prose    --prompt '/no_think Write a short paragraph describing a harbour at dawn.' \
    --genre instruct --prompt '/no_think List three things to check before deploying a database schema change.'
rc=$?

echo "report: $OUT"
exit "$rc"
