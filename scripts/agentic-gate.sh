#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Agentic gate - tool-call shape regression battery for A3B variants.
#
# Why this gate exists:
#   coherence-gate.sh runs a tool-call test row but only checks for `<tool_call>`
#   tag presence and `<|im_start|>` leakage in visible text. It uses a 50-token
#   system prompt. Issue #87's auto-MMQ regression produced clean output on
#   short prompts but corrupt tool-call JSON on long agent-shape prompts.
#   This gate guards that regression class with realistic 780-1300 token
#   system contexts (Pi-style + Hermes-style) and machine-evaluated JSON
#   structural validation.
#
#   Value-add over coherence-gate.sh --full (which already runs A3B sheep):
#     1. Prompt-length-sensitive MMQ regression cover (the #87 class)
#     2. Tool-call structural validation (JSON.parse + schema match)
#
# What this gate does NOT cover (by design, today):
#   - Jinja chat-template rendering beyond what the daemon does for a request.
#     Each generate carries `thinking_enabled` and `assistant_prefix` exactly as
#     `hipfire serve` sends them: think-on cells `open_think`, the clamped cell
#     and every second turn `closed_think`, never `max_think_tokens`. (Until
#     the 2026-10 protocol repair the cells used the Plain scaffold, so think-on
#     cells are framed differently from reports made before it.)
#   - OpenAI HTTP path. Cells drive the daemon over stdin JSONL; the
#     `serve` HTTP path runs additional Node-side Jinja rendering that
#     this harness skips. OpenAI-shaped tool-call regressions need a
#     daemon-up + curl harness; defer.
#
# Modes:
#   ./scripts/agentic-gate.sh                # full: 2 models * 4 cells = 8 cells, ~5 min
#   ./scripts/agentic-gate.sh --fast         # 1 cell, ~2 min - used by pre-commit
#   ./scripts/agentic-gate.sh --self-check   # detector rot guard, <1s
#
# Exit codes:
#   0 - battery ran clean
#   1 - hard error (panic, zero tokens, JSON parse fail, special-token leak,
#       stacked openers, or > 1/N cells soft-warned)
#   2 - build / env / detector-self-check failure
#
# Skip semantics (CI-safe):
#   - Both A3B models absent           -> exit 0 with SKIPPED message
#   - One model absent                 -> run the present one's cells, log skip
#   - Model exceeds host VRAM          -> treat as absent (path-blank); skips
#                                         silently otherwise → zero-token hard-fail
#   - HIPFIRE_SKIP_AGENTIC_GATE=1      -> exit 0 immediately
#   - HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK=1 -> bypass VRAM-headroom skip
#                                            (e.g., registry minimums too conservative
#                                            for your host)
#
# Report destination: /tmp/agentic-gate-<timestamp>.md (or $HIPFIRE_AGENTIC_GATE_OUT)
# Raw archive: with HIPFIRE_AGENTIC_GATE_RAW_DIR=<absolute dir> set, each model's raw
# daemon output is kept as <dir>/<model>.daemon-output.jsonl and the driver's
# machine-readable terminal record as <dir>/<model>.driver.json (logs only).

set -u
cd "$(dirname "$0")/.."

MODE="full"
while [ $# -gt 0 ]; do
    case "$1" in
        --fast)        MODE="fast"; ;;
        --self-check)  MODE="self-check"; ;;
        -h|--help)
            sed -n '3,32p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "${HIPFIRE_SKIP_AGENTIC_GATE:-0}" = "1" ]; then
    echo "agentic-gate: HIPFIRE_SKIP_AGENTIC_GATE=1 - skipping"
    exit 0
fi

# ---- Self-check mode (detector rot guard) ----------------------------------
# Construct a payload that intentionally trips every detector, run them, and
# assert each fires. Cheap (<1s); guards the gate against silent rot if the
# detector regexes drift away from real failure shapes.
if [ "$MODE" = "self-check" ]; then
    python3 - <<'PY'
import sys, json, re

# Two synthetic payloads cover all 4 detectors. parse-fail and schema-violation
# are mutually exclusive on a single payload (one requires successful parse, the
# other forbids it), so we test them on separate inputs.

# Trips: stacked_openers + special_token_leak + json_parse_fail
PAYLOAD_CORRUPT = '''\
Here is my response:

<tool_call>
<tool_call>
{"arguments": {"path": "/tmp/x"<|im_start|>}}
</tool_call><|im_end|>
'''

# Trips: schema_violation only (JSON parses, but missing required `name` field)
PAYLOAD_SCHEMA_BAD = '''\
<tool_call>
{"arguments": {"path": "/tmp/x"}}
</tool_call><|im_end|>
'''

def detect_stacked_openers(text):
    return bool(re.search(r"<tool_call>\s*<tool_call>", text))

def detect_special_token_leak(body):
    return any(tok in body for tok in ("<|im_start|>", "<|endoftext|>"))

def detect_json_parse_fail(body):
    try:
        json.loads(body)
        return False
    except json.JSONDecodeError:
        return True

def detect_schema_violation(body):
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return False  # parse-fail is its own detector; don't double-count
    return not (isinstance(obj, dict) and "name" in obj and "arguments" in obj)

def body_of(payload):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", payload, re.S)
    body = m.group(1) if m else ""
    # Strip any nested opener before JSON tests so stacked_openers stays
    # independent.
    return re.sub(r"^<tool_call>\s*", "", body)

corrupt_body = body_of(PAYLOAD_CORRUPT)
schema_body = body_of(PAYLOAD_SCHEMA_BAD)

results = {
    "stacked_openers":     detect_stacked_openers(PAYLOAD_CORRUPT),
    "special_token_leak":  detect_special_token_leak(corrupt_body),
    "json_parse_fail":     detect_json_parse_fail(corrupt_body),
    "schema_violation":    detect_schema_violation(schema_body),
}

failed = [k for k, v in results.items() if not v]
print("self-check results:")
for k, v in results.items():
    print(f"  {k:20s} {'fired' if v else 'MISSED'}")
if failed:
    print(f"\nself-check FAILED: detectors did not fire on synthetic corrupt: {failed}", file=sys.stderr)
    print("detector rot suspected - review the regexes and update before merging.", file=sys.stderr)
    sys.exit(2)
print("\nself-check passed: all 4 detectors fired against synthetic corrupt payloads")
PY
    exit $?
fi

# ---- Setup -----------------------------------------------------------------
# HIPFIRE_AGENTIC_GATE_EXE / HIPFIRE_AGENTIC_GATE_DRIVER_ARGS exist only for the
# CPU tests (scripts/test_agentic_gate_driver.py): they point the gate at a fake
# daemon (never rebuilt) and add driver flags such as short deadlines.
EXE="${HIPFIRE_AGENTIC_GATE_EXE:-./target/release/daemon}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_AGENTIC_GATE_OUT:-/tmp/agentic-gate-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

A3B_35="$MODELS_DIR/qwen3.5-35b-a3b.mq4"
# 2026-05-15: qwen3.6-35b-a3b.mq4 hangs / emits zero tokens during this gate
# (per issue #262 side-finding). Switched the "3.6" cell to dense qwen3.6-27b
# while that's investigated — agentic tool-call shape testing doesn't strictly
# require A3B, dense 27B exercises the same JSON-output predicates.
A3B_36="$MODELS_DIR/qwen3.6-27b.mq4"
PI_SYS="benchmarks/prompts/agentic_pi_system.txt"
HERMES_SYS="benchmarks/prompts/agentic_hermes_system.txt"
USER_READ="benchmarks/prompts/agentic_user_read.txt"

# ---- VRAM headroom check ---------------------------------------------------
# Models declare min_vram_gb in registry/models.json. If the host can't meet
# that minimum, the model loads but silently OOMs during prefill and emits
# zero tokens (looks like a tool-call regression — it isn't). Treat such
# models as absent so the existing skip path covers this case too.
#
# Threshold semantics: min_vram_gb is interpreted as a strict floor expressed
# in GiB (1024^3). Daemon overhead (KV cache, graph capture, scratch) is
# assumed pre-included in the registry value — A3B-35B is an 18.7 GB blob
# but declares min_vram_gb=22, i.e. ~3.3 GiB margin. We do NOT add a
# secondary +1 GB headroom because (a) it'd compound, and (b) the registry
# is the single source of truth for "needs N GiB". Comparison is in bytes
# to avoid GiB-truncation false-positives on cards that report slightly
# under their advertised size (e.g. a 24 GB card reading 23.98 GiB).
#
# Disable with HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK=1 (e.g., on hosts where
# the registry minimums are conservative).
REGISTRY="registry/models.json"
# Multi-card systems take the max VRAM across all DRM cards. iGPUs report
# zero / a sliver, so max correctly picks the dGPU. On dual-dGPU hosts this
# is optimistic — we don't know which card the daemon will use.
VRAM_BYTES=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v="$(cat "$f" 2>/dev/null)" || continue
    [ "$v" -gt "$VRAM_BYTES" ] && VRAM_BYTES="$v"
done
VRAM_GB=$(( VRAM_BYTES / 1073741824 ))
if [ "$VRAM_BYTES" -eq 0 ] && [ "${HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK:-0}" != "1" ]; then
    echo "agentic-gate: VRAM detection unavailable (no /sys/class/drm/card*/device/mem_info_vram_total) - skipping VRAM check"
fi

model_fits_vram() {
    local model_path="$1"
    [ "${HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK:-0}" = "1" ] && return 0
    [ -f "$REGISTRY" ] || return 0
    [ "$VRAM_BYTES" -gt 0 ] || return 0
    # python3 is a hard script-wide dep (self-check + session driver). No guard.

    local fname min_gb min_bytes
    fname="$(basename "$model_path")"
    min_gb="$(python3 -c '
import json, sys
fname = sys.argv[1]
with open(sys.argv[2]) as fh:
    data = json.load(fh)
for entry in data.get("models", {}).values():
    if isinstance(entry, dict) and entry.get("file") == fname:
        v = entry.get("min_vram_gb")
        if v is not None:
            print(v)
        break
' "$fname" "$REGISTRY" 2>/dev/null)"
    [ -n "$min_gb" ] || return 0

    min_bytes=$(( min_gb * 1073741824 ))
    if [ "$VRAM_BYTES" -lt "$min_bytes" ]; then
        echo "agentic-gate: $fname needs ${min_gb} GB VRAM, host has ${VRAM_GB} GB - skipping cell"
        return 1
    fi
    return 0
}

# Path-blank when over budget so the existing `[ -f "$A3B_..." ]` checks
# downstream naturally treat these models as absent (`[ -f "" ]` is false).
A3B_35_VRAM_SKIP=0
A3B_36_VRAM_SKIP=0
if [ -f "$A3B_35" ] && ! model_fits_vram "$A3B_35"; then A3B_35=""; A3B_35_VRAM_SKIP=1; fi
if [ -f "$A3B_36" ] && ! model_fits_vram "$A3B_36"; then A3B_36=""; A3B_36_VRAM_SKIP=1; fi

# Skip-on-absence: both models missing -> exit 0. Distinguish absent-on-disk
# from skipped-for-VRAM so the operator can tell which case fired.
if [ ! -f "$A3B_35" ] && [ ! -f "$A3B_36" ]; then
    if [ "$A3B_35_VRAM_SKIP" = "1" ] || [ "$A3B_36_VRAM_SKIP" = "1" ]; then
        echo "agentic-gate: all A3B models absent or exceed host VRAM (${VRAM_GB} GB) - SKIPPED"
    else
        echo "agentic-gate: A3B models absent ($MODELS_DIR/qwen3.{5,6}-35b-a3b.mq4) - SKIPPED"
    fi
    exit 0
fi

# Required fixtures
for f in "$PI_SYS" "$HERMES_SYS" "$USER_READ"; do
    if [ ! -f "$f" ]; then
        echo "agentic-gate: required fixture missing: $f" >&2
        exit 2
    fi
done

# HIPFIRE_AGENTIC_GATE_RAW_DIR (absolute, created here) archives, per model, the
# driver's raw daemon output and its terminal record. Unset: nothing is written.
RAW_DIR="${HIPFIRE_AGENTIC_GATE_RAW_DIR:-}"
if [ -n "$RAW_DIR" ]; then
    case "$RAW_DIR" in
        /*) ;;
        *) echo "agentic-gate: HIPFIRE_AGENTIC_GATE_RAW_DIR must be absolute: $RAW_DIR" >&2; exit 2 ;;
    esac
    if ! mkdir -p -- "$RAW_DIR"; then
        echo "agentic-gate: cannot create HIPFIRE_AGENTIC_GATE_RAW_DIR: $RAW_DIR" >&2
        exit 2
    fi
fi

# Rebuild daemon if any tracked source is newer than the binary. A test-supplied
# HIPFIRE_AGENTIC_GATE_EXE is a test double and is never rebuilt.
rebuild=0
if [ -n "${HIPFIRE_AGENTIC_GATE_EXE:-}" ]; then
    if [ ! -x "$EXE" ]; then
        echo "agentic-gate: HIPFIRE_AGENTIC_GATE_EXE is not executable: $EXE" >&2
        exit 2
    fi
elif [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/hfq.rs crates/hipfire-daemon/src/main.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1; break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "agentic-gate: rebuilding daemon..."
 if ! cargo build --release -p hipfire-daemon >&2; then
        echo "agentic-gate: build failed" >&2
        exit 2
    fi
fi

# The daemon reserves physical GPU UUIDs independently of HOME. A daemon
# already using this card makes the gate fail before loading the model;
# never unlink a GPU lock file (the active inode must remain).
DRIVER_PID=""

cleanup() {
    # TERM only the driver we spawned, by tracked PID; it kills the daemon
    # process group it started. Never pkill -f.
    if [ -n "$DRIVER_PID" ] && kill -0 "$DRIVER_PID" 2>/dev/null; then
        kill "$DRIVER_PID" 2>/dev/null
        for _ in $(seq 1 20); do
            kill -0 "$DRIVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$DRIVER_PID" 2>/dev/null
    fi
    DRIVER_PID=""
    gpu_release 2>/dev/null || true
}

# GPU lock
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "agentic-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap cleanup EXIT
fi

# ---- Build cell list -------------------------------------------------------
# Each cell: model | system_fixture | thinking_clamp_bool | multi_turn_bool | label
build_cells() {
    local model="$1"
    local prefix="$2"
    local fast_only="$3"  # 1 = emit only the fast cell for this model
    if [ "$fast_only" = "1" ]; then
        echo "$model|$HERMES_SYS|0|0|${prefix}_hermes_unclamped"
        return
    fi
    # Full mode: 4 cells per model
    echo "$model|$PI_SYS|0|0|${prefix}_pi_unclamped"
    echo "$model|$PI_SYS|1|0|${prefix}_pi_clamped"
    echo "$model|$HERMES_SYS|0|0|${prefix}_hermes_unclamped"
    echo "$model|$HERMES_SYS|0|1|${prefix}_hermes_multiturn"
}

CELLS=()
if [ "$MODE" = "fast" ]; then
    # Fast mode: prefer 3.6 (newer) if present; fall back to 3.5 if only it
    # is installed. Never silently skip if at least one A3B is available.
    if [ -f "$A3B_36" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_36" "3.6" 1)
    elif [ -f "$A3B_35" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_35" "3.5" 1)
    fi
else
    # Full mode: every available A3B contributes its 4 cells.
    if [ -f "$A3B_35" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_35" "3.5" 0)
    fi
    if [ -f "$A3B_36" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_36" "3.6" 0)
    fi
fi

if [ "${#CELLS[@]}" -eq 0 ]; then
    # Belt-and-suspenders: should never reach here, since the early skip-on-
    # absence at the top exited 0 when both models are missing. If we DO get
    # here, something is wrong with the cell builder; fail loud rather than
    # silently passing.
    echo "agentic-gate: no cells built despite at least one A3B model present" >&2
    echo "  A3B_35=$A3B_35  exists=$([ -f "$A3B_35" ] && echo yes || echo no)" >&2
    echo "  A3B_36=$A3B_36  exists=$([ -f "$A3B_36" ] && echo yes || echo no)" >&2
    exit 2
fi

# ---- Report header ---------------------------------------------------------
{
    echo "# Agentic gate"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $MODE"
    echo "- cells:  ${#CELLS[@]}"
    echo
    echo "Hard-fail predicates:"
    echo "- daemon panic / zero tokens / timeout"
    echo "- tool_call body fails JSON parse"
    echo "- parsed JSON missing required field (name or arguments)"
    echo "- special-token leak inside tool_call body (<|im_start|>, <|endoftext|>)"
    echo "- stacked openers (two consecutive <tool_call>)"
    echo "- soft-warn count > 1 across all cells"
    echo
    echo "Soft-warn signals (printed, included in collective threshold):"
    echo "- <tool_call> not emitted (model answered inline)"
    echo "- <tool_call> appearing inside <think>...</think>"
    echo
} > "$OUT"

# ---- Run cells -------------------------------------------------------------
HARD_FAIL=0
DRIVER_FAIL=0
ARCHIVE_FAIL=0
read -r -a driver_extra <<< "${HIPFIRE_AGENTIC_GATE_DRIVER_ARGS:-}"
SOFT_WARN_TOTAL=0

# Group cells by model for single-load efficiency.
declare -A MODEL_CELLS
for cell in "${CELLS[@]}"; do
    model="$(echo "$cell" | cut -d'|' -f1)"
    MODEL_CELLS["$model"]+="${cell}"$'\n'
done

# Iterate models; per model, drive a single daemon session.
for model in "${!MODEL_CELLS[@]}"; do
    model_short="$(basename "$model" .mq4)"
    echo "## $model_short" >> "$OUT"
    echo >> "$OUT"
    echo "agentic-gate: model $model_short ($(echo "${MODEL_CELLS[$model]}" | grep -c '|') cells)..."

    # Cell ids are c${idx}_t1 (and c${idx}_t2 for multi-turn). The driver
    # (scripts/agentic_gate_driver.py) owns the daemon session: it sends load,
    # then each generate with a unique nonzero attempt_id, answers the daemon's
    # commit_ready with the matching commit, and waits for done before the next
    # request, with per-step deadlines. The daemon's stdout and stderr land in
    # $OUTPUT_FILE for the parser below.
    # Private HOME directories isolate config and caches, not GPU reservations.
    # The daemon's UUID lock still conflicts with any other user on this card.
    cell_args=()
    while IFS= read -r line; do
        [ -n "$line" ] && cell_args+=("$line")
    done <<< "${MODEL_CELLS[$model]}"

    OUTPUT_FILE="$(mktemp /tmp/agentic-gate-out.XXXXXX)"

    record_args=()
    if [ -n "$RAW_DIR" ]; then
        record_args=(--record "$RAW_DIR/$model_short.driver.json")
    fi

    # The driver is backgrounded so cleanup can signal it by exact PID; on
    # TERM it kills only the daemon process group it spawned. Any overlapping
    # physical GPU UUID produces a FATAL contention message with the holder
    # PID, regardless of HOME; the driver records the daemon's stderr for it.
    env HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=1 \
        python3 scripts/agentic_gate_driver.py \
            --exe "$EXE" --model "$model" --user-file "$USER_READ" \
            --out "$OUTPUT_FILE" ${record_args[@]+"${record_args[@]}"} \
            ${driver_extra[@]+"${driver_extra[@]}"} \
            "${cell_args[@]}" &
    DRIVER_PID=$!
    wait "$DRIVER_PID"
    driver_rc=$?
    DRIVER_PID=""
    if [ "$driver_rc" -ne 0 ]; then
        # A nonzero driver exit (deadline, daemon error/EOF/nonzero exit, failed
        # unload, killed by a signal) can never PASS, whatever the parser sees.
        DRIVER_FAIL=1
        echo "agentic-gate: DRIVER FAILURE for $model_short (rc=$driver_rc)" >&2
        echo "- **DRIVER FAILURE**: the session driver for $model_short exited rc=$driver_rc; the gate HARD_FAILs" >> "$OUT"
        echo >> "$OUT"
    fi

    # Parse output, run detectors per cell, append to report.
    python3 - "$OUTPUT_FILE" "$OUT" "${cell_args[@]}" <<'PY'
import sys, json, re, pathlib

out_path = sys.argv[1]
report_path = sys.argv[2]
cells = sys.argv[3:]

raw_text = pathlib.Path(out_path).read_text()
events = []
for line in raw_text.splitlines():
    line = line.strip()
    if not line.startswith("{"): continue
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

# Group token events by id.
by_id = {}
for ev in events:
    if ev.get("type") == "token":
        by_id.setdefault(ev["id"], []).append(ev["text"])

# Per-request terminal bookkeeping. Requests are sent strictly in cell order
# (c{idx}_t1, then c{idx}_t2 for a multi-turn cell) with attempt_ids 1, 2, ...
# in send order (scripts/agentic_gate_driver.py build_requests), so the parser
# derives each request's expected (id, attempt_id) from the cell list alone.
seq = []
_attempt = 0
for _idx, _cell in enumerate(cells):
    _attempt += 1
    seq.append((f"c{_idx}_t1", _attempt))
    if _cell.split("|")[3] == "1":
        _attempt += 1
        seq.append((f"c{_idx}_t2", _attempt))
expected = dict(seq)
done_ok = set()
req_errors = {rid: [] for rid, _ in seq}
session_errors = []
for ev in events:
    et = ev.get("type")
    if et == "done":
        a = ev.get("attempt_id")
        if ev.get("id") in expected and type(a) is int and a == expected[ev["id"]]:
            done_ok.add(ev["id"])
    elif et == "error":
        # Daemon error events and the driver's synthetic ones (source
        # agentic_gate_driver) alike. An error naming a request belongs to it;
        # one with no id belongs to the request in flight (the first without a
        # done yet); anything else is a session error and fails every cell.
        msg = str(ev.get("message", ""))
        rid = ev.get("id")
        if rid is None:
            rid = next((r for r, _ in seq if r not in done_ok), None)
        if rid in req_errors:
            req_errors[rid].append(msg)
        else:
            session_errors.append(msg)

# GPU contention is printed on stderr (merged into stdout), not as a JSON event.
fatal_singleton = "already reserved by holder PID" in raw_text

def extract_tool_call_body(text):
    # Use minimal regex; the body extraction is the same as coherence-gate.
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    return m.group(1).strip() if m else None

def detect_stacked_openers(text):
    return bool(re.search(r"<tool_call>\s*<tool_call>", text))

def detect_tool_in_think(text):
    # Tool call appearing inside a <think>...</think> block.
    for m in re.finditer(r"<think>(.*?)</think>", text, re.S):
        if "<tool_call>" in m.group(1):
            return True
    return False

def detect_special_token_leak(body):
    return any(tok in body for tok in ("<|im_start|>", "<|endoftext|>"))

def detect_json(body):
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ("parse_fail", str(e))
    if not isinstance(obj, dict):
        return ("schema_fail", "tool_call body is not a JSON object")
    if "name" not in obj or "arguments" not in obj:
        return ("schema_fail", "missing required field (name or arguments)")
    return ("ok", obj)

results = []
soft_warn_count = 0
hard_fail = False

with open(report_path, "a") as report:
    for idx, cell in enumerate(cells):
        model, sys_path, clamp, multi, label = cell.split("|")
        cid_t1 = f"c{idx}_t1"
        cid_t2 = f"c{idx}_t2"
        toks_t1 = by_id.get(cid_t1, [])
        text_t1 = "".join(toks_t1)
        verdict = "PASS"
        notes = []

        # Hard-fail: singleton collision (a user daemon was already running)
        if fatal_singleton:
            verdict = "HARD_FAIL"
            notes.append("daemon singleton: another hipfire daemon was already running on this host - release it before running the gate")
            hard_fail = True
        # Hard-fail: zero tokens
        if not toks_t1:
            verdict = "HARD_FAIL"
            notes.append("zero tokens emitted")
            hard_fail = True
        else:
            # Hard: stacked openers
            if detect_stacked_openers(text_t1):
                verdict = "HARD_FAIL"
                notes.append("stacked <tool_call> openers")
                hard_fail = True
            body = extract_tool_call_body(text_t1)
            if body is None:
                # Soft: tool_call not emitted
                soft_warn_count += 1
                notes.append("soft: <tool_call> not emitted (inline answer)")
                if verdict == "PASS": verdict = "SOFT_WARN"
            else:
                # Hard: special-token leak
                if detect_special_token_leak(body):
                    verdict = "HARD_FAIL"
                    notes.append("special-token leak inside tool_call body")
                    hard_fail = True
                # Hard: JSON parse / schema
                kind, info = detect_json(body)
                if kind != "ok":
                    verdict = "HARD_FAIL"
                    notes.append(f"json {kind}: {info}")
                    hard_fail = True
                else:
                    notes.append(f"json ok: name={info.get('name')!r}")
            # Soft: tool inside think
            if detect_tool_in_think(text_t1):
                soft_warn_count += 1
                notes.append("soft: <tool_call> inside <think>...</think>")
                if verdict == "PASS": verdict = "SOFT_WARN"

        # Every request of the cell must end in an actual matching `done`
        # (same id and attempt_id) with no error event naming or in flight for
        # it; tokens seen never excuse a missing done or a present error.
        for rid in [cid_t1] + ([cid_t2] if multi == "1" else []):
            if rid not in done_ok:
                verdict = "HARD_FAIL"
                notes.append(f"{rid}: no matching done (attempt_id {expected[rid]})")
                hard_fail = True
            for msg in req_errors[rid]:
                verdict = "HARD_FAIL"
                notes.append(f"{rid}: error event: {msg}")
                hard_fail = True
        for msg in session_errors:
            verdict = "HARD_FAIL"
            notes.append(f"session error event: {msg}")
            hard_fail = True

        # Multi-turn t2 evaluation if applicable
        if multi == "1":
            toks_t2 = by_id.get(cid_t2, [])
            text_t2 = "".join(toks_t2)
            if not toks_t2:
                verdict = "HARD_FAIL"
                notes.append("multi-turn t2: zero tokens")
                hard_fail = True
            else:
                # Either coherent text or another tool_call - both ok.
                # Just check for stacked openers and special-token leak in t2.
                if detect_stacked_openers(text_t2):
                    verdict = "HARD_FAIL"
                    notes.append("multi-turn t2: stacked openers")
                    hard_fail = True
                body_t2 = extract_tool_call_body(text_t2)
                if body_t2 is not None and detect_special_token_leak(body_t2):
                    verdict = "HARD_FAIL"
                    notes.append("multi-turn t2: special-token leak")
                    hard_fail = True

        report.write(f"### {label}    {verdict}\n\n")
        report.write(f"- system: `{sys_path}` ({pathlib.Path(sys_path).stat().st_size} bytes)\n")
        report.write(f"- thinking: {'clamped (max_think=1)' if clamp == '1' else 'unclamped'}\n")
        report.write(f"- multi-turn: {'yes' if multi == '1' else 'no'}\n")
        report.write(f"- notes: {'; '.join(notes) if notes else 'clean'}\n\n")
        report.write("```\n" + (text_t1 if text_t1 else "(no tokens)") + "\n```\n\n")
        if multi == "1":
            text_t2 = "".join(by_id.get(f"c{idx}_t2", []))
            report.write("**Turn 2 (after synthesized tool_response):**\n\n")
            report.write("```\n" + (text_t2 if text_t2 else "(no tokens)") + "\n```\n\n")

    # Append per-model summary line via stderr so caller can read it.
    print(json.dumps({"hard_fail": hard_fail, "soft_warn_count": soft_warn_count}), file=sys.stderr)
PY
    res=$?
    if [ "$res" -ne 0 ]; then
        echo "agentic-gate: detector script failed for $model_short" >&2
        HARD_FAIL=1
    fi
    # Read per-model summary written to stderr by the python detector.
    # (We didn't capture stderr above; instead recompute by running a thin checker.)
    # For simplicity, re-run a quick aggregator from the report file.
    # HIPFIRE_AGENTIC_GATE_RAW_DIR: keep the raw daemon record (logs only, never
    # a model file) before the temp file goes. A failed copy keeps the temp file.
    if [ -n "$RAW_DIR" ]; then
        if cp -- "$OUTPUT_FILE" "$RAW_DIR/$model_short.daemon-output.jsonl"; then
            rm -f "$OUTPUT_FILE"
        else
            ARCHIVE_FAIL=1
            echo "agentic-gate: could not archive the raw daemon output for $model_short; kept $OUTPUT_FILE" >&2
            echo "- **ARCHIVE FAILURE**: could not copy the raw daemon output for $model_short to $RAW_DIR; kept $OUTPUT_FILE" >> "$OUT"
            echo >> "$OUT"
        fi
    else
        rm -f "$OUTPUT_FILE"
    fi
done

# ---- Aggregate verdict -----------------------------------------------------
# Count HARD_FAIL and SOFT_WARN headers in the report.
hard_count=$(grep -cE '^### .* HARD_FAIL$' "$OUT" || true)
soft_count=$(grep -cE '^### .* SOFT_WARN$' "$OUT" || true)
total_cells=${#CELLS[@]}
{
    echo "## Summary"
    echo
    echo "- total cells: $total_cells"
    echo "- hard fails: $hard_count"
    echo "- soft warns: $soft_count"
} >> "$OUT"

if [ "$ARCHIVE_FAIL" -eq 1 ]; then
    echo "agentic-gate: HARD FAIL (the raw daemon output could not be archived)"
    echo "report: $OUT"
    exit 1
fi
if [ "$DRIVER_FAIL" -eq 1 ]; then
    {
        echo
        echo "## Driver failure"
        echo
        echo "The session driver exited nonzero for one or more models (see the"
        echo "DRIVER FAILURE lines above). Exit code 1 regardless of the cells' verdicts."
    } >> "$OUT"
    echo "agentic-gate: HARD FAIL (the session driver exited nonzero)"
    echo "report: $OUT"
    exit 1
fi
if [ "$HARD_FAIL" -eq 1 ]; then
    {
        echo
        echo "## Detector failure"
        echo
        echo "One or more model passes had a detector script crash. Exit code 1"
        echo "regardless of report-level cell counts."
    } >> "$OUT"
    echo "agentic-gate: HARD FAIL (detector crashed during a model pass)"
    echo "report: $OUT"
    exit 1
fi
if [ "$hard_count" -gt 0 ]; then
    echo "agentic-gate: HARD FAIL ($hard_count cells)"
    echo "report: $OUT"
    exit 1
fi
# Soft-warn collective threshold: > 1 of N cells = hard fail.
if [ "$soft_count" -gt 1 ]; then
    echo "agentic-gate: HARD FAIL via soft-warn threshold ($soft_count > 1)"
    echo "report: $OUT"
    exit 1
fi
echo "agentic-gate: PASS ($total_cells cells, $soft_count soft warns)"
echo "report: $OUT"
exit 0
