#!/usr/bin/env bash
# Hardware check for #1058: a runtime that cannot reserve virtual address space
# (hipMemAddressReserve -> hipErrorOutOfMemory) must not fail the load.
#
#   vmm_reserve_fallback_check.sh <bindir> <model> <outdir> [max_seq]
#
#   bindir   directory holding the release `hipfire` CLI and `daemon`
#   model    registry tag or path (e.g. qwen3.8:27b-mq4-xts, qwen3.5:9b)
#   outdir   receives one log per leg
#   max_seq  default 16384
#
# Run it inside ONE fabric job on the card under test (it starts the daemon
# through the CLI and kills nothing it did not start). Three legs:
#   1. control: no fault. Automatic selection keeps the VMM backend.
#   2. HIPFIRE_FAULT_HIP=reserve:1000000 (every reservation fails like the
#      reporter's stack): the load succeeds on the legacy backend, the one
#      WARNING line carries the driver's error, nothing says "load failed",
#      and decoded text comes back.
#   3. the same fault with an explicit --kv-backend vmm: refused at admission
#      (no weights uploaded) with a message that names --kv-backend legacy.
# Exits non-zero on the first violated expectation.
set -uo pipefail

BIN=${1:?bindir}
MODEL=${2:?model}
OUT=${3:?outdir}
MAX_SEQ=${4:-16384}
PROMPT="Write a quick C++ quicksort function with no commentary."

mkdir -p "$OUT"
export HIPFIRE_DAEMON_BIN="$BIN/daemon"
fail=0

check() { # <leg-log> <description> <grep -E pattern>
    if grep -Eq -- "$3" "$1"; then
        echo "PASS  $2"
    else
        echo "FAIL  $2 (no match for: $3 in $1)"
        fail=1
    fi
}
refute() { # <leg-log> <description> <grep -E pattern>
    if grep -Eq -- "$3" "$1"; then
        echo "FAIL  $2 (unexpected match for: $3 in $1)"
        fail=1
    else
        echo "PASS  $2"
    fi
}

run_leg() { # <name> <fault spec or empty> <extra run args...>
    local name=$1 fault=$2
    shift 2
    local log="$OUT/$name.log"
    {
        echo "# leg=$name fault=${fault:-none} bin=$BIN model=$MODEL max_seq=$MAX_SEQ"
        md5sum "$BIN/daemon" "$BIN/hipfire"
        echo "# cmd: hipfire run --max-seq $MAX_SEQ $* $MODEL <prompt>"
    } >"$log"
    if [ -n "$fault" ]; then
        HIPFIRE_FAULT_HIP="$fault" timeout 900 "$BIN/hipfire" run --max-seq "$MAX_SEQ" "$@" "$MODEL" "$PROMPT" >>"$log" 2>&1
    else
        timeout 900 "$BIN/hipfire" run --max-seq "$MAX_SEQ" "$@" "$MODEL" "$PROMPT" >>"$log" 2>&1
    fi
    echo "# exit=$?" >>"$log"
    echo "$log"
}

echo "== leg 1: control (no fault) =="
log=$(run_leg control "")
check "$log" "control exits 0" '^# exit=0$'
check "$log" "control keeps the VMM backend" 'vmm backend'
refute "$log" "control logs no VMM-unavailable warning" 'automatic VMM selection unavailable'

echo "== leg 2: every hipMemAddressReserve fails, automatic backend =="
log=$(run_leg inject_auto "reserve:1000000")
check "$log" "automatic load exits 0" '^# exit=0$'
check "$log" "warning names the failed reservation" 'automatic VMM selection unavailable .*hipMemAddressReserve'
check "$log" "load summary reports the legacy backend" 'legacy backend'
refute "$log" "no load failure" 'load failed'
refute "$log" "no split-verify refusal" 'admits VMM KV only'
# The decoded text is the last block of the log; require code-shaped output.
check "$log" "decoded text is a quicksort" 'quicksort|void |int '

echo "== leg 3: every hipMemAddressReserve fails, explicit --kv-backend vmm =="
log=$(run_leg inject_explicit_vmm "reserve:1000000" --kv-backend vmm)
check "$log" "explicit vmm exits non-zero" '^# exit=[1-9]'
check "$log" "refusal names the fix" '--kv-backend legacy'
check "$log" "refusal carries the driver error" 'hipMemAddressReserve'
refute "$log" "refused before any weight was uploaded" 'loading layer'

if [ "$fail" -ne 0 ]; then
    echo "vmm_reserve_fallback_check: FAILED (logs in $OUT)"
    exit 1
fi
echo "vmm_reserve_fallback_check: all legs passed (logs in $OUT)"
