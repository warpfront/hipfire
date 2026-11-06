#!/bin/bash
# Decode bench for the Qwen4 hetero expert split: the Strix alone (gfx1151) against
# R9700 hot experts + Strix iGPU cold arm, same build, N fresh processes per arm.
# Usage: tools/hetero/bench_decode.sh PROMPT_FILE [N=3] [MAX_TOKENS=256]
# Needs this branch's binaries (the installed CLI drops topology.hetero_experts).
set -u
PROMPT=${1:?prompt file}
N=${2:-3}
TOK=${3:-256}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HF="$ROOT/target/release/hipfire"
export HIPFIRE_DAEMON_BIN="$ROOT/target/release/daemon"
[ -x "$HF" ] && [ -x "$HIPFIRE_DAEMON_BIN" ] || { echo "build first: cargo build --release -p hipfire-cli -p hipfire-daemon"; exit 1; }
OUT=${OUT:-$(mktemp -d)}
COMMON=(bench qwen3.8:flash-next --prompt-file "$PROMPT" --max-seq 32768 --runs 1 --warmups 1 --max-tokens "$TOK" --spec off)

run() { # name, then env assignments
    local name=$1; shift
    for i in $(seq "$N"); do
        env "$@" timeout 2400 "$HF" "${COMMON[@]}" ${EXTRA:-} > "$OUT/$name-$i.log" 2>&1
        echo "$name $i exit=$? $(grep -E '^ +decode +median' "$OUT/$name-$i.log" | awk '{print "decode", $3}') $(grep -E '^ +prefill +median' "$OUT/$name-$i.log" | awk '{print "prefill", $3}')"
    done
}

echo "logs: $OUT"
run strix HIPFIRE_DEVICES=gfx1151
EXTRA="--kv-backend legacy" run hetero HIPFIRE_DEVICES=gfx1201,gfx1151 HIPFIRE_TOPOLOGY_HETERO_EXPERTS=gfx1151 \
    HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_QWEN4_DECODE_SHARE=0 HIPFIRE_QWEN4_D2_DEFAULT=1
grep -L 'hetero experts: igpu arm' "$OUT"/hetero-*.log | sed 's/^/WARNING: no iGPU arm in /'
journalctl -k --since "-90min" --no-pager | grep -cE '(05:00.0|c3:00.0).*(page fault|reset)' | sed 's/^/kernel GPU faults in the last 90 min: /'
