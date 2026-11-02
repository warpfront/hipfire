#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Serve gate for sampled requests on the exact VMM batch route
# (`serve.batch_sampled`): eight concurrent sampled MTP clients with thinking
# off must all get HTTP 200 and exactly one terminal.
#
# The harness sends its default max_tokens (81,920). The singleton MTP
# speculator fits that, the batch's MTP lane engine (head cap 32,768) does
# not, so such a request must take the singleton route before the daemon
# announces it. It once entered the batch, could not be handed back after its
# start was announced, and returned HTTP 400 ("sampled request cannot be
# served exactly by the VMM batch route") on 5 of 40 C8 requests.
#
# Usage: scripts/serve_sampled_batch_gate.sh MODEL [PORT]
#   Binaries: target/release/{hipfire,daemon}; override with HIPFIRE_CLI_BIN /
#   HIPFIRE_DAEMON_BIN. Device selection is the caller's (lock + visible-device
#   env). Kernel cache: the caller's HIPFIRE_KERNEL_CACHE, if set.
#   The config lives in a private HIPFIRE_HOME; the workdir is kept on failure.

set -uo pipefail

MODEL="${1:?usage: $0 MODEL [PORT]}"
PORT="${2:-11493}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${HIPFIRE_CLI_BIN:-$ROOT/target/release/hipfire}"
DAEMON="${HIPFIRE_DAEMON_BIN:-$ROOT/target/release/daemon}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sampled-batch-gate.XXXXXX")"
SPID=""
RC=1

cleanup() {
  if [ -n "$SPID" ]; then
    # SIGTERM exactly the serve process group this script started.
    kill -TERM -- "-$SPID" 2>/dev/null
    wait "$SPID" 2>/dev/null
  fi
  if [ "$RC" = 0 ]; then rm -rf "$WORK"; else echo "workdir kept: $WORK" >&2; fi
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$CLI" ] || fail "build first: cargo build --release -p hipfire-cli (or set HIPFIRE_CLI_BIN)"
[ -x "$DAEMON" ] || fail "build first: cargo build --release -p hipfire-daemon (or set HIPFIRE_DAEMON_BIN)"

mkdir -p "$WORK/home"
cat > "$WORK/home/config.toml" <<'EOF'
[speculation]
mode = "mtp"
[serve]
vmm_batch = true
continuous_batch_size = 8
batch_sampled = true
EOF

HIPFIRE_HOME="$WORK/home" HIPFIRE_DAEMON_BIN="$DAEMON" \
  setsid "$CLI" serve "127.0.0.1:$PORT" "$MODEL" > "$WORK/serve.log" 2>&1 &
SPID=$!

ready=0
for _ in $(seq 1 900); do
  kill -0 "$SPID" 2>/dev/null || { tail -20 "$WORK/serve.log" >&2; fail "serve exited before it was ready"; }
  if python3 -I - "$PORT" <<'EOF' 2>/dev/null
import http.client, json, sys
c = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=5)
c.request("GET", "/health"); r = c.getresponse(); h = json.loads(r.read())
sys.exit(0 if r.status == 200 and h.get("model") and not h.get("loading_model") else 1)
EOF
  then ready=1; break; fi
  sleep 1
done
[ "$ready" = 1 ] || fail "serve never became ready on $PORT"
grep -q "VMM continuous batch staged" "$WORK/serve.log" || fail "the VMM batch route was not staged"

# serve_harness exits non-zero on any HTTP error or missing/duplicate terminal.
# max_tokens 40000 is above the MTP lane head cap (32,768) and inside the
# singleton speculator's context, so no request has a batch lane.
python3 -I "$ROOT/scripts/serve_harness.py" --no-spawn --port "$PORT" --model "$MODEL" \
  --mode battery --concurrent-clients 1,8 --thinking off --seed 42 --max-tokens 40000 \
  --sampling 'json:{"temperature":0.7,"top_p":0.95,"top_k":20}' \
  --out "$WORK/harness.json" > "$WORK/harness.txt" 2>&1
hrc=$?
tail -5 "$WORK/harness.txt"
[ "$hrc" = 0 ] || fail "serve_harness rc=$hrc (see $WORK/harness.txt)"

# On the recorded rows: every request at every level answered 200 with one
# terminal, C8 ran all eight clients' full batteries, every request ran the
# singleton MTP route (never an AR substitute), and with the one wire seed
# each prompt's C8 text equals its C1 text.
python3 -I - "$WORK/harness.json" <<'EOF' || fail "harness rows failed the gate"
import json, sys
rows = json.load(open(sys.argv[1]))
bad = [r for r in rows if r.get("http_status") not in (None, 200) or r.get("terminal_count") != 1]
not_mtp = [r for r in rows if not r.get("mtp") or r.get("ar_windows")]
c1 = [r for r in rows if r["concurrency"] == 1]
c8 = [r for r in rows if r["concurrency"] == 8]
ref = {r["prompt_index"]: r["content"] for r in c1}
differ = [r for r in c8 if r["content"] != ref.get(r["prompt_index"])]
print(f"rows={len(rows)} c1={len(c1)} c8={len(c8)} non200_or_bad_terminal={len(bad)} "
      f"not_singleton_mtp={len(not_mtp)} c8_text_differs_from_c1={len(differ)}")
for r in (bad + not_mtp + differ)[:8]:
    print(f"  c{r['concurrency']}/client{r['client']}/turn{r['turn']}: http={r.get('http_status')} "
          f"mtp={r.get('mtp')} ar_windows={r.get('ar_windows')} {r.get('stream_error')}")
sys.exit(0 if not (bad or not_mtp or differ) and c1 and len(c8) == 8 * len(c1) else 1)
EOF

echo "PASS: sampled MTP batch, thinking off, C1+C8: every request HTTP 200"
RC=0
