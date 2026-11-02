#!/usr/bin/env bash
# Stop this install's serve, wait for the port, optionally restart `hipfire serve`.
# Usage: serve-restart.sh [port] [--kill-only] [-- <extra serve args>]
set -uo pipefail
PORT=11435; KILL_ONLY=0; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --kill-only) KILL_ONLY=1; shift;;
  --) shift; EXTRA=("$@"); break;;
  *) PORT="$1"; shift;; esac; done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HIPFIRE="${HIPFIRE_CLI_BIN:-$ROOT/target/release/hipfire}"
[ -x "$HIPFIRE" ] || HIPFIRE="$(command -v hipfire || true)"
[ -x "$HIPFIRE" ] || { echo "[serve-restart] native hipfire CLI not found" >&2; exit 1; }
echo "[serve-restart] stopping this install's serve/daemon, freeing :$PORT"
# `stop --force` signals only this install's validated serve, pidfile and
# GPU-lock holders. Never pkill by name or `fuser -k` the port: on a shared
# host those hit other agents' and other worktrees' serves and daemons.
"$HIPFIRE" stop --force "$PORT"
for i in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "[serve-restart] WARN port still busy"; exit 1; }
echo "[serve-restart] clean"; rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head
[ "$KILL_ONLY" = 1 ] && exit 0
echo "[serve-restart] launching"
rm -f ~/.hipfire/serve.log
setsid "$HIPFIRE" serve 0.0.0.0 "$PORT" "${EXTRA[@]}" >~/.hipfire/serve.log 2>&1 & disown
for i in $(seq 1 60); do
  curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  grep -qiE "port in use|JSON Parse|FATAL" ~/.hipfire/serve.log && break
  sleep 2
done
tail -3 ~/.hipfire/serve.log
