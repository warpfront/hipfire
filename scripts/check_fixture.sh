#!/usr/bin/env bash
# Verify a model path IS the pinned canonical dense fixture before any number
# is taken on it. AGENTS.md §5 pins qwen3.8-27b.mq4-xt by size + sha256; the
# discarded non-AWQ upload (14980361216 B, 9f91556f…) has been found under
# three different paths on two boxes this week and silently produced a full
# day of numbers against the wrong quality baseline. Size check is instant and
# catches the known lookalike; pass --sha to also verify the digest (~10 s).
#
#   scripts/check_fixture.sh [--sha] [PATH]   (default: ~/.hipfire/models/qwen3.8-27b.mq4-xt)
set -euo pipefail
PIN_SIZE=14987185152
PIN_SHA=80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3
STALE_SIZE=14980361216
want_sha=0
if [ "${1:-}" = "--sha" ]; then want_sha=1; shift; fi
path="${1:-$HOME/.hipfire/models/qwen3.8-27b.mq4-xt}"
real=$(readlink -f "$path")
size=$(stat -c %s "$real")
if [ "$size" = "$STALE_SIZE" ]; then
  echo "FAIL: $path -> $real is the DISCARDED non-AWQ trunk ($STALE_SIZE B, 9f91556f…). AGENTS.md §5: not comparable; do not measure on it." >&2
  exit 2
fi
if [ "$size" != "$PIN_SIZE" ]; then
  echo "FAIL: $path -> $real size $size != pinned $PIN_SIZE" >&2
  exit 2
fi
if [ "$want_sha" = 1 ]; then
  got=$(sha256sum "$real" | cut -d' ' -f1)
  if [ "$got" != "$PIN_SHA" ]; then
    echo "FAIL: sha256 $got != pinned $PIN_SHA" >&2
    exit 2
  fi
fi
echo "OK: $path -> $real ($size B$([ "$want_sha" = 1 ] && echo ", sha256 verified"))"
