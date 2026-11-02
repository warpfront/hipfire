#!/usr/bin/env bash
# Verify a model path IS the pinned canonical dense fixture before any number
# is taken on it. AGENTS.md §5 pins qwen3.8-27b.mq4-xts (H2) by size + sha256.
# The asymmetric qwen3.8-27b.mq4-xt and the retired mq4-xts test upload have
# exactly the pinned size, and lookalikes have turned up under the canonical
# filename (e.g. a local qwen3.8-27b.mq4-xt symlinked to a symmetric file), so
# the sha256 is always verified (~10 s). The discarded non-AWQ mq4-xt upload
# (14980361216 B, 9f91556f…) is still recognised by size.
#
#   scripts/check_fixture.sh [PATH]   (default: ~/.hipfire/models/qwen3.8-27b.mq4-xts)
set -euo pipefail
PIN_SIZE=14987185152
PIN_SHA=3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae
STALE_SIZE=14980361216
path="${1:-$HOME/.hipfire/models/qwen3.8-27b.mq4-xts}"
real=$(readlink -f "$path")
size=$(stat -c %s "$real")
if [ "$size" = "$STALE_SIZE" ]; then
  echo "FAIL: $path -> $real is the DISCARDED non-AWQ mq4-xt trunk ($STALE_SIZE B, 9f91556f…). AGENTS.md §5: not comparable; do not measure on it." >&2
  exit 2
fi
if [ "$size" != "$PIN_SIZE" ]; then
  echo "FAIL: $path -> $real size $size != pinned $PIN_SIZE" >&2
  exit 2
fi
got=$(sha256sum "$real" | cut -d' ' -f1)
if [ "$got" != "$PIN_SHA" ]; then
  case "$got" in
    80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3) what=" (asymmetric qwen3.8-27b.mq4-xt, the previous fixture)" ;;
    de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5) what=" (retired mq4-xts QAT r7s200 test upload)" ;;
    *) what="" ;;
  esac
  echo "FAIL: $path -> $real sha256 $got$what != pinned $PIN_SHA" >&2
  exit 2
fi
echo "OK: $path -> $real ($size B, sha256 verified)"
