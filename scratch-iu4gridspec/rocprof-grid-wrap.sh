#!/usr/bin/env bash
set -euo pipefail
: "${HIPFIRE_ROCPROF_DAEMON_TARGET:?set target daemon}"
: "${HIPFIRE_ROCPROF_OUTPUT_DIR:?set trace output directory}"
mkdir -p "$HIPFIRE_ROCPROF_OUTPUT_DIR"
exec rocprofv3 \
  --kernel-trace \
  --hip-graph-trace \
  --stats \
  --kernel-include-regex gridspec \
  --output-format csv \
  --output-directory "$HIPFIRE_ROCPROF_OUTPUT_DIR" \
  --output-file daemon \
  -- "$HIPFIRE_ROCPROF_DAEMON_TARGET" "$@"
