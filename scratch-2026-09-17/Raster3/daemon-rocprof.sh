#!/usr/bin/env bash
# Raster3 local daemon rocprof wrapper (untracked): daemon-wrap.sh + `-S`
# so per-kernel totals print to stderr (the daemon is SIGKilled by the bench
# harness, which suppresses CSV finalization; the summary still prints).
set -euo pipefail

: "${HIPFIRE_ROCPROF_DAEMON_TARGET:?set the real daemon path}"
: "${HIPFIRE_ROCPROF_OUTPUT_DIR:?set the rocprof output directory}"

ROCPROF_BIN="${HIPFIRE_ROCPROF_BIN:-rocprofv3}"
mkdir -p "${HIPFIRE_ROCPROF_OUTPUT_DIR}"

exec "${ROCPROF_BIN}" \
    --kernel-trace \
    --stats \
    -S \
    --output-format csv \
    -d "${HIPFIRE_ROCPROF_OUTPUT_DIR}" \
    -o daemon \
    -- "${HIPFIRE_ROCPROF_DAEMON_TARGET}" "$@"
