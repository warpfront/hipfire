#!/usr/bin/env bash
set -euo pipefail
: "${HIPFIRE_ROCPROF_DAEMON_TARGET:?set daemon target}"
: "${HIPFIRE_ROCPROF_OUTPUT_DIR:?set output directory}"
mkdir -p "${HIPFIRE_ROCPROF_OUTPUT_DIR}"
exec /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --stats \
    -S \
    --output-format csv \
    -d "${HIPFIRE_ROCPROF_OUTPUT_DIR}" \
    -o daemon \
    -- "${HIPFIRE_ROCPROF_DAEMON_TARGET}" "$@"
