#!/usr/bin/env bash
set -euo pipefail
exec /opt/rocm/bin/rocprofv3 --kernel-trace --hip-trace --memory-copy-trace -f csv -d /home/kaden/ClaudeCode/warpfront/wt-ttft/scratch-2026-09-17/TtftGap/rocprof-wrapper -- /home/kaden/ClaudeCode/warpfront/wt-ttft/target/release/daemon "$@"
