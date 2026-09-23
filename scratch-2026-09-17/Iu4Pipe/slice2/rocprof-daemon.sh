#!/bin/sh
/opt/rocm/bin/rocprofv3 --kernel-trace --minimum-output-data 0 -f csv -d /home/kaden/ClaudeCode/warpfront/wt-iu4pipe/scratch-2026-09-17/Iu4Pipe/slice2/rocprof-candidate -o routed-daemon -- /home/kaden/ClaudeCode/warpfront/wt-iu4pipe/scratch-2026-09-17/Iu4Pipe/slice2/daemon-candidate "$@" <&0 >&1 2>&2 &
child=$!
wait "$child"
