#!/usr/bin/env python3
import subprocess
import sys

command = [
    "/opt/rocm/bin/rocprofv3",
    "--kernel-trace",
    "--runtime-trace",
    "--minimum-output-data", "0",
    "-f", "csv",
    "-d", "/home/kaden/ClaudeCode/warpfront/wt-iu4pipe/scratch-2026-09-17/Iu4Pipe/slice2/rocprof-candidate",
    "-o", "routed-daemon-full",
    "--",
    "/home/kaden/ClaudeCode/warpfront/wt-iu4pipe/scratch-2026-09-17/Iu4Pipe/slice2/daemon-candidate",
    *sys.argv[1:],
]
process = subprocess.Popen(command)
raise SystemExit(process.wait())
