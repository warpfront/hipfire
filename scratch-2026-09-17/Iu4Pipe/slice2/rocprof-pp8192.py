#!/usr/bin/env python3
import os
import subprocess
import sys

root = "/home/kaden/ClaudeCode/warpfront/wt-iu4pipe/scratch-2026-09-17/Iu4Pipe/slice2"
arm = os.environ["IU4_TRACE_ARM"]
command = [
    "/opt/rocm/bin/rocprofv3",
    "--kernel-trace",
    "--minimum-output-data", "0",
    "-f", "csv",
    "-d", f"{root}/rocprof-pp8192",
    "-o", f"pp8192-{arm}",
    "--",
    f"{root}/daemon-{arm}",
    *sys.argv[1:],
]
process = subprocess.Popen(command)
raise SystemExit(process.wait())
