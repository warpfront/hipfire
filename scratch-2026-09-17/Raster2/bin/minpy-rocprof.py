#!/usr/bin/env python3
"""Minimal rocprof relay: fd pass-through (no threads), rocprofv3 execs daemon.
Usage: minpy-rocprof.py <outdir> <oname> <daemon> [daemon-args...]
"""
import subprocess
import sys

outdir, oname = sys.argv[1], sys.argv[2]
dargs = sys.argv[3:]
p = subprocess.Popen(
    [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--minimum-output-data",
        "0",
        "-f",
        "csv",
        "-d",
        outdir,
        "-o",
        oname,
        "--",
    ]
    + dargs,
    stdin=sys.stdin.buffer,
    stdout=sys.stdout.buffer,
)
sys.exit(p.wait())
