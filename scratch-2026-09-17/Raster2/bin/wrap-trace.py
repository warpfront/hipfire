#!/usr/bin/env python3
"""Stdio relay: CLI <-> rocprofv3-traced daemon.

Runs rocprofv3 in FOREGROUND (dash backgrounds get /dev/null stdin) while
relaying the CLI's stdio pipes through threads. When the CLI SIGKILLs this
wrapper at bench end, the traced daemon is orphaned (pipes EOF -> it usually
exits, which finalizes rocprofv3 and flushes the CSV). If it lingers,
SIGTERM rocprofv3 manually.
Usage: wrap-trace.py <outdir> <oname> <daemon> [daemon-args...]
"""
import subprocess
import sys
import threading


def pump(src, dst):
    try:
        while True:
            chunk = src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()
    except Exception:
        pass
    finally:
        try:
            dst.close()
        except Exception:
            pass


def main():
    outdir, oname = sys.argv[1], sys.argv[2]
    dargs = sys.argv[3:]
    proc = subprocess.Popen(
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
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr.buffer,
        start_new_session=True,
    )
    t1 = threading.Thread(
        target=pump, args=(sys.stdin.buffer, proc.stdin), daemon=True
    )
    t2 = threading.Thread(
        target=pump, args=(proc.stdout, sys.stdout.buffer), daemon=True
    )
    t1.start()
    t2.start()
    sys.exit(proc.wait())


main()
