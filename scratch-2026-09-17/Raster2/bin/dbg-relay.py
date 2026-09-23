#!/usr/bin/env python3
"""Relay with step-by-step unbuffered debug log."""
import os
import subprocess
import sys
import threading

DLOG = sys.argv[1]
OUTDIR, ONAME = sys.argv[2], sys.argv[3]
DARGS = sys.argv[4:]


def log(msg):
    with open(DLOG, "ab", buffering=0) as f:
        f.write((str(os.getpid()) + "/" + threading.current_thread().name + " " + msg + "\n").encode())


def pump(src, dst, tag):
    log(tag + " start")
    try:
        while True:
            log(tag + " read.enter")
            chunk = src.read(65536)
            log(tag + " read.got %d" % len(chunk))
            if not chunk:
                break
            log(tag + " write.enter %d" % len(chunk))
            dst.write(chunk)
            dst.flush()
            log(tag + " write.done")
    except Exception as e:
        log(tag + " EXC %r" % e)
    finally:
        log(tag + " finally")
        try:
            dst.close()
        except Exception:
            pass


def main():
    log("main popen")
    proc = subprocess.Popen(
        [
            "/opt/rocm/bin/rocprofv3",
            "--kernel-trace",
            "--minimum-output-data",
            "0",
            "-f",
            "csv",
            "-d",
            OUTDIR,
            "-o",
            ONAME,
            "--",
        ]
        + DARGS,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr.buffer,
        start_new_session=True,
    )
    log("main spawned")
    t1 = threading.Thread(target=pump, args=(sys.stdin.buffer, proc.stdin, "C2D"))
    t2 = threading.Thread(target=pump, args=(proc.stdout, sys.stdout.buffer, "D2C"))
    t1.start()
    t2.start()
    log("main wait")
    sys.exit(proc.wait())


main()
