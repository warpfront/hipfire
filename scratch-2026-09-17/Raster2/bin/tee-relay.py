#!/usr/bin/env python3
"""Tee relay: log first N bytes each direction, then relay forever."""
import faulthandler
import os
import subprocess
import sys
import threading

_DBG = os.environ.get("RELAY_DEBUG_FILE")
if _DBG:
    faulthandler.dump_traceback_later(25.0, file=open(_DBG, "w"))

LOG = sys.argv[1]
CAP = 8192


def pump(src, dst, tag):
    n = 0
    try:
        while True:
            chunk = src.read(65536)
            if not chunk:
                break
            if n < CAP:
                with open(LOG, "ab") as f:
                    f.write(b"@@@" + tag + b" " + chunk[:CAP] + b"\n")
                n += len(chunk)
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
    dargs = sys.argv[2:]
    proc = subprocess.Popen(
        dargs,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr.buffer,
        start_new_session=True,
    )
    t1 = threading.Thread(
        target=pump, args=(sys.stdin.buffer, proc.stdin, b"C2D"), daemon=True
    )
    t2 = threading.Thread(
        target=pump, args=(proc.stdout, sys.stdout.buffer, b"D2C"), daemon=True
    )
    t1.start()
    t2.start()
    sys.exit(proc.wait())


main()
