#!/usr/bin/env python3
"""Double-fork launcher: detach the traced relay so the CLI's Engine-drop
SIGKILL (which targets the direct child) misses it. CLI pipes (fd 0/1) stay
open via the grandchild's inherited fds, so no EOF is observed.
Usage: det-launch.py <relay.py> <outdir> <oname> <daemon> [daemon-args...]
"""
import os
import sys

relay, outdir, oname = sys.argv[1], sys.argv[2], sys.argv[3]
dargs = sys.argv[4:]

pid = os.fork()
if pid != 0:
    os._exit(0)
os.setsid()
pid2 = os.fork()
if pid2 != 0:
    os._exit(0)
os.execvp(
    "/usr/bin/python3",
    ["python3", relay, outdir, oname] + dargs,
)
