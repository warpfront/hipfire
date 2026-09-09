#!/usr/bin/env python3
"""Model-swap residency-memo probe for perf/model-load-time audit.

Loads models sequentially in ONE daemon process to exercise the process-global
`model_pages_resident` memo in qwen35/load.rs. Sequence:
  1. load A twice  -> second load sees A resident, memo := true
  2. load B (not pre-warmed) -> memo still says true (stale) if bug present
Compares step-2 timing/trace against a fresh-process load of B where the memo
starts unset and the correct per-file probe runs.
"""
import json
import subprocess
import sys
import threading
import time


def spawn(daemon):
    env = {"HOME": "/tmp/hm-swap", "PATH": "/usr/bin:/bin"}
    p = subprocess.Popen([daemon], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    errs = []
    threading.Thread(target=lambda: [errs.append(l) for l in p.stderr],
                     daemon=True).start()
    return p, errs


def load(p, model, errs):
    t0 = time.perf_counter()
    p.stdin.write(json.dumps(
        {"type": "load", "model": model, "params": {"max_seq": 2048}}) + "\n")
    p.stdin.flush()
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("daemon died: " + "".join(errs[-20:] if errs else []))
        try:
            o = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if o.get("type") == "loaded":
            return time.perf_counter() - t0
        if o.get("type") == "error":
            raise RuntimeError("load error: " + line[:300])


def main():
    daemon = sys.argv[1]
    a = sys.argv[2]   # small model, gets warmed
    b = sys.argv[3]   # big model, never loaded before in this process

    p, errs = spawn(daemon)
    try:
        print(f"swap1  A={a.split('/')[-1]}: {load(p, a, errs)*1000:.0f} ms")
        print(f"swap2  A again (warm):   {load(p, a, errs)*1000:.0f} ms")
        print(f"swap3  B first-in-proc:  {load(p, b, errs)*1000:.0f} ms")
    finally:
        p.terminate()
    # fresh-process control for B
    p2, errs2 = spawn(daemon)
    try:
        print(f"fresh  B new-process:    {load(p2, b, errs2)*1000:.0f} ms")
    finally:
        p2.terminate()


if __name__ == "__main__":
    main()
