#!/usr/bin/env python3
"""Interleaved model-load A/B with page-residency probes and stderr capture.

Audits perf/model-load-time: runs master/branch daemons alternately (fresh
process + fresh HOME each run), records spawn->loaded wall time, captures
loader stderr traces, and mincore-probes model page residency before/between
runs. Caller must hold the /home/ghazni/gpu-coord lock.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import ctypes
import os


libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_ubyte)]

PROT_READ = 0x1
MAP_PRIVATE = 0x02


def residency_pct(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        addr = libc.mmap(None, ctypes.c_size_t(size),
                         ctypes.c_int(PROT_READ), ctypes.c_int(MAP_PRIVATE),
                         ctypes.c_int(fd), ctypes.c_long(0))
        if addr in (None, ctypes.c_void_p(-1).value):
            return -1.0
        page = 4096
        n = (size + page - 1) // page
        vec = (ctypes.c_ubyte * n)()
        rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        if rc != 0:
            return -1.0
        res = sum(1 for b in vec if b & 1)
        return res * 100.0 / n
    finally:
        os.close(fd)


def one_run(daemon, model, home_dir):
    env = dict(os.environ)
    env["HOME"] = home_dir
    proc = subprocess.Popen(
        [daemon], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env,
    )
    errbuf = []

    def drain():
        for line in proc.stderr:
            errbuf.append(line)

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    try:
        t0 = time.perf_counter()
        proc.stdin.write(json.dumps(
            {"type": "load", "model": model,
             "params": {"max_seq": 4096}}) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon exited before loaded; stderr:\n"
                                   + "".join(errbuf[-30:]))
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "loaded":
                return time.perf_counter() - t0, errbuf
            if obj.get("type") == "error":
                raise RuntimeError("load error: " + line[:400])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def trace_lines(errs):
    return [l.strip() for l in errs
            if "sweep" in l or "load-trace" in l or "warmer" in l
            or "evict" in l.lower()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()

    homes = {"master": "/tmp/hm-master", "branch": "/tmp/hm-branch"}
    results = {"master": [], "branch": []}
    print(f"residency before: {residency_pct(args.model):.0f}%", flush=True)
    for i in range(args.runs):
        for arm, daemon in (("master", args.master), ("branch", args.branch)):
            dt, errs = one_run(daemon, args.model, homes[arm])
            results[arm].append(dt)
            tr = trace_lines(errs)
            print(f"run{i} {arm}: {dt*1000:7.1f} ms  "
                  f"trace={tr[-1] if tr else '-'}", flush=True)
            print(f"      residency after {arm}: "
                  f"{residency_pct(args.model):.0f}%", flush=True)
    for arm in ("master", "branch"):
        xs = [x * 1000 for x in results[arm]]
        print(f"{arm.upper()}: median={statistics.median(xs):.1f} ms "
              f"samples={[round(x, 1) for x in xs]}")


if __name__ == "__main__":
    main()
