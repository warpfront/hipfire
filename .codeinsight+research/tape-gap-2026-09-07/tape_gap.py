#!/usr/bin/env python3
"""Retained-tape decode step: per-dispatch GPU-clock spans on Qwen3.8 MQ4-XT."""
import json, os, statistics, subprocess, sys, select
from pathlib import Path

REPO = Path("/home/kaden/ClaudeCode/warpfront/hipfire-beta")
MODEL = "/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt"
OUT = Path("/tmp/beta-val-7b16762b0/tape_gap")
OUT.mkdir(exist_ok=True)
contexts = [int(a) for a in sys.argv[1:]] or [128, 8192]

env = dict(os.environ)
env.update(HIPFIRE_REPLAY_BACKEND="shadow", HIPFIRE_REPLAY_MANUAL_CAPTURE="1", HIPFIRE_CASK_OFF="1",
           HIPFIRE_AR_GRAPH="0", HIPFIRE_GRAPH="0", HIPFIRE_KV_MODE="q8", HIP_VISIBLE_DEVICES="0",
           HIPFIRE_KERNEL_CACHE="/home/kaden/.hipfire_kernels")
log = (OUT / "daemon.log").open("w")
p = subprocess.Popen([str(REPO / "target/release/daemon")], cwd=REPO, env=env, stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)

def req(m, timeout=1800):
    p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush()
    r, _, _ = select.select([p.stdout], [], [], timeout)
    if not r: raise TimeoutError(m["type"])
    line = p.stdout.readline()
    resp = json.loads(line)
    if resp.get("type") == "error": raise RuntimeError(resp)
    return resp

try:
    loaded = req({"type": "load", "model": MODEL, "params": {"max_seq": max(contexts) + 512, "kv_mode": "q8"}})
    print("loaded", loaded.get("arch"), loaded.get("layers"), flush=True)
    for ctx in contexts:
        cap = req({"type": "bench_decode", "context_tokens": ctx, "iterations": 1, "redline_capture": True, "redline_detail": True})
        c = cap["redline_capture"]
        meas = req({"type": "bench_decode", "context_tokens": ctx, "iterations": 32})
        prof = req({"type": "redline_dispatch_profile", "context_tokens": ctx, "warmup_replays": 5, "sample_replays": 20})
        (OUT / f"profile_{ctx}.json").write_text(json.dumps(prof))
        samples = prof["samples"]
        d = prof["dispatches"]
        # samples: list per replay; find span arrays
        s0 = samples[0]
        keys = list(s0.keys()) if isinstance(s0, dict) else None
        print(f"ctx={ctx} launches={c['launches']} kernels={c['unique_kernels']} tok/s={meas['tok_s']:.1f} us/tok={meas['us_per_token']:.0f}")
        print("  sample keys:", keys, "| dispatch keys:", list(d[0].keys()) if d else None)
        print("  first sample:", json.dumps(s0)[:600])
        print("  first dispatch:", json.dumps(d[0])[:400] if d else None)
finally:
    try: req({"type": "unload"}, 60)
    except Exception: pass
    p.terminate(); p.wait(10)
