#!/usr/bin/env python3
"""Greedy-decode parity check between two hipfire daemon binaries.

Speaks the daemon stdin/stdout protocol directly:
  {"type":"load","model":...,"params":{"max_seq":...}}
  {"type":"generate","id":...,"attempt_id":N,"prompt":...,"temperature":0,"max_tokens":...}

Used by the perf/model-load-time audit to prove the loader rework
(zero-copy mmap uploads, page-cache policy change) produces byte-identical
greedy output vs master. Run under /home/ghazni/gpu-coord gpu-ctl only.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time


def sha(b):
    return hashlib.sha256(b.encode("utf-8", "replace")).hexdigest()[:16]


def run_daemon(daemon_bin, model, prompts, max_tokens, home):
    env = {"HOME": home, "PATH": "/usr/bin:/bin:/usr/local/bin"}
    proc = subprocess.Popen(
        [daemon_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env,
    )
    out = {}
    try:
        proc.stdin.write(json.dumps(
            {"type": "load", "model": model, "params": {"max_seq": 4096}}) + "\n")
        proc.stdin.flush()
        t0 = time.perf_counter()
        while True:
            line = proc.stdout.readline()
            if not line:
                err = proc.stderr.read()
                raise RuntimeError(f"daemon died during load:\n{err[-2000:]}")
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "loaded":
                print(f"  load-to-ready: {(time.perf_counter()-t0)*1000:.0f} ms",
                      file=sys.stderr)
                break
            if obj.get("type") == "error":
                raise RuntimeError(f"load error: {line[:400]}")
        for i, p in enumerate(prompts):
            pid = f"p{i}"
            req = json.dumps({
                "type": "generate", "id": pid, "attempt_id": i + 1, "prompt": p,
                "temperature": 0, "max_tokens": max_tokens})
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            text = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"daemon died during generate {pid}")
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                t = obj.get("type")
                if t == "token" and obj.get("id") == pid:
                    text.append(obj.get("text") or obj.get("token") or "")
                elif t in ("finish", "done", "finished") and obj.get("id") == pid:
                    break
                elif t == "error" and obj.get("id") == pid:
                    text.append(f"<<ERROR: {obj.get('message')}>>")
                    break
            out[p] = "".join(text)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", action="append", required=True)
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = []
    for pf in args.prompt_file:
        with open(pf, "rb") as f:
            raw = f.read()
        print(f"prompt {pf}: md5={hashlib.md5(raw).hexdigest()} bytes={len(raw)}",
              file=sys.stderr)
        prompts.append(raw.decode("utf-8"))

    res = run_daemon(args.daemon, args.model, prompts, args.max_tokens, args.home)
    with open(args.out, "w") as f:
        for p, txt in res.items():
            f.write(f"=== md5(prompt)={sha(p)} ===\n{txt}\n")
            print(f"  output sha256[:16]={sha(txt)} chars={len(txt)}",
                  file=sys.stderr)
    print(args.out)


if __name__ == "__main__":
    main()
