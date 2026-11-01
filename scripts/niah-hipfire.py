#!/usr/bin/env python3
"""
NIAH validation for hipfire automatic max_seq (see
docs/plans/local/2026-10-31-windows-legacy-autofit.md).

Fills KV to --budget tokens with non-repetitive filler, hides a needle at an
absolute token position (--needle-at, plus a 0.5 control), decodes, scores
retrieval, and records prefill/decode speed from the serve timings.

Unlike niah-pc.py (llama.cpp) this talks to `hipfire serve`:
- the load omits --max-seq, so the resident n_ctx IS the autofit value under
  test (asserted via /health before any trial);
- token counts come from usage.prompt_tokens (no /tokenize endpoint);
- prefill/decode tok/s come from the response timings object.

Usage:
    python scripts/niah-hipfire.py --model-path C:/Users/Jan/.hipfire/models/qwen3.8-27b.mq4l \
        --expect-n-ctx 152539 --budget 150000 --needle-at 140000 \
        --kv-mode q8 --port 18081 --out niah-q8.json
    python scripts/niah-hipfire.py --model-path ... --expect-n-ctx 262144 \
        --budget 250000 --needle-at 240000 --kv-k fwht4 --kv-v lloyd4 \
        --port 18082 --out niah-fwht4.json
"""

import argparse
import json
import random
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("needs 'requests': pip install requests")

_ADJ = ["quiet", "amber", "hollow", "brisk", "molten", "pale", "restless",
        "granite", "silver", "drowsy", "electric", "humble", "scarlet",
        "tangled", "luminous", "weathered", "nimble", "opaque", "gentle", "vast"]
_NOUN = ["harbor", "ledger", "orchard", "compass", "lantern", "meadow",
         "anchor", "thicket", "mirror", "station", "river", "vault",
         "garden", "signal", "window", "bridge", "candle", "forest",
         "engine", "market"]
_VERB = ["keeps", "carries", "hides", "follows", "opens", "remembers",
         "crosses", "lights", "measures", "guards"]
_TAIL = ["beneath the morning fog.", "along the northern road.",
         "while the bells ring.", "before the tide turns.",
         "under a sky of hammered tin.", "past the sleeping houses.",
         "where the wild thyme grows.", "until the last train leaves.",
         "between two slow rivers.", "as the gulls settle."]


def filler_sentence(rng):
    return (f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} "
            f"{rng.choice(_VERB)} the {rng.choice(_ADJ)} {rng.choice(_NOUN)} "
            f"{rng.choice(_TAIL)}")


class VramPeak:
    """Polls GPU VRAM in background; records peak MiB (copied from niah-pc.py).
    Windows: 'GPU Adapter Memory' performance counter via typeperf."""

    def __init__(self, interval=1.0):
        self.interval, self.peak, self._stop, self._t = interval, 0, False, None

    def _poll(self):
        while not self._stop:
            try:
                if sys.platform != "win32":
                    raise RuntimeError("Windows-only harness")
                if not hasattr(self, "_tmp"):
                    import os, tempfile
                    fd, self._tmp = tempfile.mkstemp(suffix=".csv")
                    os.close(fd)
                fh = open(self._tmp, "w")
                try:
                    subprocess.run(
                        ["typeperf",
                         r"\GPU Adapter Memory(*)\Dedicated Usage",
                         "-sc", "1"],
                        stdout=fh, stderr=subprocess.DEVNULL, timeout=15)
                finally:
                    fh.close()
                used, found = 0, False
                with open(self._tmp) as f:
                    for line in reversed(f.read().splitlines()):
                        for v in line.split(","):
                            try:
                                used = max(used,
                                           int(float(v.strip().strip('"'))
                                               / (1 << 20)))
                                found = True
                            except ValueError:
                                pass
                        if found:
                            break
                self.peak = max(self.peak, used)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        import threading
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=5)
        if getattr(self, "_tmp", None):
            try:
                import os
                os.remove(self._tmp)
            except OSError:
                pass


def wait_ready(session, base, timeout):
    """200 + resident n_ctx (the load resolved) and no load in flight."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = session.get(f"{base}/health", timeout=10)
            if r.status_code == 200:
                h = r.json()
                if h.get("n_ctx") and not h.get("loading_model"):
                    return h
        except Exception:
            pass
        time.sleep(5)
    return None


def chat(session, base, model, text, max_tokens, timeout):
    t0 = time.time()
    r = session.post(
        f"{base}/v1/chat/completions",
        json={"model": model,
              "messages": [{"role": "user", "content": text}],
              "max_tokens": max_tokens, "temperature": 0.0},
        timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    choice = d["choices"][0]["message"]
    return {"answer": (choice.get("content") or "").strip(),
            "reasoning": (choice.get("reasoning_content") or "").strip(),
            "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage", {}),
            "timings": d.get("timings", {}),
            "wall_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--hipfire", default="target/release/hipfire.exe")
    ap.add_argument("--expect-n-ctx", type=int, required=True,
                    help="autofit value under test; asserted via /health n_ctx")
    ap.add_argument("--budget", type=int, required=True,
                    help="total prompt tokens incl. needle, question, headroom")
    ap.add_argument("--needle-at", type=int, required=True,
                    help="absolute token position of the needle")
    ap.add_argument("--control-depth", type=float, default=0.5)
    ap.add_argument("--kv-mode", default=None)
    ap.add_argument("--kv-k", default=None)
    ap.add_argument("--kv-v", default=None)
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--out", default="niah-hipfire.json")
    ap.add_argument("--serve-log", default=None)
    ap.add_argument("--no-stop", action="store_true")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    log_path = a.serve_log or f"serve-niah-{a.port}.log"
    cmd = ["serve", "--model", a.model_path, f"127.0.0.1:{a.port}",
           "--spec", "off"]
    if a.kv_mode:
        cmd += ["--kv-mode", a.kv_mode]
    if a.kv_k:
        cmd += ["--kv-k", a.kv_k]
    if a.kv_v:
        cmd += ["--kv-v", a.kv_v]
    hipfire = a.hipfire
    print("starting:", hipfire, " ".join(cmd), flush=True)
    logfh = open(log_path, "w")
    server = subprocess.Popen([hipfire] + cmd, stdout=logfh,
                              stderr=subprocess.STDOUT)
    session = requests.Session()
    results = {"model_path": a.model_path, "expect_n_ctx": a.expect_n_ctx,
               "budget": a.budget, "needle_at": a.needle_at,
               "kv": {"kv_mode": a.kv_mode, "kv_k": a.kv_k,
                      "kv_v": a.kv_v},
               "trials": []}
    ok_all, vram = True, VramPeak()
    vram.__enter__()
    try:
        health = wait_ready(session, base, a.startup_timeout)
        if health is None or server.poll() is not None:
            sys.exit(f"serve failed to become ready; see {log_path}")
        n_ctx = health.get("n_ctx")
        print(f"resident n_ctx={n_ctx} (expected {a.expect_n_ctx})", flush=True)
        results["resident_n_ctx"] = n_ctx
        if abs(n_ctx - a.expect_n_ctx) / a.expect_n_ctx > 0.03:
            print(f"AUTOFIT MISMATCH: resident {n_ctx} differs from expected "
                  f"{a.expect_n_ctx} by more than 3% (free VRAM moves the "
                  f"measurement)", flush=True)
            ok_all = False
        bound = None
        # The daemon prints the max_seq line after the load completes, which
        # can lag /health (and a slow first load) by minutes; poll rather
        # than reading once, or a healthy load is misread as unmeasured.
        for _ in range(24):
            try:
                with open(log_path) as f:
                    for line in f:
                        if line.startswith("max_seq:"):
                            bound = line.split("bound=", 1)[1].split(";",
                                                                     1)[0]
                            break
            except OSError:
                pass
            if bound is not None:
                break
            time.sleep(5)
        print(f"load bound={bound}", flush=True)
        if bound != "legacy-fit":
            print("EXPECTED BOUND legacy-fit", flush=True)
            ok_all = False
        results["resident_bound"] = bound
        if a.budget >= n_ctx:
            print(f"BUDGET EXCEEDS CTX: budget {a.budget} >= resident {n_ctx}; "
                  f"refusing to send an over-long prompt", flush=True)
            ok_all = False
        models = session.get(f"{base}/v1/models", timeout=30).json()
        model = models["data"][0]["id"]
        print(f"model id: {model}", flush=True)

        secret = f"CORAL-BREEZE-{a.seed:02d}"
        needle = f"The secret password is: {secret}."
        question = ("What is the secret password mentioned in the text? "
                    "Reply with only the password.")

        if ok_all:
            # Two-point calibration: template+question, then +100 sentences.
            t0 = chat(session, base, model, question, 16,
                      a.timeout)["usage"].get("prompt_tokens", 0)
            probe = " ".join(filler_sentence(random.Random(a.seed + i))
                             for i in range(100))
            t1 = chat(session, base, model, probe + "\n\n" + question, 16,
                      a.timeout)["usage"].get("prompt_tokens", 0)
            per_sentence = max((t1 - t0) / 100.0, 1.0)
            print(f"calibration: template+question={t0}tok, "
                  f"per-sentence={per_sentence:.1f}tok", flush=True)
            results["calibration"] = {"template_question_tokens": t0,
                                      "per_sentence_tokens": per_sentence}

            def build(needle_token_pos):
                rng = random.Random(a.seed)
                n_sent = max(int((a.budget - t0 - 300) / per_sentence), 1)
                sentences = [filler_sentence(rng) for _ in range(n_sent)]
                idx = min(max(int(needle_token_pos / per_sentence), 0), n_sent)
                sentences.insert(idx, needle)
                return " ".join(sentences), idx * per_sentence

            for name, pos in [("control", a.budget * a.control_depth),
                              ("target", a.needle_at)]:
                hay, est_pos = build(pos)
                try:
                    res = chat(session, base, model, hay + "\n\n" + question,
                               a.max_tokens, a.timeout)
                    if (res["finish"] == "length"
                            and secret.lower() not in res["answer"].lower()):
                        res = chat(session, base, model,
                                   hay + "\n\n" + question,
                                   a.max_tokens * 4, a.timeout)
                        res["retried_longer"] = True
                    passed = secret.lower() in res["answer"].lower()
                except Exception as e:
                    res = {"error": str(e)}
                    passed = False
                ok_all &= passed
                trial = {"name": name, "needle_token_pos_est": round(est_pos),
                         "passed": passed, **res}
                results["trials"].append(trial)
                u, tm = res.get("usage", {}), res.get("timings", {})
                print(f"{name}: prompt_tok={u.get('prompt_tokens')} "
                      f"fill={((u.get('prompt_tokens') or 0) / n_ctx):.1%} "
                      f"prefill={tm.get('prefill_tok_s')}tok/s "
                      f"decode={tm.get('decode_tok_s')}tok/s "
                      f"ttft={tm.get('ttft_ms')}ms wall={res.get('wall_s')}s "
                      f"finish={res.get('finish')} "
                      f"-> {'PASS' if passed else 'FAIL'} "
                      f"{res.get('answer', res.get('error'))[:80]}", flush=True)
        else:
            print("skipping trials", flush=True)
    finally:
        vram.__exit__()
        if not a.no_stop:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
            logfh.close()
    results["vram_peak_mib"] = vram.peak
    print(f"\npeak VRAM: {vram.peak} MiB", flush=True)
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("max_seq:"):
                    results["serve_max_seq_line"] = line.strip()
                    print("serve:", line.strip(), flush=True)
                    break
    except OSError:
        pass
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
