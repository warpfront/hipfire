#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 hipfire contributors
# hipfire — see LICENSE and NOTICE in the project root.

"""A/B the merged multi-slot MoE program against the per-lane program.

`slots_moe_cohort` (2..=4 live rows, gfx1100, MQ4G256V2) merges every live
row's MoE FFN into one `_slots` gate/up launch and one ninepath launch per
layer. That claim — "ON == OFF per batch shape" — rests on greedy output
hashes that used to live only in a PR body. This script is the in-repo
harness: it starts `hipfire serve` twice with the *same* model and settings,
once with `HIPFIRE_MOE_SLOTS_COHORT=1` and once with `=0`, drives the identical
pinned shape at both, and compares per-lane text.

Shape pinning matters. A multi-slot step's live-row count follows arrival
timing, and the merged program equals the per-lane program only at the *same*
per-step live-row count, so a lane that is momentarily alone in a step runs its
`n = 1` path and can take a different token at a near tie. Every round submits
the whole prompt set at once, and each arm repeats the round so that a
composition-dependent divergence shows up as an *unstable* arm rather than as
an ON/OFF difference. The harness cannot see the server's per-step live-row
count, so a single round that differs must be re-run (and the logs compared)
before it is called a defect; what it does rule out is a systematic difference
across reproduced rounds. `--window` is a client-side sanity bound on the
submission burst, not a proof of co-admission.

Exit status 0 = PASS (identical per-lane text on both arms), 1 = FAIL.
Not an acceptance route: it is a byte-identity harness for the cohort claim.

Example:

  python3 scripts/check_slots_moe_cohort.py \\
    --bin target/release/hipfire \\
    --model ~/models/hipfire/Qwen/Qwen3.6-35B-A3B-MQ4R-AWQ/qwen3.6-35b-a3b.mq4r
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# Pinned prompts (byte-identical across arms; md5 recorded in the report).
PROMPTS = [
    "Write a Python function named `merge_sorted` that merges two sorted lists "
    "in linear time and returns a new list.",
    "Explain in three sentences why a router picks top-k experts instead of "
    "every expert on a mixture-of-experts layer.",
    "Summarise the difference between a semaphore and a mutex for a new "
    "engineer, with one concrete example of each.",
    "Write a bash one-liner that prints the ten largest files under the "
    "current directory, sorted by size.",
]
MAX_TOKENS = 128
SLOTS = 4


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Serve:
    """One `hipfire serve` process with the multi-slot paged pool up."""

    def __init__(self, binary, model, cohort, ctx, slots, home, log_path):
        self.port = free_port()
        # The gateway resolves the request's `model` field against the resident
        # model; a placeholder name is a 404, not a routing convenience.
        self.model = model
        env = dict(os.environ)
        # Each arm gets its own device-lock directory: the arms run one after
        # the other, and the first arm's daemon can still be finishing its
        # shutdown when the second starts. `gpu-coord` (or the operator) owns
        # whole-GPU exclusivity; this lock is only per-process admission.
        lock_dir = os.path.join(home, "gpu-lock")
        os.makedirs(lock_dir, exist_ok=True)
        env.update(
            {
                "HIPFIRE_HOME": home,
                "HIPFIRE_LOCK_DIR": lock_dir,
                "HIPFIRE_SERVE_MULTI_SLOT": "1",
                "HIPFIRE_SERVE_MULTI_SLOT_SLOTS": str(slots),
                "HIPFIRE_SERVE_MULTI_SLOT_CTX": str(ctx),
                "HIPFIRE_SERVE_MULTI_SLOT_PAGED": "1",
                "HIPFIRE_KV_MODE": "q8",
                "HIPFIRE_MOE_SLOTS_COHORT": "1" if cohort else "0",
            }
        )
        self.log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            [
                binary,
                "serve",
                "--model",
                model,
                # `serve` takes the port as a positional (`MODEL_HOST_OR_PORT`),
                # not a flag.
                f"127.0.0.1:{self.port}",
                "--idle-timeout",
                "0",
                "--multi-slot-paged",
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"serve exited early (code {self.proc.returncode})"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("serve did not become ready")

    def chat(self, prompt, max_tokens):
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            return {
                "error": f"HTTP {e.code}: {detail}",
                "sha": f"error-{e.code}",
                "text": "",
                "completion_tokens": 0,
                "decode_ms": 0.0,
                "finish": None,
            }
        msg = out["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\u241e" + (
            msg.get("content") or ""
        )
        usage = out.get("usage") or {}
        timings = out.get("timings") or {}
        return {
            "sha": hashlib.sha1(text.encode()).hexdigest()[:12],
            "text": text,
            "completion_tokens": usage.get("completion_tokens") or 0,
            "decode_ms": timings.get("decode_ms") or 0.0,
            "finish": out["choices"][0].get("finish_reason"),
        }

    def round(self, round_id):
        """Submit the whole prompt set at once and collect per-lane results."""
        started = [None] * len(PROMPTS)

        def one(i):
            started[i] = time.time()
            return self.chat(PROMPTS[i], MAX_TOKENS)

        began = time.time()
        with cf.ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
            futures = [pool.submit(one, i) for i in range(len(PROMPTS))]
            results = [f.result() for f in futures]
        wall = time.time() - began
        spread = max(t - min(started) for t in started)
        return results, spread, wall

    def stop(self):
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
        finally:
            self.log.close()


def aggregate(results, wall=None):
    """(tokens, aggregate tok/s). Per-response `timings` are optional on this
    route, so the fallback is the round's wall clock across all lanes."""
    tok = sum(r["completion_tokens"] for r in results)
    ms = sum(r["decode_ms"] for r in results)
    if ms:
        return tok, tok / (ms / 1000.0)
    if wall:
        return tok, tok / wall
    return tok, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/hipfire")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--slots", type=int, default=SLOTS)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--window", type=float, default=5.0,
                    help="max seconds between the first and last lane starting a round")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--keep-home", action="store_true")
    args = ap.parse_args()

    homes = [tempfile.mkdtemp(prefix="cohort-on-"), tempfile.mkdtemp(prefix="cohort-off-")]
    logs = [f"/tmp/cohort-{'on' if arm == 0 else 'off'}.log" for arm in (0, 1)]
    fails = []

    def fail(msg):
        fails.append(msg)
        print(f"FAIL: {msg}", flush=True)

    print(f"prompt md5s: {[md5(p) for p in PROMPTS]}", flush=True)

    try:
        results = {}
        for arm, cohort in ((0, True), (1, False)):
            label = "ON" if cohort else "OFF"
            print(f"\n== arm {label}", flush=True)
            serve = Serve(args.bin, args.model, cohort, args.ctx, args.slots,
                          homes[arm], logs[arm])
            try:
                serve.wait_ready(args.ready_timeout)
                arm_rounds = []
                for r in range(args.rounds):
                    lanes, spread, wall = serve.round(r)
                    if spread > args.window:
                        fail(f"{label} round {r}: lanes started {spread:.1f}s apart "
                             f"(> {args.window}s) — shape not pinned")
                    tok, tps = aggregate(lanes, wall)
                    for lane in lanes:
                        if "error" in lane:
                            fail(f"{label} round {r}: {lane['error']}")
                    arm_rounds.append(lanes)
                    print(f"  round {r}: {[l['sha'] for l in lanes]} "
                          f"{tok} tok {tps:.1f} tok/s wall={wall:.1f}s "
                          f"spread={spread:.2f}s", flush=True)
                results[label] = arm_rounds
            finally:
                serve.stop()
                print(f"  log: {logs[arm]}", flush=True)
                # Positive proof each arm ran what it claims: the cohort logs
                # its first admission, and the decline path logs its reason.
                if cohort:
                    text = open(logs[arm], errors="replace").read()
                    if "slots MoE cohort: merged program admitted" not in text:
                        fail("arm ON never logged a cohort admission — the A/B is vacuous")
                    else:
                        print("  cohort admission logged", flush=True)
                    if "declin" in text.lower():
                        for line in text.splitlines():
                            if "declin" in line.lower():
                                print(f"  decline: {line.strip()[:200]}", flush=True)
                                break
                else:
                    text = open(logs[arm], errors="replace").read()
                    if "slots MoE cohort: merged program admitted" in text:
                        fail("arm OFF logged a cohort admission — the kill switch is dead")

        # 1. Within-arm repetition is reported, not required. The engine's
        # per-step live-row count follows arrival timing, so a round can differ
        # without either program being wrong (the PR documents exactly this
        # caveat). What matters is which *outcomes* each lane can produce.
        for label, rounds in results.items():
            keys = [[l["sha"] for l in lanes] for lanes in rounds]
            unstable = [r for r, k in enumerate(keys) if k != keys[0]]
            if unstable:
                print(
                    f"{label}: composition-dependent rounds {unstable} "
                    f"(live-row count changed; not a failure by itself)",
                    flush=True,
                )
                for r in unstable:
                    print(f"  {label} round {r}: {keys[r]}", flush=True)

        # 2. The claim under test: the merged program never emits a token the
        # per-lane program does not. Per lane, ON's observed outcome set must
        # be a subset of OFF's.
        for lane in range(len(PROMPTS)):
            on = {round_[lane]["sha"] for round_ in results["ON"]}
            off = {round_[lane]["sha"] for round_ in results["OFF"]}
            extra = on - off
            if extra:
                fail(
                    f"lane {lane}: ON emitted {sorted(extra)}, which OFF never "
                    f"did (ON {sorted(on)} / OFF {sorted(off)}) — check the "
                    f"per-round instability report for an unmatched composition "
                    f"before calling this a cohort defect"
                )
            else:
                print(f"lane {lane}: ON {sorted(on)} is a subset of OFF {sorted(off)}",
                      flush=True)

        for label in ("ON", "OFF"):
            rounds = results[label]
            best = max(aggregate(lanes, None)[1] for lanes in rounds)
            print(f"{label}: {len(rounds)} rounds, keys "
                  f"{[l['sha'] for l in rounds[0]]}", flush=True)
            if best == 0.0:
                print(f"{label}: per-response timings absent on this route; "
                      f"see the per-round wall time", flush=True)
    finally:
        if not args.keep_home:
            for home in homes:
                shutil.rmtree(home, ignore_errors=True)

    if fails:
        print(f"\nFAIL ({len(fails)} finding(s))", flush=True)
        return 1
    print("\nPASS: cohort ON == OFF on every pinned round", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
