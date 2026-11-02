#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Multi-slot embedding acceptance: slot mode must reproduce sequential mode.

Encoding is deterministic (`encode_embeddings_batch` resets the GDN state per
segment and the daemon holds one weight copy), so the two routes are compared
value-for-value, not statistically: the same texts, one run through the
ordinary daemon and one through `experimental_multi_slot`, must agree to f32
print precision. The batched forward is the only thing that can break this —
a segment reading another segment's rows, or its delta-rule state, produces
plausible vectors — so the check is exact on purpose and the tolerance is a
canary (`0.0` unless a rounding change lands), not a quality budget.

Also reports wall-clock for both routes: the slot route batches whatever
arrives together, so N concurrent requests cost less than N serial ones even
though per-request work is unchanged.

Usage:
  python3 scripts/check_embed_multislot.py --model <dir-or-hfq> [--slots 4]
  python3 scripts/check_embed_multislot.py --model <dir> --oracle   # + NumPy leg
"""

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

PROMPTS_DIR = "benchmarks/prompts"

# Fixed pool: a sub-256 short input (Q-tiled kernel, no padding), two over it
# (eager kernel with a padded tail), and two committed fixtures. Every request
# is a distinct length, so a batch's segments never line up with its own row
# blocks.
FIXED = [
    ("query", "what is the capital of Peru"),
    ("document", "word " * 300),
    ("query", "explain the difference between a process and a thread"),
    ("document", "token " * 256),
    ("query", "hipfire"),
]


class Daemon:
    """One daemon process, multiplexed by request id (the wire is JSONL)."""

    def __init__(self, binary, model, params):
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain stderr concurrently: a daemon that logs during encodes would
        # otherwise fill the pipe and block itself mid-request, and this
        # check would hang for the wrong reason.
        self._stderr_buf = []
        def _drain_stderr():
            for line in self.proc.stderr:
                self._stderr_buf.append(line)
        threading.Thread(target=_drain_stderr, daemon=True).start()
        self._wlock = threading.Lock()
        self._pending = {}
        self._plock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.send({"type": "load", "model": model, "params": params})
        loaded = self.recv("loaded")
        if loaded is None:
            raise SystemExit("load failed")
        self.arch = loaded.get("arch")
        self.loaded = loaded

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = d.get("id")
            with self._plock:
                q = self._pending.get(rid)
            if q is not None:
                q.put(d)

    def send(self, obj):
        with self._wlock:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def recv(self, want, rid=None, timeout=600.0):
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            d = q.get(timeout=timeout)
        finally:
            with self._plock:
                self._pending.pop(rid, None)
        if d.get("type") == "error":
            raise RuntimeError(f"daemon error: {d.get('message')}")
        if want and d.get("type") != want:
            raise RuntimeError(f"expected {want}, got {d.get('type')}")
        return d

    def embed(self, rid, kind, text):
        """One synchronous encode. Registers the pending entry before sending,
        so the reply cannot race the registration."""
        q = queue.Queue()
        with self._plock:
            self._pending[rid] = q
        try:
            self.send(
                {"type": "embed", "id": rid, "kind": kind, "input": {"text": text}}
            )
            d = q.get(timeout=600.0)
        finally:
            with self._plock:
                self._pending.pop(rid, None)
        if d.get("type") != "embedded":
            raise RuntimeError(f"embed {rid}: {d}")
        return d

    def stop(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        err = "".join(self._stderr_buf)
        return err


def texts_from_prompts(names):
    out = []
    for n in names:
        path = os.path.join(PROMPTS_DIR, f"{n}.txt")
        with open(path) as f:
            out.append(("document", f.read()))
    return out


def compare(a, b):
    """(max abs diff, relative l2) between two embedding matrices."""
    if len(a) != len(b):
        return (float("inf"), float("inf"), f"length {len(a)} vs {len(b)}")
    mx = 0.0
    num = 0.0
    den = 0.0
    for x, y in zip(a, b):
        mx = max(mx, abs(x - y))
        num += (x - y) ** 2
        den += y * y
    return (mx, (num / den) ** 0.5 if den else 0.0, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", default="target/release/daemon")
    ap.add_argument("--model", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--oracle", action="store_true", help="also compare against the NumPy oracle")
    ap.add_argument("--prompts", nargs="*", default=["embed_short", "embed_medium"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases = FIXED + texts_from_prompts(args.prompts)
    n = len(cases)
    print(f"{n} inputs: {[len(t.split()) for _, t in cases]} words")

    # ── sequential: the ordinary daemon, one request at a time ─────────
    seq = Daemon(args.daemon, args.model, {})
    print(f"sequential daemon: arch {seq.arch}")
    seq_out = []
    t0 = time.perf_counter()
    for i, (kind, text) in enumerate(cases):
        d = seq.embed(f"seq{i}", kind, text)
        seq_out.append(d)
    t_seq = time.perf_counter() - t0
    seq.stop()

    # ── slot mode: the same inputs, all in flight at once ──────────────
    slot = Daemon(
        args.daemon,
        args.model,
        {
            "experimental_multi_slot": True,
            "experimental_multi_slot_slots": args.slots,
            "max_batch_tokens": args.batch_tokens,
            "max_seq": 8192,
        },
    )
    print(f"slot daemon: arch {slot.arch} (experimental_multi_slot)")
    # The whole point of this check is the batched forward — if a regression
    # made the load silently fall back to the sequential path, "slot" embeds
    # would run the same code as the reference leg and be trivially
    # identical. Assert the daemon really is in slot mode.
    if not slot.loaded.get("experimental_multi_slot"):
        print("FAIL: daemon did not enter experimental_multi_slot")
        return 1
    slot_out = [None] * n
    errors = [None] * n
    lats = [0.0] * n

    def worker(i):
        kind, text = cases[i]
        t = time.perf_counter()
        try:
            slot_out[i] = slot.embed(f"slot{i}", kind, text)
        except Exception as e:  # noqa: BLE001 — reported per input below
            errors[i] = str(e)
        lats[i] = time.perf_counter() - t

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_slot = time.perf_counter() - t0
    err = slot.stop()

    ok = True
    # Keyed by case index — inputs that errored are skipped earlier, so a
    # positional list would shift every later row and the oracle leg below
    # would attach its metrics to the wrong input.
    report = {}
    for i, (kind, text) in enumerate(cases):
        if errors[i]:
            ok = False
            print(f"  [{i}] ERROR {errors[i]}")
            continue
        a, b = seq_out[i], slot_out[i]
        if a.get("n_tokens") != b.get("n_tokens") or a.get("dim") != b.get("dim"):
            print(
                f"  [{i}] SHAPE {b.get('n_tokens')}x{b.get('dim')} != "
                f"{a.get('n_tokens')}x{a.get('dim')}"
            )
            ok = False
            continue
        mx, rel, why = compare(b["embeddings"], a["embeddings"])
        if why or mx != 0.0:
            ok = False
        report[i] = {
            "index": i,
            "kind": kind,
            "n_input_tokens": b.get("n_input_tokens"),
            "n_tokens": b.get("n_tokens"),
            "max_abs_diff": mx,
            "rel_l2": rel,
            "note": why,
        }
        flag = "ok " if (mx == 0.0 and not why) else "BAD"
        print(
            f"  [{i}] {flag} {kind:8s} in {b.get('n_input_tokens'):5d} kept "
            f"{b.get('n_tokens'):5d} | max {mx:.3e} rel {rel:.3e} | slot {lats[i] * 1e3:7.1f} ms"
        )

    print(
        f"\nserial {t_seq * 1e3:.1f} ms | slot-mode wall (all {n} concurrent) "
        f"{t_slot * 1e3:.1f} ms ({t_seq / t_slot:.2f}x) | "
        f"median slot latency {statistics.median(lats) * 1e3:.1f} ms"
    )
    if err.strip():
        print("daemon stderr tail:")
        print("\n".join(err.strip().splitlines()[-10:]))

    if args.oracle:
        sys.path.insert(0, "scripts")
        os.environ["ORACLE_MODEL"] = args.model
        try:
            import numpy as np  # noqa: PLC0415
            from oracle_pplx_embed import encode as oracle_encode  # noqa: PLC0415
            from tokenizers import Tokenizer  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"--oracle needs scripts/oracle_pplx_embed.py's deps: {e}")
        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        markers = {"query": tok.token_to_id("[Q] "), "document": tok.token_to_id("[D] ")}
        # Same host-side modules the daemon applies after the Dense head: the
        # document side drops the punctation-piece rows, both sides
        # L2-normalize. Mirrors post::skiplist_token_ids / skiplist_mask.
        skiplist = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        print("\nNumPy oracle leg (independent of the daemon's own trunk):")

        def kept(kind, ids):
            if kind != "document":
                return [True] * len(ids)
            out = []
            for i, t in enumerate(ids):
                piece = (tok.id_to_token(t) or "").lstrip("\u0120\u010a")
                out.append(i == 0 or t == markers[kind] or piece not in skiplist)
            return out

        oracle_pass = True
        # f64 NumPy vs the daemon's f16-weight GPU trunk: not bit equality —
        # a band. This is the only leg independent of the daemon's own
        # forward, so divergence here gates the run, not just prints.
        ORACLE_REL_L2 = 1e-2
        for i, (kind, text) in enumerate(cases):
            if slot_out[i] is None:
                continue
            ids = [markers[kind]] + tok.encode(text).ids
            n_in = slot_out[i].get("n_input_tokens")
            if n_in is None or n_in > len(ids):
                # Missing or larger than the prompt's tokenization is a
                # protocol/encode break, not a truncation.
                print(f"  [{i}] DIFF: daemon n_input_tokens={n_in} vs oracle {len(ids)}")
                ok = False
                oracle_pass = False
                continue
            if n_in < len(ids):
                # The daemon truncated at the kind limit; compare the kept
                # prefix, which is exactly what the oracle must reproduce.
                ids = ids[:n_in]
            e = oracle_encode(np.array(ids))
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            mask = kept(kind, ids)
            if not all(mask):
                e = e[mask]
            flat = [float(v) for row in e for v in row]
            mx, rel, why = compare(slot_out[i]["embeddings"], flat)
            flag = "ok " if rel <= ORACLE_REL_L2 and not why else "DIFF"
            print(f"  [{i}] {flag} oracle vs slot: max {mx:.3e} rel {rel:.3e} {why}")
            if rel > ORACLE_REL_L2 or why:
                ok = False
                oracle_pass = False
            if i in report:
                report[i]["oracle_max_abs_diff"] = mx
                report[i]["oracle_rel_l2"] = rel
        print(f"  oracle leg: {'PASS' if oracle_pass else 'FAIL'} (band rel_l2 <= {ORACLE_REL_L2})")

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "slots": args.slots,
                    "inputs": [report[i] for i in sorted(report)],
                    "serial_ms": round(t_seq * 1e3, 3),
                    "slot_wall_ms": round(t_slot * 1e3, 3),
                    "speedup": round(t_seq / t_slot, 3),
                    "pass": ok,
                },
                indent=2,
            )
        )

    print("\nPASS: slot mode reproduces the sequential route" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
