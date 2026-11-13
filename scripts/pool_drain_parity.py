#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Buffer-pool drain parity harness (#1049).

An out-of-memory retry returns every pooled buffer to HIP (`GpuPool` drain). The
report behind #1049 is that, on one HIP runtime, every later prefill of the
process then produced junk. This harness forces the production drain at chosen
moments inside a real daemon and requires every output to stay byte-equal to a
control process that ran the identical request sequence with no drain.

Per case (one model + MTP sidecar), two fresh daemon processes run the same
request sequence:

  control  W (warm probes) -> P (fresh-salt probes) -> R (W repeated, resumed)
  drain    W -> `test_pool_oom_drain` (idle drain) -> P with a drain armed
           mid-request (`test_fault_pool_oom_after`, between prefill chunks) -> R

Compared: the decoded text, `finish_reason` and token count of every probe,
byte for byte, greedy and a seeded sampled probe. Asserted on the drain process:
the idle drain returned a non-empty pool; every armed request consumed its
injected out-of-memory and drained again; no probe failed.

Requires a daemon built with feature ``serve-fault-inject``; without it the
``test_pool_*`` commands are unknown. Protocol is stdout JSONL only; stderr is
only scraped for evidence lines (never parsed as protocol).

``--expect-verify-graph`` additionally requires that the MTP narrow-verify
hipGraph was captured before the drain and that graphs kept launching after it,
so the case covers a graph the shared graph registry does not own. That graph
engages only on exact gfx1201 for a MoE model with host-mapped routed experts
(partial offload, e.g. ``--env HIPFIRE_MOE_EXPERT_BUDGET=24`` on a 40-layer A3B),
a Q8 KV cache and context <= 4096 (``--max-seq 4096``, short probes).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_retry_gpu_parity import DaemonSession, HarnessError  # noqa: E402


def doc_prompt(entries: int, seed: int, needles: List[Tuple[int, str]], question: str) -> str:
    rng = random.Random(seed)
    nouns = "harbor lantern orchard compass glacier meadow quarry beacon thicket canyon".split()
    verbs = "carries watches measures repairs follows crosses guards paints counts lifts".split()
    at = dict(needles)
    lines: List[str] = []
    for i in range(entries):
        lines.append(
            f"Entry {i}: the {rng.choice(nouns)} {rng.choice(verbs)} the "
            f"{rng.choice(nouns)} at hour {rng.randint(0, 23)}."
        )
        if i in at:
            lines.append(at[i])
    return "Read this log carefully.\n" + "\n".join(lines) + "\n" + question


def build_probes(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Probe definitions: name, final user prompt, optional prior turns, request extras."""
    probes: List[Dict[str, Any]] = [
        {
            "name": "fact",
            "prompt": "In two sentences, explain how binary search works and state its time complexity.",
            "max_tokens": 64,
        },
        {
            "name": "code",
            "prompt": "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively, then show fib(10).",
            "max_tokens": 64,
        },
        {
            "name": "list",
            "prompt": "List the eight planets of the solar system in order from the Sun, one per line, nothing else.",
            "max_tokens": 64,
        },
        {
            "name": "multiturn",
            "messages": [
                {"role": "user", "content": "My name is Mira and I keep bees."},
                {"role": "assistant", "content": "Nice to meet you, Mira! Beekeeping sounds rewarding."},
            ],
            "prompt": "What is my name and what do I keep? Answer in one sentence.",
            "max_tokens": 64,
        },
        {
            "name": "samp",
            "prompt": "Write a short vivid paragraph about a lighthouse keeper who discovers a message in a bottle.",
            "max_tokens": 96,
            "sampling": {"temperature": 0.8, "top_p": 0.9, "seed": 1234},
        },
        {
            "name": "longdecode",
            "prompt": "Explain, step by step and in plain prose, how a hash map handles collisions with chaining and with open addressing.",
            "max_tokens": 400,
        },
        {
            "name": "doc",
            "prompt": doc_prompt(
                args.doc_entries,
                11,
                [(args.doc_entries // 2, "Entry note: the archive keeper wrote that the secret code word is TANGERINE-4417.")],
                "Question: what is the secret code word? Answer with the code word only.",
            ),
            "max_tokens": 32,
            "expect": ["TANGERINE-4417"],
        },
    ]
    if args.long_entries > 0:
        n = args.long_entries
        probes.append(
            {
                "name": "long",
                "prompt": doc_prompt(
                    n,
                    23,
                    [
                        (40, "Note A: the first vault passphrase is MARMALADE-2208."),
                        (n // 2, "Note B: the second vault passphrase is SPARROW-9931."),
                        (n - 40, "Note C: the third vault passphrase is COBALT-5546."),
                    ],
                    "Question: list the three vault passphrases in order (A, B, C), one per line, nothing else.",
                ),
                "max_tokens": 48,
                "expect": ["MARMALADE-2208", "SPARROW-9931", "COBALT-5546"],
            }
        )
    return probes


def conv_turns() -> List[str]:
    return ["What is the project codename?", "What is the launch year?", "State both facts in one sentence."]


class Case:
    """One daemon process: load, send requests, collect results, stderr evidence."""

    def __init__(self, args: argparse.Namespace, arm: str, out: Path) -> None:
        self.args = args
        self.arm = arm
        self.log_path = out / f"{arm}.stderr.log"
        env = {"HIPFIRE_PREFILL_MAX_BATCH": str(args.prefill_batch)}
        for kv in args.env:
            k, _, v = kv.partition("=")
            env[k] = v
        if arm == "eager":
            # Reference arm: the MTP verify graph off, so every window runs the eager forward.
            env["HIPFIRE_VERIFY_GRAPH"] = "0"
        self.session = DaemonSession(
            Path(args.daemon), timeout_s=args.timeout, env=env, log_path=self.log_path
        )
        self.attempt = 0
        self.results: Dict[str, Dict[str, Any]] = {}
        self.allocs: Dict[str, int] = {}

    # -- protocol ---------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        a = self.args
        params: Dict[str, Any] = {
            "max_seq": a.max_seq,
            "kv_mode": a.kv_mode,
            "kv_backend": a.kv_backend,
            "mtp_mode": "on",
            "mtp": str(a.mtp),
            "dflash_mode": "off",
            "ngram_draft": False,
        }
        self.session.send({"type": "load", "model": str(a.model), "params": params})
        deadline = time.monotonic() + self.session.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HarnessError(f"{self.arm}: timeout waiting for loaded")
            ev = self.session.recv_line(timeout_s=remaining, context=f"{self.arm}.load")
            if ev.get("type") == "error":
                raise HarnessError(f"{self.arm}: load error: {ev}")
            if ev.get("type") == "loaded":
                eff = ev.get("mtp_mode_effective")
                if eff is not None and eff != "on":
                    raise HarnessError(f"{self.arm}: MTP not effective on this load: mtp_mode_effective={eff!r}")
                return ev

    def command(self, ty: str) -> Dict[str, Any]:
        term, _ = self.session.request_until(
            {"type": ty}, terminal_types=(ty, "error"), context=f"{self.arm}.{ty}", auto_commit=False
        )
        if term.get("type") == "error":
            raise HarnessError(
                f"{self.arm}: {ty} failed: {term} (daemon must be built with feature serve-fault-inject)"
            )
        return term

    def stats(self) -> Dict[str, Any]:
        return self.command("test_pool_stats")

    def generate(self, key: str, probe: Dict[str, Any], salt: str, *, arm_after: Optional[int] = None) -> None:
        self.attempt += 1
        prompt = probe["prompt"]
        # A request with `messages` is rendered from that list alone (its last entry
        # is the turn to answer; `prompt` is not appended), so the final user turn
        # goes in the list too.
        history = [dict(m) for m in probe.get("messages", [])]
        messages = history + [{"role": "user", "content": prompt}] if history else []
        if salt:
            if messages:
                messages[0]["content"] = f"[session {salt}] " + messages[0]["content"]
            else:
                prompt = f"[session {salt}] " + prompt
        req: Dict[str, Any] = {
            "type": "generate",
            "id": f"{self.arm}-{key}",
            "attempt_id": self.attempt,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": probe["max_tokens"],
            "max_think_tokens": 1,
            "assistant_prefix": "closed_think",
        }
        if messages:
            req["messages"] = messages
        req.update(probe.get("sampling", {}))
        if arm_after is not None:
            req["test_fault_pool_oom_after"] = arm_after
        before = self.stats()
        text: List[str] = []

        def on_event(ev: Dict[str, Any]) -> None:
            if ev.get("type") == "token":
                text.append(str(ev.get("text") or ""))

        term, _events = self.session.request_until(
            req, terminal_types=("done", "error"), context=f"{self.arm}.{key}", on_event=on_event
        )
        after = self.stats()
        if term.get("type") != "done":
            raise HarnessError(f"{self.arm}: {key}: request failed: {term}")
        self.results[key] = {
            "text": "".join(text),
            "finish_reason": term.get("finish_reason"),
            "tokens": term.get("tokens"),
            "mtp": term.get("mtp"),
            "armed_after": arm_after,
            "graph_launches": after["graph_launches"] - before["graph_launches"],
            "drains": after["drains"] - before["drains"],
            "injected_oom": after["injected_oom"] - before["injected_oom"],
            "pooled_before": before["pooled_bytes"],
            "pooled_after": after["pooled_bytes"],
        }
        self.allocs[key] = (after["pool_new"] + after["pool_reused"]) - (before["pool_new"] + before["pool_reused"])
        print(
            f"[{self.arm}] {key}: {self.results[key]['tokens']} tok, "
            f"allocs={self.allocs[key]}, drains+{self.results[key]['drains']}"
            f"{' armed@' + str(arm_after) if arm_after is not None else ''}: "
            f"{self.results[key]['text'][:90]!r}",
            flush=True,
        )

    def conversation(self, key: str, salt: str, *, arm_after: Optional[int] = None) -> None:
        """Three resumed turns: each appends the previous (deterministic) answer."""
        base = f"[session {salt}] " if salt else ""
        history: List[Dict[str, str]] = []
        user = base + "Remember: the project codename is HELIOTROPE and the launch year is 2031. Reply with just OK."
        for turn, nxt in enumerate([None] + conv_turns()):
            if nxt is not None:
                user = nxt
            probe = {"name": f"{key}{turn}", "prompt": user, "messages": history, "max_tokens": 48}
            self.generate(f"{key}{turn}", probe, "", arm_after=arm_after if turn == 0 else None)
            history = history + [
                {"role": "user", "content": user},
                {"role": "assistant", "content": self.results[f"{key}{turn}"]["text"]},
            ]

    def close(self) -> None:
        self.session.close()

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""


def run_arm(
    args: argparse.Namespace, arm: str, out: Path, probes: List[Dict[str, Any]], calib: Dict[str, int]
) -> Case:
    case = Case(args, arm, out)
    try:
        loaded = case.load()
        print(f"[{arm}] loaded {args.model.name}: kv_backend={loaded.get('kv_backend')} max_seq={loaded.get('max_seq')}", flush=True)
        drain = arm == "drain"
        for p in probes:
            case.generate(f"W:{p['name']}", p, "")
        case.conversation("W:conv", "")
        if drain:
            before_log = len(case.log_text())
            s0 = case.stats()
            ev = case.command("test_pool_oom_drain")
            s1 = case.stats()
            case.results["drain-event"] = {"event": ev, "before": s0, "after": s1, "log_offset": before_log}
            print(f"[{arm}] idle drain: {ev}", flush=True)
        # P: fresh content (cold prefill rebuilds every per-request transient).
        mid = {name for name in ("doc", "long") if any(p["name"] == name for p in probes)}
        for p in probes:
            after = None
            if drain and p["name"] in mid:
                after = max(1, calib.get(f"P:{p['name']}", 2) // 2)
            case.generate(f"P:{p['name']}", p, "post", arm_after=after)
        case.conversation("P:conv", "post")
        # R: the warm prompts again (prefix-cache / checkpoint resume across the drain).
        for p in probes:
            case.generate(f"R:{p['name']}", p, "")
        case.conversation("R:conv", "")
        return case
    finally:
        case.close()


def compare(control: Case, other: Case) -> List[str]:
    problems: List[str] = []
    for key, c in control.results.items():
        d = other.results.get(key)
        if d is None:
            problems.append(f"{key}: missing from the {other.arm} arm")
            continue
        for field in ("text", "finish_reason", "tokens"):
            if c[field] != d[field]:
                problems.append(f"{key}.{field}: control={c[field]!r:.160} {other.arm}={d[field]!r:.160}")
    return problems


def evidence(args: argparse.Namespace, probes: List[Dict[str, Any]], control: Case, drain: Case) -> List[str]:
    problems: List[str] = []
    ev = drain.results["drain-event"]["event"]
    if not ev.get("ok"):
        problems.append(f"idle drain failed: {ev}")
    if ev.get("pooled_before", 0) <= 0:
        problems.append(f"idle drain found an empty pool (nothing to drain): {ev}")
    if ev.get("drains_after", 0) != ev.get("drains_before", 0) + 1:
        problems.append(f"idle drain did not drain exactly once: {ev}")
    if ev.get("pooled_after_alloc", -1) != 0:
        problems.append(f"idle drain left buffers pooled: {ev}")
    for key, r in drain.results.items():
        if key == "drain-event":
            continue
        if r["armed_after"] is not None:
            if r["injected_oom"] != 1:
                problems.append(f"{key}: armed after {r['armed_after']} allocations but the injection fired {r['injected_oom']} times")
            if r["drains"] < 1:
                problems.append(f"{key}: injected out-of-memory did not drain a non-empty pool")
        elif r["injected_oom"] != 0:
            problems.append(f"{key}: unarmed request consumed an injection")
    for case in (control, drain):
        for key, r in case.results.items():
            if key == "drain-event":
                continue
            if len(r["text"].strip()) < 2:
                problems.append(f"{case.arm} {key}: empty or near-empty text")
    if not any(r.get("mtp") is True for k, r in drain.results.items() if k != "drain-event"):
        problems.append("no drain-arm request reported mtp=true: the MTP route did not run")
    for p in probes:
        for want in p.get("expect", []):
            for case in (control, drain):
                for phase in ("W", "P", "R"):
                    text = case.results[f"{phase}:{p['name']}"]["text"]
                    if want not in text:
                        problems.append(f"{case.arm} {phase}:{p['name']}: expected {want!r} in {text!r:.120}")
    if args.expect_verify_graph:
        log = drain.log_text()
        off = drain.results["drain-event"]["log_offset"]
        if "[mtp-verify-graph] captured for n=" not in log[:off]:
            problems.append("MTP verify graph was not captured before the drain (expected on gfx1201 + MoE + ctx <= 4096)")
        post = [r for k, r in drain.results.items() if k.startswith(("P:", "R:")) and not k.startswith(("P:long", "R:long", "P:doc", "R:doc"))]
        if not any(r["graph_launches"] > 0 for r in post):
            problems.append("no hipGraph launch after the drain")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daemon", required=True, type=Path, help="daemon built with --features serve-fault-inject")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--mtp", required=True, type=Path, help="MTP sidecar (.mtp)")
    ap.add_argument("--label", default="case")
    ap.add_argument("--out", type=Path, default=None, help="result/log directory (default: a temp dir under $PWD)")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--kv-mode", default="q8")
    ap.add_argument("--kv-backend", default="vmm")
    ap.add_argument("--prefill-batch", type=int, default=2048)
    ap.add_argument("--doc-entries", type=int, default=380, help="entries in the document probe (~4.5K tokens)")
    ap.add_argument("--long-entries", type=int, default=1500, help="entries in the long probe (~28K tokens); 0 disables")
    ap.add_argument("--expect-verify-graph", action="store_true", help="require the MTP verify graph to engage (MoE, ctx <= 4096)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--env", action="append", default=[], help="K=V for the daemon process")
    ap.add_argument(
        "--eager-reference",
        action="store_true",
        help="also run an arm with the MTP verify graph off (HIPFIRE_VERIFY_GRAPH=0) and require the graph-engaged control to match it",
    )
    args = ap.parse_args()

    out = args.out or Path(f"pool-drain-parity-{args.label}-{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)
    probes = build_probes(args)

    print(f"== control ({args.label})", flush=True)
    control = run_arm(args, "control", out, probes, {})
    calib = dict(control.allocs)
    print(f"== drain ({args.label}); allocations per probe in the control: "
          f"{ {k: v for k, v in calib.items() if k.startswith('P:') and k.split(':')[1] in ('doc', 'long')} }", flush=True)
    drain = run_arm(args, "drain", out, probes, calib)
    problems = compare(control, drain) + evidence(args, probes, control, drain)
    eager: Optional[Case] = None
    if args.eager_reference:
        print(f"== eager reference ({args.label})", flush=True)
        eager = run_arm(args, "eager", out, probes, {})
        problems += compare(control, eager)
        if "[mtp-verify-graph] captured" in eager.log_text():
            problems.append("eager reference arm captured an MTP verify graph")
    marker = "Can't find memory handle"
    summary = {
        "label": args.label,
        "daemon": str(args.daemon),
        "model": str(args.model),
        "mtp": str(args.mtp),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "problems": problems,
        "drain_event": drain.results["drain-event"]["event"],
        "mid_request": {
            k: {f: r[f] for f in ("armed_after", "injected_oom", "drains", "pooled_before", "pooled_after")}
            for k, r in drain.results.items()
            if k != "drain-event" and r["armed_after"] is not None
        },
        "gpu_pool_lines": [ln for ln in drain.log_text().splitlines() if ln.startswith("GpuPool:")],
        "verify_graph_lines": [ln for ln in drain.log_text().splitlines() if "[mtp-verify-graph]" in ln],
        "memory_handle_lines": {"control": marker in control.log_text(), "drain": marker in drain.log_text()},
        "control": control.results,
        "eager": eager.results if eager is not None else None,
        "drain": {k: v for k, v in drain.results.items() if k != "drain-event"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if problems:
        print(f"FAIL {args.label}: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    n = sum(1 for k in control.results if k != "drain-event")
    print(f"PASS {args.label}: {n} probes byte-equal across the forced drains; results in {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as exc:
        print(f"HARNESS ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
