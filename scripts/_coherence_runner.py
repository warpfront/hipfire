#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
# lifecycle: deprecated since 0.4.0, removal 0.5.0 — retired coherence-gate battery driver; historical reproduction only (docs/VALIDATION.md)

"""_coherence_runner.py — shared driver for coherence-gate-*.sh scripts.

Extracted from scripts/coherence-gate-qwen35-dspark.sh (the inline DETECT_PY
heredoc plus its run loop) so that plain-AR coherence gates (e.g.
coherence-gate-ornith15.sh) don't have to duplicate the token-frequency /
unique-token-ratio attractor detector. The token-frequency and unique-ratio
maths below are kept byte-for-byte equivalent to the original.

Speaks the daemon's JSONL protocol directly: for each --genre/--prompt pair,
writes a load/generate/unload script to the daemon's stdin, captures stdout,
extracts the committed token-id stream (requires HIPFIRE_EMIT_TOKEN_IDS=1,
set here) and the emitted text, and applies the Tier 1/2 hard-fail
thresholds.

This runner does NOT check for any speculator engagement (DSpark, MTP, ...).
Callers that attach a sidecar and need to prove it was actually exercised
(rather than silently falling back to plain AR) must add that check
themselves — see coherence-gate-qwen35-dspark.sh for the DSpark version.

Usage:
    _coherence_runner.py --exe EXE --model MODEL --out OUT.md --timeout SECS \\
        --genre NAME --prompt TEXT [--genre NAME --prompt TEXT ...] \\
        [--max-tokens N] [--max-seq N]

Exit codes: 0 no hard errors (inspect OUT.md for fluency) · 1 hard error(s)
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Ornith 1.0/1.5 pad/eos family (shared tokenizer). Confirmed against
# text_config.{pad_token_id,eos_token_id} in the downloaded config.json.
EOT_IDS = {248044, 248046}


def _iter_events(out_bytes: bytes):
    for line in out_bytes.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def detect(out_bytes: bytes) -> dict:
    """Tier 1/2 hard-fail + Tier 3 soft-warn attractor detector.

    Kept byte-for-byte equivalent to the DETECT_PY heredoc in
    coherence-gate-qwen35-dspark.sh — do not re-derive the thresholds.
    """
    toks = []
    for ev in _iter_events(out_bytes):
        if ev.get("type") == "committed" and "tok_id" in ev:
            toks.append(int(ev["tok_id"]))

    if not toks:
        return {"ok": False, "reason": "no_committed_ids"}

    trimmed = toks
    for i, t in enumerate(toks):
        if t in EOT_IDS:
            trimmed = toks[:i]
            break

    total_all = len(trimmed)

    def check_window(window, label, hard_uniq_lo, hard_freq_hi, soft_uniq_lo, soft_freq_hi):
        if len(window) < 16:
            return {"label": label, "ok": True, "reason": "short_window_ok", "n": len(window)}
        c = collections.Counter(window)
        unique = len(c)
        n = len(window)
        unique_ratio = unique / n
        max_tok, max_count = c.most_common(1)[0]
        max_freq = max_count / n
        hard_fail = max_freq > hard_freq_hi or unique_ratio < hard_uniq_lo
        soft_warn = (max_freq > soft_freq_hi or unique_ratio < soft_uniq_lo) and not hard_fail
        return {"label": label, "ok": not hard_fail, "soft_warn": soft_warn, "n": n,
                "unique": unique, "unique_ratio": round(unique_ratio, 3),
                "max_freq": round(max_freq, 3), "max_tok": max_tok}

    t1 = check_window(trimmed[:128], "tier1_first128", 0.15, 0.50, 0.25, 0.40)

    # Tier 2 inspects the TAIL, on the theory that degeneration sets in late. It
    # only means that when the tail is distinct from the head: below 256 tokens
    # the last-128 window overlaps the first-128 one, so tier 2 re-measures text
    # tier 1 already covered while applying a stricter floor (0.30 vs 0.15).
    #
    # That produced a false failure on a CORRECT answer: the arithmetic prompt
    # returned 144 tokens reading "4 hours 25 minutes" with its working shown
    # twice (once reasoning, once formatted). Restating the working is good
    # style and legitimately depresses token diversity — tier2 scored 0.211.
    #
    # This does NOT weaken degeneracy detection. A model stuck in a repetition
    # loop emits LONG output, so it always clears the 256 floor and is still
    # caught by tier 2; and tier 1 applies at every length regardless.
    TIER2_MIN_TOTAL = 256
    if total_all >= TIER2_MIN_TOTAL:
        t2 = check_window(trimmed[-128:], "tier2_last128", 0.30, 0.50, 0.40, 0.45)
    else:
        t2 = {"label": "tier2_last128", "ok": True, "soft_warn": False,
              "skipped_reason": f"total {total_all} < {TIER2_MIN_TOTAL}; tail not "
                                f"distinct from head, tier1 governs"}

    tier3_flag = False
    if total_all >= 32:
        second_half = trimmed[total_all // 2:]
        if len(second_half) >= 6:
            three_grams = [tuple(second_half[i:i + 3]) for i in range(len(second_half) - 2)]
            if three_grams:
                most_freq_3g = max(collections.Counter(three_grams).values())
                tier3_flag = (most_freq_3g / len(three_grams)) > 0.50

    return {"ok": t1["ok"] and t2["ok"], "total": total_all,
            "tier1": t1, "tier2": t2, "tier3_3gram_flag": tier3_flag}


def extract_text(out_bytes: bytes) -> str:
    text = ""
    for ev in _iter_events(out_bytes):
        # Ornith 1.5 is a thinking model: in think mode the daemon emits
        # "reasoning" events, not "token" ones. Counting only "token" reports
        # zero output for a model that is generating perfectly well, which
        # reads as total failure. Accept both.
        if ev.get("type") in ("token", "reasoning") and "text" in ev:
            text += ev["text"]
    return text


def main() -> int:
    print("warning: _coherence_runner.py is deprecated and will be removed in 0.5.0; retired coherence-gate battery, not acceptance evidence (docs/VALIDATION.md)", file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=420)
    # 200 is too tight for a thinking model: Ornith 1.5 was still mid-reasoning
    # at 220 tokens, and the daemon then rejects the turn with "open think span
    # at end of generation". A truncated think span is a harness artefact, not a
    # coherence failure, so give it room to close.
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--genre", action="append", default=[])
    ap.add_argument("--prompt", action="append", default=[])
    args = ap.parse_args()

    if len(args.genre) != len(args.prompt):
        print("_coherence_runner: --genre and --prompt counts must match", file=sys.stderr)
        return 2
    if not args.genre:
        print("_coherence_runner: no --genre/--prompt pairs given", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    hard_errors = 0
    empty_count = 0
    attractor_count = 0

    # HIPFIRE_AR_GRAPH=0: with AR graph capture enabled, qwen35 decode dies with
    # hipError(906) "hipMemcpy D2H: operation would make the legacy stream depend
    # on a capturing blocking stream". Unrelated to the quant format; disabling
    # capture is the documented workaround until that is fixed separately.
    env = {
        **os.environ,
        "HIPFIRE_EMIT_TOKEN_IDS": "1",
        "HIPFIRE_AR_GRAPH": os.environ.get("HIPFIRE_AR_GRAPH", "0"),
    }

    for genre, prompt in zip(args.genre, args.prompt):
        label = f"ornith15-{genre}"
        script_lines = [
            json.dumps({"type": "load", "model": args.model, "params": {"max_seq": args.max_seq}}),
            # attempt_id is REQUIRED by this daemon's generate contract. Without it
            # the daemon answers {"type":"error","message":"generate missing
            # attempt_id"} and exits 0 — which looks exactly like "the model
            # produced nothing". The reference gate this was lifted from predates
            # that contract.
            json.dumps({"type": "generate", "id": label, "attempt_id": 1, "prompt": prompt,
                        "temperature": 0.0, "max_tokens": args.max_tokens, "repeat_penalty": 1.0}),
            json.dumps({"type": "unload"}),
        ]
        stdin_bytes = ("\n".join(script_lines) + "\n").encode("utf-8")

        print(f"== {label} ==", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [args.exe],
                input=stdin_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                env=env,
            )
            ec = proc.returncode
            out_bytes = proc.stdout
        except subprocess.TimeoutExpired as e:
            ec = 124
            out_bytes = e.stdout or b""
        wall = time.time() - t0

        text_log = out_bytes.decode("utf-8", "replace")
        n_tokens = text_log.count('"type":"token"') + text_log.count('"type":"reasoning"')
        panic_m = re.search(r'.*(panicked|thread .* panicked|FATAL).*', text_log)
        panic = panic_m.group(0).strip() if panic_m else None
        error_ev = '"type":"error"' in text_log

        det = detect(out_bytes)

        row_hard = False
        if ec != 0 or panic or error_ev:
            row_hard = True
        if n_tokens == 0:
            empty_count += 1
            row_hard = True
        if not det.get("ok"):
            attractor_count += 1
            row_hard = True

        status = "OK"
        if row_hard:
            status = "HARD-FAIL"
            hard_errors += 1
        elif det.get("tier3_3gram_flag"):
            status = "OK(tier3-soft-flag)"

        completion = extract_text(out_bytes)

        with out_path.open("a") as f:
            f.write(f"## {genre}\n\n")
            f.write(f"- wall: {wall:.1f}s  status: **{status}**\n")
            f.write(f"- exit: {ec}  n_tokens: {n_tokens}\n")
            f.write(f"- detector: `{json.dumps(det)}`\n")
            if panic:
                f.write(f"- panic: `{panic}`\n")
            f.write("\n**Prompt:**\n\n```\n" + prompt + "\n```\n\n")
            f.write("**Output:**\n\n```\n" + completion + "\n```\n\n")

        print(f"  status={status} wall={wall:.1f}s n_tokens={n_tokens} detector={json.dumps(det)}", flush=True)

    with out_path.open("a") as f:
        f.write("\n")
        f.write(f"- hard_errors: {hard_errors}\n")
        f.write(f"- empty: {empty_count}\n")
        f.write(f"- attractor: {attractor_count}\n")
        if hard_errors > 0:
            f.write(f"\n**{hard_errors} HARD ERROR(S)**\n")
        else:
            f.write("\nno hard errors — review completions above for fluency\n")

    if hard_errors > 0:
        print(f"_coherence_runner: hard_errors={hard_errors} empty={empty_count} attractor={attractor_count} — see {args.out}", file=sys.stderr)
        return 1
    print(f"_coherence_runner: hard_errors=0 empty={empty_count} attractor={attractor_count} — review {args.out} for fluency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
