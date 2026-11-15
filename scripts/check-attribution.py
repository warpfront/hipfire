#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Check commit attribution trailers.

    check-attribution.py --range A..B [--require] [--repo DIR]
    check-attribution.py --self-test [--scratch DIR]

Rules: exactly one well-formed `Agent: <model>/<seat>` or a non-empty
`Human-Authored: <name>`; never both; an agent `Co-authored-by:` signature
without `Agent:` is invalid. `--require` makes one of the two mandatory on
every non-merge commit. Exit 1 on offences, 2 on git/usage error.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

AGENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SIGNATURES = ("claude", "anthropic", "openai", "codex", "copilot", "chatgpt",
              "gpt-", "gemini", "cursor", "devin", "noreply@anthropic.com")
TRAILER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\s*:\s*(.*)$")


class GitError(Exception):
    pass


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise GitError(p.stderr.strip() or "git %s failed" % " ".join(args))
    return p.stdout


def trailers(body):
    """Key/value lines of the final paragraph, only if it is all trailers
    (continuation lines allowed) and is not the sole paragraph (subject)."""
    paras = [p for p in re.split(r"\n\s*\n", body.strip("\n")) if p.strip()]
    if len(paras) < 2:
        return []
    out = []
    for line in paras[-1].split("\n"):
        if line[:1] in (" ", "\t") and out:
            continue
        m = TRAILER_RE.match(line)
        if not m:
            return []
        out.append((m.group(1).lower(), m.group(2).strip()))
    return out


def judge(body, merge, require):
    tr = trailers(body)
    agents = [v for k, v in tr if k == "agent"]
    humans = [v for k, v in tr if k == "human-authored"]
    coauth = [v for k, v in tr if k == "co-authored-by"]
    if agents and humans:
        return "both Agent and Human-Authored"
    if len(agents) > 1:
        return "multiple Agent trailers"
    if len(humans) > 1:
        return "multiple Human-Authored trailers"
    if agents and not AGENT_RE.match(agents[0]):
        return "malformed Agent trailer (want <model>/<seat>): %s" % agents[0]
    if humans and not humans[0]:
        return "empty Human-Authored trailer"
    if not agents:
        for v in coauth:
            if any(s in v.lower() for s in SIGNATURES):
                return "agent Co-authored-by without Agent trailer: %s" % v
    if require and not merge and not agents and not humans:
        return "missing Agent/Human-Authored trailer"
    return None


def check_range(repo, rng, require):
    out = git(repo, "log", "--format=%H%x00%P%x00%B%x01", rng)
    bad = []
    for rec in out.split("\x01"):
        rec = rec.lstrip("\n")
        if not rec:
            continue
        sha, parents, body = rec.split("\x00", 2)
        reason = judge(body, len(parents.split()) > 1, require)
        if reason:
            subject = body.strip().split("\n", 1)[0]
            bad.append("%s %s: %s" % (sha[:12], subject, reason))
    return bad


def self_test(scratch_parent):
    parent = scratch_parent
    os.makedirs(parent, exist_ok=True)
    root = tempfile.mkdtemp(prefix="check-attribution-", dir=parent)
    n = 0
    try:
        git(root, "init", "-q")
        git(root, "config", "user.email", "t@example.invalid")
        git(root, "config", "user.name", "t")
        git(root, "commit", "-q", "--allow-empty", "-m", "base")
        base = git(root, "rev-parse", "HEAD").strip()

        def case(msg, require, want_bad):
            nonlocal n
            git(root, "reset", "-q", "--hard", base)
            git(root, "commit", "-q", "--allow-empty", "-m", msg)
            bad = check_range(root, base + "..HEAD", require)
            n += 1
            assert bool(bad) == want_bad, "case %d %r require=%s: %r" % (n, msg, require, bad)

        case("s\n\nbody\n\nAgent: claude-opus-5.5/PromotionLane", True, False)
        case("s\n\nbody\n\nHuman-Authored: Kaden", True, False)
        case("s\n\nbody", False, False)
        case("s\n\nbody", True, True)
        case("s\n\nbody\n\nAgent: Claude", False, True)
        case("s\n\nbody\n\nAgent: claude-opus-5.5/X\nHuman-Authored: Kaden", False, True)
        case("s\n\nbody\n\nCo-authored-by: Claude <noreply@anthropic.com>", False, True)
        case("s\n\nbody\n\nAgent: claude-opus-5.5/X\nCo-authored-by: Claude <noreply@anthropic.com>",
             True, False)
        case("s\n\nAgent: claude-opus-5.5/X\n\nmore body text", True, True)
        case("s\n\nbody\n\nHuman-Authored:", False, True)
        print("check-attribution: self-test OK (%d checks)" % n)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--range", dest="rng")
    ap.add_argument("--require", action="store_true")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scratch")
    a = ap.parse_args()
    try:
        if a.self_test:
            top = git(a.repo, "rev-parse", "--show-toplevel").strip()
            self_test(a.scratch or os.path.dirname(top))
            return 0
        if not a.rng:
            ap.error("--range or --self-test required")
        bad = check_range(a.repo, a.rng, a.require)
    except GitError as e:
        print("check-attribution: %s" % e, file=sys.stderr)
        return 2
    except AssertionError as e:
        print("check-attribution: self-test FAILED: %s" % e, file=sys.stderr)
        return 1
    for line in bad:
        print(line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
