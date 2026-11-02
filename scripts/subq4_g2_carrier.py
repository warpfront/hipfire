#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Sub-4-bit (subq4) G2 run carrier: the real-model Flash-Next gate on ctl3-hot.

One fabric-run.sh submission (class correct, card C, 30 GB device, 44 GB pinned
host memory, hostmem lock, P1, max 7200 s) runs four child checks serially and
writes `<rcpt>/g2.json` plus per-check logs and reports under `<rcpt>/g2/`:

  fn_lane_ar         fn_lane_oracle, AR lanes, ks 1,2,4,8, contexts 128,256,
                     steps 8, max-seq 1024, --controls
  fn_lane_mtp        the same matrix with --spec mtp --k 3 --routes cycle
                     --controls and HIPFIRE_MTP_INCREMENTAL=0
  parity_state       qwen4_parity --mode state on the teacher-forced tokens
  replay             qwen4_g2_replay_gate --context 128 --iterations 5
                     --max-seq 1024 --expert-vram-layers N (CLI, no env)

N is `--expert-vram-layers` (integer, default 16): the lane and parity checks run
with HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N, the replay gate gets it on its CLI, the
replay report's params must echo it, and g2.json records it. The scheduler
declaration (30 GB device, 44 GB host) is the same for every N. Pass 15 only if
16 still runs out of device memory; the whole ks 1,2,4,8 matrix always runs.

Nothing here is implicit:
  * The scheduler is mandatory: FABRIC_RUN is /home/kaden/qcal/locks/fabric/
    fabric-run.sh. fabric.py has no no-fallback flag (it silently falls back to
    in-process admission when fabricd is down or FABRIC_NO_DAEMON=1), so the
    carrier enforces it: it refuses unless `fabric-run.sh --status --json` is
    answered by a live fabricd (that verb exits non-zero instead of falling
    back), refuses when FABRIC_NO_DAEMON/FABRIC_SANDBOX is set, and, inside the
    job, requires fabricd to list this very wrapper (FABRIC_JOB pid) as a running
    job with a `jN` id and the declared card/memory/lock/priority/owner. A
    fallback launch has no such job, so it is refused before any GPU child runs.
  * `--smoke-receipt PATH` is required; its JSON must record a PASS on the real
    ctl3-hot artifact (explicit pass/passed/ok/success true or status/verdict
    "pass"; any failing or unrecognised verdict refuses the run).
  * `--code-sha SHA` must be a prefix of `git rev-parse HEAD`, run in the G2
    worktree this script lives in (never the invoking cwd), on a clean tracked
    tree.
  * The card is validated against the known UUID/BDF map, the KFD topology and
    the scheduler's own card record; inside the job the fabric-pinned
    environment must name the same card.
  * Every child needs rc 0, a report parsed and judged strictly, zero
    mismatches on the positive runs (detected negative controls are not
    mismatches), and, for MTP, drafts, accepted drafts and rejected windows all
    exercised (an unexercised MTP run is a FAIL).

Stale outputs are refused (an existing g2.json or child output would otherwise
read as a pass). The first mismatch of any failing check is kept in g2.json with
the exact command, environment and output directory to reproduce it.

Usage (outer; submits itself to the scheduler):
  scripts/subq4_g2_carrier.py --smoke-receipt RCPT.json --code-sha SHA [--card C]
                              [--expert-vram-layers 16]
Exit 0 only when the scheduler job and every check pass, or pass with the replay gate reporting SKIP
(the loaded model is not on the product's default Redline route): g2.json status is then SKIP, never PASS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import glob
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "subq4-g2-run-v1"

REPO = Path(__file__).resolve().parent.parent
G2_WORKTREE = Path("/home/kaden/qcal/release-0.4.3/subq4-g2")
FORBIDDEN_WORKDIR = Path("/home/kaden/ClaudeCode/warpfront/hipfire-beta")
WEEK_RCPT = Path("/home/kaden/qcal/release-0.4.3/subq4-week/rcpt")
FABRIC_RUN = "/home/kaden/qcal/locks/fabric/fabric-run.sh"
KFD_TOPOLOGY = "/sys/class/kfd/kfd/topology/nodes"
HOST = "hiptrx"

ARTIFACT = Path("/home/kaden/qcal/release-0.4.3/subq4-week/artifacts/ctl3-hot/ctl3-hot.hfq")
ARTIFACT_SHA256 = "e8fb196deed493a1c1d6a825f44e934465135bb85d6325c38c2e022de33b9674"
# 113 GB: identified by size and the R2 pack receipt, never re-hashed here.
ARTIFACT_BYTES = 113653405208
PACK_RECEIPT = "r2.pack.json"

# label -> (ROCr UUID, PCI address or None when this carrier has no pinned BDF).
CARDS: dict[str, tuple[str, str | None]] = {
    "A": ("GPU-9eb7aeda51c88ffd", "0000:03:00.0"),
    "B": ("GPU-e475645fe0200397", None),
    "C": ("GPU-085289909a86cc63", "0000:e3:00.0"),
    "E": ("GPU-05f92432f2312a0e", None),
}

SCHED = {
    "class": "correct",
    "mem_gb": 30,
    "hostmem_gb": 44,
    "hostmem_lock": True,
    "priority": "P1",
    "owner": "Subq4WeekLead-2.G2Harness",
    "max_secs": 7200,
}
GIB = 1 << 30
FABRIC_JOB_ID = re.compile(r"j[0-9]+")
# fabric.py has no no-fallback flag: FABRIC_NO_DAEMON=1 forces its in-process
# admission and FABRIC_SANDBOX moves it to a test sandbox. Either set means the
# submission would not go through the real fabricd.
FABRIC_BYPASS_ENV = ("FABRIC_NO_DAEMON", "FABRIC_SANDBOX")
# Seconds of the scheduler's cap reserved for receipt writing and teardown.
DEADLINE_RESERVE_SECS = 180

DEFAULT_EXPERT_LAYERS = 16
ENV_EXPERT = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"
ENV_MTP = "HIPFIRE_MTP_INCREMENTAL"
# Pinned by the carrier per child; anything inherited under these names is dropped.
SCRUBBED_ENV = (ENV_EXPERT, ENV_MTP)
REPLAY_SCRUBBED_ENV = ("HIPFIRE_REPLAY_BACKEND", "HIPFIRE_REPLAY_MANUAL_CAPTURE", "HIPFIRE_REPLAY_TRANSPORT")

LANE_KS = [1, 2, 4, 8]
LANE_CONTEXTS = [128, 256]
LANE_STEPS = 8
MAX_SEQ = 1024
MTP_K = 3
MTP_ROUTES = "cycle"
REPLAY_FIXED = {"context": 128, "iterations": 5, "max_seq": 1024}
REPLAY_SKIP_REASON = "Redline not on the default route (config.rs:15–22)"

AR_CONTROLS = {"lane_swap_tokens", "lane_swap_row_slot", "lane_swap_rows", "wrong_position"}
REPLAY_ARMS = ("production", "plan", "hip_oracle")
REPLAY_REQUIRED = (
    "capture_window_armed_before",
    "capture_executed",
    "capture_window_effects_replayable",
    "capture_route_identities_latched",
    "capture_prepared_dispatches_equal_launches",
    "capture_forward_matches_hip_baseline",
    "hip_baseline_logits_finite_nonzero",
    "hip_baseline_ran_recorded_body",
    "rebuild_kept_prepared_plan",
    "negative_refusal_poisons_route",
    "negative_recovers_on_hip",
    "artifact_is_expert_slab",
    "replay_default_configured",
) + tuple(
    f"{phase}_{arm}_{what}"
    for arm in REPLAY_ARMS
    for phase, what in (
        ("positive", "executed"),
        ("positive", "route_stays_ready"),
        ("positive", "logits_exact"),
        ("positive", "gdn_exact"),
        ("positive", "qsa_exact"),
        ("positive", "ple_exact"),
        ("positive", "lengths_exact"),
        ("positive", "position_exact"),
        ("positive", "argmax_exact"),
        ("negative", "refused"),
        ("negative", "no_retained_execution"),
    )
)

POS_WORDS = {"pass", "passed", "ok", "success", "succeeded"}
BOOL_VERDICT_KEYS = ("pass", "passed", "ok", "success")
STR_VERDICT_KEYS = ("status", "verdict", "result", "outcome")
SMOKE_SUB_SCOPES = ("smoke", "summary", "result")


class Refusal(Exception):
    """A precondition failed: nothing was launched."""


# ── small helpers ────────────────────────────────────────────────────────


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: Path, *, hash_it: bool = True) -> dict[str, Any]:
    st = path.stat()
    out: dict[str, Any] = {"path": str(path), "bytes": st.st_size, "mtime_utc": dt.datetime.fromtimestamp(
        st.st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if hash_it:
        out["sha256"] = sha256_file(path)
    return out


def write_json_atomic(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
    return res.stdout.strip()


def tail_lines(path: Path, n: int = 40) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


# ── code identity ────────────────────────────────────────────────────────


def verify_code_sha(arg: str, repo: Path = REPO, expected_worktree: Path = G2_WORKTREE) -> dict[str, Any]:
    """`arg` must be a (>= 7 hex digit) prefix of HEAD of the G2 worktree.

    HEAD comes from `git -C <worktree> rev-parse HEAD`; the worktree is the one
    this script lives in and must be the G2 worktree, never the caller's cwd.
    """
    repo = repo.resolve()
    if repo == FORBIDDEN_WORKDIR.resolve() or FORBIDDEN_WORKDIR.resolve() in repo.parents:
        raise Refusal(f"carrier lives in the forbidden workdir {repo}")
    if repo != expected_worktree.resolve():
        raise Refusal(f"carrier worktree {repo} is not the G2 worktree {expected_worktree}")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", arg):
        raise Refusal(f"--code-sha {arg!r} is not 7-40 hex digits")
    top = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    if top != repo:
        raise Refusal(f"git toplevel {top} differs from the carrier worktree {repo}")
    head = git(repo, "rev-parse", "HEAD").lower()
    if not head.startswith(arg.lower()):
        raise Refusal(f"--code-sha {arg} does not match G2 worktree HEAD {head}")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise Refusal("G2 worktree has uncommitted tracked changes, so the sha does not name the code:\n" + dirty)
    return {
        "worktree": str(repo),
        "head": head,
        "code_sha_arg": arg.lower(),
        "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_clean": True,
    }


# ── smoke receipt ────────────────────────────────────────────────────────


def smoke_verdict(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(positive evidence, negative evidence) from the receipt's verdict fields.

    Top-level and `smoke`/`summary`/`result` sub-object fields count: a boolean
    pass/passed/ok/success, or a status/verdict/result/outcome string. A string
    outside the pass words is negative evidence, so an unknown verdict refuses.
    """
    pos: list[str] = []
    neg: list[str] = []
    scopes: list[tuple[str, dict[str, Any]]] = [("", doc)]
    scopes += [(f"{k}.", doc[k]) for k in SMOKE_SUB_SCOPES if isinstance(doc.get(k), dict)]
    for prefix, scope in scopes:
        for key in BOOL_VERDICT_KEYS:
            if key in scope:
                val = scope[key]
                if val is True:
                    pos.append(f"{prefix}{key}=true")
                else:
                    neg.append(f"{prefix}{key}={val!r}")
        for key in STR_VERDICT_KEYS:
            if key in scope and not isinstance(scope[key], dict):
                val = scope[key]
                if isinstance(val, str) and val.strip().lower() in POS_WORDS:
                    pos.append(f"{prefix}{key}={val}")
                else:
                    neg.append(f"{prefix}{key}={val!r}")
    return pos, neg


def evaluate_smoke_receipt(
    doc: Any, raw_text: str, artifact_sha: str = ARTIFACT_SHA256, artifact_path: Path = ARTIFACT
) -> dict[str, Any]:
    """Judge the RouteOwner4 real-artifact smoke receipt; raises Refusal."""
    if not isinstance(doc, dict):
        raise Refusal("smoke receipt is not a JSON object")
    pos, neg = smoke_verdict(doc)
    if neg:
        raise Refusal("smoke receipt does not record a clean PASS: " + "; ".join(neg))
    if not pos:
        raise Refusal(
            "smoke receipt records no explicit verdict (need pass/passed/ok/success true or "
            "status/verdict 'pass'); a PASS is never implied"
        )
    for key in ("synthetic", "mock", "fixture_only"):
        if doc.get(key):
            raise Refusal(f"smoke receipt is marked {key}, not a real-artifact smoke")
    lowered = raw_text.lower()
    bound = [
        what
        for what, needle in (
            ("artifact sha256", artifact_sha.lower()),
            ("artifact path", str(artifact_path).lower()),
        )
        if needle in lowered
    ]
    if not bound:
        raise Refusal(
            f"smoke receipt names neither the ctl3-hot artifact sha256 {artifact_sha} nor its path {artifact_path}"
        )
    return {"verdict_evidence": pos, "bound_to_artifact_by": bound, "receipt_code_sha": doc.get("code_sha")}


def load_smoke_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Refusal(f"--smoke-receipt {path} is not a file")
    raw = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise Refusal(f"smoke receipt {path} is not valid JSON: {exc}") from exc
    verdict = evaluate_smoke_receipt(doc, raw)
    return {**file_identity(path), **verdict}


# ── card identity ────────────────────────────────────────────────────────


def gfx_name(version: int) -> str:
    return f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"


def kfd_gpus(root: str = KFD_TOPOLOGY) -> list[dict[str, Any]]:
    gpus = []
    for node in sorted(glob.glob(root + "/*"), key=lambda p: int(p.rsplit("/", 1)[1])):
        try:
            gpu_id = int(Path(node, "gpu_id").read_text().strip())
            raw = Path(node, "properties").read_text()
        except (OSError, ValueError):
            continue
        if gpu_id == 0:
            continue
        props = dict(line.split() for line in raw.splitlines() if len(line.split()) == 2)
        loc = int(props.get("location_id", 0))
        uid = int(props.get("unique_id", 0))
        gpus.append(
            {
                "bdf": f"{int(props.get('domain', 0)):04x}:{loc >> 8:02x}:{(loc >> 3) & 31:02x}.{loc & 7}",
                "uuid": f"GPU-{uid:016x}" if uid else None,
                "arch": gfx_name(int(props.get("gfx_target_version", 0))),
            }
        )
    return gpus


def fabric_status(card: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """fabricd's status object from `fabric-run.sh --status [--card X] --json`, or (None, why).

    With --json fabric.py exits 1 when fabricd is unreachable (it never shows its
    in-process view then), so a parsed `ok` reply from a daemon whose pid is alive
    proves a live fabricd answered.
    """
    argv = [FABRIC_RUN, "--status", *(["--card", card] if card else []), "--json"]
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"fabric status failed: {exc}"
    if res.returncode != 0:
        return None, f"fabric status rc {res.returncode}: {(res.stderr or res.stdout).strip()[:300]}"
    try:
        reply = json.loads(res.stdout)
    except ValueError as exc:
        return None, f"fabric status is not JSON: {exc}"
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        return None, f"fabric status reply is not ok: {str(reply)[:300]}"
    daemon = reply.get("daemon")
    pid = daemon.get("pid") if isinstance(daemon, dict) else None
    if not isinstance(pid, int) or isinstance(pid, bool) or not Path(f"/proc/{pid}").is_dir():
        return None, f"fabric status names no live fabricd process (daemon={daemon!r})"
    return reply, None


def scheduler_card_record(card: str) -> tuple[dict[str, Any] | None, str | None]:
    """The live fabricd's record of `card`, or (None, why): None means no live scheduler."""
    reply, why = fabric_status(card)
    if reply is None:
        return None, why
    for rec in reply.get("cards") or []:
        if isinstance(rec, dict) and rec.get("name") == card:
            return rec, None
    return None, f"fabric status lists no card {card}"


def refuse_fabric_bypass(env: dict[str, str] | os._Environ[str] = os.environ) -> None:
    set_vars = [f"{k}={env[k]!r}" for k in FABRIC_BYPASS_ENV if env.get(k)]
    if set_vars:
        raise Refusal("fabric scheduler bypass in the environment (" + ", ".join(set_vars) + "): "
                      "fabric.py would not use the real fabricd; unset it")


def validate_card(
    card: str,
    gpus: list[dict[str, Any]],
    sched: dict[str, Any] | None,
    sched_why: str | None,
) -> dict[str, Any]:
    if card not in CARDS:
        raise Refusal(f"--card {card} is not a known card ({', '.join(CARDS)})")
    if sched is None:
        raise Refusal(f"no live fabricd for card {card} ({sched_why}); fabric.py would fall back to in-process "
                      "admission, which this carrier never uses: start fabricd (systemctl --user start fabricd)")
    uuid, known_bdf = CARDS[card]
    matches = [g for g in gpus if g["uuid"] == uuid]
    if len(matches) != 1:
        raise Refusal(f"card {card} ({uuid}) is not present exactly once in the KFD topology: {matches}")
    gpu = matches[0]
    if known_bdf and gpu["bdf"] != known_bdf:
        raise Refusal(f"card {card} {uuid} is at {gpu['bdf']}, expected {known_bdf}")
    out: dict[str, Any] = {"label": card, "uuid": uuid, "bdf": gpu["bdf"], "arch": gpu["arch"], "bdf_pinned": bool(known_bdf)}
    devenv = sched.get("devenv") or {}
    if sched.get("bdf") != gpu["bdf"]:
        raise Refusal(f"scheduler places card {card} at {sched.get('bdf')}, KFD at {gpu['bdf']}")
    pinned = {devenv.get("HIPFIRE_DEVICES"), devenv.get("ROCR_VISIBLE_DEVICES")}
    if pinned != {uuid}:
        raise Refusal(f"scheduler selects {sorted(map(str, pinned))} for card {card}, expected {uuid}")
    out["scheduler"] = {"status": "matched", "bdf": sched.get("bdf"), "devenv": devenv}
    return out


def check_fabric_env(card: str, env: dict[str, str] | os._Environ[str] = os.environ) -> dict[str, str]:
    """Inside the job: the scheduler's pinned selectors must name `card`."""
    uuid = CARDS[card][0]
    want = {"FABRIC_CARDS": card, "HIPFIRE_DEVICES": uuid, "ROCR_VISIBLE_DEVICES": uuid}
    got = {k: env.get(k, "") for k in want}
    if not env.get("FABRIC_JOB"):
        raise Refusal("not running inside a fabric job (FABRIC_JOB is unset): submit through the outer carrier")
    for key, value in want.items():
        if got[key] != value:
            raise Refusal(f"fabric pinned {key}={got[key]!r}, expected {value!r}")
    return {**got, "FABRIC_JOB": env["FABRIC_JOB"]}


def check_fabric_job(
    card: str, env: dict[str, str] | os._Environ[str], reply: dict[str, Any]
) -> dict[str, Any]:
    """Inside the job: fabricd must list this wrapper as the running `jN` job we declared.

    FABRIC_JOB (`<wrapper pid>:<start>:<cards>`) is set by the in-process fallback
    too, so it proves nothing about fabricd. Only a daemon job record whose pid is
    that wrapper and whose id is jN does; a fallback launch has none.
    """
    m = re.fullmatch(r"([0-9]+):[0-9]+:(.+)", env.get("FABRIC_JOB", ""))
    if not m:
        raise Refusal(f"FABRIC_JOB={env.get('FABRIC_JOB')!r} is not <pid>:<start>:<cards>")
    wrapper_pid, cards = int(m.group(1)), m.group(2)
    if cards != card:
        raise Refusal(f"FABRIC_JOB names cards {cards!r}, expected {card!r}")
    jobs = [j for j in reply.get("jobs") or [] if isinstance(j, dict) and j.get("pid") == wrapper_pid]
    if len(jobs) != 1:
        raise Refusal(f"fabricd lists {len(jobs)} jobs for wrapper pid {wrapper_pid} (fabric id {env.get('FABRIC_JOB')}): "
                      "this is a fallback (in-process) launch, not a scheduler grant")
    job = jobs[0]
    if not (isinstance(job.get("id"), str) and FABRIC_JOB_ID.fullmatch(job["id"])):
        raise Refusal(f"fabricd job id {job.get('id')!r} does not match jN")
    want = {
        "state": "running", "cls": SCHED["class"], "cards": [card], "mem": SCHED["mem_gb"] * GIB,
        "hostmem": SCHED["hostmem_gb"] * GIB, "hostmem_lock": SCHED["hostmem_lock"],
        "priority": int(SCHED["priority"][1:]), "owner": SCHED["owner"], "max_secs": SCHED["max_secs"],
    }
    for key, value in want.items():
        if job.get(key) != value:
            raise Refusal(f"fabricd job {job['id']} has {key}={job.get(key)!r}, carrier declared {value!r}")
    daemon = reply.get("daemon") or {}
    return {"id": job["id"], "wrapper_pid": wrapper_pid,
            **{k: job.get(k) for k in ("state", "cls", "cards", "mem", "hostmem", "hostmem_lock", "priority",
                                       "owner", "max_secs", "enq", "admitted", "cmd")},
            "fabricd": {k: daemon.get(k) for k in ("pid", "host", "version", "started", "drain")}}


# ── report judgment ──────────────────────────────────────────────────────


def _new_verdict() -> dict[str, Any]:
    return {"verdict": "PASS", "classification": None, "reasons": [], "mismatches": 0, "first_mismatch": None}


CLASS_PRIORITY = (
    "timeout", "crash", "mismatch", "unexercised", "control_missed", "report_missing", "report_invalid", "harness_fail",
)


def _more_specific(old: str | None, new: str | None) -> str | None:
    if new is None:
        return old
    if old is None or CLASS_PRIORITY.index(new) < CLASS_PRIORITY.index(old):
        return new
    return old


def _fail(v: dict[str, Any], classification: str, reason: str) -> None:
    v["verdict"] = "FAIL"
    # A measured difference outranks the exit code that reports it; reasons all append.
    v["classification"] = _more_specific(v["classification"], classification)
    v["reasons"].append(reason)


def _run_problem(run: dict[str, Any]) -> list[str]:
    problems = []
    if run.get("pass") is not True:
        problems.append("run.pass is not true")
    if run.get("differences") != 0:
        problems.append(f"differences={run.get('differences')!r}")
    if run.get("executor_error") is not None:
        problems.append(f"executor_error={run.get('executor_error')!r}")
    if run.get("release_errors"):
        problems.append(f"release_errors={run.get('release_errors')!r}")
    if run.get("provision_refusal") is not None:
        problems.append(f"provision_refusal={run.get('provision_refusal')!r}")
    if not (run.get("items_compared") or 0) > 0:
        problems.append("items_compared is 0")
    if not (run.get("state_checkpoints_compared") or 0) > 0:
        problems.append("state_checkpoints_compared is 0")
    return problems


def _mismatch_repro(run: dict[str, Any], source: str) -> dict[str, Any]:
    keys = (
        "k", "spec", "spec_k", "routes", "policy", "control", "steps", "promote", "differences",
        "first_divergence", "first_divergences", "executor_error", "release_errors", "provision_refusal",
    )
    return {"source": source, **{k: run[k] for k in keys if k in run}}


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "k", "spec", "spec_k", "routes", "control", "pass", "differences", "items_compared", "bytes_compared",
        "state_checkpoints_compared", "lane_steps_compared", "windows", "rejected_windows", "drafts", "accepted",
        "windows_by_route", "wall_s",
    )
    return {k: run[k] for k in keys if k in run}


def _check_lane_matrix(v: dict[str, Any], doc: dict[str, Any], *, spec: str) -> list[dict[str, Any]]:
    """Common AR/MTP judgment of `runs[]`; returns the positive runs seen."""
    args = doc.get("args") or {}
    want = {"ks": LANE_KS, "contexts": LANE_CONTEXTS, "steps": LANE_STEPS, "max_seq": MAX_SEQ,
            "controls": True, "g0": False, "spec": spec}
    if spec == "mtp":
        want.update({"k": MTP_K, "routes": MTP_ROUTES, "promote": None})
    for key, value in want.items():
        if args.get(key) != value:
            _fail(v, "harness_fail", f"report args.{key}={args.get(key)!r}, expected {value!r}")
    if doc.get("pass") is not True:
        _fail(v, "harness_fail", "summary pass is not true")
    runs = [r for r in doc.get("runs") or [] if isinstance(r, dict)]
    positive = [r for r in runs if r.get("control") == "none"]
    seen = sorted(k for k in (r.get("k") for r in positive) if isinstance(k, int))
    if seen != sorted(LANE_KS):
        _fail(v, "report_invalid", f"positive runs cover k={seen}, expected {sorted(LANE_KS)}")
    for run in positive:
        v["mismatches"] += run.get("differences") if isinstance(run.get("differences"), int) else 0
        problems = _run_problem(run)
        if spec == "mtp":
            for key, value in (("spec", "mtp"), ("spec_k", MTP_K), ("routes", MTP_ROUTES)):
                if run.get(key) != value:
                    problems.append(f"{key}={run.get(key)!r}, expected {value!r}")
        if problems:
            _fail(v, "mismatch" if (run.get("differences") or 0) > 0 else "harness_fail",
                  f"k={run.get('k')}: " + ", ".join(problems))
            if v["first_mismatch"] is None:
                v["first_mismatch"] = _mismatch_repro(run, "runs")
    return positive


def evaluate_lane_ar(doc: Any) -> dict[str, Any]:
    v = _new_verdict()
    if not isinstance(doc, dict):
        _fail(v, "report_invalid", "summary.json is not an object")
        return v
    positive = _check_lane_matrix(v, doc, spec="ar")
    if doc.get("s1_prefill_parity_k2") is not True:
        _fail(v, "harness_fail", f"s1_prefill_parity_k2={doc.get('s1_prefill_parity_k2')!r}")
    controls = [c for c in doc.get("controls") or [] if isinstance(c, dict)]
    names = {c.get("control") for c in controls}
    if names != AR_CONTROLS or len(controls) != len(AR_CONTROLS):
        _fail(v, "report_invalid", f"controls {sorted(map(str, names))} != {sorted(AR_CONTROLS)}")
    for c in controls:
        if c.get("detected") is not True or c.get("clean_after_refusal") is not True:
            _fail(v, "control_missed",
                  f"control {c.get('control')}: detected={c.get('detected')!r} how={c.get('how')!r} "
                  f"clean_after_refusal={c.get('clean_after_refusal')!r}")
    v["runs"] = [_compact_run(r) for r in positive]
    v["controls"] = [{"control": c.get("control"), "detected": c.get("detected"), "how": c.get("how")} for c in controls]
    return v


def evaluate_lane_mtp(doc: Any) -> dict[str, Any]:
    v = _new_verdict()
    if not isinstance(doc, dict):
        _fail(v, "report_invalid", "summary.json is not an object")
        return v
    positive = _check_lane_matrix(v, doc, spec="mtp")
    controls = [c for c in doc.get("controls") or [] if isinstance(c, dict)]
    by_name = {c.get("control"): c for c in controls}
    if set(by_name) != {"spec_positive_cycle", "spec_skip_rollback"} or len(controls) != 2:
        _fail(v, "report_invalid", f"controls {sorted(map(str, by_name))} != spec_positive_cycle, spec_skip_rollback")
    pos_ctl = by_name.get("spec_positive_cycle")
    if pos_ctl is not None:
        run = pos_ctl.get("run") if isinstance(pos_ctl.get("run"), dict) else {}
        diff = run.get("differences")
        v["mismatches"] += diff if isinstance(diff, int) else 0
        if pos_ctl.get("matched") is not True or pos_ctl.get("exercised") is not True or diff != 0:
            _fail(v, "mismatch" if (diff or 0) > 0 else "unexercised",
                  f"spec_positive_cycle matched={pos_ctl.get('matched')!r} exercised={pos_ctl.get('exercised')!r} "
                  f"differences={diff!r} rejected_windows={pos_ctl.get('rejected_windows')!r}")
            if (diff or 0) > 0 and v["first_mismatch"] is None:
                v["first_mismatch"] = _mismatch_repro(run, "controls.spec_positive_cycle")
    neg_ctl = by_name.get("spec_skip_rollback")
    if neg_ctl is not None and neg_ctl.get("detected") is not True:
        _fail(v, "control_missed", f"spec_skip_rollback detected={neg_ctl.get('detected')!r} how={neg_ctl.get('how')!r}")
    coverage = {
        key: sum(r.get(key) for r in positive if isinstance(r.get(key), int))
        for key in ("drafts", "accepted", "rejected_windows", "windows")
    }
    v["coverage"] = coverage
    unexercised = [k for k in ("drafts", "accepted", "rejected_windows") if not coverage[k] > 0]
    if unexercised:
        _fail(v, "unexercised", "MTP not exercised on the positive runs: no " + ", ".join(unexercised)
              + f" (coverage {coverage})")
    v["runs"] = [_compact_run(r) for r in positive]
    v["controls"] = [{k: c.get(k) for k in ("control", "matched", "detected", "exercised", "rejected_windows", "how")
                      if k in c} for c in controls]
    return v


def _walk_status(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every object whose `status` is not "pass", depth first."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        if "status" in node and node["status"] != "pass":
            found.append((path or "$", node))
        for key, val in node.items():
            found += _walk_status(val, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            found += _walk_status(val, f"{path}[{i}]")
    return found


def _walk_mismatch_counts(node: Any, path: str = "") -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    if isinstance(node, dict):
        for key, val in node.items():
            sub = f"{path}.{key}" if path else key
            if re.fullmatch(r"(?i)(mismatch(es|_count)?|differences)", key) and isinstance(val, int) \
                    and not isinstance(val, bool) and val > 0:
                found.append((sub, val))
            found += _walk_mismatch_counts(val, sub)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            found += _walk_mismatch_counts(val, f"{path}[{i}]")
    return found


def evaluate_parity_state(doc: Any) -> dict[str, Any]:
    v = _new_verdict()
    if not isinstance(doc, dict):
        _fail(v, "report_invalid", "state parity report is not an object")
        return v
    if doc.get("schema") != "hipfire.qwen4.state_parity.v1":
        _fail(v, "report_invalid", f"schema={doc.get('schema')!r}")
    for path, node in (("status", doc), ("real_model.status", doc.get("real_model")),
                       ("compact_state.status", doc.get("compact_state"))):
        status = node.get("status") if isinstance(node, dict) else None
        if status != "pass":
            _fail(v, "harness_fail", f"{path}={status!r}")
    counts = _walk_mismatch_counts(doc)
    if counts:
        v["mismatches"] = sum(n for _, n in counts)
        _fail(v, "mismatch", "nonzero mismatch counts: " + ", ".join(f"{p}={n}" for p, n in counts[:8]))
    if v["verdict"] != "PASS":
        failing = _walk_status(doc)
        if failing:
            path, node = failing[0]
            v["first_mismatch"] = {"source": "status", "path": path, "node": json.dumps(node, default=str)[:4000]}
        elif counts:
            v["first_mismatch"] = {"source": "mismatch_count", "path": counts[0][0], "count": counts[0][1]}
    mtp = (doc.get("real_model") or {}).get("ar_vs_native_mtp") if isinstance(doc.get("real_model"), dict) else None
    v["observed"] = {"gpu_arch": doc.get("gpu_arch"), "ar_vs_native_mtp": mtp}
    return v


def evaluate_replay(doc: Any, expert_layers: int) -> dict[str, Any]:
    v = _new_verdict()
    if not isinstance(doc, dict):
        _fail(v, "report_invalid", "replay report is not an object")
        return v
    if doc.get("gate") != "qwen4_g2_replay_gate":
        _fail(v, "report_invalid", f"gate={doc.get('gate')!r}")
    if doc.get("ok") is not True:
        _fail(v, "harness_fail", "report ok is not true")
    if doc.get("failed_checks") != []:
        _fail(v, "harness_fail", f"failed_checks={doc.get('failed_checks')!r}")
    params = replay_params(expert_layers)
    if doc.get("params") != params:
        _fail(v, "report_invalid", f"params={doc.get('params')!r}, expected {params}")
    if doc.get("status") == "skip":
        # The gate loaded the model and found Redline off the product's default route: it ran no
        # capture/positive/negative phase. Accepted only as exactly that report; never a PASS.
        route = doc.get("route")
        if doc.get("skipped") is not True or doc.get("reason") != REPLAY_SKIP_REASON:
            _fail(v, "report_invalid", f"skip report reason={doc.get('reason')!r} skipped={doc.get('skipped')!r}")
        if doc.get("checks") != []:
            _fail(v, "report_invalid", "a skip report must carry no check rows")
        if not isinstance(route, dict) or route.get("default_route") is not False:
            _fail(v, "report_invalid", f"skip report route={route!r} is not a non-default route")
        if v["verdict"] == "PASS":
            v["verdict"] = "SKIP"
            v["reasons"].append(REPLAY_SKIP_REASON)
        v["observed"] = {"arch": doc.get("arch"), "route": route, "elapsed_s": doc.get("elapsed_s")}
        return v
    checks = [c for c in doc.get("checks") or [] if isinstance(c, dict)]
    if not checks:
        _fail(v, "report_invalid", "report has no checks")
    names = {c.get("name") for c in checks}
    missing = [n for n in REPLAY_REQUIRED if n not in names]
    if missing:
        _fail(v, "report_invalid", f"required checks absent: {missing}")
    for c in checks:
        count = c.get("mismatch_count")
        if isinstance(count, int) and not isinstance(count, bool):
            v["mismatches"] += count
        if c.get("pass") is not True or count != 0 or c.get("error") is not None:
            _fail(v, "mismatch" if isinstance(count, int) and count > 0 else "harness_fail",
                  f"check {c.get('name')}: pass={c.get('pass')!r} mismatch_count={count!r} error={c.get('error')!r}")
            if v["first_mismatch"] is None:
                v["first_mismatch"] = {k: c.get(k) for k in ("name", "pass", "mismatch_count", "error", "detail")}
    v["observed"] = {"arch": doc.get("arch"), "checks": len(checks), "elapsed_s": doc.get("elapsed_s")}
    return v


# ── children ─────────────────────────────────────────────────────────────


def replay_params(expert_layers: int) -> dict[str, Any]:
    return {**REPLAY_FIXED, "expert_vram_layers": expert_layers}


def child_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    bins = Path(args.bin_dir)
    g2 = Path(args.rcpt_dir) / "g2"
    layers = str(args.expert_vram_layers)
    lane_common = [
        str(ARTIFACT), str(args.text), "",  # OUT_DIR filled per check
        "--ks", ",".join(map(str, LANE_KS)), "--contexts", ",".join(map(str, LANE_CONTEXTS)),
        "--steps", str(LANE_STEPS), "--max-seq", str(MAX_SEQ), "--controls",
    ]

    def lane(name: str, extra: list[str], env: dict[str, str]) -> dict[str, Any]:
        out = g2 / name
        argv = [str(bins / "fn_lane_oracle")] + [str(out) if a == "" else a for a in lane_common] + extra
        return {"name": name, "argv": argv, "set_env": env, "drop_env": list(SCRUBBED_ENV), "out_dir": out,
                "report": out / "summary.json", "evaluate": evaluate_lane_mtp if "mtp" in extra else evaluate_lane_ar}

    return [
        lane("fn_lane_ar", ["--spec", "ar"], {ENV_EXPERT: layers}),
        lane("fn_lane_mtp", ["--spec", "mtp", "--k", str(MTP_K), "--routes", MTP_ROUTES],
             {ENV_EXPERT: layers, ENV_MTP: "0"}),
        {
            "name": "parity_state",
            "argv": [str(bins / "qwen4_parity"), "--mode", "state", "--model", str(ARTIFACT),
                     "--tokens", str(args.tokens), "--out", str(g2 / "parity_state.report.json")],
            "set_env": {ENV_EXPERT: layers},
            "drop_env": list(SCRUBBED_ENV),
            "out_dir": None,
            "report": g2 / "parity_state.report.json",
            "evaluate": evaluate_parity_state,
        },
        {
            "name": "replay",
            "argv": [str(bins / "qwen4_g2_replay_gate"), "--model", str(ARTIFACT),
                     "--output", str(g2 / "replay.report.json"),
                     "--expert-vram-layers", layers, "--context", str(REPLAY_FIXED["context"]),
                     "--iterations", str(REPLAY_FIXED["iterations"]), "--max-seq", str(REPLAY_FIXED["max_seq"])],
            # The replay gate takes its placement from the CLI only.
            "set_env": {},
            "drop_env": list(SCRUBBED_ENV) + list(REPLAY_SCRUBBED_ENV),
            "out_dir": None,
            "report": g2 / "replay.report.json",
            "evaluate": functools.partial(evaluate_replay, expert_layers=args.expert_vram_layers),
        },
    ]


class Interrupted(Exception):
    pass


_CURRENT: dict[str, subprocess.Popen[bytes] | None] = {"proc": None}


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


def _on_signal(signum: int, _frame: Any) -> None:
    proc = _CURRENT["proc"]
    if proc is not None and proc.poll() is None:
        _kill_group(proc)
    raise Interrupted(f"signal {signum}")


def run_child(spec: dict[str, Any], g2: Path, deadline: float, base_env: dict[str, str]) -> dict[str, Any]:
    name = spec["name"]
    env = {k: v for k, v in base_env.items() if k not in spec["drop_env"]}
    env.update(spec["set_env"])
    if spec["out_dir"] is not None:
        Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
    stdout_log, stderr_log = g2 / f"{name}.stdout.log", g2 / f"{name}.stderr.log"
    rec: dict[str, Any] = {
        "name": name,
        "argv": spec["argv"],
        "command": shlex.join(spec["argv"]),
        "env_set": spec["set_env"],
        "env_dropped": [k for k in spec["drop_env"] if k in base_env],
        "hipfire_env": {k: v for k, v in sorted(env.items()) if k.startswith("HIPFIRE_")},
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "report": str(spec["report"]),
        "out_dir": str(spec["out_dir"]) if spec["out_dir"] is not None else None,
        "started_utc": utc_now(),
    }
    t0 = time.monotonic()
    started_wall = time.time()
    timeout = max(60.0, deadline - t0)
    rc: int | None = None
    timed_out = False
    try:
        with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
            proc = subprocess.Popen(spec["argv"], stdout=out, stderr=err, env=env, start_new_session=True)
            _CURRENT["proc"] = proc
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                rc = proc.returncode
            finally:
                _CURRENT["proc"] = None
    except OSError as exc:
        rec["spawn_error"] = str(exc)
    rec.update({"rc": rc, "timed_out": timed_out, "wall_s": round(time.monotonic() - t0, 3), "ended_utc": utc_now()})

    verdict = _new_verdict()
    report = Path(spec["report"])
    if "spawn_error" in rec:
        _fail(verdict, "crash", f"could not start: {rec['spawn_error']}")
    if timed_out:
        _fail(verdict, "timeout", f"killed after {timeout:.0f}s")
    elif rc != 0:
        _fail(verdict, "crash" if rc is None or rc < 0 or rc > 2 else "harness_fail", f"rc={rc}")
    if not report.is_file():
        _fail(verdict, "report_missing", f"no report at {report}")
    elif report.stat().st_mtime < started_wall - 1:
        _fail(verdict, "report_invalid", f"report {report} predates this child")
    else:
        try:
            doc = json.loads(report.read_text(encoding="utf-8"))
        except ValueError as exc:
            _fail(verdict, "report_invalid", f"report is not JSON: {exc}")
        else:
            judged = spec["evaluate"](doc)
            for key in ("runs", "controls", "coverage", "observed"):
                if key in judged:
                    verdict[key] = judged[key]
            verdict["mismatches"] = judged["mismatches"]
            verdict["first_mismatch"] = judged["first_mismatch"]
            if judged["verdict"] == "SKIP":
                # Only a clean run (rc 0, report present) may skip; any earlier failure stands.
                if verdict["verdict"] == "PASS":
                    verdict["verdict"] = "SKIP"
                verdict["reasons"] += judged["reasons"]
            elif judged["verdict"] != "PASS":
                verdict["verdict"] = "FAIL"
                verdict["classification"] = _more_specific(verdict["classification"], judged["classification"])
                verdict["reasons"] += judged["reasons"]
        rec["report_sha256"] = sha256_file(report)
    if verdict["verdict"] == "FAIL":
        verdict["stderr_tail"] = tail_lines(stderr_log)
        if verdict["first_mismatch"] is None:
            verdict["first_mismatch"] = {"source": "process", "rc": rc, "stderr_tail": verdict["stderr_tail"][-10:]}
        verdict["first_mismatch"]["repro"] = {
            "command": rec["command"], "env": {**spec["set_env"]}, "dropped_env": spec["drop_env"],
            "out_dir": rec["out_dir"], "report": rec["report"],
            "note": "fabric-run.sh --class correct --card C ... -- must wrap this command; the report and "
                    "per-request ref/ and lanes/ state bytes stay under out_dir",
        }
    rec.update(verdict)
    return rec


# ── pre-flight ───────────────────────────────────────────────────────────


def stale_outputs(rcpt: Path, specs: list[dict[str, Any]], *, inside_job: bool = False) -> list[str]:
    stale = []
    if (rcpt / "g2.json").exists():
        stale.append(str(rcpt / "g2.json"))
    g2 = rcpt / "g2"
    for spec in specs:
        name = spec["name"]
        paths = [g2 / f"{name}.stdout.log", g2 / f"{name}.stderr.log", Path(spec["report"])]
        if spec["out_dir"] is not None:
            paths.append(Path(spec["out_dir"]))
        stale += [str(p) for p in paths if p.exists()]
    names = ["partial.json"] if inside_job else ["scheduler.json", "partial.json"]
    stale += [str(g2 / n) for n in names if (g2 / n).exists()]
    return stale


def preflight(args: argparse.Namespace, *, inside_job: bool) -> dict[str, Any]:
    """Every check that precedes a GPU launch; raises Refusal."""
    if socket.gethostname().split(".")[0] != HOST:
        raise Refusal(f"this carrier runs on {HOST}")
    refuse_fabric_bypass()
    git_info = verify_code_sha(args.code_sha)
    smoke = load_smoke_receipt(Path(args.smoke_receipt))
    sched, why = scheduler_card_record(args.card)
    card = validate_card(args.card, kfd_gpus(), sched, why)
    if inside_job:
        card["fabric_env"] = check_fabric_env(args.card)
        reply, why = fabric_status()
        if reply is None:
            raise Refusal(f"inside the job fabricd does not answer ({why}); refusing before any GPU child")
        card["fabric_job"] = check_fabric_job(args.card, os.environ, reply)

    if not ARTIFACT.is_file():
        raise Refusal(f"artifact {ARTIFACT} is missing")
    artifact = file_identity(ARTIFACT, hash_it=False)
    if artifact["bytes"] != ARTIFACT_BYTES:
        raise Refusal(f"artifact is {artifact['bytes']} bytes, expected {ARTIFACT_BYTES}")
    artifact["sha256"] = ARTIFACT_SHA256
    artifact["sha256_source"] = "pinned in the carrier (113 GB, not re-hashed)"
    pack = WEEK_RCPT / PACK_RECEIPT
    if pack.is_file():
        out = (json.loads(pack.read_text(encoding="utf-8")).get("output") or {})
        if out.get("sha256") != ARTIFACT_SHA256 or out.get("bytes") != ARTIFACT_BYTES:
            raise Refusal(f"{pack} output identity {out.get('sha256')}/{out.get('bytes')} disagrees with the pinned artifact")
        artifact["pack_receipt"] = {"path": str(pack), "sha256": sha256_file(pack)}

    inputs: dict[str, Any] = {}
    for key, path in (("text", Path(args.text)), ("tokens_json", Path(args.tokens)),
                      ("tokens_u32le", Path(args.tokens).with_suffix(".u32le"))):
        if not path.is_file():
            raise Refusal(f"input {key} {path} is missing")
        inputs[key] = file_identity(path)

    binaries: dict[str, Any] = {}
    for name in ("fn_lane_oracle", "qwen4_parity", "qwen4_g2_replay_gate"):
        path = Path(args.bin_dir) / name
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise Refusal(f"binary {path} is missing or not executable")
        binaries[name] = file_identity(path)

    specs = child_specs(args)
    stale = stale_outputs(Path(args.rcpt_dir), specs, inside_job=inside_job)
    if stale:
        raise Refusal("refusing to reuse stale outputs (move them aside first):\n  " + "\n  ".join(stale))
    return {"git": git_info, "smoke_receipt": smoke, "card": card, "artifact": artifact, "inputs": inputs,
            "binaries": binaries, "specs": specs}


# ── inner (runs inside the fabric job) ───────────────────────────────────


def inner(args: argparse.Namespace) -> int:
    t_start = time.monotonic()
    started = utc_now()
    rcpt = Path(args.rcpt_dir)
    g2 = rcpt / "g2"
    sched_argv = json.loads(os.environ.get("G2_CARRIER_SCHED_ARGV", "null"))
    ctx = preflight(args, inside_job=True)
    specs = ctx.pop("specs")
    g2.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "code_sha": ctx["git"]["head"],
        "git": ctx["git"],
        "expert_vram_layers": args.expert_vram_layers,
        "card": ctx["card"],
        "artifact": ctx["artifact"],
        "smoke_receipt": ctx["smoke_receipt"],
        "inputs": ctx["inputs"],
        "binaries": ctx["binaries"],
        "scheduler": {"submission": {**SCHED}, "argv": sched_argv, "inner_argv": [sys.executable, *sys.argv],
                      "job_id": ctx["card"]["fabric_job"]["id"], "fabric_job": ctx["card"]["fabric_job"],
                      "fabric_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FABRIC_")}},
        "environment": {"expert_vram_layers": args.expert_vram_layers,
                        "home": os.environ.get("HOME"),
                        "home_note": "the invoking HOME is inherited unchanged; no per-run HOME is created and the "
                                     "artifact is read in place, never copied",
                        "note":
                        f"layers {args.expert_vram_layers} via {ENV_EXPERT} for fn_lane_ar/fn_lane_mtp/parity_state; "
                        f"replay uses --expert-vram-layers {args.expert_vram_layers} on its CLI with the variable removed"},
        "started_utc": started,
        "checks": [],
    }
    deadline = t_start + SCHED["max_secs"] - DEADLINE_RESERVE_SECS
    interrupted: str | None = None
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        for spec in specs:  # serial, in order; every check runs even after a failure
            rec = run_child(spec, g2, deadline, dict(os.environ))
            doc["checks"].append(rec)
            write_json_atomic(g2 / "partial.json", doc)
    except Interrupted as exc:
        interrupted = str(exc)
        doc["interrupted"] = interrupted
    done = {c["name"] for c in doc["checks"]}
    skipped = [s["name"] for s in specs if s["name"] not in done]
    doc["not_run"] = skipped
    failed = [c["name"] for c in doc["checks"] if c["verdict"] == "FAIL"]
    skip_verdicts = [c["name"] for c in doc["checks"] if c["verdict"] == "SKIP"]
    ok = not failed and not skipped and interrupted is None
    first = next((c for c in doc["checks"] if c["verdict"] == "FAIL"), None)
    doc["summary"] = {
        "checks": len(doc["checks"]),
        "failed": failed,
        "skipped": skip_verdicts,
        "not_run": skipped,
        "mismatches": sum(c["mismatches"] for c in doc["checks"]),
        "first_failure": ({"check": first["name"], "classification": first["classification"],
                           "first_mismatch": first["first_mismatch"]} if first else None),
    }
    # A SKIP child (the replay gate on a model not on the product's default Redline route) is
    # neither a failure nor a PASS: the receipt is SKIP, never PASS, and the carrier exits 0.
    doc["status"] = "FAIL" if not ok else ("SKIP" if skip_verdicts else "PASS")
    doc["ended_utc"] = utc_now()
    doc["wall_s"] = round(time.monotonic() - t_start, 3)
    write_json_atomic(rcpt / "g2.json", doc)
    (g2 / "partial.json").unlink(missing_ok=True)
    print(f"[g2-carrier] {doc['status']}: {rcpt / 'g2.json'}", flush=True)
    return 0 if ok else 1


# ── outer (submits one fabric job) ───────────────────────────────────────


def fabric_argv(args: argparse.Namespace) -> list[str]:
    s = SCHED
    inner_cmd = [sys.executable, str(Path(__file__).resolve()), "--inner",
                 "--smoke-receipt", str(Path(args.smoke_receipt).resolve()), "--code-sha", args.code_sha,
                 "--card", args.card, "--rcpt-dir", str(args.rcpt_dir), "--bin-dir", str(args.bin_dir),
                 "--text", str(args.text), "--tokens", str(args.tokens),
                 "--expert-vram-layers", str(args.expert_vram_layers)]
    return [
        FABRIC_RUN, "--class", s["class"], "--card", args.card, "--mem-gb", str(s["mem_gb"]),
        "--hostmem-gb", str(s["hostmem_gb"]), "--hostmem-lock", "--priority", s["priority"],
        "--owner", s["owner"], "--max-secs", str(s["max_secs"]), "--", *inner_cmd,
    ]


def outer(args: argparse.Namespace) -> int:
    rcpt = Path(args.rcpt_dir)
    ctx = preflight(args, inside_job=False)
    ctx.pop("specs")
    argv = fabric_argv(args)
    write_json_atomic(rcpt / "g2" / "scheduler.json", {"argv": argv, "command": shlex.join(argv), "submitted_utc": utc_now(),
                                                       "submission": SCHED, "code_sha": ctx["git"]["head"],
                                                       "expert_vram_layers": args.expert_vram_layers})
    print("[g2-carrier] submitting:", shlex.join(argv), flush=True)
    t0 = time.monotonic()
    res = subprocess.run(argv, env={**os.environ, "G2_CARRIER_SCHED_ARGV": json.dumps(argv)}, check=False)
    wall = round(time.monotonic() - t0, 3)
    sched = {"argv": argv, "command": shlex.join(argv), "rc": res.returncode, "wall_s": wall,
             "ended_utc": utc_now(), "submission": SCHED}
    receipt = rcpt / "g2.json"
    if receipt.is_file():
        doc = json.loads(receipt.read_text(encoding="utf-8"))
    else:  # refused, timed out waiting, cancelled or crashed before the inner receipt
        partial = rcpt / "g2" / "partial.json"
        doc = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else {
            "schema": SCHEMA, "code_sha": ctx["git"]["head"], "git": ctx["git"], "card": ctx["card"],
            "expert_vram_layers": args.expert_vram_layers,
            "artifact": ctx["artifact"], "smoke_receipt": ctx["smoke_receipt"], "inputs": ctx["inputs"],
            "binaries": ctx["binaries"], "checks": []}
        doc["status"] = "FAIL"
        doc["classification"] = "scheduler_or_inner_refusal" if res.returncode else "no_receipt"
        doc["reasons"] = [f"fabric-run.sh rc {res.returncode} and the job wrote no g2.json (admission refused, "
                          "wait limit, cancel, or the inner carrier refused: see this process's stderr)"]
    doc["scheduler"] = {**doc.get("scheduler", {}), **sched}
    if res.returncode != 0 and doc.get("status") in ("PASS", "SKIP"):
        was, doc["status"] = doc["status"], "FAIL"
        doc["reasons"] = [f"fabric-run.sh rc {res.returncode} despite a {was} receipt"]
    write_json_atomic(receipt, doc)
    print(f"[g2-carrier] fabric rc {res.returncode}; g2 {doc['status']}: {receipt}", flush=True)
    return 0 if (res.returncode == 0 and doc["status"] in ("PASS", "SKIP")) else 1


def _layer_count(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"{n} is negative")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke-receipt", required=True, help="RouteOwner4's real-artifact smoke receipt (JSON with an explicit PASS)")
    p.add_argument("--code-sha", required=True, help="prefix of `git rev-parse HEAD` in the G2 worktree")
    p.add_argument("--card", default="C", choices=sorted(CARDS), help="legacy card label (default C)")
    p.add_argument("--expert-vram-layers", type=_layer_count, default=DEFAULT_EXPERT_LAYERS,
                   help="expert layers resident in VRAM for every check (integer, default "
                        f"{DEFAULT_EXPERT_LAYERS}; 15 only if 16 still runs out of device memory)")
    p.add_argument("--rcpt-dir", default=str(WEEK_RCPT), help="receipt directory (g2.json and g2/ go here)")
    p.add_argument("--bin-dir", default=str(REPO.parent / "target-subq4-g2" / "release" / "examples"))
    p.add_argument("--text", default=str(REPO / "benchmarks" / "prompts" / "embed_long.txt"),
                   help="TEXT for fn_lane_oracle prompts")
    p.add_argument("--tokens", default=str(REPO / "benchmarks" / "prompts" / "qwen4-teacher-forced.tokens.json"))
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return inner(args) if args.inner else outer(args)
    except Refusal as exc:
        print(f"[g2-carrier] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
