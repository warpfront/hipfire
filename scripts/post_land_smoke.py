#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Post-land GPU smoke: drive the raw daemon, pin greedy token ids per land.

Subcommands (python3 stdlib only):

  plan     print which loads of a plan would run (and why) without running any
  run      execute a plan against a daemon binary, one daemon process per load,
           and record raw stdout/stderr plus a parsed per-load JSON
  lanes    print (JSON) the plan's fabric lanes that have >= 1 selected load, with
           the fabric flags to submit each as a hold
  hold     meant to run as the command INSIDE a fabric hold: run one lane's selected
           loads under a kernel-log watcher (see "Hold" below)
  verdict  classify the recorded cells against the pinned token ids; takes one
           --results DIR per lane (one `run --only ...` or `hold` each), merged: a dir
           without run.json, two dirs covering one load, or dirs that ran with
           different --changed-paths/--diff are usage errors (exit 2); a plan load no
           dir covers is FAIL "not run" unless the recorded filters skip it (SKIPPED);
           --partial judges only what the dirs ran. A dir whose hold.json status is
           not complete/deadline (incomplete, or a crashed hold still "running") makes
           every cell of its loads FAIL "INCOMPLETE: <reason>" and the overall
           INCOMPLETE (beats FAIL); the verdict JSON lists those lanes under "incomplete".
  pin      write the verdict's ids into the pins file (atomic). Refuses FAIL, and
           INCOMPLETE always; CHANGED needs --accept-changed REASON. With
           --allow-failed-cells REASON a FAIL verdict still pins its NEW/MATCH cells
           (never FAIL, SKIPPED, CHANGED, or cells of a failed invariant); REASON and
           the failed cells go into the history entry.

`run` never compares anything; `verdict` decides pass/fail:
  exit 0 PASS, 1 FAIL, 3 CHANGED, 4 INCOMPLETE, 2 usage/plan error.

Daemon protocol (crates/hipfire-daemon/src/main.rs, hipfire-engine terminal.rs):
  -> {"type":"load","model":M,"params":{...}}
  <- {"type":"loaded","arch":...,...}
  -> {"type":"generate","id":CELL,"attempt_id":N,"prompt":P,"temperature":0.0,"max_tokens":K,...}
  <- {"type":"committed","id":CELL,"tok_id":T,"pos":P,"t_ms":MS,"attempt_id":N}   (HIPFIRE_EMIT_TOKEN_IDS=1)
  <- {"type":"token","id":CELL,"text":"..."}
  <- {"type":"commit_ready","id":CELL,"attempt_id":N,"finish_reason":...,...}
  -> {"type":"commit","id":CELL,"attempt_id":N}
  <- {"type":"done","id":CELL,"tokens":K,"tok_s":...,"finish_reason":"stop|length",...}
  -> {"type":"unload"}   <- {"type":"unloaded"}
A generate without a nonzero numeric attempt_id is refused, and a commit_ready
that is not answered by a matching commit is aborted after 30 s, so the runner
adds attempt_id (1-based cell index within the load), answers every commit_ready
with a commit, and sends the next generate only after the previous cell's
terminal (done/error) so cells never share a batch.

Plan JSON (version 1):
  {"version":1, "prompt":"...", "max_tokens":32, "stderr_fail_patterns":[re...],
   "loads":[{"name","model","load_params":{},"env":{},"deadline_s":N,
             "enabled":true, "run_if_paths":[re...], "note":"",
             "run_if_diff":[{"path":re,"hunk":re},...],
             "stderr_require":[re...],   # route proof: each must match >=1 stderr line
                                         # of the load, else every cell is FAIL
                                         # "required route line missing: <re>"
             "loaded_expect":{key:value}, "loaded_expect_not":{key:value},
                                         # checked against the daemon's `loaded` event (dotted
                                         # key = nested path): key must exist and equal (resp.
                                         # differ from) value, else every cell is FAIL
                                         # "loaded field mismatch: key=.. want=.. got=.."
             "cells":[{"name","generate":{}}]}],
   "invariants":[[cellA,cellB],...]}
A load runs iff `enabled` (default true) OR any `run_if_paths` pattern
re.search-matches a line of --changed-paths OR any `run_if_diff` rule matches
--diff (output of `git diff -U0 --no-color PREV LAND`): a rule matches iff
re.search(rule.path, b-side path (a-side for deletions)) and
re.search(rule.hunk, hunk text, re.MULTILINE) for some hunk of that file; hunk
text is the `@@ ... @@ funcname` header line plus its +/- lines (files with no
hunks -- binary, pure rename, mode change -- use their git metadata lines).
The matching rule (index, path, matching line) is recorded in the load JSON.
Skipped loads are recorded as {"skipped":true,...}; their cells get verdict
status SKIPPED (neutral: never FAIL/CHANGED, never pinned).

Plan additions (all validated by `validate_plan`; errors exit 2):
  load "model" may contain $VAR / ${VAR} / a leading ~, expanded (os.path.expandvars +
    expanduser, the runner's own environment) at run/hold/plan/lanes time; a `$`
    left after expansion FAILs the load before any daemon is spawned with reason
    "unresolved model path: <raw>" (plan/lanes print UNRESOLVED). The expanded path is what
    the daemon is asked to load.
  load "model_bytes": N (optional int): if the expanded file's size differs the load FAILs
    before spawn with "model size mismatch: want N got M".
  "kernel_fault_patterns": [re,...] (case-insensitive; default = the hiptrx hold.sh list,
    DEFAULT_KERNEL_FAULT_PATTERNS); matched against every kernel-log line by `hold`.
  "lanes": [{"name","host","cards","mem_gb","hostmem_gb","hostmem_lock":bool,"max_secs",
             "loads":[load names]}]: when present every load is in exactly one lane and lanes
    only name known loads; `host` is a free label. A lane's loads must fit its hold cap:
    sum(deadline_s + TERM_GRACE_S) + LANE_SLACK_S <= max_secs, so a slow load ends on its own
    deadline (FAIL) and never on the fabric cap (INCOMPLETE).

`lanes --plan P [--changed-paths F] [--diff F]` prints a JSON list: per lane with >= 1 selected
load {"name","host","cards","mem_gb","hostmem_gb","hostmem_lock","max_secs","loads":[selected,
plan order],"skipped_loads":[...],"unresolved_loads":[...],"fabric_args":["--class","correct",
"--card",CARDS,"--mem-gb",..,"--hostmem-gb",..,("--hostmem-lock"),"--max-secs",..]}.

Hold: `hold --plan P --lane L --daemon D --out DIR [--changed-paths F] [--diff F]
[--home-models DIR] [--no-fabric-check] [--watch-cmd ARG ...]`. Refuses (exit 2) unless
FABRIC_JOB is set. Probes `journalctl -k -n1` (unreadable -> exit 2 "watcher unavailable",
hold.json incomplete) unless --watch-cmd is given, then starts the watcher (default
`journalctl -k -f -n0 --no-pager -o short-iso-precise`; --watch-cmd consumes the rest of the
command line) in its own session/process group (start_new_session, close_fds=True,
stdin=/dev/null: no inherited fd > 2). A reader thread appends every line to DIR/kernel.log
and matches kernel_fault_patterns. The lane's selected loads run exactly like
`run --only <those loads>` (same per-load JSON; run.json records the changed_paths/diff
inputs); with --home-models each daemon gets HOME=DIR/<load>/home whose .hipfire/models is a
symlink to DIR (otherwise HOME is inherited), and always HIPFIRE_NO_REGISTRY_FETCH=1 and
HIPFIRE_EMIT_TOKEN_IDS=1. On a watcher hit, or the watcher process ending early, DIR/STOP is
written (the matching line / reason), the running daemon's process group is killed and no
further load runs; a SIGTERM/SIGINT/SIGHUP does the same (reason "signalled"). On every exit
path the watcher's process GROUP is killed (os.killpg) while its leader is still an unreaped zombie
(only then reaped: a reaped PGID may be reused) and verified gone afterwards
(os.killpg(pgid, 0) -> ProcessLookupError), recorded as watcher.gone. A leader that is no longer ours
(ECHILD, reaped elsewhere) is never signalled, probed or retried: the load records exit_code None (or the
known rc) plus the error "process group ownership lost; not signalled" (load ownership_lost), the watcher
records ownership_lost and gone false, and the hold is INCOMPLETE. DIR/hold.json:
{"lane","status": complete|incomplete|deadline,"reason","fabric_job","cards_env":
{FABRIC_CARDS,ROCR_VISIBLE_DEVICES,HIPFIRE_DEVICES},"watcher":{"cmd","pid","pgid","lines",
"hits","gone"},"started_utc","ended_utc","loads":[...]}. Exit 0 complete; 1 a load was killed
on its deadline or failed before spawn (status deadline); 4 incomplete (kernel fault, watcher
died, signalled); 2 usage / watcher unavailable.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PLAN_VERSION = 1
PINS_VERSION = 1
TERM_GRACE_S = 5.0
LANE_SLACK_S = 15  # per-lane allowance for watcher start/stop and spawn overheads inside the hold
NATURAL_STOPS = frozenset({"stop", "eos"})
EXIT_FOR_OVERALL = {"PASS": 0, "FAIL": 1, "CHANGED": 3, "INCOMPLETE": 4}
EXIT_USAGE = 2
EXIT_HOLD_INCOMPLETE = 4
RESERVED_GENERATE_KEYS = ("type", "id", "attempt_id")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_STDERR_HITS_PER_PATTERN = 20
MAX_WATCHER_HITS = 20
WATCH_FLUSH_S = 1.0  # let the journal follower flush trailing kernel lines before the final fault check
WATCH_SETTLE_S = 0.3  # a watcher that dies within this window never covered any load
WATCH_TERM_GRACE_S = 3.0
DEFAULT_WATCH_CMD = ["journalctl", "-k", "-f", "-n0", "--no-pager", "-o", "short-iso-precise"]
WATCH_PROBE_CMD = ["journalctl", "-k", "-n1", "--no-pager"]
DIFF_CONTEXT = 3

# Default kernel-fault regexes (case-insensitive), verbatim from the hiptrx hold.sh watcher.
DEFAULT_KERNEL_FAULT_PATTERNS = [
    r"ring \S+ (timeout|test failed)", r"IB test failed",
    r"GPU reset (begin|failed|succeeded)|\bgpu reset\b|Failed to reset|mode[12] reset|\bBACO\b|device wedged",
    r"VM_L2_PROTECTION_FAULT|\[(gfxhub|mmhub)\]|GCVM_L2_PROTECTION_FAULT|amdgpu.*page fault|page fault.*amdgpu|UTCL2",
    r"amdgpu.*(\*ERROR\*|fatal|\bhang\b|\bhung\b|timed out|timeout|device (lost|removed)|GPU (lost|hang))",
    r"\*ERROR\*.*(ring|gpu|\bvm\b|fence)",
    r"fallen off the bus|AER:.*(uncorrect|fatal)|PCIe Bus Error.*(uncorrect|fatal)|pcieport.*(DPC|link down)|PCIe link (is )?down",
    r"(kfd|amdgpu).*(HW exception|queue evict\w* (fail|timeout)|\bhw hang)",
    r"(amdgpu|kfd).*(Oops|BUG:|Call Trace)|BUG: unable to handle|Kernel panic",
]


class PlanError(Exception):
    """Plan failed validation; `.errors` lists every problem found."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class PinRefused(Exception):
    """pin refused (verdict FAIL, or CHANGED without a reason)."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ids_md5(ids) -> str:
    """md5 of the comma-joined decimal token ids (identity tag, not security)."""
    data = ",".join(str(int(i)) for i in ids).encode("ascii")
    try:
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    except TypeError:  # pragma: no cover - very old Python
        return hashlib.md5(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, obj) -> None:
    """Write JSON via tmp file + rename so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def model_bytes(model):
    """Size of the model file (or directory tree); None for registry tags etc."""
    try:
        p = Path(os.path.expanduser(str(model)))
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        pass
    return None


def expand_model(raw) -> str:
    """`$VAR` / `${VAR}` / leading `~` expansion of a load's model (the runner's own environment)."""
    return os.path.expanduser(os.path.expandvars(str(raw)))


def check_model(load):
    """-> (expanded model, problem or None). Problems are pre-spawn load failures:
    "unresolved model path: <raw>" (a `$` survives expansion) or
    "model size mismatch: want N got M" (load `model_bytes` differs from the expanded file's size)."""
    raw = load["model"]
    exp = expand_model(raw)
    if "$" in exp:
        return exp, f"unresolved model path: {raw}"
    want = load.get("model_bytes")
    if want is not None:
        got = model_bytes(exp)
        if got != want:
            return exp, f"model size mismatch: want {want} got {'missing' if got is None else got}"
    return exp, None


def fmt_num(v) -> str:
    """30.3 -> '30.3', 64.0 -> '64', 600 -> '600' (fabric flag values)."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def effective_max_tokens(plan, cell) -> int:
    return int(cell.get("generate", {}).get("max_tokens", plan["max_tokens"]))


# --------------------------------------------------------------------------
# plan validation / load selection
# --------------------------------------------------------------------------


def compile_fault_regex(patterns):
    """One case-insensitive regex alternating every kernel-fault pattern."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


def _validate_lanes(lanes, load_names, deadlines, errors):
    """Validate plan-level `lanes`; every load must sit in exactly one lane, lanes only name known loads,
    and a lane's worst case (each load's deadline plus its kill grace, plus LANE_SLACK_S) fits its
    max_secs, so the fabric hold cap never cuts a load (which would make the lane INCOMPLETE)."""
    if not isinstance(lanes, list) or not lanes:
        errors.append("lanes: must be a non-empty list of lane objects")
        return []
    out: list[dict] = []
    lane_names: set[str] = set()
    owner: dict[str, str] = {}
    for i, lane in enumerate(lanes):
        where = f"lanes[{i}]"
        if not isinstance(lane, dict):
            errors.append(f"{where}: must be an object")
            continue
        name = lane.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            errors.append(f"{where}.name: required string matching {NAME_RE.pattern}, got {name!r}")
            name = f"<{i}>"
        else:
            where = f"lane {name!r}"
            if name in lane_names:
                errors.append(f"{where}: duplicate lane name")
            lane_names.add(name)
        nl: dict = {"name": name}
        host = lane.get("host")
        if not isinstance(host, str) or not host:
            errors.append(f"{where}.host: required non-empty string")
        nl["host"] = host
        cards = lane.get("cards")
        if not isinstance(cards, str) or not cards:
            errors.append(f"{where}.cards: required non-empty string (fabric --card value)")
        nl["cards"] = cards
        for key, lo_ok in (("mem_gb", False), ("hostmem_gb", True)):
            v = lane.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or (v == 0 and not lo_ok):
                errors.append(f"{where}.{key}: required {'non-negative' if lo_ok else 'positive'} number, got {v!r}")
            nl[key] = v
        lock = lane.get("hostmem_lock", False)
        if not isinstance(lock, bool):
            errors.append(f"{where}.hostmem_lock: must be a boolean")
            lock = False
        nl["hostmem_lock"] = lock
        ms = lane.get("max_secs")
        if isinstance(ms, bool) or not isinstance(ms, int) or ms <= 0:
            errors.append(f"{where}.max_secs: required positive integer, got {ms!r}")
        nl["max_secs"] = ms
        ln = lane.get("loads")
        nl_loads: list[str] = []
        if not isinstance(ln, list) or not ln or not all(isinstance(x, str) for x in ln):
            errors.append(f"{where}.loads: required non-empty list of load names")
        else:
            for lname in ln:
                if lname not in load_names:
                    errors.append(f"{where}.loads: unknown load {lname!r}")
                elif lname in owner:
                    errors.append(f"load {lname!r} is in two lanes ({owner[lname]!r} and {name!r})" if owner[lname] != name else f"{where}.loads: load {lname!r} listed twice")
                else:
                    owner[lname] = name
                    nl_loads.append(lname)
        nl["loads"] = nl_loads
        if isinstance(ms, int) and not isinstance(ms, bool) and ms > 0 and all(
            isinstance(deadlines.get(x), (int, float)) and not isinstance(deadlines.get(x), bool) for x in nl_loads
        ):
            worst = sum(deadlines[x] + TERM_GRACE_S for x in nl_loads) + LANE_SLACK_S
            if worst > ms:
                errors.append(
                    f"{where}: worst-case time {worst:g}s (sum of its loads' deadline_s, {TERM_GRACE_S:g}s kill grace "
                    f"per load, {LANE_SLACK_S}s slack) exceeds max_secs {ms}; the hold cap would cut a load"
                )
        out.append(nl)
    for lname in sorted(load_names - set(owner)):
        errors.append(f"load {lname!r} is in no lane (lanes is present, so every load must be in exactly one lane)")
    return out


def _compile_all(patterns, where, errors):
    if not isinstance(patterns, list):
        errors.append(f"{where}: must be a list of regex strings")
        return []
    ok = []
    for pat in patterns:
        if not isinstance(pat, str):
            errors.append(f"{where}: pattern {pat!r} is not a string")
            continue
        try:
            re.compile(pat)
        except re.error as e:
            errors.append(f"{where}: bad regex {pat!r}: {e}")
            continue
        ok.append(pat)
    return ok


def validate_plan(plan) -> dict:
    """Return the plan with defaults filled in, or raise PlanError (all problems)."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        raise PlanError(["plan must be a JSON object"])
    if plan.get("version") != PLAN_VERSION:
        errors.append(f"version must be {PLAN_VERSION}, got {plan.get('version')!r}")
    out: dict = {"version": PLAN_VERSION}

    prompt = plan.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt: required non-empty string")
    out["prompt"] = prompt

    max_tokens = plan.get("max_tokens", 32)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        errors.append(f"max_tokens: must be a positive integer, got {max_tokens!r}")
    out["max_tokens"] = max_tokens

    out["stderr_fail_patterns"] = _compile_all(
        plan.get("stderr_fail_patterns", []), "stderr_fail_patterns", errors
    )

    loads = plan.get("loads")
    out_loads: list[dict] = []
    if not isinstance(loads, list) or not loads:
        errors.append("loads: required non-empty list")
        loads = []
    load_names: set[str] = set()
    cell_names: dict[str, str] = {}
    for li, load in enumerate(loads):
        where = f"loads[{li}]"
        if not isinstance(load, dict):
            errors.append(f"{where}: must be an object")
            continue
        name = load.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            errors.append(f"{where}.name: required string matching {NAME_RE.pattern}, got {name!r}")
            name = f"<{li}>"
        else:
            where = f"load {name!r}"
            if name in load_names:
                errors.append(f"{where}: duplicate load name")
            load_names.add(name)
        nl: dict = {"name": name}
        model = load.get("model")
        if not isinstance(model, str) or not model:
            errors.append(f"{where}.model: required non-empty string")
        nl["model"] = model
        mb = load.get("model_bytes")
        if mb is not None and (isinstance(mb, bool) or not isinstance(mb, int) or mb < 0):
            errors.append(f"{where}.model_bytes: must be a non-negative integer, got {mb!r}")
            mb = None
        nl["model_bytes"] = mb
        lp = load.get("load_params", {})
        if not isinstance(lp, dict):
            errors.append(f"{where}.load_params: must be an object")
            lp = {}
        nl["load_params"] = lp
        env = load.get("env", {})
        norm_env: dict[str, str] = {}
        if not isinstance(env, dict):
            errors.append(f"{where}.env: must be an object")
        else:
            for k, v in env.items():
                if isinstance(v, bool) or not isinstance(v, (str, int)):
                    errors.append(f"{where}.env[{k!r}]: must be a string or integer, got {v!r}")
                else:
                    norm_env[str(k)] = str(v)
        nl["env"] = norm_env
        dl = load.get("deadline_s")
        if isinstance(dl, bool) or not isinstance(dl, (int, float)) or dl <= 0:
            errors.append(f"{where}.deadline_s: required positive number, got {dl!r}")
        nl["deadline_s"] = dl
        enabled = load.get("enabled", True)
        if not isinstance(enabled, bool):
            errors.append(f"{where}.enabled: must be a boolean")
            enabled = True
        nl["enabled"] = enabled
        nl["run_if_paths"] = _compile_all(load.get("run_if_paths", []), f"{where}.run_if_paths", errors)
        rules = load.get("run_if_diff", [])
        nl_rules: list[dict] = []
        if not isinstance(rules, list):
            errors.append(f"{where}.run_if_diff: must be a list of {{path, hunk}} objects")
        else:
            for ri, rule in enumerate(rules):
                rwhere = f"{where}.run_if_diff[{ri}]"
                if not isinstance(rule, dict) or not isinstance(rule.get("path"), str) or set(rule) - {"path", "hunk"}:
                    errors.append(f"{rwhere}: must be an object with a string 'path' and optional string 'hunk'")
                    continue
                ok = _compile_all([rule["path"], rule.get("hunk", "")], rwhere, errors)
                if len(ok) == 2:
                    nl_rules.append({"path": ok[0], "hunk": ok[1]})
        nl["run_if_diff"] = nl_rules
        nl["stderr_require"] = _compile_all(load.get("stderr_require", []), f"{where}.stderr_require", errors)
        for fld in ("loaded_expect", "loaded_expect_not"):
            exp = load.get(fld, {})
            if not isinstance(exp, dict):
                errors.append(f"{where}.{fld}: must be an object of key -> JSON value")
                exp = {}
            for k in exp:
                if not k or any(part == "" for part in k.split(".")):
                    errors.append(f"{where}.{fld}: bad key {k!r} (empty dotted-path segment)")
            nl[fld] = exp
        note = load.get("note", "")
        if not isinstance(note, str):
            errors.append(f"{where}.note: must be a string")
            note = ""
        nl["note"] = note
        cells = load.get("cells")
        nl_cells: list[dict] = []
        if not isinstance(cells, list) or not cells:
            errors.append(f"{where}.cells: required non-empty list")
            cells = []
        for ci, cell in enumerate(cells):
            cwhere = f"{where}.cells[{ci}]"
            if not isinstance(cell, dict):
                errors.append(f"{cwhere}: must be an object")
                continue
            cname = cell.get("name")
            if not isinstance(cname, str) or not NAME_RE.match(cname):
                errors.append(f"{cwhere}.name: required string matching {NAME_RE.pattern}, got {cname!r}")
                continue
            cwhere = f"cell {cname!r}"
            if cname in cell_names:
                errors.append(f"{cwhere}: duplicate cell name (also in load {cell_names[cname]!r})")
            cell_names[cname] = name
            gen = cell.get("generate", {})
            if not isinstance(gen, dict):
                errors.append(f"{cwhere}.generate: must be an object")
                gen = {}
            for key in RESERVED_GENERATE_KEYS:
                if key in gen:
                    errors.append(f"{cwhere}.generate: {key!r} is set by the runner")
            mt = gen.get("max_tokens")
            if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt <= 0):
                errors.append(f"{cwhere}.generate.max_tokens: must be a positive integer")
            nl_cells.append({"name": cname, "generate": gen})
        nl["cells"] = nl_cells
        out_loads.append(nl)
    clash = load_names & set(cell_names)
    for n in sorted(clash):
        errors.append(f"name {n!r} used for both a load and a cell (--only would be ambiguous)")
    out["loads"] = out_loads

    invariants = plan.get("invariants", [])
    out_inv: list[list[str]] = []
    if not isinstance(invariants, list):
        errors.append("invariants: must be a list of [cellA, cellB] pairs")
    else:
        for ii, pair in enumerate(invariants):
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                errors.append(f"invariants[{ii}]: must be a pair of cell names")
                continue
            for x in pair:
                if x not in cell_names:
                    errors.append(f"invariants[{ii}]: unknown cell {x!r}")
            out_inv.append(list(pair))
    out["invariants"] = out_inv

    kfp = plan.get("kernel_fault_patterns")
    if kfp is None:
        out["kernel_fault_patterns"] = list(DEFAULT_KERNEL_FAULT_PATTERNS)
    else:
        pats = _compile_all(kfp, "kernel_fault_patterns", errors)
        if isinstance(kfp, list) and not kfp:
            errors.append("kernel_fault_patterns: must be non-empty when present (omit it for the default list)")
        elif pats and len(pats) == len(kfp):
            try:
                compile_fault_regex(pats)
            except re.error as e:
                errors.append(f"kernel_fault_patterns: patterns do not combine into one regex: {e}")
        out["kernel_fault_patterns"] = pats

    if "lanes" in plan:
        deadlines = {ld["name"]: ld["deadline_s"] for ld in out["loads"] if isinstance(ld.get("name"), str)}
        out["lanes"] = _validate_lanes(plan["lanes"], load_names, deadlines, errors)

    if errors:
        raise PlanError(errors)
    return out


def load_plan(path):
    """Read + validate a plan file -> (normalized plan, sha256 of the file bytes)."""
    raw = Path(path).read_bytes()
    try:
        plan = json.loads(raw)
    except ValueError as e:
        raise PlanError([f"{path}: not valid JSON: {e}"])
    return validate_plan(plan), hashlib.sha256(raw).hexdigest()


def read_changed_paths(path) -> list[str]:
    """One repo path per line (e.g. `git diff --name-only PREV..LAND`)."""
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _diff_file_path(s):
    """Path from a `--- a/x` / `+++ b/x` / `Binary files` operand; None for /dev/null."""
    s = s.split("\t", 1)[0].strip()
    if s == "/dev/null":
        return None
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    if s[:2] in ("a/", "b/"):
        s = s[2:]
    return s


def _split_diff_git_header(rest):
    """`a/X b/Y` operands of a `diff --git` line -> (X, Y), or (None, None)."""
    if not rest.startswith("a/"):
        return None, None
    body = rest[2:]
    n = len(body)
    if n >= 5 and (n - 3) % 2 == 0:  # unambiguous when both sides are the same path
        half = (n - 3) // 2
        if body[half : half + 3] == " b/" and body[:half] == body[half + 3 :]:
            return body[:half], body[:half]
    m = re.match(r"^a/(.+) b/(.+)$", rest)
    return (m.group(1), m.group(2)) if m else (None, None)


def parse_unified_diff(text) -> list[dict]:
    """Parse `git diff -U0 --no-color` output into (path, hunk) entries.

    One entry per hunk: {"path": b-side path (a-side for deletions), "a_path",
    "b_path", "hunk": "@@ ... @@ funcname\\n+/- lines"}. A file with no hunk
    (binary, pure rename, mode change, empty new file) yields one entry whose
    hunk is its git metadata lines. Unknown/garbage lines are tolerated.
    """
    files: list[dict] = []
    cur = None
    hunk = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            a, b = _split_diff_git_header(line[len("diff --git ") :])
            cur = {"a": a, "b": b, "meta": [], "hunks": []}
            files.append(cur)
            hunk = None
            continue
        if cur is None:
            continue
        if line.startswith("@@"):
            hunk = [line]
            cur["hunks"].append(hunk)
            continue
        if hunk is not None:
            if line[:1] in ("+", "-", " ") and line:
                hunk.append(line)
            continue  # "\ No newline at end of file" and friends
        if line.startswith("--- "):
            cur["a"] = _diff_file_path(line[4:])
        elif line.startswith("+++ "):
            cur["b"] = _diff_file_path(line[4:])
        elif line.startswith("rename from ") or line.startswith("copy from "):
            cur["a"] = line.split(" from ", 1)[1]
            cur["meta"].append(line)
        elif line.startswith("rename to ") or line.startswith("copy to "):
            cur["b"] = line.split(" to ", 1)[1]
            cur["meta"].append(line)
        elif line.startswith("Binary files "):
            m = re.match(r"^Binary files (.+) and (.+) differ$", line)
            if m:
                cur["a"] = _diff_file_path(m.group(1))
                cur["b"] = _diff_file_path(m.group(2))
            cur["meta"].append(line)
        elif line.startswith("new file mode"):
            cur["a"] = None
            cur["meta"].append(line)
        elif line.startswith("deleted file mode"):
            cur["b"] = None
            cur["meta"].append(line)
        elif line.startswith(("similarity index", "dissimilarity index", "old mode", "new mode", "GIT binary patch")):
            cur["meta"].append(line)
    entries: list[dict] = []
    for f in files:
        path = f["b"] or f["a"] or ""
        hunks = ["\n".join(h) for h in f["hunks"]] or ["\n".join(f["meta"])]
        for h in hunks:
            entries.append({"path": path, "a_path": f["a"], "b_path": f["b"], "hunk": h})
    return entries


def read_diff_entries(path) -> list[dict]:
    return parse_unified_diff(Path(path).read_text(encoding="utf-8", errors="replace"))


def read_diff(path):
    """-> (entries, sha256 of the file bytes); the sha lets verdict detect differing --diff inputs."""
    raw = Path(path).read_bytes()
    return parse_unified_diff(raw.decode("utf-8", errors="replace")), hashlib.sha256(raw).hexdigest()


def match_diff_rule(rule, entries):
    """-> {"path", "line"} for the first entry the rule matches, else None."""
    prx = re.compile(rule["path"])
    hrx = re.compile(rule.get("hunk") or "", re.MULTILINE)
    for e in entries:
        if not prx.search(e["path"]):
            continue
        m = hrx.search(e["hunk"])
        if m:
            lines = e["hunk"].split("\n")
            idx = e["hunk"].count("\n", 0, m.start())
            return {"path": e["path"], "line": (lines[idx] if idx < len(lines) else "")[:200]}
    return None


def evaluate_load(load, changed_paths=None, diff_entries=None) -> dict:
    """Decide whether a load runs -> {"run": bool, "reason": str, "matched": [rule hits]}.

    `changed_paths` / `diff_entries` are None when --changed-paths / --diff were not given.
    Each hit: {kind: run_if_paths|run_if_diff, rule_index, path, line, pattern[, hunk_pattern]}.
    """
    if load.get("enabled", True):
        return {"run": True, "reason": "enabled", "matched": []}
    pats = load.get("run_if_paths") or []
    rules = load.get("run_if_diff") or []
    matched: list[dict] = []
    parts: list[str] = []
    if pats:
        if changed_paths is None:
            parts.append("run_if_paths given but --changed-paths not supplied")
        else:
            hit = False
            for i, pat in enumerate(pats):
                for path in changed_paths:
                    if re.search(pat, path):
                        matched.append(
                            {"kind": "run_if_paths", "rule_index": i, "path": path, "line": path[:200], "pattern": pat}
                        )
                        hit = True
                        break
            if not hit:
                parts.append("no changed path matches run_if_paths")
    if rules:
        if diff_entries is None:
            parts.append("run_if_diff given but --diff not supplied")
        else:
            hit = False
            for i, rule in enumerate(rules):
                hm = match_diff_rule(rule, diff_entries)
                if hm:
                    matched.append(
                        {
                            "kind": "run_if_diff",
                            "rule_index": i,
                            "path": hm["path"],
                            "line": hm["line"],
                            "pattern": rule["path"],
                            "hunk_pattern": rule.get("hunk", ""),
                        }
                    )
                    hit = True
            if not hit:
                parts.append("no diff hunk matches run_if_diff")
    if matched:
        m = matched[0]
        if m["kind"] == "run_if_paths":
            why = f"enabled by changed path {m['path']!r} (run_if_paths {m['pattern']!r})"
        else:
            why = f"enabled by diff of {m['path']!r} (run_if_diff[{m['rule_index']}], line: {m['line']!r})"
        if len(matched) > 1:
            why += f" (+{len(matched) - 1} more rule(s))"
        return {"run": True, "reason": why, "matched": matched}
    if not parts:
        return {"run": False, "reason": "disabled (enabled=false, no run_if_paths/run_if_diff)", "matched": []}
    return {"run": False, "reason": "disabled; " + "; ".join(parts), "matched": []}


def load_selection(load, changed_paths, diff_entries=None):
    """-> (should_run, reason). `changed_paths`/`diff_entries` are None when not given."""
    ev = evaluate_load(load, changed_paths, diff_entries)
    return ev["run"], ev["reason"]


def select_cells(plan, only):
    """Apply --only (load or cell names) -> {load_name: [cell dicts]}; PlanError on unknown."""
    if not only:
        return {ld["name"]: list(ld["cells"]) for ld in plan["loads"]}
    known = {ld["name"] for ld in plan["loads"]} | {c["name"] for ld in plan["loads"] for c in ld["cells"]}
    unknown = [o for o in only if o not in known]
    if unknown:
        raise PlanError([f"--only: unknown load/cell name(s): {', '.join(unknown)}"])
    sel: dict[str, list[dict]] = {}
    for ld in plan["loads"]:
        if ld["name"] in only:
            sel[ld["name"]] = list(ld["cells"])
        else:
            cells = [c for c in ld["cells"] if c["name"] in only]
            if cells:
                sel[ld["name"]] = cells
    return sel


# --------------------------------------------------------------------------
# daemon stdout / stderr parsing
# --------------------------------------------------------------------------

_REQUEST_EVENT_TYPES = ("token", "reasoning", "committed", "commit_ready", "done", "error")


def _blank_request() -> dict:
    return {
        "ids": [],
        "positions": [],
        "text": "",
        "reasoning": "",
        "finish": None,
        "done": None,
        "commit_ready": False,
        "errors": [],
    }


def parse_daemon_stdout(lines) -> dict:
    """Parse raw daemon stdout lines (str or bytes).

    Returns {"loaded": first loaded event | None, "unloaded": bool,
             "requests": {id: {ids, positions, text, reasoning, finish, done,
                               commit_ready, errors}},
             "errors": [error events with no request id],
             "garbage": [{line_no, line}], "other_types": {type: count}}.
    Non-JSON lines are recorded, never fatal.
    """
    out: dict = {
        "loaded": None,
        "unloaded": False,
        "requests": {},
        "errors": [],
        "garbage": [],
        "other_types": {},
    }
    for n, raw in enumerate(lines, 1):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        if not isinstance(ev, dict):
            out["garbage"].append({"line_no": n, "line": line[:200]})
            continue
        typ = ev.get("type")
        rid = ev.get("id")
        has_id = isinstance(rid, str) and rid != ""
        if typ == "loaded":
            if out["loaded"] is None:
                out["loaded"] = ev
        elif typ == "unloaded":
            out["unloaded"] = True
        elif typ == "error" and not has_id:
            out["errors"].append(ev)
        elif typ in _REQUEST_EVENT_TYPES and has_id:
            req = out["requests"].setdefault(rid, _blank_request())
            if typ == "committed":
                tid = ev.get("tok_id")
                if isinstance(tid, int) and not isinstance(tid, bool):
                    req["ids"].append(tid)
                    req["positions"].append(ev.get("pos"))
                else:
                    req["errors"].append({"message": f"committed event without integer tok_id: {line[:120]}"})
            elif typ == "token":
                req["text"] += str(ev.get("text", ""))
            elif typ == "reasoning":
                req["reasoning"] += str(ev.get("text", ""))
            elif typ == "commit_ready":
                req["commit_ready"] = True
            elif typ == "done":
                if req["done"] is None:
                    req["done"] = ev
                    fin = ev.get("finish_reason", ev.get("finish"))
                    req["finish"] = fin if isinstance(fin, str) else None
            else:  # error with id
                req["errors"].append(ev)
        else:
            key = str(typ) if typ is not None else "<no type>"
            out["other_types"][key] = out["other_types"].get(key, 0) + 1
    return out


_MISSING = object()


def _loaded_lookup(loaded, key):
    """Value at `key` in the loaded event: a literal top-level key first, else a dotted path."""
    if key in loaded:
        return loaded[key]
    cur = loaded
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def check_loaded(loaded, expect, expect_not) -> list[dict]:
    """-> [{key, op: "expect"|"expect_not", want, found, got, ok}] (loaded None = event absent)."""
    out: list[dict] = []
    for op, spec in (("expect", expect or {}), ("expect_not", expect_not or {})):
        for key, want in spec.items():
            got = _loaded_lookup(loaded, key) if isinstance(loaded, dict) else _MISSING
            found = got is not _MISSING
            ok = found and ((got == want and type(got) is type(want)) if op == "expect" else not (got == want and type(got) is type(want)))
            out.append({"key": key, "op": op, "want": want, "found": found, "got": None if not found else got, "ok": bool(ok)})
    return out


def scan_stderr_require(lines, patterns) -> list[dict]:
    """Route proof: -> [{pattern, matched, line}] with the FIRST stderr line matching each regex."""
    proof = [{"pattern": p, "matched": False, "line": None} for p in patterns]
    compiled = [re.compile(p) for p in patterns]
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.rstrip("\r\n")
        for rec, rx in zip(proof, compiled):
            if not rec["matched"] and rx.search(line):
                rec["matched"] = True
                rec["line"] = line[:200]
        if all(rec["matched"] for rec in proof):
            break
    return proof


def scan_stderr(lines, patterns) -> list[dict]:
    """-> [{pattern, line}] for lines matching any fail pattern (capped per pattern)."""
    compiled = [(p, re.compile(p)) for p in patterns]
    counts = {p: 0 for p in patterns}
    hits: list[dict] = []
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        line = raw.rstrip("\r\n")
        for pat, rx in compiled:
            if counts[pat] >= MAX_STDERR_HITS_PER_PATTERN:
                continue
            if rx.search(line):
                counts[pat] += 1
                hits.append({"pattern": pat, "line": line[:500]})
    return hits


def build_cell_result(req, gen_s=None, not_run=None) -> dict:
    """Per-cell record from a parsed request (None = no events at all)."""
    req = req if req is not None else _blank_request()
    ids = list(req["ids"])
    errors = list(req["errors"])
    if not_run:
        errors.append({"message": not_run})
    return {
        "ids": ids,
        "n": len(ids),
        "ids_md5": ids_md5(ids),
        "text": req["text"],
        "finish": req["finish"],
        "done_event": req["done"],
        "gen_s": gen_s,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _send(proc, obj) -> bool:
    try:
        proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
        proc.stdin.flush()
        return True
    except (BrokenPipeError, ValueError, OSError):
        return False


def _close_stdin(proc) -> None:
    try:
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        pass


class LeaderLost(Exception):
    """The group leader is no longer ours (reaped elsewhere / ECHILD): its PID/PGID may have been reused.

    `rc` is the real exit code when one is known (reaped through Popen), else None: never fabricated.
    Once raised, NO signal may be sent to that group, neither by PGID nor by PID, and it is not probed.
    """

    def __init__(self, rc=None):
        super().__init__("process group ownership lost")
        self.rc = rc


def _waitid_peek(proc):
    """Exit state of `proc` WITHOUT reaping it: the siginfo once it has exited, else None.

    Raises LeaderLost when the leader is not an unreaped (or running) child of ours any more: it was
    reaped through `proc` (returncode set) or by someone else (ECHILD).
    """
    if proc.returncode is not None:
        raise LeaderLost(proc.returncode)
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
    except ChildProcessError:
        raise LeaderLost(proc.returncode) from None


def _wait_exit_unreaped(proc, timeout) -> bool:
    """Wait up to `timeout` s for `proc` to exit without reaping it; True once it has exited.

    A group leader must stay an unreaped zombie until the LAST os.killpg on its group: while it is a
    zombie the kernel cannot reuse its PID/PGID, so the signal can only reach a group we own. Never call
    proc.wait()/poll()/communicate() on a leader before that last signal. Raises LeaderLost.
    """
    end = time.monotonic() + max(0.0, timeout)
    while True:
        if _waitid_peek(proc) is not None:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.02)


def _unreaped_rc(proc):
    """Return code of an exited-but-unreaped `proc` (Popen convention: -N when killed by signal N). Raises LeaderLost."""
    info = _waitid_peek(proc)
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _killpg_owned(proc, pgid, sig) -> None:
    """os.killpg(pgid, sig) only while the leader `proc` is provably still ours; raises LeaderLost (zero
    signals sent) otherwise. Checked before EVERY group signal, the first TERM included."""
    _waitid_peek(proc)
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_group(proc) -> bool:
    """SIGTERM the daemon's process group, TERM_GRACE_S, then SIGKILL the group, then reap the leader.

    The leader stays unreaped through the final SIGKILL (see _wait_exit_unreaped). Returns False when the
    leader was lost (LeaderLost): nothing further is signalled and the leader is not waited on.
    """
    try:
        _killpg_owned(proc, proc.pid, signal.SIGTERM)
        _wait_exit_unreaped(proc, TERM_GRACE_S)
        _killpg_owned(proc, proc.pid, signal.SIGKILL)
    except LeaderLost:
        return False
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass
    return True


def run_load(load, plan, daemon, out_dir, cells=None, daemon_sha256=None, selection=None, env_extra=None, abort=None) -> dict:
    """Run one load (one daemon process); write its raw logs and <load>.json.

    env_extra: environment overlay applied after the load's own env (hold: HOME, HIPFIRE_NO_REGISTRY_FETCH).
    abort: AbortFlag; when it is set the daemon's process group is killed and the load recorded as aborted.
    A load whose model is unresolved / the wrong size, or whose daemon cannot spawn, never starts a daemon:
    `spawned` is false and every cell is "not run".
    """
    name = load["name"]
    cells = list(load["cells"] if cells is None else cells)
    out = Path(out_dir)
    stdout_path = out / f"{name}.stdout.jsonl"
    stderr_path = out / f"{name}.stderr.log"
    model_exp, model_problem = check_model(load)
    env = dict(os.environ)
    env.update(load["env"])
    env.update(env_extra or {})
    env["HIPFIRE_EMIT_TOKEN_IDS"] = "1"
    result: dict = {
        "load": name,
        "model": load["model"],
        "model_expanded": model_exp,
        "env": dict(load["env"]),
        "env_extra": dict(env_extra or {}),
        "load_params": load["load_params"],
        "note": load.get("note", ""),
        "selection": selection,
        "started_utc": utc_now(),
        "wall_s": None,
        "load_s": None,
        "exit_code": None,
        "spawned": True,
        "killed_on_deadline": False,
        "aborted": False,
        "ownership_lost": False,
        "daemon_path": str(daemon),
        "daemon_sha256": daemon_sha256,
        "model_bytes": model_bytes(model_exp),
        "model_bytes_want": load.get("model_bytes"),
        "errors": [],
        "warnings": [],
        "stderr_hits": [],
        "route_proof": scan_stderr_require([], load.get("stderr_require", [])),
        "loaded_check": check_loaded(None, load.get("loaded_expect"), load.get("loaded_expect_not")),
        "garbage_lines": 0,
        "cells": {},
    }
    t0 = time.monotonic()
    deadline = t0 + float(load["deadline_s"])
    proc = None
    fail = model_problem
    if fail is None:
        try:
            proc = subprocess.Popen(
                [str(daemon)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as e:
            fail = f"spawn failed: {e}"
    if proc is None:
        result["spawned"] = False
        result["errors"].append({"message": fail})
        result["wall_s"] = round(time.monotonic() - t0, 3)
        for c in cells:
            result["cells"][c["name"]] = build_cell_result(
                None, None, "not run: daemon spawn failed" if model_problem is None else f"not run: {model_problem}"
            )
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        atomic_write_json(out / f"{name}.json", result)
        return result

    q: queue.Queue = queue.Queue()

    def pump_stdout():
        with open(stdout_path, "wb") as tee:
            for raw in iter(proc.stdout.readline, b""):
                tee.write(raw)
                tee.flush()
                q.put(raw.decode("utf-8", "replace"))
        q.put(None)

    def pump_stderr():
        with open(stderr_path, "wb") as tee:
            for raw in iter(proc.stderr.readline, b""):
                tee.write(raw)
                tee.flush()

    t_out = threading.Thread(target=pump_stdout, daemon=True)
    t_err = threading.Thread(target=pump_stderr, daemon=True)
    t_out.start()
    t_err.start()

    attempt_of = {c["name"]: i + 1 for i, c in enumerate(cells)}
    pending = list(cells)
    inflight = None
    sent_unload = False
    load_failed = False
    loaded_t = None
    first_commit_t: dict[str, float] = {}
    done_t: dict[str, float] = {}
    lines: list[str] = []
    killed = False
    aborted = False
    lost = False

    _send(proc, {"type": "load", "model": model_exp, "params": load["load_params"]})

    def advance():
        nonlocal inflight, sent_unload
        if inflight is not None or sent_unload:
            return
        if pending and not load_failed:
            c = pending.pop(0)
            msg = {
                "type": "generate",
                "id": c["name"],
                "attempt_id": attempt_of[c["name"]],
                "prompt": plan["prompt"],
                "temperature": 0.0,
                "max_tokens": plan["max_tokens"],
            }
            msg.update(c["generate"])
            _send(proc, msg)
            inflight = c["name"]
        else:
            _send(proc, {"type": "unload"})
            sent_unload = True

    def handle(line: str) -> None:
        nonlocal inflight, loaded_t, load_failed
        s = line.strip()
        if not s.startswith("{"):
            return
        try:
            ev = json.loads(s)
        except ValueError:
            return
        if not isinstance(ev, dict):
            return
        typ = ev.get("type")
        rid = ev.get("id")
        now = time.monotonic()
        if typ == "loaded":
            if loaded_t is None:
                loaded_t = now
                advance()
        elif typ == "unloaded":
            _close_stdin(proc)
        elif typ == "error" and (not isinstance(rid, str) or rid not in attempt_of):
            if loaded_t is None and not load_failed:
                load_failed = True
                advance()
        elif typ == "committed" and isinstance(rid, str):
            first_commit_t.setdefault(rid, now)
        elif typ == "commit_ready" and isinstance(rid, str) and rid in attempt_of:
            aid = ev.get("attempt_id")
            if isinstance(aid, bool) or not isinstance(aid, int) or aid <= 0:
                aid = attempt_of[rid]
            _send(proc, {"type": "commit", "id": rid, "attempt_id": aid})
        elif typ in ("done", "error") and isinstance(rid, str):
            if typ == "done":
                done_t.setdefault(rid, now)
            if rid == inflight:
                inflight = None
                advance()

    while True:
        if abort is not None and abort.is_set():
            aborted = True
            lost = not kill_group(proc)
            break
        remaining = deadline - time.monotonic()
        try:
            item = q.get(timeout=max(0.0, min(remaining, 0.2)))
        except queue.Empty:
            if time.monotonic() >= deadline:
                killed = True
                lost = not kill_group(proc)
                break
            continue
        if item is None:
            break
        lines.append(item)
        handle(item)

    _close_stdin(proc)
    if not (killed or aborted):
        try:
            if _wait_exit_unreaped(proc, max(0.1, deadline - time.monotonic())):
                # stdout EOF can arrive before the exit status; stray group members die with the group.
                # The leader is still an unreaped zombie here, so its PGID cannot have been reused.
                _killpg_owned(proc, proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
            else:
                killed = True
                lost = not kill_group(proc)
        except LeaderLost:
            lost = True
    else:
        # kill_group already sent the final group signal and reaped; stdout may still hold buffered lines
        t_out.join(timeout=5)
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item is not None:
            lines.append(item)
    try:
        for stream in (proc.stdout, proc.stderr):
            stream.close()
    except OSError:
        pass

    wall = time.monotonic() - t0
    parsed = parse_daemon_stdout(lines)
    result["wall_s"] = round(wall, 3)
    result["exit_code"] = proc.returncode
    result["killed_on_deadline"] = killed
    result["aborted"] = aborted
    if loaded_t is not None:
        result["load_s"] = round(loaded_t - t0, 3)
    result["garbage_lines"] = len(parsed["garbage"])
    errors = [e for e in parsed["errors"]]
    if parsed["loaded"] is None:
        errors.append({"message": "no loaded event from daemon"})
    if killed:
        errors.append({"message": f"killed on {load['deadline_s']}s deadline"})
    if aborted:
        errors.append({"message": f"aborted: {abort.reason}"})
    if lost:
        errors.append({"message": "process group ownership lost; not signalled"})
        result["ownership_lost"] = True
        if abort is not None:
            abort.set(f"load {name!r}: process group ownership lost")
    if not parsed["unloaded"] and not killed and not aborted:
        result["warnings"].append("no unloaded event")
    not_started = {c["name"] for c in pending}
    for rid, req in parsed["requests"].items():
        if rid not in attempt_of:
            for e in req["errors"]:
                errors.append(dict(e, unknown_id=rid))
    result["errors"] = errors
    try:
        stderr_lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        stderr_lines = []
    result["stderr_hits"] = scan_stderr(stderr_lines, plan["stderr_fail_patterns"])
    result["route_proof"] = scan_stderr_require(stderr_lines, load.get("stderr_require", []))
    result["loaded_check"] = check_loaded(parsed["loaded"], load.get("loaded_expect"), load.get("loaded_expect_not"))
    for c in cells:
        cname = c["name"]
        gen_s = None
        if cname in first_commit_t and cname in done_t:
            gen_s = round(done_t[cname] - first_commit_t[cname], 3)
        not_run = None
        if cname in not_started:
            not_run = "not run: load failed" if load_failed else "not run: daemon ended/killed first"
        elif cname not in parsed["requests"]:
            not_run = "no events received for this cell"
        result["cells"][cname] = build_cell_result(parsed["requests"].get(cname), gen_s, not_run)
    atomic_write_json(out / f"{name}.json", result)
    return result


class AbortFlag:
    """Set-once cross-thread abort request (kernel fault, watcher death, signal); the first reason wins."""

    def __init__(self):
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self.reason = None

    def set(self, reason) -> bool:
        with self._lock:
            if self._ev.is_set():
                return False
            self.reason = str(reason)
            self._ev.set()
            return True

    def is_set(self) -> bool:
        return self._ev.is_set()

    def wait(self, timeout) -> bool:
        return self._ev.wait(timeout)


def _print_plan_errors(e) -> int:
    for err in e.errors:
        print(f"plan error: {err}", file=sys.stderr)
    return EXIT_USAGE


def execute_run(plan, plan_sha, plan_path, daemon, out, selected, changed, diff_entries, diff_sha, only,
                env_for=None, abort=None, extra_info=None):
    """Run every selected load in plan order (one daemon each); write <load>.json files and run.json.

    selected: {load: [cells]}. env_for(load) -> env overlay for that load's daemon (hold: fresh HOME).
    abort: AbortFlag; once set, no further load starts (each recorded as not run) and a running daemon's
    process group is killed. -> (run_info, all_ok).
    """
    out = Path(out)
    daemon_sha = sha256_file(daemon)
    t0 = time.monotonic()
    run_info: dict = {
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "daemon_path": str(daemon),
        "daemon_sha256": daemon_sha,
        "out": str(out),
        "started_utc": utc_now(),
        "only": list(only or []),
        "changed_paths_given": changed is not None,
        "diff_given": diff_entries is not None,
        # the selection inputs, so `verdict` can recompute run_if_paths/run_if_diff for loads no lane ran
        "changed_paths": changed,
        "diff_sha256": diff_sha,
        "diff_entries": diff_entries,
        "selected": {},
        "loads": {},
    }
    run_info.update(extra_info or {})
    all_ok = True
    for load in plan["loads"]:
        name = load["name"]
        if name not in selected:
            continue
        cells = selected[name]
        sel = evaluate_load(load, changed, diff_entries)
        should_run, reason = sel["run"], sel["reason"]
        run_info["selected"][name] = [c["name"] for c in cells]
        if not should_run:
            rec = {
                "load": name,
                "skipped": True,
                "reason": reason,
                "note": load["note"],
                "matched": sel["matched"],
                "cells": [c["name"] for c in cells],
            }
            atomic_write_json(out / f"{name}.json", rec)
            run_info["loads"][name] = {"skipped": True, "reason": reason, "note": load["note"], "matched": sel["matched"]}
            print(f"[{name}] SKIPPED: {reason}", flush=True)
            continue
        if abort is not None and abort.is_set():
            all_ok = False
            run_info["loads"][name] = {"skipped": False, "ok": False, "not_run": True, "note": load["note"],
                                       "reason": f"not run: {abort.reason}"}
            print(f"[{name}] NOT RUN: {abort.reason}", flush=True)
            continue
        print(f"[{name}] running {len(cells)} cell(s), deadline {load['deadline_s']}s ({reason})", flush=True)
        res = run_load(load, plan, daemon, out, cells=cells, daemon_sha256=daemon_sha, selection=sel,
                       env_extra=env_for(load) if env_for else None, abort=abort)
        ok = res["spawned"] and not res["killed_on_deadline"] and not res["aborted"] and not res["ownership_lost"]
        all_ok = all_ok and ok
        run_info["loads"][name] = {
            "skipped": False,
            "ok": ok,
            "spawned": res["spawned"],
            "exit_code": res["exit_code"],
            "killed_on_deadline": res["killed_on_deadline"],
            "aborted": res["aborted"],
            "ownership_lost": res["ownership_lost"],
            "wall_s": res["wall_s"],
            "load_s": res["load_s"],
            "note": load["note"],
            "reason": reason,
            "matched": sel["matched"],
        }
        if not res["spawned"]:
            run_info["loads"][name]["problem"] = str(res["errors"][0].get("message")) if res["errors"] else "not spawned"
        print(
            f"[{name}] exit={res['exit_code']} wall={res['wall_s']}s killed={res['killed_on_deadline']} "
            + " ".join(f"{c}:n={r['n']}" for c, r in res["cells"].items()),
            flush=True,
        )
    run_info["wall_s"] = round(time.monotonic() - t0, 3)
    run_info["ok"] = all_ok
    atomic_write_json(out / "run.json", run_info)
    return run_info, all_ok


def cmd_run(args) -> int:
    try:
        plan, plan_sha = load_plan(args.plan)
        selected = select_cells(plan, args.only)
        changed = read_changed_paths(args.changed_paths) if args.changed_paths else None
        diff_entries, diff_sha = read_diff(args.diff) if args.diff else (None, None)
    except PlanError as e:
        return _print_plan_errors(e)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    daemon = Path(args.daemon)
    if not daemon.is_file() or not os.access(daemon, os.X_OK):
        print(f"error: daemon {daemon} is not an executable file", file=sys.stderr)
        return EXIT_USAGE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, all_ok = execute_run(plan, plan_sha, args.plan, daemon, out, selected, changed, diff_entries, diff_sha, args.only)
    return 0 if all_ok else 1


def cmd_plan(args) -> int:
    try:
        plan, _ = load_plan(args.plan)
        changed = read_changed_paths(args.changed_paths) if args.changed_paths else None
        diff_entries = read_diff_entries(args.diff) if args.diff else None
    except PlanError as e:
        return _print_plan_errors(e)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    for load in plan["loads"]:
        sel = evaluate_load(load, changed, diff_entries)
        should_run, reason = sel["run"], sel["reason"]
        cells = ",".join(c["name"] for c in load["cells"])
        line = f"{'RUN ' if should_run else 'SKIP'} {load['name']:<16} cells={cells}  ({reason})"
        if load["note"]:
            line += f"  note: {load['note']}"
        _, problem = check_model(load)
        if problem:
            line += f"  {'UNRESOLVED' if problem.startswith('unresolved') else 'MODEL'}: {problem}"
        print(line)
    return 0


# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------


def lane_selection(plan, lane, changed, diff_entries):
    """-> (selected load dicts in plan order, skipped load names in plan order) for one lane."""
    names = set(lane["loads"])
    chosen, skipped = [], []
    for load in plan["loads"]:
        if load["name"] not in names:
            continue
        (chosen if evaluate_load(load, changed, diff_entries)["run"] else skipped).append(load)
    return chosen, [ld["name"] for ld in skipped]


def fabric_args(lane) -> list[str]:
    args = ["--class", "correct", "--card", lane["cards"], "--mem-gb", fmt_num(lane["mem_gb"]),
            "--hostmem-gb", fmt_num(lane["hostmem_gb"])]
    if lane["hostmem_lock"]:
        args.append("--hostmem-lock")
    args += ["--max-secs", fmt_num(lane["max_secs"])]
    return args


def cmd_lanes(args) -> int:
    try:
        plan, _ = load_plan(args.plan)
        changed = read_changed_paths(args.changed_paths) if args.changed_paths else None
        diff_entries = read_diff_entries(args.diff) if args.diff else None
    except PlanError as e:
        return _print_plan_errors(e)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    if not plan.get("lanes"):
        print("plan error: plan has no `lanes`", file=sys.stderr)
        return EXIT_USAGE
    out = []
    for lane in plan["lanes"]:
        chosen, skipped = lane_selection(plan, lane, changed, diff_entries)
        if not chosen:
            continue
        unresolved = []
        for ld in chosen:
            _, problem = check_model(ld)
            if problem and problem.startswith("unresolved"):
                unresolved.append(ld["name"])
                print(f"UNRESOLVED load {ld['name']!r}: {problem}", file=sys.stderr)
        out.append(
            {
                "name": lane["name"],
                "host": lane["host"],
                "cards": lane["cards"],
                "mem_gb": lane["mem_gb"],
                "hostmem_gb": lane["hostmem_gb"],
                "hostmem_lock": lane["hostmem_lock"],
                "max_secs": lane["max_secs"],
                "loads": [ld["name"] for ld in chosen],
                "skipped_loads": skipped,
                "unresolved_loads": unresolved,
                "fabric_args": fabric_args(lane),
            }
        )
    print(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------
# hold (runs inside a fabric hold: kernel-log watcher + the lane's loads)
# --------------------------------------------------------------------------


def _group_alive(pgid) -> bool:
    """Post-reap diagnostic only: `killpg(pgid, 0)` is NOT an ownership proof."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class KernelWatcher:
    """Kernel-log follower in its own session/process group, torn down as a GROUP.

    The child gets stdin=/dev/null, close_fds=True (no fabric fd > 2) and start_new_session=True (PGID ==
    PID). A reader thread appends every line to <out>/kernel.log and matches the fault regex; a hit, or the
    child ending while not being stopped, writes <out>/STOP and sets `abort`. Only the group this object
    launched is ever signalled.
    """

    def __init__(self, cmd, out, patterns, abort):
        self.cmd = list(cmd)
        self.out = Path(out)
        self.patterns = list(patterns)
        self.rx = compile_fault_regex(self.patterns)
        self.abort = abort
        self.proc = None
        self.pid = None
        self.pgid = None
        self.lines = 0
        self.hits: list[dict] = []
        self.gone = None
        self.lost = False
        self._thread = None
        self._stopping = False

    def start(self) -> None:
        err = open(self.out / "watcher.stderr", "ab")
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                close_fds=True, start_new_session=True,
            )
        finally:
            err.close()
        self.pid = self.proc.pid
        try:
            self.pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:  # pragma: no cover - already reaped
            self.pgid = self.proc.pid
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _write_stop(self, text) -> None:
        try:
            fd = os.open(self.out / "STOP", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def _hit(self, text) -> None:
        pat = next((p for p in self.patterns if re.search(p, text, re.IGNORECASE)), None)
        if len(self.hits) < MAX_WATCHER_HITS:
            self.hits.append({"line": text[:500], "pattern": pat})
        self._write_stop(text)
        self.abort.set(f"kernel fault: {text[:300]}")

    def _read(self) -> None:
        with open(self.out / "kernel.log", "ab", buffering=0) as log:
            for raw in iter(self.proc.stdout.readline, b""):
                log.write(raw)
                self.lines += 1
                text = raw.decode("utf-8", "replace").rstrip("\r\n")
                if self.rx.search(text):
                    self._hit(text)
        if not self._stopping:
            # the leader must stay unreaped: stop() signals its group later (a reaped PGID may be reused)
            try:
                rc = _unreaped_rc(self.proc) if _wait_exit_unreaped(self.proc, 2) else None
            except LeaderLost:
                self.lost = True
                msg = "watcher died (process group ownership lost; not signalled)"
            else:
                msg = f"watcher died (exit {rc})" if rc is not None else "watcher died (stdout closed)"
            self._write_stop(msg)
            self.abort.set(msg)

    def stop(self) -> bool:
        """Kill the watcher's process group, reap it, and verify the group is gone -> `gone`.

        The leader stays an unreaped zombie (its PID/PGID cannot be reused) until the last group signal;
        only then is it reaped. `_group_alive` runs after the reap, as a diagnostic. When the leader is
        lost (LeaderLost) nothing is signalled or probed: `lost` is set and `gone` is False (unverifiable).
        """
        if self.proc is None:
            self.gone = True
            return True
        self._stopping = True
        try:
            _killpg_owned(self.proc, self.pgid, signal.SIGTERM)
            _wait_exit_unreaped(self.proc, WATCH_TERM_GRACE_S)
            # final group signal: stray members die with the group; the leader is still unreaped
            _killpg_owned(self.proc, self.pgid, signal.SIGKILL)
        except LeaderLost:
            self.lost = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._thread is None or not self._thread.is_alive():
            # a lost leader may still be running with the reader blocked in readline: closing the buffered
            # stream under it would deadlock, so leave it to the (daemon) reader thread
            try:
                self.proc.stdout.close()
            except OSError:
                pass
        if self.lost:
            self.gone = False
            return False
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        end = time.monotonic() + 3.0
        while _group_alive(self.pgid) and time.monotonic() < end:
            time.sleep(0.02)
        self.gone = not _group_alive(self.pgid)
        return self.gone


def probe_journal():
    """None when `journalctl -k` is readable, else the reason (mirrors the hiptrx hold.sh probe)."""
    try:
        pr = subprocess.run(WATCH_PROBE_CMD, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            close_fds=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return f"{' '.join(WATCH_PROBE_CMD)}: {e}"
    text = (pr.stdout + pr.stderr).strip()
    if pr.returncode != 0 or re.search(r"insufficient permissions|No journal files|not seeing messages", text, re.I):
        return f"journalctl -k unreadable rc={pr.returncode}: {text[:300]}"
    return None


def cmd_hold(args) -> int:
    if not args.no_fabric_check and not os.environ.get("FABRIC_JOB"):
        print("error: FABRIC_JOB is not set: `hold` must run inside a fabric hold (--no-fabric-check overrides)",
              file=sys.stderr)
        return EXIT_USAGE
    watch_cmd = args.watch_cmd
    if watch_cmd is not None:
        if watch_cmd and watch_cmd[0] == "--":
            watch_cmd = watch_cmd[1:]
        if not watch_cmd:
            print("error: --watch-cmd needs a command", file=sys.stderr)
            return EXIT_USAGE
    try:
        plan, plan_sha = load_plan(args.plan)
        changed = read_changed_paths(args.changed_paths) if args.changed_paths else None
        diff_entries, diff_sha = read_diff(args.diff) if args.diff else (None, None)
    except PlanError as e:
        return _print_plan_errors(e)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    lane = next((ln for ln in plan.get("lanes") or [] if ln["name"] == args.lane), None)
    if lane is None:
        known = ", ".join(ln["name"] for ln in plan.get("lanes") or []) or "<plan has no lanes>"
        print(f"error: unknown lane {args.lane!r} (known: {known})", file=sys.stderr)
        return EXIT_USAGE
    daemon = Path(args.daemon)
    if not daemon.is_file() or not os.access(daemon, os.X_OK):
        print(f"error: daemon {daemon} is not an executable file", file=sys.stderr)
        return EXIT_USAGE
    home_models = None
    if args.home_models:
        home_models = os.path.abspath(args.home_models)
        if not os.path.isdir(home_models):
            print(f"error: --home-models {home_models} is not a directory", file=sys.stderr)
            return EXIT_USAGE
    out = Path(args.out)
    if (out / "hold.json").exists():
        print(f"error: already held: {out / 'hold.json'} exists", file=sys.stderr)
        return EXIT_USAGE
    out.mkdir(parents=True, exist_ok=True)

    chosen, skipped = lane_selection(plan, lane, changed, diff_entries)
    selected = {ld["name"]: list(ld["cells"]) for ld in chosen}
    started = utc_now()
    cards_env = {k: os.environ.get(k) for k in ("FABRIC_CARDS", "ROCR_VISIBLE_DEVICES", "HIPFIRE_DEVICES")}
    cmd_used = list(watch_cmd) if watch_cmd else list(DEFAULT_WATCH_CMD)
    hold: dict = {
        "lane": lane["name"],
        "status": "running",
        "reason": None,
        "fabric_job": os.environ.get("FABRIC_JOB"),
        "cards_env": cards_env,
        "watcher": {"cmd": cmd_used, "pid": None, "pgid": None, "lines": 0, "hits": [], "gone": True,
                    "ownership_lost": False},
        "started_utc": started,
        "ended_utc": None,
        "loads": [],
        "skipped_loads": skipped,
        "out": str(out),
        "plan_sha256": plan_sha,
    }
    atomic_write_json(out / "hold.json", hold)

    abort = AbortFlag()

    def on_signal(signum, _frame):
        abort.set(f"signalled: {signal.Signals(signum).name}")

    old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            old_handlers[sig] = signal.signal(sig, on_signal)
        except ValueError:  # not the main thread: the finally block is the only teardown path
            break

    def env_for(load):
        env = {"HIPFIRE_NO_REGISTRY_FETCH": "1", "HIPFIRE_EMIT_TOKEN_IDS": "1"}
        if home_models:
            home = out / load["name"] / "home"
            (home / ".hipfire").mkdir(parents=True, exist_ok=True)
            link = home / ".hipfire" / "models"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(home_models)
            env["HOME"] = str(home)
        return env

    watcher = None
    run_info = None
    exit_override = None
    internal_error = None
    try:
        problem = None
        if watch_cmd is None:
            problem = probe_journal()
        if problem is None:
            watcher = KernelWatcher(cmd_used, out, plan["kernel_fault_patterns"], abort)
            try:
                watcher.start()
            except OSError as e:
                watcher.proc = None
                problem = f"spawn failed: {e}"
        if problem is not None:
            exit_override = EXIT_USAGE
            abort.set(f"watcher unavailable: {problem}")
            print(f"error: watcher unavailable: {problem}", file=sys.stderr)
        else:
            abort.wait(WATCH_SETTLE_S)  # a watcher that dies at once covers no load
        run_info, _ = execute_run(plan, plan_sha, args.plan, daemon, out, selected, changed, diff_entries, diff_sha,
                                  sorted(selected), env_for=env_for, abort=abort,
                                  extra_info={"lane": lane["name"], "hold": True})
        if not abort.is_set():
            abort.wait(WATCH_FLUSH_S)  # trailing kernel lines
    except Exception as e:  # noqa: BLE001 - always tear down and record
        internal_error = e
        abort.set(f"internal error: {type(e).__name__}: {e}")
    finally:
        if watcher is not None:
            watcher.stop()
        for sig, old in old_handlers.items():
            signal.signal(sig, old if old is not None else signal.SIG_DFL)

    loads_rec = []
    deadline_bad = []
    for lname, rec in ((run_info or {}).get("loads") or {}).items():
        if rec.get("skipped"):
            continue
        if rec.get("not_run"):
            st = "not_run"
        elif not rec.get("spawned", True):
            st = "failed_before_spawn"
            deadline_bad.append(f"load {lname!r}: {rec.get('problem', 'not spawned')}")
        elif rec.get("ownership_lost"):
            st = "ownership_lost"
        elif rec.get("killed_on_deadline"):
            st = "killed_on_deadline"
            deadline_bad.append(f"load {lname!r} killed on deadline")
        elif rec.get("aborted"):
            st = "aborted"
        else:
            st = "ok"
        loads_rec.append({"load": lname, "status": st, "exit_code": rec.get("exit_code"), "wall_s": rec.get("wall_s")})
    if watcher is not None and watcher.lost and not abort.is_set():
        abort.set("watcher process group ownership lost; not signalled")
    if abort.is_set():
        status, reason, code = "incomplete", abort.reason, EXIT_HOLD_INCOMPLETE
    elif deadline_bad:
        status, reason, code = "deadline", "; ".join(deadline_bad), 1
    else:
        status, reason, code = "complete", None, 0
    if exit_override is not None:
        code = exit_override
    hold.update(status=status, reason=reason, ended_utc=utc_now(), loads=loads_rec, exit_code=code)
    if watcher is not None:
        hold["watcher"] = {"cmd": cmd_used, "pid": watcher.pid, "pgid": watcher.pgid, "lines": watcher.lines,
                           "hits": watcher.hits, "gone": bool(watcher.gone), "ownership_lost": watcher.lost}
    atomic_write_json(out / "hold.json", hold)
    print(f"hold {lane['name']}: {status}{'' if reason is None else f' ({reason})'} -> {out / 'hold.json'}", flush=True)
    if internal_error is not None:
        raise internal_error
    return code


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


def first_diff(got, pin, window: int = DIFF_CONTEXT):
    """None when equal, else {index, got, pin, got_context, pin_context}."""
    got, pin = list(got), list(pin)
    if got == pin:
        return None
    idx = next((i for i, (a, b) in enumerate(zip(got, pin)) if a != b), min(len(got), len(pin)))
    lo, hi = max(0, idx - window), idx + window + 1
    return {
        "index": idx,
        "got": got[idx] if idx < len(got) else None,
        "pin": pin[idx] if idx < len(pin) else None,
        "got_context": got[lo:hi],
        "pin_context": pin[lo:hi],
    }


def cell_status(load, cell, max_tokens, pin) -> dict:
    """Classify one cell.

    load: parsed <load>.json (None = no result); cell: its cells[CELL] record
    (None = missing); pin: pins["cells"][CELL] (None = no pin).
    -> {"status": FAIL|MATCH|CHANGED|NEW, "reasons": [...], "first_diff"?: {...}}
    """
    reasons: list[str] = []
    if load is None:
        return {"status": "FAIL", "reasons": ["no result for this load"]}
    if load.get("killed_on_deadline"):
        reasons.append("daemon killed on deadline")
    if load.get("exit_code") != 0 and load.get("spawned", True):
        reasons.append(f"daemon exit code {load.get('exit_code')!r}")
    for e in load.get("errors") or []:
        reasons.append("load error: " + str(e.get("message", e))[:200])
    for h in load.get("stderr_hits") or []:
        reasons.append(f"stderr matched {h.get('pattern')!r}: {str(h.get('line', ''))[:160]}")
    for rp in load.get("route_proof") or []:
        if not rp.get("matched"):
            reasons.append(f"required route line missing: {rp.get('pattern')}")
    for lc in load.get("loaded_check") or []:
        if not lc.get("ok"):
            want = json.dumps(lc.get("want"))
            if lc.get("op") == "expect_not":
                want = "!=" + want
            got = json.dumps(lc.get("got")) if lc.get("found") else "<missing>"
            reasons.append(f"loaded field mismatch: key={lc.get('key')} want={want} got={got}")
    if cell is None:
        reasons.append("cell missing from load result")
        return {"status": "FAIL", "reasons": reasons}
    for e in cell.get("errors") or []:
        reasons.append("cell error: " + str(e.get("message", e))[:200])
    ids = cell.get("ids") or []
    n = len(ids)
    finish = cell.get("finish")
    if n == 0:
        reasons.append("no committed tokens (n == 0)")
    elif cell.get("done_event") is None:
        reasons.append("no done event")
    elif finish == "aborted":
        reasons.append("finish_reason aborted")
    elif n < max_tokens and finish not in NATURAL_STOPS:
        reasons.append(f"short: n={n} < max_tokens={max_tokens} with finish={finish!r} (no natural stop)")
    if reasons:
        return {"status": "FAIL", "reasons": reasons}
    if pin is None:
        return {"status": "NEW", "reasons": []}
    diff = first_diff(ids, pin.get("ids") or [])
    if diff is None:
        return {"status": "MATCH", "reasons": []}
    return {
        "status": "CHANGED",
        "reasons": [f"ids differ from pin at index {diff['index']}"],
        "first_diff": diff,
    }


def overall(statuses, invariant_failures) -> str:
    statuses = list(statuses)
    if "FAIL" in statuses or invariant_failures:
        return "FAIL"
    if "CHANGED" in statuses:
        return "CHANGED"
    return "PASS"


def _plain_entry(status, reason) -> dict:
    skipped = status == "SKIPPED"
    return {
        "status": status,
        "reasons": [reason],
        "ids": [],
        "n": None if skipped else 0,
        "ids_md5": None if skipped else ids_md5([]),
        "finish": None,
        "load_s": None,
        "gen_s": None,
    }


def build_verdict(plan, results, pins, sha, results_dirs=(), plan_sha256=None, selected=None, absent=None,
                  incomplete=()) -> dict:
    """Pure verdict.

    results: {load: parsed <load>.json} merged over every results dir.
    selected: {load: [cells]} the lanes covered, or None = judge every plan cell.
    absent: None = judge only what a lane ran (--partial); else {load: evaluate_load(...)} for the
      loads no lane covered: a load the selection skips is SKIPPED, any other (and any cell of a
      covered load that its lane's --only left out) is FAIL "not run".
    incomplete: [{"lane", "reason", "loads": [...]}] for hold dirs that did not finish cleanly: every
      cell of those loads is FAIL "INCOMPLETE: <reason>" and overall is INCOMPLETE (beats FAIL).
    """
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    pin_cells = (pins or {}).get("cells", {}) if isinstance(pins, dict) else {}
    cells_out: dict[str, dict] = {}
    inc_reason = {ln: i["reason"] for i in incomplete for ln in i["loads"]}
    for load in plan["loads"]:
        lname = load["name"]
        lr = results.get(lname)
        skipped_load = isinstance(lr, dict) and lr.get("skipped")
        covered = selected is None or lname in selected
        for cell in load["cells"]:
            cname = cell["name"]
            if lname in inc_reason:
                entry = _plain_entry("FAIL", f"INCOMPLETE: {inc_reason[lname]}")
            elif skipped_load:
                entry = _plain_entry("SKIPPED", str(lr.get("reason", "skipped")))
            elif not covered:
                if absent is None:
                    continue
                ev = absent.get(lname)
                if ev is not None and not ev["run"]:
                    entry = _plain_entry("SKIPPED", ev["reason"])
                else:
                    entry = _plain_entry("FAIL", f"not run: no results directory covers load {lname!r}")
            elif selected is not None and cname not in selected[lname]:
                if absent is None:
                    continue
                entry = _plain_entry("FAIL", f"not run: cell {cname!r} was not selected in its lane (--only)")
            else:
                cr = lr.get("cells", {}).get(cname) if isinstance(lr, dict) else None
                st = cell_status(lr, cr, effective_max_tokens(plan, cell), pin_cells.get(cname))
                entry = dict(st)
                entry["ids"] = list(cr["ids"]) if cr else []
                entry["n"] = cr["n"] if cr else 0
                entry["ids_md5"] = cr["ids_md5"] if cr else ids_md5([])
                entry["finish"] = cr.get("finish") if cr else None
                entry["load_s"] = lr.get("load_s") if isinstance(lr, dict) else None
                entry["gen_s"] = cr.get("gen_s") if cr else None
            entry["load"] = lname
            if load.get("note"):
                entry["note"] = load["note"]
            cells_out[cname] = entry

    invariants: list[dict] = []
    for a, b in plan.get("invariants", []):
        ea, eb = cells_out.get(a), cells_out.get(b)
        rec = {"cells": [a, b]}
        if ea is None or eb is None:
            rec.update(status="SKIPPED", detail="cell not selected in this run")
        elif ea["status"] == "SKIPPED" or eb["status"] == "SKIPPED":
            rec.update(status="SKIPPED", detail="a side was skipped")
        elif ea["status"] == "FAIL" or eb["status"] == "FAIL":
            rec.update(status="UNCHECKED", detail="a side is FAIL")
        elif ea["ids"] == eb["ids"]:
            rec.update(status="OK", detail="ids identical", ids_md5=ea["ids_md5"])
        else:
            rec.update(
                status="INVARIANT",
                detail="ids differ between cells that must be identical",
                md5=[ea["ids_md5"], eb["ids_md5"]],
                first_diff=first_diff(ea["ids"], eb["ids"]),
            )
        invariants.append(rec)
    failures = [i for i in invariants if i["status"] == "INVARIANT"]
    incomplete_lanes = [i["lane"] for i in incomplete]
    return {
        "sha": sha,
        "overall": "INCOMPLETE" if incomplete_lanes else overall((c["status"] for c in cells_out.values()), failures),
        "cells": cells_out,
        "invariants": invariants,
        "incomplete": incomplete_lanes,
        "plan_sha256": plan_sha256,
        "results_dirs": [str(d) for d in results_dirs],
    }


def format_table(verdict) -> str:
    rows = [("cell", "status", "n", "md5", "load_s", "gen_s")]

    def f(v):
        return "-" if v is None else str(v)

    for name, c in verdict["cells"].items():
        md5 = c.get("ids_md5")
        rows.append((name, c["status"], f(c.get("n")), f(md5[:12] if md5 else None), f(c.get("load_s")), f(c.get("gen_s"))))
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    lines = ["  ".join(v.ljust(widths[i]) for i, v in enumerate(r)).rstrip() for r in rows]
    for name, c in verdict["cells"].items():
        if c["status"] in ("FAIL", "CHANGED", "SKIPPED"):
            for r in c.get("reasons", []):
                lines.append(f"  {name}: {c['status']}: {r}")
    for inv in verdict["invariants"]:
        if inv["status"] == "INVARIANT":
            lines.append(f"  INVARIANT {inv['cells'][0]} != {inv['cells'][1]}: {inv['detail']}")
    if verdict.get("incomplete"):
        lines.append("incomplete lanes: " + ", ".join(verdict["incomplete"]))
    lines.append(f"overall: {verdict['overall']}  (sha {verdict['sha']})")
    return "\n".join(lines)


def _read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanError([f"{what} {path}: {e}"])


def load_result_dirs(dirs, plan):
    """Merge lane result dirs -> (results, selected, inputs, incomplete); PlanError listing every problem.

    Each dir needs run.json. Two dirs covering the same load, or disagreeing on the
    --changed-paths / --diff they ran with, is an error. results/selected only hold plan loads.
    """
    errors: list[str] = []
    incomplete: list[dict] = []
    plan_loads = {ld["name"] for ld in plan["loads"]}
    results: dict[str, dict] = {}
    selected: dict[str, list[str]] = {}
    owner: dict[str, Path] = {}
    inputs = None
    first_dir = None
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            errors.append(f"--results {d}: not a directory")
            continue
        run_json = d / "run.json"
        if not run_json.exists():
            errors.append(f"--results {d}: no run.json (lane never finished `run`?)")
            continue
        try:
            info = _read_json(run_json, "run.json")
        except PlanError as e:
            errors.extend(e.errors)
            continue
        this = {
            "changed_paths": info.get("changed_paths"),
            "diff_sha256": info.get("diff_sha256"),
            "diff_entries": info.get("diff_entries"),
        }
        if inputs is None:
            inputs, first_dir = this, d
        else:
            a, b = this["changed_paths"], inputs["changed_paths"]
            if (sorted(a) != sorted(b)) if (isinstance(a, list) and isinstance(b, list)) else (a != b):
                errors.append(f"{d} and {first_dir} ran with different --changed-paths")
            if this["diff_sha256"] != inputs["diff_sha256"]:
                errors.append(f"{d} and {first_dir} ran with different --diff")
        dir_loads: list[str] = []
        for lname, cells in (info.get("selected") or {}).items():
            if lname not in plan_loads:
                continue
            if lname in owner:
                errors.append(f"load {lname!r} is covered by both {owner[lname]} and {d}")
                continue
            owner[lname] = d
            dir_loads.append(lname)
            selected[lname] = list(cells)
            p = d / f"{lname}.json"
            if p.exists():
                try:
                    results[lname] = _read_json(p, "load result")
                except PlanError as e:
                    errors.extend(e.errors)
        hold_json = d / "hold.json"
        if hold_json.exists():
            try:
                hold = _read_json(hold_json, "hold.json")
            except PlanError as e:
                errors.extend(e.errors)
                continue
            hstatus = hold.get("status") if isinstance(hold, dict) else None
            if hstatus not in ("complete", "deadline"):
                why = (hold.get("reason") if isinstance(hold, dict) else None) or f"hold status {hstatus!r}"
                incomplete.append({"lane": (hold.get("lane") if isinstance(hold, dict) else None) or d.name,
                                   "reason": str(why), "loads": dir_loads})
    if errors:
        raise PlanError(errors)
    return results, selected, inputs or {"changed_paths": None, "diff_sha256": None, "diff_entries": None}, incomplete


def cmd_verdict(args) -> int:
    try:
        plan, plan_sha = load_plan(args.plan)
        pins = {}
        if Path(args.pins).exists():
            pins = _read_json(args.pins, "pins")
        results, selected, inputs, incomplete = load_result_dirs(args.results, plan)
    except PlanError as e:
        for err in e.errors:
            print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE
    absent = None
    if not args.partial:
        absent = {
            ld["name"]: evaluate_load(ld, inputs["changed_paths"], inputs["diff_entries"])
            for ld in plan["loads"]
            if ld["name"] not in selected
        }
    verdict = build_verdict(plan, results, pins, args.sha, args.results, plan_sha, selected, absent, incomplete)
    atomic_write_json(args.out, verdict)
    print(format_table(verdict))
    return EXIT_FOR_OVERALL[verdict["overall"]]


# --------------------------------------------------------------------------
# pin
# --------------------------------------------------------------------------


def apply_pins(pins, verdict, accept_changed=None, now=None, allow_failed=None) -> dict:
    """Return the updated pins dict or raise PinRefused. Never mutates `pins`.

    INCOMPLETE always refuses. FAIL refuses unless `allow_failed` (a reason): then only the NEW/MATCH
    cells that are not part of a failed invariant are pinned; CHANGED cells still need `accept_changed`.
    """
    now = now or utc_now()
    reason_in = (accept_changed or "").strip()
    allow_reason = (allow_failed or "").strip()
    ov = verdict.get("overall")
    if ov == "INCOMPLETE":
        raise PinRefused("verdict overall is INCOMPLETE (a lane did not finish cleanly); refusing to pin")
    if allow_failed is not None and not allow_reason:
        raise PinRefused("--allow-failed-cells needs a non-empty REASON")
    if ov == "FAIL" and not allow_reason:
        raise PinRefused("verdict overall is FAIL; refusing to pin (--allow-failed-cells REASON pins only the passing cells)")
    cells = verdict.get("cells", {})
    failed_cells: list[str] = []
    if ov == "FAIL":
        in_invariant = {c for inv in verdict.get("invariants", []) if inv.get("status") == "INVARIANT" for c in inv.get("cells", [])}
        failed_cells = sorted(c for c, v in cells.items() if v.get("status") == "FAIL" or c in in_invariant)
    changed = sorted(c for c, v in cells.items() if v.get("status") == "CHANGED")
    if changed and not reason_in:
        raise PinRefused(
            f"verdict has CHANGED cells ({', '.join(changed)}); pass --accept-changed REASON to accept"
        )
    new = copy.deepcopy(pins) if pins else {}
    new.setdefault("version", PINS_VERSION)
    new.setdefault("cells", {})
    new.setdefault("history", [])
    for cname, v in cells.items():
        st = v.get("status")
        if st not in ("NEW", "MATCH", "CHANGED") or cname in failed_cells:
            continue  # FAIL / SKIPPED (and cells of a failed invariant) never write a pin
        new["cells"][cname] = {
            "ids": list(v["ids"]),
            "ids_md5": v["ids_md5"],
            "sha": verdict.get("sha"),
            "pinned_utc": now,
            "reason": reason_in if st == "CHANGED" else ("new" if st == "NEW" else "match"),
        }
    entry = {
        "sha": verdict.get("sha"),
        "overall": verdict.get("overall"),
        "pinned_utc": now,
        "reason": reason_in,
        "changed": changed,
    }
    if ov == "FAIL":
        entry["allow_failed_cells"] = allow_reason
        entry["failed_cells"] = failed_cells
    new["history"].append(entry)
    return new


def cmd_pin(args) -> int:
    try:
        verdict = _read_json(args.verdict, "verdict")
        pins = _read_json(args.pins, "pins") if Path(args.pins).exists() else {}
    except PlanError as e:
        for err in e.errors:
            print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE
    try:
        new = apply_pins(pins, verdict, args.accept_changed, allow_failed=args.allow_failed_cells)
    except PinRefused as e:
        print(f"pin refused: {e}", file=sys.stderr)
        return 1
    atomic_write_json(args.pins, new)
    by = {}
    for c, v in verdict.get("cells", {}).items():
        by.setdefault(v.get("status"), []).append(c)
    print("pinned: " + ", ".join(f"{s}={','.join(cs)}" for s, cs in sorted(by.items())) + f" -> {args.pins}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="post_land_smoke.py",
        description="Post-land GPU smoke runner: run a plan against the raw daemon, "
        "judge token ids against pins, update pins.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="print which loads would run and why (dry run)")
    p.add_argument("--plan", required=True, help="plan JSON")
    p.add_argument("--changed-paths", help="file with one changed repo path per line")
    p.add_argument("--diff", help="`git diff -U0 --no-color PREV LAND` output (enables run_if_diff)")
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("run", help="execute a plan (one daemon process per load)")
    r.add_argument("--plan", required=True, help="plan JSON")
    r.add_argument("--daemon", required=True, help="daemon binary")
    r.add_argument("--out", required=True, help="output directory (created)")
    r.add_argument("--only", nargs="+", metavar="LOAD_OR_CELL", help="restrict to these loads/cells")
    r.add_argument("--changed-paths", help="file with one changed repo path per line (enables run_if_paths)")
    r.add_argument("--diff", help="`git diff -U0 --no-color PREV LAND` output (enables run_if_diff)")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("verdict", help="classify run results against pins; exit 0 PASS/1 FAIL/3 CHANGED")
    v.add_argument("--results", required=True, nargs="+", metavar="DIR", help="run output director(y/ies), one per lane; each needs run.json")
    v.add_argument("--plan", required=True, help="plan JSON")
    v.add_argument("--pins", required=True, help="pins JSON (missing file = all NEW)")
    v.add_argument("--sha", required=True, help="landed commit sha")
    v.add_argument("--out", required=True, help="VERDICT.json to write")
    v.add_argument("--partial", action="store_true", help="judge only the loads/cells the dirs ran (default: loads no dir ran are FAIL 'not run' unless the filters skip them)")
    v.set_defaults(fn=cmd_verdict)

    lp = sub.add_parser("lanes", help="print the plan's fabric lanes (JSON) that have >= 1 selected load")
    lp.add_argument("--plan", required=True, help="plan JSON (needs a `lanes` list)")
    lp.add_argument("--changed-paths", help="file with one changed repo path per line")
    lp.add_argument("--diff", help="`git diff -U0 --no-color PREV LAND` output (enables run_if_diff)")
    lp.set_defaults(fn=cmd_lanes)

    h = sub.add_parser(
        "hold",
        help="inside a fabric hold: run one lane's selected loads under a kernel-log watcher; "
        "exit 0 complete / 1 load killed on deadline or not spawned / 4 incomplete (kernel fault, watcher died, signalled) / 2 usage",
    )
    h.add_argument("--plan", required=True, help="plan JSON (needs a `lanes` list)")
    h.add_argument("--lane", required=True, help="lane name")
    h.add_argument("--daemon", required=True, help="daemon binary")
    h.add_argument("--out", required=True, help="output directory (created; must not hold a hold.json yet)")
    h.add_argument("--changed-paths", help="file with one changed repo path per line (enables run_if_paths)")
    h.add_argument("--diff", help="`git diff -U0 --no-color PREV LAND` output (enables run_if_diff)")
    h.add_argument("--home-models", metavar="DIR", help="per load: HOME=OUT/<load>/home with .hipfire/models -> DIR (symlink only)")
    h.add_argument("--no-fabric-check", action="store_true", help="do not require FABRIC_JOB (tests/dry runs; never on a real GPU)")
    h.add_argument(
        "--watch-cmd",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help="kernel-log follower command (default: journalctl -k -f -n0 --no-pager -o short-iso-precise; "
        "skips the journal probe); consumes the rest of the command line, so it must come last",
    )
    h.set_defaults(fn=cmd_hold)

    k = sub.add_parser("pin", help="write verdict ids into the pins file (atomic)")
    k.add_argument("--verdict", required=True, help="VERDICT.json")
    k.add_argument("--pins", required=True, help="pins JSON to create/update")
    k.add_argument("--accept-changed", metavar="REASON", help="accept CHANGED cells, recording REASON")
    k.add_argument(
        "--allow-failed-cells",
        metavar="REASON",
        help="when the verdict is FAIL (never INCOMPLETE), still pin its NEW/MATCH cells (not FAIL, "
        "SKIPPED, CHANGED, or in a failed invariant), recording REASON and the failed cells in history",
    )
    k.set_defaults(fn=cmd_pin)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
