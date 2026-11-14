#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Promotion lane: run the required hardware rows on the installed release
archive, record the verdict, merge per-class records, and check the policy.

The gate (this script, policy.json, device_mesh_matrix.py, matrix.json) comes
from the previously promoted checkout (--gate-root); the candidate contributes
only its release archive and its in-tree waivers.json. Stdlib only.

  lane.py run --archive A.tar.gz --class C --candidate SHA --out DIR
              [--gate-root R] [--repo R] [--models-dir D] [--timeout S]
              [--digest-cache FILE] [--host NAME]
  lane.py merge --candidate SHA --out CAND.json [--allow-down C]... [--rented C]... RECORD.json...
  lane.py check-release --sha SHA --records DIR [--repo R]
  lane.py classify --repo R --range BASE..TIP
  lane.py check [--repo R]
  lane.py plan --class C
  lane.py --self-test

Exit: 0 green/ok, 1 red/problems, 2 infrastructure or usage error.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone

POLICY_SCHEMA = "promotion-lane-policy-v1"
WAIVERS_SCHEMA = "promotion-lane-waivers-v1"
RECORD_SCHEMA = "promotion-record-v1"
CANDIDATE_SCHEMA = "promotion-candidate-v1"
GATES = ("candidate", "rented", "release")
TRANSITIONS = ("forced-advance", "append", "rollback")
COMPILE_MARKERS = ("hipcc compiled (", "but hipcc is unavailable")
COMPILER_PREFIXES = ("hipcc", "hipconfig", "clang", "amdclang")
LOG_SUFFIX = re.compile(r"-(?:positive|negative)-\d+\.log$")
HEX40 = re.compile(r"[0-9a-f]{40}")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAIVERS_REL = "scripts/promotion-lane/waivers.json"
POLICY_REL = "scripts/promotion-lane/policy.json"
MATRIX_REL = "scripts/device-mesh/matrix.json"


class Usage(Exception):
    """Infrastructure or usage error: exit 2."""


# --------------------------------------------------------------------------- helpers

def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(text):
    if not isinstance(text, str):
        raise ValueError("not a string")
    when = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_str_list(value, allow_empty=False):
    return (isinstance(value, list) and (allow_empty or value)
            and all(isinstance(v, str) and v for v in value))


def git(repo, *args):
    return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)


def need_sha(text, what):
    if not HEX40.fullmatch(text or ""):
        raise Usage("%s must be a full 40-hex lowercase sha, got %r" % (what, text))


# --------------------------------------------------------------------------- policy

def validate_policy(policy):
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        return ["policy: schema must be %s" % POLICY_SCHEMA]
    problems = []
    classes = policy.get("classes")
    if not isinstance(classes, dict) or not classes:
        problems.append("policy: classes must be a non-empty object")
        classes = {}
    for name, cls in classes.items():
        if not isinstance(cls, dict):
            problems.append("policy: class %s must be an object" % name)
            continue
        if cls.get("gate") not in GATES:
            problems.append("policy: class %s gate must be one of %s" % (name, "/".join(GATES)))
        if not is_int(cls.get("budget_minutes")) or cls["budget_minutes"] <= 0:
            problems.append("policy: class %s budget_minutes must be a positive integer" % name)
        req = cls.get("required")
        if not is_str_list(req) or len(set(req)) != len(req):
            problems.append("policy: class %s required must be a non-empty list of unique row ids" % name)
        if "transitions_required" in cls and not (
                is_str_list(cls["transitions_required"], allow_empty=True)
                and set(cls["transitions_required"]) <= set(TRANSITIONS)):
            problems.append("policy: class %s transitions_required must be a subset of %s" % (name, "/".join(TRANSITIONS)))
    trans = policy.get("transitions_required")
    if not is_str_list(trans) or not set(trans) <= set(TRANSITIONS):
        problems.append("policy: transitions_required must be a non-empty subset of %s" % "/".join(TRANSITIONS))
    for key in ("waiver_grantors", "gate_policy_approvers", "gate_policy_paths"):
        if not is_str_list(policy.get(key)):
            problems.append("policy: %s must be a non-empty list of strings" % key)
    if not is_int(policy.get("waiver_max_days")) or policy["waiver_max_days"] <= 0:
        problems.append("policy: waiver_max_days must be a positive integer")
    return problems


def class_transitions(policy, cls):
    """Transitions a class's required rows must cover: its own list, else the top-level one."""
    return cls["transitions_required"] if "transitions_required" in cls else policy["transitions_required"]


def load_policy(path):
    try:
        policy = load_json(path)
    except (OSError, ValueError) as exc:
        raise Usage("cannot read policy %s: %s" % (path, exc))
    problems = validate_policy(policy)
    if problems:
        raise Usage("; ".join(problems))
    return policy


# --------------------------------------------------------------------------- rule (c)

def session_ops(row):
    ops = row.get("positive_session")
    return [op for op in ops if isinstance(op, dict)] if isinstance(ops, list) else []


def sends(row):
    return [op["send"] for op in session_ops(row) if isinstance(op.get("send"), dict)]


def evidence_problem(row, transition):
    """None when the row's positive_session structurally exercises `transition`."""
    ops = session_ops(row)
    if transition == "forced-advance":
        for send in sends(row):
            budget = send.get("max_think_tokens")
            if (send.get("type") == "generate" and send.get("thinking_enabled") is True
                    and is_int(budget) and budget > 0):
                return None
        return "no generate with thinking_enabled true and max_think_tokens > 0"
    if transition == "append":
        captured = set()
        for op in ops:
            send = op.get("send") if isinstance(op.get("send"), dict) else {}
            messages = send.get("messages") if send.get("type") == "generate" else None
            for msg in messages if isinstance(messages, list) else []:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    quoted = re.findall(r"\$CAPTURE_(\w+)", json.dumps(msg.get("content")))
                    if any(name in captured for name in quoted):
                        return None
            if isinstance(op.get("capture_text"), str) and op["capture_text"]:
                captured.add(op["capture_text"])
        return "no generate whose messages hold an assistant turn quoting an earlier capture_text as $CAPTURE_<name>"
    if transition == "rollback":
        # An attempt that decoded in full and was then aborted, judged by the
        # request after it. A daemon `reset` alone is a cold start, not the
        # rollback of an attempt.
        aborted = [op["send"].get("id") for op in ops
                   if op.get("abort_at_commit_ready") and isinstance(op.get("send"), dict)
                   and op["send"].get("type") == "generate"]
        if not aborted:
            return "no abort_at_commit_ready generate"
        pred = row.get("predicate")
        if not isinstance(pred, dict) or pred.get("kind") != "wire_transitions":
            return "predicate must be wire_transitions"
        if not set(aborted) <= set(pred.get("aborted_ids") or []):
            return "aborted generate(s) %s not listed in predicate aborted_ids" % aborted
        checks = pred.get("rollback_checks")
        if not isinstance(checks, list) or not checks or not all(
                isinstance(c, list) and len(c) == 3 for c in checks):
            return "predicate needs rollback_checks [[post, warm, cold], ...]"
        return None
    return "unknown transition %r" % (transition,)


def validate_waivers(policy, doc):
    """Schema problems, plus grantor and expiry span of every open waiver."""
    if doc is None:
        return []
    if not isinstance(doc, dict) or doc.get("schema") != WAIVERS_SCHEMA or not isinstance(doc.get("waivers"), list):
        return ["waivers: schema must be %s with a waivers list" % WAIVERS_SCHEMA]
    problems = []
    for i, waiver in enumerate(doc["waivers"]):
        tag = "waivers[%d]" % i
        if not isinstance(waiver, dict):
            problems.append("%s: not an object" % tag)
            continue
        if not HEX40.fullmatch(waiver.get("sha") or ""):
            problems.append("%s: sha must be 40-hex" % tag)
        if waiver.get("class") not in policy["classes"]:
            problems.append("%s: class %r is not a policy class" % (tag, waiver.get("class")))
        elif not is_str_list(waiver.get("rows")) or not set(waiver["rows"]) <= set(policy["classes"][waiver["class"]]["required"]):
            problems.append("%s: rows must be a non-empty subset of the class's required rows" % tag)
        for key in ("reason", "granted_by", "rollback"):
            if not isinstance(waiver.get(key), str) or not waiver[key]:
                problems.append("%s: %s must be a non-empty string" % (tag, key))
        if "closed_by" not in waiver or not (waiver["closed_by"] is None or isinstance(waiver["closed_by"], str)):
            problems.append("%s: closed_by must be null or a string" % tag)
        try:
            granted, expires = parse_utc(waiver.get("granted_utc")), parse_utc(waiver.get("expires_utc"))
        except (ValueError, TypeError):
            problems.append("%s: granted_utc/expires_utc must be ISO timestamps" % tag)
            continue
        if waiver.get("closed_by") is None:
            if waiver.get("granted_by") not in policy["waiver_grantors"]:
                problems.append("%s: granted_by %r is not in waiver_grantors" % (tag, waiver.get("granted_by")))
            if not timedelta(0) < expires - granted <= timedelta(days=policy["waiver_max_days"]):
                problems.append("%s: expiry span must be within (0, %d] days" % (tag, policy["waiver_max_days"]))
    return problems


def check_tree(policy, matrix, waivers_doc):
    """CPU checks: policy rows exist for their class, every gating class covers
    the required transitions, every claimed transition is structurally evidenced."""
    problems = validate_policy(policy)
    if problems:
        return problems
    rows = {}
    for row in matrix.get("rows", []):
        if row.get("id") in rows:
            problems.append("matrix: duplicate row id %s" % row.get("id"))
        rows[row.get("id")] = row
    for name, cls in sorted(policy["classes"].items()):
        covered = set()
        for rid in cls["required"]:
            row = rows.get(rid)
            if row is None:
                problems.append("class %s: required row %s is not in matrix.json" % (name, rid))
                continue
            if name not in row.get("hosts", []):
                problems.append("class %s: required row %s does not list %s in hosts" % (name, rid, name))
            if isinstance(row.get("transitions"), list):
                covered.update(row["transitions"])
        if cls["gate"] in ("candidate", "rented"):
            missing = [t for t in class_transitions(policy, cls) if t not in covered]
            if missing:
                problems.append("class %s: required rows do not cover transitions %s" % (name, ",".join(missing)))
    if not any(set(policy["transitions_required"]) <= set(class_transitions(policy, c))
               for c in policy["classes"].values() if c["gate"] == "candidate"):
        problems.append("no candidate-gate class requires all of %s" % ",".join(policy["transitions_required"]))
    for rid, row in rows.items():
        if "transitions" not in row:
            continue
        claims = row["transitions"]
        if not isinstance(claims, list) or not set(claims) <= set(TRANSITIONS):
            problems.append("row %s: transitions must be a subset of %s" % (rid, "/".join(TRANSITIONS)))
            continue
        for transition in claims:
            why = evidence_problem(row, transition)
            if why:
                problems.append("row %s claims %s but %s" % (rid, transition, why))
    return problems + validate_waivers(policy, waivers_doc)


# --------------------------------------------------------------------------- waivers

def waiver_applies(waiver, policy, cls, candidate, repo, now):
    """(ok, reason): is this waiver valid for `cls` at `candidate` right now?"""
    if not isinstance(waiver, dict) or waiver.get("closed_by") is not None:
        return False, "closed or malformed"
    if waiver.get("class") != cls:
        return False, "class %r != %s" % (waiver.get("class"), cls)
    if waiver.get("granted_by") not in policy["waiver_grantors"]:
        return False, "granted_by %r is not a waiver grantor" % (waiver.get("granted_by"),)
    try:
        granted, expires = parse_utc(waiver.get("granted_utc")), parse_utc(waiver.get("expires_utc"))
    except (ValueError, TypeError):
        return False, "unparseable timestamps"
    if not timedelta(0) < expires - granted <= timedelta(days=policy["waiver_max_days"]):
        return False, "expiry span exceeds %d days" % policy["waiver_max_days"]
    if not now < expires:
        return False, "expired %s" % waiver["expires_utc"]
    sha = waiver.get("sha")
    if not HEX40.fullmatch(sha or "") or git(repo, "merge-base", "--is-ancestor", sha, candidate).returncode != 0:
        return False, "sha %s is not an ancestor of the candidate" % sha
    return True, ""


def candidate_waivers(repo, sha):
    """Waivers shipped in the candidate tree; absent or unreadable = none."""
    shown = git(repo, "show", "%s:%s" % (sha, WAIVERS_REL))
    if shown.returncode != 0:
        return []
    try:
        doc = json.loads(shown.stdout)
    except ValueError:
        print("warning: candidate waivers.json is not JSON; ignored", file=sys.stderr)
        return []
    waivers = doc.get("waivers") if isinstance(doc, dict) and doc.get("schema") == WAIVERS_SCHEMA else None
    return waivers if isinstance(waivers, list) else []


# --------------------------------------------------------------------------- run

def clean_path(path_value, shim_dir):
    """PATH without any directory that offers a compiler. Such a directory is
    replaced by a shim of symlinks to its non-compiler entries, so `sh` and
    friends keep resolving; the result never contains a compiler name."""
    entries = []
    for i, entry in enumerate(p for p in path_value.split(os.pathsep) if p):
        try:
            names = os.listdir(entry)
        except OSError:
            continue
        if not any(n.startswith(COMPILER_PREFIXES) for n in names):
            entries.append(entry)
            continue
        shim = os.path.join(shim_dir, str(i))
        os.makedirs(shim, exist_ok=True)
        for name in names:
            if not name.startswith(COMPILER_PREFIXES):
                with contextlib.suppress(OSError):
                    os.symlink(os.path.join(entry, name), os.path.join(shim, name))
        entries.append(shim)
    cleaned = os.pathsep.join(entries)
    if shutil.which("hipcc", path=cleaned) is not None:
        raise Usage("hipcc still resolvable on the cleaned PATH")
    return cleaned


def extract_archive(archive, dest, cls):
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise Usage("unsafe archive member %r" % member.name)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)
    for rel in ("bin/hipfire", "bin/daemon"):
        if not os.path.isfile(os.path.join(dest, rel)):
            raise Usage("archive lacks %s" % rel)
    compiled = os.path.join(dest, "bin", "kernels", "compiled")
    arches = [d for d in sorted(os.listdir(compiled))] if os.path.isdir(compiled) else []
    want = [cls] if cls.startswith("gfx") else arches
    if not want or not all(os.path.isdir(os.path.join(compiled, a)) and os.listdir(os.path.join(compiled, a))
                           for a in want):
        raise Usage("archive lacks non-empty bin/kernels/compiled/%s" % (cls if cls.startswith("gfx") else "<arch>"))


def scan_markers(matrix_dir, row_ids):
    found = {rid: [] for rid in row_ids}
    logs = os.path.join(matrix_dir, "logs")
    for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
        if not name.endswith(".log"):
            continue
        rid = LOG_SUFFIX.sub("", name) if LOG_SUFFIX.search(name) else name[:-4]
        if rid not in found:
            continue
        with open(os.path.join(logs, name), errors="replace") as fh:
            text = fh.read()
        for marker in COMPILE_MARKERS:
            if marker in text:
                found[rid].append("%s: %s" % (name, marker))
    return found


def row_records(required, summary, markers):
    by_id = {e.get("id"): e for e in (summary or {}).get("rows", []) if isinstance(e, dict)}
    rows = []
    for rid in required:
        entry = by_id.get(rid)
        disposition = entry.get("disposition") if entry else "missing"
        if markers.get(rid) or disposition == "failed":
            verdict = "failed"
        elif disposition == "current":
            verdict = "current"
        else:
            verdict = "skipped"
        note = entry.get("note", "") if entry else "row missing from summary"
        if markers.get(rid):
            note = "runtime compile: %s | %s" % ("; ".join(markers[rid]), note)
        rows.append({"id": rid, "required": True, "disposition": disposition, "verdict": verdict,
                     "elapsed_s": entry.get("elapsed_s", 0) if entry else 0, "note": note,
                     "receipt": entry.get("receipt", "") if entry else "", "compile_markers": markers.get(rid, [])})
    return rows


def verified_arch(rows):
    for row in rows:
        try:
            arch = load_json(row["receipt"]).get("platform", {}).get("gpu_arch_verified")
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        if arch:
            return arch
    return None


def cmd_run(args):
    gate = os.path.abspath(args.gate_root)
    policy = load_policy(os.path.join(gate, POLICY_REL))
    cls = policy["classes"].get(args.cls)
    if cls is None:
        raise Usage("class %s is not in policy (%s)" % (args.cls, ",".join(sorted(policy["classes"]))))
    need_sha(args.candidate, "--candidate")
    runner = os.path.join(gate, "scripts", "device_mesh_matrix.py")
    if not os.path.isfile(runner):
        raise Usage("gate root lacks scripts/device_mesh_matrix.py")
    head = git(gate, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise Usage("gate root is not a git checkout: %s" % head.stderr.strip())
    repo = os.path.abspath(args.repo or gate)
    archive = os.path.abspath(args.archive)
    if not os.path.isfile(archive):
        raise Usage("archive %s not found" % archive)

    archive_sha = sha256_file(archive)
    sidecar = archive + ".sha256"
    sidecar_ok = None
    if os.path.isfile(sidecar):
        words = open(sidecar).read().split()
        sidecar_ok = bool(words) and words[0].lower() == archive_sha
        if not sidecar_ok:
            raise Usage("archive sha256 %s does not match its sidecar %s" % (archive_sha, sidecar))

    out = os.path.abspath(args.out)
    for sub in ("install", "home", "kernel-cache", "tmp", "pathbin", "matrix"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    for stale in ("record.json", "matrix-runner.log"):
        with contextlib.suppress(OSError):
            os.remove(os.path.join(out, stale))
    install, matrix_out = (os.path.join(out, "install"), os.path.join(out, "matrix"))
    home, cache, tmp = (os.path.join(out, "home"), os.path.join(out, "kernel-cache"), os.path.join(out, "tmp"))
    for path in (install, matrix_out, home, cache, tmp):
        os.makedirs(path)
    extract_archive(archive, install, args.cls)

    env = dict(os.environ)
    env.pop("HIPFIRE_HOME", None)
    env["PATH"] = clean_path(env.get("PATH", ""), os.path.join(out, "pathbin"))
    env.update({"HIPFIRE_NO_DEVICE_COMPILER": "1",
                "HIPFIRE_CLI_BIN": os.path.join(install, "bin", "hipfire"),
                "HIPFIRE_DAEMON_BIN": os.path.join(install, "bin", "daemon"),
                "HOME": home, "MATRIX_ISOLATED_HOME": home, "HIPFIRE_KERNEL_CACHE": cache, "TMPDIR": tmp})
    if args.models_dir:
        env["HIPFIRE_MODELS_DIR"] = os.path.abspath(args.models_dir)

    cache_file = os.path.join(matrix_out, ".digest-cache.json")
    if args.digest_cache and os.path.isfile(args.digest_cache):
        shutil.copyfile(args.digest_cache, cache_file)

    required = cls["required"]
    cmd = [sys.executable, runner, "--host-class", args.cls, "--rows", ",".join(required), "--out", matrix_out]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    started = utc_now()
    with open(os.path.join(out, "matrix-runner.log"), "w") as log:
        proc = subprocess.run(cmd, cwd=gate, env=env, stdout=log, stderr=subprocess.STDOUT)
    finished = utc_now()
    if args.digest_cache and os.path.isfile(cache_file):
        staged = args.digest_cache + ".partial"
        shutil.copyfile(cache_file, staged)
        os.replace(staged, args.digest_cache)
    if proc.returncode not in (0, 1):
        raise Usage("matrix runner exited %d (see %s)" % (proc.returncode, os.path.join(out, "matrix-runner.log")))

    try:
        summary = load_json(os.path.join(matrix_out, "summary.json"))
    except (OSError, ValueError):
        summary = None
    rows = row_records(required, summary, scan_markers(matrix_out, required))

    waivers = []
    now = utc_now()
    for row in rows:
        if row["verdict"] in ("current", "waived"):
            continue
        for waiver in candidate_waivers(repo, args.candidate):
            ok, _ = waiver_applies(waiver, policy, args.cls, args.candidate, repo, now)
            if ok and row["id"] in waiver.get("rows", []):
                row["verdict"] = "waived"
                if waiver not in waivers:
                    waivers.append(waiver)
                break

    wall = round((finished - started).total_seconds(), 1)
    green = all(r["verdict"] in ("current", "waived") for r in rows)
    record = {"schema": RECORD_SCHEMA, "candidate": args.candidate, "class": args.cls,
              "host": args.host or socket.gethostname(), "gpu_arch_verified": verified_arch(rows),
              "archive": {"file": archive, "sha256": archive_sha, "sidecar_ok": sidecar_ok},
              "gate": {"root": gate, "sha": head.stdout.strip(),
                       "policy_sha256": sha256_file(os.path.join(gate, POLICY_REL)),
                       "matrix_sha256": sha256_file(os.path.join(gate, MATRIX_REL))},
              "no_compiler": {"path_hipcc": shutil.which("hipcc", path=env["PATH"]),
                              "HIPFIRE_NO_DEVICE_COMPILER": env["HIPFIRE_NO_DEVICE_COMPILER"]},
              "rows": rows, "waivers": waivers,
              "started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "wall_s": wall,
              "budget_minutes": cls["budget_minutes"], "over_budget": wall > cls["budget_minutes"] * 60,
              "verdict": "green" if green else "red"}
    write_json(os.path.join(out, "record.json"), record)
    print("LANE %s %s %s (%d rows, %.1f min)" % (args.cls, "GREEN" if green else "RED",
                                                args.candidate[:12], len(rows), wall / 60))
    for row in rows:
        if row["verdict"] != "current":
            print("  %s: %s (%s) %s" % (row["id"], row["verdict"], row["disposition"], row["note"]))
    return 0 if green else 1


# --------------------------------------------------------------------------- records

def record_problems(rec, policy):
    """Integrity problems of one record against the policy; [] when sound."""
    if not isinstance(rec, dict) or rec.get("schema") != RECORD_SCHEMA:
        return ["not a %s" % RECORD_SCHEMA]
    cls = policy["classes"].get(rec.get("class"))
    if cls is None:
        return ["class %r is not a policy class" % (rec.get("class"),)]
    problems = []
    if not HEX40.fullmatch(rec.get("candidate") or ""):
        problems.append("candidate is not a 40-hex sha")
    by_id = {r.get("id"): r for r in rec.get("rows", []) if isinstance(r, dict)}
    for rid in cls["required"]:
        if rid not in by_id:
            problems.append("required row %s absent" % rid)
        elif by_id[rid].get("verdict") not in ("current", "waived"):
            problems.append("row %s verdict %s" % (rid, by_id[rid].get("verdict")))
    nc = rec.get("no_compiler") if isinstance(rec.get("no_compiler"), dict) else {}
    if nc.get("path_hipcc") is not None or nc.get("HIPFIRE_NO_DEVICE_COMPILER") != "1":
        problems.append("run was not compiler-free")
    arch = rec.get("gpu_arch_verified")
    if rec["class"].startswith("gfx") and arch is not None and arch != rec["class"]:
        problems.append("gpu_arch_verified %s != class %s" % (arch, rec["class"]))
    arch_info = rec.get("archive") if isinstance(rec.get("archive"), dict) else {}
    if not arch_info.get("sha256") or arch_info.get("sidecar_ok") is False:
        problems.append("archive identity missing or sidecar mismatch")
    gate = rec.get("gate") if isinstance(rec.get("gate"), dict) else {}
    if not gate.get("sha"):
        problems.append("gate sha missing")
    computed = "green" if not problems else "red"
    if rec.get("verdict") != computed:
        problems.append("record verdict %r but its rows say %s" % (rec.get("verdict"), computed))
    return problems


def record_status(rec, policy):
    """('green'|'red', problems): a record is green only when its own verdict is
    green and nothing in it contradicts that."""
    problems = record_problems(rec, policy)
    ok = not problems and rec.get("verdict") == "green"
    if not ok and not problems:
        problems = ["record verdict is red"]
    return ("green" if ok else "red"), problems


def cmd_merge(args, policy):
    need_sha(args.candidate, "--candidate")
    for cls in list(args.allow_down) + list(args.rented):
        if cls not in policy["classes"]:
            raise Usage("class %s is not in policy" % cls)
    problems, records = [], {}
    for path in args.records:
        try:
            rec = load_json(path)
        except (OSError, ValueError) as exc:
            problems.append("%s: unreadable (%s)" % (path, exc))
            continue
        cls = rec.get("class") if isinstance(rec, dict) else None
        if not isinstance(rec, dict) or rec.get("schema") != RECORD_SCHEMA:
            problems.append("%s: not a %s" % (path, RECORD_SCHEMA))
        elif rec.get("candidate") != args.candidate:
            problems.append("%s: candidate %s != %s" % (path, rec.get("candidate"), args.candidate))
        elif cls not in policy["classes"]:
            problems.append("%s: class %r is not a policy class" % (path, cls))
        elif cls in records:
            problems.append("%s: second record for class %s" % (path, cls))
        else:
            records[cls] = (path, rec)
    classes, owed, open_waivers = {}, [], []
    for name, spec in sorted(policy["classes"].items()):
        required = spec["gate"] == "candidate" or (spec["gate"] == "rented" and name in args.rented)
        if name in records:
            path, rec = records[name]
            status, why = record_status(rec, policy)
            problems += ["class %s: %s" % (name, w) for w in why] if required else []
            classes[name] = {"status": status, "record": path}
            open_waivers += [w for w in rec.get("waivers", []) if isinstance(w, dict) and w.get("closed_by") is None]
        elif name in args.allow_down:
            classes[name] = {"status": "down", "record": None}
            owed.append(name)
        else:
            classes[name] = {"status": "not-run", "record": None}
            if required:
                problems.append("class %s: no record and not --allow-down" % name)
        if required and classes[name]["status"] == "red" and not any(p.startswith("class %s:" % name) for p in problems):
            problems.append("class %s: red" % name)
    green = not problems
    cand = {"schema": CANDIDATE_SCHEMA, "candidate": args.candidate, "classes": classes,
            "waivers_open": open_waivers, "verdict": "green" if green else "red", "owed": owed,
            "problems": problems}
    write_json(args.out, cand)
    print("CANDIDATE %s %s (%s)" % ("GREEN" if green else "RED", args.candidate[:12],
                                    ", ".join("%s=%s" % (k, v["status"]) for k, v in classes.items())))
    if owed:
        print("  owed: %s" % ",".join(owed))
    for problem in problems:
        print("  %s" % problem)
    return 0 if green else 1


def cmd_check_release(args, policy):
    need_sha(args.sha, "--sha")
    repo = os.path.abspath(args.repo or ROOT)
    if git(repo, "cat-file", "-e", args.sha + "^{commit}").returncode != 0:
        raise Usage("commit %s not found in %s" % (args.sha, repo))
    problems, by_class = [], {}
    try:
        names = sorted(n for n in os.listdir(args.records) if n.endswith(".json"))
    except OSError as exc:
        raise Usage("cannot list records dir: %s" % exc)
    for name in names:
        path = os.path.join(args.records, name)
        try:
            rec = load_json(path)
        except ValueError as exc:
            problems.append("%s: unreadable (%s)" % (name, exc))
            continue
        if isinstance(rec, dict) and rec.get("schema") == RECORD_SCHEMA and rec.get("candidate") == args.sha:
            by_class.setdefault(rec.get("class"), []).append((name, rec))
    for cls in sorted(policy["classes"]):
        verdicts = []
        for name, rec in by_class.get(cls, []):
            status, why = record_status(rec, policy)
            if status == "green" and (rec.get("waivers") or any(r.get("verdict") == "waived" for r in rec.get("rows", []))):
                status, why = "red", ["record is waived, release needs an unwaived record"]
            verdicts.append((status, name, why))
        if not any(v[0] == "green" for v in verdicts):
            detail = "; ".join("%s: %s" % (n, ", ".join(w)) for s, n, w in verdicts) or "no record"
            problems.append("class %s: no green unwaived record at %s (%s)" % (cls, args.sha[:12], detail))
    shown = git(repo, "show", "%s:%s" % (args.sha, WAIVERS_REL))
    if shown.returncode == 0:
        try:
            doc = json.loads(shown.stdout)
            waivers = doc["waivers"]
        except (ValueError, KeyError, TypeError):
            doc, waivers = None, []
            problems.append("waivers.json at %s is malformed" % args.sha[:12])
        for waiver in waivers:
            if isinstance(waiver, dict) and waiver.get("closed_by") is None:
                problems.append("open waiver in release tree: %s rows=%s sha=%s" % (
                    waiver.get("class"), ",".join(waiver.get("rows", [])), str(waiver.get("sha"))[:12]))
    print("RELEASE %s %s" % ("GREEN" if not problems else "RED", args.sha[:12]))
    for problem in problems:
        print("  %s" % problem)
    return 0 if not problems else 1


# --------------------------------------------------------------------------- classify, check, plan

def glob_regex(pattern):
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            c = pattern[i]
            out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def classify_paths(policy, paths):
    patterns = [glob_regex(p) for p in policy["gate_policy_paths"]]
    labelled = [("policy" if any(p.match(path) for p in patterns) else "code", path) for path in paths]
    kinds = {k for k, _ in labelled}
    kind = "mixed" if len(kinds) == 2 else (kinds.pop() if kinds else "code")
    return kind, labelled


def cmd_classify(args, policy):
    diff = git(args.repo, "diff", "--name-only", "--no-renames", args.range)
    if diff.returncode != 0:
        raise Usage("git diff %s failed: %s" % (args.range, diff.stderr.strip()))
    kind, labelled = classify_paths(policy, [p for p in diff.stdout.splitlines() if p])
    print(kind)
    for label, path in labelled:
        print("%s %s" % (label, path))
    return 0


def cmd_check(args):
    repo = os.path.abspath(args.repo or ROOT)
    policy = load_policy(os.path.join(repo, POLICY_REL))
    try:
        matrix = load_json(os.path.join(repo, MATRIX_REL))
    except (OSError, ValueError) as exc:
        raise Usage("cannot read %s: %s" % (MATRIX_REL, exc))
    try:
        waivers = load_json(os.path.join(repo, WAIVERS_REL))
    except OSError:
        waivers = None
    except ValueError as exc:
        print("waivers.json: not JSON (%s)" % exc)
        return 1
    problems = check_tree(policy, matrix, waivers)
    for problem in problems:
        print(problem)
    if not problems:
        print("promotion-lane check: OK (%d classes)" % len(policy["classes"]))
    return 1 if problems else 0


def cmd_plan(args, policy):
    cls = policy["classes"].get(args.cls)
    if cls is None:
        raise Usage("class %s is not in policy" % args.cls)
    try:
        rows = {r["id"]: r for r in load_json(os.path.join(ROOT, MATRIX_REL))["rows"]}
    except (OSError, ValueError, KeyError) as exc:
        raise Usage("cannot read %s: %s" % (MATRIX_REL, exc))
    print("class %s gate=%s budget=%dm" % (args.cls, cls["gate"], cls["budget_minutes"]))
    missing = 0
    for rid in cls["required"]:
        row = rows.get(rid)
        if row is None:
            print("  %s MISSING from matrix.json" % rid)
            missing += 1
            continue
        host = "" if args.cls in row.get("hosts", []) else " HOST-NOT-LISTED"
        print("  %s [%s] transitions=%s%s" % (rid, row.get("milestone", "?"),
                                            ",".join(row.get("transitions", [])) or "-", host))
        missing += bool(host)
    print("transitions_required: %s" % ",".join(class_transitions(policy, cls)))
    print("rows: %s" % ",".join(cls["required"]))
    return 1 if missing else 0


# --------------------------------------------------------------------------- self-test

STUB_RUNNER = r'''
import json, os, shutil, sys
args = sys.argv[1:]
rows = args[args.index("--rows") + 1].split(",")
out = args[args.index("--out") + 1]
scen = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario.json")))
cache = os.path.join(out, ".digest-cache.json")
obs = {k: os.environ.get(k) for k in ("HIPFIRE_NO_DEVICE_COMPILER", "HIPFIRE_CLI_BIN", "HIPFIRE_DAEMON_BIN", "HOME",
                                       "MATRIX_ISOLATED_HOME", "HIPFIRE_KERNEL_CACHE", "TMPDIR", "HIPFIRE_MODELS_DIR")}
obs["which_hipcc"] = shutil.which("hipcc")
obs["which_sh"] = shutil.which("sh")
obs["rows_arg"] = rows
obs["host_class"] = args[args.index("--host-class") + 1]
obs["cache_in"] = json.load(open(cache)) if os.path.exists(cache) else None
json.dump(obs, open(scen["obs"], "w"))
if scen.get("exit") not in (None, 0, 1):
    sys.exit(scen["exit"])
os.makedirs(os.path.join(out, "logs")); os.makedirs(os.path.join(out, "receipts"))
summary = {"schema": "device-mesh-matrix-summary-v1", "rows": []}
for rid in rows:
    disp = scen["rows"].get(rid, "current")
    if disp == "missing":
        continue
    receipt = os.path.join(out, "receipts", rid + ".json")
    json.dump({"platform": {"gpu_arch_verified": scen.get("arch", "gfx1100")}}, open(receipt, "w"))
    summary["rows"].append({"id": rid, "milestone": "L", "disposition": disp, "note": "stub", "elapsed_s": 1.0,
                            "receipt": receipt})
    open(os.path.join(out, "logs", rid + "-positive-0.log"), "w").write(scen.get("logs", {}).get(rid, "ok\n"))
json.dump(summary, open(os.path.join(out, "summary.json"), "w"))
json.dump({"x": 1, "y": 2}, open(cache, "w"))
sys.exit(0 if all(r["disposition"] == "current" for r in summary["rows"]) and len(summary["rows"]) == len(rows) else 1)
'''

SELF_POLICY = {
    "schema": POLICY_SCHEMA,
    "classes": {
        "gfx1100": {"gate": "candidate", "budget_minutes": 60, "required": ["rowA", "rowB"]},
        "gfx1201": {"gate": "candidate", "budget_minutes": 60, "required": ["rowA", "rowB"]},
        "gfx942": {"gate": "rented", "budget_minutes": 60, "required": ["rowA", "rowB"]},
    },
    "transitions_required": ["forced-advance", "append", "rollback"],
    "waiver_grantors": ["Kaden", "fivetide"],
    "waiver_max_days": 7,
    "gate_policy_approvers": ["Kaden", "fivetide"],
    "gate_policy_paths": ["scripts/promotion-lane/**", "scripts/device_mesh_matrix.py", "scripts/device-mesh/**",
                          "scripts/ci-gates.sh"],
}


def _gen(rid, **extra):
    return {"send": dict({"type": "generate", "id": rid, "attempt_id": 1, "prompt": "hi", "max_tokens": 8}, **extra),
            "handshake": True, "capture_text": rid, "expect_terminal": "done"}


def _matrix(**overrides):
    """A synthetic matrix whose rows satisfy SELF_POLICY; overrides replace rows by id."""
    forced = {"id": "rowA", "hosts": ["gfx1100", "gfx1201", "gfx942"], "transitions": ["forced-advance", "append"],
              "positive_session": [
                  _gen("t0", thinking_enabled=True, max_think_tokens=64),
                  {"send": {"type": "generate", "id": "t1", "attempt_id": 1, "max_tokens": 8, "messages": [
                      {"role": "user", "content": "hi"}, {"role": "assistant", "content": "$CAPTURE_t0"},
                      {"role": "user", "content": "more"}]}, "handshake": True}]}
    rollback = {"id": "rowB", "hosts": ["gfx1100", "gfx1201", "gfx942"], "transitions": ["rollback"],
                "positive_session": [_gen("a"), {"send": {"type": "generate", "id": "b", "attempt_id": 1,
                                                          "max_tokens": 8, "prompt": "hi"},
                                                 "abort_at_commit_ready": True}, _gen("c")],
                "predicate": {"kind": "wire_transitions", "identical_ids": [["a", "c"]],
                              "aborted_ids": ["b"], "rollback_checks": [["c", "a", "a"]]}}
    rows = {"rowA": forced, "rowB": rollback}
    rows.update(overrides)
    return {"rows": [r for r in rows.values() if r is not None]}


class Counter:
    def __init__(self):
        self.n, self.failed = 0, []

    def __call__(self, cond, msg):
        self.n += 1
        if not cond:
            self.failed.append(msg)


def _call(fn, *a):
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(*a)
    except Usage as exc:
        rc = 2
        err.write(str(exc))
    return rc, out.getvalue(), err.getvalue()


def _git_commit(repo, message):
    cfg = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git"] + cfg + ["-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git"] + cfg + ["-C", repo, "commit", "-q", "-m", message], check=True, capture_output=True)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def _waiver(sha, cls="gfx1100", rows=("rowB",), by="Kaden", hours=2, span_days=2, closed=None):
    now = utc_now()
    granted = now - timedelta(hours=hours)
    return {"sha": sha, "class": cls, "rows": list(rows), "reason": "r", "granted_by": by,
            "granted_utc": granted.isoformat(), "expires_utc": (granted + timedelta(days=span_days)).isoformat(),
            "rollback": "revert", "closed_by": closed}


def _self_test_check(t):
    ok_matrix = _matrix()
    t(check_tree(SELF_POLICY, ok_matrix, None) == [], "valid synthetic tree must pass check")
    t(check_tree(SELF_POLICY, ok_matrix, {"schema": WAIVERS_SCHEMA, "waivers": []}) == [], "empty waivers pass")
    t(validate_policy(json.loads(json.dumps(SELF_POLICY))) == [], "self policy valid")
    t(validate_policy({"schema": "nope"}) != [], "bad policy schema rejected")
    bad = json.loads(json.dumps(SELF_POLICY))
    bad["classes"]["gfx1100"]["gate"] = "sometimes"
    t(check_tree(bad, ok_matrix, None) != [], "bad gate value rejected")
    # (c) a class loses the row that carries a transition
    dropped = json.loads(json.dumps(SELF_POLICY))
    dropped["classes"]["gfx1201"]["required"] = ["rowA"]
    t(any("gfx1201" in p and "rollback" in p for p in check_tree(dropped, ok_matrix, None)),
      "dropping the rollback row from a class must fail check")
    override = json.loads(json.dumps(SELF_POLICY))
    override["classes"]["gfx942"]["transitions_required"] = []
    override["classes"]["gfx942"]["required"] = ["rowA"]
    t(check_tree(override, ok_matrix, None) == [], "a class-level empty transitions_required needs no transition rows")
    override["classes"]["gfx1201"]["transitions_required"] = ["rollback"]
    t(check_tree(override, ok_matrix, None) == [], "another candidate class still requires all three")
    override["classes"]["gfx1100"]["transitions_required"] = ["append"]
    t(any("no candidate-gate class requires" in p for p in check_tree(override, ok_matrix, None)),
      "no candidate-gate class requiring all three transitions must fail check")
    override["classes"]["gfx1100"]["transitions_required"] = ["bogus"]
    t(validate_policy(override) != [], "class-level transitions_required outside the known set is rejected")
    override["classes"]["gfx1100"]["transitions_required"] = ["rollback"]
    override["classes"]["gfx1201"]["required"] = ["rowA"]
    t(any("gfx1201" in p and "rollback" in p for p in check_tree(override, ok_matrix, None)),
      "a class-level list is still enforced against its own rows")
    t(any("rowB is not in matrix" in p for p in check_tree(SELF_POLICY, _matrix(rowB=None), None)),
      "required row missing from matrix must fail")
    narrow = _matrix(rowB=dict(_matrix()["rows"][1], hosts=["gfx1100"]))
    t(any("gfx1201" in p and "hosts" in p for p in check_tree(SELF_POLICY, narrow, None)),
      "required row not listing the class in hosts must fail")
    # rollback claim without an aborted attempt
    nr = _matrix(rowB=dict(_matrix()["rows"][1], positive_session=[_gen("a"), _gen("c")]))
    t(any("rowB claims rollback" in p for p in check_tree(SELF_POLICY, nr, None)),
      "rollback claim without abort_at_commit_ready must fail")
    reset = _matrix(rowB=dict(_matrix()["rows"][1], positive_session=[_gen("a"), {"send": {"type": "reset"}}, _gen("c")]))
    t(any("rowB claims rollback" in p for p in check_tree(SELF_POLICY, reset, None)),
      "a reset alone is not the rollback of an attempt")
    nid = _matrix(rowB=dict(_matrix()["rows"][1], predicate=dict(_matrix()["rows"][1]["predicate"], rollback_checks=[])))
    t(any("rowB claims rollback" in p for p in check_tree(SELF_POLICY, nid, None)),
      "rollback claim with empty rollback_checks must fail")
    nab = _matrix(rowB=dict(_matrix()["rows"][1], predicate=dict(_matrix()["rows"][1]["predicate"], aborted_ids=[])))
    t(any("rowB claims rollback" in p for p in check_tree(SELF_POLICY, nab, None)),
      "an aborted generate missing from aborted_ids must fail")
    # forced advance with a zero think budget
    row_a = _matrix()["rows"][0]
    zero = _matrix(rowA=dict(row_a, positive_session=[_gen("t0", thinking_enabled=True, max_think_tokens=0)] + row_a["positive_session"][1:]))
    t(any("rowA claims forced-advance" in p for p in check_tree(SELF_POLICY, zero, None)),
      "forced-advance with max_think_tokens 0 must fail")
    # append without an assistant $CAPTURE_ turn
    literal = json.loads(json.dumps(row_a))
    literal["positive_session"][1]["send"]["messages"][1]["content"] = "a literal answer"
    t(any("rowA claims append" in p for p in check_tree(SELF_POLICY, _matrix(rowA=literal), None)),
      "append without an assistant $CAPTURE_ turn must fail")
    undef = json.loads(json.dumps(row_a))
    undef["positive_session"][1]["send"]["messages"][1]["content"] = "$CAPTURE_nope"
    t(any("rowA claims append" in p for p in check_tree(SELF_POLICY, _matrix(rowA=undef), None)),
      "append quoting a capture no earlier op defines must fail")
    # waivers schema
    doc = {"schema": WAIVERS_SCHEMA, "waivers": [_waiver("a" * 40, by="Main")]}
    t(any("not in waiver_grantors" in p for p in check_tree(SELF_POLICY, ok_matrix, doc)), "open waiver by Main fails check")
    doc = {"schema": WAIVERS_SCHEMA, "waivers": [_waiver("a" * 40, span_days=8)]}
    t(any("expiry span" in p for p in check_tree(SELF_POLICY, ok_matrix, doc)), "8-day waiver span fails check")
    doc = {"schema": WAIVERS_SCHEMA, "waivers": [_waiver("a" * 40)]}
    t(check_tree(SELF_POLICY, ok_matrix, doc) == [], "a sound open waiver passes check")
    doc = {"schema": WAIVERS_SCHEMA, "waivers": [_waiver("a" * 40, by="Main", closed="deadbeef")]}
    t(check_tree(SELF_POLICY, ok_matrix, doc) == [], "a closed waiver is not judged by grantor")
    t(check_tree(SELF_POLICY, ok_matrix, {"schema": "x", "waivers": []}) != [], "bad waivers schema fails check")
    # the committed policy file itself must be schema-valid
    t(validate_policy(load_json(os.path.join(ROOT, POLICY_REL))) == [], "committed policy.json is valid")
    t(validate_waivers(load_json(os.path.join(ROOT, POLICY_REL)), None) == [], "absent waivers valid")


def _self_test_waivers(t, scratch):
    repo = os.path.join(scratch, "wrepo")
    os.makedirs(repo)
    git(repo, "init", "-q")
    open(os.path.join(repo, "f"), "w").write("1")
    first = _git_commit(repo, "first")
    open(os.path.join(repo, "f"), "w").write("2")
    cand = _git_commit(repo, "candidate")
    git(repo, "checkout", "-q", "--detach", first)
    open(os.path.join(repo, "g"), "w").write("side")
    side = _git_commit(repo, "side")
    now = utc_now()
    ok, why = waiver_applies(_waiver(first), SELF_POLICY, "gfx1100", cand, repo, now)
    t(ok, "valid waiver applies (%s)" % why)
    ok, _ = waiver_applies(_waiver(cand), SELF_POLICY, "gfx1100", cand, repo, now)
    t(ok, "sha equal to candidate is allowed")
    for label, waiver, cls in (
            ("grantor Main", _waiver(first, by="Main"), "gfx1100"),
            ("span > 7 days", _waiver(first, span_days=8), "gfx1100"),
            ("expired", _waiver(first, hours=72, span_days=2), "gfx1100"),
            ("sha not an ancestor", _waiver(side), "gfx1100"),
            ("wrong class", _waiver(first, cls="gfx1201"), "gfx1100"),
            ("closed", _waiver(first, closed="c0ffee"), "gfx1100")):
        ok, _ = waiver_applies(waiver, SELF_POLICY, cls, cand, repo, now)
        t(not ok, "waiver must not apply: %s" % label)
    return repo, first, cand


def _self_test_run(t, scratch, repo, first):
    gate = os.path.join(scratch, "gate")
    for rel in ("scripts/promotion-lane", "scripts/device-mesh"):
        os.makedirs(os.path.join(gate, rel))
    write_json(os.path.join(gate, POLICY_REL), SELF_POLICY)
    write_json(os.path.join(gate, MATRIX_REL), _matrix())
    open(os.path.join(gate, "scripts", "device_mesh_matrix.py"), "w").write(STUB_RUNNER)
    git(gate, "init", "-q")
    _git_commit(gate, "gate")

    stage = os.path.join(scratch, "stage")
    for rel in ("bin/hipfire", "bin/daemon", "bin/kernels/compiled/gfx1100/k.hsaco"):
        os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True)
        open(os.path.join(stage, rel), "w").write("x")
    archive = os.path.join(scratch, "hipfire-test.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(os.path.join(stage, "bin"), arcname="bin")
    open(archive + ".sha256", "w").write("%s  hipfire-test.tar.gz\n" % sha256_file(archive))
    fakebin = os.path.join(scratch, "fakebin")
    os.makedirs(fakebin)
    for name in ("hipcc", "clang++"):
        open(os.path.join(fakebin, name), "w").write("#!/bin/sh\n")
        os.chmod(os.path.join(fakebin, name), 0o755)
    dcache = os.path.join(scratch, "digest-cache.json")
    models = os.path.join(scratch, "models")

    def case(name, rows=None, logs=None, waivers=None, exit_code=None, cache=True, tail=()):
        out = os.path.join(scratch, "out-" + name)
        obs = os.path.join(scratch, "obs-%s.json" % name)
        write_json(os.path.join(gate, "scripts", "scenario.json"),
                   {"obs": obs, "rows": rows or {}, "logs": logs or {}, "exit": exit_code})
        sha = first
        if waivers is not None:
            git(repo, "checkout", "-q", "--detach", first)
            os.makedirs(os.path.join(repo, "scripts", "promotion-lane"), exist_ok=True)
            write_json(os.path.join(repo, WAIVERS_REL), {"schema": WAIVERS_SCHEMA, "waivers": waivers(first)})
            sha = _git_commit(repo, "waivers " + name)
        argv = ["run", "--archive", archive, "--class", "gfx1100", "--candidate", sha, "--out", out,
                "--gate-root", gate, "--repo", repo, "--models-dir", models, "--host", "selftest-host"]
        if cache:
            argv += ["--digest-cache", dcache]
        rc, stdout, stderr = _call(main, argv + list(tail))
        rec = load_json(os.path.join(out, "record.json")) if os.path.exists(os.path.join(out, "record.json")) else None
        observed = load_json(obs) if os.path.exists(obs) else None
        return rc, rec, observed, stdout + stderr, out

    saved = os.environ["PATH"]
    os.environ["PATH"] = fakebin + os.pathsep + saved
    try:
        write_json(dcache, {"seed": 1})
        rc, rec, obs, text, out = case("green")
        t(rc == 0 and rec and rec["verdict"] == "green", "all current -> green (%s)" % text)
        t(text.startswith("LANE gfx1100 GREEN %s (2 rows," % rec["candidate"][:12]), "LANE line format")
        t(obs["which_hipcc"] is None and rec["no_compiler"]["path_hipcc"] is None, "hipcc not on child PATH")
        t(obs["which_sh"] is not None, "cleaned PATH keeps sh resolvable")
        t(obs["HIPFIRE_NO_DEVICE_COMPILER"] == "1" and rec["no_compiler"]["HIPFIRE_NO_DEVICE_COMPILER"] == "1",
          "HIPFIRE_NO_DEVICE_COMPILER=1 in child env")
        home = os.path.join(out, "home")
        t(obs["HOME"] == home and obs["MATRIX_ISOLATED_HOME"] == home, "isolated HOME and MATRIX_ISOLATED_HOME")
        t(obs["HIPFIRE_KERNEL_CACHE"] == os.path.join(out, "kernel-cache")
          and os.listdir(obs["HIPFIRE_KERNEL_CACHE"]) == [], "fresh empty kernel cache")
        t(obs["TMPDIR"] == os.path.join(out, "tmp") and obs["HIPFIRE_MODELS_DIR"] == models, "TMPDIR and models dir")
        t(obs["HIPFIRE_CLI_BIN"] == os.path.join(out, "install", "bin", "hipfire")
          and obs["HIPFIRE_DAEMON_BIN"] == os.path.join(out, "install", "bin", "daemon"), "installed binaries in env")
        t(obs["rows_arg"] == ["rowA", "rowB"] and obs["host_class"] == "gfx1100", "runner gets policy rows and class")
        t(obs["cache_in"] == {"seed": 1} and load_json(dcache) == {"x": 1, "y": 2}, "digest cache copied in and back")
        t(rec["host"] == "selftest-host" and rec["gpu_arch_verified"] == "gfx1100", "host and verified arch recorded")
        t(rec["archive"]["sidecar_ok"] is True and len(rec["archive"]["sha256"]) == 64, "archive hash + sidecar recorded")
        t(rec["gate"]["sha"] == git(gate, "rev-parse", "HEAD").stdout.strip()
          and rec["gate"]["policy_sha256"] == sha256_file(os.path.join(gate, POLICY_REL))
          and rec["gate"]["matrix_sha256"] == sha256_file(os.path.join(gate, MATRIX_REL)), "gate identity recorded")
        t(record_status(rec, SELF_POLICY) == ("green", []), "record passes its own integrity check")

        for disp in ("hardware-blocked", "rerun-required", "failed", "missing", "historical"):
            rc, rec, _, _, _ = case("skip-" + disp, rows={"rowB": disp})
            row = rec["rows"][1]
            t(rc == 1 and rec["verdict"] == "red" and row["verdict"] == ("failed" if disp == "failed" else "skipped"),
              "required row %s must be red" % disp)
        rc, rec, _, text, _ = case("marker1", logs={"rowA": "x\nrowA: hipcc compiled (gfx1100, key ab)\n"})
        t(rc == 1 and rec["rows"][0]["verdict"] == "failed" and rec["rows"][0]["compile_markers"]
          and rec["rows"][0]["disposition"] == "current", "'hipcc compiled (' in a current row's log is red")
        rc, rec, _, _, _ = case("marker2", logs={"rowB": "k: launch, but hipcc is unavailable.\n"})
        t(rc == 1 and rec["rows"][1]["verdict"] == "failed", "'but hipcc is unavailable' is red")
        rc, rec, _, _, _ = case("exit2", exit_code=2)
        t(rc == 2 and rec is None, "runner infrastructure exit is exit 2 with no record")

        # waivers
        rc, rec, _, _, _ = case("waive-ok", rows={"rowB": "hardware-blocked"}, waivers=lambda s: [_waiver(s)])
        t(rc == 0 and rec["rows"][1]["verdict"] == "waived" and len(rec["waivers"]) == 1, "valid waiver turns a red row waived")
        for label, make in (("grantor Main", lambda s: _waiver(s, by="Main")),
                            ("span", lambda s: _waiver(s, span_days=9)),
                            ("expired", lambda s: _waiver(s, hours=100, span_days=2)),
                            ("class", lambda s: _waiver(s, cls="gfx1201")),
                            ("row", lambda s: _waiver(s, rows=("rowA",)))):
            rc, rec, _, _, _ = case("waive-" + label.replace(" ", ""), rows={"rowB": "hardware-blocked"},
                                    waivers=lambda s, make=make: [make(s)])
            t(rc == 1 and rec["rows"][1]["verdict"] == "skipped" and rec["waivers"] == [],
              "invalid waiver (%s) must not apply" % label)
        # waived record is not a release record
        rc, rec, _, _, out = case("waive-rel", rows={"rowB": "hardware-blocked"}, waivers=lambda s: [_waiver(s)])
        t(record_status(rec, SELF_POLICY)[0] == "green", "waived record is green for the candidate gate")
        rdir = os.path.join(scratch, "rel-waived")
        os.makedirs(rdir)
        write_json(os.path.join(rdir, "gfx1100.json"), rec)
        ns = argparse.Namespace(sha=rec["candidate"], records=rdir, repo=repo)
        rc, text, _ = _call(cmd_check_release, ns, SELF_POLICY)
        t(rc == 1 and "waived" in text, "check-release rejects a waived record and an open waiver")

        # archive integrity
        open(archive + ".sha256", "w").write("0" * 64 + "  x\n")
        rc, rec, _, text, _ = case("badsidecar")
        t(rc == 2 and "sidecar" in text, "sidecar mismatch is exit 2")
        open(archive + ".sha256", "w").write("%s  x\n" % sha256_file(archive))
        rc, rec, _, _, _ = case("timeout", tail=["--timeout", "5"])
        t(rc == 0 and rec is not None, "--timeout accepted")
    finally:
        os.environ["PATH"] = saved


def _fake_record(cls, cand, verdict="green", rows=("rowA", "rowB"), waived=False):
    ok = verdict == "green"
    return {"schema": RECORD_SCHEMA, "candidate": cand, "class": cls, "host": "h", "gpu_arch_verified": cls,
            "archive": {"file": "a", "sha256": "ab" * 32, "sidecar_ok": True},
            "gate": {"root": "r", "sha": "1" * 40, "policy_sha256": "p", "matrix_sha256": "m"},
            "no_compiler": {"path_hipcc": None, "HIPFIRE_NO_DEVICE_COMPILER": "1"},
            "rows": [{"id": r, "required": True, "disposition": "current" if ok else "failed",
                      "verdict": "waived" if waived else ("current" if ok else "failed"), "elapsed_s": 1, "note": "",
                      "receipt": "", "compile_markers": []} for r in rows],
            "waivers": [_waiver(cand)] if waived else [], "started_utc": "", "finished_utc": "", "wall_s": 1,
            "budget_minutes": 60, "over_budget": False, "verdict": verdict}


def _self_test_merge(t, scratch, repo, release_sha):
    cand = "c" * 40
    rdir = os.path.join(scratch, "records")
    os.makedirs(rdir)

    def merge(records, allow_down=(), rented=(), sha=cand):
        paths = []
        for i, rec in enumerate(records):
            paths.append(os.path.join(rdir, "r%d-%d.json" % (len(os.listdir(rdir)), i)))
            write_json(paths[-1], rec)
        ns = argparse.Namespace(candidate=sha, out=os.path.join(scratch, "cand.json"), allow_down=list(allow_down),
                                rented=list(rented), records=paths)
        rc, text, _ = _call(cmd_merge, ns, SELF_POLICY)
        return rc, text, load_json(ns.out) if os.path.exists(ns.out) else None

    rc, _, doc = merge([_fake_record("gfx1100", cand), _fake_record("gfx1201", cand)])
    t(rc == 0 and doc["verdict"] == "green" and doc["classes"]["gfx942"]["status"] == "not-run",
      "all candidate classes green -> green; rented class not required")
    rc, text, doc = merge([_fake_record("gfx1100", cand)])
    t(rc == 1 and doc["verdict"] == "red" and doc["classes"]["gfx1201"]["status"] == "not-run", "missing class -> red")
    rc, _, doc = merge([_fake_record("gfx1100", cand)], allow_down=["gfx1201"])
    t(rc == 0 and doc["owed"] == ["gfx1201"] and doc["classes"]["gfx1201"]["status"] == "down", "--allow-down -> green with owed")
    rc, _, doc = merge([_fake_record("gfx1100", cand), _fake_record("gfx1201", "d" * 40)])
    t(rc == 1 and any("candidate" in p for p in doc["problems"]), "mismatched candidate sha across records -> red")
    rc, _, doc = merge([_fake_record("gfx1100", cand), _fake_record("gfx1201", cand, verdict="red")])
    t(rc == 1 and doc["classes"]["gfx1201"]["status"] == "red", "red record -> red")
    lying = _fake_record("gfx1201", cand)
    lying["rows"][1]["verdict"] = "skipped"
    rc, _, doc = merge([_fake_record("gfx1100", cand), lying])
    t(rc == 1 and doc["classes"]["gfx1201"]["status"] == "red", "record claiming green over a skipped row -> red")
    thin = _fake_record("gfx1201", cand, rows=("rowA",))
    rc, _, _ = merge([_fake_record("gfx1100", cand), thin])
    t(rc == 1, "record missing a required row -> red")
    rc, _, doc = merge([_fake_record("gfx1100", cand), _fake_record("gfx1201", cand)], rented=["gfx942"])
    t(rc == 1, "--rented class without a record -> red")
    rc, _, doc = merge([_fake_record("gfx1100", cand), _fake_record("gfx1201", cand), _fake_record("gfx942", cand)],
                       rented=["gfx942"])
    t(rc == 0 and doc["classes"]["gfx942"]["status"] == "green", "--rented class with green record -> green")
    rc, _, _ = merge([_fake_record("gfx1100", cand), _fake_record("gfx1100", cand), _fake_record("gfx1201", cand)])
    t(rc == 1, "two records for one class -> red")
    rc, _, _ = merge([_fake_record("gfx1100", cand)], allow_down=["gfx9999"])
    t(rc == 2, "--allow-down with unknown class is a usage error")
    # check-release: records at release_sha, no open waivers in tree
    sha = release_sha
    rel = os.path.join(scratch, "rel")
    os.makedirs(rel)
    for cls in SELF_POLICY["classes"]:
        write_json(os.path.join(rel, cls + ".json"), _fake_record(cls, sha))
    ns = argparse.Namespace(sha=sha, records=rel, repo=repo)
    rc, text, _ = _call(cmd_check_release, ns, SELF_POLICY)
    t(rc == 0 and text.startswith("RELEASE GREEN"), "check-release green with every class green and no waivers (%s)" % text)
    os.remove(os.path.join(rel, "gfx942.json"))
    rc, text, _ = _call(cmd_check_release, ns, SELF_POLICY)
    t(rc == 1 and "gfx942" in text, "check-release needs the rented class too")
    write_json(os.path.join(rel, "gfx942.json"), _fake_record("gfx942", sha, waived=True))
    rc, text, _ = _call(cmd_check_release, ns, SELF_POLICY)
    t(rc == 1 and "waived" in text, "check-release rejects a waived record")
    write_json(os.path.join(rel, "gfx942.json"), _fake_record("gfx942", "e" * 40))
    rc, _, _ = _call(cmd_check_release, ns, SELF_POLICY)
    t(rc == 1, "check-release ignores records for another sha")


def _self_test_classify(t, scratch):
    repo = os.path.join(scratch, "crepo")
    os.makedirs(repo)
    git(repo, "init", "-q")
    open(os.path.join(repo, "a"), "w").write("0")
    base = _git_commit(repo, "base")

    def change(files, message):
        for rel in files:
            os.makedirs(os.path.dirname(os.path.join(repo, rel)) or repo, exist_ok=True)
            open(os.path.join(repo, rel), "a").write("x")
        return _git_commit(repo, message)

    code = change(["crates/x/src/lib.rs"], "code")
    pol = change(["scripts/promotion-lane/policy.json", "scripts/device-mesh/matrix.json"], "policy")
    mixed = change(["scripts/ci-gates.sh", "crates/y/lib.rs"], "mixed")
    for label, rng, want in (("code", base + ".." + code, "code"), ("policy", code + ".." + pol, "policy"),
                             ("mixed", base + ".." + mixed, "mixed")):
        ns = argparse.Namespace(repo=repo, range=rng)
        rc, text, _ = _call(cmd_classify, ns, SELF_POLICY)
        t(rc == 0 and text.splitlines()[0] == want, "classify %s range -> %s (%s)" % (label, want, text.splitlines()[:1]))
    ns = argparse.Namespace(repo=repo, range=code + ".." + pol)
    _, text, _ = _call(cmd_classify, ns, SELF_POLICY)
    t("policy scripts/device-mesh/matrix.json" in text.splitlines(), "classify labels policy paths")
    ns = argparse.Namespace(repo=repo, range=base + ".." + mixed)
    _, text, _ = _call(cmd_classify, ns, SELF_POLICY)
    t("code crates/y/lib.rs" in text.splitlines() and "policy scripts/ci-gates.sh" in text.splitlines(),
      "classify labels both kinds in a mixed range")
    t(classify_paths(SELF_POLICY, ["scripts/promotion-lane/a/b/c.py"])[0] == "policy", "** matches nested paths")
    t(classify_paths(SELF_POLICY, ["scripts/promotion-lane"])[0] == "code", "** needs a path below the directory")
    t(classify_paths(SELF_POLICY, ["xscripts/ci-gates.sh"])[0] == "code", "globs are anchored")
    rc, _, _ = _call(cmd_classify, argparse.Namespace(repo=repo, range="nope..nada"), SELF_POLICY)
    t(rc == 2, "classify with a bad range is exit 2")


def self_test():
    t = Counter()
    scratch = tempfile.mkdtemp(prefix="lane-selftest-")
    try:
        _self_test_check(t)
        repo, first, cand = _self_test_waivers(t, scratch)
        _self_test_run(t, scratch, repo, first)
        git(repo, "checkout", "-q", "--detach", cand)
        _self_test_merge(t, scratch, repo, cand)
        _self_test_classify(t, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for msg in t.failed:
        print("FAIL: %s" % msg, file=sys.stderr)
    if t.failed:
        print("promotion-lane: self-test FAILED (%d of %d checks)" % (len(t.failed), t.n))
        return 1
    print("promotion-lane: self-test OK (%d checks)" % t.n)
    return 0


# --------------------------------------------------------------------------- CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the planted-negative self-test")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="run the required rows for one class on an installed archive")
    run.add_argument("--archive", required=True)
    run.add_argument("--class", dest="cls", required=True)
    run.add_argument("--candidate", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gate-root", default=ROOT)
    run.add_argument("--repo", default=None, help="repository holding the candidate commit (default: gate root)")
    run.add_argument("--models-dir", default=None)
    run.add_argument("--timeout", type=int, default=None, help="per-command seconds for the matrix runner")
    run.add_argument("--digest-cache", default=None, help="fixture digest cache file, read before and updated after")
    run.add_argument("--host", default=None, help="host name to record (default: this host)")
    merge = sub.add_parser("merge", help="merge per-class records into a candidate verdict")
    merge.add_argument("--candidate", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--allow-down", action="append", default=[], metavar="CLASS")
    merge.add_argument("--rented", action="append", default=[], metavar="CLASS")
    merge.add_argument("records", nargs="+")
    rel = sub.add_parser("check-release", help="require green unwaived records for every class at SHA")
    rel.add_argument("--sha", required=True)
    rel.add_argument("--records", required=True)
    rel.add_argument("--repo", default=None)
    cls = sub.add_parser("classify", help="print code|policy|mixed for BASE..TIP, then the paths")
    cls.add_argument("--repo", required=True)
    cls.add_argument("--range", required=True)
    chk = sub.add_parser("check", help="CPU checks of policy, matrix and waivers")
    chk.add_argument("--repo", default=None)
    plan = sub.add_parser("plan", help="show the rows a class must run")
    plan.add_argument("--class", dest="cls", required=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.cmd:
        ap.print_usage(sys.stderr)
        return 2
    try:
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "check":
            return cmd_check(args)
        policy = load_policy(os.path.join(ROOT, POLICY_REL))
        return {"merge": cmd_merge, "check-release": cmd_check_release, "classify": cmd_classify,
                "plan": cmd_plan}[args.cmd](args, policy)
    except Usage as exc:
        print("lane.py: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
