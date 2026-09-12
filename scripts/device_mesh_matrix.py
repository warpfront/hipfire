#!/usr/bin/env python3
"""G1-G4 acceptance matrix runner (issue warpfront/hipfire#666). Catalogue:
scripts/device-mesh/matrix.json. Existing harnesses/tests plus thin
daemon-JSONL drivers for G1/G2 rows. Predicates parse structured output."""
import argparse
import hashlib
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

CARGO_RE = re.compile(r"test result:\s*(\w+)\.\s*(\d+) passed;\s*(\d+) failed")
FORBID_RUN = ("daemon error", "[unsupported", "[CLS-", "Traceback")
# Documented per-arch limits of a probe path (not tree defects). A row hitting
# one of these is hardware-blocked on that arch, exactly like an absent fixture.
CAPABILITY_GAPS = ("PM4 dispatch does not yet support scratch",)
SAMPLE_CARGO_PASS = "test result: ok. 5 passed; 0 failed; 0 ignored; finished in 1.2s\n"
SAMPLE_CARGO_FAIL = "test result: FAILED. 4 passed; 1 failed; 0 ignored; finished in 2.0s\n"
SAMPLE_SERVE_CLEAN = [{"attractor": False, "empty": False, "runaway": False, "visible": "Paris."}]
SAMPLE_SERVE_DIRTY = [{"attractor": False, "empty": False, "runaway": True, "visible": "x " * 400}]
SAMPLE_REDLINE = {"pass": True, "prefill": {"1024": {"sequence_stable": True}}, "decode": {"sequence_stable": True},
                  "aql_shadow": {"bit_exact": True}}
SAMPLE_WIRE_DONE = [{"type": "loaded", "arch": "qwen3_5", "vl": False}, {"type": "token", "id": "r1", "text": "hi"},
                    {"type": "commit_ready", "id": "r1"}, {"type": "done", "id": "r1", "tokens": 1}]

def repo_root():
    node = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(node, "registry", "v1.json")):
        parent = os.path.dirname(node)
        if parent == node:
            break
        node = parent
    return node

def load_catalogue(root):
    return json.load(open(os.path.join(root, "scripts", "device-mesh", "matrix.json")))

def find_bins(root):
    def pick(env, rels):
        if os.path.isfile(os.environ.get(env, "")):
            return os.environ[env]
        return next((c for c in (os.path.join(root, r) for r in rels) if os.path.isfile(c)), None)
    return (pick("HIPFIRE_CLI_BIN", ["target/release/hipfire", "target/debug/hipfire"]),
            pick("HIPFIRE_DAEMON_BIN", ["target/release/daemon", "target/debug/daemon"]))

def models_dir():
    return os.path.expanduser(os.environ.get("HIPFIRE_MODELS_DIR", "~/.hipfire/models"))

def resolve_fixture(root, tag):
    try:
        reg = json.load(open(os.path.join(root, "registry", "v1.json")))
        name = reg.get("models", reg).get(tag, {}).get("file")
    except (OSError, ValueError):
        name = None
    return os.path.join(models_dir(), os.path.basename(name) if name else tag)

def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_digest_cache(out):
    path = os.path.join(out, ".digest-cache.json")
    try:
        return json.load(open(path)), path
    except (OSError, ValueError):
        return {}, path

def sha256_cached(path, cache):
    # Sidecar avoids re-hashing 80 GB files; no CLI digest cache exists.
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    key = "%s|%d|%d" % (path, st.st_size, int(st.st_mtime))
    hit = cache.get(path)
    if isinstance(hit, dict) and hit.get("key") == key:
        return hit["sha256"]
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    cache[path] = {"key": key, "sha256": digest.hexdigest()}
    return cache[path]["sha256"]

def isolated_env(env_extra):
    """Child env with an isolated HIPFIRE_HOME. The operator's
    ~/.hipfire/config.toml (mtp, dflash_mode, kv overrides, per-model config)
    must never decide a matrix row's outcome — that config differs per host,
    which silently turns one host's pass into another host's failure."""
    env = dict(os.environ, **(env_extra or {}))
    env.setdefault("HIPFIRE_HOME", os.path.join(env.get("MATRIX_ISOLATED_HOME", "/tmp/device-mesh-matrix-home"), ".hipfire"))
    os.makedirs(env["HIPFIRE_HOME"], exist_ok=True)
    return env

def run_cmd(argv, env_extra, timeout):
    env = isolated_env(env_extra)
    start = time.time()
    try:
        proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
        return {"rc": proc.returncode, "out": proc.stdout + proc.stderr,
                "stdout": proc.stdout, "elapsed": round(time.time() - start, 1)}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        rc = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
        return {"rc": rc, "out": "%s" % exc, "elapsed": 0.0}

def git_identity(root):
    def git(*args):
        res = run_cmd(["git", "-C", root] + list(args), None, 30)
        return res["out"].strip() if res["rc"] == 0 else ""
    return {"head": git("rev-parse", "HEAD") or "unknown",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "base_beta": git("merge-base", "HEAD", "origin/beta") or "unknown",
            "dirty": bool(git("status", "--porcelain"))}
def best_effort(argv, limit=1024):
    res = run_cmd(argv, None, 30)
    return (res["out"][:limit] if res["rc"] == 0 else "unavailable: %s" % res["out"][:200]).strip()

def platform_probe():
    ldconfig = best_effort(["ldconfig", "-p"], 4000)
    return {"hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
            "kfd": os.path.exists("/dev/kfd"), "dri": os.path.exists("/dev/dri"),
            "hipcc": best_effort(["hipcc", "--version"]),
            "driver": best_effort(["rocm-smi", "--showdriverversion"]),
            "pci_ids": best_effort(["rocm-smi", "--showid"]),
            "rccl": "\n".join(l for l in ldconfig.splitlines() if "rccl" in l.lower())[:500] or "unavailable"}

def has_gpu():
    return os.path.exists("/dev/kfd") or os.path.exists("/dev/dri")

def parse_cargo(text):
    counts = [(int(m.group(2)), int(m.group(3))) for m in CARGO_RE.finditer(text)]
    return {"summaries": "test result:" in text, "passed": sum(p for p, _ in counts),
            "failed": sum(f for _, f in counts)}

def eval_cargo(text, minimum):
    s = parse_cargo(text)
    return (s["summaries"] and not s["failed"] and s["passed"] >= minimum,
            "cargo: passed=%d failed=%d min=%d" % (s["passed"], s["failed"], minimum))

def degenerate_text(content):
    """A lifecycle row fails on DEGENERATE output, not on truncation.
    `runaway` in serve_harness is simply finish=="length", so a healthy long
    answer that reaches the token cap trips it. Judge the text instead: an
    attractor or repetition loop collapses the distinct-word ratio, while real
    prose and code stay far above it (measured 0.34-0.65 on this fixture set)."""
    words = (content or "").split()
    if len(words) < 64:
        return True  # a cap-truncated row with almost no text is not healthy
    return len(set(words)) / len(words) < 0.2

def eval_serve_rows(rows, expect_clean):
    if not isinstance(rows, list) or not rows:
        return False, "serve: --out holds no rows"
    def flagged(r):
        if not isinstance(r, dict) or r.get("attractor") or r.get("empty"):
            return True
        # Truncation alone is not a defect; degenerate truncation is.
        return bool(r.get("runaway")) and degenerate_text(r.get("content"))
    bad = [i for i, r in enumerate(rows) if flagged(r)]
    if expect_clean:
        return not bad, "serve: %d rows, %d flagged %s" % (len(rows), len(bad), bad)
    return bool(bad), "serve-negative: %d rows, %d flagged %s" % (len(rows), len(bad), bad)

def eval_serve_expect(rows, expect_file):
    """Session fixtures that deliberately exercise fail-closed routes declare a
    per-step `expect` list of finish reasons. "0 flagged" is the wrong oracle
    there: a `fail_closed` step that finishes `stop` is the FAILURE, and its
    `error`/`length` finish is the pass. Compare each row to its declared set."""
    try:
        steps = json.load(open(expect_file)).get("steps", [])
    except (OSError, ValueError) as exc:
        return False, "serve-expect: unreadable fixture %s" % exc
    if not isinstance(rows, list) or not rows:
        return False, "serve: --out holds no rows"
    if len(rows) != len(steps):
        return False, "serve-expect: %d rows vs %d declared steps" % (len(rows), len(steps))
    bad = []
    for i, (row, step) in enumerate(zip(rows, steps)):
        want = step.get("expect_finish") or step.get("expect") or ["stop"]
        got = row.get("finish")
        # A positive step must also be non-degenerate; a fail-closed step is
        # judged only on reaching its declared terminal.
        degenerate = step.get("kind") != "fail_closed" and (row.get("attractor") or row.get("empty"))
        if got not in want or degenerate:
            bad.append("%d:%s(want %s)" % (i, got, "/".join(want)))
    return not bad, "serve-expect: %d steps, %d off-contract %s" % (len(steps), len(bad), bad)

def eval_redline(report, require_stable):
    if not isinstance(report, dict) or report.get("pass") is not True:
        return False, "redline: report pass != true"
    if require_stable:
        # `prefill` is a map {context_tokens: result}; `decode` is ONE result
        # dict. Iterating decode.values() walks its fields (`measurement`,
        # `captures`, ...) and fails a passing report on the first field that
        # happens to be a dict without the key. Pick the result nodes by shape.
        for section in ("prefill", "decode"):
            node = report.get(section, {})
            if not isinstance(node, dict):
                return False, "redline: %s missing" % section
            results = [node] if "sequence_stable" in node else [
                v for v in node.values() if isinstance(v, dict)]
            if not results:
                return False, "redline: %s reports no capture" % section
            for item in results:
                if item.get("sequence_stable") is not True:
                    return False, "redline: %s not sequence_stable" % section
        shadow = report.get("aql_shadow")
        if isinstance(shadow, dict) and shadow.get("bit_exact") is not True:
            return False, "redline: aql_shadow not bit_exact"
    return True, "redline: pass=true stable=%s" % require_stable

def eval_run_text(text, min_chars, forbid=FORBID_RUN):
    hit = [m for m in forbid if m in text]
    good = len(text.strip()) >= min_chars and not hit
    return good, "run: chars=%d min=%d forbidden=%s" % (len(text.strip()), min_chars, hit)

def event_text(events):
    parts = []
    for e in events:
        if isinstance(e, dict) and e.get("type") not in ("commit_ready", "commit"):
            parts += [e[k] for k in ("text", "token", "content", "delta")
                      if isinstance(e.get(k), str) and e[k]]
    return "".join(parts)

def id_events(run, rid):
    return [e for e in run.get("events", []) if isinstance(e, dict) and e.get("id") == rid]

def terminal_of(run, rid):
    terms = [e for e in id_events(run, rid) if e.get("type") in ("done", "error")]
    return terms[0] if len(terms) == 1 else None

def wire_texts(runs):
    return [event_text(id_events(run, rid)) for run in runs for rid in run.get("generates", [])]

def wire_errors(runs):
    return [e for run in runs for e in run.get("events", []) if e.get("type") == "error"]

def eval_wire(runs, kind, pred):
    if not runs:
        return False, "wire: no runs executed"
    if kind == "wire_same_arch_twice":
        arches = [l.get("arch") for run in runs for l in run.get("loaded", [])]
        ok = len(arches) >= 2 and all(a is not None and a == arches[0] for a in arches)
        return ok, "wire: arch across source paths: %s" % arches
    if kind == "wire_error":
        return bool(wire_errors(runs)), "wire-negative: %d error(s)" % len(wire_errors(runs))
    if kind == "wire_vl_false_then_done":
        vls = [l.get("vl") for run in runs for l in run.get("loaded", [])]
        if any(v is not False for v in vls):
            return False, "wire: loaded.vl not false: %s" % vls
        kind = "wire_done_clean"
    if kind == "wire_done_clean":
        ids = [rid for run in runs for rid in run.get("generates", [])]
        bad = [rid for run in runs for rid in run.get("generates", [])
               if (terminal_of(run, rid) or {}).get("type") != "done"
               or not event_text(id_events(run, rid)).strip()]
        return bool(ids) and not bad, "wire: %d clean done(s), bad=%s" % (len(ids), bad)
    if kind in ("wire_reload_equal", "wire_four_cycles"):
        texts = wire_texts(runs)
        if len(texts) < 2 or any(not t.strip() for t in texts):
            return False, "wire: need 2+ nonempty dones, got %d" % len(texts)
        same = all(t == texts[0] for t in texts[1:])
        return same, "wire: %d done text(s) identical=%s" % (len(texts), same)
    if kind == "wire_active_survives_bad_load":
        errs = wire_errors(runs)
        dones = [(terminal_of(r, g) or {}).get("type") == "done" and bool(event_text(id_events(r, g)).strip())
                 for r in runs for g in r.get("generates", [])]
        texts = wire_texts(runs)
        same = len(texts) >= 2 and all(t == texts[0] for t in texts[1:])
        return bool(errs) and dones and all(dones), "wire: errors=%d usable=%s identical=%s" % (len(errs), all(dones), same)
    if kind == "wire_exactly_one_terminal":
        # G4.6 race contract, per admitted request id: exactly one terminal
        # (done/error; a cancel surfaces as done with finish_reason aborted),
        # correlated to a known id, with no post-terminal bytes for that id.
        ids = [rid for run in runs for rid in run.get("generates", [])]
        if not ids:
            return False, "wire: no generate ids admitted"
        expect = pred.get("expect", {}) or {}
        bad = []
        for run in runs:
            known = set(run.get("generates", []))
            for rid in run.get("generates", []):
                evs = id_events(run, rid)
                terms = [e for e in evs if e.get("type") in ("done", "error")]
                if len(terms) != 1:
                    bad.append("%s: %d terminals" % (rid, len(terms)))
                    continue
                want = expect.get(rid)
                if want and terms[0].get("type") != want:
                    bad.append("%s: terminal=%s want %s" % (rid, terms[0].get("type"), want))
                    continue
                pos = next(i for i, e in enumerate(evs) if e is terms[0])
                if pos != len(evs) - 1:
                    bad.append("%s: %d post-terminal event(s)" % (rid, len(evs) - 1 - pos))
            for e in run.get("events", []):
                if (isinstance(e, dict) and e.get("type") in ("done", "error")
                        and e.get("id") is not None and e.get("id") not in known):
                    bad.append("unknown-id terminal: %r" % (e.get("id"),))
        return not bad, "wire: %d id(s) exactly-one-terminal, bad=%s" % (len(ids), bad)
    return False, "wire: unknown predicate %s" % kind

def subst(node, mapping):
    if isinstance(node, dict):
        return {k: subst(v, mapping) for k, v in node.items()}
    if isinstance(node, list):
        return [subst(v, mapping) for v in node]
    if isinstance(node, str):
        if not mapping:
            return node  # empty alternation matches "" and KeyErrors on every gap
        pattern = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        return re.sub(pattern, lambda m: mapping[m.group(0)], node)
    return node

def drive_session(daemon_bin, script, mapping, env, timeout):
    run = {"events": [], "generates": [], "loaded": [], "broken": ""}
    try:
        proc = subprocess.Popen([daemon_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1,
                                env=isolated_env(env))
    except OSError as exc:
        return dict(run, broken="spawn: %s" % exc)
    lines = queue.Queue()
    def pump():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    budgets = {"until": 0.0}
    def read_line(context):
        rest = budgets["until"] - time.time()
        if rest <= 0:
            raise TimeoutError("%s: timed out" % context)
        try:
            line = lines.get(timeout=rest)
        except queue.Empty:
            raise TimeoutError("%s: timed out" % context)
        if line is None:
            raise EOFError("%s: daemon closed stdout" % context)
        return line
    def wait_for(want, context):
        while True:
            event = json.loads(read_line(context))
            run["events"].append(event)
            if event.get("type") == want:
                run["loaded"].extend([event] if want == "loaded" else [])
                return event
            if event.get("type") == "error" and want != "error":
                # The daemon already answered; waiting out the rest of the
                # budget turns a one-line diagnosis into an opaque timeout.
                raise TimeoutError("%s: got error instead: %s"
                                   % (context, (event.get("message") or "")[:160]))
    try:
        for op in script:
            budgets["until"] = time.time() + timeout
            if "race_sends" in op:
                # More than one request in flight: write every send before
                # collecting any terminal, so done/error/cancel interleave at
                # the daemon instead of being serialized by the driver. A
                # sequential write-then-wait loop cannot express this: the
                # second request would never be admitted until the first
                # resolves, and no abort could land mid-flight.
                sends = [subst(s, mapping) for s in op["race_sends"]]
                attempts = {}
                for req in sends:
                    proc.stdin.write(json.dumps(req) + "\n")
                    if isinstance(req, dict) and req.get("type") == "generate" and req.get("id"):
                        run["generates"].append(req["id"])
                        attempts[req["id"]] = req.get("attempt_id", 1)
                proc.stdin.flush()
                pending = [subst(t, mapping) for t in op.get("race_terminals", [])]
                if not pending:
                    raise ValueError("race_sends needs a non-empty race_terminals list")
                while pending:
                    event = json.loads(read_line("race terminal %s" % ",".join(pending)))
                    run["events"].append(event)
                    etype, eid = event.get("type"), event.get("id")
                    if etype == "commit_ready" and eid in pending:
                        proc.stdin.write(json.dumps({"type": "commit", "id": eid,
                                                     "attempt_id": attempts.get(eid, 1)}) + "\n")
                        proc.stdin.flush()
                    elif etype in ("done", "error") and eid in pending:
                        # Any error here IS a terminal for that id (the daemon
                        # emits correlated errors only as terminals), so unlike
                        # wait_for there is no error-shortcircuit: an unknown-id
                        # error is recorded and the predicate fails it.
                        pending = [p for p in pending if p != eid]
                continue
            if "quiet_sends" in op:
                # Stale-control probe: every id is already terminal, so these
                # duplicate/late/wrong-attempt writers must produce no further
                # bytes for the retired ids. Drain a fixed window and record
                # anything heard; the predicate fails post-terminal bytes.
                for req in [subst(s, mapping) for s in op["quiet_sends"]]:
                    proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                deadline = time.time() + float(op.get("quiet_ms", 5000)) / 1000.0
                while True:
                    rest = deadline - time.time()
                    if rest <= 0:
                        break
                    try:
                        line = lines.get(timeout=rest)
                    except queue.Empty:
                        break
                    if line is None:
                        raise EOFError("quiet drain: daemon closed stdout")
                    run["events"].append(json.loads(line))
                continue
            request = subst(op["send"], mapping)
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            if op.get("handshake"):
                rid = request.get("id")
                run["generates"].append(rid)
                while True:
                    event = json.loads(read_line("generate %s" % rid))
                    run["events"].append(event)
                    if event.get("id") == rid and event.get("type") == "commit_ready":
                        commit = {"type": "commit", "id": rid, "attempt_id": request.get("attempt_id", 1)}
                        proc.stdin.write(json.dumps(commit) + "\n")
                    if event.get("id") == rid and event.get("type") in ("done", "error"):
                        break
            elif "expect_terminal" in op:
                wait_for(op["expect_terminal"], "terminal")
            elif op.get("expect") == "loaded" and "assert_loaded" in op:
                loaded = wait_for("loaded", "load")
                for key, want in op["assert_loaded"].items():
                    if loaded.get(key) != want:
                        raise ValueError("loaded.%s=%r, want %r" % (key, loaded.get(key), want))
            else:
                wait_for(op["expect"], "expect %s" % op["expect"])
    except (TimeoutError, EOFError, ValueError) as exc:
        run["broken"] = str(exc)
    finally:
        try: proc.stdin.close()
        except BrokenPipeError: pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        run["vram_raw"] = best_effort(["rocm-smi", "--showmeminfo", "vram"], 500)
    return run

def split_env(cmd):
    extra = {}
    while re.match(r"[A-Za-z_][A-Za-z0-9_]*=\S", cmd):
        head, _, cmd = cmd.partition(" ")
        key, _, val = head.partition("=")
        if len(val) >= 2 and val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        extra[key] = val
    return extra, cmd.lstrip()

def exec_probe(ctx, cmd, timeout):
    expanded = subst(cmd, ctx["mapping"])
    env_extra, stripped = split_env(expanded)
    # comments=True: catalogue commands carry trailing `# why this row exists`
    # notes. Without it shlex hands `#` and every following word to the command
    # as argv, which silently changes the prompt / flags being measured.
    argv = shlex.split(stripped, comments=True)
    if not argv:
        return {"skipped": "empty command"}
    if argv[0] == "hipfire" and ctx["cli_bin"]: argv = [ctx["cli_bin"]] + argv[1:]
    first = os.path.basename(argv[0])
    if first not in ("cargo", "python3") and argv[0] != ctx["cli_bin"]:
        return {"skipped": "display-only; covered by session driver"}
    res = run_cmd(argv, dict(ctx["row_env"], **env_extra), timeout)
    idx = argv.index("--out") if "--out" in argv else -1
    res["out_path"] = argv[idx + 1] if 0 <= idx < len(argv) - 1 else None
    res["cmd"] = stripped
    return res

def needs_gpu(row):
    if row.get("fixtures") or row.get("positive_session") or row.get("negative_session"):
        return True
    blob = " ".join(row.get("commands", {}).get("positive", []))
    return any(k in blob for k in ("serve_harness", "redline_daemon", "hipfire run"))

def row_mapping(ctx, row, out):
    tags = row.get("fixtures", [])
    model = ctx["fixture_paths"].get(tags[0], "") if tags else ""
    base = os.path.basename(model)
    return {"$MODEL": model, "$MODEL_ALT": os.path.join(models_dir(), base), "$MODEL_DIR": os.path.dirname(model),
            "$MODELS_DIR": models_dir(), "$OUT": out, "$DAEMON_BIN": ctx["daemon_bin"] or "",
            "$CLI_BIN": ctx["cli_bin"] or "", "$PROMPT": row.get("prompt", ""),
            "$DAEMON_FI_BIN": os.environ.get("HIPFIRE_DAEMON_FI_BIN", "")}
def prompt_identity(ctx, row, mapping):
    if row.get("prompt"):
        return hashlib.md5(row["prompt"].encode()).hexdigest()
    for cmd in row.get("commands", {}).get("positive", []):
        for token in shlex.split(subst(cmd, mapping)):
            if token.endswith(".json") and os.path.isfile(token):
                try:
                    return md5_file(token)
                except OSError:
                    pass
    return ""
def fixture_identity(ctx, row):
    pairs = [(t, ctx["fixture_paths"].get(t, "")) for t in row.get("fixtures", [])]
    pairs += [(os.path.basename(p), p) for p in row.get("fixture_files", [])]
    out = []
    for tag, path in pairs:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        out.append({"tag": tag, "file": path, "size_bytes": size,
                    "sha256": sha256_cached(path, ctx["digests"]) if size >= 0 else "missing"})
    return out

def extend_outs(rows, results):
    for res in results:
        try:
            with open(res.get("out_path")) as fh:
                data = json.load(fh)
            rows.extend(data if isinstance(data, list) else [])
        except (TypeError, OSError, ValueError):
            pass

def run_negatives(ctx, row, mapping, env, timeout):
    pred = row.get("negative_predicate") or {}
    kind = pred.get("kind", "")
    if row.get("negative_session"):
        runs = [drive_session(ctx["daemon_bin"], row["negative_session"], mapping, env, timeout)]
        return eval_wire(runs, kind, pred), runs
    results = [r for r in (exec_probe(ctx, c, timeout) for c in row.get("commands", {}).get("negative", []))
               if "skipped" not in r]
    if kind == "expect_nonzero":
        return (bool(results) and all(r["rc"] != 0 for r in results),
                "negative: rcs=%s" % [r["rc"] for r in results]), results
    if kind == "cargo":
        return eval_cargo("\n".join(r["out"] for r in results), pred.get("min_passed", 1)), results
    if kind == "serve":
        rows = []
        extend_outs(rows, results)
        return eval_serve_rows(rows, False), results
    return (False, "negative: unknown predicate %s" % kind), results

def execute_row(ctx, row, out, timeout):
    mapping = row_mapping(ctx, row, out)
    env = {k: subst(v, mapping) for k, v in ctx["row_env"].items()}
    sub = dict(ctx, row_env=env)
    detail = {"mapping_model": mapping["$MODEL"]}
    if row.get("probe") == "none":
        return "rerun-required", "no existing probe: %s" % row.get("note", ""), detail
    if needs_gpu(row) and not has_gpu():
        return "hardware-blocked", "no HIP GPU (/dev/kfd and /dev/dri absent)", detail
    paths = [ctx["fixture_paths"].get(t, "") for t in row.get("fixtures", [])] + row.get("fixture_files", [])
    # Rows that carry their fixture inline as an env prefix (HIPFIRE_*_FIXTURE=
    # /path[,/path]) are just as host-dependent as a declared fixture: on a host
    # without that sidecar the command runs and reports zero tests, which is
    # indistinguishable from a broken filter. Gate on the paths themselves.
    for cmd in row.get("commands", {}).get("positive", []):
        for assign in re.findall(r'HIPFIRE_(?:\w*FIXTURE|DFLASH_DRAFT|\w*RESET_MODEL)="?([^"\s]+)"?', subst(cmd, mapping)):
            paths += [p for p in assign.split(",") if p.startswith("/")]
    # A fixture may legitimately be a directory (HIPFIRE_DENSE_FIXTURE points at
    # the model store, not one file).
    missing = [p for p in paths if not (os.path.isfile(p) or os.path.isdir(p))]
    if missing:
        return "hardware-blocked", "fixtures absent: %s" % missing, detail
    if row.get("positive_session") and mapping["$MODEL"]:
        link = os.path.join(out, "%s-alt.mq4" % row["id"])
        try:
            os.path.exists(link) and os.unlink(link)
            os.symlink(mapping["$MODEL"], link)
            mapping["$MODEL_ALT"] = link
        except OSError:
            pass
    pred = row.get("predicate", {})
    kind, repeat = pred.get("kind", ""), int(row.get("repeat", 1))
    positives, sessions = [], []
    if row.get("positive_session"):
        if not ctx["daemon_bin"]:
            return "failed", "no daemon binary (HIPFIRE_DAEMON_BIN or target/*/daemon)", detail
        for _ in range(repeat):
            sessions.append(drive_session(ctx["daemon_bin"], row["positive_session"], mapping, env, timeout))
            if sessions[-1].get("broken"):
                break
    for cmd in row.get("commands", {}).get("positive", []):
        positives += [r for r in [exec_probe(sub, cmd, timeout)] if "skipped" not in r]
    broken = next((s["broken"] for s in sessions if s.get("broken")), "")
    if broken:
        return "failed", "session driver: %s" % broken, detail
    detail["probes"] = [{"session": i, "broken": s.get("broken", ""),
                         "texts": [event_text(id_events(s, g))[:200] for g in s.get("generates", [])],
                         "terminals": [(terminal_of(s, g) or {}).get("type") for g in s.get("generates", [])]}
                        for i, s in enumerate(sessions)]
    detail["probes"] += [{"cmd": r.get("cmd", ""), "rc": r.get("rc"), "tail": r.get("out", "")[-2000:]}
                         for r in positives]
    # Documented per-arch capability gaps are not tree defects and must not be
    # filed as regressions. The PM4 replay path cannot dispatch a kernel that
    # needs scratch on GFX10/GFX11, and which kernel gets selected is arch
    # dependent — gfx1100 picks a scratch-free variant, gfx1151 does not.
    for res in positives:
        gap = next((mark for mark in CAPABILITY_GAPS if mark in res.get("out", "")), "")
        if gap:
            return "hardware-blocked", "capability gap on this arch: %s" % gap, detail
    if kind == "cargo":
        good, note = eval_cargo("\n".join(r["out"] for r in positives), pred.get("min_passed", 1))
    elif kind == "serve":
        rows = []
        extend_outs(rows, positives)
        if pred.get("expect_file"):
            good, note = eval_serve_expect(rows, pred["expect_file"])
        else:
            good, note = eval_serve_rows(rows, True)
    elif kind == "redline":
        try:
            reports = [json.load(open(r["out_path"])) for r in positives if r.get("out_path")]
        except (OSError, ValueError) as exc:
            return "failed", "unreadable redline report: %s" % exc, detail
        if not reports:
            return "failed", "redline: no report produced", detail
        good, note = eval_redline(reports[0], pred.get("require_stable", False))
    elif kind == "run":
        if not positives:
            return "failed", "run: no hipfire run output captured", detail
        oks = [eval_run_text(r["out"], pred.get("min_chars", 1)) for r in positives]
        good, note = all(o for o, _ in oks), "; ".join(m for _, m in oks)
    elif kind == "run_reload_equal":
        # Byte-identity is a claim about the DECODED TEXT. `out` merges stderr,
        # which carries the GPU banner, kernel-cache lines and per-run timings —
        # two identical decodes can never match there.
        texts = [r.get("stdout", r["out"]) for r in positives if r["rc"] == 0]
        groups = pred.get("identical_text_groups", [])
        clean = len(texts) >= 2 and all(r["rc"] == 0 for r in positives)
        clean = clean and all(eval_run_text(t, pred.get("min_chars", 1))[0] for t in texts)
        good = clean and all(texts[a] == texts[b] for a, b in groups)
        note = "run: %d runs, groups %s identical=%s" % (len(texts), groups, good)
    elif kind.startswith("wire_"):
        good, note = eval_wire(sessions, kind, pred)
    else:
        return "failed", "unknown predicate %s" % kind, detail
    detail["positive_note"] = note
    if not good:
        return "failed", "positive: %s" % note, detail
    if row.get("negative_session") or row.get("commands", {}).get("negative"):
        (ngood, nnote), _ = run_negatives(ctx, row, mapping, env, timeout)
        detail["negative_note"] = nnote
        if not ngood:
            return "failed", "negative: %s" % nnote, detail
    return "current", note, detail

def make_receipt(ctx, row, out, disposition, note, detail, elapsed):
    mapping = row_mapping(ctx, row, out)
    receipt = {"schema": "device-mesh-matrix-receipt-v1", "row": row["id"],
               "milestone": row["milestone"], "route": row.get("route", ""),
               "hosts": row.get("hosts", []), "probe": row.get("probe", ""),
               "produced_utc": datetime.now(timezone.utc).isoformat(),
               "git": ctx["git"], "platform": ctx["platform"],
               "binaries": {"hipfire": ctx["cli_bin"], "hipfire_md5": ctx["cli_md5"],
                            "daemon": ctx["daemon_bin"], "daemon_md5": ctx["daemon_md5"]},
               "fixtures": fixture_identity(ctx, row), "prompt_md5": prompt_identity(ctx, row, mapping),
               "commands": row.get("commands", {}), "predicate": row.get("predicate", {}),
               "negative_predicate": row.get("negative_predicate", {}),
               "result": {"elapsed_s": elapsed, "note": note, "detail": detail},
               "disposition": disposition}
    path = os.path.join(out, "receipts", "%s.json" % row["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(receipt, open(path, "w"), indent=2)
    return path

def show_plan(rows, ctx):
    for row in rows:
        m = {"$MODEL": ctx["fixture_paths"].get((row.get("fixtures", [""]) or [""])[0], "$MODEL"),
             "$MODEL_ALT": "$MODEL_ALT", "$MODEL_DIR": "$MODEL_DIR",
             "$MODELS_DIR": models_dir(), "$OUT": "$OUT", "$DAEMON_BIN": ctx["daemon_bin"] or "$DAEMON_BIN",
             "$CLI_BIN": ctx["cli_bin"] or "hipfire", "$PROMPT": row.get("prompt", ""),
             "$DAEMON_FI_BIN": os.environ.get("HIPFIRE_DAEMON_FI_BIN", "")}
        print("== %s [%s] %s probe=%s ==" % (row["id"], row["milestone"],
                                             row.get("route", ""), row.get("probe", "")))
        print("   hosts: %s fixtures: %s" % (",".join(row.get("hosts", [])), ",".join(row.get("fixtures", [])) or "-"))
        for mark, key in (("+", "positive"), ("-", "negative")):
            for cmd in row.get("commands", {}).get(key, []):
                print("   %s %s" % (mark, subst(cmd, m)))
        for key, mark in (("positive_session", "+"), ("negative_session", "-")):
            if row.get(key):
                extra = " x%d" % row["repeat"] if key == "positive_session" and row.get("repeat", 1) > 1 else ""
                print("   %s $DAEMON_BIN JSONL session: %d ops%s" % (mark, len(row[key]), extra))
        if row.get("probe") == "none":
            print("   !! no existing probe: %s" % row.get("note", ""))

def self_test():
    checks = []
    def check(name, cond):
        checks.append((name, bool(cond)))
    good, msg = eval_cargo(SAMPLE_CARGO_PASS, 5)
    check("cargo-pass", good and "passed=5" in msg)
    check("cargo-min-bound", not eval_cargo(SAMPLE_CARGO_PASS, 6)[0])
    check("cargo-fail", not eval_cargo(SAMPLE_CARGO_FAIL, 1)[0])
    check("cargo-no-summary", not eval_cargo("nothing here", 1)[0])
    check("cargo-not-bare-ok", parse_cargo("all ok folks")["passed"] == 0)
    check("serve-clean", eval_serve_rows(SAMPLE_SERVE_CLEAN, True)[0])
    check("serve-dirty-fails", not eval_serve_rows(SAMPLE_SERVE_DIRTY, True)[0])
    check("serve-negative-accounting", eval_serve_rows(SAMPLE_SERVE_DIRTY, False)[0])
    check("serve-empty-fails", not eval_serve_rows([], True)[0])
    # Truncation is not degeneracy: a healthy answer that reaches the token cap
    # must pass, a repetition loop or a cap-hit-with-no-text must not.
    long_ok = [{"finish": "length", "runaway": True,
                "content": " ".join("w%d" % i for i in range(400))}]
    check("serve-long-truncated-passes", eval_serve_rows(long_ok, True)[0])
    check("serve-repetition-loop-fails",
          not eval_serve_rows([{"finish": "length", "runaway": True,
                                "content": "yes no " * 400}], True)[0])
    check("serve-stub-truncation-fails",
          not eval_serve_rows([{"finish": "length", "runaway": True,
                                "content": "short " * 10}], True)[0])
    check("redline-pass", eval_redline(SAMPLE_REDLINE, True)[0])
    # Real reports carry sibling fields next to `sequence_stable` (`measurement`,
    # `captures`, `context_tokens`). A shape-blind walk over decode.values()
    # failed a passing report on the `measurement` dict; pin the real shape.
    real_decode = {"pass": True, "prefill": {"2": {"sequence_stable": True}},
                   "decode": {"context_tokens": 124, "sequence_stable": True,
                              "captures": [{"type": "decode_result"}],
                              "measurement": {"tok_s": 32.9}}}
    check("redline-real-decode-shape", eval_redline(real_decode, True)[0])
    unstable = json.loads(json.dumps(real_decode))
    unstable["decode"]["sequence_stable"] = False
    check("redline-real-decode-unstable-fails", not eval_redline(unstable, True)[0])
    check("redline-tamper", not eval_redline({"pass": True, "prefill": {"1": {}}}, True)[0])
    check("redline-pass-false", not eval_redline({"pass": False}, False)[0])
    # exec_probe must RETURN its result: falling off the end made every
    # cargo/serve row raise "NoneType is not a container".
    probe_ctx = {"mapping": {}, "row_env": {}, "cli_bin": "", "daemon_bin": ""}
    check("probe-returns-dict", isinstance(exec_probe(probe_ctx, "python3 -c pass", 60), dict))
    check("probe-reports-rc", exec_probe(probe_ctx, "python3 -c pass", 60).get("rc") == 0)
    check("probe-strips-trailing-comment",
          shlex.split("cmd --flag x  # note words here", comments=True) == ["cmd", "--flag", "x"])
    check("probe-keeps-quoted-hash",
          shlex.split('cmd "a # b"', comments=True) == ["cmd", "a # b"])
    check("probe-skips-display-only",
          "skipped" in exec_probe(probe_ctx, "some-display-only-command", 60))
    check("run-text", eval_run_text("Paris is the capital.\n", 1)[0])
    check("run-forbidden", not eval_run_text("daemon error: [CLS-001] nope", 1)[0])
    run = {"events": SAMPLE_WIRE_DONE, "generates": ["r1"], "loaded": [SAMPLE_WIRE_DONE[0]]}
    check("wire-done", eval_wire([run], "wire_done_clean", {})[0])
    check("wire-text-extract", event_text(id_events(run, "r1")) == "hi")
    check("wire-single-terminal", terminal_of(run, "r1").get("type") == "done")
    check("wire-vl", eval_wire([run], "wire_vl_false_then_done", {})[0])
    bad_vl = {"events": [{"type": "loaded", "vl": True}], "generates": [], "loaded": [{"vl": True}]}
    check("wire-vl-true-fails", not eval_wire([bad_vl], "wire_vl_false_then_done", {})[0])
    # G4.6 race oracle: three concurrently admitted ids resolving to done /
    # cancel-as-done / error, with duplicate-commit + late-abort +
    # wrong-attempt-abort after the terminals producing no further bytes.
    race_clean = {"events": [
        {"type": "token", "id": "r1", "text": "hi"},
        {"type": "commit_ready", "id": "r1"},
        {"type": "error", "id": "r3", "message": "seed must be non-negative, got -1"},
        {"type": "aborted", "id": "r2", "reason": "client_cancelled", "attempt_id": 1},
        {"type": "done", "id": "r1", "tokens": 1},
        {"type": "done", "id": "r2", "finish_reason": "aborted", "completion_tokens": 0}],
        "generates": ["r1", "r2", "r3"], "loaded": []}
    race_pred = {"expect": {"r1": "done", "r2": "done", "r3": "error"}}
    check("wire-race-clean", eval_wire([race_clean], "wire_exactly_one_terminal", race_pred)[0])
    def race_variant(extra=None, drop_terminal=None, extra_terminal=None):
        run = json.loads(json.dumps(race_clean))
        if drop_terminal:
            run["events"] = [e for e in run["events"]
                             if not (e.get("id") == drop_terminal and e.get("type") in ("done", "error"))]
        if extra_terminal:
            run["events"].append(dict(extra_terminal))
        if extra:
            run["events"].append(dict(extra))
        return run
    check("wire-race-dup-terminal-fails",
          not eval_wire([race_variant(extra_terminal={"type": "done", "id": "r1", "tokens": 1})],
                        "wire_exactly_one_terminal", race_pred)[0])
    check("wire-race-post-terminal-token-fails",
          not eval_wire([race_variant(extra={"type": "token", "id": "r1", "text": "late"})],
                        "wire_exactly_one_terminal", race_pred)[0])
    check("wire-race-missing-terminal-fails",
          not eval_wire([race_variant(drop_terminal="r2")],
                        "wire_exactly_one_terminal", race_pred)[0])
    check("wire-race-unknown-id-terminal-fails",
          not eval_wire([race_variant(extra_terminal={"type": "done", "id": "rx", "tokens": 1})],
                        "wire_exactly_one_terminal", race_pred)[0])
    for name, ok in checks:
        print("   [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("device_mesh_matrix: self-test %s (%d checks)" % ("OK" if all(ok for _, ok in checks) else "FAILED", len(checks)))
    return 0 if all(ok for _, ok in checks) else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description="G1-G4 device-mesh acceptance matrix runner")
    ap.add_argument("--plan", action="store_true", help="print rows, run nothing")
    ap.add_argument("--host-class", default="", help="filter rows by host class")
    ap.add_argument("--only", default="", help="run a single row id")
    ap.add_argument("--out", default="", help="receipt directory")
    ap.add_argument("--timeout", type=int, default=900, help="per-command seconds")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    root = repo_root()
    rows = load_catalogue(root)["rows"]
    if args.only:
        rows = [r for r in rows if r["id"] == args.only]
        if not rows:
            print("unknown row: %s" % args.only, file=sys.stderr)
            return 2
    elif args.host_class:
        rows = [r for r in rows if args.host_class in r.get("hosts", [])]
    cli_bin, daemon_bin = find_bins(root)
    out = os.path.abspath(args.out) if args.out else os.path.join(root, ".matrix-out")
    ctx = {"cli_bin": cli_bin, "daemon_bin": daemon_bin, "git": git_identity(root), "platform": platform_probe(),
           "fixture_paths": {}, "cli_md5": md5_file(cli_bin) if cli_bin else "missing",
           "daemon_md5": md5_file(daemon_bin) if daemon_bin else "missing", "digests": {}}
    for tag in {t for r in rows for t in r.get("fixtures", [])}:
        ctx["fixture_paths"][tag] = resolve_fixture(root, tag)
    if args.plan:
        show_plan(rows, ctx)
        unmapped = [r["id"] for r in rows if r.get("probe") == "none"]
        print("no existing probe: %s" % ", ".join(unmapped) if unmapped else "all rows mapped")
        return 0
    # Evidence integrity: --host-class only SELECTS rows; it does not prove which
    # GPU answered. A wrong HIP_VISIBLE_DEVICES index silently files a whole run
    # under the wrong architecture (observed: a "gfx1151" run that actually ran
    # on an 8 GB gfx1010 and OOMed). Ask the daemon which device it opened.
    if args.host_class and daemon_bin:
        # The daemon initializes HIP lazily, on the first message: an immediate
        # EOF prints no banner. A ping is the cheapest thing that opens device 0.
        probe = run_cmd(["sh", "-c", 'printf \'{"type":"ping"}\\n\' | "$0"', daemon_bin], {}, 120)
        seen = re.findall(r"GPU dev \d+: (gfx\w+)", probe.get("out", ""))
        if not seen:
            print("cannot determine GPU arch from %s; refusing to file mislabeled evidence"
                  % daemon_bin, file=sys.stderr)
            return 2
        if seen[0] != args.host_class:
            print("host-class mismatch: --host-class %s but device 0 is %s "
                  "(check HIP_VISIBLE_DEVICES=%r)"
                  % (args.host_class, seen[0], os.environ.get("HIP_VISIBLE_DEVICES", "")),
                  file=sys.stderr)
            return 2
        ctx["platform"]["gpu_arch_verified"] = seen[0]
    os.makedirs(os.path.join(out, "receipts"), exist_ok=True)
    ctx["digests"], cache_path = load_digest_cache(out)
    summary = {"schema": "device-mesh-matrix-summary-v1", "produced_utc": datetime.now(timezone.utc).isoformat(),
               "git": ctx["git"], "host_class": args.host_class or "all", "rows": []}
    for row in rows:
        sub = dict(ctx, mapping=row_mapping(ctx, row, out), row_env=dict(row.get("env", {})))
        start = time.time()
        try:
            # A row may declare its own per-command budget (the vision matrix
            # legitimately runs ~15 min on gfx1201 and has an 1800 s watchdog).
            disposition, note, detail = execute_row(sub, row, out, int(row.get("timeout", args.timeout)))
        except Exception as exc:  # never abort the matrix on one row
            disposition, note, detail = "failed", "runner exception: %s" % exc, {}
        elapsed = round(time.time() - start, 1)
        path = make_receipt(ctx, row, out, disposition, note, detail, elapsed)
        summary["rows"].append({"id": row["id"], "milestone": row["milestone"], "disposition": disposition,
                                "note": note, "elapsed_s": elapsed, "receipt": path})
        print("%-28s %-6s %-16s %s" % (row["id"], row["milestone"], disposition, note))
    json.dump(ctx["digests"], open(cache_path, "w"), indent=2)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    counts = {}
    for e in summary["rows"]:
        counts[e["disposition"]] = counts.get(e["disposition"], 0) + 1
    print("summary: %s" % json.dumps(counts))
    return 0

if __name__ == "__main__":
    sys.exit(main())
