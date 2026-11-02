#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU tests for scripts/agentic_gate_driver.py (the agentic-gate daemon driver).

A fake daemon enforcing the real wire rules (scripts/tests/fake_agentic_daemon.py)
stands in for target/release/daemon. The Rust side of the proof is
`agentic_gate_driver_requests_pass_daemon_wire_validation` in
crates/hipfire-daemon/src/main.rs, which replays the same fixture through the
daemon's real validation.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import agentic_gate_driver as driver  # noqa: E402

DRIVER = SCRIPTS / "agentic_gate_driver.py"
FAKE = SCRIPTS / "tests" / "fake_agentic_daemon.py"
FIXTURE = SCRIPTS / "tests" / "fixtures" / "agentic-gate-requests.jsonl"
GATE = SCRIPTS / "agentic-gate.sh"

FIXTURE_CELLS = [
    "m.mq4|s1|1|1|clamp-multi",
    "m.mq4|s2|0|0|think-on",
    "m.mq4|s3|0|1|on-multi",
]
FIXTURE_SYSTEMS = ["SYS-1", "SYS-2", "SYS-3"]


def fixture_text():
    msgs = driver.build_session(
        "/models/m.mq4", FIXTURE_CELLS, "USER PROMPT", FIXTURE_SYSTEMS
    )
    return "".join(json.dumps(m) + "\n" for m in msgs)


def test_driver_requests_match_checked_in_fixture():
    if os.environ.get("UPDATE_FIXTURE") == "1":
        FIXTURE.write_text(fixture_text())
    assert FIXTURE.read_text() == fixture_text()


def test_request_shapes():
    msgs = driver.build_requests(FIXTURE_CELLS, "U", FIXTURE_SYSTEMS)
    assert [m["id"] for m in msgs] == ["c0_t1", "c0_t2", "c1_t1", "c2_t1", "c2_t2"]
    assert [m["attempt_id"] for m in msgs] == [1, 2, 3, 4, 5]
    for m in msgs:
        assert isinstance(m["attempt_id"], int) and m["attempt_id"] > 0
        assert "thinking" not in m and "max_think_tokens" not in m
        assert m["thinking_enabled"] is (m["assistant_prefix"] == "open_think")
    # clamp cell and every second turn think off; the other first turns think on
    assert [m["thinking_enabled"] for m in msgs] == [False, False, True, True, False]


def _cells(tmp_path):
    cells = []
    for i, (spec, text) in enumerate(zip(FIXTURE_CELLS, FIXTURE_SYSTEMS)):
        sys_file = tmp_path / f"sys{i}.txt"
        sys_file.write_text(text)
        parts = spec.split("|")
        parts[1] = str(sys_file)
        cells.append("|".join(parts))
    user = tmp_path / "user.txt"
    user.write_text("USER PROMPT")
    return cells, user


def _run_driver(tmp_path, exe, extra=(), env=None, timeout=60):
    cells, user = _cells(tmp_path)
    out = tmp_path / "out.txt"
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--exe", str(exe), "--model", "/models/m.mq4",
         "--user-file", str(user), "--out", str(out), *extra, *cells],
        capture_output=True, text=True, env=full_env, timeout=timeout,
    )
    return proc, out, cells


def _events(path):
    evs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            evs.append(json.loads(line))
    return evs


def _gate_parser():
    """The detector/report heredoc from agentic-gate.sh, verbatim."""
    lines = GATE.read_text().splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.strip().startswith('python3 - "$OUTPUT_FILE" "$OUT"') and "<<'PY'" in l
    )
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "PY")
    return "\n".join(lines[start + 1:end]) + "\n"


def test_driver_end_to_end_against_fake_daemon(tmp_path):
    proc, out, cells = _run_driver(
        tmp_path, FAKE, env={"FAKE_COMMIT_TIMEOUT": "2", "FAKE_READY_DELAY": "0.2"}
    )
    assert proc.returncode == 0, proc.stderr
    evs = _events(out)
    assert not [e for e in evs if e["type"] == "error"], evs
    assert [e["type"] for e in evs if e["type"] in ("loaded", "unloaded")] == [
        "loaded", "unloaded"
    ]
    done = [(e["id"], e["attempt_id"]) for e in evs if e["type"] == "done"]
    assert done == [("c0_t1", 1), ("c0_t2", 2), ("c1_t1", 3), ("c2_t1", 4), ("c2_t2", 5)]
    # Stderr of the daemon is captured into the same file, as `2>&1` was.
    assert "fake-daemon: started" in out.read_text()

    # The gate's own parser scores this output without a hard fail.
    report = tmp_path / "report.md"
    report.write_text("")
    parsed = subprocess.run(
        [sys.executable, "-", str(out), str(report), *cells],
        input=_gate_parser(), capture_output=True, text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    summary = json.loads(parsed.stderr.strip().splitlines()[-1])
    assert summary == {"hard_fail": False, "soft_warn_count": 0}
    text = report.read_text()
    assert text.count("HARD_FAIL") == 0
    assert "zero tokens" not in text
    assert "json ok: name='read_file'" in text


def _fake_session(tmp_path, env):
    full_env = dict(os.environ)
    full_env.update(env)
    return subprocess.Popen(
        [sys.executable, str(FAKE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, env=full_env,
    )


def _send(p, obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()


def _read_until(p, pred, limit=20):
    evs = []
    for _ in range(limit):
        line = p.stdout.readline()
        if not line:
            break
        ev = json.loads(line)
        evs.append(ev)
        if pred(ev):
            break
    return evs


@pytest.mark.parametrize("bad", [None, 0, "7", True, 1.5])
def test_fake_rejects_generate_without_valid_attempt_id(tmp_path, bad):
    p = _fake_session(tmp_path, {})
    try:
        req = {"type": "generate", "id": "x", "prompt": "p"}
        if bad is not None:
            req["attempt_id"] = bad
        _send(p, req)
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert [e["type"] for e in evs] == ["error"]
        assert evs[0]["id"] == "x"
    finally:
        p.kill()
        p.wait()


def test_fake_ignores_early_commit_then_aborts(tmp_path):
    # commit_ready is delayed; the commit sent right after generate is early.
    p = _fake_session(tmp_path, {"FAKE_READY_DELAY": "0.5", "FAKE_COMMIT_TIMEOUT": "1"})
    try:
        _send(p, {"type": "generate", "id": "x", "attempt_id": 1, "prompt": "p"})
        _send(p, {"type": "commit", "id": "x", "attempt_id": 1})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        types = [e["type"] for e in evs]
        assert "commit_ready" in types
        assert types[-1] == "error" and "done" not in types
    finally:
        p.kill()
        p.wait()


def test_fake_aborts_without_commit_and_done_needs_matching_commit(tmp_path):
    p = _fake_session(tmp_path, {"FAKE_COMMIT_TIMEOUT": "1"})
    try:
        _send(p, {"type": "generate", "id": "x", "attempt_id": 1, "prompt": "p"})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "error"
        # wrong attempt_id is not a matching commit
        _send(p, {"type": "generate", "id": "y", "attempt_id": 2, "prompt": "p"})
        _read_until(p, lambda e: e["type"] == "commit_ready")
        _send(p, {"type": "commit", "id": "y", "attempt_id": 99})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "error"
        # a matching commit completes
        _send(p, {"type": "generate", "id": "z", "attempt_id": 3, "prompt": "p"})
        _read_until(p, lambda e: e["type"] == "commit_ready")
        _send(p, {"type": "commit", "id": "z", "attempt_id": 3})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "done" and evs[-1]["attempt_id"] == 3
    finally:
        p.kill()
        p.wait()


def test_pre_protocol_requests_fail_against_fake(tmp_path):
    """The old gate's request shape (no attempt_id) would get an error, not tokens."""
    p = _fake_session(tmp_path, {})
    try:
        _send(p, {"type": "generate", "id": "c0_t1", "prompt": "p", "thinking": False,
                  "max_think_tokens": 1})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert [e["type"] for e in evs] == ["error"]
    finally:
        p.kill()
        p.wait()


def test_driver_passes_the_callers_full_environment_to_the_daemon(tmp_path):
    """A fabric hold's private lock dir and admission vars must reach the daemon."""
    sentinel = {
        "HIPFIRE_LOCK_DIR": str(tmp_path / "fabric-locks"),
        "FABRIC_JOB": "job-123",
        "ROCR_VISIBLE_DEVICES": "1",
        "HIPFIRE_DEVICES": "1",
        "AGENTIC_SENTINEL": "sentinel-value",
        "HIPFIRE_KV_MODE": "asym3",  # the gate's own `env` override
        "HIPFIRE_GRAPH": "1",
    }
    proc, out, _ = _run_driver(
        tmp_path, FAKE,
        env={**sentinel, "FAKE_ECHO_ENV": ",".join(sentinel)},
    )
    assert proc.returncode == 0, proc.stderr
    loaded = next(e for e in _events(out) if e["type"] == "loaded")
    assert loaded["env"] == sentinel


def test_gate_launches_driver_with_its_env_overrides_only():
    """The gate adds HIPFIRE_KV_MODE/HIPFIRE_GRAPH via `env`; no clean env anywhere."""
    text = GATE.read_text()
    assert re.search(
        r"env HIPFIRE_KV_MODE=asym3 \\\n\s+HIPFIRE_GRAPH=1 \\\n"
        r"\s+python3 scripts/agentic_gate_driver\.py",
        text,
    )
    assert "env -i" not in text
    assert "env -i" not in DRIVER.read_text()


def test_driver_deadline_kills_only_its_hung_daemon(tmp_path):
    pidfile = tmp_path / "daemon.pid"
    t0 = time.monotonic()
    proc, out, cells = _run_driver(
        tmp_path, FAKE,
        extra=["--generate-timeout", "1", "--unload-timeout", "1"],
        env={"FAKE_HANG": "1", "FAKE_PIDFILE": str(pidfile)},
    )
    assert time.monotonic() - t0 < 30
    assert proc.returncode == 1
    evs = _events(out)
    errs = [e for e in evs if e["type"] == "error"]
    assert errs and errs[0]["source"] == "agentic_gate_driver"
    assert errs[0]["id"] == "c0_t1" and "deadline" in errs[0]["message"]
    assert not [e for e in evs if e["type"] == "done"]
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the driver's own daemon is gone


def test_driver_records_load_deadline_and_spawn_failure(tmp_path):
    proc, out, _ = _run_driver(tmp_path, tmp_path / "does-not-exist")
    assert proc.returncode == 1
    errs = [e for e in _events(out) if e["type"] == "error"]
    assert errs and "spawn failed" in errs[0]["message"]

    silent = tmp_path / "silent.sh"
    silent.write_text("#!/bin/sh\nexec sleep 30\n")
    silent.chmod(0o755)
    proc, out, _ = _run_driver(tmp_path, silent, extra=["--load-timeout", "1"])
    assert proc.returncode == 1
    errs = [e for e in _events(out) if e["type"] == "error"]
    assert errs and "load deadline" in errs[0]["message"]


# -- fail-closed negatives: driver rc AND the real gate ------------------------
# Each scenario has valid tool-call tokens (or a clean done) and still must not
# pass: the driver exits nonzero and `scripts/agentic-gate.sh --fast` run against
# the fake daemon exits 1 with a DRIVER FAILURE report line.

SHORT = "--load-timeout 5 --generate-timeout 1 --unload-timeout 1"

NEGATIVES = {
    "eof_before_done": ({"FAKE_FAULT": "eof_before_done"}, "daemon exited during generate"),
    "error_with_id": ({"FAKE_FAULT": "error_after_tokens"}, "fake fault after tokens"),
    "error_null_id": ({"FAKE_FAULT": "error_null_id"}, "fake fault with no id"),
    "generate_deadline": ({"FAKE_FAULT": "hang_after_tokens"}, "generate deadline"),
    "exit1_after_done": ({"FAKE_EXIT_AFTER_DONE": "1", "FAKE_EXIT_CODE": "3"}, None),
    "exit1_at_unload": ({"FAKE_EXIT_ON_UNLOAD": "1", "FAKE_EXIT_CODE": "3"}, "status 3"),
    "unload_eof": ({"FAKE_EOF_ON_UNLOAD": "1"}, "exited before unloaded"),
}


def _run_gate(tmp_path, env, mode="--fast"):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "qwen3.6-27b.mq4").write_text("not a model; the fake daemon never reads it")
    report = tmp_path / "gate-report.md"
    full_env = dict(os.environ)
    full_env.pop("HIPFIRE_SKIP_AGENTIC_GATE", None)
    full_env.update({
        "HIPFIRE_AGENTIC_GATE_EXE": str(FAKE),
        "HIPFIRE_AGENTIC_GATE_DRIVER_ARGS": SHORT,
        "HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK": "1",
        "HIPFIRE_AGENTIC_GATE_OUT": str(report),
        "HIPFIRE_MODELS_DIR": str(models),
        "HIPFIRE_GPU_LOCKFILE": str(tmp_path / "gpu.lock"),
        "ROCR_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "HIPFIRE_DEVICES": "-1",
    })
    full_env.update(env)
    proc = subprocess.run(
        ["bash", str(GATE), mode], capture_output=True, text=True, env=full_env, timeout=120
    )
    return proc, report.read_text() if report.exists() else ""


def test_gate_passes_against_clean_fake_daemon(tmp_path):
    """Positive control: the harness itself is sound, so the negatives mean something."""
    proc, report = _run_gate(tmp_path, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "HARD_FAIL" not in report and "DRIVER FAILURE" not in report
    assert "json ok: name='read_file'" in report


@pytest.mark.parametrize("name", sorted(NEGATIVES))
def test_driver_and_gate_fail_closed(tmp_path, name):
    env, expect = NEGATIVES[name]
    drv_dir = tmp_path / "drv"
    drv_dir.mkdir()
    dproc, out, _ = _run_driver(drv_dir, FAKE, extra=SHORT.split(), env=env)
    assert dproc.returncode != 0, (dproc.stdout, dproc.stderr)
    errs = [e for e in _events(out) if e["type"] == "error"]
    assert errs, "no error event recorded"
    if expect:
        assert any(expect in str(e.get("message")) for e in errs), errs
    # Valid tool-call tokens were seen where the scenario reaches them.
    assert "<tool_call>" in out.read_text()

    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    proc, report = _run_gate(gate_dir, env)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "HARD FAIL" in proc.stdout
    assert "DRIVER FAILURE" in report
    assert "agentic-gate: PASS" not in proc.stdout


def test_unload_send_failure_is_an_error(tmp_path):
    class NoUnload(driver.Driver):
        def send(self, obj):
            return False if obj.get("type") == "unload" else super().send(obj)

    cells, user = _cells(tmp_path)
    d = NoUnload(str(FAKE), str(tmp_path / "o.txt"))
    reqs = driver.build_requests(cells, "U", ["a", "b", "c"])
    assert d.run("/models/m.mq4", reqs) == 1
    assert "daemon stdin closed before unload" in d.errors


# -- parser: a matching done and no error are required, whatever tokens were seen --

def _score(tmp_path, lines, multi="0"):
    sysf = tmp_path / "sys.txt"
    sysf.write_text("SYS")
    cell = f"m.mq4|{sysf}|0|{multi}|lbl"
    out = tmp_path / "events.txt"
    out.write_text("".join(json.dumps(l) + "\n" for l in lines))
    report = tmp_path / "score.md"
    report.write_text("")
    parsed = subprocess.run(
        [sys.executable, "-", str(out), str(report), cell],
        input=_gate_parser(), capture_output=True, text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    return report.read_text()


TOK = {"type": "token", "id": "c0_t1",
       "text": '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>'}
DONE = {"type": "done", "id": "c0_t1", "attempt_id": 1}


def test_parser_passes_a_cell_with_a_matching_done(tmp_path):
    report = _score(tmp_path, [TOK, DONE])
    assert "lbl    PASS" in report


@pytest.mark.parametrize("label,events", [
    ("eof before done", [TOK]),
    ("commit_ready only", [TOK, {"type": "commit_ready", "id": "c0_t1", "attempt_id": 1}]),
    ("done wrong attempt", [TOK, {"type": "done", "id": "c0_t1", "attempt_id": 2}]),
    ("done no attempt", [TOK, {"type": "done", "id": "c0_t1"}]),
    ("done wrong id", [TOK, {"type": "done", "id": "c9_t1", "attempt_id": 1}]),
    ("error with id", [TOK, {"type": "error", "id": "c0_t1", "message": "boom"}, DONE]),
    ("error null id in flight", [TOK, {"type": "error", "id": None, "message": "boom"}, DONE]),
    ("error no id key in flight", [TOK, {"type": "error", "message": "boom"}, DONE]),
    ("error after done, no id", [TOK, DONE, {"type": "error", "message": "late"}]),
    ("driver synthetic error", [TOK, DONE, {"type": "error", "source": "agentic_gate_driver",
                                            "message": "unload deadline 30s exceeded"}]),
    ("driver synthetic error with id", [TOK, {"type": "error", "source": "agentic_gate_driver",
                                              "id": "c0_t1", "message": "generate deadline"}]),
])
def test_parser_hard_fails_without_done_or_with_error(tmp_path, label, events):
    report = _score(tmp_path, events)
    assert "lbl    HARD_FAIL" in report, (label, report)


def test_parser_requires_done_for_the_multiturn_second_request(tmp_path):
    t2 = {"type": "token", "id": "c0_t2", "text": "ok"}
    report = _score(tmp_path, [TOK, DONE, t2], multi="1")
    assert "lbl    HARD_FAIL" in report and "c0_t2: no matching done (attempt_id 2)" in report
    report = _score(
        tmp_path, [TOK, DONE, t2, {"type": "done", "id": "c0_t2", "attempt_id": 2}], multi="1"
    )
    assert "lbl    PASS" in report


# -- host ownership: signal only an unreaped leader we still own --------------

@pytest.fixture
def killpg_calls(monkeypatch):
    calls = []
    real = os.killpg
    monkeypatch.setattr(driver.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)) or real(pgid, sig))
    return calls


def _started(tmp_path):
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    assert d.start()
    return d


def _finish(d):
    for t in d._threads:
        t.join(timeout=5)
    d._out.close()


def test_lost_leader_is_never_signalled(tmp_path, killpg_calls):
    d = _started(tmp_path)
    pid = d.proc.pid
    os.kill(pid, 9)           # the test, not the driver, ends and reaps the leader
    os.waitpid(pid, 0)
    d.kill()
    assert killpg_calls == []
    assert d.exit_status == "unknown"  # no exit status is invented
    assert d.errors == ["process group ownership lost; not signalled"]
    d.kill()                  # idempotent: still nothing sent, nothing re-recorded
    assert killpg_calls == [] and len(d.errors) == 1
    _finish(d)


def test_waitid_childprocesserror_means_lost_in_a_full_session(tmp_path, monkeypatch, killpg_calls):
    def lost(*_a, **_k):
        raise ChildProcessError

    monkeypatch.setattr(driver.os, "waitid", lost)
    cells, _ = _cells(tmp_path)
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    rc = d.run("/models/m.mq4", driver.build_requests(cells, "U", ["a", "b", "c"]))
    try:
        os.waitpid(d.proc.pid, 0)  # the fake exits on stdin EOF; clean up the zombie
    except ChildProcessError:
        pass
    assert rc == 1
    assert killpg_calls == []
    assert d.exit_status == "unknown"
    assert "process group ownership lost; not signalled" in d.errors


def test_signal_after_the_leader_was_reaped_sends_nothing(tmp_path, killpg_calls):
    cells, _ = _cells(tmp_path)
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    assert d.run("/models/m.mq4", driver.build_requests(cells, "U", ["a", "b", "c"])) == 0
    assert d.exit_status == 0 and d.proc.returncode == 0
    d.kill()                  # what on_signal does after run()'s wait reaped the leader
    assert killpg_calls == []
    assert d.errors == []


def test_kill_reaps_only_after_the_last_group_signal(tmp_path, killpg_calls):
    d = _started(tmp_path)
    pid = d.proc.pid
    d.kill()
    assert [c[0] for c in killpg_calls] == [pid, pid]  # TERM then the sweeping KILL
    assert d.proc.returncode is not None and d.exit_status == d.proc.returncode
    killpg_calls.clear()
    d.kill()                  # reaped: no further group signal, even for a reused pgid
    assert killpg_calls == []
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    _finish(d)


def test_sigterm_to_the_driver_kills_only_its_group_once(tmp_path):
    pidfile = tmp_path / "daemon.pid"
    cells, user = _cells(tmp_path)
    out = tmp_path / "out.txt"
    env = dict(os.environ, FAKE_HANG="1", FAKE_PIDFILE=str(pidfile))
    p = subprocess.Popen(
        [sys.executable, str(DRIVER), "--exe", str(FAKE), "--model", "/models/m.mq4",
         "--user-file", str(user), "--out", str(out), *cells],
        env=env, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if out.exists() and '"type":"loaded"' in out.read_text().replace(" ", ""):
                break
            time.sleep(0.05)
        else:
            pytest.fail("daemon never reported loaded")
        p.terminate()
        assert p.wait(timeout=30) == 143
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


# -- raw archive and terminal record -----------------------------------------

def _raw_run(tmp_path, env=None, mode_env=True):
    raw = tmp_path / "raw"
    run_env = dict(env or {})
    if mode_env:
        run_env["HIPFIRE_AGENTIC_GATE_RAW_DIR"] = str(raw)
    proc, report = _run_gate(tmp_path, run_env)
    return proc, report, raw


def _record(raw):
    return json.loads((raw / "qwen3.6-27b.driver.json").read_text())


def test_raw_dir_archives_output_and_record_for_a_clean_run(tmp_path):
    proc, _report, raw = _raw_run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = (raw / "qwen3.6-27b.daemon-output.jsonl").read_text()
    assert '"type":"done"' in log and "fake-daemon: started" in log  # stdout JSON and stderr
    rec = _record(raw)
    assert rec["driver_rc"] == 0 and rec["loaded"] is True
    (req,) = rec["requests"]
    assert (req["id"], req["attempt_id"]) == ("c0_t1", 1)
    assert req["commit_ready_seen"] and req["commit_sent"] and req["done_seen"]
    assert req["finish_reason"] == "stop" and req["tokens"] == 1 and req["errors"] == []
    assert rec["unload"] == "unloaded"
    assert rec["daemon"]["exit_status"] == 0
    assert rec["daemon"]["started"] and not rec["daemon"]["killed_by_driver"]
    assert isinstance(rec["daemon"]["peak_rss_kib"], int) and rec["daemon"]["peak_rss_kib"] > 0
    assert "VmHWM" in rec["daemon"]["peak_rss_measured"]
    assert rec["errors"] == [] and rec["session_errors"] == []
    assert not list(raw.glob("*.tmp"))


def test_raw_dir_failure_cases_record_the_real_exit_status(tmp_path):
    exit_dir = tmp_path / "exit"
    exit_dir.mkdir()
    proc, _r, raw = _raw_run(exit_dir, {"FAKE_EXIT_ON_UNLOAD": "1", "FAKE_EXIT_CODE": "3"})
    assert proc.returncode == 1
    rec = _record(raw)
    assert rec["driver_rc"] == 1
    assert rec["daemon"]["exit_status"] == 3 and not rec["daemon"]["killed_by_driver"]
    assert rec["requests"][0]["done_seen"] and rec["unload"] == "unloaded"
    assert any("status 3" in e for e in rec["errors"])
    assert (raw / "qwen3.6-27b.daemon-output.jsonl").exists()

    eof_dir = tmp_path / "eof"
    eof_dir.mkdir()
    proc, _r, raw = _raw_run(eof_dir, {"FAKE_FAULT": "eof_before_done"})
    assert proc.returncode == 1
    rec = _record(raw)
    (req,) = rec["requests"]
    assert rec["driver_rc"] == 1
    assert not req["commit_ready_seen"] and not req["commit_sent"] and not req["done_seen"]
    assert req["errors"] and req["errors"][0]["source"] == "agentic_gate_driver"
    assert rec["unload"] == "not attempted" and rec["daemon"]["exit_status"] == 0
    assert '"type":"token"' in (raw / "qwen3.6-27b.daemon-output.jsonl").read_text()


def test_without_raw_dir_no_archive_files_are_written(tmp_path):
    proc, _report, _raw = _raw_run(tmp_path, mode_env=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert not [n for n in names if n.endswith((".daemon-output.jsonl", ".driver.json"))]
    assert not (tmp_path / "raw").exists()


def test_raw_dir_must_be_absolute(tmp_path):
    proc, _report = _run_gate(tmp_path, {"HIPFIRE_AGENTIC_GATE_RAW_DIR": "relative/dir"})
    assert proc.returncode == 2 and "must be absolute" in proc.stderr


def test_driver_record_says_ownership_lost(tmp_path):
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    d._reqs = {}
    assert d.start()
    pid = d.proc.pid
    os.kill(pid, 9)
    os.waitpid(pid, 0)  # reaped elsewhere
    d.kill()
    rec_path = tmp_path / "rec.json"
    d.write_record(str(rec_path), "/m", 1)
    rec = json.loads(rec_path.read_text())
    assert rec["daemon"]["exit_status"] == "ownership lost"
    assert rec["driver_rc"] == 1 and rec["unload"] == "not attempted"
    for t in d._threads:
        t.join(timeout=5)
    d._out.close()


def test_peak_rss_is_null_with_a_reason_when_unreadable(tmp_path, monkeypatch):
    def no_proc(path, *a, **k):  # shadows the builtin inside the driver module only
        if str(path).startswith("/proc/"):
            raise OSError("procfs unavailable")
        return open(path, *a, **k)

    cells, _ = _cells(tmp_path)
    monkeypatch.setattr(driver, "open", no_proc, raising=False)
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    rc = d.run("/models/m.mq4", driver.build_requests(cells, "U", ["a", "b", "c"]))
    rec_path = tmp_path / "rec.json"
    d.write_record(str(rec_path), "/models/m.mq4", rc)
    rec = json.loads(rec_path.read_text())
    assert rc == 0 and rec["daemon"]["exit_status"] == 0
    assert rec["daemon"]["peak_rss_kib"] is None
    assert "unreadable" in rec["daemon"]["peak_rss_measured"]


# -- the memory sampler never reads /proc after the leader is reaped -----------

def _delayed_sampler(tmp_path, monkeypatch):
    """A driver whose first /proc read blocks until `gate` is set; `log` orders events."""
    import io
    import threading

    log, gate = [], threading.Event()
    first = [True]

    def slow_open(path, *a, **k):
        if str(path).startswith("/proc/"):
            if first[0]:
                first[0] = False
                log.append("read-start")
                gate.wait(30)
            log.append("read-end")
            return io.StringIO("VmHWM:\t1234 kB\nVmRSS:\t1000 kB\n")
        return open(path, *a, **k)

    real_waitpid = os.waitpid

    def logged_waitpid(*a):
        log.append("reap")
        return real_waitpid(*a)

    monkeypatch.setattr(driver, "open", slow_open, raising=False)
    monkeypatch.setattr(driver.os, "waitpid", logged_waitpid)
    d = driver.Driver(str(FAKE), str(tmp_path / "o.txt"))
    assert d.start()
    deadline = time.monotonic() + 10
    while "read-start" not in log and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log == ["read-start"]
    d._close_stdin()                      # the fake exits on stdin EOF
    assert d._wait_exit(10) == "exited"   # a zombie: still unreaped
    return d, log, gate


def _finish_driver(d):
    for t in d._threads:
        t.join(timeout=5)
    d._out.close()


def test_reap_waits_for_a_delayed_sampler_read_to_finish(tmp_path, monkeypatch):
    import threading

    d, log, gate = _delayed_sampler(tmp_path, monkeypatch)
    threading.Timer(0.5, gate.set).start()   # the sampler finishes while the reap waits
    assert d._reap() == 0
    assert log[-1] == "reap" and log.count("reap") == 1
    assert "read-end" in log[:-1]            # every read ended before the reap began
    value, how = d._peak_record()
    assert value == 1234 and "VmHWM" in how
    assert not d._mem_thread.is_alive()
    _finish_driver(d)


def test_a_sampler_that_cannot_be_stopped_reports_no_value(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "MEM_STOP_WAIT_S", 0.2)
    d, log, gate = _delayed_sampler(tmp_path, monkeypatch)
    assert d._reap() == 0                    # the daemon is reaped; the sampler is still in its read
    assert log == ["read-start", "reap"]
    gate.set()                               # the delayed read now completes AFTER the reap
    d._mem_thread.join(timeout=10)
    assert not d._mem_thread.is_alive() and "read-end" in log
    value, how = d._peak_record()
    assert value is None and "could not be stopped" in how and "not estimated" in how
    rec_path = tmp_path / "rec.json"
    d.write_record(str(rec_path), "/m", 1)
    daemon = json.loads(rec_path.read_text())["daemon"]
    assert daemon["peak_rss_kib"] is None and "could not be stopped" in daemon["peak_rss_measured"]
    _finish_driver(d)
