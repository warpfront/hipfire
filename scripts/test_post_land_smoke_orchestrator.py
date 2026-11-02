#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""scripts/post-land-smoke.sh end to end on CPU: a temp git repo, a fake daemon, a fake fabric-run.sh,
and (remote lanes) a fake ssh/rsync. No GPU, no network, no real fabric."""
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from test_post_land_smoke import FAKE_DAEMON  # noqa: E402  (the daemon protocol the runner tests use)

ORCH = SCRIPTS / "post-land-smoke.sh"
RUNNER = SCRIPTS / "post_land_smoke.py"
PROMPT = "Explain why the sky is blue in three short sentences."


def sh(cmd, cwd=None, env=None, check=True):
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, capture_output=True, text=True)


def git(repo, *args):
    e = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return sh(["git", "-C", str(repo), *args], env=e).stdout.strip()


def make_plan(remote_host=None):
    def load(name, cell, model_var, size, **extra):
        d = {
            "name": name,
            "model": "${%s}" % model_var,
            "model_bytes": size,
            "load_params": {"max_seq": 256},
            "env": {},
            "deadline_s": 60,
            "cells": [{"name": cell, "generate": {"fake_seed": "x"}}],
        }
        d.update(extra)
        return d

    host2 = remote_host or "tlocal"
    return {
        "version": 1,
        "prompt": PROMPT,
        "max_tokens": 8,
        "stderr_fail_patterns": [],
        "loads": [
            load("a-load", "a", "T_MODEL_A", 123),
            load("c-load", "c", "T_MODEL_A", 123, enabled=False, run_if_paths=["^crates/d2/"]),
            load("b-load", "b", "T_MODEL_B", 456),
        ],
        "invariants": [["a", "b"]],
        "lanes": [
            {"name": "L1", "host": "tlocal", "cards": "A,E", "mem_gb": 30.3, "hostmem_gb": 54, "hostmem_lock": True,
             "max_secs": 300, "loads": ["a-load", "c-load"]},
            {"name": "L2", "host": host2, "cards": "B", "mem_gb": 20, "hostmem_gb": 2, "hostmem_lock": False,
             "max_secs": 300, "loads": ["b-load"]},
        ],
    }


FAKE_FABRIC = r"""#!/usr/bin/env bash
# fake fabric-run.sh: records the call, sets the fabric env and runs the command after `--`
calls="$(dirname "$(readlink -f "$0")")/fabric.calls"
printf '%q ' "$@" >>"$calls"; echo >>"$calls"
cards=""
while [ $# -gt 0 ]; do
    case "$1" in
    --) shift; break ;;
    --card | --cards) cards=$2; shift 2 ;;
    --class | --mem-gb | --hostmem-gb | --max-secs | --wait-secs | --priority | --owner) shift 2 ;;
    --hostmem-lock) shift ;;
    *) echo "fake fabric: bad arg $1" >&2; exit 64 ;;
    esac
done
echo "[fabric] 12:00:00 queued as j$$ (P1, owner tester, correct, cards $cards)" >&2
export FABRIC_JOB="$$:1:$cards" FABRIC_CARDS="$cards"
if [ -n "${FAKE_FABRIC_HANG:-}" ]; then exec sleep 600; fi
exec "$@"
"""

FAKE_SSH = """#!/usr/bin/env bash
# fake ssh: ssh HOST /abs/script [args]  ->  run the script locally; refuse anything else
shift
case "$1" in /*) exec "$@" ;; *) echo "fake ssh: not an absolute script path: $1" >&2; exit 64 ;; esac
"""

FAKE_RSYNC = """#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$(dirname "$0")/rsync.calls"
args=()
while [ $# -gt 0 ]; do
    case "$1" in -e) shift 2 ;; -*) shift ;; *) args+=("$1"); shift ;; esac
done
src=${args[0]#*:}; dst=${args[1]#*:}
if [[ $src == */ ]]; then
    mkdir -p "$dst"; cp -a "${src}." "$dst"
    if [ -n "${FAKE_RSYNC_CORRUPT:-}" ] && [ -f "$dst/bin/daemon" ]; then echo x >>"$dst/bin/daemon"; fi
else
    case "$dst" in */) mkdir -p "$dst" ;; esac
    cp -a "$src" "$dst"
fi
"""


def write_exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class Env:
    """One test world: repo (c1,c2,c3), bins per commit, env file, fake tools."""

    def __init__(self, tmp, remote_host=None):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "README").write_text("base\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "c1")
        self.c1 = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "scripts" / "post-land-smoke").mkdir(parents=True)
        (self.repo / "scripts" / "post_land_smoke.py").write_text(RUNNER.read_text())
        (self.repo / "scripts" / "post-land-smoke" / "plan.json").write_text(json.dumps(make_plan(remote_host), indent=1))
        (self.repo / "crates" / "other").mkdir(parents=True)
        (self.repo / "crates" / "other" / "x.rs").write_text("fn x() {}\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "c2: runner and plan")
        self.c2 = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "crates" / "d2").mkdir(parents=True)
        (self.repo / "crates" / "d2" / "cpu_split.rs").write_text("fn d2() {}\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "c3: touches d2")
        self.c3 = git(self.repo, "rev-parse", "HEAD")

        self.bins = tmp / "bins"
        self.out = tmp / "out"
        self.pins = tmp / "state" / "pins.json"
        self.models = tmp / "models"
        self.models.mkdir()
        (self.models / "a.bin").write_bytes(b"a" * 123)
        (self.models / "b.bin").write_bytes(b"b" * 456)
        self.home_models = tmp / "home-models"
        self.home_models.mkdir()
        self.model_env = tmp / "model.env"
        self.model_env.write_text(f"# model vars\nT_MODEL_A={self.models / 'a.bin'}\nT_MODEL_B={self.models / 'b.bin'}\n")
        tools = tmp / "tools"
        tools.mkdir()
        self.fabric = write_exe(tools / "fabric-run.sh", FAKE_FABRIC)
        self.ssh = write_exe(tools / "ssh", FAKE_SSH)
        self.rsync = write_exe(tools / "rsync", FAKE_RSYNC)
        self.calls = tools / "fabric.calls"
        self.remote_root = tmp / "remote-root"
        self.remote_root.mkdir()
        self.env_extra = {}
        self.env_lines = {}

    def make_bins(self, sha, salt=0):
        d = self.bins / sha
        d.mkdir(parents=True)
        daemon = FAKE_DAEMON.replace("base = sum(map(ord, seed))", f"base = sum(map(ord, seed)) + {salt}")
        assert f"+ {salt}" in daemon
        write_exe(d / "daemon", f"#!{sys.executable}\n" + daemon)
        write_exe(d / "hipfire", "#!/bin/sh\necho hipfire\n")
        sums = "".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in ("hipfire", "daemon"))
        (d / "SHA256SUMS").write_text(sums)
        (d / "LANDED.json").write_text('{"status": "LANDED"}\n')
        return d

    def write_env(self, remote_host=None):
        lines = {
            "REPO": str(self.repo),
            "BINS_ROOT": str(self.bins),
            "PINS": str(self.pins),
            "OUT_ROOT": str(self.out),
            "FABRIC_RUN": str(self.fabric),
            "OWNER": "tester",
            "WAIT_SECS": "30",
            "LOCAL_HOST": "tlocal",
            "LOCAL_MODEL_ENV": str(self.model_env),
            "LOCAL_HOME_MODELS": str(self.home_models),
            "HOLD_EXTRA_ARGS": "--watch-cmd sleep 3600",
        }
        if remote_host:
            h = remote_host.upper()
            lines.update({
                f"REMOTE_{h}_SSH": "fakehost",
                f"REMOTE_{h}_ROOT": str(self.remote_root),
                f"REMOTE_{h}_FABRIC_RUN": str(self.fabric),
                f"REMOTE_{h}_MODEL_ENV": str(self.model_env),
                f"REMOTE_{h}_HOME_MODELS": str(self.home_models),
            })
        lines.update(self.env_lines)
        self.env_file = self.tmp / "smoke.env"
        self.env_file.write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in lines.items()))

    def run(self, *args, remote=False, **env):
        e = dict(os.environ)
        e.update(self.env_extra)
        if remote:
            e.update(SSH_BIN=str(self.ssh), RSYNC_BIN=str(self.rsync))
        e.update({k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", str(ORCH), "--env", str(self.env_file), *args], capture_output=True, text=True,
                              env=e, timeout=240)

    def land(self, sha, *args, **kw):
        return self.run("--land", sha, *args, **kw)

    def run_dirs(self, sha):
        return sorted(glob.glob(str(self.out / f"{sha}-*")), key=os.path.getmtime)

    def fabric_calls(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []


@pytest.fixture
def world(tmp_path):
    w = Env(tmp_path)
    w.make_bins(w.c2)
    w.make_bins(w.c3)
    w.write_env()
    return w


def verdict_of(w, sha):
    d = Path(w.run_dirs(sha)[-1])  # the newest run of this land
    return json.loads((d / "VERDICT.json").read_text()), d


def last_line(p):
    return p.stdout.rstrip("\n").splitlines()[-1]


def test_dry_run_prints_lanes_and_commands_and_submits_nothing(world):
    p = world.land(world.c2, "--dry-run")
    assert p.returncode == 0, p.stderr
    assert '"name": "L1"' in p.stdout and '"name": "L2"' in p.stdout
    assert str(world.fabric) in p.stdout and re.search(r"--card A\\?,E", p.stdout) and "--hostmem-lock" in p.stdout
    assert "--priority P1 --owner tester --wait-secs 30" in p.stdout
    assert "post_land_smoke.py hold --plan" in p.stdout and "--lane L1" in p.stdout
    assert "RUN  a-load" in p.stdout and "SKIP c-load" in p.stdout
    assert world.fabric_calls() == []
    assert not world.pins.exists()
    (d,) = world.run_dirs(world.c2)
    assert not list((Path(d) / "lanes").iterdir())  # nothing launched
    assert (Path(d) / "lanes.json").exists() and (Path(d) / "plan.txt").exists()


def test_full_run_pass_pins_and_summary(world):
    p = world.land(world.c2)
    assert p.returncode == 0, p.stdout + p.stderr
    v, d = verdict_of(world, world.c2)
    assert v["overall"] == "PASS"
    assert v["cells"]["a"]["status"] == "NEW" and v["cells"]["b"]["status"] == "NEW"
    assert v["cells"]["c"]["status"] == "SKIPPED"  # no run_if_paths hit in c1..c2
    assert last_line(p) == f"POST-LAND-SMOKE {world.c2} PASS {d}/VERDICT.json"
    summary = (d / "SUMMARY.txt").read_text()
    assert summary.rstrip("\n").splitlines()[-1] == last_line(p)
    assert "fabric_job=j" in summary and "hold=complete" in summary
    pins = json.loads(world.pins.read_text())
    assert set(pins["cells"]) == {"a", "b"}
    assert pins["history"][-1]["sha"] == world.c2
    # land's own runner/plan, hardlinked verified bins, selection inputs
    assert (d / "post_land_smoke.py").read_text() == RUNNER.read_text()
    assert os.stat(d / "bin" / "daemon").st_ino == os.stat(world.bins / world.c2 / "daemon").st_ino
    assert (d / "changed.txt").read_text().split() == sorted(
        ["crates/other/x.rs", "scripts/post-land-smoke/plan.json", "scripts/post_land_smoke.py"])
    # one fabric hold per lane with the plan's fabric args, P1 and the owner
    calls = world.fabric_calls()
    assert len(calls) == 2
    assert any(re.search(r"--card A\\?,E", c) for c in calls)
    assert all("--class correct" in c and "--priority P1" in c and "--owner tester" in c for c in calls)
    for lane in ("L1", "L2"):
        pid, pgid = (d / "lanes" / f"{lane}.pid").read_text().split()
        assert pid == pgid
        hold = json.loads((d / "lanes" / lane / "hold.json").read_text())
        assert hold["status"] == "complete" and hold["fabric_job"]
        assert hold["watcher"]["gone"] is True


def test_run_if_paths_load_runs_when_the_diff_touches_its_path(world):
    assert world.land(world.c2).returncode == 0
    p = world.land(world.c3)  # PREV defaults to the last pinned sha (c2)
    assert p.returncode == 0, p.stdout + p.stderr
    v, d = verdict_of(world, world.c3)
    assert v["overall"] == "PASS"
    assert v["cells"]["c"]["status"] == "NEW" and v["cells"]["a"]["status"] == "MATCH"
    assert (d / "changed.txt").read_text().split() == ["crates/d2/cpu_split.rs"]
    pins = json.loads(world.pins.read_text())
    assert set(pins["cells"]) == {"a", "b", "c"}


def test_changed_ids_give_exit_3_and_leave_pins_untouched(world):
    assert world.land(world.c2).returncode == 0
    before = world.pins.read_bytes()
    # replace the c3 bins with a daemon whose token ids differ
    shutil.rmtree(world.bins / world.c3)
    world.make_bins(world.c3, salt=7)
    p = world.land(world.c3)
    assert p.returncode == 3, p.stdout + p.stderr
    v, d = verdict_of(world, world.c3)
    assert v["overall"] == "CHANGED"
    assert v["cells"]["a"]["status"] == "CHANGED" and v["cells"]["b"]["status"] == "CHANGED"
    assert last_line(p) == f"POST-LAND-SMOKE {world.c3} CHANGED {d}/VERDICT.json"
    assert world.pins.read_bytes() == before
    assert "no pin: CHANGED" in (d / "SUMMARY.txt").read_text()


def test_bins_checksum_mismatch_exits_2_before_any_fabric_call(world):
    with open(world.bins / world.c2 / "daemon", "ab") as f:
        f.write(b"# tampered\n")
    p = world.land(world.c2)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "checksum mismatch" in p.stderr
    assert world.fabric_calls() == []
    assert not world.pins.exists()


def test_missing_bins_dir_exits_2(world):
    shutil.rmtree(world.bins / world.c2)
    p = world.land(world.c2)
    assert p.returncode == 2 and "no binaries" in p.stderr
    assert world.fabric_calls() == []


def test_lane_host_without_config_exits_2_unless_skipped(tmp_path):
    w = Env(tmp_path, remote_host="ghost")
    w.make_bins(w.c2)
    w.write_env()  # no REMOTE_GHOST_* keys
    p = w.land(w.c2)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "ghost" in p.stderr and "--skip-host" in p.stderr
    assert w.fabric_calls() == [] and not w.pins.exists()
    # --skip-host: the lane is not submitted, its cell is FAIL "not run", the others are pinned with a reason
    p = w.land(w.c2, "--skip-host", "ghost")
    assert p.returncode == 1, p.stdout + p.stderr
    v, d = verdict_of(w, w.c2)
    assert v["overall"] == "FAIL" and v["cells"]["b"]["status"] == "FAIL"
    assert "not run" in json.dumps(v["cells"]["b"])
    assert len(w.fabric_calls()) == 1
    pins = json.loads(w.pins.read_text())
    assert "a" in pins["cells"] and "b" not in pins["cells"]
    assert "post-land FAIL at" in json.dumps(pins["history"][-1])


def test_remote_lane_end_to_end(tmp_path):
    w = Env(tmp_path, remote_host="rhost")
    w.make_bins(w.c2)
    w.write_env(remote_host="rhost")
    p = w.land(w.c2, remote=True)
    assert p.returncode == 0, p.stdout + p.stderr
    v, d = verdict_of(w, w.c2)
    assert v["overall"] == "PASS" and v["cells"]["b"]["status"] == "NEW"
    # sent set verified on the remote, results fetched and verified locally
    (rdir,) = [x for x in w.remote_root.iterdir()]
    assert (rdir / "MANIFEST.sha256").exists() and (rdir / "launch-L2.sh").exists()
    assert (rdir / "RESULTS-L2.sha256").read_text().strip()
    assert (d / "lanes" / "L2" / "run.json").exists() and (d / "lanes" / "L2" / "hold.json").exists()
    assert (d / "lanes" / "L2.verify.log").exists()
    assert json.loads((d / "lanes" / "L2" / "hold.json").read_text())["status"] == "complete"
    assert len(w.fabric_calls()) == 2  # local lane L1 + remote lane L2
    launch = (rdir / "launch-L2.sh").read_text()
    assert "start_group" in launch and "hold" in launch and "--lane L2" in launch
    assert str(w.model_env) in launch
    calls = (w.rsync.parent / "rsync.calls").read_text().splitlines()
    (fetch,) = [c for c in calls if "/lanes/L2/" in c]
    for ex in ("--exclude=.hipfire/models", "--exclude=*.hfq", "--exclude=*.mq4*"):
        assert ex in fetch.split(), fetch  # a results pull never brings a model store or model file
    assert "L2" in (d / "SUMMARY.txt").read_text()


def test_remote_corrupted_transfer_fails_verify_before_any_submission(tmp_path):
    w = Env(tmp_path, remote_host="rhost")
    w.make_bins(w.c2)
    w.write_env(remote_host="rhost")
    p = w.land(w.c2, remote=True, FAKE_RSYNC_CORRUPT="1")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "remote sha256 verification failed" in p.stderr
    assert w.fabric_calls() == [] and not w.pins.exists()


def test_watch_runs_a_new_landed_head_once(world):
    origin = world.tmp / "origin.git"
    sh(["git", "clone", "-q", "--bare", str(world.repo), str(origin)])
    git(world.repo, "remote", "add", "origin", str(origin))
    git(world.repo, "push", "-q", "origin", f"{world.c3}:refs/heads/beta")
    (world.bins / world.c3 / "LANDED.json").unlink()  # not landed yet -> nothing runs
    p = world.run("--watch", "1", WATCH_ONCE="1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert world.run_dirs(world.c3) == [] and not (world.out / "last-smoked").exists()
    (world.bins / world.c3 / "LANDED.json").write_text("{}\n")
    p = world.run("--watch", "1", WATCH_ONCE="1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (world.out / "last-smoked").read_text().strip() == world.c3
    assert len(world.run_dirs(world.c3)) == 1
    p = world.run("--watch", "1", WATCH_ONCE="1")  # same head: no second run
    assert len(world.run_dirs(world.c3)) == 1
    assert (world.out / "watch.lock").exists()


def test_usage_errors_exit_2(world):
    assert world.run().returncode == 2  # neither --land nor --watch
    assert world.land("nonexistent-rev").returncode == 2
    bad = subprocess.run(["bash", str(ORCH), "--env", "/nonexistent", "--land", "x"], capture_output=True, text=True)
    assert bad.returncode == 2
    assert world.fabric_calls() == []


def test_sigterm_is_forwarded_only_to_the_recorded_lane_groups(world):
    e = dict(os.environ, FAKE_FABRIC_HANG="1")
    proc = subprocess.Popen(["bash", str(ORCH), "--env", str(world.env_file), "--land", world.c2],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        pidfiles = []
        while time.monotonic() < deadline:
            pidfiles = glob.glob(str(world.out / "*" / "lanes" / "*.pid"))
            if len(pidfiles) == 2:
                break
            time.sleep(0.1)
        assert len(pidfiles) == 2
        pgids = [int(Path(f).read_text().split()[1]) for f in pidfiles]
        assert all(g != proc.pid and g != os.getpgid(proc.pid) for g in pgids)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()  # our own child, launched by this test
    assert proc.returncode == 2, out + err
    assert "INTERRUPTED" in out.splitlines()[-1]
    for g in pgids:
        with pytest.raises(ProcessLookupError):
            os.killpg(g, 0)
    assert not world.pins.exists()


def test_sigterm_during_lane_launch_leaves_no_lane_running(world):
    # Signal as soon as the first lane's group is recorded, while the second lane may still be
    # starting: the trap only knows the groups recorded so far, so the launch loop itself must
    # signal a lane that started after the signal and submit no further lane.
    e = dict(os.environ, FAKE_FABRIC_HANG="1")
    proc = subprocess.Popen(["bash", str(ORCH), "--env", str(world.env_file), "--land", world.c2],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        pidfiles = []
        while time.monotonic() < deadline:
            pidfiles = glob.glob(str(world.out / "*" / "lanes" / "*.pid"))
            if pidfiles:
                break
            time.sleep(0.01)
        assert pidfiles
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()  # our own child, launched by this test
    assert proc.returncode == 2, out + err
    assert "INTERRUPTED" in out.splitlines()[-1]
    for f in glob.glob(str(world.out / "*" / "lanes" / "*.pid")):
        with pytest.raises(ProcessLookupError):
            os.killpg(int(Path(f).read_text().split()[1]), 0)
    assert not world.pins.exists()
