#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU tests of route_plan_walk.py's bounded gfx1100 profile against a FAKE daemon.

The fake daemon speaks just enough of the daemon's JSON-lines protocol (load, generate, unload) to pin the
walker's request plan, its load params, its run directory and its daemon-kill guarantees. Nothing here is a
recorded walk.

run: python3 scripts/test_route_plan_walk.py
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("route_plan_walk", HERE / "route_plan_walk.py")
walk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(walk)

FAKE = r'''#!/usr/bin/env python3
import json, os, sys, time
pid_file = os.environ.get("FAKE_PID_FILE")
if pid_file:
    open(pid_file, "w").write(str(os.getpid()))
hang_after = int(os.environ.get("FAKE_HANG_AFTER", "-1"))
generates = 0
for line in sys.stdin:
    msg = json.loads(line)
    t = msg["type"]
    if t == "load":
        print(json.dumps({"type": "loaded"}), flush=True)
    elif t == "generate":
        generates += 1
        if generates == hang_after:
            time.sleep(600)
        print(json.dumps({"type": "token", "text": "ok"}), flush=True)
        print(json.dumps({"type": "done", "prompt_tokens": 7}), flush=True)
    elif t == "unload":
        break
'''


def prompt_dir(root):
    d = root / "prompts"
    d.mkdir()
    for name in walk.BOUNDED_DEPTH_PROMPTS:
        (d / name).write_text("depth " + name)
    return d


def run_walker(root, route, extra=(), env=None, timeout=60):
    daemon = root / "fake-daemon"
    daemon.write_text(FAKE)
    daemon.chmod(0o755)
    cmd = [sys.executable, str(HERE / "route_plan_walk.py"), "--run", str(root / f"run-{route}"),
           "--daemon", str(daemon), "--model", "/nonexistent/model.mq4", "--route", route,
           "--profile", "xtx-fn-bounded", "--prompt-dir", str(root / "prompts"), "--max-seq", "262144", *extra]
    return subprocess.run(cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)


class BoundedPlan(unittest.TestCase):
    def test_every_generate_decodes_at_most_eight_tokens_and_depth_prompts_come_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = walk.bounded_plan(prompt_dir(pathlib.Path(tmp)))
        self.assertEqual({m["type"] for m in plan}, {"generate"})
        self.assertTrue(all(m["max_tokens"] <= 8 for m in plan))
        long_prefills = [m for m in plan if m.get("_name", "").startswith("depth_") or m.get("_words", 0) >= walk.BOUNDED_LONG_WORDS]
        self.assertEqual(len(long_prefills), 3 + sum(n >= walk.BOUNDED_LONG_WORDS for n in walk.BOUNDED_WORDS))
        self.assertTrue(all(m["max_tokens"] <= 4 for m in long_prefills), "card-drop rule: long prefills decode <= 4 tokens")
        self.assertEqual([m["_name"] for m in plan[-3:]], list(walk.BOUNDED_DEPTH_PROMPTS))
        self.assertTrue(all("_name" not in m or m["_name"] == "think-short" for m in plan[:-3]))

    def test_plan_covers_sampling_sweep_resume_and_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = walk.bounded_plan(prompt_dir(pathlib.Path(tmp)))
        sampled = [m for m in plan if m.get("prompt") == "Name three rivers in Europe." and "chain_user" not in m]
        self.assertEqual(len(sampled), len(walk.SAMPLING) - 1 + 1)   # + the resume's first turn
        self.assertEqual([m["_words"] for m in plan if m.get("temperature") == 0 and "_words" in m],
                         list(walk.BOUNDED_WORDS))
        self.assertEqual(sum("chain_user" in m for m in plan), 2)
        self.assertEqual(sum(bool(m.get("thinking_enabled")) for m in plan), 2)
        self.assertEqual(len(plan), 1 + (len(walk.SAMPLING) - 1) + 2 + len(walk.BOUNDED_WORDS) + 1 + 3 + 3)

    def test_filler_words_has_exactly_n_words(self):
        for n in (1, 8, 33, 4100):
            self.assertEqual(len(walk.filler_words(n).split()), n)


class FakeDaemonWalk(unittest.TestCase):
    def test_walk_writes_the_receipts_the_builder_reads_and_applies_load_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            r = run_walker(root, "mtp", ["--load-param", 'kv_mode="q8"', "--load-param", "ngram_draft=false",
                                         "--env", "HIPFIRE_KERNEL_CACHE=/warm/cache"])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            run = root / "run-mtp"
            command = json.loads((run / "command.json").read_text())
            self.assertEqual(command["route"], "mtp")
            self.assertEqual(command["load"]["params"], {"max_seq": 262144, "mtp_mode": "on", "kv_mode": "q8",
                                                         "ngram_draft": False})
            self.assertEqual(command["env"]["HIPFIRE_KERNEL_CACHE"], "/warm/cache")
            self.assertEqual(command["env"]["HIPFIRE_ROUTE_PLAN"], "0")
            self.assertEqual(command["env"]["HIPFIRE_MODULE_LOAD_AUDIT"], "1")
            result = json.loads((run / "result.json").read_text())
            self.assertTrue(result["ok"])
            self.assertEqual(result["errors"], 0)
            self.assertEqual(len(result["results"]), command["requests"] + 1)
            markers = [l for l in (run / "stderr.log").read_text().splitlines() if l.startswith("ROUTE_WALK_REQ")]
            self.assertEqual(len(markers), command["requests"])

    def test_ar_route_loads_with_mtp_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            self.assertEqual(run_walker(root, "ar").returncode, 0)
            command = json.loads((root / "run-ar" / "command.json").read_text())
            self.assertEqual(command["load"]["params"]["mtp_mode"], "off")

    def test_hung_daemon_is_killed_at_the_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            pid_file = root / "daemon.pid"
            t0 = time.monotonic()
            r = run_walker(root, "ar", ["--timeout", "3"], {"FAKE_PID_FILE": str(pid_file), "FAKE_HANG_AFTER": "2"})
            self.assertNotEqual(r.returncode, 0)
            self.assertLess(time.monotonic() - t0, 30)
            pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                for _ in range(50):
                    os.kill(pid, 0)
                    time.sleep(0.1)
                os.kill(pid, 0)

    def test_request_timeout_records_a_partial_walk_and_kills_the_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            pid_file = root / "daemon.pid"
            r = run_walker(root, "ar", ["--request-timeout", "3"], {"FAKE_PID_FILE": str(pid_file), "FAKE_HANG_AFTER": "5"})
            self.assertNotEqual(r.returncode, 0)
            result = json.loads((root / "run-ar" / "result.json").read_text())
            self.assertTrue(result["partial"])
            self.assertFalse(result["ok"])
            self.assertEqual(result["errors"], 0)
            self.assertEqual(len(result["results"]), 1 + 4)       # load + the four requests that finished
            self.assertIn("request timeout", result["reason"])
            with self.assertRaises(ProcessLookupError):
                for _ in range(50):
                    os.kill(int(pid_file.read_text()), 0)
                    time.sleep(0.1)
                os.kill(int(pid_file.read_text()), 0)

    def test_whole_walk_timeout_also_records_a_partial_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            run_walker(root, "ar", ["--timeout", "3"], {"FAKE_HANG_AFTER": "3"})
            result = json.loads((root / "run-ar" / "result.json").read_text())
            self.assertTrue(result["partial"])
            self.assertEqual(len(result["results"]), 1 + 2)

    def test_sigterm_of_the_walker_kills_the_daemon(self):
        import signal
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompt_dir(root)
            daemon = root / "fake-daemon"
            daemon.write_text(FAKE)
            daemon.chmod(0o755)
            pid_file = root / "daemon.pid"
            walker = subprocess.Popen(
                [sys.executable, str(HERE / "route_plan_walk.py"), "--run", str(root / "run"), "--daemon", str(daemon),
                 "--model", "/nonexistent", "--route", "ar", "--profile", "xtx-fn-bounded", "--prompt-dir",
                 str(root / "prompts"), "--timeout", "120"],
                env={**os.environ, "FAKE_PID_FILE": str(pid_file), "FAKE_HANG_AFTER": "2"},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                if pid_file.exists() and pid_file.read_text():
                    break
                time.sleep(0.1)
            time.sleep(1.0)
            pid = int(pid_file.read_text())
            os.kill(pid, 0)
            walker.send_signal(signal.SIGTERM)
            self.assertEqual(walker.wait(timeout=20), 143)
            with self.assertRaises(ProcessLookupError):
                for _ in range(50):
                    os.kill(pid, 0)
                    time.sleep(0.1)
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
