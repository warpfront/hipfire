# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""CPU test for run_jevbench.py's raw-answer guard hardening (spec §13.2,
§13.6). No GPU, no network beyond a local stub server: run_jevbench.py is
invoked as a subprocess since it runs top-level on import (same reason as
test_report.py), with JEVBENCH_DIR pointed at a fake `jevbench` module so no
external clone is needed.

Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAKE_JEVBENCH = '''
import urllib.request

ROOT = None
API_URL = None
MODEL = None
WORKERS = 1
TASKS = {"faketask": None}
EXPERIMENTS = {}


def run_task(name, n, dry_run):
    """Stands in for jevbench.py's real run_task: makes its own
    urllib.request.urlopen call, exactly as jevbench does, so the guard
    patched onto the shared urllib.request module still sees it."""
    with urllib.request.urlopen(API_URL, timeout=60) as r:
        r.read()


def save(name, rows, res, dry_run):
    pass
'''


class _Handler(http.server.BaseHTTPRequestHandler):
    """First response is raw (no calibration echoed); every response after
    that carries a calibrated x-hipfire-timing, simulating a server whose
    decide.calibration.* got configured after this collector's preflight."""
    count = 0

    def _reply(self):
        type(self).count += 1
        body = b'{"answers": {}, "usage": {}}'
        timing = {"prefill_ms": 1.0}
        if type(self).count > 1:
            timing["calibration"] = {"choice": 1.3, "score": 1.0, "noul": 1.0}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Hipfire-Timing", json.dumps(timing))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n:
            self.rfile.read(n)
        self._reply()

    def log_message(self, *_a):
        pass  # keep test output quiet


class FakeJevbenchCallsTheGuardedUrlopen(unittest.TestCase):
    def test_a_calibrated_response_to_jevbenchs_own_urlopen_call_raises(self):
        _Handler.count = 0
        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                bench_dir = Path(d) / "bench"
                bench_dir.mkdir()
                (bench_dir / "jevbench.py").write_text(FAKE_JEVBENCH)
                out_dir = Path(d) / "out"
                proc = subprocess.run(
                    [sys.executable, str(HERE / "run_jevbench.py"),
                     "--port", str(port), "--model", "fake-model",
                     "--out", str(out_dir), "faketask"],
                    env={**os.environ, "JEVBENCH_DIR": str(bench_dir)},
                    capture_output=True, text=True, cwd=HERE)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("calibration", proc.stderr)
        # The first request (this script's own preflight) got a raw
        # response and passed; the second (the fake jevbench module's own
        # run_task call, through the same patched urllib.request.urlopen)
        # got the calibrated one and raised.
        self.assertEqual(_Handler.count, 2)


if __name__ == "__main__":
    unittest.main()
