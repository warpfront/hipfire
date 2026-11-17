# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "redline" / "serve_identity_ab.py"

# Fake daemon: records the load request, then answers each generate with the token line,
# commit_ready and done in ONE write so both land in a single read (the lost-message case).
FAKE = r'''
import json, sys, os
out = os.environ["FAKE_OUT"]
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "load":
        open(out, "w").write(json.dumps(m))
        print(json.dumps({"type": "loaded"}), flush=True)
    elif m["type"] == "generate":
        sys.stdout.write(json.dumps({"type": "token", "id": m["id"], "token_id": 7}) + "\n"
                         + json.dumps({"type": "commit_ready", "id": m["id"], "attempt_id": 1}) + "\n")
        sys.stdout.flush()
        break
'''


class ServeIdentityAbTests(unittest.TestCase):
    def run_ab(self, *extra):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake.py"
            fake.write_text(FAKE)
            wrapper = Path(d) / "daemon"
            wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {fake}\n")
            wrapper.chmod(0o755)
            rec = Path(d) / "load.json"
            import os
            env = dict(os.environ, FAKE_OUT=str(rec))
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "ident", "--daemon", str(wrapper), "--model", "m",
                 "--out", d, "--arms", "hip", "--timeout", "5", *extra],
                capture_output=True, text=True, env=env, timeout=60)
            load = json.loads(rec.read_text()) if rec.exists() else None
            return r, load

    def test_help_documents_route_sku(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=30)
        self.assertIn("--route-sku", r.stdout)

    def test_route_sku_forwarded(self):
        _, load = self.run_ab("--route-sku", "mq4rs")
        self.assertEqual(load["params"]["route_sku"], "mq4rs")

    def test_route_sku_absent_by_default(self):
        _, load = self.run_ab()
        self.assertNotIn("route_sku", load["params"])


if __name__ == "__main__":
    unittest.main()
