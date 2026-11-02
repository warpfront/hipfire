#!/usr/bin/env python3
"""CPU-only checks for container harness guidance and command forwarding."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ContainerGateTests(unittest.TestCase):
    def test_no_command_fails_before_runtime(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/container-gate.sh")],
            env={**os.environ, "HIPFIRE_CONTAINER": "/nonexistent-runtime"},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        for harness in ("serve_harness.py", "redline_daemon_harness.py"):
            self.assertIn(f"python3 scripts/{harness}", result.stderr)

    def test_python_command_forwarded_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            runtime.chmod(0o755)
            command = ["python3", "scripts/serve_harness.py", "battery", "--model", "test model"]
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/container-gate.sh"), *command],
                env={**os.environ, "HIPFIRE_CONTAINER": str(runtime),
                     "HIPFIRE_SKIP_BUILD": "1", "HIPFIRE_MODELS_DIR": directory},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[-len(command):], command)

    def test_image_entrypoint_and_home(self):
        containerfile = (ROOT / "Containerfile").read_text()
        self.assertNotIn("HIPFIRE_DIR=", containerfile)
        self.assertEqual(containerfile.count("HIPFIRE_HOME=/root/.hipfire"), 2)
        gate = containerfile.split("FROM builder AS gate-runner", 1)[1]
        entrypoint = next(line for line in gate.splitlines() if line.startswith("ENTRYPOINT "))
        self.assertEqual(json.loads(entrypoint.removeprefix("ENTRYPOINT ")), [])
        cmd = next(line for line in gate.splitlines() if line.startswith("CMD "))
        self.assertEqual(json.loads(cmd.removeprefix("CMD "))[:2], ["/bin/bash", "-c"])
        for harness in ("serve_harness.py", "redline_daemon_harness.py"):
            self.assertTrue((ROOT / "scripts" / harness).is_file())
            self.assertIn(f"python3 scripts/{harness}", cmd)


if __name__ == "__main__":
    unittest.main()
