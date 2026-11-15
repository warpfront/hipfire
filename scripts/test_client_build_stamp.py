#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alex Mittell
# hipfire — see LICENSE and NOTICE in the project root.

"""Offline, no-GPU incremental build-stamp regression. Requires cached Cargo deps.

Build the real client and a version-only daemon fixture, commit an unrelated
file, then rebuild using the same target directory. A HEAD-only client stamp
is the negative control. No installed Hipfire binary or daemon is used.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PROBE = r'''
use hipfire_client::{ClientError, Engine};
use std::collections::BTreeMap;
fn main() {
    println!("client={}", env!("HIPFIRE_BUILD_COMMIT"));
    let daemon = std::env::args().nth(1).unwrap();
    match Engine::spawn(daemon, &BTreeMap::new()) {
        Ok(engine) => { drop(engine); println!("probe=accepted"); }
        Err(ClientError::DaemonBuildMismatch { .. }) => {
            println!("probe=mismatch"); std::process::exit(3);
        }
        Err(error) => { eprintln!("unexpected probe error: {error}"); std::process::exit(4); }
    }
}
'''
DAEMON = r'''
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--version") {
        println!("hipfire-daemon 0.0.0 {}", env!("HIPFIRE_BUILD_COMMIT"));
    } else {
        std::io::copy(&mut std::io::stdin(), &mut std::io::sink()).unwrap();
    }
}
'''


class IncrementalBuildStamp(unittest.TestCase):
    def test_unrelated_commit_restamps_client_and_rejects_head_only_control(self):
        with tempfile.TemporaryDirectory(prefix="hipfire-client-stamp-") as tmp:
            for mode in ("shared", "head-only-control"):
                with self.subTest(mode=mode):
                    self.run_fixture(Path(tmp) / mode, mode)

    def run_fixture(self, root, mode):
        env = {k: v for k, v in os.environ.items() if not k.startswith("HIPFIRE_")}
        env.update(HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1", CARGO_BUILD_JOBS="2")

        def run(args, cwd=root, check=True):
            result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=300)
            if check and result.returncode:
                self.fail(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
            return result

        run(["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(root)], cwd=REPO)
        run(["git", "switch", "--quiet", "-C", "build-stamp-fixture"])
        env["CARGO_TARGET_DIR"] = str(root.parent / (mode + "-target"))
        run(["git", "config", "user.name", "Build stamp fixture"])
        run(["git", "config", "user.email", "fixture@example.invalid"])
        client = root / "crates/hipfire-client"
        manifest = client / "Cargo.toml"
        self.assertIn('build = "../hipfire-cli/build.rs"', manifest.read_text())
        if mode == "head-only-control":
            script = (root / "crates/hipfire-cli/build.rs").read_text()
            index_watch = '''    if let Some(index) = git(&repo, &["rev-parse", "--git-path", "index"]) {
        println!("cargo:rerun-if-changed={index}");
    }
'''
            self.assertIn(index_watch, script)
            # Resolve the witness's watched file: a nonexistent relative watch
            # path makes Cargo rerun unconditionally and cannot expose staleness.
            script = script.replace(index_watch, "").replace(
                'println!("cargo:rerun-if-changed={head}");',
                'println!("cargo:rerun-if-changed={}", repo.join(head).display());',
            )
            (client / "build.rs").write_text(script)
            manifest.write_text(manifest.read_text().replace('build = "../hipfire-cli/build.rs"', 'build = "build.rs"'))
        examples = client / "examples"
        examples.mkdir(exist_ok=True)
        (examples / "stamp_probe.rs").write_text(PROBE)
        daemon = root / "crates/stamp-daemon-fixture"
        (daemon / "src").mkdir(parents=True)
        (daemon / "Cargo.toml").write_text('''[package]
name = "stamp-daemon-fixture"
version = "0.0.0"
edition = "2021"
build = "../hipfire-cli/build.rs"
[workspace]
''')
        (daemon / "src/main.rs").write_text(DAEMON)
        run(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(daemon / "Cargo.toml")])
        run(["git", "add", "-f", "crates/hipfire-client", "crates/stamp-daemon-fixture"])
        run(["git", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "fixture A"])

        def head():
            return run(["git", "rev-parse", "HEAD"]).stdout.strip()

        def build():
            run(["cargo", "build", "--offline", "--locked", "-p", "hipfire-client", "--example", "stamp_probe"])
            run(["cargo", "build", "--offline", "--locked", "--manifest-path", str(daemon / "Cargo.toml")])

        target = Path(env["CARGO_TARGET_DIR"]) / "debug"
        suffix = ".exe" if os.name == "nt" else ""
        probe = target / "examples" / ("stamp_probe" + suffix)
        executable = target / ("stamp-daemon-fixture" + suffix)
        a = head()
        build()
        first = run([str(probe), str(executable)])
        self.assertIn("client=" + a, first.stdout)
        self.assertIn("probe=accepted", first.stdout)
        stale_daemon = root.parent / (mode + "-daemon-a" + suffix)
        shutil.copy2(executable, stale_daemon)
        (root / "stamp-unrelated.txt").write_text("No client, daemon or manifest source changed.\n")
        run(["git", "add", "stamp-unrelated.txt"])
        run(["git", "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "fixture B"])
        b = head()
        self.assertNotEqual(a, b)
        changed = run(["git", "diff", "--name-only", a, b]).stdout.splitlines()
        self.assertEqual(changed, ["stamp-unrelated.txt"])
        build()
        version = run([str(executable), "--version"]).stdout.strip()
        self.assertTrue(version.endswith(b), version)
        second = run([str(probe), str(executable)], check=False)
        if mode == "shared":
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("client=" + b, second.stdout)
            self.assertIn("probe=accepted", second.stdout)
            stale = run([str(probe), str(stale_daemon)], check=False)
            self.assertEqual(stale.returncode, 3, stale.stdout + stale.stderr)
            self.assertIn("probe=mismatch", stale.stdout)
        else:
            self.assertEqual(second.returncode, 3, second.stdout + second.stderr)
            self.assertIn("client=" + a, second.stdout)
            self.assertIn("probe=mismatch", second.stdout)
        print(json.dumps({"mode": mode, "commit_a": a, "commit_b": b, "changed": changed,
                          "initial_probe": first.stdout, "incremental_probe": second.stdout,
                          "incremental_exit": second.returncode, "daemon_version": version}), flush=True)


if __name__ == "__main__":
    unittest.main()
