#!/usr/bin/env python3
"""CPU-only regression tests: python3 scripts/test_tiered_kv_vram_guard.py."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("tiered_kv_vram_guard.py")
SPEC = importlib.util.spec_from_file_location("tiered_kv_vram_guard", SCRIPT)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
BDF = "0000:03:00.0"


@unittest.skipUnless(sys.platform == "linux", "Linux process groups and fake sysfs")
class VramGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.drm = self.root / "drm"
        self.device = self.root / BDF
        self.device.mkdir()
        self.drm.mkdir()
        card = self.drm / "card7"
        card.mkdir()
        (card / "device").symlink_to(self.device, target_is_directory=True)
        # render nodes must not accidentally be interpreted as a second card.
        (self.drm / "renderD128").symlink_to(card, target_is_directory=True)
        self.counter = self.device / "mem_info_vram_used"
        self.counter.write_text("100\n")
        self.out = self.root / "evidence.json"

    def argv(self, code="pass", extra=(), command=None):
        return [sys.executable, str(SCRIPT), "--pci-bdf", BDF,
                "--out", str(self.out), "--_sysfs-root", str(self.drm),
                "--limit-bytes", "1000", "--safety-margin-bytes", "100",
                "--terminate-timeout-seconds", "0.1", *extra, "--",
                *(command or [sys.executable, "-c", code])]

    def invoke(self, code="pass", extra=(), command=None):
        result = subprocess.run(self.argv(code, extra, command),
                                capture_output=True, text=True, timeout=8)
        return result, json.loads(self.out.read_text())

    def test_normal_command_and_raw_samples(self):
        result, report = self.invoke("import time; print('raw stdout'); time.sleep(.12)")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "raw stdout\n")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["baseline_bytes"], 100)
        self.assertEqual(report["peak_bytes"], 100)
        self.assertEqual(report["guard_limit_bytes"], 900)
        self.assertGreaterEqual(len(report["samples"]), 3)
        self.assertTrue(all(x["raw"] == "100\n" for x in report["samples"]))
        # The unreaped leader pins the group through cleanup; signalling the
        # now-zombie leader is harmless and must not rewrite its successful code.
        self.assertEqual(report["child_exit_code"], 0)

    def test_peak_is_recorded_not_just_last_sample(self):
        code = (f"from pathlib import Path; import time; p=Path({str(self.counter)!r}); "
                "p.write_text('700\\n'); time.sleep(.2); "
                "p.write_text('200\\n'); time.sleep(.15)")
        result, report = self.invoke(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["peak_bytes"], 700)
        self.assertEqual(report["peak_delta_bytes"], 600)
        self.assertEqual(report["samples"][-1]["vram_used_bytes"], 200)
        self.assertIn("700\n", [x["raw"] for x in report["samples"]])

    def test_limit_terminates_only_owned_group(self):
        # An unrelated live process must survive, even though it runs the same
        # executable as the guarded command. No executable-name based killing.
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                 start_new_session=True)
        self.addCleanup(self.stop_process, other)
        code = (f"from pathlib import Path; import time; "
                f"Path({str(self.counter)!r}).write_text('901\\n'); time.sleep(10)")
        result, report = self.invoke(code)
        self.assertEqual(result.returncode, GUARD.EXIT_GUARD, result.stderr)
        self.assertEqual(report["status"], "limit_exceeded")
        self.assertEqual(report["peak_bytes"], 901)
        self.assertEqual(report["child_exit_code"], -signal.SIGTERM)
        self.assertIn("SIGTERM", report["termination_signals"])
        self.assertIsNone(other.poll())

    @staticmethod
    def stop_process(process):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)

    def test_baseline_over_guard_refuses_launch(self):
        marker = self.root / "must-not-launch"
        self.counter.write_text("901\n")
        result, report = self.invoke(f"from pathlib import Path; Path({str(marker)!r}).touch()")
        self.assertEqual(result.returncode, GUARD.EXIT_GUARD)
        self.assertEqual(report["status"], "limit_exceeded_before_launch")
        self.assertIsNone(report["child_pid"])
        self.assertFalse(marker.exists())

    def test_equal_to_guard_is_allowed(self):
        self.counter.write_text("900\n")
        result, report = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["peak_bytes"], 900)

    def test_monitor_read_failure_is_not_success(self):
        code = (f"from pathlib import Path; import time; "
                f"Path({str(self.counter)!r}).unlink(); time.sleep(10)")
        result, report = self.invoke(code)
        self.assertEqual(result.returncode, GUARD.EXIT_MONITOR)
        self.assertEqual(report["status"], "monitor_error")
        self.assertIn("error", report)
        self.assertEqual(report["child_exit_code"], -signal.SIGTERM)

    def test_invalid_counter_refuses_launch(self):
        for value in ("not a number\n", "-1\n", "12.5\n", ""):
            with self.subTest(value=value):
                self.counter.write_text(value)
                result, report = self.invoke()
                self.assertEqual(result.returncode, GUARD.EXIT_MONITOR)
                self.assertEqual(report["status"], "monitor_error")
                self.assertIsNone(report["child_pid"])
                self.out.unlink()

    def test_child_failure_and_signal_exit_are_preserved(self):
        for code, expected, raw in (("raise SystemExit(23)", 23, 23),
                                    ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
                                     137, -signal.SIGKILL)):
            with self.subTest(expected=expected):
                result, report = self.invoke(code)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(report["child_exit_code"], raw)
                self.assertEqual(report["status"], "completed")
                self.out.unlink()

    def test_launch_failure_has_evidence(self):
        result, report = self.invoke(command=[str(self.root / "missing-command")])
        self.assertEqual(result.returncode, 127)
        self.assertEqual(report["status"], "launch_error")
        self.assertIsNone(report["child_pid"])
        self.assertIsNone(report["child_exit_code"])

    def test_existing_evidence_is_never_overwritten(self):
        original = '{"external": "untouched"}\n'
        self.out.write_text(original)
        marker = self.root / "must-not-launch"
        result, _ = self.invoke(f"from pathlib import Path; Path({str(marker)!r}).touch()")
        self.assertEqual(result.returncode, GUARD.EXIT_MONITOR)
        self.assertEqual(self.out.read_text(), original)
        self.assertFalse(marker.exists())

    def test_evidence_symlink_is_never_followed(self):
        external = self.root / "external.json"
        original = '{"external": "untouched"}\n'
        external.write_text(original)
        self.out.symlink_to(external)
        result, _ = self.invoke()
        self.assertEqual(result.returncode, GUARD.EXIT_MONITOR)
        self.assertEqual(external.read_text(), original)
        self.assertTrue(self.out.is_symlink())

    def test_wrapper_sigterm_records_interruption_and_cleans_child(self):
        marker = self.root / "ready"
        code = ("import signal, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(marker)!r}).touch(); time.sleep(10)")
        wrapper = subprocess.Popen(self.argv(code), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop_process, wrapper)
        deadline = time.monotonic() + 3
        while not marker.exists():
            if wrapper.poll() is not None or time.monotonic() > deadline:
                self.fail("guarded command did not start")
            time.sleep(.01)
        wrapper.send_signal(signal.SIGTERM)
        _, stderr = wrapper.communicate(timeout=3)
        report = json.loads(self.out.read_text())
        self.assertEqual(wrapper.returncode, 143, stderr)
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["child_exit_code"], -signal.SIGKILL)
        self.assertEqual(report["termination_signals"], ["SIGTERM", "SIGKILL"])

    def test_term_resistant_descendant_is_killed_after_leader_exits(self):
        pidfile = self.root / "descendant.pid"
        child = ("import os, signal, time; from pathlib import Path; "
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(10)")
        # Leader exits successfully; cleanup must still find its live group.
        code = (f"import subprocess, sys, time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                f"p=Path({str(pidfile)!r}); "
                "\nwhile not p.exists(): time.sleep(.005)\n")
        result, report = self.invoke(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["termination_signals"], ["SIGTERM", "SIGKILL"])
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            status = Path(f"/proc/{pid}/status")
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(.01)
        else:
            self.fail("owned descendant survived bounded cleanup")

    def test_already_reaped_leader_never_calls_killpg(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"],
                                   start_new_session=True)
        process.wait(timeout=2)
        with mock.patch.object(GUARD.os, "killpg") as killpg:
            with self.assertRaisesRegex(ChildProcessError, "ownership lost"):
                GUARD.terminate_owned_group(process, .01)
            killpg.assert_not_called()

    def test_external_reaper_never_calls_killpg(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"],
                                   start_new_session=True)
        _, status = os.waitpid(process.pid, 0)  # Popen does not know it was reaped.
        try:
            with mock.patch.object(GUARD.os, "killpg") as killpg:
                with self.assertRaisesRegex(ChildProcessError, "ownership lost"):
                    GUARD.terminate_owned_group(process, .01)
                killpg.assert_not_called()
        finally:
            process.returncode = os.waitstatus_to_exitcode(status)

    def test_ownership_loss_after_term_prevents_any_later_group_signal(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                   start_new_session=True)
        real_killpg = os.killpg
        reaped_status = []

        def reap_after_term(pgid, sig):
            real_killpg(pgid, sig)
            if sig == signal.SIGTERM:
                reaped_status.append(os.waitpid(process.pid, 0)[1])

        try:
            with mock.patch.object(GUARD.os, "killpg", side_effect=reap_after_term) as killpg:
                with self.assertRaisesRegex(ChildProcessError, "ownership lost"):
                    GUARD.terminate_owned_group(process, .01)
                self.assertEqual(killpg.call_args_list,
                                 [mock.call(process.pid, 0),
                                  mock.call(process.pid, signal.SIGTERM)])
        finally:
            if reaped_status:
                process.returncode = os.waitstatus_to_exitcode(reaped_status[0])
            else:
                self.stop_process(process)

    def test_ownership_loss_reports_failure_without_group_signals(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"],
                                   start_new_session=True)
        process.wait(timeout=2)
        args = GUARD.parse_args(self.argv()[2:])
        report = {}
        with mock.patch.object(GUARD.subprocess, "Popen", return_value=process), \
                mock.patch.object(GUARD.os, "killpg") as killpg:
            result = GUARD.run_guard(args, report)
            killpg.assert_not_called()
        self.assertEqual(result, GUARD.EXIT_MONITOR)
        self.assertEqual(report["status"], "cleanup_error")
        self.assertIn("ownership lost", report["cleanup_error"])

    def test_wrong_pci_identity_fails_closed(self):
        result, report = self.invoke(extra=["--pci-bdf", "0000:04:00.0"])
        self.assertEqual(result.returncode, GUARD.EXIT_MONITOR)
        self.assertIsNone(report["child_pid"])
        self.assertIn("found 0", report["error"])

    def test_defaults_and_bdf_normalization(self):
        args = GUARD.parse_args(["--pci-bdf", "03:00.0", "--out", "unused", "--", "true"])
        self.assertEqual(args.pci_bdf, BDF)
        self.assertEqual(args.limit_bytes, 24 * GUARD.GIB)
        self.assertEqual(args.safety_margin_bytes, 512 * GUARD.MIB)
        self.assertEqual(GUARD.SAMPLE_INTERVAL_SECONDS, .05)


if __name__ == "__main__":
    unittest.main()
