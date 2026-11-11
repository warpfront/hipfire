import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import redline_daemon_harness as harness
from scripts.redline_daemon_harness import DFLASH_EXACT_FIELDS, dflash_shadow_failures, qwen4_recipe_failures


def valid_shadow():
    arm = {field: True for field in DFLASH_EXACT_FIELDS}
    arm["state"] = {"max_abs": 0.0, "max_rel": 0.0}
    return {
        "route": {
            "phase": "ready",
            "counters": {
                "replays": 1,
                "replay_failures": 0,
                "poison_count": 0,
                "contract_failures": 0,
                "prepare_failures": 0,
            },
        },
        "capture": {"aql_equals_unique_kernels": True},
        "prepared_identity": {
            "dispatch_equals_launches": True,
            "queue_count": 1,
            "phase_count": 1,
        },
        "parity": {
            "q8_error_feedback_enabled": True,
            "q8_byte_parity_invalid": False,
            "windows": [
                {
                    "position": 128,
                    "recorded_hip": copy.deepcopy(arm),
                    "pm4": copy.deepcopy(arm),
                }
            ],
        },
    }


def valid_qwen4_shadow():
    shadow = valid_shadow()
    arm = {field: True for field in (
        "tokens_equal", "argmax_equal", "position_equal", "lengths_equal",
        "state_bit_exact", "qsa_full_equal", "gdn_equal", "ple_equal",
    )}
    # Packed fp8 or Q8 storage can look non-finite when the diagnostic
    # formatter treats four bytes as f32. Raw equality remains mandatory.
    arm["qsa_full"] = {"max_abs": None, "max_rel": None, "bytes_equal": True}
    shadow.update(bit_exact=True, blob_bit_exact=True, failure_behavior={
        "route_poisoned": True, "error": "exceeds prepared max_position",
        "fallback_reason": "bound exceeded", "recovery_uses_hip": True,
        "recovered_bit_exact_against_clean_hip": True,
    })
    shadow["parity"]["windows"][0].update(
        recorded_hip=copy.deepcopy(arm), pm4=copy.deepcopy(arm))
    return shadow


class Qwen4RecipeTests(unittest.TestCase):
    def test_drafter_quant_does_not_admit_or_reject_trunk(self):
        trunk = "model.language_model.layers.0.self_attn.q_proj.weight"
        entries = [(trunk, 47, (128, 128)), ("mtp.layers.0.self_attn.q_proj.weight", 3, (128, 128))]
        with patch("scripts.redline_daemon_harness.read_qwen4_index", return_value=entries):
            self.assertEqual(qwen4_recipe_failures("model"), [])
            entries[0] = (trunk, 3, (128, 128))
            self.assertIn("trunk projections carry quant types [3]", qwen4_recipe_failures("model")[0])


class Qwen4KvModeTests(unittest.TestCase):
    def test_fp8_reaches_the_load_request_without_starting_a_real_daemon(self):
        class LoadObserved(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.hfq"
            binary = Path(directory) / "daemon"
            model.touch()
            binary.touch()
            for explicit, expected in ((None, "bf16"), ("bf16", "bf16"), ("fp8", "fp8")):
                with self.subTest(kv_mode=explicit):
                    argv = ["redline_daemon_harness.py", "--qwen4", "--model", str(model),
                            "--daemon", str(binary), "--state-quant", "q8"]
                    if explicit is not None:
                        argv.extend(("--kv-mode", explicit))
                    with patch("sys.argv", argv), \
                         patch.object(harness, "qwen4_recipe_failures", return_value=[]), \
                         patch.object(harness, "Daemon") as daemon:
                        # Stop at the load boundary: no model read, HIP init or
                        # child process is permitted in this CPU regression.
                        daemon.return_value.request.side_effect = LoadObserved
                        with self.assertRaises(LoadObserved):
                            harness.main()
                        self.assertEqual(daemon.call_args.args[3], expected)
                        request = daemon.return_value.request.call_args.args[0]
                        self.assertEqual(request["type"], "load")
                        self.assertEqual(request["params"]["kv_mode"], expected)
                        self.assertEqual(request["params"]["state_quant"], "q8")
                        self.assertEqual(request["params"]["mtp_mode"], "off")
                        daemon.return_value.close.assert_called_once()

    def test_other_qwen4_layouts_are_rejected_before_daemon_creation(self):
        for mode in ("q8", "fwht2", "fwht3", "fwht4"):
            with self.subTest(kv_mode=mode):
                argv = ["redline_daemon_harness.py", "--qwen4", "--model", "unused",
                        "--kv-mode", mode]
                with patch("sys.argv", argv), patch.object(harness, "Daemon") as daemon:
                    with self.assertRaisesRegex(SystemExit, "requires --kv-mode bf16 or fp8"):
                        harness.main()
                    daemon.assert_not_called()

    def test_packed_state_mismatch_cannot_hide_behind_null_float_diagnostics(self):
        shadow = valid_qwen4_shadow()
        self.assertEqual(harness.qwen4_shadow_failures(shadow), [])
        for executor in ("recorded_hip", "pm4"):
            for field in ("qsa_full_equal", "gdn_equal", "state_bit_exact"):
                with self.subTest(executor=executor, field=field):
                    bad = copy.deepcopy(shadow)
                    bad["parity"]["windows"][0][executor][field] = False
                    self.assertTrue(any(field in why for why in harness.qwen4_shadow_failures(bad)))
                    del bad["parity"]["windows"][0][executor][field]
                    self.assertTrue(any(field in why for why in harness.qwen4_shadow_failures(bad)))

    def test_recovery_error_rejects_report_without_losing_original_cause_or_windows(self):
        shadow = valid_qwen4_shadow()
        self.assertEqual(harness.qwen4_shadow_failures(shadow), [])
        original = "position 512 exceeds prepared max_position 511"
        recovery = "production-recovery prime: expert cache token is unretired"
        shadow["failure_behavior"].update(
            error=original, recovery_error=recovery,
            recovery_uses_hip=False, recovered_bit_exact_against_clean_hip=False)
        failures = harness.qwen4_shadow_failures(shadow)
        self.assertTrue(any(recovery in why for why in failures))
        self.assertIn("recovery forward is not on HIP", failures)
        self.assertIn("recovery forward is not bit-exact against clean HIP", failures)
        self.assertEqual(shadow["failure_behavior"]["error"], original)
        self.assertTrue(shadow["parity"]["windows"][0]["pm4"]["state_bit_exact"])


class DflashShadowVerdictTests(unittest.TestCase):
    def test_production_q8_ef_report_passes(self):
        self.assertEqual(dflash_shadow_failures(valid_shadow()), [])

    def test_missing_or_divergent_ef_snapshot_fails(self):
        for field in ("dn_ef_after_forward_equal", "dn_ef_after_rollback_equal"):
            with self.subTest(field=field, case="missing"):
                shadow = valid_shadow()
                del shadow["parity"]["windows"][0]["pm4"][field]
                self.assertTrue(any(field in failure for failure in dflash_shadow_failures(shadow)))
            with self.subTest(field=field, case="divergent"):
                shadow = valid_shadow()
                shadow["parity"]["windows"][0]["pm4"][field] = False
                self.assertTrue(any(field in failure for failure in dflash_shadow_failures(shadow)))

    def test_legacy_or_unreported_q8_parity_is_rejected(self):
        shadow = valid_shadow()
        del shadow["parity"]["q8_byte_parity_invalid"]
        self.assertIn(
            "parity.q8_byte_parity_invalid missing",
            dflash_shadow_failures(shadow),
        )

        shadow = valid_shadow()
        shadow["parity"]["q8_byte_parity_invalid"] = True
        self.assertTrue(
            any("production Q8 EF evidence required" in failure for failure in dflash_shadow_failures(shadow))
        )


if __name__ == "__main__":
    unittest.main()
