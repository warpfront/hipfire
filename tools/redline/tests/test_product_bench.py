#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.redline.product_bench import (
    CERTIFIED_PM4_POLICY,
    analyze_stationarity,
    backend_config_value,
    require_retained_pm4,
    run_pm4_preflight,
    validate_route_proof,
)


DEFAULTS = {
    "window": 10,
    "min_runs": 10,
    "confirmation_runs": 10,
    "max_slope_pct": 0.05,
    "max_spread_pct": 1.0,
    "max_median_drift_pct": 0.5,
}


class StationarityTests(unittest.TestCase):
    def test_stable_signal_requires_confirmation(self):
        values = [100.0 + (0.02 if i % 2 else -0.02) for i in range(20)]
        before = analyze_stationarity(values[:19], **DEFAULTS)
        after = analyze_stationarity(values, **DEFAULTS)
        self.assertFalse(before["stationary"])
        self.assertTrue(after["stationary"])
        self.assertEqual(after["candidate"]["at_row"], 10)
        self.assertEqual(after["confirmed_at_row"], 20)

    def test_false_plateau_in_tg128_trace_is_rejected(self):
        # Real gfx1100 retained-PM4 settling trace: an apparent plateau near
        # 210.8 tok/s resumes climbing before settling near 213.4 tok/s.
        values = [
            200.913, 202.040, 202.714, 202.570, 203.417,
            204.519, 204.621, 208.824, 210.386, 210.516,
            210.603, 210.746, 210.597, 210.820, 211.038,
            210.993, 210.876, 211.524, 212.203, 212.174,
            212.563, 212.876, 213.165, 213.506, 213.551,
            213.550, 213.527, 213.431, 213.536, 213.399,
        ]
        values.extend(213.45 + (0.03 if i % 2 else -0.03) for i in range(20))
        result = analyze_stationarity(values, **DEFAULTS)
        self.assertTrue(result["stationary"])
        self.assertGreaterEqual(result["candidate"]["at_row"], 30)
        self.assertGreaterEqual(result["confirmed_at_row"], 40)
        self.assertGreater(len(result["rejections"]), 0)
        self.assertGreater(result["confirmed_window"]["median"], 213.0)

    def test_continuous_ramp_never_passes(self):
        values = [100.0 + i * 0.2 for i in range(60)]
        result = analyze_stationarity(values, **DEFAULTS)
        self.assertFalse(result["stationary"])


class BackendConfigTests(unittest.TestCase):
    def test_auto_report_arm_explicitly_opts_into_redline(self):
        self.assertEqual(backend_config_value("auto"), "redline")
        self.assertEqual(backend_config_value("hip"), "hip")

    def test_product_policy_reemits_dynamic_gfx12_registers(self):
        self.assertEqual(CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_STATEFUL"], "static")
        self.assertEqual(CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_QUEUES"], "1")
        self.assertEqual(
            CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY"],
            "required-only",
        )


class RouteProofTests(unittest.TestCase):
    @staticmethod
    def route_row(iterations, delta, retained, transport="pm4"):
        return {
            "context_tokens": 129,
            "iterations": iterations,
            "redline_route": {
                "requested_backend": "auto",
                "transport": transport,
                "state": "ready",
                "fallback_reason": None,
                "execution_mode": "plain_ar",
                "prepared": {
                    "dispatches": 604,
                    "packets": 1,
                    "queue_id": 7,
                    "command_dwords": 16832 if transport == "pm4" else None,
                },
                "sequence": {
                    "launches": 604,
                    "unique_kernels": 22,
                    "hash": "440bb2b5df220117",
                },
                "observed": {
                    "count_delta": delta,
                    "first_position": 129 if delta else None,
                    "last_position": 129 + delta - 1 if delta else None,
                },
                "retained_replay_observed": retained,
            },
        }

    def test_timed_rows_require_one_replay_per_iteration(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        timed = validate_route_proof(
            rows, "auto", "pm4", require_complete_replay=True
        )
        self.assertTrue(timed["valid"], timed["errors"])

    def test_timed_rows_reject_partial_replay(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=0, retained=False),
        ]
        timed = validate_route_proof(
            rows, "auto", "pm4", require_complete_replay=True
        )
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("timed row observed no retained replay" in error for error in timed["errors"])
        )

    def test_timed_rows_reject_cumulative_position_evidence(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["observed"]["first_position"] = 128
        timed = validate_route_proof(
            [row], "auto", "pm4", require_complete_replay=True
        )
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("first replay position 128 != 129" in error for error in timed["errors"])
        )

    def test_first_auto_warmup_fast_fails_on_pm4_fallback(self):
        row = self.route_row(iterations=32, delta=0, retained=False)
        row["redline_route"]["state"] = "fallback"
        row["redline_route"]["fallback_reason"] = "kernel requires scratch"
        with self.assertRaisesRegex(RuntimeError, "kernel requires scratch"):
            require_retained_pm4(row)

    def test_first_auto_warmup_accepts_retained_pm4(self):
        row = self.route_row(iterations=32, delta=32, retained=True)
        require_retained_pm4(row)



class Pm4PreflightTests(unittest.TestCase):
    @staticmethod
    def args(work_dir):
        return SimpleNamespace(
            daemon="/unused/daemon",
            model="/unused/model",
            work_dir=work_dir,
            timeout=30.0,
            kv_mode="q8",
            max_seq=2048,
            context=128,
        )

    def test_preflight_smoke_tests_pm4_before_any_warmup(self):
        row = RouteProofTests.route_row(iterations=2, delta=2, retained=True)
        fake = unittest.mock.Mock()
        fake.request.side_effect = [{"type": "loaded"}, row]
        with tempfile.TemporaryDirectory() as work_dir:
            with patch("tools.redline.product_bench.Daemon", return_value=fake) as daemon:
                result = run_pm4_preflight(self.args(work_dir))

        daemon.assert_called_once_with(
            Path("/unused/daemon"),
            "auto",
            "pm4",
            Path(work_dir) / "product-pm4-preflight.log",
            30.0,
            "q8",
            0.0,
        )
        self.assertEqual(fake.request.call_args_list[0].args[0]["type"], "load")
        smoke = fake.request.call_args_list[1].args[0]
        self.assertEqual(smoke["type"], "bench_decode")
        self.assertEqual(smoke["iterations"], 2)
        self.assertTrue(smoke["redline_product_route"])
        self.assertEqual(result["redline_route"]["state"], "ready")
        fake.close.assert_called_once_with()

    def test_preflight_fast_fails_with_fallback_reason(self):
        row = RouteProofTests.route_row(iterations=2, delta=0, retained=False)
        row["redline_route"]["state"] = "fallback"
        row["redline_route"]["fallback_reason"] = "kernel requires scratch"
        fake = unittest.mock.Mock()
        fake.request.side_effect = [{"type": "loaded"}, row]
        with tempfile.TemporaryDirectory() as work_dir:
            with patch("tools.redline.product_bench.Daemon", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "kernel requires scratch"):
                    run_pm4_preflight(self.args(work_dir))
        fake.close.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
