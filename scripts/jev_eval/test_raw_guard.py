# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""CPU tests for raw_guard.py (spec §13.2, §13.6). No network.
Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import json
import unittest

from raw_guard import check_raw


def fake_headers(d):
    """A stand-in for http.client.HTTPResponse.headers.get."""
    return d.get


class CheckRaw(unittest.TestCase):
    def test_no_timing_header_is_fine(self):
        check_raw(fake_headers({}))  # must not raise

    def test_timing_with_no_calibration_key_is_fine(self):
        headers = {"x-hipfire-timing": json.dumps({"prefill_ms": 12.0, "decode_ms": 0.0})}
        check_raw(fake_headers(headers))  # must not raise

    def test_calibration_entry_raises(self):
        headers = {"x-hipfire-timing": json.dumps(
            {"prefill_ms": 12.0, "calibration": {"choice": 1.3, "score": 1.0, "noul": 1.0}})}
        with self.assertRaises(RuntimeError) as e:
            check_raw(fake_headers(headers))
        self.assertIn("calibration", str(e.exception))

    def test_explicit_identity_calibration_still_raises(self):
        # note_calibration only ever echoes a non-identity calibration
        # (hipfire-generate decide.rs), so a "calibration" key's mere
        # presence is enough: no need to special-case {1,1,1}.
        headers = {"x-hipfire-timing": json.dumps({"calibration": {"choice": 1.0, "score": 1.0, "noul": 1.0}})}
        with self.assertRaises(RuntimeError):
            check_raw(fake_headers(headers))


if __name__ == "__main__":
    unittest.main()
