# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""CPU test for report.py (spec §13.6). No GPU, no dataset text, no network:
report.py is invoked as a subprocess since it runs top-level on import.

Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class RestrictedSources(unittest.TestCase):
    def test_withheld_sources_never_appear_even_when_frozen_locally(self):
        # A restricted source's rows can exist locally (calibrate.py freeze
        # writes them whether or not the source ships), but report.py must
        # never print them, committed .gz or local .jsonl alike (human
        # decision, Task 6): a withheld dataset is absent from every report.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m"
            probs = out / "probs"
            probs.mkdir(parents=True)
            # Only restricted sources: jev_bench()/jev_cal() (which read the
            # external read-only clones) must never be reached for them, so
            # this test stays hermetic like test_calibrate.py/test_heldout.py.
            row = {"source": "ag-news", "type": "choice", "probs": [0.6, 0.4], "gold": 0}
            (probs / "jevbench_ag-news.jsonl").write_text(json.dumps(row) + "\n")
            row2 = {"source": "hellaswag", "type": "choice", "probs": [0.6, 0.4], "gold": 0}
            (probs / "cal_hellaswag.jsonl").write_text(json.dumps(row2) + "\n")
            subprocess.run([sys.executable, str(HERE / "report.py"), "--out", str(out)],
                            check=True, capture_output=True, text=True, cwd=HERE)
            report = (out / "report.md").read_text()
            # The explanatory prose names both sources; only a table *row* is
            # forbidden.
            self.assertNotIn("| ag-news |", report)
            self.assertNotIn("| hellaswag |", report)
            self.assertNotIn("| hellaswag ", report)


if __name__ == "__main__":
    unittest.main()
