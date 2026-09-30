# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""CPU tests for calibrate.py (spec §13.7).
Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import gzip
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import calibrate as cb


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def synthetic_rows(t_true, n=4000, k=4, seed=0, source="syn"):
    """Served probabilities softmax(z); gold labels drawn from softmax(z / t_true)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = [rng.gauss(0, 2) for _ in range(k)]
        y = rng.choices(range(k), weights=softmax(z, t_true))[0]
        rows.append({"source": source, "type": "choice", "probs": softmax(z), "gold": y})
    return rows


class ApplyT(unittest.TestCase):
    def test_identity(self):
        p = [0.7, 0.2, 0.1]
        for a, b in zip(cb.apply_t(p, 1.0), p):
            self.assertAlmostEqual(a, b, places=15)

    def test_equals_softmax_of_scaled_logits(self):
        z = [2.0, 0.0, -1.0, 4.5]
        for t in (0.5, 1.7, 4.0):
            for a, b in zip(cb.apply_t(softmax(z), t), softmax(z, t)):
                self.assertAlmostEqual(a, b, places=12)


class Fit(unittest.TestCase):
    def test_recovers_a_known_temperature(self):
        for t_true in (2.0, 0.5):
            t = cb.fit_temperature(synthetic_rows(t_true))
            self.assertLess(abs(t / t_true - 1), 0.08, (t_true, t))

    def test_nll_is_minimal_at_the_fit(self):
        rows = synthetic_rows(1.6, n=1500, seed=3)
        t = cb.fit_temperature(rows)
        best = cb.nll(rows, t)
        self.assertLessEqual(best, cb.nll(rows, t * 1.05))
        self.assertLessEqual(best, cb.nll(rows, t / 1.05))
        self.assertLess(best, cb.nll(rows, 1.0))

    def test_fit_at_a_bound_raises(self):
        # Always right at p = 0.6: sharpening always helps, the optimum is T -> 0.
        row = {"source": "s", "type": "noul", "probs": [0.6, 0.4], "gold": 0}
        with self.assertRaises(ValueError):
            cb.fit_temperature([row] * 50)

    def test_ship_rule(self):
        hot = cb.fit_types(synthetic_rows(2.0, n=1500))["choice"]
        self.assertEqual(hot["n"], cb.PER_SOURCE_CAP)
        self.assertTrue(hot["ship"], hot)
        calm = cb.fit_types(synthetic_rows(1.0, n=1500))["choice"]
        self.assertFalse(calm["ship"], calm)

    def test_pool_caps_each_source(self):
        rows = synthetic_rows(1.0, n=400, source="a") + synthetic_rows(1.0, n=50, source="b")
        self.assertEqual({s: len(v) for s, v in cb.pool(rows, "choice").items()}, {"a": 300, "b": 50})
        self.assertEqual(cb.pool(rows, "noul"), {})


class ShipDecision(unittest.TestCase):
    """Spec §13.4 ship rule: NLL gain >= 1% relative AND the 95% CI for T
    excludes 1 — either alone is not enough."""

    def test_ci_excludes_one_but_gain_under_one_percent_is_not_shipped(self):
        # A large, low-variance pool can pin T away from 1 (tight CI) while
        # the fit still buys under 1% NLL.
        self.assertFalse(cb.ship_decision(nll_raw=1.000, nll_cal=0.998, lo=1.01, hi=1.02))

    def test_gain_and_ci_both_clear_the_bar_ships(self):
        self.assertTrue(cb.ship_decision(nll_raw=1.000, nll_cal=0.980, lo=1.05, hi=1.10))

    def test_big_gain_but_ci_includes_one_is_not_shipped(self):
        self.assertFalse(cb.ship_decision(nll_raw=1.000, nll_cal=0.900, lo=0.95, hi=1.05))


class BootstrapCi(unittest.TestCase):
    def test_nearest_rank_indices_for_n_100(self):
        # Regression: the upper bound must be the 97.5th-percentile
        # nearest-rank index (ts[97] of 100 sorted values), not ts[96].
        self.assertEqual(cb._nearest_rank(100, 0.025), 2)
        self.assertEqual(cb._nearest_rank(100, 0.975), 97)

    def test_bootstrap_ci_upper_bound_is_ts_97_of_100(self):
        rows = synthetic_rows(1.6, n=1500, seed=5)
        data = cb._prep(rows)
        rng = random.Random(0)
        rng_ts = sorted(cb._fit(rng.choices(data, k=len(data)), strict=False) for _ in range(cb.BOOTSTRAP))
        lo, hi = cb.bootstrap_ci(data, n=cb.BOOTSTRAP, seed=0)
        self.assertEqual((lo, hi), (rng_ts[2], rng_ts[97]))
        self.assertNotEqual(hi, rng_ts[96], "upper bound regressed to the old off-by-one index")


class Metrics(unittest.TestCase):
    def test_ece_matches_jevbench_binning(self):
        pairs = [(0.95, 1), (0.95, 0), (0.55, 1), (0.0, 0)]
        self.assertAlmostEqual(cb.ece(pairs), 0.45 * 0.5 + 0.45 * 0.25, places=12)

    def test_summarize_keeps_accuracy_and_moves_confidence(self):
        rows = [{"source": "s", "type": "score", "probs": [0.1, 0.2, 0.7], "gold": 2},
                {"source": "s", "type": "score", "probs": [0.6, 0.3, 0.1], "gold": 1}]
        raw = cb.summarize(rows, {})
        cal = cb.summarize(rows, {"score": 2.0})
        self.assertEqual((raw["acc"], cal["acc"]), (0.5, 0.5))
        self.assertEqual(raw["ece_raw"], raw["ece_cal"])
        self.assertNotEqual(cal["ece_raw"], cal["ece_cal"])
        self.assertAlmostEqual(raw["mae_raw"], (0.4 + 0.5) / 2, places=12)
        self.assertIn("mae_cal", cal)


class Freeze(unittest.TestCase):
    def test_bench_row_shapes(self):
        n = cb.slim_bench_row("sms-spam", {"label": "false", "prob": {"true": 0.25, "false": 0.75}})
        self.assertEqual((n["type"], n["probs"], n["gold"]), ("noul", [0.25, 0.75], 1))
        self.assertNotIn("keys", n)
        s = cb.slim_bench_row("yelp-stars", {"label": "3", "prob": {str(i): 0.2 for i in range(5)}})
        self.assertEqual((s["type"], s["gold"]), ("score", 3))
        c = cb.slim_bench_row("banking77", {"label": "b", "prob": {"a": 0.5, "b": 0.5}})
        self.assertEqual((c["type"], c["gold"]), ("choice", 1))
        self.assertEqual(cb.r12(0.1234567890123456), 0.123456789012)

    def test_cal_row_shape(self):
        r = cb.slim_cal_row("synth", {"type": "noul", "option_keys": ["yes", "no"], "probs": [0.9, 0.1],
                                      "gold": "no"})
        self.assertEqual((r["source"], r["gold"], r["probs"]), ("synth", 1, [0.9, 0.1]))

    def test_freeze_checks_rows_against_saved_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m"
            (src / "jevbench/raw").mkdir(parents=True)
            (src / "jevbench/predictions").mkdir()
            (src / "calibration").mkdir()
            raw = {"state": "secret text", "label": "true", "prob": {"true": 0.8, "false": 0.2}}
            (src / "jevbench/raw/sms-spam.jsonl").write_text(json.dumps(raw) + "\n")
            pred = src / "jevbench/predictions/sms-spam.jsonl"
            pred.write_text(json.dumps({"i": 0, "p_top": 0.8, "correct": 1}) + "\n")
            out = Path(d) / "probs"
            cb.main(["freeze", "--src", str(src), "--out", str(out)])
            frozen = (out / "jevbench_sms-spam.jsonl").read_text()
            self.assertNotIn("secret", frozen)
            self.assertEqual(json.loads(frozen)["gold"], 0)
            pred.write_text(json.dumps({"i": 0, "p_top": 0.7, "correct": 1}) + "\n")
            with self.assertRaises(SystemExit):
                cb.main(["freeze", "--src", str(src), "--out", str(out)])

    def test_sources_json_paths_are_relative_to_src(self):
        # Regression: SOURCES.json must carry no local filesystem layout
        # (bench/jev/DATA-LICENSES.md, "What SOURCES.json is").
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "some" / "deep" / "checkout" / "m"
            (src / "jevbench/raw").mkdir(parents=True)
            (src / "jevbench/predictions").mkdir()
            (src / "calibration").mkdir()
            raw = {"state": "s", "label": "true", "prob": {"true": 0.8, "false": 0.2}}
            (src / "jevbench/raw/sms-spam.jsonl").write_text(json.dumps(raw) + "\n")
            (src / "jevbench/predictions/sms-spam.jsonl").write_text(
                json.dumps({"i": 0, "p_top": 0.8, "correct": 1}) + "\n")
            out = Path(d) / "probs"
            cb.main(["freeze", "--src", str(src), "--out", str(out)])
            sources = json.loads((out / "SOURCES.json").read_text())
            self.assertEqual(set(sources), {"jevbench/raw/sms-spam.jsonl", "jevbench/predictions/sms-spam.jsonl"})
            for key in sources:
                self.assertFalse(Path(key).is_absolute(), key)
                self.assertNotIn(str(d), key)


class FitCommand(unittest.TestCase):
    def test_fit_refuses_anything_but_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            probs = Path(d) / "probs"
            probs.mkdir()
            with self.assertRaises(SystemExit):
                cb.main(["fit", "--heldout", str(probs), "--model-id", "m", "--build", "b",
                         "--out", str(Path(d) / "c.json")])

    def test_fit_writes_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "heldout"
            h.mkdir()
            cb.write_rows(h / "syn.jsonl", synthetic_rows(2.0, n=1200))
            cb.main(["fit", "--heldout", str(h), "--model-id", "m", "--build", "b",
                     "--out", str(Path(d) / "c.json")])
            doc = json.loads((Path(d) / "c.json").read_text())
            self.assertEqual(doc["types"]["choice"]["sources"], {"syn": 300})
            self.assertEqual(list(doc["decide_calibration"]), ["choice"])
            self.assertIn('[models."m".overrides.decide.calibration]',
                          cb.toml_snippet("m", doc["decide_calibration"]))

    def test_fit_reads_gzipped_heldout_rows(self):
        # Regression: a heldout/ directory holding only the committed
        # .jsonl.gz form (bench/jev/*/heldout/*.jsonl.gz) must fit exactly as
        # a plain .jsonl directory would.
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "heldout"
            h.mkdir()
            rows = synthetic_rows(2.0, n=1200)
            with gzip.open(h / "syn.jsonl.gz", "wt") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            cb.main(["fit", "--heldout", str(h), "--model-id", "m", "--build", "b",
                     "--out", str(Path(d) / "c.json")])
            doc = json.loads((Path(d) / "c.json").read_text())
            self.assertEqual(doc["types"]["choice"]["sources"], {"syn": 300})
            self.assertEqual(list(doc["decide_calibration"]), ["choice"])


class Licences(unittest.TestCase):
    def test_pool_refuses_a_restricted_row(self):
        for src in ("yelp-stars", "hellaswag", "duplicates"):
            rows = synthetic_rows(1.0, n=20, source="synth-score") + synthetic_rows(1.0, n=5, source=src)
            with self.assertRaises(ValueError):
                cb.pool(rows, "choice")
            with self.assertRaises(ValueError):  # even when asked for another type
                cb.pool(rows, "noul")

    def test_fit_refuses_a_heldout_dir_with_restricted_rows(self):
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "heldout"
            h.mkdir()
            cb.write_rows(h / "syn.jsonl", synthetic_rows(2.0, n=600, source="syn"))
            cb.write_rows(h / "ag-news.jsonl", synthetic_rows(2.0, n=10, source="ag-news"))
            with self.assertRaises(SystemExit) as e:
                cb.main(["fit", "--heldout", str(h), "--model-id", "m", "--build", "b",
                         "--out", str(Path(d) / "c.json")])
            self.assertIn("ag-news", str(e.exception.code))
            self.assertFalse((Path(d) / "c.json").exists())


if __name__ == "__main__":
    unittest.main()
