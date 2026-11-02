#!/usr/bin/env python3
"""CPU tests of quality_scorer.py (no model, no GPU, no native executable).

Run: python3 -B -m unittest discover -s scripts/slowlink-gap -p 'quality_test_*.py'
"""
import copy
import hashlib
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import quality_scorer as qs  # noqa: E402

REPS = 200  # the production reps are 10000; the machinery is identical


def synth(kl16, kl4, flip=None, extra_resume=True, rows=qs.EXPECTED_ROWS):
    recs = []
    for pair, fn in (("primary-ctx16k", kl16), ("primary-ctx4k", kl4)):
        for r in range(rows):
            ref = 100 + r
            cand = ref + 1 if (flip and flip(pair, r)) else ref
            recs.append({"schema": "hipfire.qwen4.slowlink_quality.kl_row.v1", "pair": pair, "row": r,
                         "context_position": 1000 + r, "next_token": 3 * r + 1, "kl_nats": fn(r), "kl_status": "ok",
                         "reference_top1_id": ref, "candidate_top1_id": cand, "top1_match": ref == cand,
                         "reference_margin": 1.0, "candidate_margin": 0.5})
    if extra_resume:
        for r in range(32):
            recs.append({"schema": "hipfire.qwen4.slowlink_quality.kl_row.v1", "pair": "resume-s1split", "row": r,
                         "context_position": 3400 + r, "next_token": None if r == 31 else 5, "kl_nats": 9.0,
                         "kl_status": "ok", "reference_top1_id": 1, "candidate_top1_id": 2, "top1_match": False})
    return recs


class Bootstrap(unittest.TestCase):
    def test_splitmix64_reference_vector(self):
        # the canonical splitmix64 sequence for seed 0
        self.assertEqual(qs.SplitMix64(0).next(), 0xE220A8397B1DCDAF)

    def test_seed_is_sha256_of_the_fixed_label(self):
        want = int.from_bytes(hashlib.sha256(b"hipfire.q1.moving-block-bootstrap.v1").digest()[:8], "big")
        self.assertEqual(qs.seed_from_label(qs.BOOT_LABEL), want)
        self.assertEqual((qs.BOOT_BLOCK, qs.BOOT_REPS), (32, 10000))

    def test_constant_strata_collapse_and_are_deterministic(self):
        strata = {"a": [(0.5, 1.0)] * 64, "b": [(0.5, 0.0)] * 64}
        r1 = qs.stratified_block_bootstrap(strata, reps=50)
        r2 = qs.stratified_block_bootstrap(strata, reps=50)
        self.assertEqual(r1, r2)
        self.assertEqual(r1["pooled"]["kl"], (0.5, 0.5))
        self.assertEqual(r1["a"]["top1"], (1.0, 1.0))
        self.assertEqual(r1["b"]["top1"], (0.0, 0.0))

    def test_short_stratum_is_invalid(self):
        with self.assertRaises(qs.Invalid):
            qs._boot({"a": [(0.1, 1.0)] * 10}, reps=5)


class Gate(unittest.TestCase):
    def absolute(self, k16, k4, **kw):
        return qs.analyze_absolute(synth(lambda r: k16, lambda r: k4, **kw), reps=REPS)

    def test_exact_bounds_are_the_sealed_values(self):
        self.assertEqual(qs.GATE_BOUNDS_EXACT, {"ctx16k": 0.03948458079446915, "ctx4k": 0.027284316982763658})
        self.assertEqual(qs.GATE_BOUNDS_DECLARED, {"ctx16k": 0.0395, "ctx4k": 0.0273})
        self.assertEqual(len(qs.BOUNDS_FILE_SHA256), 64)

    def test_means_exactly_at_the_bounds_pass(self):
        a = self.absolute(0.03948458079446915, 0.027284316982763658)
        g = qs.evaluate_gate(a)
        self.assertTrue(g["pass"])
        self.assertAlmostEqual(g["per_context"]["ctx16k"]["margin_nats"], 0.0, places=15)

    def test_one_context_over_its_bound_fails_whatever_the_other_is(self):
        g = qs.evaluate_gate(self.absolute(0.0396, 0.001))
        self.assertFalse(g["per_context"]["ctx16k"]["pass"])
        self.assertTrue(g["per_context"]["ctx4k"]["pass"])
        self.assertFalse(g["pass"])
        g = qs.evaluate_gate(self.absolute(0.001, 0.0274))
        self.assertFalse(g["pass"])

    def test_declared_rounded_is_looser_and_reported_separately(self):
        g = qs.evaluate_gate(self.absolute(0.03949, 0.02729))  # above exact, below rounded .0395/.0273? 4k: .02729<.0273
        self.assertFalse(g["pass"])
        self.assertTrue(g["pass_declared_rounded"])

    def test_pooled_is_reported_never_gated(self):
        a = self.absolute(0.001, 0.001)
        g = qs.evaluate_gate(a)
        self.assertFalse(g["pooled_gated"])
        self.assertAlmostEqual(g["pooled_mean_kl_nats"], 0.001)
        # CI is descriptive and cannot flip the verdict
        self.assertTrue(g["pass"])
        self.assertTrue(a["contexts"]["ctx16k"]["ci_kl"]["descriptive_only"])

    def test_raw_old_010_verdict_is_preserved_separately(self):
        g = qs.evaluate_gate(self.absolute(0.03, 0.02))
        self.assertTrue(g["pass"])  # new gate
        old = g["raw_old_q1_010_diagnostic"]
        self.assertFalse(old["pass"])  # historical .010 point-mean verdict, kept apart
        self.assertEqual(sorted(old["per_context_and_pooled"]), ["ctx16k", "ctx4k", "pooled"])
        self.assertIn("direction", g["direction_note"])

    def test_resume_and_other_pairs_never_enter_the_primary_means(self):
        a = self.absolute(0.01, 0.02)
        self.assertEqual(a["ignored_pairs"], ["resume-s1split"])
        self.assertEqual(a["pooled"]["rows"], 1024)
        self.assertAlmostEqual(a["pooled"]["raw_mean_kl_nats"], 0.015)

    def test_top1_agreement_and_max(self):
        recs = synth(lambda r: 0.01 * (r == 7) + 0.001, lambda r: 0.002, flip=lambda p, r: p == "primary-ctx16k" and r % 4 == 0)
        a = qs.analyze_absolute(recs, reps=REPS)
        self.assertAlmostEqual(a["contexts"]["ctx16k"]["top1_agree"], 0.75)
        self.assertEqual(a["contexts"]["ctx4k"]["top1_agree"], 1.0)
        self.assertAlmostEqual(a["contexts"]["ctx16k"]["max_kl_nats"], 0.011)


class Refusals(unittest.TestCase):
    def good(self):
        return synth(lambda r: 0.01, lambda r: 0.01)

    def refuses(self, recs):
        with self.assertRaises(qs.Invalid):
            qs.analyze_absolute(recs, reps=5)

    def test_missing_pair(self):
        self.refuses([r for r in self.good() if r["pair"] != "primary-ctx4k"])

    def test_row_count_must_be_exactly_512(self):
        self.refuses([r for r in self.good() if not (r["pair"] == "primary-ctx16k" and r["row"] == 511)])
        self.refuses(synth(lambda r: 0.01, lambda r: 0.01, rows=511))

    def test_duplicate_row(self):
        recs = self.good()
        recs.append(copy.deepcopy(recs[3]))
        self.refuses(recs)

    def test_nonfinite_negative_and_infinite(self):
        for bad in (float("inf"), float("nan"), -1.0):
            recs = self.good()
            recs[5]["kl_nats"] = bad
            self.refuses(recs)
        recs = self.good()
        recs[5]["kl_status"] = "infinite"
        recs[5]["kl_nats"] = None
        self.refuses(recs)

    def test_inconsistent_top1(self):
        recs = self.good()
        recs[2]["top1_match"] = False
        self.refuses(recs)

    def test_null_next_token_in_a_primary_row_is_refused(self):
        recs = self.good()
        recs[0]["next_token"] = None
        self.refuses(recs)


class Delta(unittest.TestCase):
    def test_zero_delta_diagnostic_has_no_verdict(self):
        off = synth(lambda r: 0.02, lambda r: 0.02)
        d = qs.analyze_delta_diagnostic(off, copy.deepcopy(off), reps=REPS)
        self.assertIsNone(d["verdict"])
        self.assertTrue(d["diagnostic_only"])
        self.assertEqual(d["pooled"]["delta_mean_kl_nats"], 0.0)
        self.assertEqual((d["pooled"]["ci_delta_kl"]["lo"], d["pooled"]["ci_delta_kl"]["hi"]), (0.0, 0.0))

    def test_positive_delta_and_shuffled_order(self):
        off = synth(lambda r: 0.02, lambda r: 0.02)
        on = synth(lambda r: 0.03, lambda r: 0.02)
        on.reverse()
        d = qs.analyze_delta_diagnostic(off, on, reps=REPS)
        self.assertAlmostEqual(d["contexts"]["ctx16k"]["delta_mean_kl_nats"], 0.01)
        self.assertAlmostEqual(d["contexts"]["ctx4k"]["delta_mean_kl_nats"], 0.0)

    def test_history_mismatch_is_refused(self):
        off = synth(lambda r: 0.02, lambda r: 0.02)
        on = copy.deepcopy(off)
        on[10]["context_position"] += 1
        with self.assertRaises(qs.Invalid):
            qs.analyze_delta_diagnostic(off, on, reps=5)


class Identity(unittest.TestCase):
    BETA = {"primary-ctx16k-teacher": "a" * 64, "primary-ctx4k-teacher": "b" * 64}

    def summary(self, ref16="a" * 64, ref4="b" * 64, cand16="c" * 64, cand4="d" * 64, kl16=5.12, kl4=2.56, agree=(500, 510)):
        def pair(name, cell, ref, cand, ksum, ag):
            return {"schema": qs.SUMMARY_SCHEMA, "pair": name, "reference": {"cell": cell, "stream_sha256": ref},
                    "candidate": {"cell": cell, "stream_sha256": cand}, "rows": 512, "expected_rows": 512,
                    "complete": True, "infinite_rows": 0, "kl_sum_nats": ksum, "kl_mean_nats": ksum / 512,
                    "top1_agree_rows": ag}
        return {"scope": "full", "pairs": [
            pair("primary-ctx16k", "primary-ctx16k-teacher", ref16, cand16, kl16, agree[0]),
            pair("primary-ctx4k", "primary-ctx4k-teacher", ref4, cand4, kl4, agree[1]),
            {"pair": "resume-s1split", "reference": {}, "candidate": {}}]}

    def cands(self, c16="c" * 64, c4="d" * 64):
        return {"stack": {"primary-ctx16k-teacher": c16, "primary-ctx4k-teacher": c4}}

    def test_identity_ok_and_records_every_hash(self):
        ev = qs.check_reference_identity(self.BETA, {"stack": self.summary(), "off": self.summary()}, {**self.cands(), "off": self.cands()["stack"]})
        self.assertEqual(ev["primary-ctx16k"]["beta_reference_stream_sha256"], "a" * 64)
        self.assertEqual(sorted(ev["primary-ctx4k"]["arms"]), ["off", "stack"])

    def test_reference_hash_differing_between_arms_is_fatal(self):
        with self.assertRaises(qs.Invalid):
            qs.check_reference_identity(self.BETA, {"stack": self.summary(), "off": self.summary(ref16="e" * 64)})

    def test_reference_not_the_sealed_beta_is_fatal(self):
        with self.assertRaises(qs.Invalid):
            qs.check_reference_identity(self.BETA, {"stack": self.summary(ref4="e" * 64)})

    def test_scored_candidate_must_be_the_arm_s1a_stream(self):
        with self.assertRaises(qs.Invalid):
            qs.check_reference_identity(self.BETA, {"stack": self.summary()}, self.cands(c16="9" * 64))

    def test_incomplete_or_infinite_pair_is_fatal(self):
        s = self.summary()
        s["pairs"][0]["infinite_rows"] = 1
        with self.assertRaises(qs.Invalid):
            qs.check_reference_identity(self.BETA, {"stack": s})

    def test_summary_must_match_scored_rows(self):
        recs = synth(lambda r: 0.01, lambda r: 0.005)
        s = self.summary(kl16=512 * 0.01, kl4=512 * 0.005, agree=(512, 512))
        qs.check_summary_matches_rows(s, recs, "stack")
        with self.assertRaises(qs.Invalid):
            qs.check_summary_matches_rows(self.summary(kl16=1.0, kl4=512 * 0.005, agree=(512, 512)), recs, "stack")
        with self.assertRaises(qs.Invalid):
            qs.check_summary_matches_rows(self.summary(kl16=512 * 0.01, kl4=512 * 0.005, agree=(511, 512)), recs, "stack")


if __name__ == "__main__":
    unittest.main()
