#!/usr/bin/env python3
"""CPU fixtures for the EP reserve-decoupling gate (scripts/fn_ep_compare.py `reserve`, scripts/fn-ep-gate.sh).

Frozen by mgfn/ep-reserve/DECISION.md sections 7.2 (row P7) and 8. No GPU, no cards: synthetic oracle directories
and synthetic perf/C1 documents. Nothing here asserts a measured number; the thresholds are the kill rules only.
"""
import contextlib
import copy
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fn_ep_compare", REPO_ROOT / "scripts" / "fn_ep_compare.py")
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load scripts/fn_ep_compare.py")
fc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fc
spec.loader.exec_module(fc)

R, B = "resident", "borrowed"
# Hand-derived from DECISION.md 4.2/5/7.2: rows -> (forward row sequence, bank sequence, restore-before-chunk list).
# 513..4096 rows borrow; <= 512 rows (every tiny tail) use the resident bank; restore once per borrowed epoch.
TABLE = {
    8: ([8], [R], []),
    9: ([9], [R], []),
    511: ([511], [R], []),
    512: ([512], [R], []),
    513: ([513], [B], [1]),
    1024: ([1024], [B], [1]),
    2047: ([2047], [B], [1]),
    2048: ([2048], [B], [1]),
    4095: ([4095], [B], [1]),
    4096: ([4096], [B], [1]),
    4097: ([4096, 1], [B, R], [1]),
    4396: ([4096, 300], [B, R], [1]),
    4608: ([4096, 512], [B, R], [1]),         # [4096, 512] restores before the resident S1 call
    8192: ([4096, 4096], [B, B], [2]),        # stays borrowed until completion
    8204: ([4096, 4096, 12], [B, B, R], [2]),  # restores before the Full tail
    9192: ([4096, 4096, 1000], [B, B, B], [3]),
}
GATE_LENGTHS = fc.RESERVE_BOUNDARY_LENGTHS


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))


def ids_for(prompt, tokens):
    return [100 + prompt * 10 + i for i in range(tokens)]


def prompt_record(i, n, chunks, banks, points, on, ranks, phase="target", ordinal=0):
    """One merged outermost-entry record (the harvest's output shape): `prompts[i]` for (target, 0), else an entry."""
    events = [{"before_chunk": p, "bytes": 4 << 30, "restore_ms": 160.0, "barrier_wait_ms": [1.0, 2.0], "s2_dma_tail_ms": [0.5, 0.7]}
              for p in points]
    epochs = sum(1 for k, b in enumerate(banks) if b == B and (k == 0 or banks[k - 1] != B))
    p = {"prompt": i, "phase": phase, "ordinal": ordinal, "rows": n, "chunk_rows": chunks, "banks": banks, "borrowed_epochs": epochs,
         "restores": events, "elapsed_ms_incl_restore": 1000.0 + 160.0 * len(events)}
    if on:
        p.update({"donor_hash_before": ["aa", "bb"], "donor_hash_after": ["aa", "bb"]})
        p["exchange_address_before"] = [r["exchange_address"] for r in ranks]
        p["exchange_address_after"] = [r["exchange_address"] for r in ranks]
    return p


# Decode/verify-window entries of one MTP prompt: target ordinals 0.. (2..8 rows each), resident bank.
DECODE_ENTRIES = ((0, [3]), (1, [5]), (2, [1]))


def build_receipt(on, lengths, tokens, resident, mode="ar", **extra):
    first = resident - 7
    ranks = []
    for r in range(2):
        rank = {"rank": r, "resident_layers": resident, "source_generation": "src-1"}
        if on:
            rank.update({"plan_digest": 123456789, "donor_first_layer": first, "donor_count": 7, "s2_donor_layers": [first - 2, first - 1],
                         "prompt_rows": 4096, "exchange_rows": 4096, "exchange_address": "0x7f00%d000" % r,
                         "s1_admitted": True, "s2_admitted": True})
        ranks.append(rank)
    prompts = []
    for i, n in enumerate(lengths):
        chunks, banks, points = TABLE[n]
        if not on:
            banks, points = [R] * len(chunks), []
        if mode == "mtp":
            # No target prefill entry: the native-MTP prefill is the one whole-prompt entry (mtp_fill,0) and (target,0) is
            # the first decode/verify window.
            p = prompt_record(i, n, chunks, banks, points, on, ranks, "mtp_fill", 0)
            p["entries"] = [prompt_record(i, sum(c), c, [R] * len(c), [], on, ranks, "target", k) for k, c in DECODE_ENTRIES]
        else:
            p = prompt_record(i, n, chunks, banks, points, on, ranks)
        prompts.append(p)
    rec = {"schema": fc.RESERVE_RECEIPT_SCHEMA, "decouple": on, "binary_id": "bin-1", "model_sha256": "m", "seed": 7, "mode": mode,
           "prompt_chunk_rows": 4096, "max_seq": 16384, "complete": True, "ranks": ranks, "prompts": prompts,
           "token_ids": [ids_for(i, tokens) for i in range(len(lengths))]}
    rec.update(extra)
    return rec


def write_arm(root, name, on, lengths, tokens, resident, mutate=None, meta_mutate=None, mode="ar"):
    d = root / name
    meta = {"completed": True, "ranks": 2, "max_seq": 16384, "tokens": tokens, "vocab": 4, "model_sha256": "m",
            "prompts": [{"prompt": i, "token_count": n, "file_md5": "f%d" % i, "ids_md5": "i%d" % i} for i, n in enumerate(lengths)]}
    if meta_mutate:
        meta_mutate(meta)
    logits, state = [], []
    for i in range(len(lengths)):
        ids = ids_for(i, tokens)
        if mode == "mtp":
            # the MTP oracle writes one `window` record per window and a closing `tokens` record, never `logits` records
            meta["spec"] = {"kind": "mtp", "temp": 0.0, "seed": 7}
            for w in range(tokens):
                logits.append(json.dumps({"kind": "window", "prompt": i, "window": w, "position": lengths[i] + w,
                                          "target_position": lengths[i] + w, "emit": [ids[w]]}))
            logits.append(json.dumps({"kind": "tokens", "prompt": i, "windows": tokens - 1, "count": tokens, "ids_md5": "x%d" % i, "ids": ids}))
        else:
            for step in range(tokens + 1):
                sha = "h%d-%d" % (i, step)
                logits.append(json.dumps({"kind": "logits", "prompt": i, "step": step, "token": None if step == 0 else ids[step - 1],
                                          "position": step, "logits_sha256": sha, "rank_logits_sha256": [sha, sha]}))
        for rank in range(2):
            state.append(json.dumps({"kind": "state", "prompt": i, "step": 1, "rank": rank, "digest": [["position", "1"], ["gdn[0]", "s"]]}))
    write(d / "meta.json", meta)
    write(d / "logits.jsonl", "\n".join(logits) + "\n")
    write(d / "state.jsonl", "\n".join(state) + "\n")
    rec = build_receipt(on, lengths, tokens, resident, mode=mode)
    if mutate:
        mutate(rec)
    write(d / "reserve-receipt.json", rec)
    return d


def check(root, lengths=GATE_LENGTHS, tokens=4, off_mut=None, on_mut=None, meta_mut=None, mode="ar", **kw):
    off = write_arm(root, "off", False, lengths, tokens, 28, off_mut, meta_mut, mode)
    on = write_arm(root, "on", True, lengths, tokens, 33, on_mut, meta_mut, mode)
    kw.setdefault("min_tokens", tokens)
    kw.setdefault("require_lengths", lengths)
    v, _ = fc.compare_reserve_arms(str(off), str(on), **kw)
    return v


class ArmsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assertRejects(self, v, family, needle):
        msgs = [m["message"] for m in v.mismatches if m["check"] == family]
        self.assertFalse(v.ok, "expected a failure in %s" % family)
        self.assertTrue(any(needle in m for m in msgs), "no %s failure containing %r in %r" % (family, needle, v.mismatches))

    def test_table_matches_the_module_helpers(self):
        for rows, (chunks, banks, points) in TABLE.items():
            self.assertEqual(fc.reserve_chunk_plan(rows), chunks, rows)
            self.assertEqual(fc.reserve_expected_banks(chunks), banks, rows)
            self.assertEqual(fc.reserve_expected_restore_points(banks), points, rows)

    def test_boundary_matrix_passes(self):
        v = check(self.root)
        self.assertTrue(v.ok, v.mismatches)
        for family in ("receipt", "plan", "boundary", "restore", "ids", "cross", "coverage", "logits", "state"):
            self.assertGreater(v.families[family]["checked"], 0, family)

    def test_tail_fixtures_pass(self):
        v = check(self.root, lengths=(4097, 4396, 4608, 8192, 8204, 9192), require_lengths=())
        self.assertTrue(v.ok, v.mismatches)

    def test_absent_receipts_rejected(self):
        off = write_arm(self.root, "off", False, GATE_LENGTHS, 4, 28)
        on = write_arm(self.root, "on", True, GATE_LENGTHS, 4, 33)
        (on / "reserve-receipt.json").unlink()
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4)
        self.assertRejects(v, "receipt", "on arm: reserve-receipt.json absent")

    def test_absent_rank_receipts_rejected(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r.update(ranks=[])), "plan", "exactly ranks 0..1")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r.update(ranks=r["ranks"][:1])), "plan", "exactly ranks 0..1")

    def test_plan_digest_mismatch_across_ranks(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r["ranks"][1].update(plan_digest="plan-B")), "plan", "disagree on plan_digest")

    def test_source_mismatch_across_arms(self):
        self.assertRejects(check(self.root, on_mut=lambda r: [x.update(source_generation="src-2") for x in r["ranks"]]),
                           "cross", "source_generation differs")

    def test_seed_binary_and_model_mismatch(self):
        for key, value in (("seed", 8), ("binary_id", "bin-2"), ("model_sha256", "other"), ("ple_sha256", "ple")):
            with self.subTest(key=key):
                shutil.rmtree(self.root / "off", ignore_errors=True), shutil.rmtree(self.root / "on", ignore_errors=True)
                self.assertRejects(check(self.root, on_mut=lambda r, k=key, val=value: r.update({k: val})), "cross", "differ on %s" % key)

    def test_wrong_chunk_rejected(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r.update(prompt_chunk_rows=2048)), "chunk", "never lowered")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def lowered(r):
            r["prompts"][-1]["chunk_rows"] = [2048, 2048, 300]

        self.assertRejects(check(self.root, on_mut=lowered), "chunk", "forward row sequence")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["ranks"][0].update(exchange_rows=512)), "chunk", "exchange_rows")

    def test_wrong_resident_count_rejected(self):
        def low(r):
            for x in r["ranks"]:
                x.update(resident_layers=32, donor_first_layer=25, s2_donor_layers=[23, 24])

        self.assertRejects(check(self.root, on_mut=low), "plan", "32 resident layers < 33")

    def test_no_residency_gain_rejected(self):
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir()
        off = write_arm(self.root, "off", False, GATE_LENGTHS, 4, 33)
        on = write_arm(self.root, "on", True, GATE_LENGTHS, 4, 33)
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4)
        self.assertRejects(v, "cross", "no residency gain")

    def test_donor_boundaries_must_agree(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r["ranks"][0].update(s2_donor_layers=[24, 26])), "plan", "S2 donors")

    def test_omitted_restore_timing_rejected(self):
        def omit(r):
            del r["prompts"][-1]["restores"][0]["restore_ms"]

        self.assertRejects(check(self.root, on_mut=omit), "restore", "restore timing omitted")

    def test_elapsed_must_include_restore(self):
        def small(r):
            r["prompts"][4]["elapsed_ms_incl_restore"] = 100.0  # 513 rows: one 160 ms restore does not fit

        self.assertRejects(check(self.root, on_mut=small), "restore", "includes the")

    def test_repeated_or_missing_restore_rejected(self):
        def per_tile(r):
            p = r["prompts"][GATE_LENGTHS.index(4396)]
            p["restores"].insert(0, dict(p["restores"][0], before_chunk=0))

        self.assertRejects(check(self.root, on_mut=per_tile), "restore", "restores before chunks")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def none(r):
            r["prompts"][GATE_LENGTHS.index(4096)]["restores"] = []

        self.assertRejects(check(self.root, on_mut=none), "restore", "restores before chunks")

    def test_donor_hash_and_exchange_address_must_be_stable(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][4].update(donor_hash_after=["aa", "XX"])), "restore", "donor bytes differ")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][4].update(exchange_address_after=["0x1", "0x2"])), "restore", "exchange address moved")

    def test_borrowed_bank_for_small_prompt_rejected(self):
        def wrong(r):
            r["prompts"][0]["banks"] = [B]

        self.assertRejects(check(self.root, on_mut=wrong), "boundary", "bank sequence")

    def test_missing_id_arrays_rejected(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r.pop("token_ids")), "ids", "id arrays missing")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["token_ids"].__setitem__(3, [])), "ids", "id array missing or empty")

    def test_id_mismatch_rejected(self):
        def flip(r):
            r["token_ids"][2][1] += 1

        v = check(self.root, on_mut=flip)
        self.assertRejects(v, "ids", "differ from the off arm at token 1")
        self.assertRejects(v, "ids", "disagrees with logits.jsonl")

    def test_truncated_runs_rejected(self):
        self.assertRejects(check(self.root, on_mut=lambda r: r.update(complete=False)), "receipt", "truncated")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"].pop()), "boundary", "prompt receipts")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["token_ids"][0].pop()), "ids", "truncated")

    def test_too_few_committed_tokens_rejected(self):
        v = check(self.root, min_tokens=128)
        self.assertRejects(v, "ids", "at least 128 required")

    def test_uncovered_boundary_lengths_rejected(self):
        v = check(self.root, lengths=(8, 4096), require_lengths=GATE_LENGTHS)
        self.assertRejects(v, "coverage", "boundary lengths not exercised")

    def test_mode_mismatch_rejected(self):
        self.assertRejects(check(self.root, expect_mode="mtp"), "receipt", "mode 'ar', expected mtp")

    # ---- MTP: per prompt a target prefill, a native-MTP fill and decode/verify entries
    def test_mtp_receipts_with_fill_and_decode_entries_pass(self):
        v = check(self.root, mode="mtp", expect_mode="mtp")
        self.assertTrue(v.ok, v.mismatches)
        for family in ("receipt", "plan", "boundary", "restore", "ids", "cross"):
            self.assertGreater(v.families[family]["checked"], 0, family)

    def test_mtp_tail_fixtures_pass(self):
        v = check(self.root, lengths=(4097, 4396, 4608, 8192, 8204, 9192), require_lengths=(), mode="mtp", expect_mode="mtp")
        self.assertTrue(v.ok, v.mismatches)

    @staticmethod
    def _entry(rec, prompt, phase, ordinal):
        p = rec["prompts"][prompt]
        if (p["phase"], p["ordinal"]) == (phase, ordinal):
            return p
        return next(e for e in p["entries"] if (e["phase"], e["ordinal"]) == (phase, ordinal))

    def test_mtp_missing_fill_entry_rejected(self):
        def not_fill(r):  # the whole-prompt entry is not the native-MTP fill
            r["prompts"][3].update(phase="target", ordinal=0)

        needle = "prompt 3: mode mtp needs the whole-prompt entry to be (mtp_fill,0)"
        self.assertRejects(check(self.root, on_mut=not_fill, mode="mtp"), "boundary", needle)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, off_mut=not_fill, mode="mtp"), "boundary", needle)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def second_fill(r):
            r["prompts"][3]["entries"].append(prompt_record(3, 512, [512], [R], [], True, r["ranks"], "mtp_fill", 1))

        self.assertRejects(check(self.root, on_mut=second_fill, mode="mtp"), "boundary", "needs exactly one (mtp_fill,0) entry, got a further")


    def test_mtp_fill_entry_follows_the_chunk_bank_and_restore_rules_against_the_oracle_rows(self):
        i = GATE_LENGTHS.index(4396)
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][i].update(chunk_rows=[2048, 2048, 300]), mode="mtp"),
                           "chunk", "prompt %d: forward row sequence" % i)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][i].update(banks=["resident", "resident"]), mode="mtp"),
                           "boundary", "prompt %d: bank sequence" % i)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][i].update(restores=[]), mode="mtp"),
                           "restore", "prompt %d: restores before chunks" % i)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        # the fill carries the WHOLE prompt: fewer rows than the oracle prompt is a boundary failure
        self.assertRejects(check(self.root, on_mut=lambda r: r["prompts"][i].update(rows=300, chunk_rows=[300], banks=["resident"],
                                                                                      restores=[], borrowed_epochs=0), mode="mtp"),
                           "boundary", "do not match the oracle prompt")

    def test_mtp_first_target_window_is_a_generic_entry(self):
        # (target,0) is the first decode/verify window in mtp mode, not a prefill: 2..8 rows, resident, no hashing required
        self.assertEqual(self._entry(build_receipt(True, (513,), 4, 33, mode="mtp"), 0, "target", 0)["chunk_rows"], [3])
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 4, "target", 0).update(rows=513, chunk_rows=[513]), mode="mtp"),
                           "boundary", "prompt 4 entry (target,0): bank sequence")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 4, "target", 0).update(chunk_rows=[5000], rows=5000), mode="mtp"),
                           "chunk", "prompt 4 entry (target,0): forward row sequence")

    def test_mtp_logits_records_must_be_identical_between_the_arms(self):
        off = write_arm(self.root, "off", False, (513, 4097), 4, 28, mode="mtp")
        on = write_arm(self.root, "on", True, (513, 4097), 4, 33, mode="mtp")
        self.assertFalse(any(json.loads(line).get("kind") == "logits" for line in (on / "logits.jsonl").read_text().splitlines()))
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4, expect_mode="mtp")
        self.assertTrue(v.ok, v.mismatches)
        self.assertGreater(v.families["logits"]["checked"], 0)
        lines = (on / "logits.jsonl").read_text().splitlines()
        record = json.loads(lines[2])
        record["emit"] = [record["emit"][0] + 1]
        lines[2] = json.dumps(record)
        (on / "logits.jsonl").write_text("\n".join(lines) + "\n")
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4, expect_mode="mtp")
        self.assertRejects(v, "logits", "logits.jsonl record 2 (window prompt 0 window 2) differs")
        self.assertEqual(len([m for m in v.mismatches if m["check"] == "logits"]), 1)  # the first differing record is the report
        (on / "logits.jsonl").write_text("\n".join(lines[:-1]) + "\n")
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4, expect_mode="mtp")
        self.assertRejects(v, "record_set", "record counts differ")

    def test_mtp_token_chain_disagreement_rejected(self):
        def flip(r):
            r["token_ids"][1][0] += 1

        v = check(self.root, on_mut=flip, mode="mtp", expect_mode="mtp")
        self.assertRejects(v, "ids", "disagrees with the logits.jsonl tokens record")

    def test_decode_entry_hash_address_epoch_and_timing_rules(self):
        def moved_hash(r):
            self._entry(r, 4, "target", 1).update(donor_hash_after=["aa", "XX"])

        self.assertRejects(check(self.root, on_mut=moved_hash, mode="mtp"), "restore", "prompt 4 entry (target,1): donor bytes differ")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def moved_address(r):
            self._entry(r, 4, "target", 2).update(exchange_address_after=["0x1", "0x2"])

        self.assertRejects(check(self.root, on_mut=moved_address, mode="mtp"), "restore", "prompt 4 entry (target,2): exchange address moved")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def absent_hash(r):
            del self._entry(r, 4, "target", 1)["donor_hash_before"]

        self.assertRejects(check(self.root, on_mut=absent_hash, mode="mtp"), "restore", "donor_hash_before must be 2 non-empty strings")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 4, "target", 1).update(borrowed_epochs=1), mode="mtp"),
                           "restore", "prompt 4 entry (target,1): borrowed_epochs=1, expected 0")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def borrowed_without_timing(r):  # a 600-row decode-phase entry that borrowed and restored, restore_ms omitted
            e = self._entry(r, 4, "target", 1)
            e.update(rows=600, chunk_rows=[600], banks=["borrowed"], borrowed_epochs=1,
                     restores=[{"before_chunk": 1, "bytes": 1 << 30, "barrier_wait_ms": [1.0, 2.0], "s2_dma_tail_ms": [0.5, 0.7]}])

        self.assertRejects(check(self.root, on_mut=borrowed_without_timing, mode="mtp"), "restore", "prompt 4 entry (target,1): restore timing omitted")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def good_borrowed(r):
            e = self._entry(r, 4, "target", 1)
            e.update(rows=600, chunk_rows=[600], banks=["borrowed"], borrowed_epochs=1, elapsed_ms_incl_restore=900.0,
                     restores=[{"before_chunk": 1, "bytes": 1 << 30, "restore_ms": 100.0, "barrier_wait_ms": [1.0, 2.0], "s2_dma_tail_ms": [0.5, 0.7]}])

        v = check(self.root, on_mut=good_borrowed, mode="mtp")
        self.assertTrue(v.ok, v.mismatches)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def resident_unhashed(r):  # the forward hashes only entries that open on the borrowed bank
            e = self._entry(r, 4, "target", 1)
            del e["donor_hash_before"], e["donor_hash_after"]

        v = check(self.root, on_mut=resident_unhashed, mode="mtp")
        self.assertTrue(v.ok, v.mismatches)
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")

        def borrowed_unhashed(r):
            good_borrowed(r)
            e = self._entry(r, 4, "target", 1)
            del e["donor_hash_before"], e["donor_hash_after"]

        self.assertRejects(check(self.root, on_mut=borrowed_unhashed, mode="mtp"), "restore", "donor_hash_before must be 2 non-empty strings")

    def test_duplicate_entry_key_rejected(self):
        def dup(r):
            entries = r["prompts"][2]["entries"]
            entries.append(json.loads(json.dumps(entries[0])))

        self.assertRejects(check(self.root, on_mut=dup, mode="mtp"), "boundary", "duplicate entry key")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 2, "target", 1).update(ordinal=0), mode="mtp"),
                           "boundary", "duplicate entry key")

    def test_entry_key_and_phase_rules(self):
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 2, "target", 1).update(phase="prefill"), mode="mtp"),
                           "boundary", "malformed")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 2, "target", 1).update(prompt=5), mode="mtp"), "boundary", "malformed")
        shutil.rmtree(self.root / "off"), shutil.rmtree(self.root / "on")
        self.assertRejects(check(self.root, on_mut=lambda r: self._entry(r, 2, "target", 1).update(chunk_rows=[5000], rows=5000), mode="mtp"),
                           "chunk", "1..4096-row calls")

    def test_ar_run_with_a_fill_entry_rejected(self):
        def add(r):
            r["prompts"][1]["entries"] = [prompt_record(1, 9, [9], [R], [], True, r["ranks"], "mtp_fill", 0)]

        self.assertRejects(check(self.root, on_mut=add), "boundary", "(mtp_fill,0) in an ar-mode run")

    def test_state_and_logits_corruption_still_caught(self):
        off = write_arm(self.root, "off", False, (513,), 4, 28)
        on = write_arm(self.root, "on", True, (513,), 4, 33)
        lines = (on / "state.jsonl").read_text().replace('"s"', '"corrupt"', 1)
        (on / "state.jsonl").write_text(lines)
        v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=4)
        self.assertRejects(v, "state", "digest row(s) differ")

    def test_baseline_repeatability_mode(self):
        a = write_arm(self.root, "a", False, (513, 4097), 4, 28)
        b = write_arm(self.root, "b", False, (513, 4097), 4, 28)
        v, _ = fc.compare_reserve_arms(str(a), str(b), min_tokens=4, baseline_repeat=True)
        self.assertTrue(v.ok, v.mismatches)
        c = write_arm(self.root, "c", True, (513, 4097), 4, 33)
        v, _ = fc.compare_reserve_arms(str(a), str(c), min_tokens=4, baseline_repeat=True)
        self.assertFalse(v.ok)

    def test_cli_exit_codes(self):
        off = write_arm(self.root, "off", False, GATE_LENGTHS, 4, 28)
        on = write_arm(self.root, "on", True, GATE_LENGTHS, 4, 33)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fc.run(["reserve", "arms", str(off), str(on), "--min-tokens", "4", "--require-lengths", ",".join(map(str, GATE_LENGTHS)),
                           "--json", str(self.root / "v.json")])
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(json.loads((self.root / "v.json").read_text())["verdict"], "PASS")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fc.run(["reserve", "arms", str(off), str(on)]), 1)  # default 128 tokens
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(fc.run(["reserve", "arms", str(off), str(self.root / "nope")]), 2)

    def test_legacy_invocation_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fc.self_test(), 0)


def perf_doc(off_ms=1000.0, on_ms=1000.0, gate="pp32k", resident=(33, 33), **over):
    doc = {"schema": fc.RESERVE_PERF_SCHEMA, "gate": gate, "mode": "ar", "binary_id": "bin-1", "prompt_sha256": "p",
           "prompt_tokens": 32780, "generated_tokens": 1, "chunk_rows": 4096, "max_seq": 40960,
           "elapsed_includes_restore": True, "elapsed_spans_all_ranks": True,
           "launch_order": ["off", "on", "off", "on", "off", "on"], "samples": []}
    for arm, ms in (("off", off_ms), ("on", on_ms)):
        for proc in range(3):
            for req in range(-1, 3):
                s = {"arm": arm, "process": proc, "request": max(req, 0), "warmup": req < 0, "elapsed_ms": ms, "binary_id": "bin-1",
                     "prompt_sha256": "p", "mode": "ar", "chunk_rows": 4096, "generated_tokens": 1}
                if arm == "on":
                    s.update(restore_ms=160.0, resident_layers=list(resident))
                doc["samples"].append(s)
    doc.update(over)
    return doc


class PerfTest(unittest.TestCase):
    def test_pass_and_all_samples_retained(self):
        out = fc.evaluate_reserve_perf(perf_doc())
        self.assertEqual(out["verdict"], "PASS", out)
        self.assertEqual(len(out["samples"]), 24)
        self.assertEqual(out["latency_ratio"], 1.0)

    def test_latency_edge_exactly_five_percent_passes_above_kills(self):
        self.assertEqual(fc.evaluate_reserve_perf(perf_doc(on_ms=1050.0))["verdict"], "PASS")
        out = fc.evaluate_reserve_perf(perf_doc(on_ms=1050.1))
        self.assertEqual(out["verdict"], "KILL")
        self.assertTrue(any("latency ratio" in r for r in out["reasons"]), out)

    def test_throughput_edge(self):
        self.assertEqual(fc.reserve_pp32k_kill_reasons(Fraction(19, 20), Fraction(1)), [])
        reasons = fc.reserve_pp32k_kill_reasons(Fraction(949, 1000), Fraction(1))
        self.assertEqual(len(reasons), 1)
        self.assertIn("throughput", reasons[0])
        self.assertEqual(fc.reserve_pp32k_kill_reasons(Fraction(1), Fraction(21, 20)), [])
        self.assertEqual(len(fc.reserve_pp32k_kill_reasons(Fraction(1), Fraction(1051, 1000))), 1)
        # the latency rule is the stricter one: a 0.95 throughput ratio is a 1.0526 latency ratio
        self.assertEqual(len(fc.reserve_pp32k_kill_reasons(Fraction(19, 20), Fraction(20, 19))), 1)

    def test_resident_below_33_kills(self):
        out = fc.evaluate_reserve_perf(perf_doc(resident=(33, 32)))
        self.assertEqual(out["verdict"], "KILL")
        self.assertIn("32 resident layers < 33", out["reasons"][0])

    def test_noisy_pair_crossing_the_floor_earns_no_pass(self):
        doc = perf_doc()
        for s in doc["samples"]:
            if s["arm"] == "on" and s["process"] == 2:
                s["elapsed_ms"] = 1200.0
        out = fc.evaluate_reserve_perf(doc)
        self.assertEqual(out["verdict"], "INCONCLUSIVE", out)
        self.assertEqual(fc.evaluate_reserve_perf(doc)["latency_ratio"], 1.0)

    def test_process_medians_not_sample_means(self):
        doc = perf_doc()
        first = [s for s in doc["samples"] if s["arm"] == "on" and s["process"] == 0 and not s["warmup"]]
        first[0]["elapsed_ms"] = 9000.0  # one outlier request inside a process does not move its median
        self.assertEqual(fc.evaluate_reserve_perf(doc)["verdict"], "PASS")

    def test_warmup_excluded(self):
        doc = perf_doc()
        for s in doc["samples"]:
            if s["warmup"] and s["arm"] == "on":
                s["elapsed_ms"] = 9000.0
        self.assertEqual(fc.evaluate_reserve_perf(doc)["verdict"], "PASS")

    def test_invalid_documents(self):
        def bad(mutate, needle):
            doc = perf_doc()
            mutate(doc)
            out = fc.evaluate_reserve_perf(doc)
            self.assertEqual(out["verdict"], "INVALID", needle)
            self.assertTrue(any(needle in r for r in out["reasons"]), (needle, out["reasons"]))

        bad(lambda d: [s.pop("restore_ms") for s in d["samples"] if s["arm"] == "on"], "restore timing omitted")
        bad(lambda d: d.update(chunk_rows=2048), "chunk_rows")
        bad(lambda d: d.update(mode=None), "mode")
        bad(lambda d: d.update(elapsed_includes_restore=False), "elapsed_includes_restore")
        bad(lambda d: d.update(elapsed_spans_all_ranks=False), "elapsed_spans_all_ranks")
        bad(lambda d: d.update(launch_order=["off", "off", "off", "on", "on", "on"]), "launch_order")
        bad(lambda d: d.update(prompt_tokens=16000), "pp32k prompt")
        bad(lambda d: d.update(samples=[s for s in d["samples"] if s["process"] != 2]), "truncated run")
        bad(lambda d: d.update(samples=[s for s in d["samples"] if not (s["process"] == 1 and s["request"] == 2)]), "timed requests")
        bad(lambda d: d["samples"][0].update(binary_id="other"), "binary_id")
        bad(lambda d: [s.update(prompt_sha256="q") for s in d["samples"] if s["arm"] == "on"], "prompt_sha256")
        bad(lambda d: d.update(schema="x"), "schema")

    def test_invented_references_rejected(self):
        def bad(mutate):
            doc = perf_doc(gate="pp64k", prompt_tokens=65548)
            mutate(doc)
            out = fc.evaluate_reserve_perf(doc)
            self.assertEqual(out["verdict"], "INVALID", out)
            self.assertTrue(any("not accepted" in r or "off" in r for r in out["reasons"]), out["reasons"])

        bad(lambda d: d.update(reference_tok_s=4893))
        bad(lambda d: d.update(baseline_ms=13000))
        bad(lambda d: d.update(samples=[s for s in d["samples"] if s["arm"] == "on"], launch_order=["on", "off"] * 3))

    def test_other_cells_are_reported_not_judged(self):
        out = fc.evaluate_reserve_perf(perf_doc(gate="pp64k", prompt_tokens=65548, on_ms=1300.0))
        self.assertEqual(out["verdict"], "REPORTED")
        self.assertGreater(out["latency_ratio"], 1.05)
        # but residency below 33 still kills any cell
        self.assertEqual(fc.evaluate_reserve_perf(perf_doc(gate="pp8k", prompt_tokens=8204, resident=(32, 33)))["verdict"], "KILL")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.json"
            path.write_text(json.dumps(perf_doc()))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "perf", str(path)]), 0)
            path.write_text(json.dumps(perf_doc(on_ms=1100.0)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "perf", str(path)]), 1)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "perf", str(Path(tmp) / "missing.json")]), 2)


WEIGHTS = {"code": 0.3, "file_edit": 0.1, "json": 0.15, "prose": 0.15, "reasoning": 0.2, "summarization": 0.1}


def c1_summary(decode, tau=3.0):
    cats = {}
    for cat, w in WEIGHTS.items():
        d = decode[cat] if isinstance(decode, dict) else decode
        cats[cat] = {"n_timed": 20, "n_ok": 20, "decode_med": d, "weight": w, "mtp": {"tau_window_weighted": tau}}
    return {"complete": True, "categories": cats}


def c1_doc(off=100.0, on=108.0, resident=(33, 33), pairs=3):
    servers = []
    for i in range(pairs):
        for arm, val in (("off", off), ("on", on)):
            servers.append({"arm": arm, "order": len(servers), "binary_id": "bin-1", "summary": c1_summary(val),
                            "resident_layers": [28, 28] if arm == "off" else list(resident)})
    return {"schema": fc.RESERVE_C1_SCHEMA, "binary_id": "bin-1", "max_seq": 16384, "spec": "mtp",
            "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20}, "servers": servers}


class C1Test(unittest.TestCase):
    def test_plus_eight_percent_exactly_passes(self):
        out = fc.evaluate_reserve_c1(c1_doc())
        self.assertEqual(out["verdict"], "PASS", out)
        self.assertAlmostEqual(out["weighted_gain_pct"], 8.0)

    def test_below_eight_percent_kills(self):
        out = fc.evaluate_reserve_c1(c1_doc(on=107.99))
        self.assertEqual(out["verdict"], "KILL")
        self.assertIn("< +8%", out["reasons"][0])
        self.assertTrue(fc.reserve_c1_kills(Fraction(1079, 1000)))
        self.assertFalse(fc.reserve_c1_kills(Fraction(27, 25)))

    def test_resident_below_33_kills_even_with_gain(self):
        out = fc.evaluate_reserve_c1(c1_doc(on=120.0, resident=(33, 32)))
        self.assertEqual(out["verdict"], "KILL")
        self.assertIn("32 resident layers < 33", out["reasons"][0])

    def test_category_cliff_goes_to_review(self):
        doc = c1_doc()
        for s in doc["servers"]:
            if s["arm"] == "on":
                for cat, c in s["summary"]["categories"].items():
                    c["decode_med"] = 90.0 if cat == "code" else 120.0
        out = fc.evaluate_reserve_c1(doc)
        self.assertEqual(out["verdict"], "REVIEW", out)
        self.assertIn("code", out["reasons"][0])

    def test_tau_shift_is_noted(self):
        doc = c1_doc()
        for s in doc["servers"]:
            if s["arm"] == "on":
                s["summary"]["categories"]["prose"]["mtp"]["tau_window_weighted"] = 4.0
        out = fc.evaluate_reserve_c1(doc)
        self.assertEqual(out["verdict"], "PASS")
        self.assertTrue(any(n.startswith("prose") for n in out["tau_notes"]), out)

    def test_invalid_manifests(self):
        def bad(mutate, needle):
            doc = c1_doc()
            mutate(doc)
            out = fc.evaluate_reserve_c1(doc)
            self.assertEqual(out["verdict"], "INVALID", needle)
            self.assertTrue(any(needle in r for r in out["reasons"]), (needle, out["reasons"]))

        bad(lambda d: d["servers"][0]["summary"].update(complete=False), "not complete")
        bad(lambda d: d["servers"][0]["summary"]["categories"]["json"].update(n_ok=19), "successful timed requests")
        bad(lambda d: d["servers"][1]["summary"]["categories"]["json"]["mtp"].pop("tau_window_weighted"), "tau_window_weighted")
        bad(lambda d: d["servers"][0]["summary"]["categories"].pop("prose"), "categories")
        bad(lambda d: d["servers"][0]["summary"]["categories"]["code"].update(weight=0.5), "weights")
        bad(lambda d: d.update(servers=d["servers"][:4]), "need >= 3")
        bad(lambda d: d.update(max_seq=8192), "max_seq")
        bad(lambda d: d.update(spec="off"), "spec")
        bad(lambda d: d.update(sampling={"temperature": 0.0}), "sampling")
        bad(lambda d: d["servers"][2].update(order=0), "interleaved")
        bad(lambda d: d["servers"][1].update(binary_id="other"), "same binary")
        bad(lambda d: d.update(reference_c1=82.6), "not accepted")
        bad(lambda d: d["servers"][0]["summary"].update(weighted_decode=50.0), "recomputed")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c1.json"
            path.write_text(json.dumps(c1_doc()))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "c1", str(path)]), 0)
            path.write_text(json.dumps(c1_doc(on=101.0)))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "c1", str(path)]), 1)


class HarvestTest(unittest.TestCase):
    @staticmethod
    def lines(extra=()):
        recs = [{"kind": "header", "schema": fc.RESERVE_RECEIPT_SCHEMA, "decouple": True, "seed": 7},
                {"kind": "rank", "rank": 1, "resident_layers": 33}, {"kind": "rank", "rank": 0, "resident_layers": 33},
                {"kind": "prompt", "prompt": 0, "rows": 513}, {"kind": "ids", "prompt": 0, "ids": [1, 2]}, {"kind": "end", "complete": True}]
        return ["noise before\n"] + ["[t] EP_RESERVE_RECEIPT %s\n" % json.dumps(r) for r in recs + list(extra)]

    def test_assembles_in_rank_order(self):
        doc, seen = fc.reserve_assemble_receipt(self.lines())
        self.assertEqual(seen, 6)
        self.assertEqual([r["rank"] for r in doc["ranks"]], [0, 1])
        self.assertEqual(doc["token_ids"], [[1, 2]])
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["harvest_errors"], [])

    def test_missing_end_is_truncated_and_duplicates_flagged(self):
        doc, _ = fc.reserve_assemble_receipt(self.lines()[:-1])
        self.assertFalse(doc["complete"])
        doc, _ = fc.reserve_assemble_receipt(self.lines([{"kind": "rank", "rank": 0, "resident_layers": 33}]))
        self.assertTrue(any("duplicate rank" in e for e in doc["harvest_errors"]))

    @staticmethod
    def rank_lines(rank, extra=None, restore_ms=160.0, barrier=1.0):
        rec = {"kind": "prompt_rank", "rank": rank, "prompt": 0, "phase": "target", "ordinal": 0, "rows": 4396, "chunk_rows": [4096, 300],
               "banks": ["borrowed", "resident"],
               "restores": [{"before_chunk": 1, "bytes": 1 << 30, "restore_ms": restore_ms, "barrier_wait_ms": barrier + rank,
                             "s2_dma_tail_ms": 0.5 + rank}],
               "borrowed_epochs": 1, "elapsed_ms_incl_restore": 1000.0 + 10 * rank,
               "exchange_address_before": "0x%d" % rank, "exchange_address_after": "0x%d" % rank,
               "donor_hash_before": "h%d" % rank, "donor_hash_after": "h%d" % rank}
        rec.update(extra or {})
        return "EP_RESERVE_RECEIPT %s\n" % json.dumps(rec)

    def two_rank_header(self):
        return ["EP_RESERVE_RECEIPT %s\n" % json.dumps(r) for r in (
            {"kind": "header", "decouple": True, "prompt_chunk_rows": 4096, "resident_rows": 512},
            {"kind": "rank", "rank": 0, "source_generation": "5", "plan_digest": "00ff", "resident_layers": 33},
            {"kind": "header", "decouple": True, "prompt_chunk_rows": 4096, "resident_rows": 512},
            {"kind": "rank", "rank": 1, "source_generation": "5", "plan_digest": "00ff", "resident_layers": 33})]

    def test_prompt_rank_records_merge_per_prompt(self):
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(1, restore_ms=170.0), self.rank_lines(0)])
        self.assertEqual(doc["harvest_errors"], [])
        p = doc["prompts"][0]
        self.assertEqual(p["restores"], [{"before_chunk": 1, "bytes": 1 << 30, "restore_ms": 170.0, "barrier_wait_ms": [1.0, 2.0],
                                          "s2_dma_tail_ms": [0.5, 1.5]}])
        self.assertEqual(p["elapsed_ms_incl_restore"], 1010.0)
        self.assertEqual(p["exchange_address_before"], ["0x0", "0x1"])
        self.assertEqual(p["donor_hash_after"], ["h0", "h1"])
        self.assertEqual((p["rows"], p["chunk_rows"], p["banks"], p["borrowed_epochs"]), (4396, [4096, 300], ["borrowed", "resident"], 1))
        self.assertEqual(doc["prompt_chunk_rows"], 4096)

    @staticmethod
    def entry_lines(rank, prompt, phase, ordinal, extra=None):
        rec = {"kind": "prompt_rank", "rank": rank, "prompt": prompt, "phase": phase, "ordinal": ordinal, "rows": 5, "chunk_rows": [5],
               "banks": ["resident"], "restores": [], "borrowed_epochs": 0, "elapsed_ms_incl_restore": 2.0 + rank,
               "exchange_address_before": "0x%d" % rank, "exchange_address_after": "0x%d" % rank,
               "donor_hash_before": "h%d" % rank, "donor_hash_after": "h%d" % rank}
        rec.update(extra or {})
        return "EP_RESERVE_RECEIPT %s\n" % json.dumps(rec)

    def test_entries_are_keyed_by_prompt_phase_and_ordinal(self):
        rows = []
        for rank in (1, 0):  # arrival order is irrelevant
            rows += [self.entry_lines(rank, 0, "mtp_fill", 0), self.entry_lines(rank, 0, "target", 2), self.entry_lines(rank, 1, "mtp_fill", 0),
                     self.rank_lines(rank), self.entry_lines(rank, 0, "target", 1), self.entry_lines(rank, 1, "target", 0, {"rows": 8, "chunk_rows": [8]})]
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + rows)
        self.assertEqual(doc["harvest_errors"], [])
        self.assertEqual([p["prompt"] for p in doc["prompts"]], [0, 1])
        first, second = doc["prompts"]
        self.assertEqual((first["phase"], first["ordinal"], first["rows"]), ("target", 0, 4396))
        self.assertEqual([(e["phase"], e["ordinal"]) for e in first["entries"]], [("target", 1), ("target", 2), ("mtp_fill", 0)])
        self.assertEqual([(e["phase"], e["ordinal"]) for e in second["entries"]], [("mtp_fill", 0)])
        self.assertEqual(second["rows"], 8)
        self.assertEqual(first["entries"][2]["elapsed_ms_incl_restore"], 3.0)  # slowest rank
        self.assertEqual(first["entries"][2]["donor_hash_after"], ["h0", "h1"])
        self.assertNotIn("entries", first["entries"][0])  # entries are flat: only prompts[i] carries them

    def test_duplicate_key_and_bad_keys_are_harvest_errors(self):
        base = self.two_rank_header() + [self.rank_lines(0), self.rank_lines(1)]
        doc, _ = fc.reserve_assemble_receipt(base + [self.entry_lines(0, 0, "mtp_fill", 0), self.entry_lines(0, 0, "mtp_fill", 0), self.entry_lines(1, 0, "mtp_fill", 0)])
        self.assertTrue(any("duplicate prompt_rank record for prompt 0 phase mtp_fill ordinal 0 rank 0" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        # the same ordinal under another phase, or another ordinal under the same phase, is a different key
        doc, _ = fc.reserve_assemble_receipt(base + [self.entry_lines(r, 0, ph, o) for r in (0, 1) for ph, o in (("mtp_fill", 0), ("target", 1), ("mtp_fill", 1))])
        self.assertEqual(doc["harvest_errors"], [])
        self.assertEqual(len(doc["prompts"][0]["entries"]), 3)
        for bad in ({"phase": None}, {"phase": "decode"}, {"ordinal": None}, {"ordinal": -1}, {"ordinal": True}):
            doc, _ = fc.reserve_assemble_receipt(base + [self.entry_lines(0, 0, "target", 1, bad)])
            self.assertTrue(any("phase" in e and "ordinal" in e for e in doc["harvest_errors"]), (bad, doc["harvest_errors"]))
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.entry_lines(0, 0, "mtp_fill", 0), self.entry_lines(1, 0, "mtp_fill", 0)])
        self.assertTrue(any("no (target,0) entry" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        doc, _ = fc.reserve_assemble_receipt(base + [self.entry_lines(0, 0, "target", 1)])
        self.assertTrue(any("prompt 0 entry (target,1): no prompt_rank record from rank 1" in e for e in doc["harvest_errors"]), doc["harvest_errors"])

    def test_mtp_mode_builds_prompts_from_the_fill_entry(self):
        header = ["EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "header", "mode": "mtp", "seed": 1})]
        rows = []
        for rank in (0, 1):
            rows += [self.entry_lines(rank, 0, "mtp_fill", 0, {"rows": 4097, "chunk_rows": [4096, 1], "banks": ["resident", "resident"]}),
                     self.entry_lines(rank, 0, "target", 0, {"rows": 3, "chunk_rows": [3]}),
                     self.entry_lines(rank, 0, "target", 1)]
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + header + rows)
        self.assertEqual(doc["harvest_errors"], [])
        self.assertEqual(doc["mode"], "mtp")
        p = doc["prompts"][0]
        self.assertEqual((p["phase"], p["ordinal"], p["rows"], p["chunk_rows"]), ("mtp_fill", 0, 4097, [4096, 1]))
        self.assertEqual([(e["phase"], e["ordinal"]) for e in p["entries"]], [("target", 0), ("target", 1)])
        # no (mtp_fill,0) in mtp mode is a harvest error; a (target,0) alone does not stand in for it
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + header + [self.entry_lines(r, 0, "target", 0) for r in (0, 1)])
        self.assertTrue(any("no (mtp_fill,0) entry (mode mtp)" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        self.assertEqual(doc["prompts"], [])

    def test_ordinary_off_arm_log_roundtrips_through_harvest_and_check(self):
        lengths, tokens = (513, 4097), 4
        lines = ["EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "header", "binary_id": "bin-1", "model_sha256": "m", "seed": 7, "mode": "ar",
                                                          "max_seq": 16384})]
        for rank in (0, 1):  # the forward's ordinary-load records: real resident layers and chunk, no plan, no donor hashes
            lines.append("EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "header", "decouple": False, "prompt_chunk_rows": 4096}))
            lines.append("EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "rank", "rank": rank, "resident_layers": 28}))
        for i, n in enumerate(lengths):
            for rank in (0, 1):
                for ordinal, chunks in ((0, fc.reserve_chunk_plan(n)), (1, [1])):
                    lines.append("EP_RESERVE_RECEIPT %s\n" % json.dumps(
                        {"kind": "prompt_rank", "rank": rank, "prompt": i, "phase": "target", "ordinal": ordinal, "rows": sum(chunks),
                         "chunk_rows": chunks, "banks": [R] * len(chunks), "restores": [], "borrowed_epochs": 0,
                         "elapsed_ms_incl_restore": 12.5, "exchange_address_before": "0x7f00%d000" % rank,
                         "exchange_address_after": "0x7f00%d000" % rank}))
            lines.append("EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "ids", "prompt": i, "ids": ids_for(i, tokens)}))
        lines.append("EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "end", "complete": True}))
        doc, seen = fc.reserve_assemble_receipt(lines)
        self.assertEqual(doc["harvest_errors"], [])
        self.assertEqual(doc["decouple"], False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            off = write_arm(root, "off", False, lengths, tokens, 28)
            on = write_arm(root, "on", True, lengths, tokens, 33)
            write(off / "reserve-receipt.json", doc)
            v, _ = fc.compare_reserve_arms(str(off), str(on), min_tokens=tokens, require_lengths=lengths, expect_mode="ar")
            self.assertTrue(v.ok, v.mismatches)

    def test_rank_disagreement_and_missing_rank_are_harvest_errors(self):
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(0), self.rank_lines(1, {"banks": ["borrowed", "borrowed"]})])
        self.assertTrue(any("disagree on banks" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(0)])
        self.assertTrue(any("no prompt_rank record from rank 1" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(0), self.rank_lines(1, {"restores": []})])
        self.assertTrue(any("restore positions" in e for e in doc["harvest_errors"]), doc["harvest_errors"])
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(0), self.rank_lines(0), self.rank_lines(1)])
        self.assertTrue(any("duplicate prompt_rank" in e for e in doc["harvest_errors"]), doc["harvest_errors"])

    def test_conflicting_headers_and_absent_hash_rejected(self):
        lines = self.two_rank_header() + ["EP_RESERVE_RECEIPT %s\n" % json.dumps({"kind": "header", "prompt_chunk_rows": 2048})]
        doc, _ = fc.reserve_assemble_receipt(lines)
        self.assertTrue(any("header key prompt_chunk_rows disagrees" in e for e in doc["harvest_errors"]))
        no_hash = json.loads(self.rank_lines(1).split("EP_RESERVE_RECEIPT ")[1])
        for key in ("donor_hash_before", "donor_hash_after"):
            no_hash.pop(key)
        doc, _ = fc.reserve_assemble_receipt(self.two_rank_header() + [self.rank_lines(0), "EP_RESERVE_RECEIPT %s\n" % json.dumps(no_hash)])
        self.assertNotIn("donor_hash_before", doc["prompts"][0])  # the checker then rejects the prompt

    def test_cli_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stderr.txt"
            log.write_text("".join(self.lines()))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "harvest", str(log), tmp]), 0)
            first = (Path(tmp) / "reserve-receipt.json").read_text()
            log.write_text("")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "harvest", str(log), tmp]), 0)  # existing receipt kept
            self.assertEqual((Path(tmp) / "reserve-receipt.json").read_text(), first)
            (Path(tmp) / "reserve-receipt.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fc.run(["reserve", "harvest", str(log), tmp]), 1)  # no receipt lines at all


class DocsAndScriptTest(unittest.TestCase):
    def test_gate_script_parses_and_documents_reserve_mode(self):
        script = REPO_ROOT / "scripts" / "fn-ep-gate.sh"
        if shutil.which("bash") is None:
            self.skipTest("bash not available")
        self.assertEqual(subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode, 0)
        helped = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(helped.returncode, 0)
        for needle in ("--mode reserve", "HIPFIRE_EP_RESERVE_DECOUPLE", "ep2-off", "ep2-on", "--boundary-dir", "--min-resident"):
            self.assertIn(needle, helped.stdout)

    def test_docs_state_default_on_kill_switch_and_ordinary_fallback(self):
        multi = (REPO_ROOT / "docs" / "multi-gpu.md").read_text()
        env = (REPO_ROOT / "docs" / "env-vars.md").read_text()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
        self.assertIn("HIPFIRE_EP_RESERVE_DECOUPLE", env)
        self.assertIn("HIPFIRE_EP_RESERVE_DECOUPLE", multi)
        self.assertIn("HIPFIRE_EP_RESERVE_DECOUPLE", changelog)
        start = multi.index("### Flash-Next EP: EP reserve decoupling")
        section = multi[start:multi.index("\n### ", start + 10)]
        self.assertIn("ordinary", section)
        self.assertIn("kill switch", section)
        self.assertIn("HIPFIRE_EP_RESERVE_DECOUPLE=0", section)
        self.assertNotIn("not supported", section.lower())


if __name__ == "__main__":
    unittest.main()
