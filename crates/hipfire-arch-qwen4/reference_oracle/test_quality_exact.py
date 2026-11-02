# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the exact full-vocabulary KL method (no source checkpoint, no network).

Covers corpus manifest validation (masks, next-target checks, resets), the exact F64
streamed KL against independent computations and extremes, bf16 logits storage, route
agreement/recording, document bootstrap, stability gate, strata mixture, timing
projection, runtime limits, the layer-major engine contract (via stub operators) and the
export/compare provenance chain on synthetic exports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import quality
from schema import FixtureError

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the pure-numpy tests
    torch = None

VOCAB = 64


def _seq(seq_id: str, stratum: str, n: int, *, start: int = 1, positions=None, role=True) -> dict:
    tokens = [(start + 3 * i) % VOCAB for i in range(n)]
    item = {
        "id": seq_id,
        "stratum": stratum,
        "tokens": tokens,
        "score_positions": list(range(0, n - 1)) if positions is None else positions,
    }
    if stratum == "assistant" and role:
        item["role_mask"] = [0, 0] + [1] * (n - 2)
        if positions is None:
            item["score_positions"] = list(range(1, n - 1))
    return item


def _manifest(sequences=None) -> dict:
    if sequences is None:
        sequences = [
            _seq("t0", "text", 9, start=1),
            _seq("t1", "text", 7, start=5),
            _seq("a0", "assistant", 8, start=2),
            _seq("a1", "assistant", 10, start=9),
        ]
    return {"schema": quality.KL_CORPUS_SCHEMA, "name": "unit", "seed": 7, "sequences": sequences}


def _corpus(sequences=None) -> quality.KlCorpus:
    return quality.corpus_from_dict(_manifest(sequences), vocab_size=VOCAB)


def _logsoftmax(row) -> list[float]:
    row = [float(v) for v in row]
    peak = max(row)
    lse = peak + math.log(math.fsum(math.exp(v - peak) for v in row))
    return [v - lse for v in row]


class CorpusTests(unittest.TestCase):
    def test_valid_manifest_counts_and_score_index(self) -> None:
        corpus = _corpus()
        self.assertEqual(corpus.input_tokens, 9 + 7 + 8 + 10)
        self.assertEqual(corpus.scored_positions, 8 + 6 + 6 + 8)
        index = corpus.score_index()
        self.assertEqual(index.shape, (corpus.scored_positions, 3))
        for sequence_index, position, target in index.tolist():
            seq = corpus.sequences[sequence_index]
            self.assertEqual(target, seq.tokens[position + 1])
            self.assertLessEqual(position, seq.length - 2)  # last token never scored
        self.assertEqual(corpus.row_offsets(), [0, 8, 14, 20])

    def test_identity_is_deterministic_and_token_sensitive(self) -> None:
        first, second = _corpus().identity(), _corpus().identity()
        self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])
        changed = _manifest()
        changed["sequences"][0]["tokens"][3] = (changed["sequences"][0]["tokens"][3] + 1) % VOCAB
        self.assertNotEqual(quality.corpus_from_dict(changed, vocab_size=VOCAB).identity()["corpus_sha256"], first["corpus_sha256"])
        masked = _manifest()
        masked["sequences"][0]["score_positions"] = [0, 1, 2]
        self.assertNotEqual(quality.corpus_from_dict(masked, vocab_size=VOCAB).identity()["corpus_sha256"], first["corpus_sha256"])
        # meta/name edits do not change identity of tokens and masks
        meta = _manifest()
        meta["name"] = "renamed"
        meta["sequences"][1]["meta"] = {"license": "cc0"}
        self.assertEqual(quality.corpus_from_dict(meta, vocab_size=VOCAB).identity()["corpus_sha256"], first["corpus_sha256"])

    def test_last_position_has_no_target(self) -> None:
        bad = _manifest([_seq("t", "text", 6, positions=[0, 5])])
        with self.assertRaisesRegex(FixtureError, "last token has no next-token target"):
            quality.corpus_from_dict(bad, vocab_size=VOCAB)

    def test_mask_forms_agree(self) -> None:
        n = 8
        base = _seq("t", "text", n, positions=[1, 2, 5])
        as_mask = dict(base)
        del as_mask["score_positions"]
        as_mask["score_mask"] = [0, 1, 1, 0, 0, 1, 0, 0]
        as_range = dict(base)
        del as_range["score_positions"]
        as_range["score_range"] = [2, 5]
        positions = lambda spec: quality.corpus_from_dict(_manifest([spec]), vocab_size=VOCAB).sequences[0].score_positions  # noqa: E731
        self.assertEqual(positions(base), (1, 2, 5))
        self.assertEqual(positions(as_mask), (1, 2, 5))
        self.assertEqual(positions(as_range), (2, 3, 4))
        both = dict(base)
        both["score_range"] = [0, 2]
        with self.assertRaises(FixtureError):
            quality.corpus_from_dict(_manifest([both]), vocab_size=VOCAB)
        none = dict(base)
        del none["score_positions"]
        with self.assertRaises(FixtureError):
            quality.corpus_from_dict(_manifest([none]), vocab_size=VOCAB)

    def test_unsorted_empty_and_out_of_range_rejected(self) -> None:
        for positions in ([2, 1], [1, 1], [], [-1, 0], [0, 7]):
            with self.assertRaises(FixtureError, msg=str(positions)):
                quality.corpus_from_dict(_manifest([_seq("t", "text", 7, positions=positions)]), vocab_size=VOCAB)

    def test_assistant_requires_role_mask_and_assistant_targets(self) -> None:
        with self.assertRaisesRegex(FixtureError, "requires role_mask"):
            quality.corpus_from_dict(_manifest([_seq("a", "assistant", 8, role=False, positions=[2, 3])]), vocab_size=VOCAB)
        item = _seq("a", "assistant", 8)
        item["score_positions"] = [0, 2]  # target at position 1 is a prompt token
        with self.assertRaisesRegex(FixtureError, "not assistant content"):
            quality.corpus_from_dict(_manifest([item]), vocab_size=VOCAB)

    def test_declared_targets_must_match_next_tokens(self) -> None:
        item = _seq("t", "text", 6, positions=[0, 2])
        item["targets"] = [item["tokens"][1], item["tokens"][3]]
        quality.corpus_from_dict(_manifest([item]), vocab_size=VOCAB)
        item["targets"] = [item["tokens"][1], item["tokens"][2]]
        with self.assertRaisesRegex(FixtureError, "declared targets"):
            quality.corpus_from_dict(_manifest([item]), vocab_size=VOCAB)

    def test_reset_semantics_only_independent_sequences(self) -> None:
        item = _seq("t", "text", 6)
        item["reset"] = False
        with self.assertRaisesRegex(FixtureError, "independent sequences"):
            quality.corpus_from_dict(_manifest([item]), vocab_size=VOCAB)
        item["reset"] = True
        quality.corpus_from_dict(_manifest([item]), vocab_size=VOCAB)

    def test_ids_tokens_and_schema_validation(self) -> None:
        with self.assertRaisesRegex(FixtureError, "unique"):
            quality.corpus_from_dict(_manifest([_seq("x", "text", 5), _seq("x", "text", 5)]), vocab_size=VOCAB)
        bad = _seq("t", "text", 5)
        bad["tokens"][2] = VOCAB
        with self.assertRaisesRegex(FixtureError, "outside"):
            quality.corpus_from_dict(_manifest([bad]), vocab_size=VOCAB)
        bad["tokens"][2] = True
        with self.assertRaises(FixtureError):
            quality.corpus_from_dict(_manifest([bad]), vocab_size=VOCAB)
        with self.assertRaises(FixtureError):
            quality.corpus_from_dict({**_manifest(), "schema": "other"}, vocab_size=VOCAB)
        with self.assertRaises(FixtureError):
            quality.corpus_from_dict(_manifest([{**_seq("t", "text", 5), "stratum": "code"}]), vocab_size=VOCAB)
        with self.assertRaisesRegex(FixtureError, "at least two tokens"):
            quality.corpus_from_dict(_manifest([_seq("t", "text", 1, positions=[0])]), vocab_size=VOCAB)

    def test_token_payload_with_offsets_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = np.arange(20, dtype="<u4") % VOCAB
            raw = tokens.tobytes()
            (root / "tokens.u32").write_bytes(raw)
            manifest = {
                "schema": quality.KL_CORPUS_SCHEMA,
                "token_payload": {"path": "tokens.u32", "sha256": hashlib.sha256(raw).hexdigest()},
                "sequences": [
                    {"id": "p0", "stratum": "text", "offset": 0, "length": 8, "score_range": [0, 7]},
                    {"id": "p1", "stratum": "text", "offset": 8, "length": 12, "score_range": [2, 11]},
                ],
            }
            (root / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
            corpus = quality.load_kl_corpus(root / "corpus.json", vocab_size=VOCAB)
            self.assertEqual(corpus.sequences[1].tokens, tuple(range(8, 20)))
            manifest["token_payload"]["sha256"] = "0" * 64
            (root / "bad.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(FixtureError, "sha256 mismatch"):
                quality.load_kl_corpus(root / "bad.json", vocab_size=VOCAB)
            manifest["token_payload"] = {"path": "../tokens.u32", "sha256": "0" * 64}
            (root / "esc.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(FixtureError):
                quality.load_kl_corpus(root / "esc.json", vocab_size=VOCAB)

    def test_canonical_corpus_wraps_one_sequence(self) -> None:
        corpus = quality.canonical_kl_corpus(list(range(17)))
        self.assertEqual(len(corpus.sequences), 1)
        self.assertEqual(corpus.scored_positions, 16)
        self.assertEqual(corpus.score_index()[-1].tolist(), [0, 15, 16])

    def test_plan_passes_grouping(self) -> None:
        corpus = _corpus()
        self.assertEqual(quality.plan_passes(corpus, None), [[0, 1, 2, 3]])
        self.assertEqual(quality.plan_passes(corpus, 0), [[0, 1, 2, 3]])
        self.assertEqual(quality.plan_passes(corpus, 16), [[0, 1], [2], [3]])
        self.assertEqual(quality.plan_passes(corpus, 5), [[0], [1], [2], [3]])  # oversize sequences get own pass


class ExactKlTests(unittest.TestCase):
    def test_toy_distribution_matches_independent_computation(self) -> None:
        p = [0.5, 0.3, 0.2]
        q = [0.4, 0.4, 0.2]
        teacher = np.log(np.asarray([p]), dtype=np.float64)
        candidate = np.log(np.asarray([q]), dtype=np.float64)
        # Use float64 logits through the float32 path only after exact conversion below.
        t32, c32 = teacher.astype(np.float32), candidate.astype(np.float32)
        result = quality.exact_kl_streamed(t32, c32, [0], block_vocab=2)
        lt, lc = _logsoftmax(t32[0]), _logsoftmax(c32[0])
        expected = math.fsum(math.exp(a) * (a - b) for a, b in zip(lt, lc))
        self.assertAlmostEqual(float(result["kl"][0]), expected, delta=1e-10)
        self.assertAlmostEqual(float(result["nll_teacher"][0]), -lt[0], delta=1e-10)
        self.assertAlmostEqual(float(result["nll_candidate"][0]), -lc[0], delta=1e-10)
        analytic = math.fsum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        self.assertAlmostEqual(float(result["kl"][0]), analytic, delta=1e-6)  # float32 logits rounding only

    def test_identical_logits_are_exactly_zero(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.normal(0.0, 5.0, size=(6, 50)).astype(np.float32)
        result = quality.exact_kl_streamed(logits, logits.copy(), rng.integers(0, 50, 6), block_rows=4, block_vocab=7)
        self.assertLessEqual(float(np.abs(result["kl"]).max()), 1e-10)
        shifted = logits + np.float32(1024.0)  # shift invariance of softmax (exactly representable offset)
        again = quality.exact_kl_streamed(logits, shifted, [0] * 6)
        self.assertLessEqual(float(np.abs(again["kl"]).max()), 1e-6)

    def test_streamed_matches_materialized_across_block_sizes(self) -> None:
        rng = np.random.default_rng(1)
        teacher = rng.normal(0.0, 4.0, size=(11, 83)).astype(np.float32)
        candidate = (teacher + rng.normal(0.0, 0.7, size=teacher.shape)).astype(np.float32)
        reference = quality.exact_kl_materialized(teacher, candidate)
        targets = rng.integers(0, 83, 11)
        for rows, vocab in ((1, 1), (3, 7), (11, 83), (4, 1000), (64, 16384)):
            result = quality.exact_kl_streamed(teacher, candidate, targets, block_rows=rows, block_vocab=vocab)
            self.assertLessEqual(float(np.abs(result["kl"] - reference).max()), 1e-8, (rows, vocab))
        self.assertGreater(float(reference.min()), 0.0)

    def test_nll_and_top1_match_independent_recomputation(self) -> None:
        rng = np.random.default_rng(2)
        teacher = rng.normal(size=(5, 30)).astype(np.float32)
        candidate = rng.normal(size=(5, 30)).astype(np.float32)
        targets = [3, 0, 29, 14, 7]
        result = quality.exact_kl_streamed(teacher, candidate, targets, block_vocab=4)
        for row, target in enumerate(targets):
            self.assertAlmostEqual(float(result["nll_teacher"][row]), -_logsoftmax(teacher[row])[target], delta=1e-10)
            self.assertAlmostEqual(float(result["nll_candidate"][row]), -_logsoftmax(candidate[row])[target], delta=1e-10)
            self.assertEqual(int(result["top1_teacher"][row]), int(np.argmax(teacher[row])))
            self.assertEqual(int(result["top1_candidate"][row]), int(np.argmax(candidate[row])))

    def test_top1_ties_resolve_to_lowest_index_across_blocks(self) -> None:
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, [2, 7]] = 3.0
        result = quality.exact_kl_streamed(logits, logits, [0], block_vocab=3)
        self.assertEqual(int(result["top1_teacher"][0]), 2)

    def test_extreme_logits_remain_finite_and_correct(self) -> None:
        teacher = np.asarray([[1e4, 0.0, -1e4]], dtype=np.float32)
        candidate = np.asarray([[-1e4, 1e4, 0.0]], dtype=np.float32)
        result = quality.exact_kl_streamed(teacher, candidate, [0], block_vocab=1)
        self.assertTrue(math.isfinite(float(result["kl"][0])))
        self.assertAlmostEqual(float(result["kl"][0]), 2e4, delta=1e-6)
        self.assertAlmostEqual(float(result["nll_teacher"][0]), 0.0, delta=1e-9)
        huge = np.asarray([[3e4, 3e4 - 1.0, -3e4, 0.0]], dtype=np.float32)
        same = quality.exact_kl_streamed(huge, huge, [1])
        self.assertEqual(float(same["kl"][0]), 0.0)
        self.assertTrue(math.isfinite(float(same["nll_teacher"][0])))
        # tail beyond float64 underflow still contributes zero, not NaN
        tail = np.asarray([[0.0, -2e4, -2e4]], dtype=np.float32)
        out = quality.exact_kl_streamed(tail, np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32), [0])
        self.assertTrue(math.isfinite(float(out["kl"][0])))

    def test_nonfinite_and_shape_errors_fail_loudly(self) -> None:
        good = np.zeros((2, 5), dtype=np.float32)
        for bad_value in (np.nan, np.inf, -np.inf):
            bad = good.copy()
            bad[1, 3] = bad_value
            with self.assertRaisesRegex(FixtureError, "nonfinite"):
                quality.exact_kl_streamed(good, bad, [0, 1])
        with self.assertRaises(FixtureError):
            quality.exact_kl_streamed(good, np.zeros((2, 6), dtype=np.float32), [0, 1])
        with self.assertRaises(FixtureError):
            quality.exact_kl_streamed(good, good, [0])
        with self.assertRaises(FixtureError):
            quality.exact_kl_streamed(good, good, [0, 5])

    def test_materially_negative_kl_raises_and_tiny_negative_stays_raw(self) -> None:
        logits = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
        self.assertEqual(float(quality.exact_kl_streamed(logits, logits, [0])["kl"][0]), 0.0)
        original = quality.NEGATIVE_KL_TOLERANCE
        try:
            # A tolerance of -1 makes any KL < 1 "materially negative": the guard must raise,
            # never clip the value to zero.
            quality.NEGATIVE_KL_TOLERANCE = -1.0
            with self.assertRaisesRegex(FixtureError, "materially negative"):
                quality.exact_kl_streamed(logits, logits, [0])
        finally:
            quality.NEGATIVE_KL_TOLERANCE = original
        self.assertEqual(original, 1e-9)

    def test_bf16_file_storage_decodes_exactly_and_hashes(self) -> None:
        rng = np.random.default_rng(3)
        f32 = rng.normal(0.0, 3.0, size=(5, 17)).astype(np.float32)
        words = (f32.view(np.uint32) >> 16).astype("<u2")
        widened = (words.astype(np.uint32) << 16).view(np.float32)  # bf16-representable values
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logits.bf16le"
            path.write_bytes(words.tobytes())
            digest = hashlib.sha256(words.tobytes()).hexdigest()
            reader = quality._FileRows(path, 5, 17, digest, "bf16")
            np.testing.assert_array_equal(reader.block(0, 3), widened[:3])
            np.testing.assert_array_equal(reader.block(3, 5), widened[3:])
            self.assertEqual(reader.finish(), digest)
            with self.assertRaises(FixtureError):
                quality._FileRows(path, 5, 17, digest, "f32")  # wrong storage -> wrong size
            wrong = quality._FileRows(path, 5, 17, "0" * 64, "bf16")
            wrong.block(0, 5)
            with self.assertRaisesRegex(FixtureError, "sha256"):
                wrong.finish()
            skip = quality._FileRows(path, 5, 17, None, "bf16")
            with self.assertRaisesRegex(FixtureError, "sequentially"):
                skip.block(2, 4)
        # bf16 storage gives the same KL as the widened float32 matrix
        a = quality.exact_kl_streamed(widened, widened[::-1].copy(), [0] * 5)
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = Path(directory) / "r", Path(directory) / "c"
            ref.write_bytes(words.tobytes())
            cand.write_bytes(words[::-1].copy().tobytes())
            b = quality.exact_kl_streamed(quality._FileRows(ref, 5, 17, None, "bf16"), quality._FileRows(cand, 5, 17, None, "bf16"), [0] * 5)
        np.testing.assert_array_equal(a["kl"], b["kl"])


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_independent_per_stratum_and_paired_across_columns(self) -> None:
        rng = np.random.default_rng(4)
        sums = rng.normal(size=(12, 3))
        counts = rng.integers(3, 9, 12).astype(np.float64)
        a = quality.doc_bootstrap(sums, counts, 200, 5, "text")
        b = quality.doc_bootstrap(sums, counts, 200, 5, "text")
        c = quality.doc_bootstrap(sums, counts, 200, 5, "assistant")
        d = quality.doc_bootstrap(sums, counts, 200, 6, "text")
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a, d))
        # columns share document draws: duplicating a column gives identical replicates
        dup = quality.doc_bootstrap(np.stack([sums[:, 0], sums[:, 0]], axis=1), counts, 200, 5, "text")
        np.testing.assert_array_equal(dup[:, 0], dup[:, 1])
        self.assertAlmostEqual(float(a[:, 0].mean()), float(sums[:, 0].sum() / counts.sum()), delta=0.2)
        with self.assertRaises(FixtureError):
            quality.doc_bootstrap(np.zeros((0, 1)), np.zeros(0), 10, 0, "x")

    def test_bootstrap_resamples_documents_not_tokens(self) -> None:
        # one document with 1000 rows and one with 1: document resampling gives a wide CI even
        # though token-level resampling would look tight.
        sums = np.asarray([[0.0], [1.0]])
        counts = np.asarray([1000.0, 1.0])
        reps = quality.doc_bootstrap(sums, counts, 500, 1, "x")
        self.assertGreater(float(reps.max()) - float(reps.min()), 0.4)

    def test_weighted_quantile(self) -> None:
        values = np.asarray([4.0, 1.0, 3.0, 2.0])
        weights = np.ones(4)
        self.assertEqual(quality.weighted_quantile(values, weights, 0.25), 1.0)
        self.assertEqual(quality.weighted_quantile(values, weights, 0.5), 2.0)
        self.assertEqual(quality.weighted_quantile(values, weights, 0.99), 4.0)
        self.assertEqual(quality.weighted_quantile(values, np.asarray([1.0, 1.0, 1.0, 97.0]), 0.5), 2.0)

    def test_stability_gate_thresholds(self) -> None:
        self.assertTrue(quality.stability_gate(0.0077, 0.0020)["stable"])
        self.assertFalse(quality.stability_gate(0.0077, 0.0021)["stable"])
        low = quality.stability_gate(0.019, 0.0015)  # below mean floor: only the absolute limit applies
        self.assertFalse(low["relative_applies"])
        self.assertTrue(low["stable"])
        self.assertFalse(quality.stability_gate(0.02, 0.0015)["stable"])  # 5% of 0.02 is 0.001
        self.assertTrue(quality.stability_gate(0.06, 0.0020)["relative_ok"])
        high = quality.stability_gate(0.06, 0.0031)
        self.assertFalse(high["absolute_ok"])
        self.assertFalse(high["stable"])

    def _rows(self, per_doc):
        """per_doc: list of (stratum, length, [values]); returns (columns, seq_index, sequences)."""
        values, seq_index, sequences = [], [], []
        for index, (stratum, length, data) in enumerate(per_doc):
            sequences.append({"stratum": stratum, "length": length, "half": None})
            values.extend(data)
            seq_index.extend([index] * len(data))
        return {"kl": np.asarray(values)}, np.asarray(seq_index), sequences

    def test_summary_mixture_is_equal_weight_and_excludes_unscored_docs(self) -> None:
        docs = [("text", 100, [0.01] * 10)] * 4 + [("assistant", 100, [0.03] * 30)] * 4
        docs.append(("assistant", 100, []))  # a context-only document contributes nothing
        columns, seq_index, sequences = self._rows(docs)
        summary = quality.summarize_columns(columns, seq_index, sequences, bootstrap=200, seed=1)
        self.assertEqual(summary["text"]["n_documents"], 4)
        self.assertEqual(summary["assistant"]["n_documents"], 4)
        self.assertEqual(summary["text"]["n_positions"], 40)
        self.assertEqual(summary["assistant"]["n_positions"], 120)
        self.assertAlmostEqual(summary["mixture"]["metrics"]["kl"]["mean"], 0.02, places=12)  # not 0.025 (position-weighted)
        for got, want in zip(summary["text"]["metrics"]["kl"]["ci95"], (0.01, 0.01)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertTrue(summary["mixture"]["stability"]["stable"])
        self.assertEqual(summary["mixture"]["quantiles"]["p50"], 0.01)
        self.assertEqual(summary["mixture"]["quantiles"]["p99"], 0.03)

    def test_summary_flags_unstable_pilot_and_reports_strata_and_halves(self) -> None:
        docs = [("text", 100 if i % 2 else 600, [0.0 if i % 2 else 1.0] * 5) for i in range(8)]
        docs += [("assistant", 3000, [0.5] * 5) for _ in range(4)]
        columns, seq_index, sequences = self._rows(docs)
        summary = quality.summarize_columns(columns, seq_index, sequences, bootstrap=300, seed=2)
        self.assertFalse(summary["text"]["stability"]["stable"])
        self.assertGreater(summary["text"]["metrics"]["kl"]["ci95"][1], summary["text"]["metrics"]["kl"]["ci95"][0])
        self.assertEqual(set(summary["text"]["context_length_strata"]), {"len_<=128", "len_513-2048"})
        self.assertEqual(summary["assistant"]["context_length_strata"]["len_2049-8192"]["n_documents"], 4)
        self.assertIsNotNone(summary["text"]["split_half"])
        self.assertIn("agree", summary["text"]["split_half"])

    def test_summary_ppl_and_paired_columns(self) -> None:
        docs = [("text", 10, [1.0, 2.0])] * 3 + [("assistant", 10, [2.0, 4.0])] * 3
        columns, seq_index, sequences = self._rows(docs)
        columns["nll_teacher"] = columns["kl"]
        summary = quality.summarize_columns(columns, seq_index, sequences, bootstrap=100, seed=1, ppl_pairs={"teacher_ppl": "nll_teacher"})
        self.assertAlmostEqual(summary["text"]["metrics"]["teacher_ppl"]["mean"], math.exp(1.5), places=12)

    def test_route_agreement(self) -> None:
        a = np.zeros((2, 3, 4), dtype=np.uint16)
        a[:] = np.arange(4)
        b = a.copy()
        b[0, 1] = [3, 2, 1, 0]  # same set, different order
        b[1, 2] = [9, 8, 1, 0]  # half overlap
        result = quality.route_agreement(a, b)
        np.testing.assert_allclose(result["exact"], [1.0, 0.5, 0.5])
        np.testing.assert_allclose(result["overlap"], [1.0, 1.0, 0.75])
        with self.assertRaises(FixtureError):
            quality.route_agreement(a, b[:, :2])


class RuntimeTests(unittest.TestCase):
    def test_cpu_limits_reject_oversubscription_before_side_effects(self) -> None:
        for kwargs in ({"threads": 17}, {"threads": 0}, {"memory_gib": 65.0}, {"memory_gib": 0.0}):
            with self.assertRaises(FixtureError, msg=str(kwargs)):
                quality.apply_runtime_limits("cpu", **kwargs)

    def test_project_runtime_arithmetic(self) -> None:
        timing = {
            "passes": 1, "variants": 2, "input_tokens": 1000, "scored_rows": 200,
            "layers_io_seconds": 10.0, "layers_compute_seconds": 100.0, "head_io_seconds": 2.0,
            "head_compute_seconds": 20.0, "embedding_seconds": 5.0,
        }
        projection = quality.project_runtime(timing)
        self.assertAlmostEqual(projection["estimated_seconds_one_variant"], 12.0 + 0.0525 * 327_680 + 0.05 * 65_536, places=6)
        self.assertAlmostEqual(projection["estimated_seconds_lockstep_variants_run"], 12.0 + 2 * (0.0525 * 327_680 + 0.05 * 65_536), places=6)
        self.assertEqual(projection["full_passes"], 1)
        multi = quality.project_runtime(timing, pass_tokens=8192)
        self.assertEqual(multi["full_passes"], 40)
        self.assertGreater(multi["estimated_seconds_one_variant"], projection["estimated_seconds_one_variant"])

    def test_cli_rejects_bad_mode_combinations(self) -> None:
        self.assertEqual(quality.main([]), 2)
        self.assertEqual(quality.main(["--exact-export"]), 2)  # no --out-dir
        self.assertEqual(quality.main(["--exact-compare", "--exact-export"]), 2)
        self.assertEqual(quality.main(["--exact-compare"]), 2)

    def test_export_rejects_bad_orders_and_storage_early(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FixtureError):
                quality.export_exact(corpus, directory, orders=("native", "native"))
            with self.assertRaises(FixtureError):
                quality.export_exact(corpus, directory, orders=("reverse",))
            with self.assertRaises(FixtureError):
                quality.export_exact(corpus, directory, logits_storage="f16")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_export(root: Path, corpus: quality.KlCorpus, logits, *, storage: str = "f32", routes=None, order: str = "native", truncated: bool = False) -> Path:
    """Write a synthetic export directory with the production file layout and identity block."""
    root.mkdir(parents=True, exist_ok=True)
    index = corpus.score_index()
    (root / "score_index.u32le").write_bytes(index.astype("<u4").tobytes())
    logits = np.asarray(logits, dtype=np.float32)
    files = {"score_index": {"path": "score_index.u32le", "dtype": "uint32", "shape": list(index.shape)}}
    if storage == "f32":
        (root / "logits.f32le").write_bytes(logits.astype("<f4").tobytes())
        files["logits"] = {"path": "logits.f32le", "dtype": "float32", "shape": list(logits.shape)}
    else:
        words = (logits.view(np.uint32) >> 16).astype("<u2")
        (root / "logits.bf16le").write_bytes(words.tobytes())
        files["logits"] = {"path": "logits.bf16le", "dtype": "bfloat16", "shape": list(logits.shape)}
    if routes is not None:
        (root / "route_ids.u16le").write_bytes(np.asarray(routes, dtype="<u2").tobytes())
        files["route_ids"] = {"path": "route_ids.u16le", "dtype": "uint16", "shape": list(routes.shape)}
    for entry in files.values():
        entry["sha256"] = _sha(root / entry["path"])
    identity = {
        "schema": quality.EXACT_EXPORT_SCHEMA,
        "corpus": corpus.identity(),
        "reduction_order": order,
        "vocab_size": int(logits.shape[1]),
        "scored_rows": int(logits.shape[0]),
        "layers_executed": 2,
        "truncated_layers": truncated,
        "files": files,
    }
    manifest = {"identity": identity, "identity_sha256": hashlib.sha256(quality._canonical_json(identity)).hexdigest(), "runtime": {}}
    (root / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = _corpus()
        rng = np.random.default_rng(10)
        rows = self.corpus.scored_positions
        self.teacher = rng.normal(0.0, 3.0, size=(rows, VOCAB)).astype(np.float32)
        self.candidate = (self.teacher + rng.normal(0.0, 0.5, size=self.teacher.shape)).astype(np.float32)
        total = self.corpus.input_tokens
        base = np.arange(3, dtype=np.uint16)
        self.routes = np.broadcast_to(base, (2, total, 3)).copy()

    def test_self_comparison_is_zero_with_full_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = make_export(root / "a", self.corpus, self.teacher, routes=self.routes)
            report = quality.exact_compare(export, export, root / "self.json", bootstrap=200, seed=3)
            results = report["results"]
            self.assertLessEqual(abs(results["mixture"]["metrics"]["kl"]["mean"]), 1e-10)
            self.assertEqual(results["text"]["metrics"]["top1_agreement"]["mean"], 1.0)
            self.assertEqual(report["routing"]["all_positions"]["exact_ordered"], 1.0)
            self.assertEqual(results["text"]["n_positions"], 14)
            self.assertEqual(results["assistant"]["n_positions"], 14)
            self.assertEqual(report["scored_rows"], self.corpus.scored_positions)
            self.assertTrue((root / "self.positions.npz").is_file())

    def test_candidate_comparison_matches_independent_kl_and_persists_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = make_export(root / "ref", self.corpus, self.teacher, routes=self.routes)
            cand_routes = self.routes.copy()
            cand_routes[0, 3] = [2, 1, 0]
            cand = make_export(root / "cand", self.corpus, self.candidate, order="reverse-experts", routes=cand_routes)
            report = quality.exact_compare(ref, cand, root / "r.json", bootstrap=300, seed=4)
            saved = np.load(root / "r.positions.npz")
            for row in (0, 5, 17, 27):
                lt, lc = _logsoftmax(self.teacher[row]), _logsoftmax(self.candidate[row])
                expected = math.fsum(math.exp(a) * (a - b) for a, b in zip(lt, lc))
                self.assertAlmostEqual(float(saved["kl"][row]), expected, delta=1e-10)
            self.assertEqual(saved["kl"].shape[0], self.corpus.scored_positions)
            self.assertGreater(report["results"]["mixture"]["metrics"]["kl"]["mean"], 0.0)
            self.assertLess(report["routing"]["all_positions"]["exact_ordered"], 1.0)
            self.assertEqual(report["reference"]["reduction_order"], "native")
            self.assertEqual(report["candidate"]["reduction_order"], "reverse-experts")
            self.assertIn("not_additive", report)
            # mixture is the equal-weight combination of the two strata
            mean = lambda k: report["results"][k]["metrics"]["kl"]["mean"]  # noqa: E731
            self.assertAlmostEqual(mean("mixture"), 0.5 * (mean("text") + mean("assistant")), places=12)

    def test_bf16_and_f32_storage_agree(self) -> None:
        widened = (self.teacher.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
        cand = (self.candidate.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f32_ref, f32_cand = make_export(root / "r32", self.corpus, widened), make_export(root / "c32", self.corpus, cand)
            bf_ref, bf_cand = make_export(root / "rbf", self.corpus, widened, storage="bf16"), make_export(root / "cbf", self.corpus, cand, storage="bf16")
            a = quality.exact_compare(f32_ref, f32_cand, root / "a.json", bootstrap=100, seed=1)
            b = quality.exact_compare(bf_ref, bf_cand, root / "b.json", bootstrap=100, seed=1)
            self.assertEqual(a["results"], b["results"])
            self.assertEqual((root / "rbf" / "logits.bf16le").stat().st_size * 2, (root / "r32" / "logits.f32le").stat().st_size)
            mixed = quality.exact_compare(f32_ref, bf_cand, root / "m.json", bootstrap=100, seed=1)
            self.assertEqual(a["results"], mixed["results"])

    def test_provenance_mismatches_and_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = make_export(root / "ref", self.corpus, self.teacher)
            other_corpus = _manifest()
            other_corpus["sequences"][0]["tokens"][1] = (other_corpus["sequences"][0]["tokens"][1] + 1) % VOCAB
            other = quality.corpus_from_dict(other_corpus, vocab_size=VOCAB)
            bad_corpus = make_export(root / "other", other, self.candidate)
            with self.assertRaisesRegex(FixtureError, "different corpora"):
                quality.exact_compare(ref, bad_corpus, root / "x.json", bootstrap=100)
            # truncated exports are timing probes: never comparable, in either role
            truncated = make_export(root / "tr", self.corpus, self.candidate, truncated=True)
            with self.assertRaisesRegex(FixtureError, "truncated-layer"):
                quality.exact_compare(ref, truncated, root / "tr.json", bootstrap=100)
            with self.assertRaisesRegex(FixtureError, "truncated-layer"):
                quality.exact_compare(truncated, ref, root / "tr2.json", bootstrap=100)
            self.assertFalse((root / "tr.json").exists())
            manifest = json.loads((root / "ref" / "export.json").read_text())
            manifest["identity"]["reduction_order"] = "reverse-experts"
            (root / "ref" / "export.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FixtureError, "identity hash"):
                quality.exact_compare(ref, truncated, root / "t.json", bootstrap=100)
            # logits corruption is caught by the streamed hash
            good = make_export(root / "good", self.corpus, self.teacher)
            path = root / "good" / "logits.f32le"
            data = bytearray(path.read_bytes())
            data[10] ^= 0x01
            path.write_bytes(bytes(data))
            healthy = make_export(root / "healthy", self.corpus, self.candidate)
            with self.assertRaisesRegex(FixtureError, "sha256"):
                quality.exact_compare(good, healthy, root / "g.json", bootstrap=100)
            with self.assertRaises(FixtureError):
                quality.exact_compare(ref, truncated, root / "low.json", bootstrap=10)

    def test_paired_difference_uses_joint_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = make_export(root / "ref", self.corpus, self.teacher)
            near = make_export(root / "near", self.corpus, (self.teacher + 0.05).astype(np.float32) * np.float32(1.01))
            far = make_export(root / "far", self.corpus, self.candidate)
            quality.exact_compare(ref, near, root / "near.json", bootstrap=100, seed=1)
            quality.exact_compare(ref, far, root / "far.json", bootstrap=100, seed=1)
            paired = quality.paired_difference(root / "far.positions.npz", root / "near.positions.npz", ref, root / "p.json", bootstrap=200, seed=1)
            diff = paired["results"]["mixture"]["metrics"]["kl_diff"]["mean"]
            a = paired["results"]["mixture"]["metrics"]["kl_a"]["mean"]
            b = paired["results"]["mixture"]["metrics"]["kl_b"]["mean"]
            self.assertAlmostEqual(diff, a - b, places=12)


class RouteRecorderTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch unavailable")
    def test_records_all_positions_in_layer_major_tape_and_rejects_bad_routes(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = quality._RouteRecorder(root, corpus, layers=2, experts=16)
            expected = {}
            for sequence in corpus.sequences:
                recorder.current = sequence.index
                for layer in range(2):
                    ids = torch.stack([torch.randperm(16)[:3] for _ in range(sequence.length)])
                    scores = torch.rand(sequence.length, 3)
                    expected[(sequence.index, layer)] = (ids.numpy().astype(np.uint16), scores.numpy())
                    recorder.sink(layer, torch.zeros(sequence.length, 16), scores, ids)
            recorder.verify(range(4), 2)
            recorder.close()
            total = corpus.input_tokens
            ids_tape = np.fromfile(root / "route_ids.u16le", dtype="<u2").reshape(2, total, 3)
            scores_tape = np.fromfile(root / "route_scores.f32le", dtype="<f4").reshape(2, total, 3)
            for (seq_index, layer), (ids, scores) in expected.items():
                start = corpus.offsets[seq_index]
                np.testing.assert_array_equal(ids_tape[layer, start : start + len(ids)], ids)
                np.testing.assert_array_equal(scores_tape[layer, start : start + len(ids)], scores)
            self.assertEqual(recorder.k, 3)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_sink_validation(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as directory:
            recorder = quality._RouteRecorder(Path(directory), corpus, layers=2, experts=16)
            length = corpus.sequences[0].length
            ids = torch.stack([torch.arange(3) for _ in range(length)])
            scores = torch.ones(length, 3)
            with self.assertRaisesRegex(FixtureError, "outside a sequence"):
                recorder.sink(0, None, scores, ids)
            recorder.current = 0
            with self.assertRaisesRegex(FixtureError, "do not match sequence"):
                recorder.sink(0, None, scores[:-1], ids[:-1])
            with self.assertRaisesRegex(FixtureError, "out of range"):
                recorder.sink(0, None, scores, ids + 100)
            with self.assertRaisesRegex(FixtureError, "duplicate"):
                recorder.sink(0, None, scores, torch.zeros_like(ids))
            with self.assertRaisesRegex(FixtureError, "nonfinite"):
                recorder.sink(0, None, scores * float("nan"), ids)
            with self.assertRaisesRegex(FixtureError, "layer"):
                recorder.sink(5, None, scores, ids)
            recorder.sink(0, None, scores, ids)
            with self.assertRaisesRegex(FixtureError, "fired 2 times"):
                recorder.sink(0, None, scores, ids)
            with self.assertRaisesRegex(FixtureError, "incomplete"):
                recorder.verify([0], 2)
            recorder.close()


class _StubStore:
    """Counts layer weight reads; embeddings are `[token, token+1, token+2]`."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.io_seconds = 0.0
        self.current_bytes = 0
        self.peak_bytes = 0
        self.layer_reads: list[dict[str, int]] = []
        self.counters = {"expert_hits": 0, "expert_misses": 0}
        self._cache: dict[str, object] = {}

    def begin_layer(self) -> None:
        self._cache = {}
        self.layer_reads.append({})

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    def full(self, name: str):
        if name not in self._cache:
            self.layer_reads[-1][name] = self.layer_reads[-1].get(name, 0) + 1
            self._cache[name] = torch.zeros(1)
        return self._cache[name]

    def expert_rows(self, *args):
        raise AssertionError("stub operators never read experts")

    def gather(self, name: str, rows):
        return torch.tensor([[float(r), float(r) + 1, float(r) + 2] for r in rows])


class _StubOps:
    """Pure per-sequence operators; records every call so ordering/reset can be asserted."""

    def __init__(self, label: str, calls: list) -> None:
        self.label, self.calls = label, calls

    def provenance(self) -> dict:
        return {"label": self.label}

    def hyperconnection(self, hidden, weights, cfg, *, combine):
        return (hidden, hidden, torch.zeros_like(hidden)) if combine else hidden

    def gdn(self, hidden, weights, cfg, layer):
        self.calls.append(("gdn", self.label, layer, int(hidden.shape[0])))
        return hidden * 0.5, {"recurrent": hidden, "conv": hidden}

    def qsa(self, hidden, weights, cfg, layer):
        return hidden, hidden

    def ple(self, hidden, tokens, cfg, metadata, load_full, read_rows):
        return torch.zeros_like(hidden)

    def inject(self, hyper_input, block_output, injection):
        return hyper_input + block_output

    def moe(self, hidden, cfg, load_full, load_expert, layer):
        return hidden * 0.0 + float(layer + 1)


class _StubCheckpoint:
    weight_map: dict = {}

    def provenance(self) -> dict:
        return {"stub": True, "read_bytes": 0}


@unittest.skipIf(torch is None, "torch unavailable")
class EngineContractTests(unittest.TestCase):
    CFG = {"num_hidden_layers": 2, "vocab_size": 16, "hc_count": 2, "layer_types": ["linear_attention", "linear_attention"], "ple_embed_dim": 4}

    def engine(self, names=("native", "reverse-experts")):
        calls: list = []
        store = _StubStore()
        engine = quality._Engine(
            torch=torch,
            operators={name: _StubOps(name, calls) for name in names},
            checkpoint=_StubCheckpoint(),
            store=store,
            cfg=self.CFG,
            ple_metadata={},
            source_provenance=[],
            max_layers=None,
        )
        return engine, store, calls

    def test_layer_major_one_weight_read_per_layer_for_all_sequences_and_variants(self) -> None:
        engine, store, calls = self.engine()
        order = []
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        result = engine.run_pass(sequences, on_sequence=lambda name, index, layer: order.append((layer, index, name)))
        # all sequences/variants visit layer L before any visits layer L+1
        self.assertEqual([entry[0] for entry in order], sorted(entry[0] for entry in order))
        self.assertEqual(len(order), 2 * 3 * 2)
        # one read per tensor per layer, independent of sequence count and variant count
        for layer_reads in store.layer_reads[1:3]:
            self.assertTrue(layer_reads)
            self.assertTrue(all(count == 1 for count in layer_reads.values()), layer_reads)
        self.assertEqual(set(result["hidden"]), {"native", "reverse-experts"})
        self.assertEqual([tuple(h.shape) for h in result["hidden"]["native"]], [(3, 6), (2, 6), (4, 6)])
        self.assertEqual(len(result["layers"]), 2)
        self.assertEqual(result["layers"][0]["variants"], 2)
        self.assertEqual(result["layers"][0]["positions"], 9)

    def test_sequences_are_independent_no_cross_sequence_state(self) -> None:
        engine, _store, _calls = self.engine(("native",))
        together = engine.run_pass([[1, 2, 3], [4, 5, 6, 7]])["hidden"]["native"]
        alone_a = engine.run_pass([[1, 2, 3]])["hidden"]["native"][0]
        alone_b = engine.run_pass([[4, 5, 6, 7]])["hidden"]["native"][0]
        torch.testing.assert_close(together[0], alone_a, rtol=0, atol=0)
        torch.testing.assert_close(together[1], alone_b, rtol=0, atol=0)
        # every operator call saw exactly one sequence worth of positions (position 0 restarts)
        engine, _store, calls = self.engine(("native",))
        engine.run_pass([[1, 2, 3], [4, 5]])
        self.assertEqual({entry[3] for entry in calls}, {3, 2})

    def test_variants_have_independent_hidden_states(self) -> None:
        engine, _store, _calls = self.engine()
        result = engine.run_pass([[1, 2, 3]])
        a, b = result["hidden"]["native"][0], result["hidden"]["reverse-experts"][0]
        self.assertIsNot(a, b)
        self.assertEqual(a.data_ptr() == b.data_ptr(), False)
        self.assertEqual(engine.provenance("native")["operators"], {"label": "native"})

    def test_rejects_out_of_vocabulary_tokens_and_bad_layer_limit(self) -> None:
        engine, _store, _calls = self.engine(("native",))
        with self.assertRaises(FixtureError):
            engine.run_pass([[1, 99]])
        with self.assertRaises(FixtureError):
            quality._Engine(torch=torch, operators={"n": _StubOps("n", [])}, checkpoint=_StubCheckpoint(), store=_StubStore(), cfg=self.CFG, ple_metadata={}, source_provenance=[], max_layers=3)
        truncated = quality._Engine(torch=torch, operators={"n": _StubOps("n", [])}, checkpoint=_StubCheckpoint(), store=_StubStore(), cfg=self.CFG, ple_metadata={}, source_provenance=[], max_layers=1)
        self.assertTrue(truncated.truncated)
        self.assertEqual(len(truncated.run_pass([[1, 2]])["layers"]), 1)


class _FakeRawCheckpoint:
    """Raw-reader double: row r of every tensor holds the value r."""

    def __init__(self, cols: int = 4, bad: bool = False) -> None:
        self.cols, self.bad = cols, bad
        self.reads: list[tuple] = []
        self.weight_map: dict = {}

    def _record(self, rows: int, scale: float = 1.0) -> dict:
        data = torch.tensor([[float(r) * scale] * self.cols for r in range(rows)], dtype=torch.float32)
        if self.bad:
            data[0, 0] = float("nan")
        return {"dtype": "F32", "slice_shape": [rows, self.cols], "raw": bytearray(data.numpy().tobytes())}

    def tensor_rows_raw(self, name, row_start, row_count=None):
        self.reads.append(("rows", name, row_start, row_count))
        return self._record(3 if row_count is None else row_count)

    def expert_rows_raw(self, name, expert, row_start, row_count):
        self.reads.append(("expert", name, expert, row_start, row_count))
        return self._record(row_count, scale=float(expert + 1))


@unittest.skipIf(torch is None, "torch unavailable")
class StreamWeightsTests(unittest.TestCase):
    def test_full_cache_is_per_layer_and_experts_are_lru_bounded(self) -> None:
        checkpoint = _FakeRawCheckpoint()
        store = quality._StreamWeights(checkpoint, torch, expert_cache_bytes=2 * 8 * 4 * 4)  # two 8x4 float32 slices
        store.begin_layer()
        first = store.full("w")
        self.assertIs(store.full("w"), first)
        self.assertEqual(sum(1 for read in checkpoint.reads if read[0] == "rows"), 1)
        self.assertEqual(store.counters["full_hits"], 1)
        for expert in (0, 1, 0, 2, 0):
            store.expert_rows("e", expert, 0, 8)
        # 0,1 miss; 0 hit; 2 evicts LRU(1); 0 hit
        self.assertEqual(store.counters["expert_misses"], 3)
        self.assertEqual(store.counters["expert_hits"], 2)
        self.assertEqual(float(store.expert_rows("e", 2, 1, 1)[0, 0]), 0.0)  # fresh key = miss, row 0 of expert slice
        store.begin_layer()
        store.full("w")
        self.assertEqual(sum(1 for read in checkpoint.reads if read[0] == "rows"), 2)  # cache cleared across layers
        self.assertEqual(store.snapshot()["full_misses"], 2)

    def test_gather_falls_back_to_row_reads_and_nonfinite_is_rejected(self) -> None:
        checkpoint = _FakeRawCheckpoint()
        store = quality._StreamWeights(checkpoint, torch)
        rows = store.gather("t", [5, 3])
        self.assertEqual(tuple(rows.shape), (2, 4))
        with self.assertRaisesRegex(FixtureError, "nonfinite"):
            quality._StreamWeights(_FakeRawCheckpoint(bad=True), torch).rows("t", 0, 2)
        self.assertGreater(store.io_seconds, 0.0)


class _IntCheckpoint:
    """Checkpoint double for the exporter integration tests; records cleanup."""

    weight_map: dict = {}

    def __init__(self) -> None:
        self.closed = False

    def provenance(self) -> dict:
        return {"stub": "tiny-real-bf16-tensors", "read_bytes": 123}

    def close(self) -> None:
        self.closed = True


class _TinyStore(_StubStore):
    """Real (tiny, CPU) tensors: embeddings and a random lm_head in `dtype`."""

    def __init__(self, hidden: int, vocab: int, dtype) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(1234)
        self.hidden, self.dtype = hidden, dtype
        self.head = torch.randn(vocab, hidden, generator=generator).to(dtype)
        self.counters = {"expert_hits": 0, "expert_misses": 0}

    def gather(self, name: str, rows):
        values = [[((int(r) * 7) % 13 + 0.5 * j) * 0.25 for j in range(self.hidden)] for r in rows]
        return torch.tensor(values, dtype=torch.float32).to(self.dtype)

    def rows(self, name: str, row_start: int, row_count: int):
        self.layer_reads[-1][name] = self.layer_reads[-1].get(name, 0) + 1
        return self.head[row_start : row_start + row_count]


class _IntOps:
    """Per-sequence operators over real tensors. `reverse-experts` perturbs the MoE output and
    reverses the route logits so the two variants differ numerically and in routing."""

    EXPERTS, TOP_K = 6, 3

    def __init__(self, name: str, order: str, sink, fail_layer=None) -> None:
        self.name, self.order, self.sink, self.fail_layer = name, order, sink, fail_layer
        self.log: list = []

    def provenance(self) -> dict:
        return {"label": self.name, "reduction_order": self.order}

    def hyperconnection(self, hidden, weights, cfg, *, combine):
        if combine:
            return hidden, hidden, torch.zeros_like(hidden)
        return hidden[:, : cfg["hidden_size"]].contiguous()

    def gdn(self, hidden, weights, cfg, layer):
        return hidden * 0.5, {"recurrent": hidden, "conv": hidden}

    def qsa(self, hidden, weights, cfg, layer):
        return hidden, hidden

    def ple(self, hidden, tokens, cfg, metadata, load_full, read_rows):
        return torch.zeros_like(hidden)

    def inject(self, hyper_input, block_output, injection):
        return hyper_input + block_output

    def moe(self, hidden, cfg, load_full, load_expert, layer):
        if self.fail_layer == layer:
            raise RuntimeError("injected failure")
        scale = torch.arange(1, self.EXPERTS + 1, dtype=torch.float32)
        logits = torch.sin(hidden[:, : self.EXPERTS].float() * scale)
        if self.order == "reverse-experts":
            logits = logits.flip(1)
        values, indices = torch.topk(logits, self.TOP_K, dim=1)
        scores = torch.softmax(values, dim=1)
        self.log.append((layer, indices.clone(), scores.clone()))
        if self.sink is not None:
            self.sink(layer, logits, scores, indices)
        out = hidden * 0.0 + float(layer + 1)
        if self.order == "reverse-experts":
            out = out + 0.05 * hidden
        return out


@unittest.skipIf(torch is None, "torch unavailable")
class ExportIntegrationTests(unittest.TestCase):
    """Run the real `export_exact` / `run_floor` / scorer over a test-only `_Engine.open` fixture."""

    def setUp(self) -> None:
        self.corpus = _corpus()
        self.built: list = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def patch_engine(self, *, dtype=None, fail_layer=None):
        dtype = torch.bfloat16 if dtype is None else dtype
        built = self.built

        def fake_open(cls, torch_, cache_dir, local_dir=None, *, variants=(), expert_cache_bytes=0, max_layers=None, device="cpu", checkpoint_options=None):
            cfg = {
                "num_hidden_layers": 2, "vocab_size": VOCAB, "hidden_size": 8, "hc_count": 2,
                "num_experts": 6, "layer_types": ["linear_attention", "linear_attention"], "ple_embed_dim": 4,
            }
            operators = {name: _IntOps(name, order, sink, fail_layer) for name, order, sink in variants}
            checkpoint, store = _IntCheckpoint(), _TinyStore(8, VOCAB, dtype)
            built.append({"checkpoint": checkpoint, "store": store, "ops": operators, "max_layers": max_layers, "device": device})
            return cls(torch=torch_, operators=operators, checkpoint=checkpoint, store=store, cfg=cfg, ple_metadata={}, source_provenance=[{"stub": True}], max_layers=max_layers)

        return mock.patch.object(quality._Engine, "open", new=classmethod(fake_open))

    def export(self, name: str, **kwargs):
        kwargs.setdefault("orders", quality.REDUCTION_ORDERS)
        kwargs.setdefault("head_rows", VOCAB)
        with self.patch_engine(dtype=kwargs.pop("dtype", None), fail_layer=kwargs.pop("fail_layer", None)):
            return quality.export_exact(self.corpus, self.root / name, torch=torch, **kwargs)

    def test_paired_export_writes_all_artifacts_with_consistent_identity(self) -> None:
        manifests = self.export("pair", export_hidden=True)
        self.assertEqual(set(manifests), {"native", "reverse-experts"})
        total, rows, vocab = self.corpus.input_tokens, self.corpus.scored_positions, VOCAB
        for order, manifest in manifests.items():
            directory = self.root / "pair" / order
            identity = manifest["identity"]
            names = {path.name for path in directory.iterdir()}
            self.assertEqual(names, {"export.json", "score_index.u32le", "logits.f32le", "route_ids.u16le", "route_scores.f32le", "final_hidden.bfloat16le"})
            self.assertEqual(identity["reduction_order"], order)
            self.assertEqual(identity["logits_storage"], "f32")
            self.assertEqual(identity["forward_head_dtype"], "bfloat16")
            self.assertEqual(identity["hidden_dtype"], "bfloat16")
            self.assertEqual(identity["layers_executed"], 2)
            self.assertFalse(identity["truncated_layers"])
            self.assertEqual(identity["route_top_k"], 3)
            self.assertEqual(identity["source"]["operators"], {"label": order, "reduction_order": order})
            self.assertEqual(identity["execution"]["type"], "cpu")
            self.assertEqual(identity["files"]["logits"]["shape"], [rows, vocab])
            self.assertEqual(identity["files"]["route_ids"]["shape"], [2, total, 3])
            self.assertEqual((directory / "logits.f32le").stat().st_size, rows * vocab * 4)
            self.assertEqual((directory / "route_ids.u16le").stat().st_size, 2 * total * 3 * 2)
            self.assertEqual((directory / "route_scores.f32le").stat().st_size, 2 * total * 3 * 4)
            self.assertEqual((directory / "final_hidden.bfloat16le").stat().st_size, rows * 8 * 2)
            for entry in identity["files"].values():
                self.assertEqual(entry["sha256"], _sha(directory / entry["path"]))
                self.assertEqual(entry["bytes"], (directory / entry["path"]).stat().st_size)
            stored = json.loads((directory / "export.json").read_text())
            self.assertEqual(stored["identity_sha256"], hashlib.sha256(quality._canonical_json(stored["identity"])).hexdigest())
            self.assertEqual(stored["identity"]["corpus"]["corpus_sha256"], self.corpus.identity()["corpus_sha256"])
            runtime = stored["runtime"]
            self.assertEqual(runtime["lockstep_variants"], ["native", "reverse-experts"])
            self.assertTrue(runtime["timing_shared_across_lockstep_variants"])
            timing = runtime["timing"]
            self.assertEqual((timing["passes"], timing["variants"], timing["input_tokens"], timing["scored_rows"]), (1, 2, total, rows))
            self.assertEqual(len(timing["layer_records"]), 2)
            self.assertEqual(len(timing["per_layer"]), 2)
            self.assertIn("estimated_seconds_lockstep_variants_run", timing["projection_327680"])
            self.assertEqual(runtime["execution"]["type"], "cpu")
        self.assertEqual(len(self.built), 1)  # ONE engine / weight traversal for both variants
        self.assertTrue(self.built[0]["checkpoint"].closed)

    def test_logits_and_route_tapes_match_independent_recomputation(self) -> None:
        self.export("pair", export_hidden=True)
        built = self.built[0]
        head = built["store"].head
        corpus, total = self.corpus, self.corpus.input_tokens
        rows = corpus.scored_positions
        for order in quality.REDUCTION_ORDERS:
            directory = self.root / "pair" / order
            hidden = torch.from_numpy(np.fromfile(directory / "final_hidden.bfloat16le", dtype="<i2").reshape(rows, 8)).view(torch.bfloat16)
            expected = torch.matmul(hidden, head.transpose(0, 1)).float().numpy()
            logits = np.fromfile(directory / "logits.f32le", dtype="<f4").reshape(rows, VOCAB)
            np.testing.assert_array_equal(logits, expected)  # actual bf16 matmul widened to F32
            ids = np.fromfile(directory / "route_ids.u16le", dtype="<u2").reshape(2, total, 3)
            scores = np.fromfile(directory / "route_scores.f32le", dtype="<f4").reshape(2, total, 3)
            log = built["ops"][order].log
            self.assertEqual(len(log), 2 * len(corpus.sequences))  # one capture per (layer, sequence)
            for layer in range(2):
                for seq in corpus.sequences:
                    _layer, logged_ids, logged_scores = log[layer * len(corpus.sequences) + seq.index]
                    self.assertEqual(_layer, layer)
                    start = corpus.offsets[seq.index]
                    np.testing.assert_array_equal(ids[layer, start : start + seq.length], logged_ids.numpy())  # every position, scored or not
                    np.testing.assert_array_equal(scores[layer, start : start + seq.length], logged_scores.numpy())
        index = np.fromfile(self.root / "pair" / "native" / "score_index.u32le", dtype="<u4").reshape(-1, 3)
        np.testing.assert_array_equal(index, corpus.score_index())

    def test_floor_compare_over_exported_pair_and_self_sanity(self) -> None:
        self.export("pair")
        native, reverse = self.root / "pair" / "native", self.root / "pair" / "reverse-experts"
        floor = quality.exact_compare(native, reverse, self.root / "floor.json", bootstrap=200, seed=1)
        self.assertEqual(floor["routing"]["status"], "compared")
        self.assertLess(floor["routing"]["all_positions"]["exact_ordered"], 1.0)
        self.assertGreater(floor["results"]["mixture"]["metrics"]["kl"]["mean"], 0.0)
        self.assertEqual(floor["reference"]["reduction_order"], "native")
        self.assertEqual(floor["candidate"]["reduction_order"], "reverse-experts")
        self.assertEqual(floor["scored_rows"], self.corpus.scored_positions)
        self.assertFalse(floor["truncated_layers"])
        saved = np.load(self.root / "floor.positions.npz")
        self.assertEqual(saved["kl"].shape[0], self.corpus.scored_positions)
        self.assertGreaterEqual(float(saved["kl"].min()), -quality.NEGATIVE_KL_TOLERANCE)
        own = quality.exact_compare(native, native, self.root / "self.json", bootstrap=200, seed=1)
        self.assertLessEqual(abs(own["results"]["mixture"]["metrics"]["kl"]["mean"]), 1e-10)
        self.assertEqual(own["routing"]["all_positions"]["exact_ordered"], 1.0)

    def test_run_floor_end_to_end(self) -> None:
        with self.patch_engine():
            result = quality.run_floor(self.corpus, self.root / "floor", torch=torch, bootstrap=200, seed=2, head_rows=VOCAB)
        for order in quality.REDUCTION_ORDERS:
            self.assertTrue((self.root / "floor" / order / "export.json").is_file())
        for name in ("floor_native_vs_reverse.json", "floor_native_vs_reverse.positions.npz", "floor_self_native.json", "floor_self_native.positions.npz"):
            self.assertTrue((self.root / "floor" / name).is_file(), name)
        self.assertGreater(result["floor"]["results"]["mixture"]["metrics"]["kl"]["mean"], 0.0)
        self.assertLessEqual(abs(result["self_sanity"]["results"]["mixture"]["metrics"]["kl"]["mean"]), 1e-10)
        self.assertEqual(len(self.built), 1)
        with self.patch_engine():
            skipped = quality.run_floor(self.corpus, self.root / "floor2", self_sanity=False, torch=torch, bootstrap=100, head_rows=VOCAB)
        self.assertNotIn("self_sanity", skipped)

    def test_bf16_storage_is_lossless_and_halves_logits(self) -> None:
        self.export("f32", orders=("native",))
        manifests = self.export("bf", orders=("native",), logits_storage="bf16")
        identity = manifests["native"]["identity"]
        directory = self.root / "bf" / "native"
        self.assertEqual(identity["logits_storage"], "bf16")
        self.assertEqual(identity["files"]["logits"]["dtype"], "bfloat16")
        self.assertEqual(identity["files"]["logits"]["path"], "logits.bf16le")
        self.assertFalse((directory / "logits.f32le").exists())
        self.assertEqual((directory / "logits.bf16le").stat().st_size * 2, (self.root / "f32" / "native" / "logits.f32le").stat().st_size)
        words = np.fromfile(directory / "logits.bf16le", dtype="<u2").reshape(-1, VOCAB)
        widened = (words.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(widened, np.fromfile(self.root / "f32" / "native" / "logits.f32le", dtype="<f4").reshape(-1, VOCAB))
        report = quality.exact_compare(self.root / "f32" / "native", directory, self.root / "mixed.json", bootstrap=100, seed=1)
        self.assertLessEqual(abs(report["results"]["mixture"]["metrics"]["kl"]["mean"]), 1e-10)
        self.assertEqual(report["results"]["text"]["metrics"]["top1_agreement"]["mean"], 1.0)
        # paired floor in bf16 storage works too and compares against the f32 export of the same variant
        paired = self.export("bfpair", logits_storage="bf16")
        self.assertEqual({m["identity"]["files"]["logits"]["path"] for m in paired.values()}, {"logits.bf16le"})
        floor = quality.exact_compare(self.root / "bfpair" / "native", self.root / "bfpair" / "reverse-experts", self.root / "bfloor.json", bootstrap=100, seed=1)
        self.assertGreater(floor["results"]["mixture"]["metrics"]["kl"]["mean"], 0.0)

    def test_bf16_storage_refused_when_forward_is_not_bf16(self) -> None:
        with self.assertRaisesRegex(FixtureError, "bf16 logits storage refused"):
            self.export("f32fwd", orders=("native",), logits_storage="bf16", dtype=torch.float32)
        self.assertTrue(self.built[0]["checkpoint"].closed)  # cleanup still ran on refusal
        manifests = self.export("f32ok", orders=("native",), dtype=torch.float32, export_hidden=True)
        identity = manifests["native"]["identity"]
        self.assertEqual(identity["forward_head_dtype"], "float32")
        self.assertEqual(identity["hidden_dtype"], "float32")
        self.assertTrue((self.root / "f32ok" / "native" / "final_hidden.float32le").is_file())

    def test_truncated_export_cannot_stand_in_for_the_full_teacher(self) -> None:
        self.export("full", orders=("native",))
        manifests = self.export("cut", orders=("native",), max_layers=1)
        identity = manifests["native"]["identity"]
        self.assertTrue(identity["truncated_layers"])
        self.assertEqual(identity["layers_executed"], 1)
        self.assertEqual(identity["files"]["route_ids"]["shape"], [1, self.corpus.input_tokens, 3])
        self.assertEqual(self.built[1]["max_layers"], 1)
        self.assertEqual(len(manifests["native"]["runtime"]["timing"]["layer_records"]), 1)
        with self.assertRaisesRegex(FixtureError, "truncated-layer"):
            quality.exact_compare(self.root / "full" / "native", self.root / "cut" / "native", self.root / "x.json", bootstrap=100)
        other = self.export("cut2", orders=("reverse-experts",), max_layers=1)
        self.assertTrue(other["reverse-experts"]["identity"]["truncated_layers"])
        with self.assertRaisesRegex(FixtureError, "truncated-layer"):
            quality.exact_compare(self.root / "cut" / "native", self.root / "cut2" / "reverse-experts", self.root / "t.json", bootstrap=100, seed=1)
        self.assertFalse((self.root / "t.json").exists())
        self.assertFalse((self.root / "x.json").exists())

    def test_run_floor_on_truncated_probe_reports_timings_only(self) -> None:
        with self.patch_engine():
            probe = quality.run_floor(self.corpus, self.root / "probe", torch=torch, max_layers=1, bootstrap=100, head_rows=VOCAB)
        self.assertEqual(probe["status"], "truncated_timing_probe_no_quality")
        self.assertIsNone(probe["floor"])
        self.assertIsNone(probe["self_sanity"])
        for name in ("floor_native_vs_reverse.json", "floor_native_vs_reverse.positions.npz", "floor_self_native.json", "floor_self_native.positions.npz"):
            self.assertFalse((self.root / "probe" / name).exists(), name)
        for order in quality.REDUCTION_ORDERS:
            manifest = json.loads((self.root / "probe" / order / "export.json").read_text())
            self.assertTrue(manifest["identity"]["truncated_layers"])
            self.assertEqual(manifest["identity"]["layers_executed"], 1)
            self.assertEqual(len(manifest["runtime"]["timing"]["layer_records"]), 1)
        with self.patch_engine():
            full = quality.run_floor(self.corpus, self.root / "complete", torch=torch, bootstrap=100, head_rows=VOCAB)
        self.assertEqual(full["status"], "complete")

    def test_multi_pass_export_matches_single_pass_and_resets_sequences(self) -> None:
        single = self.export("one", orders=("native",))
        many = self.export("many", orders=("native",), pass_tokens=16)
        self.assertEqual(many["native"]["runtime"]["timing"]["passes"], 3)
        self.assertEqual(single["native"]["runtime"]["timing"]["passes"], 1)
        # Hidden states (hence routes) never see the pass grouping: bit-identical tapes. Logits come from a
        # bf16 matmul whose batch shape differs per pass, so require only a negligible KL, not equal bytes.
        for key in ("route_ids", "route_scores", "score_index"):
            self.assertEqual(single["native"]["identity"]["files"][key]["sha256"], many["native"]["identity"]["files"][key]["sha256"], key)
        report = quality.exact_compare(self.root / "one" / "native", self.root / "many" / "native", self.root / "pass.json", bootstrap=100, seed=1)
        self.assertLessEqual(abs(report["results"]["mixture"]["metrics"]["kl"]["mean"]), 1e-3)
        self.assertEqual(report["routing"]["all_positions"]["exact_ordered"], 1.0)

    def test_hash_flag_and_device_are_recorded_and_failures_clean_up(self) -> None:
        manifests = self.export("nohash", orders=("native",), hash_files=False)
        self.assertTrue(all(entry["sha256"] is None for entry in manifests["native"]["identity"]["files"].values()))
        self.assertEqual(self.built[0]["device"], "cpu")
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.export("boom", orders=("native",), fail_layer=1)
        self.assertTrue(self.built[1]["checkpoint"].closed)
        self.assertFalse((self.root / "boom" / "native" / "export.json").exists())  # no manifest for an aborted run


if __name__ == "__main__":
    unittest.main()
