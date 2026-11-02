# SPDX-License-Identifier: Apache-2.0
"""Focused tests for canonical corpus identity and Astrea comparison output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import quality


class QualityTests(unittest.TestCase):
    def test_canonical_corpus_is_metadata_plus_u32le(self) -> None:
        tokens, identity = quality._load_corpus("benchmarks/prompts/qwen4-teacher-forced.tokens.json")
        self.assertEqual(len(tokens), quality.CANONICAL_TOKEN_COUNT)
        self.assertEqual(identity["payload_sha256"], quality.CANONICAL_TOKEN_SHA256)
        self.assertEqual(tokens[-1], 113)

    def test_config_adapter_supplies_source_defaults(self) -> None:
        adapter = object.__new__(quality.upstream.PinnedQwen4Operators)
        config = adapter._config({"hidden_size": 8, "num_attention_heads": 2})
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.seed, 1234)

    def test_compare_emits_astrea_row(self) -> None:
        tokens, _ = quality._load_corpus("benchmarks/prompts/qwen4-teacher-forced.tokens.json")
        rows = [
            {
                "position": index,
                "input_id": token,
                "target_id": tokens[index + 1] if index + 1 < len(tokens) else None,
                "target_logit": 0.0 if index + 1 < len(tokens) else None,
                "logsumexp": 1.0,
                "top_ids": [0, 1],
                "top_logits": [0.0, -1.0],
                "top1": 0,
            }
            for index, token in enumerate(tokens)
        ]
        artifact = {
            "schema": quality.QUALITY_SCHEMA,
            "variant": "test",
            "tokens": tokens,
            "corpus_sha256": quality.CANONICAL_TOKEN_SHA256,
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            candidate = Path(directory) / "candidate.json"
            output = Path(directory) / "comparison.json"
            reference.write_text(json.dumps(artifact))
            candidate.write_text(json.dumps(artifact))
            report = quality.compare(reference, candidate, output)
            self.assertEqual(report["schema"], quality.QUALITY_SCHEMA)
            self.assertEqual(report["identity"]["n_scored"], 16)
            self.assertEqual(report["rows"][0]["scoring_mode"], "teacher_forced")
            self.assertEqual(report["comparison"]["top1_agreement"], 1.0)

    def test_compare_rejects_noncanonical_digest(self) -> None:
        artifact = {
            "schema": quality.QUALITY_SCHEMA,
            "tokens": [1] * quality.CANONICAL_TOKEN_COUNT,
            "corpus_sha256": "0" * 64,
            "rows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            candidate = Path(directory) / "candidate.json"
            output = Path(directory) / "comparison.json"
            reference.write_text(json.dumps(artifact))
            candidate.write_text(json.dumps(artifact))
            with self.assertRaises(quality.FixtureError):
                quality.compare(reference, candidate, output)
    def test_compare_rejects_nonfinite_logits(self) -> None:
        tokens, _ = quality._load_corpus("benchmarks/prompts/qwen4-teacher-forced.tokens.json")
        rows = [
            {
                "position": index,
                "input_id": token,
                "target_id": tokens[index + 1] if index + 1 < len(tokens) else None,
                "target_logit": 0.0 if index + 1 < len(tokens) else None,
                "logsumexp": 1.0,
                "top_ids": [0],
                "top_logits": [float("nan")],
                "top1": 0,
            }
            for index, token in enumerate(tokens)
        ]
        artifact = {
            "schema": quality.QUALITY_SCHEMA,
            "tokens": tokens,
            "corpus_sha256": quality.CANONICAL_TOKEN_SHA256,
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            candidate = Path(directory) / "candidate.json"
            output = Path(directory) / "comparison.json"
            reference.write_text(json.dumps(artifact, allow_nan=True))
            candidate.write_text(json.dumps(artifact, allow_nan=True))
            with self.assertRaises(quality.FixtureError):
                quality.compare(reference, candidate, output)


if __name__ == "__main__":
    unittest.main()
