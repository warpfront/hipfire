# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiny full-stack profiles (``tiny-moe`` / ``tiny-dense-dev``).

Run from the clean CPU environment (pinned torch/transformers/safetensors/
tokenizers), with the vLLM source cache reachable::

    HIPFIRE_QWEN4_ORACLE_CACHE=~/.hipfire/datasets/cachequality-qwen4-upstream \
    s0-cpu-venv/bin/python -I -m unittest discover \
        -s crates/hipfire-arch-qwen4/reference_oracle -p 'test_tiny_profiles.py' -v

The pure-Python tests (envelope, dense equation, rejection guards) need no
torch.  The full-stack tests are skipped only when torch / transformers /
safetensors / tokenizers are not importable at all; a wrong version or a hash
mismatch fails loudly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import equations
import generate_fixtures
import tiny_profiles as tp
from schema import FixtureError, read_npz, tensor

_FULL_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors", "tokenizers"))


def _dump_names(manifest: dict, dump: str) -> set[str]:
    return {item["name"] for entry in manifest["dumps"] if entry["name"] == dump for item in entry["arrays"]}


def _safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


class EnvelopeTests(unittest.TestCase):
    def test_moe_envelope_is_frozen_shape(self) -> None:
        envelope = tp.profile_envelope(tp.ProfileOptions(profile_id="tiny-moe"))
        self.assertEqual(
            envelope,
            {
                "id": "tiny-moe",
                "developer_only": True,
                "equation_source": "pinned-upstream",
                "ffn": {
                    "kind": "moe",
                    "num_experts": 3,
                    "top_k": 2,
                    "intermediate_size": 8,
                    "shared_intermediate_size": 8,
                    "norm_topk_prob": True,
                },
            },
        )

    def test_dense_envelope_declares_intermediate_size_only(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-dense-dev", ffn_intermediate_size=24)
        envelope = tp.profile_envelope(options)
        self.assertEqual(envelope["equation_source"], "non-upstream-dense")
        self.assertEqual(envelope["ffn"], {"kind": "dense", "intermediate_size": 24})
        text = tp.text_config_dict(options)
        self.assertEqual(text["intermediate_size"], 24)
        for forbidden in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size", "norm_topk_prob"):
            self.assertNotIn(forbidden, text, "dense config must not carry MoE fields (no dummy expert)")
        # HF's strict validator needs placeholders; they stay in memory only.
        hf = tp._hf_text_config_dict(options)
        self.assertEqual(hf["num_experts"], 1)
        self.assertNotIn("intermediate_size", hf)

    def test_gate_construction_values(self) -> None:
        text = tp.text_config_dict(tp.ProfileOptions())
        expected = {
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "hc_count": 2,
            "hc_lowrank": 4,
            "ple_layer_ids": [1],
            "ple_embed_dim": 16,
            "ngram_size": 3,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 7,
            "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 1,
            "ple_conv_kernel_size": 2,
            "linear_conv_kernel_dim": 2,
            "eos_token_id": 2,
            "indexer_n_heads": 1,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 4,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "max_position_embeddings": 64,
            "output_gate_type": "sigmoid",
        }
        for key, value in expected.items():
            self.assertEqual(text[key], value, key)
        self.assertEqual(text["rope_parameters"]["rope_theta"], 10000.0)
        self.assertEqual(text["rope_parameters"]["partial_rotary_factor"], 0.5)
        self.assertEqual(text["mtp"]["layer_types"], ["full_attention"])

    def test_full_config_root_carries_profile_and_envelope(self) -> None:
        config = tp.full_config_dict(tp.ProfileOptions())
        self.assertEqual(config["model_type"], "qwen4_exp")
        self.assertEqual(config["architectures"], ["Qwen4ExpForConditionalGeneration"])
        self.assertEqual(config["text_config"]["model_type"], "qwen4_exp_text")
        self.assertEqual(config["hipfire_profile"]["id"], "tiny-moe")
        self.assertTrue(config["hipfire_profile"]["developer_only"])

    def test_knob_guards(self) -> None:
        with self.assertRaisesRegex(FixtureError, "shared_intermediate_size=null"):
            tp.ProfileOptions(shared_intermediate_size=None).validate()
        with self.assertRaisesRegex(FixtureError, "1-2 layers"):
            tp.ProfileOptions(hidden_size=5120, num_layers=3).validate()
        with self.assertRaisesRegex(FixtureError, "reduced FFN"):
            tp.ProfileOptions(profile_id="tiny-dense-dev", hidden_size=5120, num_layers=2, ffn_intermediate_size=13824).validate()
        with self.assertRaisesRegex(FixtureError, "unknown tiny profile"):
            tp.ProfileOptions(profile_id="tiny-moe-x").validate()
        with self.assertRaisesRegex(FixtureError, "top_k"):
            tp.ProfileOptions(num_experts=3, top_k=3).validate()
        tp.ProfileOptions(hidden_size=5120, num_layers=1).validate()
        tp.ProfileOptions(hidden_size=5120, num_layers=2).validate()
        self.assertGreater(tp.estimated_parameter_elements(tp.ProfileOptions(hidden_size=8192, num_layers=1)), tp.MAX_PARAMETER_ELEMENTS)
        self.assertLess(tp.estimated_parameter_elements(tp.ProfileOptions(hidden_size=5120, num_layers=2)), tp.MAX_PARAMETER_ELEMENTS)

    def test_rope_knobs_default_to_the_declared_construction_and_reach_main_and_mtp_config(self) -> None:
        defaults = tp.ProfileOptions()
        self.assertEqual((defaults.rope_theta, defaults.max_position_embeddings), (10000.0, 64))
        variant = tp.ProfileOptions(rope_theta=10_000_000.0, max_position_embeddings=1024)
        variant.validate()
        text = tp.text_config_dict(variant)
        self.assertEqual(text["rope_parameters"], {"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.5})
        self.assertEqual(text["max_position_embeddings"], 1024)
        self.assertEqual(text["mtp"]["rope_theta"], 10_000_000.0)
        self.assertEqual(tp.text_config_dict(tp.ProfileOptions(rope_theta=10_000_000))["mtp"]["rope_theta"], 10_000_000.0)
        self.assertEqual(tp.full_config_dict(variant)["hipfire_profile"]["id"], "tiny-moe")
        self.assertEqual(tp._hf_text_config_dict(variant)["rope_parameters"]["rope_theta"], 10_000_000.0)
        dense = tp.ProfileOptions(profile_id="tiny-dense-dev", rope_theta=10_000_000.0, max_position_embeddings=1024)
        self.assertEqual(tp._hf_text_config_dict(dense)["max_position_embeddings"], 1024)
        self.assertEqual(tp.text_config_dict(defaults), tp.text_config_dict(tp.ProfileOptions(rope_theta=10000, max_position_embeddings=64)))

    def test_rope_knob_guards(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf"), True, "1e7"):
            with self.assertRaisesRegex(FixtureError, "rope_theta"):
                tp.ProfileOptions(rope_theta=bad).validate()
        for bad in (0, -64, 64.0, True):
            with self.assertRaisesRegex(FixtureError, "max_position_embeddings"):
                tp.ProfileOptions(max_position_embeddings=bad).validate()

    def test_cli_rope_knobs_reach_profile_options(self) -> None:
        args = generate_fixtures._parser().parse_args(["--tiny-profile", "tiny-moe", "--out", "x"])
        options = generate_fixtures._tiny_options(args)
        self.assertEqual((options.rope_theta, options.max_position_embeddings), (10000.0, 64))
        args = generate_fixtures._parser().parse_args(
            ["--tiny-profile", "tiny-dense-dev", "--out", "x", "--rope-theta", "10000000", "--max-position-embeddings", "1024"]
        )
        options = generate_fixtures._tiny_options(args)
        self.assertEqual((options.profile_id, options.rope_theta, options.max_position_embeddings), ("tiny-dense-dev", 10_000_000.0, 1024))
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in (("--rope-theta", "0"), ("--max-position-embeddings", "0")):
                code = generate_fixtures.main(["--tiny-profile", "tiny-moe", "--out", str(Path(temp) / "x"), flag, value, "--no-controls"])
                self.assertEqual(code, 2, flag)

    def test_optional_components_default_present_and_s1_absence_in_config(self) -> None:
        defaults = tp.ProfileOptions()
        self.assertEqual((defaults.with_ple, defaults.with_mtp), (True, True))
        baseline = tp.text_config_dict(defaults)
        self.assertEqual(baseline, tp.text_config_dict(tp.ProfileOptions(with_ple=True, with_mtp=True)))
        self.assertEqual(baseline["ple_layer_ids"], [1])
        self.assertEqual(baseline["mtp_num_hidden_layers"], 1)
        # S1 PLE absence: ple_layer_ids == [] and every PLE scalar absent; nothing else changes (MTP kept).
        no_ple = tp.text_config_dict(tp.ProfileOptions(with_ple=False))
        expected = {key: value for key, value in baseline.items() if key not in tp.PLE_COMPANION_KEYS}
        expected["ple_layer_ids"] = []
        self.assertEqual(no_ple, expected)
        for key in ("ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram", "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts"):
            self.assertNotIn(key, no_ple)
        self.assertEqual(no_ple["mtp"], baseline["mtp"])
        # S1 MTP absence: no mtp block, no mtp_num_hidden_layers / mtp_use_dedicated_embeddings (PLE kept).
        no_mtp = tp.text_config_dict(tp.ProfileOptions(with_mtp=False))
        self.assertEqual(no_mtp, {key: value for key, value in baseline.items() if key not in ("mtp", "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")})
        self.assertEqual(no_mtp["ple_layer_ids"], [1])
        # The root carries no PLE marker unless a PLE table was instantiated.
        for options in (tp.ProfileOptions(with_ple=False), tp.ProfileOptions(profile_id="tiny-dense-dev", with_ple=False, with_mtp=False)):
            root = tp.full_config_dict(options)
            self.assertNotIn("qwen4_ple", root)
            self.assertNotIn("ple", root["hipfire_profile"])
        # pinned HF gets the same PLE-free text config in memory.
        self.assertEqual(tp._hf_text_config_dict(tp.ProfileOptions(with_ple=False))["ple_layer_ids"], [])
        both = tp.text_config_dict(tp.ProfileOptions(profile_id="tiny-dense-dev", rope_theta=10_000_000.0, max_position_embeddings=1024, with_ple=False, with_mtp=False))
        self.assertEqual((both["rope_parameters"]["rope_theta"], both["max_position_embeddings"], both["intermediate_size"]), (10_000_000.0, 1024, 8))
        self.assertEqual((both["ple_layer_ids"], "mtp" in both, "mtp_num_hidden_layers" in both), ([], False, False))
        self.assertLess(tp.estimated_parameter_elements(tp.ProfileOptions(with_mtp=False)), tp.estimated_parameter_elements(defaults))
        self.assertEqual(defaults.with_overrides(with_ple=False).with_ple, False)
        for knob in ("with_ple", "with_mtp"):
            for bad in (0, 1, "no", None):
                with self.assertRaisesRegex(FixtureError, knob):
                    tp.ProfileOptions(**{knob: bad}).validate()

    def test_cli_optional_component_switches(self) -> None:
        def options(*flags: str):
            return generate_fixtures._tiny_options(generate_fixtures._parser().parse_args(["--tiny-profile", "tiny-dense-dev", "--out", "x", *flags]))

        for flags, expected in (((), (True, True)), (("--no-ple",), (False, True)), (("--no-mtp",), (True, False)), (("--no-ple", "--no-mtp"), (False, False))):
            parsed = options(*flags)
            self.assertEqual((parsed.with_ple, parsed.with_mtp), expected, flags)
        with tempfile.TemporaryDirectory() as temp:
            for flag in ("--no-ple", "--no-mtp"):
                self.assertEqual(generate_fixtures.main(["--out", str(Path(temp) / "x"), flag]), 2, flag)

    def test_layer_schedule(self) -> None:
        self.assertEqual(tp.layer_types_for(1), ["linear_attention"])
        self.assertEqual(tp.layer_types_for(2), ["linear_attention", "full_attention"])
        self.assertFalse(tp.has_attention_layer(tp.ProfileOptions(num_layers=1)))
        self.assertTrue(tp.has_attention_layer(tp.ProfileOptions(num_layers=2)))


class DenseEquationTests(unittest.TestCase):
    def _case(self):
        hidden = tensor((2, 3), [0.5, -1.0, 0.25, 1.5, 0.125, -0.75])
        gate = tensor((2, 3), [0.25, 0.5, -0.5, -1.0, 0.75, 0.125])
        up = tensor((2, 3), [1.0, -0.25, 0.5, 0.375, -0.5, 0.25])
        down = tensor((3, 2), [0.5, -1.0, 0.25, 0.75, -0.5, 1.5])
        return hidden, gate, up, down

    def test_matches_hand_computation(self) -> None:
        hidden, gate, up, down = self._case()
        result = equations.dense_ffn_non_upstream(hidden, gate, up, down, 2, round_bf16=False)
        # Token 0, computed independently of the implementation from nn.Linear-layout weights.
        g0 = 0.5 * 0.25 + (-1.0) * 0.5 + 0.25 * -0.5
        g1 = 0.5 * -1.0 + (-1.0) * 0.75 + 0.25 * 0.125
        u0 = 0.5 * 1.0 + (-1.0) * -0.25 + 0.25 * 0.5
        u1 = 0.5 * 0.375 + (-1.0) * -0.5 + 0.25 * 0.25
        self.assertAlmostEqual(result["gate"].data[0], g0, places=6)
        self.assertAlmostEqual(result["gate"].data[1], g1, places=6)
        self.assertAlmostEqual(result["up"].data[0], u0, places=6)
        p0 = equations.silu_scalar(g0) * u0
        p1 = equations.silu_scalar(g1) * u1
        self.assertAlmostEqual(result["product"].data[0], p0, places=5)
        o0 = p0 * 0.5 + p1 * -1.0
        self.assertAlmostEqual(result["output"].data[0], o0, places=5)
        self.assertEqual(result["output"].shape, (2, 3))

    def test_shapes_follow_declared_intermediate_size(self) -> None:
        hidden, gate, up, down = self._case()
        result = equations.dense_ffn_non_upstream(hidden, gate, up, down, 2)
        self.assertEqual(result["gate"].shape, (2, 2))
        self.assertEqual(result["output"].shape, (2, 3))
        with self.assertRaisesRegex(FixtureError, "gate/up weights"):
            equations.dense_ffn_non_upstream(hidden, gate, up, down, 3)
        with self.assertRaisesRegex(FixtureError, "down weight"):
            equations.dense_ffn_non_upstream(hidden, gate, up, tensor((2, 3), [0.0] * 6), 2)
        with self.assertRaisesRegex(FixtureError, "positive"):
            equations.dense_ffn_non_upstream(hidden, gate, up, down, 0)

    def test_rounding_boundary_is_observable(self) -> None:
        hidden, gate, up, down = self._case()
        rounded = equations.dense_ffn_non_upstream(hidden, gate, up, down, 2, round_bf16=True)
        exact = equations.dense_ffn_non_upstream(hidden, gate, up, down, 2, round_bf16=False)
        self.assertNotEqual(rounded["activation"].data, exact["activation"].data)
        for value in rounded["activation"].data + rounded["output"].data:
            self.assertEqual(value, equations.bf16_round(value), "BF16 mode must store BF16-representable values")

    def test_label_is_non_upstream(self) -> None:
        self.assertEqual(equations.DENSE_FFN_EQUATION_SOURCE, "non-upstream-dense")
        self.assertIn("NON-UPSTREAM", equations.dense_ffn_non_upstream.__doc__)


class QsaScreenTests(unittest.TestCase):
    @staticmethod
    def _row(source: str, position: int, scores: list[float]) -> dict:
        return {"source": source, "layer": 1 if source == "main" else 0, "position": position, "blocks": len(scores), "block_topk": 2, "scores": scores}

    def test_margin_predicate(self) -> None:
        # kth > 0 and kth - next >= max(0.05, 0.05 * kth)
        self.assertTrue(tp.qsa_margin_row([1.0, 0.5, 0.2], 2)["passed"])
        self.assertEqual(tp.qsa_margin_row([1.0, 0.5, 0.2], 2)["required_margin"], 0.05)
        self.assertTrue(tp.qsa_margin_row([1.0, 0.5, 0.4375], 2)["passed"])  # margin 0.0625 >= absolute 0.05
        self.assertFalse(tp.qsa_margin_row([1.0, 0.5, 0.49], 2)["passed"])
        self.assertFalse(tp.qsa_margin_row([1.0, 0.0, 0.0], 2)["passed"])  # kth == 0 (relu tie of zeros)
        self.assertFalse(tp.qsa_margin_row([2.0, 0.0, -1.0], 2)["passed"])
        # relative term dominates for large kth: required = 0.05 * 100 = 5
        big = tp.qsa_margin_row([200.0, 100.0, 94.0], 2)
        self.assertEqual((big["kth"], big["next"], big["required_margin"], big["observed_margin"], big["passed"]), (100.0, 94.0, 5.0, 6.0, True))
        self.assertFalse(tp.qsa_margin_row([200.0, 100.0, 96.0], 2)["passed"])
        # order independent; k is the 1-based boundary rank
        self.assertEqual(tp.qsa_margin_row([0.2, 1.0, 0.5], 2), tp.qsa_margin_row([1.0, 0.5, 0.2], 2))
        with self.assertRaises(FixtureError):
            tp.qsa_margin_row([1.0, 0.5], 2)  # no boundary: not a selecting row
        with self.assertRaises(FixtureError):
            tp.qsa_margin_row([1.0, float("nan"), 0.5], 2)

    def test_evaluated_rows_carry_manifest_fields(self) -> None:
        row = tp.evaluate_qsa_rows([self._row("mtp", 6, [1.0, 0.5, 0.2])])[0]
        for key in ("source", "layer", "position", "complete_blocks", "block_topk", "block_scores", "kth", "next", "required_margin", "observed_margin", "passed"):
            self.assertIn(key, row)
        self.assertEqual((row["source"], row["layer"], row["position"], row["complete_blocks"], row["block_topk"]), ("mtp", 0, 6, 3, 2))

    def _screen(self, per_seed: dict, options: "tp.ProfileOptions", max_seeds: int = 8) -> dict:
        seen: list[int] = []

        def fake(opts, sources, tokens, seed):
            seen.append(seed)
            return tp.evaluate_qsa_rows(per_seed(seed))

        with mock.patch.object(tp, "qsa_candidate_rows", fake):
            result = tp.screen_qsa_seed(options, None, [], max_seeds=max_seeds)
        self.assertEqual(seen, list(range(options.seed, result["effective_seed"] + 1)), "sequential seeds, none skipped")
        return result

    def test_seed_search_is_sequential_and_records_metadata(self) -> None:
        good = lambda: [self._row("main", 5, [1.0, 0.5, 0.2]), self._row("mtp", 5, [1.0, 0.5, 0.2])]

        def per_seed(seed: int) -> list[dict]:
            if seed == 3801:
                return good()
            if seed == 3802:  # main passes, MTP row ties: still rejected (main-only is not enough)
                return [self._row("main", 5, [1.0, 0.5, 0.2]), self._row("mtp", 5, [1.0, 0.5, 0.5])]
            if seed == 3803:  # a main row ties
                return [self._row("main", 5, [1.0, 0.5, 0.5]), self._row("mtp", 5, [1.0, 0.5, 0.2])]
            if seed == 3804:  # kth == 0
                return [self._row("main", 5, [1.0, 0.0, 0.0]), self._row("mtp", 5, [1.0, 0.5, 0.2])]
            return good()

        # 3800 is not special here: the search starts at the requested seed and 3801 is the first accepted one
        result = self._screen(lambda s: per_seed(3803 if s == 3800 else s), tp.ProfileOptions(seed=3800))
        self.assertEqual((result["start_seed"], result["effective_seed"], result["rejected_count"]), (3800, 3801, 1))
        self.assertEqual(result["rule"], "tie-screened seed")
        self.assertEqual([item["seed"] for item in result["rejected"]], [3800])
        self.assertEqual(result["rejected"][0]["first_failing_row"]["source"], "main")
        self.assertNotIn("block_scores", result["rejected"][0]["first_failing_row"])
        result = self._screen(per_seed, tp.ProfileOptions(seed=3802))
        self.assertEqual((result["start_seed"], result["effective_seed"], result["rejected_count"]), (3802, 3805, 3))
        self.assertEqual([item["seed"] for item in result["rejected"]], [3802, 3803, 3804])
        self.assertEqual(result["rejected"][0]["first_failing_row"]["source"], "mtp")
        self.assertTrue(all(row["passed"] for row in result["rows"]))
        self.assertEqual({row["source"] for row in result["rows"]}, {"main", "mtp"})

    def test_search_has_no_fallback_and_requires_main_and_mtp_rows(self) -> None:
        tie = lambda seed: [self._row("main", 5, [1.0, 0.5, 0.5]), self._row("mtp", 5, [1.0, 0.5, 0.2])]
        with self.assertRaisesRegex(FixtureError, "no seed in"):
            self._screen(tie, tp.ProfileOptions(seed=3801), max_seeds=3)
        main_only = lambda seed: [self._row("main", 5, [1.0, 0.5, 0.2])]
        with self.assertRaisesRegex(FixtureError, "needs selecting rows"):
            self._screen(main_only, tp.ProfileOptions(seed=3801))
        self._screen(main_only, tp.ProfileOptions(seed=3801, with_mtp=False))  # no MTP head: main rows are all there is
        for bad in (0, -1, True, 1.5):
            with self.assertRaisesRegex(FixtureError, "max seeds"):
                tp.screen_qsa_seed(tp.ProfileOptions(), None, [], max_seeds=bad)

    def test_cli_screening_flags_are_opt_in(self) -> None:
        parsed = generate_fixtures._parser().parse_args(["--tiny-profile", "tiny-dense-dev", "--out", "x"])
        self.assertEqual((parsed.screen_qsa_ties, parsed.screen_qsa_max_seeds, parsed.seed), (False, None, 3800))
        parsed = generate_fixtures._parser().parse_args(["--tiny-profile", "tiny-dense-dev", "--out", "x", "--seed", "3801", "--screen-qsa-ties", "--screen-qsa-max-seeds", "64"])
        self.assertEqual((parsed.screen_qsa_ties, parsed.screen_qsa_max_seeds, generate_fixtures._tiny_options(parsed).seed), (True, 64, 3801))
        with tempfile.TemporaryDirectory() as temp:
            out = str(Path(temp) / "x")
            self.assertEqual(generate_fixtures.main(["--out", out, "--screen-qsa-ties"]), 2)  # needs --tiny-profile
            self.assertEqual(generate_fixtures.main(["--tiny-profile", "tiny-dense-dev", "--out", out, "--screen-qsa-max-seeds", "4"]), 2)  # needs --screen-qsa-ties
        self.assertIs(inspect.signature(tp.generate_profile).parameters["qsa_tie_screen"].default, False)


class SourcePinTests(unittest.TestCase):
    def test_hash_mismatch_refuses_even_with_matching_version(self) -> None:
        spec = tp._spec("transformers_modeling")
        good = b"x"
        with self.assertRaisesRegex(FixtureError, "source hash mismatch for transformers_modeling"):
            tp._verify_bytes(good, spec, "test")
        record = tp._verify_bytes(good, {**spec, "sha256": hashlib.sha256(good).hexdigest()}, "test")
        self.assertTrue(record["verified_before_use"])

    @unittest.skipUnless(_FULL_STACK, "torch/transformers/safetensors/tokenizers not importable")
    def test_tampered_pin_is_rejected_before_any_model_use(self) -> None:
        tampered = tuple(
            {**spec, "sha256": "0" * 64} if spec["name"] == "transformers_modeling" else spec
            for spec in tp.UPSTREAM_SOURCE_SPECS
        )
        with mock.patch.object(tp, "UPSTREAM_SOURCE_SPECS", tampered), mock.patch.dict(tp._SOURCES_CACHE, clear=True):
            with self.assertRaisesRegex(FixtureError, "source hash mismatch"):
                tp.load_verified_sources()


@unittest.skipUnless(_FULL_STACK, "torch/transformers/safetensors/tokenizers not importable")
class FullStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.sources = tp.load_verified_sources()
        except FixtureError as exc:
            if "unable to fetch" in str(exc):
                raise unittest.SkipTest(f"pinned vLLM sources unavailable offline: {exc}") from exc
            raise
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.moe_dir = root / "moe"
        cls.dense_dir = root / "dense"
        cls.moe = tp.generate_profile(cls.moe_dir, tp.ProfileOptions(profile_id="tiny-moe"))
        cls.dense = tp.generate_profile(cls.dense_dir, tp.ProfileOptions(profile_id="tiny-dense-dev"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    # -- identity / source ---------------------------------------------------

    def test_sources_verified_before_use(self) -> None:
        for manifest in (self.moe, self.dense):
            records = {item["name"]: item for item in manifest["source_verification"]["records"]}
            for name in ("transformers_configuration", "transformers_modeling", "transformers_modular", "vllm_mtp", "vllm_model", "vllm_common_hyperconnection"):
                self.assertTrue(records[name]["verified_before_use"], name)
                self.assertEqual(records[name]["sha256"], tp._spec(name)["sha256"], name)
        self.assertEqual(self.moe["source_verification"]["environment"]["transformers_commit"], tp.HF_TRANSFORMERS_COMMIT)
        for name in ("config_module", "modeling_module"):
            module = getattr(self.sources, name)
            self.assertEqual(module.__hipfire_verified_sha256__, tp._spec("transformers_" + name.split("_")[0].replace("config", "configuration"))["sha256"])

    def test_profile_labels(self) -> None:
        self.assertEqual(self.moe["hipfire_profile"]["equation_source"], "pinned-upstream")
        self.assertEqual(self.dense["hipfire_profile"]["equation_source"], "non-upstream-dense")
        self.assertEqual(self.moe["equation_sources"]["mtp"], "pinned-vllm-mtp-forward+hf-operators")
        self.assertIn("NON-UPSTREAM", self.dense["equation_sources"]["ffn"])
        self.assertIsNone(self.moe["dense_boundary"])
        self.assertIn("non-upstream", self.dense["dense_boundary"])
        self.assertFalse(self.moe["mtp"]["pinned_hf_mtp_support"])

    def test_config_json_is_the_declared_envelope(self) -> None:
        moe = json.loads((self.moe_dir / "config.json").read_text())
        dense = json.loads((self.dense_dir / "config.json").read_text())
        self.assertEqual(
            moe["hipfire_profile"],
            {**tp.profile_envelope(tp.ProfileOptions(profile_id="tiny-moe")), "ple": {"valid_rows": 18}},
        )
        self.assertEqual(dense["hipfire_profile"]["ffn"], {"kind": "dense", "intermediate_size": 8})
        self.assertNotIn("num_experts", dense["text_config"])
        self.assertEqual(moe["text_config"]["layer_types"], ["linear_attention", "full_attention"])

    def test_tokenizer_round_trip(self) -> None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(self.moe_dir / "tokenizer.json"))
        ids = tokenizer.encode(tp.FIXED_TEXT).ids
        self.assertEqual(ids, self.moe["fixed_tokens"]["ids"])
        self.assertEqual(ids, self.dense["fixed_tokens"]["ids"])
        self.assertIn(2, ids[:-1], "an EOS must precede the last position")
        self.assertEqual(tokenizer.get_vocab_size(), 32)

    # -- inventory -----------------------------------------------------------

    def test_inventory_matches_safetensors_header_and_is_bf16(self) -> None:
        for directory, manifest in ((self.moe_dir, self.moe), (self.dense_dir, self.dense)):
            header = _safetensors_header(directory / "model.safetensors")
            header.pop("__metadata__", None)
            inventory = {item["name"]: item for item in manifest["checkpoint"]["inventory"]}
            self.assertEqual(sorted(header), sorted(inventory))
            for name, item in inventory.items():
                self.assertEqual(header[name]["dtype"], item["dtype"], name)
                self.assertEqual(header[name]["shape"], item["shape"], name)
                self.assertIn(item["dtype"], ("BF16", "I64"), name)
            self.assertEqual(sum(1 for item in inventory.values() if item["dtype"] == "I64"), 3, "PLE hash buffers")

    def test_moe_inventory_has_experts_and_dense_has_none(self) -> None:
        moe = {item["name"] for item in self.moe["checkpoint"]["inventory"]}
        dense = {item["name"] for item in self.dense["checkpoint"]["inventory"]}
        self.assertIn("model.language_model.layers.0.mlp.experts.gate_up_proj", moe)
        self.assertIn("model.language_model.layers.0.mlp.shared_expert_gate.weight", moe)
        self.assertIn("mtp.layers.0.mlp.experts.down_proj", moe)
        for name in dense:
            for fragment in (".experts.", "shared_expert", "mlp.gate.weight"):
                self.assertNotIn(fragment, name, f"dense inventory contains MoE tensor {name}")
        for layer in (0, 1):
            for role in ("gate_proj", "up_proj", "down_proj"):
                self.assertIn(f"model.language_model.layers.{layer}.mlp.{role}.weight", dense)
        self.assertIn("mtp.layers.0.mlp.gate_proj.weight", dense)
        shape = {item["name"]: item["shape"] for item in self.dense["checkpoint"]["inventory"]}
        self.assertEqual(shape["model.language_model.layers.0.mlp.gate_proj.weight"], [8, 16])
        self.assertEqual(shape["model.language_model.layers.0.mlp.down_proj.weight"], [16, 8])

    def test_mtp_and_ple_tensors_are_in_the_inventory(self) -> None:
        names = {item["name"] for item in self.moe["checkpoint"]["inventory"]}
        for required in (
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            "model.language_model.layers.0.ple.ple_embedding.layer_multipliers",
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight",
            "lm_head.weight",
        ):
            self.assertIn(required, names)
        self.assertGreater(self.moe["checkpoint"]["inventory_component_counts"]["vision"], 0)

    def test_ple_shrink_evidence(self) -> None:
        ple = self.moe["shrink"]["ple"]
        self.assertEqual(ple["head_vocab_sizes"], [7, 11])
        self.assertEqual((ple["valid_rows"], ple["padded_rows"], ple["row_width"]), (18, 24, 8))

    # -- dumps ----------------------------------------------------------------

    def test_dump_metadata_shape_and_dtype(self) -> None:
        for directory, manifest in ((self.moe_dir, self.moe), (self.dense_dir, self.dense)):
            names = {item["name"] for item in manifest["dumps"]}
            self.assertEqual(names, {"forward", "states", "mtp_step0"})
            for dump in manifest["dumps"]:
                arrays = read_npz(directory / dump["path"])
                self.assertEqual(hashlib.sha256((directory / dump["path"]).read_bytes()).hexdigest(), dump["sha256"])
                self.assertEqual({item["name"] for item in dump["arrays"]}, set(arrays))
                for item in dump["arrays"]:
                    self.assertIn(item["logical_dtype"], ("bfloat16", "float32", "int64"))
                    self.assertEqual(tuple(item["shape"]), arrays[item["name"]].shape)
            forward = {item["name"]: item for item in next(d for d in manifest["dumps"] if d["name"] == "forward")["arrays"]}
            self.assertEqual(forward["output.logits"]["shape"], [1, 9, 32])
            self.assertEqual(forward["output.logits"]["logical_dtype"], "bfloat16")
            for required in (
                "op.model.language_model.layers.0.ple.ple_embedding.ngram_embedding.in0",
                "op.model.language_model.layers.0.attn_hyper_connection.out2",
                "op.model.language_model.layers.0.linear_attn.fn.torch_chunk_gated_delta_rule.out.0",
                "op.model.language_model.layers.0.linear_attn.fn.causal_conv1d_fn.out",
                "op.model.language_model.layers.1.self_attn.indexer.out0",
                "op.model.language_model.layers.1.self_attn.fn.eager_attention_forward.out.0",
                "op.model.language_model.hyper_connection_mixer.in0",
            ):
                self.assertIn(required, forward)

    def test_moe_and_dense_op_dumps_are_distinct_sources(self) -> None:
        moe = {item["name"] for d in self.moe["dumps"] if d["name"] == "forward" for item in d["arrays"]}
        dense = {item["name"] for d in self.dense["dumps"] if d["name"] == "forward" for item in d["arrays"]}
        self.assertIn("op.model.language_model.layers.0.mlp.gate.out2", moe)
        self.assertIn("op.model.language_model.layers.0.mlp.activation", dense)
        self.assertNotIn("op.model.language_model.layers.0.mlp.gate.out2", dense)

    def test_states_cover_gdn_ple_and_qsa(self) -> None:
        states = {item["name"] for d in self.moe["dumps"] if d["name"] == "states" for item in d["arrays"]}
        for required in (
            "captured.layer0.conv_state0",
            "captured.layer0.conv_state1",
            "captured.layer0.conv_state2",
            "captured.layer0.recurrent_state0",
            "prefill.state.layer0.conv_state0",
            "prefill.state.layer0.conv_state1",
            "prefill.state.layer0.conv_state2",
            "prefill.state.layer0.recurrent_state0",
            "prefill.state.layer1.keys",
            "prefill.state.layer1.indexer_keys",
            "chunked.state.layer0.recurrent_state0",
        ):
            self.assertIn(required, states)
        self.assertTrue(self.moe["chunked_decode"]["supported"])
        self.assertLess(self.moe["chunked_decode"]["max_abs_logit_difference_vs_full_forward"], 0.25)

    def test_mtp_is_a_complete_branch_with_dumps(self) -> None:
        arrays = {item["name"]: item for d in self.moe["dumps"] if d["name"] == "mtp_step0" for item in d["arrays"]}
        for required in (
            "mtp.in.token_ids",
            "mtp.in.backbone_multi_stream",
            "mtp_op.mtp.fc_embedding.out0",
            "mtp_op.mtp.fc_hidden.out0",
            "mtp_op.mtp.pre_fc_norm_hidden.out0",
            "mtp_op.mtp.layers.0.self_attn.indexer.out0",
            "mtp_op.mtp.layers.0.mlp.gate.out2",
            "mtp.layer.hidden_before_mlp_combine",
            "mtp.out.multi_hidden",
            "mtp.out.sample_hidden",
            "mtp.out.logits",
        ):
            self.assertIn(required, arrays)
        self.assertEqual(arrays["mtp.out.logits"]["shape"], [8, 32])
        self.assertEqual(arrays["mtp.out.multi_hidden"]["shape"], [8, 32])
        self.assertEqual(arrays["mtp.out.sample_hidden"]["shape"], [8, 16])
        self.assertIn("forward_self_attributes", self.moe["mtp"]["vllm_contract"])
        self.assertIn("later draft steps", " ".join(self.moe["mtp"]["not_covered"]))
        dense_arrays = {item["name"] for d in self.dense["dumps"] if d["name"] == "mtp_step0" for item in d["arrays"]}
        self.assertIn("mtp_op.mtp.layers.0.mlp.activation", dense_arrays)

    def test_mtp_input_is_the_main_model_stream(self) -> None:
        directory = self.moe_dir
        forward = read_npz(directory / "forward.npz")
        mtp = read_npz(directory / "mtp_step0.npz")
        mixer_in = forward["op.model.language_model.hyper_connection_mixer.in0"]
        stream = mtp["mtp.in.backbone_multi_stream"]
        width = mixer_in.shape[-1]
        self.assertEqual(stream.data, mixer_in.data[: (mixer_in.shape[1] - 1) * width])
        self.assertEqual(list(mtp["mtp.in.token_ids"].data), self.moe["fixed_tokens"]["ids"][1:])

    def test_gpu_rope_variant_reaches_main_and_cloned_mtp_layer(self) -> None:
        variant = tp.ProfileOptions(profile_id="tiny-moe", rope_theta=10_000_000.0, max_position_embeddings=1024)
        assembly = tp.build_assembly(variant, self.sources, seed=variant.seed)
        evidence = tp.rope_evidence(assembly)
        self.assertEqual((evidence["rope_theta"], evidence["max_position_embeddings"]), (10_000_000.0, 1024))
        mtp_attention_config = assembly.mtp_head.layers[0].self_attn.config
        self.assertEqual(mtp_attention_config.rope_parameters["rope_theta"], 10_000_000.0)
        self.assertEqual(mtp_attention_config.max_position_embeddings, 1024)
        self.assertEqual(assembly.mtp_config.rope_parameters["rope_theta"], 10_000_000.0)
        self.assertEqual(assembly.text_model.rotary_emb.config.max_position_embeddings, 1024)
        # A clone that kept the JSON value but not the effective rope config is refused.
        assembly.mtp_config.rope_parameters = {**assembly.mtp_config.rope_parameters, "rope_theta": 10000.0}
        with self.assertRaisesRegex(FixtureError, "mtp_config has rope_theta=10000.0"):
            tp.rope_evidence(assembly)

    def test_gpu_rope_variant_profile_output_and_default_manifest_unchanged(self) -> None:
        self.assertNotIn("rope", self.moe)
        self.assertNotIn("rope_theta", self.moe["options"])
        self.assertNotIn("max_position_embeddings", self.moe["options"])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "gpu"
            options = tp.ProfileOptions(profile_id="tiny-moe", rope_theta=10_000_000.0, max_position_embeddings=1024)
            manifest = tp.generate_profile(out, options, controls=False)
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["hipfire_profile"]["id"], "tiny-moe")
            text = config["text_config"]
            self.assertEqual((text["rope_parameters"]["rope_theta"], text["max_position_embeddings"]), (10_000_000.0, 1024))
            self.assertEqual(text["rope_parameters"]["partial_rotary_factor"], 0.5)
            self.assertEqual(text["mtp"]["rope_theta"], 10_000_000.0)
            self.assertEqual(json.loads((out / "tokenizer_config.json").read_text())["model_max_length"], 1024)
            self.assertEqual(manifest["options"]["rope_theta"], 10_000_000.0)
            self.assertEqual(manifest["options"]["max_position_embeddings"], 1024)
            self.assertEqual(manifest["rope"]["rope_theta"], 10_000_000.0)
            self.assertEqual(manifest["rope"]["max_position_embeddings"], 1024)
            self.assertIn("mtp_layer_self_attn_config", manifest["rope"]["checked"])
            self.assertEqual(manifest["fixed_tokens"], self.moe["fixed_tokens"])
            self.assertEqual(manifest["seed"], self.moe["seed"])
            for name in ("forward.npz", "mtp_step0.npz"):
                self.assertNotEqual((out / name).read_bytes(), (self.moe_dir / name).read_bytes(), name)
            with self.assertRaisesRegex(FixtureError, "below the 9 fixed tokens"):
                tp.generate_profile(Path(temp) / "short", tp.ProfileOptions(max_position_embeddings=8), controls=False)

    def test_default_manifest_has_no_optional_component_keys(self) -> None:
        for manifest in (self.moe, self.dense):
            self.assertNotIn("optional_components", manifest)
            self.assertNotIn("with_ple", manifest["options"])
            self.assertNotIn("with_mtp", manifest["options"])
            self.assertIn("mtp", manifest["equation_sources"])
            self.assertTrue(manifest["mtp"]["pinned_hf_mtp_support"] is False and "vllm_contract" in manifest["mtp"])
            self.assertEqual([item["name"] for item in manifest["dumps"]], ["forward", "states", "mtp_step0"])

    def test_without_ple_variant_has_no_ple_tensors_buffers_markers_or_states(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-dense-dev", rope_theta=10_000_000.0, max_position_embeddings=1024, with_ple=False)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "nople"
            manifest = tp.generate_profile(out, options)
            config = json.loads((out / "config.json").read_text())
            text = config["text_config"]
            self.assertEqual(text["ple_layer_ids"], [])
            for key in tp.PLE_COMPANION_KEYS:
                self.assertNotIn(key, text)
            self.assertNotIn("qwen4_ple", config)
            self.assertNotIn("ple", config["hipfire_profile"])
            self.assertIn("mtp", text)  # MTP is kept
            names = {item["name"] for item in manifest["checkpoint"]["inventory"]}
            self.assertEqual(names, set(_safetensors_header(out / "model.safetensors")) - {"__metadata__"})
            for name in names:
                for fragment in (".ple.", "ngram", "layer_multipliers"):
                    self.assertNotIn(fragment, name)
            self.assertNotIn("ple", manifest["checkpoint"]["inventory_component_counts"])
            self.assertIn("mtp.fc_embedding.weight", names)
            self.assertIsNone(manifest["shrink"]["ple"])
            for dump in ("forward", "states", "mtp_step0"):
                for name in _dump_names(manifest, dump):
                    self.assertNotIn(".ple.", name, dump)
            states = _dump_names(manifest, "states")
            for name in states:
                self.assertNotRegex(name, r"conv_state[12]$")
            for required in ("captured.layer0.conv_state0", "captured.layer0.recurrent_state0", "prefill.state.layer0.conv_state0", "chunked.state.layer0.recurrent_state0"):
                self.assertIn(required, states)
            self.assertTrue(manifest["chunked_decode"]["supported"])
            components = manifest["optional_components"]
            self.assertEqual(components["ple"]["present"], False)
            self.assertEqual((components["ple"]["layers"], components["ple"]["inventory_tensors"], components["ple"]["cache_state_slots"]), ([], 0, 1))
            self.assertEqual(components["ple"]["text_config_ple_layer_ids"], [])
            self.assertTrue(components["mtp"]["present"] and components["mtp"]["inventory_tensors"] > 0)
            self.assertEqual(manifest["options"]["with_ple"], False)
            self.assertNotIn("with_mtp", manifest["options"])
            controls = {item["name"]: item for item in manifest["negative_controls"]}
            for name in ("ple_eos_history", "ple_cached_history"):
                self.assertEqual((controls[name]["status"], controls[name]["passed"], controls[name]["reason"]), ("skipped", None, tp.PLE_ABSENT_REASON))
            for name in ("identity_rebuild", "gdn_output_gate", "dense_ffn_rounding", "dense_gate_projection"):
                self.assertTrue(controls[name]["passed"], name)
            self.assertIn("mtp_layer_self_attn_config", manifest["rope"]["checked"])

    def test_without_mtp_variant_builds_no_mtp_and_keeps_the_backbone_bit_identical(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-moe", with_mtp=False)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "nomtp"
            manifest = tp.generate_profile(out, options, controls=False)
            config = json.loads((out / "config.json").read_text())
            text = config["text_config"]
            for key in tp.MTP_CONFIG_KEYS:
                self.assertNotIn(key, text)
            self.assertEqual(text["ple_layer_ids"], [1])  # PLE kept, marker kept
            self.assertIn("qwen4_ple", config)
            names = {item["name"] for item in manifest["checkpoint"]["inventory"]}
            self.assertEqual(names, set(_safetensors_header(out / "model.safetensors")) - {"__metadata__"})
            self.assertFalse([name for name in names if name.startswith("mtp.")])
            self.assertNotIn("mtp", manifest["checkpoint"]["inventory_component_counts"])
            self.assertIn("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", names)
            # No empty or fabricated MTP dump: only the actual forward and states exist.
            self.assertEqual([item["name"] for item in manifest["dumps"]], ["forward", "states"])
            self.assertEqual(sorted(path.name for path in out.glob("*.npz")), ["forward.npz", "states.npz"])
            self.assertEqual(manifest["mtp"]["present"], False)
            self.assertEqual(manifest["mtp"]["constructed"], False)
            self.assertNotIn("vllm_contract", manifest["mtp"])
            self.assertNotIn("mtp", manifest["equation_sources"])
            components = manifest["optional_components"]
            self.assertEqual((components["mtp"]["present"], components["mtp"]["inventory_tensors"]), (False, 0))
            self.assertEqual(components["mtp"]["omitted"]["dumps"], ["mtp_step0"])
            self.assertEqual((components["ple"]["present"], components["ple"]["layers"], components["ple"]["cache_state_slots"]), (True, [0], 3))
            states = _dump_names(manifest, "states")
            for required in ("captured.layer0.conv_state1", "captured.layer0.conv_state2", "prefill.state.layer0.conv_state2"):
                self.assertIn(required, states)
            # The backbone is the same model: per-tensor name-keyed init makes forward/states bit-identical to the default.
            for name in ("forward.npz", "states.npz"):
                self.assertEqual((out / name).read_bytes(), (self.moe_dir / name).read_bytes(), name)

    def test_without_both_constructs_no_mtp_and_runs_every_remaining_control(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-dense-dev", rope_theta=10_000_000.0, max_position_embeddings=1024, with_ple=False, with_mtp=False)
        with mock.patch.object(tp, "_make_mtp_head_cls", side_effect=AssertionError("MTP head constructed")) as factory:
            assembly = tp.build_assembly(options, self.sources, seed=options.seed)
        factory.assert_not_called()
        self.assertIsNone(assembly.mtp_head)
        self.assertIsNone(assembly.mtp_config)
        self.assertFalse([name for name, _ in assembly.model.named_parameters() if name.startswith("mtp")])
        self.assertEqual(tp.shrink_evidence(assembly)["parameter_elements"], sum(parameter.numel() for parameter in assembly.model.parameters()))
        self.assertIsNone(tp.derived_ple_geometry(assembly))
        self.assertTrue(all(layer.ple is None for layer in assembly.text_model.layers))
        evidence = tp.rope_evidence(assembly)
        self.assertEqual(evidence["checked"], ["main_rotary_embedding_config", "main_text_config", "rotary_inv_freq_recomputed"])
        self.assertNotIn("mtp_rotary_source", evidence)
        with self.assertRaisesRegex(FixtureError, "no MTP head"):
            tp.run_mtp_step0(assembly, None)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bare"
            manifest = tp.generate_profile(out, options)
            self.assertEqual([item["name"] for item in manifest["dumps"]], ["forward", "states"])
            controls = {item["name"]: item for item in manifest["negative_controls"]}
            self.assertEqual({name for name, item in controls.items() if item.get("status") == "skipped"}, {"ple_eos_history", "ple_cached_history"})
            for name, item in controls.items():
                if item.get("status") != "skipped":
                    self.assertTrue(item["passed"], name)
            self.assertEqual(manifest["optional_components"]["ple"]["present"], False)
            self.assertEqual(manifest["optional_components"]["mtp"]["present"], False)
            self.assertEqual((manifest["options"]["with_ple"], manifest["options"]["with_mtp"]), (False, False))
            names = {item["name"] for item in manifest["checkpoint"]["inventory"]}
            self.assertFalse([name for name in names if name.startswith("mtp.") or ".ple." in name])

    def test_dense_equation_agrees_with_reusable_mlp_math(self) -> None:
        check = self.dense["dense_equation_cross_check"]
        self.assertLessEqual(check["max_abs_difference"], 2.0**-7 * 4)
        self.assertIn("NOT upstream", check["note"])
        self.assertIsNone(self.moe["dense_equation_cross_check"])

    # -- negative controls ----------------------------------------------------

    def test_negative_controls_all_passed(self) -> None:
        moe = {item["name"]: item for item in self.moe["negative_controls"]}
        dense = {item["name"]: item for item in self.dense["negative_controls"]}
        for name in ("identity_rebuild", "gdn_output_gate", "ple_eos_history", "ple_cached_history", "moe_router_gate", "moe_norm_topk_prob", "bf16_vs_f32_rounding"):
            self.assertTrue(moe[name]["passed"], name)
        for name in ("identity_rebuild", "gdn_output_gate", "ple_eos_history", "ple_cached_history", "dense_ffn_rounding", "dense_gate_projection"):
            self.assertTrue(dense[name]["passed"], name)
        self.assertEqual(moe["identity_rebuild"]["changed_count"], 0)
        for name in ("moe_router_gate", "moe_norm_topk_prob", "ple_cached_history"):
            self.assertGreater(moe[name]["changed_count"], 0, name)

    def test_changed_gate_changes_dumps(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-moe")
        tokens = tp.build_tokenizer(options)[2]
        seed = self.moe["seed"]["effective"]
        base = tp.run_forward(tp.build_assembly(options, self.sources, seed=seed), tokens)
        assembly = tp.build_assembly(options, self.sources, seed=seed)
        torch = assembly.torch
        with torch.no_grad():
            assembly.text_model.layers[0].mlp.gate.weight.copy_(assembly.text_model.layers[0].mlp.gate.weight.flip(0))
        changed = tp.run_forward(assembly, tokens)
        keys = tp.changed_keys(base.dumps, changed.dumps, torch)
        self.assertIn("op.model.language_model.layers.0.mlp.gate.out0", keys)
        self.assertIn("output.logits", keys)
        again = tp.run_forward(tp.build_assembly(options, self.sources, seed=seed), tokens)
        self.assertEqual(tp.changed_keys(base.dumps, again.dumps, torch), [])

    def test_changed_dense_rounding_changes_dumps(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-dense-dev")
        tokens = tp.build_tokenizer(options)[2]
        seed = self.dense["seed"]["effective"]
        assembly = tp.build_assembly(options, self.sources, seed=seed)
        base = tp.run_forward(assembly, tokens)
        cls = assembly.dense_cls
        cls.round_bf16 = False
        try:
            wide = tp.run_forward(assembly, tokens)
        finally:
            cls.round_bf16 = True
        keys = tp.changed_keys(base.dumps, wide.dumps, assembly.torch)
        self.assertIn("op.model.language_model.layers.0.mlp.activation", keys)

    def test_changed_ple_history_changes_dumps(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-moe")
        tokens = tp.build_tokenizer(options)[2]
        seed = self.moe["seed"]["effective"]
        assembly = tp.build_assembly(options, self.sources, seed=seed)
        base = tp.run_forward(assembly, tokens)
        eos = options.eos_token_id
        replaced = [tokens[0] if value == eos else value for value in tokens]
        other = tp.run_forward(assembly, replaced)
        keys = tp.changed_keys(base.dumps, other.dumps, assembly.torch)
        self.assertIn("op.model.language_model.layers.0.ple.ple_embedding.ngram_embedding.in0", keys)
        self.assertIn("op.model.language_model.layers.0.ple.out0", keys)

    # -- determinism / guards ---------------------------------------------------

    def test_generation_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            again = Path(temp) / "again"
            tp.generate_profile(again, tp.ProfileOptions(profile_id="tiny-moe"), controls=False)
            for name in ("config.json", "tokenizer.json", "model.safetensors", "forward.npz", "states.npz", "mtp_step0.npz"):
                self.assertEqual(
                    hashlib.sha256((self.moe_dir / name).read_bytes()).hexdigest(),
                    hashlib.sha256((again / name).read_bytes()).hexdigest(),
                    name,
                )

    def test_qsa_tap_captures_real_selecting_rows_and_defaults_are_unscreened(self) -> None:
        options = tp.ProfileOptions(profile_id="tiny-dense-dev")
        tokens = tp.build_tokenizer(options)[2]
        rows = tp.qsa_candidate_rows(options, self.sources, tokens, 3800)
        main = [row for row in rows if row["source"] == "main"]
        mtp = [row for row in rows if row["source"] == "mtp"]
        # budget 4 / ratio 2 -> block_topk 2: a query selects once it sees >= 3 complete blocks (>= 6 visible tokens).
        self.assertEqual([(row["layer"], row["position"], row["complete_blocks"]) for row in main], [(1, 5, 3), (1, 6, 3), (1, 7, 4), (1, 8, 4)])
        self.assertEqual([(row["layer"], row["position"], row["complete_blocks"]) for row in mtp], [(0, 5, 3), (0, 6, 3), (0, 7, 4)])
        for row in rows:
            self.assertEqual(row["block_topk"], 2)
            self.assertEqual(len(row["block_scores"]), row["complete_blocks"])
            self.assertEqual(row["kth"], sorted(row["block_scores"], reverse=True)[1])
        # unscreened generation leaves the manifest untouched (seed used exactly, no screen key)
        self.assertNotIn("qsa_tie_screen", self.dense)
        self.assertEqual(self.dense["seed"]["requested"], self.dense["seed"]["effective"])

    def test_seed_is_used_exactly(self) -> None:
        self.assertEqual(self.moe["seed"]["requested"], 3800)
        self.assertEqual(self.moe["seed"]["effective"], 3800)
        self.assertEqual(self.dense["seed"]["effective"], 3800)
        self.assertNotIn("tie_free_attempt", self.moe["seed"])

    def test_captured_states_equal_dynamic_cache_states(self) -> None:
        for directory in (self.moe_dir, self.dense_dir):
            states = read_npz(directory / "states.npz")
            for slot in ("conv_state0", "conv_state1", "conv_state2", "recurrent_state0"):
                captured = states[f"captured.layer0.{slot}"]
                cached = states[f"prefill.state.layer0.{slot}"]
                self.assertEqual(captured.shape, cached.shape, slot)
                self.assertEqual(captured.dtype, cached.dtype, slot)
                self.assertEqual(captured.data, cached.data, f"{slot}: helper-capture state must equal the DynamicCache state")

    def test_config_carries_source_derived_ple_marker(self) -> None:
        def independent_multipliers(seed: int, vocab: int, layer_index: int, count: int) -> list[int]:
            mask = (1 << 64) - 1
            bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
            base = seed + 10007 * layer_index
            result = []
            for index in range(count):
                value = (base + 0x9E3779B97F4A7C15 * (index + 1)) & mask
                value = (value + 0x9E3779B97F4A7C15) & mask
                value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
                value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
                value = (value ^ (value >> 31)) & mask
                result.append(2 * (value % bound) + 1)
            return result

        for directory, manifest in ((self.moe_dir, self.moe), (self.dense_dir, self.dense)):
            config = json.loads((directory / "config.json").read_text())
            self.assertEqual(config["hipfire_profile"]["ple"], {"valid_rows": 18})
            self.assertEqual(
                config["qwen4_ple"],
                {
                    "version": 1,
                    "multipliers": independent_multipliers(1234, 32, 0, 3),
                    "head_vocab_sizes": [7, 11],
                    "head_offsets": [0, 7],
                    "padded_rows": 24,
                },
            )
            self.assertEqual(set(config["qwen4_ple"]), {"version", "multipliers", "head_vocab_sizes", "head_offsets", "padded_rows"})
            shrink = manifest["shrink"]["ple"]
            self.assertEqual(config["qwen4_ple"]["multipliers"], shrink["layer_multipliers"])
            self.assertEqual(config["qwen4_ple"]["padded_rows"], shrink["padded_rows"])
            inventory = {item["name"]: item["shape"] for item in manifest["checkpoint"]["inventory"]}
            shard = inventory["model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"]
            self.assertEqual(shard, [config["qwen4_ple"]["padded_rows"], 8])
            self.assertEqual(config["text_config"]["split_ngram_parts"], 1)

    def test_vllm_contract_drift_is_rejected(self) -> None:
        contract = tp.verify_vllm_mtp_contract(self.sources.vllm_text)
        self.assertIn("pre_fc_norm_hidden", contract["forward_self_attributes"])
        drifted = dict(self.sources.vllm_text)
        drifted["vllm_mtp"] = drifted["vllm_mtp"].replace("hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states", "hidden_states = hidden_states + inputs_embeds.unsqueeze(-2)")
        with self.assertRaisesRegex(FixtureError, "no longer contains"):
            tp.verify_vllm_mtp_contract(drifted)
        attribute_drift = dict(self.sources.vllm_text)
        attribute_drift["vllm_mtp"] = attribute_drift["vllm_mtp"].replace("inputs_embeds = self.fc_embedding(inputs_embeds)", "inputs_embeds = inputs_embeds")
        with self.assertRaisesRegex(FixtureError, "self-attribute set drifted"):
            tp.verify_vllm_mtp_contract(attribute_drift)

    def test_cli_exclusivity_and_tiny_profile(self) -> None:
        self.assertEqual(generate_fixtures.main(["--tiny-profile", "tiny-moe", "--upstream-reference", "--out", "/nonexistent"]), 2)
        with tempfile.TemporaryDirectory() as temp:
            code = generate_fixtures.main(["--tiny-profile", "tiny-dense-dev", "--out", str(Path(temp) / "x"), "--no-controls"])
            self.assertEqual(code, 0)
            manifest = json.loads((Path(temp) / "x" / "manifest.json").read_text())
            self.assertEqual(manifest["negative_controls"], [])
            self.assertEqual(manifest["hipfire_profile"]["id"], "tiny-dense-dev")

    def test_width_5120_single_layer_knob(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = tp.ProfileOptions(profile_id="tiny-dense-dev", hidden_size=5120, num_layers=1)
            manifest = tp.generate_profile(Path(temp) / "w", options, controls=False)
            shapes = {item["name"]: item["shape"] for item in manifest["checkpoint"]["inventory"]}
            self.assertEqual(shapes["model.language_model.embed_tokens.weight"], [32, 5120])
            self.assertEqual(shapes["model.language_model.layers.0.mlp.gate_proj.weight"], [8, 5120])
            self.assertEqual(shapes["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"], [4, 10240])
            parameters = manifest["shrink"]["parameter_elements"]
            self.assertGreater(parameters, 2 * 5120 * 5120, "the two [H, H] MTP fusion projections are the structural floor")
            self.assertLess(parameters, tp.MAX_PARAMETER_ELEMENTS)
            self.assertFalse(manifest["chunked_decode"]["supported"])
            self.assertEqual({item["name"] for item in manifest["dumps"]}, {"forward", "states", "mtp_step0"})
            states = {item["name"]: item for d in manifest["dumps"] if d["name"] == "states" for item in d["arrays"]}
            self.assertEqual(
                sorted(states),
                [
                    "captured.layer0.conv_state0",
                    "captured.layer0.conv_state1",
                    "captured.layer0.conv_state2",
                    "captured.layer0.recurrent_state0",
                ],
            )
            self.assertEqual(states["captured.layer0.conv_state0"]["shape"], [1, 32, 2])
            self.assertEqual(states["captured.layer0.conv_state1"]["shape"], [1, 10240, 3])
            self.assertEqual(states["captured.layer0.conv_state2"]["shape"], [1, 2])
            self.assertEqual(manifest["seed"]["effective"], 3800)


if __name__ == "__main__":
    unittest.main()
