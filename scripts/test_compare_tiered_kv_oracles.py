#!/usr/bin/env python3
"""CPU fake-evidence tests; no GPU and no model quality/performance claims."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("compare_tiered_kv_oracles.py")
SPEC = importlib.util.spec_from_file_location("compare_tiered_kv_oracles", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(arm):
    state = {key: sha(key) for key in COMPARE.STATE_HASHES}
    prefill = dict(state, visible_rows=1024)
    final = dict(state, visible_rows=1279 if arm in ("baseline", "identity") else 511)
    trace = [{"step": i, "token": i + 100, "logits_sha256": sha(str(i))} for i in range(256)]
    data = {"schema": COMPARE.SCHEMA, "pass": True, "mode": arm,
            "scope": "lab positional/state oracle; not serving/performance/quality admission",
            "model": "model.hfq", "model_sha256": sha("model"),
            "oracle_binary_sha256": sha("oracle-binary"), "prompt_file": "prompt.txt",
            "user_prompt_sha256": sha("user"), "rendered_prompt_sha256": sha("rendered"),
            "prompt_tokens_sha256": sha("tokens"), "prompt_tokens": 1024, "max_seq": 1280,
            "requested_decode_tokens": 256, "generated_tokens": 256, "finish_reason": "length",
            "route": {"arch": "gfx1100", "bdf": "0000:03:00.0", "kv_backend": "legacy",
                      "kv_backend_legacy": True, "explicit_backend": "HIPFIRE_KV_BACKEND=legacy",
                      "kv_mode": "q8/q8", "dn_quant": "Q8", "dn_ef": True,
                      "managed_limit_bytes": 20 * 1024 ** 3, "initial_physical_cap": 1280},
            "prefill_logits_sha256": sha("0"), "prefill_state": prefill, "final_state": final,
            "first_new_fa_k_sha256": sha("wrong" if arm == "wrong-rope" else "correct"),
            "managed_device_bytes_after_decode": 100, "managed_host_bytes_after_decode": 200,
            "text": "decoded text", "trace": trace, "tiered": None}
    if arm in ("bounded", "wrong-rope"):
        data["tiered"] = {"selected_block_starts": [0, 896], "selected_rows": 256,
                          "partial_prompt_rows": 0, "host_archive_bytes": 1280 * 34 * 4,
                          "committed_archived_rows": 1024, "fixed_kv_bytes": 512 * 34 * 4,
                          "physical_cap": 512, "initial_prefill_bounded": False,
                          "decode_restaging": False,
                          "staged_history_verified": True,
                          "staged_history_verified_bytes": 256 * 34 * 4,
                          "staged_history_verified_sides": 4,
                          "row_bytes_per_side": 34, "full_attention_layers": 2}
        # Later sparse attention is allowed to diverge from baseline tokens/logits.
        data["trace"][1]["token"] = 999
        data["trace"][1]["logits_sha256"] = sha("sparse")
        data["text"] = "intentionally different sparse output"
    return data


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {arm: self.root / (arm + ".json") for arm in COMPARE.ARMS}
        self.data = {arm: fixture(arm) for arm in COMPARE.ARMS}
        self.out = self.root / "comparison.json"

    def invoke(self, arms=COMPARE.ARMS, guards=None):
        argv = [sys.executable, str(SCRIPT), "--out", str(self.out)]
        originals = {}
        for arm in arms:
            raw = json.dumps(self.data[arm]).encode()
            self.paths[arm].write_bytes(raw)
            originals[self.paths[arm]] = raw
            argv += ["--" + arm, str(self.paths[arm])]
        for arm, receipt in (guards or {}).items():
            path = self.root / (arm + ".guard.json")
            path.write_text(json.dumps(receipt))
            originals[path] = path.read_bytes()
            argv += ["--guard", arm + "=" + str(path)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw, "input evidence was modified")
        return result, json.loads(self.out.read_text())

    def guard(self, arm):
        return {"schema_version": 1, "guard_kind": "sampled_not_hard_limit", "status": "completed",
                "exit_code": 0, "child_exit_code": 0, "pci_bdf": "0000:03:00.0",
                "command": ["oracle", "--mode", arm, "--out", str(self.paths[arm])],
                "limit_bytes": 1000, "safety_margin_bytes": 100, "guard_limit_bytes": 900,
                "sample_interval_seconds": .05, "baseline_bytes": 100, "peak_bytes": 200,
                "peak_delta_bytes": 100,
                "samples": [{"elapsed_seconds": 0, "vram_used_bytes": 100, "raw": "100\n"},
                            {"elapsed_seconds": .05, "vram_used_bytes": 200, "raw": "200\n"}]}

    def assert_failure(self, contains=None, **kwargs):
        result, report = self.invoke(**kwargs)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIs(report["pass"], False)
        self.assertIn("error", report)
        if contains:
            self.assertIn(contains, report["error"])
        self.out.unlink()
        return report

    def test_full_success_with_divergent_sparse_output_and_guard(self):
        result, report = self.invoke(guards={arm: self.guard(arm) for arm in COMPARE.ARMS})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["pass"])
        self.assertIn("NOT a quality, serving, or performance gate", report["scope"])
        self.assertFalse(report["sparse_later_tokens_required_equal"])
        self.assertEqual(report["oracle_schema"], "hipfire.tiered_kv_model.v2")
        self.assertEqual(len(report["inputs"]), 8)

    def test_identity_only_is_valid_without_claiming_guard(self):
        result, report = self.invoke(arms=("baseline", "identity"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["guard_evidence"], {"baseline": "not_supplied", "identity": "not_supplied"})

    def test_wrong_rope_short_natural_eos_is_valid(self):
        wrong = self.data["wrong-rope"]
        wrong.update(trace=wrong["trace"][:2], generated_tokens=2, finish_reason="stop")
        wrong["final_state"]["visible_rows"] = 257
        result, _ = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_normal_arms_short_natural_eos_is_insufficient(self):
        for arm in ("baseline", "identity", "bounded"):
            with self.subTest(arm=arm):
                original = copy.deepcopy(self.data[arm])
                self.data[arm].update(trace=self.data[arm]["trace"][:2], generated_tokens=2,
                                      finish_reason="stop")
                self.assert_failure("generated_tokens")
                self.data[arm] = original

    def test_trace_length_step_token_and_sha_are_checked(self):
        mutations = [("trace", []), ("trace", self.data["identity"]["trace"][:-1])]
        for key, value in mutations:
            original = self.data["identity"][key]
            self.data["identity"][key] = value
            self.assert_failure("trace length")
            self.data["identity"][key] = original
        for key, value in (("step", 3), ("step", True), ("token", -1), ("token", True),
                           ("logits_sha256", "not-a-hash")):
            original = self.data["identity"]["trace"][0][key]
            self.data["identity"]["trace"][0][key] = value
            self.assert_failure()
            self.data["identity"]["trace"][0][key] = original

    def test_self_reported_pass_is_not_sufficient(self):
        for arm in COMPARE.ARMS:
            self.data[arm]["pass"] = False
            self.assert_failure("pass=true")
            self.data[arm]["pass"] = True
        self.data["identity"]["trace"][200]["token"] = 4242
        self.assert_failure("strict baseline mismatch in trace")

    def test_different_model_or_prompt_rejected(self):
        for key in ("model_sha256", "user_prompt_sha256", "rendered_prompt_sha256", "prompt_tokens_sha256"):
            original = self.data["bounded"][key]
            self.data["bounded"][key] = sha("different")
            self.assert_failure(key)
            self.data["bounded"][key] = original

    def test_oracle_binary_sha_required_valid_and_same_across_arms(self):
        for arm in COMPARE.ARMS:
            original = self.data[arm].pop("oracle_binary_sha256")
            self.assert_failure("oracle_binary_sha256")
            self.data[arm]["oracle_binary_sha256"] = "malformed-hash"
            self.assert_failure("oracle_binary_sha256")
            self.data[arm]["oracle_binary_sha256"] = original
        for arm in ("identity", "bounded", "wrong-rope"):
            original = self.data[arm]["oracle_binary_sha256"]
            self.data[arm]["oracle_binary_sha256"] = sha("another-oracle-binary")
            self.assert_failure("baseline mismatch in oracle_binary_sha256")
            self.data[arm]["oracle_binary_sha256"] = original

    def test_required_fields_and_schema(self):
        for key in ("route", "final_state", "first_new_fa_k_sha256", "trace", "schema", "mode"):
            value = self.data["identity"].pop(key)
            self.assert_failure()
            self.data["identity"][key] = value
        self.data["identity"]["schema"] = "hipfire.tiered_kv_comparison.v1"
        self.assert_failure("schema")

    def test_strict_state_and_first_k_identity(self):
        for key in COMPARE.STATE_HASHES:
            original = self.data["identity"]["final_state"][key]
            self.data["identity"]["final_state"][key] = sha("different")
            self.assert_failure("final_state")
            self.data["identity"]["final_state"][key] = original
        self.data["bounded"]["first_new_fa_k_sha256"] = sha("wrong")
        self.assert_failure("must match baseline")

    def test_wrong_rope_must_be_effective_negative_control(self):
        self.data["wrong-rope"]["first_new_fa_k_sha256"] = self.data["baseline"]["first_new_fa_k_sha256"]
        self.assert_failure("negative control must differ")

    def test_capacity_must_shrink_gap_must_exist_and_visible_fit(self):
        tiered = self.data["bounded"]["tiered"]
        for key, value in (("physical_cap", 1280), ("selected_block_starts", [0, 128]),
                           ("selected_rows", 1024)):
            original = tiered[key]
            tiered[key] = value
            self.assert_failure()
            tiered[key] = original
        self.data["bounded"]["final_state"]["visible_rows"] = 513
        self.assert_failure("exceed physical cap")

    def test_staged_history_fields_required_for_both_bounded_arms(self):
        keys = ("staged_history_verified", "staged_history_verified_bytes",
                "staged_history_verified_sides", "row_bytes_per_side", "full_attention_layers")
        for arm in ("bounded", "wrong-rope"):
            for key in keys:
                with self.subTest(arm=arm, missing=key):
                    value = self.data[arm]["tiered"].pop(key)
                    self.assert_failure(key)
                    self.data[arm]["tiered"][key] = value

    def test_staged_history_false_and_wrong_counts_are_rejected(self):
        mutations = (("staged_history_verified", False), ("staged_history_verified", 1),
                     ("staged_history_verified_bytes", 256 * 34 * 2),
                     ("staged_history_verified_bytes", True),
                     ("staged_history_verified_sides", 2),
                     ("staged_history_verified_sides", 0),
                     ("row_bytes_per_side", 0), ("row_bytes_per_side", 35),
                     ("full_attention_layers", 0), ("full_attention_layers", True),
                     ("fixed_kv_bytes", 512))
        for arm in ("bounded", "wrong-rope"):
            for key, value in mutations:
                with self.subTest(arm=arm, key=key, value=value):
                    original = self.data[arm]["tiered"][key]
                    self.data[arm]["tiered"][key] = value
                    self.assert_failure(key)
                    self.data[arm]["tiered"][key] = original

    def test_staged_history_verification_includes_partial_prompt_tail(self):
        for arm, data in self.data.items():
            data["prompt_tokens"] += 17
            data["max_seq"] += 17
            data["route"]["initial_physical_cap"] += 17
            data["prefill_state"]["visible_rows"] += 17
            data["final_state"]["visible_rows"] += 17
            if data["tiered"]:
                tiered = data["tiered"]
                tiered["partial_prompt_rows"] = 17
                tiered["physical_cap"] += 17
                tiered["fixed_kv_bytes"] = tiered["physical_cap"] * 34 * 4
                tiered["staged_history_verified_bytes"] = (256 + 17) * 34 * 4
        result, _ = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.out.unlink()
        self.data["bounded"]["tiered"]["staged_history_verified_bytes"] = 256 * 34 * 4
        self.assert_failure("staged_history_verified_bytes")

    def test_same_model_bounded_arms_cannot_disagree_on_history_dimensions(self):
        original = copy.deepcopy(self.data["wrong-rope"]["tiered"])
        for key, value in (("row_bytes_per_side", 68), ("full_attention_layers", 3)):
            tiered = self.data["wrong-rope"]["tiered"]
            tiered[key] = value
            sides = tiered["full_attention_layers"] * 2
            tiered["staged_history_verified_sides"] = sides
            tiered["staged_history_verified_bytes"] = 256 * tiered["row_bytes_per_side"] * sides
            tiered["fixed_kv_bytes"] = 512 * tiered["row_bytes_per_side"] * sides
            self.assert_failure("bounded mismatch in " + key)
            self.data["wrong-rope"]["tiered"] = copy.deepcopy(original)

    def test_v1_evidence_is_not_accepted_or_implicitly_upgraded(self):
        self.data["baseline"]["schema"] = "hipfire.tiered_kv_model.v1"
        self.assert_failure("v1 evidence cannot be upgraded")
        self.data["baseline"]["schema"] = COMPARE.SCHEMA
        self.data["bounded"]["schema"] = "hipfire.tiered_kv_model.v1"
        self.data["bounded"]["tiered"].pop("staged_history_verified")
        self.assert_failure("v1 evidence cannot be upgraded")
        self.data["bounded"]["schema"] = COMPARE.SCHEMA
        self.assert_failure("staged_history_verified")

    def test_requested_decode_mismatch_and_short_length(self):
        self.data["bounded"].update(requested_decode_tokens=512, max_seq=1536, finish_reason="stop")
        for data in self.data.values():
            data["route"]["initial_physical_cap"] = 1536
        self.data["bounded"]["tiered"]["physical_cap"] = 768
        self.data["bounded"]["tiered"]["fixed_kv_bytes"] = 768 * 34 * 4
        self.assert_failure("baseline mismatch in requested_decode_tokens")
        self.data["bounded"]["requested_decode_tokens"] = 255
        self.assert_failure("requested_decode_tokens")

    def test_wrong_rope_needs_real_forward_and_trace(self):
        wrong = self.data["wrong-rope"]
        wrong.update(trace=wrong["trace"][:1], generated_tokens=1, finish_reason="stop")
        self.assert_failure("generated_tokens")

    def test_guard_receipt_cannot_stand_in_for_oracle(self):
        self.data["baseline"] = self.guard("baseline")
        self.data["baseline"]["pass"] = True
        self.assert_failure("schema")

    def test_guard_success_cannot_override_oracle_failure(self):
        self.data["baseline"]["pass"] = False
        self.assert_failure("pass=true", guards={"baseline": self.guard("baseline")})

    def test_oracle_pass_cannot_override_guard_failure(self):
        for key, value in (("status", "limit_exceeded"), ("child_exit_code", 23),
                           ("exit_code", 124), ("guard_kind", "not_a_guard"),
                           ("peak_bytes", 100), ("pci_bdf", "0000:04:00.0")):
            guard = self.guard("baseline")
            guard[key] = value
            self.assert_failure(guards={"baseline": guard})

    def test_guard_command_must_bind_arm_and_input(self):
        guard = self.guard("baseline")
        guard["command"][2] = "bounded"
        self.assert_failure("does not match oracle", guards={"baseline": guard})
        guard = self.guard("baseline")
        guard["command"][-1] = str(self.root / "unrelated.json")
        self.assert_failure("does not match oracle", guards={"baseline": guard})

    def test_malformed_json_emits_explicit_failure_json(self):
        self.paths["baseline"].write_text('{"pass":true, "pass":false}')
        self.paths["identity"].write_text("not JSON")
        result = subprocess.run([sys.executable, str(SCRIPT), "--baseline", str(self.paths["baseline"]),
                                 "--identity", str(self.paths["identity"]), "--out", str(self.out)],
                                capture_output=True, text=True, timeout=5)
        report = json.loads(self.out.read_text())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["pass"])
        self.assertIn("duplicate JSON key", report["error"])

    def test_existing_output_and_symlink_are_never_overwritten(self):
        original = '{"external": "unchanged"}\n'
        external = self.root / "external.json"
        external.write_text(original)
        for symlink in (False, True):
            if symlink:
                self.out.symlink_to(external)
            else:
                self.out.write_text(original)
            result, _ = self.invoke(arms=("baseline", "identity"))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.out.read_text(), original)
            self.assertEqual(external.read_text(), original)
            self.out.unlink()


if __name__ == "__main__":
    unittest.main()
