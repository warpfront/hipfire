#!/usr/bin/env python3
"""CPU-only fake-receipt tests; no model, HIP runtime or GPU required."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("compare_tiered_prefill_oracles.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("compare_tiered_prefill_oracles", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(arm, prompt=4099, chunk=127):
    requested, row, sides = 1024, 136, 4
    bounded = arm == "bounded"
    blocks = 2 if bounded else max(2, (prompt + 127) // 128)
    reserve = requested + 127
    cap = blocks * 128 + reserve
    limits = {"device_bytes": 20 << 30, "gtt_bytes": None, "shared_host_bytes": None, "uma": False}
    usage = {"device_bytes": 1024, "host_bytes": 0, "gtt_bytes": 0, "shared_host_bytes": 0}
    ledger = {phase: {"limits": copy.deepcopy(limits), "committed": copy.deepcopy(usage),
                      "pending": {key: 0 for key in usage}}
              for phase in ("after_bounded_load", "after_window_install", "after_prefill",
                            "after_decode", "after_cleanup")}
    prefill = []
    start = 0
    # Deliberately construct fixture policy without calling comparator helpers.
    def policy(count):
        return [i * 128 for i in (range(count) if count <= blocks
                                  else [0] + list(range(count - blocks + 1, count)))]
    def state(end, visible, packed=False):
        return {"dn_sha256": sha((end, "dn")), "hidden_sha256": sha((end, "hidden")),
                "kv_sha256": sha((end, "packed" if packed else "kv")), "visible_rows": visible}
    while start < prompt:
        n = min(chunk, prompt - start)
        end = start + n
        before = policy(start // 128)
        after = policy(end // 128)
        written = end // 128 - start // 128
        packed = len(before) * 128 + start % 128
        sources = [len(before) * 128 + i * 128 for i in range(written)]
        visible = len(after) * 128 + end % 128
        changed = sum(i >= len(before) or value != before[i] for i, value in enumerate(after))
        forward = state(end, packed + n)
        staged = state(end, visible, bounded and end // 128 > blocks)
        prefill.append({"chunk": len(prefill), "absolute_start": start, "absolute_end": end,
                        "tokens": n, "packed_start": packed, "selected_block_starts": after,
                        "selected_rows": len(after) * 128, "visible_rows": visible,
                        "archive_source_rows": sources,
                        "archived_blocks": [{"absolute_start": (start // 128 + i) * 128,
                                             "source_row": source} for i, source in enumerate(sources)],
                        "archive_committed_rows": end // 128 * 128,
                        "archive_host_bytes": end // 128 * 128 * row * sides,
                        "stage_h2d_bytes": changed * 128 * row * sides,
                        "staged_history_verified_bytes": len(after) * 128 * row * sides,
                        "carry_source_row": len(before) * 128 + written * 128,
                        "carry_rows": end % 128, "carry_destination_row": len(after) * 128,
                        "carry_verified_bytes": end % 128 * row * sides,
                        "carry_sha256": sha((end, "carry")) if end % 128 else sha(""),
                        "frontier_published": True,
                        "logits_sha256": sha((end, "logits")), "argmax": 42,
                        "forward_state": forward, "state": staged})
        start = end
    trace = [{"step": i, "token": 42 if i == 0 else 100 + i,
              "logits_sha256": prefill[-1]["logits_sha256"] if i == 0 else sha((i, "decode"))}
             for i in range(requested)]
    return {"schema": COMPARE.SCHEMA, "mode": arm, "pass": True, "model": "model.hfq",
            "prompt_file": "prompt.txt", "scope": "lab only", "text": "decoded output",
            "transient_kv_slabs": "two bounded slabs",
            "model_sha256": sha("model"), "oracle_binary_sha256": sha("binary"),
            "user_prompt_sha256": sha("user"), "rendered_prompt_sha256": sha("rendered"),
            "prompt_tokens_sha256": sha("tokens"), "prompt_tokens": prompt,
            "prefix_records": 32, "chunk_tokens": chunk, "requested_decode_tokens": requested,
            "generated_tokens": requested, "finish_reason": "length", "block_tokens": 128,
            "window_blocks": blocks, "generation_reserve_rows": reserve, "physical_cap": cap,
            "initial_physical_cap": cap, "max_seq": max(prompt + requested, cap + 1),
            "row_bytes_per_side": row, "full_attention_layers": sides // 2,
            "fixed_kv_bytes": cap * row * sides, "performance_claim": False, "quality_claim": False,
            "decode_restaging": False, "initial_prefill_bounded": bounded,
            "initial_full_history_allocation": not bounded,
            "route": {"arch": "gfx1100", "bdf": "0000:03:00.0", "kv_backend": "legacy",
                      "kv_mode": "q8/q8", "dn_quant": "Q8", "dn_ef": True,
                      "graph_forward": False, "replay_enabled": False, "pbs_lean": False,
                      "pbs_gdn_tape": False, "pbs_max_batch": chunk,
                      "prefill_max_batch_override": 512,
                      "prefill_path": "ordinary_with_pbs" if arm == "baseline" else "positioned_lab",
                      "singleton_path": "ordinary_ar" if arm == "baseline" else "positioned_ar_lab"},
            "ledger": ledger, "prefill_trace": prefill, "trace": trace,
            "prefill_logits_sha256": prefill[-1]["logits_sha256"],
            "prefill_state": copy.deepcopy(prefill[-1]["state"]),
            "selected_block_starts": policy(prompt // 128),
            "archive_committed_rows": prompt // 128 * 128,
            "archive_host_bytes": prompt // 128 * 128 * row * sides,
            "first_new_fa_k_sha256": sha("first K"),
            "final_state": state("final", prefill[-1]["visible_rows"] + requested - 1)}


def guard(arm, path):
    return {"schema_version": 1, "guard_kind": "sampled_not_hard_limit", "status": "completed",
            "exit_code": 0, "child_exit_code": 0, "pci_bdf": "0000:03:00.0",
            "command": ["oracle", "--mode", arm, "--out", str(path)],
            "limit_bytes": 1000, "safety_margin_bytes": 100, "guard_limit_bytes": 900,
            "sample_interval_seconds": .05, "samples": [
                {"vram_used_bytes": 100, "raw": "100\n", "elapsed_seconds": 0},
                {"vram_used_bytes": 200, "raw": "200\n", "elapsed_seconds": .05}],
            "baseline_bytes": 100, "peak_bytes": 200, "peak_delta_bytes": 100,
            "termination_signals": []}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = {arm: fixture(arm) for arm in COMPARE.ARMS}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def invalid(self, text=None):
        with self.assertRaisesRegex(ValueError, text or "."):
            COMPARE.compare_oracles(self.data)

    def invoke(self, guards=None, bounded=True, output=None):
        argv = [sys.executable, str(SCRIPT)]
        for arm, value in self.data.items():
            if arm == "bounded" and not bounded:
                continue
            path = self.folder / (arm + ".json")
            path.write_text(json.dumps(value))
            argv += ["--" + arm, str(path)]
        for arm, value in (guards or {}).items():
            path = self.folder / (arm + ".guard.json")
            path.write_text(json.dumps(value))
            argv += ["--guard", arm + "=" + str(path)]
        out = output or self.folder / "comparison.json"
        result = subprocess.run(argv + ["--out", str(out)], capture_output=True, text=True)
        return result, json.loads(out.read_text())

    def test_positive_all_arms_and_baseline_identity_only(self):
        statuses = COMPARE.compare_oracles(self.data)
        self.assertTrue(all(v["long_decode_pass"] for v in statuses.values()))
        result, report = self.invoke(bounded=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["pass"])
        self.assertFalse(report["sparse_later_tokens_required_equal"])

    def test_bounded_later_logits_tokens_and_state_need_not_equal(self):
        self.data["bounded"]["trace"][40]["token"] = 999
        self.data["bounded"]["trace"][40]["logits_sha256"] = sha("different")
        self.data["bounded"]["final_state"]["dn_sha256"] = sha("bounded DN")
        COMPARE.compare_oracles(self.data)

    def test_block_aligned_chunks_and_singleton_tail(self):
        self.data = {arm: fixture(arm, prompt=4097, chunk=256) for arm in COMPARE.ARMS}
        COMPARE.compare_oracles(self.data)
        self.assertEqual(self.data["identity"]["prefill_trace"][-1]["tokens"], 1)
        self.data["identity"]["prefill_trace"][0]["carry_sha256"] = sha("not empty")
        self.invalid("empty carry")

    def test_no_real_bounded_gap_or_physical_reduction_is_rejected(self):
        self.data = {arm: fixture(arm, prompt=1025, chunk=256) for arm in COMPARE.ARMS}
        self.invalid("physical cap")

    def test_missing_required_fields(self):
        for key in ("schema", "route", "ledger", "oracle_binary_sha256", "prefill_trace",
                    "final_state", "initial_physical_cap", "trace"):
            value = self.data["identity"].pop(key)
            self.invalid(key if key not in ("schema", "route") else None)
            self.data["identity"][key] = value

    def test_old_schema_refused(self):
        self.data["baseline"]["schema"] = "hipfire.tiered_kv_model.v2"
        self.invalid("schema")

    def test_booleans_are_not_integers(self):
        for key in ("chunk_tokens", "window_blocks", "physical_cap", "generated_tokens", "prefix_records"):
            old = self.data["bounded"][key]
            self.data["bounded"][key] = True
            self.invalid("integer")
            self.data["bounded"][key] = old
        item = self.data["bounded"]["prefill_trace"][0]
        item["selected_block_starts"] = [False]
        self.invalid()

    def test_each_chunk_field_required(self):
        item = self.data["bounded"]["prefill_trace"][2]
        for key in list(item):
            old = item.pop(key)
            self.invalid()
            item[key] = old

    def test_bad_map(self):
        self.data["bounded"]["prefill_trace"][-1]["selected_block_starts"] = [0, 128]
        self.invalid("selected_block_starts")

    def test_bad_carry(self):
        for key in ("carry_source_row", "carry_destination_row", "carry_rows", "carry_verified_bytes"):
            item = self.data["bounded"]["prefill_trace"][2]
            old = item[key]
            item[key] += 1
            self.invalid(key)
            item[key] = old

    def test_bad_archive_source_and_absolute_block(self):
        item = self.data["bounded"]["prefill_trace"][3]
        item["archive_source_rows"][0] += 1
        self.invalid("archive_source_rows")
        item["archive_source_rows"][0] -= 1
        item["archived_blocks"][0]["absolute_start"] += 128
        self.invalid("absolute_start")

    def test_verified_bytes_cover_all_sides(self):
        for key in ("stage_h2d_bytes", "staged_history_verified_bytes", "archive_host_bytes"):
            item = self.data["bounded"]["prefill_trace"][3]
            old = item[key]
            item[key] //= 2
            self.invalid(key)
            item[key] = old

    def test_incomplete_and_noncontiguous_prefill(self):
        trace = self.data["identity"]["prefill_trace"]
        item = trace.pop()
        self.invalid("incomplete")
        trace.append(item)
        trace[1]["absolute_start"] += 1
        self.invalid("absolute_start")

    def test_incomplete_decode(self):
        self.data["identity"]["trace"].pop()
        self.invalid("incomplete decode")

    def test_mixed_binary_prompt_route_and_geometry(self):
        for key in ("oracle_binary_sha256", "model_sha256", "prompt_tokens_sha256"):
            old = self.data["bounded"][key]
            self.data["bounded"][key] = sha("other")
            self.invalid(key)
            self.data["bounded"][key] = old
        self.data["bounded"]["route"]["bdf"] = "0000:04:00.0"
        self.invalid("bdf")

    def test_identity_logits_and_final_state_drift(self):
        item = self.data["identity"]["prefill_trace"][1]
        item["logits_sha256"] = sha("drift")
        self.invalid("prefill_trace")
        item["logits_sha256"] = self.data["baseline"]["prefill_trace"][1]["logits_sha256"]
        self.data["identity"]["final_state"]["dn_sha256"] = sha("drift")
        self.invalid("final_state")

    def test_shared_full_history_storage_corruption_is_rejected(self):
        for arm in ("baseline", "identity"):
            self.data[arm]["prefill_trace"][4]["state"]["kv_sha256"] = sha("shared corruption")
        self.invalid("GPU source changed")

    def test_storage_cannot_advance_bounded_dn(self):
        self.data["bounded"]["prefill_trace"][4]["state"]["dn_sha256"] = sha("extra DN")
        self.invalid("storage changed dn")

    def test_capacity_must_be_constant(self):
        self.data["bounded"]["initial_physical_cap"] += 128
        self.invalid("initial_physical_cap")

    def test_ledger_requires_all_phases_and_enforces_limit(self):
        phase = self.data["bounded"]["ledger"].pop("after_cleanup")
        self.invalid("after_cleanup")
        self.data["bounded"]["ledger"]["after_cleanup"] = phase
        phase["committed"]["device_bytes"] = 21 << 30
        self.invalid("exceeds limit")

    def test_short_natural_stop_is_not_long_pass(self):
        for arm, data in self.data.items():
            data["trace"] = data["trace"][:10]
            data["generated_tokens"] = 10
            data["finish_reason"] = "stop"
            data["final_state"]["visible_rows"] = data["prefill_state"]["visible_rows"] + 9
        result, report = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assertTrue(report["state_comparison_pass"])
        self.assertFalse(report["long_decode_pass"])
        self.assertTrue(report["validated_arms"]["bounded"]["natural_short_stop"])

    def test_first_token_natural_stop_has_no_forward_or_first_k(self):
        for data in self.data.values():
            data["trace"] = data["trace"][:1]
            data["generated_tokens"] = 1
            data["finish_reason"] = "stop"
            data["final_state"] = copy.deepcopy(data["prefill_state"])
            data["first_new_fa_k_sha256"] = None
        statuses = COMPARE.compare_oracles(self.data)
        self.assertTrue(all(v["natural_short_stop"] for v in statuses.values()))
        self.data["bounded"]["final_state"]["dn_sha256"] = sha("should not change")
        self.invalid("without a forwarded")

    def test_guard_positive_and_binding(self):
        value = guard("bounded", self.folder / "bounded.json")
        result, report = self.invoke({"bounded": value})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sampled_guard_passed", report["guard_evidence"]["bounded"])

    def test_completed_guard_may_have_normal_process_group_cleanup_signals(self):
        value = guard("bounded", self.folder / "bounded.json")
        value["termination_signals"] = ["SIGTERM", "SIGKILL"]
        result, report = self.invoke({"bounded": value})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["pass"])
        self.assertIn("sampled_guard_passed", report["guard_evidence"]["bounded"])

    def test_guard_bad_status_exit_trip_and_command(self):
        for index, (key, value) in enumerate((("status", "limit_exceeded"), ("exit_code", 125),
                                             ("child_exit_code", -15), ("child_exit_code", 1),
                                             ("command", ["oracle", "--mode", "identity",
                                                          "--out", "wrong.json"]))):
            with self.subTest(key=key):
                receipt = guard("bounded", self.folder / "bounded.json")
                receipt[key] = value
                out = self.folder / f"{index}-{key}.comparison.json"
                result, report = self.invoke({"bounded": receipt}, output=out)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(report["pass"])

    def test_guard_wrong_output_and_missing_mode_are_rejected(self):
        for index, command in enumerate([
                ["oracle", "--mode", "bounded", "--out", str(self.folder / "other.json")],
                ["oracle", "--out", str(self.folder / "bounded.json")]]):
            receipt = guard("bounded", self.folder / "bounded.json")
            receipt["command"] = command
            result, report = self.invoke({"bounded": receipt},
                                         output=self.folder / f"binding{index}.json")
            self.assertEqual(result.returncode, 1)
            self.assertIn("guard command", report["error"])

    def test_output_is_new_and_inputs_unchanged(self):
        result, report = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = (self.folder / "identity.json").read_bytes()
        out = self.folder / "comparison.json"
        old = out.read_bytes()
        result, _ = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(out.read_bytes(), old)
        self.assertEqual((self.folder / "identity.json").read_bytes(), original)

    def test_duplicate_json_keys_and_nonfinite_refused(self):
        for raw in ('{"schema":1,"schema":2}', '{"schema":NaN}'):
            report = {"inputs": {}}
            path = self.folder / "bad.json"
            path.write_text(raw)
            with self.assertRaises(ValueError):
                COMPARE.load_evidence(path, report, "bad")

    def test_existing_symlink_output_refused(self):
        target = self.folder / "target.json"
        target.write_text("do not replace")
        link = self.folder / "link.json"
        link.symlink_to(target)
        # invoke() reads the unchanged target after the comparator rejects it.
        argv = [sys.executable, str(SCRIPT), "--baseline", "unused", "--identity", "unused",
                "--out", str(link)]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(target.read_text(), "do not replace")


if __name__ == "__main__":
    unittest.main()
