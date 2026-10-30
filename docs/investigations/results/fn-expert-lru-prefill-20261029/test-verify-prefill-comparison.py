"""Small synthetic receipt controls; no inference, subprocess, network or GPU."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prefill_receipt_verifier", HERE / "verify-prefill-comparison.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DONOR = 10486579200


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root):
    write(root / "BASELINE.json", {"measured_restore_bytes_per_request": DONOR})
    manifests = {}
    for variant in ("old", "new"):
        manifests[variant] = {"commit": ("a" if variant == "old" else "b") * 40,
            "binaries": {"hipfire": {"size": 1, "sha256": "1" * 64},
                         "daemon_lru_" + variant: {"size": 1, "sha256": "2" * 64}}}
        write(root / ("runtime-" + variant) / "MANIFEST.json", manifests[variant])
    for index, name in enumerate(("A1", "B1", "B2", "A2", "A3", "B3")):
        variant = "old" if name[0] == "A" else "new"
        phases = ("warmup", "battery", "chain", "staged", "longdecode") if name in ("A1", "B1") else ("warmup", "battery")
        d = root / ("serve-" + name)
        d.mkdir()
        env = {"HIPFIRE_QWEN4_EXPERT_CACHE": "1", "HIPFIRE_DN_STATE_EF": "1",
               "HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT": "50", "HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT": "64",
               "HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB": "20000", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096",
               "HIPFIRE_QWEN4_DECODE_SHARE": "off", "HIPFIRE_OFFLOAD_EXEC": "pcie",
               "HIPFIRE_MTP_MODE": "off", "HIPFIRE_PLE_FORMAT": "q4",
               "HIPFIRE_CLI_BIN": str(root / ("runtime-" + variant) / "hipfire"),
               "HIPFIRE_DAEMON_BIN": str(root / ("runtime-" + variant) / ("daemon_lru_" + variant))}
        receipt = {"name": name, "variant": variant, "cache": "1", "status": "PASS", "env": env,
                   "frozen_manifest": manifests[variant], "phase_order": list(phases), "phase_results": []}
        lines = ["qwen4 state: QSA fp8 K/V (vmm context storage), GDN q8 recurrent",
                 "qwen4 long-prefill expert staging: forwards of >= 2048 rows stage experts"]
        stat = {"tokens": 0, "fills_started": 0, "fills_completed": 0, "prefill_restores": 0, "restore_bytes": 0}
        if variant == "new":
            stat.update(mapped_prefill_calls=0, mapped_prefill_rows=0)
        seq = 0
        for phase in phases:
            count = 2 if phase == "staged" else 1 if phase == "longdecode" else 5
            rows = []
            cap = {"warmup": 16, "battery": 512, "chain": 512, "staged": 128, "longdecode": 1024}[phase]
            for i in range(count):
                seq += 1
                ctx = 5103 if phase == "staged" and i == 1 else 80 if phase == "chain" and i else 44
                cached = 44 if phase == "chain" and i else 0
                gen = 16 if phase == "warmup" else 128 if phase == "staged" else 1024 if phase == "longdecode" else 50
                if seq > 1:
                    if variant == "old" or (phase == "staged" and i == 1):
                        stat["prefill_restores"] += 1
                        stat["restore_bytes"] += DONOR
                    else:
                        stat["mapped_prefill_calls"] += 1
                        stat["mapped_prefill_rows"] += ctx - cached
                stat["tokens"] += gen
                stat["fills_started"] += gen
                stat["fills_completed"] += gen
                lines.append("qwen4 exact expert cache stats: " + json.dumps(stat))
                rid = f"chatcmpl-{1000 + index}-{seq}"
                lines.append(f"HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4_ib position=512 request_id={rid} replays=5")
                rows.append({"request_id": rid, "prompt_md5": f"{100 + i:032x}", "request_md5": f"{cap + i:032x}",
                    "content": f"回答 {phase} {i}", "assistant_content": f"回答 {phase} {i}", "reasoning_content": "", "tool_calls": [],
                    "ctx": ctx, "cached": cached, "gen": gen, "finish": "length" if gen == cap else "stop",
                    "attractor": False, "empty": False, "atem_leak": False, "retrieval_missing": [],
                    "terminal_count": 1, "saw_done": True, "post_terminal_bytes": 0, "stream_error": None,
                    "runaway": gen == cap, "decode_tok_s": 30.0, "decode_estimated": False,
                    "prefill_ms": 100.0 if variant == "old" else 50.0,
                    "wall_s": gen / 30 + (0.1 if variant == "old" else 0.05),
                    "kv_mode": None, "kv_backend": None, "kv_backend_legacy": False, "kv_backend_reason": None})
            write(d / (phase + ".json"), rows)
            args = ["scripts/serve_harness.py", "--max-tokens", str(cap), "--max-seq", "16384",
                    "--sampling", "greedy", "--speculation", "off", "--thinking", "off", "--seed", "123"]
            receipt["phase_results"].append({"phase": phase, "status": "PASS", "arguments": args})
        (d / "serve.log").write_text("\n".join(lines) + "\n")
        write(d / "run.json", receipt)
    write(root / "overall-matrix.json", [{"case": n, "cache": "0" if n[0] == "A" else "1", "exit_code": 0}
                                       for n in ("A1", "B1", "B2", "A2", "A3", "B3")])


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="receipt-negative-", dir=HERE)
        self.root = Path(self.tmp.name)
        fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self, partial=False):
        return m.Audit(self.root, partial).finish()

    def mutate_row(self, name, phase, update):
        p = self.root / ("serve-" + name) / (phase + ".json")
        rows = json.loads(p.read_text())
        rows[0].update(update)
        write(p, rows)

    def mutate_stats(self, name, transform):
        p = self.root / ("serve-" + name) / "serve.log"
        result = []
        marker = "qwen4 exact expert cache stats: "
        for line in p.read_text().splitlines():
            if marker in line:
                value = json.loads(line.split(marker, 1)[1])
                transform(value)
                line = marker + json.dumps(value)
            result.append(line)
        p.write_text("\n".join(result) + "\n")

    def test_complete_and_unobserved_backend(self):
        x = self.audit()
        self.assertTrue(x["all_pass"], x["failures"])
        self.assertEqual(x["battery_timing"]["arms"]["A"]["n_fresh_processes"], 3)
        self.assertEqual(x["battery_timing"]["arms"]["B"]["n_fresh_processes"], 3)
        self.assertEqual(x["lifecycle"]["B1"]["staged"]["delta"]["restore_bytes"], DONOR)
        row = x["cases"]["B1"]["phases"]["battery"]["summary"][0]
        self.assertEqual(row["harness_backend_fields"]["kv_backend"], "UNOBSERVED")

    def test_transcript_change_fails(self):
        self.mutate_row("B2", "battery", {"content": "不同答案"})
        self.assertTrue(any("compare/battery" in e["where"] for e in self.audit()["failures"]))

    def test_terminal_failure_survives_partial(self):
        self.mutate_row("B1", "battery", {"terminal_count": 2, "saw_done": False, "post_terminal_bytes": 7})
        self.assertGreaterEqual(len(self.audit(partial=True)["failures"]), 3)

    def test_short_bulk_restore_fails(self):
        def bad(s):
            s["prefill_restores"] += 1
            s["restore_bytes"] += DONOR
        self.mutate_stats("B2", bad)
        self.assertTrue(any("short-only" in str(e) for e in self.audit()["failures"]))

    def test_wrong_long_restore_bytes_fails(self):
        def bad(s):
            if s["prefill_restores"]:
                s["restore_bytes"] += 256
        self.mutate_stats("B1", bad)
        self.assertTrue(any("exactly once" in str(e) for e in self.audit()["failures"]))

    def test_missing_case_is_partial(self):
        shutil.rmtree(self.root / "serve-A3")
        x = self.audit(partial=True)
        self.assertEqual(x["status"], "PARTIAL")
        self.assertFalse(x["all_pass"])

    def test_natural_stop_retains_coverage_gap(self):
        for name in ("A1", "B1"):
            self.mutate_row(name, "longdecode", {"gen": 512, "finish": "stop"})
        x = self.audit()
        self.assertFalse(x["all_pass"])
        self.assertFalse(x["failures"], x["failures"])
        self.assertTrue(any("1024" in e["unmet"] for e in x["coverage_limitations"]))


if __name__ == "__main__":
    unittest.main()
