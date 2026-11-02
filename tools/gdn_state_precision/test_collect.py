# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU-only tests for collect.py (no GPU, no fabric, no real binaries).

    python3 -m unittest tools/gdn_state_precision/test_collect.py -v
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gdn_collect", HERE / "collect.py")
collect = importlib.util.module_from_spec(_spec)
sys.modules["gdn_collect"] = collect
_spec.loader.exec_module(collect)  # type: ignore[union-attr]

HASH = "0123456789abcdef"
BIN_NAMES = {
    "qwen4_kld": "qwen4_kld",
    "hipfire": "hipfire",
    "daemon": "daemon",
    "gdn_state_precision_oracle": "gdn_state_precision_oracle",
    "greedy_mtp_identity_hw": f"greedy_mtp_identity_hw-{HASH}",
    "session_cache_hw": f"session_cache_hw-{HASH}",
    "sampled_mtp_distribution_hw": f"sampled_mtp_distribution_hw-{HASH}",
    "q8_ef_product_routes_hw": f"q8_ef_product_routes_hw-{HASH}",
}

#: Fake ``qwen4_kld``: writes a real-format .kldseq whose chunk KLDs depend on
#: HIPFIRE_STATE_QUANT and prints the real eval summary lines on stderr.
FAKE_KLD = """#!{python}
import hashlib, os, struct, sys
a = sys.argv
out = a[a.index("--output") + 1]
if "--append-rows" in a:
    stem, ext = os.path.splitext(out)
    out = f"{{stem}}.a{{a[a.index('--append-rows') + 1]}}{{ext}}"
mode = os.environ["HIPFIRE_STATE_QUANT"]
base = {{"q8": 0.0100, "fp32": 0.0080, "q8_ef": 0.0085}}[mode]
chunks = [(base + 0.0001 * (i % 3), base * 3, 1.5 + base) for i in range(24)]
body = b"".join(struct.pack("<ddd", *c) for c in chunks)
data = b"HFKSEQ\\0\\0" + struct.pack("<III", 2, len(chunks), 0) + body
open(out, "wb").write(data)
route = "decode" if "--decode" in a else "all-row"
sys.stderr.write(f"qwen4_kld eval: 24 chunks x 512 ({{route}} route); teacher PPL 5.0000 (t.hfq)\\n")
sys.stderr.write("  chunk 1/24  KLD 0.010000  running KLD 0.010000  1s\\n")
m = sum(c[0] for c in chunks) / len(chunks)
sys.stderr.write(f"qwen4_kld eval: mean KLD = {{m:.6f}}  mean NLL = 1.510000  PPL = 4.5267  top1 = 0.9000  (1000 tokens)  logits sha256 {{hashlib.sha256(data).hexdigest()}} -> {{out}}\\n")
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: Fake ``hipfire bench``: prints a banner on stderr (daemon log lines) and the
#: real report shape on stdout, with decode speed depending on the mode.
FAKE_HIPFIRE = """#!{python}
import json, os, sys
a = sys.argv
mode = os.environ["HIPFIRE_STATE_QUANT"]
slow = 0.9 if mode == "fp32" else 1.0
sys.stderr.write("  qwen4 state: QSA fp8 K/V, GDN " + mode + " recurrent\\n")
sys.stderr.write("daemon: max_seq 4096 card_cap 1000\\n")
if "--matrix" in a:
    rep = {{"protocol": "synthetic-pp-tg-matrix-v1",
           "prefill": [{{"tokens": 8192, "stats": {{"median": 1000 * slow}}}}],
           "decode": [{{"context": 4096, "tokens": 32, "stats": {{"median": 40 * slow}}}}]}}
elif "--ttft" in a:
    rep = {{"protocol": "client-ttft-v1", "prompt_tokens": 9000, "ttft": {{"median": 2000 / slow}}, "pp_tok_s": {{"median": 4500 * slow}}}}
else:
    rep = {{"protocol": "native-generate-v1", "prompt_tokens": 300, "max_seq": {{"value": 4096}},
           "decode_tok_s": {{"median": 100 * slow}}, "prefill_tok_s": {{"median": 900 * slow}},
           "ttft_ms": {{"median": 50}}, "spec_tau": [4.0], "spec_route": ["mtp"]}}
print("hipfire bench banner is not json")
print(json.dumps(rep, indent=2))
"""

#: Fake cargo test executable: prints the libtest lines for whatever it was asked
#: to run (``--exact NAME`` or the default case) and writes a summary file.
FAKE_TEST = """#!{python}
import os, sys
a = sys.argv
if "--list" in a:
    for n in """ + repr(list(collect.SESSION_CACHE_TESTS)) + """:
        print(n + ": test")
    sys.exit(0)
name = a[a.index("--exact") + 1] if "--exact" in a else "some_case"
out = os.environ.get("HIPFIRE_MTP_IDENTITY_OUT")
if out:
    os.makedirs(os.path.join(out, "seasons"), exist_ok=True)
    open(os.path.join(out, "seasons", "summary.txt"), "w").write("ar: 256 ids\\nbatched: identical to ar\\n")
    open(os.path.join(out, "seasons", "ar.txt"), "w").write("decoded text")
print("running 1 test")
print(f"test {{name}} ... ok")
print()
print("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s")
"""


class Fixture:
    def __init__(self, root: Path, omit: tuple[str, ...] = (), fake_kld: bool = False, fakes: dict | None = None):
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        lines = []
        fakes = dict(fakes or {})
        if fake_kld:
            fakes["qwen4_kld"] = FAKE_KLD
        for base, name in BIN_NAMES.items():
            if base in omit:
                continue
            path = self.bin / name
            if base in fakes:
                content = fakes[base].format(python=sys.executable).encode()
            else:
                content = f"#!/bin/sh\necho fake {base}\n".encode()
            path.write_bytes(content)
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
            lines.append(f"{sha256_bytes(content)}  {name}")
        (self.bin / "BINARIES.sha256").write_text("\n".join(lines) + "\n")
        self.model = root / "model.mq4"
        self.model.write_bytes(b"model")
        (root / "model.mq4.ple-q4").write_bytes(b"ple4")
        self.wt2 = root / "wt2.kldref"
        self.code = root / "code.kldref"
        self.wt2.write_bytes(b"wt2")
        self.code.write_bytes(b"code")
        self.prompt = root / "prompt.txt"
        self.prompt.write_text("p")
        self.long_prompt = root / "long.txt"
        self.long_prompt.write_text("l")
        self.out = root / "out"

    def argv(self, stage: str, *extra: str, modes: str = "q8,fp32", runs: str = "3") -> list[str]:
        return [
            "--stage", stage, "--binary-dir", str(self.bin), "--model", str(self.model),
            "--wt2-ref", str(self.wt2), "--code-ref", str(self.code), "--modes", modes,
            "--runs", runs, "--max-seq", "4096", "--out", str(self.out),
            "--prompt-file", str(self.prompt), "--long-prompt-file", str(self.long_prompt),
            *extra,
        ]

    def config(self, stage: str, *extra: str, modes: str = "q8,fp32", runs: str = "3", env=None):
        args = collect.make_parser().parse_args(self.argv(stage, "--dry-run", "--run-id", "rid", *extra, modes=modes, runs=runs))
        return collect.validate(args, env or {})


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()


class RefusalTests(TempCase):
    def run_cli(self, argv, environ=None):
        return collect.run(argv, environ if environ is not None else {})

    def test_relative_paths_refused(self):
        fx = Fixture(self.root)
        for flag in ("--model", "--binary-dir", "--wt2-ref", "--code-ref", "--out"):
            argv = fx.argv("smoke", "--dry-run")
            argv[argv.index(flag) + 1] = "relative/x"
            with self.assertRaisesRegex(collect.CollectError, "absolute"):
                self.run_cli(argv)

    def test_missing_inputs_refused(self):
        fx = Fixture(self.root)
        fx.wt2.unlink()
        with self.assertRaisesRegex(collect.CollectError, "wt2-ref"):
            self.run_cli(fx.argv("smoke", "--dry-run"))

    def test_missing_binary_refused(self):
        for omitted, stage in (("qwen4_kld", "quality"), ("hipfire", "perf"), ("daemon", "memory"),
                               ("gdn_state_precision_oracle", "smoke"), ("sampled_mtp_distribution_hw", "sampled")):
            with tempfile.TemporaryDirectory() as t:
                fx = Fixture(Path(t).resolve(), omit=(omitted,))
                with self.assertRaisesRegex(collect.CollectError, omitted):
                    self.run_cli(fx.argv(stage, "--dry-run"))

    def test_listed_but_absent_binary_refused(self):
        fx = Fixture(self.root)
        (fx.bin / "qwen4_kld").unlink()
        with self.assertRaisesRegex(collect.CollectError, "not a file"):
            self.run_cli(fx.argv("quality", "--dry-run"))

    def test_hash_mismatch_refused(self):
        fx = Fixture(self.root)
        (fx.bin / "qwen4_kld").write_bytes(b"#!/bin/sh\necho tampered\n")
        with self.assertRaisesRegex(collect.CollectError, "sha256"):
            self.run_cli(fx.argv("quality", "--dry-run"))

    def test_missing_manifest_refused(self):
        fx = Fixture(self.root)
        (fx.bin / "BINARIES.sha256").unlink()
        with self.assertRaisesRegex(collect.CollectError, "BINARIES.sha256"):
            self.run_cli(fx.argv("quality", "--dry-run"))

    def test_ambiguous_manifest_name_refused(self):
        fx = Fixture(self.root)
        extra = fx.bin / "session_cache_hw-fedcba9876543210"
        extra.write_bytes(b"x")
        extra.chmod(0o755)
        with open(fx.bin / "BINARIES.sha256", "a") as handle:
            handle.write(f"{sha256_bytes(b'x')}  {extra.name}\n")
        with self.assertRaisesRegex(collect.CollectError, "ambiguous"):
            self.run_cli(fx.argv("restore", "--dry-run"))

    def test_unknown_mode_refused(self):
        fx = Fixture(self.root)
        for bad in ("q8,fp16", "q8,f32", "q8,q8"):
            with self.assertRaisesRegex(collect.CollectError, "mode"):
                self.run_cli(fx.argv("quality", "--dry-run", modes=bad))

    def test_quality_and_perf_need_q8_baseline(self):
        fx = Fixture(self.root)
        for stage in ("quality", "perf"):
            with self.assertRaisesRegex(collect.CollectError, "baseline"):
                self.run_cli(fx.argv(stage, "--dry-run", modes="fp32"))

    def test_perf_and_memory_need_prompts(self):
        fx = Fixture(self.root)
        argv = fx.argv("perf", "--dry-run")
        i = argv.index("--long-prompt-file")
        del argv[i:i + 2]
        with self.assertRaisesRegex(collect.CollectError, "long-prompt-file"):
            self.run_cli(argv)
        argv = fx.argv("memory", "--dry-run")
        i = argv.index("--prompt-file")
        del argv[i:i + 2]
        del argv[argv.index("--long-prompt-file"):argv.index("--long-prompt-file") + 2]
        with self.assertRaisesRegex(collect.CollectError, "prompt-file"):
            self.run_cli(argv)

    def test_missing_hold_refused_for_gpu_stages(self):
        fx = Fixture(self.root)
        for stage in collect.STAGES:
            with self.assertRaisesRegex(collect.CollectError, "no fabric hold"):
                self.run_cli(fx.argv(stage, "--graph", "0"), environ={})
        self.assertFalse((fx.out / "smoke").exists())

    def test_hold_with_non_ancestor_wrapper_refused(self):
        fx = Fixture(self.root)
        bogus = f"{2**22 + 12345}:1:halo"
        with self.assertRaisesRegex(collect.CollectError, "not an ancestor"):
            self.run_cli(fx.argv("quality"), environ={"FABRIC_JOB": bogus})

    def test_existing_nonempty_run_dir_refused(self):
        fx = Fixture(self.root)
        run_dir = fx.out / "quality" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "old.txt").write_text("x")
        with self.assertRaisesRegex(collect.CollectError, "already exists"):
            self.run_cli(fx.argv("quality", "--run-id", "r1"), environ={"FABRIC_JOB": "1:1:halo"})

    def test_q8_ef_restore_and_sampled_refuse_without_ef_executable(self):
        fx = Fixture(self.root, omit=("q8_ef_product_routes_hw",))
        with self.assertRaisesRegex(collect.CollectError, "q8_ef_product_routes_hw"):
            self.run_cli(fx.argv("restore", "--dry-run", modes="q8,fp32,q8_ef"))

    def test_sampled_trials_floor(self):
        fx = Fixture(self.root)
        with self.assertRaisesRegex(collect.CollectError, "2000"):
            self.run_cli(fx.argv("sampled", "--dry-run", "--sampled-trials", "1999"))

    def test_smoke_refuses_unset_graph(self):
        fx = Fixture(self.root)
        with self.assertRaisesRegex(collect.CollectError, "smoke"):
            self.run_cli(fx.argv("smoke", "--dry-run", "--graph", "unset"))

    def test_hold_refusal_rules(self):
        chain = [100, 90, 80]
        ok = collect.hold_refusal("90:777:halo", chain, lambda pid: 777, lambda pids: True)
        self.assertIsNone(ok)
        self.assertIn("FABRIC_JOB", collect.hold_refusal("", chain, lambda p: 1, lambda p: True))
        self.assertIn("not an ancestor", collect.hold_refusal("5:1:halo", chain, lambda p: 1, lambda p: True))
        self.assertIn("stale", collect.hold_refusal("90:777:halo", chain, lambda p: 1, lambda p: True))
        seen = []
        why = collect.hold_refusal("90:777:halo", chain, lambda p: 777, lambda pids: seen.append(pids) or False)
        self.assertIn("hostmem-flashnext.lock", why)
        self.assertEqual(seen, [[100, 90]])


class HelpTests(unittest.TestCase):
    def test_help_exits_zero_and_names_frozen_flags(self):
        cp = subprocess.run([sys.executable, str(HERE / "collect.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        for flag in ("--stage", "--binary-dir", "--model", "--wt2-ref", "--code-ref", "--modes", "--runs",
                     "--max-seq", "--out", "--dry-run", "--prompt-file", "--long-prompt-file", "--graph",
                     "--analyze", "--child-timeout-secs"):
            self.assertIn(flag, cp.stdout)


class CommandConstructionTests(TempCase):
    def setUp(self):
        super().setUp()
        self.fx = Fixture(self.root)
        self.M = str(self.fx.model)
        self.bin = lambda base: str(self.fx.bin / BIN_NAMES[base])

    def plan(self, stage, *extra, modes="q8,fp32", runs="1"):
        cfg = self.fx.config(stage, *extra, modes=modes, runs=runs)
        return cfg, collect.plan_children(cfg)

    def test_dry_run_prints_and_creates_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = collect.run(self.fx.argv("quality", "--dry-run", "--run-id", "rid", runs="1"), {})
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("qwen4_kld eval --model", text)
        self.assertIn("HIPFIRE_STATE_QUANT=fp32", text)
        self.assertFalse(self.fx.out.exists())

    def test_smoke_commands(self):
        cfg, kids = self.plan("smoke", modes="q8,fp32,q8_ef")
        d = cfg.run_dir
        oracle0 = kids[0]
        self.assertEqual(
            oracle0.argv,
            [self.bin("gdn_state_precision_oracle"), "--formats", "q8,fp32,q8_ef", "--graph", "0",
             "--max-rows", "32", "--all-keeps", "--json", str(d / "run0/g0/q8,fp32,q8_ef/oracle/oracle.json")],
        )
        identity = [k for k in kids if k.kind == "identity_test"]
        self.assertEqual([k.mode for k in identity if k.graph == "0"], ["q8", "fp32"])  # EF never uses the old equality test
        first = identity[0]
        self.assertEqual(
            first.argv,
            [self.bin("greedy_mtp_identity_hw"), "--ignored", "--test-threads=1", "--exact",
             "greedy_mtp_matches_ar_on_flash_next_seasons"],
        )
        self.assertEqual(first.extra_env["HIPFIRE_GDN_TEST_STATE_QUANT"], "q8")
        self.assertEqual(first.extra_env["HIPFIRE_MTP_IDENTITY_MODEL"], self.M)
        self.assertEqual(first.extra_env["HIPFIRE_MTP_IDENTITY_OUT"], str(first.dir / "out"))
        self.assertEqual({k.graph for k in kids}, {"0", "1"})
        self.assertEqual(oracle0.extra_env.get("HIPFIRE_REPLAY_BACKEND"), None)
        self.assertEqual(first.extra_env["HIPFIRE_REPLAY_BACKEND"], "hip")
        g1 = [k for k in identity if k.graph == "1"][0]
        self.assertEqual(g1.extra_env["HIPFIRE_REPLAY_BACKEND"], "auto")
        self.assertNotIn("HIPFIRE_GRAPH", g1.extra_env)

    def test_quality_commands(self):
        cfg, kids = self.plan("quality")
        self.assertEqual(len(kids), 2 * 2 * 3)
        by = {k.id: k for k in kids}
        k = by["run0/g0/q8/code24-decode"]
        d = k.dir
        self.assertEqual(
            k.argv,
            [self.bin("qwen4_kld"), "eval", "--model", self.M, "--ref", str(self.fx.code), "--output",
             str(d / "out.kldseq"), "--max-chunks", "24", "--decode"],
        )
        k = by["run0/g0/fp32/wt2-append256"]
        self.assertEqual(k.argv[-2:], ["--append-rows", "256"])
        self.assertEqual(k.kldseq, "out.a256.kldseq")
        self.assertIn(str(self.fx.wt2), k.argv)
        k = by["run0/g0/q8/code24-whole"]
        self.assertEqual(k.argv[-2:], ["--max-chunks", "24"])
        self.assertEqual(k.extra_env["HIPFIRE_GRAPH"], "0")
        labels = sorted({k.label for k in kids})
        self.assertEqual(labels, sorted(f"{c}-{r}" for c in ("code24", "wt2") for r in ("decode", "append256", "whole")))

    def test_restore_commands(self):
        cfg, kids = self.plan("restore", modes="q8,fp32,q8_ef")
        sess = [k for k in kids if k.kind == "session_test" and k.graph == "0"]
        n = len(collect.SESSION_CACHE_TESTS)
        self.assertEqual([k.mode for k in sess], ["q8"] * n + ["fp32"] * n)
        # One process per test: each session_cache_hw test loads Flash-Next.
        for k, name in zip(sess, collect.SESSION_CACHE_TESTS * 2):
            self.assertEqual(k.argv, [self.bin("session_cache_hw"), "--ignored", "--test-threads=1", "--exact", name])
            self.assertEqual(k.expect_tests, (name,))
            self.assertEqual(k.label, "session-cache")
            self.assertEqual(k.id, f"run0/g0/{k.mode}/session-cache/{name}")
        self.assertEqual(len({k.dir for k in sess}), 2 * n)
        self.assertEqual(sess[0].extra_env["HIPFIRE_SESSION_CACHE_MODEL"], self.M)
        ident = [k for k in kids if k.kind == "identity_test" and k.graph == "0"]
        self.assertEqual(
            ident[0].argv,
            [self.bin("greedy_mtp_identity_hw"), "--ignored", "--test-threads=1", "greedy_mtp_matches_ar_on_flash_next"],
        )
        ef = [k for k in kids if k.kind == "ef_routes_test" and k.graph == "0"]
        self.assertEqual(len(ef), 1)
        self.assertEqual(ef[0].argv[0], self.bin("q8_ef_product_routes_hw"))
        self.assertEqual(ef[0].extra_env["HIPFIRE_GDN_TEST_STATE_QUANT"], "q8_ef")
        self.assertEqual(ef[0].argv[1:], ["--ignored", "--nocapture", "--test-threads=1"])

    def test_session_cache_hw_without_exact_is_refused(self):
        cfg = self.fx.config("restore", "--graph", "0", modes="fp32")
        bad = collect._child(
            cfg, 0, "0", "fp32", "session-cache", "session_test",
            [cfg.exe("session_cache_hw"), "--ignored", "--test-threads=1"], {}, "session_cache_hw",
        )
        with self.assertRaisesRegex(collect.CollectError, "one known test per process"):
            collect.require_one_session_test(bad)
        bad.argv += ["--exact", "not_a_session_test"]
        with self.assertRaisesRegex(collect.CollectError, "one known test per process"):
            collect.require_one_session_test(bad)
        ok = collect._child(
            cfg, 0, "0", "fp32", "session-cache", "session_test",
            [cfg.exe("session_cache_hw"), "--exact", collect.SESSION_CACHE_TESTS[0]], {}, "session_cache_hw",
        )
        collect.require_one_session_test(ok)

    def test_session_test_list_must_match(self):
        cfg, kids = self.plan("restore", "--graph", "0", modes="fp32")
        exe = Path(cfg.exe("session_cache_hw"))
        exe.write_text("#!/bin/sh\necho 'disk_snapshots_restore_after_restart_on_flash_next: test'\necho 'new_case: test'\n")
        with self.assertRaisesRegex(collect.CollectError, r"unknown \['new_case'\]"):
            collect.check_session_test_list(cfg, kids)
        exe.write_text(FAKE_TEST.format(python=sys.executable))
        collect.check_session_test_list(cfg, kids)

    def test_memory_commands(self):
        cfg, kids = self.plan("memory")
        self.assertEqual([k.mode for k in kids], ["q8", "fp32"])
        self.assertEqual(
            kids[0].argv,
            [self.bin("hipfire"), "bench", self.M, "--backend", "noslots", "--workload", "stateless", "--spec", "mtp",
             "--prompt-file", str(self.fx.prompt), "--max-tokens", "32", "--runs", "1", "--warmups", "0",
             "--ctx", "4096", "--json"],
        )
        self.assertTrue(kids[0].sample_memory)
        self.assertEqual(kids[0].extra_env["HIPFIRE_DAEMON_BIN"], self.bin("daemon"))
        self.assertNotIn("HIPFIRE_GRAPH", kids[0].extra_env)  # --graph defaults to unset
        _, kids2 = self.plan("memory", "--bench-max-seq")
        self.assertEqual(kids2[0].argv[-3:], ["--max-seq", "4096", "--json"])

    def test_perf_commands(self):
        cfg, kids = self.plan("perf")
        q8 = [k for k in kids if k.mode == "q8"]
        self.assertEqual([k.label for k in q8], list(collect.PERF_LABELS))
        tail = ["--runs", "1", "--warmups", "1", "--ctx", "4096", "--json"]
        hip = self.bin("hipfire")
        P, L = str(self.fx.prompt), str(self.fx.long_prompt)
        self.assertEqual(q8[0].argv, [hip, "bench", self.M, "--backend", "noslots", "--workload", "stateless", "--spec", "off",
                                      "--prompt-file", P, "--max-tokens", "256", *tail])
        self.assertEqual(q8[1].argv, [hip, "bench", self.M, "--backend", "noslots", "--workload", "stateless", "--spec", "mtp",
                                      "--prompt-file", P, "--max-tokens", "256", *tail])
        self.assertEqual(q8[2].argv, [hip, "bench", self.M, "--ttft", "--prompt-file", L, *tail])
        self.assertEqual(q8[3].argv, [hip, "bench", self.M, "--matrix", "--pp", "8192", "--tg", "32", *tail])
        self.assertTrue(all(k.extra_env["HIPFIRE_STATE_QUANT"] == k.mode for k in kids))

    def test_sampled_commands(self):
        cfg, kids = self.plan("sampled", modes="q8,fp32,q8_ef")
        self.assertEqual(len(kids), 3)
        q8, fp32, ef = kids
        exe = self.bin("sampled_mtp_distribution_hw")
        self.assertEqual(
            q8.argv,
            [exe, "--ignored", "--nocapture", "--test-threads=1", "--exact", "sampled_mtp_matches_ar_distribution_on_flash_next"],
        )
        self.assertEqual(
            ef.argv,
            [exe, "--ignored", "--nocapture", "--test-threads=1", "--exact",
             "sampled_mtp_q8ef_matches_verify_distribution_on_flash_next"],
        )
        self.assertTrue(ef.ef_sampled and not q8.ef_sampled)
        for k in kids:
            self.assertEqual(k.extra_env["HIPFIRE_SAMPLED_MTP_TRIALS"], "2000")
            self.assertEqual(k.extra_env["HIPFIRE_SAMPLED_MTP_OUT"], str(k.dir / "sampled"))
            self.assertEqual(k.extra_env["HIPFIRE_MTP_IDENTITY_MODEL"], self.M)
        self.assertEqual(ef.extra_env["HIPFIRE_GDN_TEST_STATE_QUANT"], "q8_ef")

    def test_ab_ba_interleave(self):
        self.assertEqual(collect.arm_order(("q8", "fp32"), 0), ("q8", "fp32"))
        self.assertEqual(collect.arm_order(("q8", "fp32"), 1), ("fp32", "q8"))
        self.assertEqual(collect.arm_order(("q8", "fp32", "q8_ef"), 3), ("q8_ef", "fp32", "q8"))
        cfg, kids = self.plan("perf", runs="3")
        order = []
        for k in kids:
            if k.label == "stateless-spec-off":
                order.append((k.run, k.mode))
        self.assertEqual(order, [(0, "q8"), (0, "fp32"), (1, "fp32"), (1, "q8"), (2, "q8"), (2, "fp32")])

    def test_mode_env_forwarding_per_kind(self):
        _, quality = self.plan("quality")
        for k in quality:
            self.assertEqual(k.extra_env["HIPFIRE_STATE_QUANT"], k.mode)
            self.assertNotIn("HIPFIRE_GDN_TEST_STATE_QUANT", k.extra_env)
        _, perf = self.plan("perf")
        for k in perf:
            self.assertEqual(k.extra_env["HIPFIRE_STATE_QUANT"], k.mode)
            self.assertNotIn("HIPFIRE_GDN_TEST_STATE_QUANT", k.extra_env)
        for stage in ("restore", "sampled"):
            _, tests = self.plan(stage)
            for k in tests:
                self.assertEqual(k.extra_env["HIPFIRE_GDN_TEST_STATE_QUANT"], k.mode)
                self.assertNotIn("HIPFIRE_STATE_QUANT", k.extra_env)

    def test_child_env_scrub_and_isolation(self):
        cfg, kids = self.plan("quality")
        child = kids[0]
        parent = {
            "PATH": "/usr/bin", "FABRIC_JOB": "1:2:halo", "FABRIC_CARDS": "halo",
            "HIPFIRE_DEVICES": "0", "HIPFIRE_FABRIC_ADMISSION": "x", "HIPFIRE_LOCK_DIR": "/parent/lock",
            "HIPFIRE_FOO": "1", "HIPFIRE_GRAPH": "1", "HIPFIRE_STATE_QUANT": "fp32",
            "HIPFIRE_GDN_TEST_STATE_QUANT": "fp32", "HIPFIRE_PLE_FORMAT": "q8", "HOME": "/home/x",
        }
        env = collect.build_child_env(parent, cfg, child)
        self.assertNotIn("HIPFIRE_FOO", env)
        self.assertNotIn("HIPFIRE_GDN_TEST_STATE_QUANT", env)
        self.assertEqual(env["HIPFIRE_STATE_QUANT"], "q8")  # arm mode overrides the parent's value
        self.assertEqual(env["HIPFIRE_GRAPH"], "0")
        self.assertEqual(env["HIPFIRE_DEVICES"], "0")
        self.assertEqual(env["HIPFIRE_FABRIC_ADMISSION"], "x")
        self.assertEqual(env["HIPFIRE_QWEN4_NPU_TRUNK"], "off")
        self.assertEqual(env["HIPFIRE_PLE_FORMAT"], "q4")
        self.assertEqual(env["HIPFIRE_NORMALIZE_PROMPT"], "0")
        self.assertEqual(env["HIPFIRE_KERNEL_CACHE"], str(cfg.out / "kcache"))
        self.assertEqual(env["HOME"], str(child.dir / "home"))
        self.assertEqual(env["HIPFIRE_HOME"], str(child.dir / "home" / ".hipfire"))
        self.assertEqual(env["HIPFIRE_LOCK_DIR"], str(child.dir / "lockdir"))
        self.assertEqual(env["FABRIC_JOB"], "1:2:halo")
        self.assertEqual(env["PATH"], "/usr/bin")
        extra = {k for k in env if k.startswith("HIPFIRE_")}
        self.assertEqual(
            extra,
            {"HIPFIRE_DEVICES", "HIPFIRE_FABRIC_ADMISSION", "HIPFIRE_QWEN4_NPU_TRUNK", "HIPFIRE_PLE_FORMAT",
             "HIPFIRE_NORMALIZE_PROMPT", "HIPFIRE_GRAPH", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_HOME",
             "HIPFIRE_LOCK_DIR", "HIPFIRE_STATE_QUANT"},
        )
        # isolation is per child
        self.assertNotEqual(
            collect.build_child_env(parent, cfg, kids[1])["HOME"], env["HOME"]
        )

    def test_child_env_option_reaches_build_child_env_and_dry_run(self):
        cfg, kids = self.plan(
            "quality", "--child-env", "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14", "--child-env", "HIPFIRE_QWEN4_EXPERT_HOTSET=x=y",
        )
        self.assertEqual(cfg.child_env, {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14", "HIPFIRE_QWEN4_EXPERT_HOTSET": "x=y"})
        self.assertEqual(list(cfg.child_env), ["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "HIPFIRE_QWEN4_EXPERT_HOTSET"])
        # build_child_env is called directly (bypassing validate): the unrelated parent variable is still dropped
        env = collect.build_child_env({"PATH": "/usr/bin", "HIPFIRE_FOO": "1"}, cfg, kids[0])
        self.assertEqual(env["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "14")
        self.assertEqual(env["HIPFIRE_QWEN4_EXPERT_HOTSET"], "x=y")
        self.assertNotIn("HIPFIRE_FOO", env)
        shown = collect.explicit_env(cfg, kids[0])
        self.assertEqual(shown["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "14")
        self.assertEqual(shown["HIPFIRE_QWEN4_EXPERT_HOTSET"], "x=y")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = collect.run(
                self.fx.argv("quality", "--dry-run", "--run-id", "rid", "--child-env",
                             "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14", runs="1"), {},
            )
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("# child-env: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14\n", text)
        self.assertIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14 ", text)
        # without the option: default is empty and the variable is absent
        cfg0, kids0 = self.plan("quality")
        self.assertEqual(cfg0.child_env, {})
        self.assertNotIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", collect.build_child_env({}, cfg0, kids0[0]))

    def test_child_env_owned_keys_refused(self):
        owned = [
            *collect.KEEP_HIPFIRE_ENV, "HIPFIRE_HOME", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_GRAPH",
            "HIPFIRE_REPLAY_BACKEND", "HIPFIRE_STATE_QUANT", "HIPFIRE_GDN_TEST_STATE_QUANT",
            *(k for k, _ in collect.FIXED_RECEIPT_KNOBS),
        ]
        self.assertIn("HIPFIRE_PLE_FORMAT", owned)
        for key in owned:
            with self.assertRaisesRegex(collect.CollectError, f"--child-env {key}: the collector sets {key}"):
                self.fx.config("quality", "--child-env", f"{key}=1")

    def test_child_env_malformed_values_refused(self):
        cases = (
            ("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "must be KEY=VALUE"),
            ("=1", "empty key"),
            ("PATH=/x", "must match"),
            ("HIPFIRE_lower=1", "must match"),
            ("HIPFIRE_=1", "must match"),
        )
        for raw, why in cases:
            with self.assertRaisesRegex(collect.CollectError, why):
                self.fx.config("quality", "--child-env", raw)
        with self.assertRaisesRegex(collect.CollectError, "HIPFIRE_QWEN4_EXPERT_HOTSET is given twice"):
            self.fx.config(
                "quality", "--child-env", "HIPFIRE_QWEN4_EXPERT_HOTSET=1", "--child-env", "HIPFIRE_QWEN4_EXPERT_HOTSET=2"
            )

    def test_stray_parent_hipfire_env_refused_unless_passed_through(self):
        with self.assertRaises(collect.CollectError) as cm:
            self.fx.config(
                "quality",
                env={"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14", "HIPFIRE_QWEN4_EXPERT_HOTSET": "1", "PATH": "/usr/bin"},
            )
        self.assertEqual(
            str(cm.exception),
            "parent environment sets HIPFIRE_QWEN4_EXPERT_HOTSET, HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS; the child scrub would drop "
            "them silently: pass the ones a child needs with --child-env KEY=VALUE and unset the rest",
        )
        for stage in ("smoke", "memory", "perf", "greedy"):
            with self.assertRaisesRegex(collect.CollectError, "parent environment sets HIPFIRE_QWEN4_EXPERT_HOTSET"):
                self.fx.config(stage, env={"HIPFIRE_QWEN4_EXPERT_HOTSET": "1"})
        # a parent variable that --child-env passes explicitly is not a silent drop
        cfg = self.fx.config(
            "quality", "--child-env", "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14",
            env={"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14"},
        )
        self.assertEqual(cfg.child_env, {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14"})

    def test_parent_env_with_ignored_prefix_and_keep_names_accepted(self):
        # Built from the constant: a literal host tool name would enter the env inventory.
        torch_python = collect.IGNORED_PARENT_HIPFIRE_PREFIXES[0] + "PYTHON"
        cfg = self.fx.config(
            "quality",
            env={
                torch_python: "/opt/torch/bin/python", "HIPFIRE_DEVICES": "0",
                "HIPFIRE_LOCK_DIR": "/parent/lock", "HIPFIRE_FABRIC_ADMISSION": "x",
            },
        )
        self.assertEqual(cfg.child_env, {})

    def test_analyze_and_prehash_not_refused_by_stray_parent_env(self):
        stray = {"HIPFIRE_QWEN4_EXPERT_HOTSET": "1"}
        args = collect.make_parser().parse_args(["--stage", "quality", "--analyze", "--out", str(self.root)])
        self.assertEqual(collect.validate(args, stray).stage, "quality")
        args = collect.make_parser().parse_args(self.fx.argv("quality", "--prehash"))
        self.assertEqual(collect.validate(args, stray).stage, "quality")


class StatisticsTests(unittest.TestCase):
    def test_constant_delta_has_degenerate_exact_ci(self):
        mean, lo, hi = collect.bootstrap_mean_ci([0.002] * 24)
        self.assertAlmostEqual(mean, 0.002, places=15)
        self.assertAlmostEqual(lo, 0.002, places=12)
        self.assertAlmostEqual(hi, 0.002, places=12)

    def test_symmetric_delta_ci_straddles_zero_and_is_deterministic(self):
        deltas = [0.001, -0.001] * 12
        first = collect.bootstrap_mean_ci(deltas)
        second = collect.bootstrap_mean_ci(deltas)
        self.assertEqual(first, second)
        mean, lo, hi = first
        self.assertAlmostEqual(mean, 0.0, places=15)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertLess(hi, 0.001)
        self.assertEqual(collect.rule_applied(lo, hi), "PASS")

    def test_known_result_two_point_sample(self):
        # n=2, values {0, 1}: resample mean takes 0, .5, 1 with p .25/.5/.25,
        # so the 2.5th percentile is 0 and the 97.5th is 1 (10000 draws).
        mean, lo, hi = collect.bootstrap_mean_ci([0.0, 1.0])
        self.assertEqual(mean, 0.5)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)

    def test_seed_changes_resamples_but_is_recorded(self):
        a = collect.bootstrap_mean_ci([0.0, 0.5, 1.0, 2.0], resamples=50, seed=1)
        b = collect.bootstrap_mean_ci([0.0, 0.5, 1.0, 2.0], resamples=50, seed=2)
        self.assertNotEqual(a[1:], b[1:])
        self.assertEqual(collect.BOOTSTRAP_RESAMPLES, 10000)
        self.assertIsInstance(collect.BOOTSTRAP_SEED, int)

    def test_percentile_linear(self):
        self.assertEqual(collect.percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertEqual(collect.percentile([1.0, 2.0, 3.0, 4.0], 0), 1.0)
        self.assertEqual(collect.percentile([1.0, 2.0, 3.0, 4.0], 100), 4.0)

    def test_rule_edges(self):
        self.assertEqual(collect.RULE_TEXT, "lo<=0 -> PASS; lo>0 -> PASS iff hi<=+0.010")
        self.assertEqual(collect.rule_applied(0.0, 5.0), "PASS")  # lo exactly 0 passes whatever hi is
        self.assertEqual(collect.rule_applied(-0.003, 0.02), "PASS")
        self.assertEqual(collect.rule_applied(0.0001, 0.010), "PASS")  # hi exactly at the budget
        self.assertEqual(collect.rule_applied(0.0001, 0.0101), "FAIL")
        self.assertEqual(collect.rule_applied(1e-12, 0.0), "PASS")

    def test_paired_delta_pass_and_fail(self):
        base = [(0.010, 0.1, 1.0)] * 24
        better = [(0.008, 0.1, 1.0)] * 24
        worse = [(0.030, 0.1, 1.0)] * 24
        r = collect.paired_delta(better, base)
        self.assertEqual(r["rule_applied"], "PASS")
        self.assertAlmostEqual(r["mean_delta"], -0.002, places=12)
        r = collect.paired_delta(worse, base)
        self.assertEqual(r["rule_applied"], "FAIL")
        self.assertAlmostEqual(r["ci95_lo"], 0.020, places=12)
        self.assertIn("error", collect.paired_delta(base[:3], base))
        self.assertIn("error", collect.paired_delta([(math.nan, 0.1, 1.0)] * 24, base))

    def test_gap_closure(self):
        r = collect.gap_closure(1.0, 0.0, 0.25)
        self.assertEqual(r["closure"], 0.75)
        self.assertAlmostEqual(r["closure"], 0.75, places=12)
        self.assertTrue(r["closure_ge_0.75"])
        r = collect.gap_closure(0.0100, 0.0080, 0.0090)
        self.assertAlmostEqual(r["closure"], 0.5, places=12)
        self.assertFalse(r["closure_ge_0.75"])
        self.assertIsNone(collect.gap_closure(0.008, 0.010, 0.009)["closure"])

    def test_arm_stats_counts_nonfinite(self):
        s = collect.arm_stats([(0.01, 0.02, 1.0), (math.nan, 0.02, math.inf), (0.01, math.inf, 1.0)])
        self.assertEqual(s["nonfinite_kld_values"], 2)
        self.assertEqual(s["nonfinite_nll_values"], 1)
        self.assertIsNone(s["p99_kld_of_chunk_p99"])


class KldseqTests(TempCase):
    def test_parse_real_v2_layout_built_by_hand(self):
        path = self.root / "a.kldseq"
        payload = struct.pack("<ddd", 0.5, 1.5, 2.5) + struct.pack("<ddd", 0.25, 0.75, 3.5)
        path.write_bytes(b"HFKSEQ\0\0" + struct.pack("<I", 2) + struct.pack("<I", 2) + struct.pack("<I", 0) + payload)
        seq = collect.read_kldseq(path)
        self.assertEqual(seq["version"], 2)
        self.assertEqual(seq["chunks"], [(0.5, 1.5, 2.5), (0.25, 0.75, 3.5)])
        self.assertEqual(seq["sha256"], sha256_bytes(path.read_bytes()))

    def test_parse_v1_fills_nan_nll(self):
        path = self.root / "v1.kldseq"
        path.write_bytes(b"HFKSEQ\0\0" + struct.pack("<III", 1, 1, 0) + struct.pack("<dd", 0.1, 0.2))
        seq = collect.read_kldseq(path)
        self.assertEqual(seq["chunks"][0][:2], (0.1, 0.2))
        self.assertTrue(math.isnan(seq["chunks"][0][2]))

    def test_writer_roundtrip_and_rejections(self):
        path = self.root / "w.kldseq"
        collect.write_kldseq(path, [(1.0, 2.0, 3.0)] * 3)
        self.assertEqual(len(collect.read_kldseq(path)["chunks"]), 3)
        data = path.read_bytes()
        for bad, text in (
            (data[:10], "truncated header"),
            (b"XXXXXXXX" + data[8:], "magic"),
            (data[:-1], "truncated payload"),
            (data + b"\0", "trailing"),
            (data[:8] + struct.pack("<I", 9) + data[12:], "version"),
        ):
            bad_path = self.root / "bad.kldseq"
            bad_path.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, text):
                collect.read_kldseq(bad_path)

    def test_parse_eval_stderr_in_real_format(self):
        text = (
            "qwen4_kld: ple rows: q4 from embedded model file\n"
            "qwen4_kld eval: 24 chunks x 512 (decode route); teacher PPL 5.1234 (/refs/t.hfq)\n"
            "  chunk 1/24  KLD 0.010000  running KLD 0.010000  3s\n"
            "  chunk 2/24  KLD 0.020000  running KLD 0.015000  6s\n"
            "qwen4_kld eval: mean KLD = 0.015000  mean NLL = 1.500000  PPL = 4.4817  top1 = 0.8123  "
            "(12288 tokens)  logits sha256 " + "ab" * 32 + " -> /o/out.kldseq\n"
        )
        p = collect.parse_kld_stderr(text)
        self.assertEqual(p["headers"][0], {"chunks": 24, "n_ctx": 512, "route": "decode", "teacher_ppl": 5.1234})
        s = p["summaries"][0]
        self.assertEqual((s["mean_kld"], s["mean_nll"], s["top1"], s["tokens"]), (0.015, 1.5, 0.8123, 12288))
        self.assertEqual(s["logits_sha256"], "ab" * 32)
        self.assertEqual(s["output"], "/o/out.kldseq")
        self.assertEqual(p["chunk_lines"], 2)


def synth_quality_run(root: Path, modes_deltas: dict, runs: int = 1, same_bytes: bool = True) -> tuple[Path, list[dict]]:
    """Write a run dir of real-format .kldseq files and the children records."""
    run_dir = root / "quality" / "rid"
    children = []
    for run in range(runs):
        for mode, delta in modes_deltas.items():
            for corpus, route, name in (("code24", "decode", "out.kldseq"), ("wt2", "append256", "out.a256.kldseq")):
                cdir = run_dir / f"run{run}" / "g0" / mode / f"{corpus}-{route}"
                cdir.mkdir(parents=True)
                jitter = 0.0 if same_bytes else 1e-6 * run
                chunks = [(0.010 + delta + jitter + 0.0001 * (i % 4), 0.05, 1.5) for i in range(24)]
                collect.write_kldseq(cdir / name, chunks)
                body = f"qwen4_kld eval: mean KLD = 0.010000  mean NLL = 1.5  PPL = 4.4  top1 = 0.9  (1 tokens)  logits sha256 {'cd' * 32} -> x\n"
                (cdir / "stderr.txt").write_text(body)
                children.append({"id": f"run{run}/g0/{mode}/{corpus}-{route}", "kind": "kld_eval", "dir": str(cdir.relative_to(run_dir)),
                                 "mode": mode, "graph": "0", "corpus": corpus, "route": route, "run": run, "kldseq": name})
    return run_dir, children


class QualityAnalysisTests(TempCase):
    def test_pairs_closure_and_determinism(self):
        run_dir, children = synth_quality_run(self.root, {"q8": 0.0, "fp32": -0.0020, "q8_ef": -0.0015}, runs=2)
        a = collect.analyze_quality(run_dir, children)
        self.assertEqual(a["rule"], collect.RULE_TEXT)
        self.assertEqual(a["errors"], [])
        pairs = {(p["corpus"], p["route"], p["candidate"], p["run"]): p for p in a["pairs"]}
        self.assertEqual(len(pairs), 2 * 2 * 2)
        p = pairs[("code24", "decode", "fp32", 0)]
        self.assertEqual(p["baseline"], "q8")
        self.assertAlmostEqual(p["mean_delta"], -0.0020, places=12)
        self.assertAlmostEqual(p["ci95_hi"], -0.0020, places=9)
        self.assertEqual(p["rule_applied"], "PASS")
        self.assertEqual(p["bootstrap"]["seed"], collect.BOOTSTRAP_SEED)
        self.assertTrue(a["pairs_rule_applied_all_pass"])
        closures = a["code24_decode_gap_closure"]
        self.assertEqual(len(closures), 2)
        self.assertAlmostEqual(closures[0]["closure"], 0.75, places=9)
        for d in a["same_route_determinism"]:
            self.assertTrue(d["kldseq_byte_identical"])
            self.assertEqual(len(d["kldseq_sha256"]), 2)
            self.assertTrue(d["logits_sha256_identical"])
        arm = a["arms"]["g0/code24/decode"]["fp32"]["run0"]
        self.assertEqual(arm["nonfinite_kld_values"], 0)
        self.assertAlmostEqual(arm["p99_kld_of_chunk_p99"], 0.05, places=12)
        self.assertEqual(arm["top1_from_stderr"], 0.9)

    def test_nondeterminism_and_regression_are_reported(self):
        run_dir, children = synth_quality_run(self.root, {"q8": 0.0, "fp32": 0.020}, runs=2, same_bytes=False)
        a = collect.analyze_quality(run_dir, children)
        self.assertTrue(all(d["kldseq_byte_identical"] is False for d in a["same_route_determinism"]))
        self.assertTrue(all(p["rule_applied"] == "FAIL" for p in a["pairs"]))
        self.assertFalse(a["pairs_rule_applied_all_pass"])
        self.assertEqual(a["code24_decode_gap_closure"], [])

    def test_analyze_cli_recomputes_from_existing_out(self):
        run_dir, children = synth_quality_run(self.root, {"q8": 0.0, "fp32": -0.001})
        (run_dir / "children.json").write_text(json.dumps(children))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run(["--stage", "quality", "--analyze", "--out", str(self.root)], {})
        self.assertEqual(rc, 0)
        self.assertTrue((run_dir / "analysis.json").is_file())
        self.assertEqual(json.loads(buf.getvalue())["rule"], collect.RULE_TEXT)
        with self.assertRaisesRegex(collect.CollectError, "absolute"):
            collect.run(["--stage", "quality", "--analyze", "--out", "rel"], {})
        with self.assertRaisesRegex(collect.CollectError, "quality or perf"):
            collect.run(["--stage", "smoke", "--analyze", "--out", str(self.root)], {})


class PerfAnalysisTests(unittest.TestCase):
    def test_bench_extraction_and_loss(self):
        standard = {"protocol": "native-generate-v1", "decode_tok_s": {"median": 50.0}, "prefill_tok_s": {"median": 800.0},
                    "ttft_ms": {"median": 20.0}, "spec_tau": [3.0, 5.0], "prompt_tokens": 100}
        o = collect.extract_bench(standard)
        self.assertEqual((o["decode_tok_s"], o["prefill_tok_s"]), (50.0, 800.0))
        ttft = collect.extract_bench({"protocol": "client-ttft-v1", "ttft": {"median": 1000.0}, "pp_tok_s": {"median": 4000.0}})
        self.assertEqual((ttft["ttft_ms"], ttft["pp_tok_s"]), (1000.0, 4000.0))
        matrix = collect.extract_bench({
            "protocol": "synthetic-pp-tg-matrix-v1",
            "prefill": [{"tokens": 8192, "stats": {"median": 900.0}}],
            "decode": [{"context": 4096, "tokens": 32, "stats": {"median": 40.0}}],
        })
        self.assertEqual(collect._matrix_pp(matrix), 900.0)
        self.assertEqual(collect._matrix_tg(matrix), 40.0)

        def child(mode, run, label, kind, obs):
            return {"id": f"{mode}{run}{label}", "kind": kind, "label": label, "mode": mode, "run": run, "graph": "unset", "observations": obs}

        kids = []
        for run, (q8, fp) in enumerate([(100.0, 90.0), (100.0, 96.0), (100.0, 94.0)]):
            kids.append(child("q8", run, "stateless-spec-off", "bench_standard", collect.extract_bench({**standard, "decode_tok_s": {"median": q8}})))
            kids.append(child("fp32", run, "stateless-spec-off", "bench_standard", collect.extract_bench({**standard, "decode_tok_s": {"median": fp}})))
            kids.append(child("q8", run, "ttft-long", "bench_ttft", collect.extract_bench({"protocol": "client-ttft-v1", "ttft": {"median": 1000.0}, "pp_tok_s": {"median": 4000.0}})))
            kids.append(child("fp32", run, "ttft-long", "bench_ttft", collect.extract_bench({"protocol": "client-ttft-v1", "ttft": {"median": 1100.0}, "pp_tok_s": {"median": 3800.0}})))
        a = collect.analyze_perf(kids)
        rows = {(r["label"], r["metric"], r["mode"]): r for r in a["rows"]}
        r = rows[("stateless-spec-off", "decode_tok_s", "fp32")]
        self.assertEqual(r["median"], 94.0)
        self.assertAlmostEqual(r["loss_vs_q8_of_medians"], 0.06, places=12)
        self.assertTrue(r["policy_line_observation"]["loss_of_medians_exceeds_line"])
        self.assertAlmostEqual(r["loss_vs_q8_median_of_paired_runs"], 0.06, places=12)
        t = rows[("ttft-long", "ttft_ms", "fp32")]
        self.assertAlmostEqual(t["loss_vs_q8_of_medians"], 0.10, places=12)  # lower is better: +10% TTFT is a loss
        p = rows[("ttft-long", "pp_tok_s", "fp32")]
        self.assertAlmostEqual(p["loss_vs_q8_of_medians"], 0.05, places=12)
        self.assertNotIn("loss_vs_q8_of_medians", rows[("ttft-long", "ttft_ms", "q8")])

    def test_find_json_object_skips_banner(self):
        text = 'hipfire bench\n{ not json\n{"protocol": "x", "a": {"b": 1}}\ntrailing\n'
        self.assertEqual(collect.find_json_object(text), {"protocol": "x", "a": {"b": 1}})
        self.assertIsNone(collect.find_json_object("no json here"))

    def test_vram_extraction(self):
        amd = [{"gpu": 0, "mem_usage": {"total_vram": {"value": 98304, "unit": "MB"}, "used_vram": {"value": 1234, "unit": "MB"}}}]
        self.assertEqual(collect.extract_used_vram_mib(amd), [1234.0])
        rocm = {"card0": {"VRAM Total Memory (B)": "1048576", "VRAM Total Used Memory (B)": "2097152"}}
        self.assertEqual(collect.extract_used_vram_mib(rocm), [2.0])
        s = collect.summarize_memory(
            [{"mem_available_kib": 1000, "gpu_used_vram_mib": [10.0]},
             {"mem_available_kib": 400, "gpu_used_vram_mib": [50.0]},
             {"mem_available_kib": 900, "gpu_used_vram_mib": [20.0]}], None)
        self.assertEqual(s["host_peak_drop_kib"], 600)
        self.assertEqual(s["gpu_used_vram_mib_peak_per_device"], [50.0])
        self.assertEqual(s["mem_available_kib_after"], 900)


class TestOutputTests(TempCase):
    OK = "running 1 test\ntest sampled_x ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1s\n"

    def test_cargo_parse_and_classify(self):
        p = collect.parse_cargo_test(self.OK)
        self.assertEqual((p["passed"], p["failed"]), (1, 0))
        self.assertEqual(collect.classify_test(0, False, p, ("sampled_x",))[0], "ok")
        self.assertEqual(collect.classify_test(0, False, p, ("other",))[0], "incomplete")
        self.assertEqual(collect.classify_test(1, False, p, ())[0], "failed")
        self.assertEqual(collect.classify_test(None, True, p, ())[0], "timeout")
        none = collect.parse_cargo_test("running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0s\n")
        self.assertEqual(collect.classify_test(0, False, none, ())[0], "incomplete")
        skipped = collect.parse_cargo_test("running 1 test\ntest sampled_x ... ignored\n\ntest result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n")
        self.assertEqual(collect.classify_test(0, False, skipped, ("sampled_x",))[0], "incomplete")
        self.assertEqual(collect.classify_test(0, False, collect.parse_cargo_test("garbage"), ())[0], "incomplete")

    def test_status_split_by_nested_nocapture_child(self):
        # j1749 disk_snapshots_restore_after_restart_on_flash_next stdout shape: the test re-runs
        # itself as a --nocapture child, whose output lands between the parent's `test NAME ... `
        # header and its status, so both statuses arrive on bare lines (child first).
        name = "disk_snapshots_restore_after_restart_on_flash_next"
        res = "test result: {}. 1 passed; {} failed; 0 ignored; 0 measured; 11 filtered out; finished in 1s\n"
        nested = (f"\nrunning 1 test\ntest {name} ... \nrunning 1 test\n"
                  f"test {name} ... HIPFIRE_GDN_TEST_STATE_QUANT state_quant=fp32 gdn_format=f32\nok\n\n"
                  + res.format("ok", 0) + "\nok\n\n" + res.format("ok", 0))
        p = collect.parse_cargo_test(nested)
        self.assertEqual(p["tests"], [{"name": name, "status": "ok"}, {"name": name, "status": "ok"}])
        self.assertEqual(collect.classify_test(0, False, p, (name,)), ("ok", []))
        failed = nested.replace("\n\nok\n\n" + res.format("ok", 0), "\n\nFAILED\n\n" + res.format("FAILED", 1))
        p = collect.parse_cargo_test(failed)
        self.assertEqual([t["status"] for t in p["tests"]], ["ok", "FAILED"])
        self.assertEqual(collect.classify_test(0, False, p, (name,))[0], "failed")
        # A header whose status never arrives stays incomplete.
        cut = f"running 1 test\ntest {name} ... \n" + res.format("ok", 0)
        self.assertEqual(collect.classify_test(0, False, collect.parse_cargo_test(cut), (name,))[0], "incomplete")

    def sampled_child(self, cfg_dir: Path, stdout: str, rc=0, coverage=None, config_text="mode q8_ef", graph="unset"):
        cdir = cfg_dir
        (cdir / "sampled").mkdir(parents=True)
        (cdir / "stdout.txt").write_text(stdout)
        (cdir / "stderr.txt").write_text("")
        (cdir / "sampled" / "arm.config.json").write_text(json.dumps({"env": config_text}))
        if coverage is not None:
            (cdir / "sampled" / "q8ef_coverage.json").write_text(json.dumps({"coverage": coverage}))
        child = collect.Child(
            id="x", label="sampled-law", kind="sampled_test", run=0, graph=graph, mode="q8_ef", dir=cdir,
            argv=[], extra_env={}, binary="b", expect_tests=("sampled_x",), ef_sampled=True,
        )
        return collect.observe_child(child, cdir, rc, False, graph)

    def test_q8_ef_sampled_fails_closed(self):
        full = {k: 3 for k in collect.Q8EF_REQUIRED_COVERAGE}
        outcome, obs = self.sampled_child(self.root / "a", self.OK, coverage=full)
        self.assertEqual((outcome, obs["g5_result"]), ("ok", "PASS"))
        outcome, obs = self.sampled_child(self.root / "b", self.OK, coverage=None)
        self.assertEqual((outcome, obs["g5_result"]), ("incomplete", "INCOMPLETE"))
        partial = dict(full, reject_keep0=0)
        outcome, obs = self.sampled_child(self.root / "c", self.OK, coverage=partial)
        self.assertEqual((outcome, obs["g5_result"]), ("incomplete", "INCOMPLETE"))
        self.assertIn("reject_keep0", obs["reasons"][0])
        outcome, obs = self.sampled_child(self.root / "d", self.OK.replace("1 passed", "0 passed"), coverage=full)
        self.assertEqual(obs["g5_result"], "INCOMPLETE")
        outcome, obs = self.sampled_child(self.root / "e", self.OK, rc=101, coverage=full)
        self.assertEqual((outcome, obs["g5_result"]), ("failed", "FAIL"))
        outcome, obs = self.sampled_child(self.root / "f", self.OK, coverage=full, config_text="mode q8")
        self.assertEqual(obs["g5_result"], "INCOMPLETE")
        # graph-on arms must also show capture/replay counters
        outcome, obs = self.sampled_child(self.root / "g", self.OK, coverage=full, graph="1")
        self.assertEqual(obs["g5_result"], "INCOMPLETE")
        both = dict(full, graph_capture=1, graph_replay=4)
        outcome, obs = self.sampled_child(self.root / "h", self.OK, coverage=both, graph="1")
        self.assertEqual(obs["g5_result"], "PASS")


class EndToEndTests(TempCase):
    def test_quality_stage_with_fake_binary(self):
        fx = Fixture(self.root, fake_kld=True)
        cfg = fx.config("quality", modes="q8,fp32,q8_ef", runs="2")
        cfg.dry_run = False
        environ = {"PATH": os.environ.get("PATH", ""), "FABRIC_JOB": "1:2:halo", "HIPFIRE_KV_MODE": "bf16", "HIPFIRE_STATE_QUANT": "bogus"}
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, environ)
        self.assertEqual(rc, 0)
        run_dir = cfg.run_dir
        receipt = json.loads((run_dir / "receipt.json").read_text())
        self.assertEqual(len(receipt["children"]), 2 * 3 * 6)
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["not_run"], [])
        child_dir = run_dir / "run1" / "g0" / "fp32" / "code24-decode"
        for name in ("command.txt", "env.json", "identity.json", "stdout.txt", "stderr.txt", "rc.txt",
                     "start.txt", "end.txt", "observations.json", "out.kldseq"):
            self.assertTrue((child_dir / name).is_file(), name)
        env = json.loads((child_dir / "env.json").read_text())
        self.assertNotIn("HIPFIRE_KV_MODE", env)
        self.assertEqual(env["HIPFIRE_STATE_QUANT"], "fp32")
        self.assertIn("--decode", (child_dir / "command.txt").read_text())
        ident = json.loads((child_dir / "identity.json").read_text())
        model = ident["inputs"]["model"]
        self.assertEqual(model["md5"], hashlib.md5(b"model").hexdigest())
        self.assertEqual(model["sha256"], sha256_bytes(b"model"))
        self.assertIn("model.ple-q4", ident["inputs"])
        self.assertIn("qwen4_kld", ident["binaries"])
        self.assertEqual(ident["host"]["FABRIC_JOB"], "1:2:halo")
        a = receipt["analysis"]
        self.assertEqual(a["errors"], [])
        pairs = [p for p in a["pairs"] if p["candidate"] == "fp32" and p["corpus"] == "code24" and p["route"] == "decode" and p["run"] == 0]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["rule_applied"], "PASS")
        self.assertAlmostEqual(pairs[0]["mean_delta"], -0.0020, places=9)
        self.assertEqual(len(a["code24_decode_gap_closure"]), 2)
        self.assertAlmostEqual(a["code24_decode_gap_closure"][0]["closure"], 0.75, places=6)
        self.assertTrue(all(d["kldseq_byte_identical"] for d in a["same_route_determinism"]))
        self.assertIn("lo<=0 -> PASS; lo>0 -> PASS iff hi<=+0.010", (run_dir / "report.md").read_text())
        # a second invocation into the same run dir is refused
        with self.assertRaisesRegex(collect.CollectError, "already exists"):
            collect.validate(collect.make_parser().parse_args(fx.argv("quality", "--run-id", "rid")), {"FABRIC_JOB": "1:2:halo"})

    def test_child_env_reaches_child_and_is_recorded(self):
        fx = Fixture(self.root, fake_kld=True)
        cfg = fx.config("quality", "--only", "code24-decode", "--child-env", "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14")
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", ""), "FABRIC_JOB": "1:2:halo"})
        self.assertEqual(rc, 0)
        receipt = json.loads((cfg.run_dir / "receipt.json").read_text())
        self.assertEqual(receipt["child_env"], {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14"})
        child_dir = cfg.run_dir / receipt["children"][0]["dir"]
        env = json.loads((child_dir / "env.json").read_text())
        self.assertEqual(env["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "14")
        ident = json.loads((child_dir / "identity.json").read_text())
        self.assertEqual(ident["child"]["child_env"], {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "14"})

    def test_failing_child_stops_and_exits_nonzero(self):
        fx = Fixture(self.root)  # fake qwen4_kld is a shell stub that prints nothing and writes no .kldseq
        cfg = fx.config("quality", runs="1")
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", "")})
        self.assertEqual(rc, 1)
        receipt = json.loads((cfg.run_dir / "receipt.json").read_text())
        self.assertEqual(len(receipt["children"]), 1)
        self.assertEqual(receipt["children"][0]["outcome"], "failed")
        self.assertEqual(len(receipt["not_run"]), 11)

    def test_child_timeout_is_recorded(self):
        fx = Fixture(self.root)
        slow = fx.bin / "qwen4_kld"
        content = b"#!/bin/sh\nsleep 30\n"
        slow.write_bytes(content)
        lines = [l for l in (fx.bin / "BINARIES.sha256").read_text().splitlines() if not l.endswith("  qwen4_kld")]
        lines.append(f"{sha256_bytes(content)}  qwen4_kld")
        (fx.bin / "BINARIES.sha256").write_text("\n".join(lines) + "\n")
        cfg = fx.config("quality", "--child-timeout-secs", "1", runs="1")
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", "")})
        self.assertEqual(rc, 1)
        child = json.loads((cfg.run_dir / "receipt.json").read_text())["children"][0]
        self.assertTrue(child["timed_out"])
        self.assertEqual(child["outcome"], "timeout")
        self.assertTrue((Path(cfg.run_dir) / child["dir"] / "timeout.txt").is_file())


FAKE_ORACLE = """#!{python}
import json, sys
a = sys.argv
json.dump({{"formats": a[a.index("--formats") + 1], "graph": a[a.index("--graph") + 1], "all_keeps": "--all-keeps" in a}},
          open(a[a.index("--json") + 1], "w"))
print("oracle done")
"""


class MultiStageEndToEndTests(TempCase):
    ENV = {"PATH": os.environ.get("PATH", ""), "FABRIC_JOB": "1:2:halo", "FABRIC_CARDS": "halo"}

    def run_stage(self, fx, stage, *extra, modes="q8,fp32", runs="1"):
        cfg = fx.config(stage, *extra, modes=modes, runs=runs)
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, self.ENV)
        return rc, cfg, json.loads((cfg.run_dir / "receipt.json").read_text())

    def test_smoke_stage(self):
        fx = Fixture(self.root, fakes={"gdn_state_precision_oracle": FAKE_ORACLE, "greedy_mtp_identity_hw": FAKE_TEST})
        rc, cfg, receipt = self.run_stage(fx, "smoke", "--graph", "0,1", modes="q8,fp32,q8_ef")
        self.assertEqual(rc, 0, receipt["children"])
        oracle = [c for c in receipt["children"] if c["kind"] == "oracle"]
        self.assertEqual([c["graph"] for c in oracle], ["0", "1"])
        self.assertEqual(oracle[0]["observations"]["oracle_json_scalars"]["formats"], "q8,fp32,q8_ef")
        ident = [c for c in receipt["children"] if c["kind"] == "identity_test"]
        self.assertEqual(len(ident), 4)
        self.assertIn("identical to ar", list(ident[0]["observations"]["summary_txt"].values())[0])
        self.assertEqual(ident[0]["graph_axis"], "HIPFIRE_REPLAY_BACKEND")
        env = json.loads((Path(cfg.run_dir) / ident[0]["dir"] / "env.json").read_text())
        self.assertEqual(env["HIPFIRE_GDN_TEST_STATE_QUANT"], "q8")
        self.assertNotIn("HIPFIRE_STATE_QUANT", env)

    def test_restore_stage_selects_ef_test_for_q8_ef(self):
        fx = Fixture(self.root, fakes={k: FAKE_TEST for k in ("session_cache_hw", "greedy_mtp_identity_hw", "q8_ef_product_routes_hw")})
        rc, cfg, receipt = self.run_stage(fx, "restore", "--graph", "0", modes="q8,q8_ef")
        self.assertEqual(rc, 0, receipt["children"])
        kinds = [(c["mode"], c["kind"]) for c in receipt["children"]]
        self.assertEqual(
            sorted(kinds),
            sorted([("q8", "session_test")] * len(collect.SESSION_CACHE_TESTS)
                   + [("q8", "identity_test"), ("q8_ef", "ef_routes_test")]),
        )
        # Each session_cache_hw test ran alone in its own process.
        sess = [c for c in receipt["children"] if c["kind"] == "session_test"]
        self.assertEqual([c["id"].rsplit("/", 1)[1] for c in sess], list(collect.SESSION_CACHE_TESTS))

    def test_memory_stage_records_daemon_lines_and_samples(self):
        fx = Fixture(self.root, fakes={"hipfire": FAKE_HIPFIRE})
        rc, cfg, receipt = self.run_stage(fx, "memory")
        self.assertEqual(rc, 0, receipt["children"])
        obs = receipt["children"][0]["observations"]
        self.assertTrue(any("qwen4 state:" in line for line in obs["daemon_log_lines"]))
        self.assertTrue(any("card_cap" in line for line in obs["daemon_log_lines"]))
        self.assertGreaterEqual(obs["memory"]["samples"], 2)
        sample_file = Path(cfg.run_dir) / receipt["children"][0]["dir"] / "memory_samples.jsonl"
        self.assertTrue(sample_file.is_file())
        self.assertEqual(obs["protocol"], "native-generate-v1")

    def test_perf_stage_reports_medians_and_loss(self):
        fx = Fixture(self.root, fakes={"hipfire": FAKE_HIPFIRE})
        rc, cfg, receipt = self.run_stage(fx, "perf", runs="3")
        self.assertEqual(rc, 0, receipt["children"])
        self.assertEqual(len(receipt["children"]), 3 * 2 * 4)
        rows = {(r["label"], r["metric"], r["mode"]): r for r in receipt["analysis"]["rows"]}
        r = rows[("stateless-spec-off", "decode_tok_s", "fp32")]
        self.assertEqual(r["median"], 90.0)
        self.assertAlmostEqual(r["loss_vs_q8_of_medians"], 0.1, places=12)
        self.assertTrue(r["policy_line_observation"]["loss_of_medians_exceeds_line"])
        self.assertAlmostEqual(rows[("matrix-pp8192-tg32", "pp8192_tok_s_synthetic", "fp32")]["loss_vs_q8_of_medians"], 0.1, places=12)
        self.assertEqual(rows[("stateless-spec-mtp", "tau", "q8")]["median"], 4.0)
        self.assertIn("not a verdict", (cfg.run_dir / "report.md").read_text())

    def test_sampled_stage_q8_ef_fails_closed_without_coverage(self):
        fx = Fixture(self.root, fakes={"sampled_mtp_distribution_hw": FAKE_TEST})
        rc, cfg, receipt = self.run_stage(fx, "sampled", modes="q8_ef")
        self.assertEqual(rc, 1)
        child = receipt["children"][0]
        self.assertEqual(child["outcome"], "incomplete")
        self.assertEqual(child["observations"]["g5_result"], "INCOMPLETE")


class OracleOnlyTests(TempCase):
    def test_oracle_only_commands_need_only_the_oracle_binary(self):
        fx = Fixture(self.root, omit=("greedy_mtp_identity_hw",))
        args = collect.make_parser().parse_args(
            fx.argv("smoke", "--dry-run", "--oracle-only", "--run-id", "rid", modes="q8,fp32,q8_ef", runs="1")
        )
        cfg = collect.validate(args, {})
        kids = collect.plan_children(cfg)
        self.assertEqual([k.kind for k in kids], ["oracle", "oracle"])
        self.assertEqual([k.graph for k in kids], ["0", "1"])
        self.assertEqual(kids[1].argv[1:7], ["--formats", "q8,fp32,q8_ef", "--graph", "1", "--max-rows", "32"])
        self.assertEqual(collect.input_roles(cfg), [])
        self.assertEqual(sorted(cfg.binaries), ["gdn_state_precision_oracle"])

    def test_oracle_only_is_smoke_only(self):
        fx = Fixture(self.root)
        with self.assertRaisesRegex(collect.CollectError, "smoke"):
            collect.run(fx.argv("quality", "--dry-run", "--oracle-only"), {})

    def test_oracle_only_hold_needs_wrapper_but_not_hostmem_lock(self):
        chain = [100, 90]
        self.assertIsNone(collect.hold_refusal("90:7:halo", chain, lambda p: 7, lambda pids: False, False))
        self.assertIn("hostmem", collect.hold_refusal("90:7:halo", chain, lambda p: 7, lambda pids: False))
        self.assertIn("not an ancestor", collect.hold_refusal("5:7:halo", chain, lambda p: 7, lambda pids: False, False))
        self.assertIn("no fabric hold", collect.hold_refusal("", chain, lambda p: 7, lambda pids: False, False))
        fx = Fixture(self.root)
        with self.assertRaisesRegex(collect.CollectError, "no fabric hold"):
            collect.run(fx.argv("smoke", "--oracle-only", "--graph", "0"), {})

    def test_oracle_only_stage_records_rc_verbatim(self):
        fx = Fixture(self.root)  # stub oracle writes no JSON and exits 0
        cfg = fx.config("smoke", "--oracle-only", "--graph", "0", runs="1")
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", "")})
        self.assertEqual(rc, 1)
        receipt = json.loads((cfg.run_dir / "receipt.json").read_text())
        self.assertEqual(receipt["children"][0]["rc"], 0)
        self.assertEqual(receipt["children"][0]["outcome"], "failed")  # no oracle.json: never inferred ok
        ident = json.loads((cfg.run_dir / "identity.json").read_text())
        self.assertEqual(ident["inputs"], {})


class OnlyAndMaxChunksTests(TempCase):
    def setUp(self):
        super().setUp()
        self.fx = Fixture(self.root)

    def plan(self, stage, *extra, modes="q8,fp32", runs="1"):
        cfg = self.fx.config(stage, *extra, modes=modes, runs=runs)
        return cfg, collect.plan_children(cfg)

    def test_only_keeps_selected_labels_in_planned_order(self):
        _, full = self.plan("quality")
        _, kids = self.plan("quality", "--only", "wt2-whole,code24-decode")
        self.assertEqual(
            [k.id for k in kids],
            [k.id for k in full if k.label in ("code24-decode", "wt2-whole")],
        )
        self.assertEqual(len(kids), 4)
        self.assertEqual([k.label for k in kids][:2], ["code24-decode", "wt2-whole"])
        _, perf = self.plan("perf", "--only", "ttft-long")
        self.assertEqual([(k.mode, k.label) for k in perf], [("q8", "ttft-long"), ("fp32", "ttft-long")])
        _, smoke = self.plan("smoke", "--only", "oracle")
        self.assertEqual({k.label for k in smoke}, {"oracle"})
        self.assertEqual(self.plan("memory", "--only", "memory-mtp-ctx")[0].only, ("memory-mtp-ctx",))

    def test_only_refuses_unknown_and_unplanned_labels(self):
        with self.assertRaisesRegex(collect.CollectError, "no-such-label"):
            self.fx.config("quality", "--only", "code24-decode,no-such-label")
        # ef-product-routes is planned only for q8_ef
        with self.assertRaisesRegex(collect.CollectError, "ef-product-routes"):
            self.fx.config("restore", "--only", "ef-product-routes")
        _, kids = self.plan("restore", "--only", "ef-product-routes", "--graph", "0", modes="q8,q8_ef")
        self.assertEqual([k.mode for k in kids], ["q8_ef"])
        with self.assertRaisesRegex(collect.CollectError, "--only is empty"):
            self.fx.config("quality", "--only", " , ")
        with self.assertRaisesRegex(collect.CollectError, "repeats"):
            self.fx.config("quality", "--only", "code24-decode,code24-decode")
        # the refusal also fires through the CLI entry point, before any launch
        with self.assertRaisesRegex(collect.CollectError, "bogus"):
            collect.run(self.fx.argv("perf", "--dry-run", "--only", "bogus"), {})
        self.assertFalse(self.fx.out.exists())

    def test_only_is_recorded_in_config_and_dry_run(self):
        cfg = self.fx.config("quality", "--only", "code24-decode")
        self.assertEqual(cfg.only, ("code24-decode",))
        self.assertEqual(self.fx.config("quality").only, ())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = collect.run(self.fx.argv("quality", "--dry-run", "--only", "code24-decode,wt2-whole", runs="1"), {})
        self.assertEqual(rc, 0)
        self.assertIn("# only: code24-decode,wt2-whole", buf.getvalue())
        self.assertIn("# max-chunks: 24", buf.getvalue())
        self.assertEqual(buf.getvalue().count("\n[run"), 4)

    def test_max_chunks_propagates_and_defaults(self):
        _, kids = self.plan("quality", "--max-chunks", "6")
        for k in kids:
            i = k.argv.index("--max-chunks")
            self.assertEqual(k.argv[i + 1], "6")
        _, kids = self.plan("quality")
        self.assertEqual(kids[0].argv[kids[0].argv.index("--max-chunks") + 1], "24")
        self.assertEqual(self.fx.config("quality", "--max-chunks", "2").max_chunks, 2)

    def test_max_chunks_refusals(self):
        for bad in ("1", "0", "-3"):
            with self.assertRaisesRegex(collect.CollectError, "max-chunks"):
                self.fx.config("quality", "--max-chunks", bad)
        for stage in ("perf", "memory", "smoke", "greedy"):
            with self.assertRaisesRegex(collect.CollectError, "quality only"):
                self.fx.config(stage, "--max-chunks", "8", "--graph", "0")

    def test_filtered_quality_run_records_filter_and_analyzes(self):
        (self.root / "e2e").mkdir()
        fx = Fixture(self.root / "e2e", fake_kld=True)
        cfg = fx.config("quality", "--only", "code24-decode", "--max-chunks", "5", runs="1")
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", ""), "FABRIC_JOB": "1:2:halo"})
        self.assertEqual(rc, 0)
        receipt = json.loads((cfg.run_dir / "receipt.json").read_text())
        self.assertEqual(receipt["only"], ["code24-decode"])
        self.assertEqual(receipt["max_chunks"], 5)
        self.assertIsNone(receipt["greedy_tokens"])
        self.assertEqual([c["id"] for c in receipt["children"]], ["run0/g0/q8/code24-decode", "run0/g0/fp32/code24-decode"])
        self.assertEqual(receipt["not_run"], [])
        self.assertEqual(len(receipt["analysis"]["pairs"]), 1)
        self.assertEqual(receipt["analysis"]["errors"], [])
        self.assertIn("only code24-decode", (cfg.run_dir / "report.md").read_text())
        # --analyze of the filtered run dir still works
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(collect.run(["--stage", "quality", "--analyze", "--out", str(fx.out), "--run-id", "rid"], {}), 0)
        self.assertEqual(len(json.loads(buf.getvalue())["pairs"]), 1)

    def test_quality_analysis_reports_missing_cells_without_crashing(self):
        run_dir, children = synth_quality_run(self.root, {"q8": 0.0, "fp32": -0.002}, runs=1)
        children = [c for c in children if c["id"] != "run0/g0/fp32/wt2-append256"]
        a = collect.analyze_quality(run_dir, children)
        self.assertEqual(len(a["pairs"]), 1)  # only the code24-decode cell is paired
        self.assertEqual(
            a["missing_cells"],
            [{"graph": "0", "corpus": "wt2", "route": "append256", "missing_modes": ["fp32"]}],
        )
        self.assertTrue(any("wt2/append256" in e and "missing cell" in e for e in a["errors"]))
        # a lone non-baseline arm is an error, not a crash or a pass
        only_fp32 = [c for c in children if c["mode"] == "fp32"]
        b = collect.analyze_quality(run_dir, only_fp32)
        self.assertEqual(b["pairs"], [])
        self.assertTrue(b["errors"])
        self.assertIsNone(b["pairs_rule_applied_all_pass"])

    def test_perf_analysis_of_a_filtered_run(self):
        report = {"protocol": "client-ttft-v1", "prompt_tokens": 9, "ttft": {"median": 2000.0}, "pp_tok_s": {"median": 4500.0}}
        obs = collect.extract_bench(report)
        obs["protocol"] = report["protocol"]
        kids = [
            {"id": f"run0/gunset/{m}/ttft-long", "kind": "bench_ttft", "label": "ttft-long", "graph": "unset",
             "mode": m, "run": 0, "observations": dict(obs)}
            for m in ("q8", "fp32")
        ]
        rows = collect.analyze_perf(kids)["rows"]
        self.assertEqual({r["label"] for r in rows}, {"ttft-long"})
        self.assertEqual(collect.analyze_perf([])["rows"], [])


FAKE_HIPFIRE_RUN = """#!{python}
import json, os, sys
a = sys.argv
assert a[1] == "run" and a[a.index("--temp") + 1] == "0" and "--json" in a
spec = a[a.index("--spec") + 1]
mode = os.environ["HIPFIRE_STATE_QUANT"]
sys.stderr.write("daemon: max_seq " + a[a.index("--max-seq") + 1] + " card_cap 1000\\n")
sys.stderr.write("  qwen4 state: QSA fp8 K/V, GDN " + mode + " recurrent\\n")
text = "the quick brown fox"
if spec == "mtp" and mode == "q8":
    text = "the quick brown cat jumps"
print(json.dumps({{"content": text, "tokens": 5, "tok_s": 12.5, "finish_reason": "length"}}))
"""


def greedy_child(run, mode, spec, content, outcome="ok", tokens=None, lines=None, graph="1"):
    obs = {"kind": "greedy_run"}
    if content is not None:
        obs.update(content=content, content_sha256=sha256_bytes(content.encode()), tokens=tokens)
    obs["daemon_log_lines"] = ["daemon: max_seq 4096 card_cap 1000", "GDN unrelated line"] if lines is None else lines
    return {
        "id": f"run{run}/g{graph}/{mode}/greedy-spec-{spec}", "label": f"greedy-spec-{spec}", "kind": "greedy_run",
        "run": run, "graph": graph, "mode": mode, "outcome": outcome, "observations": obs,
    }


class GreedyStageTests(TempCase):
    def setUp(self):
        super().setUp()
        self.fx = Fixture(self.root)
        self.fx.prompt.write_text("line one\nline two\n\n")
        self.hip = str(self.fx.bin / BIN_NAMES["hipfire"])
        self.daemon = str(self.fx.bin / BIN_NAMES["daemon"])

    def greedy_argv(self, *extra, runs="1", modes="q8,fp32", drop=()):
        """Greedy needs no teachers, no long prompt."""
        argv = self.fx.argv("greedy", *extra, modes=modes, runs=runs)
        for flag in ("--wt2-ref", "--code-ref", "--long-prompt-file", *drop):
            i = argv.index(flag)
            del argv[i:i + 2]
        return argv

    def config(self, *extra, **kw):
        args = collect.make_parser().parse_args(self.greedy_argv("--dry-run", "--run-id", "rid", *extra, **kw))
        return collect.validate(args, {})

    def test_plan_argv_env_and_graph_default(self):
        cfg = self.config()
        kids = collect.plan_children(cfg)
        self.assertEqual(
            [k.id for k in kids],
            ["run0/g1/q8/greedy-spec-off", "run0/g1/q8/greedy-spec-mtp",
             "run0/g1/fp32/greedy-spec-off", "run0/g1/fp32/greedy-spec-mtp"],
        )
        k = kids[1]
        self.assertEqual(
            k.argv,
            [self.hip, "run", str(self.fx.model), "--temp", "0", "--spec", "mtp", "--max-tokens", "64",
             "--max-seq", "4096", "--json", "line one\nline two"],  # trailing newlines stripped
        )
        self.assertEqual(k.kind, "greedy_run")
        self.assertEqual(k.label, "greedy-spec-mtp")
        self.assertEqual(
            k.extra_env,
            {"HIPFIRE_STATE_QUANT": "q8", "HIPFIRE_DAEMON_BIN": self.daemon, "HIPFIRE_GRAPH": "1"},
        )
        self.assertEqual(kids[2].extra_env["HIPFIRE_STATE_QUANT"], "fp32")
        self.assertEqual(kids[2].argv[kids[2].argv.index("--spec") + 1], "off")
        self.assertTrue(all(c.sample_memory for c in kids))
        self.assertEqual([r for r, _ in k.inputs], ["model", "prompt"])
        self.assertEqual(cfg.graphs, ("1",))
        self.assertEqual(cfg.runs, 1)
        self.assertEqual(sorted(cfg.binaries), ["daemon", "hipfire"])
        self.assertEqual(collect.graph_axis("greedy_run"), "HIPFIRE_GRAPH")
        self.assertEqual([r for r, _ in collect.input_roles(cfg)], ["model", "model.ple-q4", "prompt"])

    def test_default_runs_graph_override_tokens_and_ab_ba(self):
        args = collect.make_parser().parse_args(self.greedy_argv("--dry-run", "--run-id", "rid", drop=("--runs",)))
        self.assertEqual(collect.validate(args, {}).runs, 1)
        cfg = self.config("--graph", "0,1", "--greedy-tokens", "128", runs="2")
        kids = collect.plan_children(cfg)
        self.assertEqual(len(kids), 2 * 2 * 2 * 2)
        self.assertEqual({k.argv[k.argv.index("--max-tokens") + 1] for k in kids}, {"128"})
        g0 = [k for k in kids if k.graph == "0"]
        self.assertEqual(g0[0].extra_env["HIPFIRE_GRAPH"], "0")
        self.assertEqual([k.mode for k in kids if k.run == 0 and k.graph == "1" and k.label == "greedy-spec-off"], ["q8", "fp32"])
        self.assertEqual([k.mode for k in kids if k.run == 1 and k.graph == "1" and k.label == "greedy-spec-off"], ["fp32", "q8"])
        unset = collect.plan_children(self.config("--graph", "unset"))
        self.assertNotIn("HIPFIRE_GRAPH", unset[0].extra_env)

    def test_only_selects_one_spec(self):
        kids = collect.plan_children(self.config("--only", "greedy-spec-mtp"))
        self.assertEqual([k.id for k in kids], ["run0/g1/q8/greedy-spec-mtp", "run0/g1/fp32/greedy-spec-mtp"])
        with self.assertRaisesRegex(collect.CollectError, "greedy-spec-dflash"):
            self.config("--only", "greedy-spec-dflash")

    def test_required_flag_and_prompt_refusals(self):
        with self.assertRaisesRegex(collect.CollectError, "requires --prompt-file"):
            collect.run(self.greedy_argv("--dry-run", drop=("--prompt-file",)), {})
        with self.assertRaisesRegex(collect.CollectError, "--max-seq"):
            collect.run(self.greedy_argv("--dry-run", drop=("--max-seq",)), {})
        for bad in ("0", "4097", "-1"):
            with self.assertRaisesRegex(collect.CollectError, "greedy-tokens"):
                self.config("--greedy-tokens", bad)
        for stage in ("quality", "perf", "memory", "smoke"):
            with self.assertRaisesRegex(collect.CollectError, "greedy-tokens applies to --stage greedy only"):
                self.fx.config(stage, "--greedy-tokens", "8", "--graph", "0")
        self.fx.prompt.write_text("\n\n")
        with self.assertRaisesRegex(collect.CollectError, "empty"):
            self.config()
        self.fx.prompt.write_text("--looks like a flag")
        with self.assertRaisesRegex(collect.CollectError, "flag"):
            self.config()
        self.fx.prompt.write_bytes(b"\xff\xfe not utf8")
        with self.assertRaisesRegex(collect.CollectError, "UTF-8"):
            self.config()
        self.fx.prompt.write_text("x" * collect.MAX_ARG_STRLEN)
        with self.assertRaisesRegex(collect.CollectError, "MAX_ARG_STRLEN"):
            self.config()
        self.fx.prompt.write_text("x" * (collect.MAX_ARG_STRLEN - 1))
        self.config()

    def test_missing_binary_refused_and_hold_required(self):
        for omitted in ("hipfire", "daemon"):
            with tempfile.TemporaryDirectory() as t:
                fx = Fixture(Path(t).resolve(), omit=(omitted,))
                argv = fx.argv("greedy", "--dry-run")
                with self.assertRaisesRegex(collect.CollectError, omitted):
                    collect.run(argv, {})
        with self.assertRaisesRegex(collect.CollectError, "no fabric hold"):
            collect.run(self.greedy_argv(), {})

    def test_dry_run_lists_four_children(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = collect.run(self.greedy_argv("--dry-run", "--run-id", "rid"), {})
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("4 child process(es)", text)
        self.assertEqual(text.count("\n[run0/g1/"), 4)
        self.assertEqual(text.count("--temp 0 --spec off"), 2)
        self.assertEqual(text.count("--temp 0 --spec mtp"), 2)
        self.assertIn("# greedy-tokens: 64", text)
        self.assertIn("HIPFIRE_GRAPH=1", text)
        self.assertFalse(self.fx.out.exists())

    def test_observation_parses_stdout_json(self):
        cfg = self.config()
        child = collect.plan_children(cfg)[1]
        cdir = self.root / "child"
        cdir.mkdir()
        text = "h\u00e9llo wor\u2603ld"
        line = json.dumps({"content": text, "tokens": 7, "tok_s": 31.5, "finish_reason": "stop"})
        (cdir / "stdout.txt").write_text(line + "\n")
        (cdir / "stderr.txt").write_text("daemon: max_seq 4096 card_cap 900\nunrelated\nadmission refused for x\n")
        (cdir / "memory_samples.jsonl").write_text(
            json.dumps({"mem_available_kib": 1000}) + "\n" + json.dumps({"mem_available_kib": 800}) + "\n"
        )
        outcome, obs = collect.observe_child(child, cdir, 0, False, "1")
        self.assertEqual(outcome, "ok")
        self.assertEqual(obs["content"], text)
        self.assertEqual(obs["content_sha256"], sha256_bytes(text.encode("utf-8")))
        self.assertEqual((obs["tokens"], obs["tok_s"], obs["finish_reason"]), (7, 31.5, "stop"))
        self.assertEqual(obs["daemon_log_lines"], ["daemon: max_seq 4096 card_cap 900", "admission refused for x"])
        self.assertEqual(obs["refusal_lines"], ["admission refused for x"])
        self.assertEqual(obs["memory"]["host_peak_drop_kib"], 200)
        # non-zero rc, empty content, missing JSON, or a timeout are never ok
        self.assertEqual(collect.observe_child(child, cdir, 1, False, "1")[0], "failed")
        self.assertEqual(collect.observe_child(child, cdir, 0, True, "1")[0], "timeout")
        (cdir / "stdout.txt").write_text(json.dumps({"content": "", "tokens": 0}) + "\n")
        self.assertEqual(collect.observe_child(child, cdir, 0, False, "1")[0], "failed")
        (cdir / "stdout.txt").write_text("not json\n")
        outcome, obs = collect.observe_child(child, cdir, 0, False, "1")
        self.assertEqual(outcome, "failed")
        self.assertIn("no JSON", obs["error"])
        (cdir / "stdout.txt").write_text(json.dumps({"content": None, "tokens": None}) + "\n")
        self.assertEqual(collect.observe_child(child, cdir, 0, False, "1")[0], "failed")

    def test_analyze_equal(self):
        kids = []
        for mode in ("q8", "fp32"):
            kids += [greedy_child(0, mode, "off", "same text", tokens=4), greedy_child(0, mode, "mtp", "same text", tokens=4)]
        a = collect.analyze_greedy(kids)
        self.assertEqual(a["errors"], [])
        self.assertIs(a["fp32_mtp_equals_ar"], True)
        self.assertIs(a["q8_mtp_equals_ar"], True)
        self.assertEqual(a["labels"]["fp32_mtp_equals_ar"], "GATE (Track A F32 tripwire)")
        self.assertEqual(a["labels"]["q8_mtp_equals_ar"], "DIAGNOSTIC")
        self.assertIs(a["all_loaded"], True)
        p = [x for x in a["pairs"] if x["mode"] == "fp32"][0]
        self.assertEqual(
            (p["equal"], p["first_diff_char"], p["len_off"], p["len_mtp"], p["tokens_off"], p["tokens_mtp"]),
            (True, None, 9, 9, 4, 4),
        )
        self.assertEqual([x["equal"] for x in a["cross_mode_q8_vs_fp32"]], [True, True])
        self.assertEqual(a["labels"]["cross_mode_q8_vs_fp32"], "DIAGNOSTIC")
        # per-child admission lines keep only max_seq / card_cap / admission
        self.assertEqual(a["children"][0]["daemon_admission_lines"], ["daemon: max_seq 4096 card_cap 1000"])
        self.assertEqual(collect.stage_analysis("greedy", Path("/"), kids), a)

    def test_analyze_first_diff_and_cross_mode(self):
        kids = [
            greedy_child(0, "fp32", "off", "abcdef", tokens=3), greedy_child(0, "fp32", "mtp", "abcdef", tokens=3),
            greedy_child(0, "q8", "off", "abcdef", tokens=3), greedy_child(0, "q8", "mtp", "abcdXf!", tokens=4),
        ]
        a = collect.analyze_greedy(kids)
        self.assertIs(a["fp32_mtp_equals_ar"], True)
        self.assertIs(a["q8_mtp_equals_ar"], False)
        p = [x for x in a["pairs"] if x["mode"] == "q8"][0]
        self.assertEqual((p["equal"], p["first_diff_char"], p["len_off"], p["len_mtp"]), (False, 4, 6, 7))
        cross = {x["spec"]: x for x in a["cross_mode_q8_vs_fp32"]}
        self.assertTrue(cross["off"]["equal"] and cross["off"]["first_diff_char"] is None)
        self.assertFalse(cross["mtp"]["equal"])
        self.assertEqual(cross["mtp"]["first_diff_char"], 4)
        self.assertEqual((cross["mtp"]["len_q8"], cross["mtp"]["len_fp32"]), (7, 6))
        # strict prefix: first difference is the shorter length
        b = collect.analyze_greedy([greedy_child(0, "fp32", "off", "abcd"), greedy_child(0, "fp32", "mtp", "abcdef")])
        self.assertEqual(b["pairs"][0]["first_diff_char"], 4)
        self.assertIs(b["fp32_mtp_equals_ar"], False)
        self.assertIsNone(b["q8_mtp_equals_ar"])

    def test_analyze_missing_pairs_are_errors_never_passes(self):
        a = collect.analyze_greedy([greedy_child(0, "fp32", "off", "x", tokens=1)])
        self.assertIsNone(a["fp32_mtp_equals_ar"])
        self.assertEqual(a["pairs"], [])
        self.assertTrue(any("run0/g1/fp32" in e and "missing" in e for e in a["errors"]))
        self.assertEqual(a["per_mode"]["fp32"]["pairs_missing"], 1)
        # a failed child leaves its pair uncompared and all_loaded false
        kids = [
            greedy_child(0, "fp32", "off", "x", tokens=1), greedy_child(0, "fp32", "mtp", None, outcome="failed"),
            greedy_child(1, "fp32", "off", "x", tokens=1), greedy_child(1, "fp32", "mtp", "x", tokens=1),
        ]
        b = collect.analyze_greedy(kids)
        self.assertIsNone(b["fp32_mtp_equals_ar"])  # one pair equal, one missing: not a pass
        self.assertIs(b["all_loaded"], False)
        self.assertEqual(len(b["pairs"]), 1)
        self.assertTrue(any("greedy-spec-mtp" in e for e in b["errors"]))
        # a definite mismatch stays False even when another pair is missing
        c = collect.analyze_greedy(kids + [greedy_child(2, "fp32", "off", "a"), greedy_child(2, "fp32", "mtp", "b")])
        self.assertIs(c["fp32_mtp_equals_ar"], False)
        d = collect.analyze_greedy([])
        self.assertIsNone(d["fp32_mtp_equals_ar"])
        self.assertIsNone(d["q8_mtp_equals_ar"])
        self.assertIs(d["all_loaded"], False)
        self.assertTrue(d["errors"])
        # cross-mode diagnostic with a missing arm is reported, not compared
        e = collect.analyze_greedy([greedy_child(0, "q8", "off", "x"), greedy_child(0, "fp32", "off", None, outcome="failed")])
        self.assertEqual(e["cross_mode_q8_vs_fp32"], [])
        self.assertTrue(any("q8-vs-fp32" in x for x in e["errors"]))

    def test_end_to_end_with_fake_hipfire_and_analyze_cli(self):
        (self.root / "e2e").mkdir()
        fx = Fixture(self.root / "e2e", fakes={"hipfire": FAKE_HIPFIRE_RUN})
        fx.prompt.write_text("the prompt\n")
        argv = fx.argv("greedy", "--dry-run", "--run-id", "rid", runs="1")
        for flag in ("--wt2-ref", "--code-ref", "--long-prompt-file"):
            i = argv.index(flag)
            del argv[i:i + 2]
        argv[argv.index("--max-seq") + 1] = "262144"
        cfg = collect.validate(collect.make_parser().parse_args(argv), {})
        cfg.dry_run = False
        with contextlib.redirect_stderr(io.StringIO()):
            rc = collect.run_stage(cfg, {"PATH": os.environ.get("PATH", ""), "FABRIC_JOB": "1:2:halo"})
        self.assertEqual(rc, 0)
        receipt = json.loads((cfg.run_dir / "receipt.json").read_text())
        self.assertEqual(len(receipt["children"]), 4)
        self.assertEqual(receipt["greedy_tokens"], 64)
        self.assertIsNone(receipt["max_chunks"])
        a = receipt["analysis"]
        self.assertIs(a["fp32_mtp_equals_ar"], True)
        self.assertIs(a["q8_mtp_equals_ar"], False)
        self.assertIs(a["all_loaded"], True)
        self.assertEqual(a["errors"], [])
        bad = [p for p in a["pairs"] if p["mode"] == "q8"][0]
        self.assertEqual(bad["first_diff_char"], 16)
        self.assertTrue(all("max_seq 262144" in c["daemon_admission_lines"][0] for c in a["children"]))
        child_dir = Path(cfg.run_dir) / receipt["children"][1]["dir"]
        env = json.loads((child_dir / "env.json").read_text())
        self.assertEqual(env["HIPFIRE_GRAPH"], "1")
        self.assertEqual(env["HIPFIRE_STATE_QUANT"], "q8")
        self.assertIn("--spec mtp", (child_dir / "command.txt").read_text())
        # thinking off through the child's isolated global config (no env knob exists)
        self.assertEqual((child_dir / "home" / ".hipfire" / "config.toml").read_text(), '[reasoning]\nmode = "off"\n')
        self.assertEqual(json.loads((child_dir / "identity.json").read_text())["child"]["home_files"],
                         {"config.toml": '[reasoning]\nmode = "off"\n'})
        report = (cfg.run_dir / "report.md").read_text()
        self.assertIn("GATE (Track A F32 tripwire)", report)
        self.assertIn("DIAGNOSTIC", report)
        # --analyze recomputes the same analysis from the run dir
        (cfg.run_dir / "analysis.json").unlink()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(collect.run(["--stage", "greedy", "--analyze", "--out", str(fx.out), "--run-id", "rid"], {}), 0)
        self.assertEqual(json.loads(buf.getvalue()), a)
        self.assertTrue((cfg.run_dir / "analysis.json").is_file())


if __name__ == "__main__":
    unittest.main()
