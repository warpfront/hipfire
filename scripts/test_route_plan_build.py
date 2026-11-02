#!/usr/bin/env -S python3 -I
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU tests of route_plan_build.py over SYNTHETIC walk directories.

Nothing here is a recorded walk and nothing written is a route plan to check in:
every fixture (HFQM headers, audit logs, result/command receipts) is generated into
a temporary directory to pin the builder's parsing, cross-checks and sealing rules.

run: python3 scripts/test_route_plan_build.py
"""
import contextlib
import importlib.util
import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("route_plan_build", pathlib.Path(__file__).with_name("route_plan_build.py"))
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)

TRUNK = [("model.language_model.embed_tokens.weight", 3, (1024, 256)),
         ("model.language_model.layers.0.mlp.gate_proj.weight", 44, (512, 256))]
HEAD = [("eh_proj", 13, (256, 512))]
AXES = ("arch=gfx1100 model_arch=5 spec={spec} heads=16,2,256 max_seq=8192 kv_k=q8 kv_v=q8 state_quant=q8 "
        "kv_backend=vmm prefill_chunk_rows=512 num_experts=0 experts_per_tok=0 expert_placement=resident")
HIP_SHA = "a" * 64
IMG_SHA = "b" * 64


def write_hfqm(path, arch, tensors):
    meta = b'{"config":{}}'
    body = struct.pack("<I", len(tensors))
    for name, quant, shape in tensors:
        raw = name.encode()
        body += struct.pack("<H", len(raw)) + raw + struct.pack("<BB", quant, len(shape))
        body += struct.pack("<" + "I" * len(shape), *shape) + b"\0" * 12
    data = 32 + len(meta) + len(body)
    path.write_bytes(struct.pack("<4sIIIQQ", b"HFQM", 1, arch, len(tensors), 32, data) + meta + body)


def audit(arch, hip_symbols, emb_symbols):
    lines = []
    for i, (module, symbol) in enumerate(hip_symbols):
        rec = {"arch": arch, "module": module, "source_sha256": HIP_SHA, "flags": ["-O3"], "scheduler_profile": None}
        lines += [f"HIPFIRE_MODULE_LOAD_AUDIT\thip\t{json.dumps(rec)}", f"HIPFIRE_MODULE_LOAD_AUDIT\tload\th{i}",
                  f"HIPFIRE_MODULE_LOAD_AUDIT\tget_function\th{i}\t{symbol}"]
    for i, (module, symbol) in enumerate(emb_symbols, 100):
        rec = {"arch": arch, "module": module, "image_sha256": IMG_SHA}
        lines += [f"HIPFIRE_MODULE_LOAD_AUDIT\tembedded\t{json.dumps(rec)}", f"HIPFIRE_MODULE_LOAD_AUDIT\tload_data\th{i}",
                  f"HIPFIRE_MODULE_LOAD_AUDIT\tget_function\th{i}\t{symbol}"]
    return lines


def make_walk(root, name, route, model, stderr_lines, env=None, ok=True, max_seq=8192):
    walk = root / name
    walk.mkdir()
    (walk / "result.json").write_text(json.dumps({"rc": 0 if ok else 1, "ok": ok, "errors": 0, "results": []}))
    (walk / "command.json").write_text(json.dumps({
        "daemon": "d", "daemon_sha256": "ab" * 32, "route": "mtp" if route == "native_mtp" else "ar",
        "env": env or {}, "load": {"model": str(model), "params": {"max_seq": max_seq}}, "requests": 3}))
    (walk / "stderr.log").write_text("\n".join(stderr_lines) + "\n")
    return walk


def run_build(out, *argv):
    stdout = io.StringIO()
    sys.argv = ["route_plan_build.py", "--out", str(out), "--family", "synthetic", *argv]
    with contextlib.redirect_stdout(stdout):
        build.main()
    return out.read_text()


class Qwen35Builder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.model = self.root / "model.hfq"
        self.head = self.root / "model.mtp"
        write_hfqm(self.model, 5, TRUNK)
        write_hfqm(self.head, 5, HEAD)
        self.trunk_layout = build.layout_sha256(build.header_tensors(self.model)[1])
        self.mtp_layout = build.layout_sha256([*build.header_tensors(self.model)[1], *build.header_tensors(self.head)[1]])
        axes = lambda spec: f"  route plan axes: {AXES.format(spec=spec)}"
        self.ar = make_walk(self.root, "ar", "ar", self.model, [axes("Ar"), *audit("gfx1100", [("add", "add_f32")], [])])
        self.mtp = make_walk(self.root, "mtp", "native_mtp", self.model,
                             [axes("NativeMtp"), *audit("gfx1100", [("add", "add_f32"), ("mtp_only", "mtp_k")], [("pm_img", "pm_sym")])])

    def build(self, *extra, walks=None):
        walks = walks or [f"ar={self.ar}", f"native_mtp={self.mtp}"]
        argv = ["--model", str(self.model), *(x for w in walks for x in ("--walk", w)), *extra]
        return run_build(self.root / "plan.tsv", *argv)

    def test_generic_plan_records_axes_heads_layouts_and_route_rows(self):
        text = self.build("--mtp-head", str(self.head))
        lines = text.splitlines()
        self.assertIn(f"#! layout {self.trunk_layout}", lines)
        self.assertIn(f"#! mtp_layout {self.mtp_layout}", lines)
        self.assertNotEqual(self.trunk_layout, self.mtp_layout)
        self.assertIn("#! model_arch 5", lines)
        self.assertIn("#! heads 16,2,256", lines)
        self.assertIn("#! max_seq 8192", lines)
        for axis in ("kv_backend vmm", "kv_k q8", "kv_v q8", "prefill_chunk_rows 512", "state_quant q8",
                     "num_experts 0", "experts_per_tok 0", "expert_placement resident"):
            self.assertIn(f"#! axis {axis}", lines)
        self.assertFalse([l for l in lines if l.startswith("#! sealed")], "a plan is written open")
        self.assertIn(f"ar\thip\tadd\tadd_f32\t{HIP_SHA}", lines)
        self.assertNotIn(f"ar\thip\tmtp_only\tmtp_k\t{HIP_SHA}", lines)
        self.assertIn(f"native_mtp\thip\tmtp_only\tmtp_k\t{HIP_SHA}", lines)
        self.assertIn(f"native_mtp\thip\tadd\tadd_f32\t{HIP_SHA}", lines)
        self.assertIn(f"native_mtp\tembedded\tpm_img\tpm_sym\t{IMG_SHA}", lines)

    def test_moe_model_arch_6_records_its_expert_axes(self):
        moe = self.root / "moe.hfq"
        write_hfqm(moe, 6, TRUNK)
        moe_axes = (AXES.replace("model_arch=5", "model_arch=6").replace("num_experts=0", "num_experts=256")
                    .replace("experts_per_tok=0", "experts_per_tok=8"))
        walk = make_walk(self.root, "moe", "ar", moe, [f"  route plan axes: {moe_axes.format(spec='Ar')}",
                                                       *audit("gfx1100", [("add", "add_f32")], [])])
        text = run_build(self.root / "moe.tsv", "--model", str(moe), "--walk", f"ar={walk}")
        lines = text.splitlines()
        self.assertIn("#! model_arch 6", lines)
        for axis in ("num_experts 256", "experts_per_tok 8", "expert_placement resident"):
            self.assertIn(f"#! axis {axis}", lines)
        # A walk of the wrong model arch is refused.
        with self.assertRaisesRegex(SystemExit, "plan model is 6"):
            run_build(self.root / "x.tsv", "--model", str(moe), "--walk", f"ar={self.ar}", "--no-verify-walk-model")


    def test_bundled_trailer_head_offset(self):
        bundled = self.root / "bundled.hfq"
        bundled.write_bytes(self.model.read_bytes() + self.head.read_bytes())
        offset = len(self.model.read_bytes())
        text = self.build("--mtp-head", f"{bundled}@{offset}")
        self.assertIn(f"#! mtp_layout {self.mtp_layout}", text.splitlines())

    def test_native_mtp_walk_needs_the_head_header(self):
        with self.assertRaises(SystemExit):
            self.build()

    def test_missing_axes_line_needs_a_fresh_binary(self):
        stale = make_walk(self.root, "stale", "ar", self.model, audit("gfx1100", [("add", "add_f32")], []))
        with self.assertRaisesRegex(SystemExit, "no qwen4 state line or route plan axes line"):
            self.build(walks=[f"ar={stale}"])

    def test_walk_whose_resolved_spec_is_not_its_route_is_refused(self):
        # A native_mtp walk that loaded no head resolved the AR route.
        no_head = make_walk(self.root, "nohead", "native_mtp", self.model,
                            [f"  route plan axes: {AXES.format(spec='Ar')}", *audit("gfx1100", [("add", "add_f32")], [])])
        with self.assertRaisesRegex(SystemExit, "resolved the Ar route"):
            self.build("--mtp-head", str(self.head), walks=[f"ar={self.ar}", f"native_mtp={no_head}"])

    def test_walk_of_another_model_header_is_refused(self):
        other = self.root / "other.hfq"
        write_hfqm(other, 5, TRUNK[:1])
        wrong = make_walk(self.root, "wrong", "ar", other, [f"  route plan axes: {AXES.format(spec='Ar')}",
                                                            *audit("gfx1100", [("add", "add_f32")], [])])
        with self.assertRaisesRegex(SystemExit, "is not the plan layout"):
            self.build(walks=[f"ar={wrong}"])

    def test_axis_and_heads_disagreement_between_walks_is_refused(self):
        other_heads = AXES.format(spec="Ar").replace("heads=16,2,256", "heads=24,4,256")
        other = make_walk(self.root, "heads", "ar", self.model,
                          [f"  route plan axes: {other_heads}", *audit("gfx1100", [("add", "add_f32")], [])])
        with self.assertRaisesRegex(SystemExit, "disagree on heads"):
            self.build(walks=[f"ar={self.ar}", f"ar={other}"])
        # A different axis value is a second walked value, recorded on the axis line.
        legacy = AXES.format(spec="Ar").replace("kv_backend=vmm", "kv_backend=legacy")
        second = make_walk(self.root, "legacy", "ar", self.model,
                           [f"  route plan axes: {legacy}", *audit("gfx1100", [("add", "add_f32")], [])])
        self.assertIn("#! axis kv_backend legacy vmm", self.build(walks=[f"ar={self.ar}", f"ar={second}"]).splitlines())

    def test_two_resolved_routes_in_one_load_log_are_refused(self):
        twice = make_walk(self.root, "twice", "ar", self.model,
                          [f"  route plan axes: {AXES.format(spec='Ar')}",
                           f"  route plan axes: {AXES.format(spec='Ar').replace('q8', 'fp8')}",
                           *audit("gfx1100", [("add", "add_f32")], [])])
        with self.assertRaisesRegex(SystemExit, "more than one resolved route"):
            self.build(walks=[f"ar={twice}"])

    def replay(self, name, route, spec, modules, extra_lines=(), env=None, axes=None):
        lines = [f"  route plan axes: {axes or AXES.format(spec=spec)}",
                 f"  kernels: {spec} route plan ({modules} modules), compiling with up to 4 jobs during weight load",
                 f"  kernels: {spec} route preloaded and sealed", *extra_lines]
        return make_walk(self.root, name, route, self.model, lines,
                         env={"HIPFIRE_ROUTE_PLAN_SEAL": "1"} if env is None else env)

    def test_sealed_replay_seals_only_the_replayed_routes(self):
        ar = self.replay("rar", "ar", "Ar", 1)
        text = self.build("--mtp-head", str(self.head), "--sealed-replay", f"ar={ar}")
        self.assertIn("#! sealed ar", text.splitlines())
        mtp = self.replay("rmtp", "native_mtp", "NativeMtp", 3)
        text = self.build("--mtp-head", str(self.head), "--sealed-replay", f"ar={ar}", "--sealed-replay", f"native_mtp={mtp}")
        self.assertIn("#! sealed ar native_mtp", text.splitlines())
        self.assertEqual(sum(l.startswith("# replay ") for l in text.splitlines()), 2)

    def test_replays_that_do_not_prove_closure_are_refused(self):
        bad = {
            "unplanned kernel": self.replay("a", "ar", "Ar", 1, ["  kernels: WARNING unplanned kernel k (module m) loaded lazily"]),
            "predicted": self.replay("b", "ar", "Ar", 1, ["  kernels: ready, open (never sealed): 3/9 predicted x modules"]),
            "replay planned 2 modules": self.replay("c", "ar", "Ar", 2),
            "not a sealed replay": self.replay("d", "ar", "Ar", 1, env={}),
            "outside the walked plan": self.replay("e", "ar", "Ar", 1, axes=AXES.format(spec="Ar").replace("kv_k=q8", "kv_k=fp8")),
        }
        for why, replay in bad.items():
            with self.subTest(why), self.assertRaisesRegex(SystemExit, why):
                self.build("--mtp-head", str(self.head), "--sealed-replay", f"ar={replay}")
        unsealed = make_walk(self.root, "f", "ar", self.model,
                             [f"  route plan axes: {AXES.format(spec='Ar')}", "  kernels: Ar route plan (1 modules), x"],
                             env={"HIPFIRE_ROUTE_PLAN_SEAL": "1"})
        with self.assertRaisesRegex(SystemExit, "did not apply"):
            self.build("--mtp-head", str(self.head), "--sealed-replay", f"ar={unsealed}")
        failed = make_walk(self.root, "g", "ar", self.model, [], env={"HIPFIRE_ROUTE_PLAN_SEAL": "1"}, ok=False)
        with self.assertRaisesRegex(SystemExit, "not ok"):
            self.build("--mtp-head", str(self.head), "--sealed-replay", f"ar={failed}")


class FlashNextBuilder(unittest.TestCase):
    def test_flash_next_plan_keeps_its_qwen4_axes_and_gains_no_generic_directives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "fn.hfq"
            write_hfqm(model, 16, TRUNK)
            lines = ["  qwen4 state: QSA fp8 K/V (vmm context storage), GDN q8 recurrent",
                     *audit("gfx1201", [("add", "add_f32")], [])]
            walk = make_walk(root, "fn", "ar", model, lines, max_seq=131072)
            text = run_build(root / "plan.tsv", "--model", str(model), "--walk", f"ar={walk}")
            plan = text.splitlines()
            for axis in ("gdn q8", "kv_backend vmm", "qsa fp8", "qsa_f16_arena off", "qsa_pooled_bf16 off"):
                self.assertIn(f"#! axis {axis}", plan)
            self.assertFalse([l for l in plan if l.startswith(("#! heads", "#! mtp_layout", "#! sealed"))])
            self.assertIn("#! model_arch 16", plan)

    def test_walk_receipts_keep_route_env_and_drop_host_run_noise(self):
        env = {"HIPFIRE_PLE_FORMAT": "q4", "HIPFIRE_ROUTE_PLAN": "0", "HIPFIRE_FABRIC_ADMISSION": "/state/card.json#7",
               "HIPFIRE_TORCH_" + "PYTHON": "/env/bin/python", "HIPFIRE_HOME": "/run/home", "HIPFIRE_DEVICES": "GPU-x"}
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "fn.hfq"
            write_hfqm(model, 16, TRUNK)
            lines = ["  qwen4 state: QSA fp8 K/V (vmm context storage), GDN q8 recurrent",
                     *audit("gfx1201", [("add", "add_f32")], [])]
            walk = make_walk(root, "fn", "ar", model, lines, env=env, max_seq=131072)
            text = run_build(root / "plan.tsv", "--model", str(model), "--walk", f"ar={walk}")
            receipt = [l for l in text.splitlines() if l.startswith("# walk ar fn ")]
            self.assertEqual(len(receipt), 1)
            self.assertIn('env={"HIPFIRE_PLE_FORMAT": "q4", "HIPFIRE_ROUTE_PLAN": "0"}', receipt[0])


class Gfx1100FlashNext(unittest.TestCase):
    """The bounded XTX walk's shape: q8 QSA/GDN on the legacy KV backend, a walked max_seq of 262144, an AR walk and
    an MTP walk of the same layout; the plan stays open and its HIP rows become the gfx1100 trace rows."""

    STATE = "  qwen4 state: QSA q8 (Q8_0 blocks of 32) K/V (legacy context storage), GDN q8 recurrent"

    def test_partial_walk_is_refused_unless_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "fn.hfq"
            write_hfqm(model, 16, TRUNK)
            walk = make_walk(root, "ar", "ar", model, [self.STATE, *audit("gfx1100", [("add", "add_f32")], [])], max_seq=262144)
            (walk / "result.json").write_text(json.dumps({"rc": None, "ok": False, "partial": True, "reason": "walk timeout",
                                                          "errors": 0, "results": []}))
            with self.assertRaises(SystemExit):
                run_build(root / "p.tsv", "--model", str(model), "--walk", f"ar={walk}")
            text = run_build(root / "p.tsv", "--model", str(model), "--walk", f"ar={walk}", "--accept-partial")
            self.assertIn("ar\thip\tadd\tadd_f32\t" + HIP_SHA, text.splitlines())
            (walk / "result.json").write_text(json.dumps({"rc": None, "ok": False, "partial": True, "reason": "x",
                                                          "errors": 1, "results": []}))
            with self.assertRaises(SystemExit):
                run_build(root / "p.tsv", "--model", str(model), "--walk", f"ar={walk}", "--accept-partial")

    def test_plan_axes_open_seal_and_trace_rows(self):
        spec = importlib.util.spec_from_file_location("route_plan_trace_rows",
                                                      pathlib.Path(__file__).with_name("route_plan_trace_rows.py"))
        trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trace)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "fn.hfq"
            write_hfqm(model, 16, TRUNK)
            ar = make_walk(root, "ar", "ar", model,
                           [self.STATE, *audit("gfx1100", [("add", "add_f32"), ("tensor_ops", "a")], [("pm_img", "pm_w4")])],
                           max_seq=262144)
            mtp = make_walk(root, "mtp", "native_mtp", model,
                            [self.STATE, *audit("gfx1100", [("add", "add_f32"), ("tensor_ops", "b")], [])],
                            max_seq=262144)
            text = run_build(root / "qwen3.8-flash-next.gfx1100.tsv", "--model", str(model), "--walk", f"ar={ar}",
                             "--walk", f"native_mtp={mtp}")
            plan = text.splitlines()
            for line in ("#! arch gfx1100", "#! max_seq 262144", "#! axis qsa q8", "#! axis kv_backend legacy",
                         "#! axis gdn q8", "#! axis qsa_f16_arena off", "#! axis qsa_pooled_bf16 off"):
                self.assertIn(line, plan)
            self.assertFalse([l for l in plan if l.startswith("#! sealed")], "a bounded walk never seals")
            # native_mtp includes every ar row; the MTP-only symbol joins its module's row.
            self.assertIn(f"ar\thip\ttensor_ops\ta\t{HIP_SHA}", plan)
            self.assertIn(f"native_mtp\thip\ttensor_ops\ta,b\t{HIP_SHA}", plan)
            self.assertIn(f"native_mtp\tembedded\tpm_img\tpm_w4\t{IMG_SHA}", plan)
            arch, rows = trace.trace_rows(text)
            self.assertEqual(arch, "gfx1100")
            self.assertEqual(rows, [("add", ["add_f32"]), ("tensor_ops", ["a", "b"])])
            rendered = trace.render("qwen3.8-flash-next.gfx1100.tsv", text).splitlines()
            self.assertEqual(rendered[-2:], ["gfx1100\tadd\tadd_f32", "gfx1100\ttensor_ops\ta,b"])
            self.assertTrue(all(l.startswith("#") or l.count("\t") == 2 for l in rendered))


if __name__ == "__main__":
    unittest.main()
