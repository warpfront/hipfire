#!/usr/bin/env python3
"""CPU tests of quality_campaign.py: manifest / plan / env / identity / lifecycle-argument invariants,
route evidence and determinism parsing. No model, no GPU, no native executable is run.

Run: python3 -B -m unittest discover -s scripts/slowlink-gap -p 'quality_test_*.py'
"""
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import quality_campaign as qc  # noqa: E402
import quality_scorer as qs  # noqa: E402

PLAN_SHA = "74ce7b8291b15bc25254b589844ccd0c2f6b342dfb103e725736c8232b33bcf6"
LEVER_KEYS = ("HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS", "HIPFIRE_QWEN4_MOE_SYM_IU4", "HIPFIRE_FN_QSA_GATHER_GFX1100")
QSA_OFF_KEY = "HIPFIRE_QWEN4_QSA_WMMA_GATHER"


def base_plan():
    return {
        "schema": qc.PLAN_SCHEMA, "max_seq": 16897,
        "artifact": {"path": "/m/model.mq4", "size": 10, "sha256": "8" * 64},
        "expect": {"resident_layers": 9, "baseline_tile_rows": 4096, "s1_tile_rows": 768,
                   "s1_layer_major_min_rows": 1537, "ple_format": "q4", "env": dict(qc.PINNED_ENV),
                   "env_absent": [qc.EXPLICIT_LAYERS_ENV],
                   "baseline_env": {qc.EXPLICIT_LAYERS_ENV: "9", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"}},
        "determinism": {"case_ids": ["p512", "p1536", "p1537", "p8192", "irregular", "resume", "natural"]},
        "primary": {"contexts": [{"id": "ctx16k"}, {"id": "ctx4k"}]},
    }


def fake_camp(root="/r"):
    camp = {"schema": qc.CAMPAIGN_SCHEMA, "label": "t", "root": root,
            "harness": qc.harness_hashes(qc.HERE),
            "source_pin": {"commit": "a" * 40, "example_sha256": "b" * 64},
            "exe": {"path": "/x/bin/qwen4_slowlink_quality", "sha256": "c" * 64, "size": 5},
            "pack": {"digest": "d" * 64}, "config": {"path": "/c.toml", "sha256": "e" * 64},
            "model": {"path": "/m/model.mq4", "dev": 1, "ino": 2, "size": 10, "mtime_ns": 1790928378750535607,
                      "sha256": "8" * 64},
            "plan": {"path": "/p", "sha256": PLAN_SHA}, "arms": {}}
    for arm in qc.ARMS:
        runs = {ph: {"run_dir": "%s/runs/%s-%s" % (root, arm, ph), "out_dir": "%s/phases/%s-%s" % (root, arm, ph),
                     "argv": qc.native_argv("/x/bin/qwen4_slowlink_quality", ph, "/pl/%s.json" % arm,
                                            "/mf/%s-%s.json" % (arm, ph), "/m/model.mq4",
                                            "%s/phases/%s-%s" % (root, arm, ph)),
                     "manifest": {"path": "/mf/%s-%s.json" % (arm, ph), "sha256": None}} for ph in qc.PHASES}
        camp["arms"][arm] = {"levers_on": list(qc.ARMS[arm]), "lever_env": qc.lever_env(arm),
                             "plan": {"path": "/pl/%s.json" % arm, "sha256": "f" * 64}, "runs": runs}
    return camp


def fake_recipe(**over):
    r = {"schema": qc.RECIPE_SCHEMA,
         "device": {"arch": "gfx1201", "card": "e", "bdf": "0000:c3:00.0", "vendor_id": "0x1002",
                    "device_id": "0x744c", "uuid": "GPU-05f92432f2312a0e", "unique_id": "05f92432f2312a0e",
                    "hip_visible": "0"},
         "source": {"origin": "fn-accepted-fa11-gfx1201-pack-20261021-001", "pack_dir": "/x/kernels/compiled/gfx1201",
                    "pack_digest": "1" * 64, "pack_count": 3,
                    "elf": {"path": "/x/bin/qwen4_slowlink_quality", "sha256": "c" * 64, "size": 5},
                    "reference_plan_sha256": PLAN_SHA,
                    "reference_source_pin": {"commit": "a" * 40, "example_sha256": "b" * 64}},
         "verified_cache": {"home": "/real/.hipfire", "path": "/real/.hipfire/cache/verified-artifacts.json",
                            "sha256": "2" * 64, "entries": []},
         "expect": {"resident_layers": 9, "resident_layer_set": None, "placement": "slowlink-profile"},
         "resources": {"max_job_secs": 1800, "child_cap_secs": 1700, "kill_grace_secs": 20,
                       "drain_vram_bytes": 2 * 1024 ** 3, "min_mem_available_kib": 80 * 1024 * 1024,
                       "native_cpu_cap_secs": 1200}}
    for k, v in over.items():
        r[k] = v
    return r


class ExecutionRecipe(unittest.TestCase):
    def test_no_recipe_preserves_original_defaults(self):
        camp = fake_camp()
        self.assertEqual(qc.camp_device(camp), qc.DEVICE)
        self.assertEqual(qc.camp_arch(camp), "gfx1100")
        self.assertEqual(qc.pack_dir_of("/x/bin/q"), "/x/bin/kernels/compiled/gfx1100")
        m = qc.native_manifest(camp, "off", "s1a")
        self.assertNotIn("execution_recipe", m["source"])
        self.assertEqual(m["launch"]["device"], qc.DEVICE)

    def test_recipe_schema_device_source_cache_expect_resources_validated(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fake_recipe(), f)
        try:
            r, sha = qc.load_execution_recipe(f.name)
            self.assertEqual(r["schema"], qc.RECIPE_SCHEMA)
            self.assertEqual(len(sha), 64)
        finally:
            os.unlink(f.name)
        for bad in ({"schema": "other"},
                    {"device": {"arch": "gfx900", "card": "e", "bdf": "b", "vendor_id": "v",
                                "device_id": "d", "uuid": "u", "unique_id": "i", "hip_visible": "h"}},
                    {"expect": {"resident_layers": 0, "placement": "slowlink-profile"}},
                    {"resources": {"max_job_secs": 0, "child_cap_secs": 1, "kill_grace_secs": 1,
                                  "drain_vram_bytes": 1, "min_mem_available_kib": 1,
                                  "native_cpu_cap_secs": 1}},
                    {"verified_cache": {"home": "rel", "path": "rel", "sha256": "2" * 64}}):
            r = fake_recipe(**bad)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(r, f)
            try:
                with self.assertRaises(qc.Refusal):
                    qc.load_execution_recipe(f.name)
            finally:
                os.unlink(f.name)

    def test_candidate_plan_whitelist_is_execution_binding_only(self):
        base = base_plan()
        cand = copy.deepcopy(base)
        self.assertTrue(qc.plan_structural_equal_except_artifact_path(base, cand))
        moved = copy.deepcopy(base)
        moved["artifact"]["path"] = "/staged/model.mq4"
        self.assertTrue(qc.plan_structural_equal_except_artifact_path(base, moved))
        tampered = copy.deepcopy(base)
        tampered["expect"]["resident_layers"] = 12
        # no recipe: ANY count delta is refused; a changed count without a binding recipe never passes
        self.assertFalse(qc.plan_structural_equal_except_artifact_path(base, tampered))
        corpus = copy.deepcopy(base)
        corpus["primary"]["contexts"][0]["id"] = "other"
        self.assertFalse(qc.plan_structural_equal_except_artifact_path(base, corpus))
        # with recipe: ONLY artifact.path plus the recipe-bound expect.resident_layers passes.
        # The expected set lives ONLY in recipe.expect.resident_layer_set (strict Expect has no
        # set field; no new plan field introduced, native deny_unknown_fields untouched).
        recipe = fake_recipe(expect={"resident_layers": 12, "resident_layer_set": list(range(12)),
                                     "placement": "slowlink-profile"})
        bound = copy.deepcopy(base)
        bound["artifact"]["path"] = "/staged/model.mq4"
        bound["expect"]["resident_layers"] = 12
        self.assertTrue(qc.plan_structural_equal_except_artifact_path(base, bound, recipe))
        wrong_count = copy.deepcopy(bound)
        wrong_count["expect"]["resident_layers"] = 13
        self.assertFalse(qc.plan_structural_equal_except_artifact_path(base, wrong_count, recipe))
        other_drift = copy.deepcopy(bound)
        other_drift["expect"]["s1_tile_rows"] = 1536
        self.assertFalse(qc.plan_structural_equal_except_artifact_path(base, other_drift, recipe))
        set_in_plan = copy.deepcopy(bound)
        set_in_plan["expect"]["resident_layer_set"] = list(range(12))
        self.assertFalse(qc.plan_structural_equal_except_artifact_path(base, set_in_plan, recipe))

    def test_device_pack_cache_count_refused(self):
        camp = fake_camp()
        camp["execution_recipe"] = fake_recipe()
        camp["device"] = dict(camp["execution_recipe"]["device"])
        self.assertEqual(qc.camp_arch(camp), "gfx1201")
        self.assertEqual(qc.device_state(FakeHost({"/sys/bus/pci/devices/0000:c3:00.0/vendor": "0x1002\n",
                                                   "/sys/bus/pci/devices/0000:c3:00.0/device": "0x744c\n"}),
                                         camp["device"])[0], True)
        self.assertFalse(qc.device_state(FakeHost({}), camp["device"])[0])
        self.assertEqual(qc.pack_dir_of("/x/bin/q", "gfx1201"), "/x/bin/kernels/compiled/gfx1201")
        with self.assertRaises(qc.Refusal):
            qc.pack_rel_for("gfx900")
        m = qc.native_manifest(camp, "off", "s1a")
        self.assertEqual(m["source"]["execution_recipe"]["device"]["uuid"], "GPU-05f92432f2312a0e")
        self.assertEqual(m["launch"]["device"]["arch"], "gfx1201")
        self.assertEqual(m["launch"]["env"]["set"]["HIPFIRE_HOME"], "/real/.hipfire")
        self.assertEqual(m["launch"]["env"]["set"]["HOME"], m["launch"]["env"]["set"]["HOME"])
        self.assertTrue(m["launch"]["env"]["set"]["HOME"].endswith("/home"))
        self.assertEqual(m["launch"]["env"]["set"]["HIPFIRE_DEVICES"], "GPU-05f92432f2312a0e")

    def test_wrong_arch_pack_rejected_and_cache_verified(self):
        self.assertEqual(qc.pack_dir_of("/x/bin/q", "gfx1100"), "/x/bin/kernels/compiled/gfx1100")
        self.assertNotEqual(qc.pack_dir_of("/x/bin/q", "gfx1201"), qc.pack_dir_of("/x/bin/q", "gfx1100"))
        with tempfile.TemporaryDirectory() as d:
            member = os.path.join(d, "model.mq4")
            with open(member, "wb") as f:
                f.write(b"x" * 10)
            st = os.stat(member)
            vc = os.path.join(d, "verified-artifacts.json")
            with open(vc, "wb") as f:
                f.write(b"{}")
            r = fake_recipe(verified_cache={"home": d, "path": vc, "sha256": qc.sha256_bytes(b"{}"),
                                            "entries": [{"path": member, "sha256": "9" * 64, "dev": st.st_dev,
                                                         "ino": st.st_ino, "size": st.st_size,
                                                         "mtime_ns": st.st_mtime_ns}]})
            self.assertEqual(qc.verify_verified_cache(r), [])
            bad = copy.deepcopy(r)
            bad["verified_cache"]["sha256"] = "0" * 64
            self.assertTrue(qc.verify_verified_cache(bad))
            stale = copy.deepcopy(r)
            stale["verified_cache"]["entries"][0]["size"] += 1
            self.assertTrue(qc.verify_verified_cache(stale))

    def test_recipe_counts_parameterized_not_hardcoded(self):
        camp = fake_camp()
        camp["execution_recipe"] = fake_recipe(expect={"resident_layers": 12,
                                                       "resident_layer_set": list(range(12)),
                                                       "placement": "slowlink-profile"})
        camp["device"] = dict(camp["execution_recipe"]["device"])
        ev = qc.route_evidence("off", cells(), LOAD_FRONT.replace("resident layers=9", "resident layers=12"),
                               [], camp=camp)
        self.assertFalse(ev["ok"])  # helper cells still carry N9 receipts: genuine mismatch must fail
        p = base_plan()
        p["expect"]["resident_layers"] = 12
        ap = qc.arm_plan(p, "off", camp["execution_recipe"])
        self.assertEqual(ap["expect"]["resident_layers"], 12)
        with self.assertRaises(qc.Refusal):
            qc.arm_plan(p, "off")  # no-recipe replay keeps the original N9 gate

class HistoricalCampaign(unittest.TestCase):
    def test_source_binding_and_report_only_override(self):
        with tempfile.TemporaryDirectory() as root:
            camp = fake_camp(root)
            camp["harness"] = qc.harness_hashes(qc.HERE)
            data = json.dumps(camp).encode()
            qc.write_new(os.path.join(root, "campaign.json"), data)
            sha = qc.sha256_bytes(data)
            self.assertEqual(qc.load_campaign(root, sha), camp)
            with self.assertRaises(qc.Refusal):
                qc.load_campaign(root, "0" * 64)
            with self.assertRaises(qc.Refusal):
                qc.load_campaign(root, sha, recorded_harness_dir=qc.HERE)
            recorded = dict(camp["harness"], quality_campaign_sha256="2" * 64)
            with mock.patch.object(qc, "harness_hashes", return_value=recorded):
                with self.assertRaises(qc.Refusal):
                    qc.load_campaign(root, sha)
                with self.assertRaises(qc.Refusal):
                    qc.load_campaign(root, sha, recorded_harness_dir=qc.HERE, report_only=True)
            with mock.patch.object(qc, "harness_hashes", side_effect=lambda d: camp["harness"] if d == root else recorded):
                self.assertEqual(qc.load_campaign(root, sha, recorded_harness_dir=root, report_only=True), camp)
                with self.assertRaises(qc.Refusal):
                    qc.load_campaign(root, sha)

    def test_historical_options_are_only_accepted_by_report(self):
        parser = qc.build_parser()
        args = parser.parse_args(["report", "--root", "/r", "--expect-campaign-sha256", "a" * 64,
                                 "--recorded-harness-dir", "/old", "--out-dir", "/fresh"])
        self.assertEqual(args.recorded_harness_dir, "/old")
        for command, extra in (("launch", ["--arm", "off", "--phase", "s1a", "--max-secs", "1800", "--priority", "P1"]),
                               ("run-phase", ["--arm", "off", "--phase", "s1a"]),
                               ("score", ["--arm", "off"]), ("check-plans", [])):
            with self.subTest(command=command), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args([command, "--root", "/r", "--expect-campaign-sha256", "a" * 64]
                                  + extra + ["--recorded-harness-dir", "/old"])
        with self.assertRaises(qc.Refusal):
            qc.report(type("A", (), {"recorded_harness_dir": "/old", "out_dir": None})())



class LeverEnv(unittest.TestCase):
    def test_off_and_each_on_value(self):
        self.assertEqual(qc.lever_env("off"), {LEVER_KEYS[0]: "front", LEVER_KEYS[1]: "0", LEVER_KEYS[2]: "0",
                                               QSA_OFF_KEY: "0"})
        self.assertEqual(qc.lever_env("place")[LEVER_KEYS[0]], "0,3,7,15,19,23,27,39,43")
        self.assertEqual(qc.lever_env("place")[LEVER_KEYS[1]], "0")
        self.assertEqual(qc.lever_env("iu4"), {LEVER_KEYS[0]: "front", LEVER_KEYS[1]: "1", LEVER_KEYS[2]: "0",
                                               QSA_OFF_KEY: "0"})
        self.assertEqual(qc.lever_env("qsa"), {LEVER_KEYS[0]: "front", LEVER_KEYS[1]: "0", LEVER_KEYS[2]: "1",
                                               QSA_OFF_KEY: "0"})
        st = qc.lever_env("stack")
        self.assertEqual(st, {LEVER_KEYS[0]: "0,3,7,15,19,23,27,39,43", LEVER_KEYS[1]: "1", LEVER_KEYS[2]: "1",
                              QSA_OFF_KEY: "0"})
        self.assertEqual(qc.lever_env("place_iu4"),
                         {LEVER_KEYS[0]: "0,3,7,15,19,23,27,39,43", LEVER_KEYS[1]: "1", LEVER_KEYS[2]: "0",
                          QSA_OFF_KEY: "0"})

    def test_actual_qsa_gather_is_pinned_off_on_qualified_arms(self):
        for arm in ("off", "place_iu4"):
            self.assertEqual(qc.lever_env(arm)[QSA_OFF_KEY], "0")
            self.assertEqual(qc.applied_env(arm, "/rd")[QSA_OFF_KEY], "0")

    def test_place_set_is_nine_distinct_ascending(self):
        self.assertEqual(len(qc.PLACE_LAYERS), 9)
        self.assertEqual(qc.PLACE_LAYERS, sorted(set(qc.PLACE_LAYERS)))

    def test_unknown_arm_refused(self):
        with self.assertRaises(qc.Refusal):
            qc.lever_env("q8w")  # removed lever

    def test_f32_state_is_never_applied(self):
        for arm in qc.ARMS:
            self.assertEqual(qc.applied_env(arm, "/rd")["HIPFIRE_STATE_QUANT"], "q8")


class ProcessEnv(unittest.TestCase):
    PARENT = {"PATH": "/usr/bin", "HIPFIRE_QWEN4_MOE_SYM_IU4": "1", "HIPFIRE_FN_QSA_GATHER_GFX1100": "1",
              "HIPFIRE_QWEN4_QSA_WMMA_GATHER": "1",
              "HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS": "spread", "HIPFIRE_STATE_QUANT": "f32",
              qc.EXPLICIT_LAYERS_ENV: "9", "HIP_VISIBLE_DEVICES": "3", "ROCR_VISIBLE_DEVICES": "x",
              "HSA_FOO": "1", "GPU_BAR": "1", "AMD_LOG_LEVEL": "9", "HIPFIRE_QWEN4_Q8_TRUNK_WMMA": "1",
              "FABRIC_JOB": "1:2:xtx", "FABRIC_CARDS": "xtx", "HIPFIRE_FABRIC_ADMISSION": "tok", "HOME": "/h"}

    def test_inherited_lever_flags_are_stripped_then_explicitly_applied(self):
        env, unset, applied = qc.process_env(self.PARENT, "off", "/rd")
        self.assertEqual(env[LEVER_KEYS[1]], "0")           # inherited =1 did not leak into OFF
        self.assertEqual(env[LEVER_KEYS[2]], "0")
        self.assertEqual(env[QSA_OFF_KEY], "0")            # inherited =1 actual gather killed, not leaked
        self.assertEqual(env[LEVER_KEYS[0]], "front")
        self.assertNotIn(qc.EXPLICIT_LAYERS_ENV, env)
        self.assertNotIn("HIPFIRE_QWEN4_Q8_TRUNK_WMMA", env)  # q8w removed, inherited value stripped
        self.assertEqual(env["HIPFIRE_STATE_QUANT"], "q8")
        for k in ("HSA_FOO", "GPU_BAR", "AMD_LOG_LEVEL"):
            self.assertNotIn(k, env)
        self.assertIn(qc.EXPLICIT_LAYERS_ENV, unset)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], qc.DEVICE["uuid"])
        self.assertEqual(env["HOME"], "/rd/home")
        self.assertEqual(env["FABRIC_JOB"], "1:2:xtx")
        self.assertEqual(env["HIPFIRE_FABRIC_ADMISSION"], "tok")
        self.assertEqual(env[qc.SELECTOR_ENV], "1")

    def test_stack_applies_all_three_on(self):
        env, _u, applied = qc.process_env(self.PARENT, "stack", "/rd")
        self.assertEqual([env[k] for k in LEVER_KEYS], ["0,3,7,15,19,23,27,39,43", "1", "1"])
        for k, v in qc.PINNED_ENV.items():
            self.assertEqual(env[k], v)
        self.assertEqual(applied[qc.SELECTOR_ENV], "1")

    def test_place_iu4_applies_spread_and_iu4_with_qsa_off(self):
        env, _u, applied = qc.process_env(self.PARENT, "place_iu4", "/rd")
        self.assertEqual([env[k] for k in LEVER_KEYS], ["0,3,7,15,19,23,27,39,43", "1", "0"])
        for k, v in qc.PINNED_ENV.items():
            self.assertEqual(env[k], v)
        self.assertEqual(applied[qc.SELECTOR_ENV], "1")

    def test_pinned_class_is_the_native_eight(self):
        self.assertEqual(qc.PINNED_ENV, {
            "HIPFIRE_STATE_QUANT": "q8", "HIPFIRE_KV_MODE": "q8", "HIPFIRE_GRAPH": "0",
            "HIPFIRE_NORMALIZE_PROMPT": "0", "HIPFIRE_CHAT_CURRENT_DATE": "2026-10-17",
            "HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_PLE_FORMAT": "q4", "HIPFIRE_QWEN4_D2_DEFAULT": "0"})

    def test_applied_env_is_deterministic_and_run_dir_private(self):
        a = qc.applied_env("iu4", "/rd")
        self.assertEqual(a, qc.applied_env("iu4", "/rd"))
        for k in ("HOME", "HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE"):
            self.assertTrue(a[k].startswith("/rd/"))


class ArmPlans(unittest.TestCase):
    def test_plan_differs_only_in_expect_env_and_asserts_all_lever_flags(self):
        base = base_plan()
        for arm in qc.ARMS:
            p = qc.arm_plan(base, arm)
            self.assertTrue(qc.plan_diff_only_expect_env(base, p))
            for k, v in qc.lever_env(arm).items():
                self.assertEqual(p["expect"]["env"][k], v)
            for k, v in qc.PINNED_ENV.items():
                self.assertEqual(p["expect"]["env"][k], v)
            self.assertEqual(p["expect"]["env_absent"], [qc.EXPLICIT_LAYERS_ENV])
            self.assertEqual(p["expect"]["baseline_env"], base["expect"]["baseline_env"])
        self.assertNotIn(LEVER_KEYS[0], base["expect"]["env"])  # input untouched (deepcopy)
        self.assertNotIn(QSA_OFF_KEY, base["expect"]["env"])

    def test_qualified_arm_plans_pin_actual_gather_off(self):
        base = base_plan()
        for arm in ("off", "place_iu4"):
            p = qc.arm_plan(base, arm)
            self.assertEqual(p["expect"]["env"][QSA_OFF_KEY], "0")

    def test_native_counts_tiles_are_not_relaxed(self):
        for field, bad in (("resident_layers", 10), ("s1_tile_rows", 1536), ("baseline_tile_rows", 2048)):
            b = base_plan()
            b["expect"][field] = bad
            with self.assertRaises(qc.Refusal):
                qc.arm_plan(b, "off")

    def test_missing_n9_absent_assertion_or_pinned_value_refused(self):
        b = base_plan()
        b["expect"]["env_absent"] = []
        with self.assertRaises(qc.Refusal):
            qc.arm_plan(b, "off")
        b = base_plan()
        b["expect"]["env"]["HIPFIRE_STATE_QUANT"] = "f32"
        with self.assertRaises(qc.Refusal):
            qc.arm_plan(b, "off")

    def test_base_plan_already_naming_a_lever_is_refused(self):
        for where in ("env", "env_absent", "baseline_env"):
            b = base_plan()
            if where == "env_absent":
                b["expect"]["env_absent"].append(LEVER_KEYS[1])
            else:
                b["expect"][where][LEVER_KEYS[1]] = "0"
            with self.assertRaises(qc.Refusal):
                qc.arm_plan(b, "off")

    def test_schema_checked(self):
        b = base_plan()
        b["schema"] = "x"
        with self.assertRaises(qc.Refusal):
            qc.arm_plan(b, "off")

    def test_corpus_change_is_detected(self):
        base = base_plan()
        p = qc.arm_plan(base, "off")
        p["primary"]["contexts"][0]["id"] = "other"
        self.assertFalse(qc.plan_diff_only_expect_env(base, p))


class Manifests(unittest.TestCase):
    def test_each_arm_phase_manifest(self):
        camp = fake_camp()
        for arm in qc.ARMS:
            for ph in qc.PHASES:
                m = qc.native_manifest(camp, arm, ph)
                self.assertEqual(m["schema"], qc.MANIFEST_SCHEMA)
                self.assertEqual(m["binary"]["sha256"], camp["exe"]["sha256"])
                self.assertEqual(m["launch"]["lever_env"], qc.lever_env(arm))
                for k, v in qc.lever_env(arm).items():
                    self.assertEqual(m["launch"]["env"]["set"][k], v)
                self.assertEqual(m["launch"]["command"][:3], [camp["exe"]["path"], "--phase", ph])
                self.assertNotIn(qc.EXPLICIT_LAYERS_ENV, m["launch"]["env"]["set"])
                self.assertEqual(m["source"]["arm"], arm)
                self.assertEqual(m["model"]["mtime_sec"], 1790928378)
                self.assertEqual(m["model"]["mtime_nsec"], 750535607)
                self.assertEqual(m["model"]["sha256"], camp["model"]["sha256"])
                self.assertTrue(m["source"])  # native: non-empty object
                self.assertEqual(set(m), {"schema", "binary", "source", "launch", "model"})  # native refuses unknown keys

    def test_one_executable_and_distinct_outputs_for_every_run(self):
        camp = fake_camp()
        outs = {r["out_dir"] for a in camp["arms"].values() for r in a["runs"].values()}
        runs = {r["run_dir"] for a in camp["arms"].values() for r in a["runs"].values()}
        self.assertEqual(len(outs), 12)
        self.assertEqual(len(runs), 12)
        self.assertEqual({r["argv"][0] for a in camp["arms"].values() for r in a["runs"].values()},
                         {camp["exe"]["path"]})

    def test_native_argv_shape(self):
        argv = qc.native_argv("/e", "s1b", "/p", "/m", "/mod", "/o")
        self.assertEqual(argv, ["/e", "--phase", "s1b", "--plan", "/p", "--manifest", "/m", "--model", "/mod",
                                "--out-dir", "/o"])

    def test_only_s1a_and_s1b_gpu_phases(self):
        self.assertEqual(qc.PHASES, ("s1a", "s1b"))
        self.assertEqual(list(qc.ARMS), ["off", "place", "iu4", "qsa", "stack", "place_iu4"])
        self.assertEqual(len(qc.SERIAL_ORDER), 12)
        self.assertEqual(qc.SERIAL_ORDER[:4], [("off", "s1a"), ("off", "s1b"), ("place", "s1a"), ("place", "s1b")])
        self.assertEqual(qc.SERIAL_ORDER[-2:], [("place_iu4", "s1a"), ("place_iu4", "s1b")])


class LaunchArgs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camp = fake_camp(self.tmp.name)
        with open(os.path.join(self.tmp.name, "campaign.json"), "w") as f:
            json.dump(self.camp, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_cap_and_class(self):
        cmd = qc.build_launch_command(self.camp, "stack", "s1a", 1800, "P0")
        self.assertEqual(cmd[0], "bash")
        i = cmd.index("--max-secs")
        self.assertEqual(cmd[i + 1], "1800")
        for flag, val in (("--card", "xtx"), ("--mem-gb", "22.4"), ("--hostmem-gb", "67"), ("--priority", "P0")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)
        self.assertIn("--hostmem-lock", cmd)
        j = cmd.index("--")
        inner = cmd[j + 1:]
        self.assertEqual(inner[:3], ["python3", "-B", os.path.realpath(qc.__file__)])
        self.assertEqual(inner[3], "run-phase")
        self.assertEqual(inner[inner.index("--arm") + 1], "stack")
        self.assertEqual(inner[inner.index("--phase") + 1], "s1a")
        self.assertEqual(len(inner[inner.index("--expect-campaign-sha256") + 1]), 64)

    def test_exactly_one_phase_per_job(self):
        cmd = qc.build_launch_command(self.camp, "off", "s1b", 1500, "P1")
        inner = cmd[cmd.index("--") + 1:]
        self.assertEqual(inner.count("--phase"), 1)
        self.assertEqual(inner.count("--arm"), 1)

    def test_refuses_over_1800_and_bad_priority_and_phase(self):
        for secs in (1801, 0, -5, 1800.0):
            with self.assertRaises(qc.Refusal):
                qc.build_launch_command(self.camp, "off", "s1a", secs, "P1")
        with self.assertRaises(qc.Refusal):
            qc.build_launch_command(self.camp, "off", "s1a", 1800, "high")
        with self.assertRaises(qc.Refusal):
            qc.build_launch_command(self.camp, "off", "baseline", 1800, "P1")
        with self.assertRaises(qc.Refusal):
            qc.build_launch_command(self.camp, "carry", "s1a", 1800, "P1")

    def test_child_cap_leaves_100s_for_drain(self):
        self.assertEqual(qc.CHILD_CAP_SECS + 100, qc.MAX_JOB_SECS)

    def test_detach_script_closes_every_fd_above_2_and_execs(self):
        self.assertIn("${f##*/}", qc.DETACH_SCRIPT)
        self.assertIn("-gt 2", qc.DETACH_SCRIPT)
        self.assertIn("exec env OUT_PREFIX", qc.DETACH_SCRIPT)


class Lifecycle(unittest.TestCase):
    def test_group_members_finds_exactly_own_group_and_terminate_drains_it(self):
        p = subprocess.Popen(["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                if qc.group_members(p.pid):
                    break
                time.sleep(0.02)
            self.assertEqual([m[0] for m in qc.group_members(p.pid)], [p.pid])
            self.assertNotIn(os.getpid(), [m[0] for m in qc.group_members(p.pid)])
            left = qc.terminate_group(p.pid, 5)
            p.wait(timeout=10)
            self.assertEqual(left, [])
            self.assertEqual(qc.group_members(p.pid), [])
            self.assertEqual(p.returncode, -signal.SIGTERM)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()

    def test_term_ignoring_child_group_is_killed(self):
        p = subprocess.Popen(["bash", "-c", "trap '' TERM; while :; do sleep 1; done"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.3)
            left = qc.terminate_group(p.pid, 0.5)
            p.wait(timeout=15)
            self.assertEqual(left, [])
            self.assertEqual(p.returncode, -signal.SIGKILL)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()

    def test_terminate_of_an_empty_group_is_a_noop(self):
        p = subprocess.Popen(["true"], start_new_session=True)
        p.wait()
        self.assertEqual(qc.terminate_group(p.pid, 0.2), [])


class FakeHost(qc.Host):
    def __init__(self, files):
        super().__init__(environ={})
        self.files = files

    def read_text(self, path):
        if path not in self.files:
            raise OSError(path)
        return self.files[path]


class DeviceChecks(unittest.TestCase):
    BDF = "/sys/bus/pci/devices/0000:67:00.0/"

    def files(self, **kw):
        d = {self.BDF + "vendor": "0x1002\n", self.BDF + "device": "0x744c\n", self.BDF + "mem_info_vram_used": "123\n"}
        d.update({self.BDF + k: v for k, v in kw.items()})
        return d

    def test_healthy(self):
        self.assertEqual(qc.device_state(FakeHost(self.files())), (True, []))

    def test_vendor_disappearance_is_not_ok(self):
        f = self.files()
        del f[self.BDF + "vendor"]
        ok, probs = qc.device_state(FakeHost(f))
        self.assertFalse(ok)
        self.assertTrue(probs)

    def test_wrong_device_or_uuid(self):
        self.assertFalse(qc.device_state(FakeHost(self.files(device="0x7551\n")))[0])
        self.assertFalse(qc.device_state(FakeHost(self.files(unique_id="deadbeefdeadbeef\n")))[0])
        self.assertTrue(qc.device_state(FakeHost(self.files(unique_id="43390A851E296EE5\n")))[0])

    def test_vram_and_memavailable_parsers(self):
        f = self.files()
        f["/proc/meminfo"] = "MemTotal: 1 kB\nMemAvailable:   83886080 kB\n"
        h = FakeHost(f)
        self.assertEqual(qc.vram_used(h), 123)
        self.assertEqual(qc.mem_available_kib(h), 83886080)
        self.assertGreaterEqual(qc.mem_available_kib(h), qc.MIN_MEM_AVAILABLE_KIB)
        self.assertEqual(qc.vram_used(FakeHost({})), -1)

    def test_identity_constants(self):
        self.assertEqual(qc.DEVICE["bdf"], "0000:67:00.0")
        self.assertEqual(qc.DEVICE["vendor_id"], "0x1002")
        self.assertEqual(qc.DEVICE["device_id"], "0x744c")
        self.assertEqual(qc.DEVICE["uuid"], "GPU-43390a851e296ee5")


class RunPhaseGuards(unittest.TestCase):
    def test_refuses_outside_an_xtx_fabric_hold(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            camp = fake_camp(tmp.name)
            data = json.dumps(camp, indent=2, sort_keys=True).encode()
            with open(os.path.join(tmp.name, "campaign.json"), "wb") as f:
                f.write(data)
            ns = type("A", (), {"root": tmp.name, "expect_campaign_sha256": qc.sha256_bytes(data), "arm": "stack",
                                "phase": "s1a", "cap_secs": 1700})()
            with self.assertRaises(qc.Refusal):
                qc.run_phase(ns, host=FakeHost({}))                         # no FABRIC_JOB
            h = FakeHost({})
            h._environ = {"FABRIC_JOB": "1:2:xtx", "FABRIC_CARDS": "r9700"}  # wrong card
            with self.assertRaises(qc.Refusal):
                qc.run_phase(ns, host=h)
            ns.expect_campaign_sha256 = "0" * 64                              # campaign pin mismatch
            with self.assertRaises(qc.Refusal):
                qc.run_phase(ns, host=h)
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------- evidence

def receipt(iu4=False, place=None, status="ok", tiles=None, route="layer-major"):
    place = list(range(9)) if place is None else place
    mo = {"window_route": "aggregate-sym-iu4" if iu4 else "aggregate", "requested": iu4,
          "descriptor": {"arch": "gfx1100"} if iu4 else None, "source_headers_symmetric": True if iu4 else None,
          "final_headers_verified": [40, 40] if iu4 else [0, 40], "a4_sidecar_bytes": 0,
          "tiles": tiles or {"aggregate_iu4": 22 if iu4 else 0, "aggregate_wmma": 0 if iu4 else 22,
                             "aggregate_f32_tail": 1, "slab_k2": 0, "k1_or_decode": 0}}
    return {"route": route, "status": status, "resident_layers": 9, "tile_rows": 768, "retained_scratch_rows": 768,
            "resident_layer_set": place, "moe_iu4": mo}


def cells(**kw):
    return [{"cell": "primary-ctx16k-teacher", "prefill": {"receipts": [receipt(**kw)]}},
            {"cell": "primary-ctx4k-teacher", "prefill": {"receipts": [receipt(**kw)]}},
            {"cell": "det-p512", "prefill": {"receipts": [{"route": "incumbent", "bypass_reason": "TooShort"}]}}]


LOAD_FRONT = "qwen4 slowlink load: placement=slowlink-profile resident layers=9 host-mapped tensors=80 slowlink profile=true\n"
LOAD_SET = ("qwen4 slowlink load: placement=slowlink-profile resident layers=9 resident_layer_set=[0, 3, 7, 15, 19, 23, "
            "27, 39, 43] host-mapped tensors=80 slowlink profile=true\n")
QSA_LINE = "  qwen4 QSA gather workspace: 192 MiB reserved for 16897 context tokens, 192 MiB committed\n"
QSA_PACK_LINE = ("  indexed_attention_gathered_wmma: indexed package verified: /s/bin/kernels/compiled/gfx1100/"
                 "indexed_attention_gathered_wmma.hsaco\n")


class RouteEvidence(unittest.TestCase):
    def test_off_arm_clean(self):
        ev = qc.route_evidence("off", cells(), LOAD_FRONT, [])
        self.assertTrue(ev["ok"], ev)
        self.assertEqual(ev["qsa"]["strength"], "none")

    def test_place_requires_actual_set_in_receipts_and_loader_log(self):
        ok = qc.route_evidence("place", cells(place=qc.PLACE_LAYERS), LOAD_SET, [])
        self.assertTrue(ok["place"]["ok"], ok["place"])
        self.assertEqual(ok["place"]["loader_set"], qc.PLACE_LAYERS)
        bad = qc.route_evidence("place", cells(), LOAD_FRONT, [])   # env requested but front placement actually ran
        self.assertFalse(bad["place"]["ok"])
        bad = qc.route_evidence("place", cells(place=qc.PLACE_LAYERS), LOAD_FRONT, [])
        self.assertFalse(bad["place"]["ok"])
        leaked = qc.route_evidence("off", cells(place=qc.PLACE_LAYERS), LOAD_SET, [])
        self.assertFalse(leaked["place"]["ok"])

    def test_iu4_requires_sym_route_tiles_and_headers(self):
        ok = qc.route_evidence("iu4", cells(iu4=True), LOAD_FRONT, [])
        self.assertTrue(ok["iu4"]["ok"], ok["iu4"])
        self.assertGreater(ok["iu4"]["tile_totals"]["aggregate_iu4"], 0)
        bad = qc.route_evidence("iu4", cells(iu4=False), LOAD_FRONT, [])
        self.assertFalse(bad["iu4"]["ok"])
        leaked = qc.route_evidence("off", cells(iu4=True), LOAD_FRONT, [])
        self.assertFalse(leaked["iu4"]["ok"])

    def test_iu4_incumbent_fallback_tiles_detected(self):
        c = cells(iu4=True)
        c[0]["prefill"]["receipts"][0]["moe_iu4"]["tiles"]["aggregate_wmma"] = 3
        self.assertFalse(qc.route_evidence("iu4", c, LOAD_FRONT, [])["iu4"]["ok"])

    def test_qsa_needs_workspace_and_a_resolved_gather_module_and_off_arm_must_have_neither(self):
        ok = qc.route_evidence("qsa", cells(), LOAD_FRONT + QSA_LINE + QSA_PACK_LINE, [])   # pack hit: kcache stays empty
        self.assertTrue(ok["qsa"]["ok"], ok["qsa"])
        self.assertIn("first-gather-launch", ok["qsa"]["strength"])
        kc = qc.route_evidence("qsa", cells(), LOAD_FRONT + QSA_LINE, ["kcache/x/indexed_attention_gathered_wmma.hsaco"])
        self.assertTrue(kc["qsa"]["ok"], kc["qsa"])                                         # pack miss: kcache witness
        ws_only = qc.route_evidence("qsa", cells(), LOAD_FRONT + QSA_LINE, [])              # load-time config only
        self.assertFalse(ws_only["qsa"]["ok"])
        self.assertIn("NOT proof of dispatch", ws_only["qsa"]["strength"])
        none = qc.route_evidence("qsa", cells(), LOAD_FRONT, [])
        self.assertFalse(none["qsa"]["ok"])
        leaked = qc.route_evidence("off", cells(), LOAD_FRONT + QSA_LINE + QSA_PACK_LINE, [])
        self.assertFalse(leaked["qsa"]["ok"])
        leaked = qc.route_evidence("off", cells(), LOAD_FRONT + QSA_PACK_LINE, [])
        self.assertFalse(leaked["qsa"]["ok"])


    def test_common_route_problems(self):
        ev = qc.route_evidence("off", cells(status="aborted: x"), LOAD_FRONT, [])
        self.assertFalse(ev["common"]["ok"])
        ev = qc.route_evidence("off", [c for c in cells() if c["cell"] != "primary-ctx4k-teacher"], LOAD_FRONT, [])
        self.assertFalse(ev["common"]["ok"])
        self.assertFalse(qc.route_evidence("off", [], LOAD_FRONT, [])["ok"])

    def test_stack_all_three(self):
        c = cells(iu4=True, place=qc.PLACE_LAYERS)
        ev = qc.route_evidence("stack", c, LOAD_SET + QSA_LINE + QSA_PACK_LINE, [])
        self.assertTrue(ev["ok"], ev)

    def test_place_iu4_asserts_spread_plus_sym_iu4_with_qsa_off(self):
        c = cells(iu4=True, place=qc.PLACE_LAYERS)
        ev = qc.route_evidence("place_iu4", c, LOAD_SET, [])
        self.assertTrue(ev["ok"], ev)
        self.assertTrue(ev["place"]["on_expected"] and ev["iu4"]["on_expected"])
        self.assertFalse(ev["qsa"]["on_expected"])
        self.assertEqual(ev["place"]["expected_set"], qc.PLACE_LAYERS)
        self.assertGreater(ev["iu4"]["tile_totals"]["aggregate_iu4"], 0)
        leaked_qsa = qc.route_evidence("place_iu4", c, LOAD_SET + QSA_LINE + QSA_PACK_LINE, [])
        self.assertFalse(leaked_qsa["ok"], leaked_qsa)
        iu4_off = qc.route_evidence("place_iu4", cells(iu4=False, place=qc.PLACE_LAYERS), LOAD_SET, [])
        self.assertFalse(iu4_off["ok"], iu4_off)
        front = qc.route_evidence("place_iu4", cells(iu4=True), LOAD_FRONT, [])
        self.assertFalse(front["ok"], front)

    def test_s1b_all_native_det_cases_and_eligible_routes_required(self):
        cases = base_plan()["determinism"]["case_ids"]
        rows = [512, 1536, 1537, 8192, 4100, 3400, 594]
        cs = [{"cell": "det-" + cid, "prefill": {"calls": [{"rows": n}],
               "receipts": [receipt() if n >= 1537 else {"route": "incumbent"}]}}
              for cid, n in zip(cases, rows)]
        ev = qc.route_evidence("off", cs, LOAD_FRONT, [], phase="s1b", expected_cases=cases)
        self.assertTrue(ev["ok"], ev)
        self.assertFalse(qc.route_evidence("off", cs, LOAD_FRONT, [], phase="s1a",
                                           expected_cases=cases)["common"]["ok"])
        for index in range(len(cs)):
            with self.subTest(missing=cases[index]):
                self.assertFalse(qc.route_evidence("off", cs[:index] + cs[index + 1:], LOAD_FRONT, [],
                                                   phase="s1b", expected_cases=cases)["ok"])
        for index in (2, 3, 4, 5):
            bad = copy.deepcopy(cs)
            bad[index]["prefill"]["receipts"][0] = {"route": "incumbent"}
            self.assertFalse(qc.route_evidence("off", bad, LOAD_FRONT, [], phase="s1b",
                                               expected_cases=cases)["ok"])
        bad = copy.deepcopy(cs)
        bad[2]["prefill"]["receipts"][0]["status"] = "aborted"
        self.assertFalse(qc.route_evidence("off", bad, LOAD_FRONT, [], phase="s1b",
                                           expected_cases=cases)["ok"])
        self.assertFalse(qc.route_evidence("off", cs, LOAD_FRONT, [], phase="s1b")["ok"])
        self.assertFalse(qc.route_evidence("place", cs, LOAD_FRONT, [], phase="s1b", expected_cases=cases)["ok"])
        self.assertFalse(qc.route_evidence("iu4", cs, LOAD_FRONT, [], phase="s1b", expected_cases=cases)["ok"])
        self.assertFalse(qc.route_evidence("qsa", cs, LOAD_FRONT, [], phase="s1b", expected_cases=cases)["ok"])

class PackBinding(unittest.TestCase):
    SEALED = "/s/bin/kernels/compiled/gfx1100"
    HOT = "/r/kcache"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sealed = os.path.join(self.tmp.name, "kernels", "compiled", "gfx1100")
        self.hot = os.path.join(self.tmp.name, "kcache")
        os.makedirs(self.sealed)
        os.makedirs(self.hot)
        self.obj_bytes = b"fake-hsaco-bytes"
        self.obj_sha = qc.sha256_bytes(self.obj_bytes)
        idx = {"module": "add", "object_sha256": self.obj_sha}
        self.idx_bytes = json.dumps(idx).encode()
        with open(os.path.join(self.sealed, "add.hsaco"), "wb") as f:
            f.write(self.obj_bytes)
        with open(os.path.join(self.sealed, "add.index.json"), "wb") as f:
            f.write(self.idx_bytes)
        self.files = [{"rel": "add.hsaco", "size": len(self.obj_bytes), "sha256": self.obj_sha},
                      {"rel": "add.index.json", "size": len(self.idx_bytes),
                       "sha256": qc.sha256_bytes(self.idx_bytes)}]

    def tearDown(self):
        self.tmp.cleanup()

    def stderr(self, lookup=None, hits=("add",)):
        lookup = self.sealed if lookup is None else lookup
        lines = ["  pre-compiled kernels: %s" % lookup]
        for m in hits:
            lines.append("  %s: indexed package verified: %s/%s.hsaco" % (m, self.sealed, m))
        return "\n".join(lines) + "\n"

    def ev(self, stderr_text, kcache=(), pack_files="sealed"):
        return qc.pack_binding_evidence(stderr_text, list(kcache), self.sealed, self.hot,
                                        self.files if pack_files == "sealed" else pack_files)

    def test_indexed_hits_bound_to_sealed_index_and_object_pass(self):
        ev = self.ev(self.stderr())
        self.assertTrue(ev["ok"], ev)
        self.assertEqual((ev["pack_hits"], ev["pack_verified"]), (1, 1))

    def test_zero_hits_fails_closed(self):
        ev = self.ev("  pre-compiled kernels: %s\n" % self.sealed)
        self.assertFalse(ev["ok"])

    def test_missing_lookup_line_fails(self):
        ev = self.ev("  add: indexed package verified: %s/add.hsaco\n" % self.sealed)
        self.assertFalse(ev["ok"])

    def test_hit_outside_sealed_dir_fails(self):
        ev = self.ev(self.stderr() + "  x: indexed package verified: /tmp/x.hsaco\n")
        self.assertFalse(ev["ok"])

    def test_missing_index_fails(self):
        os.unlink(os.path.join(self.sealed, "add.index.json"))
        self.assertFalse(self.ev(self.stderr())["ok"])

    def test_object_sha_mismatch_fails(self):
        with open(os.path.join(self.sealed, "add.hsaco"), "wb") as f:
            f.write(b"tampered")
        self.assertFalse(self.ev(self.stderr())["ok"])

    def test_sealed_manifest_mismatch_fails(self):
        bad = [dict(self.files[0], sha256="0" * 64), self.files[1]]
        self.assertFalse(self.ev(self.stderr(), pack_files=bad)["ok"])

    def test_live_kcache_object_fails_closed(self):
        with open(os.path.join(self.hot, "add.abcdef.hsaco"), "wb") as f:
            f.write(b"jit")
        ev = self.ev(self.stderr())
        self.assertFalse(ev["ok"])
        self.assertEqual(len(ev["jit_objects"]), 1)

    def test_recorded_names_must_match_live_rescan(self):
        with open(os.path.join(self.hot, "add.abcdef.hsaco"), "wb") as f:
            f.write(b"jit")
        ev = qc.pack_binding_evidence(self.stderr(), [], self.sealed, self.hot, self.files)
        self.assertFalse(ev["ok"])

    def test_fallback_markers_fail_closed(self):
        ev = self.ev(self.stderr() + "  add: packaged object rejected (stale); falling back to JIT\n")
        self.assertFalse(ev["ok"])

    def test_foreign_lookup_dir_fails(self):
        ev = self.ev(self.stderr(lookup="/tmp/evil"))
        self.assertFalse(ev["ok"])

    def test_private_home_cold_lookup_symlink_layout(self):
        home = os.path.join(self.tmp.name, "home", ".hipfire")
        os.makedirs(os.path.join(home, "bin"))
        kernels_root = os.path.dirname(os.path.dirname(self.sealed))
        link = os.path.join(home, "bin", "kernels")
        os.symlink(kernels_root, link)
        self.assertEqual(os.path.realpath(link), os.path.realpath(kernels_root))
        self.assertEqual(os.path.realpath(os.path.join(link, "compiled", "gfx1100")),
                         os.path.realpath(self.sealed))


class Determinism(unittest.TestCase):
    CASES = ["p512", "p1536"]

    def det(self, **over):
        d = {"schema": "hipfire.qwen4.slowlink_quality.determinism.v1", "all_identical": True, "cases": []}
        for cid in self.CASES:
            d["cases"].append({"case": cid, "cell": "det-" + cid, "identical": True, "token_ids_equal": True,
                               "finish_reason": ["length", "length"], "rows": [33, 33],
                               "first_differing_row": None, "stream_sha256": [cid + "a", cid + "a"]})
        for k, v in over.items():
            d["cases"][0][k] = v
        return d

    def cells(self, text_b="hello"):
        mk = lambda t: [{"cell": "det-" + c, "stream_sha256": c + "a", "text": t,  # noqa: E731
                         "prefill": {"receipts": [receipt()]}} for c in self.CASES]
        return mk("hello"), mk(text_b)

    def test_ok(self):
        a, b = self.cells()
        r = qc.determinism_report(self.det(), self.CASES, a, b)
        self.assertTrue(r["ok"], r["problems"])
        self.assertTrue(r["route_signature_equal"])

    def test_token_or_bit_difference_and_text_difference(self):
        a, b = self.cells()
        self.assertFalse(qc.determinism_report(self.det(token_ids_equal=False), self.CASES, a, b)["ok"])
        self.assertFalse(qc.determinism_report(self.det(identical=False), self.CASES, a, b)["ok"])
        a, b = self.cells("different")
        self.assertFalse(qc.determinism_report(self.det(), self.CASES, a, b)["ok"])

    def test_missing_case_and_stream_hash_mismatch(self):
        a, b = self.cells()
        d = self.det()
        d["cases"].pop()
        self.assertFalse(qc.determinism_report(d, self.CASES, a, b)["ok"])
        a2 = copy.deepcopy(a)
        a2[0]["stream_sha256"] = "zzz"
        self.assertFalse(qc.determinism_report(self.det(), self.CASES, a2, b)["ok"])

    def test_route_signature_difference_between_a_and_b(self):
        a, b = self.cells()
        b[0]["prefill"]["receipts"] = [receipt(iu4=True)]
        r = qc.determinism_report(self.det(), self.CASES, a, b)
        self.assertFalse(r["route_signature_equal"])
        self.assertFalse(r["ok"])

    def test_finish_pair_fails_closed_on_malformed_or_different_values(self):
        a, b = self.cells()
        for value in (None, [], ["length"], ["length", "length", "length"], "length",
                      [None, None], [1, 1], ["length", "stop"]):
            with self.subTest(value=value):
                r = qc.determinism_report(self.det(finish_reason=value), self.CASES, a, b)
                self.assertFalse(r["ok"])
                self.assertFalse(r["cases"][0]["finish_equal"])
        for finish in ("length", "stop"):
            self.assertTrue(qc.determinism_report(self.det(finish_reason=[finish, finish]), self.CASES, a, b)["ok"])
        legacy = self.det()
        del legacy["cases"][0]["finish_reason"]
        legacy["cases"][0]["finish_equal"] = True
        self.assertFalse(qc.determinism_report(legacy, self.CASES, a, b)["ok"])

    def test_native_schema_and_duplicate_cases_fail_closed(self):
        a, b = self.cells()
        for value in (None, "invented"):
            det = self.det()
            det["schema"] = value
            self.assertFalse(qc.determinism_report(det, self.CASES, a, b)["ok"])
        det = self.det()
        det["cases"].append(det["cases"][0])
        self.assertFalse(qc.determinism_report(det, self.CASES, a, b)["ok"])


class TextEvidence(unittest.TestCase):
    def test_flags_and_review_requirement(self):
        cs = [{"cell": "det-a", "text": "A sentence.", "token_ids": list(range(10)), "finish_reason": "length"},
              {"cell": "det-b", "text": "", "token_ids": [7] * 12, "finish_reason": "length"},
              {"cell": "det-c", "text": "x\ufffdy", "token_ids": [1, 2, 3], "finish_reason": "stop"},
              {"cell": "no-text"}]
        t = qc.text_evidence(cs, cs)
        self.assertTrue(t["manual_review_required"])
        flagged = {c.split("@")[0] for c in t["auto_flag_cells"]}
        self.assertEqual(flagged, {"det-b", "det-c"})
        self.assertEqual(len(t["cells"]), 6)


class ReferenceAndBounds(unittest.TestCase):
    def test_bounds_file_pinned_when_present(self):
        path = qc.DEFAULT_BOUNDS_FILE
        if not os.path.exists(path):
            self.skipTest("bounds file not present on this host")
        got = qc.verify_bounds_file(path)
        self.assertEqual(got["values"], qs.GATE_BOUNDS_EXACT)

    def test_bounds_file_wrong_hash_refused(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"x": 1}, f)
        try:
            with self.assertRaises(qc.Refusal):
                qc.verify_bounds_file(f.name)
        finally:
            os.unlink(f.name)

    def test_beta_cells_include_resume_references(self):
        self.assertEqual(qc.BETA_CELLS, {"primary-ctx16k-teacher": 512, "primary-ctx4k-teacher": 512,
                                         "resume-cold": 32, "resume-split": 32})

    def test_parse_arms(self):
        self.assertEqual(qc.parse_arms("stack,off"), ["off", "stack"])
        self.assertEqual(qc.parse_arms(None), ["off", "place_iu4"])
        self.assertEqual(qc.parse_arms("place_iu4,off"), ["off", "place_iu4"])
        for bad in ("off,off", "q8w", "off,carry"):
            with self.assertRaises(qc.Refusal):
                qc.parse_arms(bad)

    def test_expected_beta_plan_pin(self):
        self.assertEqual(qc.EXPECTED_BETA_PLAN_SHA256, PLAN_SHA)

    def test_cached_model_entry_checks_live_stat_without_reading_payload(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "model.bin")
            with open(fake, "wb") as f:
                f.write(b"x" * 10)
            st = os.stat(fake)
            rc = os.path.join(d, "r.json")
            rec = {"path": fake, "size_bytes": 10, "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
                   "sha256": "9" * 64}
            with open(rc, "w") as f:
                json.dump(rec, f)
            m = qc.cached_model_entry(rc, fake, plan_artifact={"path": fake, "size": 10, "sha256": "9" * 64})
            self.assertFalse(m["payload_hashed_here"])
            self.assertEqual(m["sha256"], "9" * 64)
            rec["size_bytes"] = 11
            with open(rc, "w") as f:
                json.dump(rec, f)
            with self.assertRaises(qc.Refusal):
                qc.cached_model_entry(rc, fake)


class SerialDriver(unittest.TestCase):
    def test_serial_covers_full_off_and_place_iu4_routes_in_order(self):
        order = [pair for pair in qc.SERIAL_ORDER if pair[0] in ("off", "place_iu4")]
        self.assertEqual(order, [("off", "s1a"), ("off", "s1b"),
                                 ("place_iu4", "s1a"), ("place_iu4", "s1b")])

    def test_serial_stops_when_an_earlier_phase_has_a_failed_record_and_never_launches_next(self):
        with tempfile.TemporaryDirectory() as d:
            camp = fake_camp(d)
            run0 = camp["arms"]["off"]["runs"]["s1a"]["run_dir"]
            os.makedirs(run0)
            with open(os.path.join(run0, "run-record.json"), "w") as f:
                json.dump({"status": "INCOMPLETE"}, f)
            data = json.dumps(camp, indent=2, sort_keys=True).encode()
            with open(os.path.join(d, "campaign.json"), "wb") as f:
                f.write(data)
            ns = type("A", (), {"root": d, "expect_campaign_sha256": qc.sha256_bytes(data), "priority": "P1",
                                "max_secs": 1800, "wrapper": "/nonexistent", "fabric": "/nonexistent",
                                "max_wait_per_phase": 1, "detach": False})()
            self.assertEqual(qc.serial(ns, host=FakeHost({})), 1)
            self.assertTrue(os.path.exists(os.path.join(d, "serial", "STOPPED")))
            self.assertFalse(os.path.exists(os.path.join(d, "launch")))      # nothing was submitted
            with self.assertRaises(qc.Refusal):                               # STOPPED must be cleared by hand
                qc.serial(ns, host=FakeHost({}))

    def test_serial_refuses_to_submit_when_the_device_is_unhealthy(self):
        with tempfile.TemporaryDirectory() as d:
            camp = fake_camp(d)
            data = json.dumps(camp, indent=2, sort_keys=True).encode()
            with open(os.path.join(d, "campaign.json"), "wb") as f:
                f.write(data)
            ns = type("A", (), {"root": d, "expect_campaign_sha256": qc.sha256_bytes(data), "priority": "P1",
                                "max_secs": 1800, "wrapper": "/nonexistent", "fabric": "/nonexistent",
                                "max_wait_per_phase": 1, "detach": False})()
            self.assertEqual(qc.serial(ns, host=FakeHost({})), 1)           # sysfs unreadable => not healthy
            self.assertFalse(os.path.exists(os.path.join(d, "launch")))


class GlobalNames(unittest.TestCase):
    def test_every_function_resolves_its_global_names(self):
        """`prepare` once referenced `label`/`selected`/`bounds`/`harness` that no assignment defined (NameError on the first real
        run, invisible to every other test): each implicit global of every function must exist in the module or builtins."""
        import builtins
        import symtable
        with open(qc.__file__) as fh:
            src = fh.read()
        problems = []

        def walk(table):
            for sym in table.get_symbols():
                if table.get_type() == "function" and sym.is_global() and sym.is_referenced() and not sym.is_assigned():
                    if not hasattr(qc, sym.get_name()) and not hasattr(builtins, sym.get_name()):
                        problems.append("%s: %s" % (table.get_name(), sym.get_name()))
            for child in table.get_children():
                walk(child)

        walk(symtable.symtable(src, qc.__file__, "exec"))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
