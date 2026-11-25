#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU-only tests of the G2 carrier's judgment functions (no GPU, no scheduler)."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

_SPEC = importlib.util.spec_from_file_location("subq4_g2_carrier", Path(__file__).with_name("subq4_g2_carrier.py"))
c = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(c)


def run(k, **over):
    r = {"k": k, "control": "none", "pass": True, "differences": 0, "executor_error": None, "release_errors": [],
         "provision_refusal": None, "items_compared": 10, "state_checkpoints_compared": 3, "bytes_compared": 1}
    r.update(over)
    return r


def ar_doc():
    return {
        "pass": True, "s1_prefill_parity_k2": True,
        "args": {"ks": [1, 2, 4, 8], "contexts": [128, 256], "steps": 8, "max_seq": 1024, "controls": True,
                 "g0": False, "spec": "ar"},
        "runs": [run(k) for k in (1, 2, 4, 8)],
        "controls": [{"control": n, "detected": True, "clean_after_refusal": True, "how": "x",
                      "run": run(2, control=n, differences=5)} for n in sorted(c.AR_CONTROLS)],
    }


def mtp_doc():
    runs = [run(k, spec="mtp", spec_k=3, routes="cycle", drafts=6, accepted=3, rejected_windows=2, windows=4)
            for k in (1, 2, 4, 8)]
    return {
        "pass": True,
        "args": {"ks": [1, 2, 4, 8], "contexts": [128, 256], "steps": 8, "max_seq": 1024, "controls": True,
                 "g0": False, "spec": "mtp", "k": 3, "routes": "cycle", "promote": None},
        "runs": runs,
        "controls": [
            {"control": "spec_positive_cycle", "matched": True, "exercised": True, "rejected_windows": 2,
             "run": run(2, spec="mtp")},
            {"control": "spec_skip_rollback", "detected": True, "how": "x", "run": run(2, differences=9)},
        ],
    }


def replay_doc():
    names = list(c.REPLAY_REQUIRED)
    return {
        "gate": "qwen4_g2_replay_gate", "ok": True, "failed_checks": [],
        "params": c.replay_params(c.DEFAULT_EXPERT_LAYERS),
        "checks": [{"name": n, "pass": True, "mismatch_count": 0, "error": None, "detail": None} for n in names],
    }


class LaneTests(unittest.TestCase):
    def test_ar_pass_and_negative_controls_are_not_mismatches(self):
        v = c.evaluate_lane_ar(ar_doc())
        self.assertEqual(v["verdict"], "PASS", v["reasons"])
        self.assertEqual(v["mismatches"], 0)

    def test_ar_positive_difference_fails_with_repro(self):
        d = ar_doc()
        d["runs"][2].update(differences=3, **{"pass": False}, first_divergence="req1 logits_3: byte 4")
        v = c.evaluate_lane_ar(d)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertEqual(v["classification"], "mismatch")
        self.assertEqual(v["mismatches"], 3)
        self.assertEqual(v["first_mismatch"]["k"], 4)
        self.assertIn("byte 4", v["first_mismatch"]["first_divergence"])

    def test_ar_missed_control_fails(self):
        d = ar_doc()
        d["controls"][0]["detected"] = False
        self.assertEqual(c.evaluate_lane_ar(d)["classification"], "control_missed")

    def test_ar_missing_k_fails(self):
        d = ar_doc()
        d["runs"].pop()
        self.assertEqual(c.evaluate_lane_ar(d)["verdict"], "FAIL")

    def test_mtp_pass(self):
        v = c.evaluate_lane_mtp(mtp_doc())
        self.assertEqual(v["verdict"], "PASS", v["reasons"])
        self.assertEqual(v["coverage"]["rejected_windows"], 8)

    def test_mtp_unexercised_is_fail(self):
        for key in ("drafts", "accepted", "rejected_windows"):
            d = mtp_doc()
            for r in d["runs"]:
                r[key] = 0
            v = c.evaluate_lane_mtp(d)
            self.assertEqual(v["verdict"], "FAIL", key)
            self.assertEqual(v["classification"], "unexercised", key)

    def test_mtp_skip_rollback_must_be_detected(self):
        d = mtp_doc()
        d["controls"][1]["detected"] = False
        self.assertEqual(c.evaluate_lane_mtp(d)["classification"], "control_missed")

    def test_mtp_wrong_args_fail(self):
        d = mtp_doc()
        d["args"]["routes"] = "floor"
        self.assertEqual(c.evaluate_lane_mtp(d)["verdict"], "FAIL")


class ParityReplayTests(unittest.TestCase):
    def good_parity(self):
        return {"schema": "hipfire.qwen4.state_parity.v1", "status": "pass", "gpu_arch": "gfx1201",
                "real_model": {"status": "pass"}, "compact_state": {"status": "pass"}}

    def test_parity_pass(self):
        self.assertEqual(c.evaluate_parity_state(self.good_parity())["verdict"], "PASS")

    def test_parity_fail_status(self):
        d = self.good_parity()
        d["real_model"]["status"] = "fail"
        d["status"] = "fail"
        v = c.evaluate_parity_state(d)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertEqual(v["first_mismatch"]["path"], "$")

    def test_parity_mismatch_count(self):
        d = self.good_parity()
        d["real_model"]["mismatches"] = 4
        v = c.evaluate_parity_state(d)
        self.assertEqual((v["verdict"], v["mismatches"], v["classification"]), ("FAIL", 4, "mismatch"))

    def test_replay_pass(self):
        v = c.evaluate_replay(replay_doc(), c.DEFAULT_EXPERT_LAYERS)
        self.assertEqual(v["verdict"], "PASS", v["reasons"])

    def test_replay_params_must_echo_residency(self):
        for layers in (15, 20):
            v = c.evaluate_replay(replay_doc(), layers)
            self.assertEqual((v["verdict"], v["classification"]), ("FAIL", "report_invalid"), layers)
        d = replay_doc()
        d["params"] = c.replay_params(15)
        self.assertEqual(c.evaluate_replay(d, 15)["verdict"], "PASS")

    def test_replay_mismatch(self):
        d = replay_doc()
        d["checks"][10].update(mismatch_count=2, **{"pass": False})
        d["ok"] = False
        v = c.evaluate_replay(d, c.DEFAULT_EXPERT_LAYERS)
        self.assertEqual((v["verdict"], v["mismatches"]), ("FAIL", 2))
        self.assertEqual(v["first_mismatch"]["name"], d["checks"][10]["name"])

    def test_replay_missing_required_check(self):
        d = replay_doc()
        d["checks"] = d["checks"][1:]
        self.assertEqual(c.evaluate_replay(d, c.DEFAULT_EXPERT_LAYERS)["classification"], "report_invalid")

    def test_replay_wrong_params(self):
        d = replay_doc()
        d["params"]["iterations"] = 1
        self.assertEqual(c.evaluate_replay(d, c.DEFAULT_EXPERT_LAYERS)["verdict"], "FAIL")

    def skip_doc(self):
        return {
            "gate": "qwen4_g2_replay_gate", "ok": True, "status": "skip", "skipped": True,
            "reason": "Redline not on the default route (config.rs:15–22)", "failed_checks": [], "checks": [],
            "params": c.replay_params(c.DEFAULT_EXPERT_LAYERS), "arch": "gfx1151",
            "route": {"default_route": False, "gpu_arch": "gfx1151"},
        }

    def test_replay_skip_is_skip_not_pass_and_not_rejected(self):
        v = c.evaluate_replay(self.skip_doc(), c.DEFAULT_EXPERT_LAYERS)
        self.assertEqual(v["verdict"], "SKIP", v["reasons"])
        self.assertIsNone(v["classification"])

    def test_replay_skip_must_be_exact_and_checkless(self):
        for edit in (lambda d: d.update(reason="skipped"), lambda d: d.update(skipped=False),
                     lambda d: d.update(checks=[{"name": "x", "pass": True}]),
                     lambda d: d.update(route={"default_route": True}), lambda d: d.pop("route"),
                     lambda d: d.update(ok=False), lambda d: d.update(failed_checks=["x"]),
                     lambda d: d.update(params=c.replay_params(1))):
            d = self.skip_doc()
            edit(d)
            self.assertEqual(c.evaluate_replay(d, c.DEFAULT_EXPERT_LAYERS)["verdict"], "FAIL")

    def test_replay_without_skip_status_still_requires_every_check(self):
        d = self.skip_doc()
        d.pop("status")
        self.assertEqual(c.evaluate_replay(d, c.DEFAULT_EXPERT_LAYERS)["verdict"], "FAIL")


class SmokeReceiptTests(unittest.TestCase):
    def judge(self, doc):
        return c.evaluate_smoke_receipt(doc, json.dumps(doc))

    def bound(self, **extra):
        return {"artifact_sha256": c.ARTIFACT_SHA256, **extra}

    def test_accepts_explicit_pass_forms(self):
        for extra in ({"pass": True}, {"passed": True}, {"status": "PASS"}, {"verdict": "passed"},
                      {"smoke": {"ok": True}}):
            self.assertTrue(self.judge(self.bound(**extra))["verdict_evidence"], extra)

    def test_refuses_implicit_or_failing(self):
        for extra in ({}, {"pass": False}, {"status": "fail"}, {"pass": True, "status": "error"},
                      {"status": "complete"}, {"pass": "yes"}):
            with self.assertRaises(c.Refusal, msg=str(extra)):
                self.judge(self.bound(**extra))

    def test_refuses_unbound_or_synthetic(self):
        with self.assertRaises(c.Refusal):
            self.judge({"pass": True})
        with self.assertRaises(c.Refusal):
            self.judge(self.bound(**{"pass": True, "synthetic": True}))
        with self.assertRaises(c.Refusal):
            c.evaluate_smoke_receipt([], "[]")

    def test_artifact_path_also_binds(self):
        doc = {"pass": True, "artifact": str(c.ARTIFACT)}
        self.assertEqual(self.judge(doc)["bound_to_artifact_by"], ["artifact path"])


class CardTests(unittest.TestCase):
    GPUS = [{"bdf": "0000:e3:00.0", "uuid": "GPU-085289909a86cc63", "arch": "gfx1201"}]

    def test_card_c_matches(self):
        u = "GPU-085289909a86cc63"
        sched = {"name": "C", "bdf": "0000:e3:00.0", "devenv": {"HIPFIRE_DEVICES": u, "ROCR_VISIBLE_DEVICES": u}}
        out = c.validate_card("C", self.GPUS, sched, None)
        self.assertEqual((out["uuid"], out["bdf"]), (u, "0000:e3:00.0"))
        self.assertEqual(out["scheduler"]["status"], "matched")

    def test_no_live_fabricd_refuses(self):
        with self.assertRaisesRegex(c.Refusal, "no live fabricd"):
            c.validate_card("C", self.GPUS, None, "fabricd unreachable")

    def test_wrong_bdf_refused(self):
        gpus = copy.deepcopy(self.GPUS)
        gpus[0]["bdf"] = "0000:03:00.0"
        with self.assertRaises(c.Refusal):
            c.validate_card("C", gpus, {"name": "C", "bdf": "0000:03:00.0"}, None)

    def test_scheduler_disagreement_refused(self):
        sched = {"name": "C", "bdf": "0000:e3:00.0",
                 "devenv": {"HIPFIRE_DEVICES": "GPU-9eb7aeda51c88ffd", "ROCR_VISIBLE_DEVICES": "GPU-9eb7aeda51c88ffd"}}
        with self.assertRaises(c.Refusal):
            c.validate_card("C", self.GPUS, sched, None)

    def test_scheduler_match(self):
        u = "GPU-085289909a86cc63"
        sched = {"name": "C", "bdf": "0000:e3:00.0", "devenv": {"HIPFIRE_DEVICES": u, "ROCR_VISIBLE_DEVICES": u}}
        self.assertEqual(c.validate_card("C", self.GPUS, sched, None)["scheduler"]["status"], "matched")

    def test_fabric_env(self):
        u = "GPU-085289909a86cc63"
        env = {"FABRIC_JOB": "1:2:C", "FABRIC_CARDS": "C", "HIPFIRE_DEVICES": u, "ROCR_VISIBLE_DEVICES": u}
        self.assertEqual(c.check_fabric_env("C", env)["FABRIC_CARDS"], "C")
        with self.assertRaises(c.Refusal):
            c.check_fabric_env("C", {**env, "FABRIC_JOB": ""})
        with self.assertRaises(c.Refusal):
            c.check_fabric_env("C", {**env, "ROCR_VISIBLE_DEVICES": "GPU-9eb7aeda51c88ffd"})


def parse(*extra):
    return c.parse_args(["--smoke-receipt", "/tmp/s.json", "--code-sha", "abcdef1", *extra])


class SubmissionTests(unittest.TestCase):
    def test_fabric_argv_is_single_correct_p1_submission(self):
        argv = c.fabric_argv(parse())
        self.assertEqual(argv[0], "/home/kaden/qcal/locks/fabric/fabric-run.sh")
        self.assertEqual(argv[0], c.FABRIC_RUN)
        joined = " ".join(argv[:argv.index("--")])
        for part in ("--class correct", "--card C", "--mem-gb 30", "--hostmem-gb 44", "--hostmem-lock",
                     "--priority P1", "--owner Subq4WeekLead-2.G2Harness", "--max-secs 7200"):
            self.assertIn(part, joined)
        self.assertIn("--inner", argv)

    def test_scheduler_path_is_the_mandatory_fabric_not_fabric_sched(self):
        self.assertEqual(c.FABRIC_RUN, "/home/kaden/qcal/locks/fabric/fabric-run.sh")
        self.assertNotIn("fabric-sched", c.FABRIC_RUN)
        self.assertNotIn("fabric-sched", " ".join(c.fabric_argv(parse())))

    def test_hostmem_declaration_is_44_for_every_residency(self):
        for layers in ("16", "15"):
            argv = c.fabric_argv(parse("--expert-vram-layers", layers))
            self.assertEqual(argv[argv.index("--hostmem-gb") + 1], "44")
            self.assertEqual(argv[argv.index("--mem-gb") + 1], "30")


class ResidencyTests(unittest.TestCase):
    def test_default_is_16(self):
        self.assertEqual(c.DEFAULT_EXPERT_LAYERS, 16)
        self.assertEqual(parse().expert_vram_layers, 16)

    def test_outer_passes_layers_to_inner(self):
        for layers in ("16", "15"):
            argv = c.fabric_argv(parse("--expert-vram-layers", layers))
            self.assertEqual(argv[argv.index("--expert-vram-layers") + 1], layers)
            self.assertEqual(argv.index("--inner") < argv.index("--expert-vram-layers"), True)
        self.assertEqual(c.fabric_argv(parse())[c.fabric_argv(parse()).index("--expert-vram-layers") + 1], "16")

    def test_layers_must_be_a_non_negative_integer(self):
        for bad in ("all", "1.5", "-1", ""):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=bad):
                parse("--expert-vram-layers", bad)

    def test_every_child_uses_the_layers(self):
        for layers in ("16", "15"):
            args = parse("--expert-vram-layers", layers)
            by = {s["name"]: s for s in c.child_specs(args)}
            self.assertEqual(list(by), ["fn_lane_ar", "fn_lane_mtp", "parity_state", "replay"])
            self.assertEqual(by["fn_lane_ar"]["set_env"], {c.ENV_EXPERT: layers})
            self.assertEqual(by["fn_lane_mtp"]["set_env"], {c.ENV_EXPERT: layers, c.ENV_MTP: "0"})
            self.assertEqual(by["parity_state"]["set_env"], {c.ENV_EXPERT: layers})
            self.assertEqual(by["replay"]["set_env"], {})
            self.assertIn(c.ENV_EXPERT, by["replay"]["drop_env"])
            argv = by["replay"]["argv"]
            self.assertEqual(argv[argv.index("--expert-vram-layers") + 1], layers)
            self.assertNotIn("20", [by[n]["set_env"].get(c.ENV_EXPERT) for n in by])
            self.assertIn("--controls", by["fn_lane_ar"]["argv"])
            for ks in (by["fn_lane_ar"]["argv"], by["fn_lane_mtp"]["argv"]):
                self.assertEqual(ks[ks.index("--ks") + 1], "1,2,4,8")

    def test_replay_verdict_judges_params_against_the_chosen_layers(self):
        for layers in (16, 15):
            spec = {s["name"]: s for s in c.child_specs(parse("--expert-vram-layers", str(layers)))}["replay"]
            doc = replay_doc()
            doc["params"] = c.replay_params(layers)
            self.assertEqual(spec["evaluate"](doc)["verdict"], "PASS", layers)
            doc["params"] = c.replay_params(layers + 1)
            self.assertEqual(spec["evaluate"](doc)["verdict"], "FAIL", layers)


class FabricRequiredTests(unittest.TestCase):
    U = "GPU-085289909a86cc63"

    def reply(self, **job_over):
        job = {"id": "j7", "state": "running", "cls": "correct", "cards": ["C"], "pid": 4242, "priority": 1,
               "owner": "Subq4WeekLead-2.G2Harness", "mem": 30 * c.GIB, "hostmem": 44 * c.GIB, "hostmem_lock": True,
               "max_secs": 7200, "enq": 1.0, "admitted": 2.0, "cmd": "x"}
        job.update(job_over)
        return {"ok": True, "daemon": {"pid": os.getpid(), "host": "hiptrx", "version": "sched-1", "drain": False},
                "cards": [], "jobs": [job]}

    def status_run(self, rc, stdout, stderr=""):
        done = subprocess.CompletedProcess(["x"], rc, stdout, stderr)
        return mock.patch.object(c.subprocess, "run", return_value=done)

    def test_status_uses_the_mandatory_binary_with_json(self):
        with self.status_run(0, json.dumps(self.reply())) as run:
            reply, why = c.fabric_status("C")
        self.assertIsNone(why)
        self.assertEqual(run.call_args.args[0], [c.FABRIC_RUN, "--status", "--card", "C", "--json"])
        self.assertEqual(reply["daemon"]["pid"], os.getpid())

    def test_unreachable_fabricd_is_not_a_scheduler(self):
        # fabric-run.sh --status --json exits 1 (it does not show the in-process view) when fabricd is down.
        with self.status_run(1, "", "fabricd unreachable (No such file or directory)"):
            self.assertEqual(c.scheduler_card_record("C")[0], None)
            self.assertIn("unreachable", c.scheduler_card_record("C")[1])
        with self.status_run(0, "not json"):
            self.assertIsNone(c.fabric_status()[0])
        dead = self.reply()
        dead["daemon"]["pid"] = 2 ** 31 - 2  # no such process
        with self.status_run(0, json.dumps(dead)):
            self.assertIsNone(c.fabric_status()[0])
        no_daemon = {"ok": True, "cards": [], "jobs": []}
        with self.status_run(0, json.dumps(no_daemon)):
            self.assertIsNone(c.fabric_status()[0])

    def test_card_record_comes_from_live_daemon_only(self):
        reply = self.reply()
        reply["cards"] = [{"name": "C", "bdf": "0000:e3:00.0"}]
        with self.status_run(0, json.dumps(reply)):
            self.assertEqual(c.scheduler_card_record("C")[0]["bdf"], "0000:e3:00.0")
            self.assertIsNone(c.scheduler_card_record("B")[0])

    def test_bypass_env_refused(self):
        c.refuse_fabric_bypass({})
        c.refuse_fabric_bypass({"FABRIC_NO_DAEMON": ""})
        for var in ("FABRIC_NO_DAEMON", "FABRIC_SANDBOX"):
            with self.assertRaises(c.Refusal, msg=var):
                c.refuse_fabric_bypass({var: "1"})

    def test_inside_job_requires_a_daemon_jN_job_for_this_wrapper(self):
        env = {"FABRIC_JOB": "4242:99:C"}
        rec = c.check_fabric_job("C", env, self.reply())
        self.assertEqual((rec["id"], rec["wrapper_pid"], rec["state"]), ("j7", 4242, "running"))
        self.assertEqual(rec["fabricd"]["pid"], os.getpid())

    def test_fallback_launch_is_refused(self):
        env = {"FABRIC_JOB": "4242:99:C"}
        # fabric.py's fallback sets FABRIC_JOB too, but fabricd has no job for that wrapper pid.
        with self.assertRaisesRegex(c.Refusal, "fallback"):
            c.check_fabric_job("C", env, self.reply(pid=1))
        with self.assertRaisesRegex(c.Refusal, "fallback"):
            c.check_fabric_job("C", env, {"ok": True, "daemon": {}, "jobs": []})
        for bad_id in ("4242", "7", "J7", "j", "jx7", "4242:99:C", None):
            with self.assertRaises(c.Refusal, msg=repr(bad_id)):
                c.check_fabric_job("C", env, self.reply(id=bad_id))
        for env_bad in ({}, {"FABRIC_JOB": "bogus"}, {"FABRIC_JOB": "4242:99:B"}):
            with self.assertRaises(c.Refusal, msg=str(env_bad)):
                c.check_fabric_job("C", env_bad, self.reply())

    def test_declaration_must_match(self):
        env = {"FABRIC_JOB": "4242:99:C"}
        for over in ({"state": "waiting"}, {"cls": "timing"}, {"hostmem": 32 * c.GIB}, {"mem": 20 * c.GIB},
                     {"hostmem_lock": False}, {"priority": 2}, {"owner": "x"}, {"max_secs": 100}, {"cards": ["B"]}):
            with self.assertRaises(c.Refusal, msg=str(over)):
                c.check_fabric_job("C", env, self.reply(**over))




if __name__ == "__main__":
    unittest.main()
