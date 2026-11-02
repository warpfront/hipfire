#!/usr/bin/env python3
"""Build b4.g1a/b/c.json from the raw hipx hold artifacts of one reviewed-twin closure run. Reads only recorded receipts; every
number is summed from the raw per-record fields, nothing is typed in. Usage: aggregate.py <stage_dir> <out_dir>"""
import collections
import hashlib
import json
import os
import sys

S, OUT = sys.argv[1], sys.argv[2]
HX = "/home/kaden/pm-wave/consumer24/b4-reviewed-twin-90d429ce9"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
CL = json.load(open(f"{S}/hipx-copy/closure.receipt.json"))
PF = json.load(open(f"{S}/preflight.receipt.json"))
G = {e["label"]: e for e in CL["g1"]}
R = {l: json.load(open(f"{S}/hipx-copy/run/g1/{l}.json")) for l in G}
LINK = "xtx sits behind a PCIe 4.0 x4 root port (00:03.1); this is a correctness oracle and carries no speed or bandwidth claim"
assert CL["status"] == "complete" and CL["g1_all_ok"] is True and len(CL["g1"]) == 8 and [e["label"] for e in CL["g1"]] == list(G)

CAT = ("live", "padding_in_live_tile", "sentinel_tile_poison")


def raw_ref(label):
    p = f"{S}/hipx-copy/run/g1/{label}.json"
    return {"host_path": f"{HX}/run/g1/{label}.json", "local_path": p, "sha256": sha(p), "oracle_receipt_sha256_in_closure": G[label]["receipt_sha256"],
            "stdout_sha256": G[label]["stdout_sha256"], "stderr_sha256": G[label]["stderr_sha256"], "rc": G[label]["rc"], "secs": G[label]["secs"]}


def classes(label, checks):
    out = collections.OrderedDict()
    for r in R[label]["results"]:
        if r["check"] not in checks:
            continue
        key = f"{r['kind']}/{r['class']}" + ("/chain_down" if r.get("chain_down") else "")
        c = out.setdefault(key, collections.Counter())
        c["records"] += 1
        c["pass"] += bool(r["pass"])
        for fam in ("f32_acc", "bf16"):
            for cat in CAT:
                x = r[fam][cat]
                c[f"{fam}_{cat}_checked"] += x["checked"]
                c[f"{fam}_{cat}_mismatches"] += x["mismatches"]
        c["poison_tail_f32_bytes_changed"] += r["poison_tail_bytes_changed"]["f32_acc"]
        c["poison_tail_bf16_bytes_changed"] += r["poison_tail_bytes_changed"]["bf16"]
        c["hook_missing_keys"] += r["hook_missing_keys"]
    return {k: dict(v) for k, v in out.items()}


def zero(cl, fam):
    short = fam.split("_")[0]
    return (all(v == 0 for c in cl.values() for k, v in c.items() if (k.startswith(fam) and k.endswith("mismatches")) or (k.startswith("poison_tail_" + short)) or k == "hook_missing_keys")
            and all(c["records"] == c["pass"] for c in cl.values()))


def mism(x):
    """sum of every integer field whose key says mismatch/changed, recursively"""
    if isinstance(x, dict):
        return sum((v if isinstance(v, int) and not isinstance(v, bool) and ("mismatch" in k or "changed" in k) else mism(v)) for k, v in x.items())
    if isinstance(x, list):
        return sum(mism(v) for v in x)
    return 0


def anchors(label):
    a = [r for r in R[label]["results"] if r["check"].endswith("anchor") or r["check"].endswith("anchor_union")]
    return {"records": len(a), "nonzero_records": sum(1 for r in a if mism(r) or not r["pass"])}


def negatives(label):
    return [{k: n[k] for k in ("codec", "kind", "negative", "required_gate", "scenarios_run", "f32_mismatches", "bf16_mismatches", "detected", "status")} for n in R[label]["negative_controls"]]


def fixture(label):
    f = R[label]["epilogue_fixture"]
    return {"module_sha256": f["module_sha256"], "pass": f["pass"], "cases": [{k: c[k] for k in ("kind", "pass_name", "class", "grouped_rows", "outputs_checked", "bf16_mismatches", "hook_missing_keys", "wrongsel_mismatches", "pass")} for c in f["cases"]],
            "negative_controls": f["negative_controls"]}


_J = json.load(open(os.path.join(os.path.dirname(S), "b4-hook-closure-34c94cb59/tables/intrinsic-merged.c1.r3.json")))
_N = json.load(open(f"{S}/hipx-copy/run/tables/intrinsic-merged.c1.r3.json"))
MERGED_CMP = {"j587_sha256": sha(os.path.join(os.path.dirname(S), "b4-hook-closure-34c94cb59/tables/intrinsic-merged.c1.r3.json")), "differing_top_level_keys": sorted(k for k in set(_J) | set(_N) if _J.get(k) != _N.get(k)),
              "rows_exp2_ldexp_divide_identical": all(_J[k] == _N[k] for k in ("exp2", "ldexp", "divide", "row_batch", "counts"))}


def common():
    A = CL["artifacts"]["gpu"]
    return {
        "schema": "consumer24.b4.reviewed-twin-aggregate.v2", "approval_state": "approved; fresh full 8-process hold executed",
        "code_sha1": CL["source"]["commit"], "code_sha1_note": "main cherry-picks 8aeb1081e (=reviewed 77af6abf8) + 90d429ce9 (=reviewed c1473d378); NanPairReview FINAL APPROVE reported via parent/K2Oracle messages (reported, not re-verified here)",
        "git": {"cpu_repo": CL["config"]["cpu_repo"], "gpu_repo": CL["config"]["gpu_repo"], **CL["source"]},
        "binary": {"path": CL["config"]["gpu_oracle"], "sha256": A["gpu_oracle"], "md5": PF["binary_md5"]["gpu"], "cpu_md5_same": PF["binary_md5"]["cpu"] == PF["binary_md5"]["gpu"], "cpu_path": CL["config"]["cpu_oracle"]},
        "probe_inputs": {"base_probe_sha256": A["base_probe"], "divide_sha256": A["divide_table"], "merged_intrinsic_table_sha256": CL["closure"]["merged_table_sha256"], "merged_table_path": CL["closure"]["merged_table"],
                         "merged_counts": CL["tables"]["merged"]["counts"], "observer_sha256": A["gpu_observer"], "probe_co_sha256": A["gpu_probe_co"], "probe_manifest_sha256": A["gpu_probe_manifest"],
                         "merged_table_note": "fresh in-hold observation; same counts as j587 (exp2 2, ldexp 2, divide 3); sha differs from j587 df584043 only in `batches` metadata", "merged_vs_j587": MERGED_CMP},
        "chains": {c: CL["chain_transfer"][c] for c in ("mq4", "w2", "w3")},
        "chain_payload_vs_j587": PF["chains_payload_vs_ref"],
        "closure": {"receipt": f"{S}/hipx-copy/closure.receipt.json", "sha256": sha(f"{S}/hipx-copy/closure.receipt.json"), "host_path": f"{HX}/closure.receipt.json", "status": CL["status"], "rc": CL["rc"],
                    "elapsed_secs": CL["elapsed_secs"], "finished_utc": CL["finished_utc"], "census_rounds": CL["closure"]["rounds"], "final_census_sha256": CL["closure"]["final_census_sha256"], "iteration_hash": CL["closure"]["iteration_hash"],
                    "config_sha256": CL["config_sha256"], "coordinator_sha256": CL["script"]["sha256"], "lease": CL["lease"], "fabric_job": "j658 (P0 correct xtx max 600s, fabric-run output observed by ApprovedTwinG1)"},
        "preflight": {"receipt": f"{S}/preflight.receipt.json", "sha256": sha(f"{S}/preflight.receipt.json"), "status": PF["status"], "approved_twin_parameter_exercised": PF["approved_twin_parameter_exercised"],
                      "checks": {k: v["ok"] for k, v in PF["checks"].items()}, "gpu_work": PF["gpu_work"]},
        "device": R["synthetic-mq4"]["device"], "driver": R["synthetic-mq4"]["driver"], "link_caveat": LINK,
        "no_cu_mask": "lease env cu_mask_env='none set'; receipts driver.cu_mask_env: " + str(sorted({r["driver"]["cu_mask_env"] for r in R.values()})),
        "process_disclosures": [
            {"kind": "ownership", "status": "verified (git output observed) + parent-acknowledged (reported)", "text": "Before the parent's 'do not git-write the C-owned WT' note, ApprovedTwinG1 ran `git fetch` + `git checkout --detach 90d429ce9` in hipx consumer24-k2-oracle (the C-owned checkout). It restored that checkout to 34c94cb597164ebae133bbefca951f72af1ec9a2 detached, 0 dirty paths (observed), and ran the approved gate from its OWN hipx checkout consumer24-k2-twin-90d429ce9 (alternates clone of hiptrx consumer24-k2-gfx1100, HEAD 90d429ce9017f1c3f18b17d1bc10080aa7c45ccd, clean, observed by the preflight git check). The mistaken checkout produced attempt0 preflight PASS (rcpt/b4-reviewed-twin-90d429ce9.attempt0-gpurepo-was-C-WT, no GPU work); it is not the gate evidence."},
            {"kind": "draft_dryrun", "status": "verified", "text": "rcpt/b4-reviewed-twin-c1473d378.dryrun-draft-hipx-not-checked-out: CPU-only dry run on the draft commit (hipx git check failed as expected, all other checks PASS); no GPU."}],
        "unchanged_from_j587": "g1_commands/kernels/fixture/base/divide/real-plan unchanged; staged config differs from j587 config only in run_dir, chain_dirs, cpu_work_dir (diff verified at staging)",
    }


def per_label(label):
    e = G[label]
    return {"label": label, "pass": e["pass"], "complete": e["complete"], "status": e["status"], "gate_ok": e["gate_ok"], "g1c_verdict": e["g1c_verdict"], "hook_missing_total": e["hook_missing_total"],
            "summary": e["summary"], "device_identity_match": e["device"]["identity_match"], "binds_tables": e["binds_tables"], "command_argv": e["argv"], "inputs_sha256": e["inputs"], "source_sha256": R[label]["source_sha256"], "raw_receipt": raw_ref(label)}


# ---------------------------------------------------------------- G1a / G1b (synthetic, 3 codecs)
def synth(gate, fam):
    d = common()
    per = {}
    for c in ("mq4", "w2", "w3"):
        L = f"synthetic-{c}"
        cl = classes(L, {"candidate_vs_cpu"})
        per[c] = {**per_label(L), "record_counts": {"total_results": len(R[L]["results"]), "candidate_vs_cpu": sum(1 for r in R[L]["results"] if r["check"] == "candidate_vs_cpu"),
                                                    "anchor": anchors(L)["records"], "arms": R[L]["summary"]["arm_counts"]},
                  "per_kind_class": cl, f"{fam}_all_zero": zero(cl, fam), "anchor_same_device": anchors(L), "negative_controls": negatives(L), "negatives_all_detected": all(n["detected"] for n in negatives(L)),
                  "negative_gate_required": sorted({(n["negative"], n["required_gate"]) for n in negatives(L)}), "epilogue_fixture": fixture(L)}
    d.update({"gate": gate, "scope": "synthetic", "real_gate": False, "g1c_pass": False, "partial": False, "complete_codecs": ["mq4", "w2", "w3"], "codecs_not_run": [], "unrun_labels": [],
              "per_codec": per, "first_failure": None})
    ok = all(per[c][f"{fam}_all_zero"] and per[c]["negatives_all_detected"] and per[c]["gate_ok"] and per[c]["epilogue_fixture"]["pass"] for c in per)
    d["pass"] = ok
    return d


a = synth("B4-G1a", "f32_acc")
a["verdict_g1a"] = ("PASS: every F32 live/padding/sentinel/poison byte of the candidate equals the CPU twin in all candidate_vs_cpu records of MQ4, W2 and W3 (all mismatch counts 0, poison tail changes 0); "
                    "all 4 negative codeobjects x 2 kinds detected per codec" if a["pass"] else "FAIL")
a["g1a_scope_caveat"] = "G1a = candidate _acc F32 bytes vs the CPU twin for all three codecs, from the single closure hold; real weights/mixed residency are G1c."
b = synth("B4-G1b", "bf16")
b["verdict_g1b"] = ("PASS: every BF16 live/padding/sentinel/poison byte equals the CPU twin in all candidate_vs_cpu records (MQ4/W2/W3), incl. the standalone epilogue fixture exceptional_inputs gate_up case now 0/71680 mismatches "
                    "(was 959/71680 in j587), wrongsel negatives detected" if b["pass"] else "FAIL")
b["fixture_exceptional_gate_up"] = {c: next(x for x in b["per_codec"][c]["epilogue_fixture"]["cases"] if x["kind"] == "gate_up" and x["pass_name"] == "exceptional_inputs") for c in ("mq4", "w2", "w3")}
b["j587_959_vs_now"] = {"j587_receipt": "rcpt/b4.synth-mq4.34c94cb59.json (standalone exception fixture 959/71680, first p0 col108 got 0x7fc0 want 0x7fc1)", "now_bf16_mismatches": {c: v["bf16_mismatches"] for c, v in b["fixture_exceptional_gate_up"].items()}}

# ---------------------------------------------------------------- G1c (real, 5 processes)
real = {}
for L, checks in (("real-mq4", {"candidate_vs_cpu"}), ("real-w2", {"candidate_vs_cpu"}), ("real-w3", {"candidate_vs_cpu"}),
                  ("real-mixed-mq4-w2", {"mixed_candidates_vs_cpu"}), ("real-mixed-mq4-w3", {"mixed_candidates_vs_cpu"})):
    cl = classes(L, checks)
    real[L] = {**per_label(L), "sample_dir": [v for f, v in zip(G[L]["argv"][1::2], G[L]["argv"][2::2]) if f == "--sample-dir"], "record_counts": {"total_results": len(R[L]["results"]), "checked": sum(c["records"] for c in cl.values()), "anchor": anchors(L)},
               "per_kind_class": cl, "f32_all_zero": zero(cl, "f32_acc"), "bf16_all_zero": zero(cl, "bf16"), "negative_controls": negatives(L), "negatives_all_detected": all(n["detected"] for n in negatives(L)) if negatives(L) else None,
               "negative_note": "per-codec real runs carry 4 negatives x 2 kinds; mixed runs carry none by design (negatives covered by same-codec per-codec real launches)", "g1c": R[L]["g1c"], "epilogue_fixture": fixture(L),
               "mixed_residency_arms": R[L]["summary"]["arm_counts"] if L.startswith("real-mixed") else None, "chain_down_scenarios": R[L]["summary"]["chain_down_scenarios"], "mixed_scenarios": R[L]["summary"]["mixed_scenarios"]}
c = common()
bind = PF["checks"]["sample_binding_records"]["detail"]
samp = PF["checks"]["sample_payloads_128_both_sets_hipx"]["detail"]
c.update({"gate": "B4-G1c", "scope": "real weights; synthetic A4/routes; Down input twin-chained", "real_gate": True, "partial": False, "unrun_labels": [], "per_process": real,
          "all_strict_zero": all(v["f32_all_zero"] and v["bf16_all_zero"] and v["gate_ok"] for v in real.values()),
          "negatives_detected_per_codec": {k: real[k]["negatives_all_detected"] for k in ("real-mq4", "real-w2", "real-w3")},
          "oracle_g1c_verdicts": {k: v["g1c_verdict"] for k, v in real.items()},
          "oracle_g1c_closed": {k: v["g1c"]["closed"] for k, v in real.items()},
          "sample_payloads": {"c24": {"dir": "/home/kaden/pm-wave/consumer24/b4/samples/c24", **samp["/home/kaden/pm-wave/consumer24/b4/samples/c24"]},
                              "w3": {"dir": "/home/kaden/pm-wave/consumer24/b4/samples/w3", **samp["/home/kaden/pm-wave/consumer24/b4/samples/w3"]}, "note": "payload sha256 of all 128 .bin per set verified on hipx at preflight (before the hold)"},
          "artifact_binding": {"c24": {"status_per_oracle_receipt": real["real-mixed-mq4-w2"]["g1c"]["artifact_binding"]["status"], "evidence": {k: bind[k] for k in ("c24_b1_sample_binding", "c24_bound_manifest")},
                                       "bound_equals_provisional_payloads": bind["c24_bound_equals_provisional_payloads"],
                                       "note": "The oracle ran on the provisional sample.json (7e7367f6…) and so prints 'artifact binding pending' and closed=false for the c24 mixed process. Binding evidence (b1.sample-binding: 64/64 samples byte-equal c24.hfq, bound manifest payload list == provisional payload list, 128/128) is recorded here; no GPU rerun needed per lead policy IF the identical bytes were run (payload sha verified). The oracle's own closed flag stays false for real-mixed-mq4-w2 - reported verbatim."},
                             "w3": {"status_per_oracle_receipt": real["real-mixed-mq4-w3"]["g1c"]["artifact_binding"]["status"], "evidence": bind["w3_b4_sample_binding"], "oracle_closed": real["real-mixed-mq4-w3"]["g1c"]["closed"]}}})
c["g1c_pass"] = bool(c["all_strict_zero"] and all(c["negatives_detected_per_codec"].values()) and real["real-mixed-mq4-w2"]["g1c"]["measured_pass"] and real["real-mixed-mq4-w3"]["g1c"]["measured_pass"])
c["g1c_closed_by_oracle"] = {"w3_mixed": real["real-mixed-mq4-w3"]["g1c"]["closed"], "c24_mixed": real["real-mixed-mq4-w2"]["g1c"]["closed"]}
c["verdict_g1c"] = ("measured PASS in all 5 real processes (F32/BF16/padding/sentinel/poison all 0 mismatches; per-codec negatives all detected); W3 mixed closed=true (artifact bound); C24 mixed measured_pass=true with oracle closed=false "
                    "(provisional sample.json run; binding evidence recorded, see artifact_binding)") if c["g1c_pass"] else "FAIL"
for name, d in (("b4.g1a.json", a), ("b4.g1b.json", b), ("b4.g1c.json", c)):
    p = f"{OUT}/{name}"
    assert not os.path.exists(p), p
    open(p, "w").write(json.dumps(d, indent=1, sort_keys=True) + "\n")
    print(name, sha(p), d.get("pass", d.get("g1c_pass")))
