#!/usr/bin/env python3
"""B4 reviewed-twin full 8-check G1 hold: CPU-only staging + preflight (no lease, no GPU process, no source edit).

Run on hiptrx from a clean checkout (--repo) of the commit that carries the coordinator. Required parameters, no defaults:
  --commit CPU/GPU checkout commit (40-hex)   --cpu-oracle hiptrx oracle binary   --gpu-oracle hipx oracle binary
  --approval pending|approved                 `pending` records the receipt as NOT exercising the approved-twin parameter
Everything else is the unchanged j587 closure recipe (config via the coordinator's own --write-config, 8 raw argv from
rcpt/b4.g1c.commands.sh + the 3 synthetic plans). Fresh destinations: <rcpt>/b4-reviewed-twin-<commit9>/ (hiptrx) and
/home/kaden/pm-wave/consumer24/b4-reviewed-twin-<commit9>/ (hipx run dir, chain dirs, config, receipt).
Phases: stage config -> structure check -> git/hash/sample-payload/source checks on both hosts -> CPU census with the
existing merged table -> CPU chain prepare (3 codecs) -> CPU final census over all required scopes. The census/prepare
use the coordinator's own Run methods on separate scratch dirs (<stage>/preflight-work, <stage>/preflight-run), so the
real run_dir/chain_dirs/cpu_work_dir in the staged config stay fresh.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

Q = "/home/kaden/qcal/release-0.4.3"
KC = "/home/kaden/pm-wave/consumer24/k2-oracle/key-closure"
KREF = "f582687668f1bfb65171f83bdffcd3a055b43850"
J587 = f"{KC}/run-34c94cb59/tables/intrinsic-merged.c1.r3.json"
J587_CPU = f"{Q}/consumer24/rcpt/b4-hook-closure-34c94cb59/tables/intrinsic-merged.c1.r3.json"

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True)
ap.add_argument("--cpu-oracle", required=True)
ap.add_argument("--gpu-oracle", required=True)
ap.add_argument("--approval", required=True, choices=("pending", "approved"))
ap.add_argument("--repo", default=f"{Q}/consumer24-k2-oracle")
ap.add_argument("--gpu-repo", help="hipx checkout (default: same path as --repo)")
ap.add_argument("--kernel-repo", default=f"{Q}/consumer24-k2-gfx1100")
ap.add_argument("--merged-gpu", default=J587)
ap.add_argument("--merged-cpu", default=J587_CPU)
ap.add_argument("--lease-max-secs", type=int, default=600)
ap.add_argument("--ref-chain-root", default=f"{Q}/consumer24/rcpt/b4-hook-closure-34c94cb59/chain-c1", help="j587 chain root whose A4/H payload .bin bytes must stay identical")
ap.add_argument("--tool-load", help="load the coordinator from this path instead of <repo>/... (must have the same sha256 as the committed/hipx tool)")
a = ap.parse_args()
for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
    os.environ[k] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

TOOL = f"{a.repo}/crates/peacemaker-isa/tools/qwen4_moe_gfx1100_hook_closure.py"
GPU_REPO = a.gpu_repo or a.repo
TOOL_G = f"{GPU_REPO}/crates/peacemaker-isa/tools/qwen4_moe_gfx1100_hook_closure.py"
TOOL_LOAD = a.tool_load or TOOL
spec = importlib.util.spec_from_file_location("hook_closure", TOOL_LOAD)
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)
c9 = a.commit[:9]
D = f"{Q}/consumer24/rcpt/b4-reviewed-twin-{c9}"
HX = f"/home/kaden/pm-wave/consumer24/b4-reviewed-twin-{c9}"
os.makedirs(D, exist_ok=True)
RCPT = f"{D}/preflight.receipt.json"
for p in (f"{D}/config.json", RCPT, f"{D}/preflight-work", f"{D}/preflight-run", f"{D}/work"):
    if os.path.lexists(p):
        sys.exit(f"{p} exists; destinations must be fresh")


class Local(hc.Remote):
    """CPU host is this host: same interface as Remote, no ssh."""

    def __init__(self):
        super().__init__("hiptrx", lambda: 86400)

    def sh(self, cmd, timeout=None, stdin=None, check=True):
        p = subprocess.run(["bash", "-c", cmd], stdin=stdin if stdin is not None else subprocess.DEVNULL, capture_output=True, timeout=timeout)
        if check and p.returncode != 0:
            hc.die(f"local `{cmd[:200]}` rc={p.returncode}: {p.stderr.decode(errors='replace')[-600:]}")
        return p

    def run_cpu(self, label, argv, logs, cwd, timeout):
        os.makedirs(logs, exist_ok=True)
        with open(f"{logs}/{label}.stdout", "wb") as so, open(f"{logs}/{label}.stderr", "wb") as se:
            p = subprocess.run(hc.CPU_ENV + ["timeout", "-k", "30", str(int(timeout))] + argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=so, stderr=se)
        return p.returncode, ""


gpu, cpu = hc.Remote("hipx", lambda: 86400), Local()
rec = {"schema": "consumer24.b4.reviewed-twin-preflight.v1", "status": "in_progress", "gpu_work": False,
       "approval": a.approval, "approved_twin_parameter_exercised": a.approval == "approved",
       "params": {"commit": a.commit, "cpu_oracle": a.cpu_oracle, "gpu_oracle": a.gpu_oracle, "cpu_repo": a.repo, "gpu_repo": GPU_REPO},
       "stage_dir": D, "hipx_dir": HX, "started_utc": time.strftime("%FT%TZ", time.gmtime()), "checks": {}}
failed = []


def chk(name, fn):
    try:
        v = fn()
        rec["checks"][name] = {"ok": True, "detail": v}
    except Exception as e:  # recorded, not hidden
        rec["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        failed.append(name)


def save():
    hc.atomic_write(RCPT, hc.dumps(rec))


def req(cond, msg):
    if not cond:
        hc.die(msg)


# ---- stage + structure check (the coordinator's own generator/validator)
args = ["python3", TOOL_LOAD, "--write-config", f"{D}/config.json", "--commit", a.commit, "--cpu-repo", a.repo, "--gpu-repo", GPU_REPO,
        "--cpu-oracle", a.cpu_oracle, "--gpu-oracle", a.gpu_oracle, "--gpu-observer", f"{KC}/gfx1100_intrinsic_keys_observe",
        "--gpu-probe-co", f"{KC}/gfx1100_intrinsic_keys_probe.co", "--gpu-probe-manifest", f"{KC}/gfx1100_intrinsic_keys_probe.manifest.json",
        "--run-dir", f"{HX}/run", "--cpu-work-dir", f"{D}/work", "--lease-max-secs", str(a.lease_max_secs),
        "--cpu-base-probe", f"{Q}/consumer24/artifacts/b4-probe/observations.json",
        "--cpu-divide-table", f"{Q}/consumer24/artifacts/b4-probe/divide-observations.json",
        "--cpu-fixture", f"{Q}/consumer24/rcpt/b4-fixture/qwen4_moe_epilogue_fixture_pm_gfx1100.hxaco"]
p = subprocess.run(args, capture_output=True, text=True)
rec["write_config"] = {"argv": args, "rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[-400:]}
if p.returncode:
    rec["status"] = "FAIL"
    save()
    sys.exit(f"write-config failed: {p.stderr}")
raw = open(f"{D}/config.json", "rb").read()
cfg = hc.loads(raw, "config")
plan = hc.validate_config(cfg)
rec["config"] = {"path": f"{D}/config.json", "sha256": hc.sha256_bytes(raw), "labels": sorted(plan), "lease_max_secs": cfg["lease_max_secs"]}
rec["tool"] = {"path": TOOL, "loaded_from": TOOL_LOAD, "sha256": hc.sha256_file(TOOL_LOAD)}
rec["submit"] = (f"/home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card xtx --mem-gb 4 --hostmem-gb 1 --priority P0 "
                 f"--owner <owner> --max-secs 600 --wait-secs 43200 -- python3 {TOOL_G} --config {HX}/config.json --out {HX}/closure.receipt.json")
save()

# ---- both-host git + script
def git_side():
    out = {}
    for side, sh, rp, tl in (("cpu", cpu, a.repo, TOOL), ("gpu", gpu, GPU_REPO, TOOL_G)):
        g = lambda *x: sh.sh("git -C %s %s" % (hc.shlex.quote(rp), hc.q(list(x)))).stdout.decode().strip()
        head, dirty = g("rev-parse", "HEAD"), g("status", "--porcelain", "--untracked-files=normal")
        req(head == a.commit and not dirty, f"{side}: HEAD={head} dirty={dirty[:200]!r}")
        g("ls-files", "--error-unmatch", os.path.relpath(tl, rp))
        out[side] = {"head": head, "clean": True}
    gs = gpu.sha256(TOOL_G)[TOOL_G]
    req(gs == rec["tool"]["sha256"], f"coordinator sha256 differs on hipx: {gs}")
    out["tool_sha256_both"] = gs
    return out


chk("git_both_hosts_clean_at_commit", git_side)

# ---- hashes
Gh, Ch = {}, {}


def artifact_hashes():
    g1 = plan
    (fixture,) = {hc.one(hc.parse_pairs(x)[1], "--epilogue-codeobject", l) for l, x in g1.items()}
    gf = {"gpu_oracle": cfg["gpu_oracle"], "gpu_observer": cfg["gpu_observer"], "gpu_probe_co": cfg["gpu_probe_co"],
          "gpu_probe_manifest": cfg["gpu_probe_manifest"], "base_probe": cfg["base_probe"], "divide_table": cfg["divide_table"], "fixture": fixture}
    gh = gpu.sha256(*gf.values())
    Gh.update({k: gh[v] for k, v in gf.items()})
    cf = {"cpu_oracle": cfg["cpu_oracle"], "cpu_base_probe": cfg["cpu_base_probe"], "cpu_divide_table": cfg["cpu_divide_table"],
          "cpu_fixture": cfg["cpu_fixture"], "cpu_chain_helper": cfg["cpu_chain_helper"],
          "cpu_rotation_source": os.path.join(os.path.dirname(cfg["cpu_chain_helper"]), "mq_i4_a4_oracle.py")}
    for cod, d in cfg["cpu_samples"].items():
        cf[f"cpu_sample_manifest_{cod}"] = os.path.join(d, "sample.json")
    Ch.update({k: hc.sha256_file(v) for k, v in cf.items()})
    for x, y in (("gpu_oracle", "cpu_oracle"), ("base_probe", "cpu_base_probe"), ("divide_table", "cpu_divide_table"), ("fixture", "cpu_fixture")):
        req(Gh[x] == Ch[y], f"{x} {Gh[x]} != {y} {Ch[y]}")
    for rel, key in ((os.path.relpath(cfg["cpu_chain_helper"], a.repo), "cpu_chain_helper"), (hc.ROTATION_REL, "cpu_rotation_source")):
        blob = cpu.sh(f"git -C {a.repo} show {a.commit}:{rel} | sha256sum").stdout.decode().split()[0]
        req(blob == Ch[key], f"{key} differs from Git blob")
    return {"gpu": Gh, "cpu": Ch}


chk("artifact_hashes_equal_across_hosts", artifact_hashes)
rec["binary_md5"] = {}
chk("binary_md5", lambda: rec["binary_md5"].update({"cpu": cpu.sh(f"md5sum {a.cpu_oracle}").stdout.decode().split()[0],
                                                     "gpu": gpu.sh(f"md5sum {a.gpu_oracle}").stdout.decode().split()[0]}) or rec["binary_md5"])


def g1_inputs():
    paths, flags = [], {}
    for label, argv in plan.items():
        for f, v in hc.parse_pairs(argv)[1]:
            if f in ("--codeobject", "--mq4-anchor", "--epilogue-codeobject"):
                paths.append(v)
            elif f in ("--mixed-codeobject", "--negative-codeobject"):
                paths.append(v.split("=", 1)[1])
    paths = sorted(set(paths))
    h = gpu.sha256(*paths)
    return h


chk("g1_module_fixture_hashes_hipx", lambda: rec.setdefault("g1_inputs", g1_inputs()) and {"count": len(rec["g1_inputs"])})


def kernels_vs_ref():
    out = {}
    for name in ("qwen4_moe_iu4_sym_pm_gfx1100.hxaco", "qwen4_moe_iu4_sym_w2_pm_gfx1100.hxaco", "qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco"):
        want = hc.sha256_bytes(subprocess.run(["git", "-C", a.kernel_repo, "show", f"{KREF}:kernels/{name}"], capture_output=True, check=True).stdout)
        got = rec["g1_inputs"][f"/home/kaden/pm-wave/consumer24/b4/kernels/{name}"]
        req(want == got, f"{name}: hipx {got} != {KREF[:9]} blob {want}")
        out[name] = got
    return out


chk("candidate_kernels_equal_ref_blobs", kernels_vs_ref)


def samples():
    out = {}
    for label, argv in plan.items():
        _, _, scodec, pairs = hc.classify(argv)
        if not scodec:
            continue
        d = hc.one(pairs, "--sample-dir", label)
        gs = gpu.sha256(f"{d}/sample.json")[f"{d}/sample.json"]
        req(gs == Ch[f"cpu_sample_manifest_{scodec}"], f"{label}: hipx sample.json {gs} != cpu manifest")
        doc = json.loads(gpu.cat(f"{d}/sample.json"))
        files = [(t["file"], t["sha256"]) for s in doc["samples"] for t in s["tensors"]]
        got = gpu.sha256(*[f"{d}/{f}" for f, _ in files])
        bad = [f for f, h in files if got[f"{d}/{f}"] != h]
        req(not bad and len(files) == 128, f"{label}: payload mismatches {bad[:3]} of {len(files)}")
        out[d] = {"sample_json_sha256": gs, "payload_files_verified": len(files), "mismatch": 0, "provisional": doc.get("provisional"),
                  "artifact_sha256": doc.get("artifact_sha256")}
    return out


chk("sample_payloads_128_both_sets_hipx", samples)


def bindings():
    out = {}
    for name, p in (("c24_bound_manifest", f"{Q}/consumer24/artifacts/c24-samples/sample.bound.json"),
                    ("c24_b1_sample_binding", f"{Q}/consumer24/rcpt/b1.sample-binding.json"),
                    ("w3_b4_sample_binding", f"{Q}/consumer24/rcpt/b4.w3-sample-binding.json")):
        d = json.load(open(p))
        out[name] = {"path": p, "sha256": hc.sha256_file(p), **{k: v for k, v in d.items() if isinstance(v, (bool, int, str)) and len(str(v)) < 120}}
    b = json.load(open(f"{Q}/consumer24/artifacts/c24-samples/sample.bound.json"))
    s = json.load(open(f"{Q}/consumer24/artifacts/c24-samples/sample.json"))
    pb = [(t["file"], t["sha256"]) for x in b["samples"] for t in x["tensors"]]
    ps = [(t["file"], t["sha256"]) for x in s["samples"] for t in x["tensors"]]
    req(pb == ps and len(pb) == 128, "c24 bound manifest payload list != provisional manifest")
    out["c24_bound_equals_provisional_payloads"] = len(pb)
    return out


chk("sample_binding_records", bindings)


def fresh():
    for pth in [cfg["run_dir"], *cfg["chain_dirs"].values()]:
        req(not gpu.exists(pth), f"{pth} exists on hipx")
    req(not os.path.lexists(cfg["cpu_work_dir"]), "cpu_work_dir exists")
    req(not gpu.exists(f"{HX}/config.json") and not gpu.exists(f"{HX}/closure.receipt.json"), "hipx config/receipt path exists")
    return {"run_dir": cfg["run_dir"], "chain_dirs": cfg["chain_dirs"], "cpu_work_dir": cfg["cpu_work_dir"], "hipx_receipt": f"{HX}/closure.receipt.json"}


chk("destinations_fresh", fresh)


def pyenv():
    py = cpu.sh(hc.q(hc.CPU_ENV[:4] + [cfg["cpu_python"], "-c", "import sys,numpy,torch;print(sys.version.split()[0],numpy.__version__,torch.__version__,torch.cuda.is_available())"])).stdout.decode().split()
    req(len(py) == 4 and py[3] == "False", f"cpu python {py}")
    env = {h: sh.sh("env | grep -ci CU_MASK || true").stdout.decode().strip() for h, sh in (("cpu", cpu), ("gpu", gpu))}
    req(env == {"cpu": "0", "gpu": "0"}, f"CU_MASK env present {env}")
    return {"python": py, "cu_mask_env_vars": env}


chk("cpu_python_no_gpu_and_no_cu_mask", pyenv)

# ---- merged table: same bytes on both hosts, then CPU census / chain / final census
mg = {}


def merged():
    gs = gpu.sha256(a.merged_gpu)[a.merged_gpu]
    cs = hc.sha256_file(a.merged_cpu)
    req(gs == cs, f"merged table differs across hosts {gs} {cs}")
    doc = json.load(open(a.merged_cpu))
    mg.update({"sha256": cs, "counts": doc.get("counts")})
    return {"gpu_path": a.merged_gpu, "cpu_path": a.merged_cpu, **mg}


chk("merged_table_same_both_hosts", merged)
if not [f for f in failed if f != "git_both_hosts_clean_at_commit"]:   # git failure is recorded and makes the receipt FAIL; the other CPU phases still run
    L = dict(cfg, run_dir=f"{D}/preflight-run", cpu_work_dir=f"{D}/preflight-work", lease_max_secs=3600)
    run = hc.Run(L, raw, f"{D}/preflight.run.json")
    run.cpu, run.plan = cpu, plan
    run.base_sha, run.div_sha, run.bin_sha, run.fixture_sha = Gh["base_probe"], Gh["divide_table"], Gh["gpu_oracle"], Gh["fixture"]
    run.rec["artifacts"] = {"gpu": Gh, "cpu": Ch}
    for sub in ("", "census", "tables", "logs"):
        os.makedirs(f"{D}/preflight-run/{sub}")
    for sub in ("", "tables", "logs", "census"):
        os.makedirs(f"{D}/preflight-work/{sub}")
    run.cpu_merged = f"{D}/preflight-work/tables/intrinsic-merged.json"
    cpu.sh(f"cp -- {a.merged_cpu} {run.cpu_merged}")
    run.merged_path, run.merged_sha = a.merged_gpu, mg["sha256"]
    ref = json.load(open(f"{Q}/consumer24/rcpt/b4.closure.34c94cb59.receipt.json"))
    j587_scopes = ref["chain_cycles"][0]["final_census"]["scopes"]

    def census_a():
        v = run.census("pf.merged", None)
        s = v["summary"]
        req(s["missing_count"] == 0 and v["doc"].get("hooks_closed") is True, f"merged table incomplete: missing {s['missing_count']} first {s['first_missing']}")
        return {k: s[k] for k in ("complete", "missing_count", "full_keys", "incomplete_reasons")} | {"sha256": run.rec["rounds"][-1]["sha256"]}

    chk("census_merged_table_missing_0", census_a)

    def chains_and_final():
        root = run.prepare_chains(1)
        v = run.census("pf.final", root)
        ok, why = run.final_census_ok(v)
        cyc = run.rec["chain_cycles"][0]
        sc = {n: {"cases": s.get("cases"), "complete": s.get("complete"), "missing_count": s.get("missing_count")} for n, s in v["scopes"].items()}
        rec["final_census"] = {"ok": ok, **v["summary"], **why, "scopes": sc, "j587_full_keys_for_comparison": ref["rounds"][-1]["full_keys"],
                               "scopes_identical_to_j587": sc == j587_scopes}
        rec["chains"] = {c: {k: e[k] for k in ("chain_h_sha256", "chain_a4_sha256", "entries", "checks", "source_binding")} for c, e in cyc["codecs"].items()}
        pay = {}
        for cod in hc.CODECS:
            new, old = cpu.tree_hashes(f"{root}/{cod}"), cpu.tree_hashes(f"{a.ref_chain_root}/{cod}")
            data = lambda t: {k: h for k, h in t.items() if k.endswith(".bin")}
            pay[cod] = {"payload_files": len(data(new)), "payload_bytes_equal_ref": data(new) == data(old),
                        "manifest_files_changed": sorted(k for k in new if not k.endswith(".bin") and new[k] != old.get(k))}
        rec["chains_payload_vs_ref"] = {"ref_chain_root": a.ref_chain_root, **pay}
        req(all(v["payload_bytes_equal_ref"] and v["payload_files"] > 0 for v in pay.values()), f"A4/H payload bytes differ from reference chain: {pay}")
        req(ok, f"final census not complete: {why}")
        # all 8 G1 labels covered by a complete scope
        cover = {}
        for pre, d in run.expected_scopes():
            hit = [n for n in v["scopes"] if (n == pre if d is None else n.startswith(pre + "/") and n.endswith(d.rstrip("/")))]
            cover[f"{pre}/{d}" if d else pre] = hit
        need = {"synthetic-mq4": ["synthetic/mq4"], "synthetic-w2": ["synthetic/w2"], "synthetic-w3": ["synthetic/w3"], "real-mq4": ["real/mq4"],
                "real-w2": ["real/w2"], "real-w3": ["real/w3"], "real-mixed-mq4-w2": ["mixed/w2"], "real-mixed-mq4-w3": ["mixed/w3"]}
        lab = {l: [n for n in v["scopes"] if any(n == x or n.startswith(x + "/") for x in xs)] for l, xs in need.items()}
        req(all(lab.values()), f"labels without census scope: {[l for l, x in lab.items() if not x]}")
        rec["label_scope_coverage"] = lab
        return {"ok": ok, "labels": sorted(lab), "scopes": len(sc)}

    chk("cpu_chain_prepare_and_final_census_cover_all_8", chains_and_final)

rec["status"] = "FAIL" if failed else "PASS"
rec["failed_checks"] = failed
rec["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
save()
print(json.dumps({"receipt": RCPT, "sha256": hc.sha256_file(RCPT), "status": rec["status"], "failed": failed, "config_sha256": rec["config"]["sha256"],
                  "approved_twin_parameter_exercised": rec["approved_twin_parameter_exercised"]}))
sys.exit(1 if failed else 0)
