#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-hold coordinator for the gfx1100 E8P K2 hook-key closure and its seven G1 gates (the E8P sibling of
`qwen4_moe_gfx1100_hook_closure.py`, which stays byte-for-byte unchanged: the j658 provenance pins it).

It imports that tool as a module and reuses its infrastructure unchanged (remote execution, observer launch and strict
validation, conflict-checked merge, census reader, G1 receipt reader, fabric/lease/Git preflight). What differs is the matrix:

  G1 matrix (7 commands, stop on the FIRST failed gate):
    synthetic-e8p                     --codec e8p   (6 negatives: wrong-lut, wrong-sign-mask, reversed-epoch, wrong-fold-epoch,
                                                    wrong-scale-row, wrong-expert-pointer) + standalone epilogue fixture
    for each artifact variant V in c24, c24h6306 (sample dir basenames `V-e8p`, real E8P cold + MQ4 hot experts, sha-bound):
      real-mq4-V                      --codec mq4 on the MQ4 hot samples of that artifact (4 MQ4 negatives)
      real-e8p-V                      --codec e8p on the E8P cold samples (6 negatives), Down input CPU-twin-chained
      real-mixed-mq4-e8p-V            G1c: --mixed-codeobject mq4= and e8p=, one tile list / output, Down twin-chained
  The E8P kernels have no MQ4 anchor (|v| up to 11): the CPU twin (frozen K-E0 arithmetic + the reviewed MulUp BF16 adapter)
  is the only expectation, and `--mq4-anchor` is refused by the oracle for e8p.

Chains: per variant two chain dirs, `V-e8p` (`--prepare-chain --codec e8p`, which also prepares the mixed entries) and `V-mq4`.
Census: `--census-sample-dir e8p=DIR` once per variant (synthetic mq4 + e8p, real mq4/e8p, mixed e8p, fixture), chains `e8p=`/`mq4=`.

Usage (same contract as the base tool: run INSIDE a fabric lease the caller holds, on hipx):
  fabric-run.sh --class correct --card xtx --mem-gb 4 --max-secs <lease_max_secs> --priority P0 --owner <lead> -- \\
      python3 crates/peacemaker-isa/tools/qwen4_moe_gfx1100_e8p_closure.py --config CFG.json --out RECEIPT.json
  python3 .../qwen4_moe_gfx1100_e8p_closure.py --check-config CFG.json                 # CPU-only structure check
  python3 .../qwen4_moe_gfx1100_e8p_closure.py --write-config CFG.json --commit SHA ...   # CPU-only config generator (see --help)

Deployment: config key `gpu_repo` is the `crates/peacemaker-isa/tools` directory of a Git-fetched CLEAN worktree of `commit` on hipx
(e.g. /home/kaden/pm-wave/consumer24/k2e8p-source; never rsynced) holding this script and `qwen4_moe_gfx1100_hook_closure.py`.
Preflight requires that worktree to be a clean `commit` (HEAD == commit, empty `git status --porcelain`, both scripts tracked) and
additionally binds both scripts by sha256 to the Git blobs `crates/peacemaker-isa/tools/<name>` at `commit` on the (clean, same-commit)
CPU checkout, and binds the oracle binary, tables, fixture, probe/observer objects and sample manifests by sha256 (oracle source maps
are checked against the same commit's blobs). The GPU host needs python3 and git; run_dir must lie outside the worktree.

Config: schema consumer24.gfx1100.e8p-closure-config.v1; the base tool's keys, with
  cpu_samples  {c24, c24h6306: hiptrx sample dir}        chain_dirs {c24-e8p, c24-mq4, c24h6306-e8p, c24h6306-mq4: fresh hipx dir}
  g1_commands  the 7 raw argv lists above (no --only/--seed/--self-check/--intrinsic-table)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time

_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen4_moe_gfx1100_hook_closure.py")
_spec = importlib.util.spec_from_file_location("qwen4_moe_gfx1100_hook_closure_base", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

Fail, die = base.Fail, base.die
SELF_E8P = os.path.abspath(__file__)
CONFIG_SCHEMA = "consumer24.gfx1100.e8p-closure-config.v1"
VARIANTS = ("c24", "c24h6306")
E8P_NEGS = ("wrong-lut", "wrong-sign-mask", "reversed-epoch", "wrong-fold-epoch", "wrong-scale-row", "wrong-expert-pointer")
MQ4_NEGS = ("wrong-scale-row", "reversed-epoch", "wrong-pack-selector", "wrong-expert-pointer")
G1_LABELS = ("synthetic-e8p",) + tuple(f"{k}-{v}" for v in VARIANTS for k in ("real-mq4", "real-e8p", "real-mixed-mq4-e8p"))
CHAIN_KEYS = tuple(f"{v}-{c}" for v in VARIANTS for c in ("e8p", "mq4"))
MODULES = {"mq4": "qwen4_moe_iu4_sym_pm_gfx1100.hxaco", "e8p": "qwen4_moe_iu4_sym_e8p_pm_gfx1100.hxaco"}


def variant_of(sample_dir):
    b = os.path.basename(sample_dir.rstrip("/"))
    for v in VARIANTS:
        if b == f"{v}-e8p":
            return v
    die(f"--sample-dir {sample_dir}: basename must be one of {[f'{v}-e8p' for v in VARIANTS]}")


def classify(argv):
    """(label, chain_key|None, sample key|None, pairs); the label set is exactly G1_LABELS."""
    _, pairs = base.parse_pairs(argv)
    mixed = [x.split("=", 1) for x in base.flag_values(pairs, "--mixed-codeobject")]
    codec = base.flag_values(pairs, "--codec")
    sd = base.flag_values(pairs, "--sample-dir")
    if len(sd) > 1:
        die("--sample-dir given more than once")
    v = variant_of(sd[0]) if sd else None
    if mixed:
        if codec or sorted(k for k, _ in mixed) != ["e8p", "mq4"] or v is None:
            die(f"mixed g1 command needs --mixed-codeobject mq4=.. and e8p=.. with an E8P --sample-dir and no --codec, got {mixed}")
        return f"real-mixed-mq4-e8p-{v}", f"{v}-e8p", v, pairs
    if len(codec) != 1 or codec[0] not in ("mq4", "e8p"):
        die(f"g1 command needs exactly one --codec mq4|e8p, got {codec}")
    if v is None:
        if codec[0] != "e8p":
            die("synthetic MQ4 is already qualified; the E8P matrix has only synthetic-e8p")
        return "synthetic-e8p", None, None, pairs
    return f"real-{codec[0]}-{v}", f"{v}-{codec[0]}", v, pairs


def normalize_g1(cfg):
    cmds = cfg["g1_commands"]
    if len(cmds) != len(G1_LABELS):
        die(f"g1_commands must hold {len(G1_LABELS)} argv lists, got {len(cmds)}")
    out = {}
    for argv in cmds:
        label, chain_key, skey, pairs = classify(argv)
        if label in out:
            die(f"duplicate g1 command for {label}")
        bad = [f for f, _ in pairs if f in base.FORBIDDEN_G1]
        if bad:
            die(f"{label}: forbidden flags {bad}")
        for f, want in (("--card", base.CARD), ("--expect-uuid", base.UUID), ("--expect-bdf", base.BDF)):
            if base.one(pairs, f, label) != want:
                die(f"{label}: {f} must be {want}")
        if (chain_key is None) == bool(base.flag_values(pairs, "--chain-dir")):
            die(f"{label}: --chain-dir is required exactly for real runs")
        negs = sorted(x.split("=", 1)[0] for x in base.flag_values(pairs, "--negative-codeobject"))
        if label.startswith("real-mixed"):
            if negs or base.flag_values(pairs, "--mq4-anchor"):
                die(f"{label}: mixed runs carry no negatives and no anchor")
        else:
            want = sorted(E8P_NEGS if "e8p" in label else MQ4_NEGS)
            if negs != want:
                die(f"{label}: needs negatives {want}, got {negs}")
            if "e8p" in label and base.flag_values(pairs, "--mq4-anchor"):
                die(f"{label}: --mq4-anchor does not apply to e8p")
        out[label] = argv
    if set(out) != set(G1_LABELS):
        die(f"g1_commands matrix is {sorted(out)}, need {sorted(G1_LABELS)}")
    if len({base.one(base.parse_pairs(a)[1], "--epilogue-codeobject", l) for l, a in out.items()}) != 1:
        die("all g1 commands must name the same --epilogue-codeobject")
    return out


def validate_config(cfg):
    if not isinstance(cfg, dict):
        die("config must be a JSON object")
    probe = dict(cfg)
    extra, missing = sorted(set(probe) - set(base.CONFIG_KEYS)), sorted(set(base.CONFIG_KEYS) - set(probe))
    if extra or missing:
        die(f"config keys: unknown {extra}, missing {missing}")
    for k, t in base.CONFIG_KEYS.items():
        if not isinstance(cfg[k], t) or (t is int and isinstance(cfg[k], bool)):
            die(f"config `{k}` must be {t.__name__}")
    if cfg["schema"] != CONFIG_SCHEMA:
        die(f"config schema must be {CONFIG_SCHEMA}")
    if not base.HEX40.match(cfg["commit"]) or not base.HEX64.match(cfg["pilot_solve_sha256"]):
        die("`commit` must be 40 lowercase hex and `pilot_solve_sha256` 64 lowercase hex")
    for k in ("gpu_repo", "cpu_repo", "cpu_oracle", "gpu_oracle", "cpu_work_dir", "cpu_chain_helper", "cpu_base_probe", "cpu_divide_table",
              "cpu_fixture", "base_probe", "divide_table", "gpu_probe_co", "gpu_probe_manifest", "gpu_observer", "run_dir"):
        if not os.path.isabs(cfg[k]):
            die(f"config `{k}` must be an absolute path")
    for k in ("gpu_host", "cpu_host", "cpu_python"):
        if not cfg[k]:
            die(f"config `{k}` is empty")
    if cfg["gpu_host"] == cfg["cpu_host"]:
        die("gpu_host and cpu_host must differ")
    if cfg["lease_max_secs"] < 3 * base.SAFETY_MARGIN_SECS:
        die(f"lease_max_secs must be >= {3 * base.SAFETY_MARGIN_SECS}")
    if set(cfg["cpu_samples"]) != set(VARIANTS) or not all(isinstance(v, str) and os.path.isabs(v) for v in cfg["cpu_samples"].values()):
        die(f"cpu_samples must be {{{', '.join(VARIANTS)}: absolute dir}}")
    for v, d in cfg["cpu_samples"].items():
        if variant_of(d) != v:
            die(f"cpu_samples[{v}] must be a dir named {v}-e8p")
    if set(cfg["chain_dirs"]) != set(CHAIN_KEYS) or not all(isinstance(v, str) and os.path.isabs(v) for v in cfg["chain_dirs"].values()):
        die(f"chain_dirs must be {{{', '.join(CHAIN_KEYS)}: absolute dir}}")
    if len(set(cfg["chain_dirs"].values())) != len(CHAIN_KEYS):
        die("chain_dirs must be distinct paths")
    lim = cfg["limits"]
    if set(lim) != {"max_frontier_rounds", "max_chain_cycles", "max_keys_per_launch"} or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in lim.values()):
        die("limits must be {max_frontier_rounds, max_chain_cycles, max_keys_per_launch} positive integers")
    if lim["max_keys_per_launch"] > 4194304:
        die("limits.max_keys_per_launch must be <= 4194304")
    if cfg["observer_repeats"] < 1:
        die("observer_repeats must be >= 1")
    for k in ("cpu_chain_helper_args", "cpu_census_args"):
        if not all(isinstance(a, str) for a in cfg[k]):
            die(f"`{k}` must be a list of strings")
    if any(a.split("=")[0] in ("--seed", "--only") for a in cfg["cpu_census_args"]):
        die("cpu_census_args must not hold --seed/--only")
    if os.path.commonpath([cfg["run_dir"], cfg["gpu_repo"]]) == cfg["gpu_repo"]:
        die("run_dir must not live inside the GPU worktree tools dir (`gpu_repo`)")
    return normalize_g1(cfg)


# The base tool's remaining machinery (observer merge, census reader, G1 receipt reader, preflight, `g1`) reads these module
# globals; everything codec-specific is overridden below or routed through them.
base.classify = classify
base.G1_LABELS = G1_LABELS
base.CODECS = ("mq4", "e8p")


class E8pRun(base.Run):
    def __init__(self, cfg, cfg_bytes, out_path):
        super().__init__(cfg, cfg_bytes, out_path)
        self.rec["e8p_script"] = {"path": SELF_E8P, "sha256": base.sha256_file(SELF_E8P), "base_tool_sha256": base.sha256_file(_BASE_PATH)}
        self.rec["schema"] = "consumer24.gfx1100.e8p-closure-receipt.v1"

    def preflight(self):
        """Base preflight with the GPU-side Git requirement kept: `gpu_repo` is the tools directory of a clean Git worktree of
        `commit` on hipx (HEAD and cleanliness verified), and in addition this script and the base closure tool, the oracle binary
        and every table/fixture/sample manifest are bound by sha256 to the clean CPU checkout at `commit`. Nothing numerical is relaxed."""
        c, rec = self.cfg, self.rec
        host = base.socket.gethostname()
        if host != c["gpu_host"]:
            die(f"must run on {c['gpu_host']} (this is {host})")
        job, cards = os.environ.get("FABRIC_JOB", ""), os.environ.get("FABRIC_CARDS", "")
        if not job or cards != base.CARD:
            die(f"not inside a fabric lease for card {base.CARD}: FABRIC_JOB={job!r} FABRIC_CARDS={cards!r} (the caller acquires it)")
        pid = int(job.split(":")[0])
        anc, cur = set(), os.getpid()
        while cur > 1:
            anc.add(cur)
            try:
                with open(f"/proc/{cur}/stat") as f:
                    cur = int(f.read().rsplit(")", 1)[1].split()[1])
            except OSError:
                break
        if pid not in anc:
            die(f"FABRIC_JOB pid {pid} is not an ancestor of this process")
        masks = sorted(k for k in os.environ if "CU_MASK" in k.upper())
        if masks:
            die(f"CU-mask environment is banned on RDNA devices: {masks}")
        rec["lease"] = {"host": host, "FABRIC_JOB": job, "FABRIC_CARDS": cards, "lease_max_secs": c["lease_max_secs"],
                        "deadline_margin_secs": base.SAFETY_MARGIN_SECS, "cu_mask_env": "none set",
                        "devenv": {k: os.environ.get(k) for k in ("HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}}
        # Source binding: both checkouts clean at `commit`; the GPU-side scripts are byte-equal to that commit's blobs.
        head, dirty = (self.git_remote(c["cpu_repo"], "rev-parse", "HEAD"), self.git_remote(c["cpu_repo"], "status", "--porcelain", "--untracked-files=normal"))
        if head != c["commit"] or dirty:
            die(f"cpu checkout is not a clean {c['commit']}: HEAD={head} dirty={dirty[:200]!r}")
        stage = os.path.realpath(c["gpu_repo"])
        if os.path.realpath(os.path.dirname(SELF_E8P)) != stage or os.path.realpath(os.path.dirname(base.SELF)) != stage:
            die(f"this script and the base closure tool must both live in gpu_repo {c['gpu_repo']}")
        tool_rel = "crates/peacemaker-isa/tools/"
        gtop = os.path.realpath(self.git_local("rev-parse", "--show-toplevel"))
        if stage != os.path.join(gtop, tool_rel.rstrip("/")):
            die(f"gpu_repo {c['gpu_repo']} must be {tool_rel} of its Git worktree (worktree top is {gtop})")
        ghead, gdirty = self.git_local("rev-parse", "HEAD"), self.git_local("status", "--porcelain", "--untracked-files=normal")
        if ghead != c["commit"] or gdirty:
            die(f"gpu checkout is not a clean {c['commit']}: HEAD={ghead} dirty={gdirty[:200]!r}")
        for p in (SELF_E8P, base.SELF):
            self.git_local("ls-files", "--error-unmatch", os.path.relpath(p, stage))
        scripts = {"e8p_closure": (SELF_E8P, tool_rel + os.path.basename(SELF_E8P)), "base_closure": (base.SELF, tool_rel + os.path.basename(base.SELF))}
        manifest = {}
        for name, (path, rel) in scripts.items():
            gs, cs = base.sha256_file(path), self.blob_sha(rel)
            if gs != cs:
                die(f"stage script {path} sha256 {gs} != Git blob {rel}@{c['commit']} on {c['cpu_host']} ({cs})")
            manifest[name] = {"stage_path": path, "git_path": rel, "sha256": gs}
        helper_rel = os.path.relpath(c["cpu_chain_helper"], c["cpu_repo"])
        if helper_rel.startswith(".."):
            die("cpu_chain_helper must live inside cpu_repo")
        self.cpu.sh(f"git -C {base.shlex.quote(c['cpu_repo'])} ls-files --error-unmatch {base.shlex.quote(helper_rel)} {base.shlex.quote(base.ROTATION_REL)}")
        rec["source"] = {"commit": c["commit"], "mode": "clean GPU worktree + sha256 manifest of the stage scripts", "gpu": {"head": ghead, "clean": True, "worktree": gtop},
                         "cpu": {"head": head, "clean": True, "helper_rel": helper_rel}, "gpu_stage_dir": c["gpu_repo"], "gpu_scripts": manifest}
        # Artifact hashes: same bytes on both hosts.
        g1 = self.plan
        fixtures = {base.one(base.parse_pairs(a)[1], "--epilogue-codeobject", l) for l, a in g1.items()}
        (fixture_gpu,) = fixtures
        samples_gpu = {}
        for label, argv in g1.items():
            _, _, scodec, pairs = classify(argv)
            if scodec:
                samples_gpu[label] = (scodec, base.one(pairs, "--sample-dir", label))
        gpu_files = {"gpu_oracle": c["gpu_oracle"], "gpu_observer": c["gpu_observer"], "gpu_probe_co": c["gpu_probe_co"],
                     "gpu_probe_manifest": c["gpu_probe_manifest"], "base_probe": c["base_probe"], "divide_table": c["divide_table"], "fixture": fixture_gpu}
        gh = {k: base.sha256_file(p) for k, p in gpu_files.items()}
        cpu_files = {"cpu_oracle": c["cpu_oracle"], "cpu_base_probe": c["cpu_base_probe"], "cpu_divide_table": c["cpu_divide_table"],
                     "cpu_fixture": c["cpu_fixture"], "cpu_chain_helper": c["cpu_chain_helper"],
                     "cpu_rotation_source": os.path.join(os.path.dirname(c["cpu_chain_helper"]), "mq_i4_a4_oracle.py")}
        for cod, d in c["cpu_samples"].items():
            cpu_files[f"cpu_sample_manifest_{cod}"] = os.path.join(d, "sample.json")
        ch = {k: self.cpu.sha256(p)[p] for k, p in cpu_files.items()}
        for a_, b_ in (("gpu_oracle", "cpu_oracle"), ("base_probe", "cpu_base_probe"), ("divide_table", "cpu_divide_table"), ("fixture", "cpu_fixture")):
            if gh[a_] != ch[b_]:
                die(f"{a_} ({gh[a_]}) differs from {b_} ({ch[b_]}); only one binary/table/fixture may exist")
        for label, (cod, d) in samples_gpu.items():
            gs = base.sha256_file(os.path.join(d, "sample.json"))
            if gs != ch[f"cpu_sample_manifest_{cod}"]:
                die(f"{label}: G1 --sample-dir {d} sample.json sha256 {gs} != cpu_samples[{cod}] {ch[f'cpu_sample_manifest_{cod}']}")
        for label, argv in g1.items():
            _, pairs = base.parse_pairs(argv)
            for f, v in pairs:
                if f in ("--codeobject", "--mq4-anchor", "--epilogue-codeobject"):
                    base.sha256_file(v)
                elif f in ("--mixed-codeobject", "--negative-codeobject"):
                    base.sha256_file(v.split("=", 1)[1])
        rec["artifacts"] = {"gpu": gh, "cpu": ch, "g1_inputs": {}}
        self.base_sha, self.div_sha, self.bin_sha = gh["base_probe"], gh["divide_table"], gh["gpu_oracle"]
        self.fixture_sha = gh["fixture"]
        if ch["cpu_chain_helper"] != self.blob_sha(helper_rel) or ch["cpu_rotation_source"] != self.blob_sha(base.ROTATION_REL):
            die("chain helper / rotation source on hiptrx differ from their Git blobs")
        # fresh destinations
        for p in [c["run_dir"], *c["chain_dirs"].values()]:
            if os.path.lexists(p):
                die(f"destination {p} already exists (must be fresh)")
        if self.cpu.exists(c["cpu_work_dir"]):
            die(f"cpu_work_dir {c['cpu_work_dir']} already exists on {c['cpu_host']} (must be fresh)")
        for p in [c["run_dir"], *c["chain_dirs"].values()]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        for sub in ("", "frontier", "census", "tables", "g1", "logs"):
            os.makedirs(os.path.join(c["run_dir"], sub))
        self.cpu.sh(f"mkdir -p {base.shlex.quote(os.path.dirname(c['cpu_work_dir']))} && mkdir {base.shlex.quote(c['cpu_work_dir'])} && "
                    f"mkdir {base.shlex.quote(c['cpu_work_dir'])}/tables {base.shlex.quote(c['cpu_work_dir'])}/logs {base.shlex.quote(c['cpu_work_dir'])}/census")
        py = self.cpu.sh(base.q(base.CPU_ENV[:4] + [c["cpu_python"], "-c",
                         "import sys,numpy,torch;print(sys.version.split()[0],numpy.__version__,torch.__version__,torch.cuda.is_available())"])).stdout.decode().split()
        if len(py) != 4 or py[3] != "False":
            die(f"cpu_python must see numpy+torch and NO GPU, got {py}")
        rec["cpu_python"] = {"python": py[0], "numpy": py[1], "torch": py[2], "cuda_available": False}
        self.save()

    def census(self, tag, chain_root):
        c = self.cfg
        self.rounds += 1
        if self.rounds > c["limits"]["max_frontier_rounds"]:
            self.rec["failure_proof"] = {"cap": "max_frontier_rounds", "rounds": self.rounds - 1, "table_sha256": self.merged_sha}
            die(f"frontier did not close within max_frontier_rounds={c['limits']['max_frontier_rounds']}")
        out = f"{c['cpu_work_dir']}/census/{tag}.json"
        argv = [c["cpu_oracle"], "--collect-hook-keys", out, "--probe-table", c["cpu_base_probe"], "--divide-table", c["cpu_divide_table"]]
        for v in VARIANTS:
            argv += ["--census-sample-dir", f"e8p={c['cpu_samples'][v]}"]
        if self.cpu_merged:
            argv += ["--intrinsic-table", self.cpu_merged]
        if chain_root:
            for v in VARIANTS:
                for cod in ("e8p", "mq4"):
                    argv += ["--census-chain-dir", f"{cod}={chain_root}/{v}-{cod}"]
        argv += c["cpu_census_args"]
        t0 = time.monotonic()
        self.log(f"census {tag}: {'with' if chain_root else 'without'} chain dirs")
        rc, serr = self.cpu.run_cpu(f"census.{tag}", argv, f"{c['cpu_work_dir']}/logs", c["cpu_work_dir"], self.left())
        ent = {"tag": tag, "argv": argv, "rc": rc, "secs": round(time.monotonic() - t0, 1), "chain_root": chain_root, "ssh_stderr": serr,
               "logs": f"{c['cpu_work_dir']}/logs/census.{tag}.{{stdout,stderr}}"}
        self.rec["rounds"].append(ent)
        if rc != 0 or not self.cpu.exists(out):
            self.save()
            die(f"census {tag} failed rc={rc}")
        raw = self.cpu.cat(out)
        local = os.path.join(c["run_dir"], "census", f"{tag}.json")
        base.atomic_write(local, raw)
        doc = base.loads(raw, f"census {tag}")
        ent["sha256"], ent["local_copy"] = base.sha256_bytes(raw), local
        view = self.census_view(doc, tag)
        ent.update(view["summary"])
        self.save()
        return view

    def prepare_chains(self, cycle):
        c = self.cfg
        root = f"{c['cpu_work_dir']}/chain-c{cycle}"
        cyc = {"cycle": cycle, "root": root, "codecs": {}}
        self.rec["chain_cycles"].append(cyc)
        self.cpu.sh(f"mkdir {shlex.quote(root)}")
        for v in VARIANTS:
            for cod in ("e8p", "mq4"):
                key = f"{v}-{cod}"
                d = f"{root}/{key}"
                self.cpu.sh(f"mkdir {shlex.quote(d)}")
                prep = [c["cpu_oracle"], "--prepare-chain", d, "--codec", cod, "--sample-dir", c["cpu_samples"][v], "--probe-table", c["cpu_base_probe"],
                        "--divide-table", c["cpu_divide_table"], "--intrinsic-table", self.cpu_merged, "--out", f"{d}/prep.json"]
                helper = [c["cpu_python"], c["cpu_chain_helper"], "--chain-dir", d] + c["cpu_chain_helper_args"]
                e = {"prepare_argv": prep, "helper_argv": helper}
                cyc["codecs"][key] = e
                for stage, argv in (("prepare", prep), ("helper", helper)):
                    self.log(f"chain c{cycle} {key} {stage}")
                    t0 = time.monotonic()
                    rc, serr = self.cpu.run_cpu(f"chain-c{cycle}-{key}-{stage}", argv, f"{c['cpu_work_dir']}/logs", c["cpu_work_dir"], self.left())
                    e[stage] = {"rc": rc, "secs": round(time.monotonic() - t0, 1), "ssh_stderr": serr}
                    self.save()
                    if rc != 0:
                        die(f"chain c{cycle} {key} {stage} rc={rc} (logs {c['cpu_work_dir']}/logs/chain-c{cycle}-{key}-{stage}.stderr)")
                prep_doc = base.loads(self.cpu.cat(f"{d}/prep.json"), f"{key} prep.json")
                if prep_doc.get("pass") is not True or prep_doc.get("problems"):
                    die(f"chain prepare {key}: {prep_doc.get('problems')}")
                h = base.loads(self.cpu.cat(f"{d}/chain-h.json"), f"{key} chain-h.json")
                a4 = base.loads(self.cpu.cat(f"{d}/chain-a4.json"), f"{key} chain-a4.json")
                hashes = self.cpu.sha256(f"{d}/chain-h.json", f"{d}/chain-a4.json")
                checks = {"probe_table_sha256": h.get("probe_table_sha256") == self.base_sha, "oracle_binary_sha256": h.get("oracle_binary_sha256") == self.bin_sha,
                          "intrinsic_tables": h.get(base.CHAIN_ROOT_TABLES) == [self.merged_sha],
                          "a4_input_manifest": a4.get("input_manifest_sha256") == hashes[f"{d}/chain-h.json"],
                          "harness": a4.get("harness", {}).get("sha256") == self.rec["artifacts"]["cpu"]["cpu_chain_helper"],
                          "rotation_source": a4.get("rotation_source", {}).get("sha256") == self.rec["artifacts"]["cpu"]["cpu_rotation_source"],
                          "pilot_solve": a4.get("pilot_solve", {}).get("sha256") == c["pilot_solve_sha256"],
                          "divide_table_sha256": h.get("divide_table_sha256") == self.div_sha,
                          "a4_root_divide_table": a4.get("divide_table_sha256") == self.div_sha,
                          "a4_root_intrinsic_tables": a4.get(base.CHAIN_ROOT_TABLES) == [self.merged_sha],
                          "gpu_disabled": bool(a4.get("gpu_disabled")),
                          "root_codec": h.get("codec") == cod,
                          "sample_manifest": (h.get("sample_manifest_sha256") or {}).get(cod) == self.rec["artifacts"]["cpu"][f"cpu_sample_manifest_{v}"]}
                bad = [k for k, ok in checks.items() if not ok]
                if bad:
                    die(f"chain {key} provenance checks failed: {bad}")
                e["source_binding"] = self.verify_source_map(h.get("source_sha256"), f"chain {key}")
                e["checks"], e["chain_h_sha256"], e["chain_a4_sha256"] = checks, hashes[f"{d}/chain-h.json"], hashes[f"{d}/chain-a4.json"]
                e["entries"] = len(a4.get("entries", []))
        self.save()
        return root

    def expected_scopes(self):
        s = self.cfg["cpu_samples"]
        out = [("synthetic/mq4", None), ("synthetic/e8p", None)]
        for v in VARIANTS:
            out += [("real/mq4", s[v]), ("real/e8p", s[v]), ("mixed/e8p", s[v])]
        return out + [("fixture", None)]

    def ship_chains(self, root):
        c = self.cfg
        out = {}
        for key in CHAIN_KEYS:
            src, dst = f"{root}/{key}", c["chain_dirs"][key]
            os.mkdir(dst)
            cmd = f"cd {shlex.quote(src)} && tar -cf - ."
            p1 = subprocess.Popen(["ssh", *base.Remote.OPTS, c["cpu_host"], cmd], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
            p2 = subprocess.run(["tar", "-C", dst, "-xf", "-"], stdin=p1.stdout)
            p1.stdout.close()
            if p1.wait() != 0 or p2.returncode != 0:
                die(f"chain transfer {key} failed")
            want = self.cpu.tree_hashes(src)
            got = {}
            for dp, _, fs in os.walk(dst):
                for f in fs:
                    got[os.path.relpath(os.path.join(dp, f), dst)] = base.sha256_file(os.path.join(dp, f))
            if want != got:
                die(f"chain {key} tree differs after transfer: {sorted(set(want.items()) ^ set(got.items()))[:4]}")
            out[key] = {"cpu": src, "gpu": dst, "files": len(got), "chain_a4_sha256": got["chain-a4.json"], "chain_h_sha256": got["chain-h.json"]}
        self.rec["chain_transfer"] = out
        self.save()

    def go(self, plan):
        self.plan, c = plan, self.cfg
        self.preflight()
        view = self.census("c0.r0", None)
        final = None
        for cycle in range(1, c["limits"]["max_chain_cycles"] + 1):
            view = self.close_frontier(view, f"c{cycle}")
            if not self.cpu_merged:
                self.observe(f"c{cycle}.required", *view["full"], "actual_required_keys")
                self.write_merged(f"c{cycle}.required")
                view = self.close_frontier(self.census(f"c{cycle}.required", None), f"c{cycle}.required")
            if view["summary"]["missing_count"] != 0:
                die("internal: frontier loop returned with missing keys")
            if view["doc"].get("hooks_closed") is not True:
                die(f"census reports missing_count 0 but hooks_closed is {view['doc'].get('hooks_closed')!r}: {view['summary']['incomplete_reasons']}")
            root = self.prepare_chains(cycle)
            view = self.census(f"c{cycle}.final", root)
            ok, why = self.final_census_ok(view)
            self.rec["chain_cycles"][-1]["final_census"] = {"ok": ok, **view["summary"], **why}
            self.save()
            if ok:
                final = root
                break
            if view["summary"]["missing_count"] == 0:
                die(f"final census with chain dirs is incomplete without missing keys: {why}")
            self.log("final census found new missing keys; returning to the frontier loop")
        if final is None:
            self.rec["failure_proof"] = {"cap": "max_chain_cycles", "cycles": c["limits"]["max_chain_cycles"], "table_sha256": self.merged_sha}
            die("chain closure did not converge within max_chain_cycles")
        self.rec["closure"] = {"ok": True, "rounds": self.rounds, "merged_table_sha256": self.merged_sha, "merged_table": self.merged_path,
                               "final_census_sha256": self.rec["rounds"][-1]["sha256"], "chain_root": final,
                               "iteration_hash": base.sha256_bytes(base.dumps([r.get("sha256") for r in self.rec["rounds"]]))}
        self.ship_chains(final)
        rc = self.g1()
        self.rec["g1_all_ok"] = all(e.get("gate_ok") for e in self.rec["g1"]) and len(self.rec["g1"]) == len(G1_LABELS)
        return 0 if self.rec["g1_all_ok"] and rc == 0 else 2


# ------------------------------------------------------------------ write-config

def write_config(a):
    for n in ("commit", "cpu_repo", "gpu_repo", "cpu_oracle", "gpu_oracle", "gpu_observer", "gpu_probe_co", "gpu_probe_manifest", "run_dir",
              "cpu_work_dir", "mq4_kernels", "e8p_kernels", "gpu_sample_c24", "gpu_sample_c24h6306", "cpu_sample_c24", "cpu_sample_c24h6306"):
        if not getattr(a, n):
            die(f"--write-config needs --{n.replace('_', '-')}")
    pr = a.probe_root
    fix = a.fixture
    gs = {"c24": a.gpu_sample_c24, "c24h6306": a.gpu_sample_c24h6306}
    kd = {"mq4": a.mq4_kernels, "e8p": a.e8p_kernels}
    common_tail = ["--probe-table", f"{pr}/observations.json", "--divide-table", f"{pr}/divide-observations.json", "--epilogue-codeobject", fix]
    ident = ["--card", base.CARD, "--expect-uuid", base.UUID, "--expect-bdf", base.BDF]

    def negs(cod):
        names = E8P_NEGS if cod == "e8p" else MQ4_NEGS
        out = []
        for n in names:
            out += ["--negative-codeobject", f"{n}={kd[cod]}/qwen4_moe_gfx1100_negative/{n}/{MODULES[cod]}"]
        return out

    def single(cod, sample=None, chain=None):
        argv = [a.gpu_oracle, "--codeobject", f"{kd[cod]}/{MODULES[cod]}", "--codec", cod]
        if sample:
            argv += ["--sample-dir", sample]
        argv += common_tail + negs(cod)
        if chain:
            argv += ["--chain-dir", chain]
        return argv + ident

    def mixed(sample, chain):
        return [a.gpu_oracle, "--mixed-codeobject", f"mq4={kd['mq4']}/{MODULES['mq4']}", "--mixed-codeobject", f"e8p={kd['e8p']}/{MODULES['e8p']}",
                "--sample-dir", sample] + common_tail + ["--chain-dir", chain] + ident

    chains = {f"{v}-{c}": f"{a.run_dir}-chain/{v}-{c}" for v in VARIANTS for c in ("e8p", "mq4")}
    cmds = [single("e8p")]
    for v in VARIANTS:
        cmds += [single("mq4", gs[v], chains[f"{v}-mq4"]), single("e8p", gs[v], chains[f"{v}-e8p"]), mixed(gs[v], chains[f"{v}-e8p"])]
    cfg = {
        "schema": CONFIG_SCHEMA, "commit": a.commit, "gpu_host": "hipx", "cpu_host": "hiptrx", "gpu_repo": a.gpu_repo, "cpu_repo": a.cpu_repo,
        "lease_max_secs": a.lease_max_secs, "cpu_oracle": a.cpu_oracle, "gpu_oracle": a.gpu_oracle, "cpu_work_dir": a.cpu_work_dir, "cpu_python": a.cpu_python,
        "cpu_samples": {"c24": a.cpu_sample_c24, "c24h6306": a.cpu_sample_c24h6306},
        "cpu_chain_helper": a.chain_helper or f"{a.cpu_repo}/{base.HELPER_REL}", "cpu_chain_helper_args": [],
        "pilot_solve_sha256": a.pilot_solve_sha256, "cpu_base_probe": a.cpu_base_probe, "cpu_divide_table": a.cpu_divide_table, "cpu_fixture": a.cpu_fixture,
        "base_probe": f"{pr}/observations.json", "divide_table": f"{pr}/divide-observations.json", "gpu_probe_co": a.gpu_probe_co,
        "gpu_probe_manifest": a.gpu_probe_manifest, "gpu_observer": a.gpu_observer, "cpu_census_args": [], "run_dir": a.run_dir, "chain_dirs": chains,
        "limits": {"max_frontier_rounds": a.max_rounds, "max_chain_cycles": a.max_cycles, "max_keys_per_launch": 4194304},
        "observer_repeats": 3, "g1_commands": cmds,
    }
    validate_config(cfg)
    base.atomic_write(a.write_config, base.dumps(cfg))
    print(json.dumps({"config": a.write_config, "sha256": base.sha256_file(a.write_config), "g1": list(G1_LABELS)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--check-config", metavar="CFG")
    ap.add_argument("--write-config", metavar="CFG")
    for f in ("commit", "cpu-repo", "gpu-repo", "cpu-oracle", "gpu-oracle", "run-dir", "cpu-work-dir",
              "mq4-kernels", "e8p-kernels", "gpu-sample-c24", "gpu-sample-c24h6306", "cpu-sample-c24", "cpu-sample-c24h6306"):
        ap.add_argument(f"--{f}")
    kc = "/home/kaden/pm-wave/consumer24/k2-oracle/key-closure"
    ap.add_argument("--gpu-observer", default=f"{kc}/gfx1100_intrinsic_keys_observe")
    ap.add_argument("--gpu-probe-co", default=f"{kc}/gfx1100_intrinsic_keys_probe.co")
    ap.add_argument("--gpu-probe-manifest", default=f"{kc}/gfx1100_intrinsic_keys_probe.manifest.json")
    ap.add_argument("--probe-root", default="/home/kaden/pm-wave/consumer24/k2-oracle/probe")
    ap.add_argument("--fixture", default="/home/kaden/pm-wave/consumer24/b4/fixture/qwen4_moe_epilogue_fixture_pm_gfx1100.hxaco")
    ap.add_argument("--cpu-base-probe", default="/home/kaden/qcal/release-0.4.3/consumer24/artifacts/b4-probe/observations.json")
    ap.add_argument("--cpu-divide-table", default="/home/kaden/qcal/release-0.4.3/consumer24/artifacts/b4-probe/divide-observations.json")
    ap.add_argument("--cpu-fixture", default="/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b4-fixture/qwen4_moe_epilogue_fixture_pm_gfx1100.hxaco")
    ap.add_argument("--cpu-python", default="python3")
    ap.add_argument("--chain-helper")
    ap.add_argument("--pilot-solve-sha256", default="53965f2328e32d86c7b87460a02573a065424e220383891da210441dc9f6c0df")
    ap.add_argument("--lease-max-secs", type=int, default=14400)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--max-cycles", type=int, default=3)
    a = ap.parse_args()
    try:
        if a.write_config:
            return write_config(a) or 0
        if a.check_config:
            raw = open(a.check_config, "rb").read()
            plan = validate_config(base.loads(raw, a.check_config))
            print(json.dumps({"config_sha256": base.sha256_bytes(raw), "g1": sorted(plan), "ok": True}))
            return 0
        if not a.config or not a.out:
            ap.error("--config PATH --out RECEIPT (or --check-config / --write-config)")
        raw = open(a.config, "rb").read()
        cfg = base.loads(raw, a.config)
        plan = validate_config(cfg)
        if os.path.exists(a.out):
            die(f"receipt {a.out} already exists")
        run = E8pRun(cfg, raw, os.path.abspath(a.out))
        try:
            rc = run.go(plan)
            run.rec["status"] = "complete" if rc == 0 else "g1_failed"
        except Fail as e:
            run.rec["status"], run.rec["failure"], rc = "failed", str(e), 1
            print(f"closure: {e}", file=sys.stderr)
        except subprocess.TimeoutExpired as e:
            run.rec["status"], run.rec["failure"], rc = "failed", f"timeout: {e}", 1
        run.rec["rc"], run.rec["finished_utc"] = rc, time.strftime("%FT%TZ", time.gmtime())
        run.save()
        print(json.dumps({"receipt": a.out, "status": run.rec["status"], "rc": rc, "closure": run.rec.get("closure", {}).get("ok", False),
                          "g1": {e["label"]: e.get("gate_ok") for e in run.rec["g1"]}}))
        return rc
    except Fail as e:
        print(f"closure: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
