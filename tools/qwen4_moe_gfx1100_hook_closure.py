#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-hold coordinator for the B4-C gfx1100 K2 hook-key closure and the eight full G1 gates.

It runs ON hipx INSIDE a fabric lease the CALLER already holds (it never acquires one), drives the CPU-only oracle
passes on hiptrx over ssh (`env ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= nice -n 19`), and uses
the GPU only for (a) the independent intrinsic observer (`gfx1100_intrinsic_keys_observe`) and (b) the final 8 G1 runs.

Recipe (one lease hold):
  0  preflight: lease env (FABRIC_JOB / FABRIC_CARDS), no CU-mask env, hipx host, clean Git checkouts at ONE commit on both
     hosts (no source is ever rsynced; only binaries/artifacts are copied), same oracle binary sha256 on both hosts, same
     tables / fixture / sample manifests on both hosts, fresh run/work/chain destinations.
  1  census: collect actual required hook keys from every planned CPU case; no fabricated downstream keys.
  2  frontier loop: CPU census over EVERY case (3 synthetic, real mq4/w2/w3, mixed w2/w3, fixture) against
     base --probe-table + --divide-table + the merged --intrinsic-table; each scalar epilogue aborts at its first missing
     query (nothing downstream is derived from a placeholder). Missing keys -> observer on the card -> conflict-checked merge
     into ONE generic table with per-batch provenance -> copied to hiptrx -> repeat until zero missing.
  3  chain: re-prepare the 3 chain dirs on hiptrx with base + merged table (`--prepare-chain`), run the certified Python
     helper there, then a FINAL census WITH the new chain dirs. Any new missing key returns to step 2 (never normalised).
  4  ship only the three chain dirs (tar over ssh) to hipx, verify per-file sha256, then run all 8 G1 commands
     (binary/base/divide/--intrinsic-table/chain paths/--out rewritten, never --only); stop on the first failed gate.
The receipt (--out) records every command, scope, source/hash, rc and first missing key. Exit: 0 all green, 1 closure/setup
failure (no G1 run), 2 closure ok but at least one G1 gate failed.

Usage:
  fabric-run.sh --class correct --card xtx --mem-gb 4 --max-secs <lease_max_secs> --priority P0 --owner <lead> -- \\
      python3 crates/peacemaker-isa/tools/qwen4_moe_gfx1100_hook_closure.py --config CFG.json --out RECEIPT.json
  python3 .../qwen4_moe_gfx1100_hook_closure.py --check-config CFG.json            # CPU-only structure check
  python3 .../qwen4_moe_gfx1100_hook_closure.py --write-config CFG.json --commit SHA --cpu-repo R1 --gpu-repo R2 \\
      --cpu-oracle BIN1 --gpu-oracle BIN2 --gpu-observer OBS --gpu-probe-co CO --gpu-probe-manifest M \\
      --run-dir D --cpu-work-dir W [--real-plan FILE] [--lease-max-secs N]       # CPU-only config generator

Config (schema consumer24.gfx1100.hook-closure-config.v1; every key REQUIRED, unknown keys rejected, JSON NaN/duplicate keys
rejected). Hashed into the receipt; no value has an implicit default:
  commit             40-hex Git commit both repos must be checked out at, clean
  gpu_host/cpu_host  hipx / hiptrx (ssh name of the CPU host)         gpu_repo/cpu_repo  absolute checkout paths
  lease_max_secs     the caller's fabric --max-secs; nothing starts after lease_max_secs - 90 s
  cpu_oracle         hiptrx binary          gpu_oracle  hipx binary (sha256 must equal cpu_oracle; the 8 G1 argv[0] become it)
  cpu_work_dir       fresh hiptrx dir       cpu_python  certified Python (numpy/torch CPU) used for the chain helper
  cpu_samples        {mq4,w2: c24 dir; w3: W3 dir} on hiptrx (sample.json sha256 must equal the G1 --sample-dir on hipx)
  cpu_chain_helper   hiptrx path inside cpu_repo, tracked     cpu_chain_helper_args  extra argv list ([] allowed)
  pilot_solve_sha256 pin the helper must report in chain-a4.json
  cpu_base_probe / cpu_divide_table / cpu_fixture   hiptrx copies of base_probe / divide_table / the G1 --epilogue-codeobject
  base_probe / divide_table / gpu_probe_co / gpu_probe_manifest / gpu_observer   hipx paths
  cpu_census_args    extra census argv ([] allowed; MUST NOT hold --seed/--only)
  run_dir            fresh hipx dir         chain_dirs  {mq4,w2,w3: fresh hipx dir} the G1 --chain-dir destinations
  limits             {max_frontier_rounds, max_chain_cycles, max_keys_per_launch}; hitting one is a FAILURE, not a fallback
  observer_repeats   observer --repeats
  g1_commands        8 raw argv lists (3 synthetic, 3 real per codec, 2 mixed real); see normalize_g1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

CONFIG_SCHEMA = "consumer24.gfx1100.hook-closure-config.v1"
RECEIPT_SCHEMA = "consumer24.gfx1100.hook-closure-receipt.v1"
KEYS_SCHEMA = "consumer24.gfx1100.intrinsic-keys.v1"
OBS_SCHEMA = "consumer24.gfx1100.intrinsic-observations.v1"
CARD, UUID, BDF, ARCH = "xtx", "GPU-43390a851e296ee5", "0000:67:00.0", "gfx1100"
CODECS = ("mq4", "w2", "w3")
G1_LABELS = ("synthetic-mq4", "synthetic-w2", "synthetic-w3", "real-mq4", "real-w2", "real-w3", "real-mixed-mq4-w2", "real-mixed-mq4-w3")
NEGS = ("wrong-scale-row", "reversed-epoch", "wrong-pack-selector", "wrong-expert-pointer")
SAFETY_MARGIN_SECS = 90
CPU_ENV = ["env", "ROCR_VISIBLE_DEVICES=-1", "HIP_VISIBLE_DEVICES=-1", "CUDA_VISIBLE_DEVICES=", "nice", "-n", "19"]
RT = "crates/hipfire-runtime"
SRC_MAIN = "examples/qwen4_moe_sym_gfx1100_g1_oracle.rs"
HELPER_REL = "crates/peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py"
ROTATION_REL = "crates/peacemaker-isa/tools/mq_i4_a4_oracle.py"
# Census/prepare contract (Consumer24Lead-2.K2Oracle.KeyCensus): schema of the census output, and the chain-h/chain-a4 root field
# carrying the sorted sha256 list of the merged generic tables (the divide table is pinned as `divide_table_sha256`).
CHAIN_ROOT_TABLES = "intrinsic_tables_sha256"
CENSUS_SCHEMA = "consumer24.gfx1100.hook-census.v1"
SELF = os.path.abspath(__file__)
HEX40, HEX64 = re.compile(r"^[0-9a-f]{40}$"), re.compile(r"^[0-9a-f]{64}$")

CONFIG_KEYS = {
    "schema": str, "commit": str, "gpu_host": str, "cpu_host": str, "gpu_repo": str, "cpu_repo": str, "lease_max_secs": int,
    "cpu_oracle": str, "gpu_oracle": str, "cpu_work_dir": str, "cpu_python": str, "cpu_samples": dict, "cpu_chain_helper": str,
    "cpu_chain_helper_args": list, "pilot_solve_sha256": str, "cpu_base_probe": str, "cpu_divide_table": str, "cpu_fixture": str,
    "base_probe": str, "divide_table": str, "gpu_probe_co": str, "gpu_probe_manifest": str, "gpu_observer": str,
    "cpu_census_args": list, "run_dir": str, "chain_dirs": dict, "limits": dict, "observer_repeats": int, "g1_commands": list,
}


class Fail(Exception):
    pass


def die(msg):
    raise Fail(msg)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        die(f"cannot read {path}: {e}")
    return h.hexdigest()


def _reject_const(c):
    raise ValueError(f"non-finite JSON constant {c}")


def _no_dups(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate JSON key {k!r}")
        d[k] = v
    return d


def loads(b, what):
    try:
        return json.loads(b, parse_constant=_reject_const, object_pairs_hook=_no_dups)
    except ValueError as e:
        die(f"{what}: {e}")


def dumps(v):
    return (json.dumps(v, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_write(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def q(argv):
    return " ".join(shlex.quote(a) for a in argv)


# ------------------------------------------------------------------ config

def parse_pairs(argv):
    """[binary, --flag value, ...] -> (binary, [(flag, value)]). Every flag of the G1 oracle takes one value."""
    if not argv or any(not isinstance(a, str) for a in argv):
        die("g1 command must be a non-empty list of strings")
    pairs, i = [], 1
    while i < len(argv):
        f = argv[i]
        if not f.startswith("--") or i + 1 >= len(argv):
            die(f"g1 command: expected `--flag value`, got {argv[i:i + 2]!r}")
        pairs.append((f, argv[i + 1]))
        i += 2
    return argv[0], pairs


def flag_values(pairs, flag):
    return [v for f, v in pairs if f == flag]


def one(pairs, flag, what):
    v = flag_values(pairs, flag)
    if len(v) != 1:
        die(f"{what}: `{flag}` must appear exactly once, found {len(v)}")
    return v[0]


def classify(argv):
    """(label, codec_for_chain|None, sample_codec|None, pairs). The label set is exactly G1_LABELS."""
    _, pairs = parse_pairs(argv)
    mixed = [v.split("=", 1) for v in flag_values(pairs, "--mixed-codeobject")]
    codec = flag_values(pairs, "--codec")
    real = bool(flag_values(pairs, "--sample-dir"))
    if mixed:
        keys = [k for k, _ in mixed]
        if codec or len(keys) != 2 or keys[0] != "mq4" or keys[1] not in ("w2", "w3") or not real:
            die(f"mixed g1 command needs --mixed-codeobject mq4=.. and w2|w3=.. with --sample-dir and no --codec, got {mixed}")
        return f"real-mixed-mq4-{keys[1]}", keys[1], keys[1], pairs
    if len(codec) != 1 or codec[0] not in CODECS:
        die(f"g1 command needs exactly one --codec mq4|w2|w3, got {codec}")
    return (f"real-{codec[0]}" if real else f"synthetic-{codec[0]}"), (codec[0] if real else None), (codec[0] if real else None), pairs


FORBIDDEN_G1 = ("--only", "--self-check", "--prepare-chain", "--collect-hook-keys", "--seed", "--intrinsic-table")


def validate_config(cfg):
    """Pure structure check; returns {label: argv} for the 8 commands."""
    if not isinstance(cfg, dict):
        die("config must be a JSON object")
    extra, missing = sorted(set(cfg) - set(CONFIG_KEYS)), sorted(set(CONFIG_KEYS) - set(cfg))
    if extra or missing:
        die(f"config keys: unknown {extra}, missing {missing}")
    for k, t in CONFIG_KEYS.items():
        if not isinstance(cfg[k], t) or (t is int and isinstance(cfg[k], bool)):
            die(f"config `{k}` must be {t.__name__}")
    if cfg["schema"] != CONFIG_SCHEMA:
        die(f"config schema must be {CONFIG_SCHEMA}")
    if not HEX40.match(cfg["commit"]) or not HEX64.match(cfg["pilot_solve_sha256"]):
        die("`commit` must be 40 lowercase hex and `pilot_solve_sha256` 64 lowercase hex")
    for k in ("gpu_repo", "cpu_repo", "cpu_oracle", "gpu_oracle", "cpu_work_dir", "cpu_chain_helper", "cpu_base_probe",
              "cpu_divide_table", "cpu_fixture", "base_probe", "divide_table", "gpu_probe_co", "gpu_probe_manifest",
              "gpu_observer", "run_dir"):
        if not os.path.isabs(cfg[k]):
            die(f"config `{k}` must be an absolute path")
    for k in ("gpu_host", "cpu_host", "cpu_python"):
        if not cfg[k]:
            die(f"config `{k}` is empty")
    if cfg["gpu_host"] == cfg["cpu_host"]:
        die("gpu_host and cpu_host must differ (the CPU passes run on hiptrx, the GPU part on hipx)")
    if cfg["lease_max_secs"] < 3 * SAFETY_MARGIN_SECS:
        die(f"lease_max_secs must be >= {3 * SAFETY_MARGIN_SECS}")
    if set(cfg["cpu_samples"]) != set(CODECS) or not all(isinstance(v, str) and os.path.isabs(v) for v in cfg["cpu_samples"].values()):
        die("cpu_samples must be {mq4, w2, w3: absolute dir}")
    if cfg["cpu_samples"]["mq4"] != cfg["cpu_samples"]["w2"]:
        die("cpu_samples mq4 and w2 must be the same c24 dir (the census names it once as w2=DIR)")
    if set(cfg["chain_dirs"]) != set(CODECS) or not all(isinstance(v, str) and os.path.isabs(v) for v in cfg["chain_dirs"].values()):
        die("chain_dirs must be {mq4, w2, w3: absolute dir}")
    if len(set(cfg["chain_dirs"].values())) != 3:
        die("chain_dirs must be three distinct paths")
    lim = cfg["limits"]
    if set(lim) != {"max_frontier_rounds", "max_chain_cycles", "max_keys_per_launch"} or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in lim.values()):
        die("limits must be {max_frontier_rounds, max_chain_cycles, max_keys_per_launch} positive integers")
    if lim["max_keys_per_launch"] > 4194304:
        die("limits.max_keys_per_launch must be <= 4194304 (observer limit per list per launch)")
    if cfg["observer_repeats"] < 1:
        die("observer_repeats must be >= 1")
    for k in ("cpu_chain_helper_args", "cpu_census_args"):
        if not all(isinstance(a, str) for a in cfg[k]):
            die(f"`{k}` must be a list of strings")
    if any(a.split("=")[0] in ("--seed", "--only") for a in cfg["cpu_census_args"]):
        die("cpu_census_args must not hold --seed/--only (census inputs must equal the G1 inputs)")
    if os.path.commonpath([cfg["run_dir"], cfg["gpu_repo"]]) == cfg["gpu_repo"]:
        die("run_dir must not live inside the Git checkout")
    return normalize_g1(cfg)


def normalize_g1(cfg):
    """Check the raw 8 argv plan: exact matrix, no --only/--seed/--self-check, common card identity and epilogue fixture."""
    cmds = cfg["g1_commands"]
    if len(cmds) != 8:
        die(f"g1_commands must hold 8 argv lists, got {len(cmds)}")
    out = {}
    for argv in cmds:
        label, chain_codec, sample_codec, pairs = classify(argv)
        if label in out:
            die(f"duplicate g1 command for {label}")
        bad = [f for f, _ in pairs if f in FORBIDDEN_G1]
        if bad:
            die(f"{label}: forbidden flags {bad} (the closure rewrites --intrinsic-table; no --only/--seed/--self-check)")
        for f, want in (("--card", CARD), ("--expect-uuid", UUID), ("--expect-bdf", BDF)):
            if one(pairs, f, label) != want:
                die(f"{label}: {f} must be {want}")
        has_chain = bool(flag_values(pairs, "--chain-dir"))
        if (chain_codec is None) == has_chain:
            die(f"{label}: --chain-dir is required exactly for real runs")
        if sample_codec is None and flag_values(pairs, "--sample-dir"):
            die(f"{label}: synthetic run must not carry --sample-dir")
        if label.startswith("synthetic"):
            negs = sorted(v.split("=", 1)[0] for v in flag_values(pairs, "--negative-codeobject"))
            if negs != sorted(NEGS):
                die(f"{label}: needs the four negative controls {NEGS}, got {negs}")
        out[label] = argv
    if set(out) != set(G1_LABELS):
        die(f"g1_commands matrix is {sorted(out)}, need {sorted(G1_LABELS)}")
    fixtures = {one(parse_pairs(a)[1], "--epilogue-codeobject", l) for l, a in out.items()}
    if len(fixtures) != 1:
        die("all g1 commands must name the same --epilogue-codeobject")
    return out


def build_g1(cfg, label, argv, merged_path, run_dir):
    """Final argv: updated binary/base/divide/--intrinsic-table/chain path/--out. Returns (argv, receipt_path, plan_changes)."""
    _, pairs = parse_pairs(argv)
    _, chain_codec, _, _ = classify(argv)
    receipt = os.path.join(run_dir, "g1", f"{label}.json")
    replaced = {"--probe-table": cfg["base_probe"], "--divide-table": cfg["divide_table"], "--out": receipt}
    if chain_codec is not None:
        replaced["--chain-dir"] = cfg["chain_dirs"][chain_codec]
    new, seen = [], set()
    for f, v in pairs:
        if f in replaced:
            new.append((f, replaced[f]))
            seen.add(f)
        else:
            new.append((f, v))
    for f, v in replaced.items():
        if f not in seen:
            new.append((f, v))
    if merged_path is not None:
        new.append(("--intrinsic-table", merged_path))
    return [cfg["gpu_oracle"]] + [x for p in new for x in p], receipt


# ------------------------------------------------------------------ write-config

def default_real_plan():
    return "/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b4.g1c.commands.sh"


def write_config(a):
    real = []
    for line in open(a.real_plan, encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            real.append(shlex.split(s))
    if len(real) != 5:
        die(f"{a.real_plan}: expected 5 raw argv lines, got {len(real)}")
    b4, pr = a.b4_root, a.probe_root
    k = f"{b4}/kernels"
    fix = f"{b4}/fixture/qwen4_moe_epilogue_fixture_pm_gfx1100.hxaco"
    files = {"mq4": "qwen4_moe_iu4_sym_pm_gfx1100.hxaco", "w2": "qwen4_moe_iu4_sym_w2_pm_gfx1100.hxaco", "w3": "qwen4_moe_iu4_sym_w3_pm_gfx1100.hxaco"}
    synth = []
    for c in CODECS:
        argv = [a.gpu_oracle, "--codeobject", f"{k}/{files[c]}", "--codec", c, "--mq4-anchor", f"{k}/{files['mq4']}",
                "--probe-table", f"{pr}/observations.json", "--divide-table", f"{pr}/divide-observations.json",
                "--epilogue-codeobject", fix]
        for n in NEGS:
            argv += ["--negative-codeobject", f"{n}={k}/qwen4_moe_gfx1100_negative/{n}/{files[c]}"]
        argv += ["--card", CARD, "--expect-uuid", UUID, "--expect-bdf", BDF]
        synth.append(argv)
    cfg = {
        "schema": CONFIG_SCHEMA, "commit": a.commit, "gpu_host": "hipx", "cpu_host": "hiptrx", "gpu_repo": a.gpu_repo,
        "cpu_repo": a.cpu_repo, "lease_max_secs": a.lease_max_secs, "cpu_oracle": a.cpu_oracle, "gpu_oracle": a.gpu_oracle,
        "cpu_work_dir": a.cpu_work_dir, "cpu_python": a.cpu_python,
        "cpu_samples": {"mq4": a.cpu_c24, "w2": a.cpu_c24, "w3": a.cpu_w3},
        "cpu_chain_helper": a.chain_helper or f"{a.cpu_repo}/{HELPER_REL}", "cpu_chain_helper_args": [],
        "pilot_solve_sha256": a.pilot_solve_sha256, "cpu_base_probe": a.cpu_base_probe, "cpu_divide_table": a.cpu_divide_table,
        "cpu_fixture": a.cpu_fixture, "base_probe": f"{pr}/observations.json", "divide_table": f"{pr}/divide-observations.json",
        "gpu_probe_co": a.gpu_probe_co, "gpu_probe_manifest": a.gpu_probe_manifest, "gpu_observer": a.gpu_observer,
        "cpu_census_args": [], "run_dir": a.run_dir, "chain_dirs": {c: f"{a.run_dir}-chain/{c}" for c in CODECS},
        "limits": {"max_frontier_rounds": a.max_rounds, "max_chain_cycles": a.max_cycles, "max_keys_per_launch": 4194304},
        "observer_repeats": 3,
        "g1_commands": synth + [[a.gpu_oracle] + r[1:] for r in real],
    }
    validate_config(cfg)
    atomic_write(a.write_config, dumps(cfg))
    print(json.dumps({"config": a.write_config, "sha256": sha256_file(a.write_config), "g1": list(G1_LABELS)}))


# ------------------------------------------------------------------ remote / local execution

class Remote:
    OPTS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]

    def __init__(self, host, left):
        self.host, self.left = host, left

    def sh(self, cmd, timeout=None, stdin=None, check=True):
        t = min(timeout or 1e9, self.left())
        if t <= 0:
            die("lease time budget exhausted")
        p = subprocess.run(["ssh", *self.OPTS, self.host, cmd], stdin=stdin if stdin is not None else subprocess.DEVNULL,
                           capture_output=True, timeout=t)
        if check and p.returncode != 0:
            die(f"ssh {self.host} `{cmd[:200]}` rc={p.returncode}: {p.stderr.decode(errors='replace')[-600:]}")
        return p

    def sha256(self, *paths):
        p = self.sh("sha256sum -- " + q(paths))
        out = {}
        for line in p.stdout.decode().splitlines():
            h, _, name = line.partition("  ")
            out[name] = h
        if set(out) != set(paths):
            die(f"remote sha256sum returned {sorted(out)} for {list(paths)}")
        return out

    def cat(self, path):
        return self.sh(f"cat -- {shlex.quote(path)}").stdout

    def exists(self, path):
        return self.sh(f"test -e {shlex.quote(path)}", check=False).returncode == 0

    def tree_hashes(self, path):
        p = self.sh(f"cd {shlex.quote(path)} && find . -type f -print0 | sort -z | xargs -0 sha256sum")
        out = {}
        for line in p.stdout.decode().splitlines():
            h, _, name = line.partition("  ")
            out[name[2:] if name.startswith("./") else name] = h
        return out

    def run_cpu(self, label, argv, logs, cwd, timeout):
        """CPU-only job: GPUs hidden, nice 19, hard `timeout`, stdout/stderr kept on the CPU host. Returns rc."""
        t = int(min(timeout, self.left()))
        if t <= 0:
            die("lease time budget exhausted")
        inner = q(CPU_ENV + ["timeout", "-k", "30", str(t)] + argv)
        cmd = f"mkdir -p {shlex.quote(logs)} && cd {shlex.quote(cwd)} && {inner} > {shlex.quote(logs)}/{label}.stdout 2> {shlex.quote(logs)}/{label}.stderr"
        p = subprocess.run(["ssh", *self.OPTS, self.host, cmd], stdin=subprocess.DEVNULL, capture_output=True, timeout=t + 60)
        return p.returncode, p.stderr.decode(errors="replace")[-400:]


class Run:
    def __init__(self, cfg, cfg_bytes, out_path):
        self.cfg, self.out_path = cfg, out_path
        self.t0 = time.monotonic()
        self.deadline = self.t0 + cfg["lease_max_secs"] - SAFETY_MARGIN_SECS
        self.cpu = Remote(cfg["cpu_host"], self.left)
        self.rec = {"schema": RECEIPT_SCHEMA, "status": "in_progress", "config": cfg, "config_sha256": sha256_bytes(cfg_bytes),
                    "script": {"path": SELF, "sha256": sha256_file(SELF)}, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
                    "rounds": [], "chain_cycles": [], "g1": [], "tables": {}, "merge_batches": []}
        self.merged_rows = {"exp2": {}, "ldexp": {}, "divide": {}}   # key -> (result, batch index)
        self.merged_path = None          # hipx path of the current merged table
        self.merged_sha = None
        self.cpu_merged = None           # hiptrx copy
        self.rounds = 0
        self.blob_cache = {}

    def left(self):
        return self.deadline - time.monotonic()

    def log(self, msg):
        print(f"[closure +{time.monotonic() - self.t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    def save(self):
        self.rec["elapsed_secs"] = round(time.monotonic() - self.t0, 1)
        atomic_write(self.out_path, dumps(self.rec))

    def local(self, argv, log_stem, timeout=None, env=None):
        """GPU-host subprocess under the caller's lease environment. Returns rc; stdout/stderr go to files."""
        t = min(timeout or 1e9, self.left())
        if t <= 0:
            die("lease time budget exhausted")
        with open(log_stem + ".stdout", "wb") as so, open(log_stem + ".stderr", "wb") as se:
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=so, stderr=se, env=env, start_new_session=True)
            try:
                return p.wait(timeout=t)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                p.wait()
                die(f"`{argv[0]}` exceeded the remaining lease budget ({t:.0f}s); killed")

    # ---------------------------------------------------------------- preflight
    def git_local(self, *args):
        p = subprocess.run(["git", "-C", self.cfg["gpu_repo"], *args], capture_output=True, text=True)
        if p.returncode != 0:
            die(f"git {' '.join(args)}: {p.stderr.strip()}")
        return p.stdout.strip()

    def git_remote(self, repo, *args):
        return self.cpu.sh(f"git -C {shlex.quote(repo)} " + q(list(args))).stdout.decode().strip()

    def blob_sha(self, rel):
        if rel not in self.blob_cache:
            r = self.cpu.sh(f"git -C {shlex.quote(self.cfg['cpu_repo'])} show {self.cfg['commit']}:{shlex.quote(rel)} | sha256sum")
            self.blob_cache[rel] = r.stdout.decode().split()[0]
        return self.blob_cache[rel]

    def verify_source_map(self, m, where):
        if not isinstance(m, dict) or SRC_MAIN not in m:
            die(f"{where}: source_sha256 map missing or lacks {SRC_MAIN}")
        for k, v in m.items():
            if v != self.blob_sha(f"{RT}/{k}"):
                die(f"{where}: {k} sha256 {v} != Git blob at {self.cfg['commit']} on {self.cfg['cpu_host']}")
        return {"files": len(m), "commit": self.cfg["commit"]}

    def preflight(self):
        c, rec = self.cfg, self.rec
        host = socket.gethostname()
        if host != c["gpu_host"]:
            die(f"must run on {c['gpu_host']} (this is {host})")
        job, cards = os.environ.get("FABRIC_JOB", ""), os.environ.get("FABRIC_CARDS", "")
        if not job or cards != CARD:
            die(f"not inside a fabric lease for card {CARD}: FABRIC_JOB={job!r} FABRIC_CARDS={cards!r} (the caller acquires it)")
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
                        "deadline_margin_secs": SAFETY_MARGIN_SECS, "cu_mask_env": "none set",
                        "devenv": {k: os.environ.get(k) for k in ("HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}}
        # Git: one commit, both clean checkouts, tracked tool files.
        top = self.git_local("rev-parse", "--show-toplevel")
        if os.path.realpath(top) != os.path.realpath(c["gpu_repo"]):
            die(f"this script lives in {top}, config gpu_repo is {c['gpu_repo']}")
        rel_self = os.path.relpath(SELF, top)
        self.git_local("ls-files", "--error-unmatch", rel_self)
        src = {"commit": c["commit"], "gpu": {}, "cpu": {}}
        for side, run in (("gpu", lambda *a: self.git_local(*a)), ("cpu", lambda *a: self.git_remote(c["cpu_repo"], *a))):
            head, dirty = run("rev-parse", "HEAD"), run("status", "--porcelain", "--untracked-files=normal")
            if head != c["commit"] or dirty:
                die(f"{side} checkout is not a clean {c['commit']}: HEAD={head} dirty={dirty[:200]!r}")
            src[side] = {"head": head, "clean": True}
        helper_rel = os.path.relpath(c["cpu_chain_helper"], c["cpu_repo"])
        if helper_rel.startswith(".."):
            die("cpu_chain_helper must live inside cpu_repo")
        self.cpu.sh(f"git -C {shlex.quote(c['cpu_repo'])} ls-files --error-unmatch {shlex.quote(helper_rel)} {shlex.quote(ROTATION_REL)}")
        src["cpu"]["helper_rel"] = helper_rel
        rec["source"] = src
        # Artifact hashes: same bytes on both hosts.
        g1 = self.plan
        fixtures = {one(parse_pairs(a)[1], "--epilogue-codeobject", l) for l, a in g1.items()}
        (fixture_gpu,) = fixtures
        samples_gpu = {}
        for label, argv in g1.items():
            _, _, scodec, pairs = classify(argv)
            if scodec:
                samples_gpu[label] = (scodec, one(pairs, "--sample-dir", label))
        gpu_files = {"gpu_oracle": c["gpu_oracle"], "gpu_observer": c["gpu_observer"], "gpu_probe_co": c["gpu_probe_co"],
                     "gpu_probe_manifest": c["gpu_probe_manifest"], "base_probe": c["base_probe"], "divide_table": c["divide_table"],
                     "fixture": fixture_gpu}
        gh = {k: sha256_file(p) for k, p in gpu_files.items()}
        cpu_files = {"cpu_oracle": c["cpu_oracle"], "cpu_base_probe": c["cpu_base_probe"], "cpu_divide_table": c["cpu_divide_table"],
                     "cpu_fixture": c["cpu_fixture"], "cpu_chain_helper": c["cpu_chain_helper"],
                     "cpu_rotation_source": os.path.join(os.path.dirname(c["cpu_chain_helper"]), "mq_i4_a4_oracle.py")}
        for cod, d in c["cpu_samples"].items():
            cpu_files[f"cpu_sample_manifest_{cod}"] = os.path.join(d, "sample.json")
        ch = {k: self.cpu.sha256(p)[p] for k, p in cpu_files.items()}
        for a_, b_ in (("gpu_oracle", "cpu_oracle"), ("base_probe", "cpu_base_probe"), ("divide_table", "cpu_divide_table"),
                       ("fixture", "cpu_fixture")):
            if gh[a_] != ch[b_]:
                die(f"{a_} ({gh[a_]}) differs from {b_} ({ch[b_]}); only one binary/table/fixture may exist")
        for label, (cod, d) in samples_gpu.items():
            gs = sha256_file(os.path.join(d, "sample.json"))
            if gs != ch[f"cpu_sample_manifest_{cod}"]:
                die(f"{label}: G1 --sample-dir {d} sample.json sha256 {gs} != cpu_samples[{cod}] {ch[f'cpu_sample_manifest_{cod}']}")
        for label, argv in g1.items():
            _, pairs = parse_pairs(argv)
            for f, v in pairs:
                if f in ("--codeobject", "--mq4-anchor", "--epilogue-codeobject"):
                    sha256_file(v)
                elif f in ("--mixed-codeobject", "--negative-codeobject"):
                    sha256_file(v.split("=", 1)[1])
        rec["artifacts"] = {"gpu": gh, "cpu": ch, "g1_inputs": {}}
        self.base_sha, self.div_sha, self.bin_sha = gh["base_probe"], gh["divide_table"], gh["gpu_oracle"]
        self.fixture_sha = gh["fixture"]
        if ch["cpu_chain_helper"] != self.blob_sha(helper_rel) or ch["cpu_rotation_source"] != self.blob_sha(ROTATION_REL):
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
        self.cpu.sh(f"mkdir -p {shlex.quote(os.path.dirname(c['cpu_work_dir']))} && mkdir {shlex.quote(c['cpu_work_dir'])} && "
                    f"mkdir {shlex.quote(c['cpu_work_dir'])}/tables {shlex.quote(c['cpu_work_dir'])}/logs {shlex.quote(c['cpu_work_dir'])}/census")
        py = self.cpu.sh(q(CPU_ENV[:4] + [c["cpu_python"], "-c",
                         "import sys,numpy,torch;print(sys.version.split()[0],numpy.__version__,torch.__version__,torch.cuda.is_available())"])).stdout.decode().split()
        if len(py) != 4 or py[3] != "False":
            die(f"cpu_python must see numpy+torch and NO GPU, got {py}")
        rec["cpu_python"] = {"python": py[0], "numpy": py[1], "torch": py[2], "cuda_available": False}
        self.save()

    # ---------------------------------------------------------------- observer / merge
    @staticmethod
    def u32(v, what):
        if isinstance(v, str):
            v = int(v, 16) if v.lower().startswith("0x") else int(v)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 0xFFFFFFFF:
            die(f"{what}: not a u32: {v!r}")
        return v

    def i32(self, v, what):
        if isinstance(v, str):
            v = int(v, 0)
        if isinstance(v, bool) or not isinstance(v, int) or not -(1 << 31) <= v < (1 << 31):
            die(f"{what}: not an i32: {v!r}")
        return v

    def parse_keys(self, d, what):
        """Canonical, de-duplicated, order-preserving key lists from a probe-keys-shaped object."""
        e = [self.u32(x, f"{what}.exp2") for x in d.get("exp2", [])]
        l = [(self.u32(x["value_bits"], f"{what}.ldexp"), self.i32(x["exponent"], f"{what}.ldexp")) for x in d.get("ldexp", [])]
        v = [(self.u32(x["numerator_bits"], f"{what}.divide"), self.u32(x["denominator_bits"], f"{what}.divide")) for x in d.get("divide", [])]
        return list(dict.fromkeys(e)), list(dict.fromkeys(l)), list(dict.fromkeys(v))

    def observe(self, tag, e, l, v, purpose):
        """Run the independent observer on exact keys; validate; merge. Returns number of NEW table rows."""
        c = self.cfg
        n = c["limits"]["max_keys_per_launch"]
        new_total = 0
        for b in range(max(1, -(-max(len(e), len(l), len(v)) // n))):
            ee, ll, vv = e[b * n:(b + 1) * n], l[b * n:(b + 1) * n], v[b * n:(b + 1) * n]
            if not (ee or ll or vv):
                continue
            stem = os.path.join(c["run_dir"], "frontier", f"{tag}.b{b}")
            keys = {"schema": KEYS_SCHEMA, "exp2": ee, "ldexp": [{"value_bits": a, "exponent": x} for a, x in ll],
                    "divide": [{"numerator_bits": a, "denominator_bits": d} for a, d in vv]}
            atomic_write(stem + ".keys.json", dumps(keys))
            argv = [c["gpu_observer"], "--keys", stem + ".keys.json", "--codeobject", c["gpu_probe_co"], "--manifest", c["gpu_probe_manifest"],
                    "--card", CARD, "--expect-uuid", UUID, "--expect-bdf", BDF, "--out", stem + ".observations.json",
                    "--repeats", str(c["observer_repeats"])]
            self.log(f"observer {tag}.b{b}: {len(ee)} exp2, {len(ll)} ldexp, {len(vv)} divide keys")
            rc = self.local(argv, stem + ".observer", env=dict(os.environ))
            entry = {"tag": tag, "batch": b, "purpose": purpose, "argv": argv, "rc": rc, "keys_sha256": sha256_file(stem + ".keys.json"),
                     "counts": {"exp2": len(ee), "ldexp": len(ll), "divide": len(vv)}}
            self.rec["merge_batches"].append(entry)
            if rc != 0 or not os.path.exists(stem + ".observations.json"):
                self.save()
                die(f"observer {tag}.b{b} failed rc={rc} (stderr {stem}.observer.stderr)")
            entry["observations_sha256"] = sha256_file(stem + ".observations.json")
            res = loads(open(stem + ".observations.json", "rb").read(), f"{tag}.b{b} observations")
            new_total += self.validate_and_merge(res, ee, ll, vv, len(self.rec["merge_batches"]) - 1, entry)
            self.save()
        return new_total

    def validate_and_merge(self, res, ee, ll, vv, bidx, entry):
        c = self.cfg
        if res.get("schema") != OBS_SCHEMA:
            die(f"observer result schema {res.get('schema')!r}")
        dev, sc, pr = res.get("device", {}), res.get("selfcheck", {}), res.get("probe", {})
        if sc.get("ok") is not True or dev.get("identity_match") is not True:
            die(f"observer selfcheck/identity failed: {sc} {dev}")
        if (dev.get("arch"), dev.get("uuid"), dev.get("bdf"), dev.get("card")) != (ARCH, UUID, BDF, CARD):
            die(f"observer device is not the gfx1100 xtx: {dev}")
        co_sha, man_sha = sha256_file(c["gpu_probe_co"]), sha256_file(c["gpu_probe_manifest"])
        if pr.get("co_sha256") != co_sha or pr.get("manifest_sha256") != man_sha:
            die("observer probe co/manifest sha256 differ from the configured probe artifact")
        d = pr.get("denorm", {})
        if d.get("float_denorm_mode_32") != 3 or d.get("float_denorm_mode_16_64") != 3:
            die(f"observer denorm modes are not 3/3: {d}")
        if pr.get("keys_file_sha256") not in (None, entry["keys_sha256"]):
            die("observer keys_file_sha256 differs from the file that was submitted")
        got = {}
        for name, rows, kf, req in (("exp2", res.get("exp2"), lambda r: (r["arg"],), [(x,) for x in ee]),
                                    ("ldexp", res.get("ldexp"), lambda r: (r["arg"], r["exponent"]), ll),
                                    ("divide", res.get("divide"), lambda r: (r["numerator"], r["denominator"]), vv)):
            if not isinstance(rows, list):
                die(f"observer result lacks `{name}`")
            m = {}
            for r in rows:
                val = self.u32(r["result"], f"{name}.result")
                if name == "ldexp":
                    k = (self.u32(r["arg"], name), self.i32(r["exponent"], name))
                elif name == "divide":
                    k = (self.u32(r["numerator"], name), self.u32(r["denominator"], name))
                else:
                    k = (self.u32(r["arg"], name),)
                if k in m and m[k] != val:
                    die(f"observer returned conflicting results for {name} {k}")
                m[k] = val
            if set(m) != set(req):
                die(f"observer {name} keys differ from the request: missing {sorted(set(req) - set(m))[:5]} extra {sorted(set(m) - set(req))[:5]}")
            got[name] = m
        new = 0
        for name in ("exp2", "ldexp", "divide"):
            store = self.merged_rows[name]
            for k, val in got[name].items():
                key = k[0] if name == "exp2" else k
                if key in store:
                    if store[key][0] != val:
                        die(f"CONFLICT merging {name} {key}: table {store[key][0]:#010x} (batch {store[key][1]}) vs new {val:#010x} (batch {bidx})")
                else:
                    store[key] = (val, bidx)
                    new += 1
        entry.update({"device": dev, "selfcheck": sc, "probe": {"co_sha256": co_sha, "manifest_sha256": man_sha, "denorm": d},
                      "records_sha256": res.get("records_sha256"), "new_rows": new, "observer_counts": res.get("counts")})
        return new

    def write_merged(self, tag):
        """ONE generic table: sorted rows, parallel per-row batch-provenance arrays, batches[] metadata; copied to hiptrx."""
        c = self.cfg
        t, bs = self.merged_rows, self.rec["merge_batches"]
        e, l, v = sorted(t["exp2"].items()), sorted(t["ldexp"].items()), sorted(t["divide"].items())
        dev = bs[0]["device"]
        doc = {"schema": OBS_SCHEMA, "kind": "merged-generic-extras",
               "exp2": [{"arg": k, "result": r} for k, (r, _) in e],
               "ldexp": [{"arg": k[0], "exponent": k[1], "result": r} for k, (r, _) in l],
               "divide": [{"numerator": k[0], "denominator": k[1], "result": r} for k, (r, _) in v],
               "row_batch": {"exp2": [b for _, (_, b) in e], "ldexp": [b for _, (_, b) in l], "divide": [b for _, (_, b) in v]},
               "device": dev, "probe": bs[0]["probe"], "selfcheck": {"ok": True, "batches": len(bs), "all_batches_ok": True},
               "counts": {"exp2": len(e), "ldexp": len(l), "divide": len(v)},
               "batches": [{"index": i, "tag": b["tag"], "purpose": b["purpose"], "keys_sha256": b["keys_sha256"],
                            "observations_sha256": b["observations_sha256"], "records_sha256": b.get("records_sha256"),
                            "new_rows": b["new_rows"], "device": b["device"], "probe": b["probe"]} for i, b in enumerate(bs)]}
        for b in bs:
            if (b["device"].get("uuid"), b["device"].get("bdf"), b["device"].get("arch")) != (dev["uuid"], dev["bdf"], dev["arch"]) or b["probe"] != bs[0]["probe"]:
                die("batches disagree on device identity or probe artifact")
        data = dumps(doc)
        path = os.path.join(c["run_dir"], "tables", f"intrinsic-merged.{tag}.json")
        atomic_write(path, data)
        self.merged_path, self.merged_sha = path, sha256_bytes(data)
        self.cpu_merged = f"{c['cpu_work_dir']}/tables/intrinsic-merged.{tag}.json"
        r = self.cpu.sh(f"cat > {shlex.quote(self.cpu_merged)}", stdin=open(path, "rb"))
        if self.cpu.sha256(self.cpu_merged)[self.cpu_merged] != self.merged_sha:
            die("merged table copy to hiptrx corrupted")
        self.rec["tables"]["merged"] = {"path": path, "cpu_path": self.cpu_merged, "sha256": self.merged_sha, "counts": doc["counts"], "batches": len(bs)}
        self.rec["tables"]["base_probe_sha256"], self.rec["tables"]["divide_sha256"] = self.base_sha, self.div_sha
        self.log(f"merged table {tag}: {doc['counts']} sha256 {self.merged_sha[:16]}")

    # ---------------------------------------------------------------- census / chain
    def census(self, tag, chain_root):
        c = self.cfg
        self.rounds += 1
        if self.rounds > c["limits"]["max_frontier_rounds"]:
            self.rec["failure_proof"] = {"cap": "max_frontier_rounds", "rounds": self.rounds - 1, "table_sha256": self.merged_sha}
            die(f"frontier did not close within max_frontier_rounds={c['limits']['max_frontier_rounds']}")
        out = f"{c['cpu_work_dir']}/census/{tag}.json"
        argv = [c["cpu_oracle"], "--collect-hook-keys", out, "--probe-table", c["cpu_base_probe"], "--divide-table", c["cpu_divide_table"],
                "--census-sample-dir", f"w2={c['cpu_samples']['w2']}", "--census-sample-dir", f"w3={c['cpu_samples']['w3']}"]
        if self.cpu_merged:
            argv += ["--intrinsic-table", self.cpu_merged]
        if chain_root:
            for cod in CODECS:
                argv += ["--census-chain-dir", f"{cod}={chain_root}/{cod}"]
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
        atomic_write(local, raw)
        doc = loads(raw, f"census {tag}")
        ent["sha256"], ent["local_copy"] = sha256_bytes(raw), local
        view = self.census_view(doc, tag)
        ent.update(view["summary"])
        self.save()
        return view

    def census_view(self, doc, tag):
        """Strict reader of the census output. `keys` = FULL required set, `missing` = frontier (probe-keys shape)."""
        if doc.get("schema") != CENSUS_SCHEMA:
            die(f"census {tag}: schema {doc.get('schema')!r}, expected {CENSUS_SCHEMA}")
        for k in ("complete", "missing_count", "keys", "missing", "scopes", "binary_sha256", "tables", "incomplete_reasons", "first_missing"):
            if k not in doc:
                die(f"census {tag}: output lacks `{k}`")
        if doc["binary_sha256"] != self.bin_sha:
            die(f"census {tag}: binary sha256 {doc['binary_sha256']} != {self.bin_sha}")
        tb = doc["tables"]
        if tb.get("probe_sha256") != self.base_sha or tb.get("divide_sha256") != self.div_sha:
            die(f"census {tag}: base/divide table sha256 differ from the configured tables")
        if tb.get("intrinsic_sha256", []) != ([self.merged_sha] if self.merged_sha else []):
            die(f"census {tag}: intrinsic table sha256 list {tb.get('intrinsic_sha256')} != [{self.merged_sha}]")
        if "source_sha256" in doc:
            self.verify_source_map(doc["source_sha256"], f"census {tag}")
        full = self.parse_keys(doc["keys"], f"census {tag}.keys")
        miss = self.parse_keys(doc["missing"], f"census {tag}.missing")
        mc = doc["missing_count"]
        if not isinstance(mc, int) or mc != sum(len(x) for x in miss):
            die(f"census {tag}: missing_count {mc} != sum of frontier lists {sum(len(x) for x in miss)}")
        if (mc == 0) != (doc.get("first_missing") in (None, "")):
            die(f"census {tag}: first_missing {doc.get('first_missing')!r} inconsistent with missing_count {mc}")
        scopes = {s["name"]: s for s in doc["scopes"]}
        summary = {"complete": doc["complete"], "missing_count": mc, "first_missing": doc.get("first_missing"),
                   "incomplete_reasons": doc["incomplete_reasons"],
                   "full_keys": {"exp2": len(full[0]), "ldexp": len(full[1]), "divide": len(full[2])},
                   "scopes": {n: {"cases": s.get("cases"), "complete": s.get("complete"), "missing_count": s.get("missing_count")} for n, s in scopes.items()},
                   "dedup": doc.get("queries"), "chain_stage": doc.get("chain_stage"), "epilogue_pairs": doc.get("epilogue_pairs")}
        return {"doc": doc, "full": full, "missing": miss, "scopes": scopes, "summary": summary}

    def close_frontier(self, view, tag):
        n = 0
        while view["summary"]["missing_count"] > 0:
            n += 1
            e, l, v = view["missing"]
            if not self.observe(f"{tag}.r{self.rounds}", e, l, v, "frontier"):
                die(f"frontier keys {view['summary']['first_missing']!r} are already in the merged table; census and table disagree")
            self.write_merged(f"{tag}.r{self.rounds}")
            view = self.census(f"{tag}.r{self.rounds + 1}", None)
        return view

    def prepare_chains(self, cycle):
        c = self.cfg
        root = f"{c['cpu_work_dir']}/chain-c{cycle}"
        cyc = {"cycle": cycle, "root": root, "codecs": {}}
        self.rec["chain_cycles"].append(cyc)
        self.cpu.sh(f"mkdir {shlex.quote(root)}")
        for cod in CODECS:
            d = f"{root}/{cod}"
            self.cpu.sh(f"mkdir {shlex.quote(d)}")
            prep = [c["cpu_oracle"], "--prepare-chain", d, "--codec", cod, "--sample-dir", c["cpu_samples"][cod], "--probe-table", c["cpu_base_probe"],
                    "--divide-table", c["cpu_divide_table"], "--intrinsic-table", self.cpu_merged, "--out", f"{d}/prep.json"]
            helper = [c["cpu_python"], c["cpu_chain_helper"], "--chain-dir", d] + c["cpu_chain_helper_args"]
            e = {"prepare_argv": prep, "helper_argv": helper}
            cyc["codecs"][cod] = e
            for stage, argv in (("prepare", prep), ("helper", helper)):
                self.log(f"chain c{cycle} {cod} {stage}")
                t0 = time.monotonic()
                rc, serr = self.cpu.run_cpu(f"chain-c{cycle}-{cod}-{stage}", argv, f"{c['cpu_work_dir']}/logs", c["cpu_work_dir"], self.left())
                e[stage] = {"rc": rc, "secs": round(time.monotonic() - t0, 1), "ssh_stderr": serr}
                self.save()
                if rc != 0:
                    die(f"chain c{cycle} {cod} {stage} rc={rc} (logs {c['cpu_work_dir']}/logs/chain-c{cycle}-{cod}-{stage}.stderr)")
            prep_doc = loads(self.cpu.cat(f"{d}/prep.json"), f"{cod} prep.json")
            if prep_doc.get("pass") is not True or prep_doc.get("problems"):
                die(f"chain prepare {cod}: {prep_doc.get('problems')}")
            h = loads(self.cpu.cat(f"{d}/chain-h.json"), f"{cod} chain-h.json")
            a4_raw = self.cpu.cat(f"{d}/chain-a4.json")
            a4 = loads(a4_raw, f"{cod} chain-a4.json")
            hashes = self.cpu.sha256(f"{d}/chain-h.json", f"{d}/chain-a4.json")
            checks = {"probe_table_sha256": h.get("probe_table_sha256") == self.base_sha, "oracle_binary_sha256": h.get("oracle_binary_sha256") == self.bin_sha,
                      "intrinsic_tables": h.get(CHAIN_ROOT_TABLES) == [self.merged_sha],
                      "a4_input_manifest": a4.get("input_manifest_sha256") == hashes[f"{d}/chain-h.json"],
                      "harness": a4.get("harness", {}).get("sha256") == self.rec["artifacts"]["cpu"]["cpu_chain_helper"],
                      "rotation_source": a4.get("rotation_source", {}).get("sha256") == self.rec["artifacts"]["cpu"]["cpu_rotation_source"],
                      "pilot_solve": a4.get("pilot_solve", {}).get("sha256") == c["pilot_solve_sha256"],
                      "divide_table_sha256": h.get("divide_table_sha256") == self.div_sha,
                      "a4_root_divide_table": a4.get("divide_table_sha256") == self.div_sha,
                      "a4_root_intrinsic_tables": a4.get(CHAIN_ROOT_TABLES) == [self.merged_sha],
                      "gpu_disabled": bool(a4.get("gpu_disabled"))}
            bad = [k for k, ok in checks.items() if not ok]
            if bad:
                die(f"chain {cod} provenance checks failed: {bad}")
            e["source_binding"] = self.verify_source_map(h.get("source_sha256"), f"chain {cod}")
            e["checks"], e["chain_h_sha256"], e["chain_a4_sha256"] = checks, hashes[f"{d}/chain-h.json"], hashes[f"{d}/chain-a4.json"]
            e["entries"] = len(a4.get("entries", []))
        self.save()
        return root

    def expected_scopes(self):
        """(prefix, dir|None) of every scope the FINAL census must report complete: 3 synthetic, real per codec (mq4 on both sample
        sets), mixed w2/w3, fixture. Chain-scope names come from the census' own `required_scopes` array."""
        s = self.cfg["cpu_samples"]
        return ([(f"synthetic/{x}", None) for x in CODECS] + [("real/mq4", s["w2"]), ("real/w2", s["w2"]), ("real/mq4", s["w3"]),
                ("real/w3", s["w3"]), ("mixed/w2", s["w2"]), ("mixed/w3", s["w3"]), ("fixture", None)])

    def final_census_ok(self, view):
        s, doc = view["summary"], view["doc"]
        names = list(view["scopes"])

        def hit(pre, d):
            return [n for n in names if (n == pre if d is None else n.startswith(pre + "/") and n.endswith(d.rstrip("/")))]
        absent, scoped = [], []
        for pre, d in self.expected_scopes():
            h = hit(pre, d)
            (scoped.extend(h) if h else absent.append(f"{pre}/{d}" if d else pre))
        declared = doc.get("required_scopes", [])
        absent += [n for n in declared if n not in view["scopes"]]
        scoped += [n for n in declared if n in view["scopes"]]
        incomplete = sorted({n for n in scoped if not (view["scopes"][n].get("complete") is True and view["scopes"][n].get("missing_count") == 0)})
        stage, reasons = doc.get("chain_stage") or {}, doc.get("incomplete_reasons")
        ok = (s["complete"] is True and s["missing_count"] == 0 and reasons == [] and not absent and not incomplete
              and stage.get("ready") is True and bool(declared))
        return ok, {"absent_scopes": absent, "incomplete_scopes": incomplete, "incomplete_reasons": reasons, "chain_stage": stage,
                    "declared_required_scopes": declared}

    def ship_chains(self, root):
        c = self.cfg
        out = {}
        for cod in CODECS:
            src, dst = f"{root}/{cod}", c["chain_dirs"][cod]
            os.mkdir(dst)
            cmd = f"cd {shlex.quote(src)} && tar -cf - ."
            p1 = subprocess.Popen(["ssh", *Remote.OPTS, c["cpu_host"], cmd], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
            p2 = subprocess.run(["tar", "-C", dst, "-xf", "-"], stdin=p1.stdout)
            p1.stdout.close()
            if p1.wait() != 0 or p2.returncode != 0:
                die(f"chain transfer {cod} failed")
            want = self.cpu.tree_hashes(src)
            got = {}
            for dp, _, fs in os.walk(dst):
                for f in fs:
                    got[os.path.relpath(os.path.join(dp, f), dst)] = sha256_file(os.path.join(dp, f))
            if want != got:
                die(f"chain {cod} tree differs after transfer: {sorted(set(want.items()) ^ set(got.items()))[:4]}")
            out[cod] = {"cpu": src, "gpu": dst, "files": len(got), "chain_a4_sha256": got["chain-a4.json"], "chain_h_sha256": got["chain-h.json"]}
        self.rec["chain_transfer"] = out
        self.save()

    # ---------------------------------------------------------------- G1
    def g1(self):
        c, rc_any = self.cfg, 0
        for label in G1_LABELS:
            argv, receipt = build_g1(c, label, self.plan[label], self.merged_path, c["run_dir"])
            _, pairs = parse_pairs(argv)
            inputs = {}
            for f, v in pairs:
                p = v.split("=", 1)[1] if f in ("--mixed-codeobject", "--negative-codeobject") else v
                if f in ("--codeobject", "--mq4-anchor", "--epilogue-codeobject", "--mixed-codeobject", "--negative-codeobject", "--probe-table",
                         "--divide-table", "--intrinsic-table"):
                    inputs[f"{f} {v}"] = sha256_file(p)
                elif f == "--sample-dir":
                    inputs[f"{f} {v}"] = sha256_file(os.path.join(v, "sample.json"))
                elif f == "--chain-dir":
                    inputs[f"{f} {v}"] = {n: sha256_file(os.path.join(v, n)) for n in ("chain-h.json", "chain-a4.json")}
            ent = {"label": label, "argv": argv, "plan_argv": self.plan[label], "receipt": receipt, "inputs": inputs, "started_utc": time.strftime("%FT%TZ", time.gmtime())}
            self.rec["g1"].append(ent)
            self.log(f"g1 {label}")
            t0 = time.monotonic()
            stem = os.path.join(c["run_dir"], "g1", label)
            try:
                rc = self.local(argv, stem, env=dict(os.environ))
            except Fail as e:
                ent.update({"rc": None, "error": str(e), "gate_ok": False})
                self.rec["g1_stop"] = {"label": label, "reason": str(e), "remaining": list(G1_LABELS)[list(G1_LABELS).index(label) + 1:]}
                self.save()
                return 2
            ent.update({"rc": rc, "secs": round(time.monotonic() - t0, 1), "gate_ok": False})
            ent["stdout_sha256"], ent["stderr_sha256"] = sha256_file(stem + ".stdout"), sha256_file(stem + ".stderr")
            with open(stem + ".stderr", "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 600))
                ent["stderr_tail"] = f.read().decode(errors="replace")
            if os.path.exists(receipt):
                raw = open(receipt, "rb").read()
                ent["receipt_sha256"] = sha256_bytes(raw)
                try:
                    ent.update(self.g1_summary(label, loads(raw, f"{label} receipt"), raw))
                except Fail as e:
                    ent["receipt_error"] = str(e)
            if not ent.get("gate_ok") or rc != 0:
                ent["gate_ok"] = False
                self.rec["g1_stop"] = {"label": label, "rc": rc, "receipt": receipt,
                    "first_positive_failure": ent.get("first_positive_failure"),
                    "remaining": list(G1_LABELS)[list(G1_LABELS).index(label) + 1:]}
                self.save()
                return 2
            self.save()
        return rc_any

    def g1_summary(self, label, d, raw):
        results = d.get("results", [])
        miss = [(r.get("name") or r.get("scenario"), r.get("hook_missing_keys"), r.get("hook_missing_samples")) for r in results if r.get("hook_missing_keys")]
        dev = d.get("device", {})
        s = {"pass": d.get("pass"), "complete": d.get("complete"), "status": d.get("status"), "g1c_verdict": (d.get("g1c") or {}).get("verdict"),
             "g1c_measured_pass": (d.get("g1c") or {}).get("measured_pass"), "device": dev, "cu_mask_env": (d.get("driver") or {}).get("cu_mask_env"),
             "hook_missing_total": sum(m[1] for m in miss), "first_missing": miss[0] if miss else None, "summary": d.get("summary"),
             "first_positive_failure": next((r for r in results if r.get("pass") is False), None),
             "binds_tables": {"base_probe": self.base_sha in raw.decode(errors="replace"), "intrinsic": (self.merged_sha or "") in raw.decode(errors="replace")}}
        gate = (d.get("pass") is True and d.get("complete") is True and dev.get("identity_match") is True and dev.get("arch") == ARCH
                and dev.get("uuid") == UUID and dev.get("bdf") == BDF and not miss and all(s["binds_tables"].values()))
        if label.startswith("real-mixed"):
            gate = gate and s["g1c_measured_pass"] is True
        s["gate_ok"] = gate
        return s

    # ---------------------------------------------------------------- driver
    def go(self, plan):
        self.plan, c = plan, self.cfg
        self.preflight()
        # Device observations come only from keys required by the planned CPU cases.
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
                               "iteration_hash": sha256_bytes(dumps([r.get("sha256") for r in self.rec["rounds"]]))}
        self.ship_chains(final)
        rc = self.g1()
        self.rec["g1_all_ok"] = all(e.get("gate_ok") for e in self.rec["g1"]) and len(self.rec["g1"]) == 8
        return 0 if self.rec["g1_all_ok"] and rc == 0 else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--check-config", metavar="CFG")
    ap.add_argument("--write-config", metavar="CFG")
    for f in ("commit", "cpu-repo", "gpu-repo", "cpu-oracle", "gpu-oracle", "gpu-observer", "gpu-probe-co", "gpu-probe-manifest", "run-dir", "cpu-work-dir"):
        ap.add_argument(f"--{f}")
    ap.add_argument("--real-plan", default=default_real_plan())
    ap.add_argument("--b4-root", default="/home/kaden/pm-wave/consumer24/b4")
    ap.add_argument("--probe-root", default="/home/kaden/pm-wave/consumer24/k2-oracle/probe")
    ap.add_argument("--cpu-c24", default="/home/kaden/qcal/release-0.4.3/consumer24/artifacts/c24-samples")
    ap.add_argument("--cpu-w3", default="/home/kaden/qcal/release-0.4.3/subq4-week/artifacts/ctl3-hot/dequant-samples")
    ap.add_argument("--cpu-base-probe", default="/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b4-obs/observations.json")
    ap.add_argument("--cpu-divide-table", default="/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b4-obs/divide-observations.json")
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
            need = ("commit", "cpu_repo", "gpu_repo", "cpu_oracle", "gpu_oracle", "gpu_observer", "gpu_probe_co", "gpu_probe_manifest", "run_dir", "cpu_work_dir")
            miss = [n for n in need if not getattr(a, n)]
            if miss:
                die(f"--write-config needs {['--' + m.replace('_', '-') for m in miss]}")
            return write_config(a) or 0
        if a.check_config:
            raw = open(a.check_config, "rb").read()
            plan = validate_config(loads(raw, a.check_config))
            print(json.dumps({"config_sha256": sha256_bytes(raw), "g1": sorted(plan), "ok": True}))
            return 0
        if not a.config or not a.out:
            ap.error("--config PATH --out RECEIPT (or --check-config / --write-config)")
        raw = open(a.config, "rb").read()
        cfg = loads(raw, a.config)
        plan = validate_config(cfg)
        if os.path.exists(a.out):
            die(f"receipt {a.out} already exists")
        run = Run(cfg, raw, os.path.abspath(a.out))
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
