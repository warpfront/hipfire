#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""GDN recurrent-state precision proof collector (developer-only, stdlib only).

Launches the real, hash-verified binaries of the gate commands in
``release-0.4.3/gdn-state-precision/PLAN.md`` section 5 (G0..G5), one fresh
process per arm, interleaved AB/BA, each in an isolated HOME with a scrubbed
``HIPFIRE_*`` environment (only the fabric/device trio and repeatable ``--child-env KEY=VALUE``
pairs pass; a parent ``HIPFIRE_*`` the scrub would drop is refused, never dropped silently), and persists the exact argv/env/identity/rc plus
the parsed observations. It never downloads anything, generates no teacher or
reference, starts no second runtime, pins no hidden route, and never infers a
PASS: every verdict field names the rule it applied.

    collect.py --stage smoke|quality|restore|memory|perf|sampled|greedy \\
        --binary-dir ABS --model ABS --wt2-ref ABS --code-ref ABS \\
        --modes q8,fp32[,q8_ef] --runs 3 --max-seq N --out ABS

See README.md in this directory for the stage -> gate map and hold sizing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

STAGES = ("smoke", "quality", "restore", "memory", "perf", "sampled", "greedy")
KNOWN_MODES = ("q8", "fp32", "q8_ef")
BASELINE_MODE = "q8"

#: Rule applied to every paired per-chunk KLD delta (candidate - baseline).
RULE_TEXT = "lo<=0 -> PASS; lo>0 -> PASS iff hi<=+0.010"
HI_BUDGET = 0.010
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261017
CLOSURE_TARGET = 0.75
PERF_POLICY_LINE = 0.05
MIN_SAMPLED_TRIALS = 2000

#: HIPFIRE_* variables that survive the child scrub.
KEEP_HIPFIRE_ENV = ("HIPFIRE_DEVICES", "HIPFIRE_LOCK_DIR", "HIPFIRE_FABRIC_ADMISSION")
FIXED_RECEIPT_KNOBS = (
    ("HIPFIRE_QWEN4_NPU_TRUNK", "off"),
    ("HIPFIRE_PLE_FORMAT", "q4"),
    ("HIPFIRE_NORMALIZE_PROMPT", "0"),
)
#: HIPFIRE_* variables the collector sets itself per child (never accepted through --child-env).
COLLECTOR_OWNED_HIPFIRE_ENV = (
    "HIPFIRE_HOME", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_GRAPH", "HIPFIRE_REPLAY_BACKEND",
    "HIPFIRE_STATE_QUANT", "HIPFIRE_GDN_TEST_STATE_QUANT",
)
#: Parent HIPFIRE_* prefixes the scrub may drop without a refusal: host Python tool paths set in user
#: shells; nothing in the tree reads them.
IGNORED_PARENT_HIPFIRE_PREFIXES = ("HIPFIRE_TORCH_",)
_CHILD_ENV_KEY = re.compile(r"^HIPFIRE_[A-Z0-9_]+$")

HOSTMEM_LOCK = "hostmem-flashnext.lock"
MANIFEST_NAME = "BINARIES.sha256"

#: Executable base names (resolved through the manifest; test executables
#: carry a cargo ``-<hash>`` suffix).
BIN_QWEN4_KLD = "qwen4_kld"
BIN_HIPFIRE = "hipfire"
BIN_DAEMON = "daemon"
BIN_ORACLE = "gdn_state_precision_oracle"
BIN_IDENTITY = "greedy_mtp_identity_hw"
BIN_SESSION = "session_cache_hw"
BIN_SAMPLED = "sampled_mtp_distribution_hw"
BIN_EF_ROUTES = "q8_ef_product_routes_hw"

TEST_IDENTITY_SEASONS = "greedy_mtp_matches_ar_on_flash_next_seasons"
TEST_IDENTITY_FILTER = "greedy_mtp_matches_ar_on_flash_next"
TEST_SAMPLED_AR = "sampled_mtp_matches_ar_distribution_on_flash_next"
TEST_SAMPLED_EF = "sampled_mtp_q8ef_matches_verify_distribution_on_flash_next"

#: Every ``#[ignore]`` test of ``session_cache_hw`` (its ``--list --ignored`` on the A-de86803d7 gate bins). Each
#: test loads Flash-Next, and the test file's contract is ONE test per process (two loads in one process exhaust
#: host memory: the Halo host OOM of 2026-10-17 10:14Z, j1735, ran all of them in one process). The restore stage
#: runs one ``--exact`` child per name, refuses any session_cache_hw argv without ``--exact <one of these>``, and
#: before any GPU work refuses a binary whose ignored-test list differs from this tuple.
SESSION_CACHE_TESTS = (
    "disk_snapshots_restore_after_restart_on_flash_next",
    "message_end_snapshots_restore_on_flash_next",
    "resident_restore_matches_cold_on_flash_next",
    "restored_prefill_matches_cold_on_flash_next",
    "retired_turn_commits_live_on_flash_next",
    "tail_snapshot_restore_matches_split_cold_on_flash_next",
    "terminal_stop_inside_window_repairs_live_on_flash_next",
)

#: Counters a q8_ef sampled-law receipt must carry (each > 0). The two graph
#: counters are required only when the arm ran with HIPFIRE_REPLAY_BACKEND=auto (--graph 1). They are
#: read from any JSON under HIPFIRE_SAMPLED_MTP_OUT with a top-level
#: ``{"coverage": {<counter>: int}}`` object.
Q8EF_REQUIRED_COVERAGE = (
    "reject_keep0",
    "reject_partial",
    "full_accept",
    "post_reset_window",
    "restore_transaction",
    "restore_session",
    "restore_prefix",
    "restore_disk",
    "ring_half_0",
    "ring_half_1",
)
Q8EF_GRAPH_COVERAGE = ("graph_capture", "graph_replay")

DEFAULT_GRAPHS = {
    "smoke": ("0", "1"),
    "quality": ("0",),
    "restore": ("0", "1"),
    "memory": ("unset",),
    "perf": ("unset",),
    "sampled": ("unset",),
    "greedy": ("1",),
}
DEFAULT_RUNS = {
    "smoke": 1,
    "quality": 3,
    "restore": 3,
    "memory": 1,
    "perf": 3,
    "sampled": 3,
    "greedy": 1,
}

#: ``--max-chunks`` of the quality stage (paired analysis needs >= 2 chunks).
DEFAULT_MAX_CHUNKS = 24
MIN_MAX_CHUNKS = 2

#: Greedy kill experiment: ``hipfire run --spec`` arms compared per (run, graph, mode).
GREEDY_SPECS = ("off", "mtp")
DEFAULT_GREEDY_TOKENS = 64
MAX_GREEDY_TOKENS = 4096
#: Linux MAX_ARG_STRLEN (32 pages): one argv string must be shorter than this.
MAX_ARG_STRLEN = 131072
#: Daemon log lines the greedy analysis keeps per child (admission evidence).
GREEDY_ADMISSION_KEYS = ("max_seq", "card_cap", "admission")
GPU_STAGES = STAGES

QUALITY_CORPORA = ("code24", "wt2")
#: (route label, extra qwen4_kld args, output file name)
QUALITY_ROUTES = (
    ("decode", ("--decode",), "out.kldseq"),
    ("append256", ("--append-rows", "256"), "out.a256.kldseq"),
    ("whole", (), "out.kldseq"),
)

PERF_LABELS = (
    "stateless-spec-off",
    "stateless-spec-mtp",
    "ttft-long",
    "matrix-pp8192-tg32",
)

DAEMON_LOG_KEYS = (
    "qwen4 state:",
    "GDN",
    "partition_digest",
    "placement",
    "max_seq",
    "card_cap",
    "R=",
    "ring",
    "admission",
    "peak",
)


class CollectError(Exception):
    """A refusal: bad arguments, missing inputs, no hold, bad hash."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_json(path: Path):
    return json.loads(path.read_text())


def require_abs(label: str, value: str) -> Path:
    if not value:
        raise CollectError(f"{label} is required")
    path = Path(value)
    if not path.is_absolute():
        raise CollectError(f"{label} must be an absolute path, got {value!r}")
    return path


def require_file(label: str, value: str) -> Path:
    path = require_abs(label, value)
    if not path.is_file():
        raise CollectError(f"{label} {path} is not an existing file")
    return path


def hash_file_once(path: Path) -> dict:
    """MD5 + SHA-256 in one pass."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            md5.update(block)
            sha.update(block)
            size += len(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "size": size}


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            sha.update(block)
    return sha.hexdigest()


class Hasher:
    """Memoised MD5/SHA-256 with an optional on-disk cache keyed by
    (path, size, mtime_ns), so tens-of-GiB inputs are hashed once, outside a
    hold if desired (``--prehash``)."""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path
        self.memo: dict[str, dict] = {}
        self.cache: dict[str, dict] = {}
        if cache_path is not None and cache_path.is_file():
            try:
                self.cache = read_json(cache_path)
            except (OSError, ValueError):
                self.cache = {}

    def hash(self, path: Path) -> dict:
        key = str(path)
        if key in self.memo:
            return self.memo[key]
        st = path.stat()
        stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        cached = self.cache.get(key)
        if cached and cached.get("size") == stamp["size"] and cached.get("mtime_ns") == stamp["mtime_ns"]:
            result = {"md5": cached["md5"], "sha256": cached["sha256"], "size": stamp["size"], "cached": True}
        else:
            result = hash_file_once(path)
            result["cached"] = False
            self.cache[key] = {"md5": result["md5"], "sha256": result["sha256"], **stamp}
            if self.cache_path is not None:
                write_json(self.cache_path, self.cache)
        result["path"] = key
        self.memo[key] = result
        return result


# ---------------------------------------------------------------------------
# Binary manifest
# ---------------------------------------------------------------------------

_HEX_SUFFIX = re.compile(r"^-[0-9a-f]{8,}$")


def read_manifest(binary_dir: Path) -> dict[str, str]:
    path = binary_dir / MANIFEST_NAME
    if not path.is_file():
        raise CollectError(f"{binary_dir} has no {MANIFEST_NAME} (lines: '<sha256>  <name>')")
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise CollectError(f"{path}:{number}: expected '<sha256>  <name>', got {line!r}")
        name = parts[1].strip().lstrip("*")
        if "/" in name or name in ("", ".", ".."):
            raise CollectError(f"{path}:{number}: manifest name {name!r} must be a bare file name")
        if name in entries:
            raise CollectError(f"{path}:{number}: duplicate manifest name {name}")
        entries[name] = parts[0]
    return entries


def resolve_binary(manifest: dict[str, str], base: str) -> str:
    """The unique manifest name equal to ``base`` or ``base-<hex hash>``."""
    matches = [n for n in manifest if n == base or (n.startswith(base) and _HEX_SUFFIX.match(n[len(base):]))]
    if not matches:
        raise CollectError(f"{MANIFEST_NAME} lists no executable for {base!r}")
    if len(matches) > 1:
        raise CollectError(f"{MANIFEST_NAME} is ambiguous for {base!r}: {sorted(matches)}")
    return matches[0]


def required_binaries(stage: str, modes: tuple[str, ...], oracle_only: bool = False) -> tuple[str, ...]:
    ef = "q8_ef" in modes
    classic = any(m != "q8_ef" for m in modes)
    if stage == "smoke":
        bins = [BIN_ORACLE]
        if classic and not oracle_only:
            bins.append(BIN_IDENTITY)
        return tuple(bins)
    if stage == "quality":
        return (BIN_QWEN4_KLD,)
    if stage == "restore":
        bins: list[str] = []
        if classic:
            bins += [BIN_SESSION, BIN_IDENTITY]
        if ef:
            bins.append(BIN_EF_ROUTES)
        return tuple(bins)
    if stage in ("memory", "perf", "greedy"):
        return (BIN_HIPFIRE, BIN_DAEMON)
    if stage == "sampled":
        return (BIN_SAMPLED,)
    raise CollectError(f"unknown stage {stage!r}")


def resolve_and_verify_binaries(binary_dir: Path, bases: Iterable[str]) -> dict[str, dict]:
    """base -> {path, name, sha256}; every file exists and matches its manifest hash."""
    manifest = read_manifest(binary_dir)
    resolved: dict[str, dict] = {}
    for base in bases:
        name = resolve_binary(manifest, base)
        path = binary_dir / name
        if not path.is_file():
            raise CollectError(f"{path} is listed in {MANIFEST_NAME} but is not a file")
        if not os.access(path, os.X_OK):
            raise CollectError(f"{path} is not executable")
        actual = sha256_file(path)
        if actual != manifest[name]:
            raise CollectError(f"{path} sha256 {actual} != manifest {manifest[name]}")
        resolved[base] = {"path": path, "name": name, "sha256": actual}
    return resolved


# ---------------------------------------------------------------------------
# Fabric hold verification (mirrors session_cache_hw.rs hold_refusal)
# ---------------------------------------------------------------------------


def proc_stat(pid: int, index: int) -> Optional[int]:
    """/proc/<pid>/stat fields after the command: 1 = ppid, 19 = starttime."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return int(raw[raw.rindex(")") + 2 :].split()[index])
    except (OSError, ValueError, IndexError):
        return None


def own_chain() -> list[int]:
    chain = [os.getpid()]
    while True:
        ppid = proc_stat(chain[-1], 1)
        if not ppid or ppid <= 1:
            return chain
        chain.append(ppid)


def hostmem_lock_holders(pids: Iterable[int]) -> list[tuple[int, str]]:
    try:
        locks = Path("/proc/locks").read_text().splitlines()
    except OSError:
        locks = []
    held: list[tuple[int, str]] = []
    for pid in pids:
        try:
            fds = list(Path(f"/proc/{pid}/fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
                if os.path.basename(target) != HOSTMEM_LOCK:
                    continue
                inode = f":{os.stat(target).st_ino}"
            except OSError:
                continue
            for line in locks:
                f = line.split()
                if (
                    len(f) >= 6
                    and "->" not in f
                    and f[1] == "FLOCK"
                    and f[4] == str(pid)
                    and f[5].endswith(inode)
                    and all(p != pid for p, _ in held)
                ):
                    held.append((pid, f[3]))
    return held


def hold_refusal(
    marker: str,
    chain: list[int],
    start_of: Callable[[int], Optional[int]],
    holds_lock: Callable[[list[int]], bool],
    require_lock: bool = True,
) -> Optional[str]:
    """Why this process is not inside its own fabric hold, or ``None``."""
    parts = marker.split(":", 2)
    try:
        wrapper = int(parts[0])
        start = int(parts[1])
    except (ValueError, IndexError):
        return "no fabric hold (FABRIC_JOB unset or not 'pid:starttime:cards')"
    if wrapper not in chain:
        return f"fabric wrapper {wrapper} is not an ancestor of this process"
    if start_of(wrapper) != start:
        return f"fabric wrapper {wrapper} is gone (stale FABRIC_JOB)"
    if require_lock and not holds_lock(chain[: chain.index(wrapper) + 1]):
        return f"fabric job {marker} does not hold {HOSTMEM_LOCK} (no --hostmem-lock)"
    return None


def require_fabric_hold(environ: dict, require_lock: bool = True) -> None:
    why = hold_refusal(
        environ.get("FABRIC_JOB", ""),
        own_chain(),
        lambda pid: proc_stat(pid, 19),
        lambda pids: bool(hostmem_lock_holders(pids)),
        require_lock,
    )
    if why:
        raise CollectError(
            f"{why}. GPU stages run only inside a fabric hold: "
            "fabric-run.sh ... --hostmem-lock -- python3 tools/gdn_state_precision/collect.py ..."
        )


# ---------------------------------------------------------------------------
# Child model
# ---------------------------------------------------------------------------


@dataclass
class Config:
    stage: str
    binary_dir: Path
    model: Path
    wt2_ref: Path
    code_ref: Path
    modes: tuple[str, ...]
    runs: int
    max_seq: int
    out: Path
    run_id: str
    graphs: tuple[str, ...]
    prompt_file: Optional[Path] = None
    long_prompt_file: Optional[Path] = None
    child_timeout_secs: int = 1700
    sampled_trials: int = MIN_SAMPLED_TRIALS
    bench_max_seq: bool = False
    keep_going: bool = False
    oracle_only: bool = False
    dry_run: bool = False
    max_chunks: int = DEFAULT_MAX_CHUNKS
    greedy_tokens: int = DEFAULT_GREEDY_TOKENS
    only: tuple[str, ...] = ()
    child_env: dict = field(default_factory=dict)  # --child-env KEY=VALUE, in the order given
    binaries: dict = field(default_factory=dict)  # base -> {path,name,sha256}

    @property
    def stage_dir(self) -> Path:
        return self.out / self.stage

    @property
    def run_dir(self) -> Path:
        return self.out / self.stage / self.run_id

    def exe(self, base: str) -> str:
        return str(self.binaries[base]["path"])


@dataclass
class Child:
    id: str
    label: str
    kind: str
    run: int
    graph: str
    mode: str
    dir: Path
    argv: list[str]
    extra_env: dict
    binary: str
    inputs: list[tuple[str, Path]] = field(default_factory=list)
    expect_tests: tuple[str, ...] = ()
    corpus: Optional[str] = None
    route: Optional[str] = None
    kldseq: Optional[str] = None
    ef_sampled: bool = False
    sample_memory: bool = False
    #: Files written under the child's isolated HIPFIRE_HOME (``<dir>/home/.hipfire``) before launch.
    home_files: dict = field(default_factory=dict)


def arm_order(modes: tuple[str, ...], run: int) -> tuple[str, ...]:
    """Run i even: modes as given; odd: reversed (AB/BA)."""
    return tuple(modes) if run % 2 == 0 else tuple(reversed(modes))


def build_child_env(parent: dict, cfg: Config, child: Child) -> dict:
    """Scrubbed, isolated child environment."""
    env = {
        k: v
        for k, v in parent.items()
        if not k.startswith("HIPFIRE_") or k in KEEP_HIPFIRE_ENV
    }
    for key, value in FIXED_RECEIPT_KNOBS:
        env[key] = value
    env.update(cfg.child_env)
    # The graph axis (HIPFIRE_GRAPH or HIPFIRE_REPLAY_BACKEND) lives in extra_env.
    env["HIPFIRE_KERNEL_CACHE"] = str(cfg.out / "kcache")
    home = child.dir / "home"
    env["HOME"] = str(home)
    env["HIPFIRE_HOME"] = str(home / ".hipfire")
    env["HIPFIRE_LOCK_DIR"] = str(child.dir / "lockdir")
    env.update(child.extra_env)
    return env


def explicit_env(cfg: Config, child: Child) -> dict:
    """Only the variables the collector sets (for dry-run display)."""
    shown = {key: value for key, value in FIXED_RECEIPT_KNOBS}
    shown.update(cfg.child_env)
    # The graph axis (HIPFIRE_GRAPH or HIPFIRE_REPLAY_BACKEND) lives in extra_env.
    shown["HIPFIRE_KERNEL_CACHE"] = str(cfg.out / "kcache")
    shown["HOME"] = str(child.dir / "home")
    shown["HIPFIRE_HOME"] = str(child.dir / "home" / ".hipfire")
    shown["HIPFIRE_LOCK_DIR"] = str(child.dir / "lockdir")
    shown.update(child.extra_env)
    return shown


#: Child kinds whose graph axis is the Qwen4 test harness replay backend
#: (production Qwen4 has no hipGraph capture): "0" -> eager hip, "1" -> auto
#: (retained AR replay). Receipts label it ``replay_backend``, never hipGraph.
REPLAY_KINDS = ("identity_test", "session_test", "ef_routes_test", "sampled_test")
REPLAY_BACKEND_OF_GRAPH = {"0": "hip", "1": "auto"}


def graph_axis(kind: str) -> str:
    if kind == "oracle":
        return "oracle --graph"
    return "HIPFIRE_REPLAY_BACKEND" if kind in REPLAY_KINDS else "HIPFIRE_GRAPH"


def graph_env(kind: str, graph: str) -> dict:
    """Explicit graph-axis env of one arm (``unset`` sets nothing; the oracle
    takes ``--graph`` on its argv instead)."""
    if graph == "unset" or kind == "oracle":
        return {}
    if kind in REPLAY_KINDS:
        return {"HIPFIRE_REPLAY_BACKEND": REPLAY_BACKEND_OF_GRAPH[graph]}
    return {"HIPFIRE_GRAPH": graph}


def _child(
    cfg: Config, run: int, graph: str, mode: str, label: str, kind: str, argv, extra_env, binary,
    sub: Optional[str] = None, **kw,
) -> Child:
    """``sub`` names one of several children sharing a label (own dir and id; ``--only`` still matches the label)."""
    path = cfg.run_dir / f"run{run}" / f"g{graph}" / (mode if mode else "-") / label
    cid = f"run{run}/g{graph}/{mode}/{label}"
    if sub:
        path, cid = path / sub, f"{cid}/{sub}"
    return Child(
        id=cid,
        label=label,
        kind=kind,
        run=run,
        graph=graph,
        mode=mode,
        dir=path,
        argv=[str(a) for a in argv],
        extra_env={**graph_env(kind, graph), **extra_env},
        binary=binary,
        **kw,
    )


def _arms(cfg: Config):
    for run in range(cfg.runs):
        for graph in cfg.graphs:
            for mode in arm_order(cfg.modes, run):
                yield run, graph, mode


def plan_smoke(cfg: Config) -> list[Child]:
    children: list[Child] = []
    for run in range(cfg.runs):
        for graph in cfg.graphs:
            c = _child(
                cfg, run, graph, ",".join(cfg.modes), "oracle", "oracle", [], {}, BIN_ORACLE,
            )
            c.argv = [
                cfg.exe(BIN_ORACLE), "--formats", ",".join(cfg.modes), "--graph", graph if graph != "unset" else "0",
                "--max-rows", "32", "--all-keeps", "--json", str(c.dir / "oracle.json"),
            ]
            if graph == "unset":
                raise CollectError("smoke needs an explicit --graph 0 or 1 (the oracle takes --graph)")
            children.append(c)
            if cfg.oracle_only:
                continue
            for mode in arm_order(cfg.modes, run):
                if mode == "q8_ef":
                    continue
                children.append(
                    _child(
                        cfg, run, graph, mode, "identity-seasons", "identity_test",
                        [cfg.exe(BIN_IDENTITY), "--ignored", "--test-threads=1", "--exact", TEST_IDENTITY_SEASONS],
                        {
                            "HIPFIRE_GDN_TEST_STATE_QUANT": mode,
                            "HIPFIRE_MTP_IDENTITY_MODEL": str(cfg.model),
                            "HIPFIRE_MTP_IDENTITY_OUT": "@CHILD@/out",
                        },
                        BIN_IDENTITY,
                        expect_tests=(TEST_IDENTITY_SEASONS,),
                        inputs=[("model", cfg.model)],
                    )
                )
    return children


def plan_quality(cfg: Config) -> list[Child]:
    refs = {"code24": cfg.code_ref, "wt2": cfg.wt2_ref}
    children: list[Child] = []
    for run, graph, mode in _arms(cfg):
        for corpus in QUALITY_CORPORA:
            for route, flags, out_name in QUALITY_ROUTES:
                label = f"{corpus}-{route}"
                c = _child(
                    cfg, run, graph, mode, label, "kld_eval", [],
                    {"HIPFIRE_STATE_QUANT": mode}, BIN_QWEN4_KLD,
                    corpus=corpus, route=route, kldseq=out_name,
                    inputs=[("model", cfg.model), (f"{corpus}-ref", refs[corpus])],
                )
                c.argv = [
                    cfg.exe(BIN_QWEN4_KLD), "eval", "--model", str(cfg.model), "--ref", str(refs[corpus]),
                    "--output", str(c.dir / "out.kldseq"), "--max-chunks", str(cfg.max_chunks), *flags,
                ]
                children.append(c)
    return children


def plan_restore(cfg: Config) -> list[Child]:
    children: list[Child] = []
    for run, graph, mode in _arms(cfg):
        env = {
            "HIPFIRE_GDN_TEST_STATE_QUANT": mode,
            "HIPFIRE_MTP_IDENTITY_MODEL": str(cfg.model),
            "HIPFIRE_SESSION_CACHE_MODEL": str(cfg.model),
            "HIPFIRE_MTP_IDENTITY_OUT": "@CHILD@/out",
        }
        if mode == "q8_ef":
            children.append(
                _child(
                    cfg, run, graph, mode, "ef-product-routes", "ef_routes_test",
                    [cfg.exe(BIN_EF_ROUTES), "--ignored", "--nocapture", "--test-threads=1"],
                    env, BIN_EF_ROUTES, inputs=[("model", cfg.model)],
                )
            )
            continue
        # One process per test: each loads Flash-Next (see SESSION_CACHE_TESTS).
        session_env = {k: v for k, v in env.items() if k != "HIPFIRE_MTP_IDENTITY_OUT"}
        for name in SESSION_CACHE_TESTS:
            children.append(
                _child(
                    cfg, run, graph, mode, "session-cache", "session_test",
                    [cfg.exe(BIN_SESSION), "--ignored", "--test-threads=1", "--exact", name],
                    session_env, BIN_SESSION, sub=name, expect_tests=(name,),
                    inputs=[("model", cfg.model)],
                )
            )
        children.append(
            _child(
                cfg, run, graph, mode, "identity", "identity_test",
                [cfg.exe(BIN_IDENTITY), "--ignored", "--test-threads=1", TEST_IDENTITY_FILTER],
                {k: v for k, v in env.items() if k != "HIPFIRE_SESSION_CACHE_MODEL"},
                BIN_IDENTITY, inputs=[("model", cfg.model)],
            )
        )
    return children


def bench_base(cfg: Config) -> list[str]:
    return [cfg.exe(BIN_HIPFIRE), "bench", str(cfg.model)]


def bench_tail(cfg: Config, warmups: int) -> list[str]:
    tail = ["--runs", "1", "--warmups", str(warmups), "--ctx", str(cfg.max_seq)]
    if cfg.bench_max_seq:
        tail += ["--max-seq", str(cfg.max_seq)]
    return tail + ["--json"]


def bench_env(cfg: Config, mode: str) -> dict:
    return {"HIPFIRE_STATE_QUANT": mode, "HIPFIRE_DAEMON_BIN": cfg.exe(BIN_DAEMON)}


def plan_memory(cfg: Config) -> list[Child]:
    children: list[Child] = []
    assert cfg.prompt_file is not None
    for run, graph, mode in _arms(cfg):
        argv = bench_base(cfg) + [
            "--backend", "noslots", "--workload", "stateless", "--spec", "mtp",
            "--prompt-file", str(cfg.prompt_file), "--max-tokens", "32",
            "--runs", "1", "--warmups", "0", "--ctx", str(cfg.max_seq),
        ]
        if cfg.bench_max_seq:
            argv += ["--max-seq", str(cfg.max_seq)]
        argv.append("--json")
        children.append(
            _child(
                cfg, run, graph, mode, "memory-mtp-ctx", "bench_memory", argv,
                bench_env(cfg, mode), BIN_HIPFIRE,
                inputs=[("model", cfg.model), ("prompt", cfg.prompt_file)],
                sample_memory=True,
            )
        )
    return children


def plan_perf(cfg: Config) -> list[Child]:
    children: list[Child] = []
    assert cfg.prompt_file is not None and cfg.long_prompt_file is not None
    for run, graph, mode in _arms(cfg):
        specs = {
            "stateless-spec-off": (
                "bench_standard",
                bench_base(cfg)
                + ["--backend", "noslots", "--workload", "stateless", "--spec", "off",
                   "--prompt-file", str(cfg.prompt_file), "--max-tokens", "256"]
                + bench_tail(cfg, 1),
                cfg.prompt_file,
            ),
            "stateless-spec-mtp": (
                "bench_standard",
                bench_base(cfg)
                + ["--backend", "noslots", "--workload", "stateless", "--spec", "mtp",
                   "--prompt-file", str(cfg.prompt_file), "--max-tokens", "256"]
                + bench_tail(cfg, 1),
                cfg.prompt_file,
            ),
            "ttft-long": (
                "bench_ttft",
                bench_base(cfg) + ["--ttft", "--prompt-file", str(cfg.long_prompt_file)] + bench_tail(cfg, 1),
                cfg.long_prompt_file,
            ),
            "matrix-pp8192-tg32": (
                "bench_matrix",
                bench_base(cfg) + ["--matrix", "--pp", "8192", "--tg", "32"] + bench_tail(cfg, 1),
                None,
            ),
        }
        for label in PERF_LABELS:
            kind, argv, prompt = specs[label]
            inputs = [("model", cfg.model)]
            if prompt is not None:
                inputs.append(("prompt", prompt))
            children.append(
                _child(cfg, run, graph, mode, label, kind, argv, bench_env(cfg, mode), BIN_HIPFIRE, inputs=inputs)
            )
    return children


def plan_sampled(cfg: Config) -> list[Child]:
    children: list[Child] = []
    for run, graph, mode in _arms(cfg):
        ef = mode == "q8_ef"
        case = TEST_SAMPLED_EF if ef else TEST_SAMPLED_AR
        children.append(
            _child(
                cfg, run, graph, mode, "sampled-law" if ef else "sampled-ar", "sampled_test",
                [cfg.exe(BIN_SAMPLED), "--ignored", "--nocapture", "--test-threads=1", "--exact", case],
                {
                    "HIPFIRE_GDN_TEST_STATE_QUANT": mode,
                    "HIPFIRE_MTP_IDENTITY_MODEL": str(cfg.model),
                    "HIPFIRE_SAMPLED_MTP_TRIALS": str(cfg.sampled_trials),
                    "HIPFIRE_SAMPLED_MTP_OUT": "@CHILD@/sampled",
                },
                BIN_SAMPLED,
                expect_tests=(case,),
                ef_sampled=ef,
                inputs=[("model", cfg.model)],
            )
        )
    return children


def read_prompt_text(path: Path) -> str:
    """Greedy prompt: the file as UTF-8 text with trailing newlines stripped."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CollectError(f"--prompt-file {path} is not readable UTF-8 text: {exc}") from exc
    return text.rstrip("\n")


#: The greedy arms compare visible text, so thinking is off: with the Flash-Next default (on) a short
#: budget is all reasoning and `hipfire run --json` reports empty content (j1724). `reasoning.mode` has no
#: env knob; the child's isolated global config sets it.
GREEDY_HOME_CONFIG = '[reasoning]\nmode = "off"\n'


def plan_greedy(cfg: Config) -> list[Child]:
    """P0 kill experiment: greedy ``hipfire run`` (thinking off) with MTP off vs on, per (run, graph, mode)."""
    assert cfg.prompt_file is not None
    prompt = read_prompt_text(cfg.prompt_file)
    children: list[Child] = []
    for run, graph, mode in _arms(cfg):
        for spec in GREEDY_SPECS:
            argv = [
                cfg.exe(BIN_HIPFIRE), "run", str(cfg.model), "--temp", "0", "--spec", spec,
                "--max-tokens", str(cfg.greedy_tokens), "--max-seq", str(cfg.max_seq), "--json", prompt,
            ]
            children.append(
                _child(
                    cfg, run, graph, mode, f"greedy-spec-{spec}", "greedy_run", argv,
                    bench_env(cfg, mode), BIN_HIPFIRE,
                    inputs=[("model", cfg.model), ("prompt", cfg.prompt_file)],
                    sample_memory=True,
                    home_files={"config.toml": GREEDY_HOME_CONFIG},
                )
            )
    return children


PLANNERS = {
    "smoke": plan_smoke,
    "quality": plan_quality,
    "restore": plan_restore,
    "memory": plan_memory,
    "perf": plan_perf,
    "sampled": plan_sampled,
    "greedy": plan_greedy,
}


def plan_children(cfg: Config) -> list[Child]:
    """The ordered child processes of a stage, with ``@CHILD@`` resolved and ``--only`` applied."""
    children = PLANNERS[cfg.stage](cfg)
    if cfg.only:
        planned = {c.label for c in children}
        unknown = [label for label in cfg.only if label not in planned]
        if unknown:
            raise CollectError(
                f"--only names no planned child of stage {cfg.stage}: {', '.join(unknown)} "
                f"(planned labels: {', '.join(sorted(planned))})"
            )
        children = [c for c in children if c.label in cfg.only]
        if not children:
            raise CollectError(f"--only {','.join(cfg.only)} leaves no child to run in stage {cfg.stage}")
    for c in children:
        c.extra_env = {k: v.replace("@CHILD@", str(c.dir)) for k, v in c.extra_env.items()}
        require_one_session_test(c)
    return children


def require_one_session_test(child: Child) -> None:
    """session_cache_hw runs exactly one ``--exact`` test per process (SESSION_CACHE_TESTS)."""
    if child.binary != BIN_SESSION:
        return
    argv = child.argv
    i = argv.index("--exact") if argv.count("--exact") == 1 else -1
    if i < 0 or i + 1 >= len(argv) or argv[i + 1] not in SESSION_CACHE_TESTS:
        raise CollectError(
            f"{child.id}: session_cache_hw must run one known test per process (--exact <test>): every test loads "
            f"Flash-Next and two loads in one process exhaust host memory; argv {shlex.join(argv)}"
        )


def check_session_test_list(cfg: Config, children: list[Child]) -> None:
    """Before any GPU work: the session_cache_hw binary's ignored tests must be exactly SESSION_CACHE_TESTS
    (``--list`` touches no GPU); a test the plan does not know would otherwise go unrun."""
    if not any(c.binary == BIN_SESSION for c in children):
        return
    argv = [cfg.exe(BIN_SESSION), "--list", "--ignored", "--format", "terse"]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise CollectError(f"cannot list session_cache_hw tests ({shlex.join(argv)}): {exc}") from exc
    listed = {line[: -len(": test")] for line in out.splitlines() if line.endswith(": test")}
    if listed != set(SESSION_CACHE_TESTS):
        raise CollectError(
            "session_cache_hw ignored tests differ from SESSION_CACHE_TESTS: "
            f"unknown {sorted(listed - set(SESSION_CACHE_TESTS))}, missing {sorted(set(SESSION_CACHE_TESTS) - listed)}"
        )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(m.strip() for m in raw.split(",") if m.strip())
    if not modes:
        raise CollectError("--modes is empty")
    for m in modes:
        if m not in KNOWN_MODES:
            raise CollectError(f"unknown mode {m!r}; known: {', '.join(KNOWN_MODES)}")
    if len(set(modes)) != len(modes):
        raise CollectError(f"--modes repeats a mode: {raw}")
    return modes


def parse_graphs(raw: str) -> tuple[str, ...]:
    graphs = tuple(g.strip() for g in raw.split(",") if g.strip())
    if not graphs:
        raise CollectError("--graph is empty")
    for g in graphs:
        if g not in ("0", "1", "unset"):
            raise CollectError(f"--graph value {g!r} must be 0, 1 or unset")
    if len(set(graphs)) != len(graphs):
        raise CollectError(f"--graph repeats a value: {raw}")
    return graphs


def parse_only(raw: str) -> tuple[str, ...]:
    labels = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not labels:
        raise CollectError("--only is empty")
    if len(set(labels)) != len(labels):
        raise CollectError(f"--only repeats a label: {raw}")
    return labels


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect.py",
        description=(
            "GDN state-precision proof collector: launches the hash-verified gate binaries "
            "(PLAN section 5 G0-G5) inside a fabric hold and persists argv/env/identity/rc/observations."
        ),
    )
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--binary-dir", help="ABS dir holding the executables and BINARIES.sha256 ('<sha256>  <name>')")
    p.add_argument("--model", help="ABS model file (.hfq/.mq4); .ple-q4/.ple-q8 sidecars are hashed when present")
    p.add_argument("--wt2-ref", help="ABS wikitext2 .kldref teacher")
    p.add_argument("--code-ref", help="ABS code24 .kldref teacher")
    p.add_argument("--modes", help="comma list of q8,fp32,q8_ef (q8 is the paired baseline)")
    p.add_argument("--runs", type=int, help="fresh runs per arm (default: 3 quality/restore/perf/sampled, 1 smoke/memory)")
    p.add_argument("--max-seq", type=int, help="context N for bench --ctx / memory declared context; stage greedy: `hipfire run --max-seq N`")
    p.add_argument("--out", help="ABS output root; writes <out>/<stage>/<run-id>/")
    p.add_argument("--dry-run", action="store_true", help="print the exact commands; execute nothing; no hold needed")
    p.add_argument("--analyze", action="store_true", help="CPU-only: recompute quality/perf statistics from an existing --out run dir")
    p.add_argument("--prehash", action="store_true", help="CPU-only: fill <out>/hash-cache.json with MD5+SHA256 of every input (outside a hold)")
    p.add_argument("--run-id", help="run directory name (default: UTC timestamp+pid); with --analyze: the run to analyse (default: latest)")
    p.add_argument("--prompt-file", help="ABS prompt file for stage memory, perf and greedy (frozen committed prompt; greedy reads it as UTF-8 text, trailing newlines stripped)")
    p.add_argument("--long-prompt-file", help="ABS long prompt file for the perf --ttft command")
    p.add_argument(
        "--graph",
        help="comma list of 0,1,unset per arm. quality/memory/perf: HIPFIRE_GRAPH; identity/session/sampled/EF hw tests: "
        "HIPFIRE_REPLAY_BACKEND (0=hip eager, 1=auto retained replay; recorded as replay_backend, not hipGraph); "
        "oracle: --graph. unset sets nothing (product default). "
        "Defaults: smoke 0,1; quality 0; restore 0,1; memory/perf/sampled unset; greedy 1",
    )
    p.add_argument("--child-timeout-secs", type=int, default=1700, help="per-child GPU timeout (default 1700)")
    p.add_argument("--sampled-trials", type=int, default=MIN_SAMPLED_TRIALS, help="HIPFIRE_SAMPLED_MTP_TRIALS (>= 2000)")
    p.add_argument("--bench-max-seq", action="store_true", help="also pass `--max-seq N` to hipfire bench (plan commands pass only --ctx N)")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed child (default: stop at the first failure)")
    p.add_argument(
        "--oracle-only",
        action="store_true",
        help="stage smoke: run only the gdn_state_precision_oracle invocations (no Flash-Next load: the hold guard "
        "still requires the FABRIC_JOB wrapper ancestor but not hostmem-flashnext.lock; the model is not hashed)",
    )
    p.add_argument(
        "--only",
        help="LABEL[,LABEL...] (all stages): run only the planned children with these labels, in planned order; "
        "an unknown label is refused. Labels: smoke oracle,identity-seasons; quality <code24|wt2>-<decode|append256|whole>; "
        "restore session-cache,identity,ef-product-routes; memory memory-mtp-ctx; "
        "perf stateless-spec-off,stateless-spec-mtp,ttft-long,matrix-pp8192-tg32; sampled sampled-ar,sampled-law; "
        "greedy greedy-spec-off,greedy-spec-mtp",
    )
    p.add_argument(
        "--max-chunks", type=int, default=None,
        help=f"stage quality only: qwen4_kld eval --max-chunks N (default {DEFAULT_MAX_CHUNKS}, >= {MIN_MAX_CHUNKS}); "
        "paired analysis needs equal chunk counts between arms",
    )
    p.add_argument(
        "--greedy-tokens", type=int, default=None,
        help=f"stage greedy only: `hipfire run --max-tokens N` (default {DEFAULT_GREEDY_TOKENS}, 1..{MAX_GREEDY_TOKENS})",
    )
    p.add_argument(
        "--child-env", action="append", default=[], metavar="KEY=VALUE",
        help="passes KEY=VALUE into every child's scrubbed environment and records it in the receipts; repeatable; "
        "e.g. --child-env HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14 for the R9700 expert placement "
        "(KEY must be HIPFIRE_[A-Z0-9_]+; keys the collector sets itself are refused)",
    )
    return p


def parse_child_env(values: list[str]) -> dict:
    """``--child-env KEY=VALUE`` pairs -> ordered dict; refuses what the collector owns."""
    owned = set(KEEP_HIPFIRE_ENV) | set(COLLECTOR_OWNED_HIPFIRE_ENV) | {k for k, _ in FIXED_RECEIPT_KNOBS}
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise CollectError(f"--child-env {raw!r} must be KEY=VALUE")
        key, value = raw.split("=", 1)
        if not key:
            raise CollectError(f"--child-env {raw!r} has an empty key")
        if not _CHILD_ENV_KEY.match(key):
            raise CollectError(
                f"--child-env key {key!r} must match ^HIPFIRE_[A-Z0-9_]+$ "
                "(non-HIPFIRE variables already pass through the scrub)"
            )
        if key in owned:
            raise CollectError(f"--child-env {key}: the collector sets {key} itself")
        if key in result:
            raise CollectError(f"--child-env {key} is given twice")
        result[key] = value
    return result


def validate(args: argparse.Namespace, environ: dict) -> Config:
    """Every refusal happens here, before any process is launched."""
    out = require_abs("--out", args.out or "")
    run_id = args.run_id
    if run_id is not None and not _RUN_ID.match(run_id):
        raise CollectError(f"--run-id {run_id!r} must match [A-Za-z0-9._-]+")
    if args.analyze or args.prehash:
        if args.analyze and args.prehash:
            raise CollectError("--analyze and --prehash are mutually exclusive")
    if args.analyze:
        if args.stage not in ("quality", "perf", "greedy"):
            raise CollectError("--analyze recomputes statistics for --stage quality or perf (or greedy) only")
        if not out.is_dir():
            raise CollectError(f"--analyze: --out {out} does not exist")
        return Config(
            stage=args.stage, binary_dir=Path("/"), model=Path("/"), wt2_ref=Path("/"), code_ref=Path("/"),
            modes=(), runs=0, max_seq=0, out=out, run_id=run_id or "", graphs=(),
        )
    if args.modes is None:
        raise CollectError("--modes is required")
    modes = parse_modes(args.modes)
    binary_dir = require_abs("--binary-dir", args.binary_dir or "")
    if not binary_dir.is_dir():
        raise CollectError(f"--binary-dir {binary_dir} is not a directory")
    model = require_file("--model", args.model or "")
    if args.stage == "greedy":
        # The greedy stage reads neither teacher: they are optional and only validated when given.
        wt2 = require_file("--wt2-ref", args.wt2_ref) if args.wt2_ref else Path("/")
        code = require_file("--code-ref", args.code_ref) if args.code_ref else Path("/")
    else:
        wt2 = require_file("--wt2-ref", args.wt2_ref or "")
        code = require_file("--code-ref", args.code_ref or "")
    if args.max_seq is None or args.max_seq <= 0:
        raise CollectError("--max-seq must be a positive integer")
    runs = args.runs if args.runs is not None else DEFAULT_RUNS[args.stage]
    if runs <= 0:
        raise CollectError("--runs must be positive")
    if args.child_timeout_secs <= 0:
        raise CollectError("--child-timeout-secs must be positive")
    if args.sampled_trials < MIN_SAMPLED_TRIALS:
        raise CollectError(f"--sampled-trials must be >= {MIN_SAMPLED_TRIALS} (PLAN G5)")
    graphs = parse_graphs(args.graph) if args.graph else DEFAULT_GRAPHS[args.stage]
    if args.stage in ("quality", "perf") and BASELINE_MODE not in modes:
        raise CollectError(f"stage {args.stage} compares against the {BASELINE_MODE} baseline: add it to --modes")
    prompt = long_prompt = None
    if args.stage in ("memory", "perf", "greedy"):
        if not args.prompt_file:
            raise CollectError(f"stage {args.stage} requires --prompt-file")
        prompt = require_file("--prompt-file", args.prompt_file)
    if args.stage == "greedy":
        text = read_prompt_text(prompt)
        if not text.strip():
            raise CollectError(f"--prompt-file {prompt} is empty")
        if "\0" in text:
            raise CollectError(f"--prompt-file {prompt} contains a NUL byte (cannot be passed as an argument)")
        if text.startswith("-"):
            raise CollectError(f"--prompt-file {prompt} starts with '-': `hipfire run` would parse the prompt as a flag")
        if len(text.encode("utf-8")) >= MAX_ARG_STRLEN:
            raise CollectError(
                f"--prompt-file {prompt} is {len(text.encode('utf-8'))} bytes: a single argv string must be "
                f"< {MAX_ARG_STRLEN} bytes (Linux MAX_ARG_STRLEN) and `hipfire run` takes the prompt as an argument"
            )
    if args.max_chunks is not None:
        if args.stage != "quality":
            raise CollectError("--max-chunks applies to --stage quality only")
        if args.max_chunks < MIN_MAX_CHUNKS:
            raise CollectError(f"--max-chunks must be >= {MIN_MAX_CHUNKS} (paired analysis needs >= 2 chunks)")
    if args.greedy_tokens is not None:
        if args.stage != "greedy":
            raise CollectError("--greedy-tokens applies to --stage greedy only")
        if not 1 <= args.greedy_tokens <= MAX_GREEDY_TOKENS:
            raise CollectError(f"--greedy-tokens must be in 1..{MAX_GREEDY_TOKENS}")
    only = parse_only(args.only) if args.only is not None else ()
    if args.stage == "perf":
        if not args.long_prompt_file:
            raise CollectError("stage perf requires --long-prompt-file (the --ttft command)")
        long_prompt = require_file("--long-prompt-file", args.long_prompt_file)
    if args.stage == "smoke" and "unset" in graphs:
        raise CollectError("smoke needs an explicit --graph 0 or 1 (the oracle takes --graph)")
    if args.oracle_only and args.stage != "smoke":
        raise CollectError("--oracle-only applies to --stage smoke only")
    child_env = parse_child_env(args.child_env)
    if not args.prehash:
        stripped = sorted(
            name for name in environ
            if name.startswith("HIPFIRE_")
            and name not in KEEP_HIPFIRE_ENV
            and name not in child_env
            and not name.startswith(IGNORED_PARENT_HIPFIRE_PREFIXES)
        )
        if stripped:
            raise CollectError(
                f"parent environment sets {', '.join(stripped)}; the child scrub would drop them silently: "
                "pass the ones a child needs with --child-env KEY=VALUE and unset the rest"
            )
    run_id = run_id or (_dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-p{os.getpid()}")
    cfg = Config(
        stage=args.stage, binary_dir=binary_dir, model=model, wt2_ref=wt2, code_ref=code,
        modes=modes, runs=runs, max_seq=args.max_seq, out=out, run_id=run_id, graphs=graphs,
        prompt_file=prompt, long_prompt_file=long_prompt, child_timeout_secs=args.child_timeout_secs,
        sampled_trials=args.sampled_trials, bench_max_seq=args.bench_max_seq, keep_going=args.keep_going,
        dry_run=args.dry_run, oracle_only=args.oracle_only,
        max_chunks=args.max_chunks if args.max_chunks is not None else DEFAULT_MAX_CHUNKS,
        greedy_tokens=args.greedy_tokens if args.greedy_tokens is not None else DEFAULT_GREEDY_TOKENS,
        only=only, child_env=child_env,
    )
    cfg.binaries = resolve_and_verify_binaries(binary_dir, required_binaries(args.stage, modes, args.oracle_only))
    if cfg.only:
        plan_children(cfg)  # refuses unknown labels / an empty selection before any launch
    if not (args.dry_run or args.prehash):
        run_dir = cfg.run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise CollectError(f"run dir {run_dir} already exists and is not empty; pick a new --run-id")
        require_fabric_hold(environ, require_lock=not cfg.oracle_only)
    return cfg


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_KLD_SUMMARY = re.compile(
    r"qwen4_kld eval: mean KLD = (?P<kld>\S+)\s+mean NLL = (?P<nll>\S+)\s+PPL = (?P<ppl>\S+)\s+"
    r"top1 = (?P<top1>\S+)\s+\((?P<tokens>\d+) tokens\)\s+logits sha256 (?P<sha>[0-9a-f]+) -> (?P<path>.*)"
)
_KLD_HEADER = re.compile(
    r"qwen4_kld eval: (?P<chunks>\d+) chunks x (?P<ctx>\d+) \((?P<route>[^)]*) route\); teacher PPL (?P<ppl>\S+)"
)
_CHUNK_LINE = re.compile(r"^\s+chunk (\d+)/(\d+)\s+KLD ")


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_kld_stderr(text: str) -> dict:
    """Header/summary of ``qwen4_kld eval`` (one entry per scored pass)."""
    headers = [m.groupdict() for m in _KLD_HEADER.finditer(text)]
    summaries = []
    for m in _KLD_SUMMARY.finditer(text):
        d = m.groupdict()
        summaries.append(
            {
                "mean_kld": _float(d["kld"]), "mean_nll": _float(d["nll"]), "ppl": _float(d["ppl"]),
                "top1": _float(d["top1"]), "tokens": int(d["tokens"]),
                "logits_sha256": d["sha"], "output": d["path"].strip(),
            }
        )
    chunk_lines = sum(1 for line in text.splitlines() if _CHUNK_LINE.match(line))
    return {
        "headers": [
            {"chunks": int(h["chunks"]), "n_ctx": int(h["ctx"]), "route": h["route"], "teacher_ppl": _float(h["ppl"])}
            for h in headers
        ],
        "summaries": summaries,
        "chunk_lines": chunk_lines,
    }


def read_kldseq(path: Path) -> dict:
    """HFKSEQ v1/v2 (saddle-quant ``format::kldseq``): 20-byte header then
    24 B (v2: mean, p99, mean_nll) or 16 B (v1, NLL = NaN) f64 LE per chunk."""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path}: truncated header ({len(data)} bytes)")
    if data[:8] != b"HFKSEQ\0\0":
        raise ValueError(f"{path}: bad HFKSEQ magic {data[:8]!r}")
    version, n_chunk = struct.unpack_from("<II", data, 8)
    if version not in (1, 2):
        raise ValueError(f"{path}: unsupported HFKSEQ version {version}")
    width = 24 if version == 2 else 16
    need = 20 + n_chunk * width
    if len(data) < need:
        raise ValueError(f"{path}: truncated payload (need {need}, have {len(data)})")
    if len(data) > need:
        raise ValueError(f"{path}: trailing bytes (expected {need}, have {len(data)})")
    chunks = []
    for i in range(n_chunk):
        base = 20 + i * width
        if version == 2:
            mean, p99, nll = struct.unpack_from("<ddd", data, base)
        else:
            mean, p99 = struct.unpack_from("<dd", data, base)
            nll = math.nan
        chunks.append((mean, p99, nll))
    return {"version": version, "chunks": chunks, "sha256": hashlib.sha256(data).hexdigest()}


def write_kldseq(path: Path, chunks: list[tuple[float, float, float]]) -> None:
    """Writer for synthetic fixtures (mirrors ``kldseq::write``, v2)."""
    body = b"".join(struct.pack("<ddd", *c) for c in chunks)
    path.write_bytes(b"HFKSEQ\0\0" + struct.pack("<III", 2, len(chunks), 0) + body)


_TEST_HEAD = re.compile(r"^test (\S+) \.\.\. ?(.*)$")
_TEST_STATUS = re.compile(r"(ok|FAILED|ignored)\b")
_BARE_STATUS = re.compile(r"^(ok|FAILED|ignored)\s*$")
_TEST_RESULT = re.compile(
    r"^test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", re.M
)
_RUNNING = re.compile(r"^running (\d+) tests?", re.M)


def parse_test_lines(text: str) -> list[dict]:
    """Per-test statuses in completion order. libtest prints ``test NAME ... ``
    and then the status, but anything the test writes to the shared stdout in
    between (e.g. the ``--nocapture`` child libtest run that
    ``disk_snapshots_restore_after_restart_on_flash_next`` spawns) pushes the
    status onto a later line of its own. A header without a status waits on a
    stack; the next bare status line completes the innermost pending test (a
    nested run finishes before its parent)."""
    tests: list[dict] = []
    pending: list[str] = []
    for line in text.splitlines():
        head = _TEST_HEAD.match(line)
        if head:
            status = _TEST_STATUS.match(head.group(2))
            if status:
                tests.append({"name": head.group(1), "status": status.group(1)})
            else:
                pending.append(head.group(1))
            continue
        bare = _BARE_STATUS.match(line)
        if bare and pending:
            tests.append({"name": pending.pop(), "status": bare.group(1)})
    return tests


def parse_cargo_test(text: str) -> dict:
    results = [
        {"status": m.group(1), "passed": int(m.group(2)), "failed": int(m.group(3)),
         "ignored": int(m.group(4)), "measured": int(m.group(5)), "filtered_out": int(m.group(6))}
        for m in _TEST_RESULT.finditer(text)
    ]
    return {
        "running": [int(m.group(1)) for m in _RUNNING.finditer(text)],
        "tests": parse_test_lines(text),
        "results": results,
        "passed": sum(r["passed"] for r in results),
        "failed": sum(r["failed"] for r in results),
        "ignored": sum(r["ignored"] for r in results),
        "skip_lines": [line.strip() for line in text.splitlines() if re.search(r"\bskip(ped|ping)?\b", line, re.I)][:50],
    }


def classify_test(rc: Optional[int], timed_out: bool, parsed: dict, expect: tuple[str, ...]) -> tuple[str, list[str]]:
    """(outcome, reasons): ok | failed | timeout | incomplete. Never infers ok
    from rc alone: at least one test must have actually passed."""
    reasons: list[str] = []
    if timed_out:
        return "timeout", ["child timed out"]
    if rc != 0:
        return "failed", [f"rc={rc}"]
    if parsed["failed"] > 0:
        return "failed", [f"{parsed['failed']} test(s) failed"]
    if not parsed["results"] or parsed["passed"] == 0:
        return "incomplete", ["no test passed (0 tests ran, all ignored/filtered, or no result line)"]
    status = {t["name"]: t["status"] for t in parsed["tests"]}
    for name in expect:
        if status.get(name) != "ok":
            reasons.append(f"expected case {name} did not report ok (status: {status.get(name)})")
    if reasons:
        return "incomplete", reasons
    return "ok", []


def find_json_object(text: str):
    """First JSON object parsable from a line-start ``{`` (bench ``--json`` stdout)."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"^\{", text, re.M):
        try:
            value, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _median_of(obj, key) -> Optional[float]:
    value = obj.get(key) if isinstance(obj, dict) else None
    if isinstance(value, dict):
        value = value.get("median")
    return float(value) if isinstance(value, (int, float)) else None


def extract_bench(report: dict) -> dict:
    """Observed numbers of one ``hipfire bench --json`` report."""
    protocol = report.get("protocol")
    out: dict = {
        "protocol": protocol,
        "prompt_tokens": report.get("prompt_tokens"),
        "prompt_md5": report.get("prompt_md5"),
        "max_seq": report.get("max_seq"),
        "spec_requested": report.get("spec_requested"),
        "spec_route": report.get("spec_route"),
        "spec_tau": report.get("spec_tau"),
        "warnings": report.get("warnings"),
    }
    if protocol == "native-generate-v1":
        out.update(
            decode_tok_s=_median_of(report, "decode_tok_s"),
            prefill_tok_s=_median_of(report, "prefill_tok_s"),
            wall_tok_s=_median_of(report, "wall_tok_s"),
            ttft_ms=_median_of(report, "ttft_ms"),
        )
    elif protocol == "client-ttft-v1":
        ttft = report.get("ttft") or {}
        out.update(
            ttft_ms=float(ttft["median"]) if isinstance(ttft.get("median"), (int, float)) else None,
            pp_tok_s=_median_of(report, "pp_tok_s"),
        )
    elif protocol == "synthetic-pp-tg-matrix-v1":
        out["prefill_rows"] = [
            {"tokens": r.get("tokens"), "tok_s": _median_of(r, "stats")} for r in report.get("prefill", [])
        ]
        out["decode_rows"] = [
            {"context": r.get("context"), "tokens": r.get("tokens"), "tok_s": _median_of(r, "stats")}
            for r in report.get("decode", [])
        ]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default), q in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of empty sample")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_mean_ci(
    values: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile 95% CI of the mean, paired resampling over
    chunks with replacement, ``random.Random(seed)`` (documented seed)."""
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap of empty sample")
    rng = random.Random(seed)
    draw = rng.random
    means = []
    for _ in range(resamples):
        total = 0.0
        for _i in range(n):
            total += values[int(draw() * n)]
        means.append(total / n)
    means.sort()
    return sum(values) / n, percentile(means, 2.5), percentile(means, 97.5)


def rule_applied(lo: float, hi: float) -> str:
    """The PLAN rule: lo<=0 -> PASS; lo>0 -> PASS iff hi<=+0.010."""
    if lo <= 0:
        return "PASS"
    return "PASS" if hi <= HI_BUDGET else "FAIL"


def arm_stats(chunks: list[tuple[float, float, float]]) -> dict:
    means = [c[0] for c in chunks]
    p99s = [c[1] for c in chunks]
    nlls = [c[2] for c in chunks]
    nonfinite = sum(1 for c in chunks for v in c[:2] if not math.isfinite(v))
    nonfinite_nll = sum(1 for v in nlls if not math.isfinite(v))
    n = len(chunks)
    mean_kld = sum(means) / n if n else math.nan
    finite_nll = [v for v in nlls if math.isfinite(v)]
    mean_nll = sum(finite_nll) / len(finite_nll) if finite_nll else math.nan
    return {
        "n_chunks": n,
        "mean_kld": mean_kld,
        "p99_kld_of_chunk_p99": percentile(sorted(p99s), 99) if n and not nonfinite else None,
        "max_chunk_p99_kld": max(p99s) if p99s else None,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll) if math.isfinite(mean_nll) else None,
        "nonfinite_kld_values": nonfinite,
        "nonfinite_nll_values": nonfinite_nll,
    }


def paired_delta(cand: list[tuple[float, float, float]], base: list[tuple[float, float, float]]) -> dict:
    if len(cand) != len(base):
        return {"error": f"chunk count differs: candidate {len(cand)} vs baseline {len(base)}"}
    deltas = [c[0] - b[0] for c, b in zip(cand, base)]
    if not all(math.isfinite(d) for d in deltas):
        return {"error": "non-finite per-chunk KLD in a pair: no CI computed", "n_chunks": len(deltas)}
    mean, lo, hi = bootstrap_mean_ci(deltas)
    return {
        "n_chunks": len(deltas), "mean_delta": mean, "ci95_lo": lo, "ci95_hi": hi,
        "rule_applied": rule_applied(lo, hi),
    }


def gap_closure(kld_q8: float, kld_fp32: float, kld_ef: float) -> dict:
    gap = kld_q8 - kld_fp32
    if not (math.isfinite(gap) and math.isfinite(kld_ef)) or gap <= 0:
        return {"q8": kld_q8, "fp32": kld_fp32, "q8_ef": kld_ef, "gap_q8_minus_fp32": gap,
                "closure": None, "note": "undefined: q8-fp32 gap is not a positive finite number"}
    closure = (kld_q8 - kld_ef) / gap
    return {"q8": kld_q8, "fp32": kld_fp32, "q8_ef": kld_ef, "gap_q8_minus_fp32": gap,
            "closure": closure, f"closure_ge_{CLOSURE_TARGET}": closure >= CLOSURE_TARGET}


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------


def analyze_quality(run_dir: Path, children: list[dict]) -> dict:
    """Recompute everything from the .kldseq files and stderr of each child."""
    errors: list[str] = []
    arms: dict = {}  # (graph, corpus, route) -> mode -> run -> record
    for c in children:
        if c.get("kind") != "kld_eval":
            continue
        cdir = run_dir / c["dir"]
        seq_path = cdir / c["kldseq"]
        if not seq_path.is_file():
            errors.append(f"{c['id']}: missing {seq_path.name}")
            continue
        try:
            seq = read_kldseq(seq_path)
        except ValueError as exc:
            errors.append(f"{c['id']}: {exc}")
            continue
        summary = {}
        err_path = cdir / "stderr.txt"
        if err_path.is_file():
            sums = parse_kld_stderr(err_path.read_text(errors="replace"))["summaries"]
            summary = sums[-1] if sums else {}
        key = (c["graph"], c["corpus"], c["route"])
        arms.setdefault(key, {}).setdefault(c["mode"], {})[c["run"]] = {
            "chunks": seq["chunks"], "sha256": seq["sha256"], "summary": summary,
        }
    result_arms: dict = {}
    pairs: list[dict] = []
    determinism: list[dict] = []
    closures: list[dict] = []
    for key in sorted(arms):
        graph, corpus, route = key
        by_mode = arms[key]
        for mode in sorted(by_mode):
            runs = by_mode[mode]
            for run in sorted(runs):
                rec = runs[run]
                stats = arm_stats(rec["chunks"])
                stats["top1_from_stderr"] = rec["summary"].get("top1")
                stats["stderr_mean_kld"] = rec["summary"].get("mean_kld")
                stats["logits_sha256"] = rec["summary"].get("logits_sha256")
                stats["kldseq_sha256"] = rec["sha256"]
                result_arms.setdefault(f"g{graph}/{corpus}/{route}", {}).setdefault(mode, {})[f"run{run}"] = stats
            seq_hashes = [runs[r]["sha256"] for r in sorted(runs)]
            logit_hashes = [runs[r]["summary"].get("logits_sha256") for r in sorted(runs)]
            determinism.append(
                {
                    "graph": graph, "corpus": corpus, "route": route, "mode": mode, "runs": len(runs),
                    "kldseq_byte_identical": (len(set(seq_hashes)) == 1) if len(runs) > 1 else None,
                    "kldseq_sha256": seq_hashes,
                    "logits_sha256_identical": (len(set(logit_hashes)) == 1) if len(runs) > 1 else None,
                    "logits_sha256": logit_hashes,
                }
            )
        base_runs = by_mode.get(BASELINE_MODE, {})
        for mode in sorted(by_mode):
            if mode == BASELINE_MODE:
                continue
            for run in sorted(by_mode[mode]):
                if run not in base_runs:
                    errors.append(f"g{graph}/{corpus}/{route} run{run}: no {BASELINE_MODE} baseline for paired {mode}")
                    continue
                entry = {
                    "graph": graph, "corpus": corpus, "route": route, "run": run,
                    "candidate": mode, "baseline": BASELINE_MODE,
                    "delta": "candidate - baseline (per-chunk mean KLD)",
                    "rule": RULE_TEXT,
                    "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci": "95% percentile"},
                }
                entry.update(paired_delta(by_mode[mode][run]["chunks"], base_runs[run]["chunks"]))
                pairs.append(entry)
        if corpus == "code24" and route == "decode" and all(m in by_mode for m in ("q8", "fp32", "q8_ef")):
            for run in sorted(by_mode["q8"]):
                if run in by_mode["fp32"] and run in by_mode["q8_ef"]:
                    k = {m: arm_stats(by_mode[m][run]["chunks"])["mean_kld"] for m in ("q8", "fp32", "q8_ef")}
                    closures.append(
                        {"graph": graph, "run": run, **gap_closure(k["q8"], k["fp32"], k["q8_ef"])}
                    )
    # A filtered (--only) or partially failed run has gaps: a (graph, corpus, route) cell that lacks a mode
    # other cells have is reported as a missing cell, never silently dropped.
    seen_modes = sorted({m for by_mode in arms.values() for m in by_mode})
    missing_cells: list[dict] = []
    for key in sorted(arms):
        absent = [m for m in seen_modes if m not in arms[key]]
        if absent:
            missing_cells.append({"graph": key[0], "corpus": key[1], "route": key[2], "missing_modes": absent})
            errors.append(f"g{key[0]}/{key[1]}/{key[2]}: no arm for mode(s) {','.join(absent)} (missing cell)")
    verdicts = [p.get("rule_applied") for p in pairs]
    return {
        "rule": RULE_TEXT,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "arms": result_arms,
        "pairs": pairs,
        "pairs_rule_applied_all_pass": (all(v == "PASS" for v in verdicts) if verdicts else None),
        "code24_decode_gap_closure": closures,
        "same_route_determinism": determinism,
        "missing_cells": missing_cells,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Perf analysis
# ---------------------------------------------------------------------------

#: label -> [(metric name, extractor(observation) -> float|None, higher_is_better)]


def _matrix_pp(obs: dict) -> Optional[float]:
    for row in obs.get("prefill_rows") or []:
        if row.get("tokens") == 8192:
            return row.get("tok_s")
    return None


def _matrix_tg(obs: dict) -> Optional[float]:
    rows = [r for r in obs.get("decode_rows") or [] if r.get("tokens") == 32]
    return rows[0].get("tok_s") if rows else None


def _tau(obs: dict) -> Optional[float]:
    taus = [t for t in (obs.get("spec_tau") or []) if isinstance(t, (int, float))]
    return statistics.median(taus) if taus else None


PERF_METRICS = {
    "stateless-spec-off": (
        ("decode_tok_s", lambda o: o.get("decode_tok_s"), True),
        ("prefill_tok_s", lambda o: o.get("prefill_tok_s"), True),
    ),
    "stateless-spec-mtp": (
        ("decode_tok_s", lambda o: o.get("decode_tok_s"), True),
        ("prefill_tok_s", lambda o: o.get("prefill_tok_s"), True),
        ("tau", _tau, True),
    ),
    "ttft-long": (
        ("ttft_ms", lambda o: o.get("ttft_ms"), False),
        ("pp_tok_s", lambda o: o.get("pp_tok_s"), True),
    ),
    "matrix-pp8192-tg32": (
        ("pp8192_tok_s_synthetic", _matrix_pp, True),
        ("tg32_tok_s_synthetic", _matrix_tg, True),
    ),
}


def _loss(value: float, base: float, higher_is_better: bool) -> Optional[float]:
    """Positive = worse than the baseline."""
    if base is None or value is None or base == 0:
        return None
    return (1 - value / base) if higher_is_better else (value / base - 1)


def analyze_perf(children: list[dict]) -> dict:
    values: dict = {}  # (graph,label,metric) -> mode -> run -> value
    errors: list[str] = []
    for c in children:
        if not c.get("kind", "").startswith("bench_") or c["kind"] == "bench_memory":
            continue
        obs = c.get("observations") or {}
        if obs.get("error") or not obs.get("protocol"):
            errors.append(f"{c['id']}: no bench report ({obs.get('error', 'not parsed')})")
            continue
        for name, fn, hib in PERF_METRICS[c["label"]]:
            v = fn(obs)
            if v is not None:
                values.setdefault((c["graph"], c["label"], name, hib), {}).setdefault(c["mode"], {})[c["run"]] = v
    rows = []
    for (graph, label, name, hib), by_mode in sorted(values.items()):
        base_runs = by_mode.get(BASELINE_MODE, {})
        base_median = statistics.median(base_runs.values()) if base_runs else None
        for mode in sorted(by_mode):
            runs = by_mode[mode]
            med = statistics.median(runs.values())
            paired = [
                _loss(runs[r], base_runs[r], hib) for r in sorted(runs) if r in base_runs
            ]
            paired = [p for p in paired if p is not None]
            row = {
                "graph": graph, "label": label, "metric": name, "mode": mode,
                "higher_is_better": hib, "samples": [runs[r] for r in sorted(runs)], "median": med,
            }
            if mode != BASELINE_MODE:
                loss_of_medians = _loss(med, base_median, hib)
                row.update(
                    loss_vs_q8_of_medians=loss_of_medians,
                    loss_vs_q8_median_of_paired_runs=statistics.median(paired) if paired else None,
                    policy_line_observation={
                        "line": PERF_POLICY_LINE,
                        "note": "observation against the 5% policy line (positive loss = worse than q8); not a verdict",
                        "loss_of_medians_exceeds_line": (loss_of_medians > PERF_POLICY_LINE) if loss_of_medians is not None else None,
                    },
                )
            rows.append(row)
    return {"rows": rows, "errors": errors}


def _first_diff(a: str, b: str) -> Optional[int]:
    """Offset of the first differing character; ``None`` when equal; the shorter length for a strict prefix."""
    if a == b:
        return None
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def _all_or_none(flags: list[bool], missing: int) -> Optional[bool]:
    """False if any compared pair differs; None (never a pass) when pairs are missing or none exist."""
    if any(f is False for f in flags):
        return False
    if missing or not flags:
        return None
    return True


def analyze_greedy(children: list[dict]) -> dict:
    """P0 kill experiment: greedy MTP vs AR text equality per (run, graph, mode)."""
    errors: list[str] = []
    seen: dict = {}  # (run, graph, mode) -> {spec: child id}
    cells: dict = {}  # (run, graph, mode) -> {spec: observations of an ok child}
    per_child: list[dict] = []
    greedy = [c for c in children if c.get("kind") == "greedy_run"]
    for c in greedy:
        obs = c.get("observations") or {}
        spec = str(c.get("label", "")).removeprefix("greedy-spec-")
        key = (c["run"], c["graph"], c["mode"])
        seen.setdefault(key, {})[spec] = c["id"]
        per_child.append(
            {
                "id": c["id"], "spec": spec, "outcome": c.get("outcome"),
                "daemon_admission_lines": [
                    line for line in obs.get("daemon_log_lines") or [] if any(k in line for k in GREEDY_ADMISSION_KEYS)
                ],
            }
        )
        if spec not in GREEDY_SPECS:
            errors.append(f"{c['id']}: unknown greedy spec {spec!r}")
        elif c.get("outcome") == "ok" and isinstance(obs.get("content"), str):
            cells.setdefault(key, {})[spec] = obs
        else:
            errors.append(f"{c['id']}: outcome {c.get('outcome')} ({obs.get('error') or 'no usable content'}); not compared")
    pairs: list[dict] = []
    missing_by_mode: dict = {}
    for key in sorted(seen):
        run, graph, mode = key
        off, mtp = cells.get(key, {}).get("off"), cells.get(key, {}).get("mtp")
        if off is None or mtp is None:
            absent = [s for s, o in (("off", off), ("mtp", mtp)) if o is None]
            errors.append(f"run{run}/g{graph}/{mode}: no spec-{'/'.join(absent)} result: AR-vs-MTP pair missing")
            missing_by_mode[mode] = missing_by_mode.get(mode, 0) + 1
            continue
        pairs.append(
            {
                "run": run, "graph": graph, "mode": mode,
                "equal": off["content"] == mtp["content"],
                "first_diff_char": _first_diff(off["content"], mtp["content"]),
                "len_off": len(off["content"]), "len_mtp": len(mtp["content"]),
                "tokens_off": off.get("tokens"), "tokens_mtp": mtp.get("tokens"),
                "sha256_off": off.get("content_sha256"), "sha256_mtp": mtp.get("content_sha256"),
            }
        )
    modes = sorted({k[2] for k in seen})
    per_mode: dict = {}
    for mode in modes:
        flags = [p["equal"] for p in pairs if p["mode"] == mode]
        per_mode[mode] = {
            "pairs_compared": len(flags),
            "pairs_missing": missing_by_mode.get(mode, 0),
            "mtp_equals_ar": _all_or_none(flags, missing_by_mode.get(mode, 0)),
        }
    cross: list[dict] = []
    by_group: dict = {}
    for (run, graph, mode), specs in seen.items():
        for spec in specs:
            by_group.setdefault((run, graph, spec), set()).add(mode)
    for (run, graph, spec) in sorted(by_group):
        if not {"q8", "fp32"} <= by_group[(run, graph, spec)]:
            continue
        q8 = cells.get((run, graph, "q8"), {}).get(spec)
        f32 = cells.get((run, graph, "fp32"), {}).get(spec)
        if q8 is None or f32 is None:
            errors.append(f"run{run}/g{graph}/spec-{spec}: q8-vs-fp32 content pair missing (diagnostic not compared)")
            continue
        cross.append(
            {
                "run": run, "graph": graph, "spec": spec,
                "equal": q8["content"] == f32["content"],
                "first_diff_char": _first_diff(q8["content"], f32["content"]),
                "len_q8": len(q8["content"]), "len_fp32": len(f32["content"]),
            }
        )
    if not greedy:
        errors.append("no greedy_run children")
    return {
        "labels": {
            "fp32_mtp_equals_ar": "GATE (Track A F32 tripwire)",
            "q8_mtp_equals_ar": "DIAGNOSTIC",
            "cross_mode_q8_vs_fp32": "DIAGNOSTIC",
        },
        "pairs": pairs,
        "per_mode": per_mode,
        "fp32_mtp_equals_ar": per_mode.get("fp32", {}).get("mtp_equals_ar"),
        "q8_mtp_equals_ar": per_mode.get("q8", {}).get("mtp_equals_ar"),
        "cross_mode_q8_vs_fp32": cross,
        "all_loaded": bool(greedy) and all(c.get("outcome") == "ok" for c in greedy),
        "children": per_child,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------


def read_mem_available_kib() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _to_mib(value, unit: str) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    unit = (unit or "").strip().lower()
    factor = {"b": 1 / 2**20, "kb": 1 / 1024, "kib": 1 / 1024, "mb": 1.0, "mib": 1.0, "gb": 1024.0, "gib": 1024.0}
    return num * factor.get(unit, 1.0) if (unit in factor or unit == "") else None


def extract_used_vram_mib(obj, unit_hint: str = "") -> list[float]:
    """Every used-VRAM figure (MiB) found in an amd-smi/rocm-smi JSON value."""
    found: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = key.lower()
            if "used" in k and "vram" in k:
                hint = "b" if "(b)" in k else unit_hint
                if isinstance(value, dict) and "value" in value:
                    mib = _to_mib(value["value"], str(value.get("unit", hint)))
                elif isinstance(value, (int, float, str)):
                    mib = _to_mib(value, hint)
                else:
                    mib = None
                if mib is not None:
                    found.append(mib)
                    continue
            found.extend(extract_used_vram_mib(value, unit_hint))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(extract_used_vram_mib(item, unit_hint))
    return found


def gpu_smi_command() -> Optional[list[str]]:
    if shutil.which("amd-smi"):
        return ["amd-smi", "metric", "--mem-usage", "--json"]
    if shutil.which("rocm-smi"):
        return ["rocm-smi", "--showmeminfo", "vram", "--json"]
    return None


class MemorySampler(threading.Thread):
    """Samples MemAvailable and GPU memory at ~1 s cadence while a child runs."""

    def __init__(self, path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.cmd = gpu_smi_command()
        self.samples: list[dict] = []

    def sample(self) -> dict:
        rec: dict = {"t": utc_now(), "mem_available_kib": read_mem_available_kib()}
        if self.cmd is not None:
            try:
                cp = subprocess.run(self.cmd, capture_output=True, text=True, timeout=15)
                rec["gpu_smi_rc"] = cp.returncode
                try:
                    raw = json.loads(cp.stdout)
                    rec["gpu_used_vram_mib"] = extract_used_vram_mib(raw)
                    rec["gpu_raw"] = raw
                except ValueError:
                    rec["gpu_raw_text"] = cp.stdout[:4000]
            except (OSError, subprocess.TimeoutExpired) as exc:
                rec["gpu_smi_error"] = str(exc)
        return rec

    def run(self) -> None:
        nxt = time.monotonic()
        while not self.stop_event.is_set():
            self.samples.append(self.sample())
            nxt += self.interval
            self.stop_event.wait(max(0.0, nxt - time.monotonic()))

    def finish(self) -> None:
        self.stop_event.set()
        self.join()
        self.samples.append(self.sample())
        with open(self.path, "w") as handle:
            for rec in self.samples:
                handle.write(json.dumps(rec, sort_keys=True) + "\n")

    def summary(self) -> dict:
        return summarize_memory(self.samples, self.cmd)


def summarize_memory(samples: list[dict], cmd: Optional[list[str]]) -> dict:
    avail = [s["mem_available_kib"] for s in samples if s.get("mem_available_kib") is not None]
    vram = [s["gpu_used_vram_mib"] for s in samples if s.get("gpu_used_vram_mib")]
    out: dict = {"gpu_smi": cmd, "samples": len(samples)}
    if avail:
        out.update(
            mem_available_kib_before=avail[0],
            mem_available_kib_after=avail[-1],
            mem_available_kib_min=min(avail),
            host_peak_drop_kib=avail[0] - min(avail),
        )
    if vram:
        width = max(len(v) for v in vram)
        out.update(
            gpu_used_vram_mib_before=vram[0],
            gpu_used_vram_mib_after=vram[-1],
            gpu_used_vram_mib_peak_per_device=[max(v[i] for v in vram if len(v) > i) for i in range(width)],
        )
    return out


# ---------------------------------------------------------------------------
# Observations per child
# ---------------------------------------------------------------------------


def file_listing(root: Path, patterns: tuple[str, ...], limit: int = 400) -> list[dict]:
    items = []
    if not root.exists():
        return items
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file() and len(items) < limit:
                items.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return items


def collect_coverage(root: Path) -> dict:
    """Sum every ``{"coverage": {...}}`` object found under ``root``."""
    total: dict[str, int] = {}
    sources: list[str] = []
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        cov = data.get("coverage") if isinstance(data, dict) else None
        if isinstance(cov, dict):
            sources.append(str(path.relative_to(root)))
            for key, value in cov.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    total[key] = total.get(key, 0) + value
    return {"counters": total, "sources": sources}


def q8ef_coverage_gaps(coverage: dict, graph: str) -> list[str]:
    required = list(Q8EF_REQUIRED_COVERAGE) + (list(Q8EF_GRAPH_COVERAGE) if graph == "1" else [])
    return [key for key in required if coverage["counters"].get(key, 0) <= 0]


def refusal_lines(*texts: str) -> list[str]:
    pattern = re.compile(r"refus|error|cannot|not available|panicked|admission|out of memory|oom", re.I)
    lines = []
    for text in texts:
        lines += [line.strip() for line in text.splitlines() if pattern.search(line)]
    return lines[:200]


def daemon_log_lines(*texts: str) -> list[str]:
    lines = []
    for text in texts:
        lines += [line.rstrip() for line in text.splitlines() if any(k in line for k in DAEMON_LOG_KEYS)]
    return lines[:500]


def observe_child(child: Child, cdir: Path, rc: Optional[int], timed_out: bool, graph: str) -> tuple[str, dict]:
    """(outcome, observations) parsed purely from the files the child left."""
    stdout = (cdir / "stdout.txt").read_text(errors="replace") if (cdir / "stdout.txt").is_file() else ""
    stderr = (cdir / "stderr.txt").read_text(errors="replace") if (cdir / "stderr.txt").is_file() else ""
    obs: dict = {"kind": child.kind}
    kind = child.kind
    if timed_out:
        obs["timed_out"] = True
    if kind == "oracle":
        path = cdir / "oracle.json"
        obs["oracle_json"] = str(path) if path.is_file() else None
        if path.is_file():
            obs["oracle_json_sha256"] = sha256_file(path)
            try:
                data = read_json(path)
                obs["oracle_json_top_level_keys"] = sorted(data) if isinstance(data, dict) else None
                if isinstance(data, dict):
                    obs["oracle_json_scalars"] = {k: v for k, v in data.items() if isinstance(v, (bool, int, float, str))}
            except ValueError as exc:
                obs["oracle_json_error"] = str(exc)
        obs["stdout_tail"] = stdout.splitlines()[-40:]
        obs["oracle_exit_code_meaning"] = {0: "match", 1: "mismatch", 2: "setup/unsupported"}.get(rc, "other/unrecorded")
        outcome = "timeout" if timed_out else ("ok" if rc == 0 and path.is_file() else "failed")
        return outcome, obs
    if kind in ("identity_test", "session_test", "ef_routes_test", "sampled_test"):
        parsed = parse_cargo_test(stdout)
        obs["cargo_test"] = parsed
        out_root = cdir / ("sampled" if kind == "sampled_test" else "out")
        obs["summary_txt"] = {
            str(p.relative_to(cdir)): p.read_text(errors="replace") for p in sorted(out_root.rglob("summary.txt"))
        } if out_root.exists() else {}
        obs["output_files"] = file_listing(out_root, ("*.json", "*.txt"))
        obs["decoded_text_paths"] = [
            str(out_root / f["path"]) for f in obs["output_files"] if f["path"].endswith(".txt") and not f["path"].endswith("summary.txt")
        ]
        obs["refusal_lines"] = refusal_lines(stdout, stderr)
        outcome, reasons = classify_test(rc, timed_out, parsed, child.expect_tests)
        if kind == "sampled_test":
            blob = "".join(p.read_text(errors="replace") for p in sorted(out_root.rglob("*.json"))) if out_root.exists() else ""
            obs["sampled_configs"] = [
                {"path": str(p.relative_to(cdir)), "content": _safe_json(p)}
                for p in sorted(out_root.rglob("*.config.json"))
            ]
            obs["mode_in_receipts"] = child.mode in blob
            if child.ef_sampled:
                cov = collect_coverage(out_root)
                obs["coverage"] = cov
                gaps = q8ef_coverage_gaps(cov, graph)
                if gaps:
                    reasons.append(f"coverage counters missing or zero: {gaps}")
                if not obs["mode_in_receipts"]:
                    reasons.append("receipts/config do not record mode q8_ef")
                if reasons and outcome == "ok":
                    outcome = "incomplete"
                obs["g5_result"] = {"ok": "PASS", "timeout": "FAIL", "failed": "FAIL"}.get(outcome, "INCOMPLETE")
        if reasons:
            obs["reasons"] = reasons
        return outcome, obs
    if kind == "kld_eval":
        parsed = parse_kld_stderr(stderr)
        obs["stderr_summary"] = parsed
        seq = cdir / (child.kldseq or "")
        obs["kldseq_present"] = seq.is_file()
        ok = rc == 0 and bool(parsed["summaries"]) and seq.is_file()
        return ("timeout" if timed_out else ("ok" if ok else "failed")), obs
    if kind == "greedy_run":
        report = find_json_object(stdout)
        content = report.get("content") if report is not None else None
        if report is None:
            obs["error"] = "no JSON line on stdout"
        elif not isinstance(content, str):
            obs["error"] = "JSON line has no string content"
        else:
            obs["content"] = content
            obs["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            obs["content_chars"] = len(content)
            if content == "":
                obs["error"] = "empty content"
        if report is not None:
            obs["tokens"] = report.get("tokens")
            obs["tok_s"] = report.get("tok_s")
            obs["finish_reason"] = report.get("finish_reason")
        # The result line carries generated text: scan only the remaining stdout for daemon/refusal lines.
        log_stdout = "\n".join(line for line in stdout.splitlines() if not line.startswith('{"content"'))
        obs["daemon_log_lines"] = daemon_log_lines(stderr, log_stdout)
        obs["refusal_lines"] = refusal_lines(stderr, log_stdout)
        mem = cdir / "memory_samples.jsonl"
        if mem.is_file():
            samples = [json.loads(line) for line in mem.read_text().splitlines() if line.strip()]
            obs["memory"] = summarize_memory(samples, None)
        ok = rc == 0 and isinstance(content, str) and content != ""
        return ("timeout" if timed_out else ("ok" if ok else "failed")), obs
    if kind.startswith("bench_"):
        report = find_json_object(stdout)
        if report is None:
            obs["error"] = "no JSON report on stdout"
        else:
            obs.update(extract_bench(report))
            if kind == "bench_memory":
                obs["loaded"] = report.get("loaded")
                obs["kv_backend_fields"] = {k: v for k, v in report.items() if "kv" in k.lower()}
        if kind == "bench_memory":
            obs["daemon_log_lines"] = daemon_log_lines(stderr, stdout)
            obs["refusal_lines"] = refusal_lines(stderr, stdout)
            mem = cdir / "memory_samples.jsonl"
            if mem.is_file():
                samples = [json.loads(line) for line in mem.read_text().splitlines() if line.strip()]
                obs["memory"] = summarize_memory(samples, None)
        ok = rc == 0 and report is not None
        return ("timeout" if timed_out else ("ok" if ok else "failed")), obs
    raise CollectError(f"unhandled child kind {kind}")


def _safe_json(path: Path):
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Running children
# ---------------------------------------------------------------------------


def input_roles(cfg: Config) -> list[tuple[str, Path]]:
    if cfg.oracle_only:
        return []  # the oracle loads no model: hash nothing the run does not read
    roles: list[tuple[str, Path]] = [("model", cfg.model)]
    for suffix in (".ple-q4", ".ple-q8"):
        sidecar = Path(str(cfg.model) + suffix)
        if sidecar.is_file():
            roles.append((f"model{suffix}", sidecar))
    if cfg.stage != "greedy":
        roles += [("wt2-ref", cfg.wt2_ref), ("code-ref", cfg.code_ref)]
    if cfg.prompt_file:
        roles.append(("prompt", cfg.prompt_file))
    if cfg.long_prompt_file:
        roles.append(("long-prompt", cfg.long_prompt_file))
    return roles


def host_record(environ: dict) -> dict:
    return {
        "host": socket.gethostname(),
        "uname": " ".join(os.uname()),
        "FABRIC_JOB": environ.get("FABRIC_JOB"),
        "FABRIC_CARDS": environ.get("FABRIC_CARDS"),
        "HIPFIRE_DEVICES": environ.get("HIPFIRE_DEVICES"),
    }


def build_identity(cfg: Config, hasher: Hasher, environ: dict) -> dict:
    return {
        "binaries": {
            base: {
                "path": str(b["path"]), "name": b["name"], "manifest_sha256": b["sha256"],
                **{k: v for k, v in hasher.hash(b["path"]).items() if k in ("md5", "sha256", "size", "cached")},
            }
            for base, b in sorted(cfg.binaries.items())
        },
        "inputs": {
            role: {k: v for k, v in hasher.hash(path).items() if k in ("path", "md5", "sha256", "size", "cached")}
            for role, path in input_roles(cfg)
        },
        "modes": list(cfg.modes),
        "graphs": list(cfg.graphs),
        "max_seq": cfg.max_seq,
        "host": host_record(environ),
    }


def kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_child(cfg: Config, child: Child, identity: dict, environ: dict) -> dict:
    cdir = child.dir
    (cdir / "home" / ".hipfire").mkdir(parents=True, exist_ok=True)
    (cdir / "lockdir").mkdir(parents=True, exist_ok=True)
    for rel, text in child.home_files.items():
        write_text(cdir / "home" / ".hipfire" / rel, text)
    env = build_child_env(environ, cfg, child)
    write_text(cdir / "command.txt", shlex.join(child.argv) + "\n")
    write_json(cdir / "env.json", env)
    cid = dict(identity)
    cid["child"] = {
        "id": child.id, "mode": child.mode, "graph": child.graph, "binary": child.binary,
        "state_quant_env": {k: v for k, v in child.extra_env.items() if k in ("HIPFIRE_STATE_QUANT", "HIPFIRE_GDN_TEST_STATE_QUANT")},
        "home_files": dict(child.home_files),
        "child_env": dict(cfg.child_env),
    }
    write_json(cdir / "identity.json", cid)
    sampler = MemorySampler(cdir / "memory_samples.jsonl") if child.sample_memory else None
    start = utc_now()
    write_text(cdir / "start.txt", start + "\n")
    t0 = time.monotonic()
    timed_out = False
    rc: Optional[int]
    with open(cdir / "stdout.txt", "wb") as out, open(cdir / "stderr.txt", "wb") as err:
        if sampler:
            sampler.start()
        try:
            proc = subprocess.Popen(
                child.argv, cwd=cdir, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=cfg.child_timeout_secs)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(proc)
                rc = proc.returncode
        except OSError as exc:
            rc = 127
            err.write(f"collect.py: cannot launch {child.argv[0]}: {exc}\n".encode())
        finally:
            if sampler:
                sampler.finish()
    end = utc_now()
    write_text(cdir / "end.txt", end + "\n")
    write_text(cdir / "rc.txt", f"{rc}\n")
    if timed_out:
        write_text(cdir / "timeout.txt", f"timed out after {cfg.child_timeout_secs} s\n")
    outcome, obs = observe_child(child, cdir, rc, timed_out, child.graph)
    write_json(cdir / "observations.json", obs)
    return {
        "id": child.id, "dir": str(cdir.relative_to(cfg.run_dir)), "label": child.label, "kind": child.kind,
        "run": child.run, "graph": child.graph, "graph_axis": graph_axis(child.kind), "mode": child.mode, "corpus": child.corpus, "route": child.route,
        "kldseq": child.kldseq, "argv": child.argv, "rc": rc, "timed_out": timed_out,
        "start": start, "end": end, "seconds": round(time.monotonic() - t0, 3),
        "outcome": outcome, "observations": obs,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def stage_analysis(stage: str, run_dir: Path, children: list[dict]) -> Optional[dict]:
    if stage == "quality":
        return analyze_quality(run_dir, children)
    if stage == "perf":
        return analyze_perf(children)
    if stage == "greedy":
        return analyze_greedy(children)
    return None


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(receipt: dict) -> str:
    lines = [
        f"# GDN state-precision collect: stage {receipt['stage']}, run {receipt['run_id']}",
        "",
        f"- start {receipt['start']}  end {receipt['end']}",
        f"- modes {','.join(receipt['modes'])}  graphs {','.join(receipt['graphs'])}  runs {receipt['runs']}  max_seq {receipt['max_seq']}",
        f"- only {','.join(receipt['only'])}" if receipt.get("only") else "- only (all planned children)",
        f"- max_chunks {receipt['max_chunks']}" if receipt.get("max_chunks") is not None else "- max_chunks n/a",
        f"- greedy_tokens {receipt['greedy_tokens']}" if receipt.get("greedy_tokens") is not None else "- greedy_tokens n/a",
        f"- fabric job {receipt['host'].get('FABRIC_JOB')}  cards {receipt['host'].get('FABRIC_CARDS')}  host {receipt['host'].get('host')}",
        f"- exit code {receipt['exit_code']}  not run: {len(receipt['not_run'])}",
        "",
        "## Children",
        "",
        "| id | rc | outcome | seconds |",
        "|---|---|---|---|",
    ]
    for c in receipt["children"]:
        lines.append(f"| {c['id']} | {c['rc']} | {c['outcome']} | {c['seconds']} |")
    analysis = receipt.get("analysis")
    if analysis and receipt["stage"] == "quality":
        lines += ["", "## Paired quality (candidate - q8, per-chunk mean KLD)", "", f"Rule: `{analysis['rule']}`", "",
                  "| graph | corpus | route | run | candidate | n | mean | ci95 lo | ci95 hi | rule applied |", "|---|---|---|---|---|---|---|---|---|---|"]
        for p in analysis["pairs"]:
            lines.append(
                f"| {p['graph']} | {p['corpus']} | {p['route']} | {p['run']} | {p['candidate']} | {_fmt(p.get('n_chunks'))} | "
                f"{_fmt(p.get('mean_delta'))} | {_fmt(p.get('ci95_lo'))} | {_fmt(p.get('ci95_hi'))} | {p.get('rule_applied', p.get('error'))} |"
            )
        lines += ["", "## Arm statistics", "", "| arm | mode | run | mean KLD | p99 KLD | NLL | top1 | nonfinite |", "|---|---|---|---|---|---|---|---|"]
        for arm, by_mode in analysis["arms"].items():
            for mode, runs in by_mode.items():
                for run, s in runs.items():
                    lines.append(
                        f"| {arm} | {mode} | {run} | {_fmt(s['mean_kld'])} | {_fmt(s['p99_kld_of_chunk_p99'])} | {_fmt(s['mean_nll'])} | "
                        f"{_fmt(s['top1_from_stderr'], 4)} | {s['nonfinite_kld_values'] + s['nonfinite_nll_values']} |"
                    )
        if analysis["code24_decode_gap_closure"]:
            lines += ["", "## code24 decode gap closure (q8_ef vs q8,fp32)", ""]
            lines += [f"- {json.dumps(c, sort_keys=True)}" for c in analysis["code24_decode_gap_closure"]]
        lines += ["", "## Same-route determinism (byte-identical .kldseq across runs)", ""]
        for d in analysis["same_route_determinism"]:
            lines.append(
                f"- g{d['graph']}/{d['corpus']}/{d['route']}/{d['mode']}: identical={d['kldseq_byte_identical']} "
                f"sha256={d['kldseq_sha256']}"
            )
        if analysis["errors"]:
            lines += ["", "## Analysis errors", ""] + [f"- {e}" for e in analysis["errors"]]
    elif analysis and receipt["stage"] == "perf":
        lines += ["", "## Perf medians (loss > 0 means worse than q8; 5% line is an observation, not a verdict)", "",
                  "| graph | command | metric | mode | median | loss vs q8 (medians) | loss vs q8 (median of paired runs) | exceeds 5% line |", "|---|---|---|---|---|---|---|---|"]
        for r in analysis["rows"]:
            pol = r.get("policy_line_observation") or {}
            lines.append(
                f"| {r['graph']} | {r['label']} | {r['metric']} | {r['mode']} | {_fmt(r['median'], 3)} | "
                f"{_fmt(r.get('loss_vs_q8_of_medians'), 4)} | {_fmt(r.get('loss_vs_q8_median_of_paired_runs'), 4)} | {pol.get('loss_of_medians_exceeds_line', '-')} |"
            )
        if analysis["errors"]:
            lines += ["", "## Analysis errors", ""] + [f"- {e}" for e in analysis["errors"]]
    elif analysis and receipt["stage"] == "greedy":
        lines += ["", "## Greedy MTP vs AR text (spec off vs mtp, `--temp 0`)", "",
                  f"- fp32_mtp_equals_ar [{analysis['labels']['fp32_mtp_equals_ar']}]: {analysis['fp32_mtp_equals_ar']}",
                  f"- q8_mtp_equals_ar [{analysis['labels']['q8_mtp_equals_ar']}]: {analysis['q8_mtp_equals_ar']}",
                  f"- all_loaded (explicit --max-seq admission held for every arm): {analysis['all_loaded']}", "",
                  "| run | graph | mode | equal | first diff char | len off | len mtp | tokens off | tokens mtp |", "|---|---|---|---|---|---|---|---|---|"]
        for p in analysis["pairs"]:
            lines.append(
                f"| {p['run']} | {p['graph']} | {p['mode']} | {p['equal']} | {_fmt(p['first_diff_char'])} | {p['len_off']} | "
                f"{p['len_mtp']} | {_fmt(p['tokens_off'])} | {_fmt(p['tokens_mtp'])} |"
            )
        if analysis["cross_mode_q8_vs_fp32"]:
            lines += ["", f"### q8 vs fp32 content [{analysis['labels']['cross_mode_q8_vs_fp32']}]", ""]
            lines += [f"- {json.dumps(x, sort_keys=True)}" for x in analysis["cross_mode_q8_vs_fp32"]]
        lines += ["", "### Admission log lines (max_seq / card_cap / admission)", ""]
        for ch in analysis["children"]:
            lines.append(f"- {ch['id']} ({ch['outcome']}): {ch['daemon_admission_lines']}")
        if analysis["errors"]:
            lines += ["", "## Analysis errors", ""] + [f"- {e}" for e in analysis["errors"]]
    for c in receipt["children"]:
        obs = c.get("observations") or {}
        if obs.get("g5_result"):
            lines += ["", f"G5 {c['id']}: {obs['g5_result']}  reasons: {obs.get('reasons', [])}"]
        if c["kind"] == "bench_memory":
            lines += ["", f"Memory {c['id']}: {json.dumps(obs.get('memory'), sort_keys=True)}",
                      f"  refusals: {obs.get('refusal_lines')}"]
    if receipt["not_run"]:
        lines += ["", "## Not run (stopped at first failure)", ""] + [f"- {i}" for i in receipt["not_run"]]
    return "\n".join(lines) + "\n"


def load_children(run_dir: Path) -> list[dict]:
    path = run_dir / "children.json"
    if not path.is_file():
        raise CollectError(f"{path} not found: not a collector run dir")
    return read_json(path)


def analyze_existing(cfg: Config) -> int:
    stage_dir = cfg.out / cfg.stage
    if cfg.run_id:
        run_dir = stage_dir / cfg.run_id
    else:
        runs = sorted(p for p in stage_dir.iterdir() if p.is_dir()) if stage_dir.is_dir() else []
        if not runs:
            raise CollectError(f"--analyze: no run dirs under {stage_dir}")
        run_dir = runs[-1]
    children = load_children(run_dir)
    analysis = stage_analysis(cfg.stage, run_dir, children)
    write_json(run_dir / "analysis.json", analysis)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    print(f"analysis written to {run_dir / 'analysis.json'}", file=sys.stderr)
    return 0


def print_dry_run(cfg: Config, children: list[Child]) -> None:
    print(f"# dry run: stage {cfg.stage}  run dir {cfg.run_dir}  {len(children)} child process(es)")
    print(
        f"# scrub: unset every HIPFIRE_* except {', '.join(KEEP_HIPFIRE_ENV)} and the --child-env KEY=VALUE pairs; "
        "per-child isolated HOME/HIPFIRE_HOME/HIPFIRE_LOCK_DIR; a parent HIPFIRE_* the scrub would drop is refused "
        "(pass it with --child-env or unset it)"
    )
    if cfg.child_env:
        print("# child-env: " + " ".join(f"{k}={shlex.quote(v)}" for k, v in cfg.child_env.items()))
    print(f"# per-child timeout {cfg.child_timeout_secs} s; fabric hold required for a real run (FABRIC_JOB wrapper = ancestor, {HOSTMEM_LOCK} held)")
    if cfg.only:
        print(f"# only: {','.join(cfg.only)}")
    if cfg.stage == "quality":
        print(f"# max-chunks: {cfg.max_chunks}")
    if cfg.stage == "greedy":
        print(f"# greedy-tokens: {cfg.greedy_tokens}")
    for child in children:
        env = explicit_env(cfg, child)
        print(f"[{child.id}]")
        print("  env  " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())))
        print("  argv " + shlex.join(child.argv))


def run_stage(cfg: Config, environ: dict) -> int:
    children = plan_children(cfg)
    check_session_test_list(cfg, children)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    hasher = Hasher(cfg.out / "hash-cache.json")
    write_text(run_dir / "command.txt", shlex.join([sys.executable, *sys.argv]) + "\n")
    identity = build_identity(cfg, hasher, environ)
    write_json(run_dir / "identity.json", identity)
    start = utc_now()
    done: list[dict] = []
    not_run: list[str] = []
    failed = False
    for child in children:
        if failed and not cfg.keep_going:
            not_run.append(child.id)
            continue
        print(f"[collect] {child.id}", file=sys.stderr, flush=True)
        result = run_child(cfg, child, identity, environ)
        done.append(result)
        write_json(run_dir / "children.json", done)
        print(f"[collect] {child.id} rc={result['rc']} outcome={result['outcome']} {result['seconds']}s", file=sys.stderr, flush=True)
        if result["outcome"] != "ok":
            failed = True
    analysis = stage_analysis(cfg.stage, run_dir, done)
    if analysis is not None:
        write_json(run_dir / "analysis.json", analysis)
    exit_code = 0 if (done and all(c["outcome"] == "ok" for c in done) and not not_run) else 1
    receipt = {
        "schema": "gdn-state-precision-collect-v1",
        "note": "observed data only; rule_applied/g5_result fields name the rule they applied and are not inferred verdicts",
        "stage": cfg.stage, "run_id": cfg.run_id, "start": start, "end": utc_now(),
        "modes": list(cfg.modes), "graphs": list(cfg.graphs), "runs": cfg.runs, "max_seq": cfg.max_seq,
        "child_timeout_secs": cfg.child_timeout_secs, "host": host_record(environ),
        "identity": "identity.json", "children": done, "not_run": not_run,
        "analysis": analysis, "exit_code": exit_code, "only": list(cfg.only),
        "child_env": dict(cfg.child_env),
        "max_chunks": cfg.max_chunks if cfg.stage == "quality" else None,
        "greedy_tokens": cfg.greedy_tokens if cfg.stage == "greedy" else None,
    }
    write_json(run_dir / "receipt.json", receipt)
    write_text(run_dir / "report.md", render_report(receipt))
    print(f"[collect] receipt {run_dir / 'receipt.json'} exit {exit_code}", file=sys.stderr)
    return exit_code


def prehash(cfg: Config) -> int:
    cfg.out.mkdir(parents=True, exist_ok=True)
    hasher = Hasher(cfg.out / "hash-cache.json")
    for role, path in input_roles(cfg):
        h = hasher.hash(path)
        print(f"{role}: md5 {h['md5']} sha256 {h['sha256']} size {h['size']} cached={h['cached']} {path}")
    return 0


def run(argv: Optional[list[str]] = None, environ: Optional[dict] = None) -> int:
    environ = dict(os.environ if environ is None else environ)
    args = make_parser().parse_args(argv)
    cfg = validate(args, environ)
    if args.analyze:
        return analyze_existing(cfg)
    if args.prehash:
        return prehash(cfg)
    if cfg.dry_run:
        print_dry_run(cfg, plan_children(cfg))
        return 0
    return run_stage(cfg, environ)


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run(argv)
    except CollectError as exc:
        print(f"collect.py: refusing: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
