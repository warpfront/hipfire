#!/usr/bin/env python3
"""Shared constants + identity helpers for the S1 pp16384 >= 2000 tok/s ABBAAB performance harness.

Everything the cell, plan generator, stage helper and scorer must agree on lives here, so the scripts cannot
drift: the H2 argv, the pinned env, the arm definitions (A = all S1 levers off, B = placement spread + IU4 panels,
combination `place_iu4`, QSA gather off), the cell order, the synthetic token stream,
and the stage identity capture/compare.
CPU only. Never reads or hashes the model payload (model identity = cached stat only).
"""
import hashlib
import json
import os
import re
import stat
import struct
import sys

SCHEMA_PINS = "slowlink-gap.pins.v1"
SCHEMA_STAGE = "slowlink-gap.stage.v1"
HERE = os.path.dirname(os.path.abspath(__file__))

MODEL = "/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4"
PLE = MODEL + ".ple-q4"
UUID = "GPU-43390a851e296ee5"
BDF = "0000:67:00.0"
ARCH = "gfx1100"
PP_TOKENS = 16384
MAX_SEQ = 16416
RUNS = 3
WARMUPS = 1
# ABBAAB = A1 B1 B2 A2 A3 B3; every cell is one fresh `hipfire bench` process with 3 measured daemon runs.
ORDER = ["A1", "B1", "B2", "A2", "A3", "B3"]
GATE_TOK_S = 2000.0

# H2 argv (hipfire bench MODEL --matrix --pp16384 --tg1 --ctx1 --runs3 --warmups1 --max-seq16416 --json), in the
# exact spelling the CLI was driven with in the earlier gap2000 cells.
ARGV_TAIL = ["bench", MODEL, "--matrix", "--pp", str(PP_TOKENS), "--tg", "1", "--ctx", "1", "--runs", str(RUNS),
             "--warmups", str(WARMUPS), "--max-seq", str(MAX_SEQ), "--json"]

# Env applied to BOTH arms. All arms keep the externally committed state at q8 on the existing q8-cadence runtime
# (no f32 carry, no env switch for one, and NO global f32). D2 off. QSA gather is pinned OFF in both arms
# (the qualified combination is PLACE spread + IU4 only); there is no Q8 trunk WMMA lever on this runtime, so any
# such variable in a cell env is drift, not a lever.
PINNED_ENV = {
    "HIPFIRE_STATE_QUANT": "q8",
    "HIPFIRE_KV_MODE": "q8",
    "HIPFIRE_QWEN4_D2_DEFAULT": "0",
    "HIPFIRE_QWEN4_QSA_WMMA_GATHER": "0",
}
# Explicitly named qualified combination: PLACE spread + IU4 panels, QSA=0, Q8w inert/absent, arm A all off.
COMBINATION = "place_iu4"
# The only variables that may differ between arms: placement and IU4 expert panels.
PLACE_KEY = "HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS"
IU4_KEY = "HIPFIRE_QWEN4_MOE_SYM_IU4"
ARM_KEYS = (PLACE_KEY, IU4_KEY)
DEFAULT_PLACE = "0,3,7,15,19,23,27,39,43"
PLACE_RE = re.compile(r"^(front|spread|[0-9]{1,2}(,[0-9]{1,2})*)$")
# Fixed per-cell env the cell script adds itself (paths are per cell and not pinned).
CELL_ENV_KEYS = ("HOME", "HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_DEVICES",
                 "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HIPFIRE_NO_REGISTRY_FETCH")
CELL_PATH_KEYS = ("HOME", "HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE")


def die(msg, code=2):
    print(f"gap: {msg}", file=sys.stderr)
    sys.exit(code)


def arm_env(arm, place):
    if arm == "A":
        return {}
    if arm == "B":
        if not PLACE_RE.match(place):
            die(f"bad placement value {place!r}")
        return {PLACE_KEY: place, IU4_KEY: "1"}
    die(f"unknown arm {arm!r}")


def expected_env(arm, place):
    """Exact HIPFIRE_/HIP_/ROCR_ env (minus per-cell paths) an arm must apply."""
    env = dict(PINNED_ENV)
    env.update(arm_env(arm, place))
    env["HIPFIRE_DEVICES"] = UUID
    env["ROCR_VISIBLE_DEVICES"] = UUID
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["HIPFIRE_NO_REGISTRY_FETCH"] = "1"
    return env


def synthetic_stream_digest(n=PP_TOKENS):
    """sha256 over little-endian u32 ids of the daemon's deterministic bench_prefill stream:
    `(0..n).map(|i| 10 + (i % 1000))` (crates/hipfire-daemon/src/bench.rs, handle_bench_prefill)."""
    h = hashlib.sha256()
    h.update(struct.pack("<%dI" % n, *[10 + (i % 1000) for i in range(n)]))
    return h.hexdigest()


SYNTHETIC = {
    "formula": "ids[i] = 10 + (i % 1000), i in 0..tokens, u32 little endian",
    "source": "crates/hipfire-daemon/src/bench.rs handle_bench_prefill",
    "tokens": PP_TOKENS,
}


def hash_file(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            sha.update(b)
            md5.update(b)
            size += len(b)
    return sha.hexdigest(), md5.hexdigest(), size


def sha256_file(path):
    return hash_file(path)[0]


def file_identity(path, hashes=True):
    """Identity of one stage entry: symlink structure, canonical target, inode/size/mtime_ns, sha256+md5."""
    lst = os.lstat(path)
    d = {"is_symlink": stat.S_ISLNK(lst.st_mode)}
    if d["is_symlink"]:
        d["readlink"] = os.readlink(path)
    d["canonical"] = os.path.realpath(path)
    st = os.stat(path)
    if not stat.S_ISREG(st.st_mode):
        die(f"{path}: not a regular file after symlink resolution")
    d.update(ino=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns, mode=stat.S_IMODE(st.st_mode))
    if hashes:
        d["sha256"], d["md5"], n = hash_file(path)
        if n != st.st_size:
            die(f"{path}: size changed while hashing")
    return d


def kernel_identity(kdir, hashes=True):
    """Kernel pack arch dir (symlink or directory): every file's ino/size/mtime_ns/sha256 plus a tree digest."""
    canonical = os.path.realpath(kdir)
    if not os.path.isdir(canonical):
        die(f"{kdir}: kernel pack dir missing")
    d = {"is_symlink": os.path.islink(kdir), "canonical": canonical, "files": {}}
    if d["is_symlink"]:
        d["readlink"] = os.readlink(kdir)
    tree = hashlib.sha256()
    n_hsaco = 0
    for name in sorted(os.listdir(canonical)):
        p = os.path.join(canonical, name)
        st = os.lstat(p)
        if not stat.S_ISREG(st.st_mode):
            die(f"{p}: kernel pack entry is not a regular file")
        e = {"size": st.st_size, "ino": st.st_ino, "mtime_ns": st.st_mtime_ns}
        if hashes:
            e["sha256"] = sha256_file(p)
            tree.update(f"{name}\0{st.st_size}\0{e['sha256']}\n".encode())
        d["files"][name] = e
        n_hsaco += name.endswith(".hsaco")
    d["file_count"] = len(d["files"])
    d["hsaco_count"] = n_hsaco
    if hashes:
        d["tree_sha256"] = tree.hexdigest()
    return d


def stage_capture(bin_dir, hashes=True):
    """Capture a symlink-only stage bin dir: bin/{hipfire,daemon,<probes>} symlinks + bin/kernels/compiled/gfx1100."""
    bin_dir = os.path.abspath(bin_dir)
    if not os.path.isdir(bin_dir):
        die(f"stage bin dir missing: {bin_dir}")
    names = sorted(os.listdir(bin_dir))
    for need in ("hipfire", "daemon", "kernels"):
        if need not in names:
            die(f"stage {bin_dir}: missing {need}")
    out = {"bin_dir": bin_dir, "entries": {}}
    for name in names:
        p = os.path.join(bin_dir, name)
        if name == "kernels":
            if os.path.islink(p) or not os.path.isdir(p):
                die(f"{p}: must be a real directory holding compiled/{ARCH}")
            if os.listdir(p) != ["compiled"] or os.path.islink(os.path.join(p, "compiled")):
                die(f"{p}: must contain only a real compiled/ dir")
            if os.listdir(os.path.join(p, "compiled")) != [ARCH]:
                die(f"{p}/compiled: must contain only {ARCH}")
            out["kernels"] = kernel_identity(os.path.join(p, "compiled", ARCH), hashes)
            continue
        if not os.path.islink(p):
            die(f"{p}: stage entries must be symlinks to artifact executables (found a real file/dir)")
        e = file_identity(p, hashes)
        if not os.access(p, os.X_OK):
            die(f"{p}: target not executable")
        if os.path.dirname(e["canonical"]) == bin_dir:
            die(f"{p}: symlink resolves inside the stage")
        out["entries"][name] = e
    return out


def flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, key + "."))
        else:
            flat[key] = v
    return flat


HASH_KEYS = ("sha256", "md5", "tree_sha256")


def diff_captures(want, got, stat_only=False):
    a, b = flatten(want), flatten(got)
    if stat_only:
        a = {k: v for k, v in a.items() if k.rsplit(".", 1)[-1] not in HASH_KEYS}
        b = {k: v for k, v in b.items() if k.rsplit(".", 1)[-1] not in HASH_KEYS}
    diffs = []
    for k in sorted(set(a) | set(b)):
        if a.get(k, "<absent>") != b.get(k, "<absent>"):
            diffs.append(f"{k}: pinned={a.get(k, '<absent>')!r} actual={b.get(k, '<absent>')!r}")
    return diffs


def model_stat(path):
    st = os.stat(path)
    return {"path": path, "ino": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def load_pins(path):
    with open(path) as f:
        pins = json.load(f)
    if pins.get("schema") != SCHEMA_PINS:
        die(f"{path}: not a {SCHEMA_PINS} file")
    return pins


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
