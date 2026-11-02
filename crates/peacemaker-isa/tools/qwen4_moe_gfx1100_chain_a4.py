#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only A4 oracle prep for the Consumer24 gfx1100 qwen4 MoE chain (K2).

Input : DIR/chain-h.json  (schema consumer24.gfx1100.chain-h.v1), written by the Rust oracle `--prepare-chain DIR`;
        every entry names a raw little-endian BF16 file `h_file` (relative to DIR) of shape [rows, 640] (zero rows are
        padding and stay in place) and its `h_sha256`.
Output: DIR/<entry name>.a4.bin   epoch-major A4 sidecar [5][rows][72 B] (block_i4_128 records: {f32 d, i32 s, 64 qs},
                                  element j in byte j/2, low nibble first), the bytes the gfx1100 producer must write
        DIR/chain-a4.json         (schema consumer24.gfx1100.chain-a4.v1), written LAST, after every A4 file is on disk.

Per row the bytes are exactly those of the certified gfx1201 K3 oracle (`mq_i4_a4_oracle.expected_bytes`): the sibling
`mq_i4_a4_oracle.rotate` (R128 rotation, canonical signs gen_fwht_signs(43) / gen_fwht_signs(1043)) applied directly to
the uint16 H, then `pilot_solve.a4_emulate(rotated, candidates=2)` (token-major [rows][5][72]) transposed to epoch-major.
Rows are independent, so the work is chunked without changing a byte.

Hard failures (no fallback): pilot_solve.py hash != pin, wrong/missing/odd keys, H file size or sha256 mismatch,
non-finite BF16 in H, non-finite rotation output, seeds other than 43/1043, cols != 640, unsafe or duplicate names,
a torch build that can see a GPU. GPUs are disabled up front (HIP_VISIBLE_DEVICES=-1, ROCR_VISIBLE_DEVICES=-1,
CUDA_VISIBLE_DEVICES=) before torch is imported.

Usage:
  python3 crates/peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py --chain-dir DIR \
      [--pilot-solve /home/kaden/qcal/release-0.4.2/subq4/calib2/pilot_solve.py]
"""
from __future__ import annotations

import os

# GPU prohibition: must precede any torch import.
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import types

import numpy as np

IN_SCHEMA = "consumer24.gfx1100.chain-h.v1"
OUT_SCHEMA = "consumer24.gfx1100.chain-a4.v1"
IN_NAME = "chain-h.json"
OUT_NAME = "chain-a4.json"
PILOT = "/home/kaden/qcal/release-0.4.2/subq4/calib2/pilot_solve.py"
PILOT_SHA256 = "53965f2328e32d86c7b87460a02573a065424e220383891da210441dc9f6c0df"
ROTATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mq_i4_a4_oracle.py")
K, G, REC = 640, 5, 72
S1_SEED, S2_SEED = 43, 1043
CANDIDATES = 2
CHUNK_ROWS = 1 << 15
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ENTRY_REQUIRED = ("name", "h_file", "h_sha256", "rows", "cols", "s1_seed", "s2_seed")
ENTRY_ADDED = ("a4_file", "a4_sha256", "a4_bytes", "a4_layout")
ROOT_ADDED = ("pilot_solve", "harness", "rotation_source", "input_manifest_sha256", "input_schema", "numpy", "torch",
              "candidates", "gpu_disabled", "rotation_selftest")


def die(msg):
    raise SystemExit(f"chain_a4: {msg}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_verified(path, want_sha, what):
    with open(path, "rb") as f:
        data = f.read()
    got = sha256_bytes(data)
    if want_sha is not None and got != want_sha:
        die(f"{what} sha256 mismatch: {path} is {got}, expected {want_sha}")
    return data, got


def exec_module(name, path, data):
    """Import `path` from the already-hashed bytes (no re-read between hash check and execution)."""
    module = types.ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    exec(compile(data, path, "exec"), module.__dict__)
    return module


def int_field(entry, key, where):
    v = entry[key]
    if isinstance(v, bool) or not isinstance(v, int):
        die(f"{where}: `{key}` must be an integer, got {v!r}")
    return v


def reject_const(c):
    raise ValueError(f"non-finite JSON constant {c}")


def atomic_write(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_inside(root_real, rel, where):
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\\" in rel or "\0" in rel or ".." in rel.split("/"):
        die(f"{where}: `h_file` must be a relative path inside the chain dir, got {rel!r}")
    full = os.path.realpath(os.path.join(root_real, rel))
    if os.path.commonpath([root_real, full]) != root_real:
        die(f"{where}: `h_file` {rel!r} escapes the chain dir")
    return full


def a4_for_rows(oracle, pilot, h_u16, s1, s2):
    """[5][rows][72] uint8 for uint16 H [rows, 640]; per-chunk rotate + a4_emulate, transposed to epoch-major."""
    rows = h_u16.shape[0]
    out = np.empty((G, rows, REC), np.uint8)
    for r0 in range(0, rows, CHUNK_ROWS):
        r1 = min(rows, r0 + CHUNK_ROWS)
        rot = oracle.rotate(h_u16[r0:r1], s1, s2)
        if not np.isfinite(rot).all():
            die(f"non-finite rotation output in rows {r0}..{r1}")
        recs = np.frombuffer(pilot.a4_emulate(rot.reshape(r1 - r0, K), CANDIDATES), np.uint8)
        if recs.size != (r1 - r0) * G * REC:
            die(f"a4_emulate returned {recs.size} bytes for {r1 - r0} rows, expected {(r1 - r0) * G * REC}")
        out[:, r0:r1] = recs.reshape(r1 - r0, G, REC).transpose(1, 0, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--pilot-solve", default=PILOT)
    a = ap.parse_args()

    self_path = os.path.abspath(__file__)
    chain_dir = os.path.abspath(a.chain_dir)
    if not os.path.isdir(chain_dir):
        die(f"--chain-dir {chain_dir} is not a directory")
    root_real = os.path.realpath(chain_dir)

    # 1. Pilot hash BEFORE import; rotation harness hash recorded, then both imported from the hashed bytes.
    pilot_path = os.path.abspath(a.pilot_solve)
    pilot_src, pilot_sha = read_verified(pilot_path, PILOT_SHA256, "pilot_solve.py")
    rot_path = os.path.realpath(ROTATION)
    rot_src, rot_sha = read_verified(rot_path, None, "rotation harness")
    _, self_sha = read_verified(self_path, None, "this tool")

    import torch
    if (torch.version.cuda or torch.version.hip) and torch.cuda.device_count() != 0:
        die("torch can see a GPU despite HIP/ROCR/CUDA_VISIBLE_DEVICES masking; refusing to run")
    if torch.cuda.is_available():
        die("torch.cuda.is_available() is true; this tool is CPU-only")

    pilot = exec_module("pilot_solve", pilot_path, pilot_src)
    oracle = exec_module("mq_i4_a4_oracle", rot_path, rot_src)
    if getattr(pilot, "a4_emulate", None) is None or getattr(oracle, "rotate", None) is None or getattr(oracle, "gen_fwht_signs", None) is None:
        die("pilot_solve.a4_emulate / oracle.rotate / oracle.gen_fwht_signs missing")
    if (oracle.K, oracle.G, oracle.REC) != (K, G, REC):
        die(f"rotation harness geometry {(oracle.K, oracle.G, oracle.REC)} != {(K, G, REC)}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        oracle.selftest()  # SystemExit on mismatch of the vectorised rotation vs the lane-for-lane HIP transcription
    selftest = json.loads(buf.getvalue().strip().splitlines()[-1])
    if selftest.get("selftest") != "pass":
        die(f"rotation selftest did not pass: {selftest}")
    s1, s2 = oracle.gen_fwht_signs(S1_SEED, 128), oracle.gen_fwht_signs(S2_SEED, 128)

    # 2. Input manifest.
    in_path = os.path.join(chain_dir, IN_NAME)
    in_bytes, in_sha = read_verified(in_path, None, IN_NAME)
    try:
        root = json.loads(in_bytes, parse_constant=reject_const)
    except ValueError as e:
        die(f"{IN_NAME} is not valid JSON: {e}")
    if not isinstance(root, dict) or root.get("schema") != IN_SCHEMA:
        die(f"{IN_NAME}: schema must be {IN_SCHEMA!r}, got {root.get('schema') if isinstance(root, dict) else type(root).__name__}")
    entries = root.get("entries")
    if not isinstance(entries, list) or not entries:
        die(f"{IN_NAME}: `entries` must be a non-empty list")
    clash = [k for k in ROOT_ADDED if k in root]
    if clash:
        die(f"{IN_NAME}: root already has keys this tool adds: {clash}")

    # Validate every entry before any output exists.
    seen_names, seen_files = set(), set()
    h_paths = []
    for i, e in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(e, dict):
            die(f"{where}: not an object")
        missing = [k for k in ENTRY_REQUIRED if k not in e]
        if missing:
            die(f"{where}: missing keys {missing}")
        clash = [k for k in ENTRY_ADDED if k in e]
        if clash:
            die(f"{where}: already has keys this tool adds: {clash}")
        name = e["name"]
        if not isinstance(name, str) or not SAFE_NAME.match(name) or name.endswith(".") or name in (IN_NAME, OUT_NAME):
            die(f"{where}: unsafe `name` {name!r} (need [A-Za-z0-9][A-Za-z0-9_.-]*, <=120 chars)")
        if name.lower() in seen_names:
            die(f"{where}: duplicate name {name!r}")
        seen_names.add(name.lower())
        a4_file = f"{name}.a4.bin"
        if a4_file.lower() in seen_files:
            die(f"{where}: duplicate a4 file {a4_file}")
        seen_files.add(a4_file.lower())
        if not isinstance(e["h_sha256"], str) or not HEX64.match(e["h_sha256"]):
            die(f"{where}: `h_sha256` must be 64 lowercase hex chars")
        rows = int_field(e, "rows", where)
        if rows < 1:
            die(f"{where}: rows must be >= 1")
        if int_field(e, "cols", where) != K:
            die(f"{where}: cols must be {K}, got {e['cols']}")
        if int_field(e, "s1_seed", where) != S1_SEED or int_field(e, "s2_seed", where) != S2_SEED:
            die(f"{where}: seeds must be {S1_SEED}/{S2_SEED}, got {e['s1_seed']}/{e['s2_seed']}")
        h_paths.append(resolve_inside(root_real, e["h_file"], where))
    h_set = {os.path.normcase(p) for p in h_paths}
    for e in entries:
        if os.path.normcase(os.path.realpath(os.path.join(root_real, e["name"] + ".a4.bin"))) in h_set:
            die(f"a4 file for {e['name']!r} collides with an H file")

    # Stale-manifest guard: the manifest is written last, so a failed run must not leave an old one behind.
    out_path = os.path.join(chain_dir, OUT_NAME)
    if os.path.lexists(out_path):
        os.unlink(out_path)

    # 3. Verify, rotate, quantize, write A4 files.
    out_entries, total_rows = [], 0
    for e, h_path in zip(entries, h_paths):
        where = f"entry {e['name']!r}"
        rows = e["rows"]
        want = rows * K * 2
        if not os.path.isfile(h_path):
            die(f"{where}: H file {e['h_file']} missing")
        if os.path.getsize(h_path) != want:
            die(f"{where}: H file is {os.path.getsize(h_path)} bytes, expected {want} (rows={rows} x {K} x 2)")
        h_bytes, _ = read_verified(h_path, e["h_sha256"], f"{where} H")
        if len(h_bytes) != want:
            die(f"{where}: H file changed size while reading")
        h = np.frombuffer(h_bytes, "<u2").reshape(rows, K).astype(np.uint16)  # LE bf16 bits; padding rows stay zero
        if ((h & np.uint16(0x7F80)) == np.uint16(0x7F80)).any():
            die(f"{where}: H contains non-finite BF16 (inf/nan)")
        a4 = a4_for_rows(oracle, pilot, h, s1, s2)
        a4_bytes = a4.tobytes()
        if len(a4_bytes) != G * rows * REC:
            die(f"{where}: A4 is {len(a4_bytes)} bytes, expected {G * rows * REC}")
        a4_name = f"{e['name']}.a4.bin"
        a4_path = os.path.join(chain_dir, a4_name)
        atomic_write(a4_path, a4_bytes)
        back, back_sha = read_verified(a4_path, sha256_bytes(a4_bytes), f"{where} A4 readback")
        if len(back) != len(a4_bytes):
            die(f"{where}: A4 readback size mismatch")
        out_entries.append({**e, "a4_file": a4_name, "a4_sha256": back_sha, "a4_bytes": len(a4_bytes), "a4_layout": "epoch-major"})
        total_rows += rows

    # 4. Manifest last.
    out = dict(root)
    out.update({
        "schema": OUT_SCHEMA,
        "input_schema": IN_SCHEMA,
        "entries": out_entries,
        "pilot_solve": {"path": pilot_path, "sha256": pilot_sha},
        "harness": {"path": self_path, "sha256": self_sha},
        "rotation_source": {"path": rot_path, "sha256": rot_sha},
        "input_manifest_sha256": in_sha,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "candidates": CANDIDATES,
        "gpu_disabled": {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": ""},
        "rotation_selftest": selftest,
    })
    atomic_write(out_path, (json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"schema": OUT_SCHEMA, "manifest": out_path, "entries": len(out_entries), "rows": total_rows,
                      "pilot_solve_sha256": pilot_sha, "rotation_source_sha256": rot_sha, "harness_sha256": self_sha}))


if __name__ == "__main__":
    main()
