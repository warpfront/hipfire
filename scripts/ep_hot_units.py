#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Write and validate the `<model>.ep.hot-units` sidecar of the Flash-Next EP2 hot-unit loan.

The loader (`crates/hipfire-arch-qwen4/src/expert_hot_units.rs`) keeps, on top of the ordinary
static placement, the hottest calibrated host-demand units each rank owns resident, up to a loan of
bytes per rank (`HIPFIRE_EP_HOT_UNITS`, `HIPFIRE_EP_HOT_LOAN_MIB`). This tool builds the calibration
from `HIPFIRE_EP_SKEW_LOG` directories.

Skew log (`crates/hipfire-arch-qwen4/src/expert_unit_tail_rank.rs`, `EP_SKEW_LOG_ENV`): per rank a
file `skew-rank<r>.log`, one text line per trunk MoE call:

    rank retirement call host_ns layer slots tok...        tok = <expert><class>

class F (another rank's expert), C (static VRAM core), L (tail unit from a live slot), T (tail unit
read from pinned host), H (host-only unit). Lines are grouped into trunk forwards: a new forward
starts when `layer <= previous layer`; only COMPLETE forwards (exactly `layers` lines) count. In a
forward every distinct unit `layer * experts_per_layer + expert` of class T or H that the rank owns
(`expert % ranks == rank`) adds one to `counts[unit]`: the number of calibration forwards in which
the owner read it from pinned host. `capture_forwards[r]` is the number of forwards counted on rank
r, summed over the directories.

Usage
  ep_hot_units.py write --ranking M.ep.expert-hotset --skew-dir DIR [--skew-dir DIR ...] \\
                        --corpus STR --out F [--unit-bytes 2534400] [--ranks 2]
  ep_hot_units.py validate F --ranking M.ep.expert-hotset

The calibration corpus must be held-out mixed/diverse prompts, never the bench prompts. The
ranking is the EP hotset sidecar the calibration run was loaded under: its sha256 is bound into the
file (`base_ranking_sha256`), and so are its model/base binding and geometry. The output is
deterministic: one JSON line, fixed key order (the Rust schema is closed).
"""
import argparse
import hashlib
import json
import os
import re
import sys

KIND = "hipfire.qwen4.ep_hot_units"
VERSION = 1
RANKING_KIND = "hipfire.qwen4.expert_hotset"
DEFAULT_UNIT_BYTES = 2534400
DEFAULT_RANKS = 2
LOAN_MIB = (512, 768, 1024, 1536)
SCHEMA_KEYS = (
    "kind", "version", "model_sha256", "base_size", "base_index_sha256",
    "base_ranking_sha256", "unit_bytes", "layers", "experts_per_layer", "ranks",
    "corpus", "capture_forwards", "counts",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^(\d+)([A-Za-z])$")
CLASSES = "FCLTH"
HOST_CLASSES = "TH"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_ranking(path):
    """Return (doc, sha256 of the file bytes) of an `.ep.expert-hotset` v1 sidecar."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        doc = json.loads(data)
    except ValueError as exc:
        raise ValueError(f"{path}: not JSON: {exc}")
    if not isinstance(doc, dict) or doc.get("kind") != RANKING_KIND:
        raise ValueError(f"{path}: not a {RANKING_KIND} sidecar")
    for key in ("model_sha256", "base_size", "base_index_sha256", "layers", "experts_per_layer"):
        if key not in doc:
            raise ValueError(f"{path}: ranking sidecar lacks {key}")
    return doc, sha256_bytes(data)


def parse_skew_file(path, rank, layers, experts, ranks):
    """Per-unit counts (list of layers*experts) and the counted forward total of one rank's log."""
    counts = [0] * (layers * experts)
    forwards = 0
    incomplete = 0
    pending = []  # (layer, set of host-read owned units) of the open forward
    prev = None

    def close():
        nonlocal forwards, incomplete, pending
        if pending:
            if len(pending) == layers:
                forwards += 1
                for _, units in pending:
                    for unit in units:
                        counts[unit] += 1
            else:
                incomplete += 1
        pending = []

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    # A log cut mid-write ends without a newline: that last fragment is dropped, never parsed.
    tail = lines.pop()
    if tail.strip():
        incomplete += 1
    for number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        where = f"{path}:{number}"
        if len(fields) < 6:
            raise ValueError(f"{where}: expected `rank retirement call host_ns layer slots tok...`")
        try:
            line_rank, layer, slots = int(fields[0]), int(fields[4]), int(fields[5])
        except ValueError:
            raise ValueError(f"{where}: non-integer rank/layer/slots field")
        tokens = fields[6:]
        if line_rank != rank:
            raise ValueError(f"{where}: rank {line_rank} in the log of rank {rank}")
        if not 0 <= layer < layers:
            raise ValueError(f"{where}: layer {layer} out of range 0..{layers}")
        if len(tokens) != slots:
            raise ValueError(f"{where}: {slots} slots but {len(tokens)} tokens")
        if prev is not None and layer <= prev:
            close()
        prev = layer
        units = set()
        for token in tokens:
            match = TOKEN.match(token)
            if not match or match.group(2) not in CLASSES:
                raise ValueError(f"{where}: bad token {token!r} (<expert><class>, class in {CLASSES})")
            expert = int(match.group(1))
            if expert >= experts:
                raise ValueError(f"{where}: expert {expert} out of range 0..{experts}")
            if match.group(2) in HOST_CLASSES and expert % ranks == rank:
                units.add(layer * experts + expert)
        pending.append((layer, units))
    close()
    return counts, forwards, incomplete


def calibrate(skew_dirs, layers, experts, ranks):
    """(counts, capture_forwards, dropped_incomplete_per_rank) pooled over `skew_dirs`."""
    counts = [0] * (layers * experts)
    forwards = [0] * ranks
    incomplete = [0] * ranks
    for directory in skew_dirs:
        for rank in range(ranks):
            path = os.path.join(directory, f"skew-rank{rank}.log")
            if not os.path.isfile(path):
                raise ValueError(f"{path}: missing (every --skew-dir needs skew-rank0..{ranks - 1}.log)")
            part, n, bad = parse_skew_file(path, rank, layers, experts, ranks)
            counts = [a + b for a, b in zip(counts, part)]
            forwards[rank] += n
            incomplete[rank] += bad
    return counts, forwards, incomplete


def build_doc(ranking, ranking_sha, counts, forwards, ranks, unit_bytes, corpus):
    values = dict(
        kind=KIND, version=VERSION,
        model_sha256=ranking["model_sha256"], base_size=ranking["base_size"],
        base_index_sha256=ranking["base_index_sha256"], base_ranking_sha256=ranking_sha,
        unit_bytes=unit_bytes, layers=ranking["layers"],
        experts_per_layer=ranking["experts_per_layer"], ranks=ranks, corpus=corpus,
        capture_forwards=list(forwards), counts=list(counts),
    )
    return {key: values[key] for key in SCHEMA_KEYS}


def is_uint(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_doc(doc, ranking=None, ranking_sha=None):
    """Schema (closed, like the Rust parser) and, with `ranking`, the binding. Raises ValueError."""
    if not isinstance(doc, dict):
        raise ValueError("not a JSON object")
    keys = tuple(doc)
    if set(keys) != set(SCHEMA_KEYS):
        extra, missing = sorted(set(keys) - set(SCHEMA_KEYS)), sorted(set(SCHEMA_KEYS) - set(keys))
        raise ValueError(f"keys differ from the closed v1 schema: extra {extra}, missing {missing}")
    if doc["kind"] != KIND or doc["version"] != VERSION:
        raise ValueError(f"kind {doc['kind']!r} version {doc['version']!r}, expected {KIND!r} {VERSION}")
    for key in ("model_sha256", "base_index_sha256", "base_ranking_sha256"):
        if not isinstance(doc[key], str) or not HEX64.match(doc[key]):
            raise ValueError(f"{key} is not a full lowercase 64-hex SHA-256")
    for key in ("base_size", "unit_bytes", "layers", "experts_per_layer", "ranks"):
        if not is_uint(doc[key]):
            raise ValueError(f"{key} is not an unsigned integer")
    if doc["unit_bytes"] == 0 or doc["layers"] == 0 or doc["ranks"] == 0:
        raise ValueError("unit_bytes, layers and ranks must be positive")
    if doc["experts_per_layer"] < doc["ranks"]:
        raise ValueError("experts_per_layer is below ranks")
    if not isinstance(doc["corpus"], str) or not doc["corpus"].strip():
        raise ValueError("corpus must be a non-empty provenance string")
    forwards, counts = doc["capture_forwards"], doc["counts"]
    if not isinstance(forwards, list) or not all(is_uint(n) for n in forwards):
        raise ValueError("capture_forwards is not a list of unsigned integers")
    if len(forwards) != doc["ranks"]:
        raise ValueError(f"capture_forwards has {len(forwards)} entries for {doc['ranks']} ranks")
    total = doc["layers"] * doc["experts_per_layer"]
    if not isinstance(counts, list) or not all(is_uint(n) for n in counts):
        raise ValueError("counts is not a list of unsigned integers")
    if len(counts) != total:
        raise ValueError(f"counts has {len(counts)} entries, layers x experts_per_layer = {total}")
    for unit, count in enumerate(counts):
        owner = (unit % doc["experts_per_layer"]) % doc["ranks"]
        if count > forwards[owner]:
            raise ValueError(
                f"counts[{unit}] = {count} exceeds the {forwards[owner]} forwards of its owner rank {owner}")
    if ranking is not None:
        for key in ("model_sha256", "base_size", "base_index_sha256", "layers", "experts_per_layer"):
            if doc[key] != ranking[key]:
                raise ValueError(f"{key} {doc[key]!r} differs from the ranking's {ranking[key]!r}")
        if doc["base_ranking_sha256"] != ranking_sha:
            raise ValueError(
                f"base_ranking_sha256 {doc['base_ranking_sha256'][:12]} is not the ranking's "
                f"sha256 {ranking_sha[:12]}")


def render(doc):
    return json.dumps(doc, separators=(",", ":")) + "\n"


def coverage_lines(doc):
    """Per rank: forwards, nonzero units, and the top-of-ranking coverage at each loan size."""
    g_e, ranks, unit = doc["experts_per_layer"], doc["ranks"], doc["unit_bytes"]
    lines = []
    for rank in range(ranks):
        owned = [(count, unit_id) for unit_id, count in enumerate(doc["counts"])
                 if count > 0 and (unit_id % g_e) % ranks == rank]
        owned.sort(key=lambda item: (-item[0], item[1]))
        mass = sum(count for count, _ in owned)
        lines.append(f"rank {rank}: {doc['capture_forwards'][rank]} forwards, {len(owned)} nonzero units "
                     f"({mass} unit-reads)")
        for mib in LOAN_MIB:
            fit = min(len(owned), (mib << 20) // unit)
            covered = sum(count for count, _ in owned[:fit])
            share = 100.0 * covered / mass if mass else 0.0
            lines.append(f"  loan {mib:4d} MiB: top {fit} units = {fit * unit / (1 << 20):.1f} MiB "
                         f"cover {share:.1f}% of the rank's host reads (arena pads not counted)")
    return lines


def cmd_write(args):
    ranking, ranking_sha = read_ranking(args.ranking)
    layers, experts = ranking["layers"], ranking["experts_per_layer"]
    if not args.skew_dir:
        raise SystemExit("write needs at least one --skew-dir")
    counts, forwards, incomplete = calibrate(args.skew_dir, layers, experts, args.ranks)
    doc = build_doc(ranking, ranking_sha, counts, forwards, args.ranks, args.unit_bytes, args.corpus)
    validate_doc(doc, ranking, ranking_sha)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(doc))
    for line in coverage_lines(doc):
        print(line)
    if any(incomplete):
        print(f"dropped incomplete forwards per rank: {incomplete}")
    print(f"wrote {args.out}")
    return 0


def cmd_validate(args):
    ranking, ranking_sha = read_ranking(args.ranking)
    with open(args.file, "r", encoding="utf-8") as fh:
        doc = json.loads(fh.read(), object_pairs_hook=dict)
    validate_doc(doc, ranking, ranking_sha)
    for line in coverage_lines(doc):
        print(line)
    print(f"{args.file}: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="calibrate from skew-log directories")
    w.add_argument("--ranking", required=True, help="<model>.ep.expert-hotset the run was loaded under")
    w.add_argument("--skew-dir", action="append", default=[], help="HIPFIRE_EP_SKEW_LOG directory (repeatable)")
    w.add_argument("--corpus", required=True, help="provenance string of the calibration prompts")
    w.add_argument("--out", required=True)
    w.add_argument("--unit-bytes", type=int, default=DEFAULT_UNIT_BYTES)
    w.add_argument("--ranks", type=int, default=DEFAULT_RANKS)
    w.set_defaults(func=cmd_write)
    v = sub.add_parser("validate", help="schema + binding check")
    v.add_argument("file")
    v.add_argument("--ranking", required=True)
    v.set_defaults(func=cmd_validate)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ep_hot_units: {exc}")


if __name__ == "__main__":
    sys.exit(main())
