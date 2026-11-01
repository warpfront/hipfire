#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Write an expert-hotset v1 ranking (+ provenance sidecar) from K0 route dumps.

A K0 route dump (`HFK0RT01`, written by the lab EP oracle with
HIPFIRE_EP_LAB_ROUTE_DUMP) records, per MoE layer call, the routed expert ids
of every row slot. This tool pools the slot-read counts of the selected dumps
and writes the `hipfire.qwen4.expert_hotset` v1 file the Rust parser
(`crates/hipfire-arch-qwen4/src/expert_hotset.rs`) reads:

  rank   = every global unit id `layer * experts_per_layer + expert` exactly
           once, ordered by count descending, ties by id ascending
  counts = pooled slot reads per global id

The v1 schema is closed (unknown keys are refused), so provenance goes into
`<out>.provenance.json`, never into the ranking file.

Dump format
  header  "HFK0RT01" | u32 layers | u32 ranks | u32 top_k       (20 bytes)
  records tag 1 CALL     : u16 layer, u16 nslots, nslots * i32 expert id (-1 = pad)
          tag 2 STEP_END : u8 phase (0 decode, 1 verify), u32 step index
          tag 3 DISCARD  : (no payload) drops the open calls
Counting rule (identical to the K0 simulator, scripts/k0sim.py): calls of
closed steps count; the 1-row reference pass never emits STEP_END, so at the
first DISCARD (or at EOF when there was none) the open calls with
nslots == top_k are recovered; every other open call is dropped.
Step class: ar1 = phase 0 and 1 row, ar8 = phase 0 and > 1 rows, verify =
phase 1, other = rest (rows = sum over the step's calls on its first layer of
nslots / top_k). `--drop-class` removes whole steps of a class before counting.

Usage
  k0rank.py write    --manifest M.tsv --split cal --bind-from EXISTING.expert-hotset --out F
  k0rank.py validate F [--manifest M.tsv --split held]       (+ coverage curve)

MANIFEST columns: dump<TAB>split<TAB>domain<TAB>mode   (split cal|held).
"""
import argparse
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys
from array import array

MAGIC = b"HFK0RT01"
KIND = "hipfire.qwen4.expert_hotset"
VERSION = 1
CLASSES = ("ar1", "ar8", "verify", "other")
TOOL = "k0rank"
SCHEMA_KEYS = (
    "version", "kind", "model_sha256", "base_size", "base_index_sha256",
    "counts_sha256", "format_map", "format_map_sha256", "layers",
    "experts_per_layer", "top_k", "capture_tokens", "ranking_rule", "corpus",
    "rank", "counts",
)
BIND_KEYS = ("model_sha256", "base_size", "base_index_sha256", "format_map",
             "format_map_sha256", "layers", "experts_per_layer", "top_k")
RANKING_RULE = (
    "descending pooled slot-read count over the K0 routing dumps (all step classes unless "
    "dropped), ties by global id ascending; global id = layer*experts_per_layer+expert; "
    "counts_sha256 = sha256 of the counts as little-endian u64 in global id order"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def counts_sha256(counts):
    return hashlib.sha256(b"".join(struct.pack("<Q", c) for c in counts)).hexdigest()


def parse_dump(path, experts, drop=()):
    """Return (header, per-unit slot counts list, info dict) for one dump."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 20 or buf[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic/short header")
    layers, ranks, topk = struct.unpack_from("<III", buf, 8)
    if topk == 0:
        raise ValueError(f"{path}: top_k is 0")
    steps = []          # (phase, forced_class or None, [(layer, ids)])
    open_calls = []
    discards = 0
    reference = dict(recovered_calls=0, dropped_calls=0, steps=0)
    pos, n, truncated = 20, len(buf), False

    def recover(calls):
        """The pending 1-row reference calls -> ar1 steps (full layer run) or `other`."""
        cur = []

        def emit(group, cls):
            steps.append((0, cls, group))
            reference["steps"] += 1
            reference["recovered_calls"] += len(group)

        for layer, ids in calls:
            if len(ids) != topk:
                if cur:
                    emit(cur, 3)
                    cur = []
                reference["dropped_calls"] += 1
                continue
            if cur and layer != cur[-1][0] + 1:
                emit(cur, 3)
                cur = []
            cur.append((layer, ids))
            if len(cur) == layers:
                emit(cur, 0 if cur[0][0] == 0 else 3)
                cur = []
        if cur:
            emit(cur, 3)

    recovered = False
    while pos < n:
        tag = buf[pos]
        if tag == 1:
            if pos + 5 > n:
                truncated = True
                break
            layer, ns = struct.unpack_from("<HH", buf, pos + 1)
            end = pos + 5 + 4 * ns
            if end > n:
                truncated = True
                break
            ids = array("i")
            ids.frombytes(buf[pos + 5:end])
            open_calls.append((layer, ids))
            pos = end
        elif tag == 2:
            if pos + 6 > n:
                truncated = True
                break
            phase = buf[pos + 1]
            steps.append((phase, None, open_calls))
            open_calls = []
            pos += 6
        elif tag == 3:
            pos += 1
            if not recovered and discards == 0:
                recover(open_calls)
                recovered = True
            discards += 1
            open_calls = []
        else:
            raise ValueError(f"{path}: unknown tag {tag} at byte {pos}")
    if not recovered and discards == 0 and open_calls:
        recover(open_calls)
    dropped_open = 0 if recovered or discards == 0 else len(open_calls)

    counts = [0] * (layers * experts)
    class_steps = [0, 0, 0, 0]
    class_slots = [0, 0, 0, 0]
    dropped = {c: 0 for c in drop}
    for phase, forced, calls in steps:
        if forced is not None:
            cls = forced
        else:
            first = calls[0][0] if calls else None
            rows = sum(round(len(ids) / topk) for layer, ids in calls if layer == first)
            cls = 2 if phase == 1 else (0 if rows == 1 else 1) if phase == 0 else 3
        class_steps[cls] += 1
        if CLASSES[cls] in drop:
            dropped[CLASSES[cls]] += 1
            continue
        for layer, ids in calls:
            if layer >= layers:
                raise ValueError(f"{path}: layer {layer} >= {layers}")
            base = layer * experts
            for e in ids:
                if e < 0:
                    continue
                if e >= experts:
                    raise ValueError(f"{path}: expert id {e} >= experts_per_layer {experts}")
                counts[base + e] += 1
                class_slots[cls] += 1
    info = dict(layers=layers, ranks=ranks, top_k=topk, steps=len(steps), discards=discards,
                truncated=truncated, dropped_open_calls=dropped_open,
                recovered_reference=reference if recovered or reference["steps"] else None,
                dropped_classes=dropped, step_classes=dict(zip(CLASSES, class_steps)),
                slot_reads_by_class=dict(zip(CLASSES, class_slots)),
                slot_reads=sum(class_slots))
    return (layers, topk), counts, info


def read_manifest(path):
    rows = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) != 4 or cols[1] not in ("cal", "held") or cols[3] not in ("ar", "mtp"):
                raise ValueError(f"{path}:{ln}: expected dump<TAB>cal|held<TAB>domain<TAB>ar|mtp")
            rows.append(tuple(cols))
    return rows


def run_meta(dump):
    """Token counts and prompt digests from the oracle manifest next to the dump, if present."""
    mf = os.path.join(os.path.dirname(dump), "run", "manifest.json")
    if not os.path.exists(mf):
        return None
    with open(mf) as fh:
        m = json.load(fh)
    prompts = m.get("prompts") or []
    distinct = int(m.get("distinct_prompts") or len(prompts))
    tokens = int(m.get("tokens") or 0)
    return dict(oracle_manifest=mf, tokens_per_prompt=tokens, prompts=distinct,
                tokens=tokens * distinct, spec=m.get("spec"),
                prompt_sha256=[p.get("sha256") for p in prompts])


def pool(manifest, split, experts, drop):
    rows = [r for r in read_manifest(manifest) if r[1] == split]
    if not rows:
        raise SystemExit(f"{manifest}: no '{split}' dumps")
    total, header, dumps = None, None, []
    for path, _, domain, mode in rows:
        hdr, counts, info = parse_dump(path, experts, drop)
        if header is None:
            header = hdr
            total = [0] * len(counts)
        elif hdr != header:
            raise SystemExit(f"{path}: header {hdr} differs from {header}")
        total = [a + b for a, b in zip(total, counts)]
        meta = run_meta(path)
        dumps.append(dict(file=path, sha256=sha256_file(path), split=split, domain=domain,
                          mode=mode, run=meta, **info))
    return header, total, dumps


def order(counts):
    return sorted(range(len(counts)), key=lambda i: (-counts[i], i))


def tool_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = None
    return dict(script=os.path.basename(__file__), script_sha256=sha256_file(os.path.abspath(__file__)),
                repo_head=rev)


def load_bind(args):
    bind = {}
    if args.bind_from:
        with open(args.bind_from) as fh:
            src = json.load(fh)
        bind = {k: src[k] for k in BIND_KEYS}
    for k in BIND_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            bind[k] = v
    missing = [k for k in BIND_KEYS if k not in bind]
    if missing:
        raise SystemExit(f"base binding incomplete, give --bind-from or: {', '.join(missing)}")
    return bind


def cmd_write(args):
    bind = load_bind(args)
    experts = int(bind["experts_per_layer"])
    (layers, topk), counts, dumps = pool(args.manifest, args.split, experts, tuple(args.drop_class))
    if layers != bind["layers"] or topk != bind["top_k"]:
        raise SystemExit(f"dump layers/top_k {layers}/{topk} differ from the binding "
                         f"{bind['layers']}/{bind['top_k']}")
    tokens = sum((d["run"] or {}).get("tokens", 0) for d in dumps)
    if tokens == 0:
        tokens = args.capture_tokens
    if not tokens:
        raise SystemExit("no token count found next to the dumps; pass --capture-tokens")
    rank = order(counts)
    domains = sorted({d["domain"] for d in dumps})
    doc = dict(
        version=VERSION, kind=KIND, model_sha256=bind["model_sha256"],
        base_size=bind["base_size"], base_index_sha256=bind["base_index_sha256"],
        counts_sha256=counts_sha256(counts), format_map=bind["format_map"],
        format_map_sha256=bind["format_map_sha256"], layers=layers,
        experts_per_layer=experts, top_k=topk, capture_tokens=tokens,
        ranking_rule=RANKING_RULE + (f"; dropped step classes: {','.join(args.drop_class)}"
                                     if args.drop_class else ""),
        corpus=args.corpus or (f"K0 {args.split} route dumps: {len(dumps)} dumps, domains "
                               f"{'/'.join(domains)}, pooled"),
        rank=rank, counts=counts)
    assert tuple(doc) == SCHEMA_KEYS
    body = json.dumps(doc, separators=(",", ":")) + "\n"
    with open(args.out, "w") as fh:
        fh.write(body)
    out_sha = hashlib.sha256(body.encode()).hexdigest()
    mix = {}
    for d in dumps:
        for c, v in d["slot_reads_by_class"].items():
            mix[c] = mix.get(c, 0) + v
    prov = dict(
        tool=TOOL, tool_version=tool_rev(), date=datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        ranking_file=os.path.basename(args.out), ranking_sha256=out_sha,
        counts_sha256=doc["counts_sha256"], units=len(rank),
        base_binding=dict(bind), split=args.split, domains=domains,
        drop_class=list(args.drop_class), class_mix_slot_reads=mix,
        cal_slot_reads=sum(counts), capture_tokens=tokens,
        capture_tokens_rule="sum over dumps of oracle manifest tokens x distinct prompts "
                            "(generated tokens per prompt; one dump = one oracle run)",
        ownership="static parity e % ranks (EP2: rank = expert % 2)",
        dumps=dumps)
    with open(args.out + ".provenance.json", "w") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    print(f"wrote {args.out}: {len(rank)} units, sha256 {out_sha}, {sum(counts)} slot reads, "
          f"{tokens} tokens")


def validate_doc(doc):
    """The same checks as HotsetSidecar::from_bytes (structure, permutation, ordering)."""
    if tuple(doc) != SCHEMA_KEYS:
        raise ValueError(f"keys {tuple(doc)} differ from the v1 schema")
    if doc["kind"] != KIND or doc["version"] != VERSION:
        raise ValueError("kind/version")
    total = doc["layers"] * doc["experts_per_layer"]
    rank, counts = doc["rank"], doc["counts"]
    if len(counts) != total or len(rank) != total:
        raise ValueError(f"rank/counts length {len(rank)}/{len(counts)} != {total}")
    if sorted(rank) != list(range(total)):
        raise ValueError("rank is not a permutation of the global ids")
    for a, b in zip(rank, rank[1:]):
        if counts[a] < counts[b] or (counts[a] == counts[b] and a > b):
            raise ValueError(f"rank not (count desc, id asc) at ids {a},{b}")
    if counts_sha256(counts) != doc["counts_sha256"]:
        raise ValueError("counts_sha256 does not match counts")


def coverage(rank, held_counts, fractions):
    tot = sum(held_counts)
    out = {}
    for f in fractions:
        b = int(f * len(rank) + 0.5)
        out[f] = sum(held_counts[i] for i in rank[:b]) / tot if tot else 0.0
    return out


def cmd_validate(args):
    with open(args.file) as fh:
        doc = json.load(fh)
    validate_doc(doc)
    print(f"ok: {len(doc['rank'])} units, permutation, (count desc, id asc), counts_sha256 matches")
    if args.manifest:
        _, held, dumps = pool(args.manifest, args.split, doc["experts_per_layer"], tuple(args.drop_class))
        print(f"coverage of {args.split} slot reads ({sum(held)} over {len(dumps)} dumps):")
        for f, v in coverage(doc["rank"], held, (0.25, 0.5, 0.6875, 0.75)).items():
            print(f"  top {f * 100:5.2f}% of units -> {v * 100:.2f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--manifest", required=True)
    w.add_argument("--split", default="cal", choices=("cal", "held"))
    w.add_argument("--out", required=True)
    w.add_argument("--bind-from", help="existing .expert-hotset of the same base to copy the binding from")
    w.add_argument("--model-sha256")
    w.add_argument("--base-size", type=int)
    w.add_argument("--base-index-sha256")
    w.add_argument("--format-map")
    w.add_argument("--format-map-sha256")
    w.add_argument("--layers", type=int)
    w.add_argument("--experts-per-layer", type=int)
    w.add_argument("--top-k", type=int)
    w.add_argument("--drop-class", action="append", default=[], choices=CLASSES)
    w.add_argument("--capture-tokens", type=int, default=0)
    w.add_argument("--corpus")
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.add_argument("--manifest")
    v.add_argument("--split", default="held", choices=("cal", "held"))
    v.add_argument("--drop-class", action="append", default=[], choices=CLASSES)
    args = ap.parse_args(argv)
    {"write": cmd_write, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
