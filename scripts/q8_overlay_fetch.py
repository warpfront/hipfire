#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""LAB (measurement branch, not for landing): ranged fetch of the trunk QSA BF16 tensors that are
missing from a partial local copy of Qwen/Qwen3.8-Flash-Next.

For every QSA overlay tensor (q8_overlay_writer.QSA_RE) whose shard is absent from --src, GET exactly
that tensor's byte range from the pinned HF revision (Range request), verify the byte count against
the locally cached safetensors header of that shard (--headers DIR holds <shard> files = the first
8 + header_len bytes), and write a one-tensor safetensors file <dst>/extra-NNN.safetensors. The
<dst> dir also gets symlinks to every shard present in --src and a merged
model.safetensors.index.json, so `q8_overlay_writer.py encode --src <dst>` reads the whole set.
<dst>/q8_overlay_fetch_provenance.json records, per tensor: shard, absolute byte range, size and
sha256 of the fetched bytes. Refuses to fetch more than --max-bytes (default 0.2 GB) in total.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q8_overlay_writer as w  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def shard_header(headers_dir, shard):
    raw = (Path(headers_dir) / shard).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + n]), 8 + n


def plan(src, headers_dir):
    index = json.loads((Path(src) / "model.safetensors.index.json").read_text())["weight_map"]
    todo = []
    for name in w.qsa_names(index):
        shard = index[name]
        if (Path(src) / shard).exists():
            continue
        header, base = shard_header(headers_dir, shard)
        entry = header[name]
        lo, hi = entry["data_offsets"]
        todo.append(
            dict(name=name, shard=shard, dtype=entry["dtype"], shape=entry["shape"], lo=base + lo, hi=base + hi, header=entry)
        )
    return index, todo


def fetch_range(shard, lo, hi, revision=REVISION):
    url = f"https://huggingface.co/{REPO}/resolve/{revision}/{shard}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        content_range = resp.headers.get("Content-Range", "")
        status = resp.status
    if status != 206 or len(data) != hi - lo or not content_range.startswith(f"bytes {lo}-{hi - 1}/"):
        raise RuntimeError(f"{shard}: status {status} Content-Range {content_range!r} got {len(data)} bytes, wanted {hi - lo}")
    return data


def write_single(path, name, entry, data):
    header = {name: {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [0, len(data)]}}
    hdr = json.dumps(header).encode()
    path.write_bytes(len(hdr).to_bytes(8, "little") + hdr + data)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--headers", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-bytes", type=int, default=200_000_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    index, todo = plan(args.src, args.headers)
    total = sum(t["hi"] - t["lo"] for t in todo)
    print(f"{len(todo)} tensors missing, {total} bytes to fetch from {REPO}@{REVISION[:12]}")
    for t in todo:
        print(f"  {t['name']} {t['shard']} bytes {t['lo']}-{t['hi'] - 1} ({t['hi'] - t['lo']})")
    if total > args.max_bytes:
        raise SystemExit(f"refusing: {total} bytes exceeds --max-bytes {args.max_bytes}")
    if args.dry_run:
        return 0
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    new_index = dict(index)
    records = []
    for i, t in enumerate(todo):
        data = fetch_range(t["shard"], t["lo"], t["hi"])
        extra = f"extra-{i:03d}.safetensors"
        write_single(dst / extra, t["name"], t["header"], data)
        new_index[t["name"]] = extra
        records.append(
            dict(name=t["name"], shard=t["shard"], byte_range=[t["lo"], t["hi"]], size=len(data),
                 sha256=hashlib.sha256(data).hexdigest(), header_offsets_verified=True, extra_file=extra)
        )
        print(f"fetched {t['name']} {len(data)} {records[-1]['sha256'][:12]}", flush=True)
    for shard in sorted(set(index.values())):
        link = dst / shard
        if (Path(args.src) / shard).exists() and not link.exists():
            os.symlink(Path(args.src) / shard, link)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": new_index}))
    (dst / "q8_overlay_fetch_provenance.json").write_text(
        json.dumps(dict(repo=REPO, revision=REVISION, total_bytes=sum(r["size"] for r in records), tensors=records), indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
