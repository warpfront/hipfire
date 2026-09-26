#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned Qwen4 safetensors headers and build sparse local shards.

This is an opt-in storage admission harness.  It reads only the two header
ranges from each of the pinned 131 checkpoint shards (plus config/index JSON),
then writes those exact bytes into sparse files whose logical lengths match the
remote objects.  The resulting directory can be handed to the Rust converter's
header-only production inventory test.  No model payload byte is requested or
written.

Usage:
  python3 build_sparse_inventory.py --output .codeinsight+research/qwen4/pinned-sparse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SHARD_COUNT = 131
MAX_HEADER_BYTES = 8 << 20
RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
VISION_PREFIXES = (
    "model.visual.",
    "model.vision_tower.",
    "model.vision_projection.",
    "model.multi_modal_projector.",
    "vision_tower.",
    "visual.",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def request_range(url: str, start: int, end: int) -> tuple[bytes, dict[str, Any]]:
    if start < 0 or end < start or end - start + 1 > MAX_HEADER_BYTES:
        raise RuntimeError(f"invalid bounded range {start}-{end} for {url}")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "hipfire-qwen4-sparse-inventory/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RuntimeError(f"{url} bytes {start}-{end}: HTTP {status}, expected 206")
            content_range = response.headers.get("Content-Range", "")
            match = RANGE_RE.fullmatch(content_range)
            if match is None:
                raise RuntimeError(f"{url}: invalid Content-Range {content_range!r}")
            got_start, got_end, total = map(int, match.groups())
            if got_start != start or got_end != end or got_end >= total:
                raise RuntimeError(
                    f"{url}: Content-Range {content_range!r} does not equal bytes {start}-{end}"
                )
            body = response.read(end - start + 2)
            if len(body) != end - start + 1:
                raise RuntimeError(
                    f"{url}: range body has {len(body)} bytes, expected {end - start + 1}"
                )
            metadata = {
                "etag": response.headers.get("ETag"),
                "repo_commit": response.headers.get("X-Repo-Commit"),
                "total": total,
                "content_range": content_range,
            }
            return body, metadata
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise RuntimeError(f"unable to read {url} bytes {start}-{end}: {exc}") from exc


def read_object(url: str, *, maximum: int = MAX_HEADER_BYTES) -> tuple[bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "hipfire-qwen4-sparse-inventory/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", response.getcode())
            if status != 200:
                raise RuntimeError(f"{url}: HTTP {status}, expected 200")
            body = response.read(maximum + 1)
            if len(body) > maximum:
                raise RuntimeError(f"{url}: body exceeds {maximum} bytes")
            return body, {
                "etag": response.headers.get("ETag"),
                "repo_commit": response.headers.get("X-Repo-Commit"),
                "total": len(body),
            }
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise RuntimeError(f"unable to read {url}: {exc}") from exc


def header_for(url: str) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    prefix, prefix_meta = request_range(url, 0, 7)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length == 0 or header_length > MAX_HEADER_BYTES:
        raise RuntimeError(f"{url}: header length {header_length} outside 1..={MAX_HEADER_BYTES}")
    header, header_meta = request_range(url, 8, 8 + header_length - 1)
    if prefix_meta["total"] != header_meta["total"]:
        raise RuntimeError(f"{url}: object total changed between header ranges")
    try:
        decoded = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"{url}: invalid safetensors header: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{url}: safetensors header is not an object")
    return header, decoded, {
        "header_length": header_length,
        "header_sha256": sha256_bytes(header),
        "etag": header_meta.get("etag") or prefix_meta.get("etag"),
        "repo_commit": header_meta.get("repo_commit") or prefix_meta.get("repo_commit"),
        "total": header_meta["total"],
    }


def descriptor_bytes(name: str, descriptor: dict[str, Any]) -> tuple[str, list[int], int, int]:
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if dtype not in ("BF16", "I64"):
        raise RuntimeError(f"{name}: source dtype {dtype!r} is not BF16/I64")
    if not isinstance(shape, list) or not all(isinstance(value, int) and value >= 0 for value in shape):
        raise RuntimeError(f"{name}: malformed shape {shape!r}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) and value >= 0 for value in offsets)
        or offsets[1] < offsets[0]
    ):
        raise RuntimeError(f"{name}: malformed data_offsets {offsets!r}")
    return dtype, shape, offsets[0], offsets[1]


def write_sparse_shard(path: Path, header: bytes, total: int) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.flush()
        os.ftruncate(stream.fileno(), total)
        os.fsync(stream.fileno())
    stat = path.stat()
    return stat.st_size, getattr(stat, "st_blocks", 0) * 512


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.refresh:
            raise SystemExit(f"refusing existing output {output}; pass --refresh")
        for child in output.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    output.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"

    index_raw, index_meta = read_object(f"{base}/model.safetensors.index.json")
    config_raw, config_meta = read_object(f"{base}/config.json")
    index = json.loads(index_raw)
    config = json.loads(config_raw)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("model index has no weight_map object")
    expected_names = set(weight_map)

    (output / "config.json").write_bytes(config_raw)
    (output / "model.safetensors.index.json").write_bytes(index_raw)
    shards: list[dict[str, Any]] = []
    names: dict[str, dict[str, Any]] = {}
    counts = {"BF16": 0, "I64": 0}
    requests = 0
    requested_bytes = len(index_raw) + len(config_raw)
    for index_number in range(1, SHARD_COUNT + 1):
        shard_name = f"model-{index_number:05d}-of-{SHARD_COUNT:05d}.safetensors"
        url = f"{base}/{shard_name}"
        header, decoded, meta = header_for(url)
        requests += 2
        requested_bytes += 8 + len(header)
        header_end = 8 + len(header)
        max_relative_end = 0
        shard_names: list[str] = []
        dtype_counts = {"BF16": 0, "I64": 0}
        payload_bytes = 0
        for name, descriptor in decoded.items():
            if name == "__metadata__":
                continue
            if not isinstance(descriptor, dict):
                raise RuntimeError(f"{shard_name}: descriptor for {name} is not an object")
            dtype, shape, start, end = descriptor_bytes(name, descriptor)
            if end > max_relative_end:
                max_relative_end = end
            if name in names:
                raise RuntimeError(f"duplicate source tensor {name} in {shard_name}")
            record = {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
                "byte_length": end - start,
                "shard": shard_name,
            }
            names[name] = record
            shard_names.append(name)
            dtype_counts[dtype] += 1
            counts[dtype] += 1
            payload_bytes += end - start
        if header_end + max_relative_end > int(meta["total"]):
            raise RuntimeError(
                f"{shard_name}: header/payload end {header_end + max_relative_end} exceeds {meta['total']}"
            )
        file_path = output / shard_name
        logical_size, allocated_bytes = write_sparse_shard(file_path, header, int(meta["total"]))
        shards.append(
            {
                "name": shard_name,
                "url": url,
                "etag": meta["etag"],
                "repo_commit": meta["repo_commit"],
                "header_length": len(header),
                "header_sha256": meta["header_sha256"],
                "logical_size": logical_size,
                "allocated_bytes": allocated_bytes,
                "tensor_count": len(shard_names),
                "dtype_counts": dtype_counts,
                "payload_bytes_from_offsets": payload_bytes,
                "max_relative_end": max_relative_end,
                "tensor_names": sorted(shard_names),
            }
        )
        print(
            f"{shard_name}: headers={len(shard_names)} header_bytes={len(header)} "
            f"logical={logical_size} allocated={allocated_bytes}"
        )

    if len(shards) != SHARD_COUNT:
        raise RuntimeError(f"fetched {len(shards)} shards, expected {SHARD_COUNT}")
    if set(names) != expected_names:
        missing = sorted(expected_names - set(names))
        extra = sorted(set(names) - expected_names)
        raise RuntimeError(f"index/header inventory mismatch: missing={missing[:8]} extra={extra[:8]}")
    if len(names) != 1_658 or counts != {"BF16": 1_655, "I64": 3}:
        raise RuntimeError(f"pinned inventory counts are tensors={len(names)} dtypes={counts}")
    vision_count = sum(any(name.startswith(prefix) for prefix in VISION_PREFIXES) for name in names)
    if vision_count != 333:
        raise RuntimeError(f"positive vision inventory count is {vision_count}, expected 333")
    sparse_bytes = sum(shard["allocated_bytes"] for shard in shards)
    logical_bytes = sum(shard["logical_size"] for shard in shards)
    report = {
        "model": MODEL,
        "revision": REVISION,
        "shard_count": len(shards),
        "tensor_count": len(names),
        "dtype_counts": counts,
        "positive_vision_count": vision_count,
        "output_entry_count": len(names) - vision_count,
        "logical_shard_bytes": logical_bytes,
        "allocated_shard_bytes": sparse_bytes,
        "header_and_metadata_bytes_requested": requested_bytes,
        "range_requests": requests,
        "payload_ranges_requested": 0,
        "index_sha256": sha256_bytes(index_raw),
        "config_sha256": sha256_bytes(config_raw),
        "index_etag": index_meta.get("etag"),
        "config_etag": config_meta.get("etag"),
        "shards": shards,
        "tensors": sorted(names.values(), key=lambda record: record["name"]),
    }
    (output / "inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: {len(shards)} sparse shards, {len(names)} tensors, dtypes={counts}, "
        f"vision={vision_count}, range_requests={requests}, payload_ranges=0, "
        f"logical_bytes={logical_bytes}, allocated_bytes={sparse_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
