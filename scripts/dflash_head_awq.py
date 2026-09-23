#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Head-only AWQ repack for DFlash MQ4 drafts.

This calibrates the DFlash `fc.weight` projection against fixed target-hidden
statistics. It does not mutate the target/trunk, and it does not touch any
other drafter tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "hipfire.dflash_head_awq.v0"


def load_astrea():
    spec = importlib.util.spec_from_file_location("hipfire_astrea_for_dflash_head_awq", SCRIPT_DIR / "astrea.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import scripts/astrea.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ASTREA = load_astrea()
np = ASTREA.np


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sumsq(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if data.get("schema") != "hipfire.dflash_fc_sumsq.v0":
        raise ValueError(f"{path} is not a dflash fc sumsq artifact")
    if data.get("tensor") != "fc.weight":
        raise ValueError(f"{path} stats are for {data.get('tensor')!r}, expected 'fc.weight'")
    sumsq = np.asarray(data["sumsq"], dtype=np.float64).reshape(-1)
    rows = int(data.get("rows") or 0)
    row_stride = int(data.get("row_stride") or sumsq.size)
    if row_stride != sumsq.size:
        raise ValueError(f"row_stride {row_stride} does not match sumsq length {sumsq.size}")
    if rows <= 0:
        raise ValueError("fc sumsq artifact contains no rows")
    return {"rows": rows, "row_stride": row_stride, "sumsq": sumsq}


def compute_awq_scales(sumsq, *, rows: int, alpha: float, scale_min: float, scale_max: float):
    values = np.asarray(sumsq, dtype=np.float64).reshape(-1) / max(1, int(rows))
    log_s = (float(alpha) * 0.5) * np.log(np.maximum(values, 1.0e-12))
    log_s -= np.mean(log_s)
    scales = np.exp(log_s).astype(np.float32)
    if scale_min > 0.0 or math.isfinite(scale_max):
        scales = np.clip(scales, np.float32(scale_min), np.float32(scale_max)).astype(np.float32)
        # Preserve the AWQ normalization after clamping as closely as possible.
        log_s = np.log(np.maximum(scales.astype(np.float64), 1.0e-12))
        log_s -= np.mean(log_s)
        scales = np.exp(log_s).astype(np.float32)
        scales = np.clip(scales, np.float32(scale_min), np.float32(scale_max)).astype(np.float32)
    if not bool(np.all(np.isfinite(scales) & (scales > 0.0))):
        raise ValueError("computed non-finite or non-positive AWQ scales")
    return scales


def f16_sidecar_payload(scales) -> bytes:
    return np.asarray(scales, dtype="<f2").reshape(-1).tobytes()


def sidecar_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"expected a .weight tensor name, got {weight_name}")
    return f"{weight_name[:-len('.weight')]}.awq_scale.weight"


def insert_or_replace_sidecar(layout, weight_name: str, scales) -> dict[str, object]:
    records = layout["records"]
    by_name = {record["name"]: i for i, record in enumerate(records)}
    name = sidecar_name(weight_name)
    payload = f16_sidecar_payload(scales)
    record = {
        "name": name,
        "quant_type": 1,
        "quant_type_name": "F16",
        "shape": [len(payload) // 2],
        "group_size": 0,
        "data_size": len(payload),
        "data": payload,
    }
    old = None
    if name in by_name:
        old = {
            "data_size": records[by_name[name]]["data_size"],
            "shape": records[by_name[name]]["shape"],
            "quant_type_name": records[by_name[name]]["quant_type_name"],
        }
        records[by_name[name]].update(record)
        return {"name": name, "action": "replaced", "old": old, "new_data_size": len(payload)}
    insert_at = by_name.get(weight_name, len(records) - 1) + 1
    records.insert(insert_at, record)
    return {"name": name, "action": "inserted", "new_data_size": len(payload)}


def parse_clamp(value: str) -> tuple[float, float]:
    lo, sep, hi = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("--scale-clamp expects MIN:MAX")
    scale_min = float(lo)
    scale_max = float(hi)
    if scale_min <= 0.0 or scale_max <= scale_min:
        raise argparse.ArgumentTypeError("--scale-clamp requires 0 < MIN < MAX")
    return scale_min, scale_max


def write_candidate(args):
    if np is None:
        raise RuntimeError("numpy is required")
    base = Path(args.base).expanduser()
    source_dir = Path(args.source_dir).expanduser()
    output = Path(args.output).expanduser()
    stats_path = Path(args.stats).expanduser()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")

    stats = load_sumsq(stats_path)
    scale_min, scale_max = parse_clamp(args.scale_clamp)
    scales = compute_awq_scales(
        stats["sumsq"],
        rows=int(stats["rows"]),
        alpha=args.alpha,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    layout = ASTREA.read_hfq_layout(base)
    records = layout["records"]
    by_name = {record["name"]: record for record in records}
    if args.tensor not in by_name:
        raise ValueError(f"{base} does not contain {args.tensor}")
    fc_record = by_name[args.tensor]
    if fc_record["quant_type_name"] != "MQ4G256":
        raise ValueError(f"{args.tensor} is {fc_record['quant_type_name']}, expected MQ4G256")
    shape = [int(x) for x in fc_record["shape"]]
    if len(shape) != 2:
        raise ValueError(f"{args.tensor} shape must be 2D, got {shape}")
    m, k = shape
    if k != int(stats["row_stride"]):
        raise ValueError(f"stats row_stride {stats['row_stride']} does not match {args.tensor} K={k}")
    if scales.shape != (k,):
        raise ValueError(f"scale shape {scales.shape} does not match K={k}")

    source_summary, source_tensors = ASTREA.read_safetensors_dir_index(source_dir, max_tensors=0)
    values = ASTREA.load_safetensors_array(source_tensors, args.tensor)
    if tuple(values.shape) != (m, k):
        raise ValueError(f"source {args.tensor} shape {values.shape} does not match HFQ shape {(m, k)}")

    chunks: list[bytes] = []
    started = time.time()
    row_chunk = max(1, int(args.row_chunk))
    total_rows = int(values.shape[0])
    for start in range(0, total_rows, row_chunk):
        end = min(total_rows, start + row_chunk)
        block = np.asarray(values[start:end], dtype=np.float32) * scales[np.newaxis, :]
        packed = ASTREA.quantize_mq4g256_values_numpy(
            block,
            clip_ratio=args.clip_ratio,
            fit=args.fit,
            ls_iters=args.ls_iters,
        )
        chunks.append(packed)
        if args.progress and (start == 0 or end == total_rows or (end // row_chunk) % args.progress == 0):
            elapsed = time.time() - started
            rate = end / max(1.0e-9, elapsed)
            eta = (total_rows - end) / max(1.0e-9, rate)
            print(
                f"[dflash_head_awq] packed rows {end}/{total_rows} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    packed = b"".join(chunks)
    if len(packed) != int(fc_record["data_size"]):
        raise ValueError(f"packed size {len(packed)} does not match HFQ data_size {fc_record['data_size']}")
    fc_record["data"] = packed

    sidecar = insert_or_replace_sidecar(layout, args.tensor, scales)
    ASTREA.write_hfq_layout(output, layout)

    result = {
        "schema": SCHEMA,
        "captured_at_utc": utc_now(),
        "status": "candidate_written",
        "base": str(base),
        "base_md5": file_md5(base),
        "candidate": str(output),
        "candidate_bytes": output.stat().st_size,
        "candidate_md5": file_md5(output),
        "source": {
            "path": source_summary["path"],
            "file_count": source_summary["file_count"],
            "tensor_count": source_summary["tensor_count"],
            "tensor_names_md5": source_summary["tensor_names_md5"],
        },
        "stats": {
            "path": str(stats_path),
            "rows": int(stats["rows"]),
            "row_stride": int(stats["row_stride"]),
        },
        "mutation": {
            "tensor": args.tensor,
            "shape": shape,
            "method": "head-only-awq-mq4",
            "alpha": args.alpha,
            "clip_ratio": args.clip_ratio,
            "fit": args.fit,
            "ls_iters": args.ls_iters if args.fit == "ls" else None,
            "row_chunk": row_chunk,
            "sidecar": sidecar,
            "scale_clamp": [scale_min, scale_max],
            "scale_min": float(np.min(scales)),
            "scale_max": float(np.max(scales)),
            "scale_mean": float(np.mean(scales)),
            "scale_std": float(np.std(scales)),
        },
        "untouched_scope": "all other drafter tensors copied byte-for-byte from base layout",
        "next_step": "run fixed target + this drafter through dflash_spec_demo and rank tau/acceptance against the base drafter",
    }
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(result, indent=2) + "\n")
    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base MQ4 DFlash draft HFQ.")
    parser.add_argument("--source-dir", required=True, help="BF16/F16 DFlash safetensors directory.")
    parser.add_argument("--stats", required=True, help="JSON from dflash_spec_demo --dump-fc-sumsq.")
    parser.add_argument("--output", required=True, help="Output candidate HFQ.")
    parser.add_argument("--tensor", default="fc.weight", help="DFlash projection tensor to repack.")
    parser.add_argument("--alpha", type=float, default=0.5, help="AWQ scale exponent alpha.")
    parser.add_argument("--scale-clamp", default="0.01:100.0", help="Clamp scales as MIN:MAX before sidecar write.")
    parser.add_argument("--clip-ratio", type=float, default=1.0, help="Row max-abs clip ratio before MQ4 pack.")
    parser.add_argument("--fit", choices=("minmax", "ls"), default="minmax", help="MQ4 block fit mode.")
    parser.add_argument("--ls-iters", type=int, default=3, help="LS refit iterations when --fit ls.")
    parser.add_argument("--row-chunk", type=int, default=128, help="Rows to repack per chunk.")
    parser.add_argument("--progress", type=int, default=4, help="Print progress every N chunks; 0 disables.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--out", help="Optional JSON artifact path.")
    args = parser.parse_args()
    write_candidate(args)


if __name__ == "__main__":
    main()
