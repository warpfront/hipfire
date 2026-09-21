#!/usr/bin/env python3
"""Report per-row MQ4V2 weight-scale exponent structure."""
import collections
import json
import mmap
import struct
import sys

import numpy as np

path = sys.argv[1]
with open(path, "rb", buffering=0) as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    _, _, ntensors, meta_off, data_off = struct.unpack_from("<IIIQQ", mm, 4)
    depth = 0
    in_string = False
    escaped = False
    meta_end = None
    for pos in range(meta_off, len(mm)):
        c = mm[pos]
        if in_string:
            if escaped:
                escaped = False
            elif c == 0x5C:
                escaped = True
            elif c == 0x22:
                in_string = False
        elif c == 0x22:
            in_string = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                meta_end = pos + 1
                break
    assert meta_end is not None
    meta = json.loads(mm[meta_off:meta_end].decode("utf-8"))
    (count,) = struct.unpack_from("<I", mm, meta_end)
    assert count == ntensors
    p = meta_end + 4
    tensors = []
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", mm, p)
        p += 2
        name = mm[p:p + name_len].decode("utf-8")
        p += name_len
        quant_type = mm[p]
        ndim = mm[p + 1]
        p += 2
        shape = struct.unpack_from("<" + "I" * ndim, mm, p)
        p += 4 * ndim
        p += 4
        (data_len,) = struct.unpack_from("<Q", mm, p)
        p += 8
        tensors.append((name, quant_type, shape, data_len))

    span_hist = collections.Counter()
    distinct_hist = collections.Counter()
    adjacent_hist = collections.Counter()
    upward_sum_hist = collections.Counter()
    monotone_nonincreasing = 0
    row_count = 0
    max_span = 0
    max_up = 0
    max_down = 0
    data_pos = data_off
    family = {}
    for name, quant_type, shape, data_len in tensors:
        if quant_type != 44:
            data_pos += data_len
            continue
        rows, k = shape
        groups = k // 256
        raw = np.ndarray((rows, groups, 136), dtype=np.uint8, buffer=mm, offset=data_pos)
        words = raw[:, :, :8].view("<u2").reshape(rows, groups, 4)
        exps = (((words[:, :, (0, 2)] >> 10) & 0x1F).astype(np.int16) - 15).reshape(rows, groups * 2)
        row_min = exps.min(axis=1)
        row_max = exps.max(axis=1)
        spans = row_max - row_min
        for value, amount in zip(*np.unique(spans, return_counts=True)):
            span_hist[int(value)] += int(amount)
        max_span = max(max_span, int(spans.max()))
        # Exponents occupy a short integer interval, so count presence directly.
        distinct = np.zeros(rows, dtype=np.int16)
        for exponent in range(int(exps.min()), int(exps.max()) + 1):
            distinct += np.any(exps == exponent, axis=1)
        for value, amount in zip(*np.unique(distinct, return_counts=True)):
            distinct_hist[int(value)] += int(amount)
        delta = np.diff(exps, axis=1)
        for value, amount in zip(*np.unique(delta, return_counts=True)):
            adjacent_hist[int(value)] += int(amount)
        if delta.size:
            max_up = max(max_up, int(delta.max()))
            max_down = min(max_down, int(delta.min()))
        upward = np.maximum(delta, 0).sum(axis=1)
        for value, amount in zip(*np.unique(upward, return_counts=True)):
            upward_sum_hist[int(value)] += int(amount)
        monotone_nonincreasing += int(np.count_nonzero(np.all(delta <= 0, axis=1)))
        row_count += rows
        if ".layers.0." in name and any(key in name for key in ("gate_proj", "up_proj", "down_proj", "qkv", "q_proj", "k_proj", "v_proj")):
            family[name] = {
                "rows": rows,
                "k": k,
                "span": dict(sorted((int(v), int(n)) for v, n in zip(*np.unique(spans, return_counts=True)))),
                "distinct": dict(sorted((int(v), int(n)) for v, n in zip(*np.unique(distinct, return_counts=True)))),
            }
        data_pos += data_len
    print(f"artifact_pow2scale={meta.get('mq4v2.pow2scale')} rows={row_count} max_span={max_span} max_adjacent_up={max_up} max_adjacent_down={max_down} monotone_nonincreasing={monotone_nonincreasing}")
    print("span_hist", dict(sorted(span_hist.items())))
    print("distinct_hist", dict(sorted(distinct_hist.items())))
    print("adjacent_delta_hist", dict(sorted(adjacent_hist.items())))
    print("upward_bit_sum_hist", dict(sorted(upward_sum_hist.items())))
    print("layer0_families", json.dumps(family, sort_keys=True))
