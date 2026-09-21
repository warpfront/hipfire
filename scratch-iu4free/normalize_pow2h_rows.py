#!/usr/bin/env python3
"""Rewrite MQ4V2 pow2-half groups onto one power-of-two scale per weight row."""
import json
import mmap
import struct
import sys

import numpy as np

path = sys.argv[1]
with open(path, "r+b", buffering=0) as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    version, arch, ntensors, meta_off, data_off = struct.unpack_from("<IIIQQ", mm, 4)

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
    assert meta.get("mq4v2.pow2scale") == 2

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
        (group_size,) = struct.unpack_from("<I", mm, p)
        p += 4
        (data_len,) = struct.unpack_from("<Q", mm, p)
        p += 8
        tensors.append((name, quant_type, shape, group_size, data_len))

    data_pos = data_off
    tensor_count = 0
    row_count = 0
    changed_codes = 0
    code_count = 0
    sq_error = 0.0
    sq_reference = 0.0
    max_abs = 0.0
    span_hist = {}

    for name, quant_type, shape, group_size, data_len in tensors:
        if quant_type != 44:
            data_pos += data_len
            continue
        assert len(shape) == 2 and shape[1] % 256 == 0
        rows, k = shape
        groups = k // 256
        assert data_len == rows * groups * 136
        raw = np.ndarray((rows, groups, 136), dtype=np.uint8, buffer=mm, offset=data_pos)
        header = raw[:, :, :8]
        half_words = header.view("<u2").reshape(rows, groups, 4)
        scales = half_words[:, :, (0, 2)].copy()
        exponents = ((scales >> 10) & 0x1F).astype(np.int16) - 15
        mantissas = (scales & 0x03FF).astype(np.int16)
        assert np.all((mantissas == 0) | (mantissas == 0x200))
        row_max = exponents.max(axis=(1, 2))
        row_min = exponents.min(axis=(1, 2))
        spans, span_counts = np.unique(row_max - row_min, return_counts=True)
        for span, nspan in zip(spans.tolist(), span_counts.tolist()):
            span_hist[span] = span_hist.get(span, 0) + nspan

        ref_scale = np.ldexp(np.ones(rows, dtype=np.float32), row_max.astype(np.int32))
        ref_scale_bits = ref_scale.astype(np.float16).view(np.uint16)
        ref_zero_bits = (-8.0 * ref_scale).astype(np.float16).view(np.uint16)

        for h in range(2):
            packed = raw[:, :, 8 + 64 * h:8 + 64 * (h + 1)].copy()
            low = (packed & 0x0F).astype(np.int16) - 8
            high = ((packed >> 4) & 0x0F).astype(np.int16) - 8
            factor = np.ldexp(
                np.where(mantissas[:, :, h] == 0, 1.0, 1.5).astype(np.float32),
                exponents[:, :, h] - row_max[:, None],
            )
            low_new = np.clip(np.rint(low * factor[:, :, None]), -8, 7).astype(np.int16)
            high_new = np.clip(np.rint(high * factor[:, :, None]), -8, 7).astype(np.int16)
            changed_codes += int(np.count_nonzero(low_new != low) + np.count_nonzero(high_new != high))
            code_count += low.size + high.size
            low_ref = low.astype(np.float32) * factor[:, :, None]
            high_ref = high.astype(np.float32) * factor[:, :, None]
            low_err = low_new.astype(np.float32) - low_ref
            high_err = high_new.astype(np.float32) - high_ref
            sq_error += float(np.sum(low_err * low_err, dtype=np.float64) + np.sum(high_err * high_err, dtype=np.float64))
            sq_reference += float(np.sum(low_ref * low_ref, dtype=np.float64) + np.sum(high_ref * high_ref, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(np.abs(low_err))), float(np.max(np.abs(high_err))))
            raw[:, :, 8 + 64 * h:8 + 64 * (h + 1)] = (
                ((low_new + 8) & 15) | (((high_new + 8) & 15) << 4)
            ).astype(np.uint8)
            half_words[:, :, h * 2] = ref_scale_bits[:, None]
            half_words[:, :, h * 2 + 1] = ref_zero_bits[:, None]

        tensor_count += 1
        row_count += rows
        if tensor_count % 50 == 0:
            print(f"normalized tensors={tensor_count} rows={row_count}", flush=True)
        data_pos += data_len

    mm.flush()
    rel_rms = (sq_error / max(sq_reference, 1.0e-30)) ** 0.5
    print(
        f"ROW_NORMALIZE tensors={tensor_count} rows={row_count} changed_codes={changed_codes}/{code_count} "
        f"rel_rms={rel_rms:.9g} max_abs_ref_units={max_abs:.9g} span_hist={dict(sorted(span_hist.items()))}",
        flush=True,
    )
