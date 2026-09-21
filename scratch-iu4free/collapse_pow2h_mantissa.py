#!/usr/bin/env python3
"""Fold MQ4V2 1.5 mantissas into signed int4 codes at the same exponent."""
import json
import mmap
import struct
import sys

import numpy as np

path = sys.argv[1]
with open(path, "r+b", buffering=0) as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
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
    assert meta.get("mq4v2.pow2scale") == 2
    (count,) = struct.unpack_from("<I", mm, meta_end)
    assert count == ntensors
    p = meta_end + 4
    tensors = []
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", mm, p)
        p += 2 + name_len
        quant_type = mm[p]
        ndim = mm[p + 1]
        p += 2
        shape = struct.unpack_from("<" + "I" * ndim, mm, p)
        p += 4 * ndim
        p += 4
        (data_len,) = struct.unpack_from("<Q", mm, p)
        p += 8
        tensors.append((quant_type, shape, data_len))

    data_pos = data_off
    tensor_count = 0
    changed_codes = 0
    code_count = 0
    sq_error = 0.0
    sq_reference = 0.0
    max_abs = 0.0
    for quant_type, shape, data_len in tensors:
        if quant_type != 44:
            data_pos += data_len
            continue
        rows, k = shape
        groups = k // 256
        raw = np.ndarray((rows, groups, 136), dtype=np.uint8, buffer=mm, offset=data_pos)
        words = raw[:, :, :8].view("<u2").reshape(rows, groups, 4)
        for h in range(2):
            scale = words[:, :, h * 2]
            use = (scale & 0x03FF) == 0x0200
            if not np.any(use):
                continue
            packed = raw[:, :, 8 + 64 * h:8 + 64 * (h + 1)].copy()
            low = (packed & 15).astype(np.int16) - 8
            high = ((packed >> 4) & 15).astype(np.int16) - 8
            factor = np.where(use, 1.5, 1.0).astype(np.float32)[:, :, None]
            low_new = np.clip(np.rint(low * factor), -8, 7).astype(np.int16)
            high_new = np.clip(np.rint(high * factor), -8, 7).astype(np.int16)
            changed_codes += int(np.count_nonzero(low_new != low) + np.count_nonzero(high_new != high))
            code_count += low.size + high.size
            low_ref = low.astype(np.float32) * factor
            high_ref = high.astype(np.float32) * factor
            low_err = low_new.astype(np.float32) - low_ref
            high_err = high_new.astype(np.float32) - high_ref
            sq_error += float(np.sum(low_err * low_err, dtype=np.float64) + np.sum(high_err * high_err, dtype=np.float64))
            sq_reference += float(np.sum(low_ref * low_ref, dtype=np.float64) + np.sum(high_ref * high_ref, dtype=np.float64))
            max_abs = max(max_abs, float(np.max(np.abs(low_err))), float(np.max(np.abs(high_err))))
            raw[:, :, 8 + 64 * h:8 + 64 * (h + 1)] = (
                ((low_new + 8) & 15) | (((high_new + 8) & 15) << 4)
            ).astype(np.uint8)
            new_scale = scale & 0xFC00
            words[:, :, h * 2] = new_scale
            exp = ((new_scale >> 10) & 0x1F).astype(np.int16) - 15
            zero = np.ldexp(np.full(exp.shape, -8.0, dtype=np.float32), exp.astype(np.int32))
            words[:, :, h * 2 + 1] = zero.astype(np.float16).view(np.uint16)
        tensor_count += 1
        data_pos += data_len
    mm.flush()
    rel_rms = (sq_error / max(sq_reference, 1.0e-30)) ** 0.5
    print(f"MANTISSA_COLLAPSE tensors={tensor_count} changed_codes={changed_codes}/{code_count} rel_rms={rel_rms:.9g} max_abs_ref_units={max_abs:.9g}")
