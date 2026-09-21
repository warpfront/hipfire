#!/usr/bin/env python3
"""Scan an HFQ file: metadata flags + MQ4V2 (qt44) scale mantissa sample check."""
import json
import mmap
import struct
import sys

path = sys.argv[1]
f = open(path, "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

magic = mm[0:4]
assert magic == b"HFQM", magic
version, arch, ntensors, meta_off, data_off = struct.unpack_from("<IIIQQ", mm, 4)
print(f"magic=HFQM version={version} arch={arch} ntensors={ntensors} meta_off={meta_off} data_off={data_off}")

def find_json_end(buf, start):
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(buf)):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        else:
            if c == 0x22:
                in_str = True
            elif c == 0x7B:
                depth += 1
            elif c == 0x7D:
                depth -= 1
                if depth == 0:
                    return i + 1
    raise ValueError("unbalanced JSON")

meta_end = find_json_end(mm, meta_off)
meta = json.loads(mm[meta_off:meta_end].decode("utf-8"))
print(f"metadata_len={meta_end - meta_off} mq4v2.symmetric={meta.get('mq4v2.symmetric')} mq4v2.pow2scale={meta.get('mq4v2.pow2scale')}")
idx_off = meta_end
(n,) = struct.unpack_from("<I", mm, idx_off)
assert n == ntensors, (n, ntensors)
p = idx_off + 4
tensors = []
for _ in range(n):
    (nl,) = struct.unpack_from("<H", mm, p); p += 2
    name = mm[p:p+nl].decode("utf-8"); p += nl
    qt = mm[p]; nd = mm[p+1]; p += 2
    shape = struct.unpack_from("<" + "I"*nd, mm, p); p += 4*nd
    (gs,) = struct.unpack_from("<I", mm, p); p += 4
    (dl,) = struct.unpack_from("<Q", mm, p); p += 8
    tensors.append((name, qt, shape, gs, dl))

from collections import Counter
print("qt histogram:", dict(Counter(qt for _, qt, _, _, _ in tensors)))

def f16_to_f32(bits):
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    if e == 0:
        v = m / 1024.0 * 2.0**-14
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1 + m / 1024.0) * 2.0**(e - 15)
    return -v if s else v

off = data_off
total_groups = 0
checked = 0
bad_mantissa = 0
bad_zero = 0
zero_scale = 0
bad_examples = []
n44 = 0
for (name, qt, shape, gs, dl) in tensors:
    if qt != 44:
        off += dl
        continue
    n44 += 1
    assert dl % 136 == 0, (name, dl)
    ng = dl // 136
    total_groups += ng
    # stride across the blob, cap ~4096 groups per tensor
    stride = max(1, ng // 4096)
    for g in range(0, ng, stride):
        base = off + g * 136
        s0, z0, s1, z1 = struct.unpack_from("<HHHH", mm, base)
        for s, z in ((s0, z0), (s1, z1)):
            if s == 0:
                zero_scale += 1
                continue
            checked += 1
            if s & 0x3FF:
                bad_mantissa += 1
                if len(bad_examples) < 5:
                    bad_examples.append((name, g, hex(s)))
            else:
                # zero should be exactly -8*d for pow2 d
                d = f16_to_f32(s)
                zv = f16_to_f32(z)
                if zv != -8.0 * d:
                    bad_zero += 1
                    if len(bad_examples) < 10:
                        bad_examples.append((name, g, f"zero {hex(z)} != -8*{hex(s)}"))
    off += dl

print(f"qt44 tensors={n44} total_groups={total_groups} checked_scales={checked} "
      f"zero_scales={zero_scale} bad_mantissa={bad_mantissa} bad_zero={bad_zero}")
for e in bad_examples:
    print("  example:", e)
print("MANTISSA_CHECK:", "PASS" if bad_mantissa == 0 and checked > 0 else "FAIL")
