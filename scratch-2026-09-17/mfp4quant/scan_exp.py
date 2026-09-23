#!/usr/bin/env python3
"""Scan qt24 MFP4G32 tensors in an .hfq: per-block UE8M0 exponent stats.

Row layout: 16B header (row_scale_a f16 @0, block_count u16 @4, flags u8 @6),
then n_blocks x 17B (1B UE8M0 scale + 16B E2M1 nibbles).
Reports: global exponent min/max over blocks with any nonzero nibble,
max per-row exponent span, count of e==0 blocks with nonzero nibbles.
"""
import struct, sys, mmap

path = sys.argv[1]
f = open(path, "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
assert mm[0:4] == b"HFQM", mm[0:4]
nt = struct.unpack_from("<I", mm, 12)[0]
meta_off = struct.unpack_from("<Q", mm, 16)[0]
data_off = struct.unpack_from("<Q", mm, 24)[0]
# metadata then index
pos = 32
# metadata length unknown: index_offset = meta_off + meta_len; read tensor count at index.
# metadata is JSON; find index by scanning: index starts with count==nt.
# Simpler: index_offset follows metadata; we don't know meta len. Instead the
# writer put index right after metadata bytes; metadata_bytes len unknown from file.
# Workaround: search for index start: at index, u32 count == nt.
import json
# metadata runs from 32; try to parse increasing prefixes? Instead: data_off known;
# index = [pos .. data_start_unaligned]; parse forward from 32 by reading meta len
raw = bytes(mm[32:data_off])
# metadata is one JSON value; find its end by brace matching (string-aware).
depth = 0
in_str = False
esc = False
meta_end = None
for i, ch in enumerate(raw):
    c = chr(ch)
    if in_str:
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = False
    else:
        if c == '"':
            in_str = True
        elif c == "{" or c == "[":
            depth += 1
        elif c == "}" or c == "]":
            depth -= 1
            if depth == 0:
                meta_end = i + 1
                break
assert meta_end is not None
index_pos = 32 + meta_end
# skip whitespace between metadata and index
while index_pos < data_off and mm[index_pos:index_pos+1] in b" \t\r\n":
    index_pos += 1
count = struct.unpack_from("<I", mm, index_pos)[0]
assert count == nt, (count, nt)
index_pos += 4
tensors = []
for _ in range(nt):
    nl = struct.unpack_from("<H", mm, index_pos)[0]; index_pos += 2
    name = bytes(mm[index_pos:index_pos+nl]).decode(); index_pos += nl
    qt = mm[index_pos]; index_pos += 1
    nd = mm[index_pos]; index_pos += 1
    shape = struct.unpack_from("<" + "I"*nd, mm, index_pos); index_pos += 4*nd
    gs = struct.unpack_from("<I", mm, index_pos)[0]; index_pos += 4
    dl = struct.unpack_from("<Q", mm, index_pos)[0]; index_pos += 8
    tensors.append((name, qt, shape, dl))

QT_MFP4 = 24
gmin, gmax = 255, 0
max_span = 0
e0_nonzero = 0
n_blocks = 0
n_nz_blocks = 0
n_tensors = 0
off = data_off
for (name, qt, shape, dl) in tensors:
    if qt != QT_MFP4:
        off += dl
        continue
    n_tensors += 1
    m, k = shape[0], shape[1]
    nb = k // 32
    row_bytes = 16 + 17*nb
    assert dl == m*row_bytes, (name, dl, m*row_bytes)
    for r in range(m):
        base = off + r*row_bytes
        e_min, e_max = 255, 0
        for b in range(nb):
            bb = base + 16 + b*17
            e = mm[bb]
            # any nonzero nibble in the 16 payload bytes?
            nz = False
            for i in range(1, 17):
                if mm[bb+i] != 0:
                    nz = True
                    break
            n_blocks += 1
            if not nz:
                continue
            n_nz_blocks += 1
            if e < gmin: gmin = e
            if e > gmax: gmax = e
            if e < e_min: e_min = e
            if e > e_max: e_max = e
            if e == 0:
                e0_nonzero += 1
        if e_max >= e_min:
            span = e_max - e_min
            if span > max_span: max_span = span
    off += dl

print(f"tensors_qt24={n_tensors} blocks={n_blocks} nz_blocks={n_nz_blocks}")
print(f"exp_min={gmin} exp_max={gmax} (bias127: [{gmin-127},{gmax-127}])")
print(f"max_per_row_span={max_span}")
print(f"e0_with_nonzero_nibble={e0_nonzero}")
print(f"fold_q=e_max-127-6 normal_span_ok_11={ (gmax-127-6) if gmax>0 else 'n/a' }")
