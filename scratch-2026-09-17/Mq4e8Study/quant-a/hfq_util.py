#!/usr/bin/env python3
"""HFQ container reader for MQ4E8 study analysis (slice A).

Layout (hfq.rs write_hfq): header 32B [magic4, version u32, arch u32,
n u32, meta_off u64, data_off u64]; metadata JSON at 32 (self-delimiting);
index: count u32, entries {name_len u16, name, qt u8, ndim u8,
shape[ndim u32], group_size u32, data_len u64}; payloads concatenated in
index order from data_off (cumulative sizes).
"""
import json
import mmap
import struct

QT44 = 44


class Hfq:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self.mm
        magic, ver, self.arch, n, meta_off, data_off = struct.unpack_from(
            "<4sIIIQQ", mm, 0
        )
        assert magic == b"HFQM", path
        assert meta_off == 32
        self.data_off = data_off
        # metadata is a large (~14MB) JSON blob; locate its end by brace
        # matching on bytes (binary index follows; never UTF-8 decode it).
        buf = mm[32 : 32 + 64 * 1024 * 1024]
        assert buf[0] == 0x7B, "metadata must start with {"
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i, c in enumerate(buf):
            if in_str:
                if esc:
                    esc = False
                elif c == 0x5C:
                    esc = True
                elif c == 0x22:
                    in_str = False
            elif c == 0x22:
                in_str = True
            elif c == 0x7B:
                depth += 1
            elif c == 0x7D:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        assert end > 0, "metadata JSON end not found"
        self.meta = json.loads(bytes(buf[:end]).decode("utf-8"))
        pos = 32 + end
        (count,) = struct.unpack_from("<I", mm, pos)
        pos += 4
        assert count == n, (count, n)
        self.tensors = []
        off = data_off
        for _ in range(n):
            (nl,) = struct.unpack_from("<H", mm, pos)
            pos += 2
            name = mm[pos : pos + nl].decode()
            pos += nl
            qt = mm[pos]
            pos += 1
            nd = mm[pos]
            pos += 1
            shape = list(struct.unpack_from(f"<{nd}I", mm, pos))
            pos += 4 * nd
            (gs,) = struct.unpack_from("<I", mm, pos)
            pos += 4
            (dl,) = struct.unpack_from("<Q", mm, pos)
            pos += 8
            self.tensors.append(
                {"name": name, "qt": qt, "shape": shape, "gs": gs,
                 "len": dl, "off": off}
            )
            off += dl

    def payload(self, t):
        return self.mm[t["off"] : t["off"] + t["len"]]

    def by_name(self):
        return {t["name"]: t for t in self.tensors}

    def close(self):
        self.mm.close()
        self.f.close()


def decode_qt44(payload, n_elements):
    """Decode standard qt44 groups -> (s [ng,2] f32, z [ng,2] f32, q uint8 [ng*256])."""
    import numpy as np

    assert len(payload) % 136 == 0
    ng = len(payload) // 136
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(ng, 136)
    hdr = raw[:, :8].reshape(ng, 4, 2)
    bits = (hdr[:, :, 0].astype(np.uint16) | (hdr[:, :, 1].astype(np.uint16) << 8))
    f32v = bits.view(np.float16).astype(np.float32)
    s = f32v[:, [0, 2]]
    z = f32v[:, [1, 3]]
    nib = raw[:, 8:]
    q = np.empty((ng, 256), dtype=np.uint8)
    q[:, 0::2] = nib & 0xF
    q[:, 1::2] = (nib >> 4) & 0xF
    return s, z, q


def qt44_reconstruct(s, z, q):
    import numpy as np

    ng = s.shape[0]
    out = np.empty((ng, 256), dtype=np.float32)
    out[:, :128] = q[:, :128].astype(np.float32) * s[:, 0:1] + z[:, 0:1]
    out[:, 128:] = q[:, 128:].astype(np.float32) * s[:, 1:2] + z[:, 1:2]
    return out.reshape(-1)
