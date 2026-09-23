#!/usr/bin/env python3
"""Gate-4 metrics: per-family tail-1% MSE + max-coef MSE vs fixture codes
(slice A). Reference = parent BF16 -> f32 x exact AWQ f32 scales -> float32
FWHT (same DAG as the encoder). Decodes fixture + one artifact, accumulates
pooled per-family stats.

Usage: metrics.py <artifact.hfq> <tag>   # writes metrics_<tag>.json
Parent/AWQ/fixture paths are pinned below.
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

QA = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a"
sys.path.insert(0, QA)
from hfq_util import Hfq, decode_qt44, qt44_reconstruct  # noqa: E402
from oracle_py import fwht256, gen_signs  # noqa: E402

PARENT = "/home/kaden/qcal/parents/qwen3.8-27b"
FIXTURE = "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq"

FAMILIES = [
    ("lm_head.weight", "lm_head"),
    ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_qkv"),
    ("linear_attn.in_proj_z.weight", "linear_attn.in_proj_z"),
    ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_a"),
    ("linear_attn.in_proj_b.weight", "linear_attn.in_proj_b"),
    ("linear_attn.out_proj.weight", "linear_attn.out_proj"),
    ("mlp.down_proj.weight", "mlp.down_proj"),
    ("mlp.gate_proj.weight", "mlp.gate_proj"),
    ("mlp.up_proj.weight", "mlp.up_proj"),
    ("self_attn.q_proj.weight", "self_attn.q_proj"),
    ("self_attn.k_proj.weight", "self_attn.k_proj"),
    ("self_attn.v_proj.weight", "self_attn.v_proj"),
    ("self_attn.o_proj.weight", "self_attn.o_proj"),
]


def family_of(name):
    for suffix, fam in FAMILIES:
        if name.endswith(suffix):
            return fam
    return "other"


class ParentStore:
    """Minimal safetensors BF16 reader (exact bit-widening, no ml_dtypes)."""

    def __init__(self, d):
        self.files = sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors")
        )
        self.index = {}  # name -> (path, dtype, shape, begin, end)
        for p in self.files:
            with open(p, "rb") as f:
                (hl,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hl).decode("utf-8"))
            base = 8 + hl
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                o0, o1 = meta["data_offsets"]
                self.index[name] = (
                    p, meta["dtype"], tuple(meta["shape"]), base + o0, base + o1,
                )
        self._mm = {}

    def load_f32(self, name):
        p, dtype, shape, o0, o1 = self.index[name]
        mm = self._mmap(p)
        raw = mm[o0:o1]
        if dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).copy()
            u32 = u16.astype(np.uint32) << np.uint32(16)
            return u32.view(np.float32).reshape(shape)
        elif dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
        else:
            raise ValueError(f"{name}: unsupported dtype {dtype}")

    def _mmap(self, p):
        e = self._mm.get(p)
        if e is None:
            f = open(p, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._mm[p] = (mm, f)
            return mm
        return e[0]


def load_awq_dump():
    b = open(f"{QA}/awq_scales.bin", "rb").read()
    pos = 0
    (n,) = struct.unpack_from("<I", b, pos); pos += 4
    recs = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", b, pos); pos += 2
        name = b[pos : pos + nl].decode(); pos += nl
        m, k = struct.unpack_from("<II", b, pos); pos += 8
        awq = b[pos]; pos += 1
        sc = None
        if awq:
            (kk,) = struct.unpack_from("<I", b, pos); pos += 4
            sc = np.frombuffer(b[pos : pos + 4 * kk], dtype=np.float32).copy()
            pos += 4 * kk
        recs[name] = (m, k, sc)
    return recs


def main():
    art_path, tag = sys.argv[1], sys.argv[2]
    print(f"metrics {tag}: artifact={art_path}", flush=True)
    recs = load_awq_dump()
    ps = ParentStore(PARENT)
    fx = Hfq(FIXTURE)
    ax = Hfq(art_path)
    fn = fx.by_name()
    an = ax.by_name()
    s1 = gen_signs(42)
    s2 = gen_signs(1042)

    agg = {}  # fam -> {tail_se_f, tail_n_f, tail_se_c, tail_n_c, max_se_f, max_n_f, max_se_c, max_n_c, tensors}
    for name, (m, k, sc) in recs.items():
        fam = family_of(name)
        a = agg.setdefault(fam, {
            "tensors": 0, "tail_f": 0.0, "tail_nf": 0, "tail_c": 0.0,
            "tail_nc": 0, "max_f": 0.0, "max_nf": 0, "max_c": 0.0, "max_nc": 0,
        })
        a["tensors"] += 1
        w = ps.load_f32(name).reshape(m, k)
        if sc is not None:
            w = w * sc.reshape(1, k)
        ref = fwht256(w.reshape(-1, 256), s1, s2).reshape(-1).astype(np.float64)
        fxe = fn[name]
        axe = an[name]
        sf, zf, qf = decode_qt44(fx.payload(fxe), m * k)
        sc_, zc, qc = decode_qt44(ax.payload(axe), m * k)
        rf = qt44_reconstruct(sf, zf, qf).astype(np.float64)
        rc = qt44_reconstruct(sc_, zc, qc).astype(np.float64)
        se_f = (rf - ref) ** 2
        se_c = (rc - ref) ** 2
        p99 = float(np.quantile(np.abs(ref), 0.99))
        mask = np.abs(ref) >= p99
        a["tail_f"] += float(se_f[mask].sum()); a["tail_nf"] += int(mask.sum())
        a["tail_c"] += float(se_c[mask].sum()); a["tail_nc"] += int(mask.sum())
        # max-coef: argmax |ref| per 256-group
        amax = np.abs(ref).reshape(-1, 256).argmax(axis=1)
        idx = (np.arange(ref.size // 256) * 256 + amax)
        a["max_f"] += float(se_f[idx].sum()); a["max_nf"] += len(idx)
        a["max_c"] += float(se_c[idx].sum()); a["max_nc"] += len(idx)
        if a["tensors"] % 100 == 0:
            print(f"  ...{a['tensors']} tensors in {fam} (total done: "
                  f"{sum(v['tensors'] for v in agg.values())}/497)", flush=True)

    out = {}
    for fam, a in sorted(agg.items()):
        out[fam] = {
            "tensors": a["tensors"],
            "tail_mse_fixture": a["tail_f"] / a["tail_nf"],
            "tail_mse_artifact": a["tail_c"] / a["tail_nc"],
            "tail_ratio": (a["tail_c"] / a["tail_nc"]) / (a["tail_f"] / a["tail_nf"]),
            "maxcoef_mse_fixture": a["max_f"] / a["max_nf"],
            "maxcoef_mse_artifact": a["max_c"] / a["max_nc"],
            "maxcoef_ratio": (a["max_c"] / a["max_nc"]) / (a["max_f"] / a["max_nf"]),
        }
    json.dump(out, open(f"{QA}/metrics_{tag}.json", "w"), indent=1)
    print(f"metrics {tag}: wrote {QA}/metrics_{tag}.json")
    for fam in sorted(out):
        r = out[fam]
        print(f"  {fam:28s} tail_f={r['tail_mse_fixture']:.6e} tail_c={r['tail_mse_artifact']:.6e} "
              f"ratio={r['tail_ratio']:.3f} maxc_ratio={r['maxcoef_ratio']:.3f}")



if __name__ == "__main__":
    main()
