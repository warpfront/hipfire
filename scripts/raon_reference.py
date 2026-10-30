#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Raon-1-mini-preview reference/fixture generator (slice O, PLAN sections 6.2/6.3).

Subcommands
  fixtures     per-module oracle fixtures + negative controls + state/logits tokens
  tokenizer    HF-tokenizers ids + official chat-template renders for a corpus
  layer-dump   BF16 per-layer reference dumps (streams, RDA state, MLA latent rows,
               mHC controls, branch outputs) from the checkpoint, one layer in RAM
  kld-check    validate HFKLDR v1 teacher refs against the in-tree eval reader
               layout and re-tokenize their source texts
  selftest     CPU-only: independent NumPy/torch checks of the chart, the RDA
               recurrence, MXFP8 decode and the padded (1088->1280) FFN, plus the
               proof that every negative control differs from the true reference

The reference math is the pinned modeling file itself (REF/modeling_raon1_mini.py,
loaded as a package and executed): module classes are instantiated with real or
small configs and fed the fixture inputs.  Where a tensor boundary of the pinned
file is not reachable through its public methods (mHC selector `z`, RDA prepared
q/k/v, MLA attention) the module is driven through selector/identity projections
so the pinned forward code runs unmodified, and every re-derived quantity is
checked against an independent float64 NumPy implementation.

Fixture schema (version 1)
  <fixtures>/<module>/manifest.json
      {"schema_version":1,"module":M,"seed":S,"ref_sha256":{file:sha},
       "cases":[{"name","file","dims":{..},"mode":"bf16"|"f32","sha256",
                 "inputs":[names],"outputs":[names],
                 "weights":"<path relative to the module dir>" (optional),
                 "params":{..} (optional),"note":".."}]}
  <fixtures>/<module>/<case>.safetensors   all tensors float32; "mode":"bf16"
      cases hold BF16-rounded (RNE) values in float32 containers wherever the
      reference crosses a model-dtype boundary; scalars have shape [1]; integer
      ids (token ids, expert indices) are stored as exactly-representable float32.
  weights files (optional, manifest "weights"): extra safetensors merged with the
      case tensors (names are disjoint).  Big BF16-valued weight files use the
      safetensors BF16 dtype; convert on read.
  <fixtures>/negative_controls/<module>/...  same format; deliberately wrong
      references that a correct module MUST fail (`raon_oracle module --negative`).
  <fixtures>/MANIFEST.json, <fixtures>/fixtures.sha256   digests of every file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import struct
import sys
import time
import types
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_VERSION = 1
DEFAULT_SEED = 20261030
DEFAULT_REF = "/home/kaden/qcal/release-0.4.3/raon/ref"
DEFAULT_PARENT = "/home/kaden/qcal/parents/raon-1-mini-preview"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_PROMPTS = REPO / "benchmarks" / "prompts" / "raon"

# Predeclared engineering bands (PLAN 6.2).  Recorded in MANIFEST.json so every
# module oracle reads the same numbers; they are gates, not measured errors.
BANDS = {
    "f32": {"rel_l2": 1e-4, "max_abs_abs": 2e-5, "max_abs_rel": 2e-4},
    "f32_rda_state_long": {"rel_l2": 5e-4},
    "bf16": {"rel_l2": 1e-2, "cosine": 0.9999},
    "bitwise": {"note": "same engine/route repeated, eager vs capture: byte-identical"},
}

torch.set_num_threads(min(32, os.cpu_count() or 8))
torch.set_grad_enabled(False)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rng_for(seed: int, tag: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(tag.encode())])


def randn(rng, shape, scale=1.0) -> torch.Tensor:
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32) * np.float32(scale))


def uniform(rng, shape, lo, hi) -> torch.Tensor:
    return torch.from_numpy(rng.uniform(lo, hi, size=shape).astype(np.float32))


def bfr(x) -> torch.Tensor:
    """Round to BF16 (RNE) and return as float32."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(torch.bfloat16).to(torch.float32)


def f32c(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.detach().to(torch.float32).contiguous()


def scal(v) -> torch.Tensor:
    return torch.tensor([float(v)], dtype=torch.float32)


def rel_l2(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    den = float(b.norm())
    return float((a - b).norm()) / (den if den > 0 else 1.0)


def max_abs(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).abs().max()) if a.numel() else 0.0


def log(msg: str) -> None:
    print(f"[raon_reference {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# pinned reference modules
# ----------------------------------------------------------------------------
class Ref:
    """The pinned REF modeling/config files, loaded as a package and executed."""

    def __init__(self, ref_dir: str):
        self.dir = Path(ref_dir)
        pylib = self.dir / "teacher" / "pylib"
        try:
            import transformers  # noqa: F401
        except ImportError:
            if pylib.is_dir():
                sys.path.insert(0, str(pylib))
        pkg = types.ModuleType("raon_ref")
        pkg.__path__ = [str(self.dir)]
        sys.modules["raon_ref"] = pkg
        for n in ("configuration_raon1_mini", "modeling_raon1_mini"):
            spec = importlib.util.spec_from_file_location(f"raon_ref.{n}", self.dir / f"{n}.py")
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"raon_ref.{n}"] = m
            spec.loader.exec_module(m)
            setattr(pkg, n, m)
        self.cfgmod = sys.modules["raon_ref.configuration_raon1_mini"]
        self.mod = sys.modules["raon_ref.modeling_raon1_mini"]
        self.config_json = json.loads((self.dir / "config.json").read_text())
        self.sha = {
            f: sha256_file(self.dir / f)
            for f in (
                "modeling_raon1_mini.py",
                "configuration_raon1_mini.py",
                "config.json",
                "chat_template.jinja",
                "generation_config.json",
                "tokenizer_config.json",
            )
        }

    def cfg(self, **over):
        base = {
            k: v
            for k, v in self.config_json.items()
            if k not in ("auto_map", "architectures", "quantization_config", "model_type", "dtype")
        }
        base.update(over)
        return self.cfgmod.Raon1MiniConfig(**base)

    def rda_config(self, heads: int, dim: int) -> dict:
        d = dict(self.config_json["rda_config"])
        d["num_heads"] = heads
        d["head_dim"] = dim
        return d


class Select(nn.Module):
    """Slice the last dim: lets the pinned forward code consume already-projected tensors."""

    def __init__(self, off: int, width: int):
        super().__init__()
        self.off, self.width = off, width

    def forward(self, x):
        return x[..., self.off : self.off + self.width]


class Fixed(nn.Module):
    """Returns a precomputed tensor regardless of input (a projection fed from a fixture)."""

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.value = value

    def forward(self, x):
        return self.value


# ----------------------------------------------------------------------------
# checkpoint access (one tensor at a time, never the whole model)
# ----------------------------------------------------------------------------
class Ckpt:
    def __init__(self, parent: str):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = Path(parent)
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._h = {}

    def _handle(self, shard):
        if shard not in self._h:
            self._h[shard] = self._safe_open(str(self.dir / shard), "pt")
        return self._h[shard]

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.index[name]).get_tensor(name)

    def rows(self, name: str, rows) -> torch.Tensor:
        sl = self._handle(self.index[name]).get_slice(name)
        return torch.cat([sl[int(r) : int(r) + 1] for r in rows], dim=0)

    def has(self, name: str) -> bool:
        return name in self.index

    def layer_names(self, i: int):
        p = f"model.layers.{i}."
        return [n for n in self.index if n.startswith(p)]


# ----------------------------------------------------------------------------
# case writer
# ----------------------------------------------------------------------------
class Fixtures:
    def __init__(self, root: str, ref: Ref, seed: int):
        self.root = Path(root)
        self.ref = ref
        self.seed = seed
        self.cases: dict[tuple[str, str], list[dict]] = {}

    def _dir(self, module: str, negative: bool) -> Path:
        d = self.root / ("negative_controls" if negative else "") / module
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_weights(self, module: str, fname: str, tensors: dict, bf16: bool = False) -> str:
        from safetensors.torch import save_file

        d = self._dir(module, False)
        out = {}
        for k in sorted(tensors):
            t = tensors[k]
            out[k] = (t.to(torch.bfloat16) if bf16 else f32c(t)).contiguous()
        save_file(out, str(d / fname), metadata={"raon": json.dumps({"kind": "weights", "module": module}, sort_keys=True)})
        return fname

    def add(self, module, name, dims, mode, inputs, outputs, *, weights=None, params=None,
            note=None, negative=False, expect_fail_outputs=None):
        from safetensors.torch import save_file

        assert mode in ("bf16", "f32")
        clash = set(inputs) & set(outputs)
        assert not clash, f"{module}/{name}: tensor in both inputs and outputs: {clash}"
        tensors = {}
        for k, v in list(inputs.items()) + list(outputs.items()):
            t = f32c(v).clone()
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{module}/{name}: non-finite values in tensor {k}")
            tensors[k] = t
        d = self._dir(module, negative)
        path = d / f"{name}.safetensors"
        # one metadata key holding sorted JSON: safetensors serialises multi-key metadata in
        # hash-map order, which would make reruns differ byte-wise
        meta = {"raon": json.dumps({
            "schema_version": SCHEMA_VERSION, "module": module, "case": name, "mode": mode, "dims": dims,
            "inputs": sorted(inputs), "outputs": sorted(outputs),
            "ref_modeling_sha256": self.ref.sha["modeling_raon1_mini.py"]}, sort_keys=True)}
        save_file({k: tensors[k] for k in sorted(tensors)}, str(path), metadata=meta)
        entry = {
            "name": name,
            "file": path.name,
            "dims": dims,
            "mode": mode,
            "inputs": sorted(inputs),
            "outputs": sorted(outputs),
        }
        if weights:
            entry["weights"] = weights
        if params:
            entry["params"] = params
        if note:
            entry["note"] = note
        if negative:
            entry["expect"] = "fail"
            entry["wrong_outputs"] = sorted(expect_fail_outputs or [])
        self.cases.setdefault((module, "neg" if negative else "pos"), []).append(entry)

    def finalize(self):
        mods = {}
        for (module, kind), entries in sorted(self.cases.items()):
            negative = kind == "neg"
            d = self._dir(module, negative)
            for e in entries:
                e["sha256"] = sha256_file(d / e["file"])
            entries.sort(key=lambda e: e["name"])
            man = {
                "schema_version": SCHEMA_VERSION,
                "module": module,
                "negative_control": negative,
                "seed": self.seed,
                "ref_sha256": self.ref.sha,
                "bands": BANDS,
                "cases": entries,
            }
            (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
            mods[f"{'negative_controls/' if negative else ''}{module}"] = len(entries)
        return mods


def tree_digests(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST.json", "fixtures.sha256"):
            out[str(p.relative_to(root))] = sha256_file(p)
    return out


def save_npz_deterministic(path: Path, arrays: dict):
    """np.savez without wall-clock zip timestamps so reruns are byte-identical."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())


def write_u32(path: Path, ids):
    path.write_bytes(np.asarray(ids, dtype="<u4").tobytes())


# ----------------------------------------------------------------------------
# independent float64 NumPy implementations (cross-checks of the pinned module)
# ----------------------------------------------------------------------------
def np_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def np_logsigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.minimum(x, 0.0) - np.log1p(np.exp(-np.abs(x)))


def np_helmert(n):
    q = np.zeros((n, n - 1))
    for k in range(1, n):
        q[:k, k - 1] = 1.0
        q[k, k - 1] = -float(k)
        q[:, k - 1] /= math.sqrt(k * (k + 1.0))
    return q


def np_mhc_controls(z, alpha_pre, alpha_post, alpha_z, eta, b_pre, b_post, b_z, n, post_factor=2.0):
    """PLAN 1.4 chart, float64.  z: [N, 2n+(n-1)^2] scaled selector outputs."""
    z = np.asarray(z, dtype=np.float64)
    nm = n - 1
    q = np_helmert(n)
    pre = np_sigmoid(alpha_pre * z[:, :n] + np.asarray(b_pre, dtype=np.float64))
    post = post_factor * np_sigmoid(alpha_post * z[:, n : 2 * n] + np.asarray(b_post, dtype=np.float64))
    zm = (alpha_z * z[:, 2 * n :] + np.asarray(b_z, dtype=np.float64)).reshape(-1, nm, nm)
    e = np.einsum("im,bmk,jk->bij", q, zm, q)
    bmat = 0.5 * (np.eye(n) + np.full((n, n), 1.0 / n))
    rho = np.maximum(0.0, (-e / bmat).reshape(len(z), -1).max(axis=1))
    phi = bmat[None] + e / (1.0 + rho)[:, None, None]
    gamma = float(np_sigmoid(np.float64(eta)))
    res = (1.0 - gamma) * np.eye(n)[None] + gamma * phi
    return pre, post, res, e, rho


def np_rda_recurrence(q, k, v, log_alpha, beta, state):
    """PLAN 1.5 serial delta rule, float64.  q,k,v [T,H,K]; log_alpha [T,H,K]; beta [T,H];
    state [H,V,K].  Order: decay, predict, error, write, read-after-write."""
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    la = np.asarray(log_alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    s = np.asarray(state, dtype=np.float64).copy()
    out = np.empty_like(v)
    for t in range(q.shape[0]):
        s *= np.exp(la[t])[:, None, :]
        pred = np.matmul(s, k[t][:, :, None])[:, :, 0]
        err = v[t] - pred
        s += (beta[t][:, None] * err)[:, :, None] * k[t][:, None, :]
        out[t] = np.matmul(s, q[t][:, :, None])[:, :, 0]
    return out, s


def np_rda_scalar_gate(q, k, v, log_decay, beta, state):
    """Plain scalar-gate delta rule (one decay per head), float64; same update order."""
    la = np.repeat(np.asarray(log_decay, dtype=np.float64)[:, :, None], q.shape[-1], axis=2)
    return np_rda_recurrence(q, k, v, la, beta, state)


def np_attention(q, k, v, scale, prefix):
    """q [H,N,Dqk], k [H,T,Dqk], v [H,T,Dv] -> [H,N,Dv]; query i sees keys j <= prefix+i."""
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    s = np.einsum("hid,hjd->hij", q, k) * scale
    n, t = q.shape[1], k.shape[1]
    mask = np.arange(t)[None, :] <= (prefix + np.arange(n))[:, None]
    s = np.where(mask[None], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hij,hjd->hid", p, v)


def np_situ_glu(g, u, gb=4.0, vb=25.0):
    g = np.asarray(g, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return (gb * np.tanh(g / gb) * np_sigmoid(g)) * (vb * np.tanh(u / vb))


def np_e4m3fn_decode(byte_tensor: torch.Tensor) -> np.ndarray:
    """Independent E4M3FN decoder (bias 7, no inf, NaN = 0x7f/0xff)."""
    b = byte_tensor.view(torch.uint8).numpy().astype(np.int64)
    sign = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7.0))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return sign * val


def np_mxfp8_dequant(w: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    """Independent decode of the MXFP8 block layout [N,K] / [N,K/32] (PLAN 1.8)."""
    rows, cols = w.shape
    vals = np_e4m3fn_decode(w).reshape(rows, cols // 32, 32)
    sc = scale.view(torch.uint8).numpy().astype(np.float64)
    return (vals * (2.0 ** (sc - 127.0))[:, :, None]).reshape(rows, cols)


# ----------------------------------------------------------------------------
# norm / head / embedding broadcast
# ----------------------------------------------------------------------------
def to_mode(t: torch.Tensor, mode: str) -> torch.Tensor:
    return t.to(torch.bfloat16) if mode == "bf16" else t.to(torch.float32)


def in_mode(t: torch.Tensor, mode: str) -> torch.Tensor:
    """Fixture value of an input at a model-dtype boundary."""
    return bfr(t) if mode == "bf16" else f32c(t)


def rms_ref(ref: Ref, x, gamma, eps, mode):
    m = ref.mod.Raon1MiniRMSNorm(gamma.numel(), eps)
    m.weight.data = to_mode(gamma, "f32")
    xm = to_mode(x, mode)
    return m(xm).float(), m(xm, keep_fp32=True).float()


def gen_norm(fx: Fixtures, ck: Ckpt):
    ref = fx.ref
    gam = {
        2560: ("model.layers.1.input_layernorm.weight", 1e-5),
        128: ("model.layers.1.self_attn.o_norm.weight", 1e-5),
        512: ("model.layers.3.self_attn.kv_a_layernorm.weight", 1e-6),
        576: ("model.layers.3.self_attn.q_a_layernorm.weight", 1e-6),
    }
    specs = []
    for c, (gname, eps) in gam.items():
        for n in (1, 7, 64, 257):
            specs.append((f"rms_c{c}_n{n}", n, c, eps, gname, "bf16", "plain"))
    specs += [
        ("rms_c2560_outlier", 33, 2560, 1e-5, gam[2560][0], "bf16", "outlier"),
        ("rms_c2560_tiny_rows", 33, 2560, 1e-5, gam[2560][0], "bf16", "tiny"),
        ("rms_c512_tiny_rows_eps1e-6", 17, 512, 1e-6, gam[512][0], "bf16", "tiny"),
        ("rms_c2560_zero_row", 9, 2560, 1e-5, gam[2560][0], "bf16", "zero"),
        ("rms_c128_perhead_n280", 280, 128, 1e-5, gam[128][0], "bf16", "plain"),
        ("rms_c2560_n64_f32", 64, 2560, 1e-5, gam[2560][0], "f32", "plain"),
        ("rms_c512_n33_f32", 33, 512, 1e-6, gam[512][0], "f32", "plain"),
        ("rms_c1024_n16_randgamma_f32", 16, 1024, 1e-5, None, "f32", "plain"),
    ]
    for name, n, c, eps, gname, mode, kind in specs:
        rng = rng_for(fx.seed, "norm/" + name)
        x = randn(rng, (n, c), 1.0) * uniform(rng, (n, 1), 0.2, 4.0)
        if kind == "outlier":
            x[:, rng.integers(0, c, 5)] *= 400.0
        elif kind == "tiny":
            x[::2] *= 1e-4
            x[1::4] = x[1::4] * 1e-9
        elif kind == "zero":
            x[0] = 0.0
            x[3] = 0.0
        gamma = ck.get(gname).float() if gname else randn(rng, (c,), 0.5) + 1.0
        x = in_mode(x, mode)
        gamma = in_mode(gamma, mode)
        y, y32 = rms_ref(ref, x, gamma, eps, mode)
        fx.add("norm", name, {"N": n, "C": c}, mode,
               {"x": x, "gamma": gamma, "eps": scal(eps)},
               {"y": y, "y_f32": y32},
               note="RMSNorm plain gamma (M:51-66); y is the model-dtype output, y_f32 keep_fp32")


def gen_head(fx: Fixtures, ck: Ckpt):
    ref = fx.ref
    final_gamma = ck.get("model.norm.weight").float()
    cases = [(f"mean_n4_c2560_N{n}", n, 4, 2560, "bf16") for n in (1, 3, 64, 129)]
    cases += [("mean_n3_c1024_N16", 16, 3, 1024, "bf16"), ("mean_n4_c2560_N33_f32", 33, 4, 2560, "f32"),
              ("mean_n3_c1024_N16_f32", 16, 3, 1024, "f32")]
    for name, n, ns, c, mode in cases:
        rng = rng_for(fx.seed, "head/" + name)
        st = randn(rng, (n, ns, c), 1.0) * uniform(rng, (n, ns, 1), 0.3, 3.0)
        st[:, :, rng.integers(0, c, 3)] *= 30.0
        st = in_mode(st, mode)
        gamma = in_mode(final_gamma if c == 2560 else randn(rng, (c,), 0.3) + 1.0, mode)
        # M:1173  hidden = norm(stream.float().mean(2).to(dtype))
        mean = st.float().mean(1).to(to_mode(st, mode).dtype)
        h, _ = rms_ref(ref, mean.float(), gamma, 1e-5, mode)
        fx.add("head", name, {"N": n, "n": ns, "C": c}, mode,
               {"stream": st, "gamma": gamma, "eps": scal(1e-5)},
               {"mean": mean.float(), "h": h},
               note="final stream mean (fp32 accumulate, one cast) then RMSNorm (M:1173)")
    # embedding broadcast (M:1160): real embedding rows replicated into every stream
    for name, n, ns, c, mode in [("bcast_n4_c2560_N1", 1, 4, 2560, "bf16"), ("bcast_n4_c2560_N64", 64, 4, 2560, "bf16"),
                                 ("bcast_n4_c2560_N129", 129, 4, 2560, "bf16"),
                                 ("bcast_n3_c1024_N16_f32", 16, 3, 1024, "f32")]:
        rng = rng_for(fx.seed, "head/" + name)
        if c == 2560:
            ids = rng.integers(0, 259840, size=n)
            emb = ck.rows("model.embed_tokens.weight", ids).float()
        else:
            emb = randn(rng, (n, c), 0.02)
        emb = in_mode(emb, mode)
        bcast = to_mode(emb, mode).unsqueeze(1).expand(-1, ns, -1).contiguous().float()
        fx.add("head", name, {"N": n, "n": ns, "C": c}, mode,
               {"emb": emb}, {"bcast": bcast},
               note="X[b,t,s,c]=E[token][c] for every stream s (M:1160), no scale/BOS")


# ----------------------------------------------------------------------------
# mHC
# ----------------------------------------------------------------------------
def mhc_reference(ref: Ref, mode, n, c, x, phi, alpha_pre, alpha_post, alpha_z, eta, b_pre, b_post,
                  b_z, gamma, eps_norm, branch_out, post_factor=2.0, selector_eps=1e-5):
    """Pinned RaonMHC executed on fixture inputs.  All args are torch tensors (fixture values)."""
    mm = ref.mod
    cfg = ref.cfg(raon_mhc_streams=n, hidden_size=c, raon_mhc_post_factor=post_factor,
                  raon_mhc_selector_eps=selector_eps)
    m = mm.RaonMHC(cfg)
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    with torch.no_grad():
        m.phi_pre.data = phi[:n].clone()
        m.phi_post.data = phi[n : 2 * n].clone()
        m.res.phi_z.data = phi[2 * n :].clone()
        m.alpha_pre.data = alpha_pre.reshape(()).clone()
        m.alpha_post.data = alpha_post.reshape(()).clone()
        m.res.alpha_z.data = alpha_z.reshape(()).clone()
        m.b_pre.data = b_pre.clone()
        m.b_post.data = b_post.clone()
        m.res.b_z.data = b_z.clone()
        m = m.to(dt)
        m.res.eta.data = eta.reshape(()).to(torch.float32).clone()  # FP32, never rounded
        rows = x.shape[0]
        xs = x.to(dt).reshape(1, rows, n, c)
        h_pre, h_post, h_res = m.route(xs)
        # selector z exactly as M:169-173 (re-derived; checked against the numpy chart below)
        flat = xs.reshape(rows, n * c)
        weight = torch.cat([m.phi_pre, m.phi_post, m.res.phi_z], dim=0)
        inv = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + m.eps)
        z = F.linear(flat, weight.to(flat.dtype)).float() * inv
        sf = xs.reshape(rows, n, c).float()
        agg = mm.RaonMHC.aggregate(sf, h_pre, dt)
        norm = mm.Raon1MiniRMSNorm(c, eps_norm)
        norm.weight.data = gamma.to(torch.float32).clone()
        branch_in = norm(agg)
        x_out = mm.RaonMHC.mix(sf, branch_out.to(dt), h_post, h_res, dt)
    return {
        "z": z.float(), "pre": h_pre.float(), "post": h_post.float(), "res": h_res.float(),
        "agg": agg.float(), "branch_in": branch_in.float(), "x_out": x_out.float(),
    }, (m, sf, branch_out.to(dt), h_post, h_res, dt)


def check_mhc_against_numpy(out, n, alphas, eta, b_pre, b_post, b_z, mode, tag, want_rho_pos=None):
    pre, post, res, e, rho = np_mhc_controls(out["z"].numpy(), alphas[0], alphas[1], alphas[2], eta,
                                             b_pre.numpy(), b_post.numpy(), b_z.numpy(), n)
    tol = 1e-4 if mode == "bf16" else 2e-5
    for nm, a, b in (("pre", out["pre"].numpy(), pre), ("post", out["post"].numpy(), post),
                     ("res", out["res"].numpy(), res)):
        err = np.abs(a.astype(np.float64) - b).max() / max(1.0, np.abs(b).max())
        if err > tol:
            raise RuntimeError(f"mhc {tag}: pinned module vs float64 numpy chart mismatch on {nm}: {err:.3e}")
    # bistochastic invariant (necessary, not sufficient): rows and columns of Hres sum to 1
    for ax in (1, 2):
        if np.abs(res.sum(ax) - 1.0).max() > 1e-9:
            raise RuntimeError(f"mhc {tag}: numpy chart is not bistochastic")
    if want_rho_pos is True and not (rho > 0).any():
        raise RuntimeError(f"mhc {tag}: expected rho>0 for at least one row, got max {rho.max()}")
    return float(rho.max())


def mhc_param_set(kind, rng, n, c, ck, layer=1, site="self_attn_raon_mhc"):
    """Return dict of site parameters (float32 torch tensors, un-rounded)."""
    zd = 2 * n + (n - 1) ** 2
    if kind == "real":
        p = f"model.layers.{layer}.{site}."
        phi = torch.cat([ck.get(p + "phi_pre"), ck.get(p + "phi_post"), ck.get(p + "res.phi_z")], 0).float()
        d = dict(phi=phi, alpha_pre=ck.get(p + "alpha_pre").float().reshape(1),
                 alpha_post=ck.get(p + "alpha_post").float().reshape(1),
                 alpha_z=ck.get(p + "res.alpha_z").float().reshape(1), eta=ck.get(p + "res.eta").float().reshape(1),
                 b_pre=ck.get(p + "b_pre").float(), b_post=ck.get(p + "b_post").float(),
                 b_z=ck.get(p + "res.b_z").float())
        ln = "input_layernorm" if site == "self_attn_raon_mhc" else "post_attention_layernorm"
        d["gamma"] = ck.get(f"model.layers.{layer}.{ln}.weight").float()
        return d
    return dict(
        phi=randn(rng, (zd, n * c), 0.19),
        alpha_pre=uniform(rng, (1,), 0.03, 0.12), alpha_post=uniform(rng, (1,), 0.03, 0.12),
        alpha_z=uniform(rng, (1,), 0.4, 2.5), eta=uniform(rng, (1,), -2.0, 2.0),
        b_pre=randn(rng, (n,), 1.5), b_post=randn(rng, (n,), 0.4), b_z=randn(rng, (zd - 2 * n,), 0.6),
        gamma=randn(rng, (c,), 0.15) + 1.0,
    )


def gen_mhc(fx: Fixtures, ck: Ckpt, negatives: bool):
    ref = fx.ref
    specs = []
    for n in (1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512):
        specs.append((f"site_n4_N{n}", n, 4, 2560, "bf16", "rand", {}))
    specs += [
        ("site_n4_N33_real_l1_attn", 33, 4, 2560, "bf16", "real", dict(layer=1, site="self_attn_raon_mhc")),
        ("site_n4_N33_real_l3_mlp", 33, 4, 2560, "bf16", "real", dict(layer=3, site="mlp_raon_mhc")),
        ("site_n4_N33_real_l24_mlp", 33, 4, 2560, "bf16", "real", dict(layer=24, site="mlp_raon_mhc")),
        ("site_n4_N64_f32", 64, 4, 2560, "f32", "rand", {}),
        ("site_n3_C1024_N16", 16, 3, 1024, "bf16", "rand", {}),
        ("site_n3_C1024_N16_f32", 16, 3, 1024, "f32", "rand", {}),
        ("site_n5_C512_N16", 16, 5, 512, "bf16", "rand", {}),
        ("site_n5_C512_N16_f32", 16, 5, 512, "f32", "rand", {}),
    ]
    for eta in (-20.0, -3.0, 0.0, 3.0, 20.0):
        specs.append((f"chart_eta_{'m' if eta < 0 else 'p'}{abs(int(eta))}_N8", 8, 4, 2560, "bf16", "rand", dict(eta=eta, alpha_z=1.5)))
    specs += [
        ("chart_rho_pos_N8", 8, 4, 2560, "bf16", "rand", dict(eta=1.0, alpha_z=40.0, want_rho=True)),
        ("chart_rho_pos_N8_f32", 8, 4, 2560, "f32", "rand", dict(eta=1.0, alpha_z=40.0, want_rho=True)),
        ("chart_nonsym_z_N8", 8, 4, 2560, "bf16", "rand", dict(eta=2.0, alpha_z=3.0, nonsym=True)),
        ("sat_prepost_N8", 8, 4, 2560, "bf16", "rand", dict(eta=0.5, alpha_pre=50.0, alpha_post=-50.0,
                                                          b_pre=[30.0, -30.0, 30.0, -30.0],
                                                          b_post=[-30.0, 30.0, 30.0, -30.0])),
    ]
    neg_names = {"site_n4_N4", "site_n4_N33", "site_n4_N64", "chart_eta_p3_N8", "chart_nonsym_z_N8",
                 "site_n4_N33_real_l3_mlp"}
    rho_max = {}
    for name, n_rows, n, c, mode, kind, kw in specs:
        rng = rng_for(fx.seed, "mhc/" + name)
        ps = mhc_param_set(kind, rng, n, c, ck, **{k: kw[k] for k in ("layer", "site") if k in kw})
        if "eta" in kw:
            ps["eta"] = scal(kw["eta"])
        if "alpha_z" in kw:
            ps["alpha_z"] = scal(kw["alpha_z"])
        if "alpha_pre" in kw:
            ps["alpha_pre"] = scal(kw["alpha_pre"])
        if "alpha_post" in kw:
            ps["alpha_post"] = scal(kw["alpha_post"])
        if "b_pre" in kw:
            ps["b_pre"] = torch.tensor(kw["b_pre"])
            ps["b_post"] = torch.tensor(kw["b_post"])
        if kw.get("nonsym"):
            zz = torch.zeros((n - 1) ** 2)
            zz[0::3] = 1.5
            zz[1::3] = -0.5
            ps["b_z"] = zz * 3.0
        x = randn(rng, (n_rows, n, c), 1.0) * uniform(rng, (n_rows, n, 1), 0.4, 2.5)
        x[:, :, rng.integers(0, c, 6)] *= 25.0
        branch = randn(rng, (n_rows, c), 3.0)
        # fixture values at model-dtype boundaries
        for k in ("phi", "alpha_pre", "alpha_post", "alpha_z", "b_pre", "b_post", "b_z", "gamma"):
            ps[k] = in_mode(ps[k], mode)
        x, branch = in_mode(x, mode), in_mode(branch, mode)
        eta = ps["eta"].float()
        out, ctx = mhc_reference(ref, mode, n, c, x, ps["phi"], ps["alpha_pre"], ps["alpha_post"], ps["alpha_z"],
                                 eta, ps["b_pre"], ps["b_post"], ps["b_z"], ps["gamma"], 1e-5, branch)
        rmax = check_mhc_against_numpy(out, n, (float(ps["alpha_pre"]), float(ps["alpha_post"]),
                                                float(ps["alpha_z"])), float(eta), ps["b_pre"], ps["b_post"],
                                       ps["b_z"], mode, name, want_rho_pos=True if kw.get("want_rho") else None)
        rho_max[name] = rmax
        inputs = {"x": x, "phi": ps["phi"], "alpha_pre": ps["alpha_pre"].reshape(1),
                  "alpha_post": ps["alpha_post"].reshape(1), "alpha_z": ps["alpha_z"].reshape(1),
                  "eta": eta.reshape(1), "b_pre": ps["b_pre"], "b_post": ps["b_post"], "b_z": ps["b_z"],
                  "gamma": ps["gamma"], "eps_norm": scal(1e-5), "selector_eps": scal(1e-5),
                  "post_factor": scal(2.0), "branch_out": branch}
        outputs = {k: out[k] for k in ("z", "pre", "post", "res", "agg", "branch_in", "x_out")}
        note = f"Raon mHC site (M:120-207); rho_max={rmax:.4g}"
        fx.add("mhc", name, {"N": n_rows, "n": n, "C": c, "zdim": 2 * n + (n - 1) ** 2}, mode, inputs, outputs,
               note=note)
        if negatives and name in neg_names:
            # flipped stream orientation: use Hres^T (source/destination swapped) for the residual mix
            m, sf, bo, hpost, hres, dt = ctx
            res_t = hres.transpose(-1, -2).contiguous()
            x_out_neg = ref.mod.RaonMHC.mix(sf, bo, hpost, res_t, dt).float()
            wrong = {k: out[k] for k in outputs}
            wrong["res"] = res_t.float()
            wrong["x_out"] = x_out_neg
            fx.add("mhc", name + "__flipped_orientation", {"N": n_rows, "n": n, "C": c, "zdim": 2 * n + (n - 1) ** 2},
                   mode, inputs, wrong, negative=True, expect_fail_outputs=["res", "x_out"],
                   note="NEGATIVE CONTROL: Hres used as [source,dest] (transposed); a correct module must fail")
    return rho_max


# ----------------------------------------------------------------------------
# RDA
# ----------------------------------------------------------------------------
def np_rda_prepare(raw, b, delta, f, cw, hist, a_log, dt_bias, heads, dim, lower=-5.0, eps=1e-6):
    """Independent float64 version of M:317-357 (post-projection part).  raw/cw/hist: dict q,k,v."""
    n = raw["q"].shape[0]

    def conv(x, w, h):
        full = np.concatenate([np.asarray(h, np.float64), np.asarray(x, np.float64)], 0)
        acc = sum(np.asarray(w, np.float64)[:, j][None, :] * full[j : j + n] for j in range(4))
        return acc * np_sigmoid(acc)

    def l2(x):
        nrm = np.sqrt((x * x).sum(-1, keepdims=True))
        return x / np.maximum(nrm, eps)

    qc, kc, vc = (conv(raw[x], cw[x], hist[x]).reshape(n, heads, dim) for x in "qkv")
    q = l2(qc) * dim ** -0.5
    k = l2(kc)
    u = np.exp(np.asarray(a_log, np.float64))[None, :, None] * (
        np.asarray(f, np.float64).reshape(n, heads, dim) + np.asarray(dt_bias, np.float64).reshape(heads, dim))
    log_alpha = lower * np_sigmoid(u)
    b64, d64 = np.asarray(b, np.float64), np.asarray(delta, np.float64)
    beta = np_sigmoid(b64)
    ratio = np.exp(np_logsigmoid(b64 + d64) - np_logsigmoid(b64))
    vprime = vc * ratio[:, :, None]
    return q, k, vprime, log_alpha, beta


def rda_reference(ref: Ref, mode, heads, dim, inp, s0, hist, weights):
    """Run the pinned Raon1MiniRDA forward on already-projected inputs.

    inp: q_raw,k_raw,v_raw,z,f [N,P], b,delta [N,H].  hist: dict q,k,v [3,P] oldest->newest.
    weights: conv q,k,v [P,4], a_log [H], dt_bias [H*K], gamma [K].
    """
    mm = ref.mod
    p = heads * dim
    n = inp["q_raw"].shape[0]
    htot = 5 * p + 2 * heads
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    cfg = ref.cfg(num_attention_heads=heads, head_dim=dim, hidden_size=htot, rda_config=ref.rda_config(heads, dim))
    with torch.device("meta"):
        rda = mm.Raon1MiniRDA(cfg)
    rda.q_proj, rda.k_proj, rda.v_proj = Select(0, p), Select(p, p), Select(2 * p, p)
    rda.g_proj, rda.f_a_proj, rda.f_b_proj = Select(3 * p, p), Select(4 * p, p), nn.Identity()
    rda.b_proj, rda.rda_w_proj, rda.o_proj = Select(5 * p, heads), Select(5 * p + heads, heads), nn.Identity()
    for nm, conv in (("q", rda.q_conv1d), ("k", rda.k_conv1d), ("v", rda.v_conv1d)):
        conv.weight = nn.Parameter(weights["conv_" + nm].to(dt).clone(), requires_grad=False)
    rda.A_log = nn.Parameter(weights["a_log"].float().clone(), requires_grad=False)
    rda.dt_bias = nn.Parameter(weights["dt_bias"].float().reshape(-1).clone(), requires_grad=False)
    rda.o_norm.weight = nn.Parameter(weights["gamma"].to(dt).clone(), requires_grad=False)
    for nm, prm in rda.named_parameters():
        assert not prm.is_meta, f"meta param left: {nm}"
    cap = {}

    def core(q, k, v, log_alpha, beta, state):
        cap.update(q=q, k=k, v=v, log_alpha=log_alpha, beta=beta, state=state)
        out, fs = mm.torch_recurrent_rda(q, k, v, log_alpha, beta, state)
        cap["y"] = out
        return out, fs

    rda._core = core
    hidden = torch.cat([inp["q_raw"], inp["k_raw"], inp["v_raw"], inp["z"], inp["f"], inp["b"], inp["delta"]],
                       dim=-1).to(dt).reshape(1, n, htot)
    cache = mm.Raon1MiniDynamicCache(cfg)
    cache.conv_states[0] = torch.cat([hist[x].t().reshape(1, p, 3) for x in "qkv"], dim=1).to(dt)
    cache.recurrent_states[0] = s0.reshape(1, heads, dim, dim).float().clone()
    with torch.no_grad():
        gated = rda(hidden, cache, 0, None)
    cs = cache.conv_states[0].reshape(3, p, 3)
    out = {
        "q": cap["q"].reshape(n, p).float(), "k": cap["k"].reshape(n, p).float(),
        "vprime": cap["v"].reshape(n, p).float(), "log_alpha": cap["log_alpha"].reshape(n, p).float(),
        "beta": cap["beta"].reshape(n, heads).float(), "y": cap["y"].reshape(n, p).float(),
        "gated": gated.reshape(n, p).float(), "S_out": cache.recurrent_states[0].reshape(heads, dim, dim).float(),
        "hist_out_q": cs[0].t().float(), "hist_out_k": cs[1].t().float(), "hist_out_v": cs[2].t().float(),
    }
    return out, cap, rda


def rda_read_before_write(cap, rda, z, heads, dim, mode):
    """NEGATIVE CONTROL: read the state after decay but BEFORE the delta write."""
    q, k, v, la, beta = (cap[x].float() for x in ("q", "k", "v", "log_alpha", "beta"))
    s = cap["state"].float().clone()
    n = q.shape[1]
    out = torch.empty(1, n, heads, dim, dtype=torch.float32)
    for t in range(n):
        s = s * torch.exp(la[:, t]).unsqueeze(-2)
        out[:, t] = torch.einsum("bhvk,bhk->bhv", s, q[:, t])  # read first (wrong)
        pred = torch.einsum("bhvk,bhk->bhv", s, k[:, t])
        err = v[:, t] - pred
        s = s + beta[:, t][..., None, None] * torch.einsum("bhv,bhk->bhvk", err, k[:, t])
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    y = out.to(dt)
    normed = rda.o_norm(y.reshape(-1, dim), keep_fp32=True)
    gated = (normed * torch.sigmoid(z.to(dt).float().reshape(-1, dim))).to(dt)
    return y.reshape(n, heads * dim).float(), gated.reshape(n, heads * dim).float()


def rda_weights_real(ck, heads=40, dim=128, layer=1):
    p = f"model.layers.{layer}.self_attn."
    w = {x: ck.get(p + f"{x}_conv1d.weight").float() for x in "qkv"}
    return {"conv_q": w["q"], "conv_k": w["k"], "conv_v": w["v"], "a_log": ck.get(p + "A_log").float(),
            "dt_bias": ck.get(p + "dt_bias").float(), "gamma": ck.get(p + "o_norm.weight").float()}


def rda_inputs(rng, n, heads, dim, mode, variant="plain", ch_scale=None):
    p = heads * dim
    cs = ch_scale if ch_scale is not None else uniform(rng, (1, p), 0.5, 2.0)
    inp = {x + "_raw": randn(rng, (n, p), 1.0) * cs for x in "qkv"}
    inp["z"] = randn(rng, (n, p), 2.0)
    inp["f"] = randn(rng, (n, p), 3.0)
    inp["b"] = randn(rng, (n, heads), 2.0)
    inp["delta"] = randn(rng, (n, heads), 1.5)
    if variant == "tiny":
        inp["q_raw"] = inp["q_raw"] * 1e-8
        inp["k_raw"] = inp["k_raw"] * 1e-8
        inp["q_raw"][::3] = 0.0
        inp["k_raw"][1::4] = 0.0
    elif variant == "extreme":
        q4 = heads // 4
        b = inp["b"].clone()
        d = inp["delta"].clone()
        b[:, :q4] = 80.0
        b[:, q4 : 2 * q4] = -80.0
        b[:, 2 * q4 : 3 * q4] = 0.0
        d[:, 2 * q4 : 3 * q4] = torch.where(torch.arange(n)[:, None] % 2 == 0, 80.0, -80.0)
        b[:, 3 * q4 :] = torch.where(torch.arange(n)[:, None] % 2 == 0, 80.0, -80.0)
        inp["b"], inp["delta"] = b, d
        f = inp["f"].clone()
        f[:, : dim // 4] = 80.0
        f[:, dim // 4 : dim // 2] = -80.0
        f[:, p // 2 : p // 2 + dim // 4] = 80.0
        inp["f"] = f
    return {k: in_mode(v, mode) for k, v in inp.items()}


def rda_hist(rng, heads, dim, mode, scale=1.0):
    p = heads * dim
    return {x: in_mode(randn(rng, (3, p), scale), mode) for x in "qkv"}


def slice_inp(inp, a, b):
    return {k: v[a:b].clone() for k, v in inp.items()}


def check_rda_independent(name, inp, hist, weights, heads, dim, s0, out, cap):
    """float64 NumPy: preparation + recurrence vs the pinned module (bf16-boundary tolerance)."""
    raw = {x: inp[x + "_raw"].numpy() for x in "qkv"}
    cw = {"q": weights["conv_q"].numpy(), "k": weights["conv_k"].numpy(), "v": weights["conv_v"].numpy()}
    hs = {x: hist[x].numpy() for x in "qkv"}
    q, k, vp, la, beta = np_rda_prepare(raw, inp["b"].numpy(), inp["delta"].numpy(), inp["f"].numpy(), cw, hs,
                                        weights["a_log"].numpy(), weights["dt_bias"].numpy(), heads, dim)
    n = q.shape[0]
    for nm, a, b, tol in (("q", out["q"], q, 2e-2), ("k", out["k"], k, 2e-2), ("vprime", out["vprime"], vp, 2e-2),
                          ("log_alpha", out["log_alpha"], la, 1e-5), ("beta", out["beta"], beta, 1e-2)):
        b = b.reshape(n, -1)
        e = rel_l2(a, b)
        if e > tol:
            raise RuntimeError(f"rda {name}: pinned vs float64 numpy prepare mismatch on {nm}: rel_l2={e:.3e}")
    # recurrence on the module's own prepared (bf16-boundary) q/k/v so only the recurrence is compared
    qm, km, vm = (cap[x].float().reshape(n, heads, dim).numpy() for x in ("q", "k", "v"))
    lam = cap["log_alpha"].reshape(n, heads, dim).numpy()
    bm = cap["beta"].float().reshape(n, heads).numpy()
    y64, s64 = np_rda_recurrence(qm, km, vm, lam, bm, s0.reshape(heads, dim, dim).numpy())
    ey = rel_l2(out["y"], torch.from_numpy(y64).reshape(n, -1))
    es = rel_l2(out["S_out"], torch.from_numpy(s64))
    if ey > 1e-2 or es > 1e-4:
        raise RuntimeError(f"rda {name}: recurrence vs float64 numpy: y rel_l2={ey:.3e} S rel_l2={es:.3e}")
    return ey, es


def gen_rda(fx: Fixtures, ck: Ckpt, negatives: bool):
    ref = fx.ref
    heads, dim = 40, 128
    p = heads * dim
    wts = rda_weights_real(ck)
    wts_bf = {k: bfr(v) if k not in ("a_log", "dt_bias") else v for k, v in wts.items()}
    stats = {}

    def emit(name, n, mode, inp, s0, hist, weights, *, full=True, extra_in=None, extra_out=None,
             neg=False, note=None, check=True, dims=None):
        wr = weights
        out, cap, rda = rda_reference(ref, mode, heads_c, dim_c, inp, s0, hist, wr)
        if check and n <= 513:
            stats[name] = check_rda_independent(name, inp, hist, wr, heads_c, dim_c, s0, out, cap)
        inputs = {"q_raw": inp["q_raw"], "k_raw": inp["k_raw"], "v_raw": inp["v_raw"], "z": inp["z"],
                  "b": inp["b"], "delta": inp["delta"], "f": inp["f"],
                  "conv_w_q": wr["conv_q"], "conv_w_k": wr["conv_k"], "conv_w_v": wr["conv_v"],
                  "hist_q": hist["q"], "hist_k": hist["k"], "hist_v": hist["v"], "A_log": wr["a_log"],
                  "dt_bias": wr["dt_bias"].reshape(heads_c, dim_c), "norm_gamma": wr["gamma"], "S0": s0}
        outputs = {k: out[k] for k in ("q", "k", "vprime", "log_alpha", "beta", "y", "gated", "S_out",
                                       "hist_out_q", "hist_out_k", "hist_out_v")}
        if not full:
            for k in ("q", "k", "vprime", "log_alpha", "gated"):
                del outputs[k]
            for k in ("z",):
                del inputs[k]
        if extra_in:
            inputs.update(extra_in)
        if extra_out:
            outputs.update(extra_out)
        fx.add("rda", name, dims or {"N": n, "H": heads_c, "K": dim_c, "P": heads_c * dim_c}, mode, inputs,
               outputs, note=note)
        return out, cap, rda, inputs, outputs

    heads_c, dim_c = heads, dim
    # --- N matrix (nonzero initial state and history) -------------------------------------------
    for n in (1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512):
        rng = rng_for(fx.seed, f"rda/n{n}")
        inp = rda_inputs(rng, n, heads, dim, "bf16")
        hist = rda_hist(rng, heads, dim, "bf16")
        s0 = randn(rng, (heads, dim, dim), 0.1)
        out, cap, rda, inputs, outputs = emit(f"n{n}", n, "bf16", inp, s0, hist, wts_bf,
                                              note="nonzero S0/history; PLAN 6.2 N matrix")
        if negatives and n in (4, 33, 64, 512):
            yw, gw = rda_read_before_write(cap, rda, inp["z"], heads, dim, "bf16")
            wrong = dict(outputs)
            wrong["y"], wrong["gated"] = yw, gw
            fx.add("rda", f"n{n}__read_before_write", {"N": n, "H": heads, "K": dim, "P": p}, "bf16", inputs, wrong,
                   negative=True, expect_fail_outputs=["y", "gated"],
                   note="NEGATIVE CONTROL: y read after decay before the delta write; a correct module must fail")
    # --- zero state / f32 / tiny q,k / extreme gates ---------------------------------------------
    for name, n, mode, variant, zero in (("n1_zero_state", 1, "bf16", "plain", True),
                                         ("n64_zero_state", 64, "bf16", "plain", True),
                                         ("n64_f32", 64, "f32", "plain", False),
                                         ("n33_tiny_qk", 33, "bf16", "tiny", False),
                                         ("n64_tiny_qk", 64, "bf16", "tiny", False),
                                         ("n64_extreme_gates", 64, "bf16", "extreme", False),
                                         ("n129_extreme_gates", 129, "bf16", "extreme", False)):
        rng = rng_for(fx.seed, "rda/" + name)
        inp = rda_inputs(rng, n, heads, dim, mode, variant)
        hist = ({x: torch.zeros(3, p) for x in "qkv"} if zero else rda_hist(rng, heads, dim, mode))
        s0 = torch.zeros(heads, dim, dim) if zero else randn(rng, (heads, dim, dim), 0.1)
        w = wts_bf if mode == "bf16" else wts
        emit(name, n, mode, inp, s0, hist, w, note=f"variant={variant}")
    # --- split-call histories: a = first calls, b continues from a's state ------------------------
    for na, nb in ((1, 64), (3, 64), (63, 66), (127, 2), (512, 1)):
        name = f"split_{na}_{nb}"
        rng = rng_for(fx.seed, "rda/" + name)
        n = na + nb
        inp = rda_inputs(rng, n, heads, dim, "bf16")
        hist = rda_hist(rng, heads, dim, "bf16")
        s0 = randn(rng, (heads, dim, dim), 0.1)
        full_out, _, _ = rda_reference(ref, "bf16", heads, dim, inp, s0, hist, wts_bf)
        oa, _, _, _, _ = emit(name + "_a", na, "bf16", slice_inp(inp, 0, na), s0, hist, wts_bf,
                              note=f"first call of a split ({na}+{nb}); case _b continues from S_out/hist_out")
        hist_b = {x: oa["hist_out_" + x] for x in "qkv"}
        ob, _, _, _, _ = emit(name + "_b", nb, "bf16", slice_inp(inp, na, n), oa["S_out"], hist_b, wts_bf,
                              note="second call; S0/hist_* = _a outputs")
        # chunk invariance of the reference itself: elementwise torch ops may differ by one fp32 ulp
        # between a short and a long call (vector vs tail loop), so the bound is ulp-level, not bitwise
        for k in ("q", "k", "vprime", "y", "gated"):
            e = rel_l2(ob[k], full_out[k][na:])
            if e > 2e-3:
                raise RuntimeError(f"rda {name}: split-call output {k} differs from single call: rel_l2={e}")
        e = max_abs(ob["log_alpha"], full_out["log_alpha"][na:])
        if e > 5e-7:
            raise RuntimeError(f"rda {name}: split-call log_alpha differs from single call by {e}")
        for k in ("S_out", "hist_out_q", "hist_out_k", "hist_out_v"):
            if rel_l2(ob[k], full_out[k]) > 1e-5:
                raise RuntimeError(f"rda {name}: split final {k} differs from single call")
    # --- long trajectory ---------------------------------------------------------------------------
    rng = rng_for(fx.seed, "rda/n8192")
    inp = rda_inputs(rng, 8192, heads, dim, "bf16")
    hist = rda_hist(rng, heads, dim, "bf16")
    s0 = randn(rng, (heads, dim, dim), 0.1)
    t0 = time.time()
    out, cap, rda, _, _ = emit("n8192", 8192, "bf16", inp, s0, hist, wts_bf, full=False, check=False,
                               note="long trajectory: outputs y,beta,S_out,hist_out only (size)")
    n = 8192
    qm, km, vm = (cap[x].float().reshape(n, heads, dim).numpy() for x in ("q", "k", "v"))
    y64, s64 = np_rda_recurrence(qm, km, vm, cap["log_alpha"].reshape(n, heads, dim).numpy(),
                                 cap["beta"].float().reshape(n, heads).numpy(), s0.numpy())
    es, ey = rel_l2(out["S_out"], torch.from_numpy(s64)), rel_l2(out["y"], torch.from_numpy(y64).reshape(n, -1))
    stats["n8192"] = (ey, es)
    if es > 5e-4 or ey > 1e-2:
        raise RuntimeError(f"rda n8192 recurrence vs float64: S rel_l2 {es:.3e} y rel_l2 {ey:.3e}")
    log(f"rda n8192 done in {time.time() - t0:.0f}s (S rel_l2 vs f64 {es:.2e})")
    # --- scalar-equivalent (per-channel decay equal within each head) vs plain scalar-gate rule ---
    heads_c, dim_c = 8, 128
    ph = heads_c * dim_c
    rng = rng_for(fx.seed, "rda/scalar_equiv")
    n = 64
    w8 = {"conv_q": wts_bf["conv_q"][:ph].clone(), "conv_k": wts_bf["conv_k"][:ph].clone(),
          "conv_v": wts_bf["conv_v"][:ph].clone(), "a_log": wts["a_log"][:heads_c].clone(),
          "gamma": wts_bf["gamma"].clone()}
    dh = wts["dt_bias"].reshape(heads, dim)[:heads_c].mean(1)
    w8["dt_bias"] = dh[:, None].expand(heads_c, dim).contiguous()
    inp = rda_inputs(rng, n, heads_c, dim_c, "bf16")
    fh = randn(rng, (n, heads_c, 1), 3.0)
    inp["f"] = in_mode(fh.expand(n, heads_c, dim_c).reshape(n, ph).contiguous(), "bf16")
    hist = rda_hist(rng, heads_c, dim_c, "bf16")
    s0 = randn(rng, (heads_c, dim_c, dim_c), 0.1)
    out, cap, rda = rda_reference(ref, "bf16", heads_c, dim_c, inp, s0, hist, w8)
    la = cap["log_alpha"].reshape(n, heads_c, dim_c)
    if (la - la[..., :1]).abs().max() != 0:
        raise RuntimeError("scalar_equiv: decay is not equal across channels")
    gq, gk, gv = (cap[x].float().reshape(n, heads_c, dim_c) for x in ("q", "k", "v"))
    gbeta = cap["beta"].float().reshape(n, heads_c)
    y64, s64 = np_rda_scalar_gate(gq.numpy(), gk.numpy(), gv.numpy(), la[..., 0].numpy(), gbeta.numpy(),
                                  s0.numpy())
    gy, gs = torch.from_numpy(y64).float(), torch.from_numpy(s64).float()
    if rel_l2(out["S_out"], gs) > 1e-4 or rel_l2(out["y"].reshape(n, heads_c, dim_c), gy) > 1e-2:
        raise RuntimeError("scalar_equiv: vector-decay reference disagrees with scalar-gate rule")
    stats["scalar_equiv"] = (rel_l2(out["y"].reshape(n, heads_c, dim_c), gy), rel_l2(out["S_out"], gs))
    extra_in = {"gdn_q": gq, "gdn_k": gk, "gdn_v": gv, "gdn_log_decay": la[..., 0].contiguous(),
                "gdn_beta": gbeta, "gdn_S0": s0}
    extra_out = {"gdn_y": gy, "gdn_S_out": gs}
    emit("scalar_equiv", n, "bf16", inp, s0, hist, w8, extra_in=extra_in, extra_out=extra_out, check=True,
         note="H=8,K=V=128; decay equal per channel within a head; gdn_* = plain scalar-gate float32 delta rule "
              "(independent float64 NumPy) on the module-prepared q/k/vprime")
    return stats


# ----------------------------------------------------------------------------
# MLA
# ----------------------------------------------------------------------------
RAON_GEO = dict(H=40, nope=128, tail=64, v=128, latent=512, qlora=576)
H8_GEO = dict(H=8, nope=64, tail=32, v=96, latent=256, qlora=192)


def rope_rotate(x, pos, theta=10000.0):
    """Half-split rotary embedding on the last dim (used ONLY by the negative control)."""
    d = x.shape[-1]
    half = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def mla_module(ref: Ref, g, inp, wts, dt):
    mm = ref.mod
    h, nope, tail, v, lat, ql = (g[k] for k in ("H", "nope", "tail", "v", "latent", "qlora"))
    n = inp["q_a_out"].shape[0]
    htot = ql + lat + tail + h * v
    cfg = ref.cfg(num_attention_heads=h, hidden_size=htot, q_lora_rank=ql, kv_lora_rank=lat,
                  qk_nope_head_dim=nope, qk_rope_head_dim=tail, v_head_dim=v)
    with torch.device("meta"):
        mla = mm.Raon1MiniMLA(cfg)
    mla.q_a_proj = Select(0, ql)
    mla.q_b_proj = Fixed(inp["q_proj"].to(dt).reshape(1, n, h * (nope + tail)))
    mla.kv_a_proj_with_mqa = Select(ql, lat + tail)
    mla.g_proj = Select(ql + lat + tail, h * v)
    mla.o_proj = nn.Identity()
    kvb = nn.Linear(1, 1, bias=False)
    kvb.weight = nn.Parameter(wts["kv_b"].to(dt).clone(), requires_grad=False)
    mla.kv_b_proj = kvb
    mla.q_a_layernorm.weight = nn.Parameter(wts["q_a_norm"].to(dt).clone(), requires_grad=False)
    mla.kv_a_layernorm.weight = nn.Parameter(wts["kv_a_norm"].to(dt).clone(), requires_grad=False)
    for nm, prm in mla.named_parameters():
        assert not prm.is_meta, nm
    hidden = torch.cat([inp["q_a_out"], inp["kv_a_out"], inp["g_out"]], -1).to(dt).reshape(1, n, htot)
    return mla, cfg, hidden


def mla_prefix_kv(mla, g, prefix, dt):
    h, nope, tail, v, lat = (g[k] for k in ("H", "nope", "tail", "v", "latent"))
    p = prefix.shape[0]
    kv = F.linear(prefix[:, :lat].to(dt), mla.kv_b_proj.weight).reshape(1, p, h, nope + v)
    k_nope, value = torch.split(kv, [nope, v], dim=-1)
    key = torch.cat([k_nope, prefix[:, lat:].to(dt).reshape(1, p, 1, tail).expand(1, p, h, tail)], dim=-1)
    return key.transpose(1, 2), value.transpose(1, 2)


def mla_reference(ref: Ref, g, inp, prefix, wts, mode="bf16"):
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    h, nope, tail, v, lat = (g[k] for k in ("H", "nope", "tail", "v", "latent"))
    n, p = inp["q_a_out"].shape[0], prefix.shape[0]
    mla, cfg, hidden = mla_module(ref, g, inp, wts, dt)
    mm = ref.mod
    with torch.no_grad():
        qlat = mla.q_a_layernorm(inp["q_a_out"].to(dt))
        c_new = mla.kv_a_layernorm(inp["kv_a_out"][:, :lat].to(dt))
        new_rows = torch.cat([c_new, inp["kv_a_out"][:, lat:].to(dt)], dim=-1)
        cache_out = torch.cat([prefix.to(dt), new_rows], dim=0)
        kv_exp = F.linear(cache_out[:, :lat], mla.kv_b_proj.weight).reshape(p + n, h, nope + v)

        def run(use_gate):
            mla.use_output_gate = use_gate
            cache = mm.Raon1MiniDynamicCache(cfg)
            if p > 0:
                key, value = mla_prefix_kv(mla, g, prefix, dt)
                cache.update(key, value, 3)
            return mla(hidden, cache, 3, None).reshape(n, h * v)

        attn_y = run(False)
        gated = run(True)
    out = {"qlat": qlat.float(), "cache_out": cache_out.float(), "kv_expanded": kv_exp.float(),
           "attn_y": attn_y.float(), "gated": gated.float()}
    return out, (mla, cfg, dt)


def mla_attention_rope_wrong(g, inp, out, prefix, mla, dt):
    """NEGATIVE CONTROL: rotate the `tail` channels of q (positions p+i) and k (positions j)."""
    h, nope, tail, v, lat = (g[k] for k in ("H", "nope", "tail", "v", "latent"))
    n, p = inp["q_a_out"].shape[0], prefix.shape[0]
    t = p + n
    q = inp["q_proj"].to(dt).reshape(n, h, nope + tail).transpose(0, 1)  # [H,N,192]
    kv = out["kv_expanded"].to(dt)
    k_nope, value = kv[..., :nope], kv[..., nope:]
    kp = out["cache_out"].to(dt)[:, lat:]  # [T,tail]
    pos_q = torch.arange(p, p + n)
    pos_k = torch.arange(t)

    def attend(rotate):
        qt, kt = q[..., nope:], kp
        if rotate:
            qt = rope_rotate(qt, pos_q)
            kt = rope_rotate(kt, pos_k)
        qq = torch.cat([q[..., :nope], qt], -1).unsqueeze(0)
        kk = torch.cat([k_nope.transpose(0, 1), kt.unsqueeze(0).expand(h, t, tail)], -1).unsqueeze(0)
        vv = value.transpose(0, 1).unsqueeze(0)
        scale = (nope + tail) ** -0.5
        if n > 1 and t == n:
            y = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=scale)
        elif n > 1:
            mask = torch.ones(n, t, dtype=torch.bool).tril(t - n)[None, None]
            y = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=scale)
        else:
            y = F.scaled_dot_product_attention(qq, kk, vv, scale=scale)
        return y.transpose(1, 2).reshape(n, h * v)

    plain, wrong = attend(False), attend(True)
    gate = torch.sigmoid(inp["g_out"].to(dt))
    return plain, wrong, (wrong * gate).float()


def check_mla_numpy(name, g, inp, out, prefix):
    h, nope, tail, v = g["H"], g["nope"], g["tail"], g["v"]
    n, p = inp["q_a_out"].shape[0], prefix.shape[0]
    q = inp["q_proj"].numpy().reshape(n, h, nope + tail).transpose(1, 0, 2)
    kv = out["kv_expanded"].numpy().reshape(p + n, h, nope + v)
    tailk = out["cache_out"].numpy()[:, g["latent"] :]
    k = np.concatenate([kv[..., :nope].transpose(1, 0, 2),
                        np.broadcast_to(tailk[None], (h, p + n, tail))], -1)
    val = kv[..., nope:].transpose(1, 0, 2)
    y = np_attention(q, k, val, (nope + tail) ** -0.5, p).transpose(1, 0, 2).reshape(n, h * v)
    e = rel_l2(out["attn_y"], torch.from_numpy(y))
    if e > 1.5e-2:
        raise RuntimeError(f"mla {name}: pinned attention vs float64 numpy rel_l2={e:.3e}")
    return e


def gen_mla(fx: Fixtures, ck: Ckpt, negatives: bool):
    ref = fx.ref
    rw = {"q_a_norm": ck.get("model.layers.3.self_attn.q_a_layernorm.weight").float(),
          "kv_a_norm": ck.get("model.layers.3.self_attn.kv_a_layernorm.weight").float(),
          "kv_b": ck.get("model.layers.3.self_attn.kv_b_proj.weight").float()}
    q_b_real = ck.get("model.layers.3.self_attn.q_b_proj.weight").float()
    rng0 = rng_for(fx.seed, "mla/h8weights")
    g8 = H8_GEO
    hw = {"q_a_norm": bfr(randn(rng0, (g8["qlora"],), 0.1) + 1.0), "kv_a_norm": bfr(randn(rng0, (g8["latent"],), 0.2) + 1.5),
          "kv_b": bfr(randn(rng0, (g8["H"] * (g8["nope"] + g8["v"]), g8["latent"]), 1 / math.sqrt(g8["latent"])))}
    q_b_h8 = bfr(randn(rng0, (g8["H"] * (g8["nope"] + g8["tail"]), g8["qlora"]), 1 / math.sqrt(g8["qlora"])))
    wfiles = {
        "raon": fx.write_weights("mla", "weights_raon.safetensors", {k: bfr(v) for k, v in rw.items()}),
        "h8": fx.write_weights("mla", "weights_h8.safetensors", hw),
    }
    rw = {k: bfr(v) for k, v in rw.items()}
    specs = [(f"n{n}_p{p}", n, p, RAON_GEO, "raon", "plain") for n, p in
             ((1, 0), (1, 511), (1, 512), (2, 511), (3, 509), (31, 1), (33, 129), (64, 0), (65, 63), (129, 0), (512, 0))]
    specs.append(("n1_p511_spread", 1, 511, RAON_GEO, "raon", "spread"))
    specs.append(("n17_p300_spread", 17, 300, RAON_GEO, "raon", "spread"))
    specs += [(f"h8_n{n}_p{p}", n, p, H8_GEO, "h8", "plain") for n, p in ((5, 0), (7, 19), (64, 200), (1, 129))]
    neg = {"n1_p511", "n2_p511", "n33_p129", "n64_p0", "n65_p63"}
    errs = {}
    for name, n, p, g, wkey, variant in specs:
        rng = rng_for(fx.seed, "mla/" + name)
        h, nope, tail, v, lat, ql = (g[k] for k in ("H", "nope", "tail", "v", "latent", "qlora"))
        wts = rw if wkey == "raon" else hw
        qb = q_b_real if wkey == "raon" else q_b_h8
        inp = {"q_a_out": bfr(randn(rng, (n, ql), 2.0) * uniform(rng, (n, 1), 0.5, 2.0)),
               "kv_a_out": bfr(torch.cat([randn(rng, (n, lat), 2.0), randn(rng, (n, tail), 3.0)], -1)),
               "g_out": randn(rng, (n, h * v), 3.0)}
        gi = rng.integers(0, h * v, size=8)
        inp["g_out"][:, gi[:4]] = 80.0
        inp["g_out"][:, gi[4:]] = -80.0
        inp["g_out"] = bfr(inp["g_out"])
        nm = ref.mod.Raon1MiniRMSNorm(ql, 1e-6)
        nm.weight.data = wts["q_a_norm"].clone()
        qlat = nm(inp["q_a_out"].to(torch.bfloat16))
        inp["q_proj"] = F.linear(qlat, qb.to(torch.bfloat16)).float()
        if p > 0:
            raw = randn(rng, (p, lat), 2.0)
            nm2 = ref.mod.Raon1MiniRMSNorm(lat, 1e-6)
            nm2.weight.data = wts["kv_a_norm"].clone()
            c = nm2(raw.to(torch.bfloat16)).float()
            if variant == "spread":
                sel = torch.from_numpy(rng.random(p) < 0.2)
                c[sel] *= 10.0
            prefix = bfr(torch.cat([c, randn(rng, (p, tail), 3.0)], -1))
        else:
            prefix = torch.zeros(0, lat + tail)
        out, (mla, cfg, dt) = mla_reference(ref, g, inp, prefix, wts)
        errs[name] = check_mla_numpy(name, g, inp, out, prefix)
        inputs = dict(inp)
        inputs["cache_prefix"] = prefix
        dims = {"N": n, "P": p, "H": h, "nope": nope, "tail": tail, "v": v, "latent": lat, "qlora": ql}
        outputs = {k: out[k] for k in ("qlat", "cache_out", "kv_expanded", "attn_y", "gated")}
        fx.add("mla", name, dims, "bf16", inputs, outputs, weights=wfiles[wkey],
               note="NoPE MLA; tail channels shared across heads, never rotated; scale (nope+tail)^-0.5; "
                    "mask j<=P+i; gated=bf16(attn_y*bf16(sigmoid(g_out)))")
        if negatives and name in neg:
            plain, wrong, gated_wrong = mla_attention_rope_wrong(g, inp, out, prefix, mla, dt)
            if max_abs(plain, out["attn_y"]) != 0.0:
                raise RuntimeError(f"mla {name}: negative-control attention (no rope) differs from the pinned module")
            if rel_l2(wrong, out["attn_y"]) < 1e-2:
                raise RuntimeError(f"mla {name}: rope negative control too close to the true reference")
            wo = dict(outputs)
            wo["attn_y"], wo["gated"] = wrong.float(), gated_wrong
            fx.add("mla", name + "__rope_on_tail", dims, "bf16", inputs, wo, weights="../../mla/" + wfiles[wkey],
                   negative=True, expect_fail_outputs=["attn_y", "gated"],
                   note="NEGATIVE CONTROL: rotary embedding applied to the 64 tail channels; a correct module must fail")
    return errs


# ----------------------------------------------------------------------------
# MoE
# ----------------------------------------------------------------------------
def load_layer_experts(ck: Ckpt, ref: Ref, layer: int, experts=range(128)):
    """Dequantize routed experts exactly per M:496-511 (pinned function) to BF16, cross-checking
    the first matrices against the independent NumPy decoder."""
    sd = {}
    for e in experts:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.moe.experts.{e}.{proj}"
            w, s = ck.get(base + ".weight"), ck.get(base + ".weight_scale_inv")
            if s.dtype != torch.uint8:
                s = s.view(torch.uint8)
            d = ref.mod.dequantize_mxfp8(w, s, torch.bfloat16)
            if e in (0, 127) and proj == "gate_proj":
                indep = np_mxfp8_dequant(w, s)
                if not np.array_equal(indep.astype(np.float32), d.float().numpy()):
                    raise RuntimeError(f"layer {layer} expert {e} {proj}: pinned dequant != independent decoder")
            sd[f"experts.{e}.{proj}.weight"] = d
    return sd


def moe_module(ref: Ref, cfgkw, sd, dt, shared):
    mm = ref.mod
    cfg = ref.cfg(**cfgkw)
    with torch.device("meta"):
        m = mm.Raon1MiniMoE(cfg)
    full = {}
    for k, v in sd.items():
        full[k] = v.to(dt) if k != "gate.e_score_correction_bias" else v.float()
    if not shared:
        m.shared_experts = None
        full = {k: v for k, v in full.items() if not k.startswith("shared_experts.")}
    m.load_state_dict(full, strict=True, assign=True)
    return m, cfg


def moe_run(m, x, dt):
    n, c = x.shape
    flat = x.to(dt)
    with torch.no_grad():
        top, probs = m.gate(flat)
        router_logits = F.linear(flat.float(), m.gate.weight.float())
        scores = torch.sigmoid(router_logits)
        out = m(flat.reshape(1, n, c)).reshape(n, c)
        sh = m.shared_experts(flat) if m.shared_experts is not None else torch.zeros(n, c, dtype=dt)
        saved = m.shared_experts
        m.shared_experts = None
        routed = m(flat.reshape(1, n, c)).reshape(n, c)
        m.shared_experts = saved
    return {"scores": scores.float(), "topk_idx": top.float(), "topk_w": probs.float(), "routed": routed.float(),
            "shared": sh.float(), "out": out.float()}, top


_PINV = {}


def route_x(rng, n, c, router_w, targets, coefs, base_scale, off_logit=-2.0):
    """x rows = W^+ l + noise, where l = off_logit for every expert and coefs[j] on targets[i][j]:
    the router logits are set exactly (router rows are linearly independent), so the designed
    selection has a controlled score margin whatever the row correlations are."""
    key = id(router_w)
    if key not in _PINV:
        w = router_w.double()
        _PINV[key] = (w.t() @ torch.linalg.inv(w @ w.t())).float()
    pinv = _PINV[key]
    logits = torch.full((n, router_w.shape[0]), off_logit, dtype=torch.float32)
    for i in range(n):
        for j, e in enumerate(targets[i]):
            logits[i, e] = float(coefs[j])
    return logits @ pinv.t() + randn(rng, (n, c), base_scale)


def np_moe_generic(x, router_w, experts, shared, params, e_bias=None):
    """Independent float64 MoE for non-Raon policies (softmax router / SwiGLU) and as a cross-check
    of the SiTU/sigmoid route.  experts: dict e -> (gate [I,C], up [I,C], down [C,I]) float64."""
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    logits = x @ np.asarray(router_w, np.float64).T
    if params["router"] == "softmax":
        z = logits - logits.max(1, keepdims=True)
        scores = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        sel = scores
    else:
        scores = np_sigmoid(logits)
        sel = scores + (0.0 if e_bias is None else np.asarray(e_bias, np.float64))
    kk = params["top_k"]
    top = np.argsort(-sel, axis=1, kind="stable")[:, :kk]
    chosen = np.take_along_axis(scores, top, 1)
    w = chosen / (chosen.sum(1, keepdims=True) + (0.0 if params["router"] == "softmax" else 1e-20))

    def act(g, u):
        if params["activation"] == "situ":
            return np_situ_glu(g, u, params["gate_beta"], params["value_beta"])
        return g * np_sigmoid(g) * u

    routed = np.zeros_like(x)
    for i in range(n):
        for e in sorted(top[i]):
            j = int(np.where(top[i] == e)[0][0])
            gw, uw, dw = experts[int(e)]
            routed[i] += w[i, j] * (dw @ act(gw @ x[i], uw @ x[i]))
    sh = np.zeros_like(x)
    if shared is not None:
        gw, uw, dw = shared
        sh = (act(x @ gw.T, x @ uw.T)) @ dw.T
    return {"scores": scores, "topk_idx": top.astype(np.float64), "topk_w": w, "routed": routed, "shared": sh,
            "out": routed + sh}


def gen_moe(fx: Fixtures, ck: Ckpt, negatives: bool):
    ref = fx.ref
    layer = 1
    log("moe: dequantizing layer-1 experts (MXFP8 -> BF16, M:496-511)")
    t0 = time.time()
    sd = load_layer_experts(ck, ref, layer)
    base = f"model.layers.{layer}.moe."
    router_w = ck.get(base + "gate.weight")  # bf16
    real_bias = ck.get(base + "gate.e_score_correction_bias").float()
    for k in ("gate_proj", "up_proj", "down_proj"):
        sd[f"shared_experts.{k}.weight"] = ck.get(base + f"shared_experts.{k}.weight")
    wsave = {f"moe.{k}": v for k, v in sd.items()}
    wfile = fx.write_weights("moe", "weights_l1_real.safetensors", wsave, bf16=True)
    log(f"moe: real weights file written ({time.time() - t0:.0f}s)")
    sd["gate.weight"] = router_w
    sd["gate.e_score_correction_bias"] = real_bias
    cfgkw = dict(hidden_size=2560, num_experts=128, num_experts_per_token=6, moe_intermediate_size=1088)
    m, cfg = moe_module(ref, cfgkw, sd, torch.bfloat16, True)
    rw32 = router_w.float()
    rng = rng_for(fx.seed, "moe/targets")
    cover_targets = [[(i * 6 + j) % 128 for j in range(6)] for i in range(512)]
    skew_c, uneq_c, e0_c = [7.0, 6.0, 5.5, 5.0, 4.5, 4.0], [0.5, 1.0, 1.8, 2.8, 4.0, 6.0], [4.0, 4.0, 3.5, 3.5, 3.0, 3.0]
    skew_t, uneq_t = [3, 17, 42, 77, 100, 126], [5, 9, 40, 61, 88, 99]
    e0_ta, e0_tb = [0, 127, 1, 126, 64, 63], [0, 127, 2, 125, 90, 5]
    specs = []
    for n in (1, 2, 3, 4, 31, 64, 129):
        specs.append((f"n{n}_rand", n, dict(kind="rand", base=1.0), real_bias))
    specs += [
        ("n512_cover", 512, dict(kind="targets", targets=cover_targets, coefs=[4.0] * 6, base=0.02, cover=True), real_bias),
        ("n64_skew_all_to_one", 64, dict(kind="targets", targets=[skew_t] * 64,
                                         coefs=skew_c, base=0.02, same=True), real_bias),
        ("n33_unequal_weights", 33, dict(kind="targets", targets=[uneq_t] * 33,
                                        coefs=uneq_c, base=0.01, same=True), "zero_bias"),
        ("n16_experts_0_127", 16, dict(kind="targets", targets=[e0_ta] * 8 + [e0_tb] * 8,
                                       coefs=e0_c, base=0.02, same=False), real_bias),
        ("n32_bias_selects", 32, dict(kind="rand", base=0.6, force=[5, 9, 100]), "synthetic_bias"),
        ("n8_saturated_situ", 8, dict(kind="rand", base=6.0), real_bias),
    ]
    errs = {}
    for name, n, spec, bias in specs:
        rng = rng_for(fx.seed, "moe/" + name)
        if spec["kind"] == "rand":
            x = randn(rng, (n, 2560), spec["base"])
        else:
            x = route_x(rng, n, 2560, rw32, spec["targets"], spec["coefs"], spec["base"])
        x = bfr(x)
        if isinstance(bias, str):
            b = real_bias.clone() if bias == "synthetic_bias" else torch.zeros(128)
            if bias == "synthetic_bias":
                b[spec["force"]] += 3.0
            bias = b
        m.gate.e_score_correction_bias.data = bias.clone()
        out, top = moe_run(m, x, torch.bfloat16)
        # design assertions
        if spec.get("targets") is not None:
            for i in range(n):
                if set(top[i].tolist()) != set(spec["targets"][i]):
                    raise RuntimeError(f"moe {name}: token {i} selected {sorted(top[i].tolist())} "
                                       f"!= designed {sorted(spec['targets'][i])}")
        if spec.get("cover"):
            hit = set(top.flatten().tolist())
            if len(hit) != 128:
                raise RuntimeError(f"moe {name}: only {len(hit)} experts hit")
        if spec.get("force"):
            for i in range(n):
                if not set(spec["force"]) <= set(top[i].tolist()):
                    raise RuntimeError(f"moe {name}: forced-by-bias experts missing in token {i}")
        if name.endswith("unequal_weights"):
            wv = out["topk_w"][0]
            if float(wv.max() / wv.min()) < 1.3:
                raise RuntimeError(f"moe {name}: weights not unequal enough: {wv.tolist()}")
        if name == "n64_skew_all_to_one" and len({tuple(sorted(r)) for r in top.tolist()}) != 1:
            raise RuntimeError("moe skew: tokens did not route to one expert set")
        if name == "n16_experts_0_127":
            hit = set(top.flatten().tolist())
            if not {0, 127} <= hit:
                raise RuntimeError("moe: experts 0/127 not both hit")
        # independent float64 numpy cross-check (small N: full; large N: first 6 rows)
        nchk = min(n, 6)
        experts64 = {e: tuple(sd[f"experts.{e}.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj"))
                     for e in set(top[:nchk].flatten().tolist())}
        shared64 = tuple(sd[f"shared_experts.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj"))
        ref64 = np_moe_generic(x[:nchk].double().numpy(), router_w.double().numpy(), experts64, shared64,
                               dict(router="sigmoid_bias", activation="situ", gate_beta=4.0, value_beta=25.0, top_k=6),
                               e_bias=bias.double().numpy())
        if set(map(tuple, np.sort(ref64["topk_idx"], 1).astype(int).tolist())) != \
                set(map(tuple, np.sort(out["topk_idx"][:nchk].numpy(), 1).astype(int).tolist())):
            raise RuntimeError(f"moe {name}: numpy routing differs from pinned gate")
        ew = rel_l2(out["topk_w"][:nchk].sort(1, descending=True).values, torch.from_numpy(ref64["topk_w"]).sort(1, descending=True).values)
        eo = rel_l2(out["out"][:nchk], torch.from_numpy(ref64["out"]))
        errs[name] = (ew, eo)
        if ew > 1e-4 or eo > 3e-2:
            raise RuntimeError(f"moe {name}: pinned vs float64 numpy: weights rel_l2 {ew:.2e}, out rel_l2 {eo:.2e}")
        fx.add("moe", name, {"N": n, "E": 128, "K": 6, "C": 2560, "I": 1088, "Ipad": 1280}, "bf16",
               {"x": x, "router_w": rw32.clone(), "e_bias": bias.clone()}, out, weights=wfile,
               params={"activation": "situ", "router": "sigmoid_bias", "gate_beta": 4.0, "value_beta": 25.0,
                       "shared": True, "mode": "bf16", "weights_dtype": "bf16",
                       "weight_prefix": "moe.", "top_k": 6},
               note="layer-1 real weights; routed = fp32 sum in ascending expert id then one bf16 cast; "
                    "out = routed + shared (bf16)")
    del m, sd
    # ---- small non-Raon geometry -----------------------------------------------------------------
    gen_moe_small(fx)
    return errs


def gen_moe_small(fx: Fixtures):
    ref = fx.ref
    e_n, kk, c, i_sz = 16, 3, 256, 192
    base_cfg = dict(hidden_size=c, num_experts=e_n, num_experts_per_token=kk, moe_intermediate_size=i_sz)
    rng = rng_for(fx.seed, "moe/small_weights")

    def mk(rows, cols):
        return bfr(randn(rng, (rows, cols), 1 / math.sqrt(cols)))

    sd = {"gate.weight": bfr(randn(rng, (e_n, c), 0.3))}
    for e in range(e_n):
        sd[f"experts.{e}.gate_proj.weight"] = mk(i_sz, c) * 2.0
        sd[f"experts.{e}.up_proj.weight"] = mk(i_sz, c) * 2.0
        sd[f"experts.{e}.down_proj.weight"] = mk(c, i_sz)
    for k, (r, cc) in {"gate_proj": (i_sz, c), "up_proj": (i_sz, c), "down_proj": (c, i_sz)}.items():
        sd[f"shared_experts.{k}.weight"] = mk(r, cc)
    wtensors = {("moe." + k): v for k, v in sd.items() if k != "gate.weight"}
    wfile = fx.write_weights("moe", "weights_small.safetensors", wtensors)
    router_w = sd["gate.weight"]
    for name, n, mode, act, router, shared in (
        ("small_situ_sigmoid_N1", 1, "bf16", "situ", "sigmoid_bias", True),
        ("small_situ_sigmoid_N5", 5, "bf16", "situ", "sigmoid_bias", True),
        ("small_situ_sigmoid_N33", 33, "bf16", "situ", "sigmoid_bias", True),
        ("small_situ_sigmoid_N17_f32", 17, "f32", "situ", "sigmoid_bias", True),
        ("small_swiglu_softmax_N1_f32", 1, "f32", "swiglu", "softmax", False),
        ("small_swiglu_softmax_N7_f32", 7, "f32", "swiglu", "softmax", True),
        ("small_swiglu_softmax_N64_f32", 64, "f32", "swiglu", "softmax", True),
    ):
        r = rng_for(fx.seed, "moe/" + name)
        x = randn(r, (n, c), 1.0)
        x = bfr(x) if mode == "bf16" else f32c(x)
        bias = (randn(r, (e_n,), 0.3) if router == "sigmoid_bias" else torch.zeros(e_n))
        params = {"activation": act, "router": router, "gate_beta": 4.0, "value_beta": 25.0, "shared": shared,
                  "mode": mode, "weights_dtype": "f32", "weight_prefix": "moe.", "top_k": kk}
        if act == "situ":
            dt = torch.bfloat16 if mode == "bf16" else torch.float32
            sdm = {k: v for k, v in sd.items()}
            sdm["gate.e_score_correction_bias"] = bias
            m, _ = moe_module(ref, base_cfg, sdm, dt, shared)
            out, top = moe_run(m, x, dt)
        else:
            experts = {e: tuple(sd[f"experts.{e}.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj"))
                       for e in range(e_n)}
            sh = tuple(sd[f"shared_experts.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj")) if shared else None
            o64 = np_moe_generic(x.double().numpy(), router_w.double().numpy(), experts, sh, params)
            out = {k: torch.from_numpy(v).float() for k, v in o64.items()}
        if act == "situ":
            experts = {e: tuple(sd[f"experts.{e}.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj"))
                       for e in range(e_n)}
            sh = tuple(sd[f"shared_experts.{p}.weight"].double().numpy() for p in ("gate_proj", "up_proj", "down_proj")) if shared else None
            o64 = np_moe_generic(x.double().numpy(), router_w.double().numpy(), experts, sh, params, e_bias=bias.double().numpy())
            tol = 3e-2 if mode == "bf16" else 1e-4
            if rel_l2(out["out"], torch.from_numpy(o64["out"])) > tol:
                raise RuntimeError(f"moe {name}: pinned vs float64 numpy mismatch")
            if not np.array_equal(np.sort(o64["topk_idx"], 1).astype(int), np.sort(out["topk_idx"].numpy(), 1).astype(int)):
                raise RuntimeError(f"moe {name}: numpy routing differs")
        fx.add("moe", name, {"N": n, "E": e_n, "K": kk, "C": c, "I": i_sz, "Ipad": 256}, mode,
               {"x": x, "router_w": router_w.clone(), "e_bias": bias}, out, weights=wfile, params=params,
               note="non-Raon geometry; " + ("pinned module" if act == "situ" else "independent float64 NumPy (policy not in the pinned file)"))


# ----------------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------------
def load_tokenizers(model_dir: str, ref_dir: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    return tok


def render_chat(model_dir: str, ref_dir: str, case: dict) -> str:
    from transformers import PreTrainedTokenizerFast

    global _CHAT_TOK
    if "_CHAT_TOK" not in globals():
        _CHAT_TOK = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(model_dir) / "tokenizer.json"),
            chat_template=(Path(ref_dir) / "chat_template.jinja").read_text(),
        )
    kw = {}
    if case.get("tools"):
        kw["tools"] = case["tools"]
    if "enable_thinking" in case:
        kw["enable_thinking"] = case["enable_thinking"]
    return _CHAT_TOK.apply_chat_template(
        case["messages"], tokenize=False, add_generation_prompt=case.get("add_generation_prompt", True), **kw)


def cmd_tokenizer(a):
    tok = load_tokenizers(a.model, a.ref_dir)
    cases = []
    for ln, line in enumerate(Path(a.corpus).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        if c["kind"] == "text":
            text = c["text"]
        elif c["kind"] == "chat":
            text = render_chat(a.model, a.ref_dir, c)
        else:
            raise SystemExit(f"{a.corpus}:{ln}: unknown kind {c['kind']}")
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({
            "id": c["id"], "kind": c["kind"], "n_tokens": len(ids), "ids": ids,
            "rendered": text, "rendered_utf8_sha256": sha256_bytes(text.encode("utf-8")),
            "ids_sha256": sha256_bytes(np.asarray(ids, dtype="<u4").tobytes()),
        })
    ids_seen = [c["id"] for c in cases]
    assert len(ids_seen) == len(set(ids_seen)), "duplicate corpus ids"
    doc = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer_sha256": sha256_file(Path(a.model) / "tokenizer.json"),
        "chat_template_sha256": sha256_file(Path(a.ref_dir) / "chat_template.jinja"),
        "corpus_sha256": sha256_file(a.corpus),
        "add_special_tokens": False,
        "cases": cases,
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=True, indent=0) + "\n")
    log(f"tokenizer: {len(cases)} cases -> {a.out}")


# ----------------------------------------------------------------------------
# streamed full-model reference forward (one layer in RAM at a time)
# ----------------------------------------------------------------------------
class StreamedModel:
    def __init__(self, ref: Ref, ck: Ckpt):
        self.ref, self.ck = ref, ck
        self.cfg = ref.cfg()

    def load_layer(self, i: int):
        mm = self.ref.mod
        pre = f"model.layers.{i}."
        sd = {}
        for name in self.ck.layer_names(i):
            if name.endswith("weight_scale_inv"):
                continue
            t = self.ck.get(name)
            if t.dtype == torch.float8_e4m3fn:
                s = self.ck.get(name + "_scale_inv")
                t = mm.dequantize_mxfp8(t, s, torch.bfloat16)
            sd[name[len(pre):]] = t
        with torch.device("meta"):
            layer = mm.Raon1MiniDecoderLayer(self.cfg, i)
        layer.load_state_dict(sd, strict=True, assign=True)
        return layer.eval()

    def embed(self, tokens):
        rows = self.ck.rows("model.embed_tokens.weight", tokens)
        return rows.to(torch.bfloat16).reshape(1, len(tokens), -1)

    def run(self, tokens, layers=None, on_layer=None, final=True):
        """Returns dict with final mean/h; calls on_layer(i, layer, stream_out, captured) per layer."""
        mm = self.ref.mod
        n = len(tokens)
        emb = self.embed(tokens)
        ns = self.cfg.raon_mhc_streams
        stream = emb.unsqueeze(2).expand(-1, -1, ns, -1).contiguous()  # M:1160
        cache = mm.Raon1MiniDynamicCache(self.cfg)
        if on_layer:
            on_layer(-1, None, stream, {})
        for i in range(self.cfg.num_hidden_layers):
            t0 = time.time()
            layer = self.load_layer(i)
            cap = {}
            hooks = []
            if on_layer:
                def keep(key, sel=None):
                    def hook(m, inp, out):
                        cap[key] = out if sel is None else sel(out)
                    return hook
                hooks.append(layer.self_attn_raon_mhc.register_forward_hook(keep("attn_mhc")))
                hooks.append(layer.mlp_raon_mhc.register_forward_hook(keep("mlp_mhc")))
                hooks.append(layer.mlp_raon_mhc.register_forward_pre_hook(
                    lambda m, args, cap=cap: cap.__setitem__("mid", args[0])))
                hooks.append(layer.self_attn.register_forward_hook(keep("attn_out")))
                hooks.append((layer.moe if layer.is_moe else layer.mlp).register_forward_hook(keep("mlp_out")))
                if not layer.is_rda:
                    hooks.append(layer.self_attn.kv_a_layernorm.register_forward_hook(keep("mla_c")))
                    hooks.append(layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(keep("mla_kv_a")))
            with torch.no_grad():
                stream = layer(stream, attention_mask=None, token_mask=None, cache=cache)
            for h in hooks:
                h.remove()
            if on_layer:
                on_layer(i, layer, stream, {"cap": cap, "cache": cache})
            del layer
            log(f"layer {i} done ({time.time() - t0:.1f}s)")
        mean = stream.float().mean(2).to(torch.bfloat16)  # M:1173
        norm = mm.Raon1MiniRMSNorm(self.cfg.hidden_size, self.cfg.rms_norm_eps)
        norm.weight.data = self.ck.get("model.norm.weight").to(torch.bfloat16)
        h = norm(mean)
        return {"mean": mean[0], "h": h[0]}

    def logits(self, h, positions=None):
        w = self.ck.get("lm_head.weight").to(torch.bfloat16)
        hh = h if positions is None else h[positions]
        return F.linear(hh, w).float()


def cmd_layer_dump(a):
    ref, ck = Ref(a.ref_dir), Ckpt(a.model)
    toks = tokens_from_file(a.tokens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = None if a.layers in (None, "all") else {int(x) for x in a.layers.split(",")}
    from safetensors.torch import save_file

    sm = StreamedModel(ref, ck)
    p = sm.cfg.rda_num_heads * sm.cfg.rda_head_dim
    files = {}

    def on_layer(i, layer, stream, ctx):
        if i < 0:
            t = {"stream": f32c(stream[0])}
            fn = "embed.safetensors"
        else:
            if want is not None and i not in want:
                return
            cap, cache = ctx["cap"], ctx["cache"]
            t = {"stream_mid": f32c(cap["mid"][0]), "stream_out": f32c(stream[0])}
            for key, nm in (("attn_mhc", "attn_mhc"), ("mlp_mhc", "mlp_mhc")):
                pre, post, res = cap[key]
                t[nm + "_pre"], t[nm + "_post"], t[nm + "_res"] = f32c(pre), f32c(post), f32c(res)
            t["attn_branch_out"] = f32c(cap["attn_out"][0])
            t["mlp_branch_out"] = f32c(cap["mlp_out"][0])
            if layer.is_rda:
                t["rda_S"] = f32c(cache.recurrent_states[i][0])
                cs = cache.conv_states[i][0].reshape(3, p, 3)
                for j, x in enumerate("qkv"):
                    t["rda_hist_" + x] = f32c(cs[j].t())
            else:
                c = cap["mla_c"][0]
                kva = cap["mla_kv_a"][0]
                t["mla_latent"] = f32c(torch.cat([c, kva[:, c.shape[-1]:]], dim=-1))
            fn = f"layer{i:02d}.safetensors"
        save_file(t, str(out / fn), metadata={"raon": json.dumps({"tokens": len(toks), "kind": "layer-dump"}, sort_keys=True)})
        files[fn] = sha256_file(out / fn)

    res = sm.run(toks, on_layer=on_layer)
    t = {"mean": f32c(res["mean"]), "h": f32c(res["h"])}
    pos = list(range(len(toks))) if a.all_logits else [len(toks) - 1]
    t["logits"] = f32c(sm.logits(res["h"], pos))
    t["logit_positions"] = f32c(torch.tensor(pos, dtype=torch.float32))
    save_file(t, str(out / "final.safetensors"), metadata={"raon": json.dumps({"kind": "layer-dump"}, sort_keys=True)})
    files["final.safetensors"] = sha256_file(out / "final.safetensors")
    (out / "dump.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "n_tokens": len(toks), "tokens": toks,
        "ref_sha256": ref.sha, "files": files,
        "note": "BF16 reference, MXFP8 experts exactly dequantized (M:496-511); stream_mid (after attention site) / stream_out (after FFN site) [N,4,2560]; "
                "rda_hist_* [3,5120] oldest->newest; mla_latent [N,576]=normed c512||kpe64",
    }, indent=1, sort_keys=True) + "\n")
    log(f"layer-dump -> {out}")


def tokens_from_file(path: str):
    p = Path(path)
    if p.suffix == ".u32":
        return np.fromfile(p, dtype="<u4").astype(np.int64).tolist()
    d = json.loads(p.read_text())
    return d["tokens"] if isinstance(d, dict) else d


# ----------------------------------------------------------------------------
# kld-check (HFKLDR v1)
# ----------------------------------------------------------------------------
def read_hfkldr_v1(path):
    """Mirror of the in-tree eval reader (crates/hipfire-runtime/examples/eval_hipfire.rs:410-440:
    8B magic, 24B header = version,n_ctx,n_vocab,n_chunk (u32) top_k,flags (u16) reserved (u32),
    then n_chunk*n_ctx u32 tokens, then per scored position 8+8*top_k bytes)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        hdr = f.read(24)
    if magic != b"HFKLDR\0\0":
        raise RuntimeError(f"{path}: bad magic {magic!r}")
    version, n_ctx, n_vocab, n_chunk = struct.unpack("<IIII", hdr[:16])
    top_k, flags, reserved = struct.unpack("<HHI", hdr[16:24])
    if version != 1:
        raise RuntimeError(f"{path}: unsupported version {version}")
    scored = n_ctx - 1 - n_ctx // 2
    block = 8 + 8 * top_k
    expect = 32 + n_chunk * n_ctx * 4 + n_chunk * scored * block
    return dict(version=version, n_ctx=n_ctx, n_vocab=n_vocab, n_chunk=n_chunk, top_k=top_k, flags=flags,
                reserved=reserved, scored_per_chunk=scored, block_bytes=block, expect_size=expect,
                tokens_offset=32, blocks_offset=32 + n_chunk * n_ctx * 4)


def cmd_kld_check(a):
    tdir = Path(a.teacher_dir) if a.teacher_dir else Path(a.ref_dir) / "teacher"
    tok = load_tokenizers(a.model, a.ref_dir)
    cfg_vocab = json.loads((Path(a.ref_dir) / "config.json").read_text())["vocab_size"]
    digests = {}
    for line in (tdir / "digests.txt").read_text().splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            digests[Path(name.strip()).name] = sha
    targets = [
        ("raon-1-mini-preview.bf16.wt2.kldref.bin", "wt2-slice.txt", 2048, 24, False),
        ("raon-1-mini-preview.bf16.code24.kldref.bin", "code24.txt", 512, 24, True),
    ]
    report = {"schema_version": SCHEMA_VERSION, "refs": []}
    ok = True
    for fname, text_name, n_ctx_exp, n_chunk_exp, tail in targets:
        path = tdir / fname
        r = {"file": fname}
        h = read_hfkldr_v1(path)
        size = path.stat().st_size
        r["header"] = h
        r["file_size"] = size
        checks = {
            "size_matches_header_geometry": size == h["expect_size"],
            "sha256_matches_digests_txt": sha256_file(path) == digests.get(fname),
            "n_vocab_matches_config": h["n_vocab"] == cfg_vocab,
            "n_ctx_expected": h["n_ctx"] == n_ctx_exp, "n_chunk_expected": h["n_chunk"] == n_chunk_exp,
            "top_k_256": h["top_k"] == 256, "flags_reserved_zero": h["flags"] == 0 and h["reserved"] == 0,
        }
        n_ch, n_ctx, k = h["n_chunk"], h["n_ctx"], h["top_k"]
        toks = np.fromfile(path, dtype="<u4", count=n_ch * n_ctx, offset=h["tokens_offset"]).reshape(n_ch, n_ctx)
        checks["token_ids_below_vocab"] = bool(toks.max() < h["n_vocab"])
        rec = np.memmap(path, dtype=[("i", "<u4", (k,)), ("p", "<f4", (k,)), ("r", "<f4"), ("z", "<f4")], mode="r",
                        offset=h["blocks_offset"], shape=(n_ch * h["scored_per_chunk"],))
        assert rec.dtype.itemsize == h["block_bytes"]
        bad = {"nonfinite": 0, "idx_oob": 0, "idx_dup": 0, "logp_pos": 0, "unsorted": 0, "resid_range": 0,
               "resid_mismatch": 0, "z_nonzero": 0}
        in_top = 0
        top1 = 0
        nll_sum = 0.0
        total = 0
        step = 4096
        spc = h["scored_per_chunk"]
        for s in range(0, len(rec), step):
            blk = rec[s : s + step]
            ii, pp, rr, zz = np.asarray(blk["i"]), np.asarray(blk["p"]), np.asarray(blk["r"]), np.asarray(blk["z"])
            bad["nonfinite"] += int((~np.isfinite(pp)).sum() + (~np.isfinite(rr)).sum())
            bad["idx_oob"] += int((ii >= h["n_vocab"]).sum())
            srt = np.sort(ii, axis=1)
            bad["idx_dup"] += int((srt[:, 1:] == srt[:, :-1]).any(axis=1).sum())
            bad["logp_pos"] += int((pp > 1e-6).sum())
            bad["unsorted"] += int((pp[:, 1:] > pp[:, :-1]).sum())
            bad["resid_range"] += int(((rr < 0) | (rr > 1)).sum())
            mass = np.exp(pp.astype(np.float64)).sum(axis=1)
            bad["resid_mismatch"] += int((np.abs(1.0 - mass - rr) > 1e-4).sum())
            bad["z_nonzero"] += int((zz != 0).sum())
            # alignment: logit at scored position p predicts token p+1 of its chunk
            gidx = np.arange(s, s + len(ii))
            chunk, off = gidx // spc, gidx % spc
            target = toks[chunk, n_ctx // 2 + off + 1]
            hit = ii == target[:, None]
            in_top += int(hit.any(axis=1).sum())
            top1 += int((ii[:, 0] == target).sum())
            lp = np.where(hit, pp, -np.inf).max(axis=1)
            nll_sum += float(-lp[np.isfinite(lp)].sum())
            total += len(ii)
        r["block_defects"] = bad
        checks["blocks_clean"] = all(v == 0 for v in bad.values())
        r["scored_positions"] = total
        r["frac_target_in_top256"] = in_top / total
        r["top1_agreement"] = top1 / total
        r["mean_nll_over_in_top256"] = nll_sum / max(in_top, 1)
        checks["alignment_target_in_topk_gt_0.9"] = r["frac_target_in_top256"] > 0.9
        # re-tokenize the source text exactly as the producer script does
        text = (tdir / text_name).read_text()
        ids = tok.encode(text, add_special_tokens=False).ids
        full = len(ids) // n_ctx
        chunks = [ids[c * n_ctx : (c + 1) * n_ctx] for c in range(min(full, n_ch))]
        if len(chunks) < n_ch:
            if tail and len(chunks) == n_ch - 1 and len(ids) >= n_ctx:
                chunks.append(ids[-n_ctx:])
        r["source_text"] = text_name
        r["source_text_sha256"] = sha256_file(tdir / text_name)
        r["stream_tokens"] = len(ids)
        checks["source_text_sha_matches_digests_txt"] = r["source_text_sha256"] == digests.get(text_name)
        checks["retokenized_chunk_count"] = len(chunks) == n_ch
        mism = [c for c in range(min(len(chunks), n_ch)) if not np.array_equal(np.asarray(chunks[c]), toks[c])]
        r["chunks_with_token_mismatch"] = mism
        checks["retokenized_ids_identical"] = len(chunks) == n_ch and not mism
        r["checks"] = checks
        r["pass"] = all(checks.values())
        ok &= r["pass"]
        report["refs"].append(r)
        log(f"kld-check {fname}: {'PASS' if r['pass'] else 'FAIL'} "
            f"top256-target {r['frac_target_in_top256']:.3f} top1 {r['top1_agreement']:.3f} "
            f"nll {r['mean_nll_over_in_top256']:.3f} tokens {len(ids)}")
        for k2, v in checks.items():
            if not v:
                log(f"  FAILED check: {k2}")
    report["pass"] = ok
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------
def cmd_selftest(a):
    ref = Ref(a.ref_dir)
    results = {}

    def check(name, cond, detail=""):
        results[name] = bool(cond)
        log(f"selftest {'PASS' if cond else 'FAIL'}: {name} {detail}")

    # 1. chart: pinned module vs float64 numpy, properties of Q / B / Hres
    for n in (3, 4, 5):
        q = np_helmert(n)
        check(f"helmert_{n}_orthonormal", np.abs(q.T @ q - np.eye(n - 1)).max() < 1e-12 and np.abs(q.sum(0)).max() < 1e-12)
        q_pinned = ref.mod.helmert_basis(n).double().numpy()
        check(f"helmert_{n}_equals_pinned", np.abs(q - q_pinned).max() < 1e-6)
    rng = rng_for(DEFAULT_SEED, "selftest/mhc")
    for n, c in ((4, 64), (3, 48), (5, 40)):
        zd = 2 * n + (n - 1) ** 2
        ps = dict(phi=bfr(randn(rng, (zd, n * c), 0.3)), alpha_pre=bfr(scal(0.9)), alpha_post=bfr(scal(0.7)),
                  alpha_z=bfr(scal(3.0)), eta=scal(0.7), b_pre=bfr(randn(rng, (n,), 1.0)),
                  b_post=bfr(randn(rng, (n,), 1.0)), b_z=bfr(randn(rng, ((n - 1) ** 2,), 1.0)), gamma=bfr(randn(rng, (c,), 0.1) + 1))
        x = bfr(randn(rng, (12, n, c), 1.0))
        out, ctx = mhc_reference(ref, "bf16", n, c, x, ps["phi"], ps["alpha_pre"], ps["alpha_post"], ps["alpha_z"],
                                 ps["eta"], ps["b_pre"], ps["b_post"], ps["b_z"], ps["gamma"], 1e-5, bfr(randn(rng, (12, c), 2.0)))
        try:
            rmax = check_mhc_against_numpy(out, n, (float(ps["alpha_pre"]), float(ps["alpha_post"]), float(ps["alpha_z"])),
                                           float(ps["eta"]), ps["b_pre"], ps["b_post"], ps["b_z"], "bf16", f"selftest-n{n}")
            check(f"mhc_chart_n{n}_pinned_vs_numpy", True, f"rho_max={rmax:.3g}")
        except RuntimeError as e:
            check(f"mhc_chart_n{n}_pinned_vs_numpy", False, str(e))
        # negative: flipped orientation must differ beyond the bf16 band
        m, sf, bo, hpost, hres, dt = ctx
        xo_wrong = ref.mod.RaonMHC.mix(sf, bo, hpost, hres.transpose(-1, -2).contiguous(), dt).float()
        e = rel_l2(xo_wrong, out["x_out"])
        er = rel_l2(hres.transpose(-1, -2), hres)
        check(f"negative_mhc_flipped_orientation_n{n}_differs", er > 1e-2, f"res rel_l2={er:.3g} x_out rel_l2={e:.3g}")
    # 2. recurrence: numpy vs pinned torch_recurrent_rda; negative = read-before-write
    h, k, t = 3, 16, 24
    r = rng_for(DEFAULT_SEED, "selftest/rda")
    qn, kn, vn = (randn(r, (1, t, h, k), 1.0) for _ in range(3))
    kn = kn / kn.norm(dim=-1, keepdim=True)
    qn = qn / qn.norm(dim=-1, keepdim=True) * k ** -0.5
    la = -5.0 * torch.sigmoid(randn(r, (1, t, h, k), 2.0))
    be = torch.sigmoid(randn(r, (1, t, h), 1.0))
    s0 = randn(r, (1, h, k, k), 0.2)
    y_t, s_t = ref.mod.torch_recurrent_rda(qn, kn, vn, la, be, s0)
    y_n, s_n = np_rda_recurrence(qn[0].numpy(), kn[0].numpy(), vn[0].numpy(), la[0].numpy(), be[0].numpy(), s0[0].numpy())
    check("rda_recurrence_pinned_vs_numpy_y", rel_l2(y_t[0], torch.from_numpy(y_n)) < 1e-5)
    check("rda_recurrence_pinned_vs_numpy_state", rel_l2(s_t[0], torch.from_numpy(s_n)) < 1e-5)
    cap = {"q": qn, "k": kn, "v": vn, "log_alpha": la, "beta": be, "state": s0}

    class _N(nn.Module):
        def forward(self, x, keep_fp32=False):
            return x
    class _R: pass
    rdum = _R(); rdum.o_norm = _N()
    yw, _ = rda_read_before_write(cap, rdum, randn(r, (t, h * k), 1.0), h, k, "f32")
    check("negative_rda_read_before_write_differs", rel_l2(yw, y_t[0].reshape(t, -1)) > 1e-2,
          f"y rel_l2={rel_l2(yw, y_t[0].reshape(t, -1)):.3g}")
    # scalar-gate equivalence
    la_eq = la[..., :1].expand_as(la).contiguous()
    y_e, s_e = ref.mod.torch_recurrent_rda(qn, kn, vn, la_eq, be, s0)
    y_s, s_s = np_rda_scalar_gate(qn[0].numpy(), kn[0].numpy(), vn[0].numpy(), la_eq[0, :, :, 0].numpy(), be[0].numpy(), s0[0].numpy())
    check("rda_equal_decay_equals_scalar_gate", rel_l2(y_e[0], torch.from_numpy(y_s)) < 1e-5 and rel_l2(s_e[0], torch.from_numpy(s_s)) < 1e-5)
    # 3. MXFP8
    allb = torch.arange(256, dtype=torch.uint8)
    dec = np_e4m3fn_decode(allb)
    tdec = allb.view(torch.float8_e4m3fn).float().numpy()
    fin = np.isfinite(tdec)
    check("e4m3fn_decoder_matches_torch_all_256", np.array_equal(dec[fin], tdec[fin]) and np.array_equal(np.isnan(dec), np.isnan(tdec)))
    wb = torch.randint(0, 120, (4, 64), dtype=torch.uint8) + (torch.randint(0, 2, (4, 64), dtype=torch.uint8) * 128)
    wq = wb.view(torch.float8_e4m3fn)
    sc = torch.randint(110, 135, (4, 2), dtype=torch.uint8)
    check("mxfp8_dequant_pinned_vs_numpy", np.array_equal(ref.mod.dequantize_mxfp8(wq, sc, torch.float32).numpy().astype(np.float64), np_mxfp8_dequant(wq, sc)))
    # 4. padded unquantized FFN (1088 -> 1280): real expert, exactness of the zero-pad contract
    ck = Ckpt(a.model)
    base = "model.layers.1.moe.experts.0."
    gw = ref.mod.dequantize_mxfp8(ck.get(base + "gate_proj.weight"), ck.get(base + "gate_proj.weight_scale_inv"), torch.bfloat16).double().numpy()
    uw = ref.mod.dequantize_mxfp8(ck.get(base + "up_proj.weight"), ck.get(base + "up_proj.weight_scale_inv"), torch.bfloat16).double().numpy()
    dw = ref.mod.dequantize_mxfp8(ck.get(base + "down_proj.weight"), ck.get(base + "down_proj.weight_scale_inv"), torch.bfloat16).double().numpy()
    xr = randn(rng_for(DEFAULT_SEED, "selftest/ffn"), (5, 2560), 1.0).double().numpy()
    act = np_situ_glu(xr @ gw.T, xr @ uw.T)
    y0 = act @ dw.T
    dwp = np.concatenate([dw, np.zeros((2560, 192))], axis=1)
    actp = np.concatenate([act, np.zeros((5, 192))], axis=1)
    check("padded_ffn_exact_zero_pad", np.abs(actp @ dwp.T - y0).max() < 1e-12)
    junk = np.concatenate([act, 123.0 * np.ones((5, 192))], axis=1)
    check("padded_ffn_zero_weight_cols_ignore_pad_activation", np.abs(junk @ dwp.T - y0).max() < 1e-10)
    # same through the pinned expert module (fp32) vs the padded numpy path
    m1 = ref.mod.Raon1MiniMLP(ref.cfg(moe_intermediate_size=1088), 1088)
    m1.gate_proj.weight.data, m1.up_proj.weight.data, m1.down_proj.weight.data = (torch.from_numpy(w).float() for w in (gw, uw, dw))
    yt = m1(torch.from_numpy(xr).float())
    check("ffn_pinned_vs_numpy_padded", rel_l2(yt, torch.from_numpy(actp @ dwp.T)) < 1e-4)
    # 5. MLA rope negative differs; no-rope reimplementation equals pinned module
    g = dict(H=4, nope=16, tail=8, v=12, latent=20, qlora=14)
    rr = rng_for(DEFAULT_SEED, "selftest/mla")
    wts = {"q_a_norm": bfr(randn(rr, (14,), 0.1) + 1), "kv_a_norm": bfr(randn(rr, (20,), 0.1) + 1),
           "kv_b": bfr(randn(rr, (4 * (16 + 12), 20), 0.3))}
    n, p = 5, 7
    inp = {"q_a_out": bfr(randn(rr, (n, 14), 2.0)), "kv_a_out": bfr(randn(rr, (n, 28), 2.0)),
           "g_out": bfr(randn(rr, (n, 48), 2.0)), "q_proj": bfr(randn(rr, (n, 4 * 24), 1.0))}
    prefix = bfr(torch.cat([randn(rr, (p, 20), 1.0), randn(rr, (p, 8), 3.0)], -1))
    out, (mla, cfg, dt) = mla_reference(ref, g, inp, prefix, wts)
    e = check_mla_numpy("selftest", g, inp, out, prefix)
    check("mla_pinned_vs_float64_numpy", e < 1.5e-2, f"rel_l2={e:.3g}")
    plain, wrong, _ = mla_attention_rope_wrong(g, inp, out, prefix, mla, dt)
    check("mla_norope_reimplementation_equals_pinned", rel_l2(plain, out["attn_y"]) < 1e-3, f"{rel_l2(plain, out['attn_y']):.3g}")
    check("negative_mla_rope_on_tail_differs", rel_l2(wrong, out["attn_y"]) > 1e-2, f"rel_l2={rel_l2(wrong, out['attn_y']):.3g}")
    # 6. generated negative-control fixtures vs their true cases (when fixtures exist)
    if a.fixtures:
        negdir = Path(a.fixtures) / "negative_controls"
        from safetensors.torch import load_file

        for mdir in sorted(negdir.glob("*/manifest.json")):
            man = json.loads(mdir.read_text())
            mod = man["module"]
            for c in man["cases"]:
                base = c["name"].split("__")[0]
                neg = load_file(str(mdir.parent / c["file"]))
                pos = load_file(str(Path(a.fixtures) / mod / f"{base}.safetensors"))
                for t in c["wrong_outputs"]:
                    e = rel_l2(neg[t], pos[t])
                    check(f"fixture_negative_{mod}/{c['name']}:{t}_differs", e > 1e-2, f"rel_l2={e:.3g}")
    allok = all(results.values())
    if a.out:
        Path(a.out).write_text(json.dumps({"pass": allok, "checks": results}, indent=1, sort_keys=True) + "\n")
    log(f"selftest {'PASS' if allok else 'FAIL'} ({sum(results.values())}/{len(results)})")
    sys.exit(0 if allok else 1)


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------
def cmd_fixtures(a):
    t_start = time.time()
    ref = Ref(a.ref_dir)
    ck = Ckpt(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    only = set(a.only.split(",")) if a.only else None
    mods = ["norm", "head", "mhc", "rda", "mla", "moe"]
    neg = not a.no_negative_controls
    import shutil

    for m in mods:
        if only is None or m in only:
            shutil.rmtree(out / m, ignore_errors=True)
            shutil.rmtree(out / "negative_controls" / m, ignore_errors=True)
    fx = Fixtures(str(out), ref, a.seed)
    checks = {}
    gens = {
        "norm": lambda: gen_norm(fx, ck), "head": lambda: gen_head(fx, ck),
        "mhc": lambda: gen_mhc(fx, ck, neg), "rda": lambda: gen_rda(fx, ck, neg),
        "mla": lambda: gen_mla(fx, ck, neg), "moe": lambda: gen_moe(fx, ck, neg),
    }
    for m in mods:
        if only is None or m in only:
            t0 = time.time()
            r = gens[m]()
            checks[m] = json.loads(json.dumps(r, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))) if r is not None else None
            log(f"{m}: generated ({time.time() - t0:.0f}s)")
    counts = fx.finalize()
    extra = {"tokens": {}, "skipped": []}
    prompts = Path(a.prompts)
    if only is None:
        tok = load_tokenizers(a.model, a.ref_dir)
        for nm in ("state", "logits"):
            ids = tok.encode((prompts / f"{nm}.txt").read_text(encoding="utf-8"), add_special_tokens=False).ids
            write_u32(out / f"{nm}.tokens.u32", ids)
            extra["tokens"][nm] = {"n": len(ids), "prompt_md5": hashlib.md5((prompts / f"{nm}.txt").read_bytes()).hexdigest()}
        if a.no_model_logits:
            extra["skipped"].append("logits.npz (--no-model-logits)")
        else:
            log("logits.npz: streamed CPU BF16 reference forward of logits.txt tokens")
            ids = np.fromfile(out / "logits.tokens.u32", dtype="<u4").astype(np.int64).tolist()
            sm = StreamedModel(ref, ck)
            res = sm.run(ids)
            lg = sm.logits(res["h"])
            save_npz_deterministic(out / "logits.npz", {
                "tokens": np.asarray(ids, dtype=np.int64), "mean": res["mean"].float().numpy(),
                "hidden": res["h"].float().numpy(), "logits": lg.numpy().astype(np.float32)})
    d = tree_digests(out)
    man = {
        "schema_version": SCHEMA_VERSION, "seed": a.seed, "modules": counts, "bands": BANDS,
        "ref_sha256": ref.sha,
        "parent": {"tokenizer_json_sha256": sha256_file(Path(a.model) / "tokenizer.json"),
                   "index_sha256": sha256_file(Path(a.model) / "model.safetensors.index.json")},
        "generator_sha256": sha256_file(__file__),
        "libs": {"torch": torch.__version__, "numpy": np.__version__},
        "independent_checks": checks, "tokens": extra["tokens"], "skipped": extra["skipped"],
        "total_bytes": sum((out / k).stat().st_size for k in d), "files": d,
    }
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    (out / "fixtures.sha256").write_text("".join(f"{v}  {k}\n" for k, v in d.items()))
    log(f"fixtures done in {time.time() - t_start:.0f}s: {len(d)} files, {man['total_bytes'] / 1e9:.2f} GB -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ref-dir", default=DEFAULT_REF)
        p.add_argument("--model", default=DEFAULT_PARENT, help="checkpoint dir (PARENT)")

    p = sub.add_parser("fixtures")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--only", help="comma list of modules (skips tokens/manifest-complete outputs)")
    p.add_argument("--no-negative-controls", action="store_true")
    p.add_argument("--no-model-logits", action="store_true")
    p.set_defaults(fn=cmd_fixtures)
    p = sub.add_parser("tokenizer")
    common(p)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_tokenizer)
    p = sub.add_parser("layer-dump")
    common(p)
    p.add_argument("--tokens", required=True, help=".u32 (LE) or JSON token list")
    p.add_argument("--layers", default="all")
    p.add_argument("--out", required=True)
    p.add_argument("--all-logits", action="store_true")
    p.set_defaults(fn=cmd_layer_dump)
    p = sub.add_parser("kld-check")
    common(p)
    p.add_argument("--teacher-dir")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_kld_check)
    p = sub.add_parser("selftest")
    common(p)
    p.add_argument("--fixtures", help="also verify negative-control fixtures against their true cases")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
