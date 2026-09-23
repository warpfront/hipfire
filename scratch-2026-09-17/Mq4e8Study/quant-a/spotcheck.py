#!/usr/bin/env python3
"""3-tensor reference-chain cross-check (slice A, NOT a duplicate sweep).

Validates the Python reference chain (parent BF16 -> f32 x exact AWQ scales
-> FWHT) + qt44 decode against the planner's family tail ratios, on:
  lm_head.weight (non-AWQ), L0 mlp.gate_proj (AWQ),
  L10 linear_attn.in_proj_qkv (C1-subnormal/C4-floor boundary tensor).
Reports tail-1% MSE for fixture + C0/C2/C4.
"""
import sys

import numpy as np

QA = "/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a"
sys.path.insert(0, QA)
from hfq_util import Hfq, decode_qt44, qt44_reconstruct  # noqa: E402
from metrics import ParentStore, load_awq_dump  # noqa: E402
from oracle_py import fwht256, gen_signs  # noqa: E402

ARTS = {
    "fixture": "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq",
    "c0": "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c0.hfq",
    "c2": "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c2.hfq",
    "c4": "/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c4.hfq",
}
TENSORS = [
    "lm_head.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.10.linear_attn.in_proj_qkv.weight",
]

ps = ParentStore("/home/kaden/qcal/parents/qwen3.8-27b")
recs = load_awq_dump()
s1, s2 = gen_signs(42), gen_signs(1042)
hh = {tag: Hfq(p) for tag, p in ARTS.items()}
bn = {tag: h.by_name() for tag, h in hh.items()}

for name in TENSORS:
    m, k, sc = recs[name]
    w = ps.load_f32(name).reshape(m, k)
    if sc is not None:
        w = w * sc.reshape(1, k)
    ref = fwht256(w.reshape(-1, 256), s1, s2).reshape(-1).astype(np.float64)
    p99 = float(np.quantile(np.abs(ref), 0.99))
    mask = np.abs(ref) >= p99
    line = f"{name}ᴬᵂᑫ={'Y' if sc is not None else 'n'} p99={p99:.4e} n={m*k}"
    mses = {}
    for tag in ("fixture", "c0", "c2", "c4"):
        e = bn[tag][name]
        s, z, q = decode_qt44(hh[tag].payload(e), m * k)
        r = qt44_reconstruct(s, z, q).astype(np.float64)
        mses[tag] = float(((r - ref) ** 2)[mask].mean())
    line += "".join(f" {t}={mses[t]:.4e}" for t in mses)
    line += f" | c0/f={mses['c0']/mses['fixture']:.3f} c2/f={mses['c2']/mses['fixture']:.3f} c4/f={mses['c4']/mses['fixture']:.3f}"
    print(line, flush=True)
for h in hh.values():
    h.close()
print("spotcheck done")
