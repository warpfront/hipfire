"""NumPy oracle for the pplx-embed-v2 text encoder (arch 17 validation).

The numeric reference `hipfire-arch-qwen35-embed` is checked against. It
implements the HF `Qwen3_5Model` forward with `is_causal=false` directly on
the checkpoint's safetensors: embedding gather, zero-centered RMSNorm, the
fused per-head q+gate projection, per-head Q/K norm, partial 0.25 half-split
interleaved RoPE, bidirectional GQA attention, the GatedDeltaNet causal
recurrence (causal conv + silu, L2-normalized q/k, `g` decay, delta rule,
gated output norm), SwiGLU MLP, final norm, and the sentence-transformers
Dense head. Host post-processing (skiplist mask + row L2 normalize) is left
to the caller — compare against the daemon's `embed` output after aligning
the mask.

No torch required: numpy + safetensors + tokenizers only.

Usage:
    python3 scripts/oracle_pplx_embed.py [MODEL_DIR]

    # Defaults to ~/.hipfire/models/pplx-embed-v2-late-0.6b. Prints the token
    # ids (validating that the marker+text tokenization matches the wire
    # path) and the per-token 128-d embeddings.

Validation result (2026-10-12, gfx1101, f16 weights vs this f32 oracle):
    "hello world" (document, 3 tokens): rel-L2 1.2e-4, max abs 3e-5
    "What is the capital of France?" (query, 8 tokens): rel-L2 1.5e-4, max abs 5e-5
— the expected f16-weight-vs-f32-reference precision.
"""
import json, os, sys, numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

# Resolvable three ways so the module works standalone AND as an import:
#   $ORACLE_MODEL wins (what callers like scripts/check_embed_quant.py set),
#   then a positional argument, then the default snapshot path.
_DEFAULT_MODEL = os.path.expanduser("~/.hipfire/models/pplx-embed-v2-late-0.6b")
_POSITIONAL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
MODEL = os.environ.get("ORACLE_MODEL") or _POSITIONAL or _DEFAULT_MODEL
# `usage: ... [MODEL_DIR]` documents the positional form.
if __name__ == "__main__":
    print(f"# oracle model: {MODEL}", file=sys.stderr)
cfg = json.load(open(f"{MODEL}/config.json"))
tc, vc = cfg["text_config"], cfg["vision_config"]
d = tc["hidden_size"]; L = tc["num_hidden_layers"]
NH, NKV, HD = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
EPS = tc["rms_norm_eps"]
NROT = int(HD * tc["partial_rotary_factor"])
THETA = tc["rope_parameters"]["rope_theta"]
LKH, LVH, LKD, LVD = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
CK = tc["linear_conv_kernel_dim"]
LT = tc["layer_types"]
INTER = tc["intermediate_size"]

t = {}
with safe_open(f"{MODEL}/model.safetensors", framework="np") as f:
    for k in f.keys():
        t[k] = f.get_tensor(k)
with safe_open(f"{MODEL}/1_Dense/model.safetensors", framework="np") as f:
    for k in f.keys():
        t["dense." + k] = f.get_tensor(k)

def rmsnorm(x, w, eps=EPS):
    # HF: x/rsqrt(mean(x^2)) * (1+w)
    return x * (1.0 / np.sqrt((x.astype(np.float64)**2).mean(-1, keepdims=True) + eps)) * (1.0 + w)

def gdn_gated_norm(x, z, w, n_heads, head_dim, eps=EPS):
    # x,z: [n, n_heads*head_dim]; norm is per-head over head_dim
    sh = x.shape[:-1] + (n_heads, head_dim)
    xr = x.reshape(sh); zr = z.reshape(sh)
    xn = xr * (1.0/np.sqrt((xr.astype(np.float64)**2).mean(-1,keepdims=True)+eps))
    return ((xn * w) * (zr/(1+np.exp(-zr)))).reshape(x.shape)

def rope_partial(x, pos, n_rot=NROT, theta=THETA):
    # x: [n, heads, hd]; pairs (i, i+n_rot/2), half-split
    n = x.shape[0]; half = n_rot//2
    inv = 1.0/(theta ** (np.arange(0, n_rot, 2, dtype=np.float64)/n_rot))
    ang = np.outer(pos.astype(np.float64), inv)            # [n, half]
    c = np.cos(ang)[:, None, :]; s = np.sin(ang)[:, None, :]
    out = x.copy()
    rot = x[:, :, :n_rot]
    x1 = rot[:, :, :half]; x2 = rot[:, :, half:]
    out[:, :, :half] = x1*c - x2*s
    out[:, :, half:n_rot] = x1*s + x2*c
    return out

def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def encode(ids, hidden_only=False):
    """Final 128-d token embeddings, or the trunk's last hidden state.

    `hidden_only=True` returns the post-final-RMSNorm trunk output
    (`[n, hidden]`) before the Dense head — what HF calls
    `last_hidden_state`. Comparing trunks localizes a mismatch to the trunk or
    the head instead of only saying the two ends disagree.
    """
    n = len(ids)
    x = t["language_model.embed_tokens.weight"][ids].astype(np.float64)
    pos = np.arange(n, dtype=np.int64)
    for li in range(L):
        p = f"language_model.layers.{li}"
        res = x
        h = rmsnorm(x, t[f"{p}.input_layernorm.weight"].astype(np.float64))
        if LT[li] == "full_attention":
            qg = h @ t[f"{p}.self_attn.q_proj.weight"].astype(np.float64).T   # [n, 2*NH*HD]
            qg = qg.reshape(n, NH, 2*HD)
            q = qg[:, :, :HD].copy(); gate = qg[:, :, HD:].copy().reshape(n, NH*HD)
            k = (h @ t[f"{p}.self_attn.k_proj.weight"].astype(np.float64).T).reshape(n, NKV, HD)
            v = (h @ t[f"{p}.self_attn.v_proj.weight"].astype(np.float64).T).reshape(n, NKV, HD)
            q = rmsnorm(q, t[f"{p}.self_attn.q_norm.weight"].astype(np.float64))
            k = rmsnorm(k, t[f"{p}.self_attn.k_norm.weight"].astype(np.float64))
            q = rope_partial(q, pos); k = rope_partial(k, pos)
            rep = NH // NKV
            kk = np.repeat(k, rep, axis=1); vv = np.repeat(v, rep, axis=1)
            sc = 1.0/np.sqrt(HD)
            logits = np.einsum('nhd,mhd->hnm', q, kk) * sc        # [heads,n,n] non-causal
            w = np.exp(logits - logits.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
            ctx = np.einsum('hnm,mhd->nhd', w, vv).reshape(n, NH*HD)
            ao = ctx * (1.0/(1+np.exp(-gate)))
            mix = ao @ t[f"{p}.self_attn.o_proj.weight"].astype(np.float64).T
        else:
            qkv = h @ t[f"{p}.linear_attn.in_proj_qkv.weight"].astype(np.float64).T
            z = h @ t[f"{p}.linear_attn.in_proj_z.weight"].astype(np.float64).T
            a = h @ t[f"{p}.linear_attn.in_proj_a.weight"].astype(np.float64).T   # [n, LVH]
            b = h @ t[f"{p}.linear_attn.in_proj_b.weight"].astype(np.float64).T
            cw = t[f"{p}.linear_attn.conv1d.weight"].astype(np.float64)[:, 0, :]  # [ch, CK]
            # causal conv, left pad CK-1, silu
            pad = np.zeros((CK-1, qkv.shape[1]))
            xs = np.concatenate([pad, qkv], 0)
            conv = np.zeros_like(qkv)
            for j in range(CK):
                conv += xs[j:j+n] * cw[:, j]
            conv = conv * (1.0/(1+np.exp(-conv)))   # silu
            kd = LKH*LKD; vd = LVH*LVD
            q = conv[:, :kd].reshape(n, LKH, LKD)
            k = conv[:, kd:2*kd].reshape(n, LKH, LKD)
            v = conv[:, 2*kd:].reshape(n, LVH, LVD)
            # l2 norm + scale
            q = q / np.sqrt((q**2).sum(-1, keepdims=True) + EPS) / np.sqrt(LKD)
            k = k / np.sqrt((k**2).sum(-1, keepdims=True) + EPS)
            beta = 1.0/(1+np.exp(-b))
            A = np.exp(t[f"{p}.linear_attn.A_log"].astype(np.float64))
            g = -A * softplus(a + t[f"{p}.linear_attn.dt_bias"].astype(np.float64))
            rep = LVH // LKH
            if rep > 1:
                q = np.repeat(q, rep, axis=1); k = np.repeat(k, rep, axis=1)
            out = np.zeros((n, LVH, LVD))
            for hh in range(LVH):
                S = np.zeros((LVD, LKD))          # [dv, dk]
                for ti in range(n):
                    dec = np.exp(g[ti, hh])
                    wq = (S @ q[ti, hh]) * dec
                    wk = (S @ k[ti, hh]) * dec
                    delta = beta[ti, hh] * (v[ti, hh] - wk)
                    out[ti, hh] = wq + np.dot(q[ti,hh], k[ti,hh])*delta
                    S = dec*S + np.outer(delta, k[ti, hh])
                # note: out uses dec*(S_prev@q) + (q.k)*delta
            on = gdn_gated_norm(out.reshape(n, LVH*LVD), z, t[f"{p}.linear_attn.norm.weight"].astype(np.float64), LVH, LVD)
            mix = on @ t[f"{p}.linear_attn.out_proj.weight"].astype(np.float64).T
        x = res + mix
        res = x
        h = rmsnorm(x, t[f"{p}.post_attention_layernorm.weight"].astype(np.float64))
        gg = h @ t[f"{p}.mlp.gate_proj.weight"].astype(np.float64).T
        uu = h @ t[f"{p}.mlp.up_proj.weight"].astype(np.float64).T
        x = res + ((gg/(1+np.exp(-gg)))*uu) @ t[f"{p}.mlp.down_proj.weight"].astype(np.float64).T
    x = rmsnorm(x, t["language_model.norm.weight"].astype(np.float64))
    if hidden_only:
        return x
    return x @ t["dense.linear.weight"].astype(np.float64).T   # [n, 128]

if __name__ == "__main__":
    tok = Tokenizer.from_file(f"{MODEL}/tokenizer.json")
    for kind, marker, text in [("query","[Q] ","What is the capital of France?"),
                               ("document","[D] ","hello world")]:
        whole = tok.encode(marker + text).ids
        piece = [tok.token_to_id(marker)] + tok.encode(text).ids
        print(kind, "whole=", whole, "piece=", piece, "match=", whole == piece)
        emb = encode(np.array(whole))
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        print("  rows", emb.shape, "norm0", round(float(np.linalg.norm(emb[0])),6))
        print("  first row first 6:", [round(float(v),4) for v in emb[0][:6]])
