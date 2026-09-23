#!/usr/bin/env python3
# Slice O: throwaway host oracle for the gfx1201 GDN register-scan plan.
# Frozen ABI: (q,k,v,gate,beta,s_q8,s_scales,output,n_tokens,n_heads,head_dim,
#              frame,s_ef_residual), C=64, f64 serial vs f64 chunk algebra,
# Q8+EF seam/snapshot checks. No production dispatch, no test-suite boilerplate.
# Usage: python3 oracle.py  (exit 0 iff all checks pass)
import sys
import numpy as np

HD = 128
C = 64


def serial_f64(q, k, v, gate, beta, S0):
    """Per-token f64 recurrence. q,k,v: [N,HD]; gate,beta: [N]; S0: [HD,HD]
    (S0[v_row, key]). Returns (out [N,HD], S [HD,HD]). Matches the shipping
    fast kernel math (alpha=exp(gate); delta=(v-alpha*S@k)*beta; S=alpha*S+k*delta)."""
    N = q.shape[0]
    S = S0.copy()
    out = np.zeros((N, HD))
    for t in range(N):
        a = np.exp(gate[t])
        kv = S @ k[t]
        d = (v[t] - a * kv) * beta[t]
        S = a * S + np.outer(d, k[t])
        out[t] = S @ q[t]
    return out, S


def chunk_f64(q, k, v, gate, beta, S0):
    """Chunk-64 f64 algebra per plan 3.2. Returns (out, S). Must equal serial."""
    N = q.shape[0]
    S = S0.copy()
    out = np.zeros((N, HD))
    for c0 in range(0, N, C):
        cn = min(C, N - c0)
        G = np.cumsum(gate[c0:c0 + cn])
        E = np.exp(G)
        KK = k[c0:c0 + cn] @ k[c0:c0 + cn].T          # [cn,cn]
        QK = q[c0:c0 + cn] @ k[c0:c0 + cn].T
        L = np.zeros((cn, cn))
        M = np.zeros((cn, cn))
        for i in range(cn):
            for j in range(cn):
                dij = E[i] / E[j]
                if j < i:
                    L[i, j] = beta[c0 + i] * dij * KK[i, j]
                if j <= i:
                    M[i, j] = dij * QK[i, j]
        SK = k[c0:c0 + cn] @ S.T                       # [cn,HD]: SK[i,r] = S[r] @ k_i
        rhs = beta[c0:c0 + cn, None] * (v[c0:c0 + cn] - E[:, None] * SK)
        delta = np.linalg.solve(np.eye(cn) + L, rhs)  # [cn,HD]
        SQ = q[c0:c0 + cn] @ S.T                       # [cn,HD]
        MD = M @ delta
        out[c0:c0 + cn] = E[:, None] * SQ + MD
        elast = E[cn - 1]
        S = elast * S + ((delta.T * (elast / E)) @ k[c0:c0 + cn])
    return out, S


def commit_q8_ef(S_f32, EF_old_f16):
    """Replicates gated_delta_net_q8_fast.hip:427-465 EF path exactly:
    add old EF (f16->f32), row absmax, scale=max/127 (1 if 0), rint/clamp,
    EF_new = f16(S - q*scale). Returns (codes int8, scales, EF_new f16)."""
    S = S_f32 + EF_old_f16.astype(np.float64)
    mx = np.abs(S).max(axis=1)                        # [HD]
    scale = np.where(mx > 0, mx / 127.0, 1.0)
    inv = np.where(mx > 0, 127.0 / mx, 0.0)
    qf = np.rint(S * inv[:, None])
    qf = np.clip(qf, -128, 127)
    codes = qf.astype(np.int8)
    EF_new = (S - qf * scale[:, None]).astype(np.float16)
    return codes, scale.astype(np.float32), EF_new


def dequant(codes, scales):
    return codes.astype(np.float64) * scales[:, None]


RNG = np.random.default_rng(0xC10A)


def rand_inputs(N, key_kind="gauss", beta_kind="mid", gate_kind="mid"):
    if key_kind == "gauss":
        k = RNG.normal(0, 0.2, (N, HD))
    elif key_kind == "ortho":
        k = np.eye(HD)[RNG.choice(HD, N, replace=(N > HD))] * 0.5
    elif key_kind == "collinear":
        u = RNG.normal(0, 0.2, HD); u /= np.linalg.norm(u)
        k = u[None, :] * RNG.normal(0, 0.5, (N, 1)) + RNG.normal(0, 1e-3, (N, HD))
    elif key_kind == "repeated":
        k = np.tile(RNG.normal(0, 0.2, (4, HD)), (N // 4 + 1, 1))[:N]
    q = RNG.normal(0, 0.2, (N, HD))
    v = RNG.normal(0, 0.5, (N, HD))
    if beta_kind == "mid":
        beta = RNG.uniform(0.2, 1.0, N)
    elif beta_kind == "zero":
        beta = np.zeros(N)
    elif beta_kind == "one":
        beta = np.ones(N)
    if gate_kind == "mid":
        gate = RNG.normal(-0.5, 0.5, N)
    elif gate_kind == "zero":
        gate = np.zeros(N)
    elif gate_kind == "neg":
        gate = RNG.uniform(-6, -3, N)
    return q, k, v, gate, beta


fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + str(detail) if detail else ""))
    if not cond:
        fails.append(name)


# 1. serial vs chunk across shapes / K geometries / beta / gates / S0
CASES = []
for N in [1, 63, 64, 65, 129, 511, 512]:
    CASES.append((N, "gauss", "mid", "mid", True))
CASES += [(129, "ortho", "mid", "mid", True), (129, "collinear", "mid", "mid", True),
          (129, "repeated", "mid", "mid", True), (129, "gauss", "zero", "mid", True),
          (129, "gauss", "one", "mid", True), (129, "gauss", "mid", "zero", True),
          (129, "gauss", "mid", "neg", True), (65, "gauss", "mid", "mid", False)]
worst_out = 0.0
worst_state = 0.0
worst_resid = 0.0
for (N, kk, bk, gk, use_s0) in CASES:
    q, k, v, gate, beta = rand_inputs(N, kk, bk, gk)
    S0 = RNG.normal(0, 0.3, (HD, HD)) if use_s0 else np.zeros((HD, HD))
    o1, s1 = serial_f64(q, k, v, gate, beta, S0)
    o2, s2 = chunk_f64(q, k, v, gate, beta, S0)
    eo = np.abs(o1 - o2).max()
    es = np.abs(s1 - s2).max()
    # delta-equation residual defense (double-transform bug): recompute delta
    # from chunk outputs and verify the recurrence closes per token.
    S = S0.copy()
    maxres = 0.0
    for t in range(N):
        a = np.exp(gate[t])
        d = (v[t] - a * (S @ k[t])) * beta[t]
        Sp = a * S + np.outer(d, k[t])
        maxres = max(maxres, np.abs(Sp @ q[t] - o2[t]).max())
        S = Sp
    worst_out = max(worst_out, eo); worst_state = max(worst_state, es)
    worst_resid = max(worst_resid, maxres)
    check(f"alg N={N} {kk}/{bk}/{gk} s0={int(use_s0)}", eo < 1e-9 and es < 1e-9 and maxres < 1e-9,
          f"out={eo:.2e} state={es:.2e} resid={maxres:.2e}")
print(f"worst: out={worst_out:.2e} state={worst_state:.2e} resid={worst_resid:.2e}")

# 2. seam: 2x512 chunked vs widened 1024 in one call (identical bytes incl. EF)
N = 1024
q, k, v, gate, beta = rand_inputs(N)
S0 = RNG.normal(0, 0.3, (HD, HD))
EF0 = RNG.normal(0, 1e-3, (HD, HD)).astype(np.float16)
oA, sA = chunk_f64(q, k, v, gate, beta, S0)
cA, scA, eA = commit_q8_ef(sA, EF0)
# two 512 segments, EF folded only at the final commit (no per-chunk requant)
o1, s1 = chunk_f64(q[:512], k[:512], v[:512], gate[:512], beta[:512], S0)
o2, s2 = chunk_f64(q[512:], k[512:], v[512:], gate[512:], beta[512:], s1)
cB, scB, eB = commit_q8_ef(s2, EF0)
check("seam out", np.abs(np.vstack([o1, o2]) - oA).max() < 1e-9)
check("seam state bytes", np.abs(s2 - sA).max() < 1e-9)
check("seam codes", np.array_equal(cA, cB))
check("seam scales", np.array_equal(scA, scB))
check("seam EF bytes", np.array_equal(eA, eB))

# 3. commit edge rows: zero row -> scale 1; saturation/ties; nonzero EF
S = np.zeros((HD, HD)); EF = np.zeros((HD, HD), np.float16)
c, sc, e = commit_q8_ef(S, EF)
check("commit zero-row", sc[0] == 1.0 and (c[0] == 0).all() and (e[0] == 0).all())
S = np.full((HD, HD), 200.0); c, sc, e = commit_q8_ef(S, EF)
check("commit saturate", (c == 127).all() and np.allclose(sc, 200 / 127))
S = np.full((HD, HD), 0.5 / 127); c, sc, e = commit_q8_ef(S, EF)  # tie-ish small
check("commit finite", np.isfinite(c).all() and np.isfinite(sc).all())
EF = RNG.normal(0, 0.5, (HD, HD)).astype(np.float16)
S = RNG.normal(0, 0.3, (HD, HD))
c1, sc1, e1 = commit_q8_ef(S, EF)
# manual scalar recheck of row 7, col 9 rounding path
s70 = float(S[7, 9]) + float(EF[7, 9])
mx = max(abs(float(x) + float(y)) for x, y in zip(S[7], EF[7]))
inv = 127.0 / mx
qf = min(max(round(s70 * inv), -128), 127)
check("commit scalar row7", int(c1[7, 9]) == qf and abs(float(sc1[7]) - mx / 127) < 1e-6,
      f"code={int(c1[7,9])} expect={qf}")
# EF add happens exactly once: precommit S untouched by commit
S_before = S.copy()
commit_q8_ef(S, EF)
check("commit no-mutate", np.array_equal(S, S_before))

# 4. one-hot bit-exact case (f16-rounding-free): S0=0, K one-hot, beta=1,
# gate=0, small-int v/q -> L strictly 0, delta=v exact, out=v exact
N = 64
K = np.eye(HD)[:N]
V = (RNG.integers(-2, 3, (N, HD))).astype(float)
Q = np.eye(HD)[:N]
g0 = np.zeros(N); b1 = np.ones(N)
o1, s1 = serial_f64(Q, K, V, g0, b1, np.zeros((HD, HD)))
o2, s2 = chunk_f64(Q, K, V, g0, b1, np.zeros((HD, HD)))
check("onehot exact", np.array_equal(o1, o2) and np.array_equal(s1, s2),
      f"maxdiff={np.abs(o1-o2).max()}")
check("onehot values", np.array_equal(o1, V),
      "out must equal v (delta=v, S_out*k-free)")

print("ORACLE " + ("PASS" if not fails else f"FAIL {fails}"))
sys.exit(1 if fails else 0)
