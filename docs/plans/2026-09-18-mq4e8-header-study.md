# MQ4E8 header-constraint study — 2026-09-18

## Decision in one paragraph

**C2 passes the weight-quality gate:** WT2_24 **0.045629** versus **0.048659**; AG_24 **0.137719** versus **0.141396**. Keep it opt-in; no speedup is established. **Reject the frozen whole-K row-master route for a default:** C4 WT2_24 **0.097665**, and the compatible activation-only arm adds **0.007054** versus its iu4 baseline. C3 costs **0.001054**, also above the allowed 0.0005. C1/C4z/C4s were screened but not KLD-evaluated under Main’s narrowed scope. Integer folding still requires compatible activation scales; C2’s weight-only success is not that proof.

## Scope and evidence contract

This study changes scale relations, **not FWHT, AWQ, nibble bit width, GEMM kernels, or production defaults**. It is distinct from the retired symmetric/e8m0-group format in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-14-mq4e8-format.md`. Its old fp8 roof statement is corrected by [the supplied roof record](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/gfx1201_roofs_snapshot.txt); no speed claim here uses that obsolete roof.

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-mq4e8`, branch `mq4e8-study`, created from `mq4-lloyd @ b7169e4a9`. Evidence stays untracked in `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study`. No writes were made by this study to wt-lloyd or hipfire-beta. GPU measurements use ordinal 0, never ordinal 3.

Pinned input: `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq`; **md5 `1a48b45ebf5f3ef852efcf8ee5d8084b`**, sha256 `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3`. [Fixture verification](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/fixture_verification.log); [input/reference/executable hashes](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/inputs_md5.json).

The adoption gate is WT2 24-chunk KLD **≤0.049159**, i.e. **0.048659 + 0.0005**, on the existing fp8-prefill/GEMV route. Anything above that is not a default; accepting a quality/speed opt-in is Kaden's decision. The unchanged-weight activation-only arm has its own iu4 baseline **0.063410**, hence **0.063910** gate. These numbers are the supplied contracts, not newly inferred measurements. AG corroborates rather than replaces WT2. Short 1/2-chunk runs diagnose failures; the 24-chunk run decides the weight gate.

## 1. Input and numerical method

- Original BF16 parent: `/home/kaden/qcal/parents/qwen3.8-27b/`; imatrix: `/home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf`. The reference is **original post-AWQ/post-FWHT coefficients**, never decoded MQ4 values.
- Reproduce `compute_awq_scales` with f32 alpha 0.55 and the existing f64 log/geometric normalization. Multiply parent weights by the **pre-f16 f32 scale** before FWHT. Stored AWQ sidecars are historically truncated f16 and are not substituted for that f32 pre-scale. FWHT seeds remain 42/1042, scale 1/16.
- Historical `f32_to_f16` truncates, not RN-even. The independent CPU reconstruction matched all **200,171,520 stored half scale/zero pairs exactly**. A separate four-tensor/128-row code oracle compared **524,288 nibble bytes with zero mismatches**. [Coverage proof](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_coverage_receipt.json); [code oracle](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/baseline_code_oracle.json).
- Pilot: layers **0, 3, 31, 32, 60, 63** plus lm_head; up to 128 evenly spaced output rows per tensor, always the full K row. **46 tensors, 37,912,576 coefficients**. Main sweep: **all 497 qt44 tensors, all 100,085,760 groups, all 25,621,954,560 coefficients**, across all 64 layers plus lm_head. Non-qt44 tensors are not requantized.
- Tail-1% MSE: for each tensor choose its exact 99th percentile of `abs(original rotated coefficient)` and use that same mask for every candidate. Sum tail squared errors and tail counts within each family, then divide. These are squared raw coefficient units, not normalized-grid error and not overall MSE.
- Max-coef: each 256-group's largest-absolute original coefficient; relative error `abs(decoded-original)/abs(original)`. Tables show **mean / worst percent** across groups. Family aggregation is group-weighted. Tiny subnormal rows can dominate worst-relative-error while contributing little raw tail energy; both are reported, not hidden.
- `C0` is an unconstrained-scale, zero-refitted control. `fixture` is the actual original scale/zero/codes, not a freshly optimized substitute. Lloyd here means fixed integer levels 0..15 with local zero refinement, **not** the nonuniform per-tensor Lloyd codebook/qt52 format.

Reproducible measurement sources/logs: [header census source](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/measure_headers.py), [constraint experiment source](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/measure_constraints.py), [symmetry experiment source](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/measure_symmetry.py); [pilot raw log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/pilot_rows128.jsonl), [pilot symmetry raw log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/pilot_rows128_symmetry.jsonl), [all-coefficient raw log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_rows0.jsonl), [all-coefficient symmetry raw log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_rows0_symmetry.jsonl). Every metric below is reduced from these logs; no overall-MSE/Frobenius ranking is used.

## 2. Frozen constraints and representable wire contract

Write original half scales as `s0,s1`, group maximum `G=max(s0,s1)`. Every candidate retains separate per-half zeros unless explicitly constrained.

| Arm | Frozen scale/zero rule | What it can remove |
|---|---|---|
| C0 | Original f16 half scales; fixed-scale Lloyd zero refit | Control only |
| C1 | `G*2^-d`, `d=0..4`, nearest absolute scale; ties larger scale; only exactly f16-representable candidates | Integer combination within a weight group, **conditional on compatible activation scales** |
| C2 | `B=trunc_f16(G/16)` with low four f16 significand bits cleared; `s_h=B*m_h`, nearest `m_h=1..16` | Small-integer weight multiplier within group; same activation qualification |
| C3 | Both scales `G`; two independent fitted zeros | One weight scale per 256; not MQ4C's single zero |
| C4 | Per entire output row `B=max(trunc_f16(max_h s_h/256),2^-24)`; nearest `B*2^e`, `e=0..8` | Whole-K code-product integer fold **only with paired activation masters**; free zeros still need f32 correction |
| C4z | C4 master additionally clears low four f16 significand bits; `z_h=s_h*k_h`, nearest/refitted integer `k_h in[-15,0]` | Zero correction becomes integer MAD, but does not vanish |
| C4s | Exactly C4's scales/master; fixed `z_h=-8*s_h`, no free-zero refit | Signed-i4 recode `q xor 8`, interpreted as `q-8`; no zero correction |

C2 positive-base floor is f16 bits `0x0010`; C4z cleared-master floor is also `0x0010`. C4's floor is bits `0x0001`. Full census exercised **11 nonrepresentable C1 candidate exclusions, 226 C2 group-base floors, 3 C4 row-master floors, 28 C4z row-master floors**. Initial experiments that deliberately rejected these real subnormal boundaries are retained separately and are **not** included in final aggregates. A format must specify these boundaries, not silently break its integer relation.

C2's seven-significant-bit base multiplied by any integer up to16 needs at most eleven significant bits, so all decoded half scales are exactly f16. C4z similarly makes `s*k` exactly f16. C4/C4s powers of two preserve representability in the validated finite domain. The original reference `16*B` can exceed both actual half scales in the few floored tiny groups; the invariant is `s_h=B*m_h`, not a claim that a half must attain m16.

### Fixed-scale Lloyd rule

Initialize zero to the original f16 minmax zero. For at most eight iterations, assign

`u = f32(f32(w-z)*f32(1/s)); q = clamp(floor(f32(u+0.5)), 0, 15)`

then set `z = trunc_f16(f32(mean_f64(f32(w - f32(s*q)))))`; stop when the stored zero is stable. Always do a final assignment using the stored scale/zero. Scales never change inside Lloyd. C4z projects the mean zero onto `s*k`, `k in[-15,0]`, with the same final assignment. C4s has fixed zero and therefore only code assignment. All-zero groups use the explicitly representable constant case; the pinned fixture has no zero/nonfinite half scales.

### Quality artifacts: lossless lowering, no loader/GEMV change

The study emits **existing qt44** groups, exactly **136 B/group**:

`[0:2] f16 s0 | [2:4] f16 z0 | [4:6] f16 s1 | [6:8] f16 z1 | [8:136] 128 B nibbles`.

Both scales have already been decoded from the constrained representation, **without rounding loss**. Thus existing GEMV and fp8-prefill consume precisely the candidate's reconstruction, with no new GEMM kernel, qtype, or decoder approximation. Loader/GEMV change needed for these quality artifacts: **none**. Format flags and provenance distinguish recipes; qt44 semantics do not change. qt52 is already the Lloyd-codebook format and must not be reused. No compressed-file-size saving is claimed for lowered artifacts.

Future compact eight-byte header, if quality and performance later justify it: C1/C2 `[f16 base][f16 z0][f16 z1][u8 selector0][u8 selector1]`, with C1 rebased to the smaller half and nonnegative exponent selectors, C2 integer selectors1..16; C3 one base and reserved-zero selectors. C4 uses `[u8 e0][u8 e1][f16 z0][f16 z1][u16 reserved=0]` plus f16 row-master sidecar. C4z may store integer zero selectors; C4s needs no zero values. A distinct future qtype must be allocated through the registry, not guessed. A CPU compact→qt44 loader can expand once and compare bitwise against these quality artifacts; native compressed GPU decode is a separate, unimplemented slice.

## 3. Empirical actual-header distributions

Source: [full header census](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/header_ratios.json). `h0` is K0..127; `h1` is K128..255. Thus `s_hi/s_lo` below means **s1/s0 by K position**, not sorting scales by magnitude. Histogram bins are left-closed; values are percentages of each family's groups. Exact finer-bin counts and extrema are in the JSON. All 100,085,760 groups were inspected, with no nonfinite/zero half scales.

### Half ratio s1/s0 (%)

| Family | Groups | <.5 | .5–.75 | .75–.875 | .875–1 | 1–1.125 | 1.125–1.5 | 1.5–2 | ≥2 |
|---|---|---|---|---|---|---|---|---|---|
| lm_head | 4,966,400 | 0.0009 | 3.5080 | 16.4100 | 29.8424 | 27.1847 | 22.4958 | 0.5575 | 0.0007 |
| linear_attn.in_proj_a | 46,080 | 0.0000 | 2.8103 | 15.7856 | 31.0569 | 28.3767 | 21.5690 | 0.3993 | 0.0022 |
| linear_attn.in_proj_b | 46,080 | 0.0022 | 2.7148 | 15.6359 | 31.1372 | 28.9258 | 21.2153 | 0.3689 | 0.0000 |
| linear_attn.in_proj_qkv | 9,830,400 | 0.0007 | 3.4358 | 16.4252 | 30.0801 | 27.3174 | 22.2143 | 0.5260 | 0.0005 |
| linear_attn.in_proj_z | 5,898,240 | 0.0005 | 3.3994 | 16.3599 | 30.1158 | 27.3827 | 22.2243 | 0.5169 | 0.0006 |
| linear_attn.out_proj | 5,898,240 | 0.0006 | 3.3618 | 16.3691 | 30.1805 | 27.4095 | 22.1756 | 0.5024 | 0.0005 |
| mlp.down_proj | 22,282,240 | 0.0006 | 3.4740 | 16.4594 | 29.9534 | 27.2461 | 22.3313 | 0.5345 | 0.0007 |
| mlp.gate_proj | 22,282,240 | 0.0006 | 3.4843 | 16.4818 | 29.9608 | 27.2356 | 22.3017 | 0.5346 | 0.0006 |
| mlp.up_proj | 22,282,240 | 0.0007 | 3.5081 | 16.4764 | 29.9331 | 27.1798 | 22.3562 | 0.5452 | 0.0006 |
| self_attn.k_proj | 327,680 | 0.0003 | 3.1784 | 16.1221 | 30.6943 | 27.8198 | 21.7239 | 0.4611 | 0.0000 |
| self_attn.o_proj | 1,966,080 | 0.0006 | 3.5211 | 16.5966 | 30.0123 | 27.1591 | 22.1849 | 0.5251 | 0.0004 |
| self_attn.q_proj | 3,932,160 | 0.0005 | 3.3421 | 16.3169 | 30.3275 | 27.5539 | 21.9637 | 0.4946 | 0.0008 |
| self_attn.v_proj | 327,680 | 0.0006 | 3.4280 | 16.3049 | 30.3836 | 27.4582 | 21.9287 | 0.4953 | 0.0006 |

### Each half divided by its group maximum (%)

| Family | Half | <.5 | .5–.75 | .75–.875 | .875–.9375 | .9375–.96875 | .96875–1 | =1 |
|---|---|---|---|---|---|---|---|---|
| lm_head | h0 | 0.0007 | 3.5750 | 16.5607 | 14.2248 | 7.8582 | 7.8281 | 49.9524 |
| lm_head | h1 | 0.0009 | 3.5080 | 16.4100 | 14.1777 | 7.8562 | 7.8086 | 50.2387 |
| linear_attn.in_proj_a | h0 | 0.0022 | 2.9188 | 16.1155 | 14.6267 | 8.2161 | 8.2595 | 49.8611 |
| linear_attn.in_proj_a | h1 | 0.0000 | 2.8103 | 15.7856 | 14.5768 | 8.3138 | 8.1662 | 50.3472 |
| linear_attn.in_proj_b | h0 | 0.0000 | 2.7040 | 15.9505 | 14.7374 | 8.4158 | 8.5113 | 49.6810 |
| linear_attn.in_proj_b | h1 | 0.0022 | 2.7148 | 15.6359 | 14.6181 | 8.3008 | 8.2183 | 50.5100 |
| linear_attn.in_proj_qkv | h0 | 0.0005 | 3.4293 | 16.4043 | 14.2445 | 7.9100 | 7.8957 | 50.1157 |
| linear_attn.in_proj_qkv | h1 | 0.0007 | 3.4358 | 16.4252 | 14.2748 | 7.9030 | 7.9024 | 50.0582 |
| linear_attn.in_proj_z | h0 | 0.0005 | 3.4086 | 16.4061 | 14.2795 | 7.9397 | 7.9078 | 50.0577 |
| linear_attn.in_proj_z | h1 | 0.0005 | 3.3994 | 16.3599 | 14.2505 | 7.9567 | 7.9086 | 50.1244 |
| linear_attn.out_proj | h0 | 0.0005 | 3.3792 | 16.3816 | 14.2882 | 7.9414 | 7.9243 | 50.0848 |
| linear_attn.out_proj | h1 | 0.0006 | 3.3618 | 16.3691 | 14.3077 | 7.9494 | 7.9234 | 50.0880 |
| mlp.down_proj | h0 | 0.0007 | 3.4838 | 16.4736 | 14.2215 | 7.8839 | 7.8649 | 50.0716 |
| mlp.down_proj | h1 | 0.0006 | 3.4740 | 16.4594 | 14.2109 | 7.8814 | 7.8611 | 50.1126 |
| mlp.gate_proj | h0 | 0.0005 | 3.4734 | 16.4563 | 14.2207 | 7.8713 | 7.8706 | 50.1071 |
| mlp.gate_proj | h1 | 0.0006 | 3.4843 | 16.4818 | 14.2200 | 7.8868 | 7.8539 | 50.0724 |
| mlp.up_proj | h0 | 0.0006 | 3.5114 | 16.4814 | 14.2134 | 7.8585 | 7.8384 | 50.0963 |
| mlp.up_proj | h1 | 0.0007 | 3.5081 | 16.4764 | 14.2030 | 7.8720 | 7.8581 | 50.0818 |
| self_attn.k_proj | h0 | 0.0000 | 3.1763 | 16.0419 | 14.5236 | 8.0551 | 8.0164 | 50.1868 |
| self_attn.k_proj | h1 | 0.0003 | 3.1784 | 16.1221 | 14.5200 | 8.0774 | 8.0969 | 50.0049 |
| self_attn.o_proj | h0 | 0.0004 | 3.4494 | 16.3737 | 14.1228 | 7.8736 | 7.8808 | 50.2993 |
| self_attn.o_proj | h1 | 0.0006 | 3.5211 | 16.5966 | 14.2740 | 7.8727 | 7.8655 | 49.8695 |
| self_attn.q_proj | h0 | 0.0008 | 3.3220 | 16.2253 | 14.3095 | 8.0109 | 7.9631 | 50.1683 |
| self_attn.q_proj | h1 | 0.0005 | 3.3421 | 16.3169 | 14.3359 | 7.9990 | 7.9926 | 50.0130 |
| self_attn.v_proj | h0 | 0.0006 | 3.3511 | 16.1304 | 14.3094 | 7.9745 | 7.9477 | 50.2863 |
| self_attn.v_proj | h1 | 0.0006 | 3.4280 | 16.3049 | 14.3802 | 8.0408 | 7.9626 | 49.8828 |

## 4. Pilot quality tables

Pilot raw logs and exact selection are specified in §1. Tail MSE is absolute raw-coefficient squared error on the common tail mask.

| Family | fixture | C0 | C1 | C2 | C3 | C4 | C4z | C4s |
|---|---|---|---|---|---|---|---|---|
| lm_head | 1.183835e-06 | 1.326850e-06 | 3.423702e-06 | 1.550550e-06 | 1.663707e-06 | 3.498294e-05 | 4.485649e-05 | 4.386611e-05 |
| linear_attn.in_proj_a | 8.127148e-06 | 7.939189e-06 | 1.013483e-05 | 8.333007e-06 | 9.021941e-06 | 7.905814e-05 | 1.127795e-04 | 9.086745e-05 |
| linear_attn.in_proj_b | 4.282824e-06 | 4.219812e-06 | 5.650605e-06 | 4.313552e-06 | 4.879101e-06 | 2.632293e-05 | 3.612761e-05 | 3.063076e-05 |
| linear_attn.in_proj_qkv | 4.680571e-06 | 4.734964e-06 | 7.185391e-06 | 5.025110e-06 | 5.484036e-06 | 5.406089e-05 | 8.116487e-05 | 6.297684e-05 |
| linear_attn.in_proj_z | 2.497409e-06 | 2.665767e-06 | 5.162030e-06 | 2.974811e-06 | 3.173376e-06 | 5.605688e-05 | 7.255266e-05 | 6.923658e-05 |
| linear_attn.out_proj | 4.335105e-06 | 4.391962e-06 | 7.336182e-06 | 4.703257e-06 | 5.095756e-06 | 6.900343e-05 | 9.571716e-05 | 8.424765e-05 |
| mlp.down_proj | 9.268959e-07 | 1.062782e-06 | 2.301765e-06 | 1.245826e-06 | 1.311150e-06 | 4.129266e-05 | 5.120560e-05 | 5.065612e-05 |
| mlp.gate_proj | 9.580801e-07 | 1.074851e-06 | 2.256064e-06 | 1.252352e-06 | 1.335244e-06 | 2.489912e-05 | 3.100982e-05 | 3.171688e-05 |
| mlp.up_proj | 6.794230e-07 | 8.171343e-07 | 1.876910e-06 | 9.783231e-07 | 1.024861e-06 | 2.168515e-05 | 2.565395e-05 | 2.853460e-05 |
| self_attn.k_proj | 4.507856e-06 | 4.594958e-06 | 7.649580e-06 | 4.936494e-06 | 5.358689e-06 | 5.828937e-05 | 8.605536e-05 | 6.748306e-05 |
| self_attn.o_proj | 3.297689e-06 | 3.447704e-06 | 7.745605e-06 | 3.790851e-06 | 4.176215e-06 | 7.107469e-05 | 9.571868e-05 | 8.786735e-05 |
| self_attn.q_proj | 7.777507e-06 | 7.595943e-06 | 1.058373e-05 | 7.882020e-06 | 8.426140e-06 | 5.097276e-05 | 7.181560e-05 | 6.138004e-05 |
| self_attn.v_proj | 4.537995e-06 | 4.862756e-06 | 1.106877e-05 | 5.390508e-06 | 5.803438e-06 | 9.248161e-05 | 1.263211e-04 | 1.116652e-04 |

Max-coef **mean / worst relative percent**:

| Family | fixture | C0 | C1 | C2 | C3 | C4 | C4z | C4s |
|---|---|---|---|---|---|---|---|---|
| lm_head | 0.0407 / 0.163 | 0.9741 / 4.736 | 1.0933 / 6.194 | 1.2492 / 8.821 | 1.0933 / 6.194 | 9.0987 / 53.694 | 8.4345 / 66.146 | 12.7176 / 51.851 |
| linear_attn.in_proj_a | 0.0388 / 0.157 | 0.9837 / 4.506 | 1.1605 / 6.484 | 1.2584 / 6.977 | 1.1605 / 6.484 | 11.3848 / 54.998 | 11.5988 / 66.052 | 16.3970 / 52.240 |
| linear_attn.in_proj_b | 0.0394 / 0.173 | 0.9825 / 5.076 | 1.1182 / 6.089 | 1.2091 / 6.159 | 1.1182 / 6.089 | 10.7262 / 54.197 | 10.5394 / 65.792 | 15.7304 / 53.403 |
| linear_attn.in_proj_qkv | 0.0364 / 0.162 | 1.0163 / 5.167 | 1.1695 / 6.482 | 1.2786 / 8.200 | 1.1695 / 6.482 | 9.9257 / 54.276 | 9.5882 / 66.295 | 13.8920 / 52.100 |
| linear_attn.in_proj_z | 0.0372 / 0.178 | 1.0067 / 4.683 | 1.1645 / 6.576 | 1.2814 / 8.147 | 1.1645 / 6.576 | 11.7584 / 54.003 | 11.3294 / 66.337 | 16.2026 / 53.431 |
| linear_attn.out_proj | 0.0382 / 0.181 | 1.0176 / 4.808 | 1.1544 / 6.410 | 1.2752 / 6.595 | 1.1544 / 6.410 | 13.1432 / 55.853 | 12.9669 / 66.372 | 18.5138 / 55.266 |
| mlp.down_proj | 0.0435 / 0.178 | 1.0038 / 5.423 | 1.1373 / 6.579 | 1.2881 / 8.316 | 1.1373 / 6.579 | 13.7478 / 57.408 | 12.7786 / 66.585 | 17.9168 / 55.326 |
| mlp.gate_proj | 0.0414 / 0.182 | 0.9916 / 5.071 | 1.1469 / 6.334 | 1.2771 / 7.980 | 1.1469 / 6.334 | 9.4235 / 56.381 | 8.9052 / 66.235 | 13.1460 / 54.407 |
| mlp.up_proj | 0.0427 / 0.181 | 1.0012 / 5.139 | 1.1391 / 6.325 | 1.2744 / 8.783 | 1.1391 / 6.325 | 8.8232 / 55.875 | 8.3735 / 66.503 | 12.5467 / 53.360 |
| self_attn.k_proj | 0.0376 / 0.168 | 0.9951 / 4.914 | 1.1376 / 6.383 | 1.2606 / 7.017 | 1.1376 / 6.383 | 11.5503 / 58.434 | 11.2122 / 66.611 | 16.2321 / 53.107 |
| self_attn.o_proj | 0.0379 / 0.169 | 1.0033 / 5.058 | 1.1561 / 6.575 | 1.2853 / 8.232 | 1.1561 / 6.575 | 13.3811 / 54.461 | 13.4343 / 66.404 | 18.8063 / 55.090 |
| self_attn.q_proj | 0.0395 / 0.181 | 0.9836 / 5.197 | 1.1379 / 6.549 | 1.2499 / 7.983 | 1.1379 / 6.549 | 10.8817 / 54.981 | 10.5693 / 66.118 | 15.2117 / 52.690 |
| self_attn.v_proj | 0.0371 / 0.174 | 1.0084 / 4.690 | 1.1433 / 6.520 | 1.2612 / 7.690 | 1.1433 / 6.520 | 10.8090 / 54.054 | 10.5444 / 66.556 | 14.8726 / 51.489 |

## 5. Every-layer, every-coefficient quality tables

No row sampling in these tables. [Family summary](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_summary.json); [coverage and exactness receipt](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_coverage_receipt.json).

### Tail-1% MSE

| Family | fixture | C0 | C1 | C2 | C3 | C4 | C4z | C4s |
|---|---|---|---|---|---|---|---|---|
| lm_head | 1.066109e-06 | 1.245383e-06 | 3.134519e-06 | 1.479320e-06 | 1.570450e-06 | 3.210645e-05 | 3.992060e-05 | 4.118185e-05 |
| linear_attn.in_proj_a | 8.116900e-06 | 8.068603e-06 | 1.111479e-05 | 8.457520e-06 | 9.187647e-06 | 8.818868e-05 | 1.232229e-04 | 1.050030e-04 |
| linear_attn.in_proj_b | 1.077811e-05 | 1.039132e-05 | 1.200151e-05 | 1.052105e-05 | 1.134141e-05 | 3.790913e-05 | 5.108850e-05 | 4.561217e-05 |
| linear_attn.in_proj_qkv | 2.547409e-06 | 2.740661e-06 | 5.079525e-06 | 3.040057e-06 | 3.264865e-06 | 4.723046e-05 | 6.182398e-05 | 5.991140e-05 |
| linear_attn.in_proj_z | 3.115805e-06 | 3.241690e-06 | 5.217227e-06 | 3.506387e-06 | 3.741359e-06 | 4.584855e-05 | 5.843833e-05 | 5.871230e-05 |
| linear_attn.out_proj | 3.750681e-06 | 3.906367e-06 | 6.778897e-06 | 4.243124e-06 | 4.593132e-06 | 6.488765e-05 | 8.315813e-05 | 8.201383e-05 |
| mlp.down_proj | 8.068611e-07 | 9.380892e-07 | 2.015999e-06 | 1.114937e-06 | 1.169539e-06 | 3.740107e-05 | 4.384116e-05 | 4.684594e-05 |
| mlp.gate_proj | 7.929687e-07 | 9.204719e-07 | 2.078376e-06 | 1.090774e-06 | 1.148766e-06 | 2.325848e-05 | 2.783482e-05 | 3.013804e-05 |
| mlp.up_proj | 5.114960e-07 | 6.534158e-07 | 1.632379e-06 | 8.223198e-07 | 8.481385e-07 | 1.910447e-05 | 2.083079e-05 | 2.623323e-05 |
| self_attn.k_proj | 3.918948e-06 | 4.024115e-06 | 6.736775e-06 | 4.334837e-06 | 4.681393e-06 | 5.894558e-05 | 8.437349e-05 | 6.926956e-05 |
| self_attn.o_proj | 7.616509e-06 | 7.582610e-06 | 1.186120e-05 | 7.942575e-06 | 8.705413e-06 | 7.967964e-05 | 1.091081e-04 | 9.741103e-05 |
| self_attn.q_proj | 5.755188e-06 | 5.708750e-06 | 7.916308e-06 | 5.949073e-06 | 6.438626e-06 | 4.869171e-05 | 6.692692e-05 | 5.914769e-05 |
| self_attn.v_proj | 3.396850e-06 | 3.717794e-06 | 7.696526e-06 | 4.209018e-06 | 4.502089e-06 | 7.786202e-05 | 1.027569e-04 | 9.850718e-05 |

### Max-coef mean / worst relative percent

| Family | fixture | C0 | C1 | C2 | C3 | C4 | C4z | C4s |
|---|---|---|---|---|---|---|---|---|
| lm_head | 0.0417 / 0.191 | 1.0032 / 6.235 | 1.1433 / 6.624 | 1.2827 / 9.828 | 1.1433 / 6.624 | 8.4424 / 57.779 | 8.1292 / 66.631 | 12.0226 / 56.603 |
| linear_attn.in_proj_a | 0.0393 / 0.182 | 1.0055 / 5.207 | 1.1440 / 6.516 | 1.2717 / 7.789 | 1.1440 / 6.516 | 12.1908 / 57.973 | 12.5209 / 66.428 | 17.8985 / 56.948 |
| linear_attn.in_proj_b | 0.0398 / 0.180 | 1.0076 / 5.310 | 1.1407 / 6.468 | 1.2625 / 7.821 | 1.1407 / 6.468 | 11.9459 / 58.415 | 12.2142 / 66.490 | 17.5860 / 55.034 |
| linear_attn.in_proj_qkv | 0.0364 / 5.613 | 1.0041 / 6.532 | 1.1467 / 38.304 | 1.2561 / 61.790 | 1.1467 / 6.639 | 10.1588 / 60.252 | 9.8958 / 66.633 | 14.1134 / 57.462 |
| linear_attn.in_proj_z | 0.0377 / 0.190 | 1.0035 / 6.147 | 1.1469 / 31.161 | 1.2636 / 22.398 | 1.1468 / 6.631 | 11.1612 / 60.077 | 10.8641 / 66.638 | 15.4340 / 58.186 |
| linear_attn.out_proj | 0.0368 / 0.190 | 1.0035 / 5.820 | 1.1462 / 42.005 | 1.2578 / 9.717 | 1.1462 / 6.639 | 12.8900 / 59.472 | 12.5514 / 66.641 | 17.6152 / 57.004 |
| mlp.down_proj | 0.0441 / 0.193 | 1.0049 / 6.091 | 1.1465 / 42.946 | 1.3018 / 10.463 | 1.1465 / 6.633 | 13.2649 / 60.984 | 12.0761 / 66.649 | 17.1450 / 58.245 |
| mlp.gate_proj | 0.0430 / 0.193 | 1.0049 / 6.024 | 1.1472 / 45.111 | 1.2962 / 10.583 | 1.1472 / 6.651 | 9.6079 / 60.811 | 9.0320 / 66.647 | 13.3556 / 58.786 |
| mlp.up_proj | 0.0438 / 0.193 | 1.0045 / 6.202 | 1.1460 / 44.926 | 1.2998 / 10.313 | 1.1459 / 6.628 | 8.5890 / 59.894 | 8.1107 / 66.646 | 12.2017 / 57.765 |
| self_attn.k_proj | 0.0380 / 0.186 | 1.0087 / 5.874 | 1.1525 / 6.581 | 1.2696 / 9.417 | 1.1525 / 6.581 | 11.8651 / 58.434 | 11.5802 / 66.611 | 16.4161 / 55.518 |
| self_attn.o_proj | 0.0380 / 0.188 | 1.0029 / 5.914 | 1.1456 / 44.003 | 1.2629 / 9.950 | 1.1456 / 6.627 | 13.0719 / 61.278 | 13.0975 / 66.627 | 18.7389 / 57.941 |
| self_attn.q_proj | 0.0387 / 0.187 | 1.0041 / 6.210 | 1.1474 / 42.968 | 1.2699 / 18.394 | 1.1474 / 6.641 | 11.1120 / 60.022 | 10.8422 / 66.636 | 15.5267 / 56.806 |
| self_attn.v_proj | 0.0378 / 0.190 | 1.0042 / 5.653 | 1.1478 / 28.930 | 1.2656 / 8.799 | 1.1478 / 6.532 | 11.3571 / 57.620 | 11.1547 / 66.606 | 15.6954 / 55.673 |

C2 is the best tail-MSE constrained arm in every measured family: **0.976–1.608×** fixture tail MSE, versus C3 **1.052–1.658×**, C1 **1.114–3.191×**, C4 **3.517–46.354×**, C4z **4.740–54.335×**, C4s **4.232–58.059×**. C3 generally preserves block maxima better than C2; therefore this is **not** a KLD ranking. Even C0's zero-only refit moves extrema and can worsen tails. Raw original minmax preservation must remain the comparison, not C0 relabelled as baseline. [Tail-ratio reductions](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/all_tail_relative.json).

## 6. Integer accumulation proof and epilogue

Proof values are independently evaluated in [range_proof.json](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/range_proof.json) and [K-domain extension](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/range_extensions.json). They are algebraic bounds, not measured quality claims.

Let `A_h=sum(qW*qX)`, `S_h=sum(qX)`, with unsigned weight code `qW in[0,15]`, signed activation code `qX in[-8,7]`. Then a128-half obeys `A_h in[-15360,13440]`, `S_h in[-1024,896]`. The exact real-valued result is

`y = sum_h d_h*(s_h*A_h + z_h*S_h)`.

**C1/C2/C3 do not permit whole-K integer accumulation:** their group bases are unrelated. Current `block_i4_128` also has unrelated activation d_h per 128, so even the within-group integer fold needs a compatible activation-scale constraint. Header width alone changes neither fact. C1's tight single-group bound is261120, C2's generic bound491520, C3's bound30720; after every group a floating fold is required unless stronger invariants are introduced.

For the paired row-master recipe, `s_h=B_row*2^ew_h` and `d_h=D_token*2^ex_h`, with `ew in[0,8]`, `ex in[0,2]`, hence combined exponent at most10:

- **C4:** `T=sum_h 2^(ew+ex)*A_h`; `Z=sum_h z_h*2^ex*S_h`; `y=B*D*float(T)+D*Z`. Free f16 zeros retain a per-half f32 reduction.
- **C4z:** `T=sum_h 2^(ew+ex)*(A_h+k_h*S_h)`; `y=B*D*float(T)`. Correct the half **before** accumulating/shifting; arbitrary k cannot be eliminated by signed-i4 recoding because `q+k` spans[-15,15].
- **C4s:** reinterpret each `q xor8` nibble as signed `q-8`; `T=sum_h 2^(ew+ex)*sum((qW-8)*qX)`; `y=B*D*float(T)`. Signed×signed product magnitude is at most64, not120.

| Constraint | Max integer multiplier | Abs bound K5120 | Abs bound K17408 | Legal whole-K interpretation |
|---|---:|---:|---:|---|
| C1 / C2 | 16 | 9,830,400 | 33,423,360 | Numeric box only; unrelated bases forbid this accumulation |
| C3 | 1 | 614,400 | 2,088,960 | Numeric box only; unrelated bases forbid this accumulation |
| C4 weights, hypothetical common X scale | 256 | 157,286,400 | 534,773,760 | Requires common activation scale, unlike today's Xq |
| C4 paired / C4z paired | 1024 | 629,145,600 | 2,139,095,040 | Safe under the frozen combined-shift domain |
| C4s paired | 1024 | 335,544,320 | 1,140,850,688 | Safe signed-i4 bound |

INT32_MAX is2,147,483,647. The paired affine/C4z K17408 bound is close to the limit: next256-aligned K17664 gives2,170,552,320 and is **not admitted**. Symmetric C4s permits at most32512 on a256-aligned conservative domain. Allowing activation exponent8 alongside weight exponent8 instead gives136,902,082,560 at K=17408 and is invalid. Allowing positive zero multipliers up to15 lets `q+k=30` and invalidates the same affine bound.

`A_h`, `k`, and `S_h` fit i24 before a MAD; the running total does not. Negative signed left shift is undefined in C++, so future kernels must use defined unsigned bit shifts/adds with the signed-range proof, not depend on C++ undefined behavior. No saturating WMMA is a substitute. f32 conversion/reassociation changes today's pinned `IU4_FOLD_RN` rounding DAG; real-arithmetic equality is not bitwise numerical parity.

## 7. Activation constraint and separate quality gate

The compatible activation arm, owned by Mq4e8ActB, is opt-in `HIPFIRE_IU4_XMASTER=1`: per token, find current independently MSE-fitted half scales, use master `D=max(d_h)/4`, then choose exponent0..2 by actual per-half SSE using unchanged rint/clamp; ties choose smaller exponent. It emits decoded `D*2^e` into the existing72-byte block's f32 d field, reassigns signed-i4 codes, and writes their exact integer sum. Zero token retains `d=1,q=0,S=0`. Rotation/AWQ remain unchanged.

A token-wide reduction must finish before emission, which rereads the **same original rotated f32 X**, not reconstructed int4 values. The implementation rewrites every master/rowzero cell on every preparation; it does not cache a master by X pointer. `ScratchState::int4_mmq_xmaster_scratch` owns 8 bytes/token (f32 master array then i32 rowzero array), grows without shrinking, and shares the source lifetime and stream ordering of the existing block buffer. `Gpu::ensure_int4_mmq_x` invalidates replay before either scratch allocation grows; `ScratchState::ensure_int4_mmq_x` bumps `int4_mmq_generation` before overwrite and returns the block pointer only after all three launches enqueue successfully. Thus old prepared producer handles become invalid; an error must not publish an unconstrained partial result. Fused producers are disabled while this arm is live. These source invariants do not constitute a retained-graph/replay test: all reported model gates use `HIPFIRE_GRAPH=0`. gfx1201 standalone is measured. Current code also admits gfx1100/gfx1151 with the common wave32 standalone fallback, **untested on those architectures**; other architecture strings leave the flag inactive. Reviewer owns acceptance or narrowing of that admission.

Weight-C4/C4z/C4s KLD on fp8 and activation-only KLD on unchanged-weight iu4 are separable necessary gates. **Their deltas are not an upper bound on joint KLD.** An eventual combined route needs its own model evaluation before adoption.

## 8. Performance evidence and explicit non-goals

All rates below are **register-resident probe ceilings**, not implemented-kernel performance or model-speed forecasts. Source/results supplied by Main are snapshotted in [register probe results](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/register_probe_results_snapshot.txt).

| Shape | Measured supplied probe class | Interpretation |
|---|---|---|
| Pure iu4 | P:677–735 TOPS | No scale/zero fold, no data-fed kernel claim |
| Today's exact f32 fold per 128 | A:338–348 TOPS | Existing affine fold comparison |
| C1/C2/C3 group-fold idea | C:431–450 TOPS | Conditional group-shared scale product; unchanged Xq does not satisfy it |
| C4 free zeros | V:435–455 TOPS, one343 outlier recorded | Shift fold plus per 128 zero FMA; ~450 class, not U |
| C4z integer free zeros | [INFERENCE] V-class packet model | Integer zero MAD remains; **no separately measured C4z kernel rate** |
| C4s symmetric | U:473–501 TOPS | Only no-zero shape matches U's shift-fold premise |
| Small-integer fold-only | S:426–449 TOPS | Not a C2 whole-K implementation |

The snapshot’s last speculative sentence that fully integer zeros would recover U is not a measurement and omits the general `k_h*S_h` correction. Only the fixed symmetric signed-recode arm removes that term.

C4s signed nibble conversion, real loads, masters/exponent access, activation reduction, register occupancy and epilogue must all be included in any later benchmark. No GEMM code is written in this study. No native compact-header decoder, gfx11 port, CDNA/CUDA port, new KV/recurrent-state algorithm, or default-route cutover is claimed. CPU artifact algebra transfers across architectures; only gfx1201 has model-quality evidence here.

## 9. Composer-ready source map and ownership

Base source anchors refer to b7169e4a9 unless an implementation receipt below supersedes them:

- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-quantize/src/quant_fwht.rs:37` `cpu_fwht_256`, `:63` `gen_fwht_signs`, `:200` `quantize_mq4g256v2`; header writes243–246, nibble pack265–269.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-quantize/src/float16.rs:22` `f32_to_f16`: preserve historical truncation.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-quantize/src/calibration.rs:195` `safetensors_to_ggml_name`, `:338` `imatrix_weights_for`, `:468` `compute_awq_scales`, `:535` `awq_pre_scale_weights`, `:548` sidecar emission: unchanged.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-quantize/src/pipeline.rs:315` format parsing; dense qt44 dispatch5862–5891 with AWQ call5880 and plain calls5884/5889. Other qt44 callers3613(LFM),4848(MoE),5238/5242/5247(promoted),5550/5733(fixed tier) must not silently acquire a dense-only study recipe. All497 actual fixture qt44 tensors, including lm_head, must be covered.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-runtime/src/hfq.rs:1686` `load_weight_tensor` and`:1772` `load_weight_tensor_pread` remain ordinary qt44 loaders; no decode substitution.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/kernels/src/block_i4_128_quant.hip:36` `quantize_block_i4_128_wave`,`:95` `emit_iu4_sidecar_from_producer8`,`:130` `quantize_int4_mmq_ds128`; `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/rdna-compute/src/scratch.rs` `ensure_int4_mmq_x` owns activation preparation. A fused producer wave does not own allK and cannot invent a local master.
- `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip:195` `IU4_FOLD_RN`, d/S loads519–524; gfx11 equivalent`:182`. Both remain untouched by this study.

Implemented activation anchors: `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/rdna-compute/src/feature_flags.rs:774` `FeatureFlags::iu4_xmaster_enabled`,`:792` `iu4_producer_sidecar_enabled`; `scratch.rs:180` master scratch,`:551` `int4_mmq_xmaster_needed`,`:1458` `ScratchState::ensure_int4_mmq_x`; `dispatch.rs:2969` `Gpu::ensure_int4_mmq_x`. Activation commit: `767a2b0db` (reported by owner). No KV/recurrent-state layout or lifetime changes are introduced.

**Independent executable slices, frozen interfaces:** quantizer owner A receives original already-AWQ f32 rows plus signs and explicit constraint; produces losslessly lowered qt44 and recipe metadata. Activation owner B owns only original-X master/requant preparation and its opt-in admission/scratch lifecycle, with existing72-byte output ABI. Study owner supplies independent expected bytes/metrics, serialized model evaluation and this document. Future compact loader and native GEMM are separate slices after quality; one integrator owns shared dispatch edits, and reviewer owns final veto. Full frozen contracts: [quantizer/composer contract](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/composer-contract.txt), [kernel/state invariants](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/kernel-contract.txt).

**Transitions/invariants:** immutable parent→AWQ f32→rotated row→frozen group/row scales→half-local Lloyd/final code assignment→validated private tensor output→complete artifact publication. A row master cannot be selected after any row group emits. An interrupted encode must not publish its final artifact. All non-qt44 tensors and AWQ sidecars remain byte-identical. Loader sees unchanged qt44 ABI. Activation master reduction and block emission are stream-ordered and share source lifetime/generation. Native integer accumulation may begin only after combined exponent/K admission. Final output applies masters, optional affine correction and existing residual/bias semantics.

**Claim-scoped gates:** independent encoder oracle before full requant; exact relation/f16 decode check over every group/row; complete tensor census and unchanged non-qt44 bytes; model+binary md5 per eval log; matched WT2/AG references and scoring flags; no fmt/lint/project-wide tests in this slice. Future integer/GPU code additionally needs adversarial sign/exponent/K-boundary oracles and actual combined-route KLD. Reviewers own the final veto.

**Abandon criterion:** failing the WT2 default-quality contract stops a default kernel/format project, regardless of register ceiling. Passing independent gates still does not establish combined quality. Even if quality passes, abandon an unmeasured throughput idea unless a full data-fed, matched-shape/matched-quality benchmark and model prefill improve without decode regression. This study tests the frozen recipes, not a theorem that every possible optimized row-master fit is impossible.

## 10. Implemented study flags and artifact receipts

`--format mq4e8-c0`, `mq4e8-c1`, `mq4e8-c2`, `mq4e8-c3`, `mq4e8-c4`, `mq4e8-c4z`, `mq4e8-c4s` are opt-in study formats. Original `mq4v2` and production routing defaults remain unchanged. The pipeline admits only dense Qwen model `arch_id=5` and K divisible by 256, rejects unknown suffixes, and has no fallback to an unconstrained recipe. This is a **model-architecture** restriction, not a GPU ISA check. GGUF input does not acquire these flags. CPU quantization is ISA-independent; no non-gfx1201 model-quality claim follows.

Implementation anchors (owner receipt): `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/crates/hipfire-quantize/src/quant_mq4e8.rs:24` `Mq4HeaderConstraint`; `:63` `Mq4e8Stats`; `:648` `quantize_mq4g256v2_constrained(w,m,k,signs1,signs2,constraint) -> Result<(Vec<u8>,Mq4e8Stats),String>`. Helpers are `base_half_steps:134`, `c1_scales:159`, `c2_scales:197`, `lloyd_free_zeros:239`, `lloyd_int_zero:301`, `assign_fixed_kminus8:375`, `choose_row_exp:470`, `row_master_b:491`. `pipeline.rs:325–334` parses the flags; `MainQuantFlags::mq4e8` owns the choice; the dense dispatch at approximately`:5886` calls the constrained encoder before the ordinary MQ4V2 branch. Existing embedding handling stays Q8. Metadata records `hipfire_base_format=mq4e8-cN` and `hipfire_mq4e8_study`. Measurement commits reported by owner: `2fd1416bb`, `200af2e48`, `6344f3a94`. [Full implementation receipt](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/IMPLEMENTATION_RECEIPT.md).

Every artifact below is **14,987,185,152 B**, with **497 qt44 /801 f16 /49 Q8 tensors**. Independent consumer-side verification checked all **200,171,520 half scales**, finite headers, C4z/C4s zero constraints, and the complete L0 in_proj_a oracle bytes. All **850 non-qt44 payloads (1,361,091,584 B)** remain byte-identical, including all 496 AWQ sidecars. Seven constraint artifacts all passed. [Artifact manifest and proof links](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_manifest.json); [exhaustive verifier source](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/verify_artifact.py). These checks do not claim exhaustive code-byte parity for every emitted coefficient; the independent seven-arm full-tensor oracle covers L0 in_proj_a, while full model evaluation covers the selected quality arms.

| Arm | Artifact (absolute path) | md5 | Encode log / independent header proof |
|---|---|---|---|
| C0 | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c0.hfq` | `ccf14c7cbfe2e7602f22e56a2463f170` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c0.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C0_header_verification.json) |
| C1 | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c1.hfq` | `3676676ea4e6f135926675a93e267075` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c1.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C1_header_verification.json) |
| C2 | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c2.hfq` | `6582bc9b0d8cb4915c877711bc2a1d02` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c2.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C2_header_verification.json) |
| C3 | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c3.hfq` | `fac22847faa912ea1cd9c150ea9e91dd` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c3.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C3_header_verification.json) |
| C4 | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c4.hfq` | `e5a522a0a27a779c3db15949010e0178` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c4.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C4_header_verification.json) |
| C4z | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c4z.hfq` | `0c0278958f144bd4af5e9a5354e5679c` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c4z.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C4z_header_verification.json) |
| C4s | `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4e8-c4s.hfq` | `48e3d81b49e1723b57de1b1f6471b1c5` | [encode](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/requant_c4s.log) / [all headers](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/artifact_C4s_header_verification.json) |


Requant recipe: original parent, `--format mq4e8-cN --imatrix /home/kaden/qcal/imatrix/Qwen3.8-27B-imatrix.gguf --awq-alpha 0.55 --tier xt`, no vision inclusion. Exact runnable invocations are the adjacent `requant_cN.cmd` files. Clamp/floor report: C2 multiplier high-cap 769 groups, low-cap 0, base-floor 226; C4/C4s master-floor 3 rows; C4z master-floor 28 rows; zero-master rows 0. [Independent encoder oracle](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/quant-a/gate1_oracle.log) recorded Rust 7/7 and Python 7/7 exact matches. Owner ran only scoped quantizer/pipeline gates, not project-wide validation. Study-only oracle/dump harnesses and their measurements belong in untracked evidence, not permanent machine-local test dependencies; final code integration/cleanup and review are Main's responsibility.

## 11. Tree-built model-quality measurements

### Weight-only evaluation provenance

All weight rows use the immutable tree-built evaluator `/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_hipfire.weight`, **md5 `8b2d99f611fd201dda6c0199f3ae3eba`**. Build command: `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet`, from this worktree before activation changes. The first build's stdout was not duplicated into evidence; the successful owner build receipt is retained in [weight_eval_build.txt](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/weight_eval_build.txt). Actual executable hash is checked by the launcher on every run. This is not a prebaked baseline evaluator.

References: WT2 `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`, md5 `8a21364051d844b97c122e2c895f56d8`; AG `/home/kaden/kldrefs/qwen3.8-27b.ref_ag.bin`, md5 `b63d3bc13e3ea294e6c938050a07d36f`.

All runs pin `HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels HIPFIRE_IU4_PREFILL=0 HIPFIRE_IU4_XMASTER=0 HIPFIRE_GRAPH=0`, after clearing inherited `HIPFIRE_*` variables. Scoring is `--kv-mode q8 --kv-v q8 --scoring-mode prefill --max-chunks N`. Thus weights alone change; current fp8-prefill/GEMV consume their exact lowered reconstruction. Every log contains the full command, model/reference/binary hashes, environment, KLD and exit0. [Launcher](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/run_weight_eval.py); [raw row manifest](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/weight_eval_results.jsonl); [independent log/output-hash verification of all 18 rows](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/weight_eval_verified.json).

| Arm | Corpus | Chunks | KLD | Model md5 | Output-bin md5 | Log |
|---|---|---|---|---|---|---|
| fixture_fp8 | WT2 | 1 | 0.029580 | `1a48b45ebf5f3ef852efcf8ee5d8084b` | `e091d74e5f290efb9bc343f107a716b4` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_fixture_fp8_wt2_1.log) |
| fixture_fp8 | WT2 | 2 | 0.036694 | `1a48b45ebf5f3ef852efcf8ee5d8084b` | `3834e25a771ecfb9e4f1a02383aa3fe0` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_fixture_fp8_wt2_2.log) |
| fixture_fp8 | WT2 | 24 | 0.048659 | `1a48b45ebf5f3ef852efcf8ee5d8084b` | `deff6ed3ac7c600c3ec5e623f58db5da` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_fixture_fp8_wt2_24.log) |
| fixture_fp8 | AG | 24 | 0.141396 | `1a48b45ebf5f3ef852efcf8ee5d8084b` | `cf36f1766904527fbfe8f2a14689c39d` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_fixture_fp8_ag_24.log) |
| C0 | WT2 | 1 | 0.031786 | `ccf14c7cbfe2e7602f22e56a2463f170` | `6864d92da2637f44afc73d3b86369c9c` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C0_wt2_1.log) |
| C0 | WT2 | 2 | 0.037332 | `ccf14c7cbfe2e7602f22e56a2463f170` | `6b79e817c2fe3855ed554949f542801e` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C0_wt2_2.log) |
| C0 | WT2 | 24 | 0.047097 | `ccf14c7cbfe2e7602f22e56a2463f170` | `c16d3a7e8dfaeda96f7ea37b7c056aa7` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C0_wt2_24.log) |
| C0 | AG | 24 | 0.155170 | `ccf14c7cbfe2e7602f22e56a2463f170` | `b16a7191179c3acf0f238f507c8b7a92` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C0_ag_24.log) |
| C4 | WT2 | 1 | 0.066757 | `e5a522a0a27a779c3db15949010e0178` | `95411566cd19535c00a38cbd6ce5dae4` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C4_wt2_1.log) |
| C4 | WT2 | 2 | 0.079456 | `e5a522a0a27a779c3db15949010e0178` | `ecdbc4f39398831b970e382ae087a9b1` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C4_wt2_2.log) |
| C4 | WT2 | 24 | 0.097665 | `e5a522a0a27a779c3db15949010e0178` | `bdec3d0c332a3662db2659d4343a1945` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C4_wt2_24.log) |
| C4 | AG | 24 | 0.214142 | `e5a522a0a27a779c3db15949010e0178` | `b0d79b49cb90d33fb0b474785532f697` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C4_ag_24.log) |
| C2 | WT2 | 1 | 0.029642 | `6582bc9b0d8cb4915c877711bc2a1d02` | `9d9036d542da8f8f5f5493f264022f70` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C2_wt2_1.log) |
| C2 | WT2 | 2 | 0.035003 | `6582bc9b0d8cb4915c877711bc2a1d02` | `331ddf068bb5341dba2a9b37cf3213d4` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C2_wt2_2.log) |
| C2 | WT2 | 24 | 0.045629 | `6582bc9b0d8cb4915c877711bc2a1d02` | `2fedc607021b849961031a0c5a31bfaa` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C2_wt2_24.log) |
| C2 | AG | 24 | 0.137719 | `6582bc9b0d8cb4915c877711bc2a1d02` | `b92c39ac75e366c41deef939392e1080` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C2_ag_24.log) |
| C3 | WT2 | 24 | 0.049713 | `fac22847faa912ea1cd9c150ea9e91dd` | `9a58dc6d98d688b2b993319eb540af88` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C3_wt2_24.log) |
| C3 | AG | 24 | 0.161606 | `fac22847faa912ea1cd9c150ea9e91dd` | `576e3588ad6aca0ccab5f5e7a0796fa7` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/eval_C3_ag_24.log) |


The fresh fixture WT2_24 **0.048659** exactly reproduces the supplied fp8 baseline. C3 was run at 24 chunks directly under Main's final narrowed gate. **C1/C4z/C4s model evaluations were explicitly canceled by Main after the coefficient screen; none is represented as a measured KLD failure.** Their artifacts, all-layer coefficient metrics and algebra remain complete. C4's already-started full ladder and AG result are retained, not discarded.

### Weight quality decision

| Arm | WT2_24 | Δ WT2 | AG_24 | Δ AG | WT2 default gate |
|---|---|---|---|---|---|
| C0 | 0.047097 | -0.001562 | 0.155170 | +0.013774 | PASS |
| C2 | 0.045629 | -0.003030 | 0.137719 | -0.003677 | PASS |
| C3 | 0.049713 | +0.001054 | 0.161606 | +0.020210 | FAIL |
| C4 | 0.097665 | +0.049006 | 0.214142 | +0.072746 | FAIL |


C2 is the strongest constrained weight recipe here: WT2 improves **0.003030**, AG improves **0.003677**; it passes the default-quality threshold by **0.003530**. C3 exceeds the threshold by **0.000554** and worsens AG. C4's row-master cost is large on both corpora. C0 illustrates why the zero-refit control matters: WT2 improves but AG worsens **0.013774**. Tail/max-coef are screens, not a substitute for matched model evaluation or proof that family-MSE ranking monotonically predicts KLD. No overall-MSE/Frobenius ranking was used.

### Separate activation-only gate

All following rows use **unchanged fixture md5 `1a48b45ebf5f3ef852efcf8ee5d8084b`**, iu4 prefill, and activation-arm tree evaluator **md5 `a92a527fa9c1f1509aedba26a330cbbf`**. The activation build log is [build-eval-xm.log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/build-eval-xm.log). Owner pins `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1`, adding `HIPFIRE_IU4_XMASTER=1` only for the constrained rows; q8/q8 prefill scoring and references are unchanged. [Full activation receipt](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/RESULTS-slice-b.md); [independently verified logs and output hashes](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/activation_eval_verified.json).

| Arm | Corpus | Chunks | KLD | Output-bin md5 | Log |
|---|---|---|---|---|---|
| iu4 baseline | WT2 | 1 | 0.041329 | `032ebad84f2c1dd5e2980fa805215ac0` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/base_c1.stderr) |
| iu4 baseline | WT2 | 2 | 0.048091 | `dc7e53181662271780374f0a85fd7732` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/base_c2.stderr) |
| activation master | WT2 | 1 | 0.044127 | `9cfb377406c7a50d9565e79447c08750` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/xm_c1.stderr) |
| activation master | WT2 | 2 | 0.050674 | `f282c3e574e4ef1d9f2becfede8a8a6a` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/xm_c2.stderr) |
| activation master | WT2 | 24 | 0.070464 | `410c38048ffb7129c934f07b5c55b052` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/xm_c24.stderr) |
| iu4 baseline | AG | 24 | 0.181992 | `1966f2a9b2fe25ff46428231a7181943` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/base_ag24.stderr) |
| activation master | AG | 24 | 0.203572 | `e0cf90b1c171735eb1b93efdc91575dd` | [log](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/xm_ag24.stderr) |


The **0.063410 iu4 WT2_24 baseline is the supplied contract**, not a newly run baseline row in this study; no corresponding fresh log is invented. The new activation-only WT2_24 result **0.070464** costs **+0.007054** against that contract, exceeding its 0.063910 default gate by **0.006554**. Fresh matched AG rows worsen from **0.181992** to **0.203572**, **+0.021580**. Flag-off WT2_1/2 output hashes exactly reproduce the prior pins. The device oracle checks decoded d, exponent selection, all q and exact integer sums: e histogram 27/1/36, zero mismatches ([oracle](/home/kaden/ClaudeCode/warpfront/wt-mq4e8/scratch-2026-09-17/Mq4e8Study/slice-b/verify2.stderr)). This does not bound joint weight+activation KLD.

Activation preparation timing is measured overhead, not a GEMM speedup: N512, K5120 **88.9→143.5us**; K17408 **215.4→376.3us**, sync-delimited wall medians over 20 calls. rocprof attribution at K=5120: pass1 **65.7us**, reduce **26.5us**, fixup **24.2us**. These are owner measurements in the linked receipt; no full-model latency result is claimed.

## 12. Recommendation and closure

1. **Keep current default format/routes. Retain C2 as an opt-in weight recipe and a quality-passing control for any later group-fold study.** It passes the stated weight gate on this fixture and improves both corpora. This is a quality result, not a throughput result or a general claim across model families/GPUs.
2. **Do not adopt the frozen whole-K row-master path.** C4 weights and the independent compatible activation arm each fail their required default-quality gates. C4z/C4s were screened out without KLD under Main's explicit decision. C4s is the only proposed arm with the no-zero U-style algebra, and its all-family tail/max-coef screen is poor. No joint route or native compact-header GEMM was built or benchmarked.
3. **C3 is not a default candidate under the stated contract.** Its +0.001054 WT2 cost is above +0.0005, with corroborating AG degradation. A quality/speed opt-in beyond the contract is Kaden's call, but this study has no data-fed speed result to justify that trade.
4. **Do not mistake C2's pass for an integer-fold implementation.** Current activation half scales remain independent; group bases also vary across K. C2 alone therefore buys neither the proposed whole-K integer epilogue nor a proven group-fold speedup. A future compatible group-level activation constraint is a distinct, unmeasured hypothesis; the failed row-master arm does not validate or disprove it.

The completed deliverable is the coefficient/header census, exact study encoders and seven artifacts, scoped runtime state recipe, overflow/epilogue proof, 18 weight-quality rows, 7 activation-quality rows, and this recommendation. No production-default change is made. Performance exploration must stop at the claim-scoped quality and data-fed benchmark gates in section 9; favorable register ceilings alone do not justify implementation. Reviewer retains the final veto on source integration, architecture admission and cleanup.
