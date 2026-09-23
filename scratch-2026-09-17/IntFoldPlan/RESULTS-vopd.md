# Iu4Vopd — gfx1201 VOPD fold pairing: ADMITTED (all gates pass)

Commit `7e33ea6a6` on branch `gfx12-iu4-k32`; sole production file touched:
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip` (+47/−0).
Kernel SHA256 `cbd98f6db133ef01d2a617f79aea53f83d9aad05b65273faa2f71f2beb06dc00`.
Form 1 of the plan's §3 paired-fold experiment (helper + j/j+1 interleave);
no second form needed — ISA, oracle, TIME and gate 4 all pass on form 1.

## Change

- New `iu4_fold_rn_pair` (`__gfx1201__`-guarded): two independent
  j-components through the pinned `IU4_FOLD_RN` DAG with stages interleaved
  pairwise (t1_0,t1_1 / p_0,p_1 / t2_0,t2_1 / term_0,term_1 / sum_0,sum_1).
  Same RN mul/FMA/add nodes, same per-component order, no contraction;
  i32→f32 conversions stay scalar (no dual convert exists). By-value sums +
  float2 return (ext_vector elements cannot bind to `float&`); `__forceinline__`.
- Fold call site uses j-step-2 paired loop on gfx1201; sf hoisted per nb
  (loop-invariant; baseline CSEs to the same 4 converts/half — census
  confirms 68 cvt before and after). gfx1200 path keeps the scalar macro loop.
- Staging, WMMA bundles, barriers, LDS layout, ABI, launch geometry, host:
  untouched.

## Gate 1 — oracle: PASS 9/9 (`oracle-cand.log`, exit=0)

`cpu_bitwise=OK (mism=0)` + `repeat_identical=OK (mism=0)` on all nine
(gate/set, gate/add, down/set, down/add, m48tail, n80cols, k256/set,
k256/add, m100n100). Binaries md5: base `72096784…`, cand `c7a2000a…`
(`gate4-md5.txt`; release: hipfire `fd91d9ac…`, daemon `d1ce229e…`,
eval_hipfire `4554544f…`; OFF wt-lloyd binary `1dd5f084…`, used read-only).

## Gate 2 — metadata: PASS (production TU + JIT agree)

| entry | VGPR | SGPR | vspill/sspill/scratch | OCC (block 256, LDS 19456) |
|---|---|---|---|---|
| base (all 3) | 202 | 43/40/40 | 0/0/0 | — (landed state: 3 WG/CU) |
| cand base/disp | 202 | 43 | 0/0/0 | — |
| cand full_add | 202 | 40 | 0/0/0 | 3 WG/CU (`occ-cand.log`) |
| cand full_set | 202 | 40 | 0/0/0 | 3 WG/CU |

VGPR ≤ 202 (current), 0 spill/scratch, 3 WG/CU both entries. JIT
`radiowave.json` source_sha256 == candidate TU sha; cached `.hip`
byte-identical to prelude + candidate kernel (`bd779269c415f5a0`).

## Gate 3 — ISA census (`vopd_census.py`): PASS, exact-op parity

| symbol | fold packets base→cand | fold scalar ops | cvt |
|---|---|---|---|
| full_set | 315 → **286 (−29, −9.2%)** | 388 → 388 identical | 68 → 68 |
| full_add | 379 → **350 (−29, −7.6%)** | 452 → 452 identical | 68 → 68 |

Fold dual halves: mul 95→122, fmac 26→39, add 25→43 (73→102 dual lines).
Whole-body instr lines drop by exactly 29 in both symbols: zero copy/move
overhead. No `v_fma_f32` mix change (34/34 both). fmac::fmac adjacency
spot-checked in disasm (v24/v25, v26/v27).

## Gate 3 — TIME same session, 5 interleaved brackets (a–e, order alternates)

Fresh process per binary per bracket, ordinal 1, N=512 rows. Candidate wins
all 25 bracket×row cells, no regressions.

| row | base a/b/c/d/e (med) | cand a/b/c/d/e (med) | Δ |
|---|---|---|---|
| gate set | 491.1/493.8/496.6/499.2/499.0 (496.6) | 485.5/477.5/482.7/477.8/483.2 (482.7) | −2.8% |
| gate add | 513.9/517.4/520.3/521.7/520.3 (520.3) | 505.3/505.5/508.4/507.8/507.7 (507.7) | −2.4% |
| down add | 506.1/508.1/505.7/509.1/507.1 (507.1) | 490.8/490.7/493.3/492.7/489.5 (490.8) | −3.2% |
| qkvza set | 487.9/489.8/489.9/491.1/492.3 (489.9) | 474.4/476.8/473.8/472.8/475.5 (474.4) | −3.2% |
| qkv set | 403.2/408.5/407.9/407.4/406.6 (407.4) | 392.7/391.2/393.9/394.2/396.1 (393.9) | −3.3% |

Mean per-row gain **2.98%** ≥ 1.5% abandon threshold → ADMITTED. (>1.5% on
every prefill row = real per project rule.) The plan's 400 us stretch target
is not met (gate/set 482.7) — reported honestly as below target, not claimed.

## Gate 4 — pins, bench, decode, serve: PASS

- KLD pins under HIPFIRE_IU4_PREFILL=1 (`gate4/kld-pins.sh`): c1
  `032ebad84f2c1dd5e2980fa805215ac0` ✓ (KLD 0.041329), c2
  `dc7e53181662271780374f0a85fd7732` ✓ (KLD 0.048091), c24 slice-mean KLD
  **0.063410** ✓ — all exact, no repin.
- Interleave bench OFF/ON/OFF/ON (`gate4/bench-interleave.sh`, GRAPH=1
  LLOYD=1, rebuilt ON hipfire `fd91d9ac…` + daemon `d1ce229e…`):

| arm | pp512 | pp2048 | pp8192 | pp32768 | decode |
|---|---|---|---|---|---|
| off1 (29-state, pp2048+ not pairable) | 1484.8 | 1412.2 | 1304.1 | 1011.7 | 29.13 |
| on1 | 2189.1 | 2116.9 | 1897.7 | 1337.9 | 36.49 |
| off2 | 1480.5 | 1452.9 | 1349.0 | 1039.1 | 36.49 |
| on2 | 2189.7 | 2115.5 | 1896.9 | 1336.6 | 36.52 |

Valid 36.5-state pairs (off2 vs on1/on2): ON is tightly self-consistent
(≤0.1% arm spread). Prior ON (2151/2086/1874/1327) → current ON
(~2189/~2116/~1897/~1337): +1.8/+1.4/+1.2/+0.8%, same direction as TIME,
smaller end-to-end as expected. Cache identity: bench JIT
`gemm_mq4g256v2_residual_mmq_iu4_gfx12.bd779269c415f5a0.hip` carries 2×
`iu4_fold_rn_pair` + fetch-epoch markers and is byte-identical to
prelude + candidate kernel → ON ran the candidate.
- Decode-only (noslots stateless, 5 runs/3 warmups, 128 tok, separate run):
  OFF 36.7, ON 36.6 — both ≥ 36.4. Decode unaffected (prefill-only change).
- Serve battery + chain (ON daemon `d1ce229e…`, GRAPH=0 IU4_PREFILL=1):
  battery 5/5 stop coherent (runaway=0 empty=0 attractor=0);
  chain 3× stop + 2× length-budget turns with clean coherent tails
  (runaway=2 length-heuristic only, empty=0 attractor=0 retrieval_miss=0).
  `connection closed` teardown lines match baseline logs (benign). No
  repetition/attractor/stream errors. Full texts in
  `serve-battery.json`/`serve-chain.json`.

## Verdict

Land: exact arithmetic, −29 fold packets, +2.98% mean TIME gain, pins exact,
bench moves the right direction, decode ≥ floor, serve coherent.

## Files (`scratch-2026-09-17/iu4vopd/`)

- `vopd_census.py`, `census-base.txt`, `census-cand.txt`
- `tu-base.hip/dis`, `tu-cand.hip/dis`, `vgpr-base/cand.txt`, `cand-kernel.sha`
- `oracle-base/cand` (binaries, md5s), `build-*.log`, `oracle-cand.log`, `occ-cand.log`
- `time-brackets.sh`, `time-base/cand-{a..e}.log`
- `gate4/`: `kld-pins.sh`, `c{1,2,24}.bin/.stderr`, `bench-interleave.sh`,
  `bench-{off,on}{1,2}.json/.stderr`, `decode-{off,on}.json/.stderr`,
  `serve-{battery,chain}.json`, `serve-{battery,chain}-{harness,serve}.log`
- `RESULTS-vopd.md` (this file)
