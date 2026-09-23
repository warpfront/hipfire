# MQ4E8 slice B (activation-only paired-C4) — results, 2026-09-18

Owner: Mq4e8ActB. Branch `mq4e8-study`. Worktree `/home/kaden/ClaudeCode/warpfront/wt-mq4e8`.
GPU ordinal 0, env `HOME=/home/kaden/.hipfire-homes/ab0 ROCR_VISIBLE_DEVICES=0
HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab0/.hipfire_kernels`.

## What was built

Env flag `HIPFIRE_IU4_XMASTER=1` (default off) behind `FeatureFlags::iu4_xmaster_enabled`
(arch-gated gfx1100/gfx1151/gfx1201, same as iu4_prefill). When live, the standalone
`quantize_int4_mmq_ds128` output is post-processed on the same stream, reading the
original f32 X:
- pass 2a `int4_mmq_xmaster_reduce` (one wave32/token): `master[token] = max block d / 4`
  over the full K row (exact f32 `*0.25`), `rowzero[token]=1` iff the row is exactly zero
  (zero rows keep master 1.0 so pass-1 `d=1/q=0/s=0` bytes survive);
- pass 2b `int4_mmq_xmaster_fixup` (pass-1 grid): per 128-block `e` in 0..2 minimizing
  half SSE at `master*2^e` with the same rintf/clamp/fmaf recipe (strict `<`, ties take
  the smaller e); decoded `master*2^e` stored in the existing f32 `d` field, `q`
  recomputed, `s = sum q`. iu4 consumer untouched. `e_x<=2` keeps the paired shift
  `e_w+e_x<=10` inside the int32 bound at K=17408 (17408*15*8*2^10 = 2139095040).
- `iu4_producer_sidecar_enabled()` returns false when xmaster is live, so fused
  producers route back through the standalone quantizer — no block silently
  remains unconstrained. Scope exercised: gfx1201 standalone.

Files: `kernels/src/block_i4_128_quant.hip` (+118, new kernels only; pass-1 untouched),
`crates/rdna-compute/src/{feature_flags.rs(+27),scratch.rs(+154),dispatch.rs(+18)}`,
new oracle `crates/rdna-compute/examples/tmp_iu4_xmaster_oracle.rs`.

## Gate 1 — unset-flag byte identity: PASS

- Flag-off eval bins are md5-identical to the iu4 pins: c1 `032ebad84f2c1dd5e2980fa805215ac0`,
  c2 `dc7e53181662271780374f0a85fd7732` (KLD 0.041329 / 0.048091 reproduced exactly).
- Pass-1 kernel function textually unchanged (`git diff` shows only additions).
- Default `mq4v2` weight bytes: untouched by construction — the diff touches no file
  under `crates/hipfire-quantize/` (`git diff --name-only` proves it); the pin-md5
  reproduction above proves the full default pipeline end to end.

## Gate 2 — device oracle: PASS

`tmp_iu4_xmaster_oracle verify base_off xm_on` (n=8, k=1024, zero/const/spike/tiny/
multi-scale tokens): `e histogram: e0=27 e1=1 e2=36`, `d_fail=0 q_fail=0 s_fail=0
e_fail=0` → `ORACLE PASS`. Checks: every stored `d == master*2^e` bitwise for e in
0..2 (master from the pre-fixup dump, exact f32 ops); `q`/`s` recomputed from X and
the stored `d` (round-half-even replica) match nibbles/`s` bitwise; stored `e` matches
an exact CPU replica of the wave-tree half-SSE argmin (incl. shuffle order).

## Gate 3 — KLD (activation arm, unchanged MQ4V2 weights): rows for the planner

Model `/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt` =
`/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq`
(md5 `1a48b45ebf5f3ef852efcf8ee5d8084b`, 14987185152 B, `check_fixture.sh --sha` OK).
Binary `target/release/examples/eval_hipfire` md5 `a92a527fa9c1f1509aedba26a330cbbf`
(build log `slice-b/build-eval-xm.log`).
Env: `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1 HIPFIRE_IU4_PREFILL=1`
(+ `HIPFIRE_IU4_XMASTER=1` for xm rows), `--kv-mode q8 --kv-v q8 --scoring-mode prefill`.
Refs: WT2 md5 `8a21364051d844b97c122e2c895f56d8`, AG md5 `b63d3bc13e3ea294e6c938050a07d36f`.

| row | ref | chunks | slice-mean KLD | NLL | PPL | out-bin md5 |
|---|---|---|---|---|---|---|
| base c1 | WT2 | 1 | 0.041329 | 2.198164 | 9.0085 | 032ebad84f2c1dd5e2980fa805215ac0 |
| base c2 | WT2 | 2 | 0.048091 | 2.256271 | 9.5474 | dc7e53181662271780374f0a85fd7732 |
| xm c1 | WT2 | 1 | 0.044127 | 2.192788 | 8.9602 | 9cfb377406c7a50d9565e79447c08750 |
| xm c2 | WT2 | 2 | 0.050674 | 2.254728 | 9.5327 | f282c3e574e4ef1d9f2becfede8a8a6a |
| xm c24 | WT2 | 24 | 0.070464 | 1.871535 | 6.4983 | 410c38048ffb7129c934f07b5c55b052 |
| base ag24 | AG | 24 | 0.181992 | 2.081463 | 8.0162 | 1966f2a9b2fe25ff46428231a7181943 |
| xm ag24 | AG | 24 | 0.203572 | 2.078284 | 7.9907 | e0cf90b1c171735eb1b93efdc91575dd |

Deltas (xm − base): WT2 c1 +0.00280, c2 +0.00258, c24 +0.00705; AG c24 +0.02158.
Per plan: complete WT2 1/2/24 rows reported even though the cost is nonzero; this is the
activation-half cost only and MUST NOT be presented as an upper bound of combined
weight+activation KLD — joint eval comes later.

## Gate 4 — quant-kernel TIME at N=512: PASS (report)

Method: HIP-event-free sync-delimited wall median over 20 `ensure_int4_mmq_x` calls in the
throwaway oracle (`time` mode), plus rocprofv3 `--kernel-trace` attribution (both per
Main's brief; trace DB `slice-b/prof-k5120v2/`).

| K | flag off (med) | flag on (med) | Δ |
|---|---|---|---|
| 5120 | 88.9 us | 143.5 us | +54.6 us |
| 17408 | 215.4 us | 376.3 us | +160.9 us |

rocprofv3 medians at N=512,K=5120: pass-1 65.7 us, reduce 26.5 us, fixup 24.2 us.
Note: the first reduce version (one thread/token, uncoalesced row scan) cost 406 us;
rewriting it wave-cooperative (coalesced, [N,1,1]x[32,1,1]) cut it to 27 us with
bit-identical oracle output (re-verified PASS after the rewrite). The xmaster Δ stays
well under the ~500 us GEMM reference at both K values.

## Bound proof

`e_w<=8 + e_x<=2` → combined shift `<=10`; int32 worst case at K=17408:
`17408*15*8*2^10 = 2139095040 < 2^31`. Oracle asserts `e<=2` on every block (e_hist above).
