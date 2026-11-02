# MQ4-Lloyd (qt=52) gfx11 port: gfx1100 (RDNA3 dGPU) + gfx1151 (Strix Halo)

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`, HEAD `11953ba3d`.
Read-only investigation; this plan is the only write. No GPU program was run.
All paths below are worktree-absolute; line numbers are HEAD-pinned (`:line`).

Product decision (taken, not revisited): **xt stays uniform (+iu4) for speed on gfx11;
base/pro tiers get Lloyd.** §3 states per-tier routes and the accepted prefill cost.

> **STATUS 2026-09-15 — T0 gate FAILED; F16-LUT prefill direction abandoned.**
> XTX (gfx1100), residual K=17408 M=5120, `tmp_lloyd_gfx11_diff --time`:
> F16-LUT twin 2224 us vs uniform F16 base tile 1378 us at N=384 (+61%);
> 2986 vs 2175 at N=511 (+37%); 2980 vs 1741 at N=512 where uniform routes to
> MMQ (+71%). Abandon criterion was +10%. Correctness gate passed (rel-L2
> 2.7e-4 vs LUT GEMV, N=511/512). T2-T6 not built; T0 code reverted; T1
> (gfx1151 fused-decode admission) kept. On gfx11 a qt=52 artifact decodes
> through the scalar LUT kernels; batched prefill fails closed (gemm.rs
> `_lloyd` arms) — Lloyd tiers are gfx12-only until a prefill route that does
> not cost gfx11 MMQ/iu4 throughput exists.

## 0. Amdahl framing (denominator first)

- Decode on gfx11 is dominated by per-launch GEMV/FWHT/attention work, not by the
  LUT decode itself: the LUT adds 8 kernarg dwords + 32 B LDS + one barrier per
  row (`kernels/src/gemv_mq4g256v2.hip:78-94`), against a ~136 B/row weight stream.
  The rows=1-vs-2 decode cost on a qt44 artifact measured +0.35% (noise)
  (`crates/rdna-compute/src/gemv.rs:7819-7822`). Honest ceiling for the decode
  slice of this plan: low-single-digit % tok/s; its value is correctness/coverage
  (fused decode on Halo), not speed.
- Prefill is where the money is: on gfx12 the FP8-vs-F16 prefill gap is
  1387 vs 841 tok/s @pp512 (`CHANGELOG.md:12`), i.e. prefill GEMM choice moves total
  throughput by tens of percent. On gfx11 the uniform production route above the
  MMQ cutoff is integer MMQ, which Lloyd cannot use (integer-dot dots raw codes;
  no codebook input). So the Lloyd prefill ceiling on gfx11 is the F16-WMMA rate,
  and the plan prices that explicitly in §3. Transfer-loss warning: every number
  in §3 is measured against hipfire's tuned uniform path (X128 MMQ / iu4), never
  against a naive baseline.

## 1. Decode admission on gfx1100 and gfx1151

### 1.1 What already admits Lloyd decode on each arch

Plain / residual / SwiGLU-residual per-projection LUT GEMVs are arch-gated by
`ArchPredicate::HasWave32` for `MQ4G256V2Lloyd`
(`crates/hipfire-dispatch/src/types.rs:913-914`), i.e. **admitted on both gfx1100
and gfx1151 today** (wave32 covers all of rdna3/rdna4). The three launchers:

- `Gpu::gemv_mq4g256v2_lloyd` — `crates/rdna-compute/src/gemv.rs:7824-7844`,
  over `GEMV_MQ4G256V2_LUT_SRC`
  (`crates/rdna-compute/src/kernels.rs:1578-1581`: `-DHIPFIRE_MQ4G256V2_LUT=1`,
  distinct symbol `gemv_mq4g256v2_lloyd`). LUT = 8 dwords f16 staged to LDS
  (`kernels/src/gemv_mq4g256v2.hip:78-94`), nibble decode via `DOG_LUT`
  (`kernels/src/gemv_mq4g256v2.hip:115-128`). K%256 fail-closed (`gemv.rs:7834-7839`).
- `Gpu::gemv_hfq4g256_residual_mq4v2_lloyd` (`y +=`) — `gemv.rs:7908-7929`, over
  `GEMV_MQ4G256V2_RESIDUAL_LUT_SRC` (`kernels.rs:1584-1587`, symbol
  `gemv_mq4g256v2_residual_lloyd`).
- Fused decode launchers (live in `gemm.rs`, not `gemv.rs`):
  `fused_gate_up_mq4g256v2_lloyd` (`gemm.rs:32277-32298`),
  `fused_qkvza_mq4g256v2_lloyd` (`gemm.rs:32356-32382`),
  `fused_qkv_mq4g256v2_lloyd` (`gemm.rs:32455-32478`), over
  `FUSED_GATE_UP_MQ4G256V2_LUT_SRC` (`kernels.rs:5386-5389`),
  `FUSED_QKVZA_MQ4G256V2_LUT_SRC` (`kernels.rs:4262-4267`),
  `FUSED_QKV_MQ4G256V2_LUT_SRC` (`kernels.rs:4606-4609`).
  Fused LUT arg shape: 16 dwords gate+up (`kernels/src/fused_gate_up_mq4g256v2.hip:40-47`),
  staged per-row-source to LDS (`:60-74`); QKV 24 dwords, QKVZA 32 dwords (see
  `kernels.rs:4260-4261`, `kernels.rs:4604-4605`).
- Dispatch: `launch_fused` matches the three `*Mq4G256V2Lloyd` keys directly and
  pulls per-tensor `[u32; 8]` decode LUTs off the `WeightRef`s
  (`crates/hipfire-dispatch/src/pipeline/steps.rs:1088-1142`), failing closed on
  `lloyd_lut_f16 == None` (`steps.rs:1090-1094`). DType→key mapping admits Lloyd
  to the prerotated / residual / swiglu_residual GEMV keys
  (`types.rs:756-757`, `:813-814`, `:855-856`); rotation plan is FWHT-G256
  (`types.rs:122-126`); GEMV variant is `Prerotated` (`types.rs:154-157`).

### 1.2 What is missing: fused decode is gfx1100+gfx1201-only

All six V2 scalar-fusion guards funnel through one predicate:

```rust
fn mq4g256v2_scalar_fusion_ok(ctx: &DispatchCtx) -> bool {
    !ctx.flags.force_unfused && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1201())
}
```

(`steps.rs:161-165`; Lloyd guards `guard_qkv_mq4g256v2_lloyd`,
`guard_qkvza_mq4g256v2_lloyd`, `guard_gate_up_mq4g256v2_lloyd` at
`steps.rs:184-196`, each additionally requiring all-`MQ4G256V2Lloyd` +
`lloyd_lut_f16.is_some()` + `Prerotated` + no-AWQ via
`gemv_steps_uniform_mq4g256v2_lloyd`, `steps.rs:171-180`.)

Consequence on **gfx1151 today**: the fused patterns
(`steps.rs:564-567`, `:596-599`, `:634-637`) never fire; every Lloyd layer falls
through to per-projection LUT GEMVs. That path is correct (same kernels, same
LUTs) and costs 1–3 extra launches per layer per token — the §0 ceiling applies.
Nothing fails closed here as long as the sidecar exists; a missing sidecar fails
closed in the gemv family (`crates/hipfire-dispatch/src/families/gemv.rs:505-515`,
`:569-577`, `:622-630`).

**Exact edit (Task T1):** in `steps.rs:163-165`, admit gfx1151:

```rust
!ctx.flags.force_unfused
    && (ctx.arch.is_gfx1100() || ctx.arch.is_gfx1151() || ctx.arch.is_gfx1201())
```

(`ArchCaps::is_gfx1151` exists at `crates/rdna-compute/src/arch_caps.rs:305-307`.)
No kernel-source change: the fused LUT TUs are chip-agnostic wave32 scalar code
(the same TU already serves gfx1201). Keep `force_unfused` first (kill switch
wins). Do NOT widen to gfx1101/1102/1150/1152/1103 in this plan (no kernels
selected for them, no hardware to validate; fail-closed stays).

Trap to leave alone: `crates/hipfire-dispatch/src/families/fused_qkv.rs:78-91`
rejects any `MQ4G256V2Lloyd` weight reaching `FusedQkvFamily::run` with "no fused
LUT kernel exists". That text is stale *only* in the abstract — inside the family
it is still true (the family has no LUT ABI), and decode never routes through the
family for these keys (`launch_fused` handles them first, `steps.rs:1107-1142`).
Composers MUST NOT "fix" this guard by admitting Lloyd to the family; at most
reword the comment to name the bypass (`steps.rs:1108-1111`).

### 1.3 gfx11-specific decode variants with no LUT twin (all fall back, none fail)

- QKVZA K2048 hoist/x32: `FUSED_QKVZA_MQ4G256V2_K2048_HOIST_X32_GFX1100_SRC`
  (`kernels.rs:4268-4273`) is uniform-only and gfx1100-only. No LUT twin needed:
  a Lloyd window simply fails the (uniform-dtype) hoist gate and lands on the
  generic LUT fused kernel once §1.2 admits the arch, else per-projection LUT
  GEMVs. No edit.
- lm_head (M=248320, K=2048) + multirow: the V2-plain launcher routes every
  lm_head/k2048 specialization through the generic dual-scale source
  (`gemv.rs:7622-7655`, comment at `:7632-7635`), while Lloyd pins rows=1 with no
  multirow LUT variant (`gemv.rs:7816-7844`, rationale + measured +0.35% at
  `:7819-7822`; residual twin `:7926-7929`). So a Lloyd lm_head runs the generic
  single-row LUT kernel on every arch including gfx1151 — correct, no special
  gfx1151 lm_head/dot2/r1_hybrid_buffer LUT twin in scope. No edit.
- MoE indexed paths: `MQ4G256V2Lloyd` is explicitly refused on both indexed
  gate/up and indexed down (`crates/hipfire-dispatch/src/pipeline/mod.rs:520-523`,
  `:589-592`, "no indexed LUT kernel; refusing uniform decode"). Unchanged by
  this plan (dense-only tiers); MoE-Lloyd prefill refusals in §2.6 stand.

**Acceptance (decode slice):** on XTX (gfx1100) and Halo (gfx1151),
`hipfire bench --matrix` decode rows unchanged-or-better within noise;
1-chunk WT2 KLD prefill-vs-per-token gate per §4 recipe; fused Lloyd kernels
observed in the dispatch trace on gfx1151 (previously per-projection only).

## 2. Prefill on gfx11: dispatch sites, today's behavior, F16-WMMA-LUT twins

### 2.1 Every batched-prefill site a qt=52 tensor reaches (dense path)

Shared admission: `llama::mqv2_wmma_batchable` admits `MQ4G256V2Lloyd` exactly
like qt44 (`crates/hipfire-runtime/src/llama.rs:1995-2008`), and
`qwen35::is_batchable_la` delegates to it (`prefill.rs:1646-1648`; gfx11 kill
switch `HIPFIRE_MQV2_GFX11_WMMA` documented at `prefill.rs:1636-1638`). So on
gfx1100/gfx1151 a Lloyd model **is** batched-prefill eligible — the failure in
§2.2 happens *inside* the chunk, after admission.

Dense `*_lloyd` arms (all call `lloyd_e4m3_or_fail`, `prefill.rs:2812-2820`, and
the all-Lloyd uniformity helper `all_mq4v2_lloyd`, `prefill.rs:2826-2829`):

| # | Site (function) | All-Lloyd arm | Mixed-refusal arm |
|---|---|---|---|
| 1 | `dispatch_batched_gemm_epilogue` (shared wo/w_down dispatcher, `prefill.rs:88-92`) — plain epilogue | `prefill.rs:184-192` → `gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd` | — (single weight, no mixing possible) |
| 2 | same, `Partial` epilogue | `prefill.rs:286-292` (zero + same launcher) | — |
| 3 | `batch_chunk_delta_net_input_projection` (LA qkvza) | `prefill.rs:4631-4655` → `gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd` | `prefill.rs:4656-4667` ("mixed … refusing (quantize all four or none)") |
| 4 | `batch_chunk_delta_net_ffn_gate_up` | `prefill.rs:5503-5519` → `gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd` | `prefill.rs:5520-5527` ("quantize both or neither") |
| 5 | `batch_chunk_full_attn_input_projection` (FA qkv) | `prefill.rs:5919-5939` → `gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd` | via `qkv_same_dtype` fallthrough (`prefill.rs:5940-5941`; mixed Lloyd/uniform qkv has no dedicated refusal — §2.5 makes one mandatory) |
| 6 | `batch_chunk_full_attn_ffn_gate_up` | `prefill.rs:6682-6698` → same gate_up launcher | `prefill.rs:6699-6706` |
| 7 | `batched_gemm_single_weight` (lm_head / single-weight GEMM) | `prefill.rs:9053-9067` (zero Y + residual FP8-LUT launcher) | — |

Callers of the epilogue dispatcher for wo/w_down: LA `wo` (`prefill.rs:5050-5053`),
LA `w_down` (`prefill.rs:5627-5633`), FA `wo` (`prefill.rs:6242-6245`), FA
`w_down` (`prefill.rs:6798-6804`). FWHT-input contract already includes qt52 for
`w_down` (`prefill.rs:5596-5611`, `:6769-6784`) and the admit matchers list
`MQ4G256V2Lloyd` alongside the other prefill dtypes (`prefill.rs:4403-4404`,
`:5025-5026`, `:5352-5353`, `:5600-5601`, `:5726-5727`, `:6217-6218`,
`:6534-6535`, `:6773-6774`).

The `s4_residual_fast` f16 shortcut can never fire for Lloyd: it requires
`w_dtype == DType::MQ4G256V2` exactly (`prefill.rs:4919-4932`, predicate at
`:4929`; exact-gfx1100 + ChainVerify + Residual + n≤16). No edit needed there,
and the underlying `*_f16` producers are exact-gfx1100-only anyway
(`crates/rdna-compute/src/mq_f16_residual_producers.rs:589-592`).

### 2.2 What happens today on gfx11: hard fail-closed HipError (not a fallback)

Each `*_fp8_lloyd` prepared launcher begins with an exact-arch gate:

- gate/up: `crates/rdna-compute/src/gemm.rs:29365-29371`
  ("mq4v2-lloyd gate/up FP8 prefill requires exact gfx1201 (no LUT variant elsewhere)");
- residual: `gemm.rs:30106-30111` (same message, residual);
- qkvza: `gemm.rs:30304-30309`; qkv: `gemm.rs:30594-30599`.

Plus shape fail-closeds: every fused m must be a multiple of 16
(`gemm.rs:29398-29403` gate/up, `:30343-30347` qkvza, `:30627-30631` qkv —
"never a straddled tile decoding on the wrong codebook"); K%256 and replay/capture
rejection ride along in the same guards. The error propagates out of the chunk
via `?` (`prefill.rs:1392-1414`, inside the `forward_prefill_batch_with_pbs_opts_inner`
closure at `:1323-1431`): **there is no in-chunk per-token fallback**. The
per-token `forward_scratch` loop only runs when the whole prefill is ineligible
(`prefill.rs:1256-1295`; eligibility-fallback contract documented at
`prefill.rs:1209-1212`). Net effect on gfx11 today: a base/pro (all-qt52) artifact
fails prefill with an error naming exact-gfx1201. Escape hatches that restore the
correct-but-slow per-token path: `HIPFIRE_MQV2_GFX11_WMMA=0` (gfx11-only,
`prefill.rs:1636-1638`) or `HIPFIRE_PREFILL_BATCHED=0`
(`prefill.rs:2500-2502`; the ~14× slower fallback class is named at
`prefill.rs:1563-1564` for the Lloyd-gfx12 analogue).

Host fail-closed rules that already exist and stay: Lloyd→uniform-key misroute
refusals (`crates/hipfire-dispatch/src/families/gemm.rs:254-276`, three arms:
v1 keys, uniform-V2 keys, mq4c keys); missing-E4M3-sidecar refusal naming the
family (`prefill.rs:2816-2820`); MoE refusals (§2.6).

### 2.3 The F16-WMMA-LUT twins to build (one kernel family each)

Uniform gfx11 sources (all LDS:0, wave32 WMMA
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, interleaved-C
`acc[j]=C[2*j+(tid>>4)][tid&15]`, `__launch_bounds__(32,2)` on gfx1100 /
`(32,8)` on gfx115x):

- gate/up base `kernels/src/gemm_gate_up_mq4g256v2_wmma.hip`: header contract
  `:17-23`; packed-half2 pkrtz dequant macros `:30-69` (`DEQUANT_PK_H2` at `:43`,
  `DEQUANT_A_FRAG_PK` at `:61`, gfx1100-only `#if defined(__gfx1100__)` at `:33`);
  launch bounds `:71-76`; header loads `:121-125`; half-select `:132-133`;
  invocations `:152`, `:166`. **BT twin**
  `kernels/src/gemm_gate_up_mq4g256v2_wmma_gfx11_bt.hip:15-36` (BT6 for batch
  96..383, BT12 ≥384 on exact gfx1100; "no gfx1151 admission branch" at `:18`;
  `GEN_GATE_UP_BT` at `:27`; dequant inside the macro at `:68-79`).
- qkvza base `kernels/src/gemm_qkvza_mq4g256v2_wmma.hip`: same structure
  (macros `:31-70`, args `:77-89`, loads `:124-130`, select `:137-138`,
  invocations `:157`, `:171`). **BT twin**
  `kernels/src/gemm_qkvza_mq4g256v2_wmma_gfx11_bt.hip:17-29` (dequant `:79-89`).
- qkv base `kernels/src/gemm_qkv_mq4g256v2_wmma.hip`: inline `DQ` macro instead
  of the pkrtz path (loads `:81-86`, select `:93-94`, `DQ` + invocations
  `:110-129`, WMMA `:119`, `:129`). **BT twin**
  `kernels/src/gemm_qkv_mq4g256v2_wmma_gfx11_bt.hip` (dequant `:75-85`).
- residual base `kernels/src/gemm_mq4g256v2_residual_wmma.hip`: loads `:69-74`,
  select `:80-81`, `DQ` `:87-92`, WMMA `:96`, `Y +=` store `:100-109`. **BT twin**
  `kernels/src/gemm_mq4g256v2_residual_wmma_gfx11_bt.hip` (dequant `:55-66`).

Production defaults the twins plug into (unchanged code paths, new callees):

- `gemm_qkvza_mq4g256v2_wmma` (`gemm.rs:27943-28115`): MMQ diversion at
  batch≥128 (`gemm.rs:27961-28010`, iu4 opt-in at `:27967-27988`, beta/alpha
  M<128 small-tail at `:27995-28008`); then BT select via `mqv2_prefill_batch_tile`
  (`gemm.rs:28013-28036`) → `*_gfx1100_bt` (`gemm.rs:28127`) /
  `*_gfx1151_bt` (`gemm.rs:28248`); else base (`gemm.rs:28037-28040`).
  Frozen tile policy: `gemm.rs:379-389` (gfx1100 Qkvza|Qkv BT4 @96.. / BT12 @192..;
  gfx1151 GateUp BT12 @96.., Qkvza|Qkv|Residual BT4 @96..).
- `gemm_mq4g256v2_residual_wmma` (`gemm.rs:31105-31259`): MMQ diversion
  (`gemm.rs:31114-31129`); exact-gfx1100 verify tier (ldsstage default-on where
  K%512==0, else ksplit table, `gemm.rs:31140-31167`); MW policy MW4 @416..463 /
  MW8 @≥464 (`gemm.rs:31168-31183`); shared `mqv2_mw_waves` (`:31184-31192`);
  BT select (`:31193-31212`) → gfx1100_bt (`gemm.rs:31271`) / gfx1151_bt
  (`gemm.rs:31689`); else base (`:31214-31217`).
- gate/up (`gemm.rs:29595+` → MW @≥384 at `gemm.rs:29629-29646`, shared MW waves,
  BT select → `*_gfx1100_bt` / `*_gfx1151_bt` (`gemm.rs:29760`, `:29967`),
  small-N ldsstage default-on exact gfx1100 (`gemm.rs:29674-29678`).
  qkv mirrors with `*_gfx1151_bt` at `gemm.rs:28734`.

**Out of scope, explicitly:** ldsstage / mw_lds / ksplit LUT twins on either arch
(residual `*_gfx1100_ldsstage`/`*_ksplit_lds`/`*_mw_lds`, gate_up
`*_gfx1100_mw_lds`/`*_ldsstage`, shared `gemm_mqv2_wmma_gfx11_{bt,mw_lds}.hip`
`kernels.rs:3222-3228`). Lloyd serves base + BT only. Rationale: those tiers
serve N≤16 (verify) and N≥384–464 MW bands where Lloyd can either ride the base
kernel (small-N is launch-bound, per-token-competitive) or the BT kernel
(large-N weight reuse); porting the occupancy-tuned MW/ldsstage schedules for a
second decode arithmetic is a follow-up with its own bench round, not a
correctness need. The plan fails closed (never falls back to uniform) wherever a
twin is absent.

### 2.4 Recommended LUT mechanism: LDS-staged f16 table + wave-uniform source select

- **Do:** mirror the decode LUT exactly — 16 centered f16 levels as 8 kernarg
  dwords, staged once to 32 B of LDS by lanes 0..7 with a single `__syncthreads`
  *outside* the K loop (`kernels/src/gemv_mq4g256v2.hip:78-94`), plus a
  `DQ_LUT`/`DEQUANT_*_LUT` macro that replaces the uniform affine
  `sc*q+zp` at each §2.3 dequant site with `sc*LUT[q]+zp'` (headers already carry
  pre-folded `zp'`; container stride stays 136 B). Per-source codebook choice
  uses the FP8 kernel's wave-uniform idiom: select on the slab/tile *start row*
  (from `blockIdx`, never the lane row) so the dwords stay in SGPR across the K
  loop (`kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:296-343` for
  SLABS=2, `:369-396` for SLABS=1; "never occupy VGPRs" rationale at `:299-302`).
- **Do not use v_perm byte lookup.** The `v_perm_b32` trick is E4M3-specific:
  16 one-byte table entries as 4 immediates, top-two-nibble select
  (`gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:70-109`, LUT-arg form at `:83-95`).
  f16 entries are two bytes; the trick does not transfer without a second shuffle
  stage that costs more than the LDS read it replaces.
- **Do not use per-lane VGPR tables.** 16 f16 = 8 VGPRs per source; fused kernels
  carry 2–4 sources (16–32 VGPRs) under `__launch_bounds__(32,2)` on gfx1100
  (all four base TUs + all BT TUs, §2.3) — occupancy would drop exactly where BT
  needs it most. LDS cost is 32 B against kernels that use LDS:0 today.
- **Do not add an LDS lookup inside the K loop** (a barrier per group). Stage
  once per 16-row tile; the K loop then reads `LUT_SMEM[q]` exactly like the
  decode `DOG_LUT` path (`gemv_mq4g256v2.hip:119-127`).

Uniform symbols stay textually unchanged (hard constraint): each twin is a new
`-DHIPFIRE_MQ4G256V2_LUT=1` build of the corresponding TU with a distinct
`_lloyd` kernel symbol and `_LUT` module name, mirroring
`FUSED_GATE_UP_MQ4G256V2_LUT_SRC` (`kernels.rs:5386-5389`) and the FP8
`*_LUT_SRC` pattern (`kernels.rs:3528-3535` and siblings). New `*_for_arch`-style
constants are additive; no existing `pub const` changes bytes.

### 2.5 Kernarg plumbing, dispatch arms, fail-closed rules

- New launchers mirror the `*_fp8_*_prepared_lloyd` signatures
  (`gemm.rs:29351` gate/up, `:30096` residual, `:30282` qkvza, `:30576` qkv) but
  take decode-family LUTs: `[u32; 8]` `lloyd_lut_f16` per tensor (the same sidecar
  decode uses, `steps.rs:1090-1094`), NOT the `[u32; 4]` E4M3 form. New helper
  `lloyd_f16_or_fail` mirroring `lloyd_e4m3_or_fail` (`prefill.rs:2812-2820`)
  with the same "re-quantize … refusing uniform decode" message shape.
  Suggested names (frozen contract): 
  `gemm_gate_up_mq4g256v2_wmma_gfx11_lloyd`,
  `gemm_qkvza_mq4g256v2_wmma_gfx11_lloyd`,
  `gemm_qkv_mq4g256v2_wmma_gfx11_lloyd`,
  `gemm_mq4g256v2_residual_wmma_gfx11_lloyd`, each with a `_bt` twin taking
  `batch_tile`; BT width policy reuses `mqv2_prefill_batch_tile`
  (`gemm.rs:379-389`) so Lloyd inherits the frozen tiles (no new tuning table).
- `prefill.rs` arms (#3–#7 in §2.1 table): branch each all-Lloyd arm on arch —
  `gfx1201` keeps the existing `*_fp8_lloyd` call; `gfx1100 | gfx1151` calls the
  new `*_gfx11_lloyd` (BT-selected inside the launcher, same as uniform);
  any other arch → `HipError` fail-closed (mirrors the launcher-side
  "requires exact …" shape). Mixed-refusal arms (`:4656-4667`, `:5520-5527`,
  `:6699-6706`) stay verbatim; **add** the missing FA-qkv mixed
  Lloyd/uniform refusal at `prefill.rs:5940-5941` (fail closed like the LA/FFN
  arms — today a mixed qkv would silently take `run_fused_qkv_key`; see also the
  family-level backstop at `families/gemm.rs:263-270`).
- `s4_residual_fast` needs no change (`:4929` already exact-uniform).
- Kill switches: `HIPFIRE_MQV2_GFX11_WMMA=0` keeps working (admission-level,
  `prefill.rs:1636-1638`); add one master default-ON switch for the new route,
  e.g. `HIPFIRE_LLOYD_F16_GFX11` (=0 restores the §2.2 fail-closed-then-escape
  behavior on gfx11, uniform path byte-untouched). No per-family flags in v1.

### 2.6 MoE: stays refused

`MQ4G256V2Lloyd` attention weights in MoE layers keep the hard refusals
(`prefill.rs:6953-6958` DeltaNetMoe, `:7732-7737` FullAttnMoe — "no MoE-batched
prefill kernel … per-token fallback serves this layer") and the indexed-decode
refusals (`pipeline/mod.rs:520-523`, `:589-592`). No MoE LUT kernel in this plan.

## 3. Cost model per tier on gfx11 (accepted tradeoff, stated plainly)

Measured uniform baselines on Qwen3.8-27B XT (do not mix rounds; same-artifact
deltas only):

- X128 MMQ default (`CHANGELOG.md:13`): XTX pp512 990→1141 (+15%), pp2048
  927→1059 (+14%); Halo pp512 377→438 (+16%), pp2048 354→407 (+15%).
- iu4-direct MMQ opt-in (`CHANGELOG.md:8`): XTX pp512 1203.6→1588.8 (+32%),
  pp2048 1185.4→1557.0 (+31%), pp8192 1093.9→1405.1 (+28%); Halo pp512
  449.9→581.3 (+29%), pp2048 442.0→568.8 (+29%), pp8192 412.5→519.2 (+26%);
  decode unchanged both cards. Caller-measured F16-WMMA ~1196 vs iu4 ~1559
  @pp512 XTX is consistent with these two entries (1141 X128 / 1588.8 iu4).

Projection for base/pro (all linear tensors qt=52 → prefill runs *entirely* on
the §2.3 F16-WMMA-LUT route; no MMQ diversion exists for Lloyd):

- XTX: pp512 ≈ 1150–1200 (F16-WMMA rate + LUT overhead) vs iu4-uniform ~1559
  → **≈ −25% prefill vs xt-uniform**; pp8192 ≈ 1050–1100 vs 1405 (same ratio;
  attention share grows with context, diluting the GEMM gap — cf. the FA2 entry
  `CHANGELOG.md:12` where pp8192 attention dominates).
- Halo: pp512 ≈ 430–450 vs 581 → **≈ −25%**; pp2048/pp8192 scale likewise off
  the `CHANGELOG.md:8` Halo row.
- Decode is unaffected (same single-row LUT GEMV class; iu4 entry: decode
  48.82→48.92 XTX, 14.22→14.18 Halo).
- Quality side of the trade (already measured on gfx12, `CHANGELOG.md:4`):
  24-chunk WT2 KLD base 0.0405→0.0346, pro 0.0335→0.0284. The §4 recipe
  re-measures both deltas on gfx11 rather than assuming transfer.

Why no MMQ for Lloyd (not a gap to close later): MMQ dots integer codes through
`wmma_i32` (X128: `CHANGELOG.md:13`; iu4: `gemm.rs:18488-18510`,
`GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC` at `kernels.rs:3351-3352`, admitted exact
gfx1100/gfx1151 at `gemm.rs:18507-18511`). An integer path cannot consume a
16-entry float codebook without dequantizing first, at which point it is the F16
route again. **No blocker for residual/down_proj**: every family has a full F16
WMMA stack on both arches (residual base + `_gfx1100_bt` + `_gfx1151_bt`,
`gemm.rs:31271`/`31689`; gate_up/qkvza/qkv likewise per §2.3). The GDN
beta/alpha M=48 tails need no special kernel: 48 % 16 == 0 satisfies the
slab-select host contract, and the uniform small-tail MW4 path
(`gemm.rs:27995-28008`, measured `CHANGELOG.md` X128-followup entry) has its
analogue in the fused LUT arm's per-slab select. xt (uniform) keeps MMQ+iu4 and
pays nothing.

## 4. Bounded task breakdown (composer-sized, disjoint ownership)

Conventions for every unit: no GPU commands in this planning phase were run and
none are needed to write code; `cargo build -p <crate>` is allowed (no GPU
program). Implementers run GPU work only on the **hipx box, one daemon per card,
never two on one card**; XTX = gfx1100 seat, Halo = gfx1151 seat. Verification
per unit = §§4.1–4.3; order = T0 → T1 → T2 → T3 → T4 → T5 → T6 (T3/T4/T5 are
code-disjoint and may parallelize after T0's mechanism freezes the
`DQ_LUT` + slab-select contract).

- **T0 — Gate-first residual experiment (kill the idea cheaply).**
  Owns: one new TU build (`gemm_mq4g256v2_residual_wmma.hip` + `-D` twin only),
  one launcher (`gemm_mq4g256v2_residual_wmma_gfx11_lloyd`, base tile only, no
  BT), one throwaway differential harness (not a committed test).
  Design already frozen by §2.4; no further design allowed.
  Abandon criterion: if the differential vs `gemv_mq4g256v2_lloyd`
  (`gemv.rs:7824`) at N=511/512 exceeds rel-L2 1e-5 vs the uniform-F16-vs-uniform
  baseline gap, OR the twin is >10% slower than the uniform base residual at
  pp512 on XTX — stop the whole F16-LUT direction and report; do not build
  T3–T5. (Gate-first: magnitude unknown until measured.)
- **T1 — Decode gfx1151 admission.** Owns: `crates/hipfire-dispatch/src/pipeline/steps.rs:163-165`
  (add `is_gfx1151`), guard unit tests (`steps.rs:1409-1468` pattern — add
  gfx1151 admit + gfx1150/1152 refuse cases), coverage-table rows if the harness
  requires (`coverage_tests.rs:252-280`, `1024-1049`). No kernel/host changes.
  Acceptance: fused Lloyd fires on Halo in trace; KLD gate §4.2 on both cards.
- **T2 — Residual F16-LUT (productionizes T0).** Owns: residual twin TU +
  `GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX11_LUT_SRC`-style const (`kernels.rs`),
  base + `_bt` launchers (`gemm.rs`, BT reuses `gemm.rs:379-389` tiles),
  `dispatch_batched_gemm_epilogue` arch branch (`prefill.rs:184-192`,
  `:286-292`), `lloyd_f16_or_fail` helper next to `prefill.rs:2812`.
  Acceptance: §4.1 differential + §4.2 KLD + §4.3 matrix slice (residual-only
  model config if available, else full).
- **T3 — Gate/up F16-LUT.** Owns: gate/up twin TU + consts, base + `_bt`
  launchers, `prefill.rs:5503-5519` + `:6682-6698` arch branches. Mixed refusals
  untouched. Dequant sites: `gemm_gate_up_mq4g256v2_wmma.hip:43-69,121-133,152,166`
  (+ BT `:68-79`).
- **T4 — QKVZA F16-LUT.** Owns: qkvza twin TU + consts, base + `_bt` launchers,
  `prefill.rs:4631-4655` arch branch. Sites:
  `gemm_qkvza_mq4g256v2_wmma.hip:44-70,126-138,157,171` (+ BT `:79-89`).
- **T5 — QKV F16-LUT + FA mixed-refusal.** Owns: qkv twin TU + consts, base +
  `_bt` launchers, `prefill.rs:5919-5939` arch branch, **new** mixed
  Lloyd/uniform FA-qkv refusal at `prefill.rs:5940-5941` (copy the LA wording at
  `:4663-4667`). Sites: `gemm_qkv_mq4g256v2_wmma.hip:81-94,110-129` (+ BT `:75-85`).
- **T6 — lm_head/single-weight + BT routing + fail-closed audit.**
  Owns: `prefill.rs:9053-9067` arch branch (lm_head), `families/gemm.rs:254-276`
  (extend Lloyd→F16-LUT-key misroute tests if new keys are intro­duced; behavior
  unchanged), `fused_qkv.rs:78-91` comment refresh (no behavior change),
  end-to-end `mqv2_prefill_batch_tile` Lloyd coverage (no table change —
  verify Lloyd reaches the same tiles via the new launchers), final §4.3 full
  matrix + §4.2 KLD on both cards for base + pro artifacts.

Out of scope for all units: mw_lds/ldsstage/ksplit twins (§2.3), MoE LUT kernels
(§2.6), `FusedQkvFamily` admission (§1.2 trap), any uniform-symbol change.

### 4.1 Kernel differential (per unit, throwaway script, not a committed test)

For the unit's family at N=511 (tail/mask path) and N=512 (full tiles):
F16-LUT twin vs `gemv_mq4g256v2_lloyd` (`gemv.rs:7824`) on synthetic + real
extracted rows; gate: max rel-L2 within the uniform-F16-vs-reference envelope
(the gfx12 FP8/F16 pair was byte-identical greedy, `CHANGELOG.md:12` — same bar
here: greedy 1200-token prompt byte-identical twin-vs-per-token). Keep the script
throwaway; commit nothing that pins wording/percentages.

### 4.2 Coherence: 1-chunk WT2 KLD prefill vs per-token

Per unit (and full-model at T6): 1-chunk WT2 KLD of the batched prefill route vs
the per-token `forward_scratch` route (`prefill.rs:1256-1295`), same prompt md5
and binary md5 across runs, ship gate KLD ≤ per-token + 0.0005 (precedent:
`CHANGELOG.md:12`–`:13` entries). Re-measure the §3 quality deltas (base/pro vs
xt-uniform) on gfx11; do not assume gfx12 transfer.

### 4.3 Perf: `hipfire bench --matrix --pp 512,8192 --ctx 128,2048`, interleaved A/B

Per unit and at T6: uniform-xt vs Lloyd-base/pro on the same card, interleaved
A/B, ≥3 fresh-process runs per leg (within-session A/B drifts 10–15% with
DPM/thermal state), recorded prompt md5 + binary md5, eyeball decoded output
(tight stddev / high τ = attractor warning), and repeat the headline cell under
graph capture (tok/s alone never proves graph correctness). Claim-scoped harness
per `docs/VALIDATION.md`: kernel bucket → `scripts/serve_harness.py` battery +
`scripts/redline_daemon_harness.py` capture + HIP/PM4 parity
(`docs/VALIDATION.md:129-135`; manual path `:143-144`); merge bar stays
build + unit-tests + gates + one approving review (`:42-55`); arch-port claims
use `methodology/arch-port-validation.md` (`:268`); coherence-gate scripts are
retired — historical reproduction only, never acceptance (`:280`, `:297`).

## 5. Rejected alternatives

- **Dequantize-to-uniform-grid at load** (snap Lloyd codes to 0..15 affine on
  ingest, run uniform kernels everywhere). Rejected: discards the measured
  quality win (`CHANGELOG.md:4`: base 0.0405→0.0346, pro 0.0335→0.0284 WT2 KLD);
  silent (no fail-closed signal distinguishes it from a real Lloyd decode);
  saves nothing (same kernels, worse logits).
- **MMQ with re-quantized codes** (re-fit uniform scales around Lloyd centroids
  to keep the integer path). Rejected: integer-dot consumes raw codes
  (`gemm.rs:18488-18510`); re-quantization *is* uniform quantization plus an
  extra rounding — strictly worse than shipping uniform while paying Lloyd's
  sidecar complexity. The F16 route exists precisely because the codes are not
  affine.
- **F16 shadow weights** (decode Lloyd once at load, serve F16). Rejected: 2×
  VRAM vs 4.25 bpw effective; 27B-class models already pressure 24 GB gfx11
  (the OOM shape that motivates `HIPFIRE_PREFILL_BATCHED=0`,
  `prefill.rs:2500-2502`); defeats the tier's bytes-per-weight budget
  (`CHANGELOG.md:4`: same 136 B container as qt44).
- **FP8-WMMA-LUT port to gfx11** (instead of F16-LUT). Rejected: gfx11 WMMA is
  F16 (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` across all §2.3 TUs); there
  is no gfx11 FP8-WMMA twin of `gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip`, and
  building one is a larger ISA project than the LUT twin. The E4M3 path stays
  gfx12-only behind its exact-arch gates (§2.2).
- **Porting MW/LDS-stage/ksplit LUT twins now** (§2.3 out-of-scope list).
  Rejected for v1: three more occupancy-tuned schedules × two arches with no
  correctness need (base covers N≤16/odd-N, BT covers production N≥96 per
  `gemm.rs:379-389`); revisit only with a bench round showing a >5% prefill gap
  attributable to those bands on a Lloyd artifact.
