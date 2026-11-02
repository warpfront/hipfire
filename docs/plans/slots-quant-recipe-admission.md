# Multi-slot quant-recipe + KV admission — map & implementation plan

Branch: `feat/slots-quant-recipes` (off `a89ed0a8e`, v0.4.0 master).
Host GPU: gfx1101 (RX 7700 XT). Motivating failure: a `--format mq4v2
--tier pro` trunk fails on slots with `dense DeltaNet layer is neither
uniformly Q8_0 nor a uniformly batchable MQ-family dtype`; a `--kmap-dense`
trunk fails identically (qt15 Promote6 projections).

**Status 2026-10-04: phases A/B/C-core/V-core complete and GPU-verified on
this host.** See §7 for the evidence trail and §9 for what remains.

## 1 · How a recipe becomes gpu_dtypes (quantizer side)

`hipfire-quantize` per-tensor recipes that reach slots:

| Recipe | Trunk result |
|---|---|
| `--format mq4v2` (default) | body qt44 `MQ4G256V2`; embed/conv1d `Q8F16`→`Q8_0`; norms `F16` |
| `--tier xt/base/pro` | base lifts embed+lm_head to `Q8_0`; pro additionally lifts `ssm_out` = `linear_attn.out_proj` (wo) to `Q8_0` → **mixed layer** |
| `--fixed-tier <class>:<dtype>` | any class (lm_head, embed, router, attn, attn_full, ssm_out) → {q8, mq{2,3,4,5,6}v2} → **arbitrary per-projection mixes** |
| `--kmap-dense` | Promote6 tensors → `MQ6G256V2` (qt47) on audited loaders since B1 (was qt15 `MQ6G256` — quantizer bug, fixed by cherry-pick `f791f7f53`) |
| `--format mq{2,3,5,6}v2` | uniform bodies of qt48/46/45/47 |
| `mq4v1/magnum`, `mq4c`, hfq/hf, `mfp4`, Lloyd (`qt52`), TQ2/BQ1 | assorted; slot scope decided below |

`Q8F16` loads as `gpu_dtype = DType::Q8_0` (load.rs:875).

## 2 · Per-site admission (what replaced the uniform gate)

`forward_slots.rs` now plans each projection SITE (see its module doc):
`plan_proj_group` collapses each fused group ({wqkv,wz,wβ,wα} DeltaNet,
{wq,wk,wv} FullAttn, {w_gate,w_up} FFN) to `Uniform(dt)` (one fused launch)
or `Mixed` (per-weight plain GEMMs; every member needs a
`slots_plain_gemm_key`; AWQ sidecars in mixed groups refuse with a named
error). Residual roles (wo, w_down) dispatch on the weight's OWN dtype via
`slots_check_residual_weight` + `slots_residual_gemm_key` (NEVER the
dispatch-level `residual_gemm_key_for` wildcard — see §8 risk 1). Uniform
Q8_0 groups degrade to the plain-GEMM fork on non-WMMA archs or
`HIPFIRE_Q8_PREFILL_WMMA=0` (`plan_q8_fused_or_plain`). MoE layer attention
sides use the same machinery; the MoE FFN stays on the reference's own
shared gate (`require_batchable_moe_ffn` → `gated_moe_prefill_admissible`),
which already admits mixed-merged routed experts — admission parity with
the sequential engine holds by construction.

Admitted containers (`slots_proj_admissible`): Q8_0, MQ4G256, HFQ4G256,
MQ4G256V2, MQ4CG256, MQ6G256, HFQ6G256, MQ6G256V2, MQ5G256V2, MQ3G256,
MQ3G256V2, MQ3G256Lloyd, MQ2G256V2 (arch-sensitive half via `is_batchable_la`).
lm_head: `lm_head_slots_admissible` (14-dtype allow-list, dtype-generic
GEMV; xbatch fast path stays MQ4G256-only). Still refused with named
errors: Lloyd-V2 (qt52), PARO, E8/MFP4, G128-rotated, F32/F16/Q4K/TQ2/BQ1
projections, non-Q8 embed, fp32 DeltaNet state.

## 3 · Kernel inventory (no new kernels needed for A/B)

- Fused multi-projection kernels per container:
  `FusedQkvza*`/`FusedQkv*`/`FusedGateUp*` for qt13(hfq4)/44/45/46/47/48/49
  and `Q8_0`, `Hfq3G256`, `Hfq6G256`, Lloyd (types.rs:182-230);
  `fused_gate_up_key_for` gained explicit Q8_0/HFQ6/MQ3/Lloyd arms (was a
  silent HFQ4 catchall — closes a latent corruption path).
- Per-projection plain GEMM keys for the V2 family + `GemmQ8_0BatchedChunked`
  + `GemmHfq4G256` (+ MQ4C). Uniform-only (no plain key): HFQ6/MQ6(qt15),
  MQ3, MQ3-Lloyd, HFQ3 — they can only appear in Uniform groups.
- Residual GEMMs: `slots_residual_gemm_key` spells out every admitted
  container's residual key including `GemmHfq6G256Residual` /
  `GemmHfq3G256Residual` / `GemmMq3G256LloydResidual`.

## 4 · Quantizer Promote6 fix (B1) — done via cherry-pick

Commit `f791f7f53` ("quantize: use loader-admitted MQ6V2 for V2 K-map
Promote6", from the beta stack) is applied: `mq6v2_promote_supported(arch_id)`
gates archs {0,1,5,6,7,8}; `--format mq4v2 --kmap-dense` now emits qt47
`MQ6G256V2` for Promote6 non-expert AND expert weights; legacy MQ4/mq4c and
unaudited archs keep qt15. qt15 vs qt47 headers differ (1×f32 affine vs
2×fp16 grids) — requant required, not a re-tag. The old 6.35 GB kmap-dense
test artifact was gone; a fresh one was quantized for verification (§7 B1).

## 5 · KV ladder — measured matrix (C1)

Line-number corrigenda vs the pre-2026-10-04 draft, re-anchored to the
post-beta-rebase tree (6e0a92590): `kv_write_slots` =
forward_slots.rs:~1850 (fp8 arm at ~L2041), `tier_attend_slots` = ~L2077
(fp8 arm at ~L2313); `QWEN35_SLOTS_POLICY` = hipfire-runtime/src/kv_mode.rs:495.
The qwen35 fp8 arch gates live at kv_mode.rs:30 (`qwen35_native_eligible`)
and :226 (`qwen_auto_default_pair`); `resolve_qwen4` (kv_mode.rs) is the
qwen4 site, not qwen35.

Instrument: `crates/saddle-lab/examples/test_forward_slots_golden.rs`,
reworked 2026-10-04 into a per-mode matrix (arg 2 = comma list; reference
arm uses the production mode-generic `KvCacheExt::from_mode_with_backend`;
candidate arm builds `SlotPool::new_with_strides` + seeded tier tables
exactly as `Rig::build`; `SLOTS_GOLDEN_PAGED=1` selects the paged pool).
n_slots 1..=4, prompt lens [5,9,3,7], logits-level parity, negative control
per mode. Run through `scripts/run-bounded.sh` (HIPFIRE_MEM_CAP=6G on the
30 GiB host). Fixture: `~/.hipfire/models/qwen3.5-4b.mq4v2.hfq` (dense,
32 layers, 8 FA/24 LA).

| Mode | Legacy slab | Paged pool |
|---|---|---|
| q8 | 10/10 @ 0.000x | skipped — WMMA fast paths are slab-only; paged q8 attends via the tiled kernel, numerically different (~18x tol on worst element) from the WMMA sequential reference; kernel-level coverage exists (`test_kv_slot_desc_ports`, `test_batched_attn_slots`) |
| fwht2 | 10/10 @ 0.000x | 10/10 @ 0.000x |
| fwht3 | 10/10 @ 0.000x | 10/10 @ 0.000x |
| fwht4 | 10/10 @ 0.000x | 10/10 @ 0.000x |
| asym2/3/4 | alias to fwht2/3/4 since 0.4.0 (resolve table exercised) | — |
| bf16 | end-to-end logits diverge ~6x tolerance on one vocab element (stable before AND after switching the slots arm to the sequential-parity kernel; kernel-level slots-vs-sequential parity is BIT-EXACT per `rdna-compute/examples/test_bf16_slots_parity.rs`). The sequential bf16 attend selection is itself a heuristic (`bf16_attend_key`, flash_mode/capture-dependent), so arm-matching is insufficient — root-cause follow-up §9. Excluded from the harness default matrix; still runnable explicitly | same |
| f16 | slots-ONLY tier by design (saddle-core kv.rs: "F16 flat rows exist only on the slots engine") — no sequential constructor exists, so no golden reference is possible; excluded from the matrix with a printed note | — |
| fp8 | refused end-to-end below gfx1201 (5 stacked gates); gfx1201 slots support implemented (see C2), untested-hw | — |

Every negative control (row_slot redirect at a decode step, positions 6 vs
9) failed CORRECTLY at ~6000–6900x tolerance — the comparison is sensitive
to slot misrouting under every tier. (The pre-rework control had been
vacuous: with DECODE_STEPS=0 its "expected failure" was an index-out-of-
bounds panic; it now runs its own explicit decode step and asserts the
failure is a tolerance mismatch.)

**Mixed-recipe acceptance (V-core):** the fresh
`mimo-v2.6-distill-qwen-9b.mq4v2.tierpro` artifact (census: 224×MQ4G256V2,
50×Q8F16 = embed+lm_head+24 conv1d+24 wo, 153×F16; the wo-is-Q8-in-a-qt44-
layer mix that was this branch's motivating failure) passes 20/20 slot-steps
at 0.000x on q8 and fwht3 with correct negative controls.

**Open item (sequential side, not slots — deterministic repro):** the
sequential `forward_scratch` decode under a fwht3 filtered cache faults
with an illegal memory access — but only with allocator history. Repro
(gfx1101, qwen3.5-4b.mq4v2.hfq): `scripts/run-bounded.sh
./target/release/examples/test_forward_slots_golden
~/.hipfire/models/qwen3.5-4b.mq4v2.hfq q8,fwht2,fwht3` — the q8 and fwht2
modes pass fully (golden + control), then fwht3's golden sweep passes
10/10 and the IMA lands at the NEGATIVE CONTROL's reference decode sync.
`... fwht3` alone passes; `... fwht2,fwht3` passes; `... fwht4,fwht2`
passes. So: fwht3 sequential decode OOB read whose landing address is only
unmapped after q8+fwht2 allocation history. Pre-existing (nothing on this
branch touches sequential fwht3 decode); slots-side fwht3 is green
(isolated legacy, paged, and inside every passing prefix). Needs its own
bisect — candidate: fwht3 decode attend read extent vs the filtered
cache's physical allocation.

**KV splits** (`--kv-k`/`--kv-v`, lloyd V-tiers): refused at the daemon gate
for slots (single-KvMode resolution; V is statically Q8 on the slots ladder
except bf16/f16/fp8). Out of scope v1, unchanged.

## 6 · fp8 slots (C2) — implemented, gfx1201-gated, untested-hw

The fp8 TUs are the q8 slot-desc TUs with `#define HIPFIRE_KV_FP8_E4M3 1`
(`KV_CACHE_WRITE_FP8_E4M3_BATCHED_SRC`, `ATTENTION_FLASH_FP8_E4M3_TILE_
BATCHED_SRC`), so slots fp8 is wrapper + gating work, not new HIP kernels:
desc-passing slots wrappers around the sequential bases, paged-symbol
registration in `kv_slot_paged_symbol`, fp8 geometry in `SlotKvTierPlan`,
gfx1201-conditional admission in the rig (fail-closed named error naming
the arch elsewhere). No gfx1201 host was available — every new GPU path is
marked untested-hw in code; on gfx1101 behavior is byte-identical to before
(refusal). Same change adds the bf16 parity wrapper (§5).

## 7 · Work items — status

- [x] A1/A2: per-site dispatch (`plan_proj_group`/`GroupPlan`/
      `slots_norm_site`/`slots_plain_proj`/`slots_residual_proj`) across
      dense DeltaNet/FullAttn + both MoE attention bodies; uniform gates
      deleted; fused/plain/residual key tables widened per container.
- [x] A3: lm_head widened to the tier/fixed-tier set; embed preflight keeps
      the named non-Q8_0 error. Load-time preflight promotion remains
      follow-up (plans recompute per call; ~20 branch checks/site).
- [x] A4: CPU unit tests: `proj_group_plans_uniform_and_mixed`,
      `residual_and_plain_key_tables_cover_admitted_set`,
      `rotated_residual_keys_never_fall_back_to_hfq4` (new — locks the
      mq4_residual_proj container-selection invariant),
      `q8_uniform_degrades_to_plain_on_non_wmma` (new). 247 lib tests pass.
- [x] B1: quantizer Promote6 → qt47 (cherry-pick `f791f7f53`, conflicts
      resolved; GGUF path included). Verified end-to-end 2026-10-04: a fresh
      `--format mq4v2 --kmap-dense` MiMo-9B requant from the RAW source
      emits a census of 226×qt44 + 22×qt47(MQ6G256V2, the Promote6 tensors)
      + 26×qt3 + 153×qt1 (the pre-fix binary emitted qt15 for those 22),
      and that kmap trunk passes the slots golden harness 10/10 at 0.000x
      with a correct negative control (7403x) — the second motivating
      failure of this branch, closed.
- [x] B2 (superseded): not needed — B1 emits qt47; qt15 stays admitted for
      legacy artifacts via the v1 fused/residual keys (uniform-only).
- [x] C1: KV matrix measured on gfx1101 (§5 table) — legacy + paged.
- [x] C2: fp8 slots wrappers + gates (§6); bf16 slots arm switched to the
      sequential-parity kernel.
- [x] D1 (already done by A2): MoE attention per-site; MoE-FFN rides the
      reference's shared gate. A3B golden coverage of a MoE checkpoint via
      the serve harness remains open (the golden harness needs a MoE
      fixture whose step shapes it supports).
- [x] D2: serve-side thinking/sampling fix cherry-picks — **satisfied by the
      beta rebase (2026-10-09)**: all four landed on `beta` before
      `740818f37` in their ingested forms (`#801`→superseded by `#808`/`#811`
      commits `02e657c79`/`8c6daf00a`/`419bc8d07`/`9c9d25094`/`0d8be43fa`,
      `#805`→`84a206030`, `#807`→`0961a3464`; the fork branches named here
      were deleted after ingest). Validated on the rebased tree, gfx1101
      multi-slot: thinking-low battery routes think/answer correctly on
      completing turns; length-cap think-span runaways under the registry
      recipe (presence_penalty 1.5) reproduce identically with the
      pre-rebase serve build — sampling/model-side, not the thinking-on
      400s this item tracked.
- [x] D3: this doc + QUANTIZE.md slots-admissibility section + CHANGELOG.
- [x] V-core: tier-pro requant (6074.9 MB, census above) + golden 20/20.
- [x] V-serve (2026-10-04, gfx1101): `serve_harness.py --mode battery
      --kv q8 --thinking off` with `HIPFIRE_SERVE_MULTI_SLOT=1 _SLOTS=2` on
      the tierpro artifact through the real daemon/serve route: 5/5 turns
      (code/reason/factual/prose/instruct), runaway=0 empty=0 attractor=0
      retrieval_miss=0, coherent decoded text eyeballed on every genre,
      avg decode 35.0 tok/s, kv_backend=legacy (fixed q8 slot arena).
      Re-validated 2026-10-09 on the beta rebase (`94a9987c1`): fresh
      local requant `qwen3.5-4b-tierpro.mq4v2` (202×MQ4G256V2 + 49×Q8F16 +
      22×MQ6G256V2/qt47 + 153×F16) — golden 40/40 @ 0.000× across
      q8/fwht2/fwht3/fwht4 with correct negative controls, serve battery
      5/5 clean at avg 62.1 tok/s, daemon md5 `6d2168da…`
      (note: `hipfire-pr806:gate` bakes `HIPFIRE_DAEMON_BIN=/hipfire/...`;
      pin it to the tree's `target/release/daemon` or the harness serves
      the image's stale daemon).

## 8 · Risks / invariants (updated)

1. **Residual-key selection** — `mq4_residual_proj` must resolve through
   `slots_residual_gemm_key`, never `residual_gemm_key_for` (whose wildcard
   would send 200 B/group MQ6G256 and 104/112 B/group MQ3 headers through
   the 136 B HFQ4 v1 kernel — full-speed noise). Found in review, fixed,
   locked by `rotated_residual_keys_never_fall_back_to_hfq4`.
2. **Q8 WMMA fork** — uniform-Q8 fused groups must degrade to plain GEMMs
   off WMMA archs / with the env kill-switch (`plan_q8_fused_or_plain`);
   the WMMA kernels are arch-limited and the golden parity under
   `HIPFIRE_Q8_PREFILL_WMMA=0` depends on matching the reference's fork.
3. Mixed-group numerics: a mixed group with both variants runs
   `rmsnorm_batched` + `rotate_x_mq_batched_for` instead of the fused
   producer — same math in f32, but this combination has no sequential
   counterpart (the reference's own mixed fallback feeds every member the
   rotated buffer), so it is deliberately MORE correct, and golden-vs-
   reference cannot cover it. Cover with the serve route when a recipe
   mixes inside a fused group (today's tier recipes mix only residual
   roles, which golden DOES cover).
4. AWQ sidecars refuse mixed groups (per-weight rotated input required).
5. Graph capture: all new arms are fixed kernel launches; no host-side
   branches on slot state inside a captured region.

## 9 · What remains (follow-ups, in priority order)

1. ~~V-serve (§7)~~ — **re-validated 2026-10-09** on the beta rebase with a
   fresh local tierpro requant (§7 V-serve): golden 40/40, serve battery
   5/5 clean.
2. bf16 end-to-end parity root-cause — **localized 2026-10-09 by layer
   bisect** (`bisect_forward_slots <model> bf16`, gfx1101,
   qwen3.5-4b.mq4, 5-token prefill): divergence is EXACTLY zero through
   the pre-KV DeltaNet layers (L2, L3: 0.0000), first appears at L4 —
   the first KV-carrying (FullAttention) layer — at ~8.5e-4 abs, grows
   ~1.5x per FA layer through the GDN recurrence, and materializes at
   L20 (FullAttention, rel 0.021 vs 0.0001-0.0003 for L4-L19) when a
   compounded difference flips an attention selection. So the seed is a
   small prefill-shape difference in the bf16 KV write/attend path at
   the FIRST KV layer, not a layer-20 bug. Sharpened next step:
   `test_bf16_slots_parity.rs` covers descriptor-plumbing parity of the
   batched kernel at decode-shaped calls; extend it (or a sibling) to
   the M>1 prefill attend shape vs the sequential single-sequence
   selection (`AttnBf16Kv` scalar — pos<2048, flash off, no capture) to
   catch the seed directly.
3. gfx1201 host validation of fp8 slots (untested-hw markers in code).
4. Sequential fwht3 decode IMA bisect (§5 open item).
5. ~~D2 serve-fix cherry-picks~~ — **done 2026-10-09**: satisfied by the
   beta rebase (all four landed upstream pre-`740818f37`); thinking-low
   multi-slot battery validates the serve thinking path on gfx1101 (§7 D2).
6. Load-time preflight promotion of the per-site plans (A3 follow-up) —
   quantified 2026-10-09: `plan_proj_group_dtypes` is a pure match chain
   over ≤5 (dtype, has_awq) members, ~5µs/step total across ~100
   group sites (32 layers x ~3 groups) against a ~10ms GPU step
   (<0.05%). Recommend NOT memoizing unless a host-bound profile ever
   shows it; a global cache is unmeasurable complexity for this win.
