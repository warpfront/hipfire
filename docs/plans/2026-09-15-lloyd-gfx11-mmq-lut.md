# MQ4-Lloyd (qt=52) gfx11 batched-prefill via MMQ-LUT twins (q8-x path)

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, branch `mq4-lloyd`, HEAD `628e62213`.
**This plan file is the only write. No GPU program was run; no source was edited.**
All paths below are worktree-absolute; `:line` anchors are HEAD-pinned. Every code
claim cites `file:line`; anything not directly observed is marked `[INFERENCE]`.

## 0. Decision, denominator, and what died

**Decision:** Lloyd tiers on gfx11 (gfx1100 + gfx1151) run batched prefill through
**LUT twins of the existing q8-x int8-dot MMQ kernels** (`gemm_mq4g256v2_residual_mmq*`,
`crates/rdna-compute/src/kernels.rs:3345-3346`), not through any F16 path.
Nibble→byte through a 16-entry byte LUT `C16`, scale `sc' = sc/16`, zero-point
exactly as the uniform MMQ path handles it today.

**Dead direction (do not revisit):** F16-WMMA-LUT prefill. STATUS block
(`docs/plans/2026-09-15-lloyd-gfx11-port.md:10-19`): XTX residual K=17408 M=5120,
F16-LUT twin 2224 us vs uniform F16 base 1378 us at N=384 (+61%), abandon
criterion was +10%. T0 code reverted.

**iu4 plainly cannot take a LUT:** `HIPFIRE_GFX11_MQ4V2_IU4` (W4A4) feeds weight
nibbles *directly* to `wmma_i32_16x16x16_iu4_w32` with no nibble→byte expansion —
"no nibble->int8 expansion in the loader" (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:5-6`),
"copies the 32 raw nibble words per 256-K row straight into LDS (no expand VALU)"
(`:12-13`, loader `:204-205`), consumer at `:280-281`. A 4-bit WMMA operand has
no codebook input; there is nowhere to put 16 codebook levels. Lloyd tiers on
gfx11 therefore run the **q8-x MMQ path**, never iu4. The iu4 flag stays
uniform-only (`crates/rdna-compute/src/feature_flags.rs:42-46`,
opt-in gate `:731-736`, wired at `:464-465`).

**Amdahl denominator (measured ground truth, task contract):** residual K=17408
M=5120 on the XTX today — uniform q8-x MMQ **1741 us at N=512**; uniform F16 base
tile 1378 us at N=384; naive F16-LUT 2224 us at N=384. The MMQ-LUT twin attacks
only the *weight-decode* sub-bucket of the 1741 us (loader VALU; §4 prices it).
Honest ceiling: the twin can never beat uniform q8-x MMQ — it does strictly more
loader work per identical WMMA sequence. **Target: within 3% of uniform q8-x MMQ
per family** (contract acceptance). Transfer-loss warning: compare only against
hipfire's tuned uniform MMQ (X128 default), never a naive baseline.

**Gate-first (unmeasured levers die cheap):** per-family uniform-MMQ baselines at
the exact Lloyd shapes (gate_up, qkvza, qkv M×K×N) are unmeasured at this HEAD —
U2's first step is to measure them on the XTX before writing any kernel line.
If any family's uniform MMQ does not reproduce ≈1741-us-class throughput on its
shape, that family's twin is deprioritized, not built blind.

## 1. Inventory: what a qt=52 tensor uses in batched prefill on gfx11 today

qt=52 is `DType::MQ4G256V2Lloyd` (`crates/hipfire-arch-qwen35/src/weights.rs:248-249`;
wire contract "byte-identical to MQ4G256V2, 136 B/group" in
`crates/hipfire-runtime/src/lloyd_lut.rs:7-9`). Admission to batched prefill is
shared with qt44 (`mqv2_wmma_batchable` admits Lloyd exactly like qt44,
`crates/hipfire-runtime/src/llama.rs:1995-2008`; `is_batchable_la` delegates,
`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1474`).

### 1.1 The uniform q8-x MMQ route the twin rides (all gfx1100|gfx1151, batch≥128, batch%128==0)

Core consumer (single-source, one `A` pointer — §2.1 explains why this kills the
per-tile LUT-select problem):

- `ensure_q8_1_mmq_x_v2` — X128-vs-per-32 prelude switch, same flag read as the
  consumer (`crates/rdna-compute/src/gemm.rs:18371-18382`; X128 default documented
  `:18365-18370`).
- `gemm_mq4g256v2_mmq_prequant` — the only consumer: exact-arch gate
  (`gemm.rs:18394-18399`), K%256 gate (`:18400-18405`), full-vs-base select
  `m%128==0 && batch%128==0` (`:18407`), x128 entry select (`:18411-18419`),
  fixed module `gemm_mq4g256v2_residual_mmq` over
  `kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_SRC` (`:18420-18425`), 7-arg kernargs
  (`:18433-18441`, blob `:18459-18467`), grid `[ceil(M/128), ceil(N/128)]`,
  block `[32,8,1]`, shared-mem `(128*36 + 128*76)*4 B` (`:18442-18455`).
- Public wrappers: `gemm_mq4g256v2_mmq_set_prequant` (`:18476-18486`, SET) and
  `gemm_mq4g256v2_mmq_add_prequant` (`:18488-18498`, ADD/`y+=`).
- Kernel source `kernels/src/gemm_mq4g256v2_residual_mmq.hip` via
  `kernels.rs:3345-3346`. Six entry symbols: base + x128 + full_add/set ×
  plain/x128 (`mmq.hip:543-691`).

Per-family dispatch into that core (uniform qt44 today; Lloyd twins call the same
shape with +C16 — §3.3):

| Family | Dispatcher | MMQ diversion (q8-x) | iu4 opt-in above it |
|---|---|---|---|
| qkvza (LA) | `gemm_qkvza_mq4g256v2_wmma` (`gemm.rs:27957-27974`) | `ensure_q8_1_mmq_x_v2` + per-tensor `set_prequant` at `gemm.rs:28007-28008`; GDN beta/alpha M=48 tails divert to MW4 f16 small-tail `gemm.rs:28010-28024` | `gemm.rs:27983-28006` |
| qkv (FA) | `gemm_qkv_mq4g256v2_wmma` (`gemm.rs:28503-28517`) | same pattern `gemm.rs:28516-28519` (q/k/v three `set` calls) | `gemm.rs:28525-28529` |
| gate_up | `gemm_gate_up_mq4g256v2_wmma` (`gemm.rs:29614-29625`) | batch predicate `gemm.rs:29626-29630`, gate+up two `set` calls (`gemm.rs:29620-29622` per dispatch grep) | `gemm.rs:29615-29618` |
| residual / down+wo | `gemm_mq4g256v2_residual_wmma` (`gemm.rs:31139-31147`) | predicate `gemm.rs:31148-31152`, `mmq_add_prequant` `gemm.rs:31160-31162` | `gemm.rs:31155-31159` |

Below-MMQ-threshold and capture/replay behavior is unchanged by this plan (F16
base/BT/ksplit/mw tiers listed in the prior plan §2.3 — Lloyd never routes there;
§6 names the resulting fail-closed band explicitly).

### 1.2 Exact nibble→dot, scale, zp, and x-sum sites in the uniform kernel

Correction first: **there is no dp4a intrinsic in the gfx11 path.** The header
comment states it: "RDNA3 only (use WMMA i8 builtin). gfx906 dp4a implementation
lives in gemm_mq4g256v2_residual_mmq.gfx906.hip" (`mmq.hip:156-157`). The dot is
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` in `mma_i8` (`mmq.hip:280-293`, calls
`:291-292`). The "verify signedness" task therefore resolves to: read the two
`true` flags at `:291-292` and determine which operand may be signed — §2.2 gives
both outcomes with frozen semantics.

- **Nibble unpack → dot operand:** `load_mq4v2_tile` (`mmq.hip:298-350`).
  Nibble extracts `q0..q7 = (qs0 >> 4i) & 0xF` (`:320-327`), repacked as bytes
  `q0 | q1<<8 | q2<<16 | q3<<24` into `x_qs` LDS (`:328-329`). Downstream of this
  point **nothing assumes codes < 16**: `x_qs` feeds `load_ldmatrix_16x8` →
  `mma_i8` (`:379`, `:385`, `:392`, `:448-451`) as opaque bytes. The `<16`
  assumption lives *only* in the expand at `:320-329`, which the twin replaces.
  No packed-nibble arithmetic, no 4-bit dot variant on this path (that is the
  iu4 file, excluded above).
- **sc/zp + x group sums:** headers `hs` per 128-K half, `sc`/`zp` via
  `__half2float(__ushort_as_half(...))` (`:341-343`), packed once to
  `half2 dm = (sc, zp)` (`:344`) and replicated to LDS (`:345-347`). The float
  correction is `sum += dmA.x*dsB.x*(float)C + dmA.y*dsB.y` — per-32 path
  (`:397-399`), X128 path once per 128-K half (`:453-459`; accumulation contract
  `:406-416`: "8 i8-WMMA dots of the half accumulate in ONE i32 C_frag and the
  float correction applies once"). `dsB` is the x-side `(d, s)` from the Q8_1 /
  X128 prelude (`:387`, `:453-457`); `dmA.y*dsB.y` is the zero-point term the
  twin leaves structurally identical.
- **Accumulator headroom:** i32 `C_frag` never spans more than one 128-K half
  (converted to float at `:398` / `:458-459`; the f32 `sum[]` carries full-K).
  Worst case `|C| = 128 · 240 · 127 = 3,901,440 « 2³¹` (per-32 path: 975,360).
  Byte-valued LUT codes cannot overflow the accumulator. No handling needed
  beyond stating the bound in the twin's review.

### 1.3 What happens today when qt=52 reaches gfx11 prefill: hard fail-closed

Each `*_fp8_lloyd` launcher begins with an exact-gfx1201 gate: gate/up
(`gemm.rs:29389-29391`), residual (`gemm.rs:30130-30131`), qkvza
(`gemm.rs:30331-30332`), qkv (`gemm.rs:30626-30627`) — all "(no LUT variant
elsewhere)". The 7 prefill call sites are §3.4. Misroute backstops already
exist: Lloyd→uniform-key refusals (`crates/hipfire-dispatch/src/families/gemm.rs:254-276`),
no-uniform-key panics in `forward_slots.rs:341-344, :357-360, :373-376, :389-392`,
MoE refusals (`prefill.rs:6956-6961`, `:7735-7740`). The plan adds routes; it
does not weaken any refusal.

## 2. The LUT twin: exact spec per kernel

One twin of `gemm_mq4g256v2_residual_mmq.hip`, not four: all families share the
single-source consumer (§1.1). Family differences live in the host launchers
(§3.3), which pass per-tensor C16.

### 2.1 No per-tile LUT select — per-launch C16 (deviation from the naive sketch, with evidence)

The gfx12 FP8 kernels are *fused multi-source* (one launch, gate+up / q/k/v /
qkvza pointers), hence per-slab/per-tile codebook select keyed on the
wave-uniform start row with an m%16 host contract
(`gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:296-302`, `:370-374`). **That
mechanism does not apply here.** The gfx11 MMQ route launches *per tensor*:
qkvza fires four separate `set_prequant` calls (`gemm.rs:28007-28008`,
`28016-28024`), qkv three (`gemm.rs:28517-28519`), gate_up two
(`gemm.rs:29621-29622`), residual one (`gemm.rs:31161`). Each launch carries one
weight tensor with one codebook. The twin therefore takes **one C16 = 4 kernarg
dwords per launch** (7 args → 11), selected host-side at the already-per-tensor
call sites. No SGPR/VGPR pressure question arises; no m%tile straddle contract
is needed for LUT correctness (single source per launch; the base kernel's
`min(row0+i, M-1)` clamp + guarded store at `mmq.hip:529-537` is unchanged).
`full` (m%128==0 && N%128==0, `gemm.rs:18407`) vs base selection is untouched.

### 2.2 Nibble→byte, scale, zp, signedness (two frozen outcomes)

- **Pattern:** reuse `mq4_spread4` + `mq4_lut8` verbatim (the `#ifdef
  HIPFIRE_FP8_LUT_ARG` LUT-arg form at `fp8.gfx12.hip:83-95`, spread at
  `:76-79`, pack-to-operand at `:112-119`), retargeted: table bytes are C16
  codes, output bytes feed `x_qs` LDS in place of the `:328-329` pack.
  Uniform text stays unchanged: new code lives under `#ifdef HIPFIRE_MMQ_LUT`
  with a distinct kernel symbol per entry (`*_lloyd`), mirroring the
  `-DHIPFIRE_FP8_LUT_ARG=1` + distinct-symbol pattern (`kernels.rs:3532-3535`)
  and the GEMV `-DHIPFIRE_MQ4G256V2_LUT=1` pattern (`kernels.rs:1578-1581`).
- **Scale:** `sc' = sc·(1/16)` in fp32 immediately after the `__half2float` load
  (`mmq.hip:342`), repacked at the existing `make_half2` site (`:344`). Exact:
  ×2⁻⁴ changes only the exponent (quant scales are never subnormal). Headers on
  disk/GPU untouched.
- **Zero-point:** structurally unchanged. Headers already carry
  `zp' = fp16(zp + 7.5·sc)` via `apply_lloyd_centering`
  (`lloyd_lut.rs:155-185`); the `dmA.y*dsB.y` term (`mmq.hip:399` / `:459`)
  is byte-identical code on the same header bytes. Outcome A below needs no
  other zp work.
- **Codes:** `C16[q] = round-half-to-even(16·C[q])`, `C = L − 7.5` (§3.1 for the
  host derivation; LE packing `byte 4j+k = level 4j+k`, same convention as the
  E4M3 LUT at `lloyd_lut.rs:92-94`).
- **Signedness — composer verifies at `mmq.hip:291-292`, then takes exactly one arm:**
  - **Outcome A** (intrinsic accepts a signed weight-side operand): codes are the
    signed bytes `S[q] ∈ [-120,120]` (fits i8 with 7 headroom — bounds follow
    from `L ∈ [0,15]`). `dmA.x = sc'`, `dmA.y = zp'`; correction code untouched.
  - **Outcome B** (weight-side operand must stay unsigned): codes are
    `U[q] = S[q]+120 ∈ [0,240]` (the literal "u8 codes 0..240"); the −120 bias
    folds once at the `make_half2` site as `dmA.y' = zp' − 120·sc'` (same
    `dmA.y*dsB.y` term shape — the bias is row-constant, exactly like zp).
  Either way the uniform (non-`HIPFIRE_MMQ_LUT`) text is unchanged and headers
  are untouched. Abandon criterion: if neither arm validates bit-exact against
  the per-32 X-quant oracle (U2 gate), stop — do not invent a third encoding.
- **Determinism:** accumulation order, WMMA sequence, and f32 reduction are
  untouched → deterministic, atomics-free, hipGraph/replay-safe like the uniform
  MMQ path (which, unlike the FP8-LUT launchers, carries no eager-only gate —
  compare `gemm.rs:29393+` region). Bit-exactness vs the F16-LUT decode path is
  **not** claimed (near-center C16 rounding, §3.1); Lloyd ships under the WT2
  KLD gate per §5, with kill switch `HIPFIRE_LLOYD_MMQ_OFF=1` restoring
  fail-closed (naming TBD by Main; default-ON only after the gate passes —
  non-bit-exact ships flagged, never silent).

### 2.3 Kernel-source checklist (U2 contract)

In `kernels/src/gemm_mq4g256v2_residual_mmq.hip`, all inside
`#ifdef HIPFIRE_MMQ_LUT` except the untouched uniform text:

1. `spread4`/`lut8` helpers (copied shape from `fp8.gfx12.hip:76-95`, C16 args).
2. `load_mq4v2_tile` LUT arm: replace `:320-329` expand with spread+lut8 per
   32-bit nibble word; `sc'` at `:342-344` (Outcome A) or additionally `dmA.y'`
   (Outcome B).
3. Six entry symbols `*_lloyd` (base, x128, full_add/set × plain/x128;
   uniform entries `mmq.hip:543-691` unchanged) each taking +4 u32 kernargs.
4. New `*_SRC_*_LUT` consts in `kernels.rs` via `concat!("#define
   HIPFIRE_MMQ_LUT 1\n#define ...", include_str!(same file))` — additive only,
   existing consts (`kernels.rs:3345-3346`) byte-unchanged; distinct MODULE
   string in the launcher (uniform uses `"gemm_mq4g256v2_residual_mmq"`,
   `gemm.rs:18420`) so the code-object cache cannot alias.
5. Grid/block/shared-mem formulas (`gemm.rs:18442-18455`) unchanged — C16 rides
   kernargs, LDS budget identical.

**NOT doing:** ldsstage / mw_lds / ksplit / BT LUT twins (same rationale as the
prior plan's out-of-scope list — small-N is launch-bound, large-N rides MMQ;
those bands fail closed, never fall back to uniform); iu4 twin (impossible,
§0); lm_head special-casing (single-weight site routes the residual MMQ-LUT
launcher; M=248320 %128==0, K=2048 %256==0 — full tiles).

## 3. Runtime: derivation, plumbing, launchers, prefill branches

### 3.1 `C16 [u32;4]` derivation in `lloyd_lut.rs`

New `pub fn lloyd_c16_from_levels(levels: &[f32; 16]) -> [u32; 4]` beside
`lloyd_luts_from_levels` (`lloyd_lut.rs:99-113`, signature frozen — do not
extend its return triple; existing callers `hfq.rs:1673, :1750-1752, :1821-1824`
stay byte-identical). Packing: `out[i/4] |= code << ((i%4)*8)`, code = LE byte
of level `i` (mirrors the e4m3 arm `:100-105`).

- **Artifact constraint (fixed):** sidecar `L` in [0,15] units, E4M3-snapped by
  the constrained fit (`lloyd_lut.rs:15-18`, `:27-28`); loader centers
  `C = L − 7.5`, "bit-exact in f32 for on-grid sidecars" (`:17`).
- **Exactness proof (|e| ≥ 1):** write on-grid `L = 7.5+e`, `e` finite E4M3,
  `|e| < 8`. `fl(7.5+e)` is exact (≤12 significant bits: 7.5 contributes 2²..2⁻¹,
  E4M3 `e` extends at most to 2⁻⁹ — fits f32's 24 bits); `fl(L−7.5) = e` exactly
  (true difference representable ⇒ rounded result is `e`); `16·e` is an exact
  power-of-two scale; E4M3 steps at `|e| ≥ 1` are 2⁻³/2⁻²/2⁻¹ multiples, so
  `16·e` is an integer — `round()` is the identity. Codes exact, error 0.
- **Rounding rule (near-center, |e| < 1):** E4M3 steps there are 2⁻⁹..2⁻⁴, so
  `16·e` is fractional (halves possible, e.g. 1.5). Rule: round-half-to-EVEN
  (`f32::round_ties_even`), error ≤ 0.5 in 16C-units = `|Δw| ≤ sc/32` per
  weight. Note the mass peak sits at E4M3-zero (`lloyd_lut.rs:27`), so rounding
  is the common case, not the exception — same class of approximation as the
  shipped X128 coarser-X-quant (KLD-gated, `CHANGELOG.md:13`), strictly finer
  than the FP8 path's E4M3 rounding it replaces on gfx11.

### 3.2 Plumbing: `lloyd_lut_c16` alongside the two existing LUTs

- `WeightTensor.lloyd_lut_c16: Option<[u32;4]>` next to `lloyd_lut_e4m3` /
  `lloyd_lut_f16` (`llama.rs:522-532`), filled at the same loader sites that
  fill the pair (`hfq.rs:1749-1752`, `:1821-1824`; qwen35 `load.rs:1018-1027`,
  lm_head `:2655-2670`), `None` at every other construction site (e.g.
  `llama.rs:3580-3581` and siblings — composer enumerates all
  `lloyd_lut_e4m3: None` sites and adds the third field).
- `WeightRef.lloyd_lut_c16` next to `:46-47` in
  `crates/hipfire-dispatch/src/families/gemv.rs:45-47`; wired in
  `WeightTensor::dispatch_ref` (`llama.rs:577-594`, fields at `:592-593`).
  (GEMV decode does not consume C16 — field exists so prefill launchers pulling
  `WeightRef`s see one uniform shape; decode fail-closed on `None` f16 at
  `gemv.rs:505-512, :569-576, :622-629` is untouched.)

### 3.3 Launchers: `*_mmq_*_lloyd` twins of the prequant launchers

New core `gemm_mq4g256v2_mmq_prequant_lloyd(..., c16: [u32;4])` beside
`gemm.rs:18384-18474` (same arch/K gates `:18394-18405`, same full/base/x128
select `:18407-18419`, distinct MODULE + `*_lloyd` symbol, 11-arg kernargs), with
thin `set`/`add` wrappers twinning `:18476-18498`. Four family launchers
(mirroring the FP8-`*_lloyd` structure but with MMQ predicates, **no
exact-gfx1201 gate, no eager-only gate**):

- `gemm_qkvza_mq4g256v2_mmq_lloyd` (twin of the FP8 launcher gated at
  `gemm.rs:30331-30332`): arch gfx1100|gfx1151, K%256, batch%128==0 (+batch≥128
  per the uniform predicate `gemm.rs:27975-27979`), per-tensor C16 at the four
  `set` sites (uniform pattern `gemm.rs:28007-28024`).
- `gemm_qkv_mq4g256v2_mmq_lloyd`, `gemm_gate_up_mq4g256v2_mmq_lloyd`,
  `gemm_hfq4g256_residual_mmq_lloyd` — same, over the uniform patterns
  `gemm.rs:28516-28519` / `:29621-29622` / `:31160-31162`.
- **GDN beta/alpha M=48 tails:** the uniform route diverts M<128 to the MW4 f16
  small-tail (`gemm.rs:28010-28024`; XTX 61..90 us vs MMQ 373..458,
  `CHANGELOG.md:14-21`). That kernel decodes the uniform grid — Lloyd must
  NEVER route there (silent corruption; backstop `families/gemm.rs:263-270`).
  v1: Lloyd beta/alpha ride MMQ-LUT **base** (correct, wastes ~⅔ of the 128-row
  tile — priced in §4). A small-tail LUT twin is follow-up material gated on
  >1% in-model A/B. Fail closed if M≥128-but-not-%128 shapes ever appear where
  the launcher assumed full tiles (base handles them; the guard is for the
  LUT-correctness argument, which needs none — single source per launch, §2.1).
- Below-MMQ-threshold (batch<128 or %128≠0) Lloyd prefill on gfx11 **fails
  closed** with an error naming the predicate (no F16 fallback exists — that
  direction is dead, §0). Escape hatches restoring per-token: the
  `HIPFIRE_MQV2_GFX11_WMMA=0` / `HIPFIRE_PREFILL_BATCHED=0` pair cited in the
  prior plan (line anchors carried, re-verify at this HEAD in U4).

### 3.4 Prefill branches: 7 sites, gfx11 → MMQ-LUT, gfx12 keeps FP8-LUT

At every current `_lloyd` arm — epilogue plain (`prefill.rs:184-192`), epilogue
Partial (`:293-299`), LA qkvza (`:4637-4660`), LA gate_up (`:5507-5522`), FA qkv
(`:5923-5942`), FA gate_up (`:6685-6700`), single-weight (`:9056-9069`) — plus
helpers `lloyd_e4m3_or_fail` (`:2814-2825`) and `all_mq4v2_lloyd` (`:2832-2834`):

- Add `lloyd_c16_or_fail(w, family)` beside `:2818-2825` (same message shape,
  names `lloyd_lut_c16`).
- Each arm becomes: `if arch is gfx1100|gfx1151 → corresponding *_mmq_*_lloyd
  launcher with per-tensor C16; else → existing FP8-LUT call unchanged`
  (gfx12 byte-identical; other arches error exactly as today).
- FWHT-input matchers (`:4409-4410`, `:5029-5030`, `:5356-5357`, `:5604-5605`,
  `:5730-5731`, `:6221-6222`, `:6538-6539`, `:6776-6777`) already include qt52 —
  no change. Mixed Lloyd/uniform refusals (`:4662-4671`, `:5524-5531`,
  `:6702-6709`) stay; add the missing dedicated qkv mixed-refusal at the
  `qkv_same_dtype` fallthrough (`:5944`) — mandatory, the prior plan flagged it
  (§2.5) and this plan lands it with U4.

## 4. Perf/cost model (predictions, then measurements that can kill them)

- **vs uniform q8-x MMQ (the only comparison that matters):** identical WMMA
  sequence, identical LDS traffic, identical (d,s) correction; delta is
  loader-side only — spread (`fp8.gfx12.hip:76-79`) + two `lut8` (each 2
  `v_perm_b32` + select/mask, `:88-95`) per 8 weights, replacing the
  shift/mask/OR expand (`mmq.hip:320-329`). The loader is memory-bound (136
  B/group weight stream + Q8_1 activation tiles + `__syncthreads` per kb at
  `mmq.hip:497/501/509/512`); prediction: **<3% per family**, matching the
  contract target. Kill criterion: any family >3% slower than its own uniform
  MMQ A/B cell investigates once (occupancy/LDS spill check), then abandons the
  twin for that family (that family stays fail-closed on gfx11).
- **vs xt-iu4:** iu4 skips expansion entirely (raw nibble copy,
  `iu4.gfx11.hip:204-205`) with halved A-side LDS (`:12-14`) — measured
  **+32% pp512 on XTX (1203.6→1588.8 tok/s, `CHANGELOG.md:8`)**. Lloyd cannot
  ride iu4 (§0); this is the rung Lloyd gfx11 prefill will not reach. Do not
  benchmark against it as a target; cite it as the documented gap.
- **Anchors for the A/B sheets:** uniform MMQ 1741 us N=512 (contract); X128
  default XTX pp512 990→1141 (`CHANGELOG.md:13`); FP8-vs-F16 on gfx12 pp512
  1387 vs 841 (`CHANGELOG.md:12`); rows=1-vs-2 decode +0.35% noise
  (`gemv.rs:7819-7822`); GDN-tail MW4-vs-MMQ Halo 38..50 us vs 284, XTX 61..90
  vs 373..458 (`CHANGELOG.md:14-21` — bounds the v1 beta/alpha waste: 2 tails ×
  ~0.3 ms/chunk at N=512, visible but bounded; the follow-up tail twin exists
  only if in-model A/B attributes >1% to it).
- **Amdahl honesty:** prefill GEMM choice moves total throughput tens of percent
  on gfx12 (1387-vs-841 precedent); on gfx11 the twin's upside is *enabling*
  Lloyd prefill at ≈MMQ rate, not beating uniform. Report tok/s deltas *and*
  per-kernel us deltas; a tok/s win without the matching kernel-us win is a
  measurement artifact until proven otherwise.

## 5. Bounded composer units (disjoint files, frozen contracts, verification)

Frozen cross-unit contracts: C16 = `[u32;4]`, LE byte packing `out[i/4] |=
code<<((i%4)*8)`, code = `round_ties_even(16·(L−7.5))` ([Outcome A] signed
`[-120,120]` bit patterns; [Outcome B] +120 bias — U2 resolves, U1 emits the
unbiased integer array either way); kernel symbols
`gemm_mq4g256v2_residual_mmq{,_x128,_full_add,_full_set,_full_add_x128,_full_set_x128}_lloyd`;
MODULE `gemm_mq4g256v2_residual_mmq_lloyd`; launcher names §3.3; kill switch
`HIPFIRE_LLOYD_MMQ_OFF=1`.

- **U1 — C16 derivation + plumbing.** Files: `crates/hipfire-runtime/src/lloyd_lut.rs`
  (new fn + unit tests only), `crates/hipfire-runtime/src/llama.rs` (field +
  `dispatch_ref` + `None` at all other ctors), `crates/hipfire-dispatch/src/families/gemv.rs`
  (`WeightRef` field), `crates/hipfire-runtime/src/hfq.rs` + `crates/hipfire-arch-qwen35/src/qwen35/load.rs`
  (fill at existing LUT sites). No kernel, no dispatch logic. Verify: unit tests
  — all 256 E4M3 on-grid codebooks round-trip exact for |e|≥1; tie-rule test for
  a near-center level; `lloyd_levels_from_sidecar` malformed-sidecar still hard-errors.
- **U2 — MMQ kernel LUT twin.** Files: `kernels/src/gemm_mq4g256v2_residual_mmq.hip`
  (`#ifdef HIPFIRE_MMQ_LUT` arms only — uniform text diff must be empty) +
  `crates/rdna-compute/src/kernels.rs` (additive `*_LUT` consts). Depends on U1's
  packing contract only (can start in parallel with U1 using synthetic C16).
  Verify: (1) resolve Outcome A/B at `mmq.hip:291-292`, record the evidence;
  (2) kernel differential vs `gemv_mq4g256v2_lloyd` (`gemv.rs:7824-7844`, LDS LUT
  oracle `gemv_mq4g256v2.hip:78-94, :115-128`) at N=511/512, rel-L2 ≤ 3e-4
  (prior gate precedent); residual ADD arm with **NONZERO Y0** (else `+=` is
  untested); (3) first step before any edit: measure uniform-MMQ baselines per
  family shape on the XTX (kill/prioritize per §0).
- **U3 — Launchers.** Files: `crates/rdna-compute/src/gemm.rs` only (new core +
  set/add wrappers + 4 family launchers; no existing fn body changes).
  Depends on U1+U2 contracts. Verify: shape-guard unit tests (wrong arch / K%256 /
  batch predicate / missing C16 all fail closed with naming errors); CPU-f64
  oracle rel-L2 on one shape per family; uniform MMQ byte-identical (existing
  tests untouched, none re-pinned).
- **U4 — Prefill routing.** Files: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
  only (7 arch branches + `lloyd_c16_or_fail` + qkv mixed refusal). Depends on U3.
  Verify: dispatch trace shows MMQ-LUT on gfx1100/gfx1151 and FP8-LUT on gfx12
  for the same artifact; 1-chunk WT2 KLD prefill-vs-per-token on the XTX;
  gfx12 path bit-identical before/after (no uniform-route change).
- **U5 — Bench + gates (no code).** `hipfire bench --matrix --pp 512,8192 --ctx
  128,2048` (command shape precedent `gemv.rs:7819-7822`), interleaved A/B,
  claim-scoped harness per `docs/VALIDATION.md:231-268` (arch-port row `:268`;
  retired coherence batteries `:275-280` are never acceptance). Evidence rules
  (role non-negotiables): ≥3 fresh-process runs, prompt md5 + binary md5,
  byte-identical prompts, eyeballed decoded output (tight stddev / high τ =
  attractor warning), graph-capture correctness alongside tok/s. Promotion gate:
  per-family within 3% of uniform q8-x MMQ + WT2 KLD ≤ current+0.0005 (X128
  precedent `CHANGELOG.md:13`).
- **hipx execution order:** build once, `ssh hipx`, one daemon per card —
  `HIP_VISIBLE_DEVICES=0` → gfx1100 XTX first, then `=1` → gfx1151 Halo
  (ordering table `docs/investigations/2026-08-03-gfx1151-ds4-longctx-handoff.md:54-64`;
  confirm probe prints the arch). Never touch `/home/kaden/ClaudeCode/warpfront/hipfire-beta`.
  U1–U4 verify scoped (single test file / targeted repro); project-wide suites
  belong to Main once, after all units land.

## 6. Rejected alternatives

- **F16-WMMA-LUT prefill:** measured +61% at N=384, criterion +10% — dead
  (STATUS, `docs/plans/2026-09-15-lloyd-gfx11-port.md:10-19`).
- **Integer-grid re-quant (snap Lloyd levels back to 0..15):** destroys the
  artifact's reason to exist — the constrained E4M3-snapped fit buys +13.9% MSE
  over uniform (`lloyd_lut.rs:23-28`); re-quantizing to the uniform grid pays
  Lloyd's sidecar/plumbing cost for uniform quality. Kill evidence already in
  tree; no experiment needed.
- **2× int8 dot split (crack u8 codes into two i4/i8 halves):** doubles WMMA
  issue count plus a second correction fold on a path whose dot *is* the
  majority bucket at large N — Amdahl caps it below the v_perm approach before
  writing a line. Would need a microbenchmark showing WMMA-issue-bound (not
  LDS/memory-bound) to revive; the MMQ profile says the opposite.
- **F16 shadow weights (keep an f16 copy alongside):** doubles weight VRAM +
  a second upload path for every tensor to avoid a ~2-perm decode; the decode
  LUT exists precisely to avoid this (`gemv_mq4g256v2.hip:78-94` costs 8 dwords
  + 32 B LDS against a ~136 B/row stream). Never competitive on memory-bound
  shapes; rejected without experiment.
- **Per-16-row tile LUT select in MMQ (the gfx12 sketch):** inapplicable, not
  just rejected — §2.1 shows launches are already per-tensor. Any composer
  proposing tile-start-row select inside the MMQ kernel has misread the dispatch
  shape; point at `gemm.rs:28007-28008`.
