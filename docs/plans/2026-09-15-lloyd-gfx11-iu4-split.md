# Liu4: MQ4-Lloyd (qt=52) batched prefill riding the gfx11 iu4 W4A4 MMQ kernels

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, HEAD `29f652a77`.
**This plan file is the only write. No GPU program was run; no source was edited.**
All paths below are worktree-absolute. Anchors were observed live at plan time;
`crates/rdna-compute/src/kernels.rs` **changed under observation**
(5638 → 8575 lines between two reads — MmqLutU1/U2 are concurrently editing
`lloyd_lut.rs` / `llama.rs` / `gemv.rs` / `hfq.rs` / `load.rs`,
`kernels/src/gemm_mq4g256v2_residual_mmq.hip`, and `kernels.rs`).
Treat every `:line` below as approximate: composers MUST re-resolve by symbol
name before editing and coordinate through `hub` (I am `Liu4Planner`).
Anything not directly observed is marked `[INFERENCE]`.

Sibling plan (q8-x MMQ-LUT twin, the baseline this plan compares against):
`docs/plans/2026-09-15-lloyd-gfx11-mmq-lut.md` — cited as `mmq-lut §X`.

> **STATUS 2026-09-15 — K1 gate FAILED; iu4-split direction DEAD.**
> XTX, N=512, M=17408 K=5120: uniform iu4 full_set_occ3 1061 us; split twin
> 2813 us (2.7x) with 256 VGPR + 12-15 spills (rev1) AND 2813 us at 218 VGPR /
> 0 spills (rev2, single accumulator + per-tile fold). Correctness passed
> (rel-L2 6.53e-2 = the uniform kernel's own int4-activation floor). The iu4
> kernel is MAC/issue-bound on gfx11; doubling WMMAs costs 2.7x regardless of
> register pressure. Kernel arm reverted (history: wip commits before this).
> The q8-x MMQ byte-LUT route (docs/plans/2026-09-15-lloyd-gfx11-mmq-lut.md)
> is the gfx11 Lloyd prefill path: same WMMA count as uniform q8, ~1.5x iu4.

## 0. Decision, denominator, and the impossibility proof that bounds the design

**Decision:** build a split twin of the iu4-direct W4A4 kernel
(`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`) that serves Lloyd
tiers on gfx11 at iu4-class rate, behind the existing iu4 opt-in flag.
Gate-first: a one-entry prototype either earns the full family or kills it.

**Why two WMMAs — impossibility of one:** the iu4 intrinsic takes 4-bit weight
operands (`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32` at `iu4.gfx11.hip:280-281`,
weight side `signA=false` unsigned per the probe-verified contract at
`:24-25`). A 4-bit operand carries ≤16 distinct levels per WMMA. Lloyd's biased
code `U[q] = S[q]+120 ∈ [0,240]` (241 values; `S` = signed C16 from U1's
`lloyd_lut_c16_from_levels`, `crates/hipfire-runtime/src/lloyd_lut.rs:125-141`)
cannot fit one 4-bit operand. The split `U = 16·H+L` with `H,L ∈ 0..15`
(`H = U>>4`, `L = U&15`) is therefore the *minimal* exact encoding: two
unsigned-iu4 WMMAs per original one, `ΣU·x = 16·ΣH·x + ΣL·x`, bit-exact
integers throughout. No single-WMMA Lloyd-on-iu4 encoding exists; do not look
for one.

**Amdahl denominator (task ground truth, XTX, mq4-xt, pp512,
profile_prefill_qwen35):** iu4 total 320 ms =
`gemm_…_iu4_full_set_occ3` 588 us/call × 272 +
`…_full_add_occ3` 659 us × 128 +
`quantize_int4_mmq_ds128` 75 us × 256 (≈160+84+19 = 263 ms + unlisted rest).
Q8-x MMQ-X128 total 418 ms (full_set_x128 875 us, full_add_x128 967 us).
The split twin attacks only the 588/659 buckets (244 ms, 76% of 320 ms).
**Honest ceiling:** the twin does strictly more work per identical memory
stream than uniform iu4 (2× WMMA + nibble substitution), so it can never beat
588/659 us. Success = Lloyd rows served between uniform-iu4 rate (unreachable
for Lloyd — uniform indices ≠ Lloyd codes, mmq-lut §0) and q8-LUT rate.
Maximum conceivable prize ≈ 418 − 320 = 98 ms (23%) if the twin matched
uniform iu4 — impossible; the realistic prize is whatever fraction of that the
prototype measures. If the prototype lands above the q8-LUT line, the prize is
zero or negative and the work dies (kill criterion, §4).

**Transfer-loss warning:** compare only against hipfire's tuned uniform iu4
(`_occ3` entries, `__launch_bounds__(256,3)`, `iu4.gfx11.hip:374-380`) and the
tuned q8-LUT twin — never a naive baseline.

## 1. The iu4 kernel end-to-end (every claim cited)

File: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`.

- **Purpose/header contract (`:5-28`):** weight nibbles feed
  `wmma_i32_16x16x16_iu4` directly — "no nibble->int8 expansion in the loader"
  (`:5-6`); "copies the 32 raw nibble words per 256-K row straight into LDS
  (no expand VALU)" (`:12-13`); int4 activations via the W4A4 recipe
  (per-128 MSE-clip, `block_i4_128`); consumer issues "TWO iu4 WMMAs (one per
  16 K), same WMMA count as today's iu8 pairs; 8 wmmas accumulate into ONE i32
  C per 128-K half, then ONE fold:
  `f += sc_w*d_x*C + zp_w*d_x*s_x (s_x = exact int sum of the 128 x_q)`"
  (`:17-20`); nibble K-order "packed weight word bit [4m..4m+3] is K element
  m; X nibbles packed identically (even K low, odd K high, two's complement)"
  (`:21-23`), "signA=false unsigned, signB=true signed" (`:24-25`),
  "C map acc[j]=C[2j+(lane>>4)][lane&15] unchanged" (`:25`); "Lanes r and r+16
  load byte-identical fragments (address depends on lane%16 only)" (`:27-28`).
- **LDS/occupancy (`:30`, `:315-317`, launcher
  `crates/rdna-compute/src/gemm.rs:18564-18572`):**
  `(128*18 + 128*44)*4 = 31744 B (vs 57344 B baseline) -> 2 blocks/CU`
  (`:30`); carve `tile_y = smem`, `tile_x = tile_y + IU4_MMQ_X*IU4_TILE_Y_K`
  (`:315-317`); tile constants `IU4_TILE_Y_K 18`, `IU4_TILE_X_K 44`,
  `IU4_X_QS_WORDS 32`, `IU4_X_DM_OFF 32` (`:47-50`); grid
  `[row_tiles, batch_tiles, 1]`, block `[32,8,1]`, shared-mem formula at
  `gemm.rs:18569-18580`.
- **Register pressure notes in the kernel header:** there are NONE — the iu4
  header (`:5-30`) carries no VGPR/SGPR counts (unlike e.g. the GEMV MQ4
  gfx1100 header). Pressure reasoning therefore comes from first principles
  (§2: +8 live i32 per thread for the second accumulator) and the composer
  MUST record VGPR/SGPR/spills from the compiler log for twin vs uniform and
  confirm `_occ3` still achieves 3 blocks (else occupancy drop is diagnosed,
  not hand-waved).
- **X prelude `quantize_int4_mmq_ds128` (`:74-140`):** same `(X,Y,K,N)` grid
  contract as the Q8_1 prelude, block `(256,1,1)`, "each thread 4 floats",
  grid `[(K+1023)/1024, N]` (`:62-64`, launch at
  `crates/rdna-compute/src/scratch.rs:1261-1277`); per-128 amax, 8-candidate
  MSE-clip grid `d_j=(amax/7)*0.5*(1+j/7)`, `rintf`, clamp `[-8,7]`, strict-`<`
  argmin (`:96-117`); `|x| ≤ 8` worst case (clamp at `:105`, `:123-124` —
  NOTE: §2 headroom uses 8, not the ticket's 7, because −8 is representable);
  zero group → `d=1, q=0, s=0` (`:66-67`, `:96`, `:122`); `s` is an exact
  integer reduction (`:126-129`); packing even-K-low/odd-K-high two's
  complement (`:134-135`); `d`/`s` written once per 128-group (`:136-139`).
  Struct `block_i4_128` (`:55-60`): `f32 d`, `i32 s`, 64 B nibbles = 72 B.
  Cache launcher `ensure_int4_mmq_x` at `scratch.rs:1210-1235` (dedicated
  `int4_mmq_x_scratch` — "the layouts differ (72 B vs 144 B)", `:1207-1209`).
  Profile cost 75 us × 256 — untouched by this plan (X side identical).
- **Weight loader `load_iu4_tile<FULL>` (`:206-243`):** raw copy
  `x_qs[i*44 + txi] = *(gp + 8 + txi*4)` — 32 nibble words/row, no expand
  (`:220-225`); header loop "identical to the iu8 path (dual-fp16 headers,
  ksc select)" (`:205`): `hs` → `sc`/`zp` via `__half2float(__ushort_as_half(…))`
  (`:234-236`), `dm = make_half2(sc, zp)` replicated 4× (`:237-240`).
  Twin hooks: `dm` construction site (`:237-240`) is where `sc' = sc/16` and
  the −120 bias fold land (§2); the raw copy (`:224`) is untouched
  (substitution happens on fragments, not in LDS).
- **Consumer `vec_dot_i4_x128` (`:247-296`):** per 32-K block (`t<4` loop,
  `:273`), loads `av0/av1` (weight, `:276-277`) and `bv0/bv1` (X, `:278-279`),
  issues TWO iu4 WMMAs accumulating into ONE `int32x8_t acc` declared per
  `(j0,n)` (`:271`, `:280-281`). **Accumulator-in/out semantics:** `acc` is
  passed as the C operand and reassigned both lines (`:280-281`, trailing
  `false` = no saturation — cite as no-sat); therefore separate `accH`/`accL`
  accumulate across `t` with zero extra WMMAs (§2 choice). Fold
  (`:283-292`): `d_x`/`s_x` loads (`:283-284`), per-lane `dmA`
  (`:289-290`), `sum += dmA.x*d_x*(float)acc + dmA.y*d_x*(float)s_x`
  (`:291-292`) — the exact shape the bias fold extends.
- **Body/epilogue (`:299-372`):** kb loop over 256-groups, A-tile once +
  two X-halves (`h=0/1`, `:321-344`) with `__syncthreads` pairs
  (`:330-333`, `:341-343`); guarded Y-tile fill (`:327`, `:338`) and guarded
  store with `add` select (`:360-368`); base entry handles tails structurally
  (`FULL=false` clamps at `:222`, `:231`; Y guards at `:327`, `:360`).
- **Entries (`:374-380`):** 5 symbols — base `…_iu4` (occ2), `…_full_add`,
  `…_full_set` (occ2), `…_full_add_occ3`, `…_full_set_occ3` (occ3).
  Launcher select at `gemm.rs:18534-18539` (full→`_occ3`, tail→base);
  MODULE `"gemm_mq4g256v2_residual_mmq_iu4"` at `:18540`, source const
  `kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC` at `:18541-18544`
  (defined `crates/rdna-compute/src/kernels.rs:3351-3352`).
- **Uniform-iu4 dispatch (routing order precedent):** iu4 opt-in checked
  ABOVE the Q8_1 route in all four families — qkvza `gemm.rs:27983-28004`
  (incl. `small_tail_set_iu4` for M<128 beta/alpha at `:27987-27990`,
  `:27996-27999`), qkv `:28525-28530`, gate_up `:29633-29639`, residual
  `:31155-31158`. Gate: `gfx11_mmq_iu4_enabled()` at
  `crates/rdna-compute/src/feature_flags.rs:733-736` (explicit opt-in AND
  exact gfx1100/gfx1151), flag parsed at `:465`
  (`HIPFIRE_GFX11_MQ4V2_IU4`), doc at `:42-45`.
- **Uniform q8-x reference points (twin comparison):** dot is
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` via `mma_i8` at
  `kernels/src/gemm_mq4g256v2_residual_mmq.hip:291-292` (both flags `true`);
  X128 fold `dmA.x*ds.x*C + dmA.y*ds.y` at `:453-459` with the "8 i8-WMMA
  dots of the half accumulate in ONE i32 C_frag" contract at `:407-411`;
  launch_bounds occ2 at `:667-681`; LDS `(128*36+128*76)*4` at
  `gemm.rs:18442-18449`. (Line anchors in the uniform file per mmq-lut §§1-2;
  U2 is editing that file — re-verify.)

## 2. The split twin: exact spec

New `#ifdef HIPFIRE_MMQ_IU4_LUT` arms inside the iu4 file only, distinct
symbols (`*_lloyd`, `*_lloyd_full_add_occ3`, … — 5 entries mirroring
`:376-380`). Uniform (non-ifdef) text MUST stay byte-identical. No new file:
one twin, all families share the single-source consumer (same per-tensor
launch argument as mmq-lut §2.1 — qkvza fires per-tensor `set` calls at
`gemm.rs:27985-28003`, qkv three at `:28527-28529`, gate_up two at
`:29635-29638`, residual one at `:31157`; each launch carries one codebook).

### 2.1 Kernargs: C16 unchanged, HL derived in-shader prologue

Launcher passes the SAME `c16: [u32;4]` U1 already provides
(`lloyd_lut_c16_from_levels` at `lloyd_lut.rs:125-141`: signed
`round_ties_even(16·(L−7.5)) ∈ [-120,120]` as i8 two's-complement patterns,
`out[i/4] |= byte<<((i%4)*8)`, `:137-138`) — 7 args → 11, same growth as the
q8 twin (mmq-lut §2.3). Kernel prologue (once per threadblock, negligible)
derives ONE combined HL byte-table `T[q] = (H[q]<<4)|L[q]`, `q ∈ 0..15`,
from `U[q] = (S[q]+120)`, `H = U>>4`, `L = U&15` (`S[q]` = signed C16 byte):
16 bytes = 4 dwords in SGPRs. [INFERENCE: combined table strictly dominates
two separate H/L tables — one lookup yields both nibbles; two-table variant
is the documented fallback if the compiler spills the prologue.]

### 2.2 Nibble substitution: exact instruction sequence

Applied to each loaded weight fragment dword `W` (8 nibbles = 8 K-elements,
order preserved — substitution is per-nibble so the `:21-26` K-order contract
is invariant):

1. `lo = W & 0x0F0F0F0F` (even nibbles as bytes), `hi = (W>>4) & 0x0F0F0F0F`
   (odd nibbles as bytes) — 3 VALU.
2. `plo = v_perm_b32(T01, T23, lo)`, `phi = v_perm_b32(T01, T23, hi)` where the
   16-entry byte table `T` is split across the two perm sources (entries
   0..7 / 8..15): the control byte IS the index (bit 3 selects the source
   half, low 3 bits the byte) — 2 perms, no sub/cmp/blend. Each result byte
   = `(H<<4)|L` for one code.
3. Split: `Hlo = (plo>>4)&0x0F0F0F0F`, `Llo = plo&0x0F0F0F0F`, same for `hi`
   (4 VALU); repack `Hword = Hlo | (Hhi<<4)`, `Lword = Llo | (Lhi<<4)`
   (4 VALU: 2 shl + 2 or).

Cost per 8 codes: **2 `v_perm_b32` + ~11 VALU**. Per `(t)` iteration a thread
transforms `av0/av1` (4 dwords = 32 codes): 8 perms + ~44 VALU. Per 128-K half
(4 `t`): 16 dwords = 128 codes → 32 perms + ~176 VALU per thread. VALU hides
under the LDS/VMEM latency the kernel already pays (`__syncthreads` pairs at
`:330-343`, 136 B/group + 72 B X-tile streams) — IF the kernel is
memory/latency-bound (§4 argues it is). If the profile says otherwise, the
prototype kills the work before launchers exist.

### 2.3 Dual accumulation + fold (frozen choice)

- Accumulate `accH` (H-WMMAs) and `accL` (L-WMMAs) as SEPARATE `int32x8_t`
  across the `t` loop, reusing the acc-in/out semantics at `:280-281`
  (each WMMA reads and writes its own acc; no extra WMMAs, no intra-loop
  shift — shifting inside the loop would cost a v_lshl per lane per `t` for
  zero algebraic benefit since `16·Σ ≡ Σ·16`).
- Single combine at the existing fold site (`:291-292`):
  `C = 16*accH_i + accL_i` (exact i32), then
  `sum += sc'*d_x*(float)C + dmA.y'*d_x*(float)s_x`
  with `sc' = sc·(1/16)` (×2⁻⁴, exponent-only — same exactness argument as
  mmq-lut §2.2) and **`dmA.y' = zp − 120·sc'`** (the −120 bias folded into
  the zp/x-sum term, exactly mmq-lut Outcome B; headers on disk/GPU untouched
  — they are shared with the q8 path and U1's `apply_lloyd_centering` at
  `lloyd_lut.rs:188`). Device-side: 3 extra flops per header at the `:237`
  `make_half2` site under ifdef. Full algebra:
  `w ≈ sc·S/16 = sc'·(U−120)`, `Σw·x = sc'·(16·C_H + C_L) − 120·sc'·s_x·d_x`.
- **Bit-comparability with the q8-LUT twin:** both compute the same integer
  `C = ΣU·x` (q8: one i8-WMMA sum; split: `16·accH+accL` — identical integer
  by distributivity), the same `sc'`, the same `dmA.y'`, the same fold shape.
  If U2's twin lands with Outcome-B op order, the two twins MUST agree
  bit-exact (same f32 op order at the fold — composer verifies and records;
  any mismatch is a bug in one of them, and the pair is each other's
  strongest oracle).
- **VGPR delta:** +8 live i32 per thread (second acc; `acc` is declared inside
  the `n` loop at `:271`, so the extra set is transient per `(j0,n)`).
  LDS unchanged (substitution on registers, not LDS). `_occ3` re-verified by
  compiler log (§5 K1).

### 2.4 int32 headroom proof (worst case)

Per 128-K half, per lane: `accH, accL ≤ 128·15·8 = 15,360` (128 terms,
nibble ≤ 15 unsigned at `:24-25`, `|x| ≤ 8` from the `:105`/`:123-124`
clamp). Combined `16·accH+accL ≤ 17·15,360 = 261,120 « 2³¹` (8,200× margin;
f32 conversion exact, `< 2²⁴`). Even a hypothetical full-K accumulation at
residual K=17408: `17408·240·8 = 33.4M « 2³¹` (64× margin). No saturation
handling (no-sat flag at `:280-281`). Beyond stating this bound, no handling
needed.

## 3. Boundedness at pp512: MAC vs LDS vs VMEM, prediction, kill line

- **Structural fact:** iu4 and q8-x issue the SAME WMMA count per 128-K half
  (iu4: 2/`t` × 4 `t` at `iu4.gfx11.hip:273-282`; q8-x: 8/half, mmq-lut §1.2).
  Weight bytes identical (136 B/group both). X bytes halved in iu4
  (72 B vs 144 B per 128×batch, `block_i4_128` at `iu4.gfx11.hip:55-60`).
  A-side LDS halved (31744 B → 2 blocks/CU at `:30` vs 57344 B → 1 block/CU).
- **Inference from measurement:** iu4 full_set 588 us vs q8-x full_set_x128
  875 us (−33%) with identical MAC counts ⇒ the q8 kernel is NOT MAC-bound;
  the gain comes from the halved X stream and doubled occupancy ⇒ at pp512
  the family is **VMEM-bound (X side) and/or latency/occupancy-bound**, not
  MAC-bound. The add pair (659 vs 967, −32%) and the halved prelude (75 us
  × 256) corroborate.
- **Prediction for the split twin:** same streams and occupancy as iu4, but
  2× WMMA (16/half) + §2.2 VALU. If the MAC fraction `f` of 588 us is small
  (consistent with the −33% memory-driven win), twin ≈ `588·(1+f) + ε`,
  plausibly 650–850 us — under the q8-LUT line (~875·1.03 ≈ 901 us /
  ~967·1.03 ≈ 996 us). If `f` is large, the twin flips the kernel MAC-bound
  and lands above the line — that flip is THE central risk, and it is
  measured, not debated (K1 prototype).
- **Kill criterion (frozen):** single-entry prototype (`full_set_occ3` only,
  synthetic HL table, §5 K1) per-call time **> q8-LUT twin's per-call time on
  the same shape → DEAD** (no reason to exist: q8-LUT serves identical codes
  at that rate). Prototype must clear with margin (target ≤ 850 us vs the
  901 line) to earn the remaining 4 entries + launchers + routing. Secondary
  kill: prototype KLD ≠ q8-LUT KLD beyond noise (same math ⇒ must match), or
  `_occ3` lost to spills with no recovery.

## 4. Runtime plumbing delta (relative to mmq-lut §3)

Shared with the q8 twin (already landed at plan time — reuse, do not rebuild):
C16 derivation (`lloyd_lut.rs:125-141` + exactness tests `:326-395`),
`WeightTensor.lloyd_lut_c16` (`llama.rs:527-534`, wired in `dispatch_ref`
at `:595-596` and decode call sites `:801-802`, `:1330-1331`, `:1372-1373`,
`:1403-1404`, `:1544-1545`), `WeightRef.lloyd_lut_c16`
(`families/gemv.rs:45-49`; GEMV oracle `gemv_mq4g256v2_lloyd` at `:514`),
`None` at non-Lloyd ctors (`llama.rs:3590-3591` and siblings).
Delta for Liu4:

- **Launchers (mirror, do not touch, the iu4 family):** new core
  `gemm_mq4g256v2_mmq_prequant_iu4_lloyd(..., c16: [u32;4])` beside
  `gemm.rs:18508-18597` (same arch/K gates at `:18518-18532`, same full/base
  select at `:18534` — NOTE iu4 has no x128/per-32 select, single prelude
  `ensure_int4_mmq_x`; distinct MODULE
  e.g. `gemm_mq4g256v2_residual_mmq_iu4_lloyd`; `*_lloyd[_full_add|_full_set][_occ3]`
  symbols; 11-arg kernargs; grid/block/LDS formulas at `:18562-18580`
  unchanged), thin set/add wrappers twinning `:18599-18621`, four family
  launchers over the uniform-iu4 call sites (`:27983-28004`, `:28525-28530`,
  `:29633-29639`, `:31155-31158`) passing per-tensor C16. X prelude is
  SHARED — `ensure_int4_mmq_x` (`scratch.rs:1210-1235`) needs no twin.
  Tails: Lloyd MUST NOT take `gemm_mq4g256v2_small_tail_set_iu4`
  (`gemm.rs:27920-27946` — MW4 f16 uniform-grid decode, silent corruption for
  Lloyd, same prohibition as mmq-lut §3.3); v1 rides the iu4 BASE entry
  (structurally tail-capable: clamps at `iu4.gfx11.hip:222`, `:231`, guards
  at `:327`, `:360` — composer proves with the N=511 differential).
- **Routing order in `prefill.rs` (7 arms, all currently FP8-LUT at
  `:189-192`, `:296-299`, `:4640-4663`, `:5510-5525`, `:5926-5945`,
  `:6688-6703`, `:9069-9072`):** each Lloyd arm becomes
  `if HIPFIRE_GFX11_MQ4V2_IU4 admits (gfx1100|gfx1151, K%256,
  batch predicate) → iu4-LUT launcher; else → q8-LUT launcher`
  (iu4 FIRST — mirrors the uniform order where the iu4 check precedes the
  Q8_1 prelude in all four families, §1; iu4 is opt-in experimental so
  default-off preserves q8-LUT behavior). Helpers: `lloyd_c16_or_fail`
  beside `lloyd_e4m3_or_fail` (`:2820-2829`) per mmq-lut §3.4 (U4 owns that
  helper — K3 coordinates, single owner per arm). gfx12 arms byte-identical.
- **Fail-closed rules:** missing C16 → error naming the tensor (mirror
  `:2825-2826` message shape); wrong arch / K%256 / batch predicate →
  same errors as the uniform iu4 launcher (`gemm.rs:18518-18532`);
  below-MMQ-threshold (batch<128 or %128≠0) → error naming the predicate
  (no F16 fallback — dead per mmq-lut §0; no MW4-tail fallback — uniform
  grid); kill switches: `HIPFIRE_LLOYD_MMQ_OFF=1` (shared with q8 twin,
  mmq-lut §2.2/§5) restores fail-closed for both; iu4-split additionally
  requires `HIPFIRE_GFX11_MQ4V2_IU4=1` (default-off, `feature_flags.rs:733-736`).
  Existing misroute backstops (mmq-lut §1.3) untouched.

## 5. Bounded composer units (frozen contracts, per-unit verification)

Frozen cross-unit contracts: C16 = `[u32;4]` LE nibble…byte packing per
`lloyd_lut.rs:137-138`; combined HL table `T[q]=(H[q]<<4)|L[q]` derived
in-shader prologue; symbols (K1-frozen 2026-09-15, `_lloyd`-suffix form):
`gemm_mq4g256v2_residual_mmq_iu4_lloyd`, `…_full_add_lloyd`, `…_full_set_lloyd`,
`…_full_add_occ3_lloyd`, `…_full_set_occ3_lloyd`;
MODULE `gemm_mq4g256v2_residual_mmq_iu4_lloyd` (code-object cache cannot alias
uniform `"gemm_mq4g256v2_residual_mmq_iu4"` at `gemm.rs:18540`); launcher
names §4; kill switches §4. NO GPU commands by planners; composers run the
scoped proofs below (single test file / targeted repro — project-wide suites
belong to Main once, after all units land; never touch
`/home/kaden/ClaudeCode/warpfront/hipfire-beta`).

- **K1 — kernel twin (gate-first).** Files: `kernels/src/…iu4.gfx11.hip`
  (`#ifdef HIPFIRE_MMQ_IU4_LUT` arms only — uniform text diff MUST be empty)
  + `kernels.rs` (additive `*_IU4_LUT` consts via
  `concat!("#define HIPFIRE_MMQ_IU4_LUT 1\n", include_str!(same file))`,
  mirroring the `-DHIPFIRE_MMQ_LUT=1` pattern; re-resolve anchors — file under
  concurrent edit, §0). Ordered build: (a) resolve sign/operand facts at
  `:24-25`/`:280-281`, record; (b) ONE entry (`full_set_occ3`) + synthetic HL
  table + dual acc + fold + `sc'`/`dmA.y'` → time on XTX vs §3 kill line
  (≤850 us target, >q8-LUT per-call = abandon, no further entries); (c) on
  pass: remaining 4 entries + C16 kernarg + prologue derivation.
  Verify: differential vs `gemv_mq4g256v2_lloyd` (`families/gemv.rs:514`) at
  N=511/512, rel-L2 ≤ 3e-4 (prior gate precedent); residual ADD with NONZERO
  Y0; cross-oracle vs q8-LUT twin (bit-exact expected, §2.3 — mismatch = bug);
  compiler VGPR/SGPR/spill log twin-vs-uniform + `_occ3` occupancy confirmed.
- **K2 — launchers.** Files: `crates/rdna-compute/src/gemm.rs` ONLY (new core
  + set/add + 4 family launchers; no existing fn body changes). Depends on
  K1's frozen symbol/MODULE contract. Verify: shape-guard unit tests (wrong
  arch / K%256 / batch predicate / missing C16 all fail closed with naming
  errors); CPU-f64 oracle rel-L2 one shape per family; existing uniform-iu4
  tests untouched, none re-pinned.
- **K3 — routing.** Files: `qwen35/prefill.rs` ONLY (7 iu4-first branches +
  qkv mixed-refusal if U4 hasn't landed it — mmq-lut §3.4; exactly one owner
  per arm — message U4 via `hub` before touching). Depends on K2. Verify:
  dispatch trace shows iu4-LUT iff `HIPFIRE_GFX11_MQ4V2_IU4=1` on
  gfx1100/gfx1151 else q8-LUT; gfx12 path bit-identical before/after;
  1-chunk WT2 KLD prefill-vs-per-token on the XTX.
- **K4 — bench + gates (no code).** Interleaved A/B per ROLE evidence rules
  (≥3 fresh-process runs — within-session A/B drifts 10–15%; prompt md5 +
  binary md5, byte-identical prompts, eyeballed decoded output, graph-capture
  correctness alongside tok/s) under the claim-scoped harness
  `docs/VALIDATION.md:231-268` (retired coherence batteries never acceptance).
  Promotion: per-call ≤ q8-LUT per-call with margin (target §3) AND WT2 KLD ≤
  q8-LUT KLD + noise (same codes ⇒ must match; else bug). Report tok/s AND
  per-kernel us; tok/s win without kernel-us win = artifact until proven.

## 6. Rejected alternatives

- **Single-WMMA Lloyd-on-iu4 (any encoding):** impossible (§0 — 241 values vs
  16-level operand). No experiment needed; the proof is counting.
- **Signed-weight WMMA (`signA=true`) carrying S directly:** rejected —
  `signA=false` is probe-verified (`iu4.gfx11.hip:24-25`); beyond that, S needs
  8 bits so the H/L split is still required. Changes global nibble semantics
  for zero structural gain.
- **Pre-splitting weights offline into H/L nibble streams:** doubles weight
  VRAM + a second upload path and breaks the byte-identical artifact contract
  (mmq-lut §1: qt52 wire = qt44 bytes, 136 B/group) to save ~176 VALU/thread
  on a memory-bound kernel. Never competitive; rejected without experiment.
- **Two separate H/L kernarg tables instead of C16+prologue:** rejected —
  same 4-dword cost as C16 with a worse contract (launcher does shader work;
  diverges from the q8 twin's kernarg shape). Combined-HL fallback only if the
  prologue spills (§2.1).
- **Single-accumulator with intra-loop shift:** algebraically identical to
  §2.3 (distributivity) at higher per-iteration cost — the intrinsic's acc
  in/out (`:280-281`) makes separate accs free. Not a real alternative;
  recorded so nobody "optimizes" into it.
- **Routing Lloyd to plain iu4 / F16 fallback / MW4 small-tail for tails:**
  corruption (uniform indices), dead direction (mmq-lut §0), uniform-grid
  decode respectively. All fail closed, never fall back.
- **Per-tile LUT select in the kernel (gfx12 sketch):** inapplicable — launches
  are already per-tensor (§1, same as mmq-lut §2.1). C16 rides kernargs; no
  tile-straddle contract needed.

## 7. Explicit non-goals

No `quantize_int4` changes (X side shared); no `occ2`/base-entry removal;
no gfx12 port (RDNA4 lacks the iu4 path — separate source family); no lm_head
special-casing (routes the residual launcher; shapes already %128/%256);
no ldsstage/mw_lds/ksplit/BT twins; no change to any refusal, gate default,
or uniform text/symbol/MODULE; no project-wide test/format/lint runs by
composers.
