# Stage B1b receipt — route-N byte-copy fill + dump (diagnostic)

Base: `8509822db` (B1a). Plan §14.2/§14.3 + Main B1b brief + Main header-direct
ruling (no scratch scale publish; scales via `fa2_scale_n` accessor).
GPU: ordinal 1 (ab1). Device work ran in wt-fa2a warm tree (release).

## Change (one file: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`)

New, all under `#if HIPFIRE_FA2_KMODE == 8 && defined(HIPFIRE_FA2_FP8)`:

- `fa2_scale_n(cache, gkey, kv_h, seq_len)` — header-direct f16 scale
  accessor (row[1024+kv_h*2], stride 1032; OOR reads 0). B4 gives route Q
  the same signature from its plane; B2/B3 call only this.
- `fa2_stageb_nfill_dump<PARTIAL>` — byte-copy fill into fp8 planes
  (K at byte 0, V at byte 16384; 32768 LDS) + dump of planes and
  accessor-read scales to `out`, then return (the f16 body cannot run at
  32768 LDS; B2 replaces dump+return with the fp8 body, keeps the fill):
  - K: one lane per (key,dc16) — 1024/tile, 8/lane — one `global_load_b128`
    + two `ds_store_b64`, with a uniform fast path (key in range) and a
    cold zeroing path. Full tiles never diverge.
  - V: one lane per (sub,dc,lane) fragment — 2048/tile, 16/lane — 8
    `global_load_u8` off ONE row base with constant stride offsets + one
    `ds_store_b64`, with a per-fragment fast path (all 8 keys in range)
    and a cold per-byte-predicated fallback. Full tiles never diverge.
  - Dump per (block,kv_h,split,tile) at
    `((((bx*4+kv_h)*S+s)*512+t)*8320` u32: K plane (4096) | V plane
    (4096) | K scales f32 (64) | V scales f32 (64).
- Entries `attention_fp8_e4m3_fa2_gqa_gfx1201` / `_partial_` (frozen
  kernargs, `q8_codes` unused by the fill) + `..._merge_...` (f16 merge
  body under the stage-b symbol) + `attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201`
  (§3: amax tree, sq=amax/448 f32, `cvt_pk_fp8_f32` pack — same instruction
  as the native writer; numerics gated in B2).
- Old direct/partial/merge selectors nested one level (`#else` + inner
  `#if`); selected text for K0/K3/K8-Q0 unchanged (proven below).

One targeted correction under §11 allowance: the first cut predicated every
byte load on its key (V ≈ 1900/lane, over the abandon line — ISA showed 8
per-byte exec-mask diamonds ≈ 90 of 119 packets/iter). Restructured to
uniform fast/slow paths (full tiles: straight-line). No host impact.

## Gate 1 — K0/K3/K8-Q0 objects bit-identical to 8509822db: PASS

Same-filename `--genco` object `cmp`: all three identical. (Also K0+FP8,
K3+FP8, K8+FP8 TUs compile, 0 warnings.)

## Gate 2 — resources (K8+FP8 TU): PASS

direct 148 VGPR, partial 109, fp8-preconvert 56, merge 17; 0 spill V/S,
0 scratch everywhere; 9–16 waves/SIMD (no drop vs the 6-wave class;
higher because the diagnostic carries no Ofr/sacc state).

## Gate 3 — FILL PACKET CENSUS (the decider): K ≈ 110, V ≈ 730, FILL ≈ 840

Method: gfx1201 `save-temps` ISA of the direct stage-b entry, per-lane
dynamic packets on a full tile (uniform fast paths; slow bodies skipped —
verified uniform by construction; predicated OOR arms counted only as
their not-taken branch packets). Static-census convention (all packets,
not cycles), same unit as §6.4 (K+V per participating lane: slice-A
1216+1160 = 2376 ✓).

- K: 8 unrolled iters × 11–13 = **≈110** (1×`global_load_b128` +
  1×fused `ds_store_2addr_b64` covering both fragments + u64 addr + waits
  + uniform branch). Forecast 60–90: over ~1.3× (address/wait reality).
- V: 16 × ~46 = **≈730** (8×`global_load_u8` off one base with `off`
  stride immediates in `s_clause` batches + 1×`ds_store_b64` + one u64
  MAD + shift/or assembly + waits + uniform branches; verified by eye on
  two independent iters). Forecast 300–420: over ~2× (waits + assembly
  the plan did not cost).
- **FILL ≈ 840 vs nominal 600 (1.4×) and the ≥1200 abandon line: UNDER
  with 30% margin. Abandon NOT triggered.** §6.4 interpolation at 840 ≈
  0.63 of baseline attention work (c16=8, r=1.809). Slice-A fill 2376 →
  0.35×; Q0 decode fill ~13k → 0.065×. B2/B3 proceed on fill economics.

## Gate 4 — device fill oracle: PASS, 0 mismatches

Throwaway `crates/rdna-compute/examples/tmp_b1b_nfill_oracle.rs`
(UNTRACKED, delete before commit; vehicle per StageBHost recipe:
`ensure_kernel_public` + `launch_kernel_blob`, in-example SRC, 32768
LDS, grid [1,4,1]). Native K/V from the production fp8 writer, fixtures
seq 64 (full tile) and seq 40 (OOR tail), batch 8, all 4 kv_h; host model
of the frozen fragment layout + header scales.

- full64: compared=131584 mismatches=0
- oor40: compared=131584 mismatches=0
- (per fixture: 4 heads × (32768 code bytes + 128 scales); scales as f32
  bits, OOR keys zero codes + zero scales — proves the accessor too.)

This also proves the odd-row unaligned `global_load_b128` is byte-correct.

## Explicitly NOT gated here (B2/B3)

fp8 QK/PV legs (don't exist), preconvert numerics, partial/merge
production paths (diagnostic entries only), any timing claim.
