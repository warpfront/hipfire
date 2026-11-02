// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `--kernel qwen4_moe_verify`: R3 complete-chain workload for the Flash-Next (qwen4 FN) routed
//! experts at few rows `T in {1,2,3,4,7,8}` (`H=2560, I=640, E=512, top-10`, qt44 MQ4 gate/up +
//! qt53-wire MQ4G128 down), resident (VRAM) or host-mapped (`upload_raw_host_mapped`).
//!
//! Included by the driver with `#[path = "pm_loop/moe_verify.rs"] mod moe_verify;`; the IU4
//! candidate arm lives in `pm_loop/moe_verify_iu4.rs` (`crate::moe_verify_iu4::Iu4Arm`).
//!
//! # Shape grammar and knobs
//!
//! `--shape fn[,T=<1|2|3|4|7|8>][,res=resident|mapped][,route=uniform|skew][,seed=N]`
//! (defaults `T=4,res=resident,route=uniform`). Only the `fn` preset and those T are admitted;
//! other T / presets are refused with the reason (T=5,6 are not refused for a traced reason, they
//! are simply outside the frozen slice). `route=skew` mirrors `moe.rs`: every token picks experts
//! `0..min(5,TOP)` plus uniform others. `--candidate iu4_sym` (required unless `--null-control
//! aa`, refused under aa). No reference sampling knob: the CPU reference is complete (every
//! output element of every token).
//!
//! # Traced shipped chain (default env, eager, no prefill-invariant scope, `wide_verify_rows`
//! irrelevant at T<=8)
//!
//! Paths are relative to `crates/` of this worktree.
//!
//! **Recipe** (b): `MoeRecipe::SoftmaxGatedShared { bf16_round_trip: true, shared_after_combine:
//! true }` for decode (`hipfire-arch-qwen4/src/program.rs:1110-1113`) and grouped prefill/verify
//! (`program.rs:1230-1233`); accessors `hipfire-dispatch/src/families/moe.rs:241-259`. So gate, up
//! and the SwiGLU activation round to BF16 inside the activation kernel (`bf16_rt=1`), the
//! combine rounds every expert row / route weight / product / running sum to BF16 itself
//! (`kernels/src/moe_down_combine_top10_batched.hip`), and there is NO post-combine
//! `bf16_round_trip_f32` (`qt44_qt53_prefill.rs:2388-2392`, `mod.rs:6846-6870`: the shared
//! expert's `bf16_scaled_add` follows in the model; the shared expert is excluded here).
//!
//! **Capability** at `T < 512`: `Qt44Qt53Grouped` (never `...Symmetric`) --
//! `gpu_forward.rs:3199-3209` (`rows >= QWEN4_MOE_SYM_IU4_MIN_ROWS = 512` gemm.rs:49222). The
//! layer is assumed to be a plain uniform QT44/QT53 layer (`slab == None`, `fp8 == false`; the
//! sealed expert-slab route belongs to cold-codec layers and is not modelled).
//!
//! ## T = 2..8 (prefill-family Path 2, sealed grouped program; stage order
//! `moe_program.rs:2063-2076`: scatter, gate_up, unscatter, activation, down, combine)
//!
//! * `use_path2 = moe_grouped_gemm && arch.has_wmma() && (!narrow_verify || ..)`
//!   (`families/moe.rs:2611-2613`): `moe_grouped_gemm` defaults on (`feature_flags.rs:1159-1162`),
//!   `has_wmma` = RDNA3|RDNA4 (`arch_caps.rs:145,457`; gfx1100 HAS wmma). `narrow_verify` needs
//!   `is_verify && has_host_projection` (`moe.rs:2596-2597`) and `is_verify` is
//!   `ctx.workload == SpeculativeVerify` (`moe_program.rs:4115-4122`), which the qwen4 FN forward
//!   never sets (it seals with a `Standard` context: `moe_program.rs:4171-4175`), so host-mapped
//!   experts take the SAME route (g: no call / argument change with residency).
//! * `grouped_path`: `route_rows <= few_row_limit` so `wide_exact_route` is false
//!   (`qt44_qt53_prefill.rs:137-143`, `layer_ops.rs:238-243`). `few_row_limit` (e) =
//!   `FEW_ROW_ROUTE_ROWS = 8` outside a lab verify scope (`prefill_invariant.rs:232-241,311-317`;
//!   `wide_verify_rows` defaults 0, `dispatch.rs:3025`); `route_rows(T) = T` because the
//!   prefill-invariant scope is off in verify/AR (`gpu_forward.rs:332-364`,
//!   `prefill_invariant.rs:108-121`, `prefill_invariant.rs:293-299`).
//! * Off at T<=8: `sym_iu4` (`qt44:1573-1585` -> `gemm.rs:50245-50246`, 512-row floor), `gateup_bf16`
//!   (`qt44:1648-1654` -> `gemm.rs:48639-48645`, 512-token floor `gemm.rs:570`), `down_wmma`
//!   (`qt44:1658-1663` -> `moe.rs:3380-3386`, `10*512` rows), hence `unscatter_rotates`
//!   (`qt44:2032-2039`) is true (`mi % 128 == 0`) and `indexed_down` (`qt44:2128-2130`:
//!   `use_path2 && route_rows <= few_row_limit && !down_wmma`) is true.
//! * (a) input basis `input_basis` (`qt44:456-469`): `gateup_rotates_f16` false (`qt44:533-541`)
//!   -> `Gpu::rotate_x_mq_batched(x_norm, x_rot, gate_up_k = 2560, T)` (`gemv.rs:6798`, symbol
//!   `mq_rotate_x`): FWHT-256 per group, signs seeds 42/1042.
//! * scatter (`qt44:1631-1642`; the `moe_scatter_stable_top10` arm needs `sym_iu4` or EP exchange):
//!   `Gpu::moe_scatter_fused_top10(topk_idx, counts, offsets, sorted_slot_index, tile_ids,
//!   inverse_perm, S = 10T, 512, G, MOE_GROUPED_BLOCK_M = 16)` (`moe.rs:2502`, one CTA).
//!   (d) `G = ceil((S + 15*min(S,512)) / 16) * 16` (`program.rs:391-404`, `1188`); counts `[E]`,
//!   offsets `[E+1]`, sorted `[G]`, tile_ids `[G/16]`, inverse `[S]` all i32
//!   (`kernels/src/qwen4_moe_scatter_fused_k8.hip:19-35`).
//! * gate_up (`qt44:1873-1905`), F32 `y_gu_grouped [G, 1280]`: gfx1151 with `mi==640 &&
//!   gate_up_k==2560 && k_top==10 && 1<=T<=8` -> `Gpu::gemm_mq4g256v2_moe_grouped_top10_inverse`
//!   (`moe_few.rs:26`, entry `..._o4_r2_x4_inverse_k2560`); gfx1100/gfx1201 ->
//!   `Gpu::gemm_mq4g256v2_moe_grouped_top10` (`gemm.rs:48750`; internally, T<=8: WMMA arm off
//!   `gemm.rs:48780`, `has_gfx11_plus_simt && m==1280 && k==2560` -> `..._o4_r8_x4` module, `few`
//!   entry `..._o4_r2_x4`, `gemm.rs:48794-48826`). (f) gfx1100 has no refusal (`use_path2`
//!   holds; no arch-specific arm in this chain).
//! * unscatter (`qt44:1984-2018`): `Gpu::moe_gate_up_unscatter_silu_rotate128_top10(y_gu,
//!   sorted_slot_index, rot_batch, mi = 640, G, bf16_round_trip = true, None)` (`moe.rs:2738`):
//!   gate/up -> BF16, SwiGLU -> BF16, FWHT-128 (signs 43/1043) into `rot_batch [S, 640]` in flat
//!   (token, rank) order. `activation` is then a no-op (`path2_done`, `qt44:2072-2087`).
//! * down (`qt44:2207-2209` recursion to `down_stage(.., false)`, `qt44:2274-2284`):
//!   `Gpu::gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(down_ptrs, topk_idx, rot_batch,
//!   down_expanded, 2560, 640, T, 512, 10)` (`moe.rs:3067`; gfx1151 eager K=640 -> `..._k640_r4`
//!   `moe.rs:3103-3115`) -> unweighted F32 `[T, 10, 2560]`.
//! * combine (`qt44:2329-2394`, indexed arm `2344-2346`, `2376-2387`): `Gpu::moe_down_combine_
//!   top10_batched(down_expanded, topk_idx, topk_w, residual, 2560, T, 512)` (`moe.rs:2216`).
//!   (d) write semantics: IN-PLACE `residual += routed` -- the kernel starts from
//!   `bf16(residual)` (`moe_down_combine_top10_batched.hip:51-52`; `:26` documents `in-place +=`);
//!   `initial_zero` does not apply (`qt44:2295-2305`: needs `down_wmma || sym_iu4`). Production
//!   hands it a zero-filled `ffn_output`; this adapter hands it a non-zero deterministic
//!   residual so the `+=` is visible.
//!
//! ## T = 1 (AR decode family, `select_decode` `moe_program.rs:3048-3105`)
//!
//! Grouped route requires `SoftmaxGatedShared`, `batch_size == 1`, no EP / Paro / deferred combine
//! and `select_grouped_route` (`mod.rs:322-423`: capability admitted, geometry, formats, no
//! tags / sidecars); decode params `program.rs:1107-1128` (`batch_size: 1`, `x_rot_prerotated:
//! false`, `normalization: Provided`). Same four Gpu launchers on every arch, residency-blind:
//! * input basis `decode_input_basis_stage` (`mod.rs:6566-6604`): `Gpu::rotate_x_mq(x_norm,
//!   x_rot, 2560)` (`gemv.rs:6683`, `mq_rotate_x`).
//! * gate_up (`mod.rs:5763-5810`): `Gpu::gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
//!   gate_up_ptrs, topk_idx, x_rot, gate_batch, up_batch, 1280, 2560, 1)` (`gemv.rs:18128`; T<=8
//!   skips the `t8_pm` arm `gemv.rs:18152`; K=2560, m%8==0 -> `..._k2560_rows8_indexed_batched`
//!   `gemv.rs:18290`).
//! * activation (`mod.rs:6060-6100`, bf16): `Gpu::silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(gate,
//!   up, rot_batch, 640, 10, None)` (`gemv.rs:8360`, symbol `mq_rotate_x_128_v2_silu_bf16`; the
//!   real call fuses the shared expert's activation when present -- excluded here).
//! * down (`mod.rs:6208-6263`): the same `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`
//!   with `batch = 1`.
//! * combine (`mod.rs:6846-6870`): the same `moe_down_combine_top10_batched` with `tokens = 1`;
//!   no round trip afterwards (shared add follows: `mod.rs:6880-6884`).
//!
//! Consistency with the gfx1201 route plan `route_plans/qwen3.8-flash-next.gfx1201.tsv` (AR and
//! native_mtp walks): it loads `mq_rotate_x`, `moe_scatter_fused_top10`,
//! `gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4` (`..._o4_r2_x4`, `..._o4_r8_x4`),
//! `moe_gate_up_unscatter_silu_rotate128_top10`, `gemv_mq4g128v2_moe_down_top10_indexed_batched_
//! expanded`, `gemv_mq4g256v2_moe_gate_up_top10_indexed_batched` (+ `..._k2560_rows8_...`),
//! `mq_rotate_x_128_v2_silu_bf16` and `moe_down_combine_top10_batched` (TSV rows 29-53): exactly
//! the modules above (it also lists the grouped `moe_down_combine_grouped_top10` /
//! `qwen4_moe_scatter_stable_top10` / IU4 bundles from the >=512-row prefill walks).
//!
//! Live check: `setup` re-evaluates the public selector predicates on the actual `Gpu`
//! (`route_rows`, `few_row_limit`, `qwen4_moe_sym_iu4_applies`, `qwen4_moe_gateup_wmma_applies`,
//! `qwen4_moe_down_wmma_applies`, `has_wmma`, `moe_grouped_gemm`) and refuses when any differs from
//! the table (e.g. a lab env switch moved the shipped route).
//!
//! ## Not modelled (stated, not guessed)
//!
//! Router (`moe_router_softmax_top10_*`), shared expert, the HC streams, the `qwen4_scope` /
//! `lane_rows_scope` flags of a real verify forward (no launcher of this chain reads them at
//! T<=8: `gemv.rs:18152` and `moe.rs:3118` need `batch > 8`; `gemm.rs:48780` reads `route_rows`
//! only), and CPU-tier / EP / slab layers.
//!
//! # Reference and tolerance
//!
//! `cpu_f64_chain` recomputes the whole routed block on the host from the deterministic
//! generators (independent of every device kernel): input FWHT-256 (signs 42/1042, orthonormal
//! `1/16`), MQ4 dequant `w = sc*(u-8)` (the group headers are symmetric, `zp == -8*sc`), f64 GEMV,
//! the recipe's BF16 round trips of gate/up/SwiGLU, FWHT-128 (signs 43/1043), down GEMV, then the
//! combine's ascending-expert BF16 accumulation into the residual (rank order ties by rank). The
//! only freedom left is F32 reduction order of the device GEMVs (relative `~1e-6`, which flips a
//! BF16 rounding of an activation or of a running sum only occasionally). Baseline /
//! A-A tolerance is per output element `0.0625 * S_i` with `S_i = |residual_in| + sum_j |w_j *
//! o_ji|` (= 16 BF16 half-ulps of the largest partial the combine can hold; the combine itself
//! performs 21 BF16 roundings of partials bounded by `S_i`, a flipped upstream value moves one
//! ulp per step). `RefReport` carries the element with the largest error/bound ratio (its
//! absolute error and its own bound), and passes iff every element is within its bound. The
//! guard tail after the residual must remain the poison words.
//!
//! # IU4 reference: `cpu_iu4_chain (exact recipe; SiLU portable twin)`
//!
//! The IU4 candidate is checked against its OWN recipe, stage by stage, not against the F32 chain
//! (the A4 quantisation distance to the baseline recipe is the driver's
//! `max_abs_diff_vs_baseline`; it is not a pass criterion). The pieces are read from source:
//!
//! 1. gate/up producer `qwen4_moe_rotate256_i4` = `mq_rotate_x_i4`
//!    (`kernels/src/mq_rotate_x_i4.hip:164-217`): `x * signs1` (42), ascending-stride f32
//!    butterflies, `(v * 0.0625f) * signs2` (1042) (`rotate256`), then per 128-half the C2
//!    `block_i4_128` quantiser (`block_i4_128_quant.hip:85-215`; gfx1201 one-pass twin `:290-382`,
//!    documented bit-identical): `amax`, `base = (amax/7)*0.5`, candidates `d0 = base *
//!    0x1.b6db6ep+0f`, `d1 = base * 2`, `q = clamp(rintf(x/d), -8, 7)` (round half to even), `err =
//!    fma(-q, d, x)`, `mse = fma(err, err, mse)` over four values per lane then the xor 16..1
//!    tree, strict `<` against `1e30`, `d = 1`/`q = 0` when `amax == 0` (copied from
//!    `rotate.rs::quantize`, the transcription of `peacemaker-isa/tests/moe_rotate_i4_pm.rs:
//!    515-576`, which is the oracle of the PM twins and equals the HIP wave body above).
//! 2. gate/up GEMM + SwiGLU epilogue: `moe.rs::fold_slots` (exact int32 dot per K128 epoch,
//!    `sum = fma(RN(sc*d), float(C), sum)` from `+0`, `examples/qwen4_moe_sym.rs:262-328`), then
//!    `rt(g)`, `rt(u)` (BF16 round trips) and `moe.rs::portable_silu_mul`, BF16 RNE
//!    (`qwen4_moe_iu4_sym.gfx1151.hip:168-170`). The fold is bit-exact; the SiLU is the portable
//!    twin (`mq_i4_reference.rs:1046-1068`) of the device's `v / (1 + expf(-v))`, which differs
//!    in the last f32 bits.
//! 3. down producer `qwen4_moe_rotate128_i4` (`qwen4_moe_rotate128_i4.hip:32-67`; PM twins
//!    `moe_rotate_i4_gp_pm_gfx1201`, `mq_i4_a4_r128_k640_gfx1100`): BF16 widen, `* signs1` (43),
//!    FWHT-128, `(v * 0.0883883476f) * signs2` (1043) (`rotate.rs::rotate`), the same quantiser.
//! 4. down GEMM: `fold_slots` again, BF16 RNE store (`bf16_bits` of the f32 sum).
//! 5. combine: the grouped BF16 combine of `moe_down_combine_grouped_top10.hip:97-193`, as the
//!    baseline reference does (ascending expert id, BF16 weight / product / running sum),
//!    `residual += routed`.
//!
//! **Tolerance (derived, not a fraction).** Everything above is deterministic and bit-exact
//! except where the portable SiLU twin and the device's `expf` can round a SwiGLU activation to
//! different BF16 neighbours. That can happen only if the twin's f32 value lies within a few f32
//! ulps of a BF16 rounding midpoint (the device evaluates `expf` / `v_exp_f32` + `v_rcp_f32`
//! sequences, each ~1 ulp class, then a correctly rounded product, so the pre-rounding values are
//! expected to differ by a handful of f32 ulps; this bound is an expectation, not a proven
//! device error bound; the reference uses [`AMBIGUITY_ULPS`] = 32 as headroom). Every such activation ("ambiguous") has exactly
//! one alternative BF16 value (the other neighbour). A 1-ulp activation change is not a small
//! output change: it moves the FWHT-128 block, can change the quantiser's `d`/codes
//! (discontinuous), and so the down GEMM row. The reference therefore does not bound it
//! analytically; it ENUMERATES it: for each (slot, K128 block) with `m <= 8` ambiguous
//! activations it recomputes the block's record, the down fold and the combine for every
//! non-empty subset of the `m` flips (`2^m - 1`, the complete admissible set of that block), and
//! the per-element tolerance is the sum over (slot, block) units of the worst `|out_alt -
//! out_nominal|` plus one BF16 ulp of the output when two or more units meet in one token (the
//! combine's BF16 running sum is not additive, so the sum of single-unit deviations is a
//! first-order bound; the extra ulp covers one flipped rounding of the running sum). Where no
//! activation of any of a token's ten slots is ambiguous the tolerance is exactly `0`: the
//! candidate must reproduce the reference BIT FOR BIT. At the Flash-Next shape about 0.6
//! activations per slot are ambiguous; most alternatives change nothing after the BF16 stores, so
//! the resulting tolerance is typically 0 or a few BF16 ulps. A larger ambiguity count than
//! `AMBIGUITY_CAP` in one block makes the reference refuse (error) rather than guess.
//! Unmodelled by construction (a failure here is a finding, not noise): the device quantiser /
//! rotation differing from the sources above, the `rcp`/Newton one-pass path of gfx1201 departing
//! from its stated bit-identity, any chain stage not listed. NOT a quality/KLD gate (promotion
//! needs R6).
#![allow(dead_code)]

use crate::common::{bf16_bits, bf16_val, dev_err, f16_bits, f16_val};
use crate::moe::{fold_slots, portable_silu_mul, rt_f32, XRow};
use crate::moe_verify_iu4::Iu4Arm;
use crate::rotate::{quantize, rotate, GROUP, REC};
use crate::workload::{Arm, ChainCtx, ChainWorkload, Knobs, Object, RefReport, Result};
use peacemaker_isa::kernels::qwen4_moe_sym::Kind;
use rayon::prelude::*;
use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Geometry

const H: usize = 2560;
const I: usize = 640;
const E: usize = 512;
const TOP: usize = 10;
/// `MOE_GROUPED_BLOCK_M` (`hipfire-dispatch/src/families/moe.rs:50`).
const BLOCK_M: usize = 16;
const SUPPORTED_T: [usize; 6] = [1, 2, 3, 4, 7, 8];
const SUPPORTED_ARCH: [&str; 3] = ["gfx1201", "gfx1151", "gfx1100"];
/// Floats of poisoned guard after the residual (inside the arm's output payload).
const GUARD_FLOATS: usize = 256;
/// Finite F32 poison (`0x7A5A5A5A`).
const GUARD_BITS: u32 = 0x7A5A_5A5A;
const DEFAULT_SEED: u64 = 0x5EED_0043;
/// Declared per-element bound factor of the exact-recipe baseline arms (see module doc).
const REL_BOUND: f64 = 16.0 / 256.0;
/// A SwiGLU activation is "ambiguous" when the portable twin's f32 value is within this many f32
/// ulps of its BF16 rounding midpoint (module doc, "IU4 reference").
const AMBIGUITY_ULPS: i64 = 32;
/// Most ambiguous activations of one K128 block whose subsets are enumerated (`2^cap - 1`).
const AMBIGUITY_CAP: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geo {
    h: usize,
    i: usize,
    e: usize,
    top: usize,
}

const FN_GEO: Geo = Geo {
    h: H,
    i: I,
    e: E,
    top: TOP,
};

impl Geo {
    /// QT44 gate/up bytes per output row (`H/256` groups of 136 B).
    fn gu_row_bytes(&self) -> usize {
        self.h / 256 * 136
    }
    /// One expert's gate+up tensor (`2I` rows).
    fn gu_bytes(&self) -> usize {
        2 * self.i * self.gu_row_bytes()
    }
    /// MQ4G128 down bytes per output row (`I/128` groups of 68 B).
    fn dn_row_bytes(&self) -> usize {
        self.i / 128 * 68
    }
    fn dn_bytes(&self) -> usize {
        self.h * self.dn_row_bytes()
    }
    fn weights_bytes(&self) -> usize {
        self.e * (self.gu_bytes() + self.dn_bytes())
    }
    fn slots(&self, t: usize) -> usize {
        t * self.top
    }
    /// Tile-padded grouped rows (`program.rs:391-404`).
    fn grouped_rows(&self, t: usize) -> usize {
        let s = self.slots(t);
        (s + (BLOCK_M - 1) * s.min(self.e)).div_ceil(BLOCK_M) * BLOCK_M
    }
}

// ---------------------------------------------------------------------------
// Deterministic generators (splitmix64, as moe.rs)

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Uniform in `[0, 1)`.
    fn unit01(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f32 {
        let u = |r: &mut Self| ((r.next() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        let (a, b) = (u(self), u(self));
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

fn put_header(group: &mut [u8], off: usize, scale: f32) {
    let sc = f16_bits(scale);
    // Exact: scaling by 8 only moves the exponent of a normal f16.
    let zp = f16_bits(-8.0 * f16_val(sc));
    group[off..off + 2].copy_from_slice(&sc.to_le_bytes());
    group[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
}

fn fill_payload(r: &mut Mix, payload: &mut [u8]) {
    for chunk in payload.chunks_mut(8) {
        let w = r.next().to_le_bytes();
        chunk.copy_from_slice(&w[..chunk.len()]);
    }
}

/// Fill expert `e`'s symmetric weights. QT44 gate/up group (136 B, `kernels/src/
/// qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip:15-35`): `[f16 sc0, zp0, sc1, zp1]` +
/// 128 nibble bytes (weight `w` of the group: byte `w/2`, low nibble for even `w`; weights
/// 0..127 use grid 0, 128..255 grid 1). MQ4G128 down group (68 B, `kernels/src/gemv_mq4g128v2_moe_
/// down_top10_indexed_batched_expanded.hip:44-56`): `[f16 sc, zp]` + 64 nibble bytes (same
/// packing). Every header has `zp == -8*sc` (symmetric) and a finite positive `sc`.
fn gen_expert(geo: Geo, seed: u64, e: usize, gu: &mut [u8], dn: &mut [u8]) {
    debug_assert_eq!(gu.len(), geo.gu_bytes());
    debug_assert_eq!(dn.len(), geo.dn_bytes());
    let key = seed ^ (e as u64 + 1).wrapping_mul(GOLDEN);
    let mut r = Mix(key ^ 0x6A09_E667_F3BC_C908);
    for group in gu.chunks_exact_mut(136) {
        let s0 = 0.002 + 0.004 * r.unit01();
        let s1 = 0.002 + 0.004 * r.unit01();
        put_header(group, 0, s0);
        put_header(group, 4, s1);
        fill_payload(&mut r, &mut group[8..]);
    }
    let mut r = Mix(key ^ 0xBB67_AE85_84CA_A73B);
    for group in dn.chunks_exact_mut(68) {
        let s = 0.006 + 0.006 * r.unit01();
        put_header(group, 0, s);
        fill_payload(&mut r, &mut group[4..]);
    }
}

/// Host-side symmetric-header check (the device check `qwen4_moe_sym_check` is the gate in
/// `setup`; this one backs the unit tests).
fn check_symmetric(gu: &[u8], dn: &[u8]) -> std::result::Result<(), String> {
    let header = |g: &[u8], off: usize| -> std::result::Result<(), String> {
        let sc = f16_val(u16::from_le_bytes([g[off], g[off + 1]]));
        let zp = f16_val(u16::from_le_bytes([g[off + 2], g[off + 3]]));
        if !sc.is_finite() || !zp.is_finite() || zp != -8.0 * sc {
            return Err(format!("asymmetric or non-finite header sc={sc} zp={zp}"));
        }
        Ok(())
    };
    for g in gu.chunks_exact(136) {
        header(g, 0)?;
        header(g, 4)?;
    }
    for g in dn.chunks_exact(68) {
        header(g, 0)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shape

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteKind {
    Uniform,
    Skew,
}

impl RouteKind {
    fn name(self) -> &'static str {
        match self {
            RouteKind::Uniform => "uniform",
            RouteKind::Skew => "skew",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    t: usize,
    mapped: bool,
    route: RouteKind,
    seed: u64,
}

fn parse_shape(s: &str) -> Result<Shape> {
    let mut sh = Shape {
        t: 4,
        mapped: false,
        route: RouteKind::Uniform,
        seed: DEFAULT_SEED,
    };
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut tokens = s.split(',').map(str::trim).filter(|t| !t.is_empty());
    match tokens.next() {
        None | Some("fn") => {}
        Some(other) if !other.contains('=') => {
            return Err(format!(
                "--shape preset `{other}`: qwen4_moe_verify admits only the `fn` (Flash-Next \
                 H=2560/I=640/E=512/TOP=10) preset"
            ))
        }
        Some(first) => {
            return Err(format!(
                "--shape: expected the `fn` preset first (`fn,T=4,res=resident,...`), got `{first}`"
            ))
        }
    }
    for token in tokens {
        let (k, v) = token
            .split_once('=')
            .ok_or_else(|| format!("--shape item `{token}`: expected key=value"))?;
        let (k, v) = (k.trim(), v.trim());
        let canonical = match k {
            "T" | "t" => "T",
            "res" => "res",
            "route" => "route",
            "seed" => "seed",
            other => {
                return Err(format!(
                    "--shape key `{other}`: expected T, res, route, seed"
                ))
            }
        };
        if !seen.insert(canonical) {
            return Err(format!("--shape key `{canonical}` given twice"));
        }
        match canonical {
            "T" => {
                let t: usize = v
                    .parse()
                    .map_err(|_| format!("--shape T={v}: not an integer"))?;
                if !SUPPORTED_T.contains(&t) {
                    return Err(format!(
                        "--shape T={t}: this adapter admits T in {SUPPORTED_T:?} only (T=1 is the \
                         AR decode family, T=2..8 the verify/prefill-family few-row route; other \
                         row counts are outside the traced slice)"
                    ));
                }
                sh.t = t;
            }
            "res" => {
                sh.mapped = match v {
                    "resident" => false,
                    "mapped" => true,
                    other => return Err(format!("--shape res={other}: expected resident|mapped")),
                }
            }
            "route" => {
                sh.route = match v {
                    "uniform" => RouteKind::Uniform,
                    "skew" => RouteKind::Skew,
                    other => return Err(format!("--shape route={other}: expected uniform|skew")),
                }
            }
            _ => {
                sh.seed = v
                    .parse()
                    .map_err(|_| format!("--shape seed={v}: not a u64"))?
            }
        }
    }
    Ok(sh)
}

// ---------------------------------------------------------------------------
// Host inputs

#[derive(Clone, Debug)]
struct HostInputs {
    /// F32 `[T, H]`.
    x_norm: Vec<f32>,
    /// `[T, TOP]`.
    topk_idx: Vec<i32>,
    /// `[T, TOP]`, positive, softmax-normalised per token.
    topk_w: Vec<f32>,
    /// Initial residual F32 `[T, H]`.
    res0: Vec<f32>,
}

fn host_inputs(geo: Geo, t: usize, route: RouteKind, seed: u64) -> Result<HostInputs> {
    if geo.e < geo.top {
        return Err(format!("route needs TOP ({}) <= E ({})", geo.top, geo.e));
    }
    let hot = if route == RouteKind::Skew {
        geo.top.min(5)
    } else {
        0
    };
    let mut rx = Mix(seed ^ 0x1111_2222_3333_4444);
    let x_norm: Vec<f32> = (0..t * geo.h).map(|_| rx.normal()).collect();
    let mut rr = Mix(seed ^ 0x5555_6666_7777_8888);
    let res0: Vec<f32> = (0..t * geo.h).map(|_| 0.25 * rr.normal()).collect();
    let mut rt = Mix(seed ^ 0x0ABC_DEF0_1234_5678);
    let mut topk_idx = Vec::with_capacity(t * geo.top);
    let mut topk_w = Vec::with_capacity(t * geo.top);
    for _ in 0..t {
        let mut row: Vec<i32> = (0..hot as i32).collect();
        while row.len() < geo.top {
            let e = (hot as u64 + rt.below((geo.e - hot) as u64)) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        let logits: Vec<f32> = (0..geo.top).map(|_| rt.normal()).collect();
        let m = logits.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let sum: f32 = ex.iter().sum();
        topk_idx.extend(row);
        topk_w.extend(ex.iter().map(|v| v / sum));
    }
    Ok(HostInputs {
        x_norm,
        topk_idx,
        topk_w,
        res0,
    })
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Residual bytes of one arm: the initial residual then the poisoned guard.
fn residual_payload(res0: &[f32]) -> Vec<u8> {
    let mut b = f32_bytes(res0);
    for _ in 0..GUARD_FLOATS {
        b.extend_from_slice(&GUARD_BITS.to_le_bytes());
    }
    b
}

// ---------------------------------------------------------------------------
// CPU reference (independent of every device kernel)

fn bf16r(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

/// FWHT over `x` (power-of-two length), the device rotation's definition
/// (`kernels/src/gemv_mq4g256.hip:68-139`, `mq_rotate_x_128_v2.hip`; copied from
/// `examples/test_fwht_128_gpu_vs_cpu.rs:19-45` and `test_gemv_mfp4g32.rs:91-111`):
/// `x *= s1`, Sylvester butterfly, `x *= (1/sqrt n) * s2`.
fn fwht_f64(x: &mut [f64], s1: &[f32], s2: &[f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two() && s1.len() >= n && s2.len() >= n);
    for i in 0..n {
        x[i] *= s1[i] as f64;
    }
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 1.0 / (n as f64).sqrt();
    for i in 0..n {
        x[i] *= scale * s2[i] as f64;
    }
}

/// `row` of a QT44 gate/up tensor (`h` columns): `w = sc * (u - 8)`.
fn dequant_gu_row(gu: &[u8], row: usize, h: usize, out: &mut [f64]) {
    let groups = h / 256;
    let base = row * groups * 136;
    for g in 0..groups {
        let gp = &gu[base + g * 136..base + (g + 1) * 136];
        let sc0 = f16_val(u16::from_le_bytes([gp[0], gp[1]])) as f64;
        let sc1 = f16_val(u16::from_le_bytes([gp[4], gp[5]])) as f64;
        let payload = &gp[8..];
        for w in 0..256 {
            let byte = payload[w / 2];
            let u = if w % 2 == 0 { byte & 0xF } else { byte >> 4 } as f64;
            let sc = if w < 128 { sc0 } else { sc1 };
            out[g * 256 + w] = sc * (u - 8.0);
        }
    }
}

/// `row` of an MQ4G128 down tensor (`i` columns).
fn dequant_dn_row(dn: &[u8], row: usize, i: usize, out: &mut [f64]) {
    let groups = i / 128;
    let base = row * groups * 68;
    for g in 0..groups {
        let gp = &dn[base + g * 68..base + (g + 1) * 68];
        let sc = f16_val(u16::from_le_bytes([gp[0], gp[1]])) as f64;
        let payload = &gp[4..];
        for w in 0..128 {
            let byte = payload[w / 2];
            let u = if w % 2 == 0 { byte & 0xF } else { byte >> 4 } as f64;
            out[g * 128 + w] = sc * (u - 8.0);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One expert applied to one rotated token row: unweighted F32-valued down output `[H]`.
fn expert_down(geo: Geo, x_rot: &[f64], gu: &[u8], dn: &[u8], signs: &Signs) -> Vec<f64> {
    let (h, i) = (geo.h, geo.i);
    let mut row = vec![0.0f64; h];
    let mut gate = vec![0.0f32; i];
    let mut up = vec![0.0f32; i];
    for r in 0..2 * i {
        dequant_gu_row(gu, r, h, &mut row);
        let d = dot(&row, x_rot) as f32;
        if r < i {
            gate[r] = d;
        } else {
            up[r - i] = d;
        }
    }
    // Recipe bf16_round_trip: gate, up and the SwiGLU product round to BF16
    // (`moe_gate_up_unscatter_silu_top10.hip`, `moe_unscatter_silu`; decode: `rot128_silu_mul`,
    // `mq_rotate_x_128_v2.hip:80`).
    let mut rot: Vec<f64> = (0..i)
        .map(|k| {
            let v = bf16r(gate[k]);
            let u = bf16r(up[k]);
            let silu = v / (1.0 + (-v).exp());
            bf16r(silu * u) as f64
        })
        .collect();
    for g in rot.chunks_mut(128) {
        fwht_f64(g, &signs.s1_128, &signs.s2_128);
    }
    let mut drow = vec![0.0f64; i];
    (0..h)
        .map(|o| {
            dequant_dn_row(dn, o, i, &mut drow);
            dot(&drow, &rot)
        })
        .collect()
}

struct Signs {
    s1_256: Vec<f32>,
    s2_256: Vec<f32>,
    s1_128: Vec<f32>,
    s2_128: Vec<f32>,
}

impl Signs {
    fn new() -> Self {
        Signs {
            s1_256: gen_fwht_signs(42, 256),
            s2_256: gen_fwht_signs(1042, 256),
            s1_128: gen_fwht_signs(43, 128),
            s2_128: gen_fwht_signs(1043, 128),
        }
    }
}

struct RefData {
    /// Final residual F32 `[T, H]` (BF16-valued).
    out: Vec<f32>,
    /// `S_i = |res0_i| + sum_j |w_j o_ji|`.
    s: Vec<f64>,
}

type ExpertBytes = HashMap<usize, (Vec<u8>, Vec<u8>)>;

fn needed_experts(geo: Geo, seed: u64, topk_idx: &[i32]) -> Result<ExpertBytes> {
    let set: BTreeSet<usize> = topk_idx
        .iter()
        .map(|&e| {
            usize::try_from(e)
                .ok()
                .filter(|&e| e < geo.e)
                .ok_or_else(|| format!("route expert id {e} outside [0, {})", geo.e))
        })
        .collect::<Result<_>>()?;
    Ok(set
        .into_par_iter()
        .map(|e| {
            let mut gu = vec![0u8; geo.gu_bytes()];
            let mut dn = vec![0u8; geo.dn_bytes()];
            gen_expert(geo, seed, e, &mut gu, &mut dn);
            (e, (gu, dn))
        })
        .collect())
}

fn cpu_chain(geo: Geo, t: usize, host: &HostInputs, experts: &ExpertBytes) -> RefData {
    let signs = Signs::new();
    let (h, top) = (geo.h, geo.top);
    // Rotated activations per token.
    let x_rot: Vec<Vec<f64>> = (0..t)
        .map(|tt| {
            let mut x: Vec<f64> = host.x_norm[tt * h..(tt + 1) * h]
                .iter()
                .map(|&v| v as f64)
                .collect();
            for g in x.chunks_mut(256) {
                fwht_f64(g, &signs.s1_256, &signs.s2_256);
            }
            x
        })
        .collect();
    let downs: Vec<Vec<f64>> = (0..t * top)
        .into_par_iter()
        .map(|slot| {
            let e = host.topk_idx[slot] as usize;
            let (gu, dn) = &experts[&e];
            expert_down(geo, &x_rot[slot / top], gu, dn, &signs)
        })
        .collect();
    let mut out = vec![0.0f32; t * h];
    let mut s = vec![0.0f64; t * h];
    for tt in 0..t {
        // The combine visits experts in ascending id; equal ids by ascending rank.
        let mut order: Vec<usize> = (0..top).collect();
        order.sort_by_key(|&j| (host.topk_idx[tt * top + j], j));
        for col in 0..h {
            let r0 = host.res0[tt * h + col];
            let mut acc = bf16r(r0);
            let mut sum = r0.abs() as f64;
            for &j in &order {
                let w = bf16r(host.topk_w[tt * top + j]);
                let o = bf16r(downs[tt * top + j][col] as f32);
                let prod = bf16r(o * w);
                acc = bf16r(acc + prod);
                sum += (o as f64 * w as f64).abs();
            }
            out[tt * h + col] = acc;
            s[tt * h + col] = sum;
        }
    }
    RefData { out, s }
}

// ---------------------------------------------------------------------------
// IU4 candidate reference: `cpu_iu4_chain (exact recipe; SiLU portable twin)`

/// The sym-IU4 chain's gate/up rotation (`mq_rotate_x_i4`, kernels/src/mq_rotate_x_i4.hip:164-217)
/// of one 256-group in f32, op for op: `x * signs1`, Sylvester butterflies of strides 1, 2, 4 (the
/// lane-local steps, `:174-189`) then 8..128 (the `ds_swizzle` wave steps, `:191-207`; the lane
/// with the stride bit set computes `partner - self`), `(v * 0.0625f) * signs2` (`:209-217`).
/// Ascending strides in f32 are the same operation sequence as the device's.
fn rotate256(x: &[f32], s1: &[f32], s2: &[f32]) -> [f32; 256] {
    debug_assert!(x.len() == 256 && s1.len() >= 256 && s2.len() >= 256);
    let mut v = [0f32; 256];
    for i in 0..256 {
        v[i] = x[i] * s1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        for base in (0..256).step_by(2 * stride) {
            for j in base..base + stride {
                let (lo, hi) = (v[j], v[j + stride]);
                v[j] = lo + hi;
                v[j + stride] = lo - hi;
            }
        }
        stride *= 2;
    }
    for i in 0..256 {
        v[i] = (v[i] * 0.0625f32) * s2[i];
    }
    v
}

/// Scale and signed codes of one `block_i4_128` record (`quantize` layout: `d` f32 @0, `s` i32 @4,
/// 64 nibble bytes @8, even index = low nibble, two's complement).
fn decode_record(rec: &[u8; REC]) -> (f32, [i8; GROUP]) {
    let d = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
    let mut q = [0i8; GROUP];
    for (i, c) in q.iter_mut().enumerate() {
        let byte = rec[8 + i / 2];
        let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
        *c = ((nib << 4) as i8) >> 4;
    }
    (d, q)
}

/// The gate/up A4 sidecar of one token (`qwen4_moe_rotate256_i4`): per 256-group the f32 rotation,
/// then the two 128-halves through the C2 `block_i4_128` quantiser
/// (`quantize_block_i4_128_wave<true>`, `kernels/src/block_i4_128_quant.hip:85-215`; on gfx1201
/// the one-pass `emit_iu4_sidecar_c2_gfx12`, `:290-382`, documented bit-identical).
fn gate_xrow(x: &[f32], signs: &Signs) -> XRow {
    let mut d = Vec::with_capacity(x.len() / GROUP);
    let mut v = Vec::with_capacity(x.len());
    for g in x.chunks_exact(256) {
        let r = rotate256(g, &signs.s1_256, &signs.s2_256);
        for half in 0..2 {
            let mut blk = [0f32; GROUP];
            blk.copy_from_slice(&r[half * GROUP..(half + 1) * GROUP]);
            let (dd, q) = decode_record(&quantize(&blk));
            d.push(dd);
            v.extend_from_slice(&q);
        }
    }
    XRow { d, v }
}

/// One K128 block of the down A4 sidecar from 128 BF16 SwiGLU activations
/// (`qwen4_moe_rotate128_i4`, `kernels/src/qwen4_moe_rotate128_i4.hip:32-67`; PM twins
/// `moe_rotate_i4_gp_pm_gfx1201` / `mq_i4_a4_r128_k640_gfx1100`, all bit-identical): widen,
/// `* signs1`, FWHT-128, `(v * 0.0883883476f) * signs2`, C2 quantise.
fn down_block(act: &[u16], signs: &Signs) -> (f32, [i8; GROUP]) {
    let mut h = [0u16; GROUP];
    h.copy_from_slice(&act[..GROUP]);
    let mut s1 = [0f32; GROUP];
    let mut s2 = [0f32; GROUP];
    s1.copy_from_slice(&signs.s1_128[..GROUP]);
    s2.copy_from_slice(&signs.s2_128[..GROUP]);
    decode_record(&quantize(&rotate(&h, &s1, &s2)))
}

/// BF16 ulp of `x` (`2^(e - 7)`, `e` the unbiased exponent; subnormals at `e = -126`).
fn bf16_ulp_f32(x: f32) -> f64 {
    let e = (((x.to_bits() >> 23) & 0xff) as i32).max(1);
    2f64.powi(e - 127 - 7)
}

/// BF16 SwiGLU activations of one slot from the exact gate/up fold sums `[2I]`, plus the elements
/// whose BF16 rounding the portable SiLU twin cannot decide: the twin's f32 value lies within
/// [`AMBIGUITY_ULPS`] f32 ulps of the BF16 rounding midpoint, so the device's `expf` / `IEEE
/// divide` result may round the other way. `ambiguous` lists `(index, other-rounding bits)`.
struct Act {
    bits: Vec<u16>,
    ambiguous: Vec<(usize, u16)>,
}

fn activation(sums: &[f32], i: usize) -> Act {
    let mut bits = Vec::with_capacity(i);
    let mut ambiguous = Vec::new();
    for k in 0..i {
        let p = portable_silu_mul(rt_f32(sums[k]), rt_f32(sums[i + k]));
        let b = bf16_bits(p);
        bits.push(b);
        if p.is_finite() {
            let lo = p.to_bits() & 0xFFFF_0000;
            let mid = i64::from(lo | 0x8000);
            if (i64::from(p.to_bits()) - mid).abs() <= AMBIGUITY_ULPS {
                let lo_bits = (lo >> 16) as u16;
                ambiguous.push((k, if b == lo_bits { lo_bits + 1 } else { lo_bits }));
            }
        }
    }
    Act { bits, ambiguous }
}

/// Down GEMM of one slot (`gemm_qwen4_moe_down_iu4_sym`): exact fold, BF16 RNE store.
fn down_out(geo: Geo, x: &XRow, dn: &[u8]) -> Vec<f32> {
    fold_slots(
        Kind::Down,
        geo.h,
        geo.i / GROUP,
        geo.dn_row_bytes(),
        dn,
        std::slice::from_ref(x),
    )
    .into_iter()
    .map(bf16r)
    .collect()
}

/// One routed slot through the IU4 chain: nominal BF16 down row `o` and, per K128 activation block
/// holding ambiguous elements, every other down row the device could produce if it rounded any
/// non-empty subset of that block's ambiguous activations the other way.
struct SlotRef {
    o: Vec<f32>,
    alts: Vec<Vec<Vec<f32>>>,
    ambiguous_elements: usize,
}

fn slot_chain(geo: Geo, xrow: &XRow, gu: &[u8], dn: &[u8], signs: &Signs) -> Result<SlotRef> {
    let sums = fold_slots(
        Kind::GateUp,
        2 * geo.i,
        geo.h / GROUP,
        geo.gu_row_bytes(),
        gu,
        std::slice::from_ref(xrow),
    );
    let act = activation(&sums, geo.i);
    let blocks = geo.i / GROUP;
    let mut d = Vec::with_capacity(blocks);
    let mut v = Vec::with_capacity(geo.i);
    for b in 0..blocks {
        let (dd, q) = down_block(&act.bits[b * GROUP..], signs);
        d.push(dd);
        v.extend_from_slice(&q);
    }
    let nominal = XRow { d, v };
    let o = down_out(geo, &nominal, dn);
    let mut alts = Vec::new();
    for b in 0..blocks {
        let amb: Vec<(usize, u16)> = act
            .ambiguous
            .iter()
            .filter(|(k, _)| k / GROUP == b)
            .map(|&(k, alt)| (k % GROUP, alt))
            .collect();
        if amb.is_empty() {
            continue;
        }
        if amb.len() > AMBIGUITY_CAP {
            return Err(format!(
                "cpu_iu4_chain: {} ambiguous SiLU roundings in one K128 block (cap {AMBIGUITY_CAP}); \
                 the admissible set is not enumerated",
                amb.len()
            ));
        }
        let mut block_alts = Vec::new();
        for mask in 1u32..(1 << amb.len()) {
            let mut bits = act.bits[b * GROUP..(b + 1) * GROUP].to_vec();
            for (n, &(k, alt)) in amb.iter().enumerate() {
                if mask >> n & 1 == 1 {
                    bits[k] = alt;
                }
            }
            let (dd, q) = down_block(&bits, signs);
            let mut x = XRow {
                d: nominal.d.clone(),
                v: nominal.v.clone(),
            };
            x.d[b] = dd;
            x.v[b * GROUP..(b + 1) * GROUP].copy_from_slice(&q);
            block_alts.push(down_out(geo, &x, dn));
        }
        alts.push(block_alts);
    }
    Ok(SlotRef {
        o,
        alts,
        ambiguous_elements: act.ambiguous.len(),
    })
}

/// `moe_down_combine_grouped_top10_bf16in` of one token (`moe_down_combine_grouped_top10.hip:
/// 97-193`): ascending expert id (ties by rank), BF16 accumulator starting from `bf16(residual)`,
/// BF16 route weight, BF16 product, BF16 running sum.
fn combine_row(res0: &[f32], idx: &[i32], w: &[f32], o: &[&[f32]]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..idx.len()).collect();
    order.sort_by_key(|&j| (idx[j], j));
    (0..res0.len())
        .map(|col| {
            let mut acc = bf16r(res0[col]);
            for &j in &order {
                let prod = bf16r(bf16r(o[j][col]) * bf16r(w[j]));
                acc = bf16r(acc + prod);
            }
            acc
        })
        .collect()
}

#[derive(Debug)]
struct Iu4Ref {
    /// Final residual F32 `[T, H]` (BF16-valued) of the nominal recipe.
    out: Vec<f32>,
    /// Per-element tolerance (see the module doc): 0 where no SiLU rounding is ambiguous.
    tol: Vec<f64>,
    ambiguous_elements: usize,
    /// (slot, K128 block) units with at least one ambiguous activation.
    ambiguous_blocks: usize,
}

/// `cpu_iu4_chain`: the sym-IU4 chain's arithmetic stage by stage (module doc, "IU4 reference").
fn cpu_iu4_chain(geo: Geo, t: usize, host: &HostInputs, experts: &ExpertBytes) -> Result<Iu4Ref> {
    let signs = Signs::new();
    let (h, top) = (geo.h, geo.top);
    let xrows: Vec<XRow> = (0..t)
        .map(|tt| gate_xrow(&host.x_norm[tt * h..(tt + 1) * h], &signs))
        .collect();
    let slots: Vec<SlotRef> = (0..t * top)
        .into_par_iter()
        .map(|s| {
            let e = host.topk_idx[s] as usize;
            let (gu, dn) = &experts[&e];
            slot_chain(geo, &xrows[s / top], gu, dn, &signs)
        })
        .collect::<Result<_>>()?;
    let mut out = vec![0.0f32; t * h];
    let mut tol = vec![0.0f64; t * h];
    let (mut ambiguous_elements, mut ambiguous_blocks) = (0usize, 0usize);
    for tt in 0..t {
        let idx = &host.topk_idx[tt * top..(tt + 1) * top];
        let w = &host.topk_w[tt * top..(tt + 1) * top];
        let res0 = &host.res0[tt * h..(tt + 1) * h];
        let nominal: Vec<&[f32]> = (0..top).map(|j| &slots[tt * top + j].o[..]).collect();
        let base = combine_row(res0, idx, w, &nominal);
        let mut units = 0usize;
        let mut t_tol = vec![0.0f64; h];
        for j in 0..top {
            let slot = &slots[tt * top + j];
            ambiguous_elements += slot.ambiguous_elements;
            for block_alts in &slot.alts {
                units += 1;
                let mut worst = vec![0.0f64; h];
                for alt in block_alts {
                    let mut os = nominal.clone();
                    os[j] = alt;
                    let a = combine_row(res0, idx, w, &os);
                    for c in 0..h {
                        worst[c] = worst[c].max((a[c] as f64 - base[c] as f64).abs());
                    }
                }
                for c in 0..h {
                    t_tol[c] += worst[c];
                }
            }
        }
        // First-order superposition of the units; two or more units can also move a later BF16
        // rounding of the running sum by one ulp that no single unit shows.
        for c in 0..h {
            if units >= 2 {
                t_tol[c] += bf16_ulp_f32(base[c]);
            }
            out[tt * h + c] = base[c];
            tol[tt * h + c] = t_tol[c];
        }
        ambiguous_blocks += units;
    }
    Ok(Iu4Ref {
        out,
        tol,
        ambiguous_elements,
        ambiguous_blocks,
    })
}

fn parse_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "output payload of {} bytes is not f32",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Per-element bound of the exact-recipe baseline arms: `REL_BOUND * S_i`.
fn baseline_bounds(rd: &RefData) -> Vec<f64> {
    rd.s.iter().map(|s| REL_BOUND * s).collect()
}

/// Compare the downloaded payload (`reference.len()` residual floats then the poisoned guard) to
/// `reference` with per-element `bound` (`err <= bound`; `1e-30` keeps a zero bound an exact check).
fn compare(name: &str, reference: &[f32], bound: &[f64], payload: &[f32]) -> Result<RefReport> {
    let n = reference.len();
    debug_assert_eq!(bound.len(), n);
    if payload.len() != n + GUARD_FLOATS {
        return Err(format!(
            "output payload holds {} floats, expected {} ({} + {} guard)",
            payload.len(),
            n + GUARD_FLOATS,
            n,
            GUARD_FLOATS
        ));
    }
    let guard_ok = payload[n..].iter().all(|v| v.to_bits() == GUARD_BITS);
    let (mut worst_ratio, mut worst_err, mut worst_bound) = (0.0f64, 0.0f64, f64::INFINITY);
    let mut broken = !guard_ok;
    for i in 0..n {
        let d = payload[i];
        if !d.is_finite() {
            broken = true;
            break;
        }
        let err = (d as f64 - reference[i] as f64).abs();
        let b = bound[i] + 1e-30;
        let ratio = err / b;
        if ratio >= worst_ratio {
            worst_ratio = ratio;
            worst_err = err;
            worst_bound = b;
        }
    }
    if broken {
        return Ok(RefReport {
            reference: name.to_string(),
            max_abs_err: f64::INFINITY,
            tolerance: if worst_bound.is_finite() {
                worst_bound
            } else {
                0.0
            },
            pass: false,
        });
    }
    Ok(RefReport {
        reference: name.to_string(),
        max_abs_err: worst_err,
        tolerance: worst_bound,
        pass: worst_ratio <= 1.0,
    })
}

// ---------------------------------------------------------------------------
// Device state

/// Inputs shared by both arms (see the frozen contract in the slice brief).
pub struct ChainInputs {
    pub arch: &'static str,
    pub t: usize,
    pub h: usize,
    pub i: usize,
    pub e: usize,
    pub top: usize,
    pub host_mapped: bool,
    /// F32 `[T, H]`: the routed block input before the input-basis rotation.
    pub x_norm: GpuTensor,
    /// i32 `[T, TOP]` (a `DType::Raw` byte tensor shaped `[T, TOP, 4]`).
    pub topk_idx: GpuTensor,
    /// F32 `[T, TOP]`.
    pub topk_w: GpuTensor,
    /// Device table of `E` u64 pointers to QT44 MQ4G256V2 `[2I x H]` expert tensors.
    pub gate_up_ptrs: GpuTensor,
    /// Device table of `E` u64 pointers to MQ4G128V2 `[H x I]` expert tensors.
    pub down_ptrs: GpuTensor,
}

struct Grouped {
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tile_ids: GpuTensor,
    inverse: GpuTensor,
    y_gu: GpuTensor,
    rot: GpuTensor,
}

struct Decode {
    gate: GpuTensor,
    up: GpuTensor,
    rot: GpuTensor,
}

/// One arm's own scratch, residual and output.
struct ArmBufs {
    x_rot: GpuTensor,
    grouped: Option<Grouped>,
    decode: Option<Decode>,
    expanded: GpuTensor,
    /// `[T*H + GUARD_FLOATS]`: residual then poisoned guard.
    res_full: GpuTensor,
    /// `[T, H]` view at the head of `res_full`.
    residual: GpuTensor,
}

fn zeros_f32(gpu: &mut Gpu, n: usize) -> Result<GpuTensor> {
    gpu.zeros(&[n.max(1)], DType::F32).map_err(dev_err)
}

fn zeros_raw(gpu: &mut Gpu, bytes: usize) -> Result<GpuTensor> {
    gpu.zeros(&[bytes.max(4)], DType::Raw).map_err(dev_err)
}

fn arm_bufs(gpu: &mut Gpu, geo: Geo, t: usize) -> Result<ArmBufs> {
    let s = geo.slots(t);
    let (grouped, decode) = if t >= 2 {
        let g = geo.grouped_rows(t);
        (
            Some(Grouped {
                counts: zeros_raw(gpu, geo.e * 4)?,
                offsets: zeros_raw(gpu, (geo.e + 1) * 4)?,
                sorted: zeros_raw(gpu, g * 4)?,
                tile_ids: zeros_raw(gpu, g / BLOCK_M * 4)?,
                inverse: zeros_raw(gpu, s * 4)?,
                y_gu: zeros_f32(gpu, g * 2 * geo.i)?,
                rot: zeros_f32(gpu, s * geo.i)?,
            }),
            None,
        )
    } else {
        (
            None,
            Some(Decode {
                gate: zeros_f32(gpu, s * geo.i)?,
                up: zeros_f32(gpu, s * geo.i)?,
                rot: zeros_f32(gpu, s * geo.i)?,
            }),
        )
    };
    let res_full = zeros_f32(gpu, t * geo.h + GUARD_FLOATS)?;
    let mut residual = res_full.sub_offset(0, t * geo.h);
    residual.shape = vec![t, geo.h];
    Ok(ArmBufs {
        x_rot: zeros_f32(gpu, t * geo.h)?,
        grouped,
        decode,
        expanded: zeros_f32(gpu, s * geo.h)?,
        res_full,
        residual,
    })
}

/// Bytes of the baseline arm's own scratch + residual (analytic; GPU-free).
fn arm_scratch_bytes(geo: Geo, t: usize) -> usize {
    let s = geo.slots(t);
    let f = |n: usize| n * 4;
    let common = f(t * geo.h) + f(s * geo.h) + f(t * geo.h + GUARD_FLOATS);
    if t >= 2 {
        let g = geo.grouped_rows(t);
        common
            + f(geo.e)
            + f(geo.e + 1)
            + f(g)
            + f(g / BLOCK_M)
            + f(s)
            + f(g * 2 * geo.i)
            + f(s * geo.i)
    } else {
        common + 3 * f(s * geo.i)
    }
}

fn hip<T, E2: std::fmt::Display>(r: std::result::Result<T, E2>) -> Result<T> {
    r.map_err(dev_err)
}

/// The shipped chain at (arch, T) through the public launchers (module doc, exact literals).
fn encode_shipped(
    gpu: &mut Gpu,
    arch: &str,
    t: usize,
    inp: &ChainInputs,
    b: &ArmBufs,
) -> Result<()> {
    let (h, i, e, top) = (inp.h, inp.i, inp.e, inp.top);
    let s = t * top;
    if let Some(d) = &b.decode {
        // T = 1: AR decode family (`select_decode`).
        hip(gpu.rotate_x_mq(&inp.x_norm, &b.x_rot, h))?;
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            &inp.gate_up_ptrs,
            &inp.topk_idx,
            &b.x_rot,
            &d.gate,
            &d.up,
            2 * i,
            h,
            1,
        ))?;
        hip(gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(&d.gate, &d.up, &d.rot, i, top, None))?;
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &inp.down_ptrs,
            &inp.topk_idx,
            &d.rot,
            &b.expanded,
            h,
            i,
            1,
            e,
            top,
        ))?;
    } else {
        let g = b
            .grouped
            .as_ref()
            .ok_or("arm buffers carry neither the grouped nor the decode scratch")?;
        let grouped_rows = inp.grouped_rows();
        hip(gpu.rotate_x_mq_batched(&inp.x_norm, &b.x_rot, h, t))?;
        hip(gpu.moe_scatter_fused_top10(
            &inp.topk_idx,
            &g.counts,
            &g.offsets,
            &g.sorted,
            &g.tile_ids,
            &g.inverse,
            s,
            e,
            grouped_rows,
            BLOCK_M,
        ))?;
        if arch == "gfx1151" {
            hip(gpu.gemm_mq4g256v2_moe_grouped_top10_inverse(
                &inp.gate_up_ptrs,
                &g.tile_ids,
                &g.sorted,
                &g.inverse,
                &b.x_rot,
                &g.y_gu,
                2 * i,
                h,
                top,
                grouped_rows,
                t,
            ))?;
        } else {
            hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
                &inp.gate_up_ptrs,
                &g.tile_ids,
                &g.sorted,
                &b.x_rot,
                &g.y_gu,
                2 * i,
                h,
                top,
                grouped_rows,
                t,
            ))?;
        }
        hip(gpu.moe_gate_up_unscatter_silu_rotate128_top10(
            &g.y_gu,
            &g.sorted,
            &g.rot,
            i,
            grouped_rows,
            true,
            None,
        ))?;
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            &inp.down_ptrs,
            &inp.topk_idx,
            &g.rot,
            &b.expanded,
            h,
            i,
            t,
            e,
            top,
        ))?;
    }
    hip(gpu.moe_down_combine_top10_batched(
        &b.expanded,
        &inp.topk_idx,
        &inp.topk_w,
        &b.residual,
        h,
        t,
        e,
    ))
}

impl ChainInputs {
    fn grouped_rows(&self) -> usize {
        Geo {
            h: self.h,
            i: self.i,
            e: self.e,
            top: self.top,
        }
        .grouped_rows(self.t)
    }
}

struct Dev {
    inputs: ChainInputs,
    base: ArmBufs,
    cand: ArmBufs,
    /// Expert arenas (never freed; `GpuTensor` has no freeing `Drop`).
    _weights: Vec<GpuTensor>,
}

// ---------------------------------------------------------------------------
// The workload

/// Driver constructor (`pm_loop.rs`): GPU-free.
pub fn new(shape: &str, knobs: &mut Knobs, ctx: ChainCtx) -> Result<Box<dyn ChainWorkload>> {
    let shape = parse_shape(shape)?;
    if !SUPPORTED_ARCH.contains(&ctx.arch) {
        return Err(format!(
            "qwen4_moe_verify: arch {} is not traced (gfx1201, gfx1151, gfx1100 only)",
            ctx.arch
        ));
    }
    let candidate = knobs.take("candidate");
    let iu4 = match (ctx.null_control, candidate.as_deref()) {
        (true, Some(c)) => {
            return Err(format!(
                "--candidate {c} is refused under --null-control aa: the candidate arm is the \
                 baseline chain (A/A null control)"
            ))
        }
        (true, None) => None,
        (false, None) => {
            return Err(
                "qwen4_moe_verify needs --candidate iu4_sym (or --null-control aa)".to_string(),
            )
        }
        (false, Some("iu4_sym")) => Some(Iu4Arm::new(ctx.arch)?),
        (false, Some(other)) => {
            return Err(format!(
                "--candidate {other}: expected iu4_sym (the PM IU4 symmetric grouped chain forced \
                 at this T)"
            ))
        }
    };
    let host = host_inputs(FN_GEO, shape.t, shape.route, shape.seed)?;
    Ok(Box::new(ChainBase {
        shape,
        ctx,
        geo: FN_GEO,
        host,
        iu4,
        dev: None,
        ref_cache: OnceLock::new(),
        iu4_ref_cache: OnceLock::new(),
        experts_cache: OnceLock::new(),
        swapped: false,
    }))
}

pub struct ChainBase {
    shape: Shape,
    ctx: ChainCtx,
    geo: Geo,
    host: HostInputs,
    iu4: Option<Iu4Arm>,
    dev: Option<Dev>,
    ref_cache: OnceLock<RefData>,
    iu4_ref_cache: OnceLock<Iu4Ref>,
    experts_cache: OnceLock<ExpertBytes>,
    /// Crossover binding (`ChainWorkload::set_buffer_swap`): `false` = baseline on buffer set
    /// `base`, candidate on `cand`; `true` = the other way round.
    swapped: bool,
}

impl ChainBase {
    fn aa(&self) -> bool {
        self.ctx.null_control
    }

    fn residency(&self) -> &'static str {
        if self.shape.mapped {
            "host_mapped"
        } else {
            "resident"
        }
    }

    /// Re-evaluate the public selector predicates on the live device (module doc, "Live check").
    fn check_live_selector(&self, gpu: &Gpu) -> Result<()> {
        let t = self.shape.t;
        if gpu.arch != self.ctx.arch {
            return Err(format!(
                "device is {} but the chain was selected for {}",
                gpu.arch, self.ctx.arch
            ));
        }
        if !gpu.arch_caps.has_wmma() || !gpu.flags.moe_grouped_gemm {
            return Err(
                "qwen4_moe_verify: use_path2 is false on this device/env (has_wmma / \
                 HIPFIRE_MOE_GROUPED_GEMM); the shipped route differs from the traced one \
                 (families/moe.rs:2611-2613)"
                    .into(),
            );
        }
        if t < 2 {
            return Ok(());
        }
        let refuse = |what: &str| -> Result<()> {
            Err(format!(
                "qwen4_moe_verify T={t}: live selector disagrees with the traced route: {what}"
            ))
        };
        if gpu.route_rows(t) != t {
            return refuse("prefill-invariant scope is active (route_rows != rows)");
        }
        if t > gpu.few_row_limit() {
            return refuse("few_row_limit < T (indexed_down would not be taken)");
        }
        if gpu.qwen4_moe_sym_iu4_applies(t) {
            return refuse("the symmetric IU4 route applies at this T (sym_iu4 should be off)");
        }
        if gpu.qwen4_moe_gateup_wmma_applies(2 * I, H, t) {
            return refuse("the F16 WMMA gate/up arm applies (gateup_bf16 should be off)");
        }
        if gpu.qwen4_moe_down_wmma_applies(H, I, t * TOP) {
            return refuse("the F16 WMMA down arm applies (down_wmma should be off)");
        }
        Ok(())
    }

    fn experts(&self) -> Result<&ExpertBytes> {
        if let Some(e) = self.experts_cache.get() {
            return Ok(e);
        }
        let e = needed_experts(self.geo, self.shape.seed, &self.host.topk_idx)?;
        let _ = self.experts_cache.set(e);
        self.experts_cache
            .get()
            .ok_or_else(|| "expert cache not set".to_string())
    }

    fn ref_data(&self) -> Result<&RefData> {
        if let Some(r) = self.ref_cache.get() {
            return Ok(r);
        }
        let r = cpu_chain(self.geo, self.shape.t, &self.host, self.experts()?);
        let _ = self.ref_cache.set(r);
        self.ref_cache
            .get()
            .ok_or_else(|| "reference cache not set".to_string())
    }

    fn iu4_ref(&self) -> Result<&Iu4Ref> {
        if let Some(r) = self.iu4_ref_cache.get() {
            return Ok(r);
        }
        let r = cpu_iu4_chain(self.geo, self.shape.t, &self.host, self.experts()?)?;
        let _ = self.iu4_ref_cache.set(r);
        self.iu4_ref_cache
            .get()
            .ok_or_else(|| "iu4 reference cache not set".to_string())
    }

    /// The buffer set `arm` is bound to under the current crossover binding.
    fn arm_bufs(d: &Dev, arm: Arm, swapped: bool) -> &ArmBufs {
        match (arm, swapped) {
            (Arm::Baseline, false) | (Arm::Candidate, true) => &d.base,
            (Arm::Candidate, false) | (Arm::Baseline, true) => &d.cand,
        }
    }

    fn dev(&self) -> Result<&Dev> {
        self.dev
            .as_ref()
            .ok_or_else(|| "qwen4_moe_verify: setup has not run".to_string())
    }

    fn gpu_reset(gpu: &mut Gpu, bufs: &ArmBufs, payload: &[u8]) -> Result<()> {
        hip(gpu.hip.device_synchronize())?;
        hip(gpu.hip.memcpy_htod(&bufs.res_full.buf, payload))?;
        hip(gpu.hip.device_synchronize())
    }

    // ---- chain_json ------------------------------------------------------

    fn pred(name: &str, value: Value, source: &str) -> Value {
        json!({ "predicate": name, "value": value, "source": source })
    }

    fn selector_json(&self) -> Vec<Value> {
        let t = self.shape.t;
        let arch = self.ctx.arch;
        let p = Self::pred;
        let mut v = vec![
            p(
                "recipe = SoftmaxGatedShared{bf16_round_trip:true, shared_after_combine:true}",
                json!(true),
                "crates/hipfire-arch-qwen4/src/program.rs:1110-1113 (decode), 1230-1233 (grouped); \
                 crates/hipfire-dispatch/src/families/moe.rs:241-259",
            ),
            p(
                "route_capability(rows) = Qt44Qt53Grouped (symmetric needs rows >= 512; slab/fp8 not modelled)",
                json!("Qt44Qt53Grouped"),
                "crates/hipfire-arch-qwen4/src/gpu_forward.rs:3199-3209; crates/rdna-compute/src/gemm.rs:49222",
            ),
        ];
        if t >= 2 {
            v.extend([
                p(
                    "use_path2 = moe_grouped_gemm && arch.has_wmma() && (!narrow_verify || offloaded_uniform_verify)",
                    json!(true),
                    "crates/hipfire-dispatch/src/families/moe.rs:2611-2613; feature_flags.rs:1159-1162; arch_caps.rs:145,457",
                ),
                p(
                    "narrow_verify = is_verify && has_host_projection && batch<=4 (is_verify = ctx.workload==SpeculativeVerify; qwen4 FN seals Standard)",
                    json!(false),
                    "crates/hipfire-dispatch/src/families/moe.rs:2596-2597; pipeline/moe_program.rs:4115-4122,4171-4175",
                ),
                p(
                    "grouped_path: !wide_exact_route(route_rows, few_row_limit) (route_rows<=8)",
                    json!(true),
                    "crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:137-143; layer_ops.rs:238-243",
                ),
                p(
                    "few_row_limit() = FEW_ROW_ROUTE_ROWS = 8 (eager, wide_verify_rows = 0 outside a lab scope); route_rows(T) = T",
                    json!(8),
                    "crates/rdna-compute/src/prefill_invariant.rs:232-241,293-317; dispatch.rs:3025; gpu_forward.rs:332-364",
                ),
                p(
                    "sym_iu4 (capability symmetric && route_rows >= 512)",
                    json!(false),
                    "qt44_qt53_prefill.rs:1573-1585; gemm.rs:50245-50246,50604-50607",
                ),
                p(
                    "gateup_bf16 (qwen4_moe_gateup_wmma_applies: route_rows >= 512)",
                    json!(false),
                    "qt44_qt53_prefill.rs:1648-1654; gemm.rs:48639-48645,570",
                ),
                p(
                    "down_wmma (qwen4_moe_down_wmma_applies: route_rows*10 >= 5120)",
                    json!(false),
                    "qt44_qt53_prefill.rs:1658-1663; moe.rs:3380-3386",
                ),
                p(
                    "unscatter_rotates (!sym_iu4 && !gateup_bf16 && !down_wmma && mi % 128 == 0)",
                    json!(true),
                    "qt44_qt53_prefill.rs:2032-2039 (taken at 1984-2018)",
                ),
                p(
                    "indexed_down = use_path2 && route_rows <= few_row_limit && !down_wmma",
                    json!(true),
                    "qt44_qt53_prefill.rs:2128-2130 (down recursion 2207-2209, combine recursion 2344-2346)",
                ),
                p(
                    "gateup_rotates_f16 (input basis stays F32 mq_rotate_x)",
                    json!(false),
                    "qt44_qt53_prefill.rs:456-469,533-541",
                ),
                p(
                    "gate_up arm: gfx1151 && mi==640 && k==2560 && k_top==10 && 1<=T<=8 -> _inverse, else grouped_top10",
                    json!(if arch == "gfx1151" { "inverse" } else { "grouped_top10" }),
                    "qt44_qt53_prefill.rs:1873-1905; rdna-compute/src/moe_few.rs:26-50; gemm.rs:48750-48838",
                ),
                p(
                    "combine: indexed arm, in-place +=, no post round trip (recipe.bf16_round_trip && !shared_after_combine is false)",
                    json!("moe_down_combine_top10_batched"),
                    "qt44_qt53_prefill.rs:2329-2394; kernels/src/moe_down_combine_top10_batched.hip:26,51-76",
                ),
                p(
                    "residency does not change any call or argument (host-mapped only feeds sym_iu4 / slab, not taken at T<=8)",
                    json!(self.residency()),
                    "qt44_qt53_prefill.rs:1782-1784,1842,2203,2446; families/moe.rs:2596-2613",
                ),
            ]);
        } else {
            v.extend([
                p(
                    "select_decode: SoftmaxGatedShared && batch_size==1 && ep None && no Paro && !defer_routed_combine -> select_grouped_route",
                    json!(true),
                    "crates/hipfire-dispatch/src/pipeline/moe_program.rs:3048-3105; mod.rs:322-423; program.rs:1107-1128",
                ),
                p(
                    "route.is_qt44_qt53_grouped: decode stages take the fixed top-10 indexed launchers on every arch",
                    json!(true),
                    "pipeline/mod.rs:5763-5810 (gate_up), 6060-6100 (activation), 6208-6263 (down), 6846-6870 (combine)",
                ),
                p(
                    "decode input basis: rotate_x_mq(x_norm, x_rot_local, hidden) (x_rot_prerotated=false, normalization=Provided)",
                    json!(true),
                    "pipeline/mod.rs:6566-6604; program.rs:1115,1123",
                ),
                p(
                    "residency does not change any decode call or argument",
                    json!(self.residency()),
                    "pipeline/mod.rs:5763-5810,6208-6263",
                ),
            ]);
        }
        v
    }

    fn stages_json(&self) -> Vec<Value> {
        let t = self.shape.t;
        let arch = self.ctx.arch;
        let st = |stage: &str, launcher: &str, symbols: Vec<&str>, source: &str| {
            json!({
                "stage": stage,
                "launcher": launcher,
                "symbols": symbols,
                "origin": "hip",
                "source": source,
            })
        };
        let down = if arch == "gfx1151" {
            vec![
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4",
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
            ]
        } else {
            vec!["gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"]
        };
        let down_stage = st(
            "down",
            &format!(
                "Gpu::gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(down_ptrs, topk_idx, rot, expanded, 2560, 640, {t}, 512, 10)"
            ),
            down,
            "crates/rdna-compute/src/moe.rs:3067-3115; qt44_qt53_prefill.rs:2274-2284 (T>=2), pipeline/mod.rs:6247-6257 (T=1)",
        );
        let combine = st(
            "combine",
            &format!(
                "Gpu::moe_down_combine_top10_batched(expanded, topk_idx, topk_w, residual, 2560, {t}, 512)"
            ),
            vec!["moe_down_combine_top10_batched"],
            "crates/rdna-compute/src/moe.rs:2216; qt44_qt53_prefill.rs:2376-2387 (T>=2), pipeline/mod.rs:6858-6865 (T=1)",
        );
        if t == 1 {
            vec![
                st(
                    "input_basis",
                    "Gpu::rotate_x_mq(x_norm, x_rot, 2560)",
                    vec!["mq_rotate_x"],
                    "crates/rdna-compute/src/gemv.rs:6683; pipeline/mod.rs:6603",
                ),
                st(
                    "gate_up",
                    "Gpu::gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(gate_up_ptrs, topk_idx, x_rot, gate, up, 1280, 2560, 1)",
                    vec![
                        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
                        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
                    ],
                    "crates/rdna-compute/src/gemv.rs:18128,18262-18297; pipeline/mod.rs:5798-5807",
                ),
                st(
                    "activation",
                    "Gpu::silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(gate, up, rot, 640, 10, None)",
                    vec!["mq_rotate_x_128_v2_silu_bf16"],
                    "crates/rdna-compute/src/gemv.rs:8360; pipeline/mod.rs:6088-6095",
                ),
                down_stage,
                combine,
            ]
        } else {
            let gate_up = if arch == "gfx1151" {
                st(
                    "gate_up",
                    &format!(
                        "Gpu::gemm_mq4g256v2_moe_grouped_top10_inverse(gate_up_ptrs, tile_ids, sorted, inverse_perm, x_rot, y_gu, 1280, 2560, 10, G, {t})"
                    ),
                    vec!["gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4_inverse_k2560"],
                    "qt44_qt53_prefill.rs:1873-1893; crates/rdna-compute/src/moe_few.rs:12-13,26",
                )
            } else {
                st(
                    "gate_up",
                    &format!(
                        "Gpu::gemm_mq4g256v2_moe_grouped_top10(gate_up_ptrs, tile_ids, sorted, x_rot, y_gu, 1280, 2560, 10, G, {t})"
                    ),
                    vec!["gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4"],
                    "qt44_qt53_prefill.rs:1894-1905; crates/rdna-compute/src/gemm.rs:48750,48794-48826",
                )
            };
            vec![
                st(
                    "input_basis",
                    &format!("Gpu::rotate_x_mq_batched(x_norm, x_rot, 2560, {t})"),
                    vec!["mq_rotate_x"],
                    "qt44_qt53_prefill.rs:456-469; crates/rdna-compute/src/gemv.rs:6798",
                ),
                st(
                    "scatter",
                    &format!(
                        "Gpu::moe_scatter_fused_top10(topk_idx, counts, offsets, sorted, tile_ids, inverse_perm, {}, 512, G, 16)",
                        t * TOP
                    ),
                    vec!["moe_scatter_fused_top10"],
                    "qt44_qt53_prefill.rs:1631-1642; crates/rdna-compute/src/moe.rs:2502",
                ),
                gate_up,
                st(
                    "unscatter_activation",
                    "Gpu::moe_gate_up_unscatter_silu_rotate128_top10(y_gu, sorted, rot, 640, G, true, None)",
                    vec!["moe_gate_up_unscatter_silu_rotate128_top10"],
                    "qt44_qt53_prefill.rs:1984-2018; crates/rdna-compute/src/moe.rs:2738 (the activation stage is then a no-op: qt44:2072-2087)",
                ),
                down_stage,
                combine,
            ]
        }
    }

    fn baseline_json(&self) -> Value {
        json!({
            "selector": self.selector_json(),
            "stages": self.stages_json(),
            "residency": self.residency(),
            "scratch_bytes": arm_scratch_bytes(self.geo, self.shape.t),
            "weights_bytes": self.geo.weights_bytes(),
            "grouped_rows": if self.shape.t >= 2 { json!(self.geo.grouped_rows(self.shape.t)) } else { Value::Null },
            "shipped_recipe": RECIPE,
        })
    }
}

const RECIPE: &str = "FN qt44 recipe SoftmaxGatedShared{bf16_round_trip:true, shared_after_combine:true} \
    (program.rs:1110-1113,1230-1233): F32 expert GEMV accumulation; gate/up/SwiGLU activation rounded to BF16 \
    in the activation kernel; unweighted F32 expert rows, BF16 route weights and products and the running \
    sum rounded to BF16 inside the ascending-expert combine (in-place += into the residual); no \
    post-combine round trip (shared expert follows in the model, excluded here)";

impl ChainWorkload for ChainBase {
    fn kernel(&self) -> &'static str {
        "qwen4_moe_verify"
    }

    fn arch(&self) -> &'static str {
        self.ctx.arch
    }

    fn shape_json(&self) -> Value {
        let g = self.geo;
        let t = self.shape.t;
        json!({
            "preset": "fn",
            "T": t,
            "H": g.h, "I": g.i, "E": g.e, "TOP": g.top,
            "slots": g.slots(t),
            "grouped_rows": if t >= 2 { json!(g.grouped_rows(t)) } else { Value::Null },
            "residency": self.residency(),
            "route": self.shape.route.name(),
            "seed": self.shape.seed,
            "family": if t == 1 { "decode (select_decode)" } else { "prefill-family Path 2 (few-row indexed down)" },
            "weights": {
                "gate_up": "QT44 MQ4G256V2, 136 B/K256 group, [2I x H] per expert",
                "down": "MQ4G128V2, 68 B/K128 group, [H x I] per expert",
                "symmetric": true,
                "gate_up_bytes_per_expert": g.gu_bytes(),
                "down_bytes_per_expert": g.dn_bytes(),
            },
        })
    }

    fn candidate_knobs(&self) -> Value {
        if self.aa() {
            json!({ "candidate": "aa", "note": "candidate arm runs the baseline chain on its own buffers" })
        } else {
            json!({ "candidate": "iu4_sym" })
        }
    }

    fn chain_json(&self) -> Value {
        let baseline = self.baseline_json();
        let candidate = match (&self.iu4, self.aa()) {
            (None, _) => baseline.clone(),
            (Some(iu4), _) => iu4.stages_json(),
        };
        json!({ "baseline": baseline, "candidate": candidate })
    }

    fn numeric_contract(&self) -> Value {
        json!({
            "baseline": RECIPE,
            "candidate": match &self.iu4 {
                None => json!("same as baseline (A/A)"),
                Some(iu4) => iu4.numeric_contract(),
            },
            "relation": if self.aa() { "exact" } else { "numeric" },
            "promotion": "requires R6 KLD/state qualification; this is a microbench screen",
            "reference": {
                "baseline_and_aa": "cpu_f64_chain: f64 host recomputation of the whole routed block, BF16 stages as the recipe",
                "iu4_candidate": "cpu_iu4_chain (exact recipe; SiLU portable twin): the IU4 chain's own arithmetic stage by \
                                  stage (f32 FWHT256 + C2 A4 quantiser, exact int32 fold, portable SiLU + BF16, f32 FWHT128 + \
                                  C2 A4 quantiser, exact fold + BF16, BF16 combine); the NUMERIC distance to the baseline \
                                  recipe is the driver's max_abs_diff_vs_baseline",
            },
            "tolerance": {
                "baseline_and_aa": format!("per element {} * (|residual_in| + sum_j |w_j o_ji|)", REL_BOUND),
                "iu4_candidate": format!(
                    "per element: 0 (bit-exact) unless a SwiGLU activation's portable-twin value lies within {} f32 ulps of a \
                     BF16 rounding midpoint; then the summed worst deviation of the enumerated alternative roundings through \
                     the down A4 requantisation, fold and combine (+1 BF16 ulp when two or more such blocks meet in a token)",
                    AMBIGUITY_ULPS
                ),
            },
        })
    }

    fn pm_objects(&self, arm: Arm) -> Result<Vec<Object>> {
        match (arm, &self.iu4) {
            (Arm::Baseline, _) | (Arm::Candidate, None) => Ok(Vec::new()),
            (Arm::Candidate, Some(iu4)) => iu4.pm_objects(),
        }
    }

    fn setup(&mut self, gpu: &mut Gpu) -> Result<()> {
        self.check_live_selector(gpu)?;
        let geo = self.geo;
        let t = self.shape.t;
        let mapped = self.shape.mapped;
        let (gub, dnb) = (geo.gu_bytes(), geo.dn_bytes());
        // Deterministic distinct symmetric weights for every expert, in parallel.
        let mut gu = vec![0u8; geo.e * gub];
        let mut dn = vec![0u8; geo.e * dnb];
        let seed = self.shape.seed;
        gu.par_chunks_mut(gub)
            .zip(dn.par_chunks_mut(dnb))
            .enumerate()
            .for_each(|(e, (g, d))| gen_expert(geo, seed, e, g, d));
        let (gu_t, dn_t) = if mapped {
            (
                hip(gpu.upload_raw_host_mapped(&gu, &[gu.len()]))?,
                hip(gpu.upload_raw_host_mapped(&dn, &[dn.len()]))?,
            )
        } else {
            (
                hip(gpu.upload_raw(&gu, &[gu.len()]))?,
                hip(gpu.upload_raw(&dn, &[dn.len()]))?,
            )
        };
        drop((gu, dn));
        let table = |base: u64, stride: usize| -> Vec<u8> {
            (0..geo.e)
                .flat_map(|e| (base + (e * stride) as u64).to_le_bytes())
                .collect()
        };
        let gate_up_ptrs =
            hip(gpu.upload_raw(&table(gu_t.buf.as_ptr() as u64, gub), &[geo.e * 8]))?;
        let down_ptrs = hip(gpu.upload_raw(&table(dn_t.buf.as_ptr() as u64, dnb), &[geo.e * 8]))?;
        if !hip(gpu.qwen4_moe_sym_check(&gate_up_ptrs, 2 * geo.i, geo.h, geo.e, 136))? {
            return Err("qwen4_moe_sym_check rejected the generated QT44 gate/up headers".into());
        }
        if !hip(gpu.qwen4_moe_sym_check(&down_ptrs, geo.h, geo.i, geo.e, 68))? {
            return Err("qwen4_moe_sym_check rejected the generated MQ4G128 down headers".into());
        }
        let inputs = ChainInputs {
            arch: self.ctx.arch,
            t,
            h: geo.h,
            i: geo.i,
            e: geo.e,
            top: geo.top,
            host_mapped: mapped,
            x_norm: hip(gpu.upload_f32(&self.host.x_norm, &[t, geo.h]))?,
            topk_idx: hip(gpu.upload_raw(&i32_bytes(&self.host.topk_idx), &[t, geo.top, 4]))?,
            topk_w: hip(gpu.upload_f32(&self.host.topk_w, &[t, geo.top]))?,
            gate_up_ptrs,
            down_ptrs,
        };
        let base = arm_bufs(gpu, geo, t)?;
        let cand = arm_bufs(gpu, geo, t)?;
        let payload = residual_payload(&self.host.res0);
        Self::gpu_reset(gpu, &base, &payload)?;
        Self::gpu_reset(gpu, &cand, &payload)?;
        if let Some(iu4) = self.iu4.as_mut() {
            iu4.setup(gpu, &inputs)?;
        }
        hip(gpu.hip.device_synchronize())?;
        self.dev = Some(Dev {
            inputs,
            base,
            cand,
            _weights: vec![gu_t, dn_t],
        });
        Ok(())
    }

    fn encode_chain(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()> {
        let arch = self.ctx.arch;
        let t = self.shape.t;
        let d = self
            .dev
            .as_ref()
            .ok_or_else(|| "qwen4_moe_verify: setup has not run".to_string())?;
        let bufs = Self::arm_bufs(d, arm, self.swapped);
        match (arm, self.iu4.as_mut()) {
            (Arm::Baseline, _) | (Arm::Candidate, None) => {
                encode_shipped(gpu, arch, t, &d.inputs, bufs)
            }
            (Arm::Candidate, Some(iu4)) => iu4.encode(gpu, &d.inputs, &bufs.residual),
        }
    }

    fn set_buffer_swap(&mut self, swapped: bool) {
        self.swapped = swapped;
    }

    fn reset(&mut self, gpu: &mut Gpu, arm: Arm) -> Result<()> {
        let payload = residual_payload(&self.host.res0);
        let d = self.dev()?;
        Self::gpu_reset(gpu, Self::arm_bufs(d, arm, self.swapped), &payload)
    }

    fn outputs(&self, gpu: &mut Gpu, arm: Arm) -> Result<Vec<u8>> {
        let d = self.dev()?;
        hip(gpu.hip.device_synchronize())?;
        hip(gpu.download_raw_bytes(&Self::arm_bufs(d, arm, self.swapped).res_full))
    }

    fn reference(&self, arm: Arm, out: &[u8]) -> Result<RefReport> {
        let payload = parse_f32(out)?;
        match (arm, &self.iu4) {
            (Arm::Baseline, _) | (Arm::Candidate, None) => {
                let rd = self.ref_data()?;
                compare("cpu_f64_chain", &rd.out, &baseline_bounds(rd), &payload)
            }
            (Arm::Candidate, Some(_)) => {
                let r = self.iu4_ref()?;
                compare(
                    "cpu_iu4_chain (exact recipe; SiLU portable twin)",
                    &r.out,
                    &r.tol,
                    &payload,
                )
            }
        }
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        let (a, b) = (parse_f32(a)?, parse_f32(b)?);
        if a.len() != b.len() {
            return Err(format!(
                "output payloads differ in length ({} vs {} floats)",
                a.len(),
                b.len()
            ));
        }
        let mut m = 0.0f64;
        for (x, y) in a.iter().zip(&b) {
            if x.is_nan() != y.is_nan() {
                return Ok(f64::INFINITY);
            }
            if x.is_nan() || x.to_bits() == y.to_bits() {
                continue;
            }
            let d = (*x as f64 - *y as f64).abs();
            if d.is_nan() {
                return Ok(f64::INFINITY);
            }
            m = m.max(d);
        }
        Ok(m)
    }

    fn requires_baseline_identity(&self) -> bool {
        self.aa()
    }
}

// ---------------------------------------------------------------------------
// Tests (GPU-free)

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(arch: &'static str, aa: bool) -> ChainCtx {
        ChainCtx {
            arch,
            null_control: aa,
        }
    }

    fn knobs(pairs: &[(&str, &str)]) -> Knobs {
        let mut k = Knobs::default();
        for (a, b) in pairs {
            k.0.insert((*a).to_string(), (*b).to_string());
        }
        k
    }

    #[test]
    fn shape_defaults_and_keys() {
        let s = parse_shape("fn").unwrap();
        assert_eq!(
            s,
            Shape {
                t: 4,
                mapped: false,
                route: RouteKind::Uniform,
                seed: DEFAULT_SEED
            }
        );
        assert_eq!(parse_shape("").unwrap(), s);
        let s = parse_shape("fn,T=7, res=mapped ,route=skew,seed=9").unwrap();
        assert_eq!(
            (s.t, s.mapped, s.route, s.seed),
            (7, true, RouteKind::Skew, 9)
        );
        for t in SUPPORTED_T {
            assert_eq!(parse_shape(&format!("fn,T={t}")).unwrap().t, t);
        }
    }

    #[test]
    fn shape_refusals() {
        for bad in [
            "fn,T=5",
            "fn,T=6",
            "fn,T=0",
            "fn,T=9",
            "fn,T=512",
            "fn,T=x",
            "fn,res=ssd",
            "fn,route=all",
            "fn,T=2,T=3",
            "fn,foo=1",
            "a3b",
            "a3b,T=4",
            "T=4",
            "fn,4",
            "fn,seed=-1",
        ] {
            assert!(parse_shape(bad).is_err(), "{bad} must be refused");
        }
        let e = parse_shape("fn,T=5").unwrap_err();
        assert!(e.contains("T=5") && e.contains("[1, 2, 3, 4, 7, 8]"), "{e}");
        assert!(parse_shape("a3b").unwrap_err().contains("`fn`"));
    }

    #[test]
    fn candidate_knob_rules() {
        // aa + --candidate refused.
        let e = new(
            "fn",
            &mut knobs(&[("candidate", "iu4_sym")]),
            ctx("gfx1201", true),
        )
        .err()
        .unwrap();
        assert!(e.contains("refused under --null-control aa"), "{e}");
        // Missing candidate without aa.
        let e = new("fn", &mut knobs(&[]), ctx("gfx1201", false))
            .err()
            .unwrap();
        assert!(e.contains("--candidate iu4_sym"), "{e}");
        // Unknown candidate.
        let e = new(
            "fn",
            &mut knobs(&[("candidate", "wmma")]),
            ctx("gfx1201", false),
        )
        .err()
        .unwrap();
        assert!(e.contains("expected iu4_sym"), "{e}");
        // Unsupported arch.
        let e = new("fn", &mut knobs(&[]), ctx("gfx942", true))
            .err()
            .unwrap();
        assert!(e.contains("gfx942"), "{e}");
        // Bad shapes surface through the constructor.
        assert!(new("fn,T=6", &mut knobs(&[]), ctx("gfx1201", true)).is_err());
        // aa constructs and identity is required.
        let w = new("fn,T=3", &mut knobs(&[]), ctx("gfx1100", true)).unwrap();
        assert!(w.requires_baseline_identity());
        assert_eq!(w.kernel(), "qwen4_moe_verify");
        assert_eq!(w.numeric_contract()["relation"], "exact");
    }

    #[test]
    fn aa_candidate_chain_equals_baseline_bytes() {
        for arch in SUPPORTED_ARCH {
            for t in SUPPORTED_T {
                for res in ["resident", "mapped"] {
                    let w = new(
                        &format!("fn,T={t},res={res}"),
                        &mut knobs(&[]),
                        ctx(arch, true),
                    )
                    .unwrap();
                    let c = w.chain_json();
                    assert_eq!(
                        serde_json::to_vec(&c["baseline"]).unwrap(),
                        serde_json::to_vec(&c["candidate"]).unwrap()
                    );
                    assert!(w.pm_objects(Arm::Candidate).unwrap().is_empty());
                    assert!(w.pm_objects(Arm::Baseline).unwrap().is_empty());
                    let stages = c["baseline"]["stages"].as_array().unwrap();
                    assert_eq!(stages.len(), if t == 1 { 5 } else { 6 });
                    let gu = stages.iter().find(|s| s["stage"] == "gate_up").unwrap();
                    if t >= 2 {
                        let inverse = gu["launcher"].as_str().unwrap().contains("_inverse");
                        assert_eq!(inverse, arch == "gfx1151");
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_rows_and_scratch() {
        let g = FN_GEO;
        // G(r) of gpu_forward.rs:16488 tests: (S + 15*min(S,512)) rounded up to 16.
        assert_eq!(g.grouped_rows(1), 160);
        assert_eq!(g.grouped_rows(2), 320);
        assert_eq!(g.grouped_rows(4), 640);
        assert_eq!(g.grouped_rows(8), 1280);
        for t in SUPPORTED_T {
            assert_eq!(g.grouped_rows(t) % 16, 0);
            assert!(g.grouped_rows(t) >= g.slots(t));
            assert!(arm_scratch_bytes(g, t) > 0);
        }
        assert_eq!(g.gu_bytes(), 1_740_800);
        assert_eq!(g.dn_bytes(), 870_400);
        assert_eq!(g.weights_bytes(), 512 * 2_611_200);
    }

    #[test]
    fn generator_is_symmetric_sized_and_distinct() {
        let g = FN_GEO;
        let (mut gu0, mut dn0) = (vec![0u8; g.gu_bytes()], vec![0u8; g.dn_bytes()]);
        let (mut gu1, mut dn1) = (vec![0u8; g.gu_bytes()], vec![0u8; g.dn_bytes()]);
        gen_expert(g, 7, 0, &mut gu0, &mut dn0);
        gen_expert(g, 7, 1, &mut gu1, &mut dn1);
        assert_eq!(gu0.len(), 10 * 136 * 1280);
        assert_eq!(dn0.len(), 5 * 68 * 2560);
        check_symmetric(&gu0, &dn0).unwrap();
        check_symmetric(&gu1, &dn1).unwrap();
        assert_ne!(gu0, gu1);
        assert_ne!(dn0, dn1);
        // Deterministic.
        let (mut gu2, mut dn2) = (vec![0u8; g.gu_bytes()], vec![0u8; g.dn_bytes()]);
        gen_expert(g, 7, 0, &mut gu2, &mut dn2);
        assert_eq!((gu0, dn0), (gu2, dn2));
        // The symmetry checker rejects a broken header.
        let mut bad = gu1.clone();
        bad[2] ^= 0x01;
        assert!(check_symmetric(&bad, &dn1).is_err());
    }

    #[test]
    fn route_is_valid_and_skewed() {
        let u = host_inputs(FN_GEO, 8, RouteKind::Uniform, 3).unwrap();
        let s = host_inputs(FN_GEO, 8, RouteKind::Skew, 3).unwrap();
        for h in [&u, &s] {
            for t in 0..8 {
                let row = &h.topk_idx[t * 10..(t + 1) * 10];
                let mut d = row.to_vec();
                d.sort();
                d.dedup();
                assert_eq!(d.len(), 10);
                assert!(row.iter().all(|&e| (0..512).contains(&e)));
                let w = &h.topk_w[t * 10..(t + 1) * 10];
                assert!(w.iter().all(|&x| x > 0.0));
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
        }
        for t in 0..8 {
            assert_eq!(&s.topk_idx[t * 10..t * 10 + 5], &[0, 1, 2, 3, 4]);
        }
    }

    /// Tiny geometry for the reference self-consistency checks.
    const TINY: Geo = Geo {
        h: 256,
        i: 128,
        e: 4,
        top: 2,
    };

    fn hadamard(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| {
                        if (r & c).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Explicit-matrix evaluation of the whole block (no butterfly, no row decoder reuse).
    fn dense_chain(geo: Geo, t: usize, host: &HostInputs, seed: u64) -> Vec<f32> {
        let (h, i, top) = (geo.h, geo.i, geo.top);
        let had = |n: usize, s1: &[f32], s2: &[f32], x: &[f64]| -> Vec<f64> {
            let hm = hadamard(n);
            let pre: Vec<f64> = (0..n).map(|k| x[k] * s1[k] as f64).collect();
            (0..n)
                .map(|r| {
                    let acc: f64 = (0..n).map(|c| hm[r][c] * pre[c]).sum();
                    acc / (n as f64).sqrt() * s2[r] as f64
                })
                .collect()
        };
        let (s1_256, s2_256) = (gen_fwht_signs(42, 256), gen_fwht_signs(1042, 256));
        let (s1_128, s2_128) = (gen_fwht_signs(43, 128), gen_fwht_signs(1043, 128));
        let mut out = vec![0.0f32; t * h];
        for tt in 0..t {
            let mut xr = vec![0.0f64; h];
            for g in 0..h / 256 {
                let seg: Vec<f64> = (0..256)
                    .map(|k| host.x_norm[tt * h + g * 256 + k] as f64)
                    .collect();
                xr[g * 256..(g + 1) * 256].copy_from_slice(&had(
                    256,
                    &s1_256[..],
                    &s2_256[..],
                    &seg,
                ));
            }
            let mut contrib: Vec<(i32, f32, Vec<f32>)> = Vec::new();
            for j in 0..top {
                let e = host.topk_idx[tt * top + j] as usize;
                let (mut gu, mut dn) = (vec![0u8; geo.gu_bytes()], vec![0u8; geo.dn_bytes()]);
                gen_expert(geo, seed, e, &mut gu, &mut dn);
                let nib = |bytes: &[u8], idx: usize| -> f64 {
                    let b = bytes[idx / 2];
                    (if idx % 2 == 0 { b & 0xF } else { b >> 4 }) as f64
                };
                // Dense gate_up matrix straight from the layout description.
                let mut gate = vec![0.0f32; i];
                let mut up = vec![0.0f32; i];
                for r in 0..2 * i {
                    let mut acc = 0.0f64;
                    for c in 0..h {
                        let g = c / 256;
                        let gp = &gu[(r * (h / 256) + g) * 136..][..136];
                        let half = (c % 256) / 128;
                        let sc =
                            f16_val(u16::from_le_bytes([gp[4 * half], gp[4 * half + 1]])) as f64;
                        acc += sc * (nib(&gp[8..], c % 256) - 8.0) * xr[c];
                    }
                    if r < i {
                        gate[r] = acc as f32
                    } else {
                        up[r - i] = acc as f32
                    }
                }
                let act: Vec<f64> = (0..i)
                    .map(|k| {
                        let v = bf16r(gate[k]);
                        let u = bf16r(up[k]);
                        bf16r((v / (1.0 + (-v).exp())) * u) as f64
                    })
                    .collect();
                let rot = had(128, &s1_128[..], &s2_128[..], &act);
                let mut o = vec![0.0f32; h];
                for r in 0..h {
                    let mut acc = 0.0f64;
                    for c in 0..i {
                        let gp = &dn[(r * (i / 128) + c / 128) * 68..][..68];
                        let sc = f16_val(u16::from_le_bytes([gp[0], gp[1]])) as f64;
                        acc += sc * (nib(&gp[4..], c % 128) - 8.0) * rot[c];
                    }
                    o[r] = acc as f32;
                }
                contrib.push((host.topk_idx[tt * top + j], host.topk_w[tt * top + j], o));
            }
            contrib.sort_by_key(|c| c.0);
            for col in 0..h {
                let mut acc = bf16r(host.res0[tt * h + col]);
                for (_, w, o) in &contrib {
                    let prod = bf16r(bf16r(o[col]) * bf16r(*w));
                    acc = bf16r(acc + prod);
                }
                out[tt * h + col] = acc;
            }
        }
        out
    }

    #[test]
    fn reference_matches_dense_matrix_evaluation() {
        let t = 3;
        let seed = 11;
        let host = host_inputs(TINY, t, RouteKind::Uniform, seed).unwrap();
        let experts = needed_experts(TINY, seed, &host.topk_idx).unwrap();
        let rd = cpu_chain(TINY, t, &host, &experts);
        let dense = dense_chain(TINY, t, &host, seed);
        let mut worst = 0.0f64;
        for i in 0..rd.out.len() {
            let e = (rd.out[i] as f64 - dense[i] as f64).abs() / rd.s[i].max(1e-30);
            worst = worst.max(e);
        }
        assert!(
            worst < 0.02,
            "dense vs butterfly reference worst rel-to-S {worst}"
        );
        // The routed contribution is non-trivial and the combine moved the residual.
        let moved = rd
            .out
            .iter()
            .zip(&host.res0)
            .map(|(o, r)| (*o as f64 - *r as f64).abs())
            .fold(0.0, f64::max);
        assert!(moved > 1e-4, "routed contribution {moved}");
    }

    #[test]
    fn fwht_roundtrip_is_orthonormal() {
        let (s1, s2) = (gen_fwht_signs(43, 128), gen_fwht_signs(1043, 128));
        let x0: Vec<f64> = (0..128)
            .map(|i| ((i * 37 % 29) as f64 - 14.0) / 7.0)
            .collect();
        let mut x = x0.clone();
        fwht_f64(&mut x, &s1, &s2);
        let (n0, n1): (f64, f64) = (
            x0.iter().map(|v| v * v).sum::<f64>(),
            x.iter().map(|v| v * v).sum::<f64>(),
        );
        assert!((n0 - n1).abs() < 1e-9 * n0);
        // Inverse: swap the sign roles.
        fwht_f64(&mut x, &s2, &s1);
        for (a, b) in x.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_weights_leave_residual_rounded() {
        // All nibbles 8 => w = 0 => the routed contribution is exactly zero.
        let t = 2;
        let host = host_inputs(TINY, t, RouteKind::Uniform, 5).unwrap();
        let mut experts: ExpertBytes = HashMap::new();
        for e in 0..TINY.e {
            let (mut gu, mut dn) = (vec![0u8; TINY.gu_bytes()], vec![0u8; TINY.dn_bytes()]);
            gen_expert(TINY, 5, e, &mut gu, &mut dn);
            for g in gu.chunks_exact_mut(136) {
                g[8..].fill(0x88);
            }
            for g in dn.chunks_exact_mut(68) {
                g[4..].fill(0x88);
            }
            experts.insert(e, (gu, dn));
        }
        let rd = cpu_chain(TINY, t, &host, &experts);
        for (o, r) in rd.out.iter().zip(&host.res0) {
            assert_eq!(*o, bf16r(*r));
        }
    }

    #[test]
    fn compare_bounds_and_guard() {
        let reference = [1.0f32, 2.0];
        let bound = [0.625f64, 0.625];
        let mut payload = vec![1.0f32, 2.0];
        payload.extend(std::iter::repeat(f32::from_bits(GUARD_BITS)).take(GUARD_FLOATS));
        let ok = compare("r", &reference, &bound, &payload).unwrap();
        assert!(ok.pass && ok.max_abs_err == 0.0);
        // Error inside the per-element bound and outside.
        payload[0] = 1.5;
        assert!(compare("r", &reference, &bound, &payload).unwrap().pass);
        payload[0] = 2.0;
        let bad = compare("r", &reference, &bound, &payload).unwrap();
        assert!(!bad.pass && bad.max_abs_err > bad.tolerance);
        // A zero bound is an exact check.
        payload[0] = 1.0;
        assert!(
            compare("r", &reference, &[0.0, 0.0], &payload)
                .unwrap()
                .pass
        );
        payload[0] = f32::from_bits(1.0f32.to_bits() + 1);
        assert!(
            !compare("r", &reference, &[0.0, 0.0], &payload)
                .unwrap()
                .pass
        );
        // Guard corruption and non-finite values fail.
        payload[0] = 1.0;
        payload[2] = 0.0;
        assert!(!compare("r", &reference, &bound, &payload).unwrap().pass);
        payload[2] = f32::from_bits(GUARD_BITS);
        payload[1] = f32::NAN;
        assert!(!compare("r", &reference, &bound, &payload).unwrap().pass);
        assert!(compare("r", &reference, &bound, &payload[..2]).is_err());
        // The baseline bound is REL_BOUND * S_i.
        let rd = RefData {
            out: vec![1.0, 2.0],
            s: vec![10.0, 10.0],
        };
        assert_eq!(baseline_bounds(&rd), vec![0.625, 0.625]);
    }

    fn normals(seed: u64, n: usize) -> Vec<f32> {
        let mut r = Mix(seed);
        (0..n).map(|_| r.normal()).collect()
    }

    #[test]
    fn rotate256_is_the_f64_fwht_to_f32_precision() {
        let signs = Signs::new();
        let x = normals(21, 256);
        let got = rotate256(&x, &signs.s1_256, &signs.s2_256);
        let mut want: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        fwht_f64(&mut want, &signs.s1_256, &signs.s2_256);
        for (g, w) in got.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1e-5, "{g} vs {w}");
        }
    }

    #[test]
    fn quantiser_blocks_reconstruct_the_rotation_and_zero_is_a_unit_block() {
        let signs = Signs::new();
        let x = normals(22, 512);
        let xr = gate_xrow(&x, &signs);
        assert_eq!((xr.d.len(), xr.v.len()), (4, 512));
        for g in 0..2 {
            let r = rotate256(&x[g * 256..(g + 1) * 256], &signs.s1_256, &signs.s2_256);
            for half in 0..2 {
                let blk = g * 2 + half;
                let d = xr.d[blk];
                assert!(d > 0.0 && d.is_finite());
                for i in 0..GROUP {
                    let q = xr.v[blk * GROUP + i];
                    assert!((-8..=7).contains(&q));
                    // Round-to-nearest within the clip range (C2 candidates clip at most 1.17 d).
                    let err = (r[half * GROUP + i] - q as f32 * d).abs();
                    assert!(err <= 1.2 * d, "block {blk} elem {i}: err {err} d {d}");
                }
            }
        }
        let zero = gate_xrow(&vec![0.0f32; 256], &signs);
        assert_eq!(zero.d, vec![1.0, 1.0]);
        assert!(zero.v.iter().all(|&q| q == 0));
    }

    #[test]
    fn fold_matches_the_dequantised_dot() {
        // The exact int32-per-epoch fold equals the f64 dot of the dequantised weights with the
        // dequantised activations (d * q) to f32 accumulation precision.
        let signs = Signs::new();
        let x = normals(23, TINY.h);
        let xr = gate_xrow(&x, &signs);
        let (mut gu, mut dn) = (vec![0u8; TINY.gu_bytes()], vec![0u8; TINY.dn_bytes()]);
        gen_expert(TINY, 3, 1, &mut gu, &mut dn);
        let sums = fold_slots(
            Kind::GateUp,
            2 * TINY.i,
            TINY.h / GROUP,
            TINY.gu_row_bytes(),
            &gu,
            std::slice::from_ref(&xr),
        );
        let mut w = vec![0.0f64; TINY.h];
        for row in (0..2 * TINY.i).step_by(17) {
            dequant_gu_row(&gu, row, TINY.h, &mut w);
            let (mut want, mut mag) = (0.0f64, 0.0f64);
            for c in 0..TINY.h {
                let a = xr.d[c / GROUP] as f64 * xr.v[c] as f64;
                want += w[c] * a;
                mag += (w[c] * a).abs();
            }
            assert!(
                (sums[row] as f64 - want).abs() <= 1e-5 * mag + 1e-9,
                "row {row}: {} vs {want}",
                sums[row]
            );
        }
    }

    #[test]
    fn activation_flags_only_roundings_within_the_margin() {
        // Random fold sums over many elements: some portable-twin values land within the margin
        // of a BF16 rounding midpoint; each flagged element names the other neighbour.
        let n = 1 << 16;
        let mut r = Mix(31);
        let sums: Vec<f32> = (0..2 * n).map(|_| 0.5 * r.normal()).collect();
        let act = activation(&sums, n);
        assert_eq!(act.bits.len(), n);
        assert!(
            !act.ambiguous.is_empty(),
            "no ambiguous element in {n} draws"
        );
        assert!(act.ambiguous.len() < n / 100);
        for &(k, alt) in &act.ambiguous {
            let p = portable_silu_mul(rt_f32(sums[k]), rt_f32(sums[n + k]));
            let (lo, hi) = (p.to_bits() >> 16, (p.to_bits() >> 16) + 1);
            let b = u32::from(act.bits[k]);
            assert!(b == lo || b == hi, "rounded neighbour");
            assert_eq!(u32::from(alt), if b == lo { hi } else { lo });
        }
    }

    #[test]
    fn bf16_ulp_and_combine_basics() {
        assert_eq!(bf16_ulp_f32(1.0), 2f64.powi(-7));
        assert_eq!(bf16_ulp_f32(2.0), 2f64.powi(-6));
        assert_eq!(bf16_ulp_f32(1.5), 2f64.powi(-7));
        // Zero contributions leave the BF16-rounded residual.
        let res0 = [0.3f32, -1.7, 4.0];
        let zeros = vec![0.0f32; 3];
        let o: Vec<&[f32]> = vec![&zeros[..], &zeros[..]];
        let out = combine_row(&res0, &[1, 0], &[0.6, 0.4], &o);
        for (a, r) in out.iter().zip(&res0) {
            assert_eq!(*a, bf16r(*r));
        }
        // Ascending expert id: swapping which rank holds which expert changes only tie order.
        let (a, b) = ([0.25f32; 3], [0.5f32; 3]);
        let x = combine_row(&res0, &[3, 1], &[0.5, 0.5], &[&a[..], &b[..]]);
        let y = combine_row(&res0, &[1, 3], &[0.5, 0.5], &[&b[..], &a[..]]);
        assert_eq!(x, y);
    }

    #[test]
    fn cpu_iu4_chain_is_deterministic_a4_noise_around_the_f64_chain() {
        let t = 3;
        let seed = 17;
        let host = host_inputs(TINY, t, RouteKind::Uniform, seed).unwrap();
        let experts = needed_experts(TINY, seed, &host.topk_idx).unwrap();
        let a = cpu_iu4_chain(TINY, t, &host, &experts).unwrap();
        let b = cpu_iu4_chain(TINY, t, &host, &experts).unwrap();
        assert_eq!(a.out, b.out);
        assert_eq!(a.tol, b.tol);
        let rd = cpu_chain(TINY, t, &host, &experts);
        let delta_max = rd
            .out
            .iter()
            .zip(&host.res0)
            .map(|(o, r)| (*o as f64 - *r as f64).abs())
            .fold(0.0, f64::max);
        assert!(delta_max > 1e-4, "routed contribution {delta_max}");
        // A4 activations move the result (it is a different recipe) but only by a fraction of the
        // routed contribution.
        let dist = a
            .out
            .iter()
            .zip(&rd.out)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0, f64::max);
        assert!(dist > 0.0, "A4 reference identical to the F32 chain");
        assert!(dist < delta_max, "dist {dist} vs routed {delta_max}");
        // Tolerances are finite and non-negative; zero everywhere when nothing is ambiguous.
        assert!(a.tol.iter().all(|t| t.is_finite() && *t >= 0.0));
        if a.ambiguous_blocks == 0 {
            assert!(a.tol.iter().all(|t| *t == 0.0));
        } else {
            assert!(a.tol.iter().any(|t| *t >= 0.0));
        }
    }

    #[test]
    fn cpu_iu4_chain_with_zero_weights_leaves_the_rounded_residual() {
        let t = 2;
        let host = host_inputs(TINY, t, RouteKind::Uniform, 5).unwrap();
        let mut experts: ExpertBytes = HashMap::new();
        for e in 0..TINY.e {
            let (mut gu, mut dn) = (vec![0u8; TINY.gu_bytes()], vec![0u8; TINY.dn_bytes()]);
            gen_expert(TINY, 5, e, &mut gu, &mut dn);
            for g in gu.chunks_exact_mut(136) {
                g[8..].fill(0x88);
            }
            for g in dn.chunks_exact_mut(68) {
                g[4..].fill(0x88);
            }
            experts.insert(e, (gu, dn));
        }
        let r = cpu_iu4_chain(TINY, t, &host, &experts).unwrap();
        for (o, x) in r.out.iter().zip(&host.res0) {
            assert_eq!(*o, bf16r(*x));
        }
        assert_eq!(r.ambiguous_elements, 0);
    }

    #[test]
    fn max_abs_diff_semantics() {
        let w = new("fn", &mut knobs(&[]), ctx("gfx1201", true)).unwrap();
        let a = f32_bytes(&[1.0, 2.0, f32::NAN]);
        let b = f32_bytes(&[1.0, 2.5, f32::NAN]);
        assert_eq!(w.max_abs_diff(&a, &a).unwrap(), 0.0);
        assert_eq!(w.max_abs_diff(&a, &b).unwrap(), 0.5);
        let c = f32_bytes(&[1.0, 2.0, 3.0]);
        assert_eq!(w.max_abs_diff(&a, &c).unwrap(), f64::INFINITY);
        assert!(w.max_abs_diff(&a, &c[..8]).is_err());
    }
}
