// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 BF16-input top-k combine of the symmetric
//! IU4 grouped MoE (`HIPFIRE_MOE_SYM_IU4`, prefill Path2): the twin of
//! `moe_down_combine_grouped_k8` for the BF16 grouped down rows the symmetric
//! down GEMM writes, `moe_down_combine_grouped_bf16in_pm_gfx1201` in the module
//! `moe_slot_ops_pm_gfx1201`. gfx1201 only (gfx1100 / gfx1151 are refused).
//!
//! ABI (eight arguments, 48 bytes): `y_grouped_bf16` `*u16` @0 (BF16 bits,
//! `[grouped_rows, dim]`), `inverse_perm` `*i32` @8 (`[n_tokens * k_top]`, flat
//! slot -> grouped row, negative = unrouted), `topk_weights` `*f32` @16
//! (`[n_tokens * k_top]`), `x_residual` `*f32` @24 (`[n_tokens, dim]`, updated in
//! place), `dim` u32 @32, `k_top` u32 @36, `grouped_rows` u32 @40, `n_tokens` u32
//! @44. Grid `[n_tokens, ceil(dim / 256), 1]`, block `[256, 1, 1]`, wave32, no
//! LDS, no barrier, no scratch, no private memory, no atomics. Contract (the
//! caller's checks): `k_top` in `1..=8`, every byte offset below `2^32`,
//! `inverse_perm` and `topk_weights` hold `n_tokens * k_top` words.
//!
//! Law (per `t` = workgroup id x, `m` = 256 * workgroup id y + tid):
//! ```text
//!   return if t >= n_tokens or m >= dim                       (no write)
//!   acc = +0.0f32
//!   for k in 0..k_top ascending:
//!       slot = inverse_perm[t * k_top + k]                    (unsigned)
//!       skip if slot >= grouped_rows                          (negative = unrouted)
//!       w = topk_weights[t * k_top + k]
//!       v = f32::from_bits(y[slot * dim + m] << 16)           (BF16 widening, exact)
//!       acc = fma(w, v, acc)                                  (one fused rounding)
//!   x_residual[t * dim + m] = RN(x_residual[t * dim + m] + acc)
//! ```
//! Outside `k_top` in `1..=8` the whole workgroup returns before any VMEM access
//! (a defensive domain guard; the host never launches there), so a bad `k_top`
//! can neither read past the slot tables nor write.
//!
//! Schedule: a runtime ascending-`k` loop of `k_top` trips (one static
//! `v_fma_f32`). The slot is workgroup-uniform (it depends on `t` and `k` only), so
//! each trip loads it with SMEM and tests `slot >=u grouped_rows` on the scalar
//! unit; a skipped slot branches over the whole rest of the trip: no weight load,
//! no `y` load, no fma. A live slot loads its weight (SMEM), forms the row byte
//! offset `slot * dim * 2` (the raw-buffer `soffset`), loads the lane's BF16
//! (`buffer_load_u16`, exec-masked to `m < dim`), widens it with a 16-bit left
//! shift and runs `v_fma_f32 acc, w, v, acc` in place. After the loop one
//! `v_add_f32` adds the accumulator to the residual (loaded there) and the sum is
//! stored. There is no mask select, no clamped or dummy load.
//!
//! The same module file also holds the v2 combine (`moe_down_combine_grouped_bf16in_v2_pm_gfx1201`,
//! module `moe_slot_ops_v2_pm_gfx1201`, kernel id `moe_slot_ops_v2_pm`): eight contiguous columns
//! per lane, grid `[n_tokens, dim / 2048, 1]`. v1 above, its contract and bundle are frozen; see
//! the `v2` section at the end of the file.
use super::common::{add64, add64_imm, mem, op, s, s_add_i32, smem, sop, sr, srd_tail, v, vr};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1201, Scc, Target, WgUniform, Workgroup};

type R = Result<(), String>;

/// Threads (output columns) per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 48;
/// Largest `k_top` the kernel accepts (a workgroup with a larger `k_top` returns).
pub const MAX_K_TOP: u32 = 8;

/// The kernel-spec id (`peacemaker-isa emit --kernel`); the symbol is [`symbol`].
pub const KERNEL_ID: &str = "moe_slot_ops_pm";

pub fn symbol(arch: Arch) -> String {
    format!("moe_down_combine_grouped_bf16in_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("moe_slot_ops_pm_{}", arch.name())
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg words 0..8 -> s8..s15 (y 8:9, inverse_perm 10:11, topk_weights 12:13, x_residual 14:15);
// words 8..12 -> s16..s19 (dim, k_top, grouped_rows, n_tokens).
const Y: u8 = 8;
const INV: u8 = 10;
const WGT: u8 = 12;
const XR: u8 = 14;
const DIM: u8 = 16;
const KTOP: u8 = 17;
const GROWS: u8 = 18;
const NTOK: u8 = 19;
const WGX: u8 = 20;
const WGY: u8 = 21;
/// Trips left.
const KLEFT: u8 = 22;
/// `t * k_top * 4`: the token's first slot byte offset.
const TK4: u8 = 23;
/// `dim * 2`: BF16 row bytes.
const DIM2: u8 = 24;
/// `t * dim * 4`: the token's residual row byte offset.
const XSOFF: u8 = 25;
const BYTES: u8 = 26;
const T0: u8 = 27;
/// Running `inverse_perm + 4 (t k_top + k)` (28:29) and `topk_weights + 4 (t k_top + k)` (30:31).
const INVP: u8 = 28;
const WP: u8 = 30;
// Raw buffer descriptors (4-aligned quads).
const SRD_Y: u8 = 32;
const SRD_X: u8 = 36;
/// The trip's slot, weight and BF16 row byte offset.
const SLOT: u8 = 40;
const WK: u8 = 41;
const YOFF: u8 = 42;
/// The `m < dim` lane mask pair (the high word is never written in wave32 and held at zero).
const LV: u8 = 44;
const S_END: u8 = 48;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const VM: u8 = 1;
const VOFF2: u8 = 2;
const VOFF4: u8 = 3;
const ACC: u8 = 4;
const XV: u8 = 5;
const YV: u8 = 6;
const V_END: u8 = 8;

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("y_grouped_bf16", 0, Access::ReadOnly)
        .pointer_access("inverse_perm", 8, Access::ReadOnly)
        .pointer_access("topk_weights", 16, Access::ReadOnly)
        .pointer_access("x_residual", 24, Access::ReadWrite)
        .hidden("dim", 32, 4, "by_value")
        .hidden("k_top", 36, 4, "by_value")
        .hidden("grouped_rows", 40, 4, "by_value")
        .hidden("n_tokens", 44, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1201 only; every other target is refused before any builder exists).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!(
            "{KERNEL_ID}: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: KERNEL_ID.into(),
        variant: "combine".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The single-symbol module `moe_slot_ops_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn m(x: u8) -> RegRef {
    sr(x, 2)
}
fn cmp_wg<T: Target>(
    wg: &mut Workgroup<'_, T, Builder>,
    text: String,
    uses: &[RegRef],
) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
/// `s{dst}:{dst+3}` raw buffer descriptor of the pointer pair at `s{ptr}` with `num_records` from `s{records}`.
fn srd(b: &mut Builder, dst: u8, ptr: u8, records: u8) -> R {
    sop(b, format!("s_mov_b32 s{dst}, s{ptr}"), &[dst], &[ptr])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", dst + 1, ptr + 1),
        &[dst + 1],
        &[ptr + 1],
    )?;
    srd_tail(b, dst, Some(records))
}
/// Raw buffer load (`buffer_load_u16` / `buffer_load_b32`) at `v{voff} + s{soff}`.
fn bload_s(b: &mut Builder, name: &str, dst: u8, voff: u8, desc: u8, soff: u8) -> R {
    mem(
        b,
        format!(
            "{name} v{dst}, v{voff}, s[{desc}:{}], s{soff} offen",
            desc + 3
        ),
        &[v(dst)],
        &[v(voff), sr(desc, 4), s(soff)],
        MemoryClass::VmemLoad,
    )
}
/// Raw buffer store of one dword at `v{voff} + s{soff}`.
fn bstore_s(b: &mut Builder, data: u8, voff: u8, desc: u8, soff: u8) -> R {
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{desc}:{}], s{soff} offen",
            desc + 3
        ),
        &[],
        &[v(voff), v(data), sr(desc, 4), s(soff)],
        MemoryClass::VmemStore,
    )
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Workgroup<'_, T, Builder>) -> R {
    let end = wg.exit(".Lcombine_end")?;
    let b = wg.isa();
    smem(b, Y, 8, KARG, 0x00)?;
    smem(b, DIM, 4, KARG, 0x20)?;
    // Workgroup ids: gfx12 `ttmp9` (x) and `ttmp7[15:0]` (y).
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    // t >= n_tokens: the workgroup returns before any access.
    let outside = cmp_wg(
        wg,
        format!("s_cmp_ge_u32 s{WGX}, s{NTOK}"),
        &[s(WGX), s(NTOK)],
    )?;
    wg.exit_if(outside, &end)?;
    // k_top outside 1..=8 (`k_top - 1 >u 7`, which also rejects 0).
    let b = wg.isa();
    sop(
        b,
        format!("{} s{T0}, s{KTOP}, -1", s_add_i32(Arch::Gfx1201)),
        &[T0],
        &[KTOP],
    )?;
    let bad_k = cmp_wg(
        wg,
        format!("s_cmp_gt_u32 s{T0}, {}", MAX_K_TOP - 1),
        &[s(T0)],
    )?;
    wg.exit_if(bad_k, &end)?;
    // 256 wg_y >= dim: no lane of this workgroup has a column.
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 8"), &[T0], &[WGY])?;
    let no_cols = cmp_wg(wg, format!("s_cmp_ge_u32 s{T0}, s{DIM}"), &[s(T0), s(DIM)])?;
    wg.exit_if(no_cols, &end)?;

    let b = wg.isa();
    // Uniform byte offsets, the running slot / weight pointers and the two raw descriptors.
    sop(b, format!("s_mov_b32 s{KLEFT}, s{KTOP}"), &[KLEFT], &[KTOP])?;
    sop(
        b,
        format!("s_mul_i32 s{TK4}, s{WGX}, s{KTOP}"),
        &[TK4],
        &[WGX, KTOP],
    )?;
    sop(b, format!("s_lshl_b32 s{TK4}, s{TK4}, 2"), &[TK4], &[TK4])?;
    add64(b, INVP, INV, TK4)?;
    add64(b, WP, WGT, TK4)?;
    sop(b, format!("s_lshl_b32 s{DIM2}, s{DIM}, 1"), &[DIM2], &[DIM])?;
    sop(
        b,
        format!("s_mul_i32 s{XSOFF}, s{WGX}, s{DIM}"),
        &[XSOFF],
        &[WGX, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{XSOFF}, s{XSOFF}, 2"),
        &[XSOFF],
        &[XSOFF],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{GROWS}, s{DIM2}"),
        &[BYTES],
        &[GROWS, DIM2],
    )?;
    srd(b, SRD_Y, Y, BYTES)?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{NTOK}, s{DIM}"),
        &[BYTES],
        &[NTOK, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{BYTES}, s{BYTES}, 2"),
        &[BYTES],
        &[BYTES],
    )?;
    srd(b, SRD_X, XR, BYTES)?;
    // The lane-mask pair's high word (wave32 never writes it; VOP3 reads the pair).
    sop(b, format!("s_mov_b32 s{}, 0", LV + 1), &[LV + 1], &[])?;
    // Lane column m = 256 wg_y + tid and its byte offsets; lanes with m >= dim leave exec.
    op(
        b,
        format!("v_lshl_add_u32 v{VM}, s{WGY}, 8, v{TID}"),
        &[v(VM)],
        &[s(WGY), v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF2}, 1, v{VM}"),
        &[v(VOFF2)],
        &[v(VM)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF4}, 2, v{VM}"),
        &[v(VOFF4)],
        &[v(VM)],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{LV}, s{DIM}, v{VM}"),
        &[m(LV)],
        &[s(DIM), v(VM)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{LV}"),
        &[],
        &[m(LV)],
    )?;
    // acc = +0.
    op(b, format!("v_mov_b32_e32 v{ACC}, 0"), &[v(ACC)], &[])?;
    b.wait_all()?;

    // k = 0..k_top ascending (k_top >= 1: a do-while). Per trip the slot word is always in range of the token's k_top words.
    wg.loop_carried(".Lcombine_trip", (), |wg, ()| {
        let b = wg.isa();
        smem(b, SLOT, 1, INVP, 0)?;
        // Unrouted (negative) or out-of-range slot: skip the weight, the row and the fma.
        let dead = cmp_wg(
            wg,
            format!("s_cmp_ge_u32 s{SLOT}, s{GROWS}"),
            &[s(SLOT), s(GROWS)],
        )?;
        wg.wg_skip_if(dead, ".Lcombine_skip", (), |wg, ()| {
            let b = wg.isa();
            smem(b, WK, 1, WP, 0)?;
            sop(
                b,
                format!("s_mul_i32 s{YOFF}, s{SLOT}, s{DIM2}"),
                &[YOFF],
                &[SLOT, DIM2],
            )?;
            bload_s(b, "buffer_load_u16", YV, VOFF2, SRD_Y, YOFF)?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{YV}, 16, v{YV}"),
                &[v(YV)],
                &[v(YV)],
            )?;
            op(
                b,
                format!("v_fma_f32 v{ACC}, s{WK}, v{YV}, v{ACC}"),
                &[v(ACC)],
                &[s(WK), v(YV), v(ACC)],
            )?;
            b.wait_all()
        })?;
        let b = wg.isa();
        add64_imm(b, INVP, 4)?;
        add64_imm(b, WP, 4)?;
        sop(
            b,
            format!("{} s{KLEFT}, s{KLEFT}, -1", s_add_i32(Arch::Gfx1201)),
            &[KLEFT],
            &[KLEFT],
        )?;
        let more = cmp_wg(wg, format!("s_cmp_gt_i32 s{KLEFT}, 0"), &[s(KLEFT)])?;
        Ok(((), more))
    })?;

    // x_residual = RN(x_residual + acc).
    let b = wg.isa();
    bload_s(b, "buffer_load_b32", XV, VOFF4, SRD_X, XSOFF)?;
    op(
        b,
        format!("v_add_f32_e32 v{XV}, v{XV}, v{ACC}"),
        &[v(XV)],
        &[v(XV), v(ACC)],
    )?;
    bstore_s(b, XV, VOFF4, SRD_X, XSOFF)?;
    b.wait_all()?;
    wg.end(end)
}

/// Shape contract of the emitted kernel (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory, arithmetic and control
/// mnemonics the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b128",
        "s_load_b256",
        "buffer_load_u16",
        "buffer_load_b32",
        "buffer_store_b32",
        "v_fma_f32",
        "v_add_f32_e32",
        "s_barrier_signal",
        "s_barrier_wait",
        "s_endpgm",
    ] {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*",
            "v_mul_f32*", "v_fmac_f32*", "v_mac_f32*", "v_mad_f32*", "v_dual_*", "v_pk_*", "v_cndmask_*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}

// =====================================================================================
// v2: eight contiguous columns per lane (`moe_down_combine_grouped_bf16in_v2_pm_gfx1201`,
// module `moe_slot_ops_v2_pm_gfx1201`). Everything above is the frozen v1 emitter: it,
// its contract and `kernels/moe_slot_ops_pm_gfx1201.hxaco` are not touched by v2.
// =====================================================================================
//
// ABI: the v1 kernarg block byte for byte (eight arguments, 48 bytes; offsets 0, 8, 16, 24,
// 32, 36, 40, 44). Grid `[n_tokens, dim / 2048, 1]`, block `[256, 1, 1]`, wave32, no LDS,
// no barrier, no scratch, no atomics. Shape contract (host-validated by [`validate_shape`]
// and guarded in the kernel): `dim` positive multiple of 2048, `k_top` in `1..=8`, every
// byte offset below `2^32`, `inverse_perm` and `topk_weights` hold `n_tokens * k_top` words.
//
// Law (per `t` = workgroup id x, `m0` = 2048 * workgroup id y + 8 * tid, columns `m0..m0+8`):
// ```text
//   return if t >= n_tokens or 2048 wg_y >= dim or dim % 2048 != 0 or k_top not in 1..=8
//   acc[0..8] = +0.0f32
//   for k in 0..k_top ascending:
//       slot = inverse_perm[t * k_top + k]                    (unsigned)
//       skip if slot >= grouped_rows                          (negative = unrouted)
//       w = topk_weights[t * k_top + k]
//       for i in 0..8: acc[i] = fma(w, f32::from_bits(y[slot * dim + m0 + i] << 16), acc[i])
//   for i in 0..8: x_residual[t * dim + m0 + i] = RN(x_residual[t * dim + m0 + i] + acc[i])
// ```
// Every lane has all eight columns (the dim guard makes ragged lanes impossible), so there
// is no exec mask and no clamped load. Per live trip the slot and weight come from SMEM
// (workgroup-uniform) and are reused across the eight columns: one `buffer_load_b128` of
// the lane's eight BF16 values (soffset = `slot * dim * 2`), the eight widenings (even
// element: 16-bit left shift; odd element: `& 0xffff0000`) and eight independent
// `v_fma_f32`. The residual is two `buffer_load_b128` / `buffer_store_b128` (offsets 0 and
// 16) around eight `v_add_f32`.

/// Output columns per lane.
pub const COLS_PER_LANE_V2: u32 = 8;
/// Output columns per workgroup (`BLOCK * COLS_PER_LANE_V2`); `dim` must be a positive multiple.
pub const COLS_PER_BLOCK_V2: u32 = BLOCK * COLS_PER_LANE_V2;

/// The kernel-spec id of the v2 module (`peacemaker-isa emit --kernel`).
pub const KERNEL_ID_V2: &str = "moe_slot_ops_v2_pm";

pub fn symbol_v2(arch: Arch) -> String {
    format!("moe_down_combine_grouped_bf16in_v2_pm_{}", arch.name())
}
pub fn module_v2(arch: Arch) -> String {
    format!("moe_slot_ops_v2_pm_{}", arch.name())
}

/// Host-side shape validation of a v2 launch: `dim` a positive multiple of
/// [`COLS_PER_BLOCK_V2`] and `k_top` in `1..=`[`MAX_K_TOP`]. Returns the launch grid
/// `[n_tokens, dim / 2048, 1]`; the kernel itself returns (writes nothing) outside the domain.
pub fn validate_shape(dim: u32, k_top: u32, n_tokens: u32) -> Result<[u32; 3], String> {
    if dim == 0 || dim % COLS_PER_BLOCK_V2 != 0 {
        return Err(format!(
            "{KERNEL_ID_V2}: dim {dim} is not a positive multiple of {COLS_PER_BLOCK_V2}"
        ));
    }
    if !(1..=MAX_K_TOP).contains(&k_top) {
        return Err(format!(
            "{KERNEL_ID_V2}: k_top {k_top} outside 1..={MAX_K_TOP}"
        ));
    }
    Ok([n_tokens, dim / COLS_PER_BLOCK_V2, 1])
}

// ---- v2 VGPRs (the SGPR map above is reused; the lane-mask pair is gone) ------------
const V2_TID: u8 = 0;
/// `16 tid + 4096 wg_y`: byte offset of the lane's eight BF16 columns inside a `y` row.
const V2_VOFF2: u8 = 1;
/// `32 tid + 8192 wg_y`: byte offset of the lane's eight F32 columns inside a residual row.
const V2_VOFF4: u8 = 2;
/// Odd-element widening scratch.
const V2_TMP: u8 = 3;
/// Eight accumulators `v4..v11`.
const V2_ACC: u8 = 4;
/// The lane's four BF16 pairs `v12..v15` (`buffer_load_b128`).
const V2_YV: u8 = 12;
/// The lane's eight residual F32 `v16..v23` (two `buffer_load_b128`).
const V2_XV: u8 = 16;
const V2_V_END: u8 = 24;
/// SGPRs `s0..s43` (the v1 map minus the lane-mask pair at 44:45).
const V2_S_END: u8 = 44;

fn plan_v2() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V2_V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..40).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    p.add_range("s_blk40", Kind::S, 40, V2_S_END - 40, Live::Whole)?;
    for base in (0u8..V2_V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted v2 kernel (gfx1201 only; every other target is refused before any builder exists).
pub fn emit_v2(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!(
            "{KERNEL_ID_V2}: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: KERNEL_ID_V2.into(),
        variant: "combine_v2".into(),
        arch,
        symbol: symbol_v2(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan_v2()?);
    b.enable_delay_alu();
    kernel_v2(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The single-symbol module `moe_slot_ops_v2_pm_<arch>`.
pub fn emit_module_v2(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit_v2(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module_v2(arch))?;
    Ok((emitted, text, proof))
}

/// Raw `buffer_load_b128` of `v[dst:dst+3]` at `v{voff} + s{soff} + imm`.
fn bload128_s(b: &mut Builder, dst: u8, voff: u8, desc: u8, soff: u8, imm: u32) -> R {
    if imm > b.spec.arch.buffer_offset_max() {
        return Err(format!("buffer offset {imm} exceeds the field"));
    }
    let suffix = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    mem(
        b,
        format!(
            "buffer_load_b128 v[{dst}:{}], v{voff}, s[{desc}:{}], s{soff} offen{suffix}",
            dst + 3,
            desc + 3
        ),
        &[vr(dst, 4)],
        &[v(voff), sr(desc, 4), s(soff)],
        MemoryClass::VmemLoad,
    )
}
/// Raw `buffer_store_b128` of `v[data:data+3]` at `v{voff} + s{soff} + imm`.
fn bstore128_s(b: &mut Builder, data: u8, voff: u8, desc: u8, soff: u8, imm: u32) -> R {
    if imm > b.spec.arch.buffer_offset_max() {
        return Err(format!("buffer offset {imm} exceeds the field"));
    }
    let suffix = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    mem(
        b,
        format!(
            "buffer_store_b128 v[{data}:{}], v{voff}, s[{desc}:{}], s{soff} offen{suffix}",
            data + 3,
            desc + 3
        ),
        &[],
        &[v(voff), vr(data, 4), sr(desc, 4), s(soff)],
        MemoryClass::VmemStore,
    )
}

fn kernel_v2<T: Target>(wg: &mut Workgroup<'_, T, Builder>) -> R {
    let end = wg.exit(".Lcombine_v2_end")?;
    let b = wg.isa();
    smem(b, Y, 8, KARG, 0x00)?;
    smem(b, DIM, 4, KARG, 0x20)?;
    // Workgroup ids: gfx12 `ttmp9` (x) and `ttmp7[15:0]` (y).
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    // t >= n_tokens: the workgroup returns before any access.
    let outside = cmp_wg(
        wg,
        format!("s_cmp_ge_u32 s{WGX}, s{NTOK}"),
        &[s(WGX), s(NTOK)],
    )?;
    wg.exit_if(outside, &end)?;
    // k_top outside 1..=8 (`k_top - 1 >u 7`, which also rejects 0).
    let b = wg.isa();
    sop(
        b,
        format!("{} s{T0}, s{KTOP}, -1", s_add_i32(Arch::Gfx1201)),
        &[T0],
        &[KTOP],
    )?;
    let bad_k = cmp_wg(
        wg,
        format!("s_cmp_gt_u32 s{T0}, {}", MAX_K_TOP - 1),
        &[s(T0)],
    )?;
    wg.exit_if(bad_k, &end)?;
    // dim % 2048 != 0: no workgroup of the launch has a whole 2048-column block.
    let b = wg.isa();
    // 2047 is not an inline constant: the mask goes through the (still dead) `BYTES` scalar.
    sop(
        b,
        format!("s_mov_b32 s{BYTES}, {:#x}", COLS_PER_BLOCK_V2 - 1),
        &[BYTES],
        &[],
    )?;
    sop(
        b,
        format!("s_and_b32 s{T0}, s{DIM}, s{BYTES}"),
        &[T0],
        &[DIM, BYTES],
    )?;
    let ragged = cmp_wg(wg, format!("s_cmp_lg_u32 s{T0}, 0"), &[s(T0)])?;
    wg.exit_if(ragged, &end)?;
    // 2048 wg_y >= dim (also dim == 0): the workgroup is past the last column block.
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 11"), &[T0], &[WGY])?;
    let no_cols = cmp_wg(wg, format!("s_cmp_ge_u32 s{T0}, s{DIM}"), &[s(T0), s(DIM)])?;
    wg.exit_if(no_cols, &end)?;

    let b = wg.isa();
    // Uniform byte offsets, the running slot / weight pointers and the two raw descriptors.
    sop(b, format!("s_mov_b32 s{KLEFT}, s{KTOP}"), &[KLEFT], &[KTOP])?;
    sop(
        b,
        format!("s_mul_i32 s{TK4}, s{WGX}, s{KTOP}"),
        &[TK4],
        &[WGX, KTOP],
    )?;
    sop(b, format!("s_lshl_b32 s{TK4}, s{TK4}, 2"), &[TK4], &[TK4])?;
    add64(b, INVP, INV, TK4)?;
    add64(b, WP, WGT, TK4)?;
    sop(b, format!("s_lshl_b32 s{DIM2}, s{DIM}, 1"), &[DIM2], &[DIM])?;
    sop(
        b,
        format!("s_mul_i32 s{XSOFF}, s{WGX}, s{DIM}"),
        &[XSOFF],
        &[WGX, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{XSOFF}, s{XSOFF}, 2"),
        &[XSOFF],
        &[XSOFF],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{GROWS}, s{DIM2}"),
        &[BYTES],
        &[GROWS, DIM2],
    )?;
    srd(b, SRD_Y, Y, BYTES)?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{NTOK}, s{DIM}"),
        &[BYTES],
        &[NTOK, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{BYTES}, s{BYTES}, 2"),
        &[BYTES],
        &[BYTES],
    )?;
    srd(b, SRD_X, XR, BYTES)?;
    // Lane byte offsets: y `16 tid + 4096 wg_y`, residual `32 tid + 8192 wg_y`.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{V2_VOFF2}, 4, v{V2_TID}"),
        &[v(V2_VOFF2)],
        &[v(V2_TID)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{V2_VOFF2}, s{WGY}, 12, v{V2_VOFF2}"),
        &[v(V2_VOFF2)],
        &[s(WGY), v(V2_VOFF2)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{V2_VOFF4}, 5, v{V2_TID}"),
        &[v(V2_VOFF4)],
        &[v(V2_TID)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{V2_VOFF4}, s{WGY}, 13, v{V2_VOFF4}"),
        &[v(V2_VOFF4)],
        &[s(WGY), v(V2_VOFF4)],
    )?;
    // acc[0..8] = +0.
    for i in 0..8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", V2_ACC + i),
            &[v(V2_ACC + i)],
            &[],
        )?;
    }
    b.wait_all()?;

    // k = 0..k_top ascending (k_top >= 1: a do-while). Per trip the slot word is always in range of the token's k_top words.
    wg.loop_carried(".Lcombine_v2_trip", (), |wg, ()| {
        let b = wg.isa();
        smem(b, SLOT, 1, INVP, 0)?;
        // Unrouted (negative) or out-of-range slot: skip the weight, the row and the eight fmas.
        let dead = cmp_wg(
            wg,
            format!("s_cmp_ge_u32 s{SLOT}, s{GROWS}"),
            &[s(SLOT), s(GROWS)],
        )?;
        wg.wg_skip_if(dead, ".Lcombine_v2_skip", (), |wg, ()| {
            let b = wg.isa();
            smem(b, WK, 1, WP, 0)?;
            sop(
                b,
                format!("s_mul_i32 s{YOFF}, s{SLOT}, s{DIM2}"),
                &[YOFF],
                &[SLOT, DIM2],
            )?;
            bload128_s(b, V2_YV, V2_VOFF2, SRD_Y, YOFF, 0)?;
            for j in 0..4u8 {
                let pair = V2_YV + j;
                // Odd element first (`& 0xffff0000`), then the even element in place (`<< 16`).
                op(
                    b,
                    format!("v_and_b32_e32 v{V2_TMP}, 0xffff0000, v{pair}"),
                    &[v(V2_TMP)],
                    &[v(pair)],
                )?;
                op(
                    b,
                    format!("v_lshlrev_b32_e32 v{pair}, 16, v{pair}"),
                    &[v(pair)],
                    &[v(pair)],
                )?;
                let (even, odd) = (V2_ACC + 2 * j, V2_ACC + 2 * j + 1);
                op(
                    b,
                    format!("v_fma_f32 v{even}, s{WK}, v{pair}, v{even}"),
                    &[v(even)],
                    &[s(WK), v(pair), v(even)],
                )?;
                op(
                    b,
                    format!("v_fma_f32 v{odd}, s{WK}, v{V2_TMP}, v{odd}"),
                    &[v(odd)],
                    &[s(WK), v(V2_TMP), v(odd)],
                )?;
            }
            b.wait_all()
        })?;
        let b = wg.isa();
        add64_imm(b, INVP, 4)?;
        add64_imm(b, WP, 4)?;
        sop(
            b,
            format!("{} s{KLEFT}, s{KLEFT}, -1", s_add_i32(Arch::Gfx1201)),
            &[KLEFT],
            &[KLEFT],
        )?;
        let more = cmp_wg(wg, format!("s_cmp_gt_i32 s{KLEFT}, 0"), &[s(KLEFT)])?;
        Ok(((), more))
    })?;

    // x_residual = RN(x_residual + acc), eight columns as two 16-byte halves.
    let b = wg.isa();
    bload128_s(b, V2_XV, V2_VOFF4, SRD_X, XSOFF, 0)?;
    bload128_s(b, V2_XV + 4, V2_VOFF4, SRD_X, XSOFF, 16)?;
    for i in 0..8 {
        op(
            b,
            format!(
                "v_add_f32_e32 v{}, v{}, v{}",
                V2_XV + i,
                V2_XV + i,
                V2_ACC + i
            ),
            &[v(V2_XV + i)],
            &[v(V2_XV + i), v(V2_ACC + i)],
        )?;
    }
    bstore128_s(b, V2_XV, V2_VOFF4, SRD_X, XSOFF, 0)?;
    bstore128_s(b, V2_XV + 4, V2_VOFF4, SRD_X, XSOFF, 16)?;
    b.wait_all()?;
    wg.end(end)
}

/// Shape contract of the emitted v2 kernel (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory, arithmetic and control
/// mnemonics, the forbidden families (every v1 scalar-column buffer op included) and the
/// resource limits.
pub fn contract_v2(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b128",
        "s_load_b256",
        "buffer_load_b128",
        "buffer_store_b128",
        "buffer_load_u16",
        "buffer_load_b32",
        "buffer_store_b32",
        "v_fma_f32",
        "v_add_f32_e32",
        "v_and_b32_e32",
        "v_lshlrev_b32_e32",
        "s_barrier_signal",
        "s_barrier_wait",
        "s_endpgm",
    ] {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol_v2(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*",
            "v_mul_f32*", "v_fmac_f32*", "v_mac_f32*", "v_mad_f32*", "v_dual_*", "v_pk_*", "v_cndmask_*"],
        "vgpr_max": V2_V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
