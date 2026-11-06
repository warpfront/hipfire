// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `s_bfm_b32` receipt probe (gfx1201): a checked-Builder kernel, its input set and its host law.
//!
//! Include with `#[path = "support/s_bfm_probe.rs"] mod s_bfm_probe;`. The probe is assembled separately by the
//! host oracle; it is not part of any shipping bundle.
//!
//! ABI (16-byte kernarg): `inputs: *const [u32; 3]` at 0, `outputs: *mut [u32; 2]` at 8. Workgroup `i`
//! (`ttmp9`, grid length = `inputs().len()` workgroups of 32 lanes) reads record `i` = `[width, offset, scc]`,
//! sets SCC = (`scc != 0`) with `s_cmp_lg_u32`, executes the scalar `s_bfm_b32 width, offset`, captures SCC with
//! `s_cselect_b32` immediately after, and lane 0 (EXEC = 1, so no identical-lane store race) writes
//! `[result, scc_after]` to `outputs[i]`.
#![allow(dead_code)]
use peacemaker_isa::{
    insn::{Instruction, MemoryClass},
    kernels::common::{mem, op, s, smem, sop, sr, v},
    plan::Access,
    reg::{Kind, Live},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};

pub const SYMBOL: &str = "s_bfm_b32_probe_gfx1201";
/// Threads per workgroup (one wave32).
pub const BLOCK: u32 = 32;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 16;

// SGPR map (gfx12: user s[0:1] = kernarg pointer, workgroup id in `ttmp9`).
const KARG: u8 = 0;
const IN_PTR: u8 = 8; // s[8:9]; s_load_b128 s[8:11] = inputs, outputs
const OUT_PTR: u8 = 10; // s[10:11]
const WG: u8 = 12;
const IN_OFF: u8 = 13;
const REC: u8 = 14; // s[14:15] record address
const WIDTH: u8 = 16; // s[16:17] = width, offset (s_load_b64)
const OFFSET: u8 = 17;
const SCC_IN: u8 = 18;
const RESULT: u8 = 19;
const SCC_OUT: u8 = 20;
const OUT_OFF: u8 = 21;
const OUT_ADDR: u8 = 22; // s[22:23]
// VGPRs: v0 = 0 offset, v[2:3] = result, scc.
const V_ZERO: u8 = 0;
const V_DATA: u8 = 2;

/// `s_add_co_u32 lo, a, b; s_add_co_ci_u32 hi, a+1, 0`: 64-bit add with a proper carry.
fn add64(b: &mut Builder, dst: u8, base: u8, x: u8) -> Result<(), String> {
    sop(b, format!("s_add_co_u32 s{dst}, s{base}, s{x}"), &[dst], &[base, x])?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{}, s{}, 0", dst + 1, base + 1),
        &[dst + 1],
        &[base + 1],
    )
}

/// The probe kernel (gfx1201).
pub fn emit_probe() -> Result<Emitted, String> {
    let arch = Arch::Gfx1201;
    let mut regs = RegPlan::new(8, 104)?;
    regs.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    regs.add_range("s_blk8", Kind::S, 8, 8, Live::Whole)?;
    regs.add_range("s_blk16", Kind::S, 16, 8, Live::Whole)?;
    regs.add_range("v_blk0", Kind::V, 0, 8, Live::Whole)?;
    let kspec = KernelSpec {
        kernel_id: "s_bfm_b32_probe".into(),
        variant: "gfx1201".into(),
        arch,
        symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(KERNARG_BYTES)
            .pointer_access("inputs", 0, Access::ReadOnly)
            .pointer_access("outputs", 8, Access::WriteOnly),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut builder = Builder::new(kspec, regs);
    let b = &mut builder;
    smem(b, IN_PTR, 4, KARG, 0)?;
    sop(b, format!("s_mov_b32 s{WG}, ttmp9"), &[WG], &[])?;
    sop(b, format!("s_mul_i32 s{IN_OFF}, s{WG}, 12"), &[IN_OFF], &[WG])?;
    add64(b, REC, IN_PTR, IN_OFF)?;
    smem(b, WIDTH, 2, REC, 0)?;
    smem(b, SCC_IN, 1, REC, 8)?;
    // Establish SCC from input[2], run the instruction under test, capture SCC before anything else touches it.
    sop(b, format!("s_cmp_lg_u32 s{SCC_IN}, 0"), &[], &[SCC_IN])?;
    sop(
        b,
        format!("s_bfm_b32 s{RESULT}, s{WIDTH}, s{OFFSET}"),
        &[RESULT],
        &[WIDTH, OFFSET],
    )?;
    sop(b, format!("s_cselect_b32 s{SCC_OUT}, 1, 0"), &[SCC_OUT], &[])?;
    sop(b, format!("s_lshl_b32 s{OUT_OFF}, s{WG}, 3"), &[OUT_OFF], &[WG])?;
    add64(b, OUT_ADDR, OUT_PTR, OUT_OFF)?;
    // Lane 0 only.
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    op(b, format!("v_mov_b32_e32 v{V_ZERO}, 0"), &[v(V_ZERO)], &[])?;
    op(b, format!("v_mov_b32_e32 v{V_DATA}, s{RESULT}"), &[v(V_DATA)], &[s(RESULT)])?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{SCC_OUT}", V_DATA + 1),
        &[v(V_DATA + 1)],
        &[s(SCC_OUT)],
    )?;
    mem(
        b,
        format!(
            "global_store_b64 v{V_ZERO}, v[{V_DATA}:{}], s[{OUT_ADDR}:{}]",
            V_DATA + 1,
            OUT_ADDR + 1
        ),
        &[],
        &[v(V_ZERO), sr(V_DATA, 2), sr(OUT_ADDR, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    builder.finish()
}

fn xorshift(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

/// `[width, offset, initial_scc]` records: for every `(width, offset)` of the low five bits and `scc` in `{0, 1}`,
/// four upper-bit variants (zero, all ones, two fixed-seed random draws), 8192 records in all.
pub fn inputs() -> Vec<[u32; 3]> {
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = Vec::with_capacity(4 * 32 * 32 * 2);
    for variant in 0..4 {
        for width in 0..32u32 {
            for offset in 0..32u32 {
                for scc in 0..2u32 {
                    let (hw, ho) = match variant {
                        0 => (0, 0),
                        1 => (!31, !31),
                        _ => (xorshift(&mut rng) & !31, xorshift(&mut rng) & !31),
                    };
                    out.push([hw | width, ho | offset, scc]);
                }
            }
        }
    }
    out
}

/// Host law: `mask = ((1 << w) - 1) << o` over 32 bits with `w = input[0] & 31`, `o = input[1] & 31`;
/// SCC after = SCC before = `input[2] != 0`. Returns `[result, scc]`.
pub fn expected(input: [u32; 3]) -> [u32; 2] {
    let width = input[0] & 31;
    let mask = if width == 0 { 0 } else { u32::MAX >> (32 - width) };
    [mask.wrapping_shl(input[1]), u32::from(input[2] != 0)]
}
