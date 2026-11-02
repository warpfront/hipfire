//! Exact gfx1201 twin of `gemv_mq4g256v2_residual`.
//!
//! Frozen recipe SHA256:
//! `869f4a4bdd33adc3e5a3aa9a6faa7584b53d50b0930f1d9e737e19a77975dc0e`.
//! ABI: A/x/y pointers at 0/8/16, M/K i32 at 24/28; grid M, block 32,
//! no LDS or scratch. Four independent group streams per output row;
//! second-row DOG begins x1 then x0 in both quads and scalar tails.
//! The two-row and odd-row residual epilogues retain distinct add operands.
//! The test-only G0 probe consumes 32 packed 48-byte header/nibble/x records
//! and returns two reduced f32 values per lane.
use crate::insn::{Instruction, MemoryClass};
use crate::reg::Live;
#[cfg(test)]
use crate::reg::{S, V};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

#[cfg(test)]
const G0_SYMBOL: &str = "pm_residual_dog_g0";

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(108, 32)?;
    for (name, base) in [
        ("kernarg", 0),
        ("matrix", 4),
        ("activation", 6),
        ("residual", 8),
    ] {
        regs.s::<2>(name, base, Live::Whole)?;
    }
    regs.s::<4>("weight_resource", 20, Live::Whole)?;
    for base in [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27] {
        regs.add_range(
            &format!("scalar{base}"),
            crate::reg::Kind::S,
            base,
            1,
            Live::Whole,
        )?;
    }
    for (name, base, width) in [
        ("lane", 0, 1),
        ("weight_lane_offset", 1, 1),
        ("packed_offset", 2, 1),
        ("load_offset", 3, 1),
        ("header0", 4, 2),
        ("header1", 6, 2),
        ("x_lo", 8, 4),
        ("x_hi", 12, 4),
        ("packed0", 16, 1),
        ("packed1", 17, 1),
        ("acc", 24, 4),
        ("bcc", 28, 4),
        ("dot", 32, 2),
        ("selected_header", 34, 1),
        ("nibble", 35, 1),
        ("weight", 36, 1),
        ("shuffle_addr", 38, 1),
        ("shuffle_data", 39, 1),
        ("output_offset", 40, 1),
        ("old_y", 41, 2),
    ] {
        regs.add_range(name, crate::reg::Kind::V, base, width, Live::Whole)?;
    }
    for (name, base, width) in [
        ("quad_weights_lo", 18, 2),
        ("quad_weights_hi", 20, 4),
        ("quad_weight_gap", 37, 1),
        ("quad_weight_addresses", 44, 2),
        ("quad_x_address", 46, 1),
        ("quad_x1", 49, 8),
        ("quad_x2", 58, 8),
        ("quad_x3", 67, 8),
        ("quad_headers_lo", 76, 8),
        ("quad_headers_hi", 84, 8),
        ("quad_packed", 92, 8),
        ("quad_dots", 100, 8),
    ] {
        regs.add_range(name, crate::reg::Kind::V, base, width, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "gemv_hfq4g256_residual_mq4v2".into(),
            variant: "exact_four_stream".into(),
            arch: Arch::Gfx1201,
            symbol: "gemv_mq4g256v2_residual".into(),
            kernargs: KernargLayout::new(32)
                .pointer_access("A", 0, crate::plan::Access::ReadOnly)
                .pointer_access("x", 8, crate::plan::Access::ReadOnly)
                .pointer("y", 16)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value"),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    smem(
        &mut b,
        "s_load_b64 s[10:11], s[0:1], 0x18",
        &[10, 11],
        &[0, 1],
    )?;
    salu(&mut b, "s_lshl_b32 s12, ttmp9, 1", &[12], &[])?;
    b.wait_all()?;
    salu(&mut b, "s_cmp_ge_i32 s12, s10", &[], &[12, 10])?;
    branch(&mut b, "s_cbranch_scc1 .Lres_end")?;
    smem(
        &mut b,
        "s_load_b128 s[4:7], s[0:1], 0x0",
        &[4, 5, 6, 7],
        &[0, 1],
    )?;
    smem(&mut b, "s_load_b64 s[8:9], s[0:1], 0x10", &[8, 9], &[0, 1])?;
    b.wait_all()?;
    salu(&mut b, "s_add_co_i32 s13, s12, 1", &[13], &[12])?;
    salu(&mut b, "s_cmp_lt_i32 s13, s10", &[], &[13, 10])?;
    salu(&mut b, "s_cselect_b32 s18, 1, 0", &[18], &[])?;
    salu(&mut b, "s_cselect_b32 s13, s13, s12", &[13], &[13, 12])?;
    salu(&mut b, "s_lshr_b32 s14, s11, 8", &[14], &[11])?;
    salu(&mut b, "s_mul_i32 s19, s14, 0x88", &[19], &[14])?;
    salu(&mut b, "s_mul_i32 s16, s19, s12", &[16], &[19, 12])?;
    salu(&mut b, "s_mul_i32 s17, s19, s13", &[17], &[19, 13])?;
    salu(&mut b, "s_lshr_b32 s2, s14, 2", &[2], &[14])?;
    salu(&mut b, "s_mov_b32 s15, 0", &[15], &[])?;
    salu(&mut b, "s_mov_b32 s20, s4", &[20], &[4])?;
    salu(&mut b, "s_and_b32 s21, s5, 0xffff", &[21], &[5])?;
    salu(&mut b, "s_mov_b32 s22, -1", &[22], &[])?;
    salu(&mut b, "s_mov_b32 s23, 0x31004000", &[23], &[])?;
    valu(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[1], &[0], &[])?;
    for r in 24..32 {
        valu(&mut b, &format!("v_mov_b32_e32 v{r}, 0"), &[r], &[], &[])?;
    }
    salu(&mut b, "s_cmp_eq_u32 s2, 0", &[], &[2])?;
    branch(&mut b, "s_cbranch_scc1 .Lres_tail")?;
    b.loop_(".Lres_quad", |b| {
        quad(b)?;
        salu(b, "s_add_co_i32 s2, s2, -1", &[2], &[2])?;
        salu(b, "s_cmp_lg_u32 s2, 0", &[], &[2])?;
        branch(b, "s_cbranch_scc1 .Lres_quad")
    })?;
    b.label(".Lres_tail")?;
    for stream in 0..3 {
        salu(&mut b, "s_cmp_lt_u32 s15, s14", &[], &[15, 14])?;
        branch(&mut b, "s_cbranch_scc0 .Lres_fold")?;
        group(&mut b, stream)?;
    }
    b.label(".Lres_fold")?;
    for base in [24u8, 28] {
        valu(
            &mut b,
            &format!("v_add_f32_e32 v32, v{base}, v{}", base + 1),
            &[32],
            &[base, base + 1],
            &[],
        )?;
        valu(
            &mut b,
            &format!("v_add_f32_e32 v33, v{}, v{}", base + 2, base + 3),
            &[33],
            &[base + 2, base + 3],
            &[],
        )?;
        valu(
            &mut b,
            &format!("v_add_f32_e32 v{base}, v32, v33"),
            &[base],
            &[32, 33],
            &[],
        )?;
        reduce_full(&mut b, base)?;
    }
    valu(&mut b, "v_cmpx_eq_u32_e32 0, v0", &[], &[0], &[])?;
    branch(&mut b, "s_cbranch_execz .Lres_end")?;
    salu(&mut b, "s_lshl_b32 s27, s12, 2", &[27], &[12])?;
    valu(&mut b, "v_mov_b32_e32 v40, s27", &[40], &[], &[27])?;
    salu(&mut b, "s_cmp_eq_u32 s18, 0", &[], &[18])?;
    branch(&mut b, "s_cbranch_scc1 .Lres_single")?;
    vmem(
        &mut b,
        "global_load_b64 v[41:42], v40, s[8:9]",
        &[41, 42],
        &[40],
        &[8, 9],
        false,
    )?;
    b.wait_all()?;
    // Incumbent two-row epilogue adds acc+y, bcc+y (not y+acc).
    valu(&mut b, "v_add_f32_e32 v41, v24, v41", &[41], &[24, 41], &[])?;
    valu(&mut b, "v_add_f32_e32 v42, v28, v42", &[42], &[28, 42], &[])?;
    vmem(
        &mut b,
        "global_store_b64 v40, v[41:42], s[8:9]",
        &[],
        &[40, 41, 42],
        &[8, 9],
        true,
    )?;
    b.wait_all()?;
    branch(&mut b, "s_branch .Lres_end")?;
    b.label(".Lres_single")?;
    vmem(
        &mut b,
        "global_load_b32 v41, v40, s[8:9]",
        &[41],
        &[40],
        &[8, 9],
        false,
    )?;
    b.wait_all()?;
    valu(&mut b, "v_add_f32_e32 v41, v41, v24", &[41], &[41, 24], &[])?;
    vmem(
        &mut b,
        "global_store_b32 v40, v41, s[8:9]",
        &[],
        &[40, 41],
        &[8, 9],
        true,
    )?;
    b.wait_all()?;
    b.label(".Lres_end")?;
    branch(&mut b, "s_endpgm")?;
    Ok(vec![b.finish()?])
}

fn refs(kind: crate::reg::Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(
        text,
        refs(crate::reg::Kind::S, defs),
        refs(crate::reg::Kind::S, uses),
    ))
}
fn valu(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
) -> Result<(), String> {
    let mut uses = refs(crate::reg::Kind::V, uses);
    uses.extend(refs(crate::reg::Kind::S, scalar));
    b.push(Instruction::new(
        text,
        refs(crate::reg::Kind::V, defs),
        uses,
    ))
}
fn smem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(
        Instruction::new(
            text,
            refs(crate::reg::Kind::S, defs),
            refs(crate::reg::Kind::S, uses),
        )
        .memory(MemoryClass::SmemLoad),
    )
}
fn vmem(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
    store: bool,
) -> Result<(), String> {
    let mut uses = refs(crate::reg::Kind::V, uses);
    uses.extend(refs(crate::reg::Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(crate::reg::Kind::V, defs), uses).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}

// Four independent groups, two rows each. Rotating the x banks allows VOPD
// without exchanging even the multiplication operands in the incumbent DAG.
fn quad(b: &mut Builder) -> Result<(), String> {
    use crate::ledger::Counter;
    use crate::vopd::{Operand, VopdF32, VopdOp};
    const X: [u8; 4] = [8, 49, 58, 67];
    salu(b, "s_mul_i32 s24, s15, 0x88", &[24], &[15])?;
    salu(b, "s_add_co_i32 s25, s16, s24", &[25], &[16, 24])?;
    salu(b, "s_add_co_i32 s26, s17, s24", &[26], &[17, 24])?;
    salu(b, "s_lshl_b32 s27, s15, 10", &[27], &[15])?;
    valu(b, "v_mov_b32_e32 v3, s25", &[3], &[], &[25])?;
    valu(b, "v_add_nc_u32_e32 v2, s25, v1", &[2], &[1], &[25])?;
    valu(b, "v_mov_b32_e32 v45, s26", &[45], &[], &[26])?;
    valu(b, "v_add_nc_u32_e32 v44, s26, v1", &[44], &[1], &[26])?;
    valu(b, "v_lshlrev_b32_e32 v46, 5, v0", &[46], &[0], &[])?;
    valu(b, "v_add_nc_u32_e32 v46, s27, v46", &[46], &[46], &[27])?;
    // Incumbent first batch: sixteen weight loads and three lower x vectors.
    // Keep individual issues visible to the partial-retirement proof rather
    // than grouping them with an optional clause hint.
    for i in [0u8, 2, 3, 4, 5, 7] {
        quad_packed(b, i)?;
    }
    for i in [0u8, 2, 4, 6, 1, 3, 5, 7] {
        let h = 76 + 2 * i;
        let addr = if i % 2 == 0 { 3 } else { 45 };
        let off = u32::from(i / 2) * 136;
        let suffix = if off == 0 {
            String::new()
        } else {
            format!(" offset:{off}")
        };
        vmem(
            b,
            &format!(
                "buffer_load_b64 v[{h}:{}], v{addr}, s[20:23], null offen{suffix} scope:SCOPE_DEV",
                h + 1
            ),
            &[h, h + 1],
            &[addr],
            &[20, 21, 22, 23],
            false,
        )?;
    }
    quad_packed(b, 1)?;
    quad_packed(b, 6)?;
    for g in 0..3 {
        quad_x(b, X[g], g as u32 * 1024)?;
    }
    valu(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[0], &[])?;
    b.wait(Counter::Load, 18)?;
    quad_nibble(b, 0, 0, 16)?;
    b.wait(Counter::Load, 17)?;
    quad_nibble(b, 2, 0, 18)?;
    for (count, i) in [(12, 0u8), (11, 2), (10, 4), (9, 6), (7, 3)] {
        b.wait(Counter::Load, count)?;
        quad_header(b, i)?;
    }
    b.wait(Counter::Load, 1)?;
    for i in [1, 5, 7] {
        quad_header(b, i)?;
    }
    for i in [1u8, 3, 4, 5, 6, 7] {
        quad_nibble(b, i, 0, 16 + i)?;
    }
    b.wait(Counter::Load, 0)?;
    // The remaining five x loads overlap all eight independent dequant chains.
    quad_x(b, X[3], 3072)?;
    for (g, x) in X.into_iter().enumerate() {
        quad_x(b, x + 4, g as u32 * 1024 + 16)?;
    }
    for i in 0..8 {
        quad_mix(b, i, 16 + i)?;
    }
    for i in 0..8 {
        quad_nibble(b, i, 1, 32 + i)?;
    }
    for i in 0..8 {
        quad_mix(b, i, 32 + i)?;
    }
    for row in 0..2u8 {
        for g in [0u8, 2] {
            if row == 0 && g == 2 {
                b.wait(Counter::Load, 4)?;
            }
            let op = |g: u8| {
                let i = 2 * g + row;
                VopdOp {
                    op: VopdF32::Mul,
                    dst: 100 + 4 * row + g,
                    src0: Operand::V(X[g as usize] + row),
                    src1: if row == 0 { 16 + i } else { 32 + i },
                }
            };
            b.vopd(op(g), op(g + 1))?;
        }
    }
    for row in 0..2u8 {
        for g in [0u8, 2] {
            let op = |g: u8| {
                let i = 2 * g + row;
                VopdOp {
                    op: VopdF32::Fmac,
                    dst: 100 + 4 * row + g,
                    src0: Operand::V(X[g as usize] + 1 - row),
                    src1: if row == 0 { 32 + i } else { 16 + i },
                }
            };
            b.vopd(op(g), op(g + 1))?;
        }
    }
    for nibble in 2..8u8 {
        // Upper vectors arrive progressively rather than four full drains.
        for i in 0..8u8 {
            if nibble == 4 && i % 2 == 0 {
                b.wait(Counter::Load, 3 - i / 2)?;
            }
            let p = 92 + i;
            let w = 16 + i;
            valu(
                b,
                &format!("v_bfe_u32 v{w}, v{p}, {}, 4", nibble * 4),
                &[w],
                &[p],
                &[],
            )?;
        }
        for w in 16..24 {
            valu(
                b,
                &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"),
                &[w],
                &[w],
                &[],
            )?;
        }
        for i in 0..8 {
            quad_mix(b, i, 16 + i)?;
        }
        for row in 0..2u8 {
            for g in [0u8, 2] {
                let op = |g: u8| VopdOp {
                    op: VopdF32::Fmac,
                    dst: 100 + 4 * row + g,
                    src0: Operand::V(X[g as usize] + nibble),
                    src1: 16 + 2 * g + row,
                };
                b.vopd(op(g), op(g + 1))?;
            }
        }
    }
    for row in 0..2u8 {
        for g in [0u8, 2] {
            let op = |g: u8| {
                let acc = 24 + 4 * row + g;
                let dot = 100 + 4 * row + g;
                VopdOp {
                    op: VopdF32::Add,
                    dst: acc,
                    src0: Operand::V(if row == 0 { dot } else { acc }),
                    src1: if row == 0 { acc } else { dot },
                }
            };
            b.vopd(op(g), op(g + 1))?;
        }
    }
    salu(b, "s_add_co_i32 s15, s15, 4", &[15], &[15])?;
    b.wait_all()
}

fn quad_packed(b: &mut Builder, i: u8) -> Result<(), String> {
    let p = 92 + i;
    let addr = if i % 2 == 0 { 2 } else { 44 };
    let off = u32::from(i / 2) * 136 + 8;
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{p}, v{addr}, s[20:23], null offen offset:{off} scope:SCOPE_DEV"
        ),
        &[p],
        &[addr],
        &[20, 21, 22, 23],
        false,
    )
}
fn quad_x(b: &mut Builder, x: u8, off: u32) -> Result<(), String> {
    let suffix = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    vmem(
        b,
        &format!("global_load_b128 v[{x}:{}], v46, s[6:7]{suffix}", x + 3),
        &[x, x + 1, x + 2, x + 3],
        &[46],
        &[6, 7],
        false,
    )
}
fn quad_header(b: &mut Builder, i: u8) -> Result<(), String> {
    let h = 76 + 2 * i;
    valu(
        b,
        &format!("v_cndmask_b32_e32 v{h}, v{}, v{h}, vcc_lo", h + 1),
        &[h],
        &[h, h + 1],
        &[],
    )
}
fn quad_nibble(b: &mut Builder, i: u8, nibble: u8, w: u8) -> Result<(), String> {
    let p = 92 + i;
    valu(
        b,
        &format!("v_bfe_u32 v{w}, v{p}, {}, 4", nibble * 4),
        &[w],
        &[p],
        &[],
    )?;
    valu(
        b,
        &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"),
        &[w],
        &[w],
        &[],
    )
}
fn quad_mix(b: &mut Builder, i: u8, w: u8) -> Result<(), String> {
    let h = 76 + 2 * i;
    valu(
        b,
        &format!("v_fma_mix_f32 v{w}, v{h}, v{w}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
        &[w],
        &[h, w],
        &[],
    )
}

fn group(b: &mut Builder, stream: u8) -> Result<(), String> {
    salu(b, "s_mul_i32 s24, s15, 0x88", &[24], &[15])?;
    salu(b, "s_add_co_i32 s25, s16, s24", &[25], &[16, 24])?;
    salu(b, "s_add_co_i32 s26, s17, s24", &[26], &[17, 24])?;
    salu(b, "s_lshl_b32 s27, s15, 10", &[27], &[15])?;
    valu(b, "v_lshlrev_b32_e32 v3, 5, v0", &[3], &[0], &[])?;
    valu(b, "v_add_nc_u32_e32 v3, s27, v3", &[3], &[3], &[27])?;
    vmem(
        b,
        "global_load_b128 v[8:11], v3, s[6:7]",
        &[8, 9, 10, 11],
        &[3],
        &[6, 7],
        false,
    )?;
    vmem(
        b,
        "global_load_b128 v[12:15], v3, s[6:7] offset:16",
        &[12, 13, 14, 15],
        &[3],
        &[6, 7],
        false,
    )?;
    b.wait_all()?;
    for row in 0..2u8 {
        let header = 4 + row * 2;
        let packed = 16 + row;
        let scalar = 25 + row;
        valu(
            b,
            &format!("v_mov_b32_e32 v3, s{scalar}"),
            &[3],
            &[],
            &[scalar],
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_b64 v[{header}:{}], v3, s[20:23], null offen scope:SCOPE_DEV",
                header + 1
            ),
            &[header, header + 1],
            &[3],
            &[20, 21, 22, 23],
            false,
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v2, s{scalar}, v1"),
            &[2],
            &[1],
            &[scalar],
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{packed}, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV"
            ),
            &[packed],
            &[2],
            &[20, 21, 22, 23],
            false,
        )?;
        b.wait_all()?;
        valu(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[0], &[])?;
        valu(
            b,
            &format!("v_cndmask_b32_e32 v34, v{}, v{header}, vcc_lo", header + 1),
            &[34],
            &[header, header + 1],
            &[],
        )?;
        let dot = 32 + row;
        // Tail row1 starts x1 at 0x216c/0x23dc/0x2660, then x0 FMAC.
        let order = if row == 1 {
            [1, 0, 2, 3, 4, 5, 6, 7]
        } else {
            [0, 1, 2, 3, 4, 5, 6, 7]
        };
        for (step, nibble) in order.into_iter().enumerate() {
            valu(
                b,
                &format!("v_bfe_u32 v35, v{packed}, {}, 4", nibble * 4),
                &[35],
                &[packed],
                &[],
            )?;
            valu(b, "v_cvt_f32_ubyte0_e32 v35, v35", &[35], &[35], &[])?;
            valu(
                b,
                "v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]",
                &[36],
                &[34, 35],
                &[],
            )?;
            let x = 8 + nibble;
            if step == 0 {
                valu(
                    b,
                    &format!("v_mul_f32_e32 v{dot}, v{x}, v36"),
                    &[dot],
                    &[x, 36],
                    &[],
                )?;
            } else {
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{dot}, v{x}, v36"),
                    &[dot],
                    &[dot, x, 36],
                    &[],
                )?;
            }
        }
        let acc = 24 + row * 4 + stream;
        let (a, c) = if row == 0 { (dot, acc) } else { (acc, dot) };
        valu(
            b,
            &format!("v_add_f32_e32 v{acc}, v{a}, v{c}"),
            &[acc],
            &[a, c],
            &[],
        )?;
    }
    salu(b, "s_add_co_i32 s15, s15, 1", &[15], &[15])?;
    b.wait_all()
}

fn reduce_full(b: &mut Builder, result: u8) -> Result<(), String> {
    b.ds_crosslane(
        Instruction::new(
            format!("ds_swizzle_b32 v39, v{result} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            refs(crate::reg::Kind::V, &[39]),
            refs(crate::reg::Kind::V, &[result]),
        )
        .memory(MemoryClass::DsLoad),
    )?;
    b.wait(crate::ledger::Counter::Ds, 0)?;
    valu(
        b,
        &format!("v_add_f32_e32 v{result}, v{result}, v39"),
        &[result],
        &[result, 39],
        &[],
    )?;
    for offset in [8, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
            &[],
            &[0],
            &[],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v38, 0, {offset}, vcc_lo"),
            &[38],
            &[],
            &[],
        )?;
        valu(b, "v_add_lshl_u32 v38, v38, v0, 2", &[38], &[38, 0], &[])?;
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v39, v38, v{result}"),
                refs(crate::reg::Kind::V, &[39]),
                refs(crate::reg::Kind::V, &[38, result]),
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        valu(
            b,
            &format!("v_add_f32_e32 v{result}, v{result}, v39"),
            &[result],
            &[result, 39],
            &[],
        )?;
    }
    Ok(())
}

#[cfg(test)]
fn build_dog_g0() -> Result<Emitted, String> {
    let mut regs = RegPlan::new(32, 8)?;
    regs.s::<2>("kernarg", 0, Live::Whole)?;
    regs.s::<2>("input", 4, Live::Whole)?;
    regs.s::<2>("output", 6, Live::Whole)?;
    regs.v::<1>("lane", 0, Live::Whole)?;
    regs.v::<1>("input_offset", 1, Live::Whole)?;
    regs.v::<1>("output_offset", 2, Live::Whole)?;
    for (name, base) in [
        ("headers", 4),
        ("x_lo", 8),
        ("x_hi", 12),
        ("weights_lo", 16),
        ("weights_hi", 20),
    ] {
        regs.v::<4>(name, base, Live::Whole)?;
    }
    regs.v::<2>("results", 24, Live::Whole)?;
    regs.v::<1>("shuffle_address", 26, Live::Whole)?;
    regs.v::<1>("shuffle_result", 27, Live::Whole)?;
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "gemv_hfq4g256_residual_mq4v2".into(),
            variant: "dog_g0_only".into(),
            arch: Arch::Gfx1201,
            symbol: G0_SYMBOL.into(),
            kernargs: KernargLayout::new(16)
                .pointer("records", 0)
                .pointer("out", 8),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.push(
        Instruction::new(
            "s_load_b128 s[4:7], s[0:1], 0",
            vec![S::<2>(4).reg(), S::<2>(6).reg()],
            vec![S::<2>(0).reg()],
        )
        .memory(MemoryClass::SmemLoad),
    )?;
    b.push(Instruction::new(
        "v_mul_lo_u32 v1, 48, v0",
        vec![V::<1>(1).reg()],
        vec![V::<1>(0).reg()],
    ))?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v2, 3, v0",
        vec![V::<1>(2).reg()],
        vec![V::<1>(0).reg()],
    ))?;
    for (dst, offset) in [(4, 0), (8, 16), (12, 32)] {
        b.push(
            Instruction::new(
                format!(
                    "global_load_b128 v[{}:{}], v1, s[4:5] offset:{}",
                    dst,
                    dst + 3,
                    offset
                ),
                vec![V::<4>(dst).reg()],
                vec![V::<1>(1).reg(), S::<2>(4).reg()],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    // The row-0 chain starts x0 then x1. The row-1 chain starts x1 then x0
    // in the frozen clang24 DAG (0x1d98/0x1db4), not the source's apparent order.
    for row in 0..2u8 {
        let header = 4 + row * 2;
        let packed = header + 1;
        for nibble in 0..8u8 {
            let dst = 16 + nibble;
            b.push(Instruction::new(
                format!("v_bfe_u32 v{dst}, v{packed}, {}, 4", nibble * 4),
                vec![V::<1>(dst).reg()],
                vec![V::<1>(packed).reg()],
            ))?;
            b.push(Instruction::new(
                format!("v_cvt_f32_ubyte0_e32 v{dst}, v{dst}"),
                vec![V::<1>(dst).reg()],
                vec![V::<1>(dst).reg()],
            ))?;
            b.push(Instruction::new(format!("v_fma_mix_f32 v{dst}, v{header}, v{dst}, v{header} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),vec![V::<1>(dst).reg()],vec![V::<1>(header).reg(),V::<1>(dst).reg()]))?;
        }
        let result = 24 + row;
        let order = if row == 0 {
            [0, 1, 2, 3, 4, 5, 6, 7]
        } else {
            [1, 0, 2, 3, 4, 5, 6, 7]
        };
        for (step, nibble) in order.into_iter().enumerate() {
            let x = 8 + nibble;
            let weight = 16 + nibble;
            let (mnemonic, uses) = if step == 0 {
                ("v_mul_f32_e32", vec![V::<1>(x).reg(), V::<1>(weight).reg()])
            } else {
                (
                    "v_fmac_f32_e32",
                    vec![V::<1>(x).reg(), V::<1>(weight).reg(), V::<1>(result).reg()],
                )
            };
            b.push(Instruction::new(
                format!("{mnemonic} v{result}, v{x}, v{weight}"),
                vec![V::<1>(result).reg()],
                uses,
            ))?;
        }
    }
    // Frozen shfl-down lowering: force bit 4 for offset 16; for subsequent
    // steps an out-of-range source lane reads itself, not a wrapped lane.
    for result in [24u8, 25] {
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v27, v{result} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                vec![V::<1>(27).reg()],
                vec![V::<1>(result).reg()],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        b.wait(crate::ledger::Counter::Ds, 0)?;
        b.push(Instruction::new(
            format!("v_add_f32_e32 v{result}, v{result}, v27"),
            vec![V::<1>(result).reg()],
            vec![V::<1>(result).reg(), V::<1>(27).reg()],
        ))?;
        for offset in [8u8, 4, 2, 1] {
            b.push(Instruction::new(
                format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
                vec![],
                vec![V::<1>(0).reg()],
            ))?;
            b.push(Instruction::new(
                format!("v_cndmask_b32_e64 v26, 0, {offset}, vcc_lo"),
                vec![V::<1>(26).reg()],
                vec![],
            ))?;
            b.push(Instruction::new(
                "v_add_lshl_u32 v26, v26, v0, 2",
                vec![V::<1>(26).reg()],
                vec![V::<1>(26).reg(), V::<1>(0).reg()],
            ))?;
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v27, v26, v{result}"),
                    vec![V::<1>(27).reg()],
                    vec![V::<1>(26).reg(), V::<1>(result).reg()],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            b.wait(crate::ledger::Counter::Ds, 0)?;
            b.push(Instruction::new(
                format!("v_add_f32_e32 v{result}, v{result}, v27"),
                vec![V::<1>(result).reg()],
                vec![V::<1>(result).reg(), V::<1>(27).reg()],
            ))?;
        }
    }
    b.push(
        Instruction::new(
            "global_store_b64 v2, v[24:25], s[6:7]",
            vec![],
            vec![V::<1>(2).reg(), V::<2>(24).reg(), S::<2>(6).reg()],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    b.wait_all()?;
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
}

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;
    #[test]
    fn residual_second_row_tail_dag() {
        let emitted = build_gfx1201().expect("full residual twin").remove(0);
        // Explicit tails retain the incumbent row1 x1 -> x0 seed order.
        assert_eq!(
            emitted.s_text.matches("v_mul_f32_e32 v33, v9, v36").count(),
            3
        );
        assert_eq!(
            emitted
                .s_text
                .matches("v_fmac_f32_e32 v33, v8, v36")
                .count(),
            3
        );
        assert!(emitted
            .s_text
            .contains("v_dual_mul_f32 v104, v9, v33 :: v_dual_mul_f32 v105, v50, v35"));
        assert!(emitted
            .s_text
            .contains("v_dual_fmac_f32 v104, v8, v17 :: v_dual_fmac_f32 v105, v49, v19"));
        assert!(!emitted.s_text.contains("v_mul_f32_e32 v33, v8, v36"));
        assert!(emitted.s_text.contains("v_add_f32_e32 v41, v24, v41"));
        assert!(emitted.s_text.contains("v_add_f32_e32 v41, v41, v24"));
        assert_eq!(emitted.proof.vopd_pairs, 36);
        assert_eq!(emitted.proof.next_free_vgpr, 108);
        let quad = emitted
            .s_text
            .split(".Lres_quad:\n")
            .nth(1)
            .expect("quad label")
            .split(".Lres_tail:\n")
            .next()
            .expect("tail label");
        assert_eq!(quad.matches("s_wait_loadcnt").count(), 14);
        let first = quad
            .split("s_wait_loadcnt")
            .next()
            .expect("first load batch");
        assert_eq!(first.matches("buffer_load_").count(), 16);
        assert_eq!(first.matches("global_load_").count(), 3);
        assert_eq!(quad.matches("global_load_").count(), 8);
        crate::native::assemble(&emitted.s_text, Arch::Gfx1201).expect("native full twin");
    }
    #[test]
    fn residual_dog_g0_m7() {
        let emitted = build_dog_g0().expect("checked DOG region");
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-g0-residual-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("gemv_hfq4g256_residual_mq4v2.g0.s");
        let elf = dir.join("gemv_hfq4g256_residual_mq4v2.g0.co");
        std::fs::write(&source, &emitted.s_text).unwrap();
        std::fs::write(
            &elf,
            crate::native::assemble(&emitted.s_text, Arch::Gfx1201).expect("native assemble"),
        )
        .unwrap();
        let report = crate::pm_check::m7(&elf, "gfx1201", G0_SYMBOL).expect("M7");
        println!("{}\n{}", source.display(), report);
        assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
