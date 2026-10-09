//! Exact gfx1201 multi-column twin of `gemv_mq4g256v2_xbatch` (plain `y = W.x`).
//!
//! `gemv_mq4g256v2_xbatch_pm(A, x[B][K], y[B][M], M, K, B)`, B in 1..=8: for
//! every column `b`, `y[b]` is byte-identical to `gemv_mq4g256v2_xbatch` (and
//! so the singleton per-row DAG) on `x[b]`.
//! ABI: A/x/y pointers at 0/8/16, M/K/B i32 at 24/28/32 (36 B); the x-batch
//! launcher's grid M, block 32, no LDS or scratch. x is `[B][K]` and y is
//! `[B][M]`, both row-major f32, K % 256 == 0.
//!
//! Workgroup `w` owns the four rows `4w..4w+3`; workgroups with `4w >= M`
//! exit at once (the launcher's grid stays M). Rows past M are clamped to
//! M-1 for loads and never stored. Four rows share every activation load,
//! which is what bounds the incumbent (it re-reads x per row).
//!
//! Each 136-byte group of the four rows is loaded once (lane-selected half
//! header, packed nibbles) and dequantized once with the incumbent's
//! `fma_mix(sc, nibble, zp)`; the nibble values come from byte conversions
//! of `pk & 0x0f0f0f0f` / `(pk >> 4) & 0x0f0f0f0f` (the same integers as the
//! incumbent's bit-field extracts). Per column and row the incumbent clang
//! DAG is kept exactly: group `g` accumulates into stream `g % 4` in
//! increasing `g` (quads and scalar tails alike), each group dot is
//! `x1*w1` then fmac `x0*w0`, `x2*w2` .. `x7*w7`, the stream add is
//! `acc + dot`, the fold is `(a0 + a1) + (a2 + a3)`, and the reduction is the
//! frozen shfl_down tree (swizzle `1pppp`, then self-reading bpermutes
//! 8/4/2/1, each `own + other`). Row pairs (0,1) and (2,3) of one column are
//! VOPD packets sharing x as the gfx12 shared src1. Activations stream
//! through a register ring so later columns load while earlier ones compute.
//! B is specialized: one code path per column count, selected once.
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Kind, Live};
use crate::vopd::{Operand, VopdF32, VopdOp};

pub const SYMBOL: &str = "gemv_mq4g256v2_xbatch_pm";
pub const MAX_COLUMNS: u8 = 8;

// VGPRs.
const LANE: u8 = 0;
/// `(lane >> 4) * 4`: the lane's half-header byte offset.
const HDR_OFF: u8 = 1;
/// `lane * 4`: the lane's packed-nibble byte offset (plus 8 immediate).
const PK_OFF: u8 = 2;
/// `lane * 32`: the lane's eight activations.
const X_OFF: u8 = 3;
const SHUF_ADDR: u8 = 4;
/// Half header of (group slot j, row r).
fn hdr(j: u8, r: u8) -> u8 { 8 + 4 * j + r }
/// Packed nibbles of (group slot j, row r).
fn pk(j: u8, r: u8) -> u8 { 24 + 4 * j + r }
/// Dequantized weights of row r. Row pairs differ by 2 mod 4 so every VOPD
/// pair has distinct src0 banks.
const W: [u8; 4] = [40, 50, 60, 70];
/// Dot registers (column slot, row); even rows even, odd rows odd.
const DOT: [[u8; 4]; 2] = [[48, 49, 58, 59], [68, 69, 78, 79]];
const RING: u16 = 80;
const VGPRS: u16 = 256;
/// Activation ring slots (8 VGPRs each) of the `nb`-column path: every free
/// register after the accumulators, at most one quad of items, and few enough
/// that 32 weight loads plus the ring stay inside the 6-bit load counter.
fn slots(nb: u8) -> u16 { (4 * u16::from(nb)).min((VGPRS - RING - 16 * u16::from(nb)) / 8).min(15) }
fn ring(nb: u8, item: u16) -> u8 { (RING + 8 * (item % slots(nb))) as u8 }
/// Stream accumulator (column c, stream s, row r) of the `nb`-column path.
fn acc(nb: u8, c: u8, s: u8, r: u8) -> u8 { (RING + 8 * slots(nb)) as u8 + 16 * c + 4 * s + r }
// Epilogue registers (headers/nibbles and row-0 weights are dead by then).
fn red_tmp(c: u8, r: u8) -> u8 { 8 + 4 * c + r }
fn y_addr(c: u8) -> u8 { 40 + c }

// SGPRs.
const S_B: u8 = 3;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_NV: u8 = 13;
const S_GROUPS: u8 = 14;
const S_STRIDE: u8 = 15;
/// Weight soffset of row r (row offset plus the current quad base).
fn s_row(r: u8) -> u8 { 16 + r }
/// Activation soffset of column c (column offset plus the current quad base).
fn s_xcol(c: u8) -> u8 { 28 + c }
const S_QUADS: u8 = 36;
const S_TAIL: u8 = 37;
const S_KX4: u8 = 38;
const S_MX4: u8 = 39;
const S_T0: u8 = 40;
const S_T1: u8 = 41;
const S_ROW0X4: u8 = 42;
pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(VGPRS, 48)?;
    for (name, base, width) in [
        ("kernarg", 0, 2), ("columns", S_B, 1), ("matrix_activation", 4, 4), ("output", 8, 2),
        ("m", S_M, 1), ("k", S_K, 1), ("row0", S_ROW0, 1), ("valid_rows", S_NV, 1),
        ("groups", S_GROUPS, 1), ("stride", S_STRIDE, 1), ("row_offsets", 16, 4),
        ("weight_resource", 20, 4), ("activation_resource", 24, 4), ("column_offsets0", 28, 4),
        ("column_offsets1", 32, 4),
        ("quads", S_QUADS, 1), ("tail", S_TAIL, 1), ("k_x4", S_KX4, 1), ("m_x4", S_MX4, 1),
        ("temp0", S_T0, 1), ("temp1", S_T1, 1), ("row0_x4", S_ROW0X4, 1),
    ] {
        regs.add_range(name, Kind::S, base, width, Live::Whole)?;
    }
    for (name, base, width) in [
        ("lane", LANE, 1), ("header_offset", HDR_OFF, 1), ("packed_offset", PK_OFF, 1),
        ("activation_offset", X_OFF, 1), ("shuffle_addr", SHUF_ADDR, 1), ("headers", 8, 16),
        ("packed", 24, 16), ("weights0", W[0], 8), ("dots0", 48, 2), ("weights1", W[1], 8),
        ("dots1", 58, 2), ("weights2", W[2], 8), ("dots2", 68, 2), ("weights3", W[3], 8),
        ("dots3", 78, 2), ("ring_accumulators", RING as u8, (VGPRS - RING) as u8),
    ] {
        // The allocator admits widths 1/2/4/8: wide regions are 8-register pieces.
        let mut at = 0u8;
        while at < width {
            let piece = (width - at).min(8);
            regs.add_range(&format!("{name}{at}"), Kind::V, base + at, piece, Live::Whole)?;
            at += piece;
        }
    }
    let mut b = Builder::new(KernelSpec {
        kernel_id: "gemv_hfq4g256_xbatch_mq4v2".into(), variant: "exact_columns_r4_b8".into(),
        arch: Arch::Gfx1201, symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(36).pointer_access("A", 0, crate::plan::Access::ReadOnly)
            .pointer_access("x", 8, crate::plan::Access::ReadOnly)
            .pointer_access("y", 16, crate::plan::Access::WriteOnly)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("B", 32, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    smem(&mut b, "s_load_b64 s[10:11], s[0:1], 0x18", &[S_M, S_K], &[0, 1])?;
    // gfx12 receives the workgroup index in ttmp9.
    salu(&mut b, "s_lshl_b32 s12, ttmp9, 2", &[S_ROW0], &[])?;
    b.wait_all()?;
    salu(&mut b, "s_cmp_ge_i32 s12, s10", &[], &[S_ROW0, S_M])?;
    branch(&mut b, "s_cbranch_scc1 .Lpx_end")?;
    smem(&mut b, "s_load_b128 s[4:7], s[0:1], 0x0", &[4, 5, 6, 7], &[0, 1])?;
    smem(&mut b, "s_load_b64 s[8:9], s[0:1], 0x10", &[8, 9], &[0, 1])?;
    smem(&mut b, "s_load_b32 s3, s[0:1], 0x20", &[S_B], &[0, 1])?;
    salu(&mut b, "s_sub_co_i32 s13, s10, s12", &[S_NV], &[S_M, S_ROW0])?;
    salu(&mut b, "s_min_i32 s13, s13, 4", &[S_NV], &[S_NV])?;
    salu(&mut b, "s_add_co_i32 s40, s10, -1", &[S_T0], &[S_M])?;
    salu(&mut b, "s_lshr_b32 s14, s11, 8", &[S_GROUPS], &[S_K])?;
    salu(&mut b, "s_mul_i32 s15, s14, 0x88", &[S_STRIDE], &[S_GROUPS])?;
    for r in 0..4u8 {
        salu(&mut b, &format!("s_add_co_i32 s41, s12, {r}"), &[S_T1], &[S_ROW0])?;
        salu(&mut b, "s_min_i32 s41, s41, s40", &[S_T1], &[S_T1, S_T0])?;
        salu(&mut b, &format!("s_mul_i32 s{}, s41, s15", s_row(r)), &[s_row(r)], &[S_T1, S_STRIDE])?;
    }
    salu(&mut b, "s_lshl_b32 s38, s11, 2", &[S_KX4], &[S_K])?;
    salu(&mut b, "s_lshl_b32 s39, s10, 2", &[S_MX4], &[S_M])?;
    salu(&mut b, "s_lshl_b32 s42, s12, 2", &[S_ROW0X4], &[S_ROW0])?;
    for c in 0..MAX_COLUMNS {
        salu(&mut b, &format!("s_mul_i32 s{}, s38, {c}", s_xcol(c)), &[s_xcol(c)], &[S_KX4])?;
    }
    valu(&mut b, "v_lshrrev_b32_e32 v1, 4, v0", &[HDR_OFF], &[LANE], &[])?;
    valu(&mut b, "v_lshlrev_b32_e32 v1, 2, v1", &[HDR_OFF], &[HDR_OFF], &[])?;
    valu(&mut b, "v_lshlrev_b32_e32 v2, 2, v0", &[PK_OFF], &[LANE], &[])?;
    valu(&mut b, "v_lshlrev_b32_e32 v3, 5, v0", &[X_OFF], &[LANE], &[])?;
    b.wait_all()?;
    // Raw buffer resources (stride 0, unbounded records) over A and x.
    for (res, ptr) in [(20u8, 4u8), (24, 6)] {
        salu(&mut b, &format!("s_mov_b32 s{res}, s{ptr}"), &[res], &[ptr])?;
        salu(&mut b, &format!("s_and_b32 s{}, s{}, 0xffff", res + 1, ptr + 1), &[res + 1], &[ptr + 1])?;
        salu(&mut b, &format!("s_mov_b32 s{}, -1", res + 2), &[res + 2], &[])?;
        salu(&mut b, &format!("s_mov_b32 s{}, 0x31004000", res + 3), &[res + 3], &[])?;
    }
    for nb in 1..=MAX_COLUMNS {
        salu(&mut b, &format!("s_cmp_eq_u32 s3, {nb}"), &[], &[S_B])?;
        branch(&mut b, &format!("s_cbranch_scc1 .Lpx_b{nb}"))?;
    }
    branch(&mut b, "s_branch .Lpx_end")?;
    for nb in 1..=MAX_COLUMNS {
        section(&mut b, nb)?;
    }
    b.label(".Lpx_end")?;
    branch(&mut b, "s_endpgm")?;
    Ok(vec![b.finish()?])
}

/// The complete kernel body for `nb` columns.
fn section(b: &mut Builder, nb: u8) -> Result<(), String> {
    b.label(&format!(".Lpx_b{nb}"))?;
    for c in 0..nb {
        for s in 0..4 {
            for r in 0..4 {
                let a = acc(nb, c, s, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    salu(b, "s_lshr_b32 s36, s14, 2", &[S_QUADS], &[S_GROUPS])?;
    salu(b, "s_and_b32 s37, s14, 3", &[S_TAIL], &[S_GROUPS])?;
    salu(b, "s_cmp_eq_u32 s36, 0", &[], &[S_QUADS])?;
    branch(b, &format!("s_cbranch_scc1 .Lpx_b{nb}_tail"))?;
    b.loop_(&format!(".Lpx_b{nb}_quad"), |b| {
        body(b, nb, &[0, 1, 2, 3])?;
        for r in 0..4 {
            salu(b, &format!("s_add_co_i32 s{0}, s{0}, 0x220", s_row(r)), &[s_row(r)], &[s_row(r)])?;
        }
        for c in 0..nb {
            salu(b, &format!("s_add_co_i32 s{0}, s{0}, 0x1000", s_xcol(c)), &[s_xcol(c)], &[s_xcol(c)])?;
        }
        salu(b, "s_add_co_i32 s36, s36, -1", &[S_QUADS], &[S_QUADS])?;
        salu(b, "s_cmp_lg_u32 s36, 0", &[], &[S_QUADS])?;
        branch(b, &format!("s_cbranch_scc1 .Lpx_b{nb}_quad"))
    })?;
    b.label(&format!(".Lpx_b{nb}_tail"))?;
    // Tail group t (at the post-quad bases) accumulates into stream t.
    for t in 0..3u8 {
        salu(b, &format!("s_cmp_gt_u32 s37, {t}"), &[], &[S_TAIL])?;
        branch(b, &format!("s_cbranch_scc0 .Lpx_b{nb}_fold"))?;
        body(b, nb, &[t])?;
    }
    b.label(&format!(".Lpx_b{nb}_fold"))?;
    fold(b, nb)?;
    reduce(b, nb)?;
    store(b, nb)
}

/// Groups `js` (slot = stream = immediate index) of the four rows against
/// every column. Weights of the first group and its activations issue first,
/// then the remaining weights; activations of later items stream through the
/// ring as earlier items retire.
fn body(b: &mut Builder, nb: u8, js: &[u8]) -> Result<(), String> {
    let items = js.len() as u16 * u16::from(nb);
    let ring_slots = slots(nb);
    weight_loads(b, js[0])?;
    let mut issued = 0u16;
    while issued < ring_slots.min(u16::from(nb)) {
        x_load(b, nb, js, issued)?;
        issued += 1;
    }
    for &j in &js[1..] {
        weight_loads(b, j)?;
    }
    for (index, &j) in js.iter().enumerate() {
        dequant(b, j)?;
        let mut c = 0u8;
        while c < nb {
            let item = index as u16 * u16::from(nb) + u16::from(c);
            while issued < (item + ring_slots).min(items) {
                x_load(b, nb, js, issued)?;
                issued += 1;
            }
            let cols: Vec<(u8, u8)> = (c..nb.min(c + 2))
                .map(|col| (col, ring(nb, item + u16::from(col - c))))
                .collect();
            columns(b, nb, j, &cols)?;
            c += 2;
        }
    }
    b.wait_all()
}

/// Lane-selected half headers and packed nibbles of group slot `j`, all rows.
fn weight_loads(b: &mut Builder, j: u8) -> Result<(), String> {
    let off = u32::from(j) * 136;
    for r in 0..4u8 {
        let h = hdr(j, r);
        let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
        vmem(b, &format!("buffer_load_b32 v{h}, v1, s[20:23], s{} offen{suffix}", s_row(r)),
            &[h], &[HDR_OFF], &[20, 21, 22, 23, s_row(r)], false)?;
    }
    for r in 0..4u8 {
        let p = pk(j, r);
        vmem(b, &format!("buffer_load_b32 v{p}, v2, s[20:23], s{} offen offset:{}", s_row(r), off + 8),
            &[p], &[PK_OFF], &[20, 21, 22, 23, s_row(r)], false)?;
    }
    Ok(())
}

/// The eight activations of item `item` (group `js[item / nb]`, column
/// `item % nb`) into its ring slot.
fn x_load(b: &mut Builder, nb: u8, js: &[u8], item: u16) -> Result<(), String> {
    let j = js[usize::from(item / u16::from(nb))];
    let c = (item % u16::from(nb)) as u8;
    let x = ring(nb, item);
    for half in 0..2u8 {
        let off = u32::from(j) * 1024 + u32::from(half) * 16;
        let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
        let d = x + 4 * half;
        vmem(b, &format!("buffer_load_b128 v[{d}:{}], v3, s[24:27], s{} offen{suffix}", d + 3, s_xcol(c)),
            &[d, d + 1, d + 2, d + 3], &[X_OFF], &[24, 25, 26, 27, s_xcol(c)], false)?;
    }
    Ok(())
}

/// `w[r][n] = fma_mix(sc, nibble_n, zp)` for the four rows of group slot `j`.
/// Even nibbles are the bytes of `pk & 0x0f0f0f0f`, odd nibbles the bytes of
/// `(pk >> 4) & 0x0f0f0f0f`; slots 6/7 hold the masked words until last.
fn dequant(b: &mut Builder, j: u8) -> Result<(), String> {
    for r in 0..4u8 {
        let (w, p) = (W[usize::from(r)], pk(j, r));
        valu(b, &format!("v_and_b32_e32 v{}, 0xf0f0f0f, v{p}", w + 6), &[w + 6], &[p], &[])?;
    }
    for r in 0..4u8 {
        let (w, p) = (W[usize::from(r)], pk(j, r));
        valu(b, &format!("v_lshrrev_b32_e32 v{}, 4, v{p}", w + 7), &[w + 7], &[p], &[])?;
    }
    for r in 0..4u8 {
        let w = W[usize::from(r)];
        valu(b, &format!("v_and_b32_e32 v{0}, 0xf0f0f0f, v{0}", w + 7), &[w + 7], &[w + 7], &[])?;
    }
    for (src, nibbles) in [(6u8, [0u8, 2, 4, 6]), (7, [1, 3, 5, 7])] {
        for (byte, n) in nibbles.into_iter().enumerate() {
            for r in 0..4u8 {
                let w = W[usize::from(r)];
                valu(b, &format!("v_cvt_f32_ubyte{byte}_e32 v{}, v{}", w + n, w + src), &[w + n], &[w + src], &[])?;
            }
        }
    }
    for n in 0..8u8 {
        for r in 0..4u8 {
            let (w, h) = (W[usize::from(r)] + n, hdr(j, r));
            valu(b, &format!("v_fma_mix_f32 v{w}, v{h}, v{w}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"), &[w], &[h, w], &[])?;
        }
    }
    Ok(())
}

/// Dot the four dequantized rows against up to two columns (each `(column,
/// ring slot)`) and add the dots into stream `s`. Terms run x1, x0, x2..x7.
fn columns(b: &mut Builder, nb: u8, s: u8, cols: &[(u8, u8)]) -> Result<(), String> {
    for (term, n) in [1u8, 0, 2, 3, 4, 5, 6, 7].into_iter().enumerate() {
        let op = if term == 0 { VopdF32::Mul } else { VopdF32::Fmac };
        for (slot, &(_, x)) in cols.iter().enumerate() {
            for (ra, rb) in [(0usize, 1usize), (2, 3)] {
                b.vopd(
                    VopdOp { op, dst: DOT[slot][ra], src0: Operand::V(W[ra] + n), src1: x + n },
                    VopdOp { op, dst: DOT[slot][rb], src0: Operand::V(W[rb] + n), src1: x + n },
                )?;
            }
        }
    }
    for (slot, &(c, _)) in cols.iter().enumerate() {
        for (ra, rb) in [(0u8, 1u8), (2, 3)] {
            let (a0, a1) = (acc(nb, c, s, ra), acc(nb, c, s, rb));
            b.vopd(
                VopdOp { op: VopdF32::Add, dst: a0, src0: Operand::V(a0), src1: DOT[slot][usize::from(ra)] },
                VopdOp { op: VopdF32::Add, dst: a1, src0: Operand::V(a1), src1: DOT[slot][usize::from(rb)] },
            )?;
        }
    }
    Ok(())
}

/// `(a0 + a1) + (a2 + a3)` per column and row, into stream 0.
fn fold(b: &mut Builder, nb: u8) -> Result<(), String> {
    for c in 0..nb {
        for (dst, other) in [(0u8, 1u8), (2, 3), (0, 2)] {
            for (ra, rb) in [(0u8, 1u8), (2, 3)] {
                let (x, y) = (acc(nb, c, dst, ra), acc(nb, c, dst, rb));
                b.vopd(
                    VopdOp { op: VopdF32::Add, dst: x, src0: Operand::V(x), src1: acc(nb, c, other, ra) },
                    VopdOp { op: VopdF32::Add, dst: y, src0: Operand::V(y), src1: acc(nb, c, other, rb) },
                )?;
            }
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, nb: u8) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..nb).flat_map(|c| (0..4u8).map(move |r| (c, r))).collect();
    for &(c, r) in &values {
        let (v, t) = (acc(nb, c, 0, r), red_tmp(c, r));
        b.ds_crosslane(Instruction::new(format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            refs(Kind::V, &[t]), refs(Kind::V, &[v])).memory(MemoryClass::DsLoad))?;
    }
    for &(c, r) in &values {
        let (v, t) = (acc(nb, c, 0, r), red_tmp(c, r));
        valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
    }
    for offset in [8u8, 4, 2, 1] {
        valu(b, &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset), &[], &[LANE], &[])?;
        b.push(Instruction::new("s_wait_alu depctr_va_vcc(0)", vec![], vec![]))?;
        valu(b, &format!("v_cndmask_b32_e64 v4, 0, {offset}, vcc_lo"), &[SHUF_ADDR], &[], &[])?;
        valu(b, "v_add_lshl_u32 v4, v4, v0, 2", &[SHUF_ADDR], &[SHUF_ADDR, LANE], &[])?;
        for &(c, r) in &values {
            let (v, t) = (acc(nb, c, 0, r), red_tmp(c, r));
            b.ds_crosslane(Instruction::new(format!("ds_bpermute_b32 v{t}, v4, v{v}"),
                refs(Kind::V, &[t]), refs(Kind::V, &[SHUF_ADDR, v])).memory(MemoryClass::DsLoad))?;
        }
        for &(c, r) in &values {
            let (v, t) = (acc(nb, c, 0, r), red_tmp(c, r));
            valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
        }
    }
    Ok(())
}

/// Lane 0 stores `y[c][row0 + r]` for every column and each valid row.
fn store(b: &mut Builder, nb: u8) -> Result<(), String> {
    valu(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[LANE], &[])?;
    for c in 0..nb {
        let a = y_addr(c);
        if c == 0 {
            valu(b, &format!("v_mov_b32_e32 v{a}, s{S_ROW0X4}"), &[a], &[], &[S_ROW0X4])?;
        } else {
            salu(b, &format!("s_mul_i32 s40, s39, {c}"), &[S_T0], &[S_MX4])?;
            salu(b, "s_add_co_i32 s40, s40, s42", &[S_T0], &[S_T0, S_ROW0X4])?;
            valu(b, &format!("v_mov_b32_e32 v{a}, s40"), &[a], &[], &[S_T0])?;
        }
    }
    for r in 0..4u8 {
        if r > 0 {
            salu(b, &format!("s_cmp_le_i32 s13, {r}"), &[], &[S_NV])?;
            branch(b, &format!("s_cbranch_scc1 .Lpx_b{nb}_stored"))?;
        }
        for c in 0..nb {
            let (a, v) = (y_addr(c), acc(nb, c, 0, r));
            let suffix = if r == 0 { String::new() } else { format!(" offset:{}", 4 * r) };
            vmem(b, &format!("global_store_b32 v{a}, v{v}, s[8:9]{suffix}"), &[], &[a, v], &[8, 9], true)?;
        }
    }
    b.label(&format!(".Lpx_b{nb}_stored"))?;
    b.wait_all()?;
    branch(b, "s_branch .Lpx_end")
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices.iter().map(|&base| crate::reg::RegRef { kind, base, len: 1 }).collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses)))
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses))
}
fn smem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses)).memory(MemoryClass::SmemLoad))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses)
        .memory(if store { MemoryClass::VmemStore } else { MemoryClass::VmemLoad }))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> { b.control(Instruction::new(text, vec![], vec![])) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn register_plan_fits() {
        for nb in 1..=MAX_COLUMNS {
            assert!(slots(nb) >= 2, "nb={nb}");
            assert_eq!(acc(nb, 0, 0, 0) % 4, 0);
            assert!(u16::from(acc(nb, nb - 1, 3, 3)) < VGPRS, "nb={nb}");
            assert!(32 + 2 * slots(nb) <= 62, "nb={nb}");
        }
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn xbatch_pm_m7() {
        let emitted = build_gfx1201().expect("xbatch pm").remove(0);
        // Both rows of a pair seed x1 (shared src1), then x0, then x2..x7.
        assert!(emitted.s_text.contains("v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81"));
        assert!(emitted.s_text.contains("v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80"));
        let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1201).expect("native assemble");
        let dir = std::env::temp_dir()
            .join(format!("hipfire-isa-xbatch-pm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gemv_hfq4g256_xbatch_mq4v2.test.s"), &emitted.s_text).unwrap();
        let co = dir.join("gemv_hfq4g256_xbatch_mq4v2.test.co");
        std::fs::write(&co, &elf).unwrap();
        let report = crate::pm_check::m7(&co, "gfx1201", SYMBOL).expect("M7");
        println!("{report}");
        assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
    }
}
