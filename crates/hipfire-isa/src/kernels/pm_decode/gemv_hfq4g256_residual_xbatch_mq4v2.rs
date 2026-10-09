//! Exact gfx1201 multi-column twin of `gemv_mq4g256v2_residual`.
//!
//! `gemv_mq4g256v2_residual_xbatch(A, x[B][K], y[B][M], M, K, B)`, B in 1..=8:
//! for every column `b`, `y[b]` is byte-identical to one singleton
//! `gemv_mq4g256v2_residual` launch on `x[b]`, `y[b]`.
//! ABI: A/x/y pointers at 0/8/16, M/K/B i32 at 24/28/32; block 32, no LDS or
//! scratch; x is `[B][K]` and y is `[B][M]`, both row-major f32. Workgroup
//! `w` owns rows `4w .. 4w+4` (`ROWS_PER_WORKGROUP`): the launcher grid is
//! `ceil(M / 4)` (larger grids are safe; surplus workgroups exit).
//!
//! Each 136-byte weight group of the workgroup's rows is loaded and
//! dequantized once (`fma_mix(sc, nibble, zp)`), then applied to every column,
//! so one weight pass serves all B columns and one x load serves R rows.
//! Per column and row the singleton's per-lane DAG is kept exactly:
//! - group `g` accumulates into stream `g % 4` in increasing `g` (the
//!   singleton's quads and scalar tails alike). Streams run one after another
//!   (`g = s, s+4, ...`), so only three accumulator slots are live: stream 0
//!   in A, stream 1 in B, then `A = A + B`; stream 2 in B, stream 3 in C, then
//!   `B = B + C`, `A = A + B`, which is the singleton's `(a0+a1)+(a2+a3)`;
//! - each group dot is one `mul` plus seven `fmac`; even rows (the
//!   singleton's row0) seed x0 then x1, odd rows (row1) seed x1 then x0;
//! - the stream add is `dot + acc` for even rows and `acc + dot` for odd rows;
//! - the reduction is the frozen shfl_down tree (swizzle `1pppp`, then
//!   self-reading bpermutes 8/4/2/1);
//! - the residual epilogue is `acc + y` for a row pair and `y + acc` for an
//!   odd final row.
//!
//! Rows `2p`/`2p+1` of one column are VOPD pairs (multiplication commutes; the
//! shared x operand is the gfx12 shared-src1 form; row weight bases differ by
//! 2 mod 4, x sits one bank past the even row's weights). Weights and x of
//! the next group of a stream are issued as soon as the current group's
//! registers are consumed. B is specialized: one code path per column count,
//! selected once.
//!
//! Throughput (R9700, down 5120x17408): the dot FMAs are VALU-bound at about
//! one FMA per lane per clock (a `v_dual_fmac` pair reads five VGPRs, so it
//! does not dual-issue), so cost grows with B once the weight stream is
//! covered; see the oracle example's timing mode.
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Kind, Live};
use crate::vopd::{Operand, VopdF32, VopdOp};

pub const SYMBOL: &str = "gemv_mq4g256v2_residual_xbatch";
pub const MAX_COLUMNS: u8 = 8;
/// Output rows per workgroup (two singleton row pairs). More rows share each
/// x load but cost 3*R*B accumulators; on R9700 4 rows matched 6/8 rows at
/// every B and is the only width that fits B = 7, 8.
pub const ROWS_PER_WORKGROUP: u8 = 4;

// Fixed VGPRs.
const LANE: u8 = 0;
const LANE_X4: u8 = 1;
const X_OFF: u8 = 2;
/// Shuffle address in the reduction; next-group x offset in the stream loops.
const SHUF_ADDR: u8 = 3;
const X_NEXT: u8 = 3;

// SGPRs.
const S_COUNT: u8 = 2;
const S_B: u8 = 3;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_ROW: u8 = 13;
const S_GROUPS: u8 = 14;
const S_GOFF: u8 = 15;
const S_GNEXT: u8 = 16;
const S_GSTEP: u8 = 17;
const S_XSTEP: u8 = 18;
const S_STRIDE: u8 = 19;
fn s_xbase(b: u8) -> u8 { 28 + 2 * b }
const S_KX4: u8 = 44;
const S_MX4: u8 = 45;
const S_TMP: u8 = 24;

/// Accumulator slots (see the module docs).
const SLOT_A: u8 = 0;
const SLOT_B: u8 = 1;
const SLOT_C: u8 = 2;

/// The VGPR layout of the `nb`-column path with `rows` rows per workgroup.
#[derive(Clone, Copy)]
struct Layout {
    rows: u8,
    nb: u8,
    pk_addr: u8,
    hdr_addr: u8,
    hdr: u8,
    pk: u8,
    w: u8,
    acc: u8,
    x: u8,
    end: u16,
}

impl Layout {
    fn new(nb: u8) -> Self {
        let rows = ROWS_PER_WORKGROUP;
        let pk_addr = 4;
        let hdr_addr = pk_addr + rows;
        let hdr = hdr_addr + rows;
        let pk = hdr + 2 * rows;
        let w = (pk + rows).next_multiple_of(4);
        let acc = w + 10 * rows;
        // x_n sits one bank past w_n (even rows at bank n, odd at n+2), so no
        // dot FMA reads its two sources from one VGPR bank.
        let x0 = u16::from(acc) + 3 * u16::from(rows) * u16::from(nb);
        let x = x0 + (5 - x0 % 4) % 4;
        let end = x + 8 * u16::from(nb);
        // An oversize layout keeps `end` > 256 and is rejected by the build.
        Layout { rows, nb, pk_addr, hdr_addr, hdr, pk, w, acc, x: x.min(255) as u8, end }
    }
    fn pairs(self) -> u8 { self.rows / 2 }
    /// Dequantized weights of row `r`: even rows at 0 mod 4, odd at 2 mod 4.
    fn w(self, r: u8) -> u8 { self.w + 10 * r }
    /// Dot pair (even, odd) of (column slot `cs`, row pair `p`): the two
    /// registers after one row's weights.
    fn dot(self, cs: u8, p: u8) -> [u8; 2] {
        let g = self.w(cs * self.pairs() + p) + 8;
        [g, g + 1]
    }
    fn acc(self, b: u8, slot: u8, r: u8) -> u8 { self.acc + (b * 3 + slot) * self.rows + r }
    fn x(self, b: u8) -> u8 { self.x + 8 * b }
    fn hdr(self, r: u8) -> u8 { self.hdr + 2 * r }
    // Epilogue temporaries come from the then-dead weight block and x buffer:
    // R*nb reduction temps, R*nb y values, nb y addresses.
    fn spare(self, i: u8) -> u8 {
        let weights = self.acc - self.w;
        if i < weights { self.w + i } else { self.x + (i - weights) }
    }
    fn spare_fits(self) -> bool {
        u16::from(self.rows) * u16::from(self.nb) * 2 + u16::from(self.nb)
            <= u16::from(self.acc - self.w) + 8 * u16::from(self.nb)
    }
    fn red_tmp(self, b: u8, r: u8) -> u8 { self.spare(b * self.rows + r) }
    fn y_tmp(self, b: u8, r: u8) -> u8 { self.spare(self.nb * self.rows + b * self.rows + r) }
    fn y_addr(self, b: u8) -> u8 { self.spare(2 * self.nb * self.rows + b) }
}

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    for nb in 1..=MAX_COLUMNS {
        let l = Layout::new(nb);
        if l.rows == 0 || l.rows % 2 != 0 || !l.spare_fits() || u16::from(l.end) > 256 {
            return Err(format!("B={nb}: {} rows / {} VGPRs is not a valid layout", l.rows, l.end));
        }
    }
    let vgprs = (1..=MAX_COLUMNS).map(|nb| Layout::new(nb).end).max().unwrap_or(0);
    let mut regs = RegPlan::new(u16::from(vgprs), 48)?;
    for (name, base) in [("kernarg", 0), ("matrix", 4), ("activation", 6), ("residual", 8)] {
        regs.s::<2>(name, base, Live::Whole)?;
    }
    regs.s::<4>("weight_resource", 20, Live::Whole)?;
    for col in 0..MAX_COLUMNS { regs.add_range(&format!("column_x_base{col}"), Kind::S, s_xbase(col), 2, Live::Whole)?; }
    for base in [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 44, 45, 46, 47] {
        regs.add_range(&format!("scalar{base}"), Kind::S, base, 1, Live::Whole)?;
    }
    for (name, base) in [("lane", LANE), ("lane_x4", LANE_X4), ("x_offset", X_OFF), ("shuffle_addr", SHUF_ADDR)] {
        regs.add_range(name, Kind::V, base, 1, Live::Whole)?;
    }
    // Each B path lays out v4.. itself (`Layout`): one shared register file.
    let mut at = 4u16;
    while at < vgprs {
        let w = [8u16, 4, 2, 1].into_iter().find(|&w| at % w == 0 && at + w <= vgprs).unwrap_or(1);
        regs.add_range(&format!("path_v{at}"), Kind::V, at as u8, w as u8, Live::Whole)?;
        at += w;
    }
    let mut b = Builder::new(KernelSpec {
        kernel_id: "gemv_hfq4g256_residual_xbatch_mq4v2".into(), variant: "exact_columns_stream_major".into(),
        arch: Arch::Gfx1201, symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(36).pointer_access("A", 0, crate::plan::Access::ReadOnly)
            .pointer_access("x", 8, crate::plan::Access::ReadOnly).pointer("y", 16)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("B", 32, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    // Dependent-VALU issue hints: ~5% at B=4 (R9700, down K=17408).
    b.enable_delay_alu();
    smem(&mut b, "s_load_b128 s[4:7], s[0:1], 0x0", &[4, 5, 6, 7], &[0, 1])?;
    smem(&mut b, "s_load_b128 s[8:11], s[0:1], 0x10", &[8, 9, 10, 11], &[0, 1])?;
    smem(&mut b, "s_load_b32 s3, s[0:1], 0x20", &[S_B], &[0, 1])?;
    b.wait_all()?;
    salu(&mut b, "s_lshr_b32 s14, s11, 8", &[S_GROUPS], &[S_K])?;
    salu(&mut b, "s_mul_i32 s19, s14, 0x88", &[S_STRIDE], &[S_GROUPS])?;
    salu(&mut b, "s_mov_b32 s20, s4", &[20], &[4])?;
    salu(&mut b, "s_and_b32 s21, s5, 0xffff", &[21], &[5])?;
    salu(&mut b, "s_mov_b32 s22, -1", &[22], &[])?;
    salu(&mut b, "s_mov_b32 s23, 0x31004000", &[23], &[])?;
    salu(&mut b, "s_lshl_b32 s44, s11, 2", &[S_KX4], &[S_K])?;
    salu(&mut b, "s_lshl_b32 s45, s10, 2", &[S_MX4], &[S_M])?;
    for col in 0..MAX_COLUMNS {
        let (lo, hi) = (s_xbase(col), s_xbase(col) + 1);
        if col == 0 {
            salu(&mut b, &format!("s_mov_b32 s{lo}, s6"), &[lo], &[6])?;
            salu(&mut b, &format!("s_mov_b32 s{hi}, s7"), &[hi], &[7])?;
            continue;
        }
        salu(&mut b, &format!("s_mul_i32 s24, s44, {col}"), &[S_TMP], &[S_KX4])?;
        salu(&mut b, &format!("s_add_co_u32 s{lo}, s6, s24"), &[lo], &[6, S_TMP])?;
        salu(&mut b, &format!("s_add_co_ci_u32 s{hi}, s7, 0"), &[hi], &[7])?;
    }
    valu(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[LANE_X4], &[LANE], &[])?;
    for nb in 1..=MAX_COLUMNS {
        salu(&mut b, &format!("s_cmp_eq_u32 s3, {nb}"), &[], &[S_B])?;
        branch(&mut b, &format!("s_cbranch_scc1 .Lrx_b{nb}"))?;
    }
    branch(&mut b, "s_branch .Lrx_end")?;
    for nb in 1..=MAX_COLUMNS {
        section(&mut b, Layout::new(nb))?;
    }
    b.label(".Lrx_end")?;
    branch(&mut b, "s_endpgm")?;
    Ok(vec![b.finish()?])
}

/// The complete kernel body for one column count.
fn section(b: &mut Builder, l: Layout) -> Result<(), String> {
    let nb = l.nb;
    b.label(&format!(".Lrx_b{nb}"))?;
    salu(b, &format!("s_mul_i32 s12, ttmp9, {}", l.rows), &[S_ROW0], &[])?;
    salu(b, "s_cmp_ge_i32 s12, s10", &[], &[S_ROW0, S_M])?;
    branch(b, "s_cbranch_scc1 .Lrx_end")?;
    // Row weight bases; rows past M read row0 (the singleton's missing-row1
    // rule) and are never stored.
    for r in 0..l.rows {
        if r == 0 {
            salu(b, "s_mov_b32 s13, s12", &[S_ROW], &[S_ROW0])?;
        } else {
            salu(b, &format!("s_add_co_i32 s13, s12, {r}"), &[S_ROW], &[S_ROW0])?;
            salu(b, "s_cmp_lt_i32 s13, s10", &[], &[S_ROW, S_M])?;
            salu(b, "s_cselect_b32 s13, s13, s12", &[S_ROW], &[S_ROW, S_ROW0])?;
        }
        salu(b, "s_mul_i32 s13, s13, s19", &[S_ROW], &[S_ROW, S_STRIDE])?;
        let (h, p) = (l.hdr_addr + r, l.pk_addr + r);
        valu(b, &format!("v_mov_b32_e32 v{h}, s13"), &[h], &[], &[S_ROW])?;
        valu(b, &format!("v_add_nc_u32_e32 v{p}, s13, v1"), &[p], &[LANE_X4], &[S_ROW])?;
    }
    for col in 0..nb {
        for slot in [SLOT_A, SLOT_B, SLOT_C] {
            for r in 0..l.rows {
                let a = l.acc(col, slot, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    for stream in 0..4u8 {
        let slot = [SLOT_A, SLOT_B, SLOT_B, SLOT_C][usize::from(stream)];
        let head = format!(".Lrx_b{nb}_s{stream}");
        let done = format!(".Lrx_b{nb}_s{stream}_done");
        // Groups of this stream: ceil((G - s) / 4) (G >= 1).
        salu(b, &format!("s_add_co_i32 s2, s14, {}", 3 - stream), &[S_COUNT], &[S_GROUPS])?;
        salu(b, "s_lshr_b32 s2, s2, 2", &[S_COUNT], &[S_COUNT])?;
        salu(b, "s_cmp_eq_u32 s2, 0", &[], &[S_COUNT])?;
        branch(b, &format!("s_cbranch_scc1 {done}"))?;
        // Inline 0 for stream 0, a literal (hex) offset otherwise.
        let goff = if stream == 0 { "0".to_string() } else { format!("{:#x}", u32::from(stream) * 136) };
        salu(b, &format!("s_mov_b32 s15, {goff}"), &[S_GOFF], &[])?;
        valu(b, "v_lshlrev_b32_e32 v2, 5, v0", &[X_OFF], &[LANE], &[])?;
        if stream > 0 {
            valu(b, &format!("v_add_nc_u32_e32 v2, {:#x}, v2", u32::from(stream) * 1024), &[X_OFF], &[X_OFF], &[])?;
        }
        // Software pipeline: group g's weights and x are in flight on entry;
        // each iteration issues group g+4's (the same group again on the
        // last iteration, so no load leaves the row or the x column).
        weight_loads(b, l, S_GOFF)?;
        for col in 0..nb { x_loads(b, l, col, X_OFF)?; }
        b.loop_(&head, |b| {
            salu(b, "s_cmp_gt_u32 s2, 1", &[], &[S_COUNT])?;
            salu(b, "s_cselect_b32 s17, 0x220, 0", &[S_GSTEP], &[])?;
            salu(b, "s_cselect_b32 s18, 0x1000, 0", &[S_XSTEP], &[])?;
            salu(b, "s_add_co_i32 s16, s15, s17", &[S_GNEXT], &[S_GOFF, S_GSTEP])?;
            valu(b, "v_add_nc_u32_e32 v3, s18, v2", &[X_NEXT], &[X_OFF], &[S_XSTEP])?;
            group(b, l, slot)?;
            salu(b, "s_mov_b32 s15, s16", &[S_GOFF], &[S_GNEXT])?;
            valu(b, "v_mov_b32_e32 v2, v3", &[X_OFF], &[X_NEXT], &[])?;
            salu(b, "s_add_co_i32 s2, s2, -1", &[S_COUNT], &[S_COUNT])?;
            salu(b, "s_cmp_lg_u32 s2, 0", &[], &[S_COUNT])?;
            branch(b, &format!("s_cbranch_scc1 {head}"))
        })?;
        // The last iteration's repeat loads.
        b.wait_all()?;
        b.label(&done)?;
        match stream {
            1 => {
                fold(b, l, SLOT_A, SLOT_B)?;
                for col in 0..nb {
                    for r in 0..l.rows {
                        let a = l.acc(col, SLOT_B, r);
                        valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
                    }
                }
            }
            3 => {
                fold(b, l, SLOT_B, SLOT_C)?;
                fold(b, l, SLOT_A, SLOT_B)?;
            }
            _ => {}
        }
    }
    reduce(b, l)?;
    epilogue(b, l)
}

fn weight_loads(b: &mut Builder, l: Layout, soff: u8) -> Result<(), String> {
    for r in 0..l.rows {
        let (h, a) = (l.hdr(r), l.hdr_addr + r);
        vmem(b, &format!("buffer_load_b64 v[{h}:{}], v{a}, s[20:23], s{soff} offen scope:SCOPE_DEV", h + 1),
            &[h, h + 1], &[a], &[20, 21, 22, 23, soff], false)?;
        let (p, a) = (l.pk + r, l.pk_addr + r);
        vmem(b, &format!("buffer_load_b32 v{p}, v{a}, s[20:23], s{soff} offen offset:8 scope:SCOPE_DEV"),
            &[p], &[a], &[20, 21, 22, 23, soff], false)?;
    }
    Ok(())
}

fn x_loads(b: &mut Builder, l: Layout, col: u8, vaddr: u8) -> Result<(), String> {
    let x = l.x(col);
    let (lo, hi) = (s_xbase(col), s_xbase(col) + 1);
    for half in 0..2u8 {
        let d = x + 4 * half;
        let suffix = if half == 0 { "" } else { " offset:16" };
        vmem(b, &format!("global_load_b128 v[{d}:{}], v{vaddr}, s[{lo}:{hi}]{suffix}", d + 3),
            &[d, d + 1, d + 2, d + 3], &[vaddr], &[lo, hi], false)?;
    }
    Ok(())
}

/// One group of every row against every column, accumulated into `slot`;
/// group g+4's weights (`s16`) and x (`v3`) are issued as soon as the
/// registers of group g are consumed.
fn group(b: &mut Builder, l: Layout, slot: u8) -> Result<(), String> {
    // Dequantize: w = fma_mix(sc, (pk >> 4n) & 15, zp), half-header by lane.
    valu(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[LANE], &[])?;
    for r in 0..l.rows {
        let h = l.hdr(r);
        valu(b, &format!("v_cndmask_b32_e32 v{h}, v{}, v{h}, vcc_lo", h + 1), &[h], &[h, h + 1], &[])?;
        // Even nibbles are the bytes of pk & 0x0f0f0f0f, odd nibbles those of
        // (pk >> 4) & 0x0f0f0f0f: nibble n = byte n/2 of its half.
        let (w, p) = (l.w(r), l.pk + r);
        let (te, to) = (w + 6, w + 7);
        valu(b, &format!("v_and_b32_e32 v{te}, 0xf0f0f0f, v{p}"), &[te], &[p], &[])?;
        valu(b, &format!("v_lshrrev_b32_e32 v{to}, 4, v{p}"), &[to], &[p], &[])?;
        valu(b, &format!("v_and_b32_e32 v{to}, 0xf0f0f0f, v{to}"), &[to], &[to], &[])?;
        for n in [0u8, 2, 4, 1, 3, 5, 6, 7] {
            let src = if n % 2 == 0 { te } else { to };
            valu(b, &format!("v_cvt_f32_ubyte{}_e32 v{}, v{src}", n / 2, w + n), &[w + n], &[src], &[])?;
        }
        for n in 0..8u8 {
            valu(b, &format!("v_fma_mix_f32 v{}, v{h}, v{}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]", w + n, w + n), &[w + n], &[h, w + n], &[])?;
        }
    }
    weight_loads(b, l, S_GNEXT)?;
    // Dots: two columns interleave, each with one chain per row pair.
    let mut col = 0;
    while col < l.nb {
        let cols: Vec<u8> = (col..l.nb.min(col + 2)).collect();
        for step in 0..8u8 {
            // Even rows seed x0 then x1; odd rows seed x1 then x0.
            let (n0, n1) = match step { 0 => (0, 1), 1 => (1, 0), k => (k, k) };
            let op = if step == 0 { VopdF32::Mul } else { VopdF32::Fmac };
            for (cs, &c) in cols.iter().enumerate() {
                let x = l.x(c);
                for p in 0..l.pairs() {
                    let [d0, d1] = l.dot(cs as u8, p);
                    b.vopd(
                        VopdOp { op, dst: d0, src0: Operand::V(l.w(2 * p) + n0), src1: x + n0 },
                        VopdOp { op, dst: d1, src0: Operand::V(l.w(2 * p + 1) + n1), src1: x + n1 },
                    )?;
                }
            }
        }
        for &c in &cols { x_loads(b, l, c, X_NEXT)?; }
        for (cs, &c) in cols.iter().enumerate() {
            for p in 0..l.pairs() {
                let [d0, d1] = l.dot(cs as u8, p);
                let (a0, a1) = (l.acc(c, slot, 2 * p), l.acc(c, slot, 2 * p + 1));
                b.vopd(
                    VopdOp { op: VopdF32::Add, dst: a0, src0: Operand::V(d0), src1: a0 },
                    VopdOp { op: VopdF32::Add, dst: a1, src0: Operand::V(a1), src1: d1 },
                )?;
            }
        }
        col += 2;
    }
    Ok(())
}

/// `dst = dst + src` for every column and row.
fn fold(b: &mut Builder, l: Layout, dst: u8, src: u8) -> Result<(), String> {
    for col in 0..l.nb {
        for p in 0..l.pairs() {
            let (x, y) = (l.acc(col, dst, 2 * p), l.acc(col, dst, 2 * p + 1));
            b.vopd(
                VopdOp { op: VopdF32::Add, dst: x, src0: Operand::V(x), src1: l.acc(col, src, 2 * p) },
                VopdOp { op: VopdF32::Add, dst: y, src0: Operand::V(y), src1: l.acc(col, src, 2 * p + 1) },
            )?;
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, l: Layout) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..l.nb).flat_map(|c| (0..l.rows).map(move |r| (c, r))).collect();
    for &(c, r) in &values {
        let (v, t) = (l.acc(c, SLOT_A, r), l.red_tmp(c, r));
        b.ds_crosslane(Instruction::new(format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            refs(Kind::V, &[t]), refs(Kind::V, &[v])).memory(MemoryClass::DsLoad))?;
    }
    for &(c, r) in &values {
        let (v, t) = (l.acc(c, SLOT_A, r), l.red_tmp(c, r));
        valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
    }
    for offset in [8u8, 4, 2, 1] {
        valu(b, &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset), &[], &[LANE], &[])?;
        valu(b, &format!("v_cndmask_b32_e64 v3, 0, {offset}, vcc_lo"), &[SHUF_ADDR], &[], &[])?;
        valu(b, "v_add_lshl_u32 v3, v3, v0, 2", &[SHUF_ADDR], &[SHUF_ADDR, LANE], &[])?;
        for &(c, r) in &values {
            let (v, t) = (l.acc(c, SLOT_A, r), l.red_tmp(c, r));
            b.ds_crosslane(Instruction::new(format!("ds_bpermute_b32 v{t}, v3, v{v}"),
                refs(Kind::V, &[t]), refs(Kind::V, &[SHUF_ADDR, v])).memory(MemoryClass::DsLoad))?;
        }
        for &(c, r) in &values {
            let (v, t) = (l.acc(c, SLOT_A, r), l.red_tmp(c, r));
            valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
        }
    }
    Ok(())
}

/// Lane 0, per row pair `p` (rows `row0+2p`, `+2p+1`) while `row0+2p < M`:
/// `y = acc + y`, `y' = acc' + y'` when both rows exist, else `y = y + acc`
/// for the odd final row (the singleton's two epilogues).
fn epilogue(b: &mut Builder, l: Layout) -> Result<(), String> {
    let nb = l.nb;
    let done = format!(".Lrx_b{nb}_stored");
    valu(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[LANE], &[])?;
    salu(b, "s_lshl_b32 s46, s12, 2", &[46], &[S_ROW0])?;
    for col in 0..nb {
        let a = l.y_addr(col);
        if col == 0 {
            valu(b, &format!("v_mov_b32_e32 v{a}, s46"), &[a], &[], &[46])?;
        } else {
            salu(b, &format!("s_mul_i32 s47, s45, {col}"), &[47], &[S_MX4])?;
            salu(b, "s_add_co_i32 s47, s47, s46", &[47], &[47, 46])?;
            valu(b, &format!("v_mov_b32_e32 v{a}, s47"), &[a], &[], &[47])?;
        }
    }
    for p in 0..l.pairs() {
        let single = format!(".Lrx_b{nb}_p{p}_single");
        if p > 0 {
            salu(b, &format!("s_add_co_i32 s13, s12, {}", 2 * p), &[S_ROW], &[S_ROW0])?;
            salu(b, "s_cmp_lt_i32 s13, s10", &[], &[S_ROW, S_M])?;
            branch(b, &format!("s_cbranch_scc0 {done}"))?;
        }
        salu(b, &format!("s_add_co_i32 s13, s12, {}", 2 * p + 1), &[S_ROW], &[S_ROW0])?;
        salu(b, "s_cmp_lt_i32 s13, s10", &[], &[S_ROW, S_M])?;
        branch(b, &format!("s_cbranch_scc0 {single}"))?;
        for col in 0..nb {
            for r in [2 * p, 2 * p + 1] {
                let (y, a) = (l.y_tmp(col, r), l.y_addr(col));
                vmem(b, &format!("global_load_b32 v{y}, v{a}, s[8:9]{}", row_off(r)), &[y], &[a], &[8, 9], false)?;
            }
        }
        for col in 0..nb {
            for r in [2 * p, 2 * p + 1] {
                let (y, a, v) = (l.y_tmp(col, r), l.y_addr(col), l.acc(col, SLOT_A, r));
                valu(b, &format!("v_add_f32_e32 v{y}, v{v}, v{y}"), &[y], &[v, y], &[])?;
                vmem(b, &format!("global_store_b32 v{a}, v{y}, s[8:9]{}", row_off(r)), &[], &[a, y], &[8, 9], true)?;
            }
        }
        b.wait_all()?;
        if p + 1 < l.pairs() {
            branch(b, &format!("s_branch .Lrx_b{nb}_p{}_next", p))?;
        } else {
            branch(b, &format!("s_branch {done}"))?;
        }
        b.label(&single)?;
        let r = 2 * p;
        for col in 0..nb {
            let (y, a) = (l.y_tmp(col, r), l.y_addr(col));
            vmem(b, &format!("global_load_b32 v{y}, v{a}, s[8:9]{}", row_off(r)), &[y], &[a], &[8, 9], false)?;
        }
        for col in 0..nb {
            let (y, a, v) = (l.y_tmp(col, r), l.y_addr(col), l.acc(col, SLOT_A, r));
            valu(b, &format!("v_add_f32_e32 v{y}, v{y}, v{v}"), &[y], &[y, v], &[])?;
            vmem(b, &format!("global_store_b32 v{a}, v{y}, s[8:9]{}", row_off(r)), &[], &[a, y], &[8, 9], true)?;
        }
        b.wait_all()?;
        branch(b, &format!("s_branch {done}"))?;
        if p + 1 < l.pairs() {
            b.label(&format!(".Lrx_b{nb}_p{p}_next"))?;
        }
    }
    b.label(&done)?;
    branch(b, "s_branch .Lrx_end")
}

/// Byte offset of row `r` from the workgroup's first row in a y column.
fn row_off(r: u8) -> String {
    if r == 0 { String::new() } else { format!(" offset:{}", 4 * u32::from(r)) }
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

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;
    #[test]
    fn residual_xbatch_m7() {
        for nb in 1..=MAX_COLUMNS {
            let l = Layout::new(nb);
            // x_n one bank past even-row w_n, odd-row w_n two banks past.
            assert_eq!((l.x(0) % 4, l.w(0) % 4, l.w(1) % 4), (1, 0, 2), "B={nb}");
        }
        let emitted = build_gfx1201().expect("residual xbatch").remove(0);
        // B=8 (4 rows): row1 seeds x1 then x0 on the shared x src1.
        let l = Layout::new(8);
        let (w0, w1, x) = (l.w(0), l.w(1), l.x(0));
        let [d0, d1] = l.dot(0, 0);
        assert!(emitted.s_text.contains(&format!("v_dual_mul_f32 v{d0}, v{w0}, v{x} :: v_dual_mul_f32 v{d1}, v{}, v{}", w1 + 1, x + 1)));
        assert!(emitted.s_text.contains(&format!("v_dual_fmac_f32 v{d0}, v{}, v{} :: v_dual_fmac_f32 v{d1}, v{w1}, v{x}", w0 + 1, x + 1)));
        let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1201).expect("native assemble");
        let dir = std::env::temp_dir()
            .join(format!("hipfire-isa-residual-xbatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gemv_hfq4g256_residual_xbatch_mq4v2.test.s"), &emitted.s_text).unwrap();
        let co = dir.join("gemv_hfq4g256_residual_xbatch_mq4v2.test.co");
        std::fs::write(&co, &elf).unwrap();
        let report = crate::pm_check::m7(&co, "gfx1201", SYMBOL).expect("M7");
        println!("{report}");
        assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
    }
}
