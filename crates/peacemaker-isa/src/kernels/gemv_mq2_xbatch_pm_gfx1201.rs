// SPDX-License-Identifier: Apache-2.0
//! PeaceMaker x-batched MQ2G256V2 GEMV for gfx1201 (WideNet F05 `r4_b4_u1_native_anchor`, shipped as
//! `kernels/gemv_mq2_xbatch_pm_gfx1201.hxaco`): byte-exact PM twin and bounded variants of `gemv_mq2g256v2_xbatch`
//! (`kernels/src/gemv_mq2g256v2.hip`; the MTP draft-head ranking GEMV). The hipcc anchor itself is fixed at
//! `XBATCH_MAX = 4` (it clamps `B > 4` to 4); this emitter admits capacities up to [`XBATCH_MAX`] = 8.
//!
//! The product PM kernel is the [`SHIPPED`] tuple's committed `.hxaco`, pinned byte-for-byte by
//! `tests/gemv_mq2_xbatch_pm.rs`. Every other tuple (in particular any `cap` 8 tuple) is lab output until a
//! measured selection replaces [`SHIPPED`] and re-pins the artifact.
//!
//! ABI (36 B, identical to the anchor): `A` (read-only), `x[B][K]` (read-only), `y[B][M]` (write-only)
//! pointers at 0/8/16, `M`/`K`/`B` i32 at 24/28/32. Anchor launch: wave32, block 32, grid `M`, no LDS/scratch.
//! `A` is `M` rows of `K/256` 72-byte MQ2G256V2 groups: two `(scale, zero-point)` f16 pairs (lanes 0..15 use
//! the first dword, lanes 16..31 the second), then 64 payload bytes. Lane `l` owns the 8 two-bit fields in
//! the low 16 bits of the dword at payload byte `2l` (an unaligned 2-byte-stride dword read, exactly the
//! anchor's; the upper 16 bits are never used, but the last lane's dword spills 2 bytes past its group).
//!
//! The per-row arithmetic DAG is the anchor's (lifted from the actual hipcc gfx1201 object, AMD clang 24):
//! `w_n = fma_mix(sc_f16, float(q_n), zp_f16)`; a group dot is `w1*x1` (mul), then fmac `w0*x0`, `w2*x2`
//! .. `w7*x7`; group `g` accumulates `acc[g % 4] += dot` in increasing `g` (quads, then scalar tail groups
//! into streams 0..2); the row result is `(a0 + a1) + (a2 + a3)` followed by the frozen wave32 shfl_down
//! tree (swizzle `1pppp`, then self-reading bpermutes 8/4/2/1, each `own + other`); lane 0 stores
//! `y[b*M + row]`. No reassociation, no FMA-contraction change. `B > cap` computes `cap` columns (the hipcc anchor
//! is cap 4) and `B <= 0` stores nothing, as the anchor does.
//!
//! Grammar (`rows x cap x unroll x split x ilp` tuples, canonical order of
//! `(rows, cap, unroll, load, schedule)`): rows per workgroup `{1,2,4}` (workgroup `w` owns rows
//! `R*w..`, loads of rows past `M` clamp to `M-1`, stores only valid rows; launch grid `ceil(M/R)`);
//! x-batch capacity `{1,2,4,8}` (code paths for `B = 1..=cap`; the kernel clamps `B > cap`; column offsets use
//! `s28..s35`);
//! quads per loop trip `{1,2,4}` (main loop, then a one-quad remainder loop; tail groups unchanged);
//! activation load decomposition `{native 2xb128, split 4xb64}` (same bytes, same registers);
//! schedule `{anchor, quad_ILP}` (`quad_ILP` decodes group pairs `(0,1),(2,3)` into two weight sets and
//! interleaves their dependent fmac chains term by term; each chain's own order is unchanged).
//! The twin of the hipcc anchor is `(rows 1, cap 4, unroll 1, native, anchor)`: launch-identical to it.
//! Not every tuple closes: [`Params::validate`] checks the grid, then the builder rejects tuples whose ring/register
//! plan does not fit (e.g. `r4_b8` with `quad_ILP` has 1 ring slot but needs 4; `r4_b8` anchor schedule has 6).
//!
//! Domain of the byte-exact claim: `K % 256 == 0`, `K >= 0`, `M >= 1`, `A` addressed by a 32-bit
//! row/group byte offset (`< 4 GiB`), `y` offsets `B*M*4 < 4 GiB` (the real FN shape is 178.8 MB / 3.0 MB).

use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Kind, Live};
use crate::vopd::{Operand, VopdF32, VopdOp};
use crate::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};

pub const SYMBOL: &str = "gemv_mq2g256v2_xbatch_pm";
pub const ANCHOR_SYMBOL: &str = "gemv_mq2g256v2_xbatch";
pub const XBATCH_MAX: u8 = 8;
pub const GROUP_BYTES: u32 = 72;
const VGPR_BUDGET: u16 = 256;

/// The ordered arithmetic DAG every tuple of this family preserves (hashed into `ordered_dag_hash`).
pub const ORDERED_DAG: &str = "gemv_mq2g256v2_xbatch/gfx1201/v1;\
w_n=fma_mix(f16lo(h),f32(q_n),f16hi(h)) h=lane<16?hdr0:hdr1 q_n=(pk>>2n)&3 pk=u32@group+8+2*lane;\
dot=mul(w1,x1);dot=fmac(w0,x0);dot=fmac(w2,x2)..fmac(w7,x7);acc[g%4]=acc[g%4]+dot;g ascending(quads then tail);\
row=(a0+a1)+(a2+a3);reduce=swizzle(1pppp);then bpermute 8,4,2,1 each own+other;lane0 store y[b*M+row];B>8=>8";

/// One point of the finite variant grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Params {
    /// output rows per workgroup (`1`, `2`, `4`)
    pub rows: u8,
    /// x-batch capacity (`1`, `2`, `4`, `8`)
    pub cap: u8,
    /// quads per main-loop trip (`1`, `2`, `4`)
    pub unroll: u8,
    /// activation loads as 4xb64 instead of 2xb128
    pub split: bool,
    /// `quad_ILP` schedule
    pub ilp: bool,
}

impl Params {
    /// Canonical tuple string.
    pub fn tag(&self) -> String {
        format!(
            "r{}_b{}_u{}_{}_{}",
            self.rows,
            self.cap,
            self.unroll,
            if self.split { "split" } else { "native" },
            if self.ilp { "quadilp" } else { "anchor" }
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if ![1, 2, 4].contains(&self.rows) {
            return Err(format!("rows {} not in {{1,2,4}}", self.rows));
        }
        if ![1, 2, 4, 8].contains(&self.cap) {
            return Err(format!("cap {} not in {{1,2,4,8}}", self.cap));
        }
        if ![1, 2, 4].contains(&self.unroll) {
            return Err(format!("unroll {} not in {{1,2,4}}", self.unroll));
        }
        Ok(())
    }
}

/// The shipped tuple `r4_b4_u1_native_anchor`.
pub const SHIPPED: Params = Params {
    rows: 4,
    cap: 4,
    unroll: 1,
    split: false,
    ilp: false,
};

/// Host entry spelling of the committed `.hxaco` bundle (`clang-offload-bundler` form).
pub const HOST_TARGET: &str = crate::native::DEFAULT_HOST_TARGET;

/// First SGPR index the body never touches.
pub const SGPRS_USED: u8 = 44;

/// The complete code-object ELF of one tuple.
pub fn emit_elf(params: &Params) -> Result<Vec<u8>, String> {
    params.validate()?;
    let built = build(params)?;
    crate::native::assemble(&built.s_text, Arch::Gfx1201)
}

/// The `.hxaco` offload bundle of one tuple.
pub fn emit_bundle(params: &Params) -> Result<Vec<u8>, String> {
    Ok(crate::native::bundle(
        &emit_elf(params)?,
        Arch::Gfx1201,
        HOST_TARGET,
    ))
}

/// The CFG wait-ledger verification (`ledger_cfg::verify_emitted`) of one tuple's emitted program.
pub fn verify_waits(params: &Params) -> Result<crate::ledger_cfg::Stats, String> {
    params.validate()?;
    crate::ledger_cfg::verify_emitted(&build(params)?)
}

/// The assembly text of one tuple.
pub fn emit_text(params: &Params) -> Result<String, String> {
    params.validate()?;
    Ok(build(params)?.s_text)
}

/// The assembly text of one tuple with body labels `{prefix}_...`.
pub fn emit_text_prefixed(params: &Params, prefix: &str) -> Result<String, String> {
    params.validate()?;
    Ok(build_with(params, prefix)?.s_text)
}

/// Workgroup-index source of the body's `s_lshl_b32 s12, <wg>, k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgId {
    Ttmp9,
    Sgpr(u8),
}

/// The kernel spec of one tuple.
pub fn kernel_spec(p: &Params) -> KernelSpec {
    KernelSpec {
        kernel_id: "gemv_hfq2g256_xbatch_mq2v2".into(),
        variant: p.tag(),
        arch: Arch::Gfx1201,
        symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(36)
            .pointer_access("A", 0, crate::plan::Access::ReadOnly)
            .pointer_access("x", 8, crate::plan::Access::ReadOnly)
            .pointer_access("y", 16, crate::plan::Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("B", 32, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// First VGPR past every path's accumulators.
pub fn vgpr_top(p: &Params) -> u16 {
    Lay::new(p).top()
}

// ------------------------------------------------------------------------------------------------ layout

const LANE: u8 = 0;
/// `(lane >> 4) * 4`: the lane's half-header byte offset.
const HDR_OFF: u8 = 1;
/// `lane * 2`: the lane's payload byte offset (plus 8 immediate).
const PK_OFF: u8 = 2;
/// `lane * 32`: the lane's eight activations.
const X_OFF: u8 = 3;
const SHUF_ADDR: u8 = 4;

// SGPRs.
const S_B: u8 = 3;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_NV: u8 = 13;
const S_GROUPS: u8 = 14;
const S_STRIDE: u8 = 15;
fn s_row(r: u8) -> u8 {
    16 + r
}
fn s_xcol(c: u8) -> u8 {
    28 + c
}
const S_QUADS: u8 = 36;
const S_TAIL: u8 = 37;
const S_KX4: u8 = 38;
const S_MX4: u8 = 39;
const S_T0: u8 = 40;
const S_T1: u8 = 41;
const S_ROW0X4: u8 = 42;
const S_MAIN: u8 = 43;

/// Register map of one tuple.
struct Lay {
    r: u16,
    cap: u16,
    pk_base: u16,
    wb: u16,
    set_sz: u16,
    ring: u16,
    /// activation loads per ring item
    lpi: u16,
}

impl Lay {
    fn new(p: &Params) -> Self {
        let r = u16::from(p.rows);
        let pk_base = 8 + 4 * r;
        let wb = (pk_base + 4 * r).next_multiple_of(4);
        let set_sz = (10 * r).next_multiple_of(4);
        let sets = if p.ilp { 2 } else { 1 };
        Self {
            r,
            cap: u16::from(p.cap),
            pk_base,
            wb,
            set_sz,
            ring: (wb + sets * set_sz).next_multiple_of(8),
            lpi: if p.split { 4 } else { 2 },
        }
    }
    /// Half header of (group slot j, row r).
    fn hdr(&self, j: u8, r: u8) -> u8 {
        (8 + self.r * u16::from(j) + u16::from(r)) as u8
    }
    /// Payload dword of (group slot j, row r).
    fn pk(&self, j: u8, r: u8) -> u8 {
        (self.pk_base + self.r * u16::from(j) + u16::from(r)) as u8
    }
    /// Dequantized weight `n` of (weight set, row r). Row blocks are 10 apart so the two rows of a VOPD pair
    /// have distinct src0 banks; the two spare registers of a block hold dots.
    fn w(&self, set: u8, r: u8, n: u8) -> u8 {
        (self.wb + u16::from(set) * self.set_sz + 10 * u16::from(r) + u16::from(n)) as u8
    }
    /// Dot register of (weight set, column slot, row r); the rows of a pair are even/odd.
    fn dot(&self, set: u8, slot: u8, r: u8) -> u8 {
        let base = self.wb + u16::from(set) * self.set_sz;
        let (slot, r) = (u16::from(slot), u16::from(r));
        let off = match self.r {
            4 => 10 * (2 * slot + r / 2) + 8 + r % 2,
            2 => 10 * slot + 8 + r,
            _ => 8 + slot,
        };
        (base + off) as u8
    }
    /// Ring slots (8 VGPRs each) of the `nb`-column path: bounded by the items of one quad, the registers
    /// left after the accumulators and the 6-bit load counter (weight loads plus ring loads in flight).
    fn slots(&self, nb: u8) -> u16 {
        let nb = u16::from(nb);
        let regs = (VGPR_BUDGET - self.ring - 4 * self.r * nb) / 8;
        let loads = (63 - 8 * self.r) / self.lpi;
        (4 * nb).min(regs).min(loads)
    }
    fn ring(&self, nb: u8, item: u16) -> u8 {
        (self.ring + 8 * (item % self.slots(nb))) as u8
    }
    fn acc_base(&self, nb: u8) -> u16 {
        self.ring + 8 * self.slots(nb)
    }
    /// Stream accumulator (column c, stream s, row r) of the `nb`-column path.
    fn acc(&self, nb: u8, c: u8, s: u8, r: u8) -> u8 {
        (self.acc_base(nb) + 4 * self.r * u16::from(c) + self.r * u16::from(s) + u16::from(r)) as u8
    }
    fn red_tmp(&self, c: u8, r: u8) -> u8 {
        (8 + self.r * u16::from(c) + u16::from(r)) as u8
    }
    fn y_addr(&self, c: u8) -> u8 {
        (self.wb + u16::from(c)) as u8
    }
    /// First VGPR past every path's accumulators.
    fn top(&self) -> u16 {
        (1..=self.cap)
            .map(|nb| self.acc_base(nb as u8) + 4 * self.r * nb)
            .max()
            .unwrap_or(self.ring)
    }
    fn rowpairs(&self) -> Vec<(u8, Option<u8>)> {
        match self.r {
            4 => vec![(0, Some(1)), (2, Some(3))],
            2 => vec![(0, Some(1))],
            _ => vec![(0, None)],
        }
    }
}

// ------------------------------------------------------------------------------------------------ builder

/// Reject tuples whose ring/register plan does not close.
fn check(p: &Params, lay: &Lay) -> Result<(), String> {
    // The ring must hold one step (ILP: two groups x up to two columns) and the plan must fit.
    for nb in 1..=p.cap {
        let step = if p.ilp { 2 * nb.min(2) } else { nb.min(2) };
        if lay.slots(nb) < u16::from(step) {
            return Err(format!(
                "nb={nb}: {} ring slots cannot hold a {step}-item step",
                lay.slots(nb)
            ));
        }
    }
    let top = lay.top();
    if top > VGPR_BUDGET {
        return Err(format!("needs {top} VGPRs (> {VGPR_BUDGET})"));
    }
    Ok(())
}

/// Adds every register range the body uses. `live` applies to every body-local range; `with_inputs`
/// adds the input ranges (`kernarg` s[0:1] and `lane` v0, always `Live::Whole`).
pub fn plan_ranges(
    regs: &mut RegPlan,
    p: &Params,
    live: Live,
    with_inputs: bool,
) -> Result<(), String> {
    p.validate()?;
    let lay = Lay::new(p);
    check(p, &lay)?;
    let top = lay.top();
    if with_inputs {
        regs.add_range("kernarg", Kind::S, 0, 2, Live::Whole)?;
    }
    for (name, base, width) in [
        ("columns", S_B, 1),
        ("matrix_activation", 4, 4),
        ("output", 8, 2),
        ("m", S_M, 1),
        ("k", S_K, 1),
        ("row0", S_ROW0, 1),
        ("valid_rows", S_NV, 1),
        ("groups", S_GROUPS, 1),
        ("stride", S_STRIDE, 1),
        ("row_offsets", 16, 4),
        ("weight_resource", 20, 4),
        ("activation_resource", 24, 4),
        ("column_offsets0", 28, 4),
        ("column_offsets1", 32, 4),
        ("quads", S_QUADS, 1),
        ("tail", S_TAIL, 1),
        ("k_x4", S_KX4, 1),
        ("m_x4", S_MX4, 1),
        ("temp0", S_T0, 1),
        ("temp1", S_T1, 1),
        ("row0_x4", S_ROW0X4, 1),
        ("main_trips", S_MAIN, 1),
    ] {
        regs.add_range(name, Kind::S, base, width, live.clone())?;
    }
    let r = lay.r;
    let sets = if p.ilp { 2 } else { 1 };
    let mut vranges: Vec<(String, u16, u16)> = Vec::new();
    if with_inputs {
        vranges.push(("lane".into(), 0, 1));
    }
    vranges.extend([
        ("header_offset".into(), 1, 1),
        ("packed_offset".into(), 2, 1),
        ("activation_offset".into(), 3, 1),
        ("shuffle_addr".into(), 4, 1),
        ("headers".into(), 8, 4 * r),
        ("packed".into(), lay.pk_base, 4 * r),
        ("ring_accumulators".into(), lay.ring, top - lay.ring),
    ]);
    for s in 0..sets {
        vranges.push((
            format!("weights_set{s}"),
            lay.wb + s * lay.set_sz,
            lay.set_sz,
        ));
    }
    for (name, base, width) in vranges {
        let whole_input = name == "lane";
        let mut at = 0u16;
        while at < width {
            let mut piece = (width - at).min(8);
            // the allocator admits widths 1/2/4/8
            while !piece.is_power_of_two() {
                piece -= 1;
            }
            regs.add_range(
                &format!("{name}{at}"),
                Kind::V,
                (base + at) as u8,
                piece as u8,
                if whole_input {
                    Live::Whole
                } else {
                    live.clone()
                },
            )?;
            at += piece;
        }
    }
    Ok(())
}

fn build(p: &Params) -> Result<crate::Emitted, String> {
    build_with(p, ".Lpx")
}

fn build_with(p: &Params, prefix: &str) -> Result<crate::Emitted, String> {
    p.validate()?;
    let mut regs = RegPlan::new(VGPR_BUDGET, 48)?;
    plan_ranges(&mut regs, p, Live::Whole, true)?;
    let mut b = Builder::new(kernel_spec(p), regs);
    emit_body(&mut b, p, WgId::Ttmp9, prefix)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Everything from the first `s_load_b64 s[10:11], s[0:1], 0x18` through the final `{prefix}_end` label,
/// without the trailing `s_endpgm`. Labels are `{prefix}_end`, `{prefix}_b{nb}...`.
pub fn emit_body(b: &mut Builder, p: &Params, wg: WgId, prefix: &str) -> Result<(), String> {
    p.validate()?;
    let lay = Lay::new(p);
    check(p, &lay)?;
    let pre = prefix;
    let rows = p.rows;
    smem(b, "s_load_b64 s[10:11], s[0:1], 0x18", &[S_M, S_K], &[0, 1])?;
    // gfx12 receives the workgroup index in ttmp9 (or the given SGPR when embedded).
    match wg {
        WgId::Ttmp9 => salu(
            b,
            &format!("s_lshl_b32 s12, ttmp9, {}", rows.trailing_zeros()),
            &[S_ROW0],
            &[],
        )?,
        WgId::Sgpr(n) => salu(
            b,
            &format!("s_lshl_b32 s12, s{n}, {}", rows.trailing_zeros()),
            &[S_ROW0],
            &[n],
        )?,
    }
    b.wait_all()?;
    salu(b, "s_cmp_ge_i32 s12, s10", &[], &[S_ROW0, S_M])?;
    if hops(p) {
        // `row0 >= M` leaves through the first exit hop, one section away, not a far `_end`.
        branch(b, &format!("s_cbranch_scc0 {pre}_in"))?;
        branch(b, &format!("s_branch {}", exit_label(p, pre, 1)))?;
        b.label(&format!("{pre}_in"))?;
    } else {
        branch(b, &format!("s_cbranch_scc1 {pre}_end"))?;
    }
    smem(b, "s_load_b128 s[4:7], s[0:1], 0x0", &[4, 5, 6, 7], &[0, 1])?;
    smem(b, "s_load_b64 s[8:9], s[0:1], 0x10", &[8, 9], &[0, 1])?;
    smem(b, "s_load_b32 s3, s[0:1], 0x20", &[S_B], &[0, 1])?;
    salu(b, "s_sub_co_i32 s13, s10, s12", &[S_NV], &[S_M, S_ROW0])?;
    salu(b, &format!("s_min_i32 s13, s13, {rows}"), &[S_NV], &[S_NV])?;
    salu(b, "s_add_co_i32 s40, s10, -1", &[S_T0], &[S_M])?;
    salu(b, "s_lshr_b32 s14, s11, 8", &[S_GROUPS], &[S_K])?;
    salu(b, "s_mul_i32 s15, s14, 0x48", &[S_STRIDE], &[S_GROUPS])?;
    for row in 0..rows {
        salu(
            b,
            &format!("s_add_co_i32 s41, s12, {row}"),
            &[S_T1],
            &[S_ROW0],
        )?;
        salu(b, "s_min_i32 s41, s41, s40", &[S_T1], &[S_T1, S_T0])?;
        salu(
            b,
            &format!("s_mul_i32 s{}, s41, s15", s_row(row)),
            &[s_row(row)],
            &[S_T1, S_STRIDE],
        )?;
    }
    salu(b, "s_lshl_b32 s38, s11, 2", &[S_KX4], &[S_K])?;
    salu(b, "s_lshl_b32 s39, s10, 2", &[S_MX4], &[S_M])?;
    salu(b, "s_lshl_b32 s42, s12, 2", &[S_ROW0X4], &[S_ROW0])?;
    for c in 0..p.cap {
        salu(
            b,
            &format!("s_mul_i32 s{}, s38, {c}", s_xcol(c)),
            &[s_xcol(c)],
            &[S_KX4],
        )?;
    }
    valu(b, "v_lshrrev_b32_e32 v1, 4, v0", &[HDR_OFF], &[LANE], &[])?;
    valu(
        b,
        "v_lshlrev_b32_e32 v1, 2, v1",
        &[HDR_OFF],
        &[HDR_OFF],
        &[],
    )?;
    valu(b, "v_lshlrev_b32_e32 v2, 1, v0", &[PK_OFF], &[LANE], &[])?;
    valu(b, "v_lshlrev_b32_e32 v3, 5, v0", &[X_OFF], &[LANE], &[])?;
    b.wait_all()?;
    // Raw buffer resources (stride 0, unbounded records) over A and x.
    for (res, ptr) in [(20u8, 4u8), (24, 6)] {
        salu(b, &format!("s_mov_b32 s{res}, s{ptr}"), &[res], &[ptr])?;
        salu(
            b,
            &format!("s_and_b32 s{}, s{}, 0xffff", res + 1, ptr + 1),
            &[res + 1],
            &[ptr + 1],
        )?;
        salu(b, &format!("s_mov_b32 s{}, -1", res + 2), &[res + 2], &[])?;
        salu(
            b,
            &format!("s_mov_b32 s{}, 0x31004000", res + 3),
            &[res + 3],
            &[],
        )?;
    }
    // B > cap clamps to cap (the hipcc anchor clamps B > 4 to 4); B <= 0 matches no path and stores nothing.
    salu(b, &format!("s_min_i32 s3, s3, {}", p.cap), &[S_B], &[S_B])?;
    if hops(p) {
        // Distributed gates: the code is larger than one `s_branch` reach (+-128 KiB), so every branch spans at
        // most one section. Gate `nb` skips its section to gate `nb+1` (the last to `_end`) unless `B == nb`; a
        // finished section falls into its exit hop, a chain of one-section jumps to `_end`.
        for nb in 1..=p.cap {
            if nb > 1 {
                b.label(&format!("{pre}_d{nb}"))?;
            }
            salu(b, &format!("s_cmp_lg_u32 s3, {nb}"), &[], &[S_B])?;
            let next = if nb < p.cap {
                format!("{pre}_d{}", nb + 1)
            } else {
                format!("{pre}_end")
            };
            branch(b, &format!("s_cbranch_scc1 {next}"))?;
            section(b, p, &lay, nb, pre)?;
            if nb < p.cap {
                b.label(&format!("{pre}_x{nb}"))?;
                branch(b, &format!("s_branch {}", exit_label(p, pre, nb + 1)))?;
            }
        }
        return b.label(&format!("{pre}_end"));
    }
    for nb in 1..=p.cap {
        salu(b, &format!("s_cmp_eq_u32 s3, {nb}"), &[], &[S_B])?;
        branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}"))?;
    }
    branch(b, &format!("s_branch {pre}_end"))?;
    for nb in 1..=p.cap {
        section(b, p, &lay, nb, pre)?;
    }
    b.label(&format!("{pre}_end"))
}

/// Capacities above 4 outgrow one branch reach; the body uses the distributed-gate layout ([`emit_body`]).
/// Capacities <= 4 keep the pinned shipped layout.
fn hops(p: &Params) -> bool {
    p.cap > 4
}

/// Entry of the exit-hop chain at section `nb`: the hop after section `nb`, or `_end` past the last section.
fn exit_label(p: &Params, pre: &str, nb: u8) -> String {
    if nb >= p.cap {
        format!("{pre}_end")
    } else {
        format!("{pre}_x{nb}")
    }
}

/// The complete kernel body for `nb` columns.
fn section(b: &mut Builder, p: &Params, lay: &Lay, nb: u8, pre: &str) -> Result<(), String> {
    b.label(&format!("{pre}_b{nb}"))?;
    for c in 0..nb {
        for s in 0..4 {
            for r in 0..p.rows {
                let a = lay.acc(nb, c, s, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    salu(b, "s_lshr_b32 s36, s14, 2", &[S_QUADS], &[S_GROUPS])?;
    salu(b, "s_and_b32 s37, s14, 3", &[S_TAIL], &[S_GROUPS])?;
    let advance = |b: &mut Builder| -> Result<(), String> {
        for r in 0..p.rows {
            salu(
                b,
                &format!("s_add_co_i32 s{0}, s{0}, 0x120", s_row(r)),
                &[s_row(r)],
                &[s_row(r)],
            )?;
        }
        for c in 0..nb {
            salu(
                b,
                &format!("s_add_co_i32 s{0}, s{0}, 0x1000", s_xcol(c)),
                &[s_xcol(c)],
                &[s_xcol(c)],
            )?;
        }
        Ok(())
    };
    let quad_body = |b: &mut Builder| -> Result<(), String> {
        body(b, p, lay, nb, &[0, 1, 2, 3])?;
        advance(b)
    };
    if p.unroll > 1 {
        // main loop: quads / unroll trips of `unroll` quads, then the quads % unroll remainder.
        salu(
            b,
            &format!(
                "s_lshr_b32 s{S_MAIN}, s{S_QUADS}, {}",
                p.unroll.trailing_zeros()
            ),
            &[S_MAIN],
            &[S_QUADS],
        )?;
        salu(
            b,
            &format!("s_and_b32 s{S_QUADS}, s{S_QUADS}, {}", p.unroll - 1),
            &[S_QUADS],
            &[S_QUADS],
        )?;
        salu(b, &format!("s_cmp_eq_u32 s{S_MAIN}, 0"), &[], &[S_MAIN])?;
        branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}_rem"))?;
        b.loop_(&format!("{pre}_b{nb}_main"), |b| {
            for _ in 0..p.unroll {
                quad_body(b)?;
            }
            salu(
                b,
                &format!("s_add_co_i32 s{S_MAIN}, s{S_MAIN}, -1"),
                &[S_MAIN],
                &[S_MAIN],
            )?;
            salu(b, &format!("s_cmp_lg_u32 s{S_MAIN}, 0"), &[], &[S_MAIN])?;
            branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}_main"))
        })?;
        b.label(&format!("{pre}_b{nb}_rem"))?;
    }
    salu(b, "s_cmp_eq_u32 s36, 0", &[], &[S_QUADS])?;
    branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}_tail"))?;
    b.loop_(&format!("{pre}_b{nb}_quad"), |b| {
        quad_body(b)?;
        salu(b, "s_add_co_i32 s36, s36, -1", &[S_QUADS], &[S_QUADS])?;
        salu(b, "s_cmp_lg_u32 s36, 0", &[], &[S_QUADS])?;
        branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}_quad"))
    })?;
    b.label(&format!("{pre}_b{nb}_tail"))?;
    // Tail group t (at the post-quad bases) accumulates into stream t.
    for t in 0..3u8 {
        salu(b, &format!("s_cmp_gt_u32 s37, {t}"), &[], &[S_TAIL])?;
        branch(b, &format!("s_cbranch_scc0 {pre}_b{nb}_fold"))?;
        body(b, p, lay, nb, &[t])?;
    }
    b.label(&format!("{pre}_b{nb}_fold"))?;
    fold(b, p, lay, nb)?;
    reduce(b, p, lay, nb)?;
    store(b, p, lay, nb, pre)
}

/// One column-step: the groups (weight set, group slot) dotted against up to two columns
/// (each `(column, activation item)`).
struct GroupStep {
    set: u8,
    j: u8,
    cols: Vec<(u8, u16)>,
}

/// Groups `js` (slot = stream = immediate index) of the rows against every column. Weights of the first
/// group and its activations issue first, then the remaining weights; activations of later items stream
/// through the ring as earlier items retire. `quad_ILP` decodes consecutive group pairs into two weight
/// sets and interleaves their fmac chains term by term.
fn body(b: &mut Builder, p: &Params, lay: &Lay, nb: u8, js: &[u8]) -> Result<(), String> {
    let packets: Vec<Vec<u8>> = if p.ilp {
        js.chunks(2).map(<[u8]>::to_vec).collect()
    } else {
        js.iter().map(|&j| vec![j]).collect()
    };
    // Items (group slot, column) in consumption order, and the steps consuming them.
    let mut items: Vec<(u8, u8)> = Vec::new();
    let mut plan: Vec<(Vec<(u8, u8)>, Vec<Vec<GroupStep>>)> = Vec::new();
    for packet in &packets {
        let groups: Vec<(u8, u8)> = packet
            .iter()
            .enumerate()
            .map(|(set, &j)| (set as u8, j))
            .collect();
        let mut steps = Vec::new();
        let mut c = 0u8;
        while c < nb {
            let mut step = Vec::new();
            for &(set, j) in &groups {
                let mut cols = Vec::new();
                for col in c..nb.min(c + 2) {
                    items.push((j, col));
                    cols.push((col, (items.len() - 1) as u16));
                }
                step.push(GroupStep { set, j, cols });
            }
            steps.push(step);
            c += 2;
        }
        plan.push((groups, steps));
    }
    let total = items.len() as u16;
    let slots = lay.slots(nb);
    weight_loads(b, lay, p.rows, js[0])?;
    let mut issued = 0u16;
    while issued < slots.min(u16::from(nb)) {
        x_load(b, p, lay, nb, &items, issued)?;
        issued += 1;
    }
    for &j in &js[1..] {
        weight_loads(b, lay, p.rows, j)?;
    }
    for (groups, steps) in &plan {
        for &(set, j) in groups {
            dequant(b, lay, p.rows, set, j)?;
        }
        for step in steps {
            let first = step[0].cols[0].1;
            while issued < (first + slots).min(total) {
                x_load(b, p, lay, nb, &items, issued)?;
                issued += 1;
            }
            for term in 0..8usize {
                for g in step {
                    dot_term(b, lay, nb, g, term)?;
                }
            }
            for g in step {
                accumulate(b, lay, nb, g)?;
            }
        }
    }
    b.wait_all()
}

/// Lane-selected half headers and payload dwords of group slot `j`, all rows.
fn weight_loads(b: &mut Builder, lay: &Lay, rows: u8, j: u8) -> Result<(), String> {
    let off = u32::from(j) * GROUP_BYTES;
    for r in 0..rows {
        let h = lay.hdr(j, r);
        let suffix = if off == 0 {
            String::new()
        } else {
            format!(" offset:{off}")
        };
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{h}, v1, s[20:23], s{} offen{suffix}",
                s_row(r)
            ),
            &[h],
            &[HDR_OFF],
            &[20, 21, 22, 23, s_row(r)],
            false,
        )?;
    }
    for r in 0..rows {
        let pk = lay.pk(j, r);
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{pk}, v2, s[20:23], s{} offen offset:{}",
                s_row(r),
                off + 8
            ),
            &[pk],
            &[PK_OFF],
            &[20, 21, 22, 23, s_row(r)],
            false,
        )?;
    }
    Ok(())
}

/// The eight activations of item `item` (group `items[item].0`, column `items[item].1`) into its ring slot.
fn x_load(
    b: &mut Builder,
    p: &Params,
    lay: &Lay,
    nb: u8,
    items: &[(u8, u8)],
    item: u16,
) -> Result<(), String> {
    let (j, c) = items[usize::from(item)];
    let x = lay.ring(nb, item);
    let pieces: u8 = if p.split { 4 } else { 2 };
    let width: u8 = if p.split { 2 } else { 4 };
    for piece in 0..pieces {
        let off = u32::from(j) * 1024 + u32::from(piece) * u32::from(width) * 4;
        let suffix = if off == 0 {
            String::new()
        } else {
            format!(" offset:{off}")
        };
        let d = x + width * piece;
        let (mnemonic, regs): (&str, Vec<u8>) = if p.split {
            ("b64", vec![d, d + 1])
        } else {
            ("b128", vec![d, d + 1, d + 2, d + 3])
        };
        vmem(
            b,
            &format!(
                "buffer_load_{mnemonic} v[{d}:{}], v3, s[24:27], s{} offen{suffix}",
                d + width - 1,
                s_xcol(c)
            ),
            &regs,
            &[X_OFF],
            &[24, 25, 26, 27, s_xcol(c)],
            false,
        )?;
    }
    Ok(())
}

/// `w[r][n] = fma_mix(h, float((pk >> 2n) & 3), h)` (scale = low f16 of `h`, zero point = high f16) for every
/// row of group slot `j`, into weight set `set`.
fn dequant(b: &mut Builder, lay: &Lay, rows: u8, set: u8, j: u8) -> Result<(), String> {
    for n in 0..8u8 {
        for r in 0..rows {
            let (w, pk) = (lay.w(set, r, n), lay.pk(j, r));
            valu(
                b,
                &format!("v_bfe_u32 v{w}, v{pk}, {}, 2", 2 * n),
                &[w],
                &[pk],
                &[],
            )?;
        }
    }
    for n in 0..8u8 {
        for r in 0..rows {
            let w = lay.w(set, r, n);
            valu(
                b,
                &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"),
                &[w],
                &[w],
                &[],
            )?;
        }
    }
    for n in 0..8u8 {
        for r in 0..rows {
            let (w, h) = (lay.w(set, r, n), lay.hdr(j, r));
            valu(
                b,
                &format!("v_fma_mix_f32 v{w}, v{h}, v{w}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[w],
                &[h, w],
                &[],
            )?;
        }
    }
    Ok(())
}

/// A binary f32 op over the row pair `(ra, rb)` as one VOPD packet, or a plain VALU op for a lone row.
fn pair_op(
    b: &mut Builder,
    op: VopdF32,
    pair: (u8, Option<u8>),
    dst: impl Fn(u8) -> u8,
    src0: impl Fn(u8) -> u8,
    src1: impl Fn(u8) -> u8,
) -> Result<(), String> {
    let (ra, rb) = pair;
    match rb {
        Some(rb) => b.vopd(
            VopdOp {
                op,
                dst: dst(ra),
                src0: Operand::V(src0(ra)),
                src1: src1(ra),
            },
            VopdOp {
                op,
                dst: dst(rb),
                src0: Operand::V(src0(rb)),
                src1: src1(rb),
            },
        ),
        None => {
            let (d, s0, s1) = (dst(ra), src0(ra), src1(ra));
            match op {
                VopdF32::Mul => valu(
                    b,
                    &format!("v_mul_f32_e32 v{d}, v{s0}, v{s1}"),
                    &[d],
                    &[s0, s1],
                    &[],
                ),
                VopdF32::Fmac => valu(
                    b,
                    &format!("v_fmac_f32_e32 v{d}, v{s0}, v{s1}"),
                    &[d],
                    &[d, s0, s1],
                    &[],
                ),
                VopdF32::Add => valu(
                    b,
                    &format!("v_add_f32_e32 v{d}, v{s0}, v{s1}"),
                    &[d],
                    &[s0, s1],
                    &[],
                ),
                other => Err(format!("pair_op: unsupported {other:?}")),
            }
        }
    }
}

/// Term `term` (position in the order x1, x0, x2..x7) of the dot of group step `g` against its columns.
fn dot_term(b: &mut Builder, lay: &Lay, nb: u8, g: &GroupStep, term: usize) -> Result<(), String> {
    let n = [1u8, 0, 2, 3, 4, 5, 6, 7][term];
    let op = if term == 0 {
        VopdF32::Mul
    } else {
        VopdF32::Fmac
    };
    for (slot, &(_, item)) in g.cols.iter().enumerate() {
        let x = lay.ring(nb, item);
        for pair in lay.rowpairs() {
            pair_op(
                b,
                op,
                pair,
                |r| lay.dot(g.set, slot as u8, r),
                |r| lay.w(g.set, r, n),
                |_| x + n,
            )?;
        }
    }
    Ok(())
}

/// `acc[s] = acc[s] + dot` for the group's columns.
fn accumulate(b: &mut Builder, lay: &Lay, nb: u8, g: &GroupStep) -> Result<(), String> {
    for (slot, &(c, _)) in g.cols.iter().enumerate() {
        for pair in lay.rowpairs() {
            pair_op(
                b,
                VopdF32::Add,
                pair,
                |r| lay.acc(nb, c, g.j, r),
                |r| lay.acc(nb, c, g.j, r),
                |r| lay.dot(g.set, slot as u8, r),
            )?;
        }
    }
    Ok(())
}

/// `(a0 + a1) + (a2 + a3)` per column and row, into stream 0.
fn fold(b: &mut Builder, p: &Params, lay: &Lay, nb: u8) -> Result<(), String> {
    let _ = p;
    for c in 0..nb {
        for (dst, other) in [(0u8, 1u8), (2, 3), (0, 2)] {
            for pair in lay.rowpairs() {
                pair_op(
                    b,
                    VopdF32::Add,
                    pair,
                    |r| lay.acc(nb, c, dst, r),
                    |r| lay.acc(nb, c, dst, r),
                    |r| lay.acc(nb, c, other, r),
                )?;
            }
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, p: &Params, lay: &Lay, nb: u8) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..nb)
        .flat_map(|c| (0..p.rows).map(move |r| (c, r)))
        .collect();
    for &(c, r) in &values {
        let (v, t) = (lay.acc(nb, c, 0, r), lay.red_tmp(c, r));
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                refs(Kind::V, &[t]),
                refs(Kind::V, &[v]),
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    for &(c, r) in &values {
        let (v, t) = (lay.acc(nb, c, 0, r), lay.red_tmp(c, r));
        valu(
            b,
            &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
            &[v],
            &[v, t],
            &[],
        )?;
    }
    for offset in [8u8, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset),
            &[],
            &[LANE],
            &[],
        )?;
        b.push(Instruction::new(
            "s_wait_alu depctr_va_vcc(0)",
            vec![],
            vec![],
        ))?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v4, 0, {offset}, vcc_lo"),
            &[SHUF_ADDR],
            &[],
            &[],
        )?;
        valu(
            b,
            "v_add_lshl_u32 v4, v4, v0, 2",
            &[SHUF_ADDR],
            &[SHUF_ADDR, LANE],
            &[],
        )?;
        for &(c, r) in &values {
            let (v, t) = (lay.acc(nb, c, 0, r), lay.red_tmp(c, r));
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{t}, v4, v{v}"),
                    refs(Kind::V, &[t]),
                    refs(Kind::V, &[SHUF_ADDR, v]),
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for &(c, r) in &values {
            let (v, t) = (lay.acc(nb, c, 0, r), lay.red_tmp(c, r));
            valu(
                b,
                &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
                &[v],
                &[v, t],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Lane 0 stores `y[c][row0 + r]` for every column and each valid row.
fn store(b: &mut Builder, p: &Params, lay: &Lay, nb: u8, pre: &str) -> Result<(), String> {
    valu(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[LANE], &[])?;
    for c in 0..nb {
        let a = lay.y_addr(c);
        if c == 0 {
            valu(
                b,
                &format!("v_mov_b32_e32 v{a}, s{S_ROW0X4}"),
                &[a],
                &[],
                &[S_ROW0X4],
            )?;
        } else {
            salu(b, &format!("s_mul_i32 s40, s39, {c}"), &[S_T0], &[S_MX4])?;
            salu(b, "s_add_co_i32 s40, s40, s42", &[S_T0], &[S_T0, S_ROW0X4])?;
            valu(b, &format!("v_mov_b32_e32 v{a}, s40"), &[a], &[], &[S_T0])?;
        }
    }
    for r in 0..p.rows {
        if r > 0 {
            salu(b, &format!("s_cmp_le_i32 s13, {r}"), &[], &[S_NV])?;
            branch(b, &format!("s_cbranch_scc1 {pre}_b{nb}_stored"))?;
        }
        for c in 0..nb {
            let (a, v) = (lay.y_addr(c), lay.acc(nb, c, 0, r));
            let suffix = if r == 0 {
                String::new()
            } else {
                format!(" offset:{}", 4 * r)
            };
            vmem(
                b,
                &format!("global_store_b32 v{a}, v{v}, s[8:9]{suffix}"),
                &[],
                &[a, v],
                &[8, 9],
                true,
            )?;
        }
    }
    b.label(&format!("{pre}_b{nb}_stored"))?;
    b.wait_all()?;
    if hops(p) {
        Ok(()) // falls into the exit hop (`exit_label`) that follows the section
    } else {
        branch(b, &format!("s_branch {pre}_end"))
    }
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(
        text,
        refs(Kind::S, defs),
        refs(Kind::S, uses),
    ))
}
fn valu(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses))
}
fn smem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(
        Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses))
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
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), uses).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
