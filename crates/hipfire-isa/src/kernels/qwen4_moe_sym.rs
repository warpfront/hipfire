//! Builder-emitted Qwen4 grouped symmetric IU4 MoE GEMMs (fn-moe-sym route,
//! `HIPFIRE_QWEN4_MOE_SYM_IU4=1`): the SwiGLU gate/up and the down projection
//! over packed `block_i4_128` activations, for gfx1151
//! (`v_wmma_i32_16x16x16_iu4`, eight K16 steps per K128 epoch) and gfx1201
//! (`v_wmma_i32_16x16x32_iu4`, four K32 steps).
//!
//! Same 60-byte ABI, grid, block and output bytes as the hand-written
//! `kernels/src/qwen4_moe_iu4_sym.gfx1151.hip`:
//! `(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y_grouped,
//! M, K, x_row_div, m_total, x_src_rows)`.
//! - gate/up: block 64 (two waves), grid `(M/64, m_total/16)`; wave `w` of
//!   CTA `x` owns gate rows `32x + 16w ..+16` and the up rows `M/2` further,
//!   and stores `rt(silu(rt(g)) * rt(u))` into `Y[P][M/2]` (BF16 bits).
//! - down: block 128 (four waves), grid `(M/64, m_total/16)`; wave `w` owns
//!   rows `64x + 16w ..+16` and stores `rt(sum)` into `Y[P][M]`.
//!
//! Every wave is independent: operands are loaded straight into registers
//! (the next K128 epoch prefetched while the current one is consumed), no
//! LDS and no barrier. The tile of 16 grouped slots is `blockIdx.y`; a tile
//! past `m_total` or with expert id -1 returns before any load or store.
//! Each lane loads its slot `s = sorted[16*tile + lane%16]`; a padding slot
//! (`s < 0`) gets an out-of-range buffer offset, so its activation loads
//! return 0 without touching memory, and its outputs store +0.
//!
//! Numerics, per output and ascending epoch `h`: `C_h` is the exact int32
//! WMMA chain (A nibbles rebiased with XOR 0x88888888, signed A and X)
//! seeded with the magic 0x4b400000; `sum = fma(RN(d_h * sc_h), C_h -
//! 12582912.0, sum)` from `+0`, the HIP kernel's `fma(RN(sc*d), float(C),
//! sum)` exactly (`|C_h| <= 8192`). The row scale of each accumulator comes
//! from a `ds_swizzle_b32` broadcast: lane `(hi, k)` loads the header of the
//! row its broadcast partner needs (gfx11 C rows `2j + hi`, gfx12 `8hi + j`).
//! BF16 rounding is RNE with non-finite values passed through; the SiLU is
//! the hipcc `g / (1 + expf(-g)) * u` DAG ([`crate::kernels::gemm_uk::Epilogue::silu_dense`]).
//!
//! The gfx1151 down NT4 entry also builds row-repeat variants `..._nt4r2` and
//! `..._nt4r4` (`Spec::rr` = 2 or 4; see `nt`): one CTA folds `rr` contiguous
//! 64-row blocks of the same expert run, so the grid is `(M/(64*rr), m_total/16)`
//! and CTA `x` owns rows `64*rr*x + 64*blk + 16*wave`. `rr = 1` is the frozen
//! entry, byte for byte.
use super::bf16::Bf16;
use super::common::{self, GatherTemps, add64, add64_imm, bload, bstore_b128, lit, op, s, s_add_i32, smem, sop, srd_tail, v, SRD_WORD3};
use super::iu4_fold::{self, MAGIC, REBIAS};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V, insn::{Instruction, MemoryClass}, reg::Live};
use peacemaker_author::{End, Gfx1151, Gfx1201, Scc, Target, WgUniform, Workgroup};

type Wg<'b, T> = Workgroup<'b, T, Builder>;

/// A `K`-derived trip counter: workgroup-uniform.
fn trips_cmp<T: Target>(wg: &mut Wg<T>, cmp: &str, trips: u8) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(format!("{cmp} s{trips}, 0"), vec![], vec![s(trips)]))
}

/// Run the generic kernel body `$body::<T>(&mut Workgroup<T, Builder>, ..)`
/// with `T` the target type of `$arch`.
macro_rules! on_target {
    ($arch:expr, $b:expr, $body:ident($($arg:expr),*)) => {
        match $arch {
            Arch::Gfx1151 => $body(&mut Workgroup::<Gfx1151, Builder>::new($b)?, $($arg),*),
            Arch::Gfx1201 => $body(&mut Workgroup::<Gfx1201, Builder>::new($b)?, $($arg),*),
            a => Err(format!("qwen4_moe_sym: no typed target for {}", a.name())),
        }
    };
}

pub const KERNARG_BYTES: u32 = 60;
pub const VGPR_CEILING: u16 = 256;
/// Slot indices (`x_row_div * x_src_rows`) the activation gather resolves exactly.
pub const MAX_SLOTS: u32 = common::GATHER_MAX_INDEX;
/// SiLU elements per interleaved group (two groups per lane).
const SILU_N: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind { GateUp, Down }
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    pub fn tag(self) -> &'static str { match self { Self::GateUp => "gate_up", Self::Down => "down" } }
    /// Weight tensors folded per wave (gate and up, or down).
    fn tensors(self) -> u8 { match self { Self::GateUp => 2, Self::Down => 1 } }
    /// Output columns of one CTA (`blockIdx.x` stride of the row base).
    pub fn cta_rows(self) -> u32 { match self { Self::GateUp => 32, Self::Down => 64 } }
    pub fn group_bytes(self) -> u32 { match self { Self::GateUp => 136, Self::Down => 68 } }
}
impl std::str::FromStr for Kind {
    type Err = String;
    fn from_str(t: &str) -> Result<Self, String> {
        match t { "gate_up" => Ok(Self::GateUp), "down" => Ok(Self::Down), _ => Err(format!("qwen4_moe_sym kind {t} (gate_up|down)")) }
    }
}

/// One entry: arch, GEMM kind, expert-run tile width `nt` (16-slot tiles
/// per weight stream) and row repeat `rr` (contiguous 64-row blocks folded
/// per CTA, gfx1151 down NT4 only). `nt == 1` is the frozen 16-slot entry,
/// the byte anchor of every wider tile; `rr == 1` is the anchor of every
/// row repeat.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub kind: Kind, pub nt: u8, pub rr: u8 }

/// Tile widths built per arch: the anchor and the shipped expert-run tiles
/// (gfx1151 NT4; gfx1201 NT4 for VRAM-resident, NT8 for host-mapped experts).
pub fn tile_widths(arch: Arch) -> &'static [u8] {
    match arch { Arch::Gfx1201 => &[1, 4, 8], _ => &[1, 4] }
}

/// Row repeats of the down NT4 entry built per arch, after every
/// [`tile_widths`] entry (gfx1151 only).
pub fn row_repeats(arch: Arch) -> &'static [u8] {
    match arch { Arch::Gfx1151 => &[2, 4], _ => &[] }
}

/// Every entry of one architecture's module, in emission order: both kinds at
/// every tile width, then the down NT4 row repeats.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    tile_widths(arch).iter()
        .flat_map(|&nt| Kind::ALL.into_iter().map(move |kind| Spec { arch, kind, nt, rr: 1 }))
        .chain(row_repeats(arch).iter().map(|&rr| Spec { arch, kind: Kind::Down, nt: 4, rr }))
        .collect()
}

impl Spec {
    pub fn symbol(self) -> String {
        let base = match self.kind {
            Kind::GateUp => format!("qwen4_moe_gate_up_silu_iu4_sym_pm_{}", self.arch.name()),
            Kind::Down => format!("qwen4_moe_down_iu4_sym_pm_{}", self.arch.name()),
        };
        if self.nt == 1 { base } else if self.rr == 1 { format!("{base}_nt{}", self.nt) } else { format!("{base}_nt{}r{}", self.nt, self.rr) }
    }
    /// Contract/variant tag: `gate_up`, `down_nt4`, `down_nt4r2`.
    pub fn variant(self) -> String {
        if self.nt == 1 { self.kind.tag().into() } else if self.rr == 1 { format!("{}_nt{}", self.kind.tag(), self.nt) } else { format!("{}_nt{}r{}", self.kind.tag(), self.nt, self.rr) }
    }
    pub fn module(arch: Arch) -> String { format!("qwen4_moe_iu4_sym_pm_{}", arch.name()) }
    pub fn validate(self) -> Result<(), String> {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) { return Err("qwen4_moe_sym: built for gfx1151 and gfx1201 only".into()) }
        // gfx11 operands are twice gfx12's per lane: NT8 exceeds the VGPR file.
        let max_nt = if self.arch.gfx12() { 8 } else { 4 };
        if !matches!(self.nt, 1 | 2 | 4 | 8) || self.nt > max_nt { return Err(format!("qwen4_moe_sym: tile width {} (1, 2, 4{})", self.nt, if max_nt == 8 { " or 8" } else { "" })) }
        if !matches!(self.rr, 1 | 2 | 4) { return Err(format!("qwen4_moe_sym: row repeat {} (1, 2 or 4)", self.rr)) }
        if self.rr > 1 && (self.kind != Kind::Down || self.nt != 4 || self.arch.gfx12()) {
            return Err(format!("qwen4_moe_sym: row repeat {} is built for the gfx1151 down NT4 entry only", self.rr))
        }
        Ok(())
    }
    /// Waves per CTA: the 16-slot gate/up pairs gate and up rows in one wave
    /// (2 waves); every expert-run gate/up and every down uses 4.
    pub fn waves(self) -> u32 { if self.kind == Kind::GateUp && self.nt == 1 { 2 } else { 4 } }
    pub fn threads(self) -> u32 { self.waves() * 32 }
    /// LDS bytes: the expert-run gate/up hands RNE(gate) rows to the up waves.
    pub fn lds_bytes(self) -> u32 { if self.kind == Kind::GateUp && self.nt > 1 { u32::from(self.nt) * NT_GATE_BYTES } else { 0 } }
    pub fn kernargs(self) -> KernargLayout {
        KernargLayout::new(KERNARG_BYTES).pointer("expert_weight_ptrs", 0).pointer("expert_tile_ids", 8)
            .pointer("sorted_slot_index", 16).pointer("Xq", 24).pointer("Y_grouped", 32)
            .hidden("M", 40, 4, "by_value").hidden("K", 44, 4, "by_value").hidden("x_row_div", 48, 4, "by_value")
            .hidden("m_total", 52, 4, "by_value").hidden("x_src_rows", 56, 4, "by_value")
    }
    /// Operand dwords per lane per K128 epoch of one 16-row/16-token fragment:
    /// gfx11 lanes `r` and `r+16` both carry all 16; gfx12 lanes split K32.
    fn aw(self) -> u8 { if self.arch.gfx12() { 8 } else { 16 } }
}

/// Default-off schedule experiment. Chains belong to distinct output tiles;
/// no output's K-step or K128-fold order changes. Not part of `emit_module`.
#[derive(Clone, Copy, Debug)]
pub struct R4Spec {
    pub base: Spec,
    pub chains: u8,
    /// Register-direct epoch schedule: distance one when true, zero otherwise.
    pub prefetch: bool,
    /// Compact heaviest-first list of pad16 tile leaders. The experimental
    /// ABI is 72 bytes: the original 60, four reserved bytes, then a pointer
    /// to little-endian u32 leader indices at offset 64. Grid.y is its length.
    /// Use `gemm_uk::sched::plan(counts, 16 * nt)`, converting each item's
    /// expert-relative first_row to `(expert_pad16_prefix + first_row) / 16`.
    pub grouped: bool,
}

pub fn emit_r4(spec: R4Spec) -> Result<Emitted, String> {
    spec.base.validate()?;
    if spec.base.nt == 1 || spec.base.rr != 1 || !matches!(spec.chains, 1 | 2 | 4 | 8) || spec.chains > spec.base.nt {
        return Err("qwen4_moe_sym R4: NT >= 2, row repeat 1 and chains <= NT required".into());
    }
    let nt = u16::from(spec.base.nt);
    let aw = u16::from(spec.base.aw());
    let end = (72 + 8 * nt + 2 * aw + 2 + nt).div_ceil(8) * 8 + aw * nt
        + 8 * u16::from(spec.chains - 1);
    if end > VGPR_CEILING { return Err(format!("qwen4_moe_sym R4: {end} VGPR exceeds {VGPR_CEILING}")); }
    nt::emit_with(spec)
}

// VGPRs.
const TID: u8 = 0;
const XOFF: u8 = 16;   // d offset of this lane's slot row (or X_OOB)
const XQOFF: u8 = 17;  // nibble base offset (gfx12: + 8*hi)
const WOFF: u8 = 18;   // lr*row_bytes (gfx12: + 8*hi)
const HOFF: u8 = 19;   // header row * row_bytes
const YOFF: u8 = 20;
const MAGIC8: u8 = 24;
const CACC: u8 = 32;   // int32 chains, 8 per tensor
const SUM: u8 = 48;    // f32 sums, 8 per tensor
const SCF: u8 = 64;    // broadcast row scales, 8 per tensor
const TPROD: u8 = 80;  // fold products d*sc
const HF: u8 = 88;     // f32 scale of this lane's header row, per tensor
const SET0: u8 = 92;
// Epilogue (aliases dead K-loop registers).
const RT_TMP: u8 = 32;
const OWN: u8 = 64; const SEND: u8 = 68; const RECV: u8 = 70; const SEL: u8 = 72; const OUT: u8 = 76; const PACK: u8 = 80;

// SGPRs.
const KARG: u8 = 0;
const T0: u8 = 4; const T1: u8 = 5; const T64: u8 = 6;
const ARGS0: u8 = 8;   // ptrs 8:9, tiles 10:11, sorted 12:13, xq 14:15
const ARGS1: u8 = 16;  // Y 16:17, M 18, K 19
const ARGS2: u8 = 20;  // x_row_div 20, m_total 21
const ARGS3: u8 = 22;  // x_src_rows
const EID: u8 = 23;
const SRD_W: u8 = 24; const SRD_U: u8 = 28; const SRD_X: [u8; 2] = [32, 36]; const SRD_S: u8 = 40;
const TRIPS: u8 = 44; const XS2: u8 = 45; const ROWB: u8 = 46; const TILE16: u8 = 47; const WAVE: u8 = 48; const RBASE: u8 = 49;
const WGX: u8 = 50; const WGY: u8 = 51;
const LIVE: u8 = 52; const HIMASK: u8 = 54; const PSEL: u8 = 56; const WPTR: u8 = 58;
const MASKT: [u8; 2] = [60, 62];
const SILU_MASK: u8 = 24;
const SRD_Y: u8 = 64;

struct Gen { spec: Spec }

impl Gen {
    fn arch(&self) -> Arch { self.spec.arch }
    fn label(&self, name: &str) -> String { format!(".Lq4s_{}_{name}", self.spec.kind.tag()) }
    fn t(&self) -> u8 { self.spec.kind.tensors() }
    fn aw(&self) -> u8 { self.spec.aw() }
    /// Set `p`'s registers: A of tensor t, X, header word of tensor t, d.
    fn set_base(&self, p: usize) -> u8 { SET0 + p as u8 * self.set_len() }
    fn set_len(&self) -> u8 { (self.aw() * (self.t() + 1) + self.t() + 1).div_ceil(4) * 4 }
    fn a(&self, p: usize, t: u8) -> u8 { self.set_base(p) + self.aw() * t }
    fn x(&self, p: usize) -> u8 { self.set_base(p) + self.aw() * self.t() }
    fn h(&self, p: usize, t: u8) -> u8 { self.x(p) + self.aw() + t }
    fn d(&self, p: usize) -> u8 { self.h(p, self.t()) }
    fn srd(&self, t: u8) -> u8 { if t == 0 { SRD_W } else { SRD_U } }
    /// Header and nibble byte offsets of set `p` inside one two-epoch trip.
    fn header_off(&self, p: usize, t: u8) -> u32 { let _ = t; match self.spec.kind { Kind::GateUp => 4 * p as u32, Kind::Down => 68 * p as u32 } }
    fn nibble_off(&self, p: usize) -> u32 { match self.spec.kind { Kind::GateUp => 8 + 64 * p as u32, Kind::Down => 4 + 68 * p as u32 } }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        let epi = || Live::Between(l("epilogue"), l("end"));
        p.v::<1>("tid", TID, pro())?;
        for i in 1..16u8 { p.v::<1>("prologue_tmp", i, pro())?; }
        for (name, r) in [("x_off", XOFF), ("w_off", WOFF), ("h_off", HOFF)] { p.v::<1>(name, r, kernel())?; }
        if self.arch().gfx12() { p.v::<1>("xq_off", XQOFF, kernel())?; }
        p.v::<1>("y_off", YOFF, Live::Whole)?;
        p.v::<8>("magic8", MAGIC8, kernel())?;
        for t in 0..self.t() {
            p.v::<8>("cacc", CACC + 8 * t, kernel())?;
            p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
            p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
            p.v::<1>("scale_f32", HF + t, kernel())?;
        }
        p.v::<8>("fold_t", TPROD, kernel())?;
        // Operand sets at their load granularity (gfx11 b128, gfx12 b64):
        // A per tensor, X, then the header words and d.
        for set in 0..2 {
            for chunk in 0..(self.t() + 1) * 4 {
                let base = self.a(set, 0) + chunk * self.aw() / 4;
                if self.arch().gfx12() { p.v::<2>("operand", base, kernel())?; } else { p.v::<4>("operand", base, kernel())?; }
            }
            for t in 0..self.t() { p.v::<1>("header", self.h(set, t), kernel())?; }
            p.v::<1>("d", self.d(set), kernel())?;
        }
        // Epilogue registers over the dead K-loop ranges.
        for i in 0..8u8 { p.v::<1>("rt_tmp", RT_TMP + i, epi())?; }
        if self.spec.kind == Kind::GateUp {
            for i in 0..(6 * SILU_N) { p.v::<1>("silu_tmp", SET0 + i, epi())?; }
            for i in 0..(3 * SILU_N) { p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi())?; }
        }
        p.v::<4>("out", OUT, epi())?;
        if self.arch().gfx12() {
            // Packed rows come straight from the sums.
        } else {
            for i in 0..4u8 { p.v::<1>("own", OWN + i, epi())?; }
            for i in 0..2u8 { p.v::<1>("send", SEND + i, epi())?; p.v::<1>("recv", RECV + i, epi())?; p.v::<1>("sel", SEL + i, epi())?; }
            for i in 0..4u8 { p.v::<1>("pack", PACK + i, epi())?; }
            p.s::<2>("hi_mask", HIMASK, epi())?;
            p.s::<1>("permlane_sel", PSEL, epi())?;
        }
        p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
        p.s::<4>("srd_y", SRD_Y, epi())?;
        if !self.arch().gfx12() { p.s::<1>("wg_x_in", 2, Live::Whole)?; p.s::<1>("wg_y_in", 3, Live::Whole)?; }
        p.s::<1>("s_tmp0", T0, kernel())?;
        p.s::<1>("s_tmp1", T1, kernel())?;
        p.s::<2>("s_tmp64", T64, kernel())?;
        p.s::<8>("kernargs_0x00", ARGS0, kernel())?;
        p.s::<2>("y_base", ARGS1, Live::Whole)?;
        p.s::<2>("kernargs_m_k", ARGS1 + 2, kernel())?;
        p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
        p.s::<1>("x_src_rows", ARGS3, kernel())?;
        p.s::<1>("expert", EID, kernel())?;
        for t in 0..self.t() { p.s::<4>("srd_w", self.srd(t), kernel())?; }
        for r in SRD_X { p.s::<4>("srd_x", r, kernel())?; }
        p.s::<4>("srd_sorted", SRD_S, kernel())?;
        for (name, r) in [("trips", TRIPS), ("x_stride2", XS2), ("row_bytes", ROWB), ("tile16", TILE16), ("wave", WAVE), ("row_base", RBASE), ("wg_x", WGX), ("wg_y", WGY)] {
            p.s::<1>(name, r, kernel())?;
        }
        p.s::<2>("w_ptr", WPTR, kernel())?;
        p.s::<2>("live", LIVE, Live::Whole)?;
        for m in MASKT { p.s::<2>("cmp_mask", m, Live::Whole)?; }
        // Keep the reservation a whole 8-VGPR granule, so the descriptor's
        // allocation equals the metadata count.
        let top = p.next_free_vgpr();
        if top % 8 != 0 { p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?; }
        Ok(p)
    }
}

fn prologue<T: Target>(wg: &mut Wg<T>, g: &Gen, end: &End) -> Result<(), String> {
    let b = wg.isa();
    let a = g.arch();
    let kind = g.spec.kind;
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    if a.gfx12() {
        // Workgroup ids: ttmp9 (x), ttmp7[15:0] (y).
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
        sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
        sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
    }
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(TID)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(TID)])?;
    op(b, format!("v_readfirstlane_b32 s{WAVE}, v1"), &[s(WAVE)], &[v(1)])?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    // A tile at or past m_total, or a sentinel tile, returns before any store.
    sop(b, format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"), &[TILE16], &[WGY])?;
    let outside = wg.scmp_wg_uniform(Instruction::new(format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1), vec![], vec![s(TILE16), s(ARGS2 + 1)]))?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, ARGS0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    let sentinel = wg.scmp_wg_uniform(Instruction::new(format!("s_cmp_lt_i32 s{EID}, 0"), vec![], vec![s(EID)]))?;
    wg.exit_if(sentinel, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    add64(b, T64, ARGS0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // row_bytes = (K/256)*136 (QT44) or (K/128)*68 (QT53).
    let k = ARGS1 + 3;
    match kind {
        Kind::GateUp => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 8"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x88"), &[ROWB], &[ROWB])?; }
        Kind::Down => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 7"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x44"), &[ROWB], &[ROWB])?; }
    }
    // Two-epoch trips before the tail: (K/128 - 1) / 2.
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)), &[TRIPS], &[TRIPS])?;
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"), &[TRIPS], &[TRIPS])?;
    // Row base of this wave: x*cta_rows + 16*wave.
    let shift = if kind == Kind::GateUp { 5 } else { 6 };
    sop(b, format!("s_lshl_b32 s{RBASE}, s{WGX}, {shift}"), &[RBASE], &[WGX])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(b, format!("{} s{RBASE}, s{RBASE}, s{T0}", s_add_i32(a)), &[RBASE], &[RBASE, T0])?;
    // Sorted slots of this tile.
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 6"), &[T0], &[WGY])?;
    add64(b, SRD_S, ARGS0 + 4, T0)?;
    sop(b, format!("s_mov_b32 s{}, 64", SRD_S + 2), &[SRD_S + 2], &[])?;
    sop(b, format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)), &[SRD_S + 3], &[])?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    bload(b, 1, 6, 5, SRD_S, 0)?;
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    sop(b, format!("s_mul_i32 s{T0}, s{ARGS3}, 0x48"), &[T0], &[ARGS3])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_X[0], ARGS0 + 6), &[SRD_X[0]], &[ARGS0 + 6])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_X[0] + 1, ARGS0 + 7), &[SRD_X[0] + 1], &[ARGS0 + 7])?;
    srd_tail(b, SRD_X[0], Some(T0))?;
    add64(b, SRD_X[1], ARGS0 + 6, T0)?;
    srd_tail(b, SRD_X[1], Some(T0))?;
    sop(b, format!("s_lshl_b32 s{XS2}, s{T0}, 1"), &[XS2], &[T0])?;
    // Weight descriptors: rows rbase.. of the expert (up rows M/2 further).
    sop(b, format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"), &[T0], &[RBASE, ROWB])?;
    add64(b, SRD_W, WPTR, T0)?;
    srd_tail(b, SRD_W, None)?;
    if kind == Kind::GateUp {
        sop(b, format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2), &[T1], &[ARGS1 + 2])?;
        sop(b, format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"), &[T1], &[T1, ROWB])?;
        add64(b, SRD_U, SRD_W, T1)?;
        srd_tail(b, SRD_U, None)?;
    }
    // Lane offsets. Weights: lr*row_bytes (gfx12 + 8*hi, its K32 half).
    op(b, format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"), &[v(WOFF)], &[s(ROWB), v(3)])?;
    if a.gfx12() { op(b, format!("v_lshl_add_u32 v{WOFF}, v4, 3, v{WOFF}"), &[v(WOFF)], &[v(4), v(WOFF)])?; }
    // Header row: the row whose scale this lane's broadcast slot serves.
    if a.gfx12() {
        op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
        op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    } else {
        op(b, "v_and_b32_e32 v7, 14, v3", &[v(7)], &[v(3)])?;
        op(b, "v_or_b32_e32 v7, v7, v4", &[v(7)], &[v(7), v(4)])?;
    }
    op(b, format!("v_mul_u32_u24_e32 v{HOFF}, s{ROWB}, v7"), &[v(HOFF)], &[s(ROWB), v(7)])?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2.
    match kind {
        Kind::GateUp => sop(b, format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2), &[T1], &[ARGS1 + 2])?,
        Kind::Down => sop(b, format!("s_mov_b32 s{T1}, s{}", ARGS1 + 2), &[T1], &[ARGS1 + 2])?,
    }
    op(b, format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"), &[v(8)], &[s(TILE16), v(3)])?;
    op(b, format!("v_mul_lo_u32 v8, v8, s{T1}"), &[v(8)], &[v(8), s(T1)])?;
    op(b, format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"), &[v(9)], &[v(4), s(RBASE)])?;
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(b, format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"), &[v(YOFF)], &[v(8)])?;
    // Activation row offset: (slot / x_row_div) * 72, or the gather's
    // out-of-range offset for a padding slot so its loads read 0.
    common::gather_offset(b, XOFF, 6, Some(ARGS2), iu4_fold::XBLK_BYTES, LIVE, GatherTemps { v: [11, 12, 13, 10], mask: MASKT[0] })?;
    if a.gfx12() { op(b, format!("v_lshl_add_u32 v{XQOFF}, v4, 3, v{XOFF}"), &[v(XQOFF)], &[v(4), v(XOFF)])?; }
    load_set(b, g, 0)?;
    for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)), &[v(MAGIC8 + j)], &[])?; }
    for t in 0..g.t() { for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, 0", SUM + 8 * t + j), &[v(SUM + 8 * t + j)], &[])?; } }
    Ok(())
}

/// Issue set `p`'s loads (epoch parity p of the current trip): header words,
/// d, the activation nibbles, then the weight nibbles, as one clause.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let gfx12 = g.arch().gfx12();
    let xq = if gfx12 { XQOFF } else { XOFF };
    let (width, step) = if gfx12 { (2u8, 2u8) } else { (4u8, 4u8) };
    b.clause(|b| {
        for t in 0..g.t() { bload(b, 1, g.h(p, t), HOFF, g.srd(t), g.header_off(p, t))?; }
        bload(b, 1, g.d(p), XOFF, SRD_X[p], 0)?;
        for i in 0..4u8 { bload(b, width, g.x(p) + step * i, xq, SRD_X[p], 8 + 16 * u32::from(i))?; }
        for t in 0..g.t() {
            for i in 0..4u8 { bload(b, width, g.a(p, t) + step * i, WOFF, g.srd(t), g.nibble_off(p) + 16 * u32::from(i))?; }
        }
        Ok(())
    })
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let a = g.arch();
    // Row scales: this lane's header word -> f32 -> the eight rows of its
    // accumulators by ds_swizzle broadcast within each 16-lane half.
    for t in 0..g.t() {
        op(b, format!("v_cvt_f32_f16_e64 v{}, v{}.l", HF + t, g.h(p, t)), &[v(HF + t)], &[v(g.h(p, t))])?;
        for j in 0..8u8 {
            let k = if a.gfx12() { j } else { 2 * j };
            let text = format!("ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{k})", SCF + 8 * t + j, HF + t);
            b.ds_crosslane(crate::insn::Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)]).memory(MemoryClass::DsLoad))?;
        }
    }
    for t in 0..g.t() {
        for i in 0..g.aw() {
            let r = g.a(p, t) + i;
            op(b, format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)), &[v(r)], &[v(r)])?;
        }
    }
    for i in 0..g.aw() / 2 {
        for t in 0..g.t() {
            iu4_fold::wmma_step(b, a, V::<8>(CACC + 8 * t), V::<2>(g.a(p, t) + 2 * i), V::<2>(g.x(p) + 2 * i), i == 0, V::<8>(MAGIC8))?;
        }
    }
    for t in 0..g.t() { iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, g.d(p), TPROD)?; }
    Ok(())
}

fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.t() { add64_imm(b, g.srd(t), 136)?; }
    for r in SRD_X { add64(b, r, r, XS2)?; }
    Ok(())
}

/// Two-epoch trips over `TRIPS` (skipped at zero trips), then the tail
/// epochs.
fn kloop<T: Target>(wg: &mut Wg<T>, g: &Gen) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
    wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
        wg.loop_carried(&head, (), |wg, ()| {
            let b = wg.isa();
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            advance(b, g)?;
            load_set(b, g, 0)?;
            compute(b, g, 1)?;
            op(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)), &[s(TRIPS)], &[s(TRIPS)])?;
            Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
        })?;
        if wg.isa().ledger.shape() != entry { return Err("k loop exit ledger differs from its entry".into()) }
        Ok(())
    })?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => {
            let b = wg.isa();
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            compute(b, g, 1)
        }
        // K % 128 == 0: one epoch left when K/128 is odd, else two.
        Kind::Down => {
            let odd = wg.scmp(Instruction::new(format!("s_bitcmp1_b32 s{}, 7", ARGS1 + 3), vec![], vec![s(ARGS1 + 3)]))?;
            wg.if_else(odd, &g.label("tail_odd"), &g.label("epilogue"), |w| {
                let b = w.isa();
                load_set(b, g, 1)?;
                compute(b, g, 0)?;
                compute(b, g, 1)
            }, |w| compute(w.isa(), g, 0))
        }
    }
}

fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("epilogue"))?;
    let vals = SUM;
    if g.spec.kind == Kind::GateUp {
        // g and u rounded to BF16 before the shipped SwiGLU expression.
        for t in 0..2u8 { for j in 0..8u8 { Bf16::rne_finite_passthrough(b, SUM + 8 * t + j, RT_TMP + j, MASKT[usize::from(j % 2)], true)?; } }
        for grp in 0..2u8 { crate::kernels::gemm_uk::Epilogue::silu_dense(b, SUM + SILU_N * grp, SUM + 8 + SILU_N * grp, SET0, SILU_MASK, SILU_N)?; }
    }
    for j in 0..8u8 { Bf16::rne_finite_passthrough(b, vals + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?; }
    // Padding slots store +0.
    for j in 0..8u8 { op(b, format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{LIVE}", vals + j), &[v(vals + j)], &[v(vals + j), s(LIVE)])?; }
    let perm = |b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String| -> Result<(), String> {
        let mut uses = vec![v(hi_src), v(lo_src)];
        if sel.starts_with('v') { uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?)); }
        op(b, format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"), &[v(dst)], &uses)
    };
    if g.arch().gfx12() {
        // Lane (hi, lr) holds rows 8hi..8hi+7 of token lr: pack and store.
        for k in 0..4u8 { perm(b, OUT + k, vals + 2 * k + 1, vals + 2 * k, lit(0x0706_0302))?; }
    } else {
        // Lane (hi, lr) holds rows 2j+hi of token lr. Exchange with the
        // other half so lane hi stores rows 8hi..8hi+7: lane 0 sends rows
        // 8..14 (even), lane 1 sends rows 1..7 (odd), via v_permlanex16.
        // A VOP3 lane-mask operand is a pair for M7; the high word is unused in wave32.
        op(b, format!("s_mov_b32 s{HIMASK}, 0xffff0000"), &[s(HIMASK)], &[])?;
        op(b, format!("s_mov_b32 s{}, 0", HIMASK + 1), &[s(HIMASK + 1)], &[])?;
        op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        for k in 0..4u8 {
            op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}", OWN + k, vals + k, vals + 4 + k), &[v(OWN + k)], &[v(vals + k), v(vals + 4 + k), s(HIMASK)])?;
        }
        for m in 0..2u8 {
            // hi = 0 sends pack(H[4+2m], H[5+2m]); hi = 1 sends pack(H[2m], H[2m+1]).
            perm(b, PACK + 2 * m, vals + 5 + 2 * m, vals + 4 + 2 * m, lit(0x0706_0302))?;
            perm(b, PACK + 2 * m + 1, vals + 1 + 2 * m, vals + 2 * m, lit(0x0706_0302))?;
            op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}", SEND + m, PACK + 2 * m, PACK + 2 * m + 1), &[v(SEND + m)], &[v(PACK + 2 * m), v(PACK + 2 * m + 1), s(HIMASK)])?;
            op(b, format!("v_mov_b32_e32 v{}, v{}", RECV + m, SEND + m), &[v(RECV + m)], &[v(SEND + m)])?;
            op(b, format!("v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98", RECV + m, SEND + m), &[v(RECV + m)], &[v(RECV + m), v(SEND + m), s(PSEL)])?;
        }
        // Byte selectors (src0 = received word, src1 = own row): hi = 0
        // puts its own row low, hi = 1 puts the received row low.
        for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)].into_iter().enumerate() {
            let r = SEL + i as u8;
            op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
            op(b, format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{HIMASK}", lit(lo)), &[v(r)], &[v(r), s(HIMASK)])?;
        }
        for k in 0..4u8 { perm(b, OUT + k, RECV + k / 2, OWN + k, format!("v{}", SEL + k % 2))?; }
    }
    // Raw buffer store over Y (the same VMEM family as every load).
    sop(b, format!("s_mov_b32 s{SRD_Y}, s{ARGS1}"), &[SRD_Y], &[ARGS1])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_Y + 1, ARGS1 + 1), &[SRD_Y + 1], &[ARGS1 + 1])?;
    srd_tail(b, SRD_Y, None)?;
    bstore_b128(b, OUT, YOFF, SRD_Y, 0)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.nt > 1 { return nt::emit(spec) }
    let g = Gen { spec };
    let kspec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(), variant: spec.kind.tag().into(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    on_target!(spec.arch, &mut b, body(&g))?;
    b.finish()
}

fn body<T: Target>(wg: &mut Wg<T>, g: &Gen) -> Result<(), String> {
    let end = wg.exit(&g.label("end"))?;
    prologue(wg, g, &end)?;
    kloop(wg, g)?;
    epilogue(wg.isa(), g)?;
    wg.end(end)
}

/// Every entry of one architecture ([`module_specs`]: both kinds at every
/// [`tile_widths`] width, then the [`row_repeats`]) as one code object.
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch).into_iter().map(emit).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// LDS bytes per 16-slot tile of the expert-run gate/up handoff: RNE(gate)
/// of 2 row pairs x 32 lanes x 8 f32.
pub const NT_GATE_BYTES: u32 = 2048;

/// Expert-run tiles (`nt` 16-slot tiles of one expert per weight stream).
///
/// Grid and ABI are the 16-slot entry's; the block is 128 (four waves). CTA
/// `(x, tile)` leads when `tile`'s position inside its expert's run of tiles
/// (counted by a 32-lane walk back over `expert_tile_ids`) is a multiple of
/// `nt`; others return. The leader takes `cnt` = up to `nt` consecutive tiles
/// of the same expert (bounded by `m_total/16`); tiles past `cnt` are dead:
/// their slots read -1 (gathered from nothing), their folds are skipped by
/// uniform branches and they store nothing. Per K128 epoch every wave loads
/// its 16 weight rows once (the next epoch prefetched) and folds them with
/// each live tile's activations, which are reloaded for the next epoch right
/// after the tile's fold: each output keeps the 16-slot entry's ascending
/// per-epoch fold and rounding, so all bytes equal its bytes.
///
/// Gate/up waves are (row pair `w & 1`) x (gate if `w < 2`, else up rows
/// `M/2` further). After the K loop every wave rounds its sums to BF16
/// (round trip); the gate waves store theirs to LDS and the up waves, after
/// one barrier, read their pair's gate values in the same lane layout and
/// store `rt(silu(rt(g)) * rt(u))`. Down waves own rows `64x + 16w` as in
/// the 16-slot entry.
///
/// Row repeat (`rr` = 2 or 4, gfx1151 down NT4): the grid is
/// `(M/(64*rr), m_total/16)` and CTA `x` folds `rr` contiguous 64-row blocks
/// of its run, block `b` owning rows `64*rr*x + 64*b + 16w ..+16` of wave
/// `w`; adjacent blocks therefore stream adjacent weight rows. The prologue
/// (run walk, leader test, `cnt`, slots, activation offsets, magic) runs
/// once. The blocks are unrolled statically, each with its own labels, and
/// each block resets the activation descriptors, the K trip count and the
/// weight descriptor (`w_ptr + row_base * row_bytes`) and zeroes its sums.
/// Each block's output is the 16-slot-entry fold of those rows, so every
/// output byte equals the `rr = 1` entry's. Block `b` stores at `Y` offset
/// `+128*b` bytes (store immediate), and a dead tile skips its store
/// instead of leaving the kernel, so later blocks still run.
///
/// Between blocks the next block's first-epoch weights are issued before
/// the current block's stores, into the weight set the current block loaded
/// second (`rr = 1` parity swaps every block: block `b` reads logical set
/// `p` from physical set `p ^ (b & 1)`), so no copy is needed and the
/// stores and the next epoch's activation reload (issued after them) overlap
/// the weight latency. The store temporaries (`rt_tmp`, `own`, `send`,
/// `recv`, `sel`, `out`) move onto the dead fold registers (`scale_rows`,
/// `cacc`, `fold_t`) so no live weight set is overwritten, and the
/// gfx11 `pack` words alias the dead tile 0 activations `x(0)[0..4]`
/// until the reload. Pending stores are drained (`vscnt`) before the K loop
/// so its entry and exit ledgers agree. The register plan gives every block
/// its own adjacent body/epilogue intervals on the aliased registers.
mod nt {
    use super::*;
    use peacemaker_author::{End, Free, LdsRegion, StoreTarget, Wave};

    /// The gate/up handoff region: RNE(gate) of every tile.
    enum Gate {}

    // VGPRs (the gfx12 NT8 maximum is v231).
    const XOFF: u8 = 16;   // + j: activation row offset of tile j (or X_OOB)
    const XQOFF: u8 = 24;  // + j: gfx12 nibble offset (+ 8*hi)
    const WOFF: u8 = 32;
    const HOFF: u8 = 33;
    const YOFF: u8 = 34;
    const LDSA: u8 = 35;   // gate/up: (32*(w&1) + lane) * 32
    const HF: u8 = 36;
    const MAGIC8: u8 = 40;
    const CACC: u8 = 48;
    const TPROD: u8 = 56;
    const SCF: u8 = 64;
    const SUM0: u8 = 72;   // + 8j
    // Epilogue aliases (dead K-loop ranges).
    const SILU_TMP: u8 = 40; // 24: magic, cacc, fold products
    const RT_TMP: u8 = 64;   // 8: row scales

    // SGPRs.
    const SRD_W: u8 = 24; const WPTR: u8 = 28; const BACK: u8 = 30; const POS: u8 = 31;
    const SRD_X: [u8; 2] = [32, 36]; const SRD_S: u8 = 40;
    const TRIPS: u8 = 44; const XS2: u8 = 45; const ROWB: u8 = 46; const TILE16: u8 = 47;
    const WAVE: u8 = 48; const RBASE: u8 = 49; const WGX: u8 = 50; const WGY: u8 = 51;
    const CNT: u8 = 52; const YSTEP: u8 = 53;
    const MASKA: u8 = 54; // prologue lane mask; epilogue `hi_mask` (gfx11)
    const MASKT: [u8; 2] = [56, 58];
    const LIVE0: u8 = 60; // + 2j
    const SRD_Y: u8 = 76; const PSEL: u8 = 80;
    const SILU_MASK: u8 = 24;

    #[derive(Clone, Copy)]
    struct G { spec: Spec, blk: u8, chains: u8, prefetch: bool, grouped: bool }
    impl G {
        fn arch(&self) -> Arch { self.spec.arch }
        fn nt(&self) -> u8 { self.spec.nt }
        fn aw(&self) -> u8 { self.spec.aw() }
        fn gate_up(&self) -> bool { self.spec.kind == Kind::GateUp }
        fn rr(&self) -> u8 { self.spec.rr }
        /// Row block `blk` of the same entry.
        fn at(&self, blk: u8) -> G { G { blk, ..*self } }
        /// Block-local label; `rr = 1` keeps the frozen entry's names.
        fn label(&self, name: &str) -> String {
            if self.rr() == 1 { format!(".Lq4s_{}_nt{}_{name}", self.spec.kind.tag(), self.nt()) } else { format!(".Lq4s_{}_nt{}r{}b{}_{name}", self.spec.kind.tag(), self.nt(), self.rr(), self.blk) }
        }
        fn exit_label(&self) -> String {
            if self.rr() == 1 { self.label("end") } else { format!(".Lq4s_{}_nt{}r{}_end", self.spec.kind.tag(), self.nt(), self.rr()) }
        }
        /// A default-off R4 schedule: the original emission has none of the
        /// per-dead-tile guards below.
        fn experimental(&self) -> bool { self.chains != 1 || !self.prefetch || self.grouped }
        /// Schedules whose dead-tile guards join differently-shaped load
        /// sets (`maybe` pendings): the prologue and every trip end settle all
        /// loads, so the K loop's entry and back-edge ledgers are both empty.
        fn settles_loads(&self) -> bool { self.experimental() && (self.chains != 1 || !self.prefetch) }
        fn sum(&self, j: u8) -> u8 { SUM0 + 8 * j }
        fn sums_end(&self) -> u8 { SUM0 + 8 * self.nt() }
        /// Parity of the physical weight sets: odd row blocks swap them.
        fn flip(&self) -> usize { usize::from(self.blk & 1) }
        /// Physical weight nibble set / header word `set`.
        fn wa_set(&self, set: usize) -> u8 { self.sums_end() + self.aw() * set as u8 }
        fn wh_set(&self, set: usize) -> u8 { self.sums_end() + 2 * self.aw() + set as u8 }
        /// Weight nibbles / header of epoch parity `p` in this block: odd
        /// row blocks swap the physical sets (see the module note).
        fn wa(&self, p: usize) -> u8 { self.wa_set(p ^ self.flip()) }
        fn wh(&self, p: usize) -> u8 { self.wh_set(p ^ self.flip()) }
        fn d(&self, j: u8) -> u8 { self.sums_end() + 2 * self.aw() + 2 + j }
        fn x(&self, j: u8) -> u8 { (self.d(self.nt())).div_ceil(8) * 8 + self.aw() * j }
        fn live(&self, j: u8) -> u8 { LIVE0 + 2 * j }
        fn cacc(&self, j: u8) -> u8 {
            if j == 0 { CACC } else { self.x(self.nt()) + 8 * (j - 1) }
        }
        // Epilogue temporaries: gate values over tile 0's activations, the
        // gfx11 lane exchange over the weight sets.
        fn gval(&self) -> u8 { self.x(0) }
        // Row repeats keep every weight set live across the transition, so
        // their store temporaries sit on the dead fold registers instead.
        fn own(&self) -> u8 { if self.rr() > 1 { CACC } else { self.wa(0) } }
        fn send(&self) -> u8 { if self.rr() > 1 { CACC + 4 } else { self.wa(0) + 4 } }
        fn recv(&self) -> u8 { if self.rr() > 1 { CACC + 6 } else { self.wa(0) + 6 } }
        fn sel(&self) -> u8 { if self.rr() > 1 { TPROD } else { self.wa(0) + 8 } }
        fn out(&self) -> u8 { if self.rr() > 1 { TPROD + 4 } else { self.wa(0) + 12 } }
        fn pack(&self) -> u8 { if self.rr() > 1 { self.x(0) } else { self.wa(0) + 16 } }
        fn header_off(&self, p: usize) -> u32 { match self.spec.kind { Kind::GateUp => 4 * p as u32, Kind::Down => 68 * p as u32 } }
        fn nibble_off(&self, p: usize) -> u32 { match self.spec.kind { Kind::GateUp => 8 + 64 * p as u32, Kind::Down => 4 + 68 * p as u32 } }

        fn plan(&self) -> Result<RegPlan, String> {
            let mut p = RegPlan::new(VGPR_CEILING, 104)?;
            let rr = self.rr();
            let blk = |b: u8| self.at(b);
            let l = |n: &str| blk(0).label(n);
            let exit = self.exit_label();
            // Whole-run registers: live to the last block's epilogue.
            let kernel = || Live::Between("entry".into(), blk(rr - 1).label("epilogue"));
            let pro = || Live::Between("entry".into(), l("k_begin"));
            // Registers the store temporaries alias: block `b`'s body runs
            // from the previous block's reload (entry for block 0) to its
            // epilogue, and its temporaries from there to its own reload
            // (the exit for the last block). `rr = 1` is `kernel`/`epi`.
            let body = |b: u8| Live::Between(if b == 0 { "entry".to_string() } else { blk(b - 1).label("xreload") }, blk(b).label("epilogue"));
            let epi = |b: u8| Live::Between(blk(b).label("epilogue"), if b + 1 == rr { exit.clone() } else { blk(b).label("xreload") });
            let gfx12 = self.arch().gfx12();
            p.v::<1>("tid", 0, pro())?;
            for i in 1..16u8 { p.v::<1>("prologue_tmp", i, pro())?; }
            for j in 0..self.nt() {
                p.v::<1>("x_off", XOFF + j, kernel())?;
                if gfx12 { p.v::<1>("xq_off", XQOFF + j, kernel())?; }
                p.v::<8>("sum", self.sum(j), Live::Whole)?;
                p.v::<1>("d", self.d(j), kernel())?;
                for c in 0..4u8 {
                    if gfx12 { p.v::<2>("x", self.x(j) + 2 * c, kernel())?; }
                    else if j == 0 && c == 0 { for b in 0..rr { p.v::<4>("x", self.x(0), body(b))?; } }
                    else { p.v::<4>("x", self.x(j) + 4 * c, kernel())?; }
                }
                p.s::<2>("live", self.live(j), Live::Whole)?;
            }
            for (name, r) in [("w_off", WOFF), ("h_off", HOFF), ("scale_f32", HF)] { p.v::<1>(name, r, kernel())?; }
            p.v::<1>("y_off", YOFF, Live::Whole)?;
            if self.gate_up() { p.v::<1>("lds_addr", LDSA, Live::Whole)?; }
            p.v::<8>("magic8", MAGIC8, kernel())?;
            for b in 0..rr { p.v::<8>("cacc", CACC, body(b))?; }
            for j in 1..self.chains { p.v::<8>("tile_cacc", self.cacc(j), kernel())?; }
            for b in 0..rr { p.v::<8>("fold_t", TPROD, body(b))?; }
            for b in 0..rr { p.v::<8>("scale_rows", SCF, body(b))?; }
            for set in 0..2 {
                for c in 0..4u8 { if gfx12 { p.v::<2>("w", self.wa_set(set) + 2 * c, kernel())?; } else { p.v::<4>("w", self.wa_set(set) + 4 * c, kernel())?; } }
                p.v::<1>("header", self.wh_set(set), kernel())?;
            }
            for b in 0..rr {
                for i in 0..8u8 { p.v::<1>("rt_tmp", RT_TMP + i, epi(b))?; }
                p.v::<4>("out", self.out(), epi(b))?;
                if self.gate_up() {
                    p.v::<8>("gate", self.gval(), epi(b))?;
                    for i in 0..(6 * SILU_N) { p.v::<1>("silu_tmp", SILU_TMP + i, epi(b))?; }
                    for i in 0..(3 * SILU_N) { p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi(b))?; }
                }
                if !gfx12 {
                    for i in 0..4u8 { p.v::<1>("own", self.own() + i, epi(b))?; p.v::<1>("pack", self.pack() + i, epi(b))?; }
                    for i in 0..2u8 { p.v::<1>("send", self.send() + i, epi(b))?; p.v::<1>("recv", self.recv() + i, epi(b))?; p.v::<1>("sel", self.sel() + i, epi(b))?; }
                    p.s::<2>("hi_mask", MASKA, epi(b))?;
                    p.s::<1>("permlane_sel", PSEL, epi(b))?;
                }
            }
            if !gfx12 { p.s::<1>("wg_x_in", 2, Live::Whole)?; p.s::<1>("wg_y_in", 3, Live::Whole)?; }
            p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
            for b in 0..rr { p.s::<4>("srd_y", SRD_Y, epi(b))?; }
            p.s::<1>("s_tmp0", 4, kernel())?;
            p.s::<1>("s_tmp1", 5, kernel())?;
            p.s::<2>("s_tmp64", 6, kernel())?;
            p.s::<8>("kernargs_0x00", 8, kernel())?;
            p.s::<2>("y_base", 16, Live::Whole)?;
            p.s::<2>("kernargs_m_k", 18, kernel())?;
            p.s::<2>("kernargs_0x30", 20, kernel())?;
            p.s::<1>("x_src_rows", 22, kernel())?;
            p.s::<1>("expert", 23, kernel())?;
            p.s::<4>("srd_w", SRD_W, kernel())?;
            p.s::<2>("w_ptr", WPTR, kernel())?;
            p.s::<1>("run_back", BACK, pro())?;
            p.s::<1>("run_pos", POS, pro())?;
            for r in SRD_X { p.s::<4>("srd_x", r, kernel())?; }
            p.s::<4>("srd_tiles_sorted", SRD_S, kernel())?;
            for (name, r) in [("trips", TRIPS), ("x_stride2", XS2), ("row_bytes", ROWB), ("tile16", TILE16), ("row_base", RBASE), ("wg_x", WGX), ("wg_y", WGY)] {
                p.s::<1>(name, r, kernel())?;
            }
            p.s::<1>("wave", WAVE, Live::Whole)?;
            p.s::<1>("tile_count", CNT, Live::Whole)?;
            p.s::<1>("y_step", YSTEP, Live::Whole)?;
            p.s::<2>("lane_mask", MASKA, pro())?;
            for m in MASKT { p.s::<2>("cmp_mask", m, Live::Whole)?; }
            let top = p.next_free_vgpr();
            if top % 8 != 0 { p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?; }
            Ok(p)
        }
    }

    fn sub_i32(a: Arch) -> &'static str { if a.gfx12() { "s_sub_co_i32" } else { "s_sub_i32" } }

    /// Lane mask `m` (a pair; the high word is unused in wave32) |= `n`.
    fn or_mask(b: &mut Builder, m: u8, n: u8) -> Result<(), String> {
        sop(b, format!("s_or_b32 s{m}, s{m}, s{n}"), &[m], &[m, n])
    }

    /// Words 0..3 of both activation descriptors (epoch 0, then epoch 1)
    /// from the kernel arguments; leaves `s4 = x_src_rows * 72`.
    fn x_descriptors(b: &mut Builder) -> Result<(), String> {
        sop(b, "s_mul_i32 s4, s22, 0x48", &[4], &[22])?;
        sop(b, format!("s_mov_b32 s{}, s14", SRD_X[0]), &[SRD_X[0]], &[14])?;
        sop(b, format!("s_mov_b32 s{}, s15", SRD_X[0] + 1), &[SRD_X[0] + 1], &[15])?;
        srd_tail(b, SRD_X[0], Some(4))?;
        add64(b, SRD_X[1], 14, 4)?;
        srd_tail(b, SRD_X[1], Some(4))
    }

    /// The weight descriptor at rows `row_base..` of the expert; leaves `s4 = row_base * row_bytes`.
    fn w_descriptor(b: &mut Builder) -> Result<(), String> {
        sop(b, format!("s_mul_i32 s4, s{RBASE}, s{ROWB}"), &[4], &[RBASE, ROWB])?;
        add64(b, SRD_W, WPTR, 4)?;
        srd_tail(b, SRD_W, None)
    }

    /// Two-epoch trips before the tail: (K/128 - 1) / 2.
    fn set_trips(b: &mut Builder, a: Arch) -> Result<(), String> {
        let k = 19;
        sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
        sop(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)), &[TRIPS], &[TRIPS])?;
        sop(b, format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"), &[TRIPS], &[TRIPS])
    }

    fn prologue<T: Target>(wg: &mut Wg<T>, g: &G, end: &End) -> Result<(), String> {
        let b = wg.isa();
        let a = g.arch();
        let kind = g.spec.kind;
        let nt = g.nt();
        // The LDS address comes first: certification derives it from the
        // work-item id in the entry block. (32*(w&1) + lane) * 32 bytes.
        if g.gate_up() {
            op(b, format!("v_and_b32_e32 v{LDSA}, 63, v0"), &[v(LDSA)], &[v(0)])?;
            op(b, format!("v_lshlrev_b32_e32 v{LDSA}, 5, v{LDSA}"), &[v(LDSA)], &[v(LDSA)])?;
        }
        smem(b, 8, 8, 0, 0)?;
        smem(b, 16, 2, 0, 0x20)?;
        smem(b, 18, 2, 0, 0x28)?;
        smem(b, 20, 2, 0, 0x30)?;
        smem(b, 22, 1, 0, 0x38)?;
        if a.gfx12() {
            sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
            sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
        } else {
            sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
            sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
        }
        if g.grouped {
            smem(b, 6, 2, 0, 0x40)?;
            sop(b, format!("s_lshl_b32 s4, s{WGY}, 2"), &[4], &[WGY])?;
            add64(b, 6, 6, 4)?;
            smem(b, WGY, 1, 6, 0)?;
        }
        // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
        op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
        op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
        op(b, format!("v_readfirstlane_b32 s{WAVE}, v1"), &[s(WAVE)], &[v(1)])?;
        op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
        op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
        // A tile at or past m_total, or a sentinel tile, returns.
        sop(b, format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"), &[TILE16], &[WGY])?;
        let outside = wg.scmp_wg_uniform(Instruction::new(format!("s_cmp_ge_i32 s{TILE16}, s21"), vec![], vec![s(TILE16), s(21)]))?;
        wg.exit_if(outside, end)?;
        let b = wg.isa();
        sop(b, format!("s_lshl_b32 s4, s{WGY}, 2"), &[4], &[WGY])?;
        add64(b, 6, 10, 4)?;
        smem(b, 23, 1, 6, 0)?;
        let sentinel = wg.scmp_wg_uniform(Instruction::new("s_cmp_lt_i32 s23, 0", vec![], vec![s(23)]))?;
        wg.exit_if(sentinel, end)?;
        let b = wg.isa();
        sop(b, "s_lshl_b32 s4, s23, 3", &[4], &[23])?;
        add64(b, 6, 8, 4)?;
        smem(b, WPTR, 2, 6, 0)?;
        // Tile ids through a descriptor of m_total/16 records: an id read at
        // or past the end (or before the start) reads 0, so both bounds are
        // tested explicitly.
        sop(b, format!("s_mov_b32 s{SRD_S}, s10"), &[SRD_S], &[10])?;
        sop(b, format!("s_mov_b32 s{}, s11", SRD_S + 1), &[SRD_S + 1], &[11])?;
        sop(b, format!("s_lshr_b32 s5, s21, 2"), &[5], &[21])?;
        srd_tail(b, SRD_S, Some(5))?;
        if !g.grouped {
        // Run position: lanes test tiles tile-1-back-lane; the first that is
        // before the start or another expert ends the run.
        sop(b, format!("s_mov_b32 s{BACK}, 0"), &[BACK], &[])?;
        let (walk, found) = (g.label("run_walk"), g.label("run_found"));
        // Wave-uniform: every lane mask is a compare over the same loads.
        wg.loop_until(&walk, &found, |w, exit| {
            let b = w.isa();
            sop(b, format!("{} s4, s{WGY}, -1", s_add_i32(a)), &[4], &[WGY])?;
            sop(b, format!("{} s4, s4, s{BACK}", sub_i32(a)), &[4], &[4, BACK])?;
            op(b, "v_sub_nc_u32_e32 v5, s4, v2", &[v(5)], &[s(4), v(2)])?;
            op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
            bload(b, 1, 7, 6, SRD_S, 0)?;
            op(b, format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"), &[s(MASKA)], &[s(23), v(7)])?;
            op(b, format!("v_cmp_gt_i32_e64 s{}, 0, v5", MASKT[0]), &[s(MASKT[0])], &[v(5)])?;
            or_mask(b, MASKA, MASKT[0])?;
            let hit = w.scmp(Instruction::new(format!("s_cmp_lg_u32 s{MASKA}, 0"), vec![], vec![s(MASKA)]))?;
            w.break_if(hit, exit)?;
            sop(w.isa(), format!("{} s{BACK}, s{BACK}, 32", s_add_i32(a)), &[BACK], &[BACK])
        })?;
        let b = wg.isa();
        sop(b, format!("s_ctz_i32_b32 s{POS}, s{MASKA}"), &[POS], &[MASKA])?;
        sop(b, format!("{} s{POS}, s{POS}, s{BACK}", s_add_i32(a)), &[POS], &[POS, BACK])?;
        // Not a leader: another CTA folds this tile.
        sop(b, format!("s_and_b32 s{POS}, s{POS}, {}", nt - 1), &[POS], &[POS])?;
        let follower = wg.scmp_wg_uniform(Instruction::new(format!("s_cmp_lg_u32 s{POS}, 0"), vec![], vec![s(POS)]))?;
        wg.exit_if(follower, end)?;
        }
        let b = wg.isa();
        // cnt: the first lane l < nt whose tile tile+l is past the end or of
        // another expert (lanes >= nt always stop).
        op(b, format!("v_add_nc_u32_e32 v5, s{WGY}, v2"), &[v(5)], &[s(WGY), v(2)])?;
        op(b, "v_lshlrev_b32_e32 v6, 2, v5", &[v(6)], &[v(5)])?;
        bload(b, 1, 7, 6, SRD_S, 0)?;
        sop(b, "s_lshr_b32 s5, s21, 4", &[5], &[21])?;
        op(b, format!("v_cmp_ne_u32_e64 s{MASKA}, s23, v7"), &[s(MASKA)], &[s(23), v(7)])?;
        op(b, format!("v_cmp_le_u32_e64 s{}, s5, v5", MASKT[0]), &[s(MASKT[0])], &[s(5), v(5)])?;
        or_mask(b, MASKA, MASKT[0])?;
        op(b, format!("v_cmp_le_u32_e64 s{}, {nt}, v2", MASKT[0]), &[s(MASKT[0])], &[v(2)])?;
        or_mask(b, MASKA, MASKT[0])?;
        sop(b, format!("s_ctz_i32_b32 s{CNT}, s{MASKA}"), &[CNT], &[MASKA])?;
        // row_bytes = (K/256)*136 (QT44) or (K/128)*68 (QT53).
        let k = 19;
        match kind {
            Kind::GateUp => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 8"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x88"), &[ROWB], &[ROWB])?; }
            Kind::Down => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 7"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x44"), &[ROWB], &[ROWB])?; }
        }
        set_trips(b, a)?;
        // Output column base: gate/up 32x + 16*(w&1), down 64x + 16w. The
        // weight rows of an up wave are M/2 further.
        match kind {
            Kind::GateUp => {
                sop(b, format!("s_lshl_b32 s{RBASE}, s{WGX}, 5"), &[RBASE], &[WGX])?;
                sop(b, format!("s_and_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
            }
            Kind::Down => {
                sop(b, format!("s_lshl_b32 s{RBASE}, s{WGX}, {}", 6 + g.rr().trailing_zeros()), &[RBASE], &[WGX])?;
                sop(b, format!("s_mov_b32 s4, s{WAVE}"), &[4], &[WAVE])?;
            }
        }
        sop(b, "s_lshl_b32 s4, s4, 4", &[4], &[4])?;
        sop(b, format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)), &[RBASE], &[RBASE, 4])?;
        // Y row length and per-tile Y step (16 rows of BF16).
        match kind {
            Kind::GateUp => sop(b, "s_lshr_b32 s5, s18, 1", &[5], &[18])?,
            Kind::Down => sop(b, "s_mov_b32 s5, s18", &[5], &[18])?,
        }
        sop(b, format!("s_lshl_b32 s{YSTEP}, s5, 5"), &[YSTEP], &[5])?;
        // Y offset: ((16*tile + lr)*row_len + col + 8*hi)*2.
        op(b, format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"), &[v(8)], &[s(TILE16), v(3)])?;
        op(b, "v_mul_lo_u32 v8, v8, s5", &[v(8)], &[v(8), s(5)])?;
        op(b, format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"), &[v(9)], &[v(4), s(RBASE)])?;
        op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
        op(b, format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"), &[v(YOFF)], &[v(8)])?;
        if kind == Kind::GateUp {
            sop(b, format!("s_lshr_b32 s4, s{WAVE}, 1"), &[4], &[WAVE])?;
            sop(b, "s_mul_i32 s4, s4, s5", &[4], &[4, 5])?;
            sop(b, format!("{} s{RBASE}, s{RBASE}, s4", s_add_i32(a)), &[RBASE], &[RBASE, 4])?;
        }
        // Slots of the cnt tiles: sorted[16*(tile+j) + lr] through a
        // descriptor of cnt*64 bytes; dead tiles' slots read -1 below.
        sop(b, format!("s_lshl_b32 s4, s{WGY}, 6"), &[4], &[WGY])?;
        add64(b, SRD_S, 12, 4)?;
        sop(b, format!("s_lshl_b32 s{}, s{CNT}, 6", SRD_S + 2), &[SRD_S + 2], &[CNT])?;
        sop(b, format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)), &[SRD_S + 3], &[])?;
        op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
        for j in 0..nt {
            if g.experimental() && j > 0 {
                // A dead tile issues no sorted-slot load; its slots are -1.
                op(wg.isa(), format!("v_mov_b32_e32 v{}, -1", 6 + j), &[v(6 + j)], &[])?;
                let live = wg.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
                wg.skip_unless(live, &g.label(&format!("pro_s_dead{j}")), (), |w, ()| bload(w.isa(), 1, 6 + j, 5, SRD_S, 64 * u32::from(j)))?;
            } else {
                bload(wg.isa(), 1, 6 + j, 5, SRD_S, 64 * u32::from(j))?;
            }
        }
        let b = wg.isa();
        // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
        x_descriptors(b)?;
        sop(b, format!("s_lshl_b32 s{XS2}, s4, 1"), &[XS2], &[4])?;
        // Weight descriptor: rows rbase.. of the expert.
        w_descriptor(b)?;
        op(b, format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"), &[v(WOFF)], &[s(ROWB), v(3)])?;
        if a.gfx12() { op(b, format!("v_lshl_add_u32 v{WOFF}, v4, 3, v{WOFF}"), &[v(WOFF)], &[v(4), v(WOFF)])?; }
        if a.gfx12() {
            op(b, "v_and_b32_e32 v14, 7, v3", &[v(14)], &[v(3)])?;
            op(b, "v_lshl_or_b32 v14, v4, 3, v14", &[v(14)], &[v(4), v(14)])?;
        } else {
            op(b, "v_and_b32_e32 v14, 14, v3", &[v(14)], &[v(3)])?;
            op(b, "v_or_b32_e32 v14, v14, v4", &[v(14)], &[v(14), v(4)])?;
        }
        op(b, format!("v_mul_u32_u24_e32 v{HOFF}, s{ROWB}, v14"), &[v(HOFF)], &[s(ROWB), v(14)])?;
        load_w(b, g, 0)?;
        // Per tile: dead tiles (j >= cnt) gather slot -1, then the activation
        // row offset (or the out-of-range offset of a padding slot).
        for j in 0..nt {
            let slot = 6 + j;
            let b = wg.isa();
            if j > 0 && !g.experimental() {
                let m = MASKT[1];
                sop(b, format!("s_cmp_gt_u32 s{CNT}, {j}"), &[], &[CNT])?;
                sop(b, format!("s_cselect_b32 s{m}, -1, 0"), &[m], &[])?;
                sop(b, format!("s_mov_b32 s{}, 0", m + 1), &[m + 1], &[])?;
                op(b, format!("v_cndmask_b32_e64 v{slot}, -1, v{slot}, s{m}"), &[v(slot)], &[v(slot), s(m)])?;
            }
            common::gather_offset(b, XOFF + j, slot, Some(20), iu4_fold::XBLK_BYTES, g.live(j), GatherTemps { v: [14, 15, 1, 2], mask: MASKT[0] })?;
            if a.gfx12() { op(b, format!("v_lshl_add_u32 v{}, v4, 3, v{}", XQOFF + j, XOFF + j), &[v(XQOFF + j)], &[v(4), v(XOFF + j)])?; }
            load_x_init(wg, g, j)?;
        }
        let b = wg.isa();
        for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)), &[v(MAGIC8 + j)], &[])?; }
        for t in 0..nt { for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j), &[v(g.sum(t) + j)], &[])?; } }
        if g.settles_loads() { super::super::gemm_uk::Prefetch::<0>::wait_load(b)?; }
        Ok(())
    }

    /// Weight set `p` (epoch parity p of the current trip): header word, then
    /// the nibbles, as one clause.
    fn load_w(b: &mut Builder, g: &G, p: usize) -> Result<(), String> {
        let (width, step) = if g.arch().gfx12() { (2u8, 2u8) } else { (4u8, 4u8) };
        b.clause(|b| {
            bload(b, 1, g.wh(p), HOFF, SRD_W, g.header_off(p))?;
            for i in 0..4u8 { bload(b, width, g.wa(p) + step * i, WOFF, SRD_W, g.nibble_off(p) + 16 * u32::from(i))?; }
            Ok(())
        })
    }

    /// Tile `j`'s activations of the epoch `srd` points at: d, then nibbles.
    fn load_x(b: &mut Builder, g: &G, j: u8, srd: u8) -> Result<(), String> {
        let gfx12 = g.arch().gfx12();
        let (width, step) = if gfx12 { (2u8, 2u8) } else { (4u8, 4u8) };
        let xq = if gfx12 { XQOFF + j } else { XOFF + j };
        b.clause(|b| {
            bload(b, 1, g.d(j), XOFF + j, srd, 0)?;
            for i in 0..4u8 { bload(b, width, g.x(j) + step * i, xq, srd, 8 + 16 * u32::from(i))?; }
            Ok(())
        })
    }

    /// `load_x` of tile `j`; an experimental schedule issues no load for a
    /// dead whole tile (`j >= cnt`, wave-uniform), tile 0 is always live.
    fn load_x_live<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, j: u8, srd: u8, site: &str) -> Result<(), String> {
        if !g.experimental() || j == 0 { return load_x(w.isa(), g, j, srd); }
        let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
        w.skip_unless(live, &g.label(&format!("{site}_dead{j}")), (), |w, ()| load_x(w.isa(), g, j, srd))
    }

    /// Tile `j`'s initial activations (epoch 0). An experimental schedule
    /// defines a dead whole tile's fragments (`d` and `x`) with zeros on a
    /// dead arm instead of loading them. `if_else` branches to the live
    /// else arm when `cnt > j`; tile 0 is always live.
    fn load_x_init<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, j: u8) -> Result<(), String> {
        if !g.experimental() || j == 0 { return load_x(w.isa(), g, j, SRD_X[0]); }
        let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
        let join = g.label(&format!("pro_x_join{j}"));
        w.if_else(live, &g.label(&format!("pro_x_live{j}")), &join, |w| {
            let b = w.isa();
            op(b, format!("v_mov_b32_e32 v{}, 0", g.d(j)), &[v(g.d(j))], &[])?;
            for i in 0..g.aw() { op(b, format!("v_mov_b32_e32 v{}, 0", g.x(j) + i), &[v(g.x(j) + i)], &[])?; }
            Ok(())
        }, |w| load_x(w.isa(), g, j, SRD_X[0]))?;
        w.label(&join)
    }

    /// Fold weight set `p` into every live tile's sums; with `reload`, each
    /// tile's next-epoch activations are issued (through that descriptor)
    /// right after its fold; experimental schedules skip dead whole tiles.
    fn compute<T: Target + peacemaker_author::MmaIu4>(w: &mut Wave<'_, T, Builder>, g: &G, p: usize, site: &str, reload: Option<u8>) -> Result<(), String> {
        let b = w.isa();
        let a = g.arch();
        op(b, format!("v_cvt_f32_f16_e64 v{HF}, v{}.l", g.wh(p)), &[v(HF)], &[v(g.wh(p))])?;
        for j in 0..8u8 {
            let k = if a.gfx12() { j } else { 2 * j };
            let text = format!("ds_swizzle_b32 v{}, v{HF} offset:swizzle(BROADCAST,16,{k})", SCF + j);
            b.ds_crosslane(crate::insn::Instruction::new(text, vec![v(SCF + j)], vec![v(HF)]).memory(MemoryClass::DsLoad))?;
        }
        for i in 0..g.aw() {
            let r = g.wa(p) + i;
            op(b, format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)), &[v(r)], &[v(r)])?;
        }
        if !g.prefetch {
            super::super::gemm_uk::Prefetch::<0>::wait_load(b)?;
        }
        if g.chains == 1 { return serial_tiles(w, g, p, site, reload); }
        // The full-live path has no per-step predicates. Ragged runs retain
        // the incumbent's skipped-tile arithmetic and output ownership.
        for j in 0..g.nt() {
            let b = w.isa();
            let x = Instruction::new("", vec![], vec![v(g.d(j)), crate::reg::RegRef { kind: crate::reg::Kind::V, base: g.x(j), len: g.aw() }]);
            for (c, n, _) in b.ledger.required(&x) { b.wait(c, n)?; }
        }
        let full = w.scmp(Instruction::new(format!("s_cmp_lg_u32 s{CNT}, {}", g.nt()), vec![], vec![s(CNT)]))?;
        w.if_else(full, &g.label(&format!("{site}_partial")), &g.label(&format!("{site}_join")), |w| {
            match g.chains {
                2 => full_tiles::<2, T>(w, g, p, reload),
                4 => full_tiles::<4, T>(w, g, p, reload),
                8 => full_tiles::<8, T>(w, g, p, reload),
                _ => unreachable!("validated chain count"),
            }
        }, |w| serial_tiles(w, g, p, site, reload))?;
        w.label(&g.label(&format!("{site}_join")))
    }

    fn full_tiles<const N: usize, T: Target + peacemaker_author::MmaIu4>(
        w: &mut Wave<'_, T, Builder>, g: &G, p: usize, reload: Option<u8>,
    ) -> Result<(), String> {
        use super::super::gemm_uk::{Chain, Iu4};
        let chain = Chain::<N, Iu4>::new(std::array::from_fn(|j| V::<8>(g.cacc(j as u8))), V::<8>(MAGIC8))?;
        for first in (0..g.nt()).step_by(N) {
            for i in 0..g.aw() / 2 {
                chain.step(w, V::<2>(g.wa(p) + 2 * i),
                    std::array::from_fn(|j| V::<2>(g.x(first + j as u8) + 2 * i)), i == 0)?;
            }
            for j in first..first + N as u8 {
                let single = Chain::<1, Iu4>::new([V::<8>(g.cacc(j - first))], V::<8>(MAGIC8))?;
                single.fold(w.isa(), [g.sum(j)], SCF, [g.d(j)], TPROD)?;
                if let Some(srd) = reload { load_x(w.isa(), g, j, srd)?; }
            }
        }
        Ok(())
    }

    fn serial_tiles<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, p: usize, site: &str, reload: Option<u8>) -> Result<(), String> {
        let a = g.arch();
        // Experimental schedules reload only live tiles, inside the live
        // branch; the original reloads every tile after its (maybe skipped) fold.
        let (inside, outside) = if g.experimental() { (reload, None) } else { (None, reload) };
        for j in 0..g.nt() {
            let fold = |w: &mut Wave<'_, T, Builder>, ()| -> Result<(), String> {
                let b = w.isa();
                for i in 0..g.aw() / 2 {
                    iu4_fold::wmma_step(b, a, V::<8>(CACC), V::<2>(g.wa(p) + 2 * i), V::<2>(g.x(j) + 2 * i), i == 0, V::<8>(MAGIC8))?;
                }
                iu4_fold::fold_pass(b, CACC, g.sum(j), SCF, g.d(j), TPROD)?;
                if let Some(srd) = inside { load_x(w.isa(), g, j, srd)?; }
                Ok(())
            };
            if j > 0 {
                let b = w.isa();
                let x = Instruction::new("", vec![], vec![v(g.d(j)), crate::reg::RegRef { kind: crate::reg::Kind::V, base: g.x(j), len: g.aw() }]);
                for (c, n, _) in b.ledger.required(&x) { b.wait(c, n)?; }
                let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
                w.skip_unless(live, &g.label(&format!("{site}_dead{j}")), (), fold)?;
            } else { fold(w, ())?; }
            if let Some(srd) = outside { load_x(w.isa(), g, j, srd)?; }
        }
        Ok(())
    }

    fn advance(b: &mut Builder) -> Result<(), String> {
        add64_imm(b, SRD_W, 136)?;
        for r in SRD_X { add64(b, r, r, XS2)?; }
        Ok(())
    }

    fn kloop<T: Target + peacemaker_author::MmaIu4>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        wg.label(&g.label("k_begin"))?;
        let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
        let entry = wg.isa().ledger.shape();
        let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
        wg.wg_skip_if(no_trip, &done, (), |wg, ()| {
            wg.loop_carried(&head, (), |wg, ()| {
                if g.prefetch {
                    load_w(wg.isa(), g, 1)?;
                    compute(wg, g, 0, "l0", Some(SRD_X[1]))?;
                    advance(wg.isa())?;
                    load_w(wg.isa(), g, 0)?;
                    compute(wg, g, 1, "l1", Some(SRD_X[0]))?;
                } else {
                    compute(wg, g, 0, "l0", None)?;
                    load_epoch(wg, g, 1, "l0x")?;
                    advance(wg.isa())?;
                    compute(wg, g, 1, "l1", None)?;
                    load_epoch(wg, g, 0, "l1x")?;
                }
                if g.settles_loads() { super::super::gemm_uk::Prefetch::<0>::wait_load(wg.isa())?; }
                let b = wg.isa();
                op(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)), &[s(TRIPS)], &[s(TRIPS)])?;
                Ok(((), trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
            })?;
            if wg.isa().ledger.shape() != entry { return Err("k loop exit ledger differs from its entry".into()) }
            Ok(())
        })?;
        match g.spec.kind {
            // K % 256 == 0: an even epoch count, two epochs left.
            Kind::GateUp => {
                tail_pair(wg, g)
            }
            // K % 128 == 0: one epoch left when K/128 is odd, else two.
            Kind::Down => {
                let odd = wg.scmp(Instruction::new("s_bitcmp1_b32 s19, 7", vec![], vec![s(19)]))?;
                wg.if_else(odd, &g.label("tail_odd"), &g.label("epilogue"), |w| {
                    tail_pair(w, g)
                }, |w| compute(w, g, 0, "t2", None))
            }
        }
    }

    fn load_epoch<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, p: usize, site: &str) -> Result<(), String> {
        load_w(w.isa(), g, p)?;
        for j in 0..g.nt() { load_x_live(w, g, j, SRD_X[p], site)?; }
        Ok(())
    }

    fn tail_pair<T: Target + peacemaker_author::MmaIu4>(w: &mut Wave<'_, T, Builder>, g: &G) -> Result<(), String> {
        if g.prefetch {
            load_w(w.isa(), g, 1)?;
            compute(w, g, 0, "t0", Some(SRD_X[1]))?;
        } else {
            compute(w, g, 0, "t0", None)?;
            load_epoch(w, g, 1, "t0x")?;
        }
        compute(w, g, 1, "t1", None)
    }

    /// RNE BF16 bits of `vals` (8 rows), +0 on padding slots, packed to this
    /// lane's 8 consecutive output rows and stored at the current `SRD_Y`.
    fn store_rows(b: &mut Builder, g: &G, vals: u8, live: u8) -> Result<(), String> {
        for j in 0..8u8 { Bf16::rne_finite_passthrough(b, vals + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?; }
        for j in 0..8u8 { op(b, format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{live}", vals + j), &[v(vals + j)], &[v(vals + j), s(live)])?; }
        let perm = |b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String| -> Result<(), String> {
            let mut uses = vec![v(hi_src), v(lo_src)];
            if sel.starts_with('v') { uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?)); }
            op(b, format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"), &[v(dst)], &uses)
        };
        let out = g.out();
        if g.arch().gfx12() {
            for k in 0..4u8 { perm(b, out + k, vals + 2 * k + 1, vals + 2 * k, lit(0x0706_0302))?; }
        } else {
            // The 16-slot entry's v_permlanex16 half exchange.
            let (own, send, recv, sel, pack) = (g.own(), g.send(), g.recv(), g.sel(), g.pack());
            for k in 0..4u8 {
                op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}", own + k, vals + k, vals + 4 + k), &[v(own + k)], &[v(vals + k), v(vals + 4 + k), s(MASKA)])?;
            }
            for m in 0..2u8 {
                perm(b, pack + 2 * m, vals + 5 + 2 * m, vals + 4 + 2 * m, lit(0x0706_0302))?;
                perm(b, pack + 2 * m + 1, vals + 1 + 2 * m, vals + 2 * m, lit(0x0706_0302))?;
                op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{MASKA}", send + m, pack + 2 * m, pack + 2 * m + 1), &[v(send + m)], &[v(pack + 2 * m), v(pack + 2 * m + 1), s(MASKA)])?;
                op(b, format!("v_mov_b32_e32 v{}, v{}", recv + m, send + m), &[v(recv + m)], &[v(send + m)])?;
                op(b, format!("v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98", recv + m, send + m), &[v(recv + m)], &[v(recv + m), v(send + m), s(PSEL)])?;
            }
            for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)].into_iter().enumerate() {
                let r = sel + i as u8;
                op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
                op(b, format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{MASKA}", lit(lo)), &[v(r)], &[v(r), s(MASKA)])?;
            }
            for k in 0..4u8 { perm(b, out + k, recv + k / 2, own + k, format!("v{}", sel + k % 2))?; }
        }
        bstore_b128(b, out, YOFF, SRD_Y, 128 * u32::from(g.blk))
    }

    /// Stores of every live tile: tile 0 always, tile j while j < cnt; the
    /// descriptor base advances 16 output rows per tile.
    fn store_tiles<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G, end: &End, vals: impl Fn(&mut Wave<'_, T, Builder>, u8) -> Result<u8, String>) -> Result<(), String> {
        let b = w.isa();
        sop(b, format!("s_mov_b32 s{SRD_Y}, s16"), &[SRD_Y], &[16])?;
        sop(b, format!("s_mov_b32 s{}, s17", SRD_Y + 1), &[SRD_Y + 1], &[17])?;
        srd_tail(b, SRD_Y, None)?;
        if !g.arch().gfx12() {
            op(b, format!("s_mov_b32 s{MASKA}, 0xffff0000"), &[s(MASKA)], &[])?;
            op(b, format!("s_mov_b32 s{}, 0", MASKA + 1), &[s(MASKA + 1)], &[])?;
            op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        }
        for j in 0..g.nt() {
            if j > 0 {
                let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
                w.exit_unless(live, end)?;
                add64(w.isa(), SRD_Y, SRD_Y, YSTEP)?;
            }
            let r = vals(w, j)?;
            store_rows(w.isa(), g, r, g.live(j))?;
        }
        Ok(())
    }

    /// Stores of a non-final row block: `store_tiles` with a dead tile
    /// skipping its store (the wave goes on to the next block) in place of
    /// leaving the kernel. Gfx11 only.
    fn store_tiles_mid<T: Target>(w: &mut Wave<'_, T, Builder>, g: &G) -> Result<(), String> {
        let b = w.isa();
        sop(b, format!("s_mov_b32 s{SRD_Y}, s16"), &[SRD_Y], &[16])?;
        sop(b, format!("s_mov_b32 s{}, s17", SRD_Y + 1), &[SRD_Y + 1], &[17])?;
        srd_tail(b, SRD_Y, None)?;
        op(b, format!("s_mov_b32 s{MASKA}, 0xffff0000"), &[s(MASKA)], &[])?;
        op(b, format!("s_mov_b32 s{}, 0", MASKA + 1), &[s(MASKA + 1)], &[])?;
        op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        for j in 0..g.nt() {
            if j == 0 {
                store_rows(w.isa(), g, g.sum(0), g.live(0))?;
                continue;
            }
            let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
            w.skip_unless(live, &g.label(&format!("store_dead{j}")), (), |w: &mut Wave<'_, T, Builder>, ()| {
                add64(w.isa(), SRD_Y, SRD_Y, YSTEP)?;
                store_rows(w.isa(), g, g.sum(j), g.live(j))
            })?;
        }
        Ok(())
    }

    /// Row block `g` to block `g + 1`: reset the descriptors and the trip
    /// count, issue the next block's first-epoch weights, store this
    /// block's tiles, then reload the next block's first-epoch activations
    /// and zero the sums. Pending stores are drained so the next K loop
    /// enters with the ledger shape its back edge leaves.
    fn transition<T: Target>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        let nx = g.at(g.blk + 1);
        wg.label(&g.label("epilogue"))?;
        let b = wg.isa();
        x_descriptors(b)?;
        sop(b, format!("{} s{RBASE}, s{RBASE}, 64", s_add_i32(g.arch())), &[RBASE], &[RBASE])?;
        w_descriptor(b)?;
        set_trips(b, g.arch())?;
        load_w(b, &nx, 0)?;
        store_tiles_mid(wg, g)?;
        wg.label(&g.label("xreload"))?;
        let b = wg.isa();
        for j in 0..g.nt() { load_x(b, g, j, SRD_X[0])?; }
        for t in 0..g.nt() { for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, 0", g.sum(t) + j), &[v(g.sum(t) + j)], &[])?; } }
        b.wait(crate::ledger::Counter::Vs, 0)
    }

    /// Tile `j`'s gate publication (two b128 stores per lane) in the linear
    /// write token `out`.
    fn publish_tile<T: Target, S: StoreTarget<T::Waits, Out = S>>(w: &mut Wave<'_, T, Builder>, g: &G, j: u8, mut out: S) -> Result<S, String> {
        for h in 0..2u8 {
            let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
            let d = crate::reg::RegRef { kind: crate::reg::Kind::V, base: g.sum(j) + 4 * h, len: 4 };
            let text = format!("ds_store_b128 v{LDSA}, {d}{}", if off == 0 { String::new() } else { format!(" offset:{off}") });
            out = w.ds_store(out, Instruction::new(text, vec![], vec![v(LDSA), d]).memory(MemoryClass::DsStore))?;
        }
        Ok(out)
    }
    fn epilogue<T: Target>(wg: &mut Wg<T>, g: &G, end: End, gate: Option<LdsRegion<Gate, Free>>) -> Result<(), String> {
        let b = wg.isa();
        b.label(&g.label("epilogue"))?;
        let Some(gate) = gate else {
            store_tiles(wg, g, &end, |_, j| Ok(g.sum(j)))?;
            return wg.end(end);
        };
        // Every wave: BF16 round trip of its sums (gate or up).
        for j in 0..g.nt() { for i in 0..8u8 { Bf16::rne_finite_passthrough(b, g.sum(j) + i, RT_TMP + i, MASKT[usize::from(i % 2)], true)?; } }
        // Gate waves publish the rounded gate rows and leave; up waves meet
        // them at their own barrier, read the publication and store the
        // tiles (wave scope, up to the kernel exit).
        let up = wg.scmp(Instruction::new(format!("s_cmp_ge_u32 s{WAVE}, 2"), vec![], vec![s(WAVE)]))?;
        let gv = g.gval();
        wg.handoff(up, &g.label("up"), end, gate, |w, gate| {
            let mut out = w.begin_write(gate);
            for j in 0..g.nt() {
                if g.experimental() && j > 0 {
                    let live = w.scmp(Instruction::new(format!("s_cmp_gt_u32 s{CNT}, {j}"), vec![], vec![s(CNT)]))?;
                    out = w.skip_unless(live, &g.label(&format!("pub_dead{j}")), out, |w, out| publish_tile(w, g, j, out))?;
                } else {
                    out = publish_tile(w, g, j, out)?;
                }
            }
            Ok(out)
        }, |w, gate, end| store_tiles(w, g, end, |w, j| {
            for h in 0..2u8 {
                let off = u32::from(j) * NT_GATE_BYTES + 16 * u32::from(h);
                let d = crate::reg::RegRef { kind: crate::reg::Kind::V, base: gv + 4 * h, len: 4 };
                let text = format!("ds_load_b128 {d}, v{LDSA}{}", if off == 0 { String::new() } else { format!(" offset:{off}") });
                w.ds_load(&gate, Instruction::new(text, vec![d], vec![v(LDSA)]).memory(MemoryClass::DsLoad))?;
            }
            for grp in 0..2u8 { crate::kernels::gemm_uk::Epilogue::silu_dense(w.isa(), gv + SILU_N * grp, g.sum(j) + SILU_N * grp, SILU_TMP, SILU_MASK, SILU_N)?; }
            Ok(gv)
        }))
    }

    pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
        emit_with(R4Spec { base: spec, chains: 1, prefetch: true, grouped: false })
    }

    pub(super) fn emit_with(r4: R4Spec) -> Result<Emitted, String> {
        let spec = r4.base;
        let g = G { spec, blk: 0, chains: r4.chains, prefetch: r4.prefetch, grouped: r4.grouped };
        let experimental = g.experimental();
        let suffix = if experimental { format!("_r4c{}p{}{}", g.chains, u8::from(g.prefetch), if g.grouped { "g" } else { "" }) } else { String::new() };
        let mut kernargs = spec.kernargs();
        if g.grouped { kernargs.size = 72; kernargs = kernargs.pointer("tile_leaders", 64); }
        let kspec = KernelSpec {
            kernel_id: "qwen4_moe_sym".into(), variant: format!("{}{suffix}", spec.variant()), arch: spec.arch, symbol: format!("{}{suffix}", spec.symbol()),
            kernargs, user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
            workgroup_size: spec.threads() as u16, group_segment_fixed_size: spec.lds_bytes(), wave32: true, cu_mode: false,
        };
        let mut b = Builder::new(kspec, g.plan()?);
        b.enable_delay_alu();
        on_target!(spec.arch, &mut b, body(&g))?;
        b.finish()
    }

    fn body<T: Target + peacemaker_author::MmaIu4>(wg: &mut Wg<T>, g: &G) -> Result<(), String> {
        let gate = if g.gate_up() { Some(wg.lds::<Gate>("swiglu_gate", 0, g.spec.lds_bytes())?) } else { None };
        let end = wg.exit(&g.exit_label())?;
        prologue(wg, g, &end)?;
        for blk in 0..g.rr() - 1 {
            let gb = g.at(blk);
            kloop(wg, &gb)?;
            transition(wg, &gb)?;
        }
        let last = g.at(g.rr() - 1);
        kloop(wg, &last)?;
        epilogue(wg, &last, end, gate)
    }
}
