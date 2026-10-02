//! Builder-emitted gfx1151 (RDNA3.5, wave32) MQ4V2 x block_i4_128 GEMM with
//! the hipcc V2B algorithm (`kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip`),
//! bit for bit, for its three epilogues: SET, ADD (FFN down: residual touch
//! and the grouped raster of `_add_touch_swz`) and the F1-lite gate/up SiLU.
//!
//! Tile M256 x N256, 16 waves, one CTA per WGP; wave `w` owns rows
//! `64*(w/4)..+64` (four 16-row fragments `a`) and tokens `64*(w%4)..+64`
//! (four 16-token fragments `c`). K is walked in K128 epochs with two 32 KiB
//! LDS slots (A 16 KiB, X 16 KiB, fragment-major: K16 slice `s` of row/token
//! `r` at `((r/16)*8+s)*128 + (r%16)*8`). Epoch `e` reads slot `e%2` while
//! the staging registers carry epoch `e+1` into slot `(e+1)%2`; one
//! workgroup barrier per epoch.
//!
//! Numerics, per output and ascending epoch `e`: `C_e` is the exact int32
//! `v_wmma_i32_16x16x16_iu4` chain over the eight K16 slices seeded with the
//! magic `0x4b400000`; `sum = fma(RN(d_e * sc_e), C_e + (-12582912.0), sum)`
//! from `sum = +0`, exactly V2B's fold DAG. A is rebiased once at staging
//! (XOR 0x88888888, the symmetric `zp == -8*sc` contract).
//!
//! Every one of the 384 fold ops per epoch and wave is VOPD-paired (hipcc
//! pairs 342 in SET and 260 in gate/up). Scale words are laid out so each
//! word's rows come from one weight tensor: word `w` of lane `(hi, lr)` holds
//! the row of wave fragment `w + 2*(lr>>3)`, so pass `a` shares word `a&1`
//! from lane `8*(a>>1) + j` (gate rows and up rows never share a word), as a
//! `ds_swizzle_b32` broadcast on the LDS crossbar issued ahead of the fold.
//! Fold products sit at `t + (j^2)` so no fmac half reads one VGPR bank
//! three times. The SET stores each pass's final sums one b128 per WMMA step
//! during the last epoch; ADD and SiLU store after the K loop.
//!
//! Launch contracts (block `[512,1,1]`, dynamic LDS 65536; `M % 256 ==
//! N % 256 == 0`, `K % 256 == 0`; Y token-major `[N][M]` f32):
//! - SET  `A, Xq, Y, M, K, N`, grid `[N/256, M/256]`.
//! - ADD  `A, Xq, Y, M, K, N, GSHIFT`, `K >= 2048`, grid
//!   `[(N/256) << GSHIFT, (M/256) >> GSHIFT]`, `(M/256) % (1 << GSHIFT) == 0`:
//!   workgroup `(x, y)` computes row tile `y*G + x%G` and token tile `x/G`
//!   (`G = 1 << GSHIFT`), which is `_add_touch_swz`'s dispatch order. The
//!   residual tile is touched over epochs `E-16..E-9` like `_add_touch`.
//! - SiLU `G, U, Xq, Y, M, K, N`, grid `[N/256, 2M/256]`; `Y = h [N][M]`,
//!   `h = g / (1 + expf(-g)) * u` with hipcc's exact expf/fdiv expansion.
use super::common::{add64, add64_imm, lit, mem, op, s, sop, sr, v, vr};
use super::iu4_fold::{self, GROUP_BYTES, MAGIC, REBIAS, XBLK_BYTES};
use super::iu4_gemm::ds_offsets;
use super::iu4_v2c::off;
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V,
    insn::{Instruction, MemoryClass, Sop, Wmma},
    reg::Live, vopd::{Operand, VopdF32, VopdOp}};
use peacemaker_author::{Free, Gfx1151, Gfx11Waits, LdsWrite, MmaIu4, Pending, Published, Ring, Scc, State, Wave, WgUniform,
    Workgroup, Writing, prime, rotate};

pub const THREADS: u16 = 512;
pub const WAVES: u32 = 16;
pub const TILE: u32 = 256;
pub const LDS_BYTES: u32 = 65536;
pub const SLOT_BYTES: u32 = 32768;
pub const A_BYTES: u32 = 16384;
pub const VGPR_CEILING: u16 = 256;
/// ADD residual touch window: epochs `E-TOUCH_LEAD .. E-TOUCH_LEAD+8`.
pub const TOUCH_LEAD: u32 = 16;
/// Smallest K the ADD entry accepts (its touch window must fit the loop).
pub const ADD_MIN_K: u32 = 2048;
pub const MODULE: &str = "gemm_mq4g256v2_residual_iu4_pm_v2b_gfx1151";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { Set, Add, GateUpSilu }
impl Epi {
    pub const ALL: [Epi; 3] = [Epi::Set, Epi::Add, Epi::GateUpSilu];
    pub fn tag(self) -> &'static str { match self { Self::Set => "set", Self::Add => "add", Self::GateUpSilu => "silu" } }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(t: &str) -> Result<Self, String> {
        match t { "set" => Ok(Self::Set), "add" => Ok(Self::Add), "silu" => Ok(Self::GateUpSilu), _ => Err(format!("iu4_v2b epilogue {t} (set|add|silu)")) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub epi: Epi }

impl Spec {
    pub fn symbol(self) -> String {
        match self.epi {
            Epi::Set => format!("gemm_mq4g256v2_residual_iu4_pm_v2b_set_{}", self.arch.name()),
            Epi::Add => format!("gemm_mq4g256v2_residual_iu4_pm_v2b_add_{}", self.arch.name()),
            Epi::GateUpSilu => format!("gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_{}", self.arch.name()),
        }
    }
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1151 { return Err("iu4_v2b: the V2B tile is built for gfx1151 only (gfx1100 ships V2C)".into()) }
        Ok(())
    }
    pub fn kernargs(self) -> KernargLayout {
        match self.epi {
            Epi::Set => KernargLayout::new(36).pointer("A", 0).pointer("Xq", 8).pointer("Y", 16)
                .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("N", 32, 4, "by_value"),
            Epi::Add => KernargLayout::new(40).pointer("A", 0).pointer("Xq", 8).pointer("Y", 16)
                .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("N", 32, 4, "by_value")
                .hidden("GSHIFT", 36, 4, "by_value"),
            Epi::GateUpSilu => KernargLayout::new(44).pointer("G", 0).pointer("U", 8).pointer("Xq", 16).pointer("Y", 24)
                .hidden("M", 32, 4, "by_value").hidden("K", 36, 4, "by_value").hidden("N", 40, 4, "by_value"),
        }
    }
}

/// Physical registers. VGPRs above v159 and v0..v31 are reused by the
/// epilogue once the K loop is done.
struct Regs;
impl Regs {
    const C: u8 = 0;          // int32 WMMA chains C[c] = v[8c..8c+7]
    const ACC: u8 = 32;       // f32 sums acc[a][c] = v[32 + 8*(4a+c) ..]
    const MAGIC: u8 = 160;    // 8 x 0x4b400000, the chain seed
    const AV: [u8; 2] = [168, 172];  // A fragment pairs (slice s, s+1)
    const XV: [u8; 2] = [176, 184];  // X fragments c=0..3 of one slice
    const SCF: u8 = 192;      // row scales of the fold pass (ds_swizzle broadcast)
    const T: u8 = 200;        // fold products t = d*sc
    const DC: [u8; 2] = [208, 212];  // d of the four token fragments, by epoch parity
    const WSF: [u8; 2] = [220, 255]; // f32 scale words (v255 keeps the reservation a whole granule)
    const WS: [u8; 2] = [221, 222];  // raw f16 scale words
    const YOFF: u8 = 223;     // Y byte offset of (token wc*64+lr, row 8hi of the wave)
    const STA: u8 = 224;      // staged A (4 x b64)
    const STX: u8 = 232;      // staged X (4 x b64)
    const A_OFF: u8 = 240; const X_OFF: u8 = 241; const LWS: u8 = 242; const LXS: u8 = 243; const ST: u8 = 244;
    const AB: [[u8; 2]; 2] = [[245, 246], [247, 248]];
    const XB: [[u8; 2]; 2] = [[249, 250], [251, 252]];
    const TV: [u8; 2] = [216, 217];  // ADD residual touch destinations
    // SGPRs
    const KARG: u8 = 0; const WGX: u8 = 2; const WGY: u8 = 3;
    const T0: u8 = 4; const T1: u8 = 5; const T64: u8 = 6;
    const ARGS: u8 = 8; const ARGS2: u8 = 16; const ARGS3: u8 = 18;
    const ROWB: u8 = 20; const WAVE: u8 = 21; const TRIPS: u8 = 22; const WR: u8 = 23; const WC: u8 = 24;
    const N72: u8 = 25; const M64: u8 = 26; const BY: u8 = 27;
    const SS: u8 = 28; const SW0: u8 = 30; const SW1: u8 = 32; const SX: u8 = 34; const SY: u8 = 36; const STB: u8 = 38;
    const BX: u8 = 40; const MUL: u8 = 42;
    /// SiLU epilogue per-element masks (underflow, overflow, numerator
    /// scale), each an aligned pair: M7 sizes a VOP3 SGPR destination as a
    /// lane-mask pair.
    const MASK: u8 = 48;
    /// SET with early stores: Y bases of token fragments 1..3 (fragment 0
    /// uses SY), in the SiLU entry's mask range.
    const SYC: [u8; 3] = [48, 50, 52];
}

/// Kernel-argument SGPRs of one entry.
struct Args { a: u8, u: Option<u8>, xq: u8, y: u8, m: u8, k: u8, n: u8, gshift: Option<u8> }

/// SiLU elements folded per interleaved group.
const SILU_GROUP: u8 = 8;
const SILU_TEMPS: u8 = 6;
/// The staged packet is published after this pass's fold (see `epoch`).
const PUBLISH_AFTER_PASS: usize = 2;

/// What the fold of each epoch's last pass waits behind: it runs after the
/// epoch barrier, so the next epoch's first loads are in flight under it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sched {
    /// The next epoch's step-0 LDS loads.
    Defer,
    /// As `Defer`, with the next epoch's metadata and staging loads also
    /// issued ahead of fold(3).
    Early,
}

struct Gen { spec: Spec, args: Args, sched: Sched, early: bool }

impl Gen {
    fn new(spec: Spec) -> Self {
        let args = match spec.epi {
            Epi::Set => Args { a: 8, u: None, xq: 10, y: 12, m: 14, k: 15, n: 16, gshift: None },
            Epi::Add => Args { a: 8, u: None, xq: 10, y: 12, m: 14, k: 15, n: 16, gshift: Some(17) },
            Epi::GateUpSilu => Args { a: 8, u: Some(10), xq: 12, y: 14, m: 16, k: 17, n: 18, gshift: None },
        };
        // Screened on the Halo against hipcc V2B (clock-normalized): SET and
        // SiLU are fastest with `Early`; the ADD's residual touch loads sit
        // in the epoch head, which `Early` would move across the touch
        // loop's boundary, so it keeps `Defer`.
        let sched = if spec.epi == Epi::Add { Sched::Defer } else { Sched::Early };
        // The SET stores each pass's final sums during the last epoch
        // (`early_store`). The ADD's residual rings and the SiLU's
        // temporaries need VGPRs the K loop still holds, so they store after it.
        let early = spec.epi == Epi::Set;
        Self { spec, args, sched, early }
    }
    /// WMMA step after which pass 0's scale broadcast issues: its f16 scale
    /// word must have landed. `Early` loaded it before the previous epoch's
    /// last fold; `Defer` loads it at the epoch start, so the ADD waits for
    /// step 6 (HR4 screen, real H2 sites: SET/SiLU equal at steps 1-6, ADD
    /// +1.9% cycles at step 1, -0.7% at step 6).
    fn scales0_step(&self) -> usize {
        match self.sched { Sched::Early => 2, Sched::Defer => 6 }
    }
    fn label(&self, name: &str) -> String { format!(".Lv2b_{}_{name}", self.spec.epi.tag()) }
    fn silu(&self) -> bool { self.spec.epi == Epi::GateUpSilu }
    fn add(&self) -> bool { self.spec.epi == Epi::Add }
    /// Tile ids: the ADD entry's grouped raster, else the workgroup ids.
    fn by(&self) -> u8 { if self.add() { Regs::BY } else { Regs::WGY } }
    fn bx(&self) -> u8 { if self.add() { Regs::BX } else { Regs::WGX } }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kl = || Live::Between(l("k_begin"), l("epilogue"));
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        let epi = || Live::Between(l("epilogue"), l("end"));
        let body = || Live::Between(l("epi_body"), l("end"));
        for c in 0..4u8 { p.v::<8>("C", Regs::C + 8 * c, kl())?; }
        for i in 0..32u8 { p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, Regs::C + i, pro())?; }
        if !self.early { for i in 0..4u8 { p.v::<1>("y_off_c", Regs::C + i, epi())?; } }
        for k in 0..16u8 { p.v::<8>("acc", Regs::ACC + 8 * k, Live::Whole)?; }
        p.v::<8>("magic8", Regs::MAGIC, kernel())?;
        // The prologue issues the first epoch's step-0 fragment loads.
        for r in Regs::AV { p.v::<4>("a_frag_pair", r, kernel())?; }
        for r in Regs::XV { p.v::<8>("x_frags", r, kernel())?; }
        p.v::<8>("scale_rows", Regs::SCF, kl())?;
        p.v::<8>("fold_t", Regs::T, kl())?;
        // The early-head schedule issues epoch 0's metadata in the prologue.
        for r in Regs::DC { p.v::<4>("d_x", r, kernel())?; }
        for r in Regs::WSF { p.v::<1>("scale_f32", r, kl())?; }
        for r in Regs::WS { p.v::<1>("scale_f16", r, kernel())?; }
        p.v::<1>("y_off", Regs::YOFF, Live::Between("entry".into(), l(if self.early { "end" } else { "epi_body" })))?;
        for i in 0..4u8 { p.v::<2>("stage_a", Regs::STA + 2 * i, kernel())?; p.v::<2>("stage_x", Regs::STX + 2 * i, kernel())?; }
        for (name, r) in [("a_off", Regs::A_OFF), ("x_off", Regs::X_OFF), ("lws", Regs::LWS), ("lxs", Regs::LXS), ("st", Regs::ST),
            ("ab00", Regs::AB[0][0]), ("ab01", Regs::AB[0][1]), ("ab10", Regs::AB[1][0]), ("ab11", Regs::AB[1][1]),
            ("xb00", Regs::XB[0][0]), ("xb01", Regs::XB[0][1]), ("xb10", Regs::XB[1][0]), ("xb11", Regs::XB[1][1])] {
            p.v::<1>(name, r, kernel())?;
        }
        if self.add() { for r in Regs::TV { p.v::<1>("touch", r, kl())?; } }
        match self.spec.epi {
            Epi::Set => {}
            // Residual rings: three 32-register sets of one row fragment each.
            Epi::Add => for base in [160u8, 192, 224] { for i in 0..4u8 { p.v::<8>("residual", base + 8 * i, body())?; } },
            Epi::GateUpSilu => {
                for i in 0..(SILU_GROUP * SILU_TEMPS / 8) { p.v::<8>("silu_tmp", 160 + 8 * i, body())?; }
                for i in 0..(3 * SILU_GROUP) { p.s::<2>("silu_mask", Regs::MASK + 2 * i, body())?; }
            }
        }
        p.s::<2>("kernarg_ptr", Regs::KARG, Live::Whole)?;
        p.s::<1>("wg_x", Regs::WGX, Live::Whole)?;
        p.s::<1>("wg_y", Regs::WGY, Live::Whole)?;
        p.s::<1>("s_tmp0", Regs::T0, Live::Whole)?;
        p.s::<1>("s_tmp1", Regs::T1, Live::Whole)?;
        p.s::<2>("s_tmp64", Regs::T64, Live::Whole)?;
        p.s::<8>("kernargs_0x00", Regs::ARGS, Live::Whole)?;
        match self.spec.epi {
            Epi::Set => { p.s::<1>("kernargs_0x20", Regs::ARGS2, Live::Whole)?; }
            Epi::Add => { p.s::<2>("kernargs_0x20", Regs::ARGS2, Live::Whole)?; }
            Epi::GateUpSilu => { p.s::<2>("kernargs_0x20", Regs::ARGS2, Live::Whole)?; p.s::<1>("kernargs_0x28", Regs::ARGS3, Live::Whole)?; }
        }
        for (name, r) in [("row_bytes", Regs::ROWB), ("wave", Regs::WAVE), ("trips", Regs::TRIPS), ("wr", Regs::WR), ("wc", Regs::WC),
            ("n72", Regs::N72), ("m64", Regs::M64)] {
            p.s::<1>(name, r, Live::Whole)?;
        }
        for (name, r) in [("stage_base", Regs::SS), ("scale_base0", Regs::SW0), ("scale_base1", Regs::SW1), ("x_base", Regs::SX), ("y_base", Regs::SY), ("mul64", Regs::MUL)] {
            p.s::<2>(name, r, Live::Whole)?;
        }
        if self.add() {
            p.s::<1>("row_tile", Regs::BY, Live::Whole)?;
            p.s::<1>("token_tile", Regs::BX, Live::Whole)?;
            p.s::<2>("touch_base", Regs::STB, Live::Whole)?;
        }
        if self.early { for r in Regs::SYC { p.s::<2>("y_base_c", r, Live::Whole)?; } }
        Ok(p)
    }
}

/// LDS tags: the A (weight) and X (activation) halves of each 32 KiB slot.
pub enum A {}
pub enum X {}
type Wg<'b> = Workgroup<'b, Gfx1151, Builder>;
type Wv<'b> = Wave<'b, Gfx1151, Builder>;
/// The K loop's steady state: slot `e%2` of each ring published, the other free.
type Rings = (Ring<A, Published, Free>, Ring<X, Published, Free>);
/// A ring's next buffer with its pending stores.
type Staged<R, C> = (Ring<R, C, Writing>, Pending<Gfx11Waits, LdsWrite<R>>);

/// Slots A0, X0, A1, X1 (builder slot ids 0..3); slot 0 is written first.
fn declare_lds(wg: &mut Wg) -> Result<(Ring<A, Free, Free>, Ring<X, Free, Free>), String> {
    let (a0, x0) = (wg.lds("A0", 0, A_BYTES)?, wg.lds("X0", A_BYTES, A_BYTES)?);
    let (a1, x1) = (wg.lds("A1", SLOT_BYTES, A_BYTES)?, wg.lds("X1", SLOT_BYTES + A_BYTES, A_BYTES)?);
    Ok((Ring::new(a0, a1), Ring::new(x0, x1)))
}

/// `dst = base + x*y` (64-bit), `x*y` as a full 64-bit product.
fn mul_add64(b: &mut Builder, dst: u8, base: u8, x: u8, y: u8) -> Result<(), String> {
    let (lo, hi) = (Regs::MUL, Regs::MUL + 1);
    sop(b, format!("s_mul_i32 s{lo}, s{x}, s{y}"), &[lo], &[x, y])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{x}, s{y}"), &[hi], &[x, y])?;
    sop(b, format!("s_add_u32 s{dst}, s{base}, s{lo}"), &[dst], &[base, lo])?;
    sop(b, format!("s_addc_u32 s{}, s{}, s{hi}", dst + 1, base + 1), &[dst + 1], &[base + 1, hi])
}

/// Kernel arguments, tile bases and every lane-invariant offset.
fn prologue(wg: &mut Wg, g: &Gen, (ra, rx): (Ring<A, Free, Free>, Ring<X, Free, Free>)) -> Result<Rings, String> {
    let b = wg.isa();
    let a = &g.args;
    let (t0, t1) = (Regs::T0, Regs::T1);
    // gfx11 llvm-objdump spells a zero SMEM offset `null`; parse-back compares canonical text.
    mem(b, format!("s_load_b256 s[{}:{}], s[0:1], null", Regs::ARGS, Regs::ARGS + 7), &[sr(Regs::ARGS, 8)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    match g.spec.epi {
        Epi::Set => mem(b, format!("s_load_b32 s{}, s[0:1], 0x20", Regs::ARGS2), &[s(Regs::ARGS2)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?,
        Epi::Add => mem(b, format!("s_load_b64 s[{}:{}], s[0:1], 0x20", Regs::ARGS2, Regs::ARGS2 + 1), &[sr(Regs::ARGS2, 2)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?,
        Epi::GateUpSilu => {
            mem(b, format!("s_load_b64 s[{}:{}], s[0:1], 0x20", Regs::ARGS2, Regs::ARGS2 + 1), &[sr(Regs::ARGS2, 2)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
            mem(b, format!("s_load_b32 s{}, s[0:1], 0x28", Regs::ARGS3), &[s(Regs::ARGS3)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
        }
    }
    // v0 = tid; v1 = wave; v2 = lane; v3 = lr; v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
    op(b, format!("v_readfirstlane_b32 s{}, v1", Regs::WAVE), &[s(Regs::WAVE)], &[v(1)])?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    sop(b, format!("s_lshr_b32 s{}, s{}, 2", Regs::WR, Regs::WAVE), &[Regs::WR], &[Regs::WAVE])?;
    sop(b, format!("s_and_b32 s{}, s{}, 3", Regs::WC, Regs::WAVE), &[Regs::WC], &[Regs::WAVE])?;
    // row_bytes = (K/256)*136; trips = K/256 - 1 whole two-epoch trips before
    // the tail pair (ADD: K/256 - 8 before the touch phase).
    sop(b, format!("s_lshr_b32 s{}, s{}, 8", Regs::ROWB, a.k), &[Regs::ROWB], &[a.k])?;
    sop(b, format!("s_add_i32 s{}, s{}, {}", Regs::TRIPS, Regs::ROWB, if g.add() { "-8" } else { "-1" }), &[Regs::TRIPS], &[Regs::ROWB])?;
    sop(b, format!("s_mulk_i32 s{}, {}", Regs::ROWB, lit(GROUP_BYTES)), &[Regs::ROWB], &[Regs::ROWB])?;
    if let Some(gs) = a.gshift {
        // Grouped raster: row tile y*G + x%G, token tile x/G.
        sop(b, format!("s_lshl_b32 s{t0}, 1, s{gs}"), &[t0], &[gs])?;
        sop(b, format!("s_add_i32 s{t0}, s{t0}, -1"), &[t0], &[t0])?;
        sop(b, format!("s_and_b32 s{t0}, s{}, s{t0}", Regs::WGX), &[t0], &[Regs::WGX, t0])?;
        sop(b, format!("s_lshl_b32 s{}, s{}, s{gs}", Regs::BY, Regs::WGY), &[Regs::BY], &[Regs::WGY, gs])?;
        sop(b, format!("s_add_i32 s{0}, s{0}, s{t0}", Regs::BY), &[Regs::BY], &[Regs::BY, t0])?;
        sop(b, format!("s_lshr_b32 s{}, s{}, s{gs}", Regs::BX, Regs::WGX), &[Regs::BX], &[Regs::WGX, gs])?;
    }
    let (by, bx) = (g.by(), g.bx());
    if let Some(u) = a.u {
        // t0 = row0/2: the tile's h rows (and its gate/up row window).
        sop(b, format!("s_lshl_b32 s{t0}, s{by}, 7"), &[t0], &[by])?;
        // Staging wave w owns tile fragment w: (w odd ? U : G) rows t0 + 16*(w/2).
        sop(b, format!("s_lshr_b32 s{t1}, s{}, 1", Regs::WAVE), &[t1], &[Regs::WAVE])?;
        sop(b, format!("s_lshl_b32 s{t1}, s{t1}, 4"), &[t1], &[t1])?;
        sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
        sop(b, format!("s_bitcmp1_b32 s{}, 0", Regs::WAVE), &[], &[Regs::WAVE])?;
        op(b, format!("s_cselect_b64 s[{}:{}], s[{u}:{}], s[{}:{}]", Regs::T64, Regs::T64 + 1, u + 1, a.a, a.a + 1), &[sr(Regs::T64, 2)], &[sr(u, 2), sr(a.a, 2)])?;
        mul_add64(b, Regs::SS, Regs::T64, t1, Regs::ROWB)?;
        // Scale words: gate (word 0) and up (word 1) rows t0 + 32*wr + lane part.
        sop(b, format!("s_lshl_b32 s{t1}, s{}, 5", Regs::WR), &[t1], &[Regs::WR])?;
        sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
        mul_add64(b, Regs::SW0, a.a, t1, Regs::ROWB)?;
        mul_add64(b, Regs::SW1, u, t1, Regs::ROWB)?;
    } else {
        // t0 = row0. Staging wave w owns rows t0 + 16w.
        sop(b, format!("s_lshl_b32 s{t0}, s{by}, 8"), &[t0], &[by])?;
        sop(b, format!("s_lshl_b32 s{t1}, s{}, 4", Regs::WAVE), &[t1], &[Regs::WAVE])?;
        sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
        mul_add64(b, Regs::SS, a.a, t1, Regs::ROWB)?;
        // Scale words 0/1: rows t0 + 64*wr (+16 for word 1) + lane part.
        sop(b, format!("s_lshl_b32 s{t1}, s{}, 6", Regs::WR), &[t1], &[Regs::WR])?;
        sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
        mul_add64(b, Regs::SW0, a.a, t1, Regs::ROWB)?;
        sop(b, format!("s_lshl_b32 s{t1}, s{}, 4", Regs::ROWB), &[t1], &[Regs::ROWB])?;
        sop(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SW1, Regs::SW0), &[Regs::SW1], &[Regs::SW0, t1])?;
        sop(b, format!("s_addc_u32 s{}, s{}, 0", Regs::SW1 + 1, Regs::SW0 + 1), &[Regs::SW1 + 1], &[Regs::SW0 + 1])?;
    }
    // X base: Xq + (256*bx)*72.
    sop(b, format!("s_mul_i32 s{t1}, s{bx}, {}", lit(TILE * XBLK_BYTES)), &[t1], &[bx])?;
    sop(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SX, a.xq), &[Regs::SX], &[a.xq, t1])?;
    sop(b, format!("s_addc_u32 s{}, s{}, 0", Regs::SX + 1, a.xq + 1), &[Regs::SX + 1], &[a.xq + 1])?;
    sop(b, format!("s_mul_i32 s{}, s{}, {}", Regs::N72, a.n, lit(XBLK_BYTES)), &[Regs::N72], &[a.n])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 6", Regs::M64, a.m), &[Regs::M64], &[a.m])?;
    // Y base: Y + 4*(256*bx*M + t0), 64-bit (t0 = row0, or row0/2 for h).
    let (lo, hi) = (Regs::MUL, Regs::MUL + 1);
    sop(b, format!("s_lshl_b32 s{t1}, s{bx}, 8"), &[t1], &[bx])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{t1}, s{}", a.m), &[hi], &[t1, a.m])?;
    sop(b, format!("s_mul_i32 s{lo}, s{t1}, s{}", a.m), &[lo], &[t1, a.m])?;
    sop(b, format!("s_add_u32 s{lo}, s{lo}, s{t0}"), &[lo], &[lo, t0])?;
    sop(b, format!("s_addc_u32 s{hi}, s{hi}, 0"), &[hi], &[hi])?;
    op(b, format!("s_lshl_b64 s[{lo}:{hi}], s[{lo}:{hi}], 2"), &[sr(lo, 2)], &[sr(lo, 2)])?;
    sop(b, format!("s_add_u32 s{}, s{}, s{lo}", Regs::SY, a.y), &[Regs::SY], &[a.y, lo])?;
    sop(b, format!("s_addc_u32 s{}, s{}, s{hi}", Regs::SY + 1, a.y + 1), &[Regs::SY + 1], &[a.y + 1, hi])?;
    if g.early {
        // Token fragment c's Y base = SY + c*16*M*4 (M64 bytes per fragment).
        let mut prev = Regs::SY;
        for r in Regs::SYC {
            sop(b, format!("s_add_u32 s{r}, s{prev}, s{}", Regs::M64), &[r], &[prev, Regs::M64])?;
            sop(b, format!("s_addc_u32 s{}, s{}, 0", r + 1, prev + 1), &[r + 1], &[prev + 1])?;
            prev = r;
        }
    }

    // Staging: lane (hi, lr) of wave w loads row/token 16w + lr's K16 slice
    // pair i at +8 (header / d,s skip) + 8*hi + 16*i.
    op(b, "v_lshl_add_u32 v6, v4, 3, 8", &[v(6)], &[v(4)])?;
    op(b, format!("v_mad_u32_u24 v{}, v3, s{}, v6", Regs::A_OFF, Regs::ROWB), &[v(Regs::A_OFF)], &[v(3), s(Regs::ROWB), v(6)])?;
    op(b, format!("v_lshl_add_u32 v5, s{}, 4, v3", Regs::WAVE), &[v(5)], &[s(Regs::WAVE), v(3)])?;
    op(b, format!("v_mad_u32_u24 v{}, v5, {}, v6", Regs::X_OFF, lit(XBLK_BYTES)), &[v(Regs::X_OFF)], &[v(5), v(6)])?;
    // LDS store address: wave*1024 + hi*128 + lr*8 (fragment block w, slice 2i+hi).
    op(b, "v_lshlrev_b32_e32 v7, 3, v3", &[v(7)], &[v(3)])?;
    op(b, "v_lshl_add_u32 v16, v4, 7, v7", &[v(16)], &[v(4), v(7)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 10, v16", Regs::ST, Regs::WAVE), &[v(Regs::ST)], &[s(Regs::WAVE), v(16)])?;
    // A fragment base: wr*4096 + prow*8 (+2048 for fragments 2, 3; +32768
    // for slot 1), prow = 8*(lr&1) + (lr>>1): A lane i supplies row
    // 8*(i&1)+(i>>1), so result VGPR j of lane (hi, lr) is row 8*hi + j.
    op(b, "v_and_b32_e32 v8, 1, v3", &[v(8)], &[v(3)])?;
    op(b, "v_lshrrev_b32_e32 v9, 1, v3", &[v(9)], &[v(3)])?;
    op(b, "v_lshlrev_b32_e32 v9, 3, v9", &[v(9)], &[v(9)])?;
    op(b, "v_lshl_add_u32 v8, v8, 6, v9", &[v(8)], &[v(8), v(9)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 12, v8", Regs::AB[0][0], Regs::WR), &[v(Regs::AB[0][0])], &[s(Regs::WR), v(8)])?;
    for (dst, add) in [(Regs::AB[0][1], 2048), (Regs::AB[1][0], SLOT_BYTES), (Regs::AB[1][1], SLOT_BYTES + 2048)] {
        op(b, format!("v_add_nc_u32_e32 v{dst}, {}, v{}", lit(add), Regs::AB[0][0]), &[v(dst)], &[v(Regs::AB[0][0])])?;
    }
    // X fragment bases: A_BYTES + wc*4096 + lr*8 (+2048 for c = 2, 3).
    op(b, format!("v_lshl_add_u32 v10, s{}, 12, v7", Regs::WC), &[v(10)], &[s(Regs::WC), v(7)])?;
    for (dst, add) in [(Regs::XB[0][0], A_BYTES), (Regs::XB[0][1], A_BYTES + 2048), (Regs::XB[1][0], SLOT_BYTES + A_BYTES), (Regs::XB[1][1], SLOT_BYTES + A_BYTES + 2048)] {
        op(b, format!("v_add_nc_u32_e32 v{dst}, {}, v10", lit(add)), &[v(dst)], &[v(10)])?;
    }
    // Scale row of lane (hi, lr), relative to the word bases:
    // (K1*(lr>>3) + 8*hi + (lr&7))*row_bytes, K1 = 32 (SET/ADD) or 16 (h).
    op(b, "v_lshrrev_b32_e32 v12, 3, v3", &[v(12)], &[v(3)])?;
    op(b, format!("v_lshlrev_b32_e32 v12, {}, v12", if g.silu() { 4 } else { 5 }), &[v(12)], &[v(12)])?;
    op(b, "v_lshl_add_u32 v12, v4, 3, v12", &[v(12)], &[v(4), v(12)])?;
    op(b, "v_and_b32_e32 v13, 7, v3", &[v(13)], &[v(3)])?;
    op(b, "v_add_nc_u32_e32 v12, v12, v13", &[v(12)], &[v(12), v(13)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, s{}, v12", Regs::LWS, Regs::ROWB), &[v(Regs::LWS)], &[s(Regs::ROWB), v(12)])?;
    // d of token wc*64 + lr (+16c): (wc*64 + lr)*72.
    op(b, format!("v_lshl_add_u32 v14, s{}, 6, v3", Regs::WC), &[v(14)], &[s(Regs::WC), v(3)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, {}, v14", Regs::LXS, lit(XBLK_BYTES)), &[v(Regs::LXS)], &[v(14)])?;
    // Y offset of (token wc*64 + lr, row wr*RW + 8*hi), bytes; RW = 64, or
    // 32 h rows per wave for the SiLU entry.
    op(b, "v_lshlrev_b32_e32 v15, 3, v4", &[v(15)], &[v(4)])?;
    op(b, format!("v_lshl_add_u32 v15, s{}, {}, v15", Regs::WR, if g.silu() { 5 } else { 6 }), &[v(15)], &[s(Regs::WR), v(15)])?;
    op(b, format!("v_mad_u32_u24 v15, v14, s{}, v15", a.m), &[v(15)], &[v(14), s(a.m), v(15)])?;
    op(b, format!("v_lshlrev_b32_e32 v{}, 2, v15", Regs::YOFF), &[v(Regs::YOFF)], &[v(15)])?;

    // Stage epoch 0 into slot 0, and seed the chain constant and the sums
    // while the loads are in flight.
    stage_loads(b, 0)?;
    for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", Regs::MAGIC + j, lit(MAGIC)), &[v(Regs::MAGIC + j)], &[])?; }
    for r in 0..128u8 { op(b, format!("v_mov_b32_e32 v{}, 0", Regs::ACC + r), &[v(Regs::ACC + r)], &[])?; }
    let ((ra, pa), (rx, px)) = stage_store(wg, ra, rx)?;
    let (da, dx) = wg.wait_all((pa, px))?;
    let (ra, rx) = wg.barrier((prime(ra, da), prime(rx, dx)))?;
    step_loads(wg, &ra, &rx, 0)?;
    if g.sched == Sched::Early { head(wg.isa(), 0, true, false)?; }
    Ok((ra, rx))
}

fn global_load_b64(b: &mut Builder, dst: u8, voff: u8, base: u8, offset: u32) -> Result<(), String> {
    mem(b, format!("global_load_b64 {}, v{voff}, s[{base}:{}]{}", vr(dst, 2), base + 1, off(offset)?), &[vr(dst, 2)], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}
fn global_load_b32(b: &mut Builder, dst: u8, voff: u8, base: u8, offset: u32) -> Result<(), String> {
    mem(b, format!("global_load_b32 v{dst}, v{voff}, s[{base}:{}]{}", base + 1, off(offset)?), &[v(dst)], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}

/// Global loads of the next epoch's staging packet. `a_imm` selects the K
/// half inside the current A group (64 for an odd epoch).
fn stage_loads(b: &mut Builder, a_imm: u32) -> Result<(), String> {
    b.clause(|b| { for i in 0..4u8 { global_load_b64(b, Regs::STA + 2 * i, Regs::A_OFF, Regs::SS, a_imm + 16 * u32::from(i))?; } Ok(()) })?;
    b.clause(|b| { for i in 0..4u8 { global_load_b64(b, Regs::STX + 2 * i, Regs::X_OFF, Regs::SX, 16 * u32::from(i))?; } Ok(()) })
}

/// Rebias A and publish the staged packet into the rings' next slot: store
/// `i` fills one 256-byte block (slices 2i and 2i+1 of fragment block
/// `wave`). One address register serves both slots through the 16-bit DS offset.
fn stage_store<C: State>(w: &mut Wv, ra: Ring<A, C, Free>, rx: Ring<X, C, Free>) -> Result<(Staged<A, C>, Staged<X, C>), String> {
    let slot = ra.next_index();
    for r in 0..8u8 {
        let x = Regs::STA + r;
        op(w.isa(), format!("v_xor_b32_e32 v{x}, {}, v{x}", lit(REBIAS)), &[v(x)], &[v(x)])?;
    }
    let store = |regs: u8, region: u32, i: u8| {
        let d = regs + 2 * i;
        let o = slot as u32 * SLOT_BYTES + region + 256 * u32::from(i);
        let text = format!("ds_store_b64 v{}, {}{}", Regs::ST, vr(d, 2), if o == 0 { String::new() } else { format!(" offset:{o}") });
        Instruction::new(text, vec![], vec![v(Regs::ST), vr(d, 2)]).memory(MemoryClass::DsStore)
    };
    let mut a = w.begin_write(ra);
    for i in 0..4u8 { a = w.ds_store(a, store(Regs::STA, 0, i))?; }
    let mut x = w.begin_write(rx);
    for i in 0..4u8 { x = w.ds_store(x, store(Regs::STX, A_BYTES, i))?; }
    Ok((a, x))
}

/// Fragment loads of step `i = 8a + s` from the rings' current slot: the X
/// fragments of slice s (c = 0,1 and c = 2,3) and, at even s, the A pair (s, s+1).
fn step_loads<N: State>(w: &mut Wv, ra: &Ring<A, Published, N>, rx: &Ring<X, Published, N>, i: usize) -> Result<(), String> {
    let slot = ra.cur_index();
    let (a, s_) = ((i / 8) as u32, (i % 8) as u32);
    if s_ % 2 == 0 {
        let dst = Regs::AV[(i / 2) % 2];
        let base = Regs::AB[slot][(a / 2) as usize];
        let o = (a % 2) * 128 + 16 * s_;
        w.ds_load_cur(ra, Instruction::new(format!("ds_load_2addr_b64 {}, v{base}{}", vr(dst, 4), ds_offsets(o, o + 16)), vec![vr(dst, 4)], vec![v(base)]).memory(MemoryClass::DsLoad))?;
    }
    for half in 0..2usize {
        let dst = Regs::XV[i % 2] + 4 * half as u8;
        let base = Regs::XB[slot][half];
        w.ds_load_cur(rx, Instruction::new(format!("ds_load_2addr_b64 {}, v{base}{}", vr(dst, 4), ds_offsets(16 * s_, 16 * s_ + 128)), vec![vr(dst, 4)], vec![v(base)]).memory(MemoryClass::DsLoad))?;
    }
    Ok(())
}

fn step_wmma<T: MmaIu4>(w: &mut Wave<T, Builder>, i: usize) -> Result<(), String> {
    let b = w.isa();
    let a = V::<2>(Regs::AV[(i / 2) % 2] + 2 * (i % 2) as u8);
    for c in 0..4u8 {
        let dst = V::<8>(Regs::C + 8 * c);
        let seed = if i % 8 == 0 { V::<8>(Regs::MAGIC) } else { dst };
        b.push(Wmma::iu4(b.spec.arch, dst, a, V::<2>(Regs::XV[i % 2] + 2 * c), Some(seed)))?;
    }
    Ok(())
}

/// Scale broadcast of fold pass `a` on the LDS crossbar, off the VALU port:
/// row 16a + 8hi + j's scale is word `a&1` of lane 8*(a>>1) + j of this
/// lane's row of 16, the lane `ds_swizzle_b32 swizzle(BROADCAST,16,k)` reads
/// (DPP `row_share:k`). Issued ahead of the fold so the swizzles retire
/// under WMMA issue (see `epoch`).
fn scales(b: &mut Builder, a: u8) -> Result<(), String> {
    let w = usize::from(a & 1);
    if a < 2 {
        // True16 VOP1 reaches only v0-v127 halves; v221/v222 need the VOP3 form.
        op(b, format!("v_cvt_f32_f16_e64 v{}, v{}.l", Regs::WSF[w], Regs::WS[w]), &[v(Regs::WSF[w])], &[v(Regs::WS[w])])?;
    }
    for j in 0..8u8 {
        let text = format!("ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{})", Regs::SCF + j, Regs::WSF[w], 8 * (a >> 1) + j);
        b.ds_crosslane(Instruction::new(text, vec![v(Regs::SCF + j)], vec![v(Regs::WSF[w])]).memory(MemoryClass::DsLoad))?;
    }
    Ok(())
}

/// Fold pass `a`: sum[a][c][j] = fma(d_c * sc_j, C[c][j] - 1.5*2^23, sum),
/// with the pass's row scales already broadcast into SCF by `scales(a)`.
fn fold(b: &mut Builder, a: u8, dc: u8) -> Result<(), String> {
    for c in 0..4u8 { iu4_fold::fold_pass(b, Regs::C + 8 * c, Regs::ACC + 8 * (4 * a + c), Regs::SCF, dc + c, Regs::T)?; }
    Ok(())
}

/// Epoch `e`'s head: its metadata (scale words, token d) and, with `stage`,
/// the global loads of epoch e+1's staging packet; with `touch` (ADD), two
/// lines of the residual tile per lane ahead of the packet, whose wait
/// retires them.
fn head(b: &mut Builder, p: usize, stage: bool, touch: bool) -> Result<(), String> {
    // Metadata first: in-order VMcnt returns it before the packet.
    for (w, base) in [Regs::SW0, Regs::SW1].into_iter().enumerate() {
        mem(b, format!("global_load_u16 v{}, v{}, s[{base}:{}]{}", Regs::WS[w], Regs::LWS, base + 1, off(4 * p as u32)?),
            &[v(Regs::WS[w])], &[v(Regs::LWS), sr(base, 2)], MemoryClass::VmemLoad)?;
    }
    for c in 0..4u8 { global_load_b32(b, Regs::DC[p] + c, Regs::LXS, Regs::SX, 16 * XBLK_BYTES * u32::from(c))?; }
    if touch {
        // Lines (token wc*64 + lr + 16c, row line 4wr + l) of the tile's
        // residual, c = trip, l = 2p + k: 16 lanes x 16 loads cover the
        // wave's 256 lines over the four touch trips.
        for (k, tv) in Regs::TV.into_iter().enumerate() {
            global_load_b32(b, tv, Regs::YOFF, Regs::STB, 128 * p as u32 + 64 * k as u32)?;
        }
    }
    if stage {
        if p == 1 {
            // Epoch e+1 starts the next 136-byte A group.
            for base in [Regs::SS, Regs::SW0, Regs::SW1] { add64_imm(b, base, GROUP_BYTES)?; }
            if touch { add64(b, Regs::STB, Regs::STB, Regs::M64)?; }
        }
        add64(b, Regs::SX, Regs::SX, Regs::N72)?;
        stage_loads(b, if p == 0 { 64 } else { 0 })?;
    }
    Ok(())
}

/// An epoch's rings: still reading only, or with the next packet published
/// into the next slot (its stores pending) while the current one is read.
enum Epoch { Reading(Rings), Publishing(Staged<A, Published>, Staged<X, Published>) }
fn loads(w: &mut Wv, e: &Epoch, i: usize) -> Result<(), String> {
    match e {
        Epoch::Reading((ra, rx)) => step_loads(w, ra, rx, i),
        Epoch::Publishing((ra, _), (rx, _)) => step_loads(w, ra, rx, i),
    }
}
fn publish(w: &mut Wv, e: Epoch) -> Result<Epoch, String> {
    match e {
        Epoch::Reading((ra, rx)) => { let (a, x) = stage_store(w, ra, rx)?; Ok(Epoch::Publishing(a, x)) }
        Epoch::Publishing(..) => Err("epoch publishes its packet twice".into()),
    }
}

/// One K128 epoch reading the rings' current slot `p` (epoch parity p).
/// With `next`, the staged packet is published into slot 1-p after pass 2
/// (its LDS stores drain under pass 3's WMMAs) and the barrier rotates the
/// rings; the last epoch has neither. `succ_stages`: epoch e+1 has a
/// successor too (the early schedule issues e+1's head here). Pass 0's
/// scale broadcast issues after WMMA step `scales0_step`, pass a+1's right
/// after fold pass a released the broadcast registers.
fn epoch(wg: &mut Wg, g: &Gen, rings: Rings, next: bool, succ_stages: bool, touch: bool) -> Result<Rings, String> {
    let sched = g.sched;
    let p = rings.0.cur_index();
    if sched != Sched::Early { head(wg.isa(), p, next, touch)?; }
    // Early stores (SET): in the last epoch, pass a's final sums are stored
    // one b128 per WMMA step of pass a+1, spreading the tile's Y writes over
    // the epoch instead of bursting them after it.
    let mut pending: Vec<(u8, u8, u8)> = Vec::new();
    let mut e = Epoch::Reading(rings);
    for i in 0..32 {
        if i + 1 < 32 { loads(wg, &e, i + 1)?; }
        step_wmma(wg, i)?;
        let b = wg.isa();
        if let Some((a, c, q)) = pending.pop() { early_store(b, a, c, q)?; }
        if i == g.scales0_step() { scales(b, 0)?; }
        if i % 8 == 7 && i < 31 {
            fold(b, (i / 8) as u8, Regs::DC[p])?;
            scales(b, (i / 8) as u8 + 1)?;
            if next && i / 8 == PUBLISH_AFTER_PASS { e = publish(wg, e)?; }
            if g.early && !next {
                let a = (i / 8) as u8;
                for c in (0..4u8).rev() { for q in (0..2u8).rev() { pending.push((a, c, q)); } }
            }
        }
    }
    if !pending.is_empty() { return Err("early stores left pending".into()) }
    let rings = match (e, next) {
        (Epoch::Publishing((ra, pa), (rx, px)), true) => {
            let (da, dx) = wg.wait_all((pa, px))?;
            let (ra, rx) = wg.barrier((rotate(ra, da), rotate(rx, dx)))?;
            step_loads(wg, &ra, &rx, 0)?;
            if sched == Sched::Early { head(wg.isa(), 1 - p, succ_stages, false)?; }
            (ra, rx)
        }
        (Epoch::Reading(rings), false) => rings,
        _ => return Err("epoch publication does not match its successor".into()),
    };
    fold(wg.isa(), 3, Regs::DC[p])?;
    Ok(rings)
}

/// The trip counter is `K`-derived or a constant: workgroup-uniform.
fn trips_cmp(wg: &mut Wg, cmp: &str) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(format!("{cmp} s{}, 0", Regs::TRIPS), vec![], vec![s(Regs::TRIPS)]))
}

/// A counted loop of two-epoch trips over `Regs::TRIPS`, skipped at zero
/// trips when `may_be_empty`.
fn phase(wg: &mut Wg, g: &Gen, head: &str, touch: bool, may_be_empty: bool, rings: Rings) -> Result<Rings, String> {
    let (head, end) = (g.label(head), g.label(&format!("{head}_end")));
    // Whatever the schedule leaves in flight at a trip boundary (the next
    // epoch's step-0 loads, and with `Early` its head) is in flight on entry,
    // on every back-edge (the builder's loop fixpoint checks it) and on exit.
    let entry = wg.isa().ledger.shape();
    let trips = |wg: &mut Wg, rings: Rings| -> Result<Rings, String> {
        let rings = wg.loop_carried(&head, rings, |wg, rings| {
            let rings = epoch(wg, g, rings, true, true, touch)?;
            let rings = epoch(wg, g, rings, true, true, touch)?;
            op(wg.isa(), format!("s_add_i32 s{0}, s{0}, -1", Regs::TRIPS), &[s(Regs::TRIPS)], &[s(Regs::TRIPS)])?;
            Ok((rings, trips_cmp(wg, "s_cmp_lg_u32")?))
        })?;
        // Both paths to the end label (no trip, or after the loop) arrive with
        // the same pending loads, so the waits after it hold on either.
        if wg.isa().ledger.shape() != entry { return Err(format!("{head} exit ledger differs from its entry")) }
        Ok(rings)
    };
    if may_be_empty {
        let no_trip = trips_cmp(wg, "s_cmp_eq_u32")?;
        return wg.wg_skip_if(no_trip, &end, rings, trips)
    }
    let rings = trips(wg, rings)?;
    wg.label(&end)?;
    Ok(rings)
}

fn kloop(wg: &mut Wg, g: &Gen, rings: Rings) -> Result<Rings, String> {
    wg.label(&g.label("k_begin"))?;
    let mut rings = phase(wg, g, "k_loop", false, true, rings)?;
    if g.add() {
        // Touch trips (epochs E-16..E-9), then the three trips before the tail.
        op(wg.isa(), format!("s_mov_b32 s{}, 4", Regs::TRIPS), &[s(Regs::TRIPS)], &[])?;
        op(wg.isa(), format!("s_mov_b64 s[{}:{}], s[{}:{}]", Regs::STB, Regs::STB + 1, Regs::SY, Regs::SY + 1), &[sr(Regs::STB, 2)], &[sr(Regs::SY, 2)])?;
        rings = phase(wg, g, "touch_loop", true, false, rings)?;
        op(wg.isa(), format!("s_mov_b32 s{}, 3", Regs::TRIPS), &[s(Regs::TRIPS)], &[])?;
        rings = phase(wg, g, "late_loop", false, false, rings)?;
    }
    wg.label(&g.label("tail"))?;
    let rings = epoch(wg, g, rings, true, false, false)?;
    epoch(wg, g, rings, false, false, false)
}

/// One b128 of pass `a`, token fragment `c`, row quad `q` (SET): Y bases are
/// per-fragment SGPR pairs so the store needs no VGPR beyond YOFF.
fn early_store(b: &mut Builder, a: u8, c: u8, q: u8) -> Result<(), String> {
    let base = if c == 0 { Regs::SY } else { Regs::SYC[usize::from(c - 1)] };
    let data = vr(acc(a, c) + 4 * q, 4);
    mem(b, format!("global_store_b128 v{}, {data}, s[{base}:{}]{}", Regs::YOFF, base + 1, off(4 * (16 * u32::from(a) + 4 * u32::from(q)))?),
        &[], &[v(Regs::YOFF), data, sr(base, 2)], MemoryClass::VmemStore)
}

fn acc(a: u8, c: u8) -> u8 { Regs::ACC + 8 * (4 * a + c) }

/// Lane (hi, lr) owns rows 8hi..8hi+7 of each fragment for one token; token
/// fragment c is 16*M*4 bytes further (v0..v3 = its Y offsets; the SET's
/// early stores use per-fragment SGPR bases instead).
fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("epilogue"))?;
    if !g.early {
        op(b, format!("v_mov_b32_e32 v0, v{}", Regs::YOFF), &[v(0)], &[v(Regs::YOFF)])?;
        for c in 1..4u8 {
            op(b, format!("v_add_nc_u32_e32 v{c}, s{}, v{}", Regs::M64, c - 1), &[v(c)], &[s(Regs::M64), v(c - 1)])?;
        }
    }
    b.label(&g.label("epi_body"))?;
    let store = |b: &mut Builder, c: u8, data: u8, offset: u32| -> Result<(), String> {
        let data = vr(data, 4);
        mem(b, format!("global_store_b128 v{c}, {data}, s[{}:{}]{}", Regs::SY, Regs::SY + 1, off(offset)?), &[], &[v(c), data, sr(Regs::SY, 2)], MemoryClass::VmemStore)
    };
    match g.spec.epi {
        Epi::Set if g.early => {
            for c in 0..4u8 { for q in 0..2u8 { early_store(b, 3, c, q)?; } }
        }
        Epi::Set => {
            for c in 0..4u8 { for a in 0..4u8 { for q in 0..2u8 {
                store(b, c, acc(a, c) + 4 * q, 4 * (16 * u32::from(a) + 4 * u32::from(q)))?;
            } } }
        }
        Epi::Add => {
            // Y = RN(Y + sum): each row fragment's residual lands in one of
            // three rings, three fragments ahead; sums are added in place and
            // stored from the accumulators, so no ring waits on a store.
            let ring = [160u8, 192, 224];
            let load = |b: &mut Builder, a: u8| -> Result<(), String> {
                let r = ring[usize::from(a % 3)];
                for c in 0..4u8 { for q in 0..2u8 {
                    let dst = vr(r + 8 * c + 4 * q, 4);
                    mem(b, format!("global_load_b128 {dst}, v{c}, s[{}:{}]{}", Regs::SY, Regs::SY + 1, off(4 * (16 * u32::from(a) + 4 * u32::from(q)))?),
                        &[dst], &[v(c), sr(Regs::SY, 2)], MemoryClass::VmemLoad)?;
                } }
                Ok(())
            };
            for a in 0..3u8 { load(b, a)?; }
            for a in 0..4u8 {
                let r = ring[usize::from(a % 3)];
                for c in 0..4u8 {
                    for k in (0..8u8).step_by(2) {
                        let (y0, s0) = (acc(a, c) + k, r + 8 * c + k);
                        b.vopd(VopdOp { op: VopdF32::Add, dst: y0, src0: Operand::V(s0), src1: y0 },
                            VopdOp { op: VopdF32::Add, dst: y0 + 1, src0: Operand::V(s0 + 1), src1: y0 + 1 })?;
                    }
                }
                if a == 0 { load(b, 3)?; }
                for c in 0..4u8 { for q in 0..2u8 {
                    store(b, c, acc(a, c) + 4 * q, 4 * (16 * u32::from(a) + 4 * u32::from(q)))?;
                } }
            }
        }
        Epi::GateUpSilu => {
            // h = SILU_MUL(g, u) for gate fragment 2p and up fragment 2p+1,
            // written over the gate sums and stored from there.
            for p in 0..2u8 { for c in 0..4u8 {
                silu_group(b, acc(2 * p, c), acc(2 * p + 1, c))?;
                for q in 0..2u8 {
                    store(b, c, acc(2 * p, c) + 4 * q, 4 * (16 * u32::from(p) + 4 * u32::from(q)))?;
                }
            } }
        }
    }
    Ok(())
}

/// `g[k] = g[k] / (1 + expf(-g[k])) * u[k]` for k = 0..7, the exact op DAG
/// hipcc emits for `IU4_V2B_SILU_MUL` (LLVM's f32 `exp` lowering with range
/// checks and the IEEE `fdiv` expansion with f32 denormals enabled). The
/// eight elements are interleaved op by op.
fn silu_group(b: &mut Builder, gate: u8, up: u8) -> Result<(), String> {
    silu_mul(b, gate, up, 160, Regs::MASK, SILU_GROUP)
}

/// [`silu_group`]'s DAG over the `n` elements `gate..gate+n` / `up..up+n`
/// (h written over the gate values), interleaved op by op: the builder's one
/// SiLU emitter for gfx11 and gfx12. Element `k` uses the temporaries
/// `tmp + 6k ..+6` and the lane-mask pairs `mask + 2k` (underflow),
/// `mask + 2(n+k)` (overflow), `mask + 2(2n+k)` (numerator scale).
pub(crate) fn silu_mul(b: &mut Builder, gate: u8, up: u8, tmp: u8, mask: u8, n: u8) -> Result<(), String> {
    let t = |k: u8, i: u8| tmp + SILU_TEMPS * k + i;
    let (m_under, m_over, m_num) = (|k: u8| mask + 2 * k, |k: u8| mask + 2 * (n + k), |k: u8| mask + 2 * (2 * n + k));
    type Step<'a> = &'a dyn Fn(&mut Builder, u8) -> Result<(), String>;
    let steps: [Step; 26] = [
        // ph = RN(-log2e * g); pl = fma(-log2e, g, -ph); pl = fma(-log2e_lo, g, pl)
        &|b, k| op(b, format!("v_mul_f32_e32 v{}, 0xbfb8aa3b, v{}", t(k, 0), gate + k), &[v(t(k, 0))], &[v(gate + k)]),
        &|b, k| op(b, format!("v_fma_f32 v{}, 0xbfb8aa3b, v{}, -v{}", t(k, 1), gate + k, t(k, 0)), &[v(t(k, 1))], &[v(gate + k), v(t(k, 0))]),
        &|b, k| op(b, format!("v_fmac_f32_e32 v{}, 0xb2a5705f, v{}", t(k, 1), gate + k), &[v(t(k, 1))], &[v(t(k, 1)), v(gate + k)]),
        // e = rndne(ph); a = (ph - e) + pl; r = ldexp(exp2(a), int(e))
        &|b, k| op(b, format!("v_rndne_f32_e32 v{}, v{}", t(k, 2), t(k, 0)), &[v(t(k, 2))], &[v(t(k, 0))]),
        &|b, k| op(b, format!("v_sub_f32_e32 v{0}, v{0}, v{1}", t(k, 0), t(k, 2)), &[v(t(k, 0))], &[v(t(k, 0)), v(t(k, 2))]),
        &|b, k| op(b, format!("v_add_f32_e32 v{0}, v{0}, v{1}", t(k, 0), t(k, 1)), &[v(t(k, 0))], &[v(t(k, 0)), v(t(k, 1))]),
        &|b, k| op(b, format!("v_exp_f32_e32 v{0}, v{0}", t(k, 0)), &[v(t(k, 0))], &[v(t(k, 0))]),
        &|b, k| op(b, format!("v_cvt_i32_f32_e32 v{0}, v{0}", t(k, 2)), &[v(t(k, 2))], &[v(t(k, 2))]),
        &|b, k| op(b, format!("v_ldexp_f32 v{0}, v{0}, v{1}", t(k, 0), t(k, 2)), &[v(t(k, 0))], &[v(t(k, 0)), v(t(k, 2))]),
        // r = -g < -103.28 ? 0 : r; r = -g > 88.72 ? +inf : r
        &|b, k| op(b, format!("v_cmp_nlt_f32_e64 s{}, 0x42ce8ed0, v{}", m_under(k), gate + k), &[s(m_under(k))], &[v(gate + k)]),
        &|b, k| op(b, format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{1}", t(k, 0), m_under(k)), &[v(t(k, 0))], &[v(t(k, 0)), s(m_under(k))]),
        &|b, k| op(b, format!("v_cmp_ngt_f32_e64 s{}, 0xc2b17218, v{}", m_over(k), gate + k), &[s(m_over(k))], &[v(gate + k)]),
        &|b, k| op(b, format!("v_cndmask_b32_e64 v{0}, 0x7f800000, v{0}, s{1}", t(k, 0), m_over(k)), &[v(t(k, 0))], &[v(t(k, 0)), s(m_over(k))]),
        // d = 1 + r; q = g / d (div_scale, rcp, three fma refinements, fmas, fixup)
        &|b, k| op(b, format!("v_add_f32_e32 v{0}, 1.0, v{0}", t(k, 0)), &[v(t(k, 0))], &[v(t(k, 0))]),
        &|b, k| op(b, format!("v_div_scale_f32 v{0}, null, v{1}, v{1}, v{2}", t(k, 1), t(k, 0), gate + k), &[v(t(k, 1))], &[v(t(k, 0)), v(gate + k)]),
        &|b, k| op(b, format!("v_div_scale_f32 v{0}, s{1}, v{2}, v{3}, v{2}", t(k, 2), m_num(k), gate + k, t(k, 0)), &[v(t(k, 2)), s(m_num(k))], &[v(t(k, 0)), v(gate + k)]),
        &|b, k| op(b, format!("v_rcp_f32_e32 v{}, v{}", t(k, 3), t(k, 1)), &[v(t(k, 3))], &[v(t(k, 1))]),
        &|b, k| op(b, format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(k, 4), t(k, 1), t(k, 3)), &[v(t(k, 4))], &[v(t(k, 1)), v(t(k, 3))]),
        &|b, k| op(b, format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(k, 3), t(k, 4)), &[v(t(k, 3))], &[v(t(k, 3)), v(t(k, 4))]),
        &|b, k| op(b, format!("v_mul_f32_e32 v{}, v{}, v{}", t(k, 4), t(k, 2), t(k, 3)), &[v(t(k, 4))], &[v(t(k, 2)), v(t(k, 3))]),
        &|b, k| op(b, format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(k, 5), t(k, 1), t(k, 4), t(k, 2)), &[v(t(k, 5))], &[v(t(k, 1)), v(t(k, 4)), v(t(k, 2))]),
        &|b, k| op(b, format!("v_fmac_f32_e32 v{}, v{}, v{}", t(k, 4), t(k, 5), t(k, 3)), &[v(t(k, 4))], &[v(t(k, 4)), v(t(k, 5)), v(t(k, 3))]),
        &|b, k| op(b, format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(k, 5), t(k, 1), t(k, 4), t(k, 2)), &[v(t(k, 5))], &[v(t(k, 1)), v(t(k, 4)), v(t(k, 2))]),
        &|b, k| {
            op(b, format!("s_mov_b32 vcc_lo, s{}", m_num(k)), &[], &[s(m_num(k))])?;
            op(b, format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(k, 5), t(k, 3), t(k, 4)), &[v(t(k, 5))], &[v(t(k, 5)), v(t(k, 3)), v(t(k, 4))])
        },
        &|b, k| op(b, format!("v_div_fixup_f32 v{0}, v{0}, v{1}, v{2}", t(k, 5), t(k, 0), gate + k), &[v(t(k, 5))], &[v(t(k, 5)), v(t(k, 0)), v(gate + k)]),
        // h = q * u
        &|b, k| op(b, format!("v_mul_f32_e32 v{}, v{}, v{}", gate + k, t(k, 5), up + k), &[v(gate + k)], &[v(t(k, 5)), v(up + k)]),
    ];
    for step in steps {
        for k in 0..n { step(b, k)?; }
    }
    Ok(())
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let g = Gen::new(spec);
    let kspec = KernelSpec {
        kernel_id: "iu4_v2b".into(), variant: spec.epi.tag().into(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: THREADS, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    let mut wg = Wg::new(&mut b)?;
    let rings = declare_lds(&mut wg)?;
    let end = wg.exit(&g.label("end"))?;
    let rings = prologue(&mut wg, &g, rings)?;
    // The last epoch's slot stays published: nothing writes LDS after it.
    let _published = kloop(&mut wg, &g, rings)?;
    epilogue(wg.isa(), &g)?;
    // One CTA per WGP: release the VGPRs before the epilogue's stores drain
    // so the next CTA's waves launch (hipcc emits the same message).
    // M7 models `s_sendmsg` as reading M0; the message carries no data.
    wg.end_with(end, |w| {
        op(w.isa(), "s_mov_b32 m0, 0", &[], &[])?;
        w.isa().push(Sop::Dealloc.encode(spec.arch)?)
    })?;
    b.finish()
}

/// The three entries as one code object (SET, ADD, SiLU).
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Epi::ALL.into_iter().map(|epi| emit(Spec { arch, epi })).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

/// Instruction census of one steady-state trip (two epochs) of the first
/// K loop of `epi`.
pub fn hot_loop_census(s_text: &str, epi: Epi) -> std::collections::BTreeMap<String, u32> {
    let (head, end) = (format!(".Lv2b_{}_k_loop:", epi.tag()), format!(".Lv2b_{}_k_loop_end:", epi.tag()));
    let mut counts = std::collections::BTreeMap::new();
    let mut inside = false;
    for line in s_text.lines() {
        let t = line.trim();
        if t == head { inside = true; continue }
        if t == end { break }
        if !inside || t.ends_with(':') || t.is_empty() || t.starts_with('.') { continue }
        let name = t.split_whitespace().next().unwrap_or("").to_owned();
        *counts.entry(name.clone()).or_insert(0) += 1;
        if name.starts_with("v_dual_") { *counts.entry("vopd_packets".into()).or_insert(0) += 1 }
        if name.starts_with("v_") && !name.starts_with("v_wmma_") { *counts.entry("valu_slots".into()).or_insert(0) += 1 }
    }
    counts
}
