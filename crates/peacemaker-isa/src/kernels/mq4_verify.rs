// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 MQ4G256V2 wide exact-verify GEMMs (64 <= N <= 128 batch rows),
//! emitted with the checked ISA builder as byte-identical twins of the hipcc
//! `_vt{4,8}w{4,8}_k32` instantiations of
//! `kernels/src/gemm_mq4g256v2_wmma_gfx12_vt_core.hip` (wrappers
//! `gemm_{qkvza,qkv,gate_up}_mq4g256v2_wmma_gfx12_vt.hip` and
//! `gemm_mq4g256v2_residual_wmma_gfx12_vt.hip`), and so of the one-tile
//! singleton WMMA kernels (PLAN-CHUNK64 §3/§6).
//!
//! Exactness (byte-identical outputs): every output is one f32 chain from
//! +0 of `v_wmma_f32_16x16x16_f16` over ascending 16-K tiles with the
//! singleton's operands, so no output depends on the tiling:
//!  * A (weights, `[rows, K/256, 136]`): the header dword of a 32-K slab is
//!    word `(slab >> 2) & 1` of its 256-K group (16-K tile kt < 8 -> h0); the
//!    lane's 8 codes of 16-K tile `i` are the dword at group + 8 + 8i + 4*kg.
//!    Each nibble is converted exactly to f32 (`v_cvt_f32_ubyte{0..3}` of the
//!    nibble-spread dword; every value 0..15), then exactly to f16
//!    (`v_cvt_f16_f32`), then `v_fma_f16(sc, q, zp)` (one rounding, f16
//!    denormals kept, round to nearest even) — the singleton's
//!    `v_cvt_f32_ubyte0 -> v_cvt_f16_f32 -> v_fma_f16` chain (GATE0 §3 ISA
//!    check). Rows past `total_m` load row `total_m - 1` and are never stored.
//!  * B (activations, fp16 `[N, K]`): the same bytes, through LDS; tokens past
//!    N read 0 (a bounded buffer descriptor) and are never stored.
//!  * C map: `acc[j]` of token tile `b` is output row `rs + 8*(lane>>4) + j`,
//!    token `bs + 16*b + (lane&15)`. Overwrite ops store `acc`; the residual
//!    stores `Y + acc` once (one `v_add_f32`, commutative) after the chain.
//!
//! Freedom used (tiling, layout and scheduling only):
//!  * a wave owns 16 weight rows x `16*BT` tokens (BT accumulator tiles), so
//!    one A dequant feeds BT WMMAs; a workgroup is W waves;
//!  * X is staged per 32-K slab in LDS (80-byte token pitch: 32 K + 8 pad),
//!    double-buffered, one barrier per slab; X is loaded two slabs ahead
//!    into two register stages;
//!  * B fragments rotate through an 8-deep register ring loaded eight WMMAs
//!    ahead; A codes are loaded two slabs ahead; the next tile's dequant is
//!    interleaved with the current WMMAs;
//!  * every lane keeps 64-bit weight row pointers (the projection matrix of
//!    its own row), so a wave may straddle projection boundaries.
//!
//! Launch (the HIP twins' ABI and grid): block `32*W`, grid
//! `[ceil(total_m/(16*W)), ceil(N/(16*BT)), 1]`, no dynamic LDS, static LDS
//! `2 * 16*BT * 80` bytes. Kernargs, pointers 8 bytes and ints 4 bytes in the
//! HIP order: QKVZA 96, QKV 76, gate/up 56, residual/head 36 bytes.
//! Domain: K > 0 and K % 256 == 0 (K < 256 stores `0 + acc`/`Y + 0` without
//! loading), `N * K * 2 < 2^31`, output offsets `< 2^32` bytes per matrix.
use crate::insn::{Instruction, MemoryClass, Sop};
use crate::kernels::common::{lit, op, s, smem, sop, sr, v, vr, SRD_WORD3};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::reg::{Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The four HIP wrapper families (ABI and epilogue).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyOp {
    Qkvza,
    Qkv,
    GateUp,
    Residual,
}
impl VerifyOp {
    pub const ALL: [VerifyOp; 4] = [
        VerifyOp::Qkvza,
        VerifyOp::Qkv,
        VerifyOp::GateUp,
        VerifyOp::Residual,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Qkvza => "qkvza",
            Self::Qkv => "qkv",
            Self::GateUp => "gate_up",
            Self::Residual => "residual",
        }
    }
    /// Projection matrices (weight and output pairs).
    pub fn matrices(self) -> u8 {
        match self {
            Self::Qkvza => 4,
            Self::Qkv => 3,
            Self::GateUp => 2,
            Self::Residual => 1,
        }
    }
    /// Residual (`Y += acc`) instead of overwrite (`Y = acc`).
    pub fn accumulates(self) -> bool {
        self == Self::Residual
    }
    /// Kernarg segment bytes: n weight pointers, X, n output pointers, n row counts, K, N.
    pub fn kernarg_bytes(self) -> u32 {
        20 * u32::from(self.matrices()) + 16
    }
    /// HIP parameter names: weights, outputs, row counts.
    fn names(
        self,
    ) -> (
        &'static [&'static str],
        &'static [&'static str],
        &'static [&'static str],
    ) {
        match self {
            Self::Qkvza => (
                &["A_qkv", "A_z", "A_beta", "A_alpha"],
                &["Y_qkv", "Y_z", "Y_beta", "Y_alpha"],
                &["qkv_m", "z_m", "beta_m", "alpha_m"],
            ),
            Self::Qkv => (
                &["A_q", "A_k", "A_v"],
                &["Y_q", "Y_k", "Y_v"],
                &["q_m", "k_m", "v_m"],
            ),
            Self::GateUp => (
                &["A_gate", "A_up"],
                &["Y_gate", "Y_up"],
                &["gate_m", "up_m"],
            ),
            Self::Residual => (&["A"], &["Y"], &["M"]),
        }
    }
}
impl std::str::FromStr for VerifyOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|o| o.name() == s)
            .ok_or_else(|| format!("mq4_verify op {s} (qkvza|qkv|gate_up|residual)"))
    }
}

/// One entry: `bt` 16-token tiles per block (4: N = 64, 8: 65..128), `waves`
/// waves (16 weight rows each) per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub bt: u8,
    pub waves: u8,
    pub op: VerifyOp,
}
impl Spec {
    pub const BT: [u8; 2] = [4, 8];
    pub const WAVES: [u8; 2] = [4, 8];
    /// Every entry of the module, in its frozen order (op, then BT, then W).
    pub fn all(arch: Arch) -> Vec<Spec> {
        let mut out = Vec::with_capacity(16);
        for op in VerifyOp::ALL {
            for bt in Self::BT {
                for waves in Self::WAVES {
                    out.push(Spec {
                        arch,
                        bt,
                        waves,
                        op,
                    });
                }
            }
        }
        out
    }
    pub fn symbol(&self) -> String {
        format!(
            "mq4_verify_{}_pm_gfx1201_bt{}w{}",
            self.op.name(),
            self.bt,
            self.waves
        )
    }
    /// Contract file tag: `<op>_bt<bt>w<w>`.
    pub fn tag(&self) -> String {
        format!("{}_bt{}w{}", self.op.name(), self.bt, self.waves)
    }
    pub fn module(arch: Arch) -> String {
        format!("mq4_verify_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1201 {
            return Err("mq4_verify builder is exact-gfx1201 only".into());
        }
        if !Self::BT.contains(&self.bt) {
            return Err(format!("mq4_verify: bt {} not in {:?}", self.bt, Self::BT));
        }
        if !Self::WAVES.contains(&self.waves) {
            return Err(format!(
                "mq4_verify: waves {} not in {:?}",
                self.waves,
                Self::WAVES
            ));
        }
        Ok(())
    }
    fn threads(&self) -> u32 {
        32 * u32::from(self.waves)
    }
    /// Batch (token) rows staged per block.
    fn tokens(&self) -> u32 {
        16 * u32::from(self.bt)
    }
    /// Weight rows per block.
    fn rows(&self) -> u32 {
        16 * u32::from(self.waves)
    }
    /// One LDS X slot: `16*BT` token rows x 80 bytes.
    pub fn slot_bytes(&self) -> u32 {
        self.tokens() * PITCH
    }
    /// Static LDS: two slots.
    pub fn lds_bytes(&self) -> u32 {
        2 * self.slot_bytes()
    }
    /// X uint4 per thread per slab (`16*BT` rows x four 16-byte chunks).
    fn xpt(&self) -> u8 {
        (self.tokens() * 4 / self.threads()) as u8
    }
    /// Token rows between a thread's X uint4s.
    fn xrow_step(&self) -> u32 {
        self.threads() / 4
    }
    /// WMMAs per slab: two 16-K tiles x BT token tiles.
    fn wpc(&self) -> u8 {
        2 * self.bt
    }
    /// The WMMA after which the slab barrier sits (see `slab`).
    fn bar_at(&self) -> u8 {
        self.wpc() - 1 - RING / 2
    }
}

const PITCH: u32 = 80; // 32 K f16 + 8 pad per token row
const GROUP: u32 = 136; // bytes per 256-K weight group: h0, h1, 128 code bytes
const SLABS: u8 = 8; // 32-K slabs per 256-K group
const RING: u8 = 8; // B fragment ring depth
const NIBBLES: u32 = 0x0f0f_0f0f;

// ------------------------------------------------------------------ registers
// VGPRs: v0 tid; v1..v10 lane constants / temporaries; v11.. X offsets;
// weight row pointers; then accumulators, B ring, A fragments, codes,
// headers, dequant temporaries and the two X register stages (`Lay`).
const TID: u8 = 0;
const LDST: u8 = 1;
const LDRD: u8 = 2;
const VTOK: u8 = 3;
const VROW0: u8 = 4;
const T0: u8 = 5;
const T1: u8 = 6;
const T2: u8 = 7;
const T3: u8 = 8;
const T4: u8 = 9;
const T5: u8 = 10;
const XOFF: u8 = 11; // up to 4
/// 64-bit per-lane pointers: row base (header), row base + 8 + 4*kg (codes),
/// this group's codes, the next group's codes and header.
const BASEH: u8 = 15;
const BASEC: u8 = 17;
const PC: u8 = 19;
const PCN: u8 = 21;
const PHN: u8 = 23;
const ACC: u8 = 25;

/// Spec-dependent VGPR layout.
struct Lay {
    xpt: u8,
    ring: u8,
    af: [u8; 2],
    code: [u8; 4],
    hdr: [u8; 2],
    tlo: u8,
    thi: u8,
    q: u8,
    xr: [u8; 2],
    end: u16,
}
impl Lay {
    fn new(spec: &Spec) -> Lay {
        let xpt = spec.xpt();
        let ring = ACC + 8 * spec.bt;
        let af0 = ring + 4 * RING;
        let code0 = af0 + 8;
        let hdr0 = code0 + 8;
        let tlo = hdr0 + 4;
        let q = tlo + 2;
        let xr0 = q + 8;
        Lay {
            xpt,
            ring,
            af: [af0, af0 + 4],
            code: [code0, code0 + 2, code0 + 4, code0 + 6],
            hdr: [hdr0, hdr0 + 2],
            tlo,
            thi: tlo + 1,
            q,
            xr: [xr0, xr0 + 4 * xpt],
            end: u16::from(xr0) + 8 * u16::from(xpt),
        }
    }
    /// Epilogue tile pointers (pairs): the dead A/code/header/temp registers.
    fn eptr(&self, t: u8) -> u8 {
        self.af[0] + 2 * t
    }
}

// SGPRs.
const KARG: u8 = 0;
const SRD_X: u8 = 4;
const WGX: u8 = 8;
const WGY: u8 = 9;
const TOTM: u8 = 10;
const GPR: u8 = 11;
const STRIDE: u8 = 12;
const SK2: u8 = 13;
const SM1: u8 = 14;
const BS: u8 = 15;
/// Kernarg segment image (up to 24 dwords).
const KA: u8 = 16;
const G: u8 = 40;
const GO: u8 = 41;
const NGO: u8 = 42;
const C8: u8 = 43;
const SXOFF: u8 = 44;
const ST0: u8 = 45;
const GPRM1: u8 = 47;
/// Cumulative projection row bounds m0, m0+m1, m0+m1+m2.
const E: [u8; 3] = [48, 49, 50];
const TOTM1: u8 = 51;
/// Lane masks sit at even SGPRs with their odd neighbour unused: VOPC `_e64`
/// destinations are encoded (and analysed) as SGPR pairs.
const MASK0: u8 = 52;
const MASK1: u8 = 54;
const MASKR: u8 = 56;
const SGPRS: u16 = 58;

fn n_of(spec: &Spec) -> u8 {
    spec.op.matrices()
}
/// Kernarg SGPRs: weight pointer i, X, output pointer i, row count i, K, N.
fn ka_a(i: u8) -> u8 {
    KA + 2 * i
}
fn ka_x(spec: &Spec) -> u8 {
    KA + 2 * n_of(spec)
}
fn ka_y(spec: &Spec, i: u8) -> u8 {
    KA + 2 * (n_of(spec) + 1 + i)
}
fn ka_m(spec: &Spec, i: u8) -> u8 {
    KA + 4 * n_of(spec) + 2 + i
}
fn ka_k(spec: &Spec) -> u8 {
    KA + 5 * n_of(spec) + 2
}
fn ka_n(spec: &Spec) -> u8 {
    KA + 5 * n_of(spec) + 3
}
/// Exact SMEM loads covering the kernarg segment: (byte offset, dwords).
fn kernarg_loads(spec: &Spec) -> Vec<(u32, u8)> {
    let mut out = Vec::new();
    let (mut off, size) = (0u32, spec.op.kernarg_bytes());
    for d in [8u8, 4, 2, 1] {
        while size - off >= 4 * u32::from(d) {
            out.push((off, d));
            off += 4 * u32::from(d);
        }
    }
    out
}

fn plan(spec: &Spec, l: &Lay) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(l.end, SGPRS)?;
    let w = || Live::Whole;
    p.v::<1>("tid", TID, w())?;
    for n in LDST..=T5 {
        p.v::<1>("lane_constant", n, w())?;
    }
    for n in 0..l.xpt {
        p.v::<1>("x_offset", XOFF + n, w())?;
    }
    for ptr in [BASEH, BASEC, PC, PCN, PHN] {
        p.v::<2>("row_pointer", ptr, w())?;
    }
    for t in 0..spec.bt {
        p.v::<8>("acc", ACC + 8 * t, w())?;
    }
    for r in 0..RING {
        p.v::<4>("b_ring", l.ring + 4 * r, w())?;
    }
    for a in l.af {
        p.v::<4>("a_frag", a, w())?;
    }
    for c in l.code {
        p.v::<2>("codes", c, w())?;
    }
    for h in l.hdr {
        p.v::<2>("header", h, w())?;
    }
    for n in l.tlo..l.q + 8 {
        p.v::<1>("dequant_temp", n, w())?;
    }
    for st in l.xr {
        for i in 0..l.xpt {
            p.v::<4>("x_stage", st + 4 * i, w())?;
        }
    }
    p.s::<2>("kernarg_ptr", KARG, w())?;
    p.s::<4>("srd_x", SRD_X, w())?;
    for n in WGX..=BS {
        p.s::<1>("scalar", n, w())?;
    }
    for (off, d) in kernarg_loads(spec) {
        let base = KA + (off / 4) as u8;
        match d {
            8 => {
                p.s::<8>("kernargs", base, w())?;
            }
            4 => {
                p.s::<4>("kernargs", base, w())?;
            }
            2 => {
                p.s::<2>("kernargs", base, w())?;
            }
            _ => {
                p.s::<1>("kernargs", base, w())?;
            }
        }
    }
    for n in G..=TOTM1 {
        p.s::<1>("scalar", n, w())?;
    }
    for m in [MASK0, MASK1, MASKR] {
        p.s::<2>("lane_mask", m, w())?;
    }
    Ok(p)
}

fn wait_alu(b: &mut Builder, what: &str) -> R {
    op(b, format!("s_wait_alu {what}"), &[], &[])
}
fn rs(base: u8, len: u8) -> String {
    if len == 1 {
        format!("v{base}")
    } else {
        format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1)
    }
}
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}

/// `global_load_b{32,64,128} dst, v[ptr:ptr+1], off offset:imm`.
fn gload(b: &mut Builder, dwords: u8, dst: u8, ptr: u8, imm: u32) -> R {
    let w = match dwords {
        1 => 32,
        2 => 64,
        4 => 128,
        _ => return Err("gload width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "global_load_b{w} {}, {}, off{}",
                rs(dst, dwords),
                rs(ptr, 2),
                off(imm)
            ),
            vec![vr(dst, dwords)],
            vec![vr(ptr, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
/// `global_store_b{32,128} v[ptr:ptr+1], data, off offset:imm`.
fn gstore(b: &mut Builder, dwords: u8, ptr: u8, data: u8, imm: u32) -> R {
    let w = match dwords {
        1 => 32,
        4 => 128,
        _ => return Err("gstore width".into()),
    };
    b.push(
        Instruction::new(
            format!(
                "global_store_b{w} {}, {}, off{}",
                rs(ptr, 2),
                rs(data, dwords),
                off(imm)
            ),
            vec![],
            vec![vr(ptr, 2), vr(data, dwords)],
        )
        .memory(MemoryClass::VmemStore),
    )
}
/// X uint4: `buffer_load_b128 dst, v{voff}, s[SRD_X], s{SXOFF} offen`.
fn xload(b: &mut Builder, dst: u8, voff: u8) -> R {
    b.push(
        Instruction::new(
            format!(
                "buffer_load_b128 {}, v{voff}, s[{SRD_X}:{}], s{SXOFF} offen",
                rs(dst, 4),
                SRD_X + 3
            ),
            vec![vr(dst, 4)],
            vec![v(voff), sr(SRD_X, 4), s(SXOFF)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
/// `v[dst:dst+1] = src0 * src1 + v[base:base+1]` (64-bit, no carry out).
fn mad64(b: &mut Builder, dst: u8, src0: RegRef, src1: &str, base: u8) -> R {
    op(
        b,
        format!(
            "v_mad_co_u64_u32 {}, null, {src0}, {src1}, {}",
            rs(dst, 2),
            rs(base, 2)
        ),
        &[vr(dst, 2)],
        &[src0, vr(base, 2)],
    )
}
fn wmma(b: &mut Builder, t: u8, a: u8, bb: u8) -> R {
    let acc = ACC + 8 * t;
    op(
        b,
        format!(
            "v_wmma_f32_16x16x16_f16 {}, {}, {}, {}",
            rs(acc, 8),
            rs(a, 4),
            rs(bb, 4),
            rs(acc, 8)
        ),
        &[vr(acc, 8)],
        &[vr(a, 4), vr(bb, 4), vr(acc, 8)],
    )
}

/// One scheduled operation interleaved between WMMAs.
#[derive(Clone)]
enum Fill {
    /// B fragment of slab WMMA `k` (16-K tile k/BT, token tile k%BT) from `slot`.
    BLoad { slot: usize, k: u8 },
    /// Store X stage uint4 at `src` into `slot` at slot-relative `imm`.
    XStore { slot: usize, src: u8, imm: u32 },
    /// One code or header load: `dwords` at `ptr + imm` into `dst`.
    WLoad {
        dwords: u8,
        dst: u8,
        ptr: u8,
        imm: u32,
    },
    Valu {
        text: String,
        defs: Vec<RegRef>,
        uses: Vec<RegRef>,
    },
}
fn emit_fill(b: &mut Builder, spec: &Spec, l: &Lay, f: &Fill) -> R {
    match f {
        Fill::BLoad { slot, k } => {
            let (j, t) = (u32::from(k / spec.bt), u32::from(k % spec.bt));
            let dst = ring(l, *k);
            let imm = *slot as u32 * spec.slot_bytes() + t * 16 * PITCH + j * 32;
            b.ds_load(
                *slot,
                Instruction::new(
                    format!("ds_load_b128 {}, v{LDRD}{}", rs(dst, 4), off(imm)),
                    vec![vr(dst, 4)],
                    vec![v(LDRD)],
                )
                .memory(MemoryClass::DsLoad),
            )
        }
        Fill::XStore { slot, src, imm } => b.ds_store(
            *slot,
            Instruction::new(
                format!(
                    "ds_store_b128 v{LDST}, {}{}",
                    rs(*src, 4),
                    off(*slot as u32 * spec.slot_bytes() + imm)
                ),
                vec![],
                vec![v(LDST), vr(*src, 4)],
            )
            .memory(MemoryClass::DsStore),
        ),
        Fill::WLoad {
            dwords,
            dst,
            ptr,
            imm,
        } => gload(b, *dwords, *dst, *ptr, *imm),
        Fill::Valu { text, defs, uses } => op(b, text.clone(), defs, uses),
    }
}
fn ring(l: &Lay, k: u8) -> u8 {
    l.ring + 4 * (k % RING)
}
/// Code buffer of slab `c` (mod 4: a group's 8 slabs keep the rotation).
fn code(l: &Lay, c: u8) -> u8 {
    l.code[usize::from(c % 4)]
}
/// Header dword of slab `c` of the current group: h0 for slabs 0..3, h1 for 4..7.
fn hdr_dword(l: &Lay, c: u8) -> u8 {
    l.hdr[0] + u8::from(c >= SLABS / 2)
}
/// The two code loads of slab `c` of the group at `ptr` (16-K tiles 2c, 2c+1).
fn code_loads(l: &Lay, c_dst: u8, ptr: u8, c: u8) -> [Fill; 2] {
    let dst = code(l, c_dst);
    [
        Fill::WLoad {
            dwords: 1,
            dst,
            ptr,
            imm: 16 * u32::from(c),
        },
        Fill::WLoad {
            dwords: 1,
            dst: dst + 1,
            ptr,
            imm: 16 * u32::from(c) + 8,
        },
    ]
}
fn xstores(spec: &Spec, l: &Lay, slot: usize, stage: usize) -> Vec<Fill> {
    (0..l.xpt)
        .map(|i| Fill::XStore {
            slot,
            src: l.xr[stage] + 4 * i,
            imm: u32::from(i) * spec.xrow_step() * PITCH,
        })
        .collect()
}

/// Dequantize one 16-K tile: codes dword `w` (this lane's 8 nibbles, low
/// first), header dword `h` (sc in .l, zp in .h) into A fragment `a`
/// (half i = nibble i). Breadth-first: eight independent per-value chains.
fn dequant(l: &Lay, w: u8, h: u8, a: u8) -> Vec<Fill> {
    let mut out = Vec::new();
    let mut f = |text: String, defs: Vec<RegRef>, uses: Vec<RegRef>| {
        out.push(Fill::Valu { text, defs, uses })
    };
    let (lo, hi, q) = (l.tlo, l.thi, l.q);
    // lo bytes = even nibbles, hi bytes = odd nibbles (nibble 2k+e at byte k).
    f(
        format!("v_and_b32_e32 v{lo}, {}, v{w}", lit(NIBBLES)),
        vec![v(lo)],
        vec![v(w)],
    );
    f(
        format!("v_lshrrev_b32_e32 v{hi}, 4, v{w}"),
        vec![v(hi)],
        vec![v(w)],
    );
    f(
        format!("v_and_b32_e32 v{hi}, {}, v{hi}", lit(NIBBLES)),
        vec![v(hi)],
        vec![v(hi)],
    );
    for i in 0..8u8 {
        let src = if i % 2 == 0 { lo } else { hi };
        f(
            format!("v_cvt_f32_ubyte{}_e32 v{}, v{src}", i / 2, q + i),
            vec![v(q + i)],
            vec![v(src)],
        );
    }
    for i in 0..8u8 {
        f(
            format!("v_cvt_f16_f32_e64 v{0}.l, v{0}", q + i),
            vec![v(q + i)],
            vec![v(q + i)],
        );
    }
    for i in 0..8u8 {
        let d = a + i / 2;
        let (half, sel) = if i % 2 == 0 {
            ("l", "op_sel:[0,0,1,0]")
        } else {
            ("h", "op_sel:[0,0,1,1]")
        };
        f(
            format!("v_fma_f16 v{d}.{half}, v{h}.l, v{}.l, v{h}.h {sel}", q + i),
            vec![v(d)],
            vec![v(d), v(h), v(q + i)],
        );
    }
    out
}

/// Spread `fills` evenly over the WMMA slots `from..to`.
fn spread(slots: &mut [Vec<Fill>], fills: Vec<Fill>, from: usize, to: usize) {
    let n = fills.len().max(1);
    let span = to.max(from + 1) - from;
    for (k, f) in fills.into_iter().enumerate() {
        slots[from + k * span / n].push(f);
    }
}

/// One 32-K slab `c` (0..7) of the group loop body: 2*BT WMMAs. On entry
/// slot `c%2` holds X(s), the ring holds (or is loading) this slab's first
/// RING fragments, stage `(c+1)%2` holds X(s+1) and stage `c%2` is loading
/// X(s+2).
fn slab(b: &mut Builder, spec: &Spec, l: &Lay, c: u8) -> R {
    let (wpc, bar, bt) = (spec.wpc(), spec.bar_at(), spec.bt);
    let slot = usize::from(c % 2);
    let other = 1 - slot;
    let stage = usize::from((c + 1) % 2);
    // X(8g + c + 3) (clamped to the last slab): loaded after this slab's barrier.
    sop(
        b,
        format!("s_add_co_i32 s{SXOFF}, s{C8}, {}", c + 3),
        &[SXOFF],
        &[C8],
    )?;
    sop(
        b,
        format!("s_min_u32 s{SXOFF}, s{SXOFF}, s{SM1}"),
        &[SXOFF],
        &[SXOFF, SM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXOFF}, s{SXOFF}, 6"),
        &[SXOFF],
        &[SXOFF],
    )?;
    let mut slots: Vec<Vec<Fill>> = vec![Vec::new(); usize::from(wpc)];
    // Codes two slabs ahead (+ the next group's header two slabs before its
    // first use: the last slab's next-tile dequant).
    let (ptr, pc) = if c + 2 < SLABS {
        (PC, c + 2)
    } else {
        (PCN, c + 2 - SLABS)
    };
    slots[0].extend(code_loads(l, c + 2, ptr, pc));
    if c + 2 == SLABS {
        slots[0].push(Fill::WLoad {
            dwords: 2,
            dst: l.hdr[1],
            ptr: PHN,
            imm: 0,
        });
    }
    // B ring: WMMA i loads WMMA i+RING's fragment into the register it just read.
    for i in 0..wpc {
        if i + RING < wpc && i < bar {
            slots[usize::from(i)].push(Fill::BLoad { slot, k: i + RING });
        }
    }
    // Stage X(s+1) into the other slot before the barrier.
    spread(
        &mut slots,
        xstores(spec, l, other, stage),
        1,
        usize::from(bar),
    );
    // A(tile 1) under tile 0's WMMAs; A(tile 0) of the next slab under tile 1's.
    spread(
        &mut slots,
        dequant(l, code(l, c) + 1, hdr_dword(l, c), l.af[1]),
        0,
        usize::from(bt) - 1,
    );
    let nh = if c + 1 == SLABS {
        l.hdr[1]
    } else {
        hdr_dword(l, c + 1)
    };
    spread(
        &mut slots,
        dequant(l, code(l, c + 1), nh, l.af[0]),
        usize::from(bt),
        usize::from(wpc) - 1,
    );
    for i in 0..wpc {
        let (j, t) = (i / bt, i % bt);
        wmma(b, t, l.af[usize::from(j)], ring(l, i))?;
        for f in &slots[usize::from(i)] {
            emit_fill(b, spec, l, f)?;
        }
        if i == bar {
            // This wave's reads of `slot` have landed (its last ring loads
            // were issued before here); publish the other slot, retire this one.
            b.wait(Counter::Ds, 0)?;
            b.barrier(&[Transition::Ready(other), Transition::Retire(slot)])?;
            for x in 0..l.xpt {
                xload(b, l.xr[stage] + 4 * x, XOFF + x)?;
            }
            // The next slab's first fragments, from the published slot.
            for k in 0..(wpc - 1 - bar) {
                emit_fill(b, spec, l, &Fill::BLoad { slot: other, k })?;
            }
        } else if i > bar {
            emit_fill(
                b,
                spec,
                l,
                &Fill::BLoad {
                    slot: other,
                    k: i + RING - wpc,
                },
            )?;
        }
    }
    Ok(())
}

/// Per-lane projection select for row `row` (VGPR): `dst` (pair) = pointer
/// `ptrs[i]`, `off` = first row of projection i, `m` (optional) = its row
/// count, for the last i with `E[i-1] <= row`.
fn select(
    b: &mut Builder,
    spec: &Spec,
    row: u8,
    ptrs: &[u8],
    dst: u8,
    offv: u8,
    m: Option<u8>,
) -> R {
    op(
        b,
        format!("v_mov_b32_e32 v{dst}, s{}", ptrs[0]),
        &[v(dst)],
        &[s(ptrs[0])],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, s{}", dst + 1, ptrs[0] + 1),
        &[v(dst + 1)],
        &[s(ptrs[0] + 1)],
    )?;
    op(b, format!("v_mov_b32_e32 v{offv}, 0"), &[v(offv)], &[])?;
    if let Some(m) = m {
        op(
            b,
            format!("v_mov_b32_e32 v{m}, s{}", ka_m(spec, 0)),
            &[v(m)],
            &[s(ka_m(spec, 0))],
        )?;
    }
    for k in 1..n_of(spec) {
        let e = E[usize::from(k - 1)];
        op(
            b,
            format!("v_cmp_le_i32_e64 s{MASK1}, s{e}, v{row}"),
            &[s(MASK1)],
            &[s(e), v(row)],
        )?;
        wait_alu(b, "depctr_va_sdst(0)")?;
        let p = ptrs[usize::from(k)];
        for (d, src) in [(dst, p), (dst + 1, p + 1), (offv, e)] {
            op(
                b,
                format!("v_cndmask_b32_e64 v{d}, v{d}, s{src}, s{MASK1}"),
                &[v(d)],
                &[v(d), s(src), s(MASK1)],
            )?;
        }
        if let Some(m) = m {
            let mk = ka_m(spec, k);
            op(
                b,
                format!("v_cndmask_b32_e64 v{m}, v{m}, s{mk}, s{MASK1}"),
                &[v(m)],
                &[v(m), s(mk), s(MASK1)],
            )?;
        }
    }
    Ok(())
}

fn prologue(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    let n = n_of(spec);
    for (o, d) in kernarg_loads(spec) {
        smem(b, KA + (o / 4) as u8, d, KARG, o)?;
    }
    // Lane constants that need no kernargs.
    op(
        b,
        format!("v_and_b32_e32 v{T0}, 15, v{TID}"),
        &[v(T0)],
        &[v(TID)],
    )?; // ml
    op(
        b,
        format!("v_bfe_u32 v{T1}, v{TID}, 4, 1"),
        &[v(T1)],
        &[v(TID)],
    )?; // kg
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T2}, 4, v{T1}"),
        &[v(T2)],
        &[v(T1)],
    )?; // 16*kg
    op(
        b,
        format!("v_mad_u32_u24 v{LDRD}, v{T0}, {}, v{T2}", lit(PITCH)),
        &[v(LDRD)],
        &[v(T0), v(T2)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T3}, 2, v{TID}"),
        &[v(T3)],
        &[v(TID)],
    )?; // tid/4
    op(
        b,
        format!("v_and_b32_e32 v{T4}, 3, v{TID}"),
        &[v(T4)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T4}, 4, v{T4}"),
        &[v(T4)],
        &[v(T4)],
    )?; // (tid%4)*16
    op(
        b,
        format!("v_mad_u32_u24 v{LDST}, v{T3}, {}, v{T4}", lit(PITCH)),
        &[v(LDST)],
        &[v(T3), v(T4)],
    )?;
    for r in 0..8 * spec.bt {
        op(
            b,
            format!("v_mov_b32_e32 v{}, 0", ACC + r),
            &[v(ACC + r)],
            &[],
        )?;
    }
    b.wait(Counter::Km, 0)?;
    // total_m and the cumulative projection bounds.
    sop(
        b,
        format!("s_mov_b32 s{TOTM}, s{}", ka_m(spec, 0)),
        &[TOTM],
        &[ka_m(spec, 0)],
    )?;
    for k in 1..n {
        sop(
            b,
            format!("s_mov_b32 s{}, s{TOTM}", E[usize::from(k - 1)]),
            &[E[usize::from(k - 1)]],
            &[TOTM],
        )?;
        sop(
            b,
            format!("s_add_co_i32 s{TOTM}, s{TOTM}, s{}", ka_m(spec, k)),
            &[TOTM],
            &[TOTM, ka_m(spec, k)],
        )?;
    }
    // Block-uniform early exit: no weight rows, or no tokens, for this block.
    op(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[s(WGX)], &[])?;
    op(
        b,
        format!("s_and_b32 s{WGY}, ttmp7, 0xffff"),
        &[s(WGY)],
        &[],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{ST0}, s{TOTM}"),
        &[],
        &[ST0, TOTM],
    )?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    sop(
        b,
        format!("s_mul_i32 s{BS}, s{WGY}, {}", lit(spec.tokens())),
        &[BS],
        &[WGY],
    )?;
    sop(
        b,
        format!("s_cmp_ge_i32 s{BS}, s{}", ka_n(spec)),
        &[],
        &[BS, ka_n(spec)],
    )?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "end")), &[], &[])?;
    let (sk, sn) = (ka_k(spec), ka_n(spec));
    sop(b, format!("s_lshl_b32 s{SK2}, s{sk}, 1"), &[SK2], &[sk])?;
    sop(b, format!("s_ashr_i32 s{GPR}, s{sk}, 8"), &[GPR], &[sk])?;
    sop(
        b,
        format!("s_mul_i32 s{STRIDE}, s{GPR}, {}", lit(GROUP)),
        &[STRIDE],
        &[GPR],
    )?;
    sop(b, format!("s_lshl_b32 s{SM1}, s{GPR}, 3"), &[SM1], &[GPR])?;
    sop(
        b,
        format!("s_add_co_i32 s{SM1}, s{SM1}, -1"),
        &[SM1],
        &[SM1],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{GPRM1}, s{GPR}, -1"),
        &[GPRM1],
        &[GPR],
    )?;
    sop(
        b,
        format!("s_add_co_i32 s{TOTM1}, s{TOTM}, -1"),
        &[TOTM1],
        &[TOTM],
    )?;
    // X descriptor bounded at N*K*2 bytes: tokens past N read 0.
    let x = ka_x(spec);
    sop(b, format!("s_mov_b32 s{SRD_X}, s{x}"), &[SRD_X], &[x])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD_X + 1, x + 1),
        &[SRD_X + 1],
        &[x + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{}, s{sn}, s{SK2}", SRD_X + 2),
        &[SRD_X + 2],
        &[sn, SK2],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_X + 3, lit(SRD_WORD3)),
        &[SRD_X + 3],
        &[],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{ST0}, s{WGX}, {}", lit(spec.rows())),
        &[ST0],
        &[WGX],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    // Rows: rs = (wgx*W + wave)*16; VROW0 = rs + 8*kg (the C rows); the
    // lane's A row min(rs + ml, total_m - 1).
    op(
        b,
        format!("v_lshrrev_b32_e32 v{T5}, 5, v{TID}"),
        &[v(T5)],
        &[v(TID)],
    )?; // wave
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T5}, 4, v{T5}"),
        &[v(T5)],
        &[v(T5)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T5}, s{ST0}, v{T5}"),
        &[v(T5)],
        &[s(ST0), v(T5)],
    )?; // rs
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VROW0}, 3, v{T1}"),
        &[v(VROW0)],
        &[v(T1)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{VROW0}, v{VROW0}, v{T5}"),
        &[v(VROW0)],
        &[v(VROW0), v(T5)],
    )?;
    op(
        b,
        format!("v_or_b32_e32 v{T5}, v{T5}, v{T0}"),
        &[v(T5)],
        &[v(T5), v(T0)],
    )?;
    op(
        b,
        format!("v_min_i32_e32 v{T5}, s{TOTM1}, v{T5}"),
        &[v(T5)],
        &[s(TOTM1), v(T5)],
    )?;
    // Its projection: weight pointer (into PC, free until the loop) and first row.
    let a_ptrs: Vec<u8> = (0..n).map(ka_a).collect();
    select(b, spec, T5, &a_ptrs, PC, T2, None)?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{T5}, v{T5}, v{T2}"),
        &[v(T5)],
        &[v(T5), v(T2)],
    )?;
    mad64(b, BASEH, v(T5), &format!("s{STRIDE}"), PC)?;
    op(
        b,
        format!("v_lshl_add_u32 v{T2}, v{T1}, 2, 8"),
        &[v(T2)],
        &[v(T1)],
    )?;
    mad64(b, BASEC, v(T2), "1", BASEH)?;
    // Tokens: epilogue token bs + ml; X rows bs + i*step + tid/4.
    op(
        b,
        format!("v_add_nc_u32_e32 v{VTOK}, s{BS}, v{T0}"),
        &[v(VTOK)],
        &[s(BS), v(T0)],
    )?;
    for i in 0..l.xpt {
        let x = XOFF + i;
        op(
            b,
            format!("v_add_nc_u32_e32 v{x}, s{BS}, v{T3}"),
            &[v(x)],
            &[s(BS), v(T3)],
        )?;
        if i > 0 {
            op(
                b,
                format!(
                    "v_add_nc_u32_e32 v{x}, {}, v{x}",
                    lit(u32::from(i) * spec.xrow_step())
                ),
                &[v(x)],
                &[v(x)],
            )?;
        }
        op(
            b,
            format!("v_mul_lo_u32 v{x}, v{x}, s{SK2}"),
            &[v(x)],
            &[v(x), s(SK2)],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{x}, v{x}, v{T4}"),
            &[v(x)],
            &[v(x), v(T4)],
        )?;
    }
    // K < 256: no group; the epilogue stores the zero accumulators.
    sop(b, format!("s_cmp_lt_i32 s{sk}, {}", lit(256)), &[], &[sk])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "epi")), &[], &[])?;
    // Group 0: header, slab 0's codes and X; stage X(0) in slot 0.
    sop(b, format!("s_mov_b32 s{G}, 0"), &[G], &[])?;
    sop(b, format!("s_mov_b32 s{GO}, 0"), &[GO], &[])?;
    sop(b, format!("s_mov_b32 s{C8}, 0"), &[C8], &[])?;
    sop(b, format!("s_mov_b32 s{SXOFF}, 0"), &[SXOFF], &[])?;
    gload(b, 2, l.hdr[0], BASEH, 0)?;
    for f in code_loads(l, 0, BASEC, 0) {
        emit_fill(b, spec, l, &f)?;
    }
    for i in 0..l.xpt {
        xload(b, l.xr[0] + 4 * i, XOFF + i)?;
    }
    for f in xstores(spec, l, 0, 0) {
        emit_fill(b, spec, l, &f)?;
    }
    b.barrier(&[Transition::Ready(0)])?;
    // X(1) (stage 1), slab 1's codes, X(2) (stage 0), A of slab 0 tile 0
    // and the ring's first fragments in flight into the loop, as the body's
    // back edge leaves them.
    sop(
        b,
        format!("s_min_u32 s{SXOFF}, s{SM1}, 1"),
        &[SXOFF],
        &[SM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXOFF}, s{SXOFF}, 6"),
        &[SXOFF],
        &[SXOFF],
    )?;
    for i in 0..l.xpt {
        xload(b, l.xr[1] + 4 * i, XOFF + i)?;
    }
    for f in code_loads(l, 1, BASEC, 1) {
        emit_fill(b, spec, l, &f)?;
    }
    sop(
        b,
        format!("s_min_u32 s{SXOFF}, s{SM1}, 2"),
        &[SXOFF],
        &[SM1],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SXOFF}, s{SXOFF}, 6"),
        &[SXOFF],
        &[SXOFF],
    )?;
    for i in 0..l.xpt {
        xload(b, l.xr[0] + 4 * i, XOFF + i)?;
    }
    for f in dequant(l, code(l, 0), l.hdr[0], l.af[0]) {
        emit_fill(b, spec, l, &f)?;
    }
    for k in 0..RING {
        emit_fill(b, spec, l, &Fill::BLoad { slot: 0, k })?;
    }
    Ok(())
}

fn body(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    // This group's code pointer; the next group's code/header pointers at
    // min(g+1, gpr-1) (the last group re-reads itself).
    sop(b, format!("s_add_co_i32 s{NGO}, s{G}, 1"), &[NGO], &[G])?;
    sop(
        b,
        format!("s_min_u32 s{NGO}, s{NGO}, s{GPRM1}"),
        &[NGO],
        &[NGO, GPRM1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{NGO}, s{NGO}, {}", lit(GROUP)),
        &[NGO],
        &[NGO],
    )?;
    wait_alu(b, "depctr_sa_sdst(0)")?;
    mad64(b, PC, s(GO), "1", BASEC)?;
    mad64(b, PCN, s(NGO), "1", BASEC)?;
    mad64(b, PHN, s(NGO), "1", BASEH)?;
    for c in 0..SLABS {
        slab(b, spec, l, c)?;
    }
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", l.hdr[0], l.hdr[1]),
        &[v(l.hdr[0])],
        &[v(l.hdr[1])],
    )?;
    op(
        b,
        format!("v_mov_b32_e32 v{}, v{}", l.hdr[0] + 1, l.hdr[1] + 1),
        &[v(l.hdr[0] + 1)],
        &[v(l.hdr[1] + 1)],
    )?;
    sop(b, format!("s_add_co_i32 s{G}, s{G}, 1"), &[G], &[G])?;
    sop(
        b,
        format!("s_add_co_i32 s{GO}, s{GO}, {}", lit(GROUP)),
        &[GO],
        &[GO],
    )?;
    sop(b, format!("s_add_co_i32 s{C8}, s{C8}, 8"), &[C8], &[C8])?;
    sop(b, format!("s_cmp_lt_u32 s{G}, s{GPR}"), &[], &[G, GPR])?;
    op(b, format!("s_cbranch_scc1 {}", lbl(spec, "loop")), &[], &[])
}

/// exec = (token tile `t` < N) & MASKR. Clobbers T0.
fn tile_exec(b: &mut Builder, spec: &Spec, t: u8) -> R {
    let src = if t == 0 {
        VTOK
    } else {
        op(
            b,
            format!(
                "v_add_nc_u32_e32 v{T0}, {}, v{VTOK}",
                lit(16 * u32::from(t))
            ),
            &[v(T0)],
            &[v(VTOK)],
        )?;
        T0
    };
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASK0}, s{}, v{src}", ka_n(spec)),
        &[s(MASK0)],
        &[s(ka_n(spec)), v(src)],
    )?;
    wait_alu(b, "depctr_va_sdst(0)")?;
    op(
        b,
        format!("s_and_b32 exec_lo, s{MASK0}, s{MASKR}"),
        &[],
        &[s(MASK0), s(MASKR)],
    )
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

/// For output row `row` (VGPR): tile pointers `eptr(t)` = &Y_i[(bs+16t+ml)*m_i
/// + row - off_i] and MASKR = row < total_m. Clobbers T2..T5, PC, PCN.
fn out_pointers(b: &mut Builder, spec: &Spec, l: &Lay, row: u8) -> R {
    let y_ptrs: Vec<u8> = (0..n_of(spec)).map(|i| ka_y(spec, i)).collect();
    select(b, spec, row, &y_ptrs, PC, T3, Some(T4))?;
    op(
        b,
        format!("v_mul_lo_u32 v{T2}, v{VTOK}, v{T4}"),
        &[v(T2)],
        &[v(VTOK), v(T4)],
    )?;
    op(
        b,
        format!("v_sub_nc_u32_e32 v{T5}, v{row}, v{T3}"),
        &[v(T5)],
        &[v(row), v(T3)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{T2}, v{T2}, v{T5}"),
        &[v(T2)],
        &[v(T2), v(T5)],
    )?;
    mad64(b, l.eptr(0), v(T2), "4", PC)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{T5}, 6, v{T4}"),
        &[v(T5)],
        &[v(T4)],
    )?; // 16 tokens * 4 bytes * m
    for t in 1..spec.bt {
        mad64(b, l.eptr(t), v(T5), "1", l.eptr(t - 1))?;
    }
    op(
        b,
        format!("v_cmp_gt_i32_e64 s{MASKR}, s{TOTM}, v{row}"),
        &[s(MASKR)],
        &[s(TOTM), v(row)],
    )
}

fn epilogue(b: &mut Builder, spec: &Spec, l: &Lay) -> R {
    b.wait_all()?;
    b.label(&lbl(spec, "epi"))?;
    let n = n_of(spec);
    let bt = spec.bt;
    // Every projection row count a multiple of 8: each lane's 8-row run lies
    // in one projection and is in bounds when its first row is.
    sop(
        b,
        format!("s_mov_b32 s{ST0}, s{}", ka_m(spec, 0)),
        &[ST0],
        &[ka_m(spec, 0)],
    )?;
    for k in 1..n {
        sop(
            b,
            format!("s_or_b32 s{ST0}, s{ST0}, s{}", ka_m(spec, k)),
            &[ST0],
            &[ST0, ka_m(spec, k)],
        )?;
    }
    sop(b, format!("s_and_b32 s{ST0}, s{ST0}, 7"), &[ST0], &[ST0])?;
    sop(b, format!("s_cmp_lg_u32 s{ST0}, 0"), &[], &[ST0])?;
    op(
        b,
        format!("s_cbranch_scc1 {}", lbl(spec, "ragged")),
        &[],
        &[],
    )?;
    out_pointers(b, spec, l, VROW0)?;
    if spec.op.accumulates() {
        // Y in flight four tiles ahead through the dead B ring.
        let ybuf = |t: u8| l.ring + 8 * (t % 4);
        let load = |b: &mut Builder, t: u8| -> R {
            tile_exec(b, spec, t)?;
            gload(b, 4, ybuf(t), l.eptr(t), 0)?;
            gload(b, 4, ybuf(t) + 4, l.eptr(t), 16)?;
            exec_all(b)
        };
        for t in 0..bt.min(4) {
            load(b, t)?;
        }
        for t in 0..bt {
            let acc = ACC + 8 * t;
            for e in 0..8u8 {
                op(
                    b,
                    format!("v_add_f32_e32 v{}, v{}, v{}", acc + e, ybuf(t) + e, acc + e),
                    &[v(acc + e)],
                    &[v(ybuf(t) + e), v(acc + e)],
                )?;
            }
            tile_exec(b, spec, t)?;
            gstore(b, 4, l.eptr(t), acc, 0)?;
            gstore(b, 4, l.eptr(t), acc + 4, 16)?;
            exec_all(b)?;
            if t + 4 < bt {
                load(b, t + 4)?;
            }
        }
    } else {
        for t in 0..bt {
            let acc = ACC + 8 * t;
            tile_exec(b, spec, t)?;
            gstore(b, 4, l.eptr(t), acc, 0)?;
            gstore(b, 4, l.eptr(t), acc + 4, 16)?;
            exec_all(b)?;
        }
    }
    b.release_store_sources()?;
    op(b, format!("s_branch {}", lbl(spec, "end")), &[], &[])?;
    // Ragged projections: per row j of the lane's run, its own projection.
    b.label(&lbl(spec, "ragged"))?;
    for e in 0..8u8 {
        if e > 0 {
            b.release_store_sources()?;
        }
        let row = T1;
        op(
            b,
            format!("v_add_nc_u32_e32 v{row}, {e}, v{VROW0}"),
            &[v(row)],
            &[v(VROW0)],
        )?;
        out_pointers(b, spec, l, row)?;
        if spec.op.accumulates() {
            for t in 0..bt {
                tile_exec(b, spec, t)?;
                gload(b, 1, l.ring + t, l.eptr(t), 0)?;
                exec_all(b)?;
            }
        }
        for t in 0..bt {
            let a = ACC + 8 * t + e;
            if spec.op.accumulates() {
                op(
                    b,
                    format!("v_add_f32_e32 v{a}, v{}, v{a}", l.ring + t),
                    &[v(a)],
                    &[v(l.ring + t), v(a)],
                )?;
            }
            tile_exec(b, spec, t)?;
            gstore(b, 1, l.eptr(t), a, 0)?;
            exec_all(b)?;
        }
    }
    b.release_store_sources()?;
    b.label(&lbl(spec, "end"))?;
    b.control(Sop::End.encode(Arch::Gfx1201)?)
}

/// Local labels, qualified by entry so all sixteen share one module.
fn lbl(spec: &Spec, name: &str) -> String {
    format!(".Lmq4v_{}_{name}", spec.tag())
}

fn kernargs(spec: &Spec) -> KernargLayout {
    let (a, y, m) = spec.op.names();
    let n = u32::from(spec.op.matrices());
    let mut k = KernargLayout::new(spec.op.kernarg_bytes());
    for (i, name) in a.iter().enumerate() {
        k = k.pointer(name, 8 * i as u32);
    }
    k = k.pointer("X", 8 * n);
    for (i, name) in y.iter().enumerate() {
        k = k.pointer(name, 8 * (n + 1 + i as u32));
    }
    for (i, name) in m.iter().enumerate() {
        k = k.hidden(name, 16 * n + 8 + 4 * i as u32, 4, "by_value");
    }
    k.hidden("K", 20 * n + 8, 4, "by_value")
        .hidden("N", 20 * n + 12, 4, "by_value")
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let l = Lay::new(&spec);
    let kspec = KernelSpec {
        kernel_id: "mq4_verify".into(),
        variant: spec.tag(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs: kernargs(&spec),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: spec.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&spec, &l)?);
    b.enable_delay_alu();
    for (id, (name, base)) in [("x_slot0", 0u32), ("x_slot1", spec.slot_bytes())]
        .into_iter()
        .enumerate()
    {
        if b.lds_slot(name, base, spec.slot_bytes())? != id {
            return Err("LDS slot order".into());
        }
    }
    prologue(&mut b, &spec, &l)?;
    b.loop_(&lbl(&spec, "loop"), |b| body(b, &spec, &l))?;
    epilogue(&mut b, &spec, &l)?;
    b.finish()
}

/// Every entry as one code object (`Spec::module`), in `Spec::all` order.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Spec::all(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Mnemonics a contract pins by exact count (when present).
const COUNTED: [&str; 22] = [
    "v_wmma_f32_16x16x16_f16",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_cvt_f32_ubyte0_e32",
    "v_cvt_f32_ubyte1_e32",
    "v_cvt_f32_ubyte2_e32",
    "v_cvt_f32_ubyte3_e32",
    "v_cvt_f16_f32_e64",
    "v_fma_f16",
    "v_add_f32_e32",
    "v_mad_co_u64_u32",
    "buffer_load_b128",
    "global_load_b32",
    "global_load_b64",
    "global_load_b128",
    "global_store_b32",
    "global_store_b128",
    "ds_load_b128",
    "ds_store_b128",
    "s_load_b256",
    "s_load_b128",
    "s_endpgm",
];

/// The shape contract of one entry (`toolchain::IsaShapeContract`): exact
/// counts of its arithmetic, memory and barrier instructions, the forbidden
/// families, register ceilings (PLAN-CHUNK64 §3: VGPR <= 224, SGPR <= 64),
/// wave32, no spills or private memory, the static LDS and no dynamic LDS.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let symbol = spec.symbol();
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    serde_json::json!({
        "symbol": symbol, "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "ds_load_2addr*", "ds_store_2addr*", "ds_store_b32", "ds_store_b64", "ds_load_b32", "ds_load_b64",
                      "v_readlane*", "v_writelane*", "s_setpc*", "s_swappc*", "s_waitcnt*"],
        "vgpr_max": 224, "sgpr_max": 64,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": spec.lds_bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_deterministic_and_target_only_gfx1201() {
        for spec in Spec::all(Arch::Gfx1201) {
            let (a, b) = (emit(spec).unwrap(), emit(spec).unwrap());
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
            // Two 16-K tiles x BT token tiles per 32-K slab, eight slabs per group.
            assert_eq!(a.shape.wmma, 16 * usize::from(spec.bt), "{spec:?}");
            // One prologue barrier, one per slab.
            assert_eq!(a.shape.barriers, 9, "{spec:?}");
            assert!(
                a.shape.next_free_vgpr <= 224 && a.shape.next_free_sgpr <= 64,
                "{spec:?}: {:?}",
                a.shape
            );
        }
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            assert!(emit(Spec {
                arch,
                bt: 8,
                waves: 4,
                op: VerifyOp::Residual
            })
            .is_err());
        }
        for (bt, waves) in [(2, 4), (6, 4), (8, 2), (8, 16)] {
            assert!(emit(Spec {
                arch: Arch::Gfx1201,
                bt,
                waves,
                op: VerifyOp::Qkv
            })
            .is_err());
        }
    }

    #[test]
    fn module_exports_the_sixteen_frozen_symbols_in_order() {
        let (emitted, text, _) = emit_module(Arch::Gfx1201).unwrap();
        let got: Vec<String> = emitted.iter().map(|e| e.proof.variant.clone()).collect();
        let want: Vec<String> = Spec::all(Arch::Gfx1201).iter().map(Spec::tag).collect();
        assert_eq!(got, want);
        for spec in Spec::all(Arch::Gfx1201) {
            assert!(text.contains(&format!(".amdhsa_kernel {}\n", spec.symbol())));
        }
    }

    #[test]
    fn kernarg_abi_matches_the_hip_twins() {
        assert_eq!(VerifyOp::ALL.map(VerifyOp::kernarg_bytes), [96, 76, 56, 36]);
        for op in VerifyOp::ALL {
            let spec = Spec {
                arch: Arch::Gfx1201,
                bt: 8,
                waves: 4,
                op,
            };
            let k = kernargs(&spec);
            k.validate().unwrap();
            assert_eq!(
                k.args.last().map(|a| a.offset + a.size),
                Some(op.kernarg_bytes())
            );
            let covered: u32 = kernarg_loads(&spec)
                .iter()
                .map(|(_, d)| 4 * u32::from(*d))
                .sum();
            assert_eq!(covered, op.kernarg_bytes());
        }
    }

    /// The runtime embeds `kernels/mq4_verify_pm_gfx1201.hxaco`: it must be
    /// exactly the native writer's bundle of today's emission, and every
    /// committed shape contract exactly today's.
    #[test]
    fn committed_bundle_and_contracts_equal_fresh_emission() {
        let (emitted, text, _) = emit_module(Arch::Gfx1201).unwrap();
        let elf = crate::native::assemble(&text, Arch::Gfx1201).unwrap();
        let fresh = crate::native::bundle(&elf, Arch::Gfx1201, "host-x86_64-unknown-linux-gnu-");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/mq4_verify_pm_gfx1201.hxaco"
        );
        assert!(
            std::fs::read(path).unwrap() == fresh,
            "{path}: fresh emission differs from the committed bundle"
        );
        for (spec, e) in Spec::all(Arch::Gfx1201).into_iter().zip(&emitted) {
            let path = format!(
                "{}/kernels/mq4_verify.gfx1201.{}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                spec.tag()
            );
            let committed: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(committed, contract(spec, e), "{path}");
        }
    }

    // ------------------------------------------------- interpreted reference check
    // Every entry's emitted text runs whole grids in a small in-order wave32
    // interpreter of exactly the instructions this builder emits (waits and
    // hints are no-ops: memory completes at issue; barriers rendezvous the
    // block's waves), on small problems — projection rows straddling 8/16,
    // tokens past N, a second grid column, inactive waves, two and three
    // 256-K groups — against an independent reference: per output one chain
    // from +0 of peacemaker's probed `v_wmma_f32_16x16x16_f16` numerics over
    // ascending 16-K tiles in K order, A = fma_f16(sc, nibble, zp) rounded
    // once, residual `Y + acc`. Every byte of every output buffer (including
    // rows past total_m, tokens past N and a canary tail) is compared. This
    // checks the data movement and indexing; the GPU byte oracle is the
    // authority for the hardware arithmetic.
    use peacemaker_ir::emu::{
        convert::{f16_to_f32, f32_to_f16},
        mma::wmma_f32_f16,
    };
    use std::collections::HashMap;

    #[derive(Clone, Debug)]
    enum Opd {
        V(u16, Option<bool>),
        S(u16, u16),
        Exec,
        Null,
        Ttmp(u16),
        Imm(u32),
        Label(String),
    }
    struct Ins {
        m: String,
        o: Vec<Opd>,
        off: u32,
    }
    fn opd(t: &str) -> Opd {
        let t = t.trim();
        let range = |r: &str| {
            let (a, b) = r.trim_end_matches(']').split_once(':').unwrap();
            let (a, b): (u16, u16) = (a.parse().unwrap(), b.parse().unwrap());
            (a, b - a + 1)
        };
        if t == "exec_lo" {
            return Opd::Exec;
        }
        if t == "null" || t == "off" {
            return Opd::Null;
        }
        if t.starts_with(".L") {
            return Opd::Label(t.into());
        }
        if let Some(n) = t.strip_prefix("ttmp") {
            return Opd::Ttmp(n.parse().unwrap());
        }
        if let Some(r) = t.strip_prefix("v[") {
            return Opd::V(range(r).0, None);
        }
        if let Some(r) = t.strip_prefix("s[") {
            let (a, n) = range(r);
            return Opd::S(a, n);
        }
        if let Some(r) = t.strip_prefix('v') {
            let (n, half) = if let Some(n) = r.strip_suffix(".l") {
                (n, Some(false))
            } else if let Some(n) = r.strip_suffix(".h") {
                (n, Some(true))
            } else {
                (r, None)
            };
            if let Ok(n) = n.parse() {
                return Opd::V(n, half);
            }
        }
        if let Some(r) = t.strip_prefix('s') {
            if let Ok(n) = r.parse() {
                return Opd::S(n, 1);
            }
        }
        if let Some(h) = t.strip_prefix("0x") {
            return Opd::Imm(u32::from_str_radix(h, 16).unwrap());
        }
        Opd::Imm(t.parse::<i64>().unwrap_or_else(|_| panic!("operand {t}")) as u32)
    }
    /// The kernel's instructions and labels.
    fn parse(text: &str) -> (Vec<Ins>, HashMap<String, usize>) {
        let (mut prog, mut labels) = (Vec::new(), HashMap::new());
        for line in text
            .lines()
            .skip_while(|l| !l.ends_with(':') || l.starts_with('.'))
            .skip(1)
        {
            if line.starts_with('.') && !line.ends_with(':') {
                break;
            }
            if let Some(l) = line.trim().strip_suffix(':') {
                labels.insert(l.to_string(), prog.len());
                continue;
            }
            let line = line.trim();
            let (m, rest) = line.split_once(' ').map_or((line, ""), |(m, r)| (m, r));
            if m.starts_with("s_wait") || m == "s_delay_alu" {
                continue;
            }
            let mut rest = rest.to_string();
            if let Some(p) = rest.find(" op_sel:") {
                rest.truncate(p)
            }
            let mut off = 0;
            if let Some(p) = rest.find(" offset:") {
                off = rest[p + 8..]
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                rest.truncate(p)
            }
            if let Some(p) = rest.find(" offen") {
                rest.truncate(p)
            }
            let o = if rest.is_empty() {
                vec![]
            } else {
                rest.split(", ").map(opd).collect()
            };
            prog.push(Ins {
                m: m.to_string(),
                o,
                off,
            });
        }
        (prog, labels)
    }
    struct Mem {
        regions: Vec<(u64, Vec<u8>)>,
    }
    impl Mem {
        fn at(&mut self, addr: u64, n: usize) -> &mut [u8] {
            let r = self
                .regions
                .iter_mut()
                .find(|(b, v)| addr >= *b && addr + n as u64 <= *b + v.len() as u64)
                .unwrap_or_else(|| panic!("unmapped access {addr:#x}+{n}"));
            let o = (addr - r.0) as usize;
            &mut r.1[o..o + n]
        }
    }
    struct Wave {
        v: Vec<[u32; 32]>,
        s: [u32; 128],
        exec: u32,
        scc: bool,
        pc: usize,
        done: bool,
    }
    impl Wave {
        fn sc(&self, o: &Opd, t: [u32; 2]) -> u32 {
            match o {
                Opd::S(n, _) => self.s[usize::from(*n)],
                Opd::Imm(x) => *x,
                Opd::Ttmp(9) => t[0],
                Opd::Ttmp(7) => t[1],
                Opd::Exec => self.exec,
                Opd::Null => 0,
                _ => panic!("scalar {o:?}"),
            }
        }
        fn rd(&self, o: &Opd, l: usize, t: [u32; 2]) -> u32 {
            match o {
                Opd::V(n, h) => {
                    let x = self.v[usize::from(*n)][l];
                    match h {
                        Some(false) => x & 0xffff,
                        Some(true) => x >> 16,
                        None => x,
                    }
                }
                _ => self.sc(o, t),
            }
        }
        fn wr(&mut self, o: &Opd, l: usize, x: u32) {
            let Opd::V(n, h) = o else {
                panic!("vector destination {o:?}")
            };
            let r = &mut self.v[usize::from(*n)][l];
            *r = match h {
                Some(false) => (*r & 0xffff_0000) | (x & 0xffff),
                Some(true) => (*r & 0xffff) | (x << 16),
                None => x,
            };
        }
        fn base(o: &Opd) -> usize {
            match o {
                Opd::V(n, _) | Opd::S(n, _) => usize::from(*n),
                _ => panic!("register {o:?}"),
            }
        }
        fn sdst(&mut self, o: &Opd, x: u32) {
            match o {
                Opd::Exec => self.exec = x,
                _ => self.s[Self::base(o)] = x,
            }
        }
    }
    fn f16f(h: u32) -> f32 {
        f32::from_bits(f16_to_f32(h as u16))
    }
    fn f16(x: f32) -> u16 {
        f32_to_f16(x.to_bits())
    }
    /// f16 fma, one rounding: exact in f64 when `b` is a nibble (0..15).
    fn fma16(a: u32, b: u32, c: u32) -> u32 {
        let q = f16f(b);
        assert!(
            q.fract() == 0.0 && (0.0..16.0).contains(&q),
            "dequant multiplicand {q} is not a nibble"
        );
        let x = f64::from(f16f(a)) * f64::from(q) + f64::from(f16f(c));
        if x == 0.0 || x.abs() >= 65520.0 {
            return u32::from(f32_to_f16((x as f32).to_bits()));
        }
        let quantum = 2f64.powi((x.abs().log2().floor() as i32).max(-14) - 10);
        u32::from(f32_to_f16(
            (((x / quantum).round_ties_even() * quantum) as f32).to_bits(),
        ))
    }

    /// Run one workgroup of the parsed kernel to completion.
    fn run_wg(
        prog: &(Vec<Ins>, HashMap<String, usize>),
        mem: &mut Mem,
        lds: &mut [u8],
        waves: usize,
        kernarg: u64,
        t: [u32; 2],
    ) {
        let (code, labels) = prog;
        let mut w: Vec<Wave> = (0..waves)
            .map(|i| {
                let mut s = [0u32; 128];
                s[0] = kernarg as u32;
                s[1] = (kernarg >> 32) as u32;
                let mut v = vec![[0u32; 32]; 256];
                for l in 0..32 {
                    v[0][l] = (32 * i + l) as u32;
                }
                Wave {
                    v,
                    s,
                    exec: u32::MAX,
                    scc: false,
                    pc: 0,
                    done: false,
                }
            })
            .collect();
        loop {
            let mut waiting = 0;
            for wv in w.iter_mut().filter(|x| !x.done) {
                loop {
                    let i = &code[wv.pc];
                    wv.pc += 1;
                    if step(wv, i, labels, mem, lds, t) {
                        waiting += 1;
                        break;
                    }
                    if wv.done {
                        break;
                    }
                }
            }
            if w.iter().all(|x| x.done) {
                return;
            }
            assert_eq!(
                waiting,
                w.iter().filter(|x| !x.done).count(),
                "a wave ended while others wait at a barrier"
            );
        }
    }
    /// Execute one instruction; true at a barrier wait.
    fn step(
        w: &mut Wave,
        i: &Ins,
        labels: &HashMap<String, usize>,
        mem: &mut Mem,
        lds: &mut [u8],
        t: [u32; 2],
    ) -> bool {
        let o = &i.o;
        let lanes = |exec: u32| (0..32usize).filter(move |l| exec >> l & 1 != 0);
        let valu = |w: &mut Wave, f: &dyn Fn(&Wave, usize) -> u32| {
            for l in lanes(w.exec) {
                let x = f(w, l);
                w.wr(&o[0], l, x);
            }
        };
        let vcmp = |w: &mut Wave, f: &dyn Fn(i32, i32) -> bool| {
            let mut m = 0u32;
            for l in lanes(w.exec) {
                if f(w.rd(&o[1], l, t) as i32, w.rd(&o[2], l, t) as i32) {
                    m |= 1 << l
                }
            }
            w.sdst(&o[0], m);
        };
        let s = |w: &Wave, k: usize| w.sc(&o[k], t);
        let label = |o: &Opd| match o {
            Opd::Label(l) => labels[l],
            _ => panic!("branch target {o:?}"),
        };
        match i.m.as_str() {
            "s_barrier_signal" => {}
            "s_barrier_wait" => return true,
            "s_endpgm" => w.done = true,
            "s_branch" => w.pc = label(&o[0]),
            "s_cbranch_scc1" => {
                if w.scc {
                    w.pc = label(&o[0])
                }
            }
            "s_load_b32" | "s_load_b64" | "s_load_b128" | "s_load_b256" => {
                let p = Wave::base(&o[1]);
                let addr = u64::from(w.s[p]) | u64::from(w.s[p + 1]) << 32;
                let (Opd::S(d, n), Opd::Imm(off)) = (&o[0], &o[2]) else {
                    panic!("{}", i.m)
                };
                for k in 0..usize::from(*n) {
                    w.s[usize::from(*d) + k] = u32::from_le_bytes(
                        mem.at(addr + u64::from(*off) + 4 * k as u64, 4)
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            "s_mov_b32" => {
                let x = s(w, 1);
                w.sdst(&o[0], x)
            }
            "s_and_b32" => {
                let x = s(w, 1) & s(w, 2);
                w.sdst(&o[0], x);
                w.scc = x != 0
            }
            "s_or_b32" => {
                let x = s(w, 1) | s(w, 2);
                w.sdst(&o[0], x);
                w.scc = x != 0
            }
            "s_mul_i32" => {
                let x = s(w, 1).wrapping_mul(s(w, 2));
                w.sdst(&o[0], x)
            }
            "s_lshl_b32" => {
                let x = s(w, 1) << (s(w, 2) & 31);
                w.sdst(&o[0], x)
            }
            "s_ashr_i32" => {
                let x = ((s(w, 1) as i32) >> (s(w, 2) & 31)) as u32;
                w.sdst(&o[0], x)
            }
            "s_add_co_i32" => {
                let x = s(w, 1).wrapping_add(s(w, 2));
                w.sdst(&o[0], x)
            }
            "s_min_u32" => {
                let x = s(w, 1).min(s(w, 2));
                w.sdst(&o[0], x)
            }
            "s_cmp_ge_i32" => w.scc = s(w, 0) as i32 >= s(w, 1) as i32,
            "s_cmp_lt_i32" => w.scc = (s(w, 0) as i32) < s(w, 1) as i32,
            "s_cmp_lt_u32" => w.scc = s(w, 0) < s(w, 1),
            "s_cmp_lg_u32" => w.scc = s(w, 0) != s(w, 1),
            "v_mov_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t)),
            "v_and_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t) & w.rd(&o[2], l, t)),
            "v_or_b32_e32" => valu(w, &|w, l| w.rd(&o[1], l, t) | w.rd(&o[2], l, t)),
            "v_add_nc_u32_e32" => {
                valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_add(w.rd(&o[2], l, t)))
            }
            "v_sub_nc_u32_e32" => {
                valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_sub(w.rd(&o[2], l, t)))
            }
            "v_min_i32_e32" => valu(w, &|w, l| {
                (w.rd(&o[1], l, t) as i32).min(w.rd(&o[2], l, t) as i32) as u32
            }),
            "v_mul_lo_u32" => valu(w, &|w, l| w.rd(&o[1], l, t).wrapping_mul(w.rd(&o[2], l, t))),
            "v_lshlrev_b32_e32" => valu(w, &|w, l| w.rd(&o[2], l, t) << (w.rd(&o[1], l, t) & 31)),
            "v_lshrrev_b32_e32" => valu(w, &|w, l| w.rd(&o[2], l, t) >> (w.rd(&o[1], l, t) & 31)),
            "v_lshl_add_u32" => valu(w, &|w, l| {
                (w.rd(&o[1], l, t) << (w.rd(&o[2], l, t) & 31)).wrapping_add(w.rd(&o[3], l, t))
            }),
            "v_bfe_u32" => valu(w, &|w, l| {
                let (x, a, b) = (
                    w.rd(&o[1], l, t),
                    w.rd(&o[2], l, t) & 31,
                    w.rd(&o[3], l, t) & 31,
                );
                (x >> a) & ((1u64 << b) - 1) as u32
            }),
            "v_mad_u32_u24" => valu(w, &|w, l| {
                (w.rd(&o[1], l, t) & 0xff_ffff)
                    .wrapping_mul(w.rd(&o[2], l, t) & 0xff_ffff)
                    .wrapping_add(w.rd(&o[3], l, t))
            }),
            "v_cndmask_b32_e64" => valu(w, &|w, l| {
                if w.sc(&o[3], t) >> l & 1 != 0 {
                    w.rd(&o[2], l, t)
                } else {
                    w.rd(&o[1], l, t)
                }
            }),
            "v_add_f32_e32" => valu(w, &|w, l| {
                (f32::from_bits(w.rd(&o[1], l, t)) + f32::from_bits(w.rd(&o[2], l, t))).to_bits()
            }),
            "v_cvt_f32_ubyte0_e32"
            | "v_cvt_f32_ubyte1_e32"
            | "v_cvt_f32_ubyte2_e32"
            | "v_cvt_f32_ubyte3_e32" => {
                let sh = 8 * u32::from(i.m.as_bytes()["v_cvt_f32_ubyte".len()] - b'0');
                valu(w, &|w, l| {
                    ((w.rd(&o[1], l, t) >> sh & 0xff) as f32).to_bits()
                })
            }
            "v_cvt_f16_f32_e64" => valu(w, &|w, l| u32::from(f32_to_f16(w.rd(&o[1], l, t)))),
            "v_fma_f16" => valu(w, &|w, l| {
                fma16(w.rd(&o[1], l, t), w.rd(&o[2], l, t), w.rd(&o[3], l, t))
            }),
            "v_cmp_gt_i32_e64" => vcmp(w, &|a, b| a > b),
            "v_cmp_le_i32_e64" => vcmp(w, &|a, b| a <= b),
            "v_mad_co_u64_u32" => {
                let (d, c) = (Wave::base(&o[0]), Wave::base(&o[4]));
                for l in lanes(w.exec) {
                    let x = u64::from(w.rd(&o[2], l, t)) * u64::from(w.rd(&o[3], l, t))
                        + (u64::from(w.v[c][l]) | u64::from(w.v[c + 1][l]) << 32);
                    w.v[d][l] = x as u32;
                    w.v[d + 1][l] = (x >> 32) as u32;
                }
            }
            "v_wmma_f32_16x16x16_f16" => {
                let (d, a, b, c) = (
                    Wave::base(&o[0]),
                    Wave::base(&o[1]),
                    Wave::base(&o[2]),
                    Wave::base(&o[3]),
                );
                let half = |r: usize, l: usize, k: usize| {
                    (w.v[r + (k % 8) / 2][l] >> (16 * (k % 2))) as u16
                };
                let mut out = [[0u32; 32]; 8];
                for row in 0..16 {
                    for col in 0..16 {
                        let av: Vec<u16> =
                            (0..16).map(|k| half(a, row + 16 * (k / 8), k)).collect();
                        let bv: Vec<u16> =
                            (0..16).map(|k| half(b, col + 16 * (k / 8), k)).collect();
                        let (lane, reg) = (col + 16 * (row / 8), row % 8);
                        out[reg][lane] = wmma_f32_f16(
                            peacemaker_ir::Arch::Gfx1201,
                            &av,
                            &bv,
                            w.v[c + reg][lane],
                        );
                    }
                }
                for (r, x) in out.into_iter().enumerate() {
                    w.v[d + r] = x;
                }
            }
            "ds_load_b128" | "ds_store_b128" => {
                let load = i.m == "ds_load_b128";
                let (addr, data) = if load {
                    (&o[1], Wave::base(&o[0]))
                } else {
                    (&o[0], Wave::base(&o[1]))
                };
                for l in lanes(w.exec) {
                    let at = (w.rd(addr, l, t) + i.off) as usize;
                    assert!(at + 16 <= lds.len(), "LDS access {at} past {}", lds.len());
                    for k in 0..4 {
                        if load {
                            w.v[data + k][l] = u32::from_le_bytes(
                                lds[at + 4 * k..at + 4 * k + 4].try_into().unwrap(),
                            )
                        } else {
                            lds[at + 4 * k..at + 4 * k + 4]
                                .copy_from_slice(&w.v[data + k][l].to_le_bytes())
                        }
                    }
                }
            }
            "buffer_load_b128" => {
                let srd = Wave::base(&o[2]);
                let base = u64::from(w.s[srd]) | u64::from(w.s[srd + 1] & 0xffff) << 32;
                let d = Wave::base(&o[0]);
                for l in lanes(w.exec) {
                    let at = w.rd(&o[1], l, t) + s(w, 3) + i.off;
                    for k in 0..4u32 {
                        let x = if at + 4 * k + 4 > w.s[srd + 2] {
                            0
                        } else {
                            u32::from_le_bytes(
                                mem.at(base + u64::from(at + 4 * k), 4).try_into().unwrap(),
                            )
                        };
                        w.v[d + k as usize][l] = x;
                    }
                }
            }
            m if m.starts_with("global_load_b") || m.starts_with("global_store_b") => {
                let load = m.starts_with("global_load");
                let dw = match m.rsplit_once("_b").unwrap().1 {
                    "32" => 1,
                    "64" => 2,
                    "128" => 4,
                    x => panic!("{x}"),
                };
                let (p, d) = if load {
                    (Wave::base(&o[1]), Wave::base(&o[0]))
                } else {
                    (Wave::base(&o[0]), Wave::base(&o[1]))
                };
                for l in lanes(w.exec) {
                    let addr =
                        (u64::from(w.v[p][l]) | u64::from(w.v[p + 1][l]) << 32) + u64::from(i.off);
                    for k in 0..dw {
                        let at = mem.at(addr + 4 * k as u64, 4);
                        if load {
                            w.v[d + k][l] = u32::from_le_bytes((&*at).try_into().unwrap())
                        } else {
                            at.copy_from_slice(&w.v[d + k][l].to_le_bytes())
                        }
                    }
                }
            }
            m => panic!("interpreter: no semantics for {m}"),
        }
        false
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
    }
    const CANARY: usize = 64;
    const SENTINEL: u32 = 0x7FBA_DBAD;

    fn run_case(text: &str, spec: Spec, ms: &[u32], k: u32, n: u32, seed: u64) {
        let mut rng = Rng(seed);
        let nm = ms.len();
        let gpr = (k / 256) as usize;
        let row_bytes = gpr * GROUP as usize;
        // Weights: f16 scale/zero headers (mixed signs, a tiny zero point), random nibbles.
        let a: Vec<Vec<u8>> = ms
            .iter()
            .map(|&m| {
                let mut w = vec![0u8; m as usize * row_bytes];
                for row in w.chunks_mut(GROUP as usize) {
                    for h in 0..2 {
                        let sc = f16(((rng.next() % 2001) as f32 - 1000.0) * 1.3e-4);
                        let zp = if rng.next() % 7 == 0 {
                            0x0001
                        } else {
                            f16(((rng.next() % 2001) as f32 - 1000.0) * 9e-4)
                        };
                        row[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                        row[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
                    }
                    for byte in &mut row[8..] {
                        *byte = rng.next() as u8;
                    }
                }
                w
            })
            .collect();
        let x: Vec<u16> = (0..n as usize * k as usize)
            .map(|_| match rng.next() % 41 {
                0 => 0x8000,
                1 => 0,
                2 => 0x0003,
                _ => f16(((rng.next() % 4001) as f32 - 2000.0) * 1e-3),
            })
            .collect();
        let y0: Vec<Vec<u32>> = ms
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                (0..n as usize * m as usize + CANARY)
                    .map(|e| {
                        if spec.op.accumulates() && e < n as usize * m as usize {
                            (1.0f32 + (e + 7 * i) as f32 * 0.001).to_bits()
                                ^ (u32::from(e % 5 == 0) << 31)
                        } else {
                            SENTINEL
                        }
                    })
                    .collect()
            })
            .collect();
        // Reference.
        let deq = |i: usize, r: usize, kk: usize| -> u16 {
            let g = &a[i][r * row_bytes + (kk / 256) * GROUP as usize..];
            let h = if kk % 256 < 128 { 0 } else { 4 };
            let sc = u16::from_le_bytes([g[h], g[h + 1]]);
            let zp = u16::from_le_bytes([g[h + 2], g[h + 3]]);
            let byte = g[8 + (kk % 256) / 2];
            let q = if kk % 2 == 0 { byte & 15 } else { byte >> 4 };
            fma16(u32::from(sc), u32::from(f16(f32::from(q))), u32::from(zp)) as u16
        };
        let mut want = y0.clone();
        for (i, &m) in ms.iter().enumerate() {
            for r in 0..m as usize {
                let arow: Vec<u16> = (0..gpr * 256).map(|kk| deq(i, r, kk)).collect();
                for t in 0..n as usize {
                    let mut c = 0u32;
                    for tile in 0..gpr * 16 {
                        let xs = &x[t * k as usize + 16 * tile..][..16];
                        c = wmma_f32_f16(
                            peacemaker_ir::Arch::Gfx1201,
                            &arow[16 * tile..16 * tile + 16],
                            xs,
                            c,
                        );
                    }
                    let o = &mut want[i][t * m as usize + r];
                    *o = if spec.op.accumulates() {
                        (f32::from_bits(*o) + f32::from_bits(c)).to_bits()
                    } else {
                        c
                    };
                }
            }
        }
        // Device image.
        let le = |v: &[u32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        let a_base = |i: usize| 0x1000_0000u64 + 0x0100_0000 * i as u64;
        let y_base = |i: usize| 0x3000_0000u64 + 0x0100_0000 * i as u64;
        let (x_base, kernarg_ptr) = (0x2000_0000u64, 0x4000_0000u64);
        let mut karg = Vec::new();
        for i in 0..nm {
            karg.extend(a_base(i).to_le_bytes());
        }
        karg.extend(x_base.to_le_bytes());
        for i in 0..nm {
            karg.extend(y_base(i).to_le_bytes());
        }
        for &m in ms {
            karg.extend(m.to_le_bytes());
        }
        karg.extend(k.to_le_bytes());
        karg.extend(n.to_le_bytes());
        assert_eq!(karg.len() as u32, spec.op.kernarg_bytes());
        let mut regions: Vec<(u64, Vec<u8>)> = (0..nm).map(|i| (a_base(i), a[i].clone())).collect();
        regions.push((x_base, x.iter().flat_map(|h| h.to_le_bytes()).collect()));
        regions.extend((0..nm).map(|i| (y_base(i), le(&y0[i]))));
        regions.push((kernarg_ptr, karg));
        let mut mem = Mem { regions };
        let prog = parse(text);
        let total: u32 = ms.iter().sum();
        for gy in 0..n.div_ceil(spec.tokens()) {
            for gx in 0..total.div_ceil(spec.rows()) {
                let mut lds = vec![0xA5u8; spec.lds_bytes() as usize];
                run_wg(
                    &prog,
                    &mut mem,
                    &mut lds,
                    usize::from(spec.waves),
                    kernarg_ptr,
                    [gx, gy],
                );
            }
        }
        for (i, &m) in ms.iter().enumerate() {
            let got = mem.at(y_base(i), want[i].len() * 4).to_vec();
            for (e, w) in want[i].iter().enumerate() {
                let g = u32::from_le_bytes(got[4 * e..4 * e + 4].try_into().unwrap());
                assert_eq!(g, *w, "{} rows {ms:?} K {k} N {n}: Y{i} element {e} (token {}, row {}): got {g:#010x} want {w:#010x}",
                    spec.symbol(), e / m as usize, e % m as usize);
            }
        }
    }

    #[test]
    fn interpreted_entries_are_byte_exact_against_the_singleton_chain() {
        for spec in Spec::all(Arch::Gfx1201) {
            let text = emit(spec).unwrap().s_text;
            let (aligned, ragged): (&[u32], &[u32]) = match spec.op {
                VerifyOp::Qkvza => (&[24, 16, 8, 16], &[21, 13, 7, 22]),
                VerifyOp::Qkv => (&[16, 24, 16], &[21, 13, 22]),
                VerifyOp::GateUp => (&[32, 24], &[27, 30]),
                VerifyOp::Residual => (&[64], &[61]),
            };
            run_case(
                &text,
                spec,
                aligned,
                768,
                77,
                0x9e37_79b9_7f4a_7c15 ^ u64::from(spec.bt),
            );
            run_case(
                &text,
                spec,
                ragged,
                512,
                if spec.bt == 4 { 64 } else { 128 },
                0x2545_f491_4f6c_dd1d ^ u64::from(spec.waves),
            );
            run_case(&text, spec, aligned, 256, 64, 0x1234_5678_9abc_def1);
        }
    }
}
