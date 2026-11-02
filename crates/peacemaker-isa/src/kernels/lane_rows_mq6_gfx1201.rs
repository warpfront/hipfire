// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 (R9700, wave32) MQ6G256V2 multi-row GEMV twins, module
//! `lane_rows_mq6_pm_gfx1201` (image `kernels/lane_rows_mq6_pm_gfx1201.hxaco`).
//!
//! Every output `(row c, output n)` of every entry is bitwise the gfx1201
//! singleton `gemv_mq6g256v2` / `gemv_mq6g256v2_x4` (hipcc object, see
//! [`REFERENCE`]) of activation row `c` alone, for every live `C`. The DAG is
//! taken from that object's disassembly (both entries share one body; main
//! quad groups and the three tail groups are the same expression):
//!  * lane `l` of the output's wave covers weights `8l..8l+7` of every
//!    256-weight group (group = 200 B: half headers `h0` (lanes 0-15) / `h1`
//!    (lanes 16-31) in bytes 0..7, six payload bytes per lane at `8 + 6l`);
//!    `q0..q7` are the unsigned 6-bit fields of the payload
//!    (`q5 = ((p >> 30) | (u << 2)) & 63`, which is the object's
//!    `v_alignbit_b32(u, p, 30) & 63`; `p` the dword at `8+6l`, `u` the
//!    zero-extended u16 at `12+6l`);
//!  * `w_i = v_fma_mix_f32(sc_f16, f32(q_i), zp_f16)` (one rounding);
//!  * group dot `d = x1 * w1` (`v_mul_f32 d, x1, w1`), then `d = fma(x0, w0, d)`,
//!    `fma(x2, w2, d)` .. `fma(x7, w7, d)` as `v_fmac_f32 d, x_i, w_i`: the
//!    activation is src0 and the weight src1 in every term (the object's
//!    orientation; the gfx1151 twin's source comment says `w, x`, the gfx1201
//!    object does not);
//!  * group `g` adds into stream `g % 4` as `a_s = a_s + d` (streams start
//!    `+0`; quad loop, then up to three tail groups into streams 0..2);
//!  * fold `(a0 + a1) + (a3 + a2)`, then `ds_swizzle_b32` `"1pppp"` and
//!    `ds_bpermute_b32` +8/+4/+2/+1 (own lane when out of range),
//!    `own + other` at every level; lane 0 stores.
//! FP mode of the descriptor: RNE, denormals preserved, WGP mode, memory
//! ordered, forward progress (the reference's RSRC1).
//!
//! Entries (module order, see [`SYMBOLS`]). All kernarg ABIs are i32/ptr by
//! value, little endian: single 36 B `A@0 x@8 y@16 M@24 K@28 C@32`, x4 96 B
//! `A0..A3@0..24 x@32 y0..y3@40..64 M0..M3@72..84 K@88 C@92`. `x` is F32
//! `[C][K]`, `y` is F32 `[C][M]` (x4: `y_i` is `[C][M_i]`), `K % 256 == 0`.
//! `C` outside `1..=cap` (`c1`: exactly 1) returns before any load, store or
//! barrier; only live rows are loaded and stored.
//!  * **S1** `c1` / `xbatch_r4` / `xbatch_r8` (single and x4): block 32, one
//!    output per wave, no LDS, grid `M` (x4: `sum max(Mi, 0)`; workgroup `w`
//!    maps to the first matrix `i` with `w < M0+..+Mi`). A runtime-`C` branch
//!    selects an unrolled 1..=cap section. Per group the weights are loaded
//!    and dequantized once and every live row's X streams through a register
//!    ring (two rows per slot pair, dotted in `v_dual_*` packets whose halves
//!    share the weight register as src1: exact gfx12 `src1` sharing; src0
//!    banks differ because the pair's slots sit 9 VGPRs apart; destinations
//!    alternate parity). Weight loads use a per-output raw buffer resource
//!    (64-bit row base, no 4 GiB limit) with `scope:SCOPE_DEV`; every group
//!    body drains its loads, so loops and tails branch with empty ledgers.
//!  * **S2** `xbatch_r{4,8}_wg8` (single and x4): block 256 = eight wave32
//!    output owners (matrix-local output `8*(w - base) + wave`), grid
//!    `ceil(M/8)` (x4: `sum ceil(max(Mi, 0)/8)`, a workgroup never spans two
//!    matrices), static LDS `cap * 1024` B (`[C][256]` F32). Per K group each
//!    valid wave loads/dequantizes its weights once; wave `v < C` stages X
//!    row `v` into LDS (`(v*8 + j)*32 + lane` words), one uniform barrier,
//!    valid waves consume rows `0..C`, one uniform barrier. Waves past `M`
//!    still stage and hit every barrier and skip weights, arithmetic and
//!    stores. No barrier depends on anything but `C` and `K`.
//!  * **Rowtiles** built by [`emit_rowtiles`] use the unchanged S2 physical
//!    eight-row tile with total logical `R=1..64` in the final kernarg.
//!    Grid y is `ceil(R/8)`: tile `t` reads/writes absolute rows `8*t+j`;
//!    x4 uses each selected matrix's original M stride. Block256, LDS8192,
//!    and the eight-row register footprint do not grow with logical R.
//!    The narrow exports and their emitted instruction streams are unchanged.
//!
//! Domain: every matrix, `x` and `y` byte span below 4 GiB (32-bit voffsets).
use super::iu4_gemm::ModuleProof;
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{lit, op, smem, sop, sr, SRD_WORD3};
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::vopd::{Operand, VopdF32, VopdOp};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// Module name; the committed image is `kernels/lane_rows_mq6_pm_gfx1201.hxaco`.
pub const MODULE: &str = "lane_rows_mq6_pm_gfx1201";

/// The exports, in module order: S1 single, S1 x4, then S2 single, S2 x4.
pub const SYMBOLS: [&str; 10] = [
    "gemv_mq6g256v2_c1_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r4_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r8_pm_gfx1201",
    "gemv_mq6g256v2_x4_c1_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r4_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r8_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r4_wg8_pm_gfx1201",
    "gemv_mq6g256v2_xbatch_r8_wg8_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_mq6g256v2_x4_xbatch_r8_wg8_pm_gfx1201",
];

/// Kernarg bytes of each export, in [`SYMBOLS`] order.
pub const KERNARG_BYTES: [u32; 10] = [36, 36, 36, 96, 96, 96, 36, 36, 96, 96];

const SINGLE_ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)];
const X4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 8),
    (48, 8),
    (56, 8),
    (64, 8),
    (72, 4),
    (76, 4),
    (80, 4),
    (84, 4),
    (88, 4),
    (92, 4),
];
/// `(offset, size)` of every kernarg of each export, in [`SYMBOLS`] order.
pub const KERNARGS: [&[(u32, u32)]; 10] = [
    SINGLE_ARGS,
    SINGLE_ARGS,
    SINGLE_ARGS,
    X4_ARGS,
    X4_ARGS,
    X4_ARGS,
    SINGLE_ARGS,
    SINGLE_ARGS,
    X4_ARGS,
    X4_ARGS,
];

/// Workgroup size (threads) of each export, in [`SYMBOLS`] order.
pub const WORKGROUP_SIZE: [u32; 10] = [32, 32, 32, 32, 32, 32, 256, 256, 256, 256];

/// Static LDS bytes of each export, in [`SYMBOLS`] order.
pub const LDS_BYTES: [u32; 10] = [0, 0, 0, 0, 0, 0, 4096, 8192, 4096, 8192];

/// Most activation rows `C` of each export, in [`SYMBOLS`] order (`c1`: exactly 1).
pub const MAX_ROWS: [u8; 10] = [1, 4, 8, 1, 4, 8, 4, 8, 4, 8];

/// Launch grid (workgroups, x) of each export, in [`SYMBOLS`] order.
pub const GRID_RULE: [&str; 10] = [
    "M",
    "M",
    "M",
    "sum max(Mi,0)",
    "sum max(Mi,0)",
    "sum max(Mi,0)",
    "ceil(M/8)",
    "ceil(M/8)",
    "sum ceil(max(Mi,0)/8)",
    "sum ceil(max(Mi,0)/8)",
];

/// Per-output DAG identity: equal strings mean "same per-output DAG as the
/// singleton reference" (single and x4 are one recipe in the gfx1201 object).
pub const RECIPE_ID: &str = "mq6g256v2.gfx1201.singleton-dag.v1";
/// [`RECIPE_ID`] for each export, in [`SYMBOLS`] order.
pub const RECIPE: [&str; 10] = [RECIPE_ID; 10];

/// A singleton reference object: runtime module, kernel symbol, and the
/// sha256 of the hipcc bundle (identical to the runtime hot-cache hsaco).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reference {
    pub module: &'static str,
    pub symbol: &'static str,
    pub fixture: &'static str,
    pub sha256: &'static str,
}
/// `Gpu::gemv_mq6g256v2` -> `kernels/src/gemv_mq6g256v2.hip` (module `gemv_mq6g256v2_mq6v2`).
pub const SINGLE_REFERENCE: Reference = Reference {
    module: "gemv_mq6g256v2_mq6v2",
    symbol: "gemv_mq6g256v2",
    fixture: "gemv_mq6g256v2",
    sha256: "18b2536f2c4f6e01505d214f5816d9703af8880ea5a8063cb0ff23fc6c27df5d",
};
/// `Gpu::gemv_mq6g256v2_x4` -> `kernels/src/qwen4_gemv_mq6g256v2.hip` (module `qwen4_gemv_mq6g256v2`).
pub const X4_REFERENCE: Reference = Reference {
    module: "qwen4_gemv_mq6g256v2",
    symbol: "gemv_mq6g256v2_x4",
    fixture: "qwen4_gemv_mq6g256v2",
    sha256: "18c105a770a1f073da20d84d4a05aba2e393fb56dd31a254ce48812f2b1cf4d3",
};
/// The singleton each export is a twin of, in [`SYMBOLS`] order.
pub const REFERENCE: [Reference; 10] = [
    SINGLE_REFERENCE,
    SINGLE_REFERENCE,
    SINGLE_REFERENCE,
    X4_REFERENCE,
    X4_REFERENCE,
    X4_REFERENCE,
    SINGLE_REFERENCE,
    SINGLE_REFERENCE,
    X4_REFERENCE,
    X4_REFERENCE,
];

/// Committed image file name.
pub fn bundle_name() -> String {
    format!("{MODULE}.hxaco")
}

/// One export: single or x4, row capacity, and the S2 (`wg8`) schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub x4: bool,
    pub cap: u8,
    pub wg8: bool,
}

/// The exports in [`SYMBOLS`] order.
pub const SPECS: [Spec; 10] = [
    Spec {
        x4: false,
        cap: 1,
        wg8: false,
    },
    Spec {
        x4: false,
        cap: 4,
        wg8: false,
    },
    Spec {
        x4: false,
        cap: 8,
        wg8: false,
    },
    Spec {
        x4: true,
        cap: 1,
        wg8: false,
    },
    Spec {
        x4: true,
        cap: 4,
        wg8: false,
    },
    Spec {
        x4: true,
        cap: 8,
        wg8: false,
    },
    Spec {
        x4: false,
        cap: 4,
        wg8: true,
    },
    Spec {
        x4: false,
        cap: 8,
        wg8: true,
    },
    Spec {
        x4: true,
        cap: 4,
        wg8: true,
    },
    Spec {
        x4: true,
        cap: 8,
        wg8: true,
    },
];

impl Spec {
    pub fn symbol(&self) -> String {
        let x4 = if self.x4 { "x4_" } else { "" };
        match (self.cap, self.wg8) {
            (1, false) => format!("gemv_mq6g256v2_{x4}c1_pm_gfx1201"),
            (cap, false) => format!("gemv_mq6g256v2_{x4}xbatch_r{cap}_pm_gfx1201"),
            (cap, true) => format!("gemv_mq6g256v2_{x4}xbatch_r{cap}_wg8_pm_gfx1201"),
        }
    }
    pub fn validate(&self) -> R {
        match (self.cap, self.wg8) {
            (1 | 4 | 8, false) | (4 | 8, true) => Ok(()),
            (cap, wg8) => Err(format!(
                "mq6 gfx1201 has no capacity {cap} (wg8 {wg8}) entry"
            )),
        }
    }
    /// Threads per workgroup.
    pub fn block(&self) -> u16 {
        if self.wg8 {
            256
        } else {
            32
        }
    }
    /// Static LDS bytes.
    pub fn lds(&self) -> u32 {
        if self.wg8 {
            u32::from(self.cap) * 1024
        } else {
            0
        }
    }
    /// Label prefix, unique per export (one assembler unit holds the module).
    fn tag(&self) -> String {
        format!(
            "mq6{}{}{}",
            if self.x4 { "x4" } else { "" },
            if self.wg8 { "w8r" } else { "r" },
            self.cap
        )
    }
    fn variant(&self) -> String {
        format!(
            "{}{}_r{}",
            if self.x4 { "x4_" } else { "" },
            if self.wg8 { "wg8" } else { "rows" },
            self.cap
        )
    }
    fn kernargs(&self) -> KernargLayout {
        if self.x4 {
            let mut k = KernargLayout::new(96);
            for (i, name) in ["A0", "A1", "A2", "A3"].into_iter().enumerate() {
                k = k.pointer_access(name, 8 * i as u32, Access::ReadOnly);
            }
            k = k.pointer_access("x", 32, Access::ReadOnly);
            for (i, name) in ["y0", "y1", "y2", "y3"].into_iter().enumerate() {
                k = k.pointer_access(name, 40 + 8 * i as u32, Access::WriteOnly);
            }
            for (i, name) in ["M0", "M1", "M2", "M3"].into_iter().enumerate() {
                k = k.hidden(name, 72 + 4 * i as u32, 4, "by_value");
            }
            k.hidden("K", 88, 4, "by_value")
                .hidden("C", 92, 4, "by_value")
        } else {
            KernargLayout::new(36)
                .pointer_access("A", 0, Access::ReadOnly)
                .pointer_access("x", 8, Access::ReadOnly)
                .pointer_access("y", 16, Access::WriteOnly)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value")
                .hidden("C", 32, 4, "by_value")
        }
    }
}

// ------------------------------------------------------------------ shared
/// MQ6 group bytes.
const GROUP: u32 = 200;
/// Group-dot term order: x1 first (`v_mul`), then x0, x2 .. x7.
const TERMS: [u8; 8] = [1, 0, 2, 3, 4, 5, 6, 7];
/// gfx12 device-scope cache policy of the weight loads.
const POLICY: &str = " scope:SCOPE_DEV";

// SGPRs. s0:1 kernarg; the workgroup id is `ttmp9`.
const KARG: u8 = 0;
/// Weight buffer resource s[4:7].
const SRD: u8 = 4;
const PA: u8 = 8;
const XP: u8 = 16;
const Y0: u8 = 18;
const Y1: u8 = 20;
const Y2: u8 = 22;
const Y3: u8 = 24;
const M0: u8 = 26;
const M1: u8 = 27;
const M2: u8 = 28;
const M3: u8 = 29;
const SK: u8 = 30;
const SC: u8 = 31;
const SY: u8 = 32;
const SM: u8 = 34;
const SBASE: u8 = 35;
const CP0: u8 = 36;
const CP1: u8 = 37;
const CP2: u8 = 38;
const CP3: u8 = 39;
const SW: u8 = 40;
const ROW: u8 = 41;
const GROUPS: u8 = 42;
const STRIDE: u8 = 43;
const TMP: u8 = 44;
const TMP2: u8 = 45;
const GOFF: u8 = 46;
const KB: u8 = 49;
const MB: u8 = 50;
const ROWB: u8 = 51;
const STMP: u8 = 52;

fn r8(n: u16) -> u8 {
    u8::try_from(n).expect("register index below 256")
}
fn offset(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn refs(kind: Kind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}

/// `s_cbranch_scc0` over a lone `s_endpgm`: leaves the wave when SCC is set.
fn exit_unless(b: &mut Builder, tag: &str, name: &str) -> R {
    branch(b, &format!("s_cbranch_scc0 .L{tag}_{name}"))?;
    branch(b, "s_endpgm")?;
    b.label(&format!(".L{tag}_{name}"))
}

/// Declare registers `0..end` in aligned 8/4/2/1-register pieces.
fn declare(regs: &mut RegPlan, kind: Kind, end: u16) -> R {
    let (prefix, mut at) = (if kind == Kind::V { "v" } else { "s" }, 0u16);
    while at < end {
        let piece = [8u16, 4, 2, 1]
            .into_iter()
            .find(|p| at % p == 0 && at + p <= end)
            .unwrap_or(1);
        regs.add_range(
            &format!("{prefix}{at}"),
            kind,
            r8(at),
            piece as u8,
            Live::Whole,
        )?;
        at += piece;
    }
    Ok(())
}
fn plan(vgprs: u16, sgprs: u16) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(vgprs, 96)?;
    declare(&mut regs, Kind::V, vgprs)?;
    declare(&mut regs, Kind::S, sgprs)?;
    Ok(regs)
}

/// Kernarg loads of the entry (single: straight into the routed registers).
fn kernarg_loads(b: &mut Builder, x4: bool) -> R {
    if x4 {
        smem(b, PA, 8, KARG, 0)?; // A0..A3
        smem(b, XP, 8, KARG, 32)?; // x y0 y1 y2
        smem(b, Y3, 4, KARG, 64)?; // y3 M0 M1
        smem(b, M2, 4, KARG, 80) // M2 M3 K C
    } else {
        smem(b, SRD, 2, KARG, 0)?;
        smem(b, XP, 2, KARG, 8)?;
        smem(b, SY, 2, KARG, 16)?;
        smem(b, SM, 1, KARG, 24)?;
        smem(b, SK, 1, KARG, 28)?;
        smem(b, SC, 1, KARG, 32)
    }
}

/// x4: workgroup `w` -> the first matrix `i` with `w < CP_i` (cumulative
/// workgroup counts, matrix 3 then 2, 1, 0 overriding while `w < CP_i`):
/// its A/y/M and the first workgroup of its matrix.
fn route(b: &mut Builder) -> R {
    let pair = |n: u8| sr(n, 2);
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP2}"), &[], &[SW, CP2])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{}:{}]",
            SRD + 1,
            PA + 4,
            PA + 5,
            PA + 6,
            PA + 7
        ),
        &[pair(SRD)],
        &[pair(PA + 4), pair(PA + 6)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y2}:{}], s[{Y3}:{}]",
            SY + 1,
            Y2 + 1,
            Y3 + 1
        ),
        &[pair(SY)],
        &[pair(Y2), pair(Y3)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M2}, s{M3}"),
        &[SM],
        &[M2, M3],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{CP1}, s{CP2}"),
        &[SBASE],
        &[CP1, CP2],
    )?;
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP1}"), &[], &[SW, CP1])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{SRD}:{}]",
            SRD + 1,
            PA + 2,
            PA + 3,
            SRD + 1
        ),
        &[pair(SRD)],
        &[pair(PA + 2), pair(SRD)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y1}:{}], s[{SY}:{}]",
            SY + 1,
            Y1 + 1,
            SY + 1
        ),
        &[pair(SY)],
        &[pair(Y1), pair(SY)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M1}, s{SM}"),
        &[SM],
        &[M1, SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{CP0}, s{SBASE}"),
        &[SBASE],
        &[CP0, SBASE],
    )?;
    sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP0}"), &[], &[SW, CP0])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SRD}:{}], s[{PA}:{}], s[{SRD}:{}]",
            SRD + 1,
            PA + 1,
            SRD + 1
        ),
        &[pair(SRD)],
        &[pair(PA), pair(SRD)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{SY}:{}], s[{Y0}:{}], s[{SY}:{}]",
            SY + 1,
            Y0 + 1,
            SY + 1
        ),
        &[pair(SY)],
        &[pair(Y0), pair(SY)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{M0}, s{SM}"),
        &[SM],
        &[M0, SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, 0, s{SBASE}"),
        &[SBASE],
        &[SBASE],
    )
}

/// `SRD = A + row * stride` (64-bit row base) as a raw, unbounded 32-bit buffer.
fn weight_resource(b: &mut Builder) -> R {
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{SK}, 8"),
        &[GROUPS],
        &[SK],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{STRIDE}, s{GROUPS}, {}", lit(GROUP)),
        &[STRIDE],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{ROW}, s{STRIDE}"),
        &[TMP],
        &[ROW, STRIDE],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{TMP2}, s{ROW}, s{STRIDE}"),
        &[TMP2],
        &[ROW, STRIDE],
    )?;
    sop(
        b,
        format!("s_add_co_u32 s{SRD}, s{SRD}, s{TMP}"),
        &[SRD],
        &[SRD, TMP],
    )?;
    sop(
        b,
        format!("s_add_co_ci_u32 s{}, s{}, s{TMP2}", SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1, TMP2],
    )?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1],
    )?;
    sop(b, format!("s_mov_b32 s{}, -1", SRD + 2), &[SRD + 2], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD + 3, lit(SRD_WORD3)),
        &[SRD + 3],
        &[],
    )
}

/// `w[0..8] = fma_mix(sc, f32(q_n), zp)` from the raw header `h`, payload
/// dword `p` and zero-extended u16 `u`; `p`/`u`/`h` are read only.
fn dequant_group(b: &mut Builder, h: u8, p: u8, u: u8, w: u8) -> R {
    valu(b, &format!("v_and_b32_e32 v{w}, 63, v{p}"), &[w], &[p], &[])?;
    for (n, lsb) in [(1u8, 6u32), (2, 12), (3, 18), (4, 24)] {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{p}, {lsb}, 6", w + n),
            &[w + n],
            &[p],
            &[],
        )?;
    }
    // q5 straddles p and u: ((p >> 30) | (u << 2)) & 63 == alignbit(u, p, 30) & 63.
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{0}, 30, v{p}", w + 5),
        &[w + 5],
        &[p],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshl_or_b32 v{0}, v{u}, 2, v{0}", w + 5),
        &[w + 5],
        &[u, w + 5],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{0}, 63, v{0}", w + 5),
        &[w + 5],
        &[w + 5],
        &[],
    )?;
    valu(
        b,
        &format!("v_bfe_u32 v{}, v{u}, 4, 6", w + 6),
        &[w + 6],
        &[u],
        &[],
    )?;
    valu(
        b,
        &format!("v_bfe_u32 v{}, v{u}, 10, 6", w + 7),
        &[w + 7],
        &[u],
        &[],
    )?;
    for n in 0..8u8 {
        valu(
            b,
            &format!("v_cvt_f32_ubyte0_e32 v{0}, v{0}", w + n),
            &[w + n],
            &[w + n],
            &[],
        )?;
    }
    for n in 0..8u8 {
        let d = w + n;
        valu(
            b,
            &format!("v_fma_mix_f32 v{d}, v{h}, v{d}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
            &[d],
            &[h, d],
            &[],
        )?;
    }
    Ok(())
}

/// Lane-swizzle-16 then bpermute 8/4/2/1 `own + other` tree over `values`
/// (value register, scratch register), batched across the values.
fn lane_tree(b: &mut Builder, lane: u8, shuf: u8, values: &[(u8, u8)]) -> R {
    for &(val, t) in values {
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                refs(Kind::V, &[t]),
                refs(Kind::V, &[val]),
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    for &(val, t) in values {
        valu(
            b,
            &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
            &[val],
            &[val, t],
            &[],
        )?;
    }
    for offs in [8u32, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{lane}", 32 - offs),
            &[],
            &[lane],
            &[],
        )?;
        b.push(Instruction::new(
            "s_wait_alu depctr_va_vcc(0)",
            vec![],
            vec![],
        ))?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{shuf}, 0, {offs}, vcc_lo"),
            &[shuf],
            &[],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{shuf}, v{shuf}, v{lane}, 2"),
            &[shuf],
            &[shuf, lane],
            &[],
        )?;
        for &(val, t) in values {
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{t}, v{shuf}, v{val}"),
                    refs(Kind::V, &[t]),
                    refs(Kind::V, &[shuf, val]),
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for &(val, t) in values {
            valu(
                b,
                &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
                &[val],
                &[val, t],
                &[],
            )?;
        }
    }
    Ok(())
}

// --------------------------------------------------------------------- S1
// VGPRs.
const LANE: u8 = 0;
/// `(lane >> 4) * 4`: half-header byte offset; `lane * 6`: payload byte offset;
/// `lane * 32`: the lane's eight activations (plus the quad's byte base).
const HALF: u8 = 1;
const PKB: u8 = 2;
const XOFF: u8 = 3;
const SHUF: u8 = 4;
/// Group-dot registers (even, odd): one VOPD pair of rows.
const DOT0: u8 = 6;
const DOT1: u8 = 7;
/// Raw loads of up to four groups (header, payload dword, payload u16 each).
const RAW0: u16 = 8;
/// Dequantized weights of the current group.
const W0: u16 = 20;
const ACC0: u16 = 28;
/// Per-column x byte pointer pairs (SGPR).
const XCOL: u8 = 64;
const QUADS: u8 = 47;
const TAIL: u8 = 48;

fn raw(idx: usize, k: u8) -> u8 {
    r8(RAW0 + 3 * idx as u16 + u16::from(k))
}
fn xcol(c: u8) -> u8 {
    XCOL + 2 * c
}

/// S1 register layout of one capacity.
#[derive(Clone, Copy)]
struct Lay {
    cap: u8,
    ring0: u16,
    stride: u16,
    slots: u16,
    high: u16,
}
impl Lay {
    /// `c1`: four one-row slots (the singleton's four groups in flight, 64 VGPRs).
    /// Rows: slot pairs `A`, `A + 9` (consecutive banks mod 4) at pitch 18.
    fn new(cap: u8) -> Lay {
        if cap == 1 {
            Lay {
                cap,
                ring0: 32,
                stride: 8,
                slots: 4,
                high: 64,
            }
        } else {
            let ring0 = ACC0 + 8 * u16::from(cap.div_ceil(2));
            let slots = if cap == 4 { 4 } else { 3 };
            Lay {
                cap,
                ring0,
                stride: 18,
                slots,
                high: ring0 + 18 * (slots - 1) + 17,
            }
        }
    }
    /// Accumulator of row `c`, stream `s`: rows of a pair are adjacent
    /// registers (opposite parity and bank), `c1` is dense.
    fn acc(&self, c: u8, s: u8) -> u8 {
        if self.cap == 1 {
            r8(ACC0 + u16::from(s))
        } else {
            r8(ACC0 + 8 * u16::from(c / 2) + 2 * u16::from(s) + u16::from(c % 2))
        }
    }
    /// First register of the ring slot of item `k`.
    fn slot(&self, k: usize) -> u8 {
        r8(self.ring0 + self.stride * (k as u16 % self.slots))
    }
    fn sgprs(&self) -> u16 {
        u16::from(XCOL) + 2 * u16::from(self.cap)
    }
}

/// One ring item: group slot `j` (= stream) against row `c0` (and `c1`).
struct Item {
    idx: usize,
    j: u8,
    c0: u8,
    c1: Option<u8>,
}
fn plan_items(nb: u8, js: &[u8]) -> Vec<Item> {
    let mut items = Vec::new();
    for (idx, &j) in js.iter().enumerate() {
        let mut c = 0;
        while c < nb {
            items.push(Item {
                idx,
                j,
                c0: c,
                c1: if c + 1 < nb { Some(c + 1) } else { None },
            });
            c += 2;
        }
    }
    items
}

/// S1 export: block 32, one output per wave.
fn emit_rows(spec: Spec) -> Result<Emitted, String> {
    let lay = Lay::new(spec.cap);
    let tag = spec.tag();
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "lane_rows_mq6_gfx1201".into(),
            variant: spec.variant(),
            arch: Arch::Gfx1201,
            symbol: spec.symbol(),
            kernargs: spec.kernargs(),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: spec.block(),
            group_segment_fixed_size: spec.lds(),
            wave32: true,
            cu_mode: false,
        },
        plan(lay.high, lay.sgprs())?,
    );
    b.enable_delay_alu();
    prologue_rows(&mut b, spec, &lay, &tag)?;
    // One specialized path per live row count; each section ends in `s_endpgm`.
    for nb in 1..=spec.cap {
        sop(&mut b, format!("s_cmp_eq_u32 s{SC}, {nb}"), &[], &[SC])?;
        branch(&mut b, &format!("s_cbranch_scc0 .L{tag}_skip{nb}"))?;
        section(&mut b, &tag, &lay, nb)?;
        b.label(&format!(".L{tag}_skip{nb}"))?;
    }
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

fn prologue_rows(b: &mut Builder, spec: Spec, lay: &Lay, tag: &str) -> R {
    kernarg_loads(b, spec.x4)?;
    // Lane offsets (the object's `lane*6`, the half-header select, `lane*32`).
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{HALF}, 4, v{LANE}"),
        &[HALF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HALF}, 2, v{HALF}"),
        &[HALF],
        &[HALF],
        &[],
    )?;
    valu(
        b,
        &format!("v_mul_u32_u24_e32 v{PKB}, 6, v{LANE}"),
        &[PKB],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{XOFF}, 5, v{LANE}"),
        &[XOFF],
        &[LANE],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s{SW}, ttmp9"), &[SW], &[])?;
    b.wait_all()?;
    if spec.x4 {
        // Cumulative workgroup counts: one workgroup per output row.
        sop(b, format!("s_max_i32 s{CP0}, s{M0}, 0"), &[CP0], &[M0])?;
        for (dst, prev, m) in [(CP1, CP0, M1), (CP2, CP1, M2), (CP3, CP2, M3)] {
            sop(b, format!("s_max_i32 s{TMP}, s{m}, 0"), &[TMP], &[m])?;
            sop(
                b,
                format!("s_add_co_i32 s{dst}, s{prev}, s{TMP}"),
                &[dst],
                &[prev, TMP],
            )?;
        }
        sop(b, format!("s_cmp_ge_u32 s{SW}, s{CP3}"), &[], &[SW, CP3])?;
    } else {
        sop(b, format!("s_cmp_ge_i32 s{SW}, s{SM}"), &[], &[SW, SM])?;
    }
    exit_unless(b, tag, "w")?;
    // `C` outside the entry's range leaves before any load or store.
    if spec.cap == 1 {
        sop(b, format!("s_cmp_lg_u32 s{SC}, 1"), &[], &[SC])?;
        exit_unless(b, tag, "c")?;
    } else {
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        exit_unless(b, tag, "c0")?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, {}", spec.cap), &[], &[SC])?;
        exit_unless(b, tag, "c1")?;
    }
    if spec.x4 {
        route(b)?;
        sop(
            b,
            format!("s_sub_co_i32 s{ROW}, s{SW}, s{SBASE}"),
            &[ROW],
            &[SW, SBASE],
        )?;
    } else {
        sop(b, format!("s_mov_b32 s{ROW}, s{SW}"), &[ROW], &[SW])?;
    }
    weight_resource(b)?;
    // Per-column x pointers `x + c*K*4` and the y byte offsets.
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    for c in 0..lay.cap {
        let p = xcol(c);
        sop(b, format!("s_mul_i32 s{TMP}, s{KB}, {c}"), &[TMP], &[KB])?;
        sop(
            b,
            format!("s_add_co_u32 s{p}, s{XP}, s{TMP}"),
            &[p],
            &[XP, TMP],
        )?;
        sop(
            b,
            format!("s_add_co_ci_u32 s{}, s{}, 0", p + 1, XP + 1),
            &[p + 1],
            &[XP + 1],
        )?;
    }
    sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
    sop(b, format!("s_lshl_b32 s{ROWB}, s{ROW}, 2"), &[ROWB], &[ROW])?;
    sop(b, format!("s_mov_b32 s{GOFF}, 0"), &[GOFF], &[])
}

/// The complete kernel body for `nb` live rows (ends in `s_endpgm`).
fn section(b: &mut Builder, tag: &str, lay: &Lay, nb: u8) -> R {
    for c in 0..nb {
        for st in 0..4u8 {
            let a = lay.acc(c, st);
            valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
        }
    }
    sop(
        b,
        format!("s_lshr_b32 s{QUADS}, s{GROUPS}, 2"),
        &[QUADS],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("s_and_b32 s{TAIL}, s{GROUPS}, 3"),
        &[TAIL],
        &[GROUPS],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{QUADS}, 0"), &[], &[QUADS])?;
    branch(b, &format!("s_cbranch_scc1 .L{tag}_b{nb}_tail"))?;
    b.loop_(&format!(".L{tag}_b{nb}_quad"), |b| {
        body(b, lay, nb, &[0, 1, 2, 3])?;
        sop(
            b,
            format!("s_add_co_i32 s{GOFF}, s{GOFF}, {}", lit(4 * GROUP)),
            &[GOFF],
            &[GOFF],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{XOFF}, 0x1000, v{XOFF}"),
            &[XOFF],
            &[XOFF],
            &[],
        )?;
        sop(
            b,
            format!("s_add_co_i32 s{QUADS}, s{QUADS}, -1"),
            &[QUADS],
            &[QUADS],
        )?;
        sop(b, format!("s_cmp_lg_u32 s{QUADS}, 0"), &[], &[QUADS])?;
        branch(b, &format!("s_cbranch_scc1 .L{tag}_b{nb}_quad"))
    })?;
    b.label(&format!(".L{tag}_b{nb}_tail"))?;
    // Tail group t (at the post-quad bases) accumulates into stream t.
    for t in 0..3u8 {
        sop(b, format!("s_cmp_gt_u32 s{TAIL}, {t}"), &[], &[TAIL])?;
        branch(b, &format!("s_cbranch_scc0 .L{tag}_b{nb}_fold"))?;
        body(b, lay, nb, &[t])?;
    }
    b.label(&format!(".L{tag}_b{nb}_fold"))?;
    fold(b, lay, nb)?;
    let values: Vec<(u8, u8)> = (0..nb)
        .map(|c| (lay.acc(c, 0), r8(W0 + u16::from(c))))
        .collect();
    lane_tree(b, LANE, SHUF, &values)?;
    store(b, lay, nb)
}

/// Groups `js` (slot = stream = immediate index) against every live row:
/// the weights of all groups issue first, X streams through the ring as
/// earlier items retire, and the body drains before it returns.
fn body(b: &mut Builder, lay: &Lay, nb: u8, js: &[u8]) -> R {
    let items = plan_items(nb, js);
    for (idx, &j) in js.iter().enumerate() {
        weight_loads(b, idx, j)?;
    }
    let mut issued = 0usize;
    while issued < usize::from(lay.slots).min(items.len()) {
        x_load(b, lay, &items[issued], issued)?;
        issued += 1;
    }
    let mut dequantized = 0usize;
    for (k, it) in items.iter().enumerate() {
        if it.idx == dequantized {
            let (h, p, u) = (raw(it.idx, 0), raw(it.idx, 1), raw(it.idx, 2));
            dequant_group(b, h, p, u, r8(W0))?;
            dequantized += 1;
        }
        consume(b, lay, it, k)?;
        // Item k's slot is free: the item one ring turn later takes it.
        if issued < items.len() {
            x_load(b, lay, &items[issued], issued)?;
            issued += 1;
        }
    }
    b.wait_all()
}

/// Payload dword, payload u16 and lane-selected half header of group slot `j`.
fn weight_loads(b: &mut Builder, idx: usize, j: u8) -> R {
    let off = u32::from(j) * GROUP;
    let mut scalar: Vec<u8> = (SRD..SRD + 4).collect();
    scalar.push(GOFF);
    let (h, p, u) = (raw(idx, 0), raw(idx, 1), raw(idx, 2));
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{p}, v{PKB}, s[{SRD}:{}], s{GOFF} offen offset:{}{POLICY}",
            SRD + 3,
            off + 8
        ),
        &[p],
        &[PKB],
        &scalar,
        false,
    )?;
    vmem(
        b,
        &format!(
            "buffer_load_u16 v{u}, v{PKB}, s[{SRD}:{}], s{GOFF} offen offset:{}{POLICY}",
            SRD + 3,
            off + 12
        ),
        &[u],
        &[PKB],
        &scalar,
        false,
    )?;
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{h}, v{HALF}, s[{SRD}:{}], s{GOFF} offen{}{POLICY}",
            SRD + 3,
            offset(off)
        ),
        &[h],
        &[HALF],
        &scalar,
        false,
    )
}

/// The eight activations (two b128) of each row of item `k` into its slot.
fn x_load(b: &mut Builder, lay: &Lay, it: &Item, k: usize) -> R {
    let a = lay.slot(k);
    let rows = [(it.c0, a)].into_iter().chain(it.c1.map(|c| (c, a + 9)));
    for (c, d0) in rows {
        let p = xcol(c);
        for half in 0..2u8 {
            let off = u32::from(it.j) * 1024 + u32::from(half) * 16;
            let d = d0 + 4 * half;
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{d}:{}], v{XOFF}, s[{p}:{}]{}",
                    d + 3,
                    p + 1,
                    offset(off)
                ),
                &[d, d + 1, d + 2, d + 3],
                &[XOFF],
                &[p, p + 1],
                false,
            )?;
        }
    }
    Ok(())
}

/// The group dot of one row (or a row pair in `v_dual_*` packets) and its
/// separate stream add `a_s = a_s + d`. Terms: x1 mul, then x0, x2..x7 fmac,
/// activation src0, weight src1 (as the object).
fn consume(b: &mut Builder, lay: &Lay, it: &Item, k: usize) -> R {
    let xa = lay.slot(k);
    match it.c1 {
        Some(c1) => {
            let xb = xa + 9;
            for (t, n) in TERMS.into_iter().enumerate() {
                let kind = if t == 0 { VopdF32::Mul } else { VopdF32::Fmac };
                let w = r8(W0) + n;
                b.vopd(
                    VopdOp {
                        op: kind,
                        dst: DOT0,
                        src0: Operand::V(xa + n),
                        src1: w,
                    },
                    VopdOp {
                        op: kind,
                        dst: DOT1,
                        src0: Operand::V(xb + n),
                        src1: w,
                    },
                )?;
            }
            let (a0, a1) = (lay.acc(it.c0, it.j), lay.acc(c1, it.j));
            b.vopd(
                VopdOp {
                    op: VopdF32::Add,
                    dst: a0,
                    src0: Operand::V(a0),
                    src1: DOT0,
                },
                VopdOp {
                    op: VopdF32::Add,
                    dst: a1,
                    src0: Operand::V(a1),
                    src1: DOT1,
                },
            )
        }
        None => {
            for (t, n) in TERMS.into_iter().enumerate() {
                let (x, w) = (xa + n, r8(W0) + n);
                if t == 0 {
                    valu(
                        b,
                        &format!("v_mul_f32_e32 v{DOT0}, v{x}, v{w}"),
                        &[DOT0],
                        &[x, w],
                        &[],
                    )?;
                } else {
                    valu(
                        b,
                        &format!("v_fmac_f32_e32 v{DOT0}, v{x}, v{w}"),
                        &[DOT0],
                        &[DOT0, x, w],
                        &[],
                    )?;
                }
            }
            let a = lay.acc(it.c0, it.j);
            valu(
                b,
                &format!("v_add_f32_e32 v{a}, v{a}, v{DOT0}"),
                &[a],
                &[a, DOT0],
                &[],
            )
        }
    }
}

/// `(a0 + a1) + (a3 + a2)` per live row, into stream 0.
fn fold(b: &mut Builder, lay: &Lay, nb: u8) -> R {
    for (dst, other) in [(0u8, 1u8), (3, 2)] {
        for c in 0..nb {
            let (x, y) = (lay.acc(c, dst), lay.acc(c, other));
            valu(
                b,
                &format!("v_add_f32_e32 v{x}, v{x}, v{y}"),
                &[x],
                &[x, y],
                &[],
            )?;
        }
    }
    for c in 0..nb {
        let (x, y) = (lay.acc(c, 0), lay.acc(c, 3));
        valu(
            b,
            &format!("v_add_f32_e32 v{x}, v{x}, v{y}"),
            &[x],
            &[x, y],
            &[],
        )?;
    }
    Ok(())
}

/// Lane 0 stores `y[c*M + row]` for every live row.
fn store(b: &mut Builder, lay: &Lay, nb: u8) -> R {
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    for c in 0..nb {
        let a = r8(RAW0 + u16::from(c));
        if c == 0 {
            valu(
                b,
                &format!("v_mov_b32_e32 v{a}, s{ROWB}"),
                &[a],
                &[],
                &[ROWB],
            )?;
        } else {
            sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {c}"), &[STMP], &[MB])?;
            sop(
                b,
                format!("s_add_co_i32 s{STMP}, s{STMP}, s{ROWB}"),
                &[STMP],
                &[STMP, ROWB],
            )?;
            valu(
                b,
                &format!("v_mov_b32_e32 v{a}, s{STMP}"),
                &[a],
                &[],
                &[STMP],
            )?;
        }
    }
    for c in 0..nb {
        let (a, val) = (r8(RAW0 + u16::from(c)), lay.acc(c, 0));
        vmem(
            b,
            &format!("global_store_b32 v{a}, v{val}, s[{SY}:{}]", SY + 1),
            &[],
            &[a, val],
            &[SY, SY + 1],
            true,
        )?;
    }
    b.wait_all()?;
    branch(b, "s_endpgm")
}

// --------------------------------------------------------------------- S2
/// `wg8`: block 256, eight output waves sharing one LDS X tile per group.
pub mod wg8 {
    use super::*;
    use crate::kernels::common::{s, v};
    use crate::lds::Transition;
    use crate::ledger::Counter;

    // VGPRs.
    const TID: u8 = 0;
    const XOFF: u8 = 3;
    const SHUF: u8 = 4;
    const LANE: u8 = 5;
    const HDR: u8 = 6;
    const PACK: u8 = 7;
    const UW: u8 = 8;
    const DOT: u8 = 9;
    /// LDS producer store address `(wave & (cap-1))*1024 + 4*lane` and consumer read base `4*lane`.
    const LDST: u8 = 10;
    const LDC: u8 = 11;
    const VT: u8 = 12;
    const HALF: u8 = 13;
    const PKB: u8 = 14;
    /// Dequantized weights W0..W7.
    const WS: u8 = 16;
    /// X slots (producer staging, consumer row vector) and the epilogue scratch / store addresses.
    const XS: u8 = 24;
    const ACC0: u16 = 40;
    // SGPRs (the shared map plus the wave-specific ones).
    const GI: u8 = 47;
    const WAVE: u8 = 53;
    const XT: u8 = 54;
    const SVAL: u8 = 55;
    const XG: u8 = 56;
    const SGPRS: u16 = 57;
    // Only the rowtile entry allocates this extra SGPR. Narrow resource and
    // instruction streams retain the carried S2 layout.
    const R0: u8 = 57;

    fn high(cap: u8) -> u16 {
        ACC0 + 4 * u16::from(cap)
    }
    fn acc(row: u32, st: u32) -> u8 {
        r8(ACC0 + 4 * row as u16 + st as u16)
    }
    fn lbl(tag: &str, name: &str) -> String {
        format!(".L{tag}_{name}")
    }
    fn glbl(tag: &str, st: u32, name: &str) -> String {
        format!(".L{tag}_s{st}_{name}")
    }

    pub fn emit(spec: Spec) -> Result<Emitted, String> {
        emit_impl(spec, false)
    }

    pub(super) fn emit_impl(spec: Spec, rowtiles: bool) -> Result<Emitted, String> {
        spec.validate()?;
        if !spec.wg8 {
            return Err("wg8::emit builds the wg8 entries only".into());
        }
        let tag = if rowtiles {
            format!("{}_t8", spec.tag())
        } else {
            spec.tag()
        };
        let mut b = Builder::new(
            KernelSpec {
                kernel_id: "lane_rows_mq6_gfx1201".into(),
                variant: if rowtiles {
                    format!("{}_t8", spec.variant())
                } else {
                    spec.variant()
                },
                arch: Arch::Gfx1201,
                symbol: if rowtiles {
                    if spec.x4 {
                        "gemv_mq6g256v2_rows_x4_t8_pm_gfx1201".into()
                    } else {
                        "gemv_mq6g256v2_rows_t8_pm_gfx1201".into()
                    }
                } else {
                    spec.symbol()
                },
                kernargs: spec.kernargs(),
                user_sgpr_count: 2,
                system_sgpr_workgroup_id_y: rowtiles,
                workgroup_size: spec.block(),
                group_segment_fixed_size: spec.lds(),
                wave32: true,
                cu_mode: false,
            },
            plan(
                high(spec.cap),
                if rowtiles { u16::from(R0) + 1 } else { SGPRS },
            )?,
        );
        b.enable_delay_alu();
        let slot = b.lds_slot("x_tile", 0, spec.lds())?;
        prologue(&mut b, spec, &tag, rowtiles)?;
        for row in 0..u32::from(spec.cap) {
            for st in 0..4 {
                valu(
                    &mut b,
                    &format!("v_mov_b32_e32 v{}, 0", acc(row, st)),
                    &[acc(row, st)],
                    &[],
                    &[],
                )?;
            }
        }
        sop(&mut b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
        sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(&tag, "fold")))?;
        b.loop_(&lbl(&tag, "group"), |b| {
            for st in 0..4u32 {
                if st > 0 {
                    sop(
                        b,
                        format!("s_cmp_ge_u32 s{GI}, s{GROUPS}"),
                        &[],
                        &[GI, GROUPS],
                    )?;
                    branch(b, &format!("s_cbranch_scc1 {}", lbl(&tag, "fold")))?;
                }
                group(b, spec, &tag, slot, st)?;
                sop(b, format!("s_add_co_i32 s{GI}, s{GI}, 1"), &[GI], &[GI])?;
            }
            sop(
                b,
                format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                &[],
                &[GI, GROUPS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {}", lbl(&tag, "group")))
        })?;
        b.label(&lbl(&tag, "fold"))?;
        // Waves past M never reach another barrier: they leave here.
        sop(&mut b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(&tag, "end")))?;
        if rowtiles {
            epilogue_rowtiles(&mut b, spec, &tag)?;
        } else {
            epilogue(&mut b, spec, &tag)?;
        }
        b.label(&lbl(&tag, "end"))?;
        branch(&mut b, "s_endpgm")?;
        b.finish()
    }

    /// `dst = prev + ceil(max(m, 0) / 8)` (`prev` absent: `dst = ceil(...)`).
    fn blocks(b: &mut Builder, dst: u8, prev: Option<u8>, m: u8) -> R {
        let t = if prev.is_some() { TMP } else { dst };
        sop(b, format!("s_max_i32 s{t}, s{m}, 0"), &[t], &[m])?;
        sop(b, format!("s_add_co_i32 s{t}, s{t}, 7"), &[t], &[t])?;
        sop(b, format!("s_lshr_b32 s{t}, s{t}, 3"), &[t], &[t])?;
        if let Some(p) = prev {
            sop(
                b,
                format!("s_add_co_i32 s{dst}, s{p}, s{t}"),
                &[dst],
                &[p, t],
            )?;
        }
        Ok(())
    }

    fn prologue(b: &mut Builder, spec: Spec, tag: &str, rowtiles: bool) -> R {
        kernarg_loads(b, spec.x4)?;
        // Entry block (before the first branch): the LDS address vectors and the
        // wave id derive from the work-item id alone. LDS word of (row v,
        // element j, lane l) is (v*8 + j)*32 + l: the producer stores at
        // v*1024 + 4l (+128 j), the consumer reads rows at 4l + 1024 r + 128 j.
        // The producer's row index is masked to the capacity so that waves
        // that never produce (`wave >= C`) still carry an in-bounds address.
        valu(
            b,
            &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
            &[LANE],
            &[TID],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
            &[LDC],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
            &[VT],
            &[TID],
            &[],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{WAVE}, v{VT}"),
            &[s(WAVE)],
            &[v(VT)],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{VT}, {}, v{VT}", spec.cap - 1),
            &[VT],
            &[VT],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{VT}, 10, v{VT}"),
            &[VT],
            &[VT],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDST}, v{VT}, v{LDC}"),
            &[LDST],
            &[VT, LDC],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{HALF}, 4, v{LANE}"),
            &[HALF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{HALF}, 2, v{HALF}"),
            &[HALF],
            &[HALF],
            &[],
        )?;
        valu(
            b,
            &format!("v_mul_u32_u24_e32 v{PKB}, 6, v{LANE}"),
            &[PKB],
            &[LANE],
            &[],
        )?;
        sop(b, format!("s_mov_b32 s{SW}, ttmp9"), &[SW], &[])?;
        b.wait_all()?;
        if spec.x4 {
            blocks(b, CP0, None, M0)?;
            blocks(b, CP1, Some(CP0), M1)?;
            blocks(b, CP2, Some(CP1), M2)?;
            blocks(b, CP3, Some(CP2), M3)?;
        } else {
            blocks(b, CP3, None, SM)?;
        }
        // Invalid logical domain and dead x/y workgroups exit uniformly before
        // any barrier. SC becomes the live physical tile size only afterward.
        sop(b, format!("s_cmp_ge_u32 s{SW}, s{CP3}"), &[], &[SW, CP3])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(tag, "end")))?;
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(tag, "end")))?;
        sop(
            b,
            format!(
                "s_cmp_gt_u32 s{SC}, {}",
                if rowtiles { 64 } else { spec.cap }
            ),
            &[],
            &[SC],
        )?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(tag, "end")))?;
        if rowtiles {
            // gfx1201 packs the y workgroup id in ttmp7[15:0].
            sop(b, format!("s_and_b32 s{R0}, ttmp7, 0xffff"), &[R0], &[])?;
            sop(b, format!("s_lshl_b32 s{R0}, s{R0}, 3"), &[R0], &[R0])?;
            sop(b, format!("s_cmp_ge_u32 s{R0}, s{SC}"), &[], &[R0, SC])?;
            branch(b, &format!("s_cbranch_scc1 {}", lbl(tag, "end")))?;
            sop(
                b,
                format!("s_sub_co_i32 s{SC}, s{SC}, s{R0}"),
                &[SC],
                &[SC, R0],
            )?;
            sop(b, format!("s_min_u32 s{SC}, s{SC}, 8"), &[SC], &[SC])?;
        }
        if spec.x4 {
            route(b)?;
        } else {
            sop(b, format!("s_mov_b32 s{SBASE}, 0"), &[SBASE], &[])?;
        }
        // This wave's matrix-local output m = 8*(w - base) + wave, and whether it exists.
        sop(
            b,
            format!("s_sub_co_i32 s{ROW}, s{SW}, s{SBASE}"),
            &[ROW],
            &[SW, SBASE],
        )?;
        sop(b, format!("s_lshl_b32 s{ROW}, s{ROW}, 3"), &[ROW], &[ROW])?;
        sop(
            b,
            format!("s_add_co_i32 s{ROW}, s{ROW}, s{WAVE}"),
            &[ROW],
            &[ROW, WAVE],
        )?;
        sop(b, format!("s_cmp_lt_i32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
        sop(b, format!("s_cselect_b32 s{SVAL}, 1, 0"), &[SVAL], &[])?;
        weight_resource(b)?;
        sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
        sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
        sop(b, format!("s_lshl_b32 s{ROWB}, s{ROW}, 2"), &[ROWB], &[ROW])?;
        if rowtiles {
            // Preserve full-window bases and original per-matrix row strides.
            sop(
                b,
                format!("s_mul_i32 s{STMP}, s{R0}, s{MB}"),
                &[STMP],
                &[R0, MB],
            )?;
            sop(
                b,
                format!("s_add_co_i32 s{ROWB}, s{ROWB}, s{STMP}"),
                &[ROWB],
                &[ROWB, STMP],
            )?;
        }
        // Producer X row offset of this wave: wave * K * 4.
        sop(
            b,
            format!("s_mul_i32 s{XT}, s{WAVE}, s{KB}"),
            &[XT],
            &[WAVE, KB],
        )?;
        if rowtiles {
            sop(
                b,
                format!("s_mul_i32 s{STMP}, s{R0}, s{KB}"),
                &[STMP],
                &[R0, KB],
            )?;
            sop(
                b,
                format!("s_add_co_i32 s{XT}, s{XT}, s{STMP}"),
                &[XT],
                &[XT, STMP],
            )?;
        }
        Ok(())
    }

    /// One K group with the compile-time stream `st` (`g % 4`).
    fn group(b: &mut Builder, spec: Spec, tag: &str, slot: usize, st: u32) -> R {
        sop(
            b,
            format!("s_mul_i32 s{GOFF}, s{GI}, {}", lit(GROUP)),
            &[GOFF],
            &[GI],
        )?;
        sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 10"), &[XG], &[GI])?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{XOFF}, 5, v{LANE}"),
            &[XOFF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{XOFF}, s{XG}, v{XOFF}"),
            &[XOFF],
            &[XOFF],
            &[XG],
        )?;
        // Valid waves: the weight group, loaded and dequantized ONCE.
        let wdone = glbl(tag, st, "wdone");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
        weight_loads(b)?;
        b.wait(Counter::Load, 0)?;
        dequant_group(b, HDR, PACK, UW, WS)?;
        b.label(&wdone)?;
        tile(b, spec, tag, slot, st)
    }

    /// Header dword (lane-selected half), payload dword at `8 + 6*lane` and the
    /// zero-extended u16 at `12 + 6*lane`, group base in the soffset.
    fn weight_loads(b: &mut Builder) -> R {
        let mut scalar: Vec<u8> = (SRD..SRD + 4).collect();
        scalar.push(GOFF);
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{PACK}, v{PKB}, s[{SRD}:{}], s{GOFF} offen offset:8{POLICY}",
                SRD + 3
            ),
            &[PACK],
            &[PKB],
            &scalar,
            false,
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_u16 v{UW}, v{PKB}, s[{SRD}:{}], s{GOFF} offen offset:12{POLICY}",
                SRD + 3
            ),
            &[UW],
            &[PKB],
            &scalar,
            false,
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{HDR}, v{HALF}, s[{SRD}:{}], s{GOFF} offen{POLICY}",
                SRD + 3
            ),
            &[HDR],
            &[HALF],
            &scalar,
            false,
        )
    }

    /// The single row tile (`C <= 8`): fill, barrier, consume, barrier.
    fn tile(b: &mut Builder, spec: Spec, tag: &str, slot: usize, st: u32) -> R {
        // Producer wave `v` stages X row `v` when it is live.
        let nofill = glbl(tag, st, "nofill");
        sop(b, format!("s_cmp_ge_u32 s{WAVE}, s{SC}"), &[], &[WAVE, SC])?;
        branch(b, &format!("s_cbranch_scc1 {nofill}"))?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{VT}, s{XT}, v{XOFF}"),
            &[VT],
            &[XOFF],
            &[XT],
        )?;
        for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{}:{}], v{VT}, s[{XP}:{}]{suffix}",
                    XS + j,
                    XS + j + 3,
                    XP + 1
                ),
                &[XS + j, XS + j + 1, XS + j + 2, XS + j + 3],
                &[VT],
                &[XP, XP + 1],
                false,
            )?;
        }
        for j in 0..8u32 {
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{}{}", XS + j as u8, offset(128 * j)),
                    vec![],
                    vec![v(LDST), v(XS + j as u8)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.wait_all()?;
        b.label(&nofill)?;
        b.barrier(&[Transition::Ready(slot)])?;
        // Valid waves consume rows 0.. in ascending order while row < C.
        let cdone = glbl(tag, st, "cdone");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
        for r in 0..u32::from(spec.cap) {
            if r > 0 {
                sop(b, format!("s_cmp_le_u32 s{SC}, {r}"), &[], &[SC])?;
                branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
            }
            consume_row(b, slot, st, r, 1024 * r)?;
        }
        b.label(&cdone)?;
        b.barrier(&[Transition::Retire(slot)])
    }

    /// One row's group dot (x1 mul, then x0, x2..x7 `v_fmac`, activation src0,
    /// weight src1) and its separate stream add.
    fn consume_row(b: &mut Builder, slot: usize, st: u32, row: u32, row_off: u32) -> R {
        for n in TERMS {
            let x = XS + n;
            b.ds_load(
                slot,
                Instruction::new(
                    format!(
                        "ds_load_b32 v{x}, v{LDC}{}",
                        offset(row_off + 128 * u32::from(n))
                    ),
                    vec![v(x)],
                    vec![v(LDC)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for (term, n) in TERMS.into_iter().enumerate() {
            let (w, x) = (WS + n, XS + n);
            if term == 0 {
                valu(
                    b,
                    &format!("v_mul_f32_e32 v{DOT}, v{x}, v{w}"),
                    &[DOT],
                    &[x, w],
                    &[],
                )?;
            } else {
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{DOT}, v{x}, v{w}"),
                    &[DOT],
                    &[DOT, x, w],
                    &[],
                )?;
            }
        }
        let a = acc(row, st);
        valu(
            b,
            &format!("v_add_f32_e32 v{a}, v{a}, v{DOT}"),
            &[a],
            &[a, DOT],
            &[],
        )
    }

    fn epilogue(b: &mut Builder, spec: Spec, tag: &str) -> R {
        let cap = u32::from(spec.cap);
        // Fold (a0 + a1) + (a3 + a2) for every row (rows >= C fold zeros).
        for r in 0..cap {
            for (l, h) in [(0, 1), (3, 2), (0, 3)] {
                let (d, x) = (acc(r, l), acc(r, h));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
                    &[d],
                    &[d, x],
                    &[],
                )?;
            }
        }
        let values: Vec<(u8, u8)> = (0..cap).map(|r| (acc(r, 0), XS + r as u8)).collect();
        lane_tree(b, LANE, SHUF, &values)?;
        // Lane 0 stores y[r*M + m] for every r < C.
        valu(
            b,
            &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
            &[],
            &[LANE],
            &[],
        )?;
        for r in 0..cap {
            let a = XS + r as u8;
            if r == 0 {
                valu(
                    b,
                    &format!("v_mov_b32_e32 v{a}, s{ROWB}"),
                    &[a],
                    &[],
                    &[ROWB],
                )?;
            } else {
                sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {r}"), &[STMP], &[MB])?;
                sop(
                    b,
                    format!("s_add_co_i32 s{STMP}, s{STMP}, s{ROWB}"),
                    &[STMP],
                    &[STMP, ROWB],
                )?;
                valu(
                    b,
                    &format!("v_mov_b32_e32 v{a}, s{STMP}"),
                    &[a],
                    &[],
                    &[STMP],
                )?;
            }
        }
        let done = lbl(tag, "stored");
        for r in 0..cap {
            sop(b, format!("s_cmp_le_u32 s{SC}, {r}"), &[], &[SC])?;
            branch(b, &format!("s_cbranch_scc1 {done}"))?;
            let (a, val) = (XS + r as u8, acc(r, 0));
            vmem(
                b,
                &format!("global_store_b32 v{a}, v{val}, s[{SY}:{}]", SY + 1),
                &[],
                &[a, val],
                &[SY, SY + 1],
                true,
            )?;
        }
        b.label(&done)?;
        b.wait_all()
    }

    /// Guard before every row's fold/tree as well as its store: an inactive
    /// tail row performs no arithmetic and never reads a padded LDS row.
    fn epilogue_rowtiles(b: &mut Builder, spec: Spec, tag: &str) -> R {
        let folded = lbl(tag, "live_folded");
        for r in 0..u32::from(spec.cap) {
            sop(b, format!("s_cmp_le_u32 s{SC}, {r}"), &[], &[SC])?;
            branch(b, &format!("s_cbranch_scc1 {folded}"))?;
            for (l, h) in [(0, 1), (3, 2), (0, 3)] {
                let (d, x) = (acc(r, l), acc(r, h));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
                    &[d],
                    &[d, x],
                    &[],
                )?;
            }
            lane_tree(b, LANE, SHUF, &[(acc(r, 0), XS)])?;
        }
        b.label(&folded)?;
        valu(
            b,
            &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
            &[],
            &[LANE],
            &[],
        )?;
        let done = lbl(tag, "stored");
        for r in 0..u32::from(spec.cap) {
            sop(b, format!("s_cmp_le_u32 s{SC}, {r}"), &[], &[SC])?;
            branch(b, &format!("s_cbranch_scc1 {done}"))?;
            sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {r}"), &[STMP], &[MB])?;
            sop(
                b,
                format!("s_add_co_i32 s{STMP}, s{STMP}, s{ROWB}"),
                &[STMP],
                &[STMP, ROWB],
            )?;
            valu(
                b,
                &format!("v_mov_b32_e32 v{XS}, s{STMP}"),
                &[XS],
                &[],
                &[STMP],
            )?;
            let val = acc(r, 0);
            vmem(
                b,
                &format!("global_store_b32 v{XS}, v{val}, s[{SY}:{}]", SY + 1),
                &[],
                &[XS, val],
                &[SY, SY + 1],
                true,
            )?;
        }
        b.label(&done)?;
        b.wait_all()
    }
}

// ------------------------------------------------------------------ module
/// Build one export.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    if spec.wg8 {
        wg8::emit(spec)
    } else {
        emit_rows(spec)
    }
}

/// Build a fixed-eight-row S2 twin for total logical rows 1..=64.
///
/// The final existing row kernarg is total R; launch grid y is ceil(R/8).
/// The single/x4 x grid is ceil(M/8)/sum ceil(max(Mi,0)/8), respectively.
/// This builder deliberately does not add exports to the narrow module table:
/// image composition and registration are integration-owned.
pub fn emit_rowtiles(spec: Spec) -> Result<Emitted, String> {
    if spec.cap != 8 || !spec.wg8 {
        return Err("gfx1201 MQ6 rowtiles require cap=8 and wg8=true".into());
    }
    wg8::emit_impl(spec, true)
}

fn exported(e: &Emitted) -> Option<&str> {
    e.s_text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .map(str::trim)
}

/// Build every export and combine them into the one module source and proof.
/// The exports must be exactly [`SYMBOLS`] in order and built for exact gfx1201.
pub fn emit_module() -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let mut emitted = Vec::with_capacity(SPECS.len());
    for spec in SPECS {
        emitted.push(emit(spec).map_err(|e| format!("{MODULE}: {}: {e}", spec.symbol()))?);
    }
    let got: Vec<&str> = emitted.iter().map(|e| exported(e).unwrap_or("")).collect();
    if got != SYMBOLS {
        return Err(format!(
            "{MODULE}: builders export {got:?}, the frozen exports are {SYMBOLS:?}"
        ));
    }
    if let Some(e) = emitted.iter().find(|e| e.proof.arch != Arch::Gfx1201) {
        return Err(format!(
            "{MODULE}: {} built for {:?}, not exact gfx1201",
            exported(e).unwrap_or(""),
            e.proof.arch
        ));
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}
