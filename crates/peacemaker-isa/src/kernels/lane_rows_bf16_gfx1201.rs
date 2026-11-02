// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact gfx1201 (R9700, wave32) BF16 lane-row twins: module
//! `lane_rows_bf16_pm_gfx1201`: ten narrow entries and two fixed-eight row tiles.
//!
//! Contract (intended DAG, not yet a proven claim): every output `(row c, matrix
//! output n)` of every entry is to be bitwise the gfx1201 hipcc singleton's value
//! for activation row `c` alone. The DAG below is read from the actual objects
//! (`qwen4_gemv_bf16_xf32`, clang d6f6cb691863, `gemv_bf16_xf32_x4` and
//! `gemv_bf16_xf32_k4`; the single-matrix `gemv_bf16_xf32` object has the same
//! per-output DAG, so there is ONE ordinary recipe and ONE HC recipe).
//! The carried narrow implementation records assembly/M7 and ordinary lifted
//! singleton checks in bi-land; those are not new rowtile qualification receipts.
//! The CPU oracle compares each row to PM C1, an independent emitted DAG, and
//! the pinned actual singleton when executable. HC SALU/VALU equality and
//! division inputs outside the emulator domain remain hardware claims, not CPU
//! numerical certification of the HC singleton.
//!
//!  * `K % 8 == 0`: lane `l` owns the eight bf16 `g*256 + 8l + j` of group `g`
//!    (lanes with `8l + 8 > K - 256 g` are skipped, never zero padded). Widening
//!    is bitwise (`u32 << 16` for the even element, `u32 & 0xffff0000` for the odd
//!    one). Per group `a[j & 3] = fma(X_j, W_j, a[j & 3])`, `j = 0..7` ascending
//!    (`v_fmac_f32 acc, X, W`, chains start at `+0`, the first element is also
//!    a fused op, no separate product, no separate group add).
//!  * `K % 8 != 0` (ordinary entries): the incumbent's all-scalar path. Lane `l`
//!    owns `k = l + 32 t`, one chain `a0 = fma(X, W, a0)`; `a1..a3` stay `+0` and
//!    are still folded. Lanes with `l + 32 t >= K` are skipped.
//!  * Fold `(a3 + a2) + (a1 + a0)` (all three adds in this operand order), then
//!    the lane tree `ds_swizzle "1pppp"` (own + lane|16) followed by
//!    `ds_bpermute` +8, +4, +2, +1 with the out-of-range partner being the lane
//!    itself, every add own + other; lane 0 holds the sum.
//!  * HC k4: four K quarters of `part = K / 4` (the quarter body is the ordinary
//!    body on local `K = part`; the host guarantees `K % 32 == 0`, any other
//!    `K` returns before the first barrier and store), one per wave. The object
//!    folds the quarters on the scalar unit and runs its `hc_act` epilogue as
//!    SALU float ops (`s_add_f32`, `s_mul_f32`, `s_rndne_f32`, `s_fmamk_f32`,
//!    `s_sub_f32`, `s_cvt_i32_f32`, `s_cmp_{ngt,nlt}_f32`) plus `v_s_exp_f32`,
//!    `v_ldexp_f32` and the VALU division. By owner ruling the emitted body is
//!    ONE shared VALU mapping of that DAG, node for node, in the object's
//!    constants and operand order, identical in all six HC entries: the fold
//!    `(q0 + q1) + (q2 + q3)` as `v_add_f32`; BF16 round trip (integer bit ops
//!    on the f32 bits), `scale * x` as `v_mul_f32`, round trip; `x / (1 +
//!    exp(-x))` with `v_mul/v_xor/v_rndne/v_fma/v_sub/v_add/v_cvt_i32` for the
//!    range reduction, `v_exp_f32`, `v_ldexp_f32`, the two clamps as
//!    `v_cmp_{nlt,ngt}_f32` + `v_cndmask`, and the full `v_div_scale / v_rcp /
//!    v_fma / v_div_fmas / v_div_fixup` division; final round trip. With
//!    `hc_act == 0` the raw fold is stored. Equality of the SALU float unit and
//!    `v_s_exp_f32` with this VALU mapping on exceptional inputs (NaN payloads,
//!    denormals, overflow) is NOT CPU-proven; it is left to the H0 GPU run.
//!
//! Entries (module order): ordinary x4 `c1`, `xbatch_r4`, `xbatch_r8`; HC k4
//! `c1`, `xbatch_r4`, `xbatch_r8` (schedule S1: one output per wave / one
//! output per workgroup, X re-read from global per live row, weights loaded and
//! widened once per K tile and reused for every live row); then the S2
//! `wg8` entries (X staged once per K tile through LDS and shared by all
//! output waves): ordinary x4 `xbatch_r4_wg8`, `xbatch_r8_wg8` (block 256 = eight
//! wave32 output owners, grid `sum ceil(Mi/8)`, LDS `1024 * rows`), HC
//! `xbatch_r4_wg8`, `xbatch_r8_wg8` (block 256 = 2 outputs x 4 quarters, grid
//! `ceil(M/2)`, LDS 16384).
//!
//! ABI: ordinary 96 B `A0@0 A1@8 A2@16 A3@24 x@32 y0@40 y1@48 y2@56 y3@64 M0@72
//! M1@76 M2@80 M3@84 K@88 C@92` (`x` is `[C][K]`, `y_i` is `[C][M_i]`). HC
//! `W@0 X@8 Y@16 M@24 K@28 scale@32 hc_act@36 C@40` (logical 44 B, segment 48 B;
//! only the 44 logical bytes are read). A `C` outside the entry's range
//! (`c1`: 1; `r4`: 1..=4; `r8`: 1..=8) returns before any barrier and any store.
//! Inactive rows are neither read nor written; no zero padding arithmetic.
//!
//! `build_rowtiles(Entry { hc, wg8: false, cap: 8 })` keeps S1's eight-row
//! register/LDS capacity, interpreting C as total R in 1..=64. Grid y is
//! ceil(R/8), grid x remains `sum Mi` (ordinary) or `M` (HC). Workgroup y owns
//! rows `8*y..min(8*y+8,R)`, with full-window X/Y strides and no tail-row
//! loads, reductions, or stores. Dead grid-y tiles return before barriers.
//! HC retains four K quarters, the singleton fold, and the full exp/div DAG;
//! exceptional SALU/VALU, NaN payload, denormal and overflow parity requires
//! actual gfx1201 hardware. No shipping admission follows from this builder.
//!
//! Resources are sized per entry (VGPR high water, LDS): `c1` never inherits the
//! r8 reservation. No spills, no scratch, barriers only at uniform points
//! (wave-uniform branches on `K`, `C`, `M` and the workgroup id).
use super::iu4_gemm::ModuleProof;
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    add64, lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, sr, v,
};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// Module name; the embedded image is `kernels/lane_rows_bf16_pm_gfx1201.hxaco`.
pub const MODULE: &str = "lane_rows_bf16_pm_gfx1201";

/// The frozen exports, in module order.
pub const SYMBOLS: [&str; 10] = [
    "gemv_bf16_xf32_x4_c1_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r4_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r8_pm_gfx1201",
    "gemv_bf16_xf32_k4_c1_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r4_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r8_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_bf16_xf32_x4_xbatch_r8_wg8_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r4_wg8_pm_gfx1201",
    "gemv_bf16_xf32_k4_xbatch_r8_wg8_pm_gfx1201",
];

/// Fixed-eight S1 row tiles. The final kernarg is total logical R in 1..=64;
/// launch grid y is ceil(R/8), while grid x is `sum Mi` / `M`.
pub const ROWTILE_SYMBOLS: [&str; 2] = [
    "gemv_bf16_xf32_rows_x4_t8_pm_gfx1201",
    "gemv_bf16_xf32_k4_rows_t8_pm_gfx1201",
];

/// Kernarg segment bytes of each export (HC: 44 logical bytes padded to 48).
pub const KERNARG_BYTES: [u32; 10] = [96, 96, 96, 48, 48, 48, 96, 96, 48, 48];

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
const HC_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 4),
    (28, 4),
    (32, 4),
    (36, 4),
    (40, 4),
];

/// `(offset, size)` of every kernarg of each export, in [`SYMBOLS`] order.
pub const KERNARGS: [&[(u32, u32)]; 10] = [
    X4_ARGS, X4_ARGS, X4_ARGS, HC_ARGS, HC_ARGS, HC_ARGS, X4_ARGS, X4_ARGS, HC_ARGS, HC_ARGS,
];

/// Workgroup size (threads) of each export.
pub const WORKGROUP_SIZE: [u32; 10] = [32, 32, 32, 128, 128, 128, 256, 256, 256, 256];

/// Static LDS bytes of each export.
pub const LDS_BYTES: [u32; 10] = [0, 0, 0, 16, 64, 128, 4096, 8192, 16384, 16384];

/// Most live activation rows `C` of each export.
pub const MAX_ROWS: [u32; 10] = [1, 4, 8, 1, 4, 8, 4, 8, 4, 8];

/// VGPR high water of each export (the declared VGPR budget, checked against
/// the emitted `next_free_vgpr`).
pub const VGPR_HIGH_WATER: [u16; 10] = [36, 72, 120, 36, 72, 120, 48, 64, 48, 64];

/// Grid rule of each export (workgroups, `x` only).
pub const GRID_RULE: [&str; 10] = [
    "sum Mi",
    "sum Mi",
    "sum Mi",
    "M",
    "M",
    "M",
    "sum ceil(Mi/8)",
    "sum ceil(Mi/8)",
    "ceil(M/2)",
    "ceil(M/2)",
];

/// Ordinary per-output DAG identity (`gemv_bf16_xf32` and `gemv_bf16_xf32_x4` agree).
pub const RECIPE_X4: &str = "bf16.gfx1201.x4.singleton-dag.v1";
/// HC k4 per-output DAG identity.
pub const RECIPE_K4: &str = "bf16.gfx1201.k4.singleton-dag.v1";
/// Per-export recipe; equal strings mean the same per-output DAG.
pub const RECIPE: [&str; 10] = [
    RECIPE_X4, RECIPE_X4, RECIPE_X4, RECIPE_K4, RECIPE_K4, RECIPE_K4, RECIPE_X4, RECIPE_X4,
    RECIPE_K4, RECIPE_K4,
];

/// A singleton reference object of a recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceObject {
    pub recipe: &'static str,
    pub module: &'static str,
    pub symbol: &'static str,
    pub source: &'static str,
    pub toolchain: &'static str,
    /// sha256 of the hipcc object (`.hsaco`) the DAG was read from.
    pub object_sha256: &'static str,
}

const TOOLCHAIN: &str = "AMD clang 24.0.0git d6f6cb691863 --offload-arch=gfx1201 -O3";
const QWEN4_SRC: &str = "kernels/src/qwen4_gemv_bf16_xf32.hip";
/// Runtime hot-cache `qwen4_gemv_bf16_xf32.2360726eb8cfd46d.hsaco` (holds `gemv_bf16_xf32_x4`, `gemv_bf16_xf32_k4`).
const QWEN4_OBJECT_SHA256: &str =
    "a610695fa6df3aaaf2771d98416fcfe1e229948b8ccae868bf03eed001850325";
/// Runtime hot-cache `gemv_bf16_xf32` object (the shipped gfx1201 single-matrix kernel).
const SINGLE_OBJECT_SHA256: &str =
    "a94c1c43f95811c81213a42a4d8f4489fbbc3ec0639e4a5837b3298eaf219601";

/// The references: ordinary x4, HC k4, and the single-matrix `gemv_bf16_xf32`
/// (same per-output DAG as the x4 object, hence the same recipe).
pub const REFERENCE: [ReferenceObject; 3] = [
    ReferenceObject {
        recipe: RECIPE_X4,
        module: "qwen4_gemv_bf16_xf32",
        symbol: "gemv_bf16_xf32_x4",
        source: QWEN4_SRC,
        toolchain: TOOLCHAIN,
        object_sha256: QWEN4_OBJECT_SHA256,
    },
    ReferenceObject {
        recipe: RECIPE_K4,
        module: "qwen4_gemv_bf16_xf32",
        symbol: "gemv_bf16_xf32_k4",
        source: QWEN4_SRC,
        toolchain: TOOLCHAIN,
        object_sha256: QWEN4_OBJECT_SHA256,
    },
    ReferenceObject {
        recipe: RECIPE_X4,
        module: "gemv_bf16_xf32",
        symbol: "gemv_bf16_xf32",
        source: "kernels/src/gemv_bf16_xf32.hip",
        toolchain: TOOLCHAIN,
        object_sha256: SINGLE_OBJECT_SHA256,
    },
];

/// Committed image file name.
pub fn bundle_name() -> String {
    format!("{MODULE}.hxaco")
}

/// One export: family, schedule, row capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// HC k4 (else ordinary x4).
    pub hc: bool,
    /// S2: block 256, X shared through LDS.
    pub wg8: bool,
    /// Row capacity: 1, 4 or 8.
    pub cap: u8,
}

/// The ten entries, in [`SYMBOLS`] order.
pub const ENTRIES: [Entry; 10] = [
    Entry {
        hc: false,
        wg8: false,
        cap: 1,
    },
    Entry {
        hc: false,
        wg8: false,
        cap: 4,
    },
    Entry {
        hc: false,
        wg8: false,
        cap: 8,
    },
    Entry {
        hc: true,
        wg8: false,
        cap: 1,
    },
    Entry {
        hc: true,
        wg8: false,
        cap: 4,
    },
    Entry {
        hc: true,
        wg8: false,
        cap: 8,
    },
    Entry {
        hc: false,
        wg8: true,
        cap: 4,
    },
    Entry {
        hc: false,
        wg8: true,
        cap: 8,
    },
    Entry {
        hc: true,
        wg8: true,
        cap: 4,
    },
    Entry {
        hc: true,
        wg8: true,
        cap: 8,
    },
];

impl Entry {
    fn index(self) -> usize {
        ENTRIES
            .iter()
            .position(|e| *e == self)
            .expect("entry is in the table")
    }
    pub fn symbol(self) -> &'static str {
        SYMBOLS[self.index()]
    }
    fn tag(self) -> String {
        let fam = if self.hc { "k4" } else { "x4" };
        let cap = if self.cap == 1 {
            "c1".to_string()
        } else {
            format!("r{}", self.cap)
        };
        format!("{fam}{cap}{}", if self.wg8 { "w" } else { "" })
    }
    pub fn block(self) -> u16 {
        if self.wg8 {
            256
        } else if self.hc {
            128
        } else {
            32
        }
    }
    pub fn lds_bytes(self) -> u32 {
        LDS_BYTES[self.index()]
    }
    pub fn kernarg_bytes(self) -> u32 {
        KERNARG_BYTES[self.index()]
    }
    fn validate(self) -> R {
        if !matches!(self.cap, 1 | 4 | 8) || (self.cap == 1 && self.wg8) {
            return Err(format!("lane_rows_bf16: invalid entry {self:?}"));
        }
        Ok(())
    }
    /// First accumulator VGPR: S1 keeps one eight-VGPR X tile per live row.
    fn acc0(self) -> u16 {
        if self.wg8 {
            32
        } else {
            24 + 8 * u16::from(self.cap)
        }
    }
    /// High-water VGPR index + 1.
    pub fn vgpr_high_water(self) -> u16 {
        self.acc0() + 4 * u16::from(self.cap)
    }
}

/// Internal addressing mode; keep the public narrow entry and its resources
/// unchanged, including when narrow and tiled bodies share one emitted image.
#[derive(Clone, Copy)]
struct Body {
    entry: Entry,
    rowtiles: bool,
}

impl std::ops::Deref for Body {
    type Target = Entry;
    fn deref(&self) -> &Entry {
        &self.entry
    }
}

impl Body {
    fn symbol(self) -> &'static str {
        if self.rowtiles {
            ROWTILE_SYMBOLS[usize::from(self.hc)]
        } else {
            self.entry.symbol()
        }
    }

    fn tag(self) -> String {
        if self.rowtiles {
            format!("{}t8", if self.hc { "k4" } else { "x4" })
        } else {
            self.entry.tag()
        }
    }
}

// ----------------------------------------------------------------------- SGPRs
const KARG: u8 = 0;
/// Selected weight pointer (64-bit), advanced to this wave's row / quarter.
const WP: u8 = 2;
/// Activation pointer; selected output pointer.
const XP: u8 = 4;
const YP: u8 = 6;
/// Kernarg words 0..8 (ordinary: A0..A3; HC: W X Y M K).
const KA: u8 = 8;
/// Ordinary kernarg words 8..16 (`x y0 y1 y2`), 16..20 (`y3 M0 M1`), 20..24 (`M2 M3 K C`).
const KB_: u8 = 16;
const KC: u8 = 24;
const KD: u8 = 28;
const X4_Y: [u8; 4] = [18, 20, 22, 24];
const X4_M: [u8; 4] = [26, 27, 28, 29];
const X4_K: u8 = 30;
const X4_C: u8 = 31;
/// HC kernarg words: `M@14 K@15 scale@16 hc_act@17 C@18`.
const HC_M: u8 = 14;
const HC_K: u8 = 15;
const SCALE: u8 = 16;
const ACT: u8 = 17;
const HC_C: u8 = 18;
const SK: u8 = 32;
const SC: u8 = 33;
const SM: u8 = 34;
const SBASE: u8 = 35;
const CP0: u8 = 36;
const SW: u8 = 40;
const WAVE: u8 = 41;
const ROW: u8 = 42;
const SVAL: u8 = 43;
const GROUPS: u8 = 44;
const TMP: u8 = 45;
const TMP2: u8 = 46;
const EM: u8 = 47;
const KREM: u8 = 48;
const GOFF: u8 = 49;
const XG: u8 = 50;
const GI: u8 = 51;
const KB: u8 = 52;
const MB: u8 = 53;
const R0B: u8 = 54;
const STMP: u8 = 55;
const SPATH: u8 = 56;
const KB2: u8 = 57;
const SKL: u8 = 58;
const NT32: u8 = 59;
const QI: u8 = 60;
const MLOC: u8 = 61;
const XT: u8 = 62;
const YB: u8 = 63;
/// Per-row X pointer pairs (S1): row `r` at `XB + 2r`.
const XB: u8 = 64;
const PART: u8 = 88;
const PARTB: u8 = 89;
const K2B: u8 = 90;
const XWB: u8 = 91;
const MBASE: u8 = 92;
const IR: u8 = 93;
const IM: u8 = 94;
const WRL: u8 = 95;
const WRH: u8 = 96;
const QOFF: u8 = 97;
const COL: u8 = 98;
/// First full-window row of a fixed-eight tile; never an accumulator index.
const TILE_R0: u8 = 99;
const SGPR_BUDGET: u16 = 104;

// ----------------------------------------------------------------------- VGPRs
const TID: u8 = 0;
const HOFF: u8 = 1;
const XOFF: u8 = 2;
const SHUF: u8 = 3;
const LANE: u8 = 4;
const VT: u8 = 5;
const LDST: u8 = 6;
const LDC: u8 = 7;
const LDR: u8 = 8;
const VPART: u8 = 9;
const VYA: u8 = 10;
const VYD: u8 = 11;
/// HC item quarters (`ds_load_b128`).
const VITQ: u8 = 12;
/// HC epilogue: result and temporaries (the dead X tile region).
const T0: u8 = 24;
const T1: u8 = 25;
const T2: u8 = 26;
const T3: u8 = 27;
const T4: u8 = 28;
const T5: u8 = 29;
const VOUT: u8 = 30;
/// Widened weights W0..W7.
const WS: u8 = 16;
/// X tile of row 0 (S1) or the one shared staging / consumer tile (wg8); also
/// the reduction / store / epilogue scratch.
const XBUF: u8 = 24;

fn acc(e: Body, row: u32, st: u32) -> u8 {
    let n = usize::from(e.acc0()) + 4 * row as usize + st as usize;
    u8::try_from(n).expect("accumulator index validated by plan()")
}
/// S1: X tile of row `r`.
fn xrow(r: u32) -> u8 {
    XBUF + 8 * r as u8
}
fn xpair(r: u32) -> u8 {
    XB + 2 * r as u8
}

fn lbl(e: Body, name: &str) -> String {
    format!(".Lb16{}_{name}", e.tag())
}

fn plan(e: Body) -> Result<RegPlan, String> {
    let hw = e.vgpr_high_water();
    if hw > 256 {
        return Err(format!("{}: needs {hw} VGPRs (> 256)", e.symbol()));
    }
    if hw != VGPR_HIGH_WATER[e.index()] {
        return Err(format!(
            "{}: VGPR high water {hw} != frozen {}",
            e.symbol(),
            VGPR_HIGH_WATER[e.index()]
        ));
    }
    let mut regs = RegPlan::new(hw, SGPR_BUDGET)?;
    let mut sr_: Vec<(&str, u8, u8)> = vec![
        ("kernarg", KARG, 2),
        ("weight_ptr", WP, 2),
        ("x_ptr", XP, 2),
        ("y_ptr", YP, 2),
        ("kernarg_a", KA, 8),
        ("k", SK, 1),
        ("c", SC, 1),
        ("m_selected", SM, 1),
        ("workgroup", SW, 1),
        ("wave", WAVE, 1),
        ("row", ROW, 1),
        ("groups", GROUPS, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("exec_mask", EM, 1),
        ("k_remaining", KREM, 1),
        ("group_offset", GOFF, 1),
        ("x_group_offset", XG, 1),
        ("group_index", GI, 1),
        ("k_bytes", KB, 1),
        ("m_bytes", MB, 1),
        ("row_bytes", R0B, 1),
        ("store_temp", STMP, 1),
        ("weight_row_bytes", KB2, 1),
        ("k_local", SKL, 1),
        ("weight_row_lo", WRL, 1),
        ("weight_row_hi", WRH, 1),
    ];
    if e.rowtiles {
        sr_.push(("tile_row_start", TILE_R0, 1));
    }
    if e.hc {
        sr_.extend([
            ("hc_scale_act", SCALE, 2),
            ("hc_c", HC_C, 1),
            ("quarter", QI, 1),
            ("part", PART, 1),
            ("part_bytes", PARTB, 1),
            ("quarter_offset", QOFF, 1),
            ("column", COL, 1),
            ("item_row", IR, 1),
            ("y_bytes", YB, 1),
        ]);
        if e.wg8 {
            sr_.extend([
                ("column_valid", SVAL, 1),
                ("column_local", MLOC, 1),
                ("m_base", MBASE, 1),
                ("item_col", IM, 1),
                ("k2_bytes", K2B, 1),
                ("x_wave_base", XWB, 1),
            ]);
        }
    } else {
        sr_.extend([
            ("kernarg_b", KB_, 8),
            ("kernarg_c", KC, 4),
            ("kernarg_d", KD, 4),
            ("matrix_base", SBASE, 1),
            ("blocks_0", CP0, 1),
            ("blocks_01", CP0 + 1, 1),
            ("blocks_012", CP0 + 2, 1),
            ("blocks_0123", CP0 + 3, 1),
            ("scalar_domain", SPATH, 1),
            ("tiles32", NT32, 1),
        ]);
        if e.wg8 {
            sr_.extend([("column_valid", SVAL, 1), ("x_producer_offset", XT, 1)]);
        }
    }
    if !e.wg8 {
        // Per-row X pointer pairs of the live-row capacity.
        let mut left = 2 * usize::from(e.cap);
        let mut base = usize::from(XB);
        while left > 0 {
            let len = if left >= 8 { 8 } else { left };
            sr_.push((
                if base == usize::from(XB) {
                    "x_row_ptrs0"
                } else {
                    "x_row_ptrs1"
                },
                base as u8,
                len as u8,
            ));
            base += len;
            left -= len;
        }
    }
    for (name, base, len) in sr_ {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    let mut vr: Vec<(String, u8, u8)> = vec![
        ("tid".into(), TID, 1),
        ("weight_offset".into(), HOFF, 1),
        ("x_offset".into(), XOFF, 1),
        ("shuffle_addr".into(), SHUF, 1),
        ("lane".into(), LANE, 1),
        ("temp".into(), VT, 1),
        ("lds_store".into(), LDST, 1),
        ("lds_lane".into(), LDC, 1),
        ("lds_read".into(), LDR, 1),
        ("partial_addr".into(), VPART, 1),
        ("y_addr".into(), VYA, 1),
        ("y_data".into(), VYD, 1),
        ("item_q".into(), VITQ, 4),
        ("weights".into(), WS, 8),
    ];
    if e.wg8 {
        vr.push(("x_slots".into(), XBUF, 8));
    } else {
        for r in 0..u32::from(e.cap) {
            vr.push((format!("x_row{r}"), xrow(r), 8));
        }
    }
    for r in 0..u32::from(e.cap) {
        vr.push((format!("acc_row{r}"), acc(e, r, 0), 4));
    }
    for (name, base, len) in vr {
        regs.add_range(&name, Kind::V, base, len, Live::Whole)?;
    }
    Ok(regs)
}

// ------------------------------------------------------------- instruction glue
fn off(imm: u32) -> String {
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
fn s_sub(a: Arch) -> &'static str {
    if a.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}
fn exec_set(b: &mut Builder) -> R {
    op(b, format!("s_mov_b32 exec_lo, s{EM}"), &[], &[s(EM)])
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
fn exec_lane0(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}
/// A manual gfx12 dependency wait the builder's SGPR tracker does not model
/// (VCC as a producer/consumer).
fn alu_wait(b: &mut Builder, which: &str) -> R {
    b.push(Instruction::new(
        format!("s_wait_alu {which}"),
        vec![],
        vec![],
    ))
}
fn gpair(p: u8) -> String {
    format!("s[{p}:{}]", p + 1)
}

/// Wave id and lane of a block; the workgroup id (`ttmp9`).
fn wave_lane(b: &mut Builder) -> R {
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
        &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[LANE],
        &[TID],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s{SW}, ttmp9"), &[SW], &[])
}

/// bf16 -> f32 is bitwise: the even element `u << 16`, the odd `u & 0xffff0000`.
/// The four raw dwords sit in `WS..WS+3` (one 16-byte load); widen in place,
/// dword 3 first, so no dword is overwritten before it is read.
fn widen_weights(b: &mut Builder) -> R {
    let w = |i: u8| WS + i;
    for d in (0..4u8).rev() {
        valu(
            b,
            &format!("v_and_b32_e32 v{}, 0xffff0000, v{}", w(2 * d + 1), w(d)),
            &[w(2 * d + 1)],
            &[w(d)],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 16, v{}", w(2 * d), w(d)),
            &[w(2 * d)],
            &[w(d)],
            &[],
        )?;
    }
    Ok(())
}

/// Active-lane mask of the current tile from `SKL` (local K) and `GI`: the low
/// `KREM >> 3` lanes (K256 tile) or `KREM` lanes (K32 tile), all of them when
/// `KREM >= full`.
fn tile_mask(b: &mut Builder, shift: u32, full: u32) -> R {
    let a = b.spec.arch;
    sop(
        b,
        format!(
            "s_lshl_b32 s{KREM}, s{GI}, {}",
            if shift == 3 { 8 } else { 5 }
        ),
        &[KREM],
        &[GI],
    )?;
    sop(
        b,
        format!("{} s{KREM}, s{SKL}, s{KREM}", s_sub(a)),
        &[KREM],
        &[SKL, KREM],
    )?;
    if shift == 3 {
        sop(b, format!("s_lshr_b32 s{EM}, s{KREM}, 3"), &[EM], &[KREM])?;
    } else {
        sop(b, format!("s_mov_b32 s{EM}, s{KREM}"), &[EM], &[KREM])?;
    }
    sop(b, format!("s_lshl_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
    sop(b, format!("s_not_b32 s{EM}, s{EM}"), &[EM], &[EM])?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{KREM}, {}", lit(full)),
        &[],
        &[KREM],
    )?;
    sop(b, format!("s_cselect_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])
}

/// Per-lane byte offsets of the current K256 (`vector`) or K32 tile: X
/// (`lane*32` / `lane*4` + `XG`) and the weights (`lane*16` / `lane*2` + `GOFF`).
fn tile_offsets(b: &mut Builder, vector: bool) -> R {
    let (xs, hs) = if vector { (5, 4) } else { (2, 1) };
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{XOFF}, {xs}, v{LANE}"),
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
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HOFF}, {hs}, v{LANE}"),
        &[HOFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
        &[HOFF],
        &[HOFF],
        &[GOFF],
    )
}

/// `GOFF = GI << gs`, `XG = GI << xs` of the tile.
fn tile_bases(b: &mut Builder, gs: u32, xs: u32) -> R {
    sop(
        b,
        format!("s_lshl_b32 s{GOFF}, s{GI}, {gs}"),
        &[GOFF],
        &[GI],
    )?;
    sop(b, format!("s_lshl_b32 s{XG}, s{GI}, {xs}"), &[XG], &[GI])
}

/// Branch to `to` when the live row count is `<= r` (wave-uniform).
fn skip_dead_row(b: &mut Builder, r: u32, to: &str) -> R {
    sop(b, format!("s_cmp_le_u32 s{SC}, {r}"), &[], &[SC])?;
    branch(b, &format!("s_cbranch_scc1 {to}"))
}

/// Validate a live grid-y tile before any row load, barrier, or store. `raw`
/// is the validated total-row kernarg word: read here, never written. SC
/// receives its ONE definition in the kernel, the physical live-row count
/// `min(raw - TILE_R0, 8)`, dominating every `skip_dead_row` user outside any
/// loop. TILE_R0 remains the full-window base. The subtraction intermediate
/// lives in STMP, which holds no live value at this point (its first later
/// use redefines it). No new SGPR; same final values and addresses.
fn rowtile_domain(b: &mut Builder, e: Body, raw: u8) -> R {
    let a = b.spec.arch;
    let end = lbl(e, "end");
    sop(
        b,
        format!("s_and_b32 s{TILE_R0}, ttmp7, 0xffff"),
        &[TILE_R0],
        &[],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TILE_R0}, s{TILE_R0}, 3"),
        &[TILE_R0],
        &[TILE_R0],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{TILE_R0}, s{raw}"),
        &[],
        &[TILE_R0, raw],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(
        b,
        format!("{} s{STMP}, s{raw}, s{TILE_R0}", s_sub(a)),
        &[STMP],
        &[raw, TILE_R0],
    )?;
    sop(b, format!("s_min_u32 s{SC}, s{STMP}, 8"), &[SC], &[STMP])
}

// ------------------------------------------------------------------ S1 K tiles
/// One K256 group (vector domain), one output per wave, X of every live row
/// loaded from global: weights are loaded and widened once, then every live row
/// consumes them (`a[j&3] = fma(X_j, W_j, a[j&3])`, `j = 0..7` ascending).
fn s1_vector_group(b: &mut Builder, e: Body) -> R {
    tile_bases(b, 9, 10)?;
    tile_mask(b, 3, 256)?;
    tile_offsets(b, true)?;
    exec_set(b)?;
    let wr = |i: u8| WS + i;
    vmem(
        b,
        &format!(
            "global_load_b128 v[{WS}:{}], v{HOFF}, {}",
            WS + 3,
            gpair(WP)
        ),
        &[wr(0), wr(1), wr(2), wr(3)],
        &[HOFF],
        &[WP, WP + 1],
        false,
    )?;
    let xdone = lbl(e, "gx_done");
    let cdone = lbl(e, "gc_done");
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &xdone)?;
        }
        let x = xrow(r);
        let p = xpair(r);
        for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{}:{}], v{XOFF}, {}{suffix}",
                    x + j,
                    x + j + 3,
                    gpair(p)
                ),
                &[x + j, x + j + 1, x + j + 2, x + j + 3],
                &[XOFF],
                &[p, p + 1],
                false,
            )?;
        }
    }
    b.label(&xdone)?;
    b.wait_all()?;
    widen_weights(b)?;
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &cdone)?;
        }
        for j in 0..8u32 {
            let a = acc(e, r, j & 3);
            let x = xrow(r) + j as u8;
            valu(
                b,
                &format!("v_fmac_f32_e32 v{a}, v{x}, v{}", WS + j as u8),
                &[a],
                &[a, x, WS + j as u8],
                &[],
            )?;
        }
    }
    b.label(&cdone)?;
    exec_all(b)
}

/// One K32 tile (scalar domain, `K % 8 != 0`), one output per wave: one weight
/// u16 and one X f32 per lane and live row, one chain `a0`.
fn s1_scalar_step(b: &mut Builder, e: Body) -> R {
    tile_bases(b, 6, 7)?;
    tile_mask(b, 0, 32)?;
    tile_offsets(b, false)?;
    exec_set(b)?;
    vmem(
        b,
        &format!("global_load_u16 v{WS}, v{HOFF}, {}", gpair(WP)),
        &[WS],
        &[HOFF],
        &[WP, WP + 1],
        false,
    )?;
    let xdone = lbl(e, "sx_done");
    let cdone = lbl(e, "sc_done");
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &xdone)?;
        }
        let x = xrow(r);
        let p = xpair(r);
        vmem(
            b,
            &format!("global_load_b32 v{x}, v{XOFF}, {}", gpair(p)),
            &[x],
            &[XOFF],
            &[p, p + 1],
            false,
        )?;
    }
    b.label(&xdone)?;
    b.wait_all()?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{WS}, 16, v{WS}"),
        &[WS],
        &[WS],
        &[],
    )?;
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &cdone)?;
        }
        let a = acc(e, r, 0);
        let x = xrow(r);
        valu(
            b,
            &format!("v_fmac_f32_e32 v{a}, v{x}, v{WS}"),
            &[a],
            &[a, x, WS],
            &[],
        )?;
    }
    b.label(&cdone)?;
    exec_all(b)
}

// ------------------------------------------------------------- wg8 ordinary K tiles
/// One ordinary S2 K tile: weights once per valid wave, X rows staged once
/// into LDS by waves `0..C` (one row each), barrier, every valid wave consumes
/// all live rows, barrier. Invalid output waves still stage and hit both barriers.
fn wg8_x4_tile(b: &mut Builder, e: Body, slot: usize, vector: bool) -> R {
    if vector {
        tile_bases(b, 9, 10)?;
        tile_mask(b, 3, 256)?;
    } else {
        tile_bases(b, 6, 7)?;
        tile_mask(b, 0, 32)?;
    }
    tile_offsets(b, vector)?;
    let wdone = lbl(e, if vector { "vw_done" } else { "sw_done" });
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
    exec_set(b)?;
    if vector {
        let wr = |i: u8| WS + i;
        vmem(
            b,
            &format!(
                "global_load_b128 v[{WS}:{}], v{HOFF}, {}",
                WS + 3,
                gpair(WP)
            ),
            &[wr(0), wr(1), wr(2), wr(3)],
            &[HOFF],
            &[WP, WP + 1],
            false,
        )?;
        b.wait(Counter::Load, 0)?;
        widen_weights(b)?;
    } else {
        vmem(
            b,
            &format!("global_load_u16 v{WS}, v{HOFF}, {}", gpair(WP)),
            &[WS],
            &[HOFF],
            &[WP, WP + 1],
            false,
        )?;
        b.wait(Counter::Load, 0)?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{WS}, 16, v{WS}"),
            &[WS],
            &[WS],
            &[],
        )?;
    }
    exec_all(b)?;
    b.label(&wdone)?;
    // Producer: wave v stages row v when it is live.
    let nofill = lbl(e, if vector { "vnofill" } else { "snofill" });
    sop(b, format!("s_cmp_ge_u32 s{WAVE}, s{SC}"), &[], &[WAVE, SC])?;
    branch(b, &format!("s_cbranch_scc1 {nofill}"))?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{VT}, s{XT}, v{XOFF}"),
        &[VT],
        &[XOFF],
        &[XT],
    )?;
    exec_set(b)?;
    if vector {
        for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{}:{}], v{VT}, {}{suffix}",
                    XBUF + j,
                    XBUF + j + 3,
                    gpair(XP)
                ),
                &[XBUF + j, XBUF + j + 1, XBUF + j + 2, XBUF + j + 3],
                &[VT],
                &[XP, XP + 1],
                false,
            )?;
        }
        for j in 0..8u32 {
            let x = XBUF + j as u8;
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{x}{}", off(128 * j)),
                    vec![],
                    vec![v(LDST), v(x)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
    } else {
        vmem(
            b,
            &format!("global_load_b32 v{XBUF}, v{VT}, {}", gpair(XP)),
            &[XBUF],
            &[VT],
            &[XP, XP + 1],
            false,
        )?;
        b.ds_store(
            slot,
            Instruction::new(
                format!("ds_store_b32 v{LDST}, v{XBUF}"),
                vec![],
                vec![v(LDST), v(XBUF)],
            )
            .memory(MemoryClass::DsStore),
        )?;
    }
    b.wait_all()?;
    exec_all(b)?;
    b.label(&nofill)?;
    b.barrier(&[Transition::Ready(slot)])?;
    // Consumers: every valid wave, rows ascending while live.
    let cdone = lbl(e, if vector { "vcdone" } else { "scdone" });
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
    exec_set(b)?;
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &cdone)?;
        }
        consume_row(b, e, slot, vector, r, 1024 * r, LDC)?;
    }
    b.label(&cdone)?;
    exec_all(b)?;
    b.barrier(&[Transition::Retire(slot)])
}

/// One live row's chains advanced by one tile from LDS.
fn consume_row(
    b: &mut Builder,
    e: Body,
    slot: usize,
    vector: bool,
    row: u32,
    row_off: u32,
    base: u8,
) -> R {
    let n = if vector { 8u32 } else { 1 };
    for j in 0..n {
        let x = XBUF + j as u8;
        b.ds_load(
            slot,
            Instruction::new(
                format!("ds_load_b32 v{x}, v{base}{}", off(row_off + 128 * j)),
                vec![v(x)],
                vec![v(base)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    for j in 0..n {
        let a = acc(e, row, j & 3);
        let x = XBUF + j as u8;
        valu(
            b,
            &format!("v_fmac_f32_e32 v{a}, v{x}, v{}", WS + j as u8),
            &[a],
            &[a, x, WS + j as u8],
            &[],
        )?;
    }
    Ok(())
}

// ----------------------------------------------------------------- fold, tree
/// Per-row fold `(a3 + a2) + (a1 + a0)` and the lane tree of the first `cap`
/// rows; the value of row `r` is left in `acc(r, 3)` (lane 0 is the sum).
fn reduce_rows(b: &mut Builder, e: Body) -> R {
    let cap = u32::from(e.cap);
    let fold_done = lbl(e, "fold_rows_done");
    for r in 0..cap {
        if e.rowtiles && r > 0 {
            skip_dead_row(b, r, &fold_done)?;
        }
        for (d, x) in [(1, 0), (3, 2), (3, 1)] {
            let (d, x) = (acc(e, r, d), acc(e, r, x));
            valu(
                b,
                &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
                &[d],
                &[d, x],
                &[],
            )?;
        }
    }
    if e.rowtiles {
        b.label(&fold_done)?;
    }
    let tmp = |r: u32| XBUF + r as u8;
    let swizzle_done = lbl(e, "swizzle_rows_done");
    for r in 0..cap {
        if e.rowtiles && r > 0 {
            skip_dead_row(b, r, &swizzle_done)?;
        }
        let (val, t) = (acc(e, r, 3), tmp(r));
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                vec![v(t)],
                vec![v(val)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    if e.rowtiles {
        b.label(&swizzle_done)?;
        // The ledger counts all cap requests, but a ragged tile issued only SC.
        // A per-register wait of cap-r-1 therefore does not retire row r.
        // Drain at the join before reading any crosslane destination.
        b.wait_all()?;
    }
    let swizzle_add_done = lbl(e, "swizzle_add_rows_done");
    for r in 0..cap {
        if e.rowtiles && r > 0 {
            skip_dead_row(b, r, &swizzle_add_done)?;
        }
        let (val, t) = (acc(e, r, 3), tmp(r));
        valu(
            b,
            &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
            &[val],
            &[val, t],
            &[],
        )?;
    }
    if e.rowtiles {
        b.label(&swizzle_add_done)?;
    }
    for offs in [8u32, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - offs),
            &[],
            &[LANE],
            &[],
        )?;
        alu_wait(b, "depctr_va_vcc(0)")?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{SHUF}, 0, {offs}, vcc_lo"),
            &[SHUF],
            &[],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"),
            &[SHUF],
            &[SHUF, LANE],
            &[],
        )?;
        let permute_done = lbl(e, &format!("permute{offs}_rows_done"));
        for r in 0..cap {
            if e.rowtiles && r > 0 {
                skip_dead_row(b, r, &permute_done)?;
            }
            let (val, t) = (acc(e, r, 3), tmp(r));
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{t}, v{SHUF}, v{val}"),
                    vec![v(t)],
                    vec![v(SHUF), v(val)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        if e.rowtiles {
            b.label(&permute_done)?;
            // As above, skipped rows make static nonzero DS counts unsafe.
            // The last stage must also finish before store_rows reuses these
            // destinations as global-store addresses.
            b.wait_all()?;
        }
        let add_done = lbl(e, &format!("permute{offs}_add_rows_done"));
        for r in 0..cap {
            if e.rowtiles && r > 0 {
                skip_dead_row(b, r, &add_done)?;
            }
            let (val, t) = (acc(e, r, 3), tmp(r));
            valu(
                b,
                &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
                &[val],
                &[val, t],
                &[],
            )?;
        }
        if e.rowtiles {
            b.label(&add_done)?;
        }
    }
    Ok(())
}

/// Lane 0 stores `y[r*M + row]` of every live row (ordinary entries).
fn store_rows(b: &mut Builder, e: Body) -> R {
    let a = b.spec.arch;
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    let stored = lbl(e, "stored");
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &stored)?;
        }
        let t = XBUF + r as u8;
        sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {r}"), &[STMP], &[MB])?;
        sop(
            b,
            format!("{} s{STMP}, s{STMP}, s{R0B}", s_add_i32(a)),
            &[STMP],
            &[STMP, R0B],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{t}, s{STMP}"),
            &[t],
            &[],
            &[STMP],
        )?;
        vmem(
            b,
            &format!("global_store_b32 v{t}, v{}, {}", acc(e, r, 3), gpair(YP)),
            &[],
            &[t, acc(e, r, 3)],
            &[YP, YP + 1],
            true,
        )?;
    }
    b.label(&stored)?;
    b.wait_all()
}

fn zero_acc(b: &mut Builder, e: Body) -> R {
    let done = lbl(e, "zero_rows_done");
    for r in 0..u32::from(e.cap) {
        if e.rowtiles && r > 0 {
            skip_dead_row(b, r, &done)?;
        }
        for st in 0..4 {
            valu(
                b,
                &format!("v_mov_b32_e32 v{}, 0", acc(e, r, st)),
                &[acc(e, r, st)],
                &[],
                &[],
            )?;
        }
    }
    if e.rowtiles {
        b.label(&done)?;
    }
    Ok(())
}

fn new_builder(e: Body) -> Result<(Builder, Option<usize>), String> {
    e.validate()?;
    let kernargs = if e.hc {
        KernargLayout::new(48)
            .pointer_access("W", 0, Access::ReadOnly)
            .pointer_access("X", 8, Access::ReadOnly)
            .pointer_access("Y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("scale", 32, 4, "by_value")
            .hidden("hc_act", 36, 4, "by_value")
            .hidden("C", 40, 4, "by_value")
    } else {
        KernargLayout::new(96)
            .pointer_access("A0", 0, Access::ReadOnly)
            .pointer_access("A1", 8, Access::ReadOnly)
            .pointer_access("A2", 16, Access::ReadOnly)
            .pointer_access("A3", 24, Access::ReadOnly)
            .pointer_access("x", 32, Access::ReadOnly)
            .pointer_access("y0", 40, Access::WriteOnly)
            .pointer_access("y1", 48, Access::WriteOnly)
            .pointer_access("y2", 56, Access::WriteOnly)
            .pointer_access("y3", 64, Access::WriteOnly)
            .hidden("M0", 72, 4, "by_value")
            .hidden("M1", 76, 4, "by_value")
            .hidden("M2", 80, 4, "by_value")
            .hidden("M3", 84, 4, "by_value")
            .hidden("K", 88, 4, "by_value")
            .hidden("C", 92, 4, "by_value")
    };
    let kspec = KernelSpec {
        kernel_id: "lane_rows_bf16".into(),
        variant: format!("{}_{}", e.tag(), if e.wg8 { "wg8" } else { "s1" }),
        arch: Arch::Gfx1201,
        symbol: e.symbol().into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: e.rowtiles,
        workgroup_size: e.block(),
        group_segment_fixed_size: e.lds_bytes(),
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(e)?);
    b.enable_delay_alu();
    let slot = if e.lds_bytes() > 0 {
        Some(b.lds_slot("lds", 0, e.lds_bytes())?)
    } else {
        None
    };
    Ok((b, slot))
}

// ---------------------------------------------------------------- ordinary x4
/// Kernarg loads, matrix routing, K domain, weight and X pointers.
fn prologue_x4(b: &mut Builder, e: Body) -> R {
    let a = b.spec.arch;
    let end = lbl(e, "end");
    smem(b, KA, 8, KARG, 0)?;
    smem(b, KB_, 8, KARG, 32)?;
    smem(b, KC, 4, KARG, 64)?;
    smem(b, KD, 4, KARG, 80)?;
    wave_lane(b)?;
    b.wait_all()?;
    sop(b, format!("s_mov_b32 s{SK}, s{X4_K}"), &[SK], &[X4_K])?;
    if !e.rowtiles {
        sop(b, format!("s_mov_b32 s{SC}, s{X4_C}"), &[SC], &[X4_C])?;
    }
    op(
        b,
        format!("s_mov_b64 s[{XP}:{}], s[{KB_}:{}]", XP + 1, KB_ + 1),
        &[sr(XP, 2)],
        &[sr(KB_, 2)],
    )?;
    // Cumulative workgroup counts: Pi = Mi (S1) or ceil(Mi/8) (wg8).
    let cp = |i: u8| CP0 + i;
    if e.wg8 {
        sop(
            b,
            format!("{} s{}, s{}, 7", s_add_i32(a), cp(0), X4_M[0]),
            &[cp(0)],
            &[X4_M[0]],
        )?;
        sop(
            b,
            format!("s_lshr_b32 s{}, s{}, 3", cp(0), cp(0)),
            &[cp(0)],
            &[cp(0)],
        )?;
        for i in 1..4u8 {
            sop(
                b,
                format!("{} s{TMP}, s{}, 7", s_add_i32(a), X4_M[i as usize]),
                &[TMP],
                &[X4_M[i as usize]],
            )?;
            sop(b, format!("s_lshr_b32 s{TMP}, s{TMP}, 3"), &[TMP], &[TMP])?;
            sop(
                b,
                format!("{} s{}, s{}, s{TMP}", s_add_i32(a), cp(i), cp(i - 1)),
                &[cp(i)],
                &[cp(i - 1), TMP],
            )?;
        }
    } else {
        sop(
            b,
            format!("s_mov_b32 s{}, s{}", cp(0), X4_M[0]),
            &[cp(0)],
            &[X4_M[0]],
        )?;
        for i in 1..4u8 {
            let m = X4_M[i as usize];
            sop(
                b,
                format!("{} s{}, s{}, s{m}", s_add_i32(a), cp(i), cp(i - 1)),
                &[cp(i)],
                &[cp(i - 1), m],
            )?;
        }
    }
    // Workgroups past the sum return; so does a C outside 1..=cap (1..=64 raw
    // total for rowtiles; uniform, before the first barrier and any store).
    sop(
        b,
        format!("s_cmp_ge_u32 s{SW}, s{}", cp(3)),
        &[],
        &[SW, cp(3)],
    )?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    if e.rowtiles {
        sop(b, format!("s_cmp_eq_u32 s{X4_C}, 0"), &[], &[X4_C])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_gt_u32 s{X4_C}, 64"), &[], &[X4_C])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        rowtile_domain(b, e, X4_C)?;
    } else {
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, {}", e.cap), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
    }
    // First matrix i with w < P0+..+Pi (matrix 2 or 3, then 1, then 0).
    let wp = sr(WP, 2);
    let yp = sr(YP, 2);
    let pa = |i: u8| KA + 2 * i;
    sop(
        b,
        format!("s_cmp_lt_u32 s{SW}, s{}", cp(2)),
        &[],
        &[SW, cp(2)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{WP}:{}], s[{}:{}], s[{}:{}]",
            WP + 1,
            pa(2),
            pa(2) + 1,
            pa(3),
            pa(3) + 1
        ),
        &[wp],
        &[sr(pa(2), 2), sr(pa(3), 2)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{YP}:{}], s[{}:{}], s[{}:{}]",
            YP + 1,
            X4_Y[2],
            X4_Y[2] + 1,
            X4_Y[3],
            X4_Y[3] + 1
        ),
        &[yp],
        &[sr(X4_Y[2], 2), sr(X4_Y[3], 2)],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{}, s{}", X4_M[2], X4_M[3]),
        &[SM],
        &[X4_M[2], X4_M[3]],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{}, s{}", cp(1), cp(2)),
        &[SBASE],
        &[cp(1), cp(2)],
    )?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{SW}, s{}", cp(1)),
        &[],
        &[SW, cp(1)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{WP}:{}], s[{}:{}], s[{WP}:{}]",
            WP + 1,
            pa(1),
            pa(1) + 1,
            WP + 1
        ),
        &[wp],
        &[sr(pa(1), 2), wp],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{YP}:{}], s[{}:{}], s[{YP}:{}]",
            YP + 1,
            X4_Y[1],
            X4_Y[1] + 1,
            YP + 1
        ),
        &[yp],
        &[sr(X4_Y[1], 2), yp],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{}, s{SM}", X4_M[1]),
        &[SM],
        &[X4_M[1], SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, s{}, s{SBASE}", cp(0)),
        &[SBASE],
        &[cp(0), SBASE],
    )?;
    sop(
        b,
        format!("s_cmp_lt_u32 s{SW}, s{}", cp(0)),
        &[],
        &[SW, cp(0)],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{WP}:{}], s[{}:{}], s[{WP}:{}]",
            WP + 1,
            pa(0),
            pa(0) + 1,
            WP + 1
        ),
        &[wp],
        &[sr(pa(0), 2), wp],
    )?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{YP}:{}], s[{}:{}], s[{YP}:{}]",
            YP + 1,
            X4_Y[0],
            X4_Y[0] + 1,
            YP + 1
        ),
        &[yp],
        &[sr(X4_Y[0], 2), yp],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SM}, s{}, s{SM}", X4_M[0]),
        &[SM],
        &[X4_M[0], SM],
    )?;
    sop(
        b,
        format!("s_cselect_b32 s{SBASE}, 0, s{SBASE}"),
        &[SBASE],
        &[SBASE],
    )?;
    // This wave's matrix-local output row.
    sop(
        b,
        format!("{} s{ROW}, s{SW}, s{SBASE}", s_sub(a)),
        &[ROW],
        &[SW, SBASE],
    )?;
    if e.wg8 {
        sop(b, format!("s_lshl_b32 s{ROW}, s{ROW}, 3"), &[ROW], &[ROW])?;
        sop(
            b,
            format!("{} s{ROW}, s{ROW}, s{WAVE}", s_add_i32(a)),
            &[ROW],
            &[ROW, WAVE],
        )?;
        sop(b, format!("s_cmp_lt_i32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
        sop(b, format!("s_cselect_b32 s{SVAL}, 1, 0"), &[SVAL], &[])?;
    }
    // K domain: K256 groups for K % 8 == 0, K32 tiles for the scalar domain.
    sop(b, format!("s_and_b32 s{SPATH}, s{SK}, 7"), &[SPATH], &[SK])?;
    sop(b, format!("s_mov_b32 s{SKL}, s{SK}"), &[SKL], &[SK])?;
    sop(
        b,
        format!("{} s{GROUPS}, s{SK}, {}", s_add_i32(a), lit(255)),
        &[GROUPS],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{GROUPS}, 8"),
        &[GROUPS],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("{} s{NT32}, s{SK}, 31", s_add_i32(a)),
        &[NT32],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{NT32}, s{NT32}, 5"),
        &[NT32],
        &[NT32],
    )?;
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    sop(b, format!("s_lshl_b32 s{KB2}, s{SK}, 1"), &[KB2], &[SK])?;
    sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
    sop(b, format!("s_lshl_b32 s{R0B}, s{ROW}, 2"), &[R0B], &[ROW])?;
    // Weight pointer of this wave's row (64-bit ROW * 2K); a row past M is never loaded from.
    sop(
        b,
        format!("s_mul_i32 s{WRL}, s{ROW}, s{KB2}"),
        &[WRL],
        &[ROW, KB2],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{WRH}, s{ROW}, s{KB2}"),
        &[WRH],
        &[ROW, KB2],
    )?;
    sop(
        b,
        format!("{} s{WP}, s{WP}, s{WRL}", s_add_u32(a)),
        &[WP],
        &[WP, WRL],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, s{WRH}", s_addc_u32(a), WP + 1, WP + 1),
        &[WP + 1],
        &[WP + 1, WRH],
    )?;
    if e.wg8 {
        // Producer row offset (wave * K * 4) and the LDS lane vectors.
        sop(
            b,
            format!("s_mul_i32 s{XT}, s{WAVE}, s{KB}"),
            &[XT],
            &[WAVE, KB],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
            &[LDC],
            &[LANE],
            &[],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{TMP}, s{WAVE}, 10"),
            &[TMP],
            &[WAVE],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDST}, s{TMP}, v{LDC}"),
            &[LDST],
            &[LDC],
            &[TMP],
        )?;
    } else {
        row_pointers(b, e)?;
    }
    b.wait_all()
}

/// S1: X pointer pair of every live-row slot, `XP + r*K*4`.
fn row_pointers(b: &mut Builder, e: Body) -> R {
    if e.rowtiles {
        // Rebase full-window pointers once. Every subsequent X/Y offset stays
        // tile-local, including HC's fixed K-quarter offset already in XP.
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{TILE_R0}, s{KB}"),
            &[TMP],
            &[TILE_R0, KB],
        )?;
        add64(b, XP, XP, TMP)?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{TILE_R0}, s{MB}"),
            &[TMP],
            &[TILE_R0, MB],
        )?;
        add64(b, YP, YP, TMP)?;
    }
    let done = lbl(e, "row_pointers_done");
    for r in 0..u32::from(e.cap) {
        if e.rowtiles && r > 0 {
            skip_dead_row(b, r, &done)?;
        }
        let p = xpair(r);
        if r == 0 {
            op(
                b,
                format!("s_mov_b64 s[{p}:{}], s[{XP}:{}]", p + 1, XP + 1),
                &[sr(p, 2)],
                &[sr(XP, 2)],
            )?;
        } else {
            sop(b, format!("s_mul_i32 s{TMP}, s{KB}, {r}"), &[TMP], &[KB])?;
            add64(b, p, XP, TMP)?;
        }
    }
    if e.rowtiles {
        b.label(&done)?;
    }
    Ok(())
}

fn emit_x4(e: Body) -> Result<Emitted, String> {
    let (mut b, slot) = new_builder(e)?;
    let a = b.spec.arch;
    prologue_x4(&mut b, e)?;
    zero_acc(&mut b, e)?;
    sop(&mut b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    sop(&mut b, format!("s_cmp_lg_u32 s{SPATH}, 0"), &[], &[SPATH])?;
    branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(e, "scalar")))?;
    // Vector domain (K % 8 == 0): K256 groups.
    sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
    branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(e, "fold")))?;
    b.loop_(&lbl(e, "group"), |b| {
        match slot {
            Some(slot) => wg8_x4_tile(b, e, slot, true)?,
            None => s1_vector_group(b, e)?,
        }
        sop(b, format!("{} s{GI}, s{GI}, 1", s_add_i32(a)), &[GI], &[GI])?;
        sop(
            b,
            format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
            &[],
            &[GI, GROUPS],
        )?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(e, "group")))
    })?;
    branch(&mut b, &format!("s_branch {}", lbl(e, "fold")))?;
    // Scalar domain (K % 8 != 0): K32 tiles, one chain.
    b.label(&lbl(e, "scalar"))?;
    b.loop_(&lbl(e, "tile32"), |b| {
        match slot {
            Some(slot) => wg8_x4_tile(b, e, slot, false)?,
            None => s1_scalar_step(b, e)?,
        }
        sop(b, format!("{} s{GI}, s{GI}, 1", s_add_i32(a)), &[GI], &[GI])?;
        sop(b, format!("s_cmp_lt_u32 s{GI}, s{NT32}"), &[], &[GI, NT32])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(e, "tile32")))
    })?;
    b.label(&lbl(e, "fold"))?;
    // Waves past M never reach another barrier: they leave here.
    if e.wg8 {
        sop(&mut b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(e, "end")))?;
    }
    reduce_rows(&mut b, e)?;
    store_rows(&mut b, e)?;
    b.label(&lbl(e, "end"))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

// --------------------------------------------------------------------- HC k4
/// Kernarg loads, early returns, quarter geometry, weight / X pointers.
fn prologue_hc(b: &mut Builder, e: Body) -> R {
    let a = b.spec.arch;
    let end = lbl(e, "end");
    smem(b, KA, 8, KARG, 0)?;
    smem(b, SCALE, 2, KARG, 32)?;
    smem(b, HC_C, 1, KARG, 40)?;
    wave_lane(b)?;
    b.wait_all()?;
    op(
        b,
        format!("s_mov_b64 s[{WP}:{}], s[{KA}:{}]", WP + 1, KA + 1),
        &[sr(WP, 2)],
        &[sr(KA, 2)],
    )?;
    op(
        b,
        format!("s_mov_b64 s[{XP}:{}], s[{}:{}]", XP + 1, KA + 2, KA + 3),
        &[sr(XP, 2)],
        &[sr(KA + 2, 2)],
    )?;
    op(
        b,
        format!("s_mov_b64 s[{YP}:{}], s[{}:{}]", YP + 1, KA + 4, KA + 5),
        &[sr(YP, 2)],
        &[sr(KA + 4, 2)],
    )?;
    sop(b, format!("s_mov_b32 s{SM}, s{HC_M}"), &[SM], &[HC_M])?;
    sop(b, format!("s_mov_b32 s{SK}, s{HC_K}"), &[SK], &[HC_K])?;
    if !e.rowtiles {
        sop(b, format!("s_mov_b32 s{SC}, s{HC_C}"), &[SC], &[HC_C])?;
    }
    // Uniform early returns, before any barrier and any store.
    if e.wg8 {
        sop(b, format!("s_lshl_b32 s{MBASE}, s{SW}, 1"), &[MBASE], &[SW])?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{MBASE}, s{SM}"),
            &[],
            &[MBASE, SM],
        )?;
    } else {
        sop(b, format!("s_cmp_ge_u32 s{SW}, s{SM}"), &[], &[SW, SM])?;
    }
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    if e.rowtiles {
        sop(b, format!("s_cmp_eq_u32 s{HC_C}, 0"), &[], &[HC_C])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_gt_u32 s{HC_C}, 64"), &[], &[HC_C])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        rowtile_domain(b, e, HC_C)?;
    } else {
        sop(b, format!("s_cmp_eq_u32 s{SC}, 0"), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SC}, {}", e.cap), &[], &[SC])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
    }
    sop(b, format!("s_and_b32 s{TMP}, s{SK}, 31"), &[TMP], &[SK])?;
    sop(b, format!("s_cmp_lg_u32 s{TMP}, 0"), &[], &[TMP])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    // Output column and quarter of this wave.
    if e.wg8 {
        sop(
            b,
            format!("s_lshr_b32 s{MLOC}, s{WAVE}, 2"),
            &[MLOC],
            &[WAVE],
        )?;
        sop(b, format!("s_and_b32 s{QI}, s{WAVE}, 3"), &[QI], &[WAVE])?;
        sop(
            b,
            format!("{} s{COL}, s{MBASE}, s{MLOC}", s_add_i32(a)),
            &[COL],
            &[MBASE, MLOC],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{COL}, s{SM}"), &[], &[COL, SM])?;
        sop(b, format!("s_cselect_b32 s{SVAL}, 1, 0"), &[SVAL], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{QI}, s{WAVE}"), &[QI], &[WAVE])?;
        sop(b, format!("s_mov_b32 s{COL}, s{SW}"), &[COL], &[SW])?;
    }
    sop(b, format!("s_lshr_b32 s{PART}, s{SK}, 2"), &[PART], &[SK])?;
    sop(b, format!("s_mov_b32 s{SKL}, s{PART}"), &[SKL], &[PART])?;
    sop(
        b,
        format!("{} s{GROUPS}, s{PART}, {}", s_add_i32(a), lit(255)),
        &[GROUPS],
        &[PART],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{GROUPS}, 8"),
        &[GROUPS],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{PARTB}, s{PART}, 2"),
        &[PARTB],
        &[PART],
    )?;
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    sop(b, format!("s_lshl_b32 s{KB2}, s{SK}, 1"), &[KB2], &[SK])?;
    sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
    // Weight pointer: W + (col*K + q*part) * 2 (64-bit); a column past M is never loaded from.
    sop(
        b,
        format!("s_mul_i32 s{WRL}, s{COL}, s{KB2}"),
        &[WRL],
        &[COL, KB2],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{WRH}, s{COL}, s{KB2}"),
        &[WRH],
        &[COL, KB2],
    )?;
    sop(
        b,
        format!("{} s{WP}, s{WP}, s{WRL}", s_add_u32(a)),
        &[WP],
        &[WP, WRL],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, s{WRH}", s_addc_u32(a), WP + 1, WP + 1),
        &[WP + 1],
        &[WP + 1, WRH],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{QOFF}, s{QI}, s{PART}"),
        &[QOFF],
        &[QI, PART],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{QOFF}, s{QOFF}, 1"),
        &[QOFF],
        &[QOFF],
    )?;
    sop(
        b,
        format!("{} s{WP}, s{WP}, s{QOFF}", s_add_u32(a)),
        &[WP],
        &[WP, QOFF],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, 0", s_addc_u32(a), WP + 1, WP + 1),
        &[WP + 1],
        &[WP + 1],
    )?;
    if e.wg8 {
        sop(b, format!("s_lshl_b32 s{K2B}, s{KB}, 1"), &[K2B], &[KB])?;
        // Producer X base of this wave's vector: (wave >> 2) * K * 4 + (wave & 3) * part * 4.
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{MLOC}, s{KB}"),
            &[TMP],
            &[MLOC, KB],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{TMP2}, s{QI}, s{PARTB}"),
            &[TMP2],
            &[QI, PARTB],
        )?;
        sop(
            b,
            format!("{} s{XWB}, s{TMP}, s{TMP2}", s_add_i32(a)),
            &[XWB],
            &[TMP, TMP2],
        )?;
        // Lane vectors: producer store (wave*1024 + 4 lane), consumer read (q*1024 + 4 lane),
        // partial slot base (mloc*cap*16 + q*4).
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
            &[LDC],
            &[LANE],
            &[],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{TMP}, s{WAVE}, 10"),
            &[TMP],
            &[WAVE],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDST}, s{TMP}, v{LDC}"),
            &[LDST],
            &[LDC],
            &[TMP],
        )?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{QI}, 10"), &[TMP], &[QI])?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDR}, s{TMP}, v{LDC}"),
            &[LDR],
            &[LDC],
            &[TMP],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{MLOC}, {}", lit(16 * u32::from(e.cap))),
            &[TMP],
            &[MLOC],
        )?;
        sop(b, format!("s_lshl_b32 s{TMP2}, s{QI}, 2"), &[TMP2], &[QI])?;
        sop(
            b,
            format!("{} s{TMP}, s{TMP}, s{TMP2}", s_add_i32(a)),
            &[TMP],
            &[TMP, TMP2],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{VPART}, s{TMP}"),
            &[VPART],
            &[],
            &[TMP],
        )?;
    } else {
        // X pointer of this quarter, then the per-row pointers; partial slot base q*4.
        sop(b, format!("s_mov_b32 s{TMP}, s{PARTB}"), &[TMP], &[PARTB])?;
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{QI}, s{TMP}"),
            &[TMP],
            &[QI, TMP],
        )?;
        add64(b, XP, XP, TMP)?;
        row_pointers(b, e)?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{QI}, 2"), &[TMP], &[QI])?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{VPART}, s{TMP}"),
            &[VPART],
            &[],
            &[TMP],
        )?;
    }
    b.wait_all()
}

/// S2 HC K256 tile of this wave's quarter: weights once, then every four-row
/// stage (X of four rows x four quarters staged once, 16 KiB).
fn hc_group_wg8(b: &mut Builder, e: Body, slot: usize) -> R {
    tile_bases(b, 9, 10)?;
    tile_mask(b, 3, 256)?;
    let a = b.spec.arch;
    sop(
        b,
        format!("{} s{TMP}, s{XG}, s{XWB}", s_add_i32(a)),
        &[TMP],
        &[XG, XWB],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{XOFF}, 5, v{LANE}"),
        &[XOFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{XOFF}, s{TMP}, v{XOFF}"),
        &[XOFF],
        &[XOFF],
        &[TMP],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HOFF}, 4, v{LANE}"),
        &[HOFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
        &[HOFF],
        &[HOFF],
        &[GOFF],
    )?;
    let wdone = lbl(e, "hw_done");
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
    exec_set(b)?;
    let wr = |i: u8| WS + i;
    vmem(
        b,
        &format!(
            "global_load_b128 v[{WS}:{}], v{HOFF}, {}",
            WS + 3,
            gpair(WP)
        ),
        &[wr(0), wr(1), wr(2), wr(3)],
        &[HOFF],
        &[WP, WP + 1],
        false,
    )?;
    b.wait(Counter::Load, 0)?;
    widen_weights(b)?;
    exec_all(b)?;
    b.label(&wdone)?;
    for st in 0..u32::from(e.cap) / 4 {
        hc_stage(b, e, slot, st)?;
    }
    Ok(())
}

/// Rows `4 st .. 4 st + 3`: fill (two vectors per producer wave), barrier,
/// consume, barrier.
fn hc_stage(b: &mut Builder, e: Body, slot: usize, st: u32) -> R {
    let a = b.spec.arch;
    let rb = 4 * st;
    let done = lbl(e, &format!("st{st}_done"));
    if st > 0 {
        skip_dead_row(b, rb, &done)?;
    }
    for h in 0..2u32 {
        // Vector wave + 8h: row rb + (wave >> 2) + 2h, quarter wave & 3.
        let nofill = lbl(e, &format!("st{st}_h{h}_nofill"));
        sop(b, format!("s_lshr_b32 s{TMP}, s{WAVE}, 2"), &[TMP], &[WAVE])?;
        sop(
            b,
            format!("{} s{TMP}, s{TMP}, {}", s_add_i32(a), rb + 2 * h),
            &[TMP],
            &[TMP],
        )?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SC}"), &[], &[TMP, SC])?;
        branch(b, &format!("s_cbranch_scc1 {nofill}"))?;
        sop(b, format!("s_mul_i32 s{TMP2}, s{KB}, {rb}"), &[TMP2], &[KB])?;
        if h == 1 {
            sop(
                b,
                format!("{} s{TMP2}, s{TMP2}, s{K2B}", s_add_i32(a)),
                &[TMP2],
                &[TMP2, K2B],
            )?;
        }
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{VT}, s{TMP2}, v{XOFF}"),
            &[VT],
            &[XOFF],
            &[TMP2],
        )?;
        exec_set(b)?;
        for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{}:{}], v{VT}, {}{suffix}",
                    XBUF + j,
                    XBUF + j + 3,
                    gpair(XP)
                ),
                &[XBUF + j, XBUF + j + 1, XBUF + j + 2, XBUF + j + 3],
                &[VT],
                &[XP, XP + 1],
                false,
            )?;
        }
        for j in 0..8u32 {
            let x = XBUF + j as u8;
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{x}{}", off(8192 * h + 128 * j)),
                    vec![],
                    vec![v(LDST), v(x)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.wait_all()?;
        exec_all(b)?;
        b.label(&nofill)?;
    }
    b.barrier(&[Transition::Ready(slot)])?;
    let cdone = lbl(e, &format!("st{st}_cdone"));
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
    exec_set(b)?;
    for rl in 0..4u32 {
        if rb + rl > 0 {
            skip_dead_row(b, rb + rl, &cdone)?;
        }
        consume_row(b, e, slot, true, rb + rl, 4096 * rl, LDR)?;
    }
    b.label(&cdone)?;
    exec_all(b)?;
    b.barrier(&[Transition::Retire(slot)])?;
    b.label(&done)
}

/// Quarter sums of every live row into LDS (lane 0 of each valid wave) at
/// `((mloc*cap + r)*4 + q)*4`, then the barrier.
fn hc_partials(b: &mut Builder, e: Body, slot: usize) -> R {
    let pdone = lbl(e, "pdone");
    let stored = lbl(e, "pstored");
    if e.wg8 {
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {pdone}"))?;
    }
    reduce_rows(b, e)?;
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    for r in 0..u32::from(e.cap) {
        if r > 0 {
            skip_dead_row(b, r, &stored)?;
        }
        b.ds_store(
            slot,
            Instruction::new(
                format!("ds_store_b32 v{VPART}, v{}{}", acc(e, r, 3), off(16 * r)),
                vec![],
                vec![v(VPART), v(acc(e, r, 3))],
            )
            .memory(MemoryClass::DsStore),
        )?;
    }
    b.label(&stored)?;
    b.wait_all()?;
    exec_all(b)?;
    b.label(&pdone)?;
    b.barrier(&[Transition::Ready(slot)])
}

/// Items `n = wave + waves*k`: fold the four quarters of `(mloc, r)`, activate,
/// store `Y[r*M + col]`.
fn hc_items(b: &mut Builder, e: Body, slot: usize, diagnostic: bool) -> R {
    let a = b.spec.arch;
    let cap = u32::from(e.cap);
    let waves = if e.wg8 { 8 } else { 4 };
    let total = if e.wg8 { 2 * cap } else { cap };
    let log2cap = cap.trailing_zeros();
    for k in 0..total.div_ceil(waves) {
        let skip = lbl(e, &format!("iskip{k}"));
        sop(
            b,
            format!("{} s{TMP2}, s{WAVE}, {}", s_add_i32(a), waves * k),
            &[TMP2],
            &[WAVE],
        )?;
        if e.wg8 {
            sop(
                b,
                format!("s_lshr_b32 s{IM}, s{TMP2}, {log2cap}"),
                &[IM],
                &[TMP2],
            )?;
            sop(
                b,
                format!("s_and_b32 s{IR}, s{TMP2}, {}", cap - 1),
                &[IR],
                &[TMP2],
            )?;
        } else {
            sop(b, format!("s_mov_b32 s{IR}, s{TMP2}"), &[IR], &[TMP2])?;
        }
        sop(b, format!("s_cmp_ge_u32 s{IR}, s{SC}"), &[], &[IR, SC])?;
        branch(b, &format!("s_cbranch_scc1 {skip}"))?;
        if e.wg8 {
            sop(
                b,
                format!("{} s{TMP}, s{MBASE}, s{IM}", s_add_i32(a)),
                &[TMP],
                &[MBASE, IM],
            )?;
            sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SM}"), &[], &[TMP, SM])?;
            branch(b, &format!("s_cbranch_scc1 {skip}"))?;
            sop(b, format!("s_mov_b32 s{COL}, s{TMP}"), &[COL], &[TMP])?;
            sop(b, format!("s_lshl_b32 s{TMP}, s{TMP2}, 4"), &[TMP], &[TMP2])?;
        } else {
            sop(b, format!("s_lshl_b32 s{TMP}, s{IR}, 4"), &[TMP], &[IR])?;
        }
        valu(
            b,
            &format!("v_mov_b32_e32 v{VYA}, s{TMP}"),
            &[VYA],
            &[],
            &[TMP],
        )?;
        b.ds_load(
            slot,
            Instruction::new(
                format!("ds_load_b128 v[{VITQ}:{}], v{VYA}", VITQ + 3),
                (0..4).map(|i| v(VITQ + i)).collect(),
                vec![v(VYA)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        if diagnostic {
            // Read the actual LDS quarter payload before hc_value consumes it.
            for q in 0..4u8 {
                // TMP is dead after the LDS address setup. Materialize the
                // plane stride here: 128/192/256 are not SOP2 inline constants.
                sop(
                    b,
                    format!("s_mov_b32 s{TMP}, {}", lit(64 * (u32::from(q) + 1))),
                    &[TMP],
                    &[],
                )?;
                sop(
                    b,
                    format!("{} s{TMP}, s{IR}, s{TMP}", s_add_i32(a)),
                    &[TMP],
                    &[IR, TMP],
                )?;
                sop(
                    b,
                    format!("s_mul_i32 s{YB}, s{TMP}, s{MB}"),
                    &[YB],
                    &[TMP, MB],
                )?;
                sop(b, format!("s_lshl_b32 s{TMP}, s{COL}, 2"), &[TMP], &[COL])?;
                sop(
                    b,
                    format!("{} s{YB}, s{YB}, s{TMP}", s_add_i32(a)),
                    &[YB],
                    &[YB, TMP],
                )?;
                valu(
                    b,
                    &format!("v_mov_b32_e32 v{VYA}, s{YB}"),
                    &[VYA],
                    &[],
                    &[YB],
                )?;
                exec_lane0(b)?;
                vmem(
                    b,
                    &format!("global_store_b32 v{VYA}, v{}, {}", VITQ + q, gpair(YP)),
                    &[],
                    &[VYA, VITQ + q],
                    &[YP, YP + 1],
                    true,
                )?;
                b.wait_all()?;
                exec_all(b)?;
            }
        }
        hc_value(b, &lbl(e, &format!("item{k}")))?;
        // Y[r*M + col].
        sop(
            b,
            format!("s_mul_i32 s{YB}, s{IR}, s{MB}"),
            &[YB],
            &[IR, MB],
        )?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{COL}, 2"), &[TMP], &[COL])?;
        sop(
            b,
            format!("{} s{YB}, s{YB}, s{TMP}", s_add_i32(a)),
            &[YB],
            &[YB, TMP],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{VYA}, s{YB}"),
            &[VYA],
            &[],
            &[YB],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{VYD}, v{VOUT}"),
            &[VYD],
            &[VOUT],
            &[],
        )?;
        exec_lane0(b)?;
        vmem(
            b,
            &format!("global_store_b32 v{VYA}, v{VYD}, {}", gpair(YP)),
            &[],
            &[VYA, VYD],
            &[YP, YP + 1],
            true,
        )?;
        b.wait_all()?;
        exec_all(b)?;
        b.label(&skip)?;
    }
    Ok(())
}

/// In-place BF16 round trip of the wave-uniform f32 in `v{val}` on three scratch
/// VGPRs, the object's integer sequence (finite: `(u + ((u >> 16) & 1) + 0x7fff)
/// & 0xffff0000`; Inf/NaN, exponent field all ones: `low16(u) == 0 ? u : u |
/// 0x10000`, then the mask).
fn round_trip(b: &mut Builder, val: u8, t: [u8; 3]) -> R {
    let [t0, t1, t2] = t;
    valu(
        b,
        &format!("v_bfe_u32 v{t0}, v{val}, 16, 1"),
        &[t0],
        &[val],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{t0}, v{val}, v{t0}"),
        &[t0],
        &[val, t0],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{t0}, 0x7fff, v{t0}"),
        &[t0],
        &[t0],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{t1}, 0xffff, v{val}"),
        &[t1],
        &[val],
        &[],
    )?;
    valu(
        b,
        &format!("v_or_b32_e32 v{t2}, 0x10000, v{val}"),
        &[t2],
        &[val],
        &[],
    )?;
    valu(
        b,
        &format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{t1}"),
        &[],
        &[t1],
        &[],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        &format!("v_cndmask_b32_e32 v{t1}, v{t2}, v{val}, vcc_lo"),
        &[t1],
        &[t2, val],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{t2}, 0x7f800000, v{val}"),
        &[t2],
        &[val],
        &[],
    )?;
    valu(
        b,
        &format!("v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v{t2}"),
        &[],
        &[t2],
        &[],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        &format!("v_cndmask_b32_e32 v{t0}, v{t0}, v{t1}, vcc_lo"),
        &[t0],
        &[t0, t1],
        &[],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{val}, 0xffff0000, v{t0}"),
        &[val],
        &[t0],
        &[],
    )
}

/// The k4 epilogue of one item on the four quarters in `VITQ..VITQ+3` (the same
/// value in every lane): `out = (q0 + q1) + (q2 + q3)`; with `hc_act == 0` that is
/// the result, else `rt(rt(scale * rt(out)) / (1 + exp(-x)))`. Every node, constant
/// and operand order is the object's: the SALU `s_add/mul/fmamk/rndne/sub/cvt`
/// nodes are the `v_add/mul/fma/rndne/sub/cvt` of the same operands, `exp` is the
/// transcendental `v_exp_f32`, the clamps are the object's `ngt` / `nlt`
/// predicates (unordered true) and the division is the full `v_div_scale / v_rcp /
/// v_fma / v_div_fmas / v_div_fixup` expansion. Result in `VOUT`. (Equality of the
/// SALU float unit with the VALU on exceptional inputs is H0's claim, not the CPU
/// oracle's.)
fn hc_value(b: &mut Builder, label: &str) -> R {
    let q = |i: u8| VITQ + i;
    valu(
        b,
        &format!("v_add_f32_e32 v{T0}, v{}, v{}", q(0), q(1)),
        &[T0],
        &[q(0), q(1)],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_f32_e32 v{T1}, v{}, v{}", q(2), q(3)),
        &[T1],
        &[q(2), q(3)],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_f32_e32 v{VOUT}, v{T0}, v{T1}"),
        &[VOUT],
        &[T0, T1],
        &[],
    )?;
    let done = format!("{label}_done");
    sop(b, format!("s_cmp_eq_u32 s{ACT}, 0"), &[], &[ACT])?;
    branch(b, &format!("s_cbranch_scc1 {done}"))?;
    // x = rt(sum); x = rt(scale * x).
    round_trip(b, VOUT, [T0, T1, T2])?;
    valu(
        b,
        &format!("v_mul_f32_e32 v{VOUT}, s{SCALE}, v{VOUT}"),
        &[VOUT],
        &[VOUT],
        &[SCALE],
    )?;
    round_trip(b, VOUT, [T0, T1, T2])?;
    // exp(-x): the object's range reduction.
    valu(
        b,
        &format!("v_mul_f32_e32 v{T0}, 0xbfb8aa3b, v{VOUT}"),
        &[T0],
        &[VOUT],
        &[],
    )?;
    valu(
        b,
        &format!("v_xor_b32_e32 v{T1}, 0x80000000, v{T0}"),
        &[T1],
        &[T0],
        &[],
    )?;
    valu(
        b,
        &format!("v_rndne_f32_e32 v{T2}, v{T0}"),
        &[T2],
        &[T0],
        &[],
    )?;
    valu(
        b,
        &format!("v_fma_f32 v{T1}, v{VOUT}, 0xbfb8aa3b, v{T1}"),
        &[T1],
        &[VOUT, T1],
        &[],
    )?;
    valu(
        b,
        &format!("v_sub_f32_e32 v{T0}, v{T0}, v{T2}"),
        &[T0],
        &[T0, T2],
        &[],
    )?;
    valu(
        b,
        &format!("v_fma_f32 v{T1}, v{VOUT}, 0xb2a5705f, v{T1}"),
        &[T1],
        &[VOUT, T1],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_f32_e32 v{T0}, v{T0}, v{T1}"),
        &[T0],
        &[T0, T1],
        &[],
    )?;
    valu(
        b,
        &format!("v_cvt_i32_f32_e32 v{T3}, v{T2}"),
        &[T3],
        &[T2],
        &[],
    )?;
    valu(b, &format!("v_exp_f32_e32 v{T0}, v{T0}"), &[T0], &[T0], &[])?;
    valu(
        b,
        &format!("v_ldexp_f32 v{T0}, v{T0}, v{T3}"),
        &[T0],
        &[T0, T3],
        &[],
    )?;
    // keep e when !(x > 0x42ce8ed0); +Inf when x < 0xc2b17218 (both unordered-true).
    valu(
        b,
        &format!("v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v{VOUT}"),
        &[],
        &[VOUT],
        &[],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        &format!("v_cndmask_b32_e32 v{T0}, 0, v{T0}, vcc_lo"),
        &[T0],
        &[T0],
        &[],
    )?;
    valu(
        b,
        &format!("v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v{VOUT}"),
        &[],
        &[VOUT],
        &[],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        &format!("v_cndmask_b32_e32 v{T0}, 0x7f800000, v{T0}, vcc_lo"),
        &[T0],
        &[T0],
        &[],
    )?;
    // y = x / (1 + exp(-x)): the full division expansion.
    let [w0, w1, w2, w3, w4, w5] = [T0, T1, T2, T3, T4, T5];
    valu(
        b,
        &format!("v_add_f32_e32 v{w0}, 1.0, v{w0}"),
        &[w0],
        &[w0],
        &[],
    )?;
    valu(
        b,
        &format!("v_div_scale_f32 v{w1}, null, v{w0}, v{w0}, v{VOUT}"),
        &[w1],
        &[w0, VOUT],
        &[],
    )?;
    valu(b, &format!("v_rcp_f32_e32 v{w2}, v{w1}"), &[w2], &[w1], &[])?;
    valu(
        b,
        &format!("v_xor_b32_e32 v{w1}, 0x80000000, v{w1}"),
        &[w1],
        &[w1],
        &[],
    )?;
    valu(
        b,
        &format!("v_fma_f32 v{w3}, v{w1}, v{w2}, 1.0"),
        &[w3],
        &[w1, w2],
        &[],
    )?;
    valu(
        b,
        &format!("v_fmac_f32_e32 v{w2}, v{w3}, v{w2}"),
        &[w2],
        &[w2, w3],
        &[],
    )?;
    valu(
        b,
        &format!("v_div_scale_f32 v{w3}, vcc_lo, v{VOUT}, v{w0}, v{VOUT}"),
        &[w3],
        &[VOUT, w0],
        &[],
    )?;
    valu(
        b,
        &format!("v_mul_f32_e32 v{w4}, v{w3}, v{w2}"),
        &[w4],
        &[w3, w2],
        &[],
    )?;
    valu(
        b,
        &format!("v_fma_f32 v{w5}, v{w1}, v{w4}, v{w3}"),
        &[w5],
        &[w1, w4, w3],
        &[],
    )?;
    valu(
        b,
        &format!("v_fmac_f32_e32 v{w4}, v{w5}, v{w2}"),
        &[w4],
        &[w4, w5, w2],
        &[],
    )?;
    valu(
        b,
        &format!("v_fmac_f32_e32 v{w3}, v{w1}, v{w4}"),
        &[w3],
        &[w3, w1, w4],
        &[],
    )?;
    alu_wait(b, "depctr_va_vcc(0)")?;
    valu(
        b,
        &format!("v_div_fmas_f32 v{w1}, v{w3}, v{w2}, v{w4}"),
        &[w1],
        &[w3, w2, w4],
        &[],
    )?;
    valu(
        b,
        &format!("v_div_fixup_f32 v{w0}, v{w1}, v{w0}, v{VOUT}"),
        &[w0],
        &[w1, w0, VOUT],
        &[],
    )?;
    // out = rt(y).
    valu(
        b,
        &format!("v_mov_b32_e32 v{VOUT}, v{w0}"),
        &[VOUT],
        &[w0],
        &[],
    )?;
    round_trip(b, VOUT, [T0, T1, T2])?;
    b.label(&done)
}

fn emit_hc(e: Body) -> Result<Emitted, String> {
    emit_hc_impl(e, false)
}

fn emit_hc_impl(e: Body, diagnostic: bool) -> Result<Emitted, String> {
    let (mut b, slot) = new_builder(e)?;
    if diagnostic {
        b.spec.symbol = "gemv_bf16_xf32_k4_rows_t8_quarters_pm_gfx1201".into();
    }
    let slot = slot.ok_or("HC entries own LDS")?;
    let a = b.spec.arch;
    prologue_hc(&mut b, e)?;
    zero_acc(&mut b, e)?;
    sop(&mut b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
    branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(e, "fold")))?;
    b.loop_(&lbl(e, "group"), |b| {
        if e.wg8 {
            hc_group_wg8(b, e, slot)?;
        } else {
            s1_vector_group(b, e)?;
        }
        sop(b, format!("{} s{GI}, s{GI}, 1", s_add_i32(a)), &[GI], &[GI])?;
        sop(
            b,
            format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
            &[],
            &[GI, GROUPS],
        )?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(e, "group")))
    })?;
    b.label(&lbl(e, "fold"))?;
    hc_partials(&mut b, e, slot)?;
    hc_items(&mut b, e, slot, diagnostic)?;
    b.wait_all()?;
    b.label(&lbl(e, "end"))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Build one export.
pub fn build(e: Entry) -> Result<Emitted, String> {
    let e = Body {
        entry: e,
        rowtiles: false,
    };
    if e.hc {
        emit_hc(e)
    } else {
        emit_x4(e)
    }
}

/// Build a fixed-eight S1 twin for total logical R=1..=64. Only cap8 S1 is
/// accepted: changing R never changes K partitions, register capacity, or DAG.
/// HC exceptional-input SALU/VALU and exp/div parity still requires hardware;
/// this builder does not grant a shipping capability.
pub fn build_rowtiles(e: Entry) -> Result<Emitted, String> {
    if e.cap != 8 || e.wg8 {
        return Err(format!(
            "lane_rows_bf16: rowtiles require cap8 S1, got {e:?}"
        ));
    }
    let e = Body {
        entry: e,
        rowtiles: true,
    };
    if e.hc {
        emit_hc(e)
    } else {
        emit_x4(e)
    }
}

/// Lab-only HC observation, excluded from production modules and assets.
/// Same 48-byte HC ABI. Y must own five 64*M-f32 planes: final in plane 0,
/// actual fused quarter q in plane q+1, indexed row*M+column. Only active
/// rows are written. Registers, LDS, arithmetic and admission guards are shared
/// with the ordinary HC rowtile builder; no diagnostic instructions are emitted
/// by production entry points.
pub fn emit_hc_quarter_diagnostic() -> Result<Emitted, String> {
    emit_hc_impl(
        Body {
            entry: Entry {
                hc: true,
                wg8: false,
                cap: 8,
            },
            rowtiles: true,
        },
        true,
    )
}

fn exported(e: &Emitted) -> Option<&str> {
    e.s_text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .map(str::trim)
}

/// Build the ten exports and combine them into the one module source and proof.
/// The exports must be exactly [`SYMBOLS`] in order, built for exact gfx1201, at
/// their frozen VGPR / LDS / workgroup resources.
pub fn emit_module() -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let mut emitted = Vec::with_capacity(ENTRIES.len());
    for e in ENTRIES {
        emitted.push(build(e).map_err(|err| format!("{MODULE}: {}: {err}", e.symbol()))?);
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
    for (i, e) in emitted.iter().enumerate() {
        if e.shape.next_free_vgpr != VGPR_HIGH_WATER[i] {
            return Err(format!(
                "{MODULE}: {} reserves {} VGPRs, frozen {}",
                SYMBOLS[i], e.shape.next_free_vgpr, VGPR_HIGH_WATER[i]
            ));
        }
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}
