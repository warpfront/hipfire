// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab (ctl3-hot) GPU entries, gfx1201, gfx1100 and gfx1151 (two buckets,
//! cold = 0 and MQ4 = 1; the cold codec is W3 (1) or E8P (3) on gfx1201, E8P (3) on gfx1100 and
//! W3 (1) on gfx1151, plus E8P (3) on gfx1151 once admitted (its K1/K2 tables exist); every arch also
//! serves MQ4-only layers (`cold_codec == 0`); W2 is refused
//! everywhere, gfx1100 refuses W3, gfx1151 refuses E8P until admitted; every other arch refuses). The K1 cold-bucket
//! table is inventory ([`slab_cold_k1_metadata`]); serving selection is the admission-gated
//! [`slab_cold_kernels`]. The per-arch modules (with the hot MQ4 K1 bucket) are
//! [`SlabArchKernels`], the per-cold-codec cold K1 blobs and symbols [`SlabCold`].
//!
//! * K1 decode: [`Gpu::qwen4_slab_union_plan`] builds one immutable window
//!   descriptor, shared by gate/up and down. Above eight logical rows the PM
//!   planner emits fixed-eight-row descriptors and K1 consumes them with grid-z
//!   tiling; unused expert slots are deterministically empty.
//!   [`Gpu::mq_i4_moe_gemv_slab`] launches a projection over the non-empty
//!   buckets. Both go through `launch_maybe_blob`, so they are recorded by the
//!   replay recorder and captured by graphs like every other route launch.
//!   The planner's device `u8` table is membership only (cold = 1, MQ4 = 0).
//!   Every K1 symbol (all three archs, MQ4/W3/E8P, narrow and wide) has the 96-byte kernarg
//!   ABI: seven pointers, eight u32 words at `@56..@88`, then `n_experts@88` and a zero
//!   `reserved@92`. A workgroup whose expert id is not `< n_experts` (one unsigned,
//!   workgroup-uniform compare, so a negative id is out of range too) takes the existing
//!   no-load, no-store exit before the id indexes the pointer table, before any
//!   weight or descriptor prefetch and before the first barrier: its output tile stays
//!   untouched. [`Gpu::mq_i4_moe_gemv_slab`] passes `n_experts = SLAB_EXPERTS`, the
//!   extent of the `[512]` pointer table it validates.
//! * K2 prefill: [`Gpu::qwen4_slab_tile_masks`] splits the grouped tile ids by
//!   codec with the table-driven mask: the builder `ep_tile_mask_pm_<arch>` (gfx1100, gfx1201), or on
//!   gfx1151 its hipcc twin `qwen4_slab_tile_mask` (same law, same ids out); gfx1100 down
//!   sidecars come from [`Gpu::qwen4_moe_rotate128_i4_slab`] (K3).

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemv::WideTwinFamily;
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Module key of the embedded gfx1201 K1 certified builder image.
pub const MQ_I4_MOE_GEMV_MODULE: &str = "mq_i4_moe_gemv_pm_gfx1201";
/// Module key of the embedded gfx1151 K1 certified builder image (MQ4 and W3 symbols).
pub const MQ_I4_MOE_GEMV_GFX1151_MODULE: &str = "mq_i4_moe_gemv_pm_gfx1151";

/// Explicit arguments supplied through HIP's kernelParams launch path.
const MQ_I4_MOE_GEMV_KERNARG_COUNT: usize = 17;

/// K1 symbols (gfx1201 and gfx1151) in bucket order: `[bucket][kind]`, bucket 0 = W3 (`MQ3I4`),
/// 1 = MQ4 (`Mq4Sym`), kind 0 = gate/up (R256), 1 = down (R128).
pub const MQ_I4_MOE_GEMV_SYMBOLS: [[&str; 2]; 2] = [
    [
        "mq_i4_moe_gemv_mq3i4_r256_gate_up",
        "mq_i4_moe_gemv_mq3i4_r128_down",
    ],
    [
        "mq_i4_moe_gemv_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_mq4sym_r128_down",
    ],
];

/// Fixed-eight-row K1 exports in the gfx1201 shared image, `[bucket][kind]`.
/// Every descriptor has 80 slots, irrespective of the logical window length.
pub const MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS: [[&str; 2]; 2] = [
    [
        "mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq3i4_r128_down_t8",
    ],
    [
        "mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_mq4sym_r128_down_t8",
    ],
];

/// Fixed-eight-row exports in the gfx1151 Cn image, `[bucket][kind]`.
pub const MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS: [[&str; 2]; 2] = [
    [
        "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq3i4_r128_down_t8",
    ],
    [
        "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up_t8",
        "mq_i4_moe_gemv_cn_mq4sym_r128_down_t8",
    ],
];

/// Module key of the embedded gfx1100 MQ4-only K1 certified builder image (the hot MQ4 bucket;
/// no W2 entry). Distinct from the retired mixed `mq_i4_moe_gemv_pm_gfx1100` module.
pub const MQ_I4_MOE_GEMV_GFX1100_MODULE: &str = "mq_i4_moe_gemv_mq4only_pm_gfx1100";

/// gfx1100 MQ4 (`Mq4Sym`) K1 symbols (the hot bucket): `[kind]`, 0 = gate/up (R256), 1 = down (R128).
pub const MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
    "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100",
];

/// Module key of the embedded gfx1100 E8P K1 certified builder image (cold bucket metadata).
pub const E8P_I4_MOE_GEMV_GFX1100_MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1100";

/// gfx1100 E8P K1 symbols: `[kind]`, 0 = gate/up (R256), 1 = down (R128), contracts
/// `kernels/e8p_i4_moe_gemv.gfx1100.{r256_gate_up,r128_down}.contract.json`. Same 96-byte
/// kernarg ABI (`n_experts@88`, zero `reserved@92`; the frozen 88-byte F4 layout plus the
/// trailing guard words), grid `[M, U, 1]`, block `[32, 1, 1]` and zero dynamic LDS as the MQ4 K1 symbols
/// (static LDS 1,536 B instead of 0).
pub const E8P_I4_MOE_GEMV_GFX1100_SYMBOLS: [&str; 2] = [
    "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
    "e8p_i4_moe_gemv_r128_down_pm_gfx1100",
];

/// Module key and symbol of the gfx1100 K3 down A4 producer.
pub const MQ_I4_A4_GFX1100_MODULE: &str = "mq_i4_a4_pm_gfx1100";
pub const MQ_I4_A4_GFX1100_SYMBOL: &str = "mq_i4_a4_r128_k640_gfx1100";

/// Module key and symbol of the gfx1100 slab tile mask (module = symbol).
pub const EP_TILE_MASK_GFX1100_MODULE: &str = "ep_tile_mask_pm_gfx1100";
/// Module key of the embedded K1-C1 certified builder image (gfx1151 only).
const MQ_I4_MOE_GEMV_C1_MODULE: &str = "mq_i4_moe_gemv_c1_pm_gfx1151";

/// K1-C1 symbols: `[kind]`, 0 = gate/up (R256), 1 = down (R128). Each serves both codecs.
pub const MQ_I4_MOE_GEMV_C1_SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_c1_r256_gate_up",
    "mq_i4_moe_gemv_c1_r128_down",
];
/// Bytes of the K1-C1 kernarg blob (frozen ABI: `n_experts@72`, zero `reserved@76`).
pub const MQ_I4_MOE_GEMV_C1_KERNARG_BYTES: usize = 80;
/// Explicit arguments supplied through HIP's kernelParams launch path for K1-C1:
/// the 12 original arguments, `n_experts` and `reserved` (listed by the metadata as argument 14).
const MQ_I4_MOE_GEMV_C1_KERNARG_COUNT: usize = 14;
/// Output rows per K1-C1 workgroup (one wave32 each).
pub const MQ_I4_MOE_GEMV_C1_ROWS_PER_WG: usize = 8;
/// Threads per K1-C1 workgroup.
pub const MQ_I4_MOE_GEMV_C1_BLOCK: u32 = 256;

/// Whether the K1-C1 kernels (and K1-Cn), which carry only the MQ4 and W3 (`MQ3I4`)
/// arithmetic, serve a layer whose cold bucket is codec `cold_codec`: `0` (an MQ4-only layer)
/// or `1` (W3). Every other cold codec keeps the K1 bucket entries.
pub const fn k1_rows8_cold_codec(cold_codec: u8) -> bool {
    matches!(cold_codec, 0 | 1)
}

/// Module key of the embedded K1-Cn certified builder image (gfx1151 only).
const MQ_I4_MOE_GEMV_CN_MODULE: &str = "mq_i4_moe_gemv_cn_pm_gfx1151";

/// K1-Cn symbols in the K1 bucket order, `[bucket][kind]`: bucket 0 = W3 (`mq3i4`), 1 = MQ4
/// (`mq4sym`), kind 0 = gate/up (R256), 1 = down (R128). Same kernarg ABI as the K1 symbols.
pub const MQ_I4_MOE_GEMV_CN_SYMBOLS: [[&str; 2]; 2] = [
    [
        "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
        "mq_i4_moe_gemv_cn_mq3i4_r128_down",
    ],
    [
        "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_cn_mq4sym_r128_down",
    ],
];
/// Output rows per K1-Cn workgroup (one wave32 each).
pub const MQ_I4_MOE_GEMV_CN_ROWS_PER_WG: usize = 8;
/// Threads per K1-Cn workgroup.
pub const MQ_I4_MOE_GEMV_CN_BLOCK: u32 = 256;

/// Which of the two gfx1151 K1 schedules serves a [`Gpu::mq_i4_moe_gemv_slab`] launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabK1Module {
    /// One wave32 per output row (the bucket's [`SlabK1`] entry), any lane count 1..=8, every arch.
    K1,
    /// Eight rows per 256-thread workgroup with the expert's slots' X in LDS
    /// (`mq_i4_moe_gemv_cn_pm_gfx1151`), bitwise K1; gfx1151 MQ4/W3 layers only.
    Cn,
    /// Fixed-eight-row descriptors and grid-z scheduling. Explicit selection is
    /// the byte-oracle path, independent of the product family selector.
    RowTiles,
}

/// The module the default dispatch picks: Cn on gfx1151 when `2 <= lanes <= 8` and the switch is
/// on, K1 in every other case.
pub const fn slab_k1_default_module(gfx1151: bool, lanes: usize, cn_enabled: bool) -> SlabK1Module {
    if gfx1151 && cn_enabled && lanes >= 2 && lanes <= SLAB_NARROW_MAX_LANES {
        SlabK1Module::Cn
    } else {
        SlabK1Module::K1
    }
}

/// Module, symbol and launch geometry of one K1 bucket launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabK1Pick {
    pub module_key: &'static str,
    pub image: &'static [u8],
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// The module/symbol/geometry of bucket `bucket` (0 = W3, 1 = MQ4) of a projection.
/// `slots` is the whole-window slot count. K1 uses `[m, slots, 1]`, Cn
/// `[m / 8, slots, 1]`, and RowTiles `[ceil(m/8), 80, ceil(slots/80)]`.
/// Callers validate dimensions and codec/architecture inventory before selecting.
pub fn slab_k1_pick(
    k1: &SlabK1,
    module: SlabK1Module,
    bucket: usize,
    down: bool,
    m: usize,
    slots: usize,
) -> SlabK1Pick {
    match module {
        SlabK1Module::K1 => SlabK1Pick {
            module_key: k1.module,
            image: k1.image,
            symbol: k1.symbols[down as usize],
            grid: [m as u32, slots as u32, 1],
            block: [32, 1, 1],
        },
        SlabK1Module::Cn => SlabK1Pick {
            module_key: MQ_I4_MOE_GEMV_CN_MODULE,
            image: kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
            symbol: MQ_I4_MOE_GEMV_CN_SYMBOLS[bucket][down as usize],
            grid: [(m / MQ_I4_MOE_GEMV_CN_ROWS_PER_WG) as u32, slots as u32, 1],
            block: [MQ_I4_MOE_GEMV_CN_BLOCK, 1, 1],
        },
        SlabK1Module::RowTiles => {
            slab_rowtile_pick(k1, down, m, slots / SLAB_TOP_K).expect("validated rowtile launch")
        }
    }
}

/// Launch geometry and kernarg scalars of one K1-C1 projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabC1Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub m: u32,
    pub k: u32,
    pub x_row_div: u32,
    pub y_slot_stride: u32,
}

/// The K1-C1 launch of the gate/up (`down == false`) or down projection: grid
/// `[M / 8, 10, 1]`, block `[256, 1, 1]`.
pub const fn slab_c1_launch(down: bool) -> SlabC1Launch {
    let (m, k, x_row_div, y_slot_stride) = if down {
        (SLAB_DOWN_M, SLAB_DOWN_K, 1, SLAB_DOWN_M)
    } else {
        (
            SLAB_GATE_UP_M,
            SLAB_GATE_UP_K,
            SLAB_TOP_K,
            SLAB_GATE_UP_M / 2,
        )
    };
    SlabC1Launch {
        grid: [
            (m / MQ_I4_MOE_GEMV_C1_ROWS_PER_WG) as u32,
            SLAB_TOP_K as u32,
            1,
        ],
        block: [MQ_I4_MOE_GEMV_C1_BLOCK, 1, 1],
        m: m as u32,
        k: k as u32,
        x_row_div: x_row_div as u32,
        y_slot_stride: y_slot_stride as u32,
    }
}

/// The 80-byte K1-C1 kernarg blob in ABI order: six pointers (expert weight table, `topk`,
/// codec table, X, Y0, Y1), then `M`, `K`, `uniform_codec`, `x_row_div`, `x_row_stride` (= `K`),
/// `y_slot_stride`, `n_experts@72` (the extent of the weight pointer table) and a zero `reserved@76`.
#[allow(clippy::too_many_arguments)]
fn slab_c1_kernarg(
    weight_ptrs: *mut c_void,
    topk: *mut c_void,
    codec_table: *mut c_void,
    x: *mut c_void,
    y0: *mut c_void,
    y1: *mut c_void,
    uniform_codec: i32,
    n_experts: u32,
    geo: &SlabC1Launch,
) -> hip_bridge::KernargBlob {
    let mut b = hip_bridge::KernargBlob::with_capacity(MQ_I4_MOE_GEMV_C1_KERNARG_BYTES);
    for ptr in [weight_ptrs, topk, codec_table, x, y0, y1] {
        b.push_ptr(ptr);
    }
    b.push_u32(geo.m);
    b.push_u32(geo.k);
    b.push_i32(uniform_codec);
    b.push_u32(geo.x_row_div);
    b.push_u32(geo.k);
    b.push_u32(geo.y_slot_stride);
    b.push_u32(n_experts);
    b.push_u32(0);
    b
}
const PLAN_SYMBOL: &str = "qwen4_slab_union_plan";
/// Planner module keys: the shared one (gfx1201 and gfx1151: the historical key) and gfx1100's own.
const PLAN_MODULE_SHARED: &str = "qwen4_slab_union_plan";
const PLAN_MODULE_GFX1100: &str = "qwen4_slab_union_plan_gfx1100";
/// Module key and symbol of the gfx1151 hipcc tile mask (module = symbol).
const MASK_SYMBOL: &str = "qwen4_slab_tile_mask";
const MASK_MODULE: &str = "qwen4_slab_tile_mask";

/// Threads (tiles) per workgroup of the tile mask kernel (the builder masks and the gfx1151 hipcc twin).
const SLAB_TILE_MASK_BLOCK: u32 = 256;

/// One K1 bucket's module, blob and symbols: `[kind]`, kind 0 = gate/up (R256), 1 = down (R128).
pub struct SlabK1 {
    pub module: &'static str,
    pub image: &'static [u8],
    pub symbols: [&'static str; 2],
}

/// The cold-bucket K1 table of one cold codec on one arch (bucket 0).
pub struct SlabCold {
    /// Raw codec id of the cold bucket in the codec tables (also the cold tile-mask rank).
    pub cold_codec: u8,
    pub k1: SlabK1,
}

/// The table-driven tile mask of one arch: an embedded certified builder image or a hipcc source
/// (the repository's embedded/HIP distinction). Allocation-free, so it is a valid static table value.
#[derive(Clone, Copy)]
pub enum SlabTileMask {
    /// Embedded builder image `(module, image, symbol)`: gfx1201 and gfx1100.
    Embedded {
        module: &'static str,
        image: &'static [u8],
        symbol: &'static str,
    },
    /// hipcc-compiled source `(module, source, symbol)`: gfx1151 only.
    Hipcc {
        module: &'static str,
        source: &'static str,
        symbol: &'static str,
    },
}

/// The sealed slab's per-arch modules, blobs and symbols that do not depend on the cold codec,
/// including the hot MQ4 K1 bucket (bucket 1).
pub struct SlabArchKernels {
    pub arch: &'static str,
    /// Lane-union planner: hipcc module key and source.
    pub plan_module: &'static str,
    pub plan_src: fn() -> &'static str,
    /// Hot MQ4 K1 bucket.
    pub mq4_k1: SlabK1,
    /// K3 down A4 producer `(module, image, symbol)`; gfx1100 only (gfx1201 and gfx1151 delegate to
    /// the existing `qwen4_moe_rotate128_i4`).
    pub k3: Option<(&'static str, &'static [u8], &'static str)>,
    /// Table-driven tile mask.
    pub tile_mask: SlabTileMask,
}

fn plan_src_shared() -> &'static str {
    kernels::QWEN4_SLAB_UNION_PLAN_SRC
}

static SLAB_GFX1201: SlabArchKernels = SlabArchKernels {
    arch: "gfx1201",
    plan_module: PLAN_MODULE_SHARED,
    plan_src: plan_src_shared,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[1],
    },
    k3: None,
    tile_mask: SlabTileMask::Embedded {
        module: crate::moe::EP_TILE_MASK_PM_GFX1201_SYMBOL,
        image: kernels::EP_TILE_MASK_PM_GFX1201,
        symbol: crate::moe::EP_TILE_MASK_PM_GFX1201_SYMBOL,
    },
};

static SLAB_GFX1100: SlabArchKernels = SlabArchKernels {
    arch: "gfx1100",
    plan_module: PLAN_MODULE_GFX1100,
    plan_src: kernels::qwen4_slab_union_plan_gfx1100_src,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1100_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
        symbols: MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS,
    },
    k3: Some((
        MQ_I4_A4_GFX1100_MODULE,
        kernels::MQ_I4_A4_PM_GFX1100,
        MQ_I4_A4_GFX1100_SYMBOL,
    )),
    tile_mask: SlabTileMask::Embedded {
        module: EP_TILE_MASK_GFX1100_MODULE,
        image: kernels::EP_TILE_MASK_PM_GFX1100,
        symbol: EP_TILE_MASK_GFX1100_MODULE,
    },
};

/// gfx1151 (Strix Halo): beta's certified K1 image (MQ4 hot bucket; the W3 cold bucket is
/// [`SLAB_COLD_GFX1151_W3`] on the same image), the shared planner, the hipcc tile mask and no K3
/// (the down sidecar delegates to the existing producer, `qwen4_moe_rotate128_i4`).
static SLAB_GFX1151: SlabArchKernels = SlabArchKernels {
    arch: "gfx1151",
    plan_module: PLAN_MODULE_SHARED,
    plan_src: plan_src_shared,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1151_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[1],
    },
    k3: None,
    tile_mask: SlabTileMask::Hipcc {
        module: MASK_MODULE,
        source: kernels::QWEN4_SLAB_TILE_MASK_SRC,
        symbol: MASK_SYMBOL,
    },
};

static SLAB_COLD_GFX1201_W3: SlabCold = SlabCold {
    cold_codec: 1,
    k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[0],
    },
};

static SLAB_COLD_GFX1201_E8P: SlabCold = SlabCold {
    cold_codec: 3,
    k1: SlabK1 {
        module: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_MODULE,
        image: kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
        symbols: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_SYMBOLS,
    },
};

/// gfx1100 cold E8P K1 (`e8p_i4_moe_gemv_pm_gfx1100`), served for codec 3 once admitted.
static SLAB_COLD_GFX1100_E8P: SlabCold = SlabCold {
    cold_codec: 3,
    k1: SlabK1 {
        module: E8P_I4_MOE_GEMV_GFX1100_MODULE,
        image: kernels::E8P_I4_MOE_GEMV_PM_GFX1100,
        symbols: E8P_I4_MOE_GEMV_GFX1100_SYMBOLS,
    },
};

/// gfx1151 cold W3 K1: the same module and image as [`SLAB_GFX1151`]'s MQ4 bucket, the W3
/// (`MQ3I4`) symbols.
static SLAB_COLD_GFX1151_W3: SlabCold = SlabCold {
    cold_codec: 1,
    k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1151_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[0],
    },
};

/// gfx1151 cold E8P K1 (`e8p_i4_moe_gemv_pm_gfx1151`, the gfx1100 arm with only arch-forced
/// deltas), served for codec 3 once admitted.
static SLAB_COLD_GFX1151_E8P: SlabCold = SlabCold {
    cold_codec: 3,
    k1: SlabK1 {
        module: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_GFX1151_MODULE,
        image: kernels::E8P_I4_MOE_GEMV_PM_GFX1151,
        symbols: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_GFX1151_SYMBOLS,
    },
};

/// The per-arch slab table of `arch` (exact name), `None` off gfx1201/gfx1100/gfx1151.
pub fn slab_arch_kernels(arch: &str) -> Option<&'static SlabArchKernels> {
    match arch {
        "gfx1201" => Some(&SLAB_GFX1201),
        "gfx1100" => Some(&SLAB_GFX1100),
        "gfx1151" => Some(&SLAB_GFX1151),
        _ => None,
    }
}

/// The cold-bucket K1 table of cold codec `cold_codec` the sealed route SERVES on `arch`:
/// [`slab_cold_k1_metadata`] restricted to what [`crate::expert_capabilities::slab_serves_codec`]
/// admits, so metadata alone never launches or preloads a codec. `None` for MQ4 (0, never cold),
/// every codec the central table refuses on that arch, and W2 (2) everywhere. On gfx1100 only E8P
/// (3) resolves (its K1 [`SLAB_COLD_GFX1100_E8P`]; the matching K2 is `slab_k2_entry`); W3 (1) does
/// not. On gfx1151 W3 (1) resolves ([`SLAB_COLD_GFX1151_W3`]); E8P (3) resolves
/// ([`SLAB_COLD_GFX1151_E8P`]) only when the central table admits it.
pub fn slab_cold_kernels(arch: &str, cold_codec: u8) -> Option<&'static SlabCold> {
    if cold_codec == 0 || !crate::expert_capabilities::slab_serves_codec(arch, cold_codec) {
        return None;
    }
    slab_cold_k1_metadata(arch, cold_codec)
}

/// The cold K1 entries that exist per (`arch`, cold codec): gfx1201 W3 (1) and E8P (3), gfx1100
/// E8P (3, the qualified `e8p_i4_moe_gemv_pm_gfx1100` image), gfx1151 W3 (1) and E8P (3, the
/// `e8p_i4_moe_gemv_pm_gfx1151` image). This is inventory, NOT
/// permission: serving selection goes through [`slab_cold_kernels`] (admission-gated).
pub fn slab_cold_k1_metadata(arch: &str, cold_codec: u8) -> Option<&'static SlabCold> {
    match (arch, cold_codec) {
        ("gfx1201", 1) => Some(&SLAB_COLD_GFX1201_W3),
        ("gfx1201", 3) => Some(&SLAB_COLD_GFX1201_E8P),
        ("gfx1100", 3) => Some(&SLAB_COLD_GFX1100_E8P),
        ("gfx1151", 1) => Some(&SLAB_COLD_GFX1151_W3),
        ("gfx1151", 3) => Some(&SLAB_COLD_GFX1151_E8P),
        _ => None,
    }
}

/// Read-only provenance of the gfx1151 production one-row C1 image.
/// This is inventory, not codec admission or a request to select C1.
pub fn slab_c1_k1_metadata(arch: &str) -> Option<SlabK1> {
    (arch == "gfx1151").then_some(SlabK1 {
        module: MQ_I4_MOE_GEMV_C1_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151,
        symbols: MQ_I4_MOE_GEMV_C1_SYMBOLS,
    })
}

/// The architecture-specific rowtile image for an existing, admission-checked
/// bucket. Inventory alone does not grant codec admission.
pub fn slab_rowtile_k1(k1: &SlabK1) -> Option<SlabK1> {
    let (module, image, symbols) = match (k1.module, k1.symbols) {
        (MQ_I4_MOE_GEMV_GFX1151_MODULE, symbols) if MQ_I4_MOE_GEMV_SYMBOLS.contains(&symbols) => {
            let bucket = usize::from(symbols == MQ_I4_MOE_GEMV_SYMBOLS[1]);
            (
                MQ_I4_MOE_GEMV_CN_MODULE,
                kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
                MQ_I4_MOE_GEMV_CN_ROWTILE_SYMBOLS[bucket],
            )
        }
        (MQ_I4_MOE_GEMV_MODULE, symbols) if MQ_I4_MOE_GEMV_SYMBOLS.contains(&symbols) => {
            let bucket = usize::from(symbols == MQ_I4_MOE_GEMV_SYMBOLS[1]);
            (k1.module, k1.image, MQ_I4_MOE_GEMV_ROWTILE_SYMBOLS[bucket])
        }
        (MQ_I4_MOE_GEMV_GFX1100_MODULE, MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS) => (
            k1.module,
            k1.image,
            [
                "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100_t8",
                "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100_t8",
            ],
        ),
        ("e8p_i4_moe_gemv_pm_gfx1201", _) => (
            k1.module,
            k1.image,
            [
                "e8p_i4_moe_gemv_r256_gate_up_t8",
                "e8p_i4_moe_gemv_r128_down_t8",
            ],
        ),
        (E8P_I4_MOE_GEMV_GFX1100_MODULE, _) => (
            k1.module,
            k1.image,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100_t8",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1100_t8",
            ],
        ),
        ("e8p_i4_moe_gemv_pm_gfx1151", _) => (
            k1.module,
            k1.image,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151_t8",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1151_t8",
            ],
        ),
        _ => return None,
    };
    Some(SlabK1 {
        module,
        image,
        symbols,
    })
}

/// Read-only PM planner provenance. The module key and export have the same
/// spelling. Availability is inventory only, never an architecture admission.
pub fn slab_rowtile_planner(arch: &str) -> Option<(&'static str, &'static [u8])> {
    match arch {
        "gfx1151" => Some((
            "qwen4_slab_union_plan_t8_pm_gfx1151",
            kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1151,
        )),
        "gfx1201" => Some((
            "qwen4_slab_union_plan_t8_pm_gfx1201",
            kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1201,
        )),
        "gfx1100" => Some((
            "qwen4_slab_union_plan_t8_pm_gfx1100",
            kernels::QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1100,
        )),
        _ => None,
    }
}

fn slab_check_rowtile_pin(image: &[u8], expected: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    if expected.is_empty() {
        return Err("slab rowtile image SHA256 is not pinned".into());
    }
    let actual = format!("{:x}", Sha256::digest(image));
    if actual != expected {
        return Err(format!(
            "slab rowtile image SHA256 {actual} differs from pinned {expected}"
        ));
    }
    Ok(())
}

/// Independent qualification-pin check, once per image actually requested.
/// Module-cache identity checks still run on every embedded-kernel binding.
fn slab_verify_rowtile_pin(module: &str) -> HipResult<()> {
    macro_rules! verify {
        ($image:ident, $pin:ident) => {{
            static VERIFIED: std::sync::LazyLock<Result<(), String>> =
                std::sync::LazyLock::new(|| slab_check_rowtile_pin(kernels::$image, kernels::$pin));
            VERIFIED
                .as_ref()
                .map(|_| ())
                .map_err(|reason| hip_bridge::HipError::new(1, reason))
        }};
    }
    match module {
        "qwen4_slab_union_plan_t8_pm_gfx1151" => verify!(
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1151,
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1151_SHA256
        ),
        "qwen4_slab_union_plan_t8_pm_gfx1201" => verify!(
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1201,
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1201_SHA256
        ),
        "qwen4_slab_union_plan_t8_pm_gfx1100" => verify!(
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1100,
            QWEN4_SLAB_UNION_PLAN_T8_PM_GFX1100_SHA256
        ),
        MQ_I4_MOE_GEMV_CN_MODULE => verify!(
            MQ_I4_MOE_GEMV_CN_PM_GFX1151,
            MQ_I4_MOE_GEMV_CN_PM_GFX1151_SHA256
        ),
        MQ_I4_MOE_GEMV_MODULE => {
            verify!(MQ_I4_MOE_GEMV_PM_GFX1201, MQ_I4_MOE_GEMV_PM_GFX1201_SHA256)
        }
        MQ_I4_MOE_GEMV_GFX1100_MODULE => verify!(
            MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
            MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100_SHA256
        ),
        "e8p_i4_moe_gemv_pm_gfx1201" => verify!(
            E8P_I4_MOE_GEMV_PM_GFX1201,
            E8P_I4_MOE_GEMV_PM_GFX1201_SHA256
        ),
        "e8p_i4_moe_gemv_pm_gfx1151" => verify!(
            E8P_I4_MOE_GEMV_PM_GFX1151,
            E8P_I4_MOE_GEMV_PM_GFX1151_SHA256
        ),
        E8P_I4_MOE_GEMV_GFX1100_MODULE => verify!(
            E8P_I4_MOE_GEMV_PM_GFX1100,
            E8P_I4_MOE_GEMV_PM_GFX1100_SHA256
        ),
        _ => Err(hip_bridge::HipError::new(
            1,
            "slab rowtile image has no qualification pin",
        )),
    }
}

fn slab_plan_geometry(lanes: usize, tiled: bool) -> Option<([u32; 3], [u32; 3], i32)> {
    if lanes == 0 || lanes > SLAB_MAX_LANES || (!tiled && lanes > SLAB_NARROW_MAX_LANES) {
        return None;
    }
    Some(if tiled {
        ([lanes.div_ceil(8) as u32, 1, 1], [128, 1, 1], lanes as i32)
    } else {
        ([1, 1, 1], [512, 1, 1], (SLAB_TOP_K * lanes) as i32)
    })
}

fn slab_wide_capable(arch: &SlabArchKernels, cold: Option<&SlabCold>) -> bool {
    slab_rowtile_planner(arch.arch).is_some()
        && slab_rowtile_k1(&arch.mq4_k1).is_some()
        && cold.is_none_or(|cold| slab_rowtile_k1(&cold.k1).is_some())
}

fn slab_rowtile_pick(k1: &SlabK1, down: bool, m: usize, lanes: usize) -> Option<SlabK1Pick> {
    if lanes == 0 || lanes > SLAB_MAX_LANES || m == 0 || m > i32::MAX as usize {
        return None;
    }
    let tile = slab_rowtile_k1(k1)?;
    Some(SlabK1Pick {
        module_key: tile.module,
        image: tile.image,
        symbol: tile.symbols[down as usize],
        grid: [m.div_ceil(8) as u32, 80, lanes.div_ceil(8) as u32],
        block: [256, 1, 1],
    })
}

/// Descriptor/allocation capacity, not the architecture's row admission.
pub const SLAB_MAX_LANES: usize = 64;
/// Capacity of the unchanged decode K1 entries and one physical row tile.
pub const SLAB_NARROW_MAX_LANES: usize = 8;
/// Routed experts per lane.
pub const SLAB_TOP_K: usize = 10;
/// Experts per layer.
pub const SLAB_EXPERTS: usize = 512;
/// Gate/up output rows (gate 640 + up 640), reduction width.
pub const SLAB_GATE_UP_M: usize = 1280;
pub const SLAB_GATE_UP_K: usize = 2560;
/// Down output rows and reduction width.
pub const SLAB_DOWN_M: usize = 2560;
pub const SLAB_DOWN_K: usize = 640;

/// Bucket of codec byte `0` (MQ4) and `1` (W3) in the gfx1201 K1 order: W3 first.
pub const fn slab_bucket_of_codec(codec: u8) -> usize {
    if codec == 1 {
        0
    } else {
        1
    }
}

/// Bucket of codec id `codec` on `arch` (cold = 0, MQ4 = 1), `None` when the
/// slab does not serve that codec there ([`crate::expert_capabilities::slab_serves_codec`]).
pub fn slab_bucket_of_codec_on(arch: &str, codec: u8) -> Option<usize> {
    slab_arch_kernels(arch)?;
    if !crate::expert_capabilities::slab_serves_codec(arch, codec) {
        return None;
    }
    Some(usize::from(codec == 0))
}

/// How a layer's experts map to the slab's two buckets (cold = 0, MQ4 = 1).
#[derive(Clone, Copy)]
pub enum SlabBuckets<'a> {
    /// Every expert is in one bucket: `0` = the layer's cold codec (W3 or E8P), `1` = MQ4. Only
    /// that bucket launches.
    Uniform(usize),
    /// Per-expert bucket MEMBERSHIP (`[512]` u8), not the real codec id: `1` = cold, `0` = MQ4.
    /// Both buckets launch. (The tile mask's i32 table is the one that holds raw codec ids.)
    Table(&'a GpuTensor),
}

impl Gpu {
    /// Whether product dispatch admits a rowtile K1 launch for this layer.
    /// Requires the shared exact-row policy, the K1 family selector and an
    /// architecture-specific tile image for every admission-checked bucket.
    pub fn qwen4_slab_wide_twin(&self, lanes: usize, cold_codec: u8) -> bool {
        let Some(arch) = slab_arch_kernels(&self.arch) else {
            return false;
        };
        let cold = match cold_codec {
            0 => None,
            codec => match slab_cold_kernels(&self.arch, codec) {
                Some(cold) => Some(cold),
                None => return false,
            },
        };
        self.row_tile_twin_enabled(WideTwinFamily::K1, lanes) && slab_wide_capable(arch, cold)
    }

    /// Load the sealed slab's kernels (call before graph capture) for a layer whose cold bucket
    /// is codec `cold_codec`: `0` = no cold bucket (an MQ4-only layer, any arch), gfx1201
    /// 1 = W3 or 3 = E8P, gfx1100 3 = E8P, gfx1151 1 = W3 or 3 = E8P (E8P via the receipts switch); every
    /// other value (W2 / codec 2 everywhere, W3 on gfx1100, E8P on gfx1151 with the switch unset, unknown codecs)
    /// is refused by the shared admission matrix before
    /// anything loads. Loads the planner and,
    /// on gfx1201, the cold (if any) and MQ4 K1 symbols (the historical W3 preload, unchanged);
    /// on gfx1100 the planner and every serving entry of
    /// [`crate::kernel_registry::slab_embedded_entries`]`("gfx1100", cold_codec)`: MQ4 K1, MQ4 K2
    /// NT1 GEMMs, K3 and the tile mask, plus, for E8P layers only, the cold K1 and the E8P K2
    /// NT1 GEMMs (an MQ4-only layer loads no E8P module). Each entry is ensured once.
    ///
    /// gfx1151 exception to the no-cold-preload rule (gfx1100 and gfx1201 keep it): beta's preload
    /// is unconditional, so even `cold_codec == 0` loads the planner, the HIP tile mask and all four
    /// K1 handles (W3 gate/up, W3 down, MQ4 gate/up, MQ4 down, in that order) of the already-qualified
    /// shared image. That prepares the image only: it is not a W3 bucket launch
    /// ([`Gpu::mq_i4_moe_gemv_slab`] still refuses bucket 0 when `cold_codec == 0`) and admits no other
    /// codec. K2 loading stays lazy.
    pub fn ensure_qwen4_slab_kernels(&mut self, cold_codec: u8) -> HipResult<()> {
        let arch = self.slab_kernels("ensure_qwen4_slab_kernels")?;
        let cold = self.slab_cold_opt("ensure_qwen4_slab_kernels", cold_codec)?;
        self.bind_thread()?;
        self.ensure_kernel(arch.plan_module, (arch.plan_src)(), PLAN_SYMBOL)?;
        match arch.arch {
            "gfx1100" => {
                let inventory = crate::kernel_registry::slab_embedded_entries(
                    arch.arch, cold_codec,
                )
                .map_err(|e| {
                    hip_bridge::HipError::new(0, &format!("ensure_qwen4_slab_kernels: {e:?}"))
                })?;
                for planned in inventory {
                    let crate::kernel_registry::PlannedKernel::Embedded {
                        module,
                        image,
                        symbols,
                        ..
                    } = planned
                    else {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "ensure_qwen4_slab_kernels: the embedded slab inventory holds a hipcc entry",
                        ));
                    };
                    for symbol in symbols {
                        self.ensure_embedded_kernel(module, image, symbol)?;
                    }
                }
            }
            "gfx1151" => {
                // Beta's gfx1151 preload, unconditional (even `cold_codec == 0`): the planner (ensured
                // above), the HIP tile mask, then all four handles of the shared K1 image in beta order
                // (W3 gate/up, W3 down, MQ4 gate/up, MQ4 down). The cold codec was already admission-checked
                // (only 0, W3 and, once admitted, E8P reach here); this prepares the shared image and is not a W3 bucket launch.
                match arch.tile_mask {
                    SlabTileMask::Hipcc {
                        module,
                        source,
                        symbol,
                    } => self.ensure_kernel(module, source, symbol)?,
                    SlabTileMask::Embedded { .. } => {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "ensure_qwen4_slab_kernels: the gfx1151 tile mask is a hipcc source, not an embedded image",
                        ));
                    }
                }
                for k1 in [&SLAB_COLD_GFX1151_W3.k1, &arch.mq4_k1] {
                    for symbol in k1.symbols {
                        self.ensure_embedded_kernel(k1.module, k1.image, symbol)?;
                    }
                }
                // An admitted E8P layer (cold codec 3) additionally binds the gfx1151 E8P K1 image's
                // two symbols. The cold table is `None` for every codec the matrix refuses, so this
                // is inert until gfx1151 E8P is admitted; K2 stays lazy like every gfx1151 K2.
                if let Some(cold) = cold.filter(|cold| cold.cold_codec == 3) {
                    for symbol in cold.k1.symbols {
                        self.ensure_embedded_kernel(cold.k1.module, cold.k1.image, symbol)?;
                    }
                }
            }
            _ => {
                for k1 in cold
                    .map(|cold| &cold.k1)
                    .into_iter()
                    .chain(std::iter::once(&arch.mq4_k1))
                {
                    for symbol in k1.symbols {
                        self.ensure_embedded_kernel(k1.module, k1.image, symbol)?;
                    }
                }
            }
        }
        if WideTwinFamily::K1.enabled() && slab_wide_capable(arch, cold) {
            let (module, image) = slab_rowtile_planner(arch.arch).expect("tile-capable planner");
            slab_verify_rowtile_pin(module)?;
            self.ensure_embedded_kernel(module, image, module)?;
            for k1 in cold
                .map(|cold| &cold.k1)
                .into_iter()
                .chain(std::iter::once(&arch.mq4_k1))
            {
                let tile = slab_rowtile_k1(k1).expect("tile-capable bucket");
                slab_verify_rowtile_pin(tile.module)?;
                for symbol in tile.symbols {
                    self.ensure_embedded_kernel(tile.module, tile.image, symbol)?;
                }
            }
        }
        if self.slab_k1_c1() && k1_rows8_cold_codec(cold_codec) {
            for symbol in MQ_I4_MOE_GEMV_C1_SYMBOLS {
                self.ensure_embedded_kernel(
                    MQ_I4_MOE_GEMV_C1_MODULE,
                    kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151,
                    symbol,
                )?;
            }
        }
        if self.slab_k1_cn() && k1_rows8_cold_codec(cold_codec) {
            for symbol in MQ_I4_MOE_GEMV_CN_SYMBOLS.iter().flatten() {
                self.ensure_embedded_kernel(
                    MQ_I4_MOE_GEMV_CN_MODULE,
                    kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151,
                    symbol,
                )?;
            }
        }
        Ok(())
    }

    /// The sealed slab's per-arch module table of this device, or a refusal for every
    /// arch outside [`slab_arch_kernels`].
    fn slab_kernels(&self, what: &str) -> HipResult<&'static SlabArchKernels> {
        slab_arch_kernels(&self.arch).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "{what}: the sealed expert-slab route is gfx1100/gfx1151/gfx1201-only, got {}",
                    self.arch
                ),
            )
        })
    }

    /// The cold-bucket K1 table of cold codec `cold_codec` on this device
    /// ([`slab_cold_kernels`]); refuses codec 0 (no cold codec) and every codec the shared
    /// admission matrix refuses (W2 / codec 2 everywhere, W3 on gfx1100: a tabled K1 image,
    /// [`slab_cold_k1_metadata`], is not permission).
    fn slab_cold(&self, what: &str, cold_codec: u8) -> HipResult<&'static SlabCold> {
        slab_cold_kernels(&self.arch, cold_codec).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "{what}: cold codec {cold_codec} has no sealed expert-slab entry on {}",
                    self.arch
                ),
            )
        })
    }

    /// [`Gpu::slab_cold`] where `cold_codec == 0` means "no cold bucket" (an MQ4-only layer):
    /// `Ok(None)`.
    fn slab_cold_opt(&self, what: &str, cold_codec: u8) -> HipResult<Option<&'static SlabCold>> {
        if cold_codec == 0 {
            return Ok(None);
        }
        self.slab_cold(what, cold_codec).map(Some)
    }

    /// Reserve the slab route's scratch (union descriptor for `SLAB_MAX_LANES`,
    /// and `tiles` masked tile ids per codec) before any capture.
    pub fn reserve_qwen4_slab(&mut self, tiles: usize) -> HipResult<()> {
        self.slab_kernels("reserve_qwen4_slab")?;
        // Preflight every extent before invalidating replay or changing storage.
        let union_bytes =
            crate::scratch::qwen4_slab_descriptor_bytes(SLAB_MAX_LANES).ok_or_else(|| {
                hip_bridge::HipError::new(0, "reserve_qwen4_slab: descriptor extent overflow")
            })?;
        let tile_bytes = crate::scratch::qwen4_slab_tiles_bytes(tiles).ok_or_else(|| {
            hip_bridge::HipError::new(0, "reserve_qwen4_slab: tile mask extent overflow")
        })?;
        self.bind_thread()?;
        if crate::scratch::scratch_will_grow(
            self.scratch.qwen4_slab_union_scratch_bytes,
            self.scratch.qwen4_slab_union_scratch.is_some(),
            union_bytes,
        ) || (tiles != 0
            && crate::scratch::scratch_will_grow(
                self.scratch.qwen4_slab_tiles_scratch_bytes,
                self.scratch.qwen4_slab_tiles_scratch.is_some(),
                tile_bytes,
            ))
        {
            self.invalidate_for_scratch_growth();
        }
        self.scratch
            .reserve_qwen4_slab(&self.hip, SLAB_MAX_LANES, tiles)
    }

    fn slab_union_base(&self) -> HipResult<*mut c_void> {
        self.scratch
            .qwen4_slab_union_scratch
            .as_ref()
            .map(|buf| buf.as_ptr())
            .ok_or_else(|| hip_bridge::HipError::new(0, "slab union scratch is not reserved"))
    }

    /// Raw descriptor view for byte oracles; invalidated by the next union plan
    /// or pointer-changing scratch growth. `rowtiles` must match the planner used.
    pub fn qwen4_slab_union_descriptor_view(
        &self,
        lanes: usize,
        rowtiles: bool,
    ) -> HipResult<GpuTensor> {
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(
                0,
                "slab descriptor view: lanes must be 1..=64",
            ));
        }
        let bytes = if rowtiles {
            crate::scratch::qwen4_slab_rowtile_bytes(lanes)
        } else {
            crate::scratch::qwen4_slab_union_bytes(lanes)
        }
        .ok_or_else(|| hip_bridge::HipError::new(0, "slab descriptor view: extent overflow"))?;
        let buf = self
            .scratch
            .qwen4_slab_union_scratch
            .as_ref()
            .filter(|buf| buf.size() >= bytes)
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "slab descriptor view: scratch not reserved")
            })?;
        Ok(GpuTensor {
            buf: buf.byte_view(0, bytes),
            shape: vec![bytes / 4],
            dtype: DType::F32,
        })
    }

    /// Plan the lane union of `lanes` decode lanes from `topk` (`[lanes * 10]`
    /// i32 routed ids, slot = lane * 10 + rank) into the slab union scratch.
    /// Stream-ordered; no readback.
    pub fn qwen4_slab_union_plan(
        &mut self,
        topk: &GpuTensor,
        buckets: SlabBuckets<'_>,
        lanes: usize,
    ) -> HipResult<()> {
        self.slab_union_plan_with(topk, buckets, lanes, lanes > SLAB_NARROW_MAX_LANES)
    }

    /// Direct byte-oracle planner for the tiled layout at every R=1..64.
    /// Pair with `mq_i4_moe_gemv_slab_with(..., Some(SlabK1Module::RowTiles))`.
    pub fn qwen4_slab_union_plan_rowtiles(
        &mut self,
        topk: &GpuTensor,
        buckets: SlabBuckets<'_>,
        lanes: usize,
    ) -> HipResult<()> {
        self.slab_union_plan_with(topk, buckets, lanes, true)
    }

    fn slab_union_plan_with(
        &mut self,
        topk: &GpuTensor,
        buckets: SlabBuckets<'_>,
        lanes: usize,
        tiled: bool,
    ) -> HipResult<()> {
        let arch = self.slab_kernels("qwen4_slab_union_plan")?;
        let (grid, block, sv) = slab_plan_geometry(lanes, tiled).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                "qwen4_slab_union_plan: invalid row count or planner geometry",
            )
        })?;
        let slots = SLAB_TOP_K * lanes;
        if topk.buf.size() < slots * 4 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_slab_union_plan: topk holds fewer than 10 * lanes ids",
            ));
        }
        let (codec_ptr, uniform) = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => {
                (std::ptr::null_mut::<c_void>(), bucket as i32)
            }
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qwen4_slab_union_plan: bucket must be 0 or 1",
                ))
            }
            SlabBuckets::Table(table) => {
                if table.buf.size() < SLAB_EXPERTS {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "qwen4_slab_union_plan: codec table holds fewer than 512 bytes",
                    ));
                }
                (table.buf.as_ptr(), -1)
            }
        };
        self.bind_thread()?;
        let symbol = if tiled {
            let (module, image) = slab_rowtile_planner(arch.arch).ok_or_else(|| {
                hip_bridge::HipError::new(0, "qwen4_slab_union_plan: no rowtile planner")
            })?;
            slab_verify_rowtile_pin(module)?;
            self.ensure_embedded_kernel(module, image, module)?;
            module
        } else {
            // slab_plan_geometry limits this to <=80 slots, never >320.
            self.ensure_kernel(arch.plan_module, (arch.plan_src)(), PLAN_SYMBOL)?;
            PLAN_SYMBOL
        };
        if self.scratch.qwen4_slab_union_scratch_bytes
            < crate::scratch::qwen4_slab_descriptor_bytes(SLAB_MAX_LANES)
                .expect("bounded slab extent")
            || self.scratch.qwen4_slab_union_scratch.is_none()
        {
            self.reserve_qwen4_slab(0)?;
        }
        let out = self.slab_union_base()?;
        let tp = topk.buf.as_ptr();
        let nexp = SLAB_EXPERTS as i32;
        let mut params = [
            &tp as *const _ as *mut c_void,
            &codec_ptr as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
            &nexp as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(symbol, grid, block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(codec_ptr);
            b.push_ptr(out);
            b.push_i32(sv);
            b.push_i32(uniform);
            b.push_i32(nexp);
            b
        })
    }

    /// Launch one K1 projection over every non-empty bucket of the union
    /// [`Gpu::qwen4_slab_union_plan`] wrote for `lanes` lanes.
    ///
    /// Gate/up (`down == false`): `xrot` `[lanes, 2560]` F32 rotated
    /// activations, `y0`/`y1` `[lanes, 10, 640]` F32 gate/up. Down: `xrot`
    /// `[10 * lanes, 640]` F32, `y0` `[lanes, 10, 2560]` F32 (unweighted),
    /// `y1` `None`. `weight_ptrs` is the layer's `[512]` u64 pointer table of
    /// the projection. Output rows of every slot are written exactly once. The 96-byte kernarg
    /// ends `n_experts = SLAB_EXPERTS` and a zero reserved word; an expert id outside
    /// `[0, n_experts)` skips its workgroup without any load or store: a fault guard for a
    /// corrupt or non-finite-router id, not a routing rule.
    #[allow(clippy::too_many_arguments)]
    pub fn mq_i4_moe_gemv_slab(
        &mut self,
        down: bool,
        buckets: SlabBuckets<'_>,
        weight_ptrs: &GpuTensor,
        xrot: &GpuTensor,
        y0: &GpuTensor,
        y1: Option<&GpuTensor>,
        lanes: usize,
        cold_codec: u8,
    ) -> HipResult<()> {
        self.mq_i4_moe_gemv_slab_with(
            down,
            buckets,
            weight_ptrs,
            xrot,
            y0,
            y1,
            lanes,
            cold_codec,
            None,
        )
    }

    /// [`Gpu::mq_i4_moe_gemv_slab`] with an optionally forced oracle schedule.
    /// Product dispatch uses rowtiles above eight rows, Cn on gfx1151 MQ4/W3
    /// at 2..=8 when enabled, and the incumbent K1 schedule otherwise.
    /// Forcing `Cn` is an error off gfx1151, above 8 lanes and for any other cold codec; the oracle
    /// forces both modules over the same union plan to compare them bitwise.
    /// Explicit `RowTiles` bypasses the product family selector, not codec
    /// admission; use the rowtile planner even at R<=8 for that oracle arm.
    #[allow(clippy::too_many_arguments)]
    pub fn mq_i4_moe_gemv_slab_with(
        &mut self,
        down: bool,
        buckets: SlabBuckets<'_>,
        weight_ptrs: &GpuTensor,
        xrot: &GpuTensor,
        y0: &GpuTensor,
        y1: Option<&GpuTensor>,
        lanes: usize,
        cold_codec: u8,
        force: Option<SlabK1Module>,
    ) -> HipResult<()> {
        let arch = self.slab_kernels("mq_i4_moe_gemv_slab")?;
        let cold = self.slab_cold_opt("mq_i4_moe_gemv_slab", cold_codec)?;
        let gfx1151 = arch.arch == "gfx1151";
        let module = force.unwrap_or_else(|| {
            slab_k1_default_module(
                gfx1151,
                lanes,
                self.slab_k1_cn() && k1_rows8_cold_codec(cold_codec),
            )
        });
        if module == SlabK1Module::Cn && !gfx1151 {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: the K1-Cn kernels are gfx1151-only",
            ));
        }
        if module == SlabK1Module::Cn && lanes > SLAB_NARROW_MAX_LANES {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: the K1-Cn kernels serve 1..=8 lanes",
            ));
        }
        if module == SlabK1Module::Cn && !k1_rows8_cold_codec(cold_codec) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("mq_i4_moe_gemv_slab: the K1-Cn kernels serve MQ4/W3 layers only, got cold codec {cold_codec}"),
            ));
        }
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: lanes must be 1..=64",
            ));
        }
        let direct_tiles = module == SlabK1Module::RowTiles;
        let wide = lanes > SLAB_NARROW_MAX_LANES || direct_tiles;
        if wide && !slab_wide_capable(arch, cold) {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: a bucket has no architecture-specific rowtile image",
            ));
        }
        if lanes > SLAB_NARROW_MAX_LANES
            && !direct_tiles
            && !self.qwen4_slab_wide_twin(lanes, cold_codec)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: wide lanes require admitted K1 rowtiles",
            ));
        }
        let slots = SLAB_TOP_K * lanes;
        let (m, k, x_row_div, x_src_rows, y_stride) = if down {
            (SLAB_DOWN_M, SLAB_DOWN_K, 1usize, slots, SLAB_DOWN_M)
        } else {
            (
                SLAB_GATE_UP_M,
                SLAB_GATE_UP_K,
                SLAB_TOP_K,
                lanes,
                SLAB_GATE_UP_M / 2,
            )
        };
        if down == y1.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: gate/up needs the up output, down has none",
            ));
        }
        if weight_ptrs.buf.size() < SLAB_EXPERTS * 8
            || xrot.buf.size() < x_src_rows * k * 4
            || y0.buf.size() < slots * y_stride * 4
            || y1.is_some_and(|y| y.buf.size() < slots * y_stride * 4)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: operand smaller than the launch extent",
            ));
        }
        let present: [bool; 2] = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => [bucket == 0, bucket == 1],
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "mq_i4_moe_gemv_slab: bucket must be 0 or 1",
                ))
            }
            SlabBuckets::Table(table) => {
                if table.buf.size() < SLAB_EXPERTS {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "mq_i4_moe_gemv_slab: short codec table",
                    ));
                }
                [true, true]
            }
        };
        if present[0] && cold.is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: cold bucket requested without an admitted cold codec",
            ));
        }
        let descriptor_bytes = if wide {
            crate::scratch::qwen4_slab_rowtile_bytes(lanes)
        } else {
            crate::scratch::qwen4_slab_union_bytes(lanes)
        }
        .expect("bounded descriptor extent");
        if self.scratch.qwen4_slab_union_scratch_bytes < descriptor_bytes {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: descriptor scratch too small",
            ));
        }
        self.bind_thread()?;
        let union = self.slab_union_base()? as usize;
        let s = if wide {
            SLAB_TOP_K * SLAB_NARROW_MAX_LANES
        } else {
            slots
        };
        let wp = weight_ptrs.buf.as_ptr();
        let xp = xrot.buf.as_ptr();
        let y0p = y0.buf.as_ptr();
        let y1p = y1.map_or(std::ptr::null_mut::<c_void>(), |y| y.buf.as_ptr());
        for (bucket, present) in present.into_iter().enumerate() {
            if !present {
                continue;
            }
            let k1 = if bucket == 0 {
                &cold
                    .ok_or_else(|| {
                        hip_bridge::HipError::new(
                            0,
                            "mq_i4_moe_gemv_slab: the cold bucket launches but cold_codec is 0 (no cold codec)",
                        )
                    })?
                    .k1
            } else {
                &arch.mq4_k1
            };
            let pick = if wide {
                slab_rowtile_pick(k1, down, m, lanes).expect("validated rowtile launch")
            } else {
                slab_k1_pick(k1, module, bucket, down, m, s)
            };
            let func = pick.symbol;
            if wide {
                slab_verify_rowtile_pin(pick.module_key)?;
            }
            self.ensure_embedded_kernel(pick.module_key, pick.image, func)?;
            let ids = (union + bucket * s * 4) as *mut c_void;
            let offs = (union + (2 * s + bucket * (s + 1)) * 4) as *mut c_void;
            let slot_ids = (union + (2 * s + 2 * (s + 1) + bucket * s) * 4) as *mut c_void;
            let (mv, kv, uv, cv) = (m as u32, k as u32, s as u32, lanes as u32);
            let (dv, srows, xs, ys) = (
                x_row_div as u32,
                x_src_rows as u32,
                k as u32,
                y_stride as u32,
            );
            let nexp = SLAB_EXPERTS as u32;
            // Metadata lists reserved@92 as argument 17, not implicit struct padding.
            let reserved = 0u32;
            let mut params: [*mut c_void; MQ_I4_MOE_GEMV_KERNARG_COUNT] = [
                &wp as *const _ as *mut c_void,
                &ids as *const _ as *mut c_void,
                &offs as *const _ as *mut c_void,
                &slot_ids as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &y0p as *const _ as *mut c_void,
                &y1p as *const _ as *mut c_void,
                &mv as *const _ as *mut c_void,
                &kv as *const _ as *mut c_void,
                &uv as *const _ as *mut c_void,
                &cv as *const _ as *mut c_void,
                &dv as *const _ as *mut c_void,
                &srows as *const _ as *mut c_void,
                &xs as *const _ as *mut c_void,
                &ys as *const _ as *mut c_void,
                &nexp as *const _ as *mut c_void,
                &reserved as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(func, pick.grid, pick.block, 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(ids);
                b.push_ptr(offs);
                b.push_ptr(slot_ids);
                b.push_ptr(xp);
                b.push_ptr(y0p);
                b.push_ptr(y1p);
                for word in [mv, kv, uv, cv, dv, srows, xs, ys, nexp, 0] {
                    b.push_u32(word);
                }
                b
            })?;
        }
        Ok(())
    }

    /// Whether the slab decode uses the K1-C1 kernels at `lanes == 1`: exact gfx1151 and
    /// `HIPFIRE_QWEN4_SLAB_K1_C1` not `0` (default on). The environment is read once per process.
    pub fn slab_k1_c1(&self) -> bool {
        static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_bool("HIPFIRE_QWEN4_SLAB_K1_C1", true)
        });
        self.arch == "gfx1151" && *ENABLED
    }

    /// Whether the multi-lane slab decode (`2 <= lanes <= 8`, MTP verify) uses the K1-Cn kernels:
    /// exact gfx1151 and `HIPFIRE_QWEN4_SLAB_K1_CN` not `0` (default on). The environment is read
    /// once per process.
    pub fn slab_k1_cn(&self) -> bool {
        static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_bool("HIPFIRE_QWEN4_SLAB_K1_CN", true)
        });
        self.arch == "gfx1151" && *ENABLED
    }

    /// K1-C1: one projection of the one-lane (`lanes == 1`, AR decode) slab decode in ONE launch
    /// for both codecs, straight from `topk` (no union plan, no union scratch). Bitwise the K1
    /// result of [`Gpu::mq_i4_moe_gemv_slab`] at `lanes == 1` over the plan of the same `topk`.
    ///
    /// Operand shapes are the `lanes == 1` shapes of [`Gpu::mq_i4_moe_gemv_slab`]: `topk` holds the
    /// layer's `[10]` i32 routed ids, gate/up `xrot` `[2560]` F32 and `y0`/`y1` `[10, 640]` F32,
    /// down `xrot` `[10, 640]` F32, `y0` `[10, 2560]` F32 unweighted, `y1` `None`.
    /// [`SlabBuckets::Uniform`] names the one codec (0 = W3, 1 = MQ4: the K1 bucket index);
    /// [`SlabBuckets::Table`] is the `[512]` u8 codec table. The layer's cold codec must be `0` or
    /// `1` ([`k1_rows8_cold_codec`]); a W3 bucket needs `1`. gfx1151 only: any other device errors.
    #[allow(clippy::too_many_arguments)]
    pub fn mq_i4_moe_gemv_slab_c1(
        &mut self,
        down: bool,
        buckets: SlabBuckets<'_>,
        weight_ptrs: &GpuTensor,
        topk: &GpuTensor,
        xrot: &GpuTensor,
        y0: &GpuTensor,
        y1: Option<&GpuTensor>,
        cold_codec: u8,
    ) -> HipResult<()> {
        if self.arch != "gfx1151" {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "mq_i4_moe_gemv_slab_c1: the K1-C1 kernels are gfx1151-only, got {}",
                    self.arch
                ),
            ));
        }
        if !k1_rows8_cold_codec(cold_codec) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("mq_i4_moe_gemv_slab_c1: K1-C1 serves MQ4/W3 layers only, got cold codec {cold_codec}"),
            ));
        }
        let geo = slab_c1_launch(down);
        let slots = SLAB_TOP_K;
        if down == y1.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab_c1: gate/up needs the up output, down has none",
            ));
        }
        let x_src_rows = if down { slots } else { 1 };
        if weight_ptrs.buf.size() < SLAB_EXPERTS * 8
            || topk.buf.size() < slots * 4
            || xrot.buf.size() < x_src_rows * geo.k as usize * 4
            || y0.buf.size() < slots * geo.y_slot_stride as usize * 4
            || y1.is_some_and(|y| y.buf.size() < slots * geo.y_slot_stride as usize * 4)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab_c1: operand smaller than the launch extent",
            ));
        }
        let (codec_ptr, uniform) = match buckets {
            // K1 bucket 0 = W3 = codec 1, bucket 1 = MQ4 = codec 0.
            SlabBuckets::Uniform(0) if cold_codec == 0 => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "mq_i4_moe_gemv_slab_c1: the cold bucket launches but cold_codec is 0 (no cold codec)",
                ))
            }
            SlabBuckets::Uniform(bucket) if bucket < 2 => (std::ptr::null_mut::<c_void>(), 1 - bucket as i32),
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(0, "mq_i4_moe_gemv_slab_c1: bucket must be 0 or 1"))
            }
            SlabBuckets::Table(table) => {
                if table.buf.size() < SLAB_EXPERTS {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "mq_i4_moe_gemv_slab_c1: codec table holds fewer than 512 bytes",
                    ));
                }
                (table.buf.as_ptr(), -1)
            }
        };
        let nexp = SLAB_EXPERTS as u32;
        self.bind_thread()?;
        let func = MQ_I4_MOE_GEMV_C1_SYMBOLS[down as usize];
        self.ensure_embedded_kernel(
            MQ_I4_MOE_GEMV_C1_MODULE,
            kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151,
            func,
        )?;
        let wp = weight_ptrs.buf.as_ptr();
        let tp = topk.buf.as_ptr();
        let xp = xrot.buf.as_ptr();
        let y0p = y0.buf.as_ptr();
        let y1p = y1.map_or(std::ptr::null_mut::<c_void>(), |y| y.buf.as_ptr());
        let (mv, kv, dv, xs, ys) = (geo.m, geo.k, geo.x_row_div, geo.k, geo.y_slot_stride);
        let reserved = 0u32;
        let mut params: [*mut c_void; MQ_I4_MOE_GEMV_C1_KERNARG_COUNT] = [
            &wp as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &codec_ptr as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &y0p as *const _ as *mut c_void,
            &y1p as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &ys as *const _ as *mut c_void,
            &nexp as *const _ as *mut c_void,
            &reserved as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(func, geo.grid, geo.block, 0, &mut params, || {
            slab_c1_kernarg(wp, tp, codec_ptr, xp, y0p, y1p, uniform, nexp, &geo)
        })
    }

    /// Split the grouped tile ids `tile_ids` (`tiles` words) by codec with the
    /// table-driven tile mask: `[cold tiles, MQ4 tiles]`, each the source ids with
    /// every tile of an expert of the other codec set to `-1`. `codec_i32` is the
    /// layer's `[512]` i32 codec table of raw codec ids (`0` = MQ4); the masks use ranks
    /// `cold_codec` (gfx1201: 1 = W3 or 3 = E8P; gfx1100: 3 = E8P; gfx1151: 1 = W3, 3 = E8P when
    /// admitted) and `0`.
    /// The outputs are views of the slab tile scratch, valid until the next call.
    pub fn qwen4_slab_tile_masks(
        &mut self,
        tile_ids: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
        cold_codec: u8,
    ) -> HipResult<[GpuTensor; 2]> {
        let arch = self.slab_kernels("qwen4_slab_tile_masks")?;
        let cold = self.slab_cold("qwen4_slab_tile_masks", cold_codec)?;
        if tiles == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_slab_tile_masks: no tiles",
            ));
        }
        if self.scratch.qwen4_slab_tiles_scratch_bytes < 2 * tiles * 4
            || self.scratch.qwen4_slab_tiles_scratch.is_none()
        {
            self.reserve_qwen4_slab(tiles)?;
        }
        let base = self.scratch.qwen4_slab_tiles_scratch.as_ref().unwrap();
        let views = [0usize, 1].map(|half| GpuTensor {
            buf: base.byte_view(half * tiles * 4, tiles * 4),
            shape: vec![tiles],
            dtype: DType::Raw,
        });
        // cold codec -> views[0], MQ4 (codec 0) -> views[1].
        self.slab_tile_mask(
            arch,
            tile_ids,
            &views[0],
            codec_i32,
            tiles,
            usize::from(cold.cold_codec),
        )?;
        self.slab_tile_mask(arch, tile_ids, &views[1], codec_i32, tiles, 0)?;
        Ok(views)
    }

    /// One owner mask over the slab's `[512]` codec table: gfx1201 is the EP
    /// [`Gpu::ep_tile_mask`] (unchanged); gfx1100 launches its own certified
    /// build of the same kernel, which the owner-routed EP exchange never reaches
    /// (`ep_tile_mask` itself stays gfx1201-only); gfx1151 launches its hipcc twin
    /// `qwen4_slab_tile_mask` (the typed [`SlabTileMask::Hipcc`] descriptor: one thread
    /// per tile, the same law and ids out).
    fn slab_tile_mask(
        &mut self,
        arch: &'static SlabArchKernels,
        tile_ids: &GpuTensor,
        tile_ids_owned: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
        rank: usize,
    ) -> HipResult<()> {
        if arch.arch == "gfx1201" {
            return self.ep_tile_mask(
                tile_ids,
                tile_ids_owned,
                codec_i32,
                tiles,
                SLAB_EXPERTS,
                rank,
            );
        }
        // Both bodies keep their own refusal text: the gfx1100 embedded mask (`slab tile mask: ..`) and
        // beta's gfx1151 hipcc twin (`qwen4_slab_tile_mask: ..`).
        match arch.tile_mask {
            SlabTileMask::Embedded {
                module,
                image,
                symbol,
            } => {
                let refuse = |what: &str| {
                    Err(hip_bridge::HipError::new(
                        0,
                        &format!("slab tile mask: {what}"),
                    ))
                };
                let word = std::mem::size_of::<i32>();
                if i32::try_from(tiles).is_err() {
                    return refuse("tiles exceed i32");
                }
                if tile_ids.buf.size() < tiles * word || tile_ids_owned.buf.size() < tiles * word {
                    return refuse("tile id buffers hold fewer than `tiles` words");
                }
                if codec_i32.buf.size() < SLAB_EXPERTS * word {
                    return refuse("codec table holds fewer than 512 words");
                }
                if tile_ids.buf.as_ptr() == tile_ids_owned.buf.as_ptr() {
                    return refuse("the masked copy must not alias the source tile ids");
                }
                self.bind_thread()?;
                self.ensure_embedded_kernel(module, image, symbol)?;
                let ids = tile_ids.buf.as_ptr();
                let owned = tile_ids_owned.buf.as_ptr();
                let table = codec_i32.buf.as_ptr();
                let (tiles_i32, experts_i32, rank_i32) =
                    (tiles as i32, SLAB_EXPERTS as i32, rank as i32);
                let mut params = [
                    &ids as *const _ as *mut c_void,
                    &owned as *const _ as *mut c_void,
                    &table as *const _ as *mut c_void,
                    &tiles_i32 as *const _ as *mut c_void,
                    &experts_i32 as *const _ as *mut c_void,
                    &rank_i32 as *const _ as *mut c_void,
                ];
                let timer =
                    crate::profile::begin_timer(&self.hip, "elementwise", symbol, tiles * word * 2);
                let result = self.launch_maybe_blob(
                    symbol,
                    [(tiles as u32).div_ceil(SLAB_TILE_MASK_BLOCK), 1, 1],
                    [SLAB_TILE_MASK_BLOCK, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(ids);
                        b.push_ptr(owned);
                        b.push_ptr(table);
                        b.push_i32(tiles_i32);
                        b.push_i32(experts_i32);
                        b.push_i32(rank_i32);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                result
            }
            SlabTileMask::Hipcc {
                module,
                source,
                symbol,
            } => {
                let (owned, codec) = (tile_ids_owned, rank);
                let refuse = |what: &str| {
                    Err(hip_bridge::HipError::new(
                        0,
                        &format!("qwen4_slab_tile_mask: {what}"),
                    ))
                };
                let word = std::mem::size_of::<i32>();
                if i32::try_from(tiles).is_err() {
                    return refuse("tiles exceed i32");
                }
                if tile_ids.buf.size() < tiles * word || owned.buf.size() < tiles * word {
                    return refuse("tile id buffers hold fewer than `tiles` words");
                }
                if codec_i32.buf.size() < SLAB_EXPERTS * word {
                    return refuse("codec table holds fewer than 512 words");
                }
                if tile_ids.buf.as_ptr() == owned.buf.as_ptr() {
                    return refuse("the masked copy must not alias the source tile ids");
                }
                self.bind_thread()?;
                self.ensure_kernel(module, source, symbol)?;
                let (ids, out, table) = (
                    tile_ids.buf.as_ptr(),
                    owned.buf.as_ptr(),
                    codec_i32.buf.as_ptr(),
                );
                let (tv, ev, cv) = (tiles as i32, SLAB_EXPERTS as i32, codec as i32);
                let mut params = [
                    &ids as *const _ as *mut c_void,
                    &out as *const _ as *mut c_void,
                    &table as *const _ as *mut c_void,
                    &tv as *const _ as *mut c_void,
                    &ev as *const _ as *mut c_void,
                    &cv as *const _ as *mut c_void,
                ];
                let timer =
                    crate::profile::begin_timer(&self.hip, "elementwise", symbol, tiles * word * 2);
                let result = self.launch_maybe_blob(
                    symbol,
                    [(tiles as u32).div_ceil(SLAB_TILE_MASK_BLOCK), 1, 1],
                    [SLAB_TILE_MASK_BLOCK, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(ids);
                        b.push_ptr(out);
                        b.push_ptr(table);
                        b.push_i32(tv);
                        b.push_i32(ev);
                        b.push_i32(cv);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                result
            }
        }
    }

    /// K3 down A4 producer of the gfx1100 slab K2 prefill: grouped BF16 SwiGLU rows `h`
    /// `[grouped_rows, 640]` -> the compact flat-slot `block_i4_128` down sidecar
    /// `[5, total_slots]` (record `g * total_slots + sorted[p]`, 72 bytes), padding rows
    /// emit nothing. The row-local `mq_i4_a4_r128_k640_gfx1100` entry (certified builder
    /// module `mq_i4_a4_pm_gfx1100`, 48-byte kernarg, grid `5 * grouped_rows`, block 32);
    /// `k` must be 640. Same handle and ownership as [`Gpu::qwen4_moe_rotate128_i4`]
    /// (which refuses gfx1100). gfx1100 builds the sidecar with K3; gfx1201 and gfx1151 (no K3, `k3` is `None`)
    /// delegate to [`Gpu::qwen4_moe_rotate128_i4`] unchanged, before the K3 lookup.
    pub fn qwen4_moe_rotate128_i4_slab(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        let arch = self.slab_kernels("qwen4_moe_rotate128_i4_slab")?;
        if matches!(arch.arch, "gfx1201" | "gfx1151") {
            return self.qwen4_moe_rotate128_i4(h, sorted, k, grouped_rows, total_slots);
        }
        let Some((module, image, symbol)) = arch.k3 else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "qwen4_moe_rotate128_i4_slab: no K3 A4 producer on {}",
                    self.arch
                ),
            ));
        };
        if k != SLAB_DOWN_K || grouped_rows == 0 || total_slots == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_moe_rotate128_i4_slab: need k == 640 and nonzero rows",
            ));
        }
        crate::moe::require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i4_slab grouped BF16 rows",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i4_slab sorted slots",
        )?;
        let grid_x = u32::try_from((k / 128) * grouped_rows).map_err(|_| {
            hip_bridge::HipError::new(0, "qwen4_moe_rotate128_i4_slab: grid exceeds u32")
        })?;
        let (pv, lv) = match (u32::try_from(grouped_rows), u32::try_from(total_slots)) {
            (Ok(p), Ok(l)) => (p, l),
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qwen4_moe_rotate128_i4_slab: extent overflow",
                ))
            }
        };
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_embedded_kernel(module, image, symbol)?;
        let needed = k / 128 * total_slots * 72;
        if crate::scratch::scratch_will_grow(
            self.scratch.qwen4_moe_down_i4_scratch_bytes,
            self.scratch.qwen4_moe_down_i4_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let reservation = self
            .scratch
            .reserve_qwen4_moe_down_i4(&self.hip, k, total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            symbol,
            grouped_rows * (k * 2 + 4) + needed,
        );
        let result =
            self.launch_maybe_blob(symbol, [grid_x, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_u32(pv);
                b.push_u32(lv);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(crate::scratch::Int4MmqPrepared::from_reservation(
            reservation,
        ))
    }
    /// K3 down A4 producer of the gfx1100 aggregate K2 prefill: grouped BF16 SwiGLU rows `h`
    /// `[grouped_rows, k]` -> the compact flat-slot `block_i4_128` down sidecar
    /// `[k / 128, total_slots]` (record `g * total_slots + sorted[p]`, 72 bytes), padding rows
    /// emit nothing. `producer` is the descriptor's embedded row-local entry (module key, image,
    /// symbol, `k`; for gfx1100 `mq_i4_a4_r128_k640_gfx1100` of the certified builder module
    /// `mq_i4_a4_pm_gfx1100`, 48-byte kernarg, grid `(k / 128) * grouped_rows`, block 32) and
    /// `k` must equal the producer's `k`. Same handle and ownership as the hipcc arm of
    /// [`Gpu::qwen4_moe_rotate128_i4`], which dispatches here.
    pub(crate) fn qwen4_moe_rotate128_i4_embedded(
        &mut self,
        (module, image, symbol, producer_k): (&'static str, &'static [u8], &'static str, usize),
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        if k != producer_k || grouped_rows == 0 || total_slots == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("qwen4_moe_rotate128_i4: need k == {producer_k} and nonzero rows"),
            ));
        }
        crate::moe::require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i4 grouped BF16 rows",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i4 sorted slots",
        )?;
        let grid_x = u32::try_from((k / 128) * grouped_rows).map_err(|_| {
            hip_bridge::HipError::new(0, "qwen4_moe_rotate128_i4: grid exceeds u32")
        })?;
        let (pv, lv) = match (u32::try_from(grouped_rows), u32::try_from(total_slots)) {
            (Ok(p), Ok(l)) => (p, l),
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qwen4_moe_rotate128_i4: extent overflow",
                ))
            }
        };
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_embedded_kernel(module, image, symbol)?;
        let needed = k / 128 * total_slots * 72;
        if crate::scratch::scratch_will_grow(
            self.scratch.qwen4_moe_down_i4_scratch_bytes,
            self.scratch.qwen4_moe_down_i4_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let reservation = self
            .scratch
            .reserve_qwen4_moe_down_i4(&self.hip, k, total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            symbol,
            grouped_rows * (k * 2 + 4) + needed,
        );
        let result =
            self.launch_maybe_blob(symbol, [grid_x, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_u32(pv);
                b.push_u32(lv);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        self.moe_route_census.a4_down_producer += 1;
        Ok(crate::scratch::Int4MmqPrepared::from_reservation(
            reservation,
        ))
    }

    /// S1 slow-link layer-major prefill: apply one panel of grouped BF16 down rows to the
    /// ordered combine accumulator `acc` (`[tokens, hidden]` F32 holding the BF16 running sum).
    /// `y` holds the panel's rows `[panel_rows, hidden]` BF16 (panel-local); the panel covers
    /// the window's grouped rows `[row_base, row_base + panel_rows)` of an `inverse` map over
    /// `grouped_rows` window rows, and additionally owns the slots with a valid id in
    /// `[own_lo, own_hi)` whose grouped row is invalid (see
    /// [`slowlink_panel_owns`]).  Panels of one layer must be launched in ascending
    /// `row_base` order; [`Gpu::slowlink_combine_pad`] runs once after the last.
    #[allow(clippy::too_many_arguments)]
    pub fn slowlink_combine_panel(
        &mut self,
        y: &GpuTensor,
        inverse: &GpuTensor,
        topk: &GpuTensor,
        weights: &GpuTensor,
        acc: &GpuTensor,
        hidden: usize,
        tokens: usize,
        n_experts: usize,
        grouped_rows: usize,
        row_base: usize,
        panel_rows: usize,
        own_lo: usize,
        own_hi: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if hidden == 0 || hidden % 8 != 0 || tokens == 0 || own_lo > own_hi || own_hi > n_experts {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_combine_panel: need hidden % 8 == 0, tokens > 0 and own_lo <= own_hi <= n_experts",
            ));
        }
        if row_base
            .checked_add(panel_rows)
            .is_none_or(|end| end > grouped_rows)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_combine_panel: the panel exceeds the window's grouped rows",
            ));
        }
        for value in [
            hidden,
            tokens,
            n_experts,
            grouped_rows,
            row_base,
            panel_rows,
        ] {
            if i32::try_from(value).is_err() {
                return Err(hip_bridge::HipError::new(
                    0,
                    "slowlink_combine_panel: extent exceeds i32",
                ));
            }
        }
        crate::moe::require_span(
            y,
            &[panel_rows, hidden, 2],
            "slowlink_combine_panel panel rows",
        )?;
        crate::moe::require_span(
            inverse,
            &[tokens, 10, 4],
            "slowlink_combine_panel inverse permutation",
        )?;
        crate::moe::require_span(topk, &[tokens, 10, 4], "slowlink_combine_panel expert ids")?;
        crate::moe::require_span(weights, &[tokens, 10, 4], "slowlink_combine_panel weights")?;
        crate::moe::require_span(
            acc,
            &[tokens, hidden, 4],
            "slowlink_combine_panel accumulator",
        )?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_combine_panel";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (yp, ip, tp, wp, ap) = (
            y.buf.as_ptr(),
            inverse.buf.as_ptr(),
            topk.buf.as_ptr(),
            weights.buf.as_ptr(),
            acc.buf.as_ptr(),
        );
        let (hv, tv, nx, gr) = (
            hidden as i32,
            tokens as i32,
            n_experts as i32,
            grouped_rows as i32,
        );
        let (rb, pr, olo, ohi) = (
            row_base as i32,
            panel_rows as i32,
            own_lo as i32,
            own_hi as i32,
        );
        let mut params = [
            &yp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &rb as *const _ as *mut c_void,
            &pr as *const _ as *mut c_void,
            &olo as *const _ as *mut c_void,
            &ohi as *const _ as *mut c_void,
        ];
        let bytes = panel_rows * hidden * 2 + 2 * tokens * hidden * 4;
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", FUNC, bytes);
        let result = self.launch_maybe_blob(
            FUNC,
            [(hidden as u32 / 8).div_ceil(64), tokens as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(yp);
                b.push_ptr(ip);
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_ptr(ap);
                b.push_i32(hv);
                b.push_i32(tv);
                b.push_i32(nx);
                b.push_i32(gr);
                b.push_i32(rb);
                b.push_i32(pr);
                b.push_i32(olo);
                b.push_i32(ohi);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// The canonical combine's terminal `(10 - valid)` +0 products, once per layer after the
    /// last [`Gpu::slowlink_combine_panel`] (only a -0 accumulator changes).
    pub fn slowlink_combine_pad(
        &mut self,
        topk: &GpuTensor,
        acc: &GpuTensor,
        hidden: usize,
        tokens: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if hidden == 0 || hidden % 8 != 0 || tokens == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_combine_pad: need hidden % 8 == 0 and tokens > 0",
            ));
        }
        for value in [hidden, tokens, n_experts] {
            if i32::try_from(value).is_err() {
                return Err(hip_bridge::HipError::new(
                    0,
                    "slowlink_combine_pad: extent exceeds i32",
                ));
            }
        }
        crate::moe::require_span(topk, &[tokens, 10, 4], "slowlink_combine_pad expert ids")?;
        crate::moe::require_span(
            acc,
            &[tokens, hidden, 4],
            "slowlink_combine_pad accumulator",
        )?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_combine_pad";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (tp, ap) = (topk.buf.as_ptr(), acc.buf.as_ptr());
        let (hv, tv, nx) = (hidden as i32, tokens as i32, n_experts as i32);
        let mut params = [
            &tp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &nx as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [(hidden as u32 / 8).div_ceil(64), tokens as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(ap);
                b.push_i32(hv);
                b.push_i32(tv);
                b.push_i32(nx);
                b
            },
        )
    }

    /// Fill the first `bytes` of `tensor` with the byte `value` on the active stream (blocking
    /// when none): `0` clears F32/I32 words to +0 / 0, `0xFF` sets I32 words to -1.
    pub fn slowlink_fill_bytes(
        &mut self,
        tensor: &GpuTensor,
        value: u8,
        bytes: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if bytes > tensor.buf.size() {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_fill_bytes: the span exceeds the tensor",
            ));
        }
        match self.active_stream.as_ref() {
            Some(stream) => self
                .hip
                .memset_async(&tensor.buf, i32::from(value), bytes, stream),
            None => self.hip.memset(&tensor.buf, i32::from(value), bytes),
        }
    }

    /// Count the cohort's route ids per expert: `counts[id] += 1` per valid id (in
    /// `[0, n_experts)`) of `ids` (`n` I32); `counts` is a separate word array the caller zeroed
    /// (one i32 an expert, no packing with the K2 padded counts).
    pub fn slowlink_fold_k1_counts(
        &mut self,
        ids: &GpuTensor,
        counts: &GpuTensor,
        n: usize,
        n_experts: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if n == 0 || n_experts == 0 || [n, n_experts].iter().any(|&v| i32::try_from(v).is_err()) {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_fold_k1_counts: need n > 0, experts > 0 and i32 extents",
            ));
        }
        crate::moe::require_span(ids, &[n, 4], "slowlink_fold_k1_counts ids")?;
        crate::moe::require_span(counts, &[n_experts, 4], "slowlink_fold_k1_counts counts")?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_fold_k1_counts";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (ip, cp) = (ids.buf.as_ptr(), counts.buf.as_ptr());
        let (nv, ev) = (n as i32, n_experts as i32);
        let mut params = [
            &ip as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &ev as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(cp);
                b.push_i32(nv);
                b.push_i32(ev);
                b
            },
        )
    }

    /// `out[i] = ids[i]` when `lo <= ids[i] < hi`, else `-1`, for `n` I32 ids (the K1-cohort
    /// view of a short-tail tile's routes against one expert binding).
    pub fn slowlink_mask_ids(
        &mut self,
        ids: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        lo: usize,
        hi: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if n == 0 || lo > hi || [n, lo, hi].iter().any(|&v| i32::try_from(v).is_err()) {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_mask_ids: need n > 0, lo <= hi and i32 extents",
            ));
        }
        crate::moe::require_span(ids, &[n, 4], "slowlink_mask_ids ids")?;
        crate::moe::require_span(out, &[n, 4], "slowlink_mask_ids output")?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_mask_ids";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (ip, op) = (ids.buf.as_ptr(), out.buf.as_ptr());
        let (nv, lv, hv) = (n as i32, lo as i32, hi as i32);
        let mut params = [
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(nv);
                b.push_i32(lv);
                b.push_i32(hv);
                b
            },
        )
    }

    /// `local[i] = i` when `sorted[i] >= 0`, else `-1`, for the `rows` I32 slots of one panel
    /// (the compact local slot space of the aggregate route's panel unscatter and down GEMM).
    pub fn slowlink_remap_sorted(
        &mut self,
        sorted: &GpuTensor,
        local: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if rows == 0 || i32::try_from(rows).is_err() {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_remap_sorted: need rows > 0 and an i32 extent",
            ));
        }
        crate::moe::require_span(sorted, &[rows, 4], "slowlink_remap_sorted sorted")?;
        crate::moe::require_span(local, &[rows, 4], "slowlink_remap_sorted local")?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_remap_sorted";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (sp, lp) = (sorted.buf.as_ptr(), local.buf.as_ptr());
        let rv = rows as i32;
        let mut params = [
            &sp as *const _ as *mut c_void,
            &lp as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [(rows as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(lp);
                b.push_i32(rv);
                b
            },
        )
    }

    /// Copy the F32 rows (`width` wide) of the `slots` slots whose id in `ids` lies in
    /// `[lo, hi)` from `src` to `dst`; every other row of `dst` is left untouched (the
    /// aggregate K1 cohort's per-binding merge).
    #[allow(clippy::too_many_arguments)]
    pub fn slowlink_merge_rows(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        ids: &GpuTensor,
        slots: usize,
        width: usize,
        lo: usize,
        hi: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if slots == 0
            || width == 0
            || lo > hi
            || [slots, width, lo, hi]
                .iter()
                .any(|&v| i32::try_from(v).is_err())
            || slots
                .checked_mul(width)
                .is_none_or(|n| i32::try_from(n).is_err())
        {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_merge_rows: need slots > 0, width > 0, lo <= hi and i32 extents",
            ));
        }
        crate::moe::require_span(src, &[slots, width, 4], "slowlink_merge_rows src")?;
        crate::moe::require_span(dst, &[slots, width, 4], "slowlink_merge_rows dst")?;
        crate::moe::require_span(ids, &[slots, 4], "slowlink_merge_rows ids")?;
        const MODULE: &str = "slowlink_prefill";
        const FUNC: &str = "slowlink_merge_rows";
        self.ensure_kernel(MODULE, kernels::SLOWLINK_PREFILL_SRC, FUNC)?;
        let (sp, dp, ip) = (src.buf.as_ptr(), dst.buf.as_ptr(), ids.buf.as_ptr());
        let (sv, wv, lv, hv) = (slots as i32, width as i32, lo as i32, hi as i32);
        let mut params = [
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &wv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
            &hv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            FUNC,
            [slots as u32, (width as u32).div_ceil(256), 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(sp);
                b.push_ptr(dp);
                b.push_ptr(ip);
                b.push_i32(sv);
                b.push_i32(wv);
                b.push_i32(lv);
                b.push_i32(hv);
                b
            },
        )
    }

    /// Allocate (load time, never shrinking) the host-mapped coherent buffer the S1 count
    /// barrier's copy kernel writes: two arrays of `words` i32.
    pub fn reserve_slowlink_count_host(&mut self, words: usize) -> HipResult<()> {
        self.bind_thread()?;
        let bytes = words.checked_mul(8).ok_or_else(|| {
            hip_bridge::HipError::new(0, "reserve_slowlink_count_host: size overflows")
        })?;
        if self
            .slowlink_count_host
            .as_ref()
            .is_some_and(|buf| buf.size() >= bytes)
        {
            return Ok(());
        }
        self.slowlink_count_host = Some(self.hip.mapped_host_buffer(bytes)?);
        Ok(())
    }

    /// Shared body of the S1 count barrier: ONE `copy_regions_u32` kernel launch on the active
    /// stream copying the first `words` i32 of each source into the host-mapped coherent count
    /// buffer, one wait on that stream, then the words read out of the mapping.
    fn slowlink_count_barrier(
        &mut self,
        name: &str,
        sources: &[&GpuTensor],
        words: usize,
    ) -> HipResult<Vec<Vec<i32>>> {
        if self.active_stream.is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{name}: the S1 count barrier needs the active compute stream"),
            ));
        }
        let Some(mapped) = self.slowlink_count_host.take() else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{name}: count buffer not reserved at load (reserve_slowlink_count_host)"),
            ));
        };
        let result = self.slowlink_count_barrier_with(name, &mapped, sources, words);
        self.slowlink_count_host = Some(mapped);
        result
    }

    fn slowlink_count_barrier_with(
        &mut self,
        name: &str,
        mapped: &hip_bridge::MappedHostBuffer,
        sources: &[&GpuTensor],
        words: usize,
    ) -> HipResult<Vec<Vec<i32>>> {
        let bytes = words * 4;
        if mapped.size() < sources.len() * bytes {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{name}: count buffer not reserved at load (reserve_slowlink_count_host)"),
            ));
        }
        if words == 0 {
            return Ok(vec![Vec::new(); sources.len()]);
        }
        let words32 = u32::try_from(words)
            .map_err(|_| hip_bridge::HipError::new(0, &format!("{name}: too many words")))?;
        let base = mapped.device_buffer().as_ptr() as u64;
        let mut table = [(0u64, 0u64, 0u32); 2];
        for (index, source) in sources.iter().enumerate() {
            let dst = base + (index * bytes) as u64;
            let src = source.buf.as_ptr() as u64;
            assert!(
                (dst | src | bytes as u64) % 4 == 0,
                "{name}: count copy is not 4-byte aligned"
            );
            table[index] = (dst, src, words32);
        }
        crate::tensor_ops::launch_copy_regions(self, &table[..sources.len()])?;
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip.stream_synchronize(stream)?;
        }
        self.take_device_wait_error()?;
        // SAFETY: the active stream that ran the copy kernel was just synchronized and no
        // other GPU write targets this Gpu-owned buffer.
        let host = unsafe { mapped.host_bytes() };
        Ok((0..sources.len())
            .map(|index| {
                host[index * bytes..(index + 1) * bytes]
                    .chunks_exact(4)
                    .map(|word| i32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
                    .collect()
            })
            .collect())
    }

    /// The SP sorted readback: ONE `copy_regions_u32` kernel on the ACTIVE compute stream
    /// copying the first `words` i32 of `tensor` into the caller's host-mapped coherent `dst`
    /// (a kernel, so it never queues behind the transport's SDMA page copies, like the count
    /// barrier).  No wait: the caller proves completion with a later wait on the same stream
    /// (the S1 count barrier) before reading `dst`.
    pub fn slowlink_copy_i32_to_mapped(
        &mut self,
        tensor: &GpuTensor,
        dst: &hip_bridge::MappedHostBuffer,
        words: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let name = "slowlink_copy_i32_to_mapped";
        if self.active_stream.is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("{name}: needs the active compute stream"),
            ));
        }
        let bytes = words
            .checked_mul(4)
            .filter(|&bytes| bytes <= tensor.buf.size() && bytes <= dst.size())
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!("{name}: the span exceeds a buffer"))
            })?;
        if words == 0 {
            return Ok(());
        }
        let words32 = u32::try_from(words)
            .map_err(|_| hip_bridge::HipError::new(0, &format!("{name}: too many words")))?;
        let dst_ptr = dst.device_buffer().as_ptr() as u64;
        let src_ptr = tensor.buf.as_ptr() as u64;
        assert!(
            (dst_ptr | src_ptr | bytes as u64) % 4 == 0,
            "{name}: copy is not 4-byte aligned"
        );
        crate::tensor_ops::launch_copy_regions(self, &[(dst_ptr, src_ptr, words32)])
    }

    /// The one S1 count barrier: a kernel copy of the first `words` i32 of `tensor` into
    /// host-mapped coherent memory on the ACTIVE compute stream (so it never queues behind
    /// the transport's SDMA page copies), then one wait on that stream alone (never the
    /// transport owner's copy stream, never the legacy/null stream, never a device-wide
    /// sync).  Needs an active stream and the load-time `reserve_slowlink_count_host`.
    pub fn slowlink_sync_download_i32(
        &mut self,
        tensor: &GpuTensor,
        words: usize,
    ) -> HipResult<Vec<i32>> {
        self.bind_thread()?;
        if words
            .checked_mul(4)
            .is_none_or(|bytes| bytes > tensor.buf.size())
        {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_sync_download_i32: the span exceeds the tensor",
            ));
        }
        let mut out =
            self.slowlink_count_barrier("slowlink_sync_download_i32", &[tensor], words)?;
        Ok(out.remove(0))
    }

    /// [`Gpu::slowlink_sync_download_i32`] of the first `words` i32 of TWO tensors at the same
    /// barrier: one kernel launch copying both into host-mapped coherent memory on the active
    /// compute stream, then one wait on that stream alone.  Needs an active stream.
    pub fn slowlink_sync_download_i32_pair(
        &mut self,
        first: &GpuTensor,
        second: &GpuTensor,
        words: usize,
    ) -> HipResult<(Vec<i32>, Vec<i32>)> {
        self.bind_thread()?;
        let bytes_len = words.checked_mul(4);
        if bytes_len.is_none_or(|bytes| bytes > first.buf.size() || bytes > second.buf.size()) {
            return Err(hip_bridge::HipError::new(
                0,
                "slowlink_sync_download_i32_pair: the span exceeds a tensor",
            ));
        }
        let mut out = self.slowlink_count_barrier(
            "slowlink_sync_download_i32_pair",
            &[first, second],
            words,
        )?;
        let b = out.remove(1);
        let a = out.remove(0);
        Ok((a, b))
    }
}

/// Host-buffer lifetime guard of a stream-ordered D2H sequence (`hipMemcpyAsync` writes the host
/// buffer after the call returns; the caller owns that lifetime).  `enqueue` queues the copies
/// into `host` and sets `enqueued` as soon as the FIRST copy was accepted; `join` is the
/// confirmed join of the stream.  Rules:
///
/// * nothing accepted: no copy can write `host`; it is dropped and the enqueue error returned;
/// * an enqueue fails after one was accepted: the stream is still joined (the accepted copy
///   may be writing `host`) and the enqueue error stays the primary error; `host` is dropped
///   only if the join confirmed;
/// * the join fails: the copies may still be in flight, so `host` is QUARANTINED (leaked, never
///   freed while a device write may target it) and the error returned (the enqueue error when
///   there is one, else the join error).
///
/// No retry and no suppression: a failure is reported as it occurred.
#[cfg(test)]
fn guarded_async_download<B>(
    mut host: B,
    enqueue: impl FnOnce(&mut B, &mut bool) -> HipResult<()>,
    join: impl FnOnce() -> HipResult<()>,
) -> HipResult<B> {
    let mut enqueued = false;
    let queued = enqueue(&mut host, &mut enqueued);
    if !enqueued {
        return queued.map(|()| host);
    }
    let joined = join();
    match (queued, joined) {
        (Ok(()), Ok(())) => Ok(host),
        // Joined: every accepted copy finished, `host` is safe to drop.
        (Err(primary), Ok(())) => Err(primary),
        (Ok(()), Err(join_error)) => {
            std::mem::forget(host);
            Err(join_error)
        }
        (Err(primary), Err(_)) => {
            std::mem::forget(host);
            Err(primary)
        }
    }
}

#[cfg(test)]
mod guarded_download_tests {
    use super::guarded_async_download;
    use hip_bridge::{HipError, HipResult};
    use std::cell::RefCell;
    use std::rc::Rc;

    type Log = Rc<RefCell<Vec<&'static str>>>;

    /// A host buffer that logs when it is freed.
    struct Probe(Log);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.borrow_mut().push("drop");
        }
    }

    fn err(what: &str) -> HipError {
        HipError::new(7, what)
    }

    fn run(
        log: &Log,
        first: HipResult<()>,
        second: Option<HipResult<()>>,
        join: HipResult<()>,
    ) -> HipResult<Probe> {
        guarded_async_download(
            Probe(log.clone()),
            |_host, enqueued| {
                log.borrow_mut().push("enqueue-1");
                first?;
                *enqueued = true;
                if let Some(second) = second {
                    log.borrow_mut().push("enqueue-2");
                    second?;
                }
                Ok(())
            },
            || {
                log.borrow_mut().push("join");
                join
            },
        )
    }

    #[test]
    fn slowlink_guarded_download_success_joins_then_hands_the_buffer_over() {
        let log: Log = Rc::default();
        let got = run(&log, Ok(()), Some(Ok(())), Ok(()));
        assert!(got.is_ok());
        assert_eq!(*log.borrow(), ["enqueue-1", "enqueue-2", "join"]);
        drop(got);
        assert_eq!(log.borrow().last(), Some(&"drop"));
    }

    #[test]
    fn slowlink_guarded_download_second_enqueue_failure_joins_before_the_buffer_drops() {
        let log: Log = Rc::default();
        let got = run(&log, Ok(()), Some(Err(err("second enqueue"))), Ok(()));
        let e = got.err().expect("the enqueue error is the primary error");
        assert!(e.to_string().contains("second enqueue"), "{e}");
        // The accepted first copy is joined BEFORE the host buffer is freed.
        assert_eq!(
            *log.borrow(),
            ["enqueue-1", "enqueue-2", "join", "drop"],
            "join must precede drop"
        );
    }

    #[test]
    fn slowlink_guarded_download_unconfirmed_join_quarantines_the_buffer() {
        // Enqueue failure AND join failure: the primary (enqueue) error is returned and the
        // buffer is leaked, never dropped while a copy may still target it.
        let log: Log = Rc::default();
        let got = run(
            &log,
            Ok(()),
            Some(Err(err("second enqueue"))),
            Err(err("wait")),
        );
        let e = got.err().unwrap();
        assert!(e.to_string().contains("second enqueue"), "{e}");
        assert_eq!(*log.borrow(), ["enqueue-1", "enqueue-2", "join"]);
        // Join failure alone after both copies were accepted: the join error, still leaked.
        let log: Log = Rc::default();
        let got = run(&log, Ok(()), Some(Ok(())), Err(err("wait")));
        let e = got.err().unwrap();
        assert!(e.to_string().contains("wait"), "{e}");
        assert_eq!(*log.borrow(), ["enqueue-1", "enqueue-2", "join"]);
    }

    #[test]
    fn slowlink_guarded_download_first_enqueue_failure_needs_no_join() {
        let log: Log = Rc::default();
        let got = run(&log, Err(err("first enqueue")), Some(Ok(())), Ok(()));
        let e = got.err().unwrap();
        assert!(e.to_string().contains("first enqueue"), "{e}");
        // Nothing was accepted: no join, and the buffer is freed (no device write targets it).
        assert_eq!(*log.borrow(), ["enqueue-1", "drop"]);
    }

    #[test]
    fn slowlink_guarded_download_single_copy_join_failure_quarantines() {
        let log: Log = Rc::default();
        let got = run(&log, Ok(()), None, Err(err("wait")));
        assert!(got.is_err());
        assert_eq!(*log.borrow(), ["enqueue-1", "join"]);
        // Confirmed join: the buffer is handed to the caller (alive until the caller drops it),
        // freed only afterwards.
        let log: Log = Rc::default();
        let ok = run(&log, Ok(()), None, Ok(()));
        assert!(ok.is_ok());
        assert_eq!(*log.borrow(), ["enqueue-1", "join"]);
        drop(ok);
        assert_eq!(*log.borrow(), ["enqueue-1", "join", "drop"]);
    }
}

/// Whether the S1 ordered-combine panel `[row_base, row_base + panel_rows)` of the window's
/// `grouped_rows` rows, owning the invalid-row slots of experts `[own_lo, own_hi)`, applies the
/// route slot `(id, row)`: exactly the predicate `slowlink_combine_panel` evaluates per slot.
/// A slot whose id is outside `[0, n_experts)` is never applied here (the canonical kernel
/// skips it and pads once, [`Gpu::slowlink_combine_pad`]); a slot with a valid grouped row
/// belongs to the panel holding that row; a valid-id slot whose grouped row is invalid (the
/// canonical kernel multiplies +0 by its weight) belongs to the one panel whose ownership
/// range holds its id.  The ownership ranges of a layer's panels partition `[0, n_experts)`.
#[allow(clippy::too_many_arguments)]
pub fn slowlink_panel_owns(
    id: i32,
    row: i32,
    n_experts: usize,
    grouped_rows: usize,
    row_base: usize,
    panel_rows: usize,
    own_lo: usize,
    own_hi: usize,
) -> bool {
    if id < 0 || id as usize >= n_experts {
        return false;
    }
    if row >= 0 && (row as usize) < grouped_rows {
        let row = row as usize;
        row >= row_base && row - row_base < panel_rows
    } else {
        let id = id as usize;
        id >= own_lo && id < own_hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::{envelope::Source, inst::Arch};
    use peacemaker_lift::bundle::SourceCodec;

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn arch_table_covers_exactly_gfx1201_gfx1100_and_gfx1151() {
        assert_eq!(
            slab_arch_kernels("gfx1201").map(|k| k.arch),
            Some("gfx1201")
        );
        assert_eq!(
            slab_arch_kernels("gfx1100").map(|k| k.arch),
            Some("gfx1100")
        );
        assert_eq!(
            slab_arch_kernels("gfx1151").map(|k| k.arch),
            Some("gfx1151")
        );
        for other in [
            "gfx942", "gfx906", "gfx1101", "gfx1200", "gfx1100x", "gfx1151x", "GFX1151", "",
        ] {
            assert!(slab_arch_kernels(other).is_none(), "{other}");
        }
    }

    #[test]
    fn gfx1151_table_is_beta_k1_shared_planner_and_hipcc_mask() {
        let k = slab_arch_kernels("gfx1151").unwrap();
        // The shared planner (the same module key and source as gfx1201) and no K3 (the down sidecar
        // delegates to the existing producer).
        assert_eq!(k.plan_module, "qwen4_slab_union_plan");
        assert_eq!((k.plan_src)(), kernels::QWEN4_SLAB_UNION_PLAN_SRC);
        assert!(k.k3.is_none());
        // The real HIP mask descriptor: module/symbol `qwen4_slab_tile_mask`, the hipcc source, not an image.
        let SlabTileMask::Hipcc {
            module,
            source,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1151's tile mask is a hipcc source");
        };
        assert_eq!(
            (module, symbol),
            ("qwen4_slab_tile_mask", "qwen4_slab_tile_mask")
        );
        assert_eq!(source, kernels::QWEN4_SLAB_TILE_MASK_SRC);
        assert!(!source.is_empty());
        // K1: one image for both buckets, MQ4 hot bucket and W3 cold bucket.
        assert_eq!(k.mq4_k1.module, MQ_I4_MOE_GEMV_GFX1151_MODULE);
        assert_eq!(k.mq4_k1.module, "mq_i4_moe_gemv_pm_gfx1151");
        assert_eq!(k.mq4_k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1151);
        assert_eq!(k.mq4_k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[1]);
        let w3 = slab_cold_kernels("gfx1151", 1).unwrap();
        assert_eq!(w3.cold_codec, 1);
        assert_eq!(w3.k1.module, k.mq4_k1.module);
        assert!(std::ptr::eq(w3.k1.image, k.mq4_k1.image));
        assert_eq!(w3.k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[0]);
        for k1 in [&k.mq4_k1, &w3.k1] {
            for symbol in k1.symbols {
                assert!(
                    contains_bytes(k1.image, format!("{symbol}.kd").as_bytes()),
                    "{symbol}.kd missing"
                );
            }
        }
        // W2 has no gfx1151 table row; the E8P row has its own test (`gfx1151_e8p_k1_table_*`).
        assert!(
            slab_cold_kernels("gfx1151", 2).is_none()
                && slab_cold_k1_metadata("gfx1151", 2).is_none()
        );
    }

    /// The gfx1151 E8P K1 table row is the gfx1100 one under gfx1151 names (module, symbols,
    /// image, contracts); the table is inventory and never permission.
    #[test]
    fn gfx1151_e8p_k1_table_is_the_gfx1100_arm_under_gfx1151_names() {
        let e8p = slab_cold_k1_metadata("gfx1151", 3).unwrap();
        assert_eq!(e8p.cold_codec, 3);
        assert_eq!(e8p.k1.module, "e8p_i4_moe_gemv_pm_gfx1151");
        assert_eq!(
            e8p.k1.module,
            crate::e8p_moe_decode::E8P_I4_MOE_GEMV_GFX1151_MODULE
        );
        assert_eq!(e8p.k1.image, kernels::E8P_I4_MOE_GEMV_PM_GFX1151);
        assert_eq!(
            e8p.k1.symbols,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1151",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1151"
            ]
        );
        // Symbol for symbol the gfx1100 names with only the arch spelling changed.
        for (halo, xtx) in e8p.k1.symbols.iter().zip(E8P_I4_MOE_GEMV_GFX1100_SYMBOLS) {
            assert_eq!(*halo, xtx.replace("gfx1100", "gfx1151"));
        }
        assert_eq!(
            e8p.k1.module,
            E8P_I4_MOE_GEMV_GFX1100_MODULE.replace("gfx1100", "gfx1151")
        );
        let k = slab_arch_kernels("gfx1151").unwrap();
        assert_ne!(e8p.k1.module, k.mq4_k1.module);
        assert!(!std::ptr::eq(e8p.k1.image, k.mq4_k1.image));
        assert!(!std::ptr::eq(
            e8p.k1.image,
            kernels::E8P_I4_MOE_GEMV_PM_GFX1100
        ));
        assert!(!std::ptr::eq(
            e8p.k1.image,
            kernels::E8P_I4_MOE_GEMV_PM_GFX1201
        ));
        for symbol in e8p.k1.symbols {
            assert!(
                contains_bytes(e8p.k1.image, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
            assert!(!symbol.contains("mq2i4") && !symbol.ends_with("_acc"));
            // Each symbol is its own contract: kernels/e8p_i4_moe_gemv.gfx1151.<kind>.contract.json.
            let kind = if symbol.contains("r256_gate_up") {
                "r256_gate_up"
            } else {
                "r128_down"
            };
            let contract =
                std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    format!("../../kernels/e8p_i4_moe_gemv.gfx1151.{kind}.contract.json"),
                ))
                .unwrap();
            assert!(
                contract.contains(&format!("\"symbol\": \"{symbol}\"")),
                "{kind} contract symbol"
            );
            assert!(
                contract.contains("\"group_segment_fixed_bytes\": 1536"),
                "{kind} contract static LDS"
            );
            assert!(
                contract.contains("\"launch_dynamic_lds_bytes\": 0"),
                "{kind} contract dynamic LDS"
            );
        }
        assert!(!e8p.k1.module.contains("mq_i4") && !e8p.k1.module.contains("gfx1201"));
        assert!(!contains_bytes(e8p.k1.image, b"mq_i4_moe_gemv"));
        // Serving selection is the shared matrix: the very same table object when admitted.
        let admitted = crate::qwen4_slab_codec_admitted("gfx1151", 3);
        assert_eq!(slab_cold_kernels("gfx1151", 3).is_some(), admitted);
        if admitted {
            assert!(std::ptr::eq(slab_cold_kernels("gfx1151", 3).unwrap(), e8p));
        }
        // W2 stays refused and W3 keeps its own shared-image table.
        assert!(
            slab_cold_kernels("gfx1151", 2).is_none()
                && slab_cold_k1_metadata("gfx1151", 2).is_none()
        );
        assert!(!std::ptr::eq(
            slab_cold_k1_metadata("gfx1151", 1).unwrap(),
            e8p
        ));
    }

    /// The gfx1151 mask is beta's hipcc source (never reinterpreted as an image): its kernel is the
    /// symbol the descriptor names, with the 36-byte ABI (three pointers, i32 tiles/experts/rank), a
    /// 256-thread block, and gfx1151-only guard; gfx1100 never receives it.
    #[test]
    fn gfx1151_tile_mask_source_is_the_beta_hipcc_kernel() {
        let SlabTileMask::Hipcc { source, symbol, .. } =
            slab_arch_kernels("gfx1151").unwrap().tile_mask
        else {
            panic!("gfx1151's tile mask is a hipcc source");
        };
        assert_eq!(SLAB_TILE_MASK_BLOCK, 256);
        assert!(source.contains(&format!("void {symbol}(")));
        assert!(source.contains("__launch_bounds__(256)"));
        assert!(source
            .contains("const int* __restrict__ owner_table, int tiles, int n_experts, int rank)"));
        assert!(source.contains("gfx1151-only"));
        assert!(!source.contains("__gfx1100__"));
        for arch in ["gfx1100", "gfx1201"] {
            assert!(
                matches!(
                    slab_arch_kernels(arch).unwrap().tile_mask,
                    SlabTileMask::Embedded { .. }
                ),
                "{arch}"
            );
        }
    }

    #[test]
    fn gfx1201_table_is_the_historical_constants() {
        let k = slab_arch_kernels("gfx1201").unwrap();
        assert_eq!(k.plan_module, "qwen4_slab_union_plan");
        assert_eq!((k.plan_src)(), kernels::QWEN4_SLAB_UNION_PLAN_SRC);
        assert!(k.k3.is_none());
        let SlabTileMask::Embedded {
            module,
            image,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1201's tile mask is an embedded builder image");
        };
        assert_eq!(module, "ep_tile_mask_pm_gfx1201");
        assert_eq!(symbol, "ep_tile_mask_pm_gfx1201");
        assert_eq!(image, kernels::EP_TILE_MASK_PM_GFX1201);
        // The hot MQ4 bucket and the W3 cold bucket: the historical K1 module, image and symbols.
        assert_eq!(k.mq4_k1.module, MQ_I4_MOE_GEMV_MODULE);
        assert_eq!(k.mq4_k1.module, "mq_i4_moe_gemv_pm_gfx1201");
        assert_eq!(k.mq4_k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(k.mq4_k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[1]);
        let w3 = slab_cold_kernels("gfx1201", 1).unwrap();
        assert_eq!(w3.cold_codec, 1);
        assert_eq!(w3.k1.module, MQ_I4_MOE_GEMV_MODULE);
        assert_eq!(w3.k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(w3.k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[0]);
        // E8P cold: its own module and blob (the base's launcher constants).
        let e8p = slab_cold_kernels("gfx1201", 3).unwrap();
        assert_eq!(e8p.cold_codec, 3);
        assert_eq!(e8p.k1.module, "e8p_i4_moe_gemv_pm_gfx1201");
        assert_eq!(e8p.k1.image, kernels::E8P_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(
            e8p.k1.symbols,
            ["e8p_i4_moe_gemv_r256_gate_up", "e8p_i4_moe_gemv_r128_down"]
        );
        for k1 in [&k.mq4_k1, &w3.k1, &e8p.k1] {
            for symbol in k1.symbols {
                assert!(
                    contains_bytes(k1.image, format!("{symbol}.kd").as_bytes()),
                    "{symbol}.kd missing"
                );
            }
        }
    }

    #[test]
    fn gfx1100_table_names_and_images() {
        let k = slab_arch_kernels("gfx1100").unwrap();
        assert_eq!(k.plan_module, "qwen4_slab_union_plan_gfx1100");
        assert_eq!((k.plan_src)(), kernels::qwen4_slab_union_plan_gfx1100_src());
        // The assembled source is the shared planner file minus ONLY its exact arch guard, with no
        // fake compiler arch macro and no wave-sensitive construct in the body.
        let assembled = (k.plan_src)();
        let original = kernels::QWEN4_SLAB_UNION_PLAN_SRC;
        assert_eq!(
            original
                .matches(kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD)
                .count(),
            1
        );
        assert_eq!(
            kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD,
            "#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__) && !defined(__gfx1151__)\n\
             #error \"qwen4_slab_union_plan.hip is gfx1201/gfx1151-only\"\n\
             #endif\n"
        );
        assert_eq!(
            assembled,
            original.replacen(kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD, "", 1),
        );
        assert_eq!(
            original.len() - assembled.len(),
            kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD.len()
        );
        for forbidden in [
            "#define __gfx",
            "#undef __gfx",
            "__shfl",
            "__ballot",
            "__activemask",
            "warpSize",
            "__builtin_amdgcn",
            "__gfx",
        ] {
            assert!(
                !assembled.contains(forbidden),
                "{forbidden} in the assembled planner"
            );
        }
        // gfx1100's hot MQ4 bucket reads the `mq4sym` entries of the NEW MQ4-only K1 module.
        let hot = &k.mq4_k1;
        assert_eq!(hot.module, "mq_i4_moe_gemv_mq4only_pm_gfx1100");
        assert_eq!(hot.image, kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100);
        assert_eq!(
            hot.symbols,
            [
                "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
                "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100"
            ]
        );
        // The image names no W2 (`mq2i4`) entry, no qualification (`_acc`) twin, and no
        // `mq_i4_moe_gemv_*` name other than the two served symbols (checked below).
        // (A bare `_acc` would match the code-object metadata key `.actual_access`.)
        for needle in [&b"mq2i4"[..], b"_acc.kd", b"_acc\0"] {
            assert!(
                !contains_bytes(hot.image, needle),
                "{needle:?} in the MQ4-only image"
            );
        }
        for symbol in hot.symbols {
            assert!(
                contains_bytes(hot.image, symbol.as_bytes()),
                "{symbol} missing from its K1 image"
            );
            assert!(
                contains_bytes(hot.image, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
            assert!(!contains_bytes(
                kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
                symbol.as_bytes()
            ));
        }
        let prefix = b"mq_i4_moe_gemv_";
        for at in 0..hot.image.len().saturating_sub(prefix.len()) {
            if hot.image[at..].starts_with(prefix) {
                assert!(
                    hot.symbols
                        .iter()
                        .any(|s| hot.image[at..].starts_with(s.as_bytes())),
                    "the MQ4-only image names an unserved mq_i4_moe_gemv_* entry at {at}"
                );
            }
        }
        // E8P K1 (qualified image, two contracts): the gfx1100 cold bucket of codec 3 layers.
        let e8p = slab_cold_k1_metadata("gfx1100", 3).unwrap();
        assert_eq!(e8p.cold_codec, 3);
        assert_eq!(e8p.k1.module, "e8p_i4_moe_gemv_pm_gfx1100");
        assert_eq!(e8p.k1.image, kernels::E8P_I4_MOE_GEMV_PM_GFX1100);
        assert_eq!(
            e8p.k1.symbols,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1100"
            ]
        );
        for symbol in e8p.k1.symbols {
            assert!(
                contains_bytes(e8p.k1.image, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
            assert!(!symbol.contains("mq2i4") && !symbol.ends_with("_acc"));
            // Each symbol is its own contract: kernels/e8p_i4_moe_gemv.gfx1100.<kind>.contract.json.
            let kind = if symbol.contains("r256_gate_up") {
                "r256_gate_up"
            } else {
                "r128_down"
            };
            let contract =
                std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    format!("../../kernels/e8p_i4_moe_gemv.gfx1100.{kind}.contract.json"),
                ))
                .unwrap();
            assert!(
                contract.contains(&format!("\"symbol\": \"{symbol}\"")),
                "{kind} contract symbol"
            );
            assert!(
                contract.contains("\"group_segment_fixed_bytes\": 1536"),
                "{kind} contract static LDS"
            );
            assert!(
                contract.contains("\"launch_dynamic_lds_bytes\": 0"),
                "{kind} contract dynamic LDS"
            );
        }
        assert!(!e8p.k1.module.contains("mq_i4") && !e8p.k1.module.contains("gfx1201"));
        assert!(!contains_bytes(e8p.k1.image, b"mq_i4_moe_gemv"));
        // The admission matrix admits codec 3 on gfx1100 and runtime selection is the very same
        // table object; W3 (1) stays refused there although it is a gfx1201 codec.
        assert!(crate::qwen4_slab_codec_admitted("gfx1100", 3));
        assert!(std::ptr::eq(
            slab_cold_kernels("gfx1100", 3).unwrap(),
            slab_cold_k1_metadata("gfx1100", 3).unwrap()
        ));
        assert!(slab_cold_kernels("gfx1100", 1).is_none());
        let (module, image, symbol) = k.k3.unwrap();
        assert_eq!(
            (module, symbol),
            ("mq_i4_a4_pm_gfx1100", "mq_i4_a4_r128_k640_gfx1100")
        );
        assert!(contains_bytes(image, format!("{symbol}.kd").as_bytes()));
        let SlabTileMask::Embedded {
            module,
            image,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1100's tile mask is an embedded builder image");
        };
        assert_eq!(
            (module, symbol),
            ("ep_tile_mask_pm_gfx1100", "ep_tile_mask_pm_gfx1100")
        );
        assert!(contains_bytes(image, format!("{symbol}.kd").as_bytes()));
        assert!(!contains_bytes(
            kernels::EP_TILE_MASK_PM_GFX1201,
            symbol.as_bytes()
        ));
    }

    #[test]
    fn cold_tables_are_keyed_by_arch_and_cold_codec_and_refuse_w2() {
        for (arch, codec, served) in [
            ("gfx1201", 1u8, true),
            ("gfx1201", 3, true),
            ("gfx1201", 0, false),
            ("gfx1201", 2, false),
            ("gfx1201", 4, false),
            // gfx1100 serves the E8P cold bucket (K1 `slab_cold_k1_metadata` AND the admission
            // matrix); W3 and W2 stay refused there.
            ("gfx1100", 3, true),
            ("gfx1100", 0, false),
            ("gfx1100", 1, false),
            ("gfx1100", 2, false),
            // gfx1151 serves the W3 cold bucket (K1 metadata AND the admission matrix); W2
            // stays refused there and codec 0 is never cold.
            ("gfx1151", 1, true),
            ("gfx1151", 0, false),
            ("gfx1151", 2, false),
            // gfx1151 E8P has a K1 table row: served exactly when the admission matrix admits it.
            ("gfx1151", 3, crate::qwen4_slab_codec_admitted("gfx1151", 3)),
            ("gfx1151", 4, false),
        ] {
            let table = slab_cold_kernels(arch, codec);
            assert_eq!(table.is_some(), served, "{arch} cold codec {codec}");
            if let Some(table) = table {
                assert_eq!(table.cold_codec, codec);
                assert!(crate::qwen4_slab_codec_admitted(arch, codec));
            }
        }
    }

    #[test]
    fn cold_k1_metadata_is_not_permission() {
        // Metadata inventory: exactly the historical gfx1201 pair, the qualified gfx1100 E8P K1, the
        // gfx1151 W3 K1 and the gfx1151 E8P K1 (the gfx1100 arm under gfx1151 names).
        let mut available = Vec::new();
        for arch in [
            "gfx1201", "gfx1100", "gfx1151", "gfx942", "gfx906", "gfx1101", "",
        ] {
            for codec in 0u8..=255 {
                if let Some(table) = slab_cold_k1_metadata(arch, codec) {
                    assert_eq!(table.cold_codec, codec);
                    available.push((arch, codec));
                }
            }
        }
        assert_eq!(
            available,
            [
                ("gfx1201", 1),
                ("gfx1201", 3),
                ("gfx1100", 3),
                ("gfx1151", 1),
                ("gfx1151", 3)
            ]
        );
        // Runtime selection = metadata AND the shared admission matrix (what
        // `Gpu::ensure_qwen4_slab_kernels` and `Gpu::mq_i4_moe_gemv_slab` consult): a tabled entry
        // is served by the very same table object exactly when admitted, and no other
        // (arch, codec) pair is served. Every tabled entry except gfx1151 E8P is admitted whenever its codec matrix says so;
        // gfx1151 E8P has its K1 table (inventory) but follows the matrix alone.
        for (arch, codec) in available {
            let admitted = crate::qwen4_slab_codec_admitted(arch, codec);
            if (arch, codec) != ("gfx1151", 3) {
                assert!(admitted, "{arch} codec {codec}");
            }
            let selected = slab_cold_kernels(arch, codec);
            assert_eq!(selected.is_some(), admitted, "{arch} codec {codec}");
            if let Some(selected) = selected {
                assert!(std::ptr::eq(
                    selected,
                    slab_cold_k1_metadata(arch, codec).unwrap()
                ));
            }
        }
        assert!(
            slab_cold_kernels("gfx1100", 1).is_none() && slab_cold_kernels("gfx1100", 2).is_none()
        );
        // An MQ4-only layer's preload set never names an E8P module: the hot K1 module is its own
        // image and none of its symbols live in the E8P image.
        let hot = &slab_arch_kernels("gfx1100").unwrap().mq4_k1;
        let e8p = &slab_cold_k1_metadata("gfx1100", 3).unwrap().k1;
        assert_ne!(hot.module, e8p.module);
        assert!(!std::ptr::eq(hot.image, e8p.image));
        for symbol in hot.symbols {
            assert!(!contains_bytes(e8p.image, symbol.as_bytes()));
        }
    }

    #[test]
    fn rowtile_pin_checks_fail_closed() {
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(slab_check_rowtile_pin(b"abc", abc).is_ok());
        assert!(slab_check_rowtile_pin(b"abc", "").is_err());
        assert!(slab_check_rowtile_pin(b"abcd", abc).is_err());
        assert!(slab_check_rowtile_pin(b"abc", &abc.to_uppercase()).is_err());
    }

    #[test]
    fn c1_metadata_is_exact_arch_inventory() {
        let k1 = slab_c1_k1_metadata("gfx1151").unwrap();
        assert_eq!(k1.module, MQ_I4_MOE_GEMV_C1_MODULE);
        assert_eq!(k1.image, kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151);
        assert_eq!(k1.symbols, MQ_I4_MOE_GEMV_C1_SYMBOLS);
        for arch in ["gfx1151x", "gfx1201", "gfx1100", ""] {
            assert!(slab_c1_k1_metadata(arch).is_none());
        }
    }

    #[test]
    fn planner_geometry_rejects_wide_hip_calls_before_writes() {
        for lanes in 1usize..=64 {
            let (grid, block, rows) = slab_plan_geometry(lanes, true).unwrap();
            assert_eq!(grid, [lanes.div_ceil(8) as u32, 1, 1]);
            assert_eq!(block, [128, 1, 1]);
            assert_eq!(rows, lanes as i32);
            if lanes <= 8 {
                assert_eq!(
                    slab_plan_geometry(lanes, false),
                    Some(([1, 1, 1], [512, 1, 1], (lanes * 10) as i32)),
                );
            } else {
                assert!(slab_plan_geometry(lanes, false).is_none());
            }
        }
        for lanes in [0, 65, usize::MAX] {
            assert!(slab_plan_geometry(lanes, false).is_none());
            assert!(slab_plan_geometry(lanes, true).is_none());
        }
    }
    /// Inventory is architecture-specific; serving remains codec-admission-gated.
    #[test]
    fn rowtiles_cover_every_existing_slab_bucket_without_admitting_codecs() {
        assert_eq!(SLAB_MAX_LANES, 64);
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            let k = slab_arch_kernels(arch).unwrap();
            assert!(slab_wide_capable(k, None), "{arch}: MQ4-only layer");
            assert!(slab_rowtile_planner(arch).is_some());
            for cold_codec in [1, 3] {
                if let Some(cold) = slab_cold_k1_metadata(arch, cold_codec) {
                    assert!(
                        slab_wide_capable(k, Some(cold)),
                        "{arch}: codec {cold_codec}"
                    );
                    assert_eq!(
                        slab_cold_kernels(arch, cold_codec).is_some(),
                        crate::qwen4_slab_codec_admitted(arch, cold_codec),
                    );
                }
            }
            for k1 in std::iter::once(&k.mq4_k1).chain(
                [1, 3]
                    .into_iter()
                    .filter_map(|codec| slab_cold_k1_metadata(arch, codec).map(|c| &c.k1)),
            ) {
                let tile = slab_rowtile_k1(k1).unwrap();
                for symbol in tile.symbols {
                    assert!(
                        contains_bytes(tile.image, format!("{symbol}.kd").as_bytes()),
                        "{arch}: {symbol}"
                    );
                }
                for lanes in 1..=64 {
                    for (down, m) in [(false, SLAB_GATE_UP_M), (true, SLAB_DOWN_M)] {
                        let pick = slab_rowtile_pick(k1, down, m, lanes).unwrap();
                        assert_eq!(pick.symbol, tile.symbols[down as usize]);
                        assert_eq!(
                            pick.grid,
                            [m.div_ceil(8) as u32, 80, lanes.div_ceil(8) as u32]
                        );
                        assert_eq!(pick.block, [256, 1, 1]);
                    }
                }
                for lanes in [0, 65, usize::MAX] {
                    assert!(slab_rowtile_pick(k1, false, SLAB_GATE_UP_M, lanes).is_none());
                }
                assert!(slab_rowtile_pick(k1, false, 0, 1).is_none());
                assert!(slab_rowtile_pick(k1, false, i32::MAX as usize + 1, 1).is_none());
                for m in [1usize, 2, 7, 8, 9, 1279] {
                    assert_eq!(
                        slab_rowtile_pick(k1, false, m, 64).unwrap().grid[0],
                        m.div_ceil(8) as u32
                    );
                }
            }
        }
        for arch in ["gfx1151x", "gfx1101", "gfx942", ""] {
            assert!(slab_rowtile_planner(arch).is_none());
        }
    }

    #[test]
    fn rowtile_images_keep_the_frozen_abis() {
        for (arch_name, arch) in [
            ("gfx1151", Arch::Gfx1151),
            ("gfx1201", Arch::Gfx1201),
            ("gfx1100", Arch::Gfx1100),
        ] {
            let k = slab_arch_kernels(arch_name).unwrap();
            for k1 in std::iter::once(&k.mq4_k1).chain(
                [1, 3]
                    .into_iter()
                    .filter_map(|codec| slab_cold_k1_metadata(arch_name, codec).map(|c| &c.k1)),
            ) {
                let tile = slab_rowtile_k1(k1).unwrap();
                let (_, images) = Source::read(tile.image, arch).expect("tile image metadata");
                for symbol in tile.symbols {
                    let image = images
                        .iter()
                        .find(|image| image.name == symbol)
                        .unwrap_or_else(|| panic!("{arch_name}: missing {symbol}"));
                    let metadata = &image.metadata.parsed;
                    assert_eq!(metadata.kernarg_segment_size, 96, "{symbol}");
                    let args: Vec<_> = metadata
                        .args
                        .iter()
                        .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                        .collect();
                    assert_eq!(args.len(), MQ_I4_MOE_GEMV_KERNARG_COUNT, "{symbol}");
                    for (arg, offset) in args.iter().zip([
                        0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92,
                    ]) {
                        assert_eq!(arg.offset, offset, "{symbol}: {}", arg.name);
                    }
                    assert_eq!(args[15].name, "n_experts");
                    assert_eq!(args[16].name, "reserved");
                }
            }
            let (symbol, bytes) = slab_rowtile_planner(arch_name).unwrap();
            let (_, images) = Source::read(bytes, arch).expect("planner image metadata");
            let metadata = &images
                .iter()
                .find(|image| image.name == symbol)
                .expect("planner export")
                .metadata
                .parsed;
            assert_eq!(metadata.kernarg_segment_size, 40);
            let offsets: Vec<_> = metadata
                .args
                .iter()
                .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                .map(|arg| arg.offset)
                .collect();
            assert_eq!(offsets, [0, 8, 16, 24, 28, 32]);
        }
    }

    #[test]
    fn bucket_of_codec_on_follows_admission_with_cold_first() {
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 1), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 3), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 2), None);
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 1), None);
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 2), None);
        // The gfx1100 E8P cold bucket is a served bucket 0 (cold = 0, MQ4 = 1).
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 3), Some(0));
        // gfx1151 serves MQ4 (hot bucket 1) and W3 (cold bucket 0); W2 stays refused there. Its E8P
        // cold bucket (K1 table present) follows the shared admission matrix, never the table.
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 1), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 2), None);
        assert_eq!(
            slab_bucket_of_codec_on("gfx1151", 3),
            crate::qwen4_slab_codec_admitted("gfx1151", 3).then_some(0)
        );
        // The historical gfx1201 byte mapping is unchanged.
        assert_eq!((slab_bucket_of_codec(1), slab_bucket_of_codec(0)), (0, 1));
    }

    /// K1-Cn: pinned symbol table (K1 bucket order), grid `[M / 8, slots, 1]`, block 256; K1 keeps
    /// the one-wave-per-row geometry; the default dispatch picks Cn only on gfx1151 for 2..=8 lanes.
    #[test]
    fn cn_symbols_geometry_and_default_choice_are_pinned() {
        assert_eq!(
            MQ_I4_MOE_GEMV_CN_SYMBOLS,
            [
                [
                    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
                    "mq_i4_moe_gemv_cn_mq3i4_r128_down"
                ],
                [
                    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
                    "mq_i4_moe_gemv_cn_mq4sym_r128_down"
                ],
            ]
        );
        for (bucket, codec) in ["mq3i4", "mq4sym"].into_iter().enumerate() {
            assert_eq!(
                MQ_I4_MOE_GEMV_CN_SYMBOLS[bucket][0],
                format!("mq_i4_moe_gemv_cn_{codec}_r256_gate_up")
            );
            assert_eq!(
                MQ_I4_MOE_GEMV_CN_SYMBOLS[bucket][1],
                format!("mq_i4_moe_gemv_cn_{codec}_r128_down")
            );
        }
        for lanes in 1..=SLAB_NARROW_MAX_LANES {
            let slots = SLAB_TOP_K * lanes;
            for (bucket, down, m) in [
                (0, false, SLAB_GATE_UP_M),
                (1, false, SLAB_GATE_UP_M),
                (0, true, SLAB_DOWN_M),
                (1, true, SLAB_DOWN_M),
            ] {
                assert_eq!(m % MQ_I4_MOE_GEMV_CN_ROWS_PER_WG, 0);
                let mq4 = &slab_arch_kernels("gfx1151").unwrap().mq4_k1;
                let cn = slab_k1_pick(mq4, SlabK1Module::Cn, bucket, down, m, slots);
                assert_eq!(cn.module_key, "mq_i4_moe_gemv_cn_pm_gfx1151");
                assert_eq!(cn.symbol, MQ_I4_MOE_GEMV_CN_SYMBOLS[bucket][down as usize]);
                assert_eq!(cn.grid, [(m / 8) as u32, slots as u32, 1]);
                assert_eq!(cn.block, [256, 1, 1]);
                for arch in ["gfx1151", "gfx1201"] {
                    let entry = if bucket == 0 {
                        &slab_cold_k1_metadata(arch, 1).unwrap().k1
                    } else {
                        &slab_arch_kernels(arch).unwrap().mq4_k1
                    };
                    let k1 = slab_k1_pick(entry, SlabK1Module::K1, bucket, down, m, slots);
                    assert_eq!(
                        (k1.module_key, k1.image.as_ptr()),
                        (entry.module, entry.image.as_ptr()),
                        "{arch}"
                    );
                    assert_eq!(k1.symbol, MQ_I4_MOE_GEMV_SYMBOLS[bucket][down as usize]);
                    assert_eq!(k1.grid, [m as u32, slots as u32, 1]);
                    assert_eq!(k1.block, [32, 1, 1]);
                }
            }
        }
        for lanes in 0..=9 {
            let cn_range = (2..=8).contains(&lanes);
            assert_eq!(
                slab_k1_default_module(true, lanes, true) == SlabK1Module::Cn,
                cn_range
            );
            assert_eq!(slab_k1_default_module(true, lanes, false), SlabK1Module::K1);
            assert_eq!(slab_k1_default_module(false, lanes, true), SlabK1Module::K1);
        }
    }

    /// The embedded K1-Cn metadata matches the bounded 96-byte K1 ABI (`n_experts` @88, zero
    /// `reserved` @92) and the shared launcher's kernelParams count.
    #[test]
    fn cn_kernel_params_match_embedded_metadata() {
        let (_, images) = Source::read(kernels::MQ_I4_MOE_GEMV_CN_PM_GFX1151, Arch::Gfx1151)
            .expect("embedded K1-Cn metadata must parse");
        let mut mismatches = Vec::new();
        for symbol in MQ_I4_MOE_GEMV_CN_SYMBOLS.iter().flatten() {
            let image = images
                .iter()
                .find(|image| image.name == *symbol)
                .unwrap_or_else(|| panic!("missing K1-Cn symbol {symbol}"));
            let metadata = &image.metadata.parsed;
            assert_eq!(metadata.kernarg_segment_size, 96, "{symbol}");
            let find = |name: &str| {
                metadata
                    .args
                    .iter()
                    .find(|arg| arg.name == name)
                    .map(|arg| (arg.offset, arg.size, arg.value_kind.as_str()))
            };
            assert_eq!(find("n_experts"), Some((88, 4, "by_value")), "{symbol}");
            let reserved = metadata.args.last().expect("K1-Cn arguments");
            assert_eq!(
                (
                    reserved.name.as_str(),
                    reserved.offset,
                    reserved.size,
                    reserved.value_kind.as_str()
                ),
                ("reserved", 92, 4, "by_value"),
                "{symbol}"
            );
            let explicit = metadata
                .args
                .iter()
                .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                .count();
            if explicit != MQ_I4_MOE_GEMV_KERNARG_COUNT {
                mismatches.push(format!(
                    "{symbol}: kernelParams={MQ_I4_MOE_GEMV_KERNARG_COUNT}, declared={explicit}"
                ));
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    #[test]
    fn k1_kernel_params_match_embedded_metadata() {
        let mut mismatches = Vec::new();
        let mut checked = 0;
        for (arch, bytes) in [
            (Arch::Gfx1151, kernels::MQ_I4_MOE_GEMV_PM_GFX1151),
            (Arch::Gfx1201, kernels::MQ_I4_MOE_GEMV_PM_GFX1201),
        ] {
            let (_, images) = Source::read(bytes, arch).expect("embedded K1 metadata must parse");
            for symbol in MQ_I4_MOE_GEMV_SYMBOLS.iter().flatten() {
                let image = images
                    .iter()
                    .find(|image| image.name == *symbol)
                    .unwrap_or_else(|| panic!("{arch:?}: missing K1 symbol {symbol}"));
                let metadata = &image.metadata.parsed;
                assert_eq!(metadata.kernarg_segment_size, 96, "{arch:?}: {symbol}");
                let reserved = metadata.args.last().expect("K1 arguments");
                assert_eq!(
                    (
                        reserved.name.as_str(),
                        reserved.offset,
                        reserved.size,
                        reserved.value_kind.as_str()
                    ),
                    ("reserved", 92, 4, "by_value"),
                    "{arch:?}: {symbol}"
                );
                let explicit = metadata
                    .args
                    .iter()
                    .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                    .count();
                if explicit != MQ_I4_MOE_GEMV_KERNARG_COUNT {
                    mismatches.push(format!(
                        "{arch:?}: {symbol}: kernelParams={MQ_I4_MOE_GEMV_KERNARG_COUNT}, declared={explicit}"
                    ));
                }
                checked += 1;
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
        println!("checked {checked} embedded K1 entries against kernelParams");
    }

    /// The embedded K1-C1 metadata matches the 80-byte ABI and the launcher's kernelParams count.
    #[test]
    fn c1_kernel_params_match_embedded_metadata() {
        let (_, images) = Source::read(kernels::MQ_I4_MOE_GEMV_C1_PM_GFX1151, Arch::Gfx1151)
            .expect("embedded K1-C1 metadata must parse");
        let mut mismatches = Vec::new();
        for symbol in MQ_I4_MOE_GEMV_C1_SYMBOLS {
            let image = images
                .iter()
                .find(|image| image.name == symbol)
                .unwrap_or_else(|| panic!("missing K1-C1 symbol {symbol}"));
            let metadata = &image.metadata.parsed;
            assert_eq!(
                metadata.kernarg_segment_size as usize, MQ_I4_MOE_GEMV_C1_KERNARG_BYTES,
                "{symbol}"
            );
            let find = |name: &str| {
                metadata
                    .args
                    .iter()
                    .find(|arg| arg.name == name)
                    .map(|arg| (arg.offset, arg.size, arg.value_kind.as_str()))
            };
            assert_eq!(find("n_experts"), Some((72, 4, "by_value")), "{symbol}");
            let last = metadata.args.last().expect("K1-C1 arguments");
            assert_eq!(
                (
                    last.name.as_str(),
                    last.offset,
                    last.size,
                    last.value_kind.as_str()
                ),
                ("reserved", 76, 4, "by_value"),
                "{symbol}"
            );
            let explicit = metadata
                .args
                .iter()
                .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                .count();
            if explicit != MQ_I4_MOE_GEMV_C1_KERNARG_COUNT {
                mismatches.push(format!(
                    "{symbol}: kernelParams={MQ_I4_MOE_GEMV_C1_KERNARG_COUNT}, declared={explicit}"
                ));
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }

    /// K1-C1 / K1-Cn serve MQ4-only and W3 layers only; E8P, W2 and unknown cold codecs keep K1.
    #[test]
    fn rows8_kernels_serve_mq4_only_and_w3_layers() {
        assert!(k1_rows8_cold_codec(0) && k1_rows8_cold_codec(1));
        assert!(!k1_rows8_cold_codec(2) && !k1_rows8_cold_codec(3) && !k1_rows8_cold_codec(4));
    }

    /// The K1-C1 kernarg is the frozen 80-byte ABI: six pointers, then `M`, `K`,
    /// `uniform_codec`, `x_row_div`, `x_row_stride`, `y_slot_stride`, `n_experts`, zero `reserved`.
    #[test]
    fn c1_kernarg_is_the_frozen_80_byte_abi() {
        for (down, n_experts) in [(false, 512u32), (true, 1234)] {
            let geo = slab_c1_launch(down);
            let ptr = |n: usize| (0x1000 * n) as *mut c_void;
            let blob = slab_c1_kernarg(
                ptr(1),
                ptr(2),
                ptr(3),
                ptr(4),
                ptr(5),
                ptr(6),
                -1,
                n_experts,
                &geo,
            );
            let bytes = blob.as_bytes();
            assert_eq!(bytes.len(), MQ_I4_MOE_GEMV_C1_KERNARG_BYTES);
            for n in 1..=6usize {
                let off = 8 * (n - 1);
                assert_eq!(
                    u64::from_ne_bytes(bytes[off..off + 8].try_into().unwrap()),
                    (0x1000 * n) as u64
                );
            }
            let word = |off: usize| u32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap());
            assert_eq!(word(48), geo.m);
            assert_eq!(word(52), geo.k);
            assert_eq!(word(56), u32::MAX, "uniform_codec -1 = table");
            assert_eq!(word(60), geo.x_row_div);
            assert_eq!(word(64), geo.k, "x_row_stride = K");
            assert_eq!(word(68), geo.y_slot_stride);
            assert_eq!(word(72), n_experts, "n_experts@72");
            assert_eq!(word(76), 0, "reserved@76");
            assert_eq!(geo.m as usize % MQ_I4_MOE_GEMV_C1_ROWS_PER_WG, 0);
        }
    }
}

/// CPU models of the S1 ordered panel combine (`slowlink_combine_panel` + `slowlink_combine_pad`)
/// against the canonical monolithic BF16-row combine (`moe_combine_order_top10` +
/// `moe_down_combine_grouped_top10_bf16in`).  No HIP.
pub mod slowlink_selftest {
    #![allow(dead_code)]
    use super::slowlink_panel_owns;

    const K: usize = 10;
    /// Dimensions of the default fixture ([`make_case`]); every model function reads the
    /// case's own `h` and `n_exp`, so [`make_case_dims`] drives it at any hidden width and
    /// expert count (the real artifact's 2560 / 512 included).
    const DEFAULT_EXPERTS: usize = 512;
    const CLASSES: usize = 12;
    const DEFAULT_HIDDEN: usize = 24;

    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        pub fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        pub fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    pub fn rbf(x: f32) -> f32 {
        if x.is_nan() {
            return f32::NAN;
        }
        let u = x.to_bits();
        if (!u & 0x7f80_0000) != 0 {
            f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
        } else {
            f32::from_bits(u & 0xffff_0000)
        }
    }
    pub fn bf(x: f32) -> u16 {
        (rbf(x).to_bits() >> 16) as u16
    }
    pub fn wd(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }
    pub fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// A combine fixture: window grouped rows (pad16 per expert) with BF16 down rows,
    /// route slots, weights and initial accumulators.
    pub struct Case {
        pub tokens: usize,
        pub h: usize,
        pub n_exp: usize,
        /// Window grouped rows (pad16 per expert).
        pub g: usize,
        /// First grouped row of every expert, `[n_exp + 1]`.
        pub start: Vec<usize>,
        pub down: Vec<u16>,
        pub inv: Vec<i32>,
        pub topk: Vec<i32>,
        pub w: Vec<f32>,
        pub init: Vec<f32>,
        /// Adversarial class of every token (`[tokens]`).
        pub class: Vec<usize>,
    }

    fn palette(class: usize, rng: &mut Rng) -> u16 {
        let pick = |rng: &mut Rng, v: &[f32]| bf(v[rng.below(v.len())]);
        match class {
            6 => pick(rng, &[3.38e38, -3.38e38, 1.0e30, 2.0, 3.0e38]),
            7 => pick(rng, &[1.0e30, -1.0e30, 1.0, 1.0e-3, -1.0, 7.0e29]),
            8 => pick(rng, &[0.00390625, 1.0, -0.00390625, 0.005859375, -1.0]),
            9 => [0x0000u16, 0x8000, bf(1.0), bf(-1.0), bf(0.5)][rng.below(5)],
            11 => {
                let m = (1 + rng.below(0x7f)) as u16;
                if rng.below(2) == 0 {
                    m
                } else {
                    m | 0x8000
                }
            }
            _ => bf(rng.unit() * 2.0),
        }
    }

    fn weight(class: usize, rng: &mut Rng) -> f32 {
        match class {
            6 => [1.0f32, 2.0, -1.0, 1.0][rng.below(4)],
            7 => [1.0f32, -1.0, 1.0, 0.5][rng.below(4)],
            8 => 1.0,
            9 => [0.0f32, -0.0, 1.0, -1.0, 0.25][rng.below(5)],
            11 => [1.0e-30f32, -1.0e-30, 3.0e-39, 1.0][rng.below(4)],
            _ => rng.unit(),
        }
    }

    fn distinct(rng: &mut Rng, n_exp: usize) -> [i32; K] {
        let mut ids = [0i32; K];
        let mut n = 0;
        while n < K {
            let c = rng.below(n_exp) as i32;
            if !ids[..n].contains(&c) {
                ids[n] = c;
                n += 1;
            }
        }
        ids
    }

    /// [`make_case_dims`] at the default dimensions (24 columns, 512 experts).
    pub fn make_case(seed: u64, tokens_per_class: usize) -> Case {
        make_case_dims(seed, tokens_per_class, DEFAULT_HIDDEN, DEFAULT_EXPERTS)
    }

    /// Every adversarial class, grouped the way the stable scatter groups (`(expert, slot)`
    /// order, each expert padded to 16), with some valid ids whose grouped row is invalid, over
    /// `hidden` columns and `n_exp >= K` experts.  The fixed ids of the classes are taken modulo
    /// `n_exp` (identical at 512) and the out-of-range ones are relative to it.
    pub fn make_case_dims(seed: u64, tokens_per_class: usize, hidden: usize, n_exp: usize) -> Case {
        assert!(n_exp >= K && hidden > 0, "the fixture needs >= 10 experts");
        let ne = n_exp as i32;
        let at = |e: usize| (e % n_exp) as i32;
        let mut rng = Rng(seed | 1);
        let tokens = CLASSES * tokens_per_class;
        let mut topk = vec![0i32; tokens * K];
        let mut w = vec![0f32; tokens * K];
        let mut class_of = vec![0usize; tokens];
        for tok in 0..tokens {
            let c = tok % CLASSES;
            class_of[tok] = c;
            let ids: [i32; K] = match c {
                1 => [at(7); K],
                2 => [
                    at(5),
                    at(5),
                    at(9),
                    at(9),
                    at(9),
                    at(1),
                    at(1),
                    at(300),
                    at(300),
                    ne - 1,
                ],
                3 => [
                    -1,
                    ne,
                    at(40),
                    ne + 488,
                    -7,
                    at(3),
                    i32::MIN,
                    i32::MAX,
                    at(17),
                    at(17),
                ],
                4 => [-1; K],
                5 => {
                    let mut v = [if rng.below(2) == 0 { -1 } else { ne + 88 }; K];
                    v[rng.below(K)] = rng.below(n_exp) as i32;
                    v
                }
                8 => [2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map(|e| at(e)),
                9 => [4, 4, 4, 100, 100, 200, 200, 200, 511, 0].map(|e| at(e)),
                _ => distinct(&mut rng, n_exp),
            };
            for r in 0..K {
                topk[tok * K + r] = ids[r];
                w[tok * K + r] = weight(c, &mut rng);
            }
        }
        // Stable grouping: (expert, flat slot), pad16 per expert.
        let mut count = vec![0usize; n_exp];
        for &id in &topk {
            if (0..ne).contains(&id) {
                count[id as usize] += 1;
            }
        }
        let mut start = vec![0usize; n_exp + 1];
        for e in 0..n_exp {
            start[e + 1] = start[e] + count[e].div_ceil(16) * 16;
        }
        let g = start[n_exp].max(16);
        let mut fill = vec![0usize; n_exp];
        let mut inv = vec![-1i32; tokens * K];
        let mut down = vec![0u16; g * hidden];
        for slot in 0..tokens * K {
            let id = topk[slot];
            if !(0..ne).contains(&id) {
                continue;
            }
            let e = id as usize;
            let row = start[e] + fill[e];
            fill[e] += 1;
            let c = class_of[slot / K];
            for col in 0..hidden {
                down[row * hidden + col] = palette(c, &mut rng);
            }
            // Class 10: most slots lose their grouped row but keep their (valid) id.
            inv[slot] = if c == 10 && rng.below(3) != 0 {
                [-1i32, g as i32 + 7, i32::MAX][rng.below(3)]
            } else {
                row as i32
            };
        }
        let init = (0..tokens * hidden)
            .map(|i| match i % 5 {
                0 => 0.0,
                1 => -0.0,
                2 => rbf(rng.unit()),
                3 => rbf(1.0e30),
                _ => rbf(-3.0e38),
            })
            .collect();
        Case {
            tokens,
            h: hidden,
            n_exp,
            g,
            start,
            down,
            inv,
            topk,
            w,
            init,
            class: class_of,
        }
    }

    /// The ten (grouped row or -1, BF16 weight) slots of a token in
    /// `moe_combine_order_top10`'s order, then (-1, +0) pads.
    fn order_slots(c: &Case, tok: usize) -> [(i32, f32); K] {
        let base = tok * K;
        let mut used = [false; K];
        for r in 0..K {
            let id = c.topk[base + r];
            used[r] = !(0..c.n_exp as i32).contains(&id);
        }
        let mut out = [(-1i32, 0.0f32); K];
        for slot in out.iter_mut() {
            let mut sel = None;
            let mut sel_id = i32::MAX;
            for r in 0..K {
                if !used[r] && c.topk[base + r] < sel_id {
                    sel_id = c.topk[base + r];
                    sel = Some(r);
                }
            }
            let Some(r) = sel else { continue };
            used[r] = true;
            let row = c.inv[base + r];
            *slot = (
                if row >= 0 && (row as usize) < c.g {
                    row
                } else {
                    -1
                },
                rbf(c.w[base + r]),
            );
        }
        out
    }

    /// The canonical monolithic BF16-row combine on the CPU (`hidden` and `n_exp` come from
    /// the case): residual rounded to BF16, then the ten ranks in ascending expert id.
    pub fn canonical(c: &Case) -> Vec<f32> {
        let h = c.h;
        let mut out = vec![0f32; c.tokens * h];
        for tok in 0..c.tokens {
            let slots = order_slots(c, tok);
            for col in 0..h {
                let mut acc = rbf(c.init[tok * h + col]);
                for &(row, wt) in &slots {
                    let y = if row >= 0 {
                        wd(c.down[row as usize * h + col])
                    } else {
                        0.0
                    };
                    acc = rbf(acc + rbf(rbf(y) * wt));
                }
                out[tok * h + col] = acc;
            }
        }
        out
    }

    /// One panel of `slowlink_combine_panel`.
    fn apply_panel(c: &Case, acc: &mut [f32], row_base: usize, rows: usize, own: (usize, usize)) {
        for tok in 0..c.tokens {
            let base = tok * K;
            let mut ents: Vec<(i32, usize)> = (0..K)
                .filter(|&r| {
                    slowlink_panel_owns(
                        c.topk[base + r],
                        c.inv[base + r],
                        c.n_exp,
                        c.g,
                        row_base,
                        rows,
                        own.0,
                        own.1,
                    )
                })
                .map(|r| (c.topk[base + r], r))
                .collect();
            ents.sort();
            for (_, r) in ents {
                let row = c.inv[base + r];
                let wt = rbf(c.w[base + r]);
                for col in 0..c.h {
                    let y = if row >= 0 && (row as usize) < c.g {
                        wd(c.down[row as usize * c.h + col])
                    } else {
                        0.0
                    };
                    let a = &mut acc[tok * c.h + col];
                    *a = rbf(*a + rbf(rbf(y) * wt));
                }
            }
        }
    }

    /// `slowlink_combine_pad`.
    fn apply_pad(c: &Case, acc: &mut [f32]) {
        for tok in 0..c.tokens {
            let valid = (0..K)
                .filter(|&r| (0..c.n_exp as i32).contains(&c.topk[tok * K + r]))
                .count();
            for col in 0..c.h {
                let a = &mut acc[tok * c.h + col];
                for _ in valid..K {
                    *a = rbf(*a + rbf(0.0 * 0.0));
                }
            }
        }
    }

    fn expert_of_row(c: &Case, row: usize) -> usize {
        c.start.partition_point(|&s| s <= row) - 1
    }

    /// Panels `[cuts[k], cuts[k+1])` of the window's rows with the ownership partition the
    /// scheduler builds: the last expert of each panel closes its range, the last panel
    /// closes at the case's expert count.
    fn panels(c: &Case, cuts: &[usize]) -> Vec<(usize, usize, (usize, usize))> {
        let mut out = Vec::new();
        let mut prev_hi = 0;
        for (k, win) in cuts.windows(2).enumerate() {
            let (lo, hi) = (win[0], win[1]);
            let last = k + 2 == cuts.len();
            let own_hi = if last {
                c.n_exp
            } else if hi > lo {
                (expert_of_row(c, hi - 1) + 1).max(prev_hi)
            } else {
                prev_hi
            };
            out.push((lo, hi - lo, (prev_hi, own_hi)));
            prev_hi = own_hi;
        }
        out
    }

    fn run(c: &Case, cuts: &[usize], ascending: bool, pad: bool) -> Vec<f32> {
        let mut acc: Vec<f32> = c.init.iter().map(|&v| rbf(v)).collect();
        let ps = panels(c, cuts);
        let order: Vec<usize> = if ascending {
            (0..ps.len()).collect()
        } else {
            (0..ps.len()).rev().collect()
        };
        for i in order {
            let (base, rows, own) = ps[i];
            apply_panel(c, &mut acc, base, rows, own);
        }
        if pad {
            apply_pad(c, &mut acc);
        }
        acc
    }

    fn diffs(a: &[f32], b: &[f32]) -> usize {
        a.iter().zip(b).filter(|(x, y)| !same(**x, **y)).count()
    }

    /// Random row cuts (some mid-expert, some not tile-aligned) plus the one-row and whole-window
    /// extremes.
    fn partitions(c: &Case, rng: &mut Rng) -> Vec<Vec<usize>> {
        let mut out = vec![vec![0, c.g]];
        out.push((0..=c.g.div_ceil(16)).map(|i| (i * 16).min(c.g)).collect());
        for _ in 0..24 {
            let k = 1 + rng.below(40);
            let mut cuts: Vec<usize> = (0..k)
                .map(|_| {
                    let r = rng.below(c.g + 1);
                    if rng.below(3) == 0 {
                        r
                    } else {
                        r / 16 * 16
                    }
                })
                .collect();
            cuts.extend([0, c.g]);
            cuts.sort();
            cuts.dedup();
            out.push(cuts);
        }
        // Cuts exactly inside the heavy experts (7: all ten ranks x tokens; 5/9/300).
        for e in [7usize, 5, 9, 300, 511].map(|e| e % c.n_exp) {
            let mid = (c.start[e] + c.start[e + 1]) / 2;
            let mut cuts = vec![0, c.start[e], mid, c.start[e + 1], c.g];
            cuts.sort();
            cuts.dedup();
            out.push(cuts);
        }
        out
    }

    #[test]
    fn slowlink_combine_panels_equal_the_canonical_combine() {
        for seed in [0x51_0e_1u64, 0xC0B1, 0xDEAD_BEEF, 7] {
            let case = make_case(seed, 6);
            let want = canonical(&case);
            let mut rng = Rng(seed ^ 0x9e37_79b9);
            for cuts in partitions(&case, &mut rng) {
                let got = run(&case, &cuts, true, true);
                assert_eq!(diffs(&got, &want), 0, "seed {seed:#x}, cuts {cuts:?}");
            }
        }
    }

    /// Row cuts of a sweep whose groups hold `per_group` consecutive experts of `members`
    /// (ascending expert ids), each group's contiguous grouped-row run chunked at `max_rows`
    /// (the scheduler's `max_grouped_rows`): the panels of one binding per group.
    fn group_cuts(c: &Case, members: &[usize], per_group: usize, max_rows: usize) -> Vec<usize> {
        let mut cuts = vec![0usize];
        for group in members.chunks(per_group) {
            let (first, last) = (group[0], group[group.len() - 1]);
            let (lo, hi) = (c.start[first], c.start[last + 1]);
            let mut base = lo;
            while base < hi {
                base = (base + max_rows).min(hi);
                cuts.push(base);
            }
        }
        cuts.push(c.g);
        cuts.sort();
        cuts.dedup();
        cuts
    }

    /// TouchedOnly: S1's per-panel ordered combine over the TOUCHED experts' groups (only the
    /// experts holding a grouped row are bound; the untouched ones between them hold no row, so
    /// no panel spans anything but touched rows) equals the canonical combine, and so equals
    /// the stream-all combine over 32-expert page-order groups, bit for bit.
    #[test]
    fn slowlink_combine_touched_only_panels_equal_stream_all_panels() {
        for seed in [0x70c4_ed1u64, 0xC0B1, 7, 0xFACE] {
            let case = make_case(seed, 6);
            let want = canonical(&case);
            let touched: Vec<usize> = (0..case.n_exp)
                .filter(|&e| case.start[e + 1] > case.start[e])
                .collect();
            let all: Vec<usize> = (0..case.n_exp).collect();
            assert!(
                !touched.is_empty() && touched.len() < case.n_exp,
                "the fixture must leave experts untouched ({} of {})",
                touched.len(),
                case.n_exp
            );
            for per_group in [1usize, 7, 32] {
                for max_rows in [16usize, 2048, usize::MAX / 2] {
                    let sparse = group_cuts(&case, &touched, per_group, max_rows);
                    let dense = group_cuts(&case, &all, per_group, max_rows);
                    let sparse_out = run(&case, &sparse, true, true);
                    let dense_out = run(&case, &dense, true, true);
                    assert_eq!(
                        diffs(&sparse_out, &want),
                        0,
                        "seed {seed:#x} touched groups of {per_group}, rows <= {max_rows}"
                    );
                    assert_eq!(
                        diffs(&sparse_out, &dense_out),
                        0,
                        "seed {seed:#x} touched vs stream-all groups of {per_group}"
                    );
                    // The ownership ranges still partition [0, n_exp) in panel order.
                    let ps = panels(&case, &sparse);
                    let mut next = 0;
                    for &(_, _, (lo, hi)) in &ps {
                        assert_eq!(lo, next);
                        next = hi;
                    }
                    assert_eq!(next, case.n_exp);
                }
            }
        }
    }

    #[test]
    fn slowlink_combine_each_valid_slot_is_applied_exactly_once() {
        for seed in [3u64, 0xFEED] {
            let case = make_case(seed, 5);
            let mut rng = Rng(seed ^ 0x77);
            for cuts in partitions(&case, &mut rng) {
                let ps = panels(&case, &cuts);
                // The ownership ranges partition [0, n_exp) in panel order.
                let mut next = 0;
                for &(_, _, (lo, hi)) in &ps {
                    assert_eq!(lo, next, "ownership gap/overlap, cuts {cuts:?}");
                    assert!(hi >= lo);
                    next = hi;
                }
                assert_eq!(next, case.n_exp);
                for slot in 0..case.tokens * K {
                    let id = case.topk[slot];
                    let owners = ps
                        .iter()
                        .filter(|&&(base, rows, (lo, hi))| {
                            slowlink_panel_owns(
                                id,
                                case.inv[slot],
                                case.n_exp,
                                case.g,
                                base,
                                rows,
                                lo,
                                hi,
                            )
                        })
                        .count();
                    let valid = (0..case.n_exp as i32).contains(&id);
                    assert_eq!(
                        owners,
                        usize::from(valid),
                        "slot {slot} id {id}, cuts {cuts:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn slowlink_combine_terminal_pad_runs_once_and_only_matters_for_negative_zero() {
        let case = make_case(0xA11CE, 8);
        let want = canonical(&case);
        let cuts = vec![0, case.g / 3, case.g];
        // Without the terminal products, only accumulators that are still -0 differ: tokens
        // with fewer than ten valid ids (signed-zero and invalid-id classes).
        let bare = run(&case, &cuts, true, false);
        let wrong = diffs(&bare, &want);
        assert!(wrong > 0, "the fixture must exercise a -0 terminal pad");
        for (i, (a, b)) in bare.iter().zip(&want).enumerate() {
            if !same(*a, *b) {
                assert_eq!(a.to_bits(), 0x8000_0000, "elt {i}: only -0 may differ");
                assert_eq!(b.to_bits(), 0, "elt {i}: the pad turns -0 into +0");
            }
        }
        // One pad per layer (not per panel): padding twice is idempotent but a per-panel
        // pad would already have been wrong for a panel that precedes a negative product.
        let mut twice = run(&case, &cuts, true, true);
        apply_pad(&case, &mut twice);
        assert_eq!(diffs(&twice, &want), 0);
    }

    #[test]
    fn slowlink_combine_panel_order_matters() {
        // Negative control: ascending expert order is the contract; reversed panels differ.
        let case = make_case(0xBADD, 8);
        let want = canonical(&case);
        let cuts: Vec<usize> = (0..=case.g.div_ceil(16))
            .map(|i| (i * 16).min(case.g))
            .collect();
        assert_eq!(diffs(&run(&case, &cuts, true, true), &want), 0);
        assert!(diffs(&run(&case, &cuts, false, true), &want) > 0);
    }

    #[test]
    fn slowlink_combine_skipped_slots_add_no_zero_terms() {
        // A one-slot token under a split with absent panels must not see +0 additions from
        // panels that do not own its slot: -0 stays -0 until the single terminal pad.
        let mut case = make_case(0x2, 2);
        // Token 0: one valid slot (expert 3, positive weight, +0 payload), accumulator -0.
        for r in 0..K {
            case.topk[r] = -1;
            case.inv[r] = -1;
        }
        case.topk[4] = 3;
        case.w[4] = -1.0;
        let row = case.start[3];
        case.inv[4] = row as i32;
        for col in 0..case.h {
            case.down[row * case.h + col] = 0;
            case.init[col] = -0.0;
        }
        let want = canonical(&case);
        // -0 + (+0 * -1 = -0) stays -0, then nine +0 pads make it +0.
        for cuts in [
            vec![0, case.g],
            vec![0, row, row + 16, case.g],
            (0..=case.g / 16).map(|i| i * 16).collect(),
        ] {
            let got = run(&case, &cuts, true, true);
            assert_eq!(diffs(&got[..case.h], &want[..case.h]), 0, "cuts {cuts:?}");
            assert!(got[..case.h].iter().all(|v| v.to_bits() == 0));
        }
    }

    /// The model reads each case's own hidden width and expert count: the canonical equality
    /// and exactly-once ownership hold at a tiny shape, an odd expert count and the real
    /// artifact's 2560 x 512 (a CPU model proof only; no GEMM or GPU claim).
    #[test]
    fn slowlink_combine_model_is_generic_over_hidden_width_and_expert_count() {
        for (hidden, n_exp, per_class) in [(8usize, 16usize, 6usize), (64, 40, 5), (2560, 512, 2)] {
            let case = make_case_dims(0xD1A5 ^ hidden as u64, per_class, hidden, n_exp);
            assert_eq!((case.h, case.n_exp), (hidden, n_exp));
            assert_eq!(case.down.len(), case.g * hidden);
            assert_eq!(case.start.len(), n_exp + 1);
            assert_eq!(case.init.len(), case.tokens * hidden);
            let want = canonical(&case);
            let mut rng = Rng(0x77 ^ hidden as u64);
            for cuts in partitions(&case, &mut rng) {
                let got = run(&case, &cuts, true, true);
                assert_eq!(diffs(&got, &want), 0, "{hidden}x{n_exp}, cuts {cuts:?}");
                let ps = panels(&case, &cuts);
                assert_eq!(ps.last().map(|p| (p.2).1), Some(n_exp));
                for slot in 0..case.tokens * K {
                    let id = case.topk[slot];
                    let owners = ps
                        .iter()
                        .filter(|&&(base, rows, (lo, hi))| {
                            slowlink_panel_owns(
                                id,
                                case.inv[slot],
                                case.n_exp,
                                case.g,
                                base,
                                rows,
                                lo,
                                hi,
                            )
                        })
                        .count();
                    let valid = (0..n_exp as i32).contains(&id);
                    assert_eq!(owners, usize::from(valid), "{hidden}x{n_exp} slot {slot}");
                }
            }
        }
    }

    // ───────────── SP hybrid row table: one canonical combine over a permuted layout ─────────────

    /// The SP layer's hybrid view of a combine case: the rows of the experts for which
    /// `streamed(e)` holds are GPU-compacted first (each streamed expert's whole grouped extent,
    /// pads included, in ascending expert order), then the REAL rows of the other experts are
    /// CPU-compacted (pads never uploaded), with the matching inverse.  Valid-id slots that had
    /// no valid grouped row and invalid-id slots get -1.  Returns the hybrid case (`g` = H,
    /// `start` unused, all zero but the last word) and `G_gpu`.
    #[cfg(test)]
    fn hybrid_case(c: &Case, streamed: &dyn Fn(usize) -> bool) -> (Case, usize) {
        let ne = c.n_exp as i32;
        let mut used = vec![false; c.g];
        for slot in 0..c.tokens * K {
            let (id, row) = (c.topk[slot], c.inv[slot]);
            if (0..ne).contains(&id) && row >= 0 && (row as usize) < c.g {
                used[row as usize] = true;
            }
        }
        let mut map = vec![-1i32; c.g];
        let mut old_rows: Vec<usize> = Vec::new();
        for e in 0..c.n_exp {
            if streamed(e) {
                for r in c.start[e]..c.start[e + 1] {
                    map[r] = old_rows.len() as i32;
                    old_rows.push(r);
                }
            }
        }
        let g_gpu = old_rows.len();
        for e in 0..c.n_exp {
            if !streamed(e) {
                for r in c.start[e]..c.start[e + 1] {
                    if used[r] {
                        map[r] = old_rows.len() as i32;
                        old_rows.push(r);
                    }
                }
            }
        }
        let hg = old_rows.len();
        let mut down = vec![0u16; hg * c.h];
        for (i, &r) in old_rows.iter().enumerate() {
            down[i * c.h..(i + 1) * c.h].copy_from_slice(&c.down[r * c.h..(r + 1) * c.h]);
        }
        let inv = (0..c.tokens * K)
            .map(|slot| {
                let (id, row) = (c.topk[slot], c.inv[slot]);
                if (0..ne).contains(&id) && row >= 0 && (row as usize) < c.g {
                    map[row as usize]
                } else {
                    -1
                }
            })
            .collect();
        let mut start = vec![0usize; c.n_exp + 1];
        start[c.n_exp] = hg;
        let hybrid = Case {
            tokens: c.tokens,
            h: c.h,
            n_exp: c.n_exp,
            g: hg,
            start,
            down,
            inv,
            topk: c.topk.clone(),
            w: c.w.clone(),
            init: c.init.clone(),
            class: c.class.clone(),
        };
        (hybrid, g_gpu)
    }

    /// SP's single launch: ONE panel over the whole hybrid table owning `[0, n_exp)`, then the
    /// terminal products (`Gpu::sp_finish_combine`).
    #[cfg(test)]
    fn run_hybrid_single(hy: &Case) -> Vec<f32> {
        let mut acc: Vec<f32> = hy.init.iter().map(|&v| rbf(v)).collect();
        apply_panel(hy, &mut acc, 0, hy.g, (0, hy.n_exp));
        apply_pad(hy, &mut acc);
        acc
    }

    /// Expert placements the SP planner can produce: k = U (all streamed), k = 0 (all CPU),
    /// interleaved, random, and the heavy experts of the fixture on either side.
    #[cfg(test)]
    fn placements(n_exp: usize, rng: &mut Rng) -> Vec<(&'static str, Vec<bool>)> {
        let heavy: Vec<usize> = [7usize, 5, 9, 300, 511].iter().map(|e| e % n_exp).collect();
        let random = |rng: &mut Rng, per_mille: usize| -> Vec<bool> {
            (0..n_exp).map(|_| rng.below(1000) < per_mille).collect()
        };
        vec![
            ("all-streamed", vec![true; n_exp]),
            ("all-cpu", vec![false; n_exp]),
            ("even-streamed", (0..n_exp).map(|e| e % 2 == 0).collect()),
            ("odd-streamed", (0..n_exp).map(|e| e % 2 == 1).collect()),
            (
                "low-half-streamed",
                (0..n_exp).map(|e| e < n_exp / 2).collect(),
            ),
            ("random-50", random(rng, 500)),
            ("random-10", random(rng, 100)),
            ("random-90", random(rng, 900)),
            (
                "heavy-on-cpu",
                (0..n_exp).map(|e| !heavy.contains(&e)).collect(),
            ),
            (
                "heavy-streamed",
                (0..n_exp).map(|e| heavy.contains(&e)).collect(),
            ),
        ]
    }

    #[test]
    fn slowlink_sp_hybrid_single_launch_equals_canonical_and_the_ordered_s1_panels() {
        for (hidden, n_exp, per_class, seeds) in [
            (
                DEFAULT_HIDDEN,
                DEFAULT_EXPERTS,
                6usize,
                &[0x51_0e_1u64, 0xC0B1, 7][..],
            ),
            (8, 16, 6, &[0xD1A5][..]),
            (64, 40, 5, &[0xBEEF][..]),
            (2560, 512, 2, &[0x2560][..]),
        ] {
            for &seed in seeds {
                let case = make_case_dims(seed, per_class, hidden, n_exp);
                let want = canonical(&case);
                // The ordered multi-panel S1 result on the original layout (tile-sized panels).
                let tile_cuts: Vec<usize> = (0..=case.g.div_ceil(16))
                    .map(|i| (i * 16).min(case.g))
                    .collect();
                let ordered = run(&case, &tile_cuts, true, true);
                assert_eq!(diffs(&ordered, &want), 0, "{hidden}x{n_exp} S1 ordered");
                let mut rng = Rng(seed ^ 0x5eed);
                for (name, mask) in placements(n_exp, &mut rng) {
                    let (hy, g_gpu) = hybrid_case(&case, &|e| mask[e]);
                    let ctx = format!("{hidden}x{n_exp} seed {seed:#x} {name}");
                    // GPU rows are the streamed experts' extents, CPU rows come after them.
                    let want_gpu: usize = (0..n_exp)
                        .filter(|&e| mask[e])
                        .map(|e| case.start[e + 1] - case.start[e])
                        .sum();
                    assert_eq!(g_gpu, want_gpu, "{ctx}");
                    assert!(hy.g >= g_gpu, "{ctx}");
                    // The inverse is a bijection of the valid slots onto hybrid rows, and a slot
                    // lands in the GPU part exactly when its expert is streamed.
                    let mut seen = vec![false; hy.g];
                    for slot in 0..case.tokens * K {
                        let (id, row) = (case.topk[slot], hy.inv[slot]);
                        let valid_row = (0..n_exp as i32).contains(&id)
                            && case.inv[slot] >= 0
                            && (case.inv[slot] as usize) < case.g;
                        if !valid_row {
                            assert_eq!(row, -1, "{ctx} slot {slot}");
                            continue;
                        }
                        assert!(row >= 0 && (row as usize) < hy.g, "{ctx} slot {slot}");
                        assert!(!seen[row as usize], "{ctx}: hybrid row {row} named twice");
                        seen[row as usize] = true;
                        assert_eq!(
                            (row as usize) < g_gpu,
                            mask[id as usize],
                            "{ctx} slot {slot}"
                        );
                        // The hybrid row holds the original row's bytes.
                        let old = case.inv[slot] as usize;
                        assert_eq!(
                            hy.down[row as usize * hidden..(row as usize + 1) * hidden],
                            case.down[old * hidden..(old + 1) * hidden],
                            "{ctx} slot {slot}"
                        );
                    }
                    // CPU rows are real rows only: every CPU hybrid row is named by a slot.
                    assert!(
                        seen[g_gpu..].iter().all(|&s| s),
                        "{ctx}: an unreferenced CPU row"
                    );
                    // One launch over the permuted table == canonical == the ordered S1 panels.
                    let got = run_hybrid_single(&hy);
                    assert_eq!(diffs(&got, &want), 0, "{ctx}: vs canonical");
                    assert_eq!(diffs(&got, &ordered), 0, "{ctx}: vs ordered S1");
                    // And the monolithic canonical combine over the hybrid layout agrees.
                    assert_eq!(diffs(&canonical(&hy), &want), 0, "{ctx}: hybrid canonical");
                }
            }
        }
    }

    #[test]
    fn slowlink_sp_hybrid_layout_really_permutes_the_rows() {
        // The equality above must not be the identity layout in disguise.
        let case = make_case(0xABCD, 6);
        let (hy, g_gpu) = hybrid_case(&case, &|e| e % 2 == 0);
        assert!(g_gpu > 0 && g_gpu < hy.g);
        assert_ne!(hy.inv, case.inv);
        assert_ne!(hy.down.len(), case.down.len(), "CPU pads are dropped");
        let moved = (0..case.tokens * K)
            .filter(|&s| case.inv[s] >= 0 && hy.inv[s] >= 0 && case.inv[s] != hy.inv[s])
            .count();
        assert!(moved > case.tokens, "only {moved} slots moved");
        // CPU-all drops every pad row: H = the number of real rows.
        let (all_cpu, g0) = hybrid_case(&case, &|_| false);
        assert_eq!(g0, 0);
        let real = (0..case.tokens * K)
            .filter(|&s| {
                (0..case.n_exp as i32).contains(&case.topk[s])
                    && case.inv[s] >= 0
                    && (case.inv[s] as usize) < case.g
            })
            .count();
        assert_eq!(all_cpu.g, real);
        assert!(all_cpu.g < case.g);
        // All streamed keeps every grouped row at its place.
        let (all_gpu, g_all) = hybrid_case(&case, &|_| true);
        assert_eq!((g_all, all_gpu.g), (case.g, case.g));
        assert_eq!(all_gpu.down, case.down);
    }

    #[test]
    fn slowlink_sp_hybrid_combine_negative_controls() {
        let case = make_case(0xBADD, 8);
        let want = canonical(&case);
        let (hy, g_gpu) = hybrid_case(&case, &|e| e % 2 == 0);
        assert!(g_gpu > 0 && g_gpu < hy.g);
        assert_eq!(diffs(&run_hybrid_single(&hy), &want), 0);
        // (a) The S1 per-panel immediate combine cannot be used on the hybrid table: applying
        // the GPU part and the CPU part as two panels, in either order, breaks the ascending
        // expert order of the tokens that hold both kinds of slots.  The first panel owns the
        // invalid-row slots, the second none.
        let two_panels = |first: (usize, usize), second: (usize, usize)| {
            let mut acc: Vec<f32> = hy.init.iter().map(|&v| rbf(v)).collect();
            apply_panel(&hy, &mut acc, first.0, first.1, (0, hy.n_exp));
            apply_panel(&hy, &mut acc, second.0, second.1, (hy.n_exp, hy.n_exp));
            apply_pad(&hy, &mut acc);
            acc
        };
        let gpu_first = two_panels((0, g_gpu), (g_gpu, hy.g - g_gpu));
        let cpu_first = two_panels((g_gpu, hy.g - g_gpu), (0, g_gpu));
        assert!(
            diffs(&gpu_first, &want) > 0,
            "GPU-then-CPU panels must differ"
        );
        assert!(
            diffs(&cpu_first, &want) > 0,
            "CPU-then-GPU panels must differ"
        );
        // (b) Reordered ids: the combine order is a function of the expert ids alone, so
        // reflecting them (ids kept, rows and weights untouched) changes the result.
        let mut flipped = hybrid_case(&case, &|e| e % 2 == 0).0;
        let ne = flipped.n_exp as i32;
        for id in flipped.topk.iter_mut().filter(|id| (0..ne).contains(*id)) {
            *id = ne - 1 - *id;
        }
        assert!(
            diffs(&run_hybrid_single(&flipped), &want) > 0,
            "reordered ids must differ"
        );
        // (c) A wrong inverse (two valid slots exchange their rows) is visible too.
        let mut swapped = hybrid_case(&case, &|e| e % 2 == 0).0;
        let valid: Vec<usize> = (0..swapped.inv.len())
            .filter(|&s| swapped.inv[s] >= 0 && swapped.class[s / K] == 0)
            .collect();
        let (a, b) = (valid[0], valid[valid.len() / 2]);
        assert_ne!(swapped.inv[a], swapped.inv[b]);
        swapped.inv.swap(a, b);
        assert!(
            diffs(&run_hybrid_single(&swapped), &want) > 0,
            "swapped rows must differ"
        );
    }

    /// Result of [`canonical_combine_exact`].
    #[derive(Clone, Debug, Default)]
    pub struct CombineExactReport {
        pub tokens: usize,
        pub grouped_rows: usize,
        pub partitions: usize,
        pub panel_launches: usize,
        /// Elements of the panelled GPU combine that differ from the CPU canonical model
        /// (subnormal-class tokens excluded, see `subnormal_informational`).
        pub mismatches_vs_cpu: usize,
        /// Elements that differ from the monolithic GPU kernel
        /// (`moe_down_combine_grouped_top10_bf16in`), every class included.
        pub mismatches_vs_monolithic_gpu: usize,
        pub subnormal_informational: usize,
        /// Negative control: the reversed panel order must differ from the monolithic kernel.
        pub reversed_order_diffs: usize,
        pub ok: bool,
    }

    impl CombineExactReport {
        pub fn to_json(&self) -> String {
            format!(
                "{{\"tokens\":{},\"grouped_rows\":{},\"partitions\":{},\"panel_launches\":{},\
                 \"mismatches_vs_cpu\":{},\"mismatches_vs_monolithic_gpu\":{},\
                 \"subnormal_informational\":{},\"reversed_order_diffs\":{},\"ok\":{}}}",
                self.tokens,
                self.grouped_rows,
                self.partitions,
                self.panel_launches,
                self.mismatches_vs_cpu,
                self.mismatches_vs_monolithic_gpu,
                self.subnormal_informational,
                self.reversed_order_diffs,
                self.ok
            )
        }
    }

    /// Real-device check of the production `slowlink_combine_panel` / `slowlink_combine_pad`
    /// kernels on the adversarial fixture (every class: duplicates, invalid ids, invalid
    /// rows with valid ids, overflow, absorption, RNE ties, signed zeros, subnormals) over
    /// many panel partitions, within-expert splits included, against both the CPU canonical
    /// model and the monolithic GPU combine.  Uses only the production wrappers.
    pub fn canonical_combine_exact(
        gpu: &mut crate::dispatch::Gpu,
        seed: u64,
    ) -> hip_bridge::HipResult<CombineExactReport> {
        use crate::dispatch::DType;
        let case = make_case(seed, 6);
        let (h, t, g) = (case.h, case.tokens, case.g);
        let bytes_u16: Vec<u8> = case.down.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes_i32 = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let down_t = gpu.upload_raw(&bytes_u16, &[bytes_u16.len()])?;
        let inv_bytes = bytes_i32(&case.inv);
        let inv_t = gpu.upload_raw(&inv_bytes, &[inv_bytes.len()])?;
        let topk_bytes = bytes_i32(&case.topk);
        let topk_t = gpu.upload_raw(&topk_bytes, &[topk_bytes.len()])?;
        let w_t = gpu.upload_f32(&case.w, &[case.w.len()])?;
        let res_t = gpu.upload_f32(&case.init, &[case.init.len()])?;
        let order_t = gpu.alloc_tensor(&[t * K * 8], DType::Raw)?;
        let acc_t = gpu.alloc_tensor(&[t * h], DType::F32)?;
        let sync = |gpu: &mut crate::dispatch::Gpu| gpu.hip.device_synchronize();
        // The monolithic reference: the shipped kernels, exactly as the incumbent launches them.
        gpu.moe_down_combine_grouped_top10_bf16in(
            &down_t, &inv_t, &topk_t, &w_t, &res_t, &order_t, h, g, t, false, case.n_exp,
        )?;
        sync(gpu)?;
        let mono = gpu.download_f32(&res_t)?;
        let cpu = canonical(&case);
        let mut report = CombineExactReport {
            tokens: t,
            grouped_rows: g,
            ..Default::default()
        };
        let mut rng = Rng(seed ^ 0x9e37_79b9);
        let parts = partitions(&case, &mut rng);
        report.partitions = parts.len();
        let init: Vec<u8> = case
            .init
            .iter()
            .flat_map(|&v| rbf(v).to_le_bytes())
            .collect();
        let mut run = |gpu: &mut crate::dispatch::Gpu,
                       cuts: &[usize],
                       ascending: bool|
         -> hip_bridge::HipResult<(Vec<f32>, usize)> {
            gpu.hip.memcpy_htod(&acc_t.buf, &init)?;
            let ps = panels(&case, cuts);
            let order: Vec<usize> = if ascending {
                (0..ps.len()).collect()
            } else {
                (0..ps.len()).rev().collect()
            };
            for &i in &order {
                let (base, rows, own) = ps[i];
                let y = crate::dispatch::GpuTensor {
                    buf: down_t.buf.byte_view(base * h * 2, rows * h * 2),
                    shape: vec![rows * h],
                    dtype: DType::BF16,
                };
                gpu.slowlink_combine_panel(
                    &y, &inv_t, &topk_t, &w_t, &acc_t, h, t, case.n_exp, g, base, rows, own.0,
                    own.1,
                )?;
            }
            gpu.slowlink_combine_pad(&topk_t, &acc_t, h, t, case.n_exp)?;
            gpu.hip.device_synchronize()?;
            Ok((gpu.download_f32(&acc_t)?, order.len()))
        };
        for cuts in &parts {
            let (got, launches) = run(gpu, cuts, true)?;
            report.panel_launches += launches;
            for (i, (a, b)) in got.iter().zip(&mono).enumerate() {
                if !same(*a, *b) {
                    report.mismatches_vs_monolithic_gpu += 1;
                }
                let _ = i;
            }
            for (i, (a, b)) in got.iter().zip(&cpu).enumerate() {
                if !same(*a, *b) {
                    if case.class[i / h] == 11 {
                        report.subnormal_informational += 1;
                    } else {
                        report.mismatches_vs_cpu += 1;
                    }
                }
            }
        }
        let tile_cuts: Vec<usize> = (0..=g.div_ceil(16)).map(|i| (i * 16).min(g)).collect();
        let (reversed, _) = run(gpu, &tile_cuts, false)?;
        report.reversed_order_diffs = reversed
            .iter()
            .zip(&mono)
            .filter(|(a, b)| !same(**a, **b))
            .count();
        for tensor in [down_t, inv_t, topk_t, w_t, res_t, order_t, acc_t] {
            let _ = gpu.free_tensor(tensor);
        }
        report.ok = report.mismatches_vs_cpu == 0
            && report.mismatches_vs_monolithic_gpu == 0
            && report.reversed_order_diffs > 0
            && report.panel_launches > 0;
        Ok(report)
    }
}
