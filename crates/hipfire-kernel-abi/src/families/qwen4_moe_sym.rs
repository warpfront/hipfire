// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 grouped symmetric-IU4 MoE GEMMs over MQ4 experts (`HIPFIRE_QWEN4_MOE_SYM_IU4`
//! route): the SwiGLU gate/up GEMM and the down GEMM, as launched by
//! `rdna_compute::Gpu::qwen4_moe_sym_gemm` / `qwen4_moe_sym_nt1_gemm`.
//!
//! One 64-byte explicit kernarg image for every export (no hidden words in the PM
//! objects; `kernarg_segment_size` 64, align 8):
//!
//! | off | field | kind |
//! |---|---|---|
//! | 0 | `expert_weight_ptrs` | pointer, `n_experts` x u64 |
//! | 8 | `expert_tile_ids` | pointer, one i32 per 16-slot tile |
//! | 16 | `sorted_slot_index` | pointer, one i32 per grouped row |
//! | 24 | `Xq` | pointer, 72-byte `block_i4_128` records, epoch-major |
//! | 32 | `Y_grouped` | pointer, BF16 |
//! | 40 | `M` | i32 |
//! | 44 | `K` | i32 |
//! | 48 | `x_row_div` | i32 |
//! | 52 | `m_total` | i32 (grouped rows, a multiple of 16) |
//! | 56 | `x_src_rows` | i32 |
//! | 60 | `n_experts` | i32 (`moe_expert_table_len`: pointer-table bytes / 8) |
//!
//! Launch: grid `(M / (64 * rr), m_total / 16, 1)` (the callers pass
//! `tiles_y = grouped_rows.div_ceil(16)`, which is `m_total / 16` for every launch
//! the extent check admits, so no separate `tiles_y` launch param exists), block
//! 64 (NT1 gate/up, two waves) or 128, dynamic LDS 0. The kernels run
//! one tile per grid-y index; with NT > 1 a leader CTA folds up to NT consecutive
//! tiles of the same expert run and the others exit, so NT does not change the grid.
//!
//! Sources of truth (all in this repository):
//! * launcher: `crates/rdna-compute/src/gemm.rs` `qwen4_moe_sym_gemm`,
//!   `qwen4_moe_sym_nt1_gemm` (the same 64-byte ABI, grid `(m/64, m_total/16)`,
//!   block `QWEN4_MOE_SYM_NT1_BLOCK = [64, 128]`), `moe_sym_iu4_extents` (domain),
//!   `moe_expert_table_len`, `pm_check_moe_down_extent`, `qwen4_moe_down_gp_entry`,
//!   the `MOE_SYM_IU4_KERNELS` descriptor table and the public callers
//!   `gemm_qwen4_moe_gate_up_silu_iu4_sym` / `gemm_qwen4_moe_down_iu4_sym`;
//! * generator: `crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs` (`Spec::kernargs`,
//!   `threads`, `lds_bytes`, `KERNARG_BYTES`), `qwen4_moe_down_gp.rs`,
//!   `qwen4_moe_sym_gfx1100/mod.rs`;
//! * objects: the committed `kernels/*.hxaco` bundles pinned below (metadata read
//!   with `clang-offload-bundler --unbundle` + `llvm-readelf --notes`) and the hipcc
//!   twin `kernels/src/qwen4_moe_iu4_sym.gfx1151.hip`.
//!
//! Layouts that are NOT in this module: the gfx1151 MQ3-I4 (W3) NT4/NT1 modules and their
//! launcher (`qwen4_moe_sym_w3_nt4_gfx1151_gemm`), the `_acc` qualification twins and the
//! hipcc 16-slot anchors, which the serving launchers never select.
//!
//! `qwen4_moe_sym_nt1_gemm` is shared by three callers, each with its own contract below:
//! the MQ4 descriptor route (gfx1100 NT1: `*_GATE_UP` / `*_DOWN`), the sealed-slab E8P-I4
//! K2 route (`*_E8P`, gfx1100/gfx1151/gfx1201) and the MQ3-I4 gfx1201 NT1 route (`*_W3`).
//! The sealed-slab MQ4 K2 path (`SLAB_K2_GFX1100_MQ4`) is launched through the public MQ4
//! launchers (`qt44_qt53_prefill.rs`), i.e. the same gfx1100 entries and formats as the
//! aggregate route; `qwen4_moe_sym_slab_gemm` itself is only called with codec 3 (E8P).
//!
//! Admission stays in the launcher where it is not a function of the launch params:
//! which entry is selected (NT width, residency, `HIPFIRE_QWEN4_MOE_SYM_PM`, the
//! measured-shape stream8 allowlist `qwen4_moe_stream8_selected`, the wide-tile flag,
//! the exact-arch checks), and the host-side stale-handle check of `Int4MmqPrepared`.

use crate::{
    Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
    HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
};

// ---------------------------------------------------------------------------
// Numeric contracts
// ---------------------------------------------------------------------------

/// SwiGLU gate/up GEMM: per output and ascending K128 epoch `h`, `C_h` is the exact
/// int32 IU4 WMMA chain (A nibbles rebiased with XOR 0x88888888, signed A and X) and
/// `sum = fma(RN(d_h * sc_h), float(C_h), sum)` from `+0`; gate and up sums are each
/// rounded to BF16 (RNE, non-finite passed through) and the store is
/// `rt(silu(rt(g)) * rt(u))` with the hipcc `g / (1 + expf(-g)) * u` DAG; padding
/// slots store `+0`. Shared by every tile width, residency, arch and the hipcc twin:
/// the sources state them bitwise identical (`qwen4_moe_sym.rs` module doc "Numerics";
/// `qwen4_moe_sym_gfx1100/mod.rs` "Numerics: Exactly fold"; gemm.rs "Every entry is
/// bitwise the module's 16-slot entries"; hipcc header "byte-identical hipcc entry").
pub const NUMERIC_GATE_UP_SWIGLU: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_mq4.gate_up_swiglu_bf16_rne.v1");

/// Down GEMM: the same exact-int32 chain and `fma(RN(d_h * sc_h), float(C_h), sum)`
/// fold, then one RNE BF16 store of `sum` per output (no residual, no combine);
/// padding slots store `+0`. The grouped-position twin (`..._gp_...`) shares this id:
/// `qwen4_moe_down_gp.rs` states that "the record bytes, every fold, every rounding
/// and every store are those of the incumbent entries" (byte-identical outputs over
/// the same activations); only the activation plane's addressing differs, which is
/// the `Xq` operand format.
pub const NUMERIC_DOWN: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_mq4.down_bf16_rne.v1");

/// E8P-I4 SwiGLU gate/up: the exact int32 IU4 WMMA chain over the LUT-decoded signed
/// nibbles `o = (v - 1)/2`, so an epoch is `sum(v q) = 2C + s` with the activation sum `s`
/// (i32 `sum q` of the A4 record), folded as `sum = fma(RN(d_act * f32(d)), (2C + s)/4,
/// sum)` (exact `* 0.25`) in ascending K128 order from `+0`; then the MQ4 SwiGLU epilogue
/// (BF16 round trip of gate and up, `silu` DAG, RNE store, padding `+0`).
/// `qwen4_moe_sym_e8p.rs` "Unpack, `2C + s` fold" and `qwen4_moe_sym_gfx1100/mod.rs`
/// "Numerics". Not byte-identical to the MQ4 recipe, hence its own id. The gfx1100,
/// gfx1151 and gfx1201 modules are stated to differ only by arch-forced deltas
/// (`QWEN4_MOE_IU4_SYM_E8P_PM_GFX1151` doc).
pub const NUMERIC_GATE_UP_SWIGLU_E8P: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_e8p.gate_up_swiglu_bf16_rne.v1");

/// E8P-I4 down: the `2C + s` fold above, one RNE BF16 store of `sum` per output.
pub const NUMERIC_DOWN_E8P: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_e8p.down_bf16_rne.v1");

/// MQ3-I4 SwiGLU gate/up (gfx1201 NT1): weight `d * (2c - 7)` from 3-bit codes ((2c - 7) & 15
/// as the signed nibble), then the MQ4 entries' own exact int32 chain, `sum =
/// fma(RN(d_act * d_w), float(dot), sum)` and SwiGLU BF16 epilogue
/// (`qwen4_moe_sym_w3.rs` module doc). The weight decode differs from MQ4, so the id does.
pub const NUMERIC_GATE_UP_SWIGLU_W3: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_mq3.gate_up_swiglu_bf16_rne.v1");

/// MQ3-I4 down (gfx1201 NT1): the W3 weight decode and fold above, one RNE BF16 store.
pub const NUMERIC_DOWN_W3: NumericContractId =
    NumericContractId("qwen4_moe_sym.iu4_a4_mq3.down_bf16_rne.v1");

// ---------------------------------------------------------------------------
// Operand formats
// ---------------------------------------------------------------------------

/// Gate/up expert weight pointer table: `n_experts` little-endian u64 device
/// addresses, `table[e]` = base of expert `e`'s symmetric MQ4 QT44 slab (gate rows
/// `0..M/2` then up rows `M/2..M`, each row `K/256` groups of 136 bytes
/// `[f16 sc0, zp0, sc1, zp1][128 B nibbles]`, every header obeying `zp == -8 * sc`).
/// The slab extents are the loader's contract (header-checked by
/// `qwen4_moe_sym_check_*`), not checked by this crate.
pub const FORMAT_EXPERT_PTRS_GATE_UP_QT44: FormatId =
    FormatId("qwen4_moe.expert_ptr_table.u64.mq4_sym_qt44_gate_up");

/// Down expert weight pointer table: as above over QT53 slabs (`M` rows, each row
/// `K/128` groups of 68 bytes `[f16 sc, zp][64 B nibbles]`).
pub const FORMAT_EXPERT_PTRS_DOWN_QT53: FormatId =
    FormatId("qwen4_moe.expert_ptr_table.u64.mq4_sym_qt53_down");

/// Expert id of each 16-slot grouped tile: `m_total / 16` little-endian i32. An id
/// outside `[0, n_experts)` (the -1 sentinel or a corrupt id) is a dead tile: the
/// kernel exits before any load or store. Ids of one expert are contiguous runs.
pub const FORMAT_EXPERT_TILE_IDS: FormatId = FormatId("qwen4_moe.expert_tile_ids.i32.per_tile16");

/// Grouped-row to flat-slot map: `m_total` little-endian i32, row `16 * tile + lane`
/// holds its routed slot, or a negative value for a padding row (its activation loads
/// read 0 and its outputs store +0).
pub const FORMAT_SORTED_SLOT_INDEX: FormatId =
    FormatId("qwen4_moe.sorted_slot_index.i32.minus1_padding");

/// Gate/up activation sidecar (`mq_rotate_x_i4`: FWHT256 rotation + `block_i4_128`):
/// `K/128` epoch planes of `x_src_rows` token-row records, record index
/// `h * x_src_rows + slot / x_row_div`, 72 bytes each (`d` f32, `s` i32, 64 signed
/// nibble bytes).
pub const FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS: FormatId =
    FormatId("qwen4_moe.xq.block_i4_128.fwht256.epoch_major.token_rows");

/// Down activation sidecar, flat-slot layout (`qwen4_moe_rotate128_i4`: FWHT128
/// rotation + `block_i4_128`): `K/128` epoch planes of `x_src_rows` flat-slot
/// records, record index `h * x_src_rows + slot` (`x_row_div == 1`), 72 bytes each.
pub const FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS: FormatId =
    FormatId("qwen4_moe.xq.block_i4_128.rot128.epoch_major.flat_slots");

/// Down activation plane in grouped-position order (`moe_rotate_i4_gp_pm_gfx1201`):
/// `K/128` epoch planes of `m_total` records, record index
/// `h * m_total + 16 * tile + lane`, 72 bytes each (`d` f32, `s` i32, 64 `qs`). The
/// `..._gp` entries ignore `x_src_rows` (the plane stride is `m_total * 72`).
pub const FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION: FormatId =
    FormatId("qwen4_moe.xq.block_i4_128.rot128.epoch_major.grouped_position");

/// Gate/up output: BF16 `[m_total, M/2]` row-major, `rt(silu(rt(g)) * rt(u))`,
/// padding rows `+0`, dead tiles untouched.
pub const FORMAT_Y_GATE_UP_SWIGLU_BF16: FormatId =
    FormatId("qwen4_moe.y_grouped.bf16.swiglu.rows_m_total.cols_m_half");

/// Down output: BF16 `[m_total, M]` row-major, padding rows `+0`, dead tiles
/// untouched.
pub const FORMAT_Y_DOWN_BF16: FormatId = FormatId("qwen4_moe.y_grouped.bf16.rows_m_total.cols_m");

/// Expert pointer table of the sealed-slab E8P-I4 NT1 GEMMs (gate/up and down): `n_experts`
/// u64 addresses, `table[e]` = base of expert `e`'s planar K-E0 tensor `[M, K]`: `M*K/8`
/// little-endian u16 words (`idx | sign7 << 8 | shift << 15`) then `M*K/128` f16 scales
/// (`qwen4_moe_sym_e8p.rs` "Wire layout"). Gate/up and down share the layout; the dims
/// (`[M, K]`) are the launch params.
pub const FORMAT_EXPERT_PTRS_E8P_PLANAR: FormatId =
    FormatId("qwen4_moe.expert_ptr_table.u64.e8p_i4_planar");

/// Expert pointer table of the MQ3-I4 (W3) NT1 GEMMs: `table[e]` = base of expert `e`'s
/// planar qt131/qt132 tensor `[M, K]`: `[codes: M*K*3/8 B][scales: M*(K/128) f16]`, code `j`
/// of a row at bits `3j..3j+3` of the row's little-endian bitstream, weight `d * (2c - 7)`
/// (`qwen4_moe_sym_w3.rs`).
pub const FORMAT_EXPERT_PTRS_W3_PLANAR: FormatId =
    FormatId("qwen4_moe.expert_ptr_table.u64.mq3_i4_planar");

/// Expert pointer table of the drop-zero planar MQ4 GEMMs (DESIGN F1): `table[e]` = base of expert
/// `e`'s planar `MQ4G256V2P`/`MQ4G128V2P` tensor `[M, K]`: `[codes: M*K/2 B][scales: M*(K/128)
/// f16]` (`hipfire_runtime::mq4_planar`), weight `sc * (q - 8)`.
pub const FORMAT_EXPERT_PTRS_MQ4_PLANAR: FormatId =
    FormatId("qwen4_moe.expert_ptr_table.u64.mq4_xts_planar");

// ---------------------------------------------------------------------------
// Params and expressions
// ---------------------------------------------------------------------------

const P_M: Expr = Expr::Param("M");
const P_K: Expr = Expr::Param("K");
const P_X_ROW_DIV: Expr = Expr::Param("x_row_div");
const P_M_TOTAL: Expr = Expr::Param("m_total");
const P_X_SRC_ROWS: Expr = Expr::Param("x_src_rows");
const P_N_EXPERTS: Expr = Expr::Param("n_experts");

/// `x_row_div * x_src_rows`, the slot count `moe_sym_iu4_extents` bounds.
const SLOTS: Expr = Expr::Mul(&[P_X_ROW_DIV, P_X_SRC_ROWS]);
/// `m_total * M * 2`: the BF16 bytes of the down output `[m_total, M]`.
const DOWN_Y_BYTES: Expr = Expr::Mul(&[P_M_TOTAL, P_M, Expr::Const(2)]);

/// `K / 128` K128 epochs (every admitted K is a multiple of 128).
const EPOCHS: Expr = Expr::DivExact(&P_K, 128);

/// `expert_weight_ptrs`: entries `0..n_experts` are indexed (`e < n_experts` is
/// checked before the load); `n_experts` is `table bytes / 8`.
const EXT_TABLE: Expr = Expr::Mul(&[P_N_EXPERTS, Expr::Const(8)]);
/// `expert_tile_ids`: tile `t < ceil(m_total / 16)` (`tile * 16 >= m_total` exits;
/// the NT leader walk-back reads `t < tile`, the forward look-ahead is bounded by
/// `m_total >> 4`).
const EXT_TILES: Expr = Expr::Mul(&[Expr::DivCeil(&P_M_TOTAL, 16), Expr::Const(4)]);
/// `sorted_slot_index`: `m_total` i32 (launcher `require_span(sorted, [grouped_rows, 4])`).
const EXT_SORTED: Expr = Expr::Mul(&[P_M_TOTAL, Expr::Const(4)]);
/// Flat sidecar: `K/128` planes of `x_src_rows` 72-byte records (scratch
/// `int4_mmq_blocks_bytes`: `k / 128 * rows * 72`).
const EXT_XQ_FLAT: Expr = Expr::Mul(&[EPOCHS, P_X_SRC_ROWS, Expr::Const(72)]);
/// Grouped-position plane: `K/128` planes of `m_total` records
/// (`reserve_qwen4_moe_down_i4_plane`: `k / 128 * plane_rows * 72`).
const EXT_XQ_GP: Expr = Expr::Mul(&[EPOCHS, P_M_TOTAL, Expr::Const(72)]);
/// Gate/up Y: `[m_total, M/2]` BF16 (launcher `require_span(y, [grouped_rows, m / 2, 2])`).
const EXT_Y_GATE_UP: Expr = Expr::Mul(&[P_M_TOTAL, Expr::DivExact(&P_M, 2), Expr::Const(2)]);
/// Down Y: `[m_total, M]` BF16 (launcher `require_span(y, [grouped_rows, m, 2])`).
const EXT_Y_DOWN: Expr = DOWN_Y_BYTES;

/// Grid y: one CTA row per 16-slot tile. The callers pass
/// `grouped_rows.div_ceil(16)` (`qwen4_moe_sym_gemm`) or `dims[3] / 16`
/// (`qwen4_moe_sym_nt1_gemm`); both are `m_total / 16` once `m_total % 16 == 0`.
const GRID_Y: Expr = Expr::DivExact(&P_M_TOTAL, 16);

/// Grid x = `M / (64 * rr)` (launcher: `dims[0] / (64 * rr)`).
const GRID_X_RR1: Expr = Expr::DivExact(&P_M, 64);
const GRID_X_RR2: Expr = Expr::DivExact(&P_M, 128);
const GRID_X_RR4: Expr = Expr::DivExact(&P_M, 256);

/// `moe_sym_iu4_extents` builder bound `QWEN4_MOE_SYM_PM_MAX_SLOTS` (= `MAX_SLOTS` of the
/// generator): the f32-reciprocal slot division is exact only below 2^22 slots.
const MAX_SLOTS: u64 = 1 << 22;
/// `PM_MOE_DOWN_EXTENT_LIMIT_BYTES`: builder down entries index Y with a 32-bit byte offset.
const DOWN_Y_LIMIT_BYTES: u64 = u32::MAX as u64;

const PARAMS: [ParamV1; 6] = [
    ParamV1 {
        name: "M",
        ty: ScalarTy::I32,
        doc: "output rows: gate+up rows for gate/up (Y has M/2 columns), output channels for down",
    },
    ParamV1 {
        name: "K",
        ty: ScalarTy::I32,
        doc: "reduction width (a multiple of 256 for gate/up, of 128 for down)",
    },
    ParamV1 {
        name: "x_row_div",
        ty: ScalarTy::I32,
        doc: "activation record of slot s is s / x_row_div: top-k for gate/up, 1 for down",
    },
    ParamV1 {
        name: "m_total",
        ty: ScalarTy::I32,
        doc: "grouped rows (tile-padded), a multiple of 16; grid y = m_total / 16",
    },
    ParamV1 {
        name: "x_src_rows",
        ty: ScalarTy::I32,
        doc: "activation records per K128 epoch plane: token rows for gate/up, flat slots for down",
    },
    ParamV1 {
        name: "n_experts",
        ty: ScalarTy::I32,
        doc: "expert pointer table entries (moe_expert_table_len = table bytes / 8, saturating at i32::MAX)",
    },
];

const fn ptr_field(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 8,
        kind: FieldKind::Pointer { operand: name },
    }
}

const fn i32_field(name: &'static str, offset: u32) -> FieldV1 {
    FieldV1 {
        name,
        offset,
        width: 4,
        kind: FieldKind::Scalar {
            ty: ScalarTy::I32,
            value: Expr::Param(name),
        },
    }
}

/// The metadata layout of every PM object (offsets/sizes read from the
/// `amdhsa.kernels` `.args` of every entry listed below).
const FIELDS: [FieldV1; 11] = [
    ptr_field("expert_weight_ptrs", 0),
    ptr_field("expert_tile_ids", 8),
    ptr_field("sorted_slot_index", 16),
    ptr_field("Xq", 24),
    ptr_field("Y_grouped", 32),
    i32_field("M", 40),
    i32_field("K", 44),
    i32_field("x_row_div", 48),
    i32_field("m_total", 52),
    i32_field("x_src_rows", 56),
    i32_field("n_experts", 60),
];

const fn op_table(format: FormatId) -> OperandV1 {
    OperandV1 {
        name: "expert_weight_ptrs",
        format,
        access: Access::ReadOnly,
        align: 8,
        extent: Some(EXT_TABLE),
        nullable: false,
        note:
            "u64 per expert; entries point at symmetric MQ4 expert slabs (QT44 gate/up, QT53 down) \
               whose extent is the loader's contract (header-checked at load), not checked here; \
               the kernel indexes only e < n_experts",
    }
}

const fn op_table_planar(format: FormatId, note: &'static str) -> OperandV1 {
    OperandV1 {
        name: "expert_weight_ptrs",
        format,
        access: Access::ReadOnly,
        align: 8,
        extent: Some(EXT_TABLE),
        nullable: false,
        note,
    }
}

const NOTE_TABLE_E8P: &str = "u64 per expert; entries point at planar E8P-I4 tensors \
     [M*K/8 u16 words][M*K/128 f16 scales] (gate rows then up rows for gate/up) whose extent \
     is the loader's contract, not checked here; the kernel indexes only e < n_experts";
const NOTE_TABLE_W3: &str = "u64 per expert; entries point at planar MQ3-I4 tensors \
     [codes: M*K*3/8 B][scales: M*(K/128) f16] (gate rows then up rows for gate/up) whose \
     extent is the loader's contract, not checked here; the kernel indexes only e < n_experts";
const NOTE_TABLE_MQ4P: &str = "u64 per expert; entries point at drop-zero planar MQ4 tensors \
     [codes: M*K/2 B][scales: M*(K/128) f16] (gate rows then up rows for gate/up; the scale plane \
     starts at byte M*K/2) whose extent is the loader's contract, not checked here; the kernel \
     indexes only e < n_experts";

const OP_TILE_IDS: OperandV1 = OperandV1 {
    name: "expert_tile_ids",
    format: FORMAT_EXPERT_TILE_IDS,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(EXT_TILES),
    nullable: false,
    note: "an id outside [0, n_experts) is a dead tile (no load, no store)",
};

const OP_SORTED: OperandV1 = OperandV1 {
    name: "sorted_slot_index",
    format: FORMAT_SORTED_SLOT_INDEX,
    access: Access::ReadOnly,
    align: 4,
    extent: Some(EXT_SORTED),
    nullable: false,
    note: "negative = padding row; the launcher checks the buffer span, not the entries",
};

const fn op_xq(format: FormatId, extent: Expr, note: &'static str) -> OperandV1 {
    OperandV1 {
        name: "Xq",
        format,
        access: Access::ReadOnly,
        align: 8,
        extent: Some(extent),
        nullable: false,
        note,
    }
}

const fn op_y(format: FormatId, extent: Expr, note: &'static str) -> OperandV1 {
    OperandV1 {
        name: "Y_grouped",
        format,
        access: Access::WriteOnly,
        align: 16,
        extent: Some(extent),
        nullable: false,
        note,
    }
}

const NOTE_XQ_FLAT: &str = "the extent is the epoch-plane span; which record a live slot reads \
     (slot / x_row_div < x_src_rows) depends on the sorted_slot_index content, a producer \
     contract the launcher does not check (it bounds x_row_div * x_src_rows only); the builder \
     entries read through bounded buffer descriptors, the hipcc twin does not";
const NOTE_Y_GATE_UP: &str =
    "padding rows store +0, dead tiles are untouched; 16-byte vector stores";
const NOTE_Y_DOWN: &str = "padding rows store +0, dead tiles are untouched; 16-byte vector stores";

const OPS_GATE_UP: [OperandV1; 5] = [
    op_table(FORMAT_EXPERT_PTRS_GATE_UP_QT44),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
        EXT_XQ_FLAT,
        NOTE_XQ_FLAT,
    ),
    op_y(FORMAT_Y_GATE_UP_SWIGLU_BF16, EXT_Y_GATE_UP, NOTE_Y_GATE_UP),
];

const OPS_DOWN: [OperandV1; 5] = [
    op_table(FORMAT_EXPERT_PTRS_DOWN_QT53),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS, EXT_XQ_FLAT, NOTE_XQ_FLAT),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];

const OPS_DOWN_GP: [OperandV1; 5] = [
    op_table(FORMAT_EXPERT_PTRS_DOWN_QT53),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
        EXT_XQ_GP,
        "plane stride is m_total records per epoch (x_src_rows is ignored by the kernel); \
         sorted_slot_index still marks padding rows and dead tiles",
    ),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];

const OPS_GATE_UP_E8P: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_E8P_PLANAR, NOTE_TABLE_E8P),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
        EXT_XQ_FLAT,
        NOTE_XQ_FLAT,
    ),
    op_y(FORMAT_Y_GATE_UP_SWIGLU_BF16, EXT_Y_GATE_UP, NOTE_Y_GATE_UP),
];
const OPS_DOWN_E8P: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_E8P_PLANAR, NOTE_TABLE_E8P),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS, EXT_XQ_FLAT, NOTE_XQ_FLAT),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];
const OPS_GATE_UP_W3: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_W3_PLANAR, NOTE_TABLE_W3),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
        EXT_XQ_FLAT,
        NOTE_XQ_FLAT,
    ),
    op_y(FORMAT_Y_GATE_UP_SWIGLU_BF16, EXT_Y_GATE_UP, NOTE_Y_GATE_UP),
];
const OPS_DOWN_W3: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_W3_PLANAR, NOTE_TABLE_W3),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS, EXT_XQ_FLAT, NOTE_XQ_FLAT),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];
const OPS_GATE_UP_P: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_MQ4_PLANAR, NOTE_TABLE_MQ4P),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
        EXT_XQ_FLAT,
        NOTE_XQ_FLAT,
    ),
    op_y(FORMAT_Y_GATE_UP_SWIGLU_BF16, EXT_Y_GATE_UP, NOTE_Y_GATE_UP),
];
const OPS_DOWN_P: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_MQ4_PLANAR, NOTE_TABLE_MQ4P),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS, EXT_XQ_FLAT, NOTE_XQ_FLAT),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];
const OPS_DOWN_GP_P: [OperandV1; 5] = [
    op_table_planar(FORMAT_EXPERT_PTRS_MQ4_PLANAR, NOTE_TABLE_MQ4P),
    OP_TILE_IDS,
    OP_SORTED,
    op_xq(
        FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
        EXT_XQ_GP,
        "plane stride is m_total records per epoch (x_src_rows is ignored by the kernel); \
         sorted_slot_index still marks padding rows and dead tiles",
    ),
    op_y(FORMAT_Y_DOWN_BF16, EXT_Y_DOWN, NOTE_Y_DOWN),
];

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

const RULE_M_POSITIVE: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::Positive(P_M),
    why: "the public launchers reject m == 0 (an empty grid)",
};
const RULE_M_64: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_M, 64),
    why: "a CTA covers 64 output rows (32 gate + 32 up for gate/up); moe_sym_iu4_extents: m % 64 == 0",
};
const RULE_M_128: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_M, 128),
    why: "row repeat 2: a CTA covers two contiguous 64-row blocks; moe_sym_iu4_extents: m % (64 * rr) == 0 (HIPFIRE_QWEN4_MOE_SYM_DOWN_RR=2)",
};
const RULE_M_256: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_M, 256),
    why: "row repeat 4: a CTA covers four contiguous 64-row blocks; moe_sym_iu4_extents: m % (64 * rr) == 0 (HIPFIRE_QWEN4_MOE_SYM_DOWN_RR=4)",
};
const RULE_K_256: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_K, 256),
    why: "gate/up weights are QT44 groups of 256 (two K128 epochs per 136-byte group); gemm_qwen4_moe_gate_up_silu_iu4_sym: k % 256 == 0",
};
const RULE_K_128: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_K, 128),
    why:
        "down weights are QT53 groups of one K128 epoch; gemm_qwen4_moe_down_iu4_sym: k % 128 == 0",
};
const RULE_X_ROW_DIV: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::Positive(P_X_ROW_DIV),
    why: "the slot is divided by x_row_div; moe_sym_iu4_extents: x_row_div > 0",
};
const RULE_M_TOTAL_16: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_M_TOTAL, 16),
    why: "grouped rows are whole 16-slot tiles; moe_sym_iu4_extents: grouped_rows % 16 == 0",
};
const RULE_SLOTS: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::AtMost(SLOTS, MAX_SLOTS),
    why: "builder entries divide slot / x_row_div through an f32 reciprocal that is exact only below 2^22 slots; moe_sym_iu4_extents slot_bound",
};
const RULE_DOWN_Y_U32: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::AtMost(DOWN_Y_BYTES, DOWN_Y_LIMIT_BYTES),
    why: "builder down entries index Y with a 32-bit byte offset; pm_check_moe_down_extent: m_total * M * 2 <= u32::MAX",
};

/// Builder gate/up entries (NT1 anchors, NT4, NT8, stream8). The slot bound applies to
/// every builder entry including the NT1 launcher (`Some(QWEN4_MOE_SYM_PM_MAX_SLOTS)`).
const DOMAIN_GATE_UP: [DomainRuleV1; 6] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_256,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
];
const DOMAIN_DOWN: [DomainRuleV1; 7] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_128,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
    RULE_DOWN_Y_U32,
];
const DOMAIN_DOWN_RR2: [DomainRuleV1; 7] = [
    RULE_M_POSITIVE,
    RULE_M_128,
    RULE_K_128,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
    RULE_DOWN_Y_U32,
];
const DOMAIN_DOWN_RR4: [DomainRuleV1; 7] = [
    RULE_M_POSITIVE,
    RULE_M_256,
    RULE_K_128,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
    RULE_DOWN_Y_U32,
];
/// The incumbent hipcc entry (`HIPFIRE_QWEN4_MOE_SYM_PM=0`) is not bound by the slot or
/// the Y-byte guard: `moe_sym_iu4_extents(..., slot_bound = pm.then_some(..))` passes
/// `None` for it, so those rules are absent here (never stricter than the launcher).
const DOMAIN_GATE_UP_HIPCC: [DomainRuleV1; 5] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_256,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
];
const DOMAIN_DOWN_HIPCC: [DomainRuleV1; 5] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_128,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
];

// Planar-codec NT1 routes (sealed-slab E8P-I4 and gfx1201 MQ3-I4). `qwen4_moe_sym_nt1_gemm`
// always applies `moe_sym_iu4_extents(func, down, 1, Some(QWEN4_MOE_SYM_PM_MAX_SLOTS), dims)`,
// so the slot bound and (down) the Y-byte bound apply; the K rules are those of the public
// callers `gemm_qwen4_moe_{e8p,w3}_gate_up_silu_iu4_sym` (k % 256) and
// `gemm_qwen4_moe_{e8p,w3}_down_iu4_sym` (k % 128), the same as the MQ4 callers.
const RULE_K_256_PLANAR: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_K, 256),
    why: "gate/up K is a multiple of 256 (the shared FWHT256 activation sidecar); the E8P/W3 gate/up launchers: k % 256 == 0",
};
const RULE_K_128_PLANAR: DomainRuleV1 = DomainRuleV1 {
    rule: Rule::MultipleOf(P_K, 128),
    why: "down K is a whole number of K128 epochs; the E8P/W3 down launchers: k % 128 == 0",
};
const DOMAIN_GATE_UP_PLANAR: [DomainRuleV1; 6] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_256_PLANAR,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
];
const DOMAIN_DOWN_PLANAR: [DomainRuleV1; 7] = [
    RULE_M_POSITIVE,
    RULE_M_64,
    RULE_K_128_PLANAR,
    RULE_X_ROW_DIV,
    RULE_M_TOTAL_16,
    RULE_SLOTS,
    RULE_DOWN_Y_U32,
];

// ---------------------------------------------------------------------------
// Objects and exports
// ---------------------------------------------------------------------------

const OBJ_PM_GFX1151: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_pm_gfx1151",
    artifact: "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
    sha256: "20d6a98d700d2eb47dc5c5f8b74ee65835912c317898ce42c345cf0cddb78c0a",
};
const OBJ_PM_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_pm_gfx1201",
    artifact: "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
    sha256: "44177c21de25bec8502f63f318e682dc4fa1948420723c75dec14fa63cfed723",
};
const OBJ_PM_GFX1100: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_pm_gfx1100",
    artifact: "kernels/qwen4_moe_iu4_sym_pm_gfx1100.hxaco",
    sha256: "831db6f9f1b45e22cc4e3b99d1f67d510b305230eb32ef0a09e0c91088f24bad",
};
const OBJ_STREAM8_GFX1151: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_gate_up_stream8_pm_gfx1151",
    artifact: "kernels/qwen4_moe_gate_up_stream8_pm_gfx1151.hxaco",
    sha256: "7762f78f3988f90a45a75d3da99e2a9fe98e29c48e698f1a22ee4438fa7a7a8a",
};
const OBJ_DOWN_GP_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_down_gp_pm_gfx1201",
    artifact: "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
    sha256: "e136e40e9df122b938a7942f28d1c55be7af9359ce7a800b317aadb2e733b972",
};
/// `HIPFIRE_QWEN4_MOE_SYM_PM=0` control: hipcc-compiled at load from
/// `kernels::QWEN4_MOE_IU4_SYM_GFX1151_SRC` (this file preceded by the
/// `block_i4_128_quant.hip` prelude).
const OBJ_HIPCC_GFX1151: ObjectRefV1 = ObjectRefV1::HipccSource {
    module: "qwen4_moe_iu4_sym_gfx1151",
    source: "kernels/src/qwen4_moe_iu4_sym.gfx1151.hip",
};
const OBJ_E8P_GFX1100: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_e8p_pm_gfx1100",
    artifact: "kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1100.hxaco",
    sha256: "1dfc581125a655fc0d15969fe63e18e5a145a427e59b7935109ce0dc1501a010",
};
const OBJ_E8P_GFX1151: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_e8p_pm_gfx1151",
    artifact: "kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1151.hxaco",
    sha256: "20f3b17c86887ab5228740e6a1cc3b8584d64e6f0fd99eb811d8b6876b84abfa",
};
const OBJ_E8P_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_e8p_pm_gfx1201",
    artifact: "kernels/qwen4_moe_iu4_sym_e8p_pm_gfx1201.hxaco",
    sha256: "09bb7f0ed151eb3c18936af292f9220a8cc964ccba307c163799e5a8e95d4606",
};
const OBJ_W3_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_w3_pm_gfx1201",
    artifact: "kernels/qwen4_moe_iu4_sym_w3_pm_gfx1201.hxaco",
    sha256: "b1c05e645d424491e3d1dd69e6145ab2d6081fb97474662fd1fa14351ba5200f",
};
const OBJ_P_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_iu4_sym_p_pm_gfx1201",
    artifact: "kernels/qwen4_moe_iu4_sym_p_pm_gfx1201.hxaco",
    sha256: "7caed8e79c4d5ca07d379dfafc794475cb733a1d71874713a07d9f916b0d62dc",
};
const OBJ_DOWN_GP_P_GFX1201: ObjectRefV1 = ObjectRefV1::Embedded {
    module: "qwen4_moe_down_gp_p_pm_gfx1201",
    artifact: "kernels/qwen4_moe_down_gp_p_pm_gfx1201.hxaco",
    sha256: "96aa9191de0b24615b034ea001eaa14859a10a8c3141f594dba43e814e8b0050",
};

const fn export(
    symbol: &'static str,
    arch: Arch,
    object: ObjectRefV1,
    block_x: u32,
    group_segment_fixed: u32,
    grid_x: Expr,
    consts: &'static [(&'static str, u64)],
) -> ExportV1 {
    ExportV1 {
        symbol,
        arch,
        object,
        block: [block_x, 1, 1],
        dynamic_lds: 0,
        group_segment_fixed,
        grid: [grid_x, GRID_Y, Expr::Const(1)],
        consts,
    }
}

// Export constants: `rr` row blocks per CTA, `rows_per_cta` = 64 * rr (grid x =
// M / rows_per_cta), `tiles_per_cta` 16-slot tiles per weight stream (NT), `waves`
// = block / 32.
const C_NT1_GATE_UP: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 1),
    ("waves", 2),
];
const C_NT4_GATE_UP: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 4),
    ("waves", 4),
];
const C_NT8_GATE_UP: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 8),
    ("waves", 4),
];
const C_NT1_DOWN: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 1),
    ("waves", 4),
];
const C_NT4_DOWN: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 4),
    ("waves", 4),
];
const C_NT8_DOWN: &[(&str, u64)] = &[
    ("rr", 1),
    ("rows_per_cta", 64),
    ("tiles_per_cta", 8),
    ("waves", 4),
];
const C_NT4_DOWN_RR2: &[(&str, u64)] = &[
    ("rr", 2),
    ("rows_per_cta", 128),
    ("tiles_per_cta", 4),
    ("waves", 4),
];
const C_NT4_DOWN_RR4: &[(&str, u64)] = &[
    ("rr", 4),
    ("rows_per_cta", 256),
    ("tiles_per_cta", 4),
    ("waves", 4),
];

// Per export: block = `Spec::threads` (gate/up NT1 two waves = 64, every other
// entry four waves = 128; launcher `kernel.block_threads` / `QWEN4_MOE_SYM_NT1_BLOCK`),
// group_segment_fixed = the object's `.group_segment_fixed_size` (equals
// `Spec::lds_bytes` = NT * 2048 for NT>1 gate/up, else 0), segment 64 / align 8,
// `.max_flat_workgroup_size` = block, wavefront 32, no hidden words, in every PM entry.

// gfx1100 (NT1 only; `MOE_SYM_IU4_KERNELS[0]`, `qwen4_moe_sym_nt1_gemm`).
const GFX1100_GATE_UP: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100",
    Arch::Gfx1100,
    OBJ_PM_GFX1100,
    64,
    0,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1100_DOWN: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1100",
    Arch::Gfx1100,
    OBJ_PM_GFX1100,
    128,
    0,
    GRID_X_RR1,
    C_NT1_DOWN,
);

// gfx1151: the NT1 anchors (block 64/128, byte anchors of the oracle) and the NT4
// serving entries (`MOE_SYM_IU4_KERNELS[1]`), stream8 gate/up, down row repeats.
const GFX1151_GATE_UP_NT1: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    64,
    0,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1151_GATE_UP_NT4: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    128,
    8192,
    GRID_X_RR1,
    C_NT4_GATE_UP,
);
/// Selected only by the launcher's measured-shape allowlist
/// (`qwen4_moe_stream8_selected`: gfx1151, resident, M 1280, K 2560, top-k 10,
/// 512 experts, 7168..=8192 token rows); the kernel itself is the NT8 gate/up with the
/// same 64-byte ABI and grid (`emit_gate_up_stream8`: "The 64-byte ABI and grid are
/// unchanged"), so the shape allowlist stays a launcher selection rule, not a domain
/// rule. 16384 B gate-handoff LDS, 208 VGPR.
const GFX1151_GATE_UP_STREAM8: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
    Arch::Gfx1151,
    OBJ_STREAM8_GFX1151,
    128,
    16384,
    GRID_X_RR1,
    C_NT8_GATE_UP,
);
const GFX1151_DOWN_NT1: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1151",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    128,
    0,
    GRID_X_RR1,
    C_NT1_DOWN,
);
const GFX1151_DOWN_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    128,
    0,
    GRID_X_RR1,
    C_NT4_DOWN,
);
const GFX1151_DOWN_NT4R2: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r2",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    128,
    0,
    GRID_X_RR2,
    C_NT4_DOWN_RR2,
);
const GFX1151_DOWN_NT4R4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r4",
    Arch::Gfx1151,
    OBJ_PM_GFX1151,
    128,
    0,
    GRID_X_RR4,
    C_NT4_DOWN_RR4,
);

// gfx1201: NT1 anchors, NT4 resident and NT8 host-mapped / wide gate-up
// (`MOE_SYM_IU4_KERNELS[2]`, `QWEN4_MOE_SYM_PM_GATE_UP[1]`, `QWEN4_MOE_SYM_PM_DOWN[1]`).
const GFX1201_GATE_UP_NT1: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    64,
    0,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1201_GATE_UP_NT4: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    128,
    8192,
    GRID_X_RR1,
    C_NT4_GATE_UP,
);
const GFX1201_GATE_UP_NT8: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    128,
    16384,
    GRID_X_RR1,
    C_NT8_GATE_UP,
);
const GFX1201_DOWN_NT1: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT1_DOWN,
);
const GFX1201_DOWN_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT4_DOWN,
);
const GFX1201_DOWN_NT8: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_PM_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT8_DOWN,
);

// Grouped-position down twins (`qwen4_moe_down_gp_pm_gfx1201`; the module's first
// export `moe_rotate_i4_gp_pm_gfx1201` is the activation producer, a different ABI).
const GFX1201_DOWN_GP_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_DOWN_GP_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT4_DOWN,
);
const GFX1201_DOWN_GP_NT8: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_DOWN_GP_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT8_DOWN,
);

// gfx1151 hipcc twin (`MoeSymIu4HipccTwin.entries`: the NT4 pair `QWEN4_MOE_SYM_GATE_UP` /
// `QWEN4_MOE_SYM_DOWN`; the module's 16-slot entries are never launched by the
// launcher). No offline object exists, so `group_segment_fixed` is the static LDS the
// source declares: gate/up `sG[16][32] f32` (2048) + `sOut[16][32] u16` (1024) +
// `sym_run_pos`'s `s_min` (4) = 3076; down `sOut[16][64] u16` (2048) + `s_min` (4) =
// 2052. Both agree with a `hipcc --genco --offload-arch=gfx1151 -O3
// -mcode-object-version=6` build of the module (HIP 7.16.26385, AMD clang 24.0.0git),
// whose `.group_segment_fixed_size` is 3076 / 2052, `.max_flat_workgroup_size` 128.
const GFX1151_HIPCC_GATE_UP_NT4: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_gfx1151_nt4",
    Arch::Gfx1151,
    OBJ_HIPCC_GFX1151,
    128,
    3076,
    GRID_X_RR1,
    C_NT4_GATE_UP,
);
const GFX1151_HIPCC_DOWN_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_gfx1151_nt4",
    Arch::Gfx1151,
    OBJ_HIPCC_GFX1151,
    128,
    2052,
    GRID_X_RR1,
    C_NT4_DOWN,
);

// Sealed-slab E8P-I4 NT1 serving entries (`SLAB_K2_GFX1100_E8P`, `SLAB_K2_GFX1201_E8P`,
// `SLAB_K2_GFX1151_E8P`; never the `_acc` twins). Metadata of every entry: segment 64 /
// align 8, no hidden words, wave 32, `.group_segment_fixed_size` 1536 (the LUT[256] +
// MASK[128] LDS stage, `Spec::lds_bytes` for `WeightCodec::E8pI4`), max_flat_workgroup_size
// 64 (gate/up) / 128 (down). gfx1151 is admission-gated in the launcher (`slab_k2_entry`).
const GFX1100_GATE_UP_E8P: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1100",
    Arch::Gfx1100,
    OBJ_E8P_GFX1100,
    64,
    1536,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1100_DOWN_E8P: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1100",
    Arch::Gfx1100,
    OBJ_E8P_GFX1100,
    128,
    1536,
    GRID_X_RR1,
    C_NT1_DOWN,
);
const GFX1151_GATE_UP_E8P: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1151",
    Arch::Gfx1151,
    OBJ_E8P_GFX1151,
    64,
    1536,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1151_DOWN_E8P: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1151",
    Arch::Gfx1151,
    OBJ_E8P_GFX1151,
    128,
    1536,
    GRID_X_RR1,
    C_NT1_DOWN,
);
const GFX1201_GATE_UP_E8P: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_E8P_GFX1201,
    64,
    1536,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1201_DOWN_E8P: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_e8p_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_E8P_GFX1201,
    128,
    1536,
    GRID_X_RR1,
    C_NT1_DOWN,
);

// MQ3-I4 (W3) gfx1201 NT1 entries (`QWEN4_MOE_SYM_W3_GATE_UP` / `_DOWN`, never the `_acc`
// twins): segment 64 / align 8, no hidden words, `.group_segment_fixed_size` 0, block 64 / 128.
const GFX1201_GATE_UP_W3: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_W3_GFX1201,
    64,
    0,
    GRID_X_RR1,
    C_NT1_GATE_UP,
);
const GFX1201_DOWN_W3: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_w3_pm_gfx1201",
    Arch::Gfx1201,
    OBJ_W3_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT1_DOWN,
);

// DESIGN F1: drop-zero planar MQ4 gfx1201 NT4/NT8 entries (`qwen4_moe_iu4_sym_p_pm_gfx1201`,
// `qwen4_moe_down_gp_p_pm_gfx1201`): the shipped entries' block, LDS, grid and constants.
const GFX1201_GATE_UP_P_NT4: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_P_GFX1201,
    128,
    8192,
    GRID_X_RR1,
    C_NT4_GATE_UP,
);
const GFX1201_GATE_UP_P_NT8: ExportV1 = export(
    "qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_P_GFX1201,
    128,
    16384,
    GRID_X_RR1,
    C_NT8_GATE_UP,
);
const GFX1201_DOWN_P_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_P_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT4_DOWN,
);
const GFX1201_DOWN_P_NT8: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_p_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_P_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT8_DOWN,
);
const GFX1201_DOWN_GP_P_NT4: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt4",
    Arch::Gfx1201,
    OBJ_DOWN_GP_P_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT4_DOWN,
);
const GFX1201_DOWN_GP_P_NT8: ExportV1 = export(
    "qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt8",
    Arch::Gfx1201,
    OBJ_DOWN_GP_P_GFX1201,
    128,
    0,
    GRID_X_RR1,
    C_NT8_DOWN,
);

/// Hidden words of the hipcc NT4 entries (they read `blockDim`, so hipcc keeps the
/// implicit-arg block). Measured on the same build as above; the layout is the
/// toolchain's and is not pinned by an object, and the ordinary `kernelParams` launch
/// never encodes it (the runtime fills it). `kernarg_segment_size` 320, align 8.
const HIPCC_HIDDEN: [HiddenV1; 13] = [
    HiddenV1 {
        offset: 64,
        width: 4,
        kind: HiddenKind::BlockCount(Axis::X),
    },
    HiddenV1 {
        offset: 68,
        width: 4,
        kind: HiddenKind::BlockCount(Axis::Y),
    },
    HiddenV1 {
        offset: 72,
        width: 4,
        kind: HiddenKind::BlockCount(Axis::Z),
    },
    HiddenV1 {
        offset: 76,
        width: 2,
        kind: HiddenKind::GroupSize(Axis::X),
    },
    HiddenV1 {
        offset: 78,
        width: 2,
        kind: HiddenKind::GroupSize(Axis::Y),
    },
    HiddenV1 {
        offset: 80,
        width: 2,
        kind: HiddenKind::GroupSize(Axis::Z),
    },
    HiddenV1 {
        offset: 82,
        width: 2,
        kind: HiddenKind::Remainder(Axis::X),
    },
    HiddenV1 {
        offset: 84,
        width: 2,
        kind: HiddenKind::Remainder(Axis::Y),
    },
    HiddenV1 {
        offset: 86,
        width: 2,
        kind: HiddenKind::Remainder(Axis::Z),
    },
    HiddenV1 {
        offset: 104,
        width: 8,
        kind: HiddenKind::GlobalOffset(Axis::X),
    },
    HiddenV1 {
        offset: 112,
        width: 8,
        kind: HiddenKind::GlobalOffset(Axis::Y),
    },
    HiddenV1 {
        offset: 120,
        width: 8,
        kind: HiddenKind::GlobalOffset(Axis::Z),
    },
    HiddenV1 {
        offset: 128,
        width: 2,
        kind: HiddenKind::GridDims,
    },
];

// ---------------------------------------------------------------------------
// Contracts
// ---------------------------------------------------------------------------

/// Builder (PeaceMaker) SwiGLU gate/up: gfx1100 NT1, gfx1151 NT1/NT4/stream8, gfx1201
/// NT1/NT4/NT8.
pub static QWEN4_MOE_SYM_GATE_UP: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.gate_up.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GATE_UP_SWIGLU,
    params: &PARAMS,
    operands: &OPS_GATE_UP,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_GATE_UP,
    exports: &[
        GFX1100_GATE_UP,
        GFX1151_GATE_UP_NT1,
        GFX1151_GATE_UP_NT4,
        GFX1151_GATE_UP_STREAM8,
        GFX1201_GATE_UP_NT1,
        GFX1201_GATE_UP_NT4,
        GFX1201_GATE_UP_NT8,
    ],
};

/// Builder down, row repeat 1: gfx1100 NT1, gfx1151 NT1/NT4, gfx1201 NT1/NT4/NT8.
pub static QWEN4_MOE_SYM_DOWN: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN,
    exports: &[
        GFX1100_DOWN,
        GFX1151_DOWN_NT1,
        GFX1151_DOWN_NT4,
        GFX1201_DOWN_NT1,
        GFX1201_DOWN_NT4,
        GFX1201_DOWN_NT8,
    ],
};

/// gfx1151 down NT4 with row repeat 2 (`HIPFIRE_QWEN4_MOE_SYM_DOWN_RR=2`): grid x is
/// `M / 128`, so `M % 128 == 0` is part of this contract's domain.
pub static QWEN4_MOE_SYM_DOWN_RR2: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_rr2.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN_RR2,
    exports: &[GFX1151_DOWN_NT4R2],
};

/// gfx1151 down NT4 with row repeat 4: grid x is `M / 256`.
pub static QWEN4_MOE_SYM_DOWN_RR4: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_rr4.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN_RR4,
    exports: &[GFX1151_DOWN_NT4R4],
};

/// gfx1201 grouped-position down twins (NT4/NT8): the down numerics with the
/// grouped-position activation plane. Domain as the builder down (the launcher passes
/// the same extents and bounds for the `_gp` entry).
pub static QWEN4_MOE_SYM_DOWN_GP: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_gp.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN_GP,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN,
    exports: &[GFX1201_DOWN_GP_NT4, GFX1201_DOWN_GP_NT8],
};

/// gfx1151 hipcc twin gate/up (`HIPFIRE_QWEN4_MOE_SYM_PM=0`): the same operands and
/// numerics as the builder entries, no slot/Y-byte guard, hipcc's hidden block.
pub static QWEN4_MOE_SYM_GATE_UP_HIPCC: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.gate_up.hipcc",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GATE_UP_SWIGLU,
    params: &PARAMS,
    operands: &OPS_GATE_UP,
    fields: &FIELDS,
    hidden: &HIPCC_HIDDEN,
    explicit_bytes: 64,
    segment_bytes: 320,
    segment_align: 8,
    domain: &DOMAIN_GATE_UP_HIPCC,
    exports: &[GFX1151_HIPCC_GATE_UP_NT4],
};

/// gfx1151 hipcc twin down.
pub static QWEN4_MOE_SYM_DOWN_HIPCC: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down.hipcc",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN,
    fields: &FIELDS,
    hidden: &HIPCC_HIDDEN,
    explicit_bytes: 64,
    segment_bytes: 320,
    segment_align: 8,
    domain: &DOMAIN_DOWN_HIPCC,
    exports: &[GFX1151_HIPCC_DOWN_NT4],
};

/// Sealed-slab E8P-I4 gate/up, NT1 (`qwen4_moe_sym_slab_gemm` over `SlabK2`, codec 3):
/// gfx1100, gfx1151 and gfx1201. The `_acc` qualification twins are never launched.
pub static QWEN4_MOE_SYM_GATE_UP_E8P: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.gate_up_e8p.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GATE_UP_SWIGLU_E8P,
    params: &PARAMS,
    operands: &OPS_GATE_UP_E8P,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_GATE_UP_PLANAR,
    exports: &[
        GFX1100_GATE_UP_E8P,
        GFX1151_GATE_UP_E8P,
        GFX1201_GATE_UP_E8P,
    ],
};

/// Sealed-slab E8P-I4 down, NT1: gfx1100, gfx1151 and gfx1201.
pub static QWEN4_MOE_SYM_DOWN_E8P: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_e8p.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN_E8P,
    params: &PARAMS,
    operands: &OPS_DOWN_E8P,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN_PLANAR,
    exports: &[GFX1100_DOWN_E8P, GFX1151_DOWN_E8P, GFX1201_DOWN_E8P],
};

/// MQ3-I4 (W3) gate/up, gfx1201 NT1 (`qwen4_moe_sym_w3_gemm` -> `qwen4_moe_sym_nt1_gemm`).
/// The gfx1151 W3 NT4 launcher and its module are not part of this contract.
pub static QWEN4_MOE_SYM_GATE_UP_W3: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.gate_up_w3.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GATE_UP_SWIGLU_W3,
    params: &PARAMS,
    operands: &OPS_GATE_UP_W3,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_GATE_UP_PLANAR,
    exports: &[GFX1201_GATE_UP_W3],
};

/// MQ3-I4 (W3) down, gfx1201 NT1.
pub static QWEN4_MOE_SYM_DOWN_W3: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_w3.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN_W3,
    params: &PARAMS,
    operands: &OPS_DOWN_W3,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN_PLANAR,
    exports: &[GFX1201_DOWN_W3],
};

/// DESIGN F1: drop-zero planar MQ4 (`MQ4G256V2P`) gate/up, gfx1201 NT4/NT8. Numerics, params and
/// domain of the shipped builder gate/up; only the expert tensors' layout differs.
pub static QWEN4_MOE_SYM_GATE_UP_P: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.gate_up_p.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_GATE_UP_SWIGLU,
    params: &PARAMS,
    operands: &OPS_GATE_UP_P,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_GATE_UP,
    exports: &[GFX1201_GATE_UP_P_NT4, GFX1201_GATE_UP_P_NT8],
};

/// DESIGN F1: drop-zero planar MQ4 (`MQ4G128V2P`) down, gfx1201 NT4/NT8.
pub static QWEN4_MOE_SYM_DOWN_P: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_p.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN_P,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN,
    exports: &[GFX1201_DOWN_P_NT4, GFX1201_DOWN_P_NT8],
};

/// DESIGN F1: drop-zero planar MQ4 grouped-position down twins, gfx1201 NT4/NT8.
pub static QWEN4_MOE_SYM_DOWN_GP_P: KernelAbiV1 = KernelAbiV1 {
    id: "qwen4_moe_sym.down_gp_p.pm",
    version: 1,
    wave: 32,
    numeric: NUMERIC_DOWN,
    params: &PARAMS,
    operands: &OPS_DOWN_GP_P,
    fields: &FIELDS,
    hidden: &[],
    explicit_bytes: 64,
    segment_bytes: 64,
    segment_align: 8,
    domain: &DOMAIN_DOWN,
    exports: &[GFX1201_DOWN_GP_P_NT4, GFX1201_DOWN_GP_P_NT8],
};

/// Every contract of this family, in a stable order.
pub static ABIS: &[&KernelAbiV1] = &[
    &QWEN4_MOE_SYM_GATE_UP,
    &QWEN4_MOE_SYM_DOWN,
    &QWEN4_MOE_SYM_DOWN_RR2,
    &QWEN4_MOE_SYM_DOWN_RR4,
    &QWEN4_MOE_SYM_DOWN_GP,
    &QWEN4_MOE_SYM_GATE_UP_HIPCC,
    &QWEN4_MOE_SYM_DOWN_HIPCC,
    &QWEN4_MOE_SYM_GATE_UP_E8P,
    &QWEN4_MOE_SYM_DOWN_E8P,
    &QWEN4_MOE_SYM_GATE_UP_W3,
    &QWEN4_MOE_SYM_DOWN_W3,
    &QWEN4_MOE_SYM_GATE_UP_P,
    &QWEN4_MOE_SYM_DOWN_P,
    &QWEN4_MOE_SYM_DOWN_GP_P,
];
