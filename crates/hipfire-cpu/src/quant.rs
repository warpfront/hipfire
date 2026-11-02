// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Quant decode for the CPU-executed offload path.
//!
//! Every arm is a transcription of
//! `hipfire_runtime::weight_backend::dequantize_to_f32` (the single canonical
//! CPU decoder — see its doc comment: "there is exactly one copy") and, for
//! qt 3, of `hipfire_runtime::llama::dequantize_q8_0`. The transcription exists
//! rather than a call because this crate must not link the GPU stack (see the
//! crate docs). `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs` is what
//! keeps the two copies honest.
//!
//! Two decodes are exposed, and the difference between them is *the* thing to
//! get right in this file:
//!
//! * [`decode_group_codes`] — the affine/level decode only. These are the code
//!   values the MQ GEMV kernels dot against a **forward-rotated** activation
//!   (`rotate_x`): the weights are stored post-rotation, and
//!   `dot(rot W, rot x) == dot(W, x)` is what makes the product come out in the
//!   original basis. `crate::gemv` uses this one.
//! * [`dequant_group`] — the same affine decode followed by the *inverse* FWHT
//!   ([`fwht256_inplace`]) for the FWHT-rotated formats, i.e. the weights in the
//!   original basis, matching `dequantize_to_f32` value for value.
//!
//! Mixing them — un-rotating the weights *and* rotating the activation — is a
//! silent $\mathcal{R}^{-2}$ error, not a crash. `f32` arithmetic here is
//! ordinary `mul`/`add` in the canonical decoder's order, so
//! [`test::dequant_group_matches_canonical_decoder_bits`] holds exactly.

use std::sync::LazyLock;

/// Seeds of the MagnumQuant FWHT sign vectors, mirroring
/// `Gpu::ensure_mq_signs` (which uploads `KvCache::gen_fwht_signs(42|1042, 256)`).
const FWHT_SEED1: u32 = 42;
const FWHT_SEED2: u32 = 1042;
/// Every G256 format rotates in fixed 256-element groups.
const FWHT_N: usize = 256;

/// Quant formats the CPU offload path can decode.
///
/// This is exactly the set the qwen3.5 dense registry offers, as measured on
/// the pulled fixtures rather than assumed: `qwen3.5-2b.mq4` (qt 13),
/// `qwen3.5-2b.mq3` (**qt 20**, not qt 17 — the `-mq3` tags ship the Lloyd-Max
/// 3-bit tier), `qwen3.5-2b.mq6` (qt 15), `qwen3.5-2b.hf6` (qt 8), the `mq4v2`
/// bodies current `hipfire-quantize --format mq4` writes (qt 44), and plain
/// qt 17 for checkpoints that carry the uniform 3-bit tier — plus the blocking
/// formats those files use for norms, the embedding and structural tensors. Anything else has no CPU implementation, so a step over it stays on
/// the GPU over PCIe: correctness is unaffected, only the bandwidth win is
/// partial, and the load-time coverage line names the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuQuant {
    /// qt 13 — FWHT-rotated HFQ4-G256, 136 B / 256.
    Mq4G256,
    /// qt 44 — FWHT-rotated HFQ4-G256 v2, per-128 fp16 scale/zero, 136 B / 256.
    Mq4G256V2,
    /// qt 15 — FWHT-rotated HFQ6-G256, 200 B / 256.
    Mq6G256,
    /// qt 17 — FWHT-rotated HFQ3-G256, 104 B / 256.
    Mq3G256,
    /// qt 20 — FWHT-rotated 3-bit with a per-group 8-entry fp16 Lloyd-Max
    /// codebook, 112 B / 256. This is what the registry's `qwen3.5:2b-mq3`
    /// (and the other `-mq3` tags) actually contain — measured, not assumed.
    Mq3G256Lloyd,
    /// qt 8 — HFQ6-G256, no rotation, 200 B / 256.
    Hfq6G256,
    /// qt 11 — HFQ3-G256, no rotation, 104 B / 256.
    Hfq3G256,
    /// qt 9 — HFQ2-G256, no rotation, 72 B / 256.
    Hfq2G256,
    /// qt 7 — HFQ4-G128, no rotation, 72 B / 128.
    Hfq4G128,
    /// qt 12 — HFQ3-G128, no rotation, 56 B / 128.
    Hfq3G128,
    /// qt 10 — HFQ2-G128, no rotation, 40 B / 128.
    Hfq2G128,
    /// qt 31 — FWHT-rotated 5-bit HFQ-G256 (f32 header), 168 B / 256.
    Mq5G256,
    /// qt 18 — FWHT-rotated HFQ2-G256, 72 B / 256.
    Mq2G256,
    /// qt 19 — FWHT-rotated 2-bit with a per-group 4-entry fp16 Lloyd codebook,
    /// 72 B / 256.
    Mq2G256Lloyd,
    /// qt 51 — the *unrotated* sibling of qt 19 (byte-identical layout). Carries
    /// native-ternary checkpoints losslessly, so it must never be rotated.
    Mq2G256LloydU,
    /// qt 30 — FWHT-rotated 4-bit with a per-group 16-entry fp16 Lloyd codebook,
    /// 160 B / 256.
    Mq4G256Lloyd,
    /// qt 45 — FWHT-rotated HFQ4-G256 with an fp16 `[scale][zero]` header, 136 B / 256.
    Mq4CG256,
    /// qt 47 — FWHT-rotated 6-bit, per-128 fp16 scale/zero, 200 B / 256.
    Mq6G256V2,
    /// qt 48 — FWHT-rotated 5-bit, per-128 fp16 scale/zero, 168 B / 256.
    Mq5G256V2,
    /// qt 49 — FWHT-rotated 3-bit, per-128 fp16 scale/zero, 104 B / 256.
    Mq3G256V2,
    /// qt 50 — FWHT-rotated 2-bit, per-128 fp16 scale/zero, 72 B / 256.
    Mq2G256V2,
    /// qt 53 — FWHT-128-rotated 4-bit, row-local `ceil(k/128)` groups of 68 B:
    /// `[f16 scale][f16 zero][64 B nibbles]`, byte `i` = `q[2i] | q[2i+1] << 4`,
    /// `w = scale*q + zero`. The Flash-Next routed-expert down projection
    /// (`DType::MQ4G128V2`). Its activation rotation is the 128-wide
    /// [`rotate_x_128_v2`], not the 256-wide [`rotate_x`].
    Mq4G128V2,
    /// qt 40 — TQ2G128 ternary: fp16 `d` + 32 B of 2-bit codes, `(code-1)*d`,
    /// no rotation, 34 B / 128.
    Tq2G128,
    /// qt 41 — BQ1G128 binary: fp16 `d` + 16 B of sign bits, `±d`, no rotation,
    /// 18 B / 128.
    Bq1G128,
    /// qt 6 — HFQ4-G256, no rotation, 136 B / 256.
    Hfq4G256,
    /// qt 1 — f16, 2 B / element.
    F16,
    /// qt 2 — f32, 4 B / element.
    F32,
    /// qt 16 — bf16, 2 B / element.
    Bf16,
    /// qt 3 — `Q8F16` (`DType::Q8_0`): f16 scale + 32 i8, 34 B / 32.
    Q8F16,
    /// qt 34 — FWHT-256 E8 lattice weights: one 16 B row header,
    /// then eight 17 B blocks per 256 values.
    Mfp4G32E8,
    /// qt 131 — MQ3-I4 planar slab, FWHT-256 (gate_up). Codes plane of 3-bit
    /// codes (`M*K*3/8` B; eight codes in three bytes, LSB first), then an f16
    /// scale plane (`M*(K/128)`); `w = f32(d) * (2c - 7)`. **Planar**: see
    /// [`Self::is_planar`]. Wire source: `hipfire_runtime::mq_i4`.
    Mq3I4G128R256,
    /// qt 132 — MQ3-I4 planar slab, FWHT-128 (down; the qt 53 rotation). As
    /// [`Self::Mq3I4G128R256`].
    Mq3I4G128R128,
    /// qt 135 — E8P-I4 planar slab, FWHT-256 (gate_up). Codes plane of
    /// little-endian u16 words (`idx | sign7 << 8 | shift << 15`, eight weights
    /// each; `M*K/4` B), then an f16 scale plane (`M*(K/128)`);
    /// `w = (f32(d) * 0.25) * v`, `v` from [`e8p_decode_word`]. **Planar**: see
    /// [`Self::is_planar`]. Wire source: `hipfire_runtime::e8p_i4_reference`.
    E8pI4G128R256,
    /// qt 136 — E8P-I4 planar slab, FWHT-128 (down). As [`Self::E8pI4G128R256`].
    E8pI4G128R128,
}

/// Which forward rotation a format's activation needs before the stored codes
/// are dotted against it (see the module docs of [`crate::gemv`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRotation {
    /// Natural basis: `x` is used as is.
    None,
    /// Blockwise FWHT-256, seeds 42/1042: [`rotate_x`].
    FwhtG256,
    /// Blockwise FWHT-128 with a zero-filled tail, seeds 43/1043:
    /// [`rotate_x_128_v2`] (qt 53).
    FwhtG128V2,
}

impl CpuQuant {
    /// Map the serialized HFQ `quant_type` byte (see `hipfire_quantize`'s
    /// `QuantType`) to a CPU-decodable format, or `None` when this crate has no
    /// implementation for it.
    pub fn from_quant_type(quant_type: u8) -> Option<Self> {
        match quant_type {
            1 => Some(Self::F16),
            2 => Some(Self::F32),
            3 => Some(Self::Q8F16),
            6 => Some(Self::Hfq4G256),
            7 => Some(Self::Hfq4G128),
            8 => Some(Self::Hfq6G256),
            9 => Some(Self::Hfq2G256),
            10 => Some(Self::Hfq2G128),
            11 => Some(Self::Hfq3G256),
            12 => Some(Self::Hfq3G128),
            16 => Some(Self::Bf16),
            18 => Some(Self::Mq2G256),
            19 => Some(Self::Mq2G256Lloyd),
            30 => Some(Self::Mq4G256Lloyd),
            31 => Some(Self::Mq5G256),
            34 => Some(Self::Mfp4G32E8),
            40 => Some(Self::Tq2G128),
            41 => Some(Self::Bq1G128),
            45 => Some(Self::Mq4CG256),
            47 => Some(Self::Mq6G256V2),
            48 => Some(Self::Mq5G256V2),
            49 => Some(Self::Mq3G256V2),
            50 => Some(Self::Mq2G256V2),
            51 => Some(Self::Mq2G256LloydU),
            53 => Some(Self::Mq4G128V2),
            13 => Some(Self::Mq4G256),
            15 => Some(Self::Mq6G256),
            17 => Some(Self::Mq3G256),
            20 => Some(Self::Mq3G256Lloyd),
            44 => Some(Self::Mq4G256V2), // central entries above; see FORMATS in tests
            131 => Some(Self::Mq3I4G128R256),
            132 => Some(Self::Mq3I4G128R128),
            135 => Some(Self::E8pI4G128R256),
            136 => Some(Self::E8pI4G128R128),
            _ => None,
        }
    }

    /// Elements per group. Rows also include [`Self::row_header_bytes`].
    pub fn group_elems(self) -> usize {
        match self {
            Self::Q8F16 => 32,
            // The three G128 formats: TQ2/BQ1 are flat element formats with a
            // 128-element block, and the HFQ G128 pair packs 128 weights.
            Self::Hfq4G128
            | Self::Hfq3G128
            | Self::Hfq2G128
            | Self::Tq2G128
            | Self::Bq1G128
            | Self::Mq4G128V2
            | Self::Mq3I4G128R256
            | Self::Mq3I4G128R128
            | Self::E8pI4G128R256
            | Self::E8pI4G128R128 => 128,
            _ => 256,
        }
    }

    /// Bytes per group, including any per-group header.
    pub fn group_bytes(self) -> usize {
        match self {
            Self::Mq4G256 | Self::Mq4G256V2 | Self::Hfq4G256 | Self::Mq4CG256 => 136,
            Self::Mq6G256 | Self::Hfq6G256 | Self::Mq6G256V2 => 200,
            Self::Mq5G256 | Self::Mq5G256V2 => 168,
            Self::Mq3G256 | Self::Mq3G256V2 | Self::Hfq3G256 => 104,
            Self::Mq3G256Lloyd => 112,
            Self::Mq2G256
            | Self::Hfq2G256
            | Self::Mq2G256Lloyd
            | Self::Mq2G256LloydU
            | Self::Mq2G256V2 => 72,
            Self::Mq4G256Lloyd => 160,
            Self::Hfq4G128 => 72,
            Self::Hfq3G128 => 56,
            Self::Hfq2G128 => 40,
            Self::Tq2G128 => 34,
            Self::Bq1G128 => 18,
            Self::Mq4G128V2 => 68,
            Self::Q8F16 => 34,
            Self::F16 | Self::Bf16 => 512,
            Self::F32 => 1024,
            // Planar formats: code bytes of one 128-group only; the f16 scale
            // lives in a separate plane (see `is_planar`).
            Self::Mq3I4G128R256 | Self::Mq3I4G128R128 => 48,
            Self::E8pI4G128R256 | Self::E8pI4G128R128 => 32,
            Self::Mfp4G32E8 => 136,
        }
    }

    /// Bytes preceding the groups once per row, not once per FWHT group.
    pub fn row_header_bytes(self) -> usize {
        match self {
            Self::Mfp4G32E8 => 16,
            _ => 0,
        }
    }

    /// Whether the format's weights are stored FWHT-256-rotated, and therefore
    /// whether the activation must be forward-rotated before the dot product.
    ///
    /// Invariant: this agrees with `hipfire_dispatch::types::dtype_rotation_plan`
    /// on `RotationPlan::FwhtG256` for every dtype that maps here (pinned by
    /// `crates/hipfire-dispatch`'s `cpu_quant_rotation_agrees_with_plan`).
    pub fn is_fwht_g256(self) -> bool {
        matches!(
            self,
            Self::Mq4G256
                | Self::Mq4G256V2
                | Self::Mq4CG256
                | Self::Mq6G256
                | Self::Mq6G256V2
                | Self::Mq5G256
                | Self::Mq5G256V2
                | Self::Mq3G256
                | Self::Mq3G256V2
                | Self::Mq3G256Lloyd
                | Self::Mq2G256
                | Self::Mq2G256V2
                | Self::Mq2G256Lloyd
                | Self::Mq4G256Lloyd
                | Self::Mfp4G32E8
                | Self::Mq3I4G128R256
                | Self::E8pI4G128R256
        )
        // NOT here, deliberately: `Mq2G256LloydU` (unrotated sibling — rotating
        // it would destroy the native-ternary structure it exists to preserve),
        // the HFQ/TQ2/BQ1 formats (encoded in the natural basis), and MQ8 (whose
        // activation contract is `RotationPlan::Mq8Internal`, an int8-quantized
        // variant this crate does not implement — see the coverage note).
    }

    /// Whether the format's weights are stored FWHT-128-rotated (qt 53
    /// `Mq4G128V2`), i.e. its activation needs [`rotate_x_128_v2`]. Disjoint
    /// from [`Self::is_fwht_g256`].
    pub fn is_fwht_g128(self) -> bool {
        matches!(
            self,
            Self::Mq4G128V2 | Self::Mq3I4G128R128 | Self::E8pI4G128R128
        )
    }

    /// The forward activation rotation this format's codes are dotted against.
    /// Agrees with [`Self::is_fwht_g256`] / [`Self::is_fwht_g128`].
    pub fn input_rotation(self) -> InputRotation {
        if self.is_fwht_g256() {
            InputRotation::FwhtG256
        } else if self.is_fwht_g128() {
            InputRotation::FwhtG128V2
        } else {
            InputRotation::None
        }
    }

    /// The `k` granularity the GEMV entry points accept for this format: the
    /// 128-element-group formats take any multiple of 128 (Flash-Next's down
    /// projection is `k = 640`), everything else needs a multiple of 256.
    pub fn k_multiple(self) -> usize {
        match self {
            // Planar R256 profiles rotate in 256-wide blocks (`k % 256`); the
            // R128 ones need only the 128-wide scale group.
            Self::Mq3I4G128R256 | Self::E8pI4G128R256 => 256,
            _ if self.group_elems() == 128 => 128,
            _ => 256,
        }
    }

    /// Whether the tensor is *planar*: `[codes plane][f16 scale plane]` over the
    /// whole `[M, K]` tensor instead of self-contained per-row groups (qt 131,
    /// 132, 135, 136). Planar rows are not contiguous with their scales, so a
    /// row is addressed as a `(codes_row, scales_row)` pair (see
    /// `crate::gemv::tensor_bytes`), the bytes of one row are not
    /// `row_bytes`, and [`decode_group_codes`] / [`dequant_group`] do not apply
    /// (use [`decode_planar_row`]).
    pub fn is_planar(self) -> bool {
        matches!(
            self,
            Self::Mq3I4G128R256 | Self::Mq3I4G128R128 | Self::E8pI4G128R256 | Self::E8pI4G128R128
        )
    }
}

// ---- planar expert-slab codecs (qt 131/132/135/136) --------------------------

/// Scale-group width of the planar formats, in columns.
pub const PLANAR_GROUP: usize = 128;

/// `2|g|` of the 256 QuIP# E8P abs-grid rows, in `vq_books.json` order (`idx` =
/// row); every entry is 1, 3 or 5. Verbatim transcription of
/// `hipfire_runtime::e8p_i4_reference::E8P_ABS2` (this crate cannot depend on
/// it); `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs` asserts equality.
#[rustfmt::skip]
pub const E8P_ABS2: [[u8; 8]; 256] = [
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 1, 1, 5], [1, 1, 1, 1, 1, 1, 3, 1],
    [1, 1, 1, 1, 1, 1, 3, 3], [1, 1, 1, 1, 1, 1, 3, 5], [1, 1, 1, 1, 1, 1, 5, 1], [1, 1, 1, 1, 1, 1, 5, 3],
    [1, 1, 1, 1, 1, 3, 1, 1], [1, 1, 1, 1, 1, 3, 1, 3], [1, 1, 1, 1, 1, 3, 1, 5], [1, 1, 1, 1, 1, 3, 3, 1],
    [1, 1, 1, 1, 1, 3, 3, 3], [1, 1, 1, 1, 1, 3, 5, 1], [1, 1, 1, 1, 1, 5, 1, 1], [1, 1, 1, 1, 1, 5, 1, 3],
    [1, 1, 1, 1, 1, 5, 3, 1], [1, 1, 1, 1, 3, 1, 1, 1], [1, 1, 1, 1, 3, 1, 1, 3], [1, 1, 1, 1, 3, 1, 1, 5],
    [1, 1, 1, 1, 3, 1, 3, 1], [1, 1, 1, 1, 3, 1, 3, 3], [1, 1, 1, 1, 3, 1, 5, 1], [1, 1, 1, 1, 3, 3, 1, 1],
    [1, 1, 1, 1, 3, 3, 1, 3], [1, 1, 1, 1, 3, 3, 3, 1], [1, 1, 1, 1, 3, 3, 3, 3], [1, 1, 1, 1, 3, 5, 1, 1],
    [1, 1, 1, 1, 5, 1, 1, 1], [1, 1, 1, 1, 5, 1, 1, 3], [1, 1, 1, 1, 5, 1, 3, 1], [1, 1, 1, 1, 5, 3, 1, 1],
    [1, 1, 1, 3, 1, 1, 1, 1], [1, 1, 1, 3, 1, 1, 1, 3], [1, 1, 1, 3, 1, 1, 1, 5], [1, 1, 1, 3, 1, 1, 3, 1],
    [1, 1, 1, 3, 1, 1, 3, 3], [1, 1, 1, 3, 1, 1, 5, 1], [1, 1, 1, 3, 1, 3, 1, 1], [1, 1, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 3, 1, 3, 3, 1], [1, 1, 1, 3, 1, 3, 3, 3], [1, 1, 1, 3, 1, 5, 1, 1], [1, 1, 1, 3, 3, 1, 1, 1],
    [1, 1, 1, 3, 3, 1, 1, 3], [1, 1, 1, 3, 3, 1, 3, 1], [1, 1, 1, 3, 3, 1, 3, 3], [1, 1, 1, 3, 3, 3, 1, 1],
    [1, 1, 1, 3, 3, 3, 1, 3], [1, 1, 1, 3, 3, 3, 3, 1], [1, 1, 1, 3, 5, 1, 1, 1], [1, 1, 1, 5, 1, 1, 1, 1],
    [1, 1, 1, 5, 1, 1, 1, 3], [1, 1, 1, 5, 1, 1, 3, 1], [1, 1, 1, 5, 1, 3, 1, 1], [1, 1, 1, 5, 3, 1, 1, 1],
    [1, 1, 3, 1, 1, 1, 1, 1], [1, 1, 3, 1, 1, 1, 1, 3], [1, 1, 3, 1, 1, 1, 1, 5], [1, 1, 3, 1, 1, 1, 3, 1],
    [1, 1, 3, 1, 1, 1, 3, 3], [1, 1, 3, 1, 1, 1, 5, 1], [1, 1, 3, 1, 1, 3, 1, 1], [1, 1, 3, 1, 1, 3, 1, 3],
    [1, 1, 3, 1, 1, 3, 3, 1], [1, 1, 3, 1, 1, 3, 3, 3], [1, 1, 3, 1, 1, 5, 1, 1], [1, 1, 3, 1, 3, 1, 1, 1],
    [1, 1, 3, 1, 3, 1, 1, 3], [1, 1, 3, 1, 3, 1, 3, 1], [1, 1, 3, 1, 3, 1, 3, 3], [1, 1, 3, 1, 3, 3, 1, 1],
    [1, 1, 3, 1, 3, 3, 1, 3], [1, 1, 3, 1, 3, 3, 3, 1], [1, 1, 3, 1, 5, 1, 1, 1], [1, 1, 3, 3, 1, 1, 1, 1],
    [1, 1, 3, 3, 1, 1, 1, 3], [1, 1, 3, 3, 1, 1, 3, 1], [1, 1, 3, 3, 1, 1, 3, 3], [1, 1, 3, 3, 1, 3, 1, 1],
    [1, 1, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 1], [1, 1, 3, 3, 3, 1, 1, 1], [1, 1, 3, 3, 3, 1, 1, 3],
    [1, 1, 3, 3, 3, 1, 3, 1], [1, 1, 3, 3, 3, 3, 1, 1], [1, 1, 3, 5, 1, 1, 1, 1], [1, 1, 5, 1, 1, 1, 1, 1],
    [1, 1, 5, 1, 1, 1, 1, 3], [1, 1, 5, 1, 1, 1, 3, 1], [1, 1, 5, 1, 1, 3, 1, 1], [1, 1, 5, 1, 3, 1, 1, 1],
    [1, 1, 5, 3, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1, 3], [1, 3, 1, 1, 1, 1, 1, 5],
    [1, 3, 1, 1, 1, 1, 3, 1], [1, 3, 1, 1, 1, 1, 3, 3], [1, 3, 1, 1, 1, 1, 5, 1], [1, 3, 1, 1, 1, 3, 1, 1],
    [1, 3, 1, 1, 1, 3, 1, 3], [1, 3, 1, 1, 1, 3, 3, 1], [1, 3, 1, 1, 1, 3, 3, 3], [1, 3, 1, 1, 1, 5, 1, 1],
    [1, 3, 1, 1, 3, 1, 1, 1], [1, 3, 1, 1, 3, 1, 1, 3], [1, 3, 1, 1, 3, 1, 3, 1], [1, 3, 1, 1, 3, 1, 3, 3],
    [1, 3, 1, 1, 3, 3, 1, 1], [1, 3, 1, 1, 3, 3, 1, 3], [1, 3, 1, 1, 3, 3, 3, 1], [1, 3, 1, 1, 5, 1, 1, 1],
    [1, 3, 1, 3, 1, 1, 1, 1], [1, 3, 1, 3, 1, 1, 1, 3], [1, 3, 1, 3, 1, 1, 3, 1], [1, 3, 1, 3, 1, 1, 3, 3],
    [1, 3, 1, 3, 1, 3, 1, 1], [1, 3, 1, 3, 1, 3, 1, 3], [1, 3, 1, 3, 1, 3, 3, 1], [1, 3, 1, 3, 3, 1, 1, 1],
    [1, 3, 1, 3, 3, 1, 1, 3], [1, 3, 1, 3, 3, 1, 3, 1], [1, 3, 1, 3, 3, 3, 1, 1], [1, 3, 1, 5, 1, 1, 1, 1],
    [1, 3, 3, 1, 1, 1, 1, 1], [1, 3, 3, 1, 1, 1, 1, 3], [1, 3, 3, 1, 1, 1, 3, 1], [1, 3, 3, 1, 1, 1, 3, 3],
    [1, 3, 3, 1, 1, 3, 1, 1], [1, 3, 3, 1, 1, 3, 1, 3], [1, 3, 3, 1, 1, 3, 3, 1], [1, 3, 3, 1, 3, 1, 1, 1],
    [1, 3, 3, 1, 3, 1, 1, 3], [1, 3, 3, 1, 3, 1, 3, 1], [1, 3, 3, 1, 3, 3, 1, 1], [1, 3, 3, 3, 1, 1, 1, 1],
    [1, 3, 3, 3, 1, 1, 1, 3], [1, 3, 3, 3, 1, 1, 3, 1], [1, 3, 3, 3, 1, 3, 1, 1], [1, 3, 3, 3, 3, 1, 1, 1],
    [1, 3, 5, 1, 1, 1, 1, 1], [1, 5, 1, 1, 1, 1, 1, 1], [1, 5, 1, 1, 1, 1, 1, 3], [1, 5, 1, 1, 1, 1, 3, 1],
    [1, 5, 1, 1, 1, 3, 1, 1], [1, 5, 1, 1, 3, 1, 1, 1], [1, 5, 1, 3, 1, 1, 1, 1], [1, 5, 3, 1, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 3], [3, 1, 1, 1, 1, 1, 1, 5], [3, 1, 1, 1, 1, 1, 3, 1],
    [3, 1, 1, 1, 1, 1, 3, 3], [3, 1, 1, 1, 1, 1, 5, 1], [3, 1, 1, 1, 1, 3, 1, 1], [3, 1, 1, 1, 1, 3, 1, 3],
    [3, 1, 1, 1, 1, 3, 3, 1], [3, 1, 1, 1, 1, 3, 3, 3], [3, 1, 1, 1, 1, 5, 1, 1], [3, 1, 1, 1, 3, 1, 1, 1],
    [3, 1, 1, 1, 3, 1, 1, 3], [3, 1, 1, 1, 3, 1, 3, 1], [3, 1, 1, 1, 3, 1, 3, 3], [3, 1, 1, 1, 3, 3, 1, 1],
    [3, 1, 1, 1, 3, 3, 1, 3], [3, 1, 1, 1, 3, 3, 3, 1], [3, 1, 1, 1, 5, 1, 1, 1], [3, 1, 1, 3, 1, 1, 1, 1],
    [3, 1, 1, 3, 1, 1, 1, 3], [3, 1, 1, 3, 1, 1, 3, 1], [3, 1, 1, 3, 1, 1, 3, 3], [3, 1, 1, 3, 1, 3, 1, 1],
    [3, 1, 1, 3, 1, 3, 1, 3], [3, 1, 1, 3, 1, 3, 3, 1], [3, 1, 1, 3, 3, 1, 1, 1], [3, 1, 1, 3, 3, 1, 1, 3],
    [3, 1, 1, 3, 3, 1, 3, 1], [3, 1, 1, 3, 3, 3, 1, 1], [3, 1, 1, 5, 1, 1, 1, 1], [3, 1, 3, 1, 1, 1, 1, 1],
    [3, 1, 3, 1, 1, 1, 1, 3], [3, 1, 3, 1, 1, 1, 3, 1], [3, 1, 3, 1, 1, 1, 3, 3], [3, 1, 3, 1, 1, 3, 1, 1],
    [3, 1, 3, 1, 1, 3, 1, 3], [3, 1, 3, 1, 1, 3, 3, 1], [3, 1, 3, 1, 3, 1, 1, 1], [3, 1, 3, 1, 3, 1, 1, 3],
    [3, 1, 3, 1, 3, 1, 3, 1], [3, 1, 3, 1, 3, 3, 1, 1], [3, 1, 3, 3, 1, 1, 1, 1], [3, 1, 3, 3, 1, 1, 1, 3],
    [3, 1, 3, 3, 1, 1, 3, 1], [3, 1, 3, 3, 1, 3, 1, 1], [3, 1, 3, 3, 3, 1, 1, 1], [3, 1, 5, 1, 1, 1, 1, 1],
    [3, 3, 1, 1, 1, 1, 1, 1], [3, 3, 1, 1, 1, 1, 1, 3], [3, 3, 1, 1, 1, 1, 3, 1], [3, 3, 1, 1, 1, 1, 3, 3],
    [3, 3, 1, 1, 1, 3, 1, 1], [3, 3, 1, 1, 1, 3, 1, 3], [3, 3, 1, 1, 1, 3, 3, 1], [3, 3, 1, 1, 3, 1, 1, 1],
    [3, 3, 1, 1, 3, 1, 1, 3], [3, 3, 1, 1, 3, 1, 3, 1], [3, 3, 1, 1, 3, 3, 1, 1], [3, 3, 1, 3, 1, 1, 1, 1],
    [3, 3, 1, 3, 1, 1, 1, 3], [3, 3, 1, 3, 1, 1, 3, 1], [3, 3, 1, 3, 1, 3, 1, 1], [3, 3, 1, 3, 3, 1, 1, 1],
    [3, 3, 3, 1, 1, 1, 1, 1], [3, 3, 3, 1, 1, 1, 1, 3], [3, 3, 3, 1, 1, 1, 3, 1], [3, 3, 3, 1, 1, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 1, 1], [3, 3, 3, 3, 1, 1, 1, 1], [3, 5, 1, 1, 1, 1, 1, 1], [5, 1, 1, 1, 1, 1, 1, 1],
    [5, 1, 1, 1, 1, 1, 1, 3], [5, 1, 1, 1, 1, 1, 3, 1], [5, 1, 1, 1, 1, 3, 1, 1], [5, 1, 1, 1, 3, 1, 1, 1],
    [5, 1, 1, 3, 1, 1, 1, 1], [5, 1, 3, 1, 1, 1, 1, 1], [5, 3, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 3, 3, 3, 3],
    [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3], [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 3, 1], [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
    [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1], [3, 3, 1, 3, 1, 3, 3, 1],
    [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3], [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1],
    [3, 1, 3, 3, 3, 1, 3, 1], [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
    [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1], [1, 3, 3, 3, 1, 3, 3, 1],
    [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1],
];

/// Parity of `Σ|g|` for each abs row (1 = "odd row", 149 of 256); transcribes
/// `e8p_i4_reference::abs_row_parity`. `Σ2|g|` is always even, so the rule is
/// on `Σ|g| = Σ2|g| / 2`.
pub(crate) const E8P_ROW_PARITY: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut s = 0u32;
        let mut k = 0;
        while k < 8 {
            s += E8P_ABS2[i][k] as u32;
            k += 1;
        }
        t[i] = ((s / 2) & 1) as u8;
        i += 1;
    }
    t
};

/// `4·codeword` of one E8P wire word: eight odd integers in `[-11, 11]`.
///
/// Transcribes `e8p_i4_reference::decode8` over the wire word
/// `idx | sign7 << 8 | shift << 15`: lane `k` is
/// `2·(±E8P_ABS2[idx][k]) + (shift ? -1 : +1)`; the sign of lane `k < 7` is
/// bit `k` of `sign7`, lane 7 is negative iff
/// `popcount(sign7) + E8P_ROW_PARITY[idx]` is odd.
#[inline]
pub fn e8p_decode_word(word: u16) -> [i8; 8] {
    let idx = (word & 0xFF) as usize;
    let sign7 = ((word >> 8) & 0x7F) as u8;
    let off: i8 = if word >> 15 == 1 { -1 } else { 1 };
    let neg7 = (sign7.count_ones() as u8 + E8P_ROW_PARITY[idx]) & 1;
    let a2 = &E8P_ABS2[idx];
    let mut v = [0i8; 8];
    for k in 0..8 {
        let neg = (if k < 7 { (sign7 >> k) & 1 } else { neg7 }) == 1;
        let n = a2[k] as i8;
        v[k] = 2 * (if neg { -n } else { n }) + off;
    }
    v
}

/// One 128-group of a planar format, scale applied: `out[0..128]` are the exact
/// f32 weights. `codes` is the group's code bytes (`q.group_bytes()` of them),
/// `scale_bits` its f16 scale.
///
/// W3: `f32(d) * (2c - 7)`. E8P: `(f32(d) * 0.25) * v` (exact in f32 for every
/// finite f16 `d`).
pub(crate) fn decode_planar_group(q: CpuQuant, codes: &[u8], scale_bits: u16, out: &mut [f32]) {
    debug_assert!(q.is_planar());
    let d = half_f16_to_f32(scale_bits);
    match q {
        CpuQuant::Mq3I4G128R256 | CpuQuant::Mq3I4G128R128 => {
            for (chunk, dst) in codes[..48]
                .chunks_exact(3)
                .zip(out[..128].chunks_exact_mut(8))
            {
                let acc = chunk[0] as u32 | (chunk[1] as u32) << 8 | (chunk[2] as u32) << 16;
                for (j, o) in dst.iter_mut().enumerate() {
                    let c = ((acc >> (3 * j)) & 7) as i32;
                    *o = d * (2 * c - 7) as f32;
                }
            }
        }
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => {
            let scale = d * 0.25;
            for (word, dst) in codes[..32]
                .chunks_exact(2)
                .zip(out[..128].chunks_exact_mut(8))
            {
                let v = e8p_decode_word(u16::from_le_bytes([word[0], word[1]]));
                for (o, &lane) in dst.iter_mut().zip(v.iter()) {
                    *o = scale * lane as f32;
                }
            }
        }
        _ => unreachable!("decode_planar_group({q:?}): not a planar format"),
    }
}

/// Scalar reference decoder of one planar row: `out[0..k]` are the exact f32
/// weights (in the rotated basis, like [`decode_group_codes`]).
///
/// `codes_row` is the row's slice of the codes plane
/// (`crate::gemv::row_bytes(q, k)` bytes), `scales_row` its `k / 128`
/// little-endian f16 scales. Panics for a non-planar `q`, a `k` that is not a
/// multiple of 128, or short slices.
pub fn decode_planar_row(
    q: CpuQuant,
    codes_row: &[u8],
    scales_row: &[u8],
    k: usize,
    out: &mut [f32],
) {
    assert!(
        q.is_planar(),
        "decode_planar_row({q:?}): not a planar format"
    );
    assert!(
        k % PLANAR_GROUP == 0,
        "decode_planar_row({q:?}): k={k} is not a multiple of {PLANAR_GROUP}"
    );
    let groups = k / PLANAR_GROUP;
    let gb = q.group_bytes();
    assert!(
        codes_row.len() >= groups * gb && scales_row.len() >= 2 * groups && out.len() >= k,
        "decode_planar_row({q:?}): codes {} (need {}), scales {} (need {}), out {} (need {k})",
        codes_row.len(),
        groups * gb,
        scales_row.len(),
        2 * groups,
        out.len()
    );
    for g in 0..groups {
        let d = u16::from_le_bytes([scales_row[2 * g], scales_row[2 * g + 1]]);
        decode_planar_group(
            q,
            &codes_row[g * gb..(g + 1) * gb],
            d,
            &mut out[g * PLANAR_GROUP..(g + 1) * PLANAR_GROUP],
        );
    }
}

/// LCG behind the MagnumQuant sign vectors; mirrors
/// `KvCache::gen_fwht_signs` exactly (each step is `s = (s * 1103515245 + 12345)
/// mod 2^31` once the wrapping `u32` arithmetic and the `0x7fffffff` mask are
/// taken together).
pub fn fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn sign_table(seed: u32) -> [f32; FWHT_N] {
    fwht_signs(seed, FWHT_N).try_into().expect("gapless 256")
}

static SIGNS1: LazyLock<[f32; FWHT_N]> = LazyLock::new(|| sign_table(FWHT_SEED1));
static SIGNS2: LazyLock<[f32; FWHT_N]> = LazyLock::new(|| sign_table(FWHT_SEED2));

fn signs1() -> &'static [f32; FWHT_N] {
    &SIGNS1
}

fn signs2() -> &'static [f32; FWHT_N] {
    &SIGNS2
}

/// In-place radix-2 Hadamard butterfly over 256 elements (unnormalized: it
/// scales by 256, which the `1/16` post-scale below cancels into `1/sqrt(256)`).
fn butterfly(group: &mut [f32; FWHT_N]) {
    let mut stride = 1;
    while stride < FWHT_N {
        let mut j = 0;
        while j < FWHT_N {
            for k in 0..stride {
                let a = group[j + k];
                let b = group[j + k + stride];
                group[j + k] = a + b;
                group[j + k + stride] = a - b;
            }
            j += stride * 2;
        }
        stride <<= 1;
    }
}

/// Inverse FWHT-256: the *un*-rotation `dequantize_to_f32` applies to every
/// FWHT-rotated format — pre-scale by `signs2`, butterfly, post-scale by
/// `0.0625 * signs1` (`0.0625 = 1/16 = 1/sqrt(256)`).
///
/// A sign or normalization error here is the "token soup" attractor failure
/// mode, which is why the tests below pin both directions against independent
/// oracles rather than against each other.
pub fn fwht256_inplace(group: &mut [f32; FWHT_N], signs1: &[f32], signs2: &[f32]) {
    for i in 0..FWHT_N {
        group[i] *= signs2[i];
    }
    butterfly(group);
    let scale_inv = 0.0625;
    for i in 0..FWHT_N {
        group[i] *= scale_inv * signs1[i];
    }
}

/// Forward FWHT-256 rotation of an activation, matching the `mq_rotate_x`
/// kernels: pre-scale by `signs1`, butterfly, post-scale by `0.0625 * signs2`.
/// The sign tables appear in mirror order relative to [`fwht256_inplace`], which
/// is what makes the two transpose (and therefore mutually inverse, given the
/// `1/sqrt(256)` normalization).
///
/// `x.len()` must be a multiple of 256.
pub fn rotate_x(x: &mut [f32]) {
    assert!(
        x.len() % FWHT_N == 0,
        "rotate_x: length {} is not a multiple of {FWHT_N}",
        x.len()
    );
    for block in x.chunks_exact_mut(FWHT_N) {
        let group: &mut [f32; FWHT_N] = block.try_into().expect("chunk is 256");
        for i in 0..FWHT_N {
            group[i] *= signs1()[i];
        }
        butterfly(group);
        let scale = 0.0625;
        for i in 0..FWHT_N {
            group[i] *= scale * signs2()[i];
        }
    }
}

/// Forward FWHT-128 sign tables of the qt 53 activation rotation: the seeds
/// 43 / 1043 that `Gpu::ensure_mq_signs_128` uploads and the `MQ4G128V2`
/// encoder bakes into the weights.
const FWHT128_N: usize = 128;
const FWHT128_SEED1: u32 = 43;
const FWHT128_SEED2: u32 = 1043;
/// `1/sqrt(128)` as the `f32` literal `mq_rotate_x_128_v2.hip` uses.
const FWHT128_SCALE: f32 = 0.0883883476;

static SIGNS128_1: LazyLock<[f32; FWHT128_N]> = LazyLock::new(|| {
    fwht_signs(FWHT128_SEED1, FWHT128_N)
        .try_into()
        .expect("gapless 128")
});
static SIGNS128_2: LazyLock<[f32; FWHT128_N]> = LazyLock::new(|| {
    fwht_signs(FWHT128_SEED2, FWHT128_N)
        .try_into()
        .expect("gapless 128")
});

/// In-place radix-2 Hadamard butterfly over 128 elements, unnormalized.
fn butterfly128(group: &mut [f32; FWHT128_N]) {
    let mut stride = 1;
    while stride < FWHT128_N {
        let mut j = 0;
        while j < FWHT128_N {
            for k in 0..stride {
                let a = group[j + k];
                let b = group[j + k + stride];
                group[j + k] = a + b;
                group[j + k + stride] = a - b;
            }
            j += stride * 2;
        }
        stride <<= 1;
    }
}

/// Inverse FWHT-128 of one qt 53 group: pre-scale by `signs2`, butterfly,
/// post-scale by `1/sqrt(128) * signs1` — the mirror of [`rotate_x_128_v2`]'s
/// per-group transform, the same way [`fwht256_inplace`] mirrors [`rotate_x`].
pub fn fwht128_inplace(group: &mut [f32; FWHT128_N], signs1: &[f32], signs2: &[f32]) {
    for i in 0..FWHT128_N {
        group[i] *= signs2[i];
    }
    butterfly128(group);
    for i in 0..FWHT128_N {
        group[i] *= FWHT128_SCALE * signs1[i];
    }
}

/// Forward FWHT-128 rotation of an activation, exactly what the GPU
/// `mq_rotate_x_128_v2` kernel does to the qt 53 down projection's input, per
/// 128-element group: `x * signs1` (seed 43), radix-2 butterfly with strides
/// 1, 2, 4, …, `* 1/sqrt(128) * signs2` (seed 1043). The butterfly stage order
/// and operand order match the kernel's, so the result is bitwise the
/// kernel's.
///
/// Unlike [`rotate_x`], `x.len()` need not be a multiple of 128: the last
/// group is zero-padded for the transform and only the first `x.len() % 128`
/// outputs are kept (the kernel's tail contract, which the encoder mirrors by
/// zero-padding the weight row).
pub fn rotate_x_128_v2(x: &mut [f32]) {
    let (s1, s2) = (&*SIGNS128_1, &*SIGNS128_2);
    for block in x.chunks_mut(FWHT128_N) {
        let mut group = [0.0f32; FWHT128_N];
        for (i, v) in block.iter().enumerate() {
            group[i] = *v * s1[i];
        }
        butterfly128(&mut group);
        for (i, v) in block.iter_mut().enumerate() {
            *v = group[i] * FWHT128_SCALE * s2[i];
        }
    }
}

/// Forward-rotate `x` in place with the rotation `q`'s stored codes expect
/// ([`CpuQuant::input_rotation`]): [`rotate_x`] for the FWHT-256 formats,
/// [`rotate_x_128_v2`] for qt 53, nothing for the natural-basis formats.
pub fn rotate_input(q: CpuQuant, x: &mut [f32]) {
    match q.input_rotation() {
        InputRotation::None => {}
        InputRotation::FwhtG256 => rotate_x(x),
        InputRotation::FwhtG128V2 => rotate_x_128_v2(x),
    }
}

/// Divide an activation by a per-input-channel AWQ scale:
/// `x[i] /= scale[i]`, indexed in the **unrotated** basis.
///
/// This is the op the `rotate_x_mq_awq` / `fused_rmsnorm_mq_rotate_awq` /
/// `fused_silu_mul_rotate_mq_awq` kernels fold in ahead of the FWHT: the
/// quantizer pre-scaled those weights by `s`, so `(W·s) · (x/s) = W·x` only
/// holds if the activation is divided *before* it is rotated. Omitting it does
/// not fail — it silently computes `(W·s)·x`, a per-channel scale error on
/// every projection that carries a sidecar (`DType::supports_awq_sidecar`),
/// which is the "token soup" failure mode with no error to point at.
///
/// `scale` is the loader's already-widened f32 sidecar (`load_awq_scale_for`
/// converts the on-disk f16 to f32 on the host), one value per input channel.
pub fn divide_by_awq_scale(x: &mut [f32], scale: &[f32]) {
    assert!(
        scale.len() >= x.len(),
        "awq scale has {} channels, activation has {}",
        scale.len(),
        x.len()
    );
    for (v, s) in x.iter_mut().zip(scale) {
        *v /= *s;
    }
}

/// `(scale, zero)` from an f32 header (flat HFQ/MQ families, 8 B).
#[inline]
fn f32_hdr(packed: &[u8], off: usize) -> (f32, f32) {
    (f32_at(packed, off), f32_at(packed, off + 4))
}

/// The two per-128 halves of a V2 group's `[s0 z0 s1 z1]` fp16 header.
///
/// Widened **once per group**, not per element: `half_f16_to_f32` is a branchy
/// widening, and calling it twice per weight made the V2 family
/// conversion-bound (measured on `qwen3.8-27b.mq3-xt`: 19.9 ms/step at
/// m=12288 k=5120, i.e. ~3 GMAC/s, versus ~30 GB/s of weight streaming).
#[inline]
fn v2_halves(packed: &[u8], off: usize) -> ((f32, f32), (f32, f32)) {
    (
        (f16_at(packed, off), f16_at(packed, off + 2)),
        (f16_at(packed, off + 4), f16_at(packed, off + 6)),
    )
}

#[inline]
fn half_of(i: usize, halves: ((f32, f32), (f32, f32))) -> (f32, f32) {
    if i < 128 {
        halves.0
    } else {
        halves.1
    }
}

/// 2-bit codes, LSB-first, 4 per byte.
#[inline]
fn code2(packed: &[u8], off: usize, i: usize) -> f32 {
    ((packed[off + i / 4] >> (2 * (i % 4))) & 0x3) as f32
}

/// 3-bit codes, 8 per 3 bytes, the same cross-byte packing as HFQ3/MQ3-G256
/// (written out here so a V2 group can use it without a second scratch pass).
#[inline]
fn code3(packed: &[u8], off: usize, chunk: usize) -> [u32; 8] {
    let bo = off + chunk * 3;
    let b0 = packed[bo] as u32;
    let b1 = packed[bo + 1] as u32;
    let b2 = packed[bo + 2] as u32;
    [
        b0 & 7,
        (b0 >> 3) & 7,
        ((b0 >> 6) | (b1 << 2)) & 7,
        (b1 >> 1) & 7,
        (b1 >> 4) & 7,
        ((b1 >> 7) | (b2 << 1)) & 7,
        (b2 >> 2) & 7,
        (b2 >> 5) & 7,
    ]
}

/// 4-bit codes, 2 per byte, low nibble first.
#[inline]
fn code4(packed: &[u8], off: usize, i: usize) -> f32 {
    let b = packed[off + i / 2];
    (if i % 2 == 0 { b & 0xF } else { b >> 4 }) as f32
}

/// 5-bit codes, 8 per 5 bytes — the packing the `gemv_mq*g256v2` kernels use
/// (`DOG5` / `TAIL_DOG5`): the 40 bits run low-to-high across the five bytes.
#[inline]
fn code5(packed: &[u8], off: usize, i: usize) -> f32 {
    let bo = off + (i / 8) * 5;
    let b = &packed[bo..bo + 5];
    let (b0, b1, b2, b3, b4) = (
        b[0] as u32,
        b[1] as u32,
        b[2] as u32,
        b[3] as u32,
        b[4] as u32,
    );
    let codes = [
        b0 & 0x1f,
        ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3),
        (b1 >> 2) & 0x1f,
        ((b1 >> 7) & 0x01) | ((b2 & 0x0f) << 1),
        ((b2 >> 4) & 0x0f) | ((b3 & 0x01) << 4),
        (b3 >> 1) & 0x1f,
        ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2),
        (b4 >> 3) & 0x1f,
    ];
    codes[i % 8] as f32
}

/// 6-bit codes, 4 per 3 bytes (the packing HFQ6/MQ6-G256 use).
#[inline]
fn code6(packed: &[u8], off: usize, i: usize) -> f32 {
    let bo = off + (i / 4) * 3;
    let b0 = packed[bo] as u32;
    let b1 = packed[bo + 1] as u32;
    let b2 = packed[bo + 2] as u32;
    let codes = [
        b0 & 0x3F,
        ((b0 >> 6) | (b1 << 2)) & 0x3F,
        ((b1 >> 4) | (b2 << 4)) & 0x3F,
        (b2 >> 2) & 0x3F,
    ];
    codes[i % 4] as f32
}

#[inline]
fn f16_at(bytes: &[u8], off: usize) -> f32 {
    half_f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

#[inline]
fn f32_at(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// `half::f16::from_bits(bits).to_f32()`, inlined so this crate keeps its
/// dependency list empty of the GPU stack *and* of formatting crates.
/// `f16_widening_is_ieee` holds it against `half` for every encoding class.
#[inline]
pub(crate) fn half_f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let man = (bits & 0x3FF) as u32;
    let out = match exp {
        0 => {
            if man == 0 {
                sign << 31
            } else {
                // Subnormal: `man * 2^-24`, which is exact in f32 (10 significand
                // bits at an exponent no smaller than `2^-24`).
                f32::to_bits(man as f32 * f32::from_bits(0x3380_0000)) | (sign << 31)
            }
        }
        0x1F => (sign << 31) | 0x7F80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// E8's unsigned E4M3 block scale, including its saturated 0x7f code.
/// Powers of two and eighths are exact, matching the device decoder's bits.
#[inline]
pub(crate) fn mfp4_e8_scale(code: u8) -> f32 {
    let exp = (code >> 3) & 15;
    let mant = code & 7;
    if exp == 0 {
        mant as f32 / 512.0
    } else if exp == 15 && mant == 7 {
        448.0
    } else {
        f32::from_bits((exp as u32 + 120) << 23) * (1.0 + mant as f32 / 8.0)
    }
}

/// Decode eight AoS blocks without copying the shared row header.
pub(crate) fn decode_mfp4_e8_group(blocks: &[u8], row_scale: f32, out: &mut [f32]) {
    for block in 0..8 {
        let at = block * 17;
        // Keep the format's scale multiplication order; do not fold QUANT_STEP
        // into row_scale or defer scaling until after the dot product.
        let scale = row_scale * mfp4_e8_scale(blocks[at]) * 0.88;
        for word in 0..4 {
            let offset = at + 1 + word * 4;
            let index = u32::from_le_bytes(blocks[offset..offset + 4].try_into().unwrap());
            let coset = (index >> 31) as f32 * 0.5;
            let base = block * 32 + word * 8;
            let mut parity = 0;
            for coord in 0..7 {
                let nibble = (index >> (coord * 4)) & 15;
                parity ^= nibble;
                out[base + coord] = scale * ((nibble as i32 - 7) as f32 + coset);
            }
            let last = ((index >> 27) & 14) | (parity & 1);
            out[base + 7] = scale * ((last as i32 - 7) as f32 + coset);
        }
    }
}

/// Decode one group's *code values* into `out[0..q.group_elems()]`: the affine
/// or codebook decode, with no rotation. These are the values the MQ kernels dot
/// against a forward-rotated activation (see the module docs).
/// For E8, `packed` includes the row header followed by this group's blocks.
pub fn decode_group_codes(q: CpuQuant, packed: &[u8], out: &mut [f32]) {
    assert!(
        !q.is_planar(),
        "decode_group_codes({q:?}): planar format (codes and scales are separate planes); \
         use decode_planar_row"
    );
    let ge = q.group_elems();
    assert!(
        out.len() >= ge,
        "decode_group_codes({q:?}): out has {} elements, need {ge}",
        out.len()
    );
    assert!(
        packed.len() >= q.row_header_bytes() + q.group_bytes(),
        "decode_group_codes({q:?}): packed has {} bytes, need {}",
        packed.len(),
        q.row_header_bytes() + q.group_bytes()
    );
    match q {
        CpuQuant::Mq3I4G128R256
        | CpuQuant::Mq3I4G128R128
        | CpuQuant::E8pI4G128R256
        | CpuQuant::E8pI4G128R128 => {
            unreachable!("decode_group_codes({q:?}): planar format, rejected above")
        }
        CpuQuant::Mfp4G32E8 => {
            decode_mfp4_e8_group(&packed[16..], f16_at(packed, 0), out);
        }
        CpuQuant::F16 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f16_at(packed, 2 * i);
            }
        }
        CpuQuant::F32 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f32_at(packed, 4 * i);
            }
        }
        CpuQuant::Bf16 => {
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = f32::from_bits((f16_at_u16(packed, 2 * i) as u32) << 16);
            }
        }
        CpuQuant::Q8F16 => {
            let scale = f16_at(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = packed[2 + i] as i8 as f32 * scale;
            }
        }
        CpuQuant::Mq4G256 | CpuQuant::Hfq4G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let byte = packed[8 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                *o = scale * nibble as f32 + zero;
            }
        }
        CpuQuant::Mq4G256V2 => {
            let halves = v2_halves(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let (scale, zero) = half_of(i, halves);
                let byte = packed[8 + i / 2];
                let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                *o = scale * nibble as f32 + zero;
            }
        }
        CpuQuant::Mq4G128V2 => {
            // fp16 `[scale][zero]`, then 64 B of nibbles (low nibble = even
            // element), the `decode_mq4g128v2` / GPU down-kernel layout.
            let scale = f16_at(packed, 0);
            let zero = f16_at(packed, 2);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = scale * code4(packed, 4, i) + zero;
            }
        }
        CpuQuant::Mq6G256 | CpuQuant::Hfq6G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            let mut i = 0;
            while i < ge {
                let bo = 8 + (i / 4) * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                out[i] = scale * ((b0 & 0x3F) as f32) + zero;
                out[i + 1] = scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero;
                out[i + 2] = scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero;
                out[i + 3] = scale * (((b2 >> 2) & 0x3F) as f32) + zero;
                i += 4;
            }
        }
        CpuQuant::Mq3G256 => {
            let scale = f32_at(packed, 0);
            let zero = f32_at(packed, 4);
            for chunk in 0..32 {
                let bo = 8 + chunk * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                let base = chunk * 8;
                let codes = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) | (b1 << 2)) & 7,
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) | (b2 << 1)) & 7,
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for (k, code) in codes.iter().enumerate() {
                    out[base + k] = scale * *code as f32 + zero;
                }
            }
        }
        CpuQuant::Hfq3G256 => {
            // qt 11: f32 header + 96 B of 3-bit codes, natural basis.
            let (scale, zero) = f32_hdr(packed, 0);
            for chunk in 0..32 {
                for (k, code) in code3(packed, 8, chunk).iter().enumerate() {
                    out[chunk * 8 + k] = scale * *code as f32 + zero;
                }
            }
        }
        CpuQuant::Hfq2G256 | CpuQuant::Mq2G256 | CpuQuant::Mq2G256V2 => {
            // qt 9 flat and qt 18 / qt 50 FWHT-rotated: 2-bit codes, 4 per byte.
            let halves = v2_halves(packed, 0);
            let flat = f32_hdr(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let (scale, zero) = match q {
                    CpuQuant::Mq2G256V2 => half_of(i, halves),
                    _ => flat,
                };
                *o = scale * code2(packed, 8, i) + zero;
            }
        }
        CpuQuant::Hfq4G128 => {
            let (scale, zero) = f32_hdr(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = scale * code4(packed, 8, i) + zero;
            }
        }
        CpuQuant::Hfq3G128 => {
            let (scale, zero) = f32_hdr(packed, 0);
            for chunk in 0..16 {
                for (k, code) in code3(packed, 8, chunk).iter().enumerate() {
                    out[chunk * 8 + k] = scale * *code as f32 + zero;
                }
            }
        }
        CpuQuant::Hfq2G128 => {
            let (scale, zero) = f32_hdr(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = scale * code2(packed, 8, i) + zero;
            }
        }
        CpuQuant::Tq2G128 => {
            // fp16 `d` + 32 B of 2-bit codes; `(code - 1) * d` (ternary).
            let d = f16_at(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = (code2(packed, 2, i) - 1.0) * d;
            }
        }
        CpuQuant::Bq1G128 => {
            // fp16 `d` + 16 B of sign bits, LSB-first; `bit ? +d : -d`.
            let d = f16_at(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let bit = (packed[2 + i / 8] >> (i % 8)) & 1;
                *o = if bit == 1 { d } else { -d };
            }
        }
        CpuQuant::Mq5G256 | CpuQuant::Mq5G256V2 => {
            // 5-bit codes, 8 per 5 bytes; f32 header (flat) or fp16 halves (V2).
            let halves = v2_halves(packed, 0);
            let flat = f32_hdr(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let (scale, zero) = match q {
                    CpuQuant::Mq5G256V2 => half_of(i, halves),
                    _ => flat,
                };
                *o = scale * code5(packed, 8, i) + zero;
            }
        }
        CpuQuant::Mq6G256V2 => {
            let halves = v2_halves(packed, 0);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                let (scale, zero) = half_of(i, halves);
                *o = scale * code6(packed, 8, i) + zero;
            }
        }
        CpuQuant::Mq4CG256 => {
            // fp16 `[scale][zero]` (widened once), 4 B of padding, then nibbles.
            let scale = f16_at(packed, 0);
            let zero = f16_at(packed, 2);
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = scale * code4(packed, 8, i) + zero;
            }
        }
        CpuQuant::Mq3G256V2 => {
            let halves = v2_halves(packed, 0);
            for chunk in 0..32 {
                for (k, code) in code3(packed, 8, chunk).iter().enumerate() {
                    let i = chunk * 8 + k;
                    let (scale, zero) = half_of(i, halves);
                    out[i] = scale * *code as f32 + zero;
                }
            }
        }
        CpuQuant::Mq2G256Lloyd | CpuQuant::Mq2G256LloydU => {
            // 4-entry fp16 codebook + 64 B of 2-bit indices.
            let mut cb = [0.0f32; 4];
            for (k, c) in cb.iter_mut().enumerate() {
                *c = f16_at(packed, 2 * k);
            }
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = cb[code2(packed, 8, i) as usize];
            }
        }
        CpuQuant::Mq4G256Lloyd => {
            // 16-entry fp16 codebook + 128 B of nibbles.
            let mut cb = [0.0f32; 16];
            for (k, c) in cb.iter_mut().enumerate() {
                *c = f16_at(packed, 2 * k);
            }
            for (i, o) in out[..ge].iter_mut().enumerate() {
                *o = cb[code4(packed, 32, i) as usize];
            }
        }
        CpuQuant::Mq3G256Lloyd => {
            // 16 B codebook (8 × fp16, ascending at quant time) then 96 B of
            // 3-bit indices in the same cross-byte packing as Mq3G256.
            let mut cb = [0.0f32; 8];
            for (k, c) in cb.iter_mut().enumerate() {
                *c = f16_at(packed, 2 * k);
            }
            for chunk in 0..32 {
                let bo = 16 + chunk * 3;
                let b0 = packed[bo] as u32;
                let b1 = packed[bo + 1] as u32;
                let b2 = packed[bo + 2] as u32;
                let base = chunk * 8;
                let codes = [
                    b0 & 7,
                    (b0 >> 3) & 7,
                    ((b0 >> 6) | (b1 << 2)) & 7,
                    (b1 >> 1) & 7,
                    (b1 >> 4) & 7,
                    ((b1 >> 7) | (b2 << 1)) & 7,
                    (b2 >> 2) & 7,
                    (b2 >> 5) & 7,
                ];
                for (k, code) in codes.iter().enumerate() {
                    out[base + k] = cb[*code as usize];
                }
            }
        }
    }
}

#[inline]
fn f16_at_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([bytes[off], bytes[off + 1]])
}

/// Decode one group of `q.group_elems()` weights into `out`, in the **original
/// basis**: [`decode_group_codes`] followed by the inverse FWHT for the rotated
/// formats. Value-for-value the arithmetic of `dequantize_to_f32`.
///
/// The CPU GEMV path does *not* use this (it keeps the codes and rotates the
/// activation — see the module docs); it exists for the cross-check against the
/// canonical decoder and for any future host-side weight materialization.
pub fn dequant_group(q: CpuQuant, packed: &[u8], out: &mut [f32]) {
    assert!(
        !q.is_planar(),
        "dequant_group({q:?}): planar format has no per-group bytes; use decode_planar_row"
    );
    decode_group_codes(q, packed, out);
    if q.is_fwht_g256() {
        let group: &mut [f32; FWHT_N] = (&mut out[..FWHT_N]).try_into().expect("group is 256");
        fwht256_inplace(group, signs1(), signs2());
    } else if q.is_fwht_g128() {
        let group: &mut [f32; FWHT128_N] =
            (&mut out[..FWHT128_N]).try_into().expect("group is 128");
        fwht128_inplace(group, &*SIGNS128_1, &*SIGNS128_2);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// The single-group fixture the expectation tables below were generated
    /// over (`testfix::group_bytes(q, 0)`).
    fn fixture(q: CpuQuant) -> Vec<u8> {
        crate::testfix::group_bytes(q, 0)
    }

    /// The f16 bit patterns every implementation in this workspace must agree on,
    /// computed by hand as shortest-round-trip literals in
    /// `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs`'s generator.
    const EXPECTED_F16_BITS: [u16; 10] = [
        0x0000, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x4200, 0xc200, 0x4500, 0x2c00, 0x3a00,
    ];

    // ── expectation tables, generated from the canonical decoder ────────────
    // Recipe (a throwaway example in `hipfire-runtime`, deleted after use): for
    // each format, run
    // `weight_backend::dequantize_weight_to_f32(qt, &testfix::group_bytes(q, 0), q.group_elems())`
    // and print `v.to_bits()` as `0x%08x`. Bit patterns rather than decimal
    // literals, so the comparison is exact and immune to float-literal
    // round-tripping. Only formats with a canonical `dequantize_to_f32` arm have a
    // table: qt 45 / 47 / 48 / 49 / 50 and qt 9 / 10 / 31 have no arm there, so
    // their oracle is the GPU launcher (`gpu_gemv_parity`), not this table.
    /// qt 13, 136 B / 256 elems.
    const EXPECTED_MQ4G256: [u32; 256] = [
        0x3df00000, 0x3e200000, 0xbd800000, 0xbe280000, 0xbe840000, 0xbed00000, 0xbd400000,
        0x3eb40000, 0xbe080000, 0xbe800000, 0x3ee80000, 0x3ebc0000, 0xbe840000, 0x3d800000,
        0xbdc00000, 0x3cc00000, 0x3ee40000, 0xbe100000, 0x3e500000, 0xbe180000, 0x3e580000,
        0x3ed80000, 0x3e000000, 0x3f160000, 0xbc000000, 0xbe880000, 0xbf280000, 0x3db00000,
        0x3e380000, 0xbda00000, 0x3e500000, 0x3f0a0000, 0xbc800000, 0xbdd00000, 0x3f120000,
        0x3e500000, 0x3e980000, 0xbe680000, 0x3e580000, 0xbe400000, 0xbd400000, 0x3c000000,
        0x3e680000, 0x80000000, 0x3d400000, 0x3e280000, 0xbd900000, 0x3de00000, 0xbd400000,
        0x3e840000, 0xbef40000, 0xbe700000, 0xbc800000, 0xbe840000, 0x3e780000, 0xbe600000,
        0x3c800000, 0xbeac0000, 0x3d900000, 0x3eb00000, 0x3d400000, 0xbf060000, 0x3c000000,
        0x3e980000, 0xbe600000, 0x3e680000, 0xbd600000, 0x3ea00000, 0x3e000000, 0x3cc00000,
        0x3e480000, 0x3f140000, 0xbe100000, 0x3df00000, 0xbed40000, 0xbe200000, 0x3e700000,
        0xbdb00000, 0x3e780000, 0xbea80000, 0x3ef00000, 0x3d600000, 0xbd900000, 0x3f080000,
        0x3dc00000, 0xbe480000, 0x3f020000, 0xbea80000, 0x3e880000, 0xbefc0000, 0x3e480000,
        0xbe800000, 0x3f140000, 0x3e840000, 0xbdf00000, 0xbeb80000, 0xbe940000, 0x3ee00000,
        0x3d800000, 0x3e940000, 0x3c000000, 0x3e900000, 0xbed80000, 0x3e840000, 0x3e580000,
        0xbec80000, 0xbf900000, 0x3f360000, 0x3e480000, 0xbe500000, 0x3e980000, 0x3df00000,
        0xbefc0000, 0xbed80000, 0x3de00000, 0x3eac0000, 0xbdf00000, 0xbe500000, 0xbeb00000,
        0x3df00000, 0xbf4e0000, 0xbe200000, 0xbeb80000, 0xbe080000, 0x3f4a0000, 0x3eb00000,
        0x3f100000, 0x3f5e0000, 0xbf1a0000, 0x3ea80000, 0x3da00000, 0x3e840000, 0xbc000000,
        0xbe900000, 0x3ec80000, 0x3e9c0000, 0x3e380000, 0xbe100000, 0x3d800000, 0x3cc00000,
        0xbe9c0000, 0x3dc00000, 0x3ef00000, 0xbedc0000, 0xbe9c0000, 0x3e100000, 0xbe700000,
        0x3e840000, 0xbec40000, 0x3e400000, 0xbde00000, 0x3e180000, 0x3dd00000, 0xbe700000,
        0xbe900000, 0x3ecc0000, 0xbdb00000, 0xbd800000, 0xbe200000, 0x3e9c0000, 0x3ea00000,
        0xbd900000, 0x3e8c0000, 0xbd000000, 0xbc800000, 0xbeac0000, 0x3e480000, 0x3eb00000,
        0xbd800000, 0x3e380000, 0x3d900000, 0x3d400000, 0xbee80000, 0x3e940000, 0x3f260000,
        0xbe700000, 0xbf0c0000, 0x3ebc0000, 0xbd900000, 0x3e980000, 0x3e000000, 0xbd600000,
        0x3db00000, 0xbd400000, 0x3e300000, 0xbe480000, 0x3c000000, 0x3e900000, 0xbe000000,
        0xbe180000, 0x3df00000, 0x3dc00000, 0x3e880000, 0xbeac0000, 0x3dd00000, 0x3eb80000,
        0x3e200000, 0x3e480000, 0xbd200000, 0xbea80000, 0xbd000000, 0x3e780000, 0x3e8c0000,
        0x3e500000, 0xbe100000, 0xbee40000, 0x3cc00000, 0xbe300000, 0x3d000000, 0xbecc0000,
        0x3e180000, 0xbdc00000, 0x3ec80000, 0x3e8c0000, 0x3d200000, 0x3dc00000, 0x3da00000,
        0xbee40000, 0x3f120000, 0x3d800000, 0x3f100000, 0xbe480000, 0x3dd00000, 0x3e900000,
        0xbe840000, 0xbef00000, 0x3eb00000, 0xbedc0000, 0xbedc0000, 0x3e980000, 0x3e900000,
        0xbd600000, 0xbd600000, 0x3ec80000, 0xbe000000, 0x3ea40000, 0xbc000000, 0x3e400000,
        0xbe600000, 0x3e8c0000, 0x3dd00000, 0xbd800000, 0x3e400000, 0x3db00000, 0x3c000000,
        0x3de00000, 0x3f200000, 0xbe380000, 0xbea40000, 0x3ed80000, 0xbd000000, 0x3e080000,
        0xbd900000, 0x3f100000, 0xbf280000, 0x3db00000,
    ];
    /// qt 44, 136 B / 256 elems.
    const EXPECTED_MQ4G256V2: [u32; 256] = [
        0x40640000, 0xc0c48000, 0x408b8000, 0x40000000, 0x3f9a0000, 0xc0870000, 0xc0320000,
        0x3fea0000, 0xbf280000, 0x40878000, 0xc0190000, 0xc0180000, 0x40350000, 0x3ea00000,
        0x403e0000, 0xc01d0000, 0x3f2c0000, 0x3f580000, 0x3ec00000, 0x3ffa0000, 0xc09a0000,
        0xc0630000, 0xc0ca8000, 0xbfac0000, 0xc01d0000, 0xc0bb0000, 0x40120000, 0x40390000,
        0xc0b40000, 0x3f9a0000, 0xc0a38000, 0xbff00000, 0xc0f98000, 0x40040000, 0x3f980000,
        0xbf9e0000, 0xc0a80000, 0x40fa8000, 0xc0010000, 0x3ec00000, 0xc05d0000, 0xbef00000,
        0xbf900000, 0xbfce0000, 0x408b0000, 0x40858000, 0x410ac000, 0xc0140000, 0x3f280000,
        0xc0190000, 0x40d98000, 0x40cc0000, 0xc0130000, 0x40060000, 0x3fd80000, 0x3fea0000,
        0x40d30000, 0x3fa20000, 0xc0210000, 0xc03e0000, 0x40310000, 0xbdc00000, 0xc01a0000,
        0xbfd20000, 0x3fc60000, 0x3d000000, 0xbf600000, 0xc0090000, 0xc0ee0000, 0x3f5c0000,
        0xbfe60000, 0xc0970000, 0xc0250000, 0x40810000, 0xbe200000, 0x3fee0000, 0xc0300000,
        0xbf4c0000, 0xc1004000, 0x40600000, 0xbd000000, 0x3ed80000, 0xbf440000, 0xc0260000,
        0xc06d0000, 0x00000000, 0xc0580000, 0xc0410000, 0xc0000000, 0x408f8000, 0x40898000,
        0xbf080000, 0xc02d0000, 0xbfb40000, 0xc0500000, 0x40e08000, 0xc06d0000, 0xbfe00000,
        0xbfa80000, 0xc0c18000, 0x408b0000, 0x40330000, 0x3fda0000, 0xc0890000, 0x3f7c0000,
        0xc06c0000, 0x41518000, 0xbfe20000, 0x402e0000, 0xbe700000, 0x3fbe0000, 0xbf000000,
        0x40ca0000, 0x4108c000, 0x40410000, 0x3f080000, 0xbe500000, 0x403e0000, 0xbfa00000,
        0xbfbe0000, 0xbf780000, 0x402f0000, 0x3c800000, 0xbfb40000, 0xc07f0000, 0x40080000,
        0x401c0000, 0xc0848000, 0x40800000, 0xc0b98000, 0xc08a8000, 0xc00c0000, 0xbfae0000,
        0x408b0000, 0xc02a0000, 0x40070000, 0xbf480000, 0x409a8000, 0x40130000, 0xc0140000,
        0x405f0000, 0x00000000, 0xc0560000, 0xbfee0000, 0xbec80000, 0xbf9c0000, 0xbd800000,
        0xc01f0000, 0xc0900000, 0x40690000, 0x40a78000, 0xc00e0000, 0xc0270000, 0x40b30000,
        0xc02a0000, 0xc0430000, 0x40ae0000, 0xbf3c0000, 0xc0a28000, 0x3f700000, 0xc0e48000,
        0x3fe00000, 0xbfb00000, 0x3fca0000, 0x40a20000, 0x40e78000, 0xc01b0000, 0xbd800000,
        0x40370000, 0x3dc00000, 0xbf300000, 0x3faa0000, 0x408f0000, 0xc0988000, 0xc1044000,
        0xbff00000, 0x3f480000, 0xc03b0000, 0xc0cc8000, 0x40b20000, 0x40090000, 0xbf9c0000,
        0x3fb00000, 0xbffe0000, 0xc0c70000, 0x3f960000, 0x40430000, 0xc0160000, 0xc00b0000,
        0x3e900000, 0x3fc40000, 0x3fd60000, 0x3fb20000, 0x3ed00000, 0xbf100000, 0xc01b0000,
        0xc0f80000, 0x3ee80000, 0xc0110000, 0x409b0000, 0xc0170000, 0x407a0000, 0x3dc00000,
        0x3f8a0000, 0x400c0000, 0xbf920000, 0x4102c000, 0x40740000, 0xbef00000, 0x3f240000,
        0x3f0c0000, 0x402e0000, 0xc0770000, 0xbf700000, 0xc04c0000, 0xc03b0000, 0x3fc80000,
        0x40928000, 0xc0a08000, 0xbe900000, 0xc0470000, 0x40120000, 0xc0640000, 0xc0d98000,
        0x40570000, 0xbf700000, 0x3f800000, 0x40d08000, 0x40870000, 0xc0610000, 0xc0170000,
        0xc05a0000, 0xbf2c0000, 0xc0380000, 0x414d8000, 0xbfb60000, 0xc02e0000, 0xbf0c0000,
        0x3ffa0000, 0x3e400000, 0x40e40000, 0x40f88000, 0x407b0000, 0xbeb00000, 0xbec80000,
        0x40060000, 0xbf500000, 0xbfda0000, 0x3e600000, 0xc0550000, 0x3e980000, 0x3f280000,
        0xc0868000, 0x3fb80000, 0x40280000, 0x40430000,
    ];
    /// qt 15, 200 B / 256 elems.
    const EXPECTED_MQ6G256: [u32; 256] = [
        0xbd480000, 0xbdd80000, 0x3dcc0000, 0xbe100000, 0xbe100000, 0xbd780000, 0xbe0c0000,
        0xbe020000, 0x3e500000, 0x3e420000, 0xbee80000, 0x3d680000, 0x3e740000, 0x3d280000,
        0xbd800000, 0xbcb00000, 0xbd780000, 0x3df80000, 0x3e420000, 0xbe500000, 0xbe3c0000,
        0xbead0000, 0xbe180000, 0xbdd40000, 0x3d880000, 0xbd9c0000, 0x3d980000, 0x3cb00000,
        0xbc000000, 0x3dcc0000, 0xbd700000, 0xbe1e0000, 0xbe3e0000, 0xbdc00000, 0x3d8c0000,
        0x3e820000, 0xbe580000, 0x3eab0000, 0xbd300000, 0x3e2a0000, 0xbd600000, 0xbe6a0000,
        0xbdf00000, 0x3dd40000, 0xbc800000, 0x3e520000, 0x3e9a0000, 0xbe3e0000, 0x3c900000,
        0x3d000000, 0x3e9f0000, 0x3e9e0000, 0xbdf80000, 0xbb000000, 0x3e780000, 0x3eb10000,
        0x3c800000, 0xbdbc0000, 0xbea80000, 0xbe830000, 0x3ed00000, 0xbdc40000, 0xbe7c0000,
        0xbdcc0000, 0xbc000000, 0xbe060000, 0xbd100000, 0xbe020000, 0xbe4a0000, 0x3e5c0000,
        0x3bc00000, 0xbe380000, 0xbd700000, 0x3e260000, 0xbd900000, 0x3e7a0000, 0xbe140000,
        0xbe1a0000, 0x3ca00000, 0x3dfc0000, 0xbea60000, 0xbee10000, 0x3e980000, 0xbe020000,
        0x3dcc0000, 0x3e1c0000, 0x3c200000, 0xbdf00000, 0xbd000000, 0x3e0a0000, 0xbeaa0000,
        0x3d280000, 0xbc000000, 0xbde40000, 0x3ec80000, 0x3e5e0000, 0xbe4c0000, 0x3d940000,
        0xbeb40000, 0xbe830000, 0x3e360000, 0xbd300000, 0xbddc0000, 0xbea80000, 0xbe240000,
        0xbd280000, 0x3f080000, 0xbc900000, 0xbe800000, 0x3ecd0000, 0xbd900000, 0xbcd00000,
        0x3ed00000, 0x3ec90000, 0xbdd80000, 0x3e1a0000, 0xbe5a0000, 0x3ea20000, 0xbcf00000,
        0x3ea80000, 0x3e240000, 0x3d380000, 0x3e480000, 0xbd940000, 0xbe980000, 0xbb000000,
        0xbe300000, 0xbf248000, 0x3e7a0000, 0xbcc00000, 0xbe2a0000, 0xbe4c0000, 0xbdfc0000,
        0x3e8c0000, 0xbeed0000, 0x3c000000, 0xbe440000, 0x3e1e0000, 0xbe2c0000, 0xbd940000,
        0xbd080000, 0x3c800000, 0xbecb0000, 0x3e800000, 0xbdb40000, 0xbca00000, 0x3e0e0000,
        0xbe300000, 0x3da40000, 0xbd980000, 0xbd940000, 0xbe640000, 0x3d700000, 0x3e1e0000,
        0x3eba0000, 0xbe970000, 0x3dac0000, 0xbde80000, 0x3d8c0000, 0xbde80000, 0xbdac0000,
        0x3e300000, 0xbe2e0000, 0xbdc80000, 0x3e320000, 0x3d980000, 0x3c900000, 0xbe740000,
        0x3e080000, 0xbed30000, 0xbdd00000, 0xbdf40000, 0x3eaf0000, 0xbe680000, 0xbe5a0000,
        0xbd000000, 0x3e8d0000, 0x00000000, 0xbdec0000, 0xbeb20000, 0xbd380000, 0x3d700000,
        0x3d180000, 0x3e4c0000, 0xbe840000, 0x3df40000, 0xbcc00000, 0xbc600000, 0x3e260000,
        0x3d980000, 0x3cd00000, 0xbb800000, 0x3d700000, 0x3e2a0000, 0xbe040000, 0xbe2e0000,
        0xbe940000, 0xbd940000, 0xbb800000, 0x3e4a0000, 0xbd880000, 0xbcb00000, 0x3da80000,
        0xbe660000, 0x3d280000, 0x3e200000, 0x3e0a0000, 0x3e100000, 0x3d500000, 0xbd780000,
        0xbdd80000, 0xbdac0000, 0xbd400000, 0xbe1e0000, 0x3e540000, 0xbde40000, 0x3de80000,
        0x3c200000, 0xbe240000, 0xbb000000, 0xbd580000, 0x3e3c0000, 0xbc600000, 0xbe340000,
        0xbe400000, 0x3e910000, 0x3d000000, 0x3e2a0000, 0x3ecc0000, 0xbea70000, 0x3d500000,
        0xbe1e0000, 0xbec60000, 0xbe060000, 0x3e1c0000, 0xbe220000, 0x3d080000, 0x3d100000,
        0x3e4e0000, 0x3e4c0000, 0x3d880000, 0xbc600000, 0x3b800000, 0x3cf00000, 0xbdb80000,
        0x3d8c0000, 0x3d000000, 0xbdac0000, 0xbdd80000, 0xbec50000, 0xbe640000, 0xbe660000,
        0xbdfc0000, 0xbdc00000, 0x3e3a0000, 0xbdf00000,
    ];
    /// qt 17, 104 B / 256 elems.
    const EXPECTED_MQ3G256: [u32; 256] = [
        0xbdf00000, 0xbe9a0000, 0x3e840000, 0x3d880000, 0x3df00000, 0x3e3c0000, 0xbd400000,
        0xbe4c0000, 0x3dfc0000, 0x3e420000, 0xbecf0000, 0xbec10000, 0x3d080000, 0x3d780000,
        0x3d380000, 0xbe890000, 0xbe5c0000, 0x3d600000, 0xbdd80000, 0x3d000000, 0xbed60000,
        0xbef40000, 0xbede0000, 0xbe780000, 0xbd680000, 0x3d180000, 0x3efd0000, 0xbd280000,
        0xbea70000, 0x3e3e0000, 0xbebf0000, 0xbf278000, 0xbe9e0000, 0x3e480000, 0xbede0000,
        0xbdd00000, 0xbee60000, 0x3efc0000, 0xbdb80000, 0x3e800000, 0xbe2a0000, 0xbe1e0000,
        0xbeb30000, 0xbd780000, 0x3d180000, 0xbe120000, 0x3ea30000, 0xbe910000, 0x80000000,
        0xbdc80000, 0x3f320000, 0x3f110000, 0xbd800000, 0x3ede0000, 0xbd400000, 0x3e3c0000,
        0x3e8f0000, 0x3e830000, 0xbd840000, 0xbee30000, 0x3e760000, 0x3e890000, 0xbead0000,
        0xbe810000, 0x3e760000, 0xbea90000, 0xbd480000, 0xbe2e0000, 0xbe930000, 0x3db40000,
        0xbdac0000, 0xbeff0000, 0xbd000000, 0x3e500000, 0x3e900000, 0x3ea80000, 0xbeba0000,
        0x3dc80000, 0xbebe0000, 0x3e2c0000, 0xbec30000, 0xbe020000, 0x3d380000, 0xbee10000,
        0xbd9c0000, 0x3e870000, 0xbef90000, 0x3df40000, 0xbeba0000, 0x3ee60000, 0xbe540000,
        0x3e1c0000, 0xbeb00000, 0xbc800000, 0x3d400000, 0x3f040000, 0xbcb00000, 0xbf0a8000,
        0xbe320000, 0xbead0000, 0x3e160000, 0xbe990000, 0x3e1a0000, 0xbf058000, 0x3e040000,
        0x3da80000, 0x3fa58000, 0xbf090000, 0x80000000, 0xbc800000, 0xbe380000, 0xbe780000,
        0x3eef0000, 0x3f2c8000, 0xbd8c0000, 0x3b000000, 0xbd940000, 0x3f008000, 0x3e2a0000,
        0xbdb40000, 0x3ec00000, 0x3d000000, 0x3e480000, 0xbd900000, 0xbf2d0000, 0x3d880000,
        0xbed20000, 0xbf770000, 0x3f2d0000, 0xbec40000, 0xbe640000, 0xbe8c0000, 0x3da00000,
        0x3ece0000, 0xbeb00000, 0xbdc80000, 0xbe120000, 0x3e910000, 0xbd480000, 0xbd280000,
        0x3ec30000, 0xbddc0000, 0xbef90000, 0x3ea10000, 0x3e600000, 0xbe5c0000, 0x3e800000,
        0xbeae0000, 0x3e340000, 0xbd000000, 0x3da80000, 0xbe900000, 0xbe4a0000, 0x3ec10000,
        0x3dec0000, 0xbee30000, 0x3e3a0000, 0x3dac0000, 0xbd480000, 0xbedb0000, 0xbea80000,
        0x3d100000, 0xbea00000, 0x3e0c0000, 0x3de80000, 0x3eb80000, 0xbea20000, 0xbea00000,
        0x3cd00000, 0xbe320000, 0x3d080000, 0xbd080000, 0x3efd0000, 0xbeed0000, 0xbf2a8000,
        0x3e420000, 0x3ed60000, 0xbed00000, 0xbd880000, 0xbe200000, 0xbdd00000, 0x3e340000,
        0xbdf00000, 0x3c400000, 0xbe830000, 0x3dac0000, 0x3e520000, 0xbd8c0000, 0x3dbc0000,
        0x3dac0000, 0xbe620000, 0xbda40000, 0xbe5a0000, 0x3ebf0000, 0x3cb00000, 0xbe9f0000,
        0xbed90000, 0xbe320000, 0xbdc40000, 0x3ec90000, 0xbdb80000, 0xbca00000, 0xbe540000,
        0xbe8e0000, 0x3d500000, 0x3e040000, 0x3e860000, 0x3e9a0000, 0xbe320000, 0x3eb70000,
        0xbd9c0000, 0x3d480000, 0xbee10000, 0xbecb0000, 0xbcf00000, 0xbe6a0000, 0xbe100000,
        0x3eb00000, 0xbf360000, 0xbd800000, 0xbee80000, 0x3e940000, 0xbe700000, 0xbe900000,
        0x3e9f0000, 0x3f038000, 0xbe760000, 0x3f168000, 0x3ee10000, 0xbf0a8000, 0xbeab0000,
        0x3e0e0000, 0xbe500000, 0xbe940000, 0x3eb40000, 0xbdd00000, 0xbdf00000, 0xbe840000,
        0x3eb40000, 0xbde00000, 0x3e9b0000, 0x3e020000, 0x3e220000, 0xbe560000, 0xbdc40000,
        0xbe810000, 0xbefd0000, 0x3d280000, 0x3e540000, 0xbf010000, 0x3c400000, 0xbe540000,
        0xbd500000, 0xbed20000, 0x3f1d0000, 0xbe8e0000,
    ];
    /// qt 20, 112 B / 256 elems.
    const EXPECTED_MQ3G256LLOYD: [u32; 256] = [
        0x3ee80000, 0x3ec00000, 0x3fa00000, 0x3f4c0000, 0x3d000000, 0xbe100000, 0xbd400000,
        0x3f080000, 0xbef00000, 0x3f340000, 0xbd400000, 0xbef00000, 0xbef80000, 0xbe000000,
        0x3ea00000, 0x3f0c0000, 0x3f800000, 0x3c800000, 0xbee80000, 0xbee00000, 0xbe700000,
        0xbec00000, 0x3f200000, 0x3f820000, 0xbf240000, 0x3f280000, 0xbdc00000, 0x3f0c0000,
        0xbed00000, 0x3e500000, 0x3d400000, 0xbd000000, 0x3e700000, 0xbe600000, 0xbe200000,
        0x3da00000, 0xbec00000, 0x3eb80000, 0x3e700000, 0x3f600000, 0xbf080000, 0x3f7c0000,
        0x3de00000, 0xbf780000, 0xbe980000, 0xbe000000, 0x3f300000, 0xc01f0000, 0xbd800000,
        0x3f7c0000, 0x3e980000, 0x3f200000, 0x3f3c0000, 0x3ea00000, 0xbd800000, 0xbf440000,
        0x3e300000, 0xbee00000, 0x3d800000, 0xbf0c0000, 0x3f400000, 0xbe100000, 0xbf2c0000,
        0x3f700000, 0xbf580000, 0x3da00000, 0x3e880000, 0x80000000, 0xbec80000, 0xbf200000,
        0xbf680000, 0x3d400000, 0xbf0c0000, 0xbf080000, 0xbf500000, 0x3f340000, 0x3f400000,
        0x3e880000, 0xbf040000, 0xbef00000, 0xbf440000, 0xbec00000, 0x3f380000, 0xbef80000,
        0x3f600000, 0xbc800000, 0x3e980000, 0x3e200000, 0xbd800000, 0xbf540000, 0xbe700000,
        0x3fa40000, 0xbda00000, 0xbf100000, 0xbf180000, 0x3d400000, 0xbf600000, 0xbf1c0000,
        0xbeb80000, 0xbf280000, 0x3da00000, 0xbf780000, 0xbee00000, 0xbea80000, 0xbf1c0000,
        0x3f080000, 0xbe400000, 0x3f0c0000, 0xbec00000, 0xbde00000, 0x3ea80000, 0xbeb00000,
        0x3f2c0000, 0xbf880000, 0x3dc00000, 0x3c800000, 0xbf400000, 0x3f1c0000, 0xbfa20000,
        0x3dc00000, 0xbef00000, 0x3f540000, 0x3e300000, 0x3f500000, 0x3eb80000, 0x3ed00000,
        0x3e400000, 0xbf0c0000, 0x3ed80000, 0x3fc00000, 0xbe400000, 0xbe100000, 0xbf180000,
        0xbf820000, 0x3eb80000, 0xbeb00000, 0xbe000000, 0x3e880000, 0x3ed80000, 0x3f980000,
        0x3da00000, 0xbf180000, 0xbe900000, 0xbe500000, 0x3e800000, 0x3e880000, 0x3f8e0000,
        0xbf600000, 0x3ef80000, 0x3f200000, 0xbd800000, 0x3ed80000, 0xbe100000, 0x3e400000,
        0xbf300000, 0x3c800000, 0xbea00000, 0xbf240000, 0x3d400000, 0xbf400000, 0x3f6c0000,
        0xbec00000, 0x80000000, 0x3f960000, 0x3e900000, 0x3f2c0000, 0x3f1c0000, 0x3f580000,
        0xbf840000, 0x3ed80000, 0x3f920000, 0xbf080000, 0x3eb80000, 0x3ec00000, 0x3ec00000,
        0xbf440000, 0xbf280000, 0x3f6c0000, 0x3e500000, 0xbf280000, 0xbf3c0000, 0xbf280000,
        0x3ef00000, 0x3f140000, 0x3ef80000, 0xbe400000, 0x3e000000, 0x3e300000, 0xbf200000,
        0xbf6c0000, 0xbf860000, 0xbe000000, 0xbe000000, 0xbef80000, 0xbf6c0000, 0xbed00000,
        0x3ef80000, 0x3f840000, 0xbd800000, 0xbf0c0000, 0x3f6c0000, 0x3f940000, 0xbe000000,
        0x3f1c0000, 0x3ee00000, 0x3f0c0000, 0x3f3c0000, 0x3f380000, 0xbf4c0000, 0x3d000000,
        0x3f300000, 0x3d400000, 0x3e200000, 0x3d400000, 0xbf440000, 0x3e800000, 0xbee00000,
        0xbf2c0000, 0xbf6c0000, 0xbdc00000, 0xbf7c0000, 0x00000000, 0xbe200000, 0x3ed80000,
        0xbec00000, 0xbee80000, 0xbde00000, 0x3f280000, 0xbf740000, 0xbd000000, 0x3f300000,
        0x3f440000, 0x3f8a0000, 0x3e000000, 0x3e200000, 0x3eb80000, 0xbf580000, 0x3de00000,
        0xbe980000, 0xbf200000, 0xbef80000, 0x3f300000, 0x3dc00000, 0x3e700000, 0x3d800000,
        0x3e880000, 0xbd400000, 0xbdc00000, 0xbe000000, 0x3d400000, 0x3ed80000, 0xbe600000,
        0xbec80000, 0xbe400000, 0xbe200000, 0x3e300000,
    ];
    /// qt 30, 160 B / 256 elems.
    const EXPECTED_MQ4G256LLOYD: [u32; 256] = [
        0x3f300000, 0xbe800000, 0xbf080000, 0xbed00000, 0x3e200000, 0x3dc00000, 0xbfa00000,
        0x3ee00000, 0x3f480000, 0x3ed00000, 0x3e000000, 0x3f100000, 0xbea00000, 0xbf600000,
        0xbf180000, 0xbf280000, 0xbd000000, 0x3f780000, 0xbf400000, 0x3d800000, 0x3d800000,
        0xbec00000, 0xbeb00000, 0x3fa40000, 0xbf000000, 0xbea00000, 0x3e600000, 0xbf940000,
        0x3dc00000, 0x3f8c0000, 0xbd800000, 0x3f200000, 0x3e400000, 0x3e800000, 0x3e900000,
        0x3eb00000, 0x3ef00000, 0x3f080000, 0xbf200000, 0x3ee00000, 0xbf280000, 0x3e600000,
        0x3ec00000, 0x3ea00000, 0xbee00000, 0x00000000, 0xbe200000, 0xbf580000, 0x3e200000,
        0xbe600000, 0x3e000000, 0xbe400000, 0xbea00000, 0xbf400000, 0x3eb00000, 0x3eb00000,
        0x3ec00000, 0xbe400000, 0xbe900000, 0xbe900000, 0x3ef00000, 0xbfb40000, 0xbf500000,
        0x3f000000, 0xbe880000, 0xbf0c0000, 0x3e880000, 0xbe980000, 0x3e700000, 0x3e500000,
        0xbe700000, 0x3e980000, 0xbe300000, 0x3e880000, 0xbea80000, 0x3c800000, 0xbea80000,
        0xbef80000, 0xbea80000, 0x3c800000, 0xbe880000, 0xbee80000, 0xbe100000, 0xbe300000,
        0x3ec80000, 0x3f540000, 0x3f040000, 0xbf4c0000, 0xbc800000, 0x3d400000, 0xbef80000,
        0xbd400000, 0x3fe20000, 0x3ee80000, 0xbef80000, 0xbf0c0000, 0xbf540000, 0xbf240000,
        0xbfb20000, 0xbf4c0000, 0x3e700000, 0x3faa0000, 0xbfca0000, 0xbf8e0000, 0x3ee80000,
        0xbf9e0000, 0xbc800000, 0xbea80000, 0x3f820000, 0xbfb60000, 0xbed80000, 0xbf640000,
        0xbc800000, 0xbf340000, 0x3e980000, 0x3f640000, 0xbf4c0000, 0x3f920000, 0xbeb80000,
        0x3faa0000, 0xbfa60000, 0x3de00000, 0xbfbe0000, 0x3e500000, 0xbec80000, 0x3faa0000,
        0x3d400000, 0x3f240000, 0xbeb80000, 0x3eb80000, 0xbe700000, 0x3eb80000, 0x3e300000,
        0xbea80000, 0x3e500000, 0x3f140000, 0xbec80000, 0x3f920000, 0x3f340000, 0x3f140000,
        0xbee80000, 0x3f340000, 0x3ec80000, 0x3e700000, 0xbe300000, 0xbf8a0000, 0xbf040000,
        0x3e100000, 0xbd400000, 0x3e980000, 0xbf040000, 0xbf240000, 0x3de00000, 0x3ec80000,
        0x3e880000, 0x3f1c0000, 0x3e100000, 0xbe100000, 0x3c800000, 0xbf240000, 0xbeb80000,
        0xbc800000, 0x3d400000, 0x3e980000, 0x3f640000, 0xbc800000, 0xbd400000, 0x3d400000,
        0x3f0c0000, 0xbe300000, 0x3f540000, 0x3e300000, 0xbf140000, 0x3e500000, 0x3ed80000,
        0x3ea80000, 0x3e880000, 0x3e700000, 0x3e700000, 0x3f1c0000, 0x3e500000, 0xbe500000,
        0x3ed80000, 0x3e980000, 0x3e700000, 0x3c800000, 0x3ea80000, 0x3ee80000, 0x3da00000,
        0xbe300000, 0x3de00000, 0x3ef80000, 0xbd800000, 0xbd800000, 0xbf080000, 0x3ef00000,
        0x80000000, 0xbec00000, 0xbed00000, 0x3dc00000, 0xbf800000, 0x3e000000, 0x3e200000,
        0xbf680000, 0xbe800000, 0x3e000000, 0x3ed00000, 0x3f280000, 0xbee00000, 0xbf300000,
        0xbf280000, 0x3e200000, 0x3e400000, 0xbee00000, 0xbf580000, 0xbf580000, 0x00000000,
        0x3e000000, 0xbeb00000, 0x3f180000, 0x3d800000, 0xbf300000, 0x3ef00000, 0xbf380000,
        0x3f400000, 0x3e000000, 0x3e200000, 0x3eb00000, 0xbd800000, 0xbee00000, 0xbe600000,
        0xbe600000, 0x3fa80000, 0x3fd80000, 0x3f9c0000, 0x3ef00000, 0xbe400000, 0x3f300000,
        0x3f580000, 0x3ed00000, 0x3f200000, 0xbe000000, 0x3dc00000, 0xbf780000, 0x80000000,
        0x80000000, 0x3f180000, 0xbe900000, 0xbd800000, 0xbee00000, 0xbef00000, 0x3f180000,
        0xbec00000, 0xbec00000, 0x3e600000, 0xbe200000,
    ];
    /// qt 18, 72 B / 256 elems.
    const EXPECTED_MQ2G256: [u32; 256] = [
        0x3e420000, 0x3f058000, 0xbe910000, 0xbe360000, 0xbea50000, 0xbe990000, 0x3dcc0000,
        0x3e7e0000, 0xbe2a0000, 0xbe9f0000, 0x3f1d8000, 0x3ef10000, 0xbe7e0000, 0xbcd00000,
        0xbd9c0000, 0x3ec50000, 0x3e970000, 0xbd8c0000, 0x3dc40000, 0xbd280000, 0x3f1b8000,
        0x3f118000, 0x3f3b8000, 0x3ecd0000, 0x3d380000, 0x3dac0000, 0xbf388000, 0x3db40000,
        0x3f048000, 0xbe950000, 0x3f038000, 0x3f758000, 0x3ef90000, 0xbebd0000, 0x3f248000,
        0x3e3a0000, 0x3f268000, 0xbf378000, 0x3e160000, 0xbeb70000, 0x3ea50000, 0x3e7e0000,
        0x3ec50000, 0x3e220000, 0xbddc0000, 0x3e4a0000, 0xbef10000, 0x3ec30000, 0xbd180000,
        0x3e5e0000, 0xbf758000, 0xbf5c8000, 0x3e3a0000, 0xbf308000, 0x3d680000, 0xbea10000,
        0xbeb30000, 0xbeb70000, 0x3d580000, 0x3f208000, 0xbe760000, 0xbec10000, 0x3edb0000,
        0x3ebf0000, 0xbec50000, 0x3ea90000, 0x3d280000, 0x3e890000, 0x3ecb0000, 0xbe6e0000,
        0x3e5a0000, 0x3f378000, 0x3dac0000, 0xbe420000, 0xbecb0000, 0xbef10000, 0x3f028000,
        0xbd840000, 0x3f1e8000, 0xbec90000, 0x3f1e8000, 0x3e420000, 0x3d280000, 0x3f238000,
        0x3e7e0000, 0xbf0a8000, 0x3f3f8000, 0xbe2a0000, 0x3f078000, 0xbf0e8000, 0x3e990000,
        0xbe360000, 0x3f068000, 0x3da40000, 0xbdcc0000, 0xbf3b8000, 0xbda40000, 0x3f598000,
        0x3ea70000, 0x3ebd0000, 0xbe520000, 0x3ebd0000, 0xbe810000, 0x3f4a8000, 0xbe990000,
        0xbd940000, 0xbfedc000, 0x3f5a8000, 0x3d780000, 0x3d8c0000, 0x3ed70000, 0x3e850000,
        0xbf1d8000, 0xbf8dc000, 0x3df40000, 0x3cf00000, 0x3c600000, 0xbf268000, 0xbe990000,
        0x3e460000, 0xbf2f8000, 0xbdb40000, 0xbe870000, 0x3e2a0000, 0x3f758000, 0xbdbc0000,
        0x3f1d8000, 0x3fb44000, 0xbf778000, 0x3edb0000, 0x3e720000, 0x3ed30000, 0x3c900000,
        0xbf298000, 0x3f038000, 0x3bc00000, 0x3ea70000, 0xbf058000, 0x3b000000, 0x3df40000,
        0xbf2b8000, 0x3e160000, 0x3f2b8000, 0xbeeb0000, 0xbe8f0000, 0x3e9b0000, 0xbe910000,
        0x3f0a8000, 0xbe6a0000, 0x3d480000, 0xbd8c0000, 0x3ed10000, 0x3ea30000, 0xbf038000,
        0xbe460000, 0x3f1b8000, 0xbe910000, 0xbe460000, 0x3d380000, 0x3f228000, 0x3edd0000,
        0xbd080000, 0x3efb0000, 0xbe120000, 0xbe2a0000, 0xbf038000, 0x3ebb0000, 0x3ef90000,
        0xbd680000, 0x3e320000, 0xbdac0000, 0x3d780000, 0xbf358000, 0x3f3f8000, 0x3f728000,
        0xbea10000, 0xbf338000, 0x3f298000, 0x3d9c0000, 0x3e930000, 0x3d8c0000, 0xbe9f0000,
        0x3d940000, 0x3da40000, 0x3edb0000, 0xbe3e0000, 0xbe8b0000, 0x3e620000, 0xbe420000,
        0xbe5a0000, 0x3e890000, 0x3d280000, 0x3e7e0000, 0xbef30000, 0x3d480000, 0x3f0a8000,
        0x3f258000, 0x3ec10000, 0x3e0a0000, 0xbeff0000, 0x3dcc0000, 0x3cf00000, 0x3e950000,
        0x3ed70000, 0xbdc40000, 0xbe6a0000, 0xbec10000, 0xbeb50000, 0x3e9f0000, 0xbf098000,
        0x3e3e0000, 0xbe660000, 0x3f2b8000, 0x3f258000, 0x3d780000, 0x3e660000, 0x3e320000,
        0xbf0e8000, 0x3f85c000, 0x3cd00000, 0x3f348000, 0xbeed0000, 0x3ec10000, 0x3f0d8000,
        0xbee70000, 0xbf4a8000, 0x3eb10000, 0xbf568000, 0xbf268000, 0x3f3b8000, 0x3efd0000,
        0xbe0e0000, 0x3ebb0000, 0x3ec70000, 0xbf018000, 0x3e620000, 0x3e620000, 0x3ee50000,
        0xbf088000, 0x3c200000, 0xbed30000, 0xbe2e0000, 0xbe260000, 0x3ea70000, 0x3dd40000,
        0x3eaf0000, 0x3f198000, 0xbd480000, 0xbead0000, 0x3f408000, 0xbdcc0000, 0x3e5e0000,
        0x3d580000, 0x3f1c8000, 0xbf768000, 0x3ecf0000,
    ];
    /// qt 19, 72 B / 256 elems.
    const EXPECTED_MQ2G256LLOYD: [u32; 256] = [
        0x3ed00000, 0x3e900000, 0x3d000000, 0xbe600000, 0xbf500000, 0xbec00000, 0x3f600000,
        0xbf300000, 0x3d800000, 0xbf000000, 0xbe000000, 0xbf500000, 0xbfe40000, 0x3f380000,
        0xbeb00000, 0xbe900000, 0x3ea00000, 0xbe000000, 0x3f400000, 0x3ea00000, 0xbeb00000,
        0xbf840000, 0xbe200000, 0x3dc00000, 0xbe900000, 0x3f080000, 0x3e900000, 0x3f280000,
        0x3e400000, 0xbe000000, 0x3e000000, 0xbd800000, 0xbf200000, 0xbfe80000, 0x3ea00000,
        0x3e800000, 0x3ef00000, 0x3ed00000, 0xbed00000, 0x3f580000, 0x3f680000, 0x3fd40000,
        0xbe900000, 0xbeb00000, 0xbf900000, 0x3f100000, 0x3f500000, 0xbec00000, 0xbffc0000,
        0x3f680000, 0xbd000000, 0xbe900000, 0x3e000000, 0xbfb80000, 0x3d800000, 0xbf200000,
        0x3f800000, 0x3f500000, 0xbd800000, 0x3e000000, 0x3f580000, 0x3eb00000, 0x3f180000,
        0xbef00000, 0x3e000000, 0xbfb80000, 0xbea00000, 0xbec00000, 0xbd000000, 0xbfa40000,
        0x3f580000, 0xbe900000, 0x3f080000, 0xbed00000, 0xbdc00000, 0x3f680000, 0x00000000,
        0xbe400000, 0xbd800000, 0x3e000000, 0xbe200000, 0x3ed00000, 0xbeb00000, 0x3e900000,
        0x3fd00000, 0xbf300000, 0x3ea00000, 0x80000000, 0x3ec00000, 0x3fa80000, 0x3ea00000,
        0xbe000000, 0x3ed00000, 0x3ef00000, 0xbf680000, 0x3e200000, 0xbef00000, 0xbd000000,
        0x3f580000, 0xbf080000, 0xbf100000, 0xbf200000, 0x3f000000, 0x3f880000, 0xbea00000,
        0xbec00000, 0x3ec00000, 0x3e400000, 0x3e600000, 0x3e200000, 0x3ed00000, 0xbf180000,
        0x3e400000, 0xbf200000, 0xbf900000, 0xbf100000, 0xbf680000, 0xbf180000, 0xbeb00000,
        0x3ef00000, 0xbfb40000, 0xbf080000, 0x3ed00000, 0xbe200000, 0xbd800000, 0xbf400000,
        0xbe000000, 0xbe400000, 0xbe200000, 0xbfdc0000, 0xbf780000, 0x3eb00000, 0xbf400000,
        0xbf600000, 0x3d800000, 0xbf880000, 0x3f700000, 0xbea00000, 0xbf900000, 0x3fa00000,
        0xbf680000, 0xbef00000, 0xbfb40000, 0x3f840000, 0x3d800000, 0x3ee00000, 0x00000000,
        0x3f000000, 0x3f940000, 0xbe200000, 0xbe600000, 0x3e900000, 0x3f480000, 0xbd000000,
        0xbef00000, 0xbe900000, 0x00000000, 0xbf000000, 0xbea00000, 0x3ea00000, 0xbfd80000,
        0x3ee00000, 0xbec00000, 0x3f200000, 0x3dc00000, 0xbdc00000, 0xbf380000, 0xbf080000,
        0xbd000000, 0xc00e0000, 0xbf080000, 0x3d000000, 0x3f000000, 0x3e800000, 0xbea00000,
        0xbf300000, 0xbfdc0000, 0x3f580000, 0x3ed00000, 0x3d000000, 0x3e800000, 0xbf200000,
        0x3ea00000, 0x3ee00000, 0x3f300000, 0xbea00000, 0x3e800000, 0x3f900000, 0xbf080000,
        0xbf680000, 0xbe900000, 0xbed00000, 0xbf500000, 0x3f300000, 0xbe800000, 0x3f200000,
        0x3ed00000, 0x3ffc0000, 0xbf380000, 0x3e200000, 0x3f680000, 0xbef00000, 0x3e200000,
        0x3ef00000, 0xbf400000, 0x3f900000, 0x3ee00000, 0x3f500000, 0x3f080000, 0xbf080000,
        0x3eb00000, 0xbf8c0000, 0x3f800000, 0x3f000000, 0x3d800000, 0xbf980000, 0x3d800000,
        0x3fb80000, 0xbf200000, 0xbf200000, 0xbf280000, 0x3f680000, 0xbeb00000, 0x3f580000,
        0xbf8c0000, 0xbf940000, 0xbf840000, 0x3e900000, 0xbf200000, 0xbe000000, 0xbea00000,
        0x3f300000, 0xbf300000, 0xbe400000, 0x3e000000, 0x3f900000, 0x3f840000, 0x3f840000,
        0xbef00000, 0xbf840000, 0xbea00000, 0x3d800000, 0x40200000, 0x3f600000, 0xbfac0000,
        0x3f9c0000, 0xbef00000, 0x3e200000, 0xbf940000, 0x3e900000, 0xbf680000, 0xbfbc0000,
        0xbe000000, 0xbe800000, 0xbd800000, 0x3d800000,
    ];
    /// qt 8, 200 B / 256 elems.
    const EXPECTED_HFQ6G256: [u32; 256] = [
        0x3e8c0000, 0x3e000000, 0x3d200000, 0x3df00000, 0x3e100000, 0x3de00000, 0xbd000000,
        0x3eac0000, 0x3c000000, 0x3d800000, 0x3e8c0000, 0x3d400000, 0xbe000000, 0x3d400000,
        0x3e500000, 0x3e880000, 0x3e780000, 0x3cc00000, 0x3c000000, 0xbc800000, 0x3de00000,
        0xbcc00000, 0xbd800000, 0x3e500000, 0xbcc00000, 0xbd200000, 0x3ebc0000, 0xbdb00000,
        0x3eb00000, 0xbdc00000, 0x3e300000, 0x3e080000, 0x3e580000, 0xbde00000, 0x3dd00000,
        0x3eb40000, 0x3da00000, 0x3eb00000, 0xbdd00000, 0x3d900000, 0xbd600000, 0x3ea80000,
        0x3ea80000, 0x3e900000, 0x3ea00000, 0x3e9c0000, 0x3e080000, 0x00000000, 0x3e380000,
        0x3e840000, 0x3d800000, 0x3e600000, 0x3d400000, 0x3e780000, 0xbc000000, 0xbd800000,
        0xbdb00000, 0x3e480000, 0x3e980000, 0x3e180000, 0x3e900000, 0x3e300000, 0x3e680000,
        0x3ebc0000, 0x3e180000, 0x3e000000, 0x3d000000, 0x3db00000, 0x3c800000, 0x3de00000,
        0xbd200000, 0x3e9c0000, 0xbdf00000, 0x3d800000, 0x3e880000, 0x3c800000, 0x3e800000,
        0x3d200000, 0x3e480000, 0x3e700000, 0x3df00000, 0x3cc00000, 0x00000000, 0xbd400000,
        0xbc800000, 0xbcc00000, 0xbd900000, 0x3e300000, 0x3eb40000, 0xbd400000, 0x3eb80000,
        0xbdf00000, 0x3e600000, 0xbdc00000, 0x3e280000, 0x3dd00000, 0x3db00000, 0xbde00000,
        0x3dc00000, 0x3ea40000, 0xbd400000, 0x3eb00000, 0xbde00000, 0x3d200000, 0x3ea40000,
        0x3ea40000, 0x3ea40000, 0x3e800000, 0x3e400000, 0x3e9c0000, 0x3e000000, 0xbd000000,
        0x3d600000, 0x3e840000, 0x3d600000, 0x3e400000, 0xbda00000, 0x3e780000, 0xbc800000,
        0xbdc00000, 0x3e940000, 0x3e400000, 0x3e940000, 0x3df00000, 0x3e200000, 0x3e300000,
        0x3e600000, 0x3eac0000, 0x3cc00000, 0x3e000000, 0x3cc00000, 0x3d600000, 0xbde00000,
        0x3de00000, 0xbd400000, 0x3e8c0000, 0x3e840000, 0x3db00000, 0x3e840000, 0xbc800000,
        0x3e000000, 0x3d200000, 0x3e400000, 0x3e500000, 0xbc000000, 0x3cc00000, 0x3df00000,
        0xbda00000, 0x3eb80000, 0xbd000000, 0xbda00000, 0x3e100000, 0x3e680000, 0xbd400000,
        0x3eb40000, 0x3eb40000, 0x3dc00000, 0xbdc00000, 0x3e200000, 0x3d900000, 0xbd200000,
        0xbde00000, 0x3db00000, 0x3e940000, 0x3ea80000, 0x3ebc0000, 0xbdf00000, 0x3c000000,
        0x3e480000, 0x3ea40000, 0x3ea00000, 0x3e600000, 0x3d800000, 0x3e9c0000, 0x3e780000,
        0xbd800000, 0xbd900000, 0x3e840000, 0x3d400000, 0x3e200000, 0x3e980000, 0x3e700000,
        0xbcc00000, 0xbe000000, 0x3e280000, 0x3e400000, 0x3e900000, 0x3db00000, 0x3d000000,
        0x3e300000, 0x3e580000, 0x3e9c0000, 0xbdd00000, 0x3e000000, 0x3c800000, 0x3cc00000,
        0x3e880000, 0x3dd00000, 0xbd600000, 0x3e780000, 0x3e080000, 0x3db00000, 0x3e800000,
        0xbd400000, 0x00000000, 0x3d200000, 0x3e380000, 0x3e300000, 0x3ebc0000, 0x3c800000,
        0x3de00000, 0xbde00000, 0x3e700000, 0xbd000000, 0xbdb00000, 0x3de00000, 0x3dd00000,
        0xbd400000, 0x3eb00000, 0x3ea40000, 0xbd000000, 0xbdc00000, 0x3e180000, 0x3d200000,
        0x3eac0000, 0xbdf00000, 0x3da00000, 0x3e840000, 0x3e500000, 0x3ebc0000, 0xbe000000,
        0xbcc00000, 0x3d900000, 0x3ea40000, 0x3e9c0000, 0x3e400000, 0xbd800000, 0x3e9c0000,
        0x3e700000, 0xbdc00000, 0x3e9c0000, 0x3e800000, 0x3d200000, 0x3e000000, 0x3e300000,
        0x3e700000, 0xbd000000, 0x3eb00000, 0x3d200000, 0x3e400000, 0x3e8c0000, 0x3d600000,
        0xbdc00000, 0x3e300000, 0x3e500000, 0x3e8c0000,
    ];
    /// qt 6, 136 B / 256 elems.
    const EXPECTED_HFQ4G256: [u32; 256] = [
        0xbed00000, 0xbed00000, 0xbe800000, 0xbeb00000, 0xbdc00000, 0xbe900000, 0xbee00000,
        0xbe400000, 0xbe900000, 0xbe000000, 0xbe000000, 0xbd800000, 0xbef00000, 0xbef00000,
        0xbea00000, 0xbed00000, 0xbe200000, 0xbeb00000, 0xbf000000, 0xbe800000, 0xbeb00000,
        0xbe400000, 0xbe400000, 0xbe000000, 0xbd000000, 0xbd800000, 0xbec00000, 0xbef00000,
        0xbe600000, 0xbed00000, 0xbd800000, 0xbeb00000, 0xbed00000, 0xbe800000, 0xbe800000,
        0xbe400000, 0xbdc00000, 0xbe000000, 0xbee00000, 0xbd000000, 0xbe900000, 0xbef00000,
        0xbe000000, 0xbed00000, 0xbef00000, 0xbea00000, 0xbea00000, 0xbe800000, 0xbe200000,
        0xbe400000, 0xbf000000, 0xbdc00000, 0xbeb00000, 0xbd000000, 0xbe400000, 0xbef00000,
        0xbd000000, 0xbed00000, 0xbec00000, 0xbea00000, 0xbe600000, 0xbe800000, 0xbd800000,
        0xbe400000, 0xbed00000, 0xbdc00000, 0xbe800000, 0xbd000000, 0xbdc00000, 0xbef00000,
        0xbee00000, 0xbec00000, 0xbe900000, 0xbea00000, 0xbe000000, 0xbe800000, 0xbef00000,
        0xbe200000, 0xbea00000, 0xbdc00000, 0xbe200000, 0xbd000000, 0xbf000000, 0xbee00000,
        0xbeb00000, 0xbec00000, 0xbe400000, 0xbea00000, 0xbd000000, 0xbe800000, 0xbec00000,
        0xbe200000, 0xbe600000, 0xbdc00000, 0xbd800000, 0xbd000000, 0xbed00000, 0xbee00000,
        0xbe800000, 0xbec00000, 0xbdc00000, 0xbea00000, 0xbee00000, 0xbe600000, 0xbe900000,
        0xbe200000, 0xbe000000, 0xbdc00000, 0xbef00000, 0xbf000000, 0xbea00000, 0xbee00000,
        0xbe200000, 0xbec00000, 0xbf000000, 0xbe900000, 0xbeb00000, 0xbe600000, 0xbe400000,
        0xbe200000, 0xbd000000, 0xbdc00000, 0xbec00000, 0xbf000000, 0xbe600000, 0xbee00000,
        0xbd800000, 0xbec00000, 0xbed00000, 0xbe900000, 0xbe800000, 0xbe600000, 0xbdc00000,
        0xbe200000, 0xbee00000, 0xbd800000, 0xbe900000, 0xbf000000, 0xbe000000, 0xbee00000,
        0xbef00000, 0xbeb00000, 0xbea00000, 0xbe900000, 0xbe200000, 0xbe600000, 0xbf000000,
        0xbe000000, 0xbeb00000, 0xbd800000, 0xbe400000, 0xbf000000, 0xbd000000, 0xbee00000,
        0xbec00000, 0xbeb00000, 0xbe600000, 0xbe900000, 0xbd800000, 0xbe600000, 0xbed00000,
        0xbe000000, 0xbe800000, 0xbd800000, 0xbdc00000, 0xbf000000, 0xbee00000, 0xbed00000,
        0xbe900000, 0xbeb00000, 0xbe000000, 0xbe900000, 0xbef00000, 0xbe400000, 0xbea00000,
        0xbe000000, 0xbe200000, 0xbd800000, 0xbf000000, 0xbef00000, 0xbeb00000, 0xbed00000,
        0xbe400000, 0xbeb00000, 0xbd000000, 0xbe900000, 0xbec00000, 0xbe400000, 0xbe600000,
        0xbe000000, 0xbd800000, 0xbd800000, 0xbed00000, 0xbef00000, 0xbe800000, 0xbed00000,
        0xbdc00000, 0xbeb00000, 0xbee00000, 0xbe800000, 0xbe900000, 0xbe400000, 0xbe000000,
        0xbe000000, 0xbef00000, 0xbd000000, 0xbea00000, 0xbef00000, 0xbe200000, 0xbed00000,
        0xbf000000, 0xbea00000, 0xbeb00000, 0xbe800000, 0xbe400000, 0xbe400000, 0xbd000000,
        0xbe000000, 0xbec00000, 0xbd000000, 0xbe600000, 0xbef00000, 0xbd800000, 0xbed00000,
        0xbed00000, 0xbea00000, 0xbe800000, 0xbe800000, 0xbdc00000, 0xbe400000, 0xbee00000,
        0xbdc00000, 0xbe900000, 0xbd000000, 0xbe000000, 0xbef00000, 0xbef00000, 0xbec00000,
        0xbea00000, 0xbea00000, 0xbe200000, 0xbe800000, 0xbf000000, 0xbe200000, 0xbeb00000,
        0xbdc00000, 0xbe400000, 0xbd000000, 0xbd000000, 0xbef00000, 0xbec00000, 0xbec00000,
        0xbe600000, 0xbea00000, 0xbd800000, 0xbe800000,
    ];
    /// qt 11, 104 B / 256 elems.
    const EXPECTED_HFQ3G256: [u32; 256] = [
        0xbed00000, 0xbea00000, 0xbf000000, 0xbec00000, 0xbeb00000, 0xbee00000, 0xbe900000,
        0xbed00000, 0xbee00000, 0xbec00000, 0xbea00000, 0xbed00000, 0xbec00000, 0xbef00000,
        0xbed00000, 0xbe900000, 0xbef00000, 0xbee00000, 0xbf000000, 0xbed00000, 0xbed00000,
        0xbea00000, 0xbea00000, 0xbee00000, 0xbf000000, 0xbf000000, 0xbea00000, 0xbee00000,
        0xbee00000, 0xbeb00000, 0xbee00000, 0xbea00000, 0xbe900000, 0xbeb00000, 0xbed00000,
        0xbee00000, 0xbef00000, 0xbee00000, 0xbea00000, 0xbef00000, 0xbea00000, 0xbed00000,
        0xbeb00000, 0xbef00000, 0xbf000000, 0xbef00000, 0xbee00000, 0xbeb00000, 0xbeb00000,
        0xbef00000, 0xbed00000, 0xbef00000, 0xbe900000, 0xbe900000, 0xbeb00000, 0xbf000000,
        0xbec00000, 0xbe900000, 0xbec00000, 0xbf000000, 0xbea00000, 0xbec00000, 0xbef00000,
        0xbec00000, 0xbed00000, 0xbeb00000, 0xbee00000, 0xbf000000, 0xbeb00000, 0xbed00000,
        0xbeb00000, 0xbe900000, 0xbee00000, 0xbed00000, 0xbec00000, 0xbe900000, 0xbed00000,
        0xbf000000, 0xbef00000, 0xbed00000, 0xbef00000, 0xbef00000, 0xbee00000, 0xbe900000,
        0xbee00000, 0xbe900000, 0xbec00000, 0xbea00000, 0xbf000000, 0xbe900000, 0xbe900000,
        0xbea00000, 0xbef00000, 0xbec00000, 0xbf000000, 0xbee00000, 0xbe900000, 0xbec00000,
        0xbef00000, 0xbea00000, 0xbf000000, 0xbed00000, 0xbec00000, 0xbeb00000, 0xbea00000,
        0xbee00000, 0xbe900000, 0xbeb00000, 0xbe900000, 0xbef00000, 0xbf000000, 0xbef00000,
        0xbeb00000, 0xbf000000, 0xbef00000, 0xbeb00000, 0xbea00000, 0xbea00000, 0xbed00000,
        0xbec00000, 0xbec00000, 0xbea00000, 0xbea00000, 0xbec00000, 0xbeb00000, 0xbeb00000,
        0xbe900000, 0xbe900000, 0xbed00000, 0xbec00000, 0xbf000000, 0xbec00000, 0xbec00000,
        0xbee00000, 0xbed00000, 0xbed00000, 0xbee00000, 0xbee00000, 0xbea00000, 0xbed00000,
        0xbed00000, 0xbef00000, 0xbe900000, 0xbea00000, 0xbef00000, 0xbf000000, 0xbf000000,
        0xbed00000, 0xbee00000, 0xbea00000, 0xbee00000, 0xbee00000, 0xbf000000, 0xbea00000,
        0xbeb00000, 0xbee00000, 0xbef00000, 0xbeb00000, 0xbea00000, 0xbeb00000, 0xbe900000,
        0xbed00000, 0xbed00000, 0xbee00000, 0xbf000000, 0xbee00000, 0xbee00000, 0xbef00000,
        0xbea00000, 0xbef00000, 0xbeb00000, 0xbef00000, 0xbe900000, 0xbf000000, 0xbea00000,
        0xbec00000, 0xbeb00000, 0xbe900000, 0xbee00000, 0xbef00000, 0xbea00000, 0xbe900000,
        0xbef00000, 0xbf000000, 0xbec00000, 0xbeb00000, 0xbec00000, 0xbf000000, 0xbeb00000,
        0xbec00000, 0xbeb00000, 0xbed00000, 0xbed00000, 0xbed00000, 0xbee00000, 0xbf000000,
        0xbec00000, 0xbed00000, 0xbef00000, 0xbe900000, 0xbee00000, 0xbef00000, 0xbec00000,
        0xbe900000, 0xbee00000, 0xbf000000, 0xbeb00000, 0xbee00000, 0xbef00000, 0xbe900000,
        0xbef00000, 0xbe900000, 0xbef00000, 0xbe900000, 0xbf000000, 0xbea00000, 0xbf000000,
        0xbeb00000, 0xbe900000, 0xbea00000, 0xbf000000, 0xbec00000, 0xbec00000, 0xbef00000,
        0xbe900000, 0xbee00000, 0xbef00000, 0xbea00000, 0xbe900000, 0xbee00000, 0xbf000000,
        0xbeb00000, 0xbea00000, 0xbf000000, 0xbe900000, 0xbeb00000, 0xbea00000, 0xbef00000,
        0xbec00000, 0xbf000000, 0xbeb00000, 0xbea00000, 0xbf000000, 0xbeb00000, 0xbeb00000,
        0xbea00000, 0xbe900000, 0xbed00000, 0xbec00000, 0xbec00000, 0xbea00000, 0xbec00000,
        0xbec00000, 0xbeb00000, 0xbed00000, 0xbe900000,
    ];
    /// qt 7, 72 B / 128 elems.
    const EXPECTED_HFQ4G128: [u32; 128] = [
        0xbed00000, 0xbed00000, 0xbe800000, 0xbeb00000, 0xbdc00000, 0xbe900000, 0xbee00000,
        0xbe400000, 0xbe900000, 0xbe000000, 0xbe000000, 0xbd800000, 0xbef00000, 0xbef00000,
        0xbea00000, 0xbed00000, 0xbe200000, 0xbeb00000, 0xbf000000, 0xbe800000, 0xbeb00000,
        0xbe400000, 0xbe400000, 0xbe000000, 0xbd000000, 0xbd800000, 0xbec00000, 0xbef00000,
        0xbe600000, 0xbed00000, 0xbd800000, 0xbeb00000, 0xbed00000, 0xbe800000, 0xbe800000,
        0xbe400000, 0xbdc00000, 0xbe000000, 0xbee00000, 0xbd000000, 0xbe900000, 0xbef00000,
        0xbe000000, 0xbed00000, 0xbef00000, 0xbea00000, 0xbea00000, 0xbe800000, 0xbe200000,
        0xbe400000, 0xbf000000, 0xbdc00000, 0xbeb00000, 0xbd000000, 0xbe400000, 0xbef00000,
        0xbd000000, 0xbed00000, 0xbec00000, 0xbea00000, 0xbe600000, 0xbe800000, 0xbd800000,
        0xbe400000, 0xbed00000, 0xbdc00000, 0xbe800000, 0xbd000000, 0xbdc00000, 0xbef00000,
        0xbee00000, 0xbec00000, 0xbe900000, 0xbea00000, 0xbe000000, 0xbe800000, 0xbef00000,
        0xbe200000, 0xbea00000, 0xbdc00000, 0xbe200000, 0xbd000000, 0xbf000000, 0xbee00000,
        0xbeb00000, 0xbec00000, 0xbe400000, 0xbea00000, 0xbd000000, 0xbe800000, 0xbec00000,
        0xbe200000, 0xbe600000, 0xbdc00000, 0xbd800000, 0xbd000000, 0xbed00000, 0xbee00000,
        0xbe800000, 0xbec00000, 0xbdc00000, 0xbea00000, 0xbee00000, 0xbe600000, 0xbe900000,
        0xbe200000, 0xbe000000, 0xbdc00000, 0xbef00000, 0xbf000000, 0xbea00000, 0xbee00000,
        0xbe200000, 0xbec00000, 0xbf000000, 0xbe900000, 0xbeb00000, 0xbe600000, 0xbe400000,
        0xbe200000, 0xbd000000, 0xbdc00000, 0xbec00000, 0xbf000000, 0xbe600000, 0xbee00000,
        0xbd800000, 0xbec00000,
    ];
    /// qt 12, 56 B / 128 elems.
    const EXPECTED_HFQ3G128: [u32; 128] = [
        0xbed00000, 0xbea00000, 0xbf000000, 0xbec00000, 0xbeb00000, 0xbee00000, 0xbe900000,
        0xbed00000, 0xbee00000, 0xbec00000, 0xbea00000, 0xbed00000, 0xbec00000, 0xbef00000,
        0xbed00000, 0xbe900000, 0xbef00000, 0xbee00000, 0xbf000000, 0xbed00000, 0xbed00000,
        0xbea00000, 0xbea00000, 0xbee00000, 0xbf000000, 0xbf000000, 0xbea00000, 0xbee00000,
        0xbee00000, 0xbeb00000, 0xbee00000, 0xbea00000, 0xbe900000, 0xbeb00000, 0xbed00000,
        0xbee00000, 0xbef00000, 0xbee00000, 0xbea00000, 0xbef00000, 0xbea00000, 0xbed00000,
        0xbeb00000, 0xbef00000, 0xbf000000, 0xbef00000, 0xbee00000, 0xbeb00000, 0xbeb00000,
        0xbef00000, 0xbed00000, 0xbef00000, 0xbe900000, 0xbe900000, 0xbeb00000, 0xbf000000,
        0xbec00000, 0xbe900000, 0xbec00000, 0xbf000000, 0xbea00000, 0xbec00000, 0xbef00000,
        0xbec00000, 0xbed00000, 0xbeb00000, 0xbee00000, 0xbf000000, 0xbeb00000, 0xbed00000,
        0xbeb00000, 0xbe900000, 0xbee00000, 0xbed00000, 0xbec00000, 0xbe900000, 0xbed00000,
        0xbf000000, 0xbef00000, 0xbed00000, 0xbef00000, 0xbef00000, 0xbee00000, 0xbe900000,
        0xbee00000, 0xbe900000, 0xbec00000, 0xbea00000, 0xbf000000, 0xbe900000, 0xbe900000,
        0xbea00000, 0xbef00000, 0xbec00000, 0xbf000000, 0xbee00000, 0xbe900000, 0xbec00000,
        0xbef00000, 0xbea00000, 0xbf000000, 0xbed00000, 0xbec00000, 0xbeb00000, 0xbea00000,
        0xbee00000, 0xbe900000, 0xbeb00000, 0xbe900000, 0xbef00000, 0xbf000000, 0xbef00000,
        0xbeb00000, 0xbf000000, 0xbef00000, 0xbeb00000, 0xbea00000, 0xbea00000, 0xbed00000,
        0xbec00000, 0xbec00000, 0xbea00000, 0xbea00000, 0xbec00000, 0xbeb00000, 0xbeb00000,
        0xbe900000, 0xbe900000,
    ];
    /// qt 40, 34 B / 128 elems.
    const EXPECTED_TQ2G128: [u32; 128] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f000000, 0x3f000000, 0x3f800000,
        0x00000000, 0x3f800000, 0x3f800000, 0x00000000, 0x3f000000, 0xbf000000, 0x00000000,
        0xbf000000, 0x3f800000, 0x00000000, 0x3f000000, 0x3f000000, 0x3f800000, 0x3f000000,
        0x3f800000, 0xbf000000, 0xbf000000, 0x3f800000, 0xbf000000, 0x3f800000, 0xbf000000,
        0xbf000000, 0x3f000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000,
        0x00000000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f800000, 0x00000000,
        0xbf000000, 0x3f800000, 0xbf000000, 0x3f800000, 0x3f000000, 0x3f800000, 0x00000000,
        0xbf000000, 0x00000000, 0xbf000000, 0x3f000000, 0x00000000, 0x3f800000, 0xbf000000,
        0x3f800000, 0x3f000000, 0x00000000, 0x00000000, 0xbf000000, 0xbf000000, 0xbf000000,
        0x3f000000, 0x00000000, 0x00000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000,
        0xbf000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f000000, 0x3f800000, 0xbf000000,
        0x00000000, 0x00000000, 0xbf000000, 0x00000000, 0x3f000000, 0x3f800000, 0xbf000000,
        0x3f000000, 0x3f800000, 0x00000000, 0x00000000, 0x3f800000, 0xbf000000, 0xbf000000,
        0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x00000000, 0x3f800000,
        0xbf000000, 0x3f800000, 0x3f000000, 0xbf000000, 0x3f800000, 0x3f800000, 0x3f800000,
        0x00000000, 0x00000000, 0xbf000000, 0xbf000000, 0x3f800000, 0x3f800000, 0xbf000000,
        0x00000000, 0xbf000000, 0x3f000000, 0x00000000, 0x3f000000, 0x00000000, 0xbf000000,
        0x3f000000, 0x3f800000, 0x3f000000, 0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000,
        0x00000000, 0x3f800000,
    ];
    /// qt 41, 18 B / 128 elems.
    const EXPECTED_BQ1G128: [u32; 128] = [
        0x3f000000, 0xbf000000, 0x3f000000, 0xbf000000, 0x3f000000, 0xbf000000, 0x3f000000,
        0xbf000000, 0xbf000000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000,
        0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000,
        0xbf000000, 0xbf000000, 0x3f000000, 0xbf000000, 0xbf000000, 0x3f000000, 0xbf000000,
        0xbf000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000,
        0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0xbf000000, 0x3f000000,
        0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000, 0x3f000000,
        0x3f000000, 0xbf000000, 0xbf000000, 0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000,
        0xbf000000, 0xbf000000, 0xbf000000, 0x3f000000, 0x3f000000, 0xbf000000, 0x3f000000,
        0xbf000000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000,
        0x3f000000, 0xbf000000, 0xbf000000, 0x3f000000, 0xbf000000, 0xbf000000, 0xbf000000,
        0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0xbf000000,
        0xbf000000, 0xbf000000, 0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000, 0x3f000000,
        0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0xbf000000,
        0xbf000000, 0xbf000000, 0x3f000000, 0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000,
        0x3f000000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0xbf000000, 0xbf000000,
        0x3f000000, 0x3f000000, 0xbf000000, 0x3f000000, 0x3f000000, 0xbf000000, 0x3f000000,
        0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000, 0xbf000000,
        0xbf000000, 0x3f000000,
    ];
    /// qt 1, 512 B / 256 elems.
    const EXPECTED_F16: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 2, 1024 B / 256 elems.
    const EXPECTED_F32: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 16, 512 B / 256 elems.
    const EXPECTED_BF16: [u32; 256] = [
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000,
        0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000,
        0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000, 0x00000000,
        0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
        0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000, 0xc0400000,
        0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40400000,
        0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
        0x40400000, 0xc0400000, 0x40a00000, 0x00000000, 0x3f800000, 0xbf800000, 0x40000000,
        0xc0000000, 0x40400000, 0xc0400000, 0x40a00000,
    ];
    /// qt 3, 34 B / 32 elems.
    const EXPECTED_Q8F16: [u32; 32] = [
        0x422a0000, 0x42740000, 0xc2420000, 0xc1f00000, 0xc1380000, 0x40e00000, 0x41cc0000,
        0x42300000, 0x427a0000, 0xc23c0000, 0xc1e40000, 0xc1200000, 0x41080000, 0x41d80000,
        0x42360000, 0xc2800000, 0xc2360000, 0xc1d80000, 0xc1080000, 0x41200000, 0x41e40000,
        0x423c0000, 0xc27a0000, 0xc2300000, 0xc1cc0000, 0xc0e00000, 0x41380000, 0x41f00000,
        0x42420000, 0xc2740000, 0xc22a0000, 0xc1c00000,
    ];

    fn assert_bits(got: &[f32], expected: &[u32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g.to_bits(),
                *e,
                "{what}: element {i} = {g} (0x{:08x}) != {} (0x{e:08x})",
                g.to_bits(),
                f32::from_bits(*e)
            );
        }
    }

    #[test]
    fn dequant_group_matches_canonical_decoder_bits() {
        for (q, expected) in [
            (CpuQuant::Mq4G256, &EXPECTED_MQ4G256[..]),
            (CpuQuant::Mq4G256V2, &EXPECTED_MQ4G256V2[..]),
            (CpuQuant::Mq6G256, &EXPECTED_MQ6G256[..]),
            (CpuQuant::Mq3G256, &EXPECTED_MQ3G256[..]),
            (CpuQuant::Mq3G256Lloyd, &EXPECTED_MQ3G256LLOYD[..]),
            (CpuQuant::Mq4G256Lloyd, &EXPECTED_MQ4G256LLOYD[..]),
            (CpuQuant::Mq2G256, &EXPECTED_MQ2G256[..]),
            (CpuQuant::Mq2G256Lloyd, &EXPECTED_MQ2G256LLOYD[..]),
            (CpuQuant::Hfq6G256, &EXPECTED_HFQ6G256[..]),
            (CpuQuant::Hfq4G256, &EXPECTED_HFQ4G256[..]),
            (CpuQuant::Hfq3G256, &EXPECTED_HFQ3G256[..]),
            (CpuQuant::Hfq4G128, &EXPECTED_HFQ4G128[..]),
            (CpuQuant::Hfq3G128, &EXPECTED_HFQ3G128[..]),
            (CpuQuant::Tq2G128, &EXPECTED_TQ2G128[..]),
            (CpuQuant::Bq1G128, &EXPECTED_BQ1G128[..]),
            (CpuQuant::F16, &EXPECTED_F16[..]),
            (CpuQuant::F32, &EXPECTED_F32[..]),
            (CpuQuant::Bf16, &EXPECTED_BF16[..]),
            (CpuQuant::Q8F16, &EXPECTED_Q8F16[..]),
        ] {
            let packed = fixture(q);
            let mut out = vec![0.0f32; q.group_elems()];
            dequant_group(q, &packed, &mut out);
            assert_bits(&out, expected, &format!("dequant_group({q:?})"));
        }
    }

    #[test]
    fn decode_group_codes_is_the_affine_decode() {
        // The codes decode must be the un-rotated half of `dequant_group`: for a
        // rotated format, the affine decode has no 256-element structure of its
        // own, so its first 8 values are exactly `scale * code + zero` on the
        // nibble stream the fixture writes.
        let packed = fixture(CpuQuant::Mq4G256);
        let mut codes = [0.0f32; 256];
        decode_group_codes(CpuQuant::Mq4G256, &packed, &mut codes);
        let scale = f32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
        let zero = f32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
        for i in 0..256 {
            let byte = packed[8 + i / 2];
            let nibble = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            assert_eq!(codes[i], scale * nibble as f32 + zero, "element {i}");
        }

        // And for an unrotated format the two decodes are identical.
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        decode_group_codes(CpuQuant::Hfq4G256, &fixture(CpuQuant::Hfq4G256), &mut a);
        dequant_group(CpuQuant::Hfq4G256, &fixture(CpuQuant::Hfq4G256), &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn mfp4_e8_preserves_coordinate_and_scale_bits() {
        // Literal lattice points cover both cosets, both coordinate-7 parities,
        // and extrema. They are independent of either decoder's bit unpacking.
        const POINTS: [(u32, [f32; 8]); 6] = [
            (0x00000000, [-7.0; 8]),
            (0x80000000, [-6.5; 8]),
            (0x11111111, [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -4.0]),
            (0x91111111, [-5.5, -5.5, -5.5, -5.5, -5.5, -5.5, -5.5, -3.5]),
            (0x77777777, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]),
            (0xffffffff, [8.5; 8]),
        ];
        let mut packed = [0u8; 152];
        let mut got = [0.0f32; 256];
        for bits in [0u16, 0x2802, 0x3800, 0x3c01] {
            packed[..2].copy_from_slice(&bits.to_le_bytes());
            let row_scale = half::f16::from_bits(bits).to_f32();
            for code in 0..=255u8 {
                let exp = (code >> 3) & 15;
                let mant = code & 7;
                let block_scale = if exp == 0 {
                    2.0f32.powi(-6) * mant as f32 / 8.0
                } else if exp == 15 && mant == 7 {
                    448.0
                } else {
                    2.0f32.powi(exp as i32 - 7) * (1.0 + mant as f32 / 8.0)
                };
                for block in 0..8 {
                    let at = 16 + block * 17;
                    packed[at] = code;
                    for word in 0..4 {
                        let point = POINTS[(block * 4 + word) % POINTS.len()].0;
                        packed[at + 1 + word * 4..at + 5 + word * 4]
                            .copy_from_slice(&point.to_le_bytes());
                    }
                }
                decode_group_codes(CpuQuant::Mfp4G32E8, &packed, &mut got);
                for (i, &value) in got.iter().enumerate() {
                    let coord = POINTS[(i / 8) % POINTS.len()].1[i % 8];
                    let want = (row_scale * block_scale * 0.88) * coord;
                    assert_eq!(
                        value.to_bits(),
                        want.to_bits(),
                        "header={bits:04x} scale={code:02x} coordinate={i}"
                    );
                }
            }
        }
        for k in [256, 512, 4096] {
            assert_eq!(
                crate::gemv::row_bytes(CpuQuant::Mfp4G32E8, k),
                16 + (k / 32) * 17
            );
        }
    }

    /// Every group layout the crate decodes, pinned literally. Row headers are
    /// accounted separately by `gemv::row_bytes`.
    /// `(qt, quant, group_elems, group_bytes, rotated)`.
    const FORMATS: &[(u8, CpuQuant, usize, usize, bool)] = &[
        (1, CpuQuant::F16, 256, 512, false),
        (2, CpuQuant::F32, 256, 1024, false),
        (3, CpuQuant::Q8F16, 32, 34, false),
        (6, CpuQuant::Hfq4G256, 256, 136, false),
        (7, CpuQuant::Hfq4G128, 128, 72, false),
        (8, CpuQuant::Hfq6G256, 256, 200, false),
        (9, CpuQuant::Hfq2G256, 256, 72, false),
        (10, CpuQuant::Hfq2G128, 128, 40, false),
        (11, CpuQuant::Hfq3G256, 256, 104, false),
        (12, CpuQuant::Hfq3G128, 128, 56, false),
        (13, CpuQuant::Mq4G256, 256, 136, true),
        (15, CpuQuant::Mq6G256, 256, 200, true),
        (16, CpuQuant::Bf16, 256, 512, false),
        (17, CpuQuant::Mq3G256, 256, 104, true),
        (18, CpuQuant::Mq2G256, 256, 72, true),
        (19, CpuQuant::Mq2G256Lloyd, 256, 72, true),
        (20, CpuQuant::Mq3G256Lloyd, 256, 112, true),
        (30, CpuQuant::Mq4G256Lloyd, 256, 160, true),
        (31, CpuQuant::Mq5G256, 256, 168, true),
        (34, CpuQuant::Mfp4G32E8, 256, 136, true),
        (40, CpuQuant::Tq2G128, 128, 34, false),
        (41, CpuQuant::Bq1G128, 128, 18, false),
        (44, CpuQuant::Mq4G256V2, 256, 136, true),
        (45, CpuQuant::Mq4CG256, 256, 136, true),
        (47, CpuQuant::Mq6G256V2, 256, 200, true),
        (48, CpuQuant::Mq5G256V2, 256, 168, true),
        (49, CpuQuant::Mq3G256V2, 256, 104, true),
        (50, CpuQuant::Mq2G256V2, 256, 72, true),
        (51, CpuQuant::Mq2G256LloydU, 256, 72, false),
        (53, CpuQuant::Mq4G128V2, 128, 68, false),
    ];

    #[test]
    fn quant_type_round_trip_and_group_sizes() {
        for &(qt, q, ge, gb, rotated) in FORMATS {
            assert_eq!(CpuQuant::from_quant_type(qt), Some(q), "qt {qt}");
            assert_eq!(
                (q.group_elems(), q.group_bytes()),
                (ge, gb),
                "qt {qt} {q:?}"
            );
            assert_eq!(q.is_fwht_g256(), rotated, "qt {qt} {q:?} rotation");
            assert_eq!(
                q.is_fwht_g128(),
                q == CpuQuant::Mq4G128V2,
                "qt {qt} {q:?} g128 rotation"
            );
            assert!(
                !(q.is_fwht_g256() && q.is_fwht_g128()),
                "qt {qt} {q:?} claims both rotations"
            );
            let want = if q.is_fwht_g256() {
                InputRotation::FwhtG256
            } else if q == CpuQuant::Mq4G128V2 {
                InputRotation::FwhtG128V2
            } else {
                InputRotation::None
            };
            assert_eq!(q.input_rotation(), want, "qt {qt} {q:?} input_rotation");
            // 128-group formats take k % 128; every other format keeps k % 256.
            assert_eq!(
                q.k_multiple(),
                if ge == 128 { 128 } else { 256 },
                "qt {qt} {q:?} k_multiple"
            );
        }
        // Deliberately NOT covered, each for a structural reason (see the crate
        // and `docs/perf-checkpoints/2026-09-27-gfx1201-cpu-exec-offload.md`):
        //   0/4   Q4F16G64 / Q4K — GGUF-side formats, not in the dense HFQ set
        //   5     Q8HFQ — padded rows (`DType::row_stride`), which the group model
        //         cannot express; needs the stride to be threaded through
        //   14    MQ8G256 — `RotationPlan::Mq8Internal`, i.e. an int8-quantized
        //         activation, not a plain FWHT of f32
        //   21/24/32/33/35-37 HFP4/MFP4/... — decoders not implemented;
        //         qt 34's AoS E8 row header is handled explicitly
        //   28/29 PARO4G128(-T) — a Givens rotation on the *activation*
        //   38/39 MQ{2,3}G256GL — MoE-indexed only, no dense GEMV kernel exists
        //   22    TidI32 — a routing table, not a weight
        for qt in [
            0u8, 4, 5, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 35, 36, 37, 38, 39, 43, 46,
            255,
        ] {
            assert_eq!(CpuQuant::from_quant_type(qt), None, "qt {qt}");
        }
    }

    /// `e8p_decode_word` on words worked out by hand from the wire rule
    /// (`v_k = 2·(±abs2) + (shift ? -1 : +1)`, lane 7 negative iff
    /// `popcount(sign7) + parity(idx)` is odd).
    #[test]
    fn e8p_decode_word_matches_hand_computed_words() {
        // idx 0: abs2 = [1; 8], Σ|g| = 4 (even row).
        assert_eq!(e8p_decode_word(0x0000), [3; 8]);
        assert_eq!(e8p_decode_word(0x8000), [1; 8]);
        // sign7 = 1: lane 0 negative, popcount 1 makes lane 7 negative too.
        assert_eq!(e8p_decode_word(0x0100), [-1, 3, 3, 3, 3, 3, 3, -1]);
        // idx 1: abs2 = [1,1,1,1,1,1,1,3], Σ|g| = 5 (odd row): with no stored
        // sign the odd parity alone negates lane 7.
        assert_eq!(E8P_ROW_PARITY[1], 1);
        assert_eq!(e8p_decode_word(0x0001), [3, 3, 3, 3, 3, 3, 3, -5]);
        // ...and sign7 = 1 (popcount 1) cancels it again.
        assert_eq!(e8p_decode_word(0x0101), [-1, 3, 3, 3, 3, 3, 3, 7]);
        // Every lane of every word is odd and in [-11, 11].
        for word in 0..=u16::MAX {
            for v in e8p_decode_word(word) {
                assert!(
                    v % 2 != 0 && (-11..=11).contains(&v),
                    "word {word:#06x}: {v}"
                );
            }
        }
        assert_eq!(E8P_ROW_PARITY.iter().filter(|&&p| p == 1).count(), 149);
    }

    /// `decode_planar_row` on a hand-built W3 row: codes `0..=7` in the first
    /// chunk (LSB-first bit stream), all-zero codes elsewhere, one scale per
    /// group.
    #[test]
    fn decode_planar_row_w3_matches_hand_computed_weights() {
        let k = 256;
        let mut codes = vec![0u8; k * 3 / 8];
        let acc: u32 = (0..8).map(|j| (j as u32) << (3 * j)).sum();
        codes[..3].copy_from_slice(&acc.to_le_bytes()[..3]);
        // Second group: chunk 0 holds all-ones codes (7).
        codes[48..51].copy_from_slice(&0x00ff_ffffu32.to_le_bytes()[..3]);
        // Scales: 1.0 (0x3c00) and 0.5 (0x3800).
        let scales = [0x00u8, 0x3c, 0x00, 0x38];
        let mut out = vec![f32::NAN; k];
        decode_planar_row(CpuQuant::Mq3I4G128R256, &codes, &scales, k, &mut out);
        let want_first = [-7.0f32, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0];
        assert_eq!(&out[..8], &want_first);
        assert!(out[8..128].iter().all(|&w| w == -7.0));
        assert!(out[128..136].iter().all(|&w| w == 7.0 * 0.5));
        assert!(out[136..].iter().all(|&w| w == -7.0 * 0.5));
    }

    /// `decode_planar_row` on a hand-built E8P row: `(f32(d)·0.25)·v`.
    #[test]
    fn decode_planar_row_e8p_matches_hand_computed_weights() {
        let k = 128;
        let mut codes = vec![0u8; k / 4];
        // Word 0 = 0x0001 (v = [3,3,3,3,3,3,3,-5]), word 1 = 0x8000 (v = [1; 8]).
        codes[0..2].copy_from_slice(&0x0001u16.to_le_bytes());
        codes[2..4].copy_from_slice(&0x8000u16.to_le_bytes());
        // d = 2.0 (0x4000) -> scale 0.5.
        let scales = 0x4000u16.to_le_bytes();
        let mut out = vec![f32::NAN; k];
        decode_planar_row(CpuQuant::E8pI4G128R128, &codes, &scales, k, &mut out);
        assert_eq!(&out[..8], &[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, -2.5]);
        assert_eq!(&out[8..16], &[0.5; 8]);
        // The remaining words are 0x0000 -> v = 3 everywhere.
        assert!(out[16..].iter().all(|&w| w == 1.5));
    }

    #[test]
    #[should_panic(expected = "planar format")]
    fn decode_group_codes_rejects_planar_formats() {
        let mut out = vec![0.0f32; 128];
        decode_group_codes(CpuQuant::E8pI4G128R256, &[0u8; 64], &mut out);
    }

    #[test]
    #[should_panic(expected = "planar format")]
    fn dequant_group_rejects_planar_formats() {
        let mut out = vec![0.0f32; 256];
        dequant_group(CpuQuant::Mq3I4G128R128, &[0u8; 64], &mut out);
    }

    #[test]
    #[should_panic(expected = "not a planar format")]
    fn decode_planar_row_rejects_row_formats() {
        let mut out = vec![0.0f32; 256];
        decode_planar_row(CpuQuant::Mq4G256, &[0u8; 136], &[], 256, &mut out);
    }

    #[test]
    fn f16_widening_is_ieee() {
        // Every value the generator uses, plus the awkward encodings: subnormal,
        // zero, min/max normal, inf, NaN.
        for bits in EXPECTED_F16_BITS {
            let want = half::f16::from_bits(bits).to_f32();
            assert_eq!(half_f16_to_f32(bits), want, "bits 0x{bits:04x}");
        }
        for bits in [
            0x0001u16, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0x7c01, 0xfc00, 0x8000, 0xffff,
        ] {
            let want = half::f16::from_bits(bits).to_f32();
            let got = half_f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "bits 0x{bits:04x}: {got} is not NaN");
            } else {
                assert_eq!(got, want, "bits 0x{bits:04x}");
            }
        }
    }

    #[test]
    fn divide_by_awq_scale_is_per_channel_and_pre_rotation() {
        // (W·s)·(x/s) = W·x: the divide must cancel the pre-scaled weight, so
        // dividing then rotating a probe activation must equal rotating the
        // unscaled activation of a (W/s)-weighted... — the property that matters
        // and is checkable in isolation is that the divide is per channel and
        // precedes the rotation, i.e. R(x/s) != R(x)/s in general and equals a
        // hand-built R applied to x/s.
        let scale = [2.0f32, 4.0, 0.5, 8.0];
        let mut x = [4.0f32, 8.0, 1.0, 16.0];
        divide_by_awq_scale(&mut x, &scale);
        assert_eq!(x, [2.0, 2.0, 2.0, 2.0]);
        // A longer scale than activation is accepted (the loader's buffer is
        // exactly k, but the contract is `>=`), a shorter one is a caller bug.
        divide_by_awq_scale(&mut x, &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(x, [2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "awq scale has 3 channels")]
    fn divide_by_awq_scale_rejects_a_short_scale() {
        let mut x = [0.0f32; 4];
        divide_by_awq_scale(&mut x, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn fwht_signs_are_pinned_to_the_existing_generator() {
        // Literal transcription of `KvCache::gen_fwht_signs(42, 256)`'s first 16
        // and last 4 entries, taken from the generator that produced the tables
        // above; a change in the LCG must fail here rather than silently produce
        // different weights under the same name.
        let s1 = fwht_signs(FWHT_SEED1, FWHT_N);
        let s2 = fwht_signs(FWHT_SEED2, FWHT_N);
        let lit = |v: &[f32]| -> Vec<f32> { v.to_vec() };
        assert_eq!(s1.len(), 256);
        assert_eq!(s2.len(), 256);
        assert_eq!(
            lit(&s1[..16]),
            vec![
                1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                -1.0
            ],
            "signs1 head"
        );
        assert_eq!(
            lit(&s2[..16]),
            vec![
                1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
                1.0
            ],
            "signs2 head"
        );
        assert_eq!(lit(&s1[252..]), vec![-1.0, -1.0, 1.0, -1.0], "signs1 tail");
        assert_eq!(lit(&s2[252..]), vec![1.0, 1.0, -1.0, -1.0], "signs2 tail");
        // Balanced and unit-magnitude on any seed/length.
        for seed in [0u32, 1, 42, 1042, u32::MAX] {
            let s = fwht_signs(seed, 4096);
            assert!(s.iter().all(|v| *v == 1.0 || *v == -1.0));
            let sum: i32 = s.iter().map(|v| *v as i32).sum();
            assert!(sum.abs() < 200, "seed {seed} is badly imbalanced: {sum}");
        }
    }

    /// Independent oracle for the forward rotation: `R = 0.0625 * D2 * H * D1`
    /// where `H[i][j] = (-1)^popcount(i & j)` is the unnormalized
    /// Walsh-Hadamard matrix built from the definition, not from the butterfly
    /// (which is what `rotate_x` implements). `R e_j` is column `j`.
    #[test]
    fn rotate_x_matches_the_hadamard_matrix() {
        let s1 = fwht_signs(FWHT_SEED1, FWHT_N);
        let s2 = fwht_signs(FWHT_SEED2, FWHT_N);
        let r = |i: usize, j: usize| -> f32 {
            let sign = if ((i & j).count_ones() % 2) == 0 {
                1.0
            } else {
                -1.0
            };
            0.0625 * s2[i] * sign * s1[j]
        };
        for j in [0usize, 1, 17, 128, 255] {
            let mut x = vec![0.0f32; FWHT_N];
            x[j] = 1.0;
            rotate_x(&mut x);
            for i in 0..FWHT_N {
                assert_eq!(x[i], r(i, j), "column {j}, row {i}");
            }
        }
    }

    #[test]
    fn fwht256_inplace_inverts_rotate_x() {
        // Round trip on a spread of deterministic inputs, including a second
        // 256-block so a wrong block stride cannot pass.
        for seed in 0..3u64 {
            let mut x: Vec<f32> = (0..512)
                .map(|i| {
                    let v = (i as u64 * 2654435761 + seed * 7919) % 1000;
                    (v as f32 - 500.0) * 0.03125
                })
                .collect();
            let original = x.clone();
            rotate_x(&mut x);
            assert_ne!(x, original, "rotation is a no-op");
            for block in x.chunks_exact_mut(FWHT_N) {
                let group: &mut [f32; FWHT_N] = block.try_into().unwrap();
                fwht256_inplace(group, signs1(), signs2());
            }
            for (a, b) in x.iter().zip(&original) {
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "round trip drifted: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn rotate_x_preserves_norm_and_rejects_unaligned() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        rotate_x(&mut x);
        let after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (before - after).abs() <= 1e-3 * before,
            "rotation is not norm preserving: {before} -> {after}"
        );
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn rotate_x_panics_on_unaligned_input() {
        rotate_x(&mut [0.0f32; 255]);
    }

    // ---- qt 53 `Mq4G128V2` ------------------------------------------------

    /// A hand-built group with known values pins the wire layout against the
    /// layout in `quant_fwht.rs`/`quant_e8.rs` independent of any encoder:
    /// `[f16 scale @0][f16 zero @2][nibbles @4..68]`, low nibble = even element.
    #[test]
    fn mq4g128v2_hand_built_group_layout() {
        // Two groups of one row, distinct headers so a wrong group stride or a
        // swapped scale/zero shows up.
        let mut packed = vec![0u8; 2 * 68];
        // Group 0: scale 0.5 (0x3800), zero -1.0 (0xbc00); byte i = (i%16) | (15 - i%16) << 4.
        packed[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        packed[2..4].copy_from_slice(&0xbc00u16.to_le_bytes());
        for i in 0..64 {
            packed[4 + i] = ((i % 16) as u8) | ((15 - (i % 16) as u8) << 4);
        }
        // Group 1: scale 0.25 (0x3400), zero 2.0 (0x4000); all nibbles 0xA / 0x3.
        packed[68..70].copy_from_slice(&0x3400u16.to_le_bytes());
        packed[70..72].copy_from_slice(&0x4000u16.to_le_bytes());
        for i in 0..64 {
            packed[68 + 4 + i] = 0x3A;
        }
        let mut out = [0.0f32; 128];
        decode_group_codes(CpuQuant::Mq4G128V2, &packed, &mut out);
        for i in 0..64 {
            let (lo, hi) = ((i % 16) as f32, (15 - i % 16) as f32);
            assert_eq!(out[2 * i], 0.5 * lo - 1.0, "g0 even element {i}");
            assert_eq!(out[2 * i + 1], 0.5 * hi - 1.0, "g0 odd element {i}");
        }
        decode_group_codes(CpuQuant::Mq4G128V2, &packed[68..], &mut out);
        for i in 0..64 {
            assert_eq!(out[2 * i], 0.25 * 10.0 + 2.0, "g1 even element {i}");
            assert_eq!(out[2 * i + 1], 0.25 * 3.0 + 2.0, "g1 odd element {i}");
        }
    }

    fn lcg_floats(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Sum of three uniforms: roughly bell-shaped, in [-1.5, 1.5).
                let u = |sh: u32| ((s >> sh) & 0xFFFF) as f32 / 65536.0 - 0.5;
                u(8) + u(24) + u(40)
            })
            .collect()
    }

    /// `decode_group_codes` is bit-for-bit the reference decoder over a tensor
    /// the (transcribed) real encoder produced, and `dequant_group` — codes
    /// plus inverse FWHT-128 — recovers the original weights to 4-bit
    /// accuracy. A wrong inverse (sign tables, normalization, stage order) is
    /// a relative error near or above 1, not a quantization error.
    #[test]
    fn mq4g128v2_decode_matches_encoder_reference() {
        let (m, k) = (3usize, 640usize);
        let w = lcg_floats(0x53, m * k);
        let packed = crate::testfix::encode_mq4g128v2_ref(&w, m, k);
        assert_eq!(
            packed.len(),
            m * crate::gemv::row_bytes(CpuQuant::Mq4G128V2, k)
        );
        assert_eq!(packed.len(), m * 5 * 68);
        let reference = crate::testfix::decode_mq4g128v2_ref(&packed, m, k);
        let (mut err2, mut ref2) = (0.0f64, 0.0f64);
        for row in 0..m {
            for g in 0..k / 128 {
                let at = (row * (k / 128) + g) * 68;
                let mut codes = [0.0f32; 128];
                decode_group_codes(CpuQuant::Mq4G128V2, &packed[at..], &mut codes);
                let r0 = row * k + g * 128;
                assert_bits(
                    &codes,
                    &reference[r0..r0 + 128]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "qt53 codes vs reference decoder",
                );
                let mut orig = [0.0f32; 128];
                dequant_group(CpuQuant::Mq4G128V2, &packed[at..], &mut orig);
                for i in 0..128 {
                    let d = (orig[i] - w[r0 + i]) as f64;
                    err2 += d * d;
                    ref2 += (w[r0 + i] as f64).powi(2);
                }
            }
        }
        let rel = (err2 / ref2).sqrt();
        assert!(
            rel < 0.2,
            "qt53 dequant_group rel_l2 {rel:.4} vs the source weights"
        );
    }

    /// The first 16 and last 4 signs of the two 128-wide tables (seeds 43 /
    /// 1043), pinned literally from the generator `hipfire-quantize` and
    /// `Gpu::ensure_mq_signs_128` share.
    #[test]
    fn fwht128_signs_are_pinned() {
        let s1 = fwht_signs(FWHT128_SEED1, FWHT128_N);
        let s2 = fwht_signs(FWHT128_SEED2, FWHT128_N);
        assert_eq!(
            s1[..16],
            [
                1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
                1.0
            ]
        );
        assert_eq!(
            s2[..16],
            [
                1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
                -1.0
            ]
        );
        assert_eq!(s1[124..], [1.0, 1.0, -1.0, -1.0]);
        assert_eq!(s2[124..], [1.0, -1.0, 1.0, -1.0]);
        assert_eq!(&SIGNS128_1[..], &s1[..]);
        assert_eq!(&SIGNS128_2[..], &s2[..]);
        // The kernel's literal is the f32 nearest 1/sqrt(128).
        assert_eq!(FWHT128_SCALE, 1.0f32 / (128.0f32).sqrt());
    }

    /// Independent oracle: `R = (1/sqrt(128)) · D2 · H · D1` with `H` from its
    /// definition. `rotate_x_128_v2` of a unit vector is a column of `R`.
    #[test]
    fn rotate_x_128_v2_matches_the_hadamard_matrix() {
        let s1 = fwht_signs(43, 128);
        let s2 = fwht_signs(1043, 128);
        let norm = 1.0f32 / (128.0f32).sqrt();
        for j in [0usize, 1, 17, 64, 127] {
            let mut x = vec![0.0f32; 128];
            x[j] = 1.0;
            rotate_x_128_v2(&mut x);
            for i in 0..128 {
                let h = if (i & j).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let want = norm * s2[i] * h * s1[j];
                assert!(
                    (x[i] - want).abs() <= 1e-7,
                    "column {j}, row {i}: {} vs {want}",
                    x[i]
                );
            }
        }
    }

    /// Bitwise mirror of `mq_rotate_x_128_v2.hip`: 32 lanes x four adjacent
    /// values, in-lane stages (strides 1, 2), cross-lane XOR stages where the
    /// `tid & stride` lane computes `partner - mine`, then
    /// `* 0.0883883476 * signs2`. Includes the kernel's zero-fill of a short
    /// tail and its "only K values written" contract.
    fn gpu_kernel_mirror(x: &[f32]) -> Vec<f32> {
        let s1 = fwht_signs(43, 128);
        let s2 = fwht_signs(1043, 128);
        let mut out = vec![0.0f32; x.len()];
        for group in 0..x.len().div_ceil(128) {
            let at = |i: usize| -> f32 { x.get(group * 128 + i).copied().unwrap_or(0.0) };
            let mut v = vec![[0.0f32; 4]; 32];
            for (tid, lane) in v.iter_mut().enumerate() {
                let d0 = tid * 4;
                let (mut v0, mut v1, mut v2, mut v3) = (
                    at(d0) * s1[d0],
                    at(d0 + 1) * s1[d0 + 1],
                    at(d0 + 2) * s1[d0 + 2],
                    at(d0 + 3) * s1[d0 + 3],
                );
                let t = v0;
                v0 += v1;
                v1 = t - v1;
                let t = v2;
                v2 += v3;
                v3 = t - v3;
                let t = v0;
                v0 += v2;
                v2 = t - v2;
                let t = v1;
                v1 += v3;
                v3 = t - v3;
                *lane = [v0, v1, v2, v3];
            }
            let mut stride = 1usize;
            while stride < 32 {
                let old = v.clone();
                for tid in 0..32 {
                    for el in 0..4 {
                        let (partner, mine) = (old[tid ^ stride][el], old[tid][el]);
                        v[tid][el] = if tid & stride != 0 {
                            partner - mine
                        } else {
                            mine + partner
                        };
                    }
                }
                stride <<= 1;
            }
            for tid in 0..32 {
                for el in 0..4 {
                    let d = tid * 4 + el;
                    if group * 128 + d < x.len() {
                        out[group * 128 + d] = v[tid][el] * 0.0883883476f32 * s2[d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn rotate_x_128_v2_is_bitwise_the_gpu_kernel() {
        // k = 640 (the Flash-Next down shape, 5 groups), a single group, and
        // short/tail shapes including a non-multiple of 128.
        for len in [128usize, 640, 1, 200, 255, 130] {
            let x = lcg_floats(len as u64, len);
            let mut got = x.clone();
            rotate_x_128_v2(&mut got);
            let want = gpu_kernel_mirror(&x);
            let a: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "len {len}");
        }
    }

    /// A tail group is rotated as if zero-padded to 128, and only the first
    /// `len % 128` outputs are kept.
    #[test]
    fn rotate_x_128_v2_tail_is_zero_padded() {
        let x = lcg_floats(7, 200);
        let mut got = x.clone();
        rotate_x_128_v2(&mut got);
        let mut padded = x.clone();
        padded.resize(256, 0.0);
        rotate_x_128_v2(&mut padded);
        assert_eq!(got[..], padded[..200]);
    }

    #[test]
    fn fwht128_inplace_inverts_rotate_x_128_v2() {
        for seed in 0..3u64 {
            let x = lcg_floats(seed + 100, 384);
            let mut r = x.clone();
            rotate_x_128_v2(&mut r);
            assert_ne!(r, x, "rotation is a no-op");
            for block in r.chunks_exact_mut(FWHT128_N) {
                let group: &mut [f32; FWHT128_N] = block.try_into().unwrap();
                fwht128_inplace(group, &*SIGNS128_1, &*SIGNS128_2);
            }
            for (a, b) in r.iter().zip(&x) {
                assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
            }
        }
    }

    /// `rotate_input` picks the rotation the format's stored codes need, and
    /// leaves natural-basis formats alone.
    #[test]
    fn rotate_input_dispatches_on_the_format() {
        let x = lcg_floats(9, 256);
        let apply = |q: CpuQuant| {
            let mut v = x.clone();
            rotate_input(q, &mut v);
            v
        };
        let mut g256 = x.clone();
        rotate_x(&mut g256);
        let mut g128 = x.clone();
        rotate_x_128_v2(&mut g128);
        assert_eq!(apply(CpuQuant::Mq4G256V2), g256);
        assert_eq!(apply(CpuQuant::Mq4G128V2), g128);
        assert_ne!(g128, g256);
        assert_eq!(apply(CpuQuant::Hfq4G128), x);
        assert_eq!(apply(CpuQuant::Mq2G256LloydU), x);
    }
}
