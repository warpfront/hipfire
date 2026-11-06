// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Launch-side glue to `hipfire-kernel-abi` (PM/railgun bedrock R1): contract
//! lookup, invocation validation with `HipError` mapping, the single
//! `KernargBlob` encoder and operand bindings. Allocation-free on the success
//! path; the error paths format a message.

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use hipfire_kernel_abi::families::iu4_gdn as gdn;
use hipfire_kernel_abi::families::qwen4_moe_sym as moe;
use hipfire_kernel_abi::{
    families, Arch, FormatId, InvocationV1, KernelAbiV1, OperandBinding, ValidatedInvocationV1,
};
use std::ffi::c_void;

/// Exact-arch parse of the device's `gfx...` name.
pub(crate) fn parse_arch(arch: &str) -> HipResult<Arch> {
    Arch::parse(arch).ok_or_else(|| {
        HipError::new(
            0,
            &format!("kernel ABI: no contract family targets arch `{arch}`"),
        )
    })
}

/// The contract that owns `symbol` on `arch`.
pub(crate) fn contract(arch: &str, symbol: &str) -> HipResult<&'static KernelAbiV1> {
    let parsed = parse_arch(arch)?;
    families::find_export(parsed, symbol)
        .map(|(abi, _)| abi)
        .ok_or_else(|| HipError::new(0, &format!("{symbol}: no KernelAbiV1 export for {arch}")))
}

/// [`hipfire_kernel_abi::validate_invocation`] with the rejection mapped to a
/// `HipError` naming the symbol.
pub(crate) fn validate<'a>(
    abi: &'a KernelAbiV1,
    inv: &InvocationV1<'_>,
) -> HipResult<ValidatedInvocationV1<'a>> {
    hipfire_kernel_abi::validate_invocation(abi, inv)
        .map_err(|e| HipError::new(0, &format!("{}: {e}", inv.symbol)))
}

/// One pointer per explicit field into `v`'s image: the ordinary
/// `kernelParams` array. `v` must not move while `out` is in use.
pub(crate) fn kernel_params(
    v: &mut ValidatedInvocationV1<'_>,
    out: &mut [*mut c_void],
) -> HipResult<usize> {
    let symbol = v.export().symbol;
    v.kernel_params(out)
        .map_err(|e| HipError::new(0, &format!("{symbol}: {e}")))
}

/// The explicit kernarg image as a `KernargBlob` (graph-capture path): one
/// 4-byte word per push, so the bytes equal the explicit image exactly.
pub(crate) fn blob(explicit: &[u8]) -> KernargBlob {
    debug_assert!(
        explicit.len() % 4 == 0,
        "explicit kernarg image is word-sized"
    );
    let mut b = KernargBlob::with_capacity(explicit.len());
    for word in explicit.chunks_exact(4) {
        b.push_u32(u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
    }
    b
}

/// Binding of a whole tensor: address = buffer base, bytes = buffer size.
pub(crate) fn binding(t: &GpuTensor, format: FormatId) -> OperandBinding {
    binding_raw(t.buf.as_ptr(), Some(t.buf.size() as u64), format)
}

/// Binding of a raw device pointer. `bytes = None` is recorded by the
/// validator as an unchecked extent, never treated as proven.
pub(crate) fn binding_raw(
    ptr: *const c_void,
    bytes: Option<u64>,
    format: FormatId,
) -> OperandBinding {
    OperandBinding {
        address: ptr as usize as u64,
        bytes,
        format,
    }
}

/// Weight codec of a `qwen4_moe_sym` grouped GEMM launch: selects the expert
/// pointer-table format (the contract of the entry names the same one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoeSymCodec {
    /// Symmetric MQ4: QT44 gate/up slabs, QT53 down slabs.
    Mq4,
    /// Sealed-slab E8P-I4 planar tensors.
    E8p,
    /// MQ3-I4 (W3) planar tensors.
    W3,
    /// Drop-zero planar MQ4 tensors (`MQ4G256V2P` gate/up, `MQ4G128V2P` down, DESIGN F1).
    Mq4Planar,
}

/// Storage formats of the five operands of a `qwen4_moe_sym` launch, in contract
/// order (`expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq,
/// Y_grouped`): by weight codec, role and (down) activation-plane layout.
pub(crate) fn moe_sym_formats(
    codec: MoeSymCodec,
    down: bool,
    grouped_plane: bool,
) -> [FormatId; 5] {
    let table = match (codec, down) {
        (MoeSymCodec::Mq4, false) => moe::FORMAT_EXPERT_PTRS_GATE_UP_QT44,
        (MoeSymCodec::Mq4, true) => moe::FORMAT_EXPERT_PTRS_DOWN_QT53,
        (MoeSymCodec::E8p, _) => moe::FORMAT_EXPERT_PTRS_E8P_PLANAR,
        (MoeSymCodec::W3, _) => moe::FORMAT_EXPERT_PTRS_W3_PLANAR,
        (MoeSymCodec::Mq4Planar, _) => moe::FORMAT_EXPERT_PTRS_MQ4_PLANAR,
    };
    let (x_format, y_format) = if !down {
        (
            moe::FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
            moe::FORMAT_Y_GATE_UP_SWIGLU_BF16,
        )
    } else if grouped_plane {
        (
            moe::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION,
            moe::FORMAT_Y_DOWN_BF16,
        )
    } else {
        (
            moe::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS,
            moe::FORMAT_Y_DOWN_BF16,
        )
    };
    [
        table,
        moe::FORMAT_EXPERT_TILE_IDS,
        moe::FORMAT_SORTED_SLOT_INDEX,
        x_format,
        y_format,
    ]
}

/// The five operand bindings of a `qwen4_moe_sym` launch ([`moe_sym_formats`]
/// order). `xq_bytes` is `Some` only when `xq` is a live scratch whose size is
/// known.
pub(crate) fn moe_sym_bindings(
    codec: MoeSymCodec,
    down: bool,
    grouped_plane: bool,
    [ptrs, tiles, sorted, y]: [&GpuTensor; 4],
    xq: *mut c_void,
    xq_bytes: Option<u64>,
) -> [OperandBinding; 5] {
    let [table, tile_ids, slots, x_format, y_format] = moe_sym_formats(codec, down, grouped_plane);
    [
        binding(ptrs, table),
        binding(tiles, tile_ids),
        binding(sorted, slots),
        binding_raw(xq, xq_bytes, x_format),
        binding(y, y_format),
    ]
}

/// Validate one `qwen4_moe_sym` launch of `func` (the 64-byte, six-param ABI
/// of every export of the family): `params` = `[M, K, x_row_div, m_total,
/// x_src_rows, n_experts]`, one-dimensional `block`, no dynamic LDS.
pub(crate) fn validate_moe_sym(
    arch: &str,
    func: &str,
    params: [u64; 6],
    operands: &[OperandBinding; 5],
    grid: [u32; 3],
    block: u32,
) -> HipResult<ValidatedInvocationV1<'static>> {
    let parsed = parse_arch(arch)?;
    let abi = contract(arch, func)?;
    validate(
        abi,
        &InvocationV1 {
            arch: parsed,
            symbol: func,
            params: &params,
            operands,
            grid,
            block: [block, 1, 1],
            dynamic_lds: 0,
        },
    )
}

/// Symbol of the fused A4 GDN input projection for token (`_b1`) or slab (`_b1s`) activations.
pub(crate) fn iu4_gdn_symbol(slab: bool) -> &'static str {
    if slab {
        "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s"
    } else {
        "gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1"
    }
}

/// The twelve operand bindings of a fused A4 GDN input projection in contract order (`Wqkv, Wz, Xq,
/// Yz, Ybeta, Yalpha, ConvW, ConvState, Q, K16, V, X`), each `(device address, bytes bound)`: formats
/// by activation layout.
pub(crate) fn iu4_gdn_bindings(
    slab: bool,
    raw: [(*const c_void, Option<u64>); 12],
) -> [OperandBinding; 12] {
    use hipfire_kernel_abi::families::generic_gemm::FORMAT_BLOCK_I4_128_KB_N;
    use hipfire_kernel_abi::families::mq6_i8::FORMAT_F32_N_M;
    let formats = [
        gdn::FORMAT_MQ4G256V2_WEIGHTS,
        gdn::FORMAT_MQ4G256V2_Z_FOLD,
        if slab {
            gdn::FORMAT_BLOCK_I4_128_SLAB
        } else {
            FORMAT_BLOCK_I4_128_KB_N
        },
        FORMAT_F32_N_M,
        FORMAT_F32_N_M,
        FORMAT_F32_N_M,
        gdn::FORMAT_F32_CONV_WEIGHT_4,
        gdn::FORMAT_F32_CONV_STATE_3,
        gdn::FORMAT_F16_TOKEN_HEADS_128,
        gdn::FORMAT_F16_TOKEN_HEADS_128,
        gdn::FORMAT_F16_TOKEN_HEADS_128,
        gdn::FORMAT_F32_QKV_RAW_ROWS,
    ];
    std::array::from_fn(|i| binding_raw(raw[i].0, raw[i].1, formats[i]))
}

/// The contract params of a fused A4 GDN input projection (`iu4_gdn::PARAMS` order: `M0, Mz, Mbeta,
/// Malpha, K, N, QScaleBits, EpsBits`).
pub(crate) fn iu4_gdn_params(
    dims: [usize; 4],
    k: usize,
    n: usize,
    q_scale: f32,
    eps: f32,
) -> [u64; 8] {
    [
        dims[0] as u64,
        dims[1] as u64,
        dims[2] as u64,
        dims[3] as u64,
        k as u64,
        n as u64,
        u64::from(q_scale.to_bits()),
        u64::from(eps.to_bits()),
    ]
}

/// Validate one fused A4 GDN input projection launch: the 120-byte explicit image, grid, block and
/// dynamic LDS are the contract's (`iu4_gemm.qkvzagdn.b1` / `.b1s`).
pub(crate) fn validate_iu4_gdn(
    arch: &str,
    slab: bool,
    params: [u64; 8],
    operands: &[OperandBinding; 12],
    grid: [u32; 3],
) -> HipResult<ValidatedInvocationV1<'static>> {
    let parsed = parse_arch(arch)?;
    let symbol = iu4_gdn_symbol(slab);
    let abi = contract(arch, symbol)?;
    validate(
        abi,
        &InvocationV1 {
            arch: parsed,
            symbol,
            params: &params,
            operands,
            grid,
            block: [256, 1, 1],
            dynamic_lds: gdn::DYNAMIC_LDS,
        },
    )
}

impl Gpu {
    /// Bytes bound at `xq` when it is the live int4 MMQ activation scratch (its whole allocation);
    /// `None` for any other pointer, which the validator records as an unchecked extent.
    pub(crate) fn int4_mmq_xq_bytes(&self, xq: *mut c_void) -> Option<u64> {
        self.scratch
            .int4_mmq_x_scratch
            .as_ref()
            .filter(|b| b.as_ptr() == xq)
            .map(|b| b.size() as u64)
    }

    /// Bytes bound at `xq` when it is the live INT8 MMQ activation scratch
    /// (its whole allocation); `None` for any other pointer, which the
    /// validator records as an unchecked extent.
    pub(crate) fn int8_mmq_xq_bytes(&self, xq: *mut c_void) -> Option<u64> {
        self.scratch
            .int8_mmq_x_scratch
            .as_ref()
            .filter(|b| b.as_ptr() == xq)
            .map(|b| b.size() as u64)
    }

    /// Bytes bound at `xq` of a `qwen4_moe_sym` GEMM when it is the live
    /// activation scratch that role reads (gate/up: the shared int4 MMQ
    /// scratch; down: the down-sidecar / grouped-plane scratch); else `None`.
    pub(crate) fn moe_sym_xq_bytes(&self, down: bool, xq: *mut c_void) -> Option<u64> {
        let scratch = if down {
            &self.scratch.qwen4_moe_down_i4_scratch
        } else {
            &self.scratch.int4_mmq_x_scratch
        };
        scratch
            .as_ref()
            .filter(|b| b.as_ptr() == xq)
            .map(|b| b.size() as u64)
    }
}
