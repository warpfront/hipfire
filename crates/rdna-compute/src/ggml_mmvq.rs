// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Launchers of the GGUF-native decode GEMV set for gfx1201 (`HIPFIRE_GGUF_NATIVE=1`):
//! the `q8_1` activation producers and the quantised-weight mmvq GEMVs of two embedded
//! objects, both built from the pinned upstream ggml snapshot in `third_party/ggml/`:
//!
//! * `kernels/ggml_mmvq_shim_gfx1201.co` (S1): `ggml_q8_1_quantize_gfx1201` and
//!   `ggml_mmvq_{iq4_xs,q4_k,q6_k,q8_0}_gfx1201` (`dst[row] = dot`);
//! * `kernels/ggml_mmvq_fused_shim_gfx1201.co` (S3): `ggml_mmvq_q5_k_gfx1201`,
//!   `ggml_mmvq_{iq4_xs,q4_k,q5_k,q6_k,q8_0}_res_gfx1201` (`dst[row] = dst[row] + dot`,
//!   the residual-stream epilogue) and `ggml_q8_1_quantize_gather_gfx1201` (the producer
//!   with a 32-element block permutation).
//!
//! Numeric contracts `ggml.a8q81.<type>` (`.res` for the accumulating exports); the typed
//! contracts are `hipfire_kernel_abi::families::{ggml_mmvq, ggml_mmvq_fused}`. Every
//! launch is validated against its contract (dtype, `K` multiple of the block, operand
//! formats / alignment / byte extents, the `M * K / qk <= i32::MAX` address domain, the
//! launch geometry) BEFORE anything is loaded or launched; no export uses atomics, so
//! the results are deterministic and capture/record safe (no host sync, no allocation on
//! the success path beyond the contract-checked kernarg image).
//!
//! Image format: both objects are clang offload bundles (`__CLANG_OFFLOAD_BUNDLE__`,
//! `--no-offload-compress`) exactly as hipcc emits them; they are embedded unchanged and
//! handed to `module_load_data`, the same path the hipcc compile cache takes.

use crate::kernel_abi;
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use hipfire_kernel_abi::families::{ggml_mmvq as s1, ggml_mmvq_fused as s3};
use hipfire_kernel_abi::{
    Arch, FormatId, InvocationV1, OperandBinding, ValidatedInvocationV1, MAX_FIELDS,
};
use std::ffi::c_void;

/// The S1 object (producer + the four non-accumulating mmvq exports).
pub const S1_MODULE: &str = "ggml_mmvq_shim_gfx1201";
pub const S1_IMAGE: &[u8] = include_bytes!("../../../kernels/ggml_mmvq_shim_gfx1201.co");
/// The S3 object (`q5_k`, the `_res` exports, the gather producer).
pub const FUSED_MODULE: &str = "ggml_mmvq_fused_shim_gfx1201";
pub const FUSED_IMAGE: &[u8] = include_bytes!("../../../kernels/ggml_mmvq_fused_shim_gfx1201.co");

pub const PRODUCER_SYMBOL: &str = "ggml_q8_1_quantize_gfx1201";
pub const GATHER_SYMBOL: &str = "ggml_q8_1_quantize_gather_gfx1201";
/// Producer launch block, 256 threads along x.
pub const PRODUCER_BLOCK: [u32; 3] = [256, 1, 1];
/// mmvq launch block: `(warp_size, nwarps)`.
pub const MMVQ_BLOCK: [u32; 3] = [32, 8, 1];
/// Elements per `block_q8_1`.
const QK8_1: usize = 32;
/// Bytes per `block_q8_1` (`half2 ds` + `int8 qs[32]`).
const Q8_1_BLOCK_BYTES: usize = 36;

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("ggml_mmvq: {}", message.as_ref()))
}

/// Which embedded object holds an export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Object {
    S1,
    Fused,
}

impl Object {
    pub fn module(self) -> &'static str {
        match self {
            Object::S1 => S1_MODULE,
            Object::Fused => FUSED_MODULE,
        }
    }
    pub fn image(self) -> &'static [u8] {
        match self {
            Object::S1 => S1_IMAGE,
            Object::Fused => FUSED_IMAGE,
        }
    }
}

/// The export of one `(dtype, accumulate)` pair and everything its contract names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmvqExport {
    pub symbol: &'static str,
    pub object: Object,
    /// Elements per weight block (`qk`): 256, `Q8_0` 32.
    pub qk: usize,
    /// Bytes per weight block.
    pub block_bytes: usize,
    pub weight_format: FormatId,
}

/// The mmvq export serving `dtype`. `accumulate = false`: `y[row] = dot`; `true`:
/// `y[row] = y[row] + dot` (the `_res` exports, the residual stream is `y`). Any dtype
/// that is not one of the five ggml weight dtypes is an error naming it.
pub fn mmvq_export(dtype: DType, accumulate: bool) -> HipResult<MmvqExport> {
    let (qk, block_bytes) = dtype
        .ggml_block()
        .ok_or_else(|| err(format!("dtype {dtype:?} is not a ggml weight dtype (GgmlIQ4XS, GgmlQ4K, GgmlQ5K, GgmlQ6K, GgmlQ8_0)")))?;
    // (non-accumulating symbol, accumulating symbol, format, object of the non-accumulating one)
    let (plain, res, weight_format, plain_object) = match dtype {
        DType::GgmlIQ4XS => (
            "ggml_mmvq_iq4_xs_gfx1201",
            "ggml_mmvq_iq4_xs_res_gfx1201",
            s1::FORMAT_IQ4_XS_136,
            Object::S1,
        ),
        DType::GgmlQ4K => (
            "ggml_mmvq_q4_k_gfx1201",
            "ggml_mmvq_q4_k_res_gfx1201",
            s1::FORMAT_Q4_K_144,
            Object::S1,
        ),
        DType::GgmlQ5K => (
            "ggml_mmvq_q5_k_gfx1201",
            "ggml_mmvq_q5_k_res_gfx1201",
            s3::FORMAT_Q5_K_176,
            Object::Fused,
        ),
        DType::GgmlQ6K => (
            "ggml_mmvq_q6_k_gfx1201",
            "ggml_mmvq_q6_k_res_gfx1201",
            s1::FORMAT_Q6_K_210,
            Object::S1,
        ),
        DType::GgmlQ8_0 => (
            "ggml_mmvq_q8_0_gfx1201",
            "ggml_mmvq_q8_0_res_gfx1201",
            s1::FORMAT_Q8_0_34,
            Object::S1,
        ),
        other => return Err(err(format!("dtype {other:?} has no mmvq export"))),
    };
    let (symbol, object) = if accumulate {
        (res, Object::Fused)
    } else {
        (plain, plain_object)
    };
    Ok(MmvqExport {
        symbol,
        object,
        qk,
        block_bytes,
        weight_format,
    })
}

fn require_gfx1201(arch: &str) -> HipResult<Arch> {
    if arch != "gfx1201" {
        return Err(err(format!("requires gfx1201, this device is {arch}")));
    }
    kernel_abi::parse_arch(arch)
}

fn require_f32(t: &GpuTensor, name: &str) -> HipResult<()> {
    if t.dtype != DType::F32 {
        return Err(err(format!("{name} must be F32, is {:?}", t.dtype)));
    }
    Ok(())
}

/// Producer grid: `(ceil(k / 256), rows, 1)`.
pub fn producer_grid(k: usize, rows: usize) -> [u32; 3] {
    [k.div_ceil(256) as u32, rows as u32, 1]
}

/// Gather producer grid: `(ceil(k / 256), 1, 1)`.
pub fn gather_grid(k: usize) -> [u32; 3] {
    [k.div_ceil(256) as u32, 1, 1]
}

/// mmvq grid: one weight row per workgroup.
pub fn mmvq_grid(m: usize) -> [u32; 3] {
    [m as u32, 1, 1]
}

/// Contract-validate the S1 producer launch: `x` (`rows * k` f32, row stride `k`) into
/// `vy` (`rows * k / 32` blocks).
fn validate_producer<'a>(
    arch: Arch,
    k: usize,
    rows: usize,
    x: OperandBinding,
    vy: OperandBinding,
) -> HipResult<ValidatedInvocationV1<'a>> {
    let abi = &s1::GGML_Q8_1_PRODUCER;
    // params: K (= ne00), S01, NE0, NROWS
    kernel_abi::validate(
        abi,
        &InvocationV1 {
            arch,
            symbol: PRODUCER_SYMBOL,
            params: &[k as u64, k as u64, k as u64, rows as u64],
            operands: &[x, vy],
            grid: producer_grid(k, rows),
            block: PRODUCER_BLOCK,
            dynamic_lds: 0,
        },
    )
}

fn validate_gather<'a>(
    arch: Arch,
    k: usize,
    x: OperandBinding,
    vy: OperandBinding,
    map: OperandBinding,
) -> HipResult<ValidatedInvocationV1<'a>> {
    kernel_abi::validate(
        &s3::GGML_Q8_1_GATHER,
        &InvocationV1 {
            arch,
            symbol: GATHER_SYMBOL,
            params: &[k as u64],
            operands: &[x, vy, map],
            grid: gather_grid(k),
            block: PRODUCER_BLOCK,
            dynamic_lds: 0,
        },
    )
}

/// Contract-validate an mmvq launch of `export` for `m` rows of `k` elements.
fn validate_mmvq<'a>(
    arch: Arch,
    export: &MmvqExport,
    m: usize,
    k: usize,
    w: OperandBinding,
    q81: OperandBinding,
    y: OperandBinding,
) -> HipResult<ValidatedInvocationV1<'a>> {
    let abi = kernel_abi::contract("gfx1201", export.symbol)?;
    kernel_abi::validate(
        abi,
        &InvocationV1 {
            arch,
            symbol: export.symbol,
            params: &[m as u64, k as u64],
            operands: &[w, q81, y],
            grid: mmvq_grid(m),
            block: MMVQ_BLOCK,
            dynamic_lds: 0,
        },
    )
}

/// Launch a validated invocation of `symbol` from `object`.
fn ggml_mmvq_launch(
    gpu: &mut Gpu,
    object: Object,
    symbol: &'static str,
    mut v: ValidatedInvocationV1<'_>,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_embedded_kernel(object.module(), object.image(), symbol)?;
    let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
    let n = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
    let (grid, block) = (v.grid(), v.block());
    gpu.launch_maybe_blob(symbol, grid, block, 0, &mut ptrs[..n], || {
        kernel_abi::blob(v.explicit())
    })
}

impl Gpu {
    /// Bytes of one q8_1 activation row of `k` elements: `k / 32 * 36`.
    pub fn ggml_q81_row_bytes(k: usize) -> usize {
        k / QK8_1 * Q8_1_BLOCK_BYTES
    }

    /// Quantise `rows` F32 activation rows of `k` elements (`x` row `i` at element
    /// `i * k`, contiguous) to the mmvq `q8_1` layout in `q81` (`rows * k / 32` blocks of
    /// 36 B, raw bytes) with `ggml_q8_1_quantize_gfx1201` (`s01 = ne0 = k`). `k % 32 == 0`.
    pub fn ggml_q8_1_quantize(
        &mut self,
        x: &GpuTensor,
        q81: &GpuTensor,
        k: usize,
        rows: usize,
    ) -> HipResult<()> {
        let arch = require_gfx1201(&self.arch)?;
        require_f32(x, "x")?;
        if k == 0 || k % QK8_1 != 0 {
            return Err(err(format!(
                "quantize: k = {k} must be a positive multiple of 32"
            )));
        }
        if rows == 0 {
            return Err(err("quantize: rows must be >= 1"));
        }
        let v = validate_producer(
            arch,
            k,
            rows,
            kernel_abi::binding(x, s1::FORMAT_F32_ROWS_S01),
            kernel_abi::binding(q81, s1::FORMAT_Q8_1_MMVQ),
        )?;
        ggml_mmvq_launch(self, Object::S1, PRODUCER_SYMBOL, v)
    }

    /// Single-row quantiser with a block permutation: `q81` block `j` (32 elements) is
    /// quantised from `x[blk_map[j] * 32 .. + 32]`. `blk_map` is `k / 32` int32 entries
    /// (device tensor, raw words), every entry in `[0, k / 32)` (data-dependent: the
    /// caller's obligation). With the identity map the bytes equal
    /// [`Gpu::ggml_q8_1_quantize`]'s for one row.
    pub fn ggml_q8_1_quantize_gather(
        &mut self,
        x: &GpuTensor,
        q81: &GpuTensor,
        blk_map: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        let arch = require_gfx1201(&self.arch)?;
        require_f32(x, "x")?;
        if k == 0 || k % QK8_1 != 0 {
            return Err(err(format!(
                "gather quantize: k = {k} must be a positive multiple of 32"
            )));
        }
        let v = validate_gather(
            arch,
            k,
            kernel_abi::binding(x, s1::FORMAT_F32_ROWS_S01),
            kernel_abi::binding(q81, s1::FORMAT_Q8_1_MMVQ),
            kernel_abi::binding(blk_map, s3::FORMAT_I32_BLK_MAP),
        )?;
        ggml_mmvq_launch(self, Object::Fused, GATHER_SYMBOL, v)
    }

    /// `y = W x` (`accumulate = false`) or `y += W x` (`accumulate = true`: `y[row] = y[row]
    /// + dot`, the residual stream) for a ggml-format weight `w` (`m` rows of `k`
    /// elements, whole blocks, raw bytes; trailing bytes are ignored), the q8_1 row `q81`
    /// of the activation (one row from a producer) and the F32 output `y` (`m`). `dtype`
    /// is one of `GgmlIQ4XS`, `GgmlQ4K`, `GgmlQ5K`, `GgmlQ6K`, `GgmlQ8_0`; `k` a multiple
    /// of the block (256, `Q8_0` 32). Buffer byte lengths (`w >= m * row bytes`, `q81 >=
    /// k / 32 * 36`, `y >= m * 4`) and `m * (k / qk) <= i32::MAX` are checked before the
    /// launch.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_mmvq(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        q81: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        accumulate: bool,
    ) -> HipResult<()> {
        let export = mmvq_export(dtype, accumulate)?;
        let arch = require_gfx1201(&self.arch)?;
        if k == 0 || k % export.qk != 0 {
            return Err(err(format!(
                "{dtype:?}: k = {k} must be a positive multiple of the block ({})",
                export.qk
            )));
        }
        if m == 0 {
            return Err(err("m must be >= 1"));
        }
        require_f32(y, "y")?;
        let v = validate_mmvq(
            arch,
            &export,
            m,
            k,
            kernel_abi::binding(w, export.weight_format),
            kernel_abi::binding(q81, s1::FORMAT_Q8_1_MMVQ),
            kernel_abi::binding(y, s1::FORMAT_F32_M),
        )?;
        ggml_mmvq_launch(self, export.object, export.symbol, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_kernel_abi::AbiError;

    const ALL: [DType; 5] = [
        DType::GgmlIQ4XS,
        DType::GgmlQ4K,
        DType::GgmlQ5K,
        DType::GgmlQ6K,
        DType::GgmlQ8_0,
    ];

    fn bind(address: u64, bytes: u64, format: FormatId) -> OperandBinding {
        OperandBinding {
            address,
            bytes: Some(bytes),
            format,
        }
    }

    fn mmvq_bindings(e: &MmvqExport, m: usize, k: usize) -> [OperandBinding; 3] {
        [
            bind(
                0x1000_0000,
                (m * k / e.qk * e.block_bytes) as u64,
                e.weight_format,
            ),
            bind(
                0x4000_0000,
                Gpu::ggml_q81_row_bytes(k) as u64,
                s1::FORMAT_Q8_1_MMVQ,
            ),
            bind(0x5000_0000, (m * 4) as u64, s1::FORMAT_F32_M),
        ]
    }

    #[test]
    fn images_are_the_two_committed_bundles() {
        assert!(S1_IMAGE.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"));
        assert!(FUSED_IMAGE.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"));
        assert_ne!(S1_IMAGE, FUSED_IMAGE);
    }

    #[test]
    fn export_table_covers_the_five_dtypes_both_epilogues() {
        let want = [
            (
                DType::GgmlIQ4XS,
                "ggml_mmvq_iq4_xs_gfx1201",
                Object::S1,
                256,
                136,
            ),
            (
                DType::GgmlQ4K,
                "ggml_mmvq_q4_k_gfx1201",
                Object::S1,
                256,
                144,
            ),
            (
                DType::GgmlQ5K,
                "ggml_mmvq_q5_k_gfx1201",
                Object::Fused,
                256,
                176,
            ),
            (
                DType::GgmlQ6K,
                "ggml_mmvq_q6_k_gfx1201",
                Object::S1,
                256,
                210,
            ),
            (
                DType::GgmlQ8_0,
                "ggml_mmvq_q8_0_gfx1201",
                Object::S1,
                32,
                34,
            ),
        ];
        for (dtype, symbol, object, qk, bytes) in want {
            let plain = mmvq_export(dtype, false).unwrap();
            assert_eq!(
                (plain.symbol, plain.object, plain.qk, plain.block_bytes),
                (symbol, object, qk, bytes),
                "{dtype:?}"
            );
            let res = mmvq_export(dtype, true).unwrap();
            assert_eq!(res.object, Object::Fused, "{dtype:?} res");
            assert_eq!(
                res.symbol,
                symbol.replace("_gfx1201", "_res_gfx1201"),
                "{dtype:?} res"
            );
            assert_eq!((res.qk, res.block_bytes), (qk, bytes));
            // every export has a contract owning the symbol in the object the table names
            for e in [plain, res] {
                let abi = kernel_abi::contract("gfx1201", e.symbol).unwrap();
                let ex = abi.export(Arch::Gfx1201, e.symbol).unwrap();
                assert_eq!(ex.block, MMVQ_BLOCK, "{}", e.symbol);
                assert_eq!(ex.group_segment_fixed, 896);
            }
        }
    }

    #[test]
    fn other_dtypes_are_refused_by_name() {
        for dtype in [DType::F32, DType::MQ4G256V2, DType::Q4K, DType::HFQ4G256] {
            for acc in [false, true] {
                let e = mmvq_export(dtype, acc).unwrap_err().to_string();
                assert!(e.contains(&format!("{dtype:?}")), "{e}");
            }
        }
    }

    #[test]
    fn frozen_kernarg_abi_and_grid() {
        // vx@0 vy@8 dst@16 K@24, 28 explicit bytes, grid (M,1,1), block (32,8,1)
        for dtype in ALL {
            for acc in [false, true] {
                let e = mmvq_export(dtype, acc).unwrap();
                let (m, k) = (2560usize, 5120usize);
                let [w, q, y] = mmvq_bindings(&e, m, k);
                let v = validate_mmvq(Arch::Gfx1201, &e, m, k, w, q, y).unwrap();
                let a = v.explicit();
                assert_eq!(a.len(), 28, "{}", e.symbol);
                assert_eq!(u64::from_le_bytes(a[0..8].try_into().unwrap()), 0x1000_0000);
                assert_eq!(
                    u64::from_le_bytes(a[8..16].try_into().unwrap()),
                    0x4000_0000
                );
                assert_eq!(
                    u64::from_le_bytes(a[16..24].try_into().unwrap()),
                    0x5000_0000
                );
                assert_eq!(i32::from_le_bytes(a[24..28].try_into().unwrap()), 5120);
                assert_eq!(v.grid(), [2560, 1, 1]);
                assert_eq!(v.block(), [32, 8, 1]);
                assert_eq!(v.dynamic_lds(), 0);
                assert_eq!(kernel_abi::blob(a).as_bytes()[..28], *a);
            }
        }
        assert_eq!(mmvq_grid(17408), [17408, 1, 1]);
    }

    #[test]
    fn producer_kernarg_abi_and_grid() {
        // x@0 vy@8 ne00@16 s01@20 ne0@24; grid (ceil(k/256), rows, 1); block 256
        let (k, rows) = (5120usize, 3usize);
        let v = validate_producer(
            Arch::Gfx1201,
            k,
            rows,
            bind(0x1000, (rows * k * 4) as u64, s1::FORMAT_F32_ROWS_S01),
            bind(
                0x2000,
                (rows * Gpu::ggml_q81_row_bytes(k)) as u64,
                s1::FORMAT_Q8_1_MMVQ,
            ),
        )
        .unwrap();
        let a = v.explicit();
        assert_eq!(a.len(), 28);
        assert_eq!(u64::from_le_bytes(a[0..8].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(a[8..16].try_into().unwrap()), 0x2000);
        for off in [16, 20, 24] {
            assert_eq!(
                i32::from_le_bytes(a[off..off + 4].try_into().unwrap()),
                5120
            );
        }
        assert_eq!((v.grid(), v.block()), ([20, 3, 1], [256, 1, 1]));
        assert_eq!(producer_grid(17408, 1), [68, 1, 1]);
        assert_eq!(producer_grid(32, 1), [1, 1, 1]);
        assert_eq!(producer_grid(8224, 1), [33, 1, 1]);
    }

    #[test]
    fn gather_kernarg_abi_and_grid() {
        // x@0 vy@8 blk_map@16 ne0@24; grid (ceil(k/256),1,1); block 256
        let k = 6144usize;
        let v = validate_gather(
            Arch::Gfx1201,
            k,
            bind(0x1000, (k * 4) as u64, s1::FORMAT_F32_ROWS_S01),
            bind(
                0x2000,
                Gpu::ggml_q81_row_bytes(k) as u64,
                s1::FORMAT_Q8_1_MMVQ,
            ),
            bind(0x3000, (k / 32 * 4) as u64, s3::FORMAT_I32_BLK_MAP),
        )
        .unwrap();
        let a = v.explicit();
        assert_eq!(a.len(), 28);
        assert_eq!(u64::from_le_bytes(a[0..8].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(a[8..16].try_into().unwrap()), 0x2000);
        assert_eq!(u64::from_le_bytes(a[16..24].try_into().unwrap()), 0x3000);
        assert_eq!(i32::from_le_bytes(a[24..28].try_into().unwrap()), 6144);
        assert_eq!((v.grid(), v.block()), ([24, 1, 1], [256, 1, 1]));
        assert_eq!(gather_grid(5120), [20, 1, 1]);
        assert_eq!(gather_grid(32), [1, 1, 1]);
        assert_eq!(gather_grid(288), [2, 1, 1]);
    }

    #[test]
    fn row_bytes_of_the_q81_activation() {
        assert_eq!(Gpu::ggml_q81_row_bytes(32), 36);
        assert_eq!(Gpu::ggml_q81_row_bytes(5120), 5760);
        assert_eq!(Gpu::ggml_q81_row_bytes(17408), 19584);
    }

    #[test]
    fn short_buffers_ragged_k_and_oversized_domain_are_refused_before_launch() {
        for dtype in ALL {
            let e = mmvq_export(dtype, true).unwrap();
            let (m, k) = (37usize, 768usize);
            let ok = mmvq_bindings(&e, m, k);
            validate_mmvq(Arch::Gfx1201, &e, m, k, ok[0], ok[1], ok[2]).unwrap();
            // one byte short of each operand
            for i in 0..3 {
                let mut b = ok;
                b[i].bytes = b[i].bytes.map(|n| n - 1);
                match validate_mmvq(Arch::Gfx1201, &e, m, k, b[0], b[1], b[2]) {
                    Err(error) => assert!(
                        error.to_string().contains(["vx", "vy", "dst"][i]),
                        "{dtype:?} operand {i}: {error}"
                    ),
                    Ok(_) => panic!("{dtype:?}: operand {i} one byte short was admitted"),
                }
            }
            // a K that is not a whole number of blocks
            let bad_k = k + e.qk / 2;
            assert!(
                validate_mmvq(Arch::Gfx1201, &e, m, bad_k, ok[0], ok[1], ok[2]).is_err(),
                "{dtype:?}"
            );
            // M * (K / qk) past i32::MAX (the kernel indexes blocks in int)
            let big_m = (i32::MAX as usize) / (k / e.qk) + 1;
            let big = mmvq_bindings(&e, big_m, k); // extents sized for big_m: only the domain rule can fail
            let r = validate_mmvq(Arch::Gfx1201, &e, big_m, k, big[0], big[1], big[2]);
            assert!(r.is_err(), "{dtype:?}");
        }
        // the contract names the failure
        let e = mmvq_export(DType::GgmlQ5K, false).unwrap();
        let ok = mmvq_bindings(&e, 8, 512);
        let _: AbiError = hipfire_kernel_abi::validate_invocation(
            kernel_abi::contract("gfx1201", e.symbol).unwrap(),
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: e.symbol,
                params: &[8, 500],
                operands: &ok,
                grid: [8, 1, 1],
                block: MMVQ_BLOCK,
                dynamic_lds: 0,
            },
        )
        .unwrap_err();
    }

    #[test]
    fn misaligned_weights_are_refused() {
        // q5_k needs 4-byte alignment, q6_k / q8_0 2-byte
        let (m, k) = (4usize, 512usize);
        for (dtype, addr, ok) in [
            (DType::GgmlQ5K, 0x1000_0002u64, false),
            (DType::GgmlQ6K, 0x1000_0002, true),
            (DType::GgmlQ6K, 0x1000_0001, false),
            (DType::GgmlQ8_0, 0x1000_0002, true),
        ] {
            let e = mmvq_export(dtype, false).unwrap();
            let mut b = mmvq_bindings(&e, m, k);
            b[0].address = addr;
            let r = validate_mmvq(Arch::Gfx1201, &e, m, k, b[0], b[1], b[2]);
            assert_eq!(r.is_ok(), ok, "{dtype:?} at {addr:#x}");
        }
    }

    #[test]
    fn wrong_format_is_refused() {
        let e = mmvq_export(DType::GgmlQ5K, true).unwrap();
        let mut b = mmvq_bindings(&e, 4, 512);
        b[0].format = s1::FORMAT_Q4_K_144;
        assert!(validate_mmvq(Arch::Gfx1201, &e, 4, 512, b[0], b[1], b[2]).is_err());
    }

    #[test]
    fn only_gfx1201_is_served() {
        assert!(require_gfx1201("gfx1201").is_ok());
        for arch in ["gfx1100", "gfx1151", "gfx90a"] {
            assert!(require_gfx1201(arch)
                .unwrap_err()
                .to_string()
                .contains(arch));
        }
    }
}
