// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Launchers of the GGUF-native convert set for gfx1201 (`HIPFIRE_GGUF_NATIVE=1`), the object
//! `kernels/ggml_convert_shim_gfx1201.co` built from the pinned upstream ggml snapshot in
//! `third_party/ggml/` by `scripts/build-ggml-convert-shim.sh`:
//!
//! * block -> F16 / F32 dequantisation of a whole `m x k` weight tensor (`ggml_dequant_*`), with an
//!   optional permutation of its 128-element column chunks and, for F32, an optional bfloat16
//!   round-to-nearest-even of every element (the codec-only quality tier);
//! * the embedding row gather (`ggml_get_row*`): token ids from a device tensor, or one token by
//!   value (the decode path patches it like the engine's other embedding kernels);
//! * the in-place A8 activation round trip (`ggml_q81_roundtrip`): the `q8_1` quantiser followed by
//!   its dequantisation, written back over the F32 activations.
//!
//! Dtypes are the five ggml weight dtypes (`GgmlIQ4XS`, `GgmlQ4K`, `GgmlQ5K`, `GgmlQ6K`,
//! `GgmlQ8_0`); anything else is an error naming it. The typed contracts are
//! `hipfire_kernel_abi::families::ggml_convert`. Every launch is validated against its contract
//! (`k` a multiple of the block, operand formats / alignment / byte extents, the 32-bit element
//! domain `m * k <= i32::MAX`, the launch geometry) BEFORE anything is loaded or launched. No export
//! uses atomics or LDS: results are deterministic and independent of the launch geometry, and every
//! launch is capture / record safe (no host sync, no allocation on the success path beyond the
//! contract-checked kernarg image).
//!
//! Image format: the object is a clang offload bundle (`__CLANG_OFFLOAD_BUNDLE__`,
//! `--no-offload-compress`) exactly as hipcc emits it; it is embedded unchanged and handed to
//! `module_load_data`, the same path the hipcc compile cache and the other embedded `.co` / `.hxaco`
//! images take (`ensure_embedded_kernel`).

use crate::kernel_abi;
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use hipfire_kernel_abi::families::ggml_convert as abi;
use hipfire_kernel_abi::{
    Arch, FormatId, InvocationV1, KernelAbiV1, OperandBinding, ValidatedInvocationV1, MAX_FIELDS,
};
use std::ffi::c_void;

pub const MODULE: &str = "ggml_convert_shim_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/ggml_convert_shim_gfx1201.co");

/// Elements per column chunk of the `colmap` permutation (fixed by the contract).
pub const CHUNK: usize = 128;
/// Round-trip launch block: 256 threads along x.
pub const ROUNDTRIP_BLOCK: [u32; 3] = [256, 1, 1];
pub const ROUNDTRIP_SYMBOL: &str = "ggml_q81_roundtrip_gfx1201";
/// Largest token count of one gather launch (tokens ride grid y, HIP limit 65535).
pub const MAX_GATHER_TOKENS: usize = 65535;

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("ggml_convert: {}", message.as_ref()))
}

/// Everything the contracts name about one weight dtype.
#[derive(Clone, Copy)]
pub struct Export {
    /// Elements per weight block (`qk`): 256, `Q8_0` 32.
    pub qk: usize,
    /// Bytes per weight block.
    pub block_bytes: usize,
    pub weight_format: FormatId,
    pub dequant_f16: &'static KernelAbiV1,
    pub dequant_f32: &'static KernelAbiV1,
    pub get_row: &'static KernelAbiV1,
    pub get_row1: &'static KernelAbiV1,
}

impl std::fmt::Debug for Export {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Export")
            .field("qk", &self.qk)
            .field("block_bytes", &self.block_bytes)
            .field("dequant_f16", &self.dequant_f16.exports[0].symbol)
            .field("dequant_f32", &self.dequant_f32.exports[0].symbol)
            .field("get_row", &self.get_row.exports[0].symbol)
            .field("get_row1", &self.get_row1.exports[0].symbol)
            .finish()
    }
}

/// The exports serving `dtype`; any dtype that is not one of the five ggml weight dtypes is an error
/// naming it.
pub fn export_of(dtype: DType) -> HipResult<Export> {
    let (qk, block_bytes) = dtype.ggml_block().ok_or_else(|| {
        err(format!(
            "dtype {dtype:?} is not a ggml weight dtype (GgmlIQ4XS, GgmlQ4K, GgmlQ5K, GgmlQ6K, GgmlQ8_0)"
        ))
    })?;
    macro_rules! export {
        ($m:ident, $format:expr) => {
            Export {
                qk,
                block_bytes,
                weight_format: $format,
                dequant_f16: &abi::$m::DEQUANT_F16,
                dequant_f32: &abi::$m::DEQUANT_F32,
                get_row: &abi::$m::GET_ROW,
                get_row1: &abi::$m::GET_ROW1,
            }
        };
    }
    Ok(match dtype {
        DType::GgmlIQ4XS => export!(
            iq4_xs,
            hipfire_kernel_abi::families::ggml_mmvq::FORMAT_IQ4_XS_136
        ),
        DType::GgmlQ4K => export!(
            q4_k,
            hipfire_kernel_abi::families::ggml_mmvq::FORMAT_Q4_K_144
        ),
        DType::GgmlQ5K => export!(q5_k, abi::FORMAT_Q5_K_176),
        DType::GgmlQ6K => export!(
            q6_k,
            hipfire_kernel_abi::families::ggml_mmvq::FORMAT_Q6_K_210
        ),
        DType::GgmlQ8_0 => export!(
            q8_0,
            hipfire_kernel_abi::families::ggml_mmvq::FORMAT_Q8_0_34
        ),
        other => return Err(err(format!("dtype {other:?} has no convert export"))),
    })
}

/// Launch block of an export (its contract's).
fn block_of(abi: &KernelAbiV1) -> [u32; 3] {
    abi.exports[0].block
}

/// Dequant grid: one super-block per workgroup for the 256-element types, 512 elements per workgroup
/// for `Q8_0`. `nb` weight blocks of `qk` elements.
pub fn dequant_grid(qk: usize, nb: usize) -> [u32; 3] {
    if qk == 32 {
        [(nb * 32).div_ceil(512) as u32, 1, 1]
    } else {
        [nb as u32, 1, 1]
    }
}

/// Gather grid: `(row work-groups, n_tokens, 1)`.
pub fn gather_grid(qk: usize, k: usize, n: usize) -> [u32; 3] {
    let x = if qk == 32 { k.div_ceil(512) } else { k / 256 };
    [x as u32, n as u32, 1]
}

/// Round-trip grid: `(ceil(ne / 256), 1, 1)`.
pub fn roundtrip_grid(ne: usize) -> [u32; 3] {
    [ne.div_ceil(256) as u32, 1, 1]
}

fn require_gfx1201(arch: &str) -> HipResult<Arch> {
    if arch != "gfx1201" {
        return Err(err(format!("requires gfx1201, this device is {arch}")));
    }
    kernel_abi::parse_arch(arch)
}

fn require_dtype(t: &GpuTensor, want: DType, name: &str) -> HipResult<()> {
    if t.dtype != want {
        return Err(err(format!("{name} must be {want:?}, is {:?}", t.dtype)));
    }
    Ok(())
}

fn require_bytes(t: &GpuTensor, need: usize, name: &str) -> HipResult<()> {
    let have = t.buf.size();
    if have < need {
        return Err(err(format!("{name} needs {need} bytes, has {have}")));
    }
    Ok(())
}

/// The shape checks every dequant launch makes before the contract sees it. Returns the weight block
/// count `nb` and the element count.
fn dequant_shape(e: &Export, dtype: DType, m: usize, k: usize) -> HipResult<(usize, usize)> {
    if m == 0 {
        return Err(err("m must be >= 1"));
    }
    if k == 0 || k % e.qk != 0 {
        return Err(err(format!(
            "{dtype:?}: k = {k} must be a positive multiple of the block ({})",
            e.qk
        )));
    }
    let ne = m
        .checked_mul(k)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| {
            err(format!(
                "m * k = {m} * {k} exceeds the 32-bit element domain (i32::MAX)"
            ))
        })?;
    Ok((ne / e.qk, ne))
}

/// A null `colmap`: no address, nothing to check.
fn null_colmap() -> OperandBinding {
    kernel_abi::binding_raw(std::ptr::null(), Some(0), abi::FORMAT_I32_CHUNK_PERM)
}

#[allow(clippy::too_many_arguments)]
fn validate_dequant<'a>(
    arch: Arch,
    e: &Export,
    f16: bool,
    m: usize,
    k: usize,
    bf16_round: bool,
    w: OperandBinding,
    out: OperandBinding,
    colmap: Option<OperandBinding>,
) -> HipResult<ValidatedInvocationV1<'a>> {
    let contract = if f16 { e.dequant_f16 } else { e.dequant_f32 };
    let symbol = contract.exports[0].symbol;
    let has_colmap = colmap.is_some();
    let nb = m * k / e.qk;
    let params_f16 = [m as u64, k as u64, u64::from(has_colmap)];
    let params_f32 = [
        m as u64,
        k as u64,
        u64::from(has_colmap),
        u64::from(bf16_round),
    ];
    let params: &[u64] = if f16 { &params_f16 } else { &params_f32 };
    kernel_abi::validate(
        contract,
        &InvocationV1 {
            arch,
            symbol,
            params,
            operands: &[w, out, colmap.unwrap_or_else(null_colmap)],
            grid: dequant_grid(e.qk, nb),
            block: block_of(contract),
            dynamic_lds: 0,
        },
    )
}

/// Launch a validated invocation.
fn ggml_convert_launch(
    gpu: &mut Gpu,
    symbol: &'static str,
    mut v: ValidatedInvocationV1<'_>,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_embedded_kernel(MODULE, IMAGE, symbol)?;
    let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
    let n = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
    let (grid, block) = (v.grid(), v.block());
    gpu.launch_maybe_blob(symbol, grid, block, 0, &mut ptrs[..n], || {
        kernel_abi::blob(v.explicit())
    })
}

/// The symbol of a contract's single export (`'static`).
fn symbol_of(abi: &'static KernelAbiV1) -> &'static str {
    abi.exports[0].symbol
}

impl Gpu {
    /// Dequantise the ggml tensor `w` (`m` rows of `k` elements, whole blocks, raw bytes; trailing
    /// bytes are ignored) into the F16 tensor `out` (`m * k` elements, element `e` of the flat
    /// tensor at `out[e]`, round to nearest even as upstream's `ggml_cuda_cast<half>`).
    ///
    /// `colmap`: `None` keeps the layout. `Some(c)` (an I32 device tensor of at least `k / 128`
    /// entries, a PERMUTATION of `0..k/128`: `c[g]` is the DESTINATION chunk of source chunk `g`)
    /// stores source element `row * k + col` at `out[row * k + c[col / 128] * 128 + col % 128]`;
    /// it needs `k % 128 == 0`. The permutation values are data: the launcher cannot check them.
    ///
    /// `dtype` is one of `GgmlIQ4XS`, `GgmlQ4K`, `GgmlQ5K`, `GgmlQ6K`, `GgmlQ8_0`; `k` a multiple
    /// of the block (256, `Q8_0` 32); `m * k <= i32::MAX` (the kernels index in 32 bits). Buffer byte
    /// lengths are checked before the launch.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_dequant_f16(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        out: &GpuTensor,
        m: usize,
        k: usize,
        colmap: Option<&GpuTensor>,
    ) -> HipResult<()> {
        let e = export_of(dtype)?;
        let arch = require_gfx1201(&self.arch)?;
        let (nb, ne) = dequant_shape(&e, dtype, m, k)?;
        require_dtype(out, DType::F16, "out")?;
        require_bytes(w, nb * e.block_bytes, "w")?;
        require_bytes(out, ne * 2, "out")?;
        let colmap = colmap_binding(colmap, k)?;
        let v = validate_dequant(
            arch,
            &e,
            true,
            m,
            k,
            false,
            kernel_abi::binding(w, e.weight_format),
            kernel_abi::binding(out, abi::FORMAT_F16_FLAT),
            colmap,
        )?;
        ggml_convert_launch(self, symbol_of(e.dequant_f16), v)
    }

    /// [`Gpu::ggml_dequant_f16`] into the F32 tensor `out` (`m * k` elements). `bf16_round = true`
    /// rounds every f32 result to the nearest bfloat16 (round to nearest even, NaN quietened, Inf
    /// kept: `ggml_compute_fp32_to_bf16`) and stores it widened to f32 (the codec-only quality tier:
    /// bit for bit the host dequantisation followed by bf16 rounding); `false` stores the plain f32.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_dequant_f32(
        &mut self,
        dtype: DType,
        w: &GpuTensor,
        out: &GpuTensor,
        m: usize,
        k: usize,
        bf16_round: bool,
        colmap: Option<&GpuTensor>,
    ) -> HipResult<()> {
        let e = export_of(dtype)?;
        let arch = require_gfx1201(&self.arch)?;
        let (nb, ne) = dequant_shape(&e, dtype, m, k)?;
        require_dtype(out, DType::F32, "out")?;
        require_bytes(w, nb * e.block_bytes, "w")?;
        require_bytes(out, ne * 4, "out")?;
        let colmap = colmap_binding(colmap, k)?;
        let v = validate_dequant(
            arch,
            &e,
            false,
            m,
            k,
            bf16_round,
            kernel_abi::binding(w, e.weight_format),
            kernel_abi::binding(out, abi::FORMAT_F32_FLAT),
            colmap,
        )?;
        ggml_convert_launch(self, symbol_of(e.dequant_f32), v)
    }

    /// Embedding lookup of ONE token: dequantise row `token` of the ggml table (`dim` elements per
    /// row, the table holds `buf.size() / row_bytes` rows) into the F32 row `out` (`dim`). The token
    /// is a by-value kernarg (`tok@8`), so a decode graph patches it like `embedding_lookup_q8`'s;
    /// `token` must be a row of the table (checked here: the kernel does not).
    pub fn ggml_embedding_lookup(
        &mut self,
        dtype: DType,
        table: &GpuTensor,
        out: &GpuTensor,
        token: u32,
        dim: usize,
    ) -> HipResult<()> {
        let e = export_of(dtype)?;
        let arch = require_gfx1201(&self.arch)?;
        let rows = table_rows(&e, dtype, table, dim)?;
        require_dtype(out, DType::F32, "out")?;
        require_bytes(out, dim * 4, "out")?;
        let contract = e.get_row1;
        let v = kernel_abi::validate(
            contract,
            &InvocationV1 {
                arch,
                symbol: symbol_of(contract),
                params: &[rows as u64, dim as u64, u64::from(token)],
                operands: &[
                    kernel_abi::binding(table, e.weight_format),
                    kernel_abi::binding(out, abi::FORMAT_F32_ROWS_NK),
                ],
                grid: gather_grid(e.qk, dim, 1),
                block: block_of(contract),
                dynamic_lds: 0,
            },
        )?;
        ggml_convert_launch(self, symbol_of(contract), v)
    }

    /// Batched embedding lookup: `token_ids` is a device tensor of `n` int32 token ids (raw words),
    /// `out` the F32 rows `[n, dim]`; row `i` is the dequantised table row `token_ids[i]`. The ids are
    /// device data: they are NOT range-checked (each must be `< table rows`). `1 <= n <= 65535`.
    pub fn ggml_embedding_lookup_batched(
        &mut self,
        dtype: DType,
        table: &GpuTensor,
        out: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        let e = export_of(dtype)?;
        let arch = require_gfx1201(&self.arch)?;
        if n == 0 || n > MAX_GATHER_TOKENS {
            return Err(err(format!(
                "n = {n} tokens must be in 1..={MAX_GATHER_TOKENS} (grid y)"
            )));
        }
        let rows = table_rows(&e, dtype, table, dim)?;
        require_dtype(out, DType::F32, "out")?;
        require_bytes(out, n * dim * 4, "out")?;
        require_bytes(token_ids, n * 4, "token_ids")?;
        let contract = e.get_row;
        let v = kernel_abi::validate(
            contract,
            &InvocationV1 {
                arch,
                symbol: symbol_of(contract),
                params: &[rows as u64, dim as u64, n as u64],
                operands: &[
                    kernel_abi::binding(table, e.weight_format),
                    kernel_abi::binding(token_ids, abi::FORMAT_I32_TOKEN_IDS),
                    kernel_abi::binding(out, abi::FORMAT_F32_ROWS_NK),
                ],
                grid: gather_grid(e.qk, dim, n),
                block: block_of(contract),
                dynamic_lds: 0,
            },
        )?;
        ggml_convert_launch(self, symbol_of(contract), v)
    }

    /// In-place A8 emulation over the F32 activations `x` (`rows * k` elements, `k % 32 == 0`): every
    /// 32-element block is quantised exactly like `ggml_q8_1_quantize` (`d = amax / 127`, `q = amax
    /// == 0 ? 0 : roundf(x / d)`) and written back as `f32(f16(d)) * f32(q)`, the value the `q8_1`
    /// dot product sees. The input must keep `d = amax / 127` normal-or-subnormal nonzero and
    /// `amax <= 65504 * 127` (activations are far inside both).
    pub fn ggml_q81_roundtrip(&mut self, x: &GpuTensor, rows: usize, k: usize) -> HipResult<()> {
        let arch = require_gfx1201(&self.arch)?;
        require_dtype(x, DType::F32, "x")?;
        if rows == 0 || k == 0 || k % 32 != 0 {
            return Err(err(format!(
                "roundtrip: rows = {rows} must be >= 1 and k = {k} a positive multiple of 32"
            )));
        }
        let ne = rows
            .checked_mul(k)
            .ok_or_else(|| err("roundtrip: rows * k overflowed"))?;
        require_bytes(x, ne * 4, "x")?;
        let v = kernel_abi::validate(
            &abi::Q81_ROUNDTRIP,
            &InvocationV1 {
                arch,
                symbol: ROUNDTRIP_SYMBOL,
                params: &[ne as u64],
                operands: &[kernel_abi::binding(x, abi::FORMAT_F32_FLAT)],
                grid: roundtrip_grid(ne),
                block: ROUNDTRIP_BLOCK,
                dynamic_lds: 0,
            },
        )?;
        ggml_convert_launch(self, ROUNDTRIP_SYMBOL, v)
    }
}

/// The colmap operand binding of a dequant launch (`None` = the null operand).
fn colmap_binding(colmap: Option<&GpuTensor>, k: usize) -> HipResult<Option<OperandBinding>> {
    let Some(c) = colmap else {
        return Ok(None);
    };
    if k % CHUNK != 0 {
        return Err(err(format!(
            "a colmap permutes {CHUNK}-element chunks: k = {k} must be a multiple of {CHUNK}"
        )));
    }
    require_bytes(c, k / CHUNK * 4, "colmap")?;
    Ok(Some(kernel_abi::binding(c, abi::FORMAT_I32_CHUNK_PERM)))
}

/// Rows of an embedding table of `dim`-element rows (whole rows in the buffer).
fn table_rows(e: &Export, dtype: DType, table: &GpuTensor, dim: usize) -> HipResult<usize> {
    if dim == 0 || dim % e.qk != 0 {
        return Err(err(format!(
            "{dtype:?}: dim = {dim} must be a positive multiple of the block ({})",
            e.qk
        )));
    }
    let row_bytes = dim / e.qk * e.block_bytes;
    let rows = table.buf.size() / row_bytes;
    if rows == 0 {
        return Err(err(format!(
            "table has {} bytes, less than one row of {row_bytes}",
            table.buf.size()
        )));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

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

    fn u64_at(a: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(a[o..o + 8].try_into().unwrap())
    }
    fn i32_at(a: &[u8], o: usize) -> i32 {
        i32::from_le_bytes(a[o..o + 4].try_into().unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant(
        e: &Export,
        f16: bool,
        m: usize,
        k: usize,
        bf16: bool,
        colmap: bool,
    ) -> HipResult<ValidatedInvocationV1<'static>> {
        let nb = m * k / e.qk;
        let ne = m * k;
        validate_dequant(
            Arch::Gfx1201,
            e,
            f16,
            m,
            k,
            bf16,
            bind(0x1000_0000, (nb * e.block_bytes) as u64, e.weight_format),
            if f16 {
                bind(0x4000_0000, (ne * 2) as u64, abi::FORMAT_F16_FLAT)
            } else {
                bind(0x4000_0000, (ne * 4) as u64, abi::FORMAT_F32_FLAT)
            },
            colmap.then(|| {
                bind(
                    0x7000_0000,
                    (k / CHUNK * 4) as u64,
                    abi::FORMAT_I32_CHUNK_PERM,
                )
            }),
        )
    }

    #[test]
    fn image_is_the_committed_bundle_and_carries_every_export() {
        assert!(IMAGE.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"));
        let mut symbols = vec![ROUNDTRIP_SYMBOL.to_owned()];
        for dtype in ALL {
            let e = export_of(dtype).unwrap();
            for c in [e.dequant_f16, e.dequant_f32, e.get_row, e.get_row1] {
                symbols.push(c.exports[0].symbol.to_owned());
            }
        }
        assert_eq!(symbols.len(), 21);
        for s in &symbols {
            assert!(
                IMAGE.windows(s.len()).any(|w| w == s.as_bytes()),
                "{s} is not in the embedded image"
            );
            // every export is owned by a contract in the object this launcher embeds
            let abi = kernel_abi::contract("gfx1201", s).unwrap();
            let ex = abi.export(Arch::Gfx1201, s).unwrap();
            let hipfire_kernel_abi::ObjectRefV1::Embedded {
                module,
                artifact,
                sha256,
            } = ex.object
            else {
                panic!("{s} is not an embedded export");
            };
            assert_eq!(
                (module, artifact),
                (MODULE, "kernels/ggml_convert_shim_gfx1201.co")
            );
            // the contract pins the bytes this launcher embeds
            assert_eq!(
                sha256,
                sha2::Sha256::digest(IMAGE)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            assert_eq!(ex.group_segment_fixed, 0, "{s}");
        }
    }

    #[test]
    fn export_table_covers_the_five_dtypes() {
        let want = [
            (DType::GgmlIQ4XS, 256, 136, [32, 1, 1]),
            (DType::GgmlQ4K, 256, 144, [32, 1, 1]),
            (DType::GgmlQ5K, 256, 176, [64, 1, 1]),
            (DType::GgmlQ6K, 256, 210, [64, 1, 1]),
            (DType::GgmlQ8_0, 32, 34, [256, 1, 1]),
        ];
        for (dtype, qk, bytes, block) in want {
            let e = export_of(dtype).unwrap();
            assert_eq!((e.qk, e.block_bytes), (qk, bytes), "{dtype:?}");
            for c in [e.dequant_f16, e.dequant_f32, e.get_row, e.get_row1] {
                assert_eq!(block_of(c), block, "{}", symbol_of(c));
            }
        }
        // symbols follow `ggml_{dequant,get_row,get_row1}_<tag>_{f16,f32}_gfx1201`
        let e = export_of(DType::GgmlQ4K).unwrap();
        assert_eq!(symbol_of(e.dequant_f16), "ggml_dequant_q4_k_f16_gfx1201");
        assert_eq!(symbol_of(e.dequant_f32), "ggml_dequant_q4_k_f32_gfx1201");
        assert_eq!(symbol_of(e.get_row), "ggml_get_row_q4_k_f32_gfx1201");
        assert_eq!(symbol_of(e.get_row1), "ggml_get_row1_q4_k_f32_gfx1201");
    }

    #[test]
    fn other_dtypes_are_refused_by_name() {
        for dtype in [
            DType::F32,
            DType::MQ4G256V2,
            DType::Q4K,
            DType::HFQ4G256,
            DType::F16,
        ] {
            let e = export_of(dtype).unwrap_err().to_string();
            assert!(e.contains(&format!("{dtype:?}")), "{e}");
        }
    }

    #[test]
    fn frozen_dequant_kernarg_abi_and_grid() {
        for dtype in ALL {
            let e = export_of(dtype).unwrap();
            let (m, k) = (5120usize, 6144usize);
            let nb = m * k / e.qk;
            for colmap in [false, true] {
                // f16: vx@0 dst@8 colmap@16 nb@24 K@28 chunk@32, 36 explicit bytes
                let v = dequant(&e, true, m, k, false, colmap).unwrap();
                let a = v.explicit();
                assert_eq!(a.len(), 36, "{dtype:?}");
                assert_eq!(u64_at(a, 0), 0x1000_0000);
                assert_eq!(u64_at(a, 8), 0x4000_0000);
                assert_eq!(u64_at(a, 16), if colmap { 0x7000_0000 } else { 0 });
                assert_eq!(i32_at(a, 24), nb as i32);
                assert_eq!(i32_at(a, 28), k as i32);
                assert_eq!(i32_at(a, 32), 128);
                assert_eq!(kernel_abi::blob(a).as_bytes()[..36], *a);
                assert_eq!(
                    (v.grid(), v.block(), v.dynamic_lds()),
                    (dequant_grid(e.qk, nb), block_of(e.dequant_f16), 0)
                );
                // f32: + bf16_round@36, 40 explicit bytes
                for bf16 in [false, true] {
                    let v = dequant(&e, false, m, k, bf16, colmap).unwrap();
                    let a = v.explicit();
                    assert_eq!(a.len(), 40, "{dtype:?}");
                    assert_eq!(u64_at(a, 16), if colmap { 0x7000_0000 } else { 0 });
                    assert_eq!(i32_at(a, 24), nb as i32);
                    assert_eq!(i32_at(a, 28), k as i32);
                    assert_eq!(i32_at(a, 32), 128);
                    assert_eq!(i32_at(a, 36), i32::from(bf16));
                    assert_eq!(kernel_abi::blob(a).as_bytes()[..40], *a);
                }
            }
        }
        // grids: one super-block per workgroup (256-element types), 512 elements per workgroup (q8_0)
        assert_eq!(dequant_grid(256, 102_400), [102_400, 1, 1]);
        assert_eq!(dequant_grid(32, 16), [1, 1, 1]);
        assert_eq!(dequant_grid(32, 17), [2, 1, 1]);
        assert_eq!(dequant_grid(32, 5120 * 5120 / 32), [51_200, 1, 1]);
    }

    #[test]
    fn dequant_domain_is_enforced_before_launch() {
        let e = export_of(DType::GgmlQ4K).unwrap();
        // k not a block multiple, m = 0, 32-bit element domain
        assert!(dequant_shape(&e, DType::GgmlQ4K, 4, 5000)
            .unwrap_err()
            .to_string()
            .contains("multiple"));
        assert!(dequant_shape(&e, DType::GgmlQ4K, 0, 256).is_err());
        assert!(dequant_shape(&e, DType::GgmlQ4K, 248_320, 5120).is_ok()); // 1.27e9 < 2^31
        let big = dequant_shape(&e, DType::GgmlQ4K, 1 << 20, 4096)
            .unwrap_err()
            .to_string();
        assert!(big.contains("32-bit"), "{big}");
        // a colmap row length must be a chunk multiple (contract rule, 128 | K * COLMAP)
        let q8 = export_of(DType::GgmlQ8_0).unwrap();
        assert!(dequant(&q8, false, 4, 160, false, false).is_ok());
        let e128 = dequant(&q8, false, 4, 160, false, true)
            .unwrap_err()
            .to_string();
        assert!(e128.contains("128"), "{e128}");
        // short buffers are rejected by the contract extents
        let nb = 64 * 2048 / e.qk;
        let r = validate_dequant(
            Arch::Gfx1201,
            &e,
            true,
            64,
            2048,
            false,
            bind(
                0x1000_0000,
                (nb * e.block_bytes - 1) as u64,
                e.weight_format,
            ),
            bind(0x4000_0000, 64 * 2048 * 2, abi::FORMAT_F16_FLAT),
            None,
        );
        assert!(r.unwrap_err().to_string().contains("vx"));
        let r = validate_dequant(
            Arch::Gfx1201,
            &e,
            false,
            64,
            2048,
            false,
            bind(0x1000_0000, (nb * e.block_bytes) as u64, e.weight_format),
            bind(0x4000_0000, 64 * 2048 * 4 - 4, abi::FORMAT_F32_FLAT),
            None,
        );
        assert!(r.unwrap_err().to_string().contains("dst"));
    }

    #[test]
    fn frozen_gather_kernarg_abi_and_grid() {
        let (rows, k, n) = (248_320usize, 5120usize, 7usize);
        for dtype in ALL {
            let e = export_of(dtype).unwrap();
            let table = bind(
                0x1000_0000,
                (rows * k / e.qk * e.block_bytes) as u64,
                e.weight_format,
            );
            // vx@0 tok_ids@8 dst@16 K@24, 28 explicit bytes, grid (K/256 | ceil(K/512), n, 1)
            let c = e.get_row;
            let v = kernel_abi::validate(
                c,
                &InvocationV1 {
                    arch: Arch::Gfx1201,
                    symbol: symbol_of(c),
                    params: &[rows as u64, k as u64, n as u64],
                    operands: &[
                        table,
                        bind(0x2000_0000, (n * 4) as u64, abi::FORMAT_I32_TOKEN_IDS),
                        bind(0x3000_0000, (n * k * 4) as u64, abi::FORMAT_F32_ROWS_NK),
                    ],
                    grid: gather_grid(e.qk, k, n),
                    block: block_of(c),
                    dynamic_lds: 0,
                },
            )
            .unwrap();
            let a = v.explicit();
            assert_eq!(a.len(), 28, "{dtype:?}");
            assert_eq!(u64_at(a, 0), 0x1000_0000);
            assert_eq!(u64_at(a, 8), 0x2000_0000);
            assert_eq!(u64_at(a, 16), 0x3000_0000);
            assert_eq!(i32_at(a, 24), 5120);
            assert_eq!(v.grid(), [if e.qk == 32 { 10 } else { 20 }, 7, 1]);
            // by value: vx@0 tok@8 dst@16 K@24
            let c = e.get_row1;
            let v = kernel_abi::validate(
                c,
                &InvocationV1 {
                    arch: Arch::Gfx1201,
                    symbol: symbol_of(c),
                    params: &[rows as u64, k as u64, 248_319],
                    operands: &[
                        table,
                        bind(0x3000_0000, (k * 4) as u64, abi::FORMAT_F32_ROWS_NK),
                    ],
                    grid: gather_grid(e.qk, k, 1),
                    block: block_of(c),
                    dynamic_lds: 0,
                },
            )
            .unwrap();
            let a = v.explicit();
            assert_eq!(a.len(), 28);
            assert_eq!(u64_at(a, 0), 0x1000_0000);
            assert_eq!(i32_at(a, 8), 248_319);
            assert_eq!(u64_at(a, 16), 0x3000_0000);
            assert_eq!(i32_at(a, 24), 5120);
            assert_eq!(v.grid(), [if e.qk == 32 { 10 } else { 20 }, 1, 1]);
            // token == rows is not a row
            let r = kernel_abi::validate(
                c,
                &InvocationV1 {
                    arch: Arch::Gfx1201,
                    symbol: symbol_of(c),
                    params: &[rows as u64, k as u64, rows as u64],
                    operands: &[
                        table,
                        bind(0x3000_0000, (k * 4) as u64, abi::FORMAT_F32_ROWS_NK),
                    ],
                    grid: gather_grid(e.qk, k, 1),
                    block: block_of(c),
                    dynamic_lds: 0,
                },
            );
            assert!(
                matches!(&r, Err(e) if e.to_string().contains("ROWS")),
                "{dtype:?}: {r:?}"
            );
        }
        assert_eq!(gather_grid(256, 17408, 1), [68, 1, 1]);
        assert_eq!(gather_grid(32, 17408, 1), [34, 1, 1]);
        assert_eq!(gather_grid(32, 5121 - 1, 65535), [10, 65535, 1]);
    }

    #[test]
    fn gather_row_offsets_past_32_bits_are_admitted() {
        // The kernels form `token * row_bytes` and `out_row * K` in 64 bits. The 27B embedding table
        // (248320 x 5120, at most 1.35 GB as q8_0) stays under 2^31 bytes, but the contract does not rely on
        // it: a 248320 x 17408 q6_k table has its last row 3.5 GB in.
        let e = export_of(DType::GgmlQ6K).unwrap();
        let (rows, k) = (248_320usize, 17408usize);
        let row_bytes = k / e.qk * e.block_bytes;
        assert!((rows - 1) * row_bytes > i32::MAX as usize);
        let c = e.get_row;
        let v = kernel_abi::validate(
            c,
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: symbol_of(c),
                params: &[rows as u64, k as u64, 1],
                operands: &[
                    bind(0x1_0000_0000, (rows * row_bytes) as u64, e.weight_format),
                    bind(0x2000_0000, 4, abi::FORMAT_I32_TOKEN_IDS),
                    bind(0x3000_0000, (k * 4) as u64, abi::FORMAT_F32_ROWS_NK),
                ],
                grid: gather_grid(e.qk, k, 1),
                block: block_of(c),
                dynamic_lds: 0,
            },
        );
        assert!(v.is_ok());
        // more than 65535 tokens (grid y) is refused by the contract
        let r = kernel_abi::validate(
            c,
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: symbol_of(c),
                params: &[rows as u64, k as u64, 65536],
                operands: &[
                    bind(0x1_0000_0000, (rows * row_bytes) as u64, e.weight_format),
                    bind(0x2000_0000, 65536 * 4, abi::FORMAT_I32_TOKEN_IDS),
                    bind(0x3000_0000, 65536 * k as u64 * 4, abi::FORMAT_F32_ROWS_NK),
                ],
                grid: gather_grid(e.qk, k, 65535),
                block: block_of(c),
                dynamic_lds: 0,
            },
        );
        let e = r.unwrap_err().to_string();
        assert!(e.contains("grid y"), "{e}");
    }

    #[test]
    fn frozen_roundtrip_kernarg_abi_and_grid() {
        // x@0 ne@8, 12 explicit bytes, grid (ceil(ne/256),1,1), block 256
        let (rows, k) = (3usize, 5120usize);
        let ne = rows * k;
        let v = kernel_abi::validate(
            &abi::Q81_ROUNDTRIP,
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: ROUNDTRIP_SYMBOL,
                params: &[ne as u64],
                operands: &[bind(0x1000, (ne * 4) as u64, abi::FORMAT_F32_FLAT)],
                grid: roundtrip_grid(ne),
                block: ROUNDTRIP_BLOCK,
                dynamic_lds: 0,
            },
        )
        .unwrap();
        let a = v.explicit();
        assert_eq!(a.len(), 12);
        assert_eq!(u64_at(a, 0), 0x1000);
        assert_eq!(i32_at(a, 8), ne as i32);
        assert_eq!((v.grid(), v.block()), ([60, 1, 1], [256, 1, 1]));
        assert_eq!(roundtrip_grid(32), [1, 1, 1]);
        assert_eq!(roundtrip_grid(8224), [33, 1, 1]);
        // ne not a multiple of 32 is refused
        let r = kernel_abi::validate(
            &abi::Q81_ROUNDTRIP,
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: ROUNDTRIP_SYMBOL,
                params: &[100],
                operands: &[bind(0x1000, 400, abi::FORMAT_F32_FLAT)],
                grid: roundtrip_grid(100),
                block: ROUNDTRIP_BLOCK,
                dynamic_lds: 0,
            },
        );
        let e = r.unwrap_err().to_string();
        assert!(e.contains("ne % 32"), "{e}");
    }

    #[test]
    fn table_rows_counts_whole_rows_only() {
        // `table_rows` needs a GpuTensor; the arithmetic it applies:
        let e = export_of(DType::GgmlQ8_0).unwrap();
        let row_bytes = 5120 / e.qk * e.block_bytes;
        assert_eq!(row_bytes, 5440);
        assert_eq!(248_320 * row_bytes / row_bytes, 248_320);
        assert_eq!((248_320 * row_bytes + 5439) / row_bytes, 248_320);
    }
}
