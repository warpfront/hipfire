// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Run the generated Qwen4 operator oracle on the production HIP path,
//! execute the production HFQM carrier on the fixed teacher-forced corpus, or
//! exercise real-model state/MTP rollback plus compact arena fault scenarios.
//!
//! Fixture usage:
//!   cargo run --release -p hipfire-arch-qwen4 --features reference-parity --example qwen4_parity -- \
//!       --fixtures /path/to/reference-fixtures --out /tmp/qwen4-parity.json
//!
//! Candidate quality usage:
//!   cargo run --release -p hipfire-arch-qwen4 --features reference-parity --example qwen4_parity -- \
//!       --model /path/to/model.hfq --tokens benchmarks/prompts/qwen4-teacher-forced.tokens.json \
//!       --out /tmp/qwen4-candidate.json
//!
//! State parity usage:
//!   cargo run --release -p hipfire-arch-qwen4 --features reference-parity --example qwen4_parity -- \
//!       --mode state --model /path/to/model.hfq \
//!       --tokens benchmarks/prompts/qwen4-teacher-forced.tokens.json \
//!       --out /tmp/qwen4-state-parity.json

use hipfire_arch_qwen4::{admit_hfqm_artifact, PleHashMetadata, PleHistory, Qwen4HfqmArtifact};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::tensor_ops::{
    bf16_roundtrip_f32, gated_delta_params_f32, gated_delta_step, hyper_read, hyper_write,
    indexed_attention_attention, indexed_attention_cache_append, indexed_attention_norm_rope,
    indexed_attention_pool_rope, indexed_attention_select, Bf16Roundtrip, GatedDeltaStep,
    HyperRead, HyperWrite, IndexedAttentionAttention, IndexedAttentionCacheAppend,
    IndexedAttentionNormRope, IndexedAttentionPoolRope, IndexedAttentionSelect,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
type Result<T, E = String> = std::result::Result<T, E>;

const ORACLE_SCHEMA: &str = "hipfire.qwen4.reference_oracle.v2";
const REPORT_SCHEMA: &str = "hipfire.qwen4.parity.v1";
const PLE_HEADS: usize = 16;
const PLE_WIDTH: usize = 160;
const QSA_COMPRESS: usize = 4;
const QSA_BUDGET: usize = 8;
const RMS_EPS: f32 = 1.0e-6;

#[derive(Clone, Debug)]
struct Array {
    shape: Vec<usize>,
    dtype: String,
    values: Vec<f32>,
    integers: Option<Vec<i64>>,
}

impl Array {
    fn row(&self, row: usize) -> Result<&[f32], String> {
        if self.shape.is_empty() || row >= self.shape[0] {
            return Err(format!("row {row} out of bounds for {:?}", self.shape));
        }
        let stride = self.shape[1..].iter().product::<usize>();
        Ok(&self.values[row * stride..(row + 1) * stride])
    }

    fn ints(&self) -> Result<&[i64], String> {
        self.integers
            .as_deref()
            .ok_or_else(|| format!("array dtype {} is not an integer array", self.dtype))
    }
}

fn fail<T>(message: impl Into<String>) -> Result<T, String> {
    Err(message.into())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "u16 offset overflow".to_string())?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| format!("truncated u16 at {offset}"))?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "u32 offset overflow".to_string())?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| format!("truncated u32 at {offset}"))?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "u64 offset overflow".to_string())?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| format!("truncated u64 at {offset}"))?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, String> {
    Ok(read_u32(bytes, offset)? as i32)
}

fn read_i64(bytes: &[u8], offset: usize) -> Result<i64, String> {
    Ok(read_u64(bytes, offset)? as i64)
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let marker = "'shape':";
    let start = header
        .find(marker)
        .ok_or_else(|| "NPY header has no shape".to_string())?
        + marker.len();
    let rest = header[start..].trim_start();
    if rest.starts_with('(') {
        let close = rest
            .find(')')
            .ok_or_else(|| "unterminated NPY shape".to_string())?;
        let inside = &rest[1..close];
        let mut shape = Vec::new();
        for item in inside
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            shape.push(
                item.parse::<usize>()
                    .map_err(|error| format!("invalid NPY shape component {item:?}: {error}"))?,
            );
        }
        if shape.is_empty() {
            return Ok(vec![]);
        }
        return Ok(shape);
    }
    let digits = rest
        .split(|character: char| !character.is_ascii_digit())
        .find(|item| !item.is_empty())
        .ok_or_else(|| "invalid scalar NPY shape".to_string())?;
    Ok(vec![digits.parse::<usize>().map_err(|error| {
        format!("invalid scalar NPY shape: {error}")
    })?])
}

fn parse_descr(header: &str) -> Result<&str, String> {
    let marker = "'descr':";
    let start = header
        .find(marker)
        .ok_or_else(|| "NPY header has no descr".to_string())?
        + marker.len();
    let rest = header[start..].trim_start();
    let quote = rest
        .chars()
        .next()
        .ok_or_else(|| "empty NPY descr".to_string())?;
    if quote != '\'' && quote != '"' {
        return fail("NPY descr is not quoted");
    }
    let end = rest[1..]
        .find(quote)
        .ok_or_else(|| "unterminated NPY descr".to_string())?
        + 1;
    Ok(&rest[1..end])
}

fn read_npy(bytes: &[u8]) -> Result<Array, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return fail("NPZ member is not an NPY array");
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (read_u16(bytes, 8)? as usize, 10usize),
        2 | 3 => (read_u32(bytes, 8)? as usize, 12usize),
        other => return fail(format!("unsupported NPY major version {other}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| "NPY header length overflow".to_string())?;
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_end)
            .ok_or_else(|| "truncated NPY header".to_string())?,
    )
    .map_err(|error| format!("NPY header is not ASCII: {error}"))?;
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return fail("Fortran-order NPY arrays are not accepted");
    }
    let descr = parse_descr(header)?.to_string();
    let shape = parse_shape(header)?;
    let count = if shape.is_empty() {
        1
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| "NPY element count overflow".to_string())?
    };
    let payload = bytes
        .get(header_end..)
        .ok_or_else(|| "truncated NPY payload".to_string())?;
    let item_size = match descr.as_str() {
        "<f4" | ">f4" | "<i4" | ">i4" => 4,
        "<f8" | ">f8" | "<i8" | ">i8" => 8,
        "<u2" | ">u2" => 2,
        "|u1" | "<u1" | "|b1" | "?" => 1,
        other => return fail(format!("unsupported NPY dtype {other:?}")),
    };
    if payload.len() != count.saturating_mul(item_size) {
        return fail(format!(
            "NPY payload has {} bytes, expected {}",
            payload.len(),
            count.saturating_mul(item_size)
        ));
    }
    let mut values = Vec::with_capacity(count);
    let mut integers = None;
    match descr.as_str() {
        "<f4" => {
            for index in 0..count {
                values.push(f32::from_bits(read_u32(payload, index * 4)?));
            }
        }
        ">f4" => {
            for index in 0..count {
                values.push(f32::from_bits(u32::from_be_bytes(
                    payload[index * 4..index * 4 + 4].try_into().unwrap(),
                )));
            }
        }
        "<f8" => {
            for index in 0..count {
                values.push(f64::from_bits(read_u64(payload, index * 8)?) as f32);
            }
        }
        ">f8" => {
            for index in 0..count {
                values.push(f64::from_bits(u64::from_be_bytes(
                    payload[index * 8..index * 8 + 8].try_into().unwrap(),
                )) as f32);
            }
        }
        "<u2" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value = read_u16(payload, index * 2)?;
                values.push(value as f32);
                ints.push(value as i64);
            }
            integers = Some(ints);
        }
        ">u2" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value =
                    u16::from_be_bytes(payload[index * 2..index * 2 + 2].try_into().unwrap());
                values.push(value as f32);
                ints.push(value as i64);
            }
            integers = Some(ints);
        }
        "<i4" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value = read_i32(payload, index * 4)?;
                values.push(value as f32);
                ints.push(value as i64);
            }
            integers = Some(ints);
        }
        ">i4" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value =
                    i32::from_be_bytes(payload[index * 4..index * 4 + 4].try_into().unwrap());
                values.push(value as f32);
                ints.push(value as i64);
            }
            integers = Some(ints);
        }
        "<i8" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value = read_i64(payload, index * 8)?;
                values.push(value as f32);
                ints.push(value);
            }
            integers = Some(ints);
        }
        ">i8" => {
            let mut ints = Vec::with_capacity(count);
            for index in 0..count {
                let value =
                    i64::from_be_bytes(payload[index * 8..index * 8 + 8].try_into().unwrap());
                values.push(value as f32);
                ints.push(value);
            }
            integers = Some(ints);
        }
        "|u1" | "<u1" => {
            let mut ints = Vec::with_capacity(count);
            for &value in payload {
                values.push(value as f32);
                ints.push(value as i64);
            }
            integers = Some(ints);
        }
        "|b1" | "?" => {
            values.extend(
                payload
                    .iter()
                    .map(|&value| if value == 0 { 0.0 } else { 1.0 }),
            );
        }
        _ => unreachable!(),
    }
    if values.iter().any(|value| !value.is_finite()) {
        return fail("fixture contains non-finite values");
    }
    Ok(Array {
        shape,
        dtype: descr,
        values,
        integers,
    })
}

fn read_npz(path: &Path) -> Result<BTreeMap<String, Array>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut offset = 0usize;
    let mut arrays = BTreeMap::new();
    while offset + 4 <= bytes.len() && &bytes[offset..offset + 4] == b"PK\x03\x04" {
        if offset + 30 > bytes.len() {
            return fail(format!("truncated ZIP local header in {}", path.display()));
        }
        let flags = read_u16(&bytes, offset + 6)?;
        let compression = read_u16(&bytes, offset + 8)?;
        let compressed_size = read_u32(&bytes, offset + 18)? as usize;
        let uncompressed_size = read_u32(&bytes, offset + 22)? as usize;
        let name_len = read_u16(&bytes, offset + 26)? as usize;
        let extra_len = read_u16(&bytes, offset + 28)? as usize;
        if flags != 0 || compression != 0 {
            return fail(format!(
                "{} is not an uncompressed deterministic NPZ",
                path.display()
            ));
        }
        let name_start = offset + 30;
        let data_start = name_start
            .checked_add(name_len)
            .and_then(|value| value.checked_add(extra_len))
            .ok_or_else(|| "ZIP offset overflow".to_string())?;
        let data_end = data_start
            .checked_add(compressed_size)
            .ok_or_else(|| "ZIP data offset overflow".to_string())?;
        let name = std::str::from_utf8(
            bytes
                .get(name_start..name_start + name_len)
                .ok_or_else(|| "truncated ZIP filename".to_string())?,
        )
        .map_err(|error| format!("ZIP filename is not UTF-8: {error}"))?;
        if !name.ends_with(".npy") || name.contains('/') {
            return fail(format!("unexpected NPZ member {name:?}"));
        }
        let payload = bytes
            .get(data_start..data_end)
            .ok_or_else(|| format!("truncated NPZ member {name:?}"))?;
        if payload.len() != uncompressed_size {
            return fail(format!("NPZ member {name:?} size mismatch"));
        }
        let key = name.trim_end_matches(".npy").to_string();
        arrays.insert(key, read_npy(payload)?);
        offset = data_end;
    }
    if arrays.is_empty() {
        return fail(format!("{} contains no root NPY members", path.display()));
    }
    Ok(arrays)
}

fn required<'a>(arrays: &'a BTreeMap<String, Array>, name: &str) -> Result<&'a Array, String> {
    arrays
        .get(name)
        .ok_or_else(|| format!("fixture is missing array {name:?}"))
}

fn shape(array: &Array, expected: &[usize], name: &str) -> Result<(), String> {
    if array.shape != expected {
        return fail(format!("{name} shape {:?} != {:?}", array.shape, expected));
    }
    Ok(())
}

fn tolerance(manifest: &Value, name: &str) -> Result<(f32, f32)> {
    let node = manifest
        .pointer(&format!("/equations/tolerances/{name}"))
        .ok_or_else(|| format!("manifest has no frozen tolerance {name:?}"))?;
    let atol = node
        .get("atol")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("tolerance {name:?} has no atol"))? as f32;
    let rtol = node
        .get("rtol")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("tolerance {name:?} has no rtol"))? as f32;
    Ok((atol, rtol))
}

fn compare_f32(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
) -> Result<(f32, f32), String> {
    if actual.len() != expected.len() {
        return fail(format!(
            "{label}: {} values, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let abs = (got - want).abs();
        let rel = abs / want.abs().max(1.0e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > atol + rtol * want.abs() {
            return fail(format!(
                "{label}: mismatch at {index}: got {got:.8e}, expected {want:.8e}, abs {abs:.3e}, bound {:.3e}",
                atol + rtol * want.abs()
            ));
        }
    }
    Ok((max_abs, max_rel))
}
fn measure_f32(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
) -> Result<(f32, f32, bool), String> {
    if actual.len() != expected.len() {
        return fail(format!(
            "{label}: {} values, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut within_tolerance = true;
    for (&got, &want) in actual.iter().zip(expected) {
        let abs = (got - want).abs();
        let rel = abs / want.abs().max(1.0e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        within_tolerance &= abs <= atol + rtol * want.abs();
    }
    Ok((max_abs, max_rel, within_tolerance))
}

fn compare_i64(actual: &[i64], expected: &[i64], label: &str) -> Result<(), String> {
    if actual != expected {
        let index = actual
            .iter()
            .zip(expected)
            .position(|(got, want)| got != want)
            .unwrap_or(actual.len().min(expected.len()));
        return fail(format!(
            "{label}: integer mismatch at {index}: got {:?}, expected {:?}",
            actual.get(index),
            expected.get(index)
        ));
    }
    Ok(())
}

/// Compare top-k routes per token as expert-sorted (expert, weight) pairs.
/// Rank order carries no meaning (the combine accumulates in expert-id order)
/// and experts whose router logits tie within the F32 tolerance may swap
/// ranks; the selected set and each expert's weight must still match.
#[allow(clippy::too_many_arguments)]
fn compare_routes(
    actual_experts: &[i64],
    actual_weights: &[f32],
    expected_experts: &[i64],
    expected_weights: &[f32],
    top_k: usize,
    atol: f32,
    rtol: f32,
    label: &str,
) -> Result<(f32, f32), String> {
    if actual_experts.len() != actual_weights.len()
        || expected_experts.len() != expected_weights.len()
    {
        return fail(format!("{label}: expert and weight counts differ"));
    }
    let by_expert = |experts: &[i64], weights: &[f32]| -> (Vec<i64>, Vec<f32>) {
        let mut routes: Vec<(i64, f32)> = experts
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        for token in routes.chunks_mut(top_k) {
            token.sort_by_key(|route| route.0);
        }
        routes.into_iter().unzip()
    };
    let (actual_ids, actual_route_weights) = by_expert(actual_experts, actual_weights);
    let (expected_ids, expected_route_weights) = by_expert(expected_experts, expected_weights);
    compare_i64(
        &actual_ids,
        &expected_ids,
        &format!("{label} selected experts"),
    )?;
    compare_f32(
        &actual_route_weights,
        &expected_route_weights,
        atol,
        rtol,
        &format!("{label} routing weights"),
    )
}

fn upload_bf16(gpu: &mut Gpu, values: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    let words: Vec<u16> = values.iter().map(|value| f32_to_bf16(*value)).collect();
    let tensor = gpu
        .zeros(shape, DType::BF16)
        .map_err(|error| error.to_string())?;
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 2) };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .map_err(|error| error.to_string())?;
    Ok(tensor)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, count: usize) -> Result<Vec<i64>, String> {
    let mut values = vec![0i32; count];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, count * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .map_err(|error| error.to_string())?;
    Ok(values.into_iter().map(i64::from).collect())
}
fn free(gpu: &mut Gpu, tensor: GpuTensor) -> Result<(), String> {
    gpu.free_tensor(tensor).map_err(|error| error.to_string())
}
fn upload_bf16_words(gpu: &mut Gpu, words: &[u16], shape: &[usize]) -> Result<GpuTensor, String> {
    let tensor = gpu
        .zeros(shape, DType::BF16)
        .map_err(|error| error.to_string())?;
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 2) };
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .map_err(|error| error.to_string())?;
    Ok(tensor)
}

fn upload_raw_i32(gpu: &Gpu, values: &[i64]) -> Result<GpuTensor, String> {
    let words: Vec<i32> = values.iter().map(|value| *value as i32).collect();
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    gpu.upload_raw(bytes, &[words.len() * 4])
        .map_err(|error| error.to_string())
}

fn gpu_linear_rows(
    gpu: &mut Gpu,
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * in_dim || weight.len() != out_dim * in_dim {
        return fail("GPU linear input/weight shape mismatch");
    }
    let weight_gpu = gpu
        .upload_f32(weight, &[out_dim, in_dim])
        .map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(rows * out_dim);
    for row in 0..rows {
        let x = gpu
            .upload_f32(&input[row * in_dim..(row + 1) * in_dim], &[in_dim])
            .map_err(|error| error.to_string())?;
        let y = gpu
            .zeros(&[out_dim], DType::F32)
            .map_err(|error| error.to_string())?;
        gpu.gemm_f32_batched(&x, &weight_gpu, &y, 1, in_dim, out_dim)
            .map_err(|error| error.to_string())?;
        output.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
        free(gpu, x)?;
        free(gpu, y)?;
    }
    free(gpu, weight_gpu)?;
    Ok(output)
}

fn gpu_ple_linear_rows(
    gpu: &mut Gpu,
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, String> {
    if input.len() != rows * in_dim || weight.len() != out_dim * in_dim {
        return fail("GPU PLE linear input/weight shape mismatch");
    }
    let weight_gpu = gpu
        .upload_f32(weight, &[out_dim, in_dim])
        .map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(rows * out_dim);
    for row in 0..rows {
        let x = gpu
            .upload_f32(&input[row * in_dim..(row + 1) * in_dim], &[in_dim])
            .map_err(|error| error.to_string())?;
        let y = gpu
            .zeros(&[out_dim], DType::F32)
            .map_err(|error| error.to_string())?;
        gpu.grouped_linear_f32(&x, &weight_gpu, &y, 1, in_dim, out_dim)
            .map_err(|error| error.to_string())?;
        output.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
        free(gpu, x)?;
        free(gpu, y)?;
    }
    free(gpu, weight_gpu)?;
    Ok(output)
}

fn gpu_silu_values(gpu: &mut Gpu, values: &[f32]) -> Result<Vec<f32>, String> {
    let input = gpu
        .upload_f32(values, &[values.len()])
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[values.len()], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.silu_f32(&input, &output)
        .map_err(|error| error.to_string())?;
    let result = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    free(gpu, input)?;
    free(gpu, output)?;
    Ok(result)
}

fn gpu_sigmoid_values(gpu: &mut Gpu, values: &[f32]) -> Result<Vec<f32>, String> {
    let output = gpu
        .upload_f32(values, &[values.len()])
        .map_err(|error| error.to_string())?;
    gpu.sigmoid_f32(&output)
        .map_err(|error| error.to_string())?;
    let result = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    free(gpu, output)?;
    Ok(result)
}

fn gpu_mul_values(gpu: &mut Gpu, left: &[f32], right: &[f32]) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return fail("GPU multiply shape mismatch");
    }
    let a = gpu
        .upload_f32(left, &[left.len()])
        .map_err(|error| error.to_string())?;
    let b = gpu
        .upload_f32(right, &[right.len()])
        .map_err(|error| error.to_string())?;
    let c = gpu
        .zeros(&[left.len()], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.mul_f32(&a, &b, &c).map_err(|error| error.to_string())?;
    let result = gpu.download_f32(&c).map_err(|error| error.to_string())?;
    free(gpu, a)?;
    free(gpu, b)?;
    free(gpu, c)?;
    Ok(result)
}

fn gpu_add_values(gpu: &mut Gpu, left: &[f32], right: &[f32]) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return fail("GPU add shape mismatch");
    }
    let a = gpu
        .upload_f32(left, &[left.len()])
        .map_err(|error| error.to_string())?;
    let b = gpu
        .upload_f32(right, &[right.len()])
        .map_err(|error| error.to_string())?;
    let c = gpu
        .zeros(&[left.len()], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.add_f32(&a, &b, &c).map_err(|error| error.to_string())?;
    let result = gpu.download_f32(&c).map_err(|error| error.to_string())?;
    free(gpu, a)?;
    free(gpu, b)?;
    free(gpu, c)?;
    Ok(result)
}

fn gpu_rms_rows(
    gpu: &mut Gpu,
    values: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    if values.len() != rows * width {
        return fail("GPU RMS input shape mismatch");
    }
    let weight = gpu
        .upload_f32(&vec![1.0; width], &[width])
        .map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(values.len());
    for row in 0..rows {
        let x = gpu
            .upload_f32(&values[row * width..(row + 1) * width], &[width])
            .map_err(|error| error.to_string())?;
        let y = gpu
            .zeros(&[width], DType::F32)
            .map_err(|error| error.to_string())?;
        gpu.rmsnorm_f32(&x, &weight, &y, RMS_EPS)
            .map_err(|error| error.to_string())?;
        output.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
        free(gpu, x)?;
        free(gpu, y)?;
    }
    free(gpu, weight)?;
    Ok(output)
}

fn gpu_ple_norm_rows(
    gpu: &mut Gpu,
    values: &[f32],
    rows: usize,
    groups: usize,
    group_size: usize,
    weight: &[f32],
) -> Result<Vec<f32>, String> {
    let width = groups * group_size;
    if values.len() != rows * width || weight.len() != width {
        return fail("GPU grouped norm shape mismatch");
    }
    let input = gpu
        .upload_f32(values, &[rows, width])
        .map_err(|error| error.to_string())?;
    let norm = gpu
        .upload_f32(weight, &[width])
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[rows, width], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.grouped_norm_f32(&input, &norm, &output, rows, groups, group_size, RMS_EPS)
        .map_err(|error| error.to_string())?;
    let result = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    free(gpu, input)?;
    free(gpu, norm)?;
    free(gpu, output)?;
    Ok(result)
}

fn shifted_tokens(tokens: &[i64], previous: &[i64], shift: usize, eos_token_id: u32) -> Vec<i64> {
    let mut history = Vec::with_capacity(previous.len() + tokens.len());
    history.extend_from_slice(previous);
    history.extend_from_slice(tokens);
    let mut eos_positions = vec![-1isize; history.len()];
    for (index, value) in history.iter().enumerate() {
        if *value as u32 == eos_token_id {
            eos_positions[index] = index as isize;
        }
    }
    let mut running = -1isize;
    let mut previous_eos = vec![-1isize];
    for value in eos_positions {
        running = running.max(value);
        previous_eos.push(running);
    }
    let mut output = Vec::with_capacity(history.len());
    for position in 0..history.len() {
        let segment_start = previous_eos[position] + 1;
        let source = position.saturating_sub(shift);
        let valid = position as isize - segment_start >= shift as isize && position >= shift;
        output.push(if valid {
            history[source]
        } else {
            eos_token_id as i64
        });
    }
    output
}
fn run_ple_hash(gpu: &mut Gpu, arrays: &BTreeMap<String, Array>) -> Result<Value, String> {
    let tokens = required(arrays, "tokens")?.ints()?;
    let previous = required(arrays, "previous_context")?.ints()?;
    let token_history = required(arrays, "token_history")?.ints()?;
    let shifted = required(arrays, "shifted_tokens")?.ints()?;
    let expected_ids = required(arrays, "ple_row_ids")?.ints()?;
    if previous.len() != 2 || expected_ids.len() != tokens.len() * PLE_HEADS {
        return fail("PLE hash fixture has invalid dimensions");
    }
    let multipliers = required(arrays, "multipliers")?.ints()?.to_vec();
    let vocab_sizes = required(arrays, "head_vocab_sizes")?
        .ints()?
        .iter()
        .map(|value| u64::try_from(*value).map_err(|_| "PLE vocab size is negative".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = required(arrays, "head_offsets")?
        .ints()?
        .iter()
        .map(|value| u64::try_from(*value).map_err(|_| "PLE head offset is negative".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let eos_token_id =
        u32::try_from(previous[0]).map_err(|_| "PLE EOS token is out of range".to_string())?;
    let metadata = PleHashMetadata::from_slices(
        &multipliers,
        &vocab_sizes,
        &offsets,
        hipfire_arch_qwen4::PLE_PADDED_ROWS,
    )
    .map_err(|error| format!("fixture PLE metadata is invalid: {error}"))?;
    compare_i64(
        &metadata.multipliers().iter().copied().collect::<Vec<_>>(),
        &multipliers,
        "PLE multipliers",
    )?;
    compare_i64(
        &metadata
            .head_vocab_sizes()
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
        required(arrays, "head_vocab_sizes")?.ints()?,
        "PLE vocab sizes",
    )?;
    compare_i64(
        &metadata
            .head_offsets()
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
        required(arrays, "head_offsets")?.ints()?,
        "PLE head offsets",
    )?;
    compare_i64(
        &[previous[0], previous[1]],
        required(arrays, "previous_context")?.ints()?,
        "PLE previous context",
    )?;
    let mut history =
        PleHistory::from_previous(eos_token_id, [previous[0] as u32, previous[1] as u32]);
    let mut actual_ids = Vec::with_capacity(expected_ids.len());
    for token in tokens {
        actual_ids.extend(
            history
                .hash_token(&metadata, *token as u32)
                .iter()
                .map(|value| *value as i64),
        );
    }
    compare_i64(&actual_ids, expected_ids, "PLE row IDs")?;
    let mut actual_history = previous.to_vec();
    actual_history.extend_from_slice(tokens);
    compare_i64(&actual_history, token_history, "PLE token history")?;
    let mut actual_shifted = Vec::with_capacity(shifted.len());
    for shift in 0..3 {
        actual_shifted.extend(shifted_tokens(tokens, previous, shift, eos_token_id));
    }
    compare_i64(&actual_shifted, shifted, "PLE shifted tokens")?;

    let row_words = required(arrays, "ple_rows_bf16")?.ints()?;
    if row_words.len() != tokens.len() * PLE_HEADS * PLE_WIDTH {
        return fail("PLE BF16 row shape mismatch");
    }
    let words: Vec<u16> = row_words.iter().map(|value| *value as u16).collect();
    let staged = upload_bf16_words(gpu, &words, &[tokens.len(), PLE_HEADS, PLE_WIDTH])?;
    let widened = gpu
        .zeros(&[tokens.len(), PLE_HEADS * PLE_WIDTH], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.grouped_gather_convert_bf16(&staged, &widened, tokens.len(), PLE_HEADS, PLE_WIDTH)
        .map_err(|error| error.to_string())?;
    let actual_embedding = gpu
        .download_f32(&widened)
        .map_err(|error| error.to_string())?;
    let (max_abs, max_rel) = compare_f32(
        &actual_embedding,
        &required(arrays, "ple_embedding_f32")?.values,
        0.0,
        0.0,
        "PLE BF16 gather",
    )?;
    free(gpu, staged)?;
    free(gpu, widened)?;
    Ok(json!({
        "case":"ple_hash_history",
        "status":"pass",
        "checks":["production PleHashMetadata/PleHistory","fixture metadata arrays","EOS/history/shift byte-exact","production BF16 row gather"],
        "elements":actual_ids.len(),
        "embedding_max_abs":max_abs,
        "embedding_max_rel":max_rel
    }))
}

fn run_ple_projection(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let embedding = required(arrays, "ple_embedding_f32")?;
    let hidden = required(arrays, "hidden_states")?;
    let key_weight = required(arrays, "key_proj")?;
    let value_weight = required(arrays, "value_proj")?;
    let key_norm_weight = required(arrays, "norm_key_weight")?;
    let query_norm_weight = required(arrays, "norm_query_weight")?;
    let conv_norm_weight = required(arrays, "norm_conv_weight")?;
    let conv_weight = required(arrays, "conv1d_weight")?;
    let tokens = embedding.shape[0];
    let ple_dim = embedding.shape[1];
    let channels = hidden.shape[1];
    let hc_count = key_norm_weight.shape[0];
    let hidden_size = key_norm_weight.shape[1];
    let kernel_size = conv_weight.shape[1];
    shape(hidden, &[tokens, channels], "hidden_states")?;
    shape(key_weight, &[channels, ple_dim], "key_proj")?;
    shape(value_weight, &[hidden_size, ple_dim], "value_proj")?;
    shape(key_norm_weight, &[hc_count, hidden_size], "norm_key_weight")?;
    shape(
        query_norm_weight,
        &[hc_count, hidden_size],
        "norm_query_weight",
    )?;
    shape(
        conv_norm_weight,
        &[hc_count, hidden_size],
        "norm_conv_weight",
    )?;
    shape(conv_weight, &[channels, kernel_size], "conv1d_weight")?;
    let (atol, rtol) = tolerance(manifest, "f32_accumulation")?;
    let key_actual = gpu_ple_linear_rows(
        gpu,
        &embedding.values,
        tokens,
        ple_dim,
        &key_weight.values,
        channels,
    )?;
    let value_actual = gpu_ple_linear_rows(
        gpu,
        &embedding.values,
        tokens,
        ple_dim,
        &value_weight.values,
        hidden_size,
    )?;

    let key_abs = (0.0f32, 0.0f32);
    let value_abs = compare_f32(
        &value_actual,
        &required(arrays, "value")?.values,
        atol,
        rtol,
        "PLE value projection",
    )?;

    let key_norm_actual = gpu_ple_norm_rows(
        gpu,
        &key_actual,
        tokens,
        hc_count,
        hidden_size,
        &key_norm_weight.values,
    )?;
    let query_norm_actual = gpu_ple_norm_rows(
        gpu,
        &hidden.values,
        tokens,
        hc_count,
        hidden_size,
        &query_norm_weight.values,
    )?;
    let key_norm_err = compare_f32(
        &key_norm_actual,
        &required(arrays, "key_normed")?.values,
        atol,
        rtol,
        "PLE key norm",
    )?;
    let query_norm_err = compare_f32(
        &query_norm_actual,
        &required(arrays, "query_normed")?.values,
        atol,
        rtol,
        "PLE query norm",
    )?;

    let key_gpu = gpu
        .upload_f32(&key_actual, &[tokens, channels])
        .map_err(|error| error.to_string())?;
    let query_gpu = gpu
        .upload_f32(&hidden.values, &[tokens, channels])
        .map_err(|error| error.to_string())?;
    let value_gpu = gpu
        .upload_f32(&value_actual, &[tokens, hidden_size])
        .map_err(|error| error.to_string())?;
    let key_norm_gpu = gpu
        .upload_f32(&key_norm_weight.values, &[channels])
        .map_err(|error| error.to_string())?;
    let query_norm_gpu = gpu
        .upload_f32(&query_norm_weight.values, &[channels])
        .map_err(|error| error.to_string())?;
    let gated_gpu = gpu
        .zeros(&[tokens, channels], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.grouped_gate_f32(
        &key_gpu,
        &query_gpu,
        &value_gpu,
        &key_norm_gpu,
        &query_norm_gpu,
        &gated_gpu,
        tokens,
        hc_count,
        hidden_size,
        RMS_EPS,
    )
    .map_err(|error| error.to_string())?;
    let gated_actual = gpu
        .download_f32(&gated_gpu)
        .map_err(|error| error.to_string())?;
    let gated_err = compare_f32(
        &gated_actual,
        &required(arrays, "gated_value")?.values,
        atol,
        rtol,
        "PLE gated value",
    )?;
    let mut gate_actual = Vec::with_capacity(tokens * hc_count);
    for token in 0..tokens {
        for branch in 0..hc_count {
            let base = token * channels + branch * hidden_size;
            let mut key_sq = 0.0f32;
            let mut query_sq = 0.0f32;
            for index in 0..hidden_size {
                key_sq += key_actual[base + index] * key_actual[base + index];
                query_sq += hidden.values[base + index] * hidden.values[base + index];
            }
            let key_inv = (key_sq / hidden_size as f32 + RMS_EPS).sqrt().recip();
            let query_inv = (query_sq / hidden_size as f32 + RMS_EPS).sqrt().recip();
            let mut dot = 0.0f32;
            for index in 0..hidden_size {
                let k = key_actual[base + index]
                    * key_inv
                    * (1.0 + key_norm_weight.values[branch * hidden_size + index]);
                let q = hidden.values[base + index]
                    * query_inv
                    * (1.0 + query_norm_weight.values[branch * hidden_size + index]);
                dot += k * q;
            }
            let transformed = dot / (hidden_size as f32).sqrt();
            gate_actual.push(if transformed >= 0.0 {
                transformed.abs().sqrt()
            } else {
                -transformed.abs().sqrt()
            });
        }
    }
    let gate_err = compare_f32(
        &gate_actual,
        &required(arrays, "gate")?.values,
        atol,
        rtol,
        "PLE gate logits",
    )?;

    let conv_norm_actual = gpu_ple_norm_rows(
        gpu,
        &gated_actual,
        tokens,
        hc_count,
        hidden_size,
        &conv_norm_weight.values,
    )?;
    let conv_norm_err = compare_f32(
        &conv_norm_actual,
        &required(arrays, "gated_value_normed")?.values,
        atol,
        rtol,
        "PLE convolution norm",
    )?;
    let conv_weight_gpu = gpu
        .upload_f32(&conv_weight.values, &[channels, kernel_size])
        .map_err(|error| error.to_string())?;
    let state_rows = (kernel_size - 1) * 3;
    let state = gpu
        .zeros(&[state_rows, channels], DType::F32)
        .map_err(|error| error.to_string())?;
    let mut fused_actual = Vec::with_capacity(tokens * channels);
    for (begin, end) in [(0usize, 4usize), (4usize, tokens)] {
        let gated_chunk = gpu
            .upload_f32(
                &gated_actual[begin * channels..end * channels],
                &[end - begin, channels],
            )
            .map_err(|error| error.to_string())?;
        let norm_chunk = gpu
            .upload_f32(
                &conv_norm_actual[begin * channels..end * channels],
                &[end - begin, channels],
            )
            .map_err(|error| error.to_string())?;
        let output = gpu
            .zeros(&[end - begin, channels], DType::F32)
            .map_err(|error| error.to_string())?;
        gpu.grouped_depthwise_conv_silu_add_f32(
            &gated_chunk,
            &norm_chunk,
            &conv_weight_gpu,
            &state,
            &output,
            end - begin,
            channels,
            kernel_size,
            3,
        )
        .map_err(|error| error.to_string())?;
        fused_actual.extend(
            gpu.download_f32(&output)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, gated_chunk)?;
        free(gpu, norm_chunk)?;
        free(gpu, output)?;
    }
    let actual_state = gpu
        .download_f32(&state)
        .map_err(|error| error.to_string())?;
    let mut conv_actual = Vec::with_capacity(fused_actual.len());
    for (fused, gated) in fused_actual.iter().zip(&gated_actual) {
        conv_actual.push(fused - gated);
    }
    let conv_err = compare_f32(
        &conv_actual,
        &required(arrays, "conv_output")?.values,
        atol,
        rtol,
        "PLE convolution output",
    )?;
    let output_err = compare_f32(
        &fused_actual,
        &required(arrays, "output")?.values,
        atol,
        rtol,
        "PLE fused output",
    )?;
    let state_err = compare_f32(
        &actual_state,
        &required(arrays, "conv_history")?.values,
        atol,
        rtol,
        "PLE convolution history",
    )?;
    free(gpu, key_gpu)?;
    free(gpu, query_gpu)?;
    free(gpu, value_gpu)?;
    free(gpu, key_norm_gpu)?;
    free(gpu, query_norm_gpu)?;
    free(gpu, gated_gpu)?;
    free(gpu, conv_weight_gpu)?;
    free(gpu, state)?;
    Ok(json!({
        "case":"ple_projection_dilated_conv",
        "status":"pass",
        "layer_index":1,
        "token_boundary":4,
        "checks":["production PLE gather-fed linear key/value","production grouped norm","production branch gate","production dilated convolution/SiLU/residual","state boundary"],
        "key_max_abs":key_abs.0,
        "value_max_abs":value_abs.0,
        "key_norm_max_abs":key_norm_err.0,
        "query_norm_max_abs":query_norm_err.0,
        "gate_max_abs":gate_err.0,
        "gated_max_abs":gated_err.0,
        "conv_norm_max_abs":conv_norm_err.0,
        "conv_max_abs":conv_err.0,
        "output_max_abs":output_err.0,
        "state_max_abs":state_err.0
    }))
}

fn run_gdn_conv(
    gpu: &mut Gpu,
    input: &[f32],
    weight: &[f32],
    tokens: usize,
    channels: usize,
    segments: &[(usize, usize)],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let weight_gpu = gpu
        .upload_f32(weight, &[channels, 4])
        .map_err(|error| error.to_string())?;
    let state = gpu
        .zeros(&[channels * 3], DType::F32)
        .map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(tokens * channels);
    for &(begin, end) in segments {
        if begin > end || end > tokens {
            return fail("GDN convolution segment is out of bounds");
        }
        for token in begin..end {
            let x = gpu
                .upload_f32(
                    &input[token * channels..(token + 1) * channels],
                    &[channels],
                )
                .map_err(|error| error.to_string())?;
            let y = gpu
                .zeros(&[channels], DType::F32)
                .map_err(|error| error.to_string())?;
            gpu.conv1d_silu_f32(&y, &x, &weight_gpu, &state, channels)
                .map_err(|error| error.to_string())?;
            output.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
            free(gpu, x)?;
            free(gpu, y)?;
        }
    }
    let channel_state = gpu
        .download_f32(&state)
        .map_err(|error| error.to_string())?;
    let mut history = vec![0.0f32; 3 * channels];
    for row in 0..3 {
        for channel in 0..channels {
            history[row * channels + channel] = channel_state[channel * 3 + (2 - row)];
        }
    }
    free(gpu, weight_gpu)?;
    free(gpu, state)?;
    Ok((output, history))
}

fn run_gdn_params(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let a = required(arrays, "a_logits")?;
    let b = required(arrays, "b_logits")?;
    let a_log = required(arrays, "a_log")?;
    let dt_bias = required(arrays, "dt_bias")?;
    let tokens = a.shape[0];
    let heads = a.shape[1];
    let a_log_gpu = gpu
        .upload_f32(&a_log.values, &[heads])
        .map_err(|error| error.to_string())?;
    let dt_gpu = gpu
        .upload_f32(&dt_bias.values, &[heads])
        .map_err(|error| error.to_string())?;
    let mut gate = Vec::with_capacity(tokens * heads);
    let mut beta = Vec::with_capacity(tokens * heads);
    for token in 0..tokens {
        let a_gpu = gpu
            .upload_f32(a.row(token)?, &[heads])
            .map_err(|error| error.to_string())?;
        let b_gpu = gpu
            .upload_f32(b.row(token)?, &[heads])
            .map_err(|error| error.to_string())?;
        let gate_gpu = gpu
            .zeros(&[heads], DType::F32)
            .map_err(|error| error.to_string())?;
        let beta_gpu = gpu
            .zeros(&[heads], DType::F32)
            .map_err(|error| error.to_string())?;
        gated_delta_params_f32(
            gpu,
            &rdna_compute::tensor_ops::GatedDeltaParams {
                a: &a_gpu,
                b: &b_gpu,
                a_log: &a_log_gpu,
                dt_bias: &dt_gpu,
                gate: &gate_gpu,
                beta: &beta_gpu,
            },
            heads,
        )
        .map_err(|error| error.to_string())?;
        gate.extend(
            gpu.download_f32(&gate_gpu)
                .map_err(|error| error.to_string())?,
        );
        beta.extend(
            gpu.download_f32(&beta_gpu)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, a_gpu)?;
        free(gpu, b_gpu)?;
        free(gpu, gate_gpu)?;
        free(gpu, beta_gpu)?;
    }
    free(gpu, a_log_gpu)?;
    free(gpu, dt_gpu)?;
    Ok((gate, beta))
}

fn run_gdn_sequence(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    gate: &[f32],
    beta: &[f32],
    segments: &[(usize, usize)],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let q = required(arrays, "query_16x4")?;
    let k = required(arrays, "key_16x4")?;
    let v = required(arrays, "value_48x4")?;
    let tokens = q.shape[0];
    let key_heads = q.shape[1];
    let key_dim = q.shape[2];
    let value_heads = v.shape[1];
    let value_dim = v.shape[2];
    let state = gpu
        .zeros(&[value_heads, key_dim, value_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[value_heads, value_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let mut actual = Vec::with_capacity(tokens * value_heads * value_dim);
    for &(begin, end) in segments {
        for token in begin..end {
            let q_t = gpu
                .upload_f32(q.row(token)?, &[key_heads * key_dim])
                .map_err(|error| error.to_string())?;
            let k_t = gpu
                .upload_f32(k.row(token)?, &[key_heads * key_dim])
                .map_err(|error| error.to_string())?;
            let v_t = gpu
                .upload_f32(v.row(token)?, &[value_heads * value_dim])
                .map_err(|error| error.to_string())?;
            let gate_t = gpu
                .upload_f32(
                    &gate[token * value_heads..(token + 1) * value_heads],
                    &[value_heads],
                )
                .map_err(|error| error.to_string())?;
            let beta_t = gpu
                .upload_f32(
                    &beta[token * value_heads..(token + 1) * value_heads],
                    &[value_heads],
                )
                .map_err(|error| error.to_string())?;
            gated_delta_step(
                gpu,
                &GatedDeltaStep {
                    q: &q_t,
                    k: &k_t,
                    v: &v_t,
                    gate: &gate_t,
                    beta: &beta_t,
                    state: &state,
                    output: &output,
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                },
            )
            .map_err(|error| error.to_string())?;
            actual.extend(
                gpu.download_f32(&output)
                    .map_err(|error| error.to_string())?,
            );
            free(gpu, q_t)?;
            free(gpu, k_t)?;
            free(gpu, v_t)?;
            free(gpu, gate_t)?;
            free(gpu, beta_t)?;
        }
    }
    let state_values = gpu
        .download_f32(&state)
        .map_err(|error| error.to_string())?;
    free(gpu, state)?;
    free(gpu, output)?;
    Ok((actual, state_values))
}

fn run_gdn(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let mixed = required(arrays, "mixed_qkv")?;
    let conv_weight = required(arrays, "conv_weight")?;
    let tokens = mixed.shape[0];
    let channels = mixed.shape[1];
    let (conv_actual, conv_history) = run_gdn_conv(
        gpu,
        &mixed.values,
        &conv_weight.values,
        tokens,
        channels,
        &[(0, tokens)],
    )?;
    let (conv_chunked, conv_chunk_history) = run_gdn_conv(
        gpu,
        &mixed.values,
        &conv_weight.values,
        tokens,
        channels,
        &[(0, 3), (3, tokens)],
    )?;
    let (atol, rtol) = tolerance(manifest, "f32_state_recurrence")?;
    let (source_atol, source_rtol) = tolerance(manifest, "bf16_input_f32_accumulation")?;
    let conv_err = compare_f32(
        &conv_actual,
        &required(arrays, "conv_output")?.values,
        atol,
        rtol,
        "GDN convolution",
    )?;
    let source_conv_err = measure_f32(
        &conv_actual,
        &required(arrays, "source_conv_output")?.values,
        source_atol,
        source_rtol,
        "GDN convolution vs source",
    )?;
    let conv_state_err = compare_f32(
        &conv_history,
        &required(arrays, "conv_final_history")?.values,
        atol,
        rtol,
        "GDN convolution history",
    )?;
    compare_f32(
        &conv_chunked,
        &conv_actual,
        atol,
        rtol,
        "GDN convolution chunking",
    )?;
    compare_f32(
        &conv_chunk_history,
        &conv_history,
        atol,
        rtol,
        "GDN convolution chunk state",
    )?;

    let q = required(arrays, "query_16x4")?;
    let k = required(arrays, "key_16x4")?;
    let value = required(arrays, "value_48x4")?;
    let tokens = q.shape[0];
    let key_heads = q.shape[1];
    let key_dim = q.shape[2];
    let value_heads = value.shape[1];
    let value_dim = value.shape[2];
    let ratio = value_heads / key_heads;
    let q_gpu = gpu
        .upload_f32(&q.values, &[tokens, key_heads * key_dim])
        .map_err(|error| error.to_string())?;
    let k_gpu = gpu
        .upload_f32(&k.values, &[tokens, key_heads * key_dim])
        .map_err(|error| error.to_string())?;
    let q_exp_gpu = gpu
        .zeros(&[tokens, value_heads * key_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let k_exp_gpu = gpu
        .zeros(&[tokens, value_heads * key_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.repeat_interleave_qk_f32_batched(
        &q_gpu, &k_gpu, &q_exp_gpu, &k_exp_gpu, key_heads, ratio, key_dim, tokens,
    )
    .map_err(|error| error.to_string())?;
    let q_exp_actual = gpu
        .download_f32(&q_exp_gpu)
        .map_err(|error| error.to_string())?;
    let k_exp_actual = gpu
        .download_f32(&k_exp_gpu)
        .map_err(|error| error.to_string())?;
    let expansion_err = compare_f32(
        &q_exp_actual,
        &required(arrays, "query_expanded_48x4")?.values,
        atol,
        rtol,
        "GDN Q expansion",
    )?;
    let k_exp_err = compare_f32(
        &k_exp_actual,
        &required(arrays, "key_expanded_48x4")?.values,
        atol,
        rtol,
        "GDN K expansion",
    )?;
    free(gpu, q_gpu)?;
    free(gpu, k_gpu)?;
    free(gpu, q_exp_gpu)?;
    free(gpu, k_exp_gpu)?;

    let q_l2_gpu = gpu
        .zeros(&[tokens, value_heads * key_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let k_l2_gpu = gpu
        .zeros(&[tokens, value_heads * key_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let q_src = gpu
        .upload_f32(&q.values, &[tokens, key_heads * key_dim])
        .map_err(|error| error.to_string())?;
    let k_src = gpu
        .upload_f32(&k.values, &[tokens, key_heads * key_dim])
        .map_err(|error| error.to_string())?;
    gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
        &q_src, &k_src, &q_l2_gpu, &k_l2_gpu, key_heads, ratio, key_dim, 1.0, RMS_EPS, tokens,
    )
    .map_err(|error| error.to_string())?;
    let q_l2 = gpu
        .download_f32(&q_l2_gpu)
        .map_err(|error| error.to_string())?;
    let k_l2 = gpu
        .download_f32(&k_l2_gpu)
        .map_err(|error| error.to_string())?;
    let q_l2_err = measure_f32(
        &q_l2,
        &required(arrays, "source_query_l2")?.values,
        source_atol,
        source_rtol,
        "GDN Q L2 vs source BF16 stage",
    )?;
    let k_l2_err = measure_f32(
        &k_l2,
        &required(arrays, "source_key_l2")?.values,
        source_atol,
        source_rtol,
        "GDN K L2 vs source BF16 stage",
    )?;
    free(gpu, q_l2_gpu)?;
    free(gpu, k_l2_gpu)?;
    free(gpu, q_src)?;
    free(gpu, k_src)?;

    let (gate, beta) = run_gdn_params(gpu, arrays)?;
    let gate_err = compare_f32(
        &gate,
        &required(arrays, "g_decay")?.values,
        atol,
        rtol,
        "GDN decay parameters",
    )?;
    let beta_err = compare_f32(
        &beta,
        &required(arrays, "beta_sigmoid")?.values,
        atol,
        rtol,
        "GDN beta parameters",
    )?;
    let (whole, whole_state) = run_gdn_sequence(gpu, arrays, &gate, &beta, &[(0, tokens)])?;
    let (chunked, chunked_state) =
        run_gdn_sequence(gpu, arrays, &gate, &beta, &[(0, 3), (3, tokens)])?;
    let core_err = compare_f32(
        &whole,
        &required(arrays, "candidate_core_attention_output")?.values,
        atol,
        rtol,
        "GDN recurrent output candidate",
    )?;
    let source_core_err = measure_f32(
        &whole,
        &required(arrays, "source_core_attention_output")?.values,
        source_atol,
        source_rtol,
        "GDN recurrent output vs source",
    )?;
    let state_err = compare_f32(
        &whole_state,
        &required(arrays, "candidate_final_recurrent_state")?.values,
        atol,
        rtol,
        "GDN recurrent state candidate",
    )?;
    let source_state_err = measure_f32(
        &whole_state,
        &required(arrays, "source_final_recurrent_state")?.values,
        source_atol,
        source_rtol,
        "GDN recurrent state vs source",
    )?;
    compare_f32(&chunked, &whole, atol, rtol, "GDN recurrent chunking")?;
    compare_f32(
        &chunked_state,
        &whole_state,
        atol,
        rtol,
        "GDN recurrent chunk state",
    )?;
    // HF casts the recurrent output through BF16 before RMSNormGated.  Keep
    // this as an explicit device-side boundary in the physical oracle: one
    // BF16 scratch and one F32 destination are reused for every token.
    let recurrent_bf16 = gpu
        .zeros(&[value_heads * value_dim], DType::BF16)
        .map_err(|error| error.to_string())?;
    let recurrent_rounded = gpu
        .zeros(&[value_heads * value_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let mut rounded_whole = Vec::with_capacity(whole.len());
    for token in 0..tokens {
        let recurrent = gpu
            .upload_f32(
                &whole[token * value_heads * value_dim..(token + 1) * value_heads * value_dim],
                &[value_heads * value_dim],
            )
            .map_err(|error| error.to_string())?;
        bf16_roundtrip_f32(
            gpu,
            &Bf16Roundtrip {
                input: &recurrent,
                scratch: &recurrent_bf16,
                output: &recurrent_rounded,
                elements: value_heads * value_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        rounded_whole.extend(
            gpu.download_f32(&recurrent_rounded)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, recurrent)?;
    }
    let rounded_candidate_err = measure_f32(
        &rounded_whole,
        &required(arrays, "candidate_bf16_core_attention_output_f32")?.values,
        source_atol,
        source_rtol,
        "GDN BF16-rounded recurrent output candidate",
    )?;
    let rounded_source_err = measure_f32(
        &rounded_whole,
        &required(arrays, "source_core_attention_output")?.values,
        source_atol,
        source_rtol,
        "GDN BF16-rounded recurrent output source",
    )?;

    let norm_weight = vec![0.0f32; value_dim];
    let core_norm_raw = gpu_ple_norm_rows(
        gpu,
        &rounded_whole,
        tokens * value_heads,
        1,
        value_dim,
        &norm_weight,
    )?;
    let mut core_norm = Vec::with_capacity(core_norm_raw.len());
    for token in 0..tokens {
        let norm_input = gpu
            .upload_f32(
                &core_norm_raw
                    [token * value_heads * value_dim..(token + 1) * value_heads * value_dim],
                &[value_heads * value_dim],
            )
            .map_err(|error| error.to_string())?;
        bf16_roundtrip_f32(
            gpu,
            &Bf16Roundtrip {
                input: &norm_input,
                scratch: &recurrent_bf16,
                output: &recurrent_rounded,
                elements: value_heads * value_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        core_norm.extend(
            gpu.download_f32(&recurrent_rounded)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, norm_input)?;
    }

    let norm_err = compare_f32(
        &core_norm,
        &required(arrays, "candidate_core_norm")?.values,
        atol,
        rtol,
        "GDN output norm candidate",
    )?;
    let source_norm_err = measure_f32(
        &core_norm,
        &required(arrays, "source_core_norm")?.values,
        source_atol,
        source_rtol,
        "GDN output norm vs source",
    )?;
    let z = required(arrays, "z_output_gate")?;
    // The oracle's gate norm weight is 1.0. `grouped_norm_f32` above is
    // zero-centered (zeros mean 1), but the gate kernel applies its weight
    // directly, as production does with the checkpoint's `linear_attn.norm`.
    let norm_bf16 = upload_bf16(gpu, &vec![1.0f32; value_dim], &[value_dim])?;
    let mut gated_output = Vec::with_capacity(whole.len());
    for token in 0..tokens {
        let recurrent = gpu
            .upload_f32(
                &rounded_whole
                    [token * value_heads * value_dim..(token + 1) * value_heads * value_dim],
                &[value_heads * value_dim],
            )
            .map_err(|error| error.to_string())?;
        let z_gpu = gpu
            .upload_f32(z.row(token)?, &[value_heads * value_dim])
            .map_err(|error| error.to_string())?;
        let output = gpu
            .zeros(&[value_heads * value_dim], DType::F32)
            .map_err(|error| error.to_string())?;
        rdna_compute::tensor_ops::gated_delta_gate(
            gpu,
            &rdna_compute::tensor_ops::GatedDeltaGate {
                recurrent_output: &recurrent,
                z: &z_gpu,
                norm: &norm_bf16,
                output: &output,
                value_heads,
                value_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        gated_output.extend(
            gpu.download_f32(&output)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, recurrent)?;
        free(gpu, z_gpu)?;
        free(gpu, output)?;
    }
    let gated_err = compare_f32(
        &gated_output,
        &required(arrays, "candidate_output_gate_sigmoid")?.values,
        atol,
        rtol,
        "GDN gated output candidate",
    )?;
    let source_gated_err = measure_f32(
        &gated_output,
        &required(arrays, "source_output_gate_sigmoid")?.values,
        source_atol,
        source_rtol,
        "GDN gated output vs source",
    )?;
    let projected = gpu_ple_linear_rows(
        gpu,
        &gated_output,
        tokens,
        value_heads * value_dim,
        &required(arrays, "out_proj")?.values,
        required(arrays, "out_proj")?.shape[0],
    )?;
    let output_err = compare_f32(
        &projected,
        &required(arrays, "candidate_output")?.values,
        atol,
        rtol,
        "GDN output projection candidate",
    )?;
    let source_output_err = measure_f32(
        &projected,
        &required(arrays, "source_output")?.values,
        source_atol,
        source_rtol,
        "GDN output projection vs source",
    )?;
    free(gpu, recurrent_bf16)?;
    free(gpu, recurrent_rounded)?;
    free(gpu, norm_bf16)?;
    if !(source_conv_err.2
        && source_core_err.2
        && source_state_err.2
        && rounded_source_err.2
        && rounded_candidate_err.2
        && source_norm_err.2
        && source_gated_err.2
        && source_output_err.2)
    {
        return fail(format!(
            "GDN source BF16 parity exceeded frozen tolerance: conv={} core={} state={} rounded_source={} rounded_candidate={} norm={} gated={} output={} (max_abs conv={:.8e} core={:.8e} state={:.8e} rounded_source={:.8e} rounded_candidate={:.8e} norm={:.8e} gated={:.8e} output={:.8e})",
            source_conv_err.2,
            source_core_err.2,
            source_state_err.2,
            rounded_source_err.2,
            rounded_candidate_err.2,
            source_norm_err.2,
            source_gated_err.2,
            source_output_err.2,
            source_conv_err.0,
            source_core_err.0,
            source_state_err.0,
            rounded_source_err.0,
            rounded_candidate_err.0,
            source_norm_err.0,
            source_gated_err.0,
            source_output_err.0,
        ));
    }
    Ok(json!({
        "case":"gdn_recurrence_conv_head_expansion",
        "status":"pass",
        "chunk_boundary":3,
        "checks":["production GDN conv/SiLU","production Q/K expansion","production Q/K L2","production decay/beta","production recurrent state","production gated norm/output"],
        "conv_max_abs":conv_err.0,
        "source_conv_max_abs":source_conv_err.0,
        "source_conv_within_tolerance":source_conv_err.2,
        "conv_state_max_abs":conv_state_err.0,
        "expansion_max_abs":expansion_err.0.max(k_exp_err.0),
        "q_l2_max_abs":q_l2_err.0.max(k_l2_err.0),
        "params_max_abs":gate_err.0.max(beta_err.0),
        "core_max_abs":core_err.0,
        "source_core_max_abs":source_core_err.0,
        "source_core_within_tolerance":source_core_err.2,
        "state_max_abs":state_err.0,
        "rounded_source_max_abs":rounded_source_err.0,
        "rounded_source_within_tolerance":rounded_source_err.2,
        "rounded_candidate_max_abs":rounded_candidate_err.0,
        "rounded_candidate_within_tolerance":rounded_candidate_err.2,
        "source_state_max_abs":source_state_err.0,
        "source_state_within_tolerance":source_state_err.2,
        "norm_max_abs":norm_err.0,
        "source_norm_max_abs":source_norm_err.0,
        "source_norm_within_tolerance":source_norm_err.2,
        "gated_max_abs":gated_err.0,
        "source_gated_max_abs":source_gated_err.0,
        "source_gated_within_tolerance":source_gated_err.2,
        "output_max_abs":output_err.0,
        "source_output_max_abs":source_output_err.0,
        "source_output_within_tolerance":source_output_err.2
    }))
}

fn rope_reference(
    values: &[f32],
    rows: usize,
    heads: usize,
    dim: usize,
    positions: &[usize],
    base: f32,
    normalize: bool,
) -> Vec<f32> {
    let mut output = values.to_vec();
    let half = dim / 2;
    for row in 0..rows {
        for head in 0..heads {
            let offset = (row * heads + head) * dim;
            let inv = if normalize {
                (values[offset..offset + dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / dim as f32
                    + RMS_EPS)
                    .sqrt()
                    .recip()
            } else {
                1.0
            };
            let mut normalized = vec![0.0f32; dim];
            for index in 0..dim {
                normalized[index] = values[offset + index] * inv;
            }
            for index in 0..half {
                let angle = positions[row] as f32 / base.powf(2.0 * index as f32 / dim as f32);
                let (sine, cosine) = angle.sin_cos();
                let first = normalized[index];
                let second = normalized[index + half];
                output[offset + index] = first * cosine - second * sine;
                output[offset + index + half] = first * sine + second * cosine;
            }
            if dim > 2 * half {
                for index in 2 * half..dim {
                    output[offset + index] = normalized[index];
                }
            }
        }
    }
    output
}

/// Host model of `indexed_attention_pool_rope_f32`: the pooled mean, its
/// squares, the normalized key and the rotated output are rounded to BF16, and
/// only the first `min(dim, 64)` channels are rotated.
fn pool_rope_reference(
    raw: &[f32],
    blocks: usize,
    compress: usize,
    dim: usize,
    base: f32,
) -> Vec<f32> {
    let bf16 = |value: f32| f32::from_bits((f32_to_bf16(value) as u32) << 16);
    let rotary = dim.min(64);
    let half = rotary / 2;
    let mut pooled = vec![0.0f32; blocks * dim];
    for block in 0..blocks {
        let mut mean = vec![0.0f32; dim];
        for row in 0..compress {
            for index in 0..dim {
                mean[index] += raw[(block * compress + row) * dim + index] / compress as f32;
            }
        }
        for value in &mut mean {
            *value = bf16(*value);
        }
        let inverse = (mean.iter().map(|value| bf16(value * value)).sum::<f32>() / dim as f32
            + RMS_EPS)
            .sqrt()
            .recip();
        let normalized: Vec<f32> = mean.iter().map(|value| bf16(value * inverse)).collect();
        let mut rotated = normalized.clone();
        for index in 0..half {
            let angle = (block * compress) as f32 * base.powf(-2.0 * index as f32 / rotary as f32);
            let (sine, cosine) = angle.sin_cos();
            let first = normalized[index];
            let second = normalized[index + half];
            rotated[index] = first * cosine - second * sine;
            rotated[index + half] = first * sine + second * cosine;
        }
        for (slot, value) in pooled[block * dim..(block + 1) * dim]
            .iter_mut()
            .zip(rotated)
        {
            *slot = bf16(value);
        }
    }
    pooled
}

fn host_qsa_weights(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[i64],
    tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dim: usize,
    capacity: usize,
) -> Vec<f32> {
    let mut result = Vec::new();
    for token in 0..tokens {
        let chosen: Vec<usize> = selected[token * capacity..(token + 1) * capacity]
            .iter()
            .filter_map(|value| (*value >= 0).then_some(*value as usize))
            .collect();
        for head in 0..n_heads {
            let kv_head = head / (n_heads / n_kv_heads);
            let mut logits = Vec::with_capacity(chosen.len());
            for chosen_token in &chosen {
                let qbase = (token * n_heads + head) * dim;
                let kbase = (*chosen_token * n_kv_heads + kv_head) * dim;
                let dot = (0..dim)
                    .map(|index| queries[qbase + index] * keys[kbase + index])
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                logits.push(dot);
            }
            if logits.is_empty() {
                continue;
            }
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials: Vec<f32> =
                logits.iter().map(|value| (value - maximum).exp()).collect();
            let total = exponentials.iter().sum::<f32>();
            result.extend(exponentials.into_iter().map(|value| value / total));
        }
    }
    let _ = values;
    result
}
fn hc_read_case(
    gpu: &mut Gpu,
    input: &[f32],
    tokens: usize,
    branches: usize,
    hidden: usize,
    norm_weight: &[f32],
    down: &[f32],
    up: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let wide = branches * hidden;
    let normed = {
        let norm_gpu = upload_bf16(gpu, norm_weight, &[wide])?;
        let mut result = Vec::with_capacity(input.len());
        for row in 0..tokens {
            let x = gpu
                .upload_f32(&input[row * wide..(row + 1) * wide], &[wide])
                .map_err(|error| error.to_string())?;
            let y = gpu
                .zeros(&[wide], DType::F32)
                .map_err(|error| error.to_string())?;
            rdna_compute::tensor_ops::hyper_norm(
                gpu,
                &rdna_compute::tensor_ops::HyperNorm {
                    input: &x,
                    norm_weight: &norm_gpu,
                    normalized: &y,
                    branches,
                    hidden,
                    state_bf16: false,
                },
            )
            .map_err(|error| error.to_string())?;
            result.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
            free(gpu, x)?;
            free(gpu, y)?;
        }
        free(gpu, norm_gpu)?;
        result
    };
    let mut down_scaled = down.to_vec();
    for value in &mut down_scaled {
        *value /= branches as f32;
    }
    let low_logits = gpu_linear_rows(gpu, &normed, tokens, wide, &down_scaled, down.len() / wide)?;
    let lowrank_silu = gpu_silu_values(gpu, &low_logits)?;
    let mix_logits = gpu_linear_rows(gpu, &lowrank_silu, tokens, down.len() / wide, up, wide)?;
    let mix_gate = gpu_sigmoid_values(gpu, &mix_logits)?;
    let norm_gpu = upload_bf16(gpu, norm_weight, &[wide])?;
    let up_gpu = gpu
        .upload_f32(up, &[wide, down.len() / wide])
        .map_err(|error| error.to_string())?;
    let mut mixed = Vec::with_capacity(tokens * hidden);
    let mut normalized_from_read = Vec::with_capacity(input.len());
    for row in 0..tokens {
        let x = gpu
            .upload_f32(&input[row * wide..(row + 1) * wide], &[wide])
            .map_err(|error| error.to_string())?;
        let low = gpu
            .upload_f32(
                &lowrank_silu[row * (down.len() / wide)..(row + 1) * (down.len() / wide)],
                &[down.len() / wide],
            )
            .map_err(|error| error.to_string())?;
        let normalized = gpu
            .zeros(&[wide], DType::F32)
            .map_err(|error| error.to_string())?;
        let out = gpu
            .zeros(&[hidden], DType::F32)
            .map_err(|error| error.to_string())?;
        hyper_read(
            gpu,
            &HyperRead {
                input: &x,
                norm_weight: &norm_gpu,
                low: &low,
                up: &up_gpu,
                normalized: &normalized,
                mixed: &out,
                branches,
                hidden,
                rank: down.len() / wide,
            },
        )
        .map_err(|error| error.to_string())?;
        normalized_from_read.extend(
            gpu.download_f32(&normalized)
                .map_err(|error| error.to_string())?,
        );
        mixed.extend(gpu.download_f32(&out).map_err(|error| error.to_string())?);
        free(gpu, x)?;
        free(gpu, low)?;
        free(gpu, normalized)?;
        free(gpu, out)?;
    }
    free(gpu, norm_gpu)?;
    free(gpu, up_gpu)?;
    Ok((normalized_from_read, lowrank_silu, mix_gate, mixed))
}

fn hc_inject_case(
    gpu: &mut Gpu,
    input: &[f32],
    normalized: &[f32],
    mixed: &[f32],
    tokens: usize,
    branches: usize,
    hidden: usize,
    inject_weight: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let wide = branches * hidden;
    let logits = gpu_linear_rows(gpu, normalized, tokens, wide, inject_weight, branches)?;
    let scaled: Vec<f32> = logits
        .iter()
        .map(|value| *value / branches as f32)
        .collect();
    let sigmoid = gpu_sigmoid_values(gpu, &scaled)?;
    let weights: Vec<f32> = sigmoid.iter().map(|value| 2.0 * *value).collect();
    let mut output = Vec::with_capacity(input.len());
    for row in 0..tokens {
        let x = gpu
            .upload_f32(&input[row * wide..(row + 1) * wide], &[wide])
            .map_err(|error| error.to_string())?;
        let n = gpu
            .upload_f32(&normalized[row * wide..(row + 1) * wide], &[wide])
            .map_err(|error| error.to_string())?;
        let m = gpu
            .upload_f32(&mixed[row * hidden..(row + 1) * hidden], &[hidden])
            .map_err(|error| error.to_string())?;
        let g = gpu
            .upload_f32(&logits[row * branches..(row + 1) * branches], &[branches])
            .map_err(|error| error.to_string())?;
        let y = gpu
            .zeros(&[wide], DType::F32)
            .map_err(|error| error.to_string())?;
        hyper_write(
            gpu,
            &HyperWrite {
                input: &x,
                normalized: &n,
                mixed: &m,
                gates: &g,
                output: &y,
                branches,
                hidden,
                state_bf16: false,
            },
        )
        .map_err(|error| error.to_string())?;
        output.extend(gpu.download_f32(&y).map_err(|error| error.to_string())?);
        free(gpu, x)?;
        free(gpu, n)?;
        free(gpu, m)?;
        free(gpu, g)?;
        free(gpu, y)?;
    }
    Ok((weights, output))
}
fn run_qsa(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let index_queries = required(arrays, "index_queries")?;
    let raw_keys = required(arrays, "raw_index_keys")?;
    let selected_expected = required(arrays, "selected_indices")?.ints()?;
    let tokens = index_queries.shape[0];
    let index_heads = index_queries.shape[1];
    let index_dim = index_queries.shape[2];
    let capacity = selected_expected.len() / tokens;
    let blocks = tokens / QSA_COMPRESS;
    let (atol, rtol) = tolerance(manifest, "f32_accumulation")?;
    // The pooled keys are BF16 (see `pool_rope_reference`), so they and the
    // block scores built from them use the BF16 tolerance class.
    let (bf16_atol, bf16_rtol) = tolerance(manifest, "bf16_input_f32_accumulation")?;

    let raw_gpu = gpu
        .upload_f32(&raw_keys.values, &[tokens, index_dim])
        .map_err(|error| error.to_string())?;
    let pooled_gpu = gpu
        .zeros(&[blocks, index_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    indexed_attention_pool_rope(
        gpu,
        &IndexedAttentionPoolRope {
            raw_keys: &raw_gpu,
            pooled: &pooled_gpu,
            norm: None,
            block_count: blocks,
            compress: QSA_COMPRESS,
            index_dim,
            position: None,
            grid_bound: blocks,
        },
    )
    .map_err(|error| error.to_string())?;
    let pooled_actual = gpu
        .download_f32(&pooled_gpu)
        .map_err(|error| error.to_string())?;
    let pooled_ref = pool_rope_reference(
        &raw_keys.values,
        blocks,
        QSA_COMPRESS,
        index_dim,
        10_000_000.0,
    );
    let pool_err = compare_f32(
        &pooled_actual,
        &pooled_ref,
        bf16_atol,
        bf16_rtol,
        "QSA production pool/RoPE",
    )?;

    let mut selected_actual = Vec::with_capacity(selected_expected.len());
    for token in 0..tokens {
        let visible = token + 1;
        let block_count = visible / QSA_COMPRESS;
        if block_count == 0 {
            selected_actual.extend((0..capacity).map(|slot| {
                if slot < visible {
                    slot as i64
                } else {
                    -1
                }
            }));
            continue;
        }
        let query = gpu
            .upload_f32(index_queries.row(token)?, &[index_heads, index_dim])
            .map_err(|error| error.to_string())?;
        let selected = gpu
            .upload_raw(&vec![0u8; capacity * 4], &[capacity * 4])
            .map_err(|error| error.to_string())?;
        indexed_attention_select(
            gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: &pooled_gpu,
                selected: &selected,
                block_count,
                index_heads,
                index_dim,
                budget_blocks: QSA_BUDGET / QSA_COMPRESS,
                compress: QSA_COMPRESS,
                visible,
                capacity,
            },
        )
        .map_err(|error| error.to_string())?;
        selected_actual.extend(download_i32(gpu, &selected, capacity)?);
        free(gpu, query)?;
        free(gpu, selected)?;
    }
    compare_i64(&selected_actual, selected_expected, "QSA selected indices")?;
    compare_i64(
        &selected_actual,
        required(arrays, "selected_indices_chunked")?.ints()?,
        "QSA chunked selected indices",
    )?;
    compare_i64(
        &selected_actual,
        required(arrays, "selected_indices_incremental")?.ints()?,
        "QSA incremental selected indices",
    )?;
    let mut selected_mask = vec![0.0f32; tokens * capacity * tokens];
    let mut token_mask = vec![0.0f32; tokens * tokens];
    for row in 0..tokens {
        for slot in 0..capacity {
            let index = selected_actual[row * capacity + slot];
            if index >= 0 && (index as usize) < tokens {
                selected_mask[(row * capacity + slot) * tokens + index as usize] = 1.0;
                token_mask[row * tokens + index as usize] = 1.0;
            }
        }
    }
    compare_f32(
        &selected_mask,
        &required(arrays, "selected_mask")?.values,
        0.0,
        0.0,
        "QSA selected mask",
    )?;
    compare_f32(
        &token_mask,
        &required(arrays, "selected_token_mask")?.values,
        0.0,
        0.0,
        "QSA selected token mask",
    )?;
    let mut block_scores = vec![0.0f32; tokens * ((tokens + QSA_COMPRESS - 1) / QSA_COMPRESS)];
    for token in 0..tokens {
        let block_count = (token + 1) / QSA_COMPRESS;
        for block in 0..block_count {
            let mut score = 0.0f32;
            for head in 0..index_heads {
                let qbase = token * index_heads * index_dim + head * index_dim;
                let pbase = block * index_dim;
                let dot = (0..index_dim)
                    .map(|index| index_queries.values[qbase + index] * pooled_actual[pbase + index])
                    .sum::<f32>()
                    / (index_dim as f32).sqrt();
                score += dot.max(0.0);
            }
            block_scores[token * ((tokens + QSA_COMPRESS - 1) / QSA_COMPRESS) + block] = score;
        }
    }
    compare_f32(
        &block_scores,
        &required(arrays, "block_scores")?.values,
        bf16_atol,
        bf16_rtol,
        "QSA block scores",
    )?;

    let queries_raw = required(arrays, "attention_queries")?;
    let keys_raw = required(arrays, "attention_keys")?;
    let values = required(arrays, "attention_values")?;
    let gates = required(arrays, "attention_gate")?;
    let queries = required(arrays, "query_rope")?;
    let keys = required(arrays, "key_rope")?;
    let n_heads = queries.shape[1];
    let n_kv_heads = keys.shape[1];
    let head_dim = queries.shape[2];
    let positions: Vec<usize> = required(arrays, "positions")?
        .ints()?
        .iter()
        .map(|value| *value as usize)
        .collect();
    let query_rope_ref = rope_reference(
        &queries_raw.values,
        tokens,
        n_heads,
        head_dim,
        &positions,
        10_000_000.0,
        false,
    );
    let key_rope_ref = rope_reference(
        &keys_raw.values,
        tokens,
        n_kv_heads,
        head_dim,
        &positions,
        10_000_000.0,
        false,
    );
    compare_f32(
        &query_rope_ref,
        &queries.values,
        atol,
        rtol,
        "QSA oracle query RoPE",
    )?;
    compare_f32(
        &key_rope_ref,
        &keys.values,
        atol,
        rtol,
        "QSA oracle key RoPE",
    )?;
    let zero_norm_q = upload_bf16(gpu, &vec![0.0; head_dim], &[head_dim])?;
    let mut production_query_rope = Vec::with_capacity(queries_raw.values.len());
    for token in 0..tokens {
        let row = gpu
            .upload_f32(queries_raw.row(token)?, &[n_heads * head_dim])
            .map_err(|error| error.to_string())?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &row,
                norm: &zero_norm_q,
                heads: n_heads,
                head_dim,
                head_stride: head_dim,
                position: positions[token],
                rotary_dim: head_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        production_query_rope.extend(gpu.download_f32(&row).map_err(|error| error.to_string())?);
        free(gpu, row)?;
    }
    let production_query_ref = rope_reference(
        &queries_raw.values,
        tokens,
        n_heads,
        head_dim,
        &positions,
        10_000_000.0,
        true,
    );
    compare_f32(
        &production_query_rope,
        &production_query_ref,
        atol,
        rtol,
        "QSA production query norm/RoPE",
    )?;
    let mut production_key_rope = Vec::with_capacity(keys_raw.values.len());
    for token in 0..tokens {
        let row = gpu
            .upload_f32(keys_raw.row(token)?, &[n_kv_heads * head_dim])
            .map_err(|error| error.to_string())?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &row,
                norm: &zero_norm_q,
                heads: n_kv_heads,
                head_dim,
                head_stride: head_dim,
                position: positions[token],
                rotary_dim: head_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        production_key_rope.extend(gpu.download_f32(&row).map_err(|error| error.to_string())?);
        free(gpu, row)?;
    }
    let production_key_ref = rope_reference(
        &keys_raw.values,
        tokens,
        n_kv_heads,
        head_dim,
        &positions,
        10_000_000.0,
        true,
    );
    compare_f32(
        &production_key_rope,
        &production_key_ref,
        atol,
        rtol,
        "QSA production key norm/RoPE",
    )?;
    let rope_input = required(arrays, "rope_input")?;
    let rope_positions = [0usize, 1, 4, 7, 11];
    let mut production_rope = Vec::with_capacity(rope_input.values.len());
    for (row_index, position) in rope_positions.iter().enumerate() {
        let row = gpu
            .upload_f32(
                &rope_input.values[row_index * 16..(row_index + 1) * 16],
                &[16],
            )
            .map_err(|error| error.to_string())?;
        let rope_norm = upload_bf16(gpu, &vec![0.0; 8], &[8])?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &row,
                norm: &rope_norm,
                heads: 2,
                head_dim: 8,
                head_stride: 8,
                position: *position,
                rotary_dim: 8,
            },
        )
        .map_err(|error| error.to_string())?;
        free(gpu, rope_norm)?;
        production_rope.extend(gpu.download_f32(&row).map_err(|error| error.to_string())?);
        free(gpu, row)?;
    }
    let production_rope_ref = rope_reference(
        &rope_input.values,
        5,
        2,
        8,
        &rope_positions,
        10_000_000.0,
        true,
    );
    compare_f32(
        &production_rope,
        &production_rope_ref,
        atol,
        rtol,
        "QSA production RoPE fixture",
    )?;
    let oracle_rope_ref = rope_reference(
        &rope_input.values,
        5,
        2,
        8,
        &rope_positions,
        10_000_000.0,
        false,
    );
    compare_f32(
        &oracle_rope_ref,
        &required(arrays, "rope_output")?.values,
        atol,
        rtol,
        "QSA oracle RoPE fixture",
    )?;
    free(gpu, zero_norm_q)?;

    let full_keys = gpu
        .zeros(&[tokens * n_kv_heads * head_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    let full_values = gpu
        .zeros(&[tokens * n_kv_heads * head_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    for token in 0..tokens {
        let key = gpu
            .upload_f32(keys.row(token)?, &[n_kv_heads * head_dim])
            .map_err(|error| error.to_string())?;
        let value = gpu
            .upload_f32(values.row(token)?, &[n_kv_heads * head_dim])
            .map_err(|error| error.to_string())?;
        indexed_attention_cache_append(
            gpu,
            &IndexedAttentionCacheAppend {
                key: &key,
                value: &value,
                full_keys: &full_keys,
                full_values: &full_values,
                position: token,
                kv_width: n_kv_heads * head_dim,
            },
        )
        .map_err(|error| error.to_string())?;
        free(gpu, key)?;
        free(gpu, value)?;
    }
    let mut actual_heads = Vec::with_capacity(tokens * n_heads * head_dim);
    for token in 0..tokens {
        let mut q_gate = Vec::with_capacity(2 * n_heads * head_dim);
        let query_row = queries.row(token)?;
        let gate_row = gates.row(token)?;
        for head in 0..n_heads {
            let start = head * head_dim;
            q_gate.extend_from_slice(&query_row[start..start + head_dim]);
            q_gate.extend_from_slice(&gate_row[start..start + head_dim]);
        }
        let query = gpu
            .upload_f32(&q_gate, &[2 * n_heads * head_dim])
            .map_err(|error| error.to_string())?;
        let selected = upload_raw_i32(
            gpu,
            &selected_actual[token * capacity..(token + 1) * capacity],
        )?;
        let output = gpu
            .zeros(&[n_heads * head_dim], DType::F32)
            .map_err(|error| error.to_string())?;
        indexed_attention_attention(
            gpu,
            &IndexedAttentionAttention {
                q_with_gate: &query,
                full_keys: &full_keys,
                full_values: &full_values,
                selected: &selected,
                output: &output,
                n_heads,
                n_kv_heads,
                head_dim,
                selected_len: capacity,
                full_capacity: tokens,
            },
        )
        .map_err(|error| error.to_string())?;
        actual_heads.extend(
            gpu.download_f32(&output)
                .map_err(|error| error.to_string())?,
        );
        free(gpu, query)?;
        free(gpu, selected)?;
        free(gpu, output)?;
    }
    let head_err = compare_f32(
        &actual_heads,
        &required(arrays, "attention_head_output")?.values,
        atol,
        rtol,
        "QSA gated attention",
    )?;
    let projected = gpu_ple_linear_rows(
        gpu,
        &actual_heads,
        tokens,
        n_heads * head_dim,
        &required(arrays, "attention_output_projection")?.values,
        required(arrays, "attention_output_projection")?.shape[0],
    )?;
    let output_err = compare_f32(
        &projected,
        &required(arrays, "attention_output")?.values,
        atol,
        rtol,
        "QSA output projection",
    )?;
    let weights = host_qsa_weights(
        &queries.values,
        &keys.values,
        &values.values,
        &selected_actual,
        tokens,
        n_heads,
        n_kv_heads,
        head_dim,
        capacity,
    );
    compare_f32(
        &weights,
        &required(arrays, "attention_weights")?.values,
        atol,
        rtol,
        "QSA attention weights",
    )?;
    free(gpu, raw_gpu)?;
    free(gpu, pooled_gpu)?;
    free(gpu, full_keys)?;
    free(gpu, full_values)?;
    Ok(json!({
        "case":"qsa_pool_selection_mask_tail_rope",
        "status":"pass",
        "layer_index":3,
        "token_boundary":4,
        "pool_boundaries":[4,8,12],
        "checks":["production pool/RoPE","production norm/RoPE","production cache append","exact selection/tail/masks","production gated GQA attention","production output projection","attention weights"],
        "pool_max_abs":pool_err.0,
        "head_max_abs":head_err.0,
        "output_max_abs":output_err.0
    }))
}

fn run_hc(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let input = required(arrays, "hyper_input")?;
    let norm = required(arrays, "hc_norm_weight_zero_centered")?;
    let branches = 4usize;
    let hidden = 8usize;
    let tokens = input.shape[0];
    let wide = branches * hidden;
    shape(input, &[tokens, wide], "hyper_input")?;
    shape(norm, &[branches, hidden], "hc_norm_weight_zero_centered")?;
    let (atol, rtol) = tolerance(manifest, "bf16_input_f32_accumulation")?;
    let (normed, lowrank_silu, mix_gate, mixed) = hc_read_case(
        gpu,
        &input.values,
        tokens,
        branches,
        hidden,
        &norm.values,
        &required(arrays, "input_mix_weight_down")?.values,
        &required(arrays, "input_mix_weight_up")?.values,
    )?;
    let norm_err = compare_f32(
        &normed,
        &required(arrays, "normed")?.values,
        atol,
        rtol,
        "HC prepared norm",
    )?;
    let low_err = compare_f32(
        &lowrank_silu,
        &required(arrays, "lowrank_silu")?.values,
        atol,
        rtol,
        "HC low-rank SiLU",
    )?;
    let gate_err = compare_f32(
        &mix_gate,
        &required(arrays, "mix_gate")?.values,
        atol,
        rtol,
        "HC mix gate",
    )?;
    let mix_err = compare_f32(
        &mixed,
        &required(arrays, "mixed")?.values,
        atol,
        rtol,
        "HC mixed value",
    )?;
    let (injection_weight, injected) = hc_inject_case(
        gpu,
        &input.values,
        &normed,
        &mixed,
        tokens,
        branches,
        hidden,
        &required(arrays, "block_inject_weight")?.values,
    )?;
    let inject_weight_err = compare_f32(
        &injection_weight,
        &required(arrays, "injection_weight")?.values,
        atol,
        rtol,
        "HC injection gate",
    )?;
    let inject_err = compare_f32(
        &injected,
        &required(arrays, "injected")?.values,
        atol,
        rtol,
        "HC injected output",
    )?;
    let (final_normed, final_low, final_gate, final_mixed) = hc_read_case(
        gpu,
        &input.values,
        tokens,
        branches,
        hidden,
        &norm.values,
        &required(arrays, "final_input_mix_weight_down")?.values,
        &required(arrays, "final_input_mix_weight_up")?.values,
    )?;
    let final_norm_err = compare_f32(
        &final_normed,
        &required(arrays, "final_normed")?.values,
        atol,
        rtol,
        "HC final norm",
    )?;
    let final_low_err = compare_f32(
        &final_low,
        &required(arrays, "final_lowrank_silu")?.values,
        atol,
        rtol,
        "HC final low-rank SiLU",
    )?;
    let final_gate_err = compare_f32(
        &final_gate,
        &required(arrays, "final_mix_gate")?.values,
        atol,
        rtol,
        "HC final gate",
    )?;
    let final_mix_err = compare_f32(
        &final_mixed,
        &required(arrays, "final_mixed")?.values,
        atol,
        rtol,
        "HC final mixed value",
    )?;
    Ok(json!({
        "case":"hc_prepare_inject_final_mix",
        "status":"pass",
        "checks":["production HC norm/read projections","production mix gates","production HC inject/write","production final read/mix"],
        "norm_max_abs":norm_err.0,
        "lowrank_max_abs":low_err.0,
        "gate_max_abs":gate_err.0,
        "mix_max_abs":mix_err.0,
        "injection_weight_max_abs":inject_weight_err.0,
        "injected_max_abs":inject_err.0,
        "final_norm_max_abs":final_norm_err.0,
        "final_lowrank_max_abs":final_low_err.0,
        "final_gate_max_abs":final_gate_err.0,
        "final_mix_max_abs":final_mix_err.0
    }))
}

struct MoeActual {
    router_logits: Vec<f32>,
    selected_experts: Vec<i64>,
    routing_weights: Vec<f32>,
    routed_output: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_output: Vec<f32>,
    output: Vec<f32>,
}

fn run_moe_operator(
    gpu: &mut Gpu,
    hidden: &[f32],
    tokens: usize,
    hidden_size: usize,
    router_weight: &[f32],
    gate_up_weight: &[f32],
    down_weight: &[f32],
    shared_gate_weight: &[f32],
    shared_gate_up_weight: &[f32],
    shared_down_weight: &[f32],
) -> Result<MoeActual, String> {
    let experts = 512usize;
    let top_k = 10usize;
    let intermediate = gate_up_weight.len() / (experts * 2 * hidden_size);
    if hidden.len() != tokens * hidden_size
        || router_weight.len() != experts * hidden_size
        || gate_up_weight.len() != experts * 2 * intermediate * hidden_size
        || down_weight.len() != experts * hidden_size * intermediate
        || shared_gate_weight.len() != hidden_size
        || shared_gate_up_weight.len() != 2 * intermediate * hidden_size
        || shared_down_weight.len() != hidden_size * intermediate
    {
        return fail("MoE compact operator input shape mismatch");
    }
    let router_logits = gpu_linear_rows(gpu, hidden, tokens, hidden_size, router_weight, experts)?;
    let logits_gpu = gpu
        .upload_f32(&router_logits, &[tokens, experts])
        .map_err(|error| error.to_string())?;
    let selected_gpu = gpu
        .upload_raw(&vec![0u8; tokens * top_k * 4], &[tokens * top_k * 4])
        .map_err(|error| error.to_string())?;
    let weights_gpu = gpu
        .zeros(&[tokens * top_k], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.moe_router_softmax_top10_f32(&logits_gpu, &selected_gpu, &weights_gpu, tokens, true)
        .map_err(|error| error.to_string())?;
    let selected_experts = download_i32(gpu, &selected_gpu, tokens * top_k)?;
    let routing_weights = gpu
        .download_f32(&weights_gpu)
        .map_err(|error| error.to_string())?;
    let mut expert_outputs = Vec::with_capacity(tokens * top_k * hidden_size);
    for token in 0..tokens {
        for rank in 0..top_k {
            let expert = selected_experts[token * top_k + rank] as usize;
            let gate_up = &gate_up_weight[expert * 2 * intermediate * hidden_size
                ..(expert + 1) * 2 * intermediate * hidden_size];
            let activation = gpu_linear_rows(
                gpu,
                &hidden[token * hidden_size..(token + 1) * hidden_size],
                1,
                hidden_size,
                gate_up,
                2 * intermediate,
            )?;
            let gate = gpu_silu_values(gpu, &activation[..intermediate])?;
            let product = gpu_mul_values(gpu, &gate, &activation[intermediate..])?;
            let down = &down_weight
                [expert * hidden_size * intermediate..(expert + 1) * hidden_size * intermediate];
            let output = gpu_linear_rows(gpu, &product, 1, intermediate, down, hidden_size)?;
            expert_outputs.extend(output);
        }
    }
    let expert_gpu = gpu
        .upload_f32(&expert_outputs, &[tokens * top_k, hidden_size])
        .map_err(|error| error.to_string())?;
    let residual_gpu = gpu
        .zeros(&[tokens, hidden_size], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.moe_down_combine_top10_batched(
        &expert_gpu,
        &selected_gpu,
        &weights_gpu,
        &residual_gpu,
        hidden_size,
        tokens,
    )
    .map_err(|error| error.to_string())?;
    let routed_output = gpu
        .download_f32(&residual_gpu)
        .map_err(|error| error.to_string())?;
    let shared_gate_logits =
        gpu_linear_rows(gpu, hidden, tokens, hidden_size, shared_gate_weight, 1)?;
    let shared_gate = gpu_sigmoid_values(gpu, &shared_gate_logits)?;
    let shared_activation_all = gpu_linear_rows(
        gpu,
        hidden,
        tokens,
        hidden_size,
        shared_gate_up_weight,
        2 * intermediate,
    )?;
    let mut shared_output = Vec::with_capacity(tokens * hidden_size);
    for token in 0..tokens {
        let activation =
            &shared_activation_all[token * 2 * intermediate..(token + 1) * 2 * intermediate];
        let gate = gpu_silu_values(gpu, &activation[..intermediate])?;
        let product = gpu_mul_values(gpu, &gate, &activation[intermediate..])?;
        let output = gpu_linear_rows(
            gpu,
            &product,
            1,
            intermediate,
            shared_down_weight,
            hidden_size,
        )?;
        shared_output.extend(output);
    }
    let mut shared_scale = Vec::with_capacity(shared_output.len());
    for token in 0..tokens {
        shared_scale.extend(std::iter::repeat(shared_gate[token]).take(hidden_size));
    }
    let shared_output = gpu_mul_values(gpu, &shared_output, &shared_scale)?;
    let output = gpu_add_values(gpu, &routed_output, &shared_output)?;
    free(gpu, logits_gpu)?;
    free(gpu, selected_gpu)?;
    free(gpu, weights_gpu)?;
    free(gpu, expert_gpu)?;
    free(gpu, residual_gpu)?;
    Ok(MoeActual {
        router_logits,
        selected_experts,
        routing_weights,
        routed_output,
        shared_gate,
        shared_output,
        output,
    })
}

fn run_moe(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let hidden = required(arrays, "hidden")?;
    let router_weight = required(arrays, "router_weight")?;
    let gate_up_weight = required(arrays, "gate_up_weight")?;
    let down_weight = required(arrays, "down_weight")?;
    let shared_gate_weight = required(arrays, "shared_gate_weight")?;
    let shared_gate_up_weight = required(arrays, "shared_gate_up_weight")?;
    let shared_down_weight = required(arrays, "shared_down_weight")?;
    let tokens = hidden.shape[0];
    let hidden_size = hidden.shape[1];
    let (atol, rtol) = tolerance(manifest, "f32_accumulation")?;
    let actual = run_moe_operator(
        gpu,
        &hidden.values,
        tokens,
        hidden_size,
        &router_weight.values,
        &gate_up_weight.values,
        &down_weight.values,
        &shared_gate_weight.values,
        &shared_gate_up_weight.values,
        &shared_down_weight.values,
    )?;
    let router_err = compare_f32(
        &actual.router_logits,
        &required(arrays, "router_logits")?.values,
        atol,
        rtol,
        "MoE router logits",
    )?;
    let expected_experts = required(arrays, "selected_experts")?;
    let route_err = compare_routes(
        &actual.selected_experts,
        &actual.routing_weights,
        expected_experts.ints()?,
        &required(arrays, "routing_weights")?.values,
        *expected_experts
            .shape
            .last()
            .ok_or("selected_experts has no shape")?,
        atol,
        rtol,
        "MoE",
    )?;
    let routed_err = compare_f32(
        &actual.routed_output,
        &required(arrays, "routed_output")?.values,
        atol,
        rtol,
        "MoE routed output",
    )?;
    let gate_err = compare_f32(
        &actual.shared_gate,
        &required(arrays, "shared_gate")?.values,
        atol,
        rtol,
        "MoE shared gate",
    )?;
    let shared_err = compare_f32(
        &actual.shared_output,
        &required(arrays, "shared_output")?.values,
        atol,
        rtol,
        "MoE shared output",
    )?;
    let output_err = compare_f32(
        &actual.output,
        &required(arrays, "output")?.values,
        atol,
        rtol,
        "MoE final output",
    )?;
    Ok(json!({
        "case":"moe_top10_normalized_shared",
        "status":"pass",
        "checks":["production 512-way router","production top-10 probability normalization","production routed gate/up/down","single weighted routed combine","production shared gate/up/down","single shared add"],
        "router_max_abs":router_err.0,
        "routing_max_abs":route_err.0,
        "routed_max_abs":routed_err.0,
        "shared_gate_max_abs":gate_err.0,
        "shared_max_abs":shared_err.0,
        "output_max_abs":output_err.0
    }))
}
struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = 19650218;
        for index in 1..624 {
            state[index] = 1812433253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        let key = if seed <= u32::MAX as u64 {
            vec![seed as u32]
        } else {
            vec![seed as u32, (seed >> 32) as u32]
        };
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = 624usize.max(key.len());
        while k > 0 {
            state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = 623;
        while k > 0 {
            state[i] = (state[i]
                ^ ((state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1566083941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            k -= 1;
        }
        state[0] = 0x8000_0000;
        Self { state, index: 624 }
    }

    fn twist(&mut self) {
        for index in 0..624 {
            let value =
                (self.state[index] & 0x8000_0000) | (self.state[(index + 1) % 624] & 0x7fff_ffff);
            self.state[index] = self.state[(index + 397) % 624]
                ^ (value >> 1)
                ^ if value & 1 != 0 { 0x9908_b0df } else { 0 };
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn random(&mut self) -> f64 {
        let first = (self.next_u32() >> 5) as u64;
        let second = (self.next_u32() >> 6) as u64;
        ((first << 26) + second) as f64 / 9_007_199_254_740_992.0
    }

    fn normal(&mut self, count: usize, scale: f32) -> Vec<f32> {
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let u1 = self.random().max(2f64.powi(-53));
            let u2 = self.random();
            let radius = (-2.0 * u1.ln()).sqrt();
            values.push((radius * (2.0 * std::f64::consts::PI * u2).cos() * scale as f64) as f32);
            if values.len() < count {
                values
                    .push((radius * (2.0 * std::f64::consts::PI * u2).sin() * scale as f64) as f32);
            }
        }
        values
    }

    fn uniform(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| ((self.random() * 2.0 - 1.0) * scale as f64) as f32)
            .collect()
    }
}

fn mtp_moe_weights(
    seed: u64,
    hidden: usize,
    intermediate: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let experts = 512usize;
    let mut rng = PyRandom::new(seed);
    (
        rng.normal(experts * hidden, 0.07),
        rng.normal(experts * 2 * intermediate * hidden, 0.045),
        rng.normal(experts * hidden * intermediate, 0.045),
        rng.uniform(hidden, 0.08),
        rng.normal(2 * intermediate * hidden, 0.045),
        rng.normal(hidden * intermediate, 0.045),
    )
}

fn run_mtp(
    gpu: &mut Gpu,
    arrays: &BTreeMap<String, Array>,
    manifest: &Value,
) -> Result<Value, String> {
    let token_ids = required(arrays, "token_ids")?.ints()?;
    let tokens = token_ids.len();
    let hidden_size = required(arrays, "embedding")?.shape[1];
    let branches = 4usize;
    let wide = branches * hidden_size;
    let intermediate = 8usize;
    let (atol, rtol) = tolerance(manifest, "bf16_input_f32_accumulation")?;
    let source_native_residual_err = compare_f32(
        &required(arrays, "native_residual_input")?.values,
        &required(arrays, "source_native_residual_input")?.values,
        0.0,
        0.0,
        "MTP source/native residual capture",
    )?;
    let source_native_sample_err = compare_f32(
        &required(arrays, "native_sample_hidden")?.values,
        &required(arrays, "source_native_sample_hidden")?.values,
        0.0,
        0.0,
        "MTP source/native sample capture",
    )?;
    let source_native_multi_err = compare_f32(
        &required(arrays, "native_multi_hidden")?.values,
        &required(arrays, "source_native_multi_hidden")?.values,
        0.0,
        0.0,
        "MTP source/native multi capture",
    )?;
    let source_native_hc_err = compare_f32(
        &required(arrays, "native_hc_mixed")?.values,
        &required(arrays, "source_native_hc_mixed")?.values,
        0.0,
        0.0,
        "MTP source/native HC capture",
    )?;
    let source_native_injection_err = compare_f32(
        &required(arrays, "native_hc_injection_weight")?.values,
        &required(arrays, "source_native_hc_injection_weight")?.values,
        0.0,
        0.0,
        "MTP source/native HC injection capture",
    )?;
    let source_native_mask_err = compare_f32(
        &required(arrays, "native_qsa_selected_token_mask")?.values,
        &required(arrays, "source_native_qsa_selected_token_mask")?.values,
        0.0,
        0.0,
        "MTP source/native QSA mask capture",
    )?;
    let source_native_moe_err = compare_f32(
        &required(arrays, "native_moe_output")?.values,
        &required(arrays, "source_native_moe_output")?.values,
        0.0,
        0.0,
        "MTP source/native MoE capture",
    )?;
    compare_i64(
        required(arrays, "native_mtp_step0_indices")?.ints()?,
        required(arrays, "source_native_mtp_step0_indices")?.ints()?,
        "MTP source/native step0 indices",
    )?;
    compare_i64(
        required(arrays, "native_mtp_later_reused_indices")?.ints()?,
        required(arrays, "source_native_mtp_later_reused_indices")?.ints()?,
        "MTP source/native reused indices",
    )?;
    let source_equation_residual_err = measure_f32(
        &required(arrays, "native_residual_input")?.values,
        &required(arrays, "source_equation_residual_input")?.values,
        atol,
        rtol,
        "MTP native source-equation residual",
    )?;
    let ids_gpu = upload_raw_i32(gpu, token_ids)?;
    let table_words: Vec<u16> = required(arrays, "token_embedding_bf16")?
        .ints()?
        .iter()
        .map(|value| *value as u16)
        .collect();
    let table_shape = required(arrays, "token_embedding_bf16")?.shape.clone();
    let table_gpu = upload_bf16_words(gpu, &table_words, &table_shape)?;
    let embedding_gpu = gpu
        .zeros(&[tokens, hidden_size], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.embedding_lookup_bf16_batched(&table_gpu, &embedding_gpu, &ids_gpu, tokens, hidden_size)
        .map_err(|error| error.to_string())?;
    let embedding = gpu
        .download_f32(&embedding_gpu)
        .map_err(|error| error.to_string())?;
    let embedding_err = compare_f32(
        &embedding,
        &required(arrays, "embedding")?.values,
        atol,
        rtol,
        "MTP embedding",
    )?;
    let embedding_norm = gpu_rms_rows(gpu, &embedding, tokens, hidden_size)?;
    let embedding_norm_err = compare_f32(
        &embedding_norm,
        &required(arrays, "embedding_norm")?.values,
        atol,
        rtol,
        "MTP embedding norm",
    )?;
    let projected_embedding = gpu_linear_rows(
        gpu,
        &embedding_norm,
        tokens,
        hidden_size,
        &required(arrays, "fc_embedding")?.values,
        hidden_size,
    )?;
    let projected_embedding_err = compare_f32(
        &projected_embedding,
        &required(arrays, "projected_embedding")?.values,
        atol,
        rtol,
        "MTP projected embedding",
    )?;
    let backbone = required(arrays, "backbone_hidden")?;
    let hidden_norm = gpu_rms_rows(gpu, &backbone.values, tokens, wide)?;
    let hidden_norm_err = compare_f32(
        &hidden_norm,
        &required(arrays, "hidden_norm")?.values,
        atol,
        rtol,
        "MTP hidden norm",
    )?;
    let projected_hidden = gpu_linear_rows(
        gpu,
        &hidden_norm,
        tokens * branches,
        hidden_size,
        &required(arrays, "fc_hidden")?.values,
        hidden_size,
    )?;
    let projected_hidden_err = compare_f32(
        &projected_hidden,
        &required(arrays, "projected_hidden")?.values,
        atol,
        rtol,
        "MTP projected hidden",
    )?;
    let mut projected_embedding_broadcast = Vec::with_capacity(tokens * wide);
    for token in 0..tokens {
        for _ in 0..branches {
            projected_embedding_broadcast.extend_from_slice(
                &projected_embedding[token * hidden_size..(token + 1) * hidden_size],
            );
        }
    }
    let residual_input = gpu_add_values(gpu, &projected_embedding_broadcast, &projected_hidden)?;
    let residual_err = compare_f32(
        &residual_input,
        &required(arrays, "residual_input")?.values,
        atol,
        rtol,
        "MTP residual input",
    )?;

    let index_queries = required(arrays, "index_queries")?;
    let raw_keys = required(arrays, "raw_index_keys")?;
    let index_heads = index_queries.shape[1];
    let index_dim = index_queries.shape[2];
    let capacity = required(arrays, "step0_selected_indices")?.ints()?.len() / tokens;
    let raw_gpu = gpu
        .upload_f32(&raw_keys.values, &[tokens, index_dim])
        .map_err(|error| error.to_string())?;
    let pooled_gpu = gpu
        .zeros(&[1, index_dim], DType::F32)
        .map_err(|error| error.to_string())?;
    indexed_attention_pool_rope(
        gpu,
        &IndexedAttentionPoolRope {
            raw_keys: &raw_gpu,
            pooled: &pooled_gpu,
            norm: None,
            block_count: 1,
            compress: QSA_COMPRESS,
            index_dim,
            position: None,
            grid_bound: 1,
        },
    )
    .map_err(|error| error.to_string())?;
    let mut selected = Vec::with_capacity(tokens * capacity);
    for token in 0..tokens {
        let visible = token + 1;
        if visible < QSA_COMPRESS {
            selected
                .extend((0..capacity).map(|slot| if slot < visible { slot as i64 } else { -1 }));
            continue;
        }
        let query = gpu
            .upload_f32(index_queries.row(token)?, &[index_heads, index_dim])
            .map_err(|error| error.to_string())?;
        let selected_gpu = gpu
            .upload_raw(&vec![0u8; capacity * 4], &[capacity * 4])
            .map_err(|error| error.to_string())?;
        indexed_attention_select(
            gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: &pooled_gpu,
                selected: &selected_gpu,
                block_count: 1,
                index_heads,
                index_dim,
                budget_blocks: QSA_BUDGET / QSA_COMPRESS,
                compress: QSA_COMPRESS,
                visible,
                capacity,
            },
        )
        .map_err(|error| error.to_string())?;
        selected.extend(download_i32(gpu, &selected_gpu, capacity)?);
        free(gpu, query)?;
        free(gpu, selected_gpu)?;
    }
    compare_i64(
        &selected,
        required(arrays, "step0_selected_indices")?.ints()?,
        "MTP step0 QSA selection",
    )?;
    compare_i64(
        &selected,
        required(arrays, "later_reused_indices")?.ints()?,
        "MTP reused QSA selection",
    )?;
    let mut token_mask = vec![0.0f32; tokens * tokens];
    for token in 0..tokens {
        for index in &selected[token * capacity..(token + 1) * capacity] {
            if *index >= 0 && (*index as usize) < tokens {
                token_mask[token * tokens + *index as usize] = 1.0;
            }
        }
    }
    compare_f32(
        &token_mask,
        &required(arrays, "step0_selected_token_mask")?.values,
        0.0,
        0.0,
        "MTP selected token mask",
    )?;
    let positions: Vec<usize> = (0..tokens).collect();
    let query_rope = rope_reference(
        &required(arrays, "attention_queries")?.values,
        tokens,
        4,
        4,
        &positions,
        10_000_000.0,
        false,
    );
    let key_rope = rope_reference(
        &required(arrays, "attention_keys")?.values,
        tokens,
        2,
        4,
        &positions,
        10_000_000.0,
        false,
    );
    let full_keys = gpu
        .zeros(&[tokens * 2 * 4], DType::F32)
        .map_err(|error| error.to_string())?;
    let full_values = gpu
        .zeros(&[tokens * 2 * 4], DType::F32)
        .map_err(|error| error.to_string())?;
    for token in 0..tokens {
        let key = gpu
            .upload_f32(&key_rope[token * 8..(token + 1) * 8], &[8])
            .map_err(|error| error.to_string())?;
        let value = gpu
            .upload_f32(
                &required(arrays, "attention_values")?.values[token * 8..(token + 1) * 8],
                &[8],
            )
            .map_err(|error| error.to_string())?;
        indexed_attention_cache_append(
            gpu,
            &IndexedAttentionCacheAppend {
                key: &key,
                value: &value,
                full_keys: &full_keys,
                full_values: &full_values,
                position: token,
                kv_width: 8,
            },
        )
        .map_err(|error| error.to_string())?;
        free(gpu, key)?;
        free(gpu, value)?;
    }
    let mut attention_output = Vec::with_capacity(tokens * hidden_size);
    let attention_gate = required(arrays, "attention_gate")?;
    for token in 0..tokens {
        let mut q_gate = Vec::with_capacity(32);
        let query_row = &query_rope[token * 16..(token + 1) * 16];
        let gate_row = attention_gate.row(token)?;
        for head in 0..4 {
            let start = head * 4;
            q_gate.extend_from_slice(&query_row[start..start + 4]);
            q_gate.extend_from_slice(&gate_row[start..start + 4]);
        }
        let q_gate_gpu = gpu
            .upload_f32(&q_gate, &[32])
            .map_err(|error| error.to_string())?;
        let selected_gpu =
            upload_raw_i32(gpu, &selected[token * capacity..(token + 1) * capacity])?;
        let heads = gpu
            .zeros(&[16], DType::F32)
            .map_err(|error| error.to_string())?;
        indexed_attention_attention(
            gpu,
            &IndexedAttentionAttention {
                q_with_gate: &q_gate_gpu,
                full_keys: &full_keys,
                full_values: &full_values,
                selected: &selected_gpu,
                output: &heads,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 4,
                selected_len: capacity,
                full_capacity: tokens,
            },
        )
        .map_err(|error| error.to_string())?;
        let heads_values = gpu
            .download_f32(&heads)
            .map_err(|error| error.to_string())?;
        let projected = gpu_ple_linear_rows(
            gpu,
            &heads_values,
            1,
            16,
            &required(arrays, "attention_projection")?.values,
            hidden_size,
        )?;
        attention_output.extend(projected);
        free(gpu, q_gate_gpu)?;
        free(gpu, selected_gpu)?;
        free(gpu, heads)?;
    }
    let attention_err = compare_f32(
        &attention_output,
        &required(arrays, "attention_output")?.values,
        atol,
        rtol,
        "MTP QSA output",
    )?;
    let (attn_normed, _attn_low, _attn_gate, attn_mixed) = hc_read_case(
        gpu,
        &residual_input,
        tokens,
        branches,
        hidden_size,
        &required(arrays, "hc_norm_weight")?.values,
        &required(arrays, "hc_down")?.values,
        &required(arrays, "hc_up")?.values,
    )?;
    let attn_norm_err = compare_f32(
        &attn_normed,
        &required(arrays, "attn_hc_normed")?.values,
        atol,
        rtol,
        "MTP attention HC norm",
    )?;
    let attn_mix_err = compare_f32(
        &attn_mixed,
        &required(arrays, "attn_hc_mixed")?.values,
        atol,
        rtol,
        "MTP attention HC mixed",
    )?;
    let (_attn_weights, attn_injected) = hc_inject_case(
        gpu,
        &residual_input,
        &attn_normed,
        &attention_output,
        tokens,
        branches,
        hidden_size,
        &required(arrays, "hc_inject_weight")?.values,
    )?;
    let attn_inject_err = compare_f32(
        &attn_injected,
        &required(arrays, "attn_hc_injected")?.values,
        atol,
        rtol,
        "MTP attention HC injected",
    )?;

    let (mlp_normed, _mlp_low, _mlp_gate, mlp_mixed) = hc_read_case(
        gpu,
        &attn_injected,
        tokens,
        branches,
        hidden_size,
        &required(arrays, "hc_norm_weight")?.values,
        &required(arrays, "hc_down")?.values,
        &required(arrays, "hc_up")?.values,
    )?;
    let mlp_norm_err = compare_f32(
        &mlp_normed,
        &required(arrays, "mlp_hc_normed")?.values,
        atol,
        rtol,
        "MTP MLP HC norm",
    )?;
    let mlp_mix_err = compare_f32(
        &mlp_mixed,
        &required(arrays, "mlp_hc_mixed")?.values,
        atol,
        rtol,
        "MTP MLP HC mixed",
    )?;
    let mtp_hidden_err = compare_f32(
        &mlp_mixed,
        &required(arrays, "mtp_moe_hidden")?.values,
        atol,
        rtol,
        "MTP MoE hidden",
    )?;
    let root_seed = manifest
        .pointer("/generator/seed")
        .and_then(Value::as_u64)
        .ok_or("manifest generator seed missing")?;
    let (router, gate_up, down, shared_gate, shared_up, shared_down) =
        mtp_moe_weights(root_seed + 89, hidden_size, intermediate);
    // The chained HC input above is only BF16-close to the oracle's, which can
    // flip a top-10 choice between near-equal logits. Route the oracle's own
    // MoE hidden so the discrete expert selection is checked on equal inputs.
    let moe = run_moe_operator(
        gpu,
        &required(arrays, "mtp_moe_hidden")?.values,
        tokens,
        hidden_size,
        &router,
        &gate_up,
        &down,
        &shared_gate,
        &shared_up,
        &shared_down,
    )?;
    let moe_router_err = compare_f32(
        &moe.router_logits,
        &required(arrays, "mtp_moe_router_logits")?.values,
        atol,
        rtol,
        "MTP MoE router logits",
    )?;
    let expected_experts = required(arrays, "mtp_moe_selected_experts")?;
    let moe_route_err = compare_routes(
        &moe.selected_experts,
        &moe.routing_weights,
        expected_experts.ints()?,
        &required(arrays, "mtp_moe_routing_weights")?.values,
        *expected_experts
            .shape
            .last()
            .ok_or("mtp_moe_selected_experts has no shape")?,
        atol,
        rtol,
        "MTP MoE",
    )?;
    let (_mlp_weights, mlp_injected) = hc_inject_case(
        gpu,
        &attn_injected,
        &mlp_normed,
        &moe.output,
        tokens,
        branches,
        hidden_size,
        &required(arrays, "hc_inject_weight")?.values,
    )?;
    let mlp_inject_err = compare_f32(
        &mlp_injected,
        &required(arrays, "mlp_hc_injected")?.values,
        atol,
        rtol,
        "MTP MLP HC injected",
    )?;
    let (final_normed, final_low, final_gate, sample_hidden) = hc_read_case(
        gpu,
        &mlp_injected,
        tokens,
        branches,
        hidden_size,
        &required(arrays, "hc_norm_weight")?.values,
        &required(arrays, "final_hc_down")?.values,
        &required(arrays, "final_hc_up")?.values,
    )?;
    let final_norm_err = compare_f32(
        &final_normed,
        &required(arrays, "final_hc_normed")?.values,
        atol,
        rtol,
        "MTP final HC norm",
    )?;
    let final_low_err = compare_f32(
        &final_low,
        &required(arrays, "final_hc_lowrank_silu")?.values,
        atol,
        rtol,
        "MTP final HC low-rank",
    )?;
    let final_gate_err = compare_f32(
        &final_gate,
        &required(arrays, "final_hc_mix_gate")?.values,
        atol,
        rtol,
        "MTP final HC gate",
    )?;
    let sample_err = compare_f32(
        &sample_hidden,
        &required(arrays, "sample_hidden")?.values,
        atol,
        rtol,
        "MTP sample hidden",
    )?;
    let logits = gpu_ple_linear_rows(
        gpu,
        &sample_hidden,
        tokens,
        hidden_size,
        &required(arrays, "lm_head")?.values,
        required(arrays, "lm_head")?.shape[0],
    )?;
    let logits_err = compare_f32(
        &logits,
        &required(arrays, "logits")?.values,
        atol,
        rtol,
        "MTP LM logits",
    )?;
    free(gpu, ids_gpu)?;
    free(gpu, table_gpu)?;
    free(gpu, embedding_gpu)?;
    free(gpu, raw_gpu)?;
    free(gpu, pooled_gpu)?;
    free(gpu, full_keys)?;
    free(gpu, full_values)?;
    Ok(json!({
        "case":"native_mtp_embedding_qsa_moe_hc",
        "status":"pass",
        "checks":["production BF16 embedding lookup","production embedding/backbone RMS and FC projections","production MTP QSA pool/select/reuse/cache/attention/output","production attention HC read/write","production MTP 512-way/top10 MoE","production MLP HC read/write","production final HC and LM head"],
        "source_native_alias_max_abs":source_native_residual_err.0.max(source_native_sample_err.0).max(source_native_multi_err.0).max(source_native_hc_err.0).max(source_native_injection_err.0).max(source_native_mask_err.0).max(source_native_moe_err.0),
        "source_native_aliases_exact":true,
        "source_equation_residual_max_abs":source_equation_residual_err.0,
        "source_equation_residual_within_tolerance":source_equation_residual_err.2,
        "embedding_max_abs":embedding_err.0,
        "embedding_norm_max_abs":embedding_norm_err.0,
        "projected_embedding_max_abs":projected_embedding_err.0,
        "hidden_norm_max_abs":hidden_norm_err.0,
        "projected_hidden_max_abs":projected_hidden_err.0,
        "residual_max_abs":residual_err.0,
        "attention_max_abs":attention_err.0,
        "attn_hc_norm_max_abs":attn_norm_err.0,
        "attn_hc_mix_max_abs":attn_mix_err.0,
        "attn_hc_injected_max_abs":attn_inject_err.0,
        "mtp_hidden_max_abs":mtp_hidden_err.0,
        "moe_router_max_abs":moe_router_err.0,
        "moe_route_max_abs":moe_route_err.0,
        "mlp_hc_norm_max_abs":mlp_norm_err.0,
        "mlp_hc_mix_max_abs":mlp_mix_err.0,
        "mlp_hc_injected_max_abs":mlp_inject_err.0,
        "final_norm_max_abs":final_norm_err.0,
        "final_low_max_abs":final_low_err.0,
        "final_gate_max_abs":final_gate_err.0,
        "sample_max_abs":sample_err.0,
        "logits_max_abs":logits_err.0
    }))
}
const QUALITY_SCHEMA: &str = "hipfire.qwen4.quality.v1";
const CANONICAL_TOKEN_COUNT: usize = 17;
const CANONICAL_TOKEN_SHA256: &str =
    "e53de8c7b501eaaea637648feb6f569dd17cd564c2f669b2924ccdf1b7e52e2f";

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_path(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn read_quality_tokens(path: &Path) -> Result<(Vec<u32>, Value), String> {
    let metadata_path = if path.extension().and_then(|value| value.to_str()) == Some("json") {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    };
    let metadata_text = fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read corpus metadata {}: {error}", metadata_path.display()))?;
    let metadata: Value = serde_json::from_str(&metadata_text)
        .map_err(|error| format!("parse corpus metadata {}: {error}", metadata_path.display()))?;
    if metadata.get("schema").and_then(Value::as_str)
        != Some("hipfire.qwen4.teacher_forced_corpus.v1")
        || metadata.get("format").and_then(Value::as_str) != Some("u32le")
        || metadata.get("count").and_then(Value::as_u64) != Some(CANONICAL_TOKEN_COUNT as u64)
        || metadata.get("sha256").and_then(Value::as_str) != Some(CANONICAL_TOKEN_SHA256)
    {
        return fail("teacher corpus metadata is not the canonical 17-token u32le manifest");
    }
    let relative = metadata
        .get("path")
        .and_then(Value::as_str)
        .ok_or("teacher corpus metadata has no path")?;
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        return fail("teacher corpus metadata path must stay beside its manifest");
    }
    let payload = metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative_path);
    let bytes = fs::read(&payload)
        .map_err(|error| format!("read canonical corpus {}: {error}", payload.display()))?;
    if bytes.len() != CANONICAL_TOKEN_COUNT * 4 || sha256_hex(&bytes) != CANONICAL_TOKEN_SHA256 {
        return fail("canonical teacher corpus byte count or SHA256 does not match metadata");
    }
    let mut tokens = Vec::with_capacity(CANONICAL_TOKEN_COUNT);
    for chunk in bytes.chunks_exact(4) {
        tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((
        tokens,
        json!({
            "metadata_path": metadata_path,
            "payload_path": payload,
            "count": CANONICAL_TOKEN_COUNT,
            "sha256": CANONICAL_TOKEN_SHA256,
        }),
    ))
}

fn artifact_identity_digest(receipt: &Qwen4HfqmArtifact) -> Result<String, String> {
    let identity = receipt.source_identity.as_ref();
    let files = identity
        .files
        .iter()
        .map(|file| {
            json!({
                "path": file.canonical_path,
                "dev": file.dev,
                "ino": file.ino,
                "len": file.len,
                "mtime_secs": file.mtime_secs,
                "mtime_nanos": file.mtime_nanos,
            })
        })
        .collect::<Vec<_>>();
    let manifest = identity
        .manifest
        .iter()
        .map(|range| {
            json!({
                "name": range.name,
                "file_index": range.file_index,
                "offset": range.offset,
                "length": range.length,
                "dtype": range.dtype,
                "logical_shape": range.logical_shape,
            })
        })
        .collect::<Vec<_>>();
    let value = json!({
        "canonical_path": identity.canonical_path,
        "format": format!("{:?}", identity.format),
        "files": files,
        "metadata_json": identity.metadata_json,
        "manifest": manifest,
    });
    let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn quality_state_digest(
    bundle: &hipfire_arch_qwen4::bundle::Qwen4Bundle,
) -> Result<String, String> {
    let state = &bundle.state;
    let qsa = state
        .qsa
        .first()
        .map(|qsa| {
            json!({
                "full_len": qsa.full_len,
                "raw_len": qsa.raw_len,
                "pooled_len": qsa.pooled_len,
                "partial_len": qsa.partial_len,
                "selected_len": qsa.selected_len,
            })
        })
        .unwrap_or(Value::Null);
    let value = json!({
        "position": state.position,
        "qsa": qsa,
        "ple_history": format!("{:?}", state.ple_history),
    });
    let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn run_quality_candidate(
    model_path: &Path,
    corpus_path: &Path,
    output_path: &Path,
) -> Result<(), String> {
    let (tokens, corpus) = read_quality_tokens(corpus_path)?;
    let mut hfq = HfqFile::open(model_path)
        .map_err(|error| format!("open HFQM {}: {error}", model_path.display()))?;
    let receipt = admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let artifact_sha256 = sha256_path(model_path)?;
    let identity_sha256 = artifact_identity_digest(&receipt)?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    // Load by byte range like the serve loader and state runner. The borrowed
    // mmap path never finished on gfx1151 UMA (over an hour, stuck in SVM
    // registration); range loading takes about a minute.
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|error| error.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let placements = placements;
    let mut bundle = hipfire_arch_qwen4::bundle::Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        2048,
        metadata,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let mut nlls = Vec::with_capacity(tokens.len().saturating_sub(1));
    bundle
        .attach_forward(&mut gpu, 2048)
        .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
    let vocab = config.vocab_size;
    let logits = gpu
        .zeros(&[vocab], DType::F32)
        .map_err(|error| error.to_string())?;
    let mut rows = Vec::with_capacity(tokens.len());
    let mut state_digests = Vec::with_capacity(tokens.len());
    for (position, &token) in tokens.iter().enumerate() {
        bundle
            .forward_token(&mut gpu, token, &logits, None)
            .map_err(|error| format!("forward token {position} ({token}) failed: {error}"))?;
        let values = gpu
            .download_f32(&logits)
            .map_err(|error| error.to_string())?;
        if values.len() != vocab || values.iter().any(|value| !value.is_finite()) {
            return fail(format!(
                "nonfinite or malformed logits at token position {position}"
            ));
        }
        let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp_sum = values
            .iter()
            .map(|value| ((*value as f64) - max_value).exp())
            .sum::<f64>();
        let logsumexp = max_value + exp_sum.ln();
        let mut indices = (0..vocab).collect::<Vec<_>>();
        indices.sort_unstable_by(|left, right| {
            values[*right]
                .partial_cmp(&values[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        let top = indices.into_iter().take(32).collect::<Vec<_>>();
        let top_logits = top.iter().map(|&id| values[id] as f64).collect::<Vec<_>>();
        let top1 = top[0];
        let target_id = tokens.get(position + 1).copied();
        let target_logit = target_id.map(|id| values[id as usize] as f64);
        if let Some(target) = target_logit {
            nlls.push(logsumexp - target);
        }
        rows.push(json!({
            "position": position,
            "input_id": token,
            "target_id": target_id,
            "target_logit": target_logit,
            "logsumexp": logsumexp,
            "top_ids": top,
            "top_logits": top_logits,
            "top1": top1,
        }));
        state_digests.push(quality_state_digest(&bundle)?);
    }
    gpu.free_tensor(logits).map_err(|error| error.to_string())?;
    let final_state_sha256 =
        sha256_hex(&serde_json::to_vec(&state_digests).map_err(|error| error.to_string())?);
    bundle
        .free_gpu(&mut gpu)
        .map_err(|error| format!("qwen4 bundle teardown failed: {error}"))?;
    let binary_sha256 = env::current_exe()
        .ok()
        .and_then(|path| sha256_path(&path).ok());
    let ppl = if nlls.is_empty() {
        Value::Null
    } else {
        json!((nlls.iter().sum::<f64>() / nlls.len() as f64).exp())
    };
    let quality_rows = json!([{
        "variant": "qwen4-candidate",
        "arch": gpu.arch,
        "scoring_mode": "teacher_forced",
        "n_chunks": 1,
        "mean_kld": Value::Null,
        "mean_kld_ci_lo": Value::Null,
        "mean_kld_ci_hi": Value::Null,
        "p99_kld": Value::Null,
        "ppl": ppl,
        "notes": format!("n_tokens={}; n_scored={}", tokens.len(), nlls.len()),
    }]);
    let report = json!({
        "schema": QUALITY_SCHEMA,
        "variant": "qwen4-candidate",
        "tokens": tokens,
        "corpus": corpus,
        "corpus_sha256": CANONICAL_TOKEN_SHA256,
        "rows": rows,
        "quality_rows": quality_rows,
        "source": {
            "artifact_sha256": artifact_sha256,
            "source_identity_sha256": identity_sha256,
            "source_tensor_count": receipt.source_tensor_count,
            "binary_sha256": binary_sha256,
            "no_full_model_residency": false,
            "forward_route": "production_qwen4_gpu_forward",
        },
        "state_summary_sha256": final_state_sha256,
        "mtp": {"status": "unavailable", "reason": "native MTP adapter is not admitted"},
    });
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(
        output_path,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    Ok(())
}

fn fixture_cases(fixtures: &Value) -> Result<Vec<(String, String, Value)>, String> {
    let values = fixtures
        .as_array()
        .ok_or_else(|| "manifest fixtures is not an array".to_string())?;
    let mut cases = Vec::with_capacity(values.len());
    for fixture in values {
        let name = fixture
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "fixture has no name".to_string())?;
        let path = fixture
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("fixture {name} has no path"))?;
        let schema = fixture
            .get("schema")
            .cloned()
            .ok_or_else(|| format!("fixture {name} has no schema"))?;
        cases.push((name.to_string(), path.to_string(), schema));
    }
    Ok(cases)
}

enum ParityMode {
    Fixtures {
        directory: PathBuf,
        output: PathBuf,
    },
    Candidate {
        model: PathBuf,
        tokens: PathBuf,
        output: PathBuf,
    },
    State {
        model: PathBuf,
        tokens: PathBuf,
        output: PathBuf,
    },
    Profile {
        model: PathBuf,
        tokens: PathBuf,
        output: PathBuf,
    },
}

fn parse_args() -> Result<ParityMode, String> {
    let mut mode = None;
    let mut fixtures = None;
    let mut model = None;
    let mut tokens = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => mode = Some(args.next().ok_or("--mode needs NAME")?),
            "--fixtures" => {
                fixtures = Some(PathBuf::from(args.next().ok_or("--fixtures needs DIR")?))
            }
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs FILE")?)),
            "--tokens" => {
                tokens = Some(PathBuf::from(
                    args.next().ok_or("--tokens needs metadata JSON")?,
                ))
            }
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs FILE")?)),
            other => {
                return fail(format!(
                    "unsupported argument {other:?}; use --fixtures DIR --out FILE, --model FILE --tokens CORPUS --out FILE, --mode state --model FILE --tokens CORPUS --out FILE, or --mode profile --model FILE --tokens CORPUS --out FILE"
                ));
            }
        }
    }
    let output = out.ok_or("missing --out FILE")?;
    match (mode.as_deref(), fixtures, model, tokens) {
        (None, Some(directory), None, None) => Ok(ParityMode::Fixtures { directory, output }),
        (None, None, Some(model), Some(tokens)) => Ok(ParityMode::Candidate {
            model,
            tokens,
            output,
        }),
        (Some("state"), None, Some(model), Some(tokens)) => Ok(ParityMode::State {
            model,
            tokens,
            output,
        }),
        (Some("profile"), None, Some(model), Some(tokens)) => Ok(ParityMode::Profile {
            model,
            tokens,
            output,
        }),
        (Some(other), _, _, _) => fail(format!(
            "unsupported --mode {other:?}; only --mode state and --mode profile are available"
        )),
        _ => fail(
            "choose exactly one mode: --fixtures DIR, --model FILE --tokens CORPUS, --mode state --model FILE --tokens CORPUS, or --mode profile --model FILE --tokens CORPUS",
        ),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen4_parity: {error}");
        std::process::exit(1);
    }
}

fn run_fixtures(fixtures_dir: PathBuf, output_path: PathBuf) -> Result<(), String> {
    let manifest_path = fixtures_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_str(&manifest_text)
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(ORACLE_SCHEMA) {
        return fail(format!("manifest schema must be {ORACLE_SCHEMA}"));
    }
    let cases = fixture_cases(manifest.get("fixtures").ok_or("manifest has no fixtures")?)?;
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let mut results = Vec::with_capacity(cases.len());
    for (name, relative_path, _schema) in cases {
        let arrays = read_npz(&fixtures_dir.join(&relative_path))?;
        let result = match name.as_str() {
            "ple_hash_history" => run_ple_hash(&mut gpu, &arrays)?,
            "ple_projection_dilated_conv" => run_ple_projection(&mut gpu, &arrays, &manifest)?,
            "gdn_recurrence_conv_head_expansion" => run_gdn(&mut gpu, &arrays, &manifest)?,
            "qsa_pool_selection_mask_tail_rope" => run_qsa(&mut gpu, &arrays, &manifest)?,
            "hc_prepare_inject_final_mix" => run_hc(&mut gpu, &arrays, &manifest)?,
            "moe_top10_normalized_shared" => run_moe(&mut gpu, &arrays, &manifest)?,
            "native_mtp_embedding_qsa_moe_hc" => run_mtp(&mut gpu, &arrays, &manifest)?,
            other => return fail(format!("unknown fixture case {other}")),
        };
        results.push(result);
    }
    let report = json!({
        "schema": REPORT_SCHEMA,
        "oracle_schema": ORACLE_SCHEMA,
        "gpu_arch": gpu.arch,
        "fixtures": fixtures_dir,
        "results": results,
        "tolerances": manifest.pointer("/equations/tolerances").cloned().unwrap_or(Value::Null),
    });
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    println!("qwen4 parity PASS: {}", output_path.display());
    Ok(())
}

const SEALED_MOE_ROUTE: &str =
    "BoundMoeExperts::from_cache -> seal_decode -> execute_steps(Step::Moe)";

fn write_profile_report(
    report: hipfire_arch_qwen4::state_parity::ProfileReport,
    output_path: &Path,
) -> Result<(), String> {
    let report = report.into_json();
    for arm in ["target", "mtp"] {
        let evidence = report
            .pointer(&format!("/execution/sealed_moe_validation/{arm}"))
            .ok_or_else(|| format!("profile report has no sealed MoE evidence for {arm}"))?;
        let calls = evidence
            .get("calls")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("profile report has malformed sealed MoE calls for {arm}"))?;
        if calls == 0
            || evidence.get("sealed_route_executed") != Some(&Value::Bool(true))
            || evidence.get("dirty_reuse") != Some(&Value::Bool(true))
            || evidence.get("route").and_then(Value::as_str) != Some(SEALED_MOE_ROUTE)
        {
            return fail(format!(
                "profile report did not prove dirty sealed MoE execution for {arm}"
            ));
        }
    }
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(
        output_path,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    Ok(())
}

fn run() -> Result<(), String> {
    match parse_args()? {
        ParityMode::Fixtures { directory, output } => run_fixtures(directory, output),
        ParityMode::Candidate {
            model,
            tokens,
            output,
        } => {
            run_quality_candidate(&model, &tokens, &output)?;
            println!("qwen4 quality candidate written: {}", output.display());
            Ok(())
        }
        ParityMode::State {
            model,
            tokens,
            output,
        } => {
            let report = hipfire_arch_qwen4::state_parity::run_state_parity(&model, &tokens)?;
            report.write(&output)?;
            println!("qwen4 state parity PASS: {}", output.display());
            Ok(())
        }
        ParityMode::Profile {
            model,
            tokens,
            output,
        } => {
            let report = hipfire_arch_qwen4::state_parity::run_profile(&model, &tokens)?;
            write_profile_report(report, &output)?;
            println!("qwen4 bounded profile written: {}", output.display());
            Ok(())
        }
    }
}
