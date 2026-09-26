// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-private MQv2 projection views shared by the Qwen4 trunk and MTP.
//!
//! The packed row geometry and FWHT basis dispatch live here exactly once.  A
//! projection caller supplies logical `(m, k)` dimensions while the view keeps
//! encoded byte capacity for sealed expert-resource validation.

use hipfire_dispatch::families::gemv::WeightRef;
use rdna_compute::{DType, Gpu, GpuTensor};

pub(crate) fn row_stride(dtype: DType, k: usize) -> usize {
    dtype.row_bytes(k).unwrap_or(k * dtype.size())
}

pub(crate) fn checked_bytes(rows: usize, stride: usize) -> Option<usize> {
    rows.checked_mul(stride)
}

fn alias_tensor(source: &GpuTensor) -> GpuTensor {
    GpuTensor {
        buf: unsafe { source.buf.alias() },
        shape: source.shape.clone(),
        dtype: source.dtype,
    }
}

fn packed_view(source: &GpuTensor, byte_offset: usize, byte_len: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: source.buf.byte_view(byte_offset, byte_len),
        shape: vec![byte_len],
        dtype,
    }
}

/// A projection view with logical dimensions separated from encoded bytes.
/// Routed expert views use a packed byte-shaped `GpuTensor` because the sealed
/// live-resource validator intentionally checks encoded capacity, not logical
/// element count.
pub(crate) struct ProjectionView {
    pub(crate) tensor: GpuTensor,
    pub(crate) dtype: DType,
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) row_stride: usize,
}

impl ProjectionView {
    pub(crate) fn from_source(source: &GpuTensor, m: usize, k: usize) -> Self {
        Self {
            tensor: alias_tensor(source),
            dtype: source.dtype,
            m,
            k,
            row_stride: row_stride(source.dtype, k),
        }
    }

    pub(crate) fn from_packed(
        source: &GpuTensor,
        offset: usize,
        bytes: usize,
        dtype: DType,
        m: usize,
        k: usize,
    ) -> Self {
        Self {
            tensor: packed_view(source, offset, bytes, dtype),
            dtype,
            m,
            k,
            row_stride: row_stride(dtype, k),
        }
    }

    pub(crate) fn dispatch_ref(&self) -> WeightRef<'_> {
        WeightRef {
            buf: &self.tensor,
            dtype: self.dtype,
            m: self.m,
            k: self.k,
            row_stride: self.row_stride,
            rotation: None,
            awq_scale: None,
        }
    }
}

/// Dispatch one logical projection with the same MQv2 basis convention as the
/// ordinary Qwen4 trunk.  Inputs/outputs are F32; BF16 remains unrotated.
pub(crate) fn dispatch_gemv(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    input: &GpuTensor,
    rotation: &GpuTensor,
    output: &GpuTensor,
    m: usize,
    k: usize,
) -> hip_bridge::HipResult<()> {
    match weight.dtype {
        DType::MQ4G256V2
        | DType::MQ4G128V2
        | DType::MQ6G256V2
        | DType::MFP4G32E8SOA
        | DType::Q8_0
        | DType::BF16 => {}
        dtype => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("Qwen4 projection has unsupported resident dtype {dtype:?}"),
            ));
        }
    }
    // One projection contract: the shared lowering owns the FWHT basis
    // convention, so a packed payload is rotated and decoded exactly as the
    // trunk's own ops decode it.
    let reference = WeightRef {
        buf: weight,
        dtype: weight.dtype,
        m,
        k,
        row_stride: row_stride(weight.dtype, k),
        rotation: None,
        awq_scale: None,
    };
    hipfire_dispatch::pipeline::project_weight(gpu, &reference, input, output, 1, Some(rotation))
        .map_err(|error| hip_bridge::HipError::new(0, &error.to_string()))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingPath {
    Mq4V2,
    Bf16,
    Q8,
}

fn embedding_path(dtype: DType) -> Option<EmbeddingPath> {
    match dtype {
        DType::MQ4G256V2 | DType::MQ4G128V2 => Some(EmbeddingPath::Mq4V2),
        DType::BF16 => Some(EmbeddingPath::Bf16),
        DType::Q8_0 => Some(EmbeddingPath::Q8),
        _ => None,
    }
}

/// Dispatch one embedding lookup according to the resident table's exact
/// contract.  BF16 rows are widened directly; packed MQv2 rows use the rotated
/// staging buffer and FWHT decode; Q8 rows are block-decoded in place.
pub(crate) fn dispatch_embedding(
    gpu: &mut Gpu,
    embedding: &GpuTensor,
    rotated: &GpuTensor,
    output: &GpuTensor,
    token_ids: &GpuTensor,
    n: usize,
    dim: usize,
) -> hip_bridge::HipResult<()> {
    let path = embedding_path(embedding.dtype).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "Qwen4 embedding has unsupported resident dtype {:?}",
                embedding.dtype
            ),
        )
    })?;
    match path {
        EmbeddingPath::Mq4V2 => {
            gpu.embedding_lookup_mq4v2_batched(embedding, rotated, output, token_ids, n, dim)
        }
        EmbeddingPath::Bf16 => {
            gpu.embedding_lookup_bf16_batched(embedding, output, token_ids, n, dim)
        }
        EmbeddingPath::Q8 => {
            gpu.embedding_lookup_q8_batched(embedding, output, token_ids, n, dim)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stride the arch hands the dispatcher is a second implementation of
    /// the artifact boundary's row geometry.  These are the expert K and the
    /// trunk K values, spelled as the producer's formula so the three copies
    /// (producer, artifact boundary, here) cannot drift apart silently.
    #[test]
    fn packed_row_strides_follow_the_producer_geometry() {
        // E8 SoA: 16-byte header + ceil-to-16 scale bytes + 16 B per block.
        assert_eq!(row_stride(DType::MFP4G32E8SOA, 2560), 16 + 80 + 80 * 16);
        // K = 768 pads its 24 scale bytes to 32 — the padding case, at a K the
        // format actually admits (the routed down reduction is not 256-aligned
        // and the artifact boundary refuses it by name).
        assert_eq!(row_stride(DType::MFP4G32E8SOA, 768), 16 + 32 + 24 * 16);
        // Q8F16: 34-byte blocks of 32 weights.
        assert_eq!(row_stride(DType::Q8_0, 2560), 80 * 34);
        assert_eq!(row_stride(DType::Q8_0, 6144), 192 * 34);
    }

    #[test]
    fn embedding_dispatch_preserves_bf16_and_routed_mqv2_contracts() {
        assert_eq!(embedding_path(DType::BF16), Some(EmbeddingPath::Bf16));
        assert_eq!(embedding_path(DType::MQ4G256V2), Some(EmbeddingPath::Mq4V2));
        assert_eq!(embedding_path(DType::MQ4G128V2), Some(EmbeddingPath::Mq4V2));
        assert_eq!(embedding_path(DType::Q8_0), Some(EmbeddingPath::Q8));
    }

    #[test]
    fn embedding_dispatch_rejects_unadmitted_dtype() {
        assert!(embedding_path(DType::F32).is_none());
    }
}
