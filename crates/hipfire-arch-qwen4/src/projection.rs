// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-private MQv2 projection views shared by the Qwen4 trunk and MTP.
//!
//! The packed row geometry and FWHT basis dispatch live here exactly once.  A
//! projection caller supplies logical `(m, k)` dimensions while the view keeps
//! encoded byte capacity for sealed expert-resource validation.

use hipfire_dispatch::families::gemv::WeightRef;
use rdna_compute::{DType, GpuTensor};

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
    /// Placement of the allocation the view was cut from. The view's own
    /// buffer is `Borrowed` and must not claim host ownership, so routing reads
    /// the owner's residency from here instead of `Gpu::host_located(view)`.
    pub(crate) host_mapped: bool,
}

impl ProjectionView {
    pub(crate) fn from_source(source: &GpuTensor, m: usize, k: usize) -> Self {
        Self {
            tensor: alias_tensor(source),
            dtype: source.dtype,
            m,
            k,
            row_stride: row_stride(source.dtype, k),
            host_mapped: source.buf.is_host_mapped(),
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
            host_mapped: source.buf.is_host_mapped(),
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
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A routed expert view is a `Borrowed` byte view, yet it must report the
    /// placement of its host-mapped owner — that is what selects the NT8
    /// host-mapped expert GEMMs. A device owner must stay device.
    #[test]
    fn packed_view_reports_owner_placement() {
        let mut backing = vec![0u8; 256];
        let ptr = backing.as_mut_ptr().cast();
        // SAFETY: descriptors only; no kernel reads them and `DeviceBuffer`
        // has no Drop, so nothing is freed.
        let host = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_host_mapped(ptr, 256) },
            shape: vec![256],
            dtype: DType::F32,
        };
        let view = ProjectionView::from_packed(&host, 64, 64, DType::F32, 1, 64);
        assert!(!view.tensor.buf.is_host_mapped(), "a borrowed view must not claim host ownership");
        assert!(view.host_mapped);
        let device = GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, 256) },
            shape: vec![256],
            dtype: DType::F32,
        };
        assert!(!ProjectionView::from_packed(&device, 64, 64, DType::F32, 1, 64).host_mapped);
    }

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
}
