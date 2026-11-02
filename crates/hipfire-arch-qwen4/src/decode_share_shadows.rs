// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The weight-shadow region of a decode-share donor tail.
//!
//! On exact gfx1201 the BF16 -> F16 weight shadows
//! (`Gpu::ensure_bf16_f16_shadow`) and the shared-expert INT8 shadows
//! (`Gpu::qwen4_shared_i8_shadow`) are read only by prefill tiles of at least
//! 512 rows, and on a decode-share load every such tile is a borrowed prompt
//! tile. A load that keeps them in the donor tail charges neither the census
//! `f16_shadows` nor the `shared_i8_shadows` term; the shadows are rebuilt in
//! place at the first acquire of every borrow epoch and vanish with the next
//! restore (the donors come back).
//!
//! This module is the one pure authority on that region's layout: the
//! loader sizes the class spans from [`ShadowRegionPlan::total_bytes`] and the
//! forward places every member from the same plan, so the two can never
//! disagree. The region sits at the START of the donor tail (every class
//! dirties it); the scratch backing follows it
//! ([`crate::gpu_forward::qwen4_decode_share_class_spans`]).

use crate::expert_residency::{
    f16_shadow_entry_bytes, is_f16_shadow_owner, shared_i8_shadow_layers, SharedI8ShadowLayer,
};
use hipfire_runtime::weight_manifest::WeightEntry;

/// Byte alignment of every member of the region (the device allocation
/// alignment every shadow kernel and the 16-byte INT8 plane loads assume).
pub const SHADOW_REGION_ALIGN: u64 = 256;

/// One F16 shadow of a BF16 `[rows, cols]` weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowF16Member {
    /// Manifest name of the source weight.
    pub name: String,
    /// Manifest layer of the source weight.
    pub layer: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    /// Byte offset inside the region (256-aligned).
    pub offset: u64,
    /// `2 * rows * cols`.
    pub bytes: u64,
}

/// The INT8 planes of one layer's shared expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowI8Member {
    pub layer: usize,
    pub intermediate: usize,
    pub hidden: usize,
    pub down_rows: usize,
    pub gate_up_offset: u64,
    pub gate_up_bytes: u64,
    pub down_offset: u64,
    pub down_bytes: u64,
}

/// Layout of the shadow region: F16 members first (manifest order), then the
/// INT8 planes (layer order), each 256-aligned; no gaps other than alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowRegionPlan {
    pub f16: Vec<ShadowF16Member>,
    pub shared_i8: Vec<ShadowI8Member>,
    /// Sum of the F16 payloads (the census `f16_shadows` term).
    pub f16_bytes: u64,
    /// Sum of the INT8 planes (the census `shared_i8_shadows` term).
    pub i8_bytes: u64,
    /// Region extent: the end of the last member rounded up to 256. 0 for an
    /// empty plan.
    pub total_bytes: u64,
}

fn align_up(bytes: u64) -> Option<u64> {
    bytes
        .checked_add(SHADOW_REGION_ALIGN - 1)
        .map(|bytes| bytes & !(SHADOW_REGION_ALIGN - 1))
}

/// Which shadow families a decode-share load of `chunk_rows`-row prompt tiles
/// borrows: `(f16, shared_i8)` under the census predicates (the device can
/// build the shadows, `route_rows >= 512` for the largest tile). `Err` names
/// the reason borrowing is unsound: a call below the resident bank's rows
/// would also route to the shadow-reading arms (canonical routing rows), so a
/// shadow could be read outside a borrow epoch.
pub fn shadow_flags(
    arch: &str,
    f16_capable: bool,
    shared_i8_enabled: bool,
    chunk_rows: usize,
) -> Result<(bool, bool), &'static str> {
    let min_rows = rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS;
    if crate::expert_residency::f16_shadow_route_rows(
        arch,
        crate::gpu_forward::QWEN4_DECODE_SHARE_RESIDENT_ROWS - 1,
    ) >= min_rows
    {
        return Err("a sub-512-row call routes to the F16 WMMA arms (canonical routing rows)");
    }
    let rows = crate::expert_residency::f16_shadow_route_rows(arch, chunk_rows);
    Ok((
        f16_capable && rows >= min_rows,
        shared_i8_enabled && rows >= min_rows,
    ))
}

/// Lay out the shadow region of `weights` (the manifest entries): the F16
/// shadows when `f16` (the census predicate: shadow-capable device and
/// `route_rows >= 512`) and the INT8 shadows when `shared_i8` (route enabled
/// and `route_rows >= 512`). The members are exactly the entries
/// [`crate::expert_residency::f16_shadow_bytes`] and
/// [`crate::expert_residency::shared_i8_shadow_bytes`] charge.
pub fn plan_shadow_region(
    weights: &[WeightEntry],
    num_layers: usize,
    f16: bool,
    shared_i8: bool,
) -> Result<ShadowRegionPlan, String> {
    let overflow = || "decode-share shadow region bytes overflow".to_string();
    let mut cursor = 0u64;
    let mut f16_members = Vec::new();
    let mut f16_bytes = 0u64;
    if f16 {
        for entry in weights {
            if !is_f16_shadow_owner(entry, num_layers, &|_| true, true) {
                continue;
            }
            let bytes = f16_shadow_entry_bytes(entry)?;
            let offset = align_up(cursor).ok_or_else(overflow)?;
            cursor = offset.checked_add(bytes).ok_or_else(overflow)?;
            f16_bytes = f16_bytes.checked_add(bytes).ok_or_else(overflow)?;
            f16_members.push(ShadowF16Member {
                name: entry.name.clone(),
                layer: entry.layer,
                rows: entry.logical_shape[0],
                cols: entry.logical_shape[1],
                offset,
                bytes,
            });
        }
    }
    let mut i8_members = Vec::new();
    let mut i8_bytes = 0u64;
    if shared_i8 {
        for SharedI8ShadowLayer {
            layer,
            intermediate,
            hidden,
            down_rows,
            gate_up_bytes,
            down_bytes,
        } in shared_i8_shadow_layers(weights, num_layers)?
        {
            let gate_up_offset = align_up(cursor).ok_or_else(overflow)?;
            let down_offset = align_up(
                gate_up_offset
                    .checked_add(gate_up_bytes)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
            cursor = down_offset.checked_add(down_bytes).ok_or_else(overflow)?;
            i8_bytes = i8_bytes
                .checked_add(gate_up_bytes)
                .and_then(|bytes| bytes.checked_add(down_bytes))
                .ok_or_else(overflow)?;
            i8_members.push(ShadowI8Member {
                layer,
                intermediate,
                hidden,
                down_rows,
                gate_up_offset,
                gate_up_bytes,
                down_offset,
                down_bytes,
            });
        }
    }
    let total_bytes = if f16_members.is_empty() && i8_members.is_empty() {
        0
    } else {
        align_up(cursor).ok_or_else(overflow)?
    };
    Ok(ShadowRegionPlan {
        f16: f16_members,
        shared_i8: i8_members,
        f16_bytes,
        i8_bytes,
        total_bytes,
    })
}

impl ShadowRegionPlan {
    /// Whether the plan holds no member (nothing to borrow).
    pub fn is_empty(&self) -> bool {
        self.f16.is_empty() && self.shared_i8.is_empty()
    }

    /// `Some(self)` unless the plan holds no member.
    pub fn into_nonempty(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }

    /// Every member's `[offset, offset + bytes)` in layout order, labelled
    /// (F16 name / `layer N shared INT8 gate_up|down`): for the structural
    /// checks and the oracle's mismatch report.
    pub fn spans(&self) -> Vec<(String, u64, u64)> {
        let mut spans = Vec::with_capacity(self.f16.len() + 2 * self.shared_i8.len());
        for member in &self.f16 {
            spans.push((member.name.clone(), member.offset, member.bytes));
        }
        for member in &self.shared_i8 {
            spans.push((
                format!("layer {} shared INT8 gate_up", member.layer),
                member.gate_up_offset,
                member.gate_up_bytes,
            ));
            spans.push((
                format!("layer {} shared INT8 down", member.layer),
                member.down_offset,
                member.down_bytes,
            ));
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_residency::{f16_shadow_bytes, j1778_bf16_manifest, shared_i8_shadow_bytes};

    const LAYERS: usize = 48;

    #[test]
    fn region_members_are_exactly_the_census_owners() {
        let weights = j1778_bf16_manifest();
        let plan = plan_shadow_region(&weights, LAYERS, true, true).unwrap();
        assert_eq!(plan.f16_bytes, f16_shadow_bytes(&weights, LAYERS).unwrap());
        assert_eq!(
            plan.i8_bytes,
            shared_i8_shadow_bytes(&weights, LAYERS).unwrap()
        );
        assert_eq!(plan.f16.len(), 291);
        eprintln!(
            "shadow region: {} F16 members {} B ({:.1} MiB), {} INT8 layers {} B ({:.1} MiB), region {} B ({:.1} MiB)",
            plan.f16.len(),
            plan.f16_bytes,
            plan.f16_bytes as f64 / 1048576.0,
            plan.shared_i8.len(),
            plan.i8_bytes,
            plan.i8_bytes as f64 / 1048576.0,
            plan.total_bytes,
            plan.total_bytes as f64 / 1048576.0
        );
    }

    #[test]
    fn region_is_aligned_disjoint_and_ends_at_the_total() {
        let weights = j1778_bf16_manifest();
        let plan = plan_shadow_region(&weights, LAYERS, true, true).unwrap();
        assert!(!plan.is_empty());
        let mut spans = plan.spans();
        spans.sort_by_key(|span| span.1);
        let mut end = 0u64;
        for (label, offset, bytes) in &spans {
            assert_eq!(offset % SHADOW_REGION_ALIGN, 0, "{label}");
            assert!(*offset >= end, "{label} overlaps the previous member");
            assert!(*bytes > 0, "{label}");
            end = offset + bytes;
        }
        assert_eq!(plan.total_bytes % SHADOW_REGION_ALIGN, 0);
        assert!(plan.total_bytes >= end && plan.total_bytes - end < SHADOW_REGION_ALIGN);
        // Alignment adds at most 255 B per member.
        let payload = plan.f16_bytes + plan.i8_bytes;
        assert!(plan.total_bytes >= payload);
        assert!(plan.total_bytes - payload < SHADOW_REGION_ALIGN * spans.len() as u64);
    }

    #[test]
    fn region_follows_the_flags() {
        let weights = j1778_bf16_manifest();
        let both = plan_shadow_region(&weights, LAYERS, true, true).unwrap();
        let f16_only = plan_shadow_region(&weights, LAYERS, true, false).unwrap();
        let i8_only = plan_shadow_region(&weights, LAYERS, false, true).unwrap();
        let none = plan_shadow_region(&weights, LAYERS, false, false).unwrap();
        assert!(f16_only.shared_i8.is_empty() && f16_only.i8_bytes == 0);
        assert!(i8_only.f16.is_empty() && i8_only.f16_bytes == 0);
        assert_eq!(f16_only.f16_bytes, both.f16_bytes);
        assert_eq!(i8_only.i8_bytes, both.i8_bytes);
        assert!(none.is_empty() && none.total_bytes == 0);
        // Members keep the same shapes whatever else is borrowed.
        assert_eq!(f16_only.f16, both.f16);
    }

    /// The census predicates: capability and route rows >= 512 per family; the
    /// tile-size gate the borrowed epoch relies on holds on gfx1201.
    #[test]
    fn shadow_flags_follow_the_census_predicates() {
        assert_eq!(shadow_flags("gfx1201", true, true, 16384), Ok((true, true)));
        assert_eq!(
            shadow_flags("gfx1201", true, false, 16384),
            Ok((true, false))
        );
        assert_eq!(
            shadow_flags("gfx1201", false, true, 1024),
            Ok((false, true))
        );
        assert_eq!(
            shadow_flags("gfx1201", false, false, 16384),
            Ok((false, false))
        );
        // A chunk below the route threshold never reads a shadow.
        assert_eq!(shadow_flags("gfx1201", true, true, 511), Ok((false, false)));
    }

    #[test]
    fn region_of_a_manifest_without_owners_is_empty() {
        let plan = plan_shadow_region(&[], LAYERS, true, true).unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.total_bytes, 0);
    }
}
