// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure (no-GPU) registry of the decode-share shadow region.
//!
//! The F16 weight shadows and the Qwen4 shared-expert INT8 shadows live inside one
//! caller-provided, NON-owned device region and are rebuilt in place at the start of every
//! borrow epoch. This module holds the plan (where each shadow lives), its validation and
//! the epoch-gated lookups; [`crate::Gpu`] wires it into the lazy shadow builders.

use crate::shared_i8::shared_i8_shadow_plane_bytes;

/// Destination alignment of every shadow span.
pub const SHADOW_SPAN_ALIGN: usize = 256;

/// One F16 shadow of a BF16 `[rows, cols]` weight: converted into `dst_ptr`
/// (`2 * rows * cols` bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct F16ShadowSpan {
    pub name: String,
    pub src_ptr: usize,
    pub dst_ptr: usize,
    pub rows: usize,
    pub cols: usize,
}

impl F16ShadowSpan {
    /// `2 * rows * cols`; `None` on overflow (rejected by [`ShadowBorrowPlan::validate`]).
    pub fn checked_bytes(&self) -> Option<usize> {
        self.rows.checked_mul(self.cols)?.checked_mul(2)
    }

    /// `2 * rows * cols` (saturating; `validate` rejects overflowing spans).
    pub fn bytes(&self) -> usize {
        self.checked_bytes().unwrap_or(usize::MAX)
    }
}

/// One layer's INT8 shared-expert shadow (planes of `shared_i8_shadow_plane_bytes`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedI8ShadowSpan {
    pub layer: usize,
    pub gate_ptr: usize,
    pub up_ptr: usize,
    pub down_ptr: usize,
    pub intermediate: usize,
    pub hidden: usize,
    pub down_rows: usize,
    pub gate_up_dst: usize,
    pub down_dst: usize,
}

impl SharedI8ShadowSpan {
    /// `(gate_up plane bytes, down plane bytes)`; `None` for an inadmissible shape.
    pub fn plane_bytes(&self) -> Option<(usize, usize)> {
        let (a, b) = shared_i8_shadow_plane_bytes(self.intermediate, self.hidden, self.down_rows)?;
        Some((usize::try_from(a).ok()?, usize::try_from(b).ok()?))
    }
}

/// The whole shadow region: `[region_ptr, region_ptr + region_bytes)` is borrowed device
/// memory (NEVER freed by `Gpu`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowBorrowPlan {
    pub region_ptr: usize,
    pub region_bytes: usize,
    /// Further borrowed regions a destination may lie in (an EP donor set:
    /// one allocation per donor segment); never freed here. Empty for the
    /// single-region decode-share borrow. The byte oracle
    /// ([`Gpu::verify_borrowed_shadows`]) covers the primary region only.
    pub extra_regions: Vec<(usize, usize)>,
    pub f16: Vec<F16ShadowSpan>,
    pub shared_i8: Vec<SharedI8ShadowSpan>,
}

impl ShadowBorrowPlan {
    /// Every destination range lies inside the region, is 256-byte aligned and overlaps no
    /// other; sources are unique; dimensions are nonzero; INT8 shapes are admissible.
    pub fn validate(&self) -> Result<(), String> {
        let region_end = self
            .region_ptr
            .checked_add(self.region_bytes)
            .ok_or_else(|| "shadow region end overflows".to_string())?;
        // (start, end, label)
        let mut ranges: Vec<(usize, usize, String)> = Vec::new();
        let mut push = |start: usize, bytes: usize, label: String| -> Result<(), String> {
            if start % SHADOW_SPAN_ALIGN != 0 {
                return Err(format!(
                    "{label}: destination 0x{start:x} is not {SHADOW_SPAN_ALIGN}-byte aligned"
                ));
            }
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| format!("{label}: destination range overflows"))?;
            let within = |lo: usize, bytes: usize| {
                start >= lo && lo.checked_add(bytes).is_some_and(|hi| end <= hi)
            };
            if !within(self.region_ptr, self.region_bytes)
                && !self
                    .extra_regions
                    .iter()
                    .any(|&(lo, bytes)| within(lo, bytes))
            {
                return Err(format!(
                    "{label}: destination [0x{start:x}, 0x{end:x}) is outside the region \
                     [0x{:x}, 0x{region_end:x})",
                    self.region_ptr
                ));
            }
            ranges.push((start, end, label));
            Ok(())
        };
        let mut f16_srcs = std::collections::HashSet::new();
        for s in &self.f16 {
            if s.rows == 0 || s.cols == 0 {
                return Err(format!("{}: zero dimension {}x{}", s.name, s.rows, s.cols));
            }
            let bytes = s
                .checked_bytes()
                .ok_or_else(|| format!("{}: byte size overflows", s.name))?;
            if !f16_srcs.insert(s.src_ptr) {
                return Err(format!(
                    "{}: duplicate F16 source pointer 0x{:x}",
                    s.name, s.src_ptr
                ));
            }
            push(s.dst_ptr, bytes, s.name.clone())?;
        }
        let mut i8_srcs = std::collections::HashSet::new();
        for s in &self.shared_i8 {
            if s.intermediate == 0 || s.hidden == 0 || s.down_rows == 0 {
                return Err(format!(
                    "layer {}: zero INT8 shadow dimension ({}, {}, {})",
                    s.layer, s.intermediate, s.hidden, s.down_rows
                ));
            }
            let (gu, dn) = s.plane_bytes().ok_or_else(|| {
                format!(
                    "layer {}: INT8 shadow shape ({}, {}, {}) is not admissible",
                    s.layer, s.intermediate, s.hidden, s.down_rows
                )
            })?;
            if !i8_srcs.insert(s.gate_ptr) {
                return Err(format!(
                    "layer {}: duplicate INT8 gate pointer 0x{:x}",
                    s.layer, s.gate_ptr
                ));
            }
            push(s.gate_up_dst, gu, format!("layer {} gate_up", s.layer))?;
            push(s.down_dst, dn, format!("layer {} down", s.layer))?;
        }
        ranges.sort_by_key(|r| (r.0, r.1));
        for w in ranges.windows(2) {
            if w[1].0 < w[0].1 {
                return Err(format!("shadow ranges overlap: {} and {}", w[0].2, w[1].2));
            }
        }
        Ok(())
    }
}

/// A plan plus the borrow-epoch gate: lookups succeed only while `ready`.
#[derive(Clone, Debug)]
pub struct ShadowBorrowState {
    pub plan: ShadowBorrowPlan,
    pub ready: bool,
}

impl ShadowBorrowState {
    /// State for `plan` with `ready = false`.
    pub fn new(plan: ShadowBorrowPlan) -> Self {
        Self { plan, ready: false }
    }

    pub fn set_ready(&mut self, ready: bool) {
        self.ready = ready;
    }

    fn require_ready(&self) -> Result<(), String> {
        if self.ready {
            Ok(())
        } else {
            Err("borrowed shadow read outside a borrow epoch (shadows are not built)".to_string())
        }
    }

    /// Destination of the F16 shadow of the weight at `src_ptr`; `need_bytes` (the rows a
    /// caller reads) must not exceed the planned matrix (a row prefix is the same shadow).
    pub fn f16_lookup(&self, src_ptr: usize, need_bytes: usize) -> Result<usize, String> {
        self.require_ready()?;
        let span = self
            .plan
            .f16
            .iter()
            .find(|s| s.src_ptr == src_ptr)
            .ok_or_else(|| {
                format!(
                    "weight 0x{src_ptr:x} has no planned shadow; shadows are never allocated \
                     elsewhere on a decode-share load"
                )
            })?;
        if need_bytes > span.bytes() {
            return Err(format!(
                "{}: shadow is {} bytes but {need_bytes} were requested",
                span.name,
                span.bytes()
            ));
        }
        Ok(span.dst_ptr)
    }

    /// `(gate_up, down)` destinations of the INT8 shadow keyed on `gate_ptr`.
    pub fn i8_lookup(
        &self,
        gate_ptr: usize,
        intermediate: usize,
        hidden: usize,
        down_rows: usize,
    ) -> Result<(usize, usize), String> {
        self.require_ready()?;
        let span = self
            .plan
            .shared_i8
            .iter()
            .find(|s| s.gate_ptr == gate_ptr)
            .ok_or_else(|| {
                format!(
                    "shared gate weight 0x{gate_ptr:x} has no planned INT8 shadow; shadows are \
                     never allocated elsewhere on a decode-share load"
                )
            })?;
        if span.intermediate != intermediate || span.hidden != hidden || span.down_rows != down_rows
        {
            return Err(format!(
                "layer {}: planned INT8 shadow shape ({}, {}, {}) != requested \
                 ({intermediate}, {hidden}, {down_rows})",
                span.layer, span.intermediate, span.hidden, span.down_rows
            ));
        }
        Ok((span.gate_up_dst, span.down_dst))
    }

    /// Every span as `(region offset, bytes)`, sorted by offset (validated plans only).
    pub fn covered_ranges(&self) -> Vec<(usize, usize)> {
        let base = self.plan.region_ptr;
        let mut out = Vec::new();
        for s in &self.plan.f16 {
            if let Some(at) = s
                .dst_ptr
                .checked_sub(base)
                .filter(|o| o + s.bytes() <= self.plan.region_bytes)
            {
                out.push((at, s.bytes()));
            }
        }
        for s in &self.plan.shared_i8 {
            if let Some((gu, dn)) = s.plane_bytes() {
                if let (Some(gu_at), Some(dn_at)) = (
                    s.gate_up_dst
                        .checked_sub(base)
                        .filter(|o| o + gu <= self.plan.region_bytes),
                    s.down_dst
                        .checked_sub(base)
                        .filter(|o| o + dn <= self.plan.region_bytes),
                ) {
                    out.push((gu_at, gu));
                    out.push((dn_at, dn));
                }
            }
        }
        out.sort();
        out
    }

    /// Names the member covering `region_offset` (relative to `region_ptr`) and the byte
    /// offset inside it.
    pub fn locate(&self, region_offset: usize) -> String {
        let abs = self.plan.region_ptr.wrapping_add(region_offset);
        let hit = |start: usize, bytes: usize| abs >= start && abs - start < bytes;
        for s in &self.plan.f16 {
            if hit(s.dst_ptr, s.bytes()) {
                return format!("{} +{}", s.name, abs - s.dst_ptr);
            }
        }
        for s in &self.plan.shared_i8 {
            if let Some((gu, dn)) = s.plane_bytes() {
                if hit(s.gate_up_dst, gu) {
                    return format!("layer {} gate_up +{}", s.layer, abs - s.gate_up_dst);
                }
                if hit(s.down_dst, dn) {
                    return format!("layer {} down +{}", s.layer, abs - s.down_dst);
                }
            }
        }
        format!("region offset {region_offset} (not inside any planned member)")
    }
}

/// Index of the first differing byte of two equal-length slices (`None` when equal; for a
/// length mismatch, the shorter length).
pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let n = a.len().min(b.len());
    match a[..n].iter().zip(&b[..n]).position(|(x, y)| x != y) {
        Some(i) => Some(i),
        None if a.len() != b.len() => Some(n),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: usize = 0x10_0000;

    fn i8_span(layer: usize, gate: usize, gu: usize, dn: usize) -> SharedI8ShadowSpan {
        SharedI8ShadowSpan {
            layer,
            gate_ptr: gate,
            up_ptr: gate + 1,
            down_ptr: gate + 2,
            intermediate: 128,
            hidden: 256,
            down_rows: 256,
            gate_up_dst: gu,
            down_dst: dn,
        }
    }

    fn good() -> ShadowBorrowPlan {
        let (gu, dn) = shared_i8_shadow_plane_bytes(128, 256, 256).unwrap();
        let (gu, dn) = (gu as usize, dn as usize);
        let f16_bytes: usize = 2 * 16 * 32;
        let i8_at = BASE + f16_bytes.next_multiple_of(256);
        let gu_dst = i8_at;
        let dn_dst = (gu_dst + gu).next_multiple_of(256);
        ShadowBorrowPlan {
            region_ptr: BASE,
            region_bytes: dn_dst + dn - BASE,
            extra_regions: Vec::new(),
            f16: vec![F16ShadowSpan {
                name: "w0".into(),
                src_ptr: 0x9000,
                dst_ptr: BASE,
                rows: 16,
                cols: 32,
            }],
            shared_i8: vec![i8_span(3, 0xA000, gu_dst, dn_dst)],
        }
    }

    #[test]
    fn validate_accepts_good_plan() {
        good().validate().unwrap();
    }

    #[test]
    fn validate_rejects_bad_plans() {
        let mut p = good();
        p.region_bytes -= 256;
        assert!(p.validate().unwrap_err().contains("outside"));

        let mut p = good();
        p.f16[0].dst_ptr = p.shared_i8[0].gate_up_dst;
        assert!(p.validate().unwrap_err().contains("overlap"));

        let mut p = good();
        p.f16[0].dst_ptr += 8;
        assert!(p.validate().unwrap_err().contains("aligned"));

        let mut p = good();
        let mut dup = p.f16[0].clone();
        dup.name = "w1".into();
        dup.dst_ptr = BASE + 0x100000;
        p.region_bytes += 0x200000;
        p.f16.push(dup);
        assert!(p.validate().unwrap_err().contains("duplicate"));

        let mut p = good();
        p.f16[0].rows = 0;
        assert!(p.validate().unwrap_err().contains("zero"));

        let mut p = good();
        p.shared_i8[0].hidden = 100;
        assert!(p.validate().unwrap_err().contains("admissible"));

        let mut p = good();
        let mut dup = p.shared_i8[0].clone();
        dup.layer = 4;
        p.shared_i8.push(dup);
        assert!(p.validate().is_err());
    }

    #[test]
    fn lookups() {
        let mut st = ShadowBorrowState::new(good());
        let gu = st.plan.shared_i8[0].gate_up_dst;
        let dn = st.plan.shared_i8[0].down_dst;
        assert!(st
            .f16_lookup(0x9000, 1024)
            .unwrap_err()
            .contains("borrowed shadow read outside a borrow epoch"));
        assert!(st
            .i8_lookup(0xA000, 128, 256, 256)
            .unwrap_err()
            .contains("borrowed shadow read outside a borrow epoch"));
        st.set_ready(true);
        assert!(st
            .f16_lookup(0x1234, 1024)
            .unwrap_err()
            .contains("never allocated elsewhere"));
        assert!(st
            .f16_lookup(0x9000, 2000)
            .unwrap_err()
            .contains("2000 were requested"));
        // A prefix request (fewer rows than the planned matrix) reads the same shadow.
        assert_eq!(st.f16_lookup(0x9000, 512).unwrap(), BASE);
        assert_eq!(st.f16_lookup(0x9000, 1024).unwrap(), BASE);
        assert!(st
            .i8_lookup(0x1234, 128, 256, 256)
            .unwrap_err()
            .contains("no planned INT8 shadow"));
        assert!(st
            .i8_lookup(0xA000, 128, 512, 256)
            .unwrap_err()
            .contains("planned INT8 shadow shape"));
        assert_eq!(st.i8_lookup(0xA000, 128, 256, 256).unwrap(), (gu, dn));
        st.set_ready(false);
        assert!(st.f16_lookup(0x9000, 1024).is_err());
    }

    #[test]
    fn locate_and_first_difference() {
        let st = ShadowBorrowState::new(good());
        assert_eq!(st.locate(10), "w0 +10");
        let gu = st.plan.shared_i8[0].gate_up_dst - BASE;
        let dn = st.plan.shared_i8[0].down_dst - BASE;
        assert_eq!(st.locate(gu + 5), "layer 3 gate_up +5");
        assert_eq!(st.locate(dn + 7), "layer 3 down +7");
        assert!(st.locate(0x100000).contains("not inside"));
        assert_eq!(first_difference(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_difference(&[1, 2], &[1, 2, 3]), Some(2));
        let r = st.covered_ranges();
        assert_eq!(r[0], (0, 1024));
    }
}
