//! Canonical GDN ordinary-prefill numerical plan and absolute-position stream
//! carry contract (CHUNK-CANONICAL-DESIGN §1.1, §3.2).
//!
//! The plan is resolved once from static facts (arch, geometry, weight
//! headers, configured quantization/EF, requested precision flags) and never
//! from an outer call's row count or free VRAM. Actual extents remain the
//! caller's buffers' extents; the plan only fixes arithmetic.

/// Absolute row spacing of Q8/Q8Ef quantization seams for ordinary prefill.
pub const GDN_COMMIT_ROWS: usize = 512;

/// Version of the serialized stream metadata / carry layout. Bump whenever a
/// field's meaning or byte layout changes; restores reject mismatches.
pub const GDN_STREAM_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnScanAlgorithm {
    /// RDNA WMMA chunk scan, 64-row microtiles.
    Scan64,
    /// Qwen4 WY chunk, 16-row microtiles.
    Wy16,
    /// Sequential recurrence with FP32 accumulator continuation (C = 1).
    Sequential,
}

impl GdnScanAlgorithm {
    pub const fn micro_rows(self) -> usize {
        match self {
            Self::Scan64 => 64,
            Self::Wy16 => 16,
            Self::Sequential => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnTensorFormat {
    F32,
    Bf16,
    F16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStateQuant {
    /// Persistent FP32 state; no artificial commit seam.
    F32,
    /// Q8 state, no error feedback.
    Q8,
    /// Q8 state with F16 error feedback.
    Q8Ef,
}

/// Per-field activation/projection arithmetic. Opaque stable code owned by the
/// producer policy (S1); equality means byte-identical arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnProjectionRecipe {
    pub qkv: u32,
    pub z: u32,
    pub alpha: u32,
    pub beta: u32,
    pub out: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnPrefillPlan {
    pub recipe_id: u64,
    pub algorithm: GdnScanAlgorithm,
    pub micro_rows: usize,
    pub commit_rows: usize,
    pub projection_recipe: GdnProjectionRecipe,
    pub scan_operand_format: GdnTensorFormat,
    pub scan_output_format: GdnTensorFormat,
    pub state_quant: GdnStateQuant,
    /// Minimum physical producer tile (rows); padding is private and masked.
    pub producer_tile_rows: usize,
}

impl GdnPrefillPlan {
    /// Stable identity over every numerical field. Excludes extents, VRAM
    /// rungs, pointers and allocation addresses by construction.
    pub fn compute_recipe_id(&self) -> u64 {
        // FNV-1a over a fixed field encoding; stable across processes.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat(GDN_STREAM_SCHEMA_VERSION as u64);
        eat(self.algorithm as u64);
        eat(self.micro_rows as u64);
        eat(self.commit_rows as u64);
        let p = self.projection_recipe;
        for v in [p.qkv, p.z, p.alpha, p.beta, p.out] {
            eat(v as u64);
        }
        eat(self.scan_operand_format as u64);
        eat(self.scan_output_format as u64);
        eat(self.state_quant as u64);
        eat(self.producer_tile_rows as u64);
        h
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStreamPhase {
    /// Q8/EF (or F32) live state is authoritative; decode/verify may read it.
    Sealed,
    /// Ordinary prefill in progress; tile-entry FP32 state is authoritative.
    Open,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnStreamMeta {
    pub schema_version: u32,
    pub recipe_id: u64,
    pub phase: GdnStreamPhase,
    /// Next absolute token (exclusive committed prefix).
    pub position: u64,
    /// Genuine decode→prefill boundary, not a call start.
    pub phase_start: u64,
    /// Absolute origin of the current 512-row commit frame.
    pub frame_base: u64,
    /// Absolute origin of the current microtile.
    pub tile_base: u64,
    /// Valid raw rows retained since `tile_base`; always < micro_rows.
    pub pending_rows: u32,
    pub owner_generation: u64,
}

impl GdnStreamMeta {
    pub fn sealed_at(position: u64, owner_generation: u64) -> Self {
        Self {
            schema_version: GDN_STREAM_SCHEMA_VERSION,
            recipe_id: 0,
            phase: GdnStreamPhase::Sealed,
            position,
            phase_start: position,
            frame_base: position,
            tile_base: position,
            pending_rows: 0,
            owner_generation,
        }
    }
}

/// Geometry of a per-layer persistent carry (§5): FP32 tile-entry state,
/// tile-entry conv history, raw pre-conv QKV and raw alpha/beta for up to
/// `micro_rows` rows. Z is not retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GdnCarryGeometry {
    pub hk: usize,
    pub hv: usize,
    pub head_dim: usize,
    pub conv_hist_rows: usize,
    pub micro_rows: usize,
}

impl GdnCarryGeometry {
    pub const fn state_elems(&self) -> usize {
        self.hv * self.head_dim * self.head_dim
    }
    pub const fn qkv_width(&self) -> usize {
        (2 * self.hk + self.hv) * self.head_dim
    }
    pub const fn state_bytes(&self) -> usize {
        4 * self.state_elems()
    }
    pub const fn conv_bytes(&self) -> usize {
        4 * self.conv_hist_rows * self.qkv_width()
    }
    pub const fn raw_qkv_bytes(&self) -> usize {
        4 * self.micro_rows * self.qkv_width()
    }
    pub const fn raw_ab_bytes(&self) -> usize {
        4 * self.micro_rows * 2 * self.hv
    }
    pub const fn total_bytes(&self) -> usize {
        self.state_bytes() + self.conv_bytes() + self.raw_qkv_bytes() + self.raw_ab_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carry_bytes_match_design_table() {
        let g = GdnCarryGeometry { hk: 16, hv: 48, head_dim: 128, conv_hist_rows: 3, micro_rows: 64 };
        assert_eq!(g.total_bytes(), 5_914_624); // 5.640625 MiB
        let g16 = GdnCarryGeometry { micro_rows: 16, ..g };
        assert_eq!(g16.total_bytes(), 3_930_112); // 3.748046875 MiB
    }
}
