//! Canonical GDN ordinary-prefill numerical plan and absolute-position stream
//! carry contract (CHUNK-CANONICAL-DESIGN §1.1, §3.2).
//!
//! The plan is resolved once from static facts (arch, geometry, weight
//! headers, configured quantization/EF, requested precision flags) and never
//! from an outer call's row count or free VRAM. Actual extents remain the
//! caller's buffers' extents; the plan only fixes arithmetic.

use crate::dispatch::GpuTensor;

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

/// First absolute multiple of `grid` strictly greater than `pos`.
pub const fn next_boundary(pos: u64, grid: u64) -> u64 {
    (pos / grid + 1) * grid
}

/// Quantization of one Q8 plane triple. `ef` is present exactly for
/// [`GdnStateQuant::Q8Ef`]; a plain Q8 stream has no residual plane (the
/// kernels never touch a stand-in buffer).
#[derive(Clone, Copy)]
pub struct GdnQ8Planes<'a> {
    /// `[n_value_heads][128][128]` signed codes.
    pub codes: &'a GpuTensor,
    /// `[n_value_heads][128]` f32 row scales.
    pub scales: &'a GpuTensor,
    /// `[n_value_heads][128][128]` f16 error-feedback residual (Q8Ef only).
    pub ef: Option<&'a GpuTensor>,
}

/// Borrowed live recurrent state of one layer: the Q8 planes plus the width-4
/// convolution ring (`[channel][3]`, index 0 the newest raw row). Authoritative
/// only in [`GdnStreamPhase::Sealed`] and at a commit seam; the live ring is
/// always the ring at `position`.
#[derive(Clone, Copy)]
pub struct GdnLiveState<'a> {
    pub planes: GdnQ8Planes<'a>,
    pub conv: &'a GpuTensor,
}

/// Borrowed per-layer persistent carry (GPU buffers owned by the caller; one
/// independent owner per model/layer/lane/device). Layout follows
/// [`GdnCarryGeometry`]: `tile_entry_f32` is `[n_value_heads][128][128]` f32 in
/// the scan's state index order, `tile_entry_conv` has the live ring's layout,
/// `pending_*` hold the raw pre-conv QKV (`[micro_rows][qkv_width]`) and raw
/// alpha/beta (`[micro_rows][n_value_heads]`) rows received since `tile_base`
/// (valid prefix `pending_rows`, f32).
#[derive(Clone, Copy)]
pub struct GdnCarryView<'a> {
    pub tile_entry_f32: &'a GpuTensor,
    pub tile_entry_conv: &'a GpuTensor,
    pub pending_qkv: &'a GpuTensor,
    pub pending_alpha: &'a GpuTensor,
    pub pending_beta: &'a GpuTensor,
}

impl GdnCarryView<'_> {
    /// Byte-extent check against the layer geometry; no device access.
    pub fn check(&self, g: &GdnCarryGeometry) -> Result<(), String> {
        let need = [
            ("tile_entry_f32", self.tile_entry_f32.byte_size(), g.state_bytes()),
            ("tile_entry_conv", self.tile_entry_conv.byte_size(), g.conv_bytes()),
            ("pending_qkv", self.pending_qkv.byte_size(), g.raw_qkv_bytes()),
            ("pending_alpha", self.pending_alpha.byte_size(), 4 * g.micro_rows * g.hv),
            ("pending_beta", self.pending_beta.byte_size(), 4 * g.micro_rows * g.hv),
        ];
        for (name, have, want) in need {
            if have < want {
                return Err(format!("carry {name}: {have} bytes, geometry needs {want}"));
            }
        }
        Ok(())
    }
}

/// Prepared inputs of the sequential (C = 1) algorithm: the caller's own
/// producer already ran the convolution, normalization and gate/beta math, so
/// the stream only owns the recurrence's state continuation. `q`/`k` are
/// `[rows][n_heads / qk_head_ratio * 128]` f32, `v` and the scan `out` are
/// `[rows][n_heads * 128]`, `gate`/`beta` `[rows][n_heads]`.
#[derive(Clone, Copy)]
pub struct GdnSeqInputs<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub qk_head_ratio: usize,
}

/// Per-device edge workspace of the Scan64 stream executor (§5: charged once
/// per simultaneously executing stream, never per layer, never snapshotted):
/// one 64-row tile of raw/prepared planes, its A blocks and scan output, and a
/// conv ring scratch. Create with `Gpu::gdn_stream_workspace_alloc`, release
/// with `Gpu::gdn_stream_workspace_free`.
pub struct GdnStreamWorkspace {
    pub hk: usize,
    pub hv: usize,
    pub out_fmt: crate::norm::GdnScanOut,
    pub x: GpuTensor,
    pub alpha: GpuTensor,
    pub beta: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub a: GpuTensor,
    pub out: GpuTensor,
    pub conv: GpuTensor,
}

/// Device bytes of each plane of one [`GdnStreamWorkspace`], in allocation order (x, alpha,
/// beta, q, k, v, A, out, conv ring; `out_elem` bytes per scan output element: 4 for F32, 2
/// for BF16). `Gpu::gdn_stream_workspace_alloc` allocates exactly these.
pub const fn stream_workspace_plane_bytes(hk: usize, hv: usize, out_elem: usize) -> [usize; 9] {
    let xdim = (2 * hk + hv) * 128;
    [
        64 * xdim * 4,
        64 * hv * 4,
        64 * hv * 4,
        64 * hk * 128 * 2,
        64 * hk * 128 * 2,
        64 * hv * 128 * 2,
        64 * hv * 64 * 2,
        64 * hv * 128 * out_elem,
        xdim * 3 * 4,
    ]
}

/// Device bytes of one [`GdnStreamWorkspace`]: the sum of its planes. Pure, for admission
/// census.
pub const fn stream_workspace_bytes(hk: usize, hv: usize, out_elem: usize) -> usize {
    let p = stream_workspace_plane_bytes(hk, hv, out_elem);
    p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7] + p[8]
}

/// Model constants of one GDN layer's prep (never carried).
#[derive(Clone, Copy)]
pub struct GdnStreamConsts<'a> {
    pub conv_weight: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub q_scale: f32,
    pub eps: f32,
    pub n_key_heads: usize,
    pub n_value_heads: usize,
}

/// Canonical projected inputs of the rows being appended: post-projection,
/// pre-convolution QKV (`[rows][qkv_width]` f32) and raw alpha/beta
/// (`[rows][n_value_heads]` f32). The prep rewrites `alpha`/`beta` in place for
/// the rows it runs on the caller's buffers (gates/sigmoid), exactly like the
/// whole-call route; the raw rows a later append needs are saved first.
#[derive(Clone, Copy)]
pub struct GdnStreamRaw<'a> {
    pub qkv: &'a GpuTensor,
    pub alpha: &'a GpuTensor,
    pub beta: &'a GpuTensor,
}

/// Caller-owned prepared-plane scratch for the rows past the head tile: q/k
/// (`[rows][n_key_heads*128]`) and v (`[rows][n_value_heads*128]`) f16 planes
/// and the 64-row A blocks (`[ceil64(rows)][n_value_heads*64]` f16).
#[derive(Clone, Copy)]
pub struct GdnStreamScratch<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub a: &'a GpuTensor,
}

/// Incoming state of one chunk-scan launch. The variant names the numerical
/// reconstruction; there are no null-pointer conventions.
#[derive(Clone, Copy)]
pub enum GdnScanEntry<'a> {
    /// Plain Q8: `codes * scale`, no residual.
    Q8 { codes: &'a GpuTensor, scales: &'a GpuTensor },
    /// Q8 + F16 EF: `codes * scale + ef`, reconstructed once at entry.
    Q8Ef { codes: &'a GpuTensor, scales: &'a GpuTensor, ef: &'a GpuTensor },
    /// A tile-entry FP32 accumulator, taken unchanged.
    F32 { state: &'a GpuTensor },
}

/// Outgoing state of one chunk-scan launch.
#[derive(Clone, Copy)]
pub enum GdnScanExit<'a> {
    /// Quantize the final state once (call end / seal): codes, scales and, when
    /// `ef` is present, the F16 residual.
    Seal { codes: &'a GpuTensor, scales: &'a GpuTensor, ef: Option<&'a GpuTensor> },
    /// Keep the unquantized FP32 accumulator (a completed tile that is not a
    /// commit seam).
    F32 { state: &'a GpuTensor },
    /// Quantize at an absolute commit seam: the `Seal` stores plus `next`, the
    /// reconstructed FP32 state the following frame starts from. A plain Q8
    /// stream reconstructs with the rounding error it would have carried in
    /// registers across the seam, exactly like one whole call.
    Seam {
        codes: &'a GpuTensor,
        scales: &'a GpuTensor,
        ef: Option<&'a GpuTensor>,
        next: &'a GpuTensor,
    },
    /// Drop the end state (transient replay of an incomplete tile).
    Discard,
}

/// Kernarg mode words shared with `gdn_chunk_scan*.hip` (`GS_ENTRY_*`,
/// `GS_EXIT_*`).
pub const GS_ENTRY_Q8: i32 = 0;
pub const GS_ENTRY_Q8EF: i32 = 1;
pub const GS_ENTRY_F32: i32 = 2;
pub const GS_EXIT_SEAL: i32 = 0;
pub const GS_EXIT_F32: i32 = 1;
pub const GS_EXIT_SEAM: i32 = 2;
pub const GS_EXIT_NONE: i32 = 3;
pub const GS_EXIT_EF: i32 = 4;

/// How one append of `rows` new rows decomposes against the absolute tile and
/// seam grid. Pure arithmetic over [`GdnStreamMeta`]; the executor and the
/// oracles share it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnAppendSchedule {
    /// The head tile (pending rows replayed, or a clipped first tile) runs in
    /// the workspace and consumes `head_new` new rows.
    pub head: bool,
    pub head_pending: usize,
    pub head_new: usize,
    pub head_complete: bool,
    /// New rows after the head, in whole tiles / the incomplete final tile.
    pub bulk_complete: usize,
    pub bulk_tail: usize,
    /// Candidate metadata after a successful append.
    pub meta: GdnStreamMeta,
}

impl GdnAppendSchedule {
    /// Validate `meta` against `start_pos` and split `rows`. `micro` and
    /// `commit` are the plan's tile and seam spacings.
    pub fn new(
        meta: &GdnStreamMeta,
        start_pos: u64,
        rows: usize,
        micro: usize,
        commit: usize,
    ) -> Result<Self, String> {
        let (micro64, commit64) = (micro as u64, commit as u64);
        if meta.schema_version != GDN_STREAM_SCHEMA_VERSION {
            return Err(format!(
                "stream schema {} != {GDN_STREAM_SCHEMA_VERSION}",
                meta.schema_version
            ));
        }
        if rows == 0 {
            return Err("append of zero rows".into());
        }
        if meta.position != start_pos {
            return Err(format!("stream position {} != start_pos {start_pos}", meta.position));
        }
        let open = meta.phase == GdnStreamPhase::Open;
        let (phase_start, tile_base, pending) = if open {
            (meta.phase_start, meta.tile_base, meta.pending_rows as usize)
        } else {
            (start_pos, start_pos, 0)
        };
        if open {
            if tile_base < phase_start
                || tile_base > start_pos
                || start_pos - tile_base != pending as u64
                || pending >= micro
                || start_pos >= next_boundary(tile_base, micro64)
            {
                return Err(format!(
                    "inconsistent open stream: phase_start {phase_start} tile_base {tile_base} \
                     pending {pending} position {start_pos}"
                ));
            }
            if meta.frame_base != phase_start.max(start_pos / commit64 * commit64) {
                return Err(format!(
                    "inconsistent open stream: frame_base {} (phase_start {phase_start}, position {start_pos})",
                    meta.frame_base
                ));
            }
        }
        let tile_end = next_boundary(tile_base, micro64);
        let head = pending > 0 || tile_base % micro64 != 0;
        let head_new = if head { rows.min((tile_end - start_pos) as usize) } else { 0 };
        let head_complete = head && start_pos + head_new as u64 == tile_end;
        let rest = rows - head_new;
        let bulk_complete = rest / micro * micro;
        let bulk_tail = rest - bulk_complete;
        let end = start_pos + rows as u64;
        let new_tile_base = if bulk_tail > 0 {
            end - bulk_tail as u64
        } else if head && !head_complete {
            tile_base
        } else {
            end
        };
        let new_meta = GdnStreamMeta {
            schema_version: GDN_STREAM_SCHEMA_VERSION,
            recipe_id: meta.recipe_id,
            phase: GdnStreamPhase::Open,
            position: end,
            phase_start,
            frame_base: phase_start.max(end / commit64 * commit64),
            tile_base: new_tile_base,
            pending_rows: (end - new_tile_base) as u32,
            owner_generation: meta.owner_generation,
        };
        Ok(Self {
            head,
            head_pending: pending,
            head_new,
            head_complete,
            bulk_complete,
            bulk_tail,
            meta: new_meta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_bytes_match_the_planes() {
        // FN 16/48, BF16 output: x 2.5 MiB, alpha+beta 24 KiB, q+k 512 KiB, v 768 KiB,
        // A 384 KiB, out 768 KiB, conv ring 120 KiB.
        assert_eq!(
            stream_workspace_bytes(16, 48, 2),
            2_621_440 + 24_576 + 524_288 + 786_432 + 393_216 + 786_432 + 122_880
        );
    }

    #[test]
    fn carry_bytes_match_design_table() {
        let g = GdnCarryGeometry { hk: 16, hv: 48, head_dim: 128, conv_hist_rows: 3, micro_rows: 64 };
        assert_eq!(g.total_bytes(), 5_914_624); // 5.640625 MiB
        let g16 = GdnCarryGeometry { micro_rows: 16, ..g };
        assert_eq!(g16.total_bytes(), 3_930_112); // 3.748046875 MiB
    }

    fn open_at(pos: u64) -> GdnStreamMeta {
        GdnStreamMeta { recipe_id: 7, ..GdnStreamMeta::sealed_at(pos, 3) }
    }

    /// Final metadata of an append sequence from a sealed start; head/bulk/tail
    /// row counts must always cover the call exactly.
    fn run(start: u64, parts: &[usize]) -> GdnStreamMeta {
        let mut meta = open_at(start);
        for &n in parts {
            let s = GdnAppendSchedule::new(&meta, meta.position, n, 64, 512).unwrap();
            assert_eq!(s.head_new + s.bulk_complete + s.bulk_tail, n);
            meta = s.meta;
        }
        meta
    }

    #[test]
    fn schedule_is_partition_independent() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        for start in [0u64, 1, 63, 64, 65, 511, 512, 513, 521, 45979] {
            for _ in 0..200 {
                let total = 1 + next(1400) as usize;
                let mut parts = Vec::new();
                let mut left = total;
                while left > 0 {
                    let n = if next(3) == 0 { 1 } else { 1 + next(left as u64 + 64) as usize % left };
                    parts.push(n);
                    left -= n;
                }
                let got = run(start, &parts);
                let whole = run(start, &[total]);
                assert_eq!(got.position, whole.position, "start {start} parts {parts:?}");
                assert_eq!(got.tile_base, whole.tile_base, "start {start} parts {parts:?}");
                assert_eq!(got.pending_rows, whole.pending_rows, "start {start} parts {parts:?}");
                assert_eq!(got.frame_base, whole.frame_base, "start {start} parts {parts:?}");
            }
        }
    }

    #[test]
    fn schedule_splits_head_bulk_tail() {
        // Sealed at an off-grid position: a clipped first tile of 64 - 25 = 39 rows.
        let s = GdnAppendSchedule::new(&open_at(25), 25, 300, 64, 512).unwrap();
        assert!(s.head && s.head_pending == 0 && s.head_new == 39 && s.head_complete);
        assert_eq!((s.bulk_complete, s.bulk_tail), (256, 5));
        assert_eq!((s.meta.tile_base, s.meta.pending_rows, s.meta.frame_base), (320, 5, 25));
        // Pending rows continue the same tile.
        let m = s.meta;
        let s2 = GdnAppendSchedule::new(&m, 325, 70, 64, 512).unwrap();
        assert!(s2.head && s2.head_pending == 5 && s2.head_new == 59 && s2.head_complete);
        assert_eq!((s2.bulk_complete, s2.bulk_tail), (0, 11));
        // Singleton inside a tile stays pending and keeps the tile base.
        let s3 = GdnAppendSchedule::new(&m, 325, 1, 64, 512).unwrap();
        assert!(s3.head && !s3.head_complete);
        assert_eq!((s3.meta.tile_base, s3.meta.pending_rows), (320, 6));
        // Aligned start: no head, whole tiles, seam crossing moves the frame base.
        let a = GdnAppendSchedule::new(&open_at(512), 512, 1024, 64, 512).unwrap();
        assert!(!a.head && a.bulk_complete == 1024 && a.bulk_tail == 0);
        assert_eq!((a.meta.tile_base, a.meta.frame_base, a.meta.pending_rows), (1536, 1536, 0));
    }

    #[test]
    fn schedule_rejects_inconsistent_meta() {
        let mut m = GdnAppendSchedule::new(&open_at(0), 0, 10, 64, 512).unwrap().meta;
        assert!(GdnAppendSchedule::new(&m, 11, 1, 64, 512).is_err());
        m.pending_rows = 11;
        assert!(GdnAppendSchedule::new(&m, 10, 1, 64, 512).is_err());
        assert!(GdnAppendSchedule::new(&open_at(0), 0, 0, 64, 512).is_err());
    }
}
