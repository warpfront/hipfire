// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Per-slot RDA/conv state, VMM latent caches, bounded prefill scratch. Owner: slice S.
//!
//! One [`RaonState`] is one slot. It owns, and nothing else does:
//!
//! * the fixed-size recurrent block: per RDA layer an FP32 `S[heads, V, K]`
//!   matrix plus the raw (pre-convolution) short-conv history, carved from one
//!   device slab together with the device-resident token count. Size is
//!   independent of `max_seq`; there is no ctx-sized rollback tape;
//! * one latent cache per MLA layer, row-major `[T, latent+tail]` BF16
//!   (`c512 || kpe64`, c stored after its RMS norm). The rows live in a
//!   `hip_bridge::VmmArena` (through `Gpu::alloc_vmm_tensor`) that reserves the
//!   virtual maximum and maps a small initial range; [`RaonState::ensure_capacity`]
//!   maps ahead, fallibly, before admission or graph capture. VMM owners are
//!   released only through `Gpu::free_tensor` (arena release/quarantine/
//!   retired-VA semantics), never pooled;
//! * an optional expanded K/V mirror (token-major, per MLA layer; BF16, or INT8 codes with
//!   per-(row, head) F16 scales when [`RaonStateDims::mirror_q8`]), also VMM-backed:
//!   virtual maximum reserved lazily, physical rows mapped ahead on demand up to a
//!   memory budget computed from the free device memory at slot admission (never
//!   `max_seq` unconditionally); growth never moves an address, so it never
//!   invalidates a captured graph's pointers.
//!
//! [`RaonPrefillScratch`] is a separate owner (per invocation, not per slot):
//! one exact device slab carved into every intermediate the forward needs for
//! at most `max_rows` rows, so decode (rows = 1) and bounded prefill chunks use
//! the same buffer set and a bucket switch is one allocation.
//!
//! Capture rules honoured here: every address handed out is stable for the life
//! of the owner; sequence length/positions/token ids are device buffers;
//! growth, mirror admission, reset and snapshot/restore are outside-capture
//! operations.

use hipfire_runtime::kv_backend::{KvChunkPlan, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES};
use rdna_compute::latent_attention_tiled::q8_plane_row_bytes;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Alignment of every sub-allocation carved from a slab.
pub const SLAB_ALIGN: usize = 256;
/// Bytes per stored element of the conv history planes (F32 planes holding
/// BF16-rounded raw values; see [`RaonState::conv_q`]).
pub const CONV_ELEM_BYTES: usize = 4;
/// Bytes per latent-cache element (BF16).
pub const LATENT_ELEM_BYTES: usize = 2;
/// Rows mapped by the first VMM chunk of every latent cache.
pub const INITIAL_LATENT_ROWS: usize = 1024;
/// Free device bytes that must remain after a state or scratch admission.
pub const ADMISSION_HEADROOM_BYTES: usize = 256 << 20;
/// Free device bytes that must remain after a latent-cache growth.
pub const GROWTH_HEADROOM_BYTES: usize = 64 << 20;
/// Rows per growth step of the expanded K/V mirror's VMM planes.
pub const MIRROR_CHUNK_ROWS: usize = 1024;
/// Free device bytes held back from the slot-admission mirror estimate for the
/// scratch, work and graph buffers that do not exist yet (on top of
/// [`ADMISSION_HEADROOM_BYTES`]); a conservative guess, replaced by
/// [`RaonState::fit_mirror_budget`] once the bundle is assembled.
pub const MIRROR_RESERVE_BYTES: usize = 2 << 30;
/// Free device bytes held back from the assembled-bundle mirror budget (on top
/// of [`ADMISSION_HEADROOM_BYTES`]) for what is still allocated lazily: graph
/// instantiation, kernel module loads, dispatch-pool growth and other tenants.
/// Scratch, work, packed weights and the split-KV workspace already exist when
/// the budget is fitted, so they are measured, not reserved.
pub const MIRROR_POST_ASSEMBLY_RESERVE_BYTES: usize = 512 << 20;
/// Device bytes the lazily growing expanded mirror must always leave really free
/// (measured cliff: a card filled to 100% by lazy growth slows ~5-6x).
pub const MIRROR_REAL_FLOOR_BYTES: usize = 1 << 30;
/// Largest `max_rows` accepted by [`RaonPrefillScratch::new`].
pub const MAX_SCRATCH_ROWS: usize = 4096;
/// Rows of one bounded decompression tile (`mla_kv_tile`).
pub const MLA_TILE_ROWS: usize = 512;

fn align_up(v: usize, a: usize) -> Result<usize, String> {
    v.checked_add(a - 1)
        .map(|x| x / a * a)
        .ok_or_else(|| "raon state: byte size overflow".to_string())
}

fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "raon state: byte size overflow".to_string())
}

fn hip<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("raon state: {what}: {e:?}"))
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dimensions
// ─────────────────────────────────────────────────────────────────────────────

/// Every dimension the state and scratch owners depend on. `RaonConfig::state_dims`
/// fills this; [`RaonStateDims::raon`] is the production checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaonStateDims {
    pub hidden: usize,
    pub n_streams: usize,
    pub n_layers: usize,
    /// Number of RDA layers (recurrent blocks).
    pub n_rda: usize,
    /// Number of MLA layers (latent caches).
    pub n_mla: usize,
    pub heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    /// Conv channels per RDA layer: `q|k|v` concatenated, `heads*(2*k_dim+v_dim)`.
    pub conv_channels: usize,
    /// Raw history rows kept per channel (`kernel-1`).
    pub conv_hist: usize,
    /// Normalized latent width (512).
    pub latent: usize,
    /// Un-normalized shared key tail width (64).
    pub tail: usize,
    pub vocab: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    /// Physical K of the routed down projection (logical 1088 padded to 1280).
    pub moe_down_k: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub shared_inter: usize,
    /// MLA query bottleneck width (576).
    pub q_lora: usize,
    /// MLA `Wq_b` output width (`heads * mla_qk_head`).
    pub mla_q_out: usize,
    /// MLA `Wkv_b` output width (`heads * (nope + v)`).
    pub mla_kv_b_out: usize,
    /// Per-head query/key width (`nope + tail`, 192).
    pub mla_qk_head: usize,
    /// Per-head value width (128).
    pub mla_v_head: usize,
    /// mHC selector controls per site (17).
    pub mhc_controls: usize,
    /// RDA decay bottleneck width (128).
    pub decay_bottleneck: usize,
    /// Expanded K/V mirror stored as INT8 codes + one F16 scale per (row, head,
    /// plane) instead of BF16 ([`crate::mla::MlaPolicy::mirror_q8`]).
    pub mirror_q8: bool,
}

impl RaonStateDims {
    /// Raon-1-mini-preview constants (PLAN §1, §3.5).
    pub const fn raon() -> Self {
        Self {
            hidden: 2560,
            n_streams: 4,
            n_layers: 25,
            n_rda: 18,
            n_mla: 7,
            heads: 40,
            k_dim: 128,
            v_dim: 128,
            conv_channels: 3 * 5120,
            conv_hist: 3,
            latent: 512,
            tail: 64,
            vocab: 259_840,
            dense_inter: 12_032,
            moe_inter: 1088,
            moe_down_k: 1280,
            n_experts: 128,
            top_k: 6,
            shared_inter: 1088,
            q_lora: 576,
            mla_q_out: 40 * 192,
            mla_kv_b_out: 40 * (128 + 128),
            mla_qk_head: 192,
            mla_v_head: 128,
            mhc_controls: 17,
            decay_bottleneck: 128,
            mirror_q8: true,
        }
    }

    /// Structural consistency check; every size derived below is nonzero and
    /// the cross-field identities of the architecture hold.
    pub fn validate(&self) -> Result<(), String> {
        let fields = [
            ("hidden", self.hidden),
            ("n_streams", self.n_streams),
            ("n_layers", self.n_layers),
            ("heads", self.heads),
            ("k_dim", self.k_dim),
            ("v_dim", self.v_dim),
            ("conv_channels", self.conv_channels),
            ("conv_hist", self.conv_hist),
            ("latent", self.latent),
            ("tail", self.tail),
            ("vocab", self.vocab),
            ("dense_inter", self.dense_inter),
            ("moe_inter", self.moe_inter),
            ("moe_down_k", self.moe_down_k),
            ("n_experts", self.n_experts),
            ("top_k", self.top_k),
            ("shared_inter", self.shared_inter),
            ("q_lora", self.q_lora),
            ("mla_q_out", self.mla_q_out),
            ("mla_kv_b_out", self.mla_kv_b_out),
            ("mla_qk_head", self.mla_qk_head),
            ("mla_v_head", self.mla_v_head),
            ("mhc_controls", self.mhc_controls),
            ("decay_bottleneck", self.decay_bottleneck),
        ];
        for (name, v) in fields {
            if v == 0 {
                return Err(format!("raon state dims: {name} must be nonzero"));
            }
        }
        if self.n_rda + self.n_mla != self.n_layers {
            return Err(format!(
                "raon state dims: n_rda {} + n_mla {} != n_layers {}",
                self.n_rda, self.n_mla, self.n_layers
            ));
        }
        if self.conv_channels != self.heads * (2 * self.k_dim + self.v_dim) {
            return Err(format!(
                "raon state dims: conv_channels {} != heads*(2*k+v) = {}",
                self.conv_channels,
                self.heads * (2 * self.k_dim + self.v_dim)
            ));
        }
        if self.mla_q_out != self.heads * self.mla_qk_head {
            return Err("raon state dims: mla_q_out != heads*mla_qk_head".into());
        }
        if self.mla_qk_head <= self.tail
            || self.mla_kv_b_out != self.heads * (self.mla_qk_head - self.tail + self.mla_v_head)
        {
            return Err("raon state dims: mla_kv_b_out != heads*(nope+v)".into());
        }
        if self.moe_down_k < self.moe_inter {
            return Err("raon state dims: moe_down_k < moe_inter".into());
        }
        if self.top_k > self.n_experts {
            return Err("raon state dims: top_k > n_experts".into());
        }
        if self.mirror_q8 {
            // Kernel contract of the Q8 mirror (flash prefill K staging, 16-byte rows).
            let nope = self.mla_qk_head - self.tail;
            let rb = |w: usize| q8_plane_row_bytes(self.heads, w);
            if nope % 16 != 0
                || !(nope / 8).is_power_of_two()
                || self.mla_v_head % 16 != 0
                || rb(nope) % 16 != 0
                || rb(self.mla_v_head) % 16 != 0
            {
                return Err(format!(
                    "raon state dims: the Q8 mirror needs nope {nope} and v {} multiples of 16 \
                     (nope/8 a power of two) and 16-byte plane rows; set HIPFIRE_RAON_MIRROR_Q8=0",
                    self.mla_v_head
                ));
            }
        }
        Ok(())
    }

    /// Elements in one RDA layer's `S[heads, V, K]`.
    pub fn rda_state_elems(&self) -> usize {
        self.heads * self.v_dim * self.k_dim
    }

    /// Elements in one RDA layer's conv history `[conv_channels, conv_hist]`.
    pub fn conv_elems(&self) -> usize {
        self.conv_channels * self.conv_hist
    }

    /// Elements of the `[q, k, v]` conv history planes of one RDA layer
    /// (`conv_hist * heads * {k, k, v}`).
    pub fn conv_plane_elems(&self) -> [usize; 3] {
        [
            self.conv_hist * self.heads * self.k_dim,
            self.conv_hist * self.heads * self.k_dim,
            self.conv_hist * self.heads * self.v_dim,
        ]
    }

    /// Latent-cache row width in elements (`latent + tail`).
    pub fn latent_row_elems(&self) -> usize {
        self.latent + self.tail
    }

    /// Latent-cache row width in bytes.
    pub fn latent_row_bytes(&self) -> usize {
        self.latent_row_elems() * LATENT_ELEM_BYTES
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ledger
// ─────────────────────────────────────────────────────────────────────────────

/// One owner's footprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEntry {
    pub owner: String,
    /// Bytes backed by physical memory right now.
    pub physical_bytes: usize,
    /// Reserved virtual extent (equals `physical_bytes` for non-VMM owners).
    pub virtual_bytes: usize,
}

/// Per-owner byte accounting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateLedger {
    pub entries: Vec<LedgerEntry>,
}

impl StateLedger {
    pub fn physical_total(&self) -> usize {
        self.entries.iter().map(|e| e.physical_bytes).sum()
    }
    pub fn virtual_total(&self) -> usize {
        self.entries.iter().map(|e| e.virtual_bytes).sum()
    }
    /// Sum of physical bytes of owners whose name starts with `prefix`.
    pub fn physical_with_prefix(&self, prefix: &str) -> usize {
        self.entries
            .iter()
            .filter(|e| e.owner.starts_with(prefix))
            .map(|e| e.physical_bytes)
            .sum()
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "physical_total": self.physical_total(),
            "virtual_total": self.virtual_total(),
            "entries": self.entries.iter().map(|e| serde_json::json!({
                "owner": e.owner,
                "physical_bytes": e.physical_bytes,
                "virtual_bytes": e.virtual_bytes,
            })).collect::<Vec<_>>(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte budget (pure arithmetic; CPU-testable)
// ─────────────────────────────────────────────────────────────────────────────

/// Storage class of the latent caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentBackend {
    /// Virtual maximum reserved, physical chunks mapped on demand.
    Vmm,
    /// Whole `max_seq` committed up front (platforms without certified VMM).
    Dense,
}

impl LatentBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vmm => "vmm",
            Self::Dense => "dense",
        }
    }
}

/// Fixed carve-out of the persistent slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistLayout {
    pub rda_off: usize,
    pub rda_stride: usize,
    pub conv_off: usize,
    pub conv_stride: usize,
    pub seq_off: usize,
    pub seq_bytes: usize,
    pub total: usize,
}

/// Aligned byte size of each of the three conv history planes of one layer.
pub fn conv_plane_bytes(d: &RaonStateDims) -> Result<[usize; 3], String> {
    let e = d.conv_plane_elems();
    Ok([
        align_up(mul(e[0], CONV_ELEM_BYTES)?, SLAB_ALIGN)?,
        align_up(mul(e[1], CONV_ELEM_BYTES)?, SLAB_ALIGN)?,
        align_up(mul(e[2], CONV_ELEM_BYTES)?, SLAB_ALIGN)?,
    ])
}

pub fn persist_layout(d: &RaonStateDims) -> Result<PersistLayout, String> {
    let rda_stride = align_up(mul(d.rda_state_elems(), 4)?, SLAB_ALIGN)?;
    let conv_stride = conv_plane_bytes(d)?.iter().sum::<usize>();
    let rda_off = 0;
    let conv_off = rda_off + mul(rda_stride, d.n_rda)?;
    let seq_off = conv_off + mul(conv_stride, d.n_rda)?;
    let seq_bytes = SLAB_ALIGN;
    Ok(PersistLayout {
        rda_off,
        rda_stride,
        conv_off,
        conv_stride,
        seq_off,
        seq_bytes,
        total: seq_off + seq_bytes,
    })
}

/// Byte requirements of one slot at a given `max_seq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateBudget {
    pub persist: PersistLayout,
    /// Bytes of one latent row (all of one layer's `latent+tail` BF16).
    pub latent_row_bytes: usize,
    /// Reserved virtual bytes per latent cache.
    pub latent_reserved_per_layer: usize,
    /// Physical bytes per latent cache while only `initial_rows` are mapped.
    pub latent_initial_per_layer: usize,
    /// Physical bytes per latent cache once `max_seq` rows are mapped.
    pub latent_full_per_layer: usize,
}

impl StateBudget {
    /// Physical bytes right after construction.
    pub fn initial_physical(&self, d: &RaonStateDims) -> usize {
        self.persist.total + self.latent_initial_per_layer * d.n_mla
    }
    /// Physical bytes when every latent cache holds `max_seq` rows.
    pub fn full_physical(&self, d: &RaonStateDims) -> usize {
        self.persist.total + self.latent_full_per_layer * d.n_mla
    }
}

/// Plan the latent cache growth chunking for `granularity`.
pub fn latent_chunk_plan(
    d: &RaonStateDims,
    max_seq: usize,
    granularity: usize,
) -> Result<KvChunkPlan, String> {
    KvChunkPlan::new(
        d.latent_row_bytes(),
        max_seq,
        INITIAL_LATENT_ROWS.min(max_seq),
        granularity,
        DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
    )
    .map_err(|e| format!("raon state: latent VMM growth plan: {e}"))
}

pub fn state_budget(
    d: &RaonStateDims,
    max_seq: usize,
    backend: LatentBackend,
    granularity: usize,
) -> Result<StateBudget, String> {
    d.validate()?;
    if max_seq == 0 {
        return Err("raon state: max_seq must be nonzero".into());
    }
    let persist = persist_layout(d)?;
    let row = d.latent_row_bytes();
    let dense = mul(row, max_seq)?;
    let (reserved, initial, full) = match backend {
        LatentBackend::Dense => (dense, dense, dense),
        LatentBackend::Vmm => {
            let plan = latent_chunk_plan(d, max_seq, granularity)?;
            let initial = plan
                .mapped_bytes_for_tokens(INITIAL_LATENT_ROWS.min(max_seq))
                .map_err(|e| format!("raon state: initial latent map: {e}"))?;
            let full = plan
                .mapped_bytes_for_tokens(max_seq)
                .map_err(|e| format!("raon state: full latent map: {e}"))?;
            (plan.reserve_bytes(), initial, full)
        }
    };
    Ok(StateBudget {
        persist,
        latent_row_bytes: row,
        latent_reserved_per_layer: reserved,
        latent_initial_per_layer: initial,
        latent_full_per_layer: full,
    })
}

/// Why VMM latent caches cannot be used on `gpu`, or `None` when they can.
pub fn vmm_refusal(gpu: &Gpu) -> Option<String> {
    if cfg!(windows) {
        return Some("Windows maps whole VMM reservations; latent cache stays dense".into());
    }
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201") {
        return Some(format!(
            "latent VMM caches are certified on gfx1100/gfx1151/gfx1201, not {}",
            gpu.arch
        ));
    }
    if let Some(reason) = hipfire_config::devices::vmm_kv_platform_refusal() {
        return Some(reason);
    }
    if gpu.vmm_recommended_granularity().is_err() {
        return Some("HIP VMM symbols/granularity unavailable".into());
    }
    None
}

/// Backend [`RaonState::new`] selects on `gpu`.
pub fn auto_backend(gpu: &Gpu) -> LatentBackend {
    if vmm_refusal(gpu).is_none() {
        LatentBackend::Vmm
    } else {
        LatentBackend::Dense
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

fn zero_buf(gpu: &Gpu, buf: &hip_bridge::DeviceBuffer, bytes: usize) -> hip_bridge::HipResult<()> {
    match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.memset_async(buf, 0, bytes, stream),
        None => gpu.hip.memset(buf, 0, bytes),
    }
}

fn copy_d2d(
    gpu: &Gpu,
    dst: &hip_bridge::DeviceBuffer,
    src: &hip_bridge::DeviceBuffer,
    bytes: usize,
) -> hip_bridge::HipResult<()> {
    match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.memcpy_dtod_async_at(dst, 0, src, 0, bytes, stream),
        None => gpu.hip.memcpy_dtod(dst, src, bytes),
    }
}

fn check_free(gpu: &Gpu, need: usize, headroom: usize, what: &str) -> Result<(), String> {
    // Planned capacity (total - own - reserve), not the instantaneous free
    // figure: a transient foreign allocation must not flip the route. The
    // allocation that follows is the real check.
    let cap = hip(gpu.planned_capacity(), "query device memory")?;
    let (free, total) = (cap.bytes, cap.total);
    let required = need
        .checked_add(headroom)
        .ok_or_else(|| format!("raon state: {what}: byte overflow"))?;
    if required > free {
        return Err(format!(
            "raon state: refused {what}: needs {:.2} MiB + {:.2} MiB headroom but only {:.2} MiB of {:.2} MiB device memory is free",
            mib(need),
            mib(headroom),
            mib(free),
            mib(total)
        ));
    }
    Ok(())
}

fn slab_view(
    slab: &GpuTensor,
    off: usize,
    bytes: usize,
    dtype: DType,
    shape: Vec<usize>,
) -> GpuTensor {
    GpuTensor {
        buf: slab.buf.byte_view(off, bytes),
        shape,
        dtype,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expanded mirror
// ─────────────────────────────────────────────────────────────────────────────

/// Optional expanded K/V rows, token-major with the shared key tail left in the
/// latent cache (the layout the latent attention op reads): per MLA layer a
/// `knope` plane (`[rows, heads, nope]`) and a `vmir` plane (`[rows, heads, v]`).
///
/// * BF16 mirror (`q8 == false`): BF16 planes of exactly that shape.
/// * Q8 mirror (`q8 == true`, [`RaonStateDims::mirror_q8`]): `Raw` byte planes
///   whose row is `heads * width` INT8 codes followed by `heads` F16 scales (one
///   per (row, head); `heads * (width + 2)` bytes, see
///   [`rdna_compute::latent_attention_tiled::q8_plane_row_bytes`]). 72,800 B per
///   token over all layers instead of 143,360.
///
/// With the VMM backend every plane reserves `max_seq` rows of virtual space and
/// maps physical chunks on demand ([`RaonState::ensure_mirror_rows`]); addresses
/// never change, so growth never invalidates a captured graph's pointers. The
/// dense fallback owns exactly `rows` rows and reallocates to grow (dropping
/// the valid-row mark). Row `t` is valid for `t < RaonState::mirror_valid_rows`.
pub struct ExpandedMirror {
    /// Rows every plane is physically backed for (the attention kernels'
    /// clamp; a baked kernel argument, so a change must rebuild a graph).
    pub rows: usize,
    pub knope: Vec<GpuTensor>,
    pub vmir: Vec<GpuTensor>,
    /// The planes are Q8 byte planes (see above).
    pub q8: bool,
    /// Growth plans of the K and V planes (`None` for dense planes).
    plans: Option<[KvChunkPlan; 2]>,
}

/// Bytes per token row of the K_nope and V planes of ONE layer.
pub fn mirror_plane_row_bytes(d: &RaonStateDims) -> [usize; 2] {
    let nope = d.mla_qk_head - d.tail;
    if d.mirror_q8 {
        [
            q8_plane_row_bytes(d.heads, nope),
            q8_plane_row_bytes(d.heads, d.mla_v_head),
        ]
    } else {
        [d.heads * nope * 2, d.heads * d.mla_v_head * 2]
    }
}

/// Exact bytes per token row of the mirror over all MLA layers
/// (BF16: `n_mla * heads * (nope + v) * 2` = 143,360 for the released geometry;
/// Q8: `n_mla * heads * (nope + v + 4)` = 72,800).
pub fn mirror_row_bytes(d: &RaonStateDims) -> usize {
    let [k, v] = mirror_plane_row_bytes(d);
    (k + v) * d.n_mla
}

/// Bytes of an expanded mirror of `rows` rows over all MLA layers.
pub fn expanded_mirror_bytes(d: &RaonStateDims, rows: usize) -> Result<usize, String> {
    mul(mirror_row_bytes(d), rows)
}

/// Rows of mirror that `budget_bytes` backs (floor).
pub fn mirror_rows_for_budget(d: &RaonStateDims, budget_bytes: usize) -> usize {
    budget_bytes / mirror_row_bytes(d).max(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// RaonState
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform q/k/v conv-history geometry of the persistent slab, see
/// [`RaonState::conv_slab_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvSlabLayout {
    pub conv_off: usize,
    /// Bytes between RDA layers' conv blocks.
    pub layer_stride: usize,
    /// Bytes between the q, k and v planes of one layer.
    pub plane_pitch: usize,
    /// Channels (F32 columns) per history row.
    pub cols: usize,
    /// History rows (`kernel - 1`).
    pub hist: usize,
}

/// Snapshot of the recurrent block (RDA matrices, conv history, device token
/// count) for graph warmup/capture: warm up, then [`RaonState::restore`].
/// Latent cache rows are not copied: rows at or beyond the restored length are
/// dead and are overwritten when the tokens are consumed.
pub struct RaonStateSnapshot {
    buf: GpuTensor,
    bytes: usize,
    n_tokens: usize,
}

impl RaonStateSnapshot {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }
    /// The snapshot bytes: a verbatim copy of the persistent slab
    /// ([`RaonState::persist_tensor`]), same offsets.
    pub fn tensor(&self) -> &GpuTensor {
        &self.buf
    }
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        hip(gpu.free_tensor(self.buf), "free snapshot")
    }
}

/// One slot of Raon sequence state.
pub struct RaonState {
    dims: RaonStateDims,
    max_seq: usize,
    backend: LatentBackend,
    plan: Option<KvChunkPlan>,
    persist: GpuTensor,
    layout: PersistLayout,
    /// Per RDA layer (ordinal): FP32 `S[heads, V, K]`, value-major, key
    /// contiguous. Shape `[heads, v_dim, k_dim]`. Views of the persistent slab.
    pub rda: Vec<GpuTensor>,
    /// Per RDA layer (ordinal): raw pre-convolution history of the q, k and v
    /// channels, three separate F32 tensors holding BF16-rounded values,
    /// tap-major `[conv_hist, heads*k]` / `[conv_hist, heads*k]` /
    /// `[conv_hist, heads*v]` (element `j*C + c`, `j = 0` oldest). Zero-init.
    pub conv_q: Vec<GpuTensor>,
    pub conv_k: Vec<GpuTensor>,
    pub conv_v: Vec<GpuTensor>,
    /// Device-resident i32 count of tokens consumed (valid latent rows and the
    /// position of the next token). 4 bytes; capture-safe.
    pub seq_len: GpuTensor,
    /// Per MLA layer (ordinal): latent rows `[max_seq, latent+tail]` BF16.
    /// Only the first [`RaonState::capacity_rows`] rows are mapped.
    pub mla_cache: Vec<GpuTensor>,
    mirror: Option<ExpandedMirror>,
    /// Most rows the mirror may ever back on this device (admission-time
    /// budget, `<= max_seq`).
    mirror_budget_rows: usize,
    /// Mirror rows `[0, mirror_valid_rows)` equal the expansion of the latent
    /// cache rows with the same index.
    mirror_valid_rows: usize,
    n_tokens: usize,
    capacity_rows: usize,
}

impl RaonState {
    /// Allocate one slot with the auto-selected latent backend
    /// ([`auto_backend`]). Refuses cleanly when the device cannot hold the
    /// fixed state plus `max_seq` latent rows.
    pub fn new(gpu: &mut Gpu, dims: &RaonStateDims, max_seq: usize) -> Result<Self, String> {
        let backend = auto_backend(gpu);
        Self::new_with_backend(gpu, dims, max_seq, backend)
    }

    pub fn new_with_backend(
        gpu: &mut Gpu,
        dims: &RaonStateDims,
        max_seq: usize,
        backend: LatentBackend,
    ) -> Result<Self, String> {
        dims.validate()?;
        if backend == LatentBackend::Vmm {
            if let Some(reason) = vmm_refusal(gpu) {
                return Err(format!("raon state: VMM latent cache refused: {reason}"));
            }
        }
        let granularity = match backend {
            LatentBackend::Vmm => hip(gpu.vmm_recommended_granularity(), "VMM granularity")?,
            LatentBackend::Dense => 1,
        };
        let budget = state_budget(dims, max_seq, backend, granularity)?;
        // Physical need at the full horizon: the model is already loaded, so
        // `free` is what this slot can ever get. A context the device cannot
        // back is refused here rather than at the first long request.
        let full = budget.full_physical(dims);
        check_free(
            gpu,
            full,
            ADMISSION_HEADROOM_BYTES,
            &format!(
                "slot state for max_seq {max_seq} ({} latent, {:.2} MiB at full context)",
                backend.as_str(),
                mib(full)
            ),
        )?;

        let layout = budget.persist;
        let persist = hip(
            gpu.alloc_tensor_exact(&[layout.total], DType::Raw),
            "alloc persistent slab",
        )?;
        if let Err(e) = zero_buf(gpu, &persist.buf, layout.total) {
            let _ = gpu.free_tensor(persist);
            return Err(format!("raon state: zero persistent slab: {e:?}"));
        }

        let mut caches: Vec<GpuTensor> = Vec::with_capacity(dims.n_mla);
        let mut plan: Option<KvChunkPlan> = None;
        let build = (|| -> Result<(), String> {
            for layer in 0..dims.n_mla {
                match backend {
                    LatentBackend::Dense => {
                        let t = hip(
                            gpu.zeros(&[max_seq, dims.latent_row_elems()], DType::BF16),
                            "alloc dense latent cache",
                        )?;
                        caches.push(t);
                    }
                    LatentBackend::Vmm => {
                        // SAFETY: nothing is mapped yet; every access below
                        // stays within the mapped prefix reported by
                        // `vmm_mapped_bytes`, grown by `ensure_capacity`.
                        let mut t = hip(
                            unsafe {
                                gpu.alloc_vmm_tensor(
                                    &[max_seq, dims.latent_row_elems()],
                                    DType::BF16,
                                    0,
                                    &[],
                                )
                            },
                            "reserve latent VMM cache",
                        )?;
                        let actual = gpu
                            .vmm_granularity(&t)
                            .ok_or_else(|| "raon state: VMM owner has no granularity".to_string());
                        let actual = match actual {
                            Ok(g) => g,
                            Err(e) => {
                                caches.push(t);
                                return Err(e);
                            }
                        };
                        let layer_plan = match latent_chunk_plan(dims, max_seq, actual) {
                            Ok(p) => p,
                            Err(e) => {
                                caches.push(t);
                                return Err(e);
                            }
                        };
                        if let Some(first) = plan {
                            if first.granularity() != layer_plan.granularity() {
                                caches.push(t);
                                return Err(format!(
                                    "raon state: latent cache {layer} granularity {} differs from {}",
                                    layer_plan.granularity(),
                                    first.granularity()
                                ));
                            }
                        } else {
                            plan = Some(layer_plan);
                        }
                        let initial = match layer_plan
                            .mapped_bytes_for_tokens(INITIAL_LATENT_ROWS.min(max_seq))
                        {
                            Ok(b) => b,
                            Err(e) => {
                                caches.push(t);
                                return Err(format!("raon state: initial latent map: {e}"));
                            }
                        };
                        if let Err(e) = gpu.grow_vmm_tensor(&mut t, initial, &[]) {
                            caches.push(t);
                            return Err(format!("raon state: map initial latent range: {e:?}"));
                        }
                        caches.push(t);
                    }
                }
            }
            Ok(())
        })();
        if let Err(e) = build {
            for t in caches {
                let _ = gpu.free_tensor(t);
            }
            let _ = gpu.free_tensor(persist);
            return Err(e);
        }

        let mut rda = Vec::with_capacity(dims.n_rda);
        let mut conv_q = Vec::with_capacity(dims.n_rda);
        let mut conv_k = Vec::with_capacity(dims.n_rda);
        let mut conv_v = Vec::with_capacity(dims.n_rda);
        let plane_bytes = conv_plane_bytes(dims)?;
        let plane_elems = dims.conv_plane_elems();
        let plane_cols = [
            dims.heads * dims.k_dim,
            dims.heads * dims.k_dim,
            dims.heads * dims.v_dim,
        ];
        for i in 0..dims.n_rda {
            rda.push(slab_view(
                &persist,
                layout.rda_off + i * layout.rda_stride,
                dims.rda_state_elems() * 4,
                DType::F32,
                vec![dims.heads, dims.v_dim, dims.k_dim],
            ));
            let base = layout.conv_off + i * layout.conv_stride;
            let mut planes = [(); 3].map(|_| None);
            let mut off = base;
            for p in 0..3 {
                planes[p] = Some(slab_view(
                    &persist,
                    off,
                    plane_elems[p] * CONV_ELEM_BYTES,
                    DType::F32,
                    vec![dims.conv_hist, plane_cols[p]],
                ));
                off += plane_bytes[p];
            }
            let [q, k, v] = planes;
            conv_q.push(q.expect("plane built"));
            conv_k.push(k.expect("plane built"));
            conv_v.push(v.expect("plane built"));
        }
        let seq_len = slab_view(&persist, layout.seq_off, 4, DType::Raw, vec![4]);

        let mut state = Self {
            dims: dims.clone(),
            max_seq,
            backend,
            plan,
            persist,
            layout,
            rda,
            conv_q,
            conv_k,
            conv_v,
            seq_len,
            mla_cache: caches,
            mirror: None,
            mirror_budget_rows: 0,
            mirror_valid_rows: 0,
            n_tokens: 0,
            capacity_rows: 0,
        };
        state.capacity_rows = state.query_capacity_rows(gpu);
        // Mirror budget: what is free now, minus the latent growth this slot
        // may still map up to `max_seq`, minus the fixed reserves.
        let pending_latent = budget
            .full_physical(dims)
            .saturating_sub(budget.initial_physical(dims));
        let free = hip(gpu.planned_capacity(), "query device memory")?.bytes;
        let usable = free
            .saturating_sub(pending_latent)
            .saturating_sub(ADMISSION_HEADROOM_BYTES)
            .saturating_sub(MIRROR_RESERVE_BYTES);
        state.mirror_budget_rows = mirror_rows_for_budget(dims, usable).min(max_seq);
        Ok(state)
    }

    pub fn dims(&self) -> &RaonStateDims {
        &self.dims
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn backend(&self) -> LatentBackend {
        self.backend
    }
    /// Tokens consumed by this slot (host mirror of `seq_len`).
    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }
    /// Rows every latent cache currently has physical backing for.
    pub fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }
    pub fn persist_layout(&self) -> &PersistLayout {
        &self.layout
    }
    pub fn expanded_mirror(&self) -> Option<&ExpandedMirror> {
        self.mirror.as_ref()
    }

    fn query_capacity_rows(&self, gpu: &Gpu) -> usize {
        match (self.backend, self.plan) {
            (LatentBackend::Vmm, Some(plan)) => self
                .mla_cache
                .iter()
                .map(|t| plan.token_capacity(gpu.vmm_mapped_bytes(t).unwrap_or(0)))
                .min()
                .unwrap_or(0),
            _ => self.max_seq,
        }
    }

    /// Map every latent cache through `rows` rows (map-ahead). Call before
    /// admitting a request and before graph capture; a failure leaves every
    /// cache usable at its previous capacity and never degrades into a write
    /// to unmapped memory. Returns whether physical backing grew.
    ///
    /// Refuses before mapping anything when the whole growth (plus
    /// [`GROWTH_HEADROOM_BYTES`]) exceeds free device memory.
    pub fn ensure_capacity(&mut self, gpu: &mut Gpu, rows: usize) -> Result<bool, String> {
        if rows > self.max_seq {
            return Err(format!(
                "raon state: request needs {rows} latent rows but this slot reserved {} (raise max_seq)",
                self.max_seq
            ));
        }
        let Some(plan) = self.plan else {
            return Ok(false);
        };
        let want = rows.max(1);
        if want <= self.capacity_rows {
            return Ok(false);
        }
        let mut total = 0usize;
        for (i, t) in self.mla_cache.iter().enumerate() {
            let mapped = gpu
                .vmm_mapped_bytes(t)
                .ok_or_else(|| format!("raon state: latent cache {i} is not a VMM owner"))?;
            if let Some(g) = plan
                .growth(mapped, want)
                .map_err(|e| format!("raon state: latent cache {i} growth: {e}"))?
            {
                total += g.size_bytes;
            }
        }
        check_free(
            gpu,
            total,
            GROWTH_HEADROOM_BYTES,
            &format!("latent growth to {want} rows"),
        )?;
        let mut result = Ok(());
        for (i, t) in self.mla_cache.iter_mut().enumerate() {
            let mapped = gpu
                .vmm_mapped_bytes(t)
                .ok_or_else(|| format!("raon state: latent cache {i} is not a VMM owner"))?;
            let growth = match plan.growth(mapped, want) {
                Ok(g) => g,
                Err(e) => {
                    result = Err(format!("raon state: latent cache {i} growth: {e}"));
                    break;
                }
            };
            if let Some(g) = growth {
                if let Err(e) = gpu.grow_vmm_tensor(t, g.size_bytes, &[]) {
                    result = Err(format!("raon state: map latent cache {i}: {e:?}"));
                    break;
                }
            }
        }
        self.capacity_rows = self.query_capacity_rows(gpu);
        result.map(|()| true)
    }

    /// Set the consumed-token count on host and device (outside capture).
    pub fn set_n_tokens(&mut self, gpu: &mut Gpu, n: usize) -> Result<(), String> {
        if n > self.max_seq {
            return Err(format!(
                "raon state: n_tokens {n} exceeds reserved max_seq {}",
                self.max_seq
            ));
        }
        let v = i32::try_from(n).map_err(|_| "raon state: n_tokens exceeds i32".to_string())?;
        hip(
            gpu.hip.memcpy_htod(&self.seq_len.buf, &v.to_le_bytes()),
            "write seq_len",
        )?;
        self.n_tokens = n;
        Ok(())
    }

    /// Record `n` newly consumed tokens on the host mirror only (the device
    /// count is advanced by the forward's own kernels or by `set_n_tokens`).
    pub fn note_host_tokens(&mut self, n: usize) {
        self.n_tokens += n;
    }

    /// Start a cold request: zero recurrent state, conv history and the device
    /// length. Latent mappings, addresses and graph pointers are untouched;
    /// rows past length 0 are unreachable until rewritten.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let r = zero_buf(gpu, &self.persist.buf, self.layout.total);
        let r = hipfire_runtime::reset_core::note_hip_result(r, "raon::RaonState::reset");
        hip(r, "reset recurrent state")?;
        self.n_tokens = 0;
        self.mirror_valid_rows = 0;
        Ok(())
    }

    /// Zero every mapped latent row (debug/oracle scrub; not part of `reset`).
    pub fn scrub_latent(&self, gpu: &Gpu) -> Result<(), String> {
        for t in &self.mla_cache {
            let bytes = match self.backend {
                LatentBackend::Vmm => gpu.vmm_mapped_bytes(t).unwrap_or(0),
                LatentBackend::Dense => t.buf.size(),
            };
            if bytes > 0 {
                hip(zero_buf(gpu, &t.buf, bytes), "scrub latent cache")?;
            }
        }
        Ok(())
    }

    /// Allocate a snapshot buffer sized for this state's recurrent block.
    pub fn snapshot_alloc(&self, gpu: &mut Gpu) -> Result<RaonStateSnapshot, String> {
        let bytes = self.layout.total;
        let buf = hip(
            gpu.alloc_tensor_exact(&[bytes], DType::Raw),
            "alloc state snapshot",
        )?;
        Ok(RaonStateSnapshot {
            buf,
            bytes,
            n_tokens: 0,
        })
    }

    /// Copy RDA matrices, conv history and device length into `snap`.
    pub fn snapshot_into(&self, gpu: &Gpu, snap: &mut RaonStateSnapshot) -> Result<(), String> {
        if snap.bytes != self.layout.total {
            return Err("raon state: snapshot buffer was sized for different dims".into());
        }
        let r = copy_d2d(gpu, &snap.buf.buf, &self.persist.buf, snap.bytes);
        hip(
            hipfire_runtime::reset_core::note_hip_result(r, "raon::RaonState::snapshot"),
            "snapshot",
        )?;
        snap.n_tokens = self.n_tokens;
        Ok(())
    }

    /// Restore the recurrent block and length from `snap`. Latent rows at or
    /// beyond the restored length are dead, so the expanded mirror's valid-row
    /// watermark drops to the restored length too (it never exceeds
    /// `n_tokens`); the next forward re-expands anything it needs.
    pub fn restore(&mut self, gpu: &Gpu, snap: &RaonStateSnapshot) -> Result<(), String> {
        if snap.bytes != self.layout.total {
            return Err("raon state: snapshot was taken with different dims".into());
        }
        let r = copy_d2d(gpu, &self.persist.buf, &snap.buf.buf, snap.bytes);
        hip(
            hipfire_runtime::reset_core::note_hip_result(r, "raon::RaonState::restore"),
            "restore",
        )?;
        self.n_tokens = snap.n_tokens;
        self.clamp_mirror_valid_rows(snap.n_tokens);
        Ok(())
    }

    /// The persistent slab (RDA matrices, conv histories, device `seq_len`).
    pub fn persist_tensor(&self) -> &GpuTensor {
        &self.persist
    }

    /// The persistent slab as one session-cache state part: the buffer and the
    /// bytes of it that are state (RDA matrices, conv histories, device
    /// `seq_len`), overwritten in place by every forward.
    pub fn persist_part(&self) -> (&hip_bridge::DeviceBuffer, usize) {
        (&self.persist.buf, self.layout.total)
    }

    /// Set the host token count after a session restore. The device `seq_len`
    /// came with the restored slab, so nothing is written to the device (use
    /// [`Self::set_n_tokens`] when the device count must change too).
    pub fn set_host_tokens(&mut self, n: usize) -> Result<(), String> {
        if n > self.max_seq {
            return Err(format!(
                "raon state: n_tokens {n} exceeds reserved max_seq {}",
                self.max_seq
            ));
        }
        self.n_tokens = n;
        Ok(())
    }

    /// Uniform q/k/v conv-history layout of the slab (identical in any
    /// [`RaonStateSnapshot`] of it): layer `ri`'s plane `p` (0 q, 1 k, 2 v)
    /// starts at `conv_off + ri * layer_stride + p * plane_pitch` bytes and is
    /// `[conv_hist, cols]` F32, oldest -> newest. The three planes share one
    /// pitch and width (`heads*k_dim == heads*v_dim`), so one pitched copy
    /// moves all three; any other geometry is refused.
    pub fn conv_slab_layout(&self) -> Result<ConvSlabLayout, String> {
        let d = &self.dims;
        let planes = conv_plane_bytes(d)?;
        let elems = d.conv_plane_elems();
        if planes[0] != planes[1]
            || planes[1] != planes[2]
            || elems[0] != elems[1]
            || elems[1] != elems[2]
        {
            return Err("raon state: q/k/v conv history planes differ in size".into());
        }
        Ok(ConvSlabLayout {
            conv_off: self.layout.conv_off,
            layer_stride: self.layout.conv_stride,
            plane_pitch: planes[0],
            cols: elems[0] / d.conv_hist,
            hist: d.conv_hist,
        })
    }

    /// Download the whole persistent slab (RDA matrices, conv histories, device
    /// `seq_len`, alignment padding) after draining the device. Oracle use.
    pub fn download_persist(&self, gpu: &Gpu) -> Result<Vec<u8>, String> {
        hip(
            gpu.hip.device_synchronize(),
            "synchronize before slab download",
        )?;
        let mut out = vec![0u8; self.layout.total];
        hip(
            gpu.hip.memcpy_dtoh(&mut out, &self.persist.buf),
            "download persistent slab",
        )?;
        Ok(out)
    }

    /// Most rows the decompressed (mirror) MLA route can serve on this
    /// device: fixed at slot admission from the free device memory.
    pub fn mirror_budget_rows(&self) -> usize {
        self.mirror_budget_rows
    }

    /// Device bytes this slot may still map before it holds `through` tokens:
    /// the unmapped latent growth of every MLA layer (VMM backend; the dense
    /// backend committed `max_seq` rows up front) plus the expanded mirror's
    /// growth up to `min(through, mirror_budget_rows)` rows, rounded up to
    /// the mirror's growth chunk. An upper bound: a context that ends up on
    /// the absorbed route backs no more mirror than this.
    pub fn context_growth_bytes(&self, gpu: &Gpu, through: usize) -> u64 {
        let through = through.min(self.max_seq);
        let latent: usize = match (self.backend, self.plan) {
            (LatentBackend::Vmm, Some(plan)) => {
                let want = plan.mapped_bytes_for_tokens(through).unwrap_or(0);
                self.mla_cache
                    .iter()
                    .map(|t| want.saturating_sub(gpu.vmm_mapped_bytes(t).unwrap_or(0)))
                    .sum()
            }
            _ => 0,
        };
        let mirror = if self.dims.n_mla == 0 {
            0
        } else {
            let rows = through
                .min(self.mirror_budget_rows)
                .next_multiple_of(MIRROR_CHUNK_ROWS)
                .min(self.mirror_budget_rows);
            mirror_row_bytes(&self.dims)
                .saturating_mul(rows.saturating_sub(self.mirror_backed_rows()))
        };
        (latent as u64).saturating_add(mirror as u64)
    }

    /// Latent-cache bytes this slot may still map up to `max_seq`.
    fn pending_latent_bytes(&self, gpu: &Gpu) -> usize {
        match (self.backend, self.plan) {
            (LatentBackend::Vmm, Some(plan)) => {
                let full = plan.mapped_bytes_for_tokens(self.max_seq).unwrap_or(0);
                self.mla_cache
                    .iter()
                    .map(|t| full.saturating_sub(gpu.vmm_mapped_bytes(t).unwrap_or(0)))
                    .sum()
            }
            _ => 0,
        }
    }

    /// Re-derive the mirror budget from the memory that is actually free now:
    /// call once the bundle's scratch, work buffers and packed weights exist.
    /// Budget = rows already backed + what free memory still backs, after the
    /// latent growth to `max_seq`, [`ADMISSION_HEADROOM_BYTES`] and
    /// [`MIRROR_POST_ASSEMBLY_RESERVE_BYTES`]. Returns the new budget in rows.
    ///
    /// Format selection (capacity-based): while no mirror is backed yet and the Q8
    /// mirror is allowed ([`RaonStateDims::mirror_q8`]), the BF16 mirror is kept
    /// whenever its budget covers the whole reserved context (`max_seq`; it is
    /// 1-2% faster), and Q8 is chosen otherwise (it backs about twice the rows).
    /// The choice is made here, once, before any mirror memory exists.
    pub fn fit_mirror_budget(&mut self, gpu: &Gpu) -> Result<usize, String> {
        let planned = self.mirror_available_bytes(gpu)?;
        // Everything non-lazy is allocated by now. The planned figure ignores a
        // foreign holder, so a mirror that grows lazily into physically absent
        // memory would fill the card (measured: VRAM 100% full -> pp4096 5.8k ->
        // 1.0k, decode 73 -> 12 tok/s at 3.3 GiB foreign). Cap the growth by what
        // the device can really still back, less a fixed floor.
        let real = hip(gpu.device_mem_info(), "query device memory")?.0;
        let real_cap = real
            .saturating_sub(self.pending_latent_bytes(gpu))
            .saturating_sub(MIRROR_REAL_FLOOR_BYTES);
        let usable = planned.min(real_cap);
        if real_cap < planned {
            eprintln!(
                "raon1_mini: expanded K/V mirror growth capped by real free memory: planned {:.0} MiB, still allocatable {:.0} MiB after the {:.0} MiB floor",
                mib(planned),
                mib(real_cap),
                mib(MIRROR_REAL_FLOOR_BYTES)
            );
        }
        let (q8, rows) = self.plan_mirror(usable);
        self.dims.mirror_q8 = q8;
        self.mirror_budget_rows = rows;
        Ok(self.mirror_budget_rows)
    }

    /// Device bytes the expanded mirror, the prefill scratch and the module work
    /// buffers may still claim together right now: free memory minus the latent
    /// growth to `max_seq`, [`ADMISSION_HEADROOM_BYTES`] and
    /// [`MIRROR_POST_ASSEMBLY_RESERVE_BYTES`]. The chunk admission subtracts a
    /// candidate's scratch+work bytes from this and asks
    /// [`Self::projected_mirror_budget_rows`] what is left for the mirror.
    pub fn mirror_available_bytes(&self, gpu: &Gpu) -> Result<usize, String> {
        self.mirror_available_with_input(gpu).map(|(b, _)| b)
    }

    /// [`Self::mirror_available_bytes`] plus the planned-capacity input it was
    /// decided from (for the load log). Planned from device total minus this
    /// process's own allocations and a fixed reserve, never the instantaneous
    /// free figure.
    pub fn mirror_available_with_input(
        &self,
        gpu: &Gpu,
    ) -> Result<(usize, rdna_compute::PlannedCapacity), String> {
        let cap = hip(gpu.planned_capacity(), "query device memory")?;
        Ok((
            cap.bytes
                .saturating_sub(self.pending_latent_bytes(gpu))
                .saturating_sub(ADMISSION_HEADROOM_BYTES)
                .saturating_sub(MIRROR_POST_ASSEMBLY_RESERVE_BYTES),
            cap,
        ))
    }

    /// Mirror budget in rows [`Self::fit_mirror_budget`] would settle on if
    /// `usable` bytes were available to it (pure: no device query, no state
    /// change; same capacity-based BF16/Q8 choice).
    pub fn projected_mirror_budget_rows(&self, usable: usize) -> usize {
        self.plan_mirror(usable).1
    }

    /// `(mirror is Q8, budget rows)` for `usable` bytes.
    fn plan_mirror(&self, usable: usize) -> (bool, usize) {
        let mut q8 = self.dims.mirror_q8;
        if q8 && self.mirror.is_none() {
            let bf16 = RaonStateDims {
                mirror_q8: false,
                ..self.dims.clone()
            };
            if mirror_rows_for_budget(&bf16, usable) >= self.max_seq {
                q8 = false;
            }
        }
        let dims = RaonStateDims {
            mirror_q8: q8,
            ..self.dims.clone()
        };
        let rows = self.mirror_backed_rows() + mirror_rows_for_budget(&dims, usable);
        (q8, rows.min(self.max_seq))
    }

    /// Override the admission-time budget (oracle/test control: a lower value
    /// forces the beyond-budget routes; a higher one stays bounded by
    /// `max_seq` and by the free-memory check of every growth).
    pub fn set_mirror_budget_rows(&mut self, rows: usize) {
        self.mirror_budget_rows = rows.min(self.max_seq);
    }

    /// Rows of the mirror that equal the expansion of the latent cache.
    pub fn mirror_valid_rows(&self) -> usize {
        self.mirror_valid_rows
    }

    pub fn set_mirror_valid_rows(&mut self, rows: usize) {
        self.mirror_valid_rows = rows;
    }

    /// Rows at or beyond `n` are about to be rewritten: they stop being valid.
    pub fn clamp_mirror_valid_rows(&mut self, n: usize) {
        self.mirror_valid_rows = self.mirror_valid_rows.min(n);
    }

    /// Rows the mirror is physically backed for right now.
    pub fn mirror_backed_rows(&self) -> usize {
        self.mirror.as_ref().map_or(0, |m| m.rows)
    }

    /// Why a context of `rows` cannot be served by the mirror, naming the
    /// maximum context the decompressed route supports on this device.
    pub fn mirror_capacity_error(&self, rows: usize) -> String {
        let max = self.mirror_budget_rows.min(self.max_seq);
        format!(
            "raon state: a context of {rows} rows exceeds the maximum context of {max} tokens the \
             decompressed MLA route can hold on this device (expanded K/V mirror {} B/token, \
             admission budget {max} rows, slot reserve {} rows); \
             the absorbed route (HIPFIRE_RAON_MLA_ABSORB=1, not bit-exact) attends over the latent cache beyond it",
            mirror_row_bytes(&self.dims),
            self.max_seq
        )
    }

    /// Back the expanded mirror for `rows` rows (map-ahead). Outside capture
    /// only. VMM planes grow in place; the dense fallback reallocates (and
    /// drops the valid-row mark). Refuses with
    /// [`Self::mirror_capacity_error`] above the admission budget and when the
    /// growth does not fit the free device memory; a refusal leaves the mirror
    /// usable at its previous size. Returns whether the backing grew.
    pub fn ensure_mirror_rows(&mut self, gpu: &mut Gpu, rows: usize) -> Result<bool, String> {
        if rows == 0 {
            return Ok(false);
        }
        if rows > self.mirror_budget_rows {
            return Err(self.mirror_capacity_error(rows));
        }
        if self.mirror.as_ref().is_some_and(|m| m.rows >= rows) {
            return Ok(false);
        }
        match self.backend {
            LatentBackend::Vmm => self.ensure_mirror_vmm(gpu, rows),
            LatentBackend::Dense => self.ensure_mirror_dense(gpu, rows),
        }
    }

    fn ensure_mirror_dense(&mut self, gpu: &mut Gpu, rows: usize) -> Result<bool, String> {
        let target = rows
            .next_multiple_of(MIRROR_CHUNK_ROWS)
            .min(self.mirror_budget_rows)
            .max(rows);
        let d = self.dims.clone();
        let nope = d.mla_qk_head - d.tail;
        let need = mul(mirror_row_bytes(&d), target)?;
        // The old planes are released only after the new ones exist; the
        // free-memory check counts them as reclaimable so a growth near the
        // budget is not refused for the transient overlap alone.
        let old = self
            .mirror
            .as_ref()
            .map_or(0, |m| m.rows * mirror_row_bytes(&d));
        check_free(
            gpu,
            need.saturating_sub(old),
            GROWTH_HEADROOM_BYTES,
            &format!("dense expanded mirror of {target} rows"),
        )?;
        let mut knope = Vec::with_capacity(d.n_mla);
        let mut vmir = Vec::with_capacity(d.n_mla);
        let [kb, vb] = mirror_plane_row_bytes(&d);
        let built = (|| -> Result<(), String> {
            for _ in 0..d.n_mla {
                if d.mirror_q8 {
                    knope.push(hip(
                        gpu.zeros(&[target, kb], DType::Raw),
                        "alloc dense Q8 mirror K",
                    )?);
                    vmir.push(hip(
                        gpu.zeros(&[target, vb], DType::Raw),
                        "alloc dense Q8 mirror V",
                    )?);
                } else {
                    knope.push(hip(
                        gpu.zeros(&[target, d.heads, nope], DType::BF16),
                        "alloc dense mirror K",
                    )?);
                    vmir.push(hip(
                        gpu.zeros(&[target, d.heads, d.mla_v_head], DType::BF16),
                        "alloc dense mirror V",
                    )?);
                }
            }
            Ok(())
        })();
        if let Err(e) = built {
            for t in knope.into_iter().chain(vmir) {
                let _ = gpu.free_tensor(t);
            }
            return Err(e);
        }
        self.release_expanded_mirror(gpu)?;
        self.mirror = Some(ExpandedMirror {
            rows: target,
            knope,
            vmir,
            q8: d.mirror_q8,
            plans: None,
        });
        self.mirror_valid_rows = 0;
        Ok(true)
    }

    fn ensure_mirror_vmm(&mut self, gpu: &mut Gpu, rows: usize) -> Result<bool, String> {
        let d = self.dims.clone();
        let nope = d.mla_qk_head - d.tail;
        if self.mirror.is_none() {
            let [kb, vb] = mirror_plane_row_bytes(&d);
            let granularity = hip(gpu.vmm_recommended_granularity(), "VMM granularity")?;
            let max_seq = self.max_seq;
            let plan = |bytes: usize| {
                KvChunkPlan::new(
                    bytes,
                    max_seq,
                    MIRROR_CHUNK_ROWS.min(max_seq),
                    granularity,
                    DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
                )
                .map_err(|e| format!("raon state: mirror VMM growth plan: {e}"))
            };
            let plans = [plan(kb)?, plan(vb)?];
            let mut knope: Vec<GpuTensor> = Vec::with_capacity(d.n_mla);
            let mut vmir: Vec<GpuTensor> = Vec::with_capacity(d.n_mla);
            let built = (|| -> Result<(), String> {
                for _ in 0..d.n_mla {
                    // SAFETY: nothing is mapped yet; every access stays within
                    // the mapped prefix reported as `ExpandedMirror::rows`,
                    // grown only below.
                    let k = hip(
                        unsafe {
                            if d.mirror_q8 {
                                gpu.alloc_vmm_tensor(&[max_seq, kb], DType::Raw, 0, &[])
                            } else {
                                gpu.alloc_vmm_tensor(&[max_seq, d.heads, nope], DType::BF16, 0, &[])
                            }
                        },
                        "reserve mirror K VMM plane",
                    )?;
                    knope.push(k);
                    let v = hip(
                        unsafe {
                            if d.mirror_q8 {
                                gpu.alloc_vmm_tensor(&[max_seq, vb], DType::Raw, 0, &[])
                            } else {
                                gpu.alloc_vmm_tensor(
                                    &[max_seq, d.heads, d.mla_v_head],
                                    DType::BF16,
                                    0,
                                    &[],
                                )
                            }
                        },
                        "reserve mirror V VMM plane",
                    )?;
                    vmir.push(v);
                }
                Ok(())
            })();
            if let Err(e) = built {
                for t in knope.into_iter().chain(vmir) {
                    let _ = gpu.free_tensor(t);
                }
                return Err(e);
            }
            self.mirror = Some(ExpandedMirror {
                rows: 0,
                knope,
                vmir,
                q8: d.mirror_q8,
                plans: Some(plans),
            });
        }
        let m = self.mirror.as_mut().expect("mirror just reserved");
        let plans = m.plans.expect("VMM mirror has growth plans");
        let mut total = 0usize;
        for (planes, plan) in [(&m.knope, plans[0]), (&m.vmir, plans[1])] {
            for (i, t) in planes.iter().enumerate() {
                let mapped = gpu
                    .vmm_mapped_bytes(t)
                    .ok_or_else(|| format!("raon state: mirror plane {i} is not a VMM owner"))?;
                if let Some(g) = plan
                    .growth(mapped, rows)
                    .map_err(|e| format!("raon state: mirror plane {i} growth: {e}"))?
                {
                    total += g.size_bytes;
                }
            }
        }
        check_free(
            gpu,
            total,
            GROWTH_HEADROOM_BYTES,
            &format!("expanded mirror growth to {rows} rows"),
        )?;
        let mut result: Result<(), String> = Ok(());
        'grow: for (planes, plan) in [(&mut m.knope, plans[0]), (&mut m.vmir, plans[1])] {
            for (i, t) in planes.iter_mut().enumerate() {
                let mapped = gpu
                    .vmm_mapped_bytes(t)
                    .ok_or_else(|| format!("raon state: mirror plane {i} is not a VMM owner"))?;
                let growth = match plan.growth(mapped, rows) {
                    Ok(g) => g,
                    Err(e) => {
                        result = Err(format!("raon state: mirror plane {i} growth: {e}"));
                        break 'grow;
                    }
                };
                if let Some(g) = growth {
                    if let Err(e) = gpu.grow_vmm_tensor(t, g.size_bytes, &[]) {
                        result = Err(format!("raon state: map mirror plane {i}: {e:?}"));
                        break 'grow;
                    }
                }
            }
        }
        let mut backed = usize::MAX;
        for (planes, plan) in [(&m.knope, plans[0]), (&m.vmir, plans[1])] {
            for t in planes.iter() {
                backed = backed.min(plan.token_capacity(gpu.vmm_mapped_bytes(t).unwrap_or(0)));
            }
        }
        m.rows = if backed == usize::MAX { 0 } else { backed };
        result.map(|()| true)
    }

    pub fn release_expanded_mirror(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.mirror_valid_rows = 0;
        let Some(m) = self.mirror.take() else {
            return Ok(());
        };
        let mut first: Option<String> = None;
        for t in m.knope.into_iter().chain(m.vmir) {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(format!("raon state: free expanded mirror: {e:?}"));
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Per-owner physical/virtual bytes right now.
    pub fn bytes_ledger(&self, gpu: &Gpu) -> StateLedger {
        let l = &self.layout;
        let n = self.dims.n_rda;
        let mut entries = vec![
            LedgerEntry {
                owner: "rda_state".into(),
                physical_bytes: l.rda_stride * n,
                virtual_bytes: l.rda_stride * n,
            },
            LedgerEntry {
                owner: "conv_history".into(),
                physical_bytes: l.conv_stride * n,
                virtual_bytes: l.conv_stride * n,
            },
            LedgerEntry {
                owner: "seq_len".into(),
                physical_bytes: l.seq_bytes,
                virtual_bytes: l.seq_bytes,
            },
        ];
        for (i, t) in self.mla_cache.iter().enumerate() {
            let (physical, virt) = match self.backend {
                LatentBackend::Vmm => (
                    gpu.vmm_mapped_bytes(t).unwrap_or(0),
                    gpu.vmm_reserved_bytes(t).unwrap_or(0),
                ),
                LatentBackend::Dense => (t.buf.size(), t.buf.size()),
            };
            entries.push(LedgerEntry {
                owner: format!("latent_cache.mla{i}"),
                physical_bytes: physical,
                virtual_bytes: virt,
            });
        }
        if let Some(m) = &self.mirror {
            let mut physical = 0usize;
            let mut virt = 0usize;
            for t in m.knope.iter().chain(m.vmir.iter()) {
                match self.backend {
                    LatentBackend::Vmm => {
                        physical += gpu.vmm_mapped_bytes(t).unwrap_or(0);
                        virt += gpu.vmm_reserved_bytes(t).unwrap_or(0);
                    }
                    LatentBackend::Dense => {
                        physical += t.buf.size();
                        virt += t.buf.size();
                    }
                }
            }
            entries.push(LedgerEntry {
                owner: "expanded_mirror".into(),
                physical_bytes: physical,
                virtual_bytes: virt,
            });
        }
        StateLedger { entries }
    }

    /// Release every device owner. VMM caches go through `Gpu::free_tensor`
    /// (arena release; a failed release stays quarantined in the Gpu for
    /// `retry_vmm_cleanup`); the first error is returned after attempting all.
    pub fn free_gpu(mut self, gpu: &mut Gpu) -> Result<(), String> {
        let mut first: Option<String> = None;
        if let Some(m) = self.mirror.take() {
            for t in m.knope.into_iter().chain(m.vmir) {
                if let Err(e) = gpu.free_tensor(t) {
                    first.get_or_insert(format!("raon state: free expanded mirror: {e:?}"));
                }
            }
        }
        for (i, t) in self.mla_cache.drain(..).enumerate() {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(format!("raon state: free latent cache {i}: {e:?}"));
            }
        }
        // Views are borrowed; only the slab owner is freed.
        self.rda.clear();
        self.conv_q.clear();
        self.conv_k.clear();
        self.conv_v.clear();
        if let Err(e) = gpu.free_tensor(self.persist) {
            first.get_or_insert(format!("raon state: free persistent slab: {e:?}"));
        }
        match first {
            None => Ok(()),
            Some(e) => Err(e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prefill / decode scratch
// ─────────────────────────────────────────────────────────────────────────────

/// Element class of a scratch buffer. `U32`/`I32` buffers are `DType::Raw`
/// tensors whose shape is `[byte_len]`; `F32`/`BF16` use their dtype and the
/// logical dims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemKind {
    F32,
    BF16,
    U32,
    I32,
}

impl ElemKind {
    pub const fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::BF16 => 2,
        }
    }
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::BF16 => DType::BF16,
            Self::U32 | Self::I32 => DType::Raw,
        }
    }
}

/// One planned scratch buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchEntry {
    pub name: &'static str,
    pub kind: ElemKind,
    pub dims: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
}

/// Slab layout of a scratch set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchPlan {
    pub rows: usize,
    pub entries: Vec<ScratchEntry>,
    pub total_bytes: usize,
}

impl ScratchPlan {
    pub fn entry(&self, name: &str) -> Option<&ScratchEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
    pub fn to_ledger(&self) -> StateLedger {
        StateLedger {
            entries: self
                .entries
                .iter()
                .map(|e| LedgerEntry {
                    owner: format!("scratch.{}", e.name),
                    physical_bytes: e.bytes,
                    virtual_bytes: e.bytes,
                })
                .collect(),
        }
    }
}

fn build_plan(
    rows: usize,
    specs: Vec<(&'static str, ElemKind, Vec<usize>)>,
) -> Result<ScratchPlan, String> {
    let mut entries = Vec::with_capacity(specs.len());
    let mut offset = 0usize;
    for (name, kind, dims) in specs {
        let mut elems = 1usize;
        for &d in &dims {
            elems = mul(elems, d)?;
        }
        if elems == 0 {
            return Err(format!("raon scratch: buffer {name} has zero elements"));
        }
        let bytes = mul(elems, kind.bytes())?;
        entries.push(ScratchEntry {
            name,
            kind,
            dims,
            offset,
            bytes,
        });
        offset = align_up(offset + bytes, SLAB_ALIGN)?;
    }
    Ok(ScratchPlan {
        rows,
        entries,
        total_bytes: offset,
    })
}

fn scratch_view(slab: &GpuTensor, e: &ScratchEntry) -> GpuTensor {
    let shape = match e.kind {
        ElemKind::U32 | ElemKind::I32 => vec![e.bytes],
        _ => e.dims.clone(),
    };
    GpuTensor {
        buf: slab.buf.byte_view(e.offset, e.bytes),
        shape,
        dtype: e.kind.dtype(),
    }
}

macro_rules! scratch_buffers {
    ($d:ident, $r:ident;
     $( $(#[$meta:meta])* $name:ident : $kind:ident [ $($dim:expr),+ $(,)? ] ),+ $(,)?) => {
        /// Every intermediate the Raon forward needs for at most `rows` rows
        /// (rows = 1 is the decode set). All buffers are views of one exact
        /// device slab, zero-initialised at construction; row `i` of a
        /// `[rows, ...]` buffer starts at `i * row_stride`. F32 planes may hold
        /// BF16-rounded values.
        pub struct RaonPrefillScratch {
            pub rows: usize,
            slab: GpuTensor,
            plan: ScratchPlan,
            $( $(#[$meta])* pub $name: GpuTensor, )+
        }

        fn scratch_specs($d: &RaonStateDims, $r: usize)
            -> Vec<(&'static str, ElemKind, Vec<usize>)>
        {
            vec![ $( (stringify!($name), ElemKind::$kind, vec![$($dim),+]) ),+ ]
        }

        impl RaonPrefillScratch {
            fn from_slab(slab: GpuTensor, plan: ScratchPlan, rows: usize) -> Self {
                let mut it = plan.entries.iter();
                $( let $name = scratch_view(&slab, it.next().expect("scratch plan/field count")); )+
                Self { rows, $( $name, )+ slab, plan }
            }
        }
    };
}

scratch_buffers! { d, r;
    /// Token ids of the rows in flight (u32).
    token_ids: U32 [r],
    /// Absolute positions of the rows in flight (i32).
    positions: I32 [r],
    /// mHC stream ping-pong A/B `[rows, streams, hidden]`.
    stream_a: F32 [r, d.n_streams, d.hidden],
    stream_b: F32 [r, d.n_streams, d.hidden],
    /// Branch input after aggregate+RMS, and branch output before postmix.
    branch_in: F32 [r, d.hidden],
    branch_out: F32 [r, d.hidden],
    /// Rotated-basis activation staging, widest projection input.
    x_rot: F32 [r, d.dense_inter.max(d.hidden).max(d.heads * d.v_dim)],
    /// RDA raw projections (three separate planes, `[rows, heads*k|k|v]`),
    /// output gate, gate-ratio inputs, decay bottleneck and its expansion.
    /// The conv/prepare/recurrence intermediates are owned by the RDA module's
    /// own scratch.
    rda_q_raw: F32 [r, d.heads * d.k_dim],
    rda_k_raw: F32 [r, d.heads * d.k_dim],
    rda_v_raw: F32 [r, d.heads * d.v_dim],
    rda_z: F32 [r, d.heads * d.v_dim],
    rda_b: F32 [r, d.heads],
    rda_delta: F32 [r, d.heads],
    rda_f_lat: F32 [r, d.decay_bottleneck],
    rda_f: F32 [r, d.heads * d.k_dim],
    /// Gated, per-head-normalized recurrence output `[rows, heads*v]`.
    rda_gated: F32 [r, d.heads * d.v_dim],
    mla_qa: F32 [r, d.q_lora],
    mla_qa_norm: F32 [r, d.q_lora],
    mla_q: F32 [r, d.mla_q_out],
    /// Raw `Wkv_a` output `[rows, latent+tail]` before norm/cache write.
    mla_kv_a: F32 [r, d.latent + d.tail],
    mla_g: F32 [r, d.heads * d.mla_v_head],
    mla_attn: F32 [r, d.heads * d.mla_v_head],
    /// Bounded latent-decompression tile for a quantized `Wkv_b`: gathered
    /// normalized latent plane in, expanded `[tile, mla_kv_b_out]` out.
    mla_latent_plane: F32 [MLA_TILE_ROWS, d.latent],
    mla_kv_tile: F32 [MLA_TILE_ROWS, d.mla_kv_b_out],
    /// FFN branch output `[rows, hidden]` handed to the FFN modules; their
    /// own work buffers (router, expert activations, ...) are module-owned.
    ffn_out: F32 [r, d.hidden],
    /// Last-row head chain: stream mean, normalized hidden, logits.
    final_mean: F32 [d.hidden],
    final_norm: F32 [d.hidden],
    logits: F32 [d.vocab],
}

/// Layout of the scratch set for `rows` rows.
pub fn scratch_plan(d: &RaonStateDims, rows: usize) -> Result<ScratchPlan, String> {
    d.validate()?;
    if rows == 0 || rows > MAX_SCRATCH_ROWS {
        return Err(format!(
            "raon scratch: rows {rows} outside 1..={MAX_SCRATCH_ROWS}"
        ));
    }
    build_plan(rows, scratch_specs(d, rows))
}

impl RaonPrefillScratch {
    /// Device bytes [`RaonPrefillScratch::new`] allocates for `max_rows`.
    pub fn required_bytes(d: &RaonStateDims, max_rows: usize) -> Result<usize, String> {
        Ok(scratch_plan(d, max_rows)?.total_bytes)
    }

    /// Allocate a zeroed scratch set; refuses when it cannot fit with
    /// [`GROWTH_HEADROOM_BYTES`] to spare. `max_rows = 1` is the decode set.
    pub fn new(gpu: &mut Gpu, dims: &RaonStateDims, max_rows: usize) -> Result<Self, String> {
        let plan = scratch_plan(dims, max_rows)?;
        check_free(
            gpu,
            plan.total_bytes,
            GROWTH_HEADROOM_BYTES,
            &format!("prefill scratch for {max_rows} rows"),
        )?;
        let slab = hip(
            gpu.alloc_tensor_exact(&[plan.total_bytes], DType::Raw),
            "alloc scratch slab",
        )?;
        if let Err(e) = zero_buf(gpu, &slab.buf, plan.total_bytes) {
            let _ = gpu.free_tensor(slab);
            return Err(format!("raon scratch: zero slab: {e:?}"));
        }
        Ok(Self::from_slab(slab, plan, max_rows))
    }

    pub fn plan(&self) -> &ScratchPlan {
        &self.plan
    }
    pub fn bytes(&self) -> usize {
        self.plan.total_bytes
    }
    pub fn bytes_ledger(&self) -> StateLedger {
        self.plan.to_ledger()
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        hip(gpu.free_tensor(self.slab), "free scratch slab")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAN: usize = 2 << 20;

    #[test]
    fn raon_dims_validate() {
        RaonStateDims::raon().validate().unwrap();
        let mut bad = RaonStateDims::raon();
        bad.conv_channels += 1;
        assert!(bad.validate().is_err());
        let mut bad = RaonStateDims::raon();
        bad.n_mla = 8;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn recurrent_bytes_match_plan_and_ignore_max_seq() {
        let d = RaonStateDims::raon();
        let l = persist_layout(&d).unwrap();
        // PLAN §3.5: 18*40*128*128*4 = 45 MiB (256-aligned strides are exact here).
        assert_eq!(l.rda_stride * d.n_rda, 47_185_920);
        assert_eq!(l.conv_stride * d.n_rda, 18 * 15360 * 3 * CONV_ELEM_BYTES);
        for max_seq in [512usize, 8192, 32768, 65536, 98304, 262144] {
            let b = state_budget(&d, max_seq, LatentBackend::Vmm, GRAN).unwrap();
            assert_eq!(b.persist, l, "persist layout must not depend on max_seq");
            let b = state_budget(&d, max_seq, LatentBackend::Dense, 1).unwrap();
            assert_eq!(b.persist, l);
        }
    }

    #[test]
    fn latent_bytes_follow_plan() {
        let d = RaonStateDims::raon();
        assert_eq!(d.latent_row_bytes(), 1152);
        // 7 layers * 576 * 2 = 8064 B/token.
        assert_eq!(d.latent_row_bytes() * d.n_mla, 8064);
        let dense = state_budget(&d, 65536, LatentBackend::Dense, 1).unwrap();
        assert_eq!(dense.latent_full_per_layer * d.n_mla, 65536 * 8064);
        assert_eq!(
            dense.latent_full_per_layer * d.n_mla,
            504 * 1024 * 1024,
            "65536 tokens are 504 MiB"
        );
        let full = state_budget(&d, 262144, LatentBackend::Dense, 1).unwrap();
        assert_eq!(full.latent_full_per_layer * d.n_mla, 2016 * 1024 * 1024);
    }

    #[test]
    fn vmm_reserves_virtual_max_but_maps_small_initial_range() {
        let d = RaonStateDims::raon();
        let b = state_budget(&d, 98304, LatentBackend::Vmm, GRAN).unwrap();
        assert!(b.latent_reserved_per_layer >= 98304 * 1152);
        assert!(b.latent_initial_per_layer <= 4 << 20);
        assert!(b.latent_initial_per_layer % GRAN == 0);
        assert!(b.latent_full_per_layer >= 98304 * 1152);
        assert!(b.latent_full_per_layer <= b.latent_reserved_per_layer);
        assert!(b.initial_physical(&d) < 64 << 20);
    }

    #[test]
    fn vmm_growth_hits_chunk_boundaries() {
        let d = RaonStateDims::raon();
        let plan = latent_chunk_plan(&d, 98304, GRAN).unwrap();
        let first = plan.mapped_bytes_for_tokens(1).unwrap();
        let cap = plan.token_capacity(first);
        assert!(cap >= INITIAL_LATENT_ROWS);
        assert!(plan.growth(first, cap).unwrap().is_none());
        let g = plan.growth(first, cap + 1).unwrap().expect("must grow");
        assert_eq!(g.offset_bytes, first);
        assert!(g.token_capacity > cap);
        assert_eq!(g.size_bytes % GRAN, 0);
    }

    #[test]
    fn scratch_plan_is_disjoint_aligned_and_bounded() {
        let d = RaonStateDims::raon();
        for rows in [1usize, 512, 1024] {
            let p = scratch_plan(&d, rows).unwrap();
            assert_eq!(p.entries.len(), scratch_specs(&d, rows).len());
            let mut end = 0usize;
            for e in &p.entries {
                assert_eq!(e.offset % SLAB_ALIGN, 0, "{}", e.name);
                assert!(e.offset >= end, "{} overlaps previous", e.name);
                end = e.offset + e.bytes;
            }
            assert!(end <= p.total_bytes);
            assert_eq!(p.total_bytes % SLAB_ALIGN, 0);
        }
        let p512 = scratch_plan(&d, 512).unwrap();
        let p1024 = scratch_plan(&d, 1024).unwrap();
        assert!(p1024.total_bytes > p512.total_bytes);
        // Everything but the tile, head tables and logits scales with rows.
        assert_eq!(p512.entry("logits").unwrap().bytes, 259_840 * 4);
        assert_eq!(
            p512.entry("rda_q_raw").unwrap().bytes,
            512 * 5120 * 4,
            "raw q plane is [rows, heads*k] F32"
        );
        assert!(
            p512.entry("moe_act").is_none(),
            "FFN/MoE work buffers are module-owned, not scratch fields"
        );
        assert_eq!(p512.entry("stream_a").unwrap().bytes, 512 * 4 * 2560 * 4);
        assert!(scratch_plan(&d, 0).is_err());
        assert!(scratch_plan(&d, MAX_SCRATCH_ROWS + 1).is_err());
    }

    #[test]
    fn mirror_bytes_per_token_match_plan() {
        let mut d = RaonStateDims::raon();
        // BF16: 7*40*(128+128)*2 = 143,360 B/token (shared tail stays in the latent cache).
        d.mirror_q8 = false;
        assert_eq!(expanded_mirror_bytes(&d, 8192).unwrap(), 8192 * 143_360);
        // Q8: per plane row 40*(128+2) = 5,200 B (codes + 40 F16 scales), K and V, 7 layers.
        d.mirror_q8 = true;
        assert_eq!(mirror_plane_row_bytes(&d), [5200, 5200]);
        assert_eq!(expanded_mirror_bytes(&d, 8192).unwrap(), 8192 * 72_800);
        assert!(d.validate().is_ok());
    }

    #[test]
    fn ledger_sums() {
        let l = StateLedger {
            entries: vec![
                LedgerEntry {
                    owner: "a".into(),
                    physical_bytes: 3,
                    virtual_bytes: 10,
                },
                LedgerEntry {
                    owner: "latent_cache.x".into(),
                    physical_bytes: 4,
                    virtual_bytes: 20,
                },
            ],
        };
        assert_eq!(l.physical_total(), 7);
        assert_eq!(l.virtual_total(), 30);
        assert_eq!(l.physical_with_prefix("latent_cache"), 4);
    }
}
