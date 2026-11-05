//! Prefill on the hetero split's expert device (gfx1151, `topology.hetero_experts`).
//!
//! The primary publishes a chunk's normed rows and its route into the pinned
//! prefill mailbox. Per layer the expert device masks the hot slots (-1), zeroes
//! the mailbox partial and runs hipfire's own sealed grouped prefill
//! ([`PrefillRouteMode::HeteroCold`]) over the cold experts, combining into that
//! partial. Every operand of the call except the scratch is a view of the
//! mailbox or of the cold experts' host spans, which the expert device reads in
//! place.

use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::context::DispatchCtx;
use crate::types::DispatchError;
use crate::families::gemv::WeightRef;
use crate::families::moe::{
    MoeDtypes, MoeNormalization, MoePrefillParams, MoePrefillPrelude, MoeQ8RouterPolicy,
    MoeRecipe, MoeRoutePolicy, RoutedExpertWeights,
};
use crate::hetero_responder::ColdLayer;
use crate::pipeline::{
    checked_grouped_m_total_bound, seal_prefill, BoundMoeExperts, ExpertBindingCache,
    ExpertMetadata, ExpertResource, ExpertResources, ExpertTable, PrefillRouteMode, Step,
};
use crate::types::dtype_rotation_plan;

/// Grouped-GEMM row block of the sealed prefill (`MOE_GROUPED_BLOCK_M`).
const GROUPED_BLOCK_M: usize = 16;

fn err(msg: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("hetero prefill: {msg}"))
}

fn hip(e: impl std::fmt::Display) -> DispatchError {
    err(e)
}

/// Geometry shared by every layer of the load.
#[derive(Clone, Copy, Debug)]
pub struct ColdPrefillConfig {
    /// Hidden size (the gate/up K and the down M).
    pub dim: usize,
    /// Expert intermediate size.
    pub mi: usize,
    pub top_k: usize,
    pub n_experts: usize,
    /// Largest chunk a call may carry (the mailbox's prefill rows).
    pub max_rows: usize,
    pub gate_up_dtype: DType,
    pub down_dtype: DType,
    /// Encoded bytes of one expert's gate/up `[2 mi, dim]` and down `[dim, mi]`.
    pub gate_up_bytes: usize,
    pub down_bytes: usize,
    /// The primary's sealed route policy for this model, so the expert device
    /// lowers the same grouped route family (`None`: generic resolution).
    pub route_policy: Option<MoeRoutePolicy>,
    pub experts_all_gate_up_mq4: bool,
    /// The model's combine recipe (Qwen4: BF16 round trip, shared after combine).
    pub recipe: MoeRecipe,
}

/// Borrowed views of one layer's cold experts; hot experts are absent.
struct ColdExperts {
    views: Vec<Option<(GpuTensor, GpuTensor)>>,
    absent: (GpuTensor, GpuTensor),
    cfg: ColdPrefillConfig,
    id: u64,
}

fn view(addr: u64, bytes: usize, shape: Vec<usize>, dtype: DType) -> GpuTensor {
    // SAFETY: a non-owning view; the caller keeps the backing span alive for
    // the layer's lifetime (host-tier expert spans, or null for absent).
    let buf = unsafe { DeviceBuffer::from_raw(addr as *mut std::ffi::c_void, bytes) };
    GpuTensor { buf, shape, dtype }
}

fn absent_view(dtype: DType) -> GpuTensor {
    view(0, 0, vec![0], dtype)
}

impl ColdExperts {
    fn weight<'a>(&self, buf: &'a GpuTensor, gate_up: bool) -> WeightRef<'a> {
        let c = &self.cfg;
        let (dtype, m, k, bytes) = if gate_up {
            (c.gate_up_dtype, 2 * c.mi, c.dim, c.gate_up_bytes)
        } else {
            (c.down_dtype, c.dim, c.mi, c.down_bytes)
        };
        WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: bytes / m,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        }
    }
}

impl RoutedExpertWeights for ColdExperts {
    fn len(&self) -> usize {
        self.views.len()
    }

    fn get(&self, e: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        let (gu, dn) = match self.views.get(e)? {
            Some((gu, dn)) => (gu, dn),
            None => (&self.absent.0, &self.absent.1),
        };
        Some((self.weight(gu, true), self.weight(dn, false)))
    }

    /// The expert device reads its own RAM: host-tier spans are its memory,
    /// not memory behind a link.
    fn host_mapped_projections(&self) -> (bool, bool) {
        (false, false)
    }

    fn immutable_identity(&self) -> Option<u64> {
        Some(self.id)
    }
}

/// One layer's cold experts, bound for the sealed prefill on the expert device.
pub struct ColdPrefillLayer {
    table: ExpertTable,
    cache: ExpertBindingCache,
    experts: ColdExperts,
    gate_up_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    cold_bitmap: GpuTensor,
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl ColdPrefillLayer {
    /// `cold.gate_up[e]` / `cold.down[e]` are expert `e`'s byte addresses (0
    /// for a hot expert, which this device never runs).
    pub fn new(gpu: &mut Gpu, cfg: ColdPrefillConfig, cold: &ColdLayer) -> Result<Self, DispatchError> {
        let n = cfg.n_experts;
        if cold.gate_up.len() != n || cold.down.len() != n || cold.cold_bitmap.len() != n {
            return Err(err("cold tables do not cover every expert"));
        }
        let mut views = Vec::with_capacity(n);
        let mut metadata = Vec::with_capacity(n);
        for e in 0..n {
            let (gu, dn) = (cold.gate_up[e], cold.down[e]);
            let cold_e = cold.cold_bitmap[e] != 0;
            if cold_e != (gu != 0 && dn != 0) || (!cold_e && (gu != 0 || dn != 0)) {
                return Err(err(format!("expert {e}: cold bit and table entries disagree")));
            }
            let gu_res = ExpertResource::new(
                "hetero.cold.gate_up",
                "hetero-cold",
                vec![2 * cfg.mi, cfg.dim],
                cfg.gate_up_dtype,
                cfg.gate_up_bytes,
                cfg.gate_up_bytes / (2 * cfg.mi),
                16,
                dtype_rotation_plan(cfg.gate_up_dtype),
            )?;
            let dn_res = ExpertResource::new(
                "hetero.cold.down",
                "hetero-cold",
                vec![cfg.dim, cfg.mi],
                cfg.down_dtype,
                cfg.down_bytes,
                cfg.down_bytes / cfg.dim,
                16,
                dtype_rotation_plan(cfg.down_dtype),
            )?;
            let record = ExpertMetadata::new(e, 0, e, ExpertResources::fused(gu_res, dn_res)?)?;
            if cold_e {
                views.push(Some((
                    view(gu, cfg.gate_up_bytes, vec![cfg.gate_up_bytes], cfg.gate_up_dtype),
                    view(dn, cfg.down_bytes, vec![cfg.down_bytes], cfg.down_dtype),
                )));
                metadata.push(record);
            } else {
                views.push(None);
                metadata.push(record.into_absent());
            }
        }
        let experts = ColdExperts {
            views,
            absent: (absent_view(cfg.gate_up_dtype), absent_view(cfg.down_dtype)),
            cfg,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        let table = ExpertTable::new(metadata)?;
        let mut cache = table.prepare_binding(0, 1, gpu.device_id)?;
        let upload = |gpu: &mut Gpu, entries: &[u64]| -> Result<GpuTensor, DispatchError> {
            let bytes: Vec<u8> = entries.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let t = gpu.alloc_tensor(&[n * 2], DType::F32).map_err(hip)?;
            gpu.hip.memcpy_htod(&t.buf, &bytes).map_err(hip)?;
            Ok(t)
        };
        let gate_up_ptrs = upload(gpu, &cold.gate_up)?;
        let down_ptrs = upload(gpu, &cold.down)?;
        let cold_bitmap = gpu.alloc_tensor(&[n.div_ceil(4)], DType::F32).map_err(hip)?;
        gpu.hip.memcpy_htod(&cold_bitmap.buf, &cold.cold_bitmap).map_err(hip)?;
        cache.bind_live(&table, &experts, &gate_up_ptrs, &down_ptrs, None, None)?;
        Ok(Self { table, cache, experts, gate_up_ptrs, down_ptrs, cold_bitmap })
    }
}

/// The expert device's grouped-prefill scratch, sized for `max_rows` and
/// shared by every layer (layers run one at a time).
pub struct ColdPrefillScratch {
    cfg: ColdPrefillConfig,
    m_total_max: usize,
    topk_indices: GpuTensor,
    /// Local copies of the mailbox rows and weights, and the local partial:
    /// the mailbox is coherent (uncached) memory, which the grouped GEMMs would
    /// otherwise re-read per expert tile.
    x_norm_batch: GpuTensor,
    topk_weights: GpuTensor,
    x_batch: GpuTensor,
    x_rot_batch: GpuTensor,
    gate_batch: GpuTensor,
    up_batch: GpuTensor,
    rot_batch: GpuTensor,
    down_expanded: GpuTensor,
    expert_token_counts: GpuTensor,
    expert_offsets: GpuTensor,
    sorted_slot_index: GpuTensor,
    expert_tile_ids: GpuTensor,
    inverse_perm: GpuTensor,
    y_gate_up_grouped: GpuTensor,
    y_down_grouped: GpuTensor,
    /// Placeholder router operands: a HeteroCold call runs no router.
    router: GpuTensor,
}

impl ColdPrefillScratch {
    pub fn new(gpu: &mut Gpu, cfg: ColdPrefillConfig) -> Result<Self, DispatchError> {
        let rows = cfg.max_rows;
        let slots = rows * cfg.top_k;
        let m_total_max = checked_grouped_m_total_bound(slots, cfg.n_experts, GROUPED_BLOCK_M)?;
        let mut f32s = |n: usize| gpu.alloc_tensor(&[n.max(1)], DType::F32).map_err(hip);
        Ok(Self {
            cfg,
            m_total_max,
            topk_indices: f32s(slots)?,
            x_norm_batch: f32s(rows * cfg.dim)?,
            topk_weights: f32s(slots)?,
            x_batch: f32s(rows * cfg.dim)?,
            x_rot_batch: f32s(rows * cfg.dim)?,
            gate_batch: f32s(slots * cfg.mi)?,
            up_batch: f32s(slots * cfg.mi)?,
            rot_batch: f32s(slots * cfg.mi)?,
            down_expanded: f32s(slots * cfg.dim)?,
            expert_token_counts: f32s(cfg.n_experts)?,
            expert_offsets: f32s(cfg.n_experts + 1)?,
            sorted_slot_index: f32s(m_total_max)?,
            expert_tile_ids: f32s(m_total_max / GROUPED_BLOCK_M)?,
            inverse_perm: f32s(slots)?,
            y_gate_up_grouped: f32s(m_total_max * 2 * cfg.mi)?,
            y_down_grouped: f32s(m_total_max * cfg.dim)?,
            router: f32s(1)?,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.cfg.max_rows
    }
}

/// Device addresses of the prefill mailbox fields a call reads and writes.
#[derive(Clone, Copy, Debug)]
pub struct ColdPrefillIo {
    /// `[rows, dim]` F32 normed rows (the routed experts' natural activation).
    pub x_norm: u64,
    /// `[rows, top_k]` i32 route ids (unmasked).
    pub ids: u64,
    /// `[rows, top_k]` F32 route weights.
    pub weights: u64,
    /// `[rows, dim]` F32 cold partial (written).
    pub partial: u64,
}

fn zero(gpu: &mut Gpu, buf: &DeviceBuffer, bytes: usize) -> Result<(), DispatchError> {
    match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.memset_async(buf, 0, bytes, stream),
        None => gpu.hip.memset(buf, 0, bytes),
    }
    .map_err(hip)
}

/// Stream-ordered copy (the active stream, else the default stream).
fn copy(gpu: &mut Gpu, dst: &DeviceBuffer, src: &DeviceBuffer, bytes: usize) -> Result<(), DispatchError> {
    match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.memcpy_dtod_async_at(dst, 0, src, 0, bytes, stream),
        None => gpu.hip.memcpy_dtod_async_default_at(dst, 0, src, 0, bytes),
    }
    .map_err(hip)
}

/// Queue one layer's cold partial for `rows` rows on the current stream:
/// mask the hot slots, zero the partial, run the sealed routed tail.
pub fn run_cold_prefill(
    gpu: &mut Gpu,
    layer: &ColdPrefillLayer,
    s: &ColdPrefillScratch,
    rows: usize,
    io: ColdPrefillIo,
) -> Result<(), DispatchError> {
    let c = s.cfg;
    if rows == 0 || rows > c.max_rows {
        return Err(err(format!("{rows} rows outside 1..={}", c.max_rows)));
    }
    let slots = rows * c.top_k;
    let row_bytes = rows * c.dim * 4;
    let mb_x = view(io.x_norm, row_bytes, vec![rows, c.dim], DType::F32);
    let mb_w = view(io.weights, slots * 4, vec![slots], DType::F32);
    let mb_partial = view(io.partial, row_bytes, vec![rows, c.dim], DType::F32);
    let x_norm = GpuTensor { buf: s.x_norm_batch.buf.byte_view(0, row_bytes), shape: vec![rows, c.dim], dtype: DType::F32 };
    let weights = GpuTensor { buf: s.topk_weights.buf.byte_view(0, slots * 4), shape: vec![slots], dtype: DType::F32 };
    let partial = GpuTensor { buf: s.x_batch.buf.byte_view(0, row_bytes), shape: vec![rows, c.dim], dtype: DType::F32 };
    copy(gpu, &x_norm.buf, &mb_x.buf, row_bytes)?;
    copy(gpu, &weights.buf, &mb_w.buf, slots * 4)?;
    zero(gpu, &partial.buf, row_bytes)?;
    gpu.hetero_mask_hot(io.ids, &layer.cold_bitmap, c.n_experts as i32, slots as i32, &s.topk_indices)
        .map_err(hip)?;

    let router = WeightRef {
        buf: &s.router,
        dtype: DType::BF16,
        m: c.n_experts,
        k: c.dim,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    };
    let params = MoePrefillParams {
        dtypes: MoeDtypes {
            router: DType::BF16,
            shared: None,
            experts_all_gate_up_mq4: c.experts_all_gate_up_mq4,
            routed_gate_up: c.gate_up_dtype,
            routed_down: c.down_dtype,
            routed_has_mixed_experts: false,
            has_paro_shared: false,
            per_expert_gate_up: None,
            per_expert_down: None,
        },
        recipe: c.recipe,
        route_policy: c.route_policy,
        prelude: MoePrefillPrelude {
            normalization: MoeNormalization::Provided,
            router,
            router_logits: &s.router,
            router_scores: &s.router,
            norm_topk_prob: false,
            route: PrefillRouteMode::HeteroCold,
            shared: None,
            q8_router_policy: MoeQ8RouterPolicy::DispatcherEntry,
        },
        batch_size: rows,
        mi: c.mi,
        down_m: c.dim,
        down_k: c.mi,
        gate_up_k: c.dim,
        k_top: c.top_k,
        n_exp: c.n_experts,
        m_total_max: s.m_total_max,
        force_mq4_grouped_fp16: false,
        cpu_deferred_experts: false,
        topk_indices: &s.topk_indices,
        topk_weights: &weights,
        x_batch: &partial,
        x_norm_batch: &x_norm,
        x_rot_batch: &s.x_rot_batch,
        expert_gate_up_ptrs: &layer.gate_up_ptrs,
        expert_down_ptrs: &layer.down_ptrs,
        expert_stage_ptrs: None,
        routed_experts: &layer.experts,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        gate_batch: &s.gate_batch,
        up_batch: &s.up_batch,
        rot_batch: &s.rot_batch,
        down_expanded: &s.down_expanded,
        expert_token_counts: &s.expert_token_counts,
        expert_offsets: &s.expert_offsets,
        sorted_slot_index: &s.sorted_slot_index,
        expert_tile_ids: &s.expert_tile_ids,
        inverse_perm: &s.inverse_perm,
        y_gate_up_grouped: &s.y_gate_up_grouped,
        y_down_grouped: &s.y_down_grouped,
        paro_gate_up: None,
        paro_down: None,
        down_awq_scale: None,
        routed_out: None,
        ep_row_gather: None,
        shared_down_out: None,
        // EP-only (row gather / tile exchange owners): unused by HeteroCold.
        ep_layer_idx: 0,
        ep_tile_exchange: None,
        expert_iu4: None,
    };
    let ctx = DispatchCtx::new(gpu);
    let experts = BoundMoeExperts::from_cache(&layer.table, &layer.cache)?;
    let call = seal_prefill(experts, &ctx, params)?;
    crate::pipeline::execute_steps(gpu, &ctx, &[Step::Moe(call)])?;
    copy(gpu, &mb_partial.buf, &partial.buf, row_bytes)
}
