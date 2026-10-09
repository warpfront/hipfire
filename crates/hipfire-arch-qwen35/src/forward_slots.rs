// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// forward_batch_slots — the N-slot forward pass (SP3 Task 2).
//
// A PARALLEL entry point to `qwen35::forward_prefill_batch_with_pbs_opts`,
// deliberately NOT a modification of it. That function carries hipGraph
// capture eligibility, tree-verify, GDN-tape and MTP interactions whose
// interaction with a slot axis is unknown, and every current caller (chat,
// spec decode, MTP, the TUI) depends on it being unchanged. This file
// mirrors its per-layer kernel sequence (see `sp3-task-2-report.md` for the
// enumeration) but routes attention and KV-write through the slot-aware
// `_slots` entry points SP1 built, and DeltaNet through a per-slot loop
// over SP2's per-slot `DeltaNetState`.
//
// Scope: dense and MoE attention projections are admitted per SITE via
// `plan_proj_group` / `slots_check_residual_weight` (see `slots_proj_admissible`
// for the admitted container set): a fused group — {wqkv,wz,wβ,wα} (DeltaNet),
// {wq,wk,wv} (FullAttn), {w_gate,w_up} (FFN) — keeps its single fused launch
// when every member shares one container, and any admitted mix inside the
// group falls back to per-weight plain GEMMs with each member reading the
// activation variant its own dtype wants (`slots_norm_site` produces the
// rotated and/or normed buffer; rotated consumers MUST NOT read the
// unrotated buffer and vice versa). This is what lets the `--tier xt/base/
// pro` and `--fixed-tier` recipes — which lift individual projections to
// Q8_0 inside an MQ4V2 body — run multi-slot. Residual roles (wo, w_down)
// dispatch on the weight's own dtype; `slots_residual_gemm_key` selects the
// container (never a wildcard v1 fallback). Uniform Q8_0 groups take the
// WMMA fused kernels only on WMMA archs with the prefill WMMA switch on,
// otherwise the reference's plain-GEMM fork (`plan_q8_fused_or_plain`).
// DeltaNet state stays `StateQuant::Q8` regardless. Out of scope: Lloyd-V2,
// PARO, E8/MFP4 and G128-rotated containers (named `HipError` refusals).
// This matches the ABI the multi-slot
// infrastructure was actually built against — `SlotPool`'s per-slot
// descriptor addressing (`KvSlotDesc`, with separate legacy K/V bases for
// the tiers whose K/V strides differ), the `*_batched_slots` write/attend
// entry points, and the Q8 `gated_delta_net_q8_batch_seq` recurrence.
//
// The KV cache TIER is no longer fixed: the engine resolves a tier from the
// same policy the sequential carrier uses (`QWEN35_HFQ_POLICY`), and the
// KV-write + attend steps dispatch per tier (`SlotKvTier`, `kv_write_slots`,
// `tier_attend_slots`) across the full static ladder — q8, asym{2,3,4},
// fwht{2,3,4} (bf16's descriptor-aware batched kernels are wired too; the
// qwen35 policies simply never resolve to it). Slot addressing is a
// KV-cache-tier property, not a weight-quant property, so the rest of the
// layer body never changes no matter what a layer's projection weights — or
// the KV tier — are quantized to.
//
// `DeltaNetMoe`/`FullAttnMoe` layers (qwen3.6-35b-a3b and similar A3B
// checkpoints) admit the same uniform attention projection family, mirroring
// `forward_prefill_chunk`'s own `is_mq` dispatch fork for these layer kinds
// (rmsnorm+FWHT-rotate via `fused_rmsnorm_rotate_mq_batched_for` /
// `rotate_x_mq_batched_for`, then container-selected fused keys). This does
// NOT hold for qt=44 (MQ4G256V2): same 136 B stride and nibble payload but the
// 8 header bytes change meaning from `[0..4) f32 scale, [4..8) f32 zero` (one
// affine grid per 256 weights) to `[0..2) fp16 s0, [2..4) fp16 z0, [4..6) fp16
// s1, [6..8) fp16 z1` (s0/z0 for 0..127, s1/z1 for 128..255). A v1 kernel fed
// qt=44 bytes bit_casts fp16 pairs to f32 and decodes every weight to ~1e-14
// at full speed with no error. The MoE FFN itself is stateless per row (no
// kv_cache, no dn_state, no positions — confirmed by reading
// `moe_ffn_decode`'s signature), so it needs no slot machinery at all:
// `run_deltanet_moe_layer_slots`/`run_fullattn_moe_layer_slots` call the
// reference's own `prefill_moe_ffn_body_batched` directly over the flat N-row
// batch, gated by the reference's own `moe_ffn_batched_admissible` (uniform
// MQ4G256V2 / MQ6G256V2 shared+routed paths ride that shared gate — no
// duplicate MoE dispatch in this file). Attention and dense-FFN projections
// are admitted per SITE via `plan_proj_group` / `slots_check_residual_weight`:
// any mix of `slots_proj_admissible` containers within a fused group falls
// back to per-weight plain GEMMs (AWQ sidecars refuse mixed groups), and
// PARO/Lloyd-V2/E8/G128-rotated containers stay out of scope — every refusal
// returns a clear `HipError` rather than guessing at an untested path.
// Legacy v1 MQ6G256 (qt=15) MoE attention rides the uniform-group path: AWQ
// A3B checkpoints (`ornith-1.5:35b-a3b`, `qwen3.6-35b-a3b.mq4-awq-mi300x`)
// ship 4/40 layers — 0, 1, 38, 39 — with uniformly MQ6G256 attention; it
// shares HFQ6G256's 200 B/group container, so the container selectors route
// it to the `*Hfq6G256` keys after the same FWHT rotate, exactly the
// reference's `is_6bit` arms.
//
// DeltaNet slot-state note: only the GDN recurrence and the conv1d causal
// state are sequential-per-slot (both carry state across steps: the S
// matrix and the conv1d ring buffer respectively). The brief's Step 3 only
// spells out the GDN call, but the SAME hazard applies to conv1d — its
// ring buffer seeds the causal window for a slot's FIRST 1-2 rows from the
// PREVIOUS call, so batching rows from different slots into one launch
// would let a slot's first row convolve over a neighbouring slot's last
// rows instead of its own history. Both are looped per slot; every other
// per-layer step (rmsnorm, the QKVZA/QKV projections, sigmoid/alpha-gate,
// the Q/K L2-norm, gated-norm, wo+residual, the FFN) is stateless per row
// and runs once across all N rows regardless of slot boundaries, exactly
// like the FullAttention layer body (whose only slot-aware steps are the
// KV write and the attend call, both single launches via the `_slots`
// entry points — RoPE is slot-agnostic per SP2 Task 2 and needs no split).

pub mod vmm;

use crate::qwen35::prefill::is_batchable_la;
use crate::qwen35::{
    moe_prefill_dtypes, prefill_moe_ffn_body_batched, q8_prefill_wmma_enabled,
    run_fused_gate_up_key, run_fused_qkv_key, run_fused_qkvza_key, run_plain_gemm_key,
    run_residual_gemm_key, DeltaNetLayerWeights, DeltaNetMoeLayerWeights, DeltaNetState,
    FullAttnLayerWeights, FullAttnMoeLayerWeights, LayerType, LayerWeights, MoeFfnWeights,
    PrefillBatchScratch, Qwen35Config, Qwen35Scratch, Qwen35Weights, StateQuant,
};
use hipfire_runtime::slot_batch::SlotBatch;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::fused_qkv::fused_gate_up_key_for;
use hipfire_dispatch::families::gemm::residual_gemm_key_for;
use hipfire_dispatch::families::moe::{
    gated_moe_prefill_admissible, mq6_batched_admit_enabled_from_env,
};

use hipfire_dispatch::families::gemv::{GemvFamily, RotateInputs};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::KernelKey;
use hipfire_runtime::kv_mode::KvMode;
use hipfire_runtime::llama::{
    fused_rmsnorm_rotate_mq_batched_for, fused_silu_mul_rotate_mq_batched_for,
    rotate_x_mq_batched_for, EmbeddingFormat, WeightTensor,
};
use rdna_compute::kv_slots::{build_tiles, KvSlotDesc};
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Packed size of one `KvSlotDesc` (`kernels/src/kv_slot_desc.h`): 8+8+8+4+4.
pub const DESC_BYTES: usize = 32;

/// Pack a `KvSlotDesc` table byte-identically to `kernels/src/kv_slot_desc.h`
/// (`block_table: u64, legacy_k_base: u64, legacy_v_base: u64, seq_len: i32,
/// page_tokens: i32`, 32 bytes, no padding between fields on this target).
/// Mirrors the packer SP1's Task 7 harness uses
/// (`rdna-compute/examples/test_batched_attn_slots.rs::pack_descs`) —
/// duplicated rather than shared because that packer lives in `examples/`
/// (test-only) and this is production `src/`.
fn pack_descs(descs: &[KvSlotDesc]) -> Vec<u8> {
    debug_assert_eq!(
        std::mem::size_of::<KvSlotDesc>(),
        DESC_BYTES,
        "KvSlotDesc ABI drifted away from the packed 32-byte layout"
    );
    let mut out = Vec::with_capacity(descs.len() * DESC_BYTES);
    for d in descs {
        out.extend_from_slice(&d.block_table.to_ne_bytes());
        out.extend_from_slice(&d.legacy_k_base.to_ne_bytes());
        out.extend_from_slice(&d.legacy_v_base.to_ne_bytes());
        out.extend_from_slice(&d.seq_len.to_ne_bytes());
        out.extend_from_slice(&d.page_tokens.to_ne_bytes());
    }
    out
}

/// The resolved KV-cache tier a slots forward runs against, plus the
/// model-global rotation tables its kernels need.
///
/// The multi-slot path originally shipped Q8_0-only (see this file's header
/// comment); it is now tier-generic across the static ladder the qwen35
/// carrier resolves — q8, asym{2,3,4}, fwht{2,3,4} — under BOTH the legacy
/// slab pool and the paged pool, plus the flat native tiers bf16/f16 and
/// the native fp8 tier (gfx1201-only; every other arch refuses at load).
/// The tier touches exactly two steps of the FullAttention layer body: the
/// KV write (K through the tier's packed writer, V through the Q8_0 writer
/// resolving `legacy_v_base`; the flat tiers and fp8 write V through their
/// own writer, K and V strides equal) and the attend call (the tier's
/// descriptor-driven kernel). Everything else — projections, RoPE,
/// DeltaNet, sampling — is tier-agnostic.
///
/// The tables are model-global (shared by every slot and every layer):
/// Givens angles for the asym tiers, ±1 FWHT sign vectors for the fwht
/// tiers, none for q8/bf16/f16/fp8. They are built once by the rig with the
/// same seeds as the sequential path's constructors
/// (`gen_givens_angles(42, ·)`, `gen_fwht_signs(42|1042, ·)`), so the packed
/// cache bytes a slot pool produces are bit-identical to the sequential
/// engine's for the same tier.
pub struct SlotKvTier {
    pub mode: KvMode,
    /// Givens cos table (`[head_dim/2]` f32) — asym tiers only.
    pub givens_cos: Option<GpuTensor>,
    /// Givens sin table — asym tiers only.
    pub givens_sin: Option<GpuTensor>,
    /// FWHT signs1 (`[128]` f32) — fwht tiers only.
    pub fwht_signs1: Option<GpuTensor>,
    /// FWHT signs2 — fwht tiers only.
    pub fwht_signs2: Option<GpuTensor>,
}

impl SlotKvTier {
    /// q8 tier: no tables. The graph/WMMA fast paths remain q8-exclusive.
    pub fn q8() -> Self {
        Self {
            mode: KvMode::Q8,
            givens_cos: None,
            givens_sin: None,
            fwht_signs1: None,
            fwht_signs2: None,
        }
    }

    pub fn is_q8(&self) -> bool {
        matches!(self.mode, KvMode::Q8)
    }
}

/// Drop the rig-owned tables when the engine tears down.
impl SlotKvTier {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        if let Some(t) = self.givens_cos {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.givens_sin {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.fwht_signs1 {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.fwht_signs2 {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Persistent device staging for the two slot-addressing tables every
/// `_slots` kernel call needs: the `KvSlotDesc` table and the per-row
/// `row_slot` map. Owned by the caller across steps (like
/// `PrefillBatchScratch`) so re-uploading is a `memcpy_htod` into a fixed
/// allocation, not an alloc/free pair every step.
///
/// `descs_dev` is re-uploaded only when `pool.descriptors_dirty()` — most
/// steps touch at least one slot's `seq_len` so this rarely skips work, but
/// it's cheap to check and is what the brief asks for. `row_slot_dev` is
/// re-uploaded every step unconditionally: batch composition (which rows
/// belong to which slot) legitimately changes step to step, and `SlotBatch`
/// carries no dirty-tracking of its own.
pub struct SlotDescStaging {
    pub descs_dev: GpuTensor,
    pub row_slot_dev: GpuTensor,
    /// Flat row-tile lists for the WMMA flash-prefill kernel
    /// (`attention_q8_0_flash_prefill_wmma_slots`) — the only `_slots` kernel
    /// with `BR > 1` (fixed `M_TILE = 16`) that this file drives, so it is
    /// the only one that needs tile arrays rather than a plain `row_slot`
    /// map. Capacity is `max_rows` i32 entries: a tile always owns >= 1 row,
    /// so the tile count can never exceed the row count. Rebuilt and
    /// re-uploaded every step (like `row_slot_dev`) whenever more than one
    /// slot is active this step — batch composition legitimately changes
    /// step to step and `SlotBatch` carries no dirty-tracking of its own.
    /// Only their PREFIX (`0..n_tiles_this_step`) is meaningful; callers
    /// must track `n_tiles` themselves (see `forward_batch_slots_with_max_layer`).
    pub tile_slot_dev: GpuTensor,
    pub tile_row0_dev: GpuTensor,
    pub tile_qbase_dev: GpuTensor,
    /// Per-slot GPU buffers for paged block tables (page index arrays).
    /// Each buffer holds `max_pages_per_slot` u32 entries. Only used in
    /// paged mode; empty in legacy mode. Re-uploaded whenever a slot's
    /// block table is dirty; the descriptor's `block_table` pointer is
    /// stable for the lifetime of the staging buffers.
    pub block_table_devs: Vec<GpuTensor>,
    /// Capacity of each `block_table_devs` entry, in pages. 0 = legacy
    /// mode. Uploads are bounded by this — a slot whose table grows past
    /// it must fail the step, never write past the buffer.
    max_pages_per_slot: usize,
    n_slots: usize,
    max_rows: usize,
}

impl SlotDescStaging {
    /// `max_pages_per_slot` is the maximum number of pages a slot's block
    /// table can hold. 0 = legacy mode (no block table buffers allocated).
    pub fn new(
        gpu: &mut Gpu,
        n_slots: usize,
        max_rows: usize,
        max_pages_per_slot: usize,
    ) -> HipResult<Self> {
        // Allocate all five staging buffers, freeing whatever already
        // succeeded if a later allocation fails partway through.
        let mut allocated: Vec<GpuTensor> = Vec::new();
        let shapes: [&[usize]; 5] = [
            // One packed KvSlotDesc is 32 bytes (8+8+8+4+4, see pack_descs
            // and the size assert in rdna_compute::kv_slots) — size the
            // staging table from the ABI, not from a stale field count.
            // The 24-byte sizing predated the separate `legacy_v_base`
            // field (c949fa5c2) and only ran unharmed because the device
            // pool rounds sub-256-byte allocations up to 256 B: the
            // overflow was silently absorbed up to 8 slots and would trip
            // the memcpy_htod bound assert at 9+.
            &[n_slots * DESC_BYTES], // descs_dev
            &[max_rows * 4],         // row_slot_dev
            &[max_rows * 4],         // tile_slot_dev
            &[max_rows * 4],         // tile_row0_dev
            &[max_rows * 4],         // tile_qbase_dev
        ];
        for shape in shapes {
            match gpu.alloc_tensor(shape, DType::Raw) {
                Ok(t) => allocated.push(t),
                Err(e) => {
                    for t in allocated.drain(..) {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(e);
                }
            }
        }
        let mut it = allocated.into_iter();
        // Allocate per-slot block table buffers (paged mode only).
        let mut block_table_devs: Vec<GpuTensor> = Vec::new();
        if max_pages_per_slot > 0 {
            for _ in 0..n_slots {
                match gpu.alloc_tensor(&[max_pages_per_slot * 4], DType::Raw) {
                    Ok(t) => block_table_devs.push(t),
                    Err(e) => {
                        // Free everything allocated so far.
                        for t in block_table_devs.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        let descs = it.next().unwrap();
                        let row_slot = it.next().unwrap();
                        let tile_slot = it.next().unwrap();
                        let tile_row0 = it.next().unwrap();
                        let tile_qbase = it.next().unwrap();
                        let _ = gpu.free_tensor(descs);
                        let _ = gpu.free_tensor(row_slot);
                        let _ = gpu.free_tensor(tile_slot);
                        let _ = gpu.free_tensor(tile_row0);
                        let _ = gpu.free_tensor(tile_qbase);
                        return Err(e);
                    }
                }
            }
        }
        Ok(Self {
            descs_dev: it.next().unwrap(),
            row_slot_dev: it.next().unwrap(),
            tile_slot_dev: it.next().unwrap(),
            tile_row0_dev: it.next().unwrap(),
            tile_qbase_dev: it.next().unwrap(),
            block_table_devs,
            max_pages_per_slot,
            n_slots,
            max_rows,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.descs_dev);
        let _ = gpu.free_tensor(self.row_slot_dev);
        let _ = gpu.free_tensor(self.tile_slot_dev);
        let _ = gpu.free_tensor(self.tile_row0_dev);
        let _ = gpu.free_tensor(self.tile_qbase_dev);
        for t in self.block_table_devs {
            let _ = gpu.free_tensor(t);
        }
    }
}


/// Where one FullAttention layer's KV lives for a slots step.
///
/// `Arena` is the explicitly-legacy shared-arena route (`SlotPool` slab or
/// paged), addressed through `KvSlotDesc` offsets into one arena tensor per
/// layer. `Vmm` is the per-request route: every request owns its own
/// `KvCache` VMM reservation and this layer's `VmmKvSlotDesc` table carries
/// each request's absolute per-layer K/V base VA. The two descriptor ABIs are
/// never reinterpreted as one another.
pub(crate) enum LayerKvAddr<'a> {
    Arena {
        k_cache: &'a GpuTensor,
        v_cache: &'a GpuTensor,
        desc_staging: &'a SlotDescStaging,
        single_slot: Option<(u64, usize)>,
        n_tiles: Option<usize>,
    },
    Vmm(vmm::VmmLayerKv<'a>),
}

/// KV write + attend for one FullAttention layer step on either storage.
#[allow(clippy::too_many_arguments)]
fn layer_kv_write_attend(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    kv: &SlotKvTier,
    addr: &LayerKvAddr,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
) -> HipResult<()> {
    match addr {
        LayerKvAddr::Arena {
            k_cache,
            v_cache,
            desc_staging,
            single_slot,
            n_tiles,
        } => {
            // Batched KV write — slot-aware, ONE launch per arena across every
            // slot, on the engine's resolved KV tier (`kv.mode`). Both arenas
            // resolve through the descriptor table (block tables under paged);
            // on the rotated-K tiers the V write resolves the descriptor's
            // `legacy_v_base`, which differs from `legacy_k_base`.
            kv_write_slots(
                gpu,
                kv,
                k_cache,
                v_cache,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                &pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                n,
                &desc_staging.descs_dev,
                &desc_staging.row_slot_dev,
            )?;
            // Batched attend — slot-aware, one launch across every slot.
            // `positions[]` (not any `desc.seq_len`) is authoritative for the
            // causal bound inside the ported kernels — SP1's only Critical
            // defect came from conflating the two on a tile kernel bounded by
            // `desc.seq_len` while the shared reduce kernel stayed bounded by
            // `positions[]`. `tree_bias` is never combined with descriptors
            // here (asserted out of SP1 scope).
            tier_attend_slots(
                gpu,
                kv,
                &pbs.fa_q_batch,
                k_cache,
                v_cache,
                &pbs.fa_attn_out_batch,
                &pbs.positions,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                physical_cap,
                max_ctx_len,
                n,
                &s.flash_partials,
                &desc_staging.descs_dev,
                &desc_staging.row_slot_dev,
                *single_slot,
                n_tiles.map(|nt| {
                    (
                        &desc_staging.tile_slot_dev,
                        &desc_staging.tile_row0_dev,
                        &desc_staging.tile_qbase_dev,
                        nt,
                    )
                }),
            )
        }
        LayerKvAddr::Vmm(layer) => {
            vmm::vmm_kv_write_attend(gpu, config, layer, pbs, n)
        }
    }
}

/// Projection dtypes the multi-slot attention/FFN body can run at all
/// (uniform OR as a mixed-group member). The arch-sensitive half defers
/// to the shared `is_batchable_la` (MQ3/MQ3-Lloyd WMMA, MQ*V2 WMMA
/// eligibility, etc.) so no duplicate admit table drifts here.
fn slots_proj_admissible(dt: DType, arch: &str) -> bool {
    let member = matches!(
        dt,
        DType::Q8_0
            | DType::MQ4G256
            | DType::HFQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::HFQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ3G256Lloyd
            | DType::MQ2G256V2
    );
    member && is_batchable_la(dt, arch)
}

/// Dtypes whose stored weights are offline-FWHT-rotated over G256 groups:
/// the activation must be rotated by the matching producer before the
/// GEMM. Mirrors the `is_mq` set in `batch_chunk_delta_net_input_projection`
/// minus the containers slots refuses (MQ4G256V2Lloyd has no uniform key
/// anywhere; MFP4G32 and G128-rotated families stay out of v1).
fn slots_weight_rotated(dt: DType) -> bool {
    matches!(
        dt,
        DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ3G256Lloyd
            | DType::MQ2G256V2
    )
}

/// lm_head dtypes the multi-slot path admits. `Step::Gemv` +
/// `weights.output.dispatch_ref()` is dtype-generic, so this is an allow-list
/// of formats whose GEMV/rotate path is known-good. Covers every container
/// `--tier`/`--fixed-tier` can lift the head to (q8..mq6v2) plus the HFQ
/// siblings. Do not silently widen to Lloyd/E8 here.
fn lm_head_slots_admissible(dt: DType) -> bool {
    matches!(
        dt,
        DType::Q8_0
            | DType::MQ4G256
            | DType::HFQ4G256
            | DType::MQ4G256V2
            | DType::MQ4CG256
            | DType::MQ6G256
            | DType::HFQ6G256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::HFQ3G256
            | DType::F16
    )
}

/// Per-projection-group dispatch plan for the dense slot bodies. A "group"
/// is the operand set of one fused launch in the uniform path:
/// {wqkv,wz,w_beta,w_alpha} for DeltaNet, {wq,wk,wv} for FullAttn,
/// {w_gate,w_up} for the FFN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupPlan {
    /// Every member shares one container: one fused launch (Q8_0 keeps its
    /// WMMA-vs-plain split at the call site).
    Uniform(DType),
    /// Members differ: per-weight plain GEMMs. Plan time has already
    /// verified every member has a `slots_plain_gemm_key` and that no
    /// member carries an AWQ sidecar (mixed-dtype AWQ groups would need
    /// one pre-rotation input per weight — refused until needed).
    Mixed { needs_rot: bool, needs_norm: bool },
}

/// Plain (single-weight) batched GEMM key for a projection dtype, or None
/// when no plain kernel exists (uniform-only containers: HFQ6/MQ6, MQ3,
/// MQ3-Lloyd — those can only appear in Uniform groups).
fn slots_plain_gemm_key(dt: DType) -> Option<KernelKey> {
    Some(match dt {
        DType::Q8_0 => KernelKey::GemmQ8_0BatchedChunked,
        DType::MQ4G256 | DType::HFQ4G256 => KernelKey::GemmHfq4G256,
        DType::MQ4G256V2 => KernelKey::GemmMq4G256V2,
        DType::MQ4CG256 => KernelKey::GemmMq4CG256,
        DType::MQ6G256V2 => KernelKey::GemmMq6G256V2,
        DType::MQ5G256V2 => KernelKey::GemmMq5G256V2,
        DType::MQ3G256V2 => KernelKey::GemmMq3G256V2,
        DType::MQ2G256V2 => KernelKey::GemmMq2G256V2,
        _ => return None,
    })
}

/// Residual-GEMM key for a non-Q8 projection. `residual_gemm_key_for` only
/// knows the V2 family + defaults to HFQ4, so the HFQ6/HFQ3 containers and
/// the rotated v1 dtypes that share their payloads are spelled out here.
fn slots_residual_gemm_key(dt: DType) -> Option<KernelKey> {
    Some(match dt {
        DType::MQ4G256V2 => KernelKey::GemmMq4G256V2Residual,
        DType::MQ4CG256 => KernelKey::GemmMq4CG256Residual,
        DType::MQ6G256V2 => KernelKey::GemmMq6G256V2Residual,
        DType::MQ5G256V2 => KernelKey::GemmMq5G256V2Residual,
        DType::MQ3G256V2 => KernelKey::GemmMq3G256V2Residual,
        DType::MQ2G256V2 => KernelKey::GemmMq2G256V2Residual,
        DType::MQ4G256 | DType::HFQ4G256 => KernelKey::GemmHfq4G256Residual,
        DType::MQ6G256 | DType::HFQ6G256 => KernelKey::GemmHfq6G256Residual,
        DType::MQ3G256 => KernelKey::GemmHfq3G256Residual,
        DType::MQ3G256Lloyd => KernelKey::GemmMq3G256LloydResidual,
        _ => return None,
    })
}

/// Validate one weight for the residual-projection role (wo, w_down):
/// Q8_0, an admitted rotated container, or any admitted container with a
/// residual key. Names the projection in the error.
fn slots_check_residual_weight(w: &WeightTensor, name: &str, arch: &str) -> HipResult<()> {
    let dt = w.gpu_dtype;
    if dt == DType::Q8_0
        || (slots_proj_admissible(dt, arch)
            && (slots_weight_rotated(dt) || slots_residual_gemm_key(dt).is_some()))
    {
        return Ok(());
    }
    Err(HipError::new(
        0,
        &format!(
            "forward_batch_slots: {name} dtype {dt:?} is not admitted by the multi-slot \
             dense path on {arch} (residual role needs Q8_0, a rotated MQ-family container, \
             or a container with a residual GEMM key)"
        ),
    ))
}

/// Plan one projection group (see [`GroupPlan`]) from dtype+AWQ summaries —
/// the pure core, unit-testable without GPU buffers. Uniform groups need
/// every member admitted; mixed groups additionally need a plain key per
/// member and refuse AWQ sidecars (each AWQ weight needs its own rotated
/// input).
fn plan_proj_group_dtypes(members: &[(DType, bool)], arch: &str) -> Result<GroupPlan, String> {
    for (dt, _) in members {
        if !slots_proj_admissible(*dt, arch) {
            return Err(format!(
                "weight dtype {dt:?} is not admitted by the multi-slot dense path on {arch}"
            ));
        }
    }
    let d0 = members[0].0;
    if members.iter().all(|(dt, _)| *dt == d0) {
        return Ok(GroupPlan::Uniform(d0));
    }
    if members.iter().any(|(_, awq)| *awq) {
        return Err(
            "mixes dtypes and carries an AWQ sidecar — mixed-dtype AWQ groups are \
             refused (each scale needs its own rotated input)"
                .to_string(),
        );
    }
    for (dt, _) in members {
        if slots_plain_gemm_key(*dt).is_none() {
            return Err(format!(
                "mixes dtypes and {dt:?} has no per-projection GEMM key \
                 (uniform-only container) — quantize the whole group to one dtype"
            ));
        }
    }
    Ok(GroupPlan::Mixed {
        needs_rot: members.iter().any(|(dt, _)| slots_weight_rotated(*dt)),
        needs_norm: members.iter().any(|(dt, _)| !slots_weight_rotated(*dt)),
    })
}

/// Weight-bearing wrapper: names the group in the error and reads the AWQ
/// sidecar off each `WeightTensor`.
fn plan_proj_group(
    weights: &[&WeightTensor],
    group: &str,
    arch: &str,
) -> HipResult<GroupPlan> {
    let members: Vec<(DType, bool)> = weights
        .iter()
        .map(|w| (w.gpu_dtype, w.awq_scale.is_some()))
        .collect();
    plan_proj_group_dtypes(&members, arch)
        .map_err(|why| HipError::new(0, &format!("forward_batch_slots: {group} {why}")))
}

/// Layer-level admission preflight over the per-site machinery: every fused
/// group of the layer must plan (uniform OR mixed) and every residual role
/// must be admitted — exactly the checks the slot runners perform per step,
/// hoisted so load-time callers (the VMM executor's preflight) can refuse
/// before any GPU work. The Q8 WMMA fold is admission-neutral (it only
/// converts an admitted Uniform(Q8_0) into an admitted Mixed) and is
/// therefore not applied here.
pub(crate) fn require_batchable_deltanet_layer(
    layer: &DeltaNetLayerWeights,
    arch: &str,
) -> HipResult<()> {
    plan_proj_group(
        &[&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha],
        "DeltaNet qkvza",
        arch,
    )
    .map(|_| ())?;
    slots_check_residual_weight(&layer.wo, "DeltaNet wo", arch)?;
    plan_proj_group(&[&layer.w_gate, &layer.w_up], "dense FFN gate_up", arch).map(|_| ())?;
    slots_check_residual_weight(&layer.w_down, "dense FFN w_down", arch)
}

/// FullAttention twin of [`require_batchable_deltanet_layer`].
pub(crate) fn require_batchable_fullattn_layer(
    layer: &FullAttnLayerWeights,
    arch: &str,
) -> HipResult<()> {
    plan_proj_group(&[&layer.wq, &layer.wk, &layer.wv], "FullAttn qkv", arch).map(|_| ())?;
    slots_check_residual_weight(&layer.wo, "FullAttn wo", arch)?;
    plan_proj_group(&[&layer.w_gate, &layer.w_up], "dense FFN gate_up", arch).map(|_| ())?;
    slots_check_residual_weight(&layer.w_down, "dense FFN w_down", arch)
}

/// MoE DeltaNet attention-side preflight (the MoE FFN has its own shared
/// gate, `require_batchable_moe_ffn`, which callers chain). Takes `arch`
/// unlike the pre-per-site version: the planners consult `is_batchable_la`.
pub(crate) fn require_batchable_deltanet_moe_layer(
    layer: &DeltaNetMoeLayerWeights,
    arch: &str,
) -> HipResult<()> {
    plan_proj_group(
        &[&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha],
        "DeltaNetMoE qkvza",
        arch,
    )
    .map(|_| ())?;
    slots_check_residual_weight(&layer.wo, "DeltaNetMoE wo", arch)
}

/// MoE FullAttention twin of [`require_batchable_deltanet_moe_layer`].
pub(crate) fn require_batchable_fullattn_moe_layer(
    layer: &FullAttnMoeLayerWeights,
    arch: &str,
) -> HipResult<()> {
    plan_proj_group(&[&layer.wq, &layer.wk, &layer.wv], "FullAttnMoE qkv", arch).map(|_| ())?;
    slots_check_residual_weight(&layer.wo, "FullAttnMoE wo", arch)
}

/// Post-process a fused-group plan for the Q8 WMMA fork: a uniform Q8_0
/// group on a non-WMMA arch (or with `HIPFIRE_Q8_PREFILL_WMMA=0`, which
/// `q8_prefill_wmma_enabled` folds into `q8_wmma_arch`) must NOT take the
/// WMMA-only `FusedQkvzaQ8_0`/`FusedQkvQ8_0` kernels. Degrade it to the
/// Mixed machinery, which is the reference's plain fork for that case: one
/// `GemmQ8_0BatchedChunked` per weight over the normed input
/// (prefill.rs forks `is_q8 && q8_wmma_arch` the same way at every qkvza /
/// qkv site).
fn plan_q8_fused_or_plain(plan: GroupPlan, q8_wmma_arch: bool) -> GroupPlan {
    if !q8_wmma_arch && plan == GroupPlan::Uniform(DType::Q8_0) {
        GroupPlan::Mixed {
            needs_rot: false,
            needs_norm: true,
        }
    } else {
        plan
    }
}

/// Populate the norm-site activation buffers a projection site needs:
/// `x_norm` gets the rmsnorm output when any consumer is unrotated,
/// `x_rot` gets the FWHT-rotated normed rows when any consumer is a
/// rotated container. Uniform rotated sites keep the fused
/// rmsnorm+rotate producer (bit-identical to the reference); mixed sites
/// rotate the normed rows afterward (rmsnorm math is identical; only the
/// kernel fusion differs).
#[allow(clippy::too_many_arguments)]
fn slots_norm_site(
    gpu: &mut Gpu,
    x: &GpuTensor,
    norm: &GpuTensor,
    rot_anchor: Option<&WeightTensor>,
    x_rot: &GpuTensor,
    x_norm: &GpuTensor,
    need_rot: bool,
    need_norm: bool,
    dim: usize,
    eps: f32,
    n: usize,
) -> HipResult<()> {
    if need_norm {
        gpu.rmsnorm_batched(x, norm, x_norm, n, dim, eps)?;
    }
    if need_rot {
        let anchor = rot_anchor.ok_or_else(|| {
            HipError::new(0, "slots_norm_site: need_rot without a rotated weight")
        })?;
        if need_norm {
            rotate_x_mq_batched_for(gpu, anchor, x_norm, x_rot, dim, n)?;
        } else {
            fused_rmsnorm_rotate_mq_batched_for(gpu, x, norm, anchor, x_rot, dim, eps, n)?;
        }
    }
    Ok(())
}

/// Per-weight GEMM for a Mixed group: pick the activation variant from the
/// consuming weight's dtype (rotated containers read `x_rot`, everything
/// else reads `x_norm`). Plan time has proven the plain key exists.
fn slots_plain_proj(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_rot: &GpuTensor,
    x_norm: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let x = if slots_weight_rotated(w.gpu_dtype) { x_rot } else { x_norm };
    run_plain_gemm_key(
        gpu,
        slots_plain_gemm_key(w.gpu_dtype).expect("mixed group members have plain keys"),
        &w.buf,
        w.gpu_dtype,
        x,
        y,
        w.m,
        w.k,
        n,
    )
}

/// `y += w · x`, dispatching by `w.gpu_dtype` alone: Q8_0 residual path,
/// rotated containers rotate `x` into `rot_scratch` first, plain
/// containers project `x` directly. Replaces the layer-uniform arm the
/// old `AttnProjDtype` forced.
#[allow(clippy::too_many_arguments)]
fn slots_residual_proj(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rot_scratch: &GpuTensor,
    q8_scratch: &GpuTensor,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    let dt = w.gpu_dtype;
    if dt == DType::Q8_0 {
        return q8_residual_proj(gpu, w, x, y, q8_scratch, n, q8_wmma_arch);
    }
    if slots_weight_rotated(dt) {
        return mq4_residual_proj(gpu, w, x, y, rot_scratch, n);
    }
    let key = slots_residual_gemm_key(dt).ok_or_else(|| {
        HipError::new(
            0,
            &format!("forward_batch_slots: residual projection dtype {dt:?} has no residual key"),
        )
    })?;
    let y_n = y.sub_offset(0, n * w.m);
    run_residual_gemm_key(gpu, key, &w.buf, dt, x, &y_n, w.m, w.k, n)
}



// ── Kernel-key selection by weight CONTAINER, never hardcoded ──────────────
//
// qt=6 (HFQ4G256) and qt=13 (MQ4G256) share one container: 136 B groups holding
// `[0..4) f32 scale, [4..8) f32 zero` over all 256 weights, then 128 B of nibbles.
// MQ4 is that container plus an offline FWHT, so it has always borrowed HFQ4's
// kernel keys and that is correct — only the activations differ.
//
// qt=44 (MQ4G256V2) is a DIFFERENT container. Same 136 B stride, same nibble
// payload at the same offset, but the 8 header bytes become
// `[0..2) fp16 s0, [2..4) fp16 z0, [4..6) fp16 s1, [6..8) fp16 z1`, with s0/z0
// governing weights 0..127 and s1/z1 governing 128..255.
//
// Hand a v1 key a qt=44 weight and the kernel bit_casts an fp16 pair to f32,
// yielding ~1e-14: every weight in the tensor collapses to numerically zero. It
// cannot fail — every bit pattern is a valid finite f32, the nibbles are read
// correctly, and stride/alignment/K%256 are identical — so it runs at full speed
// and returns noise. That cost two full KLD measurement cycles (WT2 12.137559
// against a 0.043776 baseline, bit-identical across both runs) before it was
// found. Select on the container; never hardcode.

pub(crate) fn fused_qkvza_key_for(dt: DType) -> KernelKey {
    match dt {
        DType::Q8_0 => KernelKey::FusedQkvzaQ8_0,
        DType::MQ4G256V2 => KernelKey::FusedQkvzaMq4G256V2,
        DType::MQ4CG256 => KernelKey::FusedQkvzaMq4CG256,
        DType::MQ6G256V2 => KernelKey::FusedQkvzaMq6G256V2,
        DType::MQ5G256V2 => KernelKey::FusedQkvzaMq5G256V2,
        DType::MQ3G256V2 => KernelKey::FusedQkvzaMq3G256V2,
        DType::MQ2G256V2 => KernelKey::FusedQkvzaMq2G256V2,
        // qt=15/qt=8 are the 200 B/group 6-bit container: an HFQ4 key would
        // read them at the 136 B HFQ4 stride and return noise at full speed.
        DType::MQ6G256 | DType::HFQ6G256 => KernelKey::FusedQkvzaHfq6G256,
        DType::MQ3G256 => KernelKey::FusedQkvzaHfq3G256,
        DType::MQ3G256Lloyd => KernelKey::FusedQkvzaMq3G256Lloyd,
        // qt=52 must NEVER alias a uniform fused key: no fused LUT kernel exists.
        DType::MQ4G256V2Lloyd => panic!(
            "fused_qkvza_key_for: MQ4G256V2Lloyd (qt=52) has no fused key — route Lloyd prefill through gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd"
        ),
        _ => KernelKey::FusedQkvzaHfq4G256,
    }
}

pub(crate) fn fused_qkv_key_for(dt: DType) -> KernelKey {
    match dt {
        DType::Q8_0 => KernelKey::FusedQkvQ8_0,
        DType::MQ4G256V2 => KernelKey::FusedQkvMq4G256V2,
        DType::MQ4CG256 => KernelKey::FusedQkvMq4CG256,
        DType::MQ6G256V2 => KernelKey::FusedQkvMq6G256V2,
        DType::MQ5G256V2 => KernelKey::FusedQkvMq5G256V2,
        DType::MQ3G256V2 => KernelKey::FusedQkvMq3G256V2,
        DType::MQ2G256V2 => KernelKey::FusedQkvMq2G256V2,
        DType::MQ6G256 | DType::HFQ6G256 => KernelKey::FusedQkvHfq6G256,
        DType::MQ3G256 => KernelKey::FusedQkvHfq3G256,
        DType::MQ3G256Lloyd => KernelKey::FusedQkvMq3G256Lloyd,
        // qt=52 must NEVER alias a uniform fused key: no fused LUT kernel exists.
        DType::MQ4G256V2Lloyd => panic!(
            "fused_qkv_key_for: MQ4G256V2Lloyd (qt=52) has no fused key — route Lloyd prefill through gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd"
        ),
        _ => KernelKey::FusedQkvHfq4G256,
    }
}



/// MoE-FFN weight-dtype admissibility gate, delegating to the reference's own
/// `moe_ffn_batched_admissible` (same predicate `prefill_batch_pbs_eligible`
/// checks before ever entering `forward_prefill_chunk`'s MoE branches) so
/// this file's notion of "batchable MoE FFN" can never drift from the
/// function it is about to call (`prefill_moe_ffn_body_batched`). `admit_mq6`
/// is computed identically to the reference's own call site
/// (`prefill_batch_pbs_eligible`, qwen35.rs) via the same env-keyed helper.
fn require_batchable_moe_ffn(gpu: &Gpu, ffn: &MoeFfnWeights) -> HipResult<()> {
    let arch = gpu.arch.as_str();
    let admit_mq6 = mq6_batched_admit_enabled_from_env(
        hipfire_config::developer_var("HIPFIRE_MOE_MQ6_ADMIT")
            .ok()
            .as_deref(),
        arch,
    );
    let dtypes = moe_prefill_dtypes(ffn).ok_or_else(|| {
        HipError::new(
            0,
            "forward_batch_slots: MoE FFN dtype metadata is unavailable",
        )
    })?;
    if gated_moe_prefill_admissible(&dtypes, admit_mq6, arch) {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            "forward_batch_slots: MoE FFN weight dtypes are not admissible for \
             the batched prefill path (see gated_moe_prefill_admissible); this \
             file requires the same admission the reference's own \
             prefill_batch_pbs_eligible checks before entering the batched MoE \
             branches",
        ))
    }
}

/// FWHT-rotated residual projection: `y[0..n*m] += w · FWHT(x[0..n*k])` for
/// any rotated container (`slots_weight_rotated`). The residual key MUST come
/// from `slots_residual_gemm_key`, NOT `residual_gemm_key_for`: the dispatch
/// helper only knows the V2 family and defaults everything else to
/// `GemmHfq4G256Residual`, which silently mis-decodes the 200 B/group
/// MQ6G256 and 104 B/group MQ3G256/MQ3G256Lloyd headers (see the
/// container-selection note below). `scratch` is the caller's dead buffer to
/// rotate into (mirrors `pbs.dn_normed_rot_batch` / `pbs.fa_attn_out_rot_batch`
/// reuse in the dense Q8 path's `q8_residual_proj`).
fn mq4_residual_proj(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    rotate_x_mq_batched_for(gpu, w, x, scratch, w.k, n)?;
    let y_n = y.sub_offset(0, n * w.m);
    let key = slots_residual_gemm_key(w.gpu_dtype).ok_or_else(|| {
        HipError::new(
            0,
            &format!(
                "forward_batch_slots: rotated residual dtype {:?} has no residual key",
                w.gpu_dtype
            ),
        )
    })?;
    run_residual_gemm_key(
        gpu,
        key,
        &w.buf,
        w.gpu_dtype,
        scratch,
        &y_n,
        w.m,
        w.k,
        n,
    )
}

/// `y[0..n*m] += w · x[0..n*k]`, dispatched through the same
/// `GemmQ8_0ResidualWmma` / `GemmQ8_0BatchedChunked`+`add_inplace_f32`
/// fork the dense batched-prefill path uses for every Q8 residual
/// projection (wo, w_down, and — via a plain non-residual variant below —
/// the lm_head). `scratch` is the non-WMMA fallback's landing buffer for
/// `w · x` before the explicit add; callers pass a buffer that is dead at
/// this point in the layer (mirrors the reference's reuse of
/// `x_rot_batch`/`x_batch` for exactly this purpose).
#[allow(clippy::too_many_arguments)]
fn q8_residual_proj(
    gpu: &mut Gpu,
    w: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    scratch: &GpuTensor,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    if q8_wmma_arch {
        let y_n = y.sub_offset(0, n * w.m);
        run_residual_gemm_key(
            gpu,
            KernelKey::GemmQ8_0ResidualWmma,
            &w.buf,
            w.gpu_dtype,
            x,
            &y_n,
            w.m,
            w.k,
            n,
        )
    } else {
        let s = scratch.sub_offset(0, n * w.m);
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w.buf,
            w.gpu_dtype,
            x,
            &s,
            w.m,
            w.k,
            n,
        )?;
        let y_n = y.sub_offset(0, n * w.m);
        gpu.add_inplace_f32(&y_n, &s)
    }
}

/// Q8 gate+up FFN projection, WMMA-fused when available else two plain
/// batched GEMMs. Mirrors the `ffn_is_q8` fork of the dense path.
#[allow(clippy::too_many_arguments)]
fn q8_gate_up_proj(
    gpu: &mut Gpu,
    w_gate: &hipfire_runtime::llama::WeightTensor,
    w_up: &hipfire_runtime::llama::WeightTensor,
    x: &GpuTensor,
    y_gate: &GpuTensor,
    y_up: &GpuTensor,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    if q8_wmma_arch {
        run_fused_gate_up_key(
            gpu,
            KernelKey::FusedGateUpQ8_0,
            &w_gate.buf,
            &w_up.buf,
            x,
            y_gate,
            y_up,
            w_gate.m,
            w_up.m,
            w_gate.k,
            n,
        )
    } else {
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w_gate.buf,
            w_gate.gpu_dtype,
            x,
            y_gate,
            w_gate.m,
            w_gate.k,
            n,
        )?;
        run_plain_gemm_key(
            gpu,
            KernelKey::GemmQ8_0BatchedChunked,
            &w_up.buf,
            w_up.gpu_dtype,
            x,
            y_up,
            w_up.m,
            w_up.k,
            n,
        )
    }
}

/// Dense FFN body, dispatched per site from the gate/up group's dtypes and
/// `w_down`'s own dtype:
///
///   * gate/up Uniform rotated — `fused_rmsnorm_rotate_mq_batched_for` +
///     one fused gate+up launch. Uniform unrotated (HFQ4G256) — separate
///     rmsnorm + fused launch. Uniform Q8_0 — `q8_gate_up_proj`.
///   * gate/up Mixed — rmsnorm (+rotate when a member needs it) then one
///     plain GEMM per weight, each reading the variant its dtype wants.
///   * w_down — rotated: `fused_silu_mul_rotate_mq_batched_for` + the
///     container's residual key. The rotate is FUSED INTO the SwiGLU, so
///     this must NOT route through `mq4_residual_proj` (which rotates its
///     own input). Unrotated non-Q8: plain silu_mul + `slots_residual_gemm_
///     key`. Q8: `q8_residual_proj`.
#[allow(clippy::too_many_arguments)]
fn dense_ffn_body_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    ffn_norm: &GpuTensor,
    w_gate: &WeightTensor,
    w_up: &WeightTensor,
    w_down: &WeightTensor,
    pbs: &PrefillBatchScratch,
    n: usize,
    q8_wmma_arch: bool,
) -> HipResult<()> {
    let arch = gpu.arch.as_str();
    let gu_plan = plan_proj_group(&[w_gate, w_up], "dense FFN gate_up", arch)?;
    slots_check_residual_weight(w_down, "dense FFN w_down", arch)?;

    let (need_rot, need_norm) = match gu_plan {
        GroupPlan::Uniform(dt) => (slots_weight_rotated(dt), !slots_weight_rotated(dt)),
        GroupPlan::Mixed { needs_rot, needs_norm } => (needs_rot, needs_norm),
    };
    let gu_anchor = if let GroupPlan::Mixed { .. } = gu_plan {
        [w_gate, w_up]
            .iter()
            .find(|w| slots_weight_rotated(w.gpu_dtype))
            .copied()
    } else {
        slots_weight_rotated(w_gate.gpu_dtype).then_some(w_gate)
    };
    slots_norm_site(
        gpu,
        &pbs.x_batch,
        ffn_norm,
        gu_anchor,
        &pbs.x_rot_batch,
        &pbs.x_norm_batch,
        need_rot,
        need_norm,
        config.dim,
        config.norm_eps,
        n,
    )?;

    match gu_plan {
        GroupPlan::Uniform(DType::Q8_0) => q8_gate_up_proj(
            gpu,
            w_gate,
            w_up,
            &pbs.x_norm_batch,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            n,
            q8_wmma_arch,
        )?,
        GroupPlan::Uniform(dt) => run_fused_gate_up_key(
            gpu,
            fused_gate_up_key_for(dt),
            &w_gate.buf,
            &w_up.buf,
            if slots_weight_rotated(dt) { &pbs.x_rot_batch } else { &pbs.x_norm_batch },
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            w_gate.m,
            w_up.m,
            w_gate.k,
            n,
        )?,
        GroupPlan::Mixed { .. } => {
            let g = pbs.gate_ffn_batch.sub_offset(0, n * w_gate.m);
            slots_plain_proj(gpu, w_gate, &pbs.x_rot_batch, &pbs.x_norm_batch, &g, n)?;
            let u = pbs.up_batch.sub_offset(0, n * w_up.m);
            slots_plain_proj(gpu, w_up, &pbs.x_rot_batch, &pbs.x_norm_batch, &u, n)?;
        }
    }

    // w_down + residual — dispatched by w_down's own dtype. For a rotated
    // container the input rotate is fused into the SwiGLU; the x_rot/x_norm
    // buffers are free to reuse as rotate scratch here.
    let dt = w_down.gpu_dtype;
    if slots_weight_rotated(dt) {
        fused_silu_mul_rotate_mq_batched_for(
            gpu,
            w_down,
            &pbs.gate_ffn_batch,
            &pbs.up_batch,
            &pbs.ffn_hidden_batch,
            w_down.k,
            n,
        )?;
        let key = slots_residual_gemm_key(dt).expect("rotated w_down has a residual key");
        let y = pbs.x_batch.sub_offset(0, n * w_down.m);
        run_residual_gemm_key(
            gpu,
            key,
            &w_down.buf,
            dt,
            &pbs.ffn_hidden_batch,
            &y,
            w_down.m,
            w_down.k,
            n,
        )
    } else {
        gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
        slots_residual_proj(
            gpu,
            w_down,
            &pbs.ffn_hidden_batch,
            &pbs.x_batch,
            &pbs.x_rot_batch,
            &pbs.x_rot_batch,
            n,
            q8_wmma_arch,
        )
    }
}

/// Per-layer `(qkv, alpha, beta)` capture of a step's DeltaNet activations
/// for a slot's spec verify rows — the tape the post-accept DN repair
/// (`mtp_dn_repair_from_tape`) replays conv+GDN from.
///
/// The `pbs.dn_*` batch buffers are per-LAYER scratch: each DeltaNet layer
/// overwrites them, so after the forward finishes they hold only the LAST
/// layer's activations. A partial accept needs per-layer activations to
/// rewind the recurrent state over the accepted prefix, so verify steps tape
/// their rows as the layers compute. The tape is indexed per slot at
/// `slot * stride` rows (stride = verify rows per slot: `mtp_k + 1` for MTP,
/// `B` for DFlash2), independent of the batch row layout, so at most
/// `n_slots * stride` rows are ever taped — a few MB.
pub struct SpecVerifyCapture<'a> {
    pub tape: &'a mut crate::speculative::GdnTape,
    /// `verify_slots[s]`: slot s's rows this step are spec verify rows.
    pub verify_slots: &'a [bool],
    /// Rows taped per flagged slot (verify rows per slot).
    pub stride: usize,
    /// DFlash2 extract-layer hidden capture. When `Some`, every layer whose
    /// index is in `extract_layers` copies the step's post-layer `x_batch`
    /// rows into `staging[extract_idx]` at offset 0 (fixed offsets — safe
    /// inside graph capture). The engine scatters staging into each DFlash
    /// slot's `target_hidden` ring after the forward returns. `None` on
    /// steps with no DFlash rows (zero cost).
    pub hidden: Option<SpecHiddenCapture<'a>>,
}

/// Extract-layer hidden capture for DFlash2 slots: `staging[i]` receives
/// `n` rows of `x_batch` whenever the layer index equals `extract_layers[i]`.
/// One shared staging set covers the whole step (rows are slot-contiguous in
/// `x_batch`); the engine maps each slot's row range onto its own ring.
pub struct SpecHiddenCapture<'a> {
    pub staging: &'a [GpuTensor],
    pub extract_layers: &'a [usize],
    /// Row stride of `x_batch` (= model dim).
    pub dim: usize,
}

impl SpecVerifyCapture<'_> {
    #[inline]
    pub fn is_verify_slot(&self, slot: usize) -> bool {
        self.verify_slots.get(slot).copied().unwrap_or(false)
    }

    /// Copy this step's post-layer hidden rows into the DFlash2 staging
    /// buffer when `layer_idx` is an extract layer. Called once per layer
    /// from the main dispatch loop — the copy is a single contiguous D2D
    /// (all rows of the step), so it is capture-safe under hipGraph.
    #[inline]
    fn capture_hidden_rows(
        &self,
        gpu: &mut Gpu,
        pbs: &PrefillBatchScratch,
        layer_idx: usize,
        n: usize,
    ) -> HipResult<()> {
        let Some(h) = &self.hidden else { return Ok(()) };
        let Some(ext) = h.extract_layers.iter().position(|&l| l == layer_idx) else {
            return Ok(());
        };
        let row_bytes = h.dim * 4;
        gpu.hip
            .memcpy_dtod_at(&h.staging[ext].buf, 0, &pbs.x_batch.buf, 0, n * row_bytes)
    }
}

/// Tape one slot's `(qkv, alpha, beta)` rows for `delta_layer_idx` into
/// `cap` at the slot's stride offset. Three small D2D copies; the taped
/// values are exactly what `replay_gdn`-style repair needs to re-run
/// conv1d + qk-norm + GDN for those rows.
#[allow(clippy::too_many_arguments)]
fn spec_tape_layer_rows(
    gpu: &mut Gpu,
    cap: &mut SpecVerifyCapture,
    pbs: &PrefillBatchScratch,
    delta_layer_idx: usize,
    row_off: usize,
    m: usize,
    slot: usize,
    qkv_dim: usize,
    n_v_heads: usize,
) -> HipResult<()> {
    debug_assert!(
        m <= cap.stride,
        "verify rows {m} exceed tape stride {}",
        cap.stride
    );
    let tape_off = slot * cap.stride;
    let tape = &mut *cap.tape;
    gpu.memcpy_dtod_at_auto(
        &tape.qkv_bufs[delta_layer_idx].buf,
        tape_off * qkv_dim * 4,
        &pbs.dn_qkv_batch.buf,
        row_off * qkv_dim * 4,
        m * qkv_dim * 4,
    )?;
    gpu.memcpy_dtod_at_auto(
        &tape.alpha_bufs[delta_layer_idx].buf,
        tape_off * n_v_heads * 4,
        &pbs.dn_alpha_batch.buf,
        row_off * n_v_heads * 4,
        m * n_v_heads * 4,
    )?;
    gpu.memcpy_dtod_at_auto(
        &tape.beta_bufs[delta_layer_idx].buf,
        tape_off * n_v_heads * 4,
        &pbs.dn_beta_batch.buf,
        row_off * n_v_heads * 4,
        m * n_v_heads * 4,
    )?;
    Ok(())
}

/// Run one `LinearAttention` (DeltaNet) layer across the whole step.
///
/// The stateless pieces (rmsnorm, the 4-way QKVZA projection,
/// sigmoid/alpha-gate, the Q/K L2-norm, gated-norm, wo+residual, FFN) run
/// once over all `n` rows. The stateful pieces (conv1d, the GDN recurrence)
/// loop per slot with that slot's own state — see the module doc.
#[allow(clippy::too_many_arguments)]
fn run_deltanet_layer_slots<D>(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &DeltaNetLayerWeights,
    batch: &SlotBatch,
    dn_states: &D,
    pbs: &PrefillBatchScratch,
    q8_wmma_arch: bool,
    n: usize,
    delta_layer_idx: usize,
    mut spec_capture: Option<&mut SpecVerifyCapture>,
) -> HipResult<()>
where
    D: std::ops::Index<usize, Output = DeltaNetState> + ?Sized,
{
    let arch = gpu.arch.as_str();
    // Per-site plans: the qkvza group keeps one fused launch when uniform,
    // otherwise per-weight GEMMs; wo and the FFN weights validate their own
    // roles inside their dispatchers.
    let qkvza_plan = plan_q8_fused_or_plain(
        plan_proj_group(
            &[&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha],
            "DeltaNet qkvza",
            arch,
        )?,
        q8_wmma_arch,
    );
    slots_check_residual_weight(&layer.wo, "DeltaNet wo", arch)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // 1-2. rmsnorm (+FWHT-rotate for rotated containers) then the 4-way
    // QKVZA projection. Uniform rotated groups keep the fused rmsnorm+rotate
    // producer; mixed groups normalize once, rotate the normed rows, then
    // run one plain GEMM per weight (each member's dtype picks its input
    // variant — the load-bearing bit: an unrotated member MUST NOT read
    // rotated activations).
    let (need_rot, need_norm) = match qkvza_plan {
        GroupPlan::Uniform(dt) => (slots_weight_rotated(dt), !slots_weight_rotated(dt)),
        GroupPlan::Mixed { needs_rot, needs_norm } => (needs_rot, needs_norm),
    };
    let qkvza_anchor = if let GroupPlan::Uniform(dt) = qkvza_plan {
        slots_weight_rotated(dt).then_some(&layer.wqkv)
    } else {
        [&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha]
            .iter()
            .find(|w| slots_weight_rotated(w.gpu_dtype))
            .copied()
    };
    slots_norm_site(
        gpu,
        &pbs.x_batch,
        &layer.attn_norm,
        qkvza_anchor,
        &pbs.x_rot_batch,
        &pbs.x_norm_batch,
        need_rot,
        need_norm,
        config.dim,
        config.norm_eps,
        n,
    )?;
    match qkvza_plan {
        GroupPlan::Uniform(DType::Q8_0) if q8_wmma_arch => {
            run_fused_qkvza_key(
                gpu,
                KernelKey::FusedQkvzaQ8_0,
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_norm_batch,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
            )?;
        }
        GroupPlan::Uniform(dt) => {
            let x = if slots_weight_rotated(dt) { &pbs.x_rot_batch } else { &pbs.x_norm_batch };
            run_fused_qkvza_key(
                gpu,
                fused_qkvza_key_for(dt),
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                x,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
            )?;
        }
        GroupPlan::Mixed { .. } => {
            for (w, out) in [
                (&layer.wqkv, &pbs.dn_qkv_batch),
                (&layer.wz, &pbs.dn_z_batch),
                (&layer.w_beta, &pbs.dn_beta_batch),
                (&layer.w_alpha, &pbs.dn_alpha_batch),
            ] {
                let y = out.sub_offset(0, n * w.m);
                slots_plain_proj(gpu, w, &pbs.x_rot_batch, &pbs.x_norm_batch, &y, n)?;
            }
        }
    }


    // 3. Fused sigmoid(beta) + alpha_gate(alpha) — stateless, batched over N.
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;

    // 4. conv1d — STATEFUL (ring buffer), looped per slot.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if let Some(cap) = spec_capture.as_deref_mut() {
                if cap.is_verify_slot(s) {
                    spec_tape_layer_rows(
                        gpu,
                        cap,
                        pbs,
                        delta_layer_idx,
                        row_off,
                        m,
                        s,
                        qkv_dim,
                        n_v_heads,
                    )?;
                }
            }
            let q_out = pbs.dn_q_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let k_out = pbs.dn_k_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let v_out = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let input = pbs.dn_qkv_batch.sub_offset(row_off * qkv_dim, m * qkv_dim);
            gpu.conv1d_silu_split_f32_n(
                &q_out,
                &k_out,
                &v_out,
                &input,
                &layer.conv_weight,
                &dn_states[s].conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                m,
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 5. Q/K L2-norm(+scale, +repeat-interleave) — stateless, batched over N.
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
    } else {
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_num_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }

    // 6. Gated Delta Net recurrence — STATEFUL (S matrix), looped per slot.
    // Existing, unmodified `gated_delta_net_q8_batch_seq`; each launch
    // advances exactly one slot's own `DeltaNetState`. Do NOT use
    // `gated_delta_net_q8_batch_seq_slots` — it now asserts
    // `s_stride_elems == 0` (the stride design was found unsound; see
    // 8fb0e38e). No stride is needed here: one launch per slot already
    // addresses that slot's state directly.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if !matches!(dn_states[s].quant, StateQuant::Q8) {
                return Err(HipError::new(
                    0,
                    "forward_batch_slots: DeltaNetState must be StateQuant::Q8 \
                     (the multi-slot batched path is Q8-only)",
                ));
            }
            let q_view = pbs.dn_q_batch.sub_offset(row_off * v_dim, m * v_dim);
            let k_view = pbs.dn_k_batch.sub_offset(row_off * v_dim, m * v_dim);
            let v_view = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let gate_view = pbs
                .dn_alpha_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs
                .dn_beta_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let out_view = pbs.dn_attn_out_batch.sub_offset(row_off * v_dim, m * v_dim);
            gpu.gated_delta_net_q8_batch_seq(
                &q_view,
                &k_view,
                &v_view,
                &gate_view,
                &beta_view,
                &dn_states[s].s_matrices[delta_layer_idx],
                &dn_states[s].s_scales[delta_layer_idx],
                &out_view,
                m,
                n_v_heads,
                config.linear_value_head_dim,
                dn_states[s].ef_residual(delta_layer_idx),
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 7. Batched gated output norm — stateless, batched over N.
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;

    // 8. wo + residual — dispatched by wo's own dtype (Q8 residual, rotated
    // containers rotate into dn_normed_rot first, HFQ* containers direct).
    slots_residual_proj(
        gpu,
        &layer.wo,
        &pbs.dn_normed_batch,
        &pbs.x_batch,
        &pbs.dn_normed_rot_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 9. FFN: rmsnorm, gate+up, silu_mul, w_down + residual — dispatched per
    // site inside the body.
    dense_ffn_body_slots(
        gpu,
        config,
        &layer.ffn_norm,
        &layer.w_gate,
        &layer.w_up,
        &layer.w_down,
        pbs,
        n,
        q8_wmma_arch,
    )
}

/// Run one `LinearAttention` (DeltaNet) + MoE layer across the whole step.
///
/// Same shape as [`run_deltanet_layer_slots`] — the stateless attention
/// pieces run once over all `n` rows, the stateful pieces (conv1d, GDN) loop
/// per slot — except: (a) the QKVZA projection and wo admit the same
/// per-site container set as the dense body (see `plan_proj_group`), and (b) the dense
/// FFN (rmsnorm+gate/up+silu_mul+down) is replaced by a call to the
/// reference's own `prefill_moe_ffn_body_batched`. The MoE FFN is stateless
/// per row — it takes no `kv_cache`, `dn_state`, or `positions` — so no
/// slot-aware variant is needed: the flat `[n × dim]` `pbs.x_batch` this
/// function already threads through the attention body is exactly the input
/// shape `prefill_moe_ffn_body_batched` expects.
#[allow(clippy::too_many_arguments)]
fn run_deltanet_moe_layer_slots<D>(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &DeltaNetMoeLayerWeights,
    batch: &SlotBatch,
    dn_states: &D,
    pbs: &PrefillBatchScratch,
    q8_wmma_arch: bool,
    n: usize,
    delta_layer_idx: usize,
    mut spec_capture: Option<&mut SpecVerifyCapture>,
    // Whole-MODEL flag (not per-layer): true when ANY MoE layer anywhere in
    // the model has an MQ6 FFN projection. Threaded straight through to
    // `prefill_moe_ffn_body_batched`'s `model_has_mq6_moe` — see that
    // parameter's use at qwen35.rs:8488 (`force_mq4_grouped_fp16`), a
    // cross-layer kernel-selection consistency knob on gfx1151. Passing this
    // layer's own (uniform-MQ4G256, per `require_batchable_moe_ffn`) dtype
    // instead of the model-wide flag would silently diverge from the
    // reference whenever the SAME model mixes an MQ6 layer elsewhere.
    weights_moe_has_mq6: bool,
) -> HipResult<()>
where
    D: std::ops::Index<usize, Output = DeltaNetState> + ?Sized,
{
    let arch = gpu.arch.as_str();
    let qkvza_plan = plan_q8_fused_or_plain(
        plan_proj_group(
            &[&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha],
            "DeltaNetMoE qkvza",
            arch,
        )?,
        q8_wmma_arch,
    );
    slots_check_residual_weight(&layer.wo, "DeltaNetMoE wo", arch)?;
    require_batchable_moe_ffn(gpu, &layer.ffn)?;

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // 1-2. rmsnorm(+FWHT-rotate for rotated containers) then the 4-way
    // QKVZA projection — same group-plan rules as the dense DeltaNet body.
    let (need_rot, need_norm) = match qkvza_plan {
        GroupPlan::Uniform(dt) => (slots_weight_rotated(dt), !slots_weight_rotated(dt)),
        GroupPlan::Mixed { needs_rot, needs_norm } => (needs_rot, needs_norm),
    };
    let qkvza_anchor = if let GroupPlan::Uniform(dt) = qkvza_plan {
        slots_weight_rotated(dt).then_some(&layer.wqkv)
    } else {
        [&layer.wqkv, &layer.wz, &layer.w_beta, &layer.w_alpha]
            .iter()
            .find(|w| slots_weight_rotated(w.gpu_dtype))
            .copied()
    };
    slots_norm_site(
        gpu,
        &pbs.x_batch,
        &layer.attn_norm,
        qkvza_anchor,
        &pbs.x_rot_batch,
        &pbs.x_norm_batch,
        need_rot,
        need_norm,
        config.dim,
        config.norm_eps,
        n,
    )?;
    match qkvza_plan {
        GroupPlan::Uniform(DType::Q8_0) if q8_wmma_arch => {
            run_fused_qkvza_key(
                gpu,
                KernelKey::FusedQkvzaQ8_0,
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                &pbs.x_norm_batch,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
            )?;
        }
        GroupPlan::Uniform(dt) => {
            let x = if slots_weight_rotated(dt) { &pbs.x_rot_batch } else { &pbs.x_norm_batch };
            run_fused_qkvza_key(
                gpu,
                fused_qkvza_key_for(dt),
                &layer.wqkv.buf,
                &layer.wz.buf,
                &layer.w_beta.buf,
                &layer.w_alpha.buf,
                x,
                &pbs.dn_qkv_batch,
                &pbs.dn_z_batch,
                &pbs.dn_beta_batch,
                &pbs.dn_alpha_batch,
                layer.wqkv.m,
                layer.wz.m,
                layer.w_beta.m,
                layer.w_alpha.m,
                layer.wqkv.k,
                n,
            )?;
        }
        GroupPlan::Mixed { .. } => {
            for (w, out) in [
                (&layer.wqkv, &pbs.dn_qkv_batch),
                (&layer.wz, &pbs.dn_z_batch),
                (&layer.w_beta, &pbs.dn_beta_batch),
                (&layer.w_alpha, &pbs.dn_alpha_batch),
            ] {
                let y = out.sub_offset(0, n * w.m);
                slots_plain_proj(gpu, w, &pbs.x_rot_batch, &pbs.x_norm_batch, &y, n)?;
            }
        }
    }


    // 3. Fused sigmoid(beta) + alpha_gate(alpha) — stateless, batched over N.
    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n,
    )?;

    // 4. conv1d — STATEFUL (ring buffer), looped per slot. Byte-identical to
    // run_deltanet_layer_slots step 4 — conv_weight is a dequantized F32
    // GpuTensor regardless of the layer's projection weight dtype.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if let Some(cap) = spec_capture.as_deref_mut() {
                if cap.is_verify_slot(s) {
                    spec_tape_layer_rows(
                        gpu,
                        cap,
                        pbs,
                        delta_layer_idx,
                        row_off,
                        m,
                        s,
                        qkv_dim,
                        n_v_heads,
                    )?;
                }
            }
            let q_out = pbs.dn_q_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let k_out = pbs.dn_k_raw_batch.sub_offset(row_off * k_dim, m * k_dim);
            let v_out = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let input = pbs.dn_qkv_batch.sub_offset(row_off * qkv_dim, m * qkv_dim);
            gpu.conv1d_silu_split_f32_n(
                &q_out,
                &k_out,
                &v_out,
                &input,
                &layer.conv_weight,
                &dn_states[s].conv_states[delta_layer_idx],
                k_dim,
                v_dim,
                m,
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 5. Q/K L2-norm(+scale, +repeat-interleave) — stateless, batched over N.
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
    } else {
        gpu.fused_qk_l2_norm_scale_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            config.linear_num_key_heads,
            hd,
            1.0 / (hd as f32).sqrt(),
            config.norm_eps,
            n,
        )?;
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
    }

    // 6. Gated Delta Net recurrence — STATEFUL (S matrix), looped per slot.
    // Same Q8-only per-slot dispatch as run_deltanet_layer_slots step 6 —
    // the GDN recurrence dtype is a DeltaNetState property, independent of
    // this layer's projection weight dtype.
    let mut row_off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            if !matches!(dn_states[s].quant, StateQuant::Q8) {
                return Err(HipError::new(
                    0,
                    "forward_batch_slots: DeltaNetState must be StateQuant::Q8 \
                     (the multi-slot batched path is Q8-only)",
                ));
            }
            let q_view = pbs.dn_q_batch.sub_offset(row_off * v_dim, m * v_dim);
            let k_view = pbs.dn_k_batch.sub_offset(row_off * v_dim, m * v_dim);
            let v_view = pbs.dn_v_batch.sub_offset(row_off * v_dim, m * v_dim);
            let gate_view = pbs
                .dn_alpha_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let beta_view = pbs
                .dn_beta_batch
                .sub_offset(row_off * n_v_heads, m * n_v_heads);
            let out_view = pbs.dn_attn_out_batch.sub_offset(row_off * v_dim, m * v_dim);
            gpu.gated_delta_net_q8_batch_seq(
                &q_view,
                &k_view,
                &v_view,
                &gate_view,
                &beta_view,
                &dn_states[s].s_matrices[delta_layer_idx],
                &dn_states[s].s_scales[delta_layer_idx],
                &out_view,
                m,
                n_v_heads,
                config.linear_value_head_dim,
                dn_states[s].ef_residual(delta_layer_idx),
            )?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, n);

    // 7. Batched gated output norm — stateless, batched over N.
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n,
    )?;

    // 8. wo + residual — dispatched by wo's own dtype.
    slots_residual_proj(
        gpu,
        &layer.wo,
        &pbs.dn_normed_batch,
        &pbs.x_batch,
        &pbs.dn_normed_rot_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 9. Batched MoE FFN replaces the dense (rmsnorm + gate+up + silu_mul +
    // w_down) block — stateless per row, no slot machinery needed. Takes
    // pbs.x_batch as input and accumulates the FFN output residual back
    // into it (same contract as the reference's own call site).
    let ctx = DispatchCtx::new(gpu);
    prefill_moe_ffn_body_batched(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights_moe_has_mq6,
        /*routed_out=*/ None,
    )
}

/// Mirrors the `flash_optin && wmma_ok` gate that
/// `hipfire-dispatch/src/families/attention.rs` applies to
/// `KernelKey::AttnQ8_0KvBatchedMasked` BEFORE it ever reaches the LDS/tiled
/// crossover below — on gfx11xx (RDNA3/3.5) this gate is DEFAULT ON (not an
/// opt-in corner case), so any batched Q8 prefill on that hardware (any
/// `n > 1`) actually runs through `attention_q8_0_flash_prefill_wmma`
/// (f16-accumulate WMMA flash-prefill), never the LDS-backed masked kernel
/// this file's crossover assumed was the whole story. Confirmed empirically
/// (see `sp3-defect-report.md`): forcing `HIPFIRE_FLASH_PREFILL=0` on the
/// reference cuts the golden test's worst-element error from 20.49x to
/// 3.93x tolerance, and a layer-by-layer bisection shows the FIRST nonzero
/// hidden-state divergence appears exactly at the first `FullAttention`
/// layer — both point at this kernel-selection gap, not a per-layer op
/// bug.
///
/// Mirrors ONLY the unconditional gfx11 branch of the reference's
/// `flash_default_on = arch.starts_with("gfx11") || gfx12_query16_route_ok`:
/// the gfx12 disjunct reads three `hipfire-dispatch`-private eligibility
/// predicates (one needs a live `DispatchCtx`) this crate has no visibility
/// into. Restricting to `has_wmma_w32() && !has_wmma_w32_gfx12()` means this
/// NEVER opts in on gfx12 even if `HIPFIRE_FLASH_PREFILL=1` forces the
/// reference on there too — a strict narrowing (documented scope gap, same
/// category as the crossover below already carries for the scalar
/// flash-prefill ladder), not a guess at gfx12's real eligibility window.
/// The WMMA flash-prefill kernel's query-row tile size (its WMMA-fragment
/// tile, not the scalar kernel's tunable BR). Both the eligibility floor and
/// `build_tiles` must use the same value.
const WMMA_M_TILE: usize = 16;

fn q8_flash_prefill_wmma_eligible(gpu: &Gpu, head_dim: usize, batch_size: usize) -> bool {
    // Mirrors the reference's own gate exactly (`batch_size > 1`), and must
    // keep doing so: the single-active-slot reduction below uses this answer
    // to decide whether it may drop into the reference's own WMMA kernel, and
    // any divergence there breaks byte-exactness with the reference arm.
    // The decode-batch floor lives on the multi-slot tile path instead, which
    // has no reference counterpart -- see MULTI_SLOT_TILE_MIN_ROWS.
    if batch_size <= 1 {
        return false;
    }
    let default_on = gpu.arch.starts_with("gfx11");
    let flash_optin = match hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL")
        .ok()
        .as_deref()
    {
        Some("0") | Some("off") | Some("false") => false,
        Some("1") | Some("on") | Some("true") => true,
        _ => default_on,
    };
    if !flash_optin {
        return false;
    }
    let variant = hipfire_config::developer_var("HIPFIRE_FLASH_PREFILL_KERNEL")
        .unwrap_or_else(|_| "wmma".to_owned());
    variant != "scalar"
        && gpu.arch_caps.has_wmma_w32()
        && !gpu.arch_caps.has_wmma_w32_gfx12()
        && head_dim % 32 == 0
        && head_dim <= 256
}

/// Crossover between the LDS-backed masked kernel (no context ceiling issue
/// below the crossover) and the tiled no-LDS-cap flash kernel, mirroring the
/// fallback ladder in `hipfire-dispatch/src/families/attention.rs`'s
/// `AttnQ8_0KvBatchedMasked` arm. Deliberately narrower than that arm in one
/// remaining respect: the scalar flash-prefill opt-in variant (gated by
/// `HIPFIRE_FLASH_PREFILL_KERNEL=scalar` + a long-context minimum) has no
/// `_slots` port, so that one case still falls through to this crossover —
/// SP1 built the two families used here, plus (below) the WMMA flash-prefill
/// family, now ported for BOTH the single-active-slot reduction and the
/// genuinely-multi-slot case (`attention_q8_0_flash_prefill_wmma_slots`).
///
/// `single_slot`, when `Some((k_base, slab_bytes))`, means exactly one slot
/// is active in this step (true for every `n_slots == 1` call, and for a
/// larger pool whenever only one slot has live rows this step). `k_base` is
/// that slot's byte offset into the shared arena. On Q8_0 the K and V
/// per-position strides are equal, so `legacy_v_base == legacy_k_base`;
/// slot 0 of a fresh pool always has `k_base == 0`, so
/// this reduces byte-for-byte to the reference's own `k_cache`/`v_cache`
/// addressing when `n_slots == 1`. Never `Some` for a paged pool — a paged
/// slot's KV is scattered across physical pages, which this reduction's
/// pointer-shifted contiguous view cannot express; the caller gates it off
/// and every paged step takes the descriptor-driven paths below. This path
/// is kept (rather than folded into
/// the general multi-slot path below) because it is strictly cheaper: no
/// tile-array build/upload, no descriptor indirection, no extra kernarg
/// pointers or per-tile table lookups — just the plain legacy kernel against
/// a pointer-shifted view, address-identical to what SP3 verified
/// bit-identical (0.000x tolerance) against the reference.
///
/// `multi_slot_tiles`, when `Some((tile_slot_dev, tile_row0_dev,
/// tile_qbase_dev, n_tiles))`, means MORE than one slot is active this step
/// AND the reference's own gfx11 WMMA-flash-prefill gate
/// (`q8_flash_prefill_wmma_eligible`) would fire for this shape — the exact
/// condition under which the reference's single-sequence dispatch (which has
/// no concept of "slots" at all, just a flat batch) would have run this
/// kernel. `tile_*_dev` are persistent per-step staging buffers (see
/// `SlotDescStaging`) sized to `desc_staging.max_rows`; only their first
/// `n_tiles` entries are valid this step, hence the `sub_offset` views built
/// below (which exist purely to give the launcher's `tile_slot.numel()` grid
/// sizing the correct `n_tiles`, not `max_rows`).
/// Per-tier batched KV write for one FullAttention layer step: K through the
/// tier's packed writer, V through the descriptor-aware Q8_0 writer
/// (resolving `legacy_v_base`, which differs from `legacy_k_base` on every
/// rotated-K tier). Both arenas resolve through the same descriptor table /
/// block tables, so this is correct under the legacy slab pool AND the paged
/// pool. Q8 keeps its two-call form (K and V share one slab offset there —
/// the kernel reads only the K base).
#[allow(clippy::too_many_arguments)]
fn kv_write_slots(
    gpu: &mut Gpu,
    kv: &SlotKvTier,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    k_batch: &GpuTensor,
    v_batch: &GpuTensor,
    positions: &GpuTensor,
    n_kv_heads: usize,
    head_dim: usize,
    n_rows: usize,
    descs: &GpuTensor,
    row_slot: &GpuTensor,
) -> HipResult<()> {
    match kv.mode {
        KvMode::Q8 => {
            gpu.kv_cache_write_q8_0_batched_slots(
                k_cache,
                k_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
                /*use_v_base=*/ false,
            )?;
            gpu.kv_cache_write_q8_0_batched_slots(
                v_cache,
                v_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
                /*use_v_base=*/ false,
            )
        }
        KvMode::Asym2 => gpu.kv_cache_write_asym2_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym2 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym2 tier without givens sin"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        KvMode::Asym3 => gpu.kv_cache_write_asym3_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym3 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym3 tier without givens sin"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        KvMode::Asym4 => gpu.kv_cache_write_asym4_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym4 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym4 tier without givens sin"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        KvMode::Fwht2 => gpu.kv_cache_write_fwht2_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht2 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht2 tier without signs2"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        KvMode::Fwht3 => gpu.kv_cache_write_fwht3_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht3 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht3 tier without signs2"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        KvMode::Fwht4 => gpu.kv_cache_write_fwht4_batched_slots(
            k_cache,
            v_cache,
            k_batch,
            v_batch,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht4 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht4 tier without signs2"),
            n_kv_heads,
            head_dim,
            n_rows,
            Some(descs),
            Some(row_slot),
        ),
        // bf16's writer is per-arena like Q8's (flat layout, K and V share
        // the stride), and the kernel is descriptor-aware.
        KvMode::Bf16 => {
            gpu.kv_cache_write_bf16_batched(
                k_cache,
                k_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )?;
            gpu.kv_cache_write_bf16_batched(
                v_cache,
                v_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )
        }
        // f16 is the same flat layout as bf16 — per-arena writes, K and V
        // share the stride, descriptor-aware kernel.
        KvMode::F16 => {
            gpu.kv_cache_write_f16_batched(
                k_cache,
                k_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )?;
            gpu.kv_cache_write_f16_batched(
                v_cache,
                v_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )
        }
        // Native fp8 (gfx1201-only — Rig::build refuses the tier elsewhere):
        // per-arena writes like Q8/bf16 — the fp8 writer is descriptor-aware,
        // and both arenas share the fp8 row layout (`[Hkv*D codes][Hkv f16
        // scales]`, equal K/V strides), so both launches resolve the shared
        // K base and `dst` alone selects the arena.
        //
        // untested-hw: gfx1201-only path, implemented from the q8 slots
        // template; no gfx1201 host was available.
        KvMode::Fp8 => {
            gpu.kv_cache_write_fp8_e4m3_batched_slots(
                k_cache,
                k_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )?;
            gpu.kv_cache_write_fp8_e4m3_batched_slots(
                v_cache,
                v_batch,
                positions,
                n_kv_heads,
                head_dim,
                n_rows,
                Some(descs),
                Some(row_slot),
            )
        }
    }
}

/// Per-tier batched attend for one FullAttention layer step. q8 keeps the
/// full dispatch ladder (single-slot WMMA prefill fast path, scalar decode
/// below the ctx crossover); every other tier runs its descriptor-driven
/// kernel — the flash tile for the rotated/fp8 tiers, and the SEQUENTIAL
/// path's scalar bf16 batched kernel for bf16 (bit-parity with
/// `AttnBf16KvBatchedMasked`; the windowed flash tile accumulates in a
/// different order and diverged past tolerance in the golden harness).
/// These are the only paths those tiers have: they have no scalar batched
/// decode besides bf16's parity kernel, and the WMMA single-slot reduction
/// is a Q8-slab pointer trick that cannot express a descriptor.
#[allow(clippy::too_many_arguments)]
fn tier_attend_slots(
    gpu: &mut Gpu,
    kv: &SlotKvTier,
    q: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    out: &GpuTensor,
    positions: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    batch_size: usize,
    flash_partials: &GpuTensor,
    descs_dev: &GpuTensor,
    row_slot_dev: &GpuTensor,
    single_slot: Option<(u64, usize)>,
    multi_slot_tiles: Option<(&GpuTensor, &GpuTensor, &GpuTensor, usize)>,
) -> HipResult<()> {
    if kv.is_q8() {
        return q8_attend_slots(
            gpu,
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            descs_dev,
            row_slot_dev,
            single_slot,
            multi_slot_tiles,
        );
    }
    let d = Some(descs_dev);
    let r = Some(row_slot_dev);
    match kv.mode {
        KvMode::Asym2 => gpu.attention_flash_asym2_batched_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym2 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym2 tier without givens sin"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            d,
            r,
        ),
        KvMode::Asym3 => gpu.attention_flash_asym3_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym3 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym3 tier without givens sin"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            d,
            r,
        ),
        KvMode::Asym4 => gpu.attention_flash_asym4_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.givens_cos
                .as_ref()
                .expect("asym4 tier without givens cos"),
            kv.givens_sin
                .as_ref()
                .expect("asym4 tier without givens sin"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            d,
            r,
        ),
        KvMode::Fwht2 => gpu.attention_flash_fwht2_batched_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht2 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht2 tier without signs2"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            d,
            r,
        ),
        KvMode::Fwht3 => gpu.attention_flash_fwht3_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht3 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht3 tier without signs2"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            8, // V_MODE_Q8 — the static slots ladder always stores V at Q8_0
            d,
            r,
        ),
        KvMode::Fwht4 => gpu.attention_flash_fwht4_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            kv.fwht_signs1.as_ref().expect("fwht4 tier without signs1"),
            kv.fwht_signs2.as_ref().expect("fwht4 tier without signs2"),
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            d,
            r,
        ),
        // bf16 keeps the SEQUENTIAL path's scalar kernel, not the windowed
        // flash tile: the tile accumulates in a different order and the
        // golden harness measured a 6.27x-tolerance divergence on one logits
        // element against the sequential reference. `attention_bf16_kv_batched_slots`
        // runs the same TU + ABI the sequential dispatch executes
        // (`AttnBf16KvBatchedMasked` — plain causal, no window arg), with the
        // descriptor tail threaded through, so slot-pool bf16 output is
        // bit-parity with the sequential engine's instead of merely close.
        // Same arg shape the q8 scalar delegate passes (physical_cap as the
        // arena's max_seq, tree verify out of scope for slots).
        KvMode::Bf16 => gpu.attention_bf16_kv_batched_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
            d,
            r,
        ),
        KvMode::F16 => gpu.attention_flash_f16_batched_masked_windowed_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            /*window=*/ 0,
            d,
            r,
        ),
        KvMode::Q8 => unreachable!("handled by the q8 delegate above"),
        // Native fp8 (gfx1201-only — Rig::build refuses the tier elsewhere):
        // the descriptor-driven fp8 flash tile is the ONLY fp8 slot reader.
        // No scalar decode or WMMA fast path exists for it — the WMMA
        // single-slot reduction is a Q8-slab pointer trick that cannot
        // express a descriptor, and the gfx1201 VerifyAttn twins take no
        // descriptor parameters — so every shape routes here (the slots
        // ladder passes window 0 for every tier).
        //
        // untested-hw: gfx1201-only path, implemented from the q8 slots
        // template; no gfx1201 host was available.
        KvMode::Fp8 => gpu.attention_flash_fp8_e4m3_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            d,
            r,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn q8_attend_slots(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k_cache: &GpuTensor,
    v_cache: &GpuTensor,
    out: &GpuTensor,
    positions: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    batch_size: usize,
    flash_partials: &GpuTensor,
    descs_dev: &GpuTensor,
    row_slot_dev: &GpuTensor,
    single_slot: Option<(u64, usize)>,
    multi_slot_tiles: Option<(&GpuTensor, &GpuTensor, &GpuTensor, usize)>,
) -> HipResult<()> {
    if let Some((k_base, slab_bytes)) = single_slot {
        if q8_flash_prefill_wmma_eligible(gpu, head_dim, batch_size) {
            let k_view = k_cache.sub_offset(k_base as usize, slab_bytes);
            let v_view = v_cache.sub_offset(k_base as usize, slab_bytes);
            return gpu.attention_q8_0_flash_prefill_wmma(
                q,
                &k_view,
                &v_view,
                out,
                positions,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx_len,
                batch_size,
            );
        }
    } else if let Some((tile_slot_dev, tile_row0_dev, tile_qbase_dev, n_tiles)) = multi_slot_tiles {
        // Caller (forward_batch_slots_with_max_layer) only populates
        // multi_slot_tiles when q8_flash_prefill_wmma_eligible already held —
        // re-check anyway so this function's own contract doesn't depend on
        // the caller getting that right, matching this crate's usual
        // defense-in-depth style (see require_q8_fullattn_layer et al.).
        if n_tiles > 0 && q8_flash_prefill_wmma_eligible(gpu, head_dim, batch_size) {
            // Views exist only so `.numel()` reports n_tiles, not the
            // persistent buffers' max_rows capacity — see this function's
            // doc comment. Byte length as reported by the view is otherwise
            // unused (only the pointer is read downstream).
            let tile_slot_view = tile_slot_dev.sub_offset(0, n_tiles);
            let tile_row0_view = tile_row0_dev.sub_offset(0, n_tiles);
            let tile_qbase_view = tile_qbase_dev.sub_offset(0, n_tiles);
            return gpu.attention_q8_0_flash_prefill_wmma_slots(
                q,
                k_cache,
                v_cache,
                out,
                positions,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx_len,
                batch_size,
                Some(descs_dev),
                Some(&tile_slot_view),
                Some(&tile_row0_view),
                Some(&tile_qbase_view),
            );
        }
    }
    // Context length above which the flash (context-split) kernel beats the
    // scalar one. Overridable via HIPFIRE_SLOTS_ATTN_CROSSOVER.
    //
    // gfx1151's 2048 is measured on this path, not inherited: at 4096-token
    // context the scalar kernel reads 17.8 MiB of KV per FA layer in 586 us
    // (~30 GB/s) because it launches only n_kv_heads x slots = 64 workgroups
    // and each scans the whole context serially. The flash kernel splits the
    // context and recovers that parallelism. Interleaved A/B, 35B-A3B, 4
    // slots, 4096-token context, 3 reps each, no overlap between arms:
    //
    //   scalar 38.02 / 38.60 / 38.18 ms per decode step  (mean 38.27)
    //   flash  35.66 / 35.94 / 36.53 ms                  (mean 36.04)
    //
    // Flash also won at 1, 2 and 3 slots (21.46->18.24, 29.19->26.85,
    // 34.49->31.25), so this is not a batching effect -- the old 8192 was
    // simply on the wrong side for this path. At ~1086 tokens the two arms
    // were within run-to-run noise, hence 2048 rather than 0.
    //
    // Left at 8192 for other non-gfx120x parts: the measurement is gfx1151's
    // and does not transfer. gfx1200/gfx1201 keep 4096, which already puts a
    // 4096-token context on the flash side.
    let crossover = gpu.slots_attn_crossover().unwrap_or({
        if gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201() {
            4096
        } else if gpu.arch_caps.is_gfx1151() {
            2048
        } else {
            8192
        }
    });
    if max_ctx_len <= crossover {
        gpu.attention_q8_0_kv_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            None,
            0,
            0,
            Some(descs_dev),
            Some(row_slot_dev),
        )
    } else {
        gpu.attention_flash_q8_0_batched_masked_slots(
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            physical_cap,
            max_ctx_len,
            batch_size,
            flash_partials,
            None,
            0,
            0,
            Some(descs_dev),
            Some(row_slot_dev),
        )
    }
}

/// Run one `FullAttention` layer across the whole step. Every step here is
/// stateless per row and runs once over all `n` rows — the KV write and
/// attend calls are each a SINGLE launch across every slot via the
/// `_slots` entry points (that's the whole point of the descriptor table),
/// and RoPE is slot-agnostic (SP2 Task 2).
#[allow(clippy::too_many_arguments)]
fn run_fullattn_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &FullAttnLayerWeights,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    addr: &LayerKvAddr,
    kv: &SlotKvTier,
    q8_wmma_arch: bool,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    // When true, `pbs.pos3` carries per-row M-RoPE phases: dispatch the
    // batched M-RoPE kernel instead of the 1D one. Text rows carry [p, p, p]
    // (bit-identical angles); only VL image/post-image rows genuinely differ.
    use_mrope: bool,
) -> HipResult<()> {
    let arch = gpu.arch.as_str();
    // Per-site plans: {wq,wk,wv} keep one fused launch when uniform, wo and
    // the FFN weights validate their own roles inside their dispatchers.
    let qkv_plan = plan_q8_fused_or_plain(
        plan_proj_group(&[&layer.wq, &layer.wk, &layer.wv], "FullAttn qkv", arch)?,
        q8_wmma_arch,
    );
    slots_check_residual_weight(&layer.wo, "FullAttn wo", arch)?;

    let dim = config.dim;

    // 1-2. rmsnorm (+FWHT-rotate for rotated containers) then the 3-way QKV
    // projection — same group-plan rules as the DeltaNet site above.
    let (need_rot, need_norm) = match qkv_plan {
        GroupPlan::Uniform(dt) => (slots_weight_rotated(dt), !slots_weight_rotated(dt)),
        GroupPlan::Mixed { needs_rot, needs_norm } => (needs_rot, needs_norm),
    };
    let qkv_anchor = if let GroupPlan::Uniform(dt) = qkv_plan {
        slots_weight_rotated(dt).then_some(&layer.wq)
    } else {
        [&layer.wq, &layer.wk, &layer.wv]
            .iter()
            .find(|w| slots_weight_rotated(w.gpu_dtype))
            .copied()
    };
    slots_norm_site(
        gpu,
        &pbs.x_batch,
        &layer.attn_norm,
        qkv_anchor,
        &pbs.x_rot_batch,
        &pbs.x_norm_batch,
        need_rot,
        need_norm,
        dim,
        config.norm_eps,
        n,
    )?;
    match qkv_plan {
        GroupPlan::Uniform(DType::Q8_0) if q8_wmma_arch => {
            run_fused_qkv_key(
                gpu,
                KernelKey::FusedQkvQ8_0,
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_norm_batch,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }
        GroupPlan::Uniform(dt) => {
            let x = if slots_weight_rotated(dt) { &pbs.x_rot_batch } else { &pbs.x_norm_batch };
            run_fused_qkv_key(
                gpu,
                fused_qkv_key_for(dt),
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                x,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }
        GroupPlan::Mixed { .. } => {
            for (w, out) in [
                (&layer.wq, &pbs.fa_q_full_batch),
                (&layer.wk, &pbs.fa_k_batch),
                (&layer.wv, &pbs.fa_v_batch),
            ] {
                let y = out.sub_offset(0, n * w.m);
                slots_plain_proj(gpu, w, &pbs.x_rot_batch, &pbs.x_norm_batch, &y, n)?;
            }
        }
    }


    // 3. Deinterleave Q + gate.
    gpu.deinterleave_f32_batched(
        &pbs.fa_q_full_batch,
        &pbs.fa_q_batch,
        &pbs.fa_gate_batch,
        config.n_heads,
        config.head_dim,
        n,
    )?;

    // 4. Per-head Q/K rmsnorm.
    gpu.rmsnorm_batched(
        &pbs.fa_q_batch,
        &layer.q_norm,
        &pbs.fa_q_batch,
        n * config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &pbs.fa_k_batch,
        &layer.k_norm,
        &pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    // 5. RoPE — slot-agnostic (SP2 Task 2): indexes by global flat row via
    // `pbs.positions`, which SlotBatch already fills per-slot-absolute and
    // global-row-indexed. No compaction in the slot path yet, so
    // pos_offset is always 0. VL steps (use_mrope) dispatch the batched
    // M-RoPE kernel over `pbs.pos3` instead — per-row (t, h, w) phases with
    // the HF THW band mapping; text rows carry [p, p, p], which is
    // bit-identical to the 1D kernel's angles.
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    if use_mrope {
        gpu.rope_mrope_halfsplit_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pbs.pos3.buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
            config.mrope_section,
        )?;
    } else {
        gpu.rope_partial_interleaved_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pbs.positions,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
        )?;
    }

    // 6-7. Batched KV write + attend — slot-aware, one launch each across
    // every slot, on the resolved KV tier and storage (`LayerKvAddr`).
    layer_kv_write_attend(gpu, config, kv, addr, pbs, s, n, physical_cap, max_ctx_len)?;

    // 8. sigmoid(gate) * attn_out, over the `n` LIVE rows only.
    //
    // `fa_attn_out_batch` / `fa_gate_batch` are sized for the largest prefill
    // batch, and `sigmoid_mul_f32` derives its element count from the tensor
    // it is handed. Passing the whole buffer therefore did the same work on a
    // 4-row decode step as on a full prefill chunk: profiled at 8388608
    // elements (2048 rows x n_heads*head_dim) and 427.6 us per call, 10 calls
    // per step -- 4.28 ms, 10.3% of a 4-slot decode step, on 2044 dead rows.
    //
    // The reference does the same thing at its own batched call sites, but
    // only ever reaches them for prefill, where n is the batch. Its decode
    // path uses the single-row `s.fa_attn_out` scratch instead. This path is
    // the one that runs a batched buffer at decode row counts.
    //
    // Rows >= n keep whatever they held rather than being overwritten with
    // sigmoid(garbage)*garbage; nothing downstream reads them (o_proj takes
    // n rows), and the golden gate holds at 0.000x.
    let gate_elems = n * config.n_heads * config.head_dim;
    gpu.sigmoid_mul_f32(
        &pbs.fa_attn_out_batch.sub_offset(0, gate_elems),
        &pbs.fa_gate_batch.sub_offset(0, gate_elems),
    )?;

    // 9. wo + residual — dispatched by wo's own dtype (rotated containers
    // rotate into fa_attn_out_rot first, HFQ*/Q8 project directly).
    slots_residual_proj(
        gpu,
        &layer.wo,
        &pbs.fa_attn_out_batch,
        &pbs.x_batch,
        &pbs.fa_attn_out_rot_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 10. FFN: rmsnorm, gate+up, silu_mul, w_down + residual — per-site.
    dense_ffn_body_slots(
        gpu,
        config,
        &layer.ffn_norm,
        &layer.w_gate,
        &layer.w_up,
        &layer.w_down,
        pbs,
        n,
        q8_wmma_arch,
    )
}

/// Run one `FullAttention` + MoE layer across the whole step. Same shape as
/// [`run_fullattn_layer_slots`] — attention (KV write + attend) is a SINGLE
/// slot-aware launch across every slot via the `_slots` entry points, on the
/// engine's resolved KV tier regardless of this layer's projection weight
/// dtype — except: (a) the QKV projection and wo admit the same per-site
/// container set as the dense body (see `plan_proj_group`), and
/// (b) the dense FFN is replaced by the reference's own
/// `prefill_moe_ffn_body_batched` (stateless per row, no slot machinery
/// needed — see `run_deltanet_moe_layer_slots`'s doc comment).
#[allow(clippy::too_many_arguments)]
fn run_fullattn_moe_layer_slots(
    gpu: &mut Gpu,
    config: &Qwen35Config,
    layer: &FullAttnMoeLayerWeights,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    addr: &LayerKvAddr,
    kv: &SlotKvTier,
    q8_wmma_arch: bool,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    weights_moe_has_mq6: bool,
    use_mrope: bool,
) -> HipResult<()> {
    let arch = gpu.arch.as_str();
    let qkv_plan = plan_q8_fused_or_plain(
        plan_proj_group(&[&layer.wq, &layer.wk, &layer.wv], "FullAttnMoE qkv", arch)?,
        q8_wmma_arch,
    );
    slots_check_residual_weight(&layer.wo, "FullAttnMoE wo", arch)?;
    require_batchable_moe_ffn(gpu, &layer.ffn)?;

    let dim = config.dim;

    // 1-2. rmsnorm(+FWHT-rotate for rotated containers) then the 3-way QKV
    // projection — same group-plan rules as the dense FullAttn body.
    let (need_rot, need_norm) = match qkv_plan {
        GroupPlan::Uniform(dt) => (slots_weight_rotated(dt), !slots_weight_rotated(dt)),
        GroupPlan::Mixed { needs_rot, needs_norm } => (needs_rot, needs_norm),
    };
    let qkv_anchor = if let GroupPlan::Uniform(dt) = qkv_plan {
        slots_weight_rotated(dt).then_some(&layer.wq)
    } else {
        [&layer.wq, &layer.wk, &layer.wv]
            .iter()
            .find(|w| slots_weight_rotated(w.gpu_dtype))
            .copied()
    };
    slots_norm_site(
        gpu,
        &pbs.x_batch,
        &layer.attn_norm,
        qkv_anchor,
        &pbs.x_rot_batch,
        &pbs.x_norm_batch,
        need_rot,
        need_norm,
        dim,
        config.norm_eps,
        n,
    )?;
    match qkv_plan {
        GroupPlan::Uniform(DType::Q8_0) if q8_wmma_arch => {
            run_fused_qkv_key(
                gpu,
                KernelKey::FusedQkvQ8_0,
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                &pbs.x_norm_batch,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }
        GroupPlan::Uniform(dt) => {
            let x = if slots_weight_rotated(dt) { &pbs.x_rot_batch } else { &pbs.x_norm_batch };
            run_fused_qkv_key(
                gpu,
                fused_qkv_key_for(dt),
                &layer.wq.buf,
                &layer.wk.buf,
                &layer.wv.buf,
                x,
                &pbs.fa_q_full_batch,
                &pbs.fa_k_batch,
                &pbs.fa_v_batch,
                layer.wq.m,
                layer.wk.m,
                layer.wv.m,
                layer.wq.k,
                n,
            )?;
        }
        GroupPlan::Mixed { .. } => {
            for (w, out) in [
                (&layer.wq, &pbs.fa_q_full_batch),
                (&layer.wk, &pbs.fa_k_batch),
                (&layer.wv, &pbs.fa_v_batch),
            ] {
                let y = out.sub_offset(0, n * w.m);
                slots_plain_proj(gpu, w, &pbs.x_rot_batch, &pbs.x_norm_batch, &y, n)?;
            }
        }
    }


    // 3. Deinterleave Q + gate.
    gpu.deinterleave_f32_batched(
        &pbs.fa_q_full_batch,
        &pbs.fa_q_batch,
        &pbs.fa_gate_batch,
        config.n_heads,
        config.head_dim,
        n,
    )?;

    // 4. Per-head Q/K rmsnorm.
    gpu.rmsnorm_batched(
        &pbs.fa_q_batch,
        &layer.q_norm,
        &pbs.fa_q_batch,
        n * config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &pbs.fa_k_batch,
        &layer.k_norm,
        &pbs.fa_k_batch,
        n * config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    // 5. RoPE — slot-agnostic (SP2 Task 2). VL steps (use_mrope) dispatch
    // the batched M-RoPE kernel over `pbs.pos3`; text rows carry [p, p, p],
    // bit-identical to the 1D kernel's angles.
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    if use_mrope {
        gpu.rope_mrope_halfsplit_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pbs.pos3.buf,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
            config.mrope_section,
        )?;
    } else {
        gpu.rope_partial_interleaved_f32_batched(
            &pbs.fa_q_batch,
            &pbs.fa_k_batch,
            &pbs.positions,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            n_rot,
            config.rope_theta,
            n,
            0,
        )?;
    }

    // 6-7. Batched KV write + attend — slot-aware, on the engine's resolved
    // KV tier and storage regardless of this layer's projection weight dtype
    // (see module doc).
    layer_kv_write_attend(gpu, config, kv, addr, pbs, s, n, physical_cap, max_ctx_len)?;

    // 8. sigmoid(gate) * attn_out, over the `n` LIVE rows only.
    //
    // `fa_attn_out_batch` / `fa_gate_batch` are sized for the largest prefill
    // batch, and `sigmoid_mul_f32` derives its element count from the tensor
    // it is handed. Passing the whole buffer therefore did the same work on a
    // 4-row decode step as on a full prefill chunk: profiled at 8388608
    // elements (2048 rows x n_heads*head_dim) and 427.6 us per call, 10 calls
    // per step -- 4.28 ms, 10.3% of a 4-slot decode step, on 2044 dead rows.
    //
    // The reference does the same thing at its own batched call sites, but
    // only ever reaches them for prefill, where n is the batch. Its decode
    // path uses the single-row `s.fa_attn_out` scratch instead. This path is
    // the one that runs a batched buffer at decode row counts.
    //
    // Rows >= n keep whatever they held rather than being overwritten with
    // sigmoid(garbage)*garbage; nothing downstream reads them (o_proj takes
    // n rows), and the golden gate holds at 0.000x.
    let gate_elems = n * config.n_heads * config.head_dim;
    gpu.sigmoid_mul_f32(
        &pbs.fa_attn_out_batch.sub_offset(0, gate_elems),
        &pbs.fa_gate_batch.sub_offset(0, gate_elems),
    )?;

    // 9. wo + residual — dispatched by wo's own dtype.
    slots_residual_proj(
        gpu,
        &layer.wo,
        &pbs.fa_attn_out_batch,
        &pbs.x_batch,
        &pbs.fa_attn_out_rot_batch,
        &pbs.x_rot_batch,
        n,
        q8_wmma_arch,
    )?;

    // 10. Batched MoE FFN — stateless per row, no slot machinery needed.
    let ctx = DispatchCtx::new(gpu);
    prefill_moe_ffn_body_batched(
        gpu,
        &layer.ffn,
        &layer.ffn_norm,
        config,
        pbs,
        n,
        &ctx,
        weights_moe_has_mq6,
        /*routed_out=*/ None,
    )
}

/// Final output norm + per-slot last-token logits. Gathers only the last
/// row of each ACTIVE slot (`m_per_slot[s] > 0`) into a compact
/// `[n_slots × dim]` block before rmsnorm + the lm_head GEMM, rather than
/// normalizing and projecting all N rows — mirrors the reference's
/// "legacy path: only last-token logits" shortcut, generalized from one
/// row to `n_slots` rows. Idle slots' rows in `logits_out` are left
/// whatever was already there; callers must not sample them (this is the
/// same contract `SlotBatch.m_per_slot` already establishes).
///
/// Per-active-slot reduction: for EVERY active slot (not just when the pool
/// has exactly one), run the reference's own single-vector legacy-path
/// kernels (`rmsnorm_f32` + `Step::Gemv`, `qwen35.rs:11984-12008`)
/// byte-for-byte, rather than a single batched `rmsnorm_batched` +
/// `GemmQ8_0BatchedChunked(n_slots)` call. Originally an `n_slots == 1`
/// special case (SP3's fix for root cause #2, confirmed by a layer-by-layer
/// bisection — see `sp3-defect-report.md`: with the FullAttention
/// kernel-selection fix in `q8_attend_slots` also in place, EVERY layer's
/// hidden state is bit-identical between this file and the reference up to
/// and including the last layer, so the only divergence was here, a
/// `rmsnorm_batched`+`GemmQ8_0BatchedChunked` (n=1) vs `rmsnorm_f32`+GEMV
/// numeric mismatch between two mathematically-equivalent but
/// differently-implemented kernel families, not a logic bug). Generalized
/// to every slot count once the WMMA flash-prefill port (root cause #1)
/// stopped masking the same mismatch at `n_slots >= 2` — see the loop body
/// below for the empirical confirmation.
fn final_logits_per_slot(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    lm_head_skip: &[bool],
) -> HipResult<()> {
    // The lm_head goes through `Step::Gemv` + `weights.output.dispatch_ref()`
    // below, which is exactly what the reference does (qwen35.rs, the
    // `weights.output` GEMV) and is dtype-generic — the dispatcher picks the
    // kernel from `gpu_dtype`. A Q8_0-only gate here was therefore
    // over-restrictive and blocked A3B, whose untied lm_head is MQ4G256, even
    // though the very next lines would have dispatched it correctly.
    //
    // Kept as an allow-list rather than removed: an unsupported dtype should
    // still fail here with a clear message naming the lm_head, not deep inside
    // the dispatcher.
    if !lm_head_slots_admissible(weights.output.gpu_dtype) {
        return Err(HipError::new(
            0,
            &format!(
                "forward_batch_slots: lm_head (weights.output) dtype {:?} is not \
                 supported by the multi-slot path (see `lm_head_slots_admissible` \
                 for the admitted tier/fixed-tier container set)",
                weights.output.gpu_dtype
            ),
        ));
    }
    let dim = config.dim;
    let n_slots = batch.m_per_slot.len();
    assert!(
        n_slots <= pbs.max_batch,
        "forward_batch_slots: n_slots ({n_slots}) exceeds pbs.max_batch ({})",
        pbs.max_batch
    );
    assert!(
        logits_out.numel() >= n_slots * config.vocab_size,
        "forward_batch_slots: logits_out has {} elements, need >= {} (n_slots * vocab_size)",
        logits_out.numel(),
        n_slots * config.vocab_size
    );

    // Per-active-slot reduction: run the reference's own single-vector
    // legacy-path kernels (`rmsnorm_f32` + `Step::Gemv`, `qwen35.rs:11984-
    // 12008`) once per active slot, byte-for-byte, rather than a single
    // batched `rmsnorm_batched`+`GemmQ8_0BatchedChunked(n_slots)` call.
    //
    // This generalizes what was originally an `n_slots == 1` special case
    // (SP3's fix for root cause #2 — see sp3-defect-report.md) to every
    // slot count. That generalization is necessary, not merely tidier: the
    // REFERENCE this file is checked against (`run_reference_for_slot` in
    // `test_forward_slots_golden.rs`) always computes each slot through an
    // independent single-sequence `forward_prefill_batch` call, which is
    // ALWAYS one row through the GEVM path — regardless of how many slots
    // this file's own SlotBatch happens to have. A batched GEMM over M =
    // n_slots rows is a mathematically-equivalent but numerically DIFFERENT
    // kernel from M independent GEVMs (different accumulation order), so it
    // was never going to match at n_slots >= 2 either, once the FullAttention
    // kernel-selection gap (root cause #1, the WMMA flash-prefill port)
    // stopped masking it. Confirmed empirically: before this change, fixing
    // only the WMMA gap left n_slots=2 failing at ~4.4x tolerance — the same
    // residual root-cause-#2 numbers SP3 first saw at n_slots=1 before this
    // exact fix.
    // ── x-batched fast path ──────────────────────────────────────────────
    // The loop below is the last per-slot-serialised op in this file:
    // everything else is one launch across all rows. Each iteration re-reads
    // the whole lm_head weight matrix (270 MiB at vocab 248320 x dim 2048,
    // MQ4G256), so a 4-slot decode step reads it four times for four
    // different activation vectors.
    //
    // `gemv_hfq4g256_xbatch` reads it once and dots it against all B vectors.
    // It is bitwise identical to running the single-vector GEMV B times --
    // gated by `test_gemv_hfq4g256_xbatch`, which checks against the actual
    // hand-tuned gfx1151 lm_head kernel this shape selects, not a generic
    // stand-in. Measured there at M=248320 K=2048:
    //
    //   B=1 1.265 -> 1.312 ms (0.96x)   B=3 3.973 -> 1.457 ms (2.73x)
    //   B=2 2.576 -> 1.325 ms (1.94x)   B=4 5.104 -> 1.868 ms (2.73x)
    //
    // B=1 is a small loss, hence the `>= 2` gate: one slot keeps the tuned
    // single-vector kernel untouched.
    //
    // Only the GEMV is batched. The rmsnorm and the FWHT rotation stay
    // exactly as the per-slot path runs them -- same `rmsnorm_f32`, same
    // `GemvFamily::rotate` with `RotateInputs::default()`, which is what
    // `Step::Gemv { input: Raw }` does internally and is what carries the
    // weight's AWQ sidecar. They are ~8 KiB of work per slot; the 270 MiB
    // weight pass is the whole cost, and reproducing their numerics exactly
    // is worth more than batching them.
    //
    // Requires every slot active so batch row `b` IS slot `b` and the kernel
    // writes straight into `logits_out` with no scatter. That holds on every
    // decode step, which is the only place this matters.
    let all_active = batch.m_per_slot.iter().all(|&m| m > 0);
    if matches!(weights.output.gpu_dtype, DType::MQ4G256)
        && all_active
        && lm_head_skip.iter().all(|&skip| !skip)
        && (2..=Gpu::HFQ4G256_XBATCH_MAX).contains(&n_slots)
        && pbs.x_rot_batch.numel() >= n_slots * dim
    {
        let dim_row_bytes = dim * 4;
        let wr = weights.output.dispatch_ref();
        let family = GemvFamily::new();
        let mut row_off = 0usize;
        for (slot, &m) in batch.m_per_slot.iter().enumerate() {
            let last_row = row_off + m - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last_row * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            let ctx = DispatchCtx::new(gpu);
            let rot = family
                .rotate(&ctx, gpu, &wr, &s.tmp, &RotateInputs::default())
                .map_err(|e| HipError::new(0, &e.to_string()))?;
            // `rotate` hands back an alias of a single shared scratch buffer,
            // so each slot's result must be copied out before the next
            // iteration overwrites it.
            gpu.memcpy_dtod_at_auto(
                &pbs.x_rot_batch.buf,
                slot * dim_row_bytes,
                &rot.buf().buf,
                0,
                dim_row_bytes,
            )?;
            row_off += m;
        }
        debug_assert_eq!(row_off, batch.total_rows());
        let x_rot_all = pbs.x_rot_batch.sub_offset(0, n_slots * dim);
        gpu.gemv_hfq4g256_xbatch(
            wr.buf,
            &x_rot_all,
            logits_out,
            config.vocab_size,
            dim,
            n_slots,
        )?;
        return Ok(());
    }

    let mut row_off = 0usize;
    for (slot, &m) in batch.m_per_slot.iter().enumerate() {
        if m > 0 {
            // MTP verify slots: the caller derives its own logits for every
            // verify row (batched trunk lm_head over all k+1 rows), so the
            // single-row GEMV here would be discarded work that still
            // re-reads the whole lm_head weight matrix.
            if lm_head_skip.get(slot).copied().unwrap_or(false) {
                row_off += m;
                continue;
            }
            let last_row = row_off + m - 1;
            let dim_row_bytes = dim * 4;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last_row * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            let ctx = DispatchCtx::new(gpu);
            let wr = weights.output.dispatch_ref();
            let logits_view = logits_out.sub_offset(slot * config.vocab_size, config.vocab_size);
            let step = Step::Gemv {
                w: &wr,
                input: GemvInput::Raw(&s.tmp),
                out: &logits_view,
            };
            execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
        }
        row_off += m;
    }
    debug_assert_eq!(row_off, batch.total_rows());
    Ok(())
}

/// Advance every active slot in `batch` by one step: embed, run every
/// layer, and write per-slot last-token logits into `logits_out`
/// (`[n_slots × vocab_size]` f32 — row `s` is valid iff
/// `batch.m_per_slot[s] > 0`; sampling it is the caller's job, e.g. via
/// `Gpu::sample_per_slot`, Task 3's harness).
///
/// `k_arenas`/`v_arenas` hold one arena tensor per `FullAttention` layer
/// (in model layer order), each sized to `pool.arena_bytes()` — the whole
/// multi-slot arena for that layer, addressed via `pool.descriptors()`'s
/// `k_base`/`v_base` byte offsets. `dn_states` holds one `DeltaNetState`
/// per slot (indexed the same as `pool`/`desc_staging`).
///
/// Dense and MoE layers alike admit uniform `Q8_0` or uniform `MQ4G256`
/// (see module doc). A non-admitted weight dtype, a
/// non-Q8 `DeltaNetState`, or a MoE FFN dtype combination
/// `moe_ffn_batched_admissible` rejects returns `Err` rather than guessing at
/// an untested path.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
) -> HipResult<()> {
    forward_batch_slots_with_max_layer(
        gpu,
        weights,
        config,
        batch,
        pool,
        dn_states,
        k_arenas,
        v_arenas,
        desc_staging,
        kv,
        pbs,
        s,
        logits_out,
        None,
    )
}

/// Debugging/bisection variant of [`forward_batch_slots`]: early-exits the
/// layer loop at `max_layer` (exclusive), mirroring the reference's own
/// `max_layer` parameter on `forward_prefill_batch_with_pbs_opts`
/// (`qwen35.rs:6554`). `pbs.x_batch[0..n*dim]` holds the post-layer hidden
/// state on return when `max_layer` is `Some` — read it directly (it is
/// `pub`) to diff against the reference's own `pbs.x_batch` after an
/// identically-bounded call. Skips the final norm/lm_head AND the per-slot
/// KV-length advance when `max_layer` is `Some`, exactly as the reference
/// skips `do_lm_head` — an early-exit call must not mutate `pool`'s
/// bookkeeping with a partial forward's positions.
///
/// Not `#[cfg(test)]`-gated: kept as a normal `pub fn` so
/// `test_forward_slots_golden` (an `examples/` binary in a different crate)
/// can call it without a feature flag plumbing exercise. `forward_batch_slots`
/// above is the real entry point every other caller keeps using unchanged.
#[allow(clippy::too_many_arguments)]
/// Context-length bucket for the decode graph cache, in tokens.
///
/// A captured graph bakes `max_ctx_len` into kernargs, but it grows by one
/// every decode step, so capturing per step would be pointless. Instead the
/// bound is rounded up to a bucket and the graph is reused until the context
/// crosses into the next one -- one capture per 256 generated tokens.
///
/// Over-scanning inside a bucket is correct, not merely tolerable:
/// `positions[]` is the authoritative causal bound everywhere downstream
/// (SP1's only Critical defect), so positions past a row's own are masked.
/// It costs at most one bucket of extra KV scan per step.
pub const DECODE_GRAPH_CTX_BUCKET: usize = 256;

/// Identity of a captured decode graph. Any change invalidates it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DecodeGraphKey {
    n_slots: usize,
    n_rows: usize,
    ctx_bucket: usize,
    max_layer: Option<usize>,
    /// FNV-1a over the per-slot row counts, the per-slot lm_head-skip mask,
    /// and the VL step flags (M-RoPE phases present / external-embedding
    /// rows present). Both flags change which kernels the captured body
    /// launches — the M-RoPE rope branch and the embed-scatter launch — so
    /// a VL decode step captures its own graph instead of replaying the
    /// text-only one. The flags only flip when requests start or finish, so
    /// re-captures stay rare.
    m_hash: u64,
}

fn decode_graph_m_hash(
    m_per_slot: &[usize],
    lm_head_skip: &[bool],
    vl_mrope: bool,
    vl_ext: bool,
) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &m in m_per_slot {
        h ^= m as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &skip in lm_head_skip {
        h ^= skip as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= vl_mrope as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= vl_ext as u64;
    h = h.wrapping_mul(0x100000001b3);
    h
}

/// Pure predicate: is this per-slot row-count pattern a capturable spec
/// verify step at the given `verify_rows` (verify rows per spec slot:
/// `mtp_k + 1` for MTP, `B` for DFlash2)? Every slot must be idle (0), a
/// regular decode row (1), or exactly one verify window (`verify_rows`).
/// Prefill chunks (any other m) keep the plain path — their shapes vary
/// step to step and would thrash the capture cache.
pub fn spec_verify_graph_shape(m_per_slot: &[usize], verify_rows: usize) -> bool {
    verify_rows > 0
        && m_per_slot
            .iter()
            .all(|&m| m == 0 || m == 1 || m == verify_rows)
}

/// A hipGraph of one pure-decode step, reused across steps.
///
/// Why: a 4-slot decode step issues ~1390 kernel launches. Profiled on
/// gfx1151 at 4096-token context, that is 4.16 ms/step of host submission
/// (2.99 us per launch) against 9.29 ms/step of measured GPU idle -- the GPU
/// spends 23% of the step waiting to be fed. Replaying one graph replaces all
/// of those submissions with a single launch.
///
/// Only pure-decode steps are captured. Prefill batches vary in shape step to
/// step, and the win is concentrated in decode anyway: prefill launches the
/// same number of kernels but each does far more work, so submission is not
/// the bottleneck there.
#[derive(Default)]
pub struct SlotDecodeGraph {
    exec: Option<hip_bridge::GraphExec>,
    graph: Option<hip_bridge::Graph>,
    key: Option<DecodeGraphKey>,
    captures: usize,
    replays: usize,
}

impl SlotDecodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many captures and replays this cache has served. A healthy decode
    /// run shows captures ~= generated_tokens / DECODE_GRAPH_CTX_BUCKET.
    pub fn stats(&self) -> (usize, usize) {
        (self.captures, self.replays)
    }

    /// Destroy captured exec then graph handles and clear the cache key.
    /// Safe to call when empty; required before the owning Gpu is torn down.
    pub fn release(&mut self, gpu: &Gpu) {
        if let Some(exec) = self.exec.take() {
            let _ = gpu.hip.graph_exec_destroy(exec);
        }
        if let Some(graph) = self.graph.take() {
            let _ = gpu.hip.graph_destroy(graph);
        }
        self.key = None;
    }
}

/// `forward_batch_slots` with hipGraph capture/replay for decode-shaped steps.
///
/// Falls back to the plain path whenever the step is not pure decode, the
/// feature is off, or capture fails -- so this is never load-bearing for
/// correctness, only for speed.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_graphed(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    cache: &mut SlotDecodeGraph,
) -> HipResult<()> {
    forward_batch_slots_graphed_opts(
        gpu,
        weights,
        config,
        batch,
        pool,
        dn_states,
        k_arenas,
        v_arenas,
        desc_staging,
        kv,
        pbs,
        s,
        logits_out,
        cache,
        /* mtp_k */ 0,
        /* lm_head_skip */ &[],
        /* spec_capture */ None,
    )
}

/// [`forward_batch_slots_graphed`] with MTP-verify awareness.
///
/// `mtp_k > 0` additionally admits steps whose per-slot row counts are all in
/// `{0, 1, mtp_k + 1}` — the vLLM-style batched-verify shape, where an MTP
/// slot contributes its `k + 1` draft rows next to regular single-row decode
/// slots. Those steps get their own graph (keyed by the `(m, skip)` pattern),
/// so an engine in steady MTP decode replays one graph launch per step
/// exactly like pure AR decode does.
///
/// `lm_head_skip[s]` suppresses the per-slot last-row lm_head GEMV inside
/// [`final_logits_per_slot`] — set for slots whose verify logits the caller
/// computes itself (MTP verify rows read the trunk lm_head over all `k + 1`
/// rows via a batched GEMM instead; the single-row GEMV would be discarded
/// work that re-reads the whole lm_head weight). The mask is part of the
/// graph key: a captured graph bakes which GEMVs exist.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_graphed_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    cache: &mut SlotDecodeGraph,
    verify_rows: usize,
    lm_head_skip: &[bool],
    mut spec_capture: Option<&mut SpecVerifyCapture>,
) -> HipResult<()> {
    let pure_decode = !batch.is_empty() && batch.m_per_slot.iter().all(|&m| m == 1);
    // An MTP verify step is graphable when every slot's row count is a decode
    // row (0/1) or exactly one verify window (mtp_k + 1). Anything else — a
    // scheduler prefill chunk mixed in — keeps the plain path: prefill shapes
    // vary step to step and would thrash the capture cache.
    let mtp_verify_shape =
        !batch.is_empty() && spec_verify_graph_shape(&batch.m_per_slot, verify_rows);
    if !gpu.slots_decode_graph() || !(pure_decode || mtp_verify_shape) || pool.is_paged() {
        return forward_batch_slots_opts(
            gpu,
            weights,
            config,
            batch,
            pool,
            dn_states,
            k_arenas,
            v_arenas,
            desc_staging,
            kv,
            pbs,
            s,
            logits_out,
            None,
            SlotStepOpts::default(),
            lm_head_skip,
            spec_capture,
        );
    }
    // The captured step includes the per-slot lm_head `Step::Gemv`, so a spilled
    // model routes it to the CPU (a host sync point) and the graph must not be
    // used — same rule as the dense AR graph in `qwen35/forward.rs`.
    if hipfire_dispatch::cpu_offload_active(config.i_gpu_start) {
        hipfire_dispatch::log_capture_disabled_once();
        return forward_batch_slots_opts(
            gpu,
            weights,
            config,
            batch,
            pool,
            dn_states,
            k_arenas,
            v_arenas,
            desc_staging,
            kv,
            pbs,
            s,
            logits_out,
            None,
            SlotStepOpts::default(),
            lm_head_skip,
            spec_capture,
        );
    }

    let physical_cap = pool.cap_tokens();
    let true_ctx = (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
        .min(physical_cap)
        .max(1);
    let ctx_bucket = true_ctx
        .div_ceil(DECODE_GRAPH_CTX_BUCKET)
        .saturating_mul(DECODE_GRAPH_CTX_BUCKET)
        .min(physical_cap)
        .max(1);
    let key = DecodeGraphKey {
        n_slots: pool.descriptors().len(),
        n_rows: batch.total_rows(),
        ctx_bucket,
        max_layer: None,
        m_hash: decode_graph_m_hash(
            &batch.m_per_slot,
            lm_head_skip,
            batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty(),
            batch.ext_emb.len() == batch.positions.len() && batch.ext_emb.iter().any(|&e| e >= 0),
        ),
    };

    // The per-step inputs always go up outside the graph: their host source
    // buffers are temporaries, and a captured memcpy node bakes its source
    // pointer. The captured kernels read the device buffers these fill, so
    // fresh contents reach a replay without re-capturing.
    upload_step_inputs(gpu, batch, pool, desc_staging, pbs)?;

    if cache.key != Some(key) {
        cache.release(gpu);
        gpu.ensure_capture_stream()?;
        // Everything must already be warm: a kernel compile inside capture is
        // exactly what ThreadLocal capture mode forbids. One uncaptured step
        // at this shape guarantees it.
        forward_batch_slots_opts(
            gpu,
            weights,
            config,
            batch,
            pool,
            dn_states,
            k_arenas,
            v_arenas,
            desc_staging,
            kv,
            pbs,
            s,
            logits_out,
            None,
            SlotStepOpts {
                skip_uploads: true,
                ctx_override: Some(ctx_bucket),
            },
            lm_head_skip,
            spec_capture.as_deref_mut(),
        )?;
        // The warm-up step above already advanced the slot lengths; re-running
        // the capture body would advance them a second time, so roll back to
        // the pre-step lengths before capturing.
        rewind_slot_seq_lens(batch, pool)?;

        gpu.begin_stream_capture()?;
        let captured = forward_batch_slots_opts(
            gpu,
            weights,
            config,
            batch,
            pool,
            dn_states,
            k_arenas,
            v_arenas,
            desc_staging,
            kv,
            pbs,
            s,
            logits_out,
            None,
            SlotStepOpts {
                skip_uploads: true,
                ctx_override: Some(ctx_bucket),
            },
            lm_head_skip,
            spec_capture,
        );
        let graph = gpu.end_stream_capture()?;
        captured?;
        let exec = gpu.hip.graph_instantiate(&graph)?;
        cache.graph = Some(graph);
        cache.exec = Some(exec);
        cache.key = Some(key);
        cache.captures += 1;
        // Capture does not execute, and the warm-up step above already did
        // this step's work, so nothing is launched here.
        return Ok(());
    }

    let exec = cache
        .exec
        .take()
        .expect("cache.key set implies an instantiated exec");
    let launched = gpu.launch_graph(&exec);
    cache.exec = Some(exec);
    launched?;
    cache.replays += 1;
    // Host-side bookkeeping the replay cannot do for itself.
    advance_slot_seq_lens(batch, pool)?;
    Ok(())
}

/// Undo `advance_slot_seq_lens` for this batch: restore each active slot to
/// the length it had before the step. Used only by the graph path, which runs
/// the same step body twice (once to warm, once to capture).
fn rewind_slot_seq_lens(batch: &SlotBatch, pool: &mut SlotPool) -> HipResult<()> {
    for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let first_row = batch
            .row_slot
            .iter()
            .position(|&s| s as usize == slot_ix)
            .expect("m_per_slot > 0 implies at least one row for this slot");
        let old_len = batch.positions[first_row] as usize;
        pool.set_seq_len(rdna_compute::slot_pool::SlotId(slot_ix), old_len)
            .map_err(|e| HipError::new(0, &format!("rewind_slot_seq_lens: {e}")))?;
    }
    Ok(())
}

/// Knobs the graph path needs and no other caller does.
#[derive(Clone, Copy, Default)]
pub struct SlotStepOpts {
    /// The per-step H2D uploads (tokens, positions, row_slot, descriptors)
    /// were already done by the caller. Set when capturing into a hipGraph:
    /// those copies read from short-lived host buffers, and a captured memcpy
    /// node bakes its source pointer -- replaying one would read freed
    /// memory. The caller does them per step, outside the graph.
    pub skip_uploads: bool,
    /// Use this context bound instead of deriving it from `batch.positions`.
    /// A captured graph bakes `max_ctx_len` into kernargs, but it grows every
    /// decode step, so the graph path rounds it up to a bucket and re-captures
    /// only when the bucket changes. Over-scanning within a bucket is safe:
    /// `positions[]` is the authoritative causal bound (SP1's Critical
    /// defect), so the extra positions are masked out.
    pub ctx_override: Option<usize>,
}

/// Upload every dirty slot's paged block table and activate its descriptor.
///
/// Shared by the plain step path and [`upload_step_inputs`] (the graph
/// path's hoisted uploads) so the two can never drift: whichever runs, a
/// dirty table is uploaded, bounded by the staging capacity, and the
/// descriptor table is left needing exactly one upload afterwards (the
/// caller uploads descs once this returns — activation mutates them).
///
/// Slots with an EMPTY table are skipped rather than activated: their
/// `seq_len` is 0, so no kernel can read their mapping, and activating
/// would point the descriptor at stale staging contents for no benefit.
pub fn upload_block_tables(
    gpu: &mut Gpu,
    pool: &mut SlotPool,
    desc_staging: &mut SlotDescStaging,
) -> HipResult<()> {
    if !pool.is_paged() || desc_staging.block_table_devs.is_empty() {
        return Ok(());
    }
    for slot_idx in 0..pool.descriptors().len() {
        if !pool.block_table_dirty(SlotId(slot_idx)) {
            continue;
        }
        let Some(bt) = pool.block_table(SlotId(slot_idx)) else {
            continue;
        };
        let indices = bt.page_indices();
        if indices.len() > desc_staging.max_pages_per_slot {
            let page_tokens = rdna_compute::page_pool::PAGE_TOKENS;
            return Err(HipError::new(
                0,
                &format!(
                    "block table for slot {slot_idx} holds {} pages but staging \
                     was sized for {} — rebuild SlotDescStaging with \
                     max_pages_per_slot >= ceil(cap_tokens / {page_tokens})",
                    indices.len(),
                    desc_staging.max_pages_per_slot
                ),
            ));
        }
        if !indices.is_empty() {
            let bt_bytes: Vec<u8> = indices.iter().flat_map(|x| x.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&desc_staging.block_table_devs[slot_idx].buf, &bt_bytes)?;
            // Activate paged mode: point the descriptor at the uploaded
            // page index array.
            let dev_addr = desc_staging.block_table_devs[slot_idx].buf.as_ptr() as u64;
            pool.activate_paged_desc(SlotId(slot_idx), dev_addr);
        }
    }
    Ok(())
}

/// The per-step H2D uploads, hoisted so the graph path can run them outside
/// the captured region. Must run before every step, captured or not.
pub fn upload_step_inputs(
    gpu: &mut Gpu,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    desc_staging: &mut SlotDescStaging,
    pbs: &PrefillBatchScratch,
) -> HipResult<()> {
    let n = batch.total_rows();
    let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
    let tokens_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;

    let positions_host: Vec<i32> = batch.positions.clone();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;

    // VL side inputs, mirroring the plain path's step-1b/2 uploads. The
    // per-row external-embedding matrix pointers are NOT uploaded here —
    // the engine refreshes those per step (it owns the matrices); the
    // captured scatter kernel reads them through `pbs.ext_emb_row_ptr`.
    if batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty() {
        let pos3_bytes: Vec<u8> = batch
            .pos3
            .iter()
            .flat_map(|t| t.iter().flat_map(|v| v.to_ne_bytes()))
            .collect();
        gpu.hip.memcpy_htod(&pbs.pos3.buf, &pos3_bytes)?;
    }
    if batch.ext_emb.len() == batch.positions.len() && batch.ext_emb.iter().any(|&e| e >= 0) {
        let idx_bytes: Vec<u8> = batch.ext_emb.iter().flat_map(|x| x.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&pbs.ext_emb_index.buf, &idx_bytes)?;
    }

    let row_slot_bytes: Vec<u8> = batch
        .row_slot
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    gpu.hip
        .memcpy_htod(&desc_staging.row_slot_dev.buf, &row_slot_bytes)?;
    // Block tables BEFORE descs: activating a paged descriptor mutates the
    // descriptor table, so it must be uploaded after this. Skipping this in
    // paged mode would let `mark_uploaded` below clear the block-table dirty
    // flags without their contents ever reaching the device — stale page
    // mappings, i.e. silent cross-session corruption.
    upload_block_tables(gpu, pool, desc_staging)?;
    if pool.descriptors_dirty() {
        let desc_bytes = pack_descs(pool.descriptors());
        gpu.hip
            .memcpy_htod(&desc_staging.descs_dev.buf, &desc_bytes)?;
        pool.mark_uploaded();
    }
    Ok(())
}

/// Post-step host bookkeeping: advance each active slot's logical KV length.
/// Hoisted for the same reason as the uploads -- it is host code, so a graph
/// replay does not run it, and the caller must.
pub fn advance_slot_seq_lens(batch: &SlotBatch, pool: &mut SlotPool) -> HipResult<()> {
    for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let last_row = batch
            .row_slot
            .iter()
            .rposition(|&s| s as usize == slot_ix)
            .expect("m_per_slot > 0 implies at least one row for this slot");
        let new_len = (batch.positions[last_row] + 1) as usize;
        pool.set_seq_len(rdna_compute::slot_pool::SlotId(slot_ix), new_len)
            .map_err(|e| HipError::new(0, &format!("forward_batch_slots: {e}")))?;
    }
    Ok(())
}

pub fn forward_batch_slots_with_max_layer(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    max_layer: Option<usize>,
) -> HipResult<()> {
    forward_batch_slots_opts(
        gpu,
        weights,
        config,
        batch,
        pool,
        dn_states,
        k_arenas,
        v_arenas,
        desc_staging,
        kv,
        pbs,
        s,
        logits_out,
        max_layer,
        SlotStepOpts::default(),
        &[],
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_batch_slots_opts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &mut SlotPool,
    dn_states: &mut [DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &mut SlotDescStaging,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    logits_out: &GpuTensor,
    max_layer: Option<usize>,
    opts: SlotStepOpts,
    lm_head_skip: &[bool],
    mut spec_capture: Option<&mut SpecVerifyCapture>,
) -> HipResult<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let n = batch.total_rows();
    let n_slots = pool.descriptors().len();
    assert_eq!(
        batch.m_per_slot.len(),
        n_slots,
        "forward_batch_slots: batch has {} slots, pool has {}",
        batch.m_per_slot.len(),
        n_slots
    );
    assert_eq!(
        dn_states.len(),
        n_slots,
        "forward_batch_slots: dn_states.len() ({}) must equal n_slots ({n_slots})",
        dn_states.len()
    );
    assert_eq!(
        desc_staging.n_slots, n_slots,
        "forward_batch_slots: desc_staging was built for a different n_slots"
    );
    assert!(
        n <= pbs.max_batch,
        "forward_batch_slots: batch.total_rows() ({n}) exceeds pbs.max_batch ({})",
        pbs.max_batch
    );
    assert!(
        n <= desc_staging.max_rows,
        "forward_batch_slots: batch.total_rows() ({n}) exceeds desc_staging.max_rows ({})",
        desc_staging.max_rows
    );
    let n_fa_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    assert_eq!(
        k_arenas.len(),
        n_fa_layers,
        "forward_batch_slots: k_arenas.len() must equal the model's FullAttention layer count"
    );
    assert_eq!(v_arenas.len(), k_arenas.len());
    // A paged pool has nowhere to put KV without per-slot block-table staging.
    // Silent degradation here would read legacy_k_base = 0 in every kernel —
    // fail loudly instead.
    assert!(
        !pool.is_paged() || !desc_staging.block_table_devs.is_empty(),
        "forward_batch_slots: paged pool requires SlotDescStaging::new with \
         max_pages_per_slot > 0"
    );

    let dim = config.dim;

    // ── 1. Embed tokens (Q8_0 only) ──────────────────────────────────────
    if !matches!(weights.embd_format, EmbeddingFormat::Q8_0) {
        return Err(HipError::new(
            0,
            "forward_batch_slots: embedding table must be Q8_0 (the multi-slot \
             batched path is Q8_0-only, see sp3-task-2-report.md)",
        ));
    }
    if !opts.skip_uploads {
        let tokens_host: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
        let tokens_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
    }
    gpu.embedding_lookup_q8_batched(&weights.token_embd, &pbs.x_batch, &pbs.tokens, n, dim)?;

    // ── 1b. VL rows: overwrite image-pad rows with vision embeddings ─────
    // `batch.ext_emb` (per-row matrix index, -1 = token table) and the
    // per-row matrix base pointers are device-side inputs: the caller
    // uploads the pointers per step (engine-side, before this call) so a
    // captured graph replays with fresh values, and the index array is
    // uploaded here from the batch. The kernel itself is a no-op on rows
    // with a negative index, and skips null pointers defensively.
    let use_ext =
        batch.ext_emb.len() == batch.positions.len() && batch.ext_emb.iter().any(|&e| e >= 0);
    if use_ext {
        if !opts.skip_uploads {
            let idx_bytes: Vec<u8> = batch.ext_emb.iter().flat_map(|x| x.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&pbs.ext_emb_index.buf, &idx_bytes)?;
        }
        gpu.embedding_scatter_ext_batched(
            &pbs.x_batch,
            &pbs.ext_emb_index,
            &pbs.ext_emb_row_ptr,
            n,
            dim,
        )?;
    }

    // ── 2. Upload positions ──────────────────────────────────────────────
    // Per-row ABSOLUTE position within that row's own slot — authoritative
    // for the causal bound everywhere downstream (RoPE angle, KV write
    // slot-relative index, and the attend kernels' per-row seq_len). Never
    // `desc.seq_len` — see SP1's only Critical defect.
    if !opts.skip_uploads {
        let positions_host: Vec<i32> = batch.positions.clone();
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }
    // VL M-RoPE phases ([t, h, w] per row, absolute). Presence of a complete
    // pos3 array flips every FA layer's RoPE to the batched M-RoPE kernel —
    // text rows carry [p, p, p], which that kernel reduces to the same angles
    // as the 1D path, so mixed text+VL steps stay byte-identical on text.
    // KV addressing and causal bounds keep reading `positions` regardless.
    let use_mrope = batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty();
    if use_mrope && !opts.skip_uploads {
        let pos3_bytes: Vec<u8> = batch
            .pos3
            .iter()
            .flat_map(|t| t.iter().flat_map(|v| v.to_ne_bytes()))
            .collect();
        gpu.hip.memcpy_htod(&pbs.pos3.buf, &pos3_bytes)?;
    }

    // ── 2b. Provision pages for the write frontier (paged mode only) ────
    // The KV-write kernel resolves `block_table[pos / PAGE_TOKENS]` for
    // pos = positions[row], so every page this step will write through must
    // exist BEFORE the kernel runs. `advance_slot_seq_lens` only grows the
    // slot's length after the step; without provisioning here, a step that
    // crosses into a fresh page reads one entry past the uploaded table —
    // stale staging memory, i.e. silent cross-session corruption. Setting
    // seq_len to the step's last position+1 is exactly what
    // `advance_slot_seq_lens` will set it to, so the post-step call is
    // capacity-idempotent. OOM fails closed here, before any GPU write.
    if pool.is_paged() {
        for (slot_ix, &m) in batch.m_per_slot.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let max_pos = batch
                .positions
                .iter()
                .zip(batch.row_slot.iter())
                .filter(|&(_, &s)| s as usize == slot_ix)
                .map(|(&p, _)| p as usize)
                .max()
                .expect("m_per_slot > 0 implies at least one row for this slot");
            pool.set_seq_len(rdna_compute::slot_pool::SlotId(slot_ix), max_pos + 1)
                .map_err(|e| HipError::new(0, &format!("forward_batch_slots: {e}")))?;
        }
    }

    // ── 3. Upload row_slot (every step); block tables (paged, before descs
    // — activation mutates them); then the descriptor table (only when
    // dirty) — once per step, not once per layer. ────────────────────────
    if !opts.skip_uploads {
        let row_slot_bytes: Vec<u8> = batch
            .row_slot
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();
        gpu.hip
            .memcpy_htod(&desc_staging.row_slot_dev.buf, &row_slot_bytes)?;
        upload_block_tables(gpu, pool, desc_staging)?;
        if pool.descriptors_dirty() {
            let desc_bytes = pack_descs(pool.descriptors());
            gpu.hip
                .memcpy_htod(&desc_staging.descs_dev.buf, &desc_bytes)?;
            pool.mark_uploaded();
        }
    }

    let physical_cap = pool.cap_tokens();
    let max_ctx_len = opts.ctx_override.unwrap_or(
        (batch.positions.iter().copied().max().unwrap_or(0) as usize + 1)
            .min(physical_cap)
            .max(1),
    );
    let q8_wmma_arch = q8_prefill_wmma_enabled(gpu);

    // Exactly one active slot this step? (Always true for n_slots == 1;
    // also true for a larger pool when every row this step belongs to the
    // same slot.) When so, `q8_attend_slots` can safely reduce to the
    // reference's own WMMA flash-prefill kernel (see its doc comment) —
    // that kernel has no slot-descriptor concept, so it is unsafe to use
    // whenever more than one slot has live rows in the same call.
    let active_slots = batch.m_per_slot.iter().filter(|&&m| m > 0).count();
    // The single-slot reduction shifts raw pointers by the slot's legacy
    // slab base and hands them to a kernel with NO descriptor concept. A
    // paged slot's KV does not live at a contiguous slab at all, so this
    // reduction must never fire in paged mode — the descriptor-driven
    // kernels below are the only correct attend path there.
    let single_slot = if kv.is_q8() && active_slots == 1 && !pool.is_paged() {
        let slot_idx = batch
            .m_per_slot
            .iter()
            .position(|&m| m > 0)
            .expect("active_slots == 1 implies exactly one m_per_slot entry > 0");
        let k_base = pool.descriptors()[slot_idx].legacy_k_base;
        let slab_bytes = pool.arena_bytes() / n_slots;
        Some((k_base, slab_bytes))
    } else {
        None
    };

    // `n > active_slots` means at least one slot contributes more than one row
    // this step -- i.e. there is prompt left to prefill. A step where every
    // active slot contributes exactly one row is pure decode, and there the
    // WMMA *prefill* kernel is both wrong to use and slow:
    //
    //   - Wrong: the reference decodes each sequence with batch_size == 1, so
    //     its own `batch_size <= 1` gate puts it on the scalar path. Batching
    //     N sequences into N rows and then firing the WMMA kernel because
    //     N > 1 diverges from what the reference did for those same tokens.
    //   - Slow: it tiles query rows in WMMA_M_TILE (16) blocks, so 2-4 decode
    //     rows launch a full, mostly-padded tile. Measured on gfx1151,
    //     35B-A3B, 4096-token context, per decode step:
    //
    //       slots | firing on decode | pure-decode scalar | aggregate decode
    //         2   |     57.23 ms     |      33.64 ms      | 34.95 -> 59.45 tok/s
    //         3   |     61.77 ms     |      40.41 ms      | 48.57 -> 74.23 tok/s
    //         4   |     68.60 ms     |      48.47 ms      | 58.31 -> 82.52 tok/s
    //
    //     A flat ~21 ms penalty that made two concurrent users slower *in
    //     aggregate* than one. Prefill still takes the kernel and still wants
    //     it: the 4-slot run prefills in 14.7 s either way, against 19.2 s
    //     with the path disabled outright.
    //
    // Gating on row count instead (n >= 16) was tried and is wrong: it also
    // pushes small *prefill* batches (the golden gate's 14-row, 2-slot case)
    // onto the scalar kernel, which is not bit-identical to the reference's
    // WMMA kernel, and the golden gate fails at 225x tolerance.
    //
    // Genuinely multi-slot AND the reference's own gfx11 WMMA-flash-prefill
    // gate (q8_flash_prefill_wmma_eligible) would fire for this shape: the
    // exact condition under which the reference's flat, slot-unaware
    // dispatch would have run attention_q8_0_flash_prefill_wmma. Built once
    // per step here (not once per FullAttention layer — head_dim/batch_size
    // are layer-invariant config, and re-uploading per layer would repeat
    // identical work), mirroring row_slot_dev's "every step, not every
    // layer" upload policy.
    let n_tiles: Option<usize> = if kv.is_q8()
        && single_slot.is_none()
        && active_slots > 1
        && n > active_slots
        && q8_flash_prefill_wmma_eligible(gpu, config.head_dim, n)
    {
        let (tile_slot, tile_row0, tile_qbase) = build_tiles(&batch.m_per_slot, WMMA_M_TILE);
        let nt = tile_slot.len();
        assert!(
            nt <= desc_staging.max_rows,
            "forward_batch_slots: n_tiles ({nt}) exceeds desc_staging.max_rows \
             ({}) capacity — a tile always owns >= 1 row so this should be \
             unreachable unless max_rows was undersized",
            desc_staging.max_rows
        );
        if nt > 0 {
            let to_bytes =
                |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
            gpu.hip
                .memcpy_htod(&desc_staging.tile_slot_dev.buf, &to_bytes(&tile_slot))?;
            gpu.hip
                .memcpy_htod(&desc_staging.tile_row0_dev.buf, &to_bytes(&tile_row0))?;
            gpu.hip
                .memcpy_htod(&desc_staging.tile_qbase_dev.buf, &to_bytes(&tile_qbase))?;
        }
        Some(nt)
    } else {
        None
    };

    // ── 4. Per-layer loop ─────────────────────────────────────────────────
    let layer_end = config.n_layers.min(max_layer.unwrap_or(usize::MAX));
    run_layers_slots(
        gpu,
        weights,
        config,
        batch,
        &*dn_states,
        |kv_layer_idx| LayerKvAddr::Arena {
            k_cache: &k_arenas[kv_layer_idx],
            v_cache: &v_arenas[kv_layer_idx],
            desc_staging,
            single_slot,
            n_tiles,
        },
        kv,
        pbs,
        s,
        q8_wmma_arch,
        n,
        physical_cap,
        max_ctx_len,
        use_mrope,
        layer_end,
        spec_capture.as_deref_mut(),
    )?;

    if max_layer.is_some() {
        // Early-exit for bisection: mirror the reference's `do_lm_head =
        // ... && max_layer.is_none()` — skip the final norm/lm_head AND the
        // KV-length advance below (a partial forward must not tell `pool`
        // this step's positions were fully written).
        return Ok(());
    }

    // ── 5. Final norm + per-slot last-token logits ──────────────────────
    final_logits_per_slot(
        gpu,
        weights,
        config,
        batch,
        pbs,
        s,
        logits_out,
        lm_head_skip,
    )?;

    // ── 6. Advance each slot's logical KV length ────────────────────────
    //
    // This step exists because SP1 removed the device-side
    // `positions[row] + 1 <= desc.seq_len` guard: it shipped in release
    // (compiler.rs never passes -DNDEBUG) and cost 64 bytes/lane of scratch on
    // four kernels. Removing it was right for occupancy, but it left the
    // invariant unenforced — and SP3 Task 3's review found that nothing was
    // keeping `seq_len` in sync with a slot's real history either. A caller who
    // forgot to update it got no error, just quietly-wrong metadata that the
    // next step would read as the slot's length.
    //
    // Maintaining it HERE rather than asking callers to remember is
    // correct-by-construction: this function is the only thing that advances a
    // slot's KV, so it is the only thing that can get the length right.
    advance_slot_seq_lens(batch, pool)?;
    Ok(())
}

/// The per-layer body shared by the arena and VMM slots forwards. `kv_addr`
/// yields the KV storage for FullAttention layer `kv_layer_idx`.
#[allow(clippy::too_many_arguments)]
fn run_layers_slots<'a, D, F>(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    dn_states: &D,
    kv_addr: F,
    kv: &SlotKvTier,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    q8_wmma_arch: bool,
    n: usize,
    physical_cap: usize,
    max_ctx_len: usize,
    use_mrope: bool,
    layer_end: usize,
    mut spec_capture: Option<&mut SpecVerifyCapture>,
) -> HipResult<()>
where
    D: std::ops::Index<usize, Output = DeltaNetState> + ?Sized,
    F: Fn(usize) -> LayerKvAddr<'a>,
{
    let mut delta_layer_idx = 0usize;
    let mut kv_layer_idx = 0usize;
    for layer_idx in 0..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                run_deltanet_layer_slots(
                    gpu,
                    config,
                    layer,
                    batch,
                    dn_states,
                    pbs,
                    q8_wmma_arch,
                    n,
                    delta_layer_idx,
                    spec_capture.as_deref_mut(),
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                run_fullattn_layer_slots(
                    gpu,
                    config,
                    layer,
                    pbs,
                    s,
                    &kv_addr(kv_layer_idx),
                    kv,
                    q8_wmma_arch,
                    n,
                    physical_cap,
                    max_ctx_len,
                    use_mrope,
                )?;
                kv_layer_idx += 1;
            }
            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                run_deltanet_moe_layer_slots(
                    gpu,
                    config,
                    layer,
                    batch,
                    dn_states,
                    pbs,
                    q8_wmma_arch,
                    n,
                    delta_layer_idx,
                    spec_capture.as_deref_mut(),
                    weights.moe_has_mq6,
                )?;
                delta_layer_idx += 1;
            }
            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                run_fullattn_moe_layer_slots(
                    gpu,
                    config,
                    layer,
                    pbs,
                    s,
                    &kv_addr(kv_layer_idx),
                    kv,
                    q8_wmma_arch,
                    n,
                    physical_cap,
                    max_ctx_len,
                    weights.moe_has_mq6,
                    use_mrope,
                )?;
                kv_layer_idx += 1;
            }
            (_, lt) => {
                return Err(HipError::new(
                    0,
                    &format!(
                        "forward_batch_slots: layer {layer_idx} weight/type mismatch \
                         (layer_type={lt:?})"
                    ),
                ));
            }
        }
        // DFlash2 extract-layer hidden capture: copy this step's
        // post-layer hidden rows into the shared staging buffer. Runs
        // for every layer kind (extract layers can be DeltaNet or
        // FullAttention); `capture_hidden_rows` is a no-op when the
        // capture is unset or this layer is not an extract layer.
        if let Some(cap) = spec_capture.as_deref() {
            cap.capture_hidden_rows(gpu, pbs, layer_idx, n)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_dispatch::types::KernelKey;
    use rdna_compute::DType;

    #[test]
    fn v2_one_to_one_keys_no_hfq4_fallback() {
        // Contract: each admitted V2 dtype maps to its exact V2 key, never HFQ4/default.
        // Plain GEMV (via helper fallback) is covered elsewhere; here we check
        // the four dense helpers plus residual.
        let cases: &[(DType, KernelKey, KernelKey, KernelKey, KernelKey)] = &[
            (
                DType::MQ6G256V2,
                KernelKey::GemmMq6G256V2Residual,
                KernelKey::FusedQkvzaMq6G256V2,
                KernelKey::FusedQkvMq6G256V2,
                KernelKey::FusedGateUpMq6G256V2,
            ),
            (
                DType::MQ5G256V2,
                KernelKey::GemmMq5G256V2Residual,
                KernelKey::FusedQkvzaMq5G256V2,
                KernelKey::FusedQkvMq5G256V2,
                KernelKey::FusedGateUpMq5G256V2,
            ),
            (
                DType::MQ3G256V2,
                KernelKey::GemmMq3G256V2Residual,
                KernelKey::FusedQkvzaMq3G256V2,
                KernelKey::FusedQkvMq3G256V2,
                KernelKey::FusedGateUpMq3G256V2,
            ),
            (
                DType::MQ2G256V2,
                KernelKey::GemmMq2G256V2Residual,
                KernelKey::FusedQkvzaMq2G256V2,
                KernelKey::FusedQkvMq2G256V2,
                KernelKey::FusedGateUpMq2G256V2,
            ),
        ];
        for (dt, exp_resid, exp_qkvza, exp_qkv, exp_gate) in cases {
            assert_eq!(
                residual_gemm_key_for(*dt),
                *exp_resid,
                "residual mismatch for {:?}",
                dt
            );
            assert_eq!(
                fused_qkvza_key_for(*dt),
                *exp_qkvza,
                "qkvza mismatch for {:?}",
                dt
            );
            assert_eq!(
                fused_qkv_key_for(*dt),
                *exp_qkv,
                "qkv mismatch for {:?}",
                dt
            );
            assert_eq!(
                fused_gate_up_key_for(*dt),
                *exp_gate,
                "gate_up mismatch for {:?}",
                dt
            );
            // No HFQ4 leakage
            assert_ne!(residual_gemm_key_for(*dt), KernelKey::GemmHfq4G256Residual);
            assert_ne!(fused_qkvza_key_for(*dt), KernelKey::FusedQkvzaHfq4G256);
            assert_ne!(fused_qkv_key_for(*dt), KernelKey::FusedQkvHfq4G256);
            assert_ne!(fused_gate_up_key_for(*dt), KernelKey::FusedGateUpHfq4G256);
        }
        // Negative: v1 and MQ4C must NOT map to new V2 keys
        assert_eq!(
            residual_gemm_key_for(DType::MQ4G256),
            KernelKey::GemmHfq4G256Residual
        );
        assert_eq!(
            fused_qkv_key_for(DType::HFQ4G256),
            KernelKey::FusedQkvHfq4G256
        );
    }

    #[test]
    fn six_bit_g256_container_selects_hfq6_keys() {
        // MQ6G256 MoE attention and shared-expert gate/up (AWQ A3B layers
        // 0/1/38/39) route through these selectors; an HFQ4 key here reads the
        // 200 B/group container at the 136 B stride and decodes noise.
        for dt in [DType::MQ6G256, DType::HFQ6G256] {
            assert_eq!(fused_qkvza_key_for(dt), KernelKey::FusedQkvzaHfq6G256, "{dt:?}");
            assert_eq!(fused_qkv_key_for(dt), KernelKey::FusedQkvHfq6G256, "{dt:?}");
            assert_eq!(fused_gate_up_key_for(dt), KernelKey::FusedGateUpHfq6G256, "{dt:?}");
            assert_eq!(residual_gemm_key_for(dt), KernelKey::GemmHfq6G256Residual, "{dt:?}");
        }
    }

    #[test]
    fn v2_plain_gemm_resolve_no_wildcard() {
        // Plain GEMM auto-selection (GemmFamily::resolve) — not this file's logic
        // but we assert the key helpers correctly identify V2 vs wildcard.
        // The actual resolve is tested in dispatch; here we ensure dtype identity.
        assert_ne!(DType::MQ6G256V2, DType::HFQ4G256);
        assert_ne!(DType::MQ5G256V2, DType::HFQ4G256);
        assert_ne!(DType::MQ3G256V2, DType::HFQ4G256);
        assert_ne!(DType::MQ2G256V2, DType::HFQ4G256);
    }

    #[test]
    fn lm_head_slots_admissible_covers_tier_dtypes() {
        // Everything `--tier`/`--fixed-tier` can lift the head to, plus the
        // unrotated siblings the generic GEMV handles.
        for dt in [
            DType::Q8_0,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ4G256V2,
            DType::MQ4CG256,
            DType::MQ6G256,
            DType::HFQ6G256,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
            DType::HFQ3G256,
            DType::F16,
        ] {
            assert!(lm_head_slots_admissible(dt), "lm_head refuses {dt:?}");
        }
        // Do not silently widen to codebooks without slot GEMV coverage.
        for dt in [DType::MQ4G256V2Lloyd, DType::MQ3G256Lloyd] {
            assert!(!lm_head_slots_admissible(dt), "lm_head admits {dt:?}");
        }
    }

    #[test]
    fn proj_group_plans_uniform_and_mixed() {
        let arch = "gfx1101";
        let m = |dt: DType| (dt, false);
        // Uniform: the whole-V2 family plus Q8 stay one fused launch.
        for dt in [
            DType::Q8_0,
            DType::MQ4G256,
            DType::MQ4G256V2,
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
        ] {
            assert_eq!(
                plan_proj_group_dtypes(&[m(dt), m(dt)], arch),
                Ok(GroupPlan::Uniform(dt)),
                "uniform {dt:?} must plan Uniform"
            );
        }
        // MQ4CG256 is a gfx12-only container (is_batchable_la WMMA gate):
        // refused on gfx11, admitted uniform on gfx12.
        assert!(plan_proj_group_dtypes(&[m(DType::MQ4CG256), m(DType::MQ4CG256)], arch).is_err());
        assert_eq!(
            plan_proj_group_dtypes(&[m(DType::MQ4CG256), m(DType::MQ4CG256)], "gfx1201"),
            Ok(GroupPlan::Uniform(DType::MQ4CG256))
        );
        // qt44 + Q8 mixed group (tier-pro shape): both plain-capable, both
        // variants needed.
        assert_eq!(
            plan_proj_group_dtypes(&[m(DType::MQ4G256V2), m(DType::Q8_0)], arch),
            Ok(GroupPlan::Mixed {
                needs_rot: true,
                needs_norm: true
            })
        );
        // Two rotated V2 dtypes: one rot variant serves both.
        assert_eq!(
            plan_proj_group_dtypes(&[m(DType::MQ4G256V2), m(DType::MQ6G256V2)], arch),
            Ok(GroupPlan::Mixed {
                needs_rot: true,
                needs_norm: false
            })
        );
        // HFQ4 (unrotated) + MQ4 (rotated): both variants.
        assert_eq!(
            plan_proj_group_dtypes(&[m(DType::MQ4G256), m(DType::HFQ4G256)], arch),
            Ok(GroupPlan::Mixed {
                needs_rot: true,
                needs_norm: true
            })
        );
        // qt15 Promote6 member in a MIXED group: no GemmHfq6G256 plain key →
        // refuse with a named reason (uniform qt15 stays fused-legal).
        assert!(plan_proj_group_dtypes(&[m(DType::MQ6G256), m(DType::MQ4G256V2)], arch)
            .unwrap_err()
            .contains("no per-projection GEMM key"));
        // AWQ + mixed dtypes refuse; AWQ + uniform is fine.
        assert!(plan_proj_group_dtypes(
            &[(DType::MQ4G256V2, true), m(DType::Q8_0)],
            arch
        )
        .unwrap_err()
        .contains("AWQ"));
        assert_eq!(
            plan_proj_group_dtypes(&[(DType::MQ4G256V2, true), (DType::MQ4G256V2, true)], arch),
            Ok(GroupPlan::Uniform(DType::MQ4G256V2))
        );
        // Out-of-scope containers refuse at admission.
        for dt in [
            DType::MQ4G256V2Lloyd,
            DType::TQ2G128,
            DType::HFQ4G128,
            DType::F16,
        ] {
            assert!(
                plan_proj_group_dtypes(&[m(dt), m(DType::MQ4G256V2)], arch).is_err(),
                "{dt:?} must not be admitted"
            );
        }
    }

    #[test]
    fn residual_and_plain_key_tables_cover_admitted_set() {
        // Every admitted rotated container must have BOTH a plain path and
        // a residual path reachable — the tables are the contract
        // `slots_residual_proj`/`slots_plain_proj` dispatch on.
        for dt in [
            DType::MQ4G256,
            DType::MQ4G256V2,
            DType::MQ4CG256,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
        ] {
            assert!(slots_plain_gemm_key(dt).is_some(), "{dt:?} needs a plain key");
            assert!(
                slots_residual_gemm_key(dt).is_some(),
                "{dt:?} needs a residual key"
            );
            assert!(slots_weight_rotated(dt), "{dt:?} must be rotated");
        }
        // Unrotated containers: residual only (plain key is only needed for
        // mixed groups — HFQ4 has one, HFQ6/HFQ3 are uniform-only).
        assert!(slots_plain_gemm_key(DType::HFQ4G256).is_some());
        for dt in [DType::HFQ4G256, DType::HFQ6G256, DType::MQ6G256] {
            assert!(
                slots_residual_gemm_key(dt).is_some(),
                "{dt:?} needs a residual key"
            );
        }
        assert!(slots_plain_gemm_key(DType::Q8_0).is_some());
        // F16/Lloyd/etc. stay refused at admission regardless of key tables.
        assert!(!slots_proj_admissible(DType::F16, "gfx1101"));
        assert!(!slots_proj_admissible(DType::MQ4G256V2Lloyd, "gfx1101"));
    }

    #[test]
    fn rotated_residual_keys_never_fall_back_to_hfq4() {
        // The wo/w_down rotated path (`mq4_residual_proj`) requires every
        // rotated container to resolve through `slots_residual_gemm_key`.
        // The dispatch-level `residual_gemm_key_for` wildcard would silently
        // send the 200 B/group MQ6G256 and 104/112 B/group MQ3 headers
        // through the 136 B/group HFQ4 v1 kernel — full-speed noise (see the
        // container-selection note above the fused key helpers). MQ4G256 is
        // the one legitimate borrower (it IS the HFQ4 container plus an
        // offline FWHT).
        for dt in [
            DType::MQ4G256,
            DType::MQ4G256V2,
            DType::MQ4CG256,
            DType::MQ6G256,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256,
            DType::MQ3G256V2,
            DType::MQ3G256Lloyd,
            DType::MQ2G256V2,
        ] {
            assert!(slots_weight_rotated(dt), "{dt:?} is a rotated member");
            let key = slots_residual_gemm_key(dt)
                .unwrap_or_else(|| panic!("rotated container {dt:?} lost its residual key"));
            if dt != DType::MQ4G256 {
                assert_ne!(
                    key, KernelKey::GemmHfq4G256Residual,
                    "{dt:?} must not share the HFQ4 v1 residual kernel"
                );
            }
        }
        // ...and the non-136B containers get their own families explicitly.
        assert_eq!(
            slots_residual_gemm_key(DType::MQ6G256),
            Some(KernelKey::GemmHfq6G256Residual)
        );
        assert_eq!(
            slots_residual_gemm_key(DType::MQ3G256),
            Some(KernelKey::GemmHfq3G256Residual)
        );
        assert_eq!(
            slots_residual_gemm_key(DType::MQ3G256Lloyd),
            Some(KernelKey::GemmMq3G256LloydResidual)
        );
    }

    #[test]
    fn q8_uniform_degrades_to_plain_on_non_wmma() {
        // Uniform Q8_0 must degrade to the Mixed plain fork when the WMMA
        // fused kernels don't apply (non-WMMA arch or HIPFIRE_Q8_PREFILL_
        // WMMA=0 folded into q8_wmma_arch); every other plan passes through.
        assert_eq!(
            plan_q8_fused_or_plain(GroupPlan::Uniform(DType::Q8_0), false),
            GroupPlan::Mixed {
                needs_rot: false,
                needs_norm: true
            }
        );
        assert_eq!(
            plan_q8_fused_or_plain(GroupPlan::Uniform(DType::Q8_0), true),
            GroupPlan::Uniform(DType::Q8_0)
        );
        assert_eq!(
            plan_q8_fused_or_plain(GroupPlan::Uniform(DType::MQ4G256V2), false),
            GroupPlan::Uniform(DType::MQ4G256V2)
        );
        assert_eq!(
            plan_q8_fused_or_plain(
                GroupPlan::Mixed {
                    needs_rot: true,
                    needs_norm: true
                },
                false
            ),
            GroupPlan::Mixed {
                needs_rot: true,
                needs_norm: true
            }
        );
    }

    #[test]
    fn spec_verify_graph_shape_admits_only_verify_patterns() {
        // Pure decode / mixed verify+decode+idle: graphable at verify_rows=4
        // (MTP k=3 → k+1 rows; DFlash2 B=16 → 16 rows).
        assert!(spec_verify_graph_shape(&[1, 1, 1, 1], 4));
        assert!(spec_verify_graph_shape(&[4, 1, 0, 4], 4));
        assert!(spec_verify_graph_shape(&[4, 0], 4));
        // A prefill chunk (any m not in {0,1,verify_rows}) must keep the plain path.
        assert!(!spec_verify_graph_shape(&[4, 1, 7, 4], 4));
        assert!(!spec_verify_graph_shape(&[256], 4));
        // Verify rows that don't match the configured width (drift / partial window).
        assert!(!spec_verify_graph_shape(&[4, 1], 3));
        // Spec off: nothing but pure decode (handled separately) graph captures.
        assert!(!spec_verify_graph_shape(&[4, 1], 0));
    }

    #[test]
    fn decode_graph_m_hash_separates_patterns() {
        // Same total rows, different patterns → different graphs.
        assert_ne!(
            decode_graph_m_hash(&[4, 1], &[true, false], false, false),
            decode_graph_m_hash(&[1, 4], &[false, true], false, false)
        );
        // The lm_head-skip mask participates: same m-pattern with different
        // skips must not share a captured graph (the recorded GEMV set
        // differs).
        assert_ne!(
            decode_graph_m_hash(&[4, 1], &[true, false], false, false),
            decode_graph_m_hash(&[4, 1], &[false, false], false, false)
        );
        // Stable across calls.
        assert_eq!(
            decode_graph_m_hash(&[4, 1, 0], &[true, false, false], false, false),
            decode_graph_m_hash(&[4, 1, 0], &[true, false, false], false, false)
        );
        // An absent mask and an all-false mask hash differently. That is
        // acceptable, not a defect: at worst it costs one extra capture when
        // the engine toggles between MTP-on and MTP-off batches — a key
        // mismatch can never alias a wrong graph.
        assert_ne!(
            decode_graph_m_hash(&[1, 1], &[], false, false),
            decode_graph_m_hash(&[1, 1], &[false, false], false, false)
        );
        // The VL step flags participate: an M-RoPE or external-embedding
        // step captures its own graph (different rope/scatter kernel set).
        assert_ne!(
            decode_graph_m_hash(&[1, 1], &[false, false], true, false),
            decode_graph_m_hash(&[1, 1], &[false, false], false, false)
        );
        assert_ne!(
            decode_graph_m_hash(&[1, 1], &[false, false], false, true),
            decode_graph_m_hash(&[1, 1], &[false, false], false, false)
        );
    }
}
