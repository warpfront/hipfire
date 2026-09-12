// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-resident FLUX.1 MMDiT weights (upload path).
//!
//! [`GpuFluxWeights::from_host`] lifts the host `FluxWeights` f32 tables onto
//! the GPU with one [`rdna_compute::GpuTensor`] per manifest key, at the
//! manifest shape `[rows, cols]` — a structural mirror of [`FluxWeights`]
//! (same key set, same row-major layout) so a block forward consumes GPU
//! tensors in exactly the order/index the CPU reference reads host tensors.
//!
//! **Dtype choice**: split by key, in [`upload_flux_tensor`] — `.weight`
//! is f16-resident because it is the WMMA GEMM's weight operand, while
//! `.bias` and `.scale` stay f32 for `bias_add_f32` and `rmsnorm_batched`.
//! The earlier plain-fp32 upload path put every
//! tensor f32 so the block-parity gate could not be blurred by a rounding
//! step; the GEMM rewrite made f16 weights the operand the kernel wants,
//! and block parity still holds because the accumulator remains f32.
//!
//! **ACTIVATIONS are f16 between kernels** as of the f16-activation rewrite.
//! The residual streams (`img`, `txt`, `fused`) and the modulation vectors
//! stay f32 — they are what the block accumulates into and what parity is
//! measured on — but everything that flows from one kernel into the next
//! inside a block is f16, produced directly by the kernel that computes it:
//!
//! * `layernorm_modulate` emits the f16 GEMM activation in one launch, so the
//!   LayerNorm / modulate / cast chain is a single kernel.
//! * The GEMM fused epilogues (`GemmEpilogue`) emit f16, fold in GELU, and do
//!   the gated residual accumulation in the store, so `gelu`, `gated_add` and
//!   the residual `copy_of` disappear.
//! * `qk_rmsnorm_rope_flux` does QK-RMSNorm + 2D RoPE in one launch with the
//!   dtype conversion at both ends, and the attention route takes f16 Q/K/V
//!   and stores f16.
//! * The qkv GEMMs write straight into their row range of the text-first
//!   joint concat, so the six per-double-block `copy_d2d` calls are gone; the
//!   `linear2` weight is split along K at upload time, so the `[n_all, 5d]`
//!   per-token concat is gone too.
//!
//! Measured on gfx1150 at real 3072/24/128 geometry
//! (per-call census): 1786 → 988 kernel launches per denoise step,
//! 190 → 0 host-synchronous D2D copies, 18.2 → 14.0 s/step projection.
//! `HIPFIRE_FLUX_F16_ACT=0` restores the all-f32 activation path — see
//! [`f16_activations_enabled`].
//!
//! The host tables are the source of truth — `free_gpu` returns all GPU
//! buffers to the pool without touching the host copy.

use crate::config::FluxDiffusionConfig;
use crate::f16_stage::F16Stage;
use crate::flux::{
    rope_ids_for_grid, text_ids, timestep_embedding, FinalAdaLNOrder, FluxForwardInput, FluxPlan,
    FluxWeights, MlpAct,
};
use crate::manifest::{expected_flux_keys, TS_EMBED_DIM};
use hipfire_runtime::model_source::ModelSource;
use rdna_compute::gemm::{GemmEpilogue, LdsTile};
use rdna_compute::profile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// LayerNorm epsilon shared by every weightless norm in the MMDiT block. Kept
/// as a constant so the fused `layernorm_modulate` call and the legacy
/// `layernorm_batched` call cannot drift apart.
const LN_EPS: f32 = 1e-6;

/// The fixed 128×128 / 32×64 k64 macro-tile, i.e. the tile
/// `Gpu::gemm_f16_x_f16_wmma_lds_auto` falls back to. Used by the f16 path
/// when `HIPFIRE_FLUX_GEMM_WIDE=0` pins the narrow tile for an A/B, because
/// the fused-epilogue entries are tile-parameterised (`Gpu::LDS_EPI_TILES`)
/// while the standalone `gemm_f16_x_f16_wmma_lds` kernel is not.
const NARROW_TILE: LdsTile = LdsTile::new(128, 128, 32, 64, 64, false);

/// Are activations carried between kernels as f16?
///
/// Default ON. `HIPFIRE_FLUX_F16_ACT=0` restores the all-f32 activation path:
/// every GEMM casts its activation to an f16 scratch first, GELU / modulate /
/// gated-add / QK-RMSNorm / RoPE / the `linear2` row assembly stay separate
/// launches, and the six per-double-block concat copies come back. That old
/// path is retained as the A/B reference and the escape hatch for a numerical
/// regression; it is ~2× the launches and ~2.5× the activation traffic.
///
/// Read once per process: the upload path (`upload_flux_key`) and the forward
/// must agree, because the f16 path splits `single_blocks.*.linear2.weight`
/// into two tensors at upload time and the f32 path keeps the fused one.
pub fn f16_activations_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FLUX_F16_ACT").map_or(true, |v| v != "0")
    })
}

/// Are the batch-1 modulation linears run as GEMVs into one per-forward
/// buffer ([`ModAll`]) instead of one 128-row WMMA macro-tile GEMM each?
///
/// Default ON. Every modulation linear is applied to a SINGLE `d`-wide vector
/// (`silu(vec)`) — 76 of them per denoise step at FLUX geometry — so the
/// 128-row tile streams the whole `[6d, d]` / `[3d, d]` weight to compute one
/// real output row and 127 rows of padding. A GEMV reads the same weight
/// bytes and does 1/128th of the arithmetic.
///
/// `HIPFIRE_FLUX_MOD_GEMV=0` restores the per-block `Gpuf::mod_linear` /
/// `Gpuf::linear` GEMM route, so the two can be A/B-ed in one session.
///
/// Read once per process, like [`f16_activations_enabled`], so a mid-run
/// environment change cannot make two blocks disagree.
pub fn mod_gemv_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FLUX_MOD_GEMV").map_or(true, |v| v != "0")
    })
}

// ─── Per-kernel-family step profiling (`HIPFIRE_PROFILE`) ──────────────────
//
// The 3.27 s/step gfx1151 profile had ~4% unattributed. This
// section makes the whole step visible per kernel family (`gemm.qkv`,
// `attn.v2`, `norm.ln_mod`, …) instead of only per denoise-loop STAGE (which
// `HIPFIRE_IMG_PROFILE` already covers).
//
// Deliberately does NOT go through `rdna_compute::profile::start()`/
// `is_active()`/`Timer`: several kernels on this exact path already carry a
// `Timer`-based instrumentation of their own (`attention_flux`,
// `layernorm_modulate`, `qk_rmsnorm_rope_flux`) that SYNCHRONIZES the stream
// in `Timer::finish()` — correct for isolated per-kernel bandwidth
// attribution (see that module's doc), wrong here: a FLUX step is ~1,000
// kernel launches, and syncing the host thread after each one would
// serialize the whole step and inflate the per-family sum far past the
// step's real wall time. `is_active()` is a single shared switch — turning
// it on here would also turn on those other call sites. Instead this uses
// `rdna_compute::profile::{begin_deferred, resolve_deferred}`, which only
// enqueue `hipEventRecord` (never a synchronous wait) and resolve the whole
// step with ONE synchronize at the end.
//
// Zero cost when `HIPFIRE_PROFILE` is unset: `step_profile_enabled()` reads
// the env var once (`OnceLock`, matching `f16_activations_enabled` and
// friends) and every wrapped call site is a single `bool` check before
// falling straight through to the unwrapped launch — no event, no
// allocation, no branch inside the hot GEMM/kernel call itself.

/// Is per-kernel-family step profiling active? Cached like the other
/// per-process flags in this file — this is checked at every wrapped call
/// site (hundreds per step).
fn step_profile_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| hipfire_config::developer_var_os("HIPFIRE_PROFILE").is_some())
}

/// One in-flight (start recorded, stop not yet recorded) deferred timer,
/// tagged with the family it will be attributed to once resolved.
type PendingFamilyTimer = (&'static str, profile::PendingTimer);

/// Collects one forward's kernel-family attribution. GPU launches are timed
/// with `rdna_compute::profile::PendingTimer` (see the section doc above for
/// why, instead of `profile::Timer`) and resolved into a per-family
/// microsecond map by `resolve`. The `host.sampler`/`host.other` spans this
/// forward's caller cares about (the per-step scheduler/bookkeeping work in
/// `pipeline.rs`'s denoise loop) are NOT collected here — they are plain
/// wall-clock `Instant` spans with nothing to defer, so `pipeline.rs` times
/// and merges them into the table itself, after `take_step_profile()`.
struct StepProfiler {
    enabled: bool,
    gpu: Vec<PendingFamilyTimer>,
}

impl StepProfiler {
    fn new() -> Self {
        Self {
            enabled: step_profile_enabled(),
            gpu: Vec::new(),
        }
    }

    /// Begin timing a GPU launch under `family`. Call immediately before the
    /// launch; pair with `end_gpu` immediately after. No-op when profiling
    /// is off, or if the event pair could not be created (profiling must
    /// never fail the forward).
    fn begin_gpu(
        &self,
        hip: &hip_bridge::HipRuntime,
        stream: Option<&hip_bridge::Stream>,
        family: &'static str,
    ) -> Option<PendingFamilyTimer> {
        if !self.enabled {
            return None;
        }
        profile::begin_deferred(hip, stream)
            .ok()
            .map(|t| (family, t))
    }

    /// Enqueue the stop-event record (non-blocking) and stash the pair for
    /// `resolve`. No-op if `begin_gpu` returned `None`.
    fn end_gpu(
        &mut self,
        hip: &hip_bridge::HipRuntime,
        stream: Option<&hip_bridge::Stream>,
        pending: Option<PendingFamilyTimer>,
    ) {
        if let Some((family, t)) = pending {
            let _ = t.mark_stop(hip, stream);
            self.gpu.push((family, t));
        }
    }

    /// Resolve every deferred GPU span (ONE stream sync, see
    /// `profile::resolve_deferred`), summed per family in microseconds.
    fn resolve(
        self,
        hip: &hip_bridge::HipRuntime,
        stream: Option<&hip_bridge::Stream>,
    ) -> BTreeMap<&'static str, f64> {
        let mut out = BTreeMap::new();
        for (family, us) in profile::resolve_deferred(hip, stream, self.gpu) {
            *out.entry(family).or_insert(0.0) += us;
        }
        out
    }
}

impl Default for StepProfiler {
    /// Only used as the placeholder `mem::take` leaves behind when a
    /// `Gpuf::finish()` pulls the real (accumulated) profiler out — never
    /// constructed as a working profiler (use `StepProfiler::new()`, which
    /// reads `HIPFIRE_PROFILE`).
    fn default() -> Self {
        Self {
            enabled: false,
            gpu: Vec::new(),
        }
    }
}

thread_local! {
    /// The most recently resolved step profile, stashed by `Gpuf::finish()`.
    /// Mirrors `rdna_compute::profile::start()`/`stop()`'s thread-local
    /// sink/take shape, but is a SEPARATE cell — see the section doc above
    /// for why this must not share `profile`'s `is_active()` gate.
    static LAST_STEP_PROFILE: RefCell<Option<BTreeMap<&'static str, f64>>> =
        const { RefCell::new(None) };
}

/// Take the per-kernel-family attribution table (family -> microseconds)
/// from the most recently finished profiled forward (any of
/// `gpu_forward`/`gpu_forward_txt_dev`/`gpu_forward_parts`/
/// `gpu_double_block`/`gpu_single_block`). `None` when `HIPFIRE_PROFILE` was
/// unset for that call, or none has run yet in this thread. Consumes the
/// stored table, like `rdna_compute::profile::stop()`.
pub fn take_step_profile() -> Option<BTreeMap<&'static str, f64>> {
    LAST_STEP_PROFILE.with(|p| p.borrow_mut().take())
}

/// The `attn.<variant>` family label for whichever FLUX attention route
/// [`Gpuf::attention_into`] is about to dispatch to. Calls the same
/// [`rdna_compute::attention::flux_attn_route_name`] that
/// `Gpu::attention_flux_best_f16kv_f32` (`crates/rdna-compute/src/attention.rs`)
/// dispatches on — `HIPFIRE_FLUX_ATTN` override, else the per-arch measured
/// default — so the label can never drift from the route that actually runs;
/// the two used to be independent matches kept in sync by hand.
///
/// Cached like every other per-process flag in this file (`developer_var` +
/// `arch.as_str()` on every call would otherwise cost an allocation per
/// attention launch — 57 times per FLUX step — even with profiling off; the
/// call site in [`Gpuf::attention_into`] also only reaches this when
/// `self.prof.enabled`, so the cost is paid at most once per process either
/// way). The resolver's error path (an unrecognised `HIPFIRE_FLUX_ATTN`
/// value) can't reach here first: `Gpuf::attention_into` always resolves the
/// route via the same function before dispatch, so an invalid override fails
/// there with the same message, not silently here.
fn flux_attn_route_family(gpu: &Gpu) -> &'static str {
    static V: OnceLock<&'static str> = OnceLock::new();
    *V.get_or_init(|| {
        let forced = hipfire_config::developer_var("HIPFIRE_FLUX_ATTN").ok();
        match rdna_compute::attention::flux_attn_route_name(gpu.arch.as_str(), forced.as_deref()) {
            Ok("v5") => "attn.v5",
            Ok("vt") => "attn.vt",
            Ok("vtk") => "attn.vtk",
            Ok("v2") => "attn.v2",
            Ok(other) => unreachable!("flux_attn_route_name returned an unknown route `{other}`"),
            Err(_) => "attn.vt",
        }
    })
}

/// Every block's modulation vector for ONE forward, in one f32 buffer.
///
/// Layout, in `d`-wide chunks (`d` = `hidden_size`):
///
/// ```text
/// [double 0 img 6d][double 0 txt 6d] … [double L-1 txt 6d][single 0 3d] …
/// ```
///
/// so `num_layers·12d + num_single_layers·3d` f32 in total — 1,050,624
/// elements (4.0 MB) at FLUX.1-dev geometry. One allocation replaces the 76
/// per-block alloc/free pairs the GEMM route made per step, and the block
/// bodies take `sub_offset` views instead of an owned tensor.
struct ModAll {
    buf: GpuTensor,
    d: usize,
    /// Element offset of the first single-block slot (`num_layers · 12d`).
    single_base: usize,
}

impl ModAll {
    /// Element offset of the first single-block slot: the double-block slots
    /// occupy `12d` each and come first. The one definition shared by
    /// `Gpuf::build_mod_all` and the slot-layout test.
    fn single_base(num_layers: usize, d: usize) -> usize {
        num_layers * 12 * d
    }

    /// Total element count of the buffer.
    fn total(num_layers: usize, num_single_layers: usize, d: usize) -> usize {
        Self::single_base(num_layers, d) + num_single_layers * 3 * d
    }

    /// Element offset of double block `b`'s image (`img = true`) or text slot.
    fn double_off(b: usize, d: usize, img: bool) -> usize {
        (b * 12 + usize::from(!img) * 6) * d
    }

    /// Element offset of single block `b`'s slot.
    fn single_off(single_base: usize, b: usize, d: usize) -> usize {
        single_base + b * 3 * d
    }

    /// The `[6d]` view for double block `b`'s image (`img = true`) or text
    /// stream.
    fn double(&self, b: usize, img: bool) -> GpuTensor {
        self.buf
            .sub_offset(Self::double_off(b, self.d, img), 6 * self.d)
    }

    /// The `[3d]` view for single block `b`.
    fn single(&self, b: usize) -> GpuTensor {
        self.buf
            .sub_offset(Self::single_off(self.single_base, b, self.d), 3 * self.d)
    }
}

/// Where a block body gets its modulation vector.
///
/// `dv` is `silu(vec)` and `dv16` its f16 cast (present exactly when
/// [`f16_activations_enabled`]); both are hoisted out of the block loop by
/// the caller. `all` is the pre-computed GEMV buffer when
/// [`mod_gemv_enabled`], and `None` on the retained GEMM route — in which
/// case the block runs its own batch-1 linear off `dv16`/`dv`.
struct ModSrc<'a> {
    dv: &'a GpuTensor,
    dv16: Option<&'a GpuTensor>,
    all: Option<&'a ModAll>,
}

/// One block's `6d`/`3d` modulation vector: either a borrowed view into
/// [`ModAll`] or a tensor this block allocated. `Gpuf::release_mod` frees only
/// the owned form — `Gpu::free_tensor` rejects a `sub_offset` view, loudly.
struct ModVec {
    t: GpuTensor,
    owned: bool,
}

/// Install the real HIP stream the diffusion path runs on. Call **once at
/// model-load time**, before any weight upload, from whatever owns the `Gpu`.
///
/// Without an active stream every `Gpu::copy_d2d` and `Gpu::zeros` falls
/// through to the synchronous legacy-stream `hipMemcpy`/`hipMemset`, which is
/// a host stall per call. With one they go async on a blocking stream, so the
/// synchronous `memcpy_htod` / `memcpy_dtoh` of the upload and download paths
/// stay correctly ordered against them (the legacy stream synchronises with
/// blocking streams).
///
/// **This is permanent by design and process-wide.** The stream stays
/// installed on the `Gpu` for the rest of its life and every later launch —
/// diffusion, VAE, T5, CLIP — goes to it. That is why it belongs at load
/// time, in one place, rather than in a per-forward constructor: an install
/// buried in the forward would change global behaviour as a side effect of
/// the first denoise step, and would make it look like a property of the
/// activation dtype. It is not. [`f16_activations_enabled`] selects the
/// activation LAYOUT only; it does not and must not decide whether the
/// diffusion path is asynchronous.
pub fn install_forward_stream(gpu: &mut Gpu) -> Result<(), String> {
    gpu.ensure_capture_stream()
        .map_err(|e| format!("flux gpu: install forward stream: {e:?}"))
}

/// Is the LDS-staged macro-tile GEMM route enabled? `HIPFIRE_FLUX_GEMM_LDS=0`
/// pins the 16-step kernel (and, on the f16 path, the unfused epilogue) for a
/// same-session A/B. Cached: `gemm_epi` is called ~700 times per denoise step
/// and a `getenv` per call is pure launch-path overhead.
fn gemm_lds_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FLUX_GEMM_LDS").map_or(true, |v| v != "0")
    })
}

/// Is the per-arch wide macro-tile selection enabled? `HIPFIRE_FLUX_GEMM_WIDE=0`
/// pins the fixed 128x128 tile. Cached for the same reason as
/// [`gemm_lds_enabled`].
fn gemm_wide_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_FLUX_GEMM_WIDE").map_or(true, |v| v != "0")
    })
}

// ─── Padded weight rows (`HIPFIRE_FLUX_WPAD`) ─────────────────────────────
//
// Task 4 measured the row pitch of the LDS GEMM's staging reads as the
// largest single effect on the FLUX census — see the ROW PITCH note in
// `kernels/src/gemm_f16_x_f16_wmma_lds256.hip`. A block stages `bm + bn`
// rows concurrently, each a short run at a stride of the row pitch; when
// that pitch is a multiple of 1024 BYTES those runs camp on a small set of
// DRAM channels. Every FLUX.1-dev K is such a pitch (3072 / 12288 / 15360
// halves = 6144 / 24576 / 30720 bytes). Storing the weight rows 64 elements
// further apart moves all three off it: the gfx1151 census drops 1.44 s ->
// 1.16 s per step and the 3072x12288x4608 shape goes from 36 % to 73 % of
// peak. Padding the ACTIVATIONS on top of that measured as a LOSS on
// gfx1151 (`PADA=64 PADX=0` 1.17 s vs `PADX=64` 1.25 s), so only the weights
// are padded and every call site below passes `ldx = k`.
//
// Bit-exact by construction, not by luck: the kernel sums the same K
// elements in the same order, and `_auto_ld` / `_auto_epi_ld` pick their
// tile from (arch, M, batch, CU) exactly as the packed entries do — the
// pitch is not an input to `Gpu::lds_tile_for`. `HIPFIRE_FLUX_WPAD=0`
// uploads every weight packed so both routes are reachable from one binary.

/// Row pad, in elements, added to a padded FLUX weight row.
///
/// 64 elements = 128 bytes of f16. The smallest pad that both keeps the pitch
/// a multiple of 16 elements (the staging `half16` loads are 32-byte vector
/// loads and only stay aligned if it is) and puts all three FLUX K values in
/// the `pitch = 2^7·odd` band that measured best on gfx1150 AND gfx1151. It
/// costs 2 % of the weight bytes (~0.5 GB on the 23.8 GB checkpoint).
const FLUX_WEIGHT_PAD: usize = 64;

/// Device bytes spent on weight-row pad so far, for the one-line report in
/// [`log_weight_pad_once`]. Relaxed: it is a diagnostic total, never read
/// back to make a decision.
///
/// Counted by the two weight uploaders ([`upload_mmdit_weight`],
/// [`upload_weight_f16`]), each once per PERSISTED weight, at
/// `m * (pitch - k) * DType::F16.size()` — NOT inside [`upload_padded`]
/// itself, which is also used to fill a transient f32 scratch that is freed
/// before this counter would matter. Counting inside `upload_padded` double
/// counted the eager (`from_host`) path: it pads an f32 scratch tensor that
/// is freed AND leaves the persistent f16 tensor (built via `gpu.zeros` +
/// `cast_f32_to_f16`) uncounted, so the freed scratch's f32-sized pad (4
/// bytes/elem) was reported instead of the persisted f16 tensor's actual pad
/// (2 bytes/elem) — 2x the real number.
static PAD_BYTES: AtomicUsize = AtomicUsize::new(0);

/// The per-row weight pad in elements: [`FLUX_WEIGHT_PAD`] by default, `0`
/// under `HIPFIRE_FLUX_WPAD=0`. Any other integer overrides the pad, so a new
/// arch can be swept without a rebuild; a value that is not a multiple of 16
/// is rejected loudly rather than silently misaligning every staging load.
///
/// Cached like [`gemm_lds_enabled`] — this is read once per GEMM call site.
fn weight_pad() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(
        || match hipfire_config::developer_var("HIPFIRE_FLUX_WPAD") {
            Ok(v) => {
                let pad: usize = v.parse().unwrap_or_else(|_| {
                    panic!(
                        "HIPFIRE_FLUX_WPAD must be a non-negative integer of elements, got `{v}`"
                    )
                });
                assert!(
                    pad % 16 == 0,
                    "HIPFIRE_FLUX_WPAD must be a multiple of 16 elements so the staging half16 \
                 loads stay 32-byte aligned (got {pad})"
                );
                pad
            }
            Err(_) => FLUX_WEIGHT_PAD,
        },
    )
}

/// Can the GEMM route that will actually run honour a row pitch?
///
/// Only the LDS-staged `_auto_ld` / `_auto_epi_ld` entries take `lda`/`ldx`.
/// `HIPFIRE_FLUX_GEMM_WIDE=0` routes [`Gpuf::gemm_pre`] to the non-tiled
/// `gemm_f16_x_f16_wmma_lds`, and `HIPFIRE_FLUX_GEMM_LDS=0` (or a K that is
/// not a multiple of 64) routes both entries to the 16-step
/// `gemm_f16_x_f16_wmma`; none of those three takes a pitch, and each would
/// read the pad as data. So a weight is padded only when the pitch-aware
/// route is the one selected — which also keeps both existing A/B knobs
/// measuring what they measured before instead of silently changing shape.
fn pitch_route_ok(k: usize) -> bool {
    k % 64 == 0 && gemm_lds_enabled() && gemm_wide_enabled()
}

/// Is `name` a modulation weight (`*_mod.lin`, `*.modulation.lin`,
/// `final_layer.adaLN_modulation.1`)?
///
/// Those stay PACKED. Their default consumer is `gemv_f16_bias_xf32`
/// ([`mod_gemv_enabled`]), which walks a weight row contiguously and has no
/// pitch argument at all; and on the forced-GEMM route
/// (`HIPFIRE_FLUX_MOD_GEMV=0`) they are batch-1 GEMMs whose cost is streaming
/// the weight once, not the staging pattern the pad fixes.
fn is_mod_weight(name: &str) -> bool {
    name.ends_with("_mod.lin.weight")
        || name.ends_with(".modulation.lin.weight")
        || name.ends_with(".adaLN_modulation.1.weight")
        || is_flux2_packed_weight(name)
}

/// The FLUX.2 (Klein) weights that must stay PACKED at K.
///
/// Same reasoning as [`is_mod_weight`]'s FLUX.1 set: the three shared
/// modulation linears and the final head's `norm_out.linear` are consumed by
/// [`Gpuf::mod_gemv`] / a batch-1 GEMM, and `gemv_f16_bias_xf32` walks a
/// weight row contiguously with no pitch argument. The two timestep-embedder
/// linears are batch-1 too (one row of activation per forward), so their cost
/// is streaming the weight once and a pad would only add device bytes.
///
/// Matched by FULL NAME, not by suffix. A `.linear.weight` suffix test would
/// also catch FLUX.1's `final_layer.linear.weight` — a `[patch_in, d]` table
/// that IS padded today — and silently change the FLUX.1 upload layout. None
/// of these names exists in a FLUX.1 manifest, so the FLUX.1 pitch decision
/// is bit-for-bit what it was.
fn is_flux2_packed_weight(name: &str) -> bool {
    matches!(
        name,
        "double_stream_modulation_img.linear.weight"
            | "double_stream_modulation_txt.linear.weight"
            | "single_stream_modulation.linear.weight"
            | "norm_out.linear.weight"
            | "time_guidance_embed.timestep_embedder.linear_1.weight"
            | "time_guidance_embed.timestep_embedder.linear_2.weight"
    )
}

/// Is `name` a FLUX.2 per-head QK-norm scale?
///
/// Klein is bias-free, so its `[head_dim]` QK-norm scales are spelled
/// `.weight` where FLUX.1 spells them `.scale` — but they are consumed by
/// `rmsnorm_batched`, which reads an **f32** weight vector. Uploading them
/// through the `.weight` → f16 rule would hand that kernel f16 words to read
/// as f32 and produce plausible garbage, so they take the `.scale` treatment
/// instead. No FLUX.1 (or CLIP/T5) key ends in one of these four suffixes.
fn is_flux2_norm_scale(name: &str) -> bool {
    name.ends_with(".attn.norm_q.weight")
        || name.ends_with(".attn.norm_k.weight")
        || name.ends_with(".attn.norm_added_q.weight")
        || name.ends_with(".attn.norm_added_k.weight")
}

/// Does this staged f16 table contain an infinity?
///
/// FLUX.2 Klein ships BF16, whose exponent range is f32's: a weight above
/// 65504 rounds to `±inf` on the way into the f16 GEMM operand and poisons
/// the whole forward with NaNs several blocks later, far from the cause.
/// Returns the index of the first `±inf` word. NaN patterns (0x7C01..0x7FFF)
/// are deliberately NOT matched — a checkpoint that already carries a NaN is
/// a different failure and must not be reported as an f16 overflow.
fn first_f16_inf(words: &[u16]) -> Option<usize> {
    words.iter().position(|w| w & 0x7FFF == 0x7C00)
}

/// The device row pitch, in elements, at which the FLUX weight `name` with a
/// logical row width of `k` is stored.
///
/// The ONE source of truth. The uploaders ([`upload_mmdit_weight`],
/// [`upload_weight_f16`]) shape the tensor `[rows, weight_pitch]`
/// and every GEMM call site derives its `lda` from the same call, so an
/// uploader and a reader cannot disagree about where row `m` starts.
/// [`Gpuf::wlda`] additionally cross-checks the answer against the shape
/// actually stored.
fn weight_pitch(name: &str, k: usize) -> usize {
    if !name.ends_with(".weight") {
        return k;
    }
    if is_mod_weight(name) || !pitch_route_ok(k) {
        return k;
    }
    k + weight_pad()
}

/// Copy `rows` rows of `k` elements from the packed `src` into `dst`, which
/// holds the same rows at a pitch of `pitch >= k` elements.
///
/// Columns `k..pitch` of every row are left untouched — the caller owns the
/// pad's contents. Every caller here hands in a zero-filled staging buffer
/// and reuses it across chunks, so the pad that reaches the device is ZERO.
/// Not because the kernel needs it to be: it only ever addresses `k` of each
/// row, and the GEMM pitch parity suite fills the pad with poison and passes.
/// Zero is chosen so the upload is deterministic and a device dump of a
/// padded weight is readable.
fn pad_rows_into<T: Copy>(src: &[T], dst: &mut [T], rows: usize, k: usize, pitch: usize) {
    assert!(
        pitch >= k,
        "pad_rows_into: pitch {pitch} is narrower than k {k}"
    );
    assert!(
        src.len() >= rows * k,
        "pad_rows_into: src holds {} elements, {rows} rows of {k} need {}",
        src.len(),
        rows * k
    );
    assert!(
        dst.len() >= rows * pitch,
        "pad_rows_into: dst holds {} elements, {rows} rows at pitch {pitch} need {}",
        dst.len(),
        rows * pitch
    );
    for r in 0..rows {
        dst[r * pitch..r * pitch + k].copy_from_slice(&src[r * k..(r + 1) * k]);
    }
}

/// Upload a packed `[m, k]` host buffer as an `[m, pitch]` device tensor,
/// `pitch > k`.
///
/// Chunked on purpose. The streamed weight path
/// ([`GpuFluxWeights::from_stream`]) is the product path and its whole point
/// is that host RSS stays bounded — materialising a padded copy of a whole
/// weight would undo that (`single_blocks.*.linear1.weight` alone is 132 MB
/// of f16 words at FLUX.1-dev geometry). One 4 MiB staging buffer covers
/// every weight, whatever its size: the pad columns are written once, at
/// construction, and [`pad_rows_into`] never touches them again.
///
/// Only reached when the weight is actually padded; a packed weight keeps
/// its pre-existing single-shot `Gpu::upload_f16_bits` / `Gpu::upload_f32`
/// call, so `HIPFIRE_FLUX_WPAD=0` runs byte-for-byte the upload path that
/// was here before.
fn upload_padded<T: Copy + Default>(
    gpu: &mut Gpu,
    data: &[T],
    m: usize,
    k: usize,
    pitch: usize,
    dtype: DType,
) -> Result<GpuTensor, String> {
    assert_eq!(
        std::mem::size_of::<T>(),
        dtype.size(),
        "upload_padded: element type must match {dtype:?}"
    );
    assert!(
        pitch > k,
        "upload_padded: only for a padded pitch (got {pitch} for k {k})"
    );
    if data.len() != m * k {
        return Err(format!(
            "upload_padded: {} elements for a [{m}, {k}] weight",
            data.len()
        ));
    }
    // `alloc_tensor` binds the thread, which the raw `memcpy_htod` below
    // relies on. The whole `[m, pitch]` region is written by the loop, pad
    // included, so an uninitialised allocation is fine here.
    let t = gpu
        .alloc_tensor(&[m, pitch], dtype)
        .map_err(|e| format!("upload_padded: alloc [{m}, {pitch}] {dtype:?}: {e:?}"))?;
    const CHUNK_BYTES: usize = 4 << 20;
    let row_bytes = pitch * dtype.size();
    let rows_per_chunk = (CHUNK_BYTES / row_bytes.max(1)).clamp(1, m.max(1));
    let mut stage: Vec<T> = vec![T::default(); rows_per_chunk * pitch];
    let mut row0 = 0usize;
    while row0 < m {
        let rows = rows_per_chunk.min(m - row0);
        pad_rows_into(
            &data[row0 * k..(row0 + rows) * k],
            &mut stage[..rows * pitch],
            rows,
            k,
            pitch,
        );
        let view = t.sub_offset(row0 * pitch, rows * pitch);
        let bytes = unsafe {
            std::slice::from_raw_parts(stage.as_ptr().cast::<u8>(), rows * pitch * dtype.size())
        };
        if let Err(e) = gpu.hip.memcpy_htod(&view.buf, bytes) {
            let msg = format!(
                "upload_padded: htod rows {row0}..{} of [{m}, {pitch}]: {e:?}",
                row0 + rows
            );
            // Don't leak the partially-written device allocation on a
            // mid-upload failure.
            let _ = gpu.free_tensor(t);
            return Err(msg);
        }
        row0 += rows;
    }
    Ok(t)
}

/// Report the weight-pad decision and what it cost, once per process.
///
/// One line, at the end of a weight-table build, so the number that appears
/// in a bench log is the WHOLE extra device cost of the pad rather than a
/// per-tensor drip.
fn log_weight_pad_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let extra = PAD_BYTES.load(Ordering::Relaxed);
        if extra == 0 {
            eprintln!(
                "flux gpu: weight rows PACKED at K (row pad {}, pitch-aware GEMM route {})",
                weight_pad(),
                if gemm_lds_enabled() && gemm_wide_enabled() {
                    "on"
                } else {
                    "off"
                }
            );
        } else {
            eprintln!(
                "flux gpu: weight rows padded +{} elems/row, +{:.1} MB of device weight bytes \
                 (HIPFIRE_FLUX_WPAD=0 to disable)",
                weight_pad(),
                extra as f64 / (1024.0 * 1024.0)
            );
        }
    });
}

/// Is `name` the fused `linear2` weight of a single block (`[d, 5d]`)?
fn is_single_linear2(name: &str) -> bool {
    name.starts_with("single_blocks.") && name.ends_with(".linear2.weight")
}

/// Split the fused single-block `linear2` weight `[d, k]` (row-major, rows =
/// output features) along **K** into `(W_attn [d, d], W_mlp [d, k - d])`.
///
/// The K axis, not the M axis: `linear2` consumes the per-token concatenation
/// `[att(d), mlp_g(k - d)]`, so the two halves are column ranges of every row.
/// Both outputs stay row-major and contiguous, which is what the WMMA GEMM
/// wants from its weight operand.
///
/// Generic over the element type so the eager f32 upload and the streamed
/// f16 upload (`stream_into`, which holds the weight as f16 bit patterns)
/// split through the same code and cannot disagree on the column ranges.
fn split_linear2<T: Copy>(data: &[T], d: usize, k: usize) -> (Vec<T>, Vec<T>) {
    let mut w_attn: Vec<T> = Vec::with_capacity(d * d);
    let mut w_mlp: Vec<T> = Vec::with_capacity(d * (k - d));
    for r in 0..d {
        let row = &data[r * k..(r + 1) * k];
        w_attn.extend_from_slice(&row[..d]);
        w_mlp.extend_from_slice(&row[d..]);
    }
    (w_attn, w_mlp)
}

/// GPU-resident FLUX.1 weights, one tensor per manifest key.
#[derive(Default)]
pub struct GpuFluxWeights {
    pub tensors: HashMap<String, GpuTensor>,
}

impl GpuFluxWeights {
    /// Upload every host weight to a GPU tensor of manifest shape `[rows, cols]`.
    ///
    /// Iterates the manifest's deterministic key list, so the tensor set and
    /// order are pinned by the same source that validated the real checkpoint
    /// (`expected_flux_keys`). Any key the host lacks is a hard error (named),
    /// matching the CPU `load_weights` contract.
    pub fn from_host(
        gpu: &mut Gpu,
        host: &FluxWeights,
        cfg: &FluxDiffusionConfig,
    ) -> Result<Self, String> {
        let mut tensors = HashMap::with_capacity(expected_flux_keys(cfg).len());
        for key in expected_flux_keys(cfg) {
            let t = host.get(&key.name);
            let shape = [key.rows, key.cols];
            upload_flux_key(gpu, &key.name, &t.data, shape, &mut tensors).map_err(|e| {
                format!(
                    "flux gpu: upload `{}` ({}x{}): {e}",
                    key.name, key.rows, key.cols
                )
            })?;
        }
        log_weight_pad_once();
        Ok(GpuFluxWeights { tensors })
    }

    /// Upload every manifest key STRAIGHT FROM THE CHECKPOINT, without ever
    /// building an f32 host table.
    ///
    /// [`from_host`](Self::from_host) needs a `FluxWeights` that already
    /// exists, which at FLUX.1-dev geometry is ~47 GB of host `Vec<f32>` held
    /// for the life of the bundle — on a unified-memory box that is what
    /// pushes the host into zram swap and, because "VRAM" is system RAM
    /// there, drags the GPU allocations into swap with it (the VAE decode
    /// measured 359 s against 3.96 s with memory free).
    ///
    /// This walks the same manifest key list in the same order and, per key:
    /// stages the checkpoint's BF16/F16/F32 bytes into a reusable host buffer
    /// as f16 words ([`FluxPlan::stage_f16`]), uploads them into an `F16`
    /// tensor, and tells the source to drop the pages. Peak host cost is the
    /// largest single tensor (`single_blocks.*.linear1.weight`, 132 MB of f16
    /// words) plus whatever the mmap has not been asked to release yet.
    ///
    /// **Numerically identical to `from_host`.** `upload_flux_tensor` uploaded
    /// f32 and cast on the device with `(_Float16)`, i.e. round-to-nearest-even;
    /// [`crate::f16_stage::f32_to_f16_rne`] is the same rounding on the host,
    /// with tests that pin it against a reference RNE over every bf16 and
    /// every f16 bit pattern. `.bias` / `.scale` keys stay f32 exactly as
    /// before — they are small (a few MB in total) and feed `bias_add_f32` /
    /// `rmsnorm_batched`, which read f32.
    pub fn from_stream(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &FluxPlan,
        cfg: &FluxDiffusionConfig,
    ) -> Result<Self, String> {
        plan.validate(cfg)?;
        let mut tensors: HashMap<String, GpuTensor> = HashMap::new();
        match Self::stream_into(gpu, src, plan, cfg, &mut tensors) {
            Ok(()) => Ok(GpuFluxWeights { tensors }),
            Err(e) => {
                // `GpuTensor` has no `Drop`. A failure partway through (the
                // usual one is the device running out of memory at 24 GB of
                // f16 weights) would otherwise strand every tensor uploaded
                // so far for the life of the process — and the caller's
                // natural response, retrying with a smaller model or falling
                // back to the host, would then start from a device that is
                // already full.
                let freed = free_partial(gpu, tensors);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    /// The upload loop, split out so the caller owns the partially-filled map
    /// on the error path and can return it to the pool. Uses `?` freely; every
    /// early return lands in `from_stream`'s cleanup arm.
    fn stream_into(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &FluxPlan,
        cfg: &FluxDiffusionConfig,
        tensors: &mut HashMap<String, GpuTensor>,
    ) -> Result<(), String> {
        let keys = expected_flux_keys(cfg);
        tensors.reserve(keys.len());
        let mut stage = F16Stage::new();
        for key in keys {
            let shape = [key.rows, key.cols];
            let g = if key.name.ends_with(".weight") && !is_flux2_norm_scale(&key.name) {
                let words = plan.stage_f16(src, &key, &mut stage)?;
                // Spec risk (FLUX.2 only, so the FLUX.1 load stays
                // byte-identical in behaviour AND in time): a BF16 weight
                // outside f16's range becomes `±inf` here and only shows up
                // as NaN output blocks later. Fail at the named table.
                if cfg.is_flux2() {
                    if let Some(i) = first_f16_inf(words) {
                        return Err(format!(
                            "flux gpu: `{}` element {i} overflows f16 (±inf after the BF16→f16 \
                             stage); this checkpoint needs a wider GEMM operand dtype",
                            key.name
                        ));
                    }
                }
                // The f16-activation forward reads the single-block `linear2`
                // weight as two K-halves (see `upload_flux_key`): apply the
                // same split here, so the streamed map and the eager map
                // expose the same keys.
                if f16_activations_enabled() && is_single_linear2(&key.name) {
                    let (d, k) = (key.rows, key.cols);
                    if k <= d || words.len() != d * k {
                        return Err(format!(
                            "flux gpu: `{}` shape {shape:?} is not the [d, 5d] fused linear2",
                            key.name
                        ));
                    }
                    let base = key
                        .name
                        .strip_suffix(".weight")
                        .expect("is_single_linear2 matched a `.linear2.weight` key");
                    let (w_attn, w_mlp) = split_linear2(words, d, k);
                    for (suffix, buf, cols) in
                        [("w_attn.weight", w_attn, d), ("w_mlp.weight", w_mlp, k - d)]
                    {
                        let name = format!("{base}.{suffix}");
                        let g = upload_weight_f16(gpu, &name, &buf, d, cols)?;
                        tensors.insert(name, g);
                    }
                    plan.release(src, &key.name);
                    continue;
                }
                upload_weight_f16(gpu, &key.name, words, key.rows, key.cols)?
            } else {
                let t = plan.tensor(src, &key)?;
                gpu.upload_f32(&t.data, &shape).map_err(|e| {
                    format!(
                        "flux gpu: upload `{}` ({}x{}) f32: {e:?}",
                        key.name, key.rows, key.cols
                    )
                })?
            };
            plan.release(src, &key.name);
            tensors.insert(key.name, g);
        }
        stage.clear();
        log_weight_pad_once();
        Ok(())
    }

    /// Borrow a single GPU weight tensor by manifest name.
    pub fn get(&self, name: &str) -> &GpuTensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("flux gpu: missing weight `{name}`"))
    }

    /// Return every GPU buffer to the pool. Exhaustive: consumes self, so a
    /// dropped field that forgets to free a tensor fails to compile here.
    ///
    /// Returns the number of buffers freed (assertable in a lab harness).
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        for (name, tensor) in self.tensors {
            gpu.free_tensor(tensor)
                .unwrap_or_else(|e| panic!("flux gpu: free `{name}`: {e:?}"));
            freed += 1;
        }
        freed
    }
}

/// Return a partially-built weight map to the pool, best-effort.
///
/// Deliberately NOT [`GpuFluxWeights::free_gpu`], which panics if a free
/// fails: this runs while unwinding a DIFFERENT failure, and replacing the
/// real error ("device out of memory uploading `single_blocks.31.linear1`")
/// with a panic from the cleanup would destroy the only useful diagnostic.
/// A free that fails here leaks one buffer and is silent; the caller still
/// gets the error that actually mattered.
pub(crate) fn free_partial(gpu: &mut Gpu, tensors: HashMap<String, GpuTensor>) -> usize {
    let mut freed = 0usize;
    for (_, t) in tensors {
        if gpu.free_tensor(t).is_ok() {
            freed += 1;
        }
    }
    freed
}

/// Upload one `[m, k]` weight already held as f16 bit patterns, at the row
/// pitch [`weight_pitch`] gives it.
///
/// The streamed (product) path's one weight uploader. A packed weight takes
/// the pre-existing single `Gpu::upload_f16_bits` call unchanged; a padded
/// one goes through the chunked [`upload_padded`], which never builds a
/// padded copy of the whole weight on the host.
fn upload_weight_f16(
    gpu: &mut Gpu,
    name: &str,
    words: &[u16],
    m: usize,
    k: usize,
) -> Result<GpuTensor, String> {
    let pitch = weight_pitch(name, k);
    if pitch == k {
        return gpu
            .upload_f16_bits(words, &[m, k])
            .map_err(|e| format!("flux gpu: upload `{name}` ({m}x{k}) f16: {e:?}"));
    }
    let t = upload_padded(gpu, words, m, k, pitch, DType::F16)
        .map_err(|e| format!("flux gpu: upload `{name}` ({m}x{k} @ pitch {pitch}) f16: {e}"))?;
    PAD_BYTES.fetch_add(m * (pitch - k) * DType::F16.size(), Ordering::Relaxed);
    Ok(t)
}

/// Upload one FLUX host tensor with the dtype the tuned GPU forward expects.
///
/// `.weight` keys feed the WMMA GEMM (`gemm_f16_x_f16_wmma`) as the
/// f16-resident weight operand, so they are uploaded f32→f16 once and the
/// f32 scratch is returned to the pool. `.bias` keys feed `bias_add_f32` and
/// `.scale` keys feed `rmsnorm_batched`, both of which read f32 — those stay
/// f32. Same-stream launch ordering makes the transient f32 scratch free
/// safe across back-to-back uploads.
///
/// Rows are PACKED at `shape[1]` — this entry never pads. It is shared with
/// the CLIP and T5 encoder uploads (`clip_gpu.rs`, `t5_gpu.rs`), whose GEMM
/// call sites have no pitch argument; the MMDiT weights that do go through
/// [`upload_mmdit_weight`] instead. T5's `K = 4096` sits on the same
/// 1024-byte pitch cliff and is a follow-up lever, not this one.
pub fn upload_flux_tensor(
    gpu: &mut Gpu,
    name: &str,
    data: &[f32],
    shape: [usize; 2],
) -> Result<GpuTensor, String> {
    if name.ends_with(".weight") {
        let scratch = gpu
            .upload_f32(data, &shape)
            .map_err(|e| format!("upload f32 scratch `{name}`: {e:?}"))?;
        let g = gpu
            .zeros(&shape, DType::F16)
            .map_err(|e| format!("alloc f16 `{name}`: {e:?}"))?;
        gpu.cast_f32_to_f16(&scratch, &g)
            .map_err(|e| format!("cast f16 `{name}`: {e:?}"))?;
        gpu.free_tensor(scratch)
            .map_err(|e| format!("free f32 scratch `{name}`: {e:?}"))?;
        Ok(g)
    } else {
        gpu.upload_f32(data, &shape)
            .map_err(|e| format!("upload f32 `{name}`: {e:?}"))
    }
}

/// [`upload_flux_tensor`] for an MMDiT manifest key, storing a `.weight` at
/// the row pitch [`weight_pitch`] gives it.
///
/// The eager (`from_host`) twin of [`upload_weight_f16`]. The pad is applied
/// to the f32 SCRATCH, not after the cast, so the f16 weight is still
/// produced by one flat `cast_f32_to_f16` over the whole `[rows, pitch]`
/// buffer — the same device-side round-to-nearest-even, element for element,
/// that the packed route has always used, with the pad columns f32 zeros
/// casting to f16 zeros. A packed weight takes [`upload_flux_tensor`]
/// unchanged, so `HIPFIRE_FLUX_WPAD=0` runs exactly the code that was here
/// before.
fn upload_mmdit_weight(
    gpu: &mut Gpu,
    name: &str,
    data: &[f32],
    shape: [usize; 2],
) -> Result<GpuTensor, String> {
    let pitch = weight_pitch(name, shape[1]);
    if !name.ends_with(".weight") || pitch == shape[1] {
        return upload_flux_tensor(gpu, name, data, shape);
    }
    let padded = [shape[0], pitch];
    let scratch = upload_padded(gpu, data, shape[0], shape[1], pitch, DType::F32)
        .map_err(|e| format!("upload padded f32 scratch `{name}`: {e}"))?;
    let g = gpu
        .zeros(&padded, DType::F16)
        .map_err(|e| format!("alloc f16 `{name}`: {e:?}"))?;
    gpu.cast_f32_to_f16(&scratch, &g)
        .map_err(|e| format!("cast f16 `{name}`: {e:?}"))?;
    gpu.free_tensor(scratch)
        .map_err(|e| format!("free f32 scratch `{name}`: {e:?}"))?;
    // Count the pad on the PERSISTED f16 tensor, not the freed f32 scratch
    // `upload_padded` filled above — see `PAD_BYTES`'s doc comment.
    PAD_BYTES.fetch_add(
        shape[0] * (pitch - shape[1]) * DType::F16.size(),
        Ordering::Relaxed,
    );
    Ok(g)
}

/// Upload one manifest key into `tensors`, applying the layout rewrites the
/// tuned forward wants. Prefer this over [`upload_flux_tensor`] for FLUX
/// weights — it is the single place that knows a key can expand to several
/// GPU tensors, so every builder of a [`GpuFluxWeights`] map agrees.
///
/// The one rewrite today is the single-block `linear2` weight `[d, 5d]`. Its
/// K axis interleaves the attention output (first `d` columns) with the MLP
/// output (remaining `4d`), which forced the forward to materialise a
/// `[n_all, 5d]` concat of `att` and `mlp_g` per block — a 283 MB strided
/// write plus a 283 MB read at FLUX geometry, ~1 GB of traffic per single
/// block. Splitting the weight along K instead lets the two halves run as two
/// GEMMs (`W_attn·att` then `W_mlp·mlp_g` with the first as the ADDIN
/// operand), so the concat buffer disappears entirely.
///
/// The split is done on the HOST, where the f32 table is already contiguous —
/// on the device it would be `d` strided copies per block. The two halves
/// together are exactly the same size as the fused original, so only ONE
/// layout is uploaded and the checkpoint's VRAM footprint is unchanged:
/// the split when [`f16_activations_enabled`], the fused original when the
/// kill switch is set. Keeping both would have cost an extra 3.6 GB at FLUX
/// geometry (`d x 5d` f16 per single block, 38 of them), which is why the
/// choice is exclusive rather than additive. Both readers gate on the same
/// function, so they cannot disagree.
pub fn upload_flux_key(
    gpu: &mut Gpu,
    name: &str,
    data: &[f32],
    shape: [usize; 2],
    tensors: &mut HashMap<String, GpuTensor>,
) -> Result<(), String> {
    if f16_activations_enabled() && is_single_linear2(name) {
        let (d, k) = (shape[0], shape[1]);
        if k <= d || data.len() != d * k {
            return Err(format!(
                "flux gpu: `{name}` shape {shape:?} is not the [d, 5d] fused linear2"
            ));
        }
        let base = name
            .strip_suffix(".weight")
            .expect("is_single_linear2 matched a `.linear2.weight` key");
        let (w_attn, w_mlp) = split_linear2(data, d, k);
        for (suffix, buf, cols) in [("w_attn.weight", w_attn, d), ("w_mlp.weight", w_mlp, k - d)] {
            let key = format!("{base}.{suffix}");
            let g = upload_mmdit_weight(gpu, &key, &buf, [d, cols])?;
            tensors.insert(key, g);
        }
        return Ok(());
    }
    if is_flux2_norm_scale(name) {
        // Same f32 exception `stream_into` makes — kept here so the eager and
        // the streamed builders expose the same dtype per key.
        let g = gpu
            .upload_f32(data, &shape)
            .map_err(|e| format!("flux gpu: upload `{name}` f32 qk-norm scale: {e:?}"))?;
        tensors.insert(name.to_string(), g);
        return Ok(());
    }
    let g = upload_mmdit_weight(gpu, name, data, shape)?;
    tensors.insert(name.to_string(), g);
    Ok(())
}

// ─────────────── GPU MMDiT forward ──────────────────────────

/// Run the FLUX.1 MMDiT forward entirely on the GPU with the fp32 primitives
/// (matmul, attention, 2D RoPE, modulate, gated-add, silu/gelu, layernorm),
/// returning the same named intermediates as the CPU [`crate::flux::forward_parts`]
/// so a parity harness (or the block-parity gate §11.1) can compare
/// structure-for-structure. Weights are the GPU-resident [`GpuFluxWeights`].
///
/// The GPU orchestration mirrors the CPU reference exactly (conditioning
/// additive 3072-vec; text-first joint attention; per-stream double blocks;
/// fused-input single blocks with row-split qkv/mlp GEMMs and a per-token
/// cat for the `linear2` input; SiLU-then-Linear final head). Allocates its
/// own scratch; callers wanting a compact result use
/// [`gpu_forward`](Self::forward). Not a hot path — correctness-first.
pub fn gpu_forward_parts(
    gpu: &mut Gpu,
    cfg: &FluxDiffusionConfig,
    gw: &GpuFluxWeights,
    input: &FluxForwardInput,
) -> Result<Vec<(String, Vec<f32>)>, String> {
    let mut f = Gpuf::new(gpu, gw, cfg)?;
    let parts = f.forward_parts(cfg, input, true, None);
    f.finish()?;
    parts
}

/// Run the forward and return ONLY the final image stream `[n_img, patch_in]`.
///
/// Not a thin wrapper over [`gpu_forward_parts`]: that one downloads every
/// named intermediate for the parity harness, which at real geometry is about
/// **1.19 GB of device→host copies and 43 synchronous stalls per denoise
/// step** — all of it discarded here. Skipping the collection is the whole
/// point of this entry, and it is the one the denoise loop uses.
pub fn gpu_forward(
    gpu: &mut Gpu,
    cfg: &FluxDiffusionConfig,
    gw: &GpuFluxWeights,
    input: &FluxForwardInput,
) -> Result<Vec<f32>, String> {
    let mut f = Gpuf::new(gpu, gw, cfg)?;
    let parts = f.forward_parts(cfg, input, false, None);
    f.finish()?;
    parts?
        .pop()
        .map(|(_, v)| v)
        .ok_or_else(|| "gpu_forward: empty parts".to_string())
}

/// [`gpu_forward`] with the text stream **already on the device**.
///
/// `txt_dev` is an f32 `[n_txt, txt_hidden_dim]` tensor — the T5 encoder's
/// `last_hidden_state`, produced by [`crate::t5_gpu::encode`] (or uploaded
/// once from the host path). `input.txt` is ignored, so the caller passes an
/// empty vector.
///
/// This exists because the denoise loop calls the forward once per step and
/// `forward_parts` uploads `input.txt` every time it is called: at FLUX
/// geometry that is 256×4096 f32 = **4 MB of host→device traffic per step**,
/// re-uploading a tensor that cannot change within a generation. The
/// conditioning cache in `pipeline.rs` holds the device tensor for the whole
/// generation (and, on a cache hit, across generations), so this entry point
/// is what makes "upload once" possible.
///
/// The forward does NOT free `txt_dev` — it stays owned by the caller/cache.
pub fn gpu_forward_txt_dev(
    gpu: &mut Gpu,
    cfg: &FluxDiffusionConfig,
    gw: &GpuFluxWeights,
    input: &FluxForwardInput,
    txt_dev: &GpuTensor,
    n_txt: usize,
) -> Result<Vec<f32>, String> {
    let mut f = Gpuf::new(gpu, gw, cfg)?;
    let parts = f.forward_parts(cfg, input, false, Some((txt_dev, n_txt)));
    f.finish()?;
    parts?
        .pop()
        .map(|(_, v)| v)
        .ok_or_else(|| "gpu_forward_txt_dev: empty parts".to_string())
}

/// Run one GPU double block and return the downloaded `(img_out, txt_out)`
/// streams. Inputs/`vec` are hidden-width GPU tensors; `n_txt` is used only
/// to place the 2D RoPE (image rows sit after the text rows in the joint
/// concat). Public so the real-weight block-parity gate targets one block.
pub fn gpu_double_block(
    gpu: &mut Gpu,
    cfg: &FluxDiffusionConfig,
    gw: &GpuFluxWeights,
    b: usize,
    img: &GpuTensor,
    txt: &GpuTensor,
    vec: &GpuTensor,
    n_img: usize,
    n_txt: usize,
    grid: (usize, usize),
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut f = Gpuf::new(gpu, gw, cfg)?;
    // The block bodies take `silu(vec)`, which the full forward hoists out of
    // the block loop; a single-block entry point has to produce it itself.
    let (dv, dv16) = f.mod_vectors(cfg, vec)?;
    // Only block `b`'s weights need to be uploaded for this entry point, so
    // the GEMV buffer is filled for that block alone; the rest of the
    // index-addressed buffer is never read. See `Gpuf::build_mod_all`.
    let mod_all = match mod_gemv_enabled() {
        true => Some(f.build_mod_all(cfg, &dv, b..b + 1, 0..0)?),
        false => None,
    };
    let ms = ModSrc {
        dv: &dv,
        dv16: dv16.as_ref(),
        all: mod_all.as_ref(),
    };
    let (oi, ot) = f.double_block(cfg, b, img, txt, &ms, n_img, n_txt, grid)?;
    let out = (f.download(&oi)?, f.download(&ot)?);
    f.free(oi)?;
    f.free(ot)?;
    if let Some(m) = mod_all {
        f.free(m.buf)?;
    }
    if let Some(t) = dv16 {
        f.free(t)?;
    }
    f.free(dv)?;
    f.finish()?;
    Ok(out)
}

/// Run one GPU single block over a fused `[n_all, hidden]` tensor and return
/// the downloaded next fused stream. Public for the block-parity gate.
pub fn gpu_single_block(
    gpu: &mut Gpu,
    cfg: &FluxDiffusionConfig,
    gw: &GpuFluxWeights,
    b: usize,
    fused: &GpuTensor,
    vec: &GpuTensor,
    n_img: usize,
    grid: (usize, usize),
    act: MlpAct,
) -> Result<Vec<f32>, String> {
    let mut f = Gpuf::new(gpu, gw, cfg)?;
    let (dv, dv16) = f.mod_vectors(cfg, vec)?;
    // Block `b` only — see the note in [`gpu_double_block`].
    let mod_all = match mod_gemv_enabled() {
        true => Some(f.build_mod_all(cfg, &dv, 0..0, b..b + 1)?),
        false => None,
    };
    let ms = ModSrc {
        dv: &dv,
        dv16: dv16.as_ref(),
        all: mod_all.as_ref(),
    };
    let o = f.single_block(cfg, b, fused, &ms, n_img, grid, act)?;
    let out = f.download(&o)?;
    f.free(o)?;
    if let Some(m) = mod_all {
        f.free(m.buf)?;
    }
    if let Some(t) = dv16 {
        f.free(t)?;
    }
    f.free(dv)?;
    f.finish()?;
    Ok(out)
}

struct Gpuf<'a> {
    gpu: &'a mut Gpu,
    gw: &'a GpuFluxWeights,
    cfg: &'a FluxDiffusionConfig,
    /// Cached weightless-layernorm affine pair `(d, gamma=ones, beta=zeros)`.
    /// The pair is constant, so uploading it per call cost 114 host→device
    /// uploads per denoise step and leaked both buffers every time. Held for
    /// the lifetime of the `Gpuf` and released by [`Gpuf::finish`].
    ln_affine: Option<(usize, GpuTensor, GpuTensor)>,
    /// f16 activations between kernels (see [`f16_activations_enabled`]).
    f16_act: bool,
    /// dtype of the attention Q and the attention output. F16 only when the
    /// f16 activation path is on AND the tuned FLUX attention route is the one
    /// that has f16 Q/out instantiations (`vt`/`vtk`, hd 128, wave32 WMMA).
    /// `HIPFIRE_FLUX_ATTN=v5` and the portable `attention_dflash_f32` fallback
    /// are both f32-only, so they pin this back to F32.
    qo_dt: DType,
    /// dtype of K and V. F16 whenever the WMMA FLUX attention family is
    /// reachable at all — every member of it takes f16 K/V — else F32 for the
    /// portable `attention_dflash_f32` fallback.
    kv_dt: DType,
    /// Per-kernel-family attribution for this forward. See the "Per-kernel-
    /// family step profiling" section doc above.
    prof: StepProfiler,
}

impl<'a> Gpuf<'a> {
    fn new(
        gpu: &'a mut Gpu,
        gw: &'a GpuFluxWeights,
        cfg: &'a FluxDiffusionConfig,
    ) -> Result<Self, String> {
        let f16_act = f16_activations_enabled();
        // The WMMA FLUX attention family is the one `Gpuf::attention_into`
        // routes to; it is hd-128 wave32 only, and every member takes f16 K/V.
        let wmma_attn = cfg.head_dim == 128 && gpu.arch_caps.has_wmma_w32();
        let v5 = hipfire_config::developer_var("HIPFIRE_FLUX_ATTN").is_ok_and(|v| v == "v5");
        let qo_f16 = f16_act && wmma_attn && !v5;
        Ok(Self {
            gpu,
            gw,
            cfg,
            ln_affine: None,
            f16_act,
            qo_dt: if qo_f16 { DType::F16 } else { DType::F32 },
            kv_dt: if f16_act && wmma_attn {
                DType::F16
            } else {
                DType::F32
            },
            prof: StepProfiler::new(),
        })
    }

    /// Release everything the `Gpuf` itself owns. Call once, at the end of a
    /// forward. Intermediates are freed at their own scope end; this only
    /// covers the caches that outlive a single call.
    ///
    /// When `HIPFIRE_PROFILE` is set, also resolves the accumulated
    /// per-family GPU timers (one `hipStreamSynchronize`/`hipEventSynchronize`
    /// here, not one per launch — see `StepProfiler`) and stashes the result
    /// for `take_step_profile()`. `mem::take` moves the real profiler out of
    /// `self.prof` (leaving `StepProfiler::default()`, an inert placeholder)
    /// so this can run before the `&mut self` borrows below without cloning.
    fn finish(&mut self) -> Result<(), String> {
        let prof = std::mem::take(&mut self.prof);
        if prof.enabled {
            let table = prof.resolve(&self.gpu.hip, self.gpu.active_stream.as_ref());
            LAST_STEP_PROFILE.with(|p| *p.borrow_mut() = Some(table));
        }
        if let Some((_, gamma, beta)) = self.ln_affine.take() {
            self.free(gamma)?;
            self.free(beta)?;
        }
        Ok(())
    }

    /// An **uninitialized** `[shape]` F32 tensor. Use where the next kernel
    /// pure-assigns every element — which is the case for every GEMM output,
    /// every elementwise output, and every buffer this file fills by copy.
    /// `zeros` costs a full-size memset that the following kernel overwrites;
    /// at FLUX shapes that memset moves as many bytes as the kernel does.
    /// Use [`Gpuf::zeros`] only where the zero content is actually read.
    fn alloc(&mut self, shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, DType::F32)
            .map_err(|e| format!("flux gpu: alloc {shape:?}: {e:?}"))
    }

    /// A zero-filled `[shape]` F32 tensor. Only for buffers whose zero content
    /// is read (the absent-bias vector). Everything else wants [`Gpuf::alloc`].
    fn zeros(&mut self, shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .zeros(shape, DType::F32)
            .map_err(|e| format!("flux gpu: alloc {shape:?}: {e:?}"))
    }

    /// Return a tensor's buffer to the pool. `GpuTensor`/`DeviceBuffer` have no
    /// `Drop`, so a tensor that is merely dropped leaks its device allocation
    /// for the life of the process: the forward leaked 81.9 GB per denoise
    /// step before these frees existed. Never pass a `sub_offset` view —
    /// `Gpu::free_tensor` rejects borrowed buffers, loudly.
    fn free(&mut self, t: GpuTensor) -> Result<(), String> {
        self.gpu
            .free_tensor(t)
            .map_err(|e| format!("flux gpu: free: {e:?}"))
    }

    fn download(&mut self, t: &GpuTensor) -> Result<Vec<f32>, String> {
        let timer = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "host.io");
        let r = self
            .gpu
            .download_f32(t)
            .map_err(|e| format!("flux gpu: download: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), timer);
        r
    }

    /// Per-token assemble: copy `chunks` (each a contiguous `[n_rows, len]`
    /// buffer) into `dst` at row `t` starting column `dst_col`. Used for the
    /// single-block `linear2` input cat `[att(d), mlp_g(4d)]` per token, made
    /// necessary because the fused weight interleaves the two along K.
    ///
    /// One strided-copy kernel launch per chunk. The row-at-a-time `copy_d2d`
    /// loop this replaced issued `n_rows × chunks` copies — 9216 per single
    /// block at FLUX geometry — and ran launch-bound at ~14 GB/s; the kernel
    /// runs bandwidth-bound at ~75 GB/s of an 89.6 GB/s roof on gfx1150.
    fn assemble_rows(
        &mut self,
        dst: &GpuTensor,
        dst_row_stride: usize,
        n_rows: usize,
        chunks: &[(usize, &GpuTensor, usize)],
    ) -> Result<(), String> {
        for &(dcol, src, len) in chunks {
            self.gpu
                .copy_rows_strided_f32(src, dst, n_rows, len, len, dst_row_stride, dcol)
                .map_err(|e| format!("assemble_rows chunk@{dcol}: {e:?}"))?;
        }
        Ok(())
    }

    /// `y = x·Wᵀ + b` where `b` is the optional bias. `w` is the full weight
    /// tensor `[out, in]` (`w.shape[0]` = `out`); if the manifest has no
    /// `{base}.bias`, a zero buffer is used with `has_bias=false`.
    fn linear(
        &mut self,
        wname: &str,
        x: &GpuTensor,
        in_dim: usize,
        n: usize,
        family: &'static str,
    ) -> Result<GpuTensor, String> {
        let w = self.gw.get(wname);
        let out = w.shape[0];
        let lda = Self::wlda(w, wname, in_dim)?;
        // A layer with no `.bias` key still needs a bias argument, so it gets a
        // zero vector — the one place in this file where zero CONTENT is read
        // rather than immediately overwritten. It is owned scratch, so it is
        // freed after the GEMM instead of leaking one vector per call.
        let bias_key = wname
            .strip_suffix(".weight")
            .map(|base| format!("{base}.bias"));
        let real_bias = bias_key.as_ref().and_then(|bk| self.gw.tensors.get(bk));
        let held_bias = match real_bias {
            Some(_) => None,
            None => Some(self.zeros(&[out])?),
        };
        let bias_ref = match (real_bias, &held_bias) {
            (Some(bt), _) => bt,
            (None, Some(h)) => h,
            (None, None) => unreachable!("held_bias is set whenever there is no bias tensor"),
        };
        let y = self.gemm(
            x,
            w,
            bias_ref,
            n,
            out,
            in_dim,
            real_bias.is_some(),
            family,
            lda,
        )?;
        if let Some(h) = held_bias {
            self.free(h)?;
        }
        Ok(y)
    }

    /// `y = a·bᵀ + bias`, `[m, n]`, with explicit dims (so a sub_offset weight
    /// view whose `shape` no longer reads as `[n,k]` still works). Routes the
    /// linear through the tuned WMMA f16×f16→f32 GEMM (`gemm_f16_x_f16_wmma`):
    /// `b` is the f16-resident weight `[n, k]`, `a` is the f32 activation
    /// `[m, k]` (cast to an f16 scratch on the fly), and the f32 output `[m, n]`
    /// gets an optional broadcast bias-add. The WMMA kernel tiles K in steps
    /// of 16, so K must be a multiple of 16 (true for every FLUX.1-dev
    /// linear; M and batch are bounds-checked inside the kernel, so batch=1
    /// modulation linears are fine). Same-stream launch ordering keeps the
    /// scratch cast/free safe across back-to-back calls.
    ///
    /// `lda` is `b`'s device row pitch in elements — see [`weight_pitch`].
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
        family: &'static str,
        lda: usize,
    ) -> Result<GpuTensor, String> {
        let a_f16 = self.cast_act(a, m, k)?;
        let y = self.gemm_pre(&a_f16, b, bias, m, n, k, has_bias, family, lda)?;
        self.free(a_f16)?;
        Ok(y)
    }

    /// The device row pitch of the weight `w`, uploaded under `wname` with a
    /// logical row width of `k`.
    ///
    /// Derives the pitch from [`weight_pitch`] — the same call the uploader
    /// made — and then CHECKS it against the shape actually stored. A reader
    /// that passes a different `k` than the manifest's `cols` would otherwise
    /// read the wrong bytes on every row past the first and return plausible
    /// wrong numbers; here it is a named error instead. Subsumes the
    /// `shape != [out, in]` check the callers used to do inline.
    fn wlda(w: &GpuTensor, wname: &str, k: usize) -> Result<usize, String> {
        let lda = weight_pitch(wname, k);
        if w.shape.len() != 2 || w.shape[1] != lda {
            return Err(format!(
                "flux gpu: {wname} shape {:?} is not [out, {lda}] (logical K {k}, row pad {})",
                w.shape,
                lda - k
            ));
        }
        Ok(lda)
    }

    /// Cast an f32 activation `[m, k]` to an f16 scratch for the WMMA GEMM.
    /// Split out of [`Gpuf::gemm`] so a block that feeds ONE activation to
    /// several GEMMs casts it once: `qkv_prep` runs 3 GEMMs on the same `h`
    /// and `single_block` runs 4 on the same `x_mod`, which cost 3–4 casts of
    /// the same buffer per block. The scratch is fully written by the cast, so
    /// it is allocated uninitialized.
    fn cast_act(&mut self, a: &GpuTensor, m: usize, k: usize) -> Result<GpuTensor, String> {
        let a_f16 = self
            .gpu
            .alloc_tensor(&[m, k], DType::F16)
            .map_err(|e| format!("flux gpu: alloc act f16: {e:?}"))?;
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "elem.cast");
        let r = self
            .gpu
            .cast_f32_to_f16(a, &a_f16)
            .map_err(|e| format!("flux gpu: cast act f16: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r?;
        Ok(a_f16)
    }

    /// [`Gpuf::gemm`] over an activation the caller already cast to f16. The
    /// caller owns `a_f16` and frees it after its last GEMM.
    ///
    /// `lda` is `b`'s device row pitch in elements ([`weight_pitch`]); the
    /// activation is always packed, so `ldx = k`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_pre(
        &mut self,
        a_f16: &GpuTensor,
        b: &GpuTensor,
        bias: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
        family: &'static str,
        lda: usize,
    ) -> Result<GpuTensor, String> {
        if k == 0 || k % 16 != 0 {
            return Err(format!(
                "flux gpu: wmma gemm [{m}x{k}]·[{n}x{k}]: K must be a multiple of 16 (got {k})"
            ));
        }
        // A padded weight is only ever uploaded when `pitch_route_ok`, so a
        // pitch reaching a route that cannot express one is a bug in the
        // upload/read pairing, not a shape a caller can legitimately ask for.
        if lda != k && !pitch_route_ok(k) {
            return Err(format!(
                "flux gpu: wmma gemm [{m}x{k}]·[{n}x{k}]: row pitch {lda} but the pitch-aware \
                 LDS route is not selected"
            ));
        }
        // The GEMM pure-assigns every element of `y`, so it needs no memset.
        let y = self.alloc(&[m, n])?;
        // One span covers the whole GEMM (+ bias-add on the non-LDS route):
        // both/all of the branches below are `family`'s one shape class, just
        // a different kernel selection.
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), family);
        // WMMA: y[batch, m_out] = Σ_k W[m_out, k] · X[batch, k], with
        // W = b [n, k], X = a_f16 [m, k], m_out = n, batch = m.
        //
        // Prefer the LDS-staged 128×128 macro-tile kernel: it lifts arithmetic
        // intensity from 8 to 64 FLOP/byte and fuses the bias, measured ~10× on
        // the FLUX shapes. It
        // needs K % 64 == 0 — true of every FLUX.1-dev linear — so the 16-step
        // kernel stays the fallback for a ragged K.
        // `HIPFIRE_FLUX_GEMM_LDS=0` forces the old 16-step kernel, so the two
        // routes can be A/B-ed in one session on the real forward rather than
        // compared across commits.
        //
        // `_auto` picks the macro-tile per arch from measurement rather than
        // from a capability predicate. The wide tiles are worth 1.17x/1.28x/
        // 1.41x on gfx1150/gfx1151/gfx1100 over the fixed 128×128, and the
        // per-arch winners genuinely differ — gfx1100's best tile is a 27%
        // LOSS on gfx1151 — so a single tile is not available.
        // `HIPFIRE_FLUX_GEMM_WIDE=0` pins the fixed 128×128 kernel for A/B.
        // Both read through the process-cached accessors: this is on the
        // per-GEMM path (hundreds of calls per step), and a `getenv` per call
        // is both wasted work and a mid-run flip the rest of the forward would
        // not see.
        let lds_enabled = gemm_lds_enabled();
        let wide = gemm_wide_enabled();
        // Captured instead of `?`-propagated directly so `end_gpu` always
        // runs before this function returns, on every branch — otherwise an
        // early return on a launch error would skip it and leak `t`'s two
        // hipEvents (`PendingTimer` cannot free them itself: destroying a
        // HIP event needs a `HipRuntime` handle, which it deliberately does
        // not own — same reason `GpuTensor`/`DeviceBuffer` have no `Drop`).
        let r: Result<(), String> = if lds_enabled && k % 64 == 0 {
            let bias_arg = if has_bias { Some(bias) } else { None };
            if wide {
                // `lda` is the weight's stored pitch; the activation is
                // always packed, so `ldx = k`. Padding the activation as well
                // measured as a loss on gfx1151 (see the WPAD section above).
                self.gpu
                    .gemm_f16_x_f16_wmma_lds_auto_ld(b, a_f16, &y, bias_arg, n, k, m, lda, k)
                    .map_err(|e| format!("flux gpu: wmma lds auto gemm [{m}x{k}]·[{n}x{k}]: {e:?}"))
            } else {
                self.gpu
                    .gemm_f16_x_f16_wmma_lds(b, a_f16, &y, bias_arg, n, k, m)
                    .map_err(|e| format!("flux gpu: wmma lds gemm [{m}x{k}]·[{n}x{k}]: {e:?}"))
            }
        } else {
            self.gpu
                .gemm_f16_x_f16_wmma(b, a_f16, &y, n, k, m)
                .map_err(|e| format!("flux gpu: wmma gemm [{m}x{k}]·[{n}x{k}]: {e:?}"))
                .and_then(|()| {
                    if has_bias {
                        self.gpu
                            .bias_add_f32(&y, bias, m, n)
                            .map_err(|e| format!("flux gpu: bias_add: {e:?}"))
                    } else {
                        Ok(())
                    }
                })
        };
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r?;
        Ok(y)
    }

    // ─── f16-activation path helpers ────────────────────────────────────
    //
    // Everything below exists to keep activations in f16 between kernels.
    // They are only reached when `self.f16_act`; `HIPFIRE_FLUX_F16_ACT=0`
    // routes the block bodies to the f32 helpers above instead.

    /// An **uninitialized** `[shape]` tensor of an explicit dtype. Same
    /// contract as [`Gpuf::alloc`] — every caller pure-assigns the buffer.
    fn alloc_dt(&mut self, shape: &[usize], dtype: DType) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, dtype)
            .map_err(|e| format!("flux gpu: alloc {shape:?} {dtype:?}: {e:?}"))
    }

    /// The tuned WMMA GEMM with a fused epilogue, writing into a
    /// caller-owned `y`: `Y[b, m] = Σ_k W[m, k] X[b, k] + bias[m]` plus
    /// whatever `epi` selects (see [`GemmEpilogue`] for the evaluation order).
    ///
    /// Unlike [`Gpuf::gemm_pre`] this does not allocate the destination — the
    /// point of the epilogue is that the destination is frequently something
    /// the caller already owns: a row range of a text-first concat buffer, or
    /// the residual stream being updated in place.
    ///
    /// The fused-epilogue entries exist only for the LDS-staged kernel and
    /// only for `K % 64 == 0`. A ragged K, or `HIPFIRE_FLUX_GEMM_LDS=0`,
    /// falls back to [`Gpuf::gemm_epi_unfused`], which reproduces the same
    /// arithmetic with the pre-existing standalone kernels — slower, but it
    /// keeps both A/B knobs meaningful on the f16 path and gives the fused
    /// kernels an independent cross-check at lab geometry.
    #[allow(clippy::too_many_arguments)]
    fn gemm_epi(
        &mut self,
        w_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        bias: Option<&GpuTensor>,
        batch: usize,
        m: usize,
        k: usize,
        epi: &GemmEpilogue<'_>,
        family: &'static str,
        lda: usize,
    ) -> Result<(), String> {
        // Both routes tile K, the fused one in steps of 64 and the unfused
        // fallback in steps of 16. Check the weaker bound here so a bad K is
        // named at the FLUX call site rather than inside a kernel launcher.
        if k == 0 || k % 16 != 0 {
            return Err(format!(
                "flux gpu: gemm_epi [{batch}x{k}]·[{m}x{k}]: K must be a nonzero multiple of 16 \
                 (got {k})"
            ));
        }
        // Same invariant as `gemm_pre`: neither the unfused fallback below nor
        // the narrow-tile arm can express a pitch, and `weight_pitch` never
        // pads a weight whose consumer is one of them.
        if lda != k && !pitch_route_ok(k) {
            return Err(format!(
                "flux gpu: gemm_epi [{batch}x{k}]·[{m}x{k}]: row pitch {lda} but the pitch-aware \
                 LDS route is not selected"
            ));
        }
        if !gemm_lds_enabled() || k % 64 != 0 {
            return self.gemm_epi_unfused(w_f16, x_f16, y, bias, batch, m, k, epi, family);
        }
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), family);
        let r = if gemm_wide_enabled() {
            // Weight at its stored pitch, activation packed — see `gemm_pre`.
            self.gpu.gemm_f16_x_f16_wmma_lds_auto_epi_ld(
                w_f16, x_f16, y, bias, m, k, batch, epi, lda, k,
            )
        } else {
            self.gpu.gemm_f16_x_f16_wmma_lds_epi(
                w_f16,
                x_f16,
                y,
                bias,
                m,
                k,
                batch,
                NARROW_TILE,
                epi,
            )
        };
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r.map_err(|e| format!("flux gpu: wmma lds {epi:?} [{batch}x{k}]·[{m}x{k}]: {e:?}"))
    }

    /// [`Gpuf::gemm_epi`] decomposed into the standalone kernels, for the
    /// shapes and flag states the fused entries do not cover — i.e. only when
    /// `HIPFIRE_FLUX_GEMM_LDS=0` or `K % 64 != 0`, so the GEMM itself is
    /// always the 16-step kernel here.
    ///
    /// Applies the `GemmEpilogue` steps in the documented order: `acc`,
    /// `+= addin`, `+= bias`, `gelu`, then the gated store. The result is the
    /// same value **up to floating-point reassociation**, not bit-identical:
    /// the fused kernel adds `addin` before the bias and keeps the whole
    /// epilogue in one f32 register chain, while this walks the same adds as
    /// separate passes over memory in a different order. The elementwise
    /// steps run in place on the f32 accumulator — each is a
    /// thread-per-element map at the same flat index, the same aliasing
    /// `bias_add_f32` already relies on.
    #[allow(clippy::too_many_arguments)]
    fn gemm_epi_unfused(
        &mut self,
        w_f16: &GpuTensor,
        x_f16: &GpuTensor,
        y: &GpuTensor,
        bias: Option<&GpuTensor>,
        batch: usize,
        m: usize,
        k: usize,
        epi: &GemmEpilogue<'_>,
        family: &'static str,
    ) -> Result<(), String> {
        let acc = self.alloc(&[batch, m])?;
        // One span over the whole decomposed sequence: it is `family`'s one
        // GEMM, just spelled as several launches instead of one fused kernel
        // (this fallback is only reached with `HIPFIRE_FLUX_GEMM_LDS=0` or a
        // ragged K — not the default path).
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), family);
        // Captured in a closure, rather than `?`-propagated inline, so
        // `end_gpu` always runs before this function returns — an early
        // return on any one of these launches erroring would otherwise skip
        // it and leak `t`'s two hipEvents (see `gemm_pre`'s comment on why
        // `PendingTimer` cannot free them itself).
        let r: Result<(), String> = (|| {
            self.gpu
                .gemm_f16_x_f16_wmma(w_f16, x_f16, &acc, m, k, batch)
                .map_err(|e| format!("flux gpu: unfused gemm [{batch}x{k}]·[{m}x{k}]: {e:?}"))?;
            if let Some(b) = bias {
                self.gpu
                    .bias_add_f32(&acc, b, batch, m)
                    .map_err(|e| format!("flux gpu: unfused bias_add: {e:?}"))?;
            }
            // ADDIN lands before the bias in the fused kernel; both are pure
            // adds into the same f32 accumulator, so applying it after the
            // bias-add above is the same value up to fp reassociation.
            if let Some(c) = epi.addin {
                self.gpu
                    .add_f32(&acc, c, &acc)
                    .map_err(|e| format!("flux gpu: unfused addin: {e:?}"))?;
            }
            if epi.gelu {
                self.gpu
                    .gelu_tanh_f32(&acc, &acc, batch * m)
                    .map_err(|e| format!("flux gpu: unfused gelu: {e:?}"))?;
            }
            match (epi.gate, epi.residual) {
                (Some(gate), Some(res)) => {
                    // `y = res + gate * acc`. `res` may alias `y`; when it does
                    // not, seed `y` with it first (the fused kernel reads and
                    // writes the same element, so this is the only way to
                    // spell it unfused).
                    if res.buf.as_ptr() != y.buf.as_ptr() {
                        self.gpu
                            .copy_d2d(res, y, batch * m * DType::F32.size())
                            .map_err(|e| format!("flux gpu: unfused residual seed: {e:?}"))?;
                    }
                    self.gpu
                        .gated_add_f32(y, gate, &acc, batch, m)
                        .map_err(|e| format!("flux gpu: unfused gated_add: {e:?}"))?;
                }
                _ if epi.out_f16 => {
                    self.gpu
                        .cast_f32_to_f16(&acc, y)
                        .map_err(|e| format!("flux gpu: unfused f16 store: {e:?}"))?;
                }
                _ => {
                    self.gpu
                        .copy_d2d(&acc, y, batch * m * DType::F32.size())
                        .map_err(|e| format!("flux gpu: unfused f32 store: {e:?}"))?;
                }
            }
            Ok(())
        })();
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r?;
        self.free(acc)
    }

    /// `silu(vec)` plus, on the f16 path, its f16 copy — the pair every block
    /// body's modulation linears take. `forward_parts` builds these once for
    /// the whole forward; the single-block public entry points have to build
    /// their own. Both are owned by the caller.
    fn mod_vectors(
        &mut self,
        cfg: &FluxDiffusionConfig,
        vec: &GpuTensor,
    ) -> Result<(GpuTensor, Option<GpuTensor>), String> {
        let dv = self.silu(vec)?;
        let dv16 = match self.f16_act {
            true => Some(self.cast_act(&dv, 1, cfg.hidden_size)?),
            false => None,
        };
        Ok((dv, dv16))
    }

    /// A modulation linear (`*_mod.lin` / `modulation.lin`) over the
    /// **pre-cast** `silu(vec)`.
    ///
    /// These three are the only GEMMs in a block whose activation is the same
    /// buffer for every block in the model, so routing them through
    /// [`Gpuf::linear`] — which casts f32→f16 inside [`Gpuf::gemm`] — cast the
    /// identical `[1, d]` vector 76 times per denoise step (2 per double
    /// block, 1 per single block), each with its own alloc/free pair. The
    /// caller casts once per forward and hands the f16 vector down.
    ///
    /// N comes from the weight's own row count (`6d` for a double-block
    /// stream, `3d` for a single block), so one helper covers both.
    fn mod_linear(&mut self, wname: &str, dv16: &GpuTensor, d: usize) -> Result<GpuTensor, String> {
        let w = self.gw.get(wname);
        // Modulation weights are never padded (`is_mod_weight`), so this
        // resolves to `lda = d` — the packed row the GEMV route also needs.
        let lda = Self::wlda(w, wname, d)?;
        let out = w.shape[0];
        let bias = wname
            .strip_suffix(".weight")
            .and_then(|base| self.gw.tensors.get(&format!("{base}.bias")))
            .ok_or_else(|| format!("flux gpu: {wname} has no `.bias` sibling"))?;
        // Same family as `mod_gemv` below — this is the GEMM-route spelling
        // of the same "modulation linear" computation.
        self.gemm_pre(dv16, w, bias, 1, out, d, true, "mod.gemv", lda)
    }

    /// Run the modulation linears for the named blocks as GEMVs into one
    /// buffer — the [`mod_gemv_enabled`] route.
    ///
    /// `doubles`/`singles` name which blocks to fill. `forward_parts` passes
    /// the full ranges; the single-block public entry points pass just their
    /// own block, because only that block's weights are uploaded in the
    /// harnesses that call them and `GpuFluxWeights::get` panics on a missing
    /// key. The buffer is always full-size and index-addressed, so an unfilled
    /// slot is simply never read.
    ///
    /// Activation is the f32 `dv`, NOT `dv16`: the GEMV accumulates in f32
    /// against f32 input, so it drops the activation's f16 round-trip that
    /// the WMMA route needs. The result is therefore close to, but not
    /// bit-identical to, [`Gpuf::mod_linear`].
    fn build_mod_all(
        &mut self,
        cfg: &FluxDiffusionConfig,
        dv: &GpuTensor,
        doubles: impl Iterator<Item = usize>,
        singles: impl Iterator<Item = usize>,
    ) -> Result<ModAll, String> {
        let d = cfg.hidden_size;
        let single_base = ModAll::single_base(cfg.num_layers, d);
        let total = ModAll::total(cfg.num_layers, cfg.num_single_layers, d);
        // Every slot a block reads is pure-assigned by a GEMV below, so the
        // buffer needs no memset.
        let buf = self.alloc(&[total])?;
        for b in doubles {
            for (i, stem) in ["img", "txt"].into_iter().enumerate() {
                let key = format!("double_blocks.{b}.{stem}_mod.lin.weight");
                let off = ModAll::double_off(b, d, i == 0);
                self.mod_gemv(&key, dv, &buf, off, 6 * d, d)?;
            }
        }
        for b in singles {
            let key = format!("single_blocks.{b}.modulation.lin.weight");
            let off = ModAll::single_off(single_base, b, d);
            self.mod_gemv(&key, dv, &buf, off, 3 * d, d)?;
        }
        Ok(ModAll {
            buf,
            d,
            single_base,
        })
    }

    /// One modulation linear as `y = W·dv + bias` straight into `dst[off..]`.
    ///
    /// The bias is OPTIONAL **only where the family says it is**: FLUX.1's
    /// `*_mod.lin` carries one, FLUX.2 Klein's three shared modulation
    /// linears are bias-free (`cfg.bias == false`), and then a missing
    /// `.bias` sibling passes `None` to `gemv_f16_bias_xf32`, which is
    /// exactly `y = W·dv`.
    ///
    /// On a `cfg.bias` (FLUX.1) checkpoint a missing sibling is still a hard
    /// error. Making the lookup a plain `Option` for Klein's sake would turn
    /// a `*_mod.lin.bias` that failed to stage into a silent `y = W·dv` —
    /// a numerically plausible image from a model missing six shift/scale
    /// offsets per block, instead of a named load failure. The guard below is
    /// what keeps that fail-closed; `mod_gemv_keeps_the_flux1_missing_bias_guard`
    /// asserts it is still here.
    fn mod_gemv(
        &mut self,
        wname: &str,
        dv: &GpuTensor,
        dst: &GpuTensor,
        off: usize,
        out: usize,
        d: usize,
    ) -> Result<(), String> {
        let w = self.gw.get(wname);
        // `gemv_f16_bias_xf32` walks a weight row contiguously and has no
        // pitch argument, so this is also the assertion that a modulation
        // weight really was uploaded PACKED (`is_mod_weight`): a padded row
        // would fail here rather than silently read across rows.
        if w.shape.len() != 2 || w.shape[0] != out || w.shape[1] != d {
            return Err(format!(
                "flux gpu: {wname} shape {:?} not [{out},{d}]",
                w.shape
            ));
        }
        let bias = wname
            .strip_suffix(".weight")
            .and_then(|base| self.gw.tensors.get(&format!("{base}.bias")));
        if bias.is_none() && self.cfg.bias {
            return Err(format!("flux gpu: {wname} has no `.bias` sibling"));
        }
        let y = dst.sub_offset(off, out);
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "mod.gemv");
        let r = self
            .gpu
            .gemv_f16_bias_xf32(w, dv, bias, &y, out, d)
            .map_err(|e| format!("flux gpu: mod gemv {wname}: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r
    }

    /// Double block `b`'s `[6d]` modulation vector for `stem` (`"img"` /
    /// `"txt"`) — a view into [`ModAll`] on the GEMV route, a fresh batch-1
    /// linear otherwise.
    fn mod_double(
        &mut self,
        ms: &ModSrc<'_>,
        b: usize,
        stem: &str,
        d: usize,
    ) -> Result<ModVec, String> {
        if let Some(all) = ms.all {
            return Ok(ModVec {
                t: all.double(b, stem == "img"),
                owned: false,
            });
        }
        self.mod_owned(ms, &format!("double_blocks.{b}.{stem}_mod.lin.weight"), d)
    }

    /// Single block `b`'s `[3d]` modulation vector. See [`Gpuf::mod_double`].
    fn mod_single(&mut self, ms: &ModSrc<'_>, b: usize, d: usize) -> Result<ModVec, String> {
        if let Some(all) = ms.all {
            return Ok(ModVec {
                t: all.single(b),
                owned: false,
            });
        }
        self.mod_owned(ms, &format!("single_blocks.{b}.modulation.lin.weight"), d)
    }

    /// The retained per-block GEMM route: one batch-1 linear, owned by the
    /// block.
    fn mod_owned(&mut self, ms: &ModSrc<'_>, wname: &str, d: usize) -> Result<ModVec, String> {
        let t = match ms.dv16 {
            Some(dv16) => self.mod_linear(wname, dv16, d)?,
            None => self.linear(wname, ms.dv, d, 1, "mod.gemv")?,
        };
        Ok(ModVec { t, owned: true })
    }

    /// Free a [`ModVec`] iff the block owns it.
    fn release_mod(&mut self, m: ModVec) -> Result<(), String> {
        match m.owned {
            true => self.free(m.t),
            false => Ok(()),
        }
    }

    /// Weightless LayerNorm fused with the adaLN-Zero affine, emitting
    /// `out_dt` directly — one launch in place of `layernorm` + `modulate`
    /// (+ the `cast_act` that used to follow them when the consumer was a
    /// GEMM). The f32 result is bit-identical to that chain.
    #[allow(clippy::too_many_arguments)]
    fn ln_mod(
        &mut self,
        x: &GpuTensor,
        shift: &GpuTensor,
        scale: &GpuTensor,
        n_rows: usize,
        d: usize,
        out_dt: DType,
        family: &'static str,
    ) -> Result<GpuTensor, String> {
        let out = self.alloc_dt(&[n_rows, d], out_dt)?;
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), family);
        let r = self
            .gpu
            .layernorm_modulate(x, shift, scale, &out, n_rows, d, LN_EPS)
            .map_err(|e| format!("flux gpu: layernorm_modulate: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r?;
        Ok(out)
    }

    /// Per-`(row, head)` QK-RMSNorm fused with the 2D axial RoPE — one launch
    /// in place of `rmsnorm_batched` + `rope_2d_flux_f32`, with the dtype
    /// conversion folded in at both ends.
    ///
    /// Called once per (tensor, stream): the norm `scale` is a per-stream
    /// weight (`img_attn.norm.*` vs `txt_attn.norm.*` differ inside one double
    /// block), so the joint `[n_kv, d]` buffer cannot be normalised in a
    /// single launch. Text rows take `n_img = 0` and a dummy `grid_w` of 1
    /// (the kernel rejects 0 but never reads it with no image rows).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        &mut self,
        x: &GpuTensor,
        scale: &GpuTensor,
        out: &GpuTensor,
        n_txt: usize,
        n_img: usize,
        heads: usize,
        hd: usize,
        grid_w: usize,
        ids: Option<&GpuTensor>,
    ) -> Result<(), String> {
        let axes_dim = self.cfg.axes_dim;
        let theta = self.cfg.theta;
        let t = self.prof.begin_gpu(
            &self.gpu.hip,
            self.gpu.active_stream.as_ref(),
            "norm.qk_rope",
        );
        let r = self
            .gpu
            .qk_rmsnorm_rope_flux(
                x,
                scale,
                out,
                n_txt,
                n_img,
                heads,
                hd,
                grid_w.max(1),
                axes_dim,
                theta,
                ids,
            )
            .map_err(|e| format!("flux gpu: qk_rmsnorm_rope_flux: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r
    }

    /// [`Gpuf::attention`] over operands the caller already shaped and typed:
    /// Q/out in [`Gpuf::qo_dt`], K/V in [`Gpuf::kv_dt`], destination owned by
    /// the caller. Same route selection as `attention`, minus the K/V casts
    /// (the qkv GEMMs already stored f16) and minus the output allocation.
    #[allow(clippy::too_many_arguments)]
    fn attention_into(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        out: &GpuTensor,
        n_q: usize,
        heads: usize,
        hd: usize,
    ) -> Result<(), String> {
        if hd == 128 && self.gpu.arch_caps.has_wmma_w32() {
            // `flux_attn_route_family` is cheap once cached (an `OnceLock`
            // read), but only bother resolving/caching it at all when the
            // result will actually be used.
            let t = if self.prof.enabled {
                let family = flux_attn_route_family(self.gpu);
                self.prof
                    .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), family)
            } else {
                None
            };
            let r = self
                .gpu
                .attention_flux_best_f16kv_f32(q, k, v, out, n_q, n_q, heads, heads, hd)
                .map_err(|e| format!("flux gpu: flux wmma attention: {e:?}"));
            self.prof
                .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
            r
        } else {
            let t = self.prof.begin_gpu(
                &self.gpu.hip,
                self.gpu.active_stream.as_ref(),
                "attn.dflash_f32",
            );
            let r = self
                .gpu
                .attention_dflash_f32(q, k, v, out, n_q, n_q, heads, heads, hd)
                .map_err(|e| format!("flux gpu: dflash f32 attention: {e:?}"));
            self.prof
                .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
            r
        }
    }

    fn silu(&mut self, x: &GpuTensor) -> Result<GpuTensor, String> {
        let out = self.alloc(&x.shape.clone())?;
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "elem.silu");
        self.gpu
            .silu_f32(x, &out)
            .map_err(|e| format!("flux gpu: silu: {e:?}"))?;
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        Ok(out)
    }

    /// weightless layernorm: normalize each row (mean-subtract + std), eps.
    /// The affine pair is weightless — gamma is all-ones, beta all-zeros — and
    /// therefore constant, so it is uploaded once and cached on the `Gpuf`
    /// rather than re-uploaded per call.
    fn layernorm(&mut self, x: &GpuTensor, n_rows: usize, d: usize) -> Result<GpuTensor, String> {
        if !matches!(self.ln_affine, Some((cached_d, _, _)) if cached_d == d) {
            if let Some((_, gamma, beta)) = self.ln_affine.take() {
                self.free(gamma)?;
                self.free(beta)?;
            }
            let h: Vec<f32> = vec![1.0f32; d];
            let gamma = self
                .gpu
                .upload_f32(&h, &[d])
                .map_err(|e| format!("flux gpu: layernorm gamma: {e:?}"))?;
            let beta = self.zeros(&[d])?;
            self.ln_affine = Some((d, gamma, beta));
        }
        let out = self.alloc(&[n_rows, d])?;
        // Borrowed views, so the cached buffers survive this call: the kernel
        // only reads them and `free_tensor` would reject a view anyway.
        let (gamma, beta) = {
            let (_, g, b) = self
                .ln_affine
                .as_ref()
                .expect("ln_affine populated immediately above");
            (g.sub_offset(0, d), b.sub_offset(0, d))
        };
        self.gpu
            .layernorm_batched(x, &gamma, &beta, &out, n_rows, d, 1e-6)
            .map_err(|e| format!("flux gpu: layernorm: {e:?}"))?;
        Ok(out)
    }

    /// per-head QK RMSNorm: `x` is `[n, heads*hd]`, normalize each of the
    /// `n*heads` hd-wide vectors by its own RMS then scale by the per-dim
    /// learned `scale` (same `[hd]` vector for every head, as in BFL).
    fn qk_rmsnorm(
        &mut self,
        x: &GpuTensor,
        scale: &GpuTensor,
        n: usize,
        heads: usize,
        hd: usize,
    ) -> Result<GpuTensor, String> {
        let out = self.alloc(&[n, heads * hd])?;
        self.gpu
            .rmsnorm_batched(x, scale, &out, n * heads, hd, 1e-6)
            .map_err(|e| format!("flux gpu: qk_rmsnorm: {e:?}"))?;
        Ok(out)
    }

    /// elementwise add `out = a + b`.
    fn add(&mut self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
        let out = self.alloc(&a.shape.clone())?;
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "elem.add");
        self.gpu
            .add_f32(a, b, &out)
            .map_err(|e| format!("flux gpu: add: {e:?}"))?;
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        Ok(out)
    }

    /// MLPEmbedder: `linear(in) → silu → linear(out)` for a conditioning
    /// vector (time_in / guidance_in / vector_in). `x_host` is the raw
    /// sinusoidal/pooled embedding.
    fn mlp_embedder(&mut self, prefix: &str, x_host: &[f32]) -> Result<GpuTensor, String> {
        let in_dim = x_host.len();
        let timer = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "host.io");
        let xg = self
            .gpu
            .upload_f32(x_host, &[1, in_dim])
            .map_err(|e| format!("flux gpu: upload {prefix}: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), timer);
        let xg = xg?;
        let lin = self.linear(
            &format!("{prefix}.in_layer.weight"),
            &xg,
            in_dim,
            1,
            "embed.linear",
        )?;
        self.free(xg)?;
        let mid = self.silu(&lin)?;
        self.free(lin)?;
        let out = self.linear(
            &format!("{prefix}.out_layer.weight"),
            &mid,
            self.cfg.hidden_size,
            1,
            "embed.linear",
        )?;
        self.free(mid)?;
        Ok(out)
    }

    /// Full MMDiT forward over the image/text tensors, mirroring the CPU
    /// `forward_parts` structure and returning the same named intermediates.
    #[allow(clippy::too_many_lines)]
    /// `collect` controls whether the named intermediates are downloaded.
    /// The parity harness wants them; the denoise loop wants only the final
    /// tensor, and downloading the rest costs ~1.19 GB of device→host traffic
    /// and 43 pipeline stalls per step.
    ///
    /// `txt_dev`, when supplied, is a device-resident `[n_txt, txt_dim]` f32
    /// text stream that replaces the per-call `input.txt` upload (see
    /// [`gpu_forward_txt_dev`]). It is BORROWED — never freed here.
    fn forward_parts(
        &mut self,
        cfg: &FluxDiffusionConfig,
        input: &FluxForwardInput,
        collect: bool,
        txt_dev: Option<(&GpuTensor, usize)>,
    ) -> Result<Vec<(String, Vec<f32>)>, String> {
        // FLUX.2 (Klein) is a different trunk, not a variant of this one:
        // bias-free linears, one shared modulation vector for the whole
        // model, SwiGLU MLPs, fused single-block projections and 4-axis
        // id-table RoPE. It gets its own body; everything below this line is
        // the FLUX.1 forward, unchanged.
        if cfg.is_flux2() {
            return self.forward_parts_flux2(cfg, input, collect, txt_dev);
        }
        let d = cfg.hidden_size;
        let patch_in = cfg.patch_size * cfg.patch_size * cfg.latent_channels;
        let txt_dim = cfg.txt_hidden_dim;
        let n_img = input.img.len() / patch_in;
        let n_txt = match txt_dev {
            Some((_, n)) => n,
            None => input.txt.len() / txt_dim,
        };
        let n_kv = n_img + n_txt;
        debug_assert_eq!(input.grid.0 * input.grid.1, n_img);
        let mut parts: Vec<(String, Vec<f32>)> = Vec::new();

        // Conditioning: vec († additive, all d-wide).
        let te = timestep_embedding(input.timestep, TS_EMBED_DIM, 10000.0, 1000.0);
        let mut vec = self.mlp_embedder("time_in", &te)?;
        if let Some(g) = input.guidance {
            let ge = timestep_embedding(g, TS_EMBED_DIM, 10000.0, 1000.0);
            let gv = self.mlp_embedder("guidance_in", &ge)?;
            let sum = self.add(&vec, &gv)?;
            self.free(vec)?;
            self.free(gv)?;
            vec = sum;
        }
        let c = self.mlp_embedder("vector_in", &input.pooled)?;
        let sum = self.add(&vec, &c)?;
        self.free(vec)?;
        self.free(c)?;
        vec = sum;
        if collect {
            parts.push(("vec".into(), self.download(&vec)?));
        }

        // Stream embeddings.
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "host.io");
        let img_host = self
            .gpu
            .upload_f32(&input.img, &[n_img, patch_in])
            .map_err(|e| format!("upload img: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        let img_host = img_host?;
        // The text stream is either already resident (denoise loop: uploaded
        // once per generation by the conditioning cache) or uploaded here
        // (parity harnesses that hand in a host `txt`).
        let txt_host = match txt_dev {
            Some(_) => None,
            None => {
                let t =
                    self.prof
                        .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "host.io");
                let h = self
                    .gpu
                    .upload_f32(&input.txt, &[n_txt, txt_dim])
                    .map_err(|e| format!("upload txt: {e:?}"));
                self.prof
                    .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
                Some(h?)
            }
        };
        let txt_src: &GpuTensor = match (txt_dev, &txt_host) {
            (Some((t, _)), _) => t,
            (None, Some(h)) => h,
            (None, None) => unreachable!("txt_host is set whenever txt_dev is absent"),
        };
        let mut img = self.linear("img_in.weight", &img_host, patch_in, n_img, "embed.linear")?;
        let mut txt = self.linear("txt_in.weight", txt_src, txt_dim, n_txt, "embed.linear")?;
        self.free(img_host)?;
        // Only the scratch this call allocated is freed; a borrowed `txt_dev`
        // outlives the forward.
        if let Some(h) = txt_host {
            self.free(h)?;
        }
        if collect {
            parts.push(("img_in".into(), self.download(&img)?));
            parts.push(("txt_in".into(), self.download(&txt)?));
        }

        // `silu(vec)` feeds every modulation linear in the model — 2 per double
        // block, 1 per single block, 1 in the final head — and `vec` does not
        // change across blocks, so it is computed ONCE here instead of once per
        // block (57 redundant launches per denoise step at FLUX geometry).
        // Hoisting is bit-identical: silu is a deterministic elementwise map.
        let dv = self.silu(&vec)?;
        // ...and, on the f16 path, its f16 copy: the three modulation linears
        // per block all take the SAME `[1, d]` activation, so casting it here
        // instead of inside `linear` removes 76 cast launches and 76
        // alloc/free pairs per denoise step.
        let dv16 = match self.f16_act {
            true => Some(self.cast_act(&dv, 1, d)?),
            false => None,
        };
        // ...and, on the GEMV route, every block's modulation vector: 76
        // batch-1 linears whose activation is this same `dv`, run once here
        // as GEMVs into ONE buffer instead of once per block through the
        // 128-row WMMA macro-tile (127/128 padding rows) with its own
        // alloc/free pair. `HIPFIRE_FLUX_MOD_GEMV=0` — see
        // [`mod_gemv_enabled`].
        let mod_all = match mod_gemv_enabled() {
            true => {
                Some(self.build_mod_all(cfg, &dv, 0..cfg.num_layers, 0..cfg.num_single_layers)?)
            }
            false => None,
        };
        let ms = ModSrc {
            dv: &dv,
            dv16: dv16.as_ref(),
            all: mod_all.as_ref(),
        };

        // Double blocks. Each block returns fresh streams, so the previous
        // pair dies as soon as the new one exists.
        for b in 0..cfg.num_layers {
            let (ni, nt) = self.double_block(cfg, b, &img, &txt, &ms, n_img, n_txt, input.grid)?;
            self.free(img)?;
            self.free(txt)?;
            img = ni;
            txt = nt;
            if collect {
                parts.push((format!("double_{b}_img"), self.download(&img)?));
                parts.push((format!("double_{b}_txt"), self.download(&txt)?));
            }
        }

        // Single blocks on the text-first concat. Both copies together cover
        // all n_kv rows, so the concat buffer needs no memset.
        let mut fused = self.alloc(&[n_kv, d])?;
        self.copy_into(&fused, &txt, 0, n_txt, d)?;
        self.copy_into(&fused, &img, n_txt, n_img, d)?;
        self.free(img)?;
        self.free(txt)?;
        for b in 0..cfg.num_single_layers {
            let next = self.single_block(cfg, b, &fused, &ms, n_img, input.grid, input.mlp_act)?;
            self.free(fused)?;
            fused = next;
        }
        if collect {
            parts.push(("single_concat".into(), self.download(&fused)?));
        }

        // Final head. `dv`/`dv16` are the same `silu(vec)` the blocks used.
        // Its adaLN is `[2d, d]`, not one of the `[6d, d]`/`[3d, d]` block
        // shapes, so it is not a slot of `mod_all` — it stays on the GEMM
        // route as a single batch-1 launch per forward.
        self.free(vec)?;
        let adain = match ms.dv16 {
            Some(dv16) => self.mod_linear("final_layer.adaLN_modulation.1.weight", dv16, d)?,
            None => self.linear(
                "final_layer.adaLN_modulation.1.weight",
                ms.dv,
                d,
                1,
                "mod.gemv",
            )?,
        };
        if let Some(m) = mod_all {
            self.free(m.buf)?;
        }
        if let Some(t) = dv16 {
            self.free(t)?;
        }
        self.free(dv)?;
        let img_only = fused.sub_offset(n_txt * d, n_img * d);
        let (shift, scale) = match input.final_order {
            FinalAdaLNOrder::ShiftScale => (adain.sub_offset(0, d), adain.sub_offset(d, d)),
            FinalAdaLNOrder::ScaleShift => (adain.sub_offset(d, d), adain.sub_offset(0, d)),
        };
        let out = if self.f16_act {
            let h16 = self.ln_mod(
                &img_only,
                &shift,
                &scale,
                n_img,
                d,
                DType::F16,
                "final.ln_mod",
            )?;
            self.free(adain)?;
            self.free(fused)?;
            let w = self.gw.get("final_layer.linear.weight");
            let bias = self.gw.get("final_layer.linear.bias");
            let lda = Self::wlda(w, "final_layer.linear.weight", d)?;
            let out = self.gemm_pre(&h16, w, bias, n_img, patch_in, d, true, "final.gemm", lda)?;
            self.free(h16)?;
            out
        } else {
            let normed = self.layernorm(&img_only, n_img, d)?;
            let h = self.modulate(&normed, &shift, &scale, n_img, d)?;
            self.free(normed)?;
            self.free(adain)?;
            self.free(fused)?;
            let out = self.linear("final_layer.linear.weight", &h, d, n_img, "final.gemm")?;
            self.free(h)?;
            out
        };
        parts.push(("final".into(), self.download(&out)?));
        self.free(out)?;
        Ok(parts)
    }

    // ─── FLUX.2 (Klein) forward ─────────────────────────────────────────
    //
    // Structural differences from the FLUX.1 body above, all of them visible
    // in the helpers this section adds:
    //
    // * **Bias-free.** Every linear is `y = x·Wᵀ`. `Gpuf::linear` already
    //   substitutes a zero vector for a missing `.bias`, so the whole-weight
    //   linears need nothing new; the M-sliced GEMMs pass `has_bias = false`
    //   with a one-element placeholder (`nb`) the kernel never reads.
    // * **One shared modulation vector for the WHOLE model** — three linears
    //   over `silu(temb)`, not two per double block plus one per single
    //   block. `ModAll` does not apply; the three chunks live in one `[15d]`
    //   buffer built once per forward.
    // * **SwiGLU MLPs.** `linear_in` is `[2f, d]`; the two halves are M-slices
    //   (rows `0..f` gate, `f..2f` up) so the SwiGLU needs two GEMMs and one
    //   `silu_mul_f32` — no `[n, 2f]` buffer and no gather kernel.
    // * **Fused single-block projections.** `to_qkv_mlp_proj` is
    //   `[3d + 2f, d]` (q, k, v, gate, up as five M-slices) and `to_out` is
    //   `[d, d + f]` over the per-token `[att, swiglu]` concat.
    // * **4-axis id-table RoPE.** Positions come from `input.img_ids`, not
    //   from the derived grid, so a reference image can sit at a different
    //   time id than the generated grid.
    //
    // **Activations are f32 on this path regardless of
    // [`f16_activations_enabled`]** (weights are still f16 — that is the GEMM
    // operand dtype, not an activation choice). The f16 activation layout is
    // a perf wave of its own: it needs FLUX.2 spellings of `qkv_prep_f16` /
    // `proj_mlp_f16`, a SwiGLU epilogue and a `GemmEpilogue` for the fused
    // single-block projection. Correctness first; `HIPFIRE_FLUX_F16_ACT` is
    // simply inert here for now.

    /// Full FLUX.2 (Klein) MMDiT forward, returning the same named
    /// intermediates as the CPU [`crate::flux::forward_parts`] FLUX.2 branch
    /// (`temb`, `img_in`, `txt_in`, `double_{b}_{img,txt}`, `single_concat`,
    /// `final`) so the block-parity harness compares part for part.
    #[allow(clippy::too_many_lines)]
    fn forward_parts_flux2(
        &mut self,
        cfg: &FluxDiffusionConfig,
        input: &FluxForwardInput,
        collect: bool,
        txt_dev: Option<(&GpuTensor, usize)>,
    ) -> Result<Vec<(String, Vec<f32>)>, String> {
        let d = cfg.hidden_size;
        let f = cfg.mlp_width();
        let patch_in = cfg.patch_in();
        let txt_dim = cfg.txt_hidden_dim;
        let n_img = input.img.len() / patch_in;
        let n_txt = match txt_dev {
            Some((_, n)) => n,
            None => input.txt.len() / txt_dim,
        };
        let n_all = n_txt + n_img;
        let mut parts: Vec<(String, Vec<f32>)> = Vec::new();

        // ── conditioning: temb = W2·silu(W1·sincos(t·1000)) ──────────────
        let te = timestep_embedding(input.timestep, TS_EMBED_DIM, 10000.0, 1000.0);
        let te_dev = self.upload_host(&te, &[1, TS_EMBED_DIM], "timestep embedding")?;
        let h1 = self.linear(
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            &te_dev,
            TS_EMBED_DIM,
            1,
            "embed.linear",
        )?;
        self.free(te_dev)?;
        let h1s = self.silu(&h1)?;
        self.free(h1)?;
        let temb = self.linear(
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            &h1s,
            d,
            1,
            "embed.linear",
        )?;
        self.free(h1s)?;
        if collect {
            parts.push(("temb".into(), self.download(&temb)?));
        }

        // ── the model's ONE modulation source ────────────────────────────
        // `silu(temb)` feeds all three shared linears and the final head, so
        // it is computed once and the three `[6d]`/`[6d]`/`[3d]` results are
        // GEMV'd into one `[15d]` buffer — every slot is pure-assigned below,
        // so the buffer needs no memset. There are three of these per
        // forward, not 76, so [`mod_gemv_enabled`]'s per-block GEMM A/B has
        // nothing to measure here and the kill switch is not consulted.
        let stemb = self.silu(&temb)?;
        self.free(temb)?;
        let mods = self.alloc(&[15 * d])?;
        self.mod_gemv(
            "double_stream_modulation_img.linear.weight",
            &stemb,
            &mods,
            0,
            6 * d,
            d,
        )?;
        self.mod_gemv(
            "double_stream_modulation_txt.linear.weight",
            &stemb,
            &mods,
            6 * d,
            6 * d,
            d,
        )?;
        self.mod_gemv(
            "single_stream_modulation.linear.weight",
            &stemb,
            &mods,
            12 * d,
            3 * d,
            d,
        )?;
        let mod_img = mods.sub_offset(0, 6 * d);
        let mod_txt = mods.sub_offset(6 * d, 6 * d);
        let mod_single = mods.sub_offset(12 * d, 3 * d);

        // ── stream embeddings ────────────────────────────────────────────
        let img_host = self.upload_host(&input.img, &[n_img, patch_in], "img")?;
        let txt_host = match txt_dev {
            Some(_) => None,
            None => Some(self.upload_host(&input.txt, &[n_txt, txt_dim], "txt")?),
        };
        let txt_src: &GpuTensor = match (txt_dev, &txt_host) {
            (Some((t, _)), _) => t,
            (None, Some(h)) => h,
            (None, None) => unreachable!("txt_host is set whenever txt_dev is absent"),
        };
        let mut img = self.linear(
            "x_embedder.weight",
            &img_host,
            patch_in,
            n_img,
            "embed.linear",
        )?;
        let mut txt = self.linear(
            "context_embedder.weight",
            txt_src,
            txt_dim,
            n_txt,
            "embed.linear",
        )?;
        self.free(img_host)?;
        // A borrowed `txt_dev` outlives the forward; only our own scratch goes.
        if let Some(h) = txt_host {
            self.free(h)?;
        }
        if collect {
            parts.push(("img_in".into(), self.download(&img)?));
            parts.push(("txt_in".into(), self.download(&txt)?));
        }

        // ── the RoPE id table, uploaded ONCE for the whole forward ───────
        // Explicit ids are the authority on the image token count (the edit
        // path appends reference-image tokens that are not part of `grid`);
        // absent, the FLUX.1-convention `(0, row, col, 0)` grid is derived.
        let img_ids: Vec<[f32; 4]> = input
            .img_ids
            .clone()
            .unwrap_or_else(|| rope_ids_for_grid(input.grid, 0.0));
        if img_ids.len() != n_img {
            return Err(format!(
                "flux2 gpu: img_ids has {} entries for {n_img} image tokens",
                img_ids.len()
            ));
        }
        // Klein's text rows are rotated as well — `(0, 0, 0, l)` by token
        // index (ComfyUI `txt_ids_dims = [3]`; see `flux::text_ids`) — so the
        // uploaded table covers the FULL text-first concat and the blocks
        // rope every row with `row_offset = 0`.
        let mut ids: Vec<[f32; 4]> = text_ids(n_txt);
        ids.extend(img_ids);
        let ids_flat: Vec<f32> = ids.iter().flat_map(|p| p.iter().copied()).collect();
        let ids_dev = self.upload_host(&ids_flat, &[n_all, 4], "rope_ids")?;
        // `grid_w` is only read by the kernel when `ids` is absent, but it is
        // still range-checked, so pass a legal value.
        let grid_w = input.grid.1.max(1);
        // The bias operand every `has_bias = false` GEMM needs but no kernel
        // reads. One allocation for the whole forward.
        let nb = self.zeros(&[1])?;

        // ── double blocks ────────────────────────────────────────────────
        for b in 0..cfg.num_layers {
            let (ni, nt) = self.double_block_flux2(
                cfg, b, &img, &txt, &mod_img, &mod_txt, n_img, n_txt, &ids_dev, grid_w, &nb,
            )?;
            self.free(img)?;
            self.free(txt)?;
            img = ni;
            txt = nt;
            if collect {
                parts.push((format!("double_{b}_img"), self.download(&img)?));
                parts.push((format!("double_{b}_txt"), self.download(&txt)?));
            }
        }

        // ── single blocks on the text-first concat ───────────────────────
        // Both copies together cover all `n_all` rows, so no memset.
        let mut fused = self.alloc(&[n_all, d])?;
        self.copy_into(&fused, &txt, 0, n_txt, d)?;
        self.copy_into(&fused, &img, n_txt, n_img, d)?;
        self.free(img)?;
        self.free(txt)?;
        for b in 0..cfg.num_single_layers {
            let next = self.single_block_flux2(
                cfg,
                b,
                &fused,
                &mod_single,
                n_img,
                n_txt,
                f,
                &ids_dev,
                grid_w,
                &nb,
            )?;
            self.free(fused)?;
            fused = next;
        }
        if collect {
            parts.push(("single_concat".into(), self.download(&fused)?));
        }
        self.free(ids_dev)?;
        self.free(mods)?;
        self.free(nb)?;

        // ── final head ───────────────────────────────────────────────────
        // diffusers `AdaLayerNormContinuous(bias=False)`: the 2-chunk linear
        // emits (scale, shift) — SCALE FIRST, the reverse of BFL's
        // `LastLayer`. `input.final_order` is a FLUX.1 knob and is ignored.
        let adain = self.linear("norm_out.linear.weight", &stemb, d, 1, "mod.gemv")?;
        self.free(stemb)?;
        let scale = adain.sub_offset(0, d);
        let shift = adain.sub_offset(d, d);
        let img_only = fused.sub_offset(n_txt * d, n_img * d);
        let normed = self.layernorm(&img_only, n_img, d)?;
        let h = self.modulate(&normed, &shift, &scale, n_img, d)?;
        self.free(normed)?;
        self.free(adain)?;
        self.free(fused)?;
        let out = self.linear("proj_out.weight", &h, d, n_img, "final.gemm")?;
        self.free(h)?;
        parts.push(("final".into(), self.download(&out)?));
        self.free(out)?;
        Ok(parts)
    }

    /// One FLUX.2 double (dual-stream) block: per-stream modulated LayerNorm
    /// → fused qkv → QK-RMSNorm → joint attention over the text-first concat
    /// with id-table RoPE → per-stream gated projection and SwiGLU MLP.
    #[allow(clippy::too_many_arguments)]
    fn double_block_flux2(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        img: &GpuTensor,
        txt: &GpuTensor,
        mod_img: &GpuTensor,
        mod_txt: &GpuTensor,
        n_img: usize,
        n_txt: usize,
        ids: &GpuTensor,
        grid_w: usize,
        nb: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor), String> {
        let d = cfg.hidden_size;
        let f = cfg.mlp_width();
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let n_all = n_txt + n_img;
        let p = |k: &str| format!("transformer_blocks.{b}.{k}");

        let (t_q, t_k, t_v) = self.qkv_prep_flux2(
            &p("attn.add_qkv.weight"),
            &p("attn.norm_added_q.weight"),
            &p("attn.norm_added_k.weight"),
            txt,
            mod_txt,
            n_txt,
            d,
            heads,
            hd,
            nb,
        )?;
        let (i_q, i_k, i_v) = self.qkv_prep_flux2(
            &p("attn.qkv.weight"),
            &p("attn.norm_q.weight"),
            &p("attn.norm_k.weight"),
            img,
            mod_img,
            n_img,
            d,
            heads,
            hd,
            nb,
        )?;

        // Text-first joint concat; the copies cover every row, so no memset.
        let q_all = self.alloc(&[n_all, d])?;
        let k_all = self.alloc(&[n_all, d])?;
        let v_all = self.alloc(&[n_all, d])?;
        self.copy_into(&q_all, &t_q, 0, n_txt, d)?;
        self.copy_into(&q_all, &i_q, n_txt, n_img, d)?;
        self.copy_into(&k_all, &t_k, 0, n_txt, d)?;
        self.copy_into(&k_all, &i_k, n_txt, n_img, d)?;
        self.copy_into(&v_all, &t_v, 0, n_txt, d)?;
        self.copy_into(&v_all, &i_v, n_txt, n_img, d)?;
        for t in [i_q, i_k, i_v, t_q, t_k, t_v] {
            self.free(t)?;
        }
        // Klein rotates the text rows too (`ids` covers the full concat, so
        // `row_offset = 0` and every one of the `n_all` rows is rotated).
        self.rope_ids(&q_all, 0, n_all, heads, hd, grid_w, ids, "double rope q")?;
        self.rope_ids(&k_all, 0, n_all, heads, hd, grid_w, ids, "double rope k")?;
        let att = self.attention(&q_all, &k_all, &v_all, n_all, heads, hd)?;
        for t in [q_all, k_all, v_all] {
            self.free(t)?;
        }

        let txt_out = self.stream_out_flux2(
            txt,
            n_txt,
            &att.sub_offset(0, n_txt * d),
            mod_txt,
            &p("attn.to_add_out.weight"),
            &p("ff_context.linear_in.weight"),
            &p("ff_context.linear_out.weight"),
            d,
            f,
            nb,
        )?;
        let img_out = self.stream_out_flux2(
            img,
            n_img,
            &att.sub_offset(n_txt * d, n_img * d),
            mod_img,
            &p("attn.to_out.0.weight"),
            &p("ff.linear_in.weight"),
            &p("ff.linear_out.weight"),
            d,
            f,
            nb,
        )?;
        self.free(att)?;
        Ok((img_out, txt_out))
    }

    /// FLUX.2 per-stream double-block prep: modulated LayerNorm → fused qkv
    /// (three M-slices of a `[3d, lda]` bias-free weight) → QK-RMSNorm.
    /// Returns contiguous `[n, d]` q/k/v; RoPE is applied by the caller on
    /// the joint concat.
    #[allow(clippy::too_many_arguments)]
    fn qkv_prep_flux2(
        &mut self,
        qkv_name: &str,
        qnorm: &str,
        knorm: &str,
        x: &GpuTensor,
        m: &GpuTensor,
        n: usize,
        d: usize,
        heads: usize,
        hd: usize,
        nb: &GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor), String> {
        // chunk order: shift_a, scale_a, gate_a, shift_m, scale_m, gate_m.
        let normed = self.layernorm(x, n, d)?;
        let h = self.modulate(&normed, &m.sub_offset(0, d), &m.sub_offset(d, d), n, d)?;
        self.free(normed)?;
        let w = self.gw.get(qkv_name);
        let lda = Self::wlda(w, qkv_name, d)?;
        // One f32→f16 cast of `h` feeds all three qkv GEMMs.
        let h16 = self.cast_act(&h, n, d)?;
        self.free(h)?;
        let mut qkv = Vec::with_capacity(3);
        for i in 0..3 {
            qkv.push(self.gemm_pre(
                &h16,
                &w.sub_offset(i * d * lda, d * lda),
                nb,
                n,
                d,
                d,
                false,
                "gemm.qkv",
                lda,
            )?);
        }
        self.free(h16)?;
        let v = qkv.pop().expect("three qkv slices");
        let k = qkv.pop().expect("three qkv slices");
        let q = qkv.pop().expect("three qkv slices");
        let qn = self.qk_rmsnorm(&q, self.gw.get(qnorm), n, heads, hd)?;
        self.free(q)?;
        let kn = self.qk_rmsnorm(&k, self.gw.get(knorm), n, heads, hd)?;
        self.free(k)?;
        Ok((qn, kn, v))
    }

    /// FLUX.2 per-stream double-block tail: gated attention projection, then
    /// a re-modulated SwiGLU MLP gated into the same accumulator.
    #[allow(clippy::too_many_arguments)]
    fn stream_out_flux2(
        &mut self,
        org: &GpuTensor,
        n: usize,
        att_slice: &GpuTensor,
        m: &GpuTensor,
        proj_name: &str,
        ff_in: &str,
        ff_out: &str,
        d: usize,
        f: usize,
        nb: &GpuTensor,
    ) -> Result<GpuTensor, String> {
        let proj = self.linear(proj_name, att_slice, d, n, "gemm.proj")?;
        let acc = self.copy_of(org, n, d)?;
        self.acc_gated(&acc, &m.sub_offset(2 * d, d), &proj, n, d)?;
        self.free(proj)?;
        let normed2 = self.layernorm(&acc, n, d)?;
        let h2 = self.modulate(
            &normed2,
            &m.sub_offset(3 * d, d),
            &m.sub_offset(4 * d, d),
            n,
            d,
        )?;
        self.free(normed2)?;
        let g = self.swiglu_flux2(ff_in, &h2, n, d, f, "gemm.mlp_in", nb)?;
        self.free(h2)?;
        let o = self.linear(ff_out, &g, f, n, "gemm.mlp_out")?;
        self.free(g)?;
        self.acc_gated(&acc, &m.sub_offset(5 * d, d), &o, n, d)?;
        self.free(o)?;
        Ok(acc)
    }

    /// SwiGLU over a `[2f, lda]` bias-free `linear_in` weight: two GEMMs on
    /// the M-slices (rows `0..f` = gate, `f..2f` = up) into contiguous
    /// `[n, f]` buffers, then one `silu_mul_f32`.
    ///
    /// The M-slice route rather than one `[n, 2f]` GEMM plus a gather: it is
    /// the same shape trick the FLUX.1 `qkv` and `linear1` paths already use,
    /// and `silu_mul_f32` is elementwise over `gate.numel()`, so it needs the
    /// two halves contiguous — which a single fused output is not.
    #[allow(clippy::too_many_arguments)]
    fn swiglu_flux2(
        &mut self,
        ff_in: &str,
        x: &GpuTensor,
        n: usize,
        d: usize,
        f: usize,
        family: &'static str,
        nb: &GpuTensor,
    ) -> Result<GpuTensor, String> {
        let w = self.gw.get(ff_in);
        let lda = Self::wlda(w, ff_in, d)?;
        let x16 = self.cast_act(x, n, d)?;
        let gate = self.gemm_pre(
            &x16,
            &w.sub_offset(0, f * lda),
            nb,
            n,
            f,
            d,
            false,
            family,
            lda,
        )?;
        let up = self.gemm_pre(
            &x16,
            &w.sub_offset(f * lda, f * lda),
            nb,
            n,
            f,
            d,
            false,
            family,
            lda,
        )?;
        self.free(x16)?;
        let g = self.silu_mul(&gate, &up, n, f)?;
        self.free(gate)?;
        self.free(up)?;
        Ok(g)
    }

    /// One FLUX.2 single (fused-stream) block over the text-first concat:
    /// modulated LayerNorm → the fused `to_qkv_mlp_proj` as five M-slices →
    /// QK-RMSNorm + id-table RoPE → attention → SwiGLU → the fused `to_out`
    /// over the per-token `[att, swiglu]` concat → gated residual.
    #[allow(clippy::too_many_arguments)]
    fn single_block_flux2(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        fused: &GpuTensor,
        m: &GpuTensor,
        n_img: usize,
        n_txt: usize,
        f: usize,
        ids: &GpuTensor,
        grid_w: usize,
        nb: &GpuTensor,
    ) -> Result<GpuTensor, String> {
        let d = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let n_all = n_txt + n_img;
        let p = |k: &str| format!("single_transformer_blocks.{b}.{k}");

        // chunk order: shift, scale, gate.
        let normed = self.layernorm(fused, n_all, d)?;
        let x_mod = self.modulate(&normed, &m.sub_offset(0, d), &m.sub_offset(d, d), n_all, d)?;
        self.free(normed)?;

        let proj_name = p("attn.to_qkv_mlp_proj.weight");
        let w = self.gw.get(&proj_name);
        let lda = Self::wlda(w, &proj_name, d)?;
        // One f32→f16 cast of `x_mod` feeds all five M-slice GEMMs.
        let x16 = self.cast_act(&x_mod, n_all, d)?;
        self.free(x_mod)?;
        let mut qkv = Vec::with_capacity(3);
        for i in 0..3 {
            qkv.push(self.gemm_pre(
                &x16,
                &w.sub_offset(i * d * lda, d * lda),
                nb,
                n_all,
                d,
                d,
                false,
                "gemm.single_l1_qkv",
                lda,
            )?);
        }
        let gate = self.gemm_pre(
            &x16,
            &w.sub_offset(3 * d * lda, f * lda),
            nb,
            n_all,
            f,
            d,
            false,
            "gemm.single_l1_mlp",
            lda,
        )?;
        let up = self.gemm_pre(
            &x16,
            &w.sub_offset((3 * d + f) * lda, f * lda),
            nb,
            n_all,
            f,
            d,
            false,
            "gemm.single_l1_mlp",
            lda,
        )?;
        self.free(x16)?;
        let v = qkv.pop().expect("three qkv slices");
        let k = qkv.pop().expect("three qkv slices");
        let q = qkv.pop().expect("three qkv slices");
        let qn = self.qk_rmsnorm(&q, self.gw.get(&p("attn.norm_q.weight")), n_all, heads, hd)?;
        self.free(q)?;
        let kn = self.qk_rmsnorm(&k, self.gw.get(&p("attn.norm_k.weight")), n_all, heads, hd)?;
        self.free(k)?;
        self.rope_ids(&qn, 0, n_all, heads, hd, grid_w, ids, "single rope q")?;
        self.rope_ids(&kn, 0, n_all, heads, hd, grid_w, ids, "single rope k")?;
        let att = self.attention(&qn, &kn, &v, n_all, heads, hd)?;
        for t in [qn, kn, v] {
            self.free(t)?;
        }
        let g = self.silu_mul(&gate, &up, n_all, f)?;
        self.free(gate)?;
        self.free(up)?;
        // `to_out` consumes the per-token `[att(d), swiglu(f)]` concat, so
        // the two chunks are columns of every row. Both cover all `d + f`
        // columns, so the buffer needs no memset.
        let cat = self.alloc(&[n_all, d + f])?;
        self.assemble_rows(&cat, d + f, n_all, &[(0, &att, d), (d, &g, f)])?;
        self.free(att)?;
        self.free(g)?;
        let out = self.linear(
            &p("attn.to_out.weight"),
            &cat,
            d + f,
            n_all,
            "gemm.single_l2",
        )?;
        self.free(cat)?;
        let next = self.copy_of(fused, n_all, d)?;
        self.acc_gated(&next, &m.sub_offset(2 * d, d), &out, n_all, d)?;
        self.free(out)?;
        Ok(next)
    }

    /// `out = silu(gate) * up` over two contiguous `[n, f]` buffers.
    fn silu_mul(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        n: usize,
        f: usize,
    ) -> Result<GpuTensor, String> {
        let out = self.alloc(&[n, f])?;
        let t = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "elem.silu");
        let r = self
            .gpu
            .silu_mul_f32(gate, up, &out)
            .map_err(|e| format!("flux gpu: silu_mul: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r?;
        Ok(out)
    }

    /// In-place 4-axis RoPE over the trailing `n_img` rows of a joint
    /// `[n_txt + n_img, heads·hd]` buffer, positions taken from the device
    /// `ids` table. The text rows sit first and are never rotated.
    #[allow(clippy::too_many_arguments)]
    fn rope_ids(
        &mut self,
        x: &GpuTensor,
        n_txt: usize,
        n_img: usize,
        heads: usize,
        hd: usize,
        grid_w: usize,
        ids: &GpuTensor,
        what: &str,
    ) -> Result<(), String> {
        if n_img == 0 {
            return Ok(());
        }
        let axes_dim = self.cfg.axes_dim;
        let theta = self.cfg.theta;
        let t = self.prof.begin_gpu(
            &self.gpu.hip,
            self.gpu.active_stream.as_ref(),
            "norm.qk_rope",
        );
        let r = self
            .gpu
            .rope_2d_flux_f32(
                x,
                n_txt,
                n_img,
                heads,
                hd,
                grid_w,
                axes_dim,
                theta,
                Some(ids),
            )
            .map_err(|e| format!("flux2 gpu: {what}: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), t);
        r
    }

    /// Upload a host buffer, timed as host I/O like every other upload in
    /// the forward.
    fn upload_host(
        &mut self,
        data: &[f32],
        shape: &[usize],
        what: &str,
    ) -> Result<GpuTensor, String> {
        let timer = self
            .prof
            .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "host.io");
        let r = self
            .gpu
            .upload_f32(data, shape)
            .map_err(|e| format!("flux2 gpu: upload {what}: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), timer);
        r
    }

    fn modulate(
        &mut self,
        x: &GpuTensor,
        shift: &GpuTensor,
        scale: &GpuTensor,
        n_rows: usize,
        d: usize,
    ) -> Result<GpuTensor, String> {
        let out = self.alloc(&[n_rows, d])?;
        self.gpu
            .modulate_f32(x, shift, scale, &out, n_rows, d)
            .map_err(|e| format!("flux gpu: modulate: {e:?}"))?;
        Ok(out)
    }

    /// Copy `src` (a contiguous `[n_rows, d]` buffer) into `dst` starting at
    /// row `dst_row` (both `[.., d]` row-major F32).
    fn copy_into(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        dst_row: usize,
        n_rows: usize,
        d: usize,
    ) -> Result<(), String> {
        let view = dst.sub_offset(dst_row * d, n_rows * d);
        let timer =
            self.prof
                .begin_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), "elem.copy");
        let r = self
            .gpu
            .copy_d2d(src, &view, n_rows * d * DType::F32.size())
            .map_err(|e| format!("copy_into: {e:?}"));
        self.prof
            .end_gpu(&self.gpu.hip, self.gpu.active_stream.as_ref(), timer);
        r
    }

    /// A fresh `[n_rows, d]` copy (for residual bases that we then gate).
    fn copy_of(&mut self, src: &GpuTensor, n_rows: usize, d: usize) -> Result<GpuTensor, String> {
        let out = self.alloc(&[n_rows, d])?;
        self.copy_into(&out, src, 0, n_rows, d)?;
        Ok(out)
    }

    /// `acc += gate[i] * x` broadcast gated residual.
    fn acc_gated(
        &mut self,
        acc: &GpuTensor,
        gate: &GpuTensor,
        x: &GpuTensor,
        n_rows: usize,
        d: usize,
    ) -> Result<(), String> {
        self.gpu
            .gated_add_f32(acc, gate, x, n_rows, d)
            .map_err(|e| format!("flux gpu: gated_add: {e:?}"))
    }

    fn gelu(&mut self, x: &GpuTensor, total: usize) -> Result<GpuTensor, String> {
        let out = self.alloc(&[total])?;
        self.gpu
            .gelu_tanh_f32(x, &out, total)
            .map_err(|e| format!("flux gpu: gelu: {e:?}"))?;
        Ok(out)
    }

    /// Dense multi-head attention (n_q == n_kv for FLUX square blocks).
    /// Routes through the tuned bidirectional DFlash family instead of the
    /// naive thread-per-element `attention_dense_f32`: the wave32-WMMA
    /// `attention_dflash_wmma_m64_n128_f16kv_v4_f32` kernel (q/out f32, K/V
    /// cast to f16, hd==128, non-causal, N-tile=128 for 4× KV reuse) on
    /// gfx11/gfx12, with the portable tiled `attention_dflash_f32` fallback
    /// otherwise. FLUX is MHA so n_kv_heads == heads; the DFlash scale is
    /// 1/√hd (computed inside the kernel), matching the prior
    /// `attention_dense_f32` scale.
    fn attention(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        n_q: usize,
        heads: usize,
        hd: usize,
    ) -> Result<GpuTensor, String> {
        let out = self.alloc(&[n_q, heads * hd])?;
        if hd == 128 && self.gpu.arch_caps.has_wmma_w32() {
            // Both scratches are fully written by the casts below.
            let k_f16 = self
                .gpu
                .alloc_tensor(&[n_q, heads * hd], DType::F16)
                .map_err(|e| format!("flux gpu: alloc k f16: {e:?}"))?;
            let v_f16 = self
                .gpu
                .alloc_tensor(&[n_q, heads * hd], DType::F16)
                .map_err(|e| format!("flux gpu: alloc v f16: {e:?}"))?;
            self.gpu
                .cast_f32_to_f16(k, &k_f16)
                .map_err(|e| format!("flux gpu: cast k f16: {e:?}"))?;
            self.gpu
                .cast_f32_to_f16(v, &v_f16)
                .map_err(|e| format!("flux gpu: cast v f16: {e:?}"))?;
            // `attention_flux_best_f16kv_f32` picks per arch from measurement
            // rather than from a capability predicate — the vt/vtk ranking
            // inverts between gfx1150 and gfx1151/gfx1100. Both stage V
            // transposed so the PV fragment load is contiguous, and keep the
            // online softmax in registers; measured 3.0–4.0× over the v5
            // kernel this replaces. `HIPFIRE_FLUX_ATTN=v5` restores the old
            // route for a same-session A/B.
            self.gpu
                .attention_flux_best_f16kv_f32(q, &k_f16, &v_f16, &out, n_q, n_q, heads, heads, hd)
                .map_err(|e| format!("flux gpu: flux wmma attention: {e:?}"))?;
            self.gpu
                .free_tensor(k_f16)
                .map_err(|e| format!("flux gpu: free k f16: {e:?}"))?;
            self.gpu
                .free_tensor(v_f16)
                .map_err(|e| format!("flux gpu: free v f16: {e:?}"))?;
        } else {
            self.gpu
                .attention_dflash_f32(q, k, v, &out, n_q, n_q, heads, heads, hd)
                .map_err(|e| format!("flux gpu: dflash f32 attention: {e:?}"))?;
        }
        Ok(out)
    }

    /// Per-stream double-block prep: modulation → layernorm → adaLN → qkv →
    /// QK-RMSNorm. Returns `(q, k, v, modu)`, all contiguous `[n, d]`(q/k/v).
    fn qkv_prep(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        stem: &str,
        x: &GpuTensor,
        ms: &ModSrc<'_>,
        heads: usize,
        hd: usize,
        n: usize,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor, ModVec), String> {
        let d = cfg.hidden_size;
        let p = |s: &str| format!("double_blocks.{b}.{stem}_{s}");
        let modu = self.mod_double(ms, b, stem, d)?;
        let normed = self.layernorm(x, n, d)?;
        let h = self.modulate(
            &normed,
            &modu.t.sub_offset(0, d),
            &modu.t.sub_offset(d, d),
            n,
            d,
        )?;
        self.free(normed)?;
        let qkv_name = p("attn.qkv.weight");
        let qkv_w = self.gw.get(&qkv_name);
        let qkv_b = self.gw.get(&p("attn.qkv.bias"));
        // `[3d, lda]` sliced along M into three `[d, lda]` weights: row `m`
        // starts at `m·lda`, so both the offset and the length scale with the
        // stored pitch, not with the logical K.
        let lda = Self::wlda(qkv_w, &qkv_name, d)?;
        // One f32→f16 cast of `h` feeds all three qkv GEMMs.
        let h16 = self.cast_act(&h, n, d)?;
        self.free(h)?;
        let q = self.gemm_pre(
            &h16,
            &qkv_w.sub_offset(0, d * lda),
            &qkv_b.sub_offset(0, d),
            n,
            d,
            d,
            true,
            "gemm.qkv",
            lda,
        )?;
        let k = self.gemm_pre(
            &h16,
            &qkv_w.sub_offset(d * lda, d * lda),
            &qkv_b.sub_offset(d, d),
            n,
            d,
            d,
            true,
            "gemm.qkv",
            lda,
        )?;
        let v = self.gemm_pre(
            &h16,
            &qkv_w.sub_offset(2 * d * lda, d * lda),
            &qkv_b.sub_offset(2 * d, d),
            n,
            d,
            d,
            true,
            "gemm.qkv",
            lda,
        )?;
        self.free(h16)?;
        // QK-RMSNorm is out-of-place, so the pre-norm q/k die here.
        let qn = self.qk_rmsnorm(
            &q,
            self.gw.get(&p("attn.norm.query_norm.scale")),
            n,
            heads,
            hd,
        )?;
        self.free(q)?;
        let kn = self.qk_rmsnorm(
            &k,
            self.gw.get(&p("attn.norm.key_norm.scale")),
            n,
            heads,
            hd,
        )?;
        self.free(k)?;
        Ok((qn, kn, v, modu))
    }

    /// Per-stream gated residual + re-modulated MLP after joint attention.
    fn proj_mlp(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        stem: &str,
        att_slice: &GpuTensor,
        org: &GpuTensor,
        modu: &GpuTensor,
        n: usize,
    ) -> Result<GpuTensor, String> {
        let d = cfg.hidden_size;
        let p = |s: &str| format!("double_blocks.{b}.{stem}_{s}");
        let proj = self.linear(&p("attn.proj.weight"), att_slice, d, n, "gemm.proj")?;
        let acc = self.copy_of(org, n, d)?;
        self.acc_gated(&acc, &modu.sub_offset(2 * d, d), &proj, n, d)?;
        self.free(proj)?;
        let normed2 = self.layernorm(&acc, n, d)?;
        let h2 = self.modulate(
            &normed2,
            &modu.sub_offset(3 * d, d),
            &modu.sub_offset(4 * d, d),
            n,
            d,
        )?;
        self.free(normed2)?;
        let m0 = self.linear(&p("mlp.0.weight"), &h2, d, n, "gemm.mlp_in")?;
        self.free(h2)?;
        let mg = self.gelu(&m0, n * 4 * d)?;
        self.free(m0)?;
        let m1 = self.linear(&p("mlp.2.weight"), &mg, 4 * d, n, "gemm.mlp_out")?;
        self.free(mg)?;
        self.acc_gated(&acc, &modu.sub_offset(5 * d, d), &m1, n, d)?;
        self.free(m1)?;
        Ok(acc)
    }

    /// Dispatch one double block to the f16-activation body or the retained
    /// all-f32 body (`HIPFIRE_FLUX_F16_ACT=0`). `dv` is `silu(vec)`, hoisted
    /// out of the block loop by the caller.
    #[allow(clippy::too_many_arguments)]
    fn double_block(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        img: &GpuTensor,
        txt: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        n_txt: usize,
        grid: (usize, usize),
    ) -> Result<(GpuTensor, GpuTensor), String> {
        match ms.dv16 {
            Some(_) => self.double_block_f16(cfg, b, img, txt, ms, n_img, n_txt, grid),
            None => self.double_block_f32(cfg, b, img, txt, ms, n_img, n_txt, grid),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn double_block_f32(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        img: &GpuTensor,
        txt: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        n_txt: usize,
        grid: (usize, usize),
    ) -> Result<(GpuTensor, GpuTensor), String> {
        let d = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let n_kv = n_img + n_txt;
        let (i_q, i_k, i_v, i_mod) = self.qkv_prep(cfg, b, "img", img, ms, heads, hd, n_img)?;
        let (t_q, t_k, t_v, t_mod) = self.qkv_prep(cfg, b, "txt", txt, ms, heads, hd, n_txt)?;

        // Joint attention over the text-first concat; image rows get 2D RoPE.
        // The three concat buffers are written in full by the copies below
        // (n_txt + n_img == n_kv rows each), so they need no memset.
        let q_all = self.alloc(&[n_kv, d])?;
        let k_all = self.alloc(&[n_kv, d])?;
        let v_all = self.alloc(&[n_kv, d])?;
        self.copy_into(&q_all, &t_q, 0, n_txt, d)?;
        self.copy_into(&q_all, &i_q, n_txt, n_img, d)?;
        self.copy_into(&k_all, &t_k, 0, n_txt, d)?;
        self.copy_into(&k_all, &i_k, n_txt, n_img, d)?;
        self.copy_into(&v_all, &t_v, 0, n_txt, d)?;
        self.copy_into(&v_all, &i_v, n_txt, n_img, d)?;
        for t in [i_q, i_k, i_v, t_q, t_k, t_v] {
            self.free(t)?;
        }
        let axes_dim = cfg.axes_dim;
        self.gpu
            .rope_2d_flux_f32(
                &q_all, n_txt, n_img, heads, hd, grid.1, axes_dim, cfg.theta, None,
            )
            .map_err(|e| format!("double rope q: {e:?}"))?;
        self.gpu
            .rope_2d_flux_f32(
                &k_all, n_txt, n_img, heads, hd, grid.1, axes_dim, cfg.theta, None,
            )
            .map_err(|e| format!("double rope k: {e:?}"))?;
        let att = self.attention(&q_all, &k_all, &v_all, n_kv, heads, hd)?;
        for t in [q_all, k_all, v_all] {
            self.free(t)?;
        }

        let img_out = self.proj_mlp(
            cfg,
            b,
            "img",
            &att.sub_offset(n_txt * d, n_img * d),
            img,
            &i_mod.t,
            n_img,
        )?;
        let txt_out = self.proj_mlp(
            cfg,
            b,
            "txt",
            &att.sub_offset(0, n_txt * d),
            txt,
            &t_mod.t,
            n_txt,
        )?;
        self.free(att)?;
        self.release_mod(i_mod)?;
        self.release_mod(t_mod)?;
        Ok((img_out, txt_out))
    }

    /// Dispatch one single block. `dv` is `silu(vec)`, hoisted by the caller.
    #[allow(clippy::too_many_arguments)]
    fn single_block(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        fused: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        grid: (usize, usize),
        act: MlpAct,
    ) -> Result<GpuTensor, String> {
        match ms.dv16 {
            Some(_) => self.single_block_f16(cfg, b, fused, ms, n_img, grid, act),
            None => self.single_block_f32(cfg, b, fused, ms, n_img, grid, act),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn single_block_f32(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        fused: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        grid: (usize, usize),
        act: MlpAct,
    ) -> Result<GpuTensor, String> {
        let d = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let f = 4 * d;
        let n_all = fused.numel() / d;
        let modu = self.mod_single(ms, b, d)?;
        let s1 = modu.t.sub_offset(0, d);
        let c1 = modu.t.sub_offset(d, d);
        let g1 = modu.t.sub_offset(2 * d, d);
        let normed = self.layernorm(fused, n_all, d)?;
        let x_mod = self.modulate(&normed, &s1, &c1, n_all, d)?;
        self.free(normed)?;

        // linear1 fused weight split by rows → contiguous q/k/v + mlp.
        let l1_name = format!("single_blocks.{b}.linear1.weight");
        let l1_w = self.gw.get(&l1_name);
        let l1_b = self.gw.get(&format!("single_blocks.{b}.linear1.bias"));
        // Row slices of a `[7d, lda]` weight — see `qkv_prep` on why the M
        // offsets scale with the stored pitch.
        let lda = Self::wlda(l1_w, &l1_name, d)?;
        // One f32→f16 cast of `x_mod` feeds all four linear1 GEMMs.
        let x16 = self.cast_act(&x_mod, n_all, d)?;
        self.free(x_mod)?;
        let q = self.gemm_pre(
            &x16,
            &l1_w.sub_offset(0, d * lda),
            &l1_b.sub_offset(0, d),
            n_all,
            d,
            d,
            true,
            "gemm.single_l1_qkv",
            lda,
        )?;
        let k = self.gemm_pre(
            &x16,
            &l1_w.sub_offset(d * lda, d * lda),
            &l1_b.sub_offset(d, d),
            n_all,
            d,
            d,
            true,
            "gemm.single_l1_qkv",
            lda,
        )?;
        let v = self.gemm_pre(
            &x16,
            &l1_w.sub_offset(2 * d * lda, d * lda),
            &l1_b.sub_offset(2 * d, d),
            n_all,
            d,
            d,
            true,
            "gemm.single_l1_qkv",
            lda,
        )?;
        let mlp = self.gemm_pre(
            &x16,
            &l1_w.sub_offset(3 * d * lda, 4 * d * lda),
            &l1_b.sub_offset(3 * d, 4 * d),
            n_all,
            f,
            d,
            true,
            "gemm.single_l1_mlp",
            lda,
        )?;
        self.free(x16)?;
        let qn = self.qk_rmsnorm(
            &q,
            self.gw
                .get(&format!("single_blocks.{b}.norm.query_norm.scale")),
            n_all,
            heads,
            hd,
        )?;
        self.free(q)?;
        let kn = self.qk_rmsnorm(
            &k,
            self.gw
                .get(&format!("single_blocks.{b}.norm.key_norm.scale")),
            n_all,
            heads,
            hd,
        )?;
        self.free(k)?;
        let (q, k) = (qn, kn);
        // RoPE the trailing image rows before attention.
        let axes_dim = cfg.axes_dim;
        self.gpu
            .rope_2d_flux_f32(
                &q,
                n_all - n_img,
                n_img,
                heads,
                hd,
                grid.1,
                axes_dim,
                cfg.theta,
                None,
            )
            .map_err(|e| format!("single rope q: {e:?}"))?;
        self.gpu
            .rope_2d_flux_f32(
                &k,
                n_all - n_img,
                n_img,
                heads,
                hd,
                grid.1,
                axes_dim,
                cfg.theta,
                None,
            )
            .map_err(|e| format!("single rope k: {e:?}"))?;
        let att = self.attention(&q, &k, &v, n_all, heads, hd)?;
        for t in [q, k, v] {
            self.free(t)?;
        }
        let mlp_g = match act {
            MlpAct::GeluTanh => self.gelu(&mlp, n_all * f)?,
            MlpAct::Silu => self.silu(&mlp)?,
        };
        self.free(mlp)?;
        // linear2 input cat [att(d), mlp_g(4d)] per token (fused K interleave).
        // The two chunks cover all 5d columns of every row, so no memset.
        let fused2 = self.alloc(&[n_all, 5 * d])?;
        self.assemble_rows(&fused2, 5 * d, n_all, &[(0, &att, d), (d, &mlp_g, f)])?;
        self.free(att)?;
        self.free(mlp_g)?;
        let out = self.linear(
            &format!("single_blocks.{b}.linear2.weight"),
            &fused2,
            5 * d,
            n_all,
            "gemm.single_l2",
        )?;
        self.free(fused2)?;
        let next = self.copy_of(fused, n_all, d)?;
        self.acc_gated(&next, &g1, &out, n_all, d)?;
        self.free(out)?;
        // `g1` is a view into `modu`, so `modu` must outlive the gated add.
        self.release_mod(modu)?;
        Ok(next)
    }

    // ─── f16-activation block bodies ────────────────────────────────────

    /// Per-stream double-block prep, writing q/k/v **directly into the row
    /// range this stream owns in the text-first joint concat**.
    ///
    /// This is the structural difference from [`Gpuf::qkv_prep`]: there, each
    /// stream's q/k/v were three standalone `[n, d]` buffers that
    /// `double_block` then copied into the joint buffers — six `copy_d2d`
    /// calls per block, all host-synchronous because no stream was active.
    /// Here the GEMM's destination IS the sub-range, so the copies do not
    /// exist. `row0` is the stream's first row in the concat (0 for text,
    /// `n_txt` for image).
    ///
    /// `rope` carries the image grid width for the image stream and `None`
    /// for the text stream: the joint RoPE rotates only the image rows, and
    /// the QK-norm scale differs per stream (`img_attn.norm.*` vs
    /// `txt_attn.norm.*`), so the fused norm+RoPE kernel runs once per stream
    /// over that stream's rows rather than once over the joint buffer.
    #[allow(clippy::too_many_arguments)]
    fn qkv_prep_f16(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        stem: &str,
        x: &GpuTensor,
        modu: &GpuTensor,
        n: usize,
        row0: usize,
        bufs: [&GpuTensor; 5],
        heads: usize,
        hd: usize,
        rope: Option<usize>,
    ) -> Result<(), String> {
        let d = cfg.hidden_size;
        let [q_pre, k_pre, v_all, q_n, k_n] = bufs;
        let p = |s: &str| format!("double_blocks.{b}.{stem}_{s}");
        let h16 = self.ln_mod(
            x,
            &modu.sub_offset(0, d),
            &modu.sub_offset(d, d),
            n,
            d,
            DType::F16,
            "norm.ln_mod",
        )?;
        let qkv_name = p("attn.qkv.weight");
        let qkv_w = self.gw.get(&qkv_name);
        let qkv_b = self.gw.get(&p("attn.qkv.bias"));
        // Row slices of `[3d, lda]` — see the f32 `qkv_prep`.
        let lda = Self::wlda(qkv_w, &qkv_name, d)?;
        for (i, dst) in [q_pre, k_pre, v_all].into_iter().enumerate() {
            let y = dst.sub_offset(row0 * d, n * d);
            let epi = GemmEpilogue {
                out_f16: dst.dtype == DType::F16,
                ..Default::default()
            };
            self.gemm_epi(
                &qkv_w.sub_offset(i * d * lda, d * lda),
                &h16,
                &y,
                Some(&qkv_b.sub_offset(i * d, d)),
                n,
                d,
                d,
                &epi,
                "gemm.qkv",
                lda,
            )?;
        }
        self.free(h16)?;
        // Text rows: norm only. Image rows: norm + rotation, with the image
        // token index counted from this call's first row, which is exactly
        // what `rope_2d_flux_f32(row_offset = n_txt)` did over the joint
        // buffer.
        let (nt, ni, grid_w) = match rope {
            Some(g) => (0, n, g),
            None => (n, 0, 1),
        };
        for (dst, src, key) in [
            (q_n, q_pre, "attn.norm.query_norm.scale"),
            (k_n, k_pre, "attn.norm.key_norm.scale"),
        ] {
            self.qk_norm_rope(
                &src.sub_offset(row0 * d, n * d),
                self.gw.get(&p(key)),
                &dst.sub_offset(row0 * d, n * d),
                nt,
                ni,
                heads,
                hd,
                grid_w,
                None,
            )?;
        }
        Ok(())
    }

    /// Per-stream gated residual + re-modulated MLP, with every elementwise
    /// pass folded into a GEMM epilogue: three launches replace the
    /// cast/GEMM/copy/gated-add/layernorm/modulate/cast/GEMM/GELU/cast/GEMM/
    /// gated-add chain of [`Gpuf::proj_mlp`].
    ///
    /// The `proj` GEMM writes `acc = org + gate·(proj + bias)` with `org` as
    /// the residual operand and a fresh `acc` as the destination, so the
    /// caller's input stream is never mutated (the public `gpu_double_block`
    /// hands in a borrowed tensor). The `mlp.2` GEMM then gates in place, with
    /// `acc` as both residual and destination.
    #[allow(clippy::too_many_arguments)]
    fn proj_mlp_f16(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        stem: &str,
        att_slice: &GpuTensor,
        org: &GpuTensor,
        modu: &GpuTensor,
        n: usize,
    ) -> Result<GpuTensor, String> {
        let d = cfg.hidden_size;
        let p = |s: &str| format!("double_blocks.{b}.{stem}_{s}");
        // The tuned attention already stored f16; only the portable f32
        // fallback route needs a cast to feed the proj GEMM.
        let cast = if self.qo_dt == DType::F16 {
            None
        } else {
            Some(self.cast_act(att_slice, n, d)?)
        };
        let acc = self.alloc(&[n, d])?;
        {
            let att16 = cast.as_ref().unwrap_or(att_slice);
            let gate = modu.sub_offset(2 * d, d);
            let epi = GemmEpilogue {
                gate: Some(&gate),
                residual: Some(org),
                ..Default::default()
            };
            let wname = p("attn.proj.weight");
            let w = self.gw.get(&wname);
            let bias = self.gw.get(&p("attn.proj.bias"));
            let lda = Self::wlda(w, &wname, d)?;
            self.gemm_epi(w, att16, &acc, Some(bias), n, d, d, &epi, "gemm.proj", lda)?;
        }
        if let Some(c) = cast {
            self.free(c)?;
        }
        let h2 = self.ln_mod(
            &acc,
            &modu.sub_offset(3 * d, d),
            &modu.sub_offset(4 * d, d),
            n,
            d,
            DType::F16,
            "norm.ln_mod",
        )?;
        let mg = self.alloc_dt(&[n, 4 * d], DType::F16)?;
        {
            let epi = GemmEpilogue {
                out_f16: true,
                gelu: true,
                ..Default::default()
            };
            let wname = p("mlp.0.weight");
            let w = self.gw.get(&wname);
            let bias = self.gw.get(&p("mlp.0.bias"));
            let lda = Self::wlda(w, &wname, d)?;
            self.gemm_epi(
                w,
                &h2,
                &mg,
                Some(bias),
                n,
                4 * d,
                d,
                &epi,
                "gemm.mlp_in",
                lda,
            )?;
        }
        self.free(h2)?;
        {
            let gate = modu.sub_offset(5 * d, d);
            let epi = GemmEpilogue {
                gate: Some(&gate),
                residual: Some(&acc),
                ..Default::default()
            };
            let wname = p("mlp.2.weight");
            let w = self.gw.get(&wname);
            let bias = self.gw.get(&p("mlp.2.bias"));
            let lda = Self::wlda(w, &wname, 4 * d)?;
            self.gemm_epi(
                w,
                &mg,
                &acc,
                Some(bias),
                n,
                d,
                4 * d,
                &epi,
                "gemm.mlp_out",
                lda,
            )?;
        }
        self.free(mg)?;
        Ok(acc)
    }

    #[allow(clippy::too_many_arguments)]
    fn double_block_f16(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        img: &GpuTensor,
        txt: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        n_txt: usize,
        grid: (usize, usize),
    ) -> Result<(GpuTensor, GpuTensor), String> {
        let d = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let n_kv = n_img + n_txt;
        let (qo_dt, kv_dt) = (self.qo_dt, self.kv_dt);
        let i_mod = self.mod_double(ms, b, "img", d)?;
        let t_mod = self.mod_double(ms, b, "txt", d)?;

        // Joint text-first buffers. `*_pre` take the qkv GEMM stores; the
        // fused norm+RoPE is out-of-place (the kernel marks both operands
        // `__restrict__`), so Q/K get a second pair in the dtype the attention
        // route wants. All rows of every buffer are written, so none is zeroed.
        let q_pre = self.alloc_dt(&[n_kv, d], DType::F16)?;
        let k_pre = self.alloc_dt(&[n_kv, d], DType::F16)?;
        let v_all = self.alloc_dt(&[n_kv, d], kv_dt)?;
        let q_all = self.alloc_dt(&[n_kv, d], qo_dt)?;
        let k_all = self.alloc_dt(&[n_kv, d], kv_dt)?;
        let bufs = [&q_pre, &k_pre, &v_all, &q_all, &k_all];
        self.qkv_prep_f16(
            cfg, b, "txt", txt, &t_mod.t, n_txt, 0, bufs, heads, hd, None,
        )?;
        self.qkv_prep_f16(
            cfg,
            b,
            "img",
            img,
            &i_mod.t,
            n_img,
            n_txt,
            bufs,
            heads,
            hd,
            Some(grid.1),
        )?;
        self.free(q_pre)?;
        self.free(k_pre)?;

        let att = self.alloc_dt(&[n_kv, d], qo_dt)?;
        self.attention_into(&q_all, &k_all, &v_all, &att, n_kv, heads, hd)?;
        for t in [q_all, k_all, v_all] {
            self.free(t)?;
        }

        let img_out = self.proj_mlp_f16(
            cfg,
            b,
            "img",
            &att.sub_offset(n_txt * d, n_img * d),
            img,
            &i_mod.t,
            n_img,
        )?;
        let txt_out = self.proj_mlp_f16(
            cfg,
            b,
            "txt",
            &att.sub_offset(0, n_txt * d),
            txt,
            &t_mod.t,
            n_txt,
        )?;
        self.free(att)?;
        self.release_mod(i_mod)?;
        self.release_mod(t_mod)?;
        Ok((img_out, txt_out))
    }

    #[allow(clippy::too_many_arguments)]
    fn single_block_f16(
        &mut self,
        cfg: &FluxDiffusionConfig,
        b: usize,
        fused: &GpuTensor,
        ms: &ModSrc<'_>,
        n_img: usize,
        grid: (usize, usize),
        act: MlpAct,
    ) -> Result<GpuTensor, String> {
        let d = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let f = 4 * d;
        let n_all = fused.numel() / d;
        let n_txt = n_all - n_img;
        let (qo_dt, kv_dt) = (self.qo_dt, self.kv_dt);
        let sb = |s: &str| format!("single_blocks.{b}.{s}");
        let modu = self.mod_single(ms, b, d)?;
        let x16 = self.ln_mod(
            fused,
            &modu.t.sub_offset(0, d),
            &modu.t.sub_offset(d, d),
            n_all,
            d,
            DType::F16,
            "norm.ln_mod",
        )?;

        // linear1 split by output rows into q/k/v + mlp, as before, but each
        // GEMM now stores the dtype its consumer wants.
        let q_pre = self.alloc_dt(&[n_all, d], DType::F16)?;
        let k_pre = self.alloc_dt(&[n_all, d], DType::F16)?;
        let v = self.alloc_dt(&[n_all, d], kv_dt)?;
        let l1_name = sb("linear1.weight");
        let l1_w = self.gw.get(&l1_name);
        let l1_b = self.gw.get(&sb("linear1.bias"));
        // Row slices of `[7d, l1_lda]` — see the f32 `qkv_prep`.
        let l1_lda = Self::wlda(l1_w, &l1_name, d)?;
        for (i, dst) in [&q_pre, &k_pre, &v].into_iter().enumerate() {
            let epi = GemmEpilogue {
                out_f16: dst.dtype == DType::F16,
                ..Default::default()
            };
            self.gemm_epi(
                &l1_w.sub_offset(i * d * l1_lda, d * l1_lda),
                &x16,
                dst,
                Some(&l1_b.sub_offset(i * d, d)),
                n_all,
                d,
                d,
                &epi,
                "gemm.single_l1_qkv",
                l1_lda,
            )?;
        }
        // GELU-tanh is the only activation with a fused-epilogue entry; SiLU
        // (schnell-style configs) keeps the separate launches and the cast.
        let mlp16 = match act {
            MlpAct::GeluTanh => {
                let g = self.alloc_dt(&[n_all, f], DType::F16)?;
                let epi = GemmEpilogue {
                    out_f16: true,
                    gelu: true,
                    ..Default::default()
                };
                self.gemm_epi(
                    &l1_w.sub_offset(3 * d * l1_lda, 4 * d * l1_lda),
                    &x16,
                    &g,
                    Some(&l1_b.sub_offset(3 * d, f)),
                    n_all,
                    f,
                    d,
                    &epi,
                    "gemm.single_l1_mlp",
                    l1_lda,
                )?;
                g
            }
            MlpAct::Silu => {
                let raw = self.gemm_pre(
                    &x16,
                    &l1_w.sub_offset(3 * d * l1_lda, 4 * d * l1_lda),
                    &l1_b.sub_offset(3 * d, f),
                    n_all,
                    f,
                    d,
                    true,
                    "gemm.single_l1_mlp",
                    l1_lda,
                )?;
                let s = self.silu(&raw)?;
                self.free(raw)?;
                let g = self.cast_act(&s, n_all, f)?;
                self.free(s)?;
                g
            }
        };
        self.free(x16)?;

        let q = self.alloc_dt(&[n_all, d], qo_dt)?;
        let k = self.alloc_dt(&[n_all, d], kv_dt)?;
        self.qk_norm_rope(
            &q_pre,
            self.gw.get(&sb("norm.query_norm.scale")),
            &q,
            n_txt,
            n_img,
            heads,
            hd,
            grid.1,
            None,
        )?;
        self.qk_norm_rope(
            &k_pre,
            self.gw.get(&sb("norm.key_norm.scale")),
            &k,
            n_txt,
            n_img,
            heads,
            hd,
            grid.1,
            None,
        )?;
        self.free(q_pre)?;
        self.free(k_pre)?;

        let att = self.alloc_dt(&[n_all, d], qo_dt)?;
        self.attention_into(&q, &k, &v, &att, n_all, heads, hd)?;
        for t in [q, k, v] {
            self.free(t)?;
        }

        // `linear2` was split along K at upload time, so the `[n_all, 5d]`
        // per-token concat of `att` and `mlp_g` is gone: run the attention
        // half into a scratch, then fold it into the MLP half's epilogue as
        // the ADDIN operand together with the bias and the gated residual.
        let cast = if qo_dt == DType::F16 {
            None
        } else {
            Some(self.cast_act(&att, n_all, d)?)
        };
        let tmp = self.alloc(&[n_all, d])?;
        {
            let att16 = cast.as_ref().unwrap_or(&att);
            let wname = sb("linear2.w_attn.weight");
            let w_attn = self.gw.get(&wname);
            let lda = Self::wlda(w_attn, &wname, d)?;
            self.gemm_epi(
                w_attn,
                att16,
                &tmp,
                None,
                n_all,
                d,
                d,
                &GemmEpilogue::default(),
                "gemm.single_l2",
                lda,
            )?;
        }
        if let Some(c) = cast {
            self.free(c)?;
        }
        self.free(att)?;

        let next = self.alloc(&[n_all, d])?;
        {
            let gate = modu.t.sub_offset(2 * d, d);
            let epi = GemmEpilogue {
                addin: Some(&tmp),
                gate: Some(&gate),
                residual: Some(fused),
                ..Default::default()
            };
            let wname = sb("linear2.w_mlp.weight");
            let w_mlp = self.gw.get(&wname);
            let bias = self.gw.get(&sb("linear2.bias"));
            let lda = Self::wlda(w_mlp, &wname, f)?;
            self.gemm_epi(
                w_mlp,
                &mlp16,
                &next,
                Some(bias),
                n_all,
                d,
                f,
                &epi,
                "gemm.single_l2",
                lda,
            )?;
        }
        self.free(tmp)?;
        self.free(mlp16)?;
        self.release_mod(modu)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FluxDiffusionConfig;
    use serde_json::json;

    fn tiny_cfg() -> FluxDiffusionConfig {
        FluxDiffusionConfig::from_json(&json!({
            "hidden_size": 16,
            "num_attention_heads": 2,
            "head_dim": 8,
            "axes_dim": [2, 3, 3],
            "num_layers": 2,
            "num_single_layers": 1,
            "pooled_projection_dim": 8,
            "latent_channels": 4,
            "patch_size": 2,
        }))
        .unwrap()
    }

    /// The `mod_all` slots must tile the buffer exactly: 57 disjoint,
    /// gapless ranges that end at the allocated length. An off-by-one in
    /// `double_off`/`single_off` would not fail any GPU call — it would hand
    /// a block another block's modulation vector and quietly corrupt the
    /// image — so the arithmetic is pinned here, on the CPU.
    #[test]
    fn mod_all_slots_tile_the_buffer_without_gaps_or_overlap() {
        // Real FLUX.1-dev geometry.
        let (d, layers, singles) = (3072usize, 19usize, 38usize);
        // Same two expressions `Gpuf::build_mod_all` allocates from, so a
        // change to the layout cannot pass this test by moving with it.
        let single_base = ModAll::single_base(layers, d);
        let total = ModAll::total(layers, singles, d);
        // Absolute anchor, so both size helpers cannot drift together: the
        // 4.0 MB figure the `ModAll` doc comment quotes.
        assert_eq!(total, 1_050_624, "19*12d + 38*3d at d=3072");

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for b in 0..layers {
            for img in [true, false] {
                let off = ModAll::double_off(b, d, img);
                ranges.push((off, off + 6 * d));
            }
        }
        for b in 0..singles {
            let off = ModAll::single_off(single_base, b, d);
            ranges.push((off, off + 3 * d));
        }
        assert_eq!(ranges.len(), 2 * layers + singles, "one slot per linear");

        ranges.sort_unstable();
        assert_eq!(ranges[0].0, 0, "first slot starts at 0");
        for w in ranges.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "slots {:?} and {:?} are not adjacent",
                w[0], w[1]
            );
        }
        assert_eq!(
            ranges.last().unwrap().1,
            total,
            "the last slot must end at the allocated length"
        );

        // Image before text within a double block, and block b before b+1 —
        // the order `build_mod_all` fills them in.
        assert_eq!(ModAll::double_off(0, d, true), 0);
        assert_eq!(ModAll::double_off(0, d, false), 6 * d);
        assert_eq!(ModAll::double_off(1, d, true), 12 * d);
        assert_eq!(ModAll::single_off(single_base, 0, d), single_base);
    }

    #[test]
    fn split_linear2_takes_column_ranges_of_every_row() {
        // [d=2, k=5] with k - d = 3 MLP columns: rows are output features, so
        // the split must slice each row, not cut the buffer in half.
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (w_attn, w_mlp) = split_linear2(&data, 2, 5);
        assert_eq!(w_attn, vec![0.0, 1.0, 5.0, 6.0]);
        assert_eq!(w_mlp, vec![2.0, 3.0, 4.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn split_linear2_is_the_same_for_f16_bit_patterns() {
        // The streamed upload splits `u16` (f16 bits); the eager upload
        // splits `f32`. One generic helper, so the column ranges agree.
        let data: Vec<u16> = (0..10).collect();
        let (w_attn, w_mlp) = split_linear2(&data, 2, 5);
        assert_eq!(w_attn, vec![0, 1, 5, 6]);
        assert_eq!(w_mlp, vec![2, 3, 4, 7, 8, 9]);
    }

    /// The padded row copy is the whole correctness surface of the weight
    /// pad on the host side: a wrong stride here lays every row down at the
    /// wrong offset, which the kernel cannot detect — it would read the
    /// neighbouring row's data and return plausible wrong numbers.
    #[test]
    fn pad_rows_into_lays_rows_down_at_the_pitch_and_leaves_the_pad_alone() {
        // 3 rows of k = 4 at pitch 6: two pad columns per row.
        let src: Vec<u16> = (1..=12).collect();
        let mut dst = vec![0xDEADu16; 3 * 6];
        pad_rows_into(&src, &mut dst, 3, 4, 6);
        assert_eq!(
            dst,
            vec![
                1, 2, 3, 4, 0xDEAD, 0xDEAD, //
                5, 6, 7, 8, 0xDEAD, 0xDEAD, //
                9, 10, 11, 12, 0xDEAD, 0xDEAD,
            ],
            "each row starts at r*pitch and only its first k columns are written"
        );
    }

    #[test]
    fn pad_rows_into_is_a_plain_copy_at_pitch_k() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 6];
        pad_rows_into(&src, &mut dst, 2, 3, 3);
        assert_eq!(dst, src, "pitch == k must reproduce the packed layout");
    }

    /// The chunked uploader reuses ONE zero-filled staging buffer across
    /// chunks, so a later chunk must not inherit the previous chunk's pad —
    /// which is only true because `pad_rows_into` never writes the pad.
    #[test]
    fn pad_rows_into_reuses_a_staging_buffer_without_leaking_pad() {
        let mut stage = vec![0u16; 2 * 5];
        pad_rows_into(&[1, 2, 3, 4, 5, 6], &mut stage, 2, 3, 5);
        assert_eq!(stage, vec![1, 2, 3, 0, 0, 4, 5, 6, 0, 0]);
        // Second chunk, one row: the row's k-prefix is overwritten, the pad
        // columns of row 0 stay zero, and row 1 is simply not uploaded.
        pad_rows_into(&[7, 8, 9], &mut stage[..5], 1, 3, 5);
        assert_eq!(stage[..5], [7, 8, 9, 0, 0]);
    }

    /// The pad is a pure function of (name, K), and it is what both the
    /// uploader and every GEMM call site read. These are the three classes
    /// that must NOT move: modulation weights (the GEMV reads a packed row),
    /// a K the LDS route cannot take, and the kill switch.
    #[test]
    fn modulation_weights_are_never_padded() {
        for name in [
            "double_blocks.3.img_mod.lin.weight",
            "double_blocks.3.txt_mod.lin.weight",
            "single_blocks.11.modulation.lin.weight",
            "final_layer.adaLN_modulation.1.weight",
        ] {
            assert!(is_mod_weight(name), "{name} must be recognised as mod");
            assert_eq!(
                weight_pitch(name, 3072),
                3072,
                "{name} must stay packed for the GEMV"
            );
        }
        for name in [
            "double_blocks.3.img_attn.qkv.weight",
            "single_blocks.11.linear1.weight",
            "single_blocks.11.linear2.w_mlp.weight",
            "final_layer.linear.weight",
        ] {
            assert!(!is_mod_weight(name), "{name} must not be treated as mod");
        }
    }

    /// `Gpuf::mod_gemv` needs a live device, uploaded weights and a stream,
    /// so its fail-closed bias guard cannot be exercised without a GPU. What
    /// regressed was the guard's DELETION, not its logic, so this asserts the
    /// guard is still in the source — the cheapest check that catches the
    /// exact regression, and it runs in the no-GPU suite next to the other
    /// modulation-weight contract above.
    #[test]
    fn mod_gemv_keeps_the_flux1_missing_bias_guard() {
        let src = include_str!("flux_gpu.rs");
        assert!(
            src.contains("if bias.is_none() && self.cfg.bias {"),
            "flux_gpu::mod_gemv must still refuse a `cfg.bias` (FLUX.1) \
             modulation weight whose `.bias` sibling did not stage — a plain \
             Option there computes y = W*dv with no bias instead of failing"
        );
        assert!(
            src.contains("has no `.bias` sibling"),
            "the guard must name the missing tensor"
        );
    }

    #[test]
    fn a_ragged_k_is_never_padded() {
        // The pitch-aware entries live only on the LDS route, which needs
        // K % 64 == 0; everything else falls back to a kernel with no pitch.
        assert_eq!(weight_pitch("img_in.weight", 48), 48);
        assert_eq!(weight_pitch("img_in.weight", 3056), 3056);
    }

    /// The other half of the same contract: a non-modulation weight on a K
    /// the LDS route takes DOES move by exactly the configured pad. Written
    /// against `weight_pad()`/`pitch_route_ok()` rather than the literal 64
    /// so the test still means something under `HIPFIRE_FLUX_WPAD=0` and
    /// under the `HIPFIRE_FLUX_GEMM_{LDS,WIDE}=0` A/B knobs.
    #[test]
    fn a_gemm_weight_moves_by_exactly_the_configured_pad() {
        for k in [3072usize, 12288, 15360] {
            let want = if pitch_route_ok(k) {
                k + weight_pad()
            } else {
                k
            };
            assert_eq!(weight_pitch("single_blocks.0.linear1.weight", k), want);
            // Whatever the pad is, it must keep the staging half16 loads
            // 32-byte aligned — the one precondition the kernel cannot check.
            assert_eq!(weight_pitch("single_blocks.0.linear1.weight", k) % 16, 0);
        }
    }

    /// A FLUX.1 config with EVERY optional key block present (guidance
    /// included), so iterating its manifest is a complete sweep of the names
    /// the FLUX.2 classifiers below must never claim.
    fn flux1_dev_cfg() -> FluxDiffusionConfig {
        FluxDiffusionConfig::from_json(&json!({
            "hidden_size": 3072,
            "num_layers": 19,
            "num_single_layers": 38,
            "num_attention_heads": 24,
            "head_dim": 128,
            "patch_size": 2,
            "guidance_embed_dim": 256,
            "pooled_projection_dim": 768,
            "axes_dim": [16, 56, 56],
            "latent_channels": 16,
            "txt_hidden_dim": 4096,
        }))
        .unwrap()
    }

    /// The FLUX.1-byte-identity claim rests on this classifier matching by
    /// FULL NAME. The trap it exists to avoid: an `ends_with(".linear.weight")`
    /// spelling would also catch FLUX.1's `final_layer.linear.weight`, which
    /// IS padded today, and silently change the FLUX.1 upload layout — so
    /// that name is asserted false explicitly, and then the whole FLUX.1
    /// manifest is swept.
    #[test]
    fn flux2_packed_weights_are_exactly_the_six_modulation_and_embedder_tables() {
        for name in [
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
            "norm_out.linear.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
        ] {
            assert!(is_flux2_packed_weight(name), "{name} must stay packed");
            assert!(
                is_mod_weight(name),
                "{name} must reach `weight_pitch` through the packed set"
            );
            assert_eq!(
                weight_pitch(name, 3072),
                3072,
                "{name} must stay packed for the GEMV"
            );
        }
        for name in [
            "final_layer.linear.weight",
            "transformer_blocks.0.ff.linear_in.weight",
            "x_embedder.weight",
            "single_transformer_blocks.0.attn.to_out.weight",
        ] {
            assert!(
                !is_flux2_packed_weight(name),
                "{name} must keep the K+pad pitch"
            );
        }
        for key in expected_flux_keys(&flux1_dev_cfg()) {
            assert!(
                !is_flux2_packed_weight(&key.name),
                "FLUX.1 key `{}` must not be claimed by the FLUX.2 packed set",
                key.name
            );
        }
    }

    /// Klein spells its per-head QK-norm scales `.weight` where FLUX.1 spells
    /// them `.scale`, but `rmsnorm_batched` reads an f32 weight vector — so
    /// these four suffixes must leave the `.weight` → f16 rule, and nothing
    /// in a FLUX.1 manifest may follow them out.
    #[test]
    fn flux2_norm_scales_upload_f32_and_no_flux1_key_matches() {
        for name in [
            "transformer_blocks.3.attn.norm_q.weight",
            "transformer_blocks.3.attn.norm_k.weight",
            "transformer_blocks.3.attn.norm_added_q.weight",
            "transformer_blocks.3.attn.norm_added_k.weight",
            "single_transformer_blocks.7.attn.norm_q.weight",
            "single_transformer_blocks.7.attn.norm_k.weight",
        ] {
            assert!(is_flux2_norm_scale(name), "{name} must upload f32");
        }
        for name in [
            "transformer_blocks.3.attn.qkv.weight",
            "transformer_blocks.3.attn.add_qkv.weight",
            "norm_out.linear.weight",
            "single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight",
        ] {
            assert!(
                !is_flux2_norm_scale(name),
                "{name} is a GEMM operand and must stay f16"
            );
        }
        for key in expected_flux_keys(&flux1_dev_cfg()) {
            assert!(
                !is_flux2_norm_scale(&key.name),
                "FLUX.1 key `{}` must not be diverted to the f32 route",
                key.name
            );
        }
    }

    /// The scan is the load-time guard against a BF16 weight leaving f16's
    /// range. Its documented contract is narrow on purpose: ±inf only. A NaN
    /// already in the checkpoint is a DIFFERENT failure and must not be
    /// reported as an f16 overflow, so the whole NaN payload range is pinned
    /// as a non-match.
    #[test]
    fn f16_inf_scan_matches_only_infinities() {
        assert_eq!(first_f16_inf(&[0x3C00, 0x7C00]), Some(1), "+inf at index 1");
        assert_eq!(first_f16_inf(&[0xFC00]), Some(0), "-inf at index 0");
        assert_eq!(
            first_f16_inf(&[0x7C01, 0x7FFF, 0xFC01, 0x7BFF, 0x0000]),
            None,
            "NaN payloads and the largest finite f16 are not overflows"
        );
        assert_eq!(first_f16_inf(&[]), None, "an empty table has no overflow");
    }

    #[test]
    fn only_single_block_linear2_is_split() {
        assert!(is_single_linear2("single_blocks.7.linear2.weight"));
        assert!(!is_single_linear2("single_blocks.7.linear1.weight"));
        assert!(!is_single_linear2("single_blocks.7.linear2.bias"));
        assert!(!is_single_linear2("double_blocks.0.img_mlp.2.weight"));
    }

    #[test]
    fn from_host_unknown_key_panics_on_get() {
        // `from_host` needs a real Gpu (HIP) so it is exercised by the
        // `gpu_flux_stream_upload` lab example; here we pin the borrow contract only.
        let cfg = tiny_cfg();
        let host = crate::flux::FluxWeights::synthetic(&cfg);
        let names: Vec<String> = expected_flux_keys(&cfg)
            .into_iter()
            .map(|k| k.name)
            .collect();
        assert_eq!(names.len(), host.tensors.len());
    }
}
