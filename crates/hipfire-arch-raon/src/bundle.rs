// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! RaonBundle (config, weights, state, scratch, work, eos) and the generation
//! entry points built on the forward. Owner: slice A/I.
//!
//! The bundle is also the model's [`hipfire_runtime::session_cache::SessionState`]
//! (see [`crate::session`]): with a cache attached, a prefill restores the
//! snapshot `ArchModel::session_plan` returned and captures new ones on the
//! cold prefill chunk grid, byte-identical to a cold prefill.

use crate::precision::{RaonPrecisionPlan, RaonPrecisionPolicy, SiteWeights};
use rdna_compute::{Gpu, GpuTensor};

use crate::config::RaonConfig;
use crate::forward::{
    admit_decode_context, decode_binding, decode_body, forward_rows, forward_token,
    stage_decode_token, stage_decode_token_on_stream, RaonWork,
};
use crate::session::session_stride;
use crate::spec_impl::RaonSpecState;
use crate::state::{RaonPrefillScratch, RaonState};
use crate::weights::RaonWeights;
use hipfire_runtime::session_cache::{
    SessionCache, SessionCacheUsage, SessionDriver, SessionRoute,
};

/// Decode-step execution route (`HIPFIRE_GRAPH`, PLAN §3.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Whole decode step (all layers + head) recorded once per admitted
    /// context bucket and replayed per token. Product default.
    Graph,
    /// Per-launch dispatch (`HIPFIRE_GRAPH=0`): the oracle/control route.
    Eager,
}

impl DecodeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            DecodeMode::Graph => "graph",
            DecodeMode::Eager => "eager",
        }
    }

    /// Resolve once at load: `HIPFIRE_GRAPH=0` selects eager, otherwise graph.
    /// Retained replay (Redline) owning the launch stream or CPU-executed
    /// weights make capture impossible, which is reported as `Eager`.
    ///
    /// (gfx1201 note: decode there shows intermittent 2x-slow device time per
    /// step, 40 vs 82 tok/s, in whole processes or in phases that flip
    /// mid-process. It reproduces in eager decode and in llama.cpp on the same
    /// card, so it is not a property of the graph route and graph stays the
    /// default: it is the faster route when the card is in its fast state.)
    fn resolve(gpu: &Gpu) -> Self {
        if gpu.flags.graph_forward == Some(false) {
            return DecodeMode::Eager;
        }
        if gpu.replay.is_enabled() {
            eprintln!("raon1_mini: retained replay owns the launch stream; decode stays eager");
            return DecodeMode::Eager;
        }
        if gpu.owns_host_cpu_weights() {
            hipfire_dispatch::log_capture_disabled_once();
            return DecodeMode::Eager;
        }
        DecodeMode::Graph
    }
}

/// Decode graph bookkeeping of one bundle. The recorded graph itself lives in
/// `gpu.graphs` (the generic AR graph slot) keyed by `binding`.
#[derive(Default)]
struct DecodeGraph {
    /// One real eager decode step ran, so every kernel module the decode body
    /// launches is loaded before any capture (a capture cannot load modules).
    warmed: bool,
    /// Binding the graph in `gpu.graphs` was recorded with; 0 = none.
    binding: u64,
    captures: u64,
    replays: u64,
    /// Kernel nodes in the current recording.
    kernel_nodes: usize,
    /// Host seconds spent inside `hipGraphLaunch` since load.
    launch_secs: f64,
}

/// Rows of the smallest prefill chunk the loader admits (the floor of the
/// admission ladder `4096 / 2048 / 1024 / 512`, see
/// `carrier::admit_prefill_rows`). The admitted chunk is
/// [`RaonBundle::prefill_chunk_rows`].
pub const PREFILL_SCRATCH_ROWS: usize = 512;

/// Prefill route. `Batched` is the product route (N rows per module call);
/// `Serial` (one `forward_token` per prompt token) exists as the explicit
/// oracle/control and is never a default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Serial,
    Batched,
}

impl PrefillMode {
    /// `HIPFIRE_RAON_PREFILL` (`serial` | `batched`); unset means batched.
    /// Anything else is a named error, never a silent fallback.
    pub fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var("HIPFIRE_RAON_PREFILL") {
            Err(_) => Ok(Self::Batched),
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "batched" => Ok(Self::Batched),
                "serial" => Ok(Self::Serial),
                other => Err(format!(
                    "HIPFIRE_RAON_PREFILL={other:?}: expected `serial` or `batched`"
                )),
            },
        }
    }
}

/// Pin the device's activation-quantized prefill feature flags OFF for this process: the qt44
/// attention-class projections (`weight_gemm`: gfx1201 `a8_prefill`, IU4, gfx11 `gfx11_a8_prefill`)
/// never take a tier in the typed policy ([`crate::precision`]), and the device defaults of those
/// flags would otherwise quantize activations behind the policy's back. Every tiered site is
/// reached through the policy's own routes instead.
fn pin_flag_quant_routes_off(gpu: &mut Gpu) {
    let mut flags = (*gpu.flags).clone();
    flags.iu4_prefill = Some(false);
    flags.a8_prefill = false;
    flags.gfx11_a8_prefill = false;
    gpu.flags = std::sync::Arc::new(flags);
}

/// Everything one loaded Raon-1-mini model owns on the device.
pub struct RaonBundle {
    pub config: RaonConfig,
    /// Immutable weights.
    pub weights: RaonWeights,
    /// Mutable per-session state (RDA recurrent state, conv history, MLA latent caches).
    pub state: RaonState,
    /// Prefill scratch sized for the admitted chunk rows (`prefill_chunk_rows`).
    pub scratch: RaonPrefillScratch,
    /// Module work buffers (mHC controls, RDA, MoE, dense FFN) matching `scratch`.
    pub work: RaonWork,
    /// EOS token ids (`generation_config.json`: `<|im_end|>`-class ids).
    pub eos: Vec<u32>,
    prefill_mode: PrefillMode,
    decode_mode: DecodeMode,
    graph: DecodeGraph,
    /// Prefill precision policy resolved at load ([`crate::precision`]): the request and the
    /// effective tier of every site.
    precision: RaonPrecisionPlan,
    /// Speculative-decode state (DFlash2 capture, per-row verify head); empty
    /// until a speculator arms it, never touched by AR prefill or decode.
    pub(crate) spec: RaonSpecState,
    /// Device architecture string (`gfx1201`, ...), part of the snapshot scope:
    /// kernels differ per device, so their bits do too.
    pub(crate) gpu_arch: String,
    /// Prefill snapshot cache shared by every session; `None` = off. Taken out
    /// of the bundle while it drives the bundle as its `SessionState`.
    pub(crate) session: Option<SessionDriver>,
}

impl RaonBundle {
    /// Assemble the bundle from its loaded parts, allocating the module work
    /// buffers and resolving `HIPFIRE_RAON_PREFILL` once. On error every part
    /// handed in is released.
    pub fn new(
        gpu: &mut Gpu,
        config: RaonConfig,
        weights: RaonWeights,
        state: RaonState,
        scratch: RaonPrefillScratch,
    ) -> Result<Self, String> {
        let arch = gpu.arch.clone();
        let built = PrefillMode::from_env()
            .and_then(|m| RaonPrecisionPolicy::from_env(&arch).map(|p| (m, p)))
            .and_then(|(m, policy)| {
                pin_flag_quant_routes_off(gpu);
                let precision = policy.resolve(&arch, &SiteWeights::collect(&weights));
                eprintln!("{}", precision.coverage_line());
                let mut work = RaonWork::new(gpu, &config, scratch.rows)?;
                work.set_site_tiers(precision.effective);
                if let Err(e) = work.mla.pack_absorbed(gpu, &config, &weights) {
                    work.free(gpu);
                    return Err(e);
                }
                Ok((m, work, precision))
            });
        match built {
            Ok((prefill_mode, work, precision)) => {
                let mut state = state;
                match state.fit_mirror_budget(gpu) {
                    Ok(rows) => eprintln!(
                        "raon1_mini: expanded K/V mirror budget {rows} rows of {} reserved context \
                         ({} B/token, {}); contexts beyond it use the absorbed route (HIPFIRE_RAON_MLA_ABSORB)",
                        state.max_seq(),
                        crate::state::mirror_row_bytes(state.dims()),
                        if state.dims().mirror_q8 { "Q8 (HIPFIRE_RAON_MIRROR_Q8)" } else { "BF16" }
                    ),
                    Err(e) => eprintln!("raon1_mini: mirror budget refit failed ({e}); keeping the admission estimate"),
                }
                let eos = config.eos_token_ids.clone();
                Ok(Self {
                    config,
                    weights,
                    state,
                    scratch,
                    work,
                    eos,
                    prefill_mode,
                    precision,
                    decode_mode: DecodeMode::resolve(gpu),
                    graph: DecodeGraph::default(),
                    spec: RaonSpecState::default(),
                    gpu_arch: arch,
                    session: None,
                })
            }
            Err(e) => {
                if let Err(fe) = scratch.free_gpu(gpu) {
                    eprintln!("raon1_mini: scratch free after assembly failure: {fe}");
                }
                if let Err(fe) = state.free_gpu(gpu) {
                    eprintln!("raon1_mini: state free after assembly failure: {fe}");
                }
                weights.free_gpu(gpu);
                Err(format!("raon1_mini: bundle assembly failed: {e}"))
            }
        }
    }

    /// Whether `token` ends generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Prefill route selected at load (see [`PrefillMode::from_env`]).
    pub fn prefill_mode(&self) -> PrefillMode {
        self.prefill_mode
    }

    /// The prefill precision policy resolved at load: the request and the effective tier of every
    /// site ([`crate::precision`]).
    pub fn precision_plan(&self) -> &RaonPrecisionPlan {
        &self.precision
    }

    /// Rows of one batched prefill chunk: the admitted scratch/work rows (see
    /// `carrier::admit_prefill_rows`; at least [`PREFILL_SCRATCH_ROWS`] unless
    /// the reserved context is shorter).
    pub fn prefill_chunk_rows(&self) -> usize {
        self.scratch.rows.min(self.work.rows())
    }

    /// Override the load-time route (oracle/control use).
    pub fn set_prefill_mode(&mut self, mode: PrefillMode) {
        self.prefill_mode = mode;
    }

    /// Decode route in effect: `"graph"` or `"eager"` (what benches record).
    pub fn decode_mode(&self) -> &'static str {
        self.decode_mode.as_str()
    }

    /// Override the decode route (oracle/control use). Switching never touches
    /// the slot; a graph recorded earlier stays valid while its binding does.
    pub fn set_decode_mode(&mut self, mode: DecodeMode) {
        self.decode_mode = mode;
    }

    /// `(captures, replays)` of the decode graph since load: a replay is one
    /// decode step served by a recorded graph (a capture step is not a replay).
    pub fn decode_graph_stats(&self) -> (u64, u64) {
        (self.graph.captures, self.graph.replays)
    }

    /// Host seconds spent inside `hipGraphLaunch` since load (a graph replay
    /// that costs host time comparable to the step is launch-bound).
    pub fn decode_graph_launch_secs(&self) -> f64 {
        self.graph.launch_secs
    }

    /// Kernel launches recorded in the current decode graph (0 = none).
    pub fn decode_graph_kernel_nodes(&self) -> usize {
        self.graph.kernel_nodes
    }

    /// Tokens consumed by the slot so far.
    pub fn n_tokens(&self) -> usize {
        self.state.n_tokens()
    }

    /// Context rows this slot reserved.
    pub fn max_seq(&self) -> usize {
        self.state.max_seq()
    }

    /// Logits of the last row of the last `prefill`/`decode_step`: F32
    /// `[vocab]` on the device holding BF16-rounded values.
    pub fn logits(&self) -> &GpuTensor {
        &self.scratch.logits
    }

    /// Host copy of [`logits`](Self::logits) for a per-token reader. Once the
    /// graph route owns a decode stream the copy is ordered on it and the host
    /// waits on that stream alone ([`Gpu::download_f32_stream_ordered`]); the
    /// plain `download_f32` would put a null-stream hop in every token.
    pub fn download_logits(&self, gpu: &Gpu) -> Result<Vec<f32>, String> {
        gpu.download_f32_stream_ordered(&self.scratch.logits)
            .map_err(|e| format!("raon1_mini: download logits: {e:?}"))
    }

    /// Token rows one prefill step stages: 1 on the serial control route, else
    /// the admitted chunk ([`prefill_chunk_rows`](Self::prefill_chunk_rows)).
    pub(crate) fn prefill_step_rows(&self) -> usize {
        match self.prefill_mode {
            PrefillMode::Serial => 1,
            PrefillMode::Batched => self.prefill_chunk_rows(),
        }
    }

    /// One prefill step: `piece` (1..=[`prefill_step_rows`](Self::prefill_step_rows)
    /// rows) from the slot's current position. `want_logits` leaves the last
    /// row's logits in [`logits`](Self::logits).
    pub(crate) fn forward_chunk(
        &mut self,
        gpu: &mut Gpu,
        piece: &[u32],
        want_logits: bool,
    ) -> Result<(), String> {
        let pos = self.state.n_tokens() as u32;
        if piece.len() == 1 {
            forward_token(
                &self.config,
                &self.weights,
                &mut self.state,
                &mut self.scratch,
                &mut self.work,
                gpu,
                piece[0],
                pos,
                want_logits,
            )
        } else {
            forward_rows(
                &self.config,
                &self.weights,
                &mut self.state,
                &mut self.scratch,
                &mut self.work,
                gpu,
                piece,
                pos,
                want_logits,
            )
        }
    }

    /// Append `tokens` to the slot starting at `n_tokens()`, in chunks of at
    /// most `scratch.rows`; the logits of the final token are left in
    /// [`logits`](Self::logits). A failure leaves the slot partially advanced:
    /// the caller resets it.
    pub fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("raon1_mini: prefill of zero tokens".into());
        }
        let start = self.state.n_tokens();
        if start + tokens.len() > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: {start} + {} tokens exceed the reserved context of {} rows",
                tokens.len(),
                self.state.max_seq()
            ));
        }
        let chunk = self.prefill_step_rows();
        let n_chunks = tokens.len().div_ceil(chunk);
        for (ci, piece) in tokens.chunks(chunk).enumerate() {
            self.forward_chunk(gpu, piece, ci + 1 == n_chunks)?;
        }
        Ok(())
    }

    /// Attach the session cache: [`session_prefill`](Self::session_prefill)
    /// then restores and captures snapshots through it.
    pub fn attach_session_cache(&mut self, cache: SessionCache) {
        self.session = Some(SessionDriver::new(cache));
    }

    /// The attached session cache, if any.
    pub fn session_cache(&self) -> Option<&SessionCache> {
        self.session.as_ref().map(SessionDriver::cache)
    }

    /// `SessionCache::usage` of the attached cache.
    pub fn session_usage(&self) -> Option<SessionCacheUsage> {
        self.session.as_ref().map(|s| s.cache().usage())
    }

    /// Snapshot stride in prompt tokens: the smallest multiple of the prefill
    /// width that is at least [`crate::session::SESSION_STRIDE_TOKENS`].
    pub fn session_stride_tokens(&self) -> usize {
        session_stride(self.prefill_step_rows())
    }

    /// Device bytes of one stride snapshot: the persistent slab plus one stride
    /// of rows of every latent stream.
    pub fn session_snapshot_bytes(&self) -> usize {
        let (_, persist) = self.state.persist_part();
        persist
            + self.session_stride_tokens()
                * self.state.dims().latent_row_bytes()
                * self.state.mla_cache.len()
    }

    /// Device bytes the live state may still map before it holds `through`
    /// tokens: unmapped latent growth of every MLA layer plus the expanded
    /// mirror's growth up to `min(through, mirror_budget_rows)` rows.
    pub fn context_growth_bytes(&self, gpu: &Gpu, through: usize) -> u64 {
        self.state.context_growth_bytes(gpu, through)
    }

    /// Before a request that may hold up to `through` tokens: have the cache
    /// free snapshots until [`context_growth_bytes`](Self::context_growth_bytes)
    /// more device bytes fit (the cache never takes memory the context needs).
    /// No-op without a cache.
    pub fn release_session_for_growth(&mut self, gpu: &mut Gpu, through: usize) {
        if self.session.is_none() {
            return;
        }
        let need = self.context_growth_bytes(gpu, through);
        if let Some(session) = self.session.as_mut() {
            session.release_for(gpu, need);
        }
    }

    /// Run `f` with the driver taken out, so it can drive `self` as its
    /// `SessionState`. `None` without a cache.
    fn with_session<R>(&mut self, f: impl FnOnce(&mut SessionDriver, &mut Self) -> R) -> Option<R> {
        let mut session = self.session.take()?;
        let result = f(&mut session, self);
        self.session = Some(session);
        Some(result)
    }

    /// Prefill `prompt` skipping the `reused` tokens `ArchModel::session_plan`
    /// returned (0 = cold): begin through the driver (restore, or cold reset),
    /// run the rest in prefill chunks that stop at every armed snapshot
    /// boundary, and leave the LAST prompt token's logits in
    /// [`logits`](Self::logits). Without a cache `reused` must be 0. Reports
    /// `prefill_progress` after every chunk. A failure leaves the slot
    /// partially advanced: the caller resets it.
    ///
    /// The driver runs without live continuation, so every resume point is a
    /// position of the cold chunk grid and the result equals a cold prefill's.
    pub fn session_prefill(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
    ) -> Result<(), String> {
        let len = prompt.len();
        if reused >= len {
            return Err(format!(
                "raon1_mini: reusing {reused} of {len} prompt tokens leaves nothing to compute"
            ));
        }
        if len > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: {len} prompt tokens exceed the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        match self.with_session(|session, bundle| {
            session.begin(gpu, bundle, prompt, SessionRoute::Ar, reused, false)
        }) {
            Some(result) => {
                result?;
            }
            None if reused == 0 => self.state.reset(gpu)?,
            None => {
                return Err(format!(
                    "raon1_mini: session cache is not attached; cannot reuse {reused} tokens"
                ))
            }
        }
        if self.state.n_tokens() != reused {
            return Err(format!(
                "raon1_mini: session begin left the state at {} tokens, expected {reused}",
                self.state.n_tokens()
            ));
        }
        let step = self.prefill_step_rows();
        let mut pos = reused;
        while pos < len {
            let boundary = self.session.as_ref().and_then(SessionDriver::next_boundary);
            if boundary.is_some_and(|b| b <= pos) {
                return Err(format!(
                    "raon1_mini: session boundary {} is not past position {pos}",
                    boundary.unwrap_or(0)
                ));
            }
            let stop = boundary.unwrap_or(len).min(len);
            for piece in prompt[pos..stop].chunks(step) {
                pos += piece.len();
                self.forward_chunk(gpu, piece, pos == len)?;
                hipfire_runtime::prefill_progress::report(pos, len);
            }
            if boundary == Some(stop) {
                self.with_session(|session, bundle| {
                    session.at_boundary(gpu, bundle, &prompt[..stop])
                })
                .unwrap_or(Ok(()))?;
            }
        }
        Ok(())
    }

    /// One decode step: consume `token`, leave its logits in
    /// [`logits`](Self::logits). Routed by [`decode_mode`](Self::decode_mode):
    /// graph replay per token (recorded on first use per admitted context
    /// bucket) or per-launch eager dispatch. Both advance the slot exactly once.
    pub fn decode_step(&mut self, gpu: &mut Gpu, token: u32) -> Result<(), String> {
        let pos = self.state.n_tokens();
        if pos + 1 > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: decode past the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        match self.decode_mode {
            DecodeMode::Eager => forward_token(
                &self.config,
                &self.weights,
                &mut self.state,
                &mut self.scratch,
                &mut self.work,
                gpu,
                token,
                pos as u32,
                true,
            ),
            DecodeMode::Graph => self.decode_step_graph(gpu, token, pos),
        }
    }

    /// Graph route of [`decode_step`](Self::decode_step).
    ///
    /// Host work stays outside the recorded region: context admission (which
    /// may grow the latent mapping / expanded mirror and thereby change the
    /// binding) and the token id upload into the stable device id buffer. The
    /// recorded step reads the token id and `seq_len` from device memory and
    /// advances `seq_len` itself (last kernel of the graph), so one recording
    /// replays correctly at every later position of the same binding with no
    /// per-token host write of the length.
    ///
    /// Recording executes nothing (HIP captures launches into the graph), so
    /// the capturing step launches the fresh graph once and consumes its token
    /// exactly once; no snapshot/restore is needed. The very first decode step
    /// of the bundle runs the same body eagerly instead, so every kernel module
    /// is loaded before a capture.
    fn decode_step_graph(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        let end = pos + 1;
        admit_decode_context(
            &self.config,
            &self.weights,
            &mut self.state,
            &self.scratch,
            &self.work,
            gpu,
            end,
        )?;
        if !self.graph.warmed {
            stage_decode_token(&self.config, &self.scratch, gpu, token)?;
            decode_body(
                &self.config,
                &self.weights,
                &self.state,
                &self.scratch,
                &self.work,
                gpu,
            )?;
            self.graph.warmed = true;
            self.state.note_host_tokens(1);
            return Ok(());
        }
        if gpu.active_stream.is_none() {
            let stream = gpu
                .hip
                .stream_create()
                .map_err(|e| format!("raon1_mini: decode graph stream: {e:?}"))?;
            gpu.active_stream = Some(stream);
        }
        // The token id rides the decode stream (one queue, no null-stream hop).
        stage_decode_token_on_stream(&self.config, &self.scratch, gpu, token)?;
        let binding = decode_binding(&self.weights, &self.state, &self.scratch, &self.work);
        let recorded = gpu.graphs.graph_exec.is_some()
            && self.graph.binding == binding
            && gpu.graphs.ar_forward_binding == binding;
        if !recorded {
            self.record_decode_graph(gpu, binding)?;
            self.graph.captures += 1;
        } else {
            self.graph.replays += 1;
        }
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("decode graph stream just ensured");
        let t_launch = std::time::Instant::now();
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, stream)
            .map_err(|e| format!("raon1_mini: decode graph launch: {e:?}"))?;
        self.graph.launch_secs += t_launch.elapsed().as_secs_f64();
        // The recorded step advanced the device `seq_len`; mirror it on the host.
        self.state.note_host_tokens(1);
        Ok(())
    }

    /// Record the decode body into the generic AR graph slot, replacing any
    /// previous recording (an old bucket, or another model's).
    fn record_decode_graph(&mut self, gpu: &mut Gpu, binding: u64) -> Result<(), String> {
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        self.graph.binding = 0;
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("decode graph stream ensured by caller");
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, stream)
            .map_err(|e| format!("raon1_mini: begin decode graph capture: {e:?}"))?;
        let body = decode_body(
            &self.config,
            &self.weights,
            &self.state,
            &self.scratch,
            &self.work,
            gpu,
        );
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("decode graph stream ensured by caller");
        if let Err(e) = body {
            gpu.graphs
                .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
            return Err(format!(
                "raon1_mini: decode graph capture failed (no eager fallback): {e}"
            ));
        }
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, stream)
            .map_err(|e| format!("raon1_mini: end decode graph capture: {e:?}"))?;
        gpu.graphs.ar_forward_binding = binding;
        self.graph.binding = binding;
        self.graph.kernel_nodes = gpu.graphs.ar_forward_blobs.len();
        Ok(())
    }

    /// Start a cold request (zero recurrent state, conv history, length) and
    /// forget the session driver's live state, whose rows this overwrites. A
    /// recorded decode graph stays valid: addresses and extents are unchanged.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(session) = self.session.as_mut() {
            session.forget_live();
        }
        self.state.reset(gpu)
    }

    /// Reset, then prefill `n_tokens` deterministic synthetic ids through the
    /// production [`prefill`](Self::prefill) route and wait for the device.
    /// Used by the loader's prefill benchmark.
    pub fn bench_prefill_rows(&mut self, gpu: &mut Gpu, n_tokens: usize) -> Result<(), String> {
        if n_tokens == 0 {
            return Err("raon1_mini: bench prefill of zero tokens".into());
        }
        self.reset(gpu)?;
        let vocab = self.config.vocab as u64;
        // Ids stay clear of the low special-token block and are well spread.
        let lo = 1024u64.min(vocab.saturating_sub(1));
        let span = vocab - lo;
        let tokens: Vec<u32> = (0..n_tokens as u64)
            .map(|i| (lo + (i.wrapping_mul(2654435761).wrapping_add(17)) % span) as u32)
            .collect();
        self.prefill(gpu, &tokens)?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("raon1_mini: bench prefill sync: {e:?}"))
    }

    /// Return every device buffer the bundle owns (work, scratch, state, weights).
    pub fn free_all(self, gpu: &mut Gpu) {
        let RaonBundle {
            config: _,
            weights,
            state,
            scratch,
            work,
            eos: _,
            prefill_mode: _,
            precision: _,
            decode_mode: _,
            graph,
            spec,
            gpu_arch: _,
            session,
        } = self;
        if let Some(mut session) = session {
            session.clear(gpu);
        }
        // A recording of this bundle reads the buffers freed below.
        if graph.binding != 0 && gpu.graphs.ar_forward_binding == graph.binding {
            gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        }
        work.free(gpu);
        spec.free(gpu);
        if let Err(e) = scratch.free_gpu(gpu) {
            eprintln!("raon1_mini: scratch free: {e}");
        }
        if let Err(e) = state.free_gpu(gpu) {
            eprintln!("raon1_mini: state free: {e}");
        }
        weights.free_gpu(gpu);
    }
}
