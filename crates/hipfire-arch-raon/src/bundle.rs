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

use crate::band::{edge_view, RaonBand};
use crate::config::RaonConfig;
use crate::forward::{
    admit_decode_context, band_admit, band_body, decode_binding, decode_body, forward_band,
    forward_rows, forward_token, require_whole_band, stage_decode_token,
    stage_decode_token_on_stream, BandInput, RaonWork,
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
                if config.n_layers != weights.layers.len() {
                    return Err(format!(
                        "raon1_mini: config view of {} layers does not match {} ({} layers loaded)",
                        config.n_layers,
                        weights.band.label(),
                        weights.layers.len()
                    ));
                }
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

    /// The layers this bundle covers (the whole model for a regular load).
    /// `config` is the band's view ([`RaonConfig::band_view`]); `n_layers` of
    /// the whole model is `band().total`.
    pub fn band(&self) -> &RaonBand {
        &self.weights.band
    }

    /// The edge view of `rows` stream rows (`scratch.stream_a`, from row 0): a
    /// band exports its result from it at band end and imports its input into
    /// it at band start. A view: never free it.
    pub fn edge_view(&self, rows: usize) -> Result<GpuTensor, String> {
        if rows == 0 || rows > self.prefill_chunk_rows() {
            return Err(format!(
                "raon1_mini: edge of {rows} rows outside 1..={}",
                self.prefill_chunk_rows()
            ));
        }
        Ok(edge_view(&self.config, &self.scratch, rows))
    }

    /// Bytes of one stream element of the edge (the `stream_a` plane's dtype:
    /// 4 for F32, 2 for BF16 streams).
    pub fn edge_elem_bytes(&self) -> usize {
        self.scratch.stream_a.dtype.size()
    }

    /// Bytes of an edge of `rows` rows in the stream plane's dtype.
    pub fn edge_bytes(&self, rows: usize) -> usize {
        crate::band::edge_bytes(&self.config, rows, self.edge_elem_bytes())
    }

    /// One batched prefill call of this band over one chunk: the first band
    /// takes `BandInput::Tokens(piece)`, a later band `BandInput::Streams`
    /// with the previous band's rows already imported into its
    /// [`edge_view`](Self::edge_view). The slot advances by the rows. The last
    /// band's `want_logits` leaves the last row's logits in
    /// [`logits`](Self::logits). The whole band keeps [`prefill`](Self::prefill).
    pub fn prefill_band_chunk(
        &mut self,
        gpu: &mut Gpu,
        input: BandInput<'_>,
        want_logits: bool,
    ) -> Result<(), String> {
        if self.prefill_mode != PrefillMode::Batched {
            return Err(format!(
                "raon1_mini: {}: band prefill is the batched route only",
                self.weights.band.label()
            ));
        }
        let start = self.state.n_tokens();
        if start + input.rows() > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: {start} + {} rows exceed the reserved context of {} rows",
                input.rows(),
                self.state.max_seq()
            ));
        }
        forward_band(
            &self.config,
            &self.weights,
            &mut self.state,
            &mut self.scratch,
            &mut self.work,
            gpu,
            input,
            want_logits,
        )
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
        require_whole_band(&self.weights, "prefill")?;
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
    ///
    /// A partial band (a mesh rank) never attaches one: its state is a slice of
    /// the model and a snapshot of it is not restorable on its own
    /// ([`SessionState::snapshot_scope`] reports `None` there, so the loader
    /// does not offer a cache). The cache handed in is dropped.
    pub fn attach_session_cache(&mut self, cache: SessionCache) {
        if !self.weights.band.is_whole() {
            eprintln!(
                "raon1_mini: session cache not attached on {}: snapshots need the whole model on one device",
                self.weights.band.label()
            );
            return;
        }
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
        require_whole_band(&self.weights, "session prefill")?;
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
        if !self.weights.band.is_first() {
            return Err(format!(
                "raon1_mini: decode_step takes a token on {}: only the first band embeds \
                 (later bands: decode_step_from_edge)",
                self.weights.band.label()
            ));
        }
        self.decode_step_in(gpu, Some(token))
    }

    /// One decode step of a band past the first: consume the one stream row the
    /// previous band left in [`edge_view`](Self::edge_view) (`scratch.stream_a`
    /// row 0) and advance this band's slot by one. On the last band the logits
    /// are left in [`logits`](Self::logits). Same routes as
    /// [`decode_step`](Self::decode_step).
    pub fn decode_step_from_edge(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if self.weights.band.is_first() {
            return Err(format!(
                "raon1_mini: decode_step_from_edge on {}: the first band takes a token",
                self.weights.band.label()
            ));
        }
        self.decode_step_in(gpu, None)
    }

    fn decode_step_in(&mut self, gpu: &mut Gpu, token: Option<u32>) -> Result<(), String> {
        let pos = self.state.n_tokens();
        if pos + 1 > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: decode past the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        match (self.decode_mode, token) {
            (DecodeMode::Eager, Some(token)) if self.weights.band.is_whole() => forward_token(
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
            (DecodeMode::Eager, token) => {
                let ids = [token.unwrap_or(0)];
                let input = match token {
                    Some(_) => BandInput::Tokens(&ids),
                    None => BandInput::Streams { rows: 1 },
                };
                forward_band(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    &mut self.scratch,
                    &mut self.work,
                    gpu,
                    input,
                    self.weights.band.is_last(),
                )
            }
            (DecodeMode::Graph, token) => self.decode_step_graph(gpu, token, pos),
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
    fn decode_step_graph(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        pos: usize,
    ) -> Result<(), String> {
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
        self.decode_graph_enqueue(gpu, token, true)?;
        // The recorded (or warm-up) step advanced the device `seq_len`; mirror it on the host.
        self.state.note_host_tokens(1);
        Ok(())
    }

    /// Device phase of one graph-route decode step (after
    /// [`admit_decode_context`]): the warm-up step runs the body eagerly, every
    /// later step stages the token id on the decode stream, records the graph
    /// when the binding changed, then launches it. Only enqueues on the
    /// device's active stream; the warm-up's id staging is the blocking one
    /// when `blocking_warm_stage` (the single-device route) and the
    /// stream-ordered one otherwise (the mesh epoch callback). The host
    /// mirror advance is the caller's.
    fn decode_graph_enqueue(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        blocking_warm_stage: bool,
    ) -> Result<(), String> {
        if !self.graph.warmed {
            if let Some(token) = token {
                if blocking_warm_stage {
                    stage_decode_token(&self.config, &self.scratch, gpu, token)?;
                } else {
                    stage_decode_token_on_stream(&self.config, &self.scratch, gpu, token)?;
                }
            }
            decode_body(
                &self.config,
                &self.weights,
                &self.state,
                &self.scratch,
                &self.work,
                gpu,
            )?;
            self.graph.warmed = true;
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
        if let Some(token) = token {
            stage_decode_token_on_stream(&self.config, &self.scratch, gpu, token)?;
        }
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
        Ok(())
    }

    /// Host phase of one mesh epoch of this band (see [`crate::mesh`]): validate
    /// the input against the band and admit the context for `n_tokens + rows`
    /// (map-ahead of the latent caches and expanded mirror, which may move the
    /// decode binding). Eager rows also stage the first band's token ids; a
    /// graph-route decode step stages its token on the stream inside
    /// [`mesh_enqueue`](Self::mesh_enqueue). Outside any epoch submission.
    /// Returns the rows of the call.
    pub fn mesh_admit(
        &mut self,
        gpu: &mut Gpu,
        input: BandInput<'_>,
        graph_decode: bool,
    ) -> Result<usize, String> {
        if self.prefill_mode != PrefillMode::Batched {
            return Err(format!(
                "raon1_mini: {}: mesh bands run the batched route only",
                self.weights.band.label()
            ));
        }
        let rows = input.rows();
        let start = self.state.n_tokens();
        if start + rows > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: {start} + {rows} rows exceed the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        if graph_decode {
            if rows != 1 {
                return Err(format!(
                    "raon1_mini: graph decode takes one row, got {rows} on {}",
                    self.weights.band.label()
                ));
            }
            admit_decode_context(
                &self.config,
                &self.weights,
                &mut self.state,
                &self.scratch,
                &self.work,
                gpu,
                start + 1,
            )?;
            return Ok(1);
        }
        band_admit(
            &self.config,
            &self.weights,
            &mut self.state,
            &self.scratch,
            &self.work,
            gpu,
            input,
        )
    }

    /// Device phase of one mesh epoch of this band, after
    /// [`mesh_admit`](Self::mesh_admit) with the same `graph_decode`: enqueue
    /// the band's body over `rows` rows on the device's active stream with no
    /// host wait (`token` is the first band's decode token on the graph route).
    /// The last band's `want_logits` leaves the last row's logits in
    /// [`logits`](Self::logits). The host mirror advance is
    /// [`mesh_commit`](Self::mesh_commit).
    pub fn mesh_enqueue(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        token: Option<u32>,
        want_logits: bool,
        graph_decode: bool,
    ) -> Result<(), String> {
        if want_logits && !self.weights.band.is_last() {
            return Err(format!(
                "raon1_mini: logits requested on {}: only the last band reads them out",
                self.weights.band.label()
            ));
        }
        if graph_decode {
            if rows != 1 || want_logits != self.weights.band.is_last() {
                return Err(format!(
                    "raon1_mini: graph decode is one row with the last band's logits ({rows} rows on {})",
                    self.weights.band.label()
                ));
            }
            return self.decode_graph_enqueue(gpu, token, false);
        }
        band_body(
            &self.config,
            &self.weights,
            &self.state,
            &self.scratch,
            &self.work,
            gpu,
            rows,
            want_logits,
        )
    }

    /// Advance the host mirror by the `rows` an enqueued epoch consumed (the
    /// device `seq_len` advanced inside the body).
    pub fn mesh_commit(&mut self, rows: usize) {
        self.state.note_host_tokens(rows);
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
