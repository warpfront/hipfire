// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Legacy concurrent serve protocol: the experimental fixed-slot `SlotEngine`
// (`serve.multi_slot`). It stays explicitly labeled legacy and is never the
// default route.
//
// What actually serializes default serve: the CLI's `Mutex<ServeRuntime>`
// covers model/request setup only (the engine handle is cloned and the lock
// dropped before generation), but the single daemon owns one mutable resident
// model and runs one generation at a time unless a continuous-batch route is
// admitted. Batch routes are daemon-owned (`hipfire-generate::batch`): the
// existing fixed-lane Qwen/LFM/EP drivers, and the VMM continuous-batching
// route (`serve.vmm_batch`, default off) that plans steps with
// `slot_batch::BatchPlanner` over one resident weight set and per-request VMM
// state. `SlotEngine` lives outside the CLI mutex and loads its own model; it
// is not that VMM route.

// The engine LOOP lives in `hipfire-arch-qwen35::serve_engine`, not here:
// it needs `forward_batch_slots_graphed`, and that crate already depends on
// this one, so putting the loop here would be a dependency cycle. This module
// owns the protocol types both sides share — the same layering reason
// `swap::snapshot` takes an opaque buffer slice instead of a `DeltaNetState`.

use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// Visual payload for a VL request.
///
/// Image decode is CPU-side (daemon). `vision_forward` runs on the slot
/// engine thread, which exclusively owns the GPU and VisionWeights.
/// The engine splices the resulting embeddings at `<|image_pad|>` positions
/// during a per-token prefill that uses `forward_scratch_embed_mrope`.
///
/// `mrope_positions` carries the 3D (t, h, w) position for every prompt token,
/// already offset by `base` so values are absolute rope phases. `rope_delta`
/// is added to the running sequence length for decode-step positions.
pub struct VisualData {
    /// Vision-tower patch tensor from `extract_patches` (CPU).
    pub patches: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Number of post-merge visual tokens to splice at image_pad positions.
    pub n_visual_tokens: usize,
    /// 3D M-RoPE positions for every prompt token, offset by base.
    pub mrope_positions: Vec<[i32; 3]>,
    /// rope_delta for decode positions past the prompt.
    pub rope_delta: i32,
}

/// One request handed to the engine.
///
/// `reply` is the client's own channel: the engine streams this request's
/// events down it and nothing else. A dropped receiver is how the engine
/// learns the client is gone.
pub struct SubmitRequest {
    /// Full cold render of the conversation. Used when nothing is reusable.
    pub prompt_tokens: Vec<u32>,
    /// Hashes of the conversation's user turns, in order. Identity for
    /// continuation matching — see `Session::convo`.
    pub convo: Vec<u64>,
    pub continuation: Continuation,
    pub max_tokens: usize,
    /// 0.0 = greedy. Per-request; the engine installs it on the slot.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 disables top-k.
    pub top_k: i32,
    pub seed: u32,
    /// Recency window in recent tokens for the penalties below. 0 disables
    /// token penalties regardless of the penalty values.
    pub repeat_window: usize,
    /// Multiplicative recency-weighted repeat penalty; 1.0 = off.
    pub repeat_penalty: f32,
    /// OpenAI flat presence penalty; 0.0 = off. This is the mechanism that
    /// suppresses block-level repetition loops on long generations.
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (scaled by in-window count); 0.0 = off.
    pub frequency_penalty: f32,
    /// min-p cutoff; 0.0 = off.
    pub min_p: f32,
    /// Visual embeddings + M-RoPE for VL requests. None for text-only.
    pub visual_data: Option<VisualData>,
    /// JSON Schema for structured output (spec §7 G1/G2). When present, the
    /// engine compiles it into a `SchemaMatcher` and applies a pre-sampling
    /// token mask so every emitted token conforms to the schema. None for
    /// unconstrained requests. The schema object is CLI-validated and
    /// compiled at `validate_generate_caps` before submit; the engine
    /// recompiles on admit to build the per-request cursor.
    pub json_schema: Option<serde_json::Value>,
    /// Whether the assistant turn opened inside a `<think>` span (spec §7.2
    /// framing-aware grammar cursor). When true and `json_schema` is set,
    /// the grammar mask is deferred until `</think>` is emitted — the think
    /// preamble is not JSON and must not be masked by the schema.
    pub started_in_think: bool,
    /// Enforced thinking budget in think tokens (vLLM
    /// `thinking_token_budget` parity): once the cursor consumes this many
    /// think tokens, the mask allows ONLY the think close, forcing the span
    /// to end through the normal commit path. `usize::MAX` = uncapped.
    ///
    /// Scope: enforcement rides the grammar cursor, so it engages on
    /// grammar-constrained requests (the case where an over-long think
    /// span was fatal — burn-to-max then unsatisfiable). Unconstrained
    /// requests keep OpenAI-style semantics: an over-long think span ends
    /// at max_tokens with finish=length.
    pub think_budget: usize,
    /// (spec §5.3). The unified permit carries this from HTTP through daemon
    /// to slots so bytes are charged once and released exactly once. Zero
    /// when the request did not pass through the byte-bounded queue.
    pub queue_bytes: u64,
    /// Submitter-chosen cancellation identity (spec §4.6 C6): a queued
    /// request has no session id yet, so `EngineCommand::CancelWaiting`
    /// matches parked work by this tag. Idempotent — an unmatched tag is a
    /// no-op. 0 means "no cancellation identity".
    pub request_tag: u64,
    pub reply: Sender<Event>,
}

/// How this turn extends what the engine already holds.
///
/// The two reuse shapes are NOT interchangeable, and untyped tokens plus a
/// convo hash cannot tell them apart: a new user turn and a tool-result
/// iteration both arrive as "a suffix to append", but a tool-result suffix
/// pinned onto a session that never emitted those calls feeds the model a
/// `<tool_response>` for a call it did not make, and a user-turn suffix pinned
/// onto a session that already served that turn duplicates it in the KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuation {
    /// Nothing to append: the engine matches on `prompt_tokens` alone.
    Cold,
    /// A new user turn, from `prompt_frame::continuation_suffix`. Matched
    /// against a session holding exactly the preceding user turns; the tokens
    /// are APPENDED to its stored tokens, so the result is a strict extension
    /// of the KV rather than a re-render of it.
    UserTurn(Vec<u32>),
    /// Tool results answering the tool calls `session` emitted on its previous
    /// turn, from `prompt_frame::continuation_suffix_tool_results`. The caller
    /// resolved `session` from the tool-call ids it handed the client, so the
    /// engine only re-checks that the session still holds this conversation.
    ToolResults { tokens: Vec<u32>, session: u64 },
}

impl Continuation {
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Cold => &[],
            Self::UserTurn(tokens) => tokens,
            Self::ToolResults { tokens, .. } => tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoneReason {
    Eos,
    MaxTokens,
    /// The client's receiver was dropped. The session is closed and its slot
    /// freed rather than generating into a void.
    ClientGone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectClass {
    Overload,
    Validation,
    Internal,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// `reused` — prompt tokens served from the session's existing KV;
    /// `prefill` — tokens actually prefilled this turn. Together they are the
    /// client-visible prompt accounting (OpenAI `usage.prompt_tokens` =
    /// reused + prefill, `cached_tokens` = reused).
    Accepted {
        session: u64,
        reused: usize,
        prefill: usize,
    },
    Token {
        id: u32,
    },
    /// `generated` — tokens delivered to the client (terminators excluded).
    Done {
        reason: DoneReason,
        generated: usize,
    },
    Rejected {
        class: RejectClass,
        reason: String,
    },
}

/// Post an event to a client, reporting a vanished receiver as an error.
///
/// The engine must be able to see this: a request whose client has gone still
/// holds a slot, and on a 4-slot box that is 25% of capacity generating output
/// nobody will read.
pub fn send_event(tx: &Sender<Event>, e: Event) -> Result<(), String> {
    tx.send(e)
        .map_err(|_| "client receiver dropped".to_string())
}

/// Serialized as the `engine_stats` object of every multi-slot `done` event;
/// the serve gateway exports it on `/metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineStats {
    pub admitted: usize,
    pub rejected: usize,
    pub evictions: usize,
    pub restores: usize,
    /// Continuations served from a session's existing KV. Counted separately
    /// from `restores`: a hit on a still-resident session restores nothing, and
    /// conflating the two makes a gate that never restores look like it did.
    pub prefix_hits: usize,
    /// Total tokens served from the cross-session prefix cache (spec §4.5).
    /// Distinct from `prefix_hits` (session-local continuation) and
    /// `restores` (swap-in): these are tokens skipped because a prior
    /// session published the same prefix to the radix index.
    pub reused_tokens: usize,
    /// Free physical KV pages at the last completed step (A20 soak
    /// telemetry). A monotonic decline across identical request cycles is
    /// the page-level leak signature the oracle asserts against.
    pub pool_free_pages: usize,
    /// Physical KV pages in the pool; 0 when the engine is not paged.
    pub pool_total_pages: usize,
    /// Slots holding a request, out of `total_slots`: sampled before each
    /// engine step and decremented as each request finishes, before its
    /// terminal event, so a `done` snapshot counts only the slots still busy.
    pub active_slots: usize,
    pub total_slots: usize,
    /// Swap-tier failures (park or restore) since start.
    pub swap_failures: usize,
    /// Live swap-tier usage and effective budgets, refreshed by the engine
    /// each step. `/health` publishes these from the daemon's `done` event
    /// (and from the load ack before the first request).
    pub swap_host_used: u64,
    pub swap_disk_used: u64,
    pub swap_host_budget: u64,
    pub swap_disk_budget: u64,
}

impl EngineStats {
    pub fn note_admitted(&mut self) {
        self.admitted += 1;
    }
    pub fn note_rejected(&mut self) {
        self.rejected += 1;
    }
    pub fn note_eviction(&mut self) {
        self.evictions += 1;
    }
    pub fn note_restore(&mut self) {
        self.restores += 1;
    }

    pub fn note_reused_tokens(&mut self, n: usize) {
        self.reused_tokens += n;
    }
    pub fn note_prefix_hit(&mut self) {
        self.prefix_hits += 1;
    }
    pub fn note_pool_pages(&mut self, free: usize, total: usize) {
        self.pool_free_pages = free;
        self.pool_total_pages = total;
    }
    pub fn note_slots(&mut self, active: usize, total: usize) {
        self.active_slots = active;
        self.total_slots = total;
    }
    pub fn note_slot_released(&mut self) {
        self.active_slots = self.active_slots.saturating_sub(1);
    }

    /// Refresh the swap-tier snapshot from the live manager. Cheap: reads
    /// counters and three integers, no allocation.
    pub fn note_swap(&mut self, swap: &crate::swap::SwapManager) {
        let (_, _, failures) = swap.stats();
        self.swap_failures = failures;
        self.swap_host_used = swap.store().host_used();
        self.swap_disk_used = swap.store().disk_used();
        self.swap_host_budget = swap.store().host_budget();
        self.swap_disk_budget = swap.store().disk_budget();
    }
}

// ── Slot-engine handle (arch-erased multi-slot engine) ───────────────────
//
// The daemon's slot-mode dispatch is a `Carrier` hook in `hipfire-loader`.
// `hipfire-loader` cannot name a concrete arch engine type (that would be a
// loader -> arch -> loader cycle), so the engine is carried across the
// crate boundary as `Box<dyn SlotEngineHandle>`. The trait object is the
// single arch-erased surface `handle_generate` drives; the concrete engine
// (e.g. `hipfire_arch_qwen35::serve_engine::SlotEngine`) implements it.
//
// `SlotEngineConfig` is the family-neutral spawn record — every field is a
// primitive/`PathBuf`/`String`/`Option`, so a carrier can map it onto its
// own engine config without seeing arch types. It mirrors
// `hipfire_arch_qwen35::serve_engine::EngineConfig` field-for-field; keep
// them in lock-step (the qwen35 carrier maps them 1:1).

/// Family-neutral multi-slot engine spawn parameters. Constructed once by
/// the daemon from serve.* config keys + the load request, then handed to
/// `Carrier::spawn_slot_engine`, which adapts it into the concrete engine's
/// own config. Field-for-field with the qwen35 `EngineConfig`; a second
/// engine family may ignore fields it does not implement.
#[derive(Debug, Clone)]
pub struct SlotEngineConfig {
    pub model_path: PathBuf,
    pub n_slots: usize,
    /// Per-slot generation cap (tokens) used by the paged-KV budget.
    pub cap_tokens: usize,
    /// Prompt-chunk size (tokens) per prefill step.
    pub prefill_chunk: usize,
    /// Host-side swap/scratch budget in bytes.
    pub host_budget_bytes: u64,
    /// Directory the swap manager writes cold slot snapshots to.
    pub swap_dir: PathBuf,
    /// Disk-tier ceiling for swap spill files. 0 disables the disk tier.
    pub swap_disk_bytes: u64,
    /// The checkpoint is a VL model (vision tower present in the HFQ or in a
    /// separate `.vl` sidecar). Affects RoPE/M-RoPE + embed routing. On the
    /// carrier-delegated slot route this is the load's ladder-gated answer
    /// (`vision_mode=off` clears it), not a bare probe.
    pub is_vl: bool,
    /// Path to the `.vl` vision sidecar, if the vision weights live outside
    /// the text HFQ. Ladder-gated like [`Self::is_vl`]: `off` leaves it
    /// `None`, so an engine must not rediscover a sibling on its own.
    pub vl_path: Option<PathBuf>,
    /// The load's resolved `vision.mode` (`off`/`auto`/`resident`/`on`). An
    /// engine that loads a tower derives its residency from it (`resident`
    /// uploads the whole tower; every other mode streams it).
    pub vision_mode: String,
    /// MTP draft depth (K+1 verify rows per speculative cycle). 0 = off.
    pub mtp_k: usize,
    /// The `params.mtp_mode` tri-state (`off`/`auto`/`on`). `mtp_k > 0` alone
    /// cannot distinguish `auto` from an explicit `on`, so the slots arming
    /// site reads this (not `mtp_k`) for the producer-capability gate
    /// (`hipfire_config::mtp_mode_admits`: `auto` arms only a producer that
    /// declares a measured AR floor) and the host-spill decline.
    pub mtp_mode: hipfire_config::MtpMode,
    /// Raw KV-mode string from the load request (empty = resolve from
    /// env/config in the engine).
    pub kv_mode_raw: String,
    /// Effective KV backend (`legacy`/`vmm`); slot engines may refuse `vmm`.
    pub kv_backend: String,
    /// Cross-session prefix cache enabled (spec §4.5–4.6).
    pub prefix_cache: bool,
    /// Checkpoint pool max bytes; `prefix_cache` with `0` must refuse at
    /// spawn rather than fail mid-request.
    pub prefix_cache_max_bytes: u64,
    /// Global trunk-row budget per step (spec §5.2 S2).
    pub max_batch_tokens: usize,
    /// Minimum prefill quantum (spec §5.2 S2 / §5.3 S3).
    pub prefill_min_tokens: usize,
    /// Bounded waiting room: max queued request count (spec §5.3 S3).
    pub wait_max_count: usize,
    /// Bounded waiting room: max total queued bytes (spec §5.3 S3).
    pub wait_max_bytes: u64,
    /// Per-waiter timeout in scheduler ticks (spec §5.3 S3).
    pub queue_timeout_ms: u64,
    /// Structured-output jump-forward (spec §7.3 G3).
    pub structured_jump_forward: bool,
    /// DFlash2 draft path; `None` = DFlash off.
    pub dflash_draft: Option<PathBuf>,
    /// `dflash_mode=on`: a draft load failure fails the engine load.
    pub dflash_required: bool,
}

/// What the daemon's slot-mode admission and `loaded` ACK need from a
/// checkpoint *before* any GPU work: geometry, arch, tokenizer.
///
/// The daemon's own preflight reads this off the qwen35 HFQ layout, which is
/// where every slot-capable arch used to come from. An arch whose checkpoint
/// is not an HFQ at all (arch 17's encoder loads a safetensors dir as
/// happily as a container) supplies it through `Carrier::preflight_slot_model`
/// instead, and the daemon fills the same struct from it. The generate-only
/// pieces the daemon's preflight also produces (chat template, vision config,
/// recommended sampling) have no meaning on such a route and stay `None`.
pub struct SlotModelPreflight {
    pub arch_id: u32,
    /// `hipfire_loader::arch_label(arch_id)` equivalent, for the ACK.
    pub arch_str: String,
    pub dim: usize,
    pub layers: usize,
    pub vocab: usize,
    pub tokenizer: crate::tokenizer::Tokenizer,
    /// A vision tower will be resident.
    pub is_vl: bool,
    /// An audio tower will be resident (arch 18's `.vl` pack carries it next
    /// to the vision tower; every other arch answers `false`). Capability,
    /// not discovery: the load's sidecar gate turns it into residency.
    pub audio: bool,
    /// The embed route's video leg will be served (arch 18 only — its frames
    /// ride the vision tower, so this tracks the same residency as `is_vl`
    /// there and is `false` on every arch without a video leg). Capability,
    /// not discovery: the load's sidecar gate turns it into residency.
    pub video: bool,
    /// Resolved `.vl` sidecar path, when the tower lives outside the trunk.
    pub vl_path: Option<PathBuf>,
}

/// Which prompt the encoder wraps the input in. Queries and documents carry
/// different markers and different truncation limits, and the document side
/// additionally drops the multi-vector mask's punctuated positions — so the
/// kind is part of the request, not a server-side default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Query,
    Document,
}

/// One embedding encode request handed to a slot engine.
///
/// The engine owns the tokenizer, so the text rides raw — the same split the
/// sequential `embed` wire body uses. `images`/`audios` are decoded media
/// container bytes (PNG/JPEG, WAV), in request order; the engine runs the
/// vision/audio towers itself, on its own GPU thread. Empty = text-only, and
/// an arch whose tower is absent refuses a non-empty list per request rather
/// than ignoring it.
///
/// `videos` is the third modality: ONE ENTRY PER VIDEO, each a list of that
/// clip's FRAMES (encoded PNG/JPEG, in display order) — the sampler's input,
/// not a container. Video is arch 18 only, and its frames ride the vision
/// tower, so an arch without one refuses a non-empty list by name.
#[derive(Clone, Debug)]
pub struct EmbedRequest {
    pub kind: EmbedKind,
    pub text: String,
    pub images: Vec<Vec<u8>>,
    pub audios: Vec<Vec<u8>>,
    pub videos: Vec<Vec<Vec<u8>>>,
    /// The wire request id, for diagnostics only.
    pub request_tag: String,
    /// Matryoshka output width (embeddinggemma2, arch 18): truncate the
    /// pooled vector to this many components and re-normalize. `None`/0 =
    /// the model's full width. A nonzero value is refused (not ignored) by
    /// arches without MRL support — arch 17's encoder is multi-vector with a
    /// fixed 128-d head, so a `dims` there is a caller error, not a no-op.
    pub dims: Option<usize>,
}

/// One encode's result: the post-skiplist, L2-normalized token matrix
/// `[n_tokens, dim]` row-major, plus the accounting the wire reply needs.
#[derive(Clone, Debug)]
pub struct EmbedResponse {
    pub kind: EmbedKind,
    pub embeddings: Vec<f32>,
    /// Rows in `embeddings` (the mask's survivors).
    pub n_tokens: usize,
    /// Token-embedding width.
    pub dim: usize,
    /// Tokens the input tokenized to, before the mask.
    pub n_input_tokens: usize,
}

/// Arch-erased multi-slot engine. The concrete engine runs its own thread
/// and is driven through this narrow surface: submit a request, cancel a
/// queued one, close/reset a session, read stats, and shut down. The wire
/// types (`SubmitRequest`, `EngineStats`) already live in this module, so
/// the trait stays family-neutral. Object-safe; carried as
/// `Box<dyn SlotEngineHandle>` by `hipfire_loader::Carrier::spawn_slot_engine`.
pub trait SlotEngineHandle: Send + Sync {
    /// Enqueue a generate request. `Err` when the engine is shutting down.
    fn submit(&self, req: SubmitRequest) -> Result<(), String>;
    /// Cancel a QUEUED request by submitter tag. Idempotent; tag 0 is a no-op.
    fn cancel_waiting(&self, request_tag: u64);
    /// Close a session synchronously, cancelling any in-flight request and
    /// releasing its session/pool/swap state. Never silently ignored.
    fn close(&self, session: u64) -> Result<(), String>;
    /// Drop every session and swap entry back to a cold empty table. `Err`
    /// while any slot is still generating.
    fn reset(&self) -> Result<(), String>;
    /// Snapshot of engine-level counters (submitted/rejected/completed/…).
    fn stats(&self) -> EngineStats;
    /// Drain ids of sessions the engine closed on its own (the idle sweep).
    /// The daemon prunes its pending-tool broker from this list; engines
    /// without an internal sweep return an empty vec.
    fn take_closed_sessions(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Serve one batch of embedding encodes synchronously, one result per
    /// request in order.
    ///
    /// Synchronous (not the `submit`/`Event` streaming shape) because an
    /// encode has exactly one result and no incremental output: a caller
    /// hands over the requests it already has and waits. The engine is free
    /// to merge *other* callers' pending requests into the same forward — that
    /// is what a multi-slot embedding engine buys — so a single-request call
    /// may still be served as part of a larger batch.
    ///
    /// The outer `Err` is an engine-level failure (shutting down, GPU gone).
    /// The inner one is per request: one bad input must not fail its
    /// neighbours in the same forward.
    ///
    /// Defaulted: a generation engine has no synchronous encode verb, so the
    /// daemon's `embed` arm leaves the default `Err` alone and keeps the
    /// ordinary path. Only an embedding arch overrides it.
    fn encode(
        &self,
        _reqs: Vec<EmbedRequest>,
    ) -> Result<Vec<Result<EmbedResponse, String>>, String> {
        Err("this engine does not serve embeddings".into())
    }

    /// Consuming shutdown: close the command channel, fail in-flight
    /// requests, join the engine thread and free GPU state. On `Err` the
    /// caller may withhold `unloaded` to keep the load registered.
    fn shutdown(self: Box<Self>) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn a_request_whose_client_vanished_is_detected_before_work_is_done() {
        let (tx, rx) = channel::<Event>();
        drop(rx);
        assert!(
            send_event(&tx, Event::Token { id: 1 }).is_err(),
            "a dropped receiver must be visible so the engine can free the slot"
        );
    }

    #[test]
    fn a_live_client_receives_its_events_in_order() {
        let (tx, rx) = channel::<Event>();
        send_event(
            &tx,
            Event::Accepted {
                session: 4,
                reused: 10,
                prefill: 2,
            },
        )
        .unwrap();
        send_event(&tx, Event::Token { id: 7 }).unwrap();
        send_event(
            &tx,
            Event::Done {
                reason: DoneReason::Eos,
                generated: 1,
            },
        )
        .unwrap();
        assert_eq!(
            rx.recv().unwrap(),
            Event::Accepted {
                session: 4,
                reused: 10,
                prefill: 2
            }
        );
        assert_eq!(rx.recv().unwrap(), Event::Token { id: 7 });
        assert_eq!(
            rx.recv().unwrap(),
            Event::Done {
                reason: DoneReason::Eos,
                generated: 1
            }
        );
    }

    #[test]
    fn stats_start_at_zero_and_count_each_outcome() {
        let mut s = EngineStats::default();
        s.note_admitted();
        s.note_admitted();
        s.note_rejected();
        assert_eq!(s.admitted, 2);
        assert_eq!(s.rejected, 1);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.restores, 0);
    }
}
