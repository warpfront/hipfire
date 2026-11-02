// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The arch-17 multi-slot engine: one GPU thread that owns the Qwen3.5
//! multi-vector encoder and serves concurrent embed requests as **one batched
//! forward**.
//!
//! Why this is not `hipfire-arch-qwen35::serve_engine`. That engine is a
//! *generation* engine: paged KV pools, session continuation matching, swap,
//! prefix cache, a scheduler with admission windows and prefill quanta. An
//! encoder has none of those — no KV cache at all (the bundle's
//! `kv_cache_mut()` is `None`), no sessions, no incremental output. Its one
//! useful concurrency primitive is that several sequences' rows can share a
//! forward, because only GatedDeltaNet and full attention couple rows and both
//! run per sequence inside the kernel (see `forward::encode_embeddings_batch`).
//!
//! So the shape here is deliberately small: a request channel, a collector
//! that groups whatever is waiting into one row block, one
//! `encode_embeddings_batch` call per group, and a reply per request. The GPU
//! and the bundle live on the engine thread and never leave it; the handle the
//! daemon holds carries only a channel and the stats counter.
//!
//! Batching policy: the collector takes the first request, then every request
//! already queued, then waits up to [`BATCH_WINDOW`] for the tail of a burst.
//! The wait is the only latency this design adds, and only when the engine has
//! nothing else to do — a second request that arrives while the first is being
//! tokenized still joins its batch. Groups are then split so no single forward
//! exceeds the row budget, since compute scales with rows while the weight
//! traffic does not.

use crate::bundle::EmbedBundle;
use crate::carrier::{assemble_input, load_qwen35_embed_from_source};
use crate::config::InputKind;
use crate::forward::{encode_embeddings_batch, EncodeInput};
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::serve::{
    EmbedKind, EmbedRequest, EmbedResponse, EngineStats, SlotEngineConfig, SlotEngineHandle,
    SubmitRequest,
};
use rdna_compute::Gpu;
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long the collector waits for a burst's tail once the queue is empty.
/// Two orders of magnitude under a short encode (a 64-row query is ~10 ms), so
/// an isolated request pays ~1% for the chance that a second one joins it; a
/// concurrent burst pays it once per batch, not once per request.
const BATCH_WINDOW: Duration = Duration::from_micros(2000);

/// Row budget for one forward when the caller configured none. Rows are what
/// the encoder's compute scales with, and the checkpoint's own kind limits
/// (1024 query / 4096 document on the pplx release, read from
/// `sentence_bert_config.json`) already cap a single request.
const DEFAULT_BATCH_ROWS: usize = 4096;

/// `EngineConfig` analogue for the encoder: the subset of
/// [`SlotEngineConfig`] an encode actually has a use for, plus the derived
/// batch bounds.
#[derive(Clone, Debug)]
pub struct EmbedEngineConfig {
    pub model_path: std::path::PathBuf,
    /// Requests per forward. The daemon's slot count, which is the
    /// concurrency the operator asked to be able to serve at once.
    pub max_batch_requests: usize,
    /// Rows per forward (the memory/compute bound).
    pub max_batch_rows: usize,
    pub batch_window: Duration,
}

impl EmbedEngineConfig {
    /// Map the daemon's family-neutral slot config onto this engine.
    ///
    /// `n_slots` is the request count and `max_batch_tokens` the row budget —
    /// the same roles they play on the generation route, where `n_slots` is
    /// how many requests share the step and `max_batch_tokens` bounds the rows
    /// in it. `cap_tokens` (per-slot context) has no counterpart here: this
    /// model truncates each input to the length it was trained at, inside
    /// `assemble_input`, so a second cap could only be wrong.
    pub fn from_slot_config(cfg: &SlotEngineConfig) -> Self {
        Self {
            model_path: cfg.model_path.clone(),
            max_batch_requests: cfg.n_slots.max(1),
            max_batch_rows: if cfg.max_batch_tokens == 0 {
                DEFAULT_BATCH_ROWS
            } else {
                cfg.max_batch_tokens
            },
            batch_window: batch_window_from_env(),
        }
    }
}

/// `HIPFIRE_EMBED_BATCH_WINDOW_US` overrides the collector's wait. `0` turns
/// the window off (batch only what is already queued when the engine is free),
/// which is the right setting for a latency-sensitive single-request
/// deployment; a large value batches harder at the cost of a fixed delay.
///
/// Read through the process snapshot (`developer_var`): the value is captured
/// at process start like every other `HIPFIRE_*` knob, not consulted live.
/// The parsing lives in [`parse_batch_window`] so it stays unit-testable —
/// the snapshot freezes on first read, so a test cannot drive it through the
/// environment.
fn batch_window_from_env() -> Duration {
    parse_batch_window(
        hipfire_config::developer_var("HIPFIRE_EMBED_BATCH_WINDOW_US")
            .ok()
            .as_deref(),
    )
}

/// Parse one `HIPFIRE_EMBED_BATCH_WINDOW_US` value: microseconds, or the
/// default when unset or unparseable.
fn parse_batch_window(v: Option<&str>) -> Duration {
    match v {
        Some(v) => match v.trim().parse::<u64>() {
            Ok(us) => Duration::from_micros(us),
            Err(_) => BATCH_WINDOW,
        },
        None => BATCH_WINDOW,
    }
}

/// One caller's batch of requests, and where its replies go.
struct Cmd {
    reqs: Vec<EmbedRequest>,
    reply: SyncSender<Result<Vec<Result<EmbedResponse, String>>, String>>,
}

pub struct EmbedSlotEngine {
    tx: Option<Sender<Cmd>>,
    stats: Arc<Mutex<EngineStats>>,
    handle: Option<std::thread::JoinHandle<Result<(), String>>>,
}

impl EmbedSlotEngine {
    /// Load the encoder and start the engine thread. Blocks until the load has
    /// succeeded or failed, so a broken checkpoint is a load error on the
    /// daemon's `load` command rather than a dead engine behind a `loaded` ACK.
    pub fn spawn(cfg: EmbedEngineConfig) -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), String>>(1);
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);

        let handle = std::thread::Builder::new()
            .name("hipfire-embed-engine".to_string())
            .spawn(move || -> Result<(), String> {
                let mut gpu = match Gpu::init() {
                    Ok(g) => g,
                    Err(e) => {
                        let msg = format!("gpu init: {e:?}");
                        let _ = ready_tx.send(Err(msg.clone()));
                        return Err(msg);
                    }
                };
                let src = match ModelSource::from_path(&cfg.model_path.to_string_lossy()) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        return Err(e);
                    }
                };
                let bundle = match load_qwen35_embed_from_source(&mut gpu, src) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        return Err(e);
                    }
                };
                let _ = ready_tx.send(Ok(()));
                let mut bundle = bundle;
                let res = run_loop(&mut gpu, &mut bundle, rx, stats_thread, &cfg);
                bundle.weights.free_gpu(&mut gpu);
                bundle.state.free_gpu(&mut gpu);
                res
            })
            .map_err(|e| format!("spawn embed engine thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx: Some(tx),
                stats,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("embed engine thread died during startup".to_string()),
        }
    }
}

impl SlotEngineHandle for EmbedSlotEngine {
    /// This engine has no generation route: `generate` is refused by the
    /// daemon's `wire_verb` before it ever reaches here, so the generate side
    /// of the trait is a hard error rather than a silent no-op.
    fn submit(&self, _req: SubmitRequest) -> Result<(), String> {
        Err("qwen35-embed: this engine serves embeddings only; there is no generate route".into())
    }

    fn cancel_waiting(&self, _request_tag: u64) {}

    fn close(&self, _session: u64) -> Result<(), String> {
        // No sessions: an encode is complete when it returns.
        Ok(())
    }

    fn reset(&self) -> Result<(), String> {
        // No pooled state to drop — every encode starts from zero recurrence
        // (`encode_embeddings_batch` resets the GDN state itself).
        Ok(())
    }

    /// The KV/pool counters stay zero: they describe state this engine does
    /// not have. `admitted`/`rejected` count requests, which is the useful
    /// signal on this route.
    fn stats(&self) -> EngineStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    fn encode(
        &self,
        reqs: Vec<EmbedRequest>,
    ) -> Result<Vec<Result<EmbedResponse, String>>, String> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }
        let (tx, rx) = sync_channel(1);
        self.tx
            .as_ref()
            .ok_or_else(|| "qwen35-embed: engine is shut down".to_string())?
            .send(Cmd { reqs, reply: tx })
            .map_err(|_| "qwen35-embed: engine is shut down".to_string())?;
        rx.recv()
            .map_err(|_| "qwen35-embed: engine thread exited before replying".to_string())?
    }

    fn shutdown(mut self: Box<Self>) -> Result<(), String> {
        self.tx = None;
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| "qwen35-embed: engine thread panicked".to_string())?,
            None => Ok(()),
        }
    }
}

/// The engine thread: collect, assemble, encode, reply.
fn run_loop(
    gpu: &mut Gpu,
    bundle: &mut EmbedBundle,
    rx: Receiver<Cmd>,
    stats: Arc<Mutex<EngineStats>>,
    cfg: &EmbedEngineConfig,
) -> Result<(), String> {
    while let Ok(first) = rx.recv() {
        let mut cmds = vec![first];
        // `max_batch_requests` bounds REQUESTS per forward, not commands:
        // `encode` accepts a command carrying a whole batch, so counting
        // commands would let one such caller bypass the bound entirely.
        let mut reqs: usize = cmds.iter().map(|c| c.reqs.len()).sum();
        let deadline = Instant::now() + cfg.batch_window;
        while reqs < cfg.max_batch_requests {
            match rx.try_recv() {
                Ok(c) => {
                    reqs += c.reqs.len();
                    cmds.push(c);
                }
                Err(TryRecvError::Empty) => {
                    if cfg.batch_window.is_zero() || Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if let Err(e) = serve_batch(gpu, bundle, cmds, &stats, cfg) {
            // An engine-level failure: every waiter in this group is answered,
            // then the thread ends (the daemon sees the next encode fail).
            return Err(e);
        }
    }
    Ok(())
}

/// Assemble and encode one group, replying per command.
fn serve_batch(
    gpu: &mut Gpu,
    bundle: &mut EmbedBundle,
    cmds: Vec<Cmd>,
    stats: &Arc<Mutex<EngineStats>>,
    cfg: &EmbedEngineConfig,
) -> Result<(), String> {
    // Assemble every request first: it is host-side (plus the vision tower),
    // and it is what tells us the true row count for the split below.
    struct Prepared {
        cmd: usize,
        slot: usize,
        kind: InputKind,
        assembled: crate::carrier::AssembledInput,
    }
    let t_assemble0 = Instant::now();
    let mut prepared: Vec<Prepared> = Vec::new();
    // Per-request failures found before any GPU work, keyed by (cmd, slot).
    let mut early: Vec<Vec<Option<String>>> = Vec::with_capacity(cmds.len());
    for (ci, cmd) in cmds.iter().enumerate() {
        let mut errs = Vec::with_capacity(cmd.reqs.len());
        for (si, req) in cmd.reqs.iter().enumerate() {
            let kind = match req.kind {
                EmbedKind::Query => InputKind::Query,
                EmbedKind::Document => InputKind::Document,
            };
            // Mirror the sequential route's refusal: this encoder is
            // multi-vector with a fixed 128-d head and no Matryoshka
            // support, so a `dims` request must error rather than be
            // silently ignored.
            if let Some(d) = req.dims {
                if d != 0 {
                    errs.push(Some(format!(
                        "{}: `dims` ({d}) is not supported on this encoder — it is                          multi-vector with a fixed {}-d head",
                        req.request_tag, bundle.config.dense_out_features
                    )));
                    continue;
                }
            }
            // One image per request is this encoder's contract (its
            // `assemble_input` splices a single vision block). A longer list
            // is asking for the arch-18 list form, so it is refused rather
            // than silently truncated to the first image.
            if req.images.len() > 1 {
                errs.push(Some(format!(
                    "{}: this encoder takes at most one image per request (got {}); \
                     embeddinggemma2 (arch 18) accepts a list",
                    req.request_tag,
                    req.images.len()
                )));
                continue;
            }
            // No audio tower on this encoder at all: refuse the clips by name
            // rather than letting the engine encode the text and drop them.
            if !req.audios.is_empty() {
                errs.push(Some(format!(
                    "{}: this encoder has no audio tower — {} audio clip(s) cannot be \
                     encoded here; audio applies to embeddinggemma2 (arch 18)",
                    req.request_tag,
                    req.audios.len()
                )));
                continue;
            }
            // Same for video: arch 18's video leg drives ITS vision tower per
            // frame, and this encoder has no such leg.
            if !req.videos.is_empty() {
                errs.push(Some(format!(
                    "{}: this encoder has no video leg — {} video(s) cannot be encoded \
                     here; video applies to embeddinggemma2 (arch 18)",
                    req.request_tag,
                    req.videos.len()
                )));
                continue;
            }
            match assemble_input(
                bundle,
                gpu,
                kind,
                &req.text,
                req.images.first().map(|bytes| bytes.as_slice()),
            ) {
                Ok(assembled) => {
                    errs.push(None);
                    prepared.push(Prepared {
                        cmd: ci,
                        slot: si,
                        kind,
                        assembled,
                    });
                }
                Err(e) => errs.push(Some(format!("{}: {e}", req.request_tag))),
            }
        }
        early.push(errs);
    }
    if prepared.is_empty() {
        reply(cmds, early, stats);
        return Ok(());
    }

    let t_assemble = t_assemble0.elapsed();
    // Split the row block so no single forward exceeds the row budget. A
    // request is never split; one that alone exceeds the budget gets its own
    // forward (the checkpoint's own length limits already bound it).
    let rows: Vec<usize> = prepared
        .iter()
        .map(|p| crate::forward::padded_rows(p.assembled.ids.len()))
        .collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_rows = 0usize;
    for (i, r) in rows.iter().enumerate() {
        if !current.is_empty() && current_rows + r > cfg.max_batch_rows {
            groups.push(std::mem::take(&mut current));
            current_rows = 0;
        }
        current.push(i);
        current_rows += r;
    }
    if !current.is_empty() {
        groups.push(current);
    }

    let dim = bundle.config.dense_out_features;
    let mut results: Vec<Vec<Option<Result<EmbedResponse, String>>>> = early
        .into_iter()
        .map(|errs| errs.into_iter().map(|e| e.map(Err)).collect())
        .collect();

    let trace = hipfire_config::developer_var("HIPFIRE_EMBED_ENGINE_TRACE")
        .ok()
        .as_deref()
        == Some("1");
    if trace {
        eprintln!(
            "[engine] batch: {} cmds -> {} requests -> {} groups {:?} (assemble {:.1} ms)",
            cmds.len(),
            prepared.len(),
            groups.len(),
            groups
                .iter()
                .map(|g| g
                    .iter()
                    .map(|&i| prepared[i].assembled.ids.len())
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            t_assemble.as_secs_f64() * 1e3,
        );
    }
    for group in &groups {
        let inputs: Vec<EncodeInput<'_>> = group
            .iter()
            .map(|&i| EncodeInput {
                ids: &prepared[i].assembled.ids,
                image_embeds: &prepared[i].assembled.image_embeds,
                spans: &prepared[i].assembled.spans,
                pos3: &prepared[i].assembled.pos3,
            })
            .collect();
        let t_g = Instant::now();
        let raw = match encode_embeddings_batch(
            gpu,
            &bundle.weights,
            &mut bundle.state,
            &bundle.config,
            &inputs,
        ) {
            Ok(raw) => raw,
            // A failed group must not take the batch down: earlier groups'
            // results are already staged, and killing the engine would make
            // every caller (in flight AND queued) wait for a reply that can
            // never arrive. Error this group's requests and keep going.
            Err(e) => {
                for &i in group {
                    let p = &prepared[i];
                    results[p.cmd][p.slot] = Some(Err(format!("qwen35-embed: encode: {e}")));
                }
                continue;
            }
        };
        if trace {
            let rows: usize = group
                .iter()
                .map(|&i| crate::forward::padded_rows(prepared[i].assembled.ids.len()))
                .sum();
            eprintln!(
                "[engine]   group of {} ({rows} rows): encode {:.1} ms",
                group.len(),
                t_g.elapsed().as_secs_f64() * 1e3
            );
        }
        for (gi, &i) in group.iter().enumerate() {
            let p = &prepared[i];
            let raw_i = &raw[gi];
            if trace {
                let norm: f32 = raw_i.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!(
                    "[engine]   group slot {gi} (cmd {} req {}): {} tokens, {} raw floats, norm {norm:.4}",
                    p.cmd,
                    p.slot,
                    p.assembled.ids.len(),
                    raw_i.len()
                );
            }
            let out = bundle
                .finalize_embeddings(p.kind, &p.assembled.ids, raw_i)
                .map(|(embeddings, n_tokens)| EmbedResponse {
                    kind: match p.kind {
                        InputKind::Query => EmbedKind::Query,
                        InputKind::Document => EmbedKind::Document,
                    },
                    embeddings,
                    n_tokens,
                    dim,
                    n_input_tokens: p.assembled.ids.len(),
                });
            results[p.cmd][p.slot] = Some(out);
        }
    }

    let mut s = stats.lock().map_err(|_| "stats poisoned".to_string())?;
    for row in &results {
        for r in row {
            match r {
                Some(Ok(_)) => s.note_admitted(),
                Some(Err(_)) => s.note_rejected(),
                None => {}
            }
        }
    }
    drop(s);
    let mut out: Vec<Vec<Result<EmbedResponse, String>>> = Vec::with_capacity(results.len());
    for row in results {
        out.push(
            row.into_iter()
                .map(|r| r.unwrap_or_else(|| Err("qwen35-embed: request was not encoded".into())))
                .collect(),
        );
    }
    finish(cmds, out);
    Ok(())
}

fn reply(cmds: Vec<Cmd>, early: Vec<Vec<Option<String>>>, stats: &Arc<Mutex<EngineStats>>) {
    let out: Vec<Vec<Result<EmbedResponse, String>>> = early
        .into_iter()
        .map(|errs| {
            errs.into_iter()
                .map(|e| Err(e.unwrap_or_else(|| "qwen35-embed: empty request".into())))
                .collect()
        })
        .collect();
    if let Ok(mut s) = stats.lock() {
        for row in &out {
            for _ in row {
                s.note_rejected();
            }
        }
    }
    finish(cmds, out);
}

/// Hand each command its slice of the results and drop the channel (a vanished
/// caller is not an error here — the work is already done).
fn finish(cmds: Vec<Cmd>, out: Vec<Vec<Result<EmbedResponse, String>>>) {
    for (cmd, reply) in cmds.into_iter().zip(out) {
        let _ = cmd.reply.send(Ok(reply));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config() -> SlotEngineConfig {
        SlotEngineConfig {
            model_path: PathBuf::from("/models/embed"),
            n_slots: 6,
            cap_tokens: 8192,
            prefill_chunk: 1024,
            host_budget_bytes: 0,
            swap_dir: PathBuf::new(),
            swap_disk_bytes: 0,
            is_vl: true,
            vl_path: None,
            vision_mode: "off".into(),
            mtp_k: 0,
            mtp_mode: hipfire_config::MtpMode::Off,
            kv_mode_raw: String::new(),
            kv_backend: "legacy".into(),
            prefix_cache: false,
            prefix_cache_max_bytes: 0,
            max_batch_tokens: 3072,
            prefill_min_tokens: 1,
            wait_max_count: 64,
            wait_max_bytes: 1 << 28,
            queue_timeout_ms: 30000,
            structured_jump_forward: false,
            dflash_draft: None,
            dflash_required: false,
        }
    }

    #[test]
    fn the_slot_configs_two_concurrency_knobs_map_onto_requests_and_rows() {
        let cfg = EmbedEngineConfig::from_slot_config(&config());
        assert_eq!(cfg.model_path, PathBuf::from("/models/embed"));
        // `n_slots` is how many requests may share one forward.
        assert_eq!(cfg.max_batch_requests, 6);
        // `max_batch_tokens` bounds the rows in it.
        assert_eq!(cfg.max_batch_rows, 3072);
        // `cap_tokens` (per-slot context) has no counterpart: this model
        // truncates each input at the length it was trained at, inside
        // `assemble_input`, so a second cap could only be wrong.
        assert_ne!(cfg.max_batch_rows, 8192);
        // `batch_window` is the process snapshot's answer and freezes on
        // first read; its parsing is covered by the parser test above.
    }

    #[test]
    fn a_zero_row_budget_falls_back_rather_than_refusing_every_request() {
        let mut raw = config();
        raw.max_batch_tokens = 0;
        let cfg = EmbedEngineConfig::from_slot_config(&raw);
        assert_eq!(cfg.max_batch_rows, DEFAULT_BATCH_ROWS);
        // Slot count 0 is not a servable engine either; one request still runs.
        raw.n_slots = 0;
        assert_eq!(
            EmbedEngineConfig::from_slot_config(&raw).max_batch_requests,
            1
        );
    }

    #[test]
    fn the_batch_window_parses_microseconds_and_falls_back_to_the_default() {
        // The reader freezes with the process snapshot, so the parser — the
        // part with behavior — is tested directly.
        assert!(parse_batch_window(Some("0")).is_zero());
        assert_eq!(
            parse_batch_window(Some("50000")),
            Duration::from_micros(50000)
        );
        assert_eq!(
            parse_batch_window(Some(" 2000 ")),
            Duration::from_micros(2000)
        );
        assert_eq!(parse_batch_window(Some("not-a-number")), BATCH_WINDOW);
        assert_eq!(parse_batch_window(None), BATCH_WINDOW);
    }
}
