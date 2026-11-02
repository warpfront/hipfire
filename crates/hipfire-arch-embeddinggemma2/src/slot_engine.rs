// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The arch-18 multi-slot engine: one GPU thread that owns the
//! EmbeddingGemma-2 encoder and serves concurrent embed requests as **one
//! batched forward** (the qwen35-embed slot-engine shape: request channel,
//! collector with a 2 ms batch window, row-budgeted groups, one
//! `encode_embeddings_batch` per group, per-request failure isolation).

use crate::bundle::EmbedGemmaBundle;
use crate::carrier::{
    assemble_input, encode_media, load_embed_gemma_with_vision, media_rows,
    resolve_slot_vision_path, AssembledInput, NO_AUDIO_TOWER, NO_VIDEO_TOWER, NO_VISION_TOWER,
};
use crate::config::InputKind;
use crate::forward::{encode_embeddings_batch, padded_rows, EncodeInput, MediaRows};
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::serve::{
    EmbedKind, EmbedRequest, EmbedResponse, EngineStats, SlotEngineConfig, SlotEngineHandle,
    SubmitRequest,
};
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const BATCH_WINDOW: Duration = Duration::from_micros(2000);

/// Row budget for one forward when the caller configured none. The
/// checkpoint's own kind limits (8 192 both, by default) already cap a
/// single request; the eager attention's score matrix is `rows²`, so this
/// bound also bounds scratch VRAM.
const DEFAULT_BATCH_ROWS: usize = 8192;

#[derive(Clone, Debug)]
pub struct EmbedEngineConfig {
    pub model_path: std::path::PathBuf,
    pub max_batch_requests: usize,
    pub max_batch_rows: usize,
    pub batch_window: Duration,
    /// Resolved `.vl` sidecar (`None` = text-only). It carries BOTH towers
    /// (vision and audio). Resolved from the slot preflight's `vl_path`, or
    /// by [`resolve_slot_vision_path`] when the caller passed none (an
    /// explicit `HIPFIRE_VISION_MODE=off` resolves nothing).
    pub vision_path: Option<PathBuf>,
}

impl EmbedEngineConfig {
    pub fn from_slot_config(cfg: &SlotEngineConfig) -> Self {
        // The daemon's carrier preflight already applied the load's vision
        // ladder: `off` clears both fields, `auto`/`on` fill `vl_path` when a
        // tower resolved. Honor that decision instead of rediscovering — an
        // unconditional probe here is what let the tower load under an
        // explicit `off`.
        let vision_path = if cfg.vl_path.is_some() || cfg.is_vl {
            resolve_slot_vision_path(&cfg.model_path, cfg.vl_path.as_deref())
        } else {
            None
        };
        Self {
            model_path: cfg.model_path.clone(),
            max_batch_requests: cfg.n_slots.max(1),
            max_batch_rows: if cfg.max_batch_tokens == 0 {
                DEFAULT_BATCH_ROWS
            } else {
                cfg.max_batch_tokens
            },
            batch_window: batch_window_from_env(),
            vision_path,
        }
    }
}

fn batch_window_from_env() -> Duration {
    match hipfire_config::developer_var("HIPFIRE_EMBED_BATCH_WINDOW_US") {
        Ok(v) => match v.trim().parse::<u64>() {
            Ok(us) => Duration::from_micros(us),
            Err(_) => BATCH_WINDOW,
        },
        Err(_) => BATCH_WINDOW,
    }
}

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
    /// Load the encoder and start the engine thread. Blocks until the load
    /// has succeeded or failed (a broken checkpoint is a load error, not a
    /// dead engine).
    pub fn spawn(cfg: EmbedEngineConfig) -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel::<Cmd>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), String>>(1);
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);

        let handle = std::thread::Builder::new()
            .name("hipfire-eg2-embed-engine".to_string())
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
                let bundle =
                    match load_embed_gemma_with_vision(&mut gpu, src, cfg.vision_path.as_deref()) {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e.clone()));
                            return Err(e);
                        }
                    };
                let _ = ready_tx.send(Ok(()));
                let mut bundle = bundle;
                let res = run_loop(&mut gpu, &mut bundle, rx, stats_thread, &cfg);
                if let Some(vision) = bundle.vision.take() {
                    vision.free_gpu(&mut gpu);
                }
                if let Some(audio) = bundle.audio.take() {
                    audio.free_gpu(&mut gpu);
                }
                bundle.weights.free_gpu(&mut gpu);
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
    fn submit(&self, _req: SubmitRequest) -> Result<(), String> {
        Err(
            "embeddinggemma2: this engine serves embeddings only; there is no generate route"
                .into(),
        )
    }

    fn cancel_waiting(&self, _request_tag: u64) {}

    fn close(&self, _session: u64) -> Result<(), String> {
        Ok(())
    }

    fn reset(&self) -> Result<(), String> {
        Ok(())
    }

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
            .ok_or_else(|| "embeddinggemma2: engine is shut down".to_string())?
            .send(Cmd { reqs, reply: tx })
            .map_err(|_| "embeddinggemma2: engine is shut down".to_string())?;
        rx.recv()
            .map_err(|_| "embeddinggemma2: engine thread exited before replying".to_string())?
    }

    fn shutdown(mut self: Box<Self>) -> Result<(), String> {
        self.tx = None;
        match self.handle.take() {
            Some(h) => h
                .join()
                .map_err(|_| "embeddinggemma2: engine thread panicked".to_string())?,
            None => Ok(()),
        }
    }
}

fn run_loop(
    gpu: &mut Gpu,
    bundle: &mut EmbedGemmaBundle,
    rx: Receiver<Cmd>,
    stats: Arc<Mutex<EngineStats>>,
    cfg: &EmbedEngineConfig,
) -> Result<(), String> {
    while let Ok(first) = rx.recv() {
        let mut cmds = vec![first];
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
            return Err(e);
        }
    }
    Ok(())
}

fn serve_batch(
    gpu: &mut Gpu,
    bundle: &mut EmbedGemmaBundle,
    cmds: Vec<Cmd>,
    stats: &Arc<Mutex<EngineStats>>,
    cfg: &EmbedEngineConfig,
) -> Result<(), String> {
    struct Prepared {
        cmd: usize,
        slot: usize,
        kind: InputKind,
        dims: usize,
        n_input: usize,
        assembled: AssembledInput,
        /// The image towers' projected rows, one `[span.len, hidden]` block
        /// per span (empty for text-only inputs).
        media: Vec<Vec<f32>>,
    }
    let mut prepared: Vec<Prepared> = Vec::new();
    let mut early: Vec<Vec<Option<String>>> = Vec::with_capacity(cmds.len());
    for (ci, cmd) in cmds.iter().enumerate() {
        let mut errs = Vec::with_capacity(cmd.reqs.len());
        for req in cmd.reqs.iter() {
            let kind = match req.kind {
                EmbedKind::Query => InputKind::Query,
                EmbedKind::Document => InputKind::Document,
            };
            // The towers are a separate artifact (the `.vl` sidecar), so a
            // tower-less instance refuses each modality with its load-time
            // pointing error instead of silently embedding the text.
            if !req.images.is_empty() && bundle.vision.is_none() {
                errs.push(Some(format!("{}: {NO_VISION_TOWER}", req.request_tag)));
                continue;
            }
            if !req.audios.is_empty() && bundle.audio.is_none() {
                errs.push(Some(format!("{}: {NO_AUDIO_TOWER}", req.request_tag)));
                continue;
            }
            if !req.videos.is_empty() && bundle.vision.is_none() {
                errs.push(Some(format!("{}: {NO_VIDEO_TOWER}", req.request_tag)));
                continue;
            }
            // Truncate a nonsense dims ask here so it fails per-request,
            // before any GPU work, without failing its neighbours.
            let dims = req.dims.unwrap_or(0);
            let dims_ok = dims == 0 || dims <= bundle.config.embedding_dim;
            if !dims_ok {
                errs.push(Some(format!(
                    "{}: dims {dims} exceeds this model's embedding width {}",
                    req.request_tag, bundle.config.embedding_dim
                )));
                continue;
            }
            match assemble_input(
                bundle,
                kind,
                &req.text,
                &req.images,
                &req.audios,
                &req.videos,
            ) {
                Ok(assembled) => {
                    // The tower runs NOW, on the engine thread, before the
                    // batched text encode — never cached across requests.
                    match encode_media(gpu, bundle, &assembled) {
                        Ok(media) => {
                            errs.push(None);
                            prepared.push(Prepared {
                                cmd: ci,
                                slot: errs.len() - 1,
                                kind,
                                dims,
                                n_input: assembled.ids.len(),
                                assembled,
                                media,
                            });
                        }
                        Err(e) => errs.push(Some(format!("{}: {e}", req.request_tag))),
                    }
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

    // Row-budgeted groups (never splitting a request).
    let rows: Vec<usize> = prepared
        .iter()
        .map(|p| padded_rows(p.assembled.ids.len()))
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

    let mut results: Vec<Vec<Option<Result<EmbedResponse, String>>>> = early
        .into_iter()
        .map(|errs| errs.into_iter().map(|e| e.map(Err)).collect())
        .collect();

    let trace = hipfire_config::developer_var("HIPFIRE_EMBED_ENGINE_TRACE")
        .ok()
        .as_deref()
        == Some("1");
    for group in &groups {
        // The tower rows live in `prepared` (they outlive the group), so the
        // per-span views borrow from there.
        let mut media_views: Vec<Vec<MediaRows<'_>>> = Vec::with_capacity(group.len());
        let mut view_err: Option<String> = None;
        for &i in group {
            let p = &prepared[i];
            match media_rows(&p.assembled, &p.media) {
                Ok(v) => media_views.push(v),
                Err(e) => {
                    view_err = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = view_err {
            for &i in group {
                let p = &prepared[i];
                results[p.cmd][p.slot] = Some(Err(format!("embeddinggemma2: encode: {e}")));
            }
            continue;
        }
        let inputs: Vec<EncodeInput<'_>> = group
            .iter()
            .enumerate()
            .map(|(gi, &i)| EncodeInput {
                ids: &prepared[i].assembled.ids,
                media: &media_views[gi],
            })
            .collect();
        let t_g = Instant::now();
        let raw = match encode_embeddings_batch(gpu, &bundle.weights, &bundle.config, &inputs) {
            Ok(raw) => raw,
            Err(e) => {
                for &i in group {
                    let p = &prepared[i];
                    results[p.cmd][p.slot] = Some(Err(format!("embeddinggemma2: encode: {e}")));
                }
                continue;
            }
        };
        if trace {
            eprintln!(
                "[eg2-engine]   group of {} ({} rows): encode {:.1} ms",
                group.len(),
                group.iter().map(|&i| rows[i]).sum::<usize>(),
                t_g.elapsed().as_secs_f64() * 1e3
            );
        }
        for (gi, &i) in group.iter().enumerate() {
            let p = &prepared[i];
            let out = bundle
                .finalize_embedding(&raw[gi], p.n_input, p.dims)
                .map(|embeddings| EmbedResponse {
                    kind: match p.kind {
                        InputKind::Query => EmbedKind::Query,
                        InputKind::Document => EmbedKind::Document,
                    },
                    embeddings,
                    n_tokens: 1, // single-vector: one pooled embedding
                    dim: if p.dims == 0 {
                        bundle.config.embedding_dim
                    } else {
                        p.dims
                    },
                    n_input_tokens: p.n_input,
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
                .map(|r| {
                    r.unwrap_or_else(|| Err("embeddinggemma2: request was not encoded".into()))
                })
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
                .map(|e| Err(e.unwrap_or_else(|| "embeddinggemma2: empty request".into())))
                .collect()
        })
        .collect();
    if let Ok(mut s) = stats.lock() {
        for row in &out {
            for r in row {
                if r.is_err() {
                    s.note_rejected();
                }
            }
        }
    }
    finish(cmds, out);
}

fn finish(cmds: Vec<Cmd>, out: Vec<Vec<Result<EmbedResponse, String>>>) {
    for (cmd, results) in cmds.into_iter().zip(out) {
        let _ = cmd.reply.send(Ok(results));
    }
}
