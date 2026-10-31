// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Graph-captured Raon DFlash verify forward.
//!
//! The verify block (`K` = 1..=16 rows, one batched forward) is launch-bound
//! when issued eagerly (~2,100 kernel launches per step). This module records
//! that forward once per row count and replays it as one `hipGraphLaunch`.
//!
//! # Recorded region
//!
//! Everything that is pure device work over stable buffers:
//! [`verify_body`] (embedding, 25 layers, the DFlash feature capture copies,
//! the RDA rollback ring when armed, device `seq_len += K`), then
//! [`readout_logits`] of the `K` rows into the bundle's verify head and one
//! [`argmax_f32`] over them. Every kernel reads the token ids, `seq_len` and
//! all state from device memory: latent-cache row positions, mirror rows and
//! the causal bound of the attention kernels come from the device `seq_len`
//! (`LatentCache::pos`), never from the host `host_pos` (validation only), and
//! grids depend on `K` alone, so one recording replays correctly at every
//! later position of the same [`binding`](RaonBundle::verify_binding).
//!
//! # Outside the region (host work, per call)
//!
//! Context admission for `pos + K` (latent mapping, expanded mirror), the
//! `K` token ids (one `hipMemsetD32Async` immediate each, on the graph stream,
//! so no host buffer outlives the call), the recurrent snapshot, and the host
//! `n_tokens` mirror. The partial-accept rollback is not graphed here.
//!
//! # Lifecycle per `(K, with_features)`
//!
//! 1. first use: the caller runs the eager verify (loads every kernel module
//!    and sizes every shared scratch, neither of which may happen inside a
//!    capture), then [`RaonBundle::verify_graph_mark_warm`];
//! 2. next use: record (executes nothing), then launch the fresh graph once;
//! 3. later uses: launch the cached graph.
//!
//! The graphs live in `gpu.graphs.verify` (the generic per-B verify cache, not
//! the AR decode slot): `Gpu::invalidate_graph_state` (shared scratch growth,
//! e.g. a longer prefill's F16 activation staging) destroys them together with
//! the decode graph, and they re-record lazily. A graph is replayed only while
//! the binding it was recorded with still matches (weights, state, scratch,
//! work, MLA route and capacities, verify head, capture buffer, rollback ring,
//! feature flags); a mismatch re-records.
//!
//! # Capture eligibility
//!
//! Every op of the verify forward is capture-safe except `project()` over a
//! `MQ4G256V2` weight with `rows > 1`: `llama::weight_gemm` allocates and frees
//! its rotated-activation scratch per call. Raon's shipped recipe keeps every
//! attention projection and `lm_head` on Q8F16/F32 (qt44 only in the dense and
//! shared-FFN/MoE paths, which use the dispatch crate's allocation-free
//! grouped route), so [`projections_capturable`] only guards a differently
//! quantized artifact: the bundle then verifies eagerly, announced once.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu};

use crate::bundle::RaonBundle;
use crate::forward::{admit_decode_context, readout_logits, rows_binding, verify_body};
use crate::rollback::RollbackMode;
use crate::weights::{RaonMixer, RaonWeights};

/// Bookkeeping of the per-`(K, with_features)` verify graphs of one bundle. The
/// graphs themselves live in `gpu.graphs.verify`.
#[derive(Default)]
pub(crate) struct VerifyGraphs {
    /// Binding each cached graph was recorded with, by graph key.
    bound: HashMap<usize, u64>,
    /// `Some(offender)` once the projection dtypes were found not capturable.
    refused: Option<Option<&'static str>>,
    captures: u64,
    replays: u64,
    launch_secs: f64,
    kernel_nodes: usize,
}

impl VerifyGraphs {
    /// Destroy every graph this bundle recorded (they read buffers the caller
    /// is about to free).
    pub(crate) fn free(self, gpu: &mut Gpu) {
        if self.bound.is_empty() {
            return;
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_synchronize(stream);
        }
        for key in self.bound.keys() {
            gpu.graphs
                .verify_graph_destroy(&gpu.hip, gpu.device_id, *key);
        }
    }
}

/// Key of the verify graph of `rows` rows with or without the feature capture
/// copies (they are different launch sequences).
fn graph_key(rows: usize, with_features: bool) -> usize {
    (rows << 1) | with_features as usize
}

/// Name of the first projection weight (`project()` over `rows > 1`) whose dtype
/// allocates inside the batched GEMM, `None` when every one is capture-safe.
fn projections_capturable(w: &RaonWeights) -> Option<&'static str> {
    let qt44 = |t: &hipfire_runtime::llama::WeightTensor| t.gpu_dtype == DType::MQ4G256V2;
    if qt44(&w.lm_head) {
        return Some("lm_head");
    }
    for layer in &w.layers {
        let named: Vec<(&'static str, &hipfire_runtime::llama::WeightTensor)> = match &layer.mixer {
            RaonMixer::Rda(r) => vec![
                ("rda.q_proj", &r.q_proj),
                ("rda.k_proj", &r.k_proj),
                ("rda.v_proj", &r.v_proj),
                ("rda.g_proj", &r.g_proj),
                ("rda.b_proj", &r.b_proj),
                ("rda.rda_w_proj", &r.rda_w_proj),
                ("rda.f_a_proj", &r.f_a_proj),
                ("rda.f_b_proj", &r.f_b_proj),
                ("rda.o_proj", &r.o_proj),
            ],
            RaonMixer::Mla(m) => vec![
                ("mla.q_a_proj", &m.q_a_proj),
                ("mla.kv_a_proj", &m.kv_a_proj),
                ("mla.g_proj", &m.g_proj),
                ("mla.q_b_proj", &m.q_b_proj),
                ("mla.kv_b_proj", &m.kv_b_proj),
                ("mla.o_proj", &m.o_proj),
            ],
        };
        if let Some((name, _)) = named.into_iter().find(|(_, t)| qt44(t)) {
            return Some(name);
        }
    }
    None
}

impl RaonBundle {
    /// `(captures, replays)` of the verify graphs since load: a replay is a
    /// verify served by a recorded graph (the recording launch counts as a
    /// capture, not a replay).
    pub fn verify_graph_stats(&self) -> (u64, u64) {
        (
            self.spec.verify_graphs.captures,
            self.spec.verify_graphs.replays,
        )
    }

    /// Host seconds spent inside `hipGraphLaunch` for verify graphs.
    pub fn verify_graph_launch_secs(&self) -> f64 {
        self.spec.verify_graphs.launch_secs
    }

    /// Kernel launches of the most recently recorded verify graph (0 = none).
    pub fn verify_graph_kernel_nodes(&self) -> usize {
        self.spec.verify_graphs.kernel_nodes
    }

    /// Identity of every device address and baked extent a verify graph of
    /// `rows` rows reads: the decode binding's weights/state/scratch/work and
    /// MLA route of the context `pos + rows`, plus the row count, the capture
    /// flag and buffers, the verify head, the rollback ring and the feature
    /// flags (kernel routes).
    fn verify_binding(
        &self,
        gpu: &Gpu,
        rows: usize,
        with_features: bool,
        arm: bool,
    ) -> Result<u64, String> {
        let (logits, argmax) = self
            .spec
            .verify_head()
            .ok_or("raon1_mini verify graph: the verify head is not allocated")?;
        let mut extra = vec![
            rows as u64,
            with_features as u64,
            self.config.vocab as u64,
            Arc::as_ptr(&gpu.flags) as usize as u64,
            if arm { self.spec.ring_binding() } else { 0 },
            arm as u64,
            logits.buf.as_ptr() as u64,
            argmax.buf.as_ptr() as u64,
        ];
        if with_features {
            self.spec.capture(0)?.binding_words(&mut extra);
        }
        Ok(rows_binding(
            &self.weights,
            &self.state,
            &self.scratch,
            &self.work,
            rows,
            &extra,
        ))
    }

    /// The recorded region: forward body + feature capture + readout of `rows`
    /// rows into the verify head + argmax. Launches only (no host work).
    fn verify_region(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        with_features: bool,
        arm: bool,
    ) -> Result<(), String> {
        let vocab = self.config.vocab;
        let capture = if with_features {
            Some(self.spec.capture(0)?)
        } else {
            None
        };
        let ring = if arm { self.spec.ring() } else { None };
        verify_body(
            &self.config,
            &self.weights,
            &self.state,
            &self.scratch,
            &self.work,
            gpu,
            rows,
            capture.as_ref(),
            ring.as_ref(),
        )?;
        let (head_logits, head_argmax) = self
            .spec
            .verify_head()
            .ok_or("raon1_mini verify graph: the verify head is not allocated")?;
        let logits = head_logits.sub_offset(0, rows * vocab);
        readout_logits(
            &self.config,
            &self.weights,
            &mut self.scratch,
            &mut self.work,
            gpu,
            0,
            rows,
            &logits,
        )?;
        // The AR greedy token's kernel, one launch over every row.
        let mut indices = head_argmax.sub_offset(0, rows);
        indices.dtype = DType::Raw;
        indices.shape = vec![rows * 4];
        argmax_f32(
            gpu,
            &ArgmaxF32 {
                logits: &logits,
                indices: &indices,
                rows,
                vocab,
            },
        )
        .map_err(|e| format!("raon1_mini verify argmax: {e:?}"))
    }

    /// Stage the `block` token ids into the device id buffer on the graph
    /// stream: one `hipMemsetD32Async` immediate per id (no host buffer to
    /// outlive the call).
    fn stage_block_on_stream(&self, gpu: &Gpu, block: &[u32]) -> Result<(), String> {
        for (i, &token) in block.iter().enumerate() {
            if token as usize >= self.config.vocab {
                return Err(format!(
                    "raon1_mini: token id {token} outside vocab {}",
                    self.config.vocab
                ));
            }
            let cell = self.scratch.token_ids.buf.byte_view(i * 4, 4);
            gpu.write_u32_stream_ordered(&cell, token)
                .map_err(|e| format!("raon1_mini: stage verify token {i} on stream: {e:?}"))?;
        }
        Ok(())
    }

    fn sync_graph_stream(gpu: &Gpu) {
        if let Some(stream) = gpu.active_stream.as_ref() {
            let _ = gpu.hip.stream_synchronize(stream);
        }
    }

    /// Destroy the graph of `key` (stale binding) but keep its warmup: the
    /// modules and scratch sizes it needed are still resident.
    fn drop_verify_graph(&mut self, gpu: &mut Gpu, key: usize) {
        Self::sync_graph_stream(gpu);
        gpu.graphs
            .verify_graph_destroy(&gpu.hip, gpu.device_id, key);
        gpu.graphs.verify_mark_warmup_done(key);
        self.spec.verify_graphs.bound.remove(&key);
    }

    /// Record the verify region of `rows` rows into the verify cache entry
    /// `key`. Recording executes nothing.
    fn record_verify_graph(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        with_features: bool,
        arm: bool,
        key: usize,
        binding: u64,
    ) -> Result<(), String> {
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("verify graph stream ensured by caller");
        gpu.graphs
            .begin_verify_graph_capture(&gpu.hip, gpu.device_id, stream, key)
            .map_err(|e| format!("raon1_mini: begin verify graph capture (K={rows}): {e:?}"))?;
        let body = self.verify_region(gpu, rows, with_features, arm);
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("verify graph stream ensured by caller");
        if let Err(e) = body {
            gpu.graphs
                .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
            gpu.graphs.verify.capturing = None;
            return Err(format!(
                "raon1_mini: verify graph capture failed for K={rows} (no eager fallback): {e}"
            ));
        }
        gpu.graphs
            .end_verify_graph_capture(&gpu.hip, gpu.device_id, stream)
            .map_err(|e| format!("raon1_mini: end verify graph capture (K={rows}): {e:?}"))?;
        let nodes = gpu.graphs.verify.cache.get(&key).map_or(0, |e| e.2.len());
        let g = &mut self.spec.verify_graphs;
        g.bound.insert(key, binding);
        g.kernel_nodes = nodes;
        Ok(())
    }

    /// Run the verify forward of `block` (at `position == n_tokens`) through
    /// its recorded graph: logits of every row and the per-row argmax land in
    /// the verify head, the feature rows in the capture buffer when
    /// `with_features`, the device `seq_len` and the host mirror advance by
    /// `block.len()`.
    ///
    /// `Ok(false)` means nothing ran and the caller must run the eager verify
    /// (first use of this `(K, with_features)`, or an artifact whose projections
    /// are not capturable); it then calls
    /// [`verify_graph_mark_warm`](Self::verify_graph_mark_warm).
    pub(crate) fn verify_graph_try(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        with_features: bool,
    ) -> Result<bool, String> {
        let rows = block.len();
        let refused = match self.spec.verify_graphs.refused {
            Some(r) => r,
            None => {
                let r = projections_capturable(&self.weights);
                if let Some(name) = r {
                    eprintln!(
                        "raon1_mini: verify stays eager: projection `{name}` is qt44 (the batched \
                         GEMM allocates per call, not graph-capturable)"
                    );
                }
                self.spec.verify_graphs.refused = Some(r);
                r
            }
        };
        if refused.is_some() {
            return Ok(false);
        }
        let key = graph_key(rows, with_features);
        // Context admission may map latent rows / back the mirror (changes the
        // binding) and is host work: never inside the recorded region.
        admit_decode_context(
            &self.config,
            &self.weights,
            &mut self.state,
            &self.scratch,
            &self.work,
            gpu,
            position + rows,
        )?;
        let arm = self.dflash_rollback_mode()? == RollbackMode::Recurrence;
        if arm && self.spec.ring().is_none() {
            return Err("raon1_mini verify: the RDA rollback ring is not allocated \
                        (new_spec_scratch not called)"
                .into());
        }
        let binding = self.verify_binding(gpu, rows, with_features, arm)?;
        let have = gpu.graphs.verify_has_graph(key);
        let fresh = have && self.spec.verify_graphs.bound.get(&key) == Some(&binding);
        if !fresh {
            if have {
                self.drop_verify_graph(gpu, key);
            }
            if gpu.graphs.verify_needs_warmup(key) {
                return Ok(false);
            }
        }
        self.spec.clear_ring_ticket();
        if gpu.active_stream.is_none() {
            let stream = gpu
                .hip
                .stream_create()
                .map_err(|e| format!("raon1_mini: verify graph stream: {e:?}"))?;
            gpu.active_stream = Some(stream);
        }
        self.stage_block_on_stream(gpu, block)?;
        if fresh {
            self.spec.verify_graphs.replays += 1;
        } else {
            self.record_verify_graph(gpu, rows, with_features, arm, key, binding)?;
            self.spec.verify_graphs.captures += 1;
        }
        let stream = gpu
            .active_stream
            .as_ref()
            .expect("verify graph stream just ensured");
        let t_launch = Instant::now();
        gpu.graphs
            .verify_graph_launch(&gpu.hip, gpu.device_id, stream, key)
            .map_err(|e| format!("raon1_mini: verify graph launch (K={rows}): {e:?}"))?;
        self.spec.verify_graphs.launch_secs += t_launch.elapsed().as_secs_f64();
        // The graph advanced the device `seq_len`; mirror it on the host.
        self.state.note_host_tokens(rows);
        if arm {
            self.spec.mark_ring_recorded(position, rows);
        }
        Ok(true)
    }

    /// The eager verify of `rows` rows just ran: every kernel module and shared
    /// scratch it needs is resident, so the next use may record.
    pub(crate) fn verify_graph_mark_warm(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
        with_features: bool,
    ) {
        gpu.graphs
            .verify_mark_warmup_done(graph_key(rows, with_features));
    }
}
