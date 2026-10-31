// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini target side of the shared DFlash2 drafter
//! (`hipfire_runtime::dflash2_spec`): `impl SpecTarget for RaonBundle`.
//!
//! # Captured representation
//!
//! The drafter's feature of a position is the POST-FFN mHC carrier of every
//! extract layer, all four streams, stream-major, layer-major: for the layers
//! `[l0, l1, ..]` of `set_dflash_extract_layers` a row is
//! `[l0.s0 | l0.s1 | l0.s2 | l0.s3 | l1.s0 | ..]`, each stream `hidden` F32
//! wide (`layers x 4 x hidden` per row; `[1,6,11,16,21]` x 4 x 2560 = 51200 for
//! KRAFTON's draft). It is the raw, un-normalized residual after the layer's
//! FFN `hc_post` (the input of layer `l + 1`), exactly the tensor KRAFTON's
//! vLLM fork appends as `s.flatten(1)` ("tap-major, stream-major") and their
//! llama.cpp port exposes as `t_layer_inp` (see [`crate::forward::DflashCapture`]
//! and `raon/dflash2/PLAN.md` §2.1). The verify forward captures it on the
//! device with one strided copy per layer into a buffer of verify-block height.
//! Prefill and forced advances (`spec_advance`) run the AR prefill launch
//! sequence and download only the rows the drafter keeps, layer by layer,
//! through a stream tap: no device buffer of the prefill's chunk height (it
//! would cost `chunk x 200 KB`, 0.4-0.8 GB at 2048-4096-row chunks).
//!
//! # Verify and rollback
//!
//! `verify_block[_logits]` snapshots the recurrent block ([`RaonState::snapshot_into`]:
//! RDA matrices, q/k/v conv history, device `seq_len`), then runs ONE batched
//! `forward_rows` over the block (arming the RDA recurrence-input ring,
//! [`crate::rollback`]), reading every row's logits with [`readout_logits`]. A
//! full accept leaves the state as is. A partial accept ([`SpecTarget::commit_prefix`])
//! restores the snapshot and then, by [`RollbackMode`]: `recurrence` (default)
//! reruns only the RDA recurrence over the kept rows from the ring and rebuilds
//! the conv histories ([`RaonBundle::rollback_rows`]); `replay` (kill switch)
//! replays `block[..accept_len + 1]` through the same batched forward. MLA
//! latent rows past the restored length are dead and rewritten later; the
//! expanded mirror's valid-row watermark is clamped by [`RaonState::restore`].
//!
//! # Graph route
//!
//! While [`RaonBundle::decode_mode`] is `graph` (`HIPFIRE_GRAPH=0` selects eager)
//! the verify forward + feature capture + readout + argmax of a block of K rows
//! is one recorded hipGraph per `(K, capture)` ([`crate::verify_graph`]): the
//! first use of a K runs the eager path, the next records, later ones replay.
//!
//! [`RaonState::snapshot_into`]: crate::state::RaonState::snapshot_into
//! [`RaonState::restore`]: crate::state::RaonState::restore

use std::any::Any;

use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::dflash2_spec::trim_feature_rows;
use hipfire_runtime::spec::{SpecAdvance, SpecScratch, SpecTarget};
use rdna_compute::tensor_ops::{argmax_f32, argmax_f32_host_stream_ordered, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::bundle::RaonBundle;
use crate::forward::{
    embed_lookup, forward_rows_tapped, readout_logits, stage_decode_token, validate_extract_layers,
    DflashCapture, TapSite,
};
use crate::projection::{project, OutRound};
use crate::rollback::{RdaRingBuf, RollbackMode};
use crate::state::RaonStateSnapshot;
use crate::verify_graph::VerifyGraphs;
use hipfire_dispatch::pipeline::PrefillActTier;

// ───────────────────────── host planning ─────────────────────────

/// First row of an advance over `rows` positions that the drafter can still
/// read: the target keeps only the last `window` feature rows, so earlier rows
/// are never downloaded (`window = None` keeps every row).
fn keep_from(rows: usize, window: Option<usize>) -> usize {
    window.map_or(0, |w| rows.saturating_sub(w))
}

/// Rows of the chunk `[off, off + len)` whose features are kept, as
/// `(first row within the chunk, row count)`; `None` when the whole chunk lies
/// before `keep_from`.
fn chunk_span(off: usize, len: usize, keep_from: usize) -> Option<(usize, usize)> {
    let end = off + len;
    (end > keep_from).then(|| {
        let skip = keep_from.saturating_sub(off);
        (skip, len - skip)
    })
}

// ───────────────────────── bundle-owned spec state ─────────────────────────

/// Per-row verify head: logits and device argmax of up to `rows` positions.
struct VerifyHead {
    rows: usize,
    /// F32 `[rows, vocab]` (BF16-valued, [`readout_logits`] / the draft head).
    logits: GpuTensor,
    /// F32 `[rows]` holding the argmax indices as i32 bit patterns.
    argmax: GpuTensor,
}

/// Speculative-decode resources of a [`RaonBundle`]: the DFlash2 extract
/// layers, the sink window, the device capture buffer and the per-row verify
/// head. Empty (and inert) until a speculator arms it; AR prefill/decode never
/// touch it.
#[derive(Default)]
pub(crate) struct RaonSpecState {
    /// Zero-based extract layers; empty = DFlash capture disarmed.
    layers: Vec<usize>,
    /// Feature rows the drafter reads (`None` keeps every row).
    window: Option<usize>,
    /// F32 `[rows, layers.len() * streams * hidden]` capture destination.
    capture: Option<GpuTensor>,
    head: Option<VerifyHead>,
    /// Recurrence-input ring of the armed verify ([`crate::rollback`]); sized by
    /// `new_spec_scratch`.
    pub(crate) ring: Option<RdaRingBuf>,
    /// Per-bundle override of the rollback mode (`None` = process default).
    pub(crate) rollback: Option<RollbackMode>,
    /// Recorded verify graphs (one per row count and capture flag).
    pub(crate) verify_graphs: VerifyGraphs,
}

impl RaonSpecState {
    pub(crate) fn set_layers(&mut self, layers: Vec<usize>) {
        self.layers = layers;
    }

    pub(crate) fn layers(&self) -> Option<&[usize]> {
        (!self.layers.is_empty()).then_some(self.layers.as_slice())
    }

    pub(crate) fn set_window(&mut self, rows: usize) {
        self.window = (rows > 0).then_some(rows);
    }

    /// The forward-side descriptor landing a call's rows at capture row `row0`.
    pub(crate) fn capture(&self, row0: usize) -> Result<DflashCapture<'_>, String> {
        let dst = self.capture.as_ref().ok_or_else(|| {
            "raon1_mini: DFlash capture buffer is not allocated (no extract layers armed)"
                .to_string()
        })?;
        Ok(DflashCapture {
            layers: &self.layers,
            dst,
            row0,
        })
    }

    /// The verify head's `(logits, argmax)` buffers.
    pub(crate) fn verify_head(&self) -> Option<(&GpuTensor, &GpuTensor)> {
        self.head.as_ref().map(|h| (&h.logits, &h.argmax))
    }

    pub(crate) fn free(self, gpu: &mut Gpu) {
        // Recorded graphs read the buffers freed below.
        self.verify_graphs.free(gpu);
        for t in self
            .capture
            .into_iter()
            .chain(self.head.into_iter().flat_map(|h| [h.logits, h.argmax]))
            .chain(self.ring.into_iter().flat_map(|r| r.layers))
        {
            if let Err(e) = gpu.free_tensor(t) {
                eprintln!("raon1_mini: spec buffer free: {e:?}");
            }
        }
    }
}

/// Erased verify scratch of a Raon target: the pre-verify recurrent snapshot
/// and the ticket naming the verify it belongs to. The per-row logits, argmax
/// and feature capture buffers belong to the bundle ([`RaonSpecState`]).
pub struct RaonSpecScratch {
    block_size: usize,
    snapshot: RaonStateSnapshot,
    /// `(position, rows)` of the verify whose snapshot is live.
    ticket: Option<(usize, usize)>,
}

impl SpecScratch for RaonSpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        if let Err(e) = self.snapshot.free_gpu(gpu) {
            eprintln!("raon1_mini: spec snapshot free: {e}");
        }
    }

    fn clear(&mut self) {
        self.ticket = None;
    }
}

fn spec_scratch<'a>(
    scratch: &'a mut dyn SpecScratch,
    what: &str,
) -> Result<&'a mut RaonSpecScratch, String> {
    scratch
        .as_any_mut()
        .downcast_mut::<RaonSpecScratch>()
        .ok_or_else(|| format!("raon1_mini {what}: scratch is not RaonSpecScratch"))
}

/// Append rows `[from, to)` of the capture buffer (`row_width` F32 each) to `out`.
fn append_capture_rows(
    gpu: &Gpu,
    capture: &GpuTensor,
    row_width: usize,
    from: usize,
    to: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    // The capture copies may sit on the decode stream: drain every queue first.
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("raon1_mini: synchronize before feature download: {e:?}"))?;
    let elems = (to - from) * row_width;
    let start = out.len();
    out.resize(start + elems, 0.0);
    // SAFETY: `out[start..]` holds exactly `elems` f32s; any byte pattern is a valid f32.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(out[start..].as_mut_ptr() as *mut u8, elems * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &capture.sub_offset(from * row_width, elems).buf)
        .map_err(|e| format!("raon1_mini: download DFlash feature rows: {e:?}"))?;
    gpu.take_device_wait_error()
        .map_err(|e| format!("raon1_mini: download DFlash feature rows: {e:?}"))
}

// ───────────────────────── bundle helpers ─────────────────────────

impl RaonBundle {
    /// F32 width of one layer's stream rows: `streams * hidden`.
    fn stream_width(&self) -> usize {
        self.config.mhc_streams * self.config.hidden
    }

    /// F32 width of one feature row: `layers * streams * hidden` (0 disarmed).
    fn feature_row_width(&self) -> usize {
        self.spec.layers.len() * self.stream_width()
    }

    /// Size the per-row verify head for `rows` positions.
    fn ensure_head_buffers(&mut self, gpu: &mut Gpu, rows: usize) -> Result<(), String> {
        if self.spec.head.as_ref().is_some_and(|h| h.rows >= rows) {
            return Ok(());
        }
        let vocab = self.config.vocab;
        let logits = gpu
            .zeros(&[rows * vocab], DType::F32)
            .map_err(|e| format!("raon1_mini: alloc verify logits: {e:?}"))?;
        let argmax = match gpu.zeros(&[rows], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(logits);
                return Err(format!("raon1_mini: alloc verify argmax: {e:?}"));
            }
        };
        if let Some(old) = self.spec.head.replace(VerifyHead {
            rows,
            logits,
            argmax,
        }) {
            for t in [old.logits, old.argmax] {
                let _ = gpu.free_tensor(t);
            }
        }
        Ok(())
    }

    /// Validate the armed extract layers and size the capture buffer for a
    /// verify block of `rows` rows (prefill never uses it). No-op while
    /// disarmed.
    fn ensure_capture(&mut self, gpu: &mut Gpu, rows: usize) -> Result<(), String> {
        if self.spec.layers.is_empty() {
            return Ok(());
        }
        validate_extract_layers(&self.spec.layers, self.config.n_layers)?;
        let elems = rows * self.feature_row_width();
        if self
            .spec
            .capture
            .as_ref()
            .is_some_and(|c| c.numel() >= elems)
        {
            return Ok(());
        }
        let capture = gpu
            .zeros(&[elems], DType::F32)
            .map_err(|e| format!("raon1_mini: alloc DFlash capture ({elems} F32): {e:?}"))?;
        if let Some(old) = self.spec.capture.replace(capture) {
            let _ = gpu.free_tensor(old);
        }
        Ok(())
    }

    /// Snapshot, one batched forward over `block` with feature capture (when
    /// `hidden_out` is wanted and DFlash is armed) and the per-row logits of
    /// every row into the bundle's verify head. Arms the rollback ticket right
    /// after the snapshot: an error later leaves it armed for `abandon_verify`.
    fn verify_forward(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<(usize, bool), String> {
        let ss = spec_scratch(scratch, "verify")?;
        let rows = block.len();
        if rows == 0 || rows > ss.block_size {
            return Err(format!(
                "raon1_mini verify: block of {rows} rows outside 1..={}",
                ss.block_size
            ));
        }
        if ss.ticket.is_some() {
            return Err(
                "raon1_mini verify: the previous window's rollback ticket is still armed"
                    .to_string(),
            );
        }
        if position != self.state.n_tokens() {
            return Err(format!(
                "raon1_mini verify: position {position} does not match the slot's consumed tokens {}",
                self.state.n_tokens()
            ));
        }
        if position + rows > self.state.max_seq() {
            return Err(format!(
                "raon1_mini verify: {position} + {rows} rows exceed the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        self.ensure_head_buffers(gpu, rows)?;
        let with_features = hidden_out.is_some() && !self.spec.layers.is_empty();
        if with_features {
            self.ensure_capture(gpu, rows)?;
        }
        self.state.snapshot_into(gpu, &mut ss.snapshot)?;
        ss.ticket = Some((position, rows));
        // Graph route: one recorded replay of forward + capture + readout +
        // argmax. The first use of a row count (and a non-capturable artifact)
        // runs the eager forward below instead.
        let graph_mode = self.decode_mode() == "graph";
        let graphed = graph_mode && self.verify_graph_try(gpu, block, position, with_features)?;
        if !graphed {
            self.forward_verify_chunk(gpu, block, position, with_features.then_some(0))?;
            let head = self.spec.head.as_ref().expect("verify head just ensured");
            let logits = head.logits.sub_offset(0, rows * self.config.vocab);
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
            if graph_mode {
                self.verify_graph_mark_warm(gpu, rows, with_features);
            }
        }
        if let (true, Some(out)) = (with_features, hidden_out) {
            let capture = self.spec.capture.as_ref().expect("capture just ensured");
            append_capture_rows(gpu, capture, self.feature_row_width(), 0, rows, out)?;
        }
        Ok((rows, graphed))
    }

    /// One prefill step over `piece` that also appends the feature rows
    /// `skip..skip + count` of the step to `out` (capture layout: layer-major
    /// in `set_dflash_extract_layers` order, then stream-major, then hidden).
    /// The launch sequence is the AR prefill step's ([`Self::forward_chunk`]
    /// without capture), observed by a stream tap: after the FFN site of each
    /// extract layer the device is drained and that layer's kept rows (the
    /// `TapSite::Ffn(l)` buffer) are downloaded. `want_logits` as in
    /// `forward_chunk`. On error `out` is left as it was.
    fn forward_chunk_features(
        &mut self,
        gpu: &mut Gpu,
        piece: &[u32],
        want_logits: bool,
        skip: usize,
        count: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let layers = self.spec.layers.clone();
        validate_extract_layers(&layers, self.config.n_layers)?;
        if skip + count > piece.len() {
            return Err(format!(
                "raon1_mini: feature rows {skip}..{} outside a {}-row step",
                skip + count,
                piece.len()
            ));
        }
        let width = self.stream_width();
        let row_width = self.feature_row_width();
        let base = out.len();
        out.resize(base + count * row_width, 0.0);
        let dst = &mut out[base..];
        let pos = self.state.n_tokens() as u32;
        let mut tap = |gpu: &mut Gpu,
                       site: TapSite,
                       streams: &GpuTensor,
                       _rows: usize|
         -> Result<(), String> {
            let TapSite::Ffn(l) = site else {
                return Ok(());
            };
            let Some(k) = layers.iter().position(|&id| id == l) else {
                return Ok(());
            };
            gpu.hip
                .device_synchronize()
                .map_err(|e| format!("raon1_mini: feature tap synchronize L{l}: {e:?}"))?;
            let host = gpu
                .download_f32(&streams.sub_offset(skip * width, count * width))
                .map_err(|e| format!("raon1_mini: feature tap download L{l}: {e:?}"))?;
            for (r, row) in host.chunks_exact(width).enumerate() {
                let at = r * row_width + k * width;
                dst[at..at + width].copy_from_slice(row);
            }
            Ok(())
        };
        let result = forward_rows_tapped(
            &self.config,
            &self.weights,
            &mut self.state,
            &mut self.scratch,
            &mut self.work,
            gpu,
            piece,
            pos,
            want_logits,
            Some(&mut tap),
        );
        if result.is_err() {
            out.truncate(base);
        }
        result
    }

    /// The target's feature rows for `tokens` through the download tap
    /// ([`forward_rows_tapped`]): the byte-exact reference of the verify's
    /// device capture. Advances the live state over `tokens` from its current
    /// length and returns `tokens.len()` feature rows in the capture layout
    /// (layer-major in `set_dflash_extract_layers` order, then stream-major,
    /// then hidden; `layers x 4 x hidden` F32 each), each layer's rows being the
    /// `TapSite::Ffn(l)` buffer. Needs armed extract layers.
    pub fn dflash_tap_reference(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(tokens.len() * self.feature_row_width());
        for piece in tokens.chunks(self.prefill_step_rows()) {
            self.forward_chunk_features(gpu, piece, false, 0, piece.len(), &mut out)?;
        }
        Ok(out)
    }
}

// ───────────────────────── SpecTarget ─────────────────────────

impl SpecTarget for RaonBundle {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)
            .map_err(|e| format!("raon1_mini reset_recurrent: {e}"))
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(2);
        let cap = self.scratch.rows.min(self.work.rows());
        if block_size > cap {
            return Err(format!(
                "raon1_mini: speculation block {block_size} exceeds the {cap} forward rows"
            ));
        }
        self.ensure_head_buffers(gpu, block_size)?;
        self.ensure_capture(gpu, block_size)?;
        self.dflash_rollback_mode()?;
        self.ensure_rda_ring(gpu, block_size)?;
        let snapshot = self.state.snapshot_alloc(gpu)?;
        Ok(Box::new(RaonSpecScratch {
            block_size,
            snapshot,
            ticket: None,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset(gpu)
                .map_err(|e| format!("raon1_mini spec_advance reset: {e}"))?;
        }
        if tokens.is_empty() {
            return Err("raon1_mini spec_advance cannot process an empty token slice".into());
        }
        if start_pos != self.state.n_tokens() {
            return Err(format!(
                "raon1_mini spec_advance: start position {start_pos} does not match the slot's \
                 consumed tokens {}",
                self.state.n_tokens()
            ));
        }
        if start_pos + tokens.len() > self.state.max_seq() {
            return Err(format!(
                "raon1_mini spec_advance: {start_pos} + {} tokens exceed the reserved context of {} rows",
                tokens.len(),
                self.state.max_seq()
            ));
        }
        let mut hidden_out = hidden_out.filter(|_| !self.spec.layers.is_empty());
        let row_width = self.feature_row_width();
        let chunk = self.prefill_step_rows();
        let first_kept = if hidden_out.is_some() {
            let first = keep_from(tokens.len(), self.spec.window);
            if first > 0 {
                // Rows below the kept tail are never downloaded, so the rows
                // already in the sink would no longer be contiguous with it.
                if let Some(out) = hidden_out.as_deref_mut() {
                    out.clear();
                }
            }
            first
        } else {
            0
        };
        let n_chunks = tokens.len().div_ceil(chunk);
        let mut off = 0usize;
        for (ci, piece) in tokens.chunks(chunk).enumerate() {
            if abort() {
                let _ = self.reset(gpu);
                if let Some(out) = hidden_out.as_deref_mut() {
                    out.clear();
                }
                return Ok(SpecAdvance::Aborted);
            }
            let last = ci + 1 == n_chunks;
            let span = hidden_out
                .as_ref()
                .and_then(|_| chunk_span(off, piece.len(), first_kept));
            match (span, hidden_out.as_deref_mut()) {
                (Some((skip, count)), Some(out)) => {
                    self.forward_chunk_features(gpu, piece, last, skip, count, out)?;
                    if let Some(window) = self.spec.window {
                        trim_feature_rows(out, row_width, window);
                    }
                }
                _ => self.forward_chunk(gpu, piece, last)?,
            }
            off += piece.len();
        }
        let last_argmax = argmax_f32_host_stream_ordered(gpu, self.logits())
            .map_err(|e| format!("raon1_mini spec_advance argmax: {e:?}"))?;
        let logits = self.download_logits(gpu)?;
        Ok(SpecAdvance::Ready {
            last_argmax,
            last_logits: Some(logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let (rows, argmax_done) = self.verify_forward(gpu, block, position, scratch, hidden_out)?;
        let vocab = self.config.vocab;
        let head = self.spec.head.as_ref().expect("verify head just ensured");
        let logits = head.logits.sub_offset(0, rows * vocab);
        if !argmax_done {
            // The AR greedy token's kernel, one launch over every row (a
            // recorded verify graph already ran it).
            let mut indices = head.argmax.sub_offset(0, rows);
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
            .map_err(|e| format!("raon1_mini verify argmax: {e:?}"))?;
        }
        let picks = gpu
            .download_f32_stream_ordered(&head.argmax.sub_offset(0, rows))
            .map_err(|e| format!("raon1_mini verify argmax download: {e:?}"))?;
        Ok(picks.into_iter().map(f32::to_bits).collect())
    }

    fn verify_block_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let (rows, _) = self.verify_forward(gpu, block, position, scratch, hidden_out)?;
        let head = self.spec.head.as_ref().expect("verify head just ensured");
        gpu.download_f32_stream_ordered(&head.logits.sub_offset(0, rows * self.config.vocab))
            .map_err(|e| format!("raon1_mini verify logits download: {e:?}"))
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let ss = spec_scratch(scratch, "commit_prefix")?;
        if block.is_empty() {
            return Err("raon1_mini commit_prefix cannot commit an empty block".into());
        }
        let (ticket_position, ticket_rows) = ss
            .ticket
            .take()
            .ok_or("raon1_mini commit_prefix: no verify ticket is armed")?;
        if ticket_position != position || ticket_rows != block.len() {
            return Err(format!(
                "raon1_mini commit_prefix: block of {} rows at {position} does not match the \
                 verify of {ticket_rows} rows at {ticket_position}",
                block.len()
            ));
        }
        if accept_len >= block.len() {
            return Err(format!(
                "raon1_mini commit_prefix accepts {accept_len} drafts out of {}",
                block.len() - 1
            ));
        }
        let committed = accept_len + 1;
        let result = if committed < block.len() {
            // Partial accept: the verify over-advanced the recurrent state.
            // No capture or logits are needed, the drafter already read this
            // window's features.
            match self.dflash_rollback_mode() {
                Ok(RollbackMode::Recurrence) => {
                    self.rollback_rows(gpu, &ss.snapshot, position, block.len(), committed)
                }
                Ok(RollbackMode::Replay) => {
                    // Restore the pre-window snapshot (length and mirror
                    // watermark included) and replay the kept rows through the
                    // same batched forward.
                    self.state
                        .restore(gpu, &ss.snapshot)
                        .and_then(|()| self.forward_chunk(gpu, &block[..committed], false))
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(())
        };
        self.spec.clear_ring_ticket();
        result?;
        let end = position + committed;
        if self.state.n_tokens() != end {
            return Err(format!(
                "raon1_mini commit_prefix ended at {}, expected {end}",
                self.state.n_tokens()
            ));
        }
        Ok(())
    }

    fn abandon_verify(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let ss = spec_scratch(scratch, "abandon_verify")?;
        if ss.ticket.take().is_none() {
            return Ok(());
        }
        let restored = self.state.restore(gpu, &ss.snapshot);
        self.spec.clear_ring_ticket();
        restored
    }

    fn eos_token(&self) -> u32 {
        self.eos[0]
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq()
    }

    fn map_ahead_for_decode(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        remaining_new_tokens: usize,
        max_draft_rows: usize,
    ) {
        <Self as ArchModel>::map_ahead_for_decode(
            self,
            gpu,
            position,
            remaining_new_tokens,
            max_draft_rows,
        );
    }

    fn dflash_streams_per_layer(&self) -> usize {
        self.config.mhc_streams
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        self.spec.layers()
    }

    fn set_dflash_extract_layers(&mut self, layers: Vec<usize>) {
        self.spec.set_layers(layers);
    }

    fn set_dflash_feature_window(&mut self, rows: usize) {
        self.spec.set_window(rows);
    }

    /// The raw embedding row of `token_id` (no scaling), the drafter's noise
    /// input: the very lookup the forward's embedding stage runs.
    fn embed_row(&mut self, gpu: &mut Gpu, token_id: u32) -> Result<Vec<f32>, String> {
        stage_decode_token(&self.config, &self.scratch, gpu, token_id)?;
        embed_lookup(
            &self.config,
            &self.weights,
            &self.scratch,
            gpu,
            1,
            &self.scratch.branch_out,
        )?;
        gpu.download_f32(&self.scratch.branch_out.sub_offset(0, self.config.hidden))
            .map_err(|e| format!("raon1_mini embed_row download: {e:?}"))
    }

    /// Raon's `lm_head` over the draft's final rows: they are already normed by
    /// the draft's own `norm`, so only the projection runs, with the target
    /// head's precision (`OutRound::Bf16`). The `[n, vocab]` F32 view is the
    /// bundle's verify-head logits, valid until the next verify or head call.
    fn lm_head_logits_device(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<GpuTensor, String> {
        let hidden = self.config.hidden;
        let vocab = self.config.vocab;
        if n == 0 || hidden_rows.dtype != DType::F32 || hidden_rows.numel() < n * hidden {
            return Err(format!(
                "raon1_mini lm_head: needs {n} F32 rows of {hidden} (got {} {:?})",
                hidden_rows.numel(),
                hidden_rows.dtype
            ));
        }
        self.ensure_head_buffers(gpu, n)?;
        let head = self.spec.head.as_ref().expect("verify head just ensured");
        let mut logits = head.logits.sub_offset(0, n * vocab);
        project(
            gpu,
            &self.weights.lm_head,
            &hidden_rows.sub_offset(0, n * hidden),
            &logits,
            n,
            OutRound::Bf16,
            PrefillActTier::F16,
        )?;
        logits.shape = vec![n, vocab];
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_from_drops_everything_below_the_window() {
        assert_eq!(keep_from(10, None), 0);
        assert_eq!(keep_from(10, Some(2048)), 0);
        assert_eq!(keep_from(2048, Some(2048)), 0);
        assert_eq!(keep_from(3000, Some(2048)), 952);
    }

    #[test]
    fn chunk_spans_cover_exactly_the_kept_tail() {
        // 3000 rows in 512-row chunks, window 2048: rows 952.. are kept.
        let first = keep_from(3000, Some(2048));
        let mut kept = 0;
        let mut off = 0;
        let mut spans = Vec::new();
        while off < 3000 {
            let len = 512.min(3000 - off);
            if let Some((skip, count)) = chunk_span(off, len, first) {
                // The kept rows are the chunk's tail and start at `first` overall.
                assert_eq!(off + skip, first.max(off));
                assert_eq!(skip + count, len);
                kept += count;
            }
            spans.push(chunk_span(off, len, first));
            off += len;
        }
        assert_eq!(kept, 2048);
        assert_eq!(spans[0], None);
        assert_eq!(spans[1], Some((440, 72)));
        assert_eq!(spans[2], Some((0, 512)));
        // No window keeps every chunk whole.
        assert_eq!(chunk_span(512, 512, 0), Some((0, 512)));
    }

    #[test]
    fn extract_layers_and_window_setters() {
        let mut spec = RaonSpecState::default();
        assert_eq!(spec.layers(), None);
        spec.set_layers(vec![1, 6, 11, 16, 21]);
        assert_eq!(spec.layers(), Some(&[1usize, 6, 11, 16, 21][..]));
        // An empty list disarms (the drafter builder's failure cleanup).
        spec.set_layers(Vec::new());
        assert_eq!(spec.layers(), None);
        spec.set_window(2048);
        assert_eq!(spec.window, Some(2048));
        spec.set_window(0);
        assert_eq!(spec.window, None);
    }

    #[test]
    fn trimmed_sink_keeps_at_least_the_window() {
        // The sink trim shared with the drafter keeps `window` rows once it
        // holds more than twice that.
        let row = 4;
        let mut sink: Vec<f32> = (0..9 * row).map(|i| i as f32).collect();
        assert_eq!(trim_feature_rows(&mut sink, row, 4), 5);
        assert_eq!(sink.len(), 4 * row);
        assert_eq!(sink[0], (5 * row) as f32);
        let mut short: Vec<f32> = vec![0.0; 8 * row];
        assert_eq!(trim_feature_rows(&mut short, row, 4), 0);
    }
}
