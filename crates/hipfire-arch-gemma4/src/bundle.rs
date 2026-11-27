// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Published Gemma 4 owner: config, weights, state and the engine session
//! cache.
//!
//! The KV caches are the whole request state (Gemma 4 is pure attention, and
//! the drafter keeps nothing of its own), so a session snapshot is the
//! append-only KV rows of every cache slot up to its position. Rows are
//! position-indexed and never rewritten below the position, so a snapshot
//! stores only the rows above its parent.
//!
//! Every reuse point is a snapshot boundary whose rows a cold prefill wrote
//! with the same chunks, so a cached turn is byte-identical to a cold one.
//! A prefill splits its chunks at each boundary it crosses, and every
//! boundary depends only on the tokens before it ([`turn_boundaries`]), so
//! the chunks below a boundary are the same in every prompt with that
//! prefix. In the first chunk the boundaries are the position after each
//! end-of-turn token (`<turn|>`), so the next turn of a short conversation
//! resumes after the last end-of-turn token of its previous prompt. Above
//! it they are chunk multiples only (a split per message would make a long
//! history prefill in many small chunks): a long conversation resumes at
//! its previous prompt rounded down to the chunk. The decoded reply is never
//! reused: the chat template drops the thinking-off primer
//! (`<|channel>thought\n<channel|>`) of earlier model turns, so the reply has
//! other positions in the next prompt.

mod lanes;
mod mesh_family;

use crate::config::Gemma4Config;
use crate::forward;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::session_cache::{
    RowStream, SessionCache, SessionDriver, SessionRoute, SessionState, SnapshotParts, StateLayout,
};
use rdna_compute::Gpu;

/// Snapshot boundary stride in prompt tokens.
pub const SESSION_STRIDE: usize = 1024;

/// Prefill rows per forward on `gpu_arch` where batched prefill is validated
/// (`HIPFIRE_GEMMA4_PREFILL_BATCH` overrides), else per-token decode. gfx1151
/// takes 256-row batches: its grouped MoE, Q8 and attention prefill routes
/// read each weight / KV tile once per batch.
pub fn prefill_rows_for_arch(gpu_arch: &str, requested: Option<usize>) -> usize {
    requested
        .unwrap_or(match gpu_arch {
            "gfx1151" => 256,
            "gfx1100" | "gfx1201" => 64,
            _ => 1,
        })
        .clamp(1, crate::gemma4::GEMMA4_FORWARD_BATCH_MAX)
}

/// Snapshot boundaries of `prompt` in `(after, up_to]`, below the prompt end
/// (the last segment of a prompt runs in prefill-width chunks, which a longer
/// prompt runs in batch-max chunks): in the first batch-max chunk `C` the
/// position after each `turn_end` token, then `C` itself, every
/// [`SESSION_STRIDE`] tokens, and the prompt rounded down to `C`. The
/// boundaries above `C` are multiples of `C`, the chunk grid there.
fn turn_boundaries(turn_end: u32, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
    const C: usize = crate::gemma4::GEMMA4_FORWARD_BATCH_MAX;
    let grid_end = prompt.len() / C * C;
    (after + 1..=up_to.min(prompt.len().saturating_sub(1)))
        .filter(|&p| {
            (p < C && prompt[p - 1] == turn_end)
                || p == C
                || p == grid_end
                || p % SESSION_STRIDE == 0
        })
        .collect()
}

/// Published Gemma 4 owner (arch_id 13).
pub struct Gemma4Bundle {
    pub config: Gemma4Config,
    pub weights: Gemma4Weights,
    pub state: Gemma4State,
    /// End-of-turn token resolved from the tokenizer at load.
    pub eos_tok: u32,
    /// Rows per prefill forward (1 = per-token decode).
    prefill_rows: usize,
    /// Prefill snapshot cache shared by every session plus this
    /// conversation's committed live state; `None` = off.
    session: Option<SessionDriver>,
    /// The KV state the engine loop's step program declares.
    mesh_state: Vec<hipfire_runtime::weight_manifest::StateEntry>,
    /// The engine loop's lanes, when the load staged a continuous batch.
    lanes: Option<lanes::Gemma4Lanes>,
}

/// Bytes of one position of one Q8_0 cache slot (K or V): 34-byte blocks of
/// 32 elements per head.
fn q8_row_bytes(kv: &KvCache) -> usize {
    kv.n_kv_heads * (kv.head_dim / 32) * 34
}

impl Gemma4Bundle {
    pub fn new(
        config: Gemma4Config,
        weights: Gemma4Weights,
        state: Gemma4State,
        gpu: &Gpu,
    ) -> Self {
        let requested = hipfire_config::developer_var("HIPFIRE_GEMMA4_PREFILL_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let mut prefill_rows = prefill_rows_for_arch(&gpu.arch, requested);
        if prefill_rows > 1 && !forward::supports_batched_prefill(&weights, &state) {
            eprintln!(
                "  gemma4: batched prefill is unavailable for this load; using per-token prefill"
            );
            prefill_rows = 1;
        }
        Self {
            eos_tok: config.eos_token,
            mesh_state: mesh_family::gemma4_state_manifest(&config),
            config,
            weights,
            state,
            prefill_rows,
            session: None,
            lanes: None,
        }
    }

    /// Attach the session cache: prefill then restores and captures
    /// [`SESSION_STRIDE`]-token snapshots through it.
    pub fn attach_session_cache(&mut self, cache: SessionCache) {
        self.session = Some(SessionDriver::new(cache));
    }

    /// The attached session cache, if any.
    pub fn session_cache(&self) -> Option<&SessionCache> {
        self.session.as_ref().map(SessionDriver::cache)
    }

    /// Bytes of one [`SESSION_STRIDE`]-token delta snapshot.
    pub fn session_snapshot_bytes(&self) -> usize {
        let row = |kv: &KvCache| 2 * kv.k_gpu.len() * q8_row_bytes(kv);
        SESSION_STRIDE * (row(&self.state.kv_sliding) + row(&self.state.kv_full))
    }

    /// Run `f` with the driver taken out, so it can drive `self` as its
    /// [`SessionState`]. `None` without a cache.
    fn with_session<R>(&mut self, f: impl FnOnce(&mut SessionDriver, &mut Self) -> R) -> Option<R> {
        let mut session = self.session.take()?;
        let result = f(&mut session, self);
        self.session = Some(session);
        Some(result)
    }

    /// Prefill the canonical `prompt` on `route`, skipping the `reused`
    /// tokens `ArchModel::session_plan` returned: resume at the resident
    /// snapshot, restore a cached one, or start cold, then run the rest in
    /// prefill chunks that stop at every snapshot boundary. Returns the last
    /// prompt token's logits.
    pub fn prefill(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<Vec<f32>, String> {
        self.prefill_abortable(gpu, prompt, reused, route, &|| false)
            .map(|logits| logits.expect("a never-abort prefill finishes"))
    }

    /// [`Self::prefill`] that polls `abort` before every forward segment (the
    /// snapshot-boundary cadence, so the chunks are the same); `None`: the
    /// client aborted, the state holds a partial prompt.
    pub fn prefill_abortable(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
        abort: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, String> {
        if reused >= prompt.len() {
            return Err(format!(
                "gemma4 prefill: reusing {reused} of {} prompt tokens leaves nothing to compute",
                prompt.len()
            ));
        }
        match self.with_session(|session, bundle| {
            session.begin(gpu, bundle, prompt, route, reused, false)
        }) {
            Some(result) => {
                result?;
            }
            None if reused == 0 => self.state.reset(),
            None => {
                return Err(format!(
                    "gemma4 session cache is not attached; cannot reuse {reused} tokens"
                ))
            }
        }
        let mut pos = reused;
        loop {
            if abort() {
                return Ok(None);
            }
            let stop = self.segment_end(prompt, pos);
            if stop == prompt.len() {
                return self.forward_rows(gpu, &prompt[pos..], pos).map(Some);
            }
            self.forward_rows_silent(gpu, &prompt[pos..stop], pos)?;
            self.capture_at(gpu, prompt, stop)?;
            pos = stop;
        }
    }

    /// End of the prefill segment of `prompt` that starts at `pos`: the next
    /// snapshot boundary ([`turn_boundaries`]), with or without a session, so
    /// the chunks never depend on the cache.
    fn segment_end(&self, prompt: &[u32], pos: usize) -> usize {
        turn_boundaries(self.eos_tok, prompt, pos, prompt.len())
            .first()
            .copied()
            .unwrap_or(prompt.len())
    }

    /// Capture the session boundary at `end` of `prompt`, when the session
    /// armed one there.
    fn capture_at(&mut self, gpu: &mut Gpu, prompt: &[u32], end: usize) -> Result<(), String> {
        if self.session.as_ref().and_then(SessionDriver::next_boundary) != Some(end) {
            return Ok(());
        }
        self.with_session(|session, bundle| session.at_boundary(gpu, bundle, &prompt[..end]))
            .unwrap_or(Ok(()))
    }

    /// `tokens` at `[start, start + len)` in prefill chunks; returns the last
    /// token's logits.
    pub(crate) fn forward_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, String> {
        let width = self.prefill_rows;
        let mut logits = Vec::new();
        let n_chunks = tokens.len().div_ceil(width);
        for (i, chunk) in tokens.chunks(width).enumerate() {
            let pos = start + i * width;
            logits = if let [token] = chunk {
                forward::decode_step(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    gpu,
                    *token,
                    pos as u32,
                )?
            } else if i + 1 < n_chunks {
                // Only the last chunk's logits are returned; skip the LM head.
                forward::forward_prefill_batch(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    gpu,
                    chunk,
                    pos,
                )?;
                Vec::new()
            } else {
                forward::forward_batch(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    gpu,
                    chunk,
                    pos,
                )?
            };
        }
        Ok(logits)
    }

    /// As [`Self::forward_rows`] but without the logits of batched chunks.
    fn forward_rows_silent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String> {
        if self.prefill_rows == 1 {
            return self.forward_rows(gpu, tokens, start).map(drop);
        }
        forward::forward_prefill_batch(
            &self.config,
            &self.weights,
            &mut self.state,
            gpu,
            tokens,
            start,
        )
    }

    fn layout(&self, rows: usize) -> StateLayout<'_> {
        let mut streams = Vec::new();
        for kv in [&self.state.kv_sliding, &self.state.kv_full] {
            let row_bytes = q8_row_bytes(kv);
            for tensor in kv.k_gpu.iter().chain(&kv.v_gpu) {
                streams.push(RowStream {
                    buf: &tensor.buf,
                    row_bytes,
                    rows,
                });
            }
        }
        StateLayout {
            fixed: Vec::new(),
            rows: streams,
        }
    }
}

fn meta_position(meta: &[u8]) -> Result<usize, String> {
    let bytes: [u8; 8] = meta
        .try_into()
        .map_err(|_| "gemma4 session snapshot shape mismatch".to_string())?;
    Ok(u64::from_le_bytes(bytes) as usize)
}

impl SessionState for Gemma4Bundle {
    /// One scope for both routes: the drafter keeps no state and the target
    /// prefills the same way on either. The prefill width is part of it
    /// because batched and per-token prefill write different KV bits.
    fn snapshot_scope(&self, _route: SessionRoute) -> Option<String> {
        (self.state.kv_sliding.quant_q8 && self.state.kv_full.quant_q8)
            .then(|| format!("gemma4/rows{}", self.prefill_rows))
    }

    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
        turn_boundaries(self.eos_tok, prompt, after, up_to)
    }

    fn snapshot_parts(
        &mut self,
        _route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String> {
        if self.state.n_tokens != position {
            return Err(format!(
                "gemma4 session capture: state at {}, not {position}",
                self.state.n_tokens
            ));
        }
        Ok(SnapshotParts {
            meta: (position as u64).to_le_bytes().to_vec(),
            layout: self.layout(position),
        })
    }

    fn restore_parts(
        &mut self,
        _gpu: &mut Gpu,
        _route: SessionRoute,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, String> {
        let position = meta_position(meta)?;
        if position >= self.state.max_seq {
            return Err(format!(
                "gemma4 session restore: position {position} exceeds max_seq {}",
                self.state.max_seq
            ));
        }
        Ok(self.layout(position))
    }

    fn finish_restore(
        &mut self,
        _gpu: &mut Gpu,
        _route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String> {
        self.state.n_tokens = meta_position(meta)?;
        Ok(())
    }

    fn live_position(&self, _route: SessionRoute) -> Option<usize> {
        Some(self.state.n_tokens)
    }

    fn growth_reserve_bytes(&self, _gpu: &Gpu, _through: usize) -> u64 {
        // Both caches are allocated for the admitted context at load.
        0
    }

    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }
}

impl hipfire_runtime::arch_model::ArchModel for Gemma4Bundle {
    fn dim(&self) -> usize {
        self.config.dim
    }
    fn n_layers(&self) -> usize {
        self.config.n_layers
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn arch_key(&self) -> &'static str {
        "gemma4"
    }
    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        // Two caches (sliding + full) and no basis for preferring one.
        None
    }
    fn reset_session_state(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset();
        if let Some(session) = self.session.as_mut() {
            session.forget_live();
        }
        Ok(())
    }
    fn session_cache_attached(&self) -> bool {
        self.session.is_some()
    }
    fn session_plan_capped(&self, prompt: &[u32], route: SessionRoute, cap: usize) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| session.plan_capped(self, prompt, route, cap))
    }
    fn session_plan_is_live(&self, prompt: &[u32], route: SessionRoute, reused: usize) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.plan_is_live(self, prompt, route, reused))
    }
    fn session_commit(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.commit();
        }
    }
    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        if let Some(mut session) = b.session {
            session.clear(gpu);
        }
        b.state.free_gpu(gpu);
        if let Some(lanes) = b.lanes {
            lanes.free_gpu(gpu);
        }
        b.weights.free_gpu(gpu);
    }
    fn mesh_family(&mut self) -> Option<&mut dyn hipfire_runtime::mesh_family::MeshFamily> {
        Some(self)
    }
    fn spec_target(&mut self) -> Option<&mut dyn hipfire_runtime::spec::SpecTarget> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{prefill_rows_for_arch, turn_boundaries, SESSION_STRIDE};

    #[test]
    fn validated_arches_default_to_batched_prefill() {
        assert_eq!(prefill_rows_for_arch("gfx1100", None), 64);
        assert_eq!(prefill_rows_for_arch("gfx1201", None), 64);
        assert_eq!(prefill_rows_for_arch("gfx1151", None), 256);
    }

    #[test]
    fn unvalidated_arches_keep_the_sequential_default() {
        for arch in ["gfx1200", "gfx942"] {
            assert_eq!(prefill_rows_for_arch(arch, None), 1);
        }
    }

    #[test]
    fn explicit_override_wins_and_is_clamped() {
        assert_eq!(prefill_rows_for_arch("gfx1100", Some(8)), 8);
        assert_eq!(prefill_rows_for_arch("gfx1201", Some(32)), 32);
        assert_eq!(prefill_rows_for_arch("gfx1100", Some(0)), 1);
        assert_eq!(prefill_rows_for_arch("gfx1100", Some(512)), 256);
    }

    #[test]
    fn a_follow_up_has_the_turn_ends_of_its_previous_prompt() {
        const END: u32 = 106;
        // <bos> <|turn>user\n U <turn|>\n <|turn>model\n, then the primer.
        let first = [2, 105, 3, 10, 7, 7, END, 10, 105, 4, 10, 100, 101, 10, 102];
        assert_eq!(turn_boundaries(END, &first, 0, first.len()), [7]);
        // The template drops the primer: the reply follows the model header.
        let mut next = first[..11].to_vec();
        next.extend([50, 51, 52, END, 10, 105, 3, 10, 8, END, 10, 105, 4, 10]);
        next.extend(&first[11..]);
        assert_eq!(turn_boundaries(END, &next, 0, next.len()), [7, 15, 21]);
        // Nothing at or below `after`, `up_to` included.
        assert_eq!(turn_boundaries(END, &next, 7, 15), [15]);
        // Above the first chunk only its end, the stride and the prompt
        // rounded down to the chunk; never the prompt end.
        let mut long = vec![7u32; 2 * SESSION_STRIDE];
        long[100] = END;
        long[600] = END;
        assert_eq!(
            turn_boundaries(END, &long, 0, long.len()),
            [101, 256, SESSION_STRIDE]
        );
        assert_eq!(
            turn_boundaries(END, &long[..1300], 0, 1300),
            [101, 256, SESSION_STRIDE, 1280]
        );
    }
}
