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

use crate::config::Gemma4Config;
use crate::forward;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::session_cache::{
    stride_boundaries, RowStream, SessionCache, SessionDriver, SessionRoute, SessionState,
    SnapshotParts, StateLayout,
};
use rdna_compute::Gpu;

/// Snapshot boundary stride in prompt tokens (a multiple of the 256-row
/// prefill chunk, so restored prefills keep the cold chunk grid).
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
            config,
            weights,
            state,
            prefill_rows,
            session: None,
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
    /// tokens `ArchModel::session_plan` returned: continue the live state,
    /// restore a snapshot, or start cold, then run the rest in prefill
    /// chunks that stop at every snapshot boundary. Returns the last prompt
    /// token's logits.
    pub fn prefill(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<Vec<f32>, String> {
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
            let boundary = self.session.as_ref().and_then(SessionDriver::next_boundary);
            let stop = boundary.unwrap_or(prompt.len()).min(prompt.len());
            if stop == prompt.len() {
                let logits = self.forward_rows(gpu, &prompt[pos..], pos)?;
                if boundary == Some(stop) {
                    self.with_session(|session, bundle| session.at_boundary(gpu, bundle, prompt))
                        .unwrap_or(Ok(()))?;
                }
                return Ok(logits);
            }
            self.forward_rows_silent(gpu, &prompt[pos..stop], pos)?;
            self.with_session(|session, bundle| session.at_boundary(gpu, bundle, &prompt[..stop]))
                .unwrap_or(Ok(()))?;
            pos = stop;
        }
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
        stride_boundaries(SESSION_STRIDE, prompt, None, after, up_to)
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
    fn session_plan(&self, prompt: &[u32], route: SessionRoute) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| session.plan(self, prompt, route))
    }
    fn session_commit(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.commit();
        }
    }
    fn session_commit_live(&mut self, consumed: &[u32]) {
        if let Some(mut session) = self.session.take() {
            session.commit_live(self, consumed);
            self.session = Some(session);
        }
    }
    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        if let Some(mut session) = b.session {
            session.clear(gpu);
        }
        b.state.free_gpu(gpu);
        b.weights.free_gpu(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::prefill_rows_for_arch;

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
}
