// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch lane decode: `B` independent sequences decode one token
//! each through the same layer program as the singleton (`rows = B`), each
//! lane at its own position over a lane-major Q8 KV cache
//! ([`KvRoute::Lanes`](hipfire_dispatch::pipeline::sandwich::KvRoute)).
//!
//! A lane is seeded by the singleton prefill on the caller's [`GlimmerState`]
//! and [`GlimmerLanes::import`] of its KV rows, so lane KV is byte-identical
//! to the singleton's at the seam.

use crate::config::GlimmerConfig;
use crate::forward::{embed_rows, glimmer_lm_head_picks, htod_i32, rows_io, RowsScratch};
use crate::glimmer::{GlimmerState, GlimmerWeights, GLIMMER_MAX_SPEC_BLOCK};
use crate::program::{self, LaneKv};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, RowsRoute};
use hipfire_runtime::llama::KvCache;
use rdna_compute::Gpu;

/// Lane-major KV and the `max_batch`-row scratch of a lane decode.
pub struct GlimmerLanes {
    pub max_batch: usize,
    pub lane_capacity: usize,
    kv_sliding: KvCache,
    kv_full: KvCache,
    scratch: RowsScratch,
}

impl GlimmerLanes {
    /// Largest lane the per-head lane attention admits on `gpu`: its LDS
    /// grows with the lane (about 16K rows on gfx1151).
    pub fn max_lane_capacity(gpu: &Gpu, cfg: &GlimmerConfig) -> usize {
        gpu.attention_q8_0_kv_independent_max_lane_capacity(cfg.head_dim)
    }

    pub fn new(
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        max_batch: usize,
        lane_capacity: usize,
    ) -> Result<Self, String> {
        // The batched LM head shares the singleton state's verify buffers.
        if !(1..=GLIMMER_MAX_SPEC_BLOCK).contains(&max_batch) {
            return Err(format!(
                "glimmer lanes: max_batch {max_batch} outside 1..={GLIMMER_MAX_SPEC_BLOCK}"
            ));
        }
        let max_cap = Self::max_lane_capacity(gpu, cfg);
        if lane_capacity == 0 || lane_capacity > max_cap {
            return Err(format!(
                "glimmer lanes: lane capacity {lane_capacity} outside 1..={max_cap}"
            ));
        }
        let rows = max_batch * lane_capacity;
        let (n_kv, hd) = (cfg.n_kv_heads, cfg.head_dim);
        let kv_sliding = KvCache::new_gpu_q8(gpu, cfg.n_sliding_layers(), n_kv, hd, rows)
            .map_err(|e| format!("glimmer lanes: sliding kv: {e:?}"))?;
        let kv_full = match KvCache::new_gpu_q8(gpu, cfg.n_full_layers(), n_kv, hd, rows) {
            Ok(kv) => kv,
            Err(e) => {
                let _ = kv_sliding.free_gpu(gpu);
                return Err(format!("glimmer lanes: full kv: {e:?}"));
            }
        };
        let scratch = match RowsScratch::alloc(gpu, cfg, max_batch) {
            Ok(s) => s,
            Err(e) => {
                let _ = kv_sliding.free_gpu(gpu);
                let _ = kv_full.free_gpu(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            max_batch,
            lane_capacity,
            kv_sliding,
            kv_full,
            scratch,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv_sliding.free_gpu(gpu);
        let _ = self.kv_full.free_gpu(gpu);
        self.scratch.free(gpu);
    }

    /// Copy positions `0..len` of `state`'s KV (a singleton prefill) into
    /// lane `lane`.
    pub fn import(
        &self,
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        state: &GlimmerState,
        lane: usize,
        len: usize,
    ) -> Result<(), String> {
        if lane >= self.max_batch || len > self.lane_capacity || len > state.max_seq {
            return Err(format!(
                "glimmer lanes: import of {len} rows into lane {lane} (lanes {}, capacity {})",
                self.max_batch, self.lane_capacity
            ));
        }
        // Q8_0 rows of `n_kv * head_dim / 32` blocks of 34 bytes, in F32 words.
        let row = cfg.n_kv_heads * (cfg.head_dim / 32) * 34 / 4;
        let (dst_off, n) = (lane * self.lane_capacity * row, len * row);
        for (src, dst) in [
            (&state.kv_sliding, &self.kv_sliding),
            (&state.kv_full, &self.kv_full),
        ] {
            for (s, d) in src
                .k_gpu
                .iter()
                .zip(&dst.k_gpu)
                .chain(src.v_gpu.iter().zip(&dst.v_gpu))
            {
                // A kernel copy, so graph capture and replay see it.
                gpu.copy_f32_buffer(&d.sub_offset(dst_off, n), &s.sub_offset(0, n), n)
                    .map_err(|e| format!("glimmer lanes: import: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Decode one token per lane: lane `r` takes `tokens[r]` at
    /// `positions[r]`; lanes outside `active_mask` write no KV and their rows
    /// are garbage. Returns each row's argmax; `logits_out` receives the
    /// `B * vocab` logits.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        gpu: &mut Gpu,
        cfg: &GlimmerConfig,
        weights: &GlimmerWeights,
        state: &GlimmerState,
        tokens: &[u32],
        positions: &[usize],
        active_mask: u64,
        logits_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let b = tokens.len();
        if b == 0 || b > self.max_batch || positions.len() != b {
            return Err(format!(
                "glimmer lanes: {b} tokens, {} positions, {} lanes",
                positions.len(),
                self.max_batch
            ));
        }
        if b < 64 && active_mask >> b != 0 {
            return Err(format!(
                "glimmer lanes: mask {active_mask:#x} past {b} lanes"
            ));
        }
        if let Some((lane, &p)) = positions
            .iter()
            .enumerate()
            .find(|&(r, &p)| active_mask >> r & 1 == 1 && p >= self.lane_capacity)
        {
            return Err(format!(
                "glimmer lanes: lane {lane} at {p} past capacity {}",
                self.lane_capacity
            ));
        }
        let s = self.scratch.rows(cfg, b);
        htod_i32(gpu, &s.positions, positions.iter().map(|&p| p as i32))?;
        htod_i32(gpu, &s.token_ids, tokens.iter().map(|&t| t as i32))?;
        embed_rows(gpu, cfg, weights, state, &s, tokens)?;
        let ctx = DispatchCtx::new(gpu);
        let lanes = LaneKv {
            sliding: &self.kv_sliding,
            full: &self.kv_full,
            lane_capacity: self.lane_capacity,
            active_mask,
        };
        // `Prefill` picks the multi-row projection ladder; attention takes
        // the lane route, never the prefill kernels.
        let mut io = rows_io(state, &s, b, RowsRoute::Prefill, 0);
        io.lanes = Some(lanes);
        for (layer, lw) in weights.layers.iter().enumerate() {
            let steps = program::layer_steps(cfg, layer, lw, state, &io);
            execute_steps(gpu, &ctx, &steps)
                .map_err(|e| format!("glimmer lanes L{layer}: {e:?}"))?;
        }
        execute_steps(gpu, &ctx, &[io.final_norm(cfg, weights)])
            .map_err(|e| format!("glimmer lanes final norm: {e:?}"))?;
        glimmer_lm_head_picks(
            gpu,
            cfg,
            weights,
            &s.x,
            b,
            &state.logits,
            &state.logits_batch,
            &state.argmax_batch,
            logits_out,
        )
    }
}
