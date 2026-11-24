// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Rollback of a speculative verify window's rejected rows.
//!
//! A verify forward writes every row of `[seed, drafts..]` at
//! `start_pos..start_pos + rows` into the per-layer state before the accept
//! rule picks the committed prefix (the seed plus `accept_len` drafts). Two of
//! those states are finite rings in which a rejected row destroys history the
//! committed prefix still reads:
//!
//! - The SWA ring (`sliding_window` slots, slot `pos % window`): row `j`
//!   overwrites position `start_pos + j - window`. The next window starts at
//!   `n = start_pos + accept_len + 1` and still attends every position from
//!   `n + 1 - window`, so each rejected row `j >= accept_len + 2` clobbers
//!   committed history once `start_pos + j >= window`. (Row `accept_len + 1`
//!   is harmless: its slot holds `n - window`, which no later row sees, and the
//!   next window rewrites it first.)
//! - The compressor rings (the open window's `ratio` rows; at ratio 4 also the
//!   previous window): a rejected row that closes a window shifts the ring, and
//!   the rows after it overwrite committed slots of the open window. Rejected
//!   rows that close no window only touch slots the next window rewrites
//!   before it pools.
//!
//! [`VerifyRollback::arm`] runs before the verify forward and saves what a
//! rejection can destroy: the SWA columns of rows `>= 2`, and the whole rings
//! of each compressor whose window closes on a row `>= 1`. The forward then
//! stashes those compressors' projected rows
//! ([`hipfire_dispatch::pipeline::mla::compressor::CompressorStash`]).
//! [`VerifyRollback::rollback`] runs on commit: a fully accepted window keeps
//! the forward's state untouched; otherwise it restores the rejected rows' SWA
//! columns and rebuilds each compressor ring whose window a rejected row closed
//! from its saved rings plus the committed rows, as sequential decode would
//! have left it. Every copy is a kernel (D2D `hipMemcpy` runs a slow blit on
//! gfx1151, and kernel copies are recorded by captured verify bodies).

use hipfire_dispatch::pipeline::mla::compressor::CompressorStash;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::deepseek4::{IndexerLayerState, MainAttentionLayerState};
use crate::DeepseekV4Config;

/// Rows a compressor stash holds at least: every DSpark verify window (block
/// size clamped to 8, plus the seed), so a captured verify body's stash
/// pointers never move between block sizes.
const STASH_ROWS: usize = 16;

/// Saved state of the armed verify window, consumed by its commit.
#[derive(Default)]
pub struct VerifyRollback {
    /// `(start_pos, rows)` of the armed window.
    window: Option<(usize, usize)>,
    layers: Vec<LayerUndo>,
}

#[derive(Default)]
struct LayerUndo {
    /// First window row whose SWA column is saved; `None` when none is.
    swa_first: Option<usize>,
    /// Saved columns `[n_kv * head_dim, rows - swa_first]`.
    swa: Option<GpuTensor>,
    /// Main, then (ratio 4) indexer flavor.
    compressor: [CompressorUndo; 2],
}

#[derive(Default)]
struct CompressorUndo {
    armed: bool,
    /// The rings before the window.
    kv_state: Option<GpuTensor>,
    score_state: Option<GpuTensor>,
    /// The window's projected rows (`score` with its APE), stashed by the forward.
    kv_rows: Option<GpuTensor>,
    score_rows: Option<GpuTensor>,
}

/// One compressor flavor's rings and geometry.
struct CompressorRing<'a> {
    kv_state: &'a GpuTensor,
    score_state: &'a GpuTensor,
    ratio: usize,
    proj_dim: usize,
}

/// The rings of flavor `flavor` (0 main, 1 indexer) of a layer, when it has them.
fn compressor_ring<'a>(
    cfg: &DeepseekV4Config,
    layer: &'a IndexerLayerState,
    flavor: usize,
) -> Option<CompressorRing<'a>> {
    let ratio = layer.compress_ratio as usize;
    if ratio == 0 || (flavor == 1 && ratio != 4) {
        return None;
    }
    let (kv_state, score_state, head_dim) = if flavor == 1 {
        (
            layer.indexer_kv_state.as_ref()?,
            layer.indexer_score_state.as_ref()?,
            cfg.index_head_dim,
        )
    } else {
        (
            layer.main_kv_state.as_ref()?,
            layer.main_score_state.as_ref()?,
            cfg.head_dim,
        )
    };
    // The ratio-4 compressor overlaps two windows: twice the row width.
    let proj_dim = if ratio == 4 { 2 * head_dim } else { head_dim };
    Some(CompressorRing {
        kv_state,
        score_state,
        ratio,
        proj_dim,
    })
}

/// The first position in `from..=to` that closes a `ratio`-row window.
fn first_close(from: usize, to: usize, ratio: usize) -> Option<usize> {
    let close = (from + 1).div_ceil(ratio) * ratio - 1;
    (close <= to).then_some(close)
}

/// The ring-slot runs of positions `first..first + n` (`n <= window`):
/// `(slot, column, len)`, `column` counted from `first`. The second run is
/// empty unless the range wraps.
fn slot_runs(first: usize, n: usize, window: usize) -> [(usize, usize, usize); 2] {
    let slot = first % window;
    let head = n.min(window - slot);
    [(slot, 0, head), (0, head, n - head)]
}

/// `slot` holds at least `numel` F32 elements.
fn ensure(
    gpu: &mut Gpu,
    slot: &mut Option<GpuTensor>,
    numel: usize,
    what: &str,
) -> Result<(), String> {
    if slot.as_ref().is_some_and(|t| t.numel() >= numel) {
        return Ok(());
    }
    if let Some(old) = slot.take() {
        let _ = gpu.free_tensor(old);
    }
    *slot = Some(
        gpu.alloc_tensor(&[numel], DType::F32)
            .map_err(|e| format!("verify rollback: alloc {what}: {e:?}"))?,
    );
    Ok(())
}

fn copy(
    gpu: &mut Gpu,
    dst: &GpuTensor,
    src: &GpuTensor,
    n: usize,
    what: &str,
) -> Result<(), String> {
    gpu.copy_f32_buffer(dst, src, n)
        .map_err(|e| format!("verify rollback: {what}: {e:?}"))
}

/// Columns `[src_col, src_col + len)` of the `[rows, src_cols]` `src` into
/// columns `[dst_col, dst_col + len)` of the `[rows, dst_cols]` `dst`.
#[allow(clippy::too_many_arguments)]
fn copy_columns(
    gpu: &mut Gpu,
    src: &GpuTensor,
    src_cols: usize,
    src_col: usize,
    dst: &GpuTensor,
    dst_cols: usize,
    dst_col: usize,
    rows: usize,
    len: usize,
) -> Result<(), String> {
    let src = src.sub_offset(src_col, src.numel() - src_col);
    gpu.copy_rows_strided_f32(&src, dst, rows, len, src_cols, dst_cols, dst_col, None)
        .map_err(|e| format!("verify rollback: SWA columns: {e:?}"))
}

impl VerifyRollback {
    /// Before a speculative verify forward over `rows` rows at `start_pos`:
    /// save what a rejected row can destroy. `every_window` arms every
    /// compressor: a captured or retained verify body replays at other
    /// positions, so its stash steps cannot depend on this window's closes.
    pub(crate) fn arm(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        attention: &[MainAttentionLayerState],
        indexer: &[IndexerLayerState],
        start_pos: usize,
        rows: usize,
        every_window: bool,
    ) -> Result<(), String> {
        self.window = None;
        for layer in &mut self.layers {
            layer.swa_first = None;
            for flavor in &mut layer.compressor {
                flavor.armed = false;
            }
        }
        // A lone seed is always committed.
        if rows < 2 {
            return Ok(());
        }
        let n_layers = attention.len().min(indexer.len());
        if self.layers.len() < n_layers {
            self.layers.resize_with(n_layers, LayerUndo::default);
        }
        let window = cfg.sliding_window;
        if rows > window {
            return Err(format!(
                "verify rollback: {rows}-row window exceeds the {window}-slot SWA ring"
            ));
        }
        let last = start_pos + rows - 1;
        // Rows `>= 2` whose slot already holds a committed position.
        let swa_first = 2.max(window.saturating_sub(start_pos));
        for (l, undo) in self.layers.iter_mut().enumerate().take(n_layers) {
            if let (true, Some(ring)) = (swa_first < rows, attention[l].swa_k.as_ref()) {
                let cols = rows - swa_first;
                let ring_rows = ring.numel() / window;
                ensure(gpu, &mut undo.swa, ring_rows * cols, "SWA columns")?;
                let saved = undo.swa.as_ref().unwrap();
                for (slot, column, len) in slot_runs(start_pos + swa_first, cols, window) {
                    if len > 0 {
                        copy_columns(gpu, ring, window, slot, saved, cols, column, ring_rows, len)?;
                    }
                }
                undo.swa_first = Some(swa_first);
            }
            for (flavor, saved) in undo.compressor.iter_mut().enumerate() {
                let Some(ring) = compressor_ring(cfg, &indexer[l], flavor) else {
                    continue;
                };
                if !every_window && first_close(start_pos + 1, last, ring.ratio).is_none() {
                    continue;
                }
                let n = ring.kv_state.numel();
                let stash = rows * ring.proj_dim;
                // A captured or retained body keeps the stash pointers it
                // recorded: never move them under it.
                if every_window && saved.kv_rows.as_ref().is_some_and(|t| t.numel() < stash) {
                    return Err(format!(
                        "verify rollback: {rows}-row window outgrows the stash a captured \
                         verify body recorded (l{l})"
                    ));
                }
                let capacity = rows.max(STASH_ROWS) * ring.proj_dim;
                ensure(gpu, &mut saved.kv_state, n, "compressor kv ring")?;
                ensure(gpu, &mut saved.score_state, n, "compressor score ring")?;
                ensure(gpu, &mut saved.kv_rows, capacity, "compressor kv rows")?;
                ensure(
                    gpu,
                    &mut saved.score_rows,
                    capacity,
                    "compressor score rows",
                )?;
                copy(
                    gpu,
                    saved.kv_state.as_ref().unwrap(),
                    ring.kv_state,
                    n,
                    "save kv ring",
                )?;
                copy(
                    gpu,
                    saved.score_state.as_ref().unwrap(),
                    ring.score_state,
                    n,
                    "save score ring",
                )?;
                saved.armed = true;
            }
        }
        self.window = Some((start_pos, rows));
        Ok(())
    }

    /// Where the verify forward's compressor op of `layer` / flavor stashes its
    /// window's rows, when this exact window is armed for it.
    pub(crate) fn compressor_stash(
        &self,
        layer: usize,
        is_indexer: bool,
        start_pos: u32,
        rows: usize,
    ) -> Option<CompressorStash<'_>> {
        if self.window != Some((start_pos as usize, rows)) {
            return None;
        }
        let undo = &self.layers.get(layer)?.compressor[usize::from(is_indexer)];
        if !undo.armed {
            return None;
        }
        Some(CompressorStash {
            kv: undo.kv_rows.as_ref()?,
            score: undo.score_rows.as_ref()?,
            rows,
        })
    }

    /// On commit of `accept_len` drafts of the armed window: restore what its
    /// rejected rows destroyed. Disarms. An unarmed rollback is a no-op; a
    /// commit of a different window than the armed one is an error.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollback(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        attention: &[MainAttentionLayerState],
        indexer: &[IndexerLayerState],
        max_compressed: usize,
        start_pos: usize,
        rows: usize,
        accept_len: usize,
    ) -> Result<(), String> {
        let Some(armed) = self.window.take() else {
            return Ok(());
        };
        if armed != (start_pos, rows) {
            return Err(format!(
                "verify rollback: armed for window {armed:?}, commit names ({start_pos}, {rows})"
            ));
        }
        if accept_len + 1 >= rows {
            return Ok(());
        }
        let window = cfg.sliding_window;
        let committed = start_pos + accept_len;
        let last = start_pos + rows - 1;
        let n_layers = attention.len().min(indexer.len());
        for (l, undo) in self.layers.iter().enumerate().take(n_layers) {
            if let (Some(first), Some(saved)) = (undo.swa_first, undo.swa.as_ref()) {
                let from = first.max(accept_len + 2);
                let cols = rows - first;
                let rings = [&attention[l].swa_k, &attention[l].swa_v];
                for ring in rings.into_iter().flatten() {
                    let ring_rows = ring.numel() / window;
                    let n = rows.saturating_sub(from);
                    for (slot, column, len) in slot_runs(start_pos + from, n, window) {
                        if len > 0 {
                            copy_columns(
                                gpu,
                                saved,
                                cols,
                                from - first + column,
                                ring,
                                window,
                                slot,
                                ring_rows,
                                len,
                            )?;
                        }
                    }
                }
            }
            for (flavor, saved) in undo.compressor.iter().enumerate() {
                let Some(ring) = compressor_ring(cfg, &indexer[l], flavor) else {
                    continue;
                };
                if !saved.armed || first_close(committed + 1, last, ring.ratio).is_none() {
                    continue;
                }
                rebuild_compressor(gpu, &ring, saved, max_compressed, start_pos, committed)
                    .map_err(|e| format!("{e} (l{l} flavor {flavor})"))?;
            }
        }
        Ok(())
    }

    /// Drop the armed window without touching state (a new conversation).
    pub(crate) fn disarm(&mut self) {
        self.window = None;
    }

    pub(crate) fn free_gpu(&mut self, gpu: &mut Gpu) {
        self.window = None;
        for mut layer in self.layers.drain(..) {
            let flavors = layer.compressor.iter_mut().flat_map(|c| {
                [
                    c.kv_state.take(),
                    c.score_state.take(),
                    c.kv_rows.take(),
                    c.score_rows.take(),
                ]
            });
            for t in flavors.chain([layer.swa.take()]).flatten() {
                let _ = gpu.free_tensor(t);
            }
        }
    }
}

/// The rings before the window, then the committed rows `start_pos..=committed`
/// in order: each run up to a window close is written, and a close within the
/// cache shifts the ratio-4 ring (the decode route's host commit).
fn rebuild_compressor(
    gpu: &mut Gpu,
    ring: &CompressorRing<'_>,
    saved: &CompressorUndo,
    max_compressed: usize,
    start_pos: usize,
    committed: usize,
) -> Result<(), String> {
    let (ratio, proj) = (ring.ratio, ring.proj_dim);
    let overlap = ratio == 4;
    let n = ring.kv_state.numel();
    let (kv_rows, score_rows) = (
        saved.kv_rows.as_ref().unwrap(),
        saved.score_rows.as_ref().unwrap(),
    );
    copy(
        gpu,
        ring.kv_state,
        saved.kv_state.as_ref().unwrap(),
        n,
        "restore kv ring",
    )?;
    copy(
        gpu,
        ring.score_state,
        saved.score_state.as_ref().unwrap(),
        n,
        "restore score ring",
    )?;
    let mut pos = start_pos;
    while pos <= committed {
        let close = first_close(pos, committed, ratio);
        let len = close.unwrap_or(committed) - pos + 1;
        let row = pos - start_pos;
        gpu.compressor_ring_write_batched_f32(
            &kv_rows.sub_offset(row * proj, len * proj),
            &score_rows.sub_offset(row * proj, len * proj),
            ring.kv_state,
            ring.score_state,
            len as i32,
            proj as i32,
            ratio as i32,
            (pos % ratio) as i32,
            i32::from(overlap),
        )
        .map_err(|e| format!("verify rollback: compressor ring write: {e:?}"))?;
        if let (true, Some(close)) = (overlap, close) {
            if close / ratio < max_compressed {
                for state in [ring.kv_state, ring.score_state] {
                    let cur = state.sub_offset(ratio * proj, ratio * proj);
                    let prev = state.sub_offset(0, ratio * proj);
                    copy(gpu, &prev, &cur, ratio * proj, "compressor shift")?;
                }
            }
        }
        pos += len;
    }
    Ok(())
}

impl crate::DeepseekV4State {
    /// [`VerifyRollback::arm`] over the trunk layers' rings (the verify
    /// forward never writes the MTP layer's).
    pub(crate) fn arm_verify_rollback(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        start_pos: usize,
        rows: usize,
        every_window: bool,
    ) -> Result<(), String> {
        let n = cfg.num_hidden_layers;
        let Self {
            verify_rollback,
            _attention,
            _indexer,
            ..
        } = self;
        let (attention, indexer) = (
            &_attention[..n.min(_attention.len())],
            &_indexer[..n.min(_indexer.len())],
        );
        verify_rollback.arm(gpu, cfg, attention, indexer, start_pos, rows, every_window)
    }

    /// [`VerifyRollback::rollback`] on commit of `accept_len` drafts of the
    /// `rows`-row window at `start_pos`.
    pub(crate) fn rollback_verify(
        &mut self,
        gpu: &mut Gpu,
        cfg: &DeepseekV4Config,
        start_pos: usize,
        rows: usize,
        accept_len: usize,
    ) -> Result<(), String> {
        let n = cfg.num_hidden_layers;
        let max_compressed = self.compressor_capacity.active_rows();
        let Self {
            verify_rollback,
            _attention,
            _indexer,
            ..
        } = self;
        let (attention, indexer) = (
            &_attention[..n.min(_attention.len())],
            &_indexer[..n.min(_indexer.len())],
        );
        verify_rollback.rollback(
            gpu,
            cfg,
            attention,
            indexer,
            max_compressed,
            start_pos,
            rows,
            accept_len,
        )
    }
}

#[cfg(test)]
mod tests {
    //! GPU tests at the window wrap: the production ring-write, staging,
    //! attention and compressor ops run a verify window, `commit_prefix`
    //! commits a prefix of it, and the next window must read exactly what it
    //! reads after a window that only ever held the committed rows. Each case
    //! also checks the clobber without the rollback (positive control). They
    //! skip without a GPU.

    use super::*;
    use crate::deepseek4::DeepseekV4Weights;
    use crate::spec_impl::{Deepseek4Bundle, Deepseek4DsparkScratch};
    use crate::DeepseekV4State;
    use hipfire_dispatch::pipeline::mla::attention::{
        execute_swa_attention, execute_swa_ring_write, SwaAttentionOp, SwaAttentionRoute,
        SwaRingWriteOp, SwaRingWriteRoute,
    };
    use hipfire_dispatch::pipeline::mla::compressor::{
        execute_kv_compress, CompressorCommit, CompressorInput, CompressorRope, KvCompressOp,
        KvCompressRoute,
    };
    use hipfire_dispatch::pipeline::mla::rows::{execute_swa_stage, SwaStageOp};
    use hipfire_runtime::spec::SpecTarget;

    const HEAD_DIM: usize = 512;
    const WINDOW: usize = 128;
    const N_HEADS: usize = 2;

    fn cfg(compress_ratio: u32) -> DeepseekV4Config {
        DeepseekV4Config {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: N_HEADS,
            num_key_value_heads: 1,
            head_dim: HEAD_DIM,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            q_lora_rank: 8,
            o_lora_rank: 8,
            qk_rope_head_dim: 64,
            o_groups: 1,
            n_routed_experts: 1,
            n_shared_experts: 0,
            num_experts_per_tok: 1,
            moe_intermediate_size: 16,
            routed_scaling_factor: 1.0,
            topk_method: "noaux_tc".into(),
            scoring_func: "sqrtsoftplus".into(),
            norm_topk_prob: true,
            swiglu_limit: 10.0,
            hc_mult: 4,
            hc_sinkhorn_iters: 1,
            hc_eps: 1e-6,
            index_n_heads: 1,
            index_head_dim: 128,
            index_topk: 1,
            compress_ratios: vec![compress_ratio],
            compress_rope_theta: 160000.0,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_original_max_position_embeddings: 4096,
            rope_scaling_beta_fast: 32,
            rope_scaling_beta_slow: 1,
            sliding_window: WINDOW,
            num_nextn_predict_layers: 0,
            num_hash_layers: 0,
            reap_keep: None,
            load_dspark: true,
            load_mtp: true,
            mq2r: false,
            mq2rxt: false,
        }
    }

    fn bundle(cfg: DeepseekV4Config) -> Deepseek4Bundle {
        let state = DeepseekV4State::new(&cfg).expect("DeepseekV4State::new");
        Deepseek4Bundle {
            config: cfg,
            weights: DeepseekV4Weights {
                mq2r_backend: Default::default(),
                token_embd: None,
                output_norm: None,
                head: None,
                hc_head_fn: None,
                hc_head_base: None,
                hc_head_scale: 0.0,
                layers: Vec::new(),
                mtp_layer: None,
                dspark: None,
                expert_owner: None,
                _scaffold: (),
            },
            state,
            eos_tok: 0,
            pbs: None,
        }
    }

    /// Deterministic values in `[-1, 1)`: stream `seed`, `n` values per position.
    fn values(seed: u64, first: usize, count: usize, n: usize) -> Vec<f32> {
        (first..first + count)
            .flat_map(|pos| {
                (0..n).map(move |i| {
                    let mut x = seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((pos * n + i) as u64)
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x ^= x >> 31;
                    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                    x ^= x >> 29;
                    (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                })
            })
            .collect()
    }

    fn upload(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
        gpu.upload_f32(v, &[v.len()]).expect("upload")
    }

    /// `values` as i32 into an F32-typed buffer (the position-buffer convention).
    fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
        let t = gpu
            .zeros(&[values.len().max(1)], DType::F32)
            .expect("alloc");
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
        gpu.memcpy_htod_auto(&t.buf, bytes).expect("htod");
        t
    }

    fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
        gpu.hip.device_synchronize().expect("sync");
        let mut v = gpu.download_f32(t).expect("download");
        v.truncate(t.numel());
        v
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // ── SWA ring ────────────────────────────────────────────────────────

    /// The production batched ring write of `kv` (`[rows, HEAD_DIM]`) at `start`.
    fn ring_write(gpu: &mut Gpu, k: &GpuTensor, v: &GpuTensor, start: usize, kv: &[f32]) {
        let rows = kv.len() / HEAD_DIM;
        let kv = upload(gpu, kv);
        let positions = upload_i32(gpu, &[0]);
        execute_swa_ring_write(
            gpu,
            &SwaRingWriteOp {
                kv: &kv,
                swa_k: k,
                swa_v: v,
                n_kv: 1,
                head_dim: HEAD_DIM,
                win: WINDOW,
                rows,
                layer: 0,
                route: SwaRingWriteRoute::Batched {
                    positions: &positions,
                    start_pos: start as u32,
                    capture_safe: false,
                },
            },
        )
        .expect("ring write");
    }

    /// The next window's staged SWA visibility and attention output.
    fn next_window(
        gpu: &mut Gpu,
        ring: &GpuTensor,
        start: usize,
        rows: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let kv = upload(gpu, &values(7, start, rows, HEAD_DIM));
        let staged = gpu
            .zeros(&[rows * HEAD_DIM * WINDOW], DType::F32)
            .expect("staged");
        let positions = upload_i32(gpu, &[0]);
        execute_swa_stage(
            gpu,
            &SwaStageOp {
                swa_k: ring,
                kv: &kv,
                staged: &staged,
                positions: &positions,
                start_pos: start as u32,
                win: WINDOW,
                head_dim: HEAD_DIM,
                capture_safe: false,
                rows,
                layer: 0,
            },
        )
        .expect("stage");
        let n_valid: Vec<i32> = (0..rows)
            .map(|b| (start + b + 1).min(WINDOW) as i32)
            .collect();
        let n_valid = upload_i32(gpu, &n_valid);
        let q = upload(gpu, &values(8, start, rows, N_HEADS * HEAD_DIM));
        let sink = upload(gpu, &values(9, 0, 1, N_HEADS));
        let out = gpu
            .zeros(&[rows * N_HEADS * HEAD_DIM], DType::F32)
            .expect("out");
        execute_swa_attention(
            gpu,
            &SwaAttentionOp {
                q: &q,
                sink: &sink,
                out: &out,
                n_heads: N_HEADS,
                head_dim: HEAD_DIM,
                n_groups: 1,
                win: WINDOW,
                rows,
                layer: 0,
                route: SwaAttentionRoute::Batched {
                    staged: &staged,
                    n_valid: &n_valid,
                    start_pos: start as u32,
                    per_position: false,
                },
            },
        )
        .expect("attention");
        // Only the first `n_valid` staged columns of each row are defined.
        let staged = download(gpu, &staged);
        let visible = (0..rows)
            .flat_map(|b| {
                let n = (start + b + 1).min(WINDOW);
                let staged = &staged;
                (0..HEAD_DIM).flat_map(move |d| {
                    let base = (b * HEAD_DIM + d) * WINDOW;
                    staged[base..base + n].iter().copied()
                })
            })
            .collect();
        (visible, download(gpu, &out))
    }

    /// Verify `rows` rows at `start`, commit `accept_len` drafts, then run the
    /// next 3-row window; compare against a ring that only held the committed
    /// rows. Returns whether the unrolled-back ring diverged (positive control).
    fn swa_case(gpu: &mut Gpu, start: usize, rows: usize, accept_len: usize) -> bool {
        let mut b = bundle(cfg(0));
        let ring = |gpu: &mut Gpu| gpu.zeros(&[HEAD_DIM * WINDOW], DType::F32).expect("ring");
        let (k, v) = (ring(gpu), ring(gpu));
        let (ref_k, ref_v) = (ring(gpu), ring(gpu));
        let (ctl_k, ctl_v) = (ring(gpu), ring(gpu));
        // History 0..start, written in at most window-row chunks.
        let mut pos = 0;
        while pos < start {
            let n = (start - pos).min(WINDOW);
            let hist = values(1, pos, n, HEAD_DIM);
            for (rk, rv) in [(&k, &v), (&ref_k, &ref_v), (&ctl_k, &ctl_v)] {
                ring_write(gpu, rk, rv, pos, &hist);
            }
            pos += n;
        }
        b.state._attention[0].swa_k = Some(k.shallow_clone());
        b.state._attention[0].swa_v = Some(v.shallow_clone());

        // The window: committed rows carry the true tokens' KV, rejected
        // rows a draft's.
        let committed = values(1, start, accept_len + 1, HEAD_DIM);
        let mut window = committed.clone();
        window.extend(values(
            2,
            start + accept_len + 1,
            rows - accept_len - 1,
            HEAD_DIM,
        ));
        b.state
            .arm_verify_rollback(gpu, &b.config, start, rows, false)
            .expect("arm");
        ring_write(gpu, &k, &v, start, &window);
        ring_write(gpu, &ctl_k, &ctl_v, start, &window);
        ring_write(gpu, &ref_k, &ref_v, start, &committed);
        let block = vec![0u32; rows];
        b.commit_prefix(gpu, &block, accept_len, start, &mut Deepseek4DsparkScratch)
            .expect("commit_prefix");
        assert_eq!(b.state.n_tokens, (start + accept_len + 1) as u64);

        let next = start + accept_len + 1;
        let (got_staged, got_out) = next_window(gpu, &k, next, 3);
        let (want_staged, want_out) = next_window(gpu, &ref_k, next, 3);
        let (ctl_staged, ctl_out) = next_window(gpu, &ctl_k, next, 3);
        let case = format!("start={start} rows={rows} accept_len={accept_len}");
        assert_eq!(
            bits(&got_staged),
            bits(&want_staged),
            "staged SWA input, {case}"
        );
        assert_eq!(
            bits(&got_out),
            bits(&want_out),
            "SWA attention output, {case}"
        );
        // The V ring is restored too: every slot the next window can see.
        let visible = |t: &GpuTensor, gpu: &Gpu| -> Vec<u32> {
            let ring = download(gpu, t);
            let slots: Vec<usize> = (next.saturating_sub(WINDOW - 1)..next)
                .map(|p| p % WINDOW)
                .collect();
            (0..HEAD_DIM)
                .flat_map(|d| slots.iter().map(move |&s| (d, s)))
                .map(|(d, s)| ring[d * WINDOW + s].to_bits())
                .collect()
        };
        assert_eq!(visible(&v, gpu), visible(&ref_v, gpu), "V ring, {case}");
        bits(&ctl_staged) != bits(&want_staged) || bits(&ctl_out) != bits(&want_out)
    }

    #[test]
    fn swa_rollback_restores_history_across_the_window_wrap() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        // Seed at W-2 with the first draft rejected: row 2 writes position W
        // into the slot of position 0, which the next seed still attends.
        assert!(
            swa_case(&mut gpu, WINDOW - 2, 4, 0),
            "positive control: W-2, 3 drafts"
        );
        // Rows 3.. clobber positions 0..2; row 2 is the harmless a+1 row.
        assert!(
            swa_case(&mut gpu, WINDOW - 3, 6, 1),
            "positive control: W-3, 5 drafts"
        );
        // Long after the wrap, with the rejected slots themselves wrapping.
        assert!(
            swa_case(&mut gpu, 2 * WINDOW - 4, 6, 2),
            "positive control: 2W-4"
        );
        // Before the wrap a rejected row's slot holds no history.
        assert!(!swa_case(&mut gpu, WINDOW - 8, 4, 0), "before the wrap");
        // Full accept leaves nothing to restore.
        assert!(!swa_case(&mut gpu, WINDOW - 2, 4, 3), "full accept");
    }

    // ── Compressor rings ───────────────────────────────────────────────

    struct Compressor {
        ratio: usize,
        proj: usize,
        ape: GpuTensor,
        norm: GpuTensor,
    }

    struct Rings {
        kv_state: GpuTensor,
        score_state: GpuTensor,
        cache: GpuTensor,
        concat_kv: GpuTensor,
        concat_score: GpuTensor,
    }

    const CACHE_ROWS: usize = 64;

    impl Compressor {
        fn new(gpu: &mut Gpu, ratio: usize) -> Self {
            let proj = if ratio == 4 { 2 * HEAD_DIM } else { HEAD_DIM };
            Self {
                ratio,
                proj,
                ape: upload(gpu, &values(5, 0, ratio, proj)),
                norm: upload(gpu, &values(6, 0, 1, HEAD_DIM)),
            }
        }

        fn rings(&self, gpu: &mut Gpu) -> Rings {
            let state_rows = if self.ratio == 4 {
                2 * self.ratio
            } else {
                self.ratio
            };
            let n = state_rows * self.proj;
            let score_state = gpu.zeros(&[n], DType::F32).expect("score ring");
            gpu.fill_f32(&score_state, f32::NEG_INFINITY).expect("fill");
            Rings {
                kv_state: gpu.zeros(&[n], DType::F32).expect("kv ring"),
                score_state,
                cache: gpu
                    .zeros(&[CACHE_ROWS * HEAD_DIM], DType::F32)
                    .expect("cache"),
                concat_kv: gpu
                    .zeros(&[2 * self.ratio * HEAD_DIM], DType::F32)
                    .expect("concat"),
                concat_score: gpu
                    .zeros(&[2 * self.ratio * HEAD_DIM], DType::F32)
                    .expect("concat"),
            }
        }

        /// The production compressor ops over `kv`/`score` rows at `start`:
        /// the decode route one row at a time (host commit), or the batched
        /// aligned window. The last op stashes when `stash` is given.
        fn run(
            &self,
            gpu: &mut Gpu,
            rings: &Rings,
            start: usize,
            kv: &[f32],
            score: &[f32],
            batched: bool,
            mut stash: Option<CompressorStash<'_>>,
        ) {
            let (ratio, proj) = (self.ratio, self.proj);
            let n_rows = kv.len() / proj;
            let kv = upload(gpu, kv);
            let score = upload(gpu, score);
            let comp_positions = upload_i32(gpu, &[0; 16]);
            let positions: Vec<GpuTensor> = (0..n_rows)
                .map(|b| upload_i32(gpu, &[(start + b) as i32]))
                .collect();
            let rope = CompressorRope {
                freq_base: 10000.0,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: 1.0,
                corr_low: 0.0,
                corr_high: 0.0,
                rope_dim: 64,
            };
            let concat = (ratio == 4).then_some((&rings.concat_kv, &rings.concat_score));
            let op = |rows, route, stash| KvCompressOp {
                rows,
                ape: &self.ape,
                norm: &self.norm,
                kv_state: &rings.kv_state,
                score_state: &rings.score_state,
                kv_cache: &rings.cache,
                staging: None,
                rope,
                ratio,
                head_dim: HEAD_DIM,
                eps: 1e-6,
                indexer: false,
                layer: 0,
                route,
                stash,
            };
            if batched {
                assert_eq!(start % ratio, 0, "the batched window starts aligned");
                let events = if ratio - 1 < n_rows {
                    (n_rows - (ratio - 1)).div_ceil(ratio)
                } else {
                    0
                };
                let event_positions: Vec<i32> =
                    (0..events).map(|e| (start + e * ratio) as i32).collect();
                let route = KvCompressRoute::Batched {
                    kv: &kv,
                    score: &score,
                    comp_positions: &comp_positions,
                    start_pos: start as u32,
                    events,
                    local_row: Some(start / ratio),
                    event_positions,
                };
                execute_kv_compress(gpu, &op(n_rows, route, stash)).expect("batched compress");
                return;
            }
            for b in 0..n_rows {
                let pos = start + b;
                let pos_t = &positions[b];
                let route = KvCompressRoute::Decode {
                    input: CompressorInput {
                        kv: &kv,
                        score: &score,
                        row: b,
                    },
                    concat,
                    attn_state: None,
                    pos_array: pos_t,
                    pos_index: 0,
                    commit: CompressorCommit::Host {
                        ring_row: if ratio == 4 {
                            ratio + pos % ratio
                        } else {
                            pos % ratio
                        },
                        commit: (pos + 1).is_multiple_of(ratio).then_some(Some(pos / ratio)),
                    },
                    host_ape_row: Some(pos % ratio),
                    gfx1151_route: false,
                    debug: None,
                };
                let stash = if b + 1 == n_rows { stash.take() } else { None };
                execute_kv_compress(gpu, &op(1, route, stash)).expect("row compress");
            }
        }

        fn snapshot(&self, gpu: &Gpu, rings: &Rings) -> [Vec<u32>; 3] {
            [&rings.kv_state, &rings.score_state, &rings.cache].map(|t| bits(&download(gpu, t)))
        }
    }

    /// History `0..start`, a `rows`-row verify window (decode route, or the
    /// batched aligned route), commit of `accept_len` drafts, then decode
    /// through one window past the verify's last close. The rings and the
    /// compressed cache must equal a run that only ever held the committed
    /// rows. Returns whether the run without the rollback diverged.
    fn compressor_case(
        gpu: &mut Gpu,
        ratio: usize,
        start: usize,
        rows: usize,
        accept_len: usize,
        batched: bool,
    ) -> bool {
        let comp = Compressor::new(gpu, ratio);
        let proj = comp.proj;
        let mut b = bundle(cfg(ratio as u32));
        let spec = comp.rings(gpu);
        let (want, ctl) = (comp.rings(gpu), comp.rings(gpu));
        let true_rows = |first, count| {
            (
                values(11, first, count, proj),
                values(12, first, count, proj),
            )
        };
        let (hist_kv, hist_score) = true_rows(0, start);
        for rings in [&spec, &want, &ctl] {
            comp.run(gpu, rings, 0, &hist_kv, &hist_score, false, None);
        }
        let layer = &mut b.state._indexer[0];
        layer.main_kv_state = Some(spec.kv_state.shallow_clone());
        layer.main_score_state = Some(spec.score_state.shallow_clone());

        let committed = start + accept_len;
        let (mut kv, mut score) = true_rows(start, accept_len + 1);
        let (want_kv, want_score) = (kv.clone(), score.clone());
        kv.extend(values(13, committed + 1, rows - accept_len - 1, proj));
        score.extend(values(14, committed + 1, rows - accept_len - 1, proj));
        b.state
            .arm_verify_rollback(gpu, &b.config, start, rows, false)
            .expect("arm");
        let stash = b
            .state
            .verify_rollback
            .compressor_stash(0, false, start as u32, rows);
        assert!(stash.is_some(), "a window with a close on a draft row arms");
        comp.run(gpu, &spec, start, &kv, &score, batched, stash);
        comp.run(gpu, &ctl, start, &kv, &score, batched, None);
        comp.run(gpu, &want, start, &want_kv, &want_score, false, None);
        b.commit_prefix(
            gpu,
            &vec![0u32; rows],
            accept_len,
            start,
            &mut Deepseek4DsparkScratch,
        )
        .expect("commit_prefix");

        // Decode on through the window after the verify's last close.
        let last_close = (start + rows).div_ceil(ratio) * ratio - 1;
        let (next_kv, next_score) = true_rows(committed + 1, last_close + ratio - committed);
        for rings in [&spec, &want, &ctl] {
            comp.run(
                gpu,
                rings,
                committed + 1,
                &next_kv,
                &next_score,
                false,
                None,
            );
        }
        let case = format!(
            "ratio={ratio} start={start} rows={rows} accept_len={accept_len} batched={batched}"
        );
        let [got, want, ctl] = [&spec, &want, &ctl].map(|r| comp.snapshot(gpu, r));
        for (i, what) in ["kv ring", "score ring", "compressed cache"]
            .iter()
            .enumerate()
        {
            assert_eq!(got[i], want[i], "{what}, {case}");
        }
        ctl != want
    }

    #[test]
    fn compressor_rollback_rebuilds_rings_after_a_rejected_close() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        // Ratio 4, decode route: draft 2 (position 7) closes window 1 and
        // shifts the overlap ring; draft 3 then overwrites position 4.
        assert!(
            compressor_case(&mut gpu, 4, 5, 4, 0, false),
            "positive control: r4 5+4 a0"
        );
        // Ratio 4, batched aligned route: the close pools rows 4..7 and the
        // committed rows 4, 5 never reach the open-window half.
        assert!(
            compressor_case(&mut gpu, 4, 4, 4, 1, true),
            "positive control: r4 4+4 a1"
        );
        // Two closes, the first committed (shift kept), the second rejected.
        assert!(
            compressor_case(&mut gpu, 4, 2, 6, 2, false),
            "positive control: r4 2+6 a2"
        );
        // Ratio 128: the rejected close at 127, then position 128 overwrites
        // committed position 0's slot.
        assert!(
            compressor_case(&mut gpu, 128, 126, 4, 0, false),
            "positive control: r128"
        );
        // Full accept leaves the forward's state as is.
        assert!(!compressor_case(&mut gpu, 4, 5, 4, 3, false), "full accept");
    }
}
