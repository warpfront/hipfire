// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5/3.6-VL image prompts on the batched (chunked) prefill path.
//!
//! An image request differs from a text request in exactly two places:
//!
//! 1. the `<|image_pad|>` rows carry the vision tower's merged embeddings
//!    instead of the token-embedding row of `<|image_pad|>`;
//! 2. rotary phases are 3D `(t, h, w)`. Text rows use the same value on all
//!    three axes (= plain 1D RoPE), but rows after the image sit at
//!    `slot + rope_delta`, not at `slot`.
//!
//! The per-token VL body (`forward_scratch_{embed_,}mrope`) handles both but
//! runs ~40 tok/s. This module lets the ordinary `forward_prefill_batch`
//! chunks take the same two inputs:
//!
//! * [`splice_image_rows`] copies the visual embeddings over the image rows of
//!   `pbs.x_batch` right after the token-embedding step;
//! * [`stage_rope_rows`] uploads the chunk's `[n][3]` absolute phase rows into
//!   `pbs.pos3` and returns the mrope section, which the chunk puts in
//!   [`hipfire_dispatch::pipeline::batched::BatchScratch::mrope`]; the batched
//!   full-attention prep then bypasses every fused prep that bakes 1D rope
//!   (gfx12 fused/fp8-Q prep, gfx1100/gfx1151 fused prep, F5 PM Q-prep) and
//!   rotates with `rope_mrope_halfsplit_f32_batched`, the batched twin of the
//!   `rope_mrope_halfsplit_f32` kernel the per-token mrope arm dispatches.
//!
//! # Numerics contract
//!
//! Phases are NOT shifted: every prompt row at KV slot `p` rotates at
//! `MropeCtx::pos3(p)` (+ `kv_cache.compact_offset` as the kernel's
//! `pos_offset`, exactly like `upload_mrope_pos3` in the per-token arm), and
//! decode continues on `forward_scratch_mrope` with the same `MropeCtx`
//! (`pos3` falls back to `p + rope_delta` past the prompt). The batched prompt
//! therefore uses the same phases and the same embeddings as the per-token
//! path. Differences come only from the batched kernels (WMMA GEMM, chunked
//! DeltaNet scan, batched attention), exactly the contract text prefill
//! already has against per-token decode. Greedy output may differ from
//! `HIPFIRE_VL_PREFILL=token` in late tokens; that is the stated contract,
//! not a bug.
//!
//! # Threading / lifetime
//!
//! The context is a thread-local installed by [`install`] for the duration of
//! one VL prefill (the daemon runs one generation per GPU thread). It is keyed
//! by absolute KV slots, so forwards that do not overlap the installed prompt
//! range never see it. Routes that cannot honour it refuse overlapping rows
//! with a `HipError` (see [`refuse_overlap`]) instead of silently computing
//! text semantics. Device memory is freed by [`VlPrefillGuard::release`]; the
//! guard's `Drop` only clears the thread-local so a forgotten release cannot
//! leave a stale scope that corrupts a later text prefill.

use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;

use super::config::{MropeCtx, Qwen35Config};
use super::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
use hipfire_runtime::llama;

/// One contiguous run of `<|image_pad|>` rows in absolute KV slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Absolute KV slot of the first row of the run.
    pub slot_start: usize,
    /// Rows in the run.
    pub len: usize,
    /// Row of the visual-embedding matrix the run starts at.
    pub emb_row_start: usize,
}

/// `dst_row` rows of a chunk (chunk-local) are overwritten with
/// `rows` embedding rows starting at `emb_row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCopy {
    pub dst_row: usize,
    pub emb_row: usize,
    pub rows: usize,
}

/// Image-pad runs of `prompt_tokens` (which starts at KV slot `prompt_start`),
/// assigning embedding rows sequentially across runs and stopping once
/// `n_visual` rows are consumed — the same assignment the per-token loop makes
/// (`token == image_pad_id && visual_idx < n_visual`).
pub fn image_spans(
    prompt_tokens: &[u32],
    image_pad_id: u32,
    prompt_start: usize,
    n_visual: usize,
) -> Vec<ImageSpan> {
    let mut spans: Vec<ImageSpan> = Vec::new();
    let mut used = 0usize;
    let mut i = 0usize;
    while i < prompt_tokens.len() && used < n_visual {
        if prompt_tokens[i] != image_pad_id {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < prompt_tokens.len()
            && prompt_tokens[i] == image_pad_id
            && used + (i - run_start) < n_visual
        {
            i += 1;
        }
        let len = i - run_start;
        spans.push(ImageSpan {
            slot_start: prompt_start + run_start,
            len,
            emb_row_start: used,
        });
        used += len;
    }
    spans
}

/// Pure geometry of one VL prompt: where the image rows and the explicit
/// phase table live. No GPU state; everything the unit tests exercise.
#[derive(Debug, Clone)]
pub struct VlPlan {
    spans: Vec<ImageSpan>,
    /// Absolute KV slot of the first prompt row.
    prompt_start: usize,
    /// Prompt rows.
    prompt_len: usize,
    /// Absolute phases for the prompt rows (`None` = 1D slot phases).
    mrope: Option<MropeCtx>,
}

impl VlPlan {
    pub fn new(
        spans: Vec<ImageSpan>,
        prompt_start: usize,
        prompt_len: usize,
        mrope: Option<MropeCtx>,
    ) -> Result<Self, String> {
        let prompt_end = prompt_start
            .checked_add(prompt_len)
            .ok_or("VL prefill: prompt range overflows")?;
        let mut prev_end = prompt_start;
        for s in &spans {
            let end = s
                .slot_start
                .checked_add(s.len)
                .ok_or("VL prefill: image span overflows")?;
            if s.len == 0 || s.slot_start < prev_end || end > prompt_end {
                return Err(format!(
                    "VL prefill: image span [{}, {end}) is empty, overlaps a previous span or leaves the prompt [{prompt_start}, {prompt_end})",
                    s.slot_start
                ));
            }
            prev_end = end;
        }
        if let Some(mc) = mrope.as_ref() {
            if mc.base != prompt_start {
                return Err(format!(
                    "VL prefill: mrope base {} != prompt start {prompt_start}",
                    mc.base
                ));
            }
        }
        Ok(Self {
            spans,
            prompt_start,
            prompt_len,
            mrope,
        })
    }

    /// Total embedding rows the spans address (`max(emb_row_start + len)`).
    pub fn embed_rows_needed(&self) -> usize {
        self.spans
            .iter()
            .map(|s| s.emb_row_start + s.len)
            .max()
            .unwrap_or(0)
    }

    pub fn prompt_start(&self) -> usize {
        self.prompt_start
    }

    pub fn prompt_end(&self) -> usize {
        self.prompt_start + self.prompt_len
    }

    /// `[start, start+n)` intersects the prompt range.
    pub fn in_prompt(&self, start: usize, n: usize) -> bool {
        n > 0 && start < self.prompt_end() && start.saturating_add(n) > self.prompt_start
    }

    /// Embedding rows the chunk `[start, start+n)` must have spliced in.
    pub fn splice_ranges(&self, start: usize, n: usize) -> Vec<RowCopy> {
        let end = start.saturating_add(n);
        let mut out = Vec::new();
        for s in &self.spans {
            let lo = start.max(s.slot_start);
            let hi = end.min(s.slot_start + s.len);
            if lo < hi {
                out.push(RowCopy {
                    dst_row: lo - start,
                    emb_row: s.emb_row_start + (lo - s.slot_start),
                    rows: hi - lo,
                });
            }
        }
        out
    }

    pub fn splices(&self, start: usize, n: usize) -> bool {
        let end = start.saturating_add(n);
        self.spans
            .iter()
            .any(|s| start.max(s.slot_start) < end.min(s.slot_start + s.len))
    }

    /// Every chunk inside the prompt needs mrope rows when a phase table is
    /// present: rows after the image are `p + rope_delta`, not `p`.
    pub fn needs_rope(&self, start: usize, n: usize) -> bool {
        self.mrope.is_some() && self.in_prompt(start, n)
    }

    /// The chunk needs any VL handling (splice or explicit phases).
    pub fn overlaps(&self, start: usize, n: usize) -> bool {
        self.splices(start, n) || self.needs_rope(start, n)
    }

    /// Absolute `(t, h, w)` for rows `[start, start+n)`: `MropeCtx::pos3` for
    /// rows at or past the prompt start, slot phases below it (rows of an
    /// earlier turn the table does not model).
    pub fn pos3_rows(&self, start: usize, n: usize) -> Vec<[i32; 3]> {
        let mc = self.mrope.as_ref();
        (start..start + n)
            .map(|p| match mc {
                Some(mc) if p >= mc.base => mc.pos3(p),
                _ => [p as i32; 3],
            })
            .collect()
    }

    pub fn section(&self) -> Option<[usize; 3]> {
        self.mrope.as_ref().map(|m| m.section)
    }
}

/// Facts about the route a chunk is about to take, for the pre-mutation
/// refusal check.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkRoute {
    /// `BatchSemantics::Sequential` (false for independent lanes).
    pub sequential: bool,
    pub tree_verify: bool,
    /// x_batch was filled by the caller (pre-embedded or non-first band).
    pub pre_embedded: bool,
    /// Inputs were uploaded by a captured-prefill caller.
    pub pre_uploaded: bool,
    /// Multi-GPU band view (`band.is_some()`).
    pub banded: bool,
    pub mask_override: bool,
    /// HIP graph capture or Redline recording is active.
    pub capturing: bool,
    /// Per-token full-attention fallback would run.
    pub fa_unbatched: bool,
    /// A MoE full-attention layer is in the stack.
    pub moe_fa: bool,
}

impl ChunkRoute {
    /// Why a chunk overlapping a VL prompt cannot take this route.
    pub fn refusal(&self) -> Option<&'static str> {
        if !self.sequential {
            Some("independent-lane batch")
        } else if self.tree_verify {
            Some("tree verify")
        } else if self.pre_embedded {
            Some("pre-embedded / non-first-band batch")
        } else if self.pre_uploaded {
            Some("captured prefill (pre-uploaded inputs)")
        } else if self.banded {
            Some("multi-GPU band")
        } else if self.mask_override {
            Some("mask-embed override")
        } else if self.capturing {
            Some("graph capture / Redline recording")
        } else if self.fa_unbatched {
            Some("per-token full-attention fallback")
        } else if self.moe_fa {
            Some("MoE full-attention layers")
        } else {
            None
        }
    }
}

struct Active {
    plan: VlPlan,
    /// Device F32 `[n_visual, dim]`, uploaded once.
    embeds: GpuTensor,
    dim: usize,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Inputs of one image request's batched prefill.
pub struct VlPrefillInputs<'a> {
    pub spans: Vec<ImageSpan>,
    /// Absolute KV slot of the first prompt row (`m.seq_pos`).
    pub prompt_start: usize,
    pub prompt_len: usize,
    /// Merged visual embeddings, `[n_visual][dim]` f32.
    pub embeds: &'a [f32],
    /// Model width (`config.dim`).
    pub dim: usize,
    /// The per-request phase table (`None` = 1D phases, splice only).
    pub mrope: Option<&'a MropeCtx>,
}

/// RAII handle of the installed scope. [`VlPrefillGuard::release`] frees the
/// device embeddings; `Drop` only uninstalls the thread-local.
pub struct VlPrefillGuard {
    _private: (),
}

impl VlPrefillGuard {
    /// Uninstall the scope and free its device memory.
    pub fn release(self, gpu: &mut Gpu) -> HipResult<()> {
        let old = ACTIVE.with(|a| a.borrow_mut().take());
        std::mem::forget(self);
        match old {
            Some(Active { embeds, .. }) => gpu.free_tensor(embeds),
            None => Ok(()),
        }
    }
}

impl Drop for VlPrefillGuard {
    fn drop(&mut self) {
        // Forgotten `release`: drop the scope so later text prefills are
        // unaffected. The device embeddings leak until process exit.
        let _ = ACTIVE.with(|a| a.borrow_mut().take());
    }
}

/// Install the request's VL prefill context (uploads the embeddings once).
/// Errors if a scope is already installed on this thread.
pub fn install(gpu: &mut Gpu, inputs: VlPrefillInputs<'_>) -> HipResult<VlPrefillGuard> {
    if is_active() {
        return Err(HipError::new(0, "VL prefill: a scope is already installed"));
    }
    let VlPrefillInputs {
        spans,
        prompt_start,
        prompt_len,
        embeds,
        dim,
        mrope,
    } = inputs;
    if dim == 0 || embeds.len() % dim != 0 {
        return Err(HipError::new(
            0,
            "VL prefill: embedding buffer is not a whole number of rows",
        ));
    }
    let plan = VlPlan::new(spans, prompt_start, prompt_len, mrope.cloned())
        .map_err(|e| HipError::new(0, &e))?;
    let n_visual = embeds.len() / dim;
    if plan.embed_rows_needed() > n_visual {
        return Err(HipError::new(
            0,
            &format!(
                "VL prefill: spans address {} embedding rows but only {n_visual} exist",
                plan.embed_rows_needed()
            ),
        ));
    }
    let tensor = if n_visual > 0 {
        gpu.upload_f32(embeds, &[n_visual, dim])?
    } else {
        gpu.alloc_tensor(&[dim.max(1)], DType::F32)?
    };
    ACTIVE.with(|a| {
        *a.borrow_mut() = Some(Active {
            plan,
            embeds: tensor,
            dim,
        })
    });
    Ok(VlPrefillGuard { _private: () })
}

/// Whether a VL prefill context is installed on this thread.
pub fn is_active() -> bool {
    ACTIVE.with(|a| a.borrow().is_some())
}

/// Rows `[start, start+n)` need VL handling (image embeddings or explicit
/// phases). `false` without an installed scope.
pub fn overlaps(start: usize, n: usize) -> bool {
    ACTIVE.with(|a| {
        a.borrow()
            .as_ref()
            .is_some_and(|a| a.plan.overlaps(start, n))
    })
}

/// Error when a route that cannot splice image embeddings / mrope phases
/// would process VL rows.
pub fn refuse_overlap(start: usize, n: usize, site: &str) -> HipResult<()> {
    if overlaps(start, n) {
        return Err(HipError::new(
            0,
            &format!(
                "VL prefill rows [{start}, {}) reached the {site} route, which cannot splice image embeddings / mrope phases",
                start.saturating_add(n)
            ),
        ));
    }
    Ok(())
}

/// Error whenever a scope is installed, for entry points that take no
/// `start`/`n` window (multi-GPU / multi-request lanes).
pub fn refuse_active(site: &str) -> HipResult<()> {
    if is_active() {
        return Err(HipError::new(
            0,
            &format!("VL prefill scope is installed but the {site} route cannot honour it"),
        ));
    }
    Ok(())
}

/// Pre-mutation route check for one chunk `[start, start+n)`.
pub fn check_chunk_route(start: usize, n: usize, route: &ChunkRoute) -> HipResult<()> {
    if !overlaps(start, n) {
        return Ok(());
    }
    match route.refusal() {
        Some(why) => Err(HipError::new(
            0,
            &format!(
                "VL prefill rows [{start}, {}) cannot take this chunk route: {why}",
                start.saturating_add(n)
            ),
        )),
        None => Ok(()),
    }
}

/// Overwrite the image rows of a chunk's `x_batch` (rows `[start, start+n)`,
/// row stride `dim` F32) with the visual embeddings. No-op without a scope or
/// overlap.
pub(crate) fn splice_image_rows(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    start: usize,
    n: usize,
) -> HipResult<()> {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(a) = a.as_ref() else {
            return Ok(());
        };
        let row_bytes = a.dim * 4;
        if x_batch.buf.size() < n * row_bytes {
            return Err(HipError::new(
                0,
                "VL prefill: x_batch is smaller than the chunk",
            ));
        }
        for c in a.plan.splice_ranges(start, n) {
            gpu.memcpy_dtod_at_auto(
                &x_batch.buf,
                c.dst_row * row_bytes,
                &a.embeds.buf,
                c.emb_row * row_bytes,
                c.rows * row_bytes,
            )?;
        }
        Ok(())
    })
}

/// For a chunk that needs explicit phases: upload its `[n][3]` absolute phase
/// rows into `pos3` (a `pbs.pos3`, `>= n*3` i32-in-F32) and return the mrope
/// section. `None` → ordinary 1D slot phases.
pub(crate) fn stage_rope_rows(
    gpu: &mut Gpu,
    pos3: &GpuTensor,
    start: usize,
    n: usize,
) -> HipResult<Option<[usize; 3]>> {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(a) = a.as_ref() else {
            return Ok(None);
        };
        if !a.plan.needs_rope(start, n) {
            return Ok(None);
        }
        let Some(section) = a.plan.section() else {
            return Ok(None);
        };
        if pos3.buf.size() < n * 12 {
            return Err(HipError::new(
                0,
                "VL prefill: pos3 scratch is smaller than the chunk",
            ));
        }
        let rows = a.plan.pos3_rows(start, n);
        let mut bytes = Vec::with_capacity(n * 12);
        for p in &rows {
            for v in p {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
        }
        gpu.memcpy_htod_auto(&pos3.buf, &bytes)?;
        Ok(Some(section))
    })
}

/// Whether the batched prefill can host an image prompt on this model: every
/// layer dense and batch-admissible, a KV tier with batched FA keys, and the
/// ordinary batched-prefill eligibility. Pure (no GPU work). `Err` carries the
/// reason for the `[daemon/vl] prefill:` log line; the caller keeps the
/// per-token VL loop then.
///
/// True only when every chunk of the prompt takes a hooked route:
/// * no MoE layer (the MoE FA prep has no mrope arm);
/// * every layer `qwen35_layer_batch_admissible` (else the per-token FA /
///   whole-call per-token fallback would run);
/// * KV tier in {q8, asym4, asym3, asym2, native fp8} (the tiers with batched
///   write + attend keys; f32 / hfq4 KV falls back to per-token FA);
/// * `prefill_batch_pbs_eligible` at a prompt-sized `n` (batched prefill not
///   disabled by `prefill_batched = false`);
/// * a rotary section the batched mrope kernel can address;
/// * not inside HIP-graph capture or Redline recording.
pub fn vl_batched_prefill_admitted(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
) -> Result<(), String> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err("graph capture / Redline recording active".into());
    }
    if weights.layers.iter().any(|lw| {
        matches!(
            lw,
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        )
    }) {
        return Err("MoE layers".into());
    }
    let arch = gpu.arch.as_str();
    for (i, lw) in weights.layers.iter().enumerate() {
        if let Err(e) = super::prefill::qwen35_layer_batch_admissible(lw, config, arch) {
            return Err(format!("layer {i} not batch-admissible ({e})"));
        }
    }
    if !(kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2
        || kv_cache.quant_fp8)
    {
        return Err("KV tier has no batched attention".into());
    }
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    if n_rot == 0 || n_rot % 2 != 0 || config.mrope_section.iter().sum::<usize>() > n_rot / 2 {
        return Err(format!(
            "mrope section {:?} does not fit n_rot {n_rot}",
            config.mrope_section
        ));
    }
    // Prompt-sized n: the verify-decouple window (n <= 32) must not admit a
    // config that disables batched prefill for real prompts.
    if !super::prefill::prefill_batch_pbs_eligible(weights, config, dn_state, 64, arch, true) {
        return Err("batched prefill not eligible for this model/config".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_one_image(
        prompt_start: usize,
        prompt_len: usize,
        img_start: usize,
        img_len: usize,
        mrope: Option<MropeCtx>,
    ) -> VlPlan {
        VlPlan::new(
            vec![ImageSpan {
                slot_start: img_start,
                len: img_len,
                emb_row_start: 0,
            }],
            prompt_start,
            prompt_len,
            mrope,
        )
        .unwrap()
    }

    /// HF phases for [2 text, 2x2 merged image (4 rows), 3 text] at `base`:
    /// the image spans 2 phase positions for 4 tokens, so the largest phase is
    /// 6 and `rope_delta = max + 1 - n = 7 - 9 = -2`.
    fn mrope_fixture(base: usize) -> MropeCtx {
        let b = base as i32;
        let hf: Vec<[i32; 3]> = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 2],
            [2, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ]
        .into_iter()
        .map(|p| [p[0] + b, p[1] + b, p[2] + b])
        .collect();
        MropeCtx {
            base,
            positions: hf,
            rope_delta: 7 - 9,
            section: [11, 11, 10],
        }
    }

    #[test]
    fn spans_follow_per_token_assignment() {
        let pad = 9u32;
        // text, text, pad pad pad, text, pad pad, text
        let toks = [1, 2, 9, 9, 9, 3, 9, 9, 4];
        let spans = image_spans(&toks, pad, 100, 5);
        assert_eq!(
            spans,
            vec![
                ImageSpan {
                    slot_start: 102,
                    len: 3,
                    emb_row_start: 0
                },
                ImageSpan {
                    slot_start: 106,
                    len: 2,
                    emb_row_start: 3
                },
            ]
        );
        // Fewer embeddings than pads: the tail pads fall back to token rows.
        let spans = image_spans(&toks, pad, 0, 4);
        assert_eq!(
            spans,
            vec![
                ImageSpan {
                    slot_start: 2,
                    len: 3,
                    emb_row_start: 0
                },
                ImageSpan {
                    slot_start: 6,
                    len: 1,
                    emb_row_start: 3
                },
            ]
        );
        assert!(image_spans(&toks, pad, 0, 0).is_empty());
        assert!(image_spans(&[1, 2, 3], pad, 0, 4).is_empty());
    }

    #[test]
    fn splice_chunk_before_inside_after_the_run() {
        // prompt [0, 9), image rows at slots 2..6.
        let p = plan_one_image(0, 9, 2, 4, None);
        // Before.
        assert!(p.splice_ranges(0, 2).is_empty());
        assert!(!p.splices(0, 2));
        // Straddling the start.
        assert_eq!(
            p.splice_ranges(0, 4),
            vec![RowCopy {
                dst_row: 2,
                emb_row: 0,
                rows: 2
            }]
        );
        // Strictly inside.
        assert_eq!(
            p.splice_ranges(3, 2),
            vec![RowCopy {
                dst_row: 0,
                emb_row: 1,
                rows: 2
            }]
        );
        // Straddling the end.
        assert_eq!(
            p.splice_ranges(4, 4),
            vec![RowCopy {
                dst_row: 0,
                emb_row: 2,
                rows: 2
            }]
        );
        // Covering the whole run.
        assert_eq!(
            p.splice_ranges(0, 9),
            vec![RowCopy {
                dst_row: 2,
                emb_row: 0,
                rows: 4
            }]
        );
        // After.
        assert!(p.splice_ranges(6, 3).is_empty());
        assert!(!p.splices(6, 3));
    }

    #[test]
    fn splice_with_nonzero_prompt_start_and_two_runs() {
        let spans = vec![
            ImageSpan {
                slot_start: 12,
                len: 3,
                emb_row_start: 0,
            },
            ImageSpan {
                slot_start: 17,
                len: 2,
                emb_row_start: 3,
            },
        ];
        let p = VlPlan::new(spans, 10, 12, None).unwrap();
        assert_eq!(p.embed_rows_needed(), 5);
        // chunk [14, 20): last row of run 1, both rows of run 2 (slot 19 is past run 2).
        assert_eq!(
            p.splice_ranges(14, 6),
            vec![
                RowCopy {
                    dst_row: 0,
                    emb_row: 2,
                    rows: 1
                },
                RowCopy {
                    dst_row: 3,
                    emb_row: 3,
                    rows: 2
                },
            ]
        );
    }

    #[test]
    fn in_prompt_edges_and_ragged_last_chunk() {
        let p = plan_one_image(10, 9, 12, 4, Some(mrope_fixture(10)));
        assert_eq!((p.prompt_start(), p.prompt_end()), (10, 19));
        assert!(!p.in_prompt(0, 10));
        assert!(p.in_prompt(0, 11));
        assert!(p.in_prompt(18, 1));
        assert!(!p.in_prompt(19, 4));
        assert!(!p.in_prompt(10, 0));
        // Every chunk inside the prompt needs rope rows, image or not.
        assert!(p.needs_rope(10, 2)); // before the image
        assert!(p.needs_rope(16, 3)); // after the image (ragged last chunk)
        assert!(!p.needs_rope(19, 5)); // decode-side rows
        assert!(p.overlaps(16, 3));
        assert!(!p.overlaps(19, 5));
    }

    #[test]
    fn no_phase_table_means_splice_only() {
        let p = plan_one_image(0, 9, 2, 4, None);
        assert!(!p.needs_rope(0, 9));
        assert!(p.section().is_none());
        assert!(p.overlaps(0, 4));
        assert!(!p.overlaps(6, 3)); // after the image: nothing to do
    }

    #[test]
    fn pos3_rows_are_absolute_mrope_phases_not_shifted() {
        let mc = mrope_fixture(0);
        assert_eq!(mc.rope_delta, -2);
        let p = plan_one_image(0, 9, 2, 4, Some(mc.clone()));
        // Chunk inside the image run.
        assert_eq!(p.pos3_rows(3, 3), vec![[2, 2, 3], [2, 3, 2], [2, 3, 3]]);
        // Chunk after the image: HF phases (slot - 2), NOT the slot itself.
        assert_eq!(p.pos3_rows(6, 3), vec![[4, 4, 4], [5, 5, 5], [6, 6, 6]]);
        // Identical to the per-token arm's lookup for every prompt row.
        for s in 0..9 {
            assert_eq!(p.pos3_rows(s, 1)[0], mc.pos3(s));
        }
        // One chunk straddling prompt end falls back to p + rope_delta.
        assert_eq!(
            p.pos3_rows(7, 4),
            vec![
                [5, 5, 5],
                [6, 6, 6],
                [9 - 2, 9 - 2, 9 - 2],
                [10 - 2, 10 - 2, 10 - 2]
            ]
        );
    }

    #[test]
    fn pos3_rows_with_nonzero_base_and_rows_below_it() {
        let p = plan_one_image(10, 9, 12, 4, Some(mrope_fixture(10)));
        // Prompt row 0 at slot 10 has phase [10,10,10]; a row below the base is slot-phased.
        assert_eq!(p.pos3_rows(9, 2), vec![[9, 9, 9], [10, 10, 10]]);
        assert_eq!(p.pos3_rows(13, 2), vec![[12, 12, 13], [12, 13, 12]]);
        // Rows after the image: slot 17 -> HF phase 5 + base 10.
        assert_eq!(p.pos3_rows(17, 2), vec![[15, 15, 15], [16, 16, 16]]);
    }

    #[test]
    fn plan_rejects_bad_geometry() {
        let s = |slot_start, len, emb_row_start| ImageSpan {
            slot_start,
            len,
            emb_row_start,
        };
        assert!(VlPlan::new(vec![s(2, 0, 0)], 0, 9, None).is_err());
        assert!(VlPlan::new(vec![s(2, 9, 0)], 0, 9, None).is_err()); // runs past the prompt
        assert!(VlPlan::new(vec![s(2, 3, 0), s(4, 2, 3)], 0, 9, None).is_err()); // overlap
        assert!(VlPlan::new(vec![s(2, 3, 0)], 5, 9, None).is_err()); // before the prompt
        assert!(VlPlan::new(vec![s(2, 3, 0)], 0, 9, Some(mrope_fixture(1))).is_err()); // base mismatch
        assert!(VlPlan::new(vec![s(2, 3, 0)], 0, 9, Some(mrope_fixture(0))).is_ok());
    }

    #[test]
    fn route_refusals_name_the_first_blocker() {
        let ok = ChunkRoute {
            sequential: true,
            ..Default::default()
        };
        assert_eq!(ok.refusal(), None);
        for (r, why) in [
            (
                ChunkRoute {
                    sequential: false,
                    ..ok
                },
                "independent-lane batch",
            ),
            (
                ChunkRoute {
                    tree_verify: true,
                    ..ok
                },
                "tree verify",
            ),
            (
                ChunkRoute {
                    pre_embedded: true,
                    ..ok
                },
                "pre-embedded / non-first-band batch",
            ),
            (
                ChunkRoute {
                    pre_uploaded: true,
                    ..ok
                },
                "captured prefill (pre-uploaded inputs)",
            ),
            (ChunkRoute { banded: true, ..ok }, "multi-GPU band"),
            (
                ChunkRoute {
                    mask_override: true,
                    ..ok
                },
                "mask-embed override",
            ),
            (
                ChunkRoute {
                    capturing: true,
                    ..ok
                },
                "graph capture / Redline recording",
            ),
            (
                ChunkRoute {
                    fa_unbatched: true,
                    ..ok
                },
                "per-token full-attention fallback",
            ),
            (
                ChunkRoute { moe_fa: true, ..ok },
                "MoE full-attention layers",
            ),
        ] {
            assert_eq!(r.refusal(), Some(why));
        }
    }

    #[test]
    fn no_scope_means_no_overlap_and_no_refusal() {
        assert!(!is_active());
        assert!(!overlaps(0, 4096));
        assert!(refuse_overlap(0, 4096, "test").is_ok());
        assert!(refuse_active("test").is_ok());
        let bad = ChunkRoute::default();
        assert!(check_chunk_route(0, 64, &bad).is_ok());
    }
}
