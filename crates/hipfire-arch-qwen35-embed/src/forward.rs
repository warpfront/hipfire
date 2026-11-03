// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The encoder forward: tokens (+ optional merged image embeds) → 128-d
//! per-token embeddings.
//!
//! Layer loop mirrors HF `Qwen3_5Model` with `is_causal = false`:
//! full-attention layers run **bidirectional**, GDN layers keep their causal
//! recurrence (HF ignores `is_causal` for linear attention), a shared SwiGLU
//! MLP follows every mixer, then the final RMSNorm and the 128-d Dense head.
//! The layers are the typed step program in [`crate::program`]; this module
//! stages host input, sequences the programs and drops the padding rows.

use crate::config::EmbedConfig;
use crate::program::{head_steps, mixer_steps, mlp_steps, EncodeScratch};
use crate::state::GdnEncodeState;
use crate::weights::{EmbedWeights, ImageSpan};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, Step};
use rdna_compute::{Gpu, GpuTensor};

/// Input assembled by the caller (bundle / generate body): token ids, the
/// host hidden-state rows for image positions, and per-token (t, h, w)
/// mRoPE positions.
pub struct EncodeInput<'a> {
    pub ids: &'a [u32],
    /// Merged image embeds `[k, hidden]` (empty for text-only).
    pub image_embeds: &'a [f32],
    pub spans: &'a [ImageSpan],
    /// `[n][3]` (t, h, w) positions — text rows carry `[p, p, p]` (bit-equal
    /// to the 1D rope angles), image rows the 2D grid phases.
    pub pos3: &'a [[i32; 3]],
}

/// Env-gated (`HIPFIRE_EMBED_DEBUG=1`) finite-value trace of the residual
/// stream: prints absmax and the non-finite count at each stage so a NaN can
/// be localized to a layer and a mixer kind. Downloads one tensor per call,
/// so this is strictly a debugging path.
fn trace(
    gpu: &mut Gpu,
    label: impl FnOnce() -> String,
    t: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    if !debug_enabled() {
        return Ok(());
    }
    let label = label();
    let v = gpu
        .download_f32(t)
        .map_err(|e| format!("qwen35-embed: download: {e:?}"))?;
    let mut absmax = 0f32;
    let mut bad = 0usize;
    for &x in &v[..n.min(v.len())] {
        if !x.is_finite() {
            bad += 1;
        } else if x.abs() > absmax {
            absmax = x.abs();
        }
    }
    eprintln!(
        "  [embed-debug] {label}: absmax={absmax:.4} nonfinite={bad}/{}",
        v.len()
    );
    Ok(())
}

fn run(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), String> {
    execute_steps(gpu, ctx, steps).map_err(|e| format!("qwen35-embed: {e}"))
}

/// `HIPFIRE_EMBED_DEBUG=1`, read once (the trace calls are on the hot path;
/// the label closure keeps the `format!` off it when disabled).
fn debug_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_EMBED_DEBUG")
            .ok()
            .as_deref()
            == Some("1")
    });
    *ON
}

/// Run the full encoder and the Dense head; returns `[n, dense_out]`
/// row-major **unmasked, unnormalized** token embeddings (the
/// sentence-transformers modules 2/3 run host-side in [`crate::post`]).
///
/// Resets the GDN state on entry (every encode starts from zero recurrence).
pub fn encode_embeddings(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<Vec<f32>, String> {
    let mut out = encode_embeddings_batch(gpu, w, state, cfg, std::slice::from_ref(input))?;
    Ok(out.pop().expect("one input in, one embedding out"))
}

/// Several sequences in one forward, one `[n_i, dense_out]` row-major block
/// per input (same per-input shape and content as [`encode_embeddings`]).
///
/// The batch is *row-concatenation*, not padding to a common length: each
/// sequence keeps its own real length `n_i` and is padded only to its own
/// [`padded_rows`] multiple. That works because nothing in this architecture
/// couples rows across sequences except the two stateful mixers, and both are
/// run per sequence over a row slice:
///
/// - GatedDeltaNet's conv ring and delta rule carry state along rows, so each
///   segment runs with the layer state reset at its start;
/// - full attention is bidirectional *within* a sequence and must not see
///   another sequence's keys, so each segment gets its own score matrix,
///   softmax over its own rows, and its own `n_valid` bound.
///
/// Everything else (all projections, both norms, the gates, mRoPE, the MLP,
/// the Dense head) is row-independent and runs once over the whole row block.
/// Concretely this means a batch needs neither an attention mask nor a common
/// padded length — the equivalent of a `[B·L, B·L]` score matrix never exists.
///
/// The cost is that the per-sequence parts do not amortize: a batch of 8 short
/// sequences costs about as much attention and GDN work as 8 separate encodes,
/// and saves on the row-independent work (the GEMMs, dominated by the
/// weights). `T` should therefore be capped near where the weight-bound work
/// stops dominating; see the slot engine's batch window.
pub fn encode_embeddings_batch(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
) -> Result<Vec<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Err("qwen35-embed: empty batch".into());
    }
    let segs = segment_layout(inputs)?;
    let t_rows: usize = segs.iter().map(|s| s.rows).sum();
    let s = encode_hidden_segments(gpu, w, state, cfg, inputs, &segs, t_rows)?;
    // Dense head: row-independent, so it runs over the whole row block at once.
    let dense_out = cfg.dense_out_features;
    let head = run(gpu, &DispatchCtx::new(gpu), &head_steps(cfg, w, &s));
    let all = head.and_then(|_| {
        gpu.download_f32(&s.dense)
            .map_err(|e| format!("qwen35-embed: download: {e:?}"))
    });
    s.free_gpu(gpu)?;
    let all = all?;

    // Drop each segment's padding rows: the caller's mask/normalize expects
    // exactly one row per real token.
    let mut out = Vec::with_capacity(inputs.len());
    for (input, seg) in inputs.iter().zip(&segs) {
        let n = input.ids.len();
        let start = seg.row_offset * dense_out;
        out.push(all[start..start + n * dense_out].to_vec());
    }
    Ok(out)
}

/// The backbone only: returns the final-normed hidden states
/// `[n, hidden]` as a device tensor (freed by the caller). Split out so a
/// future pooled/mean variant can reuse everything above the Dense head.
///
/// Unpadded (one segment of `ids.len()` rows) — this is the debug/inspection
/// entry point, not the serving one; use [`encode_embeddings`] for correctness.
pub fn encode_hidden(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<GpuTensor, String> {
    let n = input.ids.len();
    if n == 0 {
        return Err("qwen35-embed: empty input".into());
    }
    let segs = [EncodeSegment {
        row_offset: 0,
        rows: n,
        n_valid: n,
    }];
    encode_hidden_segments(gpu, w, state, cfg, std::slice::from_ref(input), &segs, n)?
        .into_hidden(gpu)
}

/// One sequence's row block inside the concatenated batch: `rows` is its own
/// `padded_rows(n)`, `n_valid` its real token count.
pub use hipfire_dispatch::pipeline::encoder::EncoderSegment as EncodeSegment;

/// Validate the inputs and lay their row blocks out back to back.
fn segment_layout(inputs: &[EncodeInput<'_>]) -> Result<Vec<EncodeSegment>, String> {
    let mut segs = Vec::with_capacity(inputs.len());
    let mut row_offset = 0usize;
    for input in inputs {
        let n = input.ids.len();
        if n == 0 {
            return Err("qwen35-embed: empty input".into());
        }
        if input.pos3.len() != n {
            return Err(format!(
                "qwen35-embed: {} positions for {n} tokens",
                input.pos3.len()
            ));
        }
        let rows = padded_rows(n);
        segs.push(EncodeSegment {
            row_offset,
            rows,
            n_valid: n,
        });
        row_offset += rows;
    }
    Ok(segs)
}

pub(crate) use hipfire_dispatch::pipeline::encoder::padded_rows;

#[allow(clippy::too_many_arguments)]
fn encode_hidden_segments(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
    segs: &[EncodeSegment],
    n: usize,
) -> Result<EncodeScratch, String> {
    let h = cfg.hidden_size;

    // ── prologue: host gather + image splice, one upload ──────────────
    // `gather_hidden` FILLS the buffer it is handed (it clears first), so each
    // segment is gathered into its own vector and padded out to its row block
    // before being appended. The segments then land back to back in
    // `row_offset` order, which is what every slice below assumes — gathering
    // straight into `host_hidden` would let each segment overwrite the
    // previous one's rows. Trailing pos3 rows get position 0: their rope phase
    // is never read (their queries are dropped and their keys are masked).
    let mut host_hidden: Vec<f32> = Vec::with_capacity(n * h);
    let mut pos3_host: Vec<[i32; 3]> = Vec::with_capacity(n);
    for (input, seg) in inputs.iter().zip(segs) {
        let mut rows: Vec<f32> = Vec::new();
        w.gather_hidden(input.ids, input.image_embeds, input.spans, h, &mut rows)?;
        rows.resize(seg.rows * h, 0.0);
        host_hidden.extend_from_slice(&rows);
        pos3_host.extend_from_slice(input.pos3);
        pos3_host.resize(seg.row_offset + seg.rows, [0, 0, 0]);
    }
    let hidden = gpu
        .upload_f32(&host_hidden, &[n * h])
        .map_err(|e| format!("qwen35-embed: upload: {e:?}"))?;
    // mRoPE positions: [n][3] i32 uploaded as raw bits into an F32-typed
    // buffer (the rope kernel's DeviceBuffer contract).
    let pos_bits: Vec<f32> = pos3_host
        .iter()
        .flat_map(|t| t.iter().map(|v| f32::from_bits(*v as u32)))
        .collect();
    let pos3 = match gpu.upload_f32(&pos_bits, &[n * 3]) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(hidden);
            return Err(format!("qwen35-embed: upload positions: {e:?}"));
        }
    };
    let s = EncodeScratch::new(gpu, cfg, n, hidden, pos3)?;
    match run_layers(gpu, w, state, cfg, segs, &s) {
        Ok(()) => Ok(s),
        Err(e) => {
            s.free_gpu(gpu)?;
            Err(e)
        }
    }
}

fn run_layers(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &GdnEncodeState,
    cfg: &EmbedConfig,
    segs: &[EncodeSegment],
    s: &EncodeScratch,
) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    let len = s.hidden.numel();
    trace(gpu, || "embed".into(), &s.hidden, len)?;
    let mut delta = 0usize;
    for (li, layer) in w.layers.iter().enumerate() {
        run(gpu, &ctx, &mixer_steps(cfg, layer, delta, state, s, segs))
            .map_err(|e| format!("{e} (layer {li})"))?;
        if matches!(layer.mixer, crate::weights::MixerWeights::Gdn { .. }) {
            delta += 1;
            for (label, t) in s.gdn_trace_planes() {
                trace(gpu, || format!("L{li} gdn: {label}"), t, t.numel())?;
            }
        }
        trace(gpu, || format!("layer {li} post-mixer"), &s.hidden, len)?;
        run(gpu, &ctx, &mlp_steps(cfg, layer, s)).map_err(|e| format!("{e} (layer {li} mlp)"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // GPU forwards are exercised end-to-end by the daemon `embed` verb and the
    // `batch_parity` example (lab feature); the host-side math this module
    // shares (masks, normalization, scoring) is covered in crate::post. What is
    // testable without a device is the segment layout every slice in the
    // batched forward is derived from — an off-by-one there is a silent
    // cross-sequence read, not a crash.
    use super::{padded_rows, segment_layout, EncodeInput};

    fn input(len: usize) -> (Vec<u32>, Vec<[i32; 3]>) {
        (vec![0; len], (0..len as i32).map(|i| [i, i, i]).collect())
    }

    #[test]
    fn padded_rows_skips_padding_below_the_eager_threshold() {
        // Below 256 the Q-tiled kernel runs and takes any length; at and above
        // it the eager PV GEMM needs a multiple of 64 in its contraction.
        assert_eq!(padded_rows(0), 0);
        assert_eq!(padded_rows(1), 1);
        assert_eq!(padded_rows(255), 255);
        assert_eq!(padded_rows(256), 256);
        assert_eq!(padded_rows(257), 320);
        assert_eq!(padded_rows(302), 320);
        assert_eq!(padded_rows(320), 320);
        assert_eq!(padded_rows(321), 384);
    }

    #[test]
    fn segments_tile_the_row_block_in_order_with_their_own_padding() {
        let owned: Vec<_> = [7usize, 302, 10, 258].iter().map(|n| input(*n)).collect();
        let inputs: Vec<EncodeInput<'_>> = owned
            .iter()
            .map(|(ids, pos3)| EncodeInput {
                ids,
                image_embeds: &[],
                spans: &[],
                pos3,
            })
            .collect();
        let segs = segment_layout(&inputs).expect("layout");

        assert_eq!(segs.len(), 4);
        // Contiguous, in input order, each padded to its own multiple of 64 —
        // never to a common length, which is what keeps a batch from paying for
        // the longest sequence in it.
        assert_eq!(
            segs.iter().map(|s| s.row_offset).collect::<Vec<_>>(),
            vec![0, 7, 327, 337]
        );
        assert_eq!(
            segs.iter().map(|s| s.rows).collect::<Vec<_>>(),
            vec![7, 320, 10, 320]
        );
        assert_eq!(
            segs.iter().map(|s| s.n_valid).collect::<Vec<_>>(),
            vec![7, 302, 10, 258]
        );
        // The last segment's block end is the batch's row count.
        let last = segs.last().expect("non-empty");
        assert_eq!(last.row_offset + last.rows, 657);
    }

    #[test]
    fn an_empty_or_mismatched_input_is_refused() {
        let empty = input(0);
        let empty_in = EncodeInput {
            ids: &empty.0,
            image_embeds: &[],
            spans: &[],
            pos3: &empty.1,
        };
        assert!(segment_layout(&[empty_in]).is_err());

        let four = input(4);
        let pos3: Vec<[i32; 3]> = four.1[..2].to_vec();
        let mismatched = EncodeInput {
            ids: &four.0,
            image_embeds: &[],
            spans: &[],
            pos3: &pos3,
        };
        let err = segment_layout(&[mismatched]).expect_err("refused");
        assert!(err.contains("2 positions for 4 tokens"), "{err}");
    }
}
