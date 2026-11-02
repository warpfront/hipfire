// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `load_embed_gemma_bundle` — the entry point the loader's
//! `EmbeddingGemma2Carrier` calls, plus the encode-side request assembly.
//!
//! Config comes from the source's `metadata_json` (dir: `config.json`
//! contents; HFQ: embedded metadata) with the sentence-transformers
//! sidecars overlaid from the model dir (dir) or the quantizer's folded
//! `sentence_transformers` object (HFQ). The tokenizer loads from
//! `tokenizer.json` / HFQ metadata and is switched to the exact
//! merge-rank SP-BPE path this vocabulary requires (see
//! `Tokenizer::enable_sp_bpe_merges`); BOS/EOS are pinned from the model
//! config, whose ids (2/1) the generic vocab heuristics cannot infer.

use crate::audio::{decode_and_plan, mels, AudioInput, ProcessedAudio};
use crate::audio_tower::load_audio_tower;
use crate::bundle::EmbedGemmaBundle;
use crate::config::{
    parse_config, AudioFrontendConfig, EmbedConfig, InputKind, VideoFrontendConfig,
};
use crate::forward::{encode_embeddings, EncodeInput, MediaRows};
use crate::image::{process_image, ProcessedImage};
use crate::video::process_video;
use crate::vision::{load_vision_tower, VisionTower};
use crate::weights::{layer_type_summary, load_weights};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

/// What every image-carrying request gets when the vision tower is not
/// resident: the checkpoint's tower lives in a separate `.vl` artifact, so
/// this is a load-time gap, not a bad request. Callers prefix their request
/// tag.
pub const NO_VISION_TOWER: &str = "embeddinggemma2: image input but no vision tower is \
     resident — this checkpoint was loaded text-only (no `.vl` sidecar resolved); put the \
     quantizer's `<model>.vl` sidecar beside the checkpoint or set HIPFIRE_VISION_SIDECAR";

/// The audio twin: both towers ride the SAME `.vl` sidecar, so this is the
/// same load-time gap seen from the other modality. Callers prefix their
/// request tag.
pub const NO_AUDIO_TOWER: &str = "embeddinggemma2: audio input but no audio tower is \
     resident — this checkpoint was loaded text-only (no `.vl` sidecar resolved); put the \
     quantizer's `<model>.vl` sidecar beside the checkpoint or set HIPFIRE_VISION_SIDECAR";

/// The video leg has no tower of its own — its frames ride the VISION tower
/// (`Gemma4VisionModel`, one batch element per frame), so a video-carrying
/// request on a text-only load hits the same `.vl` gap seen from the third
/// modality. Callers prefix their request tag.
pub const NO_VIDEO_TOWER: &str = "embeddinggemma2: video input but no vision tower is \
     resident — this checkpoint was loaded text-only (no `.vl` sidecar resolved); video \
     frames ride the image tower, so put the quantizer's `<model>.vl` sidecar beside the \
     checkpoint or set HIPFIRE_VISION_SIDECAR";

/// The literal the tokenizer's `image_token` maps to (id 258880), i.e. the
/// marker one image compiles to. Used to resolve the marker by the
/// tokenizer's own special-token split and to recognize a literal spelling
/// the vocabulary cannot emit.
const IMAGE_MARKER: &str = "<|image|>";
/// Alias spelling some Gemma-3-family processors emit for the same marker.
const IMAGE_MARKER_ALIAS: &str = "<start_of_image>";
/// The audio twin: one `<|audio|>` (id 258881) per clip, expanding to
/// `[boa, audio_token×M, eoa]`.
const AUDIO_MARKER: &str = "<|audio|>";

/// The video twin: one `<|video|>` (id 258884) per VIDEO (not per frame) —
/// HF `validate_inputs` refuses a count mismatch. With `add_timestamps` off
/// (this checkpoint), HF's `replace_video_token` expands it to ONE
/// `[boi, video_token×S, eoi]` block per frame, back to back with no
/// separators, `S` = the frame's own soft-token count.
const VIDEO_MARKER: &str = "<|video|>";

/// `sentence_bert_config.json` / `config_sentence_transformers.json`
/// overlays, plus the audio frontend geometry from
/// `processor_config.json`. Any of them missing is fine (the defaults
/// stand); a processor config that declares something this implementation
/// cannot express is a load error.
fn overlay_sidecars(cfg: &mut EmbedConfig, dir: &std::path::Path) -> Result<(), String> {
    let read_json = |rel: &str| -> Option<serde_json::Value> {
        let p = dir.join(rel);
        let s = std::fs::read_to_string(p).ok()?;
        serde_json::from_str(&s).ok()
    };
    if let Some(sb) = read_json("sentence_bert_config.json") {
        cfg.apply_sbert_config(&sb);
    }
    if let Some(st) = read_json("config_sentence_transformers.json") {
        cfg.apply_st_config(&st);
    }
    if let Some(pc) = read_json("processor_config.json") {
        cfg.audio_frontend = AudioFrontendConfig::from_processor_json(&pc)?;
        cfg.video_frontend = VideoFrontendConfig::from_processor_json(&pc)?;
    }
    Ok(())
}

/// Apply the `sentence_transformers` object the quantizer folds into an
/// `.hfq`'s metadata (same key schema [`overlay_sidecars`] reads).
fn apply_folded_sidecars(cfg: &mut EmbedConfig, folded: &serde_json::Value) {
    cfg.apply_st_config(folded);
}

/// Load + fix up the tokenizer for either source kind.
fn load_tokenizer(st: &dyn ModelSourceTrait, cfg: &EmbedConfig) -> Result<Tokenizer, String> {
    let mut tok = match st.tokenizer_json_path() {
        Some(p) => Tokenizer::from_tokenizer_json(&p)
            .map_err(|e| format!("embeddinggemma2: tokenizer: {e}"))?
            .ok_or("embeddinggemma2: tokenizer.json present but unparseable")?,
        None => Tokenizer::from_hfq_metadata(st.metadata_json())
            .map_err(|e| format!("embeddinggemma2: tokenizer from metadata: {e}"))?,
    };
    // This vocabulary is a ▁-style BPE model, and the generic SentencePiece
    // greedy-longest path is not merge-rank BPE: on ordinary text it takes a
    // longer literal piece than the merge ranks form, so it diverges from HF
    // `tokenizers` (measured 8/29 sampled prompts, and 0/9 of this arch's
    // parity fixtures). Switch to the exact merge-rank path with byte
    // fallback. Both source kinds (dir and `.hfq`) reach this through here —
    // nothing else enables it — and the guard below fails the load outright
    // rather than let an arch-18 encode fall back to greedy.
    tok.enable_sp_bpe_merges();
    // A Gemma-shaped tokenizer.json with an EMPTY merge list would silently
    // keep the greedy SentencePiece path — the exact HF divergence this arch
    // exists to avoid — so fail the load instead.
    if !tok.sp_bpe_merges_enabled() {
        return Err(
            "embeddinggemma2: this tokenizer did not engage the merge-rank SP-BPE \
             path (no merge rules in tokenizer.json?) — the greedy fallback \
             diverges from HF on this vocabulary, refusing to load"
                .into(),
        );
    }
    // The generic bos/eos heuristics (`<|endoftext|>` / `<s>`) do not know
    // Gemma's <bos>=2 / <eos>=1; the model config's ids are authoritative.
    tok.bos_id = cfg.bos_token_id;
    tok.eos_id = cfg.eos_token_id;
    // The HF post-processor template adds <bos> … <eos>; the runtime's
    // generic add_bos would double it because assembly below adds both
    // explicitly.
    tok.add_bos = false;
    // The invariant that matters: no token id the tokenizer can EMIT may
    // exceed the embedding table (`vocab_size()` itself over-reports — the
    // runtime pads its vocab vector past the last real id).
    if let Some(max_id) = tok.max_token_id() {
        if max_id as usize >= cfg.vocab_size {
            return Err(format!(
                "embeddinggemma2: tokenizer can emit id {max_id} >= config vocab {} — \
                 that token would have no embedding row",
                cfg.vocab_size
            ));
        }
    }
    Ok(tok)
}

pub fn load_embed_gemma_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx<'_>,
) -> Result<EmbedGemmaBundle, String> {
    // Sidecar policy (the documented `vision.mode` ladder): an explicit
    // `params.vision` / `HIPFIRE_VISION_SIDECAR` path from admission wins;
    // otherwise `off` resolves nothing (text-only, no tower VRAM) while
    // `auto`/`on` discover the `<stem>.vl` sibling (HIPFIRE_VL_FILE first).
    let vision_path = match &ctx.vision_path {
        Some(p) => Some(p.clone()),
        None if ctx.vision_mode != "off" => hipfire_runtime::sidecar::resolve_vl_sidecar(ctx.path),
        None => None,
    };
    load_embed_gemma_with_vision(ctx.gpu, src, vision_path.as_deref())
}

/// The loader body without [`LoadCtx`] (the slot engine calls the
/// path-explicit variant directly; it owns its own GPU and has no load
/// context). Discovery is unconditional here — there is no `vision_mode` on
/// this entry — so the caller decides by passing (or not passing) a path.
pub fn load_embed_gemma_from_source(
    gpu: &mut rdna_compute::Gpu,
    src: ModelSource,
) -> Result<EmbedGemmaBundle, String> {
    let vision_path = source_path(&src)
        .and_then(|p| hipfire_runtime::sidecar::resolve_vl_sidecar(&p.to_string_lossy()));
    load_embed_gemma_with_vision(gpu, src, vision_path.as_deref())
}

/// The path a source was opened from (for sidecar discovery).
fn source_path(src: &ModelSource) -> Option<PathBuf> {
    let st: &dyn ModelSourceTrait = match src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    Some(st.path().to_path_buf())
}

/// The path-explicit loader body: `Some(path)` loads BOTH towers from that
/// `.vl` container (one sidecar carries `vision_tower.*`, `audio_tower.*`,
/// `embed_vision.*` and `embed_audio.*`), `None` loads text-only.
///
/// Kept under this name for its callers; `sidecar_path` is what it means.
pub fn load_embed_gemma_with_vision(
    gpu: &mut rdna_compute::Gpu,
    src: ModelSource,
    sidecar_path: Option<&Path>,
) -> Result<EmbedGemmaBundle, String> {
    let st: &dyn ModelSourceTrait = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    let meta: serde_json::Value = serde_json::from_str(st.metadata_json())
        .map_err(|e| format!("embeddinggemma2: config JSON: {e}"))?;
    let mut config = parse_config(crate::config::config_from_metadata(&meta))?;

    if src.is_dir() {
        overlay_sidecars(&mut config, st.path())?;
    } else if let Some(folded) = meta.get("sentence_transformers") {
        apply_folded_sidecars(&mut config, folded);
    }

    let tokenizer = load_tokenizer(st, &config)?;

    // Screen every container this load will decode, both sources, before any
    // GPU work: the host decode path has no arm for the GGUF/Maple-side Q4K
    // (qt=4) or the typed-Qwen4 qt=53, and the shared decoder's response to an
    // unknown type is a `panic!` — which would take the daemon and any resident
    // model down mid-load. See `refuse_unsupported_quant_types`.
    crate::weights::refuse_unsupported_quant_types(&src)?;
    if let Some(p) = sidecar_path {
        match ModelSource::from_path(&p.to_string_lossy()) {
            Ok(vsrc) => crate::weights::refuse_unsupported_quant_types(&vsrc)?,
            Err(e) => {
                return Err(format!(
                    "embeddinggemma2: open sidecar {}: {e}",
                    p.display()
                ))
            }
        }
    }

    // PLE tensors are required exactly when the config declares a PLE width
    // — a checkpoint/config mismatch is a load error, not a silent
    // zero-filled PLE.
    let has_ple = crate::weights::EmbedWeights::source_has_ple(&src);
    if (config.ple_dim > 0) != has_ple {
        return Err(format!(
            "embeddinggemma2: config declares hidden_size_per_layer_input={} but the \
             checkpoint {} the ple tensors",
            config.ple_dim,
            if has_ple { "has" } else { "lacks" }
        ));
    }

    let weights = load_weights(gpu, &src, &config)?;

    // ── the two towers (the `.vl` sidecar) ─────────────────────────────
    // A resolved but absent/mis-stamped sidecar is a LOAD ERROR: a text-only
    // bundle that then refuses every image (or every clip) would hide the
    // real problem.
    let (vision, audio) = match sidecar_path {
        None => (None, None),
        Some(p) => {
            let vsrc = ModelSource::from_path(&p.to_string_lossy())
                .map_err(|e| format!("embeddinggemma2: open sidecar {}: {e}", p.display()))?;
            let vision = match config.vision {
                None => {
                    return Err(format!(
                        "embeddinggemma2: a sidecar was resolved at {} but this \
                         checkpoint's config declares no vision_config",
                        p.display()
                    ))
                }
                Some(vcfg) => {
                    if !VisionTower::sidecar_has_tower(&vsrc) {
                        return Err(format!(
                            "embeddinggemma2: sidecar {} carries no vision tower \
                             (vision_tower.patch_embedder.input_proj.weight missing)",
                            p.display()
                        ));
                    }
                    eprintln!(
                        "  embeddinggemma2: vision tower from {} ({} layers, patch {}, \
                         position table {})",
                        p.display(),
                        vcfg.num_layers,
                        vcfg.patch_size,
                        vcfg.position_embedding_size
                    );
                    Some(load_vision_tower(gpu, &vsrc, &vcfg, config.hidden_size)?)
                }
            };
            let audio = match config.audio {
                None => {
                    return Err(format!(
                        "embeddinggemma2: a sidecar was resolved at {} but this \
                         checkpoint's config declares no audio_config",
                        p.display()
                    ))
                }
                Some(acfg) => {
                    if !crate::audio_tower::AudioTower::sidecar_has_tower(&vsrc) {
                        return Err(format!(
                            "embeddinggemma2: sidecar {} carries no audio tower \
                             (audio_tower.output_proj.weight missing)",
                            p.display()
                        ));
                    }
                    eprintln!(
                        "  embeddinggemma2: audio tower from {} ({} layers, hidden {}, \
                         conv {:?}, {} mel bins)",
                        p.display(),
                        acfg.num_layers,
                        acfg.hidden_size,
                        acfg.subsampling_conv_channels,
                        config.audio_frontend.mel_bins
                    );
                    Some(load_audio_tower(
                        gpu,
                        &vsrc,
                        &acfg,
                        config.audio_frontend.mel_bins,
                        config.hidden_size,
                    )?)
                }
            };
            (vision, audio)
        }
    };

    eprintln!(
        "  embeddinggemma2: {} layers ({}), hidden {}, embed {}→{}, window {}, \
         query/doc max {}/{} tokens, {}",
        config.num_layers,
        layer_type_summary(&config),
        config.hidden_size,
        config.hidden_size,
        config.embedding_dim,
        config.sliding_window,
        config.query_length,
        config.document_length,
        match (&vision, &audio) {
            (Some(_), Some(_)) => "vision + audio towers resident",
            (Some(_), None) => "vision tower resident",
            (None, Some(_)) => "audio tower resident",
            (None, None) => "text-only (no sidecar)",
        },
    );

    if hipfire_config::developer_var("HIPFIRE_EMBED_DEBUG")
        .ok()
        .as_deref()
        == Some("1")
    {
        let (f, t) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        eprintln!(
            "  [embed-debug] after load: VRAM {} MB free / {} MB",
            f / (1024 * 1024),
            t / (1024 * 1024)
        );
    }
    Ok(EmbedGemmaBundle {
        config,
        weights,
        tokenizer,
        vision,
        audio,
    })
}

/// What a media span's rows are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
    /// ONE FRAME of a video clip. HF's `replace_video_token` (with
    /// `add_timestamps` off) emits one `[boi, video_token×S, eoi]` block per
    /// frame, so a clip contributes one span per frame — each frame is
    /// spliced exactly like an image, which is also what the model's
    /// `masked_scatter` over the 258884 positions does.
    Video,
}

/// One media run spliced into [`AssembledInput::ids`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    /// First row of the run inside `ids` (the first inner placeholder, NOT
    /// the `boi`/`boa` marker before it).
    pub start_row: usize,
    /// Placeholder rows — the media encoder's token count.
    pub len: usize,
    pub kind: MediaKind,
}

/// The encode-side request assembly shared by the CLI, daemon verb and HTTP
/// layer: `[bos] prompt+text [eos]` with every media marker expanded in
/// place, the body truncated so the EXPANDED sequence fits the kind's limit.
pub struct AssembledInput {
    pub ids: Vec<u32>,
    /// One entry per media run, in request order — images and clips
    /// interleaved as the markers (or the append order) placed them. A video
    /// contributes one entry PER FRAME (see [`MediaKind::Video`]).
    pub media_spans: Vec<MediaSpan>,
    /// The CPU-side image processor output per image, in request order,
    /// followed by every video's frames (video 0's frames in order, then
    /// video 1's, …). Run them through
    /// [`crate::vision::VisionTower::encode_image`] — or [`encode_media`] —
    /// to get the rows the spans want: image spans take `media[..n_images]`,
    /// video spans the rest.
    pub media: Vec<ProcessedImage>,
    /// How many leading entries of `media` are images (the rest are video
    /// frames). Spans of the two kinds are interleaved in `ids`, so each
    /// needs its own cursor.
    pub n_images: usize,
    /// The CPU-side audio frontend output per clip, in request order.
    pub audio: Vec<ProcessedAudio>,
}

impl AssembledInput {
    /// A text-only assembly.
    pub fn text(ids: Vec<u32>) -> Self {
        Self {
            ids,
            media_spans: Vec::new(),
            media: Vec::new(),
            n_images: 0,
            audio: Vec::new(),
        }
    }
}

/// The projected rows for `assembled`'s spans, in span order. `rows[i]` must
/// be `[media_spans[i].len, text_hidden]`.
pub fn media_rows<'a>(
    assembled: &AssembledInput,
    rows: &'a [Vec<f32>],
) -> Result<Vec<MediaRows<'a>>, String> {
    if rows.len() != assembled.media_spans.len() {
        return Err(format!(
            "embeddinggemma2: {} projected row blocks for {} media spans",
            rows.len(),
            assembled.media_spans.len()
        ));
    }
    let mut out = Vec::with_capacity(rows.len());
    for (span, block) in assembled.media_spans.iter().zip(rows) {
        if span.len == 0 || block.len() % span.len != 0 {
            return Err(format!(
                "embeddinggemma2: media span {} carries {} rows, not a whole number of \
                 {}-wide rows",
                span.start_row,
                block.len(),
                span.len
            ));
        }
        out.push(MediaRows {
            start_row: span.start_row,
            rows: block,
        });
    }
    Ok(out)
}

/// Run every media item's tower forward: one `[span.len, text_hidden]` block
/// per span, in span order (which is what [`media_rows`] expects).
///
/// No cross-request caching: embeddings are stateless, and a cached tower
/// output would outlive the request that produced it.
pub fn encode_media(
    gpu: &mut Gpu,
    bundle: &EmbedGemmaBundle,
    assembled: &AssembledInput,
) -> Result<Vec<Vec<f32>>, String> {
    if assembled.media_spans.is_empty() {
        return Ok(Vec::new());
    }
    let hidden = bundle.config.hidden_size;
    let mut out = Vec::with_capacity(assembled.media_spans.len());
    let mut next_image = 0usize;
    let mut next_frame = assembled.n_images;
    let mut next_audio = 0usize;
    for (i, span) in assembled.media_spans.iter().enumerate() {
        let (block, what) =
            match span.kind {
                MediaKind::Image => {
                    let tower = bundle
                        .vision
                        .as_ref()
                        .ok_or_else(|| format!("{NO_VISION_TOWER} (span {i} carries an image)"))?;
                    let img = assembled.media.get(next_image).ok_or_else(|| {
                        "embeddinggemma2: more image spans than processed images — assembly \
                     bug, not a request error"
                            .to_string()
                    })?;
                    next_image += 1;
                    (tower.encode_image(gpu, img)?, "image")
                }
                MediaKind::Video => {
                    // One frame of a clip: the SAME tower forward the image leg
                    // runs, which is HF's per-frame batch element (frames never
                    // attend each other).
                    let tower = bundle.vision.as_ref().ok_or_else(|| {
                        format!("{NO_VIDEO_TOWER} (span {i} carries a video frame)")
                    })?;
                    let frame = assembled.media.get(next_frame).ok_or_else(|| {
                        "embeddinggemma2: more video spans than processed frames — assembly \
                     bug, not a request error"
                            .to_string()
                    })?;
                    next_frame += 1;
                    (tower.encode_image(gpu, frame)?, "video frame")
                }
                MediaKind::Audio => {
                    let tower = bundle.audio.as_ref().ok_or_else(|| {
                        format!("{NO_AUDIO_TOWER} (span {i} carries an audio clip)")
                    })?;
                    let clip = assembled.audio.get(next_audio).ok_or_else(|| {
                        "embeddinggemma2: more audio spans than processed clips — assembly \
                     bug, not a request error"
                            .to_string()
                    })?;
                    next_audio += 1;
                    (tower.encode_audio(gpu, clip)?, "audio")
                }
            };
        if block.len() != span.len * hidden {
            return Err(format!(
                "embeddinggemma2: {what} {i} produced {} floats for a {}-row span",
                block.len(),
                span.len
            ));
        }
        out.push(block);
    }
    Ok(out)
}

/// Assemble → media towers → batched text encode for one input: the raw
/// `[ids.len(), embedding_dim]` row block the caller pools with
/// [`EmbedGemmaBundle::finalize_embedding`].
pub fn encode_assembled(
    gpu: &mut Gpu,
    bundle: &EmbedGemmaBundle,
    assembled: &AssembledInput,
) -> Result<Vec<f32>, String> {
    let rows = encode_media(gpu, bundle, assembled)?;
    let media = media_rows(assembled, &rows)?;
    let input = EncodeInput {
        ids: &assembled.ids,
        media: &media,
    };
    encode_embeddings(gpu, &bundle.weights, &bundle.config, &input)
}

/// Tokenize + truncate + splice one input's media markers.
///
/// Sequence shape (mirrors the checkpoint's HF post-processor +
/// sentence-transformers prompt): `[bos] [prompt ∘ text] [eos]`, truncation
/// cutting **body tokens only** from the end so the frame survives — the
/// same truncation shape the HF tokenizer applies at `max_length`
/// (special-token-aware). An input whose *prompt alone* already exceeds the
/// limit is an error rather than a mid-prompt cut.
///
/// Media: each image contributes one `<|image|>` marker (token id 258880,
/// `config.image_token_id`), each audio clip one `<|audio|>` marker (id
/// 258881, `config.audio_token_id`) and each VIDEO one `<|video|>` marker
/// (id 258884, `config.video_token_id`), in request order. Each marker —
/// matched as an ID, so the tokenizer's own special-token split decides — is
/// replaced by that item's own run: `[boi, image_token×N, eoi]` for an
/// image, `[boa, audio_token×M, eoa]` for a clip, and ONE
/// `[boi, video_token×S, eoi]` block PER FRAME for a video (HF's
/// `replace_video_token` with `add_timestamps` off — no separators, no
/// timestamps), with `N`/`M`/`S` from the item's processor run. The runs
/// therefore count against the kind's token limit, and the body is cut from
/// the end to make room. With no marker for a modality in the text, that
/// modality's runs are appended before EOS (images first, then videos, then
/// clips) — the wire's "a part whose text lacks the marker appends it at the
/// part position" is what normally prevents this. A marker/item count
/// mismatch for any modality is refused (HF `validate_inputs`); a cut that
/// would drop a trailing marker is refused rather than silently serving the
/// other items.
///
/// A video's `videos[v]` is that clip's FRAMES (PNG/JPEG, display order):
/// the sampler (`crate::video::sample_indices`, HF's own `max_frames` rule)
/// picks which of them run, and each selected frame goes through the image
/// pipeline with the video processor's own patch budget.
///
/// With `images`, `audios` and `videos` all empty this is byte-identical to
/// the text-only path.
pub fn assemble_input(
    bundle: &EmbedGemmaBundle,
    kind: InputKind,
    text: &str,
    images: &[Vec<u8>],
    audios: &[Vec<u8>],
    videos: &[Vec<Vec<u8>>],
) -> Result<AssembledInput, String> {
    let cfg = &bundle.config;
    let limit = cfg.kind_length(kind);
    let bos = cfg.bos_token_id;
    let eos = cfg.eos_token_id;

    // Media first: the expansion — and therefore the truncation room —
    // depends on each item's own token count.
    let mut media: Vec<ProcessedImage> = Vec::with_capacity(images.len());
    if !images.is_empty() {
        let vcfg = cfg.vision.ok_or_else(|| {
            "embeddinggemma2: image input but this checkpoint's config declares no \
             vision_config (no Gemma-4 tower geometry to process against)"
                .to_string()
        })?;
        for (i, bytes) in images.iter().enumerate() {
            media.push(
                process_image(bytes, &vcfg)
                    .map_err(|e| format!("embeddinggemma2: image {i}: {e}"))?,
            );
        }
    }
    // Audio is decoded and PLANNED first, transformed later: the plan alone
    // decides the room, so an over-long clip is refused before its mel
    // frames are computed (which is the expensive half for a long file).
    let mut audio_in: Vec<AudioInput> = Vec::with_capacity(audios.len());
    if !audios.is_empty() {
        if cfg.audio.is_none() {
            return Err(
                "embeddinggemma2: audio input but this checkpoint's config declares no \
                 audio_config (no Gemma-4 conformer geometry to process against)"
                    .into(),
            );
        }
        for (i, bytes) in audios.iter().enumerate() {
            let input = decode_and_plan(bytes, &cfg.audio_frontend)
                .map_err(|e| format!("embeddinggemma2: audio {i}: {e}"))?;
            // A clip shorter than one mel frame has no soft tokens at all:
            // the run would be a bare `boa eoa` pair with nothing to splice,
            // so refuse it here rather than inside the tower.
            if input.plan.n_tokens == 0 {
                return Err(format!(
                    "embeddinggemma2: audio {i} is {} samples, too short for one mel frame \
                     of {} — send at least {} ms of audio",
                    input.plan.samples,
                    cfg.audio_frontend.frame_length,
                    (cfg.audio_frontend.frame_length as f64 * 1000.0
                        / cfg.audio_frontend.sampling_rate as f64)
                        .ceil() as usize
                ));
            }
            audio_in.push(input);
        }
    }
    // Video: each clip is sampled (HF's `max_frames` rule over the frames the
    // caller supplied) and every selected frame is preprocessed against the
    // VIDEO processor's patch budget, in the image pipeline. Frames land
    // after the images in `media`, each clip's frames back to back; `clips`
    // keeps the per-frame soft-token counts in the same order so the splice
    // can emit each frame's own block.
    let n_images = media.len();
    let mut clips: Vec<Vec<usize>> = Vec::with_capacity(videos.len());
    if !videos.is_empty() {
        let vcfg = cfg.frame_vision()?;
        for (i, frames) in videos.iter().enumerate() {
            let processed = process_video(frames, i, &cfg.video_frontend, &vcfg)
                .map_err(|e| format!("embeddinggemma2: {e}"))?;
            clips.push(processed.iter().map(|f| f.n_tokens).collect());
            media.extend(processed);
        }
    }
    let expansion: usize = media[..n_images].iter().map(|m| m.n_tokens + 1).sum::<usize>()
        + audio_in.iter().map(|a| a.plan.n_tokens + 1).sum::<usize>()
        // One video marker expands to `frames × (S + 2) − 1` ids: every frame
        // contributes its block (boi + placeholders + eoi) and the single
        // marker token it replaced is gone.
        + media[n_images..].iter().map(|f| f.n_tokens + 2).sum::<usize>()
        - if clips.is_empty() { 0 } else { clips.len() };
    // The frame (bos + eos) and the runs are never truncated away, so media
    // that cannot fit at all is a refusal, not a silent trim of the media.
    if expansion + 2 > limit {
        return Err(format!(
            "embeddinggemma2: this request's media expands to {expansion} placeholder \
             tokens, over the {limit}-token {kind:?} limit — send a shorter clip, image or \
             video, or split the request"
        ));
    }
    let audio: Vec<ProcessedAudio> = audio_in
        .iter()
        .map(|a| mels(&cfg.audio_frontend, a))
        .collect();

    let framed = format!("{}{}", cfg.kind_prompt(kind), text);
    let body = bundle.tokenizer.encode(&framed);
    if body.is_empty() {
        return Err(
            "embeddinggemma2: input tokenized to zero tokens (empty text after the \
             task prompt?)"
                .into(),
        );
    }

    // The markers the tokenizer itself emits: added tokens carrying
    // `config.image_token_id` / `config.audio_token_id`. The splice is by
    // ID, so the tokenizer's own longest-match special-token split decides
    // what counts as a marker — exactly what HF's processor counts in the
    // text.
    let image_marker = media_marker(
        bundle,
        "image",
        IMAGE_MARKER,
        Some(IMAGE_MARKER_ALIAS),
        text,
        cfg.image_token_id,
        !images.is_empty(),
    )?;
    let audio_marker = media_marker(
        bundle,
        "audio",
        AUDIO_MARKER,
        None,
        text,
        cfg.audio_token_id,
        !audios.is_empty(),
    )?;
    let video_marker = media_marker(
        bundle,
        "video",
        VIDEO_MARKER,
        None,
        text,
        cfg.video_token_id,
        !videos.is_empty(),
    )?;
    let markers = [
        (IMAGE_MARKER, image_marker),
        (AUDIO_MARKER, audio_marker),
        (VIDEO_MARKER, video_marker),
    ];
    for (i, (name, id)) in markers.iter().enumerate() {
        if id.is_none() {
            continue;
        }
        for (other_name, other) in markers.iter().skip(i + 1) {
            if id == other {
                return Err(format!(
                    "embeddinggemma2: the tokenizer maps {name:?} and {other_name:?} to \
                     the same id {} — the modalities cannot share a placeholder",
                    id.unwrap_or_default()
                ));
            }
        }
    }
    let image_markers = match image_marker {
        Some(id) => body.iter().filter(|t| **t == id).count(),
        None => 0,
    };
    if !images.is_empty() && image_markers != 0 && image_markers != images.len() {
        return Err(format!(
            "embeddinggemma2: {image_markers} image marker(s) in the text for {} image(s) \
             — HF's processor refuses this mismatch, so the request is refused rather \
             than mis-placed",
            images.len()
        ));
    }
    let audio_markers = match audio_marker {
        Some(id) => body.iter().filter(|t| **t == id).count(),
        None => 0,
    };
    if !audios.is_empty() && audio_markers != 0 && audio_markers != audios.len() {
        return Err(format!(
            "embeddinggemma2: {audio_markers} audio marker(s) in the text for {} clip(s) \
             — HF's processor refuses this mismatch, so the request is refused rather \
             than mis-placed",
            audios.len()
        ));
    }
    let video_markers = match video_marker {
        Some(id) => body.iter().filter(|t| **t == id).count(),
        None => 0,
    };
    if !videos.is_empty() && video_markers != 0 && video_markers != videos.len() {
        return Err(format!(
            "embeddinggemma2: {video_markers} video marker(s) in the text for {} video(s) \
             — one `<|video|>` per video (not per frame); HF's processor refuses this \
             mismatch, so the request is refused rather than mis-placed",
            videos.len()
        ));
    }

    // The frame (bos + eos) and every media expansion are never truncated
    // away; what is left is the body's room. A prompt that alone overflows
    // the limit is a configuration mismatch — the encode would be nothing
    // but a truncated prompt. Checked on the prompt's OWN tokenization (the
    // framed body check below cannot see it: truncation makes any body fit).
    let room = limit - (2 + expansion); // expansion + 2 <= limit was checked above
    let prompt_ids = bundle.tokenizer.encode(cfg.kind_prompt(kind));
    if prompt_ids.len() > room {
        return Err(format!(
            "embeddinggemma2: kind prompt tokenizes to {} tokens, leaving no room \
             for text under the {} limit (check the checkpoint's prompt/length \
             sidecars)",
            prompt_ids.len(),
            limit
        ));
    }
    let body: &[u32] = if body.len() > room {
        &body[..room]
    } else {
        &body
    };

    // ── splice ─────────────────────────────────────────────────────────
    let mut ids = Vec::with_capacity(body.len() + expansion + 2);
    let mut spans: Vec<MediaSpan> = Vec::with_capacity(media.len() + audio.len());
    ids.push(bos);
    let mut next_image = 0usize;
    let mut next_video = 0usize;
    let mut next_audio = 0usize;
    // One video marker expands to one `[boi, video_token×n, eoi]` block per
    // frame (HF `replace_video_token`, timestamps off), so a clip contributes
    // one span per frame — the same shape an image's single span has.
    let push_video = |ids: &mut Vec<u32>, spans: &mut Vec<MediaSpan>, counts: &[usize]| {
        for &n in counts {
            ids.push(cfg.boi_token_id);
            let start_row = ids.len();
            ids.extend(std::iter::repeat_n(cfg.video_token_id, n));
            ids.push(cfg.eoi_token_id);
            spans.push(MediaSpan {
                start_row,
                len: n,
                kind: MediaKind::Video,
            });
        }
    };
    for &id in body {
        if Some(id) == image_marker && next_image < n_images {
            let n = media[next_image].n_tokens;
            ids.push(cfg.boi_token_id);
            let start_row = ids.len();
            ids.extend(std::iter::repeat_n(cfg.image_token_id, n));
            ids.push(cfg.eoi_token_id);
            spans.push(MediaSpan {
                start_row,
                len: n,
                kind: MediaKind::Image,
            });
            next_image += 1;
        } else if Some(id) == video_marker && next_video < clips.len() {
            push_video(&mut ids, &mut spans, &clips[next_video]);
            next_video += 1;
        } else if Some(id) == audio_marker && next_audio < audio.len() {
            let n = audio[next_audio].plan.n_tokens;
            ids.push(cfg.boa_token_id);
            let start_row = ids.len();
            ids.extend(std::iter::repeat_n(cfg.audio_token_id, n));
            ids.push(cfg.eoa_token_id);
            spans.push(MediaSpan {
                start_row,
                len: n,
                kind: MediaKind::Audio,
            });
            next_audio += 1;
        } else {
            ids.push(id);
        }
    }
    if next_image == 0 && n_images > 0 {
        // Media present but no marker survived: append every run before EOS,
        // images first (the order the request listed them), then videos, then
        // clips.
        for m in &media[..n_images] {
            ids.push(cfg.boi_token_id);
            let start_row = ids.len();
            ids.extend(std::iter::repeat_n(cfg.image_token_id, m.n_tokens));
            ids.push(cfg.eoi_token_id);
            spans.push(MediaSpan {
                start_row,
                len: m.n_tokens,
                kind: MediaKind::Image,
            });
        }
        next_image = n_images;
    }
    if next_video == 0 && !clips.is_empty() {
        for counts in &clips {
            push_video(&mut ids, &mut spans, counts);
        }
        next_video = clips.len();
    }
    if next_audio == 0 && !audio.is_empty() {
        for a in &audio {
            let n = a.plan.n_tokens;
            ids.push(cfg.boa_token_id);
            let start_row = ids.len();
            ids.extend(std::iter::repeat_n(cfg.audio_token_id, n));
            ids.push(cfg.eoa_token_id);
            spans.push(MediaSpan {
                start_row,
                len: n,
                kind: MediaKind::Audio,
            });
        }
        next_audio = audio.len();
    }
    for (what, placed, want) in [
        ("image", next_image, n_images),
        ("video", next_video, videos.len()),
        ("audio", next_audio, audio.len()),
    ] {
        if placed != want {
            return Err(format!(
                "embeddinggemma2: the {limit}-token limit leaves room for {placed} of \
                 {want} {what} item(s) — the body cut dropped a MARKER, which would \
                 silently drop that {what}; shorten the text"
            ));
        }
    }
    ids.push(eos);

    if ids.len() < 3 {
        return Err(format!(
            "embeddinggemma2: input framed to {} tokens under the {} limit — \
             nothing left to pool",
            ids.len(),
            limit
        ));
    }
    if ids.len() > limit {
        return Err(format!(
            "embeddinggemma2: assembled {} tokens over the {limit} limit — truncation \
             accounting is wrong",
            ids.len()
        ));
    }
    Ok(AssembledInput {
        ids,
        media_spans: spans,
        media,
        n_images,
        audio,
    })
}

/// Resolve one modality's placeholder marker for THIS request.
///
/// `Ok(None)` = nothing to splice: the modality carries no media (markers
/// stay literal text, exactly what HF does for a marker with no media), or
/// the vocabulary cannot emit the marker while the text does not spell it,
/// in which case the runs are appended after the body.
fn media_marker(
    bundle: &EmbedGemmaBundle,
    what: &str,
    marker: &str,
    alias: Option<&str>,
    text: &str,
    config_id: u32,
    have_media: bool,
) -> Result<Option<u32>, String> {
    if !have_media {
        return Ok(None);
    }
    match bundle.tokenizer.special_token_id(marker) {
        Some(id) if id != config_id => Err(format!(
            "embeddinggemma2: the tokenizer maps {marker:?} to id {id} but the checkpoint \
             config's {what}_token_id is {config_id} — refusing to guess which id the \
             placeholder rows are"
        )),
        Some(id) => Ok(Some(id)),
        None => {
            let spelled =
                text.matches(marker).count() + alias.map(|a| text.matches(a).count()).unwrap_or(0);
            if spelled > 0 {
                // The literal-string fallback: the text carries the marker
                // spelling but the vocabulary cannot emit it as one token, so
                // the marker can never be placed. Refuse rather than embed
                // the spelling as text and append the item after the text.
                Err(format!(
                    "embeddinggemma2: the text carries {spelled} literal {marker:?} \
                     marker(s) but this tokenizer does not register it as a special \
                     token, so {what} cannot be placed at those positions"
                ))
            } else {
                Ok(None)
            }
        }
    }
}

/// Slot-mode admission preflight for an arch-18 checkpoint: geometry, arch
/// and tokenizer, read without touching the GPU.
///
/// `Ok(None)` = not an arch-18 source (the caller falls through).
pub fn preflight_slot(
    model_path: &std::path::Path,
) -> Result<Option<hipfire_runtime::serve::SlotModelPreflight>, String> {
    use hipfire_runtime::serve::SlotModelPreflight;

    let Some(path) = model_path.to_str() else {
        return Ok(None);
    };
    let src = match ModelSource::from_path(path) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let st: &dyn ModelSourceTrait = match &src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    };
    if st.arch_id() != crate::ARCH_ID {
        return Ok(None);
    }
    let meta: serde_json::Value = serde_json::from_str(st.metadata_json())
        .map_err(|e| format!("embeddinggemma2: config JSON: {e}"))?;
    let mut config = parse_config(crate::config::config_from_metadata(&meta))?;
    if src.is_dir() {
        overlay_sidecars(&mut config, st.path())?;
    } else if let Some(folded) = meta.get("sentence_transformers") {
        apply_folded_sidecars(&mut config, folded);
    }
    let tokenizer = load_tokenizer(st, &config)?;
    // Slot-mode sidecar resolution (see [`resolve_slot_vision_path`]).
    let vl_path = resolve_slot_vision_path(model_path, None);
    Ok(Some(SlotModelPreflight {
        arch_id: crate::ARCH_ID,
        arch_str: "embedding_gemma2".to_string(),
        dim: config.hidden_size,
        layers: config.num_layers,
        vocab: config.vocab_size,
        tokenizer,
        // A resident tower only: the daemon's slot arm passes this pair
        // through to the engine, which loads the towers from `vl_path`.
        // `audio`/`video` are capability flags, not discovery results: both
        // towers (and the video leg, which runs the vision tower per frame)
        // ride the ONE `.vl` pack, so an arch-18 pack always carries them and
        // the daemon's own gate (`&& vl_path.is_some()` after the
        // vision-mode ladder resolves the sidecar) owns the residency truth.
        is_vl: vl_path.is_some(),
        audio: true,
        video: true,
        vl_path,
    }))
}

/// The `.vl` sidecar a slot-mode load resolves.
///
/// Reports what EXISTS beside the checkpoint (an explicit path always wins):
/// the `<stem>.vl` sibling candidates, or a `HIPFIRE_VL_FILE` override inside
/// [`hipfire_runtime::sidecar::resolve_vl_sidecar`]. The load's own ladder
/// owns whether it is used — `vision_mode=off` is a hard override and `on`
/// requires a tower — and the daemon applies that gate to this report in its
/// `cpu_preflight` carrier branch, which is where the mode is known.
///
/// This used to consult `HIPFIRE_VISION_MODE` itself, which is *not* the
/// load's ladder: a multi-slot load asking for `auto` came back text-only
/// (`vl: false`, media refused), and a config-file `auto` defeated an
/// explicit `off`.
pub fn resolve_slot_vision_path(model_path: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    hipfire_runtime::sidecar::resolve_vl_sidecar(&model_path.to_string_lossy())
}
