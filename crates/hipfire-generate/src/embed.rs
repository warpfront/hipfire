// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Embeddings — the `embed` wire body (arch 17, the qwen3.5 multi-vector
//! encoder). The non-chat-verb sibling of [`crate::img`]: the daemon
//! dispatches (id parse, no-model and non-embed refusals) and this module
//! owns everything that names `hipfire_arch_qwen35_embed`.
//!
//! Wire request (JSONL over the daemon stdin):
//!
//! ```json
//! {"id":"…", "kind":"query"|"document",
//!  "input":"text",                       // or
//!  "input":{"text":"…",
//!           "images_base64":["<base64>", "data:image/png;base64,…"],
//!           "audios_base64":["<base64>", "data:audio/wav;base64,…"],
//!           "videos_base64":[["<frame base64>", …], …]}}
//! ```
//!
//! `image_base64` (one image, legacy spelling — what the CLI and the HTTP
//! route used before the list form) is still accepted; a request carrying it
//! *and* `images_base64` is refused rather than half-read. Each list entry is
//! plain base64 or a `data:…;base64,` URL, decoded here — the one decode point
//! on this verb. A list sent under any other key (`images`, `audios`,
//! `videos`) is refused by name, so media input can never silently become a
//! text encode. Audio is a Phase-2 addition: WAV/RIFF is the accepted
//! container (the arch crate refuses other formats by magic), and it is only
//! reachable on an arch-18 load with its `.vl` sidecar (which carries the audio
//! tower). Video is the third leg: `videos_base64` is ONE ENTRY PER VIDEO, each
//! that clip's FRAMES (PNG/JPEG, display order) — the arch crate samples them
//! to `max_frames` and drives the vision tower per frame.
//!
//! Wire reply — exactly one of:
//!
//! ```json
//! {"type":"embedded","id":"…","kind":"query","n_tokens":K,"dim":128,
//!  "n_input_tokens":N,"embeddings":[…K×128 floats…]}
//! {"type":"error","id":"…","message":"…"}
//! ```
//!
//! `embeddings` is the post-Dense, post-skiplist, L2-normalized token
//! matrix, row-major `[K, dim]` (late interaction: score query vs document
//! with Σ max dot — `hipfire_arch_qwen35_embed::maxsim`).

use hipfire_arch_qwen35_embed::{
    assemble_input, encode_embeddings, forward::EncodeInput, maxsim, InputKind,
};
use hipfire_engine::emit::emit_uncorrelated_error;
use hipfire_loader::LoadedModel;
use std::io::Write;

fn parse_kind(req: &serde_json::Value) -> Result<InputKind, String> {
    match req
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("document")
    {
        "query" => Ok(InputKind::Query),
        "document" | "doc" => Ok(InputKind::Document),
        other => Err(format!(
            "embed: kind {other:?} is not one of \"query\" | \"document\" \
             (queries and documents use different markers and truncation)"
        )),
    }
}

/// Extract `(text, image bytes, audio bytes, video frames)` from `input`: a
/// bare string, or an object with `text` and/or the media lists
/// `images_base64` / `audios_base64` / `videos_base64` (the one-image
/// `image_base64` field is the legacy spelling). Each image/audio entry and
/// each video FRAME tolerates a `data:…;base64,` prefix, same as the vision
/// chat path. `videos_base64` is a list of videos, each a list of that clip's
/// frames in display order.
///
/// A list under any other name is refused rather than read as text-only: an
/// unknown media field that silently degrades to a text encode is the one
/// failure this verb must not have.
fn parse_input(req: &serde_json::Value) -> Result<Input, String> {
    let input = req.get("input").ok_or(
        "embed: request has no \"input\" (string, or {text, images_base64, audios_base64, \
         videos_base64})",
    )?;
    if let Some(text) = input.as_str() {
        return Ok(Input {
            text: text.to_string(),
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
        });
    }
    let obj = input.as_object().ok_or(
        "embed: \"input\" must be a string or an object with \
         text/images_base64/audios_base64/videos_base64",
    )?;
    let text = obj
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    for (wrong, shape) in [
        ("images", "[\"<base64>\", …]"),
        ("audios", "[\"<base64>\", …]"),
        (
            "videos",
            "[[\"<frame base64>\", …], …] (one frame list per video)",
        ),
    ] {
        if obj.get(wrong).is_some_and(|v| !v.is_null()) {
            return Err(format!(
                "embed: input has an \"{wrong}\" field; this verb spells the base64 list \
                 \"{wrong}_base64\" — send {{\"text\": \"…\", \"{wrong}_base64\": {shape}}}"
            ));
        }
    }
    let legacy = obj.get("image_base64").filter(|v| !v.is_null());
    let image_list = obj.get("images_base64").filter(|v| !v.is_null());
    if legacy.is_some() && image_list.is_some() {
        return Err(
            "embed: input carries both image_base64 and images_base64 — send the list \
                    under \"images_base64\" (image_base64 is the one-image legacy spelling)"
                .into(),
        );
    }
    let mut images = Vec::new();
    if let Some(one) = legacy {
        let b64 = one
            .as_str()
            .ok_or("embed: image_base64 must be a base64 string")?;
        images.push(
            decode_media_base64(b64)
                .map_err(|e| format!("embed: image_base64 is not valid base64: {e}"))?,
        );
    }
    if let Some(list) = image_list {
        let list = list
            .as_array()
            .ok_or("embed: \"images_base64\" must be an array of base64 strings")?;
        for (i, entry) in list.iter().enumerate() {
            let b64 = entry
                .as_str()
                .ok_or_else(|| format!("embed: images_base64[{i}] must be a base64 string"))?;
            images.push(
                decode_media_base64(b64)
                    .map_err(|e| format!("embed: images_base64[{i}] is not valid base64: {e}"))?,
            );
        }
    }
    let mut audios = Vec::new();
    if let Some(list) = obj.get("audios_base64").filter(|v| !v.is_null()) {
        let list = list
            .as_array()
            .ok_or("embed: \"audios_base64\" must be an array of base64 strings")?;
        for (i, entry) in list.iter().enumerate() {
            let b64 = entry
                .as_str()
                .ok_or_else(|| format!("embed: audios_base64[{i}] must be a base64 string"))?;
            audios.push(
                decode_media_base64(b64)
                    .map_err(|e| format!("embed: audios_base64[{i}] is not valid base64: {e}"))?,
            );
        }
    }
    // Videos: one entry per clip, each entry the clip's FRAMES in display
    // order. The sampler (`EmbeddingGemma2VideoProcessor.sample_frames`) runs
    // in the arch crate, so this layer only checks the shape and decodes —
    // a non-list entry or an empty clip is a caller error named here rather
    // than an empty splice later.
    let mut videos: Vec<Vec<Vec<u8>>> = Vec::new();
    if let Some(list) = obj.get("videos_base64").filter(|v| !v.is_null()) {
        let list = list.as_array().ok_or(
            "embed: \"videos_base64\" must be an array of videos, each an array of frame \
             base64 strings — [[\"<frame>\", …], …]",
        )?;
        for (v, clip) in list.iter().enumerate() {
            let clip = clip.as_array().ok_or_else(|| {
                format!(
                    "embed: videos_base64[{v}] must be an array of frame base64 strings \
                     (one video per entry)"
                )
            })?;
            if clip.is_empty() {
                return Err(format!(
                    "embed: videos_base64[{v}] carries no frames — send a decoded clip as \
                     at least one frame (PNG/JPEG)"
                ));
            }
            let mut frames = Vec::with_capacity(clip.len());
            for (f, entry) in clip.iter().enumerate() {
                let b64 = entry.as_str().ok_or_else(|| {
                    format!("embed: videos_base64[{v}][{f}] must be a base64 string")
                })?;
                frames.push(decode_media_base64(b64).map_err(|e| {
                    format!("embed: videos_base64[{v}][{f}] is not valid base64: {e}")
                })?);
            }
            videos.push(frames);
        }
    }
    if text.is_empty() && images.is_empty() && audios.is_empty() && videos.is_empty() {
        return Err(
            "embed: input has neither text, images_base64, audios_base64 nor videos_base64".into(),
        );
    }
    Ok(Input {
        text,
        images,
        audios,
        videos,
    })
}

/// One `embed` request's decoding of `input`: the text plus the media
/// container bytes, in request order.
#[derive(Debug)]
struct Input {
    text: String,
    images: Vec<Vec<u8>>,
    audios: Vec<Vec<u8>>,
    /// One entry per video: that clip's frames, in display order.
    videos: Vec<Vec<Vec<u8>>>,
}

/// Decode one media entry: plain base64, or a `data:<mime>;base64,<payload>`
/// data URL (the OpenAI content-part spelling). Images and audio share this —
/// the container type is the bytes' business, not the encoding's.
fn decode_media_base64(raw: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(data_url_payload(raw).trim())
        .map_err(|e| e.to_string())
}

/// The base64 payload of `raw`, dropping a `data:…;base64,` prefix when one is
/// present. The `base64,` tolerance is the pre-existing vision-path behaviour
/// (a bare `base64,<payload>` decodes), kept so migrated callers cannot tell
/// the two spellings apart.
fn data_url_payload(raw: &str) -> &str {
    let raw = raw.strip_prefix("base64,").unwrap_or(raw);
    raw.split_once(',')
        .filter(|(head, _)| head.starts_with("data:") && head.ends_with(";base64"))
        .map(|(_, tail)| tail)
        .unwrap_or(raw)
}

/// The image marker the arch-18 assembler splices over: one literal `<|image>`
/// in the text per image, in image order. Zero markers with images present
/// means "append the runs before EOS" (the arch crate's rule); any other count
/// is a caller error — a request that says three images and carries two is
/// answered by refusing, never by silently dropping one.
///
/// The same literal the vocab's `image_token` maps to (id 258880), so a count
/// here is the count the arch crate's assembler sees.
const IMAGE_MARKER: &str = "<|image|>";

/// The audio marker, same contract one token over: `<|audio|>` (id 258881,
/// config `audio_token_id`), one per clip in clip order; the run expands to
/// `boa(256000) + 258881×M + eoa(258883)` with the tower's own `M`.
const AUDIO_MARKER: &str = "<|audio|>";

/// The video marker: `<|video|>` (id 258884, config `video_token_id`), one
/// per VIDEO in video order; the run expands to ONE
/// `boi(255999) + 258884×S + eoi(258882)` block PER FRAME, with `S` = that
/// frame's soft-token count (HF `replace_video_token`, `add_timestamps` off).
const VIDEO_MARKER: &str = "<|video|>";

/// The count rule all three media kinds share: no media means nothing to
/// check; no marker means "append the runs before EOS"; one marker per media
/// item means "splice in place". Anything else returns the marker count for
/// the refusal.
fn disagreeing_markers(text: &str, marker: &str, n_media: usize) -> Option<usize> {
    if n_media == 0 {
        return None;
    }
    let markers = text.matches(marker).count();
    (markers != 0 && markers != n_media).then_some(markers)
}

/// Refuse a marker/image count that the assembler would refuse anyway, but
/// name both numbers here — this is the layer that sees the wire request.
fn validate_image_markers(text: &str, n_images: usize) -> Result<(), String> {
    match disagreeing_markers(text, IMAGE_MARKER, n_images) {
        Some(markers) => Err(format!(
            "embed: text carries {markers} {IMAGE_MARKER} marker(s) but the request has \
             {n_images} image(s) — send one marker per image, or none (an image with no marker \
             is appended before EOS)"
        )),
        None => Ok(()),
    }
}

/// The audio twin of [`validate_image_markers`].
fn validate_audio_markers(text: &str, n_audios: usize) -> Result<(), String> {
    match disagreeing_markers(text, AUDIO_MARKER, n_audios) {
        Some(markers) => Err(format!(
            "embed: text carries {markers} {AUDIO_MARKER} marker(s) but the request has \
             {n_audios} audio clip(s) — send one marker per clip, or none (a clip with no \
             marker is appended before EOS)"
        )),
        None => Ok(()),
    }
}

/// The video twin: one `<|video|>` per VIDEO (not per frame), matching HF's
/// `validate_inputs`.
fn validate_video_markers(text: &str, n_videos: usize) -> Result<(), String> {
    match disagreeing_markers(text, VIDEO_MARKER, n_videos) {
        Some(markers) => Err(format!(
            "embed: text carries {markers} {VIDEO_MARKER} marker(s) but the request has \
             {n_videos} video(s) — send one marker per video (never per frame), or none \
             (a video with no marker is appended before EOS)"
        )),
        None => Ok(()),
    }
}

/// Serve one `embed` request. Emits exactly one `embedded` or one error
/// envelope; never panics across GPU calls (every arch-crate error maps to
/// an `error` event).
pub fn generate_embed(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl Write,
    id: &str,
    req: &serde_json::Value,
) {
    fn fail(stdout: &mut impl Write, id: &str, msg: String) {
        emit_uncorrelated_error(stdout, Some(id), &msg, "validation", false, false);
        let _ = stdout.flush();
    }

    let kind = match parse_kind(req) {
        Ok(k) => k,
        Err(e) => return fail(stdout, id, e),
    };
    let Input {
        text,
        images,
        audios,
        videos,
    } = match parse_input(req) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };

    // Matryoshka width (embeddinggemma2): optional `dims` on the request,
    // strictly an unsigned integer — a float or string dims silently meaning
    // "full width" is a protocol trap, not a convenience.
    let dims = match req.get("dims") {
        None | Some(serde_json::Value::Null) => 0usize,
        Some(v) => match v.as_u64() {
            Some(0) => 0usize,
            Some(d) => d as usize,
            None => {
                return fail(
                    stdout,
                    id,
                    format!("embed: `dims` must be a non-negative integer, got {v}"),
                )
            }
        },
    };

    // Arch 18 (EmbeddingGemma 2): single-vector body.
    if let Some(bundle) = m.embeddinggemma2_mut() {
        if !images.is_empty() && bundle.vision.is_none() {
            // The towers are separate artifacts (one `.vl` sidecar), so an
            // absent one is a load-time gap, not a bad request: name it.
            return fail(
                stdout,
                id,
                hipfire_arch_embeddinggemma2::carrier::NO_VISION_TOWER.into(),
            );
        }
        if !audios.is_empty() && bundle.audio.is_none() {
            return fail(
                stdout,
                id,
                hipfire_arch_embeddinggemma2::carrier::NO_AUDIO_TOWER.into(),
            );
        }
        if !videos.is_empty() && bundle.vision.is_none() {
            // Video has no tower of its own: its frames ride the vision
            // tower, so a text-only load hits the same `.vl` gap.
            return fail(
                stdout,
                id,
                hipfire_arch_embeddinggemma2::carrier::NO_VIDEO_TOWER.into(),
            );
        }
        if let Err(e) = validate_image_markers(&text, images.len()) {
            return fail(stdout, id, e);
        }
        if let Err(e) = validate_audio_markers(&text, audios.len()) {
            return fail(stdout, id, e);
        }
        if let Err(e) = validate_video_markers(&text, videos.len()) {
            return fail(stdout, id, e);
        }
        if dims != 0 && dims > bundle.config.embedding_dim {
            return fail(
                stdout,
                id,
                format!(
                    "embed: dims {dims} exceeds this model's embedding width {}",
                    bundle.config.embedding_dim
                ),
            );
        }
        let assembled = match crate::embed::assemble_input_eg2(
            bundle, kind, &text, &images, &audios, &videos,
        ) {
            Ok(a) => a,
            Err(e) => return fail(stdout, id, e),
        };
        let n_input = assembled.ids.len();
        // Vision tower (when the request carries images) then the batched text
        // encode — the arch crate owns the row-splice order.
        let raw = match hipfire_arch_embeddinggemma2::carrier::encode_assembled(
            gpu, bundle, &assembled,
        ) {
            Ok(v) => v,
            Err(e) => return fail(stdout, id, e),
        };
        let out_dim = if dims == 0 {
            bundle.config.embedding_dim
        } else {
            dims
        };
        let embeddings = match bundle.finalize_embedding(&raw, n_input, dims) {
            Ok(v) => v,
            Err(e) => return fail(stdout, id, e),
        };
        let payload = serde_json::json!({
            "type": "embedded",
            "id": id,
            "attempt_id": 0,
            "kind": if kind == InputKind::Query { "query" } else { "document" },
            "n_tokens": 1, // single-vector: one pooled embedding
            "n_input_tokens": n_input,
            "dim": out_dim,
            "embeddings": embeddings,
        });
        let _ = writeln!(stdout, "{payload}");
        let _ = stdout.flush();
        return;
    }

    // Typed handle: the daemon already refused non-embed arches.
    let Some(bundle) = m.qwen35_embed_mut() else {
        return fail(
            stdout,
            id,
            "embed: loaded model is not an embedding checkpoint (arch 17/18)".into(),
        );
    };
    if dims != 0 {
        return fail(
            stdout,
            id,
            "embed: `dims` (Matryoshka truncation) applies to embeddinggemma2 (arch 18) \
             only; the arch-17 encoder is multi-vector with a fixed 128-d head"
                .into(),
        );
    }
    // One image per request is this encoder's contract (a single spliced
    // vision block, no marker text). A list is asking for arch 18, so refuse
    // it by name rather than encoding the first image and dropping the rest.
    if images.len() > 1 {
        return fail(
            stdout,
            id,
            format!(
                "embed: this encoder takes at most one image per request (got {}); \
                 embeddinggemma2 (arch 18) accepts a list",
                images.len()
            ),
        );
    }
    // No audio tower on the arch-17 encoder at all: the arch's own answer,
    // whatever the arch-18 arm does with a resident tower.
    if !audios.is_empty() {
        return fail(
            stdout,
            id,
            format!(
                "embed: this encoder has no audio tower — {} audio clip(s) cannot be encoded \
                 here; audio applies to embeddinggemma2 (arch 18)",
                audios.len()
            ),
        );
    }
    // And no video leg: that is arch 18's (its frames ride ITS vision tower).
    if !videos.is_empty() {
        return fail(
            stdout,
            id,
            format!(
                "embed: this encoder has no video leg — {} video(s) cannot be encoded \
                 here; video applies to embeddinggemma2 (arch 18)",
                videos.len()
            ),
        );
    }
    let image_bytes = images.first().map(|bytes| bytes.as_slice());

    // Phase timing under the same knob the encoder's own trace uses. For a
    // long document the GPU kernel sum is a small fraction of wall-clock, and
    // this is what shows where the rest goes (gather, JSON of a wide output,
    // the download).
    // Its own knob, NOT HIPFIRE_EMBED_DEBUG: that one also arms the encoder's
    // per-layer finite `trace()`, which syncs the GPU every layer and inflates
    // every number here (measured 678 ms vs 334 ms on the same request).

    // Phase timing under the same knob the encoder's own trace uses. For a
    // long document the GPU kernel sum is a small fraction of wall-clock, and
    // this is what shows where the rest goes (gather, JSON of a wide output,
    // the download).
    // Its own knob, NOT HIPFIRE_EMBED_DEBUG: that one also arms the encoder's
    // per-layer finite `trace()`, which syncs the GPU every layer and inflates
    // every number here (measured 678 ms vs 334 ms on the same request).
    let timed = hipfire_config::developer_var("HIPFIRE_EMBED_PHASE")
        .ok()
        .as_deref()
        == Some("1");
    let t_phase = std::time::Instant::now();
    let assembled = match assemble_input(bundle, gpu, kind, &text, image_bytes) {
        Ok(a) => a,
        Err(e) => return fail(stdout, id, e),
    };
    let t_assemble = t_phase.elapsed();
    let n_input = assembled.ids.len();
    let input = EncodeInput {
        ids: &assembled.ids,
        image_embeds: &assembled.image_embeds,
        spans: &assembled.spans,
        pos3: &assembled.pos3,
    };
    let t_encode = std::time::Instant::now();
    let raw = match encode_embeddings(
        gpu,
        &bundle.weights,
        &mut bundle.state,
        &bundle.config,
        &input,
    ) {
        Ok(v) => v,
        Err(e) => return fail(stdout, id, e),
    };
    let d_encode = t_encode.elapsed();
    // MultiVectorMask + Normalize — via the shared `finalize_embeddings` so
    // the sequential and slot routes cannot drift (the helper is exactly
    // this mask/protect/normalize sequence; duplicating it here would make
    // a future skiplist change silently diverge the two paths).
    let dim = bundle.config.dense_out_features;
    let (embeddings, n_tokens) = match bundle.finalize_embeddings(kind, &assembled.ids, &raw) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };
    let d_post = t_encode.elapsed() - d_encode;

    let payload = serde_json::json!({
        "type": "embedded",
        "id": id,
        // See `generate_embed_slot`: the reply is correlated by `(id, 0)`, the
        // key a synchronous verb's caller registers.
        "attempt_id": 0,
        "kind": if kind == InputKind::Query { "query" } else { "document" },
        "n_tokens": n_tokens,
        "n_input_tokens": n_input,
        "dim": dim,
        "embeddings": embeddings,
    });
    let t_write = std::time::Instant::now();
    let _ = writeln!(stdout, "{payload}");
    let _ = stdout.flush();
    if timed {
        eprintln!(
            "  [embed-phase] n={n_input} kept={n_tokens}: assemble {:?} | encode {:?} | \
             mask+normalize {:?} | serialize+write {:?}",
            t_assemble,
            d_encode,
            d_post,
            t_write.elapsed()
        );
    }
}

/// The arch-18 kinds are isomorphic to the arch-17 ones; map rather than
/// re-parse.
pub(crate) fn assemble_input_eg2(
    bundle: &hipfire_arch_embeddinggemma2::EmbedGemmaBundle,
    kind: InputKind,
    text: &str,
    images: &[Vec<u8>],
    audios: &[Vec<u8>],
    videos: &[Vec<Vec<u8>>],
) -> Result<hipfire_arch_embeddinggemma2::AssembledInput, String> {
    let k = match kind {
        InputKind::Query => hipfire_arch_embeddinggemma2::InputKind::Query,
        InputKind::Document => hipfire_arch_embeddinggemma2::InputKind::Document,
    };
    hipfire_arch_embeddinggemma2::assemble_input(bundle, k, text, images, audios, videos)
}

/// Late-interaction score between two already-encoded inputs (the CLI's
/// `--score` path; the values are the `embeddings` arrays of an `embedded`
/// event each).
pub fn score_maxsim(
    query: &[f32],
    q_tokens: usize,
    doc: &[f32],
    d_tokens: usize,
    dim: usize,
) -> f32 {
    maxsim(query, q_tokens, doc, d_tokens, dim)
}

/// Serve one `embed` request through a multi-slot engine instead of the
/// ordinary resident model (slot mode owns the encoder and leaves the daemon's
/// `model` slot empty, so the sequential body cannot run).
///
/// Same wire body in, same `embedded` envelope out: the request is parsed
/// here, the engine tokenizes and encodes on its own thread, and the reply
/// carries the fields the sequential route writes. A per-request failure — a
/// bad kind, an undecodable image, an input the encoder refuses — becomes the
/// same `error` envelope [`generate_embed`] emits.
pub fn generate_embed_slot(
    engine: &dyn hipfire_runtime::serve::SlotEngineHandle,
    stdout: &mut impl Write,
    id: &str,
    req: &serde_json::Value,
) {
    use hipfire_runtime::serve::{EmbedKind, EmbedRequest};

    fn fail(stdout: &mut impl Write, id: &str, msg: String) {
        emit_uncorrelated_error(stdout, Some(id), &msg, "validation", false, false);
        let _ = stdout.flush();
    }

    let kind = match parse_kind(req) {
        Ok(k) => k,
        Err(e) => return fail(stdout, id, e),
    };
    let Input {
        text,
        images,
        audios,
        videos,
    } = match parse_input(req) {
        Ok(x) => x,
        Err(e) => return fail(stdout, id, e),
    };
    let wire_kind = match kind {
        InputKind::Query => EmbedKind::Query,
        InputKind::Document => EmbedKind::Document,
    };
    let dims = match req.get("dims") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(0) => None,
            Some(d) => Some(d as usize),
            None => {
                return fail(
                    stdout,
                    id,
                    format!("embed: `dims` must be a non-negative integer, got {v}"),
                )
            }
        },
    };
    // Marker counts are the wire layer's check on both routes; the engine then
    // runs each tower itself (and refuses a clip per request when the resident
    // bundle has no audio tower).
    if let Err(e) = validate_audio_markers(&text, audios.len()) {
        return fail(stdout, id, e);
    }
    if let Err(e) = validate_image_markers(&text, images.len()) {
        return fail(stdout, id, e);
    }
    if let Err(e) = validate_video_markers(&text, videos.len()) {
        return fail(stdout, id, e);
    }
    let mut out = match engine.encode(vec![EmbedRequest {
        kind: wire_kind,
        text,
        images,
        audios,
        videos,
        request_tag: id.to_string(),
        dims,
    }]) {
        Ok(out) => out,
        Err(e) => return fail(stdout, id, e),
    };
    let response = match out.pop() {
        Some(Ok(r)) => r,
        Some(Err(e)) => return fail(stdout, id, e),
        None => {
            return fail(
                stdout,
                id,
                "embed: engine returned no response for the request".into(),
            )
        }
    };
    let payload = serde_json::json!({
        "type": "embedded",
        "id": id,
        // Correlates the reply with the transaction that sent the request:
        // the client registers `(id, 0)` for a synchronous verb, and an
        // attempt-less reply would be routed to the shared control slot
        // instead — which holds one transaction, so concurrent embeds would
        // drop each other's replies.
        "attempt_id": 0,
        "kind": if response.kind == EmbedKind::Query {
            "query"
        } else {
            "document"
        },
        "n_tokens": response.n_tokens,
        "n_input_tokens": response.n_input_tokens,
        "dim": response.dim,
        "embeddings": response.embeddings,
    });
    let _ = writeln!(stdout, "{payload}");
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parsed(input: serde_json::Value) -> Input {
        parse_input(&json!({ "input": input })).expect("input parses")
    }

    /// The text/image view most tests want; audio has its own assertions.
    fn text_images(input: serde_json::Value) -> (String, Vec<Vec<u8>>) {
        let input = parsed(input);
        (input.text, input.images)
    }

    fn audios(input: serde_json::Value) -> Vec<Vec<u8>> {
        parsed(input).audios
    }

    fn refused(input: serde_json::Value) -> String {
        parse_input(&json!({ "input": input })).expect_err("input is refused")
    }

    /// Wire form of the old single-image request: `{text, image_base64}`.
    #[test]
    fn the_legacy_image_base64_field_is_one_image() {
        let (text, images) = text_images(json!({ "text": "a photo", "image_base64": "QUJD" }));
        assert_eq!(text, "a photo");
        assert_eq!(images, vec![b"ABC".to_vec()]);
    }

    #[test]
    fn the_legacy_field_still_strips_a_data_url_prefix() {
        let (_, images) = text_images(json!({
            "text": "",
            "image_base64": "data:image/png;base64,QUJD"
        }));
        assert_eq!(images, vec![b"ABC".to_vec()]);
    }

    #[test]
    fn a_bare_string_is_text_only() {
        let (text, images) = text_images(json!("hello"));
        assert_eq!(text, "hello");
        assert!(images.is_empty(), "a string carries no images");
    }

    #[test]
    fn the_images_list_decodes_in_order_and_tolerates_data_urls() {
        let (text, images) = text_images(json!({
            "text": "a <|image|> b <|image|>",
            "images_base64": ["QUJD", "data:image/jpeg;base64,REVG"]
        }));
        assert_eq!(text, "a <|image|> b <|image|>");
        assert_eq!(images, vec![b"ABC".to_vec(), b"DEF".to_vec()]);
    }

    #[test]
    fn an_empty_images_list_is_text_only() {
        let (_, images) = text_images(json!({ "text": "hi", "images_base64": [] }));
        assert!(images.is_empty());
    }

    /// The list is `images_base64`; the shorter spelling is refused rather
    /// than read as text-only (an image silently dropped is the failure this
    /// verb must not have).
    #[test]
    fn the_images_spelling_is_refused_by_name() {
        let message = refused(json!({ "text": "t", "images": ["QUJD"] }));
        assert!(message.contains("\"images\""), "{message}");
        assert!(message.contains("images_base64"), "{message}");
    }

    #[test]
    fn the_two_image_fields_together_are_refused_not_half_read() {
        let message = refused(json!({
            "text": "t",
            "image_base64": "QUJD",
            "images_base64": ["REVG"]
        }));
        assert!(
            message.contains("both image_base64 and images_base64")
                && message.contains("\"images_base64\""),
            "{message}"
        );
    }

    #[test]
    fn an_empty_input_is_refused_by_its_own_name() {
        let message = refused(json!({ "text": "" }));
        assert!(
            message.contains("neither text, images_base64, audios_base64 nor videos_base64"),
            "{message}"
        );
    }

    fn videos(input: serde_json::Value) -> Vec<Vec<Vec<u8>>> {
        parsed(input).videos
    }

    /// `videos_base64` is a list of VIDEOS, each a list of that clip's frames
    /// in display order — the sampler's input, not a container.
    #[test]
    fn the_video_list_decodes_one_frame_list_per_clip() {
        assert_eq!(
            videos(json!({
                "text": "a <|video|> b <|video|>",
                "videos_base64": [
                    ["QUJD", "data:image/png;base64,REVG"],
                    ["R0hJ"]
                ]
            })),
            vec![
                vec![b"ABC".to_vec(), b"DEF".to_vec()],
                vec![b"GHI".to_vec()]
            ]
        );
        assert!(videos(json!({ "text": "hi", "videos_base64": [] })).is_empty());
    }

    /// The shorter spelling is refused rather than read as text-only, and the
    /// video shape hint names the frame nesting.
    #[test]
    fn the_videos_spelling_is_refused_by_name() {
        let message = refused(json!({ "text": "t", "videos": [["QUJD"]] }));
        assert!(message.contains("\"videos\""), "{message}");
        assert!(message.contains("videos_base64"), "{message}");
        assert!(message.contains("one frame list per video"), "{message}");
    }

    #[test]
    fn a_bad_video_entry_is_refused_by_its_own_index() {
        let message = refused(json!({ "videos_base64": "QUJD" }));
        assert!(message.contains("array of videos"), "{message}");
        let message = refused(json!({ "videos_base64": ["QUJD"] }));
        assert!(message.contains("videos_base64[0]"), "{message}");
        assert!(message.contains("one video per entry"), "{message}");
        let message = refused(json!({ "videos_base64": [[]] }));
        assert!(
            message.contains("videos_base64[0] carries no frames"),
            "{message}"
        );
        let message = refused(json!({ "videos_base64": [["QUJD", 7]] }));
        assert!(message.contains("videos_base64[0][1]"), "{message}");
        let message = refused(json!({ "videos_base64": [["QUJD", "!!!"]] }));
        assert!(message.contains("videos_base64[0][1]"), "{message}");
        assert!(message.contains("not valid base64"), "{message}");
    }

    /// Video-only input is a valid request shape, refused (if at all) by the
    /// encode body — never silently text-only here.
    #[test]
    fn video_only_input_parses() {
        let input = parsed(json!({ "videos_base64": [["QUJD"]] }));
        assert_eq!(input.text, "");
        assert!(input.images.is_empty());
        assert!(input.audios.is_empty());
        assert_eq!(input.videos, vec![vec![b"ABC".to_vec()]]);
    }

    /// One `<|video|>` per VIDEO (never per frame): HF's `validate_inputs`
    /// refuses a mismatch, and the refusal says which unit is counted.
    #[test]
    fn video_markers_must_match_the_video_count_or_be_absent() {
        assert_eq!(VIDEO_MARKER, "<|video|>");
        assert!(validate_video_markers("<|video|> stray", 0).is_ok());
        assert!(validate_video_markers("a clip: ", 1).is_ok());
        assert!(validate_video_markers("<|video|> <|video|>", 2).is_ok());
        // The other modalities' markers are counted apart.
        assert!(validate_video_markers("<|image|> <|audio|>", 1).is_ok());
        let message = validate_video_markers("<|video|>", 2).expect_err("mismatch is refused");
        assert!(message.contains("1 <|video|> marker"), "{message}");
        assert!(message.contains("2 video"), "{message}");
        assert!(message.contains("never per frame"), "{message}");
        let message = validate_video_markers("<|video|> a <|video|>", 1).expect_err("extra marker");
        assert!(message.contains("2 <|video|> marker"), "{message}");
        assert!(message.contains("1 video"), "{message}");
    }

    #[test]
    fn the_audio_list_decodes_in_order_and_tolerates_data_urls() {
        assert_eq!(
            audios(json!({
                "text": "a <|audio|> b",
                "audios_base64": ["QUJD", "data:audio/wav;base64,REVG"]
            })),
            vec![b"ABC".to_vec(), b"DEF".to_vec()]
        );
        assert!(audios(json!({ "text": "hi", "audios_base64": [] })).is_empty());
    }

    /// Same posture as the image list: the shorter spelling is refused rather
    /// than read as text-only.
    #[test]
    fn the_audios_spelling_is_refused_by_name() {
        let message = refused(json!({ "text": "t", "audios": ["QUJD"] }));
        assert!(message.contains("\"audios\""), "{message}");
        assert!(message.contains("audios_base64"), "{message}");
    }

    #[test]
    fn a_bad_audio_entry_is_refused_by_index() {
        let message = refused(json!({ "audios_base64": ["QUJD", "!!!"] }));
        assert!(message.contains("audios_base64[1]"), "{message}");
        let message = refused(json!({ "audios_base64": [7] }));
        assert!(message.contains("audios_base64[0]"), "{message}");
        let message = refused(json!({ "audios_base64": "QUJD" }));
        assert!(message.contains("array of base64 strings"), "{message}");
    }

    /// Audio-only input is a valid request shape (HF: any single modality is),
    /// so it must survive parsing and be refused by the *encode* body, not
    /// here.
    #[test]
    fn audio_only_input_parses() {
        let input = parsed(json!({ "audios_base64": ["QUJD"] }));
        assert_eq!(input.text, "");
        assert!(input.images.is_empty());
        assert_eq!(input.audios, vec![b"ABC".to_vec()]);
    }

    #[test]
    fn a_bad_list_entry_is_refused_by_index() {
        let message = refused(json!({ "images_base64": ["QUJD", "!!!"] }));
        assert!(message.contains("images_base64[1]"), "{message}");
        assert!(message.contains("not valid base64"), "{message}");
        let message = refused(json!({ "images_base64": [7] }));
        assert!(message.contains("images_base64[0]"), "{message}");
        assert!(message.contains("base64 string"), "{message}");
        let message = refused(json!({ "images_base64": "QUJD" }));
        assert!(message.contains("array of base64 strings"), "{message}");
    }

    #[test]
    fn a_non_string_legacy_field_is_refused_not_dropped() {
        let message = refused(json!({ "text": "t", "image_base64": 7 }));
        assert!(message.contains("image_base64"), "{message}");
    }

    /// The marker the arch-18 assembler splices over; one per image, or none
    /// (none means the runs are appended before EOS).
    #[test]
    fn image_markers_must_match_the_image_count_or_be_absent() {
        assert_eq!(IMAGE_MARKER, "<|image|>");
        // No images: nothing to check, whatever the text says.
        assert!(validate_image_markers("<|image|> stray", 0).is_ok());
        // No markers: the append path.
        assert!(validate_image_markers("a photo of ", 1).is_ok());
        // One marker per image, in order.
        assert!(validate_image_markers("<|image|> and <|image|>", 2).is_ok());
        // Mismatch, either direction, names both counts.
        let message = validate_image_markers("<|image|>", 2).expect_err("mismatch is refused");
        assert!(message.contains("1 <|image|> marker"), "{message}");
        assert!(message.contains("2 image"), "{message}");
        let message =
            validate_image_markers("a <|image|> b <|image|> c", 1).expect_err("extra marker");
        assert!(message.contains("2 <|image|> marker"), "{message}");
        assert!(message.contains("1 image"), "{message}");
    }

    /// The audio marker is the same rule one token over: `<|audio|>` (id
    /// 258881), one per clip, or none (append before EOS).
    #[test]
    fn audio_markers_must_match_the_clip_count_or_be_absent() {
        assert_eq!(AUDIO_MARKER, "<|audio|>");
        assert!(validate_audio_markers("<|audio|> stray", 0).is_ok());
        assert!(validate_audio_markers("say this: ", 1).is_ok());
        assert!(validate_audio_markers("<|audio|> <|audio|>", 2).is_ok());
        // A clip request with an image marker in the text is not an audio
        // marker mismatch — the two markers are counted apart.
        assert!(validate_audio_markers("<|image|>", 1).is_ok());
        assert!(validate_image_markers("<|audio|>", 1).is_ok());
        let message = validate_audio_markers("<|audio|>", 2).expect_err("mismatch is refused");
        assert!(message.contains("1 <|audio|> marker"), "{message}");
        assert!(message.contains("2 audio clip"), "{message}");
        let message = validate_audio_markers("<|audio|> a <|audio|>", 1).expect_err("extra marker");
        assert!(message.contains("2 <|audio|> marker"), "{message}");
        assert!(message.contains("1 audio clip"), "{message}");
    }
}
