// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `/v1/embeddings` input lowering: the request's `input` field, in every
//! accepted spelling, onto the daemon `embed` verb's `input`.
//!
//! Accepted:
//!
//! - `"text"` — one text input.
//! - `["text"]` — the same, wrapped in a one-element list.
//! - `{"text": …, "image_base64": …}` (or `{"text": …, "images_base64": […]}`)
//!   — the object form, forwarded untouched: it predates the typed shape, so a
//!   caller that worked before this parser existed keeps byte-identical
//!   behaviour, and the daemon stays the validator of that spelling.
//! - `[{"content": [{"type":"text","text":…},
//!   {"type":"image_url","image_url":{"url":"data:image/…;base64,…"}}]}]` —
//!   the llama.cpp/OpenAI typed form, one content object per input (this
//!   endpoint takes one input per request). Text parts concatenate in order;
//!   image parts are decoded **here**, the route's one decode point (plain
//!   base64 or a data URL), and re-encoded canonical `images_base64`, so
//!   whatever the client sent, the daemon sees plain base64. The same holds
//!   for `input_audio` (→ `audios_base64`) and `input_video`, whose
//!   `input_video.data` is the clip's FRAME payloads in display order
//!   (→ `videos_base64`, one frame list per video).
//!
//! Everything else is refused with the shape it should have used. The one
//! migration worth a paragraph: a *bare* content-part array — the parts
//! without the `{"content": …}` wrapper — is rejected by name, because
//! forwarding it would only earn a generic "input must be a string or an
//! object" from the daemon.

use base64::Engine as _;
use serde_json::{json, Map, Value};

/// Lower one `/v1/embeddings` `input` onto the daemon's `input` value.
pub(crate) fn lower_embeddings_input(input: &Value) -> Result<Value, String> {
    match input {
        Value::String(_) => Ok(input.clone()),
        Value::Object(_) => {
            if input.get("content").is_some() {
                content_object(input)
            } else if is_content_part(input) {
                Err(bare_content_array())
            } else {
                Ok(input.clone())
            }
        }
        Value::Array(items) => {
            if items.iter().any(is_content_part) {
                return Err(bare_content_array());
            }
            match items.as_slice() {
                [one] => match one {
                    Value::String(_) => Ok(one.clone()),
                    Value::Object(_) if one.get("content").is_some() => content_object(one),
                    Value::Object(_) => Ok(one.clone()),
                    Value::Array(_) => Err(bare_content_array()),
                    other => Err(format!(
                        "input entries must be a string or a {{\"content\": […]}} object, \
                         not {}",
                        json_kind(other)
                    )),
                },
                other => Err(format!(
                    "input has {} entries; this endpoint accepts one input per request",
                    other.len()
                )),
            }
        }
        other => Err(format!(
            "input must be a string, a one-element list, or a {{\"content\": […]}} object, \
             not {}",
            json_kind(other)
        )),
    }
}

/// `{"content": [{"type": …}, …]}` — one input's typed parts, in order.
fn content_object(input: &Value) -> Result<Value, String> {
    // A content array IS the whole input: a sibling legacy key would be
    // dropped by the lowering, so it is refused instead of half-read.
    for key in [
        "text",
        "image_base64",
        "images_base64",
        "audios_base64",
        "videos_base64",
    ] {
        if input.get(key).is_some_and(|v| !v.is_null()) {
            return Err(format!(
                "input carries both \"content\" and \"{key}\" — put the text and image parts \
                 inside the content array instead"
            ));
        }
    }
    let parts = input
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or("input \"content\" must be an array of {\"type\": …} parts")?;
    if parts.is_empty() {
        return Err("input \"content\" is empty — send at least one text or image_url part".into());
    }
    let mut text = String::new();
    let mut images: Vec<String> = Vec::new();
    let mut audios: Vec<String> = Vec::new();
    let mut videos: Vec<Vec<String>> = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let kind = part.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
            format!(
                "input content[{i}] has no \"type\" (\"text\", \"image_url\", \"input_audio\" \
                 or \"input_video\")"
            )
        })?;
        match kind {
            "text" => {
                let piece = part.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
                    format!("input content[{i}] is a text part without a \"text\" string")
                })?;
                text.push_str(piece);
            }
            "image_url" => {
                let url = image_url(part).ok_or_else(|| {
                    format!(
                        "input content[{i}] is an image_url part without a URL — send \
                         {{\"type\": \"image_url\", \"image_url\": {{\"url\": \
                         \"data:image/png;base64,…\"}}}}"
                    )
                })?;
                let bytes = decode_base64(url).map_err(|e| format!("input content[{i}]: {e}"))?;
                images.push(base64::engine::general_purpose::STANDARD.encode(bytes));
            }
            "input_audio" => {
                let (data, format) = input_audio(part).ok_or_else(|| {
                    format!(
                        "input content[{i}] is an input_audio part without audio data — send \
                         {{\"type\": \"input_audio\", \"input_audio\": {{\"data\": \
                         \"<base64 WAV>\", \"format\": \"wav\"}}}}"
                    )
                })?;
                // The declared format is checked here because this is the only
                // layer that sees it; the container bytes themselves are the
                // arch crate's call (WAV/RIFF PCM is what it decodes).
                if let Some(format) = format {
                    if format != "wav" {
                        return Err(format!(
                            "input content[{i}] audio format {format:?} unsupported — this \
                             route takes WAV/RIFF (\"wav\")"
                        ));
                    }
                }
                let bytes = decode_base64(data).map_err(|e| format!("input content[{i}]: {e}"))?;
                audios.push(base64::engine::general_purpose::STANDARD.encode(bytes));
            }
            "input_video" => {
                let (frames, format) = input_video(part).ok_or_else(|| {
                    format!(
                        "input content[{i}] is an input_video part without frame data — send \
                         {{\"type\": \"input_video\", \"input_video\": {{\"data\": \
                         [\"data:image/png;base64,…\", …]}}}} (one entry per FRAME, in \
                         display order; a container is not a video input)"
                    )
                })?;
                // The declared format is checked here because this is the only
                // layer that sees it; the frame bytes themselves are the arch
                // crate's call (PNG/JPEG frames are what it decodes).
                if let Some(format) = format {
                    if format != "png" && format != "jpeg" && format != "jpg" {
                        return Err(format!(
                            "input content[{i}] video format {format:?} unsupported — this \
                             route takes decoded PNG/JPEG frames (\"png\" or \"jpeg\")"
                        ));
                    }
                }
                if frames.is_empty() {
                    return Err(format!(
                        "input content[{i}] input_video.data is empty — send at least one \
                         frame"
                    ));
                }
                let mut lowered_frames = Vec::with_capacity(frames.len());
                for (f, frame) in frames.iter().enumerate() {
                    let bytes = decode_base64(frame)
                        .map_err(|e| format!("input content[{i}][{f}]: {e}"))?;
                    lowered_frames.push(base64::engine::general_purpose::STANDARD.encode(bytes));
                }
                videos.push(lowered_frames);
            }
            other => {
                return Err(format!(
                    "input content[{i}] type {other:?} unsupported — this route takes \
                     \"text\", \"image_url\", \"input_audio\" and \"input_video\""
                ))
            }
        }
    }
    let mut out = Map::new();
    out.insert("text".into(), json!(text));
    if !images.is_empty() {
        out.insert("images_base64".into(), json!(images));
    }
    if !audios.is_empty() {
        out.insert("audios_base64".into(), json!(audios));
    }
    if !videos.is_empty() {
        out.insert("videos_base64".into(), json!(videos));
    }
    Ok(Value::Object(out))
}

/// The URL of an `image_url` part: the OpenAI `{"url": …}` object, or the bare
/// string spelling some SDKs send.
fn image_url(part: &Value) -> Option<&str> {
    match part.get("image_url")? {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()),
        _ => None,
    }
}

/// The `{"data": …, "format": …}` of an `input_audio` part. `format` may be
/// absent — the container bytes then speak for themselves.
fn input_audio(part: &Value) -> Option<(&str, Option<&str>)> {
    let obj = part.get("input_audio")?.as_object()?;
    let data = obj.get("data").and_then(|d| d.as_str())?;
    Some((data, obj.get("format").and_then(|f| f.as_str())))
}

/// The `{"data": […frames…], "format": …}` of an `input_video` part: `data` is
/// the list of FRAME payloads (base64 or data URLs), in display order. A bare
/// base64 string is accepted as a one-frame clip; `format` may be absent (the
/// bytes speak for themselves) or name the frame container.
fn input_video(part: &Value) -> Option<(Vec<&str>, Option<&str>)> {
    let obj = part.get("input_video")?.as_object()?;
    let format = obj.get("format").and_then(|f| f.as_str());
    match obj.get("data")? {
        Value::Array(frames) => {
            let mut out = Vec::with_capacity(frames.len());
            for frame in frames {
                out.push(frame.as_str()?);
            }
            Some((out, format))
        }
        Value::String(one) => Some((vec![one.as_str()], format)),
        _ => None,
    }
}

/// Decode one media payload: plain base64, or a `data:<mime>;base64,<payload>`
/// data URL (the OpenAI content-part spelling). Bytes only ever reach the
/// daemon re-encoded from here.
fn decode_base64(payload: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(data_url_payload(payload).trim())
        .map_err(|e| format!("media payload is not valid base64: {e}"))
}

/// The base64 payload of `raw`, dropping a `data:…;base64,` prefix when one is
/// present.
fn data_url_payload(raw: &str) -> &str {
    let raw = raw.trim();
    raw.split_once(',')
        .filter(|(head, _)| head.starts_with("data:") && head.ends_with(";base64"))
        .map(|(_, tail)| tail)
        .unwrap_or(raw)
}

/// A content *part* (`{"type": …}`) rather than a content *object*
/// (`{"content": […]}`).
fn is_content_part(v: &Value) -> bool {
    v.as_object()
        .is_some_and(|o| o.contains_key("type") && !o.contains_key("content"))
}

/// The migration error for parts sent without their wrapper. Names the
/// wrapped shape verbatim so the fix is copy-pasteable.
fn bare_content_array() -> String {
    "input looks like a bare content-part array — wrap the parts in a content object: \
     input: [{\"content\": [{\"type\": \"text\", \"text\": \"…\"}, {\"type\": \"image_url\", \
     \"image_url\": {\"url\": \"data:image/png;base64,…\"}}]}]"
        .to_string()
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lowered(input: Value) -> Value {
        lower_embeddings_input(&input).expect("input lowers")
    }

    fn refused(input: Value) -> String {
        lower_embeddings_input(&input).expect_err("input is refused")
    }

    #[test]
    fn a_bare_string_lowers_to_itself() {
        assert_eq!(lowered(json!("hello")), json!("hello"));
    }

    #[test]
    fn a_one_element_string_list_lowers_to_the_string() {
        // Legacy: the daemon sees the same single string either way.
        assert_eq!(lowered(json!(["hello"])), json!("hello"));
    }

    #[test]
    fn a_longer_list_is_refused_by_its_length() {
        let message = refused(json!(["a", "b"]));
        assert!(
            message.contains("2 entries") && message.contains("one input per request"),
            "{message}"
        );
    }

    #[test]
    fn the_legacy_object_form_is_forwarded_untouched() {
        let legacy = json!({"text": "t", "image_base64": "AAAA"});
        assert_eq!(lowered(legacy.clone()), legacy);
        let list = json!({"text": "t", "images_base64": ["AAAA"]});
        assert_eq!(lowered(list.clone()), list);
    }

    #[test]
    fn a_content_object_lowers_to_text_plus_canonical_base64() {
        // PNG magic bytes, base64 — the URL form is not passed through.
        let png = [0x89u8, 0x50, 0x4E, 0x47];
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png)
        );
        let out = lowered(json!([{
            "content": [
                {"type": "text", "text": "describe "},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "this"}
            ]
        }]));
        assert_eq!(out["text"], json!("describe this"));
        assert_eq!(
            out["images_base64"],
            json!([base64::engine::general_purpose::STANDARD.encode(png)]),
            "the wire carries plain base64, not the data URL"
        );
    }

    #[test]
    fn a_content_object_mixing_in_a_legacy_key_is_refused() {
        let message = refused(json!([{
            "content": [{"type": "text", "text": "a"}],
            "text": "b"
        }]));
        assert!(message.contains("\"content\" and \"text\""), "{message}");
    }

    #[test]
    fn a_bare_image_url_string_part_is_accepted() {
        let out = lowered(json!({"content": [{"type": "image_url", "image_url": "QUJD"}]}));
        assert_eq!(out["images_base64"], json!(["QUJD"]));
    }

    #[test]
    fn an_image_only_content_object_has_an_empty_text() {
        let out = lowered(json!([{"content": [{"type": "image_url", "image_url": "QUJD"}]}]));
        assert_eq!(out["text"], json!(""));
        assert_eq!(out["images_base64"], json!(["QUJD"]));
    }

    #[test]
    fn a_bare_content_part_array_is_rejected_with_the_wrapped_shape() {
        let message = refused(json!([{"type": "text", "text": "hello"}]));
        assert!(message.contains("bare content-part array"), "{message}");
        assert!(message.contains("{\"content\":"), "{message}");
        // The unwrapped object spelling is the same caller error.
        let message = refused(json!({"type": "text", "text": "hello"}));
        assert!(message.contains("bare content-part array"), "{message}");
        // ... and so is an array of the parts.
        let message = refused(json!([
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": "QUJD"}
        ]));
        assert!(message.contains("bare content-part array"), "{message}");
    }

    #[test]
    fn an_unknown_part_type_names_the_supported_ones() {
        let message = refused(json!([{"content": [{"type": "input_file", "data": "…"}]}]));
        assert!(
            message.contains("input_file")
                && message.contains("\"text\", \"image_url\", \"input_audio\" and \"input_video\""),
            "{message}"
        );
    }

    /// Video parts lower to `videos_base64` — one FRAME LIST per video, in
    /// part order, canonical plain base64 — beside the other media lists.
    #[test]
    fn an_input_video_part_lowers_to_frame_lists() {
        let out = lowered(json!([{
            "content": [
                {"type": "text", "text": "describe "},
                {"type": "input_video", "input_video": {
                    "data": ["QUJD", "data:image/png;base64,REVG"]
                }},
                {"type": "input_video", "input_video": {"data": "R0hJ"}}
            ]
        }]));
        assert_eq!(out["text"], json!("describe "));
        assert_eq!(
            out["videos_base64"],
            json!([["QUJD", "REVG"], ["R0hJ"]]),
            "one frame list per video, data URLs re-encoded canonical"
        );
        assert!(out.get("images_base64").is_none(), "no images: no field");
        assert!(out.get("audios_base64").is_none(), "no audio: no field");
    }

    #[test]
    fn a_blank_or_broken_video_part_is_refused_by_index() {
        let message = refused(json!([{
            "content": [{"type": "input_video", "input_video": {"data": []}}]
        }]));
        assert!(
            message.contains("content[0]") && message.contains("empty"),
            "{message}"
        );

        let message = refused(json!([{
            "content": [{"type": "input_video", "input_video": {"data": [7]}}]
        }]));
        assert!(message.contains("input_video"), "{message}");

        let message = refused(json!([{
            "content": [{"type": "input_video", "input_video": {"format": "png"}}]
        }]));
        assert!(
            message.contains("\"data\"") && message.contains("FRAME"),
            "{message}"
        );

        let message = refused(json!([{
            "content": [{"type": "input_video", "input_video": {"data": ["!!"]}}]
        }]));
        assert!(message.contains("not valid base64"), "{message}");
    }

    #[test]
    fn a_container_video_format_is_refused_at_the_route() {
        let message = refused(json!([{
            "content": [{"type": "input_video", "input_video": {"data": ["QUJD"], "format": "mp4"}}]
        }]));
        assert!(message.contains("\"mp4\""), "{message}");
        assert!(message.contains("PNG/JPEG"), "{message}");
    }

    #[test]
    fn a_content_object_mixing_in_videos_base64_is_refused() {
        let message = refused(json!([{
            "content": [{"type": "text", "text": "a"}],
            "videos_base64": [["QUJD"]]
        }]));
        assert!(
            message.contains("\"content\" and \"videos_base64\""),
            "{message}"
        );
    }

    /// Audio parts lower to `audios_base64` beside `images_base64`, OpenAI
    /// shape (`input_audio.data` + `format`), data URL accepted, and the
    /// declared format checked where it can be seen.
    #[test]
    fn an_input_audio_part_lowers_to_canonical_base64() {
        let out = lowered(json!([{
            "content": [
                {"type": "text", "text": "transcribe "},
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,REVG"}}
            ]
        }]));
        assert_eq!(out["text"], json!("transcribe "));
        assert_eq!(out["audios_base64"], json!(["QUJD", "REVG"]));
        assert!(out.get("images_base64").is_none(), "no images: no field");
        // A missing format is left to the container check downstream.
        let out = lowered(json!([{
            "content": [{"type": "input_audio", "input_audio": {"data": "QUJD"}}]
        }]));
        assert_eq!(out["audios_base64"], json!(["QUJD"]));
    }

    #[test]
    fn a_non_wav_audio_format_is_refused_at_the_route() {
        let message = refused(json!([{
            "content": [
                {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}}
            ]
        }]));
        assert!(message.contains("\"mp3\""), "{message}");
        assert!(message.contains("WAV/RIFF"), "{message}");
    }

    #[test]
    fn an_input_audio_part_without_data_names_the_shape() {
        let message = refused(json!([{
            "content": [{"type": "input_audio", "input_audio": {"format": "wav"}}]
        }]));
        assert!(message.contains("input_audio"), "{message}");
        assert!(message.contains("\"data\""), "{message}");
    }

    #[test]
    fn an_invalid_data_url_is_refused_at_the_route() {
        let message = refused(json!([{
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!"}}]
        }]));
        assert!(message.contains("not valid base64"), "{message}");
    }

    #[test]
    fn an_empty_content_array_is_refused() {
        let message = refused(json!([{"content": []}]));
        assert!(message.contains("empty"), "{message}");
    }

    #[test]
    fn a_text_part_without_its_string_is_refused_by_index() {
        let message = refused(json!([{
            "content": [{"type": "text"}, {"type": "image_url", "image_url": "QUJD"}]
        }]));
        assert!(message.contains("content[0]"), "{message}");
    }

    #[test]
    fn an_image_url_part_without_a_url_names_the_shape() {
        let message = refused(json!([{"content": [{"type": "image_url"}]}]));
        assert!(message.contains("image_url"), "{message}");
        assert!(message.contains("data:image/png;base64"), "{message}");
    }

    #[test]
    fn data_url_prefix_stripping_is_prefix_only() {
        assert_eq!(data_url_payload("data:image/png;base64,QUJD"), "QUJD");
        assert_eq!(data_url_payload(" QUJD "), "QUJD");
        // A comma that is part of the payload is not a data-URL separator.
        assert_eq!(
            data_url_payload("datax:image/png;base64,QUJD"),
            "datax:image/png;base64,QUJD"
        );
    }
}
