// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HTTP gateway: OpenAI chat completions, plus the Anthropic Messages and
//! OpenAI Responses adapters onto the same pipeline.
//!
//! Concern: request routing, JSON body limits, streaming vs non-streaming
//! framing, SSE acknowledgement, CORS/health endpoints. Isolates Hyper
//! I/O from business logic.

use crate::serve::complete::{
    complete_request_cancellable, completion_json, gate_endpoint_tools,
    openai_stream_delta_for_event, openai_stream_terminal_chunks, Completion, EndpointAdapterKind,
};
use crate::serve::metrics::Metrics;
use crate::serve::ocr::{self, OcrFormat};
use crate::serve::{anthropic, responses};
use crate::serve::{is_batch_eligible_request, ServeShared};
use crate::serve::{AdmissionError, AdmissionGuard, LoadedInfo};
use crate::{grandfather_table, list_local_models, unix_timestamp};
use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use hipfire_config::secret::BearerToken;
use hipfire_registry::is_standalone_model;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::server::conn::http1;
use hyper::{
    body::{Frame, Incoming},
    header, Method, Request, Response,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    convert::Infallible,
    future::Future,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

// ---------------------------------------------------------------------------
// Boxed body and helpers
// ---------------------------------------------------------------------------

pub(crate) type BoxBody = UnsyncBoxBody<Bytes, std::io::Error>;

fn boxed<B>(body: B) -> BoxBody
where
    B: hyper::body::Body<Data = Bytes, Error = std::io::Error> + Send + 'static,
{
    UnsyncBoxBody::new(body)
}

pub(crate) fn boxed_full(bytes: Vec<u8>) -> BoxBody {
    boxed(Full::new(bytes.into()).map_err(|never: Infallible| match never {}))
}

fn boxed_empty() -> BoxBody {
    boxed(Full::new(Bytes::new()).map_err(|never: Infallible| match never {}))
}

pub(crate) fn json_response(value: serde_json::Value, status: u16) -> Response<BoxBody> {
    match json_response_result(&value, status) {
        Ok(resp) => resp,
        Err(message) => openai_error(&message, 500),
    }
}

fn json_response_result(
    value: &serde_json::Value,
    status: u16,
) -> Result<Response<BoxBody>, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| format!("failed to encode JSON response: {err}"))?;
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed_full(bytes))
        .map_err(|err| format!("failed to build HTTP response: {err}"))
}

/// Last-resort 500 body with no serde dependency, so error rendering always
/// terminates even if JSON encoding itself is what failed.
fn static_server_error() -> Response<BoxBody> {
    Response::builder()
        .status(500)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed_full(
            br#"{"error":{"message":"internal server error","type":"server_error"}}"#.to_vec(),
        ))
        // Static status, headers, and body: the builder cannot fail on these
        // inputs, and there is no further fallback below this point.
        .expect("static 500 response builds")
}

fn openai_error_body(message: &str, status: u16) -> serde_json::Value {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    serde_json::json!({
        "error": { "message": message, "type": error_type }
    })
}

pub(crate) fn openai_error(message: &str, status: u16) -> Response<BoxBody> {
    json_response_result(&openai_error_body(message, status), status)
        .unwrap_or_else(|_| static_server_error())
}

pub(crate) fn admission_error_response(
    api: WireApi,
    metrics: &Metrics,
    error: &AdmissionError,
) -> Response<BoxBody> {
    metrics.record_admission_rejected();
    let mut resp = api.error_response(&error.message, 503);
    if let Ok(retry_after) = header::HeaderValue::from_str(&error.retry_after_seconds.to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, retry_after);
    }
    resp
}

// ---------------------------------------------------------------------------
// Generation APIs — every one lowers onto the chat-completions pipeline
// ---------------------------------------------------------------------------

/// The HTTP API a generation request arrived on. Each lowers onto the
/// chat-completions body and then shares that route's admission, tool gate
/// and worker; only request lowering and response rendering differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireApi {
    /// `POST /v1/chat/completions`.
    ChatCompletions,
    /// `POST /v1/messages` ([`anthropic`]).
    AnthropicMessages,
    /// `POST /v1/responses` ([`responses`]).
    OpenAiResponses,
}

impl WireApi {
    fn adapter_kind(self) -> EndpointAdapterKind {
        match self {
            Self::ChatCompletions => EndpointAdapterKind::OpenAiChatCompletions,
            Self::AnthropicMessages => EndpointAdapterKind::AnthropicMessages,
            Self::OpenAiResponses => EndpointAdapterKind::OpenAiResponses,
        }
    }

    /// An error response in this API's shape: Anthropic's
    /// `{"type": "error", "error": {...}}`, else OpenAI's `{"error": {...}}`.
    pub(crate) fn error_response(self, message: &str, status: u16) -> Response<BoxBody> {
        match self {
            Self::AnthropicMessages => {
                json_response_result(&anthropic::error_body(message, status), status)
                    .unwrap_or_else(|_| static_server_error())
            }
            Self::ChatCompletions | Self::OpenAiResponses => openai_error(message, status),
        }
    }

    /// Lower a request body onto the chat-completions body. A chat request
    /// passes through untouched. `Err` is a client error (400).
    fn lower(
        self,
        body: serde_json::Value,
        ocr_default: Option<OcrFormat>,
    ) -> Result<GenerationRequest, String> {
        match self {
            Self::ChatCompletions => {
                let mut body = body;
                let stream = body.get("stream").and_then(|v| v.as_bool()) == Some(true);
                let include_usage = body
                    .pointer("/stream_options/include_usage")
                    .and_then(|v| v.as_bool())
                    == Some(true);
                // `ocr_format` is answered by this hop and never forwarded: it
                // selects a rendering of the model's compact-block output, not
                // a decoder setting. With no request field the serve-level
                // default (`serve.ocr_format` / HIPFIRE_SERVE_OCR_FORMAT)
                // applies; `"raw"` opts a single request out of that default.
                // Streams decode too (incrementally, at block granularity).
                let ocr = take_ocr(&mut body, ocr_default)?;
                Ok(GenerationRequest {
                    chat: body,
                    stream,
                    encoding: ResponseEncoding::Chat { include_usage, ocr },
                })
            }
            Self::AnthropicMessages => {
                let mut body = body;
                let ocr = take_ocr(&mut body, ocr_default)?;
                let lowered = anthropic::lower_request(&body)?;
                Ok(GenerationRequest {
                    chat: lowered.chat,
                    stream: lowered.stream,
                    encoding: ResponseEncoding::Anthropic { ocr },
                })
            }
            Self::OpenAiResponses => {
                let mut body = body;
                let ocr = take_ocr(&mut body, ocr_default)?;
                let lowered = responses::lower_request(&body)?;
                Ok(GenerationRequest {
                    chat: lowered.chat,
                    stream: lowered.stream,
                    encoding: ResponseEncoding::Responses {
                        echo: lowered.echo,
                        ocr,
                    },
                })
            }
        }
    }
}

/// A generation request lowered onto the chat-completions body.
struct GenerationRequest {
    chat: serde_json::Value,
    stream: bool,
    encoding: ResponseEncoding,
}

/// How a completion (and its stream) is rendered for the client's API.
enum ResponseEncoding {
    Chat {
        include_usage: bool,
        /// Decode a compact-block (Surya-2.5) answer before returning it.
        ocr: Option<OcrFormat>,
    },
    Anthropic {
        ocr: Option<OcrFormat>,
    },
    Responses {
        echo: responses::ResponseEcho,
        ocr: Option<OcrFormat>,
    },
}

impl ResponseEncoding {
    fn api(&self) -> WireApi {
        match self {
            Self::Chat { .. } => WireApi::ChatCompletions,
            Self::Anthropic { .. } => WireApi::AnthropicMessages,
            Self::Responses { .. } => WireApi::OpenAiResponses,
        }
    }

    /// The compact-block rendering this response asks for, if any. Streams
    /// decode at the sink, before the API's encoder frames a delta.
    fn ocr(&self) -> Option<OcrFormat> {
        match self {
            Self::Chat { ocr, .. } | Self::Anthropic { ocr } | Self::Responses { ocr, .. } => *ocr,
        }
    }

    /// The non-streaming response body.
    fn completion_body(&self, completion: &Completion) -> serde_json::Value {
        match self {
            Self::Chat { ocr, .. } => {
                let mut body = completion_json(completion);
                if let Some(format) = ocr {
                    decode_ocr_content(&mut body, *format);
                }
                body
            }
            Self::Anthropic { ocr } => {
                let mut body = anthropic::message_json(completion);
                if let Some(format) = ocr {
                    decode_anthropic_content(&mut body, *format);
                }
                body
            }
            Self::Responses { echo, ocr } => {
                let mut body = responses::response_json(completion, echo);
                if let Some(format) = ocr {
                    decode_responses_content(&mut body, *format);
                }
                body
            }
        }
    }

    fn stream(self, id: &str, created: u64, model: &str) -> StreamEncoder {
        match self {
            Self::Chat { include_usage, .. } => StreamEncoder::Chat {
                id: id.to_owned(),
                created,
                model: model.to_owned(),
                include_usage,
            },
            Self::Anthropic { .. } => {
                StreamEncoder::Anthropic(anthropic::MessageStream::new(id, model))
            }
            Self::Responses { echo, .. } => {
                StreamEncoder::Responses(responses::ResponseStream::new(echo, id, created, model))
            }
        }
    }
}

/// Pull the request's `ocr_format` (an extension on every wire API) and fall
/// back to the serve default. `"raw"` opts one request out of that default.
fn take_ocr(
    body: &mut serde_json::Value,
    default: Option<OcrFormat>,
) -> Result<Option<OcrFormat>, String> {
    let Some(value) = body
        .as_object_mut()
        .and_then(|body| body.remove("ocr_format"))
    else {
        return Ok(default);
    };
    let Some(requested) = value.as_str() else {
        return Err(format!(
            "ocr_format must be {}, got {value}",
            OcrFormat::wire_values()
        ));
    };
    if requested == "raw" {
        return Ok(None);
    }
    OcrFormat::parse(requested).map(Some).ok_or_else(|| {
        format!(
            "ocr_format must be {}, got {requested:?}",
            OcrFormat::wire_values()
        )
    })
}

/// Decode one text slot in place; a plain answer is left alone.
fn decode_text_slot(slot: &mut serde_json::Value, format: OcrFormat) {
    let Some(text) = slot.as_str() else {
        return;
    };
    let decoded = match format {
        OcrFormat::Html => ocr::decode_html(text),
        OcrFormat::Blocks => ocr::blocks_json(text),
    };
    if decoded != text {
        *slot = serde_json::Value::String(decoded);
    }
}

/// Anthropic `message.content[]` text blocks (and `thinking` blocks: a model
/// served without its non-thinking default answers on the reasoning channel).
fn decode_anthropic_content(body: &mut serde_json::Value, format: OcrFormat) {
    let Some(items) = body
        .get_mut("content")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for item in items {
        match item.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get_mut("text") {
                    decode_text_slot(text, format);
                }
            }
            Some("thinking") => {
                if let Some(text) = item.get_mut("thinking") {
                    decode_text_slot(text, format);
                }
            }
            _ => {}
        }
    }
}

/// Responses `output[].content[]` parts (and the flattened `output_text`).
fn decode_responses_content(body: &mut serde_json::Value, format: OcrFormat) {
    if let Some(items) = body
        .get_mut("output")
        .and_then(serde_json::Value::as_array_mut)
    {
        for item in items {
            let Some(parts) = item
                .get_mut("content")
                .and_then(serde_json::Value::as_array_mut)
            else {
                continue;
            };
            for part in parts {
                if matches!(
                    part.get("type").and_then(serde_json::Value::as_str),
                    Some("output_text" | "text" | "reasoning_text")
                ) {
                    if let Some(text) = part.get_mut("text") {
                        decode_text_slot(text, format);
                    }
                }
            }
        }
    }
    if let Some(text) = body.get_mut("output_text") {
        decode_text_slot(text, format);
    }
}

/// Which channel a delta belongs to. Reasoning and content are decoded
/// separately: one shared decoder would emit reasoning text on the content
/// channel — as the lead-in before the first block, or as the whole answer at
/// end of generation when the answer never carried a block header — instead of
/// once, on the channel the model wrote it to.
#[derive(Clone, Copy)]
enum StreamChannelKind {
    Content,
    Reasoning,
}

/// Per-response stream decode state, owned by the sink so every wire API
/// (chat, Anthropic, Responses) sees the same decoded text deltas.
struct StreamOcr {
    content: StreamChannel,
    reasoning: StreamChannel,
}

impl StreamOcr {
    fn new(format: OcrFormat) -> Self {
        Self {
            content: StreamChannel::new(format),
            reasoning: StreamChannel::new(format),
        }
    }

    fn channel_mut(&mut self, channel: StreamChannelKind) -> &mut StreamChannel {
        match channel {
            StreamChannelKind::Content => &mut self.content,
            StreamChannelKind::Reasoning => &mut self.reasoning,
        }
    }

    /// Feed a delta from `channel`; `Some(text)` when the client should receive
    /// a frame, on that same channel.
    fn push(&mut self, channel: StreamChannelKind, text: &str) -> Option<String> {
        self.channel_mut(channel).push(text)
    }

    /// The remainder of each channel at end of generation.
    fn finish(&mut self) -> [String; 2] {
        [self.content.finish(), self.reasoning.finish()]
    }
}

/// One channel's decoder.
struct StreamChannel {
    /// No text seen: an untouched channel sends nothing at end of generation.
    seen: bool,
    state: StreamChannelState,
}

enum StreamChannelState {
    Html(ocr::BlockStream),
    /// `blocks` has to buffer: a JSON array has no prefix form.
    Blocks(String),
}

impl StreamChannel {
    fn new(format: OcrFormat) -> Self {
        let state = match format {
            OcrFormat::Html => StreamChannelState::Html(ocr::BlockStream::new()),
            OcrFormat::Blocks => StreamChannelState::Blocks(String::new()),
        };
        Self { seen: false, state }
    }

    fn push(&mut self, text: &str) -> Option<String> {
        self.seen = true;
        match &mut self.state {
            StreamChannelState::Html(stream) => {
                let chunk = stream.push(text);
                (!chunk.is_empty()).then_some(chunk)
            }
            StreamChannelState::Blocks(buffer) => {
                buffer.push_str(text);
                None
            }
        }
    }

    fn finish(&mut self) -> String {
        if !self.seen {
            return String::new();
        }
        match &mut self.state {
            StreamChannelState::Html(stream) => stream.finish(),
            StreamChannelState::Blocks(buffer) => ocr::blocks_json(buffer),
        }
    }
}

/// Replace `choices[0].message.content` with the decoded form of a
/// compact-block answer. Content that carries no block headers is left as it
/// is (see [`ocr::decode_html`]).
fn decode_ocr_content(body: &mut serde_json::Value, format: OcrFormat) {
    for slot in [
        "/choices/0/message/content",
        "/choices/0/message/reasoning_content",
    ] {
        if let Some(text) = body.pointer_mut(slot) {
            decode_text_slot(text, format);
        }
    }
}

/// SSE framing of one streamed response in the client's API.
enum StreamEncoder {
    /// OpenAI `data: {chat.completion.chunk}` frames ending in `data: [DONE]`.
    Chat {
        id: String,
        created: u64,
        model: String,
        include_usage: bool,
    },
    Anthropic(anthropic::MessageStream),
    Responses(responses::ResponseStream),
}

impl StreamEncoder {
    /// The frame the 200 commits with (chat role chunk, `message_start`,
    /// `response.created`).
    fn preamble(&mut self) -> Vec<u8> {
        match self {
            Self::Chat {
                id, created, model, ..
            } => sse_data(&serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
            })),
            Self::Anthropic(stream) => stream.preamble(),
            Self::Responses(stream) => stream.preamble(),
        }
    }

    /// One folded pipeline event; `None` when it has nothing to send.
    fn event(&mut self, event: &serde_json::Value) -> Option<Vec<u8>> {
        match self {
            Self::Chat {
                id, created, model, ..
            } => {
                let delta = openai_stream_delta_for_event(event)?;
                Some(sse_data(&serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                })))
            }
            Self::Anthropic(stream) => stream.event(event),
            Self::Responses(stream) => stream.event(event),
        }
    }

    /// The terminal frame: released tool calls, finish and usage.
    fn terminal(&mut self, completion: &Completion) -> Vec<u8> {
        match self {
            Self::Chat { include_usage, .. } => {
                let mut bytes = Vec::new();
                for chunk in openai_stream_terminal_chunks(completion, *include_usage) {
                    bytes.extend_from_slice(&sse_data(&chunk));
                }
                bytes.extend_from_slice(b"data: [DONE]\n\n");
                bytes
            }
            Self::Anthropic(stream) => stream.terminal(completion),
            Self::Responses(stream) => stream.terminal(completion),
        }
    }

    /// The frame ending a stream that fails after it committed.
    fn failure(&mut self, failure: &RequestFailure) -> Vec<u8> {
        match self {
            Self::Chat { .. } => failure.sse_event(),
            Self::Anthropic(stream) => stream.failure(&failure.message, failure.status),
            Self::Responses(stream) => stream.failure(&failure.message, failure.status),
        }
    }
}

// ---------------------------------------------------------------------------
// FlushAcks / TrackedIo — ack only after socket flush
// ---------------------------------------------------------------------------

type AckSender = std::sync::mpsc::Sender<Result<(), ()>>;

/// Per-connection FIFO of terminal-ack senders. Bodies register when they yield
/// a frame; `TrackedIo` completes them only after a successful `poll_flush`.
#[derive(Clone, Debug, Default)]
struct FlushAcks {
    queue: Arc<Mutex<Vec<AckSender>>>,
}

impl FlushAcks {
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn register(&self, ack: AckSender) {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ack);
    }

    fn drain_ok(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Ok(()));
        }
    }

    fn drain_err(&self) {
        let pending = std::mem::take(&mut *self.queue.lock().unwrap_or_else(|e| e.into_inner()));
        for ack in pending {
            let _ = ack.send(Err(()));
        }
    }
}

/// TcpStream wrapper that ties terminal acks to successful socket flushes.
struct TrackedIo {
    inner: TcpStream,
    acks: FlushAcks,
}

impl TrackedIo {
    fn new(inner: TcpStream, acks: FlushAcks) -> Self {
        Self { inner, acks }
    }
}

impl Drop for TrackedIo {
    fn drop(&mut self) {
        self.acks.drain_err();
    }
}

impl AsyncRead for TrackedIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_flush(cx);
        match &result {
            Poll::Ready(Ok(())) => self.acks.drain_ok(),
            Poll::Ready(Err(_)) => self.acks.drain_err(),
            Poll::Pending => {}
        }
        result
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        let result = Pin::new(&mut self.inner).poll_shutdown(cx);
        if let Poll::Ready(Err(_)) = &result {
            self.acks.drain_err();
        }
        result
    }
}

// ---------------------------------------------------------------------------
// AckBody / ChannelBody — preserve commit boundary
// ---------------------------------------------------------------------------

/// Terminal JSON body: exactly one frame. Ack is registered with the connection
/// tracker when the frame is yielded; Ok only after `TrackedIo` flushes. Drop
/// before registration sends Err (correlated abort).
pub(crate) struct AckBody {
    data: Option<Bytes>,
    ack: Option<AckSender>,
    tracker: FlushAcks,
}

impl AckBody {
    pub(crate) fn new(bytes: Vec<u8>, ack: AckSender, tracker: FlushAcks) -> Self {
        Self {
            data: Some(bytes.into()),
            ack: Some(ack),
            tracker,
        }
    }
}

impl Drop for AckBody {
    fn drop(&mut self) {
        if let Some(ack) = self.ack.take() {
            let _ = ack.send(Err(()));
        }
    }
}

impl hyper::body::Body for AckBody {
    type Data = Bytes;
    type Error = std::io::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if let Some(data) = self.data.take() {
            // Yield the frame and hand the ack to the connection tracker. Ok
            // arrives only after a later successful socket flush — never here.
            if let Some(ack) = self.ack.take() {
                self.tracker.register(ack);
            }
            return Poll::Ready(Some(Ok(Frame::data(data))));
        }
        Poll::Ready(None)
    }
}

/// One SSE frame: plain bytes, optional terminal ack sender, or fail marker.
#[derive(Debug)]
pub(crate) struct ResponseChunk {
    bytes: Vec<u8>,
    ack: Option<AckSender>,
    fail: bool,
    /// The response's final frame: nothing, not even a keepalive, may follow.
    last: bool,
}

impl ResponseChunk {
    pub(crate) fn plain(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ack: None,
            fail: false,
            last: false,
        }
    }
    /// The final frame of a response (chat: ends with `data: [DONE]`).
    fn last(bytes: Vec<u8>, ack: Option<AckSender>) -> Self {
        Self {
            bytes,
            ack,
            fail: false,
            last: true,
        }
    }
    pub(crate) fn fail() -> Self {
        Self {
            bytes: Vec::new(),
            ack: None,
            fail: true,
            last: false,
        }
    }
}

/// SSE comment sent after [`SSE_SILENCE_LIMIT`] without a frame. Conforming
/// SSE parsers (the OpenAI SDKs, `hipfire_client::read_openai_sse`) drop it.
const SSE_KEEPALIVE: &[u8] = b": keepalive\n\n";

/// Streaming SSE body: frames the handler already holds (`first`), then the
/// channel. Dropped receiver closes sender and callback returns Cancelled.
/// Terminal chunk ack is registered with the connection tracker when that
/// frame is yielded (not on a later body poll).
/// Until the last frame, a silence of `keepalive_every` (a long prefill, or a
/// tool call buffered to the end) yields an [`SSE_KEEPALIVE`] comment, so
/// clients and proxies with an idle-read timeout keep the connection.
/// Owns a clone of the worker cancellation flag; drop (client disconnect after
/// the response is returned) sets it so long silent towers abort promptly.
pub(crate) struct ChannelBody {
    first: VecDeque<ResponseChunk>,
    rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
    tracker: FlushAcks,
    cancelled: Arc<AtomicBool>,
    failed: bool,
    /// The last frame (`[DONE]`) has been yielded.
    done: bool,
    keepalive: Pin<Box<tokio::time::Sleep>>,
    keepalive_every: Duration,
}

impl ChannelBody {
    pub(crate) fn new(
        first: VecDeque<ResponseChunk>,
        rx: tokio::sync::mpsc::Receiver<ResponseChunk>,
        tracker: FlushAcks,
        cancelled: Arc<AtomicBool>,
        keepalive_every: Duration,
    ) -> Self {
        Self {
            first,
            rx,
            tracker,
            cancelled,
            failed: false,
            done: false,
            keepalive: Box::pin(tokio::time::sleep(keepalive_every)),
            keepalive_every,
        }
    }

    fn restart_keepalive(&mut self) {
        let next = tokio::time::Instant::now() + self.keepalive_every;
        self.keepalive.as_mut().reset(next);
    }
}

impl Drop for ChannelBody {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl hyper::body::Body for ChannelBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.failed {
            return Poll::Ready(Some(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "response body failed after terminal delivery",
            ))));
        }

        let chunk = match self.first.pop_front() {
            Some(chunk) => Poll::Ready(Some(chunk)),
            None => Pin::new(&mut self.rx).poll_recv(cx),
        };
        match chunk {
            Poll::Ready(Some(chunk)) => {
                if chunk.fail {
                    self.failed = true;
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    return Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "response body failed after terminal delivery",
                    ))));
                }
                if chunk.bytes.is_empty() {
                    // Empty chunks carry no wire bytes — never fire ack for empty.
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    // Poll again for next chunk.
                    return self.poll_frame(cx);
                }
                // Register terminal ack with the connection tracker at yield time.
                // Do not ack on a subsequent body poll — only TrackedIo::poll_flush.
                if let Some(ack) = chunk.ack {
                    self.tracker.register(ack);
                }
                self.done |= chunk.last;
                self.restart_keepalive();
                Poll::Ready(Some(Ok(Frame::data(Bytes::from(chunk.bytes)))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => {
                if !self.done && self.keepalive.as_mut().poll(cx).is_ready() {
                    // The next poll_frame polls the re-armed timer again.
                    self.restart_keepalive();
                    return Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(SSE_KEEPALIVE)))));
                }
                Poll::Pending
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cancellation guard — Hyper drops handler future on client FIN in 0ms
// ---------------------------------------------------------------------------

struct CancelOnDrop {
    token: CancellationToken,
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl CancelOnDrop {
    fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            armed: true,
        }
    }

    fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    fn cancelled(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::SeqCst);
            self.token.cancel();
        }
    }
}

// ---------------------------------------------------------------------------
// Public Hyper entry point
// ---------------------------------------------------------------------------

/// Most client connections served at once. Further connects wait in the
/// kernel's listen backlog until one closes, so idle sockets cannot use up
/// the process's descriptors (the default soft limit is 1024).
const MAX_CONNECTIONS: usize = 512;

/// A client must deliver a complete request head this soon after it connects
/// or after its previous response; hyper then closes the connection. This is
/// also what reaps idle keep-alive connections.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Pause after a failed `accept`, doubled per consecutive failure.
const ACCEPT_BACKOFF_MIN: Duration = Duration::from_millis(10);
const ACCEPT_BACKOFF_MAX: Duration = Duration::from_secs(1);

/// After shutdown, how long in-flight requests get to finish.
pub(crate) const SHUTDOWN_DRAIN: Duration = Duration::from_secs(30);

/// Accept loop. A failed `accept` (out of descriptors, a connection reset
/// before it was accepted) is logged and retried, never fatal. When
/// `shutdown` is cancelled the loop stops accepting and releases the port,
/// lets requests in flight finish for up to [`SHUTDOWN_DRAIN`], and returns.
pub(crate) async fn serve_listener_until(
    listener: TcpListener,
    shared: Arc<ServeShared>,
    shutdown: CancellationToken,
) -> Result<()> {
    let slots = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    let connections = TaskTracker::new();
    let mut backoff = ACCEPT_BACKOFF_MIN;
    loop {
        let slot = tokio::select! {
            _ = shutdown.cancelled() => break,
            slot = Arc::clone(&slots).acquire_owned() => {
                slot.expect("the connection semaphore is never closed")
            }
        };
        let accepted = tokio::select! {
            _ = shutdown.cancelled() => break,
            accepted = listener.accept() => accepted,
        };
        match accepted {
            Ok((stream, _)) => {
                backoff = ACCEPT_BACKOFF_MIN;
                connections.spawn(serve_connection(
                    stream,
                    Arc::clone(&shared),
                    shutdown.clone(),
                    slot,
                ));
            }
            Err(error) => {
                eprintln!(
                    "[hipfire] accept failed: {error}; retrying in {} ms",
                    backoff.as_millis()
                );
                tokio::select! {
                    _ = shutdown.cancelled() => break,
                    _ = tokio::time::sleep(backoff) => {}
                }
                backoff = (backoff * 2).min(ACCEPT_BACKOFF_MAX);
            }
        }
    }
    drop(listener);
    connections.close();
    if tokio::time::timeout(SHUTDOWN_DRAIN, connections.wait())
        .await
        .is_err()
    {
        eprintln!(
            "[hipfire] shutdown: requests still in flight after {} s; exiting",
            SHUTDOWN_DRAIN.as_secs()
        );
    }
    Ok(())
}

/// One client connection, holding one of the [`MAX_CONNECTIONS`] slots. On
/// shutdown it stops keep-alive: an idle connection closes at once, a busy
/// one after its response.
async fn serve_connection(
    stream: TcpStream,
    shared: Arc<ServeShared>,
    shutdown: CancellationToken,
    _slot: OwnedSemaphorePermit,
) {
    // One tracker per connection so pipelined responses share FIFO
    // flush ordering without global state.
    let acks = FlushAcks::new();
    let io = TrackedIo::new(stream, acks.clone());
    let service = hyper::service::service_fn(move |req: Request<Incoming>| {
        let shared = Arc::clone(&shared);
        let acks = acks.clone();
        async move { Ok::<_, Infallible>(handle_request(req, shared, acks).await) }
    });
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_READ_TIMEOUT);
    let conn = builder.serve_connection(TokioIo::new(io), service);
    tokio::pin!(conn);
    let mut draining = false;
    let result = loop {
        tokio::select! {
            result = conn.as_mut() => break result,
            _ = shutdown.cancelled(), if !draining => {
                draining = true;
                conn.as_mut().graceful_shutdown();
            }
        }
    };
    if let Err(err) = result {
        // Normal closes stay quiet: resets, idle connections reaped by the
        // header timeout, and a client closing while its response is still
        // open (IncompleteMessage). The last one is every streaming client
        // that stops reading at `data: [DONE]`: the chunked terminator is
        // written only after the daemon commit succeeds, so the body can
        // still fail if the commit does. A close before `[DONE]` is a client
        // cancel, which the body's drop already aborts.
        let msg = err.to_string();
        if !err.is_timeout() && !err.is_incomplete_message() && !msg.contains("reset") {
            eprintln!("[hipfire] connection error: {err:#}");
        }
    }
}

/// `/stats` `loaded_config`: the KV mode, KV backend and context the daemon
/// reported in the resident model's load ack. `null` while no model is
/// resident; individual fields are `null` when the ack did not carry them.
fn loaded_config_json(meta: &crate::serve::ServeMeta) -> serde_json::Value {
    let Some(model) = &meta.current_model else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "model": model,
        "kv_mode": meta.loaded.kv_mode,
        "kv_backend": meta.loaded.kv_backend,
        "max_seq": meta.loaded.max_seq,
    })
}

/// Format one `/v1/models` entry.
///
/// `runtime` is `Some((load facts, effective ctx))` only for the resident
/// model — the one whose KV capacity, hidden/vocab sizes and modalities are
/// known from the daemon's load ack. A non-resident entry is an on-disk
/// discovery candidate with no measured runtime facts, so it gets no
/// `meta`/`architecture`; llama.cpp fills `meta` for the loaded model only,
/// for the same reason. `created` (OpenAI-required) is supplied by the caller
/// from the file mtime and is omitted when unavailable.
fn model_entry(
    id: &str,
    created: Option<u64>,
    runtime: Option<(&LoadedInfo, u64)>,
) -> serde_json::Value {
    let mut entry = serde_json::json!({
        "id": id,
        "object": "model",
        "owned_by": "hipfire",
    });
    if let Some(created) = created {
        entry["created"] = serde_json::json!(created);
    }
    if let Some((loaded, n_ctx)) = runtime {
        let mut meta = serde_json::Map::new();
        if n_ctx > 0 {
            meta.insert("n_ctx".to_owned(), serde_json::json!(n_ctx));
        }
        if loaded.n_embd > 0 {
            meta.insert("n_embd".to_owned(), serde_json::json!(loaded.n_embd));
        }
        if loaded.n_vocab > 0 {
            meta.insert("n_vocab".to_owned(), serde_json::json!(loaded.n_vocab));
        }
        if !meta.is_empty() {
            entry["meta"] = serde_json::Value::Object(meta);
        }
        // `input_modalities` is what makes a harness enable image attachments;
        // `vl` from the load ack is the post-gate truth (a tower skipped by
        // `vision_mode=off` reports false). Gated on a known hidden size so the
        // load window — resident id set, ack not yet in — cannot advertise
        // text-only for a model whose facts are simply not known yet.
        if loaded.n_embd > 0 {
            // One entry per input the serve actually accepts, in wire order:
            // text always; image on a resident vision tower; audio on a
            // resident audio tower; video on the embed route's video leg
            // (arch 18 — its frames ride the vision tower). An arch that
            // refuses a modality by name (arch 17's audio/video refusals)
            // must not advertise it.
            let mut input_modalities = vec!["text"];
            if loaded.vision {
                input_modalities.push("image");
            }
            if loaded.audio {
                input_modalities.push("audio");
            }
            if loaded.video {
                input_modalities.push("video");
            }
            // Same facts /health publishes from `meta.loaded`, so a client
            // discovering models the OpenAI way doesn't also need a bespoke
            // /health probe to learn the effort rungs. `reasoning_contract`
            // is skipped while empty (load window: ack not yet in).
            let mut architecture = serde_json::json!({
                "input_modalities": input_modalities,
                "output_modalities": ["text"],
            });
            if !loaded.reasoning_contract.is_empty() {
                architecture["reasoning_contract"] = serde_json::json!(loaded.reasoning_contract);
                architecture["reasoning_effort_native"] =
                    serde_json::json!(loaded.reasoning_effort_native);
                architecture["reasoning_efforts"] = serde_json::json!(loaded.reasoning_efforts);
            }
            entry["architecture"] = architecture;
        }
        // The daemon's one-line "what is on/off and why" for this load;
        // absent when the ack predates it.
        if loaded.load_summary.is_object() {
            entry["load_summary"] = loaded.load_summary.clone();
        }
    }
    entry
}

/// True when the request carries exactly one `Authorization: Bearer <token>`
/// header whose token matches `key` (constant-time compare; the scheme is
/// case-insensitive per RFC 9110). Missing, malformed, repeated and wrong
/// credentials are indistinguishable to the caller.
fn bearer_authorized(headers: &hyper::HeaderMap, key: &BearerToken) -> bool {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    let (Some(value), None) = (values.next(), values.next()) else {
        return false;
    };
    let Ok(value) = std::str::from_utf8(value.as_bytes()) else {
        return false;
    };
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer") && key.matches(token.trim())
}

/// Uniform 401 for a missing or wrong bearer token. The body never says
/// which, so a probe learns nothing about the key.
fn unauthorized_response() -> Response<BoxBody> {
    Response::builder()
        .status(401)
        .header(header::WWW_AUTHENTICATE, "Bearer")
        .body(boxed_empty())
        .expect("static unauthorized response")
}

/// Unauthenticated `/health` for a keyed server: status and version only.
/// Model, pid, instance token and capabilities need the bearer token. The
/// status code still tracks the engine state so liveness probes keep working.
fn public_health(shared: &ServeShared) -> Response<BoxBody> {
    let state = shared
        .meta
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .engine_state;
    let status = if state == crate::serve::EngineState::Up {
        200
    } else {
        503
    };
    json_response(
        serde_json::json!({
            "status": state.as_str(),
            "version": env!("CARGO_PKG_VERSION"),
        }),
        status,
    )
}

async fn handle_request(
    req: Request<Incoming>,
    shared: Arc<ServeShared>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let path = req
        .uri()
        .path()
        .split('?')
        .next()
        .unwrap_or(req.uri().path())
        .to_owned();
    let method = req.method().clone();
    // Authentication gates every route before routing, body reads, and the
    // UI assets. Only a bare GET `/health` liveness status is public.
    // Preflight requires authentication too; browser pairing is unsupported.
    if let Some(key) = &shared.api_key {
        if !bearer_authorized(req.headers(), key) {
            return if method == Method::GET && path == "/health" {
                public_health(&shared)
            } else {
                unauthorized_response()
            };
        }
    }
    // Opt-in embedded chat UI (`serve.ui` / `--ui`). A GET on a UI path
    // resolves embedded assets; anything else falls through to 404.
    if shared.ui_enabled && method == Method::GET {
        if let Some(resp) = crate::serve::ui::ui_asset(&path) {
            return resp;
        }
    }

    match (method, path.as_str()) {
        (Method::GET, "/health") => {
            // Every fact here comes from `ServeMeta`, which `ensure_model`
            // updates as the load completes. `/health` must NOT take
            // `shared.runtime`: that mutex is held for the whole of a load
            // (prewarm at startup and request-time loads alike), so reading
            // through it would block this endpoint — the one every readiness
            // probe polls — for the duration and report serve as down
            // (`docs/SERVE.md` § "Detached readiness"). Reasoning capability
            // therefore publishes through `meta.loaded` too.
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            // 503 while the daemon is dead or being respawned, so probes and
            // service managers see that requests cannot be served.
            let state = meta.engine_state;
            let body = serde_json::json!({
                "status": state.as_str(),
                "model": meta.current_model,
                "loading_model": meta.loading_model,
                "pid": std::process::id(),
                "token": meta.instance_token,
                // Route capabilities: the resolved config the slot engine was
                // built from (multi-slot, prefix cache, structured output).
                "capabilities": shared.capabilities,
                "native": true,
                // Effective context of the resident model: the `max_seq` it was
                // actually loaded with (KV capacity), not the registry policy
                // and not the trained window. `null` while no model is resident
                // (`n_ctx == 0`), so a client cannot read 0 as "zero context".
                "n_ctx": (meta.n_ctx > 0).then_some(meta.n_ctx),
                // Reasoning capability is live state, published on
                // `meta.loaded` at the load ack (see `ensure_model`) so this
                // endpoint never touches the runtime lock.
                "reasoning_contract": meta.loaded.reasoning_contract,
                "reasoning_effort_native": meta.loaded.reasoning_effort_native,
                "reasoning_efforts": meta.loaded.reasoning_efforts,
                // Baseline render error of the daemon's effort probe, `null`
                // when it rendered (or no model is resident).
                "reasoning_effort_probe_error": meta.loaded.reasoning_effort_probe_error,
                // Multi-slot swap tier: effective budgets, live usage and
                // eviction/restore/failure counters from the last `done`.
                "swap": meta.loaded.swap.clone(),
            });
            let status = if state == crate::serve::EngineState::Up {
                200
            } else {
                503
            };
            json_response(body, status)
        }
        (Method::GET, "/stats") => {
            let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
            let body = serde_json::json!({
                "model": meta.current_model,
                "uptime_sec": meta.started.elapsed().as_secs(),
                "queue_depth": shared.admission.inflight(),
                "requests_served": meta.requests_served,
                "retries_attempted": meta.retries_attempted,
                "retries_succeeded": meta.retries_succeeded,
                "recent_tok_s": meta.recent_tok_s,
                "loaded_config": loaded_config_json(&meta),
                "swap": meta.loaded.swap.clone(),
            });
            json_response(body, 200)
        }
        (Method::GET, "/metrics") => {
            let body = {
                let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
                shared.metrics.render(
                    &meta,
                    shared.admission.inflight(),
                    shared.admission.capacity(),
                )
            };
            let mut resp = Response::builder()
                .status(200)
                .header(
                    header::CONTENT_TYPE,
                    "text/plain; version=0.0.4; charset=utf-8",
                )
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed_full(body.into_bytes()))
                .unwrap();
            resp
        }
        (Method::GET, "/v1/models") => {
            let (local, registry) = {
                let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let local = match list_local_models(&runtime.paths, &runtime.registry) {
                    Ok(m) => m,
                    Err(e) => return openai_error(&e.to_string(), 500),
                };
                (local, runtime.registry.clone())
            };
            // The served facts come from `ServeMeta` (see `ServeMeta::n_ctx`):
            // taking the runtime lock for them would hold this endpoint for the
            // whole of a concurrent model load, since `ensure_model` runs with
            // that lock held.
            let (resident, n_ctx, loaded) = {
                let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
                (meta.current_model.clone(), meta.n_ctx, meta.loaded.clone())
            };
            // Discovery contract: every id advertised here must be able to
            // serve a completion. A sidecar cannot — `qwen3.8-27b-vision.hfq`
            // embeds `"tokenizer": "{}"` and dies with `tokenizer metadata
            // field missing or wrong type: model` AFTER a full 64-layer load,
            // and the DFlash drafts are arch 20/22/23, not serve trunks. An
            // OpenAI-compatible client picks from this list, so a doomed id
            // here is a load failure the operator sees as a server fault.
            // They stay pullable/runnable by tag and still show in `hipfire list`.
            //
            // The resident model is always advertised first, even when it is
            // not a standalone file (a vision tower loaded by path, an
            // unlisted-suffix tier, a symlinked artifact): it IS loaded, so
            // naming it is the one id guaranteed to answer without a model
            // switch. Its validity was established at load time, not by this
            // filter.
            let mut data: Vec<serde_json::Value> = Vec::new();
            if let Some(id) = resident.clone() {
                // OpenAI's `Model` requires `created`; a local artifact's only
                // honest value is its file mtime, looked up from the on-disk
                // listing (the resident is normally a listable file; a sidecar
                // loaded by path still appears in the listing).
                let created = local
                    .iter()
                    .find(|model| {
                        model.registry_tag.as_deref() == Some(id.as_str())
                            || model.name == id
                            || model.path.to_string_lossy() == id
                    })
                    .map(|model| model.created)
                    .filter(|created| *created > 0);
                data.push(model_entry(&id, created, Some((&loaded, n_ctx))));
            }
            data.extend(
                local
                    .into_iter()
                    .filter(|model| {
                        is_standalone_model(
                            &model.path,
                            &model.name,
                            &registry,
                            &grandfather_table(),
                        )
                    })
                    .map(|model| {
                        let created = (model.created > 0).then_some(model.created);
                        (model.registry_tag.unwrap_or(model.name), created)
                    })
                    .filter(|(id, _)| Some(id) != resident.as_ref())
                    .map(|(id, created)| model_entry(&id, created, None)),
            );
            let body = serde_json::json!({
                "object": "list",
                "data": data,
            });
            json_response(body, 200)
        }
        (Method::OPTIONS, _) => {
            let mut resp = Response::builder()
                .status(204)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .header(
                    header::ACCESS_CONTROL_ALLOW_HEADERS,
                    "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta",
                )
                .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, POST, OPTIONS")
                .body(boxed_empty())
                .unwrap();
            resp
        }
        (Method::POST, "/v1/chat/completions") => {
            handle_generation(req, shared, acks, WireApi::ChatCompletions).await
        }
        (Method::POST, "/v1/messages") => {
            handle_generation(req, shared, acks, WireApi::AnthropicMessages).await
        }
        (Method::POST, "/v1/responses") => {
            handle_generation(req, shared, acks, WireApi::OpenAiResponses).await
        }
        // OpenAI-shaped embeddings. The late-interaction extension: each
        // data object carries `n_tokens` (the embedding field is the flat
        // K×dim token matrix, not a single pooled vector) and the request
        // may set `"kind": "query" | "document"` (default document) — the
        // two sides use different markers and truncation limits.
        (Method::POST, "/v1/embeddings") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };
            match handle_embeddings(Arc::clone(&shared), body_val).await {
                Ok(resp) => resp,
                Err(message) => {
                    let status = embeddings_error_status(&message);
                    // /v1/embeddings answers in the OpenAI error shape.
                    RequestFailure::new(message, status)
                        .respond(WireApi::ChatCompletions, &shared.metrics)
                }
            }
        }
        (Method::POST, "/v1/images/generations") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };
            match handle_images_generations(Arc::clone(&shared), body_val).await {
                Ok(resp) => resp,
                Err(message) => {
                    let status = images_error_status(&message);
                    RequestFailure::new(message, status)
                        .respond(WireApi::ChatCompletions, &shared.metrics)
                }
            }
        }
        // OpenAI-shaped reference edit: `multipart/form-data` with one to four
        // `image` file parts plus the text fields of `/v1/images/generations`.
        // The image bytes travel in the request; the server never reads a
        // client-named file.
        (Method::POST, "/v1/images/edits") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let boundary = req
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(multipart_boundary)
                .map(str::to_owned);
            let Some(boundary) = boundary else {
                return openai_error(
                    "/v1/images/edits takes multipart/form-data with a boundary",
                    400,
                );
            };
            let bytes = match read_body_bytes(req.into_body(), max_bytes).await {
                Ok(b) => b,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };
            let body_val = match parse_multipart(&bytes, &boundary)
                .and_then(|(f, i)| edits_form_to_body(f, i))
            {
                Ok(v) => v,
                Err(message) => return openai_error(&message, 400),
            };
            match handle_images_generations(Arc::clone(&shared), body_val).await {
                Ok(resp) => resp,
                Err(message) => {
                    let status = images_error_status(&message);
                    RequestFailure::new(message, status)
                        .respond(WireApi::ChatCompletions, &shared.metrics)
                }
            }
        }
        (Method::POST, "/v1/systemone") => {
            let max_bytes = shared.max_request_bytes;
            if req
                .headers()
                .get(header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|length| length > max_bytes)
            {
                return openai_error(&format!("request body exceeds {max_bytes} bytes"), 413);
            }
            let body_val = match read_json_body(req.into_body(), max_bytes).await {
                Ok(v) => v,
                Err(err) => {
                    let msg = err.to_string();
                    let status = if msg.contains("exceeds") { 413 } else { 400 };
                    return openai_error(&msg, status);
                }
            };

            // Same exclusive admission as non-batch chat: the single-daemon
            // backend serialises decide against chat traffic. The
            // CancellationToken lets a client disconnect while queued abort
            // the wait promptly; once acquired, `guard` moves into
            // `handle_decide`'s spawn_blocking and is held until the daemon
            // call finishes, not merely until this future is dropped.
            let mut cancel_guard = CancelOnDrop::new();
            let cancel = cancel_guard.token();
            let guard = match shared.admission.acquire_async(cancel.clone()).await {
                Ok(g) => g,
                Err(e) => {
                    return admission_error_response(WireApi::ChatCompletions, &shared.metrics, &e)
                }
            };
            let response = crate::serve::decide::handle_decide(shared, body_val, guard).await;
            cancel_guard.disarm();
            response
        }
        _ => openai_error("not found", 404),
    }
}

/// One generation request on any [`WireApi`]: body limit, lowering onto the
/// chat-completions body, then the chat route's admission (same batching
/// eligibility, same model lease), tool gate and worker. A Messages or
/// Responses request is admitted exactly as its lowered chat twin.
async fn handle_generation(
    req: Request<Incoming>,
    shared: Arc<ServeShared>,
    acks: FlushAcks,
    api: WireApi,
) -> Response<BoxBody> {
    let max_bytes = shared.max_request_bytes;
    if req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > max_bytes)
    {
        return api.error_response(&format!("request body exceeds {max_bytes} bytes"), 413);
    }
    let raw_body = match read_json_body(req.into_body(), max_bytes).await {
        Ok(v) => v,
        Err(err) => {
            let msg = err.to_string();
            let status = if msg.contains("exceeds") { 413 } else { 400 };
            return api.error_response(&msg, status);
        }
    };
    let GenerationRequest {
        chat: body_val,
        stream: is_stream,
        encoding,
    } = match api.lower(raw_body, shared.ocr_format) {
        Ok(request) => request,
        Err(message) => return api.error_response(&message, 400),
    };

    let (is_eligible, model_for_lease) = {
        let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
        let tp = runtime.tp;
        let arch = runtime.current_arch.clone();
        let batch_capable = runtime.continuous_batch_capable;
        let batch_history = runtime.continuous_batch_history;
        let multi_slot = runtime.multi_slot_enabled;
        drop(runtime);
        // The admission gate is transport concurrency, not a batch-mode
        // selector. Experimental slots overlap independent requests
        // while remaining separate from ContinuousBatchScheduler.
        let eligible = multi_slot
            || is_batch_eligible_request(
                &body_val,
                tp,
                arch.as_deref(),
                batch_capable,
                batch_history,
            );
        let model = body_val
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        (eligible, model)
    };

    let mut cancel_guard = CancelOnDrop::new();
    let cancel = cancel_guard.token();
    let cancelled = cancel_guard.cancelled();

    let guard = if is_eligible {
        match shared
            .admission
            .acquire_for_async(true, model_for_lease.as_deref(), cancel.clone())
            .await
        {
            Ok(g) => g,
            Err(e) => return admission_error_response(api, &shared.metrics, &e),
        }
    } else {
        match shared.admission.acquire_async(cancel.clone()).await {
            Ok(g) => g,
            Err(e) => return admission_error_response(api, &shared.metrics, &e),
        }
    };

    if let Err(error) = gate_endpoint_tools(&body_val, api.adapter_kind()) {
        return RequestFailure::new(error.to_string(), 400).respond(api, &shared.metrics);
    }

    let response = if is_stream {
        handle_streaming(shared, body_val, encoding, guard, cancelled, acks).await
    } else {
        handle_nonstreaming(shared, body_val, encoding, guard, cancelled, acks).await
    };
    cancel_guard.disarm();
    response
}

// ---------------------------------------------------------------------------
// Images (OpenAI-compatible txt2img)
// ---------------------------------------------------------------------------

/// `/v1/images/generations`: validate the OpenAI-shaped body, forward it as
/// one daemon `img_generate`, and return a single JSON response carrying the
/// PNG as `b64_json`. The daemon (not the gateway) is the authority on
/// sampler/geometry validation — everything it refuses surfaces as a 4xx
/// here with its message.
/// `POST /v1/embeddings` — the qwen3.5 multi-vector encoder (arch 17) and
/// EmbeddingGemma-2 (arch 18).
///
/// Body: `{"input": <string | [string] | {text, image_base64} |
/// {text, images_base64:[…]} | [{content:[{type:"text"|"image_url",…}]}]>,
/// "model"?, "kind"?: "query"|"document", "dims"?}` — see
/// [`crate::serve::embeddings::lower_embeddings_input`] for the accepted
/// input spellings and the shape each refusal names. Reply is OpenAI-shaped
/// `{object:"list", data:[{object:"embedding", index, n_tokens,
/// embedding:[f32…]}], model, usage}` where `embedding` is the flat
/// `n_tokens × dim` token matrix (late interaction), not a pooled vector; an
/// arch-18 single-vector model answers with `n_tokens: 1`.
async fn handle_embeddings(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
) -> Result<Response<BoxBody>, String> {
    let kind = match body
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("document")
    {
        "query" => "query",
        "document" | "doc" => "document",
        other => {
            return Err(format!(
                "kind {other:?} unsupported: expected \"query\" or \"document\""
            ))
        }
    };
    // One input per HTTP request; the multi-slot engine batches concurrent
    // requests behind this endpoint. Every accepted spelling (bare string,
    // one-element list, legacy object, typed content array) lowers to the
    // daemon `embed` verb's `input`; a longer list or a bare content-part
    // array refuses rather than silently dropping or half-reading inputs.
    let input = body
        .get("input")
        .ok_or_else(|| "input is required".to_string())?;
    let input = crate::serve::embeddings::lower_embeddings_input(input)?;

    let request_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Exclusive admission serialized exactly the requests a multi-slot serve
    // exists to batch; admit the sharing (batch-eligible) class whenever the
    // serve advertises multi_slot, the same gate the chat route applies. The
    // request names the resident model (enforced below), so the admission
    // lease keys on it like a chat request's.
    let is_eligible = {
        let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
        runtime.multi_slot_enabled
    };
    let _guard = if is_eligible {
        match shared
            .admission
            .acquire_for_async(true, request_model.as_deref(), CancellationToken::new())
            .await
        {
            Ok(guard) => guard,
            Err(error) => {
                return Ok(admission_error_response(
                    WireApi::ChatCompletions,
                    &shared.metrics,
                    &error,
                ))
            }
        }
    } else {
        match shared
            .admission
            .acquire_async(CancellationToken::new())
            .await
        {
            Ok(guard) => guard,
            Err(error) => {
                return Ok(admission_error_response(
                    WireApi::ChatCompletions,
                    &shared.metrics,
                    &error,
                ))
            }
        }
    };

    // Ensure a model is resident before asking the daemon. Unlike the chat
    // path, an embeddings request carries no model in the general case, so the
    // fallback is the operator's pre-warm model — which survives
    // `clear_resident` on the request policy precisely so an evicted serve can
    // reload. Without this, the first request after the idle timeout failed
    // with "no model loaded" while chat (which always names a model) kept
    // working.
    let (engine, loaded_model) = {
        let mut runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
        if runtime.current_path.is_none() {
            let name = request_model
                .clone()
                .or_else(|| runtime.request_policy.operator_model.clone());
            let Some(name) = name else {
                return Err(
                    "no model resident and this serve has no default model; name one with \
                     \"model\" or start serve with --model"
                        .to_string(),
                );
            };
            // A name the REQUESTER supplied is a request load: honor the
            // request policy (allow_request_paths / allow_pull). Only the
            // operator's own configured default is an Operator load.
            let origin = if request_model.as_deref() == Some(name.as_str()) {
                crate::serve::ModelOrigin::Request
            } else {
                crate::serve::ModelOrigin::Operator
            };
            runtime
                .ensure_model(&name, &shared.meta, origin)
                .map_err(|e| format!("loading {name}: {e}"))?;
        }
        (runtime.engine.clone(), runtime.current_path.clone())
    };
    // The name a client knows the resident model by: the spelling it was
    // served under (`meta.current_model` mirrors `served_name` from the load
    // path), never the canonicalized on-disk path — path mounts leak into
    // the wire name otherwise (`/pplx-embed-v2-late` instead of the name the
    // client asked for).
    let served_name = {
        let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
        meta.current_model.clone()
    };

    // A request that names a DIFFERENT model than the resident one would be
    // silently answered by the resident checkpoint — the response looks like
    // a 200 with embeddings from a model the caller didn't ask for. Refuse
    // instead of swapping (embeddings are cheap and bursty; an implicit
    // swap per request would thrash the GPU for every other caller).
    if let Some(name) = request_model.as_deref() {
        // Resolve the name to its on-disk path when it resolves at all; a
        // registry tag, an alias and the literal path of the resident model
        // must all compare equal. An unresolvable name is a mismatch too —
        // refusing it here is cleaner than letting the daemon answer with
        // the resident checkpoint.
        let named_path = {
            let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            crate::resolve_model_path(&runtime.paths, &runtime.registry, name)
                .ok()
                .flatten()
        };
        let matches = match (&named_path, &loaded_model) {
            (Some(named), Some(resident)) => named == resident,
            // No resident model (shouldn't happen — the ensure above either
            // loaded one or errored) or an unresolvable name: refuse.
            _ => false,
        };
        if !matches {
            return Err(format!(
                "requested model {name:?} is not the resident model ({}); \
                 this serve encodes with the resident model only",
                served_name
                    .clone()
                    .or_else(|| loaded_model.as_ref().map(|p| p.display().to_string()))
                    .unwrap_or_else(|| "none".to_string())
            ));
        }
    }
    let model_echo = request_model
        .or(served_name)
        .or_else(|| loaded_model.as_ref().map(|p| p.display().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    let id = request_id();
    // Optional Matryoshka width (embeddinggemma2): forwarded verbatim; arches
    // without MRL support refuse it in the embed body.
    let mut request = serde_json::json!({
        "type": "embed",
        "id": id,
        "kind": kind,
        "input": input,
    });
    if let Some(d) = body.get("dims") {
        request["dims"] = d.clone();
    }

    let (tx, rx) = tokio::sync::oneshot::channel::<Result<serde_json::Value, String>>();
    tokio::task::spawn_blocking(move || {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.embed(&request)));
        let result = match outcome {
            Ok(Ok(done)) => Ok(done),
            Ok(Err(error)) => Err(error.to_string()),
            Err(payload) => Err(format!(
                "embedding worker panicked: {}",
                payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string())
            )),
        };
        let _ = tx.send(result);
    });
    let done = rx
        .await
        .map_err(|_| "embedding worker disconnected".to_string())?
        .map_err(|e| e)?;

    let n_tokens = done.get("n_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let dim = done.get("dim").and_then(|v| v.as_u64()).unwrap_or(0);
    let embeddings = done
        .get("embeddings")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]));
    let payload = serde_json::json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "n_tokens": n_tokens,
            "embedding": embeddings,
        }],
        "model": model_echo,
        "usage": { "prompt_tokens": done.get("n_input_tokens").cloned().unwrap_or(serde_json::json!(0)),
                   "total_tokens": done.get("n_input_tokens").cloned().unwrap_or(serde_json::json!(0)) },
        // dim is reported top-level for clients that expect one vector.
        "dim": dim,
    });
    Ok(json_response(payload, 200))
}

async fn handle_images_generations(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
) -> Result<Response<BoxBody>, String> {
    let started = Instant::now();
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "prompt is required and must be a non-empty string".to_string())?;
    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1);
    if n != 1 {
        return Err(format!("n={n} unsupported: one image per request"));
    }
    if let Some(format) = body.get("response_format").and_then(|v| v.as_str()) {
        if format != "b64_json" {
            return Err(format!(
                "response_format {format:?} unsupported: b64_json only"
            ));
        }
    }
    // Size: OpenAI `size` string ("WxH") wins, else explicit width/height.
    // Neither is required: a reference-image edit request (`images[]`, arch
    // 44) that omits both lets the daemon default to the reference image's
    // own size, so no width/height key is inserted into the forwarded
    // request in that case.
    let (width, height): (Option<u64>, Option<u64>) =
        match body.get("size").and_then(|v| v.as_str()) {
            Some(size) => {
                let parts: Vec<&str> = size.split('x').collect();
                if parts.len() != 2 {
                    return Err(format!(
                        "size {size:?} must be WIDTHxHEIGHT, e.g. \"1024x1024\""
                    ));
                }
                let w: u64 = parts[0]
                    .parse()
                    .map_err(|_| format!("size width {:?} is not a number", parts[0]))?;
                let h: u64 = parts[1]
                    .parse()
                    .map_err(|_| format!("size height {:?} is not a number", parts[1]))?;
                (Some(w), Some(h))
            }
            None => (
                body.get("width").and_then(|v| v.as_u64()),
                body.get("height").and_then(|v| v.as_u64()),
            ),
        };
    let steps = body.get("steps").and_then(|v| v.as_u64());
    let seed = body.get("seed").and_then(|v| v.as_u64());
    let sampler = body.get("sampler").and_then(|v| v.as_str());
    let negative_prompt = body.get("negative_prompt").and_then(|v| v.as_str());
    let backend = body.get("backend").and_then(|v| v.as_str());
    if let Some(backend) = backend {
        if !matches!(backend, "cpu" | "gpu") {
            return Err(format!(
                "backend {backend:?} unsupported: expected \"cpu\" or \"gpu\""
            ));
        }
    }
    let request_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Serialize against chat traffic and cap queue depth the same way the
    // chat path does; the daemon processes messages sequentially. The wait
    // is async so a queued image request does not park a runtime worker.
    // Dropping this future (client gone) withdraws it from the queue.
    let _guard = match shared
        .admission
        .acquire_async(CancellationToken::new())
        .await
    {
        Ok(guard) => guard,
        Err(error) => {
            return Ok(admission_error_response(
                WireApi::ChatCompletions,
                &shared.metrics,
                &error,
            ))
        }
    };

    let (engine, loaded_model) = {
        let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
        (runtime.engine.clone(), runtime.current_path.clone())
    };
    // Same reporting rule as /v1/embeddings: echo the request spelling, then
    // the served name, not the canonicalized resident path.
    let served_name = {
        let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
        meta.current_model.clone()
    };
    let model_echo = request_model
        .or(served_name)
        .or_else(|| loaded_model.as_ref().map(|p| p.display().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    let id = request_id();
    let mut request = serde_json::json!({
        "type": "img_generate",
        "id": id,
        "prompt": prompt,
    });
    if let Some(width) = width {
        request["width"] = serde_json::json!(width);
    }
    if let Some(height) = height {
        request["height"] = serde_json::json!(height);
    }
    // Reference images arrive only through `/v1/images/edits` (multipart file
    // parts, the OpenAI shape), which puts their bytes under the internal
    // `_reference_images` key. A client that sends `images` here is pointed
    // at the right route; a path string is never forwarded to the daemon.
    if body.get("images").is_some() {
        return Err(
            "images is not a field of /v1/images/generations; send the reference image \
             files as multipart `image` parts to /v1/images/edits"
                .to_string(),
        );
    }
    if let Some(refs) = body.get("_reference_images") {
        request["images"] = refs.clone();
    }
    if let Some(steps) = steps {
        request["steps"] = serde_json::json!(steps);
    }
    if let Some(seed) = seed {
        request["seed"] = serde_json::json!(seed);
    }
    // Forward the fail-closed surface so the daemon's single validation
    // authority sees sampler/negative_prompt exactly as the client sent them.
    if let Some(sampler) = sampler {
        request["sampler"] = serde_json::json!(sampler);
    }
    if let Some(negative_prompt) = negative_prompt {
        request["negative_prompt"] = serde_json::json!(negative_prompt);
    }
    if let Some(backend) = backend {
        request["backend"] = serde_json::json!(backend);
    }

    let (tx, rx) = tokio::sync::oneshot::channel::<Result<serde_json::Value, String>>();
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.img_generate(&request, |_event| Ok(()))
        }));
        let result = match outcome {
            Ok(Ok(done)) => Ok(done),
            Ok(Err(error)) => Err(error.to_string()),
            Err(payload) => Err(format!(
                "image generation worker panicked: {}",
                payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string())
            )),
        };
        let _ = tx.send(result);
    });
    let done = rx
        .await
        .map_err(|_| "image generation worker disconnected".to_string())?
        .map_err(|e| e)?;

    let b64 = done
        .get("png_b64")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "daemon img_done missing png_b64".to_string())?
        .to_owned();
    let out_w = done
        .get("width")
        .and_then(|v| v.as_u64())
        .or(width)
        .unwrap_or(1024);
    let out_h = done
        .get("height")
        .and_then(|v| v.as_u64())
        .or(height)
        .unwrap_or(1024);
    {
        let mut meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
        meta.requests_served += 1;
    }
    shared
        .metrics
        .observe_request(started.elapsed().as_secs_f64() * 1000.0);
    let response = serde_json::json!({
        "created": unix_timestamp(),
        "model": model_echo,
        "data": [
            {
                "b64_json": b64,
                "size": format!("{out_w}x{out_h}"),
            }
        ],
        "hipfire": {
            "seed": done.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
            "steps": done.get("steps").and_then(|v| v.as_u64()).unwrap_or(0),
            "width": out_w,
            "height": out_h,
            "ms": done.get("ms").and_then(|v| v.as_u64()).unwrap_or(0),
        },
    });
    Ok(json_response(response, 200))
}

async fn read_json_body(body: Incoming, max_bytes: u64) -> Result<serde_json::Value> {
    let bytes = read_body_bytes(body, max_bytes).await?;
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

async fn read_body_bytes(body: Incoming, max_bytes: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut stream = body;
    while let Some(frame) = stream.frame().await {
        let frame = frame.context("failed to read request body")?;
        if let Some(data) = frame.data_ref() {
            if bytes.len() as u64 + data.len() as u64 > max_bytes {
                bail!("request body exceeds {max_bytes} bytes");
            }
            bytes.extend_from_slice(data);
        }
        if bytes.len() as u64 > max_bytes {
            bail!("request body exceeds {max_bytes} bytes");
        }
    }
    if bytes.len() as u64 > max_bytes {
        bail!("request body exceeds {max_bytes} bytes");
    }
    Ok(bytes)
}

/// The `boundary` parameter of a `multipart/form-data` content type.
fn multipart_boundary(content_type: &str) -> Option<&str> {
    let mut parts = content_type.split(';');
    if !parts
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    parts
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))
        .map(|b| b.trim_matches('"'))
        .filter(|b| !b.is_empty())
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Minimal `multipart/form-data` reader for `/v1/images/edits`: the text
/// fields as `(name, value)` and the raw bytes of every `image` file part, in
/// order. Boundaries and part headers follow RFC 7578; nested multipart and
/// transfer encodings are not accepted, which is all the OpenAI clients send.
fn parse_multipart(
    body: &[u8],
    boundary: &str,
) -> Result<(Vec<(String, String)>, Vec<Vec<u8>>), String> {
    let delim = format!("--{boundary}");
    let end_marker = format!("\r\n{delim}");
    let mut fields = Vec::new();
    let mut images = Vec::new();
    let start = find_bytes(body, delim.as_bytes()).ok_or("multipart body has no boundary")?;
    let mut rest = &body[start + delim.len()..];
    loop {
        if rest.starts_with(b"--") {
            break;
        }
        let part_start = rest
            .strip_prefix(b"\r\n")
            .ok_or("malformed multipart part delimiter")?;
        let hdr_end = find_bytes(part_start, b"\r\n\r\n")
            .ok_or("multipart part without a header terminator")?;
        let headers = std::str::from_utf8(&part_start[..hdr_end])
            .map_err(|_| "multipart part headers are not UTF-8")?;
        let content = &part_start[hdr_end + 4..];
        let body_end =
            find_bytes(content, end_marker.as_bytes()).ok_or("unterminated multipart part")?;
        let part = &content[..body_end];
        let mut name = None;
        let mut is_file = false;
        for line in headers.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if key.trim().eq_ignore_ascii_case("content-disposition") {
                for param in value.split(';').map(str::trim) {
                    if let Some(n) = param.strip_prefix("name=") {
                        name = Some(n.trim_matches('"').to_string());
                    }
                    if param.starts_with("filename=") {
                        is_file = true;
                    }
                }
            }
        }
        let name = name.ok_or("multipart part without a name")?;
        if name == "image" || name == "image[]" {
            images.push(part.to_vec());
        } else if is_file {
            return Err(format!(
                "unexpected file part {name:?}: only `image` file parts are accepted"
            ));
        } else {
            fields.push((name, String::from_utf8_lossy(part).into_owned()));
        }
        rest = &content[body_end + end_marker.len()..];
    }
    Ok((fields, images))
}

/// `/v1/images/edits` body → the JSON the generations handler consumes: the
/// text fields (numeric ones parsed), plus the image parts as base64 under
/// the internal `_reference_images` key. At most four images.
fn edits_form_to_body(
    fields: Vec<(String, String)>,
    images: Vec<Vec<u8>>,
) -> Result<serde_json::Value, String> {
    use base64::Engine as _;
    if images.is_empty() {
        return Err("an `image` file part is required".to_string());
    }
    if images.len() > 4 {
        return Err("at most 4 reference images".to_string());
    }
    let mut body = serde_json::Map::new();
    for (name, value) in fields {
        let v = match name.as_str() {
            "n" | "seed" | "steps" | "width" | "height" => serde_json::Value::from(
                value
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}"))?,
            ),
            _ => serde_json::Value::from(value),
        };
        body.insert(name, v);
    }
    let refs: Vec<String> = images
        .iter()
        .map(|b| base64::engine::general_purpose::STANDARD.encode(b))
        .collect();
    body.insert("_reference_images".into(), serde_json::json!(refs));
    Ok(serde_json::Value::Object(body))
}

/// Status for `/v1/embeddings` failures. The daemon's overload refusal
/// ("too many concurrent slot requests") is transient capacity → 429 so a
/// client retries instead of treating it as a request bug; engine and
/// worker failures are internal → 500; everything else is the caller's
/// request → 400. Mirrors [`images_error_status`].
fn embeddings_error_status(message: &str) -> u16 {
    let lower = message.to_ascii_lowercase();
    if lower.contains("too many concurrent") || lower.contains("overload") {
        return 429;
    }
    if lower.contains("engine thread exited")
        || lower.contains("engine is shut down")
        || lower.contains("worker panicked")
        || lower.contains("worker disconnected")
        || lower.contains("daemon closed")
    {
        return 500;
    }
    400
}

fn images_error_status(message: &str) -> u16 {
    let lower = message.to_ascii_lowercase();
    if [
        "refused",
        "unsupported",
        "invalid",
        "required",
        "must be",
        "at most",
        "not base64",
        "reference image",
        "not a field",
        "multipart",
        "no model loaded",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
    {
        400
    } else {
        500
    }
}

// ---------------------------------------------------------------------------
// Streaming / Non-streaming handlers
// ---------------------------------------------------------------------------

/// Longest an SSE client goes without a byte. A stream commits (200 + role
/// chunk) when the worker sends its first frame; if generation is still
/// silent this long after admission (cold load, long prefill), it commits
/// anyway, and a committed stream sends [`SSE_KEEPALIVE`] after this much
/// silence. 15 s sits under common idle-read limits (nginx
/// `proxy_read_timeout` 60 s, undici `bodyTimeout` 300 s).
const SSE_SILENCE_LIMIT: Duration = Duration::from_secs(15);

async fn handle_streaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    encoding: ResponseEncoding,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let api = encoding.api();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ResponseChunk>(32);
    let commit = Arc::new(Mutex::new(StreamCommit::Pending));
    let id = request_id();
    let created = unix_timestamp();
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();
    let ocr = encoding
        .ocr()
        .map(|format| RefCell::new(StreamOcr::new(format)));
    let mut encoder = encoding.stream(&id, created, &model);
    let mut first = VecDeque::from([ResponseChunk::plain(encoder.preamble())]);
    let sink = SseSink {
        tx,
        commit: Arc::clone(&commit),
        committed: Cell::new(false),
        terminal_sent: Cell::new(false),
        id,
        created,
        encoder: RefCell::new(encoder),
        ocr,
        stall_timeout: shared.stream_stall_timeout,
    };

    let body_cancelled = Arc::clone(&cancelled);
    let worker_shared = Arc::clone(&shared);
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            complete_request_cancellable(
                &worker_shared,
                &body,
                guard,
                Some((sink.id.clone(), sink.created)),
                &cancelled,
                |event| sink.forward_event(event),
                |completion| sink.deliver_terminal(completion),
            )
        }));
        let result = outcome.unwrap_or_else(|payload| {
            Err(anyhow!(
                "generation worker panicked: {}",
                panic_message(payload.as_ref())
            ))
        });
        sink.finish(result, &worker_shared.metrics);
    });

    // Validation, model load and the daemon's own request checks all run in
    // the worker before its first frame, so a failure there still gets its
    // real status as an ordinary JSON error instead of a 200 stream.
    tokio::select! {
        chunk = rx.recv() => match chunk {
            // The worker commits before it sends anything.
            Some(chunk) => first.push_back(chunk),
            None => {
                return match commit_stream(&commit) {
                    Err(failure) => failure.respond(api, &shared.metrics),
                    Ok(()) => RequestFailure::new("generation ended without a response", 500)
                        .respond(api, &shared.metrics),
                };
            }
        },
        _ = tokio::time::sleep(SSE_SILENCE_LIMIT) => {
            if let Err(failure) = commit_stream(&commit) {
                return failure.respond(api, &shared.metrics);
            }
        }
    }

    let body = ChannelBody::new(first, rx, acks, body_cancelled, SSE_SILENCE_LIMIT);
    Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(boxed(body))
        .unwrap()
}

async fn handle_nonstreaming(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    encoding: ResponseEncoding,
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let api = encoding.api();
    // Staged terminal channel: Ok((bytes, ack_tx)) for success, Err(failure) for preterminal.
    let (staged_tx, staged_rx) = tokio::sync::oneshot::channel::<
        Result<(Vec<u8>, std::sync::mpsc::Sender<Result<(), ()>>), RequestFailure>,
    >();
    let staged_tx = Arc::new(Mutex::new(Some(staged_tx)));

    let shared_clone = Arc::clone(&shared);
    let body_for_worker = body;
    let staged_tx_clone = Arc::clone(&staged_tx);
    let staged_tx_for_worker = Arc::clone(&staged_tx);
    let stall_timeout = shared.stream_stall_timeout;
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let staged_for_terminal = Arc::clone(&staged_tx_for_worker);
            let terminal = |completion: &Completion| {
                let bytes =
                    serde_json::to_vec(&encoding.completion_body(completion)).map_err(|err| {
                        hipfire_client::ClientError::Protocol(format!(
                            "completion json serialize failed: {err}"
                        ))
                    })?;
                if bytes.is_empty() {
                    return Err(hipfire_client::ClientError::Protocol(
                        "nonstream terminal body must be non-empty".into(),
                    ));
                }
                let (ack_tx, ack_rx) = std::sync::mpsc::channel();
                // Stage exactly one JSON frame and block for ack (RAII guard cancels while awaiting).
                {
                    let mut lock = staged_for_terminal
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    if let Some(tx) = lock.take() {
                        let _ = tx.send(Ok((bytes, ack_tx)));
                    } else {
                        return Err(hipfire_client::ClientError::Cancelled);
                    }
                }
                match wait_flush_ack(&ack_rx, stall_timeout) {
                    Ok(()) => Ok(()),
                    Err(()) => Err(hipfire_client::ClientError::Cancelled),
                }
            };
            complete_request_cancellable(
                &shared_clone,
                &body_for_worker,
                guard,
                None,
                &cancelled,
                |_event| Ok(()),
                terminal,
            )
        }));
        // If terminal was never staged, propagate preterminal error via staged channel.
        let mut lock = staged_tx_clone
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(tx) = lock.take() {
            let failure = match outcome {
                // Ok but terminal never called — report as error.
                Ok(Ok(_completion)) => {
                    RequestFailure::new("generation completed without a response body", 500)
                }
                Ok(Err(error)) => RequestFailure::from_error(&error),
                Err(payload) => RequestFailure::new(
                    format!(
                        "generation worker panicked: {}",
                        panic_message(payload.as_ref())
                    ),
                    500,
                ),
            };
            let _ = tx.send(Err(failure));
        }
    });

    // Await staged terminal with handler's CancellationToken guard alive.
    // Hyper dropping this future cancels the token and the AtomicBool watcher above.
    match staged_rx.await {
        Ok(Ok((bytes, ack_tx))) => {
            // Success: exactly one JSON frame with AckBody.
            let body = AckBody::new(bytes, ack_tx, acks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .body(boxed(body))
                .unwrap();
            resp
        }
        Ok(Err(failure)) => failure.respond(api, &shared.metrics),
        Err(_) => {
            RequestFailure::new("generation worker disconnected", 500).respond(api, &shared.metrics)
        }
    }
}

// ---------------------------------------------------------------------------
// Preserved business helpers (same shapes, adapted to tokio mpsc where needed)
// ---------------------------------------------------------------------------

/// Retry-After for a typed `transient` daemon failure (HTTP 503).
const TRANSIENT_RETRY_AFTER_SECS: u64 = 1;

/// A failed completion. Before a stream commits it is a JSON error in the
/// request's API shape with `status`; after, the stream's error frame.
#[derive(Debug)]
pub(crate) struct RequestFailure {
    message: String,
    status: u16,
}

impl RequestFailure {
    fn new(message: impl Into<String>, status: u16) -> Self {
        Self {
            message: message.into(),
            status,
        }
    }

    fn from_error(error: &anyhow::Error) -> Self {
        Self::new(error.to_string(), request_error_status(error))
    }

    /// The JSON error response ending this request; counts it as failed.
    fn respond(&self, api: WireApi, metrics: &Metrics) -> Response<BoxBody> {
        metrics.record_failure();
        let mut resp = api.error_response(&self.message, self.status);
        if self.status == 503 {
            resp.headers_mut().insert(
                header::RETRY_AFTER,
                header::HeaderValue::from(TRANSIENT_RETRY_AFTER_SECS),
            );
        }
        resp
    }

    /// The error event and `[DONE]` that end a chat stream failing after commit.
    fn sse_event(&self) -> Vec<u8> {
        let mut bytes = sse_data(&openai_error_body(&self.message, self.status));
        bytes.extend_from_slice(b"data: [DONE]\n\n");
        bytes
    }
}

/// HTTP status for a failed completion, by error TYPE, never by message. A
/// typed daemon error maps on its wire `class`: `validation`,
/// `context_length` and `unsupported` are the client's to fix (400),
/// `transient` and the multi-slot engine's `overload` (capacity) are worth a
/// retry (503), and every other class is a server fault (500). Gateway-side
/// errors are 400 when they carry [`InvalidRequest`](crate::serve::InvalidRequest),
/// 404 for [`ModelNotFound`](crate::serve::ModelNotFound), else 500.
pub(crate) fn request_error_status(error: &anyhow::Error) -> u16 {
    use hipfire_client::error_class;
    let typed = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<hipfire_client::ClientError>())
        .and_then(hipfire_client::ClientError::typed_daemon);
    if let Some(typed) = typed {
        return match typed.class.as_str() {
            error_class::VALIDATION | error_class::CONTEXT_LENGTH | error_class::UNSUPPORTED => 400,
            error_class::TRANSIENT | error_class::OVERLOAD => 503,
            _ => 500,
        };
    }
    if error
        .chain()
        .any(|cause| cause.is::<crate::serve::InvalidRequest>())
    {
        400
    } else if error
        .chain()
        .any(|cause| cause.is::<crate::serve::ModelNotFound>())
    {
        404
    } else {
        500
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

pub(crate) fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "chatcmpl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn sse_data(value: &serde_json::Value) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut bytes, value).expect("serializing a JSON value cannot fail");
    bytes.extend_from_slice(b"\n\n");
    bytes
}

/// One named SSE event (`event:` and `data:` lines), the framing of the
/// Anthropic Messages and OpenAI Responses streams.
pub(crate) fn sse_event(name: &str, value: &serde_json::Value) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"event: ");
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(b"\ndata: ");
    serde_json::to_writer(&mut bytes, value).expect("serializing a JSON value cannot fail");
    bytes.extend_from_slice(b"\n\n");
    bytes
}

/// The SSE comment frame announcing prefill progress (`: prefill 512/2048`).
/// Conforming SSE parsers ignore it; `hipfire_client::read_openai_sse` turns
/// it into `OpenAiSseEvent::PrefillProgress`.
pub(crate) fn sse_prefill_comment(done: u64, total: u64) -> Vec<u8> {
    format!(": prefill {done}/{total}\n\n").into_bytes()
}

/// The per-process part of a [`request_id`] (`chatcmpl-PID-N` → `PID-N`),
/// from which the Anthropic `msg_…` and Responses `resp_…` ids derive.
pub(crate) fn id_suffix(request_id: &str) -> &str {
    request_id.strip_prefix("chatcmpl-").unwrap_or(request_id)
}

/// Commit point of one SSE response, shared by the handler and its worker.
enum StreamCommit {
    /// Nothing sent yet: a failure can still be an ordinary JSON error.
    Pending,
    /// The 200 and the role chunk are (or are about to be) on the wire.
    Committed,
    /// The worker failed before its first frame; the handler answers with this.
    Rejected(RequestFailure),
}

/// Commit the response to 200 unless the worker already rejected it, in
/// which case its failure is handed back instead.
fn commit_stream(commit: &Mutex<StreamCommit>) -> Result<(), RequestFailure> {
    let mut state = commit.lock().unwrap_or_else(|error| error.into_inner());
    match std::mem::replace(&mut *state, StreamCommit::Committed) {
        StreamCommit::Rejected(failure) => Err(failure),
        StreamCommit::Pending | StreamCommit::Committed => Ok(()),
    }
}

/// Worker end of one SSE response, in the request's API framing.
struct SseSink {
    tx: tokio::sync::mpsc::Sender<ResponseChunk>,
    commit: Arc<Mutex<StreamCommit>>,
    /// Worker-side copy of "committed", so each token skips the lock.
    committed: Cell<bool>,
    /// The acknowledged terminal (`[DONE]`) frame was handed to the body.
    terminal_sent: Cell<bool>,
    id: String,
    created: u64,
    /// Frames the stream for the client's API; only the worker touches it.
    encoder: RefCell<StreamEncoder>,
    /// Compact-block decode state (`ocr_format` / `serve.ocr_format`). Held
    /// here, not in the encoder, so chat, Anthropic and Responses streams all
    /// see decoded text deltas from one state machine.
    ocr: Option<RefCell<StreamOcr>>,
    /// Multi-slot route only (`serve.stream_stall_timeout_ms`): how long a
    /// consumer may leave the response channel full, or the terminal frame
    /// unflushed, before the request is aborted and its permit released.
    /// `None` (standard route) waits for as long as the connection lives.
    stall_timeout: Option<Duration>,
}

/// How often a stalled send re-checks the channel.
const STALL_RETRY_INTERVAL: Duration = Duration::from_millis(10);

/// Wait for a frame's flush ack. With a stall bound, a client that never
/// reads the body is given up on after `stall_timeout` instead of pinning the
/// worker and its admission permit.
fn wait_flush_ack(
    ack_rx: &std::sync::mpsc::Receiver<Result<(), ()>>,
    stall_timeout: Option<Duration>,
) -> Result<(), ()> {
    let outcome = match stall_timeout {
        None => ack_rx.recv().map_err(|_| ()),
        Some(limit) => ack_rx.recv_timeout(limit).map_err(|error| {
            if error == std::sync::mpsc::RecvTimeoutError::Timeout {
                eprintln!(
                    "[hipfire] response not flushed within serve.stream_stall_timeout_ms; aborting"
                );
            }
        }),
    };
    outcome.and_then(|ack| ack)
}

impl SseSink {
    /// Send one frame, committing the response first if this is the first.
    /// A dropped receiver, or (multi-slot) a channel left full for the stall
    /// timeout, maps to `Cancelled`.
    fn send(&self, chunk: ResponseChunk) -> Result<(), hipfire_client::ClientError> {
        if !self.committed.get() {
            // Only this worker ever rejects, and only after its last send, so
            // committing cannot fail here.
            let _ = commit_stream(&self.commit);
            self.committed.set(true);
        }
        let Some(limit) = self.stall_timeout else {
            return self
                .tx
                .blocking_send(chunk)
                .map_err(|_| hipfire_client::ClientError::Cancelled);
        };
        let deadline = Instant::now() + limit;
        let mut chunk = chunk;
        loop {
            match self.tx.try_send(chunk) {
                Ok(()) => return Ok(()),
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    return Err(hipfire_client::ClientError::Cancelled);
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(returned)) => {
                    let now = Instant::now();
                    if now >= deadline {
                        eprintln!(
                            "[hipfire] {}: stream consumer stalled past serve.stream_stall_timeout_ms; aborting",
                            self.id
                        );
                        return Err(hipfire_client::ClientError::Cancelled);
                    }
                    std::thread::sleep(STALL_RETRY_INTERVAL.min(deadline - now));
                    chunk = returned;
                }
            }
        }
    }

    /// Forward one logical generate event. Delta-bearing events serialize to
    /// plain (no-ack) SSE bytes. No-delta mid-stream events are silent —
    /// terminal ack handles pure-tool delivery.
    fn forward_event(&self, event: &serde_json::Value) -> Result<(), hipfire_client::ClientError> {
        if event.get("type").and_then(serde_json::Value::as_str) == Some("prefill_progress") {
            return self.forward_progress(event);
        }
        // Compact-block decoding happens before the API's encoder frames a
        // delta, so chat/Anthropic/Responses stream decoded text alike. A
        // delta that completes no block sends nothing, and each channel keeps
        // its own decoder: reasoning text is only ever emitted as reasoning.
        let pending = self.ocr.as_ref().and_then(|state| {
            let channel = match event.get("type").and_then(serde_json::Value::as_str) {
                Some("token") => StreamChannelKind::Content,
                Some("reasoning") => StreamChannelKind::Reasoning,
                _ => return None,
            };
            let text = event.get("text").and_then(serde_json::Value::as_str)?;
            Some((state, channel, text))
        });
        let event = match pending {
            Some((state, channel, text)) => match state.borrow_mut().push(channel, text) {
                Some(decoded) => {
                    let mut rewritten = event.clone();
                    rewritten["text"] = serde_json::Value::String(decoded);
                    std::borrow::Cow::Owned(rewritten)
                }
                None => return Ok(()),
            },
            None => std::borrow::Cow::Borrowed(event),
        };
        let Some(bytes) = self.encoder.borrow_mut().event(&event) else {
            return Ok(());
        };
        self.send(ResponseChunk::plain(bytes))
    }

    /// Forward daemon prefill progress as a `: prefill <done>/<total>` SSE
    /// comment. Progress never commits the response: until the 200 is on the
    /// wire (a pre-commit failure must still be a JSON error) it is dropped,
    /// and nothing follows the terminal frame. Every frame it does send
    /// restarts the keepalive timer like any other.
    fn forward_progress(
        &self,
        event: &serde_json::Value,
    ) -> Result<(), hipfire_client::ClientError> {
        let (Some(done), Some(total)) = (
            event.get("done").and_then(serde_json::Value::as_u64),
            event.get("total").and_then(serde_json::Value::as_u64),
        ) else {
            return Ok(());
        };
        if self.terminal_sent.get() {
            return Ok(());
        }
        if !self.committed.get() {
            // The handler also commits on its own silence deadline, so the
            // shared state, not only this worker's copy, says whether the
            // 200 is out.
            let state = self.commit.lock().unwrap_or_else(|e| e.into_inner());
            if !matches!(*state, StreamCommit::Committed) {
                return Ok(());
            }
            self.committed.set(true);
        }
        self.send(ResponseChunk::plain(sse_prefill_comment(done, total)))
    }

    /// Serialize the terminal (released tool calls, finish, usage; chat adds
    /// `[DONE]`) into one acknowledged chunk; wait until the socket flushed it.
    fn deliver_terminal(&self, completion: &Completion) -> Result<(), hipfire_client::ClientError> {
        if let Some(state) = &self.ocr {
            let [content, reasoning] = state.borrow_mut().finish();
            for (kind, trailing) in [("token", content), ("reasoning", reasoning)] {
                if trailing.is_empty() {
                    continue;
                }
                let event = serde_json::json!({ "type": kind, "text": trailing });
                if let Some(bytes) = self.encoder.borrow_mut().event(&event) {
                    self.send(ResponseChunk::plain(bytes))?;
                }
            }
        }
        let bytes = self.encoder.borrow_mut().terminal(completion);
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        self.terminal_sent.set(true);
        self.send(ResponseChunk::last(bytes, Some(ack_tx)))?;
        wait_flush_ack(&ack_rx, self.stall_timeout)
            .map_err(|()| hipfire_client::ClientError::Cancelled)
    }

    /// Close the body after `complete_request_cancellable`.
    /// - Success or cancel: nothing more to send.
    /// - Failure before the first frame: hand it to the handler, which answers
    ///   with its status instead of a 200.
    /// - Failure after commit: the API's error frame (chat: an SSE error
    ///   event, then `[DONE]`).
    /// - Failure after the acknowledged terminal (e.g. the daemon commit):
    ///   nothing may follow it, so the body fails and the reader sees it torn.
    fn finish(self, result: Result<Completion>, metrics: &Metrics) {
        let Err(error) = result else {
            return;
        };
        let cancelled = error
            .downcast_ref::<hipfire_client::ClientError>()
            .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
        if cancelled {
            return;
        }
        eprintln!(
            "[hipfire] {}: streaming completion failed: {error:#}",
            self.id
        );
        if self.terminal_sent.get() {
            metrics.record_failure();
            let _ = self.tx.try_send(ResponseChunk::fail());
            return;
        }
        let failure = RequestFailure::from_error(&error);
        if !self.committed.get() {
            let mut state = self.commit.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(*state, StreamCommit::Pending) {
                *state = StreamCommit::Rejected(failure);
                return;
            }
            // The handler committed on its deadline while the worker was silent.
        }
        metrics.record_failure();
        let bytes = self.encoder.borrow_mut().failure(&failure);
        let _ = self.tx.blocking_send(ResponseChunk::last(bytes, None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::Body;
    use std::future::poll_fn;
    use std::task::Waker;
    use std::time::Duration;

    /// `ocr_format` selects a rendering of the compact-block form; it is
    /// stripped from the body so the daemon never sees it.
    #[test]
    fn ocr_format_is_stripped_and_selects_the_decoder() {
        let body = serde_json::json!({
            "model": "surya-ocr-2.5:q8",
            "messages": [{"role": "user", "content": "hi"}],
            "ocr_format": "html",
        });
        let request = WireApi::ChatCompletions
            .lower(body, None)
            .expect("valid request");
        assert!(
            request.chat.get("ocr_format").is_none(),
            "ocr_format is answered here, never forwarded"
        );
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, Some(OcrFormat::Html)),
            _ => panic!("expected the chat encoding"),
        }
    }

    #[test]
    fn ocr_format_accepts_streaming_and_raw_and_rejects_unknown_values() {
        let streaming = serde_json::json!({
            "model": "m",
            "stream": true,
            "ocr_format": "html",
            "messages": [],
        });
        let request = WireApi::ChatCompletions
            .lower(streaming, None)
            .expect("streams decode incrementally");
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, Some(OcrFormat::Html)),
            _ => panic!("expected the chat encoding"),
        }

        let raw = serde_json::json!({
            "model": "m",
            "ocr_format": "raw",
            "messages": [],
        });
        let request = WireApi::ChatCompletions
            .lower(raw, Some(OcrFormat::Html))
            .expect("raw opts out of the default");
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, None),
            _ => panic!("expected the chat encoding"),
        }

        let unknown = serde_json::json!({
            "model": "m",
            "ocr_format": "markdown",
            "messages": [],
        });
        let err = WireApi::ChatCompletions
            .lower(unknown, None)
            .err()
            .expect("unknown format is refused");
        assert!(err.contains("ocr_format"), "{err}");

        let not_a_string = serde_json::json!({
            "model": "m",
            "ocr_format": 1,
            "messages": [],
        });
        assert!(WireApi::ChatCompletions.lower(not_a_string, None).is_err());
    }

    /// The serve-level default (`serve.ocr_format` / HIPFIRE_SERVE_OCR_FORMAT)
    /// renders every chat turn, streamed or not; a request's own field wins.
    #[test]
    fn serve_default_applies_to_turns_and_yields_to_the_request() {
        let body = serde_json::json!({"model": "m", "messages": []});
        let request = WireApi::ChatCompletions
            .lower(body, Some(OcrFormat::Html))
            .expect("valid request");
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, Some(OcrFormat::Html)),
            _ => panic!("expected the chat encoding"),
        }

        let streaming = serde_json::json!({"model": "m", "stream": true, "messages": []});
        let request = WireApi::ChatCompletions
            .lower(streaming, Some(OcrFormat::Html))
            .expect("streams take the default too");
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, Some(OcrFormat::Html)),
            _ => panic!("expected the chat encoding"),
        }

        let explicit = serde_json::json!({
            "model": "m",
            "messages": [],
            "ocr_format": "blocks",
        });
        let request = WireApi::ChatCompletions
            .lower(explicit, Some(OcrFormat::Html))
            .expect("valid request");
        match request.encoding {
            ResponseEncoding::Chat { ocr, .. } => assert_eq!(ocr, Some(OcrFormat::Blocks)),
            _ => panic!("expected the chat encoding"),
        }
    }

    /// A streamed turn decodes at the sink, before the API's encoder frames a
    /// delta: one frame per completed block, never a raw token, and nothing
    /// at all for a delta that closes no block (whatever wire API is used).
    #[test]
    fn stream_decodes_blocks_at_the_sink_for_every_wire_api() {
        for encoding in [
            ResponseEncoding::Chat {
                include_usage: false,
                ocr: Some(OcrFormat::Html),
            },
            ResponseEncoding::Anthropic {
                ocr: Some(OcrFormat::Html),
            },
        ] {
            let (sink, _commit, mut rx) = test_sink(encoding);
            sink.forward_event(&token_event("<|Text|><|0|><|0|><|1|><|1|><p>one</p>"))
                .unwrap();
            assert!(rx.try_recv().is_err(), "nothing until a block closes");
            sink.forward_event(&token_event("<|Text|><|2|><|2|><|3|><|3|>"))
                .unwrap();
            let frame = rx.try_recv().expect("block frame");
            let text = String::from_utf8_lossy(&frame.bytes).into_owned();
            assert!(text.contains("0 0 1 1"), "{text}");
            assert!(text.contains("data-label"), "{text}");
            assert!(
                !text.contains("<|Text|>"),
                "raw tokens must not reach the wire"
            );
        }
    }

    /// Reasoning and content keep separate decoders. A thinking model's
    /// thoughts never reach the content channel — neither as the lead-in
    /// before the first block nor as the whole answer at end of generation —
    /// and a block the model wrote on the reasoning channel decodes there.
    #[test]
    fn stream_keeps_reasoning_and_content_decoders_separate() {
        let (sink, _commit, mut rx) = test_sink(ResponseEncoding::Chat {
            include_usage: false,
            ocr: Some(OcrFormat::Html),
        });
        sink.forward_event(&reasoning_event("pondering the page"))
            .unwrap();
        assert!(rx.try_recv().is_err(), "reasoning sends no content frame");
        sink.forward_event(&token_event("<|Text|><|0|><|0|><|1|><|1|><p>one</p>"))
            .unwrap();
        assert!(rx.try_recv().is_err(), "nothing until a block closes");
        sink.forward_event(&token_event("<|Text|><|2|><|2|><|3|><|3|>"))
            .unwrap();
        let mut content = String::new();
        while let Ok(frame) = rx.try_recv() {
            content.push_str(&String::from_utf8_lossy(&frame.bytes));
        }
        assert!(content.contains("data-label"), "{content}");
        assert!(!content.contains("pondering"), "{content}");
        let [content_tail, reasoning_tail] = sink.ocr.as_ref().unwrap().borrow_mut().finish();
        assert_eq!(reasoning_tail, "pondering the page");
        assert!(!content_tail.contains("pondering"), "{content_tail}");

        let (sink, _commit, mut rx) = test_sink(ResponseEncoding::Chat {
            include_usage: false,
            ocr: Some(OcrFormat::Html),
        });
        sink.forward_event(&reasoning_event(
            "<|Text|><|0|><|0|><|1|><|1|><p>thought</p>",
        ))
        .unwrap();
        sink.forward_event(&reasoning_event("<|Text|><|2|><|2|><|3|><|3|>"))
            .unwrap();
        let frame = rx.try_recv().expect("reasoning block frame");
        let text = String::from_utf8_lossy(&frame.bytes).into_owned();
        assert!(text.contains("data-label"), "{text}");
        assert!(text.contains("thought"), "{text}");
        assert!(text.contains("reasoning_content"), "{text}");
        assert!(!text.contains("<|Text|>"), "{text}");
    }

    /// Non-stream adapter bodies decode too.
    #[test]
    fn adapter_bodies_decode_compact_blocks() {
        let mut anthropic = serde_json::json!({
            "content": [{ "type": "text", "text": "<|Text|><|0|><|0|><|1|><|1|><p>x</p>" }],
        });
        decode_anthropic_content(&mut anthropic, OcrFormat::Html);
        assert!(anthropic["content"][0]["text"]
            .as_str()
            .is_some_and(|text| text.starts_with("<div data-bbox=\"0 0 1 1\"")));

        let mut responses = serde_json::json!({
            "output": [{
                "type": "message",
                "content": [{ "type": "output_text", "text": "<|Text|><|0|><|0|><|1|><|1|><p>x</p>" }],
            }],
            "output_text": "<|Text|><|0|><|0|><|1|><|1|><p>x</p>",
        });
        decode_responses_content(&mut responses, OcrFormat::Html);
        assert!(responses["output"][0]["content"][0]["text"]
            .as_str()
            .is_some_and(|text| text.starts_with("<div data-bbox=\"0 0 1 1\"")));
        assert!(responses["output_text"]
            .as_str()
            .is_some_and(|text| text.starts_with("<div data-bbox=\"0 0 1 1\"")));
    }

    /// Decoded content replaces the compact form; a plain answer is untouched.
    #[test]
    fn decoded_content_replaces_the_compact_form() {
        let mut body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<|Text|><|0|><|0|><|10|><|10|><p>x</p>",
                },
            }],
        });
        decode_ocr_content(&mut body, OcrFormat::Html);
        assert_eq!(
            body.pointer("/choices/0/message/content")
                .and_then(|value| value.as_str()),
            Some("<div data-bbox=\"0 0 10 10\" data-label=\"Text\"><p>x</p></div>")
        );

        let mut plain = serde_json::json!({"choices": [{"message": {"content": "hello"}}]});
        decode_ocr_content(&mut plain, OcrFormat::Blocks);
        assert_eq!(
            plain
                .pointer("/choices/0/message/content")
                .and_then(|value| value.as_str()),
            Some("hello")
        );
    }

    /// A model served without its non-thinking default answers on the
    /// reasoning channel; `ocr_format` decodes that slot too, on every API.
    #[test]
    fn decoded_content_covers_the_reasoning_channel() {
        let mut body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "<|Text|><|1|><|1|><|2|><|2|><p>r</p>",
                },
            }],
        });
        decode_ocr_content(&mut body, OcrFormat::Html);
        assert_eq!(
            body.pointer("/choices/0/message/reasoning_content")
                .and_then(|value| value.as_str()),
            Some("<div data-bbox=\"1 1 2 2\" data-label=\"Text\"><p>r</p></div>")
        );
        assert_eq!(
            body.pointer("/choices/0/message/content")
                .and_then(|value| value.as_str()),
            Some("")
        );

        let mut anthropic = serde_json::json!({
            "content": [
                { "type": "thinking", "thinking": "<|Text|><|1|><|1|><|2|><|2|><p>r</p>", "signature": "" },
                { "type": "text", "text": "" },
            ],
        });
        decode_anthropic_content(&mut anthropic, OcrFormat::Html);
        assert_eq!(
            anthropic
                .pointer("/content/0/thinking")
                .and_then(|v| v.as_str()),
            Some("<div data-bbox=\"1 1 2 2\" data-label=\"Text\"><p>r</p></div>")
        );

        let mut responses = serde_json::json!({
            "output": [{
                "type": "reasoning",
                "content": [{ "type": "reasoning_text", "text": "<|Text|><|1|><|1|><|2|><|2|><p>r</p>" }],
            }],
        });
        decode_responses_content(&mut responses, OcrFormat::Html);
        assert_eq!(
            responses
                .pointer("/output/0/content/0/text")
                .and_then(|v| v.as_str()),
            Some("<div data-bbox=\"1 1 2 2\" data-label=\"Text\"><p>r</p></div>")
        );
    }

    #[test]
    fn bearer_headers_require_one_matching_ascii_credential() {
        let text = "k3y!".repeat(8);
        let key = BearerToken::new(text.clone()).unwrap();
        let mut headers = hyper::HeaderMap::new();
        assert!(!bearer_authorized(&headers, &key));
        for value in [
            "Bearer".to_owned(),
            format!("Basic {text}"),
            format!("Bearer {text}x"),
        ] {
            headers.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&value).unwrap(),
            );
            assert!(!bearer_authorized(&headers, &key));
        }
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("bEaReR {text}")).unwrap(),
        );
        assert!(bearer_authorized(&headers, &key));
        headers.append(
            header::AUTHORIZATION,
            header::HeaderValue::from_static("Bearer other"),
        );
        assert!(!bearer_authorized(&headers, &key));
    }

    fn noop_cx() -> TaskContext<'static> {
        TaskContext::from_waker(Waker::noop())
    }

    async fn loopback_pair() -> (TrackedIo, TcpStream, FlushAcks) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let client = TcpStream::connect(addr).await.expect("connect");
        let (server, _) = listener.accept().await.expect("accept");
        let acks = FlushAcks::new();
        let tracked = TrackedIo::new(server, acks.clone());
        (tracked, client, acks)
    }

    async fn write_all(io: &mut TrackedIo, mut buf: &[u8]) {
        while !buf.is_empty() {
            let n = poll_fn(|cx| Pin::new(&mut *io).poll_write(cx, buf))
                .await
                .expect("write");
            assert!(n > 0, "write made no progress");
            buf = &buf[n..];
        }
    }

    async fn flush(io: &mut TrackedIo) {
        poll_fn(|cx| Pin::new(&mut *io).poll_flush(cx))
            .await
            .expect("flush");
    }

    #[tokio::test]
    async fn ack_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"{}".to_vec(), ack_tx, acks.clone());

        let mut cx = noop_cx();
        // Yield data frame — registers with tracker, must not complete Ok yet.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame.data_ref().is_some_and(|d| d.as_ref() == b"{}"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Second poll is EOF — still no Ok ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "body poll must not ack before socket flush"
        );

        // Completing the tracker flush path delivers Ok.
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_poll_alone_does_not_ack() {
        let acks = FlushAcks::new();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        tx.try_send(ResponseChunk::last(b"data: hi\n\n".to_vec(), Some(ack_tx)))
            .expect("send chunk");
        drop(tx);

        let cancelled = Arc::new(AtomicBool::new(false));
        let mut body = ChannelBody::new(
            VecDeque::new(),
            rx,
            acks.clone(),
            Arc::clone(&cancelled),
            Duration::from_secs(60),
        );
        let mut cx = noop_cx();
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(Some(Ok(frame))) => {
                assert!(frame
                    .data_ref()
                    .is_some_and(|d| d.as_ref() == b"data: hi\n\n"));
            }
            other => panic!("expected data frame, got {other:?}"),
        }
        // Further polls (EOF) must not complete the terminal ack.
        match Pin::new(&mut body).poll_frame(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("expected EOF, got {other:?}"),
        }
        assert!(
            ack_rx.try_recv().is_err(),
            "channel body poll must not ack before socket flush"
        );
        acks.drain_ok();
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
    }

    #[tokio::test]
    async fn channel_body_drop_sets_cancelled() {
        let acks = FlushAcks::new();
        let (_tx, rx) = tokio::sync::mpsc::channel::<ResponseChunk>(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        assert!(!cancelled.load(Ordering::SeqCst), "flag must start false");
        let body = ChannelBody::new(
            VecDeque::new(),
            rx,
            acks,
            Arc::clone(&cancelled),
            Duration::from_secs(60),
        );
        drop(body);
        assert!(
            cancelled.load(Ordering::SeqCst),
            "dropping ChannelBody must set cancelled"
        );
    }

    /// A silent stream gets `: keepalive` comments; after the `[DONE]` frame
    /// nothing more is written, however long the body stays open.
    #[tokio::test]
    async fn channel_body_keepalive_stops_at_the_last_frame() {
        async fn next(body: &mut ChannelBody) -> Option<Result<Frame<Bytes>, io::Error>> {
            poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await
        }
        fn data(frame: Option<Result<Frame<Bytes>, io::Error>>) -> Bytes {
            frame
                .expect("frame")
                .expect("data frame")
                .into_data()
                .expect("data")
        }
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut body = ChannelBody::new(
            VecDeque::new(),
            rx,
            FlushAcks::new(),
            Arc::new(AtomicBool::new(false)),
            Duration::from_millis(30),
        );

        assert_eq!(data(next(&mut body).await), SSE_KEEPALIVE);
        tx.send(ResponseChunk::plain(b"data: x\n\n".to_vec()))
            .await
            .unwrap();
        assert_eq!(data(next(&mut body).await), &b"data: x\n\n"[..]);
        tx.send(ResponseChunk::last(b"data: [DONE]\n\n".to_vec(), None))
            .await
            .unwrap();
        assert_eq!(data(next(&mut body).await), &b"data: [DONE]\n\n"[..]);
        assert!(
            tokio::time::timeout(Duration::from_millis(150), next(&mut body))
                .await
                .is_err(),
            "no frame may follow [DONE]"
        );
        drop(tx);
        assert!(next(&mut body).await.is_none());
    }

    fn progress_event(done: u64, total: u64) -> serde_json::Value {
        serde_json::json!({
            "type": "prefill_progress",
            "id": "chatcmpl-1-1",
            "attempt_id": 1,
            "done": done,
            "total": total,
        })
    }

    fn token_event(text: &str) -> serde_json::Value {
        serde_json::json!({ "type": "token", "id": "chatcmpl-1-1", "attempt_id": 1, "text": text })
    }

    fn reasoning_event(text: &str) -> serde_json::Value {
        serde_json::json!({ "type": "reasoning", "id": "chatcmpl-1-1", "attempt_id": 1, "text": text })
    }

    type TestSink = (
        SseSink,
        Arc<Mutex<StreamCommit>>,
        tokio::sync::mpsc::Receiver<ResponseChunk>,
    );

    fn test_sink(encoding: ResponseEncoding) -> TestSink {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let commit = Arc::new(Mutex::new(StreamCommit::Pending));
        let ocr = encoding
            .ocr()
            .map(|format| RefCell::new(StreamOcr::new(format)));
        let sink = SseSink {
            tx,
            commit: Arc::clone(&commit),
            committed: Cell::new(false),
            terminal_sent: Cell::new(false),
            id: "chatcmpl-1-1".to_owned(),
            created: 0,
            encoder: RefCell::new(encoding.stream("chatcmpl-1-1", 0, "m")),
            ocr,
            stall_timeout: None,
        };
        (sink, commit, rx)
    }

    /// Progress before the response commits is dropped: it sends nothing,
    /// does not commit, and a failure right after it is still the worker's
    /// ordinary pre-commit rejection (a JSON error, never a 200 stream).
    #[test]
    fn prefill_progress_before_commit_is_dropped_and_does_not_commit() {
        let (sink, commit, mut rx) = test_sink(ResponseEncoding::Chat {
            include_usage: false,
            ocr: None,
        });
        sink.forward_event(&progress_event(512, 2048)).unwrap();
        sink.forward_event(&progress_event(1024, 2048)).unwrap();
        assert!(rx.try_recv().is_err(), "nothing may reach the wire");
        assert!(!sink.committed.get());
        assert!(matches!(*commit.lock().unwrap(), StreamCommit::Pending));
        sink.finish(Err(anyhow!("model failed")), &Metrics::default());
        assert!(
            matches!(*commit.lock().unwrap(), StreamCommit::Rejected(_)),
            "the failure must stay a pre-commit rejection"
        );
    }

    /// After the first frame the comment goes out, exactly framed, in every
    /// API's stream; the encoders never turn it into a delta or an event.
    #[test]
    fn prefill_progress_after_commit_is_a_comment_frame_in_every_api() {
        let responses = responses::lower_request(&serde_json::json!({
            "model": "m", "input": "hi", "stream": true
        }))
        .unwrap()
        .echo;
        for encoding in [
            ResponseEncoding::Chat {
                include_usage: false,
                ocr: None,
            },
            ResponseEncoding::Anthropic { ocr: None },
            ResponseEncoding::Responses {
                echo: responses,
                ocr: None,
            },
        ] {
            let api = encoding.api();
            let (sink, commit, mut rx) = test_sink(encoding);
            assert!(
                sink.encoder
                    .borrow_mut()
                    .event(&progress_event(1, 2))
                    .is_none(),
                "{api:?}: progress is not a delta"
            );
            sink.forward_event(&token_event("hi")).unwrap();
            assert!(matches!(*commit.lock().unwrap(), StreamCommit::Committed));
            let first = rx.try_recv().expect("token frame");
            assert_ne!(first.bytes.first(), Some(&b':'), "{api:?}");
            sink.forward_event(&progress_event(512, 2048)).unwrap();
            let frame = rx.try_recv().expect("progress frame");
            assert_eq!(frame.bytes, b": prefill 512/2048\n\n", "{api:?}");
            assert!(!frame.last && frame.ack.is_none());
            // Malformed progress is ignored, not framed.
            sink.forward_event(&serde_json::json!({"type": "prefill_progress"}))
                .unwrap();
            assert!(rx.try_recv().is_err(), "{api:?}");
        }
    }

    /// The handler also commits on its silence deadline while a long prefill
    /// is still running; progress then flows even though no frame was sent.
    #[test]
    fn prefill_progress_flows_once_the_handler_has_committed() {
        let (sink, commit, mut rx) = test_sink(ResponseEncoding::Chat {
            include_usage: false,
            ocr: None,
        });
        sink.forward_event(&progress_event(512, 2048)).unwrap();
        assert!(rx.try_recv().is_err());
        commit_stream(&commit).unwrap();
        sink.forward_event(&progress_event(1024, 2048)).unwrap();
        assert_eq!(
            rx.try_recv().expect("progress frame").bytes,
            b": prefill 1024/2048\n\n"
        );
    }

    /// Nothing follows the terminal frame, progress included.
    #[test]
    fn prefill_progress_never_follows_the_terminal_frame() {
        let (sink, _commit, mut rx) = test_sink(ResponseEncoding::Chat {
            include_usage: false,
            ocr: None,
        });
        sink.forward_event(&token_event("hi")).unwrap();
        rx.try_recv().expect("token frame");
        sink.terminal_sent.set(true);
        sink.forward_event(&progress_event(2047, 2048)).unwrap();
        assert!(rx.try_recv().is_err());
    }

    /// A progress frame is activity: the keepalive timer restarts from it.
    #[tokio::test]
    async fn prefill_progress_frame_restarts_the_keepalive_timer() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let mut body = ChannelBody::new(
            VecDeque::new(),
            rx,
            FlushAcks::new(),
            Arc::new(AtomicBool::new(false)),
            Duration::from_millis(120),
        );
        let started = Instant::now();
        tx.send(ResponseChunk::plain(sse_prefill_comment(512, 2048)))
            .await
            .unwrap();
        let frame = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .expect("frame")
            .expect("data frame")
            .into_data()
            .expect("data");
        assert_eq!(frame, &b": prefill 512/2048\n\n"[..]);
        let next = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .expect("frame")
            .expect("data frame")
            .into_data()
            .expect("data");
        assert_eq!(next, SSE_KEEPALIVE);
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "keepalive fired {:?} after the progress frame",
            started.elapsed()
        );
    }

    /// Every `/v1/embeddings` input-shape refusal is the caller's request, not
    /// a server fault: bare content arrays, list arity, marker/image count and
    /// a bad data URL all answer 400.
    #[test]
    fn embeddings_refusals_are_client_errors() {
        for message in [
            "input is required",
            "input has 2 entries; this endpoint accepts one input per request",
            "input looks like a bare content-part array — wrap the parts in a content object",
            "input content[0]: image_url is not valid base64: Invalid symbol 33",
            "embed: text carries 2 <|image|> marker(s) but the request has 1 image(s)",
        ] {
            assert_eq!(embeddings_error_status(message), 400, "{message}");
        }
    }

    #[tokio::test]
    async fn tracked_flush_delivers_ok_ack() {
        let (mut tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"ok".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");

        write_all(&mut tracked, b"ok").await;
        // Write alone must not ack — only successful poll_flush.
        assert!(ack_rx.try_recv().is_err(), "no ack before flush");
        flush(&mut tracked).await;
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));

        // Prevent Drop from racing: body already registered; IO drop drains empty.
        drop(body);
        drop(tracked);
    }

    #[tokio::test]
    async fn tracked_io_drop_before_flush_fails_ack() {
        let (tracked, _client, acks) = loopback_pair().await;
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let mut body = AckBody::new(b"x".to_vec(), ack_tx, acks);

        let mut cx = noop_cx();
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut cx),
            Poll::Ready(Some(Ok(_)))
        ));
        // Registered with tracker; drop IO without flush → Cancelled/Err.
        drop(body);
        drop(tracked);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }

    #[tokio::test]
    async fn ack_body_drop_before_registration_fails() {
        let acks = FlushAcks::new();
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        let body = AckBody::new(b"x".to_vec(), ack_tx, acks);
        drop(body);
        assert_eq!(ack_rx.recv_timeout(Duration::from_secs(1)), Ok(Err(())));
    }

    /// `/v1/images/edits` reads the OpenAI multipart shape: text fields plus
    /// `image` file parts, whose bytes (binary, CRLF included) reach the
    /// daemon as base64 and never as a path.
    #[test]
    fn multipart_edit_form_carries_image_bytes_not_paths() {
        let boundary = "xYz";
        let png_bytes = b"\x89PNG\r\n\x1a\n\r\n--not-a-boundary".to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(
            b"--xYz\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it blue\r\n",
        );
        body.extend_from_slice(
            b"--xYz\r\nContent-Disposition: form-data; name=\"steps\"\r\n\r\n4\r\n",
        );
        body.extend_from_slice(
            b"--xYz\r\nContent-Disposition: form-data; name=\"image\"; filename=\"ref.png\"\r\nContent-Type: image/png\r\n\r\n",
        );
        body.extend_from_slice(&png_bytes);
        body.extend_from_slice(b"\r\n--xYz--\r\n");
        assert_eq!(
            multipart_boundary("multipart/form-data; boundary=xYz"),
            Some(boundary)
        );
        assert_eq!(multipart_boundary("application/json"), None);
        let (fields, images) = parse_multipart(&body, boundary).unwrap();
        assert_eq!(
            fields,
            vec![
                ("prompt".to_string(), "make it blue".to_string()),
                ("steps".to_string(), "4".to_string())
            ]
        );
        assert_eq!(images, vec![png_bytes.clone()]);
        let json = edits_form_to_body(fields, images).unwrap();
        assert_eq!(json["prompt"], "make it blue");
        assert_eq!(json["steps"], 4);
        use base64::Engine as _;
        assert_eq!(
            json["_reference_images"][0],
            base64::engine::general_purpose::STANDARD.encode(&png_bytes)
        );
        // A file part under any other name is refused, and no image is an error.
        let mut other = Vec::new();
        other.extend_from_slice(b"--xYz\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"m.png\"\r\n\r\nx\r\n--xYz--\r\n");
        assert!(parse_multipart(&other, boundary)
            .unwrap_err()
            .contains("only `image`"));
        assert!(edits_form_to_body(vec![], vec![])
            .unwrap_err()
            .contains("required"));
    }

    fn typed_daemon_error(class: &str, message: &str) -> anyhow::Error {
        anyhow::Error::new(hipfire_client::ClientError::Daemon(
            hipfire_client::TypedDaemonError {
                message: message.to_owned(),
                class: class.to_owned(),
                retryable: false,
                rolled_back: false,
                attempt_id: 1,
                id: None,
            },
        ))
    }

    /// A typed daemon error gets its status from `class`, a gateway error
    /// from its type; no message text is ever read.
    #[test]
    fn request_error_status_maps_daemon_class_before_message() {
        for (class, status) in [
            ("validation", 400),
            ("context_length", 400),
            ("unsupported", 400),
            ("transient", 503),
            ("internal", 500),
            ("gpu", 500),
            ("malformed", 500),
        ] {
            assert_eq!(
                request_error_status(&typed_daemon_error(class, "boom")),
                status,
                "class {class}"
            );
        }
        // Text that reads like a client error does not override a
        // server-fault class, and vice versa.
        assert_eq!(
            request_error_status(&typed_daemon_error("internal", "invalid state: required")),
            500
        );
        assert_eq!(
            request_error_status(&typed_daemon_error("validation", "seed must be an integer")),
            400
        );
        // Context layers added on the way up keep the daemon's class.
        assert_eq!(
            request_error_status(
                &typed_daemon_error("transient", "prefill glitch")
                    .context("retry aborted: admission re-acquire failed")
            ),
            503
        );
        // Gateway errors map by type, through context layers.
        assert_eq!(
            request_error_status(
                &crate::serve::invalid_request!("unsupported tool_choice value: x")
                    .context("projecting request")
            ),
            400
        );
        assert_eq!(
            request_error_status(&anyhow::Error::new(crate::serve::ModelNotFound(
                "model not found: x".into()
            ))),
            404
        );
        // Untyped errors are server faults, whatever their wording.
        for message in [
            "boom",
            "max_tokens must be between 1 and 393216",
            "model not found: x",
            "invalid state: required",
        ] {
            assert_eq!(
                request_error_status(&anyhow!("{message}")),
                500,
                "{message}"
            );
        }
        assert_eq!(
            request_error_status(&anyhow::Error::new(hipfire_client::ClientError::Protocol(
                "malformed canonical tool call: x".into()
            ))),
            500
        );
    }

    #[test]
    fn model_entry_publishes_the_load_summary_only_for_the_resident_model() {
        let summary = serde_json::json!({
            "line": "hipfire load summary: m (qwen3_5) | spec: off (no speculator armed)",
            "spec": "off (no speculator armed)",
            "kv": "fp8 (vmm, automatic)",
            "context": "4096 (user-set; model 8192, card 8192)",
            "placement": "fully resident",
            "prefill": "per-prompt chunking",
        });
        let loaded = LoadedInfo {
            n_embd: 2048,
            load_summary: summary.clone(),
            ..LoadedInfo::default()
        };
        let resident = model_entry("qwen3.5:2b", None, Some((&loaded, 4096)));
        assert_eq!(resident["load_summary"], summary);
        // A pre-summary daemon ack leaves the field absent, not null.
        let old_ack = LoadedInfo {
            n_embd: 2048,
            ..LoadedInfo::default()
        };
        assert!(model_entry("qwen3.5:2b", None, Some((&old_ack, 4096)))
            .get("load_summary")
            .is_none());
        assert!(model_entry("qwen3.6:27b", None, None)
            .get("load_summary")
            .is_none());
    }

    #[test]
    fn model_entry_publishes_runtime_facts_only_for_the_resident_model() {
        let loaded = LoadedInfo {
            vision: false,
            audio: false,
            video: false,
            n_embd: 2048,
            n_vocab: 248320,
            reasoning_contract: String::new(),
            reasoning_effort_native: false,
            reasoning_efforts: Vec::new(),
            reasoning_effort_probe_error: None,
            ..LoadedInfo::default()
        };
        let resident = model_entry("qwen3.5:2b", Some(1790385975), Some((&loaded, 82944)));
        assert_eq!(resident["created"], serde_json::json!(1790385975u64));
        assert_eq!(resident["meta"]["n_ctx"], serde_json::json!(82944u64));
        assert_eq!(resident["meta"]["n_embd"], serde_json::json!(2048u64));
        assert_eq!(resident["meta"]["n_vocab"], serde_json::json!(248320u64));
        assert_eq!(
            resident["architecture"]["input_modalities"],
            serde_json::json!(["text"])
        );
        assert_eq!(
            resident["architecture"]["output_modalities"],
            serde_json::json!(["text"])
        );

        // A non-resident candidate keeps only the spec-required fields plus
        // `created`; nothing measured is advertised for it.
        let candidate = model_entry("qwen3.6:27b", Some(1780057933), None);
        assert_eq!(candidate["created"], serde_json::json!(1780057933u64));
        assert!(candidate.get("meta").is_none());
        assert!(candidate.get("architecture").is_none());
    }

    #[test]
    fn model_entry_advertises_image_input_only_when_a_tower_loaded() {
        let vision = LoadedInfo {
            vision: true,
            audio: false,
            video: false,
            n_embd: 5120,
            n_vocab: 248320,
            reasoning_contract: String::new(),
            reasoning_effort_native: false,
            reasoning_efforts: Vec::new(),
            reasoning_effort_probe_error: None,
            ..LoadedInfo::default()
        };
        let entry = model_entry("qwen3.8:27b", None, Some((&vision, 150000)));
        assert_eq!(
            entry["architecture"]["input_modalities"],
            serde_json::json!(["text", "image"])
        );
    }

    #[test]
    fn model_entry_advertises_every_resident_embed_modality() {
        // arch 18 with both towers resident: all four inputs. The wire order
        // is text, image, audio, video — a harness keying on position or
        // membership must see the same list the embed body admits.
        let eg2 = LoadedInfo {
            vision: true,
            audio: true,
            video: true,
            n_embd: 512,
            n_vocab: 262144,
            ..LoadedInfo::default()
        };
        let entry = model_entry("embeddinggemma-2:q8f16", None, Some((&eg2, 8192)));
        assert_eq!(
            entry["architecture"]["input_modalities"],
            serde_json::json!(["text", "image", "audio", "video"])
        );

        // Audio without video cannot happen today (both ride the arch-18
        // `.vl` pack), but the list is per-fact, not a package deal: a
        // video-capable ack whose vision tower is absent advertises no
        // image and no video — the tower the frames would ride is the one
        // that refused the load.
        let audio_only = LoadedInfo {
            vision: false,
            audio: true,
            video: false,
            n_embd: 512,
            n_vocab: 262144,
            ..LoadedInfo::default()
        };
        let entry = model_entry("embeddinggemma-2:q8f16", None, Some((&audio_only, 8192)));
        assert_eq!(
            entry["architecture"]["input_modalities"],
            serde_json::json!(["text", "audio"])
        );
    }

    #[test]
    fn model_entry_omits_unmeasured_facts_before_the_load_ack() {
        // Resident id set, ack not in yet: every runtime fact is unknown, so
        // nothing may be invented — not even a text-only modality claim.
        let empty = LoadedInfo::default();
        let entry = model_entry("qwen3.5:2b", None, Some((&empty, 0)));
        assert!(entry.get("created").is_none());
        assert!(entry.get("meta").is_none());
        assert!(entry.get("architecture").is_none());
    }

    #[test]
    fn loaded_config_json_reports_the_resident_models_ack_facts() {
        let mut meta = crate::serve::ServeMeta::new("test".to_owned());
        meta.current_model = Some("qwen3.5:4b".to_owned());
        meta.loaded = LoadedInfo {
            kv_mode: Some("asym3".to_owned()),
            kv_backend: Some("paged".to_owned()),
            max_seq: Some(32768),
            ..LoadedInfo::default()
        };
        assert_eq!(
            loaded_config_json(&meta),
            serde_json::json!({
                "model": "qwen3.5:4b",
                "kv_mode": "asym3",
                "kv_backend": "paged",
                "max_seq": 32768u64,
            })
        );

        // Resident, but the ack carried none of the facts: object with nulls.
        meta.loaded = LoadedInfo::default();
        assert_eq!(
            loaded_config_json(&meta),
            serde_json::json!({
                "model": "qwen3.5:4b",
                "kv_mode": null,
                "kv_backend": null,
                "max_seq": null,
            })
        );
    }

    #[test]
    fn loaded_config_json_is_null_without_a_resident_model() {
        let meta = crate::serve::ServeMeta::new("test".to_owned());
        assert_eq!(loaded_config_json(&meta), serde_json::Value::Null);
    }

    #[test]
    fn model_entry_publishes_reasoning_facts_for_resident_model() {
        // The same facts /health serves from meta.loaded, on the model
        // discovery surface, and skipped while the contract is unknown
        // (load window / non-reasoning default-empty).
        let loaded = LoadedInfo {
            reasoning_contract: "qwen_jinja".into(),
            reasoning_effort_native: false,
            reasoning_efforts: vec!["low".into(), "high".into()],
            n_embd: 2048,
            ..LoadedInfo::default()
        };
        let entry = model_entry("qwen3.5:4b", None, Some((&loaded, 8192)));
        assert_eq!(entry["architecture"]["reasoning_contract"], "qwen_jinja");
        assert_eq!(entry["architecture"]["reasoning_effort_native"], false);
        assert_eq!(
            entry["architecture"]["reasoning_efforts"],
            serde_json::json!(["low", "high"])
        );
        // Empty contract (ack not in): no reasoning keys at all.
        let unknown = LoadedInfo {
            n_embd: 2048,
            ..LoadedInfo::default()
        };
        let bare = model_entry("qwen3.5:4b", None, Some((&unknown, 8192)));
        assert!(bare["architecture"].get("reasoning_contract").is_none());
    }
}
