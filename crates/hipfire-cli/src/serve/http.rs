// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! OpenAI HTTP gateway.
//!
//! Concern: request routing, JSON body limits, streaming vs non-streaming
//! framing, SSE acknowledgement, CORS/health endpoints. Isolates Hyper
//! I/O from business logic.

use crate::serve::complete::{
    complete_request_cancellable, completion_json, gate_chat_completions_tools,
    openai_stream_delta_for_event, openai_stream_terminal_chunks, Completion,
};
use crate::serve::metrics::Metrics;
use crate::serve::{is_batch_eligible_request, ServeShared};
use crate::serve::{AdmissionError, AdmissionGuard};
use crate::{list_local_models, unix_timestamp};
use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use http_body_util::{combinators::UnsyncBoxBody, BodyExt, Full};
use hyper::server::conn::http1;
use hyper::{
    body::{Frame, Incoming},
    header, Method, Request, Response,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    cell::Cell,
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
    time::Duration,
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

fn boxed_full(bytes: Vec<u8>) -> BoxBody {
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
    metrics: &Metrics,
    error: &AdmissionError,
) -> Response<BoxBody> {
    metrics.record_admission_rejected();
    let mut resp = openai_error(&error.message, 503);
    if let Ok(retry_after) = header::HeaderValue::from_str(&error.retry_after_seconds.to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, retry_after);
    }
    resp
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
    /// Ends with `data: [DONE]`: nothing, not even a keepalive, may follow.
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
    /// The final frame of a response (its bytes end with `data: [DONE]`).
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
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(30);

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

    match (method, path.as_str()) {
        (Method::GET, "/health") => {
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
                "native": true,
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
            });
            json_response(body, 200)
        }
        (Method::GET, "/metrics") => {
            let (uptime, model) = {
                let meta = shared.meta.lock().unwrap_or_else(|e| e.into_inner());
                (meta.started.elapsed().as_secs(), meta.current_model.clone())
            };
            let body = shared.metrics.render(
                shared.admission.inflight(),
                shared.admission.capacity(),
                uptime,
                model.as_deref(),
            );
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
            let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
            let local = match list_local_models(&runtime.paths, &runtime.registry) {
                Ok(m) => m,
                Err(e) => return openai_error(&e.to_string(), 500),
            };
            let body = serde_json::json!({
                "object": "list",
                "data": local.into_iter().map(|model| serde_json::json!({
                    "id": model.registry_tag.unwrap_or(model.name),
                    "object": "model",
                    "owned_by": "hipfire",
                })).collect::<Vec<_>>()
            });
            json_response(body, 200)
        }
        (Method::OPTIONS, _) => {
            let mut resp = Response::builder()
                .status(204)
                .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
                .header(
                    header::ACCESS_CONTROL_ALLOW_HEADERS,
                    "Content-Type, Authorization",
                )
                .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, POST, OPTIONS")
                .body(boxed_empty())
                .unwrap();
            resp
        }
        (Method::POST, "/v1/chat/completions") => {
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

            let (is_eligible, model_for_lease) = {
                let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
                let tp = runtime.tp;
                let arch = runtime.current_arch.clone();
                let batch_capable = runtime.continuous_batch_capable;
                let multi_slot = runtime.multi_slot_enabled;
                drop(runtime);
                // The admission gate is transport concurrency, not a batch-mode
                // selector. Experimental slots overlap independent requests
                // while remaining separate from ContinuousBatchScheduler.
                let eligible = multi_slot
                    || is_batch_eligible_request(&body_val, tp, arch.as_deref(), batch_capable);
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
                    Err(e) => return admission_error_response(&shared.metrics, &e),
                }
            } else {
                match shared.admission.acquire_async(cancel.clone()).await {
                    Ok(g) => g,
                    Err(e) => return admission_error_response(&shared.metrics, &e),
                }
            };

            if let Err(error) = gate_chat_completions_tools(&body_val) {
                return RequestFailure::new(error.to_string(), 400).respond(&shared.metrics);
            }

            let is_stream = body_val.get("stream").and_then(|v| v.as_bool()) == Some(true);
            let response = if is_stream {
                handle_streaming(shared, body_val, guard, cancelled, acks).await
            } else {
                handle_nonstreaming(shared, body_val, guard, cancelled, acks).await
            };
            cancel_guard.disarm();
            response
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
                    RequestFailure::new(message, status).respond(&shared.metrics)
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
                    RequestFailure::new(message, status).respond(&shared.metrics)
                }
            }
        }
        _ => openai_error("not found", 404),
    }
}

// ---------------------------------------------------------------------------
// Images (OpenAI-compatible txt2img)
// ---------------------------------------------------------------------------

/// `/v1/images/generations`: validate the OpenAI-shaped body, forward it as
/// one daemon `img_generate`, and return a single JSON response carrying the
/// PNG as `b64_json`. The daemon (not the gateway) is the authority on
/// sampler/geometry validation — everything it refuses surfaces as a 4xx
/// here with its message.
async fn handle_images_generations(
    shared: Arc<ServeShared>,
    body: serde_json::Value,
) -> Result<Response<BoxBody>, String> {
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
        Err(error) => return Ok(admission_error_response(&shared.metrics, &error)),
    };

    let (engine, loaded_model) = {
        let runtime = shared.runtime.lock().unwrap_or_else(|e| e.into_inner());
        (runtime.engine.clone(), runtime.current_path.clone())
    };
    let model_echo = loaded_model
        .as_ref()
        .map(|p| p.display().to_string())
        .or(request_model)
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
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ResponseChunk>(32);
    let commit = Arc::new(Mutex::new(StreamCommit::Pending));
    let sink = SseSink {
        tx,
        commit: Arc::clone(&commit),
        committed: Cell::new(false),
        terminal_sent: Cell::new(false),
        id: request_id(),
        created: unix_timestamp(),
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_owned(),
        include_usage: body
            .pointer("/stream_options/include_usage")
            .and_then(|v| v.as_bool())
            == Some(true),
    };
    let role = serde_json::json!({
        "id": sink.id,
        "object": "chat.completion.chunk",
        "created": sink.created,
        "model": sink.model,
        "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
    });
    let mut first = VecDeque::from([ResponseChunk::plain(sse_data(&role))]);

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
                    Err(failure) => failure.respond(&shared.metrics),
                    Ok(()) => RequestFailure::new("generation ended without a response", 500)
                        .respond(&shared.metrics),
                };
            }
        },
        _ = tokio::time::sleep(SSE_SILENCE_LIMIT) => {
            if let Err(failure) = commit_stream(&commit) {
                return failure.respond(&shared.metrics);
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
    guard: AdmissionGuard,
    cancelled: Arc<AtomicBool>,
    acks: FlushAcks,
) -> Response<BoxBody> {
    // Staged terminal channel: Ok((bytes, ack_tx)) for success, Err(failure) for preterminal.
    let (staged_tx, staged_rx) = tokio::sync::oneshot::channel::<
        Result<(Vec<u8>, std::sync::mpsc::Sender<Result<(), ()>>), RequestFailure>,
    >();
    let staged_tx = Arc::new(Mutex::new(Some(staged_tx)));

    let shared_clone = Arc::clone(&shared);
    let body_for_worker = body;
    let staged_tx_clone = Arc::clone(&staged_tx);
    let staged_tx_for_worker = Arc::clone(&staged_tx);
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let staged_for_terminal = Arc::clone(&staged_tx_for_worker);
            let terminal = |completion: &Completion| {
                let bytes = serde_json::to_vec(&completion_json(completion)).map_err(|err| {
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
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
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
        Ok(Err(failure)) => failure.respond(&shared.metrics),
        Err(_) => {
            RequestFailure::new("generation worker disconnected", 500).respond(&shared.metrics)
        }
    }
}

// ---------------------------------------------------------------------------
// Preserved business helpers (same shapes, adapted to tokio mpsc where needed)
// ---------------------------------------------------------------------------

/// Retry-After for a typed `transient` daemon failure (HTTP 503).
const TRANSIENT_RETRY_AFTER_SECS: u64 = 1;

/// A failed chat completion. Before a stream commits it is an OpenAI JSON
/// error with `status`; after, the same body goes out as an SSE error event.
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
    fn respond(&self, metrics: &Metrics) -> Response<BoxBody> {
        metrics.record_failure();
        let mut resp = openai_error(&self.message, self.status);
        if self.status == 503 {
            resp.headers_mut().insert(
                header::RETRY_AFTER,
                header::HeaderValue::from(TRANSIENT_RETRY_AFTER_SECS),
            );
        }
        resp
    }

    /// The error event and `[DONE]` that end a stream failing after commit.
    fn sse_event(&self) -> Vec<u8> {
        let mut bytes = sse_data(&openai_error_body(&self.message, self.status));
        bytes.extend_from_slice(b"data: [DONE]\n\n");
        bytes
    }
}

/// HTTP status for a failed completion. A typed daemon error maps on its wire
/// `class`, never on its message: `validation`, `context_length` and
/// `unsupported` are the client's to fix (400), `transient` is worth a retry
/// (503), and every other class is a server fault (500). Only gateway-side
/// errors, which carry no class, fall back to matching their message.
pub(crate) fn request_error_status(error: &anyhow::Error) -> u16 {
    use hipfire_client::error_class;
    let typed = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<hipfire_client::ClientError>())
        .and_then(hipfire_client::ClientError::typed_daemon);
    if let Some(typed) = typed {
        return match typed.class.as_str() {
            error_class::VALIDATION | error_class::CONTEXT_LENGTH | error_class::UNSUPPORTED => 400,
            error_class::TRANSIENT => 503,
            _ => 500,
        };
    }
    let lower = error.to_string().to_ascii_lowercase();
    if lower.contains("model not found") {
        404
    } else if lower.contains("kv budget")
        || lower.contains("max_tokens")
        || lower.contains("invalid")
        || lower.contains("required")
        || lower.contains("endpoint adapter")
        || lower.contains("lossy")
        || lower.contains("malformed canonical tool call")
    {
        400
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
    format!("data: {}\n\n", value).into_bytes()
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

/// Worker end of one OpenAI SSE response.
struct SseSink {
    tx: tokio::sync::mpsc::Sender<ResponseChunk>,
    commit: Arc<Mutex<StreamCommit>>,
    /// Worker-side copy of "committed", so each token skips the lock.
    committed: Cell<bool>,
    /// The acknowledged terminal (`[DONE]`) frame was handed to the body.
    terminal_sent: Cell<bool>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
}

impl SseSink {
    /// Send one frame, committing the response first if this is the first.
    /// A dropped receiver maps to `Cancelled`.
    fn send(&self, chunk: ResponseChunk) -> Result<(), hipfire_client::ClientError> {
        if !self.committed.get() {
            // Only this worker ever rejects, and only after its last send, so
            // committing cannot fail here.
            let _ = commit_stream(&self.commit);
            self.committed.set(true);
        }
        self.tx
            .blocking_send(chunk)
            .map_err(|_| hipfire_client::ClientError::Cancelled)
    }

    /// Forward one logical generate event. Delta-bearing events serialize to
    /// plain (no-ack) SSE bytes. No-delta mid-stream events are silent —
    /// terminal ack handles pure-tool delivery.
    fn forward_event(&self, event: &serde_json::Value) -> Result<(), hipfire_client::ClientError> {
        let Some(delta) = openai_stream_delta_for_event(event) else {
            return Ok(());
        };
        let chunk = serde_json::json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
        });
        self.send(ResponseChunk::plain(sse_data(&chunk)))
    }

    /// Serialize terminal tool_calls (if safe), finish, optional usage, and
    /// `[DONE]` into one acknowledged chunk; wait until the socket flushed it.
    fn deliver_terminal(&self, completion: &Completion) -> Result<(), hipfire_client::ClientError> {
        let mut bytes = Vec::new();
        for chunk in openai_stream_terminal_chunks(completion, self.include_usage) {
            bytes.extend_from_slice(&sse_data(&chunk));
        }
        bytes.extend_from_slice(b"data: [DONE]\n\n");
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        self.terminal_sent.set(true);
        self.send(ResponseChunk::last(bytes, Some(ack_tx)))?;
        match ack_rx.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
        }
    }

    /// Close the body after `complete_request_cancellable`.
    /// - Success or cancel: nothing more to send.
    /// - Failure before the first frame: hand it to the handler, which answers
    ///   with its status instead of a 200.
    /// - Failure after commit: an SSE error event, then `[DONE]`.
    /// - Failure after the acknowledged `[DONE]` (e.g. the daemon commit):
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
        let _ = self
            .tx
            .blocking_send(ResponseChunk::last(failure.sse_event(), None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::Body;
    use std::future::poll_fn;
    use std::task::Waker;
    use std::time::Duration;

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

    /// A typed daemon error gets its status from `class`, whatever its text
    /// says; only classless gateway errors are matched by message.
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
        // Text that the gateway fallback reads as a client error does not
        // override a server-fault class, and vice versa.
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
        // Classless gateway errors keep the message fallback.
        assert_eq!(
            request_error_status(&anyhow!("max_tokens must be between 1 and 393216")),
            400
        );
        assert_eq!(request_error_status(&anyhow!("model not found: x")), 404);
        assert_eq!(
            request_error_status(&anyhow::Error::new(hipfire_client::ClientError::Protocol(
                "failed to serialize request".into()
            ))),
            500
        );
    }
}
