// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Authenticated serve management endpoint.
//!
//! Concern: a stop channel that exists independently of the inference
//! listener. On Windows, where `hipfire stop` has no signal to send, every
//! serve binds a second loopback listener on an ephemeral port, authenticated
//! by a per-process 32-byte random credential recorded in `serve.pid` (the
//! file inherits the user-profile ACL of the hipfire home). Unix serves keep
//! SIGTERM and do not bind it; the module compiles everywhere so its tests run
//! on Linux. The only route is
//! `POST /api/v1/shutdown`, which cancels the same `CancellationToken` that
//! SIGINT/SIGTERM cancel, so the existing bounded drain applies unchanged.
//!
//! The inference listener and `ServeShared` never see this state: the
//! credential and the cancellation hook live only in this module's server task.
//! The credential is a secret: nothing here logs it, and [`ManagementCredential`]
//! has a redacted `Debug`.

use anyhow::{anyhow, Result};
use bytes::Bytes;
use hipfire_config::secret::BearerToken;
use http_body_util::Full;
use hyper::{body::Incoming, header, http::HeaderMap, server::conn::http1, Method, Request};
use hyper::{Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    convert::Infallible,
    fmt::Write as _,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

/// The one management route.
pub(crate) const SHUTDOWN_PATH: &str = "/api/v1/shutdown";

/// Most management connections served at once. The endpoint has one caller
/// (`hipfire stop`), so this only bounds what a local misbehaving client can
/// hold open.
const MAX_CONNECTIONS: usize = 8;

/// A client must deliver its complete request head this soon after connecting.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Whole-connection ceiling, including delivery of the response.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);

/// Request head size cap (hyper's minimum permitted buffer).
const MAX_HEAD_BYTES: usize = 8192;

/// Pause after a failed `accept`.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(50);

/// Per-process management secret: 32 bytes from the OS CSPRNG, hex encoded,
/// held as a [`BearerToken`] (constant-time comparison, redacted `Debug`).
///
/// `Debug` is redacted so no `{:?}` of a struct holding it can leak it.
pub(crate) struct ManagementCredential(BearerToken);

impl ManagementCredential {
    pub(crate) fn generate() -> Result<Self> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|error| {
            anyhow!("OS random source unavailable for management credential: {error}")
        })?;
        let mut hex = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(Self(BearerToken::new(hex)?))
    }

    /// The secret, for the owner-only pid record.
    pub(crate) fn expose(&self) -> &str {
        self.0.as_str()
    }

    /// Constant-time comparison against a presented credential.
    fn matches(&self, presented: &str) -> bool {
        self.0.matches(presented)
    }
}

impl std::fmt::Debug for ManagementCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ManagementCredential(<redacted>)")
    }
}

/// Accept loop. Stops accepting (and releases the port) when `shutdown` is
/// cancelled, whether by a signal or by an authenticated shutdown request.
pub(crate) async fn serve_management(
    listener: TcpListener,
    credential: Arc<ManagementCredential>,
    shutdown: CancellationToken,
) {
    let slots = Arc::new(Semaphore::new(MAX_CONNECTIONS));
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
                tokio::spawn(management_connection(
                    stream,
                    Arc::clone(&credential),
                    shutdown.clone(),
                    slot,
                ));
            }
            Err(error) => {
                eprintln!("[hipfire] management accept failed: {error}");
                tokio::select! {
                    _ = shutdown.cancelled() => break,
                    _ = tokio::time::sleep(ACCEPT_BACKOFF) => {}
                }
            }
        }
    }
}

/// One connection, one request. The shutdown token is cancelled only after the
/// connection has finished (response written and the socket closed) or hit
/// its timeout, so the caller receives its `200` before the process begins to
/// exit.
async fn management_connection(
    stream: TcpStream,
    credential: Arc<ManagementCredential>,
    shutdown: CancellationToken,
    _slot: OwnedSemaphorePermit,
) {
    let requested = Arc::new(AtomicBool::new(false));
    let service_requested = Arc::clone(&requested);
    let service = hyper::service::service_fn(move |req: Request<Incoming>| {
        let response = route(&req, &credential, &service_requested);
        async move { Ok::<_, Infallible>(response) }
    });
    let mut builder = http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_READ_TIMEOUT)
        .keep_alive(false)
        .max_buf_size(MAX_HEAD_BYTES);
    let conn = builder.serve_connection(TokioIo::new(stream), service);
    let _ = tokio::time::timeout(CONNECTION_TIMEOUT, conn).await;
    if requested.load(Ordering::Acquire) {
        eprintln!("[hipfire] shutting down: management request (finishing requests in flight)");
        shutdown.cancel();
    }
}

/// Authenticate first, so an unauthenticated peer cannot distinguish routes;
/// then match the single route exactly.
fn route<B>(
    req: &Request<B>,
    credential: &ManagementCredential,
    requested: &AtomicBool,
) -> Response<Full<Bytes>> {
    if !authorized(req.headers(), credential) {
        return reply(StatusCode::UNAUTHORIZED, "unauthorized", true);
    }
    if req.uri().path() != SHUTDOWN_PATH {
        return reply(StatusCode::NOT_FOUND, "not found", false);
    }
    if req.method() != Method::POST {
        let mut response = reply(StatusCode::METHOD_NOT_ALLOWED, "method not allowed", false);
        response
            .headers_mut()
            .insert(header::ALLOW, header::HeaderValue::from_static("POST"));
        return response;
    }
    requested.store(true, Ordering::Release);
    reply(StatusCode::OK, "shutting down", false)
}

/// Exactly one `Authorization: Bearer <credential>` header.
fn authorized(headers: &HeaderMap, credential: &ManagementCredential) -> bool {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    let (Some(value), None) = (values.next(), values.next()) else {
        return false;
    };
    value
        .to_str()
        .ok()
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|presented| credential.matches(presented))
}

fn reply(status: StatusCode, message: &str, challenge: bool) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "message": message }).to_string();
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONNECTION, "close")
        .header(header::CACHE_CONTROL, "no-store");
    if challenge {
        builder = builder.header(header::WWW_AUTHENTICATE, "Bearer");
    }
    builder
        .body(Full::new(Bytes::from(body)))
        .expect("static management response headers are valid")
}

/// Ask the serve process listening on loopback `port` to shut down.
///
/// Used by `hipfire stop` where signals cannot reach a detached serve
/// (Windows). Environment proxies are disabled so the credential only ever goes
/// to the loopback socket; the error text never contains the credential.
#[cfg(any(windows, test))]
pub(crate) fn request_shutdown(port: u16, credential: &str) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .proxy(None)
        .timeout_global(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .build()
        .into();
    let response = agent
        .post(&format!("http://127.0.0.1:{port}{SHUTDOWN_PATH}"))
        .header("Authorization", format!("Bearer {credential}"))
        .send_empty()
        .map_err(|error| anyhow!("management request to 127.0.0.1:{port} failed: {error}"))?;
    match response.status().as_u16() {
        200 => Ok(()),
        401 => Err(anyhow!(
            "serve rejected the management credential in serve.pid (stale record?)"
        )),
        status => Err(anyhow!(
            "serve management endpoint answered unexpected status {status}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn credential() -> ManagementCredential {
        ManagementCredential(BearerToken::new("c0ffee".repeat(10)).unwrap())
    }

    fn request(method: &str, path: &str, authorizations: &[&str]) -> Request<()> {
        let mut builder = Request::builder().method(method).uri(path);
        for value in authorizations {
            builder = builder.header(header::AUTHORIZATION, *value);
        }
        builder.body(()).unwrap()
    }

    #[test]
    fn generated_credentials_are_64_hex_chars_and_unique() {
        let a = ManagementCredential::generate().unwrap();
        let b = ManagementCredential::generate().unwrap();
        assert_eq!(a.expose().len(), 64);
        assert!(a.expose().bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(a.expose(), b.expose());
    }

    #[test]
    fn credential_debug_is_redacted() {
        let credential = credential();
        let rendered = format!("{credential:?} {:#?}", Some(&credential));
        assert!(!rendered.contains(credential.expose()));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn credential_match_is_exact() {
        let credential = credential();
        let token = "c0ffee".repeat(10);
        assert!(credential.matches(&token));
        assert!(!credential.matches(&format!("{token}0")));
        assert!(!credential.matches(&token[..token.len() - 1]));
        assert!(!credential.matches(&format!("{}f", &token[..token.len() - 1])));
        assert!(!credential.matches(""));
    }

    #[test]
    fn route_rejects_every_unauthenticated_shape_with_401() {
        let credential = credential();
        let good = format!("Bearer {}", credential.expose());
        let wrong = format!("Bearer {}x", credential.expose());
        let raw = credential.expose().to_owned();
        let lower = format!("bearer {}", credential.expose());
        let cases: Vec<Vec<&str>> = vec![
            vec![],
            vec![wrong.as_str()],
            vec![raw.as_str()],
            vec![lower.as_str()],
            vec!["Bearer "],
            vec![good.as_str(), good.as_str()],
            vec![good.as_str(), wrong.as_str()],
        ];
        for headers in cases {
            let requested = AtomicBool::new(false);
            let response = route(
                &request("POST", SHUTDOWN_PATH, &headers),
                &credential,
                &requested,
            );
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{headers:?}");
            assert!(!requested.load(Ordering::SeqCst), "{headers:?}");
        }
        // Unauthenticated peers get 401 even for unknown routes and methods.
        let requested = AtomicBool::new(false);
        let response = route(&request("GET", "/health", &[]), &credential, &requested);
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn route_authorized_accepts_only_post_shutdown() {
        let credential = credential();
        let good = format!("Bearer {}", credential.expose());
        let auth = [good.as_str()];

        let requested = AtomicBool::new(false);
        let response = route(
            &request("GET", SHUTDOWN_PATH, &auth),
            &credential,
            &requested,
        );
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(response.headers()[header::ALLOW], "POST");
        assert!(!requested.load(Ordering::SeqCst));

        for path in ["/health", "/v1/chat/completions", "/api/v1/shutdown/", "/"] {
            let response = route(&request("POST", path, &auth), &credential, &requested);
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
            assert!(!requested.load(Ordering::SeqCst), "{path}");
        }

        let response = route(
            &request("POST", SHUTDOWN_PATH, &auth),
            &credential,
            &requested,
        );
        assert_eq!(response.status(), StatusCode::OK);
        assert!(requested.load(Ordering::SeqCst));
    }

    struct Running {
        port: u16,
        credential: Arc<ManagementCredential>,
        shutdown: CancellationToken,
        accept_loop: tokio::task::JoinHandle<()>,
    }

    async fn start() -> Running {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let credential = Arc::new(ManagementCredential::generate().unwrap());
        let shutdown = CancellationToken::new();
        let accept_loop = tokio::spawn(serve_management(
            listener,
            Arc::clone(&credential),
            shutdown.clone(),
        ));
        Running {
            port,
            credential,
            shutdown,
            accept_loop,
        }
    }

    /// Send one raw request; returns the full response (the server closes).
    async fn exchange(port: u16, method: &str, path: &str, authorization: Option<&str>) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let mut head = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n");
        if let Some(value) = authorization {
            head.push_str(&format!("Authorization: {value}\r\n"));
        }
        head.push_str("Content-Length: 0\r\n\r\n");
        stream.write_all(head.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .expect("management response before timeout")
            .unwrap();
        String::from_utf8(response).unwrap()
    }

    #[tokio::test]
    async fn real_http_rejects_unauthorized_without_cancelling() {
        let running = start().await;
        let wrong = format!("Bearer {}", "0".repeat(64));
        for authorization in [None, Some(wrong.as_str()), Some("Bearer ")] {
            let response = exchange(running.port, "POST", SHUTDOWN_PATH, authorization).await;
            assert!(response.starts_with("HTTP/1.1 401"), "{response}");
            assert!(
                !response.contains(running.credential.expose()),
                "response must not echo the credential"
            );
        }
        let good = format!("Bearer {}", running.credential.expose());
        let response = exchange(running.port, "GET", SHUTDOWN_PATH, Some(&good)).await;
        assert!(response.starts_with("HTTP/1.1 405"), "{response}");
        let response = exchange(running.port, "POST", "/health", Some(&good)).await;
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!running.shutdown.is_cancelled());
        running.shutdown.cancel();
    }

    #[tokio::test]
    async fn real_http_authorized_shutdown_delivers_response_then_cancels() {
        let running = start().await;
        let good = format!("Bearer {}", running.credential.expose());
        let response = exchange(running.port, "POST", SHUTDOWN_PATH, Some(&good)).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("shutting down"), "{response}");
        tokio::time::timeout(Duration::from_secs(5), running.shutdown.cancelled())
            .await
            .expect("authorized request cancels the shared shutdown token");
    }

    #[tokio::test]
    async fn client_helper_shuts_down_with_the_credential_and_hides_it_on_failure() {
        let running = start().await;
        let port = running.port;
        let secret = running.credential.expose().to_owned();
        let wrong = "1".repeat(64);
        let error = tokio::task::spawn_blocking(move || request_shutdown(port, &wrong))
            .await
            .unwrap()
            .expect_err("wrong credential is refused");
        assert!(error.to_string().contains("rejected"), "{error}");
        assert!(!format!("{error:#}").contains(&secret));
        assert!(!running.shutdown.is_cancelled());

        tokio::task::spawn_blocking(move || request_shutdown(port, &secret))
            .await
            .unwrap()
            .expect("credential accepted");
        tokio::time::timeout(Duration::from_secs(5), running.shutdown.cancelled())
            .await
            .expect("client helper cancels the shared shutdown token");
    }

    /// `serve_management` owns its listener, so returning closes it. Waiting
    /// for the task, not connecting until refused: once released, the port
    /// can be rebound at once by a concurrent test's one-shot server, which
    /// such a probe would then consume.
    #[tokio::test]
    async fn accept_loop_stops_when_shutdown_is_cancelled() {
        let running = start().await;
        running.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), running.accept_loop)
            .await
            .expect("accept loop returns, closing its listener, after shutdown")
            .unwrap();
    }
}
