// SPDX-License-Identifier: Apache-2.0

//! Hugging Face download counting for `hipfire pull`.
//!
//! The Hub counts a model download server-side when a repo's query file is
//! requested (`GET` or `HEAD`); with no registered library the query files are
//! `config.json`, `config.yaml`, `hyperparams.yaml`, `params.json` and
//! `meta.yaml` (<https://huggingface.co/docs/hub/models-download-stats>).
//! hipfire's artifacts are `.mq4`/`.hfq` files, which the Hub never counts, so
//! after a pull has downloaded the base artifact it fetches the repo's tiny
//! `config.json` once. That is the same `resolve/main/<file>` request, with the
//! same endpoint and credentials, as the download itself; nothing else is sent
//! and nothing is recorded locally. A failure (missing file, offline, timeout)
//! is ignored: it never affects the pull.

use std::io::{self, Read};
use std::time::Duration;

/// The repo file whose fetch the Hub counts as one download.
pub(crate) const COUNT_FILE: &str = "config.json";
/// Upper bound on the whole count request; a pull never waits longer for it.
const COUNT_TIMEOUT: Duration = Duration::from_secs(5);
/// The count file is a few bytes; never read more than this of the body.
const COUNT_BODY_LIMIT: u64 = 64 * 1024;

#[cfg(test)]
thread_local! {
    /// Pull tests share a fixed-count HTTP fixture; the count request is off
    /// there unless a test turns it on.
    pub(crate) static ENABLED_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Count requests attempted on this thread.
    pub(crate) static ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Called once by a pull that downloaded the base artifact of `repo`.
pub(crate) fn count_download(repo: &str) {
    #[cfg(test)]
    if !ENABLED_IN_TESTS.with(|enabled| enabled.get()) {
        return;
    }
    let _ = request_count_file(repo);
}

/// One `GET <endpoint>/<repo>/resolve/main/config.json`. Returns whether the
/// Hub answered with success; every error is swallowed by the caller.
pub(crate) fn request_count_file(repo: &str) -> bool {
    #[cfg(test)]
    ATTEMPTS.with(|count| count.set(count.get() + 1));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(COUNT_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    let mut request = agent.get(&super::artifact_url_in(repo, COUNT_FILE));
    if let Ok(Some(authorization)) = hipfire_config::http_auth::hf_authorization() {
        request = request.header("Authorization", &authorization);
    }
    let Ok(mut response) = request.call() else {
        return false;
    };
    let _ = io::copy(
        &mut response.body_mut().as_reader().take(COUNT_BODY_LIMIT),
        &mut io::sink(),
    );
    response.status().is_success()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// Serves exactly `connections` requests from `files` (keyed by the request
    /// path, 404 otherwise) and returns each request line in order.
    pub(crate) fn recording_server(
        files: HashMap<String, Vec<u8>>,
        connections: usize,
    ) -> (String, JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for stream in listener.incoming().take(connections) {
                let mut stream = stream.unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut buf = [0_u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let line = String::from_utf8_lossy(&buf[..n])
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_owned();
                let path = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_owned();
                let response = match files.get(&path) {
                    Some(body) => {
                        let mut out = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        out.extend_from_slice(body);
                        out
                    }
                    None => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_vec()
                    }
                };
                let _ = stream.write_all(&response);
                let _ = stream.flush();
                seen.push(line);
            }
            seen
        });
        (base, handle)
    }

    #[test]
    fn count_request_is_one_get_of_the_repo_config_json() {
        let _env_lock = crate::tests::TEST_ENV_LOCK.lock().unwrap();
        let files = HashMap::from([(
            "/hipfire-models/qwen3.8-27b/resolve/main/config.json".to_owned(),
            br#"{"library_name":"hipfire"}"#.to_vec(),
        )]);
        let (base, server) = recording_server(files, 1);
        let _hf = crate::tests::EnvGuard::set("HIPFIRE_HF_BASE", &base);
        assert!(request_count_file("hipfire-models/qwen3.8-27b"));
        assert_eq!(
            server.join().unwrap(),
            ["GET /hipfire-models/qwen3.8-27b/resolve/main/config.json HTTP/1.1"]
        );
    }

    #[test]
    fn count_request_failures_are_reported_not_raised() {
        let _env_lock = crate::tests::TEST_ENV_LOCK.lock().unwrap();
        // The repo has no config.json yet: 404.
        let (base, server) = recording_server(HashMap::new(), 1);
        let _hf = crate::tests::EnvGuard::set("HIPFIRE_HF_BASE", &base);
        assert!(!request_count_file("hipfire-models/no-count-file"));
        assert_eq!(server.join().unwrap().len(), 1);
        // Nothing listening: the request fails fast and quietly.
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = format!("http://127.0.0.1:{}", closed.local_addr().unwrap().port());
        drop(closed);
        let _hf = crate::tests::EnvGuard::set("HIPFIRE_HF_BASE", &dead);
        let started = std::time::Instant::now();
        assert!(!request_count_file("hipfire-models/qwen3.8-27b"));
        assert!(started.elapsed() <= COUNT_TIMEOUT);
    }
}
