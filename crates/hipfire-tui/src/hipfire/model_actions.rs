// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Background runners for Models-tab actions: pull (with live progress) and rm.
//!
//! Both invoke the native `hipfire` binary on a dedicated thread so the UI
//! thread never blocks. `pull` runs non-interactively (`--yes`, stdin closed),
//! streams the CLI's stderr progress line (which is carriage-return-updated, so
//! we split on '\r' AND '\n') and reports a parsed percentage; it can be
//! cancelled, which kills and reaps the child. `remove` runs `rm <tag> --yes`
//! (the TUI does its own y/n confirm first) and reports a single outcome.
//! Events arrive on mpsc channels the App drains each frame.

use std::{
    io::Read,
    process::{Child, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::hipfire::native_cli_command;

/// Streamed events from a `pull` run.
#[derive(Clone, Debug, PartialEq)]
pub enum PullEvent {
    /// A progress update: parsed percent (None if not yet sampled) + the raw
    /// CLI progress line (bar / rate / ETA / bytes) for display.
    Progress {
        percent: Option<f64>,
        line: String,
    },
    Done,
    Failed(String),
    /// The pull was cancelled (the child was killed and reaped).
    Cancelled,
}

/// Outcome of a one-shot model action (currently `remove`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RmOutcome {
    Ok(String),
    Failed(String),
}

/// How often the pull worker polls the child (`try_wait`) and the cancel flag.
const PULL_POLL: Duration = Duration::from_millis(100);

/// Handle to a running `pull` worker: the event stream plus cancellation.
///
/// Dropping the handle cancels a still-running pull and waits (bounded) for the
/// worker to kill and reap the child, so no `hipfire pull` outlives the TUI.
pub struct PullHandle {
    /// Progress / result events from the worker.
    pub rx: Receiver<PullEvent>,
    cancel: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl PullHandle {
    /// A handle with no worker thread, for tests that drive the event channel
    /// by hand. Returns the handle and its cancel flag.
    #[cfg(test)]
    pub fn detached(rx: Receiver<PullEvent>) -> (Self, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        (
            Self {
                rx,
                cancel: Arc::clone(&cancel),
                worker: None,
            },
            cancel,
        )
    }

    /// Ask the worker to kill the child. Non-blocking; the worker answers with
    /// [`PullEvent::Cancelled`].
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Whether [`cancel`](Self::cancel) has been requested.
    pub fn cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Cancel and wait up to `timeout` for the worker to reap the child. A
    /// worker that misses the deadline is detached (it still finishes alone).
    pub fn shutdown(&mut self, timeout: Duration) {
        self.cancel();
        let Some(worker) = self.worker.take() else {
            return;
        };
        let deadline = Instant::now() + timeout;
        while !worker.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if worker.is_finished() {
            let _ = worker.join();
        }
    }
}

impl Drop for PullHandle {
    fn drop(&mut self) {
        self.shutdown(Duration::from_secs(3));
    }
}

/// Spawn native `hipfire pull <tag> --yes <extra_args...>` with stdin closed
/// (so the CLI never starts its interactive tier picker), streaming progress on
/// the returned handle's receiver and finishing with Done, Failed or Cancelled.
pub fn pull(tag: String, extra_args: Vec<String>) -> PullHandle {
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let worker = thread::spawn(move || pull_inner(tag, extra_args, tx, worker_cancel));
    PullHandle {
        rx,
        cancel,
        worker: Some(worker),
    }
}

/// Spawn native `hipfire rm <tag> --yes`; the single outcome arrives on the
/// returned receiver.
pub fn remove(tag: String) -> Receiver<RmOutcome> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(remove_inner(&tag));
    });
    rx
}

/// Extract the percentage from a CLI progress line like
/// `[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB` — find the '%' and
/// read the trailing number before it (robust to the bar's internal spaces).
pub fn parse_percent(line: &str) -> Option<f64> {
    let pct = line.find('%')?;
    let num: String = line[..pct]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    num.parse::<f64>().ok()
}

fn pull_inner(
    tag: String,
    extra_args: Vec<String>,
    tx: Sender<PullEvent>,
    cancel: Arc<AtomicBool>,
) {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            let _ = tx.send(PullEvent::Failed(
                "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)".into(),
            ));
            return;
        }
    };
    // `--yes` + a closed stdin: the CLI starts its interactive tier picker when
    // stdin is a TTY (the child would otherwise inherit the TUI's terminal and
    // swallow keystrokes behind the TUI's back).
    let mut child = match cmd
        .arg("pull")
        .arg(&tag)
        .arg("--yes")
        .args(&extra_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(PullEvent::Failed(format!("spawn: {e}")));
            return;
        }
    };

    // Drain stdout on its own thread so a full pipe can't deadlock the child.
    let stdout_handle = child.stdout.take().map(|mut so| {
        thread::spawn(move || {
            let mut s = String::new();
            let _ = so.read_to_string(&mut s);
            s
        })
    });

    // Read stderr on its own thread so the worker loop can keep polling the
    // child and the cancel flag. Flush a "line" on each '\r' or '\n' (the CLI
    // overwrites one line with '\r' and only emits '\n' at completion) and
    // retain the most-recent non-empty line so a failure reports the real
    // cause — the CLI writes its errors to stderr (the same stream as the
    // progress).
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut last_line = String::new();
            let mut buf: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                match stderr.read(&mut chunk) {
                    Ok(0) | Err(_) => break, // EOF / pipe error
                    Ok(n) => {
                        for &byte in &chunk[..n] {
                            if byte == b'\r' || byte == b'\n' {
                                if let Some(line) = flush_progress(&mut buf, &tx) {
                                    last_line = line;
                                }
                            } else {
                                buf.push(byte);
                            }
                        }
                    }
                }
            }
            if let Some(line) = flush_progress(&mut buf, &tx) {
                last_line = line;
            }
            last_line
        })
    });

    let status = loop {
        if cancel.load(Ordering::SeqCst) {
            kill_and_reap(&mut child);
            let _ = tx.send(PullEvent::Cancelled);
            return;
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => thread::sleep(PULL_POLL),
            Err(e) => break Err(e),
        }
    };

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            kill_and_reap(&mut child);
            let _ = tx.send(PullEvent::Failed(format!("wait: {e}")));
            return;
        }
    };
    // The child has exited, so both pipes hit EOF and the readers finish.
    let last_line = stderr_handle
        .map(|h| h.join().unwrap_or_default())
        .unwrap_or_default();
    let stdout = stdout_handle
        .map(|h| h.join().unwrap_or_default())
        .unwrap_or_default();
    if status.success() {
        let _ = tx.send(PullEvent::Done);
        return;
    }
    // Prefer the last stderr line (where the CLI writes errors); fall back to
    // stdout, then a generic message.
    let msg = if !last_line.is_empty() {
        last_line
    } else {
        stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("pull failed")
            .trim()
            .to_string()
    };
    let _ = tx.send(PullEvent::Failed(msg));
}

/// Kill the child (ignoring "already exited") and reap it so no zombie remains.
fn kill_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Flush the accumulated stderr line: send it as a Progress event and return it
/// (so the caller can keep the latest line for error reporting). None if empty.
fn flush_progress(buf: &mut Vec<u8>, tx: &Sender<PullEvent>) -> Option<String> {
    if buf.is_empty() {
        return None;
    }
    let line = String::from_utf8_lossy(buf).trim().to_string();
    buf.clear();
    if line.is_empty() {
        return None;
    }
    let percent = parse_percent(&line);
    let _ = tx.send(PullEvent::Progress {
        percent,
        line: line.clone(),
    });
    Some(line)
}

fn remove_inner(tag: &str) -> RmOutcome {
    let mut cmd = match native_cli_command() {
        Some(c) => c,
        None => {
            return RmOutcome::Failed(
                "native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)".into(),
            )
        }
    };
    let output = cmd.arg("rm").arg(tag).arg("--yes").output();
    match output {
        Ok(o) if o.status.success() => RmOutcome::Ok(format!("removed {tag}")),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let tail = err
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("rm failed")
                .trim();
            RmOutcome::Failed(format!("rm {tag}: {tail}"))
        }
        Err(e) => RmOutcome::Failed(format!("rm {tag}: spawn failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hipfire::ENV_LOCK;
    use std::{env, fs, os::unix::fs::PermissionsExt, path::Path, path::PathBuf};

    #[test]
    fn parse_percent_handles_real_line() {
        assert_eq!(
            parse_percent("[████▏      ]  45.2%   8.1 MB/s   ETA 1m12s   123/272 MB"),
            Some(45.2)
        );
        assert_eq!(
            parse_percent("[          ]   0.0%   —   123/272 MB"),
            Some(0.0)
        );
        assert_eq!(parse_percent("[██████████] 100.0% done"), Some(100.0));
        // Unknown-total downloads render "?%" — must yield None (drives the
        // PullEvent::Progress { percent: None } path).
        assert_eq!(parse_percent("[          ]   ?%   —   ?/? MB"), None);
        assert_eq!(parse_percent("no percent here"), None);
        // The real CLI's report_progress line (CR-prefixed, no bar).
        assert_eq!(
            parse_percent("45.20%  1.23/2.72 GB  8.1 MB/s  ETA 12s"),
            Some(45.2)
        );
    }

    #[test]
    fn flush_progress_parses_sends_and_returns_line() {
        let (tx, rx) = mpsc::channel();
        let mut buf = b"[bar] 42.0% 8 MB/s".to_vec();
        let line = flush_progress(&mut buf, &tx);
        assert_eq!(line.as_deref(), Some("[bar] 42.0% 8 MB/s"));
        assert!(buf.is_empty(), "buffer is consumed");
        match rx.try_recv() {
            Ok(PullEvent::Progress { percent, line }) => {
                assert_eq!(percent, Some(42.0));
                assert!(line.contains("42.0%"));
            }
            other => panic!("expected a Progress event, got {other:?}"),
        }
        // An empty buffer flushes nothing and emits no event.
        let mut empty = Vec::new();
        assert_eq!(flush_progress(&mut empty, &tx), None);
        assert!(rx.try_recv().is_err());
    }

    /// Fresh scratch dir under the system temp dir, unique per test.
    fn scratch(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("hipfire-pull-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write `body` as an executable `#!/bin/sh` script and point
    /// `HIPFIRE_CLI_BIN` at it. The env var is process-global and App tests
    /// that don't take `ENV_LOCK` can also spawn the "CLI" (e.g. `serve -d`), so
    /// the script ignores anything but `pull` (fails fast, like no binary).
    fn install_fake_cli(dir: &Path, body: &str) {
        let script = dir.join("hipfire");
        fs::write(
            &script,
            format!("#!/bin/sh\n[ \"$1\" = pull ] || exit 1\n{body}\n"),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        env::set_var("HIPFIRE_CLI_BIN", &script);
    }

    /// Collect events until a terminal one (Done / Failed / Cancelled) or timeout.
    fn collect_until_terminal(handle: &PullHandle, timeout: Duration) -> Vec<PullEvent> {
        let deadline = Instant::now() + timeout;
        let mut events = Vec::new();
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match handle.rx.recv_timeout(left) {
                Ok(ev) => {
                    let terminal = !matches!(ev, PullEvent::Progress { .. });
                    events.push(ev);
                    if terminal {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        events
    }

    #[test]
    fn pull_runs_noninteractive_with_yes_and_closed_stdin() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("args");
        let out = dir.join("record.txt");
        install_fake_cli(
            &dir,
            &format!(
                "echo \"$@\" >> '{out}'\n\
                 ( [ -t 0 ] && echo tty || echo notty ) >> '{out}'\n\
                 printf '\\r  45.20%%  0.10/0.22 GB  8.1 MB/s  ETA 1s' >&2\n\
                 exit 0",
                out = out.display()
            ),
        );
        let handle = pull("qwen3.5:4b".into(), Vec::new());
        let events = collect_until_terminal(&handle, Duration::from_secs(10));
        env::remove_var("HIPFIRE_CLI_BIN");

        let record = fs::read_to_string(&out).unwrap();
        let mut lines = record.lines();
        assert_eq!(lines.next(), Some("pull qwen3.5:4b --yes"));
        assert_eq!(lines.next(), Some("notty"), "stdin must not be a terminal");
        assert!(
            events.iter().any(|e| matches!(
                e,
                PullEvent::Progress { percent: Some(p), .. } if (*p - 45.2).abs() < 1e-9
            )),
            "expected Progress(45.2), got {events:?}"
        );
        assert_eq!(events.last(), Some(&PullEvent::Done));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn pull_appends_extra_args_after_yes() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("extra");
        let out = dir.join("record.txt");
        install_fake_cli(&dir, &format!("echo \"$@\" >> '{}'", out.display()));
        let handle = pull("qwen3.5:4b".into(), vec!["--with-draft".into(), "x".into()]);
        let events = collect_until_terminal(&handle, Duration::from_secs(10));
        env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(events.last(), Some(&PullEvent::Done));
        assert_eq!(
            fs::read_to_string(&out).unwrap().trim(),
            "pull qwen3.5:4b --yes --with-draft x"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn pull_failure_reports_last_stderr_line() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("fail");
        install_fake_cli(&dir, "echo 'Error: no such model' >&2\nexit 3");
        let handle = pull("nope".into(), Vec::new());
        let events = collect_until_terminal(&handle, Duration::from_secs(10));
        env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(
            events.last(),
            Some(&PullEvent::Failed("Error: no such model".into()))
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn pull_cancel_kills_and_reaps_child() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("cancel");
        let pidfile = dir.join("pid");
        install_fake_cli(
            &dir,
            &format!(
                "echo $$ > '{}'\nwhile true; do sleep 1; done",
                pidfile.display()
            ),
        );
        let handle = pull("qwen3.5:4b".into(), Vec::new());
        // The worker resolves HIPFIRE_CLI_BIN asynchronously; clear it only
        // once the child is running (below).

        // Wait for the child to record its pid.
        let deadline = Instant::now() + Duration::from_secs(5);
        let pid = loop {
            if let Some(pid) = fs::read_to_string(&pidfile)
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
            {
                break pid;
            }
            assert!(Instant::now() < deadline, "child never wrote its pid");
            thread::sleep(Duration::from_millis(10));
        };
        env::remove_var("HIPFIRE_CLI_BIN");
        assert!(Path::new(&format!("/proc/{pid}")).exists());

        handle.cancel();
        let events = collect_until_terminal(&handle, Duration::from_secs(2));
        assert_eq!(events.last(), Some(&PullEvent::Cancelled));
        assert!(
            !Path::new(&format!("/proc/{pid}")).exists(),
            "child {pid} must be killed and reaped"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dropping_the_handle_cancels_and_reaps_the_child() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = scratch("drop");
        let pidfile = dir.join("pid");
        install_fake_cli(
            &dir,
            &format!(
                "echo $$ > '{}'\nwhile true; do sleep 1; done",
                pidfile.display()
            ),
        );
        let handle = pull("qwen3.5:4b".into(), Vec::new());
        // The worker resolves HIPFIRE_CLI_BIN asynchronously; clear it only
        // once the child is running (below).
        let deadline = Instant::now() + Duration::from_secs(5);
        let pid = loop {
            if let Some(pid) = fs::read_to_string(&pidfile)
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
            {
                break pid;
            }
            assert!(Instant::now() < deadline, "child never wrote its pid");
            thread::sleep(Duration::from_millis(10));
        };
        env::remove_var("HIPFIRE_CLI_BIN");
        drop(handle);
        assert!(
            !Path::new(&format!("/proc/{pid}")).exists(),
            "drop must reap child {pid}"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
