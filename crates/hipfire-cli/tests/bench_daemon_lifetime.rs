//! `hipfire bench` owns the daemon it spawns: whatever stops the bench (a
//! SIGTERM from a timeout harness, or a failed load) must kill and reap that
//! daemon before the CLI exits. The daemon holds the machine-wide per-GPU
//! `flock`, so a daemon that outlives its CLI makes the next arm on the same
//! cards fail with "already reserved by holder PID …" (the corrective-hold
//! flash-pp3 failure). Fake daemon only; no GPU, CPU-only.
#![cfg(unix)]
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

/// Serialize: the tests write then exec scripts, and a fork in a sibling test
/// thread can otherwise inherit a script's write fd (ETXTBSY on exec).
static SERIAL: Mutex<()> = Mutex::new(());

fn root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "hipfire-bench-lifetime-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

/// A daemon that answers the bench handshake (`configure`, `ping`, `diag`),
/// records its pid, then on `load` either hangs forever (a cold multi-GPU load
/// that never answers) or reports an error — and in both cases ignores SIGTERM
/// and never exits on its own, so only an explicit kill by its owner stops it.
fn fake_daemon(root: &Path, pid_file: &Path, on_load: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = format!(
        r#"#!/usr/bin/env python3
import json, os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
def out(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    kind = req.get("type")
    if kind == "configure":
        out({{"type": "configured"}})
    elif kind == "ping":
        out({{"type": "pong"}})
    elif kind == "diag":
        out({{"type": "diag", "arch": "gfx1201", "vram_free_mb": 1}})
    elif kind == "load":
        with open({pid_file:?}, "w") as f:
            f.write(str(os.getpid()))
        if {on_load:?} == "error":
            out({{"type": "error", "message": "synthetic load failure"}})
        while True:
            time.sleep(1)
"#,
        pid_file = pid_file.display().to_string(),
        on_load = on_load,
    );
    let daemon = root.join(format!("daemon-{on_load}.py"));
    fs::write(&daemon, script).unwrap();
    fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
    daemon
}

fn bench_cli(root: &Path, daemon: &Path, model: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_hipfire"))
        .env("HIPFIRE_HOME", root)
        .env("HIPFIRE_NO_REGISTRY_FETCH", "1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .env("HIPFIRE_LOCAL", "1")
        .env("HIPFIRE_DAEMON_BIN", daemon)
        .arg("bench")
        .arg(model)
        .args(["--runs", "1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn wait_for_pid_file(pid_file: &Path, cli: &mut Child) -> libc::pid_t {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(pid) = fs::read_to_string(pid_file)
            .ok()
            .and_then(|text| text.trim().parse::<libc::pid_t>().ok())
        {
            return pid;
        }
        if let Some(status) = cli.try_wait().unwrap() {
            panic!("bench exited ({status}) before the daemon reached load");
        }
        assert!(
            Instant::now() < deadline,
            "daemon never reached load within 60 s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_exit(cli: &mut Child, what: &str) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = cli.try_wait().unwrap() {
            return status;
        }
        assert!(Instant::now() < deadline, "bench did not exit: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// True while `pid` names a process (a zombie counts: it is not reaped).
fn alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only probes for existence.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Kill a daemon the code under test failed to reap, so a failing run does not
/// leak a SIGTERM-immune process.
fn force_kill(pid: libc::pid_t) {
    // SAFETY: the pid was written by the fake daemon this test started.
    unsafe { libc::kill(pid, libc::SIGKILL) };
}

fn fixture_model(root: &Path) -> PathBuf {
    let model = root.join("fixture.mq4");
    fs::write(&model, b"fixture").unwrap();
    model
}

#[test]
fn sigterm_while_the_daemon_loads_kills_and_reaps_the_owned_daemon() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = root("sigterm");
    let pid_file = root.join("daemon.pid");
    let daemon = fake_daemon(&root, &pid_file, "hang");
    let model = fixture_model(&root);
    let mut cli = bench_cli(&root, &daemon, &model);
    let daemon_pid = wait_for_pid_file(&pid_file, &mut cli);
    assert!(alive(daemon_pid), "daemon is running while loading");

    // The harness-timeout case: SIGTERM reaches only the CLI, and the daemon
    // would ignore it anyway.
    // SAFETY: `cli.id()` is the live child this test spawned.
    assert_eq!(
        unsafe { libc::kill(cli.id() as libc::pid_t, libc::SIGTERM) },
        0
    );
    let status = wait_for_exit(&mut cli, "SIGTERM while loading");
    let leaked = alive(daemon_pid);
    if leaked {
        force_kill(daemon_pid);
    }
    assert!(
        !leaked,
        "daemon {daemon_pid} outlived the interrupted bench and would keep its GPU lock"
    );
    assert_eq!(
        status.code(),
        Some(130),
        "interrupt exits through the handler, not by dying on the signal: {status}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn failed_load_kills_and_reaps_the_owned_daemon() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = root("load-error");
    let pid_file = root.join("daemon.pid");
    let daemon = fake_daemon(&root, &pid_file, "error");
    let model = fixture_model(&root);
    let mut cli = bench_cli(&root, &daemon, &model);
    let status = wait_for_exit(&mut cli, "daemon load error");
    assert!(!status.success(), "a failed load fails the bench");
    let daemon_pid: libc::pid_t = fs::read_to_string(&pid_file)
        .expect("daemon reached load")
        .trim()
        .parse()
        .unwrap();
    let leaked = alive(daemon_pid);
    if leaked {
        force_kill(daemon_pid);
    }
    assert!(
        !leaked,
        "daemon {daemon_pid} outlived the failed bench and would keep its GPU lock"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn interrupt_leaves_an_unrelated_daemon_alone() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = root("unrelated");
    let pid_file = root.join("daemon.pid");
    let daemon = fake_daemon(&root, &pid_file, "hang");
    let model = fixture_model(&root);
    // A daemon owned by someone else (another bench, serve): same binary, but
    // not spawned by the CLI under test.
    let mut bystander = Command::new(&daemon)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let bystander_pid = bystander.id() as libc::pid_t;
    let mut cli = bench_cli(&root, &daemon, &model);
    let owned_pid = wait_for_pid_file(&pid_file, &mut cli);
    assert_ne!(owned_pid, bystander_pid);

    // SAFETY: `cli.id()` is the live child this test spawned.
    assert_eq!(
        unsafe { libc::kill(cli.id() as libc::pid_t, libc::SIGTERM) },
        0
    );
    wait_for_exit(&mut cli, "SIGTERM with a bystander daemon");
    let bystander_alive = alive(bystander_pid);
    let owned_alive = alive(owned_pid);
    let _ = bystander.kill();
    let _ = bystander.wait();
    if owned_alive {
        force_kill(owned_pid);
    }
    assert!(bystander_alive, "an unrelated daemon must not be killed");
    assert!(!owned_alive, "the owned daemon must be killed");
    let _ = fs::remove_dir_all(root);
}
