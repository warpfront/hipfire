#![cfg(unix)]
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    process::{Command, Output},
    thread,
};

fn root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "hipfire-stats-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}
fn cli(root: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_hipfire"));
    c.env("HIPFIRE_HOME", root)
        .env("HIPFIRE_NO_REGISTRY_FETCH", "1")
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1");
    c.env_remove("HIPFIRE_LOCAL");
    c
}
fn footer(output: &Output) -> Vec<&str> {
    std::str::from_utf8(&output.stderr)
        .unwrap()
        .lines()
        .filter(|s| s.starts_with("[stats]"))
        .collect()
}
#[test]
fn fake_local_daemon_stats_and_json() {
    use std::os::unix::fs::PermissionsExt;
    let root = root("local");
    let daemon = root.join("daemon.py");
    fs::write(&daemon, include_str!("../src/serve/fake_daemon.py")).unwrap();
    fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
    let model = root.join("fixture.mq4");
    fs::write(&model, b"fixture").unwrap();
    for flags in [
        vec![],
        vec!["--stats"],
        vec!["--stats", "--no-stream"],
        vec!["--stats", "--json"],
    ] {
        let output = cli(&root)
            .env("HIPFIRE_LOCAL", "1")
            .env("HIPFIRE_DAEMON_BIN", &daemon)
            .arg("run")
            .arg(&model)
            .arg("hello")
            .args(&flags)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if flags.contains(&"--json") {
            let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["tokens"], 4);
            assert!(footer(&output).is_empty());
        } else {
            assert_eq!(output.stdout, b"hello from fake daemon\n");
            if flags.contains(&"--stats") {
                assert_eq!(
                    footer(&output),
                    ["[stats] tokens=4 decode=n/a tok/s ttft=n/a ms"]
                );
            } else {
                assert!(footer(&output).is_empty());
            }
        }
    }
    let output = cli(&root)
        .env("HIPFIRE_LOCAL", "1")
        .env("HIPFIRE_DAEMON_BIN", &daemon)
        .arg("run")
        .arg(&model)
        .args(["t15-class-cancel", "--stats"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(footer(&output).is_empty());
    let metric_daemon = root.join("metric-daemon.py");
    let script = include_str!("../src/serve/fake_daemon.py").replace(
        "\"tok_s\": 12.0,",
        "\"tok_s\": 12.0, \"decode_tok_s\":123.3, \"ttft_ms\":42.1, \"tau\":7.15, \"dflash\":True,",
    );
    fs::write(&metric_daemon, script).unwrap();
    fs::set_permissions(&metric_daemon, fs::Permissions::from_mode(0o755)).unwrap();
    for flags in [vec!["--stats"], vec!["--stats", "--no-stream"]] {
        let output = cli(&root)
            .env("HIPFIRE_LOCAL", "1")
            .env("HIPFIRE_DAEMON_BIN", &metric_daemon)
            .arg("run")
            .arg(&model)
            .arg("hello")
            .args(flags)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, b"hello from fake daemon\n");
        assert_eq!(
            footer(&output),
            ["[stats] tokens=4 decode=123.30 tok/s ttft=42.1 ms tau=7.15"]
        );
    }
    fs::remove_dir_all(root).unwrap();
}

fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = stream.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let len = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|s| s.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + len {
                break;
            }
        }
    }
    String::from_utf8(bytes).unwrap()
}
#[test]
fn http_sse_usage_after_finish_and_nonstream_stats() {
    for (flags, mode) in [
        (vec!["--stats"], "ok"),
        (vec![], "ok"),
        (vec!["--stats", "--no-stream"], "ok"),
        (vec!["--stats", "--json"], "ok"),
        (vec!["--stats"], "missing"),
        (vec!["--stats"], "ar"),
        (vec!["--stats"], "error"),
        (vec!["--stats"], "eof"),
    ] {
        let root = root("http");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let setup = cli(&root)
            .args(["config", "set", "serve.port", &port.to_string()])
            .output()
            .unwrap();
        assert!(
            setup.status.success(),
            "{}",
            String::from_utf8_lossy(&setup.stderr)
        );
        let model = root.join("fixture.mq4");
        fs::write(&model, b"fixture").unwrap();
        let flags_server = flags.clone();
        let server = thread::spawn(move || loop {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream);
            if !request.starts_with("POST ") {
                let body = "{\"status\":\"ok\"}";
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                continue;
            }
            let streaming =
                !flags_server.contains(&"--no-stream") && !flags_server.contains(&"--json");
            let body = if mode == "error" {
                "data: {\"error\":{\"message\":\"cancelled\"}}\n\n"
            } else if mode == "eof" {
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"
            } else if mode == "missing" {
                concat!("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"tok_s\":99,\"decode_tok_s\":null,\"ttft_ms\":-1}}\n\n",
                        "data: [DONE]\n\n")
            } else if mode == "ar" {
                concat!("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"decode_tok_s\":123.3,\"ttft_ms\":42.1,\"tau\":1,\"dflash\":false,\"mtp\":false}}\n\n",
                        "data: {\"choices\":[],\"usage\":{\"completion_tokens\":256}}\n\n", "data: [DONE]\n\n")
            } else if streaming {
                concat!("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"decode_tok_s\":123.3,\"ttft_ms\":42.1,\"tau\":7.15}}\n\n",
                    "data: {\"choices\":[],\"usage\":{\"completion_tokens\":256}}\n\n", "data: [DONE]\n\n")
            } else {
                "{\"choices\":[{\"message\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":256},\"timings\":{\"decode_tok_s\":123.3,\"ttft_ms\":42.1,\"tau\":7.15},\"hipfire\":{\"tok_s\":2.0}}"
            };
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", if streaming {"text/event-stream"} else {"application/json"}, body.len(), body).unwrap();
            return request;
        });
        let output = cli(&root)
            .arg("run")
            .arg(model)
            .arg("hello")
            .args(&flags)
            .output()
            .unwrap();
        let request = server.join().unwrap();
        if mode == "error" || mode == "eof" {
            assert!(!output.status.success());
            assert!(footer(&output).is_empty());
            fs::remove_dir_all(root).unwrap();
            continue;
        }
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if flags.contains(&"--json") {
            let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(value["tok_s"], 2.0);
            assert!(footer(&output).is_empty());
        } else {
            assert_eq!(output.stdout, b"hello\n");
            if flags.contains(&"--stats") {
                let expected = match mode {
                    "missing" => "[stats] tokens=n/a decode=n/a tok/s ttft=n/a ms",
                    "ar" => "[stats] tokens=256 decode=123.30 tok/s ttft=42.1 ms",
                    _ => "[stats] tokens=256 decode=123.30 tok/s ttft=42.1 ms tau=7.15",
                };
                assert_eq!(footer(&output), [expected]);
            } else {
                assert!(footer(&output).is_empty());
            }
        }
        if !flags.contains(&"--no-stream") && !flags.contains(&"--json") {
            assert_eq!(
                request.contains("include_usage"),
                flags.contains(&"--stats")
            );
        }
        fs::remove_dir_all(root).unwrap();
    }
}
