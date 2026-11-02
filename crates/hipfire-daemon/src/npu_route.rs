// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Daemon surfaces of `hipfire doctor --npu`:
//!
//! - `--version`: `hipfire-daemon <version> <full build commit>`, answered
//!   before configuration, the protocol loop or any device probe.
//! - The diagnostic control message (Linux only; not a config or env knob):
//!   `→ {"type":"npu_trunk_route","route":"gpu"}` (`"gpu"` | `"auto"`)
//!   `← {"type":"npu_trunk_route","route":"gpu"}`.
//!   `gpu` routes the requests that follow through the GPU-only schedule,
//!   `auto` through the configured one; no model or GPU state is touched here.

use hipfire_engine::emit::emit_uncorrelated_error;
use std::io::Write;

/// `--version` output: `hipfire-daemon <version> <full build commit>`.
fn version_line() -> String {
    format!(
        "hipfire-daemon {} {}",
        env!("CARGO_PKG_VERSION"),
        env!("HIPFIRE_BUILD_COMMIT")
    )
}

fn version_requested(args: &[String]) -> bool {
    args.iter().skip(1).any(|arg| arg == "--version")
}

/// The process arguments. With `--version`, print the version line and exit 0.
pub(crate) fn args_or_exit_on_version() -> Vec<String> {
    let args: Vec<String> = std::env::args().collect();
    if version_requested(&args) {
        println!("{}", version_line());
        std::process::exit(0);
    }
    args
}

/// Parse an `npu_trunk_route` request: `Ok(true)` = GPU-only diagnostic
/// route, `Ok(false)` = auto. Anything else is a clear validation error.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse(msg: &serde_json::Value) -> Result<bool, String> {
    match msg.get("route").and_then(|route| route.as_str()) {
        Some("gpu") => Ok(true),
        Some("auto") => Ok(false),
        _ => Err("npu_trunk_route: \"route\" must be the string \"gpu\" or \"auto\"".to_owned()),
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn reply(gpu_route: bool) -> serde_json::Value {
    serde_json::json!({
        "type": "npu_trunk_route",
        "route": if gpu_route { "gpu" } else { "auto" },
    })
}

/// Answer one `npu_trunk_route` message.
pub(crate) fn handle(msg: &serde_json::Value, stdout: &mut impl Write) {
    #[cfg(target_os = "linux")]
    match parse(msg) {
        Ok(gpu_route) => {
            hipfire_loader::set_diagnostic_gpu_route(gpu_route);
            let _ = writeln!(stdout, "{}", reply(gpu_route));
        }
        Err(message) => {
            emit_uncorrelated_error(stdout, None, &message, "validation", false, false);
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = msg;
        emit_uncorrelated_error(
            stdout,
            None,
            "npu_trunk_route: the NPU trunk exists only on Linux",
            "unsupported",
            false,
            false,
        );
    }
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line_is_name_version_full_commit() {
        assert_eq!(
            version_line(),
            format!(
                "hipfire-daemon {} {}",
                env!("CARGO_PKG_VERSION"),
                env!("HIPFIRE_BUILD_COMMIT")
            )
        );
    }

    #[test]
    fn version_flag_only_matches_an_exact_argument() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(version_requested(&args(&["daemon", "--version"])));
        assert!(!version_requested(&args(&["daemon"])));
        assert!(!version_requested(&args(&["daemon", "--versions"])));
    }

    #[test]
    fn npu_trunk_route_accepts_only_gpu_and_auto() {
        let parse = |text: &str| parse(&serde_json::from_str::<serde_json::Value>(text).unwrap());
        assert_eq!(
            parse(r#"{"type":"npu_trunk_route","route":"gpu"}"#),
            Ok(true)
        );
        assert_eq!(
            parse(r#"{"type":"npu_trunk_route","route":"auto"}"#),
            Ok(false)
        );
        for bad in [
            r#"{"type":"npu_trunk_route"}"#,
            r#"{"type":"npu_trunk_route","route":"npu"}"#,
            r#"{"type":"npu_trunk_route","route":"GPU"}"#,
            r#"{"type":"npu_trunk_route","route":true}"#,
            r#"{"type":"npu_trunk_route","route":null}"#,
        ] {
            let message = parse(bad).unwrap_err();
            assert!(message.contains("\"gpu\" or \"auto\""), "{bad}: {message}");
        }
    }

    #[test]
    fn npu_trunk_route_reply_echoes_type_and_route() {
        assert_eq!(
            reply(true).to_string(),
            r#"{"type":"npu_trunk_route","route":"gpu"}"#
        );
        assert_eq!(
            reply(false).to_string(),
            r#"{"type":"npu_trunk_route","route":"auto"}"#
        );
    }
}
