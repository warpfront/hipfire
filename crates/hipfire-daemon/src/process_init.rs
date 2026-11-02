// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Process-start hooks that run before any config, protocol or GPU work:
//! opt-in structured logging and the one-line GPU memory fault hint.

/// Run once, first thing in `main`: structured logging, then the fault hint
/// (registered before any HIP call so the first error 700 is observed).
pub(crate) fn init() {
    init_tracing();
    hip_bridge::register_first_illegal_address_hook(report_gpu_memory_fault);
}

/// Install opt-in structured diagnostics on stderr. Stdout is reserved for the
/// daemon's JSON-lines IPC protocol and must never receive tracing output.
///
/// `HIPFIRE_LOG` accepts an EnvFilter directive such as `info` or
/// `hipfire_runtime=debug`; `RUST_LOG` is the fallback. Set
/// `HIPFIRE_LOG_FORMAT=json` for machine-readable log events. With neither
/// filter set, tracing remains off and the existing operator-facing stderr
/// messages are unchanged.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_env("HIPFIRE_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("off"));
    // Ambient env on purpose: this runs before the CLI has sent the process
    // config, and a developer_var read here would install the local fallback
    // snapshot and make the real install_process_config fail.
    let json = std::env::var("HIPFIRE_LOG_FORMAT")
        .map(|value| value.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    let result = if json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .json()
            .try_init()
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_target(false)
            .try_init()
    };

    if let Err(error) = result {
        eprintln!("warning: failed to initialize structured logging: {error}");
    }
}

/// One-line report hint for a GPU memory fault (HIP error 700). Static so the
/// hook does no allocation or formatting while the process is faulting.
const GPU_MEMORY_FAULT_HINT: &str = "hipfire: GPU memory fault (HIP error 700). To report it, capture: sudo dmesg | grep -iE 'page fault|SQC|UTCL2|gfxhub|amdgpu'; uname -r; ROCm/HIP version (/opt/rocm/.info/version or hipconfig --version); MES firmware (cat /sys/class/drm/card*/device/fw_version/mes_fw_version). See docs/KNOWN-ISSUES.md.";

/// hip-bridge fires this once, on the first raw HIP error 700 it observes
/// (before the error unwinds), whichever load or request triggered it. The
/// bridge keeps hooks registered by other layers rather than replacing them.
fn report_gpu_memory_fault() {
    eprintln!("{GPU_MEMORY_FAULT_HINT}");
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu_memory_fault_hint_is_one_line_with_capture_commands() {
        let hint = super::GPU_MEMORY_FAULT_HINT;
        assert!(!hint.contains('\n'));
        assert!(hint.starts_with("hipfire: GPU memory fault (HIP error 700)."));
        for needle in [
            "sudo dmesg | grep -iE 'page fault|SQC|UTCL2|gfxhub|amdgpu'",
            "uname -r",
            "/opt/rocm/.info/version",
            "hipconfig --version",
            "mes_fw_version",
            "docs/KNOWN-ISSUES.md",
        ] {
            assert!(hint.contains(needle), "missing {needle}");
        }
    }
}
