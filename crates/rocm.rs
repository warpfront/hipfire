// SPDX-License-Identifier: Apache-2.0
//! Shared ROCm root and explicit oracle-test opt-out.
#![allow(dead_code)]

pub fn root() -> std::path::PathBuf {
    std::env::var_os("ROCM_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/rocm"))
}

pub fn llvm(tool: &str) -> std::path::PathBuf {
    root().join("lib/llvm/bin").join(tool)
}

/// Compiler major used to key foreign-code oracle fixtures.
pub fn clang_major() -> Result<u32, String> {
    let compiler = llvm("clang");
    let output = std::process::Command::new(&compiler)
        .arg("--version")
        .output()
        .map_err(|e| format!("{}: {e}", compiler.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version failed: {}",
            compiler.display(),
            output.status
        ));
    }
    let version = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    version
        .split("clang version ")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("cannot detect clang major from {version:?}"))
}

pub fn tests_disabled() -> bool {
    std::env::var("HIPFIRE_TEST_REQUIRE_ROCM").as_deref() == Ok("0")
}
