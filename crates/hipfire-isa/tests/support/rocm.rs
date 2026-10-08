// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared ROCm-toolchain discovery for the integration tests that spawn
//! `llvm-mc`, `ld.lld`, `clang-offload-bundler` or `llvm-objdump`. CI runners
//! have no ROCm: a test that needs a tool calls `require_rocm_tool!` first and
//! returns early (with a `skipping:` line on stderr) when the tool is absent.
//! Included by path with `#[macro_use]`:
//! `#[macro_use] #[path = ".../support/rocm.rs"] mod rocm;`.
#![allow(dead_code, unused_macros)]

use std::path::PathBuf;

/// The LLVM `bin` directory of the ROCm install under test; override with
/// `HIPFIRE_TEST_ROCM_LLVM_BIN`.
pub fn llvm_bin_dir() -> PathBuf {
    std::env::var_os("HIPFIRE_TEST_ROCM_LLVM_BIN").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm/core-10.0/lib/llvm/bin"))
}

/// `llvm_bin_dir()/name` when that file exists.
pub fn rocm_tool(name: &str) -> Option<PathBuf> {
    let path = llvm_bin_dir().join(name);
    path.exists().then_some(path)
}

/// Evaluates to the tool's `PathBuf`, or prints `skipping: ROCm tool <name>
/// not present` and `return`s from the enclosing test fn.
macro_rules! require_rocm_tool {
    ($name:expr) => {
        match $crate::rocm::rocm_tool($name) {
            Some(path) => path,
            None => {
                eprintln!("skipping: ROCm tool {} not present", $name);
                return;
            }
        }
    };
}
