// SPDX-License-Identifier: Apache-2.0
//! Shared ROCm oracle discovery. Only an explicit no-ROCm marker skips tests.
#![allow(dead_code, unused_macros)]
#[path = "../../../rocm.rs"]
mod resolver;
use std::path::PathBuf;
pub fn root() -> PathBuf {
    resolver::root()
}
pub fn llvm(name: &str) -> PathBuf {
    resolver::llvm(name)
}
pub fn hipcc() -> PathBuf {
    root().join("bin/hipcc")
}
pub fn llvm_bin_dir() -> PathBuf {
    root().join("lib/llvm/bin")
}
pub fn silu_golden(gfx1100: bool) -> &'static str {
    match (resolver::clang_major().expect("detecting ROCm oracle toolchain"), gfx1100) {
        (23, false) => peacemaker_isa::kernels::iu4_gemm::region::SILU_GOLDEN,
        (23, true) => peacemaker_isa::kernels::iu4_gemm::region::SILU_GOLDEN_GFX1100,
        (24, false) => include_str!("../../kernels/iu4_gemm.silu.clang24.region.s"),
        (24, true) => include_str!("../../kernels/iu4_v2c.gfx1100.silu.clang24.region.s"),
        (major, _) => panic!("unsupported ROCm clang major {major}: regenerate and commit toolchain-keyed SiLU oracle goldens"),
    }
}
pub fn rocm_tool(name: &str) -> Option<PathBuf> {
    if resolver::tests_disabled() {
        return None;
    }
    let path = llvm(name);
    assert!(
        path.is_file(),
        "required ROCm tool {} missing; set HIPFIRE_TEST_REQUIRE_ROCM=0 only on no-ROCm hosts",
        path.display()
    );
    Some(path)
}
macro_rules! require_rocm_tool {
    ($name:expr) => {
        match $crate::rocm::rocm_tool($name) {
            Some(path) => path,
            None => {
                eprintln!("skipping: explicitly disabled ROCm tool {}", $name);
                return;
            }
        }
    };
}
