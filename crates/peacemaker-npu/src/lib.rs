// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! peacemaker-npu: build AIE2P (XDNA2, Strix / Strix Halo NPU) programs without the vendor toolchain.
//!
//! - [`isa`]: core VLIW instruction encoder, transcribed from llvm-aie's AIE2P TableGen
//!   (`AIE2PGenInstrInfo.td`, `AIE2PCompositeFormats.td`, `AIE2PMCCodeEmitterGen.inc`).
//! - [`regs`]: tile addressing and the register offsets we touch (aie-rt `xaie2pgbl_params.h`).
//! - [`dma`]: tile / shim buffer-descriptor words.
//! - [`cdo`] + [`pdi`]: the configuration image the firmware loads at CONFIG_CU (bootgen format).
//! - [`txn`]: the DPU transaction stream the firmware executes per command (mlir-aie
//!   `TxnEncoding.h` layout, firmware TXN v0.1).
pub mod cdo;
pub mod dma;
pub mod isa;
pub mod kernels;
pub mod pdi;
pub mod regs;
pub mod route;
pub mod sim;
pub mod txn;

/// Optional vendor core-ELF corpus (`AIE2P_VENDOR_CORPUS`, the `vendor-artifacts` directory of the NPU
/// reference checkout; not in this repo). Tests that cross-check against it skip when it is unset or absent.
#[cfg(test)]
pub(crate) fn vendor_corpus() -> Option<std::path::PathBuf> {
    let root = std::path::PathBuf::from(std::env::var_os("AIE2P_VENDOR_CORPUS")?);
    if root.exists() {
        Some(root)
    } else {
        eprintln!("SKIP: vendor corpus absent: {}", root.display());
        None
    }
}
