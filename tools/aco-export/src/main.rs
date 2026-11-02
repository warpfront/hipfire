// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! aco-export: offline developer tool for the PM-ACO parity program (PM-ACO-PARITY.md F1/P0).
//!
//! GLSL source -> SPIR-V (glslang) -> the REAL pinned RADV compile path (SPIR-V -> NIR -> ACO) via a
//! small Vulkan host -> captured final executable words/config/ABI -> `pm-aco-artifact-v1`.
//! No Mesa/C++ dependency enters any shipping crate; this crate is not a workspace member.

mod abi;
mod contract;
mod export;
mod mesa;
mod reference;
mod util;

use util::*;

const USAGE: &str = "aco-export <command> [options]

  build-mesa  --mesa-src DIR --build-dir DIR [--jobs N] [--meson PATH] [--ninja PATH]
      Verify the isolated pinned Mesa worktree (HEAD == pin), apply the pinned patch (sha256-bound) if the
      tree is clean, verify every patched file hash, meson-setup (pinned options) and build the patched RADV
      ICD + libamdgpu_noop_drm_shim.
  build-host  --out PATH [--cc CC]
      Compile host/vk_aco_host.c (plain Vulkan, pkg-config vulkan).
  export      --kernel NAME --out DIR --mesa-src DIR --mesa-build DIR --host PATH
              (--pci-bdf DDDD:BB:DD.F | --offline-shim) [--llvm-mc PATH] [--spec producer.json]
      shader.spv, shader.bin, shader.abi.json, shader.rawspec.json, shader.aco.dump, artifact.json, capture/.
  oracle      --artifact DIR --radiowave PATH
      Run the existing `radiowave oracle aco --input shader.aco.dump --input-artifact shader.spv ...`.
  contract    --artifact DIR --capture CAPTURE_DIR --out FILE [--odd-tail-evidence-sha256 HEX]
      Derive pm-aco-hip-contract-v1 from the sidecar/rawspec/capture.
  reference   --artifact DIR --capture CAPTURE_DIR --prepare-dir DIR --out DIR --pci-bdf BDF
              --mesa-src DIR --mesa-build DIR --host PATH
      Original Vulkan dispatch on the exact PCI device -> receipt.json (aco-vulkan-ref-v1) + framed post buffers.
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else {
        eprint!("{USAGE}");
        std::process::exit(2);
    };
    let rest = &argv[1..];
    let result = match cmd.as_str() {
        "build-mesa" => Args::parse(rest, &[]).and_then(|a| mesa::cmd_build_mesa(&a)),
        "build-host" => Args::parse(rest, &[]).and_then(|a| mesa::cmd_build_host(&a)),
        "export" => Args::parse(rest, &["offline-shim"]).and_then(|a| export::cmd_export(&a)),
        "oracle" => Args::parse(rest, &[]).and_then(|a| export::cmd_oracle(&a)),
        "contract" => Args::parse(rest, &[]).and_then(|a| contract::cmd_contract(&a)),
        "reference" => Args::parse(rest, &[]).and_then(|a| reference::cmd_reference(&a)),
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n{USAGE}")),
    };
    if let Err(e) = result {
        eprintln!("aco-export: {e}");
        std::process::exit(1);
    }
}
