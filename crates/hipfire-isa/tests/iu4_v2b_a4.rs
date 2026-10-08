// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 M512 x N128 V2B gate/up entries (`iu4_v2b_a4`, A4 fusion:
//! the stage-1 h twin and the stage-2 fused A4 epilogue): determinism, the
//! per-K256 census of the retiled K loop, assembly, byte identity of the
//! committed bundle and gfx11 certification.
use hipfire_isa::Arch;
use hipfire_isa::kernels::iu4_v2b_a4::{self, Epi, Spec};
use std::io::Write;
use std::process::{Command, Stdio};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

fn twin(epi: Epi) -> hipfire_isa::Emitted { iu4_v2b_a4::emit(Spec { arch: Arch::Gfx1151, epi }).unwrap() }

#[test]
fn m512_entries_are_deterministic_and_gfx1151_only() {
    for epi in Epi::ALL {
        let (a, b) = (twin(epi), twin(epi));
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        assert!(a.shape.next_free_vgpr <= iu4_v2b_a4::VGPR_CEILING);
        for arch in [Arch::Gfx1100, Arch::Gfx1201] { assert!(iu4_v2b_a4::emit(Spec { arch, epi }).is_err()); }
    }
}

#[test]
fn m512_k_loop_census_per_k256() {
    // One trip = two K128 periods = four K64 halves. V2B's WMMA, fold and
    // scale-broadcast work per K256 (256 K16 WMMAs, 384 VOPD fold packets,
    // 64 `ds_swizzle_b32`), the same 160 fragment loads; two barriers per
    // K128 instead of one, and five b64 staging loads/stores per K64 half
    // (A 4 + X 1) instead of eight per K128 epoch.
    for epi in Epi::ALL {
        let census = iu4_v2b_a4::hot_loop_census(&twin(epi).s_text, epi);
        for (name, n) in [("v_wmma_i32_16x16x16_iu4", 256), ("vopd_packets", 384), ("v_dual_mul_f32", 256), ("v_dual_fmac_f32", 128),
            ("ds_swizzle_b32", 64), ("v_cvt_f32_f16_e64", 4), ("v_xor_b32_e32", 32), ("s_barrier", 4),
            ("ds_load_2addr_b64", 160), ("ds_store_b64", 20), ("global_load_b64", 20), ("global_load_b32", 8), ("global_load_u16", 4)] {
            assert_eq!(census.get(name).copied().unwrap_or(0), n, "{epi:?} {name}");
        }
        for unpaired in ["v_mul_f32_e32", "v_add_f32_e32", "v_fmac_f32_e32", "v_fma_f32", "buffer_gl0_inv", "s_nop", "v_nop"] {
            assert!(!census.contains_key(unpaired), "{epi:?} {unpaired}");
        }
    }
}

#[test]
fn m512_module_assembles_for_gfx1151_with_zero_diagnostics() {
    let llvm_mc = require_rocm_tool!("llvm-mc");
    let text = iu4_v2b_a4::emit_module(Arch::Gfx1151).unwrap().1;
    let mut child = Command::new(llvm_mc).args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1151", "-filetype=obj", "-o", "/dev/null"])
        .stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success() && out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
}

/// The A4-fusion code object is its own bundle (the certified V2B bundle is
/// not re-emitted): the committed bytes are exactly today's native emission.
#[test]
fn committed_a4_bundle_equals_fresh_native_emission() {
    let path = format!("{}/../../kernels/{}.hxaco", env!("CARGO_MANIFEST_DIR"), iu4_v2b_a4::MODULE);
    let committed = std::fs::read(&path).unwrap();
    let text = iu4_v2b_a4::emit_module(Arch::Gfx1151).unwrap().1;
    let elf = hipfire_isa::native::assemble(&text, Arch::Gfx1151).unwrap();
    assert!(hipfire_isa::native::bundle(&elf, Arch::Gfx1151, hipfire_isa::native::DEFAULT_HOST_TARGET) == committed, "{path}: fresh emission differs");
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, toolchain::{build, Toolchain}};

    #[test]
    fn m512_entries_pass_gfx1151_certification_checks() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (_, text, _) = iu4_v2b_a4::emit_module(Arch::Gfx1151).unwrap();
        ledger_replay::replay_waits(&text, Arch::Gfx1151).unwrap();
        let dir = std::env::temp_dir().join(format!("hipfire-isa-v2b-a4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("v2b_a4.s");
        std::fs::write(&s, &text).unwrap();
        let build = build(&Toolchain::oracle(), &s, &dir.join("v2b_a4.hsaco"), "gfx1151").unwrap();
        for epi in Epi::ALL {
            let symbol = Spec { arch: Arch::Gfx1151, epi }.symbol();
            // The K loop's three 20 KiB K64 slots end at 60 KiB; the A4
            // epilogue's two 32 KiB token panels fill the allocation.
            let end = if epi == Epi::SiluA4 { iu4_v2b_a4::LDS_BYTES } else { 3 * iu4_v2b_a4::SLOT_BYTES };
            assert_eq!(pm_check::lds_bounds(&text, &symbol, iu4_v2b_a4::WAVES, iu4_v2b_a4::LDS_BYTES).unwrap(), end);
            assert!(pm_check::lds_bounds(&text, &symbol, iu4_v2b_a4::WAVES, end - 8).is_err());
            let m7 = pm_check::m7(&build.elf, "gfx1151", &symbol).unwrap();
            assert_eq!(m7["lift"], "byte-exact", "{symbol}");
            assert_eq!(m7["obligations"], serde_json::json!({}), "{symbol}");
        }
    }
}
