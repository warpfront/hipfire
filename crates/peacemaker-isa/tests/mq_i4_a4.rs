// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 row-local R128 / K640 A4 producer `mq_i4_a4_r128_k640` (module
//! `mq_i4_a4_pm_gfx1201`): symbol, 48-byte kernarg ABI, exact-arch gating, the
//! instruction census of the frozen numerics, the row-locality structure (straight-line,
//! three loads of exactly this row group, no LDS), the committed bundle and contract
//! identities, and (feature `toolchain`) wait replay, M7 and the radiowave certification.
//! The byte oracle against the CPU twin (`pilot_solve.a4_emulate`) is the GPU harness
//! `crates/peacemaker-isa/tools/mq_i4_a4_oracle.py`.
use peacemaker_isa::kernels::mq_i4_a4::{self, Spec};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "mq_i4_a4_pm_gfx1201";
const SYMBOL: &str = "mq_i4_a4_r128_k640";
/// Contract ABI: `h sorted xq signs1 signs2` pointers, then `P L` u32.
const ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 8), (32, 8), (40, 4), (44, 4)];

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(mnemonic))
        .count()
}
fn spec() -> Spec {
    Spec { arch: ARCH }
}

#[test]
fn module_export_is_the_frozen_symbol() {
    assert_eq!(mq_i4_a4::module_specs(ARCH), [spec()]);
    assert_eq!(spec().symbol(), SYMBOL);
    assert_eq!(spec().tag(), "r128_k640");
    assert_eq!(Spec::module(ARCH), MODULE);
    let (emitted, text, proof) = mq_i4_a4::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), 1);
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 1);
    assert!(
        text.contains(&format!(".amdhsa_kernel {SYMBOL}\n"))
            && text.contains(&format!(".globl {SYMBOL}\n"))
    );
    assert!(text.contains(&format!(".name: {SYMBOL}\n")));
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
}

#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    let (a, b) = (
        mq_i4_a4::emit(spec()).unwrap(),
        mq_i4_a4::emit(spec()).unwrap(),
    );
    assert_eq!(
        (a.proof.s_text_sha256, a.s_text),
        (b.proof.s_text_sha256, b.s_text)
    );
    let (_, ta, pa) = mq_i4_a4::emit_module(ARCH).unwrap();
    let (_, tb, pb) = mq_i4_a4::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(mq_i4_a4::emit(Spec { arch }).is_err(), "{arch:?}");
        assert!(mq_i4_a4::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Wave32, one wave per workgroup, no scratch, spills, LDS or barrier, 72 VGPRs.
#[test]
fn export_is_wave32_without_scratch_or_lds() {
    let e = mq_i4_a4::emit(spec()).unwrap();
    assert_eq!(e.proof.arch, ARCH);
    assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"), "wave32");
    assert!(
        e.s_text.contains(".private_segment_fixed_size: 0\n"),
        "private segment"
    );
    assert!(
        e.s_text.contains(".sgpr_spill_count: 0\n") && e.s_text.contains(".vgpr_spill_count: 0\n"),
        "spills"
    );
    assert!(
        e.s_text.contains(".group_segment_fixed_size: 0\n"),
        "no LDS"
    );
    assert_eq!(e.shape.barriers, 0);
    assert_eq!(e.shape.ds, 0, "no DS instruction at all");
    assert_eq!(e.shape.next_free_vgpr, mq_i4_a4::VGPRS);
    for forbidden in [
        "ds_",
        "buffer_",
        "flat_",
        "scratch_",
        "s_barrier",
        "v_wmma",
        "v_swmmac",
        "s_waitcnt",
        "v_readlane",
        "v_writelane",
        "v_fma_mix",
    ] {
        let hit = e.s_text.lines().find(|l| {
            l.split_whitespace()
                .next()
                .is_some_and(|m| m.starts_with(forbidden))
        });
        assert!(hit.is_none(), "emits {hit:?}");
    }
}

#[test]
fn kernarg_abi_matches_the_contract() {
    assert_eq!(mq_i4_a4::KERNARG_BYTES, 48);
    assert_eq!(mq_i4_a4::WORKGROUP_SIZE, 32);
    let e = mq_i4_a4::emit(spec()).unwrap();
    assert!(
        e.s_text.contains(".kernarg_segment_size: 48\n"),
        "kernarg size"
    );
    assert_eq!(
        field(&e.s_text, ".offset:"),
        ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
        "kernarg offsets"
    );
    assert_eq!(
        field(&e.s_text, ".size:"),
        ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
        "kernarg sizes"
    );
    assert_eq!(
        field(&e.s_text, ".max_flat_workgroup_size:"),
        [32],
        "block 32"
    );
    let (last_off, last_size) = *ARGS.last().unwrap();
    assert_eq!(
        last_off + last_size,
        mq_i4_a4::KERNARG_BYTES,
        "arguments cover the kernarg block"
    );
    for (name, access) in [
        ("h", "read_only"),
        ("sorted", "read_only"),
        ("xq", "write_only"),
        ("signs1", "read_only"),
        ("signs2", "read_only"),
    ] {
        assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{name} access {access}");
    }
    // Kernargs: h, sorted, xq, signs1 in one 32-byte load; signs2, P, L in one 16-byte load at 0x20.
    assert!(
        e.s_text.contains("s_load_b256 s[8:15], s[0:1], 0x0"),
        "first kernarg load"
    );
    assert!(
        e.s_text.contains("s_load_b128 s[16:19], s[0:1], 0x20"),
        "second kernarg load"
    );
}

/// The row-local structure. One straight-line function of one (row, group): only the two
/// guard branches to the single exit (no loop, no other branch), and exactly the loads of
/// that row group: the slot, the group's bf16 dwords through the `h` row-group pointer
/// s[30:31], the two 128-entry sign quads through s[14:15] / s[16:17]. The only stores are
/// the record's qs half-words and (d, s), through the record pointer s[32:33]. The row
/// count `P` (s18) reaches the code only through the grid guard.
#[test]
fn producer_is_one_straight_line_row_local_function() {
    let t = mq_i4_a4::emit(spec()).unwrap().s_text;
    assert_eq!(count(&t, "s_cbranch_scc1"), 2, "exit guards");
    for m in [
        "s_branch",
        "s_cbranch_scc0",
        "s_cbranch_vccz",
        "s_cbranch_vccnz",
        "s_cbranch_execz",
        "s_cbranch_execnz",
    ] {
        assert_eq!(count(&t, m), 0, "{m}");
    }
    // Code labels only (the metadata YAML also has `key:` lines); builder labels are emitted as tab-indented instruction lines.
    let labels: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.ends_with(':') && (l.starts_with(".L") || *l == format!("{SYMBOL}:")))
        .collect();
    assert_eq!(
        labels,
        [
            format!("{SYMBOL}:").as_str(),
            ".Lmqi4a4_exit:",
            format!(".L{SYMBOL}_end:").as_str()
        ],
        "symbol, exit and end labels only"
    );
    assert_eq!(
        t.matches(".Lmqi4a4_exit").count(),
        3,
        "two branches to one exit label plus its definition"
    );
    assert_eq!(
        (
            count(&t, "s_load_b256"),
            count(&t, "s_load_b128"),
            count(&t, "s_load_b32")
        ),
        (1, 1, 1),
        "kernargs and the slot"
    );
    let loads: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("global_load"))
        .collect();
    assert_eq!(loads.len(), 3, "{loads:?}");
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b64 v[6:7], v1, s[30:31]"))
            .count(),
        1,
        "{loads:?}"
    );
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b128 v[8:11], v2, s[14:15]"))
            .count(),
        1,
        "{loads:?}"
    );
    assert_eq!(
        loads
            .iter()
            .filter(|l| l.starts_with("global_load_b128 v[12:15], v2, s[16:17]"))
            .count(),
        1,
        "{loads:?}"
    );
    let stores: Vec<&str> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("global_store"))
        .collect();
    assert_eq!(
        stores,
        [
            "global_store_b16 v3, v43, s[32:33]",
            "global_store_b64 v4, v[69:70], s[32:33]"
        ]
    );
    // P (s18) is read by exactly the one grid-guard multiply.
    let p_uses = t
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.starts_with('.')
                && l.split_once(' ').is_some_and(|(_, ops)| {
                    ops.split(|c: char| !c.is_ascii_alphanumeric())
                        .any(|w| w == "s18")
                })
        })
        .count();
    assert_eq!(p_uses, 1, "P reaches the code only through the grid guard");
    assert_eq!(count(&t, "s_endpgm"), 1);
}

/// The frozen numerics' census: the BF16 widen / sign / butterfly / scale rotation, five
/// lane stages (DPP xor 1,2,4,8 and permlanex16 for 16), the nine full IEEE divides (one for
/// `amax / 7`, four per candidate; no reciprocal shortcut), the two mse and the amax and
/// code-sum xor trees (16,8,4,2,1), the strict winner selection and the unit block.
#[test]
fn instruction_census_matches_the_frozen_dag() {
    let t = mq_i4_a4::emit(spec()).unwrap().s_text;
    // Nine divides, each the 2 scales + rcp + 3 fma + 2 fmac + mul + fmas + fixup expansion.
    assert_eq!(count(&t, "v_div_scale_f32"), 18);
    assert_eq!(count(&t, "v_div_fmas_f32"), 9);
    assert_eq!(count(&t, "v_div_fixup_f32"), 9);
    assert_eq!(count(&t, "v_rcp_f32_e32"), 9);
    // Xor exchanges: 4 reductions (amax, mse0, mse1, code sum) x 4 DPP + 1 permlane, and the
    // lane butterflies (strides 1,2,4,8 DPP, stride 16 permlane) on each of the four elements.
    assert_eq!(count(&t, "v_mov_b32_dpp"), 4 * 4 + 4 * 4);
    assert_eq!(count(&t, "v_permlanex16_b32"), 4 + 4);
    for m in [1, 2, 4, 8] {
        assert_eq!(
            t.matches(&format!("row_xmask:{m} ")).count(),
            4 + 4,
            "row_xmask:{m}"
        );
    }
    // v_fma_f32: 3 per divide, 8 error terms, 20 lane-butterfly fma; fmac: 2 per divide + 3 per mse.
    assert_eq!(count(&t, "v_fma_f32"), 27 + 8 + 20);
    assert_eq!(count(&t, "v_fmac_f32_e32"), 18 + 6);
    // Clamp / round / select / convert.
    assert_eq!(count(&t, "v_rndne_f32_e32"), 8 + 4);
    assert_eq!(count(&t, "v_maxmin_num_f32"), 8 + 4);
    assert_eq!(
        (
            count(&t, "v_max_num_f32_e64"),
            count(&t, "v_max_num_f32_e32")
        ),
        (2, 6)
    );
    assert_eq!(count(&t, "v_cvt_i32_f32_e32"), 4);
    assert_eq!(count(&t, "v_cndmask_b32_e64"), 1 + 4 + 8 + 2);
    assert_eq!(
        (count(&t, "v_cmp_eq_f32_e64"), count(&t, "v_cmp_gt_f32_e64")),
        (1, 2)
    );
    assert_eq!(count(&t, "v_cmpx_eq_u32_e32"), 1);
    // No MQ4 XOR rebias, no reciprocal-product code path.
    assert_eq!(count(&t, "v_xor_b32_e32"), 0);
    // Frozen constants: scale, candidate 0 multiplier, 7.0, -8.0, 1e30, the permlane selector.
    for needle in [
        "v_mul_f32_e32 v16, 0x3db504f3, v16",
        "0x3fdb6db7",
        "s_mov_b32 s34, 0x40e00000",
        "s_mov_b32 s35, 0xc1000000",
        "0x7149f2ca",
        "s_mov_b32 s41, 0x76543210",
    ] {
        assert!(t.contains(needle), "missing {needle}");
    }
    // Every reduction tree (amax, mse0, mse1, code sum) exchanges lane xor 16 (permlane), then 8, 4, 2, 1.
    let exchanges: Vec<u32> = t
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("v_permlanex16_b32 v37,") || l.starts_with("v_mov_b32_dpp v37,"))
        .map(|l| {
            if l.starts_with("v_permlanex16_b32") {
                16
            } else {
                l.split("row_xmask:")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap()
            }
        })
        .collect();
    assert_eq!(exchanges, [16, 8, 4, 2, 1].repeat(4));
}

#[test]
fn constants_match_the_profile_and_the_frozen_producer() {
    assert_eq!(
        (mq_i4_a4::K, mq_i4_a4::GROUP, mq_i4_a4::GROUPS),
        (640, 128, 5)
    );
    assert_eq!((mq_i4_a4::RECORD_BYTES, mq_i4_a4::ROW_BYTES), (72, 1280));
    assert_eq!(mq_i4_a4::SIGN_SEEDS, (43, 1043));
    assert_eq!(mq_i4_a4::ROTATION_SCALE_BITS, 0.0883883476f32.to_bits());
    assert_eq!(
        mq_i4_a4::ROTATION_SCALE_BITS,
        (1.0f32 / 128.0f32.sqrt()).to_bits()
    );
    assert_eq!(mq_i4_a4::CAND0_MULT_BITS, 0x3fdb_6db7);
    // The workgroup split x -> (x / 5, x % 5) is exact for every u32 (the kernel's multiply-high).
    for x in (0..200_000u32).chain([
        u32::MAX,
        u32::MAX - 1,
        u32::MAX - 4,
        0x8000_0000,
        0xcccc_cccc,
        0xcccc_cccd,
    ]) {
        assert_eq!(((u64::from(x) * 0xcccc_cccd) >> 34) as u32, x / 5, "{x}");
    }
}

fn bundle_path() -> String {
    format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The committed `kernels/mq_i4_a4_pm_gfx1201.hxaco` must be exactly what the builder emits
/// today (native writer, no ROCm needed;
/// `peacemaker-isa emit --kernel mq_i4_a4 --arch gfx1201 ... --bundle`).
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel mq_i4_a4 --arch gfx1201 ... --bundle)"));
    let (_, text, _) = mq_i4_a4::emit_module(ARCH).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
}

/// The same `.text` through the ROCm oracle's `llvm-mc` + `ld.lld`.
#[test]
fn committed_bundle_text_equals_oracle_link() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let committed = std::fs::read(bundle_path()).unwrap();
    let text = mq_i4_a4::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), MODULE)) == text_section(&committed),
        "{MODULE}: oracle-linked emission differs from the committed bundle"
    );
}

/// The committed shape contract is exactly what `contract()` derives from fresh emission
/// (`peacemaker-isa emit --kernel mq_i4_a4 ... --co ... --shape crates/peacemaker-isa/kernels`).
#[test]
fn committed_contract_equals_fresh_contract() {
    let path = format!(
        "{}/kernels/mq_i4_a4.{}.{}.contract.json",
        env!("CARGO_MANIFEST_DIR"),
        ARCH.name(),
        spec().tag()
    );
    let committed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
            .unwrap();
    let fresh = mq_i4_a4::contract(spec(), &mq_i4_a4::emit(spec()).unwrap());
    assert_eq!(committed, fresh, "{path}");
    assert_eq!(committed["symbol"], SYMBOL);
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-mqi4a4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The export's waits replay independently (the retained wait ledger and the
    /// independent source-text replay agree), no LDS is touched (so the LDS audit has nothing
    /// to bound), and M7 lifts the natively linked object byte-exactly with no outstanding
    /// obligations (no ROCm tools).
    #[test]
    fn export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = mq_i4_a4::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4a4-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        let m7 = pm_check::m7(&co, ARCH.name(), SYMBOL).unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{SYMBOL}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The symbol certifies against the ROCm oracle build: parse-back, independent wait replay,
    /// its committed shape contract, and M7 of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = mq_i4_a4::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4a4-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let m7 = pm_check::m7(&build.elf, ARCH.name(), SYMBOL)
            .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{SYMBOL}"
        );
        let path = format!(
            "{}/kernels/mq_i4_a4.{}.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name(),
            spec().tag()
        );
        let contract =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join("manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
