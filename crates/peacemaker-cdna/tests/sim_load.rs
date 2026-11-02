// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `sim::load_code_object` against code objects built by ROCm's `llvm-mc` and
//! `ld.lld -shared` from a `.amdhsa_kernel` assembly file.
use peacemaker_cdna::sim::load_code_object;
use peacemaker_cdna::{oracle, Arch};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const INSTS: &[&str] = &[
    "s_load_dwordx2 s[2:3], s[0:1], 0x8",
    "s_waitcnt lgkmcnt(0)",
    "v_mov_b32 v1, 0x12345678",
    "v_add_u32 v1, 7, v1",
    "s_barrier",
    "s_endpgm",
];

fn kernel_source(version: u32) -> String {
    let insts: String = INSTS.iter().map(|i| format!("  {i}\n")).collect();
    format!(
        r#".amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.amdhsa_code_object_version {version}
.text
.globl pm_k
.p2align 8
.type pm_k,@function
pm_k:
{insts}.Lpm_k_end:
.size pm_k, .Lpm_k_end-pm_k

.rodata
.p2align 6
.amdhsa_kernel pm_k
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_count 2
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 8
  .amdhsa_accum_offset 16
  .amdhsa_group_segment_fixed_size 4096
  .amdhsa_kernarg_size 24
.end_amdhsa_kernel
"#
    )
}

fn scratch(tag: &str, ext: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "pm-cdna-sim-load-{}-{n}-{tag}.{ext}",
        std::process::id()
    ))
}

fn run(cmd: &mut Command) {
    let out = cmd.output().unwrap_or_else(|e| panic!("{cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Assemble and link `source`; returns the shared object's bytes.
fn build(tag: &str, source: &str) -> Vec<u8> {
    let (s, o, so) = (scratch(tag, "s"), scratch(tag, "o"), scratch(tag, "so"));
    std::fs::write(&s, source).unwrap();
    run(Command::new(oracle::tool("llvm-mc"))
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx942",
            "-filetype=obj",
            "-o",
        ])
        .arg(&o)
        .arg(&s));
    run(Command::new(oracle::tool("ld.lld"))
        .args(["-shared", "-o"])
        .arg(&so)
        .arg(&o));
    let elf = std::fs::read(&so).unwrap();
    for p in [s, o, so] {
        let _ = std::fs::remove_file(p);
    }
    elf
}

fn expected_words() -> Vec<u32> {
    let lines: Vec<String> = INSTS.iter().map(|s| s.to_string()).collect();
    let bytes: Vec<u8> = oracle::mc_encode(Arch::Gfx942, &lines)
        .unwrap()
        .into_iter()
        .flat_map(|(_, b)| b)
        .collect();
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn check_version(version: u32) {
    let elf = build(&format!("v{version}"), &kernel_source(version));
    let image = load_code_object(&elf, "pm_k").unwrap_or_else(|e| panic!("v{version}: {e}"));
    assert_eq!(image.arch, Arch::Gfx942);
    assert_eq!(image.name, "pm_k");
    assert_eq!(image.words, expected_words(), "v{version} code words");
    let kd = &image.kd;
    assert_eq!(kd.user_sgpr_count(), 2);
    assert!(kd.enable_sgpr_kernarg_segment_ptr());
    assert!(!kd.enable_sgpr_private_segment_buffer());
    assert_eq!(kd.kernarg_size, 24);
    assert_eq!(kd.group_segment_fixed_size, 4096);
    assert_eq!(
        kd.accum_offset_field(),
        3,
        "accum_offset 16 → field 16/4 - 1"
    );
    assert!(kd.enable_workgroup_id(0) && kd.enable_workgroup_id(1) && !kd.enable_workgroup_id(2));
    assert_eq!(
        kd.kernel_code_entry_byte_offset % 64,
        0,
        "code and descriptor are both 64-byte aligned"
    );
}

#[test]
fn loads_code_object_v6() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping loads_code_object_v6");
        return;
    }
    check_version(6);
}

#[test]
fn loads_code_object_v5() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping loads_code_object_v5");
        return;
    }
    check_version(5);
}

fn good_elf() -> Vec<u8> {
    build("neg", &kernel_source(6))
}

#[test]
fn rejects_missing_symbol() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping rejects_missing_symbol");
        return;
    }
    let elf = good_elf();
    let err = load_code_object(&elf, "no_such_kernel")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no_such_kernel"), "{err}");
    // The descriptor symbol alone is not a function symbol.
    assert!(load_code_object(&elf, "pm_k.kd").is_err());
}

#[test]
fn rejects_wrong_machine() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping rejects_wrong_machine");
        return;
    }
    let mut elf = good_elf();
    let flags = u32::from_le_bytes(elf[48..52].try_into().unwrap());
    elf[48..52].copy_from_slice(&((flags & !0xff) | 0x2f).to_le_bytes()); // gfx90a
    assert!(load_code_object(&elf, "pm_k")
        .unwrap_err()
        .to_string()
        .contains("EF_AMDGPU_MACH"));
    let mut elf = good_elf();
    elf[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    assert!(load_code_object(&elf, "pm_k")
        .unwrap_err()
        .to_string()
        .contains("EM_AMDGPU"));
}

#[test]
fn rejects_truncated_and_malformed() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping rejects_truncated_and_malformed");
        return;
    }
    let elf = good_elf();
    // Every strict prefix of the file is an error, never a panic.
    for len in (0..elf.len())
        .step_by(61)
        .chain([0, 1, 63, 64, elf.len() - 1])
    {
        assert!(
            load_code_object(&elf[..len], "pm_k").is_err(),
            "prefix of {len} bytes loaded"
        );
    }
    let mut bad = elf.clone();
    bad[0] = 0;
    assert!(load_code_object(&bad, "pm_k")
        .unwrap_err()
        .to_string()
        .contains("magic"));
    let mut bad = elf.clone();
    bad[4] = 1; // ELFCLASS32
    assert!(load_code_object(&bad, "pm_k").is_err());
    let mut bad = elf.clone();
    bad[5] = 2; // big-endian
    assert!(load_code_object(&bad, "pm_k").is_err());
    let mut bad = elf.clone();
    bad[40..48].copy_from_slice(&u64::MAX.to_le_bytes()); // e_shoff beyond the file
    assert!(load_code_object(&bad, "pm_k").is_err());
}

#[test]
fn rejects_entry_offset_off_symbol_and_relocatable() {
    if oracle::disabled() {
        eprintln!("ROCm oracle disabled: skipping rejects_entry_offset_off_symbol_and_relocatable");
        return;
    }
    // Shift the descriptor's entry offset by one dword so it misses the function symbol.
    let elf = good_elf();
    let kd_off = find_kd_file_offset(&elf);
    let mut bad = elf.clone();
    let entry = i64::from_le_bytes(bad[kd_off + 16..kd_off + 24].try_into().unwrap());
    bad[kd_off + 16..kd_off + 24].copy_from_slice(&(entry + 4).to_le_bytes());
    assert!(load_code_object(&bad, "pm_k")
        .unwrap_err()
        .to_string()
        .contains("kernel_code_entry_byte_offset"));
    // Non-zero reserved descriptor byte.
    let mut bad = elf.clone();
    bad[kd_off + 12] = 1;
    assert!(load_code_object(&bad, "pm_k").is_err());

    // Unlinked object: entry offset is an unapplied relocation.
    let (s, o) = (scratch("rel", "s"), scratch("rel", "o"));
    std::fs::write(&s, kernel_source(6)).unwrap();
    run(Command::new(oracle::tool("llvm-mc"))
        .args([
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx942",
            "-filetype=obj",
            "-o",
        ])
        .arg(&o)
        .arg(&s));
    let rel = std::fs::read(&o).unwrap();
    for p in [s, o] {
        let _ = std::fs::remove_file(p);
    }
    assert!(load_code_object(&rel, "pm_k")
        .unwrap_err()
        .to_string()
        .contains("relocatable"));
}

/// File offset of `pm_k.kd`'s bytes, found by scanning for the descriptor's
/// known `kernarg_size = 24`, `group_segment_fixed_size = 4096` prefix.
fn find_kd_file_offset(elf: &[u8]) -> usize {
    let mut pat = [0u8; 12];
    pat[0..4].copy_from_slice(&4096u32.to_le_bytes());
    pat[8..12].copy_from_slice(&24u32.to_le_bytes());
    elf.windows(12)
        .position(|w| w == pat)
        .expect("descriptor bytes in the linked object")
}
