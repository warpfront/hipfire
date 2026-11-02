// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use peacemaker_npu::isa::decode::{decode, executable_sections};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
fn cores(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            cores(&p, out);
        } else if p.extension().is_some_and(|e| e == "elf") {
            out.push(p);
        }
    }
}
fn normalize(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
#[test]
#[ignore = "needs an AIE2P vendor core-ELF corpus (AIE2P_VENDOR_CORPUS) and an llvm-aie objdump built from libLLVM (AIE2P_OBJDUMP, default /tmp/aie2p-llvm-objdump); neither is in the repo"]
fn vendor_decode_oracle_roundtrip() {
    let root = PathBuf::from(
        std::env::var_os("AIE2P_VENDOR_CORPUS")
            .expect("set AIE2P_VENDOR_CORPUS to the vendor core-ELF corpus"),
    );
    let oracle = std::env::var_os("AIE2P_OBJDUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/tmp/aie2p-llvm-objdump".into());
    assert!(
        oracle.exists(),
        "AIE2P llvm-objdump missing; set AIE2P_OBJDUMP"
    );
    let mut files = Vec::new();
    cores(&root, &mut files);
    files.sort();
    assert!(!files.is_empty(), "corpus present but has no core ELFs");
    let mut bundles = 0;
    let mut instructions = 0;
    let mut bytes_count = 0;
    for path in &files {
        let data = std::fs::read(path).unwrap();
        let output = std::process::Command::new(&oracle)
            .args(["-d", "-z", "--triple=aie2p", "--no-show-raw-insn"])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let mut expected = BTreeMap::new();
        for line in text.lines() {
            if let Some((address, asm)) = line.trim().split_once(':') {
                if let Ok(pc) = u64::from_str_radix(address, 16) {
                    expected.insert(pc, normalize(asm));
                }
            }
        }
        for section in executable_sections(&data).unwrap() {
            let mut offset = 0;
            while offset < section.bytes.len() {
                let pc = section.address + offset as u64;
                let inst = decode(&section.bytes[offset..], pc)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                assert_eq!(
                    inst.encode().unwrap(),
                    section.bytes[offset..offset + inst.len],
                    "{} {pc:#x} roundtrip",
                    path.display()
                );
                let want = expected
                    .remove(&pc)
                    .unwrap_or_else(|| panic!("{} {pc:#x} missing oracle line", path.display()));
                assert_eq!(
                    normalize(&inst.assembly()),
                    want,
                    "{} {pc:#x}",
                    path.display()
                );
                bundles += 1;
                instructions += inst.instructions.len();
                bytes_count += inst.len;
                offset += inst.len;
            }
        }
        assert!(
            expected.is_empty(),
            "{} unmatched oracle instructions",
            path.display()
        );
    }
    eprintln!("decoded {} ELFs: {bundles} bundles, {instructions} slot instructions, {bytes_count} bytes; oracle text and byte roundtrips match",files.len());
}
#[test]
fn decoder_rejects_invalid_elf() {
    assert!(executable_sections(b"not an ELF").is_err());
}

#[test]
fn decoder_scalar_boundaries_and_reserved_bits() {
    use peacemaker_npu::isa::{add_ri, movxm, Reg};
    for value in [-64, -1, 0, 63] {
        let bytes = add_ri(Reg::R(31), Reg::R(0), value).encode();
        let decoded = decode(&bytes, 0x100).unwrap();
        let immediate = decoded.instructions[0]
            .operands
            .iter()
            .find(|o| o.name == "imm")
            .unwrap();
        assert_eq!(immediate.value, value);
        assert_eq!(decoded.encode().unwrap(), bytes);
        for end in 0..bytes.len() {
            assert!(decode(&bytes[..end], 0x100).is_err());
        }
    }
    for value in [0, u32::MAX, 0x80000000] {
        let bytes = movxm(Reg::R(7), value).encode();
        let decoded = decode(&bytes, 0).unwrap();
        assert_eq!(
            decoded.instructions[0]
                .operands
                .iter()
                .find(|o| o.name == "i")
                .unwrap()
                .value,
            value as i32 as i64
        );
        assert_eq!(decoded.encode().unwrap(), bytes);
    }
    // Preserve every format-level don't-care bit around the fixed NOP16 slot.
    let bytes = [0xf0, 0x7f];
    let decoded = decode(&bytes, 0).unwrap();
    assert_eq!(decoded.assembly().trim(), "nop");
    assert_eq!(decoded.encode().unwrap(), bytes);
}
