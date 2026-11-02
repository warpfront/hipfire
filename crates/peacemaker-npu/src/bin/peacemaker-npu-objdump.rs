// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Offline AIE2P ELF disassembler; no vendor runtime or hardware access.
use peacemaker_npu::isa::decode::{decode, executable_sections};
fn main() {
    if let Err(e) = run() {
        eprintln!("peacemaker-npu-objdump: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: peacemaker-npu-objdump CORE.elf [...]".into());
    }
    for path in paths {
        let bytes = std::fs::read(&path)?;
        println!("{}:", std::path::Path::new(&path).display());
        for section in executable_sections(&bytes)? {
            println!("Disassembly of section {}:", section.name);
            let mut offset = 0;
            while offset < section.bytes.len() {
                let inst = decode(&section.bytes[offset..], section.address + offset as u64)?;
                print!("{:08x}: ", inst.pc);
                for byte in &section.bytes[offset..offset + inst.len] {
                    print!("{byte:02x} ");
                }
                println!("\t{}", inst.assembly());
                offset += inst.len;
            }
        }
    }
    Ok(())
}
