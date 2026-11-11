// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Opcode-table gap scan (PM-KRSI-1100 LiftProbe). Takes `llvm-objdump --mcpu=gfx1100 -d`
//! listings of a device ELF; for every instruction of every symbol asks the gfx11 codec to
//! decode and re-encode the words. Prints, per symbol, instruction count, how many do not
//! decode / re-encode, and the distinct failing mnemonics (the missing `gfx1100.tbl` rows).
//! Usage: x1_gap_scan LISTING.dis... [--only SYMBOL_SUBSTR]

use peacemaker_ir::codec::{gfx11, gfx12};
use peacemaker_ir::inst::Arch;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files = Vec::new();
    let mut only: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--only" {
            only = Some(args[i + 1].clone());
            i += 2;
        } else {
            files.push(args[i].clone());
            i += 1;
        }
    }
    let arch = Arch::Gfx1100;
    let mut global: BTreeMap<String, usize> = BTreeMap::new();
    for file in files {
        println!("== {file}");
        let text = std::fs::read_to_string(&file).unwrap();
        let mut sym = String::new();
        let mut stats: BTreeMap<String, (usize, usize, BTreeMap<String, (usize, String)>)> =
            BTreeMap::new();
        let mut order = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.split_once(" <").filter(|_| !line.starts_with('\t')) {
                sym = rest.1.trim_end_matches(">:").to_owned();
                order.push(sym.clone());
                continue;
            }
            let Some((asm, comment)) = line.strip_prefix('\t').and_then(|l| l.split_once("//"))
            else {
                continue;
            };
            let Some((_, words)) = comment.split_once(": ") else {
                continue;
            };
            let words: Vec<u32> = words
                .split_whitespace()
                .filter_map(|w| u32::from_str_radix(w, 16).ok())
                .collect();
            let name = asm.split_whitespace().next().unwrap_or("").to_owned();
            let e = stats.entry(sym.clone()).or_default();
            e.0 += 1;
            let bad = match gfx11::decode(arch, &words) {
                Err(e) => Some(format!("decode: {e}")),
                Ok((inst, n)) => match gfx12::encode_for(arch, &inst) {
                    Err(e) => Some(format!("encode: {e}")),
                    Ok(enc) if enc.as_slice() != &words[..n] || n != words.len() => {
                        Some(format!("identity: {enc:08x?} vs {words:08x?}"))
                    }
                    Ok(_) => None,
                },
            };
            if let Some(why) = bad {
                e.1 += 1;
                let key = name.clone();
                let slot =
                    e.2.entry(key)
                        .or_insert((0, format!("{asm} [{words:08x?}] {why}")));
                slot.0 += 1;
                *global.entry(name).or_default() += 1;
            }
        }
        for s in order {
            if let Some(o) = &only {
                if !s.contains(o.as_str()) {
                    continue;
                }
            }
            let (n, bad, miss) = &stats[&s];
            println!("- {s}: {n} insts, {bad} fail, {} distinct", miss.len());
            for (k, (c, ex)) in miss {
                println!("    {c:5} {k}   e.g. {}", ex.trim());
            }
        }
    }
    println!("== global failing mnemonics: {}", global.len());
    for (k, v) in &global {
        println!("{v:6} {k}");
    }
}
