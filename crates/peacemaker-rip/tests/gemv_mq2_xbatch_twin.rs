// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of the x-batched MQ2G256V2 GEMV generator
//! (`kernels/gemv_mq2_xbatch.rip` vs `kernels/gemv_mq2_xbatch_pm_gfx1201.rs`): for every
//! `(rows, cap, unroll, split, ilp)` tuple the script either refuses exactly when the generator refuses or
//! re-emits its assembly text byte for byte, and the declared point (`SHIPPED`) assembles to
//! the ELF inside the committed `.hxaco`.

use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201::{emit_elf, emit_text, Params, SHIPPED};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

const RIP: &str = include_str!("../../peacemaker-isa/src/kernels/gemv_mq2_xbatch.rip");
const SHIPPED_HXACO: &[u8] = include_bytes!("../../../kernels/gemv_mq2_xbatch_pm_gfx1201.hxaco");

/// The script with the five axis consts rewritten (each is one `const NAME: TY = LIT;` line).
fn with_point(p: &Params) -> String {
    let mut src = RIP.to_string();
    for (name, ty, old, new) in [
        ("ROWS", "u8", "4", p.rows.to_string()),
        ("CAP", "u8", "8", p.cap.to_string()),
        ("UNROLL", "u8", "1", p.unroll.to_string()),
        ("SPLIT", "bool", "false", p.split.to_string()),
        ("ILP", "bool", "false", p.ilp.to_string()),
    ] {
        let from = format!("const {name}: {ty} = {old};\n");
        assert_eq!(src.matches(&from).count(), 1, "axis const {name}");
        src = src.replace(&from, &format!("const {name}: {ty} = {new};\n"));
    }
    src
}

fn rip(p: &Params) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(&with_point(p), "gemv_mq2_xbatch.rip", Arch::Gfx1201, "xbatch")
}

fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: rust `{x}` vs rip `{y}`", i + 1);
        }
    }
    format!("lengths {} vs {} lines", a.lines().count(), b.lines().count())
}

/// The device ELF payload of an uncompressed offload bundle (second entry).
fn bundle_elf(bundle: &[u8]) -> &[u8] {
    let u64_at = |at: usize| u64::from_le_bytes(bundle[at..at + 8].try_into().unwrap()) as usize;
    assert_eq!(&bundle[..24], b"__CLANG_OFFLOAD_BUNDLE__");
    let count = u64_at(24);
    let mut at = 32;
    let mut entries = Vec::new();
    for _ in 0..count {
        let (off, size, tl) = (u64_at(at), u64_at(at + 8), u64_at(at + 16));
        entries.push((off, size));
        at += 24 + tl;
    }
    let (off, size) = entries[1];
    &bundle[off..off + size]
}

/// Every grid tuple: the script refuses exactly the tuples the generator refuses, and emits its text for
/// the rest.
#[test]
fn every_tuple_matches_the_rust_generator() {
    let (mut agree_ok, mut agree_err) = (0, 0);
    for rows in [1u8, 2, 4] {
        for cap in [1u8, 2, 4, 8] {
            for unroll in [1u8, 2, 4] {
                for split in [false, true] {
                    for ilp in [false, true] {
                        let p = Params { rows, cap, unroll, split, ilp };
                        let tag = p.tag();
                        match (emit_text(&p), rip(&p)) {
                            (Ok(rust), Ok(rip)) => {
                                assert!(
                                    rust == rip.s_text,
                                    "{tag}: {}",
                                    first_difference(&rust, &rip.s_text)
                                );
                                agree_ok += 1;
                            }
                            (Err(_), Err(_)) => agree_err += 1,
                            (Ok(_), Err(e)) => panic!("{tag}: the generator emits, the script refuses: {e}"),
                            (Err(e), Ok(_)) => panic!("{tag}: the generator refuses ({e}), the script emits"),
                        }
                    }
                }
            }
        }
    }
    eprintln!("tuples: {agree_ok} emitted identically, {agree_err} refused by both");
    assert_eq!(agree_ok + agree_err, 3 * 4 * 3 * 2 * 2);
    assert!(agree_ok > 0 && agree_err > 0);
}

/// The declared point is the shipped tuple: the script's text assembles to the committed `.hxaco`'s ELF.
#[test]
fn declared_point_is_the_shipped_artifact() {
    let emitted = peacemaker_rip::compile_named(RIP, "gemv_mq2_xbatch.rip", Arch::Gfx1201, "xbatch")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(emitted.s_text, emit_text(&SHIPPED).unwrap());
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
    assert!(elf == emit_elf(&SHIPPED).unwrap(), "script ELF differs from emit_elf(SHIPPED)");
    assert!(elf == bundle_elf(SHIPPED_HXACO), "script ELF differs from the committed .hxaco");
    assert_eq!(
        format!("{:x}", Sha256::digest(&elf)),
        "0e9b63c7e3ac07c9f73e3c9e1ad86451c81c24a2ec58e24741a38d876b221bd4"
    );
}

/// The grid consts are single lines at the shipped point; `SEARCH_AXES` names the four schedule axes (CAP is
/// not one: a CAP < 8 object differs from CAP = 8 for B > CAP).
#[test]
fn axis_consts_are_single_lines_at_the_shipped_point() {
    for line in [
        "const ROWS: u8 = 4;",
        "const CAP: u8 = 8;",
        "const UNROLL: u8 = 1;",
        "const SPLIT: bool = false;",
        "const ILP: bool = false;",
    ] {
        assert_eq!(RIP.lines().filter(|l| *l == line).count(), 1, "{line}");
    }
    assert!(RIP.contains("(\"ROWS\", \"schedule\", &[\"1\", \"2\", \"4\"])"));
    assert!(!RIP.contains("(\"CAP\","));
    assert!(RIP.contains("(\"UNROLL\", \"schedule\", &[\"1\", \"2\", \"4\"])"));
    assert!(RIP.contains("(\"SPLIT\", \"schedule\", &[\"false\", \"true\"])"));
    assert!(RIP.contains("(\"ILP\", \"schedule\", &[\"false\", \"true\"])"));
}

/// The four script-only schedule axes (PREFETCH, DELAY_ALU, PRIO, XFIRST): single-line consts at their
/// reference values, listed in `SEARCH_AXES`, every listed value compiles (at the declared grid point and
/// PREFETCH across ROWS/UNROLL/SPLIT/ILP) and a non-reference value changes the text.
#[test]
fn new_schedule_axes_compile_and_reference_values_are_unchanged() {
    let axes: [(&str, &str, &str, &[&str]); 4] = [
        ("PREFETCH", "u8", "0", &["0", "1"]),
        ("DELAY_ALU", "bool", "false", &["false", "true"]),
        ("PRIO", "u8", "0", &["0", "1", "2"]),
        ("XFIRST", "bool", "false", &["false", "true"]),
    ];
    let reference = peacemaker_rip::compile_named(RIP, "gemv_mq2_xbatch.rip", Arch::Gfx1201, "xbatch")
        .unwrap_or_else(|e| panic!("{e}"))
        .s_text;
    for (name, ty, old, values) in axes {
        let line = format!("const {name}: {ty} = {old};");
        assert_eq!(RIP.lines().filter(|l| *l == line).count(), 1, "{line}");
        let list: Vec<String> = values.iter().map(|v| format!("\"{v}\"")).collect();
        assert!(RIP.contains(&format!("(\"{name}\", \"schedule\", &[{}])", list.join(", "))), "{name}");
        for value in values {
            let src = RIP.replace(&line, &format!("const {name}: {ty} = {value};"));
            let text = peacemaker_rip::compile_named(&src, "gemv_mq2_xbatch.rip", Arch::Gfx1201, "xbatch")
                .unwrap_or_else(|e| panic!("{name}={value}: {e}"))
                .s_text;
            assert_eq!(text == reference, *value == old, "{name}={value}");
        }
    }
    // PREFETCH across the generator grid: it only refuses what the grid already refuses.
    for rows in [1u8, 2, 4] {
        for unroll in [1u8, 2, 4] {
            for split in [false, true] {
                for ilp in [false, true] {
                    let p = Params { rows, cap: 8, unroll, split, ilp };
                    let src = with_point(&p).replace("const PREFETCH: u8 = 0;", "const PREFETCH: u8 = 1;");
                    let got = peacemaker_rip::compile_named(&src, "gemv_mq2_xbatch.rip", Arch::Gfx1201, "xbatch");
                    assert_eq!(emit_text(&p).is_ok(), got.is_ok(), "{}: {:?}", p.tag(), got.err());
                }
            }
        }
    }
}
