// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of the HC8 compressed-weight GEMM family (`kernels/hc8_wmma.rip` vs
//! `kernels/hc8_wmma_gfx1201.rs` + `hc8_wmma_gfx1201/read_up.rs`): at the declared axis values every one of the
//! six served variants re-emits the generator's per-kernel assembly text byte for byte, and its assembled
//! kernel (machine code and kernel descriptor) equals the same symbol inside the committed
//! `kernels/hc8_wmma_pm_gfx1201.hxaco` (86 exports, compared through the ELF symbol table). The four schedule
//! axes compile or refuse as tabulated below, and every non-declared value changes the text.

use peacemaker_isa::kernels::hc8_wmma_gfx1201::{emit_geos, emit_read_up, Fmt, Geo};
use peacemaker_isa::{Arch, Emitted};

const RIP: &str = include_str!("../../peacemaker-isa/src/kernels/hc8_wmma.rip");
const SHIPPED_HXACO: &[u8] = include_bytes!("../../../kernels/hc8_wmma_pm_gfx1201.hxaco");

/// `(variant, shipped symbol, up, nt)`.
const VARIANTS: [(&str, &str, bool, u32); 6] = [
    ("down_n1", "hc8_gemm_e4m3_mw1_kw16_n1_act", false, 1),
    ("down_n2", "hc8_gemm_e4m3_mw1_kw16_n2_act", false, 2),
    ("down_n3", "hc8_gemm_e4m3_mw1_kw16_n3_act", false, 3),
    ("up_n1", "hc8_read_up_e4m3_n1", true, 1),
    ("up_n2", "hc8_read_up_e4m3_n2", true, 2),
    ("up_n3", "hc8_read_up_e4m3_n3", true, 3),
];

/// The generator's own `Geo::unroll()` per column-tile count.
fn default_unroll(nt: u32) -> u8 {
    match nt {
        1 => 8,
        2 => 4,
        _ => 2,
    }
}

/// The script with the four axis consts rewritten (each is one `const NAME: TY = LIT;` line).
fn with_axes(unroll: &str, loadorder: &str, epipre: &str, delay: &str) -> String {
    let mut src = RIP.to_string();
    for (name, ty, old, new) in [
        ("UNROLL", "u8", "0", unroll),
        ("LOADORDER", "u8", "0", loadorder),
        ("EPIPRE", "bool", "false", epipre),
        ("DELAY_ALU", "bool", "true", delay),
    ] {
        let from = format!("const {name}: {ty} = {old};\n");
        assert_eq!(src.matches(&from).count(), 1, "axis const {name}");
        src = src.replace(&from, &format!("const {name}: {ty} = {new};\n"));
    }
    src
}

/// One axis changed from the declared point.
fn with_axis(name: &str, value: &str) -> String {
    let mut a = ["0", "0", "false", "true"];
    let at = ["UNROLL", "LOADORDER", "EPIPRE", "DELAY_ALU"]
        .iter()
        .position(|n| *n == name)
        .unwrap();
    a[at] = value;
    with_axes(a[0], a[1], a[2], a[3])
}

fn rip(src: &str, variant: &str) -> Result<Emitted, String> {
    peacemaker_rip::compile_named(src, "hc8_wmma.rip", Arch::Gfx1201, variant)
}

/// The generator's emission of one variant.
fn generator(up: bool, nt: u32) -> Emitted {
    if up {
        return emit_read_up(nt).unwrap_or_else(|e| panic!("{e}"));
    }
    let g = Geo {
        fmt: Fmt::E4M3,
        mw: 1,
        kw: 16,
        nt,
        act: true,
    };
    emit_geos(&[g])
        .unwrap_or_else(|e| panic!("{e}"))
        .0
        .remove(0)
}

fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: rust `{x}` vs rip `{y}`", i + 1);
        }
    }
    format!(
        "lengths {} vs {} lines",
        a.lines().count(),
        b.lines().count()
    )
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

/// `(file offset, size)` of the named symbol's bytes, through the ELF64 section and symbol tables.
fn symbol_bytes<'a>(elf: &'a [u8], name: &str) -> (&'a [u8], u64) {
    let u16_ = |o: usize| u16::from_le_bytes(elf[o..o + 2].try_into().unwrap()) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(elf[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(elf[o..o + 8].try_into().unwrap()) as usize;
    assert_eq!(&elf[..4], b"\x7fELF");
    let (shoff, shnum) = (u64_(0x28), u16_(0x3c));
    let sh = |i: usize| shoff + 64 * i;
    let symtab = (0..shnum)
        .find(|&i| u32_(sh(i) + 4) == 2)
        .expect("no .symtab");
    let (st_off, st_size, st_link) = (
        u64_(sh(symtab) + 24),
        u64_(sh(symtab) + 32),
        u32_(sh(symtab) + 40),
    );
    let str_off = u64_(sh(st_link) + 24);
    let mut hit = None;
    for k in 0..st_size / 24 {
        let at = st_off + 24 * k;
        let n = &elf[str_off + u32_(at)..];
        let n = &n[..n.iter().position(|&b| b == 0).unwrap()];
        if n == name.as_bytes() {
            assert!(hit.is_none(), "{name}: two symbols");
            hit = Some(at);
        }
    }
    let at = hit.unwrap_or_else(|| panic!("{name}: no symbol"));
    let (shndx, value, size) = (u16_(at + 6), u64_(at + 8), u64_(at + 16));
    let (s_addr, s_off) = (u64_(sh(shndx) + 16), u64_(sh(shndx) + 24));
    let off = s_off + (value - s_addr);
    (&elf[off..off + size], value as u64)
}

/// Machine code and descriptor of `symbol`: `(code, kd, code_addr, kd_addr)`.
fn kernel_parts<'a>(elf: &'a [u8], symbol: &str) -> (&'a [u8], &'a [u8], u64, u64) {
    let (code, code_addr) = symbol_bytes(elf, symbol);
    let (kd, kd_addr) = symbol_bytes(elf, &format!("{symbol}.kd"));
    (code, kd, code_addr, kd_addr)
}

/// Every variant at the declared point is the generator's text, byte for byte.
#[test]
fn declared_point_is_the_generators_text() {
    for (variant, symbol, up, nt) in VARIANTS {
        let rust = generator(up, nt);
        let got = rip(RIP, variant).unwrap_or_else(|e| panic!("{variant}: {e}"));
        assert!(
            rust.s_text == got.s_text,
            "{variant}: {}",
            first_difference(&rust.s_text, &got.s_text)
        );
        assert!(got.s_text.contains(&format!(".amdhsa_kernel {symbol}\n")));
    }
}

/// The declared point assembles to the committed `.hxaco`'s kernel: the machine code bytes of the symbol and
/// its 64-byte descriptor equal those of the same symbol in the 86-export module ELF. The descriptor's
/// `kernel_code_entry_byte_offset` (bytes 16..24) is the one field that depends on where the kernel sits in
/// `.text`; it is compared as "points at the symbol's code" in each ELF instead of as a number.
#[test]
fn declared_point_is_the_shipped_artifact() {
    let module = bundle_elf(SHIPPED_HXACO);
    for (variant, symbol, ..) in VARIANTS {
        let got = rip(RIP, variant).unwrap_or_else(|e| panic!("{variant}: {e}"));
        let elf = peacemaker_isa::native::assemble(&got.s_text, Arch::Gfx1201)
            .unwrap_or_else(|e| panic!("{variant}: {e}"));
        let (code, kd, code_addr, kd_addr) = kernel_parts(&elf, symbol);
        let (m_code, m_kd, m_code_addr, m_kd_addr) = kernel_parts(module, symbol);
        assert_eq!(kd.len(), 64, "{symbol}.kd");
        assert_eq!(m_kd.len(), 64, "{symbol}.kd in the module");
        assert!(!code.is_empty(), "{symbol}: empty code");
        assert!(
            code == m_code,
            "{symbol}: machine code differs ({} vs {} bytes)",
            code.len(),
            m_code.len()
        );
        let entry = |kd: &[u8]| i64::from_le_bytes(kd[16..24].try_into().unwrap());
        assert_eq!(
            kd_addr.wrapping_add(entry(kd) as u64),
            code_addr,
            "{symbol}"
        );
        assert_eq!(
            m_kd_addr.wrapping_add(entry(m_kd) as u64),
            m_code_addr,
            "{symbol} in the module"
        );
        let mask = |kd: &[u8]| {
            let mut k = kd.to_vec();
            k[16..24].fill(0);
            k
        };
        assert!(
            mask(kd) == mask(m_kd),
            "{symbol}: kernel descriptor differs outside the entry offset"
        );
    }
}

/// The axis consts are single lines at the declared values; `SEARCH_AXES` lists exactly the four axes.
#[test]
fn axis_consts_are_single_lines_at_the_declared_point() {
    for line in [
        "const UNROLL: u8 = 0;",
        "const LOADORDER: u8 = 0;",
        "const EPIPRE: bool = false;",
        "const DELAY_ALU: bool = true;",
    ] {
        assert_eq!(RIP.lines().filter(|l| *l == line).count(), 1, "{line}");
    }
    let start = RIP.find("const SEARCH_AXES").expect("SEARCH_AXES");
    let block = &RIP[start..];
    let block = &block[..block.find("];").unwrap()];
    let rows: Vec<&str> = block.lines().skip(1).collect();
    assert_eq!(
        rows,
        [
            "    (\"UNROLL\", \"schedule\", &[\"0\", \"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"8\", \"10\"]),",
            "    (\"LOADORDER\", \"schedule\", &[\"0\", \"1\", \"2\"]),",
            "    (\"EPIPRE\", \"schedule\", &[\"false\", \"true\"]),",
            "    (\"DELAY_ALU\", \"schedule\", &[\"true\", \"false\"]),",
        ]
    );
    assert!(RIP.contains(
        "const VARIANTS: &[&str] = &[\"down_n1\", \"down_n2\", \"down_n3\", \"up_n1\", \"up_n2\", \"up_n3\"];"
    ));
}

/// The `(variant, UNROLL)` points the script refuses because the register layout does not fit (`Lay::new`:
/// more than 256 VGPRs): n1 fits every listed value, n2 up to 5, n3 up to 3.
fn unroll_refused(nt: u32, unroll: u32) -> bool {
    let u = if unroll == 0 {
        u32::from(default_unroll(nt))
    } else {
        unroll
    };
    let xb = 40 + 8 * nt;
    xb + 16 * u * nt + 4 * u > 256
}

const UNROLLS: [&str; 9] = ["0", "1", "2", "3", "4", "5", "6", "8", "10"];

/// UNROLL: refused exactly when the layout does not fit, otherwise it compiles; it changes the text exactly
/// when it differs from the generator's default for the variant, and the generator's default value spelled
/// out is the declared text.
#[test]
fn unroll_compiles_or_refuses_as_tabulated() {
    for (variant, _, up, nt) in VARIANTS {
        let reference = rip(RIP, variant).unwrap().s_text;
        let mut line = format!("{variant}:");
        for unroll in UNROLLS {
            let n: u32 = unroll.parse().unwrap();
            let got = rip(&with_axis("UNROLL", unroll), variant);
            assert_eq!(
                got.is_err(),
                unroll_refused(nt, n),
                "{variant} UNROLL={unroll}: {:?}",
                got.as_ref().err()
            );
            match got {
                Ok(e) => {
                    let same = n == 0 || n == u32::from(default_unroll(nt));
                    assert_eq!(e.s_text == reference, same, "{variant} UNROLL={unroll}");
                    line.push_str(&format!(" {unroll}:ok"));
                }
                Err(e) => {
                    assert!(e.contains("VGPRs"), "{variant} UNROLL={unroll}: {e}");
                    line.push_str(&format!(" {unroll}:refused"));
                }
            }
        }
        eprintln!("{line}");
        let _ = up;
    }
}

/// LOADORDER: every value compiles for every variant; a non-declared value changes the text.
#[test]
fn loadorder_compiles_and_changes_the_text() {
    for (variant, ..) in VARIANTS {
        let reference = rip(RIP, variant).unwrap().s_text;
        for value in ["0", "1", "2"] {
            let got = rip(&with_axis("LOADORDER", value), variant)
                .unwrap_or_else(|e| panic!("{variant} LOADORDER={value}: {e}"));
            assert_eq!(got.s_text == reference, value == "0", "{variant} {value}");
        }
    }
}

/// DELAY_ALU: both values compile; false changes the text.
#[test]
fn delay_alu_compiles_and_changes_the_text() {
    for (variant, ..) in VARIANTS {
        let reference = rip(RIP, variant).unwrap().s_text;
        for value in ["true", "false"] {
            let got = rip(&with_axis("DELAY_ALU", value), variant)
                .unwrap_or_else(|e| panic!("{variant} DELAY_ALU={value}: {e}"));
            assert_eq!(
                got.s_text == reference,
                value == "true",
                "{variant} {value}"
            );
        }
    }
}

/// The `(variant, UNROLL)` points that fit the layout but whose EPIPRE block does not: the dedicated registers
/// start at the 8-aligned end of `max(layout top, epilogue scratch end)` and need 8 VGPRs (down) or
/// `8 + 8*NT + NT` (up: scales, normalized values, per-column offsets).
fn epipre_refused(up: bool, nt: u32, unroll: u32) -> bool {
    let u = if unroll == 0 {
        u32::from(default_unroll(nt))
    } else {
        unroll
    };
    let xb = 40 + 8 * nt;
    let top = xb + 16 * u * nt + 4 * u;
    let scratch = if up { xb + 17 } else { xb + 32 };
    let pre = top.max(scratch).div_ceil(8) * 8;
    pre + if up { 8 + 9 * nt } else { 8 } > 256
}

/// EPIPRE: false is the declared point; true changes the text and compiles for every (variant, UNROLL) the
/// layout admits except where the dedicated registers do not fit in 256 VGPRs.
#[test]
fn epipre_compiles_or_refuses_as_tabulated() {
    let mut extra = Vec::new();
    for (variant, _, up, nt) in VARIANTS {
        let reference = rip(RIP, variant).unwrap().s_text;
        let off = rip(&with_axis("EPIPRE", "false"), variant).unwrap().s_text;
        assert!(off == reference, "{variant}");
        for unroll in UNROLLS {
            let n: u32 = unroll.parse().unwrap();
            let src = with_axes(unroll, "0", "true", "true");
            let got = rip(&src, variant);
            let layout = unroll_refused(nt, n);
            assert_eq!(
                got.is_err(),
                layout || epipre_refused(up, nt, n),
                "{variant} UNROLL={unroll} EPIPRE=true: {:?}",
                got.as_ref().err()
            );
            match got {
                Ok(e) => {
                    let plain = rip(&with_axes(unroll, "0", "false", "true"), variant).unwrap();
                    assert!(
                        e.s_text != plain.s_text,
                        "{variant} UNROLL={unroll}: EPIPRE=true changes nothing"
                    );
                }
                Err(e) => {
                    assert!(e.contains("VGPRs"), "{variant} UNROLL={unroll}: {e}");
                    if !layout {
                        extra.push(format!("{variant}:{unroll}"));
                    }
                }
            }
        }
    }
    eprintln!("EPIPRE=true refused beyond the layout: {extra:?}");
    assert_eq!(extra, ["up_n1:10", "up_n2:5", "up_n3:3"]);
}

/// Axis values combine: every point of the four-axis product that fits compiles (a sample: all LOADORDER and
/// DELAY_ALU values at both EPIPRE values and three UNROLL values per variant).
#[test]
fn axis_product_compiles() {
    for (variant, _, up, nt) in VARIANTS {
        for unroll in ["1", "3", "0"] {
            for order in ["0", "1", "2"] {
                for epipre in ["false", "true"] {
                    for delay in ["true", "false"] {
                        let n: u32 = unroll.parse().unwrap();
                        let fits = !unroll_refused(nt, n)
                            && !(epipre == "true" && epipre_refused(up, nt, n));
                        let got = rip(&with_axes(unroll, order, epipre, delay), variant);
                        assert_eq!(
                            got.is_ok(),
                            fits,
                            "{variant} U{unroll} O{order} E{epipre} D{delay}: {:?}",
                            got.err()
                        );
                    }
                }
            }
        }
    }
}

/// The generator spells a few operands as `llvm-objdump` never prints them back (`v_perm_b32 .., 0x07060302`,
/// `offset:0`), which the toolchain's parse-back rejects. `peacemaker_rip::objdump_spelling` normalises exactly
/// those for the certify stage; the object assembled from either spelling is the same ELF.
#[cfg(feature = "certify")]
#[test]
fn objdump_spelling_assembles_to_the_same_elf() {
    for (variant, symbol, ..) in VARIANTS {
        let text = rip(RIP, variant).unwrap().s_text;
        let canon = peacemaker_rip::objdump_spelling(&text);
        assert!(canon != text, "{variant}: nothing to normalise?");
        assert!(canon.contains("0x7060302") && !canon.contains("0x07060302"));
        assert!(!canon.contains(" offset:0\n"), "{symbol}");
        let a = peacemaker_isa::native::assemble(&text, Arch::Gfx1201).unwrap();
        let b = peacemaker_isa::native::assemble(&canon, Arch::Gfx1201).unwrap();
        assert!(a == b, "{variant}: the two spellings assemble differently");
    }
}

/// The declared point of every variant certifies with the existing passes (wait replay, `llvm-mc` + `ld.lld`
/// + objdump parse-back, M7 byte-exact lift with no obligations). Skipped where the ROCm tools are absent.
#[cfg(feature = "certify")]
#[test]
fn declared_points_certify() {
    let tc = peacemaker_isa::toolchain::Toolchain::default();
    if !tc.llvm_mc.is_file() || !tc.linker.is_file() || !tc.objdump.is_file() {
        eprintln!("skipped: no ROCm llvm tools at {}", tc.llvm_mc.display());
        return;
    }
    let dir = std::env::temp_dir().join(format!("hc8_wmma_twin_certify_{}", std::process::id()));
    for (variant, symbol, ..) in VARIANTS {
        let e = rip(RIP, variant).unwrap();
        let certs = peacemaker_rip::certify(
            std::slice::from_ref(&e),
            &e.s_text,
            Arch::Gfx1201,
            variant,
            &dir.join(variant),
        )
        .unwrap_or_else(|err| panic!("{variant}: {err}"));
        assert_eq!(certs.len(), 1);
        assert_eq!(certs[0].symbol, symbol);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
