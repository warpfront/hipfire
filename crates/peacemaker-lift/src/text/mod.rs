// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C3b: canonical objdump text (<->) typed instructions.
//!
//! The typed codec (`peacemaker-ir`) owns bytes; this module owns the
//! independent second opinion: the pinned `llvm-objdump` spelling of every
//! instruction ([`print`]) and the parser that reads it back ([`parse`]).
//! Semantic disagreement with the codec is a test failure, never a
//! normalisation. `llvm-mc` byte parity is required only where the syntax is
//! lossless; instructions carrying don't-care state the syntax cannot spell
//! land in a *reported* set instead of failing.
pub mod parse;
pub mod print;

pub use parse::{parse_line, parse_source, SourceFile, SourceKind, SourceLine};
pub use print::{canonical, kernel_lines, PrintError};

#[cfg(test)]
#[path = "../../../rocm.rs"]
mod rocm;

#[cfg(test)]
pub(crate) mod support {
    use std::path::PathBuf;

    /// Pinned second-opinion toolchain (`PEACEMAKER_ROCM`, else `ROCM_PATH`,
    /// else `/opt/rocm`). `Some` means the live tools must agree;
    /// `None` means the committed objdump fixture is the reference. A
    /// set-but-unusable `PEACEMAKER_ROCM` is a hard error, never a silent skip.
    pub(crate) struct Toolchain {
        pub(crate) objdump: PathBuf,
        pub(crate) mc: PathBuf,
    }

    pub(crate) fn toolchain() -> Result<Option<Toolchain>, String> {
        const BIN: &str = "lib/llvm/bin";
        if let Ok(root) = std::env::var("PEACEMAKER_ROCM") {
            let root = PathBuf::from(root);
            let objdump = root.join(BIN).join("llvm-objdump");
            let mc = root.join(BIN).join("llvm-mc");
            if !objdump.is_file() || !mc.is_file() {
                return Err(format!(
                    "PEACEMAKER_ROCM={} does not hold llvm-objdump/llvm-mc",
                    root.display()
                ));
            }
            return Ok(Some(Toolchain { objdump, mc }));
        }
        // Test hook: exercise the committed-fixture fallback on machines
        // that do have a toolchain.
        if std::env::var("PEACEMAKER_NO_ROCM").is_ok() || super::rocm::tests_disabled() {
            return Ok(None);
        }
        let root = super::rocm::root();
        let objdump = root.join(BIN).join("llvm-objdump");
        let mc = root.join(BIN).join("llvm-mc");
        if objdump.is_file() && mc.is_file() {
            return Ok(Some(Toolchain { objdump, mc }));
        }
        Err(format!(
            "required ROCm tools missing from {}",
            root.display()
        ))
    }

    /// Selected KT48 kernel stream as little-endian words plus base VA.
    /// Same constants as the C3 codec test (fixture pinned by provenance.json).
    pub(crate) fn kt48_stream() -> (Vec<u32>, u32) {
        const IMAGE: &[u8] = include_bytes!("../../tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        const VA: u32 = 0x7f00;
        let words = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (words, VA)
    }
    /// All six KT48 kernel symbols as `(name, base VA, byte size)`, in VA
    /// order. Same constants as the C3 codec six-stream test and the ELF
    /// `STT_FUNC` sizes (fixture pinned by provenance.json); `.text` starts
    /// at VA 0x3400, file offset 0x2400.
    pub(crate) const KT48_SYMBOLS: [(&str, u32, usize); 6] = [
        ("attention_fp8_e4m3_fa2_gqa_gfx1201", 0x3400, 7996),
        (
            "attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201",
            0x5400,
            388,
        ),
        (
            "attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201",
            0x5600,
            1484,
        ),
        ("attention_fp8_e4m3_fa2_gqa_partial_gfx1201", 0x5c00, 8148),
        ("attention_fp8_e4m3_fa2_gqa_merge_gfx1201", 0x7c00, 728),
        (
            "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
            0x7f00,
            10_604,
        ),
    ];

    /// Every KT48 kernel stream as little-endian words: `(name, base VA,
    /// words)` in VA order.
    pub(crate) fn kt48_all_streams() -> Vec<(&'static str, u32, Vec<u32>)> {
        const IMAGE: &[u8] = include_bytes!("../../tests/fixtures/kt48/hipcc.co");
        KT48_SYMBOLS
            .iter()
            .map(|(name, va, size)| {
                const BASE: usize = 0x2400;
                let start = BASE + (*va - 0x3400) as usize;
                let words = IMAGE[start..start + size]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                (*name, *va, words)
            })
            .collect()
    }

    /// One parsed objdump disassembly line.
    pub(crate) struct ObjdumpLine {
        pub(crate) addr: u32,
        pub(crate) words: Vec<u32>,
        pub(crate) text: String,
        pub(crate) ann: String,
    }

    /// Parse `llvm-objdump -d` stdout. Objump comment words are big-endian;
    /// the `//` comment separator may directly follow long lines.
    pub(crate) fn parse_objdump_stdout(s: &str) -> Vec<ObjdumpLine> {
        let mut out = Vec::new();
        for line in s.lines() {
            let body = line.strip_prefix('\t').unwrap_or("");
            if body.is_empty() {
                continue;
            }
            let Some(mark) = body.find("//") else {
                continue;
            };
            let text = body[..mark].trim_end().to_owned();
            if text.is_empty() {
                continue;
            }
            let comment = body[mark + 2..].trim();
            let mut words = Vec::new();
            let mut ann = String::new();
            let mut seen_addr = false;
            let mut addr = 0u32;
            for tok in comment.split_whitespace() {
                if !seen_addr {
                    seen_addr = true;
                    addr = match u32::from_str_radix(tok.trim_end_matches(':'), 16) {
                        Ok(a) => a,
                        Err(_) => continue,
                    };
                    continue;
                }
                if let Some(a) = tok.strip_prefix('<') {
                    ann = format!("<{a}");
                    // Remainder of the comment is the annotation tail.
                    let tail_start = comment.find(tok).unwrap() + tok.len();
                    ann.push_str(&comment[tail_start..].replace(' ', " "));
                    ann = ann.split_whitespace().collect::<Vec<_>>().join(" ");
                    break;
                }
                if tok.len() == 8 {
                    if let Ok(w) = u32::from_str_radix(tok, 16) {
                        words.push(w);
                    }
                }
            }
            if !seen_addr {
                continue;
            }
            out.push(ObjdumpLine {
                addr,
                words,
                text,
                ann,
            });
        }
        out
    }

    /// Committed pinned objdump of the selected symbol (fallback when no
    /// toolchain is present).
    pub(crate) fn pinned_objdump_fixture() -> Vec<ObjdumpLine> {
        const FIXTURE: &str = include_str!("../../tests/fixtures/kt48/hipcc.objdump.txt");
        parse_objdump_stdout(FIXTURE)
    }

    /// Committed pinned objdump of all six KT48 symbols (fallback when no
    /// toolchain is present). Tests filter lines to each symbol's
    /// `[va, va + size)` range; padding outside the ranges is ignored.
    pub(crate) fn pinned_all_objdump_fixture() -> Vec<ObjdumpLine> {
        const FIXTURE: &str = include_str!("../../tests/fixtures/kt48/hipcc.all.objdump.txt");
        parse_objdump_stdout(FIXTURE)
    }

    /// Run the pinned `llvm-mc` over bare canonical lines, returning the
    /// emitted little-endian words per input line in order. Any assembler
    /// error is returned with its line number; the caller decides whether an
    /// error is a failure or (for crafted probes) the expected outcome.
    pub(crate) fn mc_batch(
        mc: &std::path::Path,
        lines: &[String],
    ) -> Result<Vec<Vec<u32>>, String> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("pm-c3b-mc-{}-{id}.s", std::process::id()));
        std::fs::write(&path, lines.join("\n") + "\n")
            .map_err(|e| format!("mc input write: {e}"))?;
        let output = std::process::Command::new(mc)
            .args(["-arch=amdgcn", "-mcpu=gfx1201", "-show-encoding"])
            .arg(&path)
            .output()
            .map_err(|e| format!("spawn llvm-mc: {e}"))?;
        let _ = std::fs::remove_file(&path);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut encodings: Vec<Vec<u32>> = Vec::new();
        for line in stdout.lines() {
            let Some(start) = line.find("; encoding: [") else {
                continue;
            };
            let bytes = line[start + "; encoding: [".len()..]
                .trim_end_matches(']')
                .split(',')
                .map(|b| {
                    u8::from_str_radix(b.trim().trim_start_matches("0x"), 16)
                        .map_err(|_| format!("bad mc encoding byte in {line:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if bytes.len() % 4 != 0 {
                return Err(format!("mc encoding not word-aligned in {line:?}"));
            }
            encodings.push(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            );
        }
        if !output.status.success() {
            return Err(format!(
                "llvm-mc failed ({} of {} lines assembled):\n{stderr}",
                encodings.len(),
                lines.len()
            ));
        }
        if encodings.len() != lines.len() {
            return Err(format!(
                "llvm-mc emitted {} encodings for {} lines",
                encodings.len(),
                lines.len()
            ));
        }
        Ok(encodings)
    }
    /// Bits the canonical syntax cannot spell: don't-care fields plus
    /// modifiers the printer omits (op_sel on rows without 16-bit
    /// operands). `llvm-mc` fills those with its defaults; anything else
    /// must round-trip exactly.
    use peacemaker_ir::{
        codec::gfx12,
        inst::{Arch, Form, FormFields, Inst},
        isa,
    };

    /// Bits the canonical syntax cannot spell (moved here for T9/parser
    /// sharing; see the printer docs for the rule derivations).
    pub(crate) fn lossy_mask(inst: &Inst) -> [u32; 3] {
        let mut mask = [0u32; 3];
        match &inst.fields {
            FormFields::Vop3b { .. } => mask[1] |= 0x1ff << 18,
            FormFields::Bits { ignored, .. } => {
                for field in ignored {
                    match field.name {
                        "src2_unused" => mask[1] |= 0x1ff << 18,
                        "src1_unused" => mask[1] |= 0x1ff << 9,
                        "w0_extra" | "wait_unused" => {
                            if let Some(rule) = isa::lookup(Arch::Gfx1201, inst.op, inst.form)
                                .and_then(|row| row.fields.iter().find(|r| r.name == field.name))
                            {
                                mask[0] |= rule.mask;
                            }
                        }
                        "w1_extra" | "vsrc_unused" => {
                            if let Some(rule) = isa::lookup(Arch::Gfx1201, inst.op, inst.form)
                                .and_then(|row| row.fields.iter().find(|r| r.name == field.name))
                            {
                                mask[1] |= rule.mask;
                            }
                        }
                        _ => {}
                    }
                }
            }
            FormFields::None | FormFields::Vopd { .. } => {}
        }
        // Omitted op_sel modifiers (no 16-bit operand rows print none).
        let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
        let has16 = row.grammar.split(',').any(|part| part.ends_with(":16"));
        if !has16 && (inst.mods.op_sel != 0 || inst.mods.op_sel_hi != 0) {
            match inst.form {
                Form::Vop3 => mask[0] |= 0xf << 11,
                Form::Vop3p => {
                    mask[0] |= 0x7 << 11;
                    mask[1] |= 0x3 << 27;
                    mask[0] |= 0x1 << 14;
                }
                _ => {}
            }
        }
        mask
    }

    #[derive(Debug, PartialEq)]
    pub(crate) enum Parity {
        Exact,
        Reported,
        Mismatch,
    }

    pub(crate) fn classify(inst: &Inst, mc_words: &[u32]) -> Result<Parity, String> {
        let enc = gfx12::encode(inst).map_err(|e| format!("encode failed: {e}"))?;
        if enc.as_slice() == mc_words {
            return Ok(Parity::Exact);
        }
        let mask = lossy_mask(inst);
        let masked_eq = enc.len() == mc_words.len()
            && enc
                .iter()
                .zip(mc_words.iter())
                .enumerate()
                .all(|(i, (a, b))| {
                    let m = mask.get(i).copied().unwrap_or(0);
                    (a ^ b) & !m == 0 && (a ^ b) & m == (a ^ b)
                });
        if masked_eq && !mask.iter().all(|m| *m == 0) {
            // Differs, but only inside unspellable bits: reported, and the
            // semantics must still agree (re-decode check).
            let (back, n) = gfx12::decode(mc_words)
                .map_err(|e| format!("re-decode of mc bytes failed: {e}"))?;
            if n != mc_words.len()
                || back.op != inst.op
                || back.form != inst.form
                || back.operands != inst.operands
                || back.mods != inst.mods
                || back.literal != inst.literal
            {
                return Ok(Parity::Mismatch);
            }
            return Ok(Parity::Reported);
        }
        Ok(Parity::Mismatch)
    }

    /// Modifiers with textually invisible bits cleared (mirrors the
    /// printer omit rule), for semantic comparisons.
    pub(crate) fn masked_mods(inst: &Inst) -> peacemaker_ir::operand::Modifiers {
        let mut mods = inst.mods.clone();
        let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
        let has16 = row.grammar.split(',').any(|part| part.ends_with(":16"));
        if !has16 {
            mods.op_sel = 0;
            mods.op_sel_hi = 0;
        }
        mods
    }
}
