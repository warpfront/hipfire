// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! TEST ORACLE ONLY: ROCm `llvm-mc` / `llvm-objdump` for the encoding
//! round-trip tests. ROCm is found the way every PeaceMaker oracle finds it
//! (`crates/rocm.rs`: `$ROCM_PATH`, default `/opt/rocm`, tools under
//! `lib/llvm/bin`). Nothing in the production path calls this module.
use crate::arch::Arch;
use crate::Result;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "../../rocm.rs"]
mod rocm;

/// Path of a ROCm LLVM tool (`llvm-mc`, `llvm-objdump`, `ld.lld`, `clang`).
pub fn tool(name: &str) -> PathBuf {
    rocm::llvm(name)
}

/// `HIPFIRE_TEST_REQUIRE_ROCM=0` opts a machine without ROCm out of oracle
/// tests (they then return early with one logged line); otherwise a missing
/// tool is an error.
pub fn disabled() -> bool {
    rocm::tests_disabled()
}

static SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch(ext: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("pm-cdna-oracle-{}-{n}.{ext}", std::process::id()))
}

fn run(cmd: &mut Command) -> Result<(String, String)> {
    let out = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        return Err(format!("{cmd:?} failed ({}):\n{stderr}", out.status));
    }
    Ok((stdout, stderr))
}

/// Collapse runs of whitespace and drop a trailing `;`/`//` comment.
pub fn normalize(text: &str) -> String {
    let code = text
        .split(" ;")
        .next()
        .unwrap_or(text)
        .split("//")
        .next()
        .unwrap_or(text);
    code.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Assemble one instruction per line with `llvm-mc -show-encoding`; returns
/// `(llvm's printed text, encoding bytes)` per input line, in order. Any
/// assembler error fails the whole batch with llvm's diagnostics.
pub fn mc_encode(arch: Arch, lines: &[String]) -> Result<Vec<(String, Vec<u8>)>> {
    let path = scratch("s");
    std::fs::write(&path, lines.join("\n") + "\n").map_err(|e| e.to_string())?;
    let result = run(Command::new(tool("llvm-mc"))
        .args([
            "-arch=amdgcn",
            &format!("-mcpu={}", arch.name()),
            "-show-encoding",
        ])
        .arg(&path));
    let _ = std::fs::remove_file(&path);
    let (stdout, stderr) = result?;
    if stderr.contains("error") {
        return Err(stderr);
    }
    let mut out = Vec::with_capacity(lines.len());
    for line in stdout.lines() {
        let Some((text, enc)) = line.split_once("; encoding: [") else {
            continue;
        };
        let bytes = enc
            .trim_end_matches(']')
            .split(',')
            .map(|b| {
                u8::from_str_radix(b.trim().trim_start_matches("0x"), 16)
                    .map_err(|e| format!("{line}: {e}"))
            })
            .collect::<Result<Vec<u8>>>()?;
        out.push((normalize(text), bytes));
    }
    if out.len() != lines.len() {
        return Err(format!(
            "llvm-mc printed {} encodings for {} lines",
            out.len(),
            lines.len()
        ));
    }
    Ok(out)
}

/// Disassemble each instruction's bytes (one instruction per entry) with
/// `llvm-mc -disassemble` (the disassembler llvm-objdump uses); returns the
/// normalized text per entry.
pub fn mc_disassemble(arch: Arch, encodings: &[Vec<u8>]) -> Result<Vec<String>> {
    let path = scratch("txt");
    let body: String = encodings
        .iter()
        .map(|e| {
            e.iter()
                .map(|b| format!("{b:#04x}"))
                .collect::<Vec<_>>()
                .join(",")
                + "\n"
        })
        .collect();
    std::fs::write(&path, body).map_err(|e| e.to_string())?;
    let result = run(Command::new(tool("llvm-mc"))
        .args([
            "-disassemble",
            "-arch=amdgcn",
            &format!("-mcpu={}", arch.name()),
        ])
        .arg(&path));
    let _ = std::fs::remove_file(&path);
    let (stdout, stderr) = result?;
    if stderr.contains("warning") || stderr.contains("error") {
        return Err(stderr);
    }
    let out: Vec<String> = stdout
        .lines()
        .map(normalize)
        .filter(|l| !l.is_empty() && !l.starts_with('.'))
        .collect();
    if out.len() != encodings.len() {
        return Err(format!(
            "llvm-mc disassembled {} instructions from {} encodings:\n{stdout}",
            out.len(),
            encodings.len()
        ));
    }
    Ok(out)
}

/// `llvm-objdump -d --mcpu=<arch>` of a code object: the normalized
/// instruction text of every instruction line, in address order (address and
/// byte columns stripped).
pub fn objdump(arch: Arch, elf: &[u8]) -> Result<Vec<String>> {
    let path = scratch("co");
    std::fs::write(&path, elf).map_err(|e| e.to_string())?;
    let result = run(Command::new(tool("llvm-objdump"))
        .args([
            "-d",
            &format!("--mcpu={}", arch.name()),
            "--no-show-raw-insn",
            "--no-leading-addr",
        ])
        .arg(&path));
    let _ = std::fs::remove_file(&path);
    let (stdout, _) = result?;
    Ok(stdout
        .lines()
        .filter(|l| l.starts_with(char::is_whitespace) && !l.trim().is_empty())
        .map(normalize)
        .collect())
}

/// Little-endian bytes of dwords.
pub fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
