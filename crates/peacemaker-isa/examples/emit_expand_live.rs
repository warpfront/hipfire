// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `cargo run -p peacemaker-isa --features lab --example emit_expand_live -- BLOCK OUT.co`: emits the
//! standalone gfx1201 `EXPAND_LIVE` producer (`p0_expand_gfx1201::emit(BLOCK)`, BLOCK 32/64/128/256)
//! with the native PeaceMaker builder/allocator/scheduler and native assembler (no hex, no hipcc, no
//! GPU) and writes the code object to `OUT.co`. The output's parent directory must already exist.
use peacemaker_isa::kernels::{iu4_gemm, p0_expand_gfx1201 as ex};
use peacemaker_isa::{native, Arch};
use std::{fs, path::PathBuf};

const USAGE: &str = "usage: emit_expand_live BLOCK OUT.co";

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [block, out] = args.as_slice() else {
        return Err(format!("{USAGE} (got {} arguments)", args.len()));
    };
    let block: u32 = block
        .parse()
        .map_err(|e| format!("{USAGE}: invalid BLOCK {block:?}: {e}"))?;
    let out = PathBuf::from(out);
    let emitted = ex::emit(block).map_err(|e| format!("emit({block}): {e}"))?;
    let (text, _) = iu4_gemm::module(&[emitted], &format!("p0_expand_gfx1201_b{block}"))
        .map_err(|e| format!("module({block}): {e}"))?;
    let elf =
        native::assemble(&text, Arch::Gfx1201).map_err(|e| format!("assemble({block}): {e}"))?;
    fs::write(&out, &elf).map_err(|e| format!("{}: {e}", out.display()))
}
