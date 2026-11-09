// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only emitter of the single-entry HC8 objects the rip-search hardware oracle's smoke job
//! loads (`hc8_rip_oracle`): `hc8_smoke_objects OUTDIR` writes, per variant
//! (`down_n{1,2,3}` = `hc8_gemm_e4m3_mw1_kw16_n{N}_act`, `up_n{1,2,3}` = `hc8_read_up_e4m3_n{N}`),
//! `OUTDIR/<variant>/objects/<id>.co` (bare gfx1201 ELFs from the Rust generator through
//! `native::assemble`) and `OUTDIR/<variant>/manifest.tsv` (rip-search columns `id axes elf_sha256 ...`).
//! Down variants get two candidates (the single-entry module and a two-entry module that also holds
//! the `set` twin, so the ELFs differ); up variants get the single-entry object.

use peacemaker_isa::kernels::hc8_wmma_gfx1201::{self as hc8, Fmt, Geo};
use peacemaker_isa::{native, Arch};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: hc8_smoke_objects OUTDIR")?,
    );
    for nt in 1..=3u32 {
        for kind in ["down", "up"] {
            let name = format!("{kind}_n{nt}");
            let mut elfs: Vec<(String, Vec<u8>, &str)> = Vec::new();
            if kind == "down" {
                let geo = Geo {
                    fmt: Fmt::E4M3,
                    mw: 1,
                    kw: 16,
                    nt,
                    act: true,
                };
                let twin = Geo { act: false, ..geo };
                let (_, text, _) = hc8::emit_geos(&[geo])?;
                elfs.push((
                    format!("{name}-single"),
                    native::assemble(&text, Arch::Gfx1201)?,
                    "single",
                ));
                let (_, text, _) = hc8::emit_geos(&[twin, geo])?;
                elfs.push((
                    format!("{name}-pair"),
                    native::assemble(&text, Arch::Gfx1201)?,
                    "pair",
                ));
            } else {
                let e = hc8::emit_read_up(nt)?;
                elfs.push((
                    format!("{name}-single"),
                    native::assemble(&e.s_text, Arch::Gfx1201)?,
                    "single",
                ));
            }
            let dir = out.join(&name);
            std::fs::create_dir_all(dir.join("objects")).map_err(|e| e.to_string())?;
            let mut manifest = String::from("id\taxes\telf_sha256\n");
            for (id, elf, form) in &elfs {
                std::fs::write(dir.join("objects").join(format!("{id}.co")), elf)
                    .map_err(|e| e.to_string())?;
                let sha: String = Sha256::digest(elf).iter().fold(String::new(), |mut s, b| {
                    let _ = write!(s, "{b:02x}");
                    s
                });
                let _ = writeln!(manifest, "{id}\t{{\"FORM\":\"{form}\"}}\t{sha}");
            }
            std::fs::write(dir.join("manifest.tsv"), manifest).map_err(|e| e.to_string())?;
            println!("{name}: {} objects", elfs.len());
        }
    }
    Ok(())
}
