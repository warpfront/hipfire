// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab, CPU only: emit the bounded MQ2 x-batch PM variant grid, or the shipped artifact.
//!
//! `mq2_xbatch_pm_sweep OUT_DIR`: every tuple of `rows {1,2,4} x cap 8 x unroll {1,2,4} x split {0,1} x ilp {0,1}`
//! (canonical order). Each tuple the emitter accepts gets `OUT_DIR/<tag>.hxaco` (the `emit_bundle` bytes); each tuple
//! it rejects (emitter check, or `ledger_cfg::verify_emitted` on the emitted program) is recorded, with the error, and never weakened. `OUT_DIR/manifest.tsv` is tab separated: a
//! header line, then `tag rows cap unroll split ilp top legal object error` per tuple. `split`, `ilp` and `legal` are
//! `0`/`1`; `top` is `vgpr_top` (`0` when illegal); `object` is the file name relative to `OUT_DIR` (`-` when
//! illegal); `error` is the emitter message (`-` when legal, tabs and newlines flattened to spaces).
//!
//! `mq2_xbatch_pm_sweep --shipped OUT_DIR`: emit the current `SHIPPED` tuple as
//! `OUT_DIR/gemv_mq2_xbatch_pm_gfx1201.hxaco` plus `.hxaco.sha256` (hex, no newline, the committed format),
//! the device `.co`, and print the tag and SHA-256 of the bundle and ELF for the artifact pins.
//!
//! Run with `--features lab`.

use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201::{
    emit_bundle, emit_elf, verify_waits, vgpr_top, Params, SHIPPED, XBATCH_MAX,
};
use sha2::{Digest, Sha256};
use std::{fmt::Write as _, fs, path::PathBuf};

fn flat(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let shipped = args.first().is_some_and(|a| a == "--shipped");
    if shipped {
        args.remove(0);
    }
    let out = PathBuf::from(args.first().ok_or("usage: mq2_xbatch_pm_sweep [--shipped] OUT_DIR")?);
    fs::create_dir_all(&out)?;
    if shipped {
        let bundle = emit_bundle(&SHIPPED)?;
        let elf = emit_elf(&SHIPPED)?;
        let bundle_sha = format!("{:x}", Sha256::digest(&bundle));
        fs::write(out.join("gemv_mq2_xbatch_pm_gfx1201.hxaco"), &bundle)?;
        fs::write(out.join("gemv_mq2_xbatch_pm_gfx1201.hxaco.sha256"), &bundle_sha)?;
        fs::write(out.join("gemv_mq2_xbatch_pm_gfx1201.co"), &elf)?;
        println!("tag {}", SHIPPED.tag());
        println!("bundle_sha256 {bundle_sha}");
        println!("elf_sha256 {:x}", Sha256::digest(&elf));
        return Ok(());
    }
    let mut manifest = String::from("tag\trows\tcap\tunroll\tsplit\tilp\ttop\tlegal\tobject\terror\n");
    let (mut legal, mut total) = (0usize, 0usize);
    for rows in [1u8, 2, 4] {
        for unroll in [1u8, 2, 4] {
            for split in [false, true] {
                for ilp in [false, true] {
                    let p = Params {
                        rows,
                        cap: XBATCH_MAX,
                        unroll,
                        split,
                        ilp,
                    };
                    total += 1;
                    let tag = p.tag();
                    let emitted = emit_bundle(&p).and_then(|bundle| {
                        verify_waits(&p).map_err(|e| format!("ledger_cfg: {e}"))?;
                        Ok(bundle)
                    });
                    let (top, ok, object, error) = match emitted {
                        Ok(bundle) => {
                            let object = format!("{tag}.hxaco");
                            fs::write(out.join(&object), bundle)?;
                            legal += 1;
                            (vgpr_top(&p), 1, object, "-".to_string())
                        }
                        Err(e) => (0, 0, "-".to_string(), flat(&e)),
                    };
                    writeln!(
                        manifest,
                        "{tag}\t{rows}\t{}\t{unroll}\t{}\t{}\t{top}\t{ok}\t{object}\t{error}",
                        p.cap,
                        u8::from(split),
                        u8::from(ilp)
                    )?;
                }
            }
        }
    }
    fs::write(out.join("manifest.tsv"), manifest)?;
    println!("{legal}/{total} tuples legal; manifest {}", out.join("manifest.tsv").display());
    Ok(())
}
