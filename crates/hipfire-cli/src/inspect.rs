// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire inspect <file.gguf>`: read a GGUF header and report what hipfire
//! would do with it: architecture, tensor count, per-type histogram, model
//! bits per weight, and the per-tensor admission verdict. Read-only (no GPU,
//! no daemon, no model load), so it works on a header-only download too.

use anyhow::{bail, Context, Result};
use clap::Args;
use std::io::Read;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub(crate) struct InspectArgs {
    /// A `.gguf` model file.
    pub path: PathBuf,
}

/// The four GGUF magic bytes.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

pub(crate) fn inspect_command(args: InspectArgs) -> Result<()> {
    let mut magic = [0u8; 4];
    std::fs::File::open(&args.path)
        .with_context(|| format!("could not open {}", args.path.display()))?
        .read_exact(&mut magic)
        .with_context(|| format!("{} is too short to be a model file", args.path.display()))?;
    if &magic != GGUF_MAGIC {
        bail!(
            "{} is not a GGUF file; `hipfire inspect` reads GGUF files only",
            args.path.display()
        );
    }
    let report = hipfire_loader::gguf_native::inspect_report(
        &args.path,
        hipfire_loader::gguf_native::gguf_native_enabled(),
    )
    .map_err(anyhow::Error::msg)?;
    print!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_gguf_file_is_named_and_refused() {
        let dir = std::env::temp_dir().join(format!("hipfire-inspect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not-gguf.bin");
        std::fs::write(&path, b"HFQM....").unwrap();
        let err = inspect_command(InspectArgs { path: path.clone() }).unwrap_err();
        assert!(err.to_string().contains("not a GGUF file"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
