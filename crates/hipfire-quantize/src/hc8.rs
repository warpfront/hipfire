// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize hc8 build`: offline, no-GPU build of a `<model>.hc8` sidecar
//! (OCP E4M3, per-row power-of-two scale) for the Flash-Next hyper-connection mixer
//! BF16 weights of a base model.
//!
//! The container, binding and self-check are documented in
//! `hipfire_runtime::hc8_sidecar`; this is only the command-line wrapper.

use clap::{Parser, Subcommand};
use hipfire_runtime::hc8_sidecar::write_hc8_sidecar;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "hipfire-quantize hc8", about = "Flash-Next HC8 sidecar tools")]
pub(crate) struct Hc8Cli {
    #[command(subcommand)]
    pub command: Hc8Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Hc8Command {
    /// Build a `<model>.hc8` sidecar of 8-bit hyper-connection mixer weights.
    ///
    /// Every BF16 `*hyper_connection*.input_mix_weight_{down,up}.weight` tensor of the
    /// base is requantized to OCP E4M3 with a per-output-row power-of-two f32 scale
    /// (the dequantized weight is exactly a BF16). The sidecar binds to the base's
    /// file length and index digest and is re-bound and re-read as a self-check. The
    /// report is printed as JSON on stdout.
    Build(BuildArgs),
}

#[derive(Debug, Parser)]
pub(crate) struct BuildArgs {
    /// Base HFQM model file the sidecar binds to.
    #[arg(long, value_name = "BASE.hfq")]
    pub model: PathBuf,

    /// Destination sidecar (default: `<model file name>.hc8`).
    #[arg(long, value_name = "FILE.hc8")]
    pub out: Option<PathBuf>,

    /// Replace an existing destination file.
    #[arg(long)]
    pub force: bool,
}

/// Entry point for `hipfire-quantize hc8 …` (`argv` excludes the program name and the
/// `hc8` word). Returns the process exit code.
pub(crate) fn run(argv: Vec<String>) -> i32 {
    let cli = Hc8Cli::parse_from(std::iter::once("hipfire-quantize hc8".to_string()).chain(argv));
    let result = match &cli.command {
        Hc8Command::Build(args) => run_build(args),
    };
    match result {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(e) => {
            eprintln!("error: hc8 build: {e}");
            1
        }
    }
}

fn run_build(args: &BuildArgs) -> Result<String, String> {
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| hipfire_config::memory::hc8_sidecar_path(&args.model));
    if args.force && out.exists() {
        std::fs::remove_file(&out)
            .map_err(|e| format!("remove existing {}: {e}", out.display()))?;
    }
    let report = write_hc8_sidecar(&args.model, &out)?;
    serde_json::to_string_pretty(&report).map_err(|e| format!("serialize report: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_build_defaults() {
        let cli = Hc8Cli::try_parse_from(["hc8", "build", "--model", "m.mq4-xts"]).unwrap();
        let Hc8Command::Build(args) = cli.command;
        assert_eq!(args.model, PathBuf::from("m.mq4-xts"));
        assert!(args.out.is_none());
        assert!(!args.force);
    }

    #[test]
    fn parses_build_out_and_force() {
        let cli =
            Hc8Cli::try_parse_from(["hc8", "build", "--model", "m", "--out", "o.hc8", "--force"])
                .unwrap();
        let Hc8Command::Build(args) = cli.command;
        assert_eq!(args.out, Some(PathBuf::from("o.hc8")));
        assert!(args.force);
    }

    #[test]
    fn requires_model() {
        assert!(Hc8Cli::try_parse_from(["hc8", "build"]).is_err());
    }
}
