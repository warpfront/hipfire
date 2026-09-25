// SPDX-License-Identifier: Apache-2.0
use hipfire_isa::toolchain::{assemble_link_bundle, certify, read_kd, IsaShapeContract, Toolchain};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("custom") || args.next().as_deref() != Some("build") {
        return Err("usage: peacemaker custom build --arch gfx1201 --s file.s --out file.hsaco --manifest file.json [--contract shape.json --proof proof.json] [--host-target triple]".into());
    }
    let mut arch = None;
    let mut source = None;
    let mut output = None;
    let mut manifest = None;
    let mut contract = None;
    let mut proof = None;
    let mut host_target = None;
    let mut descriptor = None;
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--arch" => arch = Some(value),
            "--s" => source = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--contract" => contract = Some(PathBuf::from(value)),
            "--proof" => proof = Some(PathBuf::from(value)),
            "--host-target" => host_target = Some(value),
            "--kd" => descriptor = Some(value),
            _ => return Err(format!("unrecognized option {option}")),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let source = source.ok_or("missing --s")?;
    let output = output.ok_or("missing --out")?;
    let manifest = manifest.ok_or("missing --manifest")?;
    let contract = contract.map(|path| fs::read(&path)
        .map_err(|e| format!("{}: {e}", path.display()))
        .and_then(|bytes| serde_json::from_slice::<IsaShapeContract>(&bytes).map_err(|e| e.to_string())))
        .transpose()?;
    if contract.is_some() && proof.is_none() {
        return Err("custom shape certification requires --proof".into());
    }
    let proof_digest = proof.as_ref().map(|path| fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(&bytes)))
        .map_err(|e| e.to_string())).transpose()?.unwrap_or_default();
    let mut toolchain = Toolchain::default();
    if let Some(target) = host_target { toolchain.host_target = target; }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let build = assemble_link_bundle(&toolchain, &source, &output, &arch)?;
    certify(&toolchain, &build, &source, &arch, &manifest, contract.as_ref(),
        &proof_digest, &git_sha())?;
    if let Some(symbol) = descriptor {
        println!("{:?}", read_kd(&build.elf, &symbol)?);
    }
    println!("{}", output.display());
    Ok(())
}

fn git_sha() -> String {
    std::process::Command::new("git").args(["rev-parse", "HEAD"])
        .output().ok().filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|sha| sha.trim().to_owned()).unwrap_or_else(|| "unknown".into())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}
