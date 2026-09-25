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
    let (proof_digest, builder_git_sha) = if let Some(path) = proof.as_ref() {
        let proof_bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let source_bytes = fs::read(&source).map_err(|e| e.to_string())?;
        proof_binding(&proof_bytes, &source_bytes, &arch)?
    } else { (String::new(), String::new()) };
    let mut toolchain = Toolchain::default();
    if let Some(target) = host_target { toolchain.host_target = target; }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let build = assemble_link_bundle(&toolchain, &source, &output, &arch)?;
    certify(&toolchain, &build, &source, &arch, &manifest, contract.as_ref(),
        &proof_digest, &builder_git_sha)?;
    if let Some(symbol) = descriptor {
        println!("{:?}", read_kd(&build.elf, &symbol)?);
    }
    println!("{}", output.display());
    Ok(())
}

fn proof_binding(proof: &[u8], source: &[u8], arch: &str) -> Result<(String, String), String> {
    let decoded: serde_json::Value = serde_json::from_slice(proof).map_err(|e| e.to_string())?;
    let source_sha = format!("{:x}", Sha256::digest(source));
    if decoded["s_text_sha256"].as_str() != Some(source_sha.as_str())
        || decoded["arch"].as_str() != Some(arch)
        || decoded["builder_crate_version"].as_str() != Some(env!("CARGO_PKG_VERSION")) {
        return Err("builder proof does not bind the emitted source, architecture and crate version".into());
    }
    let builder_sha = decoded["builder_git_sha"].as_str()
        .filter(|sha| !sha.is_empty()).ok_or("builder proof has no git SHA")?;
    Ok((format!("{:x}", Sha256::digest(proof)), builder_sha.to_owned()))
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_cannot_certify_another_source_or_architecture() {
        let source = b"s_endpgm\n";
        let proof = serde_json::json!({
            "s_text_sha256": format!("{:x}", Sha256::digest(source)),
            "arch": "gfx1201",
            "builder_crate_version": env!("CARGO_PKG_VERSION"),
            "builder_git_sha": "abc123",
        });
        let encoded = serde_json::to_vec(&proof).unwrap();
        assert!(proof_binding(&encoded, source, "gfx1201").is_ok());
        assert!(proof_binding(&encoded, b"s_nop 0\ns_endpgm\n", "gfx1201").is_err());
        assert!(proof_binding(&encoded, source, "gfx1100").is_err());
    }
}
