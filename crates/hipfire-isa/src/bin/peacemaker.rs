// SPDX-License-Identifier: Apache-2.0
use hipfire_isa::audit::{self, Options as AuditOptions};
use hipfire_isa::toolchain::{build, certify, read_kd, IsaShapeContract, Toolchain};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else { return Err(usage().into()); };
    if command == "audit" { return run_audit(args); }
    if command == "profile" { return run_profile(args); }
    if command == "lint" { return run_lint(args); }
    if command == "native" { return run_native(args); }
    if command != "custom" || args.next().as_deref() != Some("build") {
        return Err(usage().into());
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
    let build = build(&toolchain, &source, &output, &arch)?;
    certify(&toolchain, &build, &source, &arch, &manifest, contract.as_ref(),
        &proof_digest, &builder_git_sha)?;
    if let Some(symbol) = descriptor {
        println!("{:?}", read_kd(&build.elf, &symbol)?);
    }
    println!("{}", output.display());
    Ok(())
}

fn usage() -> &'static str {
    "usage: peacemaker custom build --arch gfx1201 --s file.s --out file.hsaco --manifest file.json [--contract shape.json --proof proof.json] [--host-target triple]\n\
     peacemaker native --arch=gfx1201 file.s [--co=file.co] [--bundle=file.hxaco] [--host-target=triple]   (no ROCm: the code object and bundle `custom build` emits, uncertified)\n\
     peacemaker audit --arch gfx1201 (--source file.hip | --hsaco file.hsaco) [--prepend header.hip] [--define NAME=VALUE] [--flag FLAG] [--intent intent.json] [--json report.json] [--markdown report.md] [--sweep-profiles]\n\
     peacemaker audit --lds-barrier PATH...   (CPU-only: every *.hsaco/*.co under PATH; fails on lds_store_unwaited_at_barrier or an unliftable module)\n\
     peacemaker profile --arch gfx1201|gfx1100|gfx1151 --s module.s --points points.json --out diag.hsaco   (DIAGNOSTIC build: writes diag.s, diag.map.json, diag.co)\n\
     peacemaker lint OBJECT.co [--routes routes.json]   (CPU-only: JSON schedule report per lifted kernel/loop)"
}

/// The native build alone, spelled the way `toolchain::build` records it.
fn run_native(args: impl Iterator<Item=String>) -> Result<(), String> {
    let (mut arch, mut source, mut co, mut bundle) = (None, None, None, None);
    let mut host = hipfire_isa::native::DEFAULT_HOST_TARGET.to_owned();
    for arg in args {
        match arg.split_once('=') {
            Some(("--arch", v)) => arch = Some(v.parse::<hipfire_isa::Arch>()?),
            Some(("--co", v)) => co = Some(PathBuf::from(v)),
            Some(("--bundle", v)) => bundle = Some(PathBuf::from(v)),
            Some(("--host-target", v)) => host = v.to_owned(),
            _ if !arg.starts_with("--") && source.is_none() => source = Some(PathBuf::from(arg)),
            _ => return Err(format!("unrecognized native argument {arg}\n{}", usage())),
        }
    }
    let (arch, source) = (arch.ok_or("missing --arch=")?, source.ok_or("missing source .s")?);
    let text = fs::read_to_string(&source).map_err(|e| format!("{}: {e}", source.display()))?;
    let elf = hipfire_isa::native::assemble(&text, arch)?;
    if let Some(path) = &co { fs::write(path, &elf).map_err(|e| format!("{}: {e}", path.display()))? }
    if let Some(path) = &bundle { fs::write(path, hipfire_isa::native::bundle(&elf, arch, &host)).map_err(|e| format!("{}: {e}", path.display()))? }
    Ok(())
}

fn run_lint(mut args: impl Iterator<Item=String>) -> Result<(), String> {
    use hipfire_isa::cost_lint::{self, Route};
    use std::collections::BTreeMap;
    let path = args.next().ok_or_else(|| usage().to_owned())?;
    let mut routes = BTreeMap::<String, Route>::new();
    if let Some(option) = args.next() {
        if option != "--routes" { return Err(format!("unknown lint option {option}")); }
        let file = args.next().ok_or("missing --routes path")?;
        routes = serde_json::from_slice(&fs::read(&file).map_err(|e| format!("{file}: {e}"))?)
            .map_err(|e| format!("{file}: {e}"))?;
    }
    if args.next().is_some() { return Err(usage().into()); }
    let object = fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let lifted = peacemaker_lift::lift_object(&object, peacemaker_lift::Options {
        frontend: peacemaker_ir::inst::Frontend::Hipcc,
    }).map_err(|e| format!("{path}: {e}"))?;
    let report = cost_lint::analyze(&lifted.program, &routes)?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}

/// DIAGNOSTIC: instrument one kernel with timestamp records, verify the
/// rewrite, and assemble a separate object whose symbol carries the
/// `__pm_profile` suffix. The output is never a production artifact.
fn run_profile(mut args: impl Iterator<Item=String>) -> Result<(), String> {
    use hipfire_isa::{ledger_replay::replay_hazards, profile};
    let (mut arch, mut source, mut points, mut output) = (None, None, None, None);
    while let Some(option) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--arch" => arch = Some(value),
            "--s" => source = Some(PathBuf::from(value)),
            "--points" => points = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unrecognized profile option {option}\n{}", usage())),
        }
    }
    let arch = arch.ok_or("missing --arch")?;
    let target: hipfire_isa::Arch = arch.parse()?;
    let source_path = source.ok_or("missing --s")?;
    let output = output.ok_or("missing --out")?;
    let read = |p: &PathBuf| fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    let source = read(&source_path)?;
    let config: profile::Config = serde_json::from_str(&read(&points.ok_or("missing --points")?)?)
        .map_err(|e| format!("points: {e}"))?;
    let (text, map) = profile::instrument(&source, &config, target)?;
    profile::verify(&source, &text, &map)?;
    // The independent wait replay must find exactly the original's hazards
    // (none for builder kernels; hipcc's interlocked DS-source reuse otherwise).
    let original_replay = replay_hazards(&profile::kernel_body(&source, &map.symbol)?, target)?;
    let profiled_replay = replay_hazards(&profile::kernel_body(&text, &map.profiled_symbol)?, target)?;
    if original_replay != profiled_replay {
        return Err(format!("wait replay: profiled kernel has {} hazards, original {}", profiled_replay.len(), original_replay.len()));
    }
    if let Some(parent) = output.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let s_path = output.with_extension("s");
    fs::write(&s_path, &text).map_err(|e| e.to_string())?;
    let build = build(&Toolchain::default(), &s_path, &output, &arch)?;
    let mut json = serde_json::to_value(&map).map_err(|e| e.to_string())?;
    json["source"] = source_path.display().to_string().into();
    json["source_sha256"] = format!("{:x}", Sha256::digest(source.as_bytes())).into();
    json["profiled_s_sha256"] = format!("{:x}", Sha256::digest(text.as_bytes())).into();
    json["profiled_co_sha256"] = format!("{:x}", Sha256::digest(fs::read(&build.elf).map_err(|e| e.to_string())?)).into();
    json["wait_replay_hazards_original"] = original_replay.len().into();
    json["wait_replay_hazards_profiled"] = profiled_replay.len().into();
    json["parse_back"] = if build.parse_back_checked { "pass" } else { "skipped: llvm-objdump absent" }.into();
    let map_path = output.with_extension("map.json");
    fs::write(&map_path, serde_json::to_vec_pretty(&json).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    println!("DIAGNOSTIC {} sites={} delay_fixups={} vgpr {}->{} (limit {}) sgpr {}->{} kernarg {}+{}\n{}\n{}\n{}",
        map.profiled_symbol, map.sites.len(), map.delay_fixups.len(), map.vgpr_before, map.vgpr_after, map.vgpr_limit,
        map.sgpr_before, map.sgpr_after, map.kernarg_ext_offset, map.kernarg_ext_bytes,
        s_path.display(), build.elf.display(), map_path.display());
    Ok(())
}

fn run_audit(args: impl Iterator<Item=String>) -> Result<(), String> {
    let mut args = args.peekable();
    if args.next_if_eq("--lds-barrier").is_some() { return run_lds_barrier(args.map(PathBuf::from).collect()); }
    let mut options = AuditOptions::default();
    while let Some(option) = args.next() {
        if option == "--sweep-profiles" { options.sweep_profiles = true; continue; }
        let value = args.next().ok_or_else(|| format!("missing value for {option}"))?;
        match option.as_str() {
            "--arch" => options.arch = value,
            "--source" | "--hsaco" | "--input" => options.input = PathBuf::from(value),
            "--prepend" => options.prepend.push(PathBuf::from(value)),
            "--define" => options.defines.push(value),
            "--flag" => options.flags.push(value),
            "--intent" => options.intent = Some(PathBuf::from(value)),
            "--json" => options.json = Some(PathBuf::from(value)),
            "--markdown" => options.markdown = Some(PathBuf::from(value)),
            _ => return Err(format!("unrecognized audit option {option}\n{}", usage())),
        }
    }
    if options.input.as_os_str().is_empty() { return Err(usage().into()); }
    let json = options.json.clone().unwrap_or_else(|| options.input.with_extension("audit.json"));
    let markdown = options.markdown.clone().unwrap_or_else(|| options.input.with_extension("audit.md"));
    audit::run(options)?;
    println!("{}\n{}", json.display(), markdown.display());
    Ok(())
}

/// `audit --lds-barrier`: the `lds_store_unwaited_at_barrier` pass over code
/// objects (cached JIT blobs, `kernels/compiled/<arch>`). One row per arch;
/// every site and every module the lifter rejects is printed and fails.
fn run_lds_barrier(roots: Vec<PathBuf>) -> Result<(), String> {
    fn collect(path: &std::path::Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
        if path.is_dir() {
            let mut entries: Vec<_> = fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))?
                .map(|e| e.map(|e| e.path())).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
            entries.sort();
            for entry in entries { collect(&entry, out)?; }
        } else if path.extension().is_some_and(|x| x == "hsaco" || x == "co") {
            out.push(path.to_owned());
        }
        Ok(())
    }
    if roots.is_empty() { return Err(usage().into()); }
    let mut files = Vec::new();
    for root in &roots { collect(root, &mut files)?; }
    if files.is_empty() { return Err(format!("no *.hsaco or *.co under {roots:?}")); }
    #[derive(Default)]
    struct Row { modules: usize, barrier_modules: usize, kernels: usize, barrier_kernels: usize, barriers: usize, sites: usize }
    let mut rows = std::collections::BTreeMap::<String, Row>::new();
    let (mut rejected, mut sites) = (Vec::new(), 0usize);
    for file in &files {
        let bytes = fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
        match audit::lds_barrier_scan(&bytes) {
            Ok(scan) => {
                let row = rows.entry(scan.arch.clone()).or_default();
                row.modules += 1; row.barrier_modules += usize::from(scan.barriers > 0);
                row.kernels += scan.kernels; row.barrier_kernels += scan.barrier_kernels;
                row.barriers += scan.barriers; row.sites += scan.sites.len();
                for site in &scan.sites {
                    sites += 1;
                    println!("{} {} {} pc {:#x}: {}", audit::LDS_BARRIER_FINDING, file.display(), site.kernel, site.pc, site.detail);
                }
            }
            Err(error) => rejected.push(format!("{}: {error}", file.display())),
        }
    }
    println!("| arch | modules | with barriers | kernels | with barriers | barriers | {} |\n|---|---:|---:|---:|---:|---:|---:|", audit::LDS_BARRIER_FINDING);
    for (arch, r) in &rows {
        println!("| {arch} | {} | {} | {} | {} | {} | {} |", r.modules, r.barrier_modules, r.kernels, r.barrier_kernels, r.barriers, r.sites);
    }
    for line in &rejected { println!("unchecked {line}"); }
    println!("{} files: {} checked, {} unchecked, {sites} {}", files.len(), files.len() - rejected.len(), rejected.len(), audit::LDS_BARRIER_FINDING);
    if sites != 0 || !rejected.is_empty() { return Err(format!("lds-barrier audit failed: {sites} sites, {} unchecked modules", rejected.len())); }
    Ok(())
}

fn check_combined_waits(kernel: &serde_json::Value) -> Result<(), String> {
    let Some(waits) = kernel["waits"].as_array() else { return Ok(()) };
    for wait in waits {
        let Some(text) = wait["insn"].as_str() else { continue };
        let Some(imm) = text.strip_prefix("s_wait_loadcnt_dscnt 0x") else { continue };
        let encoded = u16::from_str_radix(imm, 16).map_err(|_| "invalid combined-wait proof immediate")?;
        let (field, shift) = match wait["counter"].as_str() {
            Some("Load") => ("Load", 8),
            Some("Ds") => ("Ds", 0),
            _ => return Err("combined-wait proof has the wrong counter".into()),
        };
        let count = wait["count"].as_u64().ok_or("combined-wait proof lacks count")?;
        if u64::from((encoded >> shift) & 0x3f) != count {
            return Err(format!("combined-wait {field} field disagrees with the builder proof"));
        }
        let siblings = waits.iter().filter(|other| other["pc_index"] == wait["pc_index"]
            && other["insn"] == wait["insn"]).count();
        if siblings != 2 || !waits.iter().any(|other| other["pc_index"] == wait["pc_index"]
            && other["insn"] == wait["insn"] && other["counter"] != wait["counter"]) {
            return Err("combined-wait proof requires one Load and one Ds field".into())
        }
    }
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
    if let Some(kernels) = decoded["kernels"].as_array() {
        for kernel in kernels { check_combined_waits(kernel)?; }
    } else {
        check_combined_waits(&decoded)?;
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

    #[test]
    fn combined_wait_proof_rejects_swapped_fields() {
        let source = b"s_wait_loadcnt_dscnt 0x703\ns_endpgm\n";
        let base = serde_json::json!({
            "s_text_sha256": format!("{:x}", Sha256::digest(source)),
            "arch": "gfx1201",
            "builder_crate_version": env!("CARGO_PKG_VERSION"),
            "builder_git_sha": "abc123",
            "waits": [
                {"pc_index": 0, "insn": "s_wait_loadcnt_dscnt 0x703", "counter": "Load", "count": 7},
                {"pc_index": 0, "insn": "s_wait_loadcnt_dscnt 0x703", "counter": "Ds", "count": 3}
            ]
        });
        assert!(proof_binding(&serde_json::to_vec(&base).unwrap(), source, "gfx1201").is_ok());
        let mut duplicate = base.clone();
        duplicate["waits"][1] = duplicate["waits"][0].clone();
        assert!(proof_binding(&serde_json::to_vec(&duplicate).unwrap(), source, "gfx1201").is_err());
        let mut swapped = base;
        swapped["waits"][0]["count"] = serde_json::json!(3);
        swapped["waits"][1]["count"] = serde_json::json!(7);
        assert!(proof_binding(&serde_json::to_vec(&swapped).unwrap(), source, "gfx1201").is_err());
    }
}
