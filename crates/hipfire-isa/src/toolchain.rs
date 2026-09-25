// SPDX-License-Identifier: Apache-2.0
//! Offline ROCm assembler/linker/bundler path. Enabled only for host tooling.

use radiowave::{ExistingCodeObjectRequest, PeacemakerArm, PeacemakerProducer, PeacemakerRecord,
    PeacemakerTool, PeacemakerToolRole};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub type Result<T, E = String> = std::result::Result<T, E>;

#[derive(Clone, Debug)]
pub struct Toolchain {
    pub llvm_mc: PathBuf,
    pub linker: PathBuf,
    pub bundler: PathBuf,
    pub objdump: PathBuf,
    pub readobj: PathBuf,
    pub hipcc: PathBuf,
    /// Host bundle target spelling is part of the on-disk bundle identity.
    /// ROCm `hipcc` uses the trailing-hyphen spelling; the hand-assembly
    /// control in §2.4 used the target without it.
    pub host_target: String,
}

impl Default for Toolchain {
    fn default() -> Self {
        let root = std::env::var_os("ROCM_PATH").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/opt/rocm/core-10.0"));
        let llvm = root.join("lib/llvm/bin");
        Self {
            llvm_mc: llvm.join("llvm-mc"), linker: llvm.join("ld.lld"),
            bundler: llvm.join("clang-offload-bundler"),
            objdump: llvm.join("llvm-objdump"), readobj: llvm.join("llvm-readobj"),
            hipcc: root.join("bin/hipcc"),
            host_target: "host-x86_64-unknown-linux-gnu".into(),
        }
    }
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn file_digest(path: &Path) -> Result<String> {
    fs::read(path).map(|bytes| digest(&bytes)).map_err(|e| format!("{}: {e}", path.display()))
}
fn invoke(program: &Path, args: &[String]) -> Result<String> {
    let output = Command::new(program).args(args).output()
        .map_err(|e| format!("{}: {e}", program.display()))?;
    if !output.status.success() {
        return Err(format!("{} {:?}: {}", program.display(), args,
            String::from_utf8_lossy(&output.stderr)));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}
fn identity(path: &Path, role: PeacemakerToolRole, args: Vec<String>) -> Result<PeacemakerTool> {
    Ok(PeacemakerTool { role, path: path.to_owned(),
        version: invoke(path, &["--version".into()])?.lines().next().unwrap_or("").into(),
        argv: std::iter::once(path.to_string_lossy().into_owned()).chain(args).collect(),
        sha256: file_digest(path)? })
}

/// Exact or upper/lower bound on the number of machine instructions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum CountBound { Exact(u32), Range { min: Option<u32>, max: Option<u32> } }
impl CountBound {
    fn accepts(&self, n: u32) -> bool { match self {
        Self::Exact(v) => n == *v,
        Self::Range { min, max } => min.is_none_or(|v| n >= v) && max.is_none_or(|v| n <= v),
    } }
}

/// Shape requirements for a symbol and optionally a label-delimited hot region.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsaShapeContract {
    #[serde(default)] pub symbol: String,
    #[serde(default)] pub start_label: Option<String>,
    #[serde(default)] pub end_label: Option<String>,
    #[serde(default)] pub counts: BTreeMap<String, CountBound>,
    #[serde(default)] pub forbidden: Vec<String>,
    #[serde(default)] pub vgpr_max: Option<u32>,
    #[serde(default)] pub sgpr_max: Option<u32>,
    #[serde(default)] pub require_wave32: bool,
    #[serde(default)] pub require_zero_spills: bool,
    #[serde(default)] pub require_zero_private: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IsaShapeResult {
    pub counts: BTreeMap<String, u32>,
    pub vgpr_count: u32,
    pub sgpr_count: u32,
    pub wavefront_size: u32,
    pub private_segment_fixed_size: u32,
    pub vgpr_spill_count: u32,
    pub sgpr_spill_count: u32,
}

fn instruction(line: &str) -> Option<&str> {
    let part = line.split("//").next()?.split(';').next()?.trim();
    let token = part.split_whitespace().next()?;
    if token.starts_with('.') || token.ends_with(':') || token.starts_with(';')
        || !(token.starts_with('v') || token.starts_with('s') || token.starts_with('b')
            || token.starts_with('g') || token.starts_with('d')) { return None; }
    Some(part)
}
fn count(insn: &str, counts: &mut BTreeMap<String, u32>) {
    let name = insn.split_whitespace().next().unwrap_or("");
    *counts.entry(name.to_owned()).or_default() += 1;
    if name.starts_with("v_dual_") {
        *counts.entry("vopd_packets".into()).or_default() += 1;
    }
    if name.starts_with("v_") && !name.starts_with("v_wmma_") && !name.starts_with("v_swmmac_") {
        *counts.entry("valu_slots".into()).or_default() += 1;
    }
}

/// A linked object must contain the authored instruction stream in order.
/// The assembler can accept syntax yet silently change an immediate; compare
/// canonical operands for straight-line instructions, not just mnemonic counts.
pub fn parse_back(source: &str, disassembly: &str) -> Result<()> {
    let authored: Vec<_> = source.lines().filter_map(instruction).collect();
    let mut after_endpgm = false;
    let decoded: Vec<_> = disassembly.lines().filter_map(|line| {
        if line.trim_end().ends_with(">:") { after_endpgm = false; }
        let insn = instruction(line)?;
        if after_endpgm && matches!(insn.split_whitespace().next(), Some("s_nop" | "s_code_end")) {
            return None;
        }
        if insn.starts_with("s_endpgm") { after_endpgm = true; }
        Some(insn)
    }).collect();
    if authored.len() != decoded.len() {
        return Err(format!("parse-back length: authored {}, decoded {}", authored.len(), decoded.len()));
    }
    for (index, (author, decoded)) in authored.iter().zip(decoded).enumerate() {
        let a = author.split_whitespace().next().unwrap_or("");
        let d = decoded.split_whitespace().next().unwrap_or("");
        if a != d {
            return Err(format!("parse-back instruction {index}: {a} != {d}"));
        }
        // Branch operands are labels in source and relative offsets in the
        // linked ELF. The mnemonic/order is still checked.
        if !a.starts_with("s_branch") && !a.starts_with("s_cbranch") {
            let normalize = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
            let canonical_author = if a == "s_barrier_wait" && author.ends_with(" -1") {
                author.replace(" -1", " 0xffff")
            } else { author.to_string() };
            let canonical_decoded = if (a == "v_and_b16" || a == "v_or_b16")
                && (decoded.ends_with(" op_sel:[0,0,1]") || decoded.ends_with(" op_sel:[1,0,0]"))
                && (author.contains(".h") || author.contains(".l")) {
                decoded.split(" op_sel:").next().unwrap_or(decoded).to_owned()
            } else { decoded.to_string() };
            if normalize(&canonical_author) != normalize(&canonical_decoded) {
                return Err(format!("parse-back instruction {index}: {author} != {decoded}"));
            }
        }
    }
    Ok(())
}

/// Check the *linked* object's disassembly and inspected resource metadata.
/// Unknown count names are checked as zero, so a typo cannot silently pass.
pub fn check_shape(disassembly: &str, report: &radiowave::KernelReport,
    contract: &IsaShapeContract) -> Result<IsaShapeResult> {
    if contract.symbol.is_empty() { return Err("shape contract requires symbol".into()); }
    let mut in_symbol = false;
    let mut in_region = contract.start_label.is_none();
    let mut saw_symbol = false;
    let mut saw_start = in_region;
    let mut saw_end = contract.end_label.is_none();
    let mut result = IsaShapeResult {
        vgpr_count: report.vgpr_count, sgpr_count: report.sgpr_count,
        wavefront_size: report.wavefront_size,
        private_segment_fixed_size: report.private_segment_fixed_size,
        vgpr_spill_count: report.vgpr_spill_count,
        sgpr_spill_count: report.sgpr_spill_count, ..Default::default()
    };
    for line in disassembly.lines() {
        let trimmed = line.trim();
        if let Some((_, name)) = trimmed.split_once('<') {
            if let Some(name) = name.strip_suffix(":") {
                let name = name.trim_end_matches('>');
                if name == contract.symbol { in_symbol = true; saw_symbol = true; }
                else if in_symbol && contract.start_label.as_deref() == Some(name) {
                    in_region = true; saw_start = true;
                } else if in_symbol && contract.end_label.as_deref() == Some(name) {
                    in_region = false; saw_end = true;
                } else if in_symbol && !name.starts_with(".L") && !name.contains('+') {
                    break;
                }
            }
        }
        if !in_symbol { continue; }
        if let Some(insn) = instruction(trimmed) {
            let mnemonic = insn.split_whitespace().next().unwrap_or("");
            for forbidden in &contract.forbidden {
                if mnemonic == forbidden || (forbidden.ends_with('*') && mnemonic.starts_with(forbidden.trim_end_matches('*'))) {
                    return Err(format!("forbidden instruction {mnemonic}"));
                }
            }
            if in_region { count(insn, &mut result.counts); }
        }
    }
    if !saw_symbol || !saw_start || !saw_end { return Err("missing symbol or region boundary".into()); }
    for (name, bound) in &contract.counts {
        if let CountBound::Range { min, max } = bound {
            if min.is_none() && max.is_none() || min.zip(*max).is_some_and(|(lo, hi)| lo > hi) {
                return Err(format!("{name}: invalid empty or reversed count bound"));
            }
        }
        let actual = if let Some(prefix) = name.strip_suffix('*') {
            result.counts.iter().filter(|(mnemonic, _)| mnemonic.starts_with(prefix))
                .map(|(_, n)| n).sum()
        } else { result.counts.get(name).copied().unwrap_or(0) };
        if !bound.accepts(actual) { return Err(format!("{name}: observed {actual}, contract {bound:?}")); }
    }
    if contract.require_zero_spills && (result.vgpr_spill_count != 0 || result.sgpr_spill_count != 0)
        || contract.require_zero_private && result.private_segment_fixed_size != 0
        || contract.require_wave32 && result.wavefront_size != 32
        || contract.vgpr_max.is_some_and(|max| result.vgpr_count > max)
        || contract.sgpr_max.is_some_and(|max| result.sgpr_count > max) {
        return Err(format!("resource contract failed: {result:?}"));
    }
    Ok(result)
}

/// Read back a gfx12 64-byte kernel descriptor from a linked ELF object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gfx12KernelImageFields {
    pub code_entry: u64,
    pub compute_pgm_rsrc1: u32,
    pub compute_pgm_rsrc2: u32,
    pub compute_pgm_rsrc3: u32,
    pub kernel_code_properties: u16,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub wave32: bool,
}
fn u16le(data: &[u8], n: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(data.get(n..n+2).ok_or("truncated ELF")?.try_into().unwrap()))
}
fn u32le(data: &[u8], n: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(data.get(n..n+4).ok_or("truncated ELF")?.try_into().unwrap()))
}
fn u64le(data: &[u8], n: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(data.get(n..n+8).ok_or("truncated ELF")?.try_into().unwrap()))
}
fn cstr(data: &[u8], off: usize) -> Option<&str> {
    let end = off.checked_add(data.get(off..)?.iter().position(|x| *x == 0)?)?;
    std::str::from_utf8(data.get(off..end)?).ok()
}
fn section(data: &[u8], i: usize) -> Result<(u32, u64, usize, usize, usize, usize)> {
    let base = usize::try_from(u64le(data, 40)?).map_err(|e| e.to_string())?
        .checked_add(i.checked_mul(u16le(data, 58)? as usize).ok_or("section overflow")?)
        .ok_or("section overflow")?;
    Ok((u32le(data, base+4)?, u64le(data, base+16)?,
        usize::try_from(u64le(data, base+24)?).map_err(|e| e.to_string())?,
        usize::try_from(u64le(data, base+32)?).map_err(|e| e.to_string())?,
        u32le(data, base+40)? as usize,
        usize::try_from(u64le(data, base+56)?).map_err(|e| e.to_string())?))
}
pub fn read_kd(object: &Path, symbol: &str) -> Result<Gfx12KernelImageFields> {
    let data = fs::read(object).map_err(|e| e.to_string())?;
    if data.get(..5) != Some(&b"\x7fELF\x02"[..]) || data.get(5) != Some(&1) {
        return Err("expected little-endian ELF64".into());
    }
    let sections = u16le(&data, 60)? as usize;
    let kd_name = format!("{symbol}.kd");
    for i in 0..sections {
        let (typ, _, off, size, stridx, entsize) = section(&data, i)?;
        if typ != 2 && typ != 11 { continue; } // symtab or dynsym
        if entsize < 24 || stridx >= sections { return Err("invalid ELF symbol section".into()); }
        let (_, _, str_off, str_size, _, _) = section(&data, stridx)?;
        let strings = data.get(str_off..str_off.checked_add(str_size).ok_or("string overflow")?)
            .ok_or("invalid ELF string table")?;
        for entry in (off..off.checked_add(size).ok_or("symbol overflow")?).step_by(entsize) {
            if cstr(strings, u32le(&data, entry)? as usize) != Some(kd_name.as_str()) { continue; }
            let idx = u16le(&data, entry+6)? as usize;
            if idx >= sections || idx == 0 { return Err("descriptor has no section".into()); }
            let (_, section_addr, section_off, section_size, _, _) = section(&data, idx)?;
            let addr = u64le(&data, entry+8)?;
            let delta = usize::try_from(addr.checked_sub(section_addr).ok_or("descriptor address")?)
                .map_err(|e| e.to_string())?;
            if delta.checked_add(64).is_none_or(|end| end > section_size) {
                return Err("truncated kernel descriptor".into());
            }
            let kd = data.get(section_off+delta..section_off+delta+64)
                .ok_or("truncated kernel descriptor")?;
            let relative = u64le(kd, 16)? as i64;
            let code_entry = addr.checked_add_signed(relative).ok_or("invalid code entry")?;
            let properties = u16le(kd, 56)?;
            return Ok(Gfx12KernelImageFields {
                code_entry, compute_pgm_rsrc3: u32le(kd, 44)?,
                compute_pgm_rsrc1: u32le(kd, 48)?, compute_pgm_rsrc2: u32le(kd, 52)?,
                kernel_code_properties: properties,
                group_segment_size: u32le(kd, 0)?, private_segment_size: u32le(kd, 4)?,
                wave32: properties & 0x400 != 0,
            });
        }
    }
    Err(format!("kernel descriptor {kd_name} not found in {}", object.display()))
}

#[derive(Clone, Debug)]
pub struct BuildOutput {
    pub hsaco: PathBuf,
    pub elf: PathBuf,
    pub object: PathBuf,
    pub disassembly: String,
    pub argv: Vec<String>,
    pub tools: Vec<PeacemakerTool>,
}

/// Extract and disassemble the exact device ELF in an uncompressed HIP bundle.
/// An ELF input is accepted directly for offline audits of hand-linked objects.
pub fn disassemble_code_object(toolchain: &Toolchain, input: &Path, arch: &str) -> Result<String> {
    if fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?.starts_with(b"\x7fELF") {
        return invoke(&toolchain.objdump,
            &["--disassemble".into(), format!("--mcpu={arch}"), input.display().to_string()]);
    }
    let targets = invoke(&toolchain.bundler,
        &["--type=o".into(), "--list".into(), format!("--input={}", input.display())])?;
    let target = targets.lines().map(str::trim)
        .find(|line| line.starts_with("hip") && line.ends_with(&format!("--{arch}")))
        .ok_or_else(|| format!("no HIP bundle for {arch} in {}", input.display()))?;
    let temporary = std::env::temp_dir().join(format!("peacemaker-{}-{}.co",
        std::process::id(), std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos()));
    let result = (|| {
        invoke(&toolchain.bundler, &["--type=o".into(), "--unbundle".into(),
            format!("--input={}", input.display()), format!("--targets={target}"),
            format!("--output={}", temporary.display())])?;
        invoke(&toolchain.objdump, &["--disassemble".into(), format!("--mcpu={arch}"),
            temporary.display().to_string()])
    })();
    let _ = fs::remove_file(temporary);
    result
}
/// The external commands are deliberately fixed: no HIP compilation or compression.
pub fn assemble_link_bundle(toolchain: &Toolchain, source: &Path, output: &Path,
    arch: &str) -> Result<BuildOutput> {
    if !matches!(arch, "gfx1201" | "gfx1200" | "gfx1100" | "gfx1151") {
        return Err(format!("unsupported AMDGPU architecture: {arch}"));
    }
    let stem = output.with_extension("");
    let object = stem.with_extension("o");
    let elf = stem.with_extension("co");
    let mc_args = vec!["-triple=amdgcn-amd-amdhsa".into(), format!("-mcpu={arch}"),
        "-filetype=obj".into(), source.display().to_string(), "-o".into(), object.display().to_string()];
    let ld_args = vec!["-shared".into(), object.display().to_string(), "-o".into(), elf.display().to_string()];
    let bundle_args = vec!["-type=o".into(), "-bundle-align=4096".into(),
        format!("-targets={},hipv4-amdgcn-amd-amdhsa--{arch}", toolchain.host_target),
        "-input=/dev/null".into(), format!("-input={}", elf.display()),
        format!("-output={}", output.display())];
    let tools = vec![identity(&toolchain.llvm_mc, PeacemakerToolRole::Assembler, mc_args.clone())?,
        identity(&toolchain.linker, PeacemakerToolRole::Linker, ld_args.clone())?,
        identity(&toolchain.bundler, PeacemakerToolRole::Bundler, bundle_args.clone())?];
    let mut argv = Vec::new();
    for tool in &tools { argv.extend(tool.argv.iter().cloned()); }
    invoke(&toolchain.llvm_mc, &mc_args)?;
    invoke(&toolchain.linker, &ld_args)?;
    invoke(&toolchain.bundler, &bundle_args)?;
    let disassembly = invoke(&toolchain.objdump,
        &["--disassemble".into(), format!("--mcpu={arch}"), elf.display().to_string()])?;
    let source_text = fs::read_to_string(source).map_err(|e| e.to_string())?;
    parse_back(&source_text, &disassembly)?;
    Ok(BuildOutput { hsaco: output.into(), elf, object, disassembly, argv, tools })
}

/// Bind the final HIP bundle and optional shape evidence to schema-5 radiowave.
pub fn certify(toolchain: &Toolchain, build: &BuildOutput, source: &Path, arch: &str,
    manifest: &Path, contract: Option<&IsaShapeContract>, proof_sha256: &str,
    builder_git_sha: &str) -> Result<()> {
    let mut request = ExistingCodeObjectRequest::new(source, &build.hsaco, arch)
        .hipcc(&toolchain.hipcc).manifest(manifest).command(build.argv.clone());
    if let Some(contract) = contract {
        let inspection = radiowave::Inspector::from_hipcc(&toolchain.hipcc)
            .inspect(&build.hsaco, arch).map_err(|e| e.to_string())?;
        let report = inspection.kernels.iter().find(|kernel| kernel.name == contract.symbol)
            .ok_or_else(|| format!("inspection missing kernel {}", contract.symbol))?;
        let shape = check_shape(&build.disassembly, report, contract)?;
        let kd = read_kd(&build.elf, &contract.symbol)?;
        if kd.private_segment_size != 0 || kd.group_segment_size != 0 || !kd.wave32 {
            return Err(format!("kernel descriptor violates custom-arm resources: {kd:?}"));
        }
        crate::ledger_replay::replay_waits(
            &fs::read_to_string(source).map_err(|e| e.to_string())?
        )?;
        invoke(&toolchain.readobj,
            &["--sections".into(), build.elf.display().to_string()])?;
        // The uncompressed ELF, not a source or manifest field, is the hash input.
        // Extracting .text through objcopy would add a seventh executable to the recipe;
        // use the ELF section range read directly instead.
        let elf_bytes = fs::read(&build.elf).map_err(|e| e.to_string())?;
        let section_count = u16le(&elf_bytes, 60)? as usize;
        let names_index = u16le(&elf_bytes, 62)? as usize;
        let (_, _, names_off, names_size, _, _) = section(&elf_bytes, names_index)?;
        let names = elf_bytes.get(names_off..names_off+names_size).ok_or("invalid ELF section names")?;
        let mut text_section_sha256 = None;
        for i in 0..section_count {
            let base = usize::try_from(u64le(&elf_bytes, 40)?).map_err(|e| e.to_string())?
                + i * u16le(&elf_bytes, 58)? as usize;
            if cstr(names, u32le(&elf_bytes, base)? as usize) == Some(".text") {
                let (_, _, off, size, _, _) = section(&elf_bytes, i)?;
                text_section_sha256 = Some(digest(elf_bytes.get(off..off+size).ok_or("invalid .text")?));
            }
        }
        let record = PeacemakerRecord {
            arm: PeacemakerArm::CustomIsa,
            producer: PeacemakerProducer {
                builder_crate_version: env!("CARGO_PKG_VERSION").into(),
                builder_git_sha: builder_git_sha.into(), proof_sha256: proof_sha256.into(),
                aco_commit: None,
            },
            tools: build.tools.clone(), s_text_sha256: file_digest(source)?,
            text_section_sha256: text_section_sha256.ok_or("missing .text section")?,
            shape_contract: serde_json::to_value(contract).map_err(|e| e.to_string())?,
            shape_result: serde_json::to_value(shape).map_err(|e| e.to_string())?,
            oracle_receipt: None,
        };
        request = request.peacemaker(record);
    }
    radiowave::Compiler.certify_existing(&request).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_rejects_forbidden_instruction_even_when_counts_match() {
        let contract = IsaShapeContract {
            symbol: "k".into(), forbidden: vec!["s_waitcnt".into()],
            counts: [("vopd_packets".into(), CountBound::Exact(1))].into(),
            ..Default::default()
        };
        let disasm = "000000 <k>:\n v_dual_mul_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5 // 0000\n s_waitcnt vmcnt(0) // 0008\n";
        assert!(check_shape(disasm, &radiowave::KernelReport::default(), &contract).is_err());
    }
    #[test]
    fn parse_back_rejects_silently_wrapped_immediate() {
        let source = "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen offset:4096\ns_endpgm";
        let decoded = "000000 <k>:\n buffer_load_b64 v[0:1], v2, s[4:7], s8 offen offset:0 // 0\n s_endpgm // 0";
        assert!(parse_back(source, decoded).is_err());
    }
    #[test]
    fn shape_counts_packets_not_wmma_as_valu() {
        let contract = IsaShapeContract {
            symbol: "k".into(),
            counts: [
                ("valu_slots".into(), CountBound::Exact(1)),
                ("vopd_packets".into(), CountBound::Exact(1)),
                ("v_wmma_*".into(), CountBound::Exact(1)),
            ].into(),
            ..Default::default()
        };
        let disasm = "000000 <k>:\n v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[10:11], 0 // 00\n v_dual_mul_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5 // 08\n";
        assert!(check_shape(disasm, &radiowave::KernelReport::default(), &contract).is_ok());
        let missing_packet = disasm.replace("v_dual_mul_f32", "v_mul_f32");
        assert!(check_shape(&missing_packet, &radiowave::KernelReport::default(), &contract).is_err());
    }
    #[test]
    fn misspelled_shape_limit_does_not_disable_gate() {
        let typo = r#"{"symbol":"k","counts":{"valu_slots":{"maximum":110}}}"#;
        assert!(serde_json::from_str::<IsaShapeContract>(typo).is_err());
        let unbounded = IsaShapeContract {
            symbol: "k".into(),
            counts: [("valu_slots".into(), CountBound::Range { min: None, max: None })].into(),
            ..Default::default()
        };
        assert!(check_shape("000000 <k>:\n v_mul_f32 v0, v1, v2 // 0",
            &radiowave::KernelReport::default(), &unbounded).is_err());
    }
}
