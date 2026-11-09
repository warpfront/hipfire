// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Route A ("lift-and-repass") sealing of a hipcc-compiled kernel as an embedded per-arch
//! code object. RDNA uses the `Frontend::Hipcc` IR lift and `emit::checked`; gfx942
//! uses the typed CDNA lift and checked envelope re-emission. In both cases the
//! emitted bytes must equal the input ELF byte for byte, before [`native::bundle`]
//! bundles and sha-pins them. See `examples/hip_twin.rs`.

use crate::arch::Arch;
use crate::native;
use peacemaker_ir::codec::gfx12::{decode_for, encode_for};
use peacemaker_ir::envelope::{Bundle, Envelope};
use peacemaker_ir::helpers::{helper_extents, HelperExtent};
use peacemaker_ir::inst::{Abi, Arch as IrArch, Frontend, Kernel, Program, Wave};
use peacemaker_ir::metadata::{Kernarg, KernelMeta};
use peacemaker_ir::passes::hazards;
use peacemaker_lift::bundle::BundleCodec;
use peacemaker_lift::hazards_gfx11::{check_kernel, Severity};
use peacemaker_lift::{emit, lift_object, Options};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The host entry of every committed `kernels/*_gfx1100.hxaco`.
pub const HOST_TRIPLE: &str = "host-x86_64-unknown-linux-gnu";

const OPTS: Options = Options {
    frontend: Frontend::Hipcc,
};

#[derive(Clone, Debug)]
pub struct Sealed {
    pub elf_sha256: String,
    /// sha256 of the kernel's `.text` bytes (the re-encoded instruction stream).
    pub text_sha256: String,
    pub hxaco: Vec<u8>,
    pub hxaco_sha256: String,
    pub vgprs: u32,
    pub sgprs: u32,
    pub lds: u32,
    pub scratch: u32,
    pub user_sgprs: u32,
    pub report: Value,
}

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn ir_arch(arch: Arch) -> Result<IrArch, String> {
    match arch {
        Arch::Gfx1100 => Ok(IrArch::Gfx1100),
        Arch::Gfx1151 => Ok(IrArch::Gfx1151),
        Arch::Gfx1200 => Ok(IrArch::Gfx1200),
        Arch::Gfx1201 => Ok(IrArch::Gfx1201),
        Arch::Gfx942 => Err("gfx942 uses the typed CDNA lift, not the RDNA IR".into()),
    }
}

/// An out-of-line helper extent (`peacemaker_ir::helpers`) with its file offset in the original device ELF.
struct HelperBody {
    extent: HelperExtent,
    /// File offset of the first byte in the original device ELF.
    offset: usize,
}

/// Every helper extent of `env` (see `helper_extents`: the symbol-table scan and its refusals) located in `elf`; an
/// extent outside the ELF file is an error.
fn helper_bodies(env: &Envelope, elf: &[u8]) -> Result<Vec<HelperBody>, String> {
    helper_extents(env)?
        .into_iter()
        .map(|extent| {
            let h = &env.sections[extent.section].header;
            let offset = h
                .sh_offset
                .checked_add(extent.start - h.sh_addr)
                .and_then(|o| usize::try_from(o).ok())
                .filter(|&o| {
                    usize::try_from(extent.size)
                        .ok()
                        .and_then(|n| o.checked_add(n))
                        .is_some_and(|e| e <= elf.len())
                })
                .ok_or_else(|| {
                    format!(
                        "function symbol `{}`: extent is outside the ELF file",
                        extent.names.iter().next().map_or("?", String::as_str)
                    )
                })?;
            Ok(HelperBody { extent, offset })
        })
        .collect()
}

/// Decode every instruction of every out-of-line helper body (read from the original device
/// ELF, not the envelope, whose kernel holes are zeroed), re-encode it with the typed codec and
/// require the words to be byte-exact. Returns the report entries (empty when the object has no
/// helper). This is typed-instruction re-encode coverage only: no helper CFG, hazard or
/// emulator proof is claimed.
fn certify_helpers(env: &Envelope, elf: &[u8], ir: IrArch) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for HelperBody { extent: b, offset } in helper_bodies(env, elf)? {
        let name = b.names.iter().next().expect("helper has a name").clone();
        let code = &elf[offset..offset + b.size as usize];
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().expect("dword")))
            .collect();
        let mut stream = Vec::with_capacity(code.len());
        let (mut at, mut count) = (0usize, 0usize);
        while at < words.len() {
            let va = b.start + 4 * at as u64;
            let fail = |why: String| format!("helper `{name}` at {va:#x}: {why}");
            let (inst, n) =
                decode_for(ir, &words[at..]).map_err(|e| fail(format!("decode: {e}")))?;
            if n == 0 || at + n > words.len() {
                return Err(fail(format!(
                    "instruction of {n} dword(s) leaves the helper extent"
                )));
            }
            let encoded = encode_for(ir, &inst).map_err(|e| fail(format!("re-encode: {e}")))?;
            if encoded.as_slice() != &words[at..at + n] {
                return Err(fail(format!(
                    "re-encoded {encoded:08x?} != input {:08x?}",
                    &words[at..at + n]
                )));
            }
            for w in &encoded {
                stream.extend_from_slice(&w.to_le_bytes());
            }
            at += n;
            count += 1;
        }
        if stream != code {
            return Err(format!(
                "helper `{name}`: re-encoded stream differs from the ELF bytes"
            ));
        }
        let mut entry = json!({
            "symbol": name, "section_index": b.section,
            "entry_va": b.start, "bytes": b.size, "instructions": count,
            "stream_sha256": hex(&stream),
            "coverage": "typed instruction re-encode",
        });
        if b.names.len() > 1 {
            entry["aliases"] = json!(b.names.iter().skip(1).collect::<Vec<_>>());
        }
        out.push(entry);
    }
    Ok(out)
}

/// The device ELF of `input`: the bytes themselves when they are not a bundle, else the HIP
/// device entry for `arch`.
fn device_elf(input: &[u8], arch: Arch) -> Result<&[u8], String> {
    if !Bundle::is_bundle(input) {
        return Ok(input);
    }
    let (bundle, payloads) = Bundle::read(input).map_err(|e| format!("bundle: {e}"))?;
    let idx = if arch == Arch::Gfx942 {
        let triple = format!(
            "{}gfx942",
            peacemaker_lift::bundle::HIP_DEVICE_TRIPLE_PREFIX
        );
        let mut matches = bundle.entries.iter().enumerate().filter(|(_, entry)| {
            entry.triple == triple
                || entry
                    .triple
                    .strip_prefix(&triple)
                    .is_some_and(|suffix| suffix.starts_with(':'))
        });
        let idx = matches
            .next()
            .map(|(idx, _)| idx)
            .ok_or("bundle has no gfx942 device entry")?;
        if matches.next().is_some() {
            return Err("bundle has multiple gfx942 device entries".into());
        }
        idx
    } else {
        bundle
            .device_entry(ir_arch(arch)?)
            .map_err(|e| format!("bundle has no {} device entry: {e}", arch.name()))?
    };
    Ok(payloads[idx])
}

fn kernel_of<'a>(p: &'a Program, symbol: &str) -> Result<&'a Kernel, String> {
    p.kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| {
            let have: Vec<&str> = p.kernels.iter().map(|k| k.symbol.0.as_str()).collect();
            format!("kernel `{symbol}` not found; object has {have:?}")
        })
}

fn hsa(k: &Kernel) -> Result<(&peacemaker_ir::descriptor::KernelDescriptor, &KernelMeta), String> {
    match &k.abi {
        Abi::Hsa {
            descriptor,
            metadata,
        } => Ok((descriptor, &metadata.parsed)),
        Abi::Raw(_) => Err(format!("{}: not an HSA kernel", k.symbol.0)),
    }
}

fn is_pointer(a: &Kernarg) -> bool {
    matches!(
        a.value_kind.as_str(),
        "global_buffer" | "dynamic_shared_pointer"
    )
}

/// Explicit (non-hidden) kernargs as `(offset, size, pointer?)`, in metadata order.
fn explicit_args(m: &KernelMeta) -> Vec<(u32, u32, bool)> {
    m.args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .map(|a| (a.offset, a.size, is_pointer(a)))
        .collect()
}

fn abi_diff(twin: &KernelMeta, inc: &KernelMeta) -> Option<String> {
    let (t, i) = (explicit_args(twin), explicit_args(inc));
    let mut out = Vec::new();
    if t.len() != i.len() {
        out.push(format!(
            "explicit arg count: twin {} vs incumbent {}",
            t.len(),
            i.len()
        ));
    }
    for (n, (a, b)) in t.iter().zip(&i).enumerate() {
        if a != b {
            out.push(format!(
                "arg {n}: twin (offset {}, size {}, pointer {}) vs incumbent (offset {}, size {}, pointer {})",
                a.0, a.1, a.2, b.0, b.1, b.2
            ));
        }
    }
    if twin.kernarg_segment_size != inc.kernarg_segment_size {
        out.push(format!(
            "kernarg_segment_size: twin {} vs incumbent {}",
            twin.kernarg_segment_size, inc.kernarg_segment_size
        ));
    }
    (!out.is_empty()).then(|| out.join("; "))
}

fn args_json(m: &KernelMeta) -> Value {
    Value::Array(
        m.args
            .iter()
            .map(|a| {
                json!({"name": a.name, "offset": a.offset, "size": a.size,
                       "value_kind": a.value_kind, "pointer": is_pointer(a)})
            })
            .collect(),
    )
}

/// Seal `symbol` of a raw AMDGPU ELF or offload bundle (see the module docs). `abi_ref` is the
/// incumbent object bytes and its kernel symbol; the explicit kernarg layout must match.
pub fn seal(
    input: &[u8],
    arch: Arch,
    symbol: &str,
    abi_ref: Option<(&[u8], &str)>,
) -> Result<Sealed, String> {
    if arch == Arch::Gfx942 {
        return seal_cdna(input, symbol, abi_ref);
    }
    let ir = ir_arch(arch)?;
    let elf = device_elf(input, arch)?;
    let lifted = lift_object(elf, OPTS).map_err(|e| format!("lift: {e}"))?;
    if lifted.program.target.arch != ir {
        return Err(format!(
            "object targets {:?}, requested {}",
            lifted.program.target.arch,
            arch.name()
        ));
    }
    let (emitted, _) =
        emit::checked(&lifted.program, OPTS).map_err(|e| format!("emit::checked: {e}"))?;
    if emitted != elf {
        let at = emitted
            .iter()
            .zip(elf)
            .position(|(a, b)| a != b)
            .unwrap_or(emitted.len().min(elf.len()));
        return Err(format!(
            "re-emitted ELF differs from input at byte offset {at:#x} (emitted {} bytes, input {} bytes)",
            emitted.len(),
            elf.len()
        ));
    }
    let program = &lifted.program;
    let kernel = kernel_of(program, symbol)?;
    let (desc, meta) = hsa(kernel)?;
    let source = program
        .source
        .as_ref()
        .ok_or("lifted program carries no source envelope")?;
    let helpers = certify_helpers(&source.elf, elf, ir)?;
    let text = emit::bytes(kernel, ir).map_err(|e| format!("emit text: {e}"))?;

    // Static hazards (gfx11 arches).
    let mut hz = json!({"checked": false});
    let mut hz_err = Vec::new();
    if matches!(
        arch,
        Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1200 | Arch::Gfx1201
    ) {
        let findings = check_kernel(kernel, ir, true);
        let blocking: Vec<_> = findings
            .iter()
            .filter(|f| f.severity != Severity::Advisory)
            .collect();
        let pm = hazards::analyze(&kernel.body, ir, kernel.wave)
            .map(|a| a.obligations.len())
            .map_err(|e| e.to_string());
        for f in &blocking {
            hz_err.push(format!("{}", f.to_json()));
        }
        match &pm {
            Ok(0) => {}
            Ok(n) => hz_err.push(format!("peacemaker_ir hazard replay: {n} obligation(s)")),
            Err(e) => hz_err.push(format!("peacemaker_ir hazard replay failed: {e}")),
        }
        hz = json!({"checked": true, "findings": findings.len(),
                    "blocking": blocking.len(), "advisory": findings.len() - blocking.len(),
                    "pm_ir_obligations": pm.clone().unwrap_or(usize::MAX)});
    }
    if !hz_err.is_empty() {
        return Err(format!(
            "static hazards in `{symbol}`: {}",
            hz_err.join(" | ")
        ));
    }

    if arch == Arch::Gfx1100 {
        arch.check_user_system_sgprs(symbol, desc.compute_pgm_rsrc2.0)?;
    }

    let mut abi = json!({"kernarg_segment_size": meta.kernarg_segment_size, "args": args_json(meta),
                         "compared_to_incumbent": false});
    if let Some((inc_bytes, inc_sym)) = abi_ref {
        let inc_elf = device_elf(inc_bytes, arch)?;
        let inc = lift_object(inc_elf, OPTS).map_err(|e| format!("lift abi-ref: {e}"))?;
        let inc_k = kernel_of(&inc.program, inc_sym).map_err(|e| format!("abi-ref: {e}"))?;
        let (_, inc_meta) = hsa(inc_k)?;
        if let Some(d) = abi_diff(meta, inc_meta) {
            return Err(format!(
                "kernarg ABI mismatch vs incumbent `{inc_sym}`: {d}"
            ));
        }
        abi["compared_to_incumbent"] = json!(true);
        abi["incumbent_symbol"] = json!(inc_sym);
        abi["incumbent_sha256"] = json!(hex(inc_elf));
    }

    let hxaco = native::bundle(elf, arch, HOST_TRIPLE);
    let vgprs = desc
        .compute_pgm_rsrc1
        .next_free_vgpr(kernel.wave == Wave::Wave32);
    let user_sgprs = (desc.compute_pgm_rsrc2.0 >> 1) & 0x1f;
    let mut s = Sealed {
        elf_sha256: hex(elf),
        text_sha256: hex(&text),
        hxaco_sha256: hex(&hxaco),
        hxaco,
        vgprs,
        sgprs: meta.sgpr_count,
        lds: desc.group_segment_fixed_size,
        scratch: desc.private_segment_fixed_size,
        user_sgprs,
        report: Value::Null,
    };
    s.report = json!({
        "kernel": symbol, "arch": arch.name(),
        "elf_sha256": s.elf_sha256, "elf_bytes": elf.len(),
        "text_sha256": s.text_sha256, "text_bytes": text.len(),
        "hxaco_sha256": s.hxaco_sha256, "hxaco_bytes": s.hxaco.len(),
        "lift_identity": "emit::checked ok; emitted bytes == input ELF",
        "vgprs": s.vgprs, "sgprs": s.sgprs, "lds": s.lds, "scratch": s.scratch,
        "user_sgprs": s.user_sgprs, "rsrc2": desc.compute_pgm_rsrc2.0,
        "wavefront_size": meta.wavefront_size,
        "hazards": hz, "abi": abi,
    });
    if !helpers.is_empty() {
        s.report["helpers"] = json!(helpers);
        s.report["helper_coverage"] = json!(
            "typed instruction re-encode (decode_for+encode_for byte-exact over each out-of-line helper's full extent); no helper CFG/emulator proof"
        );
    }
    Ok(s)
}

/// Route A for the typed CDNA model. The existing ELF envelope codec rebuilds
/// descriptors and metadata as well as every kernel's re-encoded instructions.
fn seal_cdna(input: &[u8], symbol: &str, abi_ref: Option<(&[u8], &str)>) -> Result<Sealed, String> {
    let arch = Arch::Gfx942;
    let elf = device_elf(input, arch)?;
    let lifted = peacemaker_lift::cdna::lift_object(elf).map_err(|e| format!("CDNA lift: {e}"))?;
    let emitted =
        peacemaker_lift::cdna::checked(&lifted).map_err(|e| format!("CDNA emit::checked: {e}"))?;
    if emitted != elf {
        let at = emitted
            .iter()
            .zip(elf)
            .position(|(a, b)| a != b)
            .unwrap_or(emitted.len().min(elf.len()));
        return Err(format!(
            "CDNA re-emitted ELF differs from input at byte offset {at:#x}"
        ));
    }
    let kernel = lifted
        .kernels
        .iter()
        .find(|k| k.image.name == symbol)
        .ok_or_else(|| format!("kernel `{symbol}` not found in CDNA object"))?;
    let desc = &kernel.image.descriptor;
    let meta = &kernel.image.metadata.parsed;
    let hz = crate::cdna::lifted::certify(kernel)
        .map_err(|e| format!("CDNA certification of `{symbol}`: {e}"))?;
    let mut abi = json!({"kernarg_segment_size": meta.kernarg_segment_size,
        "args": args_json(meta), "compared_to_incumbent": false});
    if let Some((inc_bytes, inc_sym)) = abi_ref {
        let inc_elf = device_elf(inc_bytes, arch)?;
        let inc = peacemaker_lift::cdna::lift_object(inc_elf)
            .map_err(|e| format!("CDNA lift abi-ref: {e}"))?;
        let inc_k = inc
            .kernels
            .iter()
            .find(|k| k.image.name == inc_sym)
            .ok_or_else(|| format!("abi-ref kernel `{inc_sym}` not found"))?;
        if let Some(diff) = abi_diff(meta, &inc_k.image.metadata.parsed) {
            return Err(format!(
                "kernarg ABI mismatch vs incumbent `{inc_sym}`: {diff}"
            ));
        }
        abi["compared_to_incumbent"] = json!(true);
        abi["incumbent_symbol"] = json!(inc_sym);
        abi["incumbent_sha256"] = json!(hex(inc_elf));
    }
    let mut words = Vec::with_capacity(kernel.image.code.len() / 4);
    for instruction in &kernel.instructions {
        instruction
            .inst
            .encode(peacemaker_cdna::Arch::Gfx942, &mut words)?;
    }
    let text: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    if text != kernel.image.code {
        return Err(format!("CDNA typed text differs for `{symbol}`"));
    }
    let hxaco = native::bundle(&emitted, arch, HOST_TRIPLE);
    let mut sealed = Sealed {
        elf_sha256: hex(&emitted),
        text_sha256: hex(&text),
        hxaco_sha256: hex(&hxaco),
        hxaco,
        vgprs: desc.compute_pgm_rsrc1.next_free_vgpr(false),
        sgprs: meta.sgpr_count,
        lds: desc.group_segment_fixed_size,
        scratch: desc.private_segment_fixed_size,
        user_sgprs: (desc.compute_pgm_rsrc2.0 >> 1) & 0x1f,
        report: Value::Null,
    };
    sealed.report = json!({
        "kernel": symbol, "arch": arch.name(),
        "elf_sha256": sealed.elf_sha256, "elf_bytes": emitted.len(),
        "text_sha256": sealed.text_sha256, "text_bytes": text.len(),
        "hxaco_sha256": sealed.hxaco_sha256, "hxaco_bytes": sealed.hxaco.len(),
        "lift_identity": "emit::checked ok; emitted bytes == input ELF",
        "codec": "peacemaker_cdna", "lifted_kernels": lifted.kernels.len(),
        "vgprs": sealed.vgprs, "sgprs": sealed.sgprs, "lds": sealed.lds,
        "scratch": sealed.scratch, "user_sgprs": sealed.user_sgprs,
        "rsrc2": desc.compute_pgm_rsrc2.0, "wavefront_size": meta.wavefront_size,
        "hazards": hz, "abi": abi,
    });
    Ok(sealed)
}

/// Unbundle `hxaco`, [`seal`] its device ELF, and require that re-bundling reproduces `hxaco`.
pub fn verify_hxaco(hxaco: &[u8], arch: Arch, symbol: &str) -> Result<Sealed, String> {
    if !Bundle::is_bundle(hxaco) {
        return Err("not a __CLANG_OFFLOAD_BUNDLE__".into());
    }
    let s = seal(hxaco, arch, symbol, None)?;
    if s.hxaco != hxaco {
        let at = s
            .hxaco
            .iter()
            .zip(hxaco)
            .position(|(a, b)| a != b)
            .unwrap_or(s.hxaco.len().min(hxaco.len()));
        return Err(format!(
            "re-bundling differs from the hxaco at byte offset {at:#x}"
        ));
    }
    Ok(s)
}

/// One kernel of a multi-kernel module for [`seal_many`]: its symbol and, optionally, the
/// incumbent object bytes and symbol whose explicit kernarg layout it must match.
#[derive(Clone, Copy, Debug)]
pub struct KernelSpec<'a> {
    pub symbol: &'a str,
    pub abi_ref: Option<(&'a [u8], &'a str)>,
}

/// [`seal`] every kernel of one object into a single `.hxaco`: each kernel gets its own lift
/// proof, hazard, SGPR and ABI checks; the returned [`Sealed`]s (one per spec, in order) share
/// the same bundle bytes. At least one spec is required and symbols must be distinct.
pub fn seal_many(
    input: &[u8],
    arch: Arch,
    specs: &[KernelSpec<'_>],
) -> Result<Vec<Sealed>, String> {
    if specs.is_empty() {
        return Err("seal_many: no kernels".into());
    }
    for (i, s) in specs.iter().enumerate() {
        if specs[..i].iter().any(|p| p.symbol == s.symbol) {
            return Err(format!("seal_many: duplicate kernel `{}`", s.symbol));
        }
    }
    let mut out: Vec<Sealed> = Vec::with_capacity(specs.len());
    for s in specs {
        let sealed = seal(input, arch, s.symbol, s.abi_ref)?;
        if let Some(first) = out.first() {
            if first.hxaco != sealed.hxaco {
                return Err("seal_many: kernels sealed to different bundles".into());
            }
        }
        out.push(sealed);
    }
    Ok(out)
}

/// [`verify_hxaco`] for every symbol of a multi-kernel `hxaco`.
pub fn verify_hxaco_many(
    hxaco: &[u8],
    arch: Arch,
    symbols: &[&str],
) -> Result<Vec<Sealed>, String> {
    symbols
        .iter()
        .map(|s| verify_hxaco(hxaco, arch, s))
        .collect()
}

#[cfg(test)]
mod gfx942_bundle_tests {
    use super::*;

    #[test]
    fn cdna_bundle_selects_only_its_own_device() {
        let hxaco = native::bundle(b"\x7fELF", Arch::Gfx942, HOST_TRIPLE);
        assert_eq!(device_elf(&hxaco, Arch::Gfx942).unwrap(), b"\x7fELF");
        assert!(device_elf(&hxaco, Arch::Gfx1201).is_err());
        let rdna = native::bundle(b"\x7fELF", Arch::Gfx1201, HOST_TRIPLE);
        assert!(device_elf(&rdna, Arch::Gfx942).is_err());
    }

    #[test]
    fn cdna_bundle_accepts_features_but_not_arch_prefixes() {
        let hxaco = native::bundle(b"\x7fELF", Arch::Gfx942, HOST_TRIPLE);
        let (mut bundle, payloads) = Bundle::read(&hxaco).unwrap();
        bundle.entries[1].triple.push_str(":xnack-:sramecc+");
        let features = bundle.write(&payloads).unwrap();
        assert_eq!(device_elf(&features, Arch::Gfx942).unwrap(), b"\x7fELF");
        bundle.entries[1].triple = "hipv4-amdgcn-amd-amdhsa--gfx9420".into();
        let wrong_arch = bundle.write(&payloads).unwrap();
        assert!(device_elf(&wrong_arch, Arch::Gfx942).is_err());
    }
}

#[cfg(test)]
mod gfx1200_tests {
    use super::*;

    /// gfx1200 maps to its own IR target and bundle triple, never aliasing gfx1201.
    #[test]
    fn gfx1200_has_its_own_ir_target_and_bundle_triple() {
        assert_eq!(ir_arch(Arch::Gfx1200), Ok(IrArch::Gfx1200));
        assert_eq!(ir_arch(Arch::Gfx1201), Ok(IrArch::Gfx1201));
        let hxaco = native::bundle(b"\x7fELF", Arch::Gfx1200, HOST_TRIPLE);
        let (bundle, payloads) = Bundle::read(&hxaco).unwrap();
        let idx = bundle.device_entry(IrArch::Gfx1200).unwrap();
        assert_eq!(payloads[idx], b"\x7fELF");
        assert!(bundle.device_entry(IrArch::Gfx1201).is_err());
        assert!(device_elf(&hxaco, Arch::Gfx1200).is_ok());
        assert!(device_elf(&hxaco, Arch::Gfx1201).is_err());
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;
    use peacemaker_ir::envelope::{SectionData, Symbol};
    use peacemaker_ir::helpers::{helper_symbol_name, SHT_DYNSYM, SHT_SYMTAB, STT_FUNC};

    const CO: &[u8] = include_bytes!("../tests/fixtures/hip_twin_saxpy_gfx1100.co");

    /// The fixture is a HIP bundle; helper extents are offsets into its device ELF payload.
    fn device() -> &'static [u8] {
        device_elf(CO, Arch::Gfx1100).expect("device elf")
    }

    fn envelope() -> Envelope {
        let lifted = lift_object(CO, OPTS).expect("lift");
        lifted.program.source.expect("source").elf
    }

    /// The same object with its metadata kernels forgotten: their `STT_FUNC` extents become
    /// out-of-line helpers decoded from the original ELF bytes.
    fn kernelless() -> Envelope {
        let mut env = envelope();
        env.kernels.clear();
        env
    }

    /// Run `f` on every symbol table's `STT_FUNC` symbol named `twin_saxpy`, then append
    /// `extra` clones (built by `f`) to that table.
    fn edit_symbols(env: &mut Envelope, f: impl Fn(&Symbol) -> Vec<Symbol>) {
        for table in 0..env.sections.len() {
            if !matches!(env.sections[table].header.sh_type, SHT_SYMTAB | SHT_DYNSYM) {
                continue;
            }
            let SectionData::Symbols(symbols) = &env.sections[table].data else {
                continue;
            };
            let target = symbols
                .iter()
                .find(|s| {
                    s.kind() == STT_FUNC
                        && helper_symbol_name(env, table, s).as_deref() == Ok("twin_saxpy")
                })
                .copied();
            let Some(target) = target else { continue };
            let extra = f(&target);
            let SectionData::Symbols(symbols) = &mut env.sections[table].data else {
                unreachable!()
            };
            symbols.extend(extra);
        }
    }

    #[test]
    fn kernels_alone_have_no_helpers() {
        let helpers = certify_helpers(&envelope(), device(), IrArch::Gfx1100).expect("certify");
        assert!(helpers.is_empty());
    }

    #[test]
    fn helper_extents_are_decoded_from_the_original_elf_and_reencoded_exactly() {
        let sealed = seal(CO, Arch::Gfx1100, "twin_saxpy", None).expect("seal");
        let helpers = certify_helpers(&kernelless(), device(), IrArch::Gfx1100).expect("certify");
        assert_eq!(helpers.len(), 2, "{helpers:?}");
        let saxpy = helpers
            .iter()
            .find(|h| h["symbol"] == "twin_saxpy")
            .expect("twin_saxpy extent");
        // The helper path reads the same instruction stream the kernel lift re-emits.
        assert_eq!(saxpy["stream_sha256"], sealed.text_sha256);
        assert_eq!(saxpy["bytes"], sealed.report["text_bytes"]);
        assert!(saxpy["instructions"].as_u64().unwrap() > 0);
        assert_eq!(saxpy["coverage"], "typed instruction re-encode");
        assert!(saxpy.get("aliases").is_none());
    }

    #[test]
    fn aliases_with_the_same_extent_merge_and_exact_duplicates_dedup() {
        let mut env = kernelless();
        edit_symbols(&mut env, |s| {
            // Exact duplicate (same name), and the same extent under another name.
            let mut other = *s;
            other.st_name += 1; // tail of the same string ("win_saxpy"): a distinct, valid name
            vec![*s, other]
        });
        let helpers = certify_helpers(&env, device(), IrArch::Gfx1100).expect("certify");
        assert_eq!(helpers.len(), 2);
        let saxpy = helpers
            .iter()
            .find(|h| h["symbol"] == "twin_saxpy")
            .expect("merged extent");
        assert_eq!(saxpy["aliases"], json!(["win_saxpy"]), "{saxpy}");
    }

    fn rejects(mutate: impl Fn(&Symbol) -> Vec<Symbol>, env: Envelope, why: &str) {
        let mut env = env;
        edit_symbols(&mut env, mutate);
        let e = certify_helpers(&env, device(), IrArch::Gfx1100).unwrap_err();
        assert!(e.contains(why), "expected `{why}` in `{e}`");
    }

    #[test]
    fn zero_sized_defined_function_is_rejected() {
        rejects(
            |s| {
                let mut z = *s;
                z.st_size = 0;
                vec![z]
            },
            envelope(),
            "size 0",
        );
    }

    #[test]
    fn extent_that_is_not_a_multiple_of_four_is_rejected() {
        rejects(
            |s| {
                let mut z = *s;
                z.st_size = s.st_size - 2;
                vec![z]
            },
            kernelless(),
            "multiple of 4",
        );
    }

    #[test]
    fn partial_overlap_with_a_kernel_is_rejected() {
        rejects(
            |s| {
                let mut z = *s;
                z.st_size = s.st_size - 4;
                vec![z]
            },
            envelope(),
            "partially overlaps kernel",
        );
    }

    #[test]
    fn inconsistent_helper_overlap_is_rejected() {
        rejects(
            |s| {
                let mut z = *s;
                z.st_size = s.st_size - 4;
                vec![z]
            },
            kernelless(),
            "overlaps helper",
        );
    }

    #[test]
    fn extent_outside_the_section_or_with_unresolvable_name_is_rejected() {
        rejects(
            |s| {
                let mut z = *s;
                z.st_size = 1 << 40;
                vec![z]
            },
            envelope(),
            "outside section",
        );
        rejects(
            |s| {
                let mut z = *s;
                z.st_name = u32::MAX;
                vec![z]
            },
            envelope(),
            "outside string table",
        );
    }

    #[test]
    fn undecodable_helper_words_are_rejected_not_passed_through() {
        let mut elf = device().to_vec();
        let env = kernelless();
        let bodies = helper_bodies(&env, device()).unwrap();
        let b = &bodies[0];
        for w in elf[b.offset..b.offset + b.extent.size as usize].chunks_exact_mut(4) {
            w.copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        }
        let e = certify_helpers(&env, &elf, IrArch::Gfx1100).unwrap_err();
        assert!(e.contains("helper `"), "{e}");
    }
}
