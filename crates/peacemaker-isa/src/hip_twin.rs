// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Route A ("lift-and-repass") sealing of a hipcc-compiled kernel as an embedded per-arch
//! code object. The hipcc object is lifted (`Frontend::Hipcc`), re-emitted with
//! `emit::checked`, and the emitted bytes must equal the input ELF byte for byte; the result is
//! bundled with [`native::bundle`] and sha-pinned. See `examples/hip_twin.rs`.

use crate::arch::Arch;
use crate::native;
use peacemaker_ir::envelope::Bundle;
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
        Arch::Gfx1201 => Ok(IrArch::Gfx1201),
        Arch::Gfx942 => Err("hip_twin: gfx942 is not liftable (no CDNA lift target)".into()),
    }
}

/// The device ELF of `input`: the bytes themselves when they are not a bundle, else the HIP
/// device entry for `arch`.
fn device_elf(input: &[u8], arch: Arch) -> Result<&[u8], String> {
    if !Bundle::is_bundle(input) {
        return Ok(input);
    }
    let (bundle, payloads) = Bundle::read(input).map_err(|e| format!("bundle: {e}"))?;
    let idx = bundle
        .device_entry(ir_arch(arch)?)
        .map_err(|e| format!("bundle has no {} device entry: {e}", arch.name()))?;
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
    let text = emit::bytes(kernel, ir).map_err(|e| format!("emit text: {e}"))?;

    // Static hazards (gfx11 arches).
    let mut hz = json!({"checked": false});
    let mut hz_err = Vec::new();
    if matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
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
    Ok(s)
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
