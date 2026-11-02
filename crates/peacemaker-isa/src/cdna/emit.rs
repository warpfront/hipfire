// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Code-object emission for CDNA kernels: the `.amdhsa_*` directive set and
//! descriptor (gfx90a+ layout, `peacemaker_cdna::kd`), the `.amdgpu_metadata`
//! (wavefront size 64, AGPR count), and the HSA code object v5 or v6 through
//! `crate::native::link_encoded`; plus the equivalent `.s` text that ROCm
//! `llvm-mc` + `ld.lld` (test oracle) assemble and link into the same bytes
//! except the `.comment` stamp.
//!
//! The directive set and metadata keys are what ROCm's `clang` (24.0.0git)
//! prints for a gfx942 kernel, minus the hidden kernel arguments and the
//! `.language` keys the builder has no use for:
//!
//! - `amdhsa.version` is `[1, 2]` for both code object versions; the versions
//!   differ in `.amdhsa_code_object_version`, the ELF ABI version and, in V5
//!   only, the `.amdhsa_reserve_xnack_mask 1` directive.
//! - gfx942 is wave64 (`.wavefront_size: 64`, no `.amdhsa_wavefront_size32`),
//!   has unified VGPR/AGPR files (`.amdhsa_accum_offset`, `.amdhsa_tg_split`)
//!   and no `.amdhsa_inst_pref_size`, `.amdhsa_memory_ordered` or
//!   `.amdhsa_workgroup_processor_mode`.
//! - `.amdhsa_next_free_vgpr` and metadata `.vgpr_count` are the unified total
//!   (`accum_offset + agprs`) when the kernel uses AccVGPRs, else the
//!   architectural count.
//! - metadata `.sgpr_count` is `next_free_sgpr` plus 6: LLVM counts VCC,
//!   FLAT_SCRATCH and XNACK_MASK (2 each) on gfx942 whether or not the kernel
//!   names them.
//!
//! Every input is validated against the gfx942 capability row; nothing is
//! clamped or rounded.
use super::{KernArg, KernArgKind, Kernel};
use crate::native::{self, EncodedKernel};
use crate::Arch;
use peacemaker_cdna::Inst;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeObjectVersion {
    V5,
    V6,
}

impl CodeObjectVersion {
    fn number(self) -> u8 {
        match self {
            Self::V5 => 5,
            Self::V6 => 6,
        }
    }
}

/// `.sgpr_count` minus `next_free_sgpr`: VCC, FLAT_SCRATCH, XNACK_MASK.
const EXTRA_SGPRS: u32 = 6;
const MAX_WORKGROUP: u32 = 1024;

/// A plain-scalar identifier: what the YAML reader keeps as a string and the
/// assembler accepts as a symbol (`[A-Za-z_][A-Za-z0-9_]*`, not a YAML
/// boolean/null spelling).
fn check_ident(what: &str, s: &str) -> Result<(), String> {
    const RESERVED: [&str; 24] = [
        "null", "Null", "NULL", "yes", "Yes", "YES", "no", "No", "NO", "on", "On", "ON", "off",
        "Off", "OFF", "true", "True", "TRUE", "false", "False", "FALSE", "inf", "nan", "NaN",
    ];
    let mut bytes = s.bytes();
    let ok = bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if !ok || RESERVED.contains(&s) {
        return Err(format!("{what} {s:?} is not a plain identifier"));
    }
    Ok(())
}

fn check_args(k: &Kernel) -> Result<(), String> {
    let mut end = 0u32;
    for (
        i,
        KernArg {
            name,
            offset,
            size,
            kind,
        },
    ) in k.kernargs.iter().enumerate()
    {
        check_ident(&format!("{}: kernel argument name", k.name), name)?;
        if k.kernargs[..i].iter().any(|p| p.name == *name) {
            return Err(format!("{}: duplicate kernel argument {name}", k.name));
        }
        // The size of a by-value argument is whatever the kernel declares (scalar or struct); only a pointer is fixed at 8 bytes, 8-aligned.
        if *size == 0 {
            return Err(format!("{}: argument {name}: size 0", k.name));
        }
        if matches!(kind, KernArgKind::GlobalBuffer { .. }) && (*size != 8 || offset % 8 != 0) {
            return Err(format!("{}: argument {name}: a global buffer is 8 bytes at an 8-aligned offset, got size {size} at {offset}", k.name));
        }
        if *offset < end {
            return Err(format!("{}: argument {name} at {offset} overlaps or precedes the previous argument (ends at {end})", k.name));
        }
        end = offset
            .checked_add(*size)
            .ok_or_else(|| format!("{}: argument {name} overflows the kernarg segment", k.name))?;
    }
    if end > k.resources.kernarg_size {
        return Err(format!(
            "{}: arguments end at {end} but kernarg_size is {}",
            k.name, k.resources.kernarg_size
        ));
    }
    Ok(())
}

/// Resource and argument limits of `k.arch`.
fn check_resources(k: &Kernel) -> Result<(), String> {
    let caps = k.arch.caps();
    let (r, n) = (&k.resources, &k.name);
    check_ident("kernel name", n)?;
    let system = 2 + r.workgroup_id.iter().filter(|&&on| on).count() as u32;
    if u32::from(r.next_free_sgpr) < system {
        return Err(format!(
            "{n}: next_free_sgpr {} is below the {system} system/user SGPRs",
            r.next_free_sgpr
        ));
    }
    if r.next_free_sgpr > caps.sgprs {
        return Err(format!(
            "{n}: next_free_sgpr {} exceeds {}",
            r.next_free_sgpr, caps.sgprs
        ));
    }
    if r.next_free_vgpr > caps.vgprs {
        return Err(format!(
            "{n}: next_free_vgpr {} exceeds {}",
            r.next_free_vgpr, caps.vgprs
        ));
    }
    if r.agprs > caps.agprs {
        return Err(format!("{n}: {} AGPRs exceed {}", r.agprs, caps.agprs));
    }
    // `Resources::accum_offset`: the architectural count rounded up to 4, at least 4 (the first AGPR).
    let accum = r.next_free_vgpr.next_multiple_of(4).max(4);
    if r.accum_offset != accum {
        return Err(format!(
            "{n}: accum_offset {} is not next_free_vgpr {} rounded up to 4, at least 4 ({accum})",
            r.accum_offset, r.next_free_vgpr
        ));
    }
    if total_vgprs(k) > caps.unified_vgprs {
        return Err(format!(
            "{n}: {} unified VGPRs exceed {}",
            total_vgprs(k),
            caps.unified_vgprs
        ));
    }
    if r.workitem_id > 2 {
        return Err(format!("{n}: workitem_id {} is not 0..=2", r.workitem_id));
    }
    if r.workgroup_size == 0 || r.workgroup_size > MAX_WORKGROUP {
        return Err(format!(
            "{n}: workgroup_size {} is not in 1..={MAX_WORKGROUP}",
            r.workgroup_size
        ));
    }
    if r.group_segment_fixed_size > caps.lds_bytes {
        return Err(format!(
            "{n}: LDS {} exceeds {}",
            r.group_segment_fixed_size, caps.lds_bytes
        ));
    }
    check_args(k)
}

/// The unified VGPR count of the descriptor and metadata.
fn total_vgprs(k: &Kernel) -> u16 {
    let r = &k.resources;
    if r.agprs > 0 {
        r.accum_offset + r.agprs
    } else {
        r.next_free_vgpr
    }
}

fn kernel_directives(
    k: &Kernel,
    version: CodeObjectVersion,
) -> Result<Vec<(String, String)>, String> {
    check_resources(k)?;
    let r = &k.resources;
    let mut d: Vec<(String, String)> = Vec::with_capacity(40);
    let mut put = |name: &str, value: u64| d.push((format!(".amdhsa_{name}"), value.to_string()));
    put(
        "group_segment_fixed_size",
        r.group_segment_fixed_size.into(),
    );
    put("private_segment_fixed_size", 0);
    put("kernarg_size", r.kernarg_size.into());
    // s[0:1] is the kernarg segment pointer, the only enabled user SGPR.
    put("user_sgpr_count", 2);
    put("user_sgpr_dispatch_ptr", 0);
    put("user_sgpr_queue_ptr", 0);
    put("user_sgpr_kernarg_segment_ptr", 1);
    put("user_sgpr_dispatch_id", 0);
    put("user_sgpr_kernarg_preload_length", 0);
    put("user_sgpr_kernarg_preload_offset", 0);
    put("user_sgpr_private_segment_size", 0);
    put("uses_dynamic_stack", 0);
    put("enable_private_segment", 0);
    put("system_sgpr_workgroup_id_x", r.workgroup_id[0].into());
    put("system_sgpr_workgroup_id_y", r.workgroup_id[1].into());
    put("system_sgpr_workgroup_id_z", r.workgroup_id[2].into());
    put("system_sgpr_workgroup_info", 0);
    put("system_vgpr_workitem_id", r.workitem_id.into());
    put("next_free_vgpr", total_vgprs(k).into());
    put("next_free_sgpr", r.next_free_sgpr.into());
    put("accum_offset", r.accum_offset.into());
    put("reserve_vcc", r.uses_vcc.into());
    if version == CodeObjectVersion::V5 {
        put("reserve_xnack_mask", 1)
    }
    put("float_round_mode_32", 0);
    put("float_round_mode_16_64", 0);
    put("float_denorm_mode_32", 3);
    put("float_denorm_mode_16_64", 3);
    put("dx10_clamp", 1);
    put("ieee_mode", 1);
    put("fp16_overflow", r.fp16_ovfl.into());
    put("tg_split", 0);
    for field in [
        "fp_ieee_invalid_op",
        "fp_denorm_src",
        "fp_ieee_div_zero",
        "fp_ieee_overflow",
        "fp_ieee_underflow",
        "fp_ieee_inexact",
        "int_div_zero",
    ] {
        put(&format!("exception_{field}"), 0);
    }
    Ok(d)
}

/// The `.amdhsa_*` directives (name, value) of `k`'s descriptor, exactly the
/// set the native descriptor writer accepts for gfx942 (code object V6; V5
/// adds `.amdhsa_reserve_xnack_mask 1` after `.amdhsa_reserve_vcc`). Names
/// carry the `.amdhsa_` prefix; `.amdhsa_inst_pref_size` does not exist on
/// gfx942.
pub fn directives(k: &Kernel) -> Result<Vec<(String, String)>, String> {
    kernel_directives(k, CodeObjectVersion::V6)
}

/// The `.amdgpu_metadata` YAML body (between `---` and `...`) for `kernels`.
pub fn metadata_yaml(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<String, String> {
    check_kernels(kernels)?;
    // Both versions carry metadata version 1.2 (LLVM `MetadataStreamerMsgPackV5`/`V6`).
    let (major, minor) = match version {
        CodeObjectVersion::V5 | CodeObjectVersion::V6 => (1, 2),
    };
    let mut y = String::from("amdhsa.kernels:\n");
    for k in kernels {
        check_resources(k)?;
        let r = &k.resources;
        let name = &k.name;
        // Keys in LLVM's sorted order; the first key carries the list dash.
        let _ = writeln!(y, "  - .agpr_count: {}", r.agprs);
        y.push_str("    .args:\n");
        for a in &k.kernargs {
            let mut lead = "      - ";
            let mut line = |key: &str, value: &dyn std::fmt::Display| {
                let _ = writeln!(y, "{lead}{key}: {value}");
                lead = "        "
            };
            if let KernArgKind::GlobalBuffer { readonly: true } = a.kind {
                line(".actual_access", &"read_only")
            }
            if let KernArgKind::GlobalBuffer { .. } = a.kind {
                line(".address_space", &"global")
            }
            if let KernArgKind::GlobalBuffer { readonly: true } = a.kind {
                line(".is_const", &"true")
            }
            line(".name", &a.name);
            line(".offset", &a.offset);
            line(".size", &a.size);
            line(
                ".value_kind",
                &match a.kind {
                    KernArgKind::GlobalBuffer { .. } => "global_buffer",
                    KernArgKind::ByValue => "by_value",
                },
            );
        }
        let _ = write!(y, "    .group_segment_fixed_size: {}\n    .kernarg_segment_align: 8\n    .kernarg_segment_size: {}\n    .max_flat_workgroup_size: {}\n    \
            .name: {name}\n    .private_segment_fixed_size: 0\n    .sgpr_count: {}\n    .sgpr_spill_count: 0\n    .symbol: {name}.kd\n    \
            .uniform_work_group_size: 1\n    .uses_dynamic_stack: false\n    .vgpr_count: {}\n    .vgpr_spill_count: 0\n    .wavefront_size: 64\n",
            r.group_segment_fixed_size, r.kernarg_size, r.workgroup_size, u32::from(r.next_free_sgpr) + EXTRA_SGPRS, total_vgprs(k));
    }
    let _ = writeln!(
        y,
        "amdhsa.target: amdgcn-amd-amdhsa--{}\namdhsa.version: [{major}, {minor}]",
        kernels[0].arch.name()
    );
    Ok(y)
}

/// Non-empty, one architecture, unique names.
fn check_kernels(kernels: &[&Kernel]) -> Result<(), String> {
    let first = kernels.first().ok_or("no kernel to emit")?;
    for (i, k) in kernels.iter().enumerate() {
        if k.arch != first.arch {
            return Err(format!(
                "{}: arch {} differs from {}",
                k.name, k.arch, first.arch
            ));
        }
        if kernels[..i].iter().any(|p| p.name == k.name) {
            return Err(format!("duplicate kernel {}", k.name));
        }
    }
    Ok(())
}

/// Re-encode `k.insts` and prove it is `k.words`; returns each instruction's
/// dword offset and the boundary-checked labels (`.L` prefix stripped).
fn layout(k: &Kernel) -> Result<(Vec<usize>, Vec<(String, usize)>), String> {
    let mut words = Vec::with_capacity(k.words.len());
    let mut starts = Vec::with_capacity(k.insts.len());
    for (i, inst) in k.insts.iter().enumerate() {
        starts.push(words.len());
        inst.encode(k.arch, &mut words)
            .map_err(|e| format!("{}: instruction {i} ({inst}): {e}", k.name))?;
    }
    if words != k.words {
        return Err(format!("{}: words are not the encoding of insts", k.name));
    }
    if words.is_empty() {
        return Err(format!("{}: empty kernel", k.name));
    }
    let mut labels = Vec::with_capacity(k.labels.len());
    for (name, at) in &k.labels {
        let plain = name.strip_prefix(".L").unwrap_or(name);
        if plain.is_empty()
            || !plain
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'$'))
        {
            return Err(format!(
                "{}: label {name:?} is not a plain local label",
                k.name
            ));
        }
        if plain == "end" {
            return Err(format!(
                "{}: label {name} collides with the kernel end label",
                k.name
            ));
        }
        if labels.iter().any(|(p, _)| p == plain) {
            return Err(format!("{}: duplicate label {name}", k.name));
        }
        if *at != words.len() && starts.binary_search(at).is_err() {
            return Err(format!(
                "{}: label {name} at dword {at} is not an instruction boundary",
                k.name
            ));
        }
        labels.push((plain.to_owned(), *at));
    }
    Ok((starts, labels))
}

/// The native code object.
pub fn code_object(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<Vec<u8>, String> {
    check_kernels(kernels)?;
    let arch = match kernels[0].arch {
        peacemaker_cdna::Arch::Gfx942 => Arch::Gfx942,
    };
    let sets = kernels
        .iter()
        .map(|k| kernel_directives(k, version))
        .collect::<Result<Vec<_>, _>>()?;
    let yaml = metadata_yaml(kernels, version)?;
    let encoded: Vec<EncodedKernel<'_>> = kernels
        .iter()
        .zip(&sets)
        .map(|(k, d)| EncodedKernel {
            symbol: &k.name,
            words: &k.words,
            directives: d,
        })
        .collect();
    native::link_encoded_versioned(&encoded, &yaml, arch, u32::from(version.number()))
}

/// `.s` text for the ROCm oracle (instructions printed by their `Display`).
pub fn assembly(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<String, String> {
    check_kernels(kernels)?;
    let mut s = format!(
        ".amdgcn_target \"amdgcn-amd-amdhsa--{}\"\n.amdhsa_code_object_version {}\n",
        kernels[0].arch.name(),
        version.number()
    );
    for k in kernels {
        let (starts, labels) = layout(k)?;
        let sym = &k.name;
        let dir = kernel_directives(k, version)?;
        let _ = write!(
            s,
            ".text\n.protected {sym}\n.globl {sym}\n.p2align 8\n.type {sym},@function\n{sym}:\n"
        );
        let label_at = |at: usize| labels.iter().find(|(_, o)| *o == at).map(|(n, _)| n);
        let mut at = 0usize;
        for (index, inst) in k.insts.iter().enumerate() {
            for (name, _) in labels.iter().filter(|(_, o)| *o == at) {
                let _ = writeln!(s, ".L{sym}_{name}:");
            }
            match (inst, inst.branch_offset()) {
                (Inst::Sopp(p), Some(offset)) => {
                    let target = at as i64 + 1 + i64::from(offset);
                    if target < 0
                        || target as usize > k.words.len()
                        || (target as usize != k.words.len()
                            && starts.binary_search(&(target as usize)).is_err())
                    {
                        return Err(format!("{sym}: {inst} at dword {at} targets dword {target}, not an instruction boundary of the kernel"));
                    }
                    match label_at(target as usize) {
                        Some(name) => {
                            let _ = writeln!(s, "\t{} .L{sym}_{name}", p.op.name());
                        }
                        None => {
                            let _ = writeln!(s, "\t{} {offset}", p.op.name());
                        }
                    }
                }
                _ => {
                    let _ = writeln!(s, "\t{inst}");
                }
            }
            at = starts.get(index + 1).copied().unwrap_or(k.words.len());
        }
        for (name, _) in labels.iter().filter(|(_, o)| *o == k.words.len()) {
            let _ = writeln!(s, ".L{sym}_{name}:");
        }
        let _ = write!(s, ".L{sym}_end:\n.size {sym}, .L{sym}_end-{sym}\n.section .rodata,\"a\",@progbits\n.p2align 6\n.amdhsa_kernel {sym}\n");
        for (name, value) in &dir {
            let _ = writeln!(s, "\t{name} {value}");
        }
        s.push_str(".end_amdhsa_kernel\n");
    }
    s.push_str(".text\n.amdgpu_metadata\n---\n");
    s.push_str(&metadata_yaml(kernels, version)?);
    s.push_str("...\n.end_amdgpu_metadata\n");
    Ok(s)
}
