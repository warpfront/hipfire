// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `.amdhsa_kernel` directives → the 64-byte kernel descriptor (AMDGPU
//! code object v5/v6; LLVM `AMDGPUUsage`, "Kernel Descriptor").
//!
//! gfx11/gfx12 (wave32): exactly the directive set the builder writes is
//! accepted, each once and each within its field; the code entry offset is
//! filled in by the linker layout. `.amdhsa_next_free_sgpr` and
//! `.amdhsa_reserve_vcc` are checked but encode nothing on GFX10+
//! (`GRANULATED_WAVEFRONT_SGPR_COUNT` is zero).
//!
//! gfx942 (CDNA3, wave64, GFX90A-family layout): [`encode_cdna`] mirrors the
//! `llvm-mc` (AMD LLVM 24.0.0git) `.amdhsa_kernel` parser: the same directive
//! names, ranges, defaults and diagnostics, the unified VGPR+AGPR count with
//! `.amdhsa_accum_offset`, the GFX9 SGPR extras, and the descriptor assembled
//! through `peacemaker_cdna::kd::KernelDescriptor`. There is no
//! `.amdhsa_inst_pref_size` and no wavefront-size directive on gfx942.
use crate::Arch;
use std::collections::BTreeMap;

pub(crate) const KD_SIZE: usize = 64;
/// Byte offset of `kernel_code_entry_byte_offset` (i64).
pub(crate) const ENTRY_OFFSET: usize = 16;

/// Where a directive's value goes: (word byte offset, low bit, width).
fn field(arch: Arch, name: &str) -> Option<(usize, u32, u32)> {
    const RSRC3: usize = 44;
    const RSRC1: usize = 48;
    const RSRC2: usize = 52;
    const PROPS: usize = 56;
    Some(match name {
        "group_segment_fixed_size" => (0, 0, 32),
        "private_segment_fixed_size" => (4, 0, 32),
        "kernarg_size" => (8, 0, 32),
        "user_sgpr_dispatch_ptr" => (PROPS, 1, 1),
        "user_sgpr_queue_ptr" => (PROPS, 2, 1),
        "user_sgpr_kernarg_segment_ptr" => (PROPS, 3, 1),
        "user_sgpr_dispatch_id" => (PROPS, 4, 1),
        "user_sgpr_private_segment_size" => (PROPS, 6, 1),
        "wavefront_size32" => (PROPS, 10, 1),
        "uses_dynamic_stack" => (PROPS, 11, 1),
        "enable_private_segment" => (RSRC2, 0, 1),
        "user_sgpr_count" => (RSRC2, 1, 5),
        "system_sgpr_workgroup_id_x" => (RSRC2, 7, 1),
        "system_sgpr_workgroup_id_y" => (RSRC2, 8, 1),
        "system_sgpr_workgroup_id_z" => (RSRC2, 9, 1),
        "system_sgpr_workgroup_info" => (RSRC2, 10, 1),
        "system_vgpr_workitem_id" => (RSRC2, 11, 2),
        "exception_fp_ieee_invalid_op" => (RSRC2, 24, 1),
        "exception_fp_denorm_src" => (RSRC2, 25, 1),
        "exception_fp_ieee_div_zero" => (RSRC2, 26, 1),
        "exception_fp_ieee_overflow" => (RSRC2, 27, 1),
        "exception_fp_ieee_underflow" => (RSRC2, 28, 1),
        "exception_fp_ieee_inexact" => (RSRC2, 29, 1),
        "exception_int_div_zero" => (RSRC2, 30, 1),
        "float_round_mode_32" => (RSRC1, 12, 2),
        "float_round_mode_16_64" => (RSRC1, 14, 2),
        "float_denorm_mode_32" => (RSRC1, 16, 2),
        "float_denorm_mode_16_64" => (RSRC1, 18, 2),
        "fp16_overflow" => (RSRC1, 26, 1),
        "workgroup_processor_mode" => (RSRC1, 29, 1),
        "memory_ordered" => (RSRC1, 30, 1),
        "forward_progress" => (RSRC1, 31, 1),
        // GFX12 renames bit 21 (`WG_RR_EN`); GFX11 has DX10_CLAMP there and IEEE at 23.
        "round_robin_scheduling" if arch.gfx12() => (RSRC1, 21, 1),
        "dx10_clamp" if !arch.gfx12() => (RSRC1, 21, 1),
        "ieee_mode" if !arch.gfx12() => (RSRC1, 23, 1),
        "shared_vgpr_count" if !arch.gfx12() => (RSRC3, 0, 4),
        "inst_pref_size" => (RSRC3, 4, if arch.gfx12() { 8 } else { 6 }),
        _ => return None,
    })
}

/// Directives that are range-checked but encode nothing on GFX10+.
const CHECKED_ONLY: [&str; 3] = ["next_free_vgpr", "next_free_sgpr", "reserve_vcc"];

/// `instprefsize(size)`: instruction cache lines (128 B on GFX11+) the
/// kernel's code spans, saturated at the field maximum.
fn inst_pref_size(arch: Arch, code_bytes: u64) -> u64 {
    code_bytes
        .div_ceil(128)
        .min(if arch.gfx12() { 0xff } else { 0x3f })
}

/// `((instprefsize(.Lend-sym)<<4)&MASK)>>4`, the builder's spelling.
fn inst_pref_expr(
    arch: Arch,
    value: &str,
    end_label: &str,
    symbol: &str,
    code_bytes: u64,
) -> Result<u64, String> {
    let mask = if arch.gfx12() { 4080 } else { 1008 };
    let inner = value
        .strip_prefix("((instprefsize(")
        .and_then(|v| v.strip_suffix(&format!(")<<4)&{mask})>>4")))
        .ok_or_else(|| format!(".amdhsa_inst_pref_size: unsupported expression {value}"))?;
    if inner != format!("{end_label}-{symbol}") {
        return Err(format!(
            ".amdhsa_inst_pref_size: {inner} is not the extent of {symbol}"
        ));
    }
    Ok(((inst_pref_size(arch, code_bytes) << 4) & mask) >> 4)
}

/// The descriptor of `symbol` from its `.amdhsa_kernel` block (entry offset zero).
pub(crate) fn encode(
    arch: Arch,
    symbol: &str,
    directives: &[(&str, &str)],
    end_label: &str,
    code_bytes: u64,
) -> Result<[u8; KD_SIZE], String> {
    let mut values: BTreeMap<&str, &str> = BTreeMap::new();
    for &(name, value) in directives {
        let key = name
            .strip_prefix(".amdhsa_")
            .ok_or_else(|| format!("{symbol}: unexpected directive {name}"))?;
        if values.insert(key, value).is_some() {
            return Err(format!("{symbol}: duplicate .amdhsa_{key}"));
        }
    }
    if arch.wave64() {
        return encode_cdna(symbol, &values);
    }
    let mut expected: Vec<&str> = [
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_size",
        "user_sgpr_count",
        "user_sgpr_dispatch_ptr",
        "user_sgpr_queue_ptr",
        "user_sgpr_kernarg_segment_ptr",
        "user_sgpr_dispatch_id",
        "user_sgpr_private_segment_size",
        "wavefront_size32",
        "uses_dynamic_stack",
        "enable_private_segment",
        "system_sgpr_workgroup_id_x",
        "system_sgpr_workgroup_id_y",
        "system_sgpr_workgroup_id_z",
        "system_sgpr_workgroup_info",
        "system_vgpr_workitem_id",
        "next_free_vgpr",
        "next_free_sgpr",
        "reserve_vcc",
        "float_round_mode_32",
        "float_round_mode_16_64",
        "float_denorm_mode_32",
        "float_denorm_mode_16_64",
        "fp16_overflow",
        "workgroup_processor_mode",
        "memory_ordered",
        "forward_progress",
        "inst_pref_size",
        "exception_fp_ieee_invalid_op",
        "exception_fp_denorm_src",
        "exception_fp_ieee_div_zero",
        "exception_fp_ieee_overflow",
        "exception_fp_ieee_underflow",
        "exception_fp_ieee_inexact",
        "exception_int_div_zero",
    ]
    .into();
    if arch.gfx12() {
        expected.push("round_robin_scheduling")
    } else {
        expected.extend(["dx10_clamp", "ieee_mode", "shared_vgpr_count"])
    }
    if let Some(extra) = values.keys().find(|k| !expected.contains(k)) {
        return Err(format!(
            "{symbol}: unsupported .amdhsa_{extra} on {}",
            arch.name()
        ));
    }
    if let Some(missing) = expected.iter().find(|k| !values.contains_key(*k)) {
        return Err(format!("{symbol}: missing .amdhsa_{missing}"));
    }

    let number = |key: &str| -> Result<u64, String> {
        let v = values[key];
        let parsed = if let Some(hex) = v.strip_prefix("0x") {
            u64::from_str_radix(hex, 16)
        } else {
            v.parse()
        };
        parsed.map_err(|_| format!("{symbol}: .amdhsa_{key} {v} is not an integer"))
    };
    if number("wavefront_size32")? != 1 {
        return Err(format!(
            "{symbol}: the native writer emits wave32 kernels only"
        ));
    }
    let mut kd = [0u8; KD_SIZE];
    for (&key, &value) in &values {
        if CHECKED_ONLY.contains(&key) {
            continue;
        }
        let (offset, low, width) =
            field(arch, key).ok_or_else(|| format!("{symbol}: unsupported .amdhsa_{key}"))?;
        let v = if key == "inst_pref_size" {
            inst_pref_expr(arch, value, end_label, symbol, code_bytes)?
        } else {
            number(key)?
        };
        if width < 64 && v >> width != 0 {
            return Err(format!(
                "{symbol}: .amdhsa_{key} {v} exceeds its {width}-bit field"
            ));
        }
        let word_bytes = if offset == 56 { 2 } else { 4 };
        let mut word = [0u8; 8];
        word[..word_bytes].copy_from_slice(&kd[offset..offset + word_bytes]);
        let merged = u64::from_le_bytes(word) | (v << low);
        kd[offset..offset + word_bytes].copy_from_slice(&merged.to_le_bytes()[..word_bytes]);
    }
    // GRANULATED_WORKITEM_VGPR_COUNT: wave32 encoding granule of 8 VGPRs.
    let vgprs = number("next_free_vgpr")?;
    if vgprs > 256 {
        return Err(format!(
            "{symbol}: .amdhsa_next_free_vgpr {vgprs} exceeds 256"
        ));
    }
    let blocks = (vgprs.max(1).div_ceil(8) - 1) as u32;
    let rsrc1 = u32::from_le_bytes(kd[48..52].try_into().unwrap()) | blocks;
    kd[48..52].copy_from_slice(&rsrc1.to_le_bytes());
    if number("next_free_sgpr")? > 106 || number("reserve_vcc")? > 1 {
        return Err(format!("{symbol}: SGPR count out of range"));
    }
    // `.amdhsa_user_sgpr_count` must cover the enabled user SGPRs.
    let implied = 2
        * (number("user_sgpr_dispatch_ptr")?
            + number("user_sgpr_queue_ptr")?
            + number("user_sgpr_kernarg_segment_ptr")?
            + number("user_sgpr_dispatch_id")?)
        + number("user_sgpr_private_segment_size")?;
    if number("user_sgpr_count")? < implied {
        return Err(format!(
            "{symbol}: .amdhsa_user_sgpr_count is below the {implied} enabled user SGPRs"
        ));
    }
    let rsrc2 = u32::from_le_bytes(kd[52..56].try_into().unwrap());
    arch.check_user_system_sgprs(symbol, rsrc2)?;
    Ok(kd)
}

/// Directives `llvm-mc` accepts for a gfx942 kernel (every one optional except
/// `next_free_vgpr`, `next_free_sgpr` and `accum_offset`).
const CDNA_DIRECTIVES: &[&str] = &[
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "kernarg_size",
    "user_sgpr_count",
    "user_sgpr_dispatch_ptr",
    "user_sgpr_queue_ptr",
    "user_sgpr_kernarg_segment_ptr",
    "user_sgpr_dispatch_id",
    "user_sgpr_kernarg_preload_length",
    "user_sgpr_kernarg_preload_offset",
    "user_sgpr_private_segment_size",
    "uses_dynamic_stack",
    "enable_private_segment",
    "system_sgpr_workgroup_id_x",
    "system_sgpr_workgroup_id_y",
    "system_sgpr_workgroup_id_z",
    "system_sgpr_workgroup_info",
    "system_vgpr_workitem_id",
    "next_free_vgpr",
    "next_free_sgpr",
    "accum_offset",
    "reserve_vcc",
    "reserve_xnack_mask",
    "float_round_mode_32",
    "float_round_mode_16_64",
    "float_denorm_mode_32",
    "float_denorm_mode_16_64",
    "dx10_clamp",
    "ieee_mode",
    "fp16_overflow",
    "tg_split",
    "exception_fp_ieee_invalid_op",
    "exception_fp_denorm_src",
    "exception_fp_ieee_div_zero",
    "exception_fp_ieee_overflow",
    "exception_fp_ieee_underflow",
    "exception_fp_ieee_inexact",
    "exception_int_div_zero",
];

/// Directives `llvm-mc` knows but rejects on gfx942: the GFX10+/GFX11+/GFX12 ones, and the
/// flat-scratch-setup ones (gfx942 has architected flat scratch: no private segment buffer,
/// flat scratch init, scratch wave offset SGPR, and `reserve_flat_scratch` is not settable).
const CDNA_REJECTED: &[&str] = &[
    "reserve_flat_scratch",
    "user_sgpr_private_segment_buffer",
    "user_sgpr_flat_scratch_init",
    "system_sgpr_private_segment_wavefront_offset",
    "wavefront_size32",
    "workgroup_processor_mode",
    "memory_ordered",
    "forward_progress",
    "inst_pref_size",
    "shared_vgpr_count",
    "round_robin_scheduling",
    "named_barrier_count",
];

/// SGPRs the hardware reserves above `.amdhsa_next_free_sgpr` on GFX9 (LLVM `getNumExtraSGPRs`): VCC
/// (2), the XNACK mask (+2) and flat scratch (+2) make 6. gfx942's target ID is `xnack any` (the XNACK
/// mask is always reserved: `.amdhsa_reserve_xnack_mask 0` is an `llvm-mc` error) and it has
/// architected flat scratch (counted as in use whatever `.amdhsa_reserve_vcc` says: `llvm-mc` encodes
/// the same SGPR granule for `reserve_vcc 0` and `1`).
const CDNA_EXTRA_SGPRS: u64 = 2 + 2 + 2;
/// `.amdhsa_next_free_sgpr` ceiling (GFX8+ addressable SGPRs).
const CDNA_MAX_SGPR: u64 = 102;
/// `.amdhsa_next_free_vgpr` ceiling: unified VGPR + AGPR file (512 registers).
const CDNA_MAX_VGPR: u64 = 512;
/// User SGPR ceiling.
const CDNA_MAX_USER_SGPR: u64 = 16;

/// An unsigned decimal or `0x` hexadecimal directive value.
fn cdna_number(symbol: &str, key: &str, text: &str) -> Result<u64, String> {
    let parsed = match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None if text.bytes().all(|b| b.is_ascii_digit()) && !text.is_empty() => text.parse(),
        None => {
            return Err(format!(
                "{symbol}: .amdhsa_{key} {text} is not an unsigned integer"
            ))
        }
    };
    parsed.map_err(|_| format!("{symbol}: .amdhsa_{key} {text} is out of range"))
}

/// The gfx942 descriptor of `symbol` from its `.amdhsa_kernel` directives (code entry offset zero),
/// bit for bit what `llvm-mc` assembles for them.
fn encode_cdna(symbol: &str, values: &BTreeMap<&str, &str>) -> Result<[u8; KD_SIZE], String> {
    use peacemaker_cdna::kd::KernelDescriptor;
    for key in values.keys() {
        if CDNA_REJECTED.contains(key) {
            return Err(format!(
                "{symbol}: .amdhsa_{key} is not supported on gfx942"
            ));
        }
        if !CDNA_DIRECTIVES.contains(key) {
            return Err(format!("{symbol}: unknown .amdhsa_{key} directive"));
        }
    }
    // A directive's value within `0..=max`, or `default` when absent (`llvm-mc`'s defaults).
    let get = |key: &str, default: u64, max: u64| -> Result<u64, String> {
        match values.get(key) {
            None => Ok(default),
            Some(text) => {
                let n = cdna_number(symbol, key, text)?;
                if n > max {
                    return Err(format!(
                        "{symbol}: .amdhsa_{key} {text} is out of range (at most {max})"
                    ));
                }
                Ok(n)
            }
        }
    };
    let required = |key: &str, max: u64| -> Result<u64, String> {
        if !values.contains_key(key) {
            return Err(format!("{symbol}: .amdhsa_{key} directive is required"));
        }
        get(key, 0, max)
    };
    let next_free_vgpr = required("next_free_vgpr", CDNA_MAX_VGPR)?;
    let next_free_sgpr = required("next_free_sgpr", CDNA_MAX_SGPR)?;
    let accum_offset = required("accum_offset", 256)?;
    get("reserve_vcc", 1, 1)?;
    if get("reserve_xnack_mask", 1, 1)? != 1 {
        return Err(format!(
            "{symbol}: .amdhsa_reserve_xnack_mask does not match target id (gfx942 is xnack any)"
        ));
    }

    // ACCUM_OFFSET: the first AGPR, a multiple of 4 in 4..=256, within the VGPR allocation.
    if accum_offset < 4 || accum_offset % 4 != 0 {
        return Err(format!(
            "{symbol}: accum_offset should be in range [4..256] in increments of 4"
        ));
    }
    if accum_offset > next_free_vgpr.max(1).next_multiple_of(4) {
        return Err(format!(
            "{symbol}: accum_offset exceeds total VGPR allocation"
        ));
    }

    // User SGPRs: dispatch_ptr, queue_ptr, kernarg_segment_ptr and dispatch_id take 2 each, the private
    // segment size 1, kernarg preload one per preloaded dword.
    let preload_length = get("user_sgpr_kernarg_preload_length", 0, CDNA_MAX_USER_SGPR)?;
    let implied = 2
        * (get("user_sgpr_dispatch_ptr", 0, 1)?
            + get("user_sgpr_queue_ptr", 0, 1)?
            + get("user_sgpr_kernarg_segment_ptr", 0, 1)?
            + get("user_sgpr_dispatch_id", 0, 1)?)
        + get("user_sgpr_private_segment_size", 0, 1)?
        + preload_length;
    if implied > CDNA_MAX_USER_SGPR {
        return Err(format!("{symbol}: too many user SGPRs enabled, found {implied}, but only {CDNA_MAX_USER_SGPR} are supported"));
    }
    let user_sgpr_count = get("user_sgpr_count", implied, CDNA_MAX_USER_SGPR)?;
    if user_sgpr_count < implied {
        return Err(format!("{symbol}: .amdhsa_user_sgpr_count {user_sgpr_count} is smaller than the {implied} implied by the enabled user SGPRs"));
    }

    // COMPUTE_PGM_RSRC1 (GFX9 layout). VGPR granule 8 (wave64, unified file), SGPR encoding granule 8.
    let vgpr_blocks = next_free_vgpr.max(1).div_ceil(8) - 1;
    let sgpr_blocks = (next_free_sgpr + CDNA_EXTRA_SGPRS).max(1).div_ceil(8) - 1;
    let rsrc1 = vgpr_blocks
        | sgpr_blocks << 6
        | get("float_round_mode_32", 0, 3)? << 12
        | get("float_round_mode_16_64", 0, 3)? << 14
        | get("float_denorm_mode_32", 0, 3)? << 16
        | get("float_denorm_mode_16_64", 3, 3)? << 18
        | get("dx10_clamp", 1, 1)? << 21
        | get("ieee_mode", 1, 1)? << 23
        | get("fp16_overflow", 0, 1)? << 26;
    // COMPUTE_PGM_RSRC2.
    let mut rsrc2 = get("enable_private_segment", 0, 1)?
        | user_sgpr_count << 1
        | get("system_sgpr_workgroup_id_x", 1, 1)? << 7
        | get("system_sgpr_workgroup_id_y", 0, 1)? << 8
        | get("system_sgpr_workgroup_id_z", 0, 1)? << 9
        | get("system_sgpr_workgroup_info", 0, 1)? << 10
        | get("system_vgpr_workitem_id", 0, 3)? << 11;
    for (bit, key) in [
        "exception_fp_ieee_invalid_op",
        "exception_fp_denorm_src",
        "exception_fp_ieee_div_zero",
        "exception_fp_ieee_overflow",
        "exception_fp_ieee_underflow",
        "exception_fp_ieee_inexact",
        "exception_int_div_zero",
    ]
    .into_iter()
    .enumerate()
    {
        rsrc2 |= get(key, 0, 1)? << (24 + bit);
    }
    // COMPUTE_PGM_RSRC3 (GFX90A/GFX942): ACCUM_OFFSET [5:0] = offset / 4 - 1, TG_SPLIT [16].
    let rsrc3 = (accum_offset / 4 - 1) | get("tg_split", 0, 1)? << 16;
    let properties = get("user_sgpr_dispatch_ptr", 0, 1)? << 1
        | get("user_sgpr_queue_ptr", 0, 1)? << 2
        | get("user_sgpr_kernarg_segment_ptr", 0, 1)? << 3
        | get("user_sgpr_dispatch_id", 0, 1)? << 4
        | get("user_sgpr_private_segment_size", 0, 1)? << 6
        | get("uses_dynamic_stack", 0, 1)? << 11;
    // KERNARG_PRELOAD_SPEC: LENGTH [6:0], OFFSET [15:7].
    let preload = preload_length | get("user_sgpr_kernarg_preload_offset", 0, 511)? << 7;

    let kd = KernelDescriptor {
        group_segment_fixed_size: get("group_segment_fixed_size", 0, u32::MAX.into())? as u32,
        private_segment_fixed_size: get("private_segment_fixed_size", 0, u32::MAX.into())? as u32,
        kernarg_size: get("kernarg_size", 0, u32::MAX.into())? as u32,
        kernel_code_entry_byte_offset: 0,
        compute_pgm_rsrc3: rsrc3 as u32,
        compute_pgm_rsrc1: rsrc1 as u32,
        compute_pgm_rsrc2: rsrc2 as u32,
        kernel_code_properties: properties as u16,
        kernarg_preload: preload as u16,
    };
    Ok(kd.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kd(list: &[(&str, &str)]) -> Result<[u8; KD_SIZE], String> {
        let owned: Vec<(String, String)> = list
            .iter()
            .map(|&(k, v)| (format!(".amdhsa_{k}"), v.to_owned()))
            .collect();
        let borrowed: Vec<(&str, &str)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        encode(Arch::Gfx942, "k", &borrowed, ".Lk_end", 8)
    }
    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
    const MIN: [(&str, &str); 3] = [
        ("next_free_vgpr", "8"),
        ("next_free_sgpr", "16"),
        ("accum_offset", "8"),
    ];

    // The expected bytes below are `llvm-mc -mcpu=gfx942` (AMD LLVM 24.0.0git) output for the same directives.
    #[test]
    fn minimal_directives_take_llvm_defaults() {
        assert_eq!(
            hex(&kd(&MIN).unwrap()),
            "0".repeat(88) + "01000000" + "8000ac00" + "80000000" + "00000000" + "00000000"
        );
    }

    #[test]
    fn every_directive_lands_in_its_field() {
        let full = [
            ("group_segment_fixed_size", "4096"),
            ("private_segment_fixed_size", "16"),
            ("kernarg_size", "56"),
            ("user_sgpr_count", "8"),
            ("user_sgpr_dispatch_ptr", "1"),
            ("user_sgpr_queue_ptr", "0"),
            ("user_sgpr_kernarg_segment_ptr", "1"),
            ("user_sgpr_dispatch_id", "0"),
            ("user_sgpr_kernarg_preload_length", "2"),
            ("user_sgpr_kernarg_preload_offset", "3"),
            ("user_sgpr_private_segment_size", "0"),
            ("uses_dynamic_stack", "1"),
            ("enable_private_segment", "1"),
            ("system_sgpr_workgroup_id_x", "1"),
            ("system_sgpr_workgroup_id_y", "1"),
            ("system_sgpr_workgroup_id_z", "1"),
            ("system_sgpr_workgroup_info", "1"),
            ("system_vgpr_workitem_id", "2"),
            ("next_free_vgpr", "100"),
            ("next_free_sgpr", "37"),
            ("accum_offset", "64"),
            ("reserve_vcc", "0"),
            ("reserve_xnack_mask", "1"),
            ("float_round_mode_32", "1"),
            ("float_round_mode_16_64", "2"),
            ("float_denorm_mode_32", "3"),
            ("float_denorm_mode_16_64", "1"),
            ("dx10_clamp", "0"),
            ("ieee_mode", "0"),
            ("fp16_overflow", "1"),
            ("tg_split", "1"),
            ("exception_fp_ieee_invalid_op", "1"),
            ("exception_fp_denorm_src", "0"),
            ("exception_fp_ieee_div_zero", "1"),
            ("exception_fp_ieee_overflow", "0"),
            ("exception_fp_ieee_underflow", "1"),
            ("exception_fp_ieee_inexact", "0"),
            ("exception_int_div_zero", "1"),
        ];
        assert_eq!(hex(&kd(&full).unwrap()),
            "00100000100000003800000000000000000000000000000000000000000000000000000000000000000000000f0001004c910704911700550a08820100000000");
    }

    #[test]
    fn kernarg_and_workgroup_y() {
        let mut d = MIN.to_vec();
        d.extend([
            ("user_sgpr_kernarg_segment_ptr", "1"),
            ("kernarg_size", "24"),
            ("system_sgpr_workgroup_id_y", "1"),
        ]);
        assert_eq!(hex(&kd(&d).unwrap()),
            "0000000000000000180000000000000000000000000000000000000000000000000000000000000000000000010000008000ac00840100000800000000000000");
    }

    /// `(next_free_vgpr, accum_offset, next_free_sgpr, RSRC1, RSRC3)` from `llvm-mc`.
    #[test]
    fn register_granules_follow_llvm() {
        for (vgpr, accum, sgpr, rsrc1, rsrc3) in [
            (1, 4, 16, 0xac_0080u32, 0),
            (8, 4, 16, 0xac_0080, 0),
            (9, 12, 16, 0xac_0081, 2),
            (17, 20, 16, 0xac_0082, 4),
            (64, 32, 16, 0xac_0087, 7),
            (256, 128, 16, 0xac_009f, 0x1f),
            (257, 256, 16, 0xac_00a0, 0x3f),
            (512, 256, 16, 0xac_00bf, 0x3f),
            (8, 8, 0, 0xac_0000, 1),
            (8, 8, 10, 0xac_0040, 1),
            (8, 8, 19, 0xac_00c0, 1),
            (8, 8, 98, 0xac_0300, 1),
            (8, 8, 99, 0xac_0340, 1),
            (8, 8, 102, 0xac_0340, 1),
        ] {
            let b = kd(&[
                ("next_free_vgpr", &vgpr.to_string()),
                ("next_free_sgpr", &sgpr.to_string()),
                ("accum_offset", &accum.to_string()),
            ])
            .unwrap();
            let d = peacemaker_cdna::kd::KernelDescriptor::from_bytes(&b).unwrap();
            assert_eq!(
                (d.compute_pgm_rsrc1, d.compute_pgm_rsrc3),
                (rsrc1, rsrc3),
                "vgpr {vgpr} accum {accum} sgpr {sgpr}"
            );
        }
    }

    #[test]
    fn llvm_diagnostics_are_reproduced() {
        let with = |extra: &[(&str, &str)]| {
            let mut d = MIN.to_vec();
            d.extend_from_slice(extra);
            kd(&d)
        };
        for (extra, message) in [
            (
                &[("user_sgpr_count", "1"), ("user_sgpr_dispatch_ptr", "1")][..],
                "smaller than the 2 implied",
            ),
            (&[("user_sgpr_count", "17")], "out of range (at most 16)"),
            (&[("user_sgpr_kernarg_preload_length", "17")], "at most 16"),
            (
                &[("user_sgpr_kernarg_preload_offset", "512")],
                "at most 511",
            ),
            (&[("wavefront_size32", "0")], "not supported on gfx942"),
            (&[("inst_pref_size", "1")], "not supported on gfx942"),
            (&[("reserve_flat_scratch", "0")], "not supported on gfx942"),
            (
                &[("user_sgpr_private_segment_buffer", "1")],
                "not supported on gfx942",
            ),
            (&[("reserve_xnack_mask", "0")], "does not match target id"),
            (&[("reserve_vcc", "2")], "at most 1"),
            (&[("dx10_clamp", "2")], "at most 1"),
            (&[("float_round_mode_32", "4")], "at most 3"),
            (&[("system_vgpr_workitem_id", "4")], "at most 3"),
            (
                &[("group_segment_fixed_size", "4294967296")],
                "at most 4294967295",
            ),
            (&[("no_such_thing", "1")], "unknown"),
            (&[("next_free_sgpr", "16")], "duplicate"),
        ] {
            assert!(with(extra).unwrap_err().contains(message), "{extra:?}");
        }
        assert!(kd(&[
            ("next_free_vgpr", "8"),
            ("next_free_sgpr", "16"),
            ("accum_offset", "12")
        ])
        .unwrap_err()
        .contains("exceeds total VGPR"));
        assert!(kd(&[
            ("next_free_vgpr", "8"),
            ("next_free_sgpr", "16"),
            ("accum_offset", "0")
        ])
        .unwrap_err()
        .contains("[4..256]"));
        assert!(kd(&[
            ("next_free_vgpr", "8"),
            ("next_free_sgpr", "16"),
            ("accum_offset", "6")
        ])
        .unwrap_err()
        .contains("[4..256]"));
        assert!(kd(&[
            ("next_free_vgpr", "513"),
            ("next_free_sgpr", "16"),
            ("accum_offset", "256")
        ])
        .unwrap_err()
        .contains("at most 512"));
        assert!(kd(&[
            ("next_free_vgpr", "8"),
            ("next_free_sgpr", "103"),
            ("accum_offset", "8")
        ])
        .unwrap_err()
        .contains("at most 102"));
        assert!(kd(&[("next_free_vgpr", "8"), ("accum_offset", "8")])
            .unwrap_err()
            .contains("next_free_sgpr directive is required"));
        assert!(kd(&[("next_free_vgpr", "8"), ("next_free_sgpr", "8")])
            .unwrap_err()
            .contains("accum_offset directive is required"));
    }
}
