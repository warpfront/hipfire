//! `.amdhsa_kernel` directives → the 64-byte kernel descriptor (AMDGPU
//! code object v6, gfx11/gfx12; LLVM `AMDGPUUsage`, "Kernel Descriptor").
//!
//! Exactly the directive set the builder writes is accepted, each once and
//! each within its field; the code entry offset is filled in by the linker
//! layout. `.amdhsa_next_free_sgpr` and `.amdhsa_reserve_vcc` are checked but
//! encode nothing on GFX10+ (`GRANULATED_WAVEFRONT_SGPR_COUNT` is zero).
use crate::Arch;
use std::collections::BTreeMap;

pub(crate) const KD_SIZE: usize = 64;
/// Byte offset of `kernel_code_entry_byte_offset` (i64).
pub(crate) const ENTRY_OFFSET: usize = 16;

/// Where a directive's value goes: (word byte offset, low bit, width).
fn field(arch: Arch, name: &str) -> Option<(usize, u32, u32)> {
    const RSRC3: usize = 44; const RSRC1: usize = 48; const RSRC2: usize = 52; const PROPS: usize = 56;
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
    code_bytes.div_ceil(128).min(if arch.gfx12() { 0xff } else { 0x3f })
}

/// `((instprefsize(.Lend-sym)<<4)&MASK)>>4`, the builder's spelling.
fn inst_pref_expr(arch: Arch, value: &str, end_label: &str, symbol: &str, code_bytes: u64) -> Result<u64, String> {
    let mask = if arch.gfx12() { 4080 } else { 1008 };
    let inner = value.strip_prefix("((instprefsize(").and_then(|v| v.strip_suffix(&format!(")<<4)&{mask})>>4")))
        .ok_or_else(|| format!(".amdhsa_inst_pref_size: unsupported expression {value}"))?;
    if inner != format!("{end_label}-{symbol}") {
        return Err(format!(".amdhsa_inst_pref_size: {inner} is not the extent of {symbol}"));
    }
    Ok(((inst_pref_size(arch, code_bytes) << 4) & mask) >> 4)
}

/// The descriptor of `symbol` from its `.amdhsa_kernel` block (entry offset zero).
pub(crate) fn encode(arch: Arch, symbol: &str, directives: &[(&str, &str)], end_label: &str, code_bytes: u64) -> Result<[u8; KD_SIZE], String> {
    let mut values: BTreeMap<&str, &str> = BTreeMap::new();
    for &(name, value) in directives {
        let key = name.strip_prefix(".amdhsa_").ok_or_else(|| format!("{symbol}: unexpected directive {name}"))?;
        if values.insert(key, value).is_some() { return Err(format!("{symbol}: duplicate .amdhsa_{key}")) }
    }
    let mut expected: Vec<&str> = ["group_segment_fixed_size", "private_segment_fixed_size", "kernarg_size", "user_sgpr_count",
        "user_sgpr_dispatch_ptr", "user_sgpr_queue_ptr", "user_sgpr_kernarg_segment_ptr", "user_sgpr_dispatch_id",
        "user_sgpr_private_segment_size", "wavefront_size32", "uses_dynamic_stack", "enable_private_segment",
        "system_sgpr_workgroup_id_x", "system_sgpr_workgroup_id_y", "system_sgpr_workgroup_id_z", "system_sgpr_workgroup_info",
        "system_vgpr_workitem_id", "next_free_vgpr", "next_free_sgpr", "reserve_vcc", "float_round_mode_32",
        "float_round_mode_16_64", "float_denorm_mode_32", "float_denorm_mode_16_64", "fp16_overflow",
        "workgroup_processor_mode", "memory_ordered", "forward_progress", "inst_pref_size",
        "exception_fp_ieee_invalid_op", "exception_fp_denorm_src", "exception_fp_ieee_div_zero",
        "exception_fp_ieee_overflow", "exception_fp_ieee_underflow", "exception_fp_ieee_inexact", "exception_int_div_zero"].into();
    if arch.gfx12() { expected.push("round_robin_scheduling") } else { expected.extend(["dx10_clamp", "ieee_mode", "shared_vgpr_count"]) }
    if let Some(extra) = values.keys().find(|k| !expected.contains(k)) { return Err(format!("{symbol}: unsupported .amdhsa_{extra} on {}", arch.name())) }
    if let Some(missing) = expected.iter().find(|k| !values.contains_key(*k)) { return Err(format!("{symbol}: missing .amdhsa_{missing}")) }

    let number = |key: &str| -> Result<u64, String> {
        let v = values[key];
        let parsed = if let Some(hex) = v.strip_prefix("0x") { u64::from_str_radix(hex, 16) } else { v.parse() };
        parsed.map_err(|_| format!("{symbol}: .amdhsa_{key} {v} is not an integer"))
    };
    if number("wavefront_size32")? != 1 { return Err(format!("{symbol}: the native writer emits wave32 kernels only")) }
    let mut kd = [0u8; KD_SIZE];
    for (&key, &value) in &values {
        if CHECKED_ONLY.contains(&key) { continue }
        let (offset, low, width) = field(arch, key).ok_or_else(|| format!("{symbol}: unsupported .amdhsa_{key}"))?;
        let v = if key == "inst_pref_size" { inst_pref_expr(arch, value, end_label, symbol, code_bytes)? } else { number(key)? };
        if width < 64 && v >> width != 0 { return Err(format!("{symbol}: .amdhsa_{key} {v} exceeds its {width}-bit field")) }
        let word_bytes = if offset == 56 { 2 } else { 4 };
        let mut word = [0u8; 8];
        word[..word_bytes].copy_from_slice(&kd[offset..offset + word_bytes]);
        let merged = u64::from_le_bytes(word) | (v << low);
        kd[offset..offset + word_bytes].copy_from_slice(&merged.to_le_bytes()[..word_bytes]);
    }
    // GRANULATED_WORKITEM_VGPR_COUNT: wave32 encoding granule of 8 VGPRs.
    let vgprs = number("next_free_vgpr")?;
    if vgprs > 256 { return Err(format!("{symbol}: .amdhsa_next_free_vgpr {vgprs} exceeds 256")) }
    let blocks = (vgprs.max(1).div_ceil(8) - 1) as u32;
    let rsrc1 = u32::from_le_bytes(kd[48..52].try_into().unwrap()) | blocks;
    kd[48..52].copy_from_slice(&rsrc1.to_le_bytes());
    if number("next_free_sgpr")? > 106 || number("reserve_vcc")? > 1 { return Err(format!("{symbol}: SGPR count out of range")) }
    // `.amdhsa_user_sgpr_count` must cover the enabled user SGPRs.
    let implied = 2 * (number("user_sgpr_dispatch_ptr")? + number("user_sgpr_queue_ptr")? + number("user_sgpr_kernarg_segment_ptr")? + number("user_sgpr_dispatch_id")?)
        + number("user_sgpr_private_segment_size")?;
    if number("user_sgpr_count")? < implied { return Err(format!("{symbol}: .amdhsa_user_sgpr_count is below the {implied} enabled user SGPRs")) }
    Ok(kd)
}
