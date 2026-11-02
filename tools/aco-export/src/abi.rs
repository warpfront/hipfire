// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Decode the RADV/ACO capture (`radv-aco-capture-v1`) into the physical entry-ABI sidecar
//! (`pm-aco-abi-v1`), the decoded shader config / FP state, and the RawLift entry facts.
//!
//! Policy (PM-ACO-PARITY.md F1): nothing is inferred. Every ABI value is either read from the
//! captured driver structures or is a cited Mesa-source fact (the hardware-initialised TTMP /
//! packed-local-id conventions that have no `ac_arg` slot). Any slot, config bit or symbol that is
//! not understood is a hard rejection, never an omission.

use crate::util::*;
use serde_json::{json, Value};

const GFX12: u64 = 16; // enum amd_gfx_level: GFX12 (amd_family.h; GFX11=14, GFX11_5=15)

const SRC_ARGS: &str = "src/amd/vulkan/radv_shader_args.c";

fn bits(v: u64, lo: u32, hi: u32) -> u64 {
    (v >> lo) & ((1u64 << (hi - lo + 1)) - 1)
}

fn bit(v: u64, b: u32) -> bool {
    (v >> b) & 1 == 1
}

fn axis(i: usize) -> &'static str {
    ["x", "y", "z"][i]
}

pub struct Analysis {
    pub abi: Value,
    pub entry: Vec<Value>,
    pub resources: Value,
    pub fp: Value,
    pub config: Value,
    pub relocations: Vec<Value>,
    pub wave: u64,
    pub arch: String,
}

fn reject<T>(what: impl std::fmt::Display) -> Res<T> {
    Err(format!("REJECT: {what}"))
}

/// Decode `COMPUTE_PGM_RSRC1/2/3` for gfx12 (src/amd/registers/gfx12.json field tables), rejecting
/// every set bit that is not understood and execution-relevant.
fn decode_rsrc(cap_cfg: &Value, info: &Value, wave: u64, user_sgprs: u64) -> Res<(Value, Value)> {
    let r1 = ju64(cap_cfg, "rsrc1")?;
    let r2 = ju64(cap_cfg, "rsrc2")?;
    let r3 = ju64(cap_cfg, "rsrc3")?;

    // RSRC1 (gfx12): VGPRS 0-5, SGPRS 6-9, PRIORITY 10-11, FLOAT_MODE 12-19, PRIV 20, WG_RR_EN 21,
    // DEBUG_MODE 22, DISABLE_PERF 23, BULKY 24, CDBG_USER 25, FP16_OVFL 26, WGP_MODE 29,
    // MEM_ORDERED 30, FWD_PROGRESS 31.
    for (name, lo, hi) in [
        ("PRIORITY", 10, 11),
        ("PRIV", 20, 20),
        ("DEBUG_MODE", 22, 22),
        ("DISABLE_PERF", 23, 23),
        ("BULKY", 24, 24),
        ("CDBG_USER", 25, 25),
        ("unnamed27_28", 27, 28),
    ] {
        if bits(r1, lo, hi) != 0 {
            return reject(format!(
                "RSRC1.{name} is set (0x{r1:08x}); not representable in pm-raw-kernel-spec-v1"
            ));
        }
    }
    let float_mode = bits(r1, 12, 19);
    if float_mode != ju64(cap_cfg, "float_mode")? {
        return reject(format!(
            "config.float_mode {} != RSRC1.FLOAT_MODE {float_mode}",
            ju64(cap_cfg, "float_mode")?
        ));
    }
    let vgprs_field = bits(r1, 0, 5);
    let granule = if wave == 32 { 8 } else { 4 };
    let vgprs_allocated = (vgprs_field + 1) * granule;

    // RSRC2 (gfx12): SCRATCH_EN 0, USER_SGPR 1-5, DYNAMIC_VGPR 6, TGID_X/Y/Z_EN 7/8/9, TG_SIZE_EN 10,
    // TIDIG_COMP_CNT 11-12, EXCP_EN_MSB 13-14, LDS_SIZE 15-23, EXCP_EN 24-30, WGP_TAKEOVER 31.
    if bit(r2, 0) {
        return reject("RSRC2.SCRATCH_EN set: scratch is not certifiable for the default route");
    }
    if bit(r2, 6) {
        return reject("RSRC2.DYNAMIC_VGPR set");
    }
    if bit(r2, 31) {
        return reject("RSRC2.WGP_TAKEOVER set");
    }
    if bits(r2, 1, 5) != user_sgprs {
        return reject(format!(
            "RSRC2.USER_SGPR {} != args.num_user_sgprs {user_sgprs}",
            bits(r2, 1, 5)
        ));
    }
    for (i, b) in [7u32, 8, 9].iter().enumerate() {
        let used = jget(info, "cs.uses_block_id")?
            .get(i)
            .and_then(Value::as_bool)
            .ok_or("uses_block_id")?;
        if bit(r2, *b) != used {
            return reject(format!(
                "RSRC2.TGID_{}_EN {} != cs.uses_block_id[{i}] {used}",
                axis(i).to_uppercase(),
                bit(r2, *b)
            ));
        }
    }
    let uses_idx = jbool(info, "cs.uses_local_invocation_idx")?;
    if bit(r2, 10) != uses_idx {
        return reject("RSRC2.TG_SIZE_EN != cs.uses_local_invocation_idx");
    }
    let tid = jget(info, "cs.uses_thread_id")?;
    let tid_cnt = if tid[2].as_bool() == Some(true) {
        2
    } else if tid[1].as_bool() == Some(true) {
        1
    } else {
        0
    };
    if bits(r2, 11, 12) != tid_cnt {
        return reject(format!(
            "RSRC2.TIDIG_COMP_CNT {} != derived {tid_cnt}",
            bits(r2, 11, 12)
        ));
    }
    let exceptions = bits(r2, 24, 30) | (bits(r2, 13, 14) << 7);

    // RSRC3 (gfx12): SHARED_VGPR_CNT 0-3, INST_PREF_SIZE 4-11, GLG_EN 13, IMAGE_OP 31.
    if bits(r3, 0, 3) != 0 {
        return reject("RSRC3.SHARED_VGPR_CNT != 0");
    }
    if bit(r3, 13) || bit(r3, 31) || bit(r3, 12) || bits(r3, 14, 30) != 0 {
        return reject(format!(
            "RSRC3 has unknown/execution-affecting bits set (0x{r3:08x})"
        ));
    }

    let fp = json!({
        "round32": bits(float_mode, 0, 1), "round16_64": bits(float_mode, 2, 3),
        "denorm32": bits(float_mode, 4, 5), "denorm16_64": bits(float_mode, 6, 7),
        // gfx12 RSRC1 bit 21 is WG_RR_EN (workgroup round robin), NOT DX10_CLAMP; bit 23 is DISABLE_PERF
        // (rejected above if set). gfx12 has no DX10_CLAMP / IEEE_MODE fields (src/amd/registers/gfx12.json).
        "dx10_clamp": false, "ieee_mode": false, "wg_round_robin": bit(r1, 21),
        "fp16_overflow": bit(r1, 26), "wgp_mode": bit(r1, 29), "mem_ordered": bit(r1, 30),
        "forward_progress": bit(r1, 31), "exceptions": exceptions,
    });
    let decoded = json!({
        "rsrc1": format!("0x{r1:08x}"), "rsrc2": format!("0x{r2:08x}"), "rsrc3": format!("0x{r3:08x}"),
        "rsrc1_fields": {"VGPRS": vgprs_field, "SGPRS": bits(r1, 6, 9), "FLOAT_MODE": float_mode, "WG_RR_EN": bit(r1, 21),
                         "FP16_OVFL": bit(r1, 26), "WGP_MODE": bit(r1, 29), "MEM_ORDERED": bit(r1, 30), "FWD_PROGRESS": bit(r1, 31)},
        "rsrc2_fields": {"USER_SGPR": bits(r2, 1, 5), "TGID_EN": [bit(r2, 7), bit(r2, 8), bit(r2, 9)], "TG_SIZE_EN": bit(r2, 10),
                         "TIDIG_COMP_CNT": bits(r2, 11, 12), "LDS_SIZE": bits(r2, 15, 23), "EXCP_EN": bits(r2, 24, 30), "EXCP_EN_MSB": bits(r2, 13, 14)},
        "rsrc3_fields": {"INST_PREF_SIZE": bits(r3, 4, 11)},
        "vgprs_allocated": vgprs_allocated, "vgpr_alloc_granule": granule,
    });
    Ok((fp, decoded))
}

/// Dword indices (ascending) of the set bits in the inline push-constant mask.
fn mask_dwords(mask: u64) -> Vec<u64> {
    (0..64).filter(|b| (mask >> b) & 1 == 1).collect()
}

fn push_field_for_dword<'a>(fields: &'a [Value], dword: u64) -> Option<&'a Value> {
    fields.iter().find(|f| {
        let off = f["offset"].as_u64().unwrap_or(u64::MAX);
        let size = f["size"].as_u64().unwrap_or(0);
        off <= dword * 4 && dword * 4 < off + size
    })
}

pub fn analyze(kernel: &Value, cap: &Value) -> Res<Analysis> {
    let arch = jstr(kernel, "arch")?.to_owned();
    if arch != "gfx1201" {
        return reject(format!(
            "exporter pilot decodes gfx1201 captures only, spec arch is {arch}"
        ));
    }
    if jstr(cap, "schema")? != "radv-aco-capture-v1" || jstr(cap, "stage")? != "compute" {
        return reject("capture schema/stage mismatch");
    }
    let pd = jget(cap, "physical_device")?;
    if ju64(pd, "gfx_level")? != GFX12 || jstr(pd, "family_name")? != "GFX1201" {
        return reject(format!(
            "driver compiled for gfx_level {} family {}, spec wants gfx1201",
            ju64(pd, "gfx_level")?,
            jstr(pd, "family_name")?
        ));
    }
    if ju64(jget(cap, "aco_options")?, "gfx_level")? != GFX12 {
        return reject("aco_options.gfx_level != GFX12");
    }
    let info = jget(cap, "radv_shader_info")?;
    let wave = ju64(info, "wave_size")?;
    if wave != ju64(kernel, "wave")? || wave != ju64(jget(cap, "aco_info")?, "wave_size")? {
        return reject(format!(
            "wave size mismatch: radv {wave}, aco {}, spec {}",
            ju64(jget(cap, "aco_info")?, "wave_size")?,
            ju64(kernel, "wave")?
        ));
    }
    let local: Vec<u64> = jarr(kernel, "local_size")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let block: Vec<u64> = jarr(info, "cs.block_size")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    if local != block {
        return reject(format!("block size {block:?} != spec local_size {local:?}"));
    }
    if jbool(info, "cs.uses_grid_size")? {
        return reject(
            "shader reads the dispatch grid size (AC_UD_CS_GRID_SIZE); HIP has no such SGPR",
        );
    }
    if !jarr(cap, "symbols")?.is_empty() {
        return reject(format!(
            "ACO emitted unresolved symbols {:?}",
            cap["symbols"]
        ));
    }

    let cfg = jget(cap, "config")?;
    let aco_cfg = jget(cap, "aco_callback_config")?;
    for (name, c) in [("config", cfg), ("aco_callback_config", aco_cfg)] {
        for f in [
            "spilled_sgprs",
            "spilled_vgprs",
            "scratch_bytes_per_wave",
            "num_shared_vgprs",
        ] {
            if ju64(c, f)? != 0 {
                return reject(format!(
                    "{name}.{f} != 0 (spills/scratch/shared VGPRs are not certifiable)"
                ));
            }
        }
    }
    let args = jget(cap, "shader_args")?;
    let num_user = ju64(args, "num_user_sgprs")?;
    let (fp, mut decoded) = decode_rsrc(cfg, info, wave, num_user)?;

    let exec_size = ju64(cap, "exec_size")?;
    let code_size = ju64(cap, "code_size")?;
    if exec_size == 0 || exec_size % 4 != 0 || code_size < exec_size {
        return reject(format!("bad exec_size {exec_size} / code_size {code_size}"));
    }
    let ra = jget(cap, "ra_max_used")?;
    let vgprs_allocated = ju64(&decoded, "vgprs_allocated")?;
    let vgprs_actual = ju64(ra, "vgprs")?;
    let sgprs_actual = ju64(ra, "sgprs")?;
    if vgprs_actual == 0 || vgprs_actual > vgprs_allocated {
        return reject(format!(
            "vgprs_actual {vgprs_actual} vs allocated {vgprs_allocated}"
        ));
    }
    // gfx10+ has no per-wave SGPR allocation field in RSRC1 (RADV writes SGPRS only for gfx<10); the
    // driver's final config.num_sgprs is the allocation RADV commits to.
    let sgprs_allocated = ju64(cfg, "num_sgprs")?;
    if sgprs_actual > sgprs_allocated {
        return reject(format!(
            "sgprs_actual {sgprs_actual} > final config.num_sgprs {sgprs_allocated}"
        ));
    }
    let lds_bytes = ju64(cfg, "lds_size")?;
    let resources = json!({
        "lds_bytes": lds_bytes, "scratch_bytes_per_lane": 0,
        "vgprs_actual": vgprs_actual, "vgprs_allocated": vgprs_allocated,
        "sgprs_actual": sgprs_actual, "sgprs_allocated": sgprs_allocated,
    });

    // ---------------------------------------------------------------- argument slots
    let push = jget(kernel, "push")?;
    let push_fields = jarr(push, "fields")?;
    let push_size = ju64(push, "size")?;
    let mask = u64::from_str_radix(jstr(args, "inline_push_const_mask")?, 16).map_err(err)?;
    let mask_bits = mask_dwords(mask);
    let slots = jarr(args, "args")?;

    let mut inline_slot_iter = 0usize; // i-th inline_push_consts slot <-> i-th set bit (radv_nir_apply_pipeline_layout.c:364)
                                       // (phys sgpr index, dword index, field index in push_fields)
    let mut inline_phys: Vec<(u64, u64)> = Vec::new();
    let mut sgpr_entries: Vec<(u64, Value)> = Vec::new();
    let mut vgpr_entries: Vec<Value> = Vec::new();
    let mut entry: Vec<Value> = Vec::new();
    let mut max_sgpr_end = 0u64;
    let mut last_user_end = 0u64;

    for s in slots {
        let slot = ju64(s, "slot")?;
        if jbool(s, "skip")? {
            return reject(format!(
                "arg slot {slot} is `skip` (pixel-shader compaction); unexpected for compute"
            ));
        }
        let field = match s["field"].as_str() {
            Some(f) => f,
            None => {
                return reject(format!(
                    "arg slot {slot} (file {}, offset {}, size {}) has no named ac_arg field",
                    s["file"], s["offset"], s["size"]
                ))
            }
        };
        if ju64(s, "field_alias_count")? != 1 {
            return reject(format!(
                "arg slot {slot} is aliased by {} named fields",
                s["field_alias_count"]
            ));
        }
        let offset = ju64(s, "offset")?;
        let size = ju64(s, "size")?;
        let findex = s["field_index"].as_i64().unwrap_or(-1);
        match jstr(s, "file")? {
            "sgpr" => {
                let is_user = offset < num_user;
                if is_user {
                    if offset != last_user_end {
                        return reject(format!(
                            "user SGPR slots are not contiguous at slot {slot}"
                        ));
                    }
                    last_user_end = offset + size;
                } else if offset != max_sgpr_end {
                    return reject(format!(
                        "system SGPR slots are not contiguous at slot {slot}"
                    ));
                }
                max_sgpr_end = max_sgpr_end.max(offset + size);
                let mut e = json!({
                    "phys": format!("s{offset}"), "sgpr_index": offset, "dwords": size, "slot": slot,
                    "source": if is_user { "inline_user_sgpr" } else { "system_sgpr" },
                    "mesa_field": field, "const_addr": jstr(s, "type")? == "const_addr",
                });
                match field {
                    "ac.inline_push_consts" => {
                        let dword = *mask_bits.get(inline_slot_iter).ok_or("more inline push slots than mask bits")?;
                        if findex != inline_slot_iter as i64 {
                            return reject(format!("inline push slot order mismatch at slot {slot}"));
                        }
                        inline_slot_iter += 1;
                        inline_phys.push((offset, dword));
                        e["role"] = json!("push_u32");
                        e["push_byte_offset"] = json!(dword * 4);
                        // role / hip_arg refined after merging pointer halves below
                    }
                    "ac.ring_offsets" => {
                        if size != 2 {
                            return reject("ring_offsets is not 2 dwords");
                        }
                        e["role"] = json!("ring");
                        e["hip_arg"] = Value::Null;
                    }
                    "ac.workgroup_ids" => {
                        e["role"] = json!(format!("wgid_{}", axis(findex as usize)));
                        e["axis"] = json!(axis(findex as usize));
                        e["hip_arg"] = Value::Null;
                        entry.push(json!({"loc": {"kind": "sgpr", "base": offset, "len": 1}, "value": {"kind": "workgroup_id", "axis": axis(findex as usize)}}));
                    }
                    "ac.tg_size" => {
                        e["role"] = json!("other");
                        e["hip_arg"] = Value::Null;
                    }
                    f if f.starts_with("descriptors") => return reject(format!("descriptor-set pointer SGPR s{offset} (slot {slot}): BDA pilot must not use descriptors")),
                    "ac.push_constants" => return reject(format!("non-inlined push-constant pointer in s{offset} (slot {slot}): not every push dword was inlined")),
                    other => return reject(format!("unsupported SGPR arg `{other}` in s{offset} (slot {slot})")),
                }
                sgpr_entries.push((offset, e));
            }
            "vgpr" => {
                match field {
                    "ac.local_invocation_ids_packed" => {
                        if size != 1 {
                            return reject("packed local id VGPR is not 1 dword");
                        }
                        let mut fields = Vec::new();
                        let mut layout = serde_json::Map::new();
                        for (i, b) in block.iter().enumerate() {
                            if *b > 1 {
                                if *b > 1024 {
                                    return reject("block dimension > 1024");
                                }
                                fields.push(json!({"axis": axis(i), "shift": 10 * i, "bits": 10, "max": b - 1}));
                                layout.insert(
                                    axis(i).to_owned(),
                                    json!({"shift": 10 * i, "bits": 10, "max": b - 1}),
                                );
                            }
                        }
                        let value = if fields.is_empty() {
                            json!({"kind": "exact", "value": 0})
                        } else {
                            json!({"kind": "local_id", "fields": fields})
                        };
                        entry.push(json!({"loc": {"kind": "vgpr", "base": offset, "len": 1}, "value": value}));
                        vgpr_entries.push(json!({
                        "phys": format!("v{offset}"), "vgpr_index": offset, "dwords": 1, "role": "local_id_packed", "source": "hw",
                        "layout": layout, "mesa_field": field,
                        "evidence": [format!("{SRC_ARGS}:588-590 (gfx11+/MI200: local_invocation_ids_packed in VGPR0)"),
                                     "src/amd/common/nir/ac_nir_lower_intrinsics_to_args.c:160-200 (10 bits per component, packed in VGPR0; upper_bound = (block-1) per axis)"]
                    }));
                    }
                    other => {
                        return reject(format!("unsupported VGPR arg `{other}` (slot {slot})"))
                    }
                }
            }
            f => return reject(format!("arg slot {slot}: unknown register file {f}")),
        }
    }
    if max_sgpr_end != ju64(args, "num_sgprs_used")? || last_user_end != num_user {
        return reject("SGPR slots do not exactly cover num_sgprs_used / num_user_sgprs");
    }
    if inline_slot_iter != mask_bits.len() {
        return reject("inline push mask bit count != number of inline push slots");
    }

    // Merge inline push dwords according to the kernel's push-block fields.
    let mut covered = std::collections::BTreeSet::new();
    for f in push_fields {
        let off = ju64(f, "offset")?;
        let size = ju64(f, "size")?;
        let kind = jstr(f, "kind")?;
        if off + size > push_size || off % 4 != 0 || size % 4 != 0 {
            return reject(format!(
                "push field {} is outside/misaligned in the {push_size}-byte block",
                f["name"]
            ));
        }
        let dwords: Vec<u64> = (off / 4..(off + size) / 4).collect();
        let inlined: Vec<Option<u64>> = dwords
            .iter()
            .map(|d| inline_phys.iter().find(|(_, dd)| dd == d).map(|(p, _)| *p))
            .collect();
        let present = inlined.iter().filter(|x| x.is_some()).count();
        if present == 0 {
            continue; // field never loaded by the shader
        }
        if present != dwords.len() {
            return reject(format!("push field `{}` is only partially inlined (a truncated 64-bit pointer / split value)", f["name"]));
        }
        let first = inlined[0].unwrap();
        for (i, p) in inlined.iter().enumerate() {
            if p.unwrap() != first + i as u64 {
                return reject(format!(
                    "push field `{}` dwords are not in consecutive SGPRs",
                    f["name"]
                ));
            }
            covered.insert(dwords[i]);
        }
        let role = match kind {
            "u64_ptr" => "bda_ptr",
            "i32" | "u32" | "f32" => "push_u32",
            k => return reject(format!("unknown push field kind {k}")),
        };
        // remove the per-dword entries and replace with one merged entry
        sgpr_entries.retain(|(p, _)| !(first..first + dwords.len() as u64).contains(p));
        sgpr_entries.push((first, json!({
            "phys": format!("s{first}"), "sgpr_index": first, "dwords": dwords.len(), "source": "inline_user_sgpr",
            "mesa_field": "ac.inline_push_consts", "role": role, "hip_arg": f["hip_arg"], "push_field": f["name"],
            "push_byte_offset": off, "push_field_kind": kind, "const_addr": false, "access": f["access"],
            "dwords_covered": dwords.iter().map(|d| json!({"dword": d, "byte_offset": d * 4})).collect::<Vec<_>>(),
        })));
    }
    for (_, d) in &inline_phys {
        if !covered.contains(d) {
            return reject(format!(
                "inlined push dword {d} (byte {}) belongs to no declared push field",
                d * 4
            ));
        }
        let _ = push_field_for_dword(push_fields, *d);
    }
    sgpr_entries.sort_by_key(|(p, _)| *p);
    let mut user_sgprs: Vec<Value> = Vec::new();
    for (i, (_, mut e)) in sgpr_entries.into_iter().enumerate() {
        e["index"] = json!(i);
        user_sgprs.push(e);
    }

    // ---------------------------------------------------------------- hardware-initialised registers
    let mut hw: Vec<Value> = Vec::new();
    let implicit: Vec<bool> = jarr(args, "gfx12_implicit_workgroup_ids")?
        .iter()
        .map(|b| b.as_bool().unwrap_or(false))
        .collect();
    if implicit.len() != 3 {
        return reject("gfx12_implicit_workgroup_ids malformed");
    }
    let src_wg = "src/amd/compiler/instruction_selection/aco_select_nir_intrinsics.cpp:4056-4073 (nir_intrinsic_load_workgroup_id, GFX12: x=ttmp9 (PhysReg 108+9), y/z=ttmp7)";
    if implicit[0] {
        entry.push(json!({"loc": {"kind": "ttmp", "base": 9, "len": 1}, "value": {"kind": "workgroup_id", "axis": "x"}}));
        hw.push(json!({"phys": "ttmp9", "ttmp_index": 9, "dwords": 1, "role": "ttmp_wgid_x", "axis": "x", "source": "hw", "hip_arg": null, "evidence": [src_wg, format!("{SRC_ARGS}:572-578 (gfx12 workgroup_ids[i].used=true, no SGPR slot)")]}));
    }
    if implicit[2] {
        return reject("gfx12 workgroup z used: ttmp7 packs y[15:0]/z[31:16]; not representable in pm-raw-kernel-spec-v1");
    }
    if implicit[1] {
        entry.push(json!({"loc": {"kind": "ttmp", "base": 7, "len": 1}, "value": {"kind": "workgroup_id", "axis": "y"}}));
        hw.push(json!({"phys": "ttmp7", "ttmp_index": 7, "dwords": 1, "role": "wgid_y", "axis": "y", "source": "hw", "hip_arg": null, "evidence": [src_wg]}));
    }
    entry.push(json!({"loc": {"kind": "special", "name": "exec"}, "value": {"kind": "defined"}}));
    hw.push(json!({"phys": "exec", "role": "exec", "source": "hw", "evidence": ["wave launch initialises EXEC to the launched lanes (full wave for a block that is a multiple of the wave size)"]}));
    // User + system SGPR facts: defined (non-workgroup-id ones).
    for e in &user_sgprs {
        let role = e["role"].as_str().unwrap_or("");
        if role.starts_with("wgid_") {
            continue; // already emitted as workgroup_id facts above
        }
        entry.push(json!({"loc": {"kind": "sgpr", "base": e["sgpr_index"], "len": e["dwords"]}, "value": {"kind": "defined"}}));
    }

    // ---------------------------------------------------------------- config / dispatch facts
    let regs = jget(cap, "regs")?;
    let config = {
        decoded["wave_size"] = json!(wave);
        decoded["exec_size"] = json!(exec_size);
        decoded["code_size"] = json!(code_size);
        decoded["lds_bytes"] = json!(lds_bytes);
        decoded["scratch_bytes_per_wave"] = json!(0);
        decoded["float_mode"] = cfg["float_mode"].clone();
        decoded["fp"] = fp.clone();
        decoded["vgprs_actual"] = json!(vgprs_actual);
        decoded["sgprs_actual"] = json!(sgprs_actual);
        decoded["sgprs_allocated"] = json!(sgprs_allocated);
        decoded["vgprs_aco_alloc"] = aco_cfg["num_vgprs"].clone();
        decoded["sgprs_aco_alloc"] = aco_cfg["num_sgprs"].clone();
        decoded["vgprs_final_config"] = cfg["num_vgprs"].clone();
        decoded["actual_source"] = ra["source"].clone();
        decoded["sgprs_allocated_source"] = json!("radv_postprocess_binary_config: MAX2(aco config.num_sgprs, args.num_sgprs_used + 2 + 3); gfx10+ has no RSRC1 SGPR allocation field");
        decoded["compute_resource_limits"] = regs["compute_resource_limits"].clone();
        decoded["compute_num_thread"] = json!([
            regs["compute_num_thread_x"],
            regs["compute_num_thread_y"],
            regs["compute_num_thread_z"]
        ]);
        decoded
    };

    let inline_dwords: Vec<Value> = inline_phys
        .iter()
        .map(|(p, d)| json!({"dword": d, "byte_offset": d * 4, "phys": format!("s{p}")}))
        .collect();
    let abi = json!({
        "schema": "pm-aco-abi-v1",
        "symbol": jstr(kernel, "symbol")?,
        "arch": arch, "wave": wave,
        "capture_schema": "radv-aco-capture-v1",
        "user_sgprs": user_sgprs,
        "vgprs": vgpr_entries,
        "hw": hw,
        "workgroup": {
            "local_size": block, "uses_block_id": info["cs"]["uses_block_id"], "uses_thread_id": info["cs"]["uses_thread_id"],
            "uses_local_invocation_idx": info["cs"]["uses_local_invocation_idx"], "uses_grid_size": false,
        },
        "push_block": {
            "size": push_size, "fields": push_fields,
            "inline_mask_hex": jstr(args, "inline_push_const_mask")?, "inline_dwords": inline_dwords,
            "evidence": [
                "src/amd/vulkan/radv_shader_args.c:allocate_inline_push_consts + declare_global_input_sgprs (inline_push_consts[i] slots follow descriptors/push_constants/dynamic_descriptors in user SGPRs)",
                "src/amd/vulkan/nir/radv_nir_apply_pipeline_layout.c:347-365 (push dword d -> slot popcount(mask & (bit(d)-1)))",
                "src/amd/vulkan/radv_cmd_buffer.c:6395-6410 (emitted in ascending mask-bit order)",
            ],
        },
        "pointers": {"width_bits": 64, "truncation": false, "address32_hi": jget(cap, "aco_options.address32_hi")?, "rule":
            "u64_ptr push fields occupy two consecutive inline user SGPRs holding the full 64-bit address; any partially inlined pointer is rejected; const_addr args (32-bit) are rejected unless named ring_offsets (2 dwords)"},
        "config": config,
        "dispatch": {
            "user_data_0_reg": info["user_data_0"], "compute_pgm_rsrc_regs": [regs["pgm_rsrc1_reg"], regs["pgm_rsrc2_reg"], regs["pgm_rsrc3_reg"]],
            "device_dispatch_initiator": regs["device_dispatch_initiator"],
            "per_dispatch_initiator_rules": [
                "CS_W32_EN set iff wave_size == 32 (src/amd/vulkan/radv_cmd_buffer.c:13538-13540)",
                "ORDER_MODE cleared iff dispatch_info.ordered (radv_cmd_buffer.c:13542-13543)",
                "USE_THREAD_DIMENSIONS only for indirect unaligned dispatch (radv_cmd_buffer.c:13559-13562); FORCE_START_AT_000 for indirect",
            ],
            "radv_emit_compute": "src/amd/vulkan/radv_cmd_buffer.c:2836-2849 emits shader.config.rsrc3 and compute_resource_limits verbatim",
        },
        "unhandled_by_design": ["hardware ttmp8 (subgroup id) / other ttmp reads are NOT declared; the lift rejects an undefined read"],
    });
    let relocations: Vec<Value> = Vec::new(); // symbols were rejected above if any
    Ok(Analysis {
        abi,
        entry,
        resources,
        fp,
        config,
        relocations,
        wave,
        arch,
    })
}
