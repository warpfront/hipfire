// SPDX-License-Identifier: Apache-2.0
//! Independent, source-text replay of async memory waits and register locks.
//! Consumes assembly text, not the builder's pending-event ledger.
//!
//! gfx12 counters are LOADcnt, STOREcnt, DScnt and KMcnt. On gfx11 (RDNA3 ISA
//! §16.5) `s_waitcnt` carries VMcnt (loads), LGKMcnt (DS *and* SMEM/message
//! returns, one counter) and EXPcnt; `s_waitcnt_vscnt` waits on stores. A
//! nonzero LGKMcnt retires only the oldest DS operations when no SMEM shares
//! the counter (SMEM returns out of order), as in ROCm LLVM `SIInsertWaitcnts`.
//! gfx12 LOADcnt retires `buffer_load*` and `global_load*` in issue order
//! together: ROCm 10.0 `SIInsertWaitcnts.cpp` maps both (MUBUF non-image via
//! `VmemReadMapping[VMEM_NOSAMPLER]`, FLAT-global directly) to the single
//! `VMEM_ACCESS` LOAD_CNT event, so `counterOutOfOrder(LOAD_CNT)` is false.
//! Other VMEM forms (flat, scratch, image) and every gfx11 mix stay separate.
use crate::Arch;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
struct Pending<'a> {
    defs: BTreeSet<u16>,
    locks: BTreeSet<u16>,
    kind: Kind,
    opcode: &'a str,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Vmem,
    Store,
    Ds,
    Km,
}

pub(crate) fn registers(operands: &str) -> BTreeSet<u16> {
    let bytes = operands.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if !matches!(bytes[i], b'v' | b's')
            || (i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
        {
            i += 1;
            continue;
        }
        let class = if bytes[i] == b's' { 256 } else { 0 };
        i += 1;
        let bracket = bytes.get(i) == Some(&b'[');
        if bracket {
            i += 1;
        }
        let begin = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        let Some(first) = operands.get(begin..i).and_then(|s| s.parse::<u16>().ok()) else {
            continue;
        };
        let last = if bracket && bytes.get(i) == Some(&b':') {
            i += 1;
            let begin = i;
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
            operands
                .get(begin..i)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(first)
        } else {
            first
        };
        if last >= first && last <= 255 {
            found.extend((first + class)..=(last + class));
        }
    }
    found
}
fn wait_count(rest: &str) -> Option<usize> {
    let token = rest.split_whitespace().next()?.split('_').next()?;
    if let Some(hex) = token.strip_prefix("0x") {
        usize::from_str_radix(hex, 16).ok()
    } else {
        token.parse().ok()
    }
}

/// In-order class of a pending VMEM load (see the module comment).
fn vmem_class(opcode: &str, gfx12: bool) -> &str {
    if gfx12 && (opcode.starts_with("buffer_load") || opcode.starts_with("global_load")) {
        return "vmem_read";
    }
    opcode.split('_').next().unwrap_or(opcode)
}

fn retire(pending: &mut Vec<Pending<'_>>, kind: Kind, count: usize, gfx12: bool) {
    let mut outstanding = pending.iter().filter(|item| item.kind == kind).count();
    // SMEM, stores, and mixed VMEM load classes have no in-order guarantee.
    if count != 0
        && (matches!(kind, Kind::Km | Kind::Store)
            || kind == Kind::Vmem && {
                let mut types = pending
                    .iter()
                    .filter(|item| item.kind == Kind::Vmem)
                    .map(|item| vmem_class(item.opcode, gfx12));
                let first = types.next();
                types.any(|family| Some(family) != first)
            })
    {
        return;
    }
    pending.retain(|item| {
        if item.kind == kind && outstanding > count {
            outstanding -= 1;
            false
        } else {
            true
        }
    });
}

/// Reject RAW, WAW and store-source WAR hazards from machine-readable text.
/// Combined waits decode both counters independently.
pub fn replay_waits(assembly: &str, arch: Arch) -> Result<(), String> {
    replay(assembly, arch, false).map(|_| ())
}

/// Every hazard the replay finds, as the offending instruction text, in
/// order. Used to show a rewrite adds no hazard to foreign (hipcc) code whose
/// hardware-interlocked DS-source reuse the strict replay already flags.
pub fn replay_hazards(assembly: &str, arch: Arch) -> Result<Vec<String>, String> {
    replay(assembly, arch, true)
}

/// gfx11 LGKMcnt: DS and SMEM share one counter. With any SMEM pending only
/// a full drain retires anything; otherwise the oldest DS operations retire.
fn retire_lgkm(pending: &mut Vec<Pending<'_>>, count: usize) {
    if count == 0 {
        pending.retain(|item| !matches!(item.kind, Kind::Ds | Kind::Km));
    } else if !pending.iter().any(|item| item.kind == Kind::Km) {
        retire(pending, Kind::Ds, count, false);
    }
}

/// Fields of a gfx11 `s_waitcnt`: `vmcnt(n) expcnt(n) lgkmcnt(n)` in any
/// order (absent = no wait), or a raw immediate VM[15:10] LGKM[9:4] EXP[2:0].
fn gfx11_waitcnt(operands: &str) -> Option<(Option<usize>, Option<usize>)> {
    let operands = operands.trim();
    if operands.starts_with("0x") || operands.bytes().next().is_some_and(|b| b.is_ascii_digit()) {
        let imm = wait_count(operands)?;
        let (vm, lgkm) = ((imm >> 10) & 0x3f, (imm >> 4) & 0x3f);
        return Some(((vm != 0x3f).then_some(vm), (lgkm != 0x3f).then_some(lgkm)));
    }
    let (mut vm, mut lgkm) = (None, None);
    for field in operands
        .split(|c: char| c.is_whitespace() || c == '&' || c == ',')
        .filter(|f| !f.is_empty())
    {
        let (name, value) = field.split_once('(')?;
        let value: usize = value.strip_suffix(')')?.parse().ok()?;
        match name {
            "vmcnt" => vm = Some(value),
            "lgkmcnt" => lgkm = Some(value),
            "expcnt" => {}
            _ => return None,
        }
    }
    Some((vm, lgkm))
}

fn replay(assembly: &str, arch: Arch, all: bool) -> Result<Vec<String>, String> {
    let gfx11 = !arch.gfx12();
    let mut hazards = Vec::new();
    let mut pending = Vec::<Pending>::new();
    for (line_no, source) in assembly.lines().enumerate() {
        let line = source.split(';').next().unwrap_or("").trim();
        if line.starts_with("s_endpgm") {
            pending.clear();
            continue;
        }
        let Some((name, operands)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        if name.starts_with('.') || name.ends_with(':') {
            continue;
        }
        if gfx11 {
            let count = || {
                operands
                    .rsplit(',')
                    .next()
                    .and_then(wait_count)
                    .ok_or_else(|| format!("line {}: invalid wait", line_no + 1))
            };
            match name {
                "s_waitcnt" => {
                    let (vm, lgkm) = gfx11_waitcnt(operands)
                        .ok_or_else(|| format!("line {}: invalid s_waitcnt", line_no + 1))?;
                    if let Some(n) = vm {
                        retire(&mut pending, Kind::Vmem, n, false);
                    }
                    if let Some(n) = lgkm {
                        retire_lgkm(&mut pending, n);
                    }
                    continue;
                }
                "s_waitcnt_vmcnt" => {
                    retire(&mut pending, Kind::Vmem, count()?, false);
                    continue;
                }
                "s_waitcnt_vscnt" => {
                    retire(&mut pending, Kind::Store, count()?, false);
                    continue;
                }
                "s_waitcnt_lgkmcnt" => {
                    retire_lgkm(&mut pending, count()?);
                    continue;
                }
                "s_waitcnt_depctr" if operands.contains("depctr_vm_vsrc(0)") => {
                    for item in pending.iter_mut().filter(|item| item.kind == Kind::Store) {
                        item.locks.clear();
                    }
                    continue;
                }
                _ if name.starts_with("s_wait_") => {
                    return Err(format!("line {}: {name} is not a gfx11 wait", line_no + 1))
                }
                _ => {}
            }
        } else if name == "s_waitcnt" || name.starts_with("s_waitcnt_") {
            return Err(format!("line {}: {name} is not a gfx12 wait", line_no + 1));
        }
        let wait_kind = match name {
            "s_wait_loadcnt" => Some(Kind::Vmem),
            "s_wait_storecnt" => Some(Kind::Store),
            "s_wait_dscnt" => Some(Kind::Ds),
            "s_wait_kmcnt" => Some(Kind::Km),
            _ => None,
        };
        if let Some(kind) = wait_kind {
            let count = wait_count(operands)
                .ok_or_else(|| format!("line {}: invalid wait", line_no + 1))?;
            retire(&mut pending, kind, count, true);
            continue;
        }
        if name == "s_wait_loadcnt_dscnt" {
            let encoded = wait_count(operands)
                .ok_or_else(|| format!("line {}: invalid combined wait", line_no + 1))?;
            if encoded > 0x3f3f || encoded & 0xc0c0 != 0 {
                return Err(format!("line {}: out-of-range combined wait", line_no + 1));
            }
            retire(&mut pending, Kind::Vmem, (encoded >> 8) & 0x3f, true);
            retire(&mut pending, Kind::Ds, encoded & 0x3f, true);
            continue;
        }
        // VM_VSRC counts VMEM instructions that have not yet read their
        // source registers; at 0 every pending store has consumed its sources.
        if name == "s_wait_alu" && operands.contains("depctr_vm_vsrc(0)") {
            for item in pending.iter_mut().filter(|item| item.kind == Kind::Store) {
                item.locks.clear();
            }
            continue;
        }
        let kind = if name.starts_with("buffer_load") || name.starts_with("global_load") {
            Some((Kind::Vmem, true))
        } else if name.starts_with("buffer_store") || name.starts_with("global_store") {
            Some((Kind::Store, false))
        } else if name.starts_with("ds_load")
            || name == "ds_swizzle_b32"
            || name == "ds_bpermute_b32"
        {
            // ds_swizzle_b32 / ds_bpermute_b32 return their lane exchange through the DS queue like a load.
            Some((Kind::Ds, true))
        } else if name.starts_with("ds_store") {
            Some((Kind::Ds, false))
        } else if name.starts_with("s_load")
            || name.starts_with("s_buffer_load")
            || name.starts_with("s_sendmsg_rtn")
        {
            Some((Kind::Km, true))
        } else if name.starts_with("s_store") || name.starts_with("s_buffer_store") {
            Some((Kind::Km, false))
        } else {
            None
        };
        let all_regs = registers(operands);
        let first = operands.split(',').next().unwrap_or("");
        let defs = if kind.is_some_and(|(_, load)| load)
            || name.starts_with("v_")
            || name.starts_with("s_mov")
            || name.starts_with("s_add")
            || name.starts_with("s_sub")
        {
            let mut result = registers(first);
            if name.starts_with("v_dual_") {
                if let Some((_, second)) = operands.split_once("::") {
                    // The second half starts with its mnemonic, then its destination.
                    if let Some((_, args)) = second.trim().split_once(char::is_whitespace) {
                        result.extend(registers(args.split(',').next().unwrap_or("")));
                    }
                }
            }
            result
        } else {
            BTreeSet::new()
        };
        if pending
            .iter()
            .any(|event| !event.defs.is_disjoint(&all_regs) || !event.locks.is_disjoint(&defs))
        {
            if !all {
                return Err(format!(
                    "line {}: {name} touches unfinished memory operands",
                    line_no + 1
                ));
            }
            hazards.push(line.to_string());
        }
        if let Some((kind, load)) = kind {
            pending.push(Pending {
                defs: if load { defs } else { BTreeSet::new() },
                locks: if load { BTreeSet::new() } else { all_regs },
                kind,
                opcode: name,
            });
        }
    }
    Ok(hazards)
}

#[cfg(test)]
mod tests {
    use super::replay_waits as gfx12_replay;
    fn replay_waits(text: &str) -> Result<(), String> {
        gfx12_replay(text, crate::Arch::Gfx1201)
    }
    #[test]
    fn load_wait_uses_only_retired_destinations() {
        let legal = "buffer_load_b32 v0, v8, s[4:7], s9 offen\n\
            buffer_load_b32 v1, v8, s[4:7], s9 offen\n\
            s_wait_loadcnt 0x1\nv_add_f32 v2, v0, v3\n";
        assert!(replay_waits(legal).is_ok());
        assert!(replay_waits(&legal.replace("0x1", "0x2")).is_err());
    }
    #[test]
    fn combined_wait_keeps_young_vmem_pending() {
        let stream = "buffer_load_b32 v0, v8, s[4:7], s9 offen\n\
            ds_load_b32 v1, v9\n\
            s_wait_loadcnt_dscnt 0x100\n\
            v_add_f32 v2, v1, v3\n";
        assert!(replay_waits(stream).is_ok());
        assert!(replay_waits(&(stream.to_owned() + "v_add_f32 v2, v0, v3\n")).is_err());
    }
    #[test]
    fn asymmetric_combined_wait_decodes_load_then_ds() {
        let mut source = String::new();
        for i in 0..8 {
            source.push_str(&format!("buffer_load_b32 v{i}, v20, s[4:7], s9 offen\n"));
        }
        for i in 0..4 {
            source.push_str(&format!("ds_load_b32 v{}, v21\n", i + 8));
        }
        source.push_str("s_wait_loadcnt_dscnt 0x703\nv_add_f32 v22, v0, v8\n");
        assert!(replay_waits(&source).is_ok());
        assert!(replay_waits(&source.replace("0x703", "0x307")).is_err());
        assert!(replay_waits(&source.replace("0x703", "0x707")).is_err());
    }
    #[test]
    fn store_lock_requires_storecnt_before_redefinition() {
        let source = "buffer_store_b32 v0, v1, s[4:7], s8 offen\n\
            v_mov_b32 v0, 0\n";
        assert!(replay_waits(source).is_err());
        assert!(replay_waits(&source.replace("v_mov_b32", "s_wait_storecnt 0\nv_mov_b32")).is_ok());
        assert!(replay_waits(
            "buffer_store_b32 v0, v1, s[4:7], s8 offen\n\
            v_add_f32 v2, v0, v3\n"
        )
        .is_ok());
    }
    #[test]
    fn smem_load_requires_kmcnt_before_scalar_use() {
        let source = "s_load_b32 s4, s[0:1], 0\ns_add_u32 s5, s4, s6\n";
        assert!(replay_waits(source).is_err());
        assert!(replay_waits(&source.replace("s_add_u32", "s_wait_kmcnt 0\ns_add_u32")).is_ok());
    }
    #[test]
    fn vm_vsrc_releases_store_sources_and_realtime_needs_kmcnt() {
        let store = "global_store_addtid_b32 v9, s[4:5] offset:-8\nv_writelane_b32 v9, s6, 0\n";
        assert!(replay_waits(store).is_err());
        assert!(replay_waits(
            &store.replace("v_writelane", "s_wait_alu depctr_vm_vsrc(0)\nv_writelane")
        )
        .is_ok());
        let clock =
            "s_sendmsg_rtn_b64 s[6:7], sendmsg(MSG_RTN_GET_REALTIME)\nv_writelane_b32 v9, s6, 0\n";
        assert!(replay_waits(clock).is_err());
        assert!(
            replay_waits(&clock.replace("v_writelane", "s_wait_kmcnt 0x0\nv_writelane")).is_ok()
        );
    }
    /// gfx12 LOADcnt: buffer and global loads share one in-order VMEM_ACCESS
    /// event, so a partial wait retires exactly the oldest of the mixed window.
    /// Flat/scratch loads, and gfx11 VMcnt, keep requiring a full drain.
    #[test]
    fn gfx12_mixed_buffer_global_loads_retire_in_issue_order() {
        let window = "buffer_load_b64 v[0:1], v8, s[4:7], s9 offen\n\
            global_load_b32 v2, v8, s[4:5]\n\
            buffer_load_b32 v5, v8, s[4:7], s9 offen offset:8\n";
        // loadcnt 2 retires only the oldest (buffer v[0:1]).
        assert!(replay_waits(&format!("{window}s_wait_loadcnt 2\nv_add_f32 v3, v0, v4\n")).is_ok());
        assert!(
            replay_waits(&format!("{window}s_wait_loadcnt 2\nv_add_f32 v3, v2, v4\n")).is_err(),
            "global still pending"
        );
        assert!(replay_waits(&format!("{window}s_wait_loadcnt 1\nv_add_f32 v3, v2, v4\n")).is_ok());
        // Insufficient wait for the youngest (buffer v5).
        assert!(
            replay_waits(&format!("{window}s_wait_loadcnt 1\nv_add_f32 v3, v5, v4\n")).is_err()
        );
        assert!(replay_waits(&format!("{window}s_wait_loadcnt 0\nv_add_f32 v3, v5, v4\n")).is_ok());
        // gfx11 VMcnt keeps the per-family rule.
        let gfx11 = "buffer_load_b64 v[0:1], v8, s[4:7], s9 offen\nglobal_load_b32 v2, v8, s[4:5]\ns_waitcnt vmcnt(1)\nv_add_f32 v3, v0, v4\n";
        assert!(super::replay_waits(gfx11, crate::Arch::Gfx1100).is_err());
    }
    #[test]
    fn gfx11_waitcnt_fields_and_shared_lgkm() {
        let gfx11 = |text: &str| super::replay_waits(text, crate::Arch::Gfx1100);
        let loads = "global_load_b32 v0, v8, s[4:5]\nglobal_load_b32 v1, v8, s[4:5] offset:4\n\
            ds_load_b32 v2, v9\nds_load_b32 v3, v9 offset:4\n";
        assert!(gfx11(&format!(
            "{loads}s_waitcnt vmcnt(1) lgkmcnt(1)\nv_add_f32 v4, v0, v2\n"
        ))
        .is_ok());
        assert!(gfx11(&format!(
            "{loads}s_waitcnt vmcnt(1) lgkmcnt(1)\nv_add_f32 v4, v1, v2\n"
        ))
        .is_err());
        assert!(gfx11(&format!(
            "{loads}s_waitcnt vmcnt(1)\nv_add_f32 v4, v0, v2\n"
        ))
        .is_err());
        // SMEM shares LGKMcnt and returns out of order: only lgkmcnt(0) retires.
        let mixed = "s_load_b32 s6, s[0:1], 0x0\nds_load_b32 v2, v9\nds_load_b32 v3, v9 offset:4\n";
        assert!(gfx11(&format!(
            "{mixed}s_waitcnt lgkmcnt(1)\nv_add_f32 v4, v2, v2\n"
        ))
        .is_err());
        assert!(gfx11(&format!(
            "{mixed}s_waitcnt lgkmcnt(0)\nv_add_f32 v4, v2, v2\n"
        ))
        .is_ok());
        // Raw immediate: VM[15:10] = 0, LGKM[9:4] = 63 (no wait), EXP[2:0] = 7.
        assert!(gfx11(&format!("{loads}s_waitcnt 0x3f7\nv_add_f32 v4, v1, v1\n")).is_ok());
        assert!(gfx11(&format!("{loads}s_waitcnt 0x3f7\nv_add_f32 v4, v2, v2\n")).is_err());
        // Stores: only vscnt(0) or vm_vsrc(0) releases their sources.
        let store = "global_store_b32 v1, v0, s[4:5]\nv_mov_b32_e32 v0, 0\n";
        assert!(gfx11(store).is_err());
        assert!(gfx11(&store.replace("v_mov", "s_waitcnt_vscnt null, 0x0\nv_mov")).is_ok());
        assert!(
            gfx11(&store.replace("v_mov", "s_waitcnt_depctr depctr_vm_vsrc(0)\nv_mov")).is_ok()
        );
        // Architecture-foreign spellings are rejected, not ignored.
        assert!(gfx11(&format!(
            "{loads}s_wait_loadcnt 0x0\nv_add_f32 v4, v1, v1\n"
        ))
        .is_err());
        assert!(super::replay_waits("s_waitcnt vmcnt(0)\n", crate::Arch::Gfx1201).is_err());
    }
    /// A ds_swizzle_b32 lane exchange returns through the DS queue: its
    /// destination is unfinished until an lgkmcnt wait retires it, in order
    /// with the DS loads around it.
    #[test]
    fn gfx11_swizzle_results_wait_like_ds_loads() {
        let gfx11 = |text: &str| super::replay_waits(text, crate::Arch::Gfx1100);
        let stream = "ds_load_b32 v2, v9\nds_swizzle_b32 v3, v4 offset:swizzle(BROADCAST,16,0)\nds_load_b32 v5, v9 offset:4\n";
        assert!(gfx11(&format!("{stream}v_add_f32 v6, v3, v3\n")).is_err());
        assert!(gfx11(&format!(
            "{stream}s_waitcnt lgkmcnt(2)\nv_add_f32 v6, v3, v3\n"
        ))
        .is_err());
        assert!(gfx11(&format!(
            "{stream}s_waitcnt lgkmcnt(1)\nv_add_f32 v6, v3, v3\n"
        ))
        .is_ok());
        // lgkmcnt(1) leaves the youngest load, behind the swizzle, pending.
        assert!(gfx11(&format!(
            "{stream}s_waitcnt lgkmcnt(1)\nv_add_f32 v6, v5, v5\n"
        ))
        .is_err());
    }
    /// A ds_bpermute_b32 lane exchange returns through the gfx12 DS queue in
    /// order with the DS loads around it, exactly like a swizzle.
    #[test]
    fn gfx12_bpermute_results_wait_like_ds_loads() {
        let gfx12 = |text: &str| super::replay_waits(text, crate::Arch::Gfx1201);
        let stream =
            "ds_load_b32 v2, v9\nds_bpermute_b32 v3, v4, v7\nds_load_b32 v5, v9 offset:4\n";
        assert!(gfx12(&format!("{stream}v_add_f32_e32 v6, v3, v3\n")).is_err());
        assert!(gfx12(&format!(
            "{stream}s_wait_dscnt 0x2\nv_add_f32_e32 v6, v3, v3\n"
        ))
        .is_err());
        assert!(gfx12(&format!(
            "{stream}s_wait_dscnt 0x1\nv_add_f32_e32 v6, v3, v3\n"
        ))
        .is_ok());
        assert!(gfx12(&format!(
            "{stream}s_wait_dscnt 0x1\nv_add_f32_e32 v6, v5, v5\n"
        ))
        .is_err());
    }
}
