// SPDX-License-Identifier: Apache-2.0
//! Independent, source-text replay of async memory waits and register locks.
//! Consumes assembly text, not the builder's pending-event ledger.
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
struct Pending<'a> { defs: BTreeSet<u16>, locks: BTreeSet<u16>, kind: Kind, opcode: &'a str }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind { Vmem, Store, Ds, Km }

fn registers(operands: &str) -> BTreeSet<u16> {
    let bytes = operands.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if !matches!(bytes[i], b'v' | b's')
            || (i > 0 && (bytes[i-1].is_ascii_alphanumeric() || bytes[i-1] == b'_')) {
            i += 1; continue;
        }
        let class = if bytes[i] == b's' { 256 } else { 0 };
        i += 1;
        let bracket = bytes.get(i) == Some(&b'[');
        if bracket { i += 1; }
        let begin = i;
        while bytes.get(i).is_some_and(u8::is_ascii_digit) { i += 1; }
        let Some(first) = operands.get(begin..i).and_then(|s| s.parse::<u16>().ok()) else { continue };
        let last = if bracket && bytes.get(i) == Some(&b':') {
            i += 1;
            let begin = i;
            while bytes.get(i).is_some_and(u8::is_ascii_digit) { i += 1; }
            operands.get(begin..i).and_then(|s| s.parse::<u16>().ok()).unwrap_or(first)
        } else { first };
        if last >= first && last <= 255 {
            found.extend((first + class)..=(last + class));
        }
    }
    found
}
fn wait_count(rest: &str) -> Option<usize> {
    let token = rest.split_whitespace().next()?.split('_').next()?;
    if let Some(hex) = token.strip_prefix("0x") { usize::from_str_radix(hex, 16).ok() }
    else { token.parse().ok() }
}

fn retire(pending: &mut Vec<Pending<'_>>, kind: Kind, count: usize) {
    let mut outstanding = pending.iter().filter(|item| item.kind == kind).count();
    // SMEM, stores, and mixed VMEM load types have no in-order guarantee.
    if count != 0 && (matches!(kind, Kind::Km | Kind::Store)
        || kind == Kind::Vmem && {
            let mut types = pending.iter().filter(|item| item.kind == Kind::Vmem)
                .map(|item| item.opcode.split('_').next().unwrap_or(item.opcode));
            let first = types.next();
            types.any(|family| Some(family) != first)
        }) { return; }
    pending.retain(|item| {
        if item.kind == kind && outstanding > count {
            outstanding -= 1;
            false
        } else { true }
    });
}

/// Reject RAW, WAW and store-source WAR hazards from machine-readable text.
/// Combined waits decode both counters independently.
pub fn replay_waits(assembly: &str) -> Result<(), String> {
    let mut pending = Vec::<Pending>::new();
    for (line_no, source) in assembly.lines().enumerate() {
        let line = source.split(';').next().unwrap_or("").trim();
        if line.starts_with("s_endpgm") {
            pending.clear();
            continue;
        }
        let Some((name, operands)) = line.split_once(char::is_whitespace) else { continue };
        if name.starts_with('.') || name.ends_with(':') { continue; }
        let wait_kind = match name {
            "s_wait_loadcnt" => Some(Kind::Vmem),
            "s_wait_storecnt" => Some(Kind::Store),
            "s_wait_dscnt" => Some(Kind::Ds),
            "s_wait_kmcnt" => Some(Kind::Km),
            _ => None,
        };
        if let Some(kind) = wait_kind {
            let count = wait_count(operands).ok_or_else(|| format!("line {}: invalid wait", line_no+1))?;
            retire(&mut pending, kind, count);
            continue;
        }
        if name == "s_wait_loadcnt_dscnt" {
            let encoded = wait_count(operands)
                .ok_or_else(|| format!("line {}: invalid combined wait", line_no+1))?;
            if encoded > 0x3f3f || encoded & 0xc0c0 != 0 {
                return Err(format!("line {}: out-of-range combined wait", line_no+1));
            }
            retire(&mut pending, Kind::Vmem, encoded & 0x3f);
            retire(&mut pending, Kind::Ds, (encoded >> 8) & 0x3f);
            continue;
        }
        let kind = if name.starts_with("buffer_load") || name.starts_with("global_load") {
            Some((Kind::Vmem, true))
        } else if name.starts_with("buffer_store") || name.starts_with("global_store") {
            Some((Kind::Store, false))
        } else if name.starts_with("ds_load") {
            Some((Kind::Ds, true))
        } else if name.starts_with("ds_store") {
            Some((Kind::Ds, false))
        } else if name.starts_with("s_load") || name.starts_with("s_buffer_load") {
            Some((Kind::Km, true))
        } else if name.starts_with("s_store") || name.starts_with("s_buffer_store") {
            Some((Kind::Km, false))
        } else { None };
        let all_regs = registers(operands);
        let first = operands.split(',').next().unwrap_or("");
        let defs = if kind.is_some_and(|(_, load)| load)
            || name.starts_with("v_") || name.starts_with("s_mov")
            || name.starts_with("s_add") || name.starts_with("s_sub") {
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
        } else { BTreeSet::new() };
        if pending.iter().any(|event| !event.defs.is_disjoint(&all_regs)
            || !event.locks.is_disjoint(&defs)) {
            return Err(format!("line {}: {name} touches unfinished memory operands", line_no+1));
        }
        if let Some((kind, load)) = kind {
            pending.push(Pending {
                defs: if load { defs } else { BTreeSet::new() },
                locks: if load { BTreeSet::new() } else { all_regs },
                kind, opcode: name,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::replay_waits;
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
            s_wait_loadcnt_dscnt 0x1\n\
            v_add_f32 v2, v1, v3\n";
        assert!(replay_waits(stream).is_ok());
        assert!(replay_waits(&(stream.to_owned() + "v_add_f32 v2, v0, v3\n")).is_err());
    }
    #[test]
    fn store_lock_requires_storecnt_before_redefinition() {
        let source = "buffer_store_b32 v0, v1, s[4:7], s8 offen\n\
            v_mov_b32 v0, 0\n";
        assert!(replay_waits(source).is_err());
        assert!(replay_waits(&source.replace("v_mov_b32", "s_wait_storecnt 0\nv_mov_b32")).is_ok());
        assert!(replay_waits("buffer_store_b32 v0, v1, s[4:7], s8 offen\n\
            v_add_f32 v2, v0, v3\n").is_ok());
    }
    #[test]
    fn smem_load_requires_kmcnt_before_scalar_use() {
        let source = "s_load_b32 s4, s[0:1], 0\ns_add_u32 s5, s4, s6\n";
        assert!(replay_waits(source).is_err());
        assert!(replay_waits(&source.replace("s_add_u32", "s_wait_kmcnt 0\ns_add_u32")).is_ok());
    }
    #[test]
    fn mixed_vmem_loads_require_zero_wait() {
        let source = "buffer_load_b64 v[0:1], v8, s[4:7], s9 offen\n\
            global_load_b32 v2, v8, s[4:5]\n\
            s_wait_loadcnt 1\nv_add_f32 v3, v0, v4\n";
        assert!(replay_waits(source).is_err());
        assert!(replay_waits(&source.replace("s_wait_loadcnt 1", "s_wait_loadcnt 0")).is_ok());
        assert!(replay_waits("global_load_b32 v0, v8, s[4:5]\n\
            global_load_b64 v[1:2], v9, s[4:5]\n\
            s_wait_loadcnt 1\nv_add_f32 v3, v0, v4\n").is_ok());
    }
}
