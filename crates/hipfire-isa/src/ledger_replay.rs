// SPDX-License-Identifier: Apache-2.0
//! Independent, source-text replay of VMEM/DS load RAW waits. This consumes
//! assembly text, not the builder's pending-event ledger.
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
struct Load { regs: BTreeSet<u16>, kind: Kind }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind { Vmem, Ds }

fn registers(operands: &str) -> BTreeSet<u16> {
    let bytes = operands.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'v' || (i > 0 && (bytes[i-1].is_ascii_alphanumeric() || bytes[i-1] == b'_')) {
            i += 1; continue;
        }
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
        if last >= first && last - first <= 255 {
            found.extend(first..=last);
        }
    }
    found
}
fn wait_count(rest: &str) -> Option<usize> {
    let token = rest.split_whitespace().next()?.split('_').next()?;
    if let Some(hex) = token.strip_prefix("0x") { usize::from_str_radix(hex, 16).ok() }
    else { token.parse().ok() }
}

fn retire(pending: &mut Vec<Load>, kind: Kind, count: usize) {
    let mut outstanding = pending.iter().filter(|item| item.kind == kind).count();
    pending.retain(|item| {
        if item.kind == kind && outstanding > count {
            outstanding -= 1;
            false
        } else { true }
    });
}

/// Reject reads or overwrites of unfinished VMEM/DS load destinations.
/// Combined waits decode both counters independently, without assuming a full drain.
pub fn replay_waits(assembly: &str) -> Result<(), String> {
    let mut pending = Vec::<Load>::new();
    for (line_no, source) in assembly.lines().enumerate() {
        if source.trim_start().starts_with("s_endpgm") {
            pending.clear();
            continue;
        }
        let line = source.split(';').next().unwrap_or("").trim();
        let Some((name, operands)) = line.split_once(char::is_whitespace) else { continue };
        if name.starts_with('.') || name.ends_with(':') { continue; }
        let kind = if name.starts_with("buffer_load") || name.starts_with("global_load") {
            Some(Kind::Vmem)
        } else if name.starts_with("ds_load") { Some(Kind::Ds) } else { None };
        if name == "s_wait_loadcnt" || name == "s_wait_dscnt" {
            let target = if name == "s_wait_loadcnt" { Kind::Vmem } else { Kind::Ds };
            let count = wait_count(operands).ok_or_else(|| format!("line {}: invalid wait", line_no+1))?;
            retire(&mut pending, target, count);
            continue;
        }
        if name == "s_wait_loadcnt_dscnt" {
            let encoded = wait_count(operands)
                .ok_or_else(|| format!("line {}: invalid combined wait", line_no+1))?;
            if encoded > 0x3f3f {
                return Err(format!("line {}: out-of-range combined wait", line_no+1));
            }
            retire(&mut pending, Kind::Vmem, encoded & 0x3f);
            retire(&mut pending, Kind::Ds, (encoded >> 8) & 0x3f);
            continue;
        }
        if let Some(kind) = kind {
            let destination = operands.split(',').next().unwrap_or("");
            let address = operands.strip_prefix(destination).unwrap_or("");
            let addresses = registers(address);
            if pending.iter().any(|load| !load.regs.is_disjoint(&addresses)) {
                return Err(format!("line {}: address uses outstanding load", line_no+1));
            }
            let defs = registers(destination);
            if pending.iter().any(|load| !load.regs.is_disjoint(&defs)) {
                return Err(format!("line {}: overwrites outstanding load", line_no+1));
            }
            pending.push(Load { regs: defs, kind });
        } else if !name.starts_with('s') || name.starts_with("s_store") || name.starts_with("s_barrier") {
            let used = registers(operands);
            if pending.iter().any(|load| !load.regs.is_disjoint(&used)) {
                return Err(format!("line {}: {name} reads outstanding load", line_no+1));
            }
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
}
