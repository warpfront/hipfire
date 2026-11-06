// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Correlated branches for the VGPR allocator's reaching definitions (`regalloc`).
//!
//! Two conditional branches that test the same scalar predicate (`s_bitcmp0_b32 sM, 3` ... `s_bitcmp1_b32 sM, 3`,
//! with no write of `sM` between them) take the same side. The CFG does not know that: a value defined in the
//! prologue and redefined under `if (bit 3 of sM)` reaches every later read under the same `if` along the path
//! that skipped the first one, so the prologue definition looks live across all the guarded regions in between,
//! although no execution can read it.
//!
//! `refine` recomputes the reaching definitions of every unit with more than one candidate definition, per
//! predicate that is known where the unit is referenced, over the CFG split by the predicate's value (unknown /
//! true / false, reset by a write of one of its SGPRs); an edge contradicting the known value is not taken. Each
//! refined set is a superset of what real executions see, so their intersection is too, and replaces the
//! path-insensitive set only when it is not empty.
use crate::insn::Instruction;
use crate::reg::Kind;
use crate::regalloc::{Block, Item, Op, Reaching, UNDEF, UNITS};
use std::collections::HashMap;

/// A branch on predicate `key` is taken when the predicate equals `taken_when`.
struct Guard {
    key: usize,
    taken_when: bool,
}

struct Keys {
    ids: HashMap<String, usize>,
    /// SGPR units each predicate reads.
    sgprs: Vec<Vec<u8>>,
}

fn sgpr_tokens(text: &str) -> Vec<u8> {
    let b = text.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let prev_word = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if b[i] == b's' && !prev_word {
            let j = i + 1;
            if j < b.len() && b[j] == b'[' {
                if let Some(close) = text[j..].find(']').map(|k| k + j) {
                    if let Some((lo, hi)) = text[j + 1..close].split_once(':') {
                        if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<u8>(), hi.trim().parse::<u8>()) {
                            out.extend(lo..=hi);
                            i = close + 1;
                            continue;
                        }
                    }
                }
            } else {
                let mut k = j;
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                let next_word = k < b.len() && (b[k].is_ascii_alphanumeric() || b[k] == b'_');
                if k > j && !next_word {
                    if let Ok(n) = text[j..k].parse::<u8>() {
                        out.push(n);
                        i = k;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// The SGPR units an instruction writes: its declared SGPR defs and, for an instruction whose first operand is
/// a destination, that operand.
fn sgpr_defs(insn: &Instruction) -> Vec<u8> {
    let mut out = vec![];
    for r in insn.defs.iter().filter(|r| r.kind == Kind::S) {
        out.extend(r.base..r.base.saturating_add(r.len));
    }
    let m = insn.mnemonic();
    let reads_first = m.starts_with("s_cmp")
        || m.starts_with("s_bitcmp")
        || m.starts_with("s_cbranch")
        || m.starts_with("s_wait")
        || m.starts_with("s_store")
        || m.starts_with("s_buffer_store")
        || m == "s_branch"
        || m == "s_endpgm";
    if !reads_first {
        if let Some((_, rest)) = insn.text.split_once(char::is_whitespace) {
            let first = rest.split(',').next().unwrap_or("").trim();
            if first.starts_with('s') {
                out.extend(sgpr_tokens(first));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Predicate of a compare, normalised so complementary compares share a key: `(key text, sense)`, where the
/// compare's result is `predicate == sense`.
fn predicate(insn: &Instruction) -> Option<(String, bool, Vec<u8>)> {
    let m = insn.mnemonic();
    let (_, rest) = insn.text.split_once(char::is_whitespace)?;
    let ops: Vec<&str> = rest.split(',').map(str::trim).collect();
    let [a, b] = ops.as_slice() else { return None };
    let (key, sense) = if let Some(s) = m.strip_prefix("s_bitcmp") {
        let sense = match s {
            "1_b32" | "1_b64" => true,
            "0_b32" | "0_b64" => false,
            _ => return None,
        };
        (format!("bit{} {a},{b}", &s[2..]), sense)
    } else if let Some(t) = m.strip_prefix("s_cmp_eq_") {
        (format!("eq_{t} {a},{b}"), true)
    } else if let Some(t) = m.strip_prefix("s_cmp_lg_") {
        (format!("eq_{t} {a},{b}"), false)
    } else if m.starts_with("s_cmp_") {
        (format!("{m} {a},{b}"), true)
    } else {
        return None;
    };
    let mut sg = sgpr_tokens(a);
    sg.extend(sgpr_tokens(b));
    sg.sort_unstable();
    sg.dedup();
    Some((key, sense, sg))
}

fn guards(
    items: &[Item],
    blocks: &[Block],
    insns: &[&Instruction],
) -> (Vec<Option<Guard>>, Keys) {
    let mut keys = Keys {
        ids: HashMap::new(),
        sgprs: vec![],
    };
    let mut out: Vec<Option<Guard>> = Vec::with_capacity(blocks.len());
    for blk in blocks {
        let mut g = None;
        if let Item::Insn(j) = &items[blk.last] {
            let m = insns[*j].mnemonic();
            if m == "s_cbranch_scc0" || m == "s_cbranch_scc1" {
                let scc1 = m == "s_cbranch_scc1";
                // the nearest preceding scalar instruction decides SCC (VALU never writes it)
                let cmp = items[blk.first..blk.last].iter().rev().find_map(|it| match it {
                    Item::Insn(k) if !insns[*k].mnemonic().starts_with("v_") => Some(*k),
                    _ => None,
                });
                if let Some(k) = cmp {
                    if let Some((key, sense, sg)) = predicate(insns[k]) {
                        let n = keys.ids.len();
                        let id = *keys.ids.entry(key).or_insert_with(|| {
                            keys.sgprs.push(sg);
                            n
                        });
                        g = Some(Guard {
                            key: id,
                            taken_when: scc1 == sense,
                        });
                    }
                }
            }
        }
        out.push(g);
    }
    (out, keys)
}

const U: u8 = 0;
const T: u8 = 1;
const F: u8 = 2;

fn val(b: bool) -> u8 {
    if b {
        T
    } else {
        F
    }
}

/// Refine `rd.use_rd` in place (see the module doc).
pub(crate) fn refine(
    items: &[Item],
    blocks: &[Block],
    insns: &[&Instruction],
    ops: &[Op],
    live_in: &[bool],
    rd: &mut Reaching,
) {
    let n = blocks.len();
    if n == 0 {
        return;
    }
    let (guard, keys) = guards(items, blocks, insns);
    let nk = keys.ids.len();
    if nk == 0 {
        return;
    }
    // keys killed by a write of one of their SGPRs, per block
    let mut by_sgpr: Vec<Vec<usize>> = vec![vec![]; 256];
    for (k, sg) in keys.sgprs.iter().enumerate() {
        for &u in sg {
            by_sgpr[u as usize].push(k);
        }
    }
    let mut killed = vec![vec![false; nk]; n];
    let mut block_of_item = vec![0usize; items.len()];
    for (b, blk) in blocks.iter().enumerate() {
        for k in blk.first..=blk.last {
            block_of_item[k] = b;
            if let Item::Insn(i) = &items[k] {
                for u in sgpr_defs(insns[*i]) {
                    for &key in &by_sgpr[u as usize] {
                        killed[b][key] = true;
                    }
                }
            }
        }
    }
    let mut preds: Vec<Vec<usize>> = vec![vec![]; n];
    for (b, blk) in blocks.iter().enumerate() {
        for &s in &blk.succ {
            preds[s].push(b);
        }
    }
    // Edge value of key `key` leaving `b` towards successor index `si`, given the value at the block's end.
    let edge_value = |b: usize, si: usize, key: usize, at_end: u8| -> Option<u8> {
        match &guard[b] {
            Some(g) if g.key == key && blocks[b].succ.len() == 2 => {
                let on_taken = si == 0;
                let want = if on_taken { g.taken_when } else { !g.taken_when };
                let opposite = val(!want);
                if at_end == opposite {
                    None
                } else {
                    Some(val(want))
                }
            }
            _ => Some(at_end),
        }
    };
    // Must-facts at block entry: per key U/T/F, `None` = no path seen yet.
    let mut must: Vec<Vec<Option<u8>>> = vec![vec![None; nk]; n];
    for key in 0..nk {
        must[0][key] = Some(U);
    }
    loop {
        let mut changed = false;
        for b in 0..n {
            for key in 0..nk {
                let mut acc: Option<u8> = if b == 0 { Some(U) } else { None };
                for &p in &preds[b] {
                    let Some(pin) = must[p][key] else { continue };
                    let at_end = if killed[p][key] { U } else { pin };
                    let si = blocks[p].succ.iter().position(|&s| s == b).unwrap();
                    let Some(v) = edge_value(p, si, key, at_end) else { continue };
                    acc = Some(match acc {
                        None => v,
                        Some(a) if a == v => a,
                        Some(_) => U,
                    });
                }
                if acc.is_some() && acc != must[b][key] {
                    must[b][key] = acc;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // references per unit: instruction indices in item order
    let mut refs: Vec<Vec<usize>> = vec![vec![]; UNITS];
    for it in items {
        if let Item::Insn(i) = it {
            if !rd.reachable[*i] {
                continue;
            }
            for &u in ops[*i].defs.iter().chain(&ops[*i].uses) {
                if refs[u as usize].last() != Some(i) {
                    refs[u as usize].push(*i);
                }
            }
        }
    }
    let mut item_of_insn = vec![0usize; insns.len()];
    for (k, it) in items.iter().enumerate() {
        if let Item::Insn(i) = it {
            item_of_insn[*i] = k;
        }
    }
    for u in 0..UNITS {
        // units whose uses have more than one candidate definition
        let uncertain = refs[u].iter().any(|&i| {
            ops[i]
                .uses
                .iter()
                .position(|x| *x as usize == u)
                .is_some_and(|k| rd.use_rd[i][k].len() > 1)
        });
        if !uncertain {
            continue;
        }
        let mut cand: Vec<usize> = vec![];
        for &i in &refs[u] {
            let b = block_of_item[item_of_insn[i]];
            for key in 0..nk {
                if matches!(must[b][key], Some(T) | Some(F)) && !cand.contains(&key) {
                    cand.push(key);
                }
            }
        }
        if cand.is_empty() {
            continue;
        }
        // per block: the references of `u` in order
        let mut block_refs: HashMap<usize, Vec<usize>> = HashMap::new();
        for &i in &refs[u] {
            block_refs
                .entry(block_of_item[item_of_insn[i]])
                .or_default()
                .push(i);
        }
        let entry_def = if live_in[u] { u as u32 } else { UNDEF };
        let mut meet: HashMap<(usize, usize), Vec<u32>> = HashMap::new(); // (insn, k) -> intersection
        for &key in &cand {
            // nodes: (block, ctx); in-state = candidate definitions of `u` at block entry
            let mut st: Vec<[Option<Vec<u32>>; 3]> = vec![[None, None, None]; n];
            st[0][U as usize] = Some(vec![entry_def]);
            let mut work: std::collections::VecDeque<(usize, u8)> = [(0usize, U)].into();
            let out_of = |b: usize, s: &Vec<u32>| -> Vec<u32> {
                match block_refs.get(&b).and_then(|r| {
                    r.iter()
                        .rev()
                        .find_map(|&i| ops[i].defs.iter().position(|x| *x as usize == u).map(|j| rd.def_id[i][j]))
                }) {
                    Some(d) => vec![d],
                    None => s.clone(),
                }
            };
            while let Some((b, ctx)) = work.pop_front() {
                let s_in = st[b][ctx as usize].clone().unwrap();
                let s_out = out_of(b, &s_in);
                let at_end = if killed[b][key] { U } else { ctx };
                for (si, &s) in blocks[b].succ.iter().enumerate() {
                    let Some(v) = edge_value(b, si, key, at_end) else { continue };
                    let slot = &mut st[s][v as usize];
                    let mut changed = false;
                    match slot {
                        None => {
                            *slot = Some(s_out.clone());
                            changed = true;
                        }
                        Some(into) => {
                            for &x in &s_out {
                                if let Err(p) = into.binary_search(&x) {
                                    into.insert(p, x);
                                    changed = true;
                                }
                            }
                        }
                    }
                    if changed {
                        work.push_back((s, v));
                    }
                }
            }
            // record the uses
            let mut seen: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
            for (b, rs) in &block_refs {
                for ctx in 0..3 {
                    let Some(s_in) = &st[*b][ctx] else { continue };
                    let mut cur = s_in.clone();
                    for &i in rs {
                        if let Some(k) = ops[i].uses.iter().position(|x| *x as usize == u) {
                            let e = seen.entry((i, k)).or_default();
                            for &x in &cur {
                                if let Err(p) = e.binary_search(&x) {
                                    e.insert(p, x);
                                }
                            }
                        }
                        if let Some(j) = ops[i].defs.iter().position(|x| *x as usize == u) {
                            cur = vec![rd.def_id[i][j]];
                        }
                    }
                }
            }
            for (k, v) in seen {
                match meet.get_mut(&k) {
                    None => {
                        meet.insert(k, v);
                    }
                    Some(m) => m.retain(|x| v.binary_search(x).is_ok()),
                }
            }
        }
        for ((i, k), set) in meet {
            if !set.is_empty() && set.len() < rd.use_rd[i][k].len() {
                rd.use_rd[i][k] = set;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::{build_cfg, exec_dirty_of, operands, reaching, reaching_refined, StreamItem};
    use crate::reg::RegRef;

    fn v(base: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len: 1,
        }
    }
    fn s(base: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len: 1,
        }
    }
    fn i(text: &str, defs: &[RegRef], uses: &[RegRef]) -> StreamItem {
        StreamItem::Insn {
            insn: Instruction::new(text, defs.to_vec(), uses.to_vec()),
            subs: [0, usize::MAX],
        }
    }

    /// Reaching definitions of every use, plain and refined, for a stream.
    fn chains(stream: &[StreamItem]) -> (Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<u32>>>, Vec<Vec<u32>>) {
        let mut insns: Vec<&Instruction> = vec![];
        let mut items: Vec<Item> = vec![];
        for it in stream {
            match it {
                StreamItem::Insn { insn, .. } => {
                    items.push(Item::Insn(insns.len()));
                    insns.push(insn);
                }
                StreamItem::Label(l) => items.push(Item::Label(l.clone())),
            }
        }
        let blocks = build_cfg(&items, &insns).unwrap();
        let dirty = exec_dirty_of(&items, &blocks, &insns);
        let ops: Vec<Op> = insns
            .iter()
            .enumerate()
            .map(|(k, ins)| operands(ins, dirty[k]).unwrap())
            .collect();
        let mut live = vec![false; UNITS];
        live[0] = true;
        let plain = reaching(&items, &blocks, &ops, &live);
        let refined = reaching_refined(&items, &blocks, &insns, &ops, &live);
        (plain.use_rd, refined.use_rd, refined.def_id)
    }

    /// `v5 = 0; if (bit 3 of s1) v5 = ..; ..; if (bit 3 of s1) use v5`: the second guard implies the first.
    fn guarded(second_cmp: &str, second_branch: &str, rewrite_between: bool) -> Vec<StreamItem> {
        let mut st = vec![
            i("v_mov_b32_e32 v5, 0", &[v(5)], &[]),
            i("s_bitcmp0_b32 s1, 3", &[], &[s(1)]),
            i("s_cbranch_scc1 .L1", &[], &[]),
            i("v_add_f32_e32 v5, 0, v6", &[v(5)], &[v(6)]),
            StreamItem::Label(".L1".into()),
            i("v_mov_b32_e32 v9, 1", &[v(9)], &[]),
        ];
        if rewrite_between {
            st.push(i("s_mov_b32 s1, 7", &[s(1)], &[]));
        }
        st.extend([
            i(second_cmp, &[], &[s(1)]),
            i(second_branch, &[], &[]),
            i("v_add_f32_e32 v8, v5, v6", &[v(8)], &[v(5), v(6)]),
            StreamItem::Label(".L2".into()),
            i("v_add_f32_e32 v10, v9, v9", &[v(10)], &[v(9), v(9)]),
            i("s_endpgm", &[], &[]),
        ]);
        st
    }
    fn use_of_v5(st: &[StreamItem], sets: &[Vec<Vec<u32>>]) -> Vec<u32> {
        let at = st
            .iter()
            .filter_map(|it| match it {
                StreamItem::Insn { insn, .. } => Some(insn),
                _ => None,
            })
            .position(|ins| ins.text.starts_with("v_add_f32_e32 v8"))
            .unwrap();
        sets[at][0].clone()
    }

    #[test]
    fn a_repeated_guard_hides_the_prologue_definition_from_the_guarded_read() {
        let st = guarded("s_bitcmp0_b32 s1, 3", "s_cbranch_scc1 .L2", false);
        let (plain, refined, ids) = chains(&st);
        let both = use_of_v5(&st, &plain);
        assert_eq!(both.len(), 2, "the CFG alone lets both definitions reach: {both:?}");
        let only = use_of_v5(&st, &refined);
        assert_eq!(only, vec![ids[3][0]], "only the guarded definition reaches the guarded read");
    }

    #[test]
    fn the_complementary_test_of_the_same_bit_correlates_too() {
        // skips when the bit is set, so the read runs when it is clear: the first guard's body did not run
        let st = guarded("s_bitcmp1_b32 s1, 3", "s_cbranch_scc1 .L2", false);
        let (_, refined, ids) = chains(&st);
        assert_eq!(use_of_v5(&st, &refined), vec![ids[0][0]]);
        // taken when bit 3 is clear (scc0 after bitcmp1 skips on clear): the read runs when the bit is set, as the
        // first guard's body did
        let st = guarded("s_bitcmp1_b32 s1, 3", "s_cbranch_scc0 .L2", false);
        let (_, refined, ids) = chains(&st);
        assert_eq!(use_of_v5(&st, &refined), vec![ids[3][0]]);
    }

    #[test]
    fn another_bit_or_a_rewritten_register_correlates_nothing() {
        let st = guarded("s_bitcmp0_b32 s1, 4", "s_cbranch_scc1 .L2", false);
        assert_eq!(use_of_v5(&st, &chains(&st).1).len(), 2);
        let st = guarded("s_bitcmp0_b32 s1, 3", "s_cbranch_scc1 .L2", true);
        assert_eq!(use_of_v5(&st, &chains(&st).1).len(), 2);
    }
}
