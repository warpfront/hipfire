// SPDX-License-Identifier: Apache-2.0
//! gfx11 certification of a linked builder object, beyond the text-level
//! parse-back and wait replay:
//!
//! - `m7`: the linked ELF is lifted with `peacemaker-lift`, which decodes
//!   every instruction word through M7's pinned gfx1100/gfx1151 opcode tables
//!   and rejects unknown encodings; `emit::module` must reproduce the object
//!   byte for byte. M7's whole-program analyses then run on the lifted
//!   kernel: wait replay (`passes::waits`, gfx11 VM/LGKM/VS counters), the
//!   gfx11 hazard replay (`passes::hazards`: gfx1100 TRANS-use, VCMPX ->
//!   PERMLANE, WMMA chaining, joined over loop back-edges), barrier pairing
//!   and the barrier LDS-drain rule (`passes::barriers`), `s_clause` /
//!   `s_delay_alu` windows and register resources. Every obligation fails
//!   certification.
//! - `lds_bounds`: every DS address register is evaluated for every lane of
//!   every wave of the workgroup by executing the straight-line entry block
//!   (up to the first label or branch) that derives it from the work-item id (kernel arguments and workgroup ids are
//!   unknown and must not reach an address), and every DS access must end
//!   inside the launch allocation.
use crate::toolchain::Result;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

fn ir_arch(arch: &str) -> Result<peacemaker_ir::Arch> {
    match arch {
        "gfx1100" => Ok(peacemaker_ir::Arch::Gfx1100),
        "gfx1151" => Ok(peacemaker_ir::Arch::Gfx1151),
        "gfx1201" => Ok(peacemaker_ir::Arch::Gfx1201),
        _ => Err(format!("M7 certification does not cover {arch}")),
    }
}

/// Lift, re-emit and analyze `symbol` in the linked ELF `elf`.
pub fn m7(elf: &Path, arch: &str, symbol: &str) -> Result<Value> {
    use peacemaker_ir::passes::{barriers, windows};
    let target = ir_arch(arch)?;
    let bytes = std::fs::read(elf).map_err(|e| format!("{}: {e}", elf.display()))?;
    let lifted = peacemaker_lift::lift_object(&bytes, peacemaker_lift::Options { frontend: peacemaker_ir::inst::Frontend::Builder })
        .map_err(|e| format!("M7 lift rejected {}: {e}", elf.display()))?;
    if lifted.program.target.arch != target { return Err(format!("M7 lift read {:?}, expected {arch}", lifted.program.target.arch)) }
    if peacemaker_lift::emit::module(&lifted.program).map_err(|e| e.to_string())? != bytes {
        return Err("M7 emit::module does not reproduce the linked object".into());
    }
    let kernel = lifted.program.kernels.iter().find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("M7 lift has no kernel {symbol}"))?;
    let instructions = peacemaker_lift::emit::insts(kernel, target).map_err(|e| e.to_string())?.len();
    let analysis = peacemaker_ir::edit::analyze(lifted.program.clone(), &kernel.symbol).map_err(|e| e.to_string())?;
    let pairings = barriers::analyze(&kernel.body, target).map_err(|e| e.to_string())?;
    let win = windows::check_windows(&kernel.body).map_err(|e| e.to_string())?;
    let mut obligations = BTreeMap::<String, usize>::new();
    let mut details = Vec::new();
    for o in analysis.obligations.iter().chain(&pairings.obligations) {
        *obligations.entry(format!("{:?}/{}", o.kind, o.rule_id)).or_default() += 1;
        let pcs: Vec<String> = o.insts.iter().filter_map(|id| kernel.body.insts.get(*id)).filter_map(|i| i.prov.pc).map(|pc| format!("{pc:#x}")).collect();
        details.push(format!("{:?}/{} at {pcs:?}: {}", o.kind, o.rule_id, o.text));
    }
    let ambiguous = win.delays.iter().filter(|d| d.is_ambiguous()).count();
    let resources = analysis.facts.resources.first().ok_or("M7 analysis has no resource summary")?;
    let unknown = analysis.facts.lds.accesses.iter().filter(|(a, _)| matches!(a.addr, peacemaker_ir::lds::AddrFact::Unknown)).count();
    let summary = json!({
        "lift": "byte-exact", "table": arch, "instructions": instructions,
        "obligations": obligations,
        "wait_facts": analysis.facts.waits.len(),
        "barrier_pairs": pairings.pairs.len(),
        "clauses": win.clauses.len(), "delays": win.delays.len(), "ambiguous_delays": ambiguous,
        "vgpr_high_water": resources.max_vgpr, "sgpr_high_water": resources.max_sgpr,
        "lds_accesses": analysis.facts.lds.accesses.len(), "lds_unknown_addresses": unknown,
        "lds_max_known_end": analysis.facts.lds.max_end,
    });
    if !obligations.is_empty() || ambiguous != 0 {
        return Err(format!("M7 analysis of {symbol} left obligations: {summary}\n{}", details.join("\n")));
    }
    Ok(summary)
}

/// Lane values of one wave: `None` is a value the prologue cannot know.
type Lanes = [Option<u32>; 32];

fn parse_imm(t: &str) -> Option<u32> {
    let t = t.trim();
    if let Some(h) = t.strip_prefix("0x") { u32::from_str_radix(h, 16).ok() }
    else if let Some(n) = t.strip_prefix('-') { n.parse::<u32>().ok().map(|n| n.wrapping_neg()) }
    else { t.parse().ok() }
}

struct Wave { v: BTreeMap<u16, Lanes>, s: BTreeMap<u16, Option<u32>> }
impl Wave {
    fn operand(&self, t: &str) -> Lanes {
        let t = t.trim();
        if let Some(n) = t.strip_prefix('v').and_then(|n| n.parse::<u16>().ok()) { return self.v.get(&n).copied().unwrap_or([None; 32]) }
        if let Some(n) = t.strip_prefix('s').and_then(|n| n.parse::<u16>().ok()) { return [self.s.get(&n).copied().flatten(); 32] }
        [parse_imm(t); 32]
    }
    fn scalar(&self, t: &str) -> Option<u32> {
        let t = t.trim();
        if let Some(n) = t.strip_prefix('s').and_then(|n| n.parse::<u16>().ok()) { return self.s.get(&n).copied().flatten() }
        parse_imm(t)
    }
}

fn dst(first: &str) -> Option<(char, u16)> {
    let t = first.trim();
    let class = t.chars().next()?;
    if !matches!(class, 'v' | 's') { return None }
    t[1..].parse().ok().map(|n| (class, n))
}

/// `v[lo:hi]` / `s[lo:hi]`.
fn range_dst(first: &str) -> Option<(char, u16, u16)> {
    let t = first.trim();
    let class = t.chars().next()?;
    if !matches!(class, 'v' | 's') { return None }
    let (lo, hi) = t[1..].strip_prefix('[')?.strip_suffix(']')?.split_once(':')?;
    Some((class, lo.parse().ok()?, hi.parse().ok()?))
}

/// Execute one straight-line instruction for all lanes. Unknown operations
/// make their destination unknown; scalar compares write only SCC, and a
/// register-range destination makes every register of the range unknown.
fn step(w: &mut Wave, name: &str, ops: &[&str]) {
    if name.starts_with("s_cmp") || name.starts_with("s_bitcmp") { return }
    if let Some((class, lo, hi)) = ops.first().and_then(|o| range_dst(o)) {
        for n in lo..=hi { if class == 'v' { w.v.insert(n, [None; 32]); } else { w.s.insert(n, None); } }
        return;
    }
    let Some((class, n)) = ops.first().and_then(|o| dst(o)) else { return };
    let lanes = |f: &dyn Fn(usize) -> Option<u32>| -> Lanes { std::array::from_fn(f) };
    let a = |i: usize| ops.get(i).map(|o| w.operand(o)).unwrap_or([None; 32]);
    let bin = |f: fn(u32, u32) -> u32| { let (x, y) = (a(1), a(2)); lanes(&|l| Some(f(x[l]?, y[l]?))) };
    let tri = |f: fn(u32, u32, u32) -> u32| { let (x, y, z) = (a(1), a(2), a(3)); lanes(&|l| Some(f(x[l]?, y[l]?, z[l]?))) };
    let value: Lanes = match name {
        "v_mov_b32_e32" => a(1),
        "v_and_b32_e32" => bin(|x, y| x & y),
        "v_or_b32_e32" => bin(|x, y| x | y),
        "v_add_nc_u32_e32" => bin(|x, y| x.wrapping_add(y)),
        "v_sub_nc_u32_e32" => bin(|x, y| x.wrapping_sub(y)),
        "v_lshrrev_b32_e32" => bin(|s, x| x >> (s & 31)),
        "v_lshlrev_b32_e32" => bin(|s, x| x << (s & 31)),
        "v_mul_u32_u24_e32" => bin(|x, y| (x & 0xff_ffff).wrapping_mul(y & 0xff_ffff)),
        "v_lshl_add_u32" => tri(|x, s, y| (x << (s & 31)).wrapping_add(y)),
        "v_mad_u32_u24" => tri(|x, y, z| (x & 0xff_ffff).wrapping_mul(y & 0xff_ffff).wrapping_add(z)),
        "v_readfirstlane_b32" => { let x = a(1); [x[0]; 32] }
        "s_lshr_b32" | "s_and_b32" | "s_lshl_b32" | "s_mov_b32" | "s_add_i32" | "s_mul_i32" => {
            let x = w.scalar(ops.get(1).unwrap_or(&""));
            let y = ops.get(2).and_then(|o| w.scalar(o));
            let r = match name {
                "s_mov_b32" => x,
                "s_lshr_b32" => x.zip(y).map(|(x, y)| x >> (y & 31)),
                "s_lshl_b32" => x.zip(y).map(|(x, y)| x << (y & 31)),
                "s_and_b32" => x.zip(y).map(|(x, y)| x & y),
                "s_add_i32" => x.zip(y).map(|(x, y)| x.wrapping_add(y)),
                _ => x.zip(y).map(|(x, y)| x.wrapping_mul(y)),
            };
            [r; 32]
        }
        _ => [None; 32],
    };
    if class == 'v' { w.v.insert(n, value); } else { w.s.insert(n, value[0]); }
}

/// Byte ranges `(offset, width)` a DS instruction touches relative to its address VGPR.
fn ds_ranges(name: &str, ops: &str) -> Option<Vec<(u32, u32)>> {
    let field = |key: &str| ops.split_whitespace().find_map(|t| t.strip_prefix(key)).and_then(parse_imm).unwrap_or(0);
    let width = |suffix: &str| match suffix { "b32" => Some(4), "b64" => Some(8), "b96" => Some(12), "b128" => Some(16), _ => None };
    let op = name.strip_prefix("ds_load_").or_else(|| name.strip_prefix("ds_store_"))?;
    if let Some(rest) = op.strip_prefix("2addr_stride64_") {
        let w = width(rest)?;
        Some(vec![(field("offset0:") * 64 * w, w), (field("offset1:") * 64 * w, w)])
    } else if let Some(rest) = op.strip_prefix("2addr_") {
        let w = width(rest)?;
        Some(vec![(field("offset0:") * w, w), (field("offset1:") * w, w)])
    } else {
        Some(vec![(field("offset:"), width(op)?)])
    }
}

/// Every DS access of every lane of `waves` waves of `symbol`, whose address
/// VGPR the entry block (the straight-line code before the first label or
/// branch) derives from the work-item id and which nothing after it
/// redefines, ends at or before `limit` bytes. Returns the maximum end. A
/// kernel without DS memory accesses passes only with no allocation
/// (`limit == 0`).
pub fn lds_bounds(source: &str, symbol: &str, waves: u32, limit: u32) -> Result<u32> {
    let marker = format!("\n{symbol}:\n");
    let start = source.find(&marker).ok_or_else(|| format!("{symbol}: missing kernel label"))? + marker.len();
    let body = &source[start - 1..];
    let body = &body[..body.find(&format!("\n.L{symbol}_end:")).ok_or("missing kernel end label")?];
    let lines: Vec<(&str, Vec<&str>, &str)> = body.lines().filter_map(|l| {
        let t = l.split(';').next()?.trim();
        if t.is_empty() || t.starts_with('.') && !t.ends_with(':') { return None }
        let (name, rest) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
        let ops = if name.starts_with("v_dual_") { vec![] } else { rest.split(',').map(str::trim).collect() };
        Some((name, ops, rest))
    }).collect();
    let head = lines.iter().position(|(n, _, _)| n.ends_with(':') || n.starts_with("s_branch") || n.starts_with("s_cbranch"))
        .ok_or("kernel has no block boundary")?;
    let mut end = 0u32;
    let mut checked = 0usize;
    for wave in 0..waves {
        let mut w = Wave { v: BTreeMap::new(), s: BTreeMap::new() };
        w.v.insert(0, std::array::from_fn(|l| Some(wave * 32 + l as u32)));
        for (i, (name, ops, rest)) in lines.iter().enumerate() {
            // ds_swizzle_b32 exchanges lanes on the LDS crossbar and touches no LDS memory.
            if name.starts_with("ds_") && *name != "ds_swizzle_b32" {
                let addr_op = if name.starts_with("ds_store") { ops.first() } else { ops.get(1) };
                let reg = addr_op.map(|o| o.split_whitespace().next().unwrap_or("")).and_then(|o| o.strip_prefix('v')).and_then(|n| n.parse::<u16>().ok())
                    .ok_or_else(|| format!("{name}: unparsed LDS address in `{rest}`"))?;
                if i > head && lines[head + 1..].iter().any(|(n2, o2, _)| n2.starts_with('v') && o2.first().and_then(|o| dst(o)) == Some(('v', reg))) {
                    return Err(format!("LDS address v{reg} is redefined inside the K loop"));
                }
                let addr = w.v.get(&reg).copied().unwrap_or([None; 32]);
                for (lane, value) in addr.iter().enumerate() {
                    let base = value.ok_or_else(|| format!("{name} {rest}: lane {lane} of wave {wave} has an address the prologue cannot derive"))?;
                    for (offset, width) in ds_ranges(name, rest).ok_or_else(|| format!("unsupported DS instruction {name}"))? {
                        let e = base.checked_add(offset).and_then(|x| x.checked_add(width)).ok_or("LDS address overflow")?;
                        if (base + offset) % width.min(8) != 0 { return Err(format!("{name} {rest}: misaligned LDS access {} in wave {wave} lane {lane}", base + offset)) }
                        if e > limit { return Err(format!("{name} {rest}: wave {wave} lane {lane} reaches byte {e} of {limit}")) }
                        end = end.max(e);
                    }
                }
                checked += 1;
            } else if i < head {
                step(&mut w, name, ops);
            }
        }
    }
    if checked == 0 && limit != 0 { return Err("no LDS access found".into()) }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::lds_bounds;
    const K: &str = "\nk:\n\tv_and_b32_e32 v1, 31, v0\n\tv_lshlrev_b32_e32 v1, 3, v1\n\tv_readfirstlane_b32 s4, v0\n\ts_lshr_b32 s4, s4, 5\n\tv_lshl_add_u32 v2, s4, 8, v1\n.Lloop:\n\tds_load_2addr_b64 v[4:7], v2 offset1:32\n.Lk_end:\n";
    #[test]
    fn every_lane_of_every_wave_is_bounded() {
        // wave w, lane l: 256w + 8l; offset1 32*8 = 256 -> end 256w + 8*31 + 256 + 8.
        assert_eq!(lds_bounds(K, "k", 2, 1024).unwrap(), 256 + 248 + 256 + 8);
        assert!(lds_bounds(K, "k", 4, 1024).is_err());
        let unknown = K.replace("v_lshl_add_u32 v2, s4, 8, v1", "v_lshl_add_u32 v2, s9, 8, v1");
        assert!(lds_bounds(&unknown, "k", 1, 1 << 20).is_err());
        let redefined = K.replace(".Lk_end:", "\tv_add_nc_u32_e32 v2, 8, v2\n.Lk_end:");
        assert!(lds_bounds(&redefined, "k", 1, 1 << 20).is_err());
    }
    /// ds_swizzle_b32 touches no LDS memory: its operand is lane data that
    /// the loop may redefine, and it adds nothing to the access bound.
    #[test]
    fn swizzle_is_not_an_lds_access() {
        let swizzle = K.replace(".Lk_end:", "\tv_cvt_f32_f16_e64 v9, v8.l\n\tds_swizzle_b32 v10, v9 offset:swizzle(BROADCAST,16,3)\n.Lk_end:");
        assert_eq!(lds_bounds(&swizzle, "k", 2, 1024).unwrap(), 256 + 248 + 256 + 8);
    }
    /// An LDS-free kernel (swizzles only) bounds at 0 bytes when it
    /// allocates none; an allocation it never touches is refused.
    #[test]
    fn lds_free_kernel_needs_zero_allocation() {
        let free = K.replace("\tds_load_2addr_b64 v[4:7], v2 offset1:32\n", "\tds_swizzle_b32 v10, v9 offset:swizzle(BROADCAST,16,3)\n");
        assert_eq!(lds_bounds(&free, "k", 2, 0).unwrap(), 0);
        assert!(lds_bounds(&free, "k", 2, 1024).is_err());
    }
    /// A scalar compare writes only SCC: the SGPR it reads stays known. A
    /// register-range write (an SMEM load, a 64-bit select) makes every
    /// register of the range unknown, so an address derived from it fails.
    #[test]
    fn compares_keep_their_operands_and_range_writes_clobber() {
        let compare = K.replace("\tv_lshl_add_u32 v2", "\ts_bitcmp1_b32 s4, 0\n\ts_cmp_eq_u32 s4, 0\n\tv_lshl_add_u32 v2");
        assert_eq!(lds_bounds(&compare, "k", 2, 1024).unwrap(), 256 + 248 + 256 + 8);
        for clobber in ["s_load_b64 s[4:5], s[0:1], 0x0", "s_cselect_b64 s[4:5], s[6:7], s[8:9]"] {
            let text = K.replace("\tv_lshl_add_u32 v2", &format!("\t{clobber}\n\tv_lshl_add_u32 v2"));
            assert!(lds_bounds(&text, "k", 1, 1 << 20).is_err(), "{clobber}");
        }
    }
}
