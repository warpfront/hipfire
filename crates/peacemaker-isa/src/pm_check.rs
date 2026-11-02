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
//! - `lds_bounds`: every DS address is evaluated for every lane of every
//!   wave from the entry path, or independently rederived in its access's
//!   block from the unchanged work-item id. Conditional branches directly
//!   to `s_endpgm` do not bypass any continuing path's definitions. Unknown
//!   inputs are refused unless an immediate mask bounds the address, and
//!   every DS access must end inside the launch allocation.
use crate::toolchain::Result;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
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
///
/// Successful receipts are cached by exact object bytes, architecture, symbol
/// and the running checker's identity. See `cert_cache` for configuration and
/// integrity rules. Cache failures never change certification's outcome.
pub fn m7(elf: &Path, arch: &str, symbol: &str) -> Result<Value> {
    m7_status(elf, arch, symbol).map(|(receipt, _)| receipt)
}

/// How [`m7_status`] obtained its receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum M7Cache {
    /// A stored success receipt for exactly these inputs was returned; M7 did not run.
    Hit,
    /// The cache was consulted, M7 ran and its success receipt was stored.
    Miss,
    /// No cache applies (disabled or no executable identity): M7 ran uncached.
    Disabled,
}

impl M7Cache {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Disabled => "disabled",
        }
    }
}

/// [`m7`] reporting whether the receipt came from the certification cache.
pub fn m7_status(elf: &Path, arch: &str, symbol: &str) -> Result<(Value, M7Cache)> {
    let target = ir_arch(arch)?;
    let bytes = std::fs::read(elf).map_err(|e| format!("{}: {e}", elf.display()))?;
    let cache = crate::cert_cache::Cache::for_request(&bytes, arch, symbol);
    if let Some(receipt) = cache.as_ref().and_then(crate::cert_cache::Cache::load) {
        return Ok((receipt, M7Cache::Hit));
    }
    let receipt = m7_uncached(elf, &bytes, arch, symbol, target)?;
    let status = if cache.is_some() {
        M7Cache::Miss
    } else {
        M7Cache::Disabled
    };
    if let Some(cache) = cache {
        cache.store(&receipt);
    }
    Ok((receipt, status))
}

fn m7_uncached(
    elf: &Path,
    bytes: &[u8],
    arch: &str,
    symbol: &str,
    target: peacemaker_ir::Arch,
) -> Result<Value> {
    use peacemaker_ir::passes::{barriers, windows};
    #[cfg(test)]
    M7_RUNS.with(|runs| runs.set(runs.get() + 1));
    let lifted = peacemaker_lift::lift_object(
        bytes,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(|e| format!("M7 lift rejected {}: {e}", elf.display()))?;
    if lifted.program.target.arch != target {
        return Err(format!(
            "M7 lift read {:?}, expected {arch}",
            lifted.program.target.arch
        ));
    }
    if peacemaker_lift::emit::module(&lifted.program).map_err(|e| e.to_string())? != bytes {
        return Err("M7 emit::module does not reproduce the linked object".into());
    }
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("M7 lift has no kernel {symbol}"))?;
    let instructions = peacemaker_lift::emit::insts(kernel, target)
        .map_err(|e| e.to_string())?
        .len();
    // LLVM FeatureUserSGPRInit16Bug (gfx1100): the descriptor must enable >= 16 user+system SGPRs.
    if let peacemaker_ir::inst::Abi::Hsa { descriptor, .. } = &kernel.abi {
        arch.parse::<crate::arch::Arch>()?
            .check_user_system_sgprs(symbol, descriptor.compute_pgm_rsrc2.0)?;
    }
    let analysis = peacemaker_ir::edit::analyze(lifted.program.clone(), &kernel.symbol)
        .map_err(|e| e.to_string())?;
    let pairings = barriers::analyze(&kernel.body, target).map_err(|e| e.to_string())?;
    let win = windows::check_windows(&kernel.body).map_err(|e| e.to_string())?;
    let mut obligations = BTreeMap::<String, usize>::new();
    let mut details = Vec::new();
    for o in analysis.obligations.iter().chain(&pairings.obligations) {
        *obligations
            .entry(format!("{:?}/{}", o.kind, o.rule_id))
            .or_default() += 1;
        let pcs: Vec<String> = o
            .insts
            .iter()
            .filter_map(|id| kernel.body.insts.get(*id))
            .filter_map(|i| i.prov.pc)
            .map(|pc| format!("{pc:#x}"))
            .collect();
        details.push(format!("{:?}/{} at {pcs:?}: {}", o.kind, o.rule_id, o.text));
    }
    let ambiguous = win.delays.iter().filter(|d| d.is_ambiguous()).count();
    let resources = analysis
        .facts
        .resources
        .first()
        .ok_or("M7 analysis has no resource summary")?;
    let unknown = analysis
        .facts
        .lds
        .accesses
        .iter()
        .filter(|(a, _)| matches!(a.addr, peacemaker_ir::lds::AddrFact::Unknown))
        .count();
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
        return Err(format!(
            "M7 analysis of {symbol} left obligations: {summary}\n{}",
            details.join("\n")
        ));
    }
    Ok(summary)
}

#[cfg(test)]
thread_local! {
    static M7_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Lane values of one wave: `None` is a value the prologue cannot know.
type Lanes = [Option<u32>; 32];

fn parse_imm(t: &str) -> Option<u32> {
    let t = t.trim();
    if let Some(h) = t.strip_prefix("0x") {
        u32::from_str_radix(h, 16).ok()
    } else if let Some(n) = t.strip_prefix('-') {
        n.parse::<u32>().ok().map(|n| n.wrapping_neg())
    } else {
        t.parse().ok()
    }
}

struct Wave {
    v: BTreeMap<u16, Lanes>,
    s: BTreeMap<u16, Option<u32>>,
}
impl Wave {
    fn operand(&self, t: &str) -> Lanes {
        let t = t.trim();
        if let Some(n) = t.strip_prefix('v').and_then(|n| n.parse::<u16>().ok()) {
            return self.v.get(&n).copied().unwrap_or([None; 32]);
        }
        if let Some(n) = t.strip_prefix('s').and_then(|n| n.parse::<u16>().ok()) {
            return [self.s.get(&n).copied().flatten(); 32];
        }
        [parse_imm(t); 32]
    }
    fn scalar(&self, t: &str) -> Option<u32> {
        let t = t.trim();
        if let Some(n) = t.strip_prefix('s').and_then(|n| n.parse::<u16>().ok()) {
            return self.s.get(&n).copied().flatten();
        }
        parse_imm(t)
    }
}

fn dst(first: &str) -> Option<(char, u16)> {
    let t = first.trim();
    let class = t.chars().next()?;
    if !matches!(class, 'v' | 's') {
        return None;
    }
    t[1..].parse().ok().map(|n| (class, n))
}

/// `v[lo:hi]` / `s[lo:hi]`.
fn range_dst(first: &str) -> Option<(char, u16, u16)> {
    let t = first.trim();
    let class = t.chars().next()?;
    if !matches!(class, 'v' | 's') {
        return None;
    }
    let (lo, hi) = t[1..]
        .strip_prefix('[')?
        .strip_suffix(']')?
        .split_once(':')?;
    Some((class, lo.parse().ok()?, hi.parse().ok()?))
}

/// Execute one straight-line instruction for all lanes. Unknown operations
/// make their destination unknown; scalar compares write only SCC, and a
/// register-range destination makes every register of the range unknown.
fn step(w: &mut Wave, name: &str, ops: &[&str]) {
    if name.starts_with("s_cmp") || name.starts_with("s_bitcmp") {
        return;
    }
    if let Some((class, lo, hi)) = ops.first().and_then(|o| range_dst(o)) {
        for n in lo..=hi {
            if class == 'v' {
                w.v.insert(n, [None; 32]);
            } else {
                w.s.insert(n, None);
            }
        }
        return;
    }
    let Some((class, n)) = ops.first().and_then(|o| dst(o)) else {
        return;
    };
    let lanes = |f: &dyn Fn(usize) -> Option<u32>| -> Lanes { std::array::from_fn(f) };
    let a = |i: usize| ops.get(i).map(|o| w.operand(o)).unwrap_or([None; 32]);
    let bin = |f: fn(u32, u32) -> u32| {
        let (x, y) = (a(1), a(2));
        lanes(&|l| Some(f(x[l]?, y[l]?)))
    };
    let tri = |f: fn(u32, u32, u32) -> u32| {
        let (x, y, z) = (a(1), a(2), a(3));
        lanes(&|l| Some(f(x[l]?, y[l]?, z[l]?)))
    };
    let value: Lanes = match name {
        "v_mov_b32_e32" => a(1),
        "v_and_b32_e32" => bin(|x, y| x & y),
        "v_or_b32_e32" => bin(|x, y| x | y),
        "v_xor_b32_e32" => bin(|x, y| x ^ y),
        "v_add_nc_u32_e32" => bin(|x, y| x.wrapping_add(y)),
        "v_sub_nc_u32_e32" => bin(|x, y| x.wrapping_sub(y)),
        "v_lshrrev_b32_e32" => bin(|s, x| x >> (s & 31)),
        "v_lshlrev_b32_e32" => bin(|s, x| x << (s & 31)),
        "v_mul_u32_u24_e32" => bin(|x, y| (x & 0xff_ffff).wrapping_mul(y & 0xff_ffff)),
        "v_lshl_add_u32" => tri(|x, s, y| (x << (s & 31)).wrapping_add(y)),
        "v_mad_u32_u24" => {
            tri(|x, y, z| (x & 0xff_ffff).wrapping_mul(y & 0xff_ffff).wrapping_add(z))
        }
        "v_bfe_u32" => tri(|x, o, w| (x >> (o & 31)) & ((1u32 << (w & 31)) - 1)),
        "v_readfirstlane_b32" => {
            let x = a(1);
            [x[0]; 32]
        }
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
    if class == 'v' {
        w.v.insert(n, value);
    } else {
        w.s.insert(n, value[0]);
    }
}

/// Byte ranges `(offset, width)` a DS instruction touches relative to its address VGPR.
fn ds_ranges(name: &str, ops: &str) -> Option<Vec<(u32, u32)>> {
    let field = |key: &str| {
        ops.split_whitespace()
            .find_map(|t| t.strip_prefix(key))
            .and_then(parse_imm)
            .unwrap_or(0)
    };
    let width = |suffix: &str| match suffix {
        "b16" | "b16_d16_hi" => Some(2),
        "b32" => Some(4),
        "b64" => Some(8),
        "b96" => Some(12),
        "b128" => Some(16),
        _ => None,
    };
    let op = name
        .strip_prefix("ds_load_")
        .or_else(|| name.strip_prefix("ds_store_"))?;
    if let Some(rest) = op.strip_prefix("2addr_stride64_") {
        let w = width(rest)?;
        Some(vec![
            (field("offset0:") * 64 * w, w),
            (field("offset1:") * 64 * w, w),
        ])
    } else if let Some(rest) = op.strip_prefix("2addr_") {
        let w = width(rest)?;
        Some(vec![(field("offset0:") * w, w), (field("offset1:") * w, w)])
    } else {
        Some(vec![(field("offset:"), width(op)?)])
    }
}

/// Every DS access of every lane of `waves` waves of `symbol` ends at or
/// before `limit` bytes. Its address VGPR is either one the entry path (before
/// the first label or branch, except conditional exits directly to `s_endpgm`)
/// derives from the work-item id and which nothing after it redefines, one
/// rederived in the access's block from the unchanged work-item id, or a
/// masked address: in the access's own straight-line block its last definition is
/// `v_and_b32_e32 vR, M, vX` with an immediate `M`, and nothing between them
/// names `vR` except DS accesses addressed by it, so every lane's address is
/// a submask of `M` (at most `M`, aligned to `M`'s trailing zeros). Returns
/// the maximum end. A kernel without DS memory accesses passes only with no
/// allocation (`limit == 0`).
pub fn lds_bounds(source: &str, symbol: &str, waves: u32, limit: u32) -> Result<u32> {
    lds_bounds_host(source, symbol, waves, limit, &[]).map(|(end, _)| end)
}

/// [`lds_bounds`] for a kernel whose `host_bounded` VGPRs address a
/// launch-sized dynamic LDS list (the QSA token list): each of their
/// definitions must be `v_add_nc_u32_e32 vR, sN, vL` with `vL` a lane offset
/// the entry block derives (at most 124 bytes), or `v_mov_b32_e32 vR, sN`,
/// and each access through them spans at most 64 bytes past `vR`. Their
/// scalar base is the kernel's list position, bounded by the launch
/// contract, not by this check. Every other access is bounded as in
/// [`lds_bounds`]. Returns the static maximum end and the number of
/// host-bounded accesses.
pub fn lds_bounds_host(
    source: &str,
    symbol: &str,
    waves: u32,
    limit: u32,
    host_bounded: &[u16],
) -> Result<(u32, usize)> {
    let marker = format!("\n{symbol}:\n");
    let start = source
        .find(&marker)
        .ok_or_else(|| format!("{symbol}: missing kernel label"))?
        + marker.len();
    let body = &source[start - 1..];
    let body = &body[..body
        .find(&format!("\n.L{symbol}_end:"))
        .ok_or("missing kernel end label")?];
    let lines: Vec<(&str, Vec<&str>, &str)> = body
        .lines()
        .filter_map(|l| {
            let t = l.split(';').next()?.trim();
            if t.is_empty() || t.starts_with('.') && !t.ends_with(':') {
                return None;
            }
            let (name, rest) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
            let ops = if name.starts_with("v_dual_") {
                vec![]
            } else {
                rest.split(',').map(str::trim).collect()
            };
            Some((name, ops, rest))
        })
        .collect();
    // An early refusal that only terminates the wave cannot bypass a
    // definition on any path reaching LDS. All other branches still end
    // entry evaluation, including forward branches that rejoin and back-edges.
    let exits: BTreeSet<_> = lines
        .windows(2)
        .filter(|pair| pair[1].0 == "s_endpgm")
        .filter_map(|pair| pair[0].0.strip_suffix(':'))
        .collect();
    let head = lines
        .iter()
        .position(|(n, _, rest)| {
            n.ends_with(':')
                || n.starts_with("s_branch")
                || n.starts_with("s_cbranch") && !exits.contains(rest)
        })
        .ok_or("kernel has no block boundary")?;
    // Conservatively refuse the work-item-id seed if any instruction could
    // overwrite v0, including range and dual destinations.
    let tid_immutable = lines.iter().all(|(name, ops, rest)| {
        if name.starts_with("v_dual_") {
            !names_vgpr(rest, 0)
        } else {
            !ops.first().is_some_and(|o| names_vgpr(o, 0))
        }
    });
    let mut end = 0u32;
    let mut checked = 0usize;
    let mut hosted = 0usize;
    for wave in 0..waves {
        let mut w = Wave {
            v: BTreeMap::new(),
            s: BTreeMap::new(),
        };
        w.v.insert(0, std::array::from_fn(|l| Some(wave * 32 + l as u32)));
        for (i, (name, ops, rest)) in lines.iter().enumerate() {
            // ds_swizzle_b32 / ds_bpermute_b32 exchange lanes on the LDS crossbar and touch no LDS memory.
            if !name.starts_with("ds_") || *name == "ds_swizzle_b32" || *name == "ds_bpermute_b32" {
                if i < head {
                    step(&mut w, name, ops);
                }
                continue;
            }
            let addr_op = if name.starts_with("ds_store") {
                ops.first()
            } else {
                ops.get(1)
            };
            let reg = addr_op
                .map(|o| o.split_whitespace().next().unwrap_or(""))
                .and_then(|o| o.strip_prefix('v'))
                .and_then(|n| n.parse::<u16>().ok())
                .ok_or_else(|| format!("{name}: unparsed LDS address in `{rest}`"))?;
            let ranges = ds_ranges(name, rest)
                .ok_or_else(|| format!("unsupported DS instruction {name}"))?;
            if host_bounded.contains(&reg) {
                if i < head {
                    return Err(format!(
                        "{name} {rest}: host-bounded access in the entry block"
                    ));
                }
                if ranges.iter().any(|&(o, wd)| o + wd > 64) {
                    return Err(format!(
                        "{name} {rest}: host-bounded access spans past 64 bytes"
                    ));
                }
                if wave == 0 {
                    for (n2, o2, r2) in &lines[..] {
                        if !n2.starts_with('v')
                            || o2.first().and_then(|o| dst(o)) != Some(('v', reg))
                        {
                            continue;
                        }
                        let lane_ok = |o: &str| {
                            o.strip_prefix('v')
                                .and_then(|n| n.parse::<u16>().ok())
                                .and_then(|n| w.v.get(&n))
                                .is_some_and(|l| l.iter().all(|x| x.is_some_and(|x| x <= 124)))
                                && lines[head + 1..].iter().all(|(n3, o3, _)| {
                                    !n3.starts_with('v')
                                        || o3.first().and_then(|o| dst(o))
                                            != o.strip_prefix('v')
                                                .and_then(|n| n.parse().ok())
                                                .map(|n| ('v', n))
                                })
                        };
                        let scalar = |o: &str| o.starts_with('s') && o[1..].parse::<u16>().is_ok();
                        let ok = match (*n2, o2.as_slice()) {
                            ("v_add_nc_u32_e32", [_, s, l]) => scalar(s) && lane_ok(l),
                            ("v_mov_b32_e32", [_, s]) => scalar(s),
                            _ => false,
                        };
                        if !ok {
                            return Err(format!(
                                "host-bounded LDS address v{reg} defined by `{n2} {r2}`"
                            ));
                        }
                    }
                    hosted += 1;
                }
                continue;
            }
            let redefined = i > head
                && lines[head + 1..].iter().any(|(n2, o2, _)| {
                    n2.starts_with('v') && o2.first().and_then(|o| dst(o)) == Some(('v', reg))
                });
            let local = redefined
                .then(|| {
                    local_address(
                        &lines,
                        i,
                        reg,
                        if tid_immutable {
                            std::array::from_fn(|l| Some(wave * 32 + l as u32))
                        } else {
                            [None; 32]
                        },
                        &exits,
                    )
                })
                .flatten();
            if redefined && local.is_none() {
                let mask = masked_address(&lines, head, i, reg)
                    .ok_or_else(|| format!("LDS address v{reg} is redefined inside the K loop"))?;
                for &(offset, width) in &ranges {
                    let align = width.min(8);
                    if mask % align != 0 || offset % align != 0 {
                        return Err(format!("{name} {rest}: masked LDS address 0x{mask:x} + {offset} is not {align}-aligned"));
                    }
                    let e = mask
                        .checked_add(offset)
                        .and_then(|x| x.checked_add(width))
                        .ok_or("LDS address overflow")?;
                    if e > limit {
                        return Err(format!(
                            "{name} {rest}: masked address reaches byte {e} of {limit}"
                        ));
                    }
                    end = end.max(e);
                }
                checked += 1;
                continue;
            }
            let addr = local.unwrap_or_else(|| w.v.get(&reg).copied().unwrap_or([None; 32]));
            for (lane, value) in addr.iter().enumerate() {
                let base = value.ok_or_else(|| format!("{name} {rest}: lane {lane} of wave {wave} has an address the prologue cannot derive"))?;
                for &(offset, width) in &ranges {
                    let e = base
                        .checked_add(offset)
                        .and_then(|x| x.checked_add(width))
                        .ok_or("LDS address overflow")?;
                    if (base + offset) % width.min(8) != 0 {
                        return Err(format!(
                            "{name} {rest}: misaligned LDS access {} in wave {wave} lane {lane}",
                            base + offset
                        ));
                    }
                    if e > limit {
                        return Err(format!(
                            "{name} {rest}: wave {wave} lane {lane} reaches byte {e} of {limit}"
                        ));
                    }
                    end = end.max(e);
                }
            }
            checked += 1;
        }
    }
    if checked == 0 && hosted == 0 && limit != 0 {
        return Err("no LDS access found".into());
    }
    Ok((end, hosted))
}

fn names_vgpr(text: &str, reg: u16) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '[' || c == ']' || c == ':'))
        .any(|t| {
            t.strip_prefix('v').and_then(|n| n.parse::<u16>().ok()) == Some(reg)
                || t.strip_prefix("v[")
                    .and_then(|r| r.strip_suffix(']'))
                    .and_then(|r| r.split_once(':'))
                    .and_then(|(a, b)| Some((a.parse::<u16>().ok()?, b.parse::<u16>().ok()?)))
                    .is_some_and(|(a, b)| a <= reg && reg <= b)
        })
}

/// Replay only the access's block, with no values carried across its entry
/// except an immutable work-item id. A fresh local definition may reuse a
/// register that held an unrelated (e.g. global-memory) address in the K loop.
fn local_address(
    lines: &[(&str, Vec<&str>, &str)],
    at: usize,
    reg: u16,
    tid: Lanes,
    exits: &BTreeSet<&str>,
) -> Option<Lanes> {
    let start = lines[..at].iter().rposition(|(name, _, rest)| {
        name.ends_with(':')
            || name.starts_with("s_branch")
            || name.starts_with("s_cbranch") && !exits.contains(rest)
            || name.starts_with("s_setpc")
            || name.starts_with("s_swappc")
    })? + 1;
    let mut w = Wave {
        v: BTreeMap::from([(0, tid)]),
        s: BTreeMap::new(),
    };
    for (name, ops, _) in &lines[start..at] {
        if name.starts_with("v_dual_") {
            w.v.clear();
        } else {
            step(&mut w, name, ops);
        }
    }
    let addr = w.v.get(&reg).copied()?;
    addr.iter().all(Option::is_some).then_some(addr)
}

/// The immediate mask `M` of the `v_and_b32_e32 vR, M, vX` that is the last
/// definition of `vR` before line `at`, in the same straight-line block
/// (after the entry block), when nothing between them names `vR` except DS
/// accesses that only read it as their address.
fn masked_address(
    lines: &[(&str, Vec<&str>, &str)],
    head: usize,
    at: usize,
    reg: u16,
) -> Option<u32> {
    let names = |text: &str| names_vgpr(text, reg);
    for j in (head + 1..at).rev() {
        let (name, ops, rest) = &lines[j];
        if name.ends_with(':')
            || name.starts_with("s_branch")
            || name.starts_with("s_cbranch")
            || name.starts_with("s_setpc")
            || name.starts_with("s_swappc")
        {
            return None;
        }
        if *name == "v_and_b32_e32" && ops.len() == 3 && dst(ops[0]) == Some(('v', reg)) {
            return parse_imm(ops[1])
                .filter(|_| ops[2].starts_with('v') && ops[2][1..].parse::<u16>().is_ok());
        }
        if !names(rest) {
            continue;
        }
        let address = if name.starts_with("ds_store") {
            ops.first()
        } else if name.starts_with("ds_load") {
            ops.get(1)
        } else {
            None
        };
        let writes = name.starts_with("ds_load") && ops.first().is_some_and(|o| names(o));
        if address.and_then(|o| dst(o.split_whitespace().next().unwrap_or(""))) != Some(('v', reg))
            || writes
        {
            return None;
        }
        // A DS store naming vR only as its address (not as data) reads it.
        if name.starts_with("ds_store") && ops[1..].iter().any(|o| names(o)) {
            return None;
        }
    }
    None
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
        let unknown = K.replace(
            "v_lshl_add_u32 v2, s4, 8, v1",
            "v_lshl_add_u32 v2, s9, 8, v1",
        );
        assert!(lds_bounds(&unknown, "k", 1, 1 << 20).is_err());
        let redefined = K.replace(".Lk_end:", "\tv_add_nc_u32_e32 v2, 8, v2\n.Lk_end:");
        assert!(lds_bounds(&redefined, "k", 1, 1 << 20).is_err());
    }
    /// Conditional refusals may precede the address setup: only their
    /// fallthrough reaches LDS, so they do not end entry evaluation.
    #[test]
    fn early_exits_preserve_entry_address_proof() {
        let guarded = K
            .replace(
                "\tv_and_b32_e32 v1",
                "\ts_cmp_eq_u32 s8, 0\n\ts_cbranch_scc1 .Lexit\n\tv_and_b32_e32 v1",
            )
            .replace(
                "\tv_readfirstlane_b32 s4",
                "\ts_cmp_eq_u32 s9, 0\n\ts_cbranch_scc0 .Lexit\n\tv_readfirstlane_b32 s4",
            )
            .replace(".Lk_end:", ".Lexit:\n\ts_endpgm\n.Lk_end:");
        assert_eq!(
            lds_bounds(&guarded, "k", 2, 1024).unwrap(),
            256 + 248 + 256 + 8
        );
        assert!(lds_bounds(&guarded, "k", 4, 1024).is_err());
        // A rejoining branch may skip setup; a nonterminal target may still
        // access LDS. Neither is an early exit. Real loop redefinitions,
        // unknown inputs and out-of-bounds addresses must still fail.
        for bad in [
            guarded.replace("s_cbranch_scc1 .Lexit", "s_cbranch_scc1 .Lloop"),
            guarded.replace(
                ".Lexit:\n\ts_endpgm",
                ".Lexit:\n\tds_load_b32 v9, v2\n\ts_endpgm",
            ),
            guarded.replace(
                ".Lexit:",
                "\tv_add_nc_u32_e32 v2, 8, v2\n\ts_branch .Lloop\n.Lexit:",
            ),
            guarded.replace(
                "v_lshl_add_u32 v2, s4, 8, v1",
                "v_lshl_add_u32 v2, s9, 8, v1",
            ),
            guarded.replace(
                "v_lshl_add_u32 v2, s4, 8, v1",
                "v_lshl_add_u32 v2, s4, 8, 0x100000",
            ),
        ] {
            assert!(lds_bounds(&bad, "k", 2, 1024).is_err(), "{bad}");
        }
    }
    /// A reused register can be independently rederived in a later block,
    /// as in the HC partial-read epilogue. No unknown loop state is carried.
    #[test]
    fn local_addresses_are_rederived_from_immutable_thread_ids() {
        let local = K.replace(
            ".Lk_end:",
            "\tv_add_nc_u32_e32 v9, s9, v1\n.Lepilogue:\n\tv_lshlrev_b32_e32 v9, 5, v0\n\ts_cbranch_execz .Lexit\n\tds_load_b128 v[16:19], v9\n\tds_load_b128 v[20:23], v9 offset:16\n.Lexit:\n\ts_endpgm\n.Lk_end:",
        );
        assert_eq!(lds_bounds(&local, "k", 2, 2048).unwrap(), 2048);
        assert!(lds_bounds(&local, "k", 2, 2047).is_err());
        for bad in [
            local.replace("v9, 5, v0", "v9, 5, v8"),
            local.replace("v9, 5, v0", "v9, 5, v9"),
            local.replace("v9, 5, v0", "v9, 5, v0\n\tv_add_nc_u32_e32 v9, s9, v9"),
            local.replace("v9, 5, v0", "v9, 5, v0\n.Ljoin:"),
            local.replace("v9, 5, v0", "v9, 5, v0\n\ts_cbranch_scc1 .Lloop"),
            local.replace(".Lepilogue:", "\tv_mov_b32_e32 v0, s9\n.Lepilogue:"),
            local.replace(".Lepilogue:", "\tds_load_b64 v[0:1], v2\n.Lepilogue:"),
            local.replace(
                ".Lepilogue:",
                "\tv_dual_mov_b32 v0, v8 :: v_dual_mov_b32 v9, v8\n.Lepilogue:",
            ),
            local.replace(
                "\tds_load_b128 v[16:19], v9",
                "\tv_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v10, v8\n\tds_load_b128 v[16:19], v9",
            ),
            local.replace("v[16:19], v9", "v[8:11], v9"),
        ] {
            assert!(lds_bounds(&bad, "k", 2, 2048).is_err(), "{bad}");
        }
    }
    /// ds_swizzle_b32 touches no LDS memory: its operand is lane data that
    /// the loop may redefine, and it adds nothing to the access bound.
    #[test]
    fn swizzle_is_not_an_lds_access() {
        let swizzle = K.replace(".Lk_end:", "\tv_cvt_f32_f16_e64 v9, v8.l\n\tds_swizzle_b32 v10, v9 offset:swizzle(BROADCAST,16,3)\n.Lk_end:");
        assert_eq!(
            lds_bounds(&swizzle, "k", 2, 1024).unwrap(),
            256 + 248 + 256 + 8
        );
    }
    /// An LDS-free kernel (swizzles only) bounds at 0 bytes when it
    /// allocates none; an allocation it never touches is refused.
    #[test]
    fn lds_free_kernel_needs_zero_allocation() {
        let free = K.replace(
            "\tds_load_2addr_b64 v[4:7], v2 offset1:32\n",
            "\tds_swizzle_b32 v10, v9 offset:swizzle(BROADCAST,16,3)\n",
        );
        assert_eq!(lds_bounds(&free, "k", 2, 0).unwrap(), 0);
        assert!(lds_bounds(&free, "k", 2, 1024).is_err());
    }
    /// ds_bpermute_b32's address operand is a lane-byte index, not an LDS
    /// address: an LDS-free bpermute kernel bounds at 0 bytes.
    #[test]
    fn bpermute_is_not_an_lds_access() {
        let free = K.replace(
            "\tds_load_2addr_b64 v[4:7], v2 offset1:32\n",
            "\tds_bpermute_b32 v10, v11, v9\n",
        );
        assert_eq!(lds_bounds(&free, "k", 2, 0).unwrap(), 0);
        assert!(lds_bounds(&free, "k", 2, 1024).is_err());
    }
    /// A scalar compare writes only SCC: the SGPR it reads stays known. A
    /// register-range write (an SMEM load, a 64-bit select) makes every
    /// register of the range unknown, so an address derived from it fails.
    #[test]
    fn compares_keep_their_operands_and_range_writes_clobber() {
        let compare = K.replace(
            "\tv_lshl_add_u32 v2",
            "\ts_bitcmp1_b32 s4, 0\n\ts_cmp_eq_u32 s4, 0\n\tv_lshl_add_u32 v2",
        );
        assert_eq!(
            lds_bounds(&compare, "k", 2, 1024).unwrap(),
            256 + 248 + 256 + 8
        );
        for clobber in [
            "s_load_b64 s[4:5], s[0:1], 0x0",
            "s_cselect_b64 s[4:5], s[6:7], s[8:9]",
        ] {
            let text = K.replace(
                "\tv_lshl_add_u32 v2",
                &format!("\t{clobber}\n\tv_lshl_add_u32 v2"),
            );
            assert!(lds_bounds(&text, "k", 1, 1 << 20).is_err(), "{clobber}");
        }
    }
    /// A loop-defined address is bounded by its immediate `v_and_b32` mask
    /// only when that mask is its last definition in the access's block.
    #[test]
    fn masked_loop_addresses_are_bounded_by_their_mask() {
        let base = K.replace(".Lk_end:", "\tv_xor_b32_e32 v3, 0x100, v2\n\tv_and_b32_e32 v3, 0x7f8, v3\n\tds_load_b64 v[8:9], v3 offset:2048\n\tds_store_b64 v3, v[10:11] offset:2048\n.Lk_end:");
        assert_eq!(lds_bounds(&base, "k", 2, 4096).unwrap(), 0x7f8 + 2048 + 8);
        assert!(lds_bounds(&base, "k", 2, 4096 - 1024).is_err());
        // Unmasked, written after the mask, across a label, misaligned mask, overwritten by a load, or by a dual op.
        for bad in [
            base.replace("\tv_and_b32_e32 v3, 0x7f8, v3\n", ""),
            base.replace(
                "\tds_load_b64 v[8:9], v3",
                "\tv_add_nc_u32_e32 v3, 8, v3\n\tds_load_b64 v[8:9], v3",
            ),
            base.replace(
                "\tds_load_b64 v[8:9], v3",
                ".Lmid:\n\tds_load_b64 v[8:9], v3",
            ),
            base.replace("0x7f8", "0x7fc"),
            base.replace("ds_load_b64 v[8:9], v3", "ds_load_b64 v[2:3], v3"),
            base.replace(
                "\tds_store_b64 v3, v[10:11]",
                "\tv_dual_mov_b32 v3, v4 :: v_dual_mov_b32 v5, v6\n\tds_store_b64 v3, v[10:11]",
            ),
        ] {
            assert!(lds_bounds(&bad, "k", 2, 1 << 20).is_err(), "{bad}")
        }
    }
}

// Environment configuration is exercised in fresh test subprocesses rather
// than mutating process-global variables while other tests are running.
#[cfg(all(test, target_os = "linux"))]
mod cache_tests {
    use super::{m7, M7_RUNS};
    use std::path::{Path, PathBuf};

    fn fixture() -> (Vec<u8>, &'static str) {
        use crate::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};
        use peacemaker_author::{Gfx1151, Workgroup};
        let mut regs = RegPlan::new(16, 8).unwrap();
        regs.s::<4>("abi", 0, crate::reg::Live::Whole).unwrap();
        regs.v::<1>("tid", 0, crate::reg::Live::Whole).unwrap();
        let mut builder = Builder::new(
            KernelSpec {
                kernel_id: "probe".into(),
                variant: "default".into(),
                arch: Arch::Gfx1151,
                symbol: "probe".into(),
                kernargs: KernargLayout::new(8),
                user_sgpr_count: 2,
                system_sgpr_workgroup_id_y: false,
                workgroup_size: 64,
                group_segment_fixed_size: 0,
                wave32: true,
                cu_mode: false,
            },
            regs,
        );
        let mut workgroup = Workgroup::<Gfx1151, Builder>::new(&mut builder).unwrap();
        let exit = workgroup.exit(".Lend").unwrap();
        workgroup.end(exit).unwrap();
        let emitted = builder.finish().unwrap();
        (
            crate::native::assemble(&emitted.s_text, Arch::Gfx1151).unwrap(),
            "probe",
        )
    }

    fn isolated(case: &str) {
        let root = std::env::temp_dir().join(format!(
            "pm-m7-{}-{}-{case}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = root.join("cache");
        if case == "unwritable" {
            std::fs::write(&cache, b"not a directory").unwrap();
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "pm_check::cache_tests::cache_child",
                "--ignored",
                "--nocapture",
            ])
            .env("HIPFIRE_M7_TEST_CASE", case)
            .env("HIPFIRE_M7_TEST_ROOT", &root)
            .env(
                "HIPFIRE_CERT_CACHE",
                if case == "disabled" {
                    Path::new("0")
                } else {
                    &cache
                },
            )
            .output()
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        assert!(
            output.status.success(),
            "child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if case == "unwritable" {
            assert_eq!(
                String::from_utf8_lossy(&output.stderr)
                    .matches("warning: M7 certification cache")
                    .count(),
                1
            );
        }
    }

    #[test]
    fn repeated_receipt_is_byte_identical_and_hits() {
        isolated("repeat");
    }

    #[test]
    fn changed_elf_bytes_miss() {
        isolated("bytes");
    }

    #[test]
    fn changed_architecture_misses() {
        isolated("arch");
    }

    #[test]
    fn changed_symbol_misses() {
        isolated("symbol");
    }

    #[test]
    fn corrupt_entry_recomputes() {
        isolated("corrupt");
    }

    #[test]
    fn disabled_cache_bypasses() {
        isolated("disabled");
    }

    #[test]
    fn unwritable_cache_warns_once_and_certifies() {
        isolated("unwritable");
    }

    #[test]
    #[ignore = "invoked in an isolated environment by the parent tests"]
    fn cache_child() {
        let case = std::env::var("HIPFIRE_M7_TEST_CASE").unwrap();
        let root = PathBuf::from(std::env::var_os("HIPFIRE_M7_TEST_ROOT").unwrap());
        let elf = root.join("fixture.co");
        let (bytes, symbol) = fixture();
        std::fs::write(&elf, &bytes).unwrap();
        let first = m7(&elf, "gfx1151", symbol).unwrap();
        assert_eq!(M7_RUNS.with(|runs| runs.get()), 1);
        match case.as_str() {
            "repeat" | "disabled" | "unwritable" => {
                let second = m7(&elf, "gfx1151", symbol).unwrap();
                assert_eq!(
                    serde_json::to_vec(&first).unwrap(),
                    serde_json::to_vec(&second).unwrap()
                );
                if case == "disabled" {
                    assert!(!root.join("cache").exists());
                }
            }
            "bytes" => {
                let mut changed = bytes.clone();
                changed[0] ^= 0xff;
                std::fs::write(&elf, changed).unwrap();
                assert!(m7(&elf, "gfx1151", symbol).is_err());
            }
            "arch" => assert!(m7(&elf, "gfx1100", symbol).is_err()),
            "symbol" => assert!(m7(&elf, "gfx1151", "missing_kernel").is_err()),
            "corrupt" => {
                let cache =
                    crate::cert_cache::Cache::for_request(&bytes, "gfx1151", symbol).unwrap();
                std::fs::write(cache.path(), b"{truncated").unwrap();
                let second = m7(&elf, "gfx1151", symbol).unwrap();
                assert_eq!(
                    serde_json::to_vec(&first).unwrap(),
                    serde_json::to_vec(&second).unwrap()
                );
                assert_eq!(cache.load(), Some(first));
            }
            _ => panic!("unknown isolated test case {case}"),
        }
        assert_eq!(
            M7_RUNS.with(|runs| runs.get()),
            if case == "repeat" { 1 } else { 2 }
        );
    }
}
