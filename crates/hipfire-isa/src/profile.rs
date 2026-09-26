// SPDX-License-Identifier: Apache-2.0
//! DIAGNOSTIC ONLY: `peacemaker profile` in-kernel timestamp instrumentation.
//!
//! The pass rewrites one wave32 gfx12 kernel in an assembly module (builder
//! output or a hipcc `-save-temps` `.s`) into `<symbol>__pm_profile`, which
//! stores a per-wave record at every named point. The profiled symbol is a
//! different kernel with a longer kernarg segment; it must never be embedded
//! or selected by a runtime.
//!
//! Clocks. gfx1201 has no `s_memtime`/`s_memrealtime` (llvm-mc rejects both).
//! Every point reads `SHADER_CYCLES_LO` with `s_getreg_b32`, an SALU read with
//! no memory counter. Entry and exit also read the 100 MHz constant clock with
//! `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, which returns through KMcnt, so it
//! gets its own `s_wait_kmcnt 0` before its SGPRs reach the store.
//!
//! Kernarg extension, at `ext = align8(original kernarg size)`: `u64 base`,
//! `u32 slot_bytes`, `u32 waves_per_wg`, `u32 grid_x`, `u32 grid_y` (in
//! workgroups). Wave `w = (x + grid_x*(y + grid_y*z))*waves_per_wg + tid.x/32`
//! (1D workgroups) owns `base + w*slot_bytes`:
//! - header, 32 B: magic, HW_ID1, HW_ID2, linear WG, wave in WG, realtime
//!   lo/hi at entry, SHADER_CYCLES_HI at entry;
//! - point record, 8 B: SHADER_CYCLES_LO, site id;
//! - exit record, 16 B: SHADER_CYCLES_LO, site id | `EXIT_FLAG`, realtime
//!   lo/hi. The host pre-fills slots with 0xff and stops at an id of !0.
//!
//! Isolation. The pass uses only registers the original kernel never
//! references: one VGPR below the occupancy ceiling (the gfx1201 wave32
//! allocation granule is 24 VGPRs, so the ceiling defaults to
//! `next_free_vgpr` rounded up to 24) and four SGPRs (record pointer pair,
//! timestamp, EXEC save). Entry code may also clobber SGPRs that are not
//! kernel inputs, because the original program has not run yet; the exit
//! record sits before `s_endpgm` (above a VGPR-dealloc message), where every register is dead.
//! SCC, VCC and M0 are never written after entry; EXEC is saved and restored
//! around each store. `s_add_nc_u64` advances the pointer without SCC.
//!
//! Counters and hazards. The stores count in STORECNT only, so every original
//! `s_wait_storecnt` must be 0 (checked). `LOADcnt`/`DScnt` are untouched and
//! the only KMcnt producers are drained by their own wait. An SGPR written by
//! SALU is read by `v_writelane` only after `s_wait_alu depctr_sa_sdst(0)`;
//! `depctr_vm_vsrc(0)` also guarantees the previous record store has read the
//! scratch VGPR before it is rewritten. `depctr_va_sdst(0)` precedes the EXEC
//! read when the kernel has a VALU EXEC writer (`v_cmpx`). No point splits an
//! `s_clause` window or an `s_delay_alu` from its target(s): `before` anchors
//! move up to the window start, `after`/label anchors down past it. Each
//! point adds two VALU instructions, so an original `VALU_DEP_n` hint whose
//! producer precedes a point is rewritten to `VALU_DEP_{n+2k}` (or `NO_DEP`
//! beyond 4). Hints never change results; they are kept exact.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PROFILED_SUFFIX: &str = "__pm_profile";
pub const MAGIC: u32 = 0x504d_5046;
pub const HEADER_BYTES: u32 = 32;
pub const RECORD_BYTES: u32 = 8;
pub const EXIT_RECORD_BYTES: u32 = 16;
pub const EXIT_FLAG: u32 = 0x8000_0000;
pub const KERNARG_EXT_BYTES: u32 = 24;
pub const BEGIN: &str = "; @pm-profile begin";
pub const END: &str = "; @pm-profile end";
const SGPR_LIMIT: u16 = 106;
const VGPR_GRANULE: u16 = 24;
const POINT_VALU: usize = 2;

/// One named point family. Exactly one of `label`, `before`, `after`.
/// `before`/`after` match instructions whose text starts with the prefix;
/// `first_after` keeps only the first match after each match of that prefix;
/// `occurrences` keeps the listed 0-based matches.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rule {
    pub name: String,
    #[serde(default)] pub label: Option<String>,
    #[serde(default)] pub before: Option<String>,
    #[serde(default)] pub after: Option<String>,
    #[serde(default)] pub first_after: Option<String>,
    #[serde(default)] pub occurrences: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub symbol: String,
    #[serde(default)] pub vgpr_limit: Option<u16>,
    pub rules: Vec<Rule>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Site { pub id: u32, pub rule: String, pub line: usize, pub anchor: String, pub moved: bool }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DelayFixup { pub line: usize, pub before: String, pub after: String }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Registers {
    pub scratch_vgpr: u16, pub pointer: u16, pub timestamp: u16, pub exec_save: u16,
    pub entry_quad: u16, pub entry_pair: u16, pub realtime_pair: u16, pub entry_singles: Vec<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Map {
    pub diagnostic: String,
    pub symbol: String,
    pub profiled_symbol: String,
    pub kernarg_size: u32,
    pub kernarg_ext_offset: u32,
    pub kernarg_ext_bytes: u32,
    pub header_bytes: u32,
    pub record_bytes: u32,
    pub exit_record_bytes: u32,
    pub exit_flag: u32,
    pub magic: u32,
    pub vgpr_before: u16, pub vgpr_after: u16, pub vgpr_limit: u16,
    pub sgpr_before: u16, pub sgpr_after: u16,
    pub registers: Registers,
    pub valu_exec_writer: bool,
    pub sites: Vec<Site>,
    pub delay_fixups: Vec<DelayFixup>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind { Insn, Label, Other }

fn code(raw: &str) -> &str { raw.split(';').next().unwrap_or("").trim() }
fn kind(raw: &str) -> Kind {
    let c = code(raw);
    if c.is_empty() { Kind::Other } else if c.ends_with(':') && !c.contains(char::is_whitespace) { Kind::Label }
    else if c.starts_with('.') { Kind::Other } else { Kind::Insn }
}
fn mnemonic(raw: &str) -> &str { code(raw).split_whitespace().next().unwrap_or("") }

/// VGPR (`class` 'v') or SGPR ('s') numbers named in one line of code.
pub fn registers(text: &str, class: u8) -> BTreeSet<u16> {
    let b = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != class || (i > 0 && (b[i-1].is_ascii_alphanumeric() || b[i-1] == b'_')) { i += 1; continue; }
        i += 1;
        let bracket = b.get(i) == Some(&b'[');
        if bracket { i += 1; }
        let start = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) { i += 1; }
        let Ok(first) = text[start..i].parse::<u16>() else { continue };
        let mut last = first;
        if bracket && b.get(i) == Some(&b':') {
            i += 1;
            let start = i;
            while b.get(i).is_some_and(u8::is_ascii_digit) { i += 1; }
            last = text[start..i].parse().unwrap_or(first);
        }
        if bracket && b.get(i) != Some(&b']') { continue; }
        out.extend(first..=last);
    }
    out
}

struct Descriptor { kernarg: u32, vgpr: u16, sgpr: u16, inputs: u16 }

fn descriptor(lines: &[&str], symbol: &str) -> Result<(usize, usize, Descriptor), String> {
    let start = lines.iter().position(|l| code(l) == format!(".amdhsa_kernel {symbol}"))
        .ok_or("missing .amdhsa_kernel block")?;
    let end = start + lines[start..].iter().position(|l| code(l) == ".end_amdhsa_kernel")
        .ok_or("unterminated .amdhsa_kernel block")?;
    let get = |key: &str| -> Option<u32> {
        lines[start..end].iter().find_map(|l| {
            let mut t = code(l).split_whitespace();
            (t.next() == Some(key)).then(|| t.next().and_then(|v| v.parse().ok())).flatten()
        })
    };
    if get(".amdhsa_wavefront_size32") != Some(1) { return Err("profile supports wave32 kernels only".into()) }
    let sys = [".amdhsa_system_sgpr_workgroup_id_x", ".amdhsa_system_sgpr_workgroup_id_y",
        ".amdhsa_system_sgpr_workgroup_id_z", ".amdhsa_system_sgpr_workgroup_info",
        ".amdhsa_enable_private_segment"].iter().map(|k| get(k).unwrap_or(0)).sum::<u32>();
    let d = Descriptor {
        kernarg: get(".amdhsa_kernarg_size").ok_or("missing .amdhsa_kernarg_size")?,
        vgpr: get(".amdhsa_next_free_vgpr").ok_or("missing literal .amdhsa_next_free_vgpr")? as u16,
        sgpr: get(".amdhsa_next_free_sgpr").ok_or("missing literal .amdhsa_next_free_sgpr")? as u16,
        inputs: (get(".amdhsa_user_sgpr_count").ok_or("missing .amdhsa_user_sgpr_count")? + sys) as u16,
    };
    Ok((start, end, d))
}

/// Forbidden insertion gaps: inside an `s_clause` window or between an
/// `s_delay_alu` and its last target. Returned as (window start line, last line).
fn windows(lines: &[&str], body: std::ops::Range<usize>) -> Result<Vec<(usize, usize)>, String> {
    let insns = |from: usize| (from..body.end).filter(|&j| kind(lines[j]) == Kind::Insn);
    let mut out = Vec::new();
    for i in body.clone() {
        if kind(lines[i]) != Kind::Insn { continue }
        let c = code(lines[i]);
        let last = if let Some(n) = c.strip_prefix("s_clause ") {
            let n = parse_imm(n.trim())? as usize;
            insns(i + 1).nth(n).ok_or("s_clause runs past the kernel")?
        } else if c.starts_with("s_delay_alu") {
            let skip = delay_skip(c);
            let t1 = insns(i + 1).next().ok_or("s_delay_alu without target")?;
            // Count the skip without other hints so the window is the larger reading.
            let mut t = t1;
            let mut left = skip;
            let mut j = t1 + 1;
            while left > 0 && j < body.end {
                if kind(lines[j]) == Kind::Insn && !mnemonic(lines[j]).starts_with("s_delay_alu") { t = j; left -= 1; }
                j += 1;
            }
            t
        } else { continue };
        if (i..=last).any(|j| kind(lines[j]) == Kind::Label) {
            return Err(format!("line {}: scheduling window crosses a label", i + 1))
        }
        out.push((i, last));
    }
    Ok(out)
}

fn parse_imm(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x") { u32::from_str_radix(h, 16) } else { s.parse() }
        .map_err(|_| format!("bad immediate {s}"))
}

#[derive(Clone, Debug, PartialEq)]
struct Delay { id0: String, skip: usize, id1: Option<String> }
fn delay_fields(c: &str) -> Delay {
    let mut d = Delay { id0: "NO_DEP".into(), skip: 0, id1: None };
    for part in c.trim_start_matches("s_delay_alu").split('|') {
        let p = part.trim();
        let inner = |p: &str| p.split_once('(').map(|(_, r)| r.trim_end_matches(')').to_string());
        if p.starts_with("instid0") { d.id0 = inner(p).unwrap_or_default() }
        else if p.starts_with("instid1") { d.id1 = inner(p) }
        else if p.starts_with("instskip") {
            let v = inner(p).unwrap_or_default();
            d.skip = if v == "SAME" { 0 } else if v == "NEXT" { 1 }
                else { v.trim_start_matches("SKIP_").parse::<usize>().map(|n| n + 1).unwrap_or(0) };
        }
    }
    d
}
fn delay_skip(c: &str) -> usize { let d = delay_fields(c); if d.id1.is_some() { d.skip } else { 0 } }
/// Canonical (disassembler) spelling: NO_DEP fields are omitted, an empty
/// hint is `s_delay_alu 0`.
fn delay_text(d: &Delay) -> String {
    let mut parts = Vec::new();
    if d.id0 != "NO_DEP" { parts.push(format!("instid0({})", d.id0)) }
    if let Some(id1) = d.id1.as_ref().filter(|id| *id != "NO_DEP") {
        let skip = match d.skip { 0 => "SAME".to_string(), 1 => "NEXT".to_string(), n => format!("SKIP_{}", n - 1) };
        parts.push(format!("instskip({skip})"));
        parts.push(format!("instid1({id1})"));
    }
    if parts.is_empty() { "s_delay_alu 0".into() } else { format!("s_delay_alu {}", parts.join(" | ")) }
}

struct Gap { at: usize, rule: String, anchor: String, moved: bool }

fn resolve(lines: &[&str], body: std::ops::Range<usize>, rule: &Rule, win: &[(usize, usize)]) -> Result<Vec<Gap>, String> {
    let insn = |i: usize| kind(lines[i]) == Kind::Insn;
    let (prefix, after) = match (&rule.label, &rule.before, &rule.after) {
        (Some(l), None, None) => {
            let target = format!("{l}:");
            let i = body.clone().find(|&i| code(lines[i]) == target)
                .ok_or_else(|| format!("rule {}: label {l} not in kernel", rule.name))?;
            return Ok(vec![settle(i + 1, true, rule, win, &target)]);
        }
        (None, Some(p), None) => (p.as_str(), false),
        (None, None, Some(p)) => (p.as_str(), true),
        _ => return Err(format!("rule {}: give exactly one of label/before/after", rule.name)),
    };
    let mut hits: Vec<usize> = body.clone().filter(|&i| insn(i) && code(lines[i]).starts_with(prefix)).collect();
    if let Some(f) = &rule.first_after {
        let marks: Vec<usize> = body.clone().filter(|&i| insn(i) && code(lines[i]).starts_with(f.as_str())).collect();
        hits = marks.iter().enumerate().filter_map(|(k, &m)| {
            let limit = marks.get(k + 1).copied().unwrap_or(body.end);
            hits.iter().copied().find(|&h| h > m && h < limit)
        }).collect();
    }
    if let Some(keep) = &rule.occurrences {
        if keep.iter().any(|&k| k >= hits.len()) { return Err(format!("rule {}: occurrence out of range ({} matches)", rule.name, hits.len())) }
        hits = keep.iter().map(|&k| hits[k]).collect();
    }
    if hits.is_empty() { return Err(format!("rule {}: no instruction starts with {prefix:?}", rule.name)) }
    Ok(hits.into_iter().map(|h| settle(if after { h + 1 } else { h }, after, rule, win, code(lines[h]))).collect())
}

/// Move a gap out of any window: up to the window start for `before`, down
/// past its last line for `after` anchors.
fn settle(mut at: usize, down: bool, rule: &Rule, win: &[(usize, usize)], anchor: &str) -> Gap {
    let mut moved = false;
    while let Some(&(s, e)) = win.iter().find(|&&(s, e)| at > s && at <= e) {
        at = if down { e + 1 } else { s };
        moved = true;
    }
    Gap { at, rule: rule.name.clone(), anchor: anchor.to_string(), moved }
}

struct Regs { v: u16, p: u16, t: u16, e: u16, q: u16, r: u16, rt: u16, singles: Vec<u16> }

fn pick(dsc: &Descriptor, vgpr_limit: u16, used_v: &BTreeSet<u16>, used_s: &BTreeSet<u16>) -> Result<Regs, String> {
    let v = (0..vgpr_limit).rev().find(|n| !used_v.contains(n))
        .ok_or_else(|| format!("no unreferenced VGPR below the occupancy ceiling v{vgpr_limit}"))?;
    let mut taken: BTreeSet<u16> = (0..dsc.inputs).collect();
    let free = |taken: &BTreeSet<u16>, n: u16, align: u16, unused_only: bool| (0..SGPR_LIMIT).step_by(align as usize)
        .find(|&b| b + n <= SGPR_LIMIT && (b..b + n).all(|r| !taken.contains(&r) && (!unused_only || !used_s.contains(&r))));
    let take = |taken: &mut BTreeSet<u16>, n: u16, align: u16, unused_only: bool| -> Result<u16, String> {
        let b = free(taken, n, align, unused_only).or_else(|| if unused_only { None } else { free(taken, n, align, true) })
            .ok_or("not enough SGPRs for the profile record")?;
        taken.extend(b..b + n);
        Ok(b)
    };
    let p = take(&mut taken, 2, 2, true)?;
    let t = take(&mut taken, 1, 1, true)?;
    let e = take(&mut taken, 1, 1, true)?;
    // Entry-only temporaries may reuse registers the original writes later.
    let q = take(&mut taken, 4, 4, false)?;
    let r = take(&mut taken, 2, 2, false)?;
    let rt = take(&mut taken, 2, 2, false)?;
    let singles = (0..6).map(|_| take(&mut taken, 1, 1, false)).collect::<Result<Vec<_>, _>>()?;
    Ok(Regs { v, p, t, e, q, r, rt, singles })
}

/// Immediates as the disassembler prints them: inline constants in decimal,
/// literals in hex, so parse-back compares text exactly.
fn imm(v: u32) -> String { if v <= 64 { v.to_string() } else { format!("{v:#x}") } }

fn store(r: &Regs, lanes: u32, bytes: u32, va_sdst: bool) -> Vec<String> {
    let (p, p1) = (r.p, r.p + 1);
    let mut s = Vec::new();
    if va_sdst { s.push("s_wait_alu depctr_va_sdst(0)".into()) }
    s.push(format!("s_mov_b32 s{}, exec_lo", r.e));
    s.push(format!("s_mov_b32 exec_lo, {}", imm((1u32 << lanes) - 1)));
    s.push(format!("s_add_nc_u64 s[{p}:{p1}], s[{p}:{p1}], {bytes}"));
    s.push(format!("global_store_addtid_b32 v{}, s[{p}:{p1}] offset:-{bytes} th:TH_STORE_NT", r.v));
    s.push(format!("s_mov_b32 exec_lo, s{}", r.e));
    s
}

fn point(r: &Regs, id: u32, name: &str, va_sdst: bool) -> Vec<String> {
    let mut s = vec![format!("{BEGIN} site {id} {name}"),
        format!("s_getreg_b32 s{}, hwreg(HW_REG_SHADER_CYCLES_LO)", r.t),
        "s_wait_alu depctr_sa_sdst(0) depctr_vm_vsrc(0)".into(),
        format!("v_writelane_b32 v{}, s{}, 0", r.v, r.t),
        format!("v_writelane_b32 v{}, {}, 1", r.v, imm(id))];
    s.extend(store(r, 2, RECORD_BYTES, va_sdst));
    s.push(END.into());
    s
}

fn entry(r: &Regs, ext: u32, va_sdst: bool) -> Vec<String> {
    let [w, h1, h2, ch, z, l] = [r.singles[0], r.singles[1], r.singles[2], r.singles[3], r.singles[4], r.singles[5]];
    let (q, rr, rt, p) = (r.q, r.r, r.rt, r.p);
    let mut s = vec![format!("{BEGIN} entry header (DIAGNOSTIC: never embed)"),
        format!("s_load_b128 s[{q}:{}], s[0:1], {ext:#x}", q + 3),
        format!("s_load_b64 s[{rr}:{}], s[0:1], {:#x}", rr + 1, ext + 16),
        format!("s_getreg_b32 s{h1}, hwreg(HW_REG_WAVE_HW_ID1)"),
        format!("s_getreg_b32 s{h2}, hwreg(HW_REG_WAVE_HW_ID2)"),
        format!("s_getreg_b32 s{ch}, hwreg(HW_REG_SHADER_CYCLES_HI)"),
        format!("v_readfirstlane_b32 s{w}, v0"),
        format!("s_sendmsg_rtn_b64 s[{rt}:{}], sendmsg(MSG_RTN_GET_REALTIME)", rt + 1),
        "s_wait_kmcnt 0x0".into(),
        "s_wait_alu depctr_va_sdst(0)".into(),
        format!("s_and_b32 s{w}, s{w}, 0x3ff"),
        format!("s_lshr_b32 s{w}, s{w}, 5"),
        format!("s_lshr_b32 s{z}, ttmp7, 16"),
        format!("s_mul_i32 s{z}, s{z}, s{}", rr + 1),
        format!("s_and_b32 s{l}, ttmp7, 0xffff"),
        format!("s_add_co_i32 s{z}, s{z}, s{l}"),
        format!("s_mul_i32 s{z}, s{z}, s{rr}"),
        format!("s_add_co_i32 s{z}, s{z}, ttmp9"),
        format!("s_mul_i32 s{l}, s{z}, s{}", q + 3),
        format!("s_add_co_i32 s{l}, s{l}, s{w}"),
        format!("s_mul_i32 s{p}, s{l}, s{}", q + 2),
        format!("s_mul_hi_u32 s{}, s{l}, s{}", p + 1, q + 2),
        format!("s_add_co_u32 s{p}, s{p}, s{q}"),
        format!("s_add_co_ci_u32 s{}, s{}, s{}", p + 1, p + 1, q + 1),
        "s_wait_alu depctr_sa_sdst(0)".into(),
        format!("v_writelane_b32 v{}, {MAGIC:#x}, 0", r.v)];
    for (lane, src) in [h1, h2, z, w, rt, rt + 1, ch].into_iter().enumerate() {
        s.push(format!("v_writelane_b32 v{}, s{src}, {}", r.v, lane + 1));
    }
    s.extend(store(r, 8, HEADER_BYTES, va_sdst));
    s.push(END.into());
    s.extend(point(r, 0, "entry", va_sdst));
    s
}

fn exit(r: &Regs, id: u32, va_sdst: bool) -> Vec<String> {
    let mut s = vec![format!("{BEGIN} site {id} exit"),
        format!("s_getreg_b32 s{}, hwreg(HW_REG_SHADER_CYCLES_LO)", r.t),
        format!("s_sendmsg_rtn_b64 s[{}:{}], sendmsg(MSG_RTN_GET_REALTIME)", r.rt, r.rt + 1),
        "s_wait_kmcnt 0x0".into(),
        "s_wait_alu depctr_sa_sdst(0) depctr_vm_vsrc(0)".into(),
        format!("v_writelane_b32 v{}, s{}, 0", r.v, r.t),
        format!("v_writelane_b32 v{}, {:#x}, 1", r.v, id | EXIT_FLAG),
        format!("v_writelane_b32 v{}, s{}, 2", r.v, r.rt),
        format!("v_writelane_b32 v{}, s{}, 3", r.v, r.rt + 1)];
    s.extend(store(r, 4, EXIT_RECORD_BYTES, va_sdst));
    s.push(END.into());
    s
}

fn rename(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len() + 256);
    let mut rest = text;
    while let Some(i) = rest.find(from) {
        let next = rest.as_bytes().get(i + from.len()).copied();
        out.push_str(&rest[..i]);
        out.push_str(if next.is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_') { from } else { to });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

fn set_value(line: &str, value: impl std::fmt::Display) -> String {
    let c = code(line);
    let key = c.split(|ch: char| ch.is_whitespace() || ch == ',').next().unwrap_or("");
    let indent = &line[..line.len() - line.trim_start().len()];
    if c.starts_with(".set") {
        let name = c.trim_start_matches(".set").trim().split(',').next().unwrap_or("").trim();
        format!("{indent}.set {name}, {value}")
    } else if key.ends_with(':') { format!("{indent}{key} {value}") } else { format!("{indent}{key} {value}") }
}

/// Instrument `symbol` in `source`. Returns the rewritten module and the map.
pub fn instrument(source: &str, cfg: &Config) -> Result<(String, Map), String> {
    let sym = cfg.symbol.as_str();
    let lines: Vec<&str> = source.lines().collect();
    let start = lines.iter().position(|l| code(l) == format!("{sym}:")).ok_or_else(|| format!("kernel label {sym}: not found"))?;
    let end = start + lines[start..].iter().position(|l| code(l).starts_with(".size") && code(l).contains(sym))
        .ok_or("kernel .size directive not found")?;
    let body = start + 1..end;
    let (_, _, dsc) = descriptor(&lines, sym)?;
    let mut used_v = BTreeSet::new();
    let mut used_s = BTreeSet::new();
    let mut valu_exec_writer = false;
    for i in body.clone() {
        if kind(lines[i]) != Kind::Insn { continue }
        let c = code(lines[i]);
        used_v.extend(registers(c, b'v'));
        used_s.extend(registers(c, b's'));
        valu_exec_writer |= mnemonic(c).starts_with("v_cmpx");
        if let Some(rest) = c.strip_prefix("s_wait_storecnt_dscnt ") {
            if parse_imm(rest)? >> 8 != 0 { return Err(format!("line {}: nonzero store wait would count profile stores", i + 1)) }
        } else if let Some(rest) = c.strip_prefix("s_wait_storecnt ") {
            if parse_imm(rest)? != 0 { return Err(format!("line {}: nonzero store wait would count profile stores", i + 1)) }
        }
    }
    let vgpr_limit = cfg.vgpr_limit.unwrap_or(dsc.vgpr.div_ceil(VGPR_GRANULE) * VGPR_GRANULE).min(256);
    let regs = pick(&dsc, vgpr_limit, &used_v, &used_s)?;
    let win = windows(&lines, body.clone())?;

    // Gaps: entry before the first instruction, exits at each s_endpgm, rules.
    let first = body.clone().find(|&i| kind(lines[i]) == Kind::Insn).ok_or("empty kernel")?;
    let mut gaps: Vec<Gap> = Vec::new();
    for rule in &cfg.rules { gaps.extend(resolve(&lines, body.clone(), rule, &win)?); }
    for i in body.clone().filter(|&i| mnemonic(lines[i]) == "s_endpgm") {
        if win.iter().any(|&(s, e)| i > s && i <= e) { return Err("s_endpgm inside a scheduling window".into()) }
        // The exit record needs its VGPR: go above LLVM's `s_nop 0` +
        // `s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)` epilogue, which touch no register.
        let mut at = i;
        let prev = |j: usize| (body.start..j).rev().find(|&k| kind(lines[k]) != Kind::Other);
        if let Some(k) = prev(at).filter(|&k| code(lines[k]) == "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)") {
            at = k;
            if let Some(n) = prev(at).filter(|&n| code(lines[n]) == "s_nop 0") { at = n }
        }
        if (at..i).any(|k| kind(lines[k]) == Kind::Label) { return Err("label between the exit record and s_endpgm".into()) }
        gaps.push(Gap { at, rule: "exit".into(), anchor: "s_endpgm".into(), moved: at != i });
    }
    gaps.sort_by_key(|g| g.at);
    let mut sites = vec![Site { id: 0, rule: "entry".into(), line: first + 1, anchor: code(lines[first]).into(), moved: false }];
    for (k, g) in gaps.iter().enumerate() {
        sites.push(Site { id: k as u32 + 1, rule: g.rule.clone(), line: g.at + 1, anchor: g.anchor.clone(), moved: g.moved });
    }

    // s_delay_alu fix-ups for VALU_DEP hints that reach back across a point.
    let valu = |i: usize| kind(lines[i]) == Kind::Insn && mnemonic(lines[i]).starts_with("v_");
    let inserted = |lo: usize, hi: usize| gaps.iter().filter(|g| g.at > lo && g.at <= hi && g.rule != "exit").count() * POINT_VALU;
    let mut fixups = Vec::new();
    let mut replaced: Vec<Option<String>> = vec![None; lines.len()];
    for i in body.clone() {
        if kind(lines[i]) != Kind::Insn || !mnemonic(lines[i]).starts_with("s_delay_alu") { continue }
        let mut d = delay_fields(code(lines[i]));
        let targets: Vec<usize> = body.clone().filter(|&j| j > i && kind(lines[j]) == Kind::Insn && !mnemonic(lines[j]).starts_with("s_delay_alu")).take(d.skip + 1).collect();
        let t1 = *targets.first().ok_or("s_delay_alu without target")?;
        let t2 = targets.last().copied().unwrap_or(t1);
        let fix = |id: &mut String, t: usize| {
            let Some(n) = id.strip_prefix("VALU_DEP_").and_then(|n| n.parse::<usize>().ok()) else { return };
            let mut seen = 0;
            let mut p = t;
            while p > body.start && seen < n { p -= 1; if valu(p) { seen += 1 } }
            let k = inserted(p, t);
            if k > 0 { *id = if n + k <= 4 { format!("VALU_DEP_{}", n + k) } else { "NO_DEP".into() } }
        };
        let before = d.clone();
        fix(&mut d.id0, t1);
        if let Some(id1) = d.id1.as_mut() { fix(id1, t2) }
        if d != before {
            let indent = &lines[i][..lines[i].len() - lines[i].trim_start().len()];
            let text = format!("{indent}{}", delay_text(&d));
            fixups.push(DelayFixup { line: i + 1, before: lines[i].to_string(), after: text.clone() });
            replaced[i] = Some(text);
        }
    }

    let (dstart, dend, _) = descriptor(&lines, sym)?;
    let ext = dsc.kernarg.div_ceil(8) * 8;
    let new_vgpr = dsc.vgpr.max(regs.v + 1);
    let ours = [regs.p + 1, regs.t, regs.e, regs.q + 3, regs.r + 1, regs.rt + 1].into_iter().chain(regs.singles.iter().copied()).max().unwrap_or(0);
    let new_sgpr = dsc.sgpr.max(ours + 1);
    if new_sgpr > SGPR_LIMIT { return Err("profile SGPRs exceed the gfx12 limit".into()) }

    let mut out: Vec<String> = Vec::with_capacity(lines.len() + gaps.len() * 12 + 64);
    let mut gi = 0;
    let mut site = 1u32;
    for (i, raw) in lines.iter().enumerate() {
        if i == start {
            out.push(format!("; DIAGNOSTIC: {sym}{PROFILED_SUFFIX} is a peacemaker profile build; never embed."));
        }
        if i == first { out.extend(entry(&regs, ext, valu_exec_writer).into_iter().map(|l| format!("\t{l}"))); }
        while gi < gaps.len() && gaps[gi].at == i {
            let seq = if gaps[gi].rule == "exit" { exit(&regs, site, valu_exec_writer) } else { point(&regs, site, &gaps[gi].rule, valu_exec_writer) };
            out.extend(seq.into_iter().map(|l| format!("\t{l}")));
            gi += 1; site += 1;
        }
        let line = if let Some(r) = &replaced[i] { r.clone() } else if (dstart..dend).contains(&i) {
            match code(raw).split_whitespace().next() {
                Some(".amdhsa_kernarg_size") => set_value(raw, ext + KERNARG_EXT_BYTES),
                Some(".amdhsa_next_free_vgpr") => set_value(raw, new_vgpr),
                Some(".amdhsa_next_free_sgpr") => set_value(raw, new_sgpr),
                _ => raw.to_string(),
            }
        } else if code(raw).starts_with(&format!(".set .L{sym}.num_vgpr,")) { set_value(raw, new_vgpr) }
        else if code(raw).starts_with(&format!(".set .L{sym}.numbered_sgpr,")) { set_value(raw, new_sgpr) }
        else { raw.to_string() };
        out.push(line);
    }
    let text = metadata(&out.join("\n"), sym, ext, new_vgpr, new_sgpr)? + if source.ends_with('\n') { "\n" } else { "" };
    let text = rename(&text, sym, &format!("{sym}{PROFILED_SUFFIX}"));
    let map = Map {
        diagnostic: "peacemaker profile build: timing records only, never embed".into(),
        symbol: sym.into(), profiled_symbol: format!("{sym}{PROFILED_SUFFIX}"),
        kernarg_size: ext + KERNARG_EXT_BYTES, kernarg_ext_offset: ext, kernarg_ext_bytes: KERNARG_EXT_BYTES,
        header_bytes: HEADER_BYTES, record_bytes: RECORD_BYTES, exit_record_bytes: EXIT_RECORD_BYTES,
        exit_flag: EXIT_FLAG, magic: MAGIC,
        vgpr_before: dsc.vgpr, vgpr_after: new_vgpr, vgpr_limit, sgpr_before: dsc.sgpr, sgpr_after: new_sgpr,
        registers: Registers { scratch_vgpr: regs.v, pointer: regs.p, timestamp: regs.t, exec_save: regs.e,
            entry_quad: regs.q, entry_pair: regs.r, realtime_pair: regs.rt, entry_singles: regs.singles.clone() },
        valu_exec_writer, sites, delay_fixups: fixups,
    };
    Ok((text, map))
}

/// Append the kernarg extension to the kernel's `.args`, update its sizes.
fn metadata(text: &str, sym: &str, ext: u32, vgpr: u16, sgpr: u16) -> Result<String, String> {
    let lines: Vec<&str> = text.lines().collect();
    let meta = lines.iter().position(|l| code(l) == ".amdgpu_metadata").ok_or("missing .amdgpu_metadata")?;
    let name = lines.iter().enumerate().skip(meta)
        .position(|(_, l)| { let c = code(l); c.starts_with(".name:") && c.trim_start_matches(".name:").trim() == sym })
        .map(|p| p + meta).ok_or("kernel metadata entry not found")?;
    let kstart = (meta..name).rev().find(|&i| lines[i].starts_with("  - ")).ok_or("metadata entry start not found")?;
    let kend = (name..lines.len()).find(|&i| lines[i].starts_with("  - ") || (!lines[i].starts_with("    ") && !lines[i].trim().is_empty()))
        .ok_or("metadata entry end not found")?;
    let args = (kstart..kend).find(|&i| lines[i].trim_start_matches("  - ").trim_start().starts_with(".args:"))
        .ok_or("kernel .args not found")?;
    let args_end = (args + 1..kend).find(|&i| !lines[i].starts_with("      ")).unwrap_or(kend);
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 16);
    for (i, l) in lines.iter().enumerate() {
        if i == args_end {
            out.push("      - .address_space: global".into());
            out.push("        .name: pm_profile_records".into());
            out.push(format!("        .offset: {ext}"));
            out.push("        .size: 8".into());
            out.push("        .value_kind: global_buffer".into());
            for (k, n) in ["pm_slot_bytes", "pm_waves_per_wg", "pm_grid_x", "pm_grid_y"].into_iter().enumerate() {
                out.push(format!("      - .name: {n}"));
                out.push(format!("        .offset: {}", ext + 8 + 4 * k as u32));
                out.push("        .size: 4".into());
                out.push("        .value_kind: by_value".into());
            }
        }
        let c = code(l);
        let inside = i > kstart && i < kend;
        let s = if inside && c.starts_with(".kernarg_segment_size:") { set_value(l, ext + KERNARG_EXT_BYTES) }
            else if inside && c.starts_with(".vgpr_count:") { let v: u16 = c[12..].trim().parse().unwrap_or(0); set_value(l, v.max(vgpr)) }
            else if inside && c.starts_with(".sgpr_count:") { let v: u16 = c[12..].trim().parse().unwrap_or(0); set_value(l, v.max(sgpr)) }
            else { l.to_string() };
        out.push(s);
    }
    Ok(out.join("\n"))
}

/// Lines strictly between `symbol:` and its `.size` directive.
pub fn kernel_body(text: &str, symbol: &str) -> Result<String, String> {
    let l: Vec<&str> = text.lines().collect();
    let s = l.iter().position(|x| code(x) == format!("{symbol}:")).ok_or_else(|| format!("kernel label {symbol}: missing"))?;
    let e = s + l[s..].iter().position(|x| code(x).starts_with(".size") && code(x).contains(symbol))
        .ok_or("kernel .size missing")?;
    Ok(l[s + 1..e].join("\n"))
}

/// Independent check of an instrumented module against its source and map:
/// removing the marked blocks and undoing the recorded hint fix-ups gives the
/// original kernel body; the marked blocks touch only the mapped registers;
/// the original body never names the persistent profile registers.
pub fn verify(source: &str, instrumented: &str, map: &Map) -> Result<(), String> {
    let sym = &map.symbol;
    let body = |text: &str, name: &str| -> Result<Vec<String>, String> {
        Ok(kernel_body(text, name)?.lines().map(str::to_string).collect())
    };
    let original = body(source, sym)?;
    let profiled = body(instrumented, &map.profiled_symbol)?;
    let r = &map.registers;
    let persistent_s: BTreeSet<u16> = [r.pointer, r.pointer + 1, r.timestamp, r.exec_save].into();
    let mut entry_s = persistent_s.clone();
    entry_s.extend(r.entry_quad..r.entry_quad + 4);
    entry_s.extend([r.entry_pair, r.entry_pair + 1, r.realtime_pair, r.realtime_pair + 1, 0, 1]);
    entry_s.extend(r.entry_singles.iter().copied());
    let mut stripped = Vec::new();
    let mut block: Option<String> = None;
    let mut blocks = 0;
    for line in &profiled {
        let t = line.trim();
        if t.starts_with(BEGIN) { if block.is_some() { return Err("nested profile block".into()) } block = Some(t.to_string()); blocks += 1; continue }
        if t.starts_with(END) { block.take().ok_or("unmatched profile end")?; continue }
        if let Some(b) = &block {
            let c = code(line);
            let v = registers(c, b'v');
            let s = registers(c, b's');
            let entry_block = b.contains("entry header");
            let allowed_v = |n: &u16| *n == r.scratch_vgpr || (entry_block && *n == 0);
            if !v.iter().all(allowed_v) { return Err(format!("profile block touches VGPRs {v:?}: {c}")) }
            let allowed_s = if entry_block { &entry_s } else { &persistent_s };
            let exit_block = b.ends_with(" exit");
            if !s.iter().all(|n| allowed_s.contains(n) || (exit_block && (*n == r.realtime_pair || *n == r.realtime_pair + 1))) {
                return Err(format!("profile block touches SGPRs {s:?}: {c}"))
            }
            // Points run inside live code: only SCC/VCC/M0-neutral opcodes.
            // Entry code runs before the kernel and may use SALU arithmetic.
            let m = mnemonic(c);
            let point_ok = matches!(m, "s_getreg_b32" | "s_wait_alu" | "v_writelane_b32" | "s_mov_b32"
                | "s_add_nc_u64" | "global_store_addtid_b32" | "s_sendmsg_rtn_b64" | "s_wait_kmcnt");
            if m.contains("branch") || c.contains("vcc") || c.contains("m0") || (!entry_block && !point_ok) {
                return Err(format!("profile block touches control state: {c}"))
            }
            continue;
        }
        stripped.push(line.clone());
    }
    if block.is_some() { return Err("unterminated profile block".into()) }
    if blocks != map.sites.len() + 1 { return Err(format!("{blocks} blocks for {} sites", map.sites.len())) }
    for f in &map.delay_fixups {
        let k = stripped.iter().position(|l| *l == f.after).ok_or("delay fix-up not found")?;
        stripped[k] = f.before.clone();
    }
    let renamed: Vec<String> = original.iter().map(|l| rename(l, sym, &map.profiled_symbol)).collect();
    let diag = format!("; DIAGNOSTIC: {}", map.profiled_symbol);
    let stripped: Vec<String> = stripped.into_iter().filter(|l| !l.starts_with(&diag)).collect();
    if stripped != renamed {
        let k = stripped.iter().zip(&renamed).position(|(a, b)| a != b).unwrap_or(stripped.len().min(renamed.len()));
        return Err(format!("stripped body differs from the original at body line {}", k + 1))
    }
    for line in &original {
        if kind(line) != Kind::Insn { continue }
        let c = code(line);
        if registers(c, b'v').contains(&r.scratch_vgpr) || !registers(c, b's').is_disjoint(&persistent_s) {
            return Err(format!("original kernel uses a profile register: {c}"))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "\t.text\nk:\n\ts_load_b64 s[2:3], s[0:1], 0x0\n\ts_wait_kmcnt 0x0\n\
        \tv_add_f32_e32 v1, v2, v3\n\tv_mul_f32_e32 v4, v5, v6\n\
        \ts_barrier_signal -1\n\ts_barrier_wait 0xffff\n\
        \ts_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)\n\
        \tv_add_f32_e32 v7, v1, v1\n\tv_add_f32_e32 v8, v4, v4\n\
        \ts_clause 0x1\n\tglobal_load_b32 v9, v10, s[2:3]\n\tglobal_load_b32 v11, v10, s[2:3] offset:4\n\
        \ts_wait_loadcnt 0x0\n\tglobal_store_b32 v10, v9, s[2:3]\n\ts_endpgm\n.Lk_end:\n\t.size k, .Lk_end-k\n\
        .amdhsa_kernel k\n\t.amdhsa_kernarg_size 12\n\t.amdhsa_user_sgpr_count 2\n\t.amdhsa_wavefront_size32 1\n\
        \t.amdhsa_system_sgpr_workgroup_id_x 1\n\t.amdhsa_next_free_vgpr 12\n\t.amdhsa_next_free_sgpr 4\n.end_amdhsa_kernel\n\
        .amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n      - .address_space: global\n        .offset: 0\n        .size: 8\n        .value_kind: global_buffer\n\
        \x20   .kernarg_segment_size: 12\n    .name: k\n    .symbol: k.kd\n    .vgpr_count: 12\namdhsa.target: amdgcn-amd-amdhsa--gfx1201\n.end_amdgpu_metadata\n";

    fn cfg(rules: &str) -> Config {
        serde_json::from_str(&format!("{{\"symbol\":\"k\",\"rules\":{rules}}}")).unwrap()
    }

    #[test]
    fn points_keep_windows_and_retarget_valu_hints() {
        let c = cfg(r#"[{"name":"bar","after":"s_barrier_wait"},{"name":"ld","after":"s_clause"}]"#);
        let (text, map) = instrument(KERNEL, &c).unwrap();
        // The barrier point sits between both producers and their consumers:
        // two inserted VALUs push VALU_DEP_2 to VALU_DEP_4.
        assert_eq!(map.delay_fixups.len(), 1);
        assert!(map.delay_fixups[0].after.ends_with("instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)"));
        // The clause anchor moves past the clause's last load.
        let ld = map.sites.iter().find(|s| s.rule == "ld").unwrap();
        assert!(ld.moved);
        let lines: Vec<&str> = text.lines().collect();
        let clause = lines.iter().position(|l| l.trim() == "s_clause 0x1").unwrap();
        assert!(lines[clause + 1].contains("global_load_b32 v9") && lines[clause + 2].contains("global_load_b32 v11"));
        verify(KERNEL, &text, &map).unwrap();
        assert!(text.contains(".kernarg_segment_size: 40") && text.contains(".amdhsa_kernarg_size 40"));
        assert!(text.contains("k__pm_profile:") && !text.contains("\nk:\n"));
    }

    #[test]
    fn nonzero_store_wait_is_rejected() {
        let src = KERNEL.replace("\ts_endpgm", "\ts_wait_storecnt 0x1\n\ts_endpgm");
        assert!(instrument(&src, &cfg("[]")).is_err());
    }

    #[test]
    fn verify_rejects_a_block_that_clobbers_kernel_state() {
        let (text, map) = instrument(KERNEL, &cfg(r#"[{"name":"bar","before":"s_barrier_signal"}]"#)).unwrap();
        let bad = text.replacen("s_mov_b32 exec_lo, 3", "s_mov_b32 exec_lo, 3\n\tv_mov_b32 v1, 0", 1);
        assert_ne!(bad, text);
        assert!(verify(KERNEL, &bad, &map).is_err());
        let site = format!("v_writelane_b32 v{}, 1, 1", map.registers.scratch_vgpr);
        assert!(verify(KERNEL, &text.replacen(&site, &format!("{site}\n\ts_cmp_eq_u32 s{0}, 0", map.registers.timestamp), 1), &map).is_err());
    }
}
