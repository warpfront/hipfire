// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `emu_race_gate`: run ONE production hipcc object launch (a JSON launch spec) through the pre-card
//! emulator gate (`Emulator::hw_model`: in-flight loads honouring `s_waitcnt`, LDS happens-before race
//! detection, adversarial wave schedules, strict LDS/workspace reads) at the spec's real grid, block,
//! dynamic LDS, kernarg values and exact-size guarded buffers. No device. PM-KRSI-1100 DecodeAudit.
//! Wave32 and wave64 objects (gfx1100/gfx1151/gfx1201): the wavefront size comes from the lifted kernel (KD
//! ENABLE_WAVEFRONT_SIZE32), `block` threads are split into waves of that size, EXEC/VCC are 64-bit in wave64 and the
//! gate model (in-flight loads, LDS races, schedules) runs per wave of either size.
//!
//! Usage: emu_race_gate SPEC.json [--wg x,y,z]... [--order forward|reverse|random:SEED] [--json OUT.json] [--max-findings N]
//!   no --wg: every workgroup of `grid_wg`. `--order` permutes the order the workgroups run in (inter-workgroup
//!   schedule: the gate does not interleave workgroups); the report carries sha256 of every output buffer so runs
//!   in different orders can be compared (tools/order_diff.py). Exit 0 = clean, 1 = at least one finding, 2 = the spec/object could
//!   not be run (lift refused, an opcode has no emulator semantics, bad spec): that is NOT a clean verdict.
//!
//! Spec (spec_version 1): `{name, object (path relative to the spec dir), object_sha256, arch, symbol,
//!   grid_wg[3], block[3], dynamic_lds, args[], buffers[], notes, nan_policy?}`. `nan_policy` = `strict` (default: an
//!   unmeasured NaN layout of a qualified opcode is a finding) | `meets_nan` (`Emulator::nan_meets_nan`: the lab
//!   exactness contract, a NaN operand propagates a NaN, payload not claimed; needed for hipcc lane kernels fed Wild
//!   headers).
//!   args (the kernel's explicit arguments, in declaration order; count and byte size are checked against the object's
//!   kernarg metadata, hidden arguments are filled from the launch geometry as the HIP runtime does):
//!     `{name, buf, off?}` pointer to a buffer (+ byte offset), `{name, i32|u32|f32}`, `{name, null:true}`,
//!     `{name, bytes_hex}` by-value struct (little-endian raw bytes).
//!   buffers: `{name, bytes, file?, init: seeded|undef|ptrtable, output, entries?}`. Every buffer is its own
//!     exact-size region with `GUARD_BYTES` of unmapped space around it, so a one-byte overrun is an OOB finding.
//!     `seeded`: bytes from `file`; `undef`: poison, a load of a byte no store wrote is an `[uninit]` finding;
//!     `ptrtable`: u64 entries, each `null` (0) or `{buf, offset}` (address of that buffer + offset) or `{poison:true}`.
//!     A store to a buffer with `output:false` (and to the kernarg segment) is a finding.
//! Findings: class race | in-flight | uninit | schedule | oob | write-to-input | other, with the pc / instruction /
//! workgroup the emulator names (`inst=PC name` in its message) and the full message.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Abi, Frontend};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

const GUARD_BYTES: u64 = 1 << 20;
const POISON: u8 = 0xA5;

type Res<T> = Result<T, String>;

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("emu_race_gate: {e}");
            std::process::exit(2);
        }
    }
}

fn parse_wg(s: &str) -> Res<[u32; 3]> {
    let v: Vec<u32> = s
        .split(',')
        .map(|x| x.parse::<u32>().map_err(|e| format!("--wg {s}: {e}")))
        .collect::<Res<_>>()?;
    if v.len() != 3 {
        return Err("--wg requires x,y,z".into());
    }
    Ok([v[0], v[1], v[2]])
}

fn u32s(v: &Value, key: &str) -> Res<[u32; 3]> {
    let a = v[key].as_array().ok_or_else(|| format!("missing {key}"))?;
    if a.len() != 3 {
        return Err(format!("{key} must have 3 axes"));
    }
    let mut r = [0u32; 3];
    for i in 0..3 {
        r[i] = a[i]
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(|| format!("{key}[{i}] not a u32"))?;
    }
    Ok(r)
}

fn classify(msg: &str) -> &'static str {
    if msg.contains("[race]") {
        "race"
    } else if msg.contains("[in-flight]") {
        "in-flight"
    } else if msg.contains("[uninit]") || msg.contains("never written") {
        "uninit"
    } else if msg.contains("[schedule]") {
        "schedule"
    } else if msg.contains("unmapped memory")
        || msg.contains("out of bounds")
        || msg.contains("out of range")
    {
        "oob"
    } else {
        "other"
    }
}

/// `inst=PC name` as the emulator names the faulting instruction.
fn pc_inst(msg: &str) -> (Option<u64>, Option<String>) {
    let Some(at) = msg.find("inst=") else {
        return (None, None);
    };
    let rest = &msg[at + 5..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let pc = digits.parse::<u64>().ok();
    let name = rest[digits.len()..]
        .trim_start()
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .next()
        .map(str::to_owned);
    (pc, name.filter(|n| !n.is_empty()))
}

fn lift(image: &[u8]) -> Res<(peacemaker_ir::Program, &'static str)> {
    let mut errs = Vec::new();
    for (name, frontend) in [("hipcc", Frontend::Hipcc), ("builder", Frontend::Builder)] {
        match peacemaker_lift::lift_object(image, peacemaker_lift::Options { frontend }) {
            Ok(l) => return Ok((l.program, name)),
            Err(e) => errs.push(format!("{name}: {e:?}")),
        }
    }
    Err(format!("lift refused the object ({})", errs.join("; ")))
}

fn run() -> Res<i32> {
    let mut args = std::env::args().skip(1);
    let spec_path =
        PathBuf::from(args.next().ok_or(
            "usage: emu_race_gate SPEC.json [--wg x,y,z]... [--json OUT] [--max-findings N]",
        )?);
    let mut selected: Vec<[u32; 3]> = Vec::new();
    let mut json_out: Option<PathBuf> = None;
    let mut max_findings = 50usize;
    let mut order_opt = "forward".to_owned();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--order" => order_opt = args.next().ok_or("missing --order value")?,
            "--wg" => selected.push(parse_wg(&args.next().ok_or("missing --wg value")?)?),
            "--json" => json_out = Some(PathBuf::from(args.next().ok_or("missing --json value")?)),
            "--max-findings" => {
                max_findings = args
                    .next()
                    .ok_or("missing value")?
                    .parse()
                    .map_err(|e| format!("--max-findings: {e}"))?
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    let started = Instant::now();
    let dir = spec_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let spec: Value = serde_json::from_slice(
        &std::fs::read(&spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?,
    )
    .map_err(|e| format!("spec json: {e}"))?;
    if spec["spec_version"].as_u64() != Some(1) {
        return Err("spec_version must be 1".into());
    }
    let name = spec["name"].as_str().unwrap_or("?").to_owned();
    let symbol = spec["symbol"].as_str().ok_or("missing symbol")?.to_owned();
    let image = std::fs::read(dir.join(spec["object"].as_str().ok_or("missing object")?))
        .map_err(|e| format!("object: {e}"))?;
    let sha = format!("{:x}", Sha256::digest(&image));
    if Some(sha.as_str()) != spec["object_sha256"].as_str() {
        return Err(format!("object sha256 {sha} differs from the spec"));
    }
    let grid = u32s(&spec, "grid_wg")?;
    let block = u32s(&spec, "block")?;
    let dynamic_lds = u32::try_from(spec["dynamic_lds"].as_u64().ok_or("missing dynamic_lds")?)
        .map_err(|_| "dynamic_lds")?;

    let (program, frontend) = lift(&image)?;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("the object does not define `{symbol}`"))?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err("not an HSA kernel".into());
    };
    let mut missing = std::collections::BTreeMap::<&str, usize>::new();
    for id in &kernel.body.layout {
        let inst = kernel
            .body
            .insts
            .get(*id)
            .ok_or("layout names a missing instruction")?;
        match peacemaker_ir::isa::lookup(program.target.arch, inst.op, inst.form) {
            Some(row) if peacemaker_ir::emu::semantic_class(row.name).is_some() => {}
            Some(row) => *missing.entry(row.name).or_default() += 1,
            None => *missing.entry("<undecodable>").or_default() += 1,
        }
    }
    if !missing.is_empty() {
        let l: Vec<String> = missing.iter().map(|(n, c)| format!("{n} x{c}")).collect();
        return Err(format!("no emulator semantics for: {}", l.join(", ")));
    }

    // ---- buffers ----
    struct Buf {
        name: String,
        output: bool,
        init: String,
        bytes: Vec<u8>,
        addr: u64,
    }
    let bspec = spec["buffers"].as_array().ok_or("missing buffers")?;
    let mut bufs: Vec<Buf> = Vec::new();
    for b in bspec {
        let bname = b["name"].as_str().ok_or("buffer without name")?.to_owned();
        let init = b["init"].as_str().unwrap_or("seeded").to_owned();
        let n = usize::try_from(b["bytes"].as_u64().ok_or("buffer without bytes")?)
            .map_err(|_| "bytes")?;
        let bytes = match init.as_str() {
            "seeded" => {
                let f = b["file"]
                    .as_str()
                    .ok_or_else(|| format!("seeded buffer {bname} has no file"))?;
                let v =
                    std::fs::read(dir.join(f)).map_err(|e| format!("buffer {bname}: {f}: {e}"))?;
                if v.len() != n {
                    return Err(format!(
                        "buffer {bname}: file is {} B, spec says {n}",
                        v.len()
                    ));
                }
                v
            }
            "undef" => vec![POISON; n],
            "ptrtable" => vec![0u8; n],
            other => return Err(format!("buffer {bname}: unknown init {other}")),
        };
        bufs.push(Buf {
            name: bname,
            output: b["output"].as_bool().unwrap_or(false),
            init,
            bytes,
            addr: 0,
        });
    }
    let mut mem = Memory::default();
    // map every non-ptrtable buffer first (ptrtable entries need their targets' addresses)
    let mut order: Vec<usize> = (0..bufs.len())
        .filter(|&i| bufs[i].init != "ptrtable")
        .collect();
    order.extend((0..bufs.len()).filter(|&i| bufs[i].init == "ptrtable"));
    for &i in &order {
        if bufs[i].init == "ptrtable" {
            let entries = bspec[i]["entries"]
                .as_array()
                .ok_or_else(|| format!("ptrtable {} has no entries", bufs[i].name))?;
            if entries.len() * 8 != bufs[i].bytes.len() {
                return Err(format!(
                    "ptrtable {}: {} entries x 8 != {} B",
                    bufs[i].name,
                    entries.len(),
                    bufs[i].bytes.len()
                ));
            }
            for (k, e) in entries.iter().enumerate() {
                let v: u64 = if e.is_null() {
                    0
                } else if e["poison"].as_bool() == Some(true) {
                    0xDEAD_BEEF_0000_0000
                } else {
                    let tn = e["buf"].as_str().ok_or("ptrtable entry without buf")?;
                    let t = bufs
                        .iter()
                        .find(|x| x.name == tn)
                        .ok_or_else(|| format!("ptrtable entry -> unknown buffer {tn}"))?;
                    t.addr + e["offset"].as_u64().unwrap_or(0)
                };
                bufs[i].bytes[k * 8..k * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
        }
        let data = std::mem::take(&mut bufs[i].bytes);
        bufs[i].addr = mem
            .map_guarded(bufs[i].name.clone(), data, GUARD_BYTES)
            .map_err(|e| format!("map {}: {e}", bufs[i].name))?;
        if bufs[i].init == "undef" {
            mem.set_strict_read(&bufs[i].name)
                .map_err(|e| format!("strict_read {}: {e}", bufs[i].name))?;
        }
    }

    // ---- kernarg segment (explicit args from the spec, hidden args from the geometry) ----
    let segment = metadata
        .parsed
        .kernarg_segment_size
        .max(descriptor.kernarg_size) as usize;
    let explicit: Vec<_> = metadata
        .parsed
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .collect();
    let aspec = spec["args"].as_array().ok_or("missing args")?;
    if explicit.len() != aspec.len() {
        return Err(format!(
            "abi: the object declares {} explicit kernel arguments, the spec passes {}",
            explicit.len(),
            aspec.len()
        ));
    }
    let end = metadata
        .parsed
        .args
        .iter()
        .map(|a| (a.offset + a.size) as usize)
        .max()
        .unwrap_or(0);
    let mut kern = vec![0u8; segment.max(end)];
    for (i, (m, a)) in explicit.iter().zip(aspec).enumerate() {
        let bytes: Vec<u8> = if let Some(bn) = a["buf"].as_str() {
            let b = bufs
                .iter()
                .find(|x| x.name == bn)
                .ok_or_else(|| format!("arg {i}: unknown buffer {bn}"))?;
            let off = a["off"].as_u64().unwrap_or(0);
            let len = bspec
                .iter()
                .find(|x| x["name"].as_str() == Some(bn))
                .and_then(|x| x["bytes"].as_u64())
                .unwrap_or(0);
            if off > len {
                return Err(format!("arg {i}: off {off} beyond buffer {bn} ({len} B)"));
            }
            (b.addr + off).to_le_bytes().to_vec()
        } else if a["null"].as_bool() == Some(true) {
            0u64.to_le_bytes().to_vec()
        } else if let Some(x) = a["i32"].as_i64() {
            i32::try_from(x)
                .map_err(|_| format!("arg {i}: i32 out of range"))?
                .to_le_bytes()
                .to_vec()
        } else if let Some(x) = a["u32"].as_u64() {
            u32::try_from(x)
                .map_err(|_| format!("arg {i}: u32 out of range"))?
                .to_le_bytes()
                .to_vec()
        } else if let Some(x) = a["f32"].as_f64() {
            (x as f32).to_le_bytes().to_vec()
        } else if let Some(h) = a["bytes_hex"].as_str() {
            (0..h.len() / 2)
                .map(|k| {
                    u8::from_str_radix(&h[2 * k..2 * k + 2], 16)
                        .map_err(|e| format!("arg {i}: bytes_hex: {e}"))
                })
                .collect::<Res<_>>()?
        } else {
            return Err(format!("arg {i}: unrecognised value {a}"));
        };
        if bytes.len() != m.size as usize {
            return Err(format!(
                "abi: explicit argument {i} `{}` is {} B in the object, {} B in the spec",
                a["name"].as_str().unwrap_or(""),
                m.size,
                bytes.len()
            ));
        }
        let off = m.offset as usize;
        kern[off..off + bytes.len()].copy_from_slice(&bytes);
    }
    for a in metadata
        .parsed
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_"))
    {
        let (off, size) = (a.offset as usize, a.size as usize);
        let v: u64 = match a.value_kind.as_str() {
            "hidden_block_count_x" => u64::from(grid[0]),
            "hidden_block_count_y" => u64::from(grid[1]),
            "hidden_block_count_z" => u64::from(grid[2]),
            "hidden_group_size_x" => u64::from(block[0]),
            "hidden_group_size_y" => u64::from(block[1]),
            "hidden_group_size_z" => u64::from(block[2]),
            "hidden_remainder_x" | "hidden_remainder_y" | "hidden_remainder_z" => 0,
            "hidden_grid_dims" => 3,
            "hidden_dynamic_lds_size" => u64::from(dynamic_lds),
            _ => 0,
        };
        kern[off..off + size].copy_from_slice(&v.to_le_bytes()[..size.min(8)]);
    }
    kern.resize(kern.len().next_multiple_of(16), 0);
    let kva = mem
        .map_guarded("kernarg".to_owned(), kern, GUARD_BYTES)
        .map_err(|e| format!("map kernarg: {e}"))?;

    // ---- run ----
    let mut emu = Emulator::new(&program, &symbol)
        .map_err(|e| format!("emulator: {e}"))?
        .hw_model()
        .gfx1100_as_gfx1151();
    let nan_policy = spec["nan_policy"].as_str().unwrap_or("strict").to_owned();
    match nan_policy.as_str() {
        "strict" => {}
        "meets_nan" => emu = emu.nan_meets_nan(),
        other => return Err(format!("nan_policy {other}: expected strict|meets_nan")),
    }
    let wave = kernel.wave.lanes();
    let wgs: Vec<[u32; 3]> = if selected.is_empty() {
        let mut v = Vec::new();
        for z in 0..grid[2] {
            for y in 0..grid[1] {
                for x in 0..grid[0] {
                    v.push([x, y, z]);
                }
            }
        }
        v
    } else {
        for w in &selected {
            if (0..3).any(|i| w[i] >= grid[i]) {
                return Err(format!("--wg {w:?} outside the grid {grid:?}"));
            }
        }
        selected.clone()
    };
    let mut wgs = wgs;
    match order_opt.as_str() {
        "forward" => {}
        "reverse" => wgs.reverse(),
        o if o.starts_with("random:") => {
            let mut x: u64 = o[7..]
                .parse::<u64>()
                .map_err(|e| format!("--order random: {e}"))?
                ^ 0x9E37_79B9_7F4A_7C15;
            for i in (1..wgs.len()).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                wgs.swap(i, (x % (i as u64 + 1)) as usize);
            }
        }
        other => {
            return Err(format!(
                "--order {other}: expected forward|reverse|random:SEED"
            ))
        }
    }
    let (mut instructions, mut ran, mut nfind) = (0u64, 0u64, 0usize);
    let mut findings: Vec<Value> = Vec::new();
    let mut schedules: Vec<String> = Vec::new();
    let mut by_class = std::collections::BTreeMap::<&'static str, usize>::new();
    fn note(
        by_class: &mut std::collections::BTreeMap<&'static str, usize>,
        wg: [u32; 3],
        msg: String,
        class: Option<&'static str>,
        nfind: &mut usize,
        findings: &mut Vec<Value>,
        max_findings: usize,
    ) {
        let class = class.unwrap_or_else(|| classify(&msg));
        *by_class.entry(class).or_default() += 1;
        *nfind += 1;
        let (pc, inst) = pc_inst(&msg);
        if findings.len() < max_findings {
            findings.push(
                json!({"class": class, "workgroup": wg, "pc": pc, "inst": inst, "message": msg}),
            );
        }
    }
    for wg in &wgs {
        let launch = Launch {
            kernarg_ptr: kva,
            workgroup: *wg,
            block,
            dynamic_lds,
        };
        match catch_unwind(AssertUnwindSafe(|| {
            emu.run_workgroup_report(&launch, &mut mem)
        })) {
            Ok(Ok(rep)) => {
                instructions += rep.instructions;
                ran += 1;
                if schedules.is_empty() {
                    schedules = rep.schedules;
                }
            }
            Ok(Err(e)) => note(
                &mut by_class,
                *wg,
                e,
                None,
                &mut nfind,
                &mut findings,
                max_findings,
            ),
            Err(p) => {
                let m = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                note(
                    &mut by_class,
                    *wg,
                    format!("emulator panic: {m}"),
                    None,
                    &mut nfind,
                    &mut findings,
                    max_findings,
                );
            }
        }
    }
    // stores outside the declared output buffers
    let mut outputs = Vec::new();
    for b in &bufs {
        let r = mem
            .regions
            .iter()
            .find(|r| r.name == b.name)
            .ok_or("mapped region vanished")?;
        let w = r.written.count();
        if b.output {
            outputs.push(json!({"buffer": b.name, "written_bytes": w, "bytes": r.bytes.len(), "sha256": format!("{:x}", Sha256::digest(&r.bytes))}));
        } else if w != 0 {
            note(
                &mut by_class,
                [0, 0, 0],
                format!(
                    "{w} byte(s) of the read-only buffer `{}` were stored",
                    b.name
                ),
                Some("write-to-input"),
                &mut nfind,
                &mut findings,
                max_findings,
            );
        }
    }
    if let Some(r) = mem.regions.iter().find(|r| r.name == "kernarg") {
        if r.written.count() != 0 {
            note(
                &mut by_class,
                [0, 0, 0],
                format!(
                    "{} byte(s) of the kernarg segment were stored",
                    r.written.count()
                ),
                Some("write-to-input"),
                &mut nfind,
                &mut findings,
                max_findings,
            );
        }
    }
    let verdict = if nfind == 0 { "clean" } else { "FINDINGS" };
    let report = json!({
        "spec": name, "symbol": symbol, "object_sha256": sha, "frontend": frontend, "verdict": verdict,
        "order": order_opt, "grid_wg": grid, "block": block, "dynamic_lds": dynamic_lds,
        "workgroups_total": wgs.len(), "workgroups_clean": ran, "instructions_sched0": instructions,
        "schedules": schedules, "findings_total": nfind, "findings_by_class": by_class, "findings": findings,
        "output_buffers": outputs, "elapsed_ms": started.elapsed().as_millis() as u64,
        "private_segment_fixed_size": descriptor.private_segment_fixed_size, "wave": wave, "nan_policy": nan_policy,
        "group_segment_fixed_size": descriptor.group_segment_fixed_size,
        "model": format!("Emulator::hw_model(){}.gfx1100_as_gfx1151() (arch {:?}, wave{wave})", if nan_policy == "meets_nan" { ".nan_meets_nan()" } else { "" }, program.target.arch),
        "inter_workgroup_races": "not modelled: workgroups run one after another on shared memory",
    });
    println!(
        "{} {} wg {}/{} clean, {} finding(s) {:?}  [{} ms]",
        verdict,
        name,
        ran,
        wgs.len(),
        nfind,
        by_class,
        started.elapsed().as_millis()
    );
    for f in &findings {
        println!(
            "  {} wg={:?} pc={} inst={} :: {}",
            f["class"].as_str().unwrap_or(""),
            f["workgroup"],
            f["pc"],
            f["inst"],
            f["message"].as_str().unwrap_or("")
        );
    }
    if let Some(p) = json_out {
        std::fs::write(
            &p,
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(if nfind == 0 { 0 } else { 1 })
}
