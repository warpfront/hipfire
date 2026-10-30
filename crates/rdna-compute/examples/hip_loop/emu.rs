// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The pre-card emulator gate of `hip_loop --emu`: no candidate reaches the XTX before its code
//! object has run, at a small shape and over the FULL launch grid, on the PeaceMaker CPU emulator
//! (`peacemaker-ir::emu`, objects lifted by `peacemaker_lift::lift_object`) next to the
//! incumbent's object, from the same host-side seeded inputs.
//!
//! What fails the gate (every one of these is a card fault or a silent wrong answer there):
//! * any global access outside a mapped buffer: each buffer is its own exact-size emulator region
//!   with [`GUARD_BYTES`] of unmapped space on both sides, so a one-byte overrun is an error;
//! * LDS access beyond `group_segment_fixed_size + dynamic LDS`;
//! * a read of an LDS byte this workgroup never wrote (`Emulator::strict_lds`);
//! * a write into an input buffer or the kernarg segment (the kernel's contract is `out` only);
//! * a barrier deadlock, an instruction-limit overrun, an instruction the emulator has no
//!   semantics for (the error names the opcode: the gate fails closed), a panic in the emulator;
//! * output bytes not bit-identical to the emulated incumbent's at the same shape and inputs.
//!
//! The kernarg segment is mapped as its own region at the object's own `kernarg_segment_size`:
//! the explicit arguments at the metadata offsets, the hidden arguments the runtime fills
//! (block counts, group sizes, remainders, `hidden_dynamic_lds_size`) from the arm's launch
//! geometry, everything else zero (a zero pointer a kernel dereferences is an unmapped access).
//!
//! Included with `#[path = "hip_loop/emu.rs"] mod emu;`.
#![allow(dead_code)]

use crate::elf::sha256_hex;
use crate::node::{EmuArg, EmuCase, Geometry, Result, Xs64, PAD_OWNER};
use peacemaker_ir::emu::{Emulator, Launch, Memory, WriteSet};
use peacemaker_ir::inst::{Abi, Frontend};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

/// Unmapped gap below and above every region (>= 64 KiB required by the gate's contract).
pub const GUARD_BYTES: u64 = 1 << 20;

/// What one emulated arm produced.
pub struct ArmRun {
    pub out: Vec<u8>,
    pub instructions: u64,
    pub workgroups: u64,
    pub json: Value,
    /// Bytes of the output buffer the arm wrote.
    pub out_written: usize,
    /// Which output bytes the arm wrote.
    pub written: Option<WriteSet>,
}

/// An arm that failed, with the work done before it did.
pub struct ArmFail {
    pub reason: String,
    pub instructions: u64,
    pub workgroups: u64,
    pub json: Value,
}

fn fail(reason: impl Into<String>, json: Value) -> ArmFail {
    ArmFail {
        reason: reason.into(),
        instructions: 0,
        workgroups: 0,
        json,
    }
}

/// The kernarg segment of one arm: explicit arguments from `case` (addresses of the mapped
/// buffers) at the object's metadata offsets, hidden arguments from the launch geometry.
fn kernarg_bytes(
    case: &EmuCase,
    addrs: &[u64],
    args: &[peacemaker_ir::metadata::Kernarg],
    segment: usize,
    geo: &Geometry,
) -> Result<Vec<u8>> {
    let explicit: Vec<_> = args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .collect();
    if explicit.len() != case.args.len() {
        return Err(format!(
            "abi: the object declares {} explicit kernel arguments, the node passes {}",
            explicit.len(),
            case.args.len()
        ));
    }
    let end = args
        .iter()
        .map(|a| (a.offset + a.size) as usize)
        .max()
        .unwrap_or(0);
    let mut k = vec![0u8; segment.max(end)];
    for (i, (a, v)) in explicit.iter().zip(&case.args).enumerate() {
        let (off, size) = (a.offset as usize, a.size as usize);
        let bytes: Vec<u8> = match v {
            EmuArg::Buf(b) => addrs
                .get(*b)
                .ok_or("abi: argument names a missing buffer")?
                .to_le_bytes()
                .to_vec(),
            EmuArg::BufOff(b, off) => (addrs
                .get(*b)
                .ok_or("abi: argument names a missing buffer")?
                + off)
                .to_le_bytes()
                .to_vec(),
            EmuArg::Null => 0u64.to_le_bytes().to_vec(),
            EmuArg::I32(x) => x.to_le_bytes().to_vec(),
            EmuArg::U32(x) => x.to_le_bytes().to_vec(),
            EmuArg::U64(x) => x.to_le_bytes().to_vec(),
            EmuArg::F32(x) => x.to_le_bytes().to_vec(),
        };
        if bytes.len() != size {
            return Err(format!(
                "abi: explicit argument {i} `{}` is {size} B in the object, {} B in the node",
                a.name,
                bytes.len()
            ));
        }
        k[off..off + size].copy_from_slice(&bytes);
    }
    let [bx, by, bz] = geo.block;
    let [gx, gy, gz] = geo.grid;
    for a in args.iter().filter(|a| a.value_kind.starts_with("hidden_")) {
        let (off, size) = (a.offset as usize, a.size as usize);
        let value: u64 = match a.value_kind.as_str() {
            "hidden_block_count_x" => u64::from(gx),
            "hidden_block_count_y" => u64::from(gy),
            "hidden_block_count_z" => u64::from(gz),
            "hidden_group_size_x" => u64::from(bx),
            "hidden_group_size_y" => u64::from(by),
            "hidden_group_size_z" => u64::from(bz),
            // The grid is a whole number of workgroups.
            "hidden_remainder_x" | "hidden_remainder_y" | "hidden_remainder_z" => 0,
            "hidden_grid_dims" => 3,
            "hidden_dynamic_lds_size" => u64::from(geo.lds),
            // Global offsets, queue/printf/hostcall/heap pointers, ...: zero.
            _ => 0,
        };
        k[off..off + size].copy_from_slice(&value.to_le_bytes()[..size.min(8)]);
    }
    Ok(k)
}

/// Lift `image` (hipcc first, then the builder frontend for PM twins) and return the program and
/// the frontend that accepted it.
fn lift(image: &[u8]) -> Result<(peacemaker_ir::Program, &'static str)> {
    let mut errs = Vec::new();
    for (name, frontend) in [("hipcc", Frontend::Hipcc), ("builder", Frontend::Builder)] {
        match peacemaker_lift::lift_object(image, peacemaker_lift::Options { frontend }) {
            Ok(l) => return Ok((l.program, name)),
            Err(e) => errs.push(format!("{name}: {e:?}")),
        }
    }
    Err(format!("lift refused the object ({})", errs.join("; ")))
}

/// The wavefront size (32 or 64) of `symbol`'s kernel in a lifted program.
fn kernel_wave(program: &peacemaker_ir::Program, symbol: &str) -> Option<u32> {
    use peacemaker_ir::inst::Wave;
    program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .map(|k| match k.wave {
            Wave::Wave32 => 32,
            Wave::Wave64 => 64,
        })
}

/// The receipt's `numerics_model` for an arm (`arch` from the lifted program).
fn numerics_model(arm: &Value) -> Value {
    match arm["arch"].as_str() {
        None | Some("Gfx1100") => json!("gfx1100 program decoded with gfx1100 tables; WMMA/transcendental/NaN rules of the gfx1151 model (both arms identical; no claim about gfx1100 hardware ulps)"),
        Some(arch) => json!(format!(
            "{arch} program decoded and executed with its own tables and the emulator's native {arch} model (wave size {} from the kernel descriptor; both arms identical)",
            arm["wave"]
        )),
    }
}

/// The static hazard checks on one kernel of an object: `peacemaker_lift::hazards_gfx11` (every
/// non-advisory finding, i.e. Required or Unchecked, fails) and `peacemaker_ir::passes::hazards::analyze`
/// (an analysis error, or any obligation it reports, fails; the object's own arch and wave size
/// are used). `pass` is the verdict; the receipt carries every finding and the analysis result.
pub fn hazards(image: &[u8], symbol: &str) -> (bool, Value) {
    use peacemaker_ir::passes::hazards as pm;
    use peacemaker_lift::hazards_gfx11::{check_kernel, Severity};
    let (program, frontend) = match lift(image) {
        Ok(l) => l,
        Err(e) => {
            return (
                false,
                json!({"pass": false, "reason": format!("lift: {e}")}),
            )
        }
    };
    let arch = program.target.arch;
    let Some(kernel) = program.kernels.iter().find(|k| k.symbol.0 == symbol) else {
        return (
            false,
            json!({"pass": false, "reason": format!("the object does not define `{symbol}`")}),
        );
    };
    let findings = check_kernel(kernel, arch, true);
    let blocking: Vec<&_> = findings
        .iter()
        .filter(|f| f.severity != Severity::Advisory)
        .collect();
    let advisory = findings.len() - blocking.len();
    let analysis = pm::analyze(&kernel.body, arch, kernel.wave);
    let mut reasons: Vec<String> = Vec::new();
    if !blocking.is_empty() {
        reasons.push(format!(
            "{} required/unchecked hazard finding(s)",
            blocking.len()
        ));
    }
    let replay = match &analysis {
        Ok(a) => {
            if !a.obligations.is_empty() {
                reasons.push(format!(
                    "{} hazard obligation(s) from the PM IR replay",
                    a.obligations.len()
                ));
            }
            json!({"count": a.obligations.len(), "obligations": a.obligations.iter().map(|o| {
                let pc = o.insts.first().and_then(|id| kernel.body.insts.get(*id)).and_then(|i| i.prov.pc);
                json!({"rule_id": o.rule_id, "pc": pc.map(|p| format!("{p:#x}")), "text": o.text})
            }).collect::<Vec<_>>()})
        }
        Err(e) => {
            reasons.push(format!("hazard analysis error: {e}"));
            json!({"error": e.to_string()})
        }
    };
    let pass = reasons.is_empty();
    (
        pass,
        json!({
            "pass": pass,
            "reason": if pass { Value::Null } else { json!(reasons.join("; ")) },
            "arch": format!("{arch:?}"),
            "wave": kernel_wave(&program, symbol),
            "frontend": frontend,
            "findings_blocking": blocking.iter().map(|f| f.to_json()).collect::<Vec<_>>(),
            "findings_advisory_count": advisory,
            "pm_ir_hazard_replay": replay,
            "checker": "peacemaker_lift::hazards_gfx11::check_kernel + peacemaker_ir::passes::hazards::analyze",
        }),
    )
}

/// Which workgroups of the launch grid an arm runs.
pub enum WgSel<'a> {
    All,
    List(&'a [[u32; 3]]),
}

/// Emulate the selected workgroups of `geo` for `symbol` in `image` on `case`. The case's buffers are moved
/// into the emulator's regions (no copy: large zero-filled caches stay untouched pages) and moved back
/// afterwards, the output buffer reset to its pristine value, so the same case serves the next arm.
pub fn run_arm(
    tag: &str,
    image: &[u8],
    symbol: &str,
    geo: &Geometry,
    case: &mut EmuCase,
    sel: &WgSel,
) -> std::result::Result<ArmRun, ArmFail> {
    run_arm_model(tag, image, symbol, geo, case, sel, Model::Gate)
}

/// Which emulator model an arm runs under. Every gate path uses `Gate` (in-flight memory, LDS races, schedules).
/// `Legacy` (the pre-0.4.3 gate: strict LDS, one interleaving, loads retire at issue) and `StrictStoreSources`
/// (`Gate` plus the PM builder's store-source rule on gfx11) exist for tests and audits only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Model {
    Legacy,
    Gate,
    StrictStoreSources,
}

/// `run_arm` under an explicit [`Model`].
pub fn run_arm_model(
    tag: &str,
    image: &[u8],
    symbol: &str,
    geo: &Geometry,
    case: &mut EmuCase,
    sel: &WgSel,
    model: Model,
) -> std::result::Result<ArmRun, ArmFail> {
    debug_assert!(
        model == Model::Gate || cfg!(test),
        "only tests may run a non-gate emulator model"
    );
    let pristine = case.bufs[case.out].bytes.clone();
    let mut mem = Memory::default();
    let r = exec_arm(tag, image, symbol, geo, case, sel, &mut mem, model);
    let mut taken = None;
    for reg in std::mem::take(&mut mem.regions) {
        if let Some(i) = case.bufs.iter().position(|b| b.name == reg.name) {
            if i == case.out {
                taken = Some((reg.bytes, reg.written));
            } else {
                case.bufs[i].bytes = reg.bytes;
            }
        }
    }
    case.bufs[case.out].bytes = pristine;
    r.map(|mut run| {
        if let Some((bytes, written)) = taken {
            run.out = bytes;
            run.written = Some(written);
        }
        run
    })
}

fn exec_arm(
    tag: &str,
    image: &[u8],
    symbol: &str,
    geo: &Geometry,
    case: &mut EmuCase,
    sel: &WgSel,
    mem: &mut Memory,
    model: Model,
) -> std::result::Result<ArmRun, ArmFail> {
    let started = Instant::now();
    let mut info = json!({"tag": tag, "symbol": symbol, "object_sha256": sha256_hex(image)});
    let (program, frontend) = lift(image).map_err(|e| fail(format!("lift: {e}"), info.clone()))?;
    info["frontend"] = json!(frontend);
    info["arch"] = json!(format!("{:?}", program.target.arch));
    info["wave"] = json!(kernel_wave(&program, symbol));
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| {
            fail(
                format!("the object does not define `{symbol}`"),
                info.clone(),
            )
        })?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err(fail("not an HSA kernel", info));
    };
    info["group_segment_fixed_size"] = json!(descriptor.group_segment_fixed_size);
    info["private_segment_fixed_size"] = json!(descriptor.private_segment_fixed_size);
    info["kernarg_segment_size"] = json!(metadata.parsed.kernarg_segment_size);
    // Every opcode of the kernel the emulator cannot execute, named at once (fail closed).
    let mut missing = std::collections::BTreeMap::<&str, usize>::new();
    for id in &kernel.body.layout {
        let inst = kernel
            .body
            .insts
            .get(*id)
            .expect("layout names an instruction");
        match peacemaker_ir::isa::lookup(program.target.arch, inst.op, inst.form) {
            Some(row) if peacemaker_ir::emu::semantic_class(row.name).is_some() => {}
            Some(row) => *missing.entry(row.name).or_default() += 1,
            None => *missing.entry("<undecodable>").or_default() += 1,
        }
    }
    if !missing.is_empty() {
        let list: Vec<String> = missing.iter().map(|(n, c)| format!("{n} x{c}")).collect();
        return Err(fail(
            format!("no emulator semantics for: {}", list.join(", ")),
            info,
        ));
    }
    let emu = Emulator::new(&program, symbol)
        .map_err(|e| fail(format!("emulator: {e}"), info.clone()))?
        .strict_lds();
    let emu = match model {
        Model::Legacy => emu,
        Model::Gate => emu.hw_model(),
        Model::StrictStoreSources => emu.hw_model_strict_store_sources(),
    }
    .gfx1100_as_gfx1151();

    let mut addrs = Vec::with_capacity(case.bufs.len());
    for b in &mut case.bufs {
        let bytes = std::mem::take(&mut b.bytes);
        addrs.push(
            mem.map_guarded(b.name.to_owned(), bytes, GUARD_BYTES)
                .map_err(|e| fail(format!("map {}: {e}", b.name), info.clone()))?,
        );
    }
    let segment = metadata
        .parsed
        .kernarg_segment_size
        .max(descriptor.kernarg_size) as usize;
    let kern = kernarg_bytes(case, &addrs, &metadata.parsed.args, segment, geo)
        .map_err(|e| fail(e, info.clone()))?;
    // The compiler merges neighbouring kernarg dwords into one 16-byte `s_load_b128`, which can span past
    // the declared segment end (the runtime pads kernarg allocations): zero-pad the region to 16 bytes.
    let mut kern = kern;
    kern.resize(kern.len().next_multiple_of(16), 0);
    info["kernarg_bytes_mapped"] = json!(kern.len());
    let kva = mem
        .map_guarded("kernarg".to_owned(), kern, GUARD_BYTES)
        .map_err(|e| fail(format!("map kernarg: {e}"), info.clone()))?;

    let all: Vec<[u32; 3]>;
    let list: &[[u32; 3]] = match sel {
        WgSel::List(l) => l,
        WgSel::All => {
            let mut v = Vec::new();
            for z in 0..geo.grid[2] {
                for y in 0..geo.grid[1] {
                    for x in 0..geo.grid[0] {
                        v.push([x, y, z]);
                    }
                }
            }
            all = v;
            &all
        }
    };
    let (mut instructions, mut workgroups) = (0u64, 0u64);
    let mut schedules: Vec<String> = Vec::new();
    for &[x, y, z] in list {
        if x >= geo.grid[0] || y >= geo.grid[1] || z >= geo.grid[2] {
            return Err(ArmFail {
                reason: format!("workgroup ({x},{y},{z}) is outside the grid {:?}", geo.grid),
                instructions,
                workgroups,
                json: info,
            });
        }
        let launch = Launch {
            kernarg_ptr: kva,
            workgroup: [x, y, z],
            block: geo.block,
            dynamic_lds: geo.lds,
        };
        let r = catch_unwind(AssertUnwindSafe(|| emu.run_workgroup_report(&launch, mem)));
        match r {
            Ok(Ok(rep)) => {
                instructions += rep.instructions;
                workgroups += 1;
                if schedules.is_empty() {
                    schedules = rep.schedules;
                }
            }
            Ok(Err(e)) => {
                return Err(ArmFail {
                    reason: format!("workgroup ({x},{y},{z}): {e}"),
                    instructions,
                    workgroups,
                    json: info,
                });
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                return Err(ArmFail {
                    reason: format!("workgroup ({x},{y},{z}): emulator panic: {msg}"),
                    instructions,
                    workgroups,
                    json: info,
                });
            }
        }
    }
    let mut violations = Vec::new();
    for (i, b) in case.bufs.iter().enumerate() {
        let r = mem
            .regions
            .iter()
            .find(|r| r.name == b.name)
            .expect("every mapped region is still mapped");
        if i != case.out && r.written.count() != 0 {
            violations.push(format!(
                "wrote {} byte(s) of the read-only input buffer `{}`",
                r.written.count(),
                b.name
            ));
        }
    }
    if let Some(r) = mem.regions.iter().find(|r| r.name == "kernarg") {
        if r.written.count() != 0 {
            violations.push(format!(
                "wrote {} byte(s) of the kernarg segment",
                r.written.count()
            ));
        }
    }
    if !violations.is_empty() {
        return Err(ArmFail {
            reason: violations.join("; "),
            instructions,
            workgroups,
            json: info,
        });
    }
    let out_written = mem
        .regions
        .iter()
        .find(|r| r.name == case.bufs[case.out].name)
        .expect("the output region is mapped")
        .written
        .count();
    info["workgroups"] = json!(workgroups);
    info["instructions"] = json!(instructions);
    info["schedules"] = json!(schedules);
    info["geometry"] =
        json!({"grid_workgroups": geo.grid, "block": geo.block, "lds_dynamic_bytes": geo.lds});
    info["output_written_bytes"] = json!(out_written);
    info["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    Ok(ArmRun {
        out: Vec::new(),
        instructions,
        workgroups,
        json: info,
        out_written,
        written: None,
    })
}

/// First differing byte and the number of differing bytes.
fn diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), a.len().abs_diff(b.len())));
    }
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some((first, a.iter().zip(b).filter(|(x, y)| x != y).count()))
}

type ArmResult = std::result::Result<ArmRun, ArmFail>;

fn counts(r: &ArmResult) -> (u64, u64) {
    match r {
        Ok(a) => (a.instructions, a.workgroups),
        Err(f) => (f.instructions, f.workgroups),
    }
}

fn arm_json(r: &ArmResult) -> Value {
    match r {
        Ok(a) => a.json.clone(),
        Err(f) => f.json.clone(),
    }
}

fn failure_reasons(b: &ArmResult, c: &ArmResult) -> Option<String> {
    let mut why = Vec::new();
    if let Err(f) = b {
        why.push(format!("baseline: {}", f.reason));
    }
    if let Err(f) = c {
        why.push(format!("candidate: {}", f.reason));
    }
    (!why.is_empty()).then(|| why.join(" | "))
}

/// Run both arms over the FULL small grid and build the `emu_gate` receipt. `pass` is the verdict.
pub fn gate(
    symbol: &str,
    case: &mut EmuCase,
    base: (&[u8], &Geometry),
    cand: (&[u8], &Geometry),
    shape: Value,
    seed: u64,
) -> (bool, Value) {
    // The two arms are independent runs on the same inputs: the baseline on its own copy of the case, side by side.
    let mut base_case = case.clone();
    let (b, c) = std::thread::scope(|s| {
        let hb = s.spawn(|| {
            run_arm(
                "baseline",
                base.0,
                symbol,
                base.1,
                &mut base_case,
                &WgSel::All,
            )
        });
        let c = run_arm("candidate", cand.0, symbol, cand.1, case, &WgSel::All);
        (
            hb.join().unwrap_or_else(|_| {
                Err(fail(
                    "baseline arm thread panicked",
                    json!({"tag": "baseline"}),
                ))
            }),
            c,
        )
    });
    let (bn, cn) = (counts(&b), counts(&c));
    let mut rec = json!({
        "pass": false,
        "reason": Value::Null,
        "shape": shape,
        "seed": seed,
        "strict_lds": true,
        "guard_bytes": GUARD_BYTES,
        "numerics_model": numerics_model(&arm_json(&b)),
        "instructions": {"baseline": bn.0, "candidate": cn.0},
        "workgroups": {"baseline": bn.1, "candidate": cn.1},
        "arms": {"baseline": arm_json(&b), "candidate": arm_json(&c)},
        "output_bytes": case.bufs[case.out].bytes.len(),
        "output_sha256": {"baseline": Value::Null, "candidate": Value::Null},
        "identical_to_baseline": Value::Null,
    });
    if let Ok(a) = &b {
        rec["output_sha256"]["baseline"] = json!(sha256_hex(&a.out));
    }
    if let Ok(a) = &c {
        rec["output_sha256"]["candidate"] = json!(sha256_hex(&a.out));
    }
    if let Some(why) = failure_reasons(&b, &c) {
        rec["reason"] = json!(why);
        return (false, rec);
    }
    let (b, c) = (b.ok().unwrap(), c.ok().unwrap());
    if b.out_written == 0 {
        rec["reason"] =
            json!("the baseline wrote none of the output buffer: the launch is not exercised");
        return (false, rec);
    }
    let pristine = &case.bufs[case.out].bytes;
    if diff(pristine, &b.out).is_none() {
        rec["reason"] =
            json!("the baseline left the output buffer unchanged: the launch is not exercised");
        return (false, rec);
    }
    if let Some((first, n)) = diff(&b.out, &c.out) {
        rec["identical_to_baseline"] = json!(false);
        rec["reason"] = json!(format!(
            "output differs from the emulated incumbent: first differing byte {first}, {n} differing byte(s) of {}",
            b.out.len()
        ));
        return (false, rec);
    }
    rec["identical_to_baseline"] = json!(true);
    // Cross-workgroup order independence: the workgroups of one launch have no defined order, but the emulator
    // runs them sequentially, so a dependence between workgroups (one reads what another writes, e.g. an in-place
    // buffer) is invisible to a single pass. Re-run the candidate with the workgroup order reversed: both orders
    // must store the same bytes.
    let mut order: Vec<[u32; 3]> = Vec::new();
    for z in 0..cand.1.grid[2] {
        for y in 0..cand.1.grid[1] {
            for x in 0..cand.1.grid[0] {
                order.push([x, y, z]);
            }
        }
    }
    order.reverse();
    let r = run_arm(
        "candidate-reversed",
        cand.0,
        symbol,
        cand.1,
        case,
        &WgSel::List(&order),
    );
    let verdict = match &r {
        Err(f) => Err(format!("candidate with reversed workgroup order: {}", f.reason)),
        Ok(a) => match diff(&c.out, &a.out) {
            None => Ok(()),
            Some((first, n)) => Err(format!(
                "candidate output depends on workgroup order (reversed order: first differing byte {first}, {n} \
                 differing byte(s)): workgroups communicate through memory"
            )),
        },
    };
    rec["wg_order_reversed"] = json!({
        "pass": verdict.is_ok(),
        "output_sha256": r.as_ref().ok().map(|a| sha256_hex(&a.out)),
    });
    match verdict {
        Ok(()) => {
            rec["pass"] = json!(true);
            (true, rec)
        }
        Err(why) => {
            rec["reason"] = json!(why);
            (false, rec)
        }
    }
}

// ---------------------------------------------------------------------------
// Production-extent cases: sampled workgroups

/// Output-owner map: which workgroup writes output element `e` (4-byte elements).
pub type Owner<'a> = &'a dyn Fn(usize) -> [u32; 3];

/// Upper bound on the workgroups either arm runs in one extent case (the sampled set plus the closure
/// that makes both arms cover the same output elements).
pub const MAX_EXTENT_WGS: usize = 2000;

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The sampled workgroups of a grid: every corner (first/last along each dim), the last tile of each dim
/// next to the first/middle tiles of the others (the partial-dim edges), and eight seeded random ones.
pub fn sample_wgs(grid: [u32; 3], seed: u64, label: &str) -> Vec<[u32; 3]> {
    let mut set = BTreeSet::new();
    let last = [grid[0] - 1, grid[1] - 1, grid[2] - 1];
    let mid = [grid[0] / 2, grid[1] / 2, grid[2] / 2];
    for m in 0..8u32 {
        set.insert([
            if m & 1 != 0 { last[0] } else { 0 },
            if m & 2 != 0 { last[1] } else { 0 },
            if m & 4 != 0 { last[2] } else { 0 },
        ]);
    }
    for d in 0..3 {
        let (a, b) = ((d + 1) % 3, (d + 2) % 3);
        for m in 0..4u32 {
            let mut w = [0u32; 3];
            w[d] = last[d];
            w[a] = if m & 1 != 0 { mid[a] } else { 0 };
            w[b] = if m & 2 != 0 { mid[b] } else { 0 };
            set.insert(w);
        }
    }
    let mut r = Xs64::new(seed ^ fnv(label));
    for _ in 0..8 {
        set.insert([
            (r.next_u64() % u64::from(grid[0])) as u32,
            (r.next_u64() % u64::from(grid[1])) as u32,
            (r.next_u64() % u64::from(grid[2])) as u32,
        ]);
    }
    set.into_iter().collect()
}

fn lin(g: [u32; 3], w: [u32; 3]) -> Option<usize> {
    (w[0] < g[0] && w[1] < g[1] && w[2] < g[2])
        .then(|| w[0] as usize + g[0] as usize * (w[1] as usize + g[1] as usize * w[2] as usize))
}

fn grid_len(g: [u32; 3]) -> usize {
    g[0] as usize * g[1] as usize * g[2] as usize
}

fn wg_list(g: [u32; 3], picked: &[bool]) -> Vec<[u32; 3]> {
    picked
        .iter()
        .enumerate()
        .filter(|(_, p)| **p)
        .map(|(i, _)| {
            let (gx, gy) = (g[0] as usize, g[1] as usize);
            [
                (i % gx) as u32,
                (i / gx % gy) as u32,
                (i / (gx * gy)) as u32,
            ]
        })
        .collect()
}

/// One production-extent case. Both arms run only SAMPLED workgroups of their own grids, chosen so that
/// they cover the same set `E` of output elements: the baseline's sampled workgroups (`sample_wgs`) give
/// elements, the candidate workgroups owning them give a larger element set, the baseline workgroups
/// owning THAT give the final set `E`. `E` is compared bitwise between the arms. Each arm must have
/// written every element of `E` and nothing outside its own workgroups' elements.
pub fn extent_gate(
    label: &str,
    symbol: &str,
    case: &mut EmuCase,
    base: (&[u8], &Geometry, Owner),
    cand: (&[u8], &Geometry, Owner),
    shape: Value,
    seed: u64,
) -> (bool, Value) {
    let (gb, gc) = (base.1.grid, cand.1.grid);
    let nelem = case.bufs[case.out].bytes.len() / 4;
    let mut rec = json!({
        "label": label, "pass": false, "reason": Value::Null, "shape": shape, "seed": seed,
        "grid": {"baseline": gb, "candidate": gc},
        "output_elements": nelem,
    });
    let stop = |rec: &mut Value, why: String| {
        rec["reason"] = json!(why);
        (false, rec.clone())
    };
    if case.bufs[case.out].bytes.len() % 4 != 0 {
        return stop(
            &mut rec,
            "output buffer is not a whole number of 4-byte elements".into(),
        );
    }
    // 1. the baseline's sampled workgroups -> their output elements -> the candidate workgroups owning them.
    let sampled = sample_wgs(gb, seed, label);
    let mut sel_b = vec![false; grid_len(gb)];
    for w in &sampled {
        sel_b[lin(gb, *w).expect("sample is inside the grid")] = true;
    }
    let mut sel_c = vec![false; grid_len(gc)];
    for e in 0..nelem {
        if (base.2)(e) == PAD_OWNER {
            continue;
        }
        let Some(lb) = lin(gb, (base.2)(e)) else {
            return stop(
                &mut rec,
                format!("baseline geometry does not cover output element {e}"),
            );
        };
        if sel_b[lb] {
            let Some(lc) = lin(gc, (cand.2)(e)) else {
                return stop(&mut rec, format!("candidate geometry does not cover output element {e} (owner {:?}, grid {gc:?})", (cand.2)(e)));
            };
            sel_c[lc] = true;
        }
    }
    // 2. the elements those candidate workgroups own (E) -> the baseline workgroups owning E.
    let mut in_e = vec![false; nelem];
    let mut sel_b2 = vec![false; grid_len(gb)];
    let mut n_e = 0usize;
    for e in 0..nelem {
        if (base.2)(e) == PAD_OWNER || (cand.2)(e) == PAD_OWNER {
            continue;
        }
        let Some(lc) = lin(gc, (cand.2)(e)) else {
            return stop(&mut rec, format!("candidate geometry does not cover output element {e} (owner {:?}, grid {gc:?})", (cand.2)(e)));
        };
        if sel_c[lc] {
            in_e[e] = true;
            n_e += 1;
            sel_b2[lin(gb, (base.2)(e)).expect("checked")] = true;
        }
    }
    let (wb, wc) = (wg_list(gb, &sel_b2), wg_list(gc, &sel_c));
    rec["sampled_wgs"] =
        json!({"baseline_seed": sampled.len(), "baseline": wb.len(), "candidate": wc.len()});
    rec["elements_compared"] = json!(n_e);
    if wb.len() > MAX_EXTENT_WGS || wc.len() > MAX_EXTENT_WGS {
        return stop(&mut rec, format!("the closure of the sampled workgroups is too large ({} baseline, {} candidate; limit {MAX_EXTENT_WGS})", wb.len(), wc.len()));
    }
    rec["first_wgs"] = json!({"baseline": wb.iter().take(6).collect::<Vec<_>>(), "candidate": wc.iter().take(6).collect::<Vec<_>>()});
    // 3. run both arms on their lists.
    let b = run_arm("baseline", base.0, symbol, base.1, case, &WgSel::List(&wb));
    let c = run_arm("candidate", cand.0, symbol, cand.1, case, &WgSel::List(&wc));
    let (bn, cn) = (counts(&b), counts(&c));
    rec["instructions"] = json!({"baseline": bn.0, "candidate": cn.0});
    rec["workgroups"] = json!({"baseline": bn.1, "candidate": cn.1});
    rec["arms"] = json!({"baseline": arm_json(&b), "candidate": arm_json(&c)});
    if let Some(why) = failure_reasons(&b, &c) {
        return stop(&mut rec, why);
    }
    let (b, c) = (b.ok().unwrap(), c.ok().unwrap());
    let (Some(wrb), Some(wrc)) = (&b.written, &c.written) else {
        return stop(&mut rec, "internal: no written set".into());
    };
    // 4. write discipline, coverage of E, bitwise identity on E.
    let pristine = &case.bufs[case.out].bytes;
    let mut sha_b = sha2::Sha256::default();
    let mut sha_c = sha2::Sha256::default();
    let mut changed = false;
    for e in 0..nelem {
        if (base.2)(e) == PAD_OWNER || (cand.2)(e) == PAD_OWNER {
            for (arm, wr) in [("baseline", wrb), ("candidate", wrc)] {
                if (0..4).any(|k| wr.contains(e * 4 + k)) {
                    rec["identical_to_baseline"] = json!(false);
                    return stop(&mut rec, format!("{arm} wrote the pad element {e} between packed output segments (an out-of-segment store)"));
                }
            }
            continue;
        }
        for (arm, wr, grid, owner, list) in [
            ("baseline", wrb, gb, base.2, &sel_b2),
            ("candidate", wrc, gc, cand.2, &sel_c),
        ] {
            if wr.contains(e * 4) && !list[lin(grid, owner(e)).expect("covered")] {
                return stop(&mut rec, format!("{arm} wrote output element {e} outside the elements of its sampled workgroups"));
            }
        }
        if !in_e[e] {
            continue;
        }
        for (arm, wr) in [("baseline", wrb), ("candidate", wrc)] {
            if !(0..4).all(|k| wr.contains(e * 4 + k)) {
                return stop(
                    &mut rec,
                    format!(
                        "{arm} did not write output element {e} (owner workgroup {:?})",
                        if arm == "baseline" {
                            (base.2)(e)
                        } else {
                            (cand.2)(e)
                        }
                    ),
                );
            }
        }
        let r = e * 4..e * 4 + 4;
        use sha2::Digest;
        sha_b.update(&b.out[r.clone()]);
        sha_c.update(&c.out[r.clone()]);
        changed |= b.out[r.clone()] != pristine[r.clone()];
        if b.out[r.clone()] != c.out[r.clone()] {
            let (x, y) = (
                u32::from_le_bytes(b.out[r.clone()].try_into().unwrap()),
                u32::from_le_bytes(c.out[r.clone()].try_into().unwrap()),
            );
            rec["identical_to_baseline"] = json!(false);
            return stop(&mut rec, format!("output element {e} differs from the emulated incumbent: baseline {x:#010x}, candidate {y:#010x} (baseline owner {:?}, candidate owner {:?})", (base.2)(e), (cand.2)(e)));
        }
    }
    use sha2::Digest;
    rec["output_sha256"] = json!({"baseline": format!("{:x}", sha_b.finalize()), "candidate": format!("{:x}", sha_c.finalize())});
    if !changed {
        return stop(&mut rec, "the baseline left every compared element at its pristine value: the launch is not exercised".into());
    }
    rec["identical_to_baseline"] = json!(true);
    rec["pass"] = json!(true);
    (true, rec)
}
