// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// One-card gfx1201 hardware smoke of railgun's persistent-program completion
// cancel (lab only; run through the fabric on ONE gfx1201 card):
//
//   cargo run --release --features lab -p rdna-compute \
//     --example railgun_persistent_cancel_smoke -- OUT.json [--arm-b-unsafe-escalation]
//
// Three PeaceMaker-emitted kernels (one wave32 workgroup each, emitted below
// through the public `peacemaker_isa::Builder`, certified by `railgun-cert`,
// registered through `Gpu::railgun_register_pinned_object`) exercise
// `submit_queue_nowait_watched` against a host-mapped poison cell
// (`hipMallocSignalMemory`, `hip_bridge::SignalWord`):
//
//   * `poll`   (arm A): lane 0 stores `ENTERED` to out[0], then spins on the
//     poison cell with a device-scope acquire `event_wait` (unsigned >= 1, i.e.
//     any non-zero value); on exit it stores `EXITED` to out[1] and ends. It
//     never ends on its own, so the watched completion MUST time out, the host
//     MUST store the poison value, and the window MUST retire:
//     `FailureKind::Poisoned` with `Quiescence::Proven`, wall time in
//     [budget, budget + grace + slack), out = [ENTERED, EXITED]. The same queue
//     is then reused (cell zeroed, out zeroed, second watched submit, same
//     result), and a second, ordinary unwatched submit of the finite `done`
//     kernel (lane 0 stores `DONE` to out[2]) on its own queue MUST complete
//     Ok afterwards. The device MUST NOT be quarantined throughout.
//     THE VISIBILITY CAVEAT: whether a device-scope acquire poll observes the
//     host's store into fine-grained signal memory is not provable from the
//     source; arm A proves it on hardware (a failure here reads as a TornDown
//     outcome instead of Poisoned).
//   * `ignore` (arm B, ONLY with `--arm-b-unsafe-escalation`, run LAST): never
//     exits for a poison of 1. It polls the cell through an `event_wait` whose
//     threshold is 0xFFFFFFFF, so the cell is read but the poison value never
//     satisfies it (a pure `s_sleep` + `s_branch` loop with no reachable
//     `s_endpgm` is rejected by `railgun_cert::certify`: "no reachable
//     s_endpgm"; the host-only test below pins that every probe certifies).
//     The watched wait MUST fail with
//     `FailureKind::TornDown` (never Ok, never Poisoned): the queue is torn
//     down. If retirement was observed (`Quiescence::Proven`) the device stays
//     usable and the torn queue refuses with TornDown; if not (`Unknown`) the
//     device is quarantined until process restart, and every later registration,
//     lowering and submit refuses with the recorded reason. Either case is
//     printed. Arm B can leave the card busy: the process exits right after
//     writing the report, without running destructors, so the card needs a
//     process restart (and possibly more) afterwards.
//
// Usage: railgun_persistent_cancel_smoke OUT.json [--arm-b-unsafe-escalation]
// Exit 0 only when every assertion held.

use std::sync::Arc;
use std::time::{Duration, Instant};

use hip_bridge::DeviceBuffer;
use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::plan::Access;
use peacemaker_isa::reg::{Kind, Live, RegRef};
use peacemaker_isa::sync::{GlobalAddr, Ordering, Scope};
use peacemaker_isa::{native, Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, S, V};
use railgun::kernel::{author, KernelProgram, LaunchRecord, NodeFacts};
use railgun::plan::{plan, CacheTable, Pm4Plan, RowPolicy};
use railgun::watchdog::{FailureKind, ResolvedWatchdog};
use railgun::window::{
    submit_queue_nowait, submit_queue_nowait_watched, NodeImage, QueueFailure, Quiescence,
};
use railgun::Binding;
use rdna_compute::railgun_window::{HostPoisonCell, RailgunLoweringContext, RailgunWindowQueue};
use serde_json::{json, Value};

const MODULE_POLL: &str = "pm_persist_cancel_probe_poll_gfx1201";
const MODULE_IGNORE: &str = "pm_persist_cancel_probe_ignore_gfx1201";
const MODULE_DONE: &str = "pm_persist_cancel_probe_done_gfx1201";
const SYMBOL_POLL: &str = "pm_persist_cancel_probe_poll";
const SYMBOL_IGNORE: &str = "pm_persist_cancel_probe_ignore";
const SYMBOL_DONE: &str = "pm_persist_cancel_probe_done";
const KERNARG_BYTES: u32 = 24;

/// out[0]: lane 0 entered the kernel. out[1]: the poll arm saw the poison and exited.
/// out[2]: the finite kernel ran.
const ENTERED: u32 = 0xE17E_0001;
const EXITED: u32 = 0xE17E_00A5;
const DONE: u32 = 0xD0DE_0001;

const BUDGET: Duration = Duration::from_millis(300);
const GRACE: Duration = Duration::from_millis(300);
const ESCALATION: Duration = Duration::from_millis(500);
/// Scheduling slack on top of `budget + grace` for the poisoned wall-time bound.
const SLACK: Duration = Duration::from_millis(500);
const POISON_VALUE: u32 = 1;

// ---------------------------------------------------------------- the kernels

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Probe {
    /// Arm A: spins on the poison cell, exits when it is non-zero.
    Poll,
    /// Arm B: polls the cell with an unreachable threshold (0xFFFFFFFF), so a
    /// poison of 1 is never observed as satisfying it; never exits.
    Ignore,
    /// Finite follow-up kernel.
    Done,
}

impl Probe {
    fn symbol(self) -> &'static str {
        match self {
            Probe::Poll => SYMBOL_POLL,
            Probe::Ignore => SYMBOL_IGNORE,
            Probe::Done => SYMBOL_DONE,
        }
    }
    fn module(self) -> &'static str {
        match self {
            Probe::Poll => MODULE_POLL,
            Probe::Ignore => MODULE_IGNORE,
            Probe::Done => MODULE_DONE,
        }
    }
    fn variant(self) -> &'static str {
        match self {
            Probe::Poll => "poll",
            Probe::Ignore => "ignore_poll_unreachable",
            Probe::Done => "done",
        }
    }
}

fn s(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 1,
    }
}
fn v(n: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    }
}
fn sp(n: u8) -> RegRef {
    RegRef {
        kind: Kind::S,
        base: n,
        len: 2,
    }
}
fn op(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
fn smem(b: &mut Builder, text: &str, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(MemoryClass::SmemLoad))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}
/// `out[offset / 4] = v2` by lane 0 (v1 = 0 is the address VGPR).
fn store_out(b: &mut Builder, offset: u32) -> Result<(), String> {
    b.push(
        Instruction::new(
            format!("global_store_b32 v1, v2, s[6:7] offset:{offset}"),
            vec![],
            vec![v(1), v(2), sp(6)],
        )
        .memory(MemoryClass::VmemStore),
    )
}

fn probe_spec(probe: Probe) -> KernelSpec {
    KernelSpec {
        kernel_id: "pm_persist_cancel_probe".into(),
        variant: probe.variant().into(),
        arch: Arch::Gfx1201,
        symbol: probe.symbol().into(),
        kernargs: KernargLayout::new(KERNARG_BYTES)
            .pointer_access("poison", 0, Access::ReadOnly)
            .pointer_access("out", 8, Access::WriteOnly)
            .hidden("mode", 16, 4, "by_value"),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// Kernarg: `poison` ptr at 0, `out` ptr at 8, `mode` word at 16 (unused).
fn emit(probe: Probe) -> Result<Emitted, String> {
    let mut regs = RegPlan::new(64, 48)?;
    for (name, base, len) in [
        ("kernarg", 0u8, 2u8),
        ("poison", 4, 2),
        ("out", 6, 2),
        ("expected", 9, 1),
        ("observed", 10, 1),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    for (name, base) in [("lane", 0u8), ("zero", 1), ("value", 2), ("poll", 3)] {
        regs.add_range(name, Kind::V, base, 1, Live::Whole)?;
    }
    let mut b = Builder::new(probe_spec(probe), regs);
    smem(
        &mut b,
        "s_load_b128 s[4:7], s[0:1], 0x0",
        &[sp(4), sp(6)],
        &[sp(0)],
    )?;
    // The poll threshold: unsigned `>= 1` (any non-zero value) for arm A; the
    // unreachable 0xFFFFFFFF for arm B.
    let expected = if probe == Probe::Ignore { "-1" } else { "1" };
    op(&mut b, &format!("s_mov_b32 s9, {expected}"), &[s(9)], &[])?;
    op(&mut b, "s_mov_b32 exec_lo, 1", &[], &[])?;
    op(&mut b, "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    if probe == Probe::Done {
        op(
            &mut b,
            &format!("v_mov_b32_e32 v2, {DONE:#x}"),
            &[v(2)],
            &[],
        )?;
        b.wait_all()?;
        store_out(&mut b, 8)?;
        b.wait_all()?;
        branch(&mut b, "s_endpgm")?;
        return b.finish();
    }
    op(
        &mut b,
        &format!("v_mov_b32_e32 v2, {ENTERED:#x}"),
        &[v(2)],
        &[],
    )?;
    b.wait_all()?;
    store_out(&mut b, 0)?;
    // The marker is complete (store counter drained) before the kernel can spin.
    b.wait_all()?;
    match probe {
        Probe::Poll | Probe::Ignore => {
            b.event_wait(
                ".Lspin",
                GlobalAddr {
                    vaddr: V::<1>(1),
                    base: S::<2>(4),
                    offset: 0,
                },
                S::<1>(9),
                V::<1>(3),
                S::<1>(10),
                Scope::Dev,
                Ordering::Acquire,
                8,
                None,
            )?;
            if probe == Probe::Poll {
                op(
                    &mut b,
                    &format!("v_mov_b32_e32 v2, {EXITED:#x}"),
                    &[v(2)],
                    &[],
                )?;
                b.wait_all()?;
                store_out(&mut b, 4)?;
                b.wait_all()?;
            }
        }
        Probe::Done => unreachable!(),
    }
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

struct Built {
    probe: Probe,
    elf: Vec<u8>,
    sha: String,
    receipt_json: Vec<u8>,
    instructions: usize,
}

fn build_object(probe: Probe) -> Result<Built, String> {
    let emitted = emit(probe).map_err(|e| format!("emit {}: {e}", probe.variant()))?;
    let elf = native::assemble(&emitted.s_text, Arch::Gfx1201)
        .map_err(|e| format!("assemble {}: {e}", probe.variant()))?;
    let sha = railgun_corpus::sha256_hex(&elf);
    let receipt = railgun_cert::certify(&elf, None)
        .map_err(|e| format!("certify {}: {e}", probe.variant()))?;
    let receipt_json = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
    Ok(Built {
        probe,
        elf,
        sha,
        receipt_json,
        instructions: emitted.shape.instructions,
    })
}

fn register(gpu: &mut rdna_compute::Gpu, built: &Built) -> Result<Arc<NodeFacts>, String> {
    let image: Arc<[u8]> = Arc::from(built.elf.as_slice());
    Ok(gpu
        .railgun_register_pinned_object(
            built.probe.module(),
            image,
            &built.sha,
            &built.receipt_json,
            &[built.probe.symbol()],
        )?
        .remove(0))
}

// ---------------------------------------------------------------- lowering

struct Lowered {
    queue: RailgunWindowQueue,
    program: KernelProgram,
    pm4: Pm4Plan,
    image: Vec<u8>,
    grid: [u32; 3],
    /// True when `railgun_probe_allocation` could not resolve the poison
    /// (signal-memory) pointer and the binding was constructed by hand.
    poison_binding_by_hand: bool,
}

fn kernarg(poison: u64, out: u64) -> Vec<u8> {
    let mut ka = vec![0u8; KERNARG_BYTES as usize];
    ka[0..8].copy_from_slice(&poison.to_le_bytes());
    ka[8..16].copy_from_slice(&out.to_le_bytes());
    ka
}

#[allow(clippy::too_many_arguments)]
fn lower_probe(
    gpu: &rdna_compute::Gpu,
    ctx: &mut RailgunLoweringContext,
    table: &CacheTable,
    facts: &Arc<NodeFacts>,
    symbol: &str,
    ka: &[u8],
    poison: Binding,
) -> Result<Lowered, String> {
    let mut allocations = Vec::new();
    let mut poison_binding_by_hand = false;
    for off in facts.facts.pointer_offsets() {
        let o = off as usize;
        let addr = u64::from_le_bytes(ka[o..o + 8].try_into().unwrap());
        let binding = match gpu.railgun_probe_allocation(addr) {
            Ok(b) => Ok(b),
            // hipMemGetAddressRange cannot describe signal memory: bind the
            // poison word by hand (its exact base and size).
            Err(_) if addr == poison.base => {
                poison_binding_by_hand = true;
                Ok(poison)
            }
            Err(e) => Err(e),
        };
        allocations.push((off, binding));
    }
    let program = author(
        &gpu.arch,
        &[LaunchRecord {
            symbol: symbol.to_owned(),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            dynamic_lds: 0,
            kernarg: ka.to_vec(),
            declared: Vec::new(),
            facts: Ok(Arc::clone(facts)),
            allocations,
        }],
    );
    let node = &program.nodes[0];
    if !node.lowering.is_pm4() {
        return Err(format!("{symbol}: node is not PM4: {:?}", node.lowering));
    }
    let grid = node.grid;
    let segment = node.segment_bytes.ok_or("no certified kernarg segment")? as usize;
    let words = program.authored_words();
    let image = program.kernarg_image(0, segment, &words)?;
    let pm4 = plan(&program, table, RowPolicy::UnreceiptedSystem);
    if !pm4.is_single_segment(program.nodes.len()) {
        return Err(format!("{symbol}: {} PM4 segments", pm4.segments.len()));
    }
    let queue = gpu.lower_railgun_window(ctx, &program, &pm4)?;
    Ok(Lowered {
        queue,
        program,
        pm4,
        image,
        grid,
        poison_binding_by_hand,
    })
}

// ---------------------------------------------------------------- helpers

fn check(fails: &mut Vec<String>, ok: bool, msg: impl Into<String>) {
    if !ok {
        let msg = msg.into();
        eprintln!("FAIL: {msg}");
        fails.push(msg);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn read_out(gpu: &rdna_compute::Gpu, out: &DeviceBuffer) -> Result<[u32; 4], String> {
    let mut bytes = [0u8; 16];
    gpu.hip
        .memcpy_dtoh(&mut bytes, out)
        .map_err(|e| e.to_string())?;
    let mut w = [0u32; 4];
    for (i, c) in bytes.chunks(4).enumerate() {
        w[i] = u32::from_le_bytes(c.try_into().unwrap());
    }
    Ok(w)
}

fn zero_out(gpu: &rdna_compute::Gpu, out: &DeviceBuffer) -> Result<(), String> {
    gpu.hip
        .memcpy_htod(out, &[0u8; 16])
        .map_err(|e| e.to_string())
}

fn run_watched(
    queue: &mut RailgunWindowQueue,
    nodes: &[NodeImage<'_>],
    wd: &ResolvedWatchdog,
) -> (Result<(), QueueFailure>, Duration) {
    let t0 = Instant::now();
    let r = match submit_queue_nowait_watched(queue, nodes, &[], wd) {
        Ok(completion) => completion.wait(),
        Err(f) => Err(f),
    };
    (r, t0.elapsed())
}

fn describe(r: &Result<(), QueueFailure>, dt: Duration) -> Value {
    match r {
        Ok(()) => json!({"ok": true, "millis": ms(dt)}),
        Err(f) => json!({
            "ok": false,
            "kind": format!("{:?}", f.kind),
            "quiescence": format!("{:?}", f.quiescence),
            "message": f.message,
            "millis": ms(dt),
        }),
    }
}

fn object_json(built: &Built, facts: &NodeFacts) -> Value {
    json!({
        "module": built.probe.module(),
        "symbol": built.probe.symbol(),
        "variant": built.probe.variant(),
        "object_sha256": built.sha,
        "instructions": built.instructions,
        "receipt_sha256": facts.kref.receipt_sha256,
        "receipt_tier": facts.kref.tier,
        "obligations": facts.kref.obligations,
    })
}

fn write_report(out: &str, report: &Value) -> Result<(), String> {
    std::fs::write(out, serde_json::to_string_pretty(report).unwrap())
        .map_err(|e| format!("{out}: {e}"))
}

fn finish(pass: bool) -> std::process::ExitCode {
    if pass {
        println!("railgun_persistent_cancel_smoke: PASS");
        std::process::ExitCode::SUCCESS
    } else {
        println!("railgun_persistent_cancel_smoke: FAIL");
        std::process::ExitCode::FAILURE
    }
}

// ---------------------------------------------------------------- the smoke

fn run(out_path: &str, arm_b: bool) -> Result<bool, String> {
    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
    if gpu.arch != "gfx1201" {
        return Err(format!(
            "device is {}; this smoke runs only on gfx1201",
            gpu.arch
        ));
    }
    if let Some(reason) = gpu.railgun_device_quarantine() {
        return Err(format!("device is already quarantined: {reason}"));
    }
    let mut fails: Vec<String> = Vec::new();

    // The poison cell (host-mapped hipMallocSignalMemory, zeroed) and the
    // output buffer. `poison_word` is declared before every queue so it drops
    // after them.
    let poison_word = gpu
        .hip
        .signal_word()
        .map_err(|e| format!("signal_word: {e}"))?;
    check(
        &mut fails,
        poison_word.load() == 0,
        "poison cell is not zero at start",
    );
    let queue_cell = HostPoisonCell::from_signal_word(&poison_word);
    let host_cell = HostPoisonCell::from_signal_word(&poison_word);
    let poison_addr = queue_cell.device_address();
    let poison_binding = Binding {
        base: poison_addr,
        bytes: poison_word.device_buffer().size() as u64,
    };
    let out_buf = gpu.hip.malloc(16).map_err(|e| e.to_string())?;
    zero_out(&gpu, &out_buf)?;
    let ka = kernarg(poison_addr, out_buf.as_ptr() as u64);
    let wd = ResolvedWatchdog::new(BUDGET, GRACE, ESCALATION, poison_addr, POISON_VALUE)?;

    let table = CacheTable::for_arch(&gpu.arch).ok_or("no gfx1201 cache table")?;
    let mut ctx = gpu.railgun_lowering_context()?;

    // ---------------------------------------------------------------- arm A
    let built_a = build_object(Probe::Poll)?;
    let facts_a = register(&mut gpu, &built_a)?;
    println!(
        "registered {} object {} tier {}",
        Probe::Poll.module(),
        built_a.sha,
        facts_a.kref.tier
    );
    let mut low_a = lower_probe(
        &gpu,
        &mut ctx,
        &table,
        &facts_a,
        Probe::Poll.symbol(),
        &ka,
        poison_binding,
    )?;
    low_a.queue.set_poison_cell(queue_cell);
    let nodes_a = [NodeImage {
        kernarg: &low_a.image,
        grid: low_a.grid,
    }];

    let mut reps = Vec::new();
    for rep in 0..2 {
        host_cell.store(0);
        zero_out(&gpu, &out_buf)?;
        let (r, dt) = run_watched(&mut low_a.queue, &nodes_a, &wd);
        let words = read_out(&gpu, &out_buf)?;
        let (kind, quiescence) = match &r {
            Ok(()) => (None, None),
            Err(f) => (Some(f.kind), Some(f.quiescence)),
        };
        check(
            &mut fails,
            kind == Some(FailureKind::Poisoned),
            format!(
                "arm A rep {rep}: expected Err(Poisoned), got {}; if the host store was not seen by the device-scope acquire poll the window was torn down instead",
                match &r {
                    Ok(()) => "Ok".to_owned(),
                    Err(f) => format!("{:?}: {}", f.kind, f.message),
                }
            ),
        );
        check(
            &mut fails,
            quiescence == Some(Quiescence::Proven),
            format!("arm A rep {rep}: quiescence {quiescence:?}, expected Proven"),
        );
        check(
            &mut fails,
            dt >= BUDGET,
            format!(
                "arm A rep {rep}: completed in {:.1} ms, before the {BUDGET:?} budget",
                ms(dt)
            ),
        );
        check(
            &mut fails,
            dt < BUDGET + GRACE + SLACK,
            format!(
                "arm A rep {rep}: took {:.1} ms, beyond budget + grace + slack",
                ms(dt)
            ),
        );
        check(
            &mut fails,
            words[0] == ENTERED,
            format!(
                "arm A rep {rep}: out[0] = {:#x}, expected {ENTERED:#x}",
                words[0]
            ),
        );
        check(
            &mut fails,
            words[1] == EXITED,
            format!(
                "arm A rep {rep}: out[1] = {:#x}, expected {EXITED:#x} (the kernel saw the poison and exited)",
                words[1]
            ),
        );
        check(
            &mut fails,
            gpu.railgun_device_quarantine().is_none(),
            format!("arm A rep {rep}: device quarantined"),
        );
        println!(
            "arm A rep {rep}: {} out {words:x?} {:.1} ms",
            match &r {
                Ok(()) => "Ok".to_owned(),
                Err(f) => format!("{:?}/{:?}", f.kind, f.quiescence),
            },
            ms(dt)
        );
        let mut entry = describe(&r, dt);
        entry["rep"] = json!(rep);
        entry["out_words"] = json!(words);
        reps.push(entry);
    }

    // A following ordinary submit works after the poison cancels: the finite
    // `done` kernel on its own queue, unwatched.
    host_cell.store(0);
    zero_out(&gpu, &out_buf)?;
    let built_d = build_object(Probe::Done)?;
    let facts_d = register(&mut gpu, &built_d)?;
    let mut low_d = lower_probe(
        &gpu,
        &mut ctx,
        &table,
        &facts_d,
        Probe::Done.symbol(),
        &ka,
        poison_binding,
    )?;
    let nodes_d = [NodeImage {
        kernarg: &low_d.image,
        grid: low_d.grid,
    }];
    let t0 = Instant::now();
    let normal = match submit_queue_nowait(&mut low_d.queue, &nodes_d, &[]) {
        Ok(c) => c.wait(),
        Err(f) => Err(f),
    };
    let normal_dt = t0.elapsed();
    let words = read_out(&gpu, &out_buf)?;
    check(
        &mut fails,
        normal.is_ok(),
        format!(
            "normal follow-up submit failed: {:?}",
            normal.as_ref().err()
        ),
    );
    check(
        &mut fails,
        words[2] == DONE,
        format!(
            "normal follow-up: out[2] = {:#x}, expected {DONE:#x}",
            words[2]
        ),
    );
    check(
        &mut fails,
        gpu.railgun_device_quarantine().is_none(),
        "device quarantined after arm A",
    );
    println!("normal follow-up: {:?} out {words:x?}", normal.is_ok());
    let mut normal_json = describe(&normal, normal_dt);
    normal_json["out_words"] = json!(words);

    let mut report = json!({
        "device_arch": gpu.arch,
        "budget_ms": ms(BUDGET),
        "grace_ms": ms(GRACE),
        "escalation_grace_ms": ms(ESCALATION),
        "poison_address": format!("{poison_addr:#x}"),
        "poison_binding_by_hand": low_a.poison_binding_by_hand || low_d.poison_binding_by_hand,
        "objects": {
            "poll": object_json(&built_a, &facts_a),
            "done": object_json(&built_d, &facts_d),
        },
        "arm_a": {"reps": reps, "normal_follow_up": normal_json},
        "arm_b": if arm_b { json!("pending") } else { json!("not run (no --arm-b-unsafe-escalation)") },
        "failures": fails,
        "pass": fails.is_empty(),
    });
    write_report(out_path, &report)?;
    if !arm_b {
        drop(low_d);
        drop(low_a);
        gpu.hip.free(out_buf).map_err(|e| e.to_string())?;
        return Ok(fails.is_empty());
    }

    // ---------------------------------------------------------------- arm B
    // LAST: the poison is ignored, so the watchdog escalates to queue teardown.
    println!("arm B: poison-ignoring kernel (unsafe escalation); the card may need a process restart afterwards");
    host_cell.store(0);
    zero_out(&gpu, &out_buf)?;
    let built_b = build_object(Probe::Ignore)?;
    let facts_b = register(&mut gpu, &built_b)?;
    let mut low_b = lower_probe(
        &gpu,
        &mut ctx,
        &table,
        &facts_b,
        Probe::Ignore.symbol(),
        &ka,
        poison_binding,
    )?;
    low_b
        .queue
        .set_poison_cell(HostPoisonCell::from_signal_word(&poison_word));
    let nodes_b = [NodeImage {
        kernarg: &low_b.image,
        grid: low_b.grid,
    }];
    let (rb, dt_b) = run_watched(&mut low_b.queue, &nodes_b, &wd);
    let words_b = read_out(&gpu, &out_buf).unwrap_or([0; 4]);
    let mut arm_b_json = describe(&rb, dt_b);
    arm_b_json["variant"] = json!(built_b.probe.variant());
    arm_b_json["out_words"] = json!(words_b);
    arm_b_json["object"] = object_json(&built_b, &facts_b);
    arm_b_json["lowered_poison_binding_by_hand"] = json!(low_b.poison_binding_by_hand);

    let outcome = match &rb {
        Ok(()) => {
            check(
                &mut fails,
                false,
                "arm B: wait returned Ok for a kernel that never exits",
            );
            "unexpected_ok"
        }
        Err(f) => {
            check(
                &mut fails,
                f.kind == FailureKind::TornDown,
                format!(
                    "arm B: expected FailureKind::TornDown, got {:?} ({})",
                    f.kind, f.message
                ),
            );
            check(
                &mut fails,
                dt_b >= BUDGET,
                format!(
                    "arm B: returned after {:.1} ms, before the budget",
                    ms(dt_b)
                ),
            );
            match f.quiescence {
                Quiescence::Proven => "torn_down_proven",
                Quiescence::Unknown => "torn_down_unknown_quarantined",
            }
        }
    };
    let quarantine = gpu.railgun_device_quarantine();
    arm_b_json["outcome"] = json!(outcome);
    arm_b_json["quarantine_reason"] = json!(quarantine);

    // A new submit on the torn queue.
    let torn_submit = match submit_queue_nowait(&mut low_b.queue, &nodes_b, &[]) {
        Ok(c) => {
            let r = c.wait();
            Err(format!("torn queue accepted a submit (wait: {r:?})"))
        }
        Err(f) => Ok(f),
    };
    match (outcome, &torn_submit) {
        ("torn_down_proven", Ok(f)) => {
            check(
                &mut fails,
                quarantine.is_none(),
                "arm B: retirement was proven but the device is quarantined",
            );
            check(
                &mut fails,
                f.kind == FailureKind::TornDown,
                format!(
                    "arm B: torn queue refused with {:?}, expected TornDown",
                    f.kind
                ),
            );
            println!("arm B: retirement observed after inactivate; device NOT quarantined");
        }
        ("torn_down_unknown_quarantined", Ok(f)) => {
            check(
                &mut fails,
                matches!(
                    f.kind,
                    FailureKind::TornDown | FailureKind::DeviceQuarantined
                ),
                format!(
                    "arm B: torn queue refused with {:?}, expected TornDown or DeviceQuarantined",
                    f.kind
                ),
            );
            match &quarantine {
                None => check(
                    &mut fails,
                    false,
                    "arm B: quiescence Unknown but the device is not quarantined",
                ),
                Some(reason) => {
                    println!("arm B: retirement NOT observed; device quarantined: {reason}");
                    let reg = register(&mut gpu, &built_d);
                    let reg_err = reg.err();
                    check(
                        &mut fails,
                        reg_err.as_deref().is_some_and(|e| e.contains(reason)),
                        format!("arm B: register after quarantine: {reg_err:?}, expected an Err containing the reason"),
                    );
                    let lower = gpu.lower_railgun_window(&mut ctx, &low_b.program, &low_b.pm4);
                    let lower_err = lower.err();
                    check(
                        &mut fails,
                        lower_err.as_deref().is_some_and(|e| e.contains(reason)),
                        format!("arm B: lower after quarantine: {lower_err:?}, expected an Err containing the reason"),
                    );
                    arm_b_json["refusals"] = json!({
                        "torn_queue_submit_kind": format!("{:?}", f.kind),
                        "torn_queue_submit_message": f.message,
                        "register_error": reg_err,
                        "lower_error": lower_err,
                    });
                }
            }
        }
        (_, Err(e)) => check(&mut fails, false, format!("arm B: {e}")),
        _ => {}
    }
    if let Ok(f) = &torn_submit {
        arm_b_json["torn_queue_submit"] = json!({
            "kind": format!("{:?}", f.kind),
            "quiescence": format!("{:?}", f.quiescence),
            "message": f.message,
        });
    }
    println!(
        "arm B: {} {:?} out {words_b:x?} {:.1} ms",
        outcome,
        rb.as_ref().err().map(|f| (f.kind, f.quiescence)),
        ms(dt_b)
    );

    report["arm_b"] = arm_b_json;
    report["failures"] = json!(fails);
    report["pass"] = json!(fails.is_empty());
    write_report(out_path, &report)?;
    // Arm B can leave the card busy: skip every destructor (queue/word/HIP
    // teardown can wait on a window that never retires) and leave.
    let pass = fails.is_empty();
    println!(
        "railgun_persistent_cancel_smoke: {}",
        if pass { "PASS" } else { "FAIL" }
    );
    std::process::exit(if pass { 0 } else { 1 });
}

fn main() -> std::process::ExitCode {
    let mut out = None;
    let mut arm_b = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--arm-b-unsafe-escalation" => arm_b = true,
            a if out.is_none() && !a.starts_with("--") => out = Some(arg),
            _ => {
                eprintln!(
                    "usage: railgun_persistent_cancel_smoke OUT.json [--arm-b-unsafe-escalation]"
                );
                return std::process::ExitCode::from(2);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("usage: railgun_persistent_cancel_smoke OUT.json [--arm-b-unsafe-escalation]");
        return std::process::ExitCode::from(2);
    };
    match run(&out, arm_b) {
        Ok(pass) => finish(pass),
        Err(e) => {
            eprintln!("railgun_persistent_cancel_smoke: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Host-only: every probe emits, assembles and certifies without a GPU, so a
/// kernel the fabric job could not register is caught before it is submitted
/// (`cargo test -p rdna-compute --features lab --example
/// railgun_persistent_cancel_smoke`).
#[cfg(test)]
mod tests {
    use super::*;

    fn differential_statuses(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                if let Some(d) = m.get("differential") {
                    if let Some(st) = d.get("status").and_then(Value::as_str) {
                        out.push(st.to_owned());
                    }
                }
                m.values().for_each(|x| differential_statuses(x, out));
            }
            Value::Array(a) => a.iter().for_each(|x| differential_statuses(x, out)),
            _ => {}
        }
    }

    #[test]
    fn every_probe_emits_assembles_certifies_and_authors_pm4() {
        for probe in [Probe::Poll, Probe::Ignore, Probe::Done] {
            let name = probe.variant();
            let built = build_object(probe).unwrap_or_else(|e| panic!("{e}"));
            assert!(built.instructions > 0, "{name}");
            assert_eq!(built.sha.len(), 64, "{name}");

            // G8: A1's per-arg access set is inside the metadata-conservative set.
            let receipt: Value = serde_json::from_slice(&built.receipt_json).unwrap();
            let mut statuses = Vec::new();
            differential_statuses(&receipt, &mut statuses);
            assert!(!statuses.is_empty(), "{name}: receipt has no differential");
            assert!(
                statuses.iter().all(|s| s == "pass"),
                "{name}: differential {statuses:?}\n{receipt:#}"
            );

            // The node authored from the pinned receipt lowers to PM4.
            let pinned = railgun_corpus::PinnedReceipt::admit(
                probe.module(),
                "gfx1201",
                &built.elf,
                &built.sha,
                &built.receipt_json,
            )
            .unwrap_or_else(|e| panic!("{name}: admit: {e}"));
            let facts = Arc::new(
                NodeFacts::from_pinned(&pinned, probe.symbol())
                    .unwrap_or_else(|e| panic!("{name}: facts: {e}")),
            );
            let ka = kernarg(0x7000_0000, 0x7001_0000);
            let allocations = facts
                .facts
                .pointer_offsets()
                .into_iter()
                .map(|off| {
                    let base =
                        u64::from_le_bytes(ka[off as usize..off as usize + 8].try_into().unwrap());
                    (
                        off,
                        Ok(Binding {
                            base,
                            bytes: 0x1000,
                        }),
                    )
                })
                .collect();
            let program = author(
                "gfx1201",
                &[LaunchRecord {
                    symbol: probe.symbol().to_owned(),
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    dynamic_lds: 0,
                    kernarg: ka,
                    declared: Vec::new(),
                    facts: Ok(facts),
                    allocations,
                }],
            );
            assert!(
                program.nodes[0].lowering.is_pm4(),
                "{name}: {:?}",
                program.nodes[0].lowering
            );
        }
    }
}
