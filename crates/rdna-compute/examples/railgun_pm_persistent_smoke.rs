// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// One-card gfx1201 smoke of railgun submitting a PeaceMaker-emitted persistent
// kernel (lab only; run through the fabric):
//
// 1. Registration: PeaceMaker emits the trivial persistent dispatcher
//    (`peacemaker_isa::kernels::persistent_dispatch_gfx1201::dispatcher`, the
//    F05 MQ2 x-batched GEMV body as a region behind a claim counter and a
//    Release completion counter); `railgun-cert` certifies exactly those bytes;
//    `Gpu::railgun_register_pinned_object` admits the receipt against the
//    pinned SHA-256 and loads the object. No JIT corpus is involved.
// 2. The node is authored from the receipt facts (`railgun::kernel::author`),
//    planned (`plan`, strict `UnreceiptedSystem` rows), lowered through
//    railgun's own gfx12 transport and submitted with `submit_queue_nowait`;
//    the returned completion handle proves quiescence.
// 3. Byte identity: the same object launched directly through HIP on identical
//    inputs. Every `y` byte, the claim counter and the completion counter must
//    match, for several shapes and two submits of one lowered window.
// 4. Submit without completion: a second submit while the first is in flight
//    is refused before anything is written; `wait` then completes the first.
//
// Run (gfx1201): cargo run --release --features lab -p rdna-compute \
//   --example railgun_pm_persistent_smoke -- OUT.json
// Exit 0 only when every case is byte-identical and the refusal held.

use std::sync::Arc;
use std::time::Instant;

use peacemaker_isa::kernels::persistent_dispatch_gfx1201 as pd;
use peacemaker_isa::{native, Arch};
use railgun::kernel::{author, LaunchRecord};
use railgun::plan::{plan, CacheTable, RowPolicy};
use railgun::window::{submit_queue_nowait, NodeImage, WindowQueue};
use serde_json::{json, Value};

const MODULE: &str = "pm_persistent_mq2_xbatch_dispatch_gfx1201";
const POISON: u8 = 0xA5;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// MQ2 weights (72-byte groups of 256) and f32 activations, as the PeaceMaker
/// smoke generates them.
fn mq2_inputs(m: usize, k: usize, b: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Lcg(seed);
    let groups = k / 256;
    let mut a = vec![0u8; m * groups * 72 + 64];
    for g in 0..m * groups {
        let at = g * 72;
        for h in 0..4 {
            let v = ((rng.next() & 0x8000)
                | (((10 + rng.next() % 5) << 10) & 0x7c00)
                | (rng.next() & 0x3ff)) as u16;
            a[at + 2 * h..at + 2 * h + 2].copy_from_slice(&v.to_le_bytes());
        }
        for byte in &mut a[at + 8..at + 72] {
            *byte = rng.next() as u8;
        }
    }
    let x = (0..b * k)
        .flat_map(|_| ((rng.next() % 2001) as f32 / 1000.0 - 1.0).to_le_bytes())
        .collect();
    (a, x)
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The dispatcher's 56-byte kernarg (module docs of `persistent_dispatch_gfx1201`).
fn kernarg(a: u64, x: u64, y: u64, m: u32, k: u32, b: u32, queue: u64, tasks: u32) -> Vec<u8> {
    let mut k8 = Vec::with_capacity(pd::DISPATCH_KERNARG_BYTES as usize);
    for p in [a, x, y] {
        k8.extend(p.to_le_bytes());
    }
    for v in [m, k, b] {
        k8.extend(v.to_le_bytes());
    }
    k8.resize(40, 0);
    k8.extend(queue.to_le_bytes());
    k8.extend(tasks.to_le_bytes());
    k8.resize(pd::DISPATCH_KERNARG_BYTES as usize, 0);
    k8
}

fn run(out: &str) -> Result<bool, String> {
    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
    if gpu.arch != "gfx1201" {
        return Err(format!(
            "device is {}; this smoke runs only on gfx1201",
            gpu.arch
        ));
    }

    // 1. Emit, certify, register.
    let emitted = pd::dispatcher()?;
    let elf = native::assemble(&emitted.s_text, Arch::Gfx1201)?;
    let sha = railgun_corpus::sha256_hex(&elf);
    let receipt = railgun_cert::certify(&elf, None).map_err(|e| format!("certify: {e}"))?;
    let receipt_json = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
    let image: Arc<[u8]> = Arc::from(elf.as_slice());
    let facts = gpu
        .railgun_register_pinned_object(MODULE, image, &sha, &receipt_json, &[pd::DISPATCH_SYMBOL])?
        .remove(0);
    println!(
        "registered {MODULE} object {sha} receipt {} tier {}",
        facts.kref.receipt_sha256, facts.kref.tier
    );
    // A wrong pin is refused before anything is loaded.
    let mut wrong = sha.clone().into_bytes();
    wrong[0] = if wrong[0] == b'0' { b'1' } else { b'0' };
    let wrong_pin_refused = gpu
        .railgun_register_pinned_object(
            MODULE,
            Arc::from(elf.as_slice()),
            std::str::from_utf8(&wrong).unwrap(),
            &receipt_json,
            &[pd::DISPATCH_SYMBOL],
        )
        .is_err();

    let table = CacheTable::for_arch(&gpu.arch).ok_or("no gfx1201 cache table")?;
    let mut ctx = gpu.railgun_lowering_context()?;
    let mut cases = Vec::new();
    let mut all_ok = wrong_pin_refused;
    let mut refusal_checked = false;
    for (m, k, b, grid) in [
        (13usize, 512usize, 1usize, 3u32),
        (1000, 2560, 4, 16),
        (100_276, 2560, 4, 256),
    ] {
        let (a, x) = mq2_inputs(m, k, b, 0x5eed ^ m as u64 ^ (b as u64) << 40);
        let tasks = m.div_ceil(4) as u32;
        let y_bytes = b * m * 4;
        let hip = &gpu.hip;
        let up = |bytes: &[u8]| -> Result<hip_bridge::DeviceBuffer, String> {
            let d = hip.malloc(bytes.len().max(4)).map_err(|e| e.to_string())?;
            hip.memcpy_htod(&d, bytes).map_err(|e| e.to_string())?;
            Ok(d)
        };
        let read = |d: &hip_bridge::DeviceBuffer, n: usize| -> Result<Vec<u8>, String> {
            let mut v = vec![0u8; n];
            hip.memcpy_dtoh(&mut v, d).map_err(|e| e.to_string())?;
            Ok(v)
        };
        let (da, dx) = (up(&a)?, up(&x)?);
        let (y_direct, q_direct) = (up(&vec![POISON; y_bytes])?, up(&[0u8; 8])?);
        let (y_rg, q_rg) = (up(&vec![POISON; y_bytes])?, up(&[0u8; 8])?);
        let ka = |y: &hip_bridge::DeviceBuffer, q: &hip_bridge::DeviceBuffer| {
            kernarg(
                da.as_ptr() as u64,
                dx.as_ptr() as u64,
                y.as_ptr() as u64,
                m as u32,
                k as u32,
                b as u32,
                q.as_ptr() as u64,
                tasks,
            )
        };

        // Direct HIP launch of the very same registered object.
        let mut direct_ka = ka(&y_direct, &q_direct);
        gpu.launch_kernel_blob(
            pd::DISPATCH_SYMBOL,
            [grid, 1, 1],
            [32, 1, 1],
            0,
            &mut direct_ka,
        )
        .map_err(|e| format!("direct launch: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        let want_y = read(&y_direct, y_bytes)?;
        let want_q = u32s(&read(&q_direct, 8)?);

        // railgun: author from the pinned receipt facts, plan, lower, submit.
        let rg_ka = ka(&y_rg, &q_rg);
        let mut allocations = Vec::new();
        for off in facts.facts.pointer_offsets() {
            let o = off as usize;
            let addr = u64::from_le_bytes(rg_ka[o..o + 8].try_into().unwrap());
            allocations.push((off, gpu.railgun_probe_allocation(addr)));
        }
        let program = author(
            &gpu.arch,
            &[LaunchRecord {
                symbol: pd::DISPATCH_SYMBOL.to_owned(),
                grid: [grid, 1, 1],
                block: [32, 1, 1],
                dynamic_lds: 0,
                kernarg: rg_ka.clone(),
                declared: Vec::new(),
                facts: Ok(Arc::clone(&facts)),
                allocations,
            }],
        );
        let node = &program.nodes[0];
        if !node.lowering.is_pm4() {
            return Err(format!("dispatcher node is not PM4: {:?}", node.lowering));
        }
        let words = program.authored_words();
        let segment = node.segment_bytes.ok_or("no certified kernarg segment")? as usize;
        let image = program.kernarg_image(0, segment, &words)?;
        let pm4 = plan(&program, &table, RowPolicy::UnreceiptedSystem);
        if !pm4.is_single_segment(program.nodes.len()) {
            return Err(format!("{} PM4 segments", pm4.segments.len()));
        }
        let mut queue = gpu.lower_railgun_window(&mut ctx, &program, &pm4)?;
        let nodes = [NodeImage {
            kernarg: &image,
            grid: node.grid,
        }];
        let mut submits = Vec::new();
        let mut case_ok = true;
        for rep in 0..2 {
            gpu.hip
                .memcpy_htod(&y_rg, &vec![POISON; y_bytes])
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memcpy_htod(&q_rg, &[0u8; 8])
                .map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            let completion = submit_queue_nowait(&mut queue, &nodes, &[])
                .map_err(|f| format!("submit: {} ({:?})", f.message, f.quiescence))?;
            let submit_us = t0.elapsed().as_secs_f64() * 1e6;
            completion
                .wait()
                .map_err(|f| format!("completion: {} ({:?})", f.message, f.quiescence))?;
            let complete_us = t0.elapsed().as_secs_f64() * 1e6;
            let got_y = read(&y_rg, y_bytes)?;
            let got_q = u32s(&read(&q_rg, 8)?);
            let mismatched = want_y
                .chunks(4)
                .zip(got_y.chunks(4))
                .filter(|(w, g)| w != g)
                .count();
            let unwritten = got_y.chunks(4).filter(|w| *w == [POISON; 4]).count();
            let ok = mismatched == 0
                && unwritten == 0
                && got_q == want_q
                && want_q[1] == tasks
                && want_q[0] == tasks + grid;
            case_ok &= ok;
            println!(
                "M={m} K={k} B={b} grid={grid} rep {rep}: mismatched {mismatched}/{} unwritten {unwritten} queue direct {want_q:?} railgun {got_q:?} submit {submit_us:.0} us complete {complete_us:.0} us -> {}",
                b * m,
                if ok { "OK" } else { "FAIL" }
            );
            submits.push(
                json!({"rep": rep, "mismatched_words": mismatched, "unwritten_words": unwritten,
                "queue_direct": want_q, "queue_railgun": got_q,
                "host_submit_return_us": submit_us, "host_complete_us": complete_us, "pass": ok}),
            );
        }

        // Submit without completion: refused before any write, then wait completes the first.
        let mut refusal = Value::Null;
        if !refusal_checked {
            refusal_checked = true;
            gpu.hip
                .memcpy_htod(&q_rg, &[0u8; 8])
                .map_err(|e| e.to_string())?;
            queue
                .submit_nowait(&nodes, &[])
                .map_err(|f| format!("submit: {}", f.message))?;
            let second = queue.submit_nowait(&nodes, &[]);
            let waited = queue.wait();
            let got_q = u32s(&read(&q_rg, 8)?);
            let held = second.is_err() && waited.is_ok() && got_q == want_q;
            println!(
                "submit without completion: second submit {:?}; wait {:?}; queue {got_q:?} -> {}",
                second.as_ref().err().map(|f| &f.message),
                waited.as_ref().err().map(|f| &f.message),
                if held { "OK" } else { "FAIL" }
            );
            case_ok &= held;
            refusal = json!({"second_submit_refused": second.is_err(),
                "refusal": second.err().map(|f| f.message),
                "wait_ok": waited.is_ok(), "queue_after_wait": got_q, "pass": held});
        }
        drop(queue);
        for d in [da, dx, y_direct, q_direct, y_rg, q_rg] {
            gpu.hip.free(d).map_err(|e| e.to_string())?;
        }
        all_ok &= case_ok;
        cases.push(json!({"m": m, "k": k, "b": b, "grid": grid, "tasks": tasks,
            "submits": submits, "submit_without_completion": refusal, "pass": case_ok}));
    }
    let report = json!({
        "device_arch": gpu.arch,
        "module": MODULE,
        "symbol": pd::DISPATCH_SYMBOL,
        "object_sha256": sha,
        "receipt_sha256": facts.kref.receipt_sha256,
        "receipt_tier": facts.kref.tier,
        "obligations": facts.kref.obligations,
        "wrong_pin_refused": wrong_pin_refused,
        "cases": cases,
        "pass": all_ok,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    Ok(all_ok)
}

fn main() -> std::process::ExitCode {
    let Some(out) = std::env::args().nth(1) else {
        eprintln!("usage: railgun_pm_persistent_smoke OUT.json");
        return std::process::ExitCode::from(2);
    };
    match run(&out) {
        Ok(true) => {
            println!("railgun_pm_persistent_smoke: PASS");
            std::process::ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("railgun_pm_persistent_smoke: FAIL");
            std::process::ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("railgun_pm_persistent_smoke: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
