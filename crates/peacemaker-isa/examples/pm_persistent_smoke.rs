// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! One-card gfx1201 smoke of the PeaceMaker persistent-megakernel prerequisites
//! (`kernels::persistent_dispatch_gfx1201`). Lab only, run through the fabric:
//!
//! 1. **Region identity.** The shipped MQ2 x-batched GEMV module
//!    (`gemv_mq2_xbatch_pm_gfx1201`, grid `ceil(M/4)`) and the same body as a
//!    region inside the persistent dispatcher (a few workgroups claiming tasks)
//!    on identical inputs: every `y` byte must match, the completion counter must
//!    equal the task count and the claim counter `tasks + grid`.
//! 2. **Release/acquire handoff.** Workgroup 0 publishes `WORDS*32` dwords with a
//!    `Release` signal after a delay; the other workgroups primed their caches
//!    with the old values, `event_wait(Acquire)` and re-read. Counted over many
//!    launches: stale words, watchdog timeouts. Negative controls run the same
//!    instruction stream with the acquire's `global_inv` removed, and with the
//!    release's `s_wait_loadcnt/s_wait_storecnt` removed (both rejected by
//!    `sync::verify`; here only to see whether the hardware shows the race).
//!
//! `pm_persistent_smoke OUT.json [--trials N]`; exit 0 when (1) is byte-identical
//! and the fenced handoff has no stale word or timeout.

use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use peacemaker_isa::kernels::gemv_mq2_xbatch_pm_gfx1201 as mq2;
use peacemaker_isa::kernels::persistent_dispatch_gfx1201 as pd;
use peacemaker_isa::{native, sync, Arch};
use serde_json::{json, Value};
use std::process::ExitCode;
use std::time::Instant;

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

struct Card {
    rt: HipRuntime,
}
impl Card {
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, String> {
        let d = self
            .rt
            .malloc(bytes.len().max(4))
            .map_err(|e| format!("hipMalloc: {e}"))?;
        self.rt
            .memcpy_htod(&d, bytes)
            .map_err(|e| format!("upload: {e}"))?;
        Ok(d)
    }
    fn read(&self, d: &DeviceBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.rt
            .memcpy_dtoh(&mut v, d)
            .map_err(|e| format!("readback: {e}"))?;
        Ok(v)
    }
    fn launch(&self, f: &Function, grid: u32, blob: &mut [u8]) -> Result<f64, String> {
        let t = Instant::now();
        // SAFETY: `blob` is the kernel's full kernarg image; every pointer in it is a live
        // allocation of this harness, sized for the launch.
        unsafe {
            self.rt
                .launch_kernel_blob(f, [grid, 1, 1], [32, 1, 1], 0, None, blob)
        }
        .map_err(|e| format!("launch: {e}"))?;
        self.rt
            .device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))?;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    fn function(&self, elf: &[u8], symbol: &str) -> Result<Function, String> {
        let m = self
            .rt
            .module_load_data(elf)
            .map_err(|e| format!("hipModuleLoadData({symbol}): {e}"))?;
        let f = self
            .rt
            .module_get_function(&m, symbol)
            .map_err(|e| format!("hipModuleGetFunction({symbol}): {e}"))?;
        std::mem::forget(m); // modules live for the process
        Ok(f)
    }
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn region_identity(card: &Card, solo: &Function, disp: &Function) -> Result<(bool, Value), String> {
    let mut cases = Vec::new();
    let mut ok = true;
    for (m, k, b, grid) in [
        (13usize, 512usize, 1usize, 3u32),
        (1000, 2560, 4, 16),
        (100_276, 2560, 1, 64),
        (100_276, 2560, 4, 256),
        (100_276, 2560, 3, 1024),
    ] {
        let (a, x) = mq2_inputs(m, k, b, 0x5eed ^ m as u64 ^ (b as u64) << 40);
        let (da, dx) = (card.upload(&a)?, card.upload(&x)?);
        let y_bytes = b * m * 4;
        let poison = vec![0xa5u8; y_bytes];
        let tasks = m.div_ceil(4) as u32;
        let kernarg = |y: &DeviceBuffer, q: Option<&DeviceBuffer>| {
            let mut k8 = Vec::new();
            for p in [&da, &dx, y] {
                k8.extend((p.as_ptr() as u64).to_le_bytes());
            }
            for v in [m, k, b] {
                k8.extend((v as u32).to_le_bytes());
            }
            if let Some(q) = q {
                k8.resize(40, 0);
                k8.extend((q.as_ptr() as u64).to_le_bytes());
                k8.extend(tasks.to_le_bytes());
                k8.resize(pd::DISPATCH_KERNARG_BYTES as usize, 0);
            }
            k8
        };
        let y_solo = card.upload(&poison)?;
        let t_solo = card.launch(solo, tasks, &mut kernarg(&y_solo, None))?;
        let want = card.read(&y_solo, y_bytes)?;
        let y_disp = card.upload(&poison)?;
        let queue = card.upload(&[0u8; 8])?;
        let t_disp = card.launch(disp, grid, &mut kernarg(&y_disp, Some(&queue)))?;
        let got = card.read(&y_disp, y_bytes)?;
        let q = u32s(&card.read(&queue, 8)?);
        let mismatched = want
            .chunks(4)
            .zip(got.chunks(4))
            .filter(|(w, g)| w != g)
            .count();
        let unwritten = want.chunks(4).filter(|w| *w == [0xa5; 4]).count();
        let case_ok = mismatched == 0 && unwritten == 0 && q[1] == tasks && q[0] == tasks + grid;
        ok &= case_ok;
        println!(
            "region M={m} K={k} B={b} grid={grid}: mismatched {mismatched}/{} unwritten {unwritten} claims {} done {} (tasks {tasks}) solo {t_solo:.0} us, persistent {t_disp:.0} us -> {}",
            b * m,
            q[0],
            q[1],
            if case_ok { "OK" } else { "FAIL" }
        );
        cases.push(json!({"m": m, "k": k, "b": b, "grid": grid, "tasks": tasks,
            "mismatched_words": mismatched, "unwritten_words": unwritten,
            "claim_counter": q[0], "completion_counter": q[1],
            "first_launch_wall_us": {"standalone": t_solo, "persistent": t_disp}, "pass": case_ok}));
        for d in [da, dx, y_solo, y_disp, queue] {
            card.rt.free(d).map_err(|e| format!("hipFree: {e}"))?;
        }
    }
    Ok((ok, Value::Array(cases)))
}

/// Remove the release waits before the event signal (the two lines before the producer's
/// `global_atomic_add_u32`) or the acquire's `global_inv`.
fn mutate(text: &str, drop_release: bool, drop_acquire: bool) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let signal = lines
        .iter()
        .position(|l| l.trim().starts_with("global_atomic_add_u32"))
        .expect("producer signal");
    lines
        .iter()
        .enumerate()
        .filter(|(i, l)| {
            let t = l.trim();
            !(drop_release && (*i + 2 == signal || *i + 1 == signal) && t.starts_with("s_wait_"))
                && !(drop_acquire && t.starts_with("global_inv"))
        })
        .map(|(_, l)| format!("{l}\n"))
        .collect()
}

fn handoff(
    card: &Card,
    f: &Function,
    grid: u32,
    trials: usize,
    delay: u32,
) -> Result<Value, String> {
    let words = (pd::WORDS * 32) as usize;
    let rec = pd::RECORD_DWORDS as usize;
    let (mut stale, mut stale_launches, mut timeouts, mut pre_fresh) =
        (0usize, 0usize, 0usize, 0usize);
    let mut latency_ticks = Vec::new();
    let data = card.upload(&vec![0u8; words * 4])?;
    let event = card.upload(&[0u8; 4])?;
    let out = card.upload(&vec![0u8; rec * 4 * grid as usize])?;
    for _ in 0..trials {
        card.rt
            .memset(&data, 0, words * 4)
            .map_err(|e| format!("memset: {e}"))?;
        card.rt
            .memset(&event, 0, 4)
            .map_err(|e| format!("memset: {e}"))?;
        card.rt
            .memset(&out, 0, rec * 4 * grid as usize)
            .map_err(|e| format!("memset: {e}"))?;
        let mut k = Vec::new();
        for p in [&data, &event, &out] {
            k.extend((p.as_ptr() as u64).to_le_bytes());
        }
        k.extend(delay.to_le_bytes());
        k.extend(4_000_000u32.to_le_bytes()); // watchdog polls (~4M x s_sleep 1)
        k.extend(1u32.to_le_bytes()); // expected event value
        k.resize(pd::HANDOFF_KERNARG_BYTES as usize, 0);
        card.launch(f, grid, &mut k)?;
        let o = u32s(&card.read(&out, rec * 4 * grid as usize)?);
        let t_pub = u64::from(o[2 * words + 2]) | u64::from(o[2 * words + 3]) << 32;
        let mut launch_stale = 0;
        for c in 1..grid as usize {
            let r = &o[c * rec..(c + 1) * rec];
            if r[2 * words] != 1 {
                timeouts += 1;
                continue;
            }
            for j in 0..pd::WORDS as usize {
                for lane in 0..32 {
                    let idx = j * 32 + lane;
                    let want = (idx + 1) as u32;
                    if r[idx] != want {
                        launch_stale += 1;
                    }
                    if r[words + idx] == want {
                        pre_fresh += 1;
                    }
                }
            }
            let t_obs = u64::from(r[2 * words + 2]) | u64::from(r[2 * words + 3]) << 32;
            latency_ticks.push(t_obs.wrapping_sub(t_pub) as i64);
        }
        stale += launch_stale;
        stale_launches += usize::from(launch_stale > 0);
    }
    for d in [data, event, out] {
        card.rt.free(d).map_err(|e| format!("hipFree: {e}"))?;
    }
    latency_ticks.sort_unstable();
    let p = |q: f64| {
        latency_ticks
            .get(((latency_ticks.len() as f64 - 1.0) * q) as usize)
            .copied()
    };
    Ok(
        json!({"grid": grid, "trials": trials, "delay_sleeps": delay,
        "stale_words": stale, "launches_with_stale_words": stale_launches,
        "consumer_timeouts": timeouts, "primed_words_already_fresh": pre_fresh,
        "publish_to_observe_realtime_ticks_100MHz": {"p50": p(0.5), "p95": p(0.95), "max": latency_ticks.last()}}),
    )
}

fn run(out: &str, trials: usize) -> Result<bool, String> {
    let rt = HipRuntime::load().map_err(|e| format!("HipRuntime::load: {e}"))?;
    rt.set_device(0).map_err(|e| format!("hipSetDevice: {e}"))?;
    let arch = rt.get_arch(0).map_err(|e| format!("get_arch: {e}"))?;
    if !arch.starts_with("gfx1201") {
        return Err(format!(
            "device 0 is {arch}; this smoke runs only on gfx1201"
        ));
    }
    let card = Card { rt };
    let solo = card.function(&mq2::emit_elf(&mq2::SHIPPED)?, mq2::SYMBOL)?;
    let dispatcher = pd::dispatcher()?;
    let disp = card.function(
        &native::assemble(&dispatcher.s_text, Arch::Gfx1201)?,
        pd::DISPATCH_SYMBOL,
    )?;
    let (identity_ok, identity) = region_identity(&card, &solo, &disp)?;

    let h = pd::handoff()?;
    sync::verify(&h.program, &h.proof.syncs)?;
    let mut arms = serde_json::Map::new();
    let mut fenced_ok = true;
    for (name, drop_release, drop_acquire) in [
        ("release_acquire", false, false),
        ("no_acquire_inv", false, true),
        ("no_release_waits", true, false),
    ] {
        let text = if drop_release || drop_acquire {
            mutate(&h.s_text, drop_release, drop_acquire)
        } else {
            h.s_text.clone()
        };
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let at = lines
            .iter()
            .position(|l| l.starts_with("global_atomic_add_u32"))
            .unwrap();
        let inv = lines.iter().filter(|l| l.starts_with("global_inv")).count();
        println!(
            "{name}: before the signal {:?}; global_inv lines {inv}",
            &lines[at.saturating_sub(3)..=at]
        );
        let f = card.function(&native::assemble(&text, Arch::Gfx1201)?, pd::HANDOFF_SYMBOL)?;
        let mut runs = Vec::new();
        for grid in [2u32, 32] {
            let r = handoff(&card, &f, grid, trials, 300)?;
            println!("handoff {name} grid {grid}: {r}");
            if name == "release_acquire" {
                fenced_ok &= r["stale_words"] == 0 && r["consumer_timeouts"] == 0;
            }
            runs.push(r);
        }
        arms.insert(name.into(), Value::Array(runs));
    }
    let receipt = json!({
        "device_arch": arch,
        "dispatcher_symbol": pd::DISPATCH_SYMBOL,
        "region_proof": dispatcher.proof.regions,
        "region_identity": identity,
        "handoff": arms,
        "pass": identity_ok && fenced_ok,
    });
    std::fs::write(out, serde_json::to_string_pretty(&receipt).unwrap())
        .map_err(|e| format!("{out}: {e}"))?;
    Ok(identity_ok && fenced_ok)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(out) = args.get(1) else {
        eprintln!("usage: pm_persistent_smoke OUT.json [--trials N]");
        return ExitCode::from(2);
    };
    let trials = args
        .iter()
        .position(|a| a == "--trials")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    match run(out, trials) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            println!("FAIL");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
