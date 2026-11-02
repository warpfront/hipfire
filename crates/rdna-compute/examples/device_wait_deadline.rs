// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Single-card GPU test of the gfx1201 TP barrier wait deadline.
//!
//! Signal 2 is never published ("one card never signals"), so every
//! `tp_graph_signal_wait3_gfx1201` on it can only end through the device-clock
//! deadline (`DEVICE_WAIT_TIMEOUT_MS`). Every wait here is bounded by that
//! deadline, and a host watchdog exits the process if the whole run overruns.
//!
//! Exit codes: 0 all PASS, 1 any FAIL, 2 not a gfx1201 card (SKIP).
//!
//! Run (one gfx1201 card): `device_wait_deadline`

use hip_bridge::DeviceBuffer;
use rdna_compute::device_wait::DEVICE_WAIT_TIMEOUT_MS;
use rdna_compute::{DType, Gpu};
use std::time::{Duration, Instant};

// Covers runtime init and a cold JIT compile of the wait kernels on top of
// about 2.5 s of bounded timeouts.
const WATCHDOG: Duration = Duration::from_secs(30);
const NORMAL_MAX_MS: f64 = 50.0;
const FLOOD_LAUNCHES: u32 = 1000;

type Step = Result<String, String>;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Run one named step, print its PASS/FAIL line, and count failures.
fn report(fails: &mut u32, name: &str, step: Step) {
    match step {
        Ok(msg) => println!("PASS {name}: {msg}"),
        Err(msg) => {
            *fails += 1;
            println!("FAIL {name}: {msg}");
        }
    }
}

fn api<T, E: std::fmt::Display>(what: &str, r: Result<T, E>) -> Result<T, String> {
    r.map_err(|e| format!("{what} failed: {e}"))
}

/// The error text of a pending device-wait record, or a failure if none is
/// pending or it lacks the expected kernel/epoch identification.
fn expect_timeout(what: &str, r: Result<(), hip_bridge::HipError>) -> Result<String, String> {
    let msg = match r {
        Ok(()) => return Err(format!("{what}: expected a device wait error, got Ok")),
        Err(e) => e.to_string(),
    };
    for needle in [
        "tp_graph_signal_wait3_gfx1201",
        "epoch 1",
        "device wait timeout on GPU",
    ] {
        if !msg.contains(needle) {
            return Err(format!("{what}: error lacks {needle:?}: {msg}"));
        }
    }
    Ok(msg)
}

fn expect_clear(what: &str, r: Result<(), hip_bridge::HipError>) -> Result<(), String> {
    r.map_err(|e| format!("{what}: expected the error to be cleared, got: {e}"))
}

fn within(what: &str, elapsed: f64, lo: f64, hi: f64) -> Result<(), String> {
    if (lo..=hi).contains(&elapsed) {
        Ok(())
    } else {
        Err(format!("{what}: {elapsed:.1} ms outside [{lo}, {hi}] ms"))
    }
}

fn normal_wait2(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    let t = Instant::now();
    api(
        "wait2",
        gpu.tp_graph_signal_wait2_gfx1201([&s[0], &s[1]], 1),
    )?;
    api("sync", gpu.hip.device_synchronize())?;
    let e = ms(t);
    within("wait2", e, 0.0, NORMAL_MAX_MS)?;
    expect_clear("wait2", gpu.take_device_wait_error())?;
    Ok(format!("{e:.2} ms, no error"))
}

fn timeout_wait3(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    let t = Instant::now();
    api(
        "wait3",
        gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
    )?;
    api("sync", gpu.hip.device_synchronize())?;
    let e = ms(t);
    within("wait3 timeout", e, 400.0, 1500.0)?;
    let msg = expect_timeout("take", gpu.take_device_wait_error())?;
    expect_clear("second take", gpu.take_device_wait_error())?;
    Ok(format!("{e:.1} ms, error cleared by take: {msg}"))
}

fn download_reports_timeout(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    let tensor = api("zeros", gpu.zeros(&[4], DType::F32))?;
    let t = Instant::now();
    api(
        "wait3",
        gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
    )?;
    api("sync", gpu.hip.device_synchronize())?;
    let e = ms(t);
    within("wait3 timeout", e, 400.0, 1500.0)?;
    let first = gpu.download_f32(&tensor);
    let msg = expect_timeout("download_f32", first.map(|_| ()))?;
    api("second download_f32", gpu.download_f32(&tensor))?;
    api("free_tensor", gpu.free_tensor(tensor))?;
    Ok(format!(
        "{e:.1} ms, download failed: {msg}; next download Ok"
    ))
}

fn fail_fast_after_timeout(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    // Enqueue all three before the GPU runs the first, so the launcher's
    // pending-error check passes at each enqueue. Waits 2 and 3 see the
    // recorded code and stop within 16 polls.
    let t = Instant::now();
    for i in 0..3 {
        api(
            &format!("wait3 #{i}"),
            gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
        )?;
    }
    api("sync", gpu.hip.device_synchronize())?;
    let e = ms(t);
    within("3 chained waits", e, 0.0, 1000.0)?;
    let msg = expect_timeout("take", gpu.take_device_wait_error())?;
    expect_clear("second take", gpu.take_device_wait_error())?;
    Ok(format!("{e:.1} ms for 3 chained waits: {msg}"))
}

fn launcher_reports_pending(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    api(
        "wait3",
        gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
    )?;
    api("sync", gpu.hip.device_synchronize())?;
    // The error is now pending and has not been taken.
    let t = Instant::now();
    let launch = gpu.tp_graph_signal_wait2_gfx1201([&s[0], &s[1]], 1);
    let e = ms(t);
    let msg = match launch {
        Ok(()) => return Err("wait2 enqueue returned Ok with a pending timeout".into()),
        Err(err) => err.to_string(),
    };
    if !msg.contains("device wait timeout on GPU") {
        return Err(format!("enqueue error is not the pending timeout: {msg}"));
    }
    expect_clear("take after enqueue error", gpu.take_device_wait_error())?;
    Ok(format!(
        "enqueue refused in {e:.2} ms: {msg}; take Ok after"
    ))
}

fn normal_after_timeouts(gpu: &mut Gpu, s: &[DeviceBuffer; 3]) -> Step {
    api("store s2", gpu.tp_graph_signal_store_gfx1201(&s[2], 1))?;
    let t = Instant::now();
    api(
        "wait3",
        gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
    )?;
    api("sync", gpu.hip.device_synchronize())?;
    let e = ms(t);
    within("satisfied wait3", e, 0.0, NORMAL_MAX_MS)?;
    expect_clear("take", gpu.take_device_wait_error())?;

    let t = Instant::now();
    for i in 0..FLOOD_LAUNCHES {
        api(
            &format!("wait3 flood #{i}"),
            gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[2]], 1),
        )?;
    }
    api("flood sync", gpu.hip.device_synchronize())?;
    let flood = ms(t);
    expect_clear("take after flood", gpu.take_device_wait_error())?;
    Ok(format!(
        "{e:.2} ms single; {FLOOD_LAUNCHES} back-to-back launches {flood:.1} ms = {:.1} us/launch (info)",
        flood * 1e3 / FLOOD_LAUNCHES as f64
    ))
}

fn main() {
    // Every device wait is bounded; this only guards a hung HIP call.
    std::thread::spawn(|| {
        std::thread::sleep(WATCHDOG);
        println!("FAIL watchdog: run exceeded {} s", WATCHDOG.as_secs());
        std::process::exit(1);
    });
    let run = Instant::now();

    let mut gpu = Gpu::init_with_device(0).expect("Gpu::init_with_device(0)");
    println!(
        "arch {} | DEVICE_WAIT_TIMEOUT_MS {}",
        gpu.arch, DEVICE_WAIT_TIMEOUT_MS
    );
    if !gpu.arch_caps.is_gfx1201() {
        println!("SKIP device_wait_deadline: needs gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }

    // Signals 0 and 1 publish epoch 1; signal 2 never does.
    let mut signals = Vec::new();
    for _ in 0..3 {
        let buf = gpu.hip.malloc_signal(8).expect("malloc_signal");
        gpu.hip.memset(&buf, 0, 8).expect("memset signal");
        signals.push(buf);
    }
    let s: [DeviceBuffer; 3] = signals.try_into().ok().expect("3 signals");
    gpu.tp_graph_signal_store_gfx1201(&s[0], 1)
        .expect("store s0");
    gpu.tp_graph_signal_store_gfx1201(&s[1], 1)
        .expect("store s1");

    // Warm-up: compile both wait kernels and allocate the wait block outside
    // the timed steps. [s0, s1, s0] is fully published, so neither times out.
    gpu.tp_graph_signal_wait2_gfx1201([&s[0], &s[1]], 1)
        .expect("warm wait2");
    gpu.tp_graph_signal_wait3_gfx1201([&s[0], &s[1], &s[0]], 1)
        .expect("warm wait3");
    gpu.hip.device_synchronize().expect("warm sync");
    gpu.take_device_wait_error()
        .expect("warm-up recorded a timeout");

    let mut fails = 0;
    report(&mut fails, "normal_wait2", normal_wait2(&mut gpu, &s));
    report(&mut fails, "timeout_wait3", timeout_wait3(&mut gpu, &s));
    report(
        &mut fails,
        "download_reports_timeout",
        download_reports_timeout(&mut gpu, &s),
    );
    report(
        &mut fails,
        "fail_fast_after_timeout",
        fail_fast_after_timeout(&mut gpu, &s),
    );
    report(
        &mut fails,
        "launcher_reports_pending",
        launcher_reports_pending(&mut gpu, &s),
    );
    report(
        &mut fails,
        "normal_after_timeouts",
        normal_after_timeouts(&mut gpu, &s),
    );

    for buf in s {
        gpu.hip.free(buf).expect("free signal");
    }
    println!(
        "{} in {:.0} ms",
        if fails == 0 { "ALL PASS" } else { "FAILED" },
        ms(run)
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
