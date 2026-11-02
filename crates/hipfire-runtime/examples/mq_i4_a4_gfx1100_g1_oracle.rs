// SPDX-License-Identifier: Apache-2.0
//! Raw PM K3 gfx1100 (R128 / K640 A4 producer) G1 GPU runner; never participates in runtime dispatch.
//! Build: cargo build -p hipfire-runtime --release --features lab --example mq_i4_a4_gfx1100_g1_oracle
//! (auto-discovered example; no Cargo.toml entry, no registry or dispatch wiring: standalone qualification only).
//! Driven by `crates/peacemaker-isa/tools/mq_i4_a4_gfx1100_oracle.py` (see docs/MQ_I4_A4_GFX1100_G1_ORACLE.md):
//!   mq_i4_a4_gfx1100_g1_oracle --codeobject /abs/mq_i4_a4_pm_gfx1100.hxaco --cases DIR \
//!     [--expect-uuid GPU-..] [--expect-bdf 0000:..] [--iterations N [--warmup M]]
//! `DIR/manifest.json` is `{"cases":[{"name":N,"p":P,"l":L}, ..]}` (extra keys ignored) and per case `N.h.bin`
//! (P*640 u16 LE bf16 bits), `N.sorted.bin` (P i32 LE, -1 = pad row), `N.signs1.bin` / `N.signs2.bin` (128 f32 LE). The
//! runner pre-fills the 5*L*72-byte down sidecar with 0xA5, launches `mq_i4_a4_r128_k640_gfx1100`
//! (grid [5P,1,1], block [32,1,1]) once, reads it back to `N.xq.out.bin`, and only then (optionally) times
//! `--warmup` + `--iterations` further launches. The byte comparison against the CPU oracle is the driver's.
//! Timing: per iteration the host wall clock around `launch_kernel_blob` + `device_synchronize` (launch + sync, no
//! event timing, no memset or copy inside the window). The expected card sits on a PCIe 4.0 x4 root-port link; the
//! figures are a short launch+sync wall clock of this kernel only and are NOT an end-to-end claim.
//! It prints one JSON receipt line with the device identity (uuid, bdf, arch), the launch list and the timings.
use hip_bridge::HipRuntime;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
const SCHEMA: &str = "mq-i4-a4-gfx1100-g1-runner-v1";
const ARCH: &str = "gfx1100";
const SYMBOL: &str = "mq_i4_a4_r128_k640_gfx1100";
const K: usize = 640;
const GROUPS: usize = K / 128;
const RECORD: usize = 72;
const KERNARG: usize = 48;
const MAX_ITERATIONS: usize = 1_000_000;
const LINK_CAVEAT: &str = "PCIe 4.0 x4: the expected card's upstream root port is a PCIe 4.0 x4 link; timings are launch+sync host wall clock of this kernel alone, not an end-to-end claim.";
const USAGE: &str = "usage: mq_i4_a4_gfx1100_g1_oracle --codeobject PATH --cases DIR [--expect-uuid UUID] [--expect-bdf BDF] [--iterations N [--warmup M]]";

fn read(dir: &Path, name: &str, suffix: &str) -> Result<Vec<u8>> {
    let path = dir.join(format!("{name}.{suffix}"));
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn count(flag: &str, value: &str) -> Result<usize> {
    let n: usize = value
        .parse()
        .map_err(|_| format!("{flag}: not a non-negative integer: {value}"))?;
    if n > MAX_ITERATIONS {
        return Err(format!("{flag}: at most {MAX_ITERATIONS}"));
    }
    Ok(n)
}

fn timing(us: &mut [f64], warmup: usize) -> Value {
    us.sort_by(|a, b| a.total_cmp(b));
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    json!({"iterations": us.len(), "warmup": warmup, "min_us": us[0], "median_us": us[us.len() / 2], "mean_us": mean, "max_us": us[us.len() - 1]})
}

fn run() -> Result<()> {
    let (mut codeobject, mut cases, mut expect_uuid, mut expect_bdf) = (None, None, None, None);
    let (mut iterations, mut warmup) = (0usize, None);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--codeobject" => codeobject = Some(value),
            "--cases" => cases = Some(PathBuf::from(value)),
            "--expect-uuid" => expect_uuid = Some(value),
            "--expect-bdf" => expect_bdf = Some(value.to_ascii_lowercase()),
            "--iterations" => iterations = count(&flag, &value)?,
            "--warmup" => warmup = Some(count(&flag, &value)?),
            _ => return Err(USAGE.into()),
        }
    }
    if iterations == 0 && warmup.is_some() {
        return Err("--warmup requires --iterations N with N > 0".into());
    }
    let warmup = warmup.unwrap_or(0);
    let (codeobject, dir) = (codeobject.ok_or(USAGE)?, cases.ok_or(USAGE)?);
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match = arch == ARCH
        && expect_uuid.as_deref().is_none_or(|u| u == uuid)
        && expect_bdf.as_deref().is_none_or(|b| b == bdf);
    let device = json!({"uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": ARCH, "expected_uuid": expect_uuid, "expected_bdf": expect_bdf, "identity_match": identity_match});
    if !identity_match {
        println!(
            "{}",
            json!({"schema": SCHEMA, "device": device, "error": "MQ-I4 A4 gfx1100 device identity mismatch"})
        );
        return Err("MQ-I4 A4 gfx1100 device identity mismatch".into());
    }
    let image = std::fs::read(&codeobject).map_err(err)?;
    let module = hip.module_load_data(&image).map_err(err)?;
    let f = hip.module_get_function(&module, SYMBOL).map_err(err)?;
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).map_err(err)?)
            .map_err(err)?;
    let mut launches = Vec::new();
    for case in manifest["cases"].as_array().ok_or("manifest: cases")? {
        let name = case["name"].as_str().ok_or("case name")?;
        if name.is_empty() || name.contains(['/', '\\']) {
            return Err(format!("{name:?}: invalid case name"));
        }
        let (p, l) = (
            case["p"].as_u64().ok_or("case p")? as usize,
            case["l"].as_u64().ok_or("case l")? as usize,
        );
        if p == 0 || l == 0 || p.checked_mul(GROUPS).is_none_or(|g| g > u32::MAX as usize) {
            return Err(format!("{name}: P, L must be nonzero and 5P must fit u32"));
        }
        let (h, sorted) = (read(&dir, name, "h.bin")?, read(&dir, name, "sorted.bin")?);
        let (s1, s2) = (
            read(&dir, name, "signs1.bin")?,
            read(&dir, name, "signs2.bin")?,
        );
        if h.len() != p * K * 2 || sorted.len() != p * 4 || s1.len() != 512 || s2.len() != 512 {
            return Err(format!("{name}: input sizes disagree with P={p}"));
        }
        let xq_bytes = GROUPS * l * RECORD;
        let (dh, dsorted, dxq, d1, d2) = (
            hip.malloc(h.len()).map_err(err)?,
            hip.malloc(sorted.len()).map_err(err)?,
            hip.malloc(xq_bytes).map_err(err)?,
            hip.malloc(512).map_err(err)?,
            hip.malloc(512).map_err(err)?,
        );
        hip.memcpy_htod(&dh, &h).map_err(err)?;
        hip.memcpy_htod(&dsorted, &sorted).map_err(err)?;
        hip.memcpy_htod(&d1, &s1).map_err(err)?;
        hip.memcpy_htod(&d2, &s2).map_err(err)?;
        hip.memset(&dxq, 0xa5, xq_bytes).map_err(err)?;
        let mut kernarg = [0u8; KERNARG];
        for (off, ptr) in [
            (0, dh.as_ptr()),
            (8, dsorted.as_ptr()),
            (16, dxq.as_ptr()),
            (24, d1.as_ptr()),
            (32, d2.as_ptr()),
        ] {
            kernarg[off..off + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
        }
        kernarg[40..44].copy_from_slice(&(p as u32).to_le_bytes());
        kernarg[44..48].copy_from_slice(&(l as u32).to_le_bytes());
        let grid = [(p * GROUPS) as u32, 1, 1];
        // SAFETY: exact 48-byte ABI; every allocation stays live through every synchronize below.
        unsafe { hip.launch_kernel_blob(&f, grid, [32, 1, 1], 0, None, &mut kernarg) }
            .map_err(err)?;
        hip.device_synchronize().map_err(err)?;
        let mut out = vec![0u8; xq_bytes];
        hip.memcpy_dtoh(&mut out, &dxq).map_err(err)?;
        std::fs::write(dir.join(format!("{name}.xq.out.bin")), &out).map_err(err)?;
        let mut launch =
            json!({"case": name, "p": p, "l": l, "grid_x": p * GROUPS, "xq_bytes": xq_bytes});
        if iterations > 0 {
            // Timed launches follow the output read-back; they rewrite identical records and never feed the comparison.
            let mut us = Vec::with_capacity(iterations);
            for i in 0..warmup + iterations {
                let t0 = Instant::now();
                // SAFETY: same ABI and live allocations as the launch above.
                unsafe { hip.launch_kernel_blob(&f, grid, [32, 1, 1], 0, None, &mut kernarg) }
                    .map_err(err)?;
                hip.device_synchronize().map_err(err)?;
                let dt = t0.elapsed().as_secs_f64() * 1e6;
                if i >= warmup {
                    us.push(dt)
                }
            }
            launch["timing"] = timing(&mut us, warmup);
        }
        launches.push(launch);
    }
    let timing_note = (iterations > 0).then_some("per iteration host wall clock around launch_kernel_blob + device_synchronize; kernel only; not an end-to-end claim");
    println!(
        "{}",
        json!({"schema": SCHEMA, "symbol": SYMBOL, "codeobject": codeobject, "device": device, "kernarg_bytes": KERNARG,
        "timing_requested": {"iterations": iterations, "warmup": warmup}, "timing_scope": timing_note, "link_caveat": LINK_CAVEAT, "launches": launches})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
