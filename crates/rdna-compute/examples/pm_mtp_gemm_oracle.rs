// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Synthetic byte oracle for the exact gfx1201 MTP fill GEMM. Usage: --out FILE.
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sha(v: &[u8]) -> String {
    Sha256::digest(v)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn random(s: &mut u64) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s as u32
}
fn download(g: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    g.hip.device_synchronize().map_err(err)?;
    let mut v = vec![0; t.buf.size()];
    g.hip.memcpy_dtoh(&mut v, &t.buf).map_err(err)?;
    Ok(v)
}
fn timing(g: &mut Gpu, mut f: impl FnMut(&mut Gpu) -> hip_bridge::HipResult<()>) -> Result<Value> {
    let (a, b) = (
        g.hip.event_create().map_err(err)?,
        g.hip.event_create().map_err(err)?,
    );
    for _ in 0..5 {
        f(g).map_err(err)?;
    }
    g.hip.device_synchronize().map_err(err)?;
    let mut v = Vec::with_capacity(20);
    for _ in 0..20 {
        g.hip.event_record(&a, None).map_err(err)?;
        f(g).map_err(err)?;
        g.hip.event_record(&b, None).map_err(err)?;
        g.hip.event_synchronize(&b).map_err(err)?;
        v.push(g.hip.event_elapsed_ms(&a, &b).map_err(err)?);
    }
    g.hip.event_destroy(a).map_err(err)?;
    g.hip.event_destroy(b).map_err(err)?;
    v.sort_by(f32::total_cmp);
    Ok(json!({"median_ms":v[10],"samples_ms":v}))
}
fn run() -> Result<Value> {
    let mut g = Gpu::init().map_err(err)?;
    if g.arch != "gfx1201" {
        return Err(format!("wrong arch {}", g.arch));
    }
    use hipfire_config::devices::{self, DeviceRoot, IdentitySource, ObservedDevice};
    if g.hip.device_count().map_err(err)? != 1 {
        return Err("exactly one visible device required".into());
    }
    let root = DeviceRoot::system();
    let seen = ObservedDevice {
        logical: 0,
        arch: g.hip.get_arch(0).map_err(err)?,
        pci_bus_id: g.hip.device_pci_bus_id(0).map_err(err)?,
        uuid: if root.identity_source() == IdentitySource::Hip {
            Some(g.hip.device_uuid(0).map_err(err)?)
        } else {
            None
        },
    };
    let active = devices::active_devices().ok_or("fabric must select the device")?;
    active
        .verify_visible(std::slice::from_ref(&seen))
        .map_err(err)?;
    let device = &active.devices[0];
    if device.uuid().as_deref() != Some("GPU-085289909a86cc63")
        || seen.pci_bus_id.to_lowercase() != "0000:e3:00.0"
    {
        return Err(format!(
            "wrong device uuid={:?} pci={}",
            device.uuid(),
            seen.pci_bus_id
        ));
    }
    eprintln!(
        "[devices] verified logical 0: HIP {} {} = index {} {} rocr={}",
        seen.arch,
        seen.pci_bus_id,
        device.index,
        device.lock_identity(),
        device.rocr_selector()
    );
    let mut cases = Vec::new();
    let mut failures = 0;
    for (m, k, ns) in [
        (5120usize, 10240usize, vec![1, 7, 8, 9, 63, 512, 1264, 1688]),
        (1024, 5120, vec![1, 8, 512, 1264, 1688]),
        (7, 256, vec![1, 7, 8, 9, 63]),
        (1027, 5120, vec![9, 1688]),
    ] {
        let mut seed = 0x5a771d235a771d23 ^ m as u64 ^ k as u64;
        let mut weights = Vec::with_capacity(m * (k / 256) * 136);
        for i in 0..m * (k / 256) {
            let scale = [0.0000001f32, 0.0001, 0.1, 1.0, 100.0, 10000.0][i % 6];
            let zero = (random(&mut seed) as i32 as f32 / i32::MAX as f32) * scale * 8.0;
            weights.extend_from_slice(&scale.to_le_bytes());
            weights.extend_from_slice(&zero.to_le_bytes());
            for _ in 0..32 {
                weights.extend_from_slice(&random(&mut seed).to_le_bytes());
            }
        }
        let a = g.upload_raw(&weights, &[weights.len()]).map_err(err)?;
        for n in ns {
            let xs: Vec<f32> = (0..n * k)
                .map(|_| random(&mut seed) as i32 as f32 / i32::MAX as f32)
                .collect();
            let x = g.upload_f32(&xs, &[n, k]).map_err(err)?;
            // Guard both ends and compare every byte, including guards.
            let ya = g.zeros(&[n * m + 128], DType::F32).map_err(err)?;
            let yb = g.zeros(&[n * m + 128], DType::F32).map_err(err)?;
            g.hip.memset(&ya.buf, 0x7f, ya.buf.size()).map_err(err)?;
            g.hip.memset(&yb.buf, 0x7f, yb.buf.size()).map_err(err)?;
            let av = ya.sub_offset(64, n * m);
            let bv = yb.sub_offset(64, n * m);
            g.gemm_hfq4g256(&a, &x, &av, m, k, n).map_err(err)?;
            g.gemm_hfq4g256_rows_pm(&a, &x, &bv, m, k, n).map_err(err)?;
            let ra = download(&g, &ya)?;
            let rb = download(&g, &yb)?;
            let first = ra.iter().zip(&rb).position(|(a, b)| a != b);
            let bad = ra.iter().zip(&rb).filter(|(a, b)| a != b).count();
            failures += usize::from(bad > 0);
            let mismatch = first.map(|i|json!({"byte":i,"row":i.saturating_sub(256)/4%m,"column":i.saturating_sub(256)/4/m,"incumbent":ra[i],"pm":rb[i]}));
            let mut row = json!({"M":m,"K":k,"N":n,"incumbent_sha256":sha(&ra),"pm_sha256":sha(&rb),"mismatch_bytes":bad,"first_mismatch":mismatch});
            if n == 1688 && (m == 5120 || m == 1024) {
                row["incumbent_timing"] =
                    timing(&mut g, |g| g.gemm_hfq4g256(&a, &x, &av, m, k, n))?;
                row["pm_timing"] =
                    timing(&mut g, |g| g.gemm_hfq4g256_rows_pm(&a, &x, &bv, m, k, n))?;
            }
            eprintln!("{row}");
            cases.push(row);
        }
    }
    Ok(
        json!({"schema":"pm-mtp-gemm-oracle-v1","identical":failures==0,"failures":failures,"cases":cases}),
    )
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .iter()
        .position(|v| v == "--out")
        .and_then(|i| args.get(i + 1))
        .expect("--out FILE required");
    match run() {
        Ok(v) => {
            std::fs::write(out, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
            if v["failures"] != 0 {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
