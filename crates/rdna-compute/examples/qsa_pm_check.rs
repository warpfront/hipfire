// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only byte-equality and timing of the gathered QSA route on the
//! hipcc kernels against the builder module (`HIPFIRE_QWEN4_QSA_PM`).
//!
//! `qsa_pm_check SNAPSHOT_DIR... --out OUT.jsonl [--time N]`: every
//! `qsa-source-v1` snapshot (`snapshot.json` under each directory, recursively,
//! `pre-prologue` dumps skipped) runs the producer and attention on both
//! paths over a poisoned output, compares the output bytes with each other
//! and with the snapshot's `eager.f32`, and for the first `N` snapshots
//! (default 0) times both paths ABBA, 20 calls per arm per round, 3 rounds.

use rdna_compute::tensor_ops::{indexed_attention_gathered_batch, IndexedAttentionAttentionBatch, QsaKvFormat};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String { format!("{e:?}") }
fn sha256(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }

fn snapshots(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(err)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "pre-prologue") { continue }
            snapshots(&p, out)?;
        } else if p.file_name().is_some_and(|n| n == "snapshot.json") {
            out.push(p);
        }
    }
    Ok(())
}

struct Case { q: GpuTensor, k: GpuTensor, v: GpuTensor, s: GpuTensor, out: GpuTensor, g: Value, fp8: bool }
impl Case {
    fn params(&self) -> IndexedAttentionAttentionBatch<'_> {
        let n = |k: &str| self.g[k].as_u64().unwrap() as usize;
        IndexedAttentionAttentionBatch {
            q_with_gate: &self.q, full_keys: &self.k, full_values: &self.v, selected: &self.s, output: &self.out,
            rows: n("rows"), position_start: n("position_start"), n_heads: n("heads"), n_kv_heads: n("kv_heads"), head_dim: n("dim"),
            budget_blocks: n("budget_blocks"), compress: n("compress"), capacity: n("capacity"), full_capacity: n("full_capacity"),
            format: if self.fp8 { QsaKvFormat::Fp8 } else { QsaKvFormat::F32 }, shape_selected: n("capacity"),
        }
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [self.q, self.k, self.v, self.s, self.out] { gpu.free_tensor(t).map_err(err)?; }
        Ok(())
    }
}

fn run(gpu: &mut Gpu, c: &Case, pm: bool) -> Result<Vec<u8>> {
    gpu.hip.memset(&c.out.buf, 0x7f, c.out.byte_size()).map_err(err)?;
    indexed_attention_gathered_batch(gpu, &c.params(), pm).map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0u8; c.out.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &c.out.buf).map_err(err)?;
    Ok(bytes)
}

fn median_ms(gpu: &mut Gpu, c: &Case, pm: bool) -> Result<f32> {
    let p = c.params();
    for _ in 0..3 { indexed_attention_gathered_batch(gpu, &p, pm).map_err(err)?; }
    gpu.hip.device_synchronize().map_err(err)?;
    let (a, b) = (gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?);
    let mut t = Vec::new();
    for _ in 0..20 {
        gpu.hip.event_record(&a, None).map_err(err)?;
        indexed_attention_gathered_batch(gpu, &p, pm).map_err(err)?;
        gpu.hip.event_record(&b, None).map_err(err)?;
        gpu.hip.event_synchronize(&b).map_err(err)?;
        t.push(gpu.hip.event_elapsed_ms(&a, &b).map_err(err)?);
    }
    gpu.hip.event_destroy(a).map_err(err)?;
    gpu.hip.event_destroy(b).map_err(err)?;
    t.sort_by(f32::total_cmp);
    Ok((t[9] + t[10]) * 0.5)
}

fn main() -> Result<()> {
    if std::env::var_os("HIPFIRE_LOCK_DIR").is_none() { return Err("private HIPFIRE_LOCK_DIR required".into()) }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut dirs, mut out, mut timed) = (Vec::new(), None, 0usize);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next(),
            "--time" => timed = it.next().ok_or("--time N")?.parse().map_err(err)?,
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    let out = out.ok_or("--out OUT.jsonl required")?;
    let mut paths = Vec::new();
    for d in &dirs { snapshots(d, &mut paths)?; }
    if paths.is_empty() { return Err("no snapshot.json found".into()) }
    let mut gpu = Gpu::init().map_err(err)?;
    gpu.dpm_warmup(10.0).map_err(err)?;
    let mut file = std::fs::File::create(&out).map_err(err)?;
    let (mut equal, mut eager_equal) = (0usize, 0usize);
    for (i, path) in paths.iter().enumerate() {
        let root = path.parent().unwrap();
        let header: Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if header["schema"] != "qsa-source-v1" { return Err(format!("{}: unsupported schema", path.display())) }
        let get = |name: &str| -> Result<Vec<u8>> {
            let b = std::fs::read(root.join(name)).map_err(err)?;
            if header["files"][name]["sha256"] != sha256(&b) { return Err(format!("{}: hash mismatch {name}", path.display())) }
            Ok(b)
        };
        let g = header["geometry"].clone();
        let fp8 = g["format"] == "fp8";
        if fp8 != gpu.arch_caps.is_gfx1201() { return Err(format!("{}: snapshot format {} on {}", path.display(), g["format"], gpu.arch)) }
        let f32s = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let q = f32s(get("qgate.f32")?);
        let cache = |gpu: &mut Gpu, b: Vec<u8>| if fp8 { gpu.upload_raw(&b, &[b.len()]).map_err(err) } else { let v = f32s(b); gpu.upload_f32(&v, &[v.len()]).map_err(err) };
        let k = cache(&mut gpu, get("keys.source")?)?;
        let v = cache(&mut gpu, get("values.source")?)?;
        let sb = get("selected.i32")?;
        let eager = get("eager.f32")?;
        let n = |k: &str| g[k].as_u64().unwrap() as usize;
        let c = Case {
            q: gpu.upload_f32(&q, &[q.len()]).map_err(err)?, k, v,
            s: gpu.upload_raw(&sb, &[sb.len()]).map_err(err)?,
            out: gpu.zeros(&[n("rows") * n("heads") * n("dim")], DType::F32).map_err(err)?,
            g: g.clone(), fp8,
        };
        let hip = run(&mut gpu, &c, false)?;
        let pm = run(&mut gpu, &c, true)?;
        let differing = hip.chunks_exact(4).zip(pm.chunks_exact(4)).filter(|(a, b)| a != b).count();
        let same = differing == 0;
        equal += same as usize;
        eager_equal += (pm == eager) as usize;
        let mut rec = json!({
            "snapshot": path, "identity": header["identity"], "geometry": g,
            "hip_sha256": sha256(&hip), "pm_sha256": sha256(&pm), "eager_sha256": sha256(&eager),
            "pm_equals_hip": same, "differing_f32": differing, "pm_equals_eager": pm == eager,
        });
        if i < timed {
            // ABBA, three rounds: per-arm medians of 20 calls each.
            let mut arms = json!({"hip": [], "pm": []});
            for _ in 0..3 {
                for pm_arm in [false, true, true, false] {
                    let ms = median_ms(&mut gpu, &c, pm_arm)?;
                    arms[if pm_arm { "pm" } else { "hip" }].as_array_mut().unwrap().push(json!(ms));
                }
            }
            rec["ms_per_call"] = arms;
        }
        writeln!(file, "{rec}").map_err(err)?;
        eprintln!("{}/{} {} equal={same} differing={differing} eager={}", i + 1, paths.len(), path.display(), pm == eager);
        c.free(&mut gpu)?;
    }
    println!("{}", json!({"arch": gpu.arch, "snapshots": paths.len(), "pm_equals_hip": equal, "pm_equals_eager": eager_equal, "out": out}));
    if equal != paths.len() { return Err(format!("{} of {} snapshots differ", paths.len() - equal, paths.len())) }
    Ok(())
}
