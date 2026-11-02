// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gp NT1 tile-epoch timing of one resident layer: E8P (one or two bounded variants) against the XTS MQ4 originals and
//! W3 (`mq3i4`) on the SAME recorded real route, the SAME selected expert ids, the SAME tiles/padded accounting
//! (the R6 `qwen4_gp_k2_timing` method), one card, one session, arms interleaved rep by rep.
//!
//! Timing: HIP events around each launch sequence (gate/up then down, stream order, one sync per pass), `--warmup`
//! untimed full passes (default 5) then `--reps` timed passes (default 30) per arm. Arms rotate their position every
//! rep so drift hits all equally. Verdict: aggregate-tiling pass (gate/up + down) tile-epoch throughput of each E8P
//! variant over XTS MQ4 >= 0.85 (`xts_median_s / e8p_median_s`); per-kind (gate/up, down) ratios are reported too.

use std::path::{Path, PathBuf};

use hip_bridge::{DeviceBuffer, Event, Function, HipRuntime};
use hipfire_arch_qwen4::expert_slab_table::build_expert_slab_table;
use hipfire_arch_qwen4::expert_slabs::{
    self, expert_entry_name, ExpertFormat, ExpertProj, EXPERTS_PER_LAYER,
};
use hipfire_runtime::hfq::HfqFile;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::common::*;
use super::hw::E8pModule;
use super::records::{E8pLayer, EXPERTS};
use super::route::{build_tiles, load_route, Route, TileSet, Tiling};
use super::twin::gen_x_timing;

pub const SCHEMA: &str = "gp.e8p.nt1.timing.v1";
const G1_SCHEMA: &str = "g1.k2.real";
const E8P_G1_SCHEMA: &str = "g1.e8p.real";
pub const THRESHOLD: f64 = 0.85;
const MIN_REPS: usize = 20;
const MIN_WARMUP: usize = 3;

pub const W3_SYMS: [&str; 2] = [
    "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1201",
    "qwen4_moe_down_iu4_sym_w3_pm_gfx1201",
];
pub const MQ4_SYMS: [&str; 2] = [
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
    "qwen4_moe_down_iu4_sym_pm_gfx1201",
];

fn expert_proj(p: Proj) -> ExpertProj {
    match p {
        Proj::GateUp => ExpertProj::GateUp,
        Proj::Down => ExpertProj::Down,
    }
}
fn mq4_qt(p: Proj) -> u8 {
    match p {
        Proj::GateUp => ExpertFormat::Mq4Xts.pair().0,
        Proj::Down => ExpertFormat::Mq4Xts.pair().1,
    }
}
fn mq4_group(p: Proj) -> u32 {
    match p {
        Proj::GateUp => 256,
        Proj::Down => 128,
    }
}
fn aggregate_name(layer: usize, proj: Proj) -> String {
    format!(
        "model.language_model.layers.{layer}.mlp.experts.{}",
        expert_proj(proj).suffix()
    )
}

// ---------------------------------------------------------------------------------------------------------------
// guards (files only; evaluated before any device is touched)
// ---------------------------------------------------------------------------------------------------------------

/// The W3/MQ4 real-G1 certificate (`g1.k2.real`), exactly the R6 guard.
fn g1_guard(
    path: &Path,
    artifact_sha: &str,
    w3_sha: &str,
    mq4_sha: &str,
    layer: usize,
) -> Result<Value> {
    let (g, raw) = read_json(path)?;
    let mut checks = serde_json::Map::new();
    let mut fail: Vec<String> = Vec::new();
    let mut chk = |name: &str, ok: bool| {
        checks.insert(name.into(), json!(ok));
        if !ok {
            fail.push(name.to_owned());
        }
    };
    chk("schema", g["schema"].as_str() == Some(G1_SCHEMA));
    chk("pass", g["pass"].as_bool() == Some(true));
    chk(
        "exact_mismatch_total_0",
        g["exact_mismatch_total"].as_u64() == Some(0),
    );
    chk("acc_mismatches_0", g["acc_mismatches"].as_u64() == Some(0));
    chk("h_mismatches_0", g["h_mismatches"].as_u64() == Some(0));
    chk(
        "capture_failures_0",
        g["capture_failures"].as_u64() == Some(0),
    );
    chk(
        "source_mode_full_artifact",
        g["source_mode"].as_str() == Some("full_artifact"),
    );
    chk(
        "artifact_sha256",
        g["artifact_sha256"].as_str() == Some(artifact_sha),
    );
    chk(
        "w3_codeobject_sha256",
        g["code"]["w3"]["codeobject"]["sha256"].as_str() == Some(w3_sha),
    );
    chk(
        "mq4_codeobject_sha256",
        g["code"]["mq4"]["codeobject"]["sha256"].as_str() == Some(mq4_sha),
    );
    chk(
        "w3_bundle_equals_elf",
        g["code"]["w3"]["bundle"]["device_entry_equals_loaded_elf"].as_bool() == Some(true),
    );
    chk(
        "mq4_bundle_equals_elf",
        g["code"]["mq4"]["bundle"]["device_entry_equals_loaded_elf"].as_bool() == Some(true),
    );
    let sym = &g["code"]["symbols"];
    chk(
        "w3_symbols",
        sym["w3_gate_up"].as_str() == Some(W3_SYMS[0])
            && sym["w3_down"].as_str() == Some(W3_SYMS[1]),
    );
    chk(
        "mq4_symbols",
        sym["mq4_gate_up"].as_str() == Some(MQ4_SYMS[0])
            && sym["mq4_down"].as_str() == Some(MQ4_SYMS[1]),
    );
    chk(
        "device_arch_gfx1201",
        g["device"]["arch"].as_str() == Some("gfx1201")
            && g["device"]["identity_match"].as_bool() == Some(true),
    );
    let certified: Vec<u64> = g["layers"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        .unwrap_or_default();
    chk("layer_certified", certified.contains(&(layer as u64)));
    if !fail.is_empty() {
        return Err(format!(
            "W3/MQ4 G1 guard {}: failed checks {fail:?} (no GPU time spent)",
            path.display()
        ));
    }
    Ok(
        json!({"receipt": path.display().to_string(), "receipt_file_sha256": sha(&raw), "checks": checks, "certified_layers": certified, "g1_device": g["device"].clone()}),
    )
}

/// The real-layer E8P G1 certificate (`g1.e8p.real`) of one variant: pass, zero mismatches, all 512 experts exact, same
/// code-object bytes, same symbols, same pinned record files.
#[allow(clippy::too_many_arguments)]
fn e8p_g1_guard(
    path: &Path,
    variant: &str,
    e8p_sha: &str,
    syms: &[String; 2],
    layer: usize,
    manifest_sha: &str,
    gate_sha: &str,
    down_sha: &str,
) -> Result<Value> {
    let (g, raw) = read_json(path)?;
    let mut checks = serde_json::Map::new();
    let mut fail: Vec<String> = Vec::new();
    let mut chk = |name: &str, ok: bool| {
        checks.insert(name.into(), json!(ok));
        if !ok {
            fail.push(name.to_owned());
        }
    };
    chk("schema", g["schema"].as_str() == Some(E8P_G1_SCHEMA));
    chk("pass", g["pass"].as_bool() == Some(true));
    chk("variant", g["variant"].as_str() == Some(variant));
    chk(
        "exact_mismatch_total_0",
        g["exact_mismatch_total"].as_u64() == Some(0),
    );
    chk("acc_mismatches_0", g["acc_mismatches"].as_u64() == Some(0));
    chk("h_mismatches_0", g["h_mismatches"].as_u64() == Some(0));
    chk(
        "mismatching_slots_0",
        g["mismatching_slots"].as_u64() == Some(0),
    );
    chk(
        "writer_guard_clean",
        g["writer_guard_clean"].as_bool() == Some(true),
    );
    chk(
        "layer_certified",
        g["layers"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x.as_u64() == Some(layer as u64))),
    );
    chk(
        "all_512_resident",
        g["scope"]["experts_resident"].as_u64() == Some(EXPERTS as u64),
    );
    chk(
        "coverage_gate_up_all_512_exact",
        g["scope"]["coverage"]["gate_up"]["all_512_exact"].as_bool() == Some(true),
    );
    chk(
        "coverage_down_all_512_exact",
        g["scope"]["coverage"]["down"]["all_512_exact"].as_bool() == Some(true),
    );
    chk(
        "e8p_codeobject_sha256",
        g["code"]["e8p"]["module"]["codeobject"]["sha256"].as_str() == Some(e8p_sha),
    );
    chk(
        "e8p_symbols",
        g["code"]["e8p"]["symbols"]["gate_up"].as_str() == Some(syms[0].as_str())
            && g["code"]["e8p"]["symbols"]["down"].as_str() == Some(syms[1].as_str()),
    );
    chk(
        "manifest_sha256",
        g["inputs"]["manifest"]["sha256"].as_str() == Some(manifest_sha),
    );
    chk(
        "gate_up_record_sha256",
        g["inputs"]["files"]["gate_up"]["sha256"].as_str() == Some(gate_sha),
    );
    chk(
        "down_record_sha256",
        g["inputs"]["files"]["down"]["sha256"].as_str() == Some(down_sha),
    );
    chk(
        "device_arch_gfx1201",
        g["device"]["arch"].as_str() == Some("gfx1201")
            && g["device"]["identity_match"].as_bool() == Some(true),
    );
    if !fail.is_empty() {
        return Err(format!(
            "E8P G1 guard {} ({variant}): failed checks {fail:?} (no GPU time spent)",
            path.display()
        ));
    }
    Ok(
        json!({"receipt": path.display().to_string(), "receipt_file_sha256": sha(&raw), "variant": variant, "checks": checks, "g1_device": g["device"].clone(),
        "g1_scope": g["scope"]["coverage"].clone()}),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// expert acquisition
// ---------------------------------------------------------------------------------------------------------------

/// Same ranking as `qwen4_moe_sym_real_g1::select_experts`, W3 members only.
fn g1_w3_ids(layer: usize, row: &[ExpertFormat], per: usize) -> Result<Vec<usize>> {
    let mut v: Vec<([u8; 32], usize)> = row
        .iter()
        .enumerate()
        .filter(|(_, f)| **f == ExpertFormat::Mq3I4)
        .map(|(e, _)| {
            (
                Sha256::digest(format!("hipfire.g1real.select.v1|L{layer}|w3|E{e}").as_bytes())
                    .into(),
                e,
            )
        })
        .collect();
    v.sort();
    if v.len() < per {
        return Err(format!("layer {layer}: {} W3 experts, need {per}", v.len()));
    }
    Ok(v.into_iter().take(per).map(|x| x.1).collect())
}

fn source_slice<'a>(src: &'a HfqFile, layer: usize, expert: usize, proj: Proj) -> Result<&'a [u8]> {
    let name = aggregate_name(layer, proj);
    let (info, data) = src
        .tensor_data(&name)
        .ok_or_else(|| format!("source: aggregate `{name}` missing"))?;
    let per = ExpertFormat::Mq4Xts.gate_up_bytes() * (1 - proj.idx())
        + ExpertFormat::Mq4Xts.down_bytes() * proj.idx();
    let want = vec![EXPERTS_PER_LAYER, proj.m() as u32, proj.k() as u32];
    if info.quant_type != mq4_qt(proj)
        || info.shape != want
        || info.group_size != mq4_group(proj)
        || data.len() != EXPERTS_PER_LAYER as usize * per
    {
        return Err(format!(
            "source aggregate `{name}`: qt{} shape {:?} group {} {} B != expected qt{} shape {want:?} group {} {} B",
            info.quant_type,
            info.shape,
            info.group_size,
            data.len(),
            mq4_qt(proj),
            mq4_group(proj),
            EXPERTS_PER_LAYER as usize * per
        ));
    }
    Ok(&data[expert * per..(expert + 1) * per])
}

struct ExpertBytes<'a> {
    id: usize,
    w3: Option<[&'a [u8]; 2]>,
    xts: [&'a [u8]; 2],
    e8p: [&'a [u8]; 2],
    span_json: Value,
}

// ---------------------------------------------------------------------------------------------------------------
// device arms
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArmKind {
    W3,
    Xts,
    E8p,
}

struct Arm<'a> {
    kind: ArmKind,
    /// `gp_bench.py`-style route name: `mq3i4`, `xts_mq4`, `e8p_<variant>`.
    route: String,
    syms: [String; 2],
    co_sha: String,
    fns: &'a [Function; 2],
    ka: [usize; 2],
    slab: DeviceBuffer,
    ptrs: [DeviceBuffer; 2],
    slab_bytes: usize,
}

fn upload_arm<'a>(
    hip: &HipRuntime,
    kind: ArmKind,
    route: String,
    syms: [String; 2],
    co_sha: String,
    fns: &'a [Function; 2],
    ka: [usize; 2],
    blobs: &[[&[u8]; 2]],
) -> Result<Arm<'a>> {
    let mut offsets = Vec::with_capacity(blobs.len());
    let mut cur = GUARD;
    for b in blobs {
        offsets.push(cur);
        cur = align(cur + b[0].len() + b[1].len()) + GUARD;
    }
    let slab = hip.malloc(cur).map_err(err)?;
    hip.memset(&slab, 0xff, cur).map_err(err)?;
    for (b, &o) in blobs.iter().zip(&offsets) {
        hip.memcpy_htod_offset(&slab, o, b[0]).map_err(err)?;
        hip.memcpy_htod_offset(&slab, o + b[0].len(), b[1])
            .map_err(err)?;
    }
    let base = slab.as_ptr() as u64;
    let table = |p: Proj| -> Vec<u8> {
        blobs
            .iter()
            .zip(&offsets)
            .flat_map(|(b, o)| {
                (base + (*o + if p == Proj::Down { b[0].len() } else { 0 }) as u64).to_le_bytes()
            })
            .collect()
    };
    let ptrs = [up(hip, &table(Proj::GateUp))?, up(hip, &table(Proj::Down))?];
    Ok(Arm {
        kind,
        route,
        syms,
        co_sha,
        fns,
        ka,
        slab,
        ptrs,
        slab_bytes: cur,
    })
}

struct Launch {
    proj: Proj,
    ntiles: usize,
    /// Index into the tile-set device buffers shared by gate_up and down of one set.
    set: usize,
    x: DeviceBuffer,
    x_rows: usize,
    div: usize,
}

struct PassTimes {
    gate_up_s: f64,
    down_s: f64,
    pass_s: f64,
}

fn one_pass(
    hip: &HipRuntime,
    arm: &Arm,
    launches: &[Launch],
    blobs: &mut [Vec<u8>],
    ev: &[Event],
) -> Result<PassTimes> {
    hip.event_record(&ev[0], None).map_err(err)?;
    for (i, (l, blob)) in launches.iter().zip(blobs.iter_mut()).enumerate() {
        // SAFETY: the shared PM 64-byte kernarg ABI (5 pointers, 6 u32 incl. n_experts@60 = the table's actual entry count), pointers are live device buffers that outlive the
        // synchronize below; the blob is copied by the launch.
        unsafe {
            hip.launch_kernel_blob(
                &arm.fns[l.proj.idx()],
                [l.proj.grid_x(), l.ntiles as u32, 1],
                [l.proj.threads(), 1, 1],
                0,
                None,
                blob,
            )
        }
        .map_err(err)?;
        hip.event_record(&ev[i + 1], None).map_err(err)?;
    }
    hip.event_synchronize(&ev[launches.len()]).map_err(err)?;
    let (mut gu, mut dn) = (0f64, 0f64);
    for (i, l) in launches.iter().enumerate() {
        let s = f64::from(hip.event_elapsed_ms(&ev[i], &ev[i + 1]).map_err(err)?) / 1000.0;
        match l.proj {
            Proj::GateUp => gu += s,
            Proj::Down => dn += s,
        }
    }
    let pass = f64::from(
        hip.event_elapsed_ms(&ev[0], &ev[launches.len()])
            .map_err(err)?,
    ) / 1000.0;
    Ok(PassTimes {
        gate_up_s: gu,
        down_s: dn,
        pass_s: pass,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// main flow
// ---------------------------------------------------------------------------------------------------------------

fn need<T: Clone>(o: &Option<T>, what: &str) -> Result<T> {
    o.clone()
        .ok_or_else(|| format!("timing mode requires {what}"))
}

pub struct TimingPrep<'a> {
    pub args: &'a super::Args,
    pub argv: &'a [String],
    pub layer: &'a E8pLayer,
    pub harness: &'a Value,
    pub out_dir: &'a Path,
    pub records_file: &'a Value,
}

struct VariantRun {
    name: String,
    co: PathBuf,
    img: Vec<u8>,
    bind: Value,
    g1: Value,
}

pub fn run_timing(p: &TimingPrep) -> Result<()> {
    let a = p.args;
    let layer_no = a.layer;
    if a.reps < MIN_REPS || a.warmup < MIN_WARMUP {
        return Err(format!(
            "--reps must be >= {MIN_REPS} and --warmup >= {MIN_WARMUP} (gp_bench.py refuses fewer)"
        ));
    }
    if !a.tilings.contains(&Tiling::Aggregate) {
        return Err(
            "the verdict is taken on the aggregate tiling: --tiling must be aggregate or both"
                .into(),
        );
    }
    let (w3_co, mq4_co, w3_sha, mq4_sha, g1_path, art_path, art_sha, src_path, src_sha, trace_dir) = (
        need(&a.w3_co, "--w3-codeobject")?,
        need(&a.mq4_co, "--mq4-codeobject")?,
        need(&a.w3_sha, "--w3-sha256")?,
        need(&a.mq4_sha, "--mq4-sha256")?,
        need(&a.g1_receipt, "--g1-receipt")?,
        need(&a.artifact, "--artifact")?,
        need(&a.artifact_sha, "--artifact-sha256")?,
        need(&a.source, "--source")?,
        need(&a.source_sha, "--source-sha256")?,
        need(&a.trace_dir, "--trace-dir")?,
    );
    let (manifest_sha, gate_sha, down_sha) = (
        need(&a.e8p_manifest_sha, "--e8p-manifest-sha256")?,
        need(&a.e8p_gate_up_sha, "--e8p-gate-up-sha256")?,
        need(&a.e8p_down_sha, "--e8p-down-sha256")?,
    );
    let out_dir = p.out_dir;
    let receipt_path = out_dir.join("gp.e8p.timing.receipt.json");

    // ---- guards that need no device ------------------------------------------------------------------------
    let (w3_img, mq4_img) = (read_file(&w3_co)?, read_file(&mq4_co)?);
    let w3_bind = bind_module("w3", &w3_co, &w3_img, Some(&w3_sha), a.w3_bundle.as_deref())?;
    let mq4_bind = bind_module(
        "mq4",
        &mq4_co,
        &mq4_img,
        Some(&mq4_sha),
        a.mq4_bundle.as_deref(),
    )?;
    let g1 = g1_guard(&g1_path, &art_sha, &w3_sha, &mq4_sha, layer_no)?;
    let syms = a.e8p_symbols();
    let mut variants: Vec<VariantRun> = Vec::new();
    for name in &a.variants {
        let cfg = a.variant_cfg(name)?;
        let img = read_file(&cfg.co)?;
        let bind = bind_module(
            &format!("e8p/{name}"),
            &cfg.co,
            &img,
            cfg.sha.as_deref(),
            cfg.bundle.as_deref(),
        )?;
        let g1p = cfg.g1_receipt.clone().ok_or_else(|| {
            format!(
                "variant {name}: its G1 receipt flag is required (the E8P G1 guard gates timing)"
            )
        })?;
        let g1v = e8p_g1_guard(
            &g1p,
            name,
            &sha(&img),
            &syms,
            layer_no,
            &manifest_sha,
            &gate_sha,
            &down_sha,
        )?;
        variants.push(VariantRun {
            name: name.clone(),
            co: cfg.co,
            img,
            bind,
            g1: g1v,
        });
    }

    let art_canon = std::fs::canonicalize(&art_path).map_err(err)?;
    let hfq = HfqFile::open(&art_canon).map_err(|e| format!("{}: {e}", art_canon.display()))?;
    if let Some(b) = a.artifact_bytes {
        if b != hfq.file_len() {
            return Err(format!(
                "artifact is {} bytes, --artifact-bytes {b}",
                hfq.file_len()
            ));
        }
    }
    let meta = expert_slabs::parse(&hfq.metadata_json)
        .map_err(err)?
        .ok_or("artifact has no qwen4.expert_slabs metadata")?;
    let table = build_expert_slab_table(&meta, hfq.tensors())
        .map_err(|e| format!("ExpertSlabTable: {e}"))?;
    let src_canon = std::fs::canonicalize(&src_path).map_err(err)?;
    let src = HfqFile::open(&src_canon).map_err(|e| format!("{}: {e}", src_canon.display()))?;
    if let Some(b) = a.source_bytes {
        if b != src.file_len() {
            return Err(format!(
                "source is {} bytes, --source-bytes {b}",
                src.file_len()
            ));
        }
    }
    let pinned =
        |what: &str| format!("{what} pinned by CLI sha256 (file not re-hashed by this run)");

    // ---- route, scope, tiles ----------------------------------------------------------------------------------
    let route: Route = load_route(
        &trace_dir.join(format!("layer{layer_no}.trace.json")),
        a.route_bin.as_deref(),
        layer_no,
    )?;
    let row: Vec<ExpertFormat> = (0..EXPERTS_PER_LAYER as usize)
        .map(|e| {
            table
                .slab(layer_no, e)
                .map(|s| s.format)
                .ok_or(format!("slab table has no L{layer_no} E{e}"))
        })
        .collect::<Result<_>>()?;
    let w3_all: Vec<usize> = (0..row.len())
        .filter(|&e| row[e] == ExpertFormat::Mq3I4)
        .collect();
    let mq4_all: Vec<usize> = (0..row.len())
        .filter(|&e| row[e] == ExpertFormat::Mq4Xts)
        .collect();
    let full_layer_scope = a.scope == "full_layer";
    let mut pool: Vec<usize> = match a.scope.as_str() {
        "w3_pool" => w3_all
            .iter()
            .copied()
            .filter(|&e| route.rows_per_expert[e] > 0)
            .collect(),
        "g1_w3" => g1_w3_ids(layer_no, &row, a.per_codec)?,
        "full_layer" => (0..row.len())
            .filter(|&e| route.rows_per_expert[e] > 0)
            .collect(),
        _ => a.expert_ids.clone(),
    };
    pool.sort_unstable();
    pool.dedup();
    if !full_layer_scope {
        if let Some(&bad) = pool
            .iter()
            .find(|&&e| e >= row.len() || row[e] != ExpertFormat::Mq3I4)
        {
            return Err(format!(
                "L{layer_no} E{bad}: not a W3 expert in the packed artifact (format {:?})",
                row.get(bad)
            ));
        }
    }
    if let Some(&z) = pool.iter().find(|&&e| route.rows_per_expert[e] == 0) {
        return Err(format!(
            "L{layer_no} E{z}: the real route sends no row to this expert (nothing to time)"
        ));
    }
    if pool.is_empty() {
        return Err(format!("L{layer_no}: empty scope"));
    }
    let mut pool_pos = vec![-1i32; row.len()];
    for (pi, &e) in pool.iter().enumerate() {
        pool_pos[e] = pi as i32;
    }
    let rpe = &route.rows_per_expert;
    let selected_rows: usize = pool.iter().map(|&e| rpe[e]).sum();
    let total_rows: usize = rpe.iter().sum();
    let hot_rows: usize = mq4_all.iter().map(|&e| rpe[e]).sum();
    let tiles_agg: usize = pool.iter().map(|&e| rpe[e].div_ceil(TILE_ROWS)).sum();
    let mut tiles_chunked = 0usize;
    let mut per_chunk_rows: Vec<Vec<usize>> = Vec::new();
    for c in &route.chunks {
        let mut cnt = vec![0usize; row.len()];
        for &e in &c.ids {
            cnt[e as usize] += 1;
        }
        tiles_chunked += pool
            .iter()
            .map(|&e| cnt[e].div_ceil(TILE_ROWS))
            .sum::<usize>();
        per_chunk_rows.push(pool.iter().map(|&e| cnt[e]).collect());
    }
    let hit_experts = rpe.iter().filter(|&&r| r > 0).count();
    let scope_json = json!({
        "mode": a.scope, "layer": layer_no, "w3_experts_in_layer": w3_all.len(), "mq4_experts_in_layer": mq4_all.len(),
        "w3_experts_hit_by_route": w3_all.iter().filter(|&&e| rpe[e] > 0).count(), "experts_hit_by_route": hit_experts,
        "selected_experts": pool, "selected_expert_count": pool.len(), "selected_routed_rows": selected_rows,
        "full_layer_routed_rows": total_rows, "token_count": route.token_count, "top_k": route.topk,
        "selected_fraction_of_layer_rows": selected_rows as f64 / total_rows as f64,
        "hot_mq4_routed_rows_outside_scope": if full_layer_scope { 0 } else { hot_rows },
        "w3_routed_rows_outside_scope": w3_all.iter().filter(|e| pool_pos[**e] < 0).map(|&e| rpe[e]).sum::<usize>(),
        "full_layer": if full_layer_scope { pool.len() == hit_experts } else { pool.len() == w3_all.iter().filter(|&&e| rpe[e] > 0).count() && hot_rows == 0 },
        "arms": if full_layer_scope { json!(["xts_mq4", "e8p"]) } else { json!(["mq3i4", "xts_mq4", "e8p"]) },
        "claim": if full_layer_scope {
            "full routed layer: every expert the real route hits, E8P (real solved records) vs original XTS MQ4 of the same ids; W3 has no twin for the hot-MQ4 experts so no W3 arm in this scope"
        } else {
            "resident-subset measurement: the selected W3 experts of the packed artifact, their original XTS MQ4 and their real E8P records, same ids on the same real route; NOT a full-layer or full-model speed claim unless full_layer is true"
        },
        "tiles": {"aggregate": tiles_agg, "chunked": tiles_chunked, "tile_rows": TILE_ROWS, "k_epochs": {"gate_up": Proj::GateUp.groups(), "down": Proj::Down.groups(), "pass": Proj::GateUp.groups() + Proj::Down.groups()}},
        "selected_rows_per_chunk": per_chunk_rows, "chunks": route.chunks.iter().map(|c| json!({"rows": c.rows, "start_pos": c.start_pos})).collect::<Vec<_>>(),
    });
    let mut trace_out = route.trace_json.clone();
    {
        let obj = trace_out
            .as_object_mut()
            .ok_or("trace json is not an object")?;
        obj.insert("selected_experts".into(), json!(pool));
        obj.insert("selected_routed_rows".into(), json!(selected_rows));
        obj.insert("original_trace_sha256".into(), json!(route.route_sha256));
        obj.insert("driver_scope".into(), scope_json.clone());
    }
    let mut tilesets: Vec<(Tiling, Vec<TileSet>)> = Vec::new();
    for &t in &a.tilings {
        tilesets.push((t, build_tiles(&route, &pool_pos, pool.len(), t)?));
    }
    for (t, sets) in &tilesets {
        let (n, live): (usize, usize) = (
            sets.iter().map(|s| s.ntiles()).sum(),
            sets.iter().map(|s| s.live_slots).sum(),
        );
        let want = if *t == Tiling::Aggregate {
            tiles_agg
        } else {
            tiles_chunked
        };
        if n != want || live != selected_rows {
            return Err(format!("L{layer_no} {}: built {n} tiles / {live} live slots, expected {want} / {selected_rows}", t.name()));
        }
    }

    // ---- bytes: W3 (packed), XTS (original) and E8P (real records) of the same ids ---------------------------
    let mut slice_checks = Vec::new();
    {
        let mut checked = Vec::new();
        for e in (0..EXPERTS_PER_LAYER as usize)
            .filter(|&e| {
                table
                    .slab(layer_no, e)
                    .is_some_and(|s| s.format == ExpertFormat::Mq4Xts)
            })
            .take(3)
        {
            for proj in Proj::ALL {
                let (_, packed) = hfq
                    .tensor_data(&expert_entry_name(layer_no, e, expert_proj(proj)))
                    .ok_or_else(|| {
                        format!("L{layer_no} E{e} {}: packed tensor missing", proj.name())
                    })?;
                if packed != source_slice(&src, layer_no, e, proj)? {
                    return Err(format!("L{layer_no} E{e} {}: packed hot-MQ4 bytes differ from the original XTS slice (source slicing is wrong or the artifact was repacked)", proj.name()));
                }
            }
            checked.push(e);
        }
        slice_checks.push(
            json!({"layer": layer_no, "hot_mq4_experts_byte_identical_to_source_slices": checked}),
        );
    }
    let mut bytes: Vec<ExpertBytes> = Vec::new();
    for &e in &pool {
        let mut w3: Option<[&[u8]; 2]> = None;
        let mut xts: [&[u8]; 2] = [&[], &[]];
        let mut spans = serde_json::Map::new();
        for proj in Proj::ALL {
            xts[proj.idx()] = source_slice(&src, layer_no, e, proj)?;
            let mut ent = serde_json::Map::new();
            ent.insert("xts_original".into(), json!({"tensor": aggregate_name(layer_no, proj), "slice_offset_in_tensor": e * xts[proj.idx()].len(), "len": xts[proj.idx()].len(), "quant_type": mq4_qt(proj), "sha256": sha(xts[proj.idx()])}));
            if row[e] == ExpertFormat::Mq3I4 {
                let slab = table.slab(layer_no, e).ok_or("slab")?;
                let name = expert_entry_name(layer_no, e, expert_proj(proj));
                let (info, data) = hfq
                    .tensor_data(&name)
                    .ok_or_else(|| format!("{name}: tensor_data missing"))?;
                let (span, qt) = match proj {
                    Proj::GateUp => (slab.gate_up, ExpertFormat::Mq3I4.pair().0),
                    Proj::Down => (slab.down, ExpertFormat::Mq3I4.pair().1),
                };
                if info.data_offset != span.offset
                    || info.data_size != span.len
                    || data.len() != span.len
                    || info.quant_type != qt
                {
                    return Err(format!("{name}: HFQ entry != table span / qt{qt}"));
                }
                w3.get_or_insert([&[], &[]])[proj.idx()] = data;
                ent.insert("packed_w3".into(), json!({"tensor": name, "offset": span.offset, "len": span.len, "quant_type": qt, "sha256": sha(data)}));
            }
            let blob = &p.layer.blobs[e][proj.idx()];
            ent.insert(
                "e8p_record".into(),
                json!({"expert": e, "len": blob.len(), "blob_sha256": sha(blob)}),
            );
            spans.insert(proj.name().into(), Value::Object(ent));
        }
        bytes.push(ExpertBytes {
            id: e,
            w3,
            xts,
            e8p: [
                p.layer.blobs[e][0].as_slice(),
                p.layer.blobs[e][1].as_slice(),
            ],
            span_json: Value::Object(spans),
        });
    }

    let variant_names: Vec<String> = variants.iter().map(|v| v.name.clone()).collect();
    let common = |layer_json: Vec<Value>, status: &str| -> Value {
        json!({
            "schema": SCHEMA, "status": status, "slice": "R8", "gate": "Gp", "arch": "gfx1201", "nt": 1, "tile_rows": TILE_ROWS,
            "threshold": THRESHOLD, "variants": variant_names,
            "g1_guard": {"w3_mq4": g1, "e8p": variants.iter().map(|v| v.g1.clone()).collect::<Vec<_>>()},
            "objects": {"w3": w3_bind, "mq4": mq4_bind, "e8p": variants.iter().map(|v| json!({"variant": v.name, "binding": v.bind, "symbols": syms})).collect::<Vec<_>>(),
                "symbols": {"mq3i4": W3_SYMS, "xts_mq4": MQ4_SYMS, "e8p": syms}},
            "artifact": {"path": art_canon.display().to_string(), "bytes": hfq.file_len(), "sha256": art_sha, "sha256_origin": pinned("artifact"),
                "slab_counts": {"w3": table.counts().w3, "w2": table.counts().w2, "mq4": table.counts().mq4}},
            "source_xts": {"path": src_canon.display().to_string(), "bytes": src.file_len(), "sha256": src_sha, "sha256_origin": pinned("source"), "slicing_checks": slice_checks.clone()},
            "e8p_inputs": p.layer.inputs, "e8p_records_file": p.records_file,
            "reps": a.reps, "warmup": a.warmup, "arms_interleaved": "rotated every rep",
            "activation_provenance": "deterministic bounded synthetic A4 blocks (d = 1..16/1024, random signed nibbles, correct block sums); the real route fixes which rows/slots each expert sees, NOT the activation values; kernel time assumed activation-value independent (assumption, not measured)",
            "weights_provenance": "real: packed artifact W3 experts; original XTS MQ4 slices of the same ids; real solved E8P records (vq-e8p, BF16 source) of the same ids",
            "layers": layer_json,
            "command": {"argv": p.argv, "cwd": std::env::current_dir().ok().map(|d| d.display().to_string()), "rc": "exit code recorded by the fabric wrapper (0 timed/plan ok, 1 error)"},
            "code": {"harness": p.harness},
        })
    };
    let tp = out_dir
        .join(format!("L{layer_no}"))
        .join("route_trace.json");
    let tb = write_json(&tp, &trace_out)?;
    let experts_json: Vec<Value> = bytes
        .iter()
        .map(|b| {
            let rows = rpe[b.id];
            json!({"expert": b.id, "routed_rows": rows, "tiles_aggregate": rows.div_ceil(TILE_ROWS), "bytes": b.span_json.clone()})
        })
        .collect();
    let layer_receipt = json!({"layer": layer_no, "scope": scope_json.clone(), "route": {"trace": route.trace_path.display().to_string(), "trace_file_sha256": route.trace_raw_sha256,
        "route_bin": route.route_path.display().to_string(), "route_bin_sha256": route.route_sha256, "route_bin_bytes": route.route_bytes,
        "token_count": route.token_count, "top_k": route.topk, "counts_rederived_from_route_bin": true},
        "route_trace_file": {"path": tp.display().to_string(), "sha256": sha(&tb)}, "experts": experts_json, "results": []});
    if a.plan_only {
        let r = common(vec![layer_receipt], "plan_only_no_timing");
        let b = write_json(&receipt_path, &r)?;
        println!(
            "{}",
            json!({"schema": SCHEMA, "status": "plan_only_no_timing", "receipt": receipt_path.display().to_string(), "receipt_sha256": sha(&b)})
        );
        return Ok(());
    }

    // ---- device ---------------------------------------------------------------------------------------------------
    let card = need(&a.card, "--card")?;
    let fabric = fabric_provenance(
        &card,
        "timing",
        a.fabric_owner.as_deref(),
        a.fabric_wrapper.as_deref(),
    )?;
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (
        hip.device_uuid(id).map_err(err)?,
        hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase(),
        hip.get_arch(id).map_err(err)?,
    );
    if arch != "gfx1201" || a.expect_bdf.as_deref().is_some_and(|b| b != bdf) {
        return Err(format!(
            "device identity mismatch: arch={arch} bdf={bdf} expected gfx1201 bdf={:?}",
            a.expect_bdf
        ));
    }
    let device = json!({"arch": arch, "card": card, "uuid": uuid, "bdf": bdf});
    let g1_uuid = g1["g1_device"]["uuid"].as_str().map(str::to_owned);
    let device_full = json!({"arch": arch, "card": card, "uuid": uuid, "bdf": bdf, "device_id": id,
        "same_uuid_as_w3_mq4_g1_receipt": g1_uuid.as_deref() == Some(uuid.as_str()),
        "same_uuid_as_e8p_g1_receipts": variants.iter().map(|v| json!({"variant": v.name, "same": v.g1["g1_device"]["uuid"].as_str() == Some(uuid.as_str())})).collect::<Vec<_>>(),
        "env": {"HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(), "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok()}});
    let (m_w3, m_mq4) = (
        hip.module_load_data(&w3_img).map_err(err)?,
        hip.module_load_data(&mq4_img).map_err(err)?,
    );
    let get = |m: &hip_bridge::Module, n: &str| {
        hip.module_get_function(m, n)
            .map_err(|e| format!("{n}: {e}"))
    };
    let f3 = [get(&m_w3, W3_SYMS[0])?, get(&m_w3, W3_SYMS[1])?];
    let fx = [get(&m_mq4, MQ4_SYMS[0])?, get(&m_mq4, MQ4_SYMS[1])?];
    let mut e8p_mods: Vec<E8pModule> = Vec::new();
    for v in &variants {
        let cfg = a.variant_cfg(&v.name)?;
        e8p_mods.push(super::hw::load_e8p(
            &hip,
            &v.co,
            &v.img,
            cfg.sha.as_deref(),
            cfg.bundle.as_deref(),
            syms.clone(),
            false,
        )?);
    }
    let events: Vec<Event> = (0..=2 * tilesets.iter().map(|(_, s)| s.len()).max().unwrap_or(1))
        .map(|_| hip.event_create().map_err(err))
        .collect::<Result<_>>()?;

    let mut arms: Vec<Arm> = Vec::new();
    if !full_layer_scope {
        let blobs: Vec<[&[u8]; 2]> = bytes
            .iter()
            .map(|b| {
                b.w3.ok_or_else(|| format!("expert {} has no W3 bytes", b.id))
            })
            .collect::<Result<_>>()?;
        arms.push(upload_arm(
            &hip,
            ArmKind::W3,
            "mq3i4".into(),
            W3_SYMS.map(String::from),
            w3_sha.clone(),
            &f3,
            [64, 64],
            &blobs,
        )?);
    }
    {
        let blobs: Vec<[&[u8]; 2]> = bytes.iter().map(|b| b.xts).collect();
        arms.push(upload_arm(
            &hip,
            ArmKind::Xts,
            "xts_mq4".into(),
            MQ4_SYMS.map(String::from),
            mq4_sha.clone(),
            &fx,
            [64, 64],
            &blobs,
        )?);
    }
    for (v, md) in variants.iter().zip(&e8p_mods) {
        let blobs: Vec<[&[u8]; 2]> = bytes.iter().map(|b| b.e8p).collect();
        arms.push(upload_arm(
            &hip,
            ArmKind::E8p,
            format!("e8p_{}", v.name),
            md.syms.clone(),
            md.image_sha.clone(),
            &md.ship,
            md.ka,
            &blobs,
        )?);
    }
    let n_arms = arms.len();
    let xts_i = arms.iter().position(|x| x.kind == ArmKind::Xts).unwrap();
    let w3_i = arms.iter().position(|x| x.kind == ArmKind::W3);

    let mut results: Vec<Value> = Vec::new();
    let mut verdict_src: Option<Value> = None;
    for (tiling, sets) in &tilesets {
        let nonempty: Vec<&TileSet> = sets.iter().filter(|s| s.ntiles() > 0).collect();
        let max_tiles = nonempty.iter().map(|s| s.ntiles()).max().unwrap_or(0);
        let mut ybuf: Vec<DeviceBuffer> = Vec::new();
        for proj in Proj::ALL {
            ybuf.push(
                hip.malloc(max_tiles * TILE_ROWS * proj.h() * 2)
                    .map_err(err)?,
            );
        }
        let mut set_dev: Vec<(DeviceBuffer, DeviceBuffer)> = Vec::new();
        let mut launches: Vec<Launch> = Vec::new();
        for (si, s) in nonempty.iter().enumerate() {
            set_dev.push((
                up(&hip, &i32s_le(&s.tile_expert))?,
                up(&hip, &i32s_le(&s.sorted))?,
            ));
            for proj in Proj::ALL {
                let (rows, div) = match proj {
                    Proj::GateUp => (s.tokens, s.topk),
                    Proj::Down => (s.tokens * s.topk, 1),
                };
                let key = a.seed
                    ^ (layer_no as u64).wrapping_mul(0x1000_0000_01B3)
                    ^ ((si as u64 + 1) << 40)
                    ^ ((proj.idx() as u64 + 1) << 56);
                let x = up(&hip, &gen_x_timing(proj.groups(), rows, key))?;
                launches.push(Launch {
                    proj,
                    ntiles: s.ntiles(),
                    set: si,
                    x,
                    x_rows: rows,
                    div,
                });
            }
        }
        let routed_rows: usize = nonempty.iter().map(|s| s.live_slots).sum();
        let tiles_total: usize = nonempty.iter().map(|s| s.ntiles()).sum();
        let mut blobs: Vec<Vec<Vec<u8>>> = Vec::new();
        for arm in &arms {
            blobs.push(
                launches
                    .iter()
                    .map(|l| {
                        kernarg(
                            &arm.ptrs[l.proj.idx()],
                            &set_dev[l.set].0,
                            &set_dev[l.set].1,
                            &l.x,
                            &ybuf[l.proj.idx()],
                            l.proj.m(),
                            l.proj.k(),
                            l.div,
                            l.ntiles * TILE_ROWS,
                            l.x_rows,
                            arm.ka[l.proj.idx()],
                        )
                    })
                    .collect(),
            );
        }
        // Liveness: sentinel-filled outputs must be overwritten at the first slot of the last launch of each projection.
        let mut liveness = Vec::new();
        for (ai, arm) in arms.iter().enumerate() {
            for proj in Proj::ALL {
                hip.memset(
                    &ybuf[proj.idx()],
                    0xa5,
                    max_tiles * TILE_ROWS * proj.h() * 2,
                )
                .map_err(err)?;
            }
            one_pass(&hip, arm, &launches, &mut blobs[ai], &events)?;
            for proj in Proj::ALL {
                let mut r = vec![0u8; proj.h() * 2];
                hip.memcpy_dtoh(&mut r, &ybuf[proj.idx()]).map_err(err)?;
                let sentinel = r
                    .chunks_exact(2)
                    .filter(|c| c[0] == 0xa5 && c[1] == 0xa5)
                    .count();
                if sentinel > proj.h() / 2 {
                    return Err(format!("L{layer_no} {} {} {}: first output row still sentinel ({sentinel}/{} elements): kernel did not write", tiling.name(), arm.route, proj.name(), proj.h()));
                }
                liveness.push(json!({"arm": arm.route, "proj": proj.name(), "sentinel_elements_in_first_row": sentinel, "row_elements": proj.h()}));
            }
        }
        // Warmup then rotated timed reps.
        for w in 0..a.warmup {
            for i in 0..n_arms {
                let ai = (i + w) % n_arms;
                one_pass(&hip, &arms[ai], &launches, &mut blobs[ai], &events)?;
            }
        }
        let mut samples: Vec<Vec<[f64; 3]>> = vec![Vec::new(); n_arms];
        for r in 0..a.reps {
            for i in 0..n_arms {
                let ai = (i + r) % n_arms;
                let t = one_pass(&hip, &arms[ai], &launches, &mut blobs[ai], &events)?;
                samples[ai].push([t.gate_up_s, t.down_s, t.pass_s]);
            }
        }
        // Per label: runs (gp_bench driver-run schema per arm), medians, ratios, verdict.
        let (k_gu, k_dn) = (Proj::GateUp.groups(), Proj::Down.groups());
        let labels: [(&str, usize, usize); 3] = [
            ("pass", k_gu + k_dn, 2),
            ("gate_up", k_gu, 0),
            ("down", k_dn, 1),
        ];
        let mut files = Vec::new();
        let mut ratios = serde_json::Map::new();
        for (label, kep, sel) in &labels {
            let mut runs = Vec::new();
            let per_arm: Vec<Vec<f64>> = (0..n_arms)
                .map(|ai| samples[ai].iter().map(|t| t[*sel]).collect())
                .collect();
            let meds: Vec<f64> = per_arm.iter().map(|s| median(s)).collect();
            for (ai, arm) in arms.iter().enumerate() {
                let symbol = match *label {
                    "pass" => format!("{}+{}", arm.syms[0], arm.syms[1]),
                    "gate_up" => arm.syms[0].clone(),
                    _ => arm.syms[1].clone(),
                };
                runs.push(json!({
                    "codec": arm.route, "symbol": symbol, "code_object_sha": arm.co_sha, "timing_source": "device", "warmup_runs": a.warmup,
                    "samples_s": per_arm[ai], "routed_rows": routed_rows, "tile_epochs": tiles_total * kep,
                    "fabric": fabric.clone(), "device": device.clone(),
                    "label": label, "tiling": tiling.name(), "k_epochs_per_tile": kep, "tiles": tiles_total, "padded_rows": tiles_total * TILE_ROWS,
                    "launches_per_sample": launches.iter().filter(|l| *label == "pass" || l.proj.name() == *label).count(),
                    "layer": layer_no, "scope": scope_json.clone(), "summary": summary(&per_arm[ai]),
                    "tile_epochs_per_s_median": (tiles_total * kep) as f64 / meds[ai],
                    "weights": match arm.kind { ArmKind::W3 => "packed artifact W3 experts", ArmKind::Xts => "original XTS MQ4 slices of the same expert ids", ArmKind::E8p => "real solved E8P records of the same expert ids" },
                    "slab_bytes": arm.slab_bytes, "timer": "hipEventElapsedTime around the launch sequence, one sync per pass, default stream",
                }));
            }
            let mut per_variant = serde_json::Map::new();
            for (ai, arm) in arms
                .iter()
                .enumerate()
                .filter(|(_, x)| x.kind == ArmKind::E8p)
            {
                let (t_ci, thr_ci) = ratio_ci(&per_arm[ai], &per_arm[xts_i], BOOTSTRAP_RESAMPLES);
                let thr = meds[xts_i] / meds[ai];
                let mut v = json!({"e8p_over_xts_time": meds[ai] / meds[xts_i], "e8p_over_xts_throughput": thr,
                    "ci95": {"e8p_over_xts_time": t_ci, "e8p_over_xts_throughput": thr_ci, "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "prng": "splitmix64", "paired": false, "method": "gp_bench.py _ratio_ci"}},
                    "meets_threshold_point": thr >= THRESHOLD});
                if let Some(wi) = w3_i {
                    v["e8p_over_w3_throughput"] = json!(meds[wi] / meds[ai]);
                    v["e8p_over_w3_time"] = json!(meds[ai] / meds[wi]);
                }
                per_variant.insert(arm.route.clone(), v);
            }
            let mut block = json!({"median_s": arms.iter().zip(&meds).map(|(x, m)| (x.route.clone(), json!(m))).collect::<serde_json::Map<_, _>>(),
                "tile_epochs": tiles_total * kep, "tiles": tiles_total, "routed_rows": routed_rows, "e8p": Value::Object(per_variant),
                "basis": "median seconds, identical tiles/slots/x on every arm, one card/session; throughput = tile_epochs / median_s so throughput ratios are inverse time ratios"});
            if let Some(wi) = w3_i {
                block["w3_over_xts_throughput"] = json!(meds[xts_i] / meds[wi]);
            }
            ratios.insert((*label).to_owned(), block);
            let la = json!({"layer": layer_no, "nt": 1, "tile_rows": TILE_ROWS, "k_epochs_per_tile": kep, "arch": "gfx1201"});
            let dir = out_dir.join(format!("L{layer_no}")).join(tiling.name());
            let lab = write_json(&dir.join(format!("{label}.layer_artifact.json")), &la)?;
            let tim = write_json(
                &dir.join(format!("{label}.timing.json")),
                &json!({"runs": runs, "schema": SCHEMA, "label": label, "tiling": tiling.name(), "layer": layer_no, "scope": scope_json.clone(),
                "route_trace_file": layer_receipt["route_trace_file"].clone()}),
            )?;
            files.push(json!({"label": label, "layer_artifact": dir.join(format!("{label}.layer_artifact.json")).display().to_string(), "layer_artifact_sha256": sha(&lab),
                "timing": dir.join(format!("{label}.timing.json")).display().to_string(), "timing_sha256": sha(&tim), "k_epochs_per_tile": kep, "tile_epochs": tiles_total * kep}));
        }
        if *tiling == Tiling::Aggregate {
            verdict_src = Some(Value::Object(ratios.clone()));
        }
        results.push(json!({"tiling": tiling.name(), "launch_sets": nonempty.len(), "tiles": tiles_total, "routed_rows": routed_rows, "padded_rows": tiles_total * TILE_ROWS,
            "tile_sets": nonempty.iter().map(|s| json!({"chunk": s.chunk, "tokens": s.tokens, "tiles": s.ntiles(), "live_slots": s.live_slots})).collect::<Vec<_>>(),
            "tiles_sha256": sha(&nonempty.iter().flat_map(|s| u32s_le(&s.tile_expert.iter().map(|&x| x as u32).collect::<Vec<_>>())).collect::<Vec<u8>>()),
            "sorted_slots_sha256": sha(&nonempty.iter().flat_map(|s| u32s_le(&s.sorted.iter().map(|&x| x as u32).collect::<Vec<_>>())).collect::<Vec<u8>>()),
            "liveness": liveness, "ratios": Value::Object(ratios), "files": files}));
        for l in launches {
            hip.free(l.x).map_err(err)?;
        }
        for (t, s) in set_dev {
            hip.free(t).map_err(err)?;
            hip.free(s).map_err(err)?;
        }
        for y in ybuf {
            hip.free(y).map_err(err)?;
        }
    }
    for e in events {
        hip.event_destroy(e).map_err(err)?;
    }
    for arm in arms {
        let Arm {
            slab, ptrs: [x, y], ..
        } = arm;
        hip.free(slab).map_err(err)?;
        hip.free(x).map_err(err)?;
        hip.free(y).map_err(err)?;
    }

    // ---- verdict (aggregate tiling, pass label = gate_up + down per tile epoch) -------------------------------------
    let vs = verdict_src.ok_or("no aggregate tiling result")?;
    let mut verdicts = Vec::new();
    for v in &variants {
        let key = format!("e8p_{}", v.name);
        let pass = &vs["pass"]["e8p"][&key];
        let thr = pass["e8p_over_xts_throughput"]
            .as_f64()
            .ok_or("verdict: missing pass ratio")?;
        let lo = pass["ci95"]["e8p_over_xts_throughput"]["lo"]
            .as_f64()
            .unwrap_or(f64::NAN);
        verdicts.push(json!({
            "variant": v.name, "arm": key, "threshold": THRESHOLD,
            "basis": {"tiling": "aggregate", "label": "pass", "metric": "tile-epoch throughput of E8P over XTS MQ4 = xts_median_s / e8p_median_s (same tiles, same slots, same card/session)"},
            "e8p_over_xts_throughput": thr, "ci95_lo": lo, "verdict": if thr >= THRESHOLD { "PASS" } else { "FAIL" },
            "ci95_lo_meets_threshold": lo >= THRESHOLD,
            "per_kind": {"gate_up": vs["gate_up"]["e8p"][&key]["e8p_over_xts_throughput"].clone(), "down": vs["down"]["e8p"][&key]["e8p_over_xts_throughput"].clone()},
            "e8p_over_w3_throughput": pass["e8p_over_w3_throughput"].clone(),
            "scope": {"mode": a.scope, "full_layer": scope_json["full_layer"].clone(), "selected_experts": pool.len()},
            "next": if thr >= THRESHOLD { "gate met on this scope" } else { "below threshold: parent authorizes at most one bounded LDS alternative (--variant alt), requalify with --mode g1 --variant alt, then re-time all arms in the same session" },
        }));
    }
    let mut layer_json = layer_receipt;
    layer_json["results"] = Value::Array(results);
    let mut r = common(vec![layer_json], "timed");
    r["device"] = device_full;
    r["fabric"] = fabric.clone();
    r["verdicts"] = Value::Array(verdicts);
    r["verdict_overall"] = json!(if r["verdicts"]
        .as_array()
        .unwrap()
        .iter()
        .all(|v| v["verdict"] == "PASS")
    {
        "PASS"
    } else {
        "FAIL"
    });
    let b = write_json(&receipt_path, &r)?;
    println!(
        "{}",
        json!({"schema": SCHEMA, "status": "timed", "receipt": receipt_path.display().to_string(), "receipt_sha256": sha(&b), "device": r["device"], "verdicts": r["verdicts"]})
    );
    Ok(())
}
