// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G0 byte oracle of the Q8F16 (qt3) trunk-overlay lane row-twins
//! (`QsaIn([Q8_0; 4])` / `GdnIn([Q8_0; 4])` / `Projection(Q8_0)` proof rows of
//! `gpu_forward_lanes`: qsa_in, qsa_out, gdn_in, gdn_out and the head).
//!
//! ```text
//! lane_q8_rows_oracle <BASE.mq4-xts> <BASE.mq4-xts.trunk-overlay>
//! ```
//!
//! Every overlay tensor (every payload sha256-verified against the sidecar
//! manifest by `bind_trunk_overlay` + `read_verified`) is uploaded as
//! `DType::Q8_0` and grouped by the lane stage that consumes it:
//!
//! * a QSA layer: qsa_in `[indexer_qk, q, k, v]` (one shared x) + qsa_out `o`;
//! * a GDN layer: gdn_in `[qkv, z, a, b]` (one shared x) + gdn_out `out_proj`;
//! * the head: `lm_head` alone (M = 248320).
//!
//! A group must be complete (a partial layer is refused: the lane stage proof
//! registers whole groups). For every group, `R` in `1..=8` (`LANE_MAX_ROWS`)
//! and every activation set:
//!
//! * the lane route is the dispatch's own shared projection entry
//!   (`project_weights` over the four same-x in members, `project_weight` for
//!   the single out / head member) with `R` activation rows, exactly the call a
//!   once-over-all-rows lane stage makes;
//! * the singleton is the rows == 1 decode route (`project_weights` /
//!   `project_weight` with one row, i.e. `gemv_q8_0`) on each activation row
//!   alone, plus the raw `Gpu::gemv_q8_0` launch on the same row (a second,
//!   dispatch-independent reference);
//! * every output row of the lane route must equal the singleton byte for byte,
//!   in the given row order AND with the rows reversed (a row's bytes must not
//!   depend on its position or its neighbours).
//!
//! Each output lives in `64 guard floats | R * M floats | 64 guard floats`
//! memset to the `0xFF` NaN sentinel; a launch may write only its `R` rows. The
//! activations and the weights are downloaded after every group and must equal
//! what was uploaded. The lane route must run the multi-row hipcc kernels
//! (`q8_0_rows_exact_route != PerRowSingleton`); on any other architecture the
//! oracle refuses (exit 2): the proof rows are gfx1201.
//!
//! Control (information only, never gating): the pre-twin route
//! `gemm_q8_0_batched_f32_chunked` at `R` in `2..=8` is compared with the
//! singleton too (in groups), so a zero-mismatch result cannot come from an
//! oracle that is blind to a row-order change.
//!
//! Last result line (the card script greps it):
//! `lane_q8_rows_oracle: <C> cells, <W> bytes compared, 0 mismatching bytes`.
//! Exit 0 pass, 1 mismatch, 2 environment.

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{project_weight, project_weights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::trunk_overlay::{bind_trunk_overlay, TrunkOverlayFormat};
use rdna_compute::gemv::{q8_0_rows_exact_route, Q8RowsExactRoute};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::path::Path;

const GUARD: usize = 64;
const MAX_R: usize = 8;
const SEEDS: [u64; 3] = [
    0x71a8_0001_c0ff_ee01,
    0x0fed_cba9_8765_4321,
    0x5151_aaaa_5555_0f0f,
];
/// Activation scale classes (each seed x class is one activation set).
const AMPS: [f32; 4] = [1.0, 1.0 / 1024.0, 48.0, 0.0625];

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Roughly normal in [-6, 6): the sum of four uniforms, centred.
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 1.7320508
    }
}

/// One `[R, k]` activation set: gaussian with heavy-tailed outliers, exact
/// zeros and signed zeros, scaled by `amp`.
fn gen_x(rows: usize, k: usize, amp: f32, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        for _ in 0..k {
            let v = match rng.next() % 64 {
                0 => 0.0,
                1 => -0.0,
                2 => rng.gauss() * 40.0,
                3 => rng.gauss() * 0.001,
                _ => rng.gauss(),
            };
            x.push(v * amp);
        }
    }
    x
}

#[derive(Default)]
struct Stats {
    cells: u64,
    bytes: u64,
    mismatches: u64,
    control_cells: u64,
    control_bad_bytes: u64,
    control_bad_cells: u64,
}

/// One lane-stage group: `n_in` same-x in-projection members followed by one
/// out / head member.
struct Layer {
    index: usize,
    kind: &'static str,
    /// QSA: indexer_qk, q, k, v, o. GDN: qkv, z, a, b, out. Head: lm_head
    /// (`n_in == 0`).
    weights: Vec<(String, GpuTensor, usize, usize)>,
    n_in: usize,
    hosts: Vec<Vec<u8>>,
}

fn weight_ref<'a>(buf: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf,
        dtype: DType::Q8_0,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn alloc_guarded(gpu: &mut Gpu, elems: usize) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[2 * GUARD + elems], DType::F32)
        .map_err(err)?;
    gpu.hip.memset(&t.buf, 0xff, t.buf.size()).map_err(err)?;
    Ok(t)
}

fn reset_guarded(gpu: &mut Gpu, t: &GpuTensor) -> Result<(), String> {
    gpu.hip.memset(&t.buf, 0xff, t.buf.size()).map_err(err)
}

/// Compare a guarded output allocation with the reference rows (`rows * m`
/// words) and the untouched sentinel everywhere else.
fn check(
    gpu: &Gpu,
    y: &GpuTensor,
    rows: usize,
    m: usize,
    want: &[f32],
    tag: &str,
    st: &mut Stats,
) -> Result<(), String> {
    st.cells += 1;
    let got = gpu.download_f32(y).map_err(err)?;
    let mut bad = 0u64;
    for (i, g) in got.iter().enumerate() {
        let expect = if i < GUARD || i >= GUARD + rows * m {
            0xffff_ffffu32
        } else {
            want[i - GUARD].to_bits()
        };
        if g.to_bits() != expect {
            bad += (g.to_bits() ^ expect)
                .to_le_bytes()
                .iter()
                .filter(|b| **b != 0)
                .count() as u64;
            if bad <= 4 {
                eprintln!(
                    "MISMATCH {tag} word {i} (row {} col {}): got {:#010x} want {expect:#010x}",
                    i.wrapping_sub(GUARD) / m.max(1),
                    i.wrapping_sub(GUARD) % m.max(1),
                    g.to_bits()
                );
            }
        }
    }
    st.bytes += (got.len() * 4) as u64;
    st.mismatches += bad;
    Ok(())
}

/// The rows == 1 decode route on each activation row alone: the dispatch entry
/// with one row (`project_weights` / `project_weight`), and the raw
/// `gemv_q8_0` launch; the two must agree with each other too.
fn singleton_rows(
    gpu: &mut Gpu,
    x: &GpuTensor,
    ws: &[&WeightRef<'_>],
    rows: usize,
    st: &mut Stats,
    tag: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let mut out: Vec<Vec<f32>> = ws.iter().map(|_| Vec::new()).collect();
    for r in 0..rows {
        let k = ws[0].k;
        let xr = x.sub_offset(r * k, k);
        for (p, w) in ws.iter().enumerate() {
            let via_dispatch = gpu.zeros(&[w.m], DType::F32).map_err(err)?;
            project_weight(gpu, w, &xr, &via_dispatch, 1, None).map_err(err)?;
            let raw = gpu.zeros(&[w.m], DType::F32).map_err(err)?;
            gpu.gemv_q8_0(w.buf, &xr, &raw, w.m, w.k).map_err(err)?;
            let a = gpu.download_f32(&via_dispatch).map_err(err)?;
            let b = gpu.download_f32(&raw).map_err(err)?;
            st.cells += 1;
            st.bytes += (a.len() * 4) as u64;
            for (i, (u, v)) in a.iter().zip(&b).enumerate() {
                if u.to_bits() != v.to_bits() {
                    st.mismatches += 4;
                    eprintln!(
                        "MISMATCH {tag} row {r} member {p} col {i}: dispatch rows==1 vs gemv_q8_0"
                    );
                }
            }
            out[p].extend_from_slice(&b);
            gpu.free_tensor(via_dispatch).map_err(err)?;
            gpu.free_tensor(raw).map_err(err)?;
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_layer(gpu: &mut Gpu, layer: &Layer, seed: u64, st: &mut Stats) -> Result<(), String> {
    let n_in = layer.n_in;
    let k_out = layer.weights[n_in].3;
    // A head has no in group: x_in stays a (never launched) 1-wide placeholder.
    let k_in = if n_in > 0 { layer.weights[0].3 } else { k_out };
    for (ai, amp) in AMPS.iter().enumerate() {
        let mut rng = Rng(seed ^ ((layer.index as u64) << 32) ^ ((ai as u64) << 8) ^ 0x9e37_79b9);
        let x_in_host = gen_x(MAX_R, k_in, *amp, &mut rng);
        let x_out_host = gen_x(MAX_R, k_out, *amp, &mut rng);
        let x_in = gpu.upload_f32(&x_in_host, &[MAX_R, k_in]).map_err(err)?;
        let x_out = gpu.upload_f32(&x_out_host, &[MAX_R, k_out]).map_err(err)?;
        let refs: Vec<WeightRef<'_>> = layer
            .weights
            .iter()
            .map(|(_, t, m, k)| weight_ref(t, *m, *k))
            .collect();
        let in_ws: Vec<&WeightRef<'_>> = refs[..n_in].iter().collect();
        let out_ws: Vec<&WeightRef<'_>> = refs[n_in..].iter().collect();
        let tag0 = format!("{} {} seed {seed:#x} amp {amp}", layer.kind, layer.index);
        // Singleton reference rows, all MAX_R rows once; every R reuses a prefix
        // and the reversed arm indexes rows backwards.
        let ref_in = if n_in > 0 {
            singleton_rows(gpu, &x_in, &in_ws, MAX_R, st, &format!("{tag0} in"))?
        } else {
            Vec::new()
        };
        let ref_out = singleton_rows(gpu, &x_out, &out_ws, MAX_R, st, &format!("{tag0} out"))?;
        for rows in 1..=MAX_R {
            for reversed in [false, true] {
                let order: Vec<usize> = if reversed {
                    (0..rows).rev().collect()
                } else {
                    (0..rows).collect()
                };
                // Activations in this row order.
                let pick = |host: &[f32], k: usize| -> Vec<f32> {
                    order
                        .iter()
                        .flat_map(|&r| host[r * k..(r + 1) * k].to_vec())
                        .collect()
                };
                let xi = gpu
                    .upload_f32(&pick(&x_in_host, k_in), &[rows, k_in])
                    .map_err(err)?;
                let xo = gpu
                    .upload_f32(&pick(&x_out_host, k_out), &[rows, k_out])
                    .map_err(err)?;
                // in: the four same-x members in one shared call (none for a head).
                let ys: Vec<GpuTensor> = in_ws
                    .iter()
                    .map(|w| alloc_guarded(gpu, rows * w.m))
                    .collect::<Result<_, _>>()?;
                let views: Vec<GpuTensor> = ys
                    .iter()
                    .zip(&in_ws)
                    .map(|(y, w)| y.sub_offset(GUARD, rows * w.m))
                    .collect();
                let projections: Vec<(&WeightRef<'_>, &GpuTensor)> =
                    in_ws.iter().copied().zip(views.iter()).collect();
                if n_in > 0 {
                    project_weights(gpu, &xi, rows, None, &projections).map_err(err)?;
                }
                for (p, w) in in_ws.iter().enumerate() {
                    let want: Vec<f32> = order
                        .iter()
                        .flat_map(|&r| ref_in[p][r * w.m..(r + 1) * w.m].to_vec())
                        .collect();
                    let tag = format!("{tag0} in member {p} rows {rows} rev {reversed}");
                    check(gpu, &ys[p], rows, w.m, &want, &tag, st)?;
                }
                // out: the single projection (the head too).
                let wo = out_ws[0];
                let yo = alloc_guarded(gpu, rows * wo.m)?;
                let vo = yo.sub_offset(GUARD, rows * wo.m);
                project_weight(gpu, wo, &xo, &vo, rows, None).map_err(err)?;
                let want: Vec<f32> = order
                    .iter()
                    .flat_map(|&r| ref_out[0][r * wo.m..(r + 1) * wo.m].to_vec())
                    .collect();
                let tag = format!("{tag0} out rows {rows} rev {reversed}");
                check(gpu, &yo, rows, wo.m, &want, &tag, st)?;

                // Control: the pre-twin batched route (information only).
                if rows >= 2 && !reversed && *amp == 1.0 {
                    for (p, w) in in_ws.iter().enumerate() {
                        reset_guarded(gpu, &ys[p])?;
                        gpu.gemm_q8_0_batched_f32_chunked(w.buf, &xi, &views[p], w.m, w.k, rows)
                            .map_err(err)?;
                        let got = gpu.download_f32(&ys[p]).map_err(err)?;
                        let mut bad = 0u64;
                        for r in 0..rows {
                            for c in 0..w.m {
                                let g = got[GUARD + r * w.m + c];
                                let s = ref_in[p][r * w.m + c];
                                bad += (g.to_bits() ^ s.to_bits())
                                    .to_le_bytes()
                                    .iter()
                                    .filter(|b| **b != 0)
                                    .count() as u64;
                            }
                        }
                        st.control_cells += 1;
                        st.control_bad_bytes += bad;
                        st.control_bad_cells += u64::from(bad != 0);
                    }
                }
                for y in ys {
                    gpu.free_tensor(y).map_err(err)?;
                }
                gpu.free_tensor(yo).map_err(err)?;
                gpu.free_tensor(xi).map_err(err)?;
                gpu.free_tensor(xo).map_err(err)?;
            }
        }
        // Inputs untouched.
        for (t, host) in [(&x_in, &x_in_host), (&x_out, &x_out_host)] {
            let d = gpu.download_f32(t).map_err(err)?;
            st.cells += 1;
            st.bytes += (d.len() * 4) as u64;
            if d.iter().zip(host).any(|(a, b)| a.to_bits() != b.to_bits()) {
                eprintln!("MISMATCH activations modified ({tag0})");
                st.mismatches += 1;
            }
        }
        gpu.free_tensor(x_in).map_err(err)?;
        gpu.free_tensor(x_out).map_err(err)?;
    }
    // Weights untouched.
    for ((name, t, _, _), host) in layer.weights.iter().zip(&layer.hosts) {
        let mut d = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut d, &t.buf).map_err(err)?;
        st.cells += 1;
        st.bytes += host.len() as u64;
        if d[..host.len()] != host[..] {
            eprintln!("MISMATCH weights modified: {name}");
            st.mismatches += 1;
        }
    }
    Ok(())
}

/// `(kind, layer, slot)` of an overlay tensor in its lane-stage group.
/// QSA slots: indexer_qk, q, k, v (in), o (out). GDN slots: qkv, z, a, b (in),
/// out_proj (out). The head is `("head", 0, 0)`.
fn parse_member(name: &str) -> Option<(&'static str, usize, usize)> {
    const QSA: [&str; 5] = [
        "self_attn.indexer.index_qk_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ];
    const GDN: [&str; 5] = [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.out_proj.weight",
    ];
    if name == "lm_head.weight" {
        return Some(("head", 0, 0));
    }
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (index, role) = rest.split_once('.')?;
    let index = index.parse().ok()?;
    if let Some(slot) = QSA.iter().position(|r| *r == role) {
        return Some(("qsa", index, slot));
    }
    GDN.iter()
        .position(|r| *r == role)
        .map(|slot| ("gdn", index, slot))
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: lane_q8_rows_oracle <BASE> <TRUNK-OVERLAY>".to_string());
    }
    let base = HfqFile::open(Path::new(&args[1])).map_err(|e| format!("base: {e}"))?;
    let bound = bind_trunk_overlay(&base, Path::new(&args[2]))?;
    let payloads = bound.read_verified()?;
    println!(
        "overlay: {} tensors, {} payload bytes, marker {} (every payload sha256-verified)",
        payloads.len(),
        bound.payload_bytes(),
        bound.marker()
    );
    let mut gpu = Gpu::init().map_err(err)?;
    println!("device arch: {}", gpu.arch);
    // (kind, layer) -> member slots.
    let mut groups: BTreeMap<(&'static str, usize), Vec<Option<usize>>> = BTreeMap::new();
    for (i, p) in payloads.iter().enumerate() {
        if p.format != TrunkOverlayFormat::Qt3 {
            return Err(format!("tensor {} is not a qt3 (Q8F16) payload", p.name));
        }
        let (kind, layer, slot) = parse_member(&p.name)
            .ok_or_else(|| format!("tensor {} is not a trunk lane-stage projection", p.name))?;
        let width = if kind == "head" { 1 } else { 5 };
        let slots = groups
            .entry((kind, layer))
            .or_insert_with(|| vec![None; width]);
        if slots[slot].replace(i).is_some() {
            return Err(format!("duplicate overlay tensor {}", p.name));
        }
    }
    if groups.is_empty() || groups.values().any(|s| s.iter().any(Option::is_none)) {
        return Err(format!(
            "an overlay lane-stage group is incomplete ({} tensors over {} groups)",
            payloads.len(),
            groups.len()
        ));
    }
    for kind in ["qsa", "gdn", "head"] {
        println!(
            "groups {kind}: {}",
            groups.keys().filter(|(k, _)| *k == kind).count()
        );
    }
    let mut ks: Vec<usize> = payloads.iter().map(|p| p.shape[1]).collect();
    ks.sort_unstable();
    ks.dedup();
    for &k in &ks {
        let route = q8_0_rows_exact_route(&gpu.arch, k);
        println!("K={k}: rows-exact route {route:?}");
        if route == Q8RowsExactRoute::PerRowSingleton {
            eprintln!(
                "REFUSAL: arch {} runs the per-row singleton for K={k}; this oracle proves the multi-row kernels (gfx1201)",
                gpu.arch
            );
            std::process::exit(2);
        }
    }
    if gpu.arch != "gfx1201" {
        eprintln!(
            "REFUSAL: arch {} is not gfx1201 (the registered proof rows)",
            gpu.arch
        );
        std::process::exit(2);
    }
    println!(
        "few_row_limit {} (lane groups carry at most {MAX_R} rows)",
        gpu.few_row_limit()
    );
    if gpu.few_row_limit() < MAX_R {
        return Err(format!("few_row_limit {} < {MAX_R}", gpu.few_row_limit()));
    }

    let mut st = Stats::default();
    for (&(kind, index), slots) in &groups {
        let mut layer = Layer {
            index,
            kind,
            weights: Vec::new(),
            n_in: if kind == "head" { 0 } else { 4 },
            hosts: Vec::new(),
        };
        for slot in slots.iter().flatten() {
            let p = &payloads[*slot];
            let mut t = gpu.upload_raw(&p.bytes, &[p.bytes.len()]).map_err(err)?;
            t.dtype = DType::Q8_0;
            layer
                .weights
                .push((p.name.clone(), t, p.shape[0], p.shape[1]));
            layer.hosts.push(p.bytes.clone());
        }
        // The in members share x: one K.
        let n_in = layer.n_in;
        if n_in > 0 {
            let k0 = layer.weights[0].3;
            if layer.weights[..n_in].iter().any(|w| w.3 != k0) {
                return Err(format!("{kind} {index}: in members differ in K"));
            }
        }
        for seed in SEEDS {
            run_layer(&mut gpu, &layer, seed, &mut st)?;
        }
        println!(
            "{kind} {index}: ms {:?} k_out {} ok ({} cells so far, {} mismatching bytes)",
            layer.weights.iter().map(|w| w.2).collect::<Vec<_>>(),
            layer.weights[n_in].3,
            st.cells,
            st.mismatches
        );
        for (_, t, _, _) in layer.weights {
            gpu.free_tensor(t).map_err(err)?;
        }
    }
    println!(
        "control (information only): the pre-twin gemm_q8_0_batched_f32_chunked differs from the singleton in {}/{} in-group cells ({} bytes)",
        st.control_bad_cells, st.control_cells, st.control_bad_bytes
    );
    println!(
        "lane_q8_rows_oracle: {} cells, {} bytes compared, {} mismatching bytes",
        st.cells, st.bytes, st.mismatches
    );
    Ok(st.mismatches == 0)
}

fn main() {
    match run() {
        Ok(true) => println!("PASS"),
        Ok(false) => {
            println!("FAIL");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("lane_q8_rows_oracle: {e}");
            std::process::exit(2);
        }
    }
}
