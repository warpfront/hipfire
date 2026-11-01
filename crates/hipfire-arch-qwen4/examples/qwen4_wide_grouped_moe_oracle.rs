// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_wide_grouped_moe_oracle — lever-A step A4g: byte oracle and read census of
//! the routed-MoE chain of the sealed QT44/QT53 route at 9..=32 verify rows, on REAL
//! artifact expert tensors (single card, no EP gather).
//!
//! The chain mirrors `hipfire-dispatch::pipeline::qt44_qt53_prefill` stage for stage on
//! the same `rdna-compute` launchers (gate/up, activation, down; the combine is not
//! touched by `HIPFIRE_QWEN4_WIDE_GROUPED_MOE`):
//!   * `indexed-wide` (today's wide_exact route, wide scope open with `N` rows):
//!     `gemv_mq4g256v2_moe_gate_up_top10_indexed_batched` ->
//!     `silu_mul_bf16_rt_rotate_x_mq_128_v2_shared` ->
//!     `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded`;
//!   * `grouped-wide` (the flag's route, wide scope open): `moe_scatter_fused_top10` ->
//!     `gemm_mq4g256v2_moe_grouped_top10` (r8 tile, `x_src_rows > 8`) ->
//!     `moe_gate_up_unscatter_silu_rotate128_top10` -> the same per-slot down;
//!   * `groups8`: the grouped chain over groups of at most eight rows, scope closed
//!     (the r2 few-row tile; what the 4x8-row verify runs today);
//!   * `single`: the grouped chain one row at a time (the AR decode route).
//! Compared, per layer and row set: the activation rows (`rot`, `[N*10, 640]`) and the
//! expanded down outputs (`[N*10, 2560]`), all four pairwise against `grouped-wide`
//! plus `indexed-wide` vs `groups8`; every byte, guards on the outputs.
//!
//! Router: the real `mlp.gate.weight` (BF16) applied on the CPU to the synthetic
//! activation pool rows; top-10 by logit. Activations are synthetic (not hidden states),
//! so distinct-expert counts are higher than K0's measured 144-151 per layer at 30-32
//! rows; the census prints both. Assumes `recipe.bf16_round_trip()` (Flash-Next).
//!
//! Read census per layer-call, from the router's slot counts `c_e`:
//!   gate/up expert reads: indexed-wide `N*10`; grouped-wide `sum_e ceil(c_e/8)`;
//!   groups8 (r2 tiles) `sum_groups sum_e ceil(c_{g,e}/2)`; distinct `|{e: c_e>0}|`.
//!   down (all routes, per-slot indexed): `N*10`.
//! Bytes per read: gate/up 1,740,800, down 870,400.
//!
//! Exit: 0 all pairs byte-equal; 1 any mismatch or launch error.

use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

type Res<T> = Result<T, String>;

const GUARD: usize = 64;
const NAN_BITS: u32 = 0x7fc0_0000;
const POOL: usize = 64;
const MIN_N: usize = 9;
const MAX_N: usize = 32;
const HIDDEN: usize = 2560;
const MI: usize = 640;
const K_TOP: usize = 10;
const N_EXP: usize = 512;
const BLOCK_M: usize = 16;
const GU_BYTES: usize = 1_740_800; // 1280 * (2560/256) * 136
const DN_BYTES: usize = 870_400; // 2560 * (640/128) * 68
const DEFAULT_SEED: u64 = 0x4134_6772_6f75_7001;
const AMPS: [f32; 12] = [
    1.0,
    1.0 / 1024.0,
    1.0 / 16.0,
    4.0,
    64.0,
    0.5,
    0.0,
    8.0,
    1.0 / 256.0,
    2.0,
    32.0,
    0.25,
];

fn he<T, E: std::fmt::Debug>(r: Result<T, E>) -> Res<T> {
    r.map_err(|e| format!("{e:?}"))
}

// ------------------------------------------------------------------- CPU side

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn pool_row(seed: u64, idx: usize, k: usize) -> Vec<f32> {
    let mut rng = Rng(seed ^ (idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((k as u64) << 32) | 1);
    let amp = AMPS[idx % AMPS.len()];
    (0..k)
        .map(|c| match (idx + c) % 97 {
            0 => 0.0,
            1 => -0.0,
            _ => rng.unit() * amp,
        })
        .collect()
}

fn pick_rows(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..POOL).collect();
    for i in 0..n {
        let j = i + rng.below(POOL - i);
        idx.swap(i, j);
    }
    idx.truncate(n);
    idx
}

fn row_set_sizes(rng: &mut Rng, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            if i < MAX_N - MIN_N + 1 {
                MIN_N + i
            } else {
                MIN_N + rng.below(MAX_N - MIN_N + 1)
            }
        })
        .collect()
}

fn groups8(rows: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(8)
        .map(|s| (s, 8.min(rows - s)))
        .collect()
}

fn bf16_to_f32(b: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(b) as u32) << 16)
}

/// Top-10 expert ids of every pool row under the real router weight (`[512, 2560]` BF16).
fn route_pool(router: &[u8], pool: &[Vec<f32>]) -> Vec<[i32; K_TOP]> {
    let w: Vec<f32> = router
        .chunks_exact(2)
        .map(|c| bf16_to_f32([c[0], c[1]]))
        .collect();
    pool.iter()
        .map(|x| {
            let mut logits: Vec<(f32, usize)> = (0..N_EXP)
                .map(|e| {
                    let row = &w[e * HIDDEN..(e + 1) * HIDDEN];
                    (row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>(), e)
                })
                .collect();
            logits.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            std::array::from_fn(|i| logits[i].1 as i32)
        })
        .collect()
}

/// Per-expert slot counts of the given rows' routes.
fn slot_counts(routes: &[[i32; K_TOP]]) -> Vec<usize> {
    let mut c = vec![0usize; N_EXP];
    for r in routes {
        for &e in r {
            c[e as usize] += 1;
        }
    }
    c
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
struct Census {
    distinct: usize,
    indexed: usize,
    grouped_wide_r8: usize,
    groups8_r2: usize,
}

fn read_census(routes: &[[i32; K_TOP]]) -> Census {
    let c = slot_counts(routes);
    let mut g8 = 0;
    for (first, n) in groups8(routes.len()) {
        g8 += slot_counts(&routes[first..first + n])
            .iter()
            .map(|&x| x.div_ceil(2))
            .sum::<usize>();
    }
    Census {
        distinct: c.iter().filter(|&&x| x > 0).count(),
        indexed: routes.len() * K_TOP,
        grouped_wide_r8: c.iter().map(|&x| x.div_ceil(8)).sum(),
        groups8_r2: g8,
    }
}

/// `sum_e align_up(count_e, 16)` worst-case bound used by the pipeline (`m_total_bound`).
fn grouped_rows_bound(total_slots: usize) -> usize {
    let live = total_slots.min(N_EXP);
    (total_slots + live * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
struct Diff {
    bytes: usize,
    rows: usize,
    first: Option<(usize, usize, u8, u8)>,
}

fn diff_rows(got: &[u32], want: &[u32], width: usize) -> Diff {
    let mut d = Diff::default();
    let rows = got.len().min(want.len()) / width.max(1);
    for r in 0..rows {
        let mut bad = false;
        for c in 0..width {
            let (a, b) = (got[r * width + c], want[r * width + c]);
            if a != b {
                let (ab, bb) = (a.to_le_bytes(), b.to_le_bytes());
                for i in 0..4 {
                    if ab[i] != bb[i] {
                        d.bytes += 1;
                        bad = true;
                        if d.first.is_none() {
                            d.first = Some((r, c * 4 + i, ab[i], bb[i]));
                        }
                    }
                }
            }
        }
        d.rows += bad as usize;
    }
    if got.len() != want.len() {
        d.bytes += got.len().abs_diff(want.len()) * 4;
    }
    d
}

// ----------------------------------------------------------------- GPU helpers

fn nan_tensor(gpu: &mut Gpu, n: usize) -> Res<GpuTensor> {
    he(gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n]))
}

fn zero_raw(gpu: &mut Gpu, bytes: usize) -> Res<GpuTensor> {
    he(gpu.upload_raw(&vec![0u8; bytes], &[bytes]))
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Res<Vec<u32>> {
    he(gpu.hip.device_synchronize())?;
    Ok(he(gpu.download_f32(t))?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

type Symbols = Arc<Mutex<Vec<String>>>;

fn take_symbols(s: &Symbols) -> BTreeSet<String> {
    std::mem::take(&mut *s.lock().unwrap_or_else(|e| e.into_inner()))
        .into_iter()
        .collect()
}

fn ptr_table(gpu: &mut Gpu, base: &GpuTensor, per: usize) -> Res<GpuTensor> {
    let b = base.buf.as_ptr() as usize as u64;
    let bytes: Vec<u8> = (0..N_EXP)
        .flat_map(|i| (b + (i * per) as u64).to_le_bytes())
        .collect();
    he(gpu.upload_raw(&bytes, &[bytes.len()]))
}

struct LayerW {
    gu: GpuTensor,
    dn: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
}

fn load_layer(gpu: &mut Gpu, hfq: &HfqFile, layer: usize) -> Res<(LayerW, Vec<u8>)> {
    let p = format!("model.language_model.layers.{layer}.mlp");
    let get = |name: String| -> Res<Vec<u8>> {
        hfq.tensor_data(&name)
            .map(|(_, d)| d.to_vec())
            .ok_or_else(|| format!("tensor {name} missing"))
    };
    let gu_b = get(format!("{p}.experts.gate_up_proj"))?;
    let dn_b = get(format!("{p}.experts.down_proj"))?;
    if gu_b.len() != N_EXP * GU_BYTES || dn_b.len() != N_EXP * DN_BYTES {
        return Err(format!(
            "layer {layer}: expert bytes gate_up {} (want {}), down {} (want {})",
            gu_b.len(),
            N_EXP * GU_BYTES,
            dn_b.len(),
            N_EXP * DN_BYTES
        ));
    }
    let router = get(format!("{p}.gate.weight"))?;
    let gu = he(gpu.upload_raw(&gu_b, &[gu_b.len()]))?;
    let dn = he(gpu.upload_raw(&dn_b, &[dn_b.len()]))?;
    let gu_ptrs = ptr_table(gpu, &gu, GU_BYTES)?;
    let dn_ptrs = ptr_table(gpu, &dn, DN_BYTES)?;
    Ok((
        LayerW {
            gu,
            dn,
            gu_ptrs,
            dn_ptrs,
        },
        router,
    ))
}

fn down_call(gpu: &mut Gpu, w: &LayerW, topk: &GpuTensor, rot: &GpuTensor, out: &GpuTensor, n: usize) -> Res<()> {
    he(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &w.dn_ptrs, topk, rot, out, HIDDEN, MI, n, N_EXP, K_TOP,
    ))
}

/// Today's wide_exact route.
fn indexed_chain(gpu: &mut Gpu, w: &LayerW, n: usize, x: &GpuTensor, topk: &GpuTensor, rot: &GpuTensor, out: &GpuTensor) -> Res<()> {
    let total = n * K_TOP;
    let gate = nan_tensor(gpu, total * MI)?;
    let up = nan_tensor(gpu, total * MI)?;
    he(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
        &w.gu_ptrs, topk, x, &gate, &up, 2 * MI, HIDDEN, n,
    ))?;
    he(gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(&gate, &up, rot, MI, total, None))?;
    down_call(gpu, w, topk, rot, out, n)?;
    he(gpu.free_tensor(gate))?;
    he(gpu.free_tensor(up))
}

/// The grouped (expert-unioned) route of the <=8-row call; the flag's route at 9..32.
fn grouped_chain(gpu: &mut Gpu, w: &LayerW, n: usize, x: &GpuTensor, topk: &GpuTensor, rot: &GpuTensor, out: &GpuTensor) -> Res<()> {
    let total = n * K_TOP;
    let gr = grouped_rows_bound(total);
    let counts = zero_raw(gpu, N_EXP * 4)?;
    let offsets = zero_raw(gpu, (N_EXP + 1) * 4)?;
    let sorted = zero_raw(gpu, gr * 4)?;
    let tiles = zero_raw(gpu, (gr / BLOCK_M) * 4)?;
    let inverse = zero_raw(gpu, total * 4)?;
    let y = he(gpu.upload_f32(&vec![0f32; gr * 2 * MI], &[gr * 2 * MI]))?;
    he(gpu.moe_scatter_fused_top10(topk, &counts, &offsets, &sorted, &tiles, &inverse, total, N_EXP, gr, BLOCK_M))?;
    he(gpu.gemm_mq4g256v2_moe_grouped_top10(
        &w.gu_ptrs, &tiles, &sorted, x, &y, 2 * MI, HIDDEN, K_TOP, gr, n,
    ))?;
    he(gpu.moe_gate_up_unscatter_silu_rotate128_top10(&y, &sorted, rot, MI, gr, true, None))?;
    down_call(gpu, w, topk, rot, out, n)?;
    for t in [counts, offsets, sorted, tiles, inverse, y] {
        he(gpu.free_tensor(t))?;
    }
    Ok(())
}

type Chain = fn(&mut Gpu, &LayerW, usize, &GpuTensor, &GpuTensor, &GpuTensor, &GpuTensor) -> Res<()>;

struct Outs {
    rot: GpuTensor,
    out: GpuTensor,
}

fn alloc_outs(gpu: &mut Gpu, n: usize) -> Res<Outs> {
    Ok(Outs {
        rot: nan_tensor(gpu, GUARD + n * K_TOP * MI + GUARD)?,
        out: nan_tensor(gpu, GUARD + n * K_TOP * HIDDEN + GUARD)?,
    })
}

/// Run `chain` over `groups` of rows into `o`, with the wide scope `scope` (0 = closed).
#[allow(clippy::too_many_arguments)]
fn run_groups(
    gpu: &mut Gpu,
    w: &LayerW,
    chain: Chain,
    groups: &[(usize, usize)],
    scope: usize,
    x: &GpuTensor,
    topk: &GpuTensor,
    o: &Outs,
) -> Res<()> {
    for &(first, cnt) in groups {
        let xv = x.sub_offset(first * HIDDEN, cnt * HIDDEN);
        let tv = topk.sub_offset(first * K_TOP * 4, cnt * K_TOP * 4);
        let rv = o.rot.sub_offset(GUARD + first * K_TOP * MI, cnt * K_TOP * MI);
        let ov = o.out.sub_offset(GUARD + first * K_TOP * HIDDEN, cnt * K_TOP * HIDDEN);
        gpu.wide_verify_rows = scope;
        let r = chain(gpu, w, cnt, &xv, &tv, &rv, &ov);
        gpu.wide_verify_rows = 0;
        r?;
    }
    Ok(())
}

struct Pair {
    rot: Diff,
    out: Diff,
    guard_bad: usize,
}

fn guards_clean(bits: &[u32], at: usize, len: usize) -> bool {
    bits[..at].iter().all(|&w| w == NAN_BITS) && bits[at + len..].iter().all(|&w| w == NAN_BITS)
}

struct Fetched {
    rot: Vec<u32>,
    out: Vec<u32>,
    guard_bad: usize,
}

fn fetch_outs(gpu: &Gpu, o: &Outs, n: usize) -> Res<Fetched> {
    let (r, d) = (fetch(gpu, &o.rot)?, fetch(gpu, &o.out)?);
    let mut guard_bad = 0;
    if !guards_clean(&r, GUARD, n * K_TOP * MI) {
        guard_bad += 1;
    }
    if !guards_clean(&d, GUARD, n * K_TOP * HIDDEN) {
        guard_bad += 1;
    }
    Ok(Fetched {
        rot: r[GUARD..GUARD + n * K_TOP * MI].to_vec(),
        out: d[GUARD..GUARD + n * K_TOP * HIDDEN].to_vec(),
        guard_bad,
    })
}

fn compare(a: &Fetched, b: &Fetched) -> Pair {
    Pair {
        rot: diff_rows(&a.rot, &b.rot, K_TOP * MI),
        out: diff_rows(&a.out, &b.out, K_TOP * HIDDEN),
        guard_bad: a.guard_bad + b.guard_bad,
    }
}

#[derive(Default)]
struct Stat {
    calls: usize,
    /// per pair name -> (rot bytes, out bytes, rot rows, out rows, guard)
    pairs: BTreeMap<&'static str, [usize; 5]>,
    first: BTreeMap<&'static str, String>,
    census: Census,
    errors: usize,
}

struct Args(Vec<String>);

impl Args {
    fn value(&self, flag: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == flag)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn flag(&self, flag: &str) -> bool {
        self.0.iter().any(|a| a == flag)
    }
}

fn run() -> Res<i32> {
    let args = Args(std::env::args().skip(1).collect());
    let model = args.value("--model").ok_or("--model required")?;
    let layers: Vec<usize> = args
        .value("--layers")
        .unwrap_or("0,20,47")
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|e| format!("--layers: {e}")))
        .collect::<Res<_>>()?;
    let row_sets: usize = args
        .value("--row-sets")
        .unwrap_or("100")
        .parse()
        .map_err(|e| format!("--row-sets: {e}"))?;
    let seed: u64 = match args.value("--seed") {
        Some(s) => s.parse().map_err(|e| format!("--seed: {e}"))?,
        None => DEFAULT_SEED,
    };
    let verbose = args.flag("--verbose");
    let hfq = HfqFile::open(Path::new(model)).map_err(|e| format!("open {model}: {e}"))?;
    let mut gpu = he(Gpu::init())?;
    gpu.qwen4_scope = true;
    gpu.lane_rows_scope = true;
    let syms: Symbols = Arc::new(Mutex::new(Vec::new()));
    {
        let s = syms.clone();
        gpu.set_lab_launch_observer(Some(Arc::new(
            move |_g: &Gpu, w: &rdna_compute::LabLaunchWitness<'_>| -> HipResult<()> {
                s.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(w.symbol.to_string());
                Ok(())
            },
        )));
    }
    println!(
        "arch {} few_row_limit(default)={} wide_grouped_moe(env)={} layers={layers:?} row_sets={row_sets} seed={seed}",
        gpu.arch,
        gpu.few_row_limit(),
        gpu.wide_grouped_moe()
    );
    let pool: Vec<Vec<f32>> = (0..POOL).map(|i| pool_row(seed, i, HIDDEN)).collect();
    let mut rng = Rng(seed | 1);
    let sizes = row_set_sizes(&mut rng, row_sets);
    let picks: Vec<Vec<usize>> = sizes.iter().map(|&n| pick_rows(&mut rng, n)).collect();

    let mut stats: BTreeMap<(usize, usize), Stat> = BTreeMap::new();
    let mut symbols: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
    let mut distinct_by_n: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &layer in &layers {
        let (w, router) = load_layer(&mut gpu, &hfq, layer)?;
        let routes_pool = route_pool(&router, &pool);
        println!("layer {layer}: experts uploaded, router applied to {} pool rows", pool.len());
        for (set_id, (&n, pick)) in sizes.iter().zip(&picks).enumerate() {
            let host_x: Vec<f32> = pick.iter().flat_map(|&i| pool[i].iter().copied()).collect();
            let routes: Vec<[i32; K_TOP]> = pick.iter().map(|&i| routes_pool[i]).collect();
            let topk_bytes: Vec<u8> = routes.iter().flatten().flat_map(|e| e.to_le_bytes()).collect();
            let x = he(gpu.upload_f32(&host_x, &[host_x.len()]))?;
            let topk = he(gpu.upload_raw(&topk_bytes, &[topk_bytes.len()]))?;
            let st = stats.entry((layer, n)).or_default();
            st.calls += 1;
            let c = read_census(&routes);
            st.census.distinct += c.distinct;
            st.census.indexed += c.indexed;
            st.census.grouped_wide_r8 += c.grouped_wide_r8;
            st.census.groups8_r2 += c.groups8_r2;
            distinct_by_n.entry(n).or_default().push(c.distinct);

            let all = [(0usize, n)];
            let ones: Vec<(usize, usize)> = (0..n).map(|r| (r, 1)).collect();
            let routes_to_run: [(&'static str, Chain, Vec<(usize, usize)>, usize); 4] = [
                ("indexed_wide", indexed_chain, all.to_vec(), n),
                ("grouped_wide", grouped_chain, all.to_vec(), n),
                ("groups8", grouped_chain, groups8(n), 0),
                ("single", grouped_chain, ones, 0),
            ];
            let mut got: BTreeMap<&'static str, Fetched> = BTreeMap::new();
            let mut failed = false;
            for (name, chain, groups, scope) in routes_to_run {
                let o = alloc_outs(&mut gpu, n)?;
                let _ = take_symbols(&syms);
                let r = run_groups(&mut gpu, &w, chain, &groups, scope, &x, &topk, &o);
                symbols.entry(name).or_default().extend(take_symbols(&syms));
                match r {
                    Ok(()) => {
                        got.insert(name, fetch_outs(&gpu, &o, n)?);
                    }
                    Err(e) => {
                        st.errors += 1;
                        st.first
                            .entry(name)
                            .or_insert_with(|| format!("set {set_id} n={n}: launch error: {e}"));
                        eprintln!("ERROR layer {layer} set {set_id} n={n} route {name}: {e}");
                        failed = true;
                    }
                }
                he(gpu.free_tensor(o.rot))?;
                he(gpu.free_tensor(o.out))?;
            }
            if !failed {
                let pairs: [(&'static str, &str, &str); 4] = [
                    ("grouped_wide_vs_indexed_wide", "grouped_wide", "indexed_wide"),
                    ("grouped_wide_vs_groups8", "grouped_wide", "groups8"),
                    ("grouped_wide_vs_single", "grouped_wide", "single"),
                    ("indexed_wide_vs_groups8", "indexed_wide", "groups8"),
                ];
                for (pname, a, b) in pairs {
                    let d = compare(&got[a], &got[b]);
                    let e = st.pairs.entry(pname).or_default();
                    e[0] += d.rot.bytes;
                    e[1] += d.out.bytes;
                    e[2] += d.rot.rows;
                    e[3] += d.out.rows;
                    e[4] += d.guard_bad;
                    for (what, df) in [("rot", &d.rot), ("down", &d.out)] {
                        if let Some((r, byte, g, wv)) = df.first {
                            st.first.entry(pname).or_insert_with(|| {
                                format!(
                                    "set {set_id} n={n} {what}: first differing row {r} byte {byte} got {g:#04x} want {wv:#04x}; {} bytes / {} rows",
                                    df.bytes, df.rows
                                )
                            });
                        }
                    }
                    if verbose {
                        println!(
                            "layer {layer} set {set_id} n={n} {pname}: rot {} down {} bytes",
                            d.rot.bytes, d.out.bytes
                        );
                    }
                }
            }
            he(gpu.free_tensor(x))?;
            he(gpu.free_tensor(topk))?;
        }
        for t in [w.gu, w.dn, w.gu_ptrs, w.dn_ptrs] {
            he(gpu.free_tensor(t))?;
        }
    }

    // Receipt.
    let mut total_bad = 0usize;
    let mut total_err = 0usize;
    let mut by_pair: BTreeMap<&'static str, [usize; 5]> = BTreeMap::new();
    let mut per: BTreeMap<String, Value> = BTreeMap::new();
    for ((layer, n), st) in &stats {
        total_err += st.errors;
        for (p, v) in &st.pairs {
            let e = by_pair.entry(p).or_default();
            for i in 0..5 {
                e[i] += v[i];
            }
            total_bad += v[0] + v[1] + v[4];
        }
        let calls = st.calls.max(1) as f64;
        let mb = |reads: usize| reads as f64 / calls * GU_BYTES as f64 / 1e6;
        per.entry(format!("L{layer}"))
            .or_insert_with(|| json!({}))[n.to_string()] = json!({
            "calls": st.calls,
            "pairs": st.pairs.iter().map(|(k, v)| (k.to_string(), json!({
                "mismatch_bytes_rot": v[0], "mismatch_bytes_down": v[1],
                "mismatch_rows_rot": v[2], "mismatch_rows_down": v[3], "guard_violations": v[4]}))).collect::<BTreeMap<_,_>>(),
            "first_difference": st.first,
            "launch_errors": st.errors,
            "read_census_per_call_mean": {
                "distinct_experts": st.census.distinct as f64 / calls,
                "gate_up_reads_indexed_wide": st.census.indexed as f64 / calls,
                "gate_up_reads_grouped_wide_r8": st.census.grouped_wide_r8 as f64 / calls,
                "gate_up_reads_groups8_r2": st.census.groups8_r2 as f64 / calls,
                "down_reads_all_routes": st.census.indexed as f64 / calls,
                "gate_up_MB_indexed_wide": mb(st.census.indexed),
                "gate_up_MB_grouped_wide_r8": mb(st.census.grouped_wide_r8),
                "gate_up_MB_groups8_r2": mb(st.census.groups8_r2),
                "down_MB_all_routes": st.census.indexed as f64 / calls * DN_BYTES as f64 / 1e6,
            },
        });
        if total_err > 0 && st.errors > 0 {
            println!("ERRORS layer {layer} n={n}: {}", st.errors);
        }
    }
    let distinct_summary: BTreeMap<String, Value> = distinct_by_n
        .iter()
        .map(|(n, v)| {
            let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
            (n.to_string(), json!({"mean": mean, "min": v.iter().min(), "max": v.iter().max(), "samples": v.len()}))
        })
        .collect();
    let receipt = json!({
        "tool": "qwen4_wide_grouped_moe_oracle",
        "model": model,
        "arch": gpu.arch,
        "seed": seed,
        "layers": layers,
        "row_sets": sizes.len(),
        "row_set_sizes": sizes,
        "mismatch_bytes_by_pair": by_pair.iter().map(|(k, v)| (k.to_string(), json!({
            "rot": v[0], "down": v[1], "rot_rows": v[2], "down_rows": v[3], "guard_violations": v[4]}))).collect::<BTreeMap<_,_>>(),
        "launch_errors": total_err,
        "symbols_by_route": symbols,
        "distinct_experts_by_rows_synthetic_router_input": distinct_summary,
        "k0_distinct_per_layer_at_30_32_rows": [144, 151],
        "layers_detail": per,
    });
    if let Some(out) = args.value("--out") {
        std::fs::write(out, serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write {out}: {e}"))?;
    }
    println!("SUMMARY row_sets={} launch_errors={total_err} mismatching_bytes_or_guards={total_bad}", sizes.len());
    for (k, v) in &by_pair {
        println!("  pair {k}: rot bytes {} down bytes {} guard {}", v[0], v[1], v[4]);
    }
    for (r, s) in &symbols {
        println!("  route {r} symbols {s:?}");
    }
    for ((layer, n), st) in &stats {
        for (k, f) in &st.first {
            println!("MISMATCH layer {layer} n={n} {k}: {f}");
        }
    }
    let bad = total_bad != 0 || total_err != 0;
    println!("RESULT {}", if bad { "FAIL" } else { "PASS" });
    Ok(bad as i32)
}

fn main() {
    match run() {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("qwen4_wide_grouped_moe_oracle: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(ids: [i32; K_TOP]) -> [i32; K_TOP] {
        ids
    }

    #[test]
    fn grouped_rows_bound_is_tile_aligned_and_covers_padding() {
        for n in 1..=64 {
            let t = n * K_TOP;
            let b = grouped_rows_bound(t);
            assert_eq!(b % BLOCK_M, 0);
            assert!(b >= t);
        }
        assert_eq!(grouped_rows_bound(320), 5120);
        assert_eq!(grouped_rows_bound(80), 1280);
    }

    #[test]
    fn census_unions_across_the_call_not_per_tile() {
        // 9 rows all routing to experts 0..10: one distinct set.
        let rows: Vec<[i32; K_TOP]> = (0..9).map(|_| route([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).collect();
        let c = read_census(&rows);
        assert_eq!(c.distinct, 10);
        assert_eq!(c.indexed, 90);
        // 9 slots per expert: ceil(9/8) = 2 tiles each.
        assert_eq!(c.grouped_wide_r8, 20);
        // groups of 8 and 1: ceil(8/2)=4 per expert + ceil(1/2)=1 per expert.
        assert_eq!(c.groups8_r2, 10 * 4 + 10);
    }

    #[test]
    fn disjoint_rows_read_once_per_slot_everywhere() {
        let rows: Vec<[i32; K_TOP]> = (0..9)
            .map(|r| std::array::from_fn(|i| (r * K_TOP + i) as i32))
            .collect();
        let c = read_census(&rows);
        assert_eq!((c.distinct, c.indexed, c.grouped_wide_r8), (90, 90, 90));
        assert_eq!(c.groups8_r2, 90);
    }

    #[test]
    fn diff_rows_first_difference() {
        let want = vec![1u32, 2, 3, 4];
        let mut got = want.clone();
        got[2] ^= 0x100;
        let d = diff_rows(&got, &want, 2);
        assert_eq!((d.bytes, d.rows), (1, 1));
        assert_eq!(d.first, Some((1, 1, 0x01, 0x00)));
    }
}
