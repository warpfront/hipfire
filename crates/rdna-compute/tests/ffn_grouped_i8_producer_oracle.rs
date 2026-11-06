// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: `Gpu::ffn_activation_grouped_rotate_i8_f32`
//! (`ffn_*_grouped_rotate_i8_gfx1201`) must emit exactly the `block_i8_128` blocks, and the
//! retained F32 rotated rows, of the chain it replaces on the Raon grouped-MoE down site:
//! `ffn_activation_grouped_f32` -> `rotate_x_mq_batched` -> `ensure_int8_mmq_x`
//! (K = 1280 = 1088 logical + 192 zero pad, 128 experts, top-6, SiTU 4/25, per-expert AWQ,
//! BF16 casts), and the grouped A8 GEMM over them must therefore produce the same bytes.
//!
//! Every byte of the whole `[K/128][slots]` block array and of the whole F32 rotated array is
//! compared for rows {1, 3, 63, 64, 512, 777, 2048, 4093, 4096} on lognormal-skew and
//! uniform expert histograms, with the data variants: Gaussian, injected NaN/+-inf/+-0/
//! subnormal/extreme-magnitude gate/up values, all-zero, tie (every value equal), per-expert
//! AWQ, a null AWQ table entry, out-of-range expert tile ids, no AWQ table, all three
//! activations and both cast settings. The destination scratch is poisoned between the two
//! producers, so a block the fused kernel left unwritten cannot pass. Padding rows of the
//! grouped order carry NaN gate/up values.
//!
//! `HIPFIRE_G5_CAPTURE` (a prefix of `<prefix>.gateup.f32`, `<prefix>.topk.i32`,
//! `<prefix>.awq.f32`, written from a real prefill) adds a captured real node to the oracle
//! and the microbench.
//!
//! `#[ignore]`d: needs an exact gfx1201 with a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test ffn_grouped_i8_producer_oracle \
//!       -- --ignored --test-threads=1 --nocapture
//!
//! `microbench` prints hipEvent medians of the stages of the chain, the existing
//! `rotate_x_mq_i8_batched` two-launch form, the fused launch, and the A8 GEMM they feed, and the
//! gate/up and shared-down chains for the same node.

use half::f16;
use rdna_compute::ffn_activation::{
    FfnActivation, FfnActivationGrouped, FfnActivationGroupedI8, FfnActivationRows,
    FfnActivationRowsI8, FfnAwq,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;

const E: usize = 128;
const TOP_K: usize = 6;
const HIDDEN: usize = 2560;
const LOGICAL: usize = 1088;
const PHYS: usize = 1280;
const BLOCK_M: usize = 16;
const BLOCK_BYTES: usize = 136;
const SITU: FfnActivation = FfnActivation::Situ {
    gate_beta: 4.0,
    value_beta: 25.0,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unit(), self.unit());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn bound_of(slots: usize) -> usize {
    let live = slots.min(E);
    (slots + live * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M
}

/// Gumbel-top-k over lognormal popularities (`sigma` 0 = uniform).
fn route(rows: usize, sigma: f32, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let logp: Vec<f32> = (0..E).map(|_| sigma * rng.normal()).collect();
    let mut topk = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut sc: Vec<(f32, usize)> = (0..E)
            .map(|e| (logp[e] - (-(rng.unit().ln())).ln() as f32, e))
            .collect();
        sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        topk.extend(sc.iter().take(TOP_K).map(|s| s.1 as i32));
    }
    topk
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dl_i32(gpu: &Gpu, t: &GpuTensor) -> Vec<i32> {
    gpu.download_raw_bytes(t)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dtoh(gpu: &Gpu, ptr: *mut c_void, bytes: usize) -> Vec<u8> {
    // SAFETY: `ptr` is the live A8 scratch with at least `bytes` bytes; the borrowed wrapper
    // neither owns nor frees it.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &buf).expect("dtoh");
    v
}

fn poison(gpu: &Gpu, ptr: *mut c_void, bytes: usize) {
    // SAFETY: as in `dtoh`.
    let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
    gpu.hip.memset(&buf, 0xA5, bytes).expect("poison");
}

fn require_gfx1201() -> Option<Gpu> {
    let gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("skip: needs gfx1201, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

/// One qt44 weight matrix `[rows, k]`: 136 B per 256 K, symmetric headers.
fn build_qt44(rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; rows * groups * 136];
    for g in 0..rows * groups {
        let o = g * 136;
        for half in 0..2 {
            let sc = f16::from_f32(0.004 + 0.012 * rng.unit() as f32);
            let zp = f16::from_f32(-8.0 * sc.to_f32());
            out[o + 4 * half..o + 4 * half + 2].copy_from_slice(&sc.to_bits().to_le_bytes());
            out[o + 4 * half + 2..o + 4 * half + 4].copy_from_slice(&zp.to_bits().to_le_bytes());
        }
        for i in (0..128).step_by(8) {
            out[o + 8 + i..o + 16 + i].copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Data {
    Gaussian,
    Specials,
    Zero,
    Tie,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Awq {
    PerExpert,
    NullEntry,
    BadTiles,
    None,
}

/// A routed node: routing tables, gate/up rows in grouped order and the AWQ table.
struct Node {
    rows: usize,
    slots: usize,
    bound: usize,
    topk: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inv: GpuTensor,
    y: GpuTensor,
    awq_table: Option<GpuTensor>,
    awq_vecs: Vec<GpuTensor>,
    m_total: usize,
    live_tiles: usize,
    max_expert_rows: usize,
}

/// Special F32 gate/up values: NaN (two payloads), +-inf, +-0, subnormals, extremes.
const SPECIAL_F32: [u32; 14] = [
    0x7fc0_0000,
    0x7fa0_0001,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x0080_0000,
    0x4cbe_bc20,
    0xccbe_bc20,
];

fn awq_vector(rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![1.0f32; PHYS];
    for x in v.iter_mut().take(LOGICAL) {
        *x = (0.3 * rng.normal()).exp();
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    gpu: &mut Gpu,
    topk: Vec<i32>,
    rows: usize,
    data: Data,
    awq: Awq,
    gate_up: Option<Vec<f32>>,
    awq_host: Option<Vec<Vec<f32>>>,
    seed: u64,
) -> Node {
    let slots = rows * TOP_K;
    let bound = bound_of(slots);
    let topk_t = gpu.upload_raw(&i32s(&topk), &[slots * 4]).unwrap();
    let counts = gpu.zeros(&[E], DType::F32).unwrap();
    let offsets = gpu.zeros(&[E + 1], DType::F32).unwrap();
    let sorted = gpu.zeros(&[bound], DType::F32).unwrap();
    let tiles = gpu.zeros(&[bound / BLOCK_M], DType::F32).unwrap();
    let inv = gpu.zeros(&[slots], DType::F32).unwrap();
    gpu.moe_scatter_fused_k8(
        &topk_t, &counts, &offsets, &sorted, &tiles, &inv, slots, E, bound, BLOCK_M,
    )
    .unwrap();
    let sorted_h = dl_i32(gpu, &sorted);
    let mut tiles_h = dl_i32(gpu, &tiles);
    let offs = dl_i32(gpu, &offsets);
    let m_total = offs[E] as usize;
    let live_tiles = tiles_h.iter().filter(|&&t| t >= 0).count();
    let max_expert_rows = (0..E)
        .map(|e| (offs[e + 1] - offs[e]) as usize)
        .max()
        .unwrap();
    let mut tiles = tiles;
    if awq == Awq::BadTiles && live_tiles >= 3 {
        // One tile names an expert past the table, one a negative expert: whole zero rows.
        tiles_h[0] = 999;
        tiles_h[1] = -1;
        tiles_h[live_tiles - 1] = E as i32;
        gpu.free_tensor(tiles).unwrap();
        tiles = gpu
            .upload_raw(&i32s(&tiles_h), &[tiles_h.len() * 4])
            .unwrap();
    }

    let mut rng = Rng(seed ^ 0xD1B5_4A32_D192_ED03 | 1);
    let mut y = vec![0f32; bound * 2 * LOGICAL];
    match gate_up {
        Some(c) => {
            assert_eq!(c.len(), y.len(), "captured gate/up extent");
            y = c;
        }
        None => {
            let tie = 0.7311f32;
            for v in y.iter_mut() {
                *v = match data {
                    Data::Gaussian => 2.0 * rng.normal(),
                    Data::Zero => 0.0,
                    Data::Tie => tie,
                    Data::Specials => {
                        let r = rng.below(64);
                        if r < 4 {
                            f32::from_bits(SPECIAL_F32[rng.below(14) as usize])
                        } else if r < 8 {
                            // large but finite: forces the rounding/saturation edges
                            2000.0 * rng.normal()
                        } else {
                            2.0 * rng.normal()
                        }
                    }
                };
            }
        }
    }
    // Padding rows of the grouped order are never read: NaN-fill them.
    for (g, s) in sorted_h.iter().enumerate() {
        if *s < 0 {
            y[g * 2 * LOGICAL..(g + 1) * 2 * LOGICAL].fill(f32::NAN);
        }
    }
    let y = gpu.upload_f32(&y, &[bound, 2 * LOGICAL]).unwrap();

    let (awq_table, awq_vecs) = if awq == Awq::None {
        (None, Vec::new())
    } else {
        let mut vecs = Vec::new();
        let mut ptrs = Vec::with_capacity(E * 8);
        for e in 0..E {
            let host = match &awq_host {
                Some(h) => h[e].clone(),
                None => awq_vector(&mut rng),
            };
            let t = gpu.upload_f32(&host, &[PHYS]).unwrap();
            let p = if awq == Awq::NullEntry && e % 7 == 3 {
                0u64
            } else {
                t.buf.as_ptr() as u64
            };
            ptrs.extend_from_slice(&p.to_le_bytes());
            vecs.push(t);
        }
        (Some(gpu.upload_raw(&ptrs, &[ptrs.len()]).unwrap()), vecs)
    };
    Node {
        rows,
        slots,
        bound,
        topk: topk_t,
        counts,
        offsets,
        sorted,
        tiles,
        inv,
        y,
        awq_table,
        awq_vecs,
        m_total,
        live_tiles,
        max_expert_rows,
    }
}

fn free_node(gpu: &mut Gpu, n: Node) {
    for t in [n.topk, n.counts, n.offsets, n.sorted, n.tiles, n.inv, n.y]
        .into_iter()
        .chain(n.awq_table)
        .chain(n.awq_vecs)
    {
        gpu.free_tensor(t).unwrap();
    }
}

struct Chain {
    act: GpuTensor,
    rot: GpuTensor,
    xo: GpuTensor,
}

fn chain_bufs(gpu: &mut Gpu, slots: usize) -> Chain {
    Chain {
        act: gpu.zeros(&[slots, PHYS], DType::F32).unwrap(),
        rot: gpu.zeros(&[slots, PHYS], DType::F32).unwrap(),
        xo: gpu.zeros(&[slots, PHYS], DType::F32).unwrap(),
    }
}

fn free_chain(gpu: &mut Gpu, c: Chain) {
    for t in [c.act, c.rot, c.xo] {
        gpu.free_tensor(t).unwrap();
    }
}

#[derive(Clone, Copy)]
struct Cfg {
    act: FfnActivation,
    cast: bool,
}

fn grouped_op<'a>(n: &'a Node, out: &'a GpuTensor, cfg: Cfg) -> FfnActivationGrouped<'a> {
    FfnActivationGrouped {
        activation: cfg.act,
        y_grouped: &n.y,
        sorted_slot_index: &n.sorted,
        expert_tile_ids: &n.tiles,
        out,
        m_total: n.bound,
        block_m: BLOCK_M,
        logical_n: LOGICAL,
        phys_n: PHYS,
        round_inputs: cfg.cast,
        round_output: cfg.cast,
        awq_table: n.awq_table.as_ref(),
        n_experts: E,
    }
}

/// The incumbent chain: activation, rotation, standalone A8 quantizer. Returns the sidecar.
fn chain_incumbent(gpu: &mut Gpu, n: &Node, c: &Chain, cfg: Cfg) -> *mut c_void {
    gpu.ffn_activation_grouped_f32(&grouped_op(n, &c.act, cfg))
        .unwrap();
    gpu.rotate_x_mq_batched(&c.act, &c.rot, PHYS, n.slots)
        .unwrap();
    gpu.ensure_int8_mmq_x(&c.rot, n.slots, PHYS).unwrap()
}

/// The existing two-launch fused-rotate form: activation, then `rotate_x_mq_i8_batched`.
fn chain_two_launch(gpu: &mut Gpu, n: &Node, c: &Chain, cfg: Cfg, keep_f32: bool) -> *mut c_void {
    gpu.ffn_activation_grouped_f32(&grouped_op(n, &c.act, cfg))
        .unwrap();
    let r = gpu.reserve_int8_mmq(PHYS, n.slots).unwrap();
    let p = gpu
        .rotate_x_mq_i8_batched(&c.act, keep_f32.then_some(&c.xo), r, PHYS, n.slots)
        .unwrap();
    gpu.int8_mmq_prepared_ptr(&p, PHYS, n.slots).unwrap()
}

/// The single fused launch.
fn chain_fused(gpu: &mut Gpu, n: &Node, c: &Chain, cfg: Cfg, keep_f32: bool) -> *mut c_void {
    let r = gpu.reserve_int8_mmq(PHYS, n.slots).unwrap();
    let p = gpu
        .ffn_activation_grouped_rotate_i8_f32(
            &FfnActivationGroupedI8 {
                activation: cfg.act,
                y_grouped: &n.y,
                sorted_slot_index: &n.sorted,
                expert_tile_ids: &n.tiles,
                x_out: keep_f32.then_some(&c.xo),
                m_total: n.bound,
                block_m: BLOCK_M,
                logical_n: LOGICAL,
                phys_n: PHYS,
                total_slots: n.slots,
                round_inputs: cfg.cast,
                round_output: cfg.cast,
                awq_table: n.awq_table.as_ref(),
                n_experts: E,
            },
            r,
        )
        .unwrap();
    gpu.int8_mmq_prepared_ptr(&p, PHYS, n.slots).unwrap()
}

fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    (
        a.iter().zip(b).filter(|(x, y)| x != y).count(),
        a.iter().zip(b).position(|(x, y)| x != y),
    )
}

struct Stats {
    blocks: usize,
    nonzero_d: usize,
}

/// Compare the incumbent chain with the existing two-launch form and the fused launch,
/// blocks and retained F32, for one node and config.
fn compare_node(gpu: &mut Gpu, n: &Node, cfg: Cfg, tag: &str) -> Stats {
    let c = chain_bufs(gpu, n.slots);
    let bytes = (PHYS / 128) * n.slots * BLOCK_BYTES;
    let ref_ptr = chain_incumbent(gpu, n, &c, cfg);
    let reference = dtoh(gpu, ref_ptr, bytes);
    let rot_ref = gpu.download_raw_bytes(&c.rot).unwrap();

    for keep in [false, true] {
        poison(gpu, ref_ptr, bytes);
        gpu.hip.memset(&c.xo.buf, 0xA5, c.xo.buf.size()).unwrap();
        let p = chain_two_launch(gpu, n, &c, cfg, keep);
        assert_eq!(gpu.last_launched_kernel(), Some("mq_rotate_x_i8_gfx12"));
        assert_eq!(p, ref_ptr, "both producers use the one A8 scratch");
        let (bad, first) = diff(&reference, &dtoh(gpu, p, bytes));
        assert_eq!(
            bad, 0,
            "{tag}: two-launch blocks differ: {bad} bytes, first {first:?}"
        );
        if keep {
            let (bad, first) = diff(&rot_ref, &gpu.download_raw_bytes(&c.xo).unwrap());
            assert_eq!(
                bad, 0,
                "{tag}: two-launch F32 differs: {bad} bytes, first {first:?}"
            );
        }
    }
    for keep in [false, true] {
        poison(gpu, ref_ptr, bytes);
        gpu.hip.memset(&c.xo.buf, 0xA5, c.xo.buf.size()).unwrap();
        let p = chain_fused(gpu, n, &c, cfg, keep);
        assert_eq!(
            gpu.last_launched_kernel(),
            Some(match cfg.act {
                FfnActivation::Situ { .. } => "ffn_situ_grouped_rotate_i8_gfx1201",
                FfnActivation::SwiGlu => "ffn_swiglu_grouped_rotate_i8_gfx1201",
                FfnActivation::SwiGluClamp { .. } => "ffn_swiglu_clamp_grouped_rotate_i8_gfx1201",
            })
        );
        assert_eq!(p, ref_ptr, "both producers use the one A8 scratch");
        let fused = dtoh(gpu, p, bytes);
        let (bad, first) = diff(&reference, &fused);
        assert_eq!(
            bad,
            0,
            "{tag}: fused blocks differ: {bad} of {bytes} bytes, first {first:?} (block {:?}, byte {:?})",
            first.map(|f| f / BLOCK_BYTES),
            first.map(|f| f % BLOCK_BYTES)
        );
        if keep {
            let (bad, first) = diff(&rot_ref, &gpu.download_raw_bytes(&c.xo).unwrap());
            assert_eq!(
                bad, 0,
                "{tag}: fused retained F32 differs: {bad} bytes, first {first:?}"
            );
        }
    }
    let nonzero_d = reference
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .filter(|b| b[..4] != [0; 4])
        .count();
    let blocks = reference.len() / BLOCK_BYTES;
    free_chain(gpu, c);
    Stats { blocks, nonzero_d }
}

const ROWS: [usize; 9] = [1, 3, 63, 64, 512, 777, 2048, 4093, 4096];

#[test]
#[ignore = "needs an exact gfx1201"]
fn blocks_are_bytewise_the_three_launch_chain() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let mut cases = 0usize;
    for (i, &rows) in ROWS.iter().enumerate() {
        for (sigma, route_name) in [(1.2f32, "skew"), (0.0, "uniform")] {
            if route_name == "uniform" && rows != 4096 && rows != 512 {
                continue;
            }
            for (data, awq, cfg) in [
                (
                    Data::Gaussian,
                    Awq::PerExpert,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Specials,
                    Awq::PerExpert,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Specials,
                    Awq::NullEntry,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Zero,
                    Awq::PerExpert,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Tie,
                    Awq::PerExpert,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    Awq::BadTiles,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    Awq::None,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    Awq::PerExpert,
                    Cfg {
                        act: SITU,
                        cast: false,
                    },
                ),
                (
                    Data::Specials,
                    Awq::PerExpert,
                    Cfg {
                        act: FfnActivation::SwiGlu,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    Awq::PerExpert,
                    Cfg {
                        act: FfnActivation::SwiGluClamp { limit: 7.0 },
                        cast: true,
                    },
                ),
            ] {
                // The full variant matrix on the sizes that matter, a thinner one elsewhere.
                if !matches!(rows, 64 | 512 | 4096)
                    && !matches!(
                        (data, awq),
                        (Data::Gaussian, Awq::PerExpert) | (Data::Specials, Awq::NullEntry)
                    )
                {
                    continue;
                }
                let topk = route(rows, sigma, 0x5EED_0000 + i as u64);
                let node = build_node(
                    &mut gpu,
                    topk,
                    rows,
                    data,
                    awq,
                    None,
                    None,
                    0xA11C + i as u64,
                );
                let tag = format!(
                    "rows {rows} {route_name} {data:?} {awq:?} {:?} cast {}",
                    cfg.act, cfg.cast
                );
                let st = compare_node(&mut gpu, &node, cfg, &tag);
                if data == Data::Gaussian && rows >= 4 {
                    assert!(st.nonzero_d > 0, "{tag}: no block has a scale");
                }
                if data == Data::Zero {
                    assert_eq!(st.nonzero_d, 0, "{tag}: zero input must give zero scales");
                }
                eprintln!(
                    "{tag}: {} blocks identical ({} with scale), m_total {} live_tiles {} max_expert_rows {}",
                    st.blocks, st.nonzero_d, node.m_total, node.live_tiles, node.max_expert_rows
                );
                free_node(&mut gpu, node);
                cases += 1;
            }
        }
    }
    eprintln!("{cases} cases, every block and retained-F32 byte identical");
}

// ───────────────────────── dense rows: shared expert / dense FFN down ─────────────────────────

/// Gate/up planes `[rows, mi]` (each row-strided by `mi`) and an optional F32 AWQ vector `[k]`.
struct RowsNode {
    rows: usize,
    mi: usize,
    k: usize,
    gate: GpuTensor,
    up: GpuTensor,
    awq: Option<GpuTensor>,
}

fn build_rows_node(
    gpu: &mut Gpu,
    rows: usize,
    mi: usize,
    k: usize,
    data: Data,
    awq: bool,
    seed: u64,
) -> RowsNode {
    let mut rng = Rng(seed | 1);
    let mut plane = |rng: &mut Rng| -> Vec<f32> {
        (0..rows * mi)
            .map(|_| match data {
                Data::Gaussian => 2.0 * rng.normal(),
                Data::Zero => 0.0,
                Data::Tie => 0.7311,
                Data::Specials => {
                    let r = rng.below(64);
                    if r < 4 {
                        f32::from_bits(SPECIAL_F32[rng.below(14) as usize])
                    } else if r < 8 {
                        2000.0 * rng.normal()
                    } else {
                        2.0 * rng.normal()
                    }
                }
            })
            .collect()
    };
    let g = plane(&mut rng);
    let u = plane(&mut rng);
    let gate = gpu.upload_f32(&g, &[rows, mi]).unwrap();
    let up = gpu.upload_f32(&u, &[rows, mi]).unwrap();
    let awq = awq.then(|| {
        let mut v = vec![1.0f32; k];
        for x in v.iter_mut().take(mi) {
            *x = (0.3 * rng.normal()).exp();
        }
        gpu.upload_f32(&v, &[k]).unwrap()
    });
    RowsNode {
        rows,
        mi,
        k,
        gate,
        up,
        awq,
    }
}

fn free_rows_node(gpu: &mut Gpu, n: RowsNode) {
    for t in [n.gate, n.up].into_iter().chain(n.awq) {
        gpu.free_tensor(t).unwrap();
    }
}

fn rows_incumbent(
    gpu: &mut Gpu,
    n: &RowsNode,
    act: &GpuTensor,
    rot: &GpuTensor,
    cfg: Cfg,
) -> *mut c_void {
    gpu.ffn_activation_rows_f32(&FfnActivationRows {
        activation: cfg.act,
        gate: &n.gate,
        gate_offset: 0,
        gate_ld: n.mi,
        up: &n.up,
        up_offset: 0,
        up_ld: n.mi,
        out: act,
        rows: n.rows,
        logical_n: n.mi,
        phys_n: n.k,
        round_inputs: cfg.cast,
        round_output: cfg.cast,
        awq: n.awq.as_ref().map_or(FfnAwq::None, FfnAwq::Vector),
    })
    .unwrap();
    gpu.rotate_x_mq_batched(act, rot, n.k, n.rows).unwrap();
    gpu.ensure_int8_mmq_x(rot, n.rows, n.k).unwrap()
}

fn rows_fused(gpu: &mut Gpu, n: &RowsNode, xo: Option<&GpuTensor>, cfg: Cfg) -> *mut c_void {
    let r = gpu.reserve_int8_mmq(n.k, n.rows).unwrap();
    let p = gpu
        .ffn_activation_rows_rotate_i8_f32(
            &FfnActivationRowsI8 {
                activation: cfg.act,
                gate: &n.gate,
                gate_offset: 0,
                gate_ld: n.mi,
                up: &n.up,
                up_offset: 0,
                up_ld: n.mi,
                x_out: xo,
                rows: n.rows,
                logical_n: n.mi,
                phys_n: n.k,
                round_inputs: cfg.cast,
                round_output: cfg.cast,
                awq: n.awq.as_ref(),
            },
            r,
        )
        .unwrap();
    gpu.int8_mmq_prepared_ptr(&p, n.k, n.rows).unwrap()
}

#[test]
#[ignore = "needs an exact gfx1201"]
fn rows_blocks_are_bytewise_the_three_launch_chain() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let mut cases = 0usize;
    // (mi, k): the Raon shared expert (1088 -> 1280) and the first-layer dense MLP (12032 -> 12032).
    for (mi, k) in [(LOGICAL, PHYS), (12032usize, 12032usize)] {
        for (i, &rows) in ROWS.iter().enumerate() {
            for (data, awq, cfg) in [
                (
                    Data::Gaussian,
                    true,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    false,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Specials,
                    true,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Zero,
                    true,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Tie,
                    true,
                    Cfg {
                        act: SITU,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    true,
                    Cfg {
                        act: SITU,
                        cast: false,
                    },
                ),
                (
                    Data::Specials,
                    true,
                    Cfg {
                        act: FfnActivation::SwiGlu,
                        cast: true,
                    },
                ),
                (
                    Data::Gaussian,
                    true,
                    Cfg {
                        act: FfnActivation::SwiGluClamp { limit: 7.0 },
                        cast: true,
                    },
                ),
            ] {
                if !matches!(rows, 64 | 512 | 4096)
                    && !matches!((data, awq), (Data::Gaussian, true) | (Data::Specials, true))
                {
                    continue;
                }
                let n = build_rows_node(&mut gpu, rows, mi, k, data, awq, 0x7A5 + i as u64);
                let act = gpu.zeros(&[rows, k], DType::F32).unwrap();
                let rot = gpu.zeros(&[rows, k], DType::F32).unwrap();
                let xo = gpu.zeros(&[rows, k], DType::F32).unwrap();
                let tag = format!(
                    "rows-site mi {mi} k {k} rows {rows} {data:?} awq {awq} {:?} cast {}",
                    cfg.act, cfg.cast
                );
                let bytes = (k / 128) * rows * BLOCK_BYTES;
                let ref_ptr = rows_incumbent(&mut gpu, &n, &act, &rot, cfg);
                let reference = dtoh(&gpu, ref_ptr, bytes);
                let rot_ref = gpu.download_raw_bytes(&rot).unwrap();
                for keep in [false, true] {
                    poison(&gpu, ref_ptr, bytes);
                    gpu.hip.memset(&xo.buf, 0xA5, xo.buf.size()).unwrap();
                    let p = rows_fused(&mut gpu, &n, keep.then_some(&xo), cfg);
                    assert_eq!(p, ref_ptr, "{tag}: one A8 scratch");
                    let (bad, first) = diff(&reference, &dtoh(&gpu, p, bytes));
                    assert_eq!(
                        bad, 0,
                        "{tag}: fused blocks differ: {bad} of {bytes} bytes, first {first:?}"
                    );
                    if keep {
                        let (bad, first) = diff(&rot_ref, &gpu.download_raw_bytes(&xo).unwrap());
                        assert_eq!(
                            bad, 0,
                            "{tag}: retained F32 differs: {bad} bytes, first {first:?}"
                        );
                    }
                }
                let nonzero_d = reference
                    .as_chunks::<BLOCK_BYTES>()
                    .0
                    .iter()
                    .filter(|b| b[..4] != [0; 4])
                    .count();
                if data == Data::Gaussian && rows >= 4 {
                    assert!(nonzero_d > 0, "{tag}: no block has a scale");
                }
                if data == Data::Zero {
                    assert_eq!(nonzero_d, 0, "{tag}: zero input must give zero scales");
                }
                eprintln!(
                    "{tag}: {} blocks identical ({nonzero_d} with scale)",
                    reference.len() / BLOCK_BYTES
                );
                for t in [act, rot, xo] {
                    gpu.free_tensor(t).unwrap();
                }
                free_rows_node(&mut gpu, n);
                cases += 1;
            }
        }
    }
    eprintln!("{cases} rows-site cases, every block and retained-F32 byte identical");
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A captured real node: (rows, topk, gate/up in grouped order, per-expert AWQ vectors).
fn load_capture(prefix: &str) -> (usize, Vec<i32>, Vec<f32>, Vec<Vec<f32>>) {
    let topk: Vec<i32> = std::fs::read(format!("{prefix}.topk.i32"))
        .unwrap_or_else(|e| panic!("{prefix}.topk.i32: {e}"))
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(topk.len() % TOP_K, 0);
    let y = read_f32(&format!("{prefix}.gateup.f32"));
    let awq = read_f32(&format!("{prefix}.awq.f32"));
    assert_eq!(awq.len(), E * PHYS, "captured AWQ extent");
    let awq: Vec<Vec<f32>> = awq.chunks_exact(PHYS).map(|c| c.to_vec()).collect();
    (topk.len() / TOP_K, topk, y, awq)
}

fn capture_prefixes() -> Vec<String> {
    std::env::var("HIPFIRE_G5_CAPTURE")
        .map(|s| s.split(',').map(str::to_owned).collect())
        .unwrap_or_default()
}

#[test]
#[ignore = "needs an exact gfx1201 and HIPFIRE_G5_CAPTURE"]
fn captured_real_nodes_are_bytewise_the_three_launch_chain() {
    let prefixes = capture_prefixes();
    if prefixes.is_empty() {
        eprintln!("skip: HIPFIRE_G5_CAPTURE unset");
        return;
    }
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    for prefix in prefixes {
        let (rows, topk, y, awq) = load_capture(&prefix);
        let node = build_node(
            &mut gpu,
            topk,
            rows,
            Data::Gaussian,
            Awq::PerExpert,
            Some(y),
            Some(awq),
            1,
        );
        let st = compare_node(
            &mut gpu,
            &node,
            Cfg {
                act: SITU,
                cast: true,
            },
            &prefix,
        );
        eprintln!(
            "captured {prefix}: rows {rows}: {} blocks identical ({} with scale), m_total {} live_tiles {} max_expert_rows {}",
            st.blocks, st.nonzero_d, node.m_total, node.live_tiles, node.max_expert_rows
        );
        free_node(&mut gpu, node);
    }
}

/// Down-projection weights of every expert plus the pointer table.
struct DownWeights {
    ptrs: GpuTensor,
    _owners: Vec<GpuTensor>,
}

fn down_weights(gpu: &mut Gpu, experts: usize, m: usize, k: usize) -> DownWeights {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut owners = Vec::new();
    let mut ptrs = Vec::new();
    for _ in 0..experts {
        let w = build_qt44(m, k, &mut rng);
        let t = gpu.upload_raw(&w, &[w.len()]).unwrap();
        ptrs.extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
        owners.push(t);
    }
    DownWeights {
        ptrs: gpu.upload_raw(&ptrs, &[ptrs.len()]).unwrap(),
        _owners: owners,
    }
}

fn median_us(gpu: &mut Gpu, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..5 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut s = Vec::with_capacity(iters);
    for _ in 0..iters {
        gpu.hip.event_record(&start, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&stop, None).unwrap();
        gpu.hip.event_synchronize(&stop).unwrap();
        s.push(gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1000.0);
    }
    gpu.hip.event_destroy(start).unwrap();
    gpu.hip.event_destroy(stop).unwrap();
    s.sort_by(f64::total_cmp);
    s[s.len() / 2]
}

fn gemm_down(gpu: &mut Gpu, n: &Node, w: &DownWeights, xq: *mut c_void, y: &GpuTensor) {
    let (nt, rf) = rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(n.bound, HIDDEN);
    gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
        &w.ptrs, &n.tiles, &n.offsets, &n.sorted, xq, y, HIDDEN, PHYS, 1, n.bound, n.slots, nt, rf,
    )
    .unwrap();
}

#[test]
#[ignore = "needs an exact gfx1201; timing"]
fn gemm_bytes_are_unchanged() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let w = down_weights(&mut gpu, E, HIDDEN, PHYS);
    for (i, rows) in [512usize, 4096].into_iter().enumerate() {
        let topk = route(rows, 1.2, 0x77 + i as u64);
        let node = build_node(
            &mut gpu,
            topk,
            rows,
            Data::Gaussian,
            Awq::PerExpert,
            None,
            None,
            3,
        );
        let c = chain_bufs(&mut gpu, node.slots);
        let cfg = Cfg {
            act: SITU,
            cast: true,
        };
        let y_ref = gpu.zeros(&[node.bound, HIDDEN], DType::F32).unwrap();
        let y_new = gpu.zeros(&[node.bound, HIDDEN], DType::F32).unwrap();
        let xq = chain_incumbent(&mut gpu, &node, &c, cfg);
        gemm_down(&mut gpu, &node, &w, xq, &y_ref);
        let xq = chain_fused(&mut gpu, &node, &c, cfg, false);
        gemm_down(&mut gpu, &node, &w, xq, &y_new);
        let (a, b) = (
            gpu.download_raw_bytes(&y_ref).unwrap(),
            gpu.download_raw_bytes(&y_new).unwrap(),
        );
        let (bad, first) = diff(&a, &b);
        assert_eq!(
            bad, 0,
            "rows {rows}: {bad} GEMM bytes differ, first {first:?}"
        );
        eprintln!("rows {rows}: {} down GEMM output bytes identical", a.len());
        for t in [y_ref, y_new] {
            gpu.free_tensor(t).unwrap();
        }
        free_chain(&mut gpu, c);
        free_node(&mut gpu, node);
    }
}

fn bench_node(gpu: &mut Gpu, w: &DownWeights, n: &Node, label: &str) {
    let c = chain_bufs(gpu, n.slots);
    let cfg = Cfg {
        act: SITU,
        cast: true,
    };
    let y = gpu.zeros(&[n.bound, HIDDEN], DType::F32).unwrap();
    // JIT everything once.
    let xq = chain_incumbent(gpu, n, &c, cfg);
    gemm_down(gpu, n, w, xq, &y);
    let _ = chain_two_launch(gpu, n, &c, cfg, false);
    let _ = chain_fused(gpu, n, &c, cfg, false);
    let it = 30;
    let t_act = median_us(gpu, it, |g| {
        g.ffn_activation_grouped_f32(&grouped_op(n, &c.act, cfg))
            .unwrap();
    });
    let t_rot = median_us(gpu, it, |g| {
        g.rotate_x_mq_batched(&c.act, &c.rot, PHYS, n.slots)
            .unwrap();
    });
    let t_quant = median_us(gpu, it, |g| {
        g.ensure_int8_mmq_x(&c.rot, n.slots, PHYS).unwrap();
    });
    let t_inc = median_us(gpu, it, |g| {
        chain_incumbent(g, n, &c, cfg);
    });
    let t_two = median_us(gpu, it, |g| {
        chain_two_launch(g, n, &c, cfg, false);
    });
    let t_fused = median_us(gpu, it, |g| {
        chain_fused(g, n, &c, cfg, false);
    });
    let t_fused_f32 = median_us(gpu, it, |g| {
        chain_fused(g, n, &c, cfg, true);
    });
    let xq = chain_incumbent(gpu, n, &c, cfg);
    let t_gemm = median_us(gpu, it, |g| gemm_down(g, n, w, xq, &y));
    // Whole chains, producer + GEMM, back to back, the way the dispatcher runs them.
    let t_full_inc = median_us(gpu, it, |g| {
        let xq = chain_incumbent(g, n, &c, cfg);
        gemm_down(g, n, w, xq, &y);
    });
    let t_full_two = median_us(gpu, it, |g| {
        let xq = chain_two_launch(g, n, &c, cfg, false);
        gemm_down(g, n, w, xq, &y);
    });
    let t_full_fused = median_us(gpu, it, |g| {
        let xq = chain_fused(g, n, &c, cfg, false);
        gemm_down(g, n, w, xq, &y);
    });
    let prod = t_act + t_rot + t_quant;
    println!(
        "G5DOWN {label} rows={} slots={} bound={} m_total={} live_tiles={} max_expert_rows={} | \
         act={t_act:.1} rot={t_rot:.1} quant={t_quant:.1} sum={prod:.1} chain_inc={t_inc:.1} \
         two_launch={t_two:.1} fused={t_fused:.1} fused+f32={t_fused_f32:.1} gemm={t_gemm:.1} | \
         full_inc={t_full_inc:.1} full_two={t_full_two:.1} full_fused={t_full_fused:.1} | \
         producer_share_of_inc={:.1}% gain_two={:.2}% gain_fused={:.2}%",
        n.rows,
        n.slots,
        n.bound,
        n.m_total,
        n.live_tiles,
        n.max_expert_rows,
        100.0 * t_inc / t_full_inc,
        100.0 * (t_full_inc - t_full_two) / t_full_inc,
        100.0 * (t_full_inc - t_full_fused) / t_full_inc,
    );
    gpu.free_tensor(y).unwrap();
    free_chain(gpu, c);
}

/// One-expert grouping tables + weight of a dense `[m, k]` qt44 projection driven through the
/// grouped A8 core exactly as `gated_ffn_from_basis` does (`index` all zero doubles as offsets).
struct OneExpert {
    ptrs: GpuTensor,
    tiles: GpuTensor,
    index: GpuTensor,
    sorted: GpuTensor,
    rows_pad: usize,
    m: usize,
    k: usize,
    _w: GpuTensor,
}

fn one_expert(gpu: &mut Gpu, rows: usize, m: usize, k: usize) -> OneExpert {
    let rows_pad = rows.div_ceil(16) * 16;
    let mut rng = Rng(0xABCD_0123_4567_89EF);
    let w = build_qt44(m, k, &mut rng);
    let wt = gpu.upload_raw(&w, &[w.len()]).unwrap();
    let ptrs = gpu
        .upload_raw(&(wt.buf.as_ptr() as u64).to_le_bytes(), &[8])
        .unwrap();
    let sorted_h: Vec<i32> = (0..rows_pad)
        .map(|i| if i < rows { i as i32 } else { -1 })
        .collect();
    OneExpert {
        ptrs,
        tiles: gpu
            .upload_raw(&i32s(&vec![0; rows_pad / 16]), &[rows_pad / 16 * 4])
            .unwrap(),
        index: gpu
            .upload_raw(&i32s(&vec![0; rows_pad]), &[rows_pad * 4])
            .unwrap(),
        sorted: gpu.upload_raw(&i32s(&sorted_h), &[rows_pad * 4]).unwrap(),
        rows_pad,
        m,
        k,
        _w: wt,
    }
}

fn gemm_one(gpu: &mut Gpu, o: &OneExpert, rows: usize, xq: *mut c_void, y: &GpuTensor) {
    let (nt, rf) = rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(o.rows_pad, o.m);
    gpu.gemm_mq4g256v2_moe_grouped_a8_sym(
        &o.ptrs, &o.tiles, &o.index, &o.sorted, xq, y, o.m, o.k, 1, o.rows_pad, rows, nt, rf,
    )
    .unwrap();
}

/// Shared-expert / dense-FFN down chain (`gated_ffn_from_basis`): activation rows, rotation,
/// quantizer, GEMM, against the fused launch.
fn bench_rows_site(gpu: &mut Gpu, rows: usize, mi: usize, k: usize, label: &str) {
    let n = build_rows_node(gpu, rows, mi, k, Data::Gaussian, true, 0x51);
    let o = one_expert(gpu, rows, HIDDEN, k);
    let act = gpu.zeros(&[rows, k], DType::F32).unwrap();
    let rot = gpu.zeros(&[rows, k], DType::F32).unwrap();
    let y = gpu.zeros(&[o.rows_pad, HIDDEN], DType::F32).unwrap();
    let cfg = Cfg {
        act: SITU,
        cast: true,
    };
    let xq = rows_incumbent(gpu, &n, &act, &rot, cfg);
    gemm_one(gpu, &o, rows, xq, &y);
    let _ = rows_fused(gpu, &n, None, cfg);
    let it = 30;
    let t_act = median_us(gpu, it, |g| {
        g.ffn_activation_rows_f32(&FfnActivationRows {
            activation: cfg.act,
            gate: &n.gate,
            gate_offset: 0,
            gate_ld: mi,
            up: &n.up,
            up_offset: 0,
            up_ld: mi,
            out: &act,
            rows,
            logical_n: mi,
            phys_n: k,
            round_inputs: true,
            round_output: true,
            awq: n.awq.as_ref().map_or(FfnAwq::None, FfnAwq::Vector),
        })
        .unwrap();
    });
    let t_rot = median_us(gpu, it, |g| {
        g.rotate_x_mq_batched(&act, &rot, k, rows).unwrap();
    });
    let t_quant = median_us(gpu, it, |g| {
        g.ensure_int8_mmq_x(&rot, rows, k).unwrap();
    });
    let t_inc = median_us(gpu, it, |g| {
        rows_incumbent(g, &n, &act, &rot, cfg);
    });
    let t_fused = median_us(gpu, it, |g| {
        rows_fused(g, &n, None, cfg);
    });
    let xq = rows_incumbent(gpu, &n, &act, &rot, cfg);
    let t_gemm = median_us(gpu, it, |g| gemm_one(g, &o, rows, xq, &y));
    let t_full_inc = median_us(gpu, it, |g| {
        let xq = rows_incumbent(g, &n, &act, &rot, cfg);
        gemm_one(g, &o, rows, xq, &y);
    });
    let t_full_fused = median_us(gpu, it, |g| {
        let xq = rows_fused(g, &n, None, cfg);
        gemm_one(g, &o, rows, xq, &y);
    });
    println!(
        "G5ROWS {label} rows={rows} mi={mi} k={k} | act={t_act:.1} rot={t_rot:.1} quant={t_quant:.1} \
         sum={:.1} chain_inc={t_inc:.1} fused={t_fused:.1} gemm={t_gemm:.1} | \
         full_inc={t_full_inc:.1} full_fused={t_full_fused:.1} | producer_share_of_inc={:.1}% gain_fused={:.2}%",
        t_act + t_rot + t_quant,
        100.0 * t_inc / t_full_inc,
        100.0 * (t_full_inc - t_full_fused) / t_full_inc,
    );
    for t in [act, rot, y] {
        gpu.free_tensor(t).unwrap();
    }
    free_rows_node(gpu, n);
}

/// Routed gate/up input chain: AWQ rotation, standalone quantizer, grouped GEMM, with the
/// existing AWQ-less `rotate_x_mq_i8_batched` timed as a lower-bound proxy of any AWQ-capable
/// one-pass producer (the AWQ divide adds one load per element to it).
fn bench_gu(gpu: &mut Gpu, wgu: &DownWeights, n: &Node, label: &str) {
    let rows = n.rows;
    let mut rng = Rng(0x6B5 + rows as u64);
    let x = gpu
        .upload_f32(
            &(0..rows * HIDDEN)
                .map(|_| rng.normal())
                .collect::<Vec<f32>>(),
            &[rows, HIDDEN],
        )
        .unwrap();
    let awq_h: Vec<f32> = (0..HIDDEN).map(|_| (0.3 * rng.normal()).exp()).collect();
    let awq = gpu.upload_f32(&awq_h, &[HIDDEN]).unwrap();
    let x_rot = gpu.zeros(&[rows, HIDDEN], DType::F32).unwrap();
    let y = gpu.zeros(&[n.bound, 2 * LOGICAL], DType::F32).unwrap();
    let (nt, rf) = rdna_compute::moe_grouped_qt44_iu4::select_entry_a8(n.bound, 2 * LOGICAL);
    let gemm = |g: &mut Gpu, xq: *mut c_void| {
        g.gemm_mq4g256v2_moe_grouped_a8_sym(
            &wgu.ptrs,
            &n.tiles,
            &n.offsets,
            &n.sorted,
            xq,
            &y,
            2 * LOGICAL,
            HIDDEN,
            TOP_K,
            n.bound,
            rows,
            nt,
            rf,
        )
        .unwrap();
    };
    let inc = |g: &mut Gpu| -> *mut c_void {
        g.rotate_x_mq_awq_batched(&x, &awq, &x_rot, HIDDEN, rows)
            .unwrap();
        g.ensure_int8_mmq_x(&x_rot, rows, HIDDEN).unwrap()
    };
    let proxy = |g: &mut Gpu, keep: bool| -> *mut c_void {
        let r = g.reserve_int8_mmq(HIDDEN, rows).unwrap();
        let p = g
            .rotate_x_mq_i8_batched(&x, keep.then_some(&x_rot), r, HIDDEN, rows)
            .unwrap();
        g.int8_mmq_prepared_ptr(&p, HIDDEN, rows).unwrap()
    };
    let xq = inc(gpu);
    gemm(gpu, xq);
    let _ = proxy(gpu, false);
    let it = 30;
    let t_rot = median_us(gpu, it, |g| {
        g.rotate_x_mq_awq_batched(&x, &awq, &x_rot, HIDDEN, rows)
            .unwrap();
    });
    let t_quant = median_us(gpu, it, |g| {
        g.ensure_int8_mmq_x(&x_rot, rows, HIDDEN).unwrap();
    });
    let t_inc = median_us(gpu, it, |g| {
        inc(g);
    });
    let t_proxy = median_us(gpu, it, |g| {
        proxy(g, false);
    });
    let t_proxy_f32 = median_us(gpu, it, |g| {
        proxy(g, true);
    });
    let xq = inc(gpu);
    let t_gemm = median_us(gpu, it, |g| gemm(g, xq));
    let t_full_inc = median_us(gpu, it, |g| {
        let xq = inc(g);
        gemm(g, xq);
    });
    let t_full_proxy = median_us(gpu, it, |g| {
        let xq = proxy(g, true);
        gemm(g, xq);
    });
    println!(
        "G5GU {label} rows={rows} m_total={} | rot_awq={t_rot:.1} quant={t_quant:.1} chain_inc={t_inc:.1} \
         awqless_onepass={t_proxy:.1} awqless_onepass+f32={t_proxy_f32:.1} gemm={t_gemm:.1} | \
         full_inc={t_full_inc:.1} full_onepass_proxy+f32={t_full_proxy:.1} | \
         producer_share_of_inc={:.1}% max_gain_if_awq_free={:.2}%",
        n.m_total,
        100.0 * t_inc / t_full_inc,
        100.0 * (t_full_inc - t_full_proxy) / t_full_inc,
    );
    for t in [x, awq, x_rot, y] {
        gpu.free_tensor(t).unwrap();
    }
}

#[test]
#[ignore = "needs an exact gfx1201; timing"]
fn microbench() {
    let Some(mut gpu) = require_gfx1201() else {
        return;
    };
    let synthetic = std::env::var_os("HIPFIRE_G5_ONLY_CAPTURE").is_none();
    let wgu = down_weights(&mut gpu, E, 2 * LOGICAL, HIDDEN);
    for rows in [64usize, 512, 2048, 4096].into_iter().filter(|_| synthetic) {
        bench_rows_site(&mut gpu, rows, LOGICAL, PHYS, "shared");
    }
    if synthetic {
        bench_rows_site(&mut gpu, 4096, 12032, 12032, "dense-l0");
    }
    for rows in [64usize, 512, 2048, 4096].into_iter().filter(|_| synthetic) {
        let topk = route(rows, 1.2, 0xBEEF + rows as u64);
        let node = build_node(
            &mut gpu,
            topk,
            rows,
            Data::Gaussian,
            Awq::PerExpert,
            None,
            None,
            9,
        );
        bench_gu(&mut gpu, &wgu, &node, "skew");
        free_node(&mut gpu, node);
    }
    let w = down_weights(&mut gpu, E, HIDDEN, PHYS);
    for rows in [64usize, 512, 2048, 4096].into_iter().filter(|_| synthetic) {
        for (sigma, name) in [(1.2f32, "skew"), (0.0, "uniform")] {
            let topk = route(rows, sigma, 0xBEEF + rows as u64);
            let node = build_node(
                &mut gpu,
                topk,
                rows,
                Data::Gaussian,
                Awq::PerExpert,
                None,
                None,
                9,
            );
            bench_node(&mut gpu, &w, &node, name);
            free_node(&mut gpu, node);
        }
    }
    for prefix in capture_prefixes() {
        let (rows, topk, y, awq) = load_capture(&prefix);
        let node = build_node(
            &mut gpu,
            topk,
            rows,
            Data::Gaussian,
            Awq::PerExpert,
            Some(y),
            Some(awq),
            1,
        );
        bench_node(&mut gpu, &w, &node, &format!("captured:{prefix}"));
        free_node(&mut gpu, node);
    }
}
