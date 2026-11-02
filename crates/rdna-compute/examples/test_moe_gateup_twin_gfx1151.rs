// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Bitwise oracle + timing for the gfx1151 QT44 grouped top-10 MoE gate/up
//! unrolled twin, against the production incumbent:
//!
//!   incumbent  `gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4`
//!              module = symbol, block [128,1,1],
//!              grid [ceil(M/16), ceil(grouped_rows/8), 1]
//!   twin       `gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe_gfx1151` (module
//!              `gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe`), same geometry.
//!              The scheduling variants this oracle ranked on Halo (W waves per
//!              block, prefetch depth, double-buffered LDS; commit 0b0fd7848)
//!              were reduced to the winner, W = 4 with one group in flight.
//!
//! The variants live in the `VARIANTS` table: adding/removing a twin is one line.
//! All kernels take the same nine production arguments (expert_weight_ptrs,
//! expert_tile_ids, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div,
//! grouped_rows) and are compiled straight from the `.hip` sources
//! (`include_str!`); no dispatch/production code is touched. A short SHA-256
//! prefix of each source string is printed at startup.
//!
//! ## What is verified
//! Every f32 bit of `Y_grouped` of each twin (the whole `grouped_rows * M`
//! region PLUS a 16-row guard band after it) equals the incumbent's. Before each
//! variant's correctness launch the buffer is refilled with the NaN-payload
//! sentinel `0x7fc01234`. Cells the incumbent leaves untouched (sentinel tiles
//! with a negative expert id are skipped by design) must be left untouched by
//! the twins too, so unwritten cells are reported, not rejected. Required: the
//! guard band is untouched, no written cell is non-finite, and the incumbent has
//! at least one finite nonzero cell whenever live slots exist.
//!
//! ## Routing / layout (production scatter, not a CPU imitation)
//! `sorted_slot_index` / `expert_tile_ids` come from `Gpu::moe_scatter_fused_top10`
//! (block_m = 16) over seeded top-10-without-replacement routing (uniform or
//! Zipf(1.1)-skewed) with E = 512. `grouped_rows` is the production worst-case
//! padded bound `(slots + 15 * min(slots, E)).next_multiple_of(16)` unless the
//! case says otherwise; tiles past the live padded count are `-1` sentinels. The
//! scatter output is structurally validated on the host before any GEMM runs.
//! X row = flat_slot / 10 = token (`x_row_div = 10`), `X_src` is [tokens, K].
//!
//! ## Data
//! E = 512, M = 1280, K = 2560. MQ4G256V2 weights: per row 10 groups x 136 B
//! (row stride 1360 B). Group = f16 scale + f16 zero for values 0..127, f16
//! scale + f16 zero for values 128..255, then 128 random payload bytes. Scales
//! are random finite signed f16 with |scale| in ~[1e-3, 0.13], zeros |zero| <=
//! 0.5, so sums stay finite. X is seeded random f32 in [-2, 2). Weights are built
//! once per process (~0.9 GB) and shared by every case.
//!
//! ## Cases (default run)
//!   u{T}     uniform routing, T in 9,17,33,65,97,129,145,...,497,529 (step 32)
//!   skew{T}  Zipf(1.1) routing, T in 145, 273, 401
//!   partial145   grouped_rows = live_padded - 5 (not a multiple of 8 or 16)
//!   invalid145   invalid router ids injected before the scatter
//!   mutate145    sentinel tiles (-1, -7), -1 slots inside live tiles, an
//!                all-dead live tile and a half-dead live tile (no out-of-range
//!                tile ids or X rows: the gate/up kernels have no bounds check)
//! `--tokens=a,b,c` replaces the uniform list (no skewed / edge cases).
//!
//! ## Timing
//! Per variant WARMUP launches, then SAMPLES individually HIP-event-timed
//! launches, launch order cyclically shifted per sample. Reported: median and
//! min per launch, incumbent_median / variant_median, and effective GB/s (1 GB =
//! 1e9 B) over the "ideal unique traffic" denominator
//!   W = (# experts owning >= 1 tile with >= 1 live slot) * M * 1360
//!   X = (# distinct referenced X rows) * K * 4
//!   Y = grouped_rows * M * 4
//! which is NOT measured DRAM traffic (reuse is credited as free).
//!
//! GFX1151 ONLY: refuses any other arch (exit 2). Exit 1 on any mismatch.
//!
//! Run:
//!   cargo run --release -p rdna-compute --features lab --example test_moe_gateup_twin_gfx1151 [-- --tokens=145,273 --variants=pipe --precheck] [case-name-filter]

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};

const SRC_INCUMBENT: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4.gfx1151.hip");
const SRC_PIPE: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe.gfx1151.hip");
const MODULE_INCUMBENT: &str = "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4";
const MODULE_PIPE: &str = "gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe";

#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    module: &'static str,
    func: &'static str,
    src: &'static str,
    /// Waves per block (block = 32 * waves, 4 output rows per wave).
    waves: usize,
}

const fn pipe(label: &'static str, func: &'static str, waves: usize) -> Variant {
    Variant {
        label,
        module: MODULE_PIPE,
        func,
        src: SRC_PIPE,
        waves,
    }
}

/// Index 0 is the incumbent (the bitwise reference); the rest are twins.
const VARIANTS: [Variant; 2] = [
    Variant {
        label: "incumbent",
        module: MODULE_INCUMBENT,
        func: "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4",
        src: SRC_INCUMBENT,
        waves: 4,
    },
    pipe(
        "pipe",
        "gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe_gfx1151",
        4,
    ),
];

const E: usize = 512;
const M: usize = 1280;
const K: usize = 2560;
const X_DIV: usize = 10;
const K_TOP: usize = 10;
const BLOCK_M: usize = 16;
const SLOTS_PER_BLOCK: usize = 8; // R: slots per subtile (grid.y)
const GROUP_BYTES: usize = 136;
const GROUPS_PER_ROW: usize = K / 256;
const ROW_BYTES: usize = GROUPS_PER_ROW * GROUP_BYTES; // 1360
const SEED: u64 = 0x5154_3434_4741_5455; // "QT44GATU"
const WARMUP: usize = 4;
const SAMPLES: usize = 21;
const GUARD_ROWS: usize = 16;
const SENTINEL_BITS: u32 = 0x7fc0_1234;

const UNIFORM_TOKENS: [usize; 19] = [
    9, 17, 33, 65, 97, 129, 145, 177, 209, 241, 273, 305, 337, 369, 401, 433, 465, 497, 529,
];
const SKEWED_TOKENS: [usize; 3] = [145, 273, 401];
const EDGE_TOKENS: usize = 145;
/// The uniform window the per-twin geometric-mean ratio is taken over.
const GEOMEAN_RANGE: std::ops::RangeInclusive<usize> = 145..=497;

// ── RNG ────────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 11) as u128 * n as u128 >> 53) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A finite, normal, signed FP16 with biased exponent in `lo..=hi`.
    fn half_bits(&mut self, lo: u64, hi: u64) -> u16 {
        let r = self.next_u64();
        let sign = (r & 1) as u16;
        let exp = lo + (r >> 1) % (hi - lo + 1);
        let mant = ((r >> 20) & 0x3ff) as u16;
        (sign << 15) | ((exp as u16) << 10) | mant
    }
}

// ── weights ────────────────────────────────────────────────────────────────

struct Weights {
    tensors: Vec<GpuTensor>,
    ptr_t: GpuTensor,
}

impl Weights {
    fn build(gpu: &mut Gpu, seed: u64) -> Weights {
        let mut bytes = vec![0u8; M * ROW_BYTES];
        let mut tensors = Vec::with_capacity(E);
        let mut ptrs = Vec::with_capacity(E * 8);
        for expert in 0..E {
            let mut rng = Rng(seed ^ (expert as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            for chunk in bytes.chunks_exact_mut(GROUP_BYTES) {
                // |scale| in ~[1e-3, 0.13] (biased exp 5..=11), |zero| <= 0.5
                // (biased exp 3..=13): finite sums for K = 2560.
                let hdr = [
                    rng.half_bits(5, 11),
                    rng.half_bits(3, 13),
                    rng.half_bits(5, 11),
                    rng.half_bits(3, 13),
                ];
                for (i, h) in hdr.iter().enumerate() {
                    chunk[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                }
                for w in chunk[8..].chunks_exact_mut(8) {
                    w.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
            }
            let t = gpu
                .upload_raw(&bytes, &[bytes.len()])
                .expect("expert upload");
            ptrs.extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
            tensors.push(t);
        }
        let ptr_t = gpu
            .upload_raw(&ptrs, &[ptrs.len()])
            .expect("ptr table upload");
        Weights { tensors, ptr_t }
    }
    fn free(self, gpu: &mut Gpu) {
        for t in self.tensors {
            gpu.free_tensor(t).expect("free expert");
        }
        gpu.free_tensor(self.ptr_t).expect("free ptr table");
    }
}

// ── routing ────────────────────────────────────────────────────────────────

/// Uniform top-10 WITHOUT replacement per token (partial Fisher-Yates).
fn route_uniform(rng: &mut Rng, e: usize, tokens: usize) -> Vec<i32> {
    let mut perm: Vec<i32> = (0..e as i32).collect();
    let mut out = Vec::with_capacity(tokens * K_TOP);
    for _ in 0..tokens {
        for j in 0..K_TOP {
            let r = j + rng.below(e - j);
            perm.swap(j, r);
            out.push(perm[j]);
        }
    }
    out
}

/// Skewed top-10 WITHOUT replacement: Zipf(1.1) weights over a random expert
/// ranking, sequential weighted draws with removal.
fn route_skewed(rng: &mut Rng, e: usize, tokens: usize) -> Vec<i32> {
    let mut order: Vec<i32> = (0..e as i32).collect();
    for i in (1..e).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let base: Vec<f64> = (0..e).map(|r| 1.0 / ((r + 1) as f64).powf(1.1)).collect();
    let mut out = Vec::with_capacity(tokens * K_TOP);
    for _ in 0..tokens {
        let mut w = base.clone();
        for _ in 0..K_TOP {
            let total: f64 = w.iter().sum();
            let mut u = rng.unit() * total;
            let mut pick = e - 1;
            for (r, &x) in w.iter().enumerate() {
                if x > 0.0 && u < x {
                    pick = r;
                    break;
                }
                u -= x;
            }
            if w[pick] == 0.0 {
                pick = w.iter().position(|&x| x > 0.0).expect("weights exhausted");
            }
            w[pick] = 0.0;
            out.push(order[pick]);
        }
    }
    out
}

// ── cases ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Routing {
    Uniform,
    Skewed,
}

struct Spec {
    name: String,
    tokens: usize,
    routing: Routing,
    /// Seed salt: a case is seeded by (kind, tokens), never by list position.
    salt: u64,
    /// Pass `grouped_rows = live_padded - 5` (partial final subtile/tile).
    partial_grouped: bool,
    /// Inject invalid expert ids into the router output (scatter must skip them).
    invalid_topk: bool,
    /// Corrupt the scatter output afterwards: sentinel tiles and -1 slots.
    mutate: bool,
    /// Plain uniform case (counts toward the geometric mean when in range).
    uniform: bool,
}

fn plain(name: String, tokens: usize, routing: Routing) -> Spec {
    Spec {
        name,
        tokens,
        routing,
        salt: if routing == Routing::Uniform {
            0x11
        } else {
            0x22
        },
        partial_grouped: false,
        invalid_topk: false,
        mutate: false,
        uniform: routing == Routing::Uniform,
    }
}

fn specs(tokens_override: Option<&[usize]>) -> Vec<Spec> {
    if let Some(toks) = tokens_override {
        return toks
            .iter()
            .map(|&t| plain(format!("u{t}"), t, Routing::Uniform))
            .collect();
    }
    let mut v: Vec<Spec> = UNIFORM_TOKENS
        .iter()
        .map(|&t| plain(format!("u{t}"), t, Routing::Uniform))
        .collect();
    v.extend(
        SKEWED_TOKENS
            .iter()
            .map(|&t| plain(format!("skew{t}"), t, Routing::Skewed)),
    );
    v.push(Spec {
        partial_grouped: true,
        salt: 0x33,
        uniform: false,
        ..plain(
            format!("partial{EDGE_TOKENS}"),
            EDGE_TOKENS,
            Routing::Uniform,
        )
    });
    v.push(Spec {
        invalid_topk: true,
        salt: 0x44,
        uniform: false,
        ..plain(
            format!("invalid{EDGE_TOKENS}"),
            EDGE_TOKENS,
            Routing::Uniform,
        )
    });
    v.push(Spec {
        mutate: true,
        salt: 0x55,
        uniform: false,
        ..plain(
            format!("mutate{EDGE_TOKENS}"),
            EDGE_TOKENS,
            Routing::Uniform,
        )
    });
    v
}

// ── host helpers ───────────────────────────────────────────────────────────

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zeros_raw(gpu: &Gpu, bytes: usize) -> GpuTensor {
    gpu.upload_raw(&vec![0u8; bytes], &[bytes])
        .expect("zero upload")
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid byte patterns; length in bytes is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn sha_prefix(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Structural check of the production scatter against the router output.
fn validate_scatter(
    topk: &[i32],
    offsets: &[i32],
    sorted: &[i32],
    tiles: &[i32],
    counts_dev: &[i32],
) -> usize {
    let mut flats: Vec<Vec<i32>> = vec![Vec::new(); E];
    for (flat, &x) in topk.iter().enumerate() {
        if x >= 0 && (x as usize) < E {
            flats[x as usize].push(flat as i32);
        }
    }
    let mut off = 0usize;
    for ex in 0..E {
        let padded = flats[ex].len().next_multiple_of(BLOCK_M);
        assert_eq!(offsets[ex] as usize, off, "scatter offset of expert {ex}");
        assert_eq!(
            counts_dev[ex] as usize, padded,
            "scatter padded count of expert {ex}"
        );
        let region = &sorted[off..off + padded];
        let mut got: Vec<i32> = region.iter().copied().filter(|&s| s >= 0).collect();
        assert!(region.iter().all(|&s| s >= -1), "bad slot value");
        got.sort_unstable();
        assert_eq!(got, flats[ex], "scatter slots of expert {ex}");
        for t in off / BLOCK_M..(off + padded) / BLOCK_M {
            assert_eq!(tiles[t], ex as i32, "tile {t} expert");
        }
        off += padded;
    }
    assert_eq!(offsets[E] as usize, off, "scatter total");
    assert!(
        sorted[off..].iter().all(|&s| s == -1),
        "slot tail must be -1"
    );
    assert!(
        tiles[off / BLOCK_M..].iter().all(|&t| t == -1),
        "tile tail must be -1"
    );
    off
}

struct Traffic {
    touched_experts: usize,
    unique_x_rows: usize,
    live_slots: usize,
    w_bytes: u64,
    x_bytes: u64,
    y_bytes: u64,
}

/// Mirrors the kernel's own validity predicates (tile id >= 0, slot < gr, flat >= 0).
fn traffic(sorted: &[i32], tiles: &[i32], gr: usize, x_rows: usize) -> Traffic {
    let mut touched = vec![false; E];
    let mut seen_x = vec![false; x_rows];
    let mut live_slots = 0usize;
    for t in 0..gr.div_ceil(BLOCK_M) {
        let ex = tiles[t];
        if ex < 0 || ex as usize >= E {
            continue;
        }
        for slot in 0..BLOCK_M {
            let gs = t * BLOCK_M + slot;
            if gs >= gr {
                continue;
            }
            let flat = sorted[gs];
            if flat < 0 {
                continue;
            }
            touched[ex as usize] = true;
            seen_x[flat as usize / X_DIV] = true;
            live_slots += 1;
        }
    }
    let touched_experts = touched.iter().filter(|&&b| b).count();
    let unique_x_rows = seen_x.iter().filter(|&&b| b).count();
    Traffic {
        touched_experts,
        unique_x_rows,
        live_slots,
        w_bytes: (touched_experts * M * ROW_BYTES) as u64,
        x_bytes: (unique_x_rows * K * 4) as u64,
        y_bytes: (gr * M * 4) as u64,
    }
}

// ── one case ───────────────────────────────────────────────────────────────

struct CaseOutcome {
    name: String,
    tokens: usize,
    uniform: bool,
    live_tiles: usize,
    ok: bool,
    /// Indexed by VARIANTS index; NaN for variants that were not run.
    median_us: Vec<f64>,
}

fn run_case(gpu: &mut Gpu, spec: &Spec, w: &Weights, active: &[usize]) -> CaseOutcome {
    assert_eq!(
        active.first(),
        Some(&0),
        "incumbent must be the reference variant"
    );
    let mut rng = Rng(SEED ^ ((spec.tokens as u64) << 32) ^ (spec.salt << 8) ^ 0xA5A5);
    let slots = spec.tokens * K_TOP;
    let prod_gr = (slots + 15 * slots.min(E)).next_multiple_of(BLOCK_M);

    // Routing.
    let mut topk = match spec.routing {
        Routing::Uniform => route_uniform(&mut rng, E, spec.tokens),
        Routing::Skewed => route_skewed(&mut rng, E, spec.tokens),
    };
    if spec.invalid_topk {
        for (i, bad) in [
            (3usize, -1i32),
            (17, E as i32),
            (40, -9),
            (41, E as i32 + 100),
        ] {
            if i < topk.len() {
                topk[i] = bad;
            }
        }
    }

    // Production scatter.
    let topk_t = gpu
        .upload_raw(&i32_bytes(&topk), &[slots * 4])
        .expect("topk");
    let counts_t = zeros_raw(gpu, E * 4);
    let offsets_t = zeros_raw(gpu, (E + 1) * 4);
    let sorted_t = zeros_raw(gpu, prod_gr * 4);
    let tiles_t = zeros_raw(gpu, prod_gr / BLOCK_M * 4);
    let inverse_t = zeros_raw(gpu, slots * 4);
    gpu.moe_scatter_fused_top10(
        &topk_t, &counts_t, &offsets_t, &sorted_t, &tiles_t, &inverse_t, slots, E, prod_gr, BLOCK_M,
    )
    .expect("production scatter");
    gpu.hip.device_synchronize().expect("scatter sync");
    let offsets = i32s(&gpu.download_raw_bytes(&offsets_t).expect("offsets"));
    let counts = i32s(&gpu.download_raw_bytes(&counts_t).expect("counts"));
    let mut sorted = i32s(&gpu.download_raw_bytes(&sorted_t).expect("sorted"));
    let mut tiles = i32s(&gpu.download_raw_bytes(&tiles_t).expect("tiles"));
    let live_padded = validate_scatter(&topk, &offsets, &sorted, &tiles, &counts);

    // Optional corruption: sentinel tiles and dead slots inside live tiles.
    // Only values the kernel legitimately handles: negative tile ids and -1
    // slots (there is no X bounds check, so no out-of-range slots/ids).
    if spec.mutate {
        let live_tiles = live_padded / BLOCK_M;
        assert!(live_tiles >= 10, "mutation case needs >= 10 live tiles");
        tiles[1] = -1; // sentinel tile that still has live slots
        tiles[3] = -7;
        for (t, slot) in [(4usize, 3usize), (5, 0), (5, 9), (0, 2), (2, 7), (8, 15)] {
            sorted[t * BLOCK_M + slot] = -1;
        }
        // A live tile whose slots are all dead (subtile without a live slot).
        for slot in 0..BLOCK_M {
            sorted[6 * BLOCK_M + slot] = -1;
        }
        // A live tile whose first subtile is dead and second is live.
        for slot in 0..SLOTS_PER_BLOCK {
            sorted[7 * BLOCK_M + slot] = -1;
        }
        gpu.hip
            .memcpy_htod(&sorted_t.buf, &i32_bytes(&sorted))
            .expect("mutate sorted");
        gpu.hip
            .memcpy_htod(&tiles_t.buf, &i32_bytes(&tiles))
            .expect("mutate tiles");
    }
    let gr = if spec.partial_grouped && live_padded > 2 * BLOCK_M {
        live_padded - 5
    } else {
        prod_gr
    };
    assert!(gr > 0 && gr <= prod_gr);

    // X: [tokens, K]; X row = flat / 10 = token.
    let x_rows = spec.tokens;
    let xs: Vec<f32> = (0..x_rows * K)
        .map(|_| (rng.unit() * 4.0 - 2.0) as f32)
        .collect();
    let x_t = gpu.upload_f32(&xs, &[xs.len()]).expect("x upload");

    // Y with a guard band; every variant starts from the NaN-payload sentinel.
    let y_floats = (gr + GUARD_ROWS) * M;
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); y_floats];
    let y_t = gpu.upload_f32(&sentinel, &[y_floats]).expect("y upload");

    let tr = traffic(&sorted, &tiles, gr, x_rows);
    let total_bytes = tr.w_bytes + tr.x_bytes + tr.y_bytes;
    let tiles_launched = gr.div_ceil(BLOCK_M);
    println!(
        "\n=== {} ===\n  shape E={E} M={M} K={K} tokens={} slots={slots} routing={} x_row_div={X_DIV}\n  \
         grouped_rows(arg)={gr} (production bound {prod_gr}, live padded {live_padded}, live tiles {}, \
         launched tiles {tiles_launched})\n  touched experts={} live slots={} unique X rows={} | \
         bytes: W={:.2} MB X={:.2} MB Y={:.2} MB total={:.2} MB",
        spec.name,
        spec.tokens,
        if spec.routing == Routing::Skewed { "skewed-zipf1.1" } else { "uniform" },
        live_padded / BLOCK_M,
        tr.touched_experts,
        tr.live_slots,
        tr.unique_x_rows,
        tr.w_bytes as f64 / 1e6,
        tr.x_bytes as f64 / 1e6,
        tr.y_bytes as f64 / 1e6,
        total_bytes as f64 / 1e6,
    );

    // Kernarg blobs (production argument order and types, ten args: n_experts last).
    let blob = || -> Vec<u8> {
        let mut b = KernargBlob::new();
        b.push_u64(w.ptr_t.buf.as_ptr() as u64);
        b.push_u64(tiles_t.buf.as_ptr() as u64);
        b.push_u64(sorted_t.buf.as_ptr() as u64);
        b.push_u64(x_t.buf.as_ptr() as u64);
        b.push_u64(y_t.buf.as_ptr() as u64);
        b.push_i32(M as i32);
        b.push_i32(K as i32);
        b.push_i32(X_DIV as i32);
        b.push_i32(gr as i32);
        b.push_i32((w.ptr_t.buf.size() / 8) as i32);
        b.into_vec()
    };
    let mut blobs: Vec<Vec<u8>> = VARIANTS.iter().map(|_| blob()).collect();
    let grid = |v: &Variant| {
        [
            M.div_ceil(4 * v.waves) as u32,
            gr.div_ceil(SLOTS_PER_BLOCK) as u32,
            1u32,
        ]
    };
    let block = |v: &Variant| [(32 * v.waves) as u32, 1u32, 1u32];

    // Correctness: one fresh-sentinel run per variant, compared on the fly
    // against the incumbent's output (the first one run).
    let main_cells = gr * M;
    let mut ok = true;
    let mut reference: Vec<f32> = Vec::new();
    for (pos, &vi) in active.iter().enumerate() {
        let v = &VARIANTS[vi];
        gpu.hip
            .memcpy_htod(&y_t.buf, f32_bytes(&sentinel))
            .expect("reset y");
        gpu.launch_kernel_blob(v.func, grid(v), block(v), 0, &mut blobs[vi])
            .expect("launch");
        gpu.hip.device_synchronize().expect("sync");
        let out = gpu.download_f32(&y_t).expect("download y");
        let unwritten = out[..main_cells]
            .iter()
            .filter(|x| x.to_bits() == SENTINEL_BITS)
            .count();
        let guard_bad = out[main_cells..]
            .iter()
            .filter(|x| x.to_bits() != SENTINEL_BITS)
            .count();
        let nonfinite = out[..main_cells]
            .iter()
            .filter(|x| !x.is_finite() && x.to_bits() != SENTINEL_BITS)
            .count();
        let nonzero = out[..main_cells]
            .iter()
            .filter(|x| x.is_finite() && x.to_bits() << 1 != 0)
            .count();
        let mut line = format!(
            "  [{:<9}] unwritten={unwritten} guard_clobbered={guard_bad} nonfinite={nonfinite} \
             nonzero_cells={nonzero}",
            v.label
        );
        let mut v_ok = guard_bad == 0 && nonfinite == 0;
        if pos == 0 {
            if tr.live_slots > 0 && nonzero == 0 {
                v_ok = false;
                line.push_str(" ALL-ZERO");
            }
            line.push_str(if v_ok {
                " (reference sane)"
            } else {
                " REFERENCE-BAD"
            });
            reference = out;
        } else {
            let mut count = 0usize;
            let mut first = None;
            for j in 0..out.len() {
                if out[j].to_bits() != reference[j].to_bits() {
                    count += 1;
                    first.get_or_insert(j);
                }
            }
            match first {
                None => line.push_str(" | BITWISE-IDENTICAL to incumbent"),
                Some(j) => {
                    v_ok = false;
                    line.push_str(&format!(
                        " | MISMATCH {count} cells; first at grouped_slot {} row {}: {:#010x} vs \
                         incumbent {:#010x}",
                        j / M,
                        j % M,
                        out[j].to_bits(),
                        reference[j].to_bits()
                    ));
                }
            }
            if !v_ok && guard_bad + nonfinite > 0 {
                line.push_str(" | (guard/nonfinite violation)");
            }
        }
        ok &= v_ok;
        println!("{line}");
    }
    drop(reference);

    // Timing: individually event-timed launches, order cyclically shifted per sample.
    let ev0 = gpu.hip.event_create().expect("event");
    let ev1 = gpu.hip.event_create().expect("event");
    let time_one = |gpu: &mut Gpu, vi: usize, blobs: &mut Vec<Vec<u8>>| -> f64 {
        let v = &VARIANTS[vi];
        gpu.hip.event_record(&ev0, None).expect("rec");
        gpu.launch_kernel_blob(v.func, grid(v), block(v), 0, &mut blobs[vi])
            .expect("launch");
        gpu.hip.event_record(&ev1, None).expect("rec");
        gpu.hip.event_synchronize(&ev1).expect("evsync");
        gpu.hip.event_elapsed_ms(&ev0, &ev1).expect("elapsed") as f64 * 1000.0
    };
    let n = active.len();
    for s in 0..WARMUP {
        for j in 0..n {
            time_one(gpu, active[(s + j) % n], &mut blobs);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); VARIANTS.len()];
    for s in 0..SAMPLES {
        for j in 0..n {
            let vi = active[(s + j) % n];
            let t = time_one(gpu, vi, &mut blobs);
            samples[vi].push(t);
        }
    }
    gpu.hip.event_destroy(ev0).expect("destroy");
    gpu.hip.event_destroy(ev1).expect("destroy");

    let mut median_us = vec![f64::NAN; VARIANTS.len()];
    println!(
        "  timing: {SAMPLES} individual launches/variant after {WARMUP} warmup, order cyclically \
         shifted per sample"
    );
    for &vi in active {
        let min = samples[vi].iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut samples[vi]);
        median_us[vi] = med;
        println!(
            "  {:<9} median {:>9.2} us  min {:>9.2} us  eff {:>7.1} GB/s  ({:.3}x incumbent median)",
            VARIANTS[vi].label,
            med,
            min,
            total_bytes as f64 / 1e9 / (med * 1e-6),
            median_us[0] / med
        );
    }
    println!(
        "  VERDICT {}: {}",
        spec.name,
        if ok { "PASS (bitwise)" } else { "FAIL" }
    );

    for t in [
        topk_t, counts_t, offsets_t, sorted_t, tiles_t, inverse_t, x_t, y_t,
    ] {
        gpu.free_tensor(t).expect("free");
    }
    CaseOutcome {
        name: spec.name.clone(),
        tokens: spec.tokens,
        uniform: spec.uniform,
        live_tiles: live_padded / BLOCK_M,
        ok,
        median_us,
    }
}

fn main() {
    let mut tokens_override: Option<Vec<usize>> = None;
    let mut variant_filter: Option<Vec<String>> = None;
    let mut filter: Option<String> = None;
    // `--precheck`: correctness pre-check on a gfx1100 / gfx1201 card (the same
    // kernels compile there; the incumbent is routed on every gfx11+ SIMT arch):
    // bitwise verdicts are meaningful, timings are not gfx1151 numbers.
    let mut precheck = false;
    for a in std::env::args().skip(1) {
        if let Some(list) = a.strip_prefix("--tokens=") {
            let v: Vec<usize> = list
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("bad --tokens entry {s:?}"))
                })
                .collect();
            assert!(
                !v.is_empty() && v.iter().all(|&t| t > 0),
                "--tokens needs positive counts"
            );
            tokens_override = Some(v);
        } else if let Some(list) = a.strip_prefix("--variants=") {
            variant_filter = Some(
                list.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            );
        } else if a == "--precheck" {
            precheck = true;
        } else if a.starts_with("--") {
            eprintln!("unknown flag {a} (supported: --tokens=a,b,c, --variants=pipe, --precheck)");
            std::process::exit(2);
        } else {
            filter = Some(a);
        }
    }

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch = {}", gpu.arch);
    let gfx1151 = gpu.arch_caps.is_gfx1151() && gpu.arch.starts_with("gfx1151");
    if !gfx1151
        && !(precheck && (gpu.arch.starts_with("gfx1100") || gpu.arch.starts_with("gfx1201")))
    {
        eprintln!(
            "REFUSING: test_moe_gateup_twin_gfx1151 is a gfx1151-only oracle (found {}; \
             --precheck admits a gfx1100/gfx1201 correctness pre-check).",
            gpu.arch
        );
        std::process::exit(2);
    }
    if !gfx1151 {
        println!(
            "NOTE: {} correctness pre-check; timings below are NOT gfx1151 numbers",
            gpu.arch
        );
    }

    // The incumbent (index 0) always runs; twins are substring-matched on the symbol.
    let active: Vec<usize> = (0..VARIANTS.len())
        .filter(|&i| {
            i == 0
                || variant_filter.as_ref().map_or(true, |f| {
                    f.iter().any(|p| VARIANTS[i].func.contains(p.as_str()))
                })
        })
        .collect();
    if active.len() == 1 && variant_filter.is_some() {
        eprintln!("--variants matched no twin symbol");
        std::process::exit(2);
    }

    println!(
        "source sha256[..8]: incumbent {} ({MODULE_INCUMBENT}.gfx1151.hip), pipe {} \
         ({MODULE_PIPE}.gfx1151.hip)",
        sha_prefix(SRC_INCUMBENT),
        sha_prefix(SRC_PIPE)
    );
    // One module for all pipe symbols: each `ensure_kernel_public` call binds one
    // more symbol of the already-compiled/loaded module.
    for &i in &active {
        let v = &VARIANTS[i];
        gpu.ensure_kernel_public(v.module, v.src, v.func)
            .unwrap_or_else(|e| panic!("compile {}: {e:?}", v.func));
        println!(
            "  loaded {:<9} {} (block {}, W={})",
            v.label,
            v.func,
            32 * v.waves,
            v.waves
        );
    }
    println!(
        "bytes denominator: W (unique touched experts, once each) + X (unique referenced rows, \
         once each) + Y (grouped_rows*M*4); GB/s = bytes/1e9/median_s"
    );

    let all_specs = specs(tokens_override.as_deref());
    let mut weights: Option<Weights> = None;
    let mut outcomes = Vec::new();
    for sp in all_specs.iter() {
        if filter.as_deref().is_some_and(|f| !sp.name.contains(f)) {
            continue;
        }
        if weights.is_none() {
            println!(
                "\n[building {E} experts of {M}x{K} (MQ4G256V2, {:.2} MB each)]",
                (M * ROW_BYTES) as f64 / 1e6
            );
            weights = Some(Weights::build(&mut gpu, SEED ^ 0x4D51_3447));
        }
        outcomes.push(run_case(&mut gpu, sp, weights.as_ref().unwrap(), &active));
    }
    if let Some(old) = weights.take() {
        old.free(&mut gpu);
    }

    let mut all_ok = !outcomes.is_empty();
    println!("\n================ SUMMARY (median us; ratio = incumbent median / variant median) ================");
    print!(
        "{:<12} {:>6} {:>6} {:>7} | {:>9}",
        "case", "tokens", "live_t", "verdict", "inc us"
    );
    for &i in active.iter().skip(1) {
        print!(" | {:>9} {:>6}", format!("{} us", VARIANTS[i].label), "x");
    }
    println!();
    for o in &outcomes {
        all_ok &= o.ok;
        print!(
            "{:<12} {:>6} {:>6} {:>7} | {:>9.2}",
            o.name,
            o.tokens,
            o.live_tiles,
            if o.ok { "PASS" } else { "FAIL" },
            o.median_us[0]
        );
        for &i in active.iter().skip(1) {
            print!(
                " | {:>9.2} {:>6.3}",
                o.median_us[i],
                o.median_us[0] / o.median_us[i]
            );
        }
        println!();
    }

    let geo: Vec<&CaseOutcome> = outcomes
        .iter()
        .filter(|o| o.uniform && GEOMEAN_RANGE.contains(&o.tokens))
        .collect();
    if geo.is_empty() {
        println!("\ngeomean ratio: no uniform 145..=497 cases ran");
    } else {
        println!(
            "\ngeometric-mean ratio (incumbent/variant median) over {} uniform 145..=497 cases:",
            geo.len()
        );
        for &i in active.iter().skip(1) {
            let ln: f64 = geo
                .iter()
                .map(|o| (o.median_us[0] / o.median_us[i]).ln())
                .sum::<f64>()
                / geo.len() as f64;
            println!("  {:<9} {:.4}x", VARIANTS[i].label, ln.exp());
        }
    }
    println!(
        "OVERALL: {}",
        if all_ok {
            "PASS (all bitwise identical)"
        } else {
            "FAIL"
        }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
