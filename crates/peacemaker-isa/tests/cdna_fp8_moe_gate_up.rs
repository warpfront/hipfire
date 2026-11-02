// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C for the gfx942 FP8 × FP8 grouped MoE gate/up tile
//! (`gemm_fp8b128_moe_grouped_fp8mfma_pm`): the PeaceMaker kernel is built,
//! emitted as a code object, loaded into the CPU wave64 emulator and launched
//! over several experts and 16-row tiles (partial last tiles, `-1` padding
//! rows, a skipped `-1` tile); the whole `Y` buffer (including the elements
//! the kernel must not touch) is compared bitwise with
//! `fp8_moe_gate_up::reference` under the same `AccumModel` the emulator
//! runs. Weight codes include many `0x80` (OCP −0); activation codes are every
//! FNUZ code except `0x80` (NaN).
//!
//! `reference` itself is pinned under `AccumModel::Exact` to a formula
//! written out here in `f64` (OCP / FNUZ values from their definitions, each
//! MFMA = one rounding of the exact sum).
//!
//! The `0x80 → 0x00` weight remap formula is checked against a per-byte
//! specification for every byte value at every position and every adjacent
//! byte pair; one (expert, activation row) pair is shown to give a bitwise
//! identical `Y` row at any tile position, occupancy and neighbourhood.
//!
//! The HIP twin is compiled offline for gfx942 (ROCm oracle,
//! `HIPFIRE_TEST_REQUIRE_ROCM=0` opts out); no GPU is used.
use peacemaker_cdna::sim::mfma::AccumModel;
use peacemaker_cdna::sim::{self, Config, KernelImage, Launch, Memory};
use peacemaker_isa::cdna::emit::{code_object, metadata_yaml, CodeObjectVersion};
use peacemaker_isa::kernels::cdna::fp8_moe_gate_up::{
    build, expert_stride, grid, reference, Problem, Shape, COLS_PER_WG, HIP_SYMBOL, KERNARG_BYTES,
    KERNARG_OFFSETS, LDS_BUF_BYTES, SYMBOL, TILE_ROWS, WORKGROUP,
};
use peacemaker_isa::Arch;
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const W_ADDR: u64 = 0x1_0000_0000;
const TD_ADDR: u64 = 0x2000_0000;
const SORTED_ADDR: u64 = 0x2100_0000;
const X_ADDR: u64 = 0x3000_0000;
const XS_ADDR: u64 = 0x3800_0000;
const Y_ADDR: u64 = 0x4000_0000;
const KARG_ADDR: u64 = 0x100;
/// `Y` poison: elements the kernel must not write keep it.
const Y_POISON: u32 = 0xeeee_eeee;

/// The production gate/up shape the kernel is built for. The emitted kernel reads `R`/`K` from its kernargs at
/// run time, so every launch below (any `R`, `K`) runs the one image built for this shape.
const SHAPE: Shape = Shape { r: 1280, k: 2560 };

static IMAGE: LazyLock<KernelImage> = LazyLock::new(|| {
    let kernel = build(Arch::Gfx942, SHAPE)
        .unwrap_or_else(|e| panic!("build(gfx942, R={} K={}): {e}", SHAPE.r, SHAPE.k));
    let elf = code_object(&[&kernel], CodeObjectVersion::V5)
        .unwrap_or_else(|e| panic!("code_object: {e}"));
    sim::load_code_object(&elf, SYMBOL)
        .unwrap_or_else(|e| panic!("load_code_object({SYMBOL}): {e}"))
});

// ----------------------------------------------------------------------------
// Inputs

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Case {
    r: usize,
    k: usize,
    x_row_div: usize,
    w: Vec<u8>,
    stride: usize,
    tile_desc: Vec<[i32; 4]>,
    sorted_rows: Vec<i32>,
    x: Vec<u8>,
    x_scale: Vec<f32>,
    n_experts: usize,
}

impl Case {
    fn problem(&self) -> Problem<'_> {
        Problem {
            w: &self.w,
            expert_stride: self.stride,
            tile_desc: &self.tile_desc,
            sorted_rows: &self.sorted_rows,
            x: &self.x,
            x_scale: &self.x_scale,
            r: self.r,
            k: self.k,
            x_row_div: self.x_row_div,
            n_experts: self.n_experts,
        }
    }
}

/// `runs`: per expert run `(expert, rows, pad16)`: `pad16` pads the run's
/// last tile to 16 rows with `-1` (the stable scatter's layout), otherwise
/// the last tile is partial (`nrows < 16`). `skip_tile` appends a `-1` tile.
fn case(
    seed: u64,
    r: usize,
    k: usize,
    x_row_div: usize,
    n_experts: usize,
    runs: &[(i32, usize, bool)],
    skip_tile: bool,
) -> Case {
    let mut rng = Rng(seed);
    let stride = expert_stride(r, k);
    let mut w = vec![0u8; n_experts * stride];
    let (kb, sr) = (k / 128, r.div_ceil(128));
    for e in 0..n_experts {
        let blk = &mut w[e * stride..(e + 1) * stride];
        for c in &mut blk[..r * k] {
            *c = loop {
                let b = if rng.below(8) == 0 {
                    0x80
                } else {
                    rng.next() as u8
                };
                if b & 0x7f != 0x7f {
                    break b;
                }
            };
        }
        for i in 0..sr * kb {
            // Positive BF16 scales 2^-8 .. 2^7 with a random mantissa.
            let bits = ((119 + rng.below(16) as u16) << 7) | rng.below(128) as u16;
            blk[r * k + 2 * i..r * k + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    // Tokens and flat slots token * x_row_div + rank.
    let total: usize = runs.iter().map(|r| r.1).sum();
    let tokens = total.div_ceil(x_row_div.min(4)).max(1) + 1;
    let mut tile_desc = Vec::new();
    let mut sorted_rows = Vec::new();
    for &(expert, rows, pad16) in runs {
        let mut left = rows;
        while left > 0 {
            let n = left.min(16);
            tile_desc.push([
                expert,
                sorted_rows.len() as i32,
                if pad16 { 16 } else { n as i32 },
                0,
            ]);
            for _ in 0..n {
                let tok = rng.below(tokens as u64) as usize;
                let rank = rng.below(x_row_div as u64) as usize;
                sorted_rows.push((tok * x_row_div + rank) as i32);
            }
            if pad16 {
                sorted_rows.extend(std::iter::repeat_n(-1, 16 - n))
            }
            left -= n;
        }
    }
    if skip_tile {
        tile_desc.push([-1, sorted_rows.len() as i32, 16, 0]);
        sorted_rows.extend(std::iter::repeat_n(-1, 16));
    }
    let x = (0..tokens * k)
        .map(|_| loop {
            let b = rng.next() as u8;
            if b != 0x80 {
                break b;
            }
        })
        .collect();
    let x_scale = (0..tokens)
        .map(|_| 2f32.powi(rng.below(10) as i32 - 6))
        .collect();
    Case {
        r,
        k,
        x_row_div,
        w,
        stride,
        tile_desc,
        sorted_rows,
        x,
        x_scale,
        n_experts,
    }
}

fn kernarg(c: &Case) -> Vec<u8> {
    let mut a = Vec::with_capacity(KERNARG_BYTES as usize);
    a.extend(W_ADDR.to_le_bytes());
    a.extend((c.stride as u64).to_le_bytes());
    for p in [TD_ADDR, SORTED_ADDR, X_ADDR, XS_ADDR, Y_ADDR] {
        a.extend(p.to_le_bytes())
    }
    for v in [c.r, c.k, c.x_row_div, c.n_experts] {
        a.extend((v as u32).to_le_bytes())
    }
    assert_eq!(a.len(), KERNARG_BYTES as usize);
    a
}

fn run_sim(c: &Case, accum: AccumModel) -> Vec<u32> {
    let mut mem = Memory::default();
    mem.map_readonly("W", W_ADDR, c.w.clone()).unwrap();
    mem.map_readonly(
        "tile_desc",
        TD_ADDR,
        c.tile_desc
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .unwrap();
    mem.map_readonly(
        "sorted_rows",
        SORTED_ADDR,
        c.sorted_rows.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map_readonly("X", X_ADDR, c.x.clone()).unwrap();
    mem.map_readonly(
        "x_scale",
        XS_ADDR,
        c.x_scale.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map(
        "Y",
        Y_ADDR,
        Y_POISON.to_le_bytes().repeat(c.sorted_rows.len() * c.r),
    )
    .unwrap();
    let launch = Launch {
        grid: grid(c.r as u32, c.tile_desc.len() as u32),
        block: [WORKGROUP, 1, 1],
        kernarg: kernarg(c),
        kernarg_addr: KARG_ADDR,
        dynamic_lds: 0,
    };
    let cfg = Config {
        accum,
        ..Config::default()
    };
    sim::run(&IMAGE, &launch, &mut mem, &cfg)
        .unwrap_or_else(|e| panic!("sim (R={} K={}): {e}", c.r, c.k));
    mem.region("Y")
        .unwrap()
        .bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn reference_bits(c: &Case, accum: AccumModel) -> Vec<u32> {
    let mut y = vec![f32::from_bits(Y_POISON); c.sorted_rows.len() * c.r];
    reference(&c.problem(), accum, &mut y).unwrap_or_else(|e| panic!("reference: {e}"));
    y.iter().map(|v| v.to_bits()).collect()
}

fn tier_c(c: &Case, accum: AccumModel) {
    assert!(c.w.contains(&0x80), "the case must exercise weight 0x80");
    let got = run_sim(c, accum);
    let want = reference_bits(c, accum);
    let written = want.iter().filter(|&&b| b != Y_POISON).count();
    let live_rows: usize = c
        .tile_desc
        .iter()
        .filter(|t| t[0] >= 0 && (t[0] as usize) < c.n_experts)
        .map(|t| t[2] as usize)
        .sum();
    assert_eq!(written, live_rows * c.r, "reference writes every live row");
    let bad: Vec<_> = got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .take(8)
        .map(|(i, (g, w))| format!("Y[{}][{}] = {g:#010x}, want {w:#010x}", i / c.r, i % c.r))
        .collect();
    assert!(
        bad.is_empty(),
        "{accum:?} R={} K={}: {} mismatches, first: {bad:#?}",
        c.r,
        c.k,
        got.iter().zip(&want).filter(|(g, w)| g != w).count()
    );
}

// ----------------------------------------------------------------------------
// Tier C

/// K = 256, three experts (expert 1 unused), partial and padded tiles, a skipped tile.
#[test]
fn tier_c_k256_gate_up() {
    let c = case(
        0x9e37_79b9_7f4a_7c15,
        256,
        256,
        10,
        3,
        &[(0, 37, false), (2, 21, true), (0, 5, false)],
        true,
    );
    tier_c(&c, Config::default().accum);
}

/// Same launch under the other accumulation models: the reference tracks the model the emulator runs.
#[test]
fn tier_c_k256_accum_models() {
    let c = case(
        0x51ed_270b_2a3c_9d41,
        128,
        256,
        10,
        2,
        &[(1, 19, false), (0, 16, true)],
        false,
    );
    for accum in [
        AccumModel::Exact,
        AccumModel::Serial,
        AccumModel::Grouped(4),
    ] {
        tier_c(&c, accum)
    }
}

/// Full gate/up K = 2560 over two experts and several tiles (narrow R to bound runtime).
#[test]
fn tier_c_k2560() {
    let c = case(
        0x2545_f491_4f6c_dd1d,
        128,
        2560,
        10,
        2,
        &[(1, 23, false), (0, 9, true)],
        true,
    );
    tier_c(&c, Config::default().accum);
}

/// The production gate/up shape R = 1280, K = 2560, one partial tile.
#[test]
fn tier_c_full_gate_up_shape() {
    let c = case(
        0xd1b5_4a32_d192_ed03,
        1280,
        2560,
        10,
        1,
        &[(0, 7, false)],
        false,
    );
    tier_c(&c, Config::default().accum);
}

/// Down-style launch: x_row_div = 1, K = 640.
#[test]
fn tier_c_down_style() {
    let c = case(
        0x8cb9_2ba7_2f3d_8dd7,
        64,
        640,
        1,
        2,
        &[(1, 17, true), (0, 3, false)],
        true,
    );
    tier_c(&c, Config::default().accum);
}

/// Fail closed: a tile whose expert id is outside `[0, n_experts)` (`n_experts`, `i32::MAX`: a non-finite
/// router row's id, never a valid index) is skipped before `W + expert * expert_stride` is formed, like the
/// `-1` sentinel: its `Y` rows keep the poison, the valid tiles around it match `reference` bitwise, and the
/// emulator (which maps only `n_experts * stride` bytes of `W`) never faults.
#[test]
fn out_of_range_expert_tiles_are_skipped() {
    let base = case(
        0x4f1b_bcdc_bcc1_2a77,
        128,
        256,
        10,
        2,
        &[(0, 16, true), (1, 16, true), (0, 16, true), (1, 16, true)],
        false,
    );
    let mut c = Case {
        tile_desc: base.tile_desc.clone(),
        sorted_rows: base.sorted_rows.clone(),
        w: base.w.clone(),
        x: base.x.clone(),
        x_scale: base.x_scale.clone(),
        ..base
    };
    c.tile_desc[1][0] = c.n_experts as i32;
    c.tile_desc[3][0] = i32::MAX;
    let got = run_sim(&c, Config::default().accum);
    let want = reference_bits(&c, Config::default().accum);
    assert_eq!(got, want);
    for t in [1usize, 3] {
        let (row0, rows) = (c.tile_desc[t][1] as usize, c.tile_desc[t][2] as usize);
        assert!(
            got[row0 * c.r..(row0 + rows) * c.r]
                .iter()
                .all(|&b| b == Y_POISON),
            "tile {t} wrote Y"
        );
    }
    for t in [0usize, 2] {
        let (row0, rows) = (c.tile_desc[t][1] as usize, c.tile_desc[t][2] as usize);
        assert!(
            got[row0 * c.r..(row0 + rows) * c.r]
                .iter()
                .all(|&b| b != Y_POISON),
            "valid tile {t} unwritten"
        );
    }
}

// ----------------------------------------------------------------------------
// M-placement invariance

/// `base`'s weights and activations under another tile layout. `tiles`: `(expert, rows)` per tile; `rows`
/// (1..=16 entries, `-1` = padding) become `nrows = rows.len()`, an `expert < 0` tile is skipped.
fn relayout(base: &Case, tiles: &[(i32, Vec<i32>)]) -> Case {
    let (mut tile_desc, mut sorted_rows) = (Vec::new(), Vec::new());
    for (expert, rows) in tiles {
        assert!(
            (1..=16).contains(&rows.len()),
            "tile of {} rows",
            rows.len()
        );
        tile_desc.push([*expert, sorted_rows.len() as i32, rows.len() as i32, 0]);
        sorted_rows.extend_from_slice(rows);
    }
    Case {
        r: base.r,
        k: base.k,
        x_row_div: base.x_row_div,
        w: base.w.clone(),
        stride: base.stride,
        tile_desc,
        sorted_rows,
        x: base.x.clone(),
        x_scale: base.x_scale.clone(),
        n_experts: base.n_experts,
    }
}

fn put(mut rows: Vec<i32>, pos: usize, slot: i32) -> Vec<i32> {
    rows[pos] = slot;
    rows
}

/// The result for one (expert, activation row) pair does not depend on where the row sits in its tile, on the
/// tile's position in the grid, on the tile's occupancy (partial / `-1` padded / full), on the neighbouring rows
/// (other tokens, the same row repeated, other experts' tiles, a skipped tile), nor on which flat slot (rank)
/// of the token selects the row: its `Y` row is bitwise one value, and every launch matches `reference`.
#[test]
fn row_result_is_independent_of_tile_placement() {
    let base = case(
        0x6d2b_79f5_a1c3_e847,
        128,
        256,
        10,
        2,
        &[(0, 40, false)],
        false,
    );
    let (div, x_rows, r) = (base.x_row_div, base.x_scale.len(), base.r);
    let token = 3usize;
    assert!(x_rows > token);
    // Two flat slots (ranks 7 and 0) of the same token select the same X row.
    let (t_a, t_b) = ((token * div + 7) as i32, (token * div) as i32);
    let mut rng = Rng(0x2f0b_9a41_77d5_c3e1);
    let mut nb = |n: usize| -> Vec<i32> {
        (0..n)
            .map(|_| {
                (rng.below(x_rows as u64) as usize * div + rng.below(div as u64) as usize) as i32
            })
            .collect()
    };
    let pad16 = |mut v: Vec<i32>| {
        v.resize(16, -1);
        v
    };
    let mut goldens: Vec<Vec<u32>> = Vec::new();
    for e in 0..2i32 {
        let o = 1 - e;
        // (tiles, intended (tile, position) of the target row)
        let layouts: Vec<(Vec<(i32, Vec<i32>)>, (usize, usize))> = vec![
            (vec![(e, vec![t_a])], (0, 0)),
            (vec![(e, put(nb(16), 15, t_a))], (0, 15)),
            (vec![(e, pad16(put(nb(8), 7, t_b)))], (0, 7)),
            (
                vec![(o, nb(16)), (e, nb(16)), (e, put(nb(9), 3, t_b))],
                (2, 3),
            ),
            (
                vec![(-1, vec![-1; 16]), (o, nb(5)), (e, put(nb(16), 0, t_a))],
                (2, 0),
            ),
            (vec![(e, vec![t_a; 16])], (0, 9)),
            (vec![(o, nb(2)), (e, put(nb(2), 1, t_b))], (1, 1)),
        ];
        let mut golden: Option<Vec<u32>> = None;
        for (li, (tiles, (ti, pos))) in layouts.iter().enumerate() {
            assert!(
                tiles[*ti].0 == e && (tiles[*ti].1[*pos] == t_a || tiles[*ti].1[*pos] == t_b),
                "layout {li}: bad target"
            );
            let c = relayout(&base, tiles);
            let got = run_sim(&c, Config::default().accum);
            let want = reference_bits(&c, Config::default().accum);
            if let Some(i) = got.iter().zip(&want).position(|(g, w)| g != w) {
                panic!(
                    "expert {e} layout {li}: Y[{}][{}] = {:#010x}, reference {:#010x}",
                    i / r,
                    i % r,
                    got[i],
                    want[i]
                );
            }
            // Every row of the target token under expert `e`, wherever it sits.
            let mut found = 0;
            for (t, (ex, rows)) in tiles.iter().enumerate() {
                for (p, &f) in rows.iter().enumerate() {
                    if *ex != e || f < 0 || f as usize / div != token {
                        continue;
                    }
                    let i = c.tile_desc[t][1] as usize + p;
                    let row = &got[i * r..(i + 1) * r];
                    if golden.is_none() {
                        golden = Some(row.to_vec());
                    } else {
                        assert!(golden.as_deref() == Some(row), "expert {e} layout {li}: row at tile {t} position {p} differs from the first placement");
                    }
                    found += 1;
                }
            }
            assert!(found >= 1, "layout {li}: no target row found");
        }
        let golden = golden.unwrap();
        assert!(
            golden.iter().any(|&b| b != 0 && b != Y_POISON),
            "expert {e}: degenerate (all-zero) target row"
        );
        goldens.push(golden);
    }
    assert!(
        goldens[0] != goldens[1],
        "the two experts must give different rows for the same activation"
    );
}

// ----------------------------------------------------------------------------
// Reference pinned to an independent f64 formula (AccumModel::Exact)

fn ocp_f64(b: u8) -> f64 {
    let (e, m) = (i32::from(b >> 3 & 15), f64::from(b & 7));
    let v = if e == 0 {
        m / 512.0
    } else {
        (8.0 + m) / 8.0 * 2f64.powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}
fn fnuz_f64(b: u8) -> f64 {
    let (e, m) = (i32::from(b >> 3 & 15), f64::from(b & 7));
    let v = if e == 0 {
        m / 1024.0
    } else {
        (8.0 + m) / 8.0 * 2f64.powi(e - 8)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

#[test]
fn reference_matches_exact_formula() {
    let c = case(
        0x0123_4567_89ab_cdef,
        128,
        384,
        10,
        2,
        &[(0, 18, false), (1, 4, true)],
        true,
    );
    let got = reference_bits(&c, AccumModel::Exact);
    let (r, k, nkb) = (c.r, c.k, c.k / 128);
    for t in c.tile_desc.iter().filter(|t| t[0] >= 0) {
        let blk = &c.w[t[0] as usize * c.stride..];
        for n in 0..t[2] as usize {
            let i = t[1] as usize + n;
            let flat = c.sorted_rows[i];
            for col in 0..r {
                let mut total = 0f32;
                for kb in 0..nkb {
                    let mut partial = 0f32;
                    for step in 0..4 {
                        let mut sum = f64::from(partial);
                        for j in 0..32 {
                            let kk = 128 * kb + 32 * step + j;
                            if flat < 0 {
                                continue;
                            }
                            let xr = flat as usize / c.x_row_div;
                            // OCP weight / 2 (−0 is +0 after the remap; the product sign of zero is irrelevant to the sum).
                            sum += fnuz_f64(c.x[xr * k + kk]) * ocp_f64(blk[col * k + kk]) / 2.0;
                        }
                        partial = sum as f32;
                    }
                    let si = (col / 128) * nkb + kb;
                    let s_w = f32::from_bits(
                        u32::from(u16::from_le_bytes([
                            blk[r * k + 2 * si],
                            blk[r * k + 2 * si + 1],
                        ])) << 16,
                    );
                    let a = if flat < 0 {
                        1.0
                    } else {
                        c.x_scale[flat as usize / c.x_row_div]
                    };
                    total = partial.mul_add((4.0 * a) * s_w, total);
                }
                assert_eq!(got[i * r + col], total.to_bits(), "Y[{i}][{col}]");
            }
        }
    }
}

#[test]
fn reference_refuses_contract_violations() {
    let mut c = case(7, 64, 128, 10, 1, &[(0, 3, false)], false);
    let mut y = vec![0f32; c.sorted_rows.len() * c.r];
    let xr = c.sorted_rows[0] as usize / c.x_row_div;
    let saved = c.x[xr * c.k];
    c.x[xr * c.k] = 0x80;
    assert!(reference(&c.problem(), AccumModel::Exact, &mut y)
        .unwrap_err()
        .contains("0x80"));
    c.x[xr * c.k] = saved;
    c.w[0] = 0x7f;
    assert!(reference(&c.problem(), AccumModel::Exact, &mut y)
        .unwrap_err()
        .contains("NaN"));
    c.w[0] = 0;
    c.x_scale[xr] = 3.0;
    assert!(reference(&c.problem(), AccumModel::Exact, &mut y)
        .unwrap_err()
        .contains("power of two"));
}

// ----------------------------------------------------------------------------
// In-kernel weight remap (`0x80 -> 0x00`)

/// The kernel's per-dword remap, in the emitted instruction order (`v_and_b32`, `v_add_u32`, `v_or_b32`,
/// `v_and_b32`; `0x7f7f7f7f` lives in an SGPR). The formula is the module's; the expectation below is not.
fn remap_dword(w: u32) -> u32 {
    const C: u32 = 0x7f7f_7f7f;
    let d = C & w;
    let d = C
        .checked_add(d)
        .expect("per-byte `(b & 0x7f) + 0x7f <= 0xfe` never carries out of the dword");
    let d = C | d;
    d & w
}

/// Per-byte specification: OCP −0 (`0x80`) becomes `+0`; every other code is unchanged.
fn remap_byte(b: u8) -> u8 {
    if b == 0x80 {
        0
    } else {
        b
    }
}

fn remap_spec(w: u32) -> u32 {
    u32::from_le_bytes(w.to_le_bytes().map(remap_byte))
}

#[test]
fn weight_remap_formula_matches_per_byte_spec() {
    let mut rng = Rng(0xa076_1d64_78bd_642f);
    let check = |w: u32| assert_eq!(remap_dword(w), remap_spec(w), "w = {w:#010x}");
    // Every byte value at every byte position, in random and in boundary contexts.
    let contexts: Vec<u32> = [
        0,
        u32::MAX,
        0x8080_8080,
        0x7f7f_7f7f,
        0x0101_0101,
        0x8000_8000,
        0x0080_0080,
        0xff80_80ff,
        0x807f_7f80,
    ]
    .into_iter()
    .chain((0..64).map(|_| rng.next() as u32))
    .collect();
    for pos in 0..4 {
        for v in 0..=255u32 {
            for &ctx in &contexts {
                check(ctx & !(0xff << (8 * pos)) | v << (8 * pos))
            }
        }
    }
    // Every pair of adjacent bytes (all 65536 values), over random and boundary neighbours.
    for pos in 0..3 {
        for pair in 0..=0xffffu32 {
            for ctx in [
                0,
                u32::MAX,
                0x8080_8080,
                0x7f7f_7f7f,
                rng.next() as u32,
                rng.next() as u32,
            ] {
                check(ctx & !(0xffff << (8 * pos)) | pair << (8 * pos));
            }
        }
    }
    // Random dwords.
    for _ in 0..(1 << 18) {
        check(rng.next() as u32)
    }
    // Per-byte meaning: the remapped code is never FNUZ NaN (0x80) and is the OCP value's FNUZ fold (`fn(b) / 2`).
    for b in 0..=255u8 {
        assert_ne!(remap_byte(b), 0x80);
        if b & 0x7f == 0x7f {
            continue;
        } // OCP NaN, outside the contract
        assert_eq!(fnuz_f64(remap_byte(b)), ocp_f64(b) / 2.0, "code {b:#04x}");
    }
}

// ----------------------------------------------------------------------------
// ABI and HIP twin

fn pm_kernarg_offsets() -> Vec<u32> {
    let kernel = build(Arch::Gfx942, SHAPE).unwrap();
    let yaml = metadata_yaml(&[&kernel], CodeObjectVersion::V5).unwrap();
    yaml.lines()
        .filter_map(|l| {
            l.trim()
                .trim_start_matches("- ")
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse().unwrap())
        })
        .collect()
}

#[test]
fn pm_kernarg_layout() {
    assert_eq!(pm_kernarg_offsets(), KERNARG_OFFSETS);
    let kernel = build(Arch::Gfx942, SHAPE).unwrap();
    assert_eq!(kernel.resources.kernarg_size, KERNARG_BYTES);
    assert_eq!(kernel.resources.group_segment_fixed_size, 4608);
    assert_eq!(
        kernel.resources.group_segment_fixed_size,
        2 * LDS_BUF_BYTES,
        "two double-buffered activation tiles"
    );
    assert_eq!(kernel.resources.workgroup_size, WORKGROUP);
}

/// The kernel reads `R`/`K` at run time: the emitted code object does not depend on the build shape.
#[test]
fn emitted_kernel_is_shape_independent() {
    let elf = |shape: Shape| {
        let kernel = build(Arch::Gfx942, shape).unwrap();
        code_object(&[&kernel], CodeObjectVersion::V5).unwrap()
    };
    let want = elf(SHAPE);
    for (r, k) in [(2560, 640), (128, 256), (64, 128)] {
        assert!(
            want == elf(Shape { r, k }),
            "code object for R={r} K={k} differs from the R={} K={} one",
            SHAPE.r,
            SHAPE.k
        );
    }
}

#[test]
fn launch_geometry_constants() {
    assert_eq!(WORKGROUP, 256, "four wave64");
    assert_eq!(
        WORKGROUP / 64 * 16,
        COLS_PER_WG,
        "each wave owns 16 output columns"
    );
    assert_eq!(TILE_ROWS, 16);
    assert_eq!(
        LDS_BUF_BYTES,
        TILE_ROWS * 144,
        "128 code bytes + 16 B pad per row"
    );
    assert_eq!(grid(1280, 5), [20, 5, 1]);
    assert_eq!(grid(2560, 1), [40, 1, 1]);
}

#[test]
fn hip_twin_compiles_for_gfx942() {
    let objdump = require_rocm_tool!("llvm-objdump");
    let readelf = require_rocm_tool!("llvm-readelf");
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/src/pm_cdna/fp8_moe_gate_up.gfx942.hip");
    let out =
        std::env::temp_dir().join(format!("pm-cdna-fp8-moe-gate-up-{}.co", std::process::id()));
    let status = std::process::Command::new(rocm::hipcc())
        .args([
            "--offload-arch=gfx942",
            "-O3",
            "--cuda-device-only",
            "--no-gpu-bundle-output",
            "-o",
        ])
        .arg(&out)
        .arg(&src)
        .status()
        .expect("running hipcc");
    assert!(status.success(), "hipcc failed for {}", src.display());
    let dis = std::process::Command::new(objdump)
        .args(["-d", "--mcpu=gfx942"])
        .arg(&out)
        .output()
        .unwrap();
    let notes = std::process::Command::new(readelf)
        .arg("--notes")
        .arg(&out)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&out);
    let dis = String::from_utf8_lossy(&dis.stdout);
    let notes = String::from_utf8_lossy(&notes.stdout);
    assert!(
        dis.contains(&format!("<{HIP_SYMBOL}>:")),
        "symbol {HIP_SYMBOL} missing"
    );
    assert_eq!(
        dis.matches("v_mfma_f32_16x16x32_fp8_fp8").count(),
        4,
        "four FP8 MFMAs per 128-K block"
    );
    assert!(
        notes.contains(&format!(".kernarg_segment_size: {KERNARG_BYTES}")),
        "HIP kernarg size"
    );
    assert!(
        notes.contains(".group_segment_fixed_size: 4608"),
        "HIP LDS bytes"
    );
    // HIP declares expert_stride as one u64 at 8; PM as two u32 halves at 8 and 12: same bytes.
    let hip: Vec<u32> = notes
        .lines()
        .filter_map(|l| {
            l.trim()
                .trim_start_matches("- ")
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse().unwrap())
        })
        .collect();
    let pm: Vec<u32> = KERNARG_OFFSETS
        .iter()
        .copied()
        .filter(|&o| o != 12)
        .collect();
    assert_eq!(hip, pm);
}
