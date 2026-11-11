// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sequence-parallel QSA selection exchange (`qsa_sp_pack_blocks` /
//! `qsa_sp_expand_blocks`) against one full-tile batched select.
//!
//! Two ranks are simulated on one GPU. Each rank selects its own contiguous row range
//! (`position_start = P + r0`, `block_count = (P + r0 + m) / compress`), packs its rows,
//! receives the peer's compact rows (a host copy stands in for the all-gather) and expands
//! them. Both ranks' full selection buffers must equal the full-tile select byte for byte,
//! and the mirror of the rank whose range ends at the tile's last row must equal the
//! full-tile mirror.
//!
//!   cargo test -p rdna-compute --release --test qsa_sp_select_hw -- --ignored --nocapture

use rdna_compute::qsa_sp_select::QsaSpBlocks;
use rdna_compute::tensor_ops::{
    indexed_attention_select_batch, indexed_attention_select_batch_mirrored,
    IndexedAttentionSelectBatch,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const HEADS: usize = 4;
const DIM: usize = 128;
const QUERY_STRIDE: usize = 640;
const BUDGET_BLOCKS: usize = 512;
const COMPRESS: usize = 4;
const CAPACITY: usize = BUDGET_BLOCKS * COMPRESS + COMPRESS - 1;
const POOLED_CAPACITY: usize = 10240;
const JUNK: u8 = 0xa5;

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

/// Round-to-nearest-even F32 -> BF16 bits (finite inputs).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn pooled_bits(ties: bool, seed: u32) -> Vec<u16> {
    let mut rng = Lcg(seed);
    let palette: Vec<Vec<u16>> = (0..4)
        .map(|_| (0..DIM).map(|_| bf16_bits(rng.signed())).collect())
        .collect();
    let mut bits = Vec::with_capacity(POOLED_CAPACITY * DIM);
    for b in 0..POOLED_CAPACITY {
        if ties && b % 8 != 0 {
            bits.extend_from_slice(&palette[(b / 8) % 4]);
        } else {
            bits.extend((0..DIM).map(|_| bf16_bits(rng.signed())));
        }
    }
    bits
}

fn query_f32(rows: usize, seed: u32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut q = vec![0.0f32; rows * QUERY_STRIDE];
    for r in 0..rows {
        for v in &mut q[r * QUERY_STRIDE..r * QUERY_STRIDE + HEADS * DIM] {
            *v = rng.signed();
        }
    }
    q
}

/// Balanced split: smallest `s` in `[1, n-1]` with left work >= right work, compared with
/// `s - 1` by `max(left, right)` (tie -> smaller `s`). Work of row `r` is `P + r + 1`.
fn balanced_split(p: usize, n: usize) -> usize {
    let total = n * p + n * (n + 1) / 2;
    let left = |s: usize| s * p + s * (s + 1) / 2;
    let cost = |s: usize| left(s).max(total - left(s));
    let s = (1..n)
        .find(|&s| left(s) >= total - left(s))
        .unwrap_or(n - 1);
    if s > 1 && cost(s - 1) <= cost(s) {
        s - 1
    } else {
        s
    }
}

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::init() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skip: no GPU ({e})");
            None
        }
    }
}

fn raw(gpu: &mut Gpu, bytes: usize) -> GpuTensor {
    gpu.alloc_tensor(&[bytes], DType::Raw).expect("alloc raw")
}

fn fill_junk(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .memcpy_htod(&t.buf, &vec![JUNK; t.buf.size()])
        .expect("junk fill");
}

fn words(bytes: &[u8]) -> Vec<i32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|c| i32::from_le_bytes(*c)).collect()
}

/// Panic at the first differing `(row, slot)` of two `[rows][width]` i32 buffers.
fn assert_rows_equal(label: &str, got: &[u8], want: &[u8], width: usize) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if got == want {
        return;
    }
    let (g, w) = (words(got), words(want));
    let i = g.iter().zip(&w).position(|(a, b)| a != b).unwrap();
    panic!(
        "{label}: first difference at row {} slot {}: got {} want {}",
        i / width,
        i % width,
        g[i],
        w[i]
    );
}

struct Tile<'a> {
    query: &'a GpuTensor,
    pooled: &'a GpuTensor,
    position: usize,
    rows: usize,
}

/// Select tile rows `[r0, r0 + m)` into `selected`'s rows; with `mirror` the call must end
/// at the tile's last row. Returns the mirror bytes when requested (copied from the last
/// selected row if the route did not persist it).
fn select_range(
    gpu: &mut Gpu,
    t: &Tile<'_>,
    selected: &GpuTensor,
    r0: usize,
    m: usize,
    mirror: Option<&GpuTensor>,
) -> Option<Vec<u8>> {
    let query = t.query.sub_offset(r0 * QUERY_STRIDE, m * QUERY_STRIDE);
    let sel = selected.sub_offset(r0 * CAPACITY * 4, m * CAPACITY * 4);
    let position_start = t.position + r0;
    let p = IndexedAttentionSelectBatch {
        query: &query,
        pooled: t.pooled,
        pooled_bf16: None,
        pooled_bf16_valid_blocks: 0,
        selected: &sel,
        rows: m,
        query_row_stride: QUERY_STRIDE,
        block_count: (position_start + m) / COMPRESS,
        index_heads: HEADS,
        index_dim: DIM,
        budget_blocks: BUDGET_BLOCKS,
        compress: COMPRESS,
        position_start,
        capacity: CAPACITY,
        shape_blocks: POOLED_CAPACITY,
    };
    match mirror {
        None => {
            indexed_attention_select_batch(gpu, &p).expect("select");
            None
        }
        Some(mirror) => {
            assert_eq!(r0 + m, t.rows, "mirror needs the tile's last row");
            let persisted =
                indexed_attention_select_batch_mirrored(gpu, &p, mirror).expect("mirrored select");
            gpu.hip.device_synchronize().expect("sync");
            if !persisted {
                let all = gpu.download_raw_bytes(selected).expect("selected download");
                let last = &all[(t.rows - 1) * CAPACITY * 4..t.rows * CAPACITY * 4];
                gpu.hip.memcpy_htod(&mirror.buf, last).expect("mirror copy");
            }
            Some(gpu.download_raw_bytes(mirror).expect("mirror download"))
        }
    }
}

fn run_case(gpu: &mut Gpu, pooled: &GpuTensor, label: &str, position: usize, n: usize, s: usize) {
    let q = query_f32(n, 0x51ed_0001 ^ position as u32 ^ n as u32);
    let query = gpu.upload_f32(&q, &[q.len()]).expect("query upload");
    let t = Tile {
        query: &query,
        pooled,
        position,
        rows: n,
    };
    let sel_bytes = n * CAPACITY * 4;
    let cmp_bytes = n * BUDGET_BLOCKS * 4;

    // Reference: one full-tile mirrored select.
    let ref_sel = raw(gpu, sel_bytes);
    let ref_mirror = raw(gpu, CAPACITY * 4);
    fill_junk(gpu, &ref_sel);
    fill_junk(gpu, &ref_mirror);
    let ref_mirror_bytes =
        select_range(gpu, &t, &ref_sel, 0, n, Some(&ref_mirror)).expect("reference mirror");
    gpu.hip.device_synchronize().expect("sync");
    let ref_bytes = gpu
        .download_raw_bytes(&ref_sel)
        .expect("reference download");

    let ranges = [(0usize, s), (s, n - s)];
    let sel: Vec<GpuTensor> = (0..2).map(|_| raw(gpu, sel_bytes)).collect();
    let cmp: Vec<GpuTensor> = (0..2).map(|_| raw(gpu, cmp_bytes)).collect();
    let mirror1 = raw(gpu, CAPACITY * 4);
    for r in 0..2 {
        fill_junk(gpu, &sel[r]);
        fill_junk(gpu, &cmp[r]);
    }
    fill_junk(gpu, &mirror1);

    // Each rank: select own rows, pack own rows.
    let mut mirror1_bytes = None;
    for r in 0..2 {
        let (r0, m) = ranges[r];
        let mirror = (r == 1).then_some(&mirror1);
        let got = select_range(gpu, &t, &sel[r], r0, m, mirror);
        if r == 1 {
            mirror1_bytes = got;
        }
        let blocks = QsaSpBlocks {
            selected: &sel[r],
            compact: &cmp[r],
            tile_rows: n,
            position_start: position,
            block_count: (position + n) / COMPRESS,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            capacity: CAPACITY,
        };
        gpu.qsa_sp_pack_blocks(&blocks, r0, m).expect("pack");
    }
    gpu.hip.device_synchronize().expect("sync");

    // Stand-in all-gather: each rank's compact buffer receives the peer's own rows.
    let rb = BUDGET_BLOCKS * 4;
    let c0 = gpu.download_raw_bytes(&cmp[0]).expect("compact 0");
    let c1 = gpu.download_raw_bytes(&cmp[1]).expect("compact 1");
    let mut m0 = c0.clone();
    m0[s * rb..].copy_from_slice(&c1[s * rb..]);
    let mut m1 = c1.clone();
    m1[..s * rb].copy_from_slice(&c0[..s * rb]);
    gpu.hip.memcpy_htod(&cmp[0].buf, &m0).expect("gather 0");
    gpu.hip.memcpy_htod(&cmp[1].buf, &m1).expect("gather 1");

    // Each rank expands the peer's rows.
    for r in 0..2 {
        let (p0, pm) = ranges[1 - r];
        let blocks = QsaSpBlocks {
            selected: &sel[r],
            compact: &cmp[r],
            tile_rows: n,
            position_start: position,
            block_count: (position + n) / COMPRESS,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            capacity: CAPACITY,
        };
        gpu.qsa_sp_expand_blocks(&blocks, p0, pm).expect("expand");
    }
    gpu.hip.device_synchronize().expect("sync");

    for (r, rank_sel) in sel.iter().enumerate() {
        let got = gpu.download_raw_bytes(rank_sel).expect("selected download");
        assert_rows_equal(
            &format!("{label} rank{r} selected"),
            &got,
            &ref_bytes,
            CAPACITY,
        );
    }
    assert_rows_equal(
        &format!("{label} rank1 mirror"),
        &mirror1_bytes.expect("rank1 mirror"),
        &ref_mirror_bytes,
        CAPACITY,
    );
    let got0 = gpu.download_raw_bytes(&sel[0]).expect("selected download");
    assert_rows_equal(
        &format!("{label} rank0 last row vs mirror"),
        &got0[(n - 1) * CAPACITY * 4..],
        &ref_mirror_bytes,
        CAPACITY,
    );
    eprintln!("ok: {label} P={position} n={n} split={s}");

    for t in [ref_sel, ref_mirror, mirror1]
        .into_iter()
        .chain(sel)
        .chain(cmp)
    {
        gpu.free_tensor(t).expect("free");
    }
    gpu.free_tensor(query).expect("free query");
}

fn run_cases(ties: bool, cases: &[(usize, usize)], label: &str) {
    let Some(mut gpu) = gpu_or_skip() else {
        return;
    };
    let bits = pooled_bits(ties, 0x9e37_79b9);
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let pooled = gpu
        .alloc_tensor(&[POOLED_CAPACITY * DIM], DType::BF16)
        .expect("alloc pooled");
    gpu.hip
        .memcpy_htod(&pooled.buf, &bytes)
        .expect("pooled upload");
    for &(position, n) in cases {
        let balanced = balanced_split(position, n);
        let even = n.div_ceil(2);
        for (kind, s) in [("balanced", balanced), ("even", even)] {
            run_case(
                &mut gpu,
                &pooled,
                &format!("{label}/{kind}"),
                position,
                n,
                s,
            );
        }
    }
    gpu.free_tensor(pooled).expect("free pooled");
}

#[test]
#[ignore = "needs a GPU"]
fn sp_select_matches_full_tile_random_keys() {
    run_cases(false, &[(0, 4096), (28672, 4096), (32768, 3830)], "random");
}

#[test]
#[ignore = "needs a GPU"]
fn sp_select_matches_full_tile_tied_keys() {
    run_cases(true, &[(8192, 4096)], "ties");
}

#[test]
#[ignore = "needs a GPU"]
fn sp_select_matches_full_tile_short_tiles() {
    // Tails 0..3 and tiles whose budget is not reached (rows see fewer than 512 blocks).
    run_cases(false, &[(0, 7), (0, 2051), (5, 33)], "short");
}
