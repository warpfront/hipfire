// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gate for `moe_route_scatter_top10_f32`
//! (`HIPFIRE_QWEN4_MOE_ROUTE_SCATTER=1`, module `moe_route_scatter_top10_f32`): the E512/top-10
//! router and the grouped scatter in one single-CTA launch, rows 1..=8.
//!
//! For every row count 1..=8, both softmax recipes (normalize x BF16 round) and seeded logits
//! (uniform, heavy ties, all-equal, BF16-grid, shared peaked rows that pile onto the same
//! experts, +-80 spreads, rows with NaN / +Inf / -Inf), the production two-launch chain
//! (`moe_router_softmax_top10_f32` then `moe_scatter_fused_top10`) and the fused launch run into
//! canary-filled buffers; every word of `topk_idx`, `topk_w`, `expert_token_counts`,
//! `expert_offsets`, `sorted_slot_index`, `expert_tile_ids`, `inverse_perm` (whole allocations,
//! trailing canary included) and the router fault word must be identical, with ONE documented
//! exception: the scatter's bucket fill claims each slot's position inside its expert with an LDS
//! `atomicAdd` from every wave, so which slot lands first among slots of the SAME expert is a
//! wave-arrival race in the two-launch chain itself (measured here: the chain differs from the
//! ascending-slot order in ~40% of the cases and from its own rerun in ~0.3%). The fused entry
//! fills each bucket in ascending slot order on one wave (`ORDERED`), every time. So
//! `sorted_slot_index` and `inverse_perm` of the fused launch are compared, strictly, with a host
//! model of that order (a stable counting sort of `topk_idx` over the chain's `expert_offsets`),
//! and with the chain's copies as permutations (equal bucket contents, `sorted[inverse[i]] == i`,
//! same -1 cells); the chain's own deviations from the model are counted and reported, never
//! failed. Every other output is compared bytewise against the chain.
//! Every other output is compared bytewise against the chain. The combine reads
//! slots through `inverse_perm` and every grouped row is computed independently, so the order
//! inside a bucket does not reach the output. Each row of the fused
//! outputs is also compared with the incumbent one-workgroup-per-token router run on that row
//! alone. A real logits dump replaces the seeded rows when `MOE_ROUTE_SCATTER_LOGITS` names a
//! raw little-endian f32 file of `[rows >= 8, 512]`.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test moe_route_scatter_hw -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        let sign = self.next_u32() & 1;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0;
        if sign == 1 {
            -v
        } else {
            v
        }
    }
}

const E: usize = 512;
const K: usize = 10;
const BLOCK_M: usize = 16;
const PAD: usize = 64; // trailing canary words per buffer
const CANARY: u32 = 0x7FC0_DEAD;

fn m_total_bound(slots: usize) -> usize {
    (slots + slots.min(E) * (BLOCK_M - 1)).div_ceil(BLOCK_M) * BLOCK_M
}

fn upload_u32(gpu: &mut Gpu, dtype: DType, data: &[u32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[data.len()], dtype).expect("alloc");
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
    t
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// One set of outputs, every buffer `len + PAD` words of canary.
struct Outputs {
    idx: GpuTensor,
    w: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    lens: [usize; 7],
}
impl Outputs {
    fn new(gpu: &mut Gpu, rows: usize) -> Self {
        let slots = rows * K;
        let m = m_total_bound(slots);
        let lens = [slots, slots, E, E + 1, m, m / BLOCK_M, slots];
        let mk = |gpu: &mut Gpu, n: usize| upload_u32(gpu, DType::F32, &vec![CANARY; n + PAD]);
        Self {
            idx: mk(gpu, lens[0]),
            w: mk(gpu, lens[1]),
            counts: mk(gpu, lens[2]),
            offsets: mk(gpu, lens[3]),
            sorted: mk(gpu, lens[4]),
            tiles: mk(gpu, lens[5]),
            inverse: mk(gpu, lens[6]),
            lens,
        }
    }
    fn dump(&self, gpu: &Gpu) -> Vec<Vec<u32>> {
        let ts = [
            &self.idx,
            &self.w,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.tiles,
            &self.inverse,
        ];
        ts.iter()
            .zip(self.lens)
            .map(|(t, n)| download_u32(gpu, t, n + PAD))
            .collect()
    }
}
const NAMES: [&str; 7] = [
    "topk_idx",
    "topk_w",
    "expert_token_counts",
    "expert_offsets",
    "sorted_slot_index",
    "expert_tile_ids",
    "inverse_perm",
];

fn gen_row(rng: &mut Lcg, mode: u32, shared: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = match mode {
        0 | 4 | 5 | 6 => (0..E).map(|_| rng.f32_signed() * 4.0).collect(),
        1 => (0..E).map(|_| (rng.next_u32() % 6) as f32 * 0.5).collect(),
        2 => vec![1.25; E],
        3 => (0..E)
            .map(|_| f32::from_bits((rng.f32_signed() * 3.0).to_bits() & 0xFFFF_0000))
            .collect(),
        7 => shared.iter().map(|v| v + rng.f32_signed() * 1e-3).collect(),
        8 => (0..E).map(|_| rng.f32_signed() * 80.0).collect(),
        _ => vec![-1.0e30; E],
    };
    let at = (rng.next_u32() as usize) % E;
    match mode {
        4 => r[at] = f32::NAN,
        5 => r[at] = f32::INFINITY,
        6 => r[at] = f32::NEG_INFINITY,
        _ => {}
    }
    r
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_route_scatter_matches_two_launch_chain_hw() {
    // Read once per process by the launcher.
    std::env::set_var("HIPFIRE_QWEN4_MOE_ROUTE_SCATTER", "1");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    gpu.ensure_router_fault_word().expect("fault word");
    let real: Option<Vec<f32>> = std::env::var("MOE_ROUTE_SCATTER_LOGITS").ok().map(|path| {
        let b = std::fs::read(&path).expect("logits file");
        assert!(b.len() >= 8 * E * 4, "logits file needs >= 8 rows");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    });

    // Admission edges: nothing is launched outside 1..=8 rows / E512 / top-10.
    {
        let o = Outputs::new(&mut gpu, 8);
        let logits = upload_u32(&mut gpu, DType::F32, &vec![0u32; 16 * E]);
        for (rows, e, k) in [(0usize, E, K), (9, E, K), (8, 256, K), (8, E, 8)] {
            let launched = gpu
                .moe_route_scatter_top10_f32(
                    &logits,
                    &o.idx,
                    &o.w,
                    rows,
                    true,
                    true,
                    0,
                    0,
                    &o.counts,
                    &o.offsets,
                    &o.sorted,
                    &o.tiles,
                    &o.inverse,
                    e,
                    k,
                    m_total_bound(80),
                    BLOCK_M,
                )
                .expect("decline");
            assert!(!launched, "rows {rows} e {e} k {k} must decline");
        }
    }

    let (mut cases, mut words, mut bad) = (0usize, 0usize, 0usize);
    let mut fault_cases = 0usize;
    let (mut strict_diff, mut chain_self_diff, mut chain_vs_model) =
        ([0usize; 7], [0usize; 7], [0usize; 7]);
    for rows in 1usize..=8 {
        let slots = rows * K;
        let m_total = m_total_bound(slots);
        for seed in 0..48u64 {
            let mut rng = Lcg(0x9E37_0000 + 1000 * rows as u64 + seed);
            let shared: Vec<f32> = (0..E).map(|_| rng.f32_signed() * 3.0).collect();
            // Row modes: seed-dependent mixes; seeds 40.. force one bad row, 44.. several.
            let mut modes: Vec<u32> = (0..rows)
                .map(|r| ((seed as u32 / 3) + r as u32 * 5 + (rng.next_u32() % 3)) % 10)
                .collect();
            if seed >= 40 {
                let bad_rows = if seed >= 44 { 2.min(rows) } else { 1 };
                for b in 0..bad_rows {
                    modes[(rng.next_u32() as usize + b) % rows] = 4 + (seed as u32 + b as u32) % 3;
                }
            }
            let mut logits: Vec<f32> = Vec::with_capacity(rows * E);
            for (r, &mode) in modes.iter().enumerate() {
                if let Some(real) = real.as_ref() {
                    let src = ((r + seed as usize) % (real.len() / E)) * E;
                    logits.extend_from_slice(&real[src..][..E]);
                } else {
                    logits.extend(gen_row(&mut rng, mode, &shared));
                }
            }
            let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
            let logits_t = upload_u32(&mut gpu, DType::F32, &bits);
            let nonfinite_rows: Vec<usize> = (0..rows)
                .filter(|&r| logits[r * E..][..E].iter().any(|v| !v.is_finite()))
                .collect();
            let layer = 3 + (seed as usize % 40);
            let row_base = 17 * (seed as usize % 5);
            for (normalize, round) in [(true, false), (true, true), (false, false), (false, true)] {
                // Reference: the production two-launch chain.
                let want = Outputs::new(&mut gpu, rows);
                gpu.clear_router_fault();
                gpu.moe_router_softmax_top10_f32(
                    &logits_t, &want.idx, &want.w, rows, normalize, round, layer, row_base,
                )
                .expect("router");
                gpu.moe_scatter_fused_top10(
                    &want.idx,
                    &want.counts,
                    &want.offsets,
                    &want.sorted,
                    &want.tiles,
                    &want.inverse,
                    slots,
                    E,
                    m_total,
                    BLOCK_M,
                )
                .expect("scatter");
                gpu.hip.device_synchronize().expect("sync");
                let want_fault = gpu.take_router_fault();
                let want_d = want.dump(&gpu);
                // A second chain run: the within-bucket slot order is an atomic race.
                let want2 = Outputs::new(&mut gpu, rows);
                gpu.moe_router_softmax_top10_f32(
                    &logits_t, &want2.idx, &want2.w, rows, normalize, round, layer, row_base,
                )
                .expect("router2");
                gpu.moe_scatter_fused_top10(
                    &want2.idx,
                    &want2.counts,
                    &want2.offsets,
                    &want2.sorted,
                    &want2.tiles,
                    &want2.inverse,
                    slots,
                    E,
                    m_total,
                    BLOCK_M,
                )
                .expect("scatter2");
                gpu.hip.device_synchronize().expect("sync");
                let _ = gpu.take_router_fault();
                let want2_d = want2.dump(&gpu);

                // Candidate: the fused launch.
                let got = Outputs::new(&mut gpu, rows);
                gpu.clear_router_fault();
                let launched = gpu
                    .moe_route_scatter_top10_f32(
                        &logits_t,
                        &got.idx,
                        &got.w,
                        rows,
                        normalize,
                        round,
                        layer,
                        row_base,
                        &got.counts,
                        &got.offsets,
                        &got.sorted,
                        &got.tiles,
                        &got.inverse,
                        E,
                        K,
                        m_total,
                        BLOCK_M,
                    )
                    .expect("fused");
                assert!(launched, "fused launch declined (flag not read?)");
                gpu.hip.device_synchronize().expect("sync");
                let got_fault = gpu.take_router_fault();
                let got_d = got.dump(&gpu);

                // Host model of the ordered fill (stable counting sort of the chain's ids).
                let (model_sorted, model_inverse) = {
                    let ids = &want_d[0];
                    let offsets = &want_d[3];
                    let mut sorted = got_d[4].clone();
                    let mut inverse = got_d[6].clone();
                    for v in sorted[..m_total].iter_mut() {
                        *v = u32::MAX;
                    }
                    let mut fill = vec![0u32; E];
                    for i in 0..slots {
                        let e = ids[i] as i32;
                        if (0..E as i32).contains(&e) {
                            let pos = offsets[e as usize] + fill[e as usize];
                            fill[e as usize] += 1;
                            sorted[pos as usize] = i as u32;
                            inverse[i] = pos;
                        } else {
                            inverse[i] = u32::MAX;
                        }
                    }
                    (sorted, inverse)
                };
                for (i, name) in NAMES.iter().enumerate() {
                    let model: Option<&Vec<u32>> = match i {
                        4 => Some(&model_sorted),
                        6 => Some(&model_inverse),
                        _ => None,
                    };
                    let reference = model.unwrap_or(&want_d[i]);
                    let n = got_d[i]
                        .iter()
                        .zip(reference)
                        .filter(|(g, w)| g != w)
                        .count();
                    if n != 0 {
                        eprintln!(
                            "MISMATCH rows {rows} seed {seed} norm {normalize} round {round}: {name} {n} words"
                        );
                        bad += n;
                    }
                    if model.is_some() {
                        if want_d[i] != *reference {
                            chain_vs_model[i] += 1;
                        }
                        if want_d[i] != got_d[i] {
                            strict_diff[i] += 1;
                        }
                        if want2_d[i] != want_d[i] {
                            chain_self_diff[i] += 1;
                        }
                    }
                    words += got_d[i].len();
                }
                // The same scatter permutation as the chain's up to the order inside an expert's
                // bucket: equal bucket contents, `sorted[inverse[i]] == i`, the same -1 cells.
                {
                    let (offsets, sorted, inverse) = (&got_d[3], &got_d[4], &got_d[6]);
                    let (wsorted, winverse) = (&want_d[4], &want_d[6]);
                    let mut ok = true;
                    for e in 0..E {
                        let (lo, hi) = (offsets[e] as usize, offsets[e + 1] as usize);
                        let mut x: Vec<u32> = sorted[lo..hi].to_vec();
                        let mut y: Vec<u32> = wsorted[lo..hi].to_vec();
                        x.sort_unstable();
                        y.sort_unstable();
                        ok &= x == y;
                    }
                    for i in 0..slots {
                        ok &= if inverse[i] == u32::MAX {
                            winverse[i] == u32::MAX
                        } else {
                            sorted[inverse[i] as usize] == i as u32
                        };
                    }
                    if !ok {
                        eprintln!("MISMATCH rows {rows} seed {seed}: scatter permutation not equivalent to the chain's");
                        bad += 1;
                    }
                }
                // Router fault word: one bad row is deterministic; with several the last store
                // wins in both launches, so either must name one of the bad rows.
                let key = |f: &Option<rdna_compute::router_fault::RouterFault>| {
                    f.as_ref().map(|f| (f.layer, f.row))
                };
                if nonfinite_rows.len() <= 1 {
                    if key(&got_fault) != key(&want_fault) {
                        eprintln!(
                            "MISMATCH rows {rows} seed {seed}: fault {:?} vs {:?}",
                            key(&got_fault),
                            key(&want_fault)
                        );
                        bad += 1;
                    }
                    assert_eq!(got_fault.is_some(), !nonfinite_rows.is_empty());
                } else {
                    for f in [&got_fault, &want_fault] {
                        let f = f.as_ref().expect("fault recorded");
                        assert!(
                            nonfinite_rows
                                .iter()
                                .any(|&r| f.row == (row_base + r) as u64)
                                && f.layer as usize == layer,
                            "fault ({}, {}) names no bad row of {nonfinite_rows:?}",
                            f.layer,
                            f.row
                        );
                    }
                }
                if !nonfinite_rows.is_empty() {
                    fault_cases += 1;
                }

                // Each fused row against the incumbent one-workgroup-per-token router.
                for r in 0..rows {
                    let one = upload_u32(&mut gpu, DType::F32, &bits[r * E..][..E]);
                    let inc_idx = upload_u32(&mut gpu, DType::F32, &[CANARY; K]);
                    let inc_w = upload_u32(&mut gpu, DType::F32, &[CANARY; K]);
                    gpu.moe_router_softmax_top10_f32(
                        &one,
                        &inc_idx,
                        &inc_w,
                        1,
                        normalize,
                        round,
                        layer,
                        row_base + r,
                    )
                    .expect("incumbent");
                    gpu.hip.device_synchronize().expect("sync");
                    let _ = gpu.take_router_fault();
                    let (ii, iw) = (
                        download_u32(&gpu, &inc_idx, K),
                        download_u32(&gpu, &inc_w, K),
                    );
                    if ii != got_d[0][r * K..][..K] || iw != got_d[1][r * K..][..K] {
                        eprintln!(
                            "MISMATCH rows {rows} seed {seed} row {r}: fused vs incumbent router"
                        );
                        bad += 1;
                    }
                }
                cases += 1;
            }
        }
    }
    eprintln!(
        "moe_route_scatter: {cases} cases (rows 1..=8 x 48 seeds x 4 recipes, {fault_cases} with non-finite rows), {words} words compared, mismatches {bad}"
    );
    eprintln!(
        "sorted_slot_index / inverse_perm (indices 4, 6 of {NAMES:?}): cases where fused != chain bytewise {strict_diff:?}; chain != ascending-order model {chain_vs_model:?}; chain != chain rerun {chain_self_diff:?}"
    );
    assert_eq!(
        bad, 0,
        "fused route+scatter differs from the two-launch chain"
    );
}

/// Run alone (separate process, flag `=0`, the default-on kill switch): the launcher declines and
/// touches nothing.
#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn moe_route_scatter_declines_without_flag_hw() {
    if std::env::var("HIPFIRE_QWEN4_MOE_ROUTE_SCATTER").as_deref() == Ok("1") {
        eprintln!("SKIP: flag forced on in this process (run this test alone).");
        return;
    }
    // Read once per process by the launcher; the flag is default on, so kill it explicitly.
    std::env::set_var("HIPFIRE_QWEN4_MOE_ROUTE_SCATTER", "0");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let o = Outputs::new(&mut gpu, 8);
    let logits = upload_u32(&mut gpu, DType::F32, &vec![0u32; 8 * E]);
    assert!(!gpu.moe_route_scatter_top10_admitted(8, E, K));
    let launched = gpu
        .moe_route_scatter_top10_f32(
            &logits,
            &o.idx,
            &o.w,
            8,
            true,
            true,
            0,
            0,
            &o.counts,
            &o.offsets,
            &o.sorted,
            &o.tiles,
            &o.inverse,
            E,
            K,
            m_total_bound(80),
            BLOCK_M,
        )
        .expect("decline");
    assert!(!launched);
    gpu.hip.device_synchronize().expect("sync");
    for (i, d) in o.dump(&gpu).iter().enumerate() {
        assert!(d.iter().all(|w| *w == CANARY), "{} touched", NAMES[i]);
    }
}
