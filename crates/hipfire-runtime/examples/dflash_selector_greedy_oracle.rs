// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU-vs-CPU byte oracle for `dflash_selector_greedy_exact` (Design A §4,
// slice A3). The CPU reference is `propose_candidates_from_rows_capture`
// (the unchanged `propose_inner` scorer). For every case the device scores
// `[rows*K]` and the published block `[seed, id_0..id_{rows-1}]` must equal
// the CPU scores (u32 bit equality; two NaNs compare equal) and chosen ids.
//
// Classes: real 27B draft codebooks (random, ties, adversarial predecessors,
// subnormal projections, all row counts 1..15), synthetic f32/f16 codebooks
// (signed zeros, f16/f32 subnormals, extreme magnitudes, inf/NaN), malformed
// ids (status word, no publication past the fault), and the pinned owner
// input staging transport (bytes + in-flight refusals, canaries).
//
// Usage:
//   cargo run --release -p hipfire-runtime --features deltanet \
//     --example dflash_selector_greedy_oracle -- \
//     ~/.hipfire/models/qwen38-27b-dflash-mq4.hfq [--trials N]
// (`deltanet` is a default feature.) Exit status is non-zero on any mismatch.

use hipfire_runtime::dflash::{
    self, DflashConfig, DflashWeights, PinnedWindowInputs, SelectorCodebook,
    SelectorDeviceCodebooks, SelectorGreedyBlock,
};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu};
use std::path::Path;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

struct Ctx<'a> {
    gpu: &'a mut Gpu,
    fails: usize,
    cases: usize,
}

impl Ctx<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        label: &str,
        weights: &DflashWeights,
        books: &SelectorDeviceCodebooks,
        rows: usize,
        proj: &[f32],
        ids: &[u32],
        vals: &[f32],
        seed: u32,
    ) {
        self.cases += 1;
        let k = weights.selector_top_k.unwrap();
        let rank = weights.selector_rank.unwrap();
        let blk = SelectorGreedyBlock::new(self.gpu, weights, rows).expect("block alloc");
        let pt = self.gpu.upload_f32(proj, &[rows * rank]).expect("proj");
        let it = self
            .gpu
            .upload_raw(&bytes_u32(ids), &[rows * k])
            .expect("ids");
        let vt = self.gpu.upload_f32(vals, &[rows * k]).expect("vals");
        self.gpu
            .hip
            .memcpy_htod(blk.seed_buffer(), &seed.to_le_bytes())
            .unwrap();
        // Poison every output so an unwritten word cannot match by accident.
        self.gpu
            .hip
            .memcpy_htod(blk.status_buffer(), &[0xFF; 4])
            .unwrap();
        self.gpu
            .hip
            .memcpy_htod(blk.block_buffer(), &vec![0xFF; (rows + 1) * 4])
            .unwrap();
        self.gpu
            .hip
            .memcpy_htod(blk.scores_buffer(), &vec![0xFF; rows * k * 4])
            .unwrap();
        blk.enqueue(self.gpu, books, &pt, &it, &vt)
            .expect("enqueue");
        self.gpu.hip.device_synchronize().unwrap();
        let got = blk.read_back(self.gpu).expect("readback");

        let mut cpu_scores = Vec::new();
        let cpu = dflash::propose_candidates_from_rows_capture(
            weights,
            proj,
            ids,
            vals,
            rows,
            seed,
            0.0,
            None,
            Some(&mut cpu_scores),
        )
        .expect("cpu reference");
        let mut ok = got.status == 0 && got.block[0] == seed;
        if got.block[1..] != cpu.tokens[..] {
            ok = false;
        }
        if got.scores.len() != cpu_scores.len()
            || !got
                .scores
                .iter()
                .zip(&cpu_scores)
                .all(|(a, b)| same_f32(*a, *b))
        {
            ok = false;
        }
        if !ok {
            self.fails += 1;
            let first = got
                .scores
                .iter()
                .zip(&cpu_scores)
                .position(|(a, b)| !same_f32(*a, *b));
            eprintln!(
                "FAIL {label}: status={:#x} block_gpu={:?} block_cpu={:?} first_score_diff={first:?}",
                got.status, got.block, cpu.tokens
            );
        }
        for t in [pt, it, vt] {
            let _ = self.gpu.hip.free(t.buf);
        }
        blk.free(self.gpu);
    }
}

fn rand_ids(rng: &mut Rng, n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|_| rng.below(vocab) as u32).collect()
}

fn synth_book(
    rng: &mut Rng,
    vocab: usize,
    rank: usize,
    f16: bool,
    special: bool,
) -> SelectorCodebook {
    let n = vocab * rank;
    if f16 {
        let specials: [u16; 12] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00,
            0x0200,
        ];
        let nan_inf: [u16; 4] = [0x7C00, 0xFC00, 0x7E00, 0x7D55];
        let data = (0..n)
            .map(|_| {
                let r = rng.below(8);
                if special && r == 0 {
                    nan_inf[rng.below(4)]
                } else if r < 4 {
                    specials[rng.below(specials.len())]
                } else {
                    let v = rng.unit();
                    hipfire_runtime::llama::f32_to_f16(v * 4.0)
                }
            })
            .collect();
        SelectorCodebook {
            vocab,
            rank,
            f16_data: Some(data),
            f32_data: None,
        }
    } else {
        let specials: [f32; 12] = [
            0.0,
            -0.0,
            1e-40,
            -1e-40,
            1.4e-45,
            -1.4e-45,
            1.1754942e-38,
            3.0e38,
            -3.0e38,
            1.0,
            -1.0,
            1e-20,
        ];
        let nan_inf = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let data = (0..n)
            .map(|_| {
                let r = rng.below(8);
                if special && r == 0 {
                    nan_inf[rng.below(3)]
                } else if r < 4 {
                    specials[rng.below(specials.len())]
                } else {
                    rng.unit() * 4.0
                }
            })
            .collect();
        SelectorCodebook {
            vocab,
            rank,
            f16_data: None,
            f32_data: Some(data),
        }
    }
}

fn synth_vec(rng: &mut Rng, n: usize, special: bool) -> Vec<f32> {
    let specials: [f32; 10] = [
        0.0, -0.0, 1e-40, -1e-40, 1.4e-45, -1.4e-45, 3.0e38, -3.0e38, 1e-30, -1e-30,
    ];
    (0..n)
        .map(|_| {
            let r = rng.below(6);
            if special && r == 0 {
                [f32::INFINITY, f32::NEG_INFINITY, f32::NAN][rng.below(3)]
            } else if r < 3 {
                specials[rng.below(specials.len())]
            } else {
                rng.unit() * 3.0
            }
        })
        .collect()
}

fn input_staging_checks(ctx: &mut Ctx<'_>) {
    ctx.cases += 1;
    let gpu = &*ctx.gpu;
    let mut fail = |what: &str, ctx_fails: &mut usize| {
        *ctx_fails += 1;
        eprintln!("FAIL input staging: {what}");
    };
    let mut st = PinnedWindowInputs::new(gpu, 256).expect("staging");
    let stream = gpu.hip.stream_create().expect("stream");
    let dst = gpu.hip.malloc(512).expect("dst");
    gpu.hip.memcpy_htod(&dst, &vec![0xA5u8; 512]).unwrap();
    let q: Vec<u32> = (0..16).map(|i| 1000 + i * 3).collect();
    let v: Vec<u32> = (0..16).map(|i| 5000 + i).collect();
    let off = st
        .pack_seed_mask_positions(0x1234_5678, 248_070, &q, &v)
        .unwrap();
    st.submit(gpu, &dst, &stream).unwrap();
    // In flight: repack and resubmit are refused; the pinned bytes are frozen.
    if st.pack(&[&[1u32]]).is_ok() {
        fail("pack accepted while in flight", &mut ctx.fails);
    }
    if st.submit(gpu, &dst, &stream).is_ok() {
        fail("second submit accepted while in flight", &mut ctx.fails);
    }
    st.complete(gpu).unwrap();
    let mut got = vec![0u8; 512];
    gpu.hip.memcpy_dtoh(&mut got, &dst).unwrap();
    let mut want = vec![0xA5u8; 512];
    let mut words = vec![0x1234_5678u32, 248_070];
    words.extend(&q);
    words.extend(&v);
    want[..words.len() * 4].copy_from_slice(&bytes_u32(&words));
    if got != want
        || off.seed != 0
        || off.mask != 4
        || off.q_positions != 8
        || off.verify_positions != 8 + 64
        || off.total_bytes != words.len() * 4
    {
        fail("uploaded bytes / offsets differ", &mut ctx.fails);
    }
    // Reuse after completion; over-capacity and oversize-for-destination refused.
    if st.pack(&[&[7u32, 8]]).is_err() {
        fail("repack after complete refused", &mut ctx.fails);
    }
    if st.pack(&[&vec![0u32; 65]]).is_ok() {
        fail("over-capacity pack accepted", &mut ctx.fails);
    }
    let small = gpu.hip.malloc(4).unwrap();
    st.pack(&[&[7u32, 8]]).unwrap();
    if st.submit(gpu, &small, &stream).is_ok() {
        fail("submit larger than destination accepted", &mut ctx.fails);
    }
    st.destroy(gpu).unwrap();
    let _ = gpu.hip.free(small);
    let _ = gpu.hip.free(dst);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dflash_selector_greedy_oracle <draft.hfq> [--trials N]");
        std::process::exit(2);
    }
    let mut trials = 24usize;
    if let Some(i) = args.iter().position(|a| a == "--trials") {
        trials = args[i + 1].parse().unwrap();
    }
    let hfq = HfqFile::open(Path::new(&args[1])).expect("open draft");
    let cfg = DflashConfig::from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("gpu");
    let mut weights = DflashWeights::load(&mut gpu, &hfq, &cfg).expect("load weights");
    assert!(weights.has_candidate_selector(), "draft has no selector");
    let rank = weights.selector_rank.unwrap_or(256);
    let k = weights.selector_top_k.unwrap_or(16);
    weights.selector_rank = Some(rank);
    weights.selector_top_k = Some(k);
    let vocab = weights.predecessor_codebook.as_ref().unwrap().vocab;
    eprintln!("selector: vocab={vocab} rank={rank} k={k} gpu={}", gpu.arch);
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    let mut ctx = Ctx {
        gpu: &mut gpu,
        fails: 0,
        cases: 0,
    };

    // ---- real codebooks ----------------------------------------------------
    let books = SelectorDeviceCodebooks::upload(ctx.gpu, &weights).expect("upload books");
    for t in 0..trials {
        let rows = 15;
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let ids = rand_ids(&mut rng, rows * k, vocab);
        let vals: Vec<f32> = (0..rows * k).map(|_| rng.unit() * 8.0).collect();
        let seed = rng.below(vocab) as u32;
        ctx.run(
            &format!("real random t={t}"),
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            seed,
        );
    }
    for rows in 1..=15usize {
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let ids = rand_ids(&mut rng, rows * k, vocab);
        let vals: Vec<f32> = (0..rows * k).map(|_| rng.unit() * 4.0).collect();
        ctx.run(
            &format!("real rows={rows}"),
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            1,
        );
    }
    // Ties: duplicate ids with equal unaries tie exactly; the first wins.
    for t in 0..trials {
        let rows = 15;
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let mut ids = vec![0u32; rows * k];
        let mut vals = vec![0f32; rows * k];
        for r in 0..rows {
            let base = rng.below(vocab) as u32;
            let u = rng.unit();
            for j in 0..k {
                ids[r * k + j] = if t % 2 == 0 {
                    base
                } else {
                    [base, (base ^ 1) % vocab as u32][j % 2]
                };
                vals[r * k + j] = u;
            }
        }
        ctx.run(
            &format!("real ties t={t}"),
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            7,
        );
    }
    // Zero projection: scores are the unaries (signed zeros included).
    {
        let rows = 15;
        let proj = vec![0f32; rows * rank];
        let ids = rand_ids(&mut rng, rows * k, vocab);
        let vals: Vec<f32> = (0..rows * k).map(|i| [0.0, -0.0][i % 2]).collect();
        ctx.run(
            "real zero-proj signed-zero unaries",
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            3,
        );
        let proj = vec![-0.0f32; rows * rank];
        ctx.run(
            "real negzero-proj",
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            3,
        );
    }
    // Subnormal / extreme projections against the real books.
    for (name, scale) in [
        ("1e-40", 1e-40f32),
        ("1e-38", 1e-38),
        ("1e-45", 1.4e-45),
        ("1e30", 1e30),
    ] {
        let rows = 15;
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit() * scale).collect();
        let ids = rand_ids(&mut rng, rows * k, vocab);
        let vals: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
        ctx.run(
            &format!("real proj*{name}"),
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            11,
        );
    }
    // Adversarial predecessors: edges of the vocabulary, candidate == previous
    // predecessor, a repeating chain.
    for seed in [0u32, vocab as u32 - 1, 1, vocab as u32 / 2] {
        let rows = 15;
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let mut ids = rand_ids(&mut rng, rows * k, vocab);
        for r in 0..rows {
            ids[r * k] = 0;
            ids[r * k + k - 1] = vocab as u32 - 1;
            ids[r * k + k / 2] = seed;
        }
        let vals: Vec<f32> = (0..rows * k).map(|_| rng.unit()).collect();
        ctx.run(
            &format!("real edge seed={seed}"),
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            seed,
        );
    }
    {
        // Every row offers the same ids so the chain revisits predecessors.
        let rows = 15;
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let row_ids = rand_ids(&mut rng, k, vocab);
        let ids: Vec<u32> = (0..rows).flat_map(|_| row_ids.clone()).collect();
        let vals = vec![0.5f32; rows * k];
        ctx.run(
            "real repeating chain",
            &weights,
            &books,
            rows,
            &proj,
            &ids,
            &vals,
            row_ids[0],
        );
    }

    // ---- malformed ids: status word, nothing published past the fault -----
    {
        ctx.cases += 1;
        let rows = 6;
        let blk = SelectorGreedyBlock::new(ctx.gpu, &weights, rows).unwrap();
        let proj: Vec<f32> = (0..rows * rank).map(|_| rng.unit()).collect();
        let mut ids = rand_ids(&mut rng, rows * k, vocab);
        ids[3 * k + 1] = vocab as u32; // row 3, candidate 1
        let vals = vec![0f32; rows * k];
        let pt = ctx.gpu.upload_f32(&proj, &[rows * rank]).unwrap();
        let it = ctx.gpu.upload_raw(&bytes_u32(&ids), &[rows * k]).unwrap();
        let vt = ctx.gpu.upload_f32(&vals, &[rows * k]).unwrap();
        ctx.gpu
            .hip
            .memcpy_htod(blk.seed_buffer(), &5u32.to_le_bytes())
            .unwrap();
        ctx.gpu
            .hip
            .memcpy_htod(blk.status_buffer(), &[0xFF; 4])
            .unwrap();
        ctx.gpu
            .hip
            .memcpy_htod(blk.block_buffer(), &vec![0xFF; (rows + 1) * 4])
            .unwrap();
        blk.enqueue(ctx.gpu, &books, &pt, &it, &vt).unwrap();
        ctx.gpu.hip.device_synchronize().unwrap();
        let g = blk.read_back(ctx.gpu).unwrap();
        if g.status != (3 | (4 << 8))
            || g.block[4..].iter().any(|&w| w != 0xFFFF_FFFF)
            || g.block[0] != 5
        {
            ctx.fails += 1;
            eprintln!(
                "FAIL bad candidate id: status={:#x} block={:?}",
                g.status, g.block
            );
        }
        // Bad seed.
        ctx.gpu
            .hip
            .memcpy_htod(blk.seed_buffer(), &(vocab as u32).to_le_bytes())
            .unwrap();
        ctx.gpu
            .hip
            .memcpy_htod(blk.block_buffer(), &vec![0xFF; (rows + 1) * 4])
            .unwrap();
        blk.enqueue(ctx.gpu, &books, &pt, &it, &vt).unwrap();
        ctx.gpu.hip.device_synchronize().unwrap();
        let g = blk.read_back(ctx.gpu).unwrap();
        if g.status != 2 || g.block.iter().any(|&w| w != 0xFFFF_FFFF) {
            ctx.fails += 1;
            eprintln!("FAIL bad seed: status={:#x} block={:?}", g.status, g.block);
        }
        for t in [pt, it, vt] {
            let _ = ctx.gpu.hip.free(t.buf);
        }
        blk.free(ctx.gpu);
    }
    books.free(ctx.gpu);

    // ---- synthetic codebooks: f32/f16 mixes, ranks, k, specials -----------
    let real_pred = weights.predecessor_codebook.take();
    let real_succ = weights.successor_codebook.take();
    for t in 0..trials.max(8) {
        let sv = 48usize;
        let srank = [1usize, 2, 7, 33, 64, 256][t % 6];
        let sk = [1usize, 2, 3, 5, 8, 15, 16][t % 7];
        let special = t % 3 == 2;
        let pf = (t & 1) == 1;
        let sf = (t & 2) == 2;
        weights.predecessor_codebook = Some(synth_book(&mut rng, sv, srank, pf, special));
        weights.successor_codebook = Some(synth_book(&mut rng, sv, srank, sf, special));
        weights.selector_rank = Some(srank);
        weights.selector_top_k = Some(sk);
        let sbooks = SelectorDeviceCodebooks::upload(ctx.gpu, &weights).expect("upload synth");
        for rows in [1usize, 4, 15] {
            let proj = synth_vec(&mut rng, rows * srank, special);
            let ids = rand_ids(&mut rng, rows * sk, sv);
            let vals = synth_vec(&mut rng, rows * sk, special);
            let seed = rng.below(sv) as u32;
            ctx.run(
                &format!("synth t={t} rank={srank} k={sk} rows={rows} pf16={pf} sf16={sf} special={special}"),
                &weights, &sbooks, rows, &proj, &ids, &vals, seed,
            );
        }
        sbooks.free(ctx.gpu);
    }
    weights.predecessor_codebook = real_pred;
    weights.successor_codebook = real_succ;

    input_staging_checks(&mut ctx);

    println!(
        "dflash_selector_greedy_oracle: {} cases, {} failures",
        ctx.cases, ctx.fails
    );
    if ctx.fails != 0 {
        std::process::exit(1);
    }
    let _ = DType::F32;
}
