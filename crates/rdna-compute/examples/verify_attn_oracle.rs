// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Byte oracle for VerifyAttn (`Gpu::try_attention_verify_gqa`) against the
// batched flash tile + reduce it replaces
// (`attention_flash_{fp8_e4m3,q8_0}_tile_batched` +
// `attention_flash_asym_reduce_batched`), H2 shape (24 q heads, 4 kv heads,
// head_dim 256). On gfx1100 (Q8 only) it also checks the multi-row twin
// (`Gpu::try_attention_verify_gqa_rows`) against the R4/R8 tile + reduce
// (`attention_flash_q8_0_rows_masked`), the eager Q8 verify route there.
// On gfx1151 (Q8 only) it checks `Gpu::attention_verify_wmma_with` (the
// production geometry and the other P.V walkers) against the single-slot WMMA
// flash prefill it replaces (`attention_q8_0_flash_prefill_wmma_slots`); the
// reference writes no partials, so the twin's S-tile scratch may hold anything
// and the partials past it must keep their poison.
//
// Every run poisons the whole output and partials allocations (including guard
// regions around the views the launch receives), and the KV cache past the
// last written position holds NaN codes/scales, so a read or write outside the
// reference footprint shows up as a byte difference. The reference and
// candidate must match byte for byte over the full allocations.
//
// Modes:
//   oracle [fp8|q8|both]      B {1,4,16} (+2,5,9,32) x ctx {1,127,128,2K,8K,16K,32K,64K},
//                             eager and graph-capture max_ctx_len, full and
//                             sub-batched partials, cap 262144 capture grid.
//                             gfx1100: plus the R4/R8 reference, B {4,16}
//                             (+5,7,8,9,13,32) x ctx {1..64K, 4097}.
//   stress <reps>             poisoned repeats (rotating 0xFF/0x7F/0x00/0xA5),
//                             each compared with the reference and the first
//                             candidate launch (gfx1151: also its live S tiles).
//                             Run it under ROC_GLOBAL_CU_MASK=0x1 and as 2 and 4
//                             concurrent processes for the co-resident modes.
//                             VA_STRESS_SHAPES="b,ctx,capture,cap;..." picks the
//                             shapes; VA_STRESS_REF=1 loops the reference
//                             instead (control). Mismatches print STRESS_DIFF.
//   soak <secs>               back-to-back candidate launches (B16 32K fp8/q8),
//                             every output compared with the reference.
//   bench                     wall-clock per call, reference vs candidate.
//   bench_rows                Q8 eager: multi-row R4/R8 vs VerifyAttn wall-clock
//                             (gfx1100: its R4/R8 twin; gfx1201: the tile twin).
//   sweep                     (gfx1151) candidate wall-clock per launch geometry.
// Exit 0 = pass. Run: cargo run --release -p rdna-compute --example verify_attn_oracle -- oracle both

use rdna_compute::attention::{VerifyKv, VerifyWmmaGeometry, VerifyWmmaPv};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const TILE: usize = 128;
const STRIDE: usize = 2 + HD;
const QDIM: usize = NH * HD;
/// Guard elements on each side of the output view and after the partials view.
const GUARD: usize = 4096;
const POS_GUARD: i32 = 0x3fff_ffff;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn f16_bits(x: f32) -> u16 {
    half_from_f32(x)
}

// Round-to-nearest-even f32 -> f16 bits (no external crate in examples).
fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && (half & 1) == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { half + 1 } else { half };
    sign | r as u16
}

fn row_bytes(kv: VerifyKv) -> usize {
    match kv {
        VerifyKv::Q8 => NKV * (HD / 32) * 34,
        VerifyKv::Fp8 => NKV * (HD + 2),
    }
}

/// Block/head scale: mostly positive normal f16 in [2^-9, 2^-4], with some
/// negative, zero and subnormal scales to pin signed-zero and denormal paths.
fn scale_f16(rng: &mut Rng) -> u16 {
    match rng.below(64) {
        0 => 0x0000,
        1 => 0x8000,
        2 => 0x0001 + rng.below(0x3ff) as u16,
        3 | 4 => f16_bits(-(0.002 + 0.06 * rng.unit())),
        _ => f16_bits(0.002 + 0.06 * rng.unit()),
    }
}

/// K or V cache: `valid` random rows, then NaN-poisoned rows up to `cap`.
fn fill_kv(kv: VerifyKv, rng: &mut Rng, valid: usize, cap: usize) -> Vec<u8> {
    let rb = row_bytes(kv);
    let mut buf = vec![0u8; cap * rb];
    for t in 0..cap {
        let row = &mut buf[t * rb..(t + 1) * rb];
        let poison = t >= valid;
        match kv {
            VerifyKv::Q8 => {
                for blk in 0..NKV * (HD / 32) {
                    let o = blk * 34;
                    let s = if poison { 0x7e00 } else { scale_f16(rng) };
                    row[o..o + 2].copy_from_slice(&s.to_le_bytes());
                    for j in 0..32 {
                        row[o + 2 + j] = if poison {
                            0x7f
                        } else if rng.below(16) == 0 {
                            [0x80u8, 0x00, 0x7f, 0x81][rng.below(4) as usize]
                        } else {
                            rng.next() as u8
                        };
                    }
                }
            }
            VerifyKv::Fp8 => {
                for i in 0..NKV * HD {
                    row[i] = if poison {
                        0x7f
                    } else {
                        loop {
                            let c = rng.next() as u8;
                            if c & 0x7f != 0x7f {
                                break c;
                            }
                        }
                    };
                }
                for h in 0..NKV {
                    let s = if poison { 0x7e00 } else { scale_f16(rng) };
                    let o = NKV * HD + 2 * h;
                    row[o..o + 2].copy_from_slice(&s.to_le_bytes());
                }
            }
        }
    }
    buf
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kv: VerifyKv,
    b: usize,
    ctx: usize,
    /// Graph-capture shape: `max_ctx_len = cap` instead of the logical context.
    capture: bool,
    cap: usize,
    /// Partials capacity in rows (at this case's max_tiles); < b sub-batches.
    partial_rows: usize,
    /// Reference is the multi-row R4/R8 tile (Q8, b >= 4) instead of tile_batched.
    rows: bool,
    /// gfx1151: the WMMA twin at this geometry against the WMMA reference
    /// (Q8 only; `partial_rows` unused, the scratch is exactly the S tiles).
    wmma: Option<VerifyWmmaGeometry>,
}

impl Case {
    fn start(&self) -> usize {
        self.ctx - 1
    }
    fn valid(&self) -> usize {
        self.start() + self.b
    }
    fn max_ctx_len(&self) -> usize {
        if self.capture {
            self.cap
        } else {
            self.valid()
        }
    }
    fn max_tiles(&self) -> usize {
        self.max_ctx_len().div_ceil(TILE)
    }
    fn partials_len(&self) -> usize {
        if self.wmma.is_some() {
            self.wmma_scratch_bytes() / 4
        } else {
            self.partial_rows * NH * self.max_tiles() * STRIDE
        }
    }
    /// f16 S tiles of the gfx1151 twin: row groups x kv heads x 6 x key tiles x 512 B.
    fn wmma_scratch_bytes(&self) -> usize {
        self.b.div_ceil(16) * NKV * 6 * self.max_ctx_len().div_ceil(16) * 512
    }
    /// The multi-row reference's rows per block (`flash_rows_per_block`).
    fn group_rows(&self) -> usize {
        if self.b >= 8 {
            8
        } else {
            4
        }
    }
    fn ref_name(&self) -> &'static str {
        if self.wmma.is_some() {
            "wmma"
        } else if self.rows {
            "rows"
        } else {
            "tile"
        }
    }
}

struct Bufs {
    q: GpuTensor,
    q_full: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    out: GpuTensor,
    out_full: GpuTensor,
    part: GpuTensor,
    part_full: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu, c: &Case, seed: u64) -> Bufs {
        let mut rng = Rng(seed);
        let mut qh = vec![0f32; c.b * QDIM + GUARD];
        for (i, x) in qh.iter_mut().enumerate() {
            *x = if i >= c.b * QDIM {
                f32::NAN
            } else {
                // Rows alternate between ordinary and 12x-scaled queries so
                // some tiles run their softmax deep into expf underflow.
                let row = i / QDIM;
                let amp = if row % 3 == 2 { 12.0 } else { 1.0 };
                (rng.unit() * 2.0 - 1.0) * amp
            };
        }
        let q_full = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
        let q = q_full.sub_offset(0, c.b * QDIM);
        let kh = fill_kv(c.kv, &mut rng, c.valid(), c.cap);
        let vh = fill_kv(c.kv, &mut rng, c.valid(), c.cap);
        let k = gpu.upload_raw(&kh, &[kh.len()]).expect("k");
        let v = gpu.upload_raw(&vh, &[vh.len()]).expect("v");
        let mut posh: Vec<i32> = (0..c.b).map(|i| (c.start() + i) as i32).collect();
        posh.extend(std::iter::repeat(POS_GUARD).take(8));
        let pb: Vec<u8> = posh.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let pos = gpu.upload_raw(&pb, &[posh.len()]).expect("pos");
        let out_full = gpu.zeros(&[c.b * QDIM + 2 * GUARD], DType::F32).expect("out");
        let out = out_full.sub_offset(GUARD, c.b * QDIM);
        let part_full = gpu.zeros(&[c.partials_len() + GUARD], DType::F32).expect("partials");
        let part = part_full.sub_offset(0, c.partials_len());
        Bufs { q, q_full, k, v, pos, out, out_full, part, part_full }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.q_full, self.k, self.v, self.pos, self.out_full, self.part_full] {
            let _ = gpu.free_tensor(t);
        }
        let _ = (self.q, self.out, self.part);
    }
}

fn poison(gpu: &Gpu, t: &GpuTensor, byte: u8) {
    gpu.hip.memset(&t.buf, byte as i32, t.buf.size()).expect("memset");
}

fn bytes_of(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let mut v = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("dtoh");
    v
}

/// One launch of the reference (`candidate == false`) or VerifyAttn.
fn launch(gpu: &mut Gpu, c: &Case, bf: &Bufs, candidate: bool) {
    std::sync::Arc::make_mut(&mut gpu.flags).verify_attn = candidate;
    if let Some(geom) = c.wmma {
        if candidate {
            let ran = gpu
                .attention_verify_wmma_with(
                    &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part, geom,
                )
                .expect("verify_attn wmma launch");
            assert!(ran, "VerifyAttn declined an admitted case: {c:?}");
        } else {
            gpu.attention_q8_0_flash_prefill_wmma_slots(
                &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, None, None, None, None,
            )
            .expect("reference wmma");
        }
        return;
    }
    if c.rows {
        let ran = if candidate {
            gpu.try_attention_verify_gqa_rows(
                &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part, c.group_rows(),
            )
            .expect("verify_attn rows launch")
        } else {
            gpu.attention_flash_q8_0_rows_masked(
                &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part,
            )
            .expect("reference rows")
        };
        assert!(ran, "declined an admitted rows case (candidate={candidate}): {c:?}");
        return;
    }
    if candidate {
        let ran = gpu
            .try_attention_verify_gqa(
                c.kv, &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part,
                None, 0,
            )
            .expect("verify_attn launch");
        assert!(ran, "VerifyAttn declined an admitted case: {c:?}");
    } else {
        match c.kv {
            VerifyKv::Fp8 => gpu
                .attention_flash_fp8_e4m3_tile_batched(
                    &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.cap, c.max_ctx_len(), c.b,
                    &bf.part, None, 0, 0,
                )
                .expect("reference fp8"),
            VerifyKv::Q8 => gpu
                .attention_flash_q8_0_batched_masked(
                    &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.cap, c.max_ctx_len(), c.b,
                    &bf.part, None, 0, 0,
                )
                .expect("reference q8"),
        }
    }
}

/// Poison both allocations with `byte`, launch, return (out_full, part_full).
fn run(gpu: &mut Gpu, c: &Case, bf: &Bufs, candidate: bool, byte: u8) -> (Vec<u8>, Vec<u8>) {
    poison(gpu, &bf.out_full, byte);
    poison(gpu, &bf.part_full, byte);
    launch(gpu, c, bf, candidate);
    (bytes_of(gpu, &bf.out_full), bytes_of(gpu, &bf.part_full))
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or(if a.len() != b.len() { Some(a.len().min(b.len())) } else { None })
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn out_view(c: &Case, full: &[u8]) -> Vec<f32> {
    as_f32(&full[GUARD * 4..(GUARD + c.b * QDIM) * 4])
}

/// Byte ranges of the partials the reference writes (tiles below each row's
/// causal bound; for the multi-row reference, below its row group's bound),
/// in chunk-local rows of the sub-batched launch. gfx1151: the twin's S-tile
/// scratch (any content; the reference writes none).
fn written_ranges(c: &Case) -> Vec<(usize, usize)> {
    if c.wmma.is_some() {
        return vec![(0, c.wmma_scratch_bytes())];
    }
    let mt = c.max_tiles();
    let sub = c.partial_rows.min(c.b);
    let mut r = Vec::new();
    let mut off = 0;
    while off < c.b {
        let chunk = (c.b - off).min(sub);
        for lr in 0..chunk {
            let last = if c.rows { (lr - lr % c.group_rows() + c.group_rows()).min(chunk) - 1 } else { lr };
            let seq = c.start() + off + last + 1;
            let nt = seq.div_ceil(TILE).min(mt);
            for h in 0..NH {
                let base = ((lr * NH + h) * mt) * STRIDE * 4;
                r.push((base, base + nt * STRIDE * 4));
            }
        }
        off += chunk;
    }
    r
}

fn kv_name(kv: VerifyKv) -> &'static str {
    match kv {
        VerifyKv::Fp8 => "fp8",
        VerifyKv::Q8 => "q8",
    }
}

fn oracle_cases(kvs: &[VerifyKv], rows_twin: bool) -> Vec<Case> {
    const CAP: usize = 65536 + 256;
    let mut cases = Vec::new();
    for &kv in kvs {
        let tile = |b, ctx, capture, cap, partial_rows| Case { kv, b, ctx, capture, cap, partial_rows, rows: false, wmma: None };
        for &b in &[1usize, 4, 16] {
            for &ctx in &[1usize, 127, 128, 2048, 8192, 16384, 32768, 65536] {
                for &capture in &[false, true] {
                    cases.push(tile(b, ctx, capture, CAP, 16));
                }
            }
        }
        // Odd batch sizes, ragged row groups, the 32-row ceiling, sub-batching.
        for &(b, ctx, rows) in &[(2, 129, 16), (5, 2047, 16), (9, 4097, 16), (32, 3000, 32), (16, 8191, 5), (13, 300, 4)] {
            cases.push(tile(b, ctx, false, CAP, rows));
            cases.push(tile(b, ctx, true, CAP, rows));
        }
        // The real graph-capture shape: physical cap 262144 (2048 partial tiles).
        for &(b, ctx) in &[(16, 2048), (4, 20000), (16, 9000)] {
            cases.push(tile(b, ctx, true, 262144, 16));
        }
        if rows_twin && kv == VerifyKv::Q8 {
            // Multi-row reference: `capture` here only sizes max_ctx_len to
            // the cap (the reference itself is eager-only).
            let rows = |b, ctx, capture, partial_rows| Case { kv, b, ctx, capture, cap: CAP, partial_rows, rows: true, wmma: None };
            for &b in &[4usize, 16] {
                for &ctx in &[1usize, 127, 128, 2048, 4097, 8192, 16384, 32768, 65536] {
                    for &capture in &[false, true] {
                        cases.push(rows(b, ctx, capture, 16));
                    }
                }
            }
            // R4 with two row groups, ragged R8 groups, the 32-row ceiling,
            // sub-batching (chunks that split R8 groups).
            for &(b, ctx, pr) in &[(5, 2047, 16), (7, 20000, 16), (8, 129, 16), (9, 4097, 16), (13, 300, 4), (32, 3000, 32), (16, 8191, 5), (6, 255, 16)] {
                cases.push(rows(b, ctx, false, pr));
                cases.push(rows(b, ctx, true, pr));
            }
        }
    }
    cases
}

/// gfx1151: production geometry over the B x ctx x capture grid, odd and
/// two-row-group batches, the reference fragment layout, both P.V walkers in
/// both fragment layouts with one-split and many-split S launches, and the
/// physical-cap capture shape.
fn wmma_oracle_cases() -> Vec<Case> {
    const CAP: usize = 65536 + 256;
    let prod = Some(VerifyWmmaGeometry::default());
    let mut cases = Vec::new();
    let mut push = |b: usize, ctx: usize, capture: bool, cap: usize, wmma| {
        cases.push(Case { kv: VerifyKv::Q8, b, ctx, capture, cap, partial_rows: 16, rows: false, wmma });
    };
    for &b in &[1usize, 4, 16] {
        for &ctx in &[1usize, 127, 128, 2048, 8192, 16384, 32768, 65536] {
            for &capture in &[false, true] {
                push(b, ctx, capture, CAP, prod);
            }
        }
    }
    for &(b, ctx) in &[(2, 129), (3, 17), (5, 2047), (8, 15), (9, 4097), (13, 300), (17, 1000), (24, 2049), (32, 3000)] {
        push(b, ctx, false, CAP, prod);
        push(b, ctx, true, CAP, prod);
    }
    for &(b, ctx) in &[(1, 127), (4, 8192), (5, 2047), (16, 16384), (32, 3000)] {
        for &capture in &[false, true] {
            push(b, ctx, capture, CAP, Some(VerifyWmmaGeometry { packed: false, ..VerifyWmmaGeometry::default() }));
        }
    }
    for &(b, ctx) in &[(4, 4099), (16, 8192), (9, 1025), (1, 300), (32, 3000)] {
        for (i, &pv) in VerifyWmmaPv::ALL.iter().enumerate() {
            for (j, &qk_waves) in [1usize, 64, 1_000_000].iter().enumerate() {
                for packed in [true, false] {
                    push(b, ctx, (i + j) % 2 == 1, CAP, Some(VerifyWmmaGeometry { packed, qk_waves, pv: Some(pv) }));
                }
            }
        }
    }
    for &(b, ctx) in &[(16, 2048), (4, 20000), (16, 9000)] {
        push(b, ctx, true, 262144, prod);
    }
    cases
}

fn parse_kvs(arg: Option<&str>) -> Vec<VerifyKv> {
    match arg.unwrap_or("both") {
        "fp8" => vec![VerifyKv::Fp8],
        "q8" => vec![VerifyKv::Q8],
        _ => vec![VerifyKv::Fp8, VerifyKv::Q8],
    }
}

fn oracle(gpu: &mut Gpu, kvs: &[VerifyKv]) -> bool {
    let mut ok = true;
    let mut n = 0;
    let rows_twin = gpu.arch == "gfx1100";
    let cases = if gpu.arch == "gfx1151" { wmma_oracle_cases() } else { oracle_cases(kvs, rows_twin) };
    for (i, c) in cases.iter().enumerate() {
        let bf = Bufs::new(gpu, c, 0x5eed_0000 + i as u64);
        let (ro, rp) = run(gpu, c, &bf, false, 0xA5);
        let (co, cp) = run(gpu, c, &bf, true, 0xA5);
        let od = first_diff(&ro, &co);
        // gfx1151: the reference writes no partials; past the S tiles they keep their poison.
        let skip = if c.wmma.is_some() { c.wmma_scratch_bytes() } else { 0 };
        let pd = first_diff(&rp[skip..], &cp[skip..]).map(|d| d + skip);
        let rv = out_view(c, &ro);
        let nan = rv.iter().filter(|x| !x.is_finite()).count();
        let pass = od.is_none() && pd.is_none() && nan == 0;
        let (mut abs, mut rel) = (0f32, 0f32);
        if od.is_some() {
            let cv = out_view(c, &co);
            for (a, b) in rv.iter().zip(&cv) {
                let d = (a - b).abs();
                abs = abs.max(d);
                rel = rel.max(d / a.abs().max(1e-30));
            }
        }
        println!(
            "ORACLE kv={} ref={} B={} ctx={} capture={} cap={} max_tiles={} partial_rows={} out_bytes={} part_bytes={} \
             ref_nonfinite={} out_first_diff={:?} part_first_diff={:?} max_abs={:e} max_rel={:e} {}",
            kv_name(c.kv), c.ref_name(), c.b, c.ctx, c.capture, c.cap, c.max_tiles(), c.partial_rows, ro.len(), rp.len(), nan,
            od, pd, abs, rel, if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
        n += 1;
        bf.free(gpu);
    }
    println!("ORACLE_SUMMARY cases={n} {}", if ok { "PASS" } else { "FAIL" });
    ok
}

/// Byte ranges of the gfx1151 twin's live S tiles (the fragments and key
/// tiles each row group computes; the rest of the scratch keeps the poison).
fn wmma_live_s(c: &Case) -> Vec<(usize, usize)> {
    let geom = c.wmma.expect("wmma case");
    let t_stride = c.max_ctx_len().div_ceil(16);
    let mut r = Vec::new();
    for rg in 0..c.b.div_ceil(16) {
        let rows = (c.b - 16 * rg).min(16);
        let pack = if geom.packed { rows } else { 16 };
        let nf = (6 * pack).div_ceil(16);
        let n_tiles = (c.start() + 16 * rg + rows).div_ceil(16).min(t_stride);
        for kv in 0..NKV {
            for f in 0..nf {
                let base = ((rg * NKV + kv) * 6 + f) * t_stride * 512;
                r.push((base, base + n_tiles * 512));
            }
        }
    }
    r
}

/// Output bytes: differing count and the first difference as (row, head, dim).
fn describe_out(what: &str, a: &[u8], b: &[u8]) -> String {
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    match first_diff(a, b) {
        None => format!("{what}=0"),
        Some(i) => {
            let e = i / 4;
            let (row, rem) = (e / QDIM, e % QDIM);
            let got = f32::from_ne_bytes(a[e * 4..e * 4 + 4].try_into().unwrap());
            let want = f32::from_ne_bytes(b[e * 4..e * 4 + 4].try_into().unwrap());
            format!("{what}={n}B first(row={row} head={} dim={} got={got:e} want={want:e})", rem / HD, rem % HD)
        }
    }
}

/// Live S-tile bytes of two scratch copies: differing count and the first
/// difference as (row group, kv head, fragment, key tile, fragment row, key).
fn describe_s(c: &Case, live: &[(usize, usize)], a: &[u8], b: &[u8]) -> String {
    let n: usize = live.iter().map(|&(s, e)| a[s..e].iter().zip(&b[s..e]).filter(|(x, y)| x != y).count()).sum();
    let Some(i) = live.iter().find_map(|&(s, e)| first_diff(&a[s..e], &b[s..e]).map(|d| s + d)) else {
        return "s_vs_first=0".into();
    };
    let t_stride = c.max_ctx_len().div_ceil(16);
    let tile = i / 512;
    let (t, f, kv, rg) = (tile % t_stride, tile / t_stride % 6, tile / t_stride / 6 % NKV, tile / t_stride / 6 / NKV);
    format!("s_vs_first={n}B first(rg={rg} kv={kv} frag={f} tile={t} row={} key={})", i % 512 / 32, i % 32 / 2)
}

/// Stress shapes (b, ctx, capture, cap, rows). `VA_STRESS_SHAPES="b,ctx,capture,cap;..."` replaces the
/// tile/WMMA list (capture 0/1).
fn stress_shapes(kv: VerifyKv, rows_twin: bool, wmma: bool) -> Vec<(usize, usize, bool, usize, bool)> {
    if let Ok(s) = std::env::var("VA_STRESS_SHAPES") {
        return s
            .split(';')
            .filter(|x| !x.trim().is_empty())
            .map(|x| {
                let v: Vec<usize> = x.split(',').map(|n| n.trim().parse().expect("VA_STRESS_SHAPES")).collect();
                assert_eq!(v.len(), 4, "VA_STRESS_SHAPES entry b,ctx,capture,cap: {x}");
                (v[0], v[1], v[2] != 0, v[3], false)
            })
            .collect();
    }
    let mut shapes = vec![
        (16usize, 8192usize, true, 65536 + 256, false),
        (16, 1000, false, 4096, false),
        (4, 16384, false, 20000, false),
        (1, 127, true, 4096, false),
        (9, 2049, true, 4096, false),
    ];
    if wmma {
        // Ragged row groups at the 16-key tile boundary (live length ctx - 1 + B
        // = 2048 - 1, 2048, 2048 + 1) and a capture cap one past a tile.
        shapes.extend([
            (32, 3000, true, 4096, false),
            (31, 2017, true, 4096, false),
            (17, 2032, true, 4096, false),
            (15, 2035, true, 4096, false),
            (9, 4089, true, 4097, false),
        ]);
    }
    if rows_twin && kv == VerifyKv::Q8 {
        shapes.extend([
            (16usize, 8192usize, false, 8192 + 256, true),
            (4, 16384, false, 20000, true),
            (9, 5000, false, 8192, true),
            (5, 4097, true, 8192, true),
        ]);
    }
    shapes
}

/// `VA_STRESS_REF=1`: the loop launches the reference instead (control run:
/// each launch against the first reference launch).
fn stress(gpu: &mut Gpu, reps: usize, kvs: &[VerifyKv]) -> bool {
    let mut ok = true;
    let mut launches = 0usize;
    let mut mism = 0usize;
    let rows_twin = gpu.arch == "gfx1100";
    let wmma = (gpu.arch == "gfx1151").then(VerifyWmmaGeometry::default);
    let control = std::env::var("VA_STRESS_REF").as_deref() == Ok("1");
    for &kv in kvs {
        for (b, ctx, capture, cap, rows) in stress_shapes(kv, rows_twin, wmma.is_some()) {
            let c = Case { kv, b, ctx, capture, cap, partial_rows: 16, rows, wmma };
            let bf = Bufs::new(gpu, &c, 0xC0FFEE ^ (b * 131 + ctx) as u64);
            let (ro, rp) = run(gpu, &c, &bf, false, 0xA5);
            let ref_out = out_view(&c, &ro);
            let ranges = if control && c.wmma.is_some() { Vec::new() } else { written_ranges(&c) };
            let mut covered = vec![false; rp.len()];
            for &(s, e) in &ranges {
                covered[s..e].iter_mut().for_each(|x| *x = true);
            }
            // Live S tiles, compared with the first candidate launch's.
            let live_s = if c.wmma.is_some() && !control { wmma_live_s(&c) } else { Vec::new() };
            let mut first: Option<(Vec<u8>, Vec<u8>)> = None;
            let mut bad = 0usize;
            for r in 0..reps {
                let byte = [0xFFu8, 0x7F, 0x00, 0xA5][r % 4];
                let (co, cp) = run(gpu, &c, &bf, !control, byte);
                let cv = out_view(&c, &co);
                let out_ok = cv.iter().zip(&ref_out).all(|(a, b)| a.to_bits() == b.to_bits());
                // Guards keep this launch's poison byte.
                let guard_ok = co[..GUARD * 4].iter().all(|&x| x == byte)
                    && co[(GUARD + c.b * QDIM) * 4..].iter().all(|&x| x == byte);
                // Written partials equal the reference's (gfx1151: the S-tile
                // scratch may hold anything); the rest keeps the poison.
                let part_ok = c.wmma.is_some() || ranges.iter().all(|&(s, e)| cp[s..e] == rp[s..e]);
                let stray = cp.iter().zip(&covered).filter(|&(&x, &cov)| !cov && x != byte).count();
                let outb = co[GUARD * 4..(GUARD + c.b * QDIM) * 4].to_vec();
                let (first_ok, s_ok) = match &first {
                    None => {
                        first = Some((outb.clone(), if live_s.is_empty() { Vec::new() } else { cp.clone() }));
                        (true, true)
                    }
                    Some((f, fs)) => (*f == outb, live_s.iter().all(|&(s, e)| cp[s..e] == fs[s..e])),
                };
                let same = out_ok && guard_ok && part_ok && stray == 0 && first_ok && s_ok;
                launches += 1;
                if !same {
                    bad += 1;
                    if bad <= 4 {
                        let refb: Vec<u8> = ref_out.iter().flat_map(|x| x.to_ne_bytes()).collect();
                        let (f, fs) = first.as_ref().unwrap();
                        println!(
                            "STRESS_DIFF B={b} ctx={ctx} capture={capture} cap={cap} rep={r} byte={byte:#04x} {} {} \
                             guards_ok={guard_ok} partials_ok={part_ok} stray_partial_bytes={stray} {}",
                            describe_out("out_vs_ref", &outb, &refb),
                            describe_out("out_vs_first", &outb, f),
                            if live_s.is_empty() { "s_vs_first=n/a".to_string() } else { describe_s(&c, &live_s, &cp, fs) },
                        );
                    }
                }
            }
            println!(
                "STRESS kv={} ref={} B={} ctx={} capture={} cap={} reps={} mismatched={} {}{}",
                kv_name(kv), c.ref_name(), b, ctx, capture, cap, reps, bad, if bad == 0 { "PASS" } else { "FAIL" },
                if control { " (control: reference vs reference)" } else { "" }
            );
            mism += bad;
            ok &= bad == 0;
            bf.free(gpu);
        }
    }
    println!(
        "STRESS_SUMMARY launches={launches} mismatched_launches={mism} cu_mask={} pid={}{} {}",
        std::env::var("ROC_GLOBAL_CU_MASK").unwrap_or_else(|_| "none".into()),
        std::process::id(),
        if control { " control=reference" } else { "" },
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn soak(gpu: &mut Gpu, secs: u64) -> bool {
    let cap = 32768 + 256;
    let prod = Some(VerifyWmmaGeometry::default());
    let cases: Vec<Case> = if gpu.arch == "gfx1151" {
        vec![
            Case { kv: VerifyKv::Q8, b: 16, ctx: 32768, capture: true, cap, partial_rows: 16, rows: false, wmma: prod },
            Case { kv: VerifyKv::Q8, b: 4, ctx: 32768, capture: false, cap, partial_rows: 16, rows: false, wmma: prod },
        ]
    } else if gpu.arch == "gfx1100" {
        vec![
            Case { kv: VerifyKv::Q8, b: 16, ctx: 32768, capture: true, cap, partial_rows: 16, rows: false, wmma: None },
            Case { kv: VerifyKv::Q8, b: 16, ctx: 32768, capture: false, cap, partial_rows: 16, rows: true, wmma: None },
            Case { kv: VerifyKv::Q8, b: 4, ctx: 32768, capture: false, cap, partial_rows: 16, rows: true, wmma: None },
        ]
    } else {
        vec![
            Case { kv: VerifyKv::Fp8, b: 16, ctx: 32768, capture: true, cap, partial_rows: 16, rows: false, wmma: None },
            Case { kv: VerifyKv::Q8, b: 16, ctx: 32768, capture: true, cap, partial_rows: 16, rows: false, wmma: None },
        ]
    };
    let bufs: Vec<Bufs> = cases.iter().enumerate().map(|(i, c)| Bufs::new(gpu, c, 0x50A4 + i as u64)).collect();
    let refs: Vec<Vec<f32>> = cases
        .iter()
        .zip(&bufs)
        .map(|(c, bf)| {
            let (o, _) = run(gpu, c, bf, false, 0xA5);
            out_view(c, &o)
        })
        .collect();
    let t0 = Instant::now();
    let (mut launches, mut bad) = (0usize, 0usize);
    while t0.elapsed().as_secs() < secs {
        for (i, c) in cases.iter().enumerate() {
            for _ in 0..16 {
                launch(gpu, c, &bufs[i], true);
                launches += 1;
            }
            let o = bytes_of(gpu, &bufs[i].out_full);
            let cv = out_view(c, &o);
            if !cv.iter().zip(&refs[i]).all(|(a, b)| a.to_bits() == b.to_bits()) {
                bad += 1;
            }
        }
    }
    println!(
        "SOAK secs={} launches={} mismatched_checks={} {}",
        t0.elapsed().as_secs_f64(),
        launches,
        bad,
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    for bf in bufs {
        bf.free(gpu);
    }
    bad == 0
}

fn bench(gpu: &mut Gpu, kvs: &[VerifyKv]) {
    let wmma = (gpu.arch == "gfx1151").then(VerifyWmmaGeometry::default);
    let bs: &[usize] = if wmma.is_some() { &[1, 4, 16] } else { &[4, 16] };
    for &kv in kvs {
        for &b in bs {
            for &ctx in &[2048usize, 8192, 16384, 32768, 65536] {
                for &(capture, cap) in &[(false, ctx + 64), (true, 262144usize)] {
                    let c = Case { kv, b, ctx, capture, cap: cap.max(ctx + 64), partial_rows: 16, rows: false, wmma };
                    let bf = Bufs::new(gpu, &c, 0xBE4C);
                    let mut ms = [0f64; 2];
                    for (arm, cand) in [(0usize, false), (1, true)] {
                        ms[arm] = time_ms(gpu, &c, &bf, cand, 20);
                    }
                    println!(
                        "BENCH kv={} B={} ctx={} capture={} ref_ms={:.4} verify_attn_ms={:.4} speedup={:.2}",
                        kv_name(kv), b, ctx, capture, ms[0], ms[1], ms[0] / ms[1]
                    );
                    bf.free(gpu);
                }
            }
        }
    }
}

fn time_ms(gpu: &mut Gpu, c: &Case, bf: &Bufs, cand: bool, iters: usize) -> f64 {
    for _ in 0..3 {
        launch(gpu, c, bf, cand);
    }
    gpu.hip.device_synchronize().expect("sync");
    let t = Instant::now();
    for _ in 0..iters {
        launch(gpu, c, bf, cand);
    }
    gpu.hip.device_synchronize().expect("sync");
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// gfx1151: candidate wall-clock per launch geometry around the production
/// one (S-launch waves, every P.V walker, fragment packing), reference alongside.
fn sweep(gpu: &mut Gpu) {
    let d = VerifyWmmaGeometry::default();
    let mut geoms = vec![d];
    for qk_waves in [80usize, 320, 640] {
        geoms.push(VerifyWmmaGeometry { qk_waves, ..d });
    }
    for pv in VerifyWmmaPv::ALL {
        geoms.push(VerifyWmmaGeometry { pv: Some(pv), ..d });
    }
    geoms.push(VerifyWmmaGeometry { packed: false, ..d });
    for &b in &[1usize, 4, 16] {
        for &ctx in &[2048usize, 8192, 32768] {
            let base = Case {
                kv: VerifyKv::Q8, b, ctx, capture: false, cap: ctx + 64, partial_rows: 16, rows: false, wmma: Some(d),
            };
            let bf = Bufs::new(gpu, &base, 0xBE4C);
            let ref_ms = time_ms(gpu, &base, &bf, false, 10);
            for g in &geoms {
                let c = Case { wmma: Some(*g), ..base };
                let ms = time_ms(gpu, &c, &bf, true, 20);
                println!(
                    "SWEEP B={b} ctx={ctx} packed={} qk_waves={} pv={:?} ref_ms={ref_ms:.4} verify_attn_ms={ms:.4} \
                     speedup={:.2}",
                    g.packed, g.qk_waves, g.pv, ref_ms / ms
                );
            }
            bf.free(gpu);
        }
    }
}

/// Q8 eager verify above 4K runs the multi-row R4/R8 kernel
/// (`attention_flash_q8_0_rows_masked`) when admitted; compare it with
/// VerifyAttn on the same eager shapes. gfx1100 times its byte-identical
/// R4/R8 twin; gfx1201 times the tile twin (not byte-compared: different
/// arithmetic).
fn bench_rows(gpu: &mut Gpu) {
    let rows_twin = gpu.arch == "gfx1100";
    for &b in &[4usize, 16] {
        for &ctx in &[4096usize, 8192, 16384, 32768, 65536] {
            let c = Case { kv: VerifyKv::Q8, b, ctx, capture: false, cap: ctx + 64, partial_rows: 16, rows: rows_twin, wmma: None };
            let bf = Bufs::new(gpu, &c, 0xBE4C);
            let mut ms = [0f64; 2];
            for arm in 0..2 {
                let go = |gpu: &mut Gpu| {
                    if arm == 0 {
                        std::sync::Arc::make_mut(&mut gpu.flags).verify_attn = false;
                        assert!(gpu
                            .attention_flash_q8_0_rows_masked(
                                &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part,
                            )
                            .expect("rows launch"));
                    } else {
                        launch(gpu, &c, &bf, true);
                    }
                };
                for _ in 0..3 {
                    go(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                let iters = 20;
                let t = Instant::now();
                for _ in 0..iters {
                    go(gpu);
                }
                gpu.hip.device_synchronize().expect("sync");
                ms[arm] = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
            }
            println!(
                "BENCH_ROWS kv=q8 B={} ctx={} rows_r{}_ms={:.4} verify_attn_ms={:.4} speedup={:.2}",
                b, ctx, if b >= 8 { 8 } else { 4 }, ms[0], ms[1], ms[0] / ms[1]
            );
            bf.free(gpu);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("oracle");
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        matches!(gpu.arch.as_str(), "gfx1201" | "gfx1100" | "gfx1151"),
        "VerifyAttn oracle runs on gfx1201, gfx1100 or gfx1151, got {}",
        gpu.arch
    );
    println!("VERIFY_ATTN_ORACLE arch={} mode={mode}", gpu.arch);
    // fp8 KV (and its VerifyAttn twin) is gfx1201-only.
    let kvs = |arg: Option<&String>, gpu: &Gpu| -> Vec<VerifyKv> {
        let mut v = parse_kvs(arg.map(String::as_str));
        if gpu.arch != "gfx1201" {
            v.retain(|&k| k == VerifyKv::Q8);
        }
        v
    };
    let ok = match mode {
        "oracle" => {
            let k = kvs(args.get(2), &gpu);
            oracle(&mut gpu, &k)
        }
        "stress" => {
            let reps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
            let k = kvs(args.get(3), &gpu);
            stress(&mut gpu, reps, &k)
        }
        "soak" => soak(&mut gpu, args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60)),
        "bench" => {
            let k = kvs(args.get(2), &gpu);
            bench(&mut gpu, &k);
            true
        }
        "sweep" if gpu.arch == "gfx1151" => {
            sweep(&mut gpu);
            true
        }
        "bench_rows" if gpu.arch != "gfx1151" => {
            bench_rows(&mut gpu);
            true
        }
        m => panic!("unknown mode {m}"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
