// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GPU identity oracle for the native-fp8 GQA decode and VerifyAttn kernels at
// the A3B shape (H16/KV2, GQA group 8, head_dim 256) and, as the "27B
// unchanged" regression, the H2 shape (H24/KV4, group 6). gfx1201 only.
//
// No tolerance anywhere: every comparison is byte equality over the WHOLE
// output and partials allocations (poisoned before each launch, with guard
// regions around the views the launch receives).
// Modes:
//   verify [--shape a3b|h2|all]
//       `Gpu::try_attention_verify_gqa` (fp8) against
//       `attention_flash_fp8_e4m3_tile_batched` + `attention_flash_asym_reduce_batched`
//       (verify_attn flag off), in one process. B 1..32 (all small MTP/DFlash
//       row counts, ragged row groups of 6 and 8), ctx {1,127,128,129,2K,8K}
//       and 32K/64K for B {1,6,16}, eager and graph-capture max_ctx_len, the
//       262144-cap capture grid, sub-batched partials (chunk boundaries inside
//       row groups), and non-consecutive row positions (rollback replays,
//       tile-edge rows, descending, repeated).
//   stress <reps>
//       poisoned candidate repeats (rotating 0xFF/0x7F/0x00/0xA5), each compared
//       with the reference run under the same poison. Never set
//       ROC_GLOBAL_CU_MASK or HSA_CU_MASK on RDNA.
//   decode [--shape a3b|h2|all]
//       prints one DECODE line (FNV-1a of the whole out and partials
//       allocations) per case for whatever route `Gpu::attention_flash_fp8_e4m3_tile`
//       takes. The decode flag is a process snapshot, so the reference is a second
//       process:
//         a3b_gqa8_oracle decode --dump-dir cand
//         HIPFIRE_FP8_DECODE_ATTN_GQA=0 a3b_gqa8_oracle decode --dump-dir ref
//         diff -rq cand ref             # compares every byte, including guards
//       Directories must be new; hashes alone are not an identity gate.
//       Which pair ran is visible with HIPFIRE_MODULE_LOAD_AUDIT=1 (the candidate
//       run loads attention_flash_fp8_e4m3_tile_gqa8_gfx1201 + the dsplit reduce,
//       the reference run attention_flash_fp8_e4m3_tile + attention_flash_q8_0_reduce).
//       Cases: ctx 1..70000 (tile-tail and tile-boundary lengths, short, 8K,
//       > 512 tiles so the grid-y loop wraps), max_seq == ctx, ctx + 129 and the
//       262144 capture layout.
//   bench
//       wall-clock per call: decode (the env-selected route) and verify
//       (reference vs candidate). Run decode twice as above for the A/B.
//
// Exit 0 = pass (decode: all cases ran at tile 128).
// Run: cargo run --release -p rdna-compute --example a3b_gqa8_oracle -- verify

use rdna_compute::attention::{q8_flash_tile_size, VerifyKv};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const HD: usize = 256;
const TILE: usize = 128;
const STRIDE: usize = 2 + HD;
/// Guard elements on each side of the output view and after the partials view.
const GUARD: usize = 4096;
const POS_GUARD: i32 = 0x3fff_ffff;
const VERIFY_CAP: usize = 65536 + 256;

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    nh: usize,
    nkv: usize,
}
const A3B: Shape = Shape {
    name: "a3b-h16kv2",
    nh: 16,
    nkv: 2,
};
const H2: Shape = Shape {
    name: "h2-h24kv4",
    nh: 24,
    nkv: 4,
};

impl Shape {
    fn qdim(&self) -> usize {
        self.nh * HD
    }
    /// fp8 native KV row: nkv x 256 codes, then nkv f16 scales.
    fn row_bytes(&self) -> usize {
        self.nkv * (HD + 2)
    }
}

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
        let r = if rem > mid || (rem == mid && (half & 1) == 1) {
            half + 1
        } else {
            half
        };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half + 1
    } else {
        half
    };
    sign | r as u16
}

/// Head scale: mostly positive normal f16 in [2^-9, 2^-4], with some negative,
/// zero and subnormal scales to pin signed-zero and denormal paths.
fn scale_f16(rng: &mut Rng) -> u16 {
    match rng.below(64) {
        0 => 0x0000,
        1 => 0x8000,
        2 => 0x0001 + rng.below(0x3ff) as u16,
        3 | 4 => half_from_f32(-(0.002 + 0.06 * rng.unit())),
        _ => half_from_f32(0.002 + 0.06 * rng.unit()),
    }
}

/// K or V cache: `valid` random rows (no NaN codes), then NaN-poisoned rows up
/// to `cap`, so a read past the causal bound shows up as a byte difference.
fn fill_kv(s: Shape, rng: &mut Rng, valid: usize, cap: usize) -> Vec<u8> {
    let rb = s.row_bytes();
    let codes = s.nkv * HD;
    let mut buf = vec![0u8; cap * rb];
    for t in 0..cap {
        let row = &mut buf[t * rb..(t + 1) * rb];
        if t >= valid {
            row[..codes].fill(0x7f);
            for h in 0..s.nkv {
                row[codes + 2 * h..codes + 2 * h + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
            }
            continue;
        }
        for chunk in row[..codes].chunks_mut(8) {
            let word = rng.next().to_le_bytes();
            for (dst, &b) in chunk.iter_mut().zip(word.iter()) {
                // E4M3FN NaN is S.1111.111: keep every code finite.
                *dst = if b & 0x7f == 0x7f {
                    (b & 0x80) | 0x3a
                } else {
                    b
                };
            }
        }
        for h in 0..s.nkv {
            let sc = scale_f16(rng);
            row[codes + 2 * h..codes + 2 * h + 2].copy_from_slice(&sc.to_le_bytes());
        }
    }
    buf
}

fn query(s: Shape, rng: &mut Rng, rows: usize) -> Vec<f32> {
    (0..rows * s.qdim())
        .map(|i| {
            // Rows (and heads) alternate between ordinary and 12x-scaled queries
            // so some tiles run their softmax deep into expf underflow.
            let amp = if (i / HD) % 3 == 2 { 12.0 } else { 1.0 };
            (rng.unit() * 2.0 - 1.0) * amp
        })
        .collect()
}

fn poison(gpu: &Gpu, t: &GpuTensor, byte: u8) {
    gpu.hip
        .memset(&t.buf, byte as i32, t.buf.size())
        .expect("memset");
}

fn bytes_of(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let mut v = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("dtoh");
    v
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or(if a.len() != b.len() {
            Some(a.len().min(b.len()))
        } else {
            None
        })
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn parse_shapes(args: &[String]) -> Vec<Shape> {
    let which = args
        .iter()
        .position(|a| a == "--shape")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    match which {
        Some("a3b") => vec![A3B],
        Some("h2") => vec![H2],
        None | Some("all") => vec![A3B, H2],
        Some(other) => panic!("unknown --shape {other} (a3b|h2|all)"),
    }
}

// ───────────────────────────── VerifyAttn ─────────────────────────────

/// Row position patterns (`start` = ctx - 1).
#[derive(Clone, Copy, Debug)]
enum Pat {
    /// start + i: the ordinary verify block.
    Consecutive,
    /// start + (i % 3): positions repeat and step back (rollback replays).
    Rollback,
    /// start + 128 * (i % 3) - (i % 2): rows alternating either side of tile edges.
    TileEdge,
    /// start + b - 1 - i.
    Descending,
    /// every row at start.
    Repeat,
}

#[derive(Clone, Debug)]
struct VCase {
    shape: Shape,
    pat: Pat,
    b: usize,
    ctx: usize,
    /// Graph-capture shape: `max_ctx_len = cap` instead of the live extent.
    capture: bool,
    cap: usize,
    /// Partials capacity in rows (at this case's max_tiles); < b sub-batches.
    partial_rows: usize,
}

impl VCase {
    fn new(
        shape: Shape,
        pat: Pat,
        b: usize,
        ctx: usize,
        capture: bool,
        cap: usize,
        partial_rows: usize,
    ) -> VCase {
        let mut c = VCase {
            shape,
            pat,
            b,
            ctx,
            capture,
            cap,
            partial_rows,
        };
        c.cap = cap.max(c.valid() + 64);
        c
    }
    fn positions(&self) -> Vec<i32> {
        let s = self.ctx as i64 - 1;
        (0..self.b)
            .map(|i| {
                let i = i as i64;
                let p = match self.pat {
                    Pat::Consecutive => s + i,
                    Pat::Rollback => s + (i % 3),
                    Pat::TileEdge => s + 128 * (i % 3) - (i % 2),
                    Pat::Descending => s + self.b as i64 - 1 - i,
                    Pat::Repeat => s,
                };
                p.max(0) as i32
            })
            .collect()
    }
    fn valid(&self) -> usize {
        self.positions().into_iter().max().unwrap_or(0) as usize + 1
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
        self.partial_rows * self.shape.nh * self.max_tiles() * STRIDE
    }
}

struct VBufs {
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

impl VBufs {
    fn new(gpu: &mut Gpu, c: &VCase, seed: u64) -> VBufs {
        let s = c.shape;
        let mut rng = Rng(seed);
        let mut qh = query(s, &mut rng, c.b);
        qh.extend(std::iter::repeat(f32::NAN).take(GUARD));
        let q_full = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
        let q = q_full.sub_offset(0, c.b * s.qdim());
        let kh = fill_kv(s, &mut rng, c.valid(), c.cap);
        let vh = fill_kv(s, &mut rng, c.valid(), c.cap);
        let k = gpu.upload_raw(&kh, &[kh.len()]).expect("k");
        let v = gpu.upload_raw(&vh, &[vh.len()]).expect("v");
        let mut posh = c.positions();
        posh.extend(std::iter::repeat(POS_GUARD).take(8));
        let pb: Vec<u8> = posh.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let pos = gpu.upload_raw(&pb, &[posh.len()]).expect("pos");
        let out_full = gpu
            .zeros(&[c.b * s.qdim() + 2 * GUARD], DType::F32)
            .expect("out");
        let out = out_full.sub_offset(GUARD, c.b * s.qdim());
        let part_full = gpu
            .zeros(&[c.partials_len() + GUARD], DType::F32)
            .expect("partials");
        let part = part_full.sub_offset(0, c.partials_len());
        VBufs {
            q,
            q_full,
            k,
            v,
            pos,
            out,
            out_full,
            part,
            part_full,
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [
            self.q_full,
            self.k,
            self.v,
            self.pos,
            self.out_full,
            self.part_full,
        ] {
            let _ = gpu.free_tensor(t);
        }
        let _ = (self.q, self.out, self.part);
    }
}

/// One launch of the per-head reference (`candidate == false`) or VerifyAttn.
fn vlaunch(gpu: &mut Gpu, c: &VCase, bf: &VBufs, candidate: bool) {
    let s = c.shape;
    std::sync::Arc::make_mut(&mut gpu.flags).verify_attn = candidate;
    if candidate {
        let ran = gpu
            .try_attention_verify_gqa(
                VerifyKv::Fp8,
                &bf.q,
                &bf.k,
                &bf.v,
                &bf.out,
                &bf.pos,
                s.nh,
                s.nkv,
                HD,
                c.max_ctx_len(),
                c.b,
                &bf.part,
                None,
                0,
                None,
            )
            .expect("verify_attn launch");
        assert!(ran, "VerifyAttn declined an admitted case: {c:?}");
    } else {
        gpu.attention_flash_fp8_e4m3_tile_batched(
            &bf.q,
            &bf.k,
            &bf.v,
            &bf.out,
            &bf.pos,
            s.nh,
            s.nkv,
            HD,
            c.cap,
            c.max_ctx_len(),
            c.b,
            &bf.part,
            None,
            0,
            0,
            None,
        )
        .expect("reference fp8");
    }
}

/// Poison both allocations with `byte`, launch, return (out_full, part_full).
fn vrun(gpu: &mut Gpu, c: &VCase, bf: &VBufs, candidate: bool, byte: u8) -> (Vec<u8>, Vec<u8>) {
    poison(gpu, &bf.out_full, byte);
    poison(gpu, &bf.part_full, byte);
    vlaunch(gpu, c, bf, candidate);
    (bytes_of(gpu, &bf.out_full), bytes_of(gpu, &bf.part_full))
}

fn verify_cases(shapes: &[Shape]) -> Vec<VCase> {
    let mut cases = Vec::new();
    for &shape in shapes {
        let mk = |pat, b: usize, ctx, capture, partial_rows| {
            VCase::new(
                shape,
                pat,
                b,
                ctx,
                capture,
                if capture { VERIFY_CAP } else { 0 },
                partial_rows,
            )
        };
        // Every small row count (MTP K+1, DFlash B, n-gram), row groups of 6 and
        // 8 (ragged and full), up to the 32-row ceiling.
        for &b in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 32] {
            for &ctx in &[1usize, 127, 128, 129, 2048, 8192] {
                for &capture in &[false, true] {
                    cases.push(mk(Pat::Consecutive, b, ctx, capture, b.min(16)));
                }
            }
        }
        // Long contexts: many tiles per split (grid-y wraps), eager.
        for &b in &[1usize, 6, 16] {
            for &ctx in &[32768usize, 65536] {
                cases.push(mk(Pat::Consecutive, b, ctx, false, b.min(16)));
            }
        }
        // Non-consecutive row positions: rollback replays, tile edges, descending, repeated.
        for pat in [Pat::Rollback, Pat::TileEdge, Pat::Descending, Pat::Repeat] {
            for &b in &[4usize, 5, 6, 7, 9, 13, 16] {
                for &ctx in &[300usize, 8192] {
                    cases.push(mk(pat, b, ctx, false, b.min(16)));
                    cases.push(mk(pat, b, ctx, true, b.min(16)));
                }
            }
        }
        // Sub-batched partials: chunk boundaries inside 6- and 8-row groups.
        for &(b, ctx, pr) in &[
            (16usize, 8191usize, 5usize),
            (13, 300, 4),
            (32, 3000, 7),
            (12, 129, 6),
            (7, 2048, 6),
            (32, 8192, 32),
        ] {
            cases.push(mk(Pat::Consecutive, b, ctx, false, pr));
            cases.push(mk(Pat::Consecutive, b, ctx, true, pr));
        }
        // The real graph-capture shape: physical cap 262144 (2048 partial tiles).
        for &(b, ctx) in &[(16usize, 2048usize), (4, 20000), (6, 9000), (1, 129)] {
            cases.push(VCase::new(
                shape,
                Pat::Consecutive,
                b,
                ctx,
                true,
                262144,
                16,
            ));
        }
    }
    cases
}

fn verify(gpu: &mut Gpu, shapes: &[Shape]) -> bool {
    let mut ok = true;
    let mut n = 0;
    for (i, c) in verify_cases(shapes).iter().enumerate() {
        let bf = VBufs::new(gpu, c, 0x5eed_0000 + i as u64);
        let byte = [0xA5u8, 0x00][i % 2];
        let (ro, rp) = vrun(gpu, c, &bf, false, byte);
        let (co, cp) = vrun(gpu, c, &bf, true, byte);
        let od = first_diff(&ro, &co);
        let pd = first_diff(&rp, &cp);
        let out0 = GUARD * 4;
        let nonfinite = ro[out0..out0 + c.b * c.shape.qdim() * 4]
            .chunks_exact(4)
            .filter(|w| !f32::from_ne_bytes([w[0], w[1], w[2], w[3]]).is_finite())
            .count();
        let pass = od.is_none() && pd.is_none() && nonfinite == 0;
        println!(
            "VERIFY shape={} pat={:?} B={} ctx={} capture={} cap={} max_tiles={} partial_rows={} out_bytes={} \
             part_bytes={} ref_nonfinite={nonfinite} out_first_diff={od:?} part_first_diff={pd:?} {}",
            c.shape.name, c.pat, c.b, c.ctx, c.capture, c.cap, c.max_tiles(), c.partial_rows, ro.len(), rp.len(),
            if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
        n += 1;
        bf.free(gpu);
    }
    println!(
        "VERIFY_SUMMARY cases={n} {}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn stress(gpu: &mut Gpu, reps: usize, shapes: &[Shape]) -> bool {
    let mut ok = true;
    for &shape in shapes {
        for &(b, ctx, capture) in &[
            (1usize, 129usize, false),
            (5, 8192, false),
            (6, 8192, true),
            (9, 2048, false),
            (16, 8192, true),
        ] {
            let c = VCase::new(
                shape,
                Pat::Consecutive,
                b,
                ctx,
                capture,
                if capture { VERIFY_CAP } else { 0 },
                b.min(16),
            );
            let bf = VBufs::new(gpu, &c, 0x57e5_5000 + (b * 131 + ctx) as u64);
            let mut bad = 0;
            for rep in 0..reps {
                let byte = [0xFFu8, 0x7F, 0x00, 0xA5][rep % 4];
                let (ro, rp) = vrun(gpu, &c, &bf, false, byte);
                let (co, cp) = vrun(gpu, &c, &bf, true, byte);
                if ro != co || rp != cp {
                    bad += 1;
                    println!("STRESS_DIFF shape={} B={b} ctx={ctx} rep={rep} out_diff={:?} part_diff={:?}", shape.name, first_diff(&ro, &co), first_diff(&rp, &cp));
                }
            }
            println!(
                "STRESS shape={} B={b} ctx={ctx} capture={capture} reps={reps} bad={bad} {}",
                shape.name,
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            ok &= bad == 0;
            bf.free(gpu);
        }
    }
    ok
}

// ───────────────────────────── decode ─────────────────────────────

/// (ctx = seq_len, max_seq, mode): mode 0 plain, 1 output gate, 2 gate + AWQ scale.
fn decode_cases() -> Vec<(usize, usize, u8)> {
    let mut cases = Vec::new();
    for &ctx in &[
        1usize, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1000, 2047, 2048, 2049, 4095, 4096,
        4097, 8191, 8192, 8193, 16384, 32767, 65536, 65537, 70000,
    ] {
        cases.push((ctx, ctx, 0));
        cases.push((ctx, ctx + 129, 0));
    }
    // Graph-capture layout: partials sized for the 262144-position cap.
    for &ctx in &[1usize, 128, 8192, 20000] {
        cases.push((ctx, 262144, 0));
    }
    // The gated epilogue route (A3B): short, tile edges, 8K, > 512 tiles.
    for &mode in &[1u8, 2] {
        for &ctx in &[1usize, 127, 128, 129, 2048, 8193, 65537] {
            cases.push((ctx, ctx + 129, mode));
        }
        cases.push((8192, 262144, mode));
    }
    cases
}

fn decode(gpu: &mut Gpu, shapes: &[Shape], dump_dir: &std::path::Path) -> bool {
    let mut ok = true;
    println!("DECODE_ORACLE arch={}", gpu.arch);
    for &s in shapes {
        for (n, &(ctx, max_seq, mode)) in decode_cases().iter().enumerate() {
            let tile = q8_flash_tile_size(&gpu.arch, s.nh, s.nkv, HD, max_seq);
            if tile != TILE {
                println!(
                    "DECODE shape={} ctx={ctx} max_seq={max_seq} NOT_ADMITTED tile={tile}",
                    s.name
                );
                ok = false;
                continue;
            }
            let mut rng = Rng(0xDEC0_0000 + n as u64 * 977 + s.nh as u64);
            let qh = query(s, &mut rng, 1);
            let q = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
            let kh = fill_kv(s, &mut rng, ctx, max_seq);
            let vh = fill_kv(s, &mut rng, ctx, max_seq);
            let k = gpu.upload_raw(&kh, &[kh.len()]).expect("k");
            let v = gpu.upload_raw(&vh, &[vh.len()]).expect("v");
            let pb = ((ctx - 1) as i32).to_ne_bytes().to_vec();
            let pos = gpu.upload_raw(&pb, &[1]).expect("pos");
            let max_tiles = max_seq.div_ceil(TILE);
            let part_len = s.nh * max_tiles * STRIDE;
            let out_full = gpu.zeros(&[s.qdim() + 2 * GUARD], DType::F32).expect("out");
            let out = out_full.sub_offset(GUARD, s.qdim());
            let part_full = gpu
                .zeros(&[part_len + GUARD], DType::F32)
                .expect("partials");
            let part = part_full.sub_offset(0, part_len);
            poison(gpu, &out_full, 0xA5);
            poison(gpu, &part_full, 0xA5);
            // The gated epilogue (A3B on gfx1201: gate + MQ rotation, optionally
            // the AWQ scale) consumes the tile's partials instead of the dsplit
            // reduce; `out` stays [n_heads * 256] f32.
            let gate = (mode >= 1).then(|| {
                let gh: Vec<f32> = (0..s.qdim()).map(|_| rng.unit() * 4.0 - 2.0).collect();
                gpu.upload_f32(&gh, &[gh.len()]).expect("gate")
            });
            let awq = (mode >= 2).then(|| {
                let ah: Vec<f32> = (0..s.qdim()).map(|_| 0.5 + rng.unit()).collect();
                gpu.upload_f32(&ah, &[ah.len()]).expect("awq")
            });
            gpu.attention_flash_fp8_e4m3_tile(
                &q,
                &k,
                &v,
                &out,
                &pos.buf,
                ctx,
                s.nh,
                s.nkv,
                HD,
                max_seq,
                &part,
                gate.as_ref(),
                awq.as_ref(),
            )
            .expect("decode launch");
            let ob = bytes_of(gpu, &out_full);
            let pbts = bytes_of(gpu, &part_full);
            let nonfinite = ob[GUARD * 4..(GUARD + s.qdim()) * 4]
                .chunks_exact(4)
                .filter(|w| !f32::from_ne_bytes([w[0], w[1], w[2], w[3]]).is_finite())
                .count();
            if nonfinite != 0 {
                ok = false;
            }
            // Full bytes, not hashes, are the identity evidence. The printed
            // hashes are only diagnostics; compare both directories with diff -rq.
            let stem = format!("{}-ctx{ctx}-max{max_seq}-mode{mode}", s.name);
            std::fs::write(dump_dir.join(format!("{stem}.out.bin")), &ob).expect("write output");
            std::fs::write(dump_dir.join(format!("{stem}.partials.bin")), &pbts)
                .expect("write partials");
            println!(
                "DECODE shape={} mode={} ctx={ctx} max_seq={max_seq} out={:016x} part={:016x} nonfinite={nonfinite}",
                s.name,
                ["plain", "gate", "gate+awq"][mode as usize],
                fnv64(&ob),
                fnv64(&pbts)
            );
            for t in [q, k, v, pos, out_full, part_full] {
                let _ = gpu.free_tensor(t);
            }
            if let Some(g) = gate {
                let _ = gpu.free_tensor(g);
            }
            if let Some(a) = awq {
                let _ = gpu.free_tensor(a);
            }
            let _ = (out, part);
        }
    }
    ok
}

fn time_ms(gpu: &mut Gpu, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..5 {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    let t = Instant::now();
    for _ in 0..iters {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn bench(gpu: &mut Gpu, shapes: &[Shape]) {
    for &s in shapes {
        for &ctx in &[512usize, 8192, 32768] {
            let max_seq = ctx + 129;
            let mut rng = Rng(0xBE4C + ctx as u64);
            let qh = query(s, &mut rng, 1);
            let q = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
            let kh = fill_kv(s, &mut rng, ctx, max_seq);
            let vh = fill_kv(s, &mut rng, ctx, max_seq);
            let k = gpu.upload_raw(&kh, &[kh.len()]).expect("k");
            let v = gpu.upload_raw(&vh, &[vh.len()]).expect("v");
            let pos = gpu
                .upload_raw(&((ctx - 1) as i32).to_ne_bytes(), &[1])
                .expect("pos");
            let out = gpu.zeros(&[s.qdim()], DType::F32).expect("out");
            let part = gpu
                .zeros(&[s.nh * max_seq.div_ceil(TILE) * STRIDE], DType::F32)
                .expect("partials");
            let ms = time_ms(gpu, 200, |g| {
                g.attention_flash_fp8_e4m3_tile(
                    &q, &k, &v, &out, &pos.buf, ctx, s.nh, s.nkv, HD, max_seq, &part, None, None,
                )
                .expect("decode")
            });
            println!("BENCH_DECODE shape={} ctx={ctx} ms_per_call={ms:.4} (route = env HIPFIRE_FP8_DECODE_ATTN_GQA)", s.name);
            for t in [q, k, v, pos, out, part] {
                let _ = gpu.free_tensor(t);
            }
        }
        for &b in &[1usize, 2, 4, 5, 6, 8, 16] {
            for &ctx in &[512usize, 8192] {
                for &capture in &[false, true] {
                    let c = VCase::new(
                        s,
                        Pat::Consecutive,
                        b,
                        ctx,
                        capture,
                        if capture { 262144 } else { 0 },
                        16,
                    );
                    let bf = VBufs::new(gpu, &c, 0xBE4D);
                    let mut ms = [0f64; 2];
                    for (arm, cand) in [(0usize, false), (1, true)] {
                        ms[arm] = time_ms(gpu, 50, |g| vlaunch(g, &c, &bf, cand));
                    }
                    println!(
                        "BENCH_VERIFY shape={} B={b} ctx={ctx} capture={capture} ref_ms={:.4} verify_attn_ms={:.4} speedup={:.2}",
                        s.name, ms[0], ms[1], ms[0] / ms[1]
                    );
                    bf.free(gpu);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("verify");
    // The fp8 reference is the VerifyAttn tile + reduce. The native-fp8
    // split-KV verifier (default on from 8,192 tokens on gfx1201, H24/KV4)
    // would replace it at long contexts, so pin it off before the process
    // config snapshot is first read.
    std::env::set_var("HIPFIRE_GFX12_FA2_SPLIT_VERIFY_FP8", "0");
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        gpu.arch == "gfx1201",
        "a3b_gqa8_oracle runs on gfx1201, got {}",
        gpu.arch
    );
    let shapes = parse_shapes(&args);
    println!("A3B_GQA8_ORACLE arch={} mode={mode}", gpu.arch);
    let ok = match mode {
        "verify" => verify(&mut gpu, &shapes),
        "stress" => {
            let reps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            stress(&mut gpu, reps, &shapes)
        }
        "decode" => {
            let dir = args
                .iter()
                .position(|a| a == "--dump-dir")
                .and_then(|i| args.get(i + 1))
                .expect("decode requires --dump-dir <new-directory>");
            let dir = std::path::Path::new(dir);
            std::fs::create_dir(dir).expect("dump directory must not already exist");
            decode(&mut gpu, &shapes, dir)
        }
        "bench" => {
            bench(&mut gpu, &shapes);
            true
        }
        m => panic!("unknown mode {m} (verify|stress|decode|bench)"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
