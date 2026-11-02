// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bit-exact contraction gate for the MQ4G256V2 expanded MoE-down producer
//! (`gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded`).
//!
//! WHY. The shipped fused decode path (`gemv_mq4g256v2_moe_ninepath_d4`)
//! evaluates each 8-weight group dot as
//! `fma(d7,x7, ... fma(d2,x2, fma(d0,x0, round(d1*x1))) ...)`: the first
//! product is `d1*x1`, separately rounded, and `d0*x0` is FUSED into it. The
//! expanded producer used to be written as one `a += d0*x0 + d1*x1 + ...`
//! expression and left to `-ffp-contract`; the compiler rounded `d0*x0` and
//! fused `d1*x1` instead (an audited ISA difference in K512 group 1), so
//! expanded/EP outputs differed from D4 in the last bit. The kernel now pins the
//! chain with explicit `__fmul_rn` / `__fmaf_rn`. The orientation is a
//! compiler choice per call site, so a probe kernel cannot reproduce it; this
//! test runs the REAL kernel through the production launcher and compares raw
//! f32 bits to a CPU model of the pinned chain.
//!
//! The CPU model is exactly the kernel contract:
//!   * dequant `d_i = fmaf(sc, (f32)q_i, zp)` (f16 header decoded exactly;
//!     lanes 0..15 use pair 0, lanes 16..31 pair 1; nibble i = `(pk>>4i)&15`),
//!   * `sum = d1*x1` (rounded); `sum = fmaf(d0,x0,sum)`; then `d2..d7`
//!     ascending `fmaf`,
//!   * `acc += sum` per group (never seeded from `acc`), K512: `acc0+acc1`,
//!   * the `__shfl_down` tree 16,8,4,2,1 into lane 0.
//! Rust `f32` ops are never contracted and `f32::mul_add` is a true fused
//! multiply-add, so the model is the unambiguous reference.
//!
//! Failure semantics:
//!   * Pre-fix kernel (compiler-chosen contraction): the pinned-chain
//!     assertion FAILS with mismatching raw bits, reported per `row % 4`
//!     (the audited divergence is rows 0/1/2 of K512 group 1).
//!   * Fixed kernel: the pinned-chain assertion passes with ZERO mismatches
//!     for K512 (unrolled two-group body) and K768 (general group loop).
//!   * The same inputs under the OPPOSITE orientation (`d0*x0` rounded
//!     first) MUST differ from the kernel bits in many outputs: that proves
//!     the fixture is contraction-sensitive, so a pass is not vacuous.
//!
//! `#[ignore]`d: needs an RDNA wave32 GPU with a working HIP toolchain; no
//! model files. Developer-only:
//!
//!   cargo test -p rdna-compute --release --test moe_down_mq4v2_contraction_bits -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};

const M: usize = 64; // 16 workgroups x 4 rows: every row%4 position is exercised
const N_EXP: usize = 2;
const K_TOP: usize = 8;
const BATCH: usize = 2;
/// Opposite orientation must disagree with the kernel on at least this
/// fraction of outputs (measured ~48% when every group flips; require 1/8).
const SENSITIVITY_DIVISOR: usize = 8;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Full-mantissa f32 in [-1, 1).
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF; // 23 mantissa bits
        let sign = self.next_u32() & 1;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0; // [0,1) with 23 random bits
        if sign == 1 {
            -v
        } else {
            v
        }
    }
    /// Normal f16 bit pattern with exponent in `lo..=hi` (biased).
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

/// Exact f16 -> f32 for the normal numbers this fixture generates.
fn f16_normal_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1F) as u32,
        (h & 0x3FF) as u32,
    );
    assert!(exp != 0 && exp != 0x1F, "fixture emits normal f16 only");
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// MQ4G256V2 expert: `m` rows x `k/256` groups of 136 B
/// `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][32 x u32 nibbles]`.
fn synth_mq4v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    let mut rng = Lcg(seed);
    for row in 0..m {
        for g in 0..groups {
            let off = (row * groups + g) * 136;
            for half in 0..2 {
                let sc = rng.f16_normal(8, 13, false); // 2^-7 .. 2^-2
                let zp = rng.f16_normal(8, 14, true); // +/- 2^-7 .. 2^-1
                out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
                out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
            }
            for w in 0..32 {
                let pk = rng.next_u32() ^ (rng.next_u32() << 16);
                out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

/// One lane's group dot. `d1_first` = D4 orientation: round `d1*x1`, then FMA
/// `d0*x0` into it. The opposite (`!d1_first`) is the compiler-contracted
/// orientation the fix removed.
fn group_sum(d: &[f32; 8], x: &[f32; 8], d1_first: bool) -> f32 {
    let mut s = if d1_first {
        let p = d[1] * x[1];
        d[0].mul_add(x[0], p)
    } else {
        let p = d[0] * x[0];
        d[1].mul_add(x[1], p)
    };
    for i in 2..8 {
        s = d[i].mul_add(x[i], s);
    }
    s
}

/// One output row of one (token, krank): lane accumulation + shfl_down tree.
fn row_dot(packed: &[u8], row: usize, groups: usize, x: &[f32], d1_first: bool) -> f32 {
    let mut lane = [0f32; 32];
    for (tid, slot) in lane.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..groups {
            let off = (row * groups + g) * 136;
            let hdr = off + 4 * (tid / 16);
            let sc = f16_normal_to_f32(u16::from_le_bytes([packed[hdr], packed[hdr + 1]]));
            let zp = f16_normal_to_f32(u16::from_le_bytes([packed[hdr + 2], packed[hdr + 3]]));
            let p = off + 8 + 4 * tid;
            let pk = u32::from_le_bytes(packed[p..p + 4].try_into().unwrap());
            let mut d = [0f32; 8];
            for (i, di) in d.iter_mut().enumerate() {
                *di = sc.mul_add(((pk >> (4 * i)) & 0xF) as f32, zp);
            }
            let xb = g * 256 + tid * 8;
            let xg: [f32; 8] = x[xb..xb + 8].try_into().unwrap();
            // `(a) += sum`: separate add; K512 `acc0 + acc1` is the same value
            // as this sequential form for two groups.
            acc += group_sum(&d, &xg, d1_first);
        }
        *slot = acc;
    }
    for off in [16usize, 8, 4, 2, 1] {
        let old = lane;
        for l in 0..32 {
            let src = if l + off < 32 { old[l + off] } else { old[l] };
            lane[l] = old[l] + src;
        }
    }
    lane[0]
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

#[test]
#[ignore]
fn mq4v2_expanded_down_matches_pinned_d4_chain_bits() {
    let mut gpu = Gpu::init().expect("GPU required for this developer-only bit gate");
    if !gpu.arch_caps.is_wave32() {
        eprintln!("SKIP — requires an RDNA wave32 device.");
        return;
    }

    // K512 = unrolled two-group body (the audited divergence); K768 = general
    // sequential group loop. Both bodies share the DOG_X8 macro.
    for k in [512usize, 768usize] {
        let groups = k / 256;

        let mut hosts: Vec<Vec<u8>> = Vec::with_capacity(N_EXP);
        let mut keep: Vec<GpuTensor> = Vec::with_capacity(N_EXP);
        let mut ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
        for e in 0..N_EXP {
            let packed = synth_mq4v2(M, k, 0x4D51_3456 ^ ((k as u64) << 8) ^ e as u64);
            let t = upload_u8(&mut gpu, &packed);
            ptrs.push(t.buf.as_ptr() as u64);
            keep.push(t);
            hosts.push(packed);
        }
        let topk: Vec<i32> = (0..BATCH * K_TOP).map(|i| (i % N_EXP) as i32).collect();
        let mut rng = Lcg(0xC0FF_EE00 ^ k as u64);
        let rot: Vec<f32> = (0..BATCH * K_TOP * k).map(|_| rng.f32_signed()).collect();

        let expert_ptrs = upload_u64(&mut gpu, &ptrs);
        let topk_t = upload_i32(&mut gpu, &topk);
        let rot_t = upload_f32(&mut gpu, &rot);
        let out_t = upload_f32(&mut gpu, &vec![0.0f32; BATCH * K_TOP * M]);

        gpu.gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded(
            &expert_ptrs,
            &topk_t,
            &rot_t,
            &out_t,
            M,
            k,
            K_TOP,
            BATCH,
        )
        .expect("expanded down launch");
        gpu.hip.device_synchronize().expect("sync");
        let got = download_f32(&gpu, &out_t, BATCH * K_TOP * M);

        let mut pinned_bad = [0usize; 4];
        let mut first_bad: Option<(usize, u32, u32)> = None;
        let mut opposite_diff = 0usize;
        for slot in 0..BATCH * K_TOP {
            let blob = &hosts[topk[slot] as usize];
            let x = &rot[slot * k..(slot + 1) * k];
            for row in 0..M {
                let idx = slot * M + row;
                let want = row_dot(blob, row, groups, x, true);
                let alt = row_dot(blob, row, groups, x, false);
                if got[idx].to_bits() != want.to_bits() {
                    pinned_bad[row % 4] += 1;
                    first_bad.get_or_insert((idx, got[idx].to_bits(), want.to_bits()));
                }
                if got[idx].to_bits() != alt.to_bits() {
                    opposite_diff += 1;
                }
            }
        }

        // Fixture sensitivity first: if the opposite orientation does not
        // move the bits, the equality below proves nothing.
        let total = BATCH * K_TOP * M;
        assert!(
            opposite_diff >= total / SENSITIVITY_DIVISOR,
            "[K={k}] fixture is not contraction-sensitive: opposite orientation differs in only \
             {opposite_diff}/{total} outputs"
        );
        assert_eq!(
            pinned_bad,
            [0; 4],
            "[K={k}] expanded MQ4V2 down kernel bits differ from the pinned D4 chain \
             (round(d1*x1) then fma(d0,x0), d2..d7 ascending fma, `a += sum`): mismatches per \
             row%4 = {pinned_bad:?} of {total} outputs; first (index, got bits, want bits) = {first_bad:x?}"
        );
    }
}
