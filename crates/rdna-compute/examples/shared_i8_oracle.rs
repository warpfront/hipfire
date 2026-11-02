// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU oracle of the Qwen4 shared-expert INT8 route (`HIPFIRE_QWEN4_SHARED_INT8=1`,
//! `crates/rdna-compute/src/shared_i8.rs`) on exact gfx1201.
//!
//! Byte checks (any mismatch exits 1):
//!   1. Weight shadows: the rotated, A8-quantized gate/up (interleaved plane) and down planes
//!      against the production rotations (`rotate_x_mq_batched`, `rotate_x_mq_128_v2`) of the
//!      BF16-widened weight rows followed by the `block_i8_128` quantizer contract on the host.
//!   2. GEMMs: the down entry's F32 rows, and the gate/up entry's BF16 hidden, against a host
//!      model of the kernel's exact integer + `fma(RN(d_w*d_x), C, acc)` fold over the shadow
//!      plane and the production A8 blocks of the activations (the SwiGLU through the
//!      production `shared_expert_activation_bf16_f32` on the exact accumulators).
//!   3. Hidden producer: A8 blocks of the BF16 hidden against the production
//!      `rotate_x_mq_128_v2` + the host quantizer, and the selector sigmoid against the
//!      production activation launch.
//!
//! Numeric report (informational, per row count): the whole shared expert (gate/up -> SwiGLU ->
//! down) on random weights and activations with outlier channels, the INT8 route and the F16
//! WMMA route each against an f64 model of the exact BF16 recipe on sampled tokens: relative
//! RMS and max-abs error of the final rows, and the INT8-vs-F16 route difference.
//!
//! Exit 0 pass, 1 mismatch, 2 the route does not apply (needs `HIPFIRE_QWEN4_SHARED_INT8=1` on
//! gfx1201). `ROWS=a,b,..` overrides the row list; receipt `SHARED_I8_ORACLE_RECEIPT`
//! (default `./shared-i8-oracle.txt`).
//!
//! Run: HIPFIRE_QWEN4_SHARED_INT8=1 cargo run --release --features lab --example shared_i8_oracle -p rdna-compute

use hip_bridge::DeviceBuffer;
use rdna_compute::shared_i8::SharedI8Weights;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;

const I: usize = 640;
const H: usize = 2560;
const BLOCK: usize = 136;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn bf16_rt(v: f32) -> f32 {
    bf16_f32(bf16_bits(v))
}

fn upload_bytes(gpu: &mut Gpu, bytes: &[u8], dtype: DType, n: usize) -> GpuTensor {
    let mut t = gpu.upload_raw(bytes, &[n]).expect("upload");
    t.dtype = dtype;
    t
}

fn upload_bf16(gpu: &mut Gpu, v: &[u16]) -> GpuTensor {
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for h in v {
        bytes.extend_from_slice(&h.to_le_bytes());
    }
    upload_bytes(gpu, &bytes, DType::BF16, v.len())
}

fn download_raw(gpu: &Gpu, ptr: *mut std::ffi::c_void, bytes: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().unwrap();
    let view = unsafe { DeviceBuffer::from_raw(ptr, bytes) };
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &view).unwrap();
    out
}

fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `quantize_int8_mmq_ds128` contract on 128 values: `(d, s, qs)` with `qs` in the nibble-pair
/// byte order.
fn quant_block(x: &[f32]) -> [u8; BLOCK] {
    assert_eq!(x.len(), 128);
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
    let mut q = [0i32; 128];
    for (i, v) in x.iter().enumerate() {
        q[i] = if amax == 0.0 {
            0
        } else {
            (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
        };
    }
    let s: i32 = q.iter().sum();
    let mut out = [0u8; BLOCK];
    out[0..4].copy_from_slice(&d.to_le_bytes());
    out[4..8].copy_from_slice(&s.to_le_bytes());
    for g in 0..16 {
        for j in 0..4 {
            out[8 + 8 * g + j] = q[8 * g + 2 * j] as i8 as u8;
            out[8 + 8 * g + 4 + j] = q[8 * g + 2 * j + 1] as i8 as u8;
        }
    }
    out
}

/// Plane `[k/128][rows]` of blocks from row-major `x` (`[rows][k]`), plane row of `row` =
/// `dst(row)`.
fn quant_plane(
    x: &[f32],
    k: usize,
    rows: usize,
    dst_rows: usize,
    dst: impl Fn(usize) -> usize,
    plane: &mut [u8],
) {
    for row in 0..rows {
        for kb in 0..k / 128 {
            let b = quant_block(&x[row * k + kb * 128..row * k + kb * 128 + 128]);
            let at = (kb * dst_rows + dst(row)) * BLOCK;
            plane[at..at + BLOCK].copy_from_slice(&b);
        }
    }
}

fn block_scale(plane: &[u8], kb: usize, rows: usize, row: usize) -> f32 {
    let at = (kb * rows + row) * BLOCK;
    f32::from_le_bytes(plane[at..at + 4].try_into().unwrap())
}

fn block_qs(plane: &[u8], kb: usize, rows: usize, row: usize) -> &[u8] {
    let at = (kb * rows + row) * BLOCK + 8;
    &plane[at..at + 128]
}

/// The kernel's accumulator for one (row, token): exact integer C per K128 block, folded with
/// `fma(RN(d_w * d_x), (float)C, acc)` in block order.
fn ref_acc(w: &[u8], m: usize, x: &[u8], n: usize, k: usize, row: usize, tok: usize) -> f32 {
    let mut acc = 0f32;
    for kb in 0..k / 128 {
        let (qw, qx) = (block_qs(w, kb, m, row), block_qs(x, kb, n, tok));
        let c: i32 = qw
            .iter()
            .zip(qx)
            .map(|(&a, &b)| (a as i8 as i32) * (b as i8 as i32))
            .sum();
        let scale = block_scale(w, kb, m, row) * block_scale(x, kb, n, tok);
        acc = scale.mul_add(c as f32, acc);
    }
    acc
}

fn sample_tokens(n: usize, rng: &mut Rng) -> Vec<usize> {
    if n <= 600 {
        return (0..n).collect();
    }
    let mut t = vec![
        0,
        1,
        63,
        64,
        127,
        128,
        129,
        n - 129,
        n - 128,
        n - 127,
        n - 2,
        n - 1,
    ];
    for _ in 0..40 {
        t.push(rng.below(n));
    }
    t.sort_unstable();
    t.dedup();
    t
}

struct Report {
    text: String,
    failures: usize,
}

impl Report {
    fn line(&mut self, pass: bool, msg: String) {
        if !pass {
            self.failures += 1;
        }
        let tag = if pass { "PASS" } else { "FAIL" };
        println!("ORACLE {tag} {msg}");
        writeln!(self.text, "ORACLE {tag} {msg}").unwrap();
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.to_string();
    let switch = std::env::var("HIPFIRE_QWEN4_SHARED_INT8").unwrap_or_default();
    println!("arch {arch} HIPFIRE_QWEN4_SHARED_INT8={switch:?}");

    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![512, 513, 1000, 2048, 4099]);
    let max_rows = *rows_list.iter().max().unwrap();

    let mut rng = Rng(0x5a5a_1234);
    let wscale = 0.03f32;
    let gen_w = |rng: &mut Rng, n: usize| -> Vec<u16> {
        (0..n).map(|_| bf16_bits(rng.gauss() * wscale)).collect()
    };
    let gate_h = gen_w(&mut rng, I * H);
    let up_h = gen_w(&mut rng, I * H);
    let down_h = gen_w(&mut rng, H * I);
    let gate = upload_bf16(&mut gpu, &gate_h);
    let up = upload_bf16(&mut gpu, &up_h);
    let down = upload_bf16(&mut gpu, &down_h);
    let w = SharedI8Weights {
        gate: &gate,
        up: &up,
        down: &down,
        intermediate: I,
        hidden: H,
        down_rows: H,
    };
    if !gpu.qwen4_shared_i8_applies(&w, 4096) {
        eprintln!(
            "the shared-expert INT8 route does not apply (HIPFIRE_QWEN4_SHARED_INT8={switch:?}, \
             arch {arch}): the oracle needs the switch on exact gfx1201"
        );
        std::process::exit(2);
    }

    let mut report = Report {
        text: String::new(),
        failures: 0,
    };
    writeln!(report.text, "shared_i8_oracle receipt").unwrap();
    writeln!(report.text, "arch {arch} I={I} H={H} switch={switch:?}").unwrap();

    // ---- 1. weight shadows ----
    let (gu_plane, dn_plane) = gpu.qwen4_shared_i8_shadow_host(&w).unwrap();
    let widen = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| bf16_f32(b)).collect() };
    let rotate256 = |gpu: &mut Gpu, v: &[u16], rows: usize| -> Vec<f32> {
        let x = gpu.upload_f32(&widen(v), &[rows * H]).unwrap();
        let xr = gpu.zeros(&[rows * H], DType::F32).unwrap();
        gpu.rotate_x_mq_batched(&x, &xr, H, rows).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.download_f32(&xr).unwrap()
    };
    let rg = rotate256(&mut gpu, &gate_h, I);
    let ru = rotate256(&mut gpu, &up_h, I);
    let mut gu_ref = vec![0u8; gu_plane.len()];
    quant_plane(&rg, H, I, 2 * I, |r| ((r >> 2) << 3) | (r & 3), &mut gu_ref);
    let mut up_ref = vec![0u8; gu_plane.len()];
    quant_plane(
        &ru,
        H,
        I,
        2 * I,
        |r| ((r >> 2) << 3) | 4 | (r & 3),
        &mut up_ref,
    );
    // The two writers cover disjoint plane rows: merge by row.
    let mut gu_expect = gu_ref.clone();
    for kb in 0..H / 128 {
        for r in 0..2 * I {
            if (r & 4) != 0 {
                let at = (kb * 2 * I + r) * BLOCK;
                gu_expect[at..at + BLOCK].copy_from_slice(&up_ref[at..at + BLOCK]);
            }
        }
    }
    let gu_bad = gu_plane
        .iter()
        .zip(&gu_expect)
        .filter(|(a, b)| a != b)
        .count();
    report.line(
        gu_bad == 0,
        format!(
            "shadow gate/up plane bytes={} mismatches={gu_bad}",
            gu_plane.len()
        ),
    );

    let down_x = {
        let x = gpu.upload_f32(&widen(&down_h), &[H * I]).unwrap();
        let xr = gpu.zeros(&[H * I], DType::F32).unwrap();
        gpu.rotate_x_mq_128_v2(&x, &xr, I, H).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.download_f32(&xr).unwrap()
    };
    let mut dn_expect = vec![0u8; dn_plane.len()];
    quant_plane(&down_x, I, H, H, |r| r, &mut dn_expect);
    let dn_bad = dn_plane
        .iter()
        .zip(&dn_expect)
        .filter(|(a, b)| a != b)
        .count();
    report.line(
        dn_bad == 0,
        format!(
            "shadow down plane bytes={} mismatches={dn_bad}",
            dn_plane.len()
        ),
    );

    // ---- per-row-count checks ----
    // Activations: N(0,1) with a few outlier channels (post-norm massive channels).
    let mut x_host = vec![0f32; max_rows * H];
    let outlier: Vec<usize> = (0..6).map(|_| rng.below(H)).collect();
    for r in 0..max_rows {
        for c in 0..H {
            let mut v = rng.gauss();
            if outlier.contains(&c) {
                v *= 40.0;
            }
            x_host[r * H + c] = v;
        }
    }
    let sel_host: Vec<f32> = (0..max_rows).map(|_| rng.gauss() * 2.0).collect();

    for &n in &rows_list {
        let x = gpu.upload_f32(&x_host[..n * H], &[n * H]).unwrap();
        let hid = gpu.zeros(&[n * I], DType::F32).unwrap(); // BF16 hidden bytes (n*I*2 <= n*I*4)
        let blocks = gpu.zeros(&[n * I], DType::F32).unwrap(); // A8 hidden blocks (n*I/128*136 bytes)
        let sel = gpu.upload_f32(&sel_host[..n], &[n]).unwrap();
        let out = gpu.zeros(&[n * H], DType::F32).unwrap();

        // Production A8 blocks of x (the producer the route calls).
        let xq_ptr = gpu.mq6_i8_prepare_x(&x, H, n).unwrap();
        let xq = download_raw(&gpu, xq_ptr, (H / 128) * n * BLOCK);

        // 0. The fused front producer's three outputs against the standalone chain it extends:
        // the F16 X and the IU4 sidecar of `qwen4_moe_front_producer` and the A8 blocks of
        // `mq6_i8_prepare_x`.
        if gpu.arch == "gfx1201" && gpu.qwen4_moe_front_fuse_applies() {
            let (p4, x16r) = gpu.qwen4_moe_front_producer(&x, H, n).unwrap();
            let p4r = download_raw(
                &gpu,
                gpu.int4_mmq_prepared_ptr(&p4, H, n).unwrap(),
                (H / 128) * n * 72,
            );
            let x16r_b = download_raw(&gpu, x16r.buf.as_ptr(), n * H * 2);
            let (p4n, x16n, p8) = gpu.qwen4_moe_front_producer_i8(&x, H, n).unwrap();
            let p4n_b = download_raw(
                &gpu,
                gpu.int4_mmq_prepared_ptr(&p4n, H, n).unwrap(),
                (H / 128) * n * 72,
            );
            let x16n_b = download_raw(&gpu, x16n.buf.as_ptr(), n * H * 2);
            let p8_b = download_raw(
                &gpu,
                gpu.int8_mmq_prepared_ptr(&p8, H, n).unwrap(),
                (H / 128) * n * BLOCK,
            );
            let bad4 = p4r.iter().zip(&p4n_b).filter(|(a, b)| a != b).count();
            let bad16 = x16r_b.iter().zip(&x16n_b).filter(|(a, b)| a != b).count();
            let bad8 = xq.iter().zip(&p8_b).filter(|(a, b)| a != b).count();
            report.line(
                bad4 == 0 && bad16 == 0 && bad8 == 0,
                format!("front producer i8 rows={n} iu4_mismatches={bad4} f16x_mismatches={bad16} a8_mismatches={bad8}"),
            );
        } else if n == rows_list[0] {
            println!("front producer check skipped: qwen4_moe_front_fuse_applies() is false here");
        }

        gpu.qwen4_shared_i8_gate_up(&w, &x, &hid, n).unwrap();
        let hid_bits = le_u16(&download_raw(&gpu, hid.buf.as_ptr(), n * I * 2));
        let toks = sample_tokens(n, &mut rng);

        // 2a. gate/up GEMM -> exact accumulators -> production activation on them.
        let mut g_acc = vec![0f32; toks.len() * I];
        let mut u_acc = vec![0f32; toks.len() * I];
        for (ti, &t) in toks.iter().enumerate() {
            for c in 0..I {
                let (gr, ur) = (((c >> 2) << 3) | (c & 3), ((c >> 2) << 3) | 4 | (c & 3));
                g_acc[ti * I + c] = ref_acc(&gu_plane, 2 * I, &xq, n, H, gr, t);
                u_acc[ti * I + c] = ref_acc(&gu_plane, 2 * I, &xq, n, H, ur, t);
            }
        }
        let gt = gpu.upload_f32(&g_acc, &[g_acc.len()]).unwrap();
        let ut = gpu.upload_f32(&u_acc, &[u_acc.len()]).unwrap();
        let hr = gpu.zeros(&[g_acc.len()], DType::F32).unwrap();
        let sdummy = gpu.zeros(&[toks.len()], DType::F32).unwrap();
        gpu.shared_expert_activation_bf16_f32(&gt, &ut, &hr, &sdummy, toks.len())
            .unwrap();
        let hr_f = gpu.download_f32(&hr).unwrap();
        let mut bad = 0usize;
        let mut first = None;
        for (ti, &t) in toks.iter().enumerate() {
            for c in 0..I {
                let want = hr_f[ti * I + c].to_bits() >> 16;
                let got = hid_bits[t * I + c] as u32;
                if want != got {
                    bad += 1;
                    first.get_or_insert((t, c, want, got));
                }
            }
        }
        report.line(
            bad == 0,
            format!(
                "gate_up gemm+silu rows={n} tokens_checked={} mismatches={bad} first={first:?}",
                toks.len()
            ),
        );

        // 3. hidden producer: blocks + selector vs production rotate128 + host quantizer.
        gpu.qwen4_shared_i8_hidden(&w, &hid, &blocks, &sel, n)
            .unwrap();
        let blk = download_raw(&gpu, blocks.buf.as_ptr(), (I / 128) * n * BLOCK);
        let hid_f: Vec<f32> = hid_bits.iter().map(|&b| bf16_f32(b)).collect();
        let hx = gpu.upload_f32(&hid_f, &[n * I]).unwrap();
        let hxr = gpu.zeros(&[n * I], DType::F32).unwrap();
        gpu.rotate_x_mq_128_v2(&hx, &hxr, I, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let hrot = gpu.download_f32(&hxr).unwrap();
        let mut blk_expect = vec![0u8; blk.len()];
        quant_plane(&hrot, I, n, n, |r| r, &mut blk_expect);
        let blk_bad = blk.iter().zip(&blk_expect).filter(|(a, b)| a != b).count();
        report.line(
            blk_bad == 0,
            format!(
                "hidden producer rows={n} bytes={} mismatches={blk_bad}",
                blk.len()
            ),
        );
        let selr = gpu.upload_f32(&sel_host[..n], &[n]).unwrap();
        let gz = gpu.zeros(&[n * I], DType::F32).unwrap();
        let oz = gpu.zeros(&[n * I], DType::F32).unwrap();
        gpu.shared_expert_activation_bf16_f32(&gz, &gz, &oz, &selr, n)
            .unwrap();
        let sel_got = le_f32(&download_raw(&gpu, sel.buf.as_ptr(), n * 4));
        let sel_want = gpu.download_f32(&selr).unwrap();
        let sel_bad = sel_got
            .iter()
            .zip(&sel_want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        report.line(
            sel_bad == 0,
            format!("selector sigmoid rows={n} mismatches={sel_bad}"),
        );

        // 2b. down GEMM rows vs the host fold over the shadow plane and the produced blocks.
        gpu.qwen4_shared_i8_down(&w, &blocks, &out, n).unwrap();
        let out_f = le_f32(&download_raw(&gpu, out.buf.as_ptr(), n * H * 4));
        let mut bad = 0usize;
        let mut first = None;
        for &t in &toks {
            for r in 0..H {
                let want = ref_acc(&dn_plane, H, &blk, n, I, r, t);
                if want.to_bits() != out_f[t * H + r].to_bits() {
                    bad += 1;
                    first.get_or_insert((t, r, want, out_f[t * H + r]));
                }
            }
        }
        report.line(
            bad == 0,
            format!(
                "down gemm rows={n} tokens_checked={} mismatches={bad} first={first:?}",
                toks.len()
            ),
        );

        // ---- numeric report vs the f64 recipe model and the F16 WMMA route ----
        let g16 = gpu.zeros(&[n * I], DType::F32).unwrap();
        let u16t = gpu.zeros(&[n * I], DType::F32).unwrap();
        let ok = gpu
            .gemm_bf16_xf32_f16_wmma_qwen4_groups(&[&[(&gate, &g16, I), (&up, &u16t, I)]], &x, H, n)
            .unwrap();
        assert!(
            ok,
            "the F16 WMMA route does not apply to the reference chain"
        );
        let h16 = gpu.zeros(&[n * I], DType::F32).unwrap();
        let sel16 = gpu.upload_f32(&sel_host[..n], &[n]).unwrap();
        gpu.shared_expert_activation_bf16_f32(&g16, &u16t, &h16, &sel16, n)
            .unwrap();
        let out16 = gpu.zeros(&[n * H], DType::F32).unwrap();
        let ok = gpu
            .gemm_bf16_xf32_f16_wmma_qwen4(&[(&down, &out16, H)], &h16, I, n)
            .unwrap();
        assert!(
            ok,
            "the F16 WMMA route does not apply to the reference down"
        );
        let out16_f = le_f32(&download_raw(&gpu, out16.buf.as_ptr(), n * H * 4));

        let wg: Vec<f32> = widen(&gate_h);
        let wu: Vec<f32> = widen(&up_h);
        let wd: Vec<f32> = widen(&down_h);
        let ntr = toks.len().min(48);
        let (mut num8, mut num16, mut numd, mut den, mut max8, mut max16, mut maxt) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for &t in toks.iter().take(ntr) {
            let xr = &x_host[t * H..(t + 1) * H];
            let mut hid_t = vec![0f64; I];
            for c in 0..I {
                let (mut g, mut u) = (0f64, 0f64);
                for k in 0..H {
                    g += wg[c * H + k] as f64 * xr[k] as f64;
                    u += wu[c * H + k] as f64 * xr[k] as f64;
                }
                let (g, u) = (bf16_rt(g as f32) as f64, bf16_rt(u as f32) as f64);
                hid_t[c] = bf16_rt(((g / (1.0 + (-g).exp())) * u) as f32) as f64;
            }
            for r in 0..H {
                let mut o = 0f64;
                for c in 0..I {
                    o += wd[r * I + c] as f64 * hid_t[c];
                }
                let (a8, a16) = (out_f[t * H + r] as f64, out16_f[t * H + r] as f64);
                num8 += (a8 - o) * (a8 - o);
                num16 += (a16 - o) * (a16 - o);
                numd += (a8 - a16) * (a8 - a16);
                den += o * o;
                max8 = max8.max((a8 - o).abs());
                max16 = max16.max((a16 - o).abs());
                maxt = maxt.max(o.abs());
            }
        }
        // Stage attribution: INT8 gate/up + F16 down (hidden from the INT8 GEMM), and F16 gate/up +
        // INT8 down (the F16 hidden through the INT8 hidden producer and down GEMM).
        let mut hid_i8_f32 = gpu.zeros(&[n * I], DType::F32).unwrap();
        {
            let v: Vec<f32> = hid_bits.iter().map(|&b| bf16_f32(b)).collect();
            hid_i8_f32 = gpu.upload_f32(&v, &[n * I]).unwrap();
        }
        let outa = gpu.zeros(&[n * H], DType::F32).unwrap();
        gpu.gemm_bf16_xf32_f16_wmma_qwen4(&[(&down, &outa, H)], &hid_i8_f32, I, n)
            .unwrap();
        let outa_f = le_f32(&download_raw(&gpu, outa.buf.as_ptr(), n * H * 4));
        let h16_f = le_f32(&download_raw(&gpu, h16.buf.as_ptr(), n * I * 4));
        let h16_bits: Vec<u16> = h16_f.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
        let h16_bf = upload_bf16(&mut gpu, &h16_bits);
        let blocks_b = gpu.zeros(&[n * I], DType::F32).unwrap();
        let selb = gpu.upload_f32(&sel_host[..n], &[n]).unwrap();
        gpu.qwen4_shared_i8_hidden(&w, &h16_bf, &blocks_b, &selb, n)
            .unwrap();
        let outb = gpu.zeros(&[n * H], DType::F32).unwrap();
        gpu.qwen4_shared_i8_down(&w, &blocks_b, &outb, n).unwrap();
        let outb_f = le_f32(&download_raw(&gpu, outb.buf.as_ptr(), n * H * 4));
        let (mut numa, mut numb) = (0f64, 0f64);
        for &t in toks.iter().take(ntr) {
            let xr = &x_host[t * H..(t + 1) * H];
            let mut hid_t = vec![0f64; I];
            for c in 0..I {
                let (mut g, mut u) = (0f64, 0f64);
                for k in 0..H {
                    g += wg[c * H + k] as f64 * xr[k] as f64;
                    u += wu[c * H + k] as f64 * xr[k] as f64;
                }
                let (g, u) = (bf16_rt(g as f32) as f64, bf16_rt(u as f32) as f64);
                hid_t[c] = bf16_rt(((g / (1.0 + (-g).exp())) * u) as f32) as f64;
            }
            for r in 0..H {
                let mut o = 0f64;
                for c in 0..I {
                    o += wd[r * I + c] as f64 * hid_t[c];
                }
                numa += (outa_f[t * H + r] as f64 - o).powi(2);
                numb += (outb_f[t * H + r] as f64 - o).powi(2);
            }
        }
        println!(
            "NUMERIC-STAGES rows={n} rel_rms int8 gate/up + f16 down={:.3e}  f16 gate/up + int8 down={:.3e}",
            (numa / den).sqrt(),
            (numb / den).sqrt()
        );
        let line = format!(
            "NUMERIC rows={n} tokens={ntr} rel_rms int8={:.3e} f16={:.3e} int8_vs_f16={:.3e} | max_abs/max|ref| int8={:.3e} f16={:.3e}",
            (num8 / den).sqrt(),
            (num16 / den).sqrt(),
            (numd / den).sqrt(),
            max8 / maxt,
            max16 / maxt,
        );
        println!("{line}");
        writeln!(report.text, "{line}").unwrap();
    }

    let path = std::env::var("SHARED_I8_ORACLE_RECEIPT")
        .unwrap_or_else(|_| "./shared-i8-oracle.txt".into());
    writeln!(report.text, "failures {}", report.failures).unwrap();
    std::fs::write(&path, &report.text).expect("receipt");
    println!("receipt {path}: failures={}", report.failures);
    std::process::exit(if report.failures == 0 { 0 } else { 1 });
}
