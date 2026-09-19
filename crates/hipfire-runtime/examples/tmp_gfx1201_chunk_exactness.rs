// SPDX-License-Identifier: Apache-2.0
// Temporary gfx1201 chunk-widening exactness oracle (plan §7, arms 1-6).
// O-owned throwaway: public forward/wrapper APIs only. No production hooks,
// no kernel sources, no TU changes.
//
// Contract: `--chunk C` runs the candidate C-row schedule against the frozen
// 512-row baseline schedule and exits nonzero on ANY bit/guard/state
// mismatch. Baseline is always the legacy per-512 commit sequence.
// Post-integration, H's production entry logs
// `prefill_chunk: requested=<r> admitted=<a> commit_stride=512`; this oracle
// prints the same line for the schedule IT executed (requested == admitted by
// construction here: the oracle drives HIPFIRE_PREFILL_MAX_BATCH directly).
//
// Usage (ordinal 2 box):
//   HOME=/home/kaden/.hipfire-homes/ab2 ROCR_VISIBLE_DEVICES=2 \
//   HIPFIRE_KERNEL_CACHE=$HOME/.hipfire_kernels/gfx1201 \
//   cargo run --release -p hipfire-runtime --features deltanet \
//     --example tmp_gfx1201_chunk_exactness -- \
//     --model /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt --chunk 1024 \
//     --out-dir .redline-work/gfx1201-chunk
// Flags: --arms 1,2,3,4,5,6 (default all), --skip-model (synthetic arms only).
// Exit: 0 all exact; 1 any mismatch/launch failure; 2 usage/environment error.

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("tmp_gfx1201_chunk_exactness: build with --features deltanet");
    std::process::exit(2);
}

#[cfg(feature = "deltanet")]
mod ora {
    use hipfire_arch_qwen35::qwen35::{
        self, DeltaNetState, PrefillBatchScratch, Qwen35Scratch,
    };
    use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::norm::{gdn_requant_frame_checkpoint, restore_gdn_requant_frame_checkpoint};
    use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};
    use std::path::{Path, PathBuf};

    // ── deterministic PRNG ──────────────────────────────────────────────
    pub fn prng_u32(i: u64, salt: u32) -> u32 {
        let mut x = (i as u32)
            .wrapping_mul(0x9e3779b1)
            .wrapping_add(salt)
            .wrapping_mul(0x85ebca6b);
        x ^= x >> 16;
        x = x.wrapping_mul(0xc2b2ae35);
        x ^ (x >> 16)
    }
    pub fn prng_f32(i: u64, salt: u32) -> f32 {
        (prng_u32(i, salt) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    pub fn f32_to_f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 31) & 1) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 0 {
            return sign << 15;
        }
        if exp == 0xff {
            let m = if mant != 0 { 0x200 } else { 0 };
            return (sign << 15) | 0x7c00 | m;
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1f {
            return (sign << 15) | 0x7c00;
        }
        if new_exp <= 0 {
            return sign << 15;
        }
        let new_mant = (mant >> 13) as u16;
        (sign << 15) | ((new_exp as u16) << 10) | new_mant
    }

    // ── byte compare helpers ────────────────────────────────────────────
    /// First differing byte offset, or None when equal (length mismatch counts
    /// as a difference at the common-prefix length).
    pub fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
        let n = a.len().min(b.len());
        for i in 0..n {
            if a[i] != b[i] {
                return Some(i);
            }
        }
        if a.len() != b.len() {
            return Some(n);
        }
        None
    }

    pub struct Cmp {
        pub failures: usize,
        pub bytes_compared: u64,
    }
    impl Cmp {
        pub fn new() -> Self {
            Self {
                failures: 0,
                bytes_compared: 0,
            }
        }
        /// Bit-exact compare; logs arm/layer/position/first-offset on mismatch.
        pub fn check(&mut self, arm: &str, what: &str, a: &[u8], b: &[u8]) {
            self.bytes_compared += (a.len() as u64) + (b.len() as u64);
            match first_diff(a, b) {
                None => eprintln!("  ok   {arm} {what} ({} B each)", a.len()),
                Some(off) => {
                    self.failures += 1;
                    let (ab, bb) = (
                        if off < a.len() {
                            format!("{:02x}", a[off])
                        } else {
                            "<end>".into()
                        },
                        if off < b.len() {
                            format!("{:02x}", b[off])
                        } else {
                            "<end>".into()
                        },
                    );
                    eprintln!(
                        "  FAIL {arm} {what}: len_a={} len_b={} first_diff_off={off} a={ab} b={bb}",
                        a.len(),
                        b.len()
                    );
                }
            }
        }
    }

    pub fn download_raw(gpu: &mut Gpu, t: &GpuTensor) -> Vec<u8> {
        // Physical buffer is ground truth: production Q8 state declares
        // shape=[s_size] with dtype F32 over a byte-sized malloc (weights.rs),
        // so shape-product x dtype-size can exceed the allocation. Clamp.
        let logical: usize = t.shape.iter().product::<usize>() * t.dtype.size();
        let n = logical.min(t.buf.size());
        let mut out = vec![0u8; n];
        gpu.hip
            .memcpy_dtoh(&mut out, &t.buf)
            .unwrap_or_else(|e| panic!("download_raw (logical {logical} physical {}): {e:?}", t.buf.size()));
        out
    }
    pub fn download_f32_vec(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
        let v = gpu
            .download_f32(t)
            .unwrap_or_else(|e| panic!("download_f32: {e:?}"));
        let n = v.len() * 4;
        let mut out = vec![0u8; n];
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) });
        out
    }
    /// Snapshot Gpu-owned scratch behind a raw pointer (Borrowed wrapper: never freed).
    pub fn download_ptr(gpu: &Gpu, ptr: *mut std::ffi::c_void, nbytes: usize) -> Vec<u8> {
        let buf =
            unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, nbytes) };
        let mut out = vec![0u8; nbytes];
        gpu.hip
            .memcpy_dtoh(&mut out, &buf)
            .unwrap_or_else(|e| panic!("download_ptr ({nbytes} B): {e:?}"));
        // Borrowed: drops without freeing. Sync first so bytes are settled.
        gpu.hip.device_synchronize().unwrap();
        out
    }
    pub fn sync(gpu: &Gpu) {
        gpu.hip.device_synchronize().unwrap();
    }
    pub fn upload_i8(gpu: &mut Gpu, data: &[i8], shape: &[usize]) -> GpuTensor {
        let t = gpu.alloc_tensor(shape, DType::Raw).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
        t
    }
    pub fn upload_u16(gpu: &mut Gpu, data: &[u16], shape: &[usize]) -> GpuTensor {
        let t = gpu.alloc_tensor(shape, DType::F16).unwrap();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
        t
    }

    // ══ ARM 1: producer differential (FP8 prepare + GEMM, whole-C vs 512 views)
    // Shapes from plan §7.1 (model dims: D5120/I17408, LA K16/V48/D128).
    // Synthetic MQ4G256V2 (qt=44) blobs via host pack; identical both arms, so
    // the differential isolates chunking (per-row pack, K-order, alignment).
    // LUT selectors are synthetic zeros, identical both arms (documented).
    const GROUP: usize = 256;
    const HALF_G: usize = 128;
    const GROUP_BYTES: usize = 136;
    fn pack_v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
        use half::f16;
        let gpr = k / GROUP;
        let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
        for r in 0..m {
            for g in 0..gpr {
                let src = r * k + g * GROUP;
                let dst = (r * gpr + g) * GROUP_BYTES;
                let mut q = [0u8; GROUP];
                for h in 0..2 {
                    let off = h * HALF_G;
                    let s = &w[src + off..src + off + HALF_G];
                    let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
                    let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                    let sb = if hi == lo {
                        0u16
                    } else {
                        f16::from_f32(step).to_bits()
                    };
                    let zb = f16::from_f32(lo).to_bits();
                    blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&sb.to_le_bytes());
                    blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&zb.to_le_bytes());
                    let st = f16::from_bits(sb).to_f32();
                    let z = f16::from_bits(zb).to_f32();
                    if st == 0.0 {
                        continue;
                    }
                    let inv = 1.0 / st;
                    for i in 0..HALF_G {
                        q[off + i] = ((s[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                    }
                }
                for i in 0..HALF_G {
                    blob[dst + 8 + i] = (q[2 * i] & 0xF) | ((q[2 * i + 1] & 0xF) << 4);
                }
            }
        }
        blob
    }
    fn det_weights(m: usize, k: usize, salt: u32) -> Vec<f32> {
        // Realistic post-FWHT scale (sigma ~0.011 Gaussian via Box-Muller).
        let mut w = vec![0.0f32; m * k];
        for (i, v) in w.iter_mut().enumerate() {
            let u1 = (prng_u32(i as u64, salt) as f64 / u32::MAX as f64).max(1e-7);
            let u2 = prng_u32(i as u64, salt ^ 0x9abc_def0) as f64 / u32::MAX as f64;
            *v = ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos() * 0.011) as f32;
        }
        w
    }
    fn det_x(n: usize, k: usize, salt: u32) -> Vec<f32> {
        (0..n * k).map(|i| prng_f32(i as u64, salt)).collect()
    }
    /// Snapshot prepared FP8 bytes for (x_view, n, k): codes + half-sums + row-scales.
    fn snap_prepared(gpu: &mut Gpu, x: &GpuTensor, n: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let p = gpu
            .prepare_mq4v2_fp8_x(x, n, k, 1)
            .unwrap_or_else(|e| panic!("prepare_mq4v2_fp8_x n={n} k={k}: {e:?}"));
        sync(&mut *gpu);
        let groups = k / 256;
        let a = download_ptr(&mut *gpu, p.x_fp8, n * k);
        let b = download_ptr(&mut *gpu, p.half_sums, n * groups * 8);
        let c = download_ptr(&mut *gpu, p.row_scales, n * 4);
        (a, b, c)
    }

    fn arm1_residual(
        gpu: &mut Gpu,
        cmp: &mut Cmp,
        c: usize,
        name: &str,
        m: usize,
        k: usize,
        nonzero_y: bool,
    ) {
        // y = Wx (+ residual): whole-C vs per-512 views on cloned buffers.
        let w = det_weights(m, k, 0xA11CE);
        let blob = pack_v2(&w, m, k);
        let x_h = det_x(c, k, 0xB0B);
        let y0_h: Vec<f32> = if nonzero_y {
            (0..c * m).map(|i| prng_f32(i as u64, 0xADD) * 0.25).collect()
        } else {
            vec![0.0; c * m]
        };
        let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let d_y = gpu.upload_f32(&y0_h, &[c, m]).unwrap();
        // Whole-C prepared snapshot, then launch.
        let (pa, ha, sa) = snap_prepared(&mut *gpu, &d_x, c, k);
        gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
            &d_a, &d_x, &d_y, m, k, c, 1, [0, 0, 0, 0],
        )
        .unwrap_or_else(|e| panic!("arm1/{name} whole-C: {e:?}"));
        sync(&mut *gpu);
        // Segmented: prepare+launch per 512-row view of clones.
        let d_xb = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let d_yb = gpu.upload_f32(&y0_h, &[c, m]).unwrap();
        let nseg = c.div_ceil(512);
        for s in 0..nseg {
            let o = s * 512;
            let n = (c - o).min(512);
            let xv = d_xb.sub_offset(o * k, n * k);
            let (pb, hb, sb) = snap_prepared(&mut *gpu, &xv, n, k);
            // Prepared bytes for this segment must equal the whole-C slice.
            let groups = k / 256;
            cmp.check(
                "arm1",
                &format!("{name}/prep-codes-seg{s}"),
                &pa[o * k..(o + n) * k],
                &pb,
            );
            cmp.check(
                "arm1",
                &format!("{name}/prep-halfsums-seg{s}"),
                &ha[o * groups * 8..(o + n) * groups * 8],
                &hb,
            );
            cmp.check(
                "arm1",
                &format!("{name}/prep-rowscales-seg{s}"),
                &sa[o * 4..(o + n) * 4],
                &sb,
            );
            let yv = d_yb.sub_offset(o * m, n * m);
            gpu.gemm_hfq4g256_residual_wmma_gfx12_mq4v2_fp8_lloyd(
                &d_a, &xv, &yv, m, k, n, 1, [0, 0, 0, 0],
            )
            .unwrap_or_else(|e| panic!("arm1/{name} seg{s}: {e:?}"));
            sync(&mut *gpu);
        }
        let ya = download_f32_vec(&mut *gpu, &d_y);
        let yb = download_f32_vec(&mut *gpu, &d_yb);
        cmp.check("arm1", &format!("{name}/f32-out"), &ya, &yb);
        for t in [d_a, d_x, d_y, d_xb, d_yb] {
            gpu.free_tensor(t).unwrap();
        }
    }

    fn arm1_gate_up(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        let (gm, um, k) = (17408usize, 17408usize, 5120usize);
        let bg = pack_v2(&det_weights(gm, k, 0x6A7E), gm, k);
        let bu = pack_v2(&det_weights(um, k, 0x01CE), um, k);
        let x_h = det_x(c, k, 0x9A7);
        let d_ag = gpu.upload_raw(&bg, &[bg.len()]).unwrap();
        let d_au = gpu.upload_raw(&bu, &[bu.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let d_yg = gpu.zeros(&[c * gm], DType::F32).unwrap();
        let d_yu = gpu.zeros(&[c * um], DType::F32).unwrap();
        let (pa, ha, sa) = snap_prepared(&mut *gpu, &d_x, c, k);
        gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
            &d_ag, &d_au, &d_x, &d_yg, &d_yu, gm, um, k, c, 1, [0, 0, 0, 0], [0, 0, 0, 0],
        )
        .unwrap_or_else(|e| panic!("arm1/gate_up whole-C: {e:?}"));
        sync(&mut *gpu);
        let d_xb = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let d_ygb = gpu.zeros(&[c * gm], DType::F32).unwrap();
        let d_yub = gpu.zeros(&[c * um], DType::F32).unwrap();
        for s in 0..c.div_ceil(512) {
            let (o, n) = (s * 512, (c - s * 512).min(512));
            let xv = d_xb.sub_offset(o * k, n * k);
            let (pb, hb, sb) = snap_prepared(&mut *gpu, &xv, n, k);
            let groups = k / 256;
            cmp.check("arm1", &format!("gate_up/prep-codes-seg{s}"), &pa[o*k..(o+n)*k], &pb);
            cmp.check(
                "arm1",
                &format!("gate_up/prep-halfsums-seg{s}"),
                &ha[o*groups*8..(o+n)*groups*8],
                &hb,
            );
            cmp.check(
                "arm1",
                &format!("gate_up/prep-rowscales-seg{s}"),
                &sa[o*4..(o+n)*4],
                &sb,
            );
            let ygv = d_ygb.sub_offset(o * gm, n * gm);
            let yuv = d_yub.sub_offset(o * um, n * um);
            gpu.gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &d_ag, &d_au, &xv, &ygv, &yuv, gm, um, k, n, 1, [0, 0, 0, 0], [0, 0, 0, 0],
            )
            .unwrap_or_else(|e| panic!("arm1/gate_up seg{s}: {e:?}"));
            sync(&mut *gpu);
        }
        cmp.check("arm1", "gate_up/f32-gate", &download_f32_vec(&mut *gpu, &d_yg), &download_f32_vec(&mut *gpu, &d_ygb));
        cmp.check("arm1", "gate_up/f32-up", &download_f32_vec(&mut *gpu, &d_yu), &download_f32_vec(&mut *gpu, &d_yub));
        for t in [d_ag, d_au, d_x, d_yg, d_yu, d_xb, d_ygb, d_yub] {
            gpu.free_tensor(t).unwrap();
        }
    }

    fn arm1_qkvza(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        let (qm, zm, bm, am, k) = (10240usize, 6144usize, 48usize, 48usize, 5120usize);
        let bq = pack_v2(&det_weights(qm, k, 0x9A0), qm, k);
        let bz = pack_v2(&det_weights(zm, k, 0x2E7), zm, k);
        let bb = pack_v2(&det_weights(bm, k, 0xBE7), bm, k);
        let ba = pack_v2(&det_weights(am, k, 0xA1A), am, k);
        let x_h = det_x(c, k, 0x515);
        let d_q = gpu.upload_raw(&bq, &[bq.len()]).unwrap();
        let d_z = gpu.upload_raw(&bz, &[bz.len()]).unwrap();
        let d_b = gpu.upload_raw(&bb, &[bb.len()]).unwrap();
        let d_a = gpu.upload_raw(&ba, &[ba.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let (yq, yz, yb, ya) = (
            gpu.zeros(&[c * qm], DType::F32).unwrap(),
            gpu.zeros(&[c * zm], DType::F32).unwrap(),
            gpu.zeros(&[c * bm], DType::F32).unwrap(),
            gpu.zeros(&[c * am], DType::F32).unwrap(),
        );
        let (pa, ha, sa) = snap_prepared(&mut *gpu, &d_x, c, k);
        gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
            &d_q, &d_z, &d_b, &d_a, &d_x, &yq, &yz, &yb, &ya, qm, zm, bm, am, k, c, 1,
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        )
        .unwrap_or_else(|e| panic!("arm1/qkvza whole-C: {e:?}"));
        sync(&mut *gpu);
        let d_xb = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let (yqb, yzb, ybb, yab) = (
            gpu.zeros(&[c * qm], DType::F32).unwrap(),
            gpu.zeros(&[c * zm], DType::F32).unwrap(),
            gpu.zeros(&[c * bm], DType::F32).unwrap(),
            gpu.zeros(&[c * am], DType::F32).unwrap(),
        );
        for s in 0..c.div_ceil(512) {
            let (o, n) = (s * 512, (c - s * 512).min(512));
            let xv = d_xb.sub_offset(o * k, n * k);
            let (pb, hb, sb) = snap_prepared(&mut *gpu, &xv, n, k);
            let groups = k / 256;
            cmp.check("arm1", &format!("qkvza/prep-codes-seg{s}"), &pa[o*k..(o+n)*k], &pb);
            cmp.check("arm1", &format!("qkvza/prep-halfsums-seg{s}"), &ha[o*groups*8..(o+n)*groups*8], &hb);
            cmp.check("arm1", &format!("qkvza/prep-rowscales-seg{s}"), &sa[o*4..(o+n)*4], &sb);
            gpu.gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &d_q, &d_z, &d_b, &d_a, &xv,
                &yqb.sub_offset(o * qm, n * qm),
                &yzb.sub_offset(o * zm, n * zm),
                &ybb.sub_offset(o * bm, n * bm),
                &yab.sub_offset(o * am, n * am),
                qm, zm, bm, am, k, n, 1,
                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
            )
            .unwrap_or_else(|e| panic!("arm1/qkvza seg{s}: {e:?}"));
            sync(&mut *gpu);
        }
        cmp.check("arm1", "qkvza/f32-qkv", &download_f32_vec(&mut *gpu, &yq), &download_f32_vec(&mut *gpu, &yqb));
        cmp.check("arm1", "qkvza/f32-z", &download_f32_vec(&mut *gpu, &yz), &download_f32_vec(&mut *gpu, &yzb));
        cmp.check("arm1", "qkvza/f32-beta", &download_f32_vec(&mut *gpu, &yb), &download_f32_vec(&mut *gpu, &ybb));
        cmp.check("arm1", "qkvza/f32-alpha", &download_f32_vec(&mut *gpu, &ya), &download_f32_vec(&mut *gpu, &yab));
        for t in [d_q, d_z, d_b, d_a, d_x, yq, yz, yb, ya, d_xb, yqb, yzb, ybb, yab] {
            gpu.free_tensor(t).unwrap();
        }
    }

    fn arm1_qkv(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        let (qm, km, vm, k) = (6144usize, 1024usize, 1024usize, 5120usize);
        let bq = pack_v2(&det_weights(qm, k, 0xF90), qm, k);
        let bk = pack_v2(&det_weights(km, k, 0xE11), km, k);
        let bv = pack_v2(&det_weights(vm, k, 0xA6), vm, k);
        let x_h = det_x(c, k, 0xC0C);
        let d_q = gpu.upload_raw(&bq, &[bq.len()]).unwrap();
        let d_k = gpu.upload_raw(&bk, &[bk.len()]).unwrap();
        let d_v = gpu.upload_raw(&bv, &[bv.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let (yq, yk, yv) = (
            gpu.zeros(&[c * qm], DType::F32).unwrap(),
            gpu.zeros(&[c * km], DType::F32).unwrap(),
            gpu.zeros(&[c * vm], DType::F32).unwrap(),
        );
        let (pa, ha, sa) = snap_prepared(&mut *gpu, &d_x, c, k);
        gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
            &d_q, &d_k, &d_v, &d_x, &yq, &yk, &yv, qm, km, vm, k, c, 1,
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        )
        .unwrap_or_else(|e| panic!("arm1/fa-qkv whole-C: {e:?}"));
        sync(&mut *gpu);
        let d_xb = gpu.upload_f32(&x_h, &[c, k]).unwrap();
        let (yqb, ykb, yvb) = (
            gpu.zeros(&[c * qm], DType::F32).unwrap(),
            gpu.zeros(&[c * km], DType::F32).unwrap(),
            gpu.zeros(&[c * vm], DType::F32).unwrap(),
        );
        for s in 0..c.div_ceil(512) {
            let (o, n) = (s * 512, (c - s * 512).min(512));
            let xv = d_xb.sub_offset(o * k, n * k);
            let (pb, hb, sb) = snap_prepared(&mut *gpu, &xv, n, k);
            let groups = k / 256;
            cmp.check("arm1", &format!("fa-qkv/prep-codes-seg{s}"), &pa[o*k..(o+n)*k], &pb);
            cmp.check("arm1", &format!("fa-qkv/prep-halfsums-seg{s}"), &ha[o*groups*8..(o+n)*groups*8], &hb);
            cmp.check("arm1", &format!("fa-qkv/prep-rowscales-seg{s}"), &sa[o*4..(o+n)*4], &sb);
            gpu.gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8_lloyd(
                &d_q, &d_k, &d_v, &xv,
                &yqb.sub_offset(o * qm, n * qm),
                &ykb.sub_offset(o * km, n * km),
                &yvb.sub_offset(o * vm, n * vm),
                qm, km, vm, k, n, 1,
                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
            )
            .unwrap_or_else(|e| panic!("arm1/fa-qkv seg{s}: {e:?}"));
            sync(&mut *gpu);
        }
        cmp.check("arm1", "fa-qkv/f32-q", &download_f32_vec(&mut *gpu, &yq), &download_f32_vec(&mut *gpu, &yqb));
        cmp.check("arm1", "fa-qkv/f32-k", &download_f32_vec(&mut *gpu, &yk), &download_f32_vec(&mut *gpu, &ykb));
        cmp.check("arm1", "fa-qkv/f32-v", &download_f32_vec(&mut *gpu, &yv), &download_f32_vec(&mut *gpu, &yvb));
        for t in [d_q, d_k, d_v, d_x, yq, yk, yv, d_xb, yqb, ykb, yvb] {
            gpu.free_tensor(t).unwrap();
        }
    }

    pub fn run_arm1(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        eprintln!("arm1: producer differential C={c} (whole-C vs 512 views)");
        arm1_gate_up(&mut *gpu, cmp, c);
        arm1_qkvza(&mut *gpu, cmp, c);
        arm1_qkv(&mut *gpu, cmp, c);
        arm1_residual(&mut *gpu, cmp, c, "resid-out-zero", 5120, 6144, false);
        arm1_residual(&mut *gpu, cmp, c, "resid-out-nonzero", 5120, 6144, true);
        arm1_residual(&mut *gpu, cmp, c, "resid-down-zero", 5120, 17408, false);
        arm1_residual(&mut *gpu, cmp, c, "resid-down-nonzero", 5120, 17408, true);
    }
    const GH: usize = 48;
    const GD: usize = 128;
    const GVD: usize = GH * GD; // 6144
    /// Host match of fast-kernel EF requant (per-row absmax over HD).
    fn quantize_s(s_f32: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<u16>) {
        let n_rows = GH * GD;
        assert_eq!(s_f32.len(), n_rows * GD);
        let mut codes = vec![0i8; n_rows * GD];
        let mut scales = vec![0f32; n_rows];
        let mut ef = vec![0u16; n_rows * GD];
        for row in 0..n_rows {
            let base = row * GD;
            let mut my_max = 0.0f32;
            for &x in &s_f32[base..base + GD] {
                my_max = my_max.max(x.abs());
            }
            let inv = if my_max > 0.0 { 127.0 / my_max } else { 0.0 };
            let sc = if my_max > 0.0 { my_max / 127.0 } else { 1.0 };
            scales[row] = sc;
            for cc in 0..GD {
                let s = s_f32[base + cc];
                let qf = (s * inv).round().clamp(-128.0, 127.0);
                codes[base + cc] = qf as i8;
                ef[base + cc] = f32_to_f16_bits(s - qf * sc);
            }
        }
        (codes, scales, ef)
    }
    struct GdnInputs {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        gate: Vec<f32>,
        beta: Vec<f32>,
        s0: Vec<f32>,
    }
    fn gdn_inputs(n: usize, salt: u32, gate_mode: u32) -> GdnInputs {
        let q = (0..n * GVD).map(|i| prng_f32(i as u64, salt) * 0.5).collect();
        let k = (0..n * GVD).map(|i| prng_f32(i as u64, salt ^ 1) * 0.5).collect();
        let v = (0..n * GVD).map(|i| prng_f32(i as u64, salt ^ 2) * 0.5).collect();
        let gate = (0..n * GH)
            .map(|i| match gate_mode {
                0 => -0.001 + prng_f32(i as u64, salt ^ 3) * 0.0001,
                1 => -0.1 + prng_f32(i as u64, salt ^ 3) * 0.01,
                _ => prng_f32(i as u64, salt ^ 3) * 3.0,
            })
            .collect();
        let beta = (0..n * GH)
            .map(|i| 0.2 + 0.8 * (prng_f32(i as u64, salt ^ 4) * 0.5 + 0.5))
            .collect();
        let s0 = (0..GH * GD * GD)
            .map(|i| prng_f32(i as u64, salt ^ 5) * 0.05)
            .collect();
        GdnInputs { q, k, v, gate, beta, s0 }
    }
    struct GdnState {
        sq: GpuTensor,
        sc: GpuTensor,
        ef: Option<GpuTensor>,
    }
    fn upload_gdn_state(gpu: &mut Gpu, codes: &[i8], scales: &[f32], ef: Option<&[u16]>) -> GdnState {
        let sq = upload_i8(&mut *gpu, codes, &[GH * GD * GD]);
        let sc = gpu.upload_f32(scales, &[GH * GD]).unwrap();
        let ef_t = ef.map(|e| upload_u16(&mut *gpu, e, &[GH * GD * GD]));
        GdnState { sq, sc, ef: ef_t }
    }
    fn download_gdn_state(gpu: &mut Gpu, st: &GdnState) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let a = download_raw(&mut *gpu, &st.sq);
        let b = download_f32_vec(&mut *gpu, &st.sc);
        let c = st.ef.as_ref().map(|t| download_raw(&mut *gpu, t)).unwrap_or_default();
        (a, b, c)
    }
    fn gdn_commit(
        gpu: &mut Gpu,
        st: &GdnState,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        g: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) {
        gpu.gated_delta_net_q8_batch_seq(
            q, k, v, g, b, &st.sq, &st.sc, out, n, GH, GD, st.ef.as_ref(),
        )
        .unwrap_or_else(|e| panic!("gdn commit n={n}: {e:?}"));
        sync(&mut *gpu);
    }
    /// Legacy schedule: independent per-commit tensors, sequential 512 commits.
    /// Candidate: single shared buffers + 512 views, same commit sequence, same
    /// unsliced S/scales/EF owners. Compares outputs + state after EACH commit.
    fn arm2_case(
        gpu: &mut Gpu,
        cmp: &mut Cmp,
        tag: &str,
        inp: &GdnInputs,
        n: usize,
        zero_state: bool,
        use_ef: bool,
    ) {
        let (codes, scales, efv) = if zero_state {
            (vec![0i8; GH * GD * GD], vec![1.0f32; GH * GD], vec![0u16; GH * GD * GD])
        } else {
            quantize_s(&inp.s0)
        };
        let ef_opt = use_ef.then(|| efv.clone());
        // Shared (candidate) buffers.
        let frame0 = gdn_requant_frame_checkpoint();
        restore_gdn_requant_frame_checkpoint(frame0);
        let cq = gpu.upload_f32(&inp.q, &[n, GVD]).unwrap();
        let ck = gpu.upload_f32(&inp.k, &[n, GVD]).unwrap();
        let cv = gpu.upload_f32(&inp.v, &[n, GVD]).unwrap();
        let cg = gpu.upload_f32(&inp.gate, &[n, GH]).unwrap();
        let cb = gpu.upload_f32(&inp.beta, &[n, GH]).unwrap();
        let cout = gpu.zeros(&[n, GVD], DType::F32).unwrap();
        let cst = upload_gdn_state(&mut *gpu, &codes, &scales, ef_opt.as_deref());
        // Legacy (independent) state + per-segment input clones.
        restore_gdn_requant_frame_checkpoint(frame0);
        let lst = upload_gdn_state(&mut *gpu, &codes, &scales, ef_opt.as_deref());
        let mut lout_h = vec![0u8; 0];
        let nseg = n.div_ceil(512);
        for s in 0..nseg {
            let (o, m) = (s * 512, (n - s * 512).min(512));
            // Candidate commit on views.
            restore_gdn_requant_frame_checkpoint(frame0 + (o as u32));
            gdn_commit(&mut *gpu, &cst,
            &cq.sub_offset(o * GVD, m * GVD),
            &ck.sub_offset(o * GVD, m * GVD),
            &cv.sub_offset(o * GVD, m * GVD),
            &cg.sub_offset(o * GH, m * GH),
            &cb.sub_offset(o * GH, m * GH),
            &cout.sub_offset(o * GVD, m * GVD),
            m,);
            // Legacy commit on independent clones of the same bytes.
            restore_gdn_requant_frame_checkpoint(frame0 + (o as u32));
            let lq = gpu.upload_f32(&inp.q[o * GVD..(o + m) * GVD], &[m, GVD]).unwrap();
            let lk = gpu.upload_f32(&inp.k[o * GVD..(o + m) * GVD], &[m, GVD]).unwrap();
            let lv = gpu.upload_f32(&inp.v[o * GVD..(o + m) * GVD], &[m, GVD]).unwrap();
            let lg = gpu.upload_f32(&inp.gate[o * GH..(o + m) * GH], &[m, GH]).unwrap();
            let lb = gpu.upload_f32(&inp.beta[o * GH..(o + m) * GH], &[m, GH]).unwrap();
            let lo = gpu.zeros(&[m, GVD], DType::F32).unwrap();
            gdn_commit(&mut *gpu, &lst, &lq, &lk, &lv, &lg, &lb, &lo, m);
            // Compare this commit's outputs + full state.
            let co = download_raw(&mut *gpu, &cout.sub_offset(o * GVD, m * GVD));
            let loh = download_f32_vec(&mut *gpu, &lo);
            cmp.check("arm2", &format!("{tag}/out-seg{s}"), &co, &loh);
            let (csq, csc, cef) = download_gdn_state(&mut *gpu, &cst);
            let (lsq, lsc, lef) = download_gdn_state(&mut *gpu, &lst);
            cmp.check("arm2", &format!("{tag}/S-seg{s}"), &csq, &lsq);
            cmp.check("arm2", &format!("{tag}/scales-seg{s}"), &csc, &lsc);
            if use_ef {
                cmp.check("arm2", &format!("{tag}/EF-seg{s}"), &cef, &lef);
            }
            lout_h.extend_from_slice(&loh);
            for t in [lq, lk, lv, lg, lb, lo] {
                gpu.free_tensor(t).unwrap();
            }
        }
        restore_gdn_requant_frame_checkpoint(frame0 + (n as u32));
        for t in [cq, ck, cv, cg, cb, cout, cst.sq, cst.sc] {
            gpu.free_tensor(t).unwrap();
        }
        if let Some(t) = cst.ef {
            gpu.free_tensor(t).unwrap();
        }
        for t in [lst.sq, lst.sc] {
            gpu.free_tensor(t).unwrap();
        }
        if let Some(t) = lst.ef {
            gpu.free_tensor(t).unwrap();
        }
        let _ = lout_h;
    }

    pub fn run_arm2(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        eprintln!("arm2: GDN seam differential (views vs independent, per-512 commits)");
        for &gate_mode in &[0u32, 1, 2] {
            arm2_case(&mut *gpu, cmp, &format!("c{c}-gate{gate_mode}"), &gdn_inputs(c, 0x6D4, gate_mode), c, false, true);
        }
        // Zero-state fixture.
        let zn = c.min(1024);
        arm2_case(&mut *gpu, cmp, "zero-state", &gdn_inputs(zn, 0x2E0, 2), zn, true, true);
        // Tails keep the original schedule: same commit sequence both arms.
        for &tn in &[513usize, 1025] {
            arm2_case(&mut *gpu, cmp, &format!("tail{tn}"), &gdn_inputs(tn, 0x7A1, 2), tn, false, true);
        }
        // EF-off determinism under identical frame checkpoints (oracle-only restore).
        let en = 512usize;
        arm2_case(&mut *gpu, cmp, "ef-off", &gdn_inputs(en, 0x0EF0, 2), en, false, false);
        // Per-token-requant exclusion route: same schedule both arms, must agree.
        let prev = std::env::var("HIPFIRE_DN_REQUANT_PER_TOKEN").ok();
        std::env::set_var("HIPFIRE_DN_REQUANT_PER_TOKEN", "1");
        let rn = 1024usize;
        arm2_case(&mut *gpu, cmp, "rpt-on", &gdn_inputs(rn, 0x09A7, 2), rn, false, true);
        match prev {
            Some(v) => std::env::set_var("HIPFIRE_DN_REQUANT_PER_TOKEN", v),
            None => std::env::remove_var("HIPFIRE_DN_REQUANT_PER_TOKEN"),
        }
    }
    // ══ ARM 3: convolution whole-C vs 512 sequence (identical nonzero ring)
    const CK: usize = 2048;
    const CV: usize = 6144;
    const CCH: usize = 2 * CK + CV; // 10240
    fn conv_case(gpu: &mut Gpu, cmp: &mut Cmp, tag: &str, n: usize) {
        let inp_h: Vec<f32> = (0..n * CCH).map(|i| prng_f32(i as u64, 0xC04) * 0.5).collect();
        let w_h: Vec<f32> = (0..CCH * 4).map(|i| prng_f32(i as u64, 0xE16) * 0.25).collect();
        let ring0: Vec<f32> = (0..CCH * 3).map(|i| prng_f32(i as u64, 0x916) * 0.25).collect();
        // Candidate: one whole-n call on shared buffers.
        let di = gpu.upload_f32(&inp_h, &[n, CCH]).unwrap();
        let dw = gpu.upload_f32(&w_h, &[CCH * 4]).unwrap();
        let ds = gpu.upload_f32(&ring0, &[CCH * 3]).unwrap();
        let (dq, dk, dv) = (
            gpu.zeros(&[n * CK], DType::F32).unwrap(),
            gpu.zeros(&[n * CK], DType::F32).unwrap(),
            gpu.zeros(&[n * CV], DType::F32).unwrap(),
        );
        gpu.conv1d_silu_split_f32_n(&dq, &dk, &dv, &di, &dw, &ds, CK, CV, n)
            .unwrap_or_else(|e| panic!("arm3/{tag} whole: {e:?}"));
        sync(&mut *gpu);
        // Legacy: sequential 512 (+tail) commits on views of clones.
        let di2 = gpu.upload_f32(&inp_h, &[n, CCH]).unwrap();
        let ds2 = gpu.upload_f32(&ring0, &[CCH * 3]).unwrap();
        let (dq2, dk2, dv2) = (
            gpu.zeros(&[n * CK], DType::F32).unwrap(),
            gpu.zeros(&[n * CK], DType::F32).unwrap(),
            gpu.zeros(&[n * CV], DType::F32).unwrap(),
        );
        let mut o = 0usize;
        let mut seg = 0usize;
        while o < n {
            let m = (n - o).min(512);
            gpu.conv1d_silu_split_f32_n(
                &dq2.sub_offset(o * CK, m * CK),
                &dk2.sub_offset(o * CK, m * CK),
                &dv2.sub_offset(o * CV, m * CV),
                &di2.sub_offset(o * CCH, m * CCH),
                &dw, &ds2, CK, CV, m,
            )
            .unwrap_or_else(|e| panic!("arm3/{tag} seg{seg}: {e:?}"));
            sync(&mut *gpu);
            o += m;
            seg += 1;
        }
        cmp.check("arm3", &format!("{tag}/q-raw"), &download_f32_vec(&mut *gpu, &dq), &download_f32_vec(&mut *gpu, &dq2));
        cmp.check("arm3", &format!("{tag}/k-raw"), &download_f32_vec(&mut *gpu, &dk), &download_f32_vec(&mut *gpu, &dk2));
        cmp.check("arm3", &format!("{tag}/v-raw"), &download_f32_vec(&mut *gpu, &dv), &download_f32_vec(&mut *gpu, &dv2));
        cmp.check("arm3", &format!("{tag}/ring-final"), &download_f32_vec(&mut *gpu, &ds), &download_f32_vec(&mut *gpu, &ds2));
        for t in [di, dw, ds, dq, dk, dv, di2, ds2, dq2, dk2, dv2] {
            gpu.free_tensor(t).unwrap();
        }
    }
    pub fn run_arm3(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        eprintln!("arm3: conv whole-C vs 512 sequence (nonzero ring)");
        conv_case(&mut *gpu, cmp, &format!("c{c}"), c);
        conv_case(&mut *gpu, cmp, "tail513", 513);
        conv_case(&mut *gpu, cmp, "tail1025", 1025);
    }

    // ══ ARM 4: attention 512-tile views vs independent operands (gfx1201 FA2)
    const ANH: usize = 24;
    const ANKV: usize = 4;
    const AHD: usize = 256;
    const ABPH: usize = AHD / 32;
    const AQ8B: usize = 34;
    const AROW: usize = ANKV * ABPH * AQ8B; // 1088 B/row Q8 cache
    const AQO: usize = ANH * AHD; // 6144
    const AF3H: usize = 100;
    const AF3P: usize = ANKV * AF3H; // 400 B/row fwht3 K
    fn pack_q8_blk(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
        let mut out = [0u8; 34];
        let s = f32_to_f16_bits(scale);
        out[0] = (s & 0xff) as u8;
        out[1] = (s >> 8) as u8;
        for w in 0..8 {
            let mut u = 0u32;
            for cc in 0..4 {
                u |= ((codes[w * 4 + cc] as u8) as u32) << (cc * 8);
            }
            out[2 + w * 4..2 + w * 4 + 4].copy_from_slice(&u.to_le_bytes());
        }
        out
    }
    fn fill_q8(seq_len: usize, salt: u32) -> Vec<u8> {
        let mut buf = vec![0u8; seq_len * AROW];
        for g in 0..seq_len {
            for kv in 0..ANKV {
                for b in 0..ABPH {
                    let mut codes = [0i8; 32];
                    for cc in 0..32 {
                        let r = prng_u32((g * 64 + kv * 8 + b) as u64 * 32 + cc as u64, salt);
                        codes[cc] = ((r % 61) as i8).wrapping_sub(30);
                    }
                    let scale = 0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
                    let blk = pack_q8_blk(scale, &codes);
                    let off = g * AROW + (kv * ABPH + b) * AQ8B;
                    buf[off..off + 34].copy_from_slice(&blk);
                }
            }
        }
        buf
    }
    fn fill_f3k(seq_len: usize, salt: u32) -> Vec<u8> {
        let mut buf = vec![0u8; seq_len * AF3P];
        for g in 0..seq_len {
            for kv in 0..ANKV {
                let base = g * AF3P + kv * AF3H;
                let cnorm = 0.05 + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
                buf[base..base + 4].copy_from_slice(&cnorm.to_le_bytes());
                for g3 in 0..32 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        packed |= ((prng_u32((g * 256 + kv * 64 + g3 * 8 + i) as u64, salt) % 8) as u32) << (3 * i);
                    }
                    buf[base + 4 + g3 * 3..base + 4 + g3 * 3 + 3].copy_from_slice(&packed.to_le_bytes()[..3]);
                }
            }
        }
        buf
    }
    fn pos_bytes_of(pos: &[i32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(pos.len() * 4);
        for &p in pos {
            b.extend_from_slice(&p.to_le_bytes());
        }
        b
    }
    fn check_pos_view(gpu: &mut Gpu, cmp: &mut Cmp, arm: &str, tag: &str, base: &GpuTensor, view: &GpuTensor, start: usize, row0: usize, n: usize) {
        // Geometry: view byte pointer must sit exactly 4*row0 past the base
        // (Raw positions: 4 bytes per i32; mirrors the repaired F1 contract).
        let delta = (view.buf.as_ptr() as usize).wrapping_sub(base.buf.as_ptr() as usize);
        if delta != 4 * row0 {
            cmp.failures += 1;
            eprintln!("  FAIL {arm} {tag}/pos-geometry: byte_delta={delta} want={}", 4 * row0);
        }
        let got = download_raw(&mut *gpu, view);
        let mut exp = Vec::with_capacity(n * 4);
        for b in 0..n {
            exp.extend_from_slice(&((start + row0 + b) as i32).to_le_bytes());
        }
        cmp.check(arm, &format!("{tag}/pos-readback"), &got, &exp);
    }
    /// Tiled-views arm vs independent-operands arm at (batch=C, start, seq_len).
    /// Per-tile launch args identical: batch=512, tile-local max_ctx=start+o+512.
    fn attn_case(gpu: &mut Gpu, cmp: &mut Cmp, tag: &str, batch: usize, start: usize, seq_len: usize, fwht3: bool, s1: &GpuTensor, s2: &GpuTensor) {
        assert!(batch % 512 == 0 && batch >= 512);
        let pos_h: Vec<i32> = (0..batch).map(|b| (start + b) as i32).collect();
        let q_h: Vec<f32> = (0..batch * AQO).map(|i| prng_f32(i as u64, 0x911) * 0.5).collect();
        let k_h = if fwht3 { fill_f3k(seq_len, 0x922) } else { fill_q8(seq_len, 0x922) };
        let v_h = fill_q8(seq_len, 0x933);
        let pos_b = pos_bytes_of(&pos_h);
        let launch = |g: &mut Gpu, q: &GpuTensor, kk: &GpuTensor, vv: &GpuTensor, o: &GpuTensor, p: &GpuTensor, bsz: usize, mctx: usize| {
            if fwht3 {
                g.attention_q8_0_fa2_gqa_fwht3k_gfx1201(q, kk, vv, o, p, s1, s2, ANH, ANKV, AHD, mctx, bsz)
            } else {
                g.attention_q8_0_fa2_gqa_gfx1201(q, kk, vv, o, p, ANH, ANKV, AHD, mctx, bsz)
            }
            .unwrap_or_else(|e| panic!("arm4/{tag} launch bsz={bsz}: {e:?}"));
            sync(g);
        };
        // Candidate: shared buffers + views.
        let dk = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
        let dv = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
        let dq = gpu.upload_f32(&q_h, &[batch, ANH, AHD]).unwrap();
        let dp = gpu.upload_raw(&pos_b, &[pos_b.len()]).unwrap();
        let dout = gpu.zeros(&[batch * AQO], DType::F32).unwrap();
        for s in 0..batch / 512 {
            let o = s * 512;
            let qv = dq.sub_offset(o * AQO, 512 * AQO);
            let ov = dout.sub_offset(o * AQO, 512 * AQO);
            let pv = dp.sub_offset(4 * o, 4 * 512);
            check_pos_view(&mut *gpu, cmp, "arm4", &format!("{tag}/tile{s}"), &dp, &pv, start, o, 512);
            launch(&mut *gpu, &qv, &dk, &dv, &ov, &pv, 512, start + o + 512);
        }
        // Legacy: independent operands per tile (same bytes, fresh tensors).
        let mut indep_out = vec![0u8; 0];
        let mut indep_q = vec![0u8; 0];
        for s in 0..batch / 512 {
            let o = s * 512;
            let dk2 = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
            let dv2 = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
            let dq2 = gpu.upload_f32(&q_h[o * AQO..(o + 512) * AQO], &[512, ANH, AHD]).unwrap();
            let dp2 = gpu.upload_raw(&pos_b[4 * o..4 * (o + 512)], &[4 * 512]).unwrap();
            let do2 = gpu.zeros(&[512 * AQO], DType::F32).unwrap();
            let pv2 = dp2.sub_offset(0, 4 * 512);
            launch(&mut *gpu, &dq2, &dk2, &dv2, &do2, &pv2, 512, start + o + 512);
            indep_out.extend_from_slice(&download_f32_vec(&mut *gpu, &do2));
            indep_q.extend_from_slice(&download_f32_vec(&mut *gpu, &dq2));
            // Cache bytes after each tile must be untouched.
            cmp.check("arm4", &format!("{tag}/tile{s}/k-guard"), &download_raw(&mut *gpu, &dk2), &k_h);
            cmp.check("arm4", &format!("{tag}/tile{s}/v-guard"), &download_raw(&mut *gpu, &dv2), &v_h);
            for t in [dk2, dv2, dq2, dp2, do2] {
                gpu.free_tensor(t).unwrap();
            }
        }
        cmp.check("arm4", &format!("{tag}/out"), &download_f32_vec(&mut *gpu, &dout), &indep_out);
        cmp.check("arm4", &format!("{tag}/q-unchanged"), &download_f32_vec(&mut *gpu, &dq), &indep_q);
        cmp.check("arm4", &format!("{tag}/k-cache"), &download_raw(&mut *gpu, &dk), &k_h);
        cmp.check("arm4", &format!("{tag}/v-cache"), &download_raw(&mut *gpu, &dv), &v_h);
        cmp.check("arm4", &format!("{tag}/pos-guard"), &download_raw(&mut *gpu, &dp), &pos_b);
        for t in [dk, dv, dq, dp, dout] {
            gpu.free_tensor(t).unwrap();
        }
    }
    pub fn run_arm4(gpu: &mut Gpu, cmp: &mut Cmp, c: usize) {
        eprintln!("arm4: FA2 512-tile views vs independent operands (gfx1201)");
        let s1h = gen_fwht_signs(42, 256);
        let s2h = gen_fwht_signs(1042, 256);
        let d_s1 = gpu.upload_f32(&s1h, &[256]).unwrap();
        let d_s2 = gpu.upload_f32(&s2h, &[256]).unwrap();
        // (batch, start, seq_len): aligned, unaligned resume, near 8192/32768.
        let cases: &[(usize, usize, usize)] = &[
            (512, 0, 512),
            (512, 1000, 1512),
            (512, 7680, 8192),
            (512, 32256, 32768),
            (1024, 0, 1024),
            (1024, 7168, 8192),
        ];
        for &(b, st, seqlen) in cases {
            attn_case(&mut *gpu, cmp, &format!("q8-b{b}-s{st}-L{seqlen}"), b, st, seqlen, false, &d_s1, &d_s2);
            attn_case(&mut *gpu, cmp, &format!("f3-b{b}-s{st}-L{seqlen}"), b, st, seqlen, true, &d_s1, &d_s2);
        }
        if c >= 1024 {
            attn_case(&mut *gpu, cmp, &format!("q8-b{c}-s0"), c, 0, c, false, &d_s1, &d_s2);
            attn_case(&mut *gpu, cmp, &format!("f3-b{c}-s0"), c, 0, c, true, &d_s1, &d_s2);
        }
        gpu.free_tensor(d_s1).unwrap();
        gpu.free_tensor(d_s2).unwrap();
    }
    // ══ ARM 5: full model, fresh + resumed-prefix, 512 schedule vs C schedule
    pub struct ModelCtx {
        pub config: qwen35::Qwen35Config,
        pub weights: qwen35::Qwen35Weights,
    }
    pub fn load_model(gpu: &mut Gpu, model: &Path) -> ModelCtx {
        let mut hfq = HfqFile::open(model).expect("open model");
        let config = qwen35::config_from_hfq(&hfq).expect("read config");
        eprintln!(
            "model: dim={} layers={} kv_heads={} head_dim={} vocab={} la_k={}x{} la_v={}x{}",
            config.dim, config.n_layers, config.n_kv_heads, config.head_dim,
            config.vocab_size, config.linear_num_key_heads, config.linear_key_head_dim,
            config.linear_num_value_heads, config.linear_value_head_dim,
        );
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        let weights = qwen35::load_weights(&mut src, std::slice::from_mut(&mut *gpu), &layout)
            .expect("load weights");
        ModelCtx { config, weights }
    }
    fn det_tokens(l: usize, vocab: usize) -> Vec<u32> {
        (0..l).map(|i| (((i * 131) % (vocab - 1)) + 1) as u32).collect()
    }
    pub struct Snap {
        pub dn_s: Vec<Vec<u8>>,
        pub dn_sc: Vec<Vec<u8>>,
        pub dn_ef: Vec<Vec<u8>>,
        pub conv: Vec<Vec<u8>>,
        pub kv_k: Vec<Vec<u8>>,
        pub kv_v: Vec<Vec<u8>>,
        pub kv_ks: Vec<Vec<u8>>,
        pub kv_vs: Vec<Vec<u8>>,
        pub hidden: Vec<u8>,
        pub logits: Vec<u8>,
    }
    fn snap_state(gpu: &mut Gpu, dn: &DeltaNetState, kv: &KvCache, scratch: &Qwen35Scratch, hbuf: &GpuTensor) -> Snap {
        let dn_s = dn.s_matrices.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let dn_sc = dn.s_scales.iter().map(|t| download_f32_vec(&mut *gpu, t)).collect();
        let dn_ef = dn.s_ef_residual.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let conv = dn.conv_states.iter().map(|t| download_f32_vec(&mut *gpu, t)).collect();
        let kv_k = kv.k_gpu.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let kv_v = kv.v_gpu.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let kv_ks = kv.k_scales.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let kv_vs = kv.v_scales.iter().map(|t| download_raw(&mut *gpu, t)).collect();
        let hidden = download_f32_vec(&mut *gpu, hbuf);
        let logits = download_f32_vec(&mut *gpu, &scratch.logits);
        Snap { dn_s, dn_sc, dn_ef, conv, kv_k, kv_v, kv_ks, kv_vs, hidden, logits }
    }
    fn cmp_snap(cmp: &mut Cmp, arm: &str, tag: &str, a: &Snap, b: &Snap, skip_kv: bool) {
        assert_eq!(a.dn_s.len(), b.dn_s.len(), "dn layer count");
        assert!(!a.dn_s.is_empty(), "no DN layers?");
        for (li, (x, y)) in a.dn_s.iter().zip(b.dn_s.iter()).enumerate() {
            cmp.check(arm, &format!("{tag}/L{li}/S"), x, y);
        }
        for (li, (x, y)) in a.dn_sc.iter().zip(b.dn_sc.iter()).enumerate() {
            cmp.check(arm, &format!("{tag}/L{li}/scales"), x, y);
        }
        assert_eq!(a.dn_ef.len(), b.dn_ef.len(), "EF layer count");
        assert!(!a.dn_ef.is_empty(), "{tag}: EF residual empty — not Q8+EF, admission void");
        for (li, (x, y)) in a.dn_ef.iter().zip(b.dn_ef.iter()).enumerate() {
            cmp.check(arm, &format!("{tag}/L{li}/EF"), x, y);
        }
        for (li, (x, y)) in a.conv.iter().zip(b.conv.iter()).enumerate() {
            cmp.check(arm, &format!("{tag}/L{li}/conv"), x, y);
        }
        if !skip_kv {
            for (li, (x, y)) in a.kv_k.iter().zip(b.kv_k.iter()).enumerate() {
                cmp.check(arm, &format!("{tag}/L{li}/kv-k"), x, y);
            }
            for (li, (x, y)) in a.kv_v.iter().zip(b.kv_v.iter()).enumerate() {
                cmp.check(arm, &format!("{tag}/L{li}/kv-v"), x, y);
            }
            for (li, (x, y)) in a.kv_ks.iter().zip(b.kv_ks.iter()).enumerate() {
                cmp.check(arm, &format!("{tag}/L{li}/kv-ksc"), x, y);
            }
            for (li, (x, y)) in a.kv_vs.iter().zip(b.kv_vs.iter()).enumerate() {
                cmp.check(arm, &format!("{tag}/L{li}/kv-vsc"), x, y);
            }
        }
        cmp.check(arm, &format!("{tag}/hidden"), &a.hidden, &b.hidden);
        cmp.check(arm, &format!("{tag}/logits"), &a.logits, &b.logits);
    }
    /// One full prefill run under `ceiling`: fresh KV/DN/scratch, tokens[0..l].
    /// If `resume_p > 0`, two calls (prefix + extend) on the same state.
    fn full_run(gpu: &mut Gpu, m: &ModelCtx, l: usize, ceiling: usize, resume_p: usize) -> Snap {
        std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", format!("{ceiling}"));
        let kv_seq = (l + 16).max(512);
        let mut kv = KvCache::new_gpu_q8(&mut *gpu, m.config.n_layers, m.config.n_kv_heads, m.config.head_dim, kv_seq)
            .unwrap_or_else(|e| panic!("kv q8 kv_seq={kv_seq}: {e:?}"));
        let mut dn = DeltaNetState::new(&mut *gpu, &m.config).expect("dn fresh");
        assert!(!dn.s_ef_residual.is_empty(), "fresh DN has no EF — not Q8+EF");
        let admitted = qwen35::ordinary_prefill_chunk_limit(&*gpu, &m.weights, &m.config, &dn, &kv, None);
        match &admitted {
            Ok(a) => eprintln!("  run L={l} resume_p={resume_p}: requested={ceiling} admitted={a}"),
            Err(e) => eprintln!("  run L={l} resume_p={resume_p}: requested={ceiling} admission-query failed: {e:?}"),
        }
        let scratch = Qwen35Scratch::new_with_kv_max(&mut *gpu, &m.config, 128, kv_seq)
            .unwrap_or_else(|e| panic!("scratch kv_seq={kv_seq}: {e:?}"));
        let hbuf = gpu.alloc_tensor(&[l, m.config.dim], DType::F32).expect("hbuf");
        let toks = det_tokens(l, m.config.vocab_size);
        if resume_p == 0 {
            qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &toks, 0, &mut kv, &mut dn, &scratch, None, Some(&hbuf), None, None)
                .unwrap_or_else(|e| panic!("forward l={l} ceiling={ceiling}: {e:?}"));
        } else {
            let hbuf_p = gpu.alloc_tensor(&[resume_p, m.config.dim], DType::F32).expect("hbuf-p");
            qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &toks[..resume_p], 0, &mut kv, &mut dn, &scratch, None, Some(&hbuf_p), None, None)
                .unwrap_or_else(|e| panic!("forward prefix p={resume_p}: {e:?}"));
            let hbuf2 = gpu.alloc_tensor(&[(l - resume_p), m.config.dim], DType::F32).expect("hbuf2");
            qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &toks[resume_p..], resume_p, &mut kv, &mut dn, &scratch, None, Some(&hbuf2), None, None)
                .unwrap_or_else(|e| panic!("forward extend l={l} p={resume_p}: {e:?}"));
            sync(&mut *gpu);
            // Splice prefix + extend hidden rows into hbuf for comparison.
            let pre = download_f32_vec(&mut *gpu, &hbuf_p);
            let ext = download_f32_vec(&mut *gpu, &hbuf2);
            let mut full = vec![0u8; l * m.config.dim * 4];
            full[..resume_p * m.config.dim * 4].copy_from_slice(&pre);
            full[resume_p * m.config.dim * 4..].copy_from_slice(&ext);
            gpu.hip.memcpy_htod(&hbuf.buf, &full).unwrap();
            gpu.free_tensor(hbuf_p).unwrap();
            gpu.free_tensor(hbuf2).unwrap();
        }
        sync(&mut *gpu);
        let s = snap_state(&mut *gpu, &dn, &kv, &scratch, &hbuf);
        gpu.free_tensor(hbuf).unwrap();
        kv.free_gpu(&mut *gpu).unwrap();
        dn.free_gpu(&mut *gpu);
        scratch.free_gpu(&mut *gpu).unwrap();
        s
    }
    pub fn run_arm5(gpu: &mut Gpu, cmp: &mut Cmp, c: usize, m: &ModelCtx) {
        eprintln!("arm5: full model fresh + resumed-prefix, 512 vs {c}");
        let lens: &[usize] = &[512, 1024, 2048, 4096, 8192, 32768, 511, 513, 1023, 1025, 1537, 2049, 4097, 8193];
        for &l in lens {
            let p = l / 2;
            eprintln!("  L={l} (resume_p={p}) ...");
            let f512 = full_run(&mut *gpu, m, l, 512, 0);
            let fc = full_run(&mut *gpu, m, l, c, 0);
            cmp_snap(cmp, "arm5", &format!("L{l}/fresh"), &f512, &fc, false);
            // Same call sequence across schedules is the invariant (fresh512 vs
            // freshC above, resume512 vs resumeC above). Resume-vs-fresh is NOT
            // asserted: different API call sequences legitimately differ (GDN
            // single-end requant boundaries + batch-size-dependent GEMM/FA
            // numerics), as the 512 baseline itself demonstrates.
            let r512 = full_run(&mut *gpu, m, l, 512, p);
            let rc = full_run(&mut *gpu, m, l, c, p);
            cmp_snap(cmp, "arm5", &format!("L{l}/resume"), &r512, &rc, false);
        }
    }
    // ══ ARM 6: transitions / exclusions
    pub fn run_arm6(gpu: &mut Gpu, cmp: &mut Cmp, c: usize, m: Option<&ModelCtx>) {
        eprintln!("arm6: transitions/exclusions");
        // 6a. GDN save→restore→extend.
        {
            let n = 1024usize;
            let inp = gdn_inputs(n, 0x6A, 2);
            let (codes, scales, efv) = quantize_s(&inp.s0);
            let frame0 = gdn_requant_frame_checkpoint();
            let q = gpu.upload_f32(&inp.q, &[n, GVD]).unwrap();
            let k = gpu.upload_f32(&inp.k, &[n, GVD]).unwrap();
            let v = gpu.upload_f32(&inp.v, &[n, GVD]).unwrap();
            let g = gpu.upload_f32(&inp.gate, &[n, GH]).unwrap();
            let b = gpu.upload_f32(&inp.beta, &[n, GH]).unwrap();
            let o = gpu.zeros(&[n, GVD], DType::F32).unwrap();
            let st = upload_gdn_state(&mut *gpu, &codes, &scales, Some(&efv));
            restore_gdn_requant_frame_checkpoint(frame0);
            gdn_commit(&mut *gpu, &st, &q.sub_offset(0, 512*GVD), &k.sub_offset(0, 512*GVD), &v.sub_offset(0, 512*GVD), &g.sub_offset(0, 512*GH), &b.sub_offset(0, 512*GH), &o.sub_offset(0, 512*GVD), 512);
            let saved = download_gdn_state(&mut *gpu, &st);
            let frame1 = gdn_requant_frame_checkpoint();
            gdn_commit(&mut *gpu, &st, &q.sub_offset(512*GVD, 512*GVD), &k.sub_offset(512*GVD, 512*GVD), &v.sub_offset(512*GVD, 512*GVD), &g.sub_offset(512*GH, 512*GH), &b.sub_offset(512*GH, 512*GH), &o.sub_offset(512*GVD, 512*GVD), 512);
            let out_full = download_raw(&mut *gpu, &o.sub_offset(512*GVD, 512*GVD));
            let st_full = download_gdn_state(&mut *gpu, &st);
            // Restore snapshot bytes + frame, re-extend with the same rows.
            gpu.hip.memcpy_htod(&st.sq.buf, &saved.0).unwrap();
            gpu.hip.memcpy_htod(&st.sc.buf, &saved.1).unwrap();
            if let Some(t) = st.ef.as_ref() {
                gpu.hip.memcpy_htod(&t.buf, &saved.2).unwrap();
            }
            restore_gdn_requant_frame_checkpoint(frame1);
            let o2 = gpu.zeros(&[512, GVD], DType::F32).unwrap();
            gdn_commit(&mut *gpu, &st, &q.sub_offset(512*GVD, 512*GVD), &k.sub_offset(512*GVD, 512*GVD), &v.sub_offset(512*GVD, 512*GVD), &g.sub_offset(512*GH, 512*GH), &b.sub_offset(512*GH, 512*GH), &o2, 512);
            cmp.check("arm6", "save-restore/out", &download_f32_vec(&mut *gpu, &o2), &out_full);
            let st2 = download_gdn_state(&mut *gpu, &st);
            cmp.check("arm6", "save-restore/S", &st2.0, &st_full.0);
            cmp.check("arm6", "save-restore/scales", &st2.1, &st_full.1);
            cmp.check("arm6", "save-restore/EF", &st2.2, &st_full.2);
            restore_gdn_requant_frame_checkpoint(frame0 + (n as u32));
            for t in [q, k, v, g, b, o, o2, st.sq, st.sc] {
                gpu.free_tensor(t).unwrap();
            }
            if let Some(t) = st.ef {
                gpu.free_tensor(t).unwrap();
            }
        }
        // 6d. EF-off frame determinism (oracle-only checkpoint restore).
        {
            let n = 512usize;
            let inp = gdn_inputs(n, 0x60, 2);
            let (codes, scales, _) = quantize_s(&inp.s0);
            let frame0 = gdn_requant_frame_checkpoint();
            let run = |g: &mut Gpu| -> (Vec<u8>, Vec<u8>) {
                restore_gdn_requant_frame_checkpoint(frame0);
                let q = g.upload_f32(&inp.q, &[n, GVD]).unwrap();
                let k = g.upload_f32(&inp.k, &[n, GVD]).unwrap();
                let vv = g.upload_f32(&inp.v, &[n, GVD]).unwrap();
                let gg = g.upload_f32(&inp.gate, &[n, GH]).unwrap();
                let b = g.upload_f32(&inp.beta, &[n, GH]).unwrap();
                let o = g.zeros(&[n, GVD], DType::F32).unwrap();
                let st = upload_gdn_state(g, &codes, &scales, None);
                gdn_commit(g, &st, &q, &k, &vv, &gg, &b, &o, n);
                let oh = download_f32_vec(g, &o);
                let sh = download_gdn_state(g, &st);
                for t in [q, k, vv, gg, b, o, st.sq, st.sc] {
                    g.free_tensor(t).unwrap();
                }
                (oh, sh.0)
            };
            let (o1, s1) = run(&mut *gpu);
            let (o2, s2) = run(&mut *gpu);
            cmp.check("arm6", "ef-off-determinism/out", &o1, &o2);
            cmp.check("arm6", "ef-off-determinism/S", &s1, &s2);
            restore_gdn_requant_frame_checkpoint(frame0);
        }
        // 6e. Excluded capture/recording path: must be off in this process.
        {
            let cap = gpu.graphs.capture_mode;
            let rec = gpu.replay.is_recording();
            eprintln!("  arm6/capture-mode={cap} replay-recording={rec}");
            if cap || rec {
                cmp.failures += 1;
                eprintln!("  FAIL arm6 capture/recording active — widened route would be misadmitted");
            }
        }
        // 6f. Allocation failure before publication: state untouched.
        {
            let n = 512usize;
            let inp = gdn_inputs(n, 0x6F, 2);
            let (codes, scales, efv) = quantize_s(&inp.s0);
            let st = upload_gdn_state(&mut *gpu, &codes, &scales, Some(&efv));
            let before = download_gdn_state(&mut *gpu, &st);
            let huge: Result<GpuTensor, _> = gpu.alloc_tensor(&[(1usize << 40)], DType::Raw);
            match huge {
                Ok(t) => {
                    gpu.free_tensor(t).unwrap();
                    cmp.failures += 1;
                    eprintln!("  FAIL arm6 alloc-failure: 1TB alloc unexpectedly succeeded");
                }
                Err(_) => eprintln!("  ok   arm6 alloc-failure refused as expected"),
            }
            let after = download_gdn_state(&mut *gpu, &st);
            cmp.check("arm6", "alloc-failure/S-untouched", &before.0, &after.0);
            cmp.check("arm6", "alloc-failure/scales-untouched", &before.1, &after.1);
            cmp.check("arm6", "alloc-failure/EF-untouched", &before.2, &after.2);
            for t in [st.sq, st.sc] {
                gpu.free_tensor(t).unwrap();
            }
            if let Some(t) = st.ef {
                gpu.free_tensor(t).unwrap();
            }
        }
        // 6g. Negative control: reused-without-reset state MUST differ (non-vacuous).
        {
            let n = 512usize;
            let inp = gdn_inputs(n, 0x66, 2);
            let (codes, scales, efv) = quantize_s(&inp.s0);
            let frame0 = gdn_requant_frame_checkpoint();
            let q = gpu.upload_f32(&inp.q, &[n, GVD]).unwrap();
            let k = gpu.upload_f32(&inp.k, &[n, GVD]).unwrap();
            let vv = gpu.upload_f32(&inp.v, &[n, GVD]).unwrap();
            let gg = gpu.upload_f32(&inp.gate, &[n, GH]).unwrap();
            let b = gpu.upload_f32(&inp.beta, &[n, GH]).unwrap();
            let st = upload_gdn_state(&mut *gpu, &codes, &scales, Some(&efv));
            restore_gdn_requant_frame_checkpoint(frame0);
            let o1 = gpu.zeros(&[n, GVD], DType::F32).unwrap();
            gdn_commit(&mut *gpu, &st, &q, &k, &vv, &gg, &b, &o1, n);
            let s_after_first = download_gdn_state(&mut *gpu, &st).0;
            let o2 = gpu.zeros(&[n, GVD], DType::F32).unwrap();
            gdn_commit(&mut *gpu, &st, &q, &k, &vv, &gg, &b, &o2, n);
            let s_after_second = download_gdn_state(&mut *gpu, &st).0;
            if first_diff(&s_after_first, &s_after_second).is_none() {
                cmp.failures += 1;
                eprintln!("  FAIL arm6 negative-control: reused state identical — comparisons vacuous");
            } else {
                eprintln!("  ok   arm6 negative-control: reused state differs as expected");
            }
            restore_gdn_requant_frame_checkpoint(frame0);
            for t in [q, k, vv, gg, b, o1, o2, st.sq, st.sc] {
                gpu.free_tensor(t).unwrap();
            }
            if let Some(t) = st.ef {
                gpu.free_tensor(t).unwrap();
            }
        }
        // Model-gated checks.
        if let Some(m) = m {
            // 6b. reset→same prompt equals fresh.
            {
                let l = 1024usize;
                std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", "512");
                let toks = det_tokens(l, m.config.vocab_size);
                let kv_seq = (l + 16).max(512);
                let fresh = full_run(&mut *gpu, m, l, 512, 0);
                let mut kv = KvCache::new_gpu_q8(&mut *gpu, m.config.n_layers, m.config.n_kv_heads, m.config.head_dim, kv_seq).unwrap();
                let mut dn = DeltaNetState::new(&mut *gpu, &m.config).unwrap();
                let scratch = Qwen35Scratch::new_with_kv_max(&mut *gpu, &m.config, 128, kv_seq).unwrap();
                // Pollute, then reset, then run the same prompt.
                let junk: Vec<u32> = vec![7; 256];
                qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &junk, 0, &mut kv, &mut dn, &scratch, None, None, None, None).unwrap();
                dn.reset(&mut *gpu).unwrap();
                let mut kv2 = KvCache::new_gpu_q8(&mut *gpu, m.config.n_layers, m.config.n_kv_heads, m.config.head_dim, kv_seq).unwrap();
                let hbuf = gpu.alloc_tensor(&[l, m.config.dim], DType::F32).unwrap();
                qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &toks, 0, &mut kv2, &mut dn, &scratch, None, Some(&hbuf), None, None).unwrap();
                sync(&mut *gpu);
                let reset_snap = snap_state(&mut *gpu, &dn, &kv2, &scratch, &hbuf);
                // Fresh-vs-reset: DN + hidden + logits must match (KV fresh both).
                cmp_snap(cmp, "arm6", "reset-vs-fresh", &fresh, &reset_snap, true);
                gpu.free_tensor(hbuf).unwrap();
                kv.free_gpu(&mut *gpu).unwrap();
                kv2.free_gpu(&mut *gpu).unwrap();
                dn.free_gpu(&mut *gpu);
                scratch.free_gpu(&mut *gpu).unwrap();
            }
            // 6c. Tiny caller PBS: the explicit cap wins at any ceiling — the
            // capped path must be ceiling-invariant (both sides chunk at 256).
            // (Capped-vs-uncapped is NOT asserted: different chunk sizes
            // legitimately differ, as the baseline sensitivity check shows.)
            {
                let l = 1024usize;
                let toks = det_tokens(l, m.config.vocab_size);
                let kv_seq = (l + 16).max(512);
                let mut tiny_at = |ceiling: usize| -> Snap {
                    std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", format!("{ceiling}"));
                    let mut kv = KvCache::new_gpu_q8(&mut *gpu, m.config.n_layers, m.config.n_kv_heads, m.config.head_dim, kv_seq).unwrap();
                    let mut dn = DeltaNetState::new(&mut *gpu, &m.config).unwrap();
                    let scratch = Qwen35Scratch::new_with_kv_max(&mut *gpu, &m.config, 128, kv_seq).unwrap();
                    let pbs = PrefillBatchScratch::new(&mut *gpu, &m.config, 256).expect("tiny pbs");
                    let hbuf = gpu.alloc_tensor(&[l, m.config.dim], DType::F32).unwrap();
                    qwen35::forward_prefill_batch_with_pbs(&mut *gpu, &m.weights, &m.config, &toks, 0, &mut kv, &mut dn, &scratch, None, Some(&hbuf), None, None, Some(&pbs), None, None).unwrap();
                    sync(&mut *gpu);
                    let s = snap_state(&mut *gpu, &dn, &kv, &scratch, &hbuf);
                    gpu.free_tensor(hbuf).unwrap();
                    kv.free_gpu(&mut *gpu).unwrap();
                    dn.free_gpu(&mut *gpu);
                    scratch.free_gpu(&mut *gpu).unwrap();
                    pbs.free_gpu(&mut *gpu).unwrap();
                    s
                };
                let t512 = tiny_at(512);
                let tc = tiny_at(c);
                cmp_snap(cmp, "arm6", "tiny-pbs-ceiling-invariant", &t512, &tc, false);
            }
            // 6h. Hidden-ring wrap: the ring-capped path (chunks ≤256 via
            // ring.max_batch) must be ceiling-invariant. (Ring-vs-plain is NOT
            // asserted: the cap re-chunks by construction.)
            {
                let l = 600usize;
                let toks = det_tokens(l, m.config.vocab_size);
                let kv_seq = (l + 16).max(512);
                let mid = m.config.n_layers / 2;
                let mut ring_at = |ceiling: usize| -> Snap {
                    std::env::set_var("HIPFIRE_PREFILL_MAX_BATCH", format!("{ceiling}"));
                    let mut kv = KvCache::new_gpu_q8(&mut *gpu, m.config.n_layers, m.config.n_kv_heads, m.config.head_dim, kv_seq).unwrap();
                    let mut dn = DeltaNetState::new(&mut *gpu, &m.config).unwrap();
                    let scratch = Qwen35Scratch::new_with_kv_max(&mut *gpu, &m.config, 128, kv_seq).unwrap();
                    let mut ring = HiddenStateRingBuffer::new_for_layers(&mut *gpu, &[mid], m.config.dim, 256, 256).expect("ring");
                    let hbuf = gpu.alloc_tensor(&[l, m.config.dim], DType::F32).unwrap();
                    qwen35::forward_prefill_batch(&mut *gpu, &m.weights, &m.config, &toks, 0, &mut kv, &mut dn, &scratch, Some(&mut ring), Some(&hbuf), None, None).unwrap();
                    sync(&mut *gpu);
                    let s = snap_state(&mut *gpu, &dn, &kv, &scratch, &hbuf);
                    gpu.free_tensor(hbuf).unwrap();
                    kv.free_gpu(&mut *gpu).unwrap();
                    dn.free_gpu(&mut *gpu);
                    scratch.free_gpu(&mut *gpu).unwrap();
                    for t in ring.layer_bufs.into_iter().chain(ring.staging_bufs) {
                        gpu.free_tensor(t).unwrap();
                    }
                    s
                };
                let r512 = ring_at(512);
                let rc = ring_at(c);
                cmp_snap(cmp, "arm6", "hidden-ring-ceiling-invariant", &r512, &rc, false);
            }
        }
        let _ = c;
    }
    // ══ driver ══
    pub struct Args {
        pub model: Option<PathBuf>,
        pub chunk: usize,
        pub out_dir: PathBuf,
        pub arms: Vec<usize>,
    }
    pub fn parse_args() -> Args {
        let a: Vec<String> = std::env::args().collect();
        let mut model = None;
        let mut chunk = 0usize;
        let mut out_dir = PathBuf::from(".redline-work/gfx1201-chunk");
        let mut arms = vec![1, 2, 3, 4, 5, 6];
        let mut i = 1;
        while i < a.len() {
            match a[i].as_str() {
                "--model" => { model = Some(PathBuf::from(&a[i + 1])); i += 2; }
                "--chunk" => { chunk = a[i + 1].parse().unwrap_or(0); i += 2; }
                "--out-dir" => { out_dir = PathBuf::from(&a[i + 1]); i += 2; }
                "--arms" => {
                    arms = a[i + 1].split(',').map(|s| s.parse().expect("--arms values")).collect();
                    i += 2;
                }
                "--skip-model" => {
                    arms.retain(|x| *x != 5);
                    i += 1;
                }
                h => {
                    eprintln!("unknown flag {h}; want --model --chunk --out-dir [--arms] [--skip-model]");
                    std::process::exit(2);
                }
            }
        }
        if ![512, 1024, 2048, 4096, 8192].contains(&chunk) {
            eprintln!("--chunk must be one of 512,1024,2048,4096,8192 (got {chunk})");
            std::process::exit(2);
        }
        Args { model, chunk, out_dir, arms }
    }
    pub fn real_main() {
        let args = parse_args();
        let c = args.chunk;
        std::fs::create_dir_all(&args.out_dir).unwrap_or_else(|e| {
            eprintln!("out-dir: {e:?}");
            std::process::exit(2);
        });
        let mut gpu = Gpu::init().unwrap_or_else(|e| {
            eprintln!("gpu init failed: {e:?}");
            std::process::exit(2);
        });
        if gpu.arch != "gfx1201" {
            eprintln!("oracle requires exact gfx1201, got {}", gpu.arch);
            std::process::exit(2);
        }
        eprintln!("tmp_gfx1201_chunk_exactness arch={} chunk={c} arms={:?}", gpu.arch, args.arms);
        // Oracle-executed schedule line (mirrors H's production log contract).
        println!("prefill_chunk: requested={c} admitted={c} commit_stride=512");
        let need_model = args.arms.iter().any(|x| *x == 5);
        let model_path = match (&args.model, need_model) {
            (Some(p), _) => Some(p.clone()),
            (None, true) => {
                eprintln!("arm 5 needs --model (or drop 5 from --arms)");
                std::process::exit(2);
            }
            (None, false) => None,
        };
        let m = model_path.as_ref().map(|p| load_model(&mut gpu, p));
        let mut cmp = Cmp::new();
        if args.arms.contains(&1) {
            run_arm1(&mut gpu, &mut cmp, c);
        }
        if args.arms.contains(&2) {
            run_arm2(&mut gpu, &mut cmp, c);
        }
        if args.arms.contains(&3) {
            run_arm3(&mut gpu, &mut cmp, c);
        }
        if args.arms.contains(&4) {
            run_arm4(&mut gpu, &mut cmp, c);
        }
        if args.arms.contains(&5) {
            run_arm5(&mut gpu, &mut cmp, c, m.as_ref().expect("arm5 needs --model"));
        }
        if args.arms.contains(&6) {
            run_arm6(&mut gpu, &mut cmp, c, m.as_ref());
        }
        let pass = cmp.failures == 0;
        let env_show = |k: &str| std::env::var(k).unwrap_or("(unset)".into());
        let summary = format!(
            "tmp_gfx1201_chunk_exactness chunk={c} arch={} arms={:?} failures={} bytes_compared={} => {}\nmodel={:?}\nDN_STATE_EF={} DN_REQUANT_PER_TOKEN={} ROCR_VISIBLE_DEVICES={} HOME={}\n",
            gpu.arch, args.arms, cmp.failures, cmp.bytes_compared,
            if pass { "PASS" } else { "FAIL" },
            args.model,
            env_show("HIPFIRE_DN_STATE_EF"),
            env_show("HIPFIRE_DN_REQUANT_PER_TOKEN"), env_show("ROCR_VISIBLE_DEVICES"),
            env_show("HOME"),
        );
        eprint!("{summary}");
        std::fs::write(args.out_dir.join(format!("oracle_results_{c}.txt")), &summary)
            .unwrap_or_else(|e| panic!("write summary: {e:?}"));
        if !pass {
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "deltanet")]
fn main() {
    ora::real_main();
}
