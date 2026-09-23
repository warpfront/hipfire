// THROWAWAY B1b device oracle. DELETE BEFORE COMMIT. Not for production.
//
// Route-N byte-copy fill: dumps K/V fp8 planes + accessor-read scales for
// one KT64 tile and compares bytewise against the host fragment model
// (codes verbatim, OOR keys zero, scales = f32(header f16) or 0).
// Fixtures: seq 64 (full tile) and seq 40 (OOR tail), batch 8, all kv_h.
use rdna_compute::{DType, Gpu};

const SRC: &str = concat!(
    "#define HIPFIRE_FA2_FP8 1\n",
    "#define HIPFIRE_FA2_KMODE 8\n",
    include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip")
);
const SYMBOL: &str = "attention_fp8_e4m3_fa2_gqa_gfx1201";

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BATCH: usize = 8;
const CAP: usize = 128;
const ROW: usize = NKV * (HD + 2); // 1032
const DUMP_WORDS: usize = 8320; // 4096 K + 4096 V + 64 Ksc + 64 Vsc (u32)

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_u64() as f64 / u64::MAX as f64).max(1e-12);
        let u2 = self.next_u64() as f64 / u64::MAX as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let b = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
            let mut e2 = 127u32 - 14 - 10;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            m2 &= 0x3ff;
            (s << 31) | (e2 << 23) | (m2 << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(b)
}

fn run_case(
    gpu: &mut Gpu,
    k_host: &[f32],
    v_host: &[f32],
    seq: usize,
    tag: &str,
) -> (usize, usize) {
    // Cache rows 0..63 carry k_host/v_host rows; bounds rows force seq_len.
    let nrows = 64usize;
    let mut pos_host: Vec<i32> = (0..nrows as i32).collect();
    let k = gpu.alloc_tensor(&[nrows * NKV * HD], DType::F32).unwrap();
    let v = gpu.alloc_tensor(&[nrows * NKV * HD], DType::F32).unwrap();
    let positions = gpu.alloc_tensor(&[nrows], DType::F32).unwrap();
    let bytes_f32 = |x: &[f32]| unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    };
    gpu.hip.memcpy_htod(&k.buf, bytes_f32(k_host)).unwrap();
    gpu.hip.memcpy_htod(&v.buf, bytes_f32(v_host)).unwrap();
    gpu.hip
        .memcpy_htod(
            &positions.buf,
            unsafe {
                std::slice::from_raw_parts(
                    pos_host.as_ptr() as *const u8,
                    pos_host.len() * 4,
                )
            },
        )
        .unwrap();
    let k_cache = gpu.alloc_tensor(&[CAP * ROW], DType::Raw).unwrap();
    let v_cache = gpu.alloc_tensor(&[CAP * ROW], DType::Raw).unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&k_cache, &k, &positions, NKV, HD, nrows)
        .unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&v_cache, &v, &positions, NKV, HD, nrows)
        .unwrap();
    // Bounds: first 8 rows decide seq_len = max+1.
    for p in pos_host.iter_mut().take(BATCH) {
        *p = (seq - 1) as i32;
    }
    gpu.hip
        .memcpy_htod(
            &positions.buf,
            unsafe {
                std::slice::from_raw_parts(
                    pos_host.as_ptr() as *const u8,
                    pos_host.len() * 4,
                )
            },
        )
        .unwrap();

    let q8 = gpu.alloc_tensor(&[65536], DType::Raw).unwrap();
    // Dump covers head slots ((bx*4+kh)*512+t)*8320 with t=0: kh=3 peaks.
    let out_words = (3 * 512 + 1) * DUMP_WORDS;
    let out = gpu.alloc_tensor(&[out_words * 4], DType::Raw).unwrap();

    gpu.ensure_kernel_public("b1b_oracle", SRC, SYMBOL).unwrap();
    let scale = 1.0f32 / (HD as f32).sqrt();
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(q8.buf.as_ptr() as *const std::ffi::c_void);
    blob.push_ptr(k_cache.buf.as_ptr() as *const std::ffi::c_void);
    blob.push_ptr(v_cache.buf.as_ptr() as *const std::ffi::c_void);
    blob.push_ptr(out.buf.as_ptr() as *const std::ffi::c_void);
    blob.push_ptr(positions.buf.as_ptr() as *const std::ffi::c_void);
    blob.push_i32(NH as i32);
    blob.push_i32(NKV as i32);
    blob.push_i32(HD as i32);
    blob.push_i32(BATCH as i32);
    blob.push_f32(scale);
    gpu.launch_kernel_blob(SYMBOL, [1, 4, 1], [128, 1, 1], 32768, blob.as_mut_slice())
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let mut dump = vec![0u32; out_words];
    gpu.hip
        .memcpy_dtoh(
            unsafe {
                std::slice::from_raw_parts_mut(dump.as_mut_ptr() as *mut u8, dump.len() * 4)
            },
            &out.buf,
        )
        .unwrap();
    let mut kc = vec![0u8; CAP * ROW];
    let mut vc = vec![0u8; CAP * ROW];
    gpu.hip.memcpy_dtoh(&mut kc, &k_cache.buf).unwrap();
    gpu.hip.memcpy_dtoh(&mut vc, &v_cache.buf).unwrap();

    // Host model + compare.
    let mut bad = 0usize;
    let mut total = 0usize;
    let dump_b = |w: &[u32]| unsafe {
        std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4)
    };
    for kh in 0..NKV {
        let base = (kh * 512) * DUMP_WORDS; // bx=0, split 0, tile 0
        let plane = &dump[base..base + 8192];
        let pb = dump_b(plane);
        let sc_k = &dump[base + 8192..base + 8256];
        let sc_v = &dump[base + 8256..base + 8320];
        // K plane: frag(sub,dc,lane) byte j = code[key][dim] or 0.
        for sub in 0..4 {
            for dc in 0..16 {
                for lane in 0..32 {
                    let ml = lane & 15;
                    let kg = lane >> 4;
                    let key = sub * 16 + ml;
                    for j in 0..8 {
                        let dim = dc * 16 + kg * 8 + j;
                        let want = if key < seq {
                            kc[key * ROW + kh * HD + dim]
                        } else {
                            0
                        };
                        let got = pb[((dc * 4 + sub) * 32 + lane) * 8 + j];
                        total += 1;
                        if got != want {
                            bad += 1;
                            if bad < 10 {
                                eprintln!("{tag} K kh={kh} key={key} dim={dim}: got={got:02x} want={want:02x}");
                            }
                        }
                    }
                }
            }
        }
        // V plane: frag byte j = code[key][dim] or 0.
        for sub in 0..4 {
            for dc in 0..16 {
                for lane in 0..32 {
                    let ml = lane & 15;
                    let kg = lane >> 4;
                    let dim = dc * 16 + ml;
                    for j in 0..8 {
                        let key = sub * 16 + kg * 8 + j;
                        let want = if key < seq {
                            vc[key * ROW + kh * HD + dim]
                        } else {
                            0
                        };
                        let got = pb[16384 + ((sub * 16 + dc) * 32 + lane) * 8 + j];
                        total += 1;
                        if got != want {
                            bad += 1;
                            if bad < 10 {
                                eprintln!("{tag} V kh={kh} key={key} dim={dim}: got={got:02x} want={want:02x}");
                            }
                        }
                    }
                }
            }
        }
        // Scales via accessor: f32(header f16) or 0.
        for key in 0..64 {
            for (side, row, sc) in [("K", &kc, sc_k), ("V", &vc, sc_v)] {
                let want = if key < seq {
                    let lo = row[key * ROW + 1024 + kh * 2];
                    let hi = row[key * ROW + 1024 + kh * 2 + 1];
                    f16_bits_to_f32((lo as u16) | ((hi as u16) << 8)).to_bits()
                } else {
                    0u32
                };
                let got = sc[key];
                total += 1;
                if got != want {
                    bad += 1;
                    if bad < 10 {
                        eprintln!("{tag} S{side} kh={kh} key={key}: got={got:08x} want={want:08x}");
                    }
                }
            }
        }
    }
    println!("{tag}: compared={total} mismatches={bad}");
    (total, bad)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("=== SKIP === exact gfx1201 only (arch={})", gpu.arch);
        return;
    }
    let nrows = 64usize;
    let mut rng = Lcg(0xB1B0B1B);
    let k_host: Vec<f32> = (0..nrows * NKV * HD)
        .map(|_| rng.next_normal() * 1.5)
        .collect();
    let v_host: Vec<f32> = (0..nrows * NKV * HD)
        .map(|_| rng.next_normal() * 0.7)
        .collect();
    let mut fail = 0;
    for (seq, tag) in [(64usize, "full64"), (40usize, "oor40")] {
        let (total, bad) = run_case(&mut gpu, &k_host, &v_host, seq, tag);
        assert!(total == 4 * (16384 + 16384 + 128), "compare count {total}");
        if bad > 0 {
            fail += 1;
        }
    }
    if fail > 0 {
        eprintln!("B1b ORACLE FAIL");
        std::process::exit(1);
    }
    eprintln!("B1b ORACLE PASS");
}
