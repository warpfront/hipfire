// THROWAWAY B2 QK-leg compare. DELETE BEFORE COMMIT. Not for production.
//
// Layer-35 real-data taps: native fp8 KV (production writer) + f32 Q.
// Runs Q0-device (f16 body, flag off) and B2-device (fp8 QK + f16 PV,
// flag on) via the S3 launchers, compares both against the production
// f16 kernel O (o_kern_final, arm1) and against each other (QK-leg delta;
// PV is bit-exact Q0 by construction: f16(decode*sv), f32 mul commutes).
fn tapdir() -> String {
    std::env::var("B2_TAPDIR").unwrap_or_else(|_| "/home/kaden/ClaudeCode/warpfront/wt-fa2a/scratch-2026-09-17/StageB/Screen/dump35/layer35".to_string())
}
use rdna_compute::{DType, Gpu};
const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const NQ: usize = 384;
const NTOK: usize = 4224;
const ROW: usize = NKV * (HD + 2);

fn load_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert!(b.len() % 4 == 0, "{path}");
    let mut v = vec![0f32; b.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            b.as_ptr(),
            v.as_mut_ptr() as *mut u8,
            b.len(),
        )
    };
    v
}

fn stats(tag: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut pairs: Vec<(f32, f32)> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x, (x - y).abs()))
        .collect();
    for &(_, d) in &pairs {
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    pairs.sort_by(|p, q| q.0.abs().partial_cmp(&p.0.abs()).unwrap());
    let tail_n = (n / 100).max(1);
    let tail_sum: f64 = pairs[..tail_n].iter().map(|&(_, d)| d as f64).sum();
    let mut max_rel = 0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.abs() > 1e-3 {
            max_rel = max_rel.max((x - y).abs() / x.abs());
        }
    }
    let bad = a
        .iter()
        .zip(b.iter())
        .filter(|(&x, &y)| !x.is_finite() || !y.is_finite())
        .count();
    println!(
        "{tag}: n={n} max_abs={max_abs:.6e} mean_abs={:.6e} tail1pct={:.6e} max_rel={max_rel:.6e} nonfinite={bad}",
        sum_abs / n as f64,
        tail_sum / tail_n as f64
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("=== SKIP === exact gfx1201 only (arch={})", gpu.arch);
        return;
    }
    let tap = tapdir();
    let q_host = load_f32(&format!("{tap}/q_final.f32"));
    let k_host = load_f32(&format!("{tap}/k_raw.f32"));
    let v_host = load_f32(&format!("{tap}/v_raw.f32"));
    let o_ref = load_f32(&format!("{tap}/o_kern_final.f32"));
    assert_eq!(q_host.len(), NQ * NH * HD);
    assert_eq!(k_host.len(), NTOK * NKV * HD);
    assert_eq!(v_host.len(), NTOK * NKV * HD);
    assert_eq!(o_ref.len(), NQ * NH * HD);
    let pb = std::fs::read(format!("{tap}/positions.i32")).unwrap();
    assert_eq!(pb.len(), NQ * 4);
    let mut pos_host = vec![0i32; NQ];
    unsafe {
        std::ptr::copy_nonoverlapping(
            pb.as_ptr(),
            pos_host.as_mut_ptr() as *mut u8,
            pb.len(),
        )
    };

    let q = gpu.alloc_tensor(&[NQ * NH * HD], DType::F32).unwrap();
    let k = gpu.alloc_tensor(&[NTOK * NKV * HD], DType::F32).unwrap();
    let v = gpu.alloc_tensor(&[NTOK * NKV * HD], DType::F32).unwrap();
    let positions = gpu.alloc_tensor(&[NQ], DType::F32).unwrap();
    let bytes_f32 = |x: &[f32]| unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    };
    gpu.hip.memcpy_htod(&q.buf, bytes_f32(&q_host)).unwrap();
    gpu.hip.memcpy_htod(&k.buf, bytes_f32(&k_host)).unwrap();
    gpu.hip.memcpy_htod(&v.buf, bytes_f32(&v_host)).unwrap();
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

    // Native KV for all 4224 rows (positions 0..4223).
    let pos_all: Vec<i32> = (0..NTOK as i32).collect();
    let positions_all = gpu.alloc_tensor(&[NTOK], DType::F32).unwrap();
    gpu.hip
        .memcpy_htod(
            &positions_all.buf,
            unsafe {
                std::slice::from_raw_parts(
                    pos_all.as_ptr() as *const u8,
                    pos_all.len() * 4,
                )
            },
        )
        .unwrap();
    let k_cache = gpu.alloc_tensor(&[NTOK * ROW], DType::Raw).unwrap();
    let v_cache = gpu.alloc_tensor(&[NTOK * ROW], DType::Raw).unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&k_cache, &k, &positions_all, NKV, HD, NTOK)
        .unwrap();
    gpu.kv_cache_write_fp8_e4m3_batched(&v_cache, &v, &positions_all, NKV, HD, NTOK)
        .unwrap();

    let out_q0 = gpu.alloc_tensor(&[NQ * NH * HD], DType::F32).unwrap();
    let out_b2 = gpu.alloc_tensor(&[NQ * NH * HD], DType::F32).unwrap();
    // Q0 arm (flag off at process start): f16 body on native KV.
    gpu.attention_fp8_e4m3_fa2_gqa_f16_gfx1201(
        &q, &k_cache, &v_cache, &out_q0, &positions, NH, NKV, HD, NTOK, NQ,
    )
    .unwrap();
    // NOTE: flag flip requires a fresh process (flags bind at init);
    // run the B2 arm via B2_THROWAWAY_FP8=1 sub-process convention below.
    // Here we only dump Q0; B2 runs in a second process (see main_fp8).
    let mut hq = vec![0f32; NQ * NH * HD];
    gpu.hip
        .memcpy_dtoh(
            unsafe {
                std::slice::from_raw_parts_mut(
                    hq.as_mut_ptr() as *mut u8,
                    hq.len() * 4,
                )
            },
            &out_q0.buf,
        )
        .unwrap();
    stats("Q0-vs-arm1", &o_ref, &hq);
    let out_path = std::env::var("B2_OUT").unwrap_or_default();
    if !out_path.is_empty() {
        std::fs::write(
            &out_path,
            unsafe {
                std::slice::from_raw_parts(
                    hq.as_ptr() as *const u8,
                    hq.len() * 4,
                )
            },
        )
        .unwrap();
        eprintln!("wrote {out_path}");
    }

    if std::env::var("B2_FP8").as_deref() == Ok("1") {
        gpu.attention_fp8_e4m3_fa2_gqa_fp8_gfx1201(
            &q, &k_cache, &v_cache, &out_b2, &positions, NH, NKV, HD, NTOK, NQ,
        )
        .unwrap();
        let mut hb = vec![0f32; NQ * NH * HD];
        gpu.hip
            .memcpy_dtoh(
                unsafe {
                    std::slice::from_raw_parts_mut(
                        hb.as_mut_ptr() as *mut u8,
                        hb.len() * 4,
                    )
                },
                &out_b2.buf,
            )
            .unwrap();
        stats("B2-vs-arm1", &o_ref, &hb);
        stats("B2-vs-Q0", &hq, &hb);
        if !out_path.is_empty() {
            let p2 = out_path + ".b2";
            std::fs::write(
                &p2,
                unsafe {
                    std::slice::from_raw_parts(
                        hb.as_ptr() as *const u8,
                        hb.len() * 4,
                    )
                },
            )
            .unwrap();
            eprintln!("wrote {p2}");
        }
    }
}
