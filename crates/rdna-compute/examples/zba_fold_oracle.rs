// SPDX-License-Identifier: Apache-2.0
//! gfx1201 GDN Z/beta/alpha fold SET + split byte oracle (GPU, lab only).
//!
//! Compares, for each batch size in {1, 8, 16, 64, 100, 128}, the three
//! separate `_b1s`/MMQ SET launches (Z over the first `z_m` rows of the fold
//! buffer, beta and alpha over buffers holding only their 48 rows) with
//! `Gpu::gemm_zba_fold_g12_iu4_prepared` (one SET over the loader's Z fold,
//! then `split_mq4v2_z_betaalpha_g12`). Raw output bytes must be identical.
//! The fold buffer is assembled exactly like the loader's
//! `append_betaalpha_to_z`: Z rows, beta at row `z_m`, alpha at row `z_m + 64`,
//! zero rows elsewhere. Activations are SLAB activations from the product's AWQ
//! fused-RMSNorm producer twin (`fused_rmsnorm_rotate_mq_i4_gfx12_batched` with a
//! unit weight and a synthetic AWQ scale), the layout the `_b1s` bundle and the
//! merged QKV + Z-fold launch require; the oracle fails when the merged launch is
//! not admitted. Needs `Gpu::a4_slab_active` (default env on gfx1201: no
//! `HIPFIRE_A4_SLAB=0`, `HIPFIRE_G12_IU4_B1_CONTROL`, `HIPFIRE_G12_A4C2=0`,
//! `HIPFIRE_G12_IU4_ISA=0`, `HIPFIRE_IU4_SYMFOLD=0` or `HIPFIRE_G12_NORM=0`).
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab --example zba_fold_oracle
//! ```
//! Exit 0 = every n identical, 1 = mismatch, 2 = unsupported configuration.

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP_BYTES: usize = 136;
const SENTINEL: i32 = 0xA5;
const Z_M: usize = 6144;
const QKV_M: usize = 10240;
const K: usize = 5120;
const NS: [usize; 6] = [1, 8, 16, 64, 100, 128];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

/// Random valid MQ4G256V2 rows: `m` rows of `k / 256` groups of 136 B.
fn synth_mq4g256v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = 0.004 + (rng.next() & 0x3f) as f32 * 2.0e-4;
            let zero = -7.5 * scale + ((rng.next() & 0xf) as f32 - 8.0) * 1.0e-4;
            out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            out.extend_from_slice(&f16::from_f32(zero).to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    out
}

fn synth_x(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..k {
            out.push(((rng.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0) * scale);
        }
    }
    out
}

fn sentinel(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .memset(&t.buf, SENTINEL, t.byte_size())
        .expect("sentinel memset");
}

fn first_mismatch(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("zba_fold_oracle: arch {} is not gfx1201", gpu.arch);
        std::process::exit(2);
    }
    // The loader claims the symmetric contract after verifying headers; the
    // synthetic weights are only compared against themselves.
    gpu.mq4v2_symmetric = true;
    if !gpu.iu4_gdn_fold_active([QKV_M, Z_M, 48, 48], K) {
        eprintln!("zba_fold_oracle: Gpu::iu4_gdn_fold_active is false (check HIPFIRE_IU4_PREFILL / HIPFIRE_GDN_PREP_FUSED / G12 IU4 flags)");
        std::process::exit(2);
    }

    let row_bytes = K / 256 * GROUP_BYTES;
    let folded_m = Gpu::iu4_gdn_fold_rows(Z_M).0;
    let z_bytes = synth_mq4g256v2(Z_M, K, 0x2b4a_0001);
    let beta_bytes = synth_mq4g256v2(48, K, 0x2b4a_0002);
    let alpha_bytes = synth_mq4g256v2(48, K, 0x2b4a_0003);
    // Fold layout of the loader (`append_betaalpha_to_z`).
    let mut fold = vec![0u8; folded_m * row_bytes];
    fold[..Z_M * row_bytes].copy_from_slice(&z_bytes);
    fold[Z_M * row_bytes..(Z_M + 48) * row_bytes].copy_from_slice(&beta_bytes);
    fold[(Z_M + 64) * row_bytes..(Z_M + 112) * row_bytes].copy_from_slice(&alpha_bytes);
    let qkv_bytes = synth_mq4g256v2(QKV_M, K, 0x2b4a_0004);
    let w_qkv = gpu
        .upload_raw(&qkv_bytes, &[qkv_bytes.len()])
        .expect("upload qkv");
    let w_fold = gpu.upload_raw(&fold, &[fold.len()]).expect("upload fold");
    let w_beta = gpu
        .upload_raw(&beta_bytes, &[beta_bytes.len()])
        .expect("upload beta");
    let w_alpha = gpu
        .upload_raw(&alpha_bytes, &[alpha_bytes.len()])
        .expect("upload alpha");

    // The merged QKV + Z-fold launch and the `_b1s` bundle read only SLAB activations. Those come
    // from the product's AWQ fused-RMSNorm producer twin (`_v2_slab`), so the oracle feeds that
    // producer with a unit RMSNorm weight and a synthetic AWQ scale (K f32 near 1.0, no reciprocal
    // planes: `awq_has_rcp_planes` is false, hence the non-fdiv slab twin).
    if !gpu.a4_slab_active() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "<unset>".into());
        eprintln!(
            "zba_fold_oracle: Gpu::a4_slab_active is false (slab producer twins unavailable): \
             HIPFIRE_A4_SLAB={} HIPFIRE_G12_IU4_B1_CONTROL={} HIPFIRE_IU4_SYMFOLD={} \
             HIPFIRE_G12_IU4_ISA={} HIPFIRE_G12_A4C2={} g12_iu4_isa={} hipcc_extra_flags={:?} (needs -DIU4_A4_CANDIDATES=2)",
            env("HIPFIRE_A4_SLAB"),
            env("HIPFIRE_G12_IU4_B1_CONTROL"),
            env("HIPFIRE_IU4_SYMFOLD"),
            env("HIPFIRE_G12_IU4_ISA"),
            env("HIPFIRE_G12_A4C2"),
            gpu.flags.g12_iu4_isa,
            gpu.flags.hipcc_extra_flags,
        );
        std::process::exit(2);
    }
    let unit_w = gpu
        .upload_f32(&vec![1.0f32; K], &[K])
        .expect("upload rmsnorm unit weight");
    let awq_scale = {
        let mut rng = Rng(0x2b4a_a000);
        let v: Vec<f32> = (0..K)
            .map(|_| 0.9 + (rng.next() & 0xffff) as f32 / 65_536.0 * 0.2)
            .collect();
        gpu.upload_f32(&v, &[K]).expect("upload awq scale")
    };

    // Open the IU4 verify scope so the slab producers admit n < 64 (the DFlash verify batches).
    let _scope = gpu.iu4_verify_enter(true);
    let mut all_ok = true;
    for n in NS {
        let x = gpu
            .upload_f32(&synth_x(n, K, 0x2b4a_1000 + n as u64), &[n, K])
            .expect("upload x");
        // The shared-IU4 predicate wants batch >= 64; the verify scope opened above admits the producers for smaller n.
        if n >= 64 && !gpu.mq4g256v2_shared_iu4_supported(K, n) {
            eprintln!("zba_fold_oracle: IU4 prepared path unsupported for k={K} n={n}");
            std::process::exit(2);
        }
        let res = gpu.reserve_int4_mmq(K, n).expect("reserve int4 mmq");
        let prep = gpu
            .fused_rmsnorm_rotate_mq_i4_gfx12_batched(
                &x,
                &unit_w,
                Some(&awq_scale),
                None,
                res,
                K,
                1.0e-6,
                n,
            )
            .expect("slab producer");
        let slab = {
            let p = gpu
                .int4_mmq_prepared_ptr(&prep, K, n)
                .expect("prepared ptr");
            gpu.scratch.int4_mmq_slab_at(p)
        };
        if !slab {
            eprintln!("n={n}: FAIL producer did not mark slab activations (HIPFIRE_G12_NORM=0?)");
            std::process::exit(2);
        }

        let alloc =
            |gpu: &mut Gpu, len: usize| gpu.alloc_tensor(&[len], DType::F32).expect("alloc");
        let (z_ref, b_ref, a_ref) = (
            alloc(&mut gpu, n * Z_M),
            alloc(&mut gpu, n * 48),
            alloc(&mut gpu, n * 48),
        );
        let (z_new, b_new, a_new) = (
            alloc(&mut gpu, n * Z_M),
            alloc(&mut gpu, n * 48),
            alloc(&mut gpu, n * 48),
        );
        let y_fold = alloc(&mut gpu, n * folded_m);
        let (q_ref, q_new) = (alloc(&mut gpu, n * QKV_M), alloc(&mut gpu, n * QKV_M));
        let (zm_new, bm_new, am_new) = (
            alloc(&mut gpu, n * Z_M),
            alloc(&mut gpu, n * 48),
            alloc(&mut gpu, n * 48),
        );
        for t in [&q_ref, &q_new, &zm_new, &bm_new, &am_new] {
            sentinel(&gpu, t);
        }
        for t in [&z_ref, &b_ref, &a_ref, &z_new, &b_new, &a_new, &y_fold] {
            sentinel(&gpu, t);
        }

        // (a) three separate SETs.
        let xq = gpu
            .int4_mmq_prepared_ptr(&prep, K, n)
            .expect("prepared ptr");
        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w_fold, xq, &z_ref, Z_M, K, n)
            .expect("z SET");
        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w_beta, xq, &b_ref, 48, K, n)
            .expect("beta SET");
        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w_alpha, xq, &a_ref, 48, K, n)
            .expect("alpha SET");

        gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&w_qkv, xq, &q_ref, QKV_M, K, n)
            .expect("qkv SET");
        // (c) merged QKV + Z fold launch.
        let merged = gpu
            .gemm_qkvza_set_g12_iu4_prepared(
                &w_qkv,
                &w_fold,
                &prep,
                &q_new,
                &zm_new,
                &bm_new,
                &am_new,
                [QKV_M, Z_M, 48, 48],
                K,
                n,
            )
            .expect("merged qkv + z fold SET");
        println!(
            "n={n}: route {}",
            if merged {
                "merged (qkvzaset_b1s)"
            } else {
                "NOT merged"
            }
        );

        // (b) fold SET + split.
        let launched = gpu
            .gemm_zba_fold_g12_iu4_prepared(
                &w_fold, &prep, &y_fold, &z_new, &b_new, &a_new, QKV_M, Z_M, K, n,
            )
            .expect("fold SET + split");
        gpu.hip.device_synchronize().expect("device synchronize");
        if !launched {
            eprintln!("n={n}: FAIL fold path not admitted (Ok(false))");
            all_ok = false;
            continue;
        }

        if !merged {
            eprintln!("n={n}: FAIL merged launch not admitted (Ok(false))");
            all_ok = false;
            continue;
        }
        let mut ok = true;
        for (name, a, b) in [
            ("z", &z_ref, &z_new),
            ("beta", &b_ref, &b_new),
            ("alpha", &a_ref, &a_new),
            ("merged qkv", &q_ref, &q_new),
            ("merged z", &z_ref, &zm_new),
            ("merged beta", &b_ref, &bm_new),
            ("merged alpha", &a_ref, &am_new),
        ] {
            let ra = gpu.download_raw_bytes(a).expect("download ref");
            let rb = gpu.download_raw_bytes(b).expect("download new");
            if let Some(i) = first_mismatch(&ra, &rb) {
                eprintln!("n={n}: {name} first mismatch at byte {i} (of {})", ra.len());
                ok = false;
            }
        }
        println!("n={n}: {}", if ok { "PASS" } else { "FAIL" });
        all_ok &= ok;
    }
    println!("{}", if all_ok { "ALL PASS" } else { "FAILED" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
