//! Focused gfx942 correctness oracle for the MagnumQuant G256 activation rotation.

use rdna_compute::{gen_fwht_signs, Gpu};

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| (lcg(&mut state) as f32 / u32::MAX as f32 - 0.5) * 0.5)
        .collect()
}

fn rotate_reference(x: &[f32], k: usize) -> Vec<f32> {
    assert_eq!(x.len() % k, 0);
    assert_eq!(k % 256, 0);
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let mut out = vec![0.0f32; x.len()];

    for (src_row, dst_row) in x.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
        for (src, dst) in src_row
            .chunks_exact(256)
            .zip(dst_row.chunks_exact_mut(256))
        {
            let mut values = [0.0f32; 256];
            for i in 0..256 {
                values[i] = src[i] * signs1[i];
            }
            let mut stride = 1;
            while stride < 256 {
                for base in (0..256).step_by(stride * 2) {
                    for lane in 0..stride {
                        let a = values[base + lane];
                        let b = values[base + lane + stride];
                        values[base + lane] = a + b;
                        values[base + lane + stride] = a - b;
                    }
                }
                stride *= 2;
            }
            for i in 0..256 {
                dst[i] = values[i] * 0.0625 * signs2[i];
            }
        }
    }
    out
}

fn compare(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (i, (&actual, &reference)) in got.iter().zip(expected).enumerate() {
        let abs = (actual - reference).abs();
        max_abs = max_abs.max(abs);
        if actual.to_bits() != reference.to_bits() {
            mismatches += 1;
            first.get_or_insert((i, actual, reference, abs));
        }
    }
    println!(
        "{label}: mismatches={mismatches}/{} max_abs={max_abs:.8e} first={first:?}",
        got.len()
    );
}

fn main() {
    const K: usize = 4096;
    const BATCH: usize = 7;

    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx942" {
        println!("SKIP: detected {}, requires gfx942", gpu.arch);
        return;
    }

    let single_host = make_x(K, 0x9420_0101);
    let single = gpu.upload_f32(&single_host, &[K]).expect("upload single");
    let single_out = gpu
        .upload_f32(&vec![0.0; K], &[K])
        .expect("allocate single output");
    gpu.rotate_x_mq(&single, &single_out, K)
        .expect("single rotate launch");
    gpu.hip.device_synchronize().expect("single rotate sync");
    compare(
        "single",
        &gpu.download_f32(&single_out).expect("download single"),
        &rotate_reference(&single_host, K),
    );

    let batch_host = make_x(BATCH * K, 0x9420_0202);
    let batch = gpu
        .upload_f32(&batch_host, &[BATCH, K])
        .expect("upload batch");
    let batch_out = gpu
        .upload_f32(&vec![0.0; BATCH * K], &[BATCH, K])
        .expect("allocate batch output");
    gpu.rotate_x_mq_batched(&batch, &batch_out, K, BATCH)
        .expect("batched rotate launch");
    gpu.hip.device_synchronize().expect("batched rotate sync");
    compare(
        "batched",
        &gpu.download_f32(&batch_out).expect("download batch"),
        &rotate_reference(&batch_host, K),
    );

    const FUSED_K: usize = 2048;
    let gate_host = make_x(FUSED_K, 0x9420_0303);
    let up_host = make_x(FUSED_K, 0x9420_0404);
    let gate = gpu.upload_f32(&gate_host, &[FUSED_K]).expect("upload gate");
    let up = gpu.upload_f32(&up_host, &[FUSED_K]).expect("upload up");
    let silu_plain = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate silu plain");
    let silu_reference = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate silu reference");
    let silu_fused = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate silu fused");
    gpu.deepseek4_silu_mul_clamp_f32(&gate, &up, &silu_plain, 7.0)
        .expect("standalone SwiGLU launch");
    gpu.rotate_x_mq(&silu_plain, &silu_reference, FUSED_K)
        .expect("standalone SwiGLU rotate launch");
    gpu.deepseek4_fused_silu_mul_clamp_mq_rotate(&gate, &up, &silu_fused, FUSED_K, 7.0)
        .expect("fused SwiGLU rotate launch");
    gpu.hip.device_synchronize().expect("SwiGLU rotate sync");
    compare(
        "fused-swiglu-rotate",
        &gpu.download_f32(&silu_fused).expect("download fused SwiGLU"),
        &gpu
            .download_f32(&silu_reference)
            .expect("download reference SwiGLU"),
    );

    let rms_x_host = make_x(FUSED_K, 0x9420_0505);
    let rms_weight_host: Vec<_> = make_x(FUSED_K, 0x9420_0606)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect();
    let rms_x = gpu.upload_f32(&rms_x_host, &[FUSED_K]).expect("upload rms x");
    let rms_weight = gpu
        .upload_f32(&rms_weight_host, &[FUSED_K])
        .expect("upload rms weight");
    let rms_plain_reference = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate rms plain reference");
    let rms_rot_reference = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate rms rotate reference");
    let rms_plain_fused = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate fused rms plain");
    let rms_rot_fused = gpu
        .upload_f32(&vec![0.0; FUSED_K], &[FUSED_K])
        .expect("allocate fused rms rotate");
    gpu.rmsnorm_f32(&rms_x, &rms_weight, &rms_plain_reference, 1.0e-6)
        .expect("standalone RMSNorm launch");
    gpu.rotate_x_mq(&rms_plain_reference, &rms_rot_reference, FUSED_K)
        .expect("standalone RMSNorm rotate launch");
    gpu.fused_rmsnorm_rotate_mq_plain(
        &rms_x,
        &rms_weight,
        &rms_rot_fused,
        &rms_plain_fused,
        FUSED_K,
        1.0e-6,
    )
    .expect("fused RMSNorm rotate launch");
    gpu.hip.device_synchronize().expect("RMSNorm rotate sync");
    compare(
        "fused-rmsnorm-plain",
        &gpu.download_f32(&rms_plain_fused).expect("download fused rms plain"),
        &gpu
            .download_f32(&rms_plain_reference)
            .expect("download reference rms plain"),
    );
    compare(
        "fused-rmsnorm-rotate",
        &gpu.download_f32(&rms_rot_fused).expect("download fused rms rotate"),
        &gpu
            .download_f32(&rms_rot_reference)
            .expect("download reference rms rotate"),
    );
}
