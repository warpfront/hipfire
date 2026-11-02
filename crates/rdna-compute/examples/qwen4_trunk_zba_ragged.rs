//! Bitwise hardware oracle for the production in-place ragged zab route.
use rdna_compute::gemm::qwen4_trunk_iu4_ragged_rows;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

const N: usize = 8192;
const K: usize = 2560;
const Z: usize = 6144;
const POISON: u32 = 0x7fc0_dead;

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let a = self.unit().max(1e-7);
        let b = self.unit();
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}
fn poison(gpu: &mut Gpu, width: usize) -> GpuTensor {
    let raw = POISON.to_le_bytes().repeat(N * width);
    let mut t = gpu.upload_raw(&raw, &[N, width]).expect("upload poison");
    t.dtype = DType::F32;
    t
}
fn words(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_raw_bytes(t)
        .expect("download")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn main() {
    let start = std::time::Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let dir = arg("--payloads").expect("--payloads DIR is required (real QT44 only)");
    let layer: usize = arg("--layer").map_or(0, |v| v.parse().expect("layer"));
    assert_ne!(layer % 4, 3, "layer must be GDN");
    let rows_list: Vec<usize> = arg("--rows")
        .map_or(vec![513, 777, 1124, 1127, 4097, 8151, 8186, 8191], |v| {
            v.split(',').map(|r| r.parse().expect("rows")).collect()
        });
    for r in [511, 512, 8064, 8192] {
        assert_eq!(qwen4_trunk_iu4_ragged_rows(r), None);
    }
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1151");
    gpu.ensure_mq_signs().expect("mq signs");
    let rb = K / 256 * 136;
    let mut fold = vec![0u8; (Z + 256) * rb];
    for (tag, m, offset) in [
        ("in_proj_z", Z, 0),
        ("in_proj_b", 48, Z),
        ("in_proj_a", 48, Z + 48),
    ] {
        let name =
            format!("{dir}/model.language_model.layers.{layer}.linear_attn.{tag}.weight.bin");
        let raw = std::fs::read(&name).expect("real QT44 payload");
        assert_eq!(raw.len(), m * rb, "{name}: QT44 size");
        let w = gpu.upload_raw(&raw, &[raw.len()]).expect("weight upload");
        assert!(
            gpu.qwen4_trunk_sym_check(&w, m, K).expect("sym check"),
            "{name}: asymmetric"
        );
        gpu.free_tensor(w).expect("free weight");
        fold[offset * rb..(offset + m) * rb].copy_from_slice(&raw);
    }
    let fold = gpu.upload_raw(&fold, &[fold.len()]).expect("fold upload");
    gpu.mq4v2_symmetric = true;
    gpu.qwen4_trunk_iu4 = true;
    gpu.qwen4_trunk_iu4_prefill_scope = true;
    // Resolve the production flag exactly once, without changing env or spawning.
    let disabled =
        hipfire_config::developer_var("HIPFIRE_QWEN4_TRUNK_IU4_RAGGED").as_deref() == Ok("0");
    if disabled {
        assert_eq!(gpu.qwen4_trunk_iu4_ragged(8186), None);
        let report =
            json!({"pass": true, "kill_switch": true, "admission_checks": true, "cases": []});
        let text = report.to_string();
        if let Some(out) = arg("--json") {
            std::fs::write(out, &text).expect("write json");
        }
        println!("{text}");
        gpu.free_tensor(fold).expect("free fold");
        return;
    }
    let capture = arg("--act").map(|p| {
        let raw = std::fs::read(p).expect("activation file");
        assert_eq!(raw.len() % (K * 4), 0, "activation row size");
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    });
    let mut reports = Vec::new();
    let mut pass = true;
    for rows in rows_list {
        assert!(rows < N);
        let n_pad = qwen4_trunk_iu4_ragged_rows(rows).expect("ragged rows >=512");
        assert_eq!(n_pad, rows.next_multiple_of(256));
        assert_eq!(gpu.qwen4_trunk_iu4_ragged(rows), Some(n_pad));
        let mut rng = Rng(0x1234_5678);
        let mut x: Vec<f32> = (0..N * K).map(|_| rng.normal()).collect();
        if let Some(c) = &capture {
            assert!(c.len() >= rows * K, "capture too short");
            x[..rows * K].copy_from_slice(&c[..rows * K]);
        }
        // Unrelated random reference tail, rather than candidate zero padding.
        let mut tail_rng = Rng(0xabc0_1234 + rows as u64);
        for v in &mut x[rows * K..] {
            *v = tail_rng.normal();
        }
        let reference_input = gpu.upload_f32(&x, &[N, K]).expect("reference input");
        let reference: Vec<_> = [Z, 48, 48]
            .into_iter()
            .map(|m| poison(&mut gpu, m))
            .collect();
        let res = gpu.reserve_int4_mmq(K, N).expect("reserve reference");
        let prep = gpu
            .rotate_x_mq_i4_batched(&reference_input, None, None, res, K, N)
            .expect("reference producer");
        assert!(gpu
            .gemm_qwen4_trunk_zba_iu4(
                &fold,
                &prep,
                &reference[0],
                &reference[1],
                &reference[2],
                Z,
                K,
                N
            )
            .expect("reference V2B"));
        gpu.hip.device_synchronize().expect("reference sync");
        gpu.free_tensor(reference_input)
            .expect("free reference input");
        for v in &mut x[rows * K..] {
            *v = f32::from_bits(POISON);
        }
        let candidate_input = gpu.upload_f32(&x, &[N, K]).expect("candidate input");
        let candidate: Vec<_> = [Z, 48, 48]
            .into_iter()
            .map(|m| poison(&mut gpu, m))
            .collect();
        let took = gpu
            .qwen4_trunk_zba_ragged(
                &candidate_input,
                rows,
                n_pad,
                K,
                &fold,
                Z,
                &candidate[0],
                &candidate[1],
                &candidate[2],
            )
            .expect("production ragged route");
        gpu.hip.device_synchronize().expect("candidate sync");
        let mut outputs = Vec::new();
        for (i, (name, m)) in [("z", Z), ("beta", 48), ("alpha", 48)]
            .into_iter()
            .enumerate()
        {
            let r = words(&gpu, &reference[i]);
            let c = words(&gpu, &candidate[i]);
            let mut first = None;
            let mut mismatch = 0usize;
            for j in 0..rows * m {
                if r[j] != c[j] {
                    mismatch += 1;
                    if first.is_none() {
                        first = Some(
                            json!({"row": j / m, "col": j % m, "reference": r[j], "candidate": c[j]}),
                        );
                    }
                }
            }
            let pad_nonzero = c[rows * m..n_pad * m].iter().filter(|&&w| w != 0).count();
            let poison_overwrites = c[n_pad * m..].iter().filter(|&&w| w != POISON).count();
            pass &= took && mismatch == 0 && pad_nonzero == 0 && poison_overwrites == 0;
            outputs.push(
                json!({"name": name, "mismatch": mismatch, "first_mismatch": first,
                "pad_nonzero_words": pad_nonzero, "poison_overwrites": poison_overwrites}),
            );
        }
        reports.push(json!({"rows": rows, "n_pad": n_pad, "took": took, "outputs": outputs}));
        for t in reference
            .into_iter()
            .chain(candidate)
            .chain([candidate_input])
        {
            gpu.free_tensor(t).expect("free case");
        }
    }
    gpu.free_tensor(fold).expect("free fold");
    let seconds = start.elapsed().as_secs_f64();
    pass &= seconds < 90.0;
    let report = json!({"pass": pass, "layer": layer, "kill_switch": false,
        "admission_checks": true, "seconds": seconds, "cases": reports});
    let text = report.to_string();
    if let Some(out) = arg("--json") {
        std::fs::write(out, &text).expect("write json");
    }
    println!("{text}");
    std::process::exit(i32::from(!pass));
}
