// SPDX-License-Identifier: Apache-2.0
// Slice E: disposable dual-module device oracle — route-N stage-b vs packet
// body on identical frozen code-Q/sq/K/V/positions. Separate entry symbols
// force separate module compiles (symbol-keyed function cache), so this is
// old-vs-new kernel coverage, not aliasing. Direct raw-O compare plus
// partial-record compare at splits 1 and 8 with each arm's own merge.
//
// Run (ordinal 2): HOME=... ROCR_VISIBLE_DEVICES=2 cargo run --release -p
// hipfire-runtime --example tmp_fa_packet_oracle -- --out <dir>

use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const ROWB: usize = 1032;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}
fn download_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor, n_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n_bytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .unwrap_or_else(|e| panic!("download_bytes: {e:?}"));
    bytes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut outdir = String::from("scratch-2026-09-17/fapkt/oracle");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                outdir = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    std::fs::create_dir_all(&outdir).unwrap();
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("packet oracle on {}", gpu.arch);
    // Retained shape rows (plan G2) plus ownership/tail edges: batch
    // 1,6,7,8,9,12,15,16,17,22,24,31,32,33 straddle WG boundaries
    // (rows = batch*6 vs 128-row blocks); ctx tails 15/16/17, 31/32/33,
    // 63/64/65 plus long non-multiples.
    let batches: Vec<usize> = vec![1, 6, 7, 8, 9, 12, 15, 16, 17, 22, 24, 31, 32, 33];
    let ctxs: Vec<usize> = vec![15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 200, 511, 1000];
    let mut fails = 0usize;
    let mut cases = 0usize;
    for &batch in &batches {
        for &ctx in &ctxs {
            if ctx < batch {
                continue;
            }
            let mut rng = Rng(0x9E3779B97F4A7C15u64 ^ ((batch as u64) << 32 | ctx as u64));
            // Native fp8 KV rows: pattern codes + f16 scale headers.
            let mut kv = vec![0u8; ctx * ROWB];
            let mut kv2 = vec![0u8; ctx * ROWB];
            for r in 0..ctx {
                for d in 0..1024 {
                    kv[r * ROWB + d] = rng.below(256) as u8;
                    kv2[r * ROWB + d] = rng.below(256) as u8;
                }
                for kv_h in 0..NKV {
                    let ku: u16 = (0x3800 + rng.below(0x800) as u16) as u16;
                    let vu: u16 = (0x3400 + rng.below(0x800) as u16) as u16;
                    kv[r * ROWB + 1024 + kv_h * 2] = (ku & 0xff) as u8;
                    kv[r * ROWB + 1024 + kv_h * 2 + 1] = (ku >> 8) as u8;
                    kv2[r * ROWB + 1024 + kv_h * 2] = (vu & 0xff) as u8;
                    kv2[r * ROWB + 1024 + kv_h * 2 + 1] = (vu >> 8) as u8;
                }
            }
            // Zero-scale row + zero-code row edges.
            if ctx > 5 {
                kv[5 * ROWB + 1024] = 0;
                kv[5 * ROWB + 1025] = 0;
                for d in 0..1024 {
                    kv2[3 * ROWB + d] = 0;
                }
            }
            // Q: deterministic full-range f32.
            let qn = batch * NH * HD;
            let mut q = vec![0f32; qn];
            for v in q.iter_mut() {
                *v = (rng.below(20001) as f32 / 10000.0 - 1.0) * 4.0;
            }
            // Positions: causal prefix; duplicate/nonmonotone variant every
            // third case (same for both arms).
            let mut pos: Vec<i32> = (0..batch).map(|qq| (ctx - batch + qq) as i32).collect();
            if (batch + ctx) % 3 == 0 && batch >= 3 {
                pos[1] = pos[0];
                pos[2] = pos[0];
            }
            // All-masked variant every fifth case: positions -1 (empty
            // prefix; both arms must agree, finite, no invalid reads).
            if (batch + ctx) % 5 == 0 {
                for p in pos.iter_mut() {
                    *p = -1;
                }
            }
            let mut pos_bytes = vec![0u8; batch * 4];
            for (qq, p) in pos.iter().enumerate() {
                pos_bytes[qq * 4..qq * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
            let d_k = gpu.upload_raw(&kv, &[kv.len()]).unwrap();
            let d_v = gpu.upload_raw(&kv2, &[kv2.len()]).unwrap();
            let d_q = gpu.upload_f32(&q, &[batch, NH, HD]).unwrap();
            let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
            let d_old = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
            let d_new = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
            gpu.attention_fp8_e4m3_fa2_gqa_fp8_gfx1201(
                &d_q, &d_k, &d_v, &d_old, &d_pos, NH, NKV, HD, ctx.max(1), batch,
            )
            .unwrap_or_else(|e| panic!("old direct b={batch} c={ctx}: {e:?}"));
            gpu.attention_fp8_e4m3_fa2_gqa_packet_gfx1201(
                &d_q, &d_k, &d_v, &d_new, &d_pos, NH, NKV, HD, ctx.max(1), batch,
            )
            .unwrap_or_else(|e| panic!("packet direct b={batch} c={ctx}: {e:?}"));
            let o: Vec<f32> = gpu.download_f32(&d_old).unwrap();
            let n: Vec<f32> = gpu.download_f32(&d_new).unwrap();
            cases += 1;
            let mut diff = 0usize;
            let mut first = None;
            for idx in 0..qn {
                if o[idx].to_bits() != n[idx].to_bits() {
                    if first.is_none() {
                        first = Some((idx, o[idx], n[idx]));
                    }
                    diff += 1;
                }
                if !o[idx].is_finite() || !n[idx].is_finite() {
                    // Both non-finite at the same lane is parity; one-sided
                    // is a failure recorded below via bits.
                }
            }
            // Input immutability: re-download Q/K/V/positions, compare bytes.
            let q2: Vec<f32> = gpu.download_f32(&d_q).unwrap();
            let k2 = download_bytes(&gpu, &d_k, kv.len());
            let v2 = download_bytes(&gpu, &d_v, kv2.len());
            let p2 = download_bytes(&gpu, &d_pos, pos_bytes.len());
            assert!(q2.iter().zip(q.iter()).all(|(a, b)| a.to_bits() == b.to_bits()), "Q mutated b={batch} c={ctx}");
            assert_eq!(k2, kv, "K mutated b={batch} c={ctx}");
            assert_eq!(v2, kv2, "V mutated b={batch} c={ctx}");
            assert_eq!(p2, pos_bytes, "positions mutated b={batch} c={ctx}");
            if diff != 0 {
                fails += 1;
                eprintln!("DIRECT-DIFF b={batch} c={ctx} diffs={diff}/{qn} first={first:?}");
            }
            // Partial records at splits 1 and 8 + own merges.
            for &ns in &[1usize, 8usize] {
                let np = ns * batch * NH * (HD + 2);
                let d_po = gpu.zeros(&[np], DType::F32).unwrap();
                let d_pp = gpu.zeros(&[np], DType::F32).unwrap();
                let d_mo = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
                let d_mp = gpu.zeros(&[batch * NH * HD], DType::F32).unwrap();
                gpu.attention_fp8_e4m3_fa2_gqa_stageb_split_gfx1201_bench(
                    &d_q, &d_k, &d_v, &d_mo, &d_pos, &d_po, NH, NKV, HD, batch, ns,
                )
                .unwrap_or_else(|e| panic!("old split b={batch} c={ctx} s={ns}: {e:?}"));
                gpu.attention_fp8_e4m3_fa2_gqa_packet_split_gfx1201_bench(
                    &d_q, &d_k, &d_v, &d_mp, &d_pos, &d_pp, NH, NKV, HD, batch, ns,
                )
                .unwrap_or_else(|e| panic!("packet split b={batch} c={ctx} s={ns}: {e:?}"));
                let po: Vec<f32> = gpu.download_f32(&d_po).unwrap();
                let pp: Vec<f32> = gpu.download_f32(&d_pp).unwrap();
                let mo: Vec<f32> = gpu.download_f32(&d_mo).unwrap();
                let mp: Vec<f32> = gpu.download_f32(&d_mp).unwrap();
                cases += 1;
                let pd = po.iter().zip(pp.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let md = mo.iter().zip(mp.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if pd != 0 || md != 0 {
                    fails += 1;
                    eprintln!("SPLIT-DIFF b={batch} c={ctx} s={ns} prec={pd}/{np} merge={md}/{qn}");
                }
            }
        }
    }
    eprintln!("oracle: {cases} cases, {fails} failures");
    std::fs::write(format!("{outdir}/summary.txt"), format!("cases={cases} fails={fails}\n")).unwrap();
    if fails != 0 {
        std::process::exit(1);
    }
}
