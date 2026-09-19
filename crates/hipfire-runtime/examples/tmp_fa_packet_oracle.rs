// SPDX-License-Identifier: Apache-2.0
// Device-resident route-N vs packet oracle. Inputs are uploaded once; cases
// vary batch/context/position views. All output/record comparisons and poison
// checks execute on device, with one compact result transfer and line per case.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::time::Instant;

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const ROWB: usize = 1032;
const MAX_BATCH: usize = 512;
const MAX_CTX: usize = 32768;
const GUARD_WORDS: usize = 64;
const GUARD_BYTES: usize = GUARD_WORDS * 4;
const POISON: u32 = 0x4f12_3456;
const METRICS_PER_STAGE: usize = 5;
const STAGES: usize = 5;
const METRICS_PER_CASE: usize = METRICS_PER_STAGE * STAGES;

const CHECK_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void fa_packet_fill_u32(
    unsigned int* p, unsigned int n, unsigned int value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}
extern "C" __global__ void fa_packet_scan_pair(
    const unsigned int* a, const unsigned int* b, unsigned int n,
    unsigned int poison, unsigned int* counts) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = a[i], y = b[i];
    if (x != y) atomicAdd(counts + 0, 1u);
    if (x == poison) atomicAdd(counts + 1, 1u);
    if (y == poison) atomicAdd(counts + 2, 1u);
}
extern "C" __global__ void fa_packet_scan_guard(
    const unsigned int* p, unsigned int n, unsigned int poison,
    unsigned int* count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] != poison) atomicAdd(count, 1u);
}
"#;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Case {
    batch: usize,
    ctx: usize,
    mode: usize,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.0
    }
}

fn cases() -> Vec<Case> {
    let ctxs = [
        1usize, 15, 16, 17, 63, 64, 65, 127, 128, 129, 511, 512, 4095, 4096,
        4097, 8191, 8192,
    ];
    let edge_batches = [
        1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512,
    ];
    let mut set = BTreeSet::new();
    // Every batch 1..=512 is exercised; contexts rotate over the complete
    // required tail set and position modes rotate independently.
    for batch in 1..=MAX_BATCH {
        set.insert(Case {
            batch,
            ctx: ctxs[(batch - 1) % ctxs.len()],
            mode: batch & 3,
        });
    }
    // Cross the ownership/tail boundaries with every required context.
    for (bi, &batch) in edge_batches.iter().enumerate() {
        for (ci, &ctx) in ctxs.iter().enumerate() {
            set.insert(Case {
                batch,
                ctx,
                mode: (bi + ci) & 3,
            });
        }
    }
    // At the first Q128 ownership tail, force every mask/position mode for
    // every context, including all-masked and duplicate/nonmonotone rows.
    for &ctx in &ctxs {
        for mode in 0..4 {
            set.insert(Case {
                batch: 129,
                ctx,
                mode,
            });
        }
    }
    // Retained plan rows, including the long contexts beyond the minimum list.
    for &(batch, ctx) in &[
        (4, 256),
        (8, 512),
        (32, 2048),
        (64, 4096),
        (128, 8192),
        (256, 16384),
        (512, 32768),
    ] {
        set.insert(Case {
            batch,
            ctx,
            mode: 0,
        });
    }
    set.into_iter().collect()
}

fn ptr_add(ptr: *mut c_void, bytes: usize) -> *mut c_void {
    unsafe { (ptr as *mut u8).add(bytes) as *mut c_void }
}

fn fill_u32(gpu: &Gpu, ptr: *mut c_void, n: usize) {
    let mut b = KernargBlob::new();
    b.push_ptr(ptr);
    b.push_u32(n as u32);
    b.push_u32(POISON);
    gpu.launch_kernel_blob(
        "fa_packet_fill_u32",
        [n.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("fill_u32");
}

fn scan_pair(
    gpu: &Gpu,
    a: *mut c_void,
    bptr: *mut c_void,
    n: usize,
    result: *mut c_void,
) {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(bptr);
    b.push_u32(n as u32);
    b.push_u32(POISON);
    b.push_ptr(result);
    gpu.launch_kernel_blob(
        "fa_packet_scan_pair",
        [n.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("scan_pair");
}

fn scan_guard(gpu: &Gpu, ptr: *mut c_void, n: usize, result: *mut c_void) {
    let mut b = KernargBlob::new();
    b.push_ptr(ptr);
    b.push_u32(n as u32);
    b.push_u32(POISON);
    b.push_ptr(result);
    gpu.launch_kernel_blob(
        "fa_packet_scan_guard",
        [n.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("scan_guard");
}

fn prepare_region(gpu: &Gpu, owner: &GpuTensor, used_words: usize) -> GpuTensor {
    fill_u32(gpu, owner.buf.as_ptr(), GUARD_WORDS + used_words + GUARD_WORDS);
    owner.sub_offset(GUARD_WORDS, used_words)
}

fn compare_stage(
    gpu: &Gpu,
    a_owner: &GpuTensor,
    b_owner: &GpuTensor,
    used_words: usize,
    result_base: *mut c_void,
) {
    let a = ptr_add(a_owner.buf.as_ptr(), GUARD_BYTES);
    let b = ptr_add(b_owner.buf.as_ptr(), GUARD_BYTES);
    scan_pair(gpu, a, b, used_words, result_base);
    let guard_a = ptr_add(result_base, 3 * 4);
    let guard_b = ptr_add(result_base, 4 * 4);
    scan_guard(gpu, a_owner.buf.as_ptr(), GUARD_WORDS, guard_a);
    scan_guard(gpu, b_owner.buf.as_ptr(), GUARD_WORDS, guard_b);
    scan_guard(
        gpu,
        ptr_add(a, used_words * 4),
        GUARD_WORDS,
        guard_a,
    );
    scan_guard(
        gpu,
        ptr_add(b, used_words * 4),
        GUARD_WORDS,
        guard_b,
    );
}

fn result_words(gpu: &Gpu, results: &GpuTensor, case_index: usize) -> [u32; METRICS_PER_CASE] {
    let byte_off = case_index * METRICS_PER_CASE * 4;
    let view = results.sub_offset(byte_off, METRICS_PER_CASE * 4);
    let mut bytes = vec![0u8; METRICS_PER_CASE * 4];
    gpu.hip.memcpy_dtoh(&mut bytes, &view.buf).expect("result dtoh");
    let mut out = [0u32; METRICS_PER_CASE];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

fn stage_text(v: &[u32], stage: usize) -> String {
    let x = &v[stage * METRICS_PER_STAGE..(stage + 1) * METRICS_PER_STAGE];
    format!("{}/{}/{}/{}/{}", x[0], x[1], x[2], x[3], x[4])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut outdir = String::from("scratch-2026-09-17/fapkt2/oracle");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" {
            outdir = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }
    std::fs::create_dir_all(&outdir).expect("create oracle evidence directory");
    let all_cases = cases();
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201");
    gpu.ensure_kernel_public("fa_packet_check", CHECK_SRC, "fa_packet_fill_u32")
        .expect("compile fill");
    gpu.ensure_kernel_public("fa_packet_check", CHECK_SRC, "fa_packet_scan_pair")
        .expect("compile pair scan");
    gpu.ensure_kernel_public("fa_packet_check", CHECK_SRC, "fa_packet_scan_guard")
        .expect("compile guard scan");

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let max_qn = MAX_BATCH * NH * HD;
    let mut q_owner_host = vec![f32::from_bits(POISON); GUARD_WORDS + max_qn + GUARD_WORDS];
    for x in &mut q_owner_host[GUARD_WORDS..GUARD_WORDS + max_qn] {
        *x = ((rng.next() % 20001) as f32 / 10000.0 - 1.0) * 4.0;
    }
    let q_ref = gpu.upload_f32(&q_owner_host, &[q_owner_host.len()]).expect("upload Q ref");
    let q_owner = gpu.upload_f32(&q_owner_host, &[q_owner_host.len()]).expect("upload Q");
    let q = q_owner.sub_offset(GUARD_WORDS, max_qn);

    let kv_bytes = MAX_CTX * ROWB;
    let mut k_owner_host = vec![0x56u8; GUARD_BYTES + kv_bytes + GUARD_BYTES];
    let mut v_owner_host = vec![0x56u8; GUARD_BYTES + kv_bytes + GUARD_BYTES];
    for row in 0..MAX_CTX {
        let kbase = GUARD_BYTES + row * ROWB;
        let vbase = GUARD_BYTES + row * ROWB;
        for d in 0..1024 {
            let mut kc = rng.next() as u8;
            let mut vc = rng.next() as u8;
            if kc & 0x7f == 0x7f {
                kc ^= 1;
            }
            if vc & 0x7f == 0x7f {
                vc ^= 1;
            }
            k_owner_host[kbase + d] = kc;
            v_owner_host[vbase + d] = vc;
        }
        for kh in 0..NKV {
            let ks = 0x3000u16 + ((row * 37 + kh * 211) & 0x0fff) as u16;
            let vs = 0x2c00u16 + ((row * 53 + kh * 173) & 0x0fff) as u16;
            k_owner_host[kbase + 1024 + 2 * kh..kbase + 1026 + 2 * kh]
                .copy_from_slice(&ks.to_le_bytes());
            v_owner_host[vbase + 1024 + 2 * kh..vbase + 1026 + 2 * kh]
                .copy_from_slice(&vs.to_le_bytes());
        }
    }
    // Zero codes/scales and head-varying headers are permanent resident edges.
    for d in 0..1024 {
        v_owner_host[GUARD_BYTES + 3 * ROWB + d] = 0;
    }
    for kh in 0..NKV {
        k_owner_host[GUARD_BYTES + 5 * ROWB + 1024 + 2 * kh..GUARD_BYTES + 5 * ROWB + 1026 + 2 * kh]
            .copy_from_slice(&0u16.to_le_bytes());
        v_owner_host[GUARD_BYTES + 7 * ROWB + 1024 + 2 * kh..GUARD_BYTES + 7 * ROWB + 1026 + 2 * kh]
            .copy_from_slice(&0u16.to_le_bytes());
    }
    let k_ref = gpu.upload_raw(&k_owner_host, &[k_owner_host.len()]).expect("upload K ref");
    let v_ref = gpu.upload_raw(&v_owner_host, &[v_owner_host.len()]).expect("upload V ref");
    let k_owner = gpu.upload_raw(&k_owner_host, &[k_owner_host.len()]).expect("upload K");
    let v_owner = gpu.upload_raw(&v_owner_host, &[v_owner_host.len()]).expect("upload V");
    let k = k_owner.sub_offset(GUARD_BYTES, kv_bytes);
    let v = v_owner.sub_offset(GUARD_BYTES, kv_bytes);

    let pos_stride_bytes = MAX_BATCH * 4;
    let mut pos_owner_host = vec![0x56u8; GUARD_BYTES + all_cases.len() * pos_stride_bytes + GUARD_BYTES];
    for (ci, case) in all_cases.iter().enumerate() {
        let dst = GUARD_BYTES + ci * pos_stride_bytes;
        for qq in 0..MAX_BATCH {
            let p = if qq >= case.batch || case.mode == 3 {
                -1
            } else if case.mode == 2 {
                case.ctx as i32 - 1
            } else if case.mode == 1 {
                let width = case.ctx.min(17);
                (case.ctx - 1 - ((qq * 7) % width)) as i32
            } else {
                let start = case.ctx.saturating_sub(case.batch);
                (start + qq).min(case.ctx - 1) as i32
            };
            pos_owner_host[dst + qq * 4..dst + qq * 4 + 4].copy_from_slice(&p.to_le_bytes());
        }
    }
    let pos_ref = gpu.upload_raw(&pos_owner_host, &[pos_owner_host.len()]).expect("upload positions ref");
    let pos_owner = gpu.upload_raw(&pos_owner_host, &[pos_owner_host.len()]).expect("upload positions");

    let max_out_words = max_qn;
    let max_partial_words = 8 * MAX_BATCH * NH * (HD + 2);
    let old_out_owner = gpu.zeros(&[GUARD_WORDS + max_out_words + GUARD_WORDS], DType::F32).expect("old out");
    let new_out_owner = gpu.zeros(&[GUARD_WORDS + max_out_words + GUARD_WORDS], DType::F32).expect("new out");
    let old_partial_owner = gpu.zeros(&[GUARD_WORDS + max_partial_words + GUARD_WORDS], DType::F32).expect("old partial");
    let new_partial_owner = gpu.zeros(&[GUARD_WORDS + max_partial_words + GUARD_WORDS], DType::F32).expect("new partial");
    let results = gpu.zeros(&[all_cases.len() * METRICS_PER_CASE * 4], DType::Raw).expect("results");

    let started = Instant::now();
    let mut failed_cases = 0usize;
    let mut lines = String::new();
    for (ci, case) in all_cases.iter().enumerate() {
        let qn = case.batch * NH * HD;
        let pos = pos_owner.sub_offset(GUARD_BYTES + ci * pos_stride_bytes, case.batch * 4);
        let result_case = ptr_add(results.buf.as_ptr(), ci * METRICS_PER_CASE * 4);

        let old_out = prepare_region(&gpu, &old_out_owner, qn);
        let new_out = prepare_region(&gpu, &new_out_owner, qn);
        gpu.attention_fp8_e4m3_fa2_gqa_fp8_gfx1201(
            &q, &k, &v, &old_out, &pos, NH, NKV, HD, case.ctx, case.batch,
        )
        .unwrap_or_else(|e| panic!("route direct {case:?}: {e:?}"));
        gpu.attention_fp8_e4m3_fa2_gqa_packet_gfx1201(
            &q, &k, &v, &new_out, &pos, NH, NKV, HD, case.ctx, case.batch,
        )
        .unwrap_or_else(|e| panic!("packet direct {case:?}: {e:?}"));
        compare_stage(&gpu, &old_out_owner, &new_out_owner, qn, result_case);

        for (split_index, &n_splits) in [1usize, 8].iter().enumerate() {
            let stage_record = 1 + split_index * 2;
            let stage_merge = stage_record + 1;
            let np = n_splits * case.batch * NH * (HD + 2);
            let old_partial = prepare_region(&gpu, &old_partial_owner, np);
            let new_partial = prepare_region(&gpu, &new_partial_owner, np);
            let old_merged = prepare_region(&gpu, &old_out_owner, qn);
            let new_merged = prepare_region(&gpu, &new_out_owner, qn);
            gpu.attention_fp8_e4m3_fa2_gqa_stageb_split_gfx1201_bench(
                &q,
                &k,
                &v,
                &old_merged,
                &pos,
                &old_partial,
                NH,
                NKV,
                HD,
                case.batch,
                n_splits,
            )
            .unwrap_or_else(|e| panic!("route split {case:?} s={n_splits}: {e:?}"));
            gpu.attention_fp8_e4m3_fa2_gqa_packet_split_gfx1201_bench(
                &q,
                &k,
                &v,
                &new_merged,
                &pos,
                &new_partial,
                NH,
                NKV,
                HD,
                case.batch,
                n_splits,
            )
            .unwrap_or_else(|e| panic!("packet split {case:?} s={n_splits}: {e:?}"));
            compare_stage(
                &gpu,
                &old_partial_owner,
                &new_partial_owner,
                np,
                ptr_add(result_case, stage_record * METRICS_PER_STAGE * 4),
            );
            compare_stage(
                &gpu,
                &old_out_owner,
                &new_out_owner,
                qn,
                ptr_add(result_case, stage_merge * METRICS_PER_STAGE * 4),
            );
        }

        let words = result_words(&gpu, &results, ci);
        let failed = words.iter().any(|&x| x != 0);
        failed_cases += usize::from(failed);
        let line = format!(
            "CASE {:04} b={} c={} mode={} direct={} s1rec={} s1merge={} s8rec={} s8merge={} {}\n",
            ci,
            case.batch,
            case.ctx,
            case.mode,
            stage_text(&words, 0),
            stage_text(&words, 1),
            stage_text(&words, 2),
            stage_text(&words, 3),
            stage_text(&words, 4),
            if failed { "FAIL" } else { "PASS" },
        );
        eprint!("{line}");
        lines.push_str(&line);
    }

    // Device-side whole-owner input immutability, including poisoned guards.
    let input_results = gpu.zeros(&[4 * 3 * 4], DType::Raw).expect("input results");
    scan_pair(&gpu, q_owner.buf.as_ptr(), q_ref.buf.as_ptr(), q_owner_host.len(), input_results.buf.as_ptr());
    scan_pair(
        &gpu,
        k_owner.buf.as_ptr(),
        k_ref.buf.as_ptr(),
        k_owner_host.len() / 4,
        ptr_add(input_results.buf.as_ptr(), 3 * 4),
    );
    scan_pair(
        &gpu,
        v_owner.buf.as_ptr(),
        v_ref.buf.as_ptr(),
        v_owner_host.len() / 4,
        ptr_add(input_results.buf.as_ptr(), 6 * 4),
    );
    scan_pair(
        &gpu,
        pos_owner.buf.as_ptr(),
        pos_ref.buf.as_ptr(),
        pos_owner_host.len() / 4,
        ptr_add(input_results.buf.as_ptr(), 9 * 4),
    );
    let mut input_bytes = vec![0u8; 4 * 3 * 4];
    gpu.hip.memcpy_dtoh(&mut input_bytes, &input_results.buf).expect("input result dtoh");
    let input_words: Vec<u32> = input_bytes
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let input_diffs = [input_words[0], input_words[3], input_words[6], input_words[9]];
    let elapsed = started.elapsed();
    let summary = format!(
        "SUMMARY cases={} failed={} input_diffs(Q/K/V/P)={:?} elapsed_s={:.3}\n",
        all_cases.len(), failed_cases, input_diffs, elapsed.as_secs_f64(),
    );
    eprint!("{summary}");
    lines.push_str(&summary);
    std::fs::write(format!("{outdir}/oracle.log"), lines).expect("write oracle log");
    if failed_cases != 0 || input_diffs.iter().any(|&x| x != 0) {
        std::process::exit(1);
    }
}
