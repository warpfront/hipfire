// SPDX-License-Identifier: Apache-2.0
//! LAB-only A3B q8 attention magnitude screen. No production admission changes.
//!
//! CPU: `fa_a3b_screen --plan` or `--self-check` (neither initializes HIP).
//! GPU hold: `fa_a3b_screen > fa_a3b_screen.json` on an exclusively held gfx1201.
//! Times include two runtime Q8_0 KV writes and the complete attention sequence;
//! component measurements exclude KV writes. Allocation/JIT are outside timing.
//! The lab FA2 copy uses four compute waves plus one helper at GQA8 (160 lanes),
//! retaining the first 128 loader lanes and the production arithmetic. Full-n
//! and split-n calls deliberately exceed Rust production ingress row limits.
//! `candidate_16x48_vs_production` is the required historical arm name; here it
//! checks H24/KV4/D256, NOT 16/48 (which is a GDN geometry).
use hip_bridge::{Function, KernargBlob, Module};
use rdna_compute::{DType, Gpu, GpuTensor, KernelCompiler};
use serde_json::{json, Value};
use std::{error::Error, ffi::c_void};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;
const D: usize = 256;
const LAYERS: usize = 10;
const SHAPES: [(usize, usize); 5] = [(512, 0), (4096, 0), (8192, 0), (512, 3584), (4096, 4096)];
const LAB_SRC: &str = include_str!("lab_kernels/fa_a3b_fa2.hip");
const DIRECT: &str = "attention_q8_0_fa2_gqa_gfx1201";
const PARTIAL: &str = "attention_q8_0_fa2_gqa_partial_gfx1201";
const MERGE: &str = "attention_q8_0_fa2_gqa_merge_gfx1201";
const PRE: &str = "attention_fa2_q_preconvert_gfx1201";

fn bytes<T>(x: &[T]) -> &[u8] {
    // SAFETY: only initialized f32/i32 slices are passed; no padding.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
fn ptr(t: &GpuTensor, byte_offset: usize) -> *mut c_void {
    // SAFETY: offsets are checked by shape self-check and tensor construction.
    unsafe { t.buf.as_ptr().cast::<u8>().add(byte_offset).cast() }
}
fn tensor(gpu: &mut Gpu, n: usize, dtype: DType) -> Result<GpuTensor> {
    Ok(gpu.alloc_tensor(&[n], dtype)?)
}
fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>> {
    let mut b = vec![0u8; t.numel() * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf)?;
    Ok(b.chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect())
}

#[derive(Clone, Copy)]
struct Mode {
    segment: usize,
    splits: usize,
}
impl Mode {
    fn name(self) -> String {
        format!(
            "{}_segment{}_splits{}",
            if self.splits == 1 { "direct" } else { "split" },
            self.segment,
            self.splits
        )
    }
}
fn modes(n: usize) -> [Mode; 3] {
    [
        Mode {
            segment: 512,
            splits: 1,
        },
        Mode {
            segment: n,
            splits: 1,
        },
        Mode {
            segment: n,
            splits: 8,
        },
    ]
}
fn launch_plan(n: usize, start: usize, h: usize, kv: usize, mode: Mode) -> Value {
    let seg = n.min(mode.segment);
    json!({"segment_rows": seg, "segments": n.div_ceil(seg), "absolute_positions": [start, start+n-1],
        "kv_write_each": {"symbol":"kv_cache_write_q8_0_batched","count":2,"grid":[kv*(D/32),n,1],"block":[32,1,1],"lds_bytes":0},
        "q_preconvert": {"grid": [(seg*h).div_ceil(4),1,1], "block": [128,1,1], "lds_bytes":0},
        "attention": {"symbol": if mode.splits==1 {DIRECT} else {PARTIAL}, "grid":[seg.div_ceil(8),kv,mode.splits],
            "block":[(h/kv/2+1)*32,1,1], "lds_bytes":65536},
        "merge": if mode.splits>1 {json!({"grid":[(seg*h).div_ceil(8),1,1],"block":[256,1,1],"lds_bytes":0})} else {Value::Null},
        "scratch_bytes": {"q16":seg*h*D*2,"partials": if mode.splits>1 {seg*h*mode.splits*(D+1)*4} else {0}},
        "reserved_scratch_bytes": {"q16": if h==24 {512*h*D*2} else {n*h*D*2},
            "partials": if h==24 {256*h*8*(D+1)*4} else {n*h*8*(D+1)*4}},
        "kv_cache_bytes_each":(start+n)*kv*(D/32)*34})
}
fn self_check() {
    for (n, start) in SHAPES {
        for (h, kv) in [(16, 2), (24, 4)] {
            assert_eq!(h % kv, 0);
            assert_eq!((h / kv) % 2, 0);
            let mut visits = vec![0; h * 8];
            for kh in 0..kv {
                for wave in 0..h / kv / 2 {
                    for ml in 0..16 {
                        let head = (h / kv) * kh + 2 * wave + (ml >> 3);
                        visits[head * 8 + (ml & 7)] += 1;
                    }
                }
            }
            assert!(visits.iter().all(|&v| v == 1));
            for mode in modes(n) {
                let mut seen = 0;
                for off in (0..n).step_by(mode.segment) {
                    let rows = (n - off).min(mode.segment);
                    assert_eq!((start + off..start + off + rows).len(), rows);
                    assert!(off * h * D + rows * h * D <= n * h * D);
                    assert!(rows * h * D * 2 <= mode.segment * h * D * 2);
                    seen += rows;
                }
                assert_eq!(seen, n);
            }
            assert_eq!((start + n) * kv * (D / 32) * 34, (start + n) * kv * 272);
        }
    }
}
fn plan() -> Value {
    let mut cells = Vec::new();
    for (n, start) in SHAPES {
        cells.push(json!({"arm":"incumbent","rows":n,"start_pos":start,"layers":LAYERS,
            "sequence":["kv_cache_write_q8_0_batched(K)","kv_cache_write_q8_0_batched(V)","attention_q8_0_flash_prefill_wmma"],
            "launch_configs":{"kv_write_each":{"symbol":"kv_cache_write_q8_0_batched","count":2,"grid":[16,n,1],"block":[32,1,1],"lds_bytes":0},
                "attention":{"grid":[n.div_ceil(16),16,1],"block":[32,1,1],"lds_bytes":18176}},"scratch_bytes":0}));
        for mode in modes(n) {
            cells.push(json!({"arm":"candidate","mode":mode.name(),"rows":n,"start_pos":start,"layers":LAYERS,"launch_configs":launch_plan(n,start,16,2,mode)}));
        }
        for mode in [
            Mode {
                segment: 512,
                splits: 1,
            },
            Mode {
                segment: 256,
                splits: 8,
            },
        ] {
            cells.push(json!({"arm":"candidate_16x48_vs_production","rows":n,"start_pos":start,"layers":LAYERS,
                "geometry":"H24/KV4/D256", "mode":mode.name(),"launch_configs":launch_plan(n,start,24,4,mode),
                "required":"byte-identical outputs"}));
        }
    }
    json!({"screen":"fa_a3b","arch":"gfx1201","plan":true,"cpu_shape_self_check":"passed","cells":cells,
        "verdict_inputs":{"warmups":3,"timed_reps":10,"layers":LAYERS,"timing_scope":"two KV writes + complete attention sequence",
        "core_envelope_gain_threshold_ms_at_4096":25,"nonzero_start_causal_check":"Future K/V zeroing must preserve first-query bytes; CPU f64 reference samples first/middle/last queries and first/last heads",
        "production_limits_bypassed_in_lab_only":{"heads":"16/2 instead of24/4","batch":"full-n direct/split; split512+ instead of <=384"},
        "production_anchors":["hipfire-arch-qwen35/src/qwen35/prefill.rs:12849-12867","hipfire-arch-qwen35/src/qwen35/prefill.rs:9863-9879","hipfire-arch-qwen35/src/qwen35/prefill.rs:10215-10232","hipfire-dispatch/src/families/attention.rs:823-838","hipfire-dispatch/src/families/attention.rs:970-983","hipfire-dispatch/src/families/attention.rs:2359-2370","rdna-compute/src/attention.rs:5569-5592","rdna-compute/src/attention.rs:5737-5752","rdna-compute/src/attention.rs:5976-5980","rdna-compute/src/attention.rs:6152-6167","rdna-compute/src/attention.rs:8773-8825"]}})
}

struct Kernels {
    _module: Module,
    pre: Function,
    direct: Function,
    partial: Function,
    merge: Function,
}
impl Kernels {
    fn load(gpu: &Gpu, h: usize, kv: usize) -> Result<Self> {
        let source = format!("#define LAB_FA_HEADS {h}\n#define LAB_FA_KV_HEADS {kv}\n{LAB_SRC}");
        let mut compiler = KernelCompiler::new("gfx1201", gpu.flags.hipcc_extra_flags.clone())?;
        let object = compiler
            .compile(&format!("lab_fa_a3b_h{h}_kv{kv}"), &source)?
            .to_owned();
        let module = gpu.hip.module_load_data(&std::fs::read(object)?)?;
        let pre = gpu.hip.module_get_function(&module, PRE)?;
        let direct = gpu.hip.module_get_function(&module, DIRECT)?;
        let partial = gpu.hip.module_get_function(&module, PARTIAL)?;
        let merge = gpu.hip.module_get_function(&module, MERGE)?;
        Ok(Self {
            _module: module,
            pre,
            direct,
            partial,
            merge,
        })
    }
}
fn launch(
    gpu: &Gpu,
    f: &Function,
    grid: [u32; 3],
    block: u32,
    lds: u32,
    mut args: KernargBlob,
) -> Result<()> {
    // SAFETY: blobs match the production ABI; allocations outlive all launches.
    unsafe {
        gpu.hip
            .launch_kernel_blob(f, grid, [block, 1, 1], lds, None, args.as_mut_slice())
    }?;
    Ok(())
}
struct Data {
    n: usize,
    start: usize,
    h: usize,
    kv: usize,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    kc: GpuTensor,
    vc: GpuTensor,
    out: GpuTensor,
    reference: GpuTensor,
    q16: GpuTensor,
    partials: GpuTensor,
    qh: Vec<f32>,
    kh: Vec<u8>,
    vh: Vec<u8>,
}
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn normal(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
}
impl Data {
    fn new(gpu: &mut Gpu, n: usize, start: usize, h: usize, kv: usize) -> Result<Self> {
        let mut rng = Rng(0x41a3b060);
        let ctx = start + n;
        let qh: Vec<f32> = (0..n * h * D).map(|_| rng.normal()).collect();
        let kh: Vec<f32> = (0..ctx * kv * D).map(|_| rng.normal()).collect();
        let vh: Vec<f32> = (0..ctx * kv * D).map(|_| rng.normal() * 0.7).collect();
        let q = tensor(gpu, qh.len(), DType::F32)?;
        let k = tensor(gpu, kh.len(), DType::F32)?;
        let v = tensor(gpu, vh.len(), DType::F32)?;
        let pos = tensor(gpu, n, DType::F32)?;
        let allpos = tensor(gpu, ctx, DType::F32)?;
        gpu.hip.memcpy_htod(&q.buf, bytes(&qh))?;
        gpu.hip.memcpy_htod(&k.buf, bytes(&kh))?;
        gpu.hip.memcpy_htod(&v.buf, bytes(&vh))?;
        gpu.hip.memcpy_htod(
            &pos.buf,
            bytes(&(start as i32..ctx as i32).collect::<Vec<_>>()),
        )?;
        gpu.hip
            .memcpy_htod(&allpos.buf, bytes(&(0..ctx as i32).collect::<Vec<_>>()))?;
        let cache_bytes = ctx * kv * (D / 32) * 34;
        let kc = tensor(gpu, cache_bytes, DType::Raw)?;
        let vc = tensor(gpu, cache_bytes, DType::Raw)?;
        gpu.kv_cache_write_q8_0_batched(&kc, &k, &allpos, kv, D, ctx)?;
        gpu.kv_cache_write_q8_0_batched(&vc, &v, &allpos, kv, D, ctx)?;
        let mut kh = vec![0; cache_bytes];
        let mut vh = vec![0; cache_bytes];
        gpu.hip.memcpy_dtoh(&mut kh, &kc.buf)?;
        gpu.hip.memcpy_dtoh(&mut vh, &vc.buf)?;
        Ok(Self {
            n,
            start,
            h,
            kv,
            q,
            k,
            v,
            pos,
            kc,
            vc,
            out: tensor(gpu, n * h * D, DType::F32)?,
            reference: tensor(gpu, n * h * D, DType::F32)?,
            // H24 parity only uses direct512/split256; A3B also runs full-n.
            q16: tensor(
                gpu,
                if h == 24 {
                    512 * h * D * 2
                } else {
                    n * h * D * 2
                },
                DType::Raw,
            )?,
            partials: tensor(
                gpu,
                if h == 24 {
                    256 * h * 8 * (D + 1)
                } else {
                    n * h * 8 * (D + 1)
                },
                DType::F32,
            )?,
            qh,
            kh,
            vh,
        })
    }
    fn writes(&self, gpu: &mut Gpu) -> Result<()> {
        let k = self
            .k
            .sub_offset(self.start * self.kv * D, self.n * self.kv * D);
        let v = self
            .v
            .sub_offset(self.start * self.kv * D, self.n * self.kv * D);
        gpu.kv_cache_write_q8_0_batched(&self.kc, &k, &self.pos, self.kv, D, self.n)?;
        gpu.kv_cache_write_q8_0_batched(&self.vc, &v, &self.pos, self.kv, D, self.n)?;
        Ok(())
    }
    fn incumbent(&self, gpu: &mut Gpu) -> Result<()> {
        gpu.attention_q8_0_flash_prefill_wmma_incumbent_gfx1201_bench(
            &self.q,
            &self.kc,
            &self.vc,
            &self.reference,
            &self.pos,
            self.h,
            self.kv,
            D,
            self.start + self.n,
            self.n,
        )?;
        Ok(())
    }
    fn pre(&self, gpu: &Gpu, k: &Kernels, off: usize, n: usize) -> Result<()> {
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&self.q, off * self.h * D * 4));
        b.push_ptr(ptr(&self.q16, 0));
        b.push_ptr(std::ptr::null::<c_void>());
        b.push_ptr(std::ptr::null::<c_void>());
        b.push_i32(n as i32);
        b.push_i32(0);
        launch(
            gpu,
            &k.pre,
            [(n * self.h).div_ceil(4) as u32, 1, 1],
            128,
            0,
            b,
        )
    }
    fn body(&self, gpu: &Gpu, k: &Kernels, off: usize, n: usize, splits: usize) -> Result<()> {
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&self.q16, 0));
        b.push_ptr(ptr(&self.kc, 0));
        b.push_ptr(ptr(&self.vc, 0));
        b.push_ptr(if splits == 1 {
            ptr(&self.out, off * self.h * D * 4)
        } else {
            ptr(&self.partials, 0)
        });
        b.push_ptr(ptr(&self.pos, off * 4));
        for x in [self.h, self.kv, D, n] {
            b.push_i32(x as i32);
        }
        b.push_f32(1.0 / 16.0);
        if splits > 1 {
            b.push_i32(splits as i32);
        }
        launch(
            gpu,
            if splits == 1 { &k.direct } else { &k.partial },
            [n.div_ceil(8) as u32, self.kv as u32, splits as u32],
            ((self.h / self.kv / 2 + 1) * 32) as u32,
            65536,
            b,
        )
    }
    fn merge(&self, gpu: &Gpu, k: &Kernels, off: usize, n: usize, splits: usize) -> Result<()> {
        let mut b = KernargBlob::new();
        b.push_ptr(ptr(&self.partials, 0));
        b.push_ptr(ptr(&self.out, off * self.h * D * 4));
        for x in [n, self.h, D, splits] {
            b.push_i32(x as i32);
        }
        launch(
            gpu,
            &k.merge,
            [(n * self.h).div_ceil(8) as u32, 1, 1],
            256,
            0,
            b,
        )
    }
    fn candidate(&self, gpu: &Gpu, k: &Kernels, mode: Mode) -> Result<()> {
        for off in (0..self.n).step_by(mode.segment) {
            let n = (self.n - off).min(mode.segment);
            self.pre(gpu, k, off, n)?;
            self.body(gpu, k, off, n, mode.splits)?;
            if mode.splits > 1 {
                self.merge(gpu, k, off, n, mode.splits)?;
            }
        }
        Ok(())
    }
    fn production(&self, gpu: &mut Gpu, mode: Mode) -> Result<()> {
        for off in (0..self.n).step_by(mode.segment) {
            let n = (self.n - off).min(mode.segment);
            let q = self.q.sub_offset(off * self.h * D, n * self.h * D);
            let pos = self.pos.sub_offset(off, n);
            let out = self.reference.sub_offset(off * self.h * D, n * self.h * D);
            if mode.splits == 1 {
                gpu.attention_q8_0_fa2_gqa_gfx1201(
                    &q,
                    &self.kc,
                    &self.vc,
                    &out,
                    &pos,
                    self.h,
                    self.kv,
                    D,
                    self.start + off + n,
                    n,
                )?;
            } else {
                gpu.attention_q8_0_fa2_gqa_split_gfx1201(
                    &q,
                    &self.kc,
                    &self.vc,
                    &out,
                    &pos,
                    &self.partials,
                    self.h,
                    self.kv,
                    D,
                    n,
                    mode.splits,
                )?;
            }
        }
        Ok(())
    }
}
fn timed(gpu: &mut Gpu, mut f: impl FnMut(&mut Gpu) -> Result<()>) -> Result<Vec<f64>> {
    for _ in 0..3 {
        f(gpu)?;
    }
    gpu.hip.device_synchronize()?;
    let a = gpu.hip.event_create()?;
    let b = gpu.hip.event_create()?;
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        gpu.hip.event_record(&a, None)?;
        f(gpu)?;
        gpu.hip.event_record(&b, None)?;
        gpu.hip.event_synchronize(&b)?;
        samples.push(gpu.hip.event_elapsed_ms(&a, &b)? as f64);
    }
    gpu.hip.event_destroy(a)?;
    gpu.hip.event_destroy(b)?;
    Ok(samples)
}
fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    (v[4] + v[5]) * 0.5
}
fn compare(a: &[f32], b: &[f32]) -> Result<Value> {
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("nonfinite output or length mismatch".into());
    }
    let (mut ma, mut mr, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut mismatch = 0;
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a as f64, b as f64);
        let e = (a - b).abs();
        ma = ma.max(e);
        mr = mr.max(e / a.abs().max(1e-6));
        dot += a * b;
        aa += a * a;
        bb += b * b;
        if (a as f32).to_bits() != (b as f32).to_bits() {
            mismatch += 1;
        }
    }
    if aa == 0.0 || bb == 0.0 {
        return Err("degenerate zero attention output".into());
    }
    Ok(
        json!({"max_abs":ma,"max_rel":mr,"relative_denominator_floor":1e-6,"cosine":dot/(aa*bb).sqrt(),"mismatched_f32":mismatch,"byte_identical":mismatch==0,"elements":a.len()}),
    )
}
fn decode(cache: &[u8], pos: usize, head: usize, d: usize, kv: usize) -> f64 {
    let at = ((pos * kv + head) * (D / 32) + d / 32) * 34;
    let scale = half::f16::from_bits(u16::from_ne_bytes([cache[at], cache[at + 1]])).to_f32();
    (scale * (cache[at + 2 + d % 32] as i8 as f32)) as f64
}
fn causal_reference(data: &Data, output: &[f32]) -> Result<Value> {
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for row in [0, data.n / 2, data.n - 1] {
        for head in [0, data.h - 1] {
            let kh = head / (data.h / data.kv);
            let end = data.start + row + 1;
            let mut scores = Vec::with_capacity(end);
            for p in 0..end {
                let mut s = 0.0;
                for d in 0..D {
                    s += data.qh[(row * data.h + head) * D + d] as f64
                        * decode(&data.kh, p, kh, d, data.kv);
                }
                scores.push(s / 16.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max).exp();
                    *s
                })
                .sum();
            for d in 0..D {
                let v: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(p, s)| s * decode(&data.vh, p, kh, d, data.kv))
                    .sum();
                expected.push((v / denom) as f32);
                actual.push(output[(row * data.h + head) * D + d]);
            }
        }
    }
    let mut result = compare(&expected, &actual)?;
    result["causal_prefix_only"] = json!(true);
    result["nonzero_start_pos"] = json!(data.start > 0);
    result["sample_rows"] = json!([0, data.n / 2, data.n - 1]);
    result["sample_heads"] = json!([0, data.h - 1]);
    Ok(result)
}
// Change every future K/V byte while retaining the first query's causal prefix.
// The entire sequence still runs, but only row zero is compared. This detects
// future-cache leakage independently of incumbent/candidate agreement.
fn causal_probe(
    gpu: &mut Gpu,
    data: &Data,
    baseline: &[f32],
    output: &GpuTensor,
    mut run: impl FnMut(&mut Gpu) -> Result<()>,
) -> Result<Value> {
    let mut k = data.kh.clone();
    let mut v = data.vh.clone();
    let future = (data.start + 1) * data.kv * (D / 32) * 34;
    k[future..].fill(0);
    v[future..].fill(0);
    gpu.hip.memcpy_htod(&data.kc.buf, &k)?;
    gpu.hip.memcpy_htod(&data.vc.buf, &v)?;
    run(gpu)?;
    let changed = download(gpu, output)?;
    gpu.hip.memcpy_htod(&data.kc.buf, &data.kh)?;
    gpu.hip.memcpy_htod(&data.vc.buf, &data.vh)?;
    let row = data.h * D;
    let comparison = compare(&baseline[..row], &changed[..row])?;
    if comparison["byte_identical"] != true {
        return Err("causal mask failure: future K/V changed the first query".into());
    }
    Ok(
        json!({"future_kv_zeroed": true, "first_query_position": data.start,
        "first_query_byte_identical": true, "comparison": comparison}),
    )
}
fn cell(arm: &str, data: &Data, times: Vec<f64>, correctness: Value) -> Value {
    let m = median(&times);
    json!({"arm":arm,"rows":data.n,"start_pos":data.start,"per_layer_ms_median":m,"per_layer_ms_all":times,"layers":LAYERS,"envelope_ms":m*LAYERS as f64,"correctness":correctness,
        "reserved_scratch_bytes":{"q16":data.q16.numel(),"partials":data.partials.numel()*4}})
}
fn run() -> Result<()> {
    self_check();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|s| s == "--plan" || s == "--self-check") {
        println!("{}", plan());
        return Ok(());
    }
    if !args.is_empty() {
        return Err("usage: fa_a3b_screen [--plan|--self-check]".into());
    }
    // Reject alternate incumbent tuning rather than silently mislabel a screen.
    for (name, allowed) in [
        ("HIPFIRE_FLASH_PREFILL_SPLITQ", "0"),
        ("HIPFIRE_FLASH_PREFILL_FIXED_HD", "1"),
        ("HIPFIRE_FLASH_PREFILL_PREFETCH_V", "1"),
    ] {
        if let Ok(v) = std::env::var(name) {
            if v != allowed {
                return Err(
                    format!("unset {name}; screen requires default incumbent tuning").into(),
                );
            }
        }
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let kernels = Kernels::load(&gpu, 16, 2)?;
    let parity_kernels = Kernels::load(&gpu, 24, 4)?;
    let mut cells = Vec::new();
    let mut parity_pass = true;
    for (n, start) in SHAPES {
        let data = Data::new(&mut gpu, n, start, 16, 2)?;
        let inc = timed(&mut gpu, |g| {
            data.writes(g)?;
            data.incumbent(g)
        })?;
        let reference = download(&gpu, &data.reference)?;
        let inc_causal = causal_probe(&mut gpu, &data, &reference, &data.reference, |g| {
            data.incumbent(g)
        })?;
        let kv_time = timed(&mut gpu, |g| data.writes(g))?;
        let core_time = timed(&mut gpu, |g| data.incumbent(g))?;
        let mut c = cell(
            "incumbent",
            &data,
            inc,
            json!({"cpu_causal_reference":causal_reference(&data,&reference)?, "future_perturbation":inc_causal}),
        );
        c["pieces"] = json!({"kv_write_ms_all":kv_time,"attention_core_ms_all":core_time});
        cells.push(c);
        for mode in modes(n) {
            let times = timed(&mut gpu, |g| {
                data.writes(g)?;
                data.candidate(g, &kernels, mode)
            })?;
            let actual = download(&gpu, &data.out)?;
            let mut correctness = compare(&reference, &actual)?;
            correctness["cpu_causal_reference"] = causal_reference(&data, &actual)?;
            correctness["future_perturbation"] =
                causal_probe(&mut gpu, &data, &actual, &data.out, |g| {
                    data.candidate(g, &kernels, mode)
                })?;
            let core = timed(&mut gpu, |g| data.candidate(g, &kernels, mode))?;
            let pre = timed(&mut gpu, |g| {
                for off in (0..n).step_by(mode.segment) {
                    data.pre(g, &kernels, off, (n - off).min(mode.segment))?;
                }
                Ok(())
            })?;
            let merge = if mode.splits > 1 {
                timed(&mut gpu, |g| data.merge(g, &kernels, 0, n, mode.splits))?
            } else {
                Vec::new()
            };
            let mut c = cell("candidate", &data, times, correctness);
            c["mode"] = json!(mode.name());
            c["launch_configs"] = launch_plan(n, start, 16, 2, mode);
            c["pieces"] = json!({"attention_core_ms_all":core,"q_preconvert_ms_all":pre,"merge_ms_all":merge});
            c["core_envelope_saved_ms"] =
                json!((median(&core_time) - median(&core)) * LAYERS as f64);
            cells.push(c);
        }
        let check = Data::new(&mut gpu, n, start, 24, 4)?;
        for mode in [
            Mode {
                segment: 512,
                splits: 1,
            },
            Mode {
                segment: 256,
                splits: 8,
            },
        ] {
            check.production(&mut gpu, mode)?;
            let prod = download(&gpu, &check.reference)?;
            let times = timed(&mut gpu, |g| {
                check.writes(g)?;
                check.candidate(g, &parity_kernels, mode)
            })?;
            let lab = download(&gpu, &check.out)?;
            let comparison = compare(&prod, &lab)?;
            parity_pass &= comparison["byte_identical"] == true;
            let mut c = cell("candidate_16x48_vs_production", &check, times, comparison);
            c["geometry"] = json!("H24/KV4/D256");
            c["mode"] = json!(mode.name());
            cells.push(c);
        }
    }
    let report = json!({"screen":"fa_a3b","arch":gpu.arch,"cells":cells,"verdict_inputs":{"parity_byte_identical":parity_pass,"warmups":3,"timed_reps":10,"layers":LAYERS,
        "core_envelope_gain_threshold_ms_at_4096":25,"timing_scope":"two KV writes + complete attention sequence","core_timings_in":"pieces.attention_core_ms_all",
        "accuracy_is_measurement_not_admission":true,"causal_check":"Future K/V zeroing must preserve first-query bytes; CPU f64 reference on decoded runtime Q8_0 samples first/middle/last rows and first/last heads",
        "lab_copy_needed":true,"not_reproduced":"projection, QK norm/RoPE, output gate/projection; inputs synthetic N(0,1), V N(0,0.7); full-n and split-n bypass production row limits"}});
    println!("{report}");
    if !parity_pass {
        return Err("lab default H24/KV4 output not byte-identical to production".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("fa_a3b_screen: {e}");
        std::process::exit(1);
    }
}
