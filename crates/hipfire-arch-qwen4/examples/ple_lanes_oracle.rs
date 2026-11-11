// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle + microbench of the lane-batched PLE chain
//! (`HIPFIRE_QWEN4_LANE_PLE_BATCH`, `GroupedDepthwiseOp::lane_states`).
//!
//! ```text
//! HIPFIRE_QWEN4_LANE_PLE_BATCH=1 ple_lanes_oracle <BASE.mq4-xts> [--steps N] [--dilation D] [--bench N]
//! ```
//!
//! The real PLE key / value / norm / convolution weights are read from the
//! artifact (`...layers.<L>.ple.{key_proj,value_proj,norm_key,norm_query,
//! norm_conv}.weight`, `conv1d.weight`; kernel size from the conv tensor,
//! dilation default 3 = the config's `ple_conv_dilation`). For every lane count
//! `L` in 2..=8 (the batched chunk sizes of the lane step):
//!
//! * incumbent: `L` separate `execute_grouped_depthwise` steps (rows = 1, own
//!   conv state, own row views: exactly the per-lane `Step::GroupedDepthwise`
//!   of `forward_lanes`);
//! * batched: ONE `execute_grouped_depthwise` with `lane_states` (rows = L).
//!
//! Both sides start from identical random streams / PLE rows / convolution
//! states (each lane different) and run `--steps` consecutive steps (default 16,
//! more than the 9 history rows, so every state row is replaced by a
//! computed one) with fresh streams and PLE rows each step. After every step
//! the streams, the normed rows, the key / value scratch rows and every lane's
//! convolution state must be byte-identical; the guard words past every batched
//! buffer must be untouched. Lanes 1 and 8 (indices 0 and L-1) are reported
//! separately; every lane is compared.
//!
//! The microbench (`--bench N`, default 200) times L = 8 end to end (host wall,
//! device synchronised around the loop) and counts HIP kernel launches per
//! step for the incumbent and the batched route.
//!
//! Last result line: `ple_lanes_oracle: <C> cells, <B> bytes compared, <M>
//! mismatching bytes`. Exit 0 pass, 1 mismatch, 2 environment.

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{execute_grouped_depthwise, GroupedDepthwiseOp};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::Path;
use std::time::Instant;

const GUARD: usize = 64;
const SENTINEL: u32 = 0xffff_ffff;
const HC: usize = 4;
const EPS: f32 = 1e-6;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 1.7320508
    }
    /// Gaussian with heavy-tailed outliers and exact zeros, scaled by `amp`.
    fn sample(&mut self, amp: f32) -> f32 {
        let v = match self.next() % 64 {
            0 => 0.0,
            1 => -0.0,
            2 => self.gauss() * 20.0,
            3 => self.gauss() * 0.001,
            _ => self.gauss(),
        };
        v * amp
    }
}

fn gen(rng: &mut Rng, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|_| rng.sample(amp)).collect()
}

/// A guarded F32 buffer: `elems` payload floats followed by `GUARD` sentinel words.
fn guarded(gpu: &mut Gpu, host: &[f32]) -> Result<GpuTensor, String> {
    let mut all = host.to_vec();
    all.extend(std::iter::repeat_n(f32::from_bits(SENTINEL), GUARD));
    gpu.upload_f32(&all, &[all.len()]).map_err(err)
}

fn rewrite(gpu: &mut Gpu, t: &GpuTensor, host: &[f32]) -> Result<(), String> {
    let mut all = host.to_vec();
    all.extend(std::iter::repeat_n(f32::from_bits(SENTINEL), GUARD));
    gpu.hip
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(all.as_ptr() as *const u8, all.len() * 4)
        })
        .map_err(err)
}

fn bf16_tensor(gpu: &mut Gpu, bytes: &[u8], numel: usize) -> Result<GpuTensor, String> {
    if bytes.len() != numel * 2 {
        return Err(format!(
            "bf16 tensor has {} bytes for {numel} elements",
            bytes.len()
        ));
    }
    let mut t = gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)?;
    t.dtype = DType::BF16;
    t.shape = vec![numel];
    Ok(t)
}

fn weight<'a>(buf: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf,
        dtype: DType::BF16,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

struct Weights {
    key: GpuTensor,
    value: GpuTensor,
    norm_key: GpuTensor,
    norm_query: GpuTensor,
    norm_conv: GpuTensor,
    conv: GpuTensor,
    channels: usize,
    hidden: usize,
    embed: usize,
    kernel: usize,
}

fn load_weights(gpu: &mut Gpu, base: &HfqFile) -> Result<Weights, String> {
    let key_name = base
        .tensors()
        .iter()
        .map(|t| t.name.as_str())
        .find(|n| n.ends_with(".ple.key_proj.weight"))
        .ok_or("no .ple.key_proj.weight tensor in the artifact")?
        .to_string();
    let prefix = key_name.trim_end_matches("key_proj.weight").to_string();
    let get = |suffix: &str| -> Result<(Vec<usize>, Vec<u8>), String> {
        let name = format!("{prefix}{suffix}");
        let (info, bytes) = base
            .tensor_data(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        Ok((
            info.shape.iter().map(|&d| d as usize).collect(),
            bytes.to_vec(),
        ))
    };
    let (key_shape, key_bytes) = get("key_proj.weight")?;
    let (value_shape, value_bytes) = get("value_proj.weight")?;
    let (conv_shape, conv_bytes) = get("conv1d.weight")?;
    let (nk_shape, nk) = get("norm_key.weight")?;
    let (nq_shape, nq) = get("norm_query.weight")?;
    let (nc_shape, nc) = get("norm_conv.weight")?;
    let channels = key_shape[0];
    let embed = key_shape[1];
    let hidden = value_shape[0];
    if value_shape[1] != embed
        || conv_shape.len() != 3
        || conv_shape[0] != channels
        || channels != HC * hidden
        || nk_shape != [channels]
        || nq_shape != [channels]
        || nc_shape != [channels]
    {
        return Err(format!(
            "unexpected PLE shapes key {key_shape:?} value {value_shape:?} conv {conv_shape:?}"
        ));
    }
    let kernel = conv_shape[2];
    println!(
        "ple tensors: prefix {prefix} channels {channels} hidden {hidden} embed {embed} kernel {kernel}"
    );
    Ok(Weights {
        key: bf16_tensor(gpu, &key_bytes, channels * embed)?,
        value: bf16_tensor(gpu, &value_bytes, hidden * embed)?,
        conv: bf16_tensor(gpu, &conv_bytes, channels * kernel)?,
        norm_key: bf16_tensor(gpu, &nk, channels)?,
        norm_query: bf16_tensor(gpu, &nq, channels)?,
        norm_conv: bf16_tensor(gpu, &nc, channels)?,
        channels,
        hidden,
        embed,
        kernel,
    })
}

/// One side's buffers for `lanes` rows.
struct Side {
    streams: GpuTensor,
    rows: GpuTensor,
    key: GpuTensor,
    value: GpuTensor,
    normed: GpuTensor,
    query: GpuTensor,
    gated: GpuTensor,
    output: GpuTensor,
    rotation: GpuTensor,
    states: Vec<GpuTensor>,
}

impl Side {
    fn new(
        gpu: &mut Gpu,
        w: &Weights,
        lanes: usize,
        streams: &[f32],
        rows: &[f32],
        states: &[Vec<f32>],
    ) -> Result<Self, String> {
        let zeros = |n: usize| vec![0.0f32; n];
        let wide = lanes * w.channels;
        Ok(Self {
            streams: guarded(gpu, streams)?,
            rows: guarded(gpu, rows)?,
            key: guarded(gpu, &zeros(wide))?,
            value: guarded(gpu, &zeros(lanes * w.hidden))?,
            normed: guarded(gpu, &zeros(wide))?,
            query: guarded(gpu, &zeros(wide))?,
            gated: guarded(gpu, &zeros(wide))?,
            output: guarded(gpu, &zeros(wide))?,
            rotation: guarded(gpu, &zeros(lanes * w.embed.max(w.hidden)))?,
            // Conv states are exact-sized (the wrappers require exact extents).
            states: states
                .iter()
                .map(|s| gpu.upload_f32(s, &[s.len()]).map_err(err))
                .collect::<Result<_, _>>()?,
        })
    }
}

#[derive(Default)]
struct Stats {
    cells: u64,
    bytes: u64,
    mismatches: u64,
}

fn compare(
    gpu: &Gpu,
    st: &mut Stats,
    tag: &str,
    got: &GpuTensor,
    want: &GpuTensor,
    payload: usize,
) -> Result<(), String> {
    let g = gpu.download_f32(got).map_err(err)?;
    let w = gpu.download_f32(want).map_err(err)?;
    st.cells += 1;
    st.bytes += (payload * 4) as u64;
    let mut bad = 0u64;
    for i in 0..payload {
        if g[i].to_bits() != w[i].to_bits() {
            bad += (g[i].to_bits() ^ w[i].to_bits())
                .to_le_bytes()
                .iter()
                .filter(|b| **b != 0)
                .count() as u64;
            if bad <= 4 {
                eprintln!(
                    "MISMATCH {tag} word {i}: batched {:#010x} incumbent {:#010x}",
                    g[i].to_bits(),
                    w[i].to_bits()
                );
            }
        }
    }
    // Guard words past the payload must be untouched on the batched side.
    for (i, v) in g.iter().enumerate().skip(payload) {
        if v.to_bits() != SENTINEL {
            bad += 4;
            eprintln!(
                "MISMATCH {tag} guard word {} overwritten ({:#010x})",
                i - payload,
                v.to_bits()
            );
        }
    }
    st.mismatches += bad;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn op<'a>(
    w: &'a Weights,
    s: &'a Side,
    streams: &'a GpuTensor,
    rows_t: &'a GpuTensor,
    keyv: &'a GpuTensor,
    valuev: &'a GpuTensor,
    normed: &'a GpuTensor,
    state: &'a GpuTensor,
    lane_states: &'a [&'a GpuTensor],
    rows: usize,
    dilation: usize,
) -> GroupedDepthwiseOp<'a> {
    GroupedDepthwiseOp {
        rotation: &s.rotation,
        state_bf16: false,
        rows_f16: false,
        key: weight(&w.key, w.channels, w.embed),
        value: weight(&w.value, w.hidden, w.embed),
        norm_key: &w.norm_key,
        norm_query: &w.norm_query,
        norm_conv: &w.norm_conv,
        conv: &w.conv,
        state,
        streams,
        rows_tensor: rows_t,
        query: &s.query,
        key_scratch: keyv,
        value_scratch: valuev,
        gated: &s.gated,
        normed,
        output: &s.output,
        rows,
        branches: HC,
        hidden: w.hidden,
        embed_dim: w.embed,
        kernel_size: w.kernel,
        dilation,
        epsilon: EPS,
        lane_states,
        projected: false,
    }
}

/// The incumbent: one `GroupedDepthwiseOp` (rows = 1) per lane over that lane's
/// row views of `s`.
fn run_incumbent(
    gpu: &mut Gpu,
    w: &Weights,
    s: &Side,
    lanes: usize,
    dilation: usize,
) -> Result<(), String> {
    for l in 0..lanes {
        let streams = s.streams.sub_offset(l * w.channels, w.channels);
        let rows_t = s.rows.sub_offset(l * w.embed, w.embed);
        let keyv = s.key.sub_offset(l * w.channels, w.channels);
        let valuev = s.value.sub_offset(l * w.hidden, w.hidden);
        let normed = s.normed.sub_offset(l * w.channels, w.channels);
        let query = s.query.sub_offset(l * w.channels, w.channels);
        let gated = s.gated.sub_offset(l * w.channels, w.channels);
        let output = s.output.sub_offset(l * w.channels, w.channels);
        let mut o = op(
            w,
            s,
            &streams,
            &rows_t,
            &keyv,
            &valuev,
            &normed,
            &s.states[l],
            &[],
            1,
            dilation,
        );
        o.query = &query;
        o.gated = &gated;
        o.output = &output;
        execute_grouped_depthwise(gpu, &o).map_err(err)?;
    }
    Ok(())
}

fn run_batched(
    gpu: &mut Gpu,
    w: &Weights,
    s: &Side,
    lanes: usize,
    dilation: usize,
) -> Result<(), String> {
    let states: Vec<&GpuTensor> = s.states[..lanes].iter().collect();
    let o = op(
        w,
        s,
        &s.streams,
        &s.rows,
        &s.key,
        &s.value,
        &s.normed,
        &s.states[0],
        &states,
        lanes,
        dilation,
    );
    execute_grouped_depthwise(gpu, &o).map_err(err)
}

fn run_lanes(
    gpu: &mut Gpu,
    w: &Weights,
    lanes: usize,
    steps: usize,
    dilation: usize,
    st: &mut Stats,
) -> Result<(), String> {
    let hist = (w.kernel - 1) * dilation * w.channels;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((lanes as u64) << 40));
    let states0: Vec<Vec<f32>> = (0..lanes).map(|_| gen(&mut rng, hist, 0.3)).collect();
    let streams0 = gen(&mut rng, lanes * w.channels, 1.0);
    let rows0 = gen(&mut rng, lanes * w.embed, 0.05);
    let inc = Side::new(gpu, w, lanes, &streams0, &rows0, &states0)?;
    let bat = Side::new(gpu, w, lanes, &streams0, &rows0, &states0)?;
    for step in 0..steps {
        if step > 0 {
            let amp = [1.0f32, 0.05, 6.0, 1.0][step % 4];
            let streams = gen(&mut rng, lanes * w.channels, amp);
            let rows = gen(&mut rng, lanes * w.embed, 0.05 * amp.max(0.2));
            for side in [&inc, &bat] {
                rewrite(gpu, &side.streams, &streams)?;
                rewrite(gpu, &side.rows, &rows)?;
            }
        }
        run_incumbent(gpu, w, &inc, lanes, dilation)?;
        run_batched(gpu, w, &bat, lanes, dilation)?;
        let wide = lanes * w.channels;
        let tag = |what: &str| format!("lanes {lanes} step {step} {what}");
        compare(gpu, st, &tag("streams"), &bat.streams, &inc.streams, wide)?;
        compare(gpu, st, &tag("normed"), &bat.normed, &inc.normed, wide)?;
        compare(gpu, st, &tag("key scratch"), &bat.key, &inc.key, wide)?;
        compare(
            gpu,
            st,
            &tag("value scratch"),
            &bat.value,
            &inc.value,
            lanes * w.hidden,
        )?;
        for l in 0..lanes {
            compare(
                gpu,
                st,
                &tag(&format!("lane {l} state")),
                &bat.states[l],
                &inc.states[l],
                hist,
            )?;
        }
        // The per-lane report of the acceptance lanes 1 and 8 (indices 0, L-1).
        if step + 1 == steps && (lanes == 8 || lanes == 2) {
            println!(
                "lanes {lanes}: lane 1 / lane {lanes} checked at every step (streams, normed, key, value, conv state); mismatching bytes so far {}",
                st.mismatches
            );
        }
    }
    Ok(())
}

fn launches() -> u64 {
    hip_bridge::memory_effects::snapshot().launches
}

fn bench(gpu: &mut Gpu, w: &Weights, iters: usize, dilation: usize) -> Result<(), String> {
    let lanes = 8usize;
    let hist = (w.kernel - 1) * dilation * w.channels;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let states0: Vec<Vec<f32>> = (0..lanes).map(|_| gen(&mut rng, hist, 0.3)).collect();
    let streams0 = gen(&mut rng, lanes * w.channels, 1.0);
    let rows0 = gen(&mut rng, lanes * w.embed, 0.05);
    let inc = Side::new(gpu, w, lanes, &streams0, &rows0, &states0)?;
    let bat = Side::new(gpu, w, lanes, &streams0, &rows0, &states0)?;
    // Launch counts of one step.
    let before = launches();
    run_incumbent(gpu, w, &inc, lanes, dilation)?;
    let inc_launches = launches() - before;
    let before = launches();
    run_batched(gpu, w, &bat, lanes, dilation)?;
    let bat_launches = launches() - before;
    gpu.hip.device_synchronize().map_err(err)?;
    println!("bench launches/step (8 lanes): incumbent {inc_launches}, batched {bat_launches}");
    for round in 0..3 {
        for (name, batched) in [("incumbent", false), ("batched", true)] {
            for _ in 0..10 {
                if batched {
                    run_batched(gpu, w, &bat, lanes, dilation)?;
                } else {
                    run_incumbent(gpu, w, &inc, lanes, dilation)?;
                }
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let t = Instant::now();
            for _ in 0..iters {
                if batched {
                    run_batched(gpu, w, &bat, lanes, dilation)?;
                } else {
                    run_incumbent(gpu, w, &inc, lanes, dilation)?;
                }
            }
            gpu.hip.device_synchronize().map_err(err)?;
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("bench round {round} {name}: {us:.1} us / 8-lane PLE chain ({iters} iters)");
        }
    }
    Ok(())
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut path = None;
    let (mut steps, mut dilation, mut bench_iters) = (16usize, 3usize, 200usize);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                steps = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--steps N")?;
                i += 2;
            }
            "--dilation" => {
                dilation = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--dilation D")?;
                i += 2;
            }
            "--bench" => {
                bench_iters = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--bench N")?;
                i += 2;
            }
            other => {
                path = Some(other.to_string());
                i += 1;
            }
        }
    }
    let path =
        path.ok_or("usage: ple_lanes_oracle <BASE> [--steps N] [--dilation D] [--bench N]")?;
    let base = HfqFile::open(Path::new(&path)).map_err(|e| format!("base: {e}"))?;
    let mut gpu = Gpu::init().map_err(err)?;
    println!("device arch: {}", gpu.arch);
    if gpu.arch != "gfx1201" || !gpu.flags.qwen4_lane_ple_batch_enabled() {
        eprintln!(
            "REFUSAL: needs exact gfx1201 with HIPFIRE_QWEN4_LANE_PLE_BATCH=1 (arch {}, flag {})",
            gpu.arch,
            gpu.flags.qwen4_lane_ple_batch_enabled()
        );
        std::process::exit(2);
    }
    println!(
        "few_row_limit {} dilation {dilation} steps {steps}",
        gpu.few_row_limit()
    );
    let w = load_weights(&mut gpu, &base)?;
    let mut st = Stats::default();
    for lanes in 2..=8 {
        run_lanes(&mut gpu, &w, lanes, steps, dilation, &mut st)?;
        println!(
            "lanes {lanes}: ok ({} cells so far, {} mismatching bytes)",
            st.cells, st.mismatches
        );
    }
    bench(&mut gpu, &w, bench_iters, dilation)?;
    println!(
        "ple_lanes_oracle: {} cells, {} bytes compared, {} mismatching bytes",
        st.cells, st.bytes, st.mismatches
    );
    Ok(st.mismatches == 0)
}

fn main() {
    match run() {
        Ok(true) => println!("PASS"),
        Ok(false) => {
            println!("FAIL");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("ple_lanes_oracle: {e}");
            std::process::exit(2);
        }
    }
}
