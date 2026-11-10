// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dispatch oracle of the gfx942 batched GDN lanes Core (`pipeline/gdn.rs`,
//! `HIPFIRE_QWEN4_GDN_LANES=1` on exact gfx942) against the per-lane Core it replaces.
//!
//! Both arms are driven through the real dispatch entry points `validate_steps` /
//! `execute_validated_steps` with `DispatchCtx::new(&gpu)`:
//!
//! * reference: one step list per lane holding that lane's `Step::GatedDeltaNet` Core op (a
//!   group of one never groups): conv + gate parameters, `gated_delta_step_batched`
//!   (`gated_delta_step_pipe128_f32`), gated norm, three launches per lane;
//! * candidate: ONE step list of all N lanes' Core ops with the lanes flag on: the grouper of
//!   `steps.rs` hands them to `execute_core_lanes_f32_gfx942`, which runs every lane's conv +
//!   params, ONE `qwen4_gdn_lanes_step_f32_gfx942` launch, every lane's gated norm.
//!
//! Each lane owns separate allocations of every tensor it touches (state, conv ring, projection
//! and conv output rows, a / b, gate / beta, recurrent output, z, gated output): the lane table
//! cannot lean on a shared scratch stride. Lanes share the conv / norm / a_log / dt_bias
//! weights. Per step every lane gets fresh random inputs and its own cursor (`position % 3`);
//! outputs are sentinel-filled first; after each step every lane tensor of both arms is
//! compared as bytes (a sentinel left behind fails). The launch count of each arm is checked
//! (`hip_bridge::memory_effects`: reference 3N, candidate 2N + 1) so a silent per-lane fallback
//! cannot pass, and negative controls MUST be detected: the candidate with two lanes' state
//! tensors swapped, and a candidate lane at the wrong position (its cursor off by one).
//!
//! `cargo run --release -p hipfire-dispatch --example gdn_lanes_f32_gfx942_dispatch_oracle --
//!    [--lanes 2,8,33,64] [--steps 3] [--seed U64]`
//!
//! Prints one line per case / control and a final `PASS|FAIL gdn_lanes_f32_gfx942_dispatch_oracle`.
//! Exit 0 = PASS, 1 = a mismatch / undetected control / wrong launch count, 2 = not exact gfx942.

use hip_bridge::memory_effects;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, GatedDeltaNetOp, MixerPhase, Step,
};
use rdna_compute::tensor_ops::GdnStateFormat;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Arc;

const NK: usize = 16;
const NV: usize = 48;
const HD: usize = 128;
const QKV: usize = 2 * NK * HD + NV * HD;
const VALUE: usize = NV * HD;
const STATE: usize = NV * HD * HD;
const HIDDEN: usize = 2560;
const SENT: u32 = 0xFFFF_FFFF;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u32 << 24) as f32)
    }
}

fn rv(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.range(lo, hi)).collect()
}

fn bf16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

fn bf16_tensor(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect();
    let t = gpu.alloc_tensor(&[v.len()], DType::BF16).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("upload");
    t
}

fn f32_tensor(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload")
}

fn htod(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    assert_eq!(t.numel(), v.len());
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    gpu.bind_thread().expect("bind");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

fn dtoh(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(t).expect("dtoh")
}

fn words_diff(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// One lane's tensors in one arm.
struct LaneBufs {
    projection: GpuTensor,
    projection2: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    recurrent: GpuTensor,
    conv_state: GpuTensor,
    recurrent_output: GpuTensor,
    z: GpuTensor,
    output: GpuTensor,
}

impl LaneBufs {
    fn new(gpu: &mut Gpu, state: &[f32], ring: &[f32]) -> Self {
        Self {
            projection: f32_tensor(gpu, &vec![0.0; QKV]),
            projection2: f32_tensor(gpu, &vec![0.0; QKV]),
            a: f32_tensor(gpu, &vec![0.0; NV]),
            b: f32_tensor(gpu, &vec![0.0; NV]),
            gate: f32_tensor(gpu, &vec![0.0; NV]),
            beta: f32_tensor(gpu, &vec![0.0; NV]),
            recurrent: f32_tensor(gpu, state),
            conv_state: f32_tensor(gpu, ring),
            recurrent_output: f32_tensor(gpu, &vec![0.0; VALUE]),
            z: f32_tensor(gpu, &vec![0.0; VALUE]),
            output: f32_tensor(gpu, &vec![0.0; VALUE]),
        }
    }
    fn tensors(&self) -> [(&'static str, &GpuTensor); 9] {
        [
            ("projection2 (conv out)", &self.projection2),
            ("gate", &self.gate),
            ("beta", &self.beta),
            ("state", &self.recurrent),
            ("conv ring", &self.conv_state),
            ("recurrent output", &self.recurrent_output),
            ("gated output", &self.output),
            ("projection (input)", &self.projection),
            ("z", &self.z),
        ]
    }
}

struct Shared {
    conv: GpuTensor,
    a_log: GpuTensor,
    dt_bias: GpuTensor,
    norm: GpuTensor,
    out_w: GpuTensor,
    bf16_scratch: GpuTensor,
    rotation: GpuTensor,
    input: GpuTensor,
    output_tensor: GpuTensor,
}

fn weight<'a>(w: &'a GpuTensor, m: usize, k: usize) -> WeightRef<'a> {
    WeightRef {
        buf: w,
        dtype: DType::BF16,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn core_op<'a>(s: &'a Shared, l: &'a LaneBufs, position: usize) -> GatedDeltaNetOp<'a> {
    GatedDeltaNetOp {
        qkv: weight(&s.out_w, QKV, HIDDEN),
        conv: &s.conv,
        in_proj_a: weight(&s.out_w, NV, HIDDEN),
        in_proj_b: weight(&s.out_w, NV, HIDDEN),
        a_log: &s.a_log,
        dt_bias: &s.dt_bias,
        z: weight(&s.out_w, VALUE, HIDDEN),
        norm: &s.norm,
        output: weight(&s.out_w, HIDDEN, VALUE),
        recurrent: &l.recurrent,
        format: GdnStateFormat::F32,
        residual: None,
        conv_state: &l.conv_state,
        projection: &l.projection,
        projection2: &l.projection2,
        a: &l.a,
        b: &l.b,
        gate: &l.gate,
        beta: &l.beta,
        recurrent_output: &l.recurrent_output,
        bf16_scratch: &s.bf16_scratch,
        z_output: &l.z,
        output_scratch: &l.output,
        input: &s.input,
        output_tensor: &s.output_tensor,
        rows: 1,
        start_position: position,
        key_heads: NK,
        value_heads: NV,
        key_dim: HD,
        value_dim: HD,
        conv_kernel: 4,
        input_width: HIDDEN,
        rotation: &s.rotation,
        row_capture: None,
        zba_fold: None,
        trunk_a4: 0,
        phase: MixerPhase::Core,
    }
}

fn run(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), String> {
    validate_steps(gpu, steps).map_err(|e| format!("validate: {e:?}"))?;
    execute_validated_steps(gpu, ctx, steps).map_err(|e| format!("execute: {e:?}"))
}

#[derive(Clone, Copy, PartialEq)]
enum Control {
    None,
    SwappedStates,
    WrongPosition,
}

fn run_case(gpu: &mut Gpu, ctx: &DispatchCtx, sh: &Shared, n: usize, steps: usize, seed: u64) -> bool {
    let mut rng = Rng(seed ^ ((n as u64) << 32));
    let init: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
        .map(|_| (rv(&mut rng, STATE, -0.5, 0.5), rv(&mut rng, 3 * QKV, -1.5, 1.5)))
        .collect();
    let mut positions: Vec<usize> = (0..n).map(|i| (rng.next() % 1000) as usize + i).collect();
    let mut ok = true;
    for control in [Control::None, Control::SwappedStates, Control::WrongPosition] {
        if control != Control::None && n < 2 {
            continue;
        }
        let refr: Vec<LaneBufs> = init.iter().map(|(s, r)| LaneBufs::new(gpu, s, r)).collect();
        let cand: Vec<LaneBufs> = init.iter().map(|(s, r)| LaneBufs::new(gpu, s, r)).collect();
        let mut pos = positions.clone();
        let mut detail = Vec::new();
        let (mut total_diff, mut launch_ok, mut sentinel_left) = (0usize, true, 0usize);
        for step in 0..steps {
            for l in 0..n {
                let (p, a, b, z) = (
                    rv(&mut rng, QKV, -1.0, 1.0),
                    rv(&mut rng, NV, -2.0, 2.0),
                    rv(&mut rng, NV, -3.0, 3.0),
                    rv(&mut rng, VALUE, -3.0, 3.0),
                );
                for arm in [&refr[l], &cand[l]] {
                    htod(gpu, &arm.projection, &p);
                    htod(gpu, &arm.a, &a);
                    htod(gpu, &arm.b, &b);
                    htod(gpu, &arm.z, &z);
                    let sent = |len| vec![f32::from_bits(SENT); len];
                    htod(gpu, &arm.projection2, &sent(QKV));
                    htod(gpu, &arm.gate, &sent(NV));
                    htod(gpu, &arm.beta, &sent(NV));
                    htod(gpu, &arm.recurrent_output, &sent(VALUE));
                    htod(gpu, &arm.output, &sent(VALUE));
                }
            }
            // Reference: one list per lane, flag off.
            Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = false;
            let m0 = memory_effects::snapshot();
            for l in 0..n {
                let list = [Step::GatedDeltaNet(core_op(sh, &refr[l], pos[l]))];
                run(gpu, ctx, &list).expect("reference lane");
            }
            let ref_launches = memory_effects::snapshot().since(m0).launches;
            // Candidate: one list, flag on.
            Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = true;
            let mut order: Vec<usize> = (0..n).collect();
            let mut cpos = pos.clone();
            if control == Control::WrongPosition {
                cpos[n - 1] += 1;
            }
            let swapped: Vec<&LaneBufs> = (0..n)
                .map(|l| match (control, l) {
                    (Control::SwappedStates, 0) => &cand[1],
                    (Control::SwappedStates, 1) => &cand[0],
                    _ => &cand[l],
                })
                .collect();
            let list: Vec<Step<'_>> = order
                .iter()
                .map(|&l| {
                    let mut op = core_op(sh, &cand[l], cpos[l]);
                    // Swap control: lane 0 runs on lane 1's state and ring, and vice versa.
                    op.recurrent = &swapped[l].recurrent;
                    op.conv_state = &swapped[l].conv_state;
                    Step::GatedDeltaNet(op)
                })
                .collect();
            let m1 = memory_effects::snapshot();
            run(gpu, ctx, &list).expect("candidate step list");
            let cand_launches = memory_effects::snapshot().since(m1).launches;
            order.clear();
            Arc::make_mut(&mut gpu.flags).qwen4_gdn_lanes = false;
            let launches_good =
                ref_launches == 3 * n as u64 && cand_launches == 2 * n as u64 + 1;
            if control == Control::None {
                launch_ok &= launches_good;
                detail.push(format!(
                    "step {step}: launches reference {ref_launches} (want {}) candidate {cand_launches} (want {})",
                    3 * n,
                    2 * n + 1
                ));
            }
            for l in 0..n {
                for ((name, r), (_, c)) in refr[l].tensors().into_iter().zip(cand[l].tensors()) {
                    let (rv_, cv_) = (dtoh(gpu, r), dtoh(gpu, c));
                    let d = words_diff(&rv_, &cv_);
                    if d != 0 && control == Control::None {
                        detail.push(format!("  lane {l} {name}: {d} differing words"));
                    }
                    total_diff += d;
                    if matches!(name, "projection2 (conv out)" | "gate" | "beta" | "recurrent output" | "gated output") {
                        sentinel_left += rv_.iter().chain(&cv_).filter(|x| x.to_bits() == SENT).count();
                    }
                }
            }
            for p in &mut pos {
                *p += 1;
            }
        }
        match control {
            Control::None => {
                let good = total_diff == 0 && launch_ok && sentinel_left == 0;
                ok &= good;
                println!(
                    "[lanes={n}] dispatch lanes route vs per-lane Core, {steps} steps: differing words {total_diff}, sentinel words left {sentinel_left}, launch counts {} => {}",
                    if launch_ok { "as expected" } else { "WRONG" },
                    if good { "BYTE-EQUAL" } else { "MISMATCH" }
                );
                for line in detail.iter().filter(|l| l.starts_with(' ') || !launch_ok) {
                    println!("{line}");
                }
            }
            Control::SwappedStates | Control::WrongPosition => {
                let name = if control == Control::SwappedStates {
                    "two lanes' states + rings swapped"
                } else {
                    "last lane's position off by one"
                };
                let detected = total_diff > 0;
                ok &= detected;
                println!(
                    "[lanes={n}] control ({name}): {total_diff} differing words (must be detected) => {}",
                    if detected { "ok" } else { "UNDETECTED" }
                );
            }
        }
        positions = positions.iter().map(|p| p + steps).collect();
    }
    ok
}

fn main() {
    let mut lanes = vec![2usize, 8, 33, 64];
    let mut steps = 3usize;
    let mut seed = 0x6469_7370_3234_3200u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--lanes" => lanes = v.split(',').map(|s| s.parse().expect("lanes")).collect(),
            "--steps" => steps = v.parse().expect("steps"),
            "--seed" => seed = v.parse().expect("seed"),
            other => panic!("unknown argument {other}"),
        }
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.try_gfx942().is_none() {
        eprintln!("gdn_lanes_f32_gfx942_dispatch_oracle: needs exact gfx942, this is {}", gpu.arch);
        std::process::exit(2);
    }
    let ctx = DispatchCtx::new(&gpu);
    let mut rng = Rng(seed);
    let sh = Shared {
        conv: bf16_tensor(&mut gpu, &rv(&mut rng, QKV * 4, -1.0, 1.0)),
        a_log: bf16_tensor(&mut gpu, &rv(&mut rng, NV, 0.0, 2.7)),
        dt_bias: bf16_tensor(&mut gpu, &rv(&mut rng, NV, -1.0, 1.0)),
        norm: bf16_tensor(&mut gpu, &rv(&mut rng, HD, 0.5, 1.5)),
        out_w: bf16_tensor(&mut gpu, &vec![0.0; HIDDEN * VALUE]),
        bf16_scratch: bf16_tensor(&mut gpu, &vec![0.0; VALUE]),
        rotation: f32_tensor(&mut gpu, &vec![0.0; VALUE]),
        input: f32_tensor(&mut gpu, &vec![0.0; HIDDEN]),
        output_tensor: f32_tensor(&mut gpu, &vec![0.0; HIDDEN]),
    };
    let mut all = true;
    for &n in &lanes {
        all &= run_case(&mut gpu, &ctx, &sh, n, steps, seed);
    }
    println!(
        "{} gdn_lanes_f32_gfx942_dispatch_oracle",
        if all { "PASS" } else { "FAIL" }
    );
    std::process::exit(if all { 0 } else { 1 });
}
