// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU parity of `GroupedMoeProtocol::RowsExact`: every row of a K-row call
//! (K = 1..=16) is bit-identical to the same row run alone as a `rows == 1`
//! decode MoE call (routed + shared + combine), with random qt44 experts at the
//! Raon dims (hidden 2560, expert width 1088, down K 1280, 128 experts, top-6).
//!
//! Needs a GPU; run only through the GPU fabric:
//! `cargo test -p hipfire-arch-raon --test moe_rows_exact_gpu --release -- --ignored --nocapture`
//! It prints decode-step timings of `RowsExact` vs the default (`Auto`) route at
//! K = 1, 2, 4, 8, 16.

use hipfire_arch_raon::moe::{raon_moe_op, RaonMoeDims, RaonMoeLayerRefs, RaonMoeScratch};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_grouped_moe, CastPolicy, GroupedMoeProtocol, GroupedMoeShared,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const GROUP_BYTES: usize = 136;
const MAX_K: usize = 16;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// bf16-valued F32 (the layer input contract): drop the low 16 mantissa bits.
fn bf16_valued(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

/// Random symmetric qt44 bytes for an `[m, k]` weight: valid f16 headers
/// (`zp == -8 * sc`, exact) and random nibbles.
fn qt44_bytes(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    // (scale, zero) f16 bits: 1/16 & -1/2, 1/32 & -1/4, 1/8 & -1.
    const HEADERS: [(u16, u16); 3] = [(0x2c00, 0xb800), (0x2800, 0xb400), (0x3000, 0xbc00)];
    let groups = m * (k / 256);
    let mut out = vec![0u8; groups * GROUP_BYTES];
    for g in out.chunks_exact_mut(GROUP_BYTES) {
        for h in 0..2 {
            let (s, z) = HEADERS[(rng.next() % 3) as usize];
            g[4 * h..4 * h + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&z.to_le_bytes());
        }
        for chunk in g[8..].chunks_exact_mut(8) {
            chunk.copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    out
}

fn up_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    gpu.upload_f32(data, &[data.len()]).expect("upload f32")
}

fn up_raw(gpu: &Gpu, data: &[u8]) -> GpuTensor {
    gpu.upload_raw(data, &[data.len()]).expect("upload raw")
}

fn ptr_table(gpu: &Gpu, ts: &[GpuTensor]) -> GpuTensor {
    let mut b = Vec::with_capacity(ts.len() * 8);
    for t in ts {
        b.extend_from_slice(&(t.buf.as_ptr() as usize as u64).to_le_bytes());
    }
    up_raw(gpu, &b)
}

fn qt44_ref(t: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf: t,
        dtype: DType::MQ4G256V2,
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

struct Run<'a> {
    gpu: &'a mut Gpu,
    ctx: &'a DispatchCtx,
    dims: RaonMoeDims,
    layer: RaonMoeLayerRefs<'a>,
    scratch: &'a RaonMoeScratch,
    cast: CastPolicy,
}

impl Run<'_> {
    /// One MoE call of `rows` rows on `x`; returns (out, routed_out) words.
    fn call(
        &mut self,
        protocol: GroupedMoeProtocol,
        rows: usize,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let hidden = self.dims.hidden;
        let xt = up_f32(self.gpu, x);
        let nan = vec![f32::NAN; rows * hidden];
        let out = up_f32(self.gpu, &nan);
        let routed = up_f32(self.gpu, &nan);
        let mut op = raon_moe_op(
            self.ctx,
            &self.dims,
            &self.layer,
            protocol,
            rows,
            &xt,
            self.scratch,
            &out,
            Some(&routed),
        );
        op.policy.cast = self.cast;
        let result = execute_grouped_moe(self.gpu, &op).map_err(|e| e.to_string());
        self.gpu
            .hip
            .device_synchronize()
            .map_err(|e| e.to_string())?;
        let got = result.map(|()| {
            (
                self.gpu.download_f32(&out).expect("download out"),
                self.gpu.download_f32(&routed).expect("download routed"),
            )
        });
        for t in [xt, out, routed] {
            let _ = self.gpu.free_tensor(t);
        }
        got
    }

    /// Mean wall ms of a call (inputs resident), `reps` timed after 3 warmups.
    fn time(&mut self, protocol: GroupedMoeProtocol, rows: usize, x: &[f32]) -> Option<f64> {
        let reps = 20;
        for _ in 0..3 {
            self.call(protocol, rows, x).ok()?;
        }
        let hidden = self.dims.hidden;
        let xt = up_f32(self.gpu, x);
        let out = up_f32(self.gpu, &vec![0.0; rows * hidden]);
        let mut op = raon_moe_op(
            self.ctx,
            &self.dims,
            &self.layer,
            protocol,
            rows,
            &xt,
            self.scratch,
            &out,
            None,
        );
        op.policy.cast = self.cast;
        self.gpu.hip.device_synchronize().ok()?;
        let t0 = Instant::now();
        for _ in 0..reps {
            execute_grouped_moe(self.gpu, &op).ok()?;
        }
        self.gpu.hip.device_synchronize().ok()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        for t in [xt, out] {
            let _ = self.gpu.free_tensor(t);
        }
        Some(ms)
    }
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn rows_exact_is_bit_identical_to_single_row_decode_for_k_1_to_16() {
    let mut gpu = Gpu::init().expect("gpu init");
    let ctx = DispatchCtx::new(&gpu);
    let dims = RaonMoeDims::raon();
    let (c, mi, down_k, e) = (dims.hidden, dims.logical_mi, dims.down_k, dims.n_experts);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let gate_up: Vec<GpuTensor> = (0..e)
        .map(|_| up_raw(&gpu, &qt44_bytes(&mut rng, 2 * mi, c)))
        .collect();
    let down: Vec<GpuTensor> = (0..e)
        .map(|_| up_raw(&gpu, &qt44_bytes(&mut rng, c, down_k)))
        .collect();
    let gate_up_ptrs = ptr_table(&gpu, &gate_up);
    let down_ptrs = ptr_table(&gpu, &down);
    let sh_gate = up_raw(&gpu, &qt44_bytes(&mut rng, mi, c));
    let sh_up = up_raw(&gpu, &qt44_bytes(&mut rng, mi, c));
    let sh_down = up_raw(&gpu, &qt44_bytes(&mut rng, c, down_k));
    let router_w: Vec<f32> = (0..e * c).map(|_| rng.unit() * 0.05).collect();
    let router_w = up_f32(&mut gpu, &router_w);
    let bias: Vec<f32> = (0..e).map(|_| rng.unit() * 0.01).collect();
    let bias = up_f32(&mut gpu, &bias);
    let x: Vec<f32> = (0..MAX_K * c).map(|_| bf16_valued(rng.unit())).collect();

    let scratch = RaonMoeScratch::new(&mut gpu, &dims, MAX_K).expect("scratch");
    let layer = RaonMoeLayerRefs {
        router_weight: &router_w,
        correction_bias: &bias,
        gate_up_ptrs: &gate_up_ptrs,
        down_ptrs: &down_ptrs,
        gate_up_awq: None,
        down_awq_ptrs: None,
        shared: GroupedMoeShared {
            gate: qt44_ref(&sh_gate, mi, c),
            up: qt44_ref(&sh_up, mi, c),
            down: qt44_ref(&sh_down, c, down_k),
        },
        iu4_symmetric: false,
    };

    for (label, cast) in [
        ("bf16", CastPolicy::Bf16Boundaries),
        ("exact32", CastPolicy::Exact32),
    ] {
        let mut run = Run {
            gpu: &mut gpu,
            ctx: &ctx,
            dims,
            layer,
            scratch: &scratch,
            cast,
        };
        // Reference: MAX_K single-row decode calls (Auto resolves to the
        // indexed core at one row).
        let mut want_out = Vec::new();
        let mut want_routed = Vec::new();
        for r in 0..MAX_K {
            let (o, rt) = run
                .call(GroupedMoeProtocol::Auto, 1, &x[r * c..(r + 1) * c])
                .expect("single-row decode call");
            want_out.extend(o);
            want_routed.extend(rt);
        }
        assert!(want_out.iter().all(|v| v.is_finite()), "{label}: reference");

        for k in 1..=MAX_K {
            let (o, rt) = run
                .call(GroupedMoeProtocol::RowsExact, k, &x[..k * c])
                .unwrap_or_else(|e| panic!("{label} K={k}: {e}"));
            for (name, got, want) in [("out", &o, &want_out), ("routed_out", &rt, &want_routed)] {
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "{label} K={k} {name}: row {} col {} differs ({g} vs {w})",
                        i / c,
                        i % c
                    );
                }
            }
        }
        eprintln!("{label}: RowsExact == single-row decode bitwise for K=1..={MAX_K}");

        // Informational: does the default route (grouped core at K>1) match?
        for k in [2usize, 16] {
            if let Ok((o, _)) = run.call(GroupedMoeProtocol::Auto, k, &x[..k * c]) {
                let diff = o
                    .iter()
                    .zip(want_out.iter())
                    .filter(|(g, w)| g.to_bits() != w.to_bits())
                    .count();
                eprintln!(
                    "{label}: default route K={k}: {diff}/{} words differ",
                    k * c
                );
            }
        }

        if label == "bf16" {
            eprintln!("timings ({label}, ms per MoE layer call, 20 reps):");
            for k in [1usize, 2, 4, 8, 16] {
                let exact = run.time(GroupedMoeProtocol::RowsExact, k, &x[..k * c]);
                let auto = run.time(GroupedMoeProtocol::Auto, k, &x[..k * c]);
                eprintln!("  K={k:2}: rows_exact {exact:?}  default {auto:?}");
            }
        }
    }
}
