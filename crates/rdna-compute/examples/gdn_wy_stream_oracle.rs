// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! WY (C16) stream partition oracle (CHUNK-CANONICAL-DESIGN §3.3/§3.4, G1).
//!
//! One append of all `N` rows (then the explicit seal) is the reference. Every
//! partition of the same stream into appends — r = 1..15 head/tail splits,
//! splits either side of the absolute 512 seams, singleton appends, randomized
//! partitions — must reproduce, byte for byte, every published output row, the
//! sealed live slot (Q8 codes+scales, or the F32 state), the Q8Ef residual and
//! the stream cursor. Initial state and residual are nonzero. Formats F32, Q8,
//! Q8Ef; starts 0 (grid aligned) and 7 (off-grid phase start).
//!
//! usage: gdn_wy_stream_oracle [--rows N] [--random K] [--seed S] [--json PATH]

use rdna_compute::gdn_stream::{GdnStreamMeta, GdnStreamPhase};
use rdna_compute::tensor_ops::gdn_wy_stream::{
    gdn_wy_stream_append, gdn_wy_stream_seal, wy_stream_admitted, GdnWyAppend, GdnWyCarry,
    GdnWyEdge, GdnWySeal, WY_TILE_ROWS,
};
use rdna_compute::tensor_ops::{GatedDeltaGateBatched, GatedDeltaStepBatched, GdnStateFormat};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::Cell;

const KH: usize = 2;
const VH: usize = 4;
const HD: usize = 128;
const QKV: usize = (2 * KH + VH) * HD;
const VALUE: usize = VH * HD;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xffff) as f32 / 32768.0 - 1.0
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn up(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    gpu.hip.device_synchronize().unwrap();
    gpu.hip.memcpy_htod(&t.buf, bytes).unwrap();
}

fn down(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().unwrap();
    let mut host = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut host, &t.buf).unwrap();
    host
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Inputs {
    proj: Vec<u8>,
    gate: Vec<u8>,
    beta: Vec<u8>,
    z: Vec<u8>,
}

fn inputs(rng: &mut Rng, n: usize) -> Inputs {
    let proj: Vec<u8> = (0..n * QKV)
        .flat_map(|_| bf16_bits(rng.unit() * 1.5).to_le_bytes())
        .collect();
    let gate = f32_bytes(
        &(0..n * VH)
            .map(|_| -0.02 - 0.5 * (rng.unit() * 0.5 + 0.5))
            .collect::<Vec<_>>(),
    );
    let beta = f32_bytes(
        &(0..n * VH)
            .map(|_| 0.2 + 0.7 * (rng.unit() * 0.5 + 0.5))
            .collect::<Vec<_>>(),
    );
    let z = f32_bytes(&(0..n * VALUE).map(|_| rng.unit()).collect::<Vec<_>>());
    Inputs {
        proj,
        gate,
        beta,
        z,
    }
}

/// Nonzero initial live state: `(slot bytes, residual bytes)`.
fn initial_state(rng: &mut Rng, format: GdnStateFormat) -> (Vec<u8>, Vec<u8>) {
    match format {
        GdnStateFormat::F32 => (
            f32_bytes(
                &(0..VH * HD * HD)
                    .map(|_| rng.unit() * 0.3)
                    .collect::<Vec<_>>(),
            ),
            Vec::new(),
        ),
        _ => {
            let codes = VH * HD * HD;
            let mut slot: Vec<u8> = (0..codes).map(|_| (rng.next() & 0xff) as u8).collect();
            slot.extend(f32_bytes(
                &(0..VH * HD)
                    .map(|_| 0.002 + 0.001 * (rng.unit() * 0.5 + 0.5))
                    .collect::<Vec<_>>(),
            ));
            let ef = if format.has_residual() {
                (0..VH * HD * HD)
                    .flat_map(|_| f16_bits(rng.unit() * 0.001).to_le_bytes())
                    .collect()
            } else {
                Vec::new()
            };
            (slot, ef)
        }
    }
}

struct Rig {
    format: GdnStateFormat,
    state: GpuTensor,
    ef: Option<GpuTensor>,
    tile_entry: GpuTensor,
    pending_conv: GpuTensor,
    pending_params: GpuTensor,
    edge_z: GpuTensor,
    edge_out: GpuTensor,
    norm: GpuTensor,
    proj: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    z: GpuTensor,
    out: GpuTensor,
    n: usize,
}

impl Rig {
    fn new(gpu: &mut Gpu, format: GdnStateFormat, n: usize, inp: &Inputs, rng: &mut Rng) -> Self {
        let (state_t, ef) = match format {
            GdnStateFormat::F32 => (
                gpu.zeros(&[VH * HD * HD], DType::F32).unwrap(),
                None,
            ),
            _ => (
                gpu.zeros(&[format.storage().state_units(VH, HD, HD)], DType::Raw)
                    .unwrap(),
                format
                    .has_residual()
                    .then(|| gpu.zeros(&[VH * HD * HD], DType::F16).unwrap()),
            ),
        };
        let norm = gpu.zeros(&[HD], DType::BF16).unwrap();
        let nw: Vec<u8> = (0..HD)
            .flat_map(|_| bf16_bits(0.8 + 0.2 * rng.unit()).to_le_bytes())
            .collect();
        up(gpu, &norm, &nw);
        let rig = Self {
            format,
            state: state_t,
            ef,
            tile_entry: gpu.zeros(&[VH * HD * HD], DType::F32).unwrap(),
            pending_conv: gpu.zeros(&[WY_TILE_ROWS * QKV], DType::BF16).unwrap(),
            pending_params: gpu.zeros(&[2 * WY_TILE_ROWS * VH], DType::F32).unwrap(),
            edge_z: gpu.zeros(&[WY_TILE_ROWS * VALUE], DType::F32).unwrap(),
            edge_out: gpu.zeros(&[WY_TILE_ROWS * VALUE], DType::F32).unwrap(),
            norm,
            proj: gpu.zeros(&[n * QKV], DType::BF16).unwrap(),
            gate: gpu.zeros(&[n * VH], DType::F32).unwrap(),
            beta: gpu.zeros(&[n * VH], DType::F32).unwrap(),
            z: gpu.zeros(&[n * VALUE], DType::F32).unwrap(),
            out: gpu.zeros(&[n * VALUE], DType::F32).unwrap(),
            n,
        };
        up(gpu, &rig.proj, &inp.proj);
        up(gpu, &rig.gate, &inp.gate);
        up(gpu, &rig.beta, &inp.beta);
        up(gpu, &rig.z, &inp.z);
        rig
    }

    fn reset(&self, gpu: &Gpu, slot: &[u8], ef: &[u8], poison: bool) {
        up(gpu, &self.state, slot);
        if let Some(t) = &self.ef {
            up(gpu, t, ef);
        }
        let fill = if poison { 0xA5u8 } else { 0 };
        up(gpu, &self.tile_entry, &vec![fill; self.tile_entry.byte_size()]);
        up(gpu, &self.pending_conv, &vec![fill; self.pending_conv.byte_size()]);
        up(gpu, &self.pending_params, &vec![fill; self.pending_params.byte_size()]);
        up(gpu, &self.out, &vec![0xFF; self.out.byte_size()]);
    }
}

#[derive(PartialEq, Eq, Debug)]
struct Result {
    out: Vec<u8>,
    slot: Vec<u8>,
    ef: Vec<u8>,
    meta: GdnStreamMeta,
}

fn run(
    gpu: &mut Gpu,
    rig: &Rig,
    start: usize,
    parts: &[usize],
    initial: &(Vec<u8>, Vec<u8>),
    poison: bool,
) -> Result {
    let total: usize = parts.iter().sum();
    assert!(total <= rig.n);
    rig.reset(gpu, &initial.0, &initial.1, poison);
    let meta = Cell::new(GdnStreamMeta::sealed_at(start as u64, 0));
    let carry = GdnWyCarry {
        tile_entry: &rig.tile_entry,
        pending_conv: &rig.pending_conv,
        pending_params: &rig.pending_params,
        meta: &meta,
        recipe_id: 0x5eed,
    };
    let edge = GdnWyEdge {
        z: &rig.edge_z,
        out: &rig.edge_out,
    };
    let mut row = 0;
    for &k in parts {
        let proj = rig.proj.sub_offset(row * QKV, k * QKV);
        let gate = rig.gate.sub_offset(row * VH, k * VH);
        let beta = rig.beta.sub_offset(row * VH, k * VH);
        let z = rig.z.sub_offset(row * VALUE, k * VALUE);
        let out = rig.out.sub_offset(row * VALUE, k * VALUE);
        let step = GatedDeltaStepBatched {
            projection: &proj,
            gate: &gate,
            beta: &beta,
            state: &rig.state,
            format: rig.format,
            residual: rig.ef.as_ref(),
            output: &out,
            row_states: None,
            rows: k,
            qkv_width: QKV,
            key_heads: KH,
            value_heads: VH,
            key_dim: HD,
            value_dim: HD,
            position: start + row,
        };
        let gated = GatedDeltaGateBatched {
            recurrent_output: &out,
            z: &z,
            norm: &rig.norm,
            output: &out,
            rows: k,
            value_heads: VH,
            value_dim: HD,
        };
        let m = gdn_wy_stream_append(
            gpu,
            &GdnWyAppend {
                step: &step,
                gate: &gated,
                carry,
                edge: &edge,
            },
        )
        .expect("append");
        assert_eq!(m.position as usize, start + row + k);
        row += k;
    }
    let sealed = gdn_wy_stream_seal(
        gpu,
        &GdnWySeal {
            state: &rig.state,
            format: rig.format,
            residual: rig.ef.as_ref(),
            carry,
            edge: &edge,
            key_heads: KH,
            value_heads: VH,
            norm: &rig.norm,
        },
    )
    .expect("seal");
    assert_eq!(sealed.phase, GdnStreamPhase::Sealed);
    Result {
        out: down(gpu, &rig.out, total * VALUE * 4),
        slot: down(gpu, &rig.state, rig.state.byte_size()),
        ef: rig
            .ef
            .as_ref()
            .map_or(Vec::new(), |t| down(gpu, t, t.byte_size())),
        meta: sealed,
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = arg(&args, "--rows").map_or(1100, |v| v.parse().unwrap());
    let random: usize = arg(&args, "--random").map_or(24, |v| v.parse().unwrap());
    let seed: u64 = arg(&args, "--seed").map_or(0x57a11, |v| v.parse().unwrap());
    let mut gpu = Gpu::init().expect("GPU");
    if !wy_stream_admitted(&gpu, KH, VH, HD, HD) {
        eprintln!("WY stream not admitted on {}", gpu.arch);
        std::process::exit(2);
    }
    let mut rng = Rng(seed | 1);
    let inp = inputs(&mut rng, n);
    let mut report = Vec::new();
    let mut failures = 0usize;
    let mut cases = 0usize;
    for format in [GdnStateFormat::F32, GdnStateFormat::Q8, GdnStateFormat::Q8Ef] {
        let initial = initial_state(&mut rng, format);
        let rig = Rig::new(&mut gpu, format, n, &inp, &mut rng);
        if let Some(spec) = arg(&args, "--sweep") {
            // --sweep START,ROW,L0,L1 (F32): row ROW of a single call of L rows, L0..=L1.
            let v: Vec<usize> = spec.split(',').map(|v| v.parse().unwrap()).collect();
            let (st, row, l0, l1) = (v[0], v[1], v[2], v[3]);
            let base = run(&mut gpu, &rig, st, &[l0], &initial, false);
            let pick = |r: &Result| r.out[row * VALUE * 4..(row + 1) * VALUE * 4].to_vec();
            let b0 = pick(&base);
            for l in l0..=l1 {
                let r = run(&mut gpu, &rig, st, &[l], &initial, false);
                let rb = pick(&r);
                let d: Vec<usize> = (0..VALUE).filter(|e| rb[e * 4..e * 4 + 4] != b0[e * 4..e * 4 + 4]).collect();
                println!("SWEEP L={l} nn={} diff_elems={d:?}", l - (l / 16) * 16);
            }
            std::process::exit(0);
        }
        if let Some(spec) = arg(&args, "--probe") {
            // --probe START:p1,p2,..  (F32 only): reference [sum] vs the partition.
            let (s, ps) = spec.split_once(':').unwrap();
            let st: usize = s.parse().unwrap();
            let parts: Vec<usize> = ps.split(',').map(|v| v.parse().unwrap()).collect();
            let total: usize = parts.iter().sum();
            let reference = run(&mut gpu, &rig, st, &[total], &initial, false);
            let got = run(&mut gpu, &rig, st, &parts, &initial, true);
            let mut rows_diff = Vec::new();
            for row in 0..total {
                let lo = row * VALUE * 4;
                let cnt = (0..VALUE)
                    .filter(|e| got.out[lo + e * 4..lo + e * 4 + 4] != reference.out[lo + e * 4..lo + e * 4 + 4])
                    .count();
                if cnt > 0 {
                    rows_diff.push(format!("row{row}:{cnt}"));
                }
            }
            println!("PROBE start={st} parts={parts:?} slot_eq={} diffrows={}", got.slot == reference.slot, rows_diff.join(","));
            std::process::exit(0);
        }
        for start in [0usize, 7] {
            let reference = run(&mut gpu, &rig, start, &[n], &initial, false);
            // The seal quantized state must differ from the initial (nonzero path).
            assert_ne!(reference.slot, initial.0, "reference moved the state");
            let mut partitions: Vec<(String, Vec<usize>)> = Vec::new();
            for r in 1..=15 {
                partitions.push((format!("split{r}"), vec![r, n - r]));
                partitions.push((format!("tail{r}"), vec![n - r, r]));
            }
            for at in [511usize, 512, 513, 1023, 1024, 1025] {
                if at < n {
                    partitions.push((format!("seam{at}"), vec![at, n - at]));
                }
            }
            for at in [16usize, 17, 31, 64, 65] {
                partitions.push((format!("grid{at}"), vec![at, n - at]));
            }
            let mut singles = vec![1usize; 40];
            singles.push(n - 40);
            partitions.push(("singletons40".into(), singles));
            let mut tail_single = vec![n - 20];
            tail_single.extend(vec![1usize; 20]);
            partitions.push(("singleton_tail20".into(), tail_single));
            for k in 0..random {
                let mut left = n;
                let mut parts = Vec::new();
                while left > 0 {
                    let take = (1 + (rng.next() as usize % 300)).min(left);
                    parts.push(take);
                    left -= take;
                }
                partitions.push((format!("random{k}"), parts));
            }
            for (name, parts) in partitions {
                cases += 1;
                // Poisoned carry scratch must not leak into the result.
                let got = run(&mut gpu, &rig, start, &parts, &initial, true);
                let ok = got == reference;
                if !ok {
                    failures += 1;
                    let first = got
                        .out
                        .iter()
                        .zip(&reference.out)
                        .position(|(a, b)| a != b);
                    eprintln!(
                        "MISMATCH {format:?} start={start} {name} parts={parts:?} out_first_diff_row={:?} slot_eq={} ef_eq={} meta_eq={}",
                        first.map(|b| b / (VALUE * 4)),
                        got.slot == reference.slot,
                        got.ef == reference.ef,
                        got.meta == reference.meta
                    );
                    let mut rows_diff: Vec<String> = Vec::new();
                    for row in 0..n {
                        let lo = row * VALUE * 4;
                        let mut cnt = 0;
                        let mut firstd = None;
                        for e in 0..VALUE {
                            let o = lo + e * 4;
                            if got.out[o..o + 4] != reference.out[o..o + 4] {
                                cnt += 1;
                                if firstd.is_none() {
                                    let a = f32::from_le_bytes(got.out[o..o + 4].try_into().unwrap());
                                    let b = f32::from_le_bytes(reference.out[o..o + 4].try_into().unwrap());
                                    firstd = Some((e, a, b));
                                }
                            }
                        }
                        if cnt > 0 && rows_diff.len() < 12 {
                            rows_diff.push(format!("row{row}:{cnt}diff first={firstd:?}"));
                        }
                    }
                    eprintln!("  DIFFROWS {}", rows_diff.join(" | "));
                }
                report.push(format!(
                    "{{\"format\":\"{format:?}\",\"start\":{start},\"case\":\"{name}\",\"appends\":{},\"byte_exact\":{ok}}}",
                    parts.len()
                ));
            }
        }
    }
    let json = format!(
        "{{\"oracle\":\"gdn_wy_stream\",\"arch\":\"{}\",\"rows\":{n},\"cases\":{cases},\"failures\":{failures},\"results\":[{}]}}",
        gpu.arch,
        report.join(",")
    );
    if let Some(path) = arg(&args, "--json") {
        std::fs::write(path, &json).unwrap();
    }
    println!(
        "gdn_wy_stream_oracle: arch={} cases={cases} failures={failures}",
        gpu.arch
    );
    std::process::exit(i32::from(failures != 0));
}
