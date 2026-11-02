// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The two folded exports of `iu4_c8_producer_pm_gfx1201`, `iu4_c8_sigmoid_rotate_pm_gfx1201` and
//! `iu4_c8_rmsnorm_out_rotate_pm_gfx1201`, run to completion on the PeaceMaker emulator: the emitted module is
//! assembled, lifted and executed unmodified (80-byte kernarg, exactly sized buffers, every output byte must be
//! written, inputs untouched). `iu4_c8_producer.rs` proves the unfused `rotate` export against the incumbent graph
//! on the same interpreter; a folded export must leave every byte of the unfused chain unchanged, so each is checked
//! against exactly that chain:
//!
//!  * `sigmoid_rotate(attn, gate)` equals `rotate(attn * (1 / (1 + expf(-gate))))` (the incumbent `sigmoid_mul_f32`
//!    expression, hipcc's range-reduced `expf`, IEEE division), byte for byte in the rotated plane and the sidecar;
//!  * `rmsnorm_out_rotate(x, w)` stores `y = (x * w) * rms` with `rms` from the standalone `rmsnorm_f32` reduction
//!    order (thread partial `fma` chain, then the `sdata[t] += sdata[t + s]` tree `s = 128 .. 1`, `mean = sum /
//!    2048`, the vector `v_rsq_f32`), and its rotated plane and sidecar equal `rotate(y)` (same AWQ and signs).
use peacemaker_ir::emu::{trans, Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::iu4_c8_producer::{self as c8, Export};
use peacemaker_isa::{native, Arch};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const IR_ARCH: peacemaker_ir::Arch = peacemaker_ir::Arch::Gfx1201;
const KARG: u64 = 0x1_0000;
const XB: u64 = 0x100_0000;
const ROTB: u64 = 0x200_0000;
const SCB: u64 = 0x300_0000;
const S1B: u64 = 0x400_0000;
const S2B: u64 = 0x401_0000;
const WB: u64 = 0x402_0000;
const AB: u64 = 0x403_0000;
const EXTB: u64 = 0x500_0000;
const K: usize = 2048;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Outputs {
    rot: Vec<u8>,
    sc: Vec<u8>,
    ext: Vec<u8>,
}

struct Rig {
    program: peacemaker_ir::Program,
}
impl Rig {
    fn new() -> Self {
        let (_, text, _) = c8::emit_module(ARCH).unwrap();
        let elf = native::assemble(&text, ARCH).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Rig { program }
    }

    /// One whole launch of `export`. `x` is `[rows, K]`; `ext` is the folded input (`gate`) for the sigmoid export;
    /// `weight` feeds the norm exports. Returns the rotated plane, the sidecar and (norm_out) the normalized plane.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        export: Export,
        rows: usize,
        x: &[f32],
        ext: Option<&[f32]>,
        weight: &[f32],
        awq: Option<&[f32]>,
        signs: (&[f32], &[f32]),
        eps: f32,
    ) -> Outputs {
        let emu = Emulator::new(&self.program, export.symbol()).unwrap();
        let mut karg = Vec::new();
        for ptr in [
            XB,
            ROTB,
            SCB,
            S1B,
            S2B,
            if export.is_norm() { WB } else { 0 },
            if awq.is_some() { AB } else { 0 },
        ] {
            karg.extend(ptr.to_le_bytes());
        }
        karg.extend((rows as i32).to_le_bytes());
        karg.extend((K as i32).to_le_bytes());
        karg.extend(eps.to_le_bytes());
        karg.extend([0u8; 4]);
        if export.extended() {
            karg.extend(EXTB.to_le_bytes());
        }
        assert_eq!(karg.len(), export.kernarg_bytes() as usize);
        let mut mem = Memory::default();
        mem.map("karg".into(), KARG, karg).unwrap();
        mem.map("x".into(), XB, bytes(x)).unwrap();
        mem.map("rot".into(), ROTB, vec![0xa5; rows * K * 4])
            .unwrap();
        mem.map("sc".into(), SCB, vec![0xa5; (K / 128) * rows * 72])
            .unwrap();
        mem.map("s1".into(), S1B, bytes(signs.0)).unwrap();
        mem.map("s2".into(), S2B, bytes(signs.1)).unwrap();
        if export.is_norm() {
            mem.map("w".into(), WB, bytes(weight)).unwrap();
        }
        if let Some(a) = awq {
            mem.map("awq".into(), AB, bytes(a)).unwrap();
        }
        match export {
            Export::SigmoidRotate => mem
                .map("gate".into(), EXTB, bytes(ext.expect("gate")))
                .unwrap(),
            Export::NormOut => mem
                .map("norm_out".into(), EXTB, vec![0xa5; rows * K * 4])
                .unwrap(),
            _ => {}
        }
        let grid = if export.is_norm() {
            rows
        } else {
            rows * K / 256
        };
        for wg in 0..grid {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [wg as u32, 0, 0],
                    block: [export.block(), 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{export:?} wg {wg}: {e}"));
        }
        let region = |n: &str| mem.regions.iter().find(|r| r.name == n).unwrap();
        for n in ["rot", "sc"] {
            assert_eq!(
                region(n).written.count(),
                region(n).bytes.len(),
                "{export:?}: {n} not fully written"
            );
        }
        let mut inputs = vec!["x", "s1", "s2", "karg"];
        if export == Export::SigmoidRotate {
            inputs.push("gate");
        }
        for n in inputs {
            assert_eq!(
                region(n).written.count(),
                0,
                "{export:?}: input {n} was written"
            );
        }
        let ext_bytes = if export == Export::NormOut {
            let r = region("norm_out");
            assert_eq!(
                r.written.count(),
                r.bytes.len(),
                "norm_out not fully written"
            );
            r.bytes.clone()
        } else {
            Vec::new()
        };
        Outputs {
            rot: region("rot").bytes.clone(),
            sc: region("sc").bytes.clone(),
            ext: ext_bytes,
        }
    }
}

// ---- hipcc `expf` and the sigmoid expression (the instruction graph of `expf(-g)` as the producer emits it) ----
fn exp2_hw(x: f32) -> f32 {
    f32::from_bits(trans::exp_f32(IR_ARCH, x.to_bits()))
}
fn ldexp(x: f32, n: i32) -> f32 {
    (f64::from(x) * 2f64.powi(n.clamp(-400, 400))) as f32
}
fn expf_neg(x: f32) -> f32 {
    let (c1, c2) = (f32::from_bits(0xbfb8aa3b), f32::from_bits(0xb2a5705f));
    let prod = x * c1;
    let lo = c1.mul_add(x, -prod);
    let n = prod.round_ties_even();
    let lo = c2.mul_add(x, lo);
    let r = (prod - n) + lo;
    let e = ldexp(exp2_hw(r), n as i32);
    // `v_cmp_ngt / nlt` are true on NaN: no NaN reaches here.
    let e = if !(f32::from_bits(0x42ce8ed0) < x) {
        e
    } else {
        0.0
    };
    if !(f32::from_bits(0xc2b17218) > x) {
        e
    } else {
        f32::INFINITY
    }
}
fn sigmoid_mul(attn: f32, g: f32) -> f32 {
    attn * (1.0f32 / (1.0f32 + expf_neg(g)))
}

fn setup(rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    let x: Vec<f32> = (0..rows * K).map(|_| rng.unit() * 3.0).collect();
    let mut gate: Vec<f32> = (0..rows * K).map(|_| rng.unit() * 8.0).collect();
    // The sigmoid's own range edges: far negative (exp overflow -> 0), far positive (underflow), zero.
    gate[0] = 0.0;
    gate[1] = 120.0;
    gate[2] = -120.0;
    gate[3] = 88.0;
    let weight: Vec<f32> = (0..K).map(|_| 0.5 + rng.unit().abs()).collect();
    let awq: Vec<f32> = (0..K).map(|_| 0.5 + rng.unit().abs() * 2.0).collect();
    let sign = |rng: &mut Rng| -> Vec<f32> {
        (0..256)
            .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
            .collect()
    };
    let (s1, s2) = (sign(&mut rng), sign(&mut rng));
    (x, gate, weight, s1, s2, awq)
}

#[test]
fn sigmoid_rotate_equals_the_unfused_gate_then_rotate_chain() {
    require_rocm_tool!("llvm-mc");
    let rig = Rig::new();
    for (rows, seed, with_awq) in [(1usize, 11u64, false), (3, 12, true)] {
        let (attn, gate, weight, s1, s2, awq) = setup(rows, seed);
        let awq = with_awq.then_some(awq.as_slice());
        let folded = rig.run(
            Export::SigmoidRotate,
            rows,
            &attn,
            Some(&gate),
            &weight,
            awq,
            (&s1, &s2),
            0.0,
        );
        let gated: Vec<f32> = attn
            .iter()
            .zip(&gate)
            .map(|(&a, &g)| sigmoid_mul(a, g))
            .collect();
        let unfused = rig.run(
            Export::Rotate,
            rows,
            &gated,
            None,
            &weight,
            awq,
            (&s1, &s2),
            0.0,
        );
        assert_eq!(
            folded.rot, unfused.rot,
            "rows={rows} awq={with_awq}: rotated plane"
        );
        assert_eq!(folded.sc, unfused.sc, "rows={rows} awq={with_awq}: sidecar");
    }
}

/// The `rmsnorm_f32` reduction of one row (block 256): thread partials, then the `sdata[t] += sdata[t + s]` tree.
fn rms_of(row: &[f32], eps: f32) -> f32 {
    let mut sdata = [0f32; 256];
    for (t, p) in sdata.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..K / 256 {
            let v = row[t + 256 * j];
            acc = v.mul_add(v, acc);
        }
        *p = acc;
    }
    let mut s = 128;
    while s >= 1 {
        for t in 0..s {
            sdata[t] += sdata[t + s];
        }
        s /= 2;
    }
    let mean = sdata[0] / 2048.0f32;
    let arg = eps + mean;
    // The vector v_rsq_f32 with its denormal guard.
    let r = if arg.to_bits() < 0x0080_0000 {
        f32::from_bits(trans::rsq_f32(IR_ARCH, (arg * 16777216.0f32).to_bits())) * 4096.0f32
    } else {
        f32::from_bits(trans::rsq_f32(IR_ARCH, arg.to_bits()))
    };
    r
}

#[test]
fn rmsnorm_out_rotate_stores_the_normalized_row_and_rotates_it() {
    require_rocm_tool!("llvm-mc");
    let rig = Rig::new();
    for (rows, seed, with_awq) in [(1usize, 21u64, false), (3, 22, true)] {
        let (x, _, weight, s1, s2, awq) = setup(rows, seed);
        let awq = with_awq.then_some(awq.as_slice());
        let eps = 1.0e-6f32;
        let folded = rig.run(
            Export::NormOut,
            rows,
            &x,
            None,
            &weight,
            awq,
            (&s1, &s2),
            eps,
        );
        let mut y = vec![0f32; rows * K];
        for r in 0..rows {
            let rms = rms_of(&x[r * K..][..K], eps);
            for c in 0..K {
                y[r * K + c] = (x[r * K + c] * weight[c]) * rms;
            }
        }
        let got: Vec<u32> = folded
            .ext
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for (idx, (&g, w)) in got.iter().zip(&y).enumerate() {
            assert_eq!(
                g,
                w.to_bits(),
                "rows={rows} norm_out[{idx}] (row {}, col {})",
                idx / K,
                idx % K
            );
        }
        let unfused = rig.run(
            Export::Rotate,
            rows,
            &y,
            None,
            &weight,
            awq,
            (&s1, &s2),
            eps,
        );
        assert_eq!(
            folded.rot, unfused.rot,
            "rows={rows} awq={with_awq}: rotated plane"
        );
        assert_eq!(folded.sc, unfused.sc, "rows={rows} awq={with_awq}: sidecar");
    }
}
