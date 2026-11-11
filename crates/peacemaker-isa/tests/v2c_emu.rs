// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1100 iu4_v2c SET/ADD/gate-up kernels on the PeaceMaker CPU emulator.
//!
//! gfx1100 declares 14 user SGPRs (LLVM `FeatureUserSGPRInit16Bug` padding), so the workgroup ids are `s14`/`s15`:
//! a kernel that read `s2`/`s3` would see zeros and write one tile. SET/ADD are checked against the exact CPU fold
//! (`support/iu4_emu.rs`). `v2c_old_object_vs_new_object_are_byte_identical` (ignored; needs
//! `PM_V2C_OLD_OBJECT=<previous .hxaco>`) runs the same launches through a previous object and the freshly emitted
//! one and requires byte-identical outputs for all three entries.
#![allow(dead_code)]
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

use iu4_emu::{reference, Problem};
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::iu4_v2c::{self, Epi, Spec};
use peacemaker_isa::Arch;

const KARG: u64 = 0x1000_0000;
const A: u64 = 0x2000_0000;
const U: u64 = 0x3000_0000;
const XQ: u64 = 0x4000_0000;
const Y: u64 = 0x6000_0000;
const LDS: u32 = 32768;

fn fresh_object() -> Vec<u8> {
    let (_, text, _) = iu4_v2c::module(Arch::Gfx1100, &Epi::ALL).unwrap();
    peacemaker_isa::native::assemble(&text, Arch::Gfx1100).unwrap()
}

fn lift(object: &[u8]) -> peacemaker_ir::inst::Program {
    peacemaker_lift::lift_object(
        object,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn ptr(b: &mut Vec<u8>, p: u64) {
    b.extend_from_slice(&p.to_le_bytes());
}

/// One launch of `epi` over `workgroups` (x = token tile, y = row tile), returning the whole output buffer.
/// SET/ADD: `A Xq Y M K N`; gate/up: `G U Xq H M K N` with `M` the gate rows (grid y covers `2M/128`).
struct Case {
    epi: Epi,
    m: usize,
    n: usize,
    k: usize,
    /// Weights `[M x K/256 x 136]` (gate) and, for gate/up, up weights.
    w: Vec<u8>,
    w_up: Vec<u8>,
    recs: Vec<u8>,
    y0: Vec<f32>,
}

impl Case {
    fn new(epi: Epi, m: usize, n: usize, k: usize, seed: u64) -> Self {
        let p = Problem::synth(m, n, k, seed);
        let w_up = if epi == Epi::Silu {
            Problem::synth(m, n, k, seed ^ 0x5555).w
        } else {
            Vec::new()
        };
        Self {
            epi,
            m,
            n,
            k,
            w: p.w,
            w_up,
            recs: p.recs,
            y0: p.y0,
        }
    }
    fn grid(&self) -> (usize, usize) {
        let rows = if self.epi == Epi::Silu {
            2 * self.m
        } else {
            self.m
        };
        (self.n / 128, rows / 128)
    }
    fn run(&self, program: &peacemaker_ir::inst::Program, workgroups: &[(u32, u32)]) -> Vec<u8> {
        let symbol = Spec {
            arch: Arch::Gfx1100,
            epi: self.epi,
        }
        .symbol();
        // The emulator has no hardware-qualified gfx1100 WMMA model: gfx1151 numerics (exact for the integer WMMA chain),
        // valid for comparing arms with each other and with the exact CPU fold.
        let emu = Emulator::new(program, &symbol)
            .unwrap()
            .gfx1100_as_gfx1151();
        let mut kern = Vec::new();
        let out_len = self.n * self.m * 4;
        if self.epi == Epi::Silu {
            for p in [A, U, XQ, Y] {
                ptr(&mut kern, p);
            }
        } else {
            for p in [A, XQ, Y] {
                ptr(&mut kern, p);
            }
        }
        for v in [self.m, self.k, self.n] {
            kern.extend_from_slice(&(v as i32).to_le_bytes());
        }
        let mut mem = Memory::default();
        mem.map("kernarg".into(), KARG, kern).unwrap();
        let y0: Vec<u8> = if self.epi == Epi::Add {
            self.y0.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            vec![0; out_len]
        };
        let mut bufs = vec![("a", A, self.w.clone()), ("xq", XQ, self.recs.clone())];
        if self.epi == Epi::Silu {
            bufs.push(("u", U, self.w_up.clone()));
        }
        bufs.push(("y", Y, y0));
        for (name, va, bytes) in bufs {
            let len = bytes.len() as u64;
            mem.map(name.into(), va, bytes).unwrap();
            mem.map(format!("{name}_tail"), va + len, vec![0; 4096])
                .unwrap();
        }
        for &(x, y) in workgroups {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [x, y, 0],
                    block: [256, 1, 1],
                    dynamic_lds: LDS,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{:?} workgroup ({x},{y}): {e}", self.epi));
        }
        for name in ["a_tail", "xq_tail", "y_tail"] {
            let r = mem.regions.iter().find(|r| r.name == name).unwrap();
            assert!(
                r.bytes.iter().all(|&b| b == 0),
                "{:?}: wrote past {name}",
                self.epi
            );
        }
        mem.read(Y, out_len).unwrap().to_vec()
    }
    fn full_grid(&self) -> Vec<(u32, u32)> {
        let (gx, gy) = self.grid();
        (0..gy)
            .flat_map(|y| (0..gx).map(move |x| (x as u32, y as u32)))
            .collect()
    }
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn set_and_add_equal_the_cpu_fold_on_the_whole_grid() {
    let object = fresh_object();
    let lifted = lift(&object);
    // Small (one tile), a 2x2 grid with two K epochs per group, and K = 3 groups.
    for (m, n, k) in [(128, 128, 256), (256, 256, 512), (128, 256, 768)] {
        for epi in [Epi::Set, Epi::Add] {
            let case = Case::new(epi, m, n, k, 0x2c0ffee ^ (m * 31 + n) as u64);
            let got = as_f32(&case.run(&lifted, &case.full_grid()));
            let p = Problem {
                m,
                n,
                k,
                w: case.w.clone(),
                recs: case.recs.clone(),
                y0: case.y0.clone(),
            };
            let want = reference(&p, epi == Epi::Add);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{epi:?} M{m} N{n} K{k}"
            );
        }
    }
}

/// Sampled workgroups of a production-extent grid: corners, quartile tiles and the middle.
fn sampled(gx: usize, gy: usize) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for x in [0, gx / 4, gx / 2, 3 * gx / 4, gx - 1] {
        for y in [0, gy / 4, gy / 2, 3 * gy / 4, gy - 1] {
            v.push((x as u32, y as u32));
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Previous object vs freshly emitted object, byte for byte, every entry, small full grids and production-extent
/// sampled grids (the banded raster is active at N >= 4096).
#[test]
#[ignore = "needs PM_V2C_OLD_OBJECT=<previous gemm_mq4g256v2_residual_iu4_pm_gfx1100.hxaco>"]
fn v2c_old_object_vs_new_object_are_byte_identical() {
    let old = std::fs::read(std::env::var("PM_V2C_OLD_OBJECT").unwrap()).unwrap();
    let old = lift(&old);
    let new = lift(&fresh_object());
    let mut compared = 0usize;
    for epi in Epi::ALL {
        // Full grids at small shapes.
        for (m, n, k) in [(128, 128, 256), (256, 256, 512), (128, 512, 768)] {
            let case = Case::new(epi, m, n, k, 0x71 ^ (m + n) as u64);
            let wg = case.full_grid();
            let a = case.run(&old, &wg);
            let b = case.run(&new, &wg);
            assert!(a == b, "{epi:?} M{m} N{n} K{k}: old != new");
            assert!(a.iter().any(|&x| x != 0), "{epi:?}: all-zero output");
            compared += a.len();
        }
        // Production extents, sampled workgroups: M = 6144 (qkv/z-like), K = 2560, N = 4096 tokens.
        let (m, n, k) = (6144, 4096, 2560);
        let mut case = Case::new(epi, m, n, k, 0xabc ^ epi as u64);
        if epi == Epi::Silu {
            // Keep the gate/up sums inside the exponent range the emulator's f32 divide model has measured.
            for r in case.recs.chunks_exact_mut(iu4_emu::REC) {
                let d = f32::from_le_bytes(r[0..4].try_into().unwrap()) / 64.0;
                r[0..4].copy_from_slice(&d.to_le_bytes());
            }
        }
        let (gx, gy) = case.grid();
        let wg = sampled(gx, gy);
        let a = case.run(&old, &wg);
        let b = case.run(&new, &wg);
        assert!(a == b, "{epi:?} production extent: old != new");
        assert!(a.iter().any(|&x| x != 0));
        compared += a.len();
    }
    eprintln!("compared {compared} output bytes old vs new");
}
