// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Emulator receipts of `peacemaker_isa::libm`: the libm receipt kernels (`kernels::libm_exact`, built only through
//! the shared emitters) are assembled, lifted and executed on the PeaceMaker emulator, and every output word must
//! equal the fresh hipcc object's (`examples/libm_exact_ref.hip`, ROCm product flags) lifted and executed on the same
//! emulator, bit for bit, on a sampled input set per function and architecture (gfx1100, gfx1151, gfx1201): random
//! bit patterns, all IEEE classes, NaN payloads, the Cody-Waite / Payne-Hanek threshold (`|x| = 2^17` +- ulps),
//! huge arguments, RoPE angles, and for `powf` the edge grid and mixed random pairs. (The GPU-exhaustive receipts
//! against the hardware hipcc objects are `examples/libm_exact.rs`.)
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::libm_exact::{emit, symbol, Fun};
use peacemaker_isa::{native, Arch};
use std::process::Command;

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const KARG: u64 = 0x1_0000;
const XB: u64 = 0x100_0000;
const EB: u64 = 0x200_0000;
const YB: u64 = 0x300_0000;
const Y2B: u64 = 0x400_0000;
const BLOCK: u32 = 256;
/// Regular (finite) samples per function: whole workgroups.
const REGULAR: usize = 16384;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn ir_arch(a: Arch) -> peacemaker_ir::Arch {
    match a {
        Arch::Gfx1100 => peacemaker_ir::Arch::Gfx1100,
        Arch::Gfx1151 => peacemaker_ir::Arch::Gfx1151,
        _ => peacemaker_ir::Arch::Gfx1201,
    }
}

/// The fresh hipcc object of `libm_exact_ref.hip` for `arch`, lifted.
fn hipcc_program(arch: Arch) -> peacemaker_ir::Program {
    let hipcc = rocm::hipcc();
    assert!(
        hipcc.is_file(),
        "required ROCm compiler missing: {}",
        hipcc.display()
    );
    let dir =
        std::env::temp_dir().join(format!("libm-exact-{}-{}", arch.name(), std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("ref.co");
    let status = Command::new(hipcc)
        .args([
            "--genco",
            &format!("--offload-arch={}", arch.name()),
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-mcode-object-version=6",
            "-o",
        ])
        .arg(&out)
        .arg(format!(
            "{}/examples/libm_exact_ref.hip",
            env!("CARGO_MANIFEST_DIR")
        ))
        .status()
        .unwrap();
    assert!(status.success(), "hipcc failed");
    let bytes = std::fs::read(&out).unwrap();
    peacemaker_lift::lift_object(
        &bytes,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    )
    .unwrap_or_else(|e| panic!("lift hipcc {}: {e}", arch.name()))
    .program
}

fn pm_program(arch: Arch, fun: Fun, masked: bool) -> peacemaker_ir::Program {
    let e = emit(arch, fun, masked).unwrap();
    let elf = native::assemble(&e.s_text, arch).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// hipcc kernarg image: explicit arguments (natural alignment) then the v5 hidden block count / group size at
/// the next 8-byte boundary.
fn hip_karg(parts: &[(u64, usize)], grid: u32) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    for &(x, size) in parts {
        while v.len() % size != 0 {
            v.push(0);
        }
        v.extend(&x.to_le_bytes()[..size]);
    }
    while v.len() % 8 != 0 {
        v.push(0);
    }
    for x in [grid, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    for x in [BLOCK as u16, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    v.resize(v.len().max(256), 0);
    v
}

fn pm_karg(n: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for p in [YB, Y2B, XB, EB] {
        v.extend(p.to_le_bytes());
    }
    v.extend(n.to_le_bytes());
    v
}

/// Per-workgroup outcome: the 256 `y` / `y2` words, or the emulator's refusal (an operand domain the
/// hardware-measured numerics tables do not cover, e.g. a NaN under an abs/neg source modifier).
static REASONS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

type Wg = Result<(Vec<u32>, Vec<u32>), String>;

fn run(
    program: &peacemaker_ir::Program,
    symbol: &str,
    karg: Vec<u8>,
    xs: &[u32],
    es: Option<&[u32]>,
) -> Vec<Wg> {
    let n = xs.len();
    let emu = Emulator::new(program, symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
    let zeros = vec![0u32; n];
    let es = es.unwrap_or(&zeros);
    let mut out = Vec::new();
    for wg in 0..(n as u32).div_ceil(BLOCK) {
        // One launch per workgroup on fresh memory: a refusal stays local to its 256 elements.
        let (lo, hi) = (wg as usize * 256, ((wg as usize + 1) * 256).min(n));
        let mut mem = Memory::default();
        mem.map("karg".into(), KARG, karg.clone()).unwrap();
        mem.map("x".into(), XB, bytes(xs)).unwrap();
        mem.map("e".into(), EB, bytes(es)).unwrap();
        mem.map("y".into(), YB, vec![0xa5; n * 4]).unwrap();
        mem.map("y2".into(), Y2B, vec![0xa5; n * 4]).unwrap();
        let r = emu.run_workgroup(
            &Launch {
                kernarg_ptr: KARG,
                workgroup: [wg, 0, 0],
                block: [BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        );
        out.push(r.map_err(|e| format!("{e:#}")).map(|()| {
            let words = |name: &str| -> Vec<u32> {
                mem.regions.iter().find(|r| r.name == name).unwrap().bytes[lo * 4..hi * 4]
                    .chunks(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            (words("y"), words("y2"))
        }));
    }
    out
}

/// Compare two per-workgroup runs word for word on the workgroups both executed; returns
/// `(compared_wgs, refused_wgs, mismatching_words)`. `pick` selects the output (`y` or `y2`) of the PM run.
fn compare(a: &[Wg], b: &[Wg], second: bool, what: &str) -> (usize, usize, usize) {
    let (mut compared, mut refused, mut bad) = (0, 0, 0);
    for (i, (ra, rb)) in a.iter().zip(b).enumerate() {
        match (ra, rb) {
            (Ok(ra), Ok(rb)) => {
                compared += 1;
                let (ya, yb) = (&ra.0, if second { &rb.1 } else { &rb.0 });
                for (j, (u, v)) in ya.iter().zip(yb).enumerate() {
                    if u != v {
                        bad += 1;
                        if bad <= 4 {
                            eprintln!("{what}: wg {i} word {j}: {u:#010x} != {v:#010x}");
                        }
                    }
                }
            }
            (ra, rb) => {
                refused += 1;
                for e in [ra, rb].into_iter().filter_map(|r| r.as_ref().err()) {
                    let key: String = e.chars().take(150).collect();
                    let key = key.split(" | arch=").next().unwrap_or(&key).to_string();
                    if !REASONS.lock().unwrap().contains(&format!("{what}: {key}")) {
                        REASONS.lock().unwrap().push(format!("{what}: {key}"));
                    }
                }
            }
        }
    }
    (compared, refused, bad)
}

fn is_special(bits: u32) -> bool {
    bits & 0x7fff_ffff >= 0x7f80_0000
}

/// `list` as workgroups of one repeated value (a refusal then costs only that value).
fn broadcast(list: &[u32]) -> Vec<u32> {
    list.iter()
        .flat_map(|&v| std::iter::repeat(v).take(256))
        .collect()
}

/// Regular samples (finite: zeros, subnormals, normals up to the huge Payne-Hanek range) in whole workgroups,
/// then the NaN / inf values as one broadcast workgroup each.
fn trig_samples() -> Vec<u32> {
    let mut rng = Rng(0x1ba5_e11e_dead_beef);
    let mut v: Vec<u32> = vec![
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x8000_0001,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f80_0000,
        0xbf80_0000,
    ];
    for d in -8i64..=8 {
        for base in [
            0x4800_0000i64,
            0xc800_0000u32 as i64,
            0x3fc9_0fdb,
            0x4049_0fdb,
            0x42b4_0000,
        ] {
            v.push((base + d) as u32);
        }
    }
    for e in (120u32..255).step_by(5) {
        for m in [0u32, 0x1234, 0x7f_ffff, 0x40_0000] {
            v.push(e << 23 | m);
            v.push(0x8000_0000 | e << 23 | m);
        }
    }
    for pos in [1u32, 7, 255, 4095, 32767, 131071, 262143, 1_000_000] {
        for i in 0..8 {
            let f = 1.0f32 / 10000f32.powf((2 * i) as f32 / 64.0);
            v.push((pos as f32 * f).to_bits());
        }
    }
    while v.len() < REGULAR {
        let r = rng.next() as u32;
        if !is_special(r) {
            v.push(r);
        }
    }
    v.truncate(REGULAR);
    v.extend(broadcast(&[
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xffff_ffff,
        0x7fc1_2345,
    ]));
    v
}

fn pow_samples() -> (Vec<u32>, Vec<u32>) {
    let mut rng = Rng(0x0dd5_eed0_f00d_cafe);
    let fin = [
        0x0000_0000u32,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0001,
        0x3f7f_ffff,
        0x4000_0000,
        0xc000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x4040_0000,
        0x4120_0000,
        0x461c_4000,
        0x4b18_9680,
        0x4c18_9680,
        0x2edb_e6ff,
    ];
    let (mut xs, mut es) = (Vec::new(), Vec::new());
    for &x in &fin {
        for &y in &fin {
            xs.push(x);
            es.push(y);
        }
    }
    for base in [1e4f32, 1e6, 5e6, 1e7] {
        for n in [64u32, 128, 256] {
            for i in 0..n / 2 {
                let e = (2 * i) as f32 / n as f32;
                for y in [e, -e] {
                    xs.push(base.to_bits());
                    es.push(y.to_bits());
                }
            }
        }
    }
    while xs.len() < REGULAR {
        let (a, b) = (rng.next(), rng.next());
        let x = f32::from_bits((a as u32 & 0x80ff_ffff) | ((100 + (a >> 40) as u32 % 56) << 23));
        let y = f32::from_bits((b as u32 & 0x80ff_ffff) | ((110 + (b >> 40) as u32 % 24) << 23));
        xs.push(x.to_bits());
        es.push(y.to_bits());
    }
    xs.truncate(REGULAR);
    es.truncate(REGULAR);
    // inf / NaN pairs: one broadcast workgroup each.
    let sp = [
        0x7f80_0000u32,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
    ];
    let mixed = [
        0x3f80_0000u32,
        0x4000_0000,
        0xc040_0000,
        0x0000_0000,
        0x8000_0000,
        0x3f00_0000,
    ];
    for &x in sp.iter().chain(&mixed) {
        for &y in sp.iter().chain(&mixed) {
            if is_special(x) || is_special(y) {
                xs.extend(std::iter::repeat(x).take(256));
                es.extend(std::iter::repeat(y).take(256));
            }
        }
    }
    (xs, es)
}

fn check(arch: Arch) {
    let hip = hipcc_program(arch);
    let _ = ir_arch(arch);
    let xs = trig_samples();
    let n = xs.len();
    let grid = (n as u32).div_ceil(BLOCK);
    let hip_trig = |sym: &str| {
        run(
            &hip,
            sym,
            hip_karg(&[(XB, 8), (YB, 8), (n as u64, 4)], grid),
            &xs,
            None,
        )
    };
    let (ref_sin, ref_cos) = (hip_trig("k_sinf"), hip_trig("k_cosf"));
    let large = xs
        .iter()
        .filter(|&&x| f32::from_bits(x).abs() >= 131072.0 && !is_special(x))
        .count();
    assert!(
        large > 100 && large < REGULAR - 100,
        "regular sample must straddle 2^17 ({large})"
    );
    let regular_wgs = REGULAR / 256;
    let mut report = Vec::new();
    for masked in [false, true] {
        for fun in [Fun::Sinf, Fun::Cosf, Fun::Sincosf] {
            let p = pm_program(arch, fun, masked);
            let pm = run(&p, &symbol(arch, fun, masked), pm_karg(n as u32), &xs, None);
            let mut one = |reference: &[Wg], second: bool, what: &str| {
                // `compare` reads the PM output at `.0`/`.1`: swap so the PM side is always `a`.
                let (c, r, b) = compare(
                    reference,
                    &pm,
                    second,
                    &format!("{arch:?} {what} masked={masked}"),
                );
                report.push(format!(
                    "{arch:?} {what} masked={masked}: compared {c} wg, refused {r}, mismatches {b}"
                ));
                assert_eq!(b, 0, "{arch:?} {what} masked={masked}");
                assert!(
                    c >= regular_wgs,
                    "{arch:?} {what}: only {c} workgroups were modelled by the emulator"
                );
            };
            match fun {
                Fun::Sinf => one(&ref_sin, false, "sinf"),
                Fun::Cosf => one(&ref_cos, false, "cosf"),
                _ => {
                    one(&ref_sin, false, "sincosf.sin");
                    one(&ref_cos, true, "sincosf.cos");
                }
            }
        }
    }
    let (px, pe) = pow_samples();
    let grid = (px.len() as u32).div_ceil(BLOCK);
    let ref_pow = run(
        &hip,
        "k_powf",
        hip_karg(&[(XB, 8), (EB, 8), (YB, 8), (px.len() as u64, 4)], grid),
        &px,
        Some(&pe),
    );
    let p = pm_program(arch, Fun::Powf, false);
    let pm = run(
        &p,
        &symbol(arch, Fun::Powf, false),
        pm_karg(px.len() as u32),
        &px,
        Some(&pe),
    );
    let (c, r, b) = compare(&ref_pow, &pm, false, &format!("{arch:?} powf"));
    report.push(format!(
        "{arch:?} powf: compared {c} wg, refused {r}, mismatches {b}"
    ));
    if arch == Arch::Gfx1100 {
        // The gfx1100 interpreter has no measured v_rcp_f32 / v_exp_f32 table ("unqualified numerical architecture"):
        // it executes no powf at all, from either object. The GPU-exhaustive receipts cover gfx1100 hardware.
        for line in &report {
            eprintln!("{line}");
        }
        assert_eq!(c, 0);
        assert!(REASONS
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.contains("Gfx1100 powf")
                && l.contains("unqualified numerical architecture Gfx1100")));
        return;
    }
    for line in &report {
        eprintln!("{line}");
    }
    for line in REASONS.lock().unwrap().iter() {
        eprintln!("refused: {line}");
    }
    assert_eq!(b, 0, "{arch:?} powf");
    assert!(
        c >= regular_wgs,
        "{arch:?} powf: only {c} workgroups were modelled by the emulator"
    );
}

#[test]
fn gfx1201_emulator_matches_hipcc() {
    let _ = require_rocm_tool!("clang");
    check(Arch::Gfx1201);
}

#[test]
fn gfx1100_emulator_matches_hipcc() {
    let _ = require_rocm_tool!("clang");
    check(Arch::Gfx1100);
}

#[test]
fn gfx1151_emulator_matches_hipcc() {
    let _ = require_rocm_tool!("clang");
    check(Arch::Gfx1151);
}
