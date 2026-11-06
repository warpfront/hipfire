// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU byte oracle for the PM prefill dequant exports `hc8_dequant_bf16` / `hc8_dequant_f16` of module
//! `hc8_wmma_pm_gfx1201` (`emit_module` -> `native::assemble` -> `lift_object(Builder)` ->
//! `Emulator::run_workgroup` per workgroup).
//!
//! Oracle (independent of the kernel): the swizzle is spelled out here from the spec formula (and cross-checked against
//! `swizzle_hc8`); the E4M3 value is the closed form `m * 2^-9` / `(8+m) * 2^(e-10)`; the scale multiply is the exact
//! host f32 product; BF16 / F16 rounding is the emulator's bit-level RNE (`f32_to_bf16`, `f32_to_f16`: gradual
//! half subnormals, overflow to inf). Every output byte is compared. The 254 finite codes (signed zeros, subnormals,
//! `0x7e`/`0xfe` max finite; the NaN codes `0x7f`/`0xff` are never produced) cross every row scale
//! (power of two: half-underflow, half-subnormal, and half-overflow edges included) in the 16-row case.
//! Each launch also runs ONE EXTRA workgroup past `ceil(M*K/256)`: its lanes must read and write nothing.
//! Guards around every region are poisoned and checked; inputs must stay unmodified and the written byte set must be
//! exactly the output extent.

use peacemaker_ir::emu::convert::{f32_to_bf16, f32_to_f16};
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::hc8_wmma_gfx1201::{
    emit_dequant, emit_module, swizzle_hc8, DequantOut, DEQUANT_BLOCK, DEQUANT_KERNARG_BYTES,
};
use peacemaker_isa::Arch;

const ARCH: Arch = Arch::Gfx1201;
const GUARD: usize = 128;
const KERNARG: u64 = 0x1_0000;
const QBASE: u64 = 0x2_0000_0000;
const SBASE: u64 = 0x3_0000_0000;
const OBASE: u64 = 0x4_0000_0000;
const SENT: u8 = 0xa5;

/// Row-scale exponents: 2^-24 (f16 min subnormal), -23, -15/-14 (f16 min normal region), -9, -1, 0, 1, 7..10
/// (448 * 2^8 overflows f16), plus far f32 exponents (BF16 stays exact).
const EXPS: [i32; 16] = [
    0, -24, -23, -15, -14, -9, -1, 1, 7, 8, 9, 10, -30, 16, -100, 100,
];

/// Shapes `(M, K)`: multiple tiles in both dims, a non-power-of-two `K/32` (3, 31, 10), and the 16-row cross case.
const SHAPES: [(usize, usize); 7] = [
    (16, 32),
    (32, 64),
    (48, 96),
    (16, 256),
    (160, 320),
    (48, 992),
    (64, 128),
];

fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

fn finite_codes() -> Vec<u8> {
    (0..=255u8).filter(|c| c & 0x7f != 0x7f).collect()
}

/// Exact E4M3 value (closed form).
fn e4m3(code: u8) -> f32 {
    let e = i32::from((code >> 3) & 15);
    let m = u32::from(code & 7);
    let mag = if e == 0 {
        m as f32 * pow2(-9)
    } else {
        (8 + m) as f32 * pow2(e - 10)
    };
    if code & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// Spec swizzle, verbatim.
fn swizzle_spec(w: &[u8], m: usize, k: usize) -> Vec<u8> {
    let mut q = vec![0u8; m * k];
    for r in 0..m {
        for c in 0..k {
            let (mt, rr, kt, kk) = (r >> 4, r & 15, c >> 5, c & 31);
            let (step, rem) = (kk >> 4, kk & 15);
            let h = (rem >> 2) & 1;
            let j = (rem & 3) + 4 * (rem >> 3);
            q[(mt * (k >> 5) + kt) * 512 + (h * 16 + rr) * 16 + step * 8 + j] = w[r * k + c];
        }
    }
    q
}

struct Case {
    m: usize,
    k: usize,
    swz: Vec<u8>,
    scales: Vec<f32>,
    /// Row-major scaled f32 values.
    vals: Vec<f32>,
}

fn case(m: usize, k: usize) -> Case {
    let codes = finite_codes();
    assert_eq!(codes.len(), 254);
    let w: Vec<u8> = (0..m * k)
        .map(|i| {
            let r = i / k;
            codes[(i + 3 * r) % 254]
        })
        .collect();
    let swz = swizzle_spec(&w, m, k);
    assert_eq!(
        swz,
        swizzle_hc8(&w, m, k),
        "spec swizzle vs swizzle_hc8 ({m}x{k})"
    );
    let scales: Vec<f32> = (0..m).map(|r| pow2(EXPS[r % EXPS.len()])).collect();
    let vals = (0..m * k).map(|i| e4m3(w[i]) * scales[i / k]).collect();
    Case {
        m,
        k,
        swz,
        scales,
        vals,
    }
}

fn expected(out: DequantOut, vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| {
            let bits = v.to_bits();
            match out {
                DequantOut::Bf16 => f32_to_bf16(bits),
                DequantOut::F16 => f32_to_f16(bits),
            }
            .to_le_bytes()
        })
        .collect()
}

fn lift_text(text: &str) -> Program {
    let elf = peacemaker_isa::native::assemble(text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn program() -> &'static Program {
    static PROGRAM: std::sync::LazyLock<Program> = std::sync::LazyLock::new(|| {
        let (_, text, _) = emit_module().unwrap_or_else(|e| panic!("emit_module: {e}"));
        lift_text(&text)
    });
    &PROGRAM
}

fn map_guarded(mem: &mut Memory, name: &str, base: u64, data: &[u8], pat: u8) -> u64 {
    let mut bytes = vec![pat; GUARD];
    bytes.extend_from_slice(data);
    bytes.extend(std::iter::repeat(pat).take(GUARD));
    mem.map(name.to_string(), base, bytes).unwrap();
    base + GUARD as u64
}

fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
}

fn run(out: DequantOut, c: &Case, extra_groups: u32) -> Vec<u8> {
    let emu = Emulator::new(program(), out.symbol()).unwrap();
    let total = c.m * c.k;
    let mut mem = Memory::default();
    let qp = map_guarded(&mut mem, "q", QBASE, &c.swz, 0x5a);
    let sb: Vec<u8> = c.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let sp = map_guarded(&mut mem, "s", SBASE, &sb, 0x5a);
    let op = map_guarded(&mut mem, "out", OBASE, &vec![SENT; total * 2], SENT);
    let mut ka = vec![0u8; 64];
    ka[0..8].copy_from_slice(&qp.to_le_bytes());
    ka[8..16].copy_from_slice(&sp.to_le_bytes());
    ka[16..24].copy_from_slice(&op.to_le_bytes());
    ka[24..28].copy_from_slice(&(c.m as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&(c.k as u32).to_le_bytes());
    mem.map("kernarg".into(), KERNARG, ka).unwrap();
    let before: Vec<(&str, Vec<u8>)> = ["q", "s"]
        .iter()
        .map(|n| (*n, region(&mem, n).bytes.clone()))
        .collect();
    let grid = (total as u32).div_ceil(DEQUANT_BLOCK) + extra_groups;
    for x in 0..grid {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KERNARG,
                workgroup: [x, 0, 0],
                block: [DEQUANT_BLOCK, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("{} {}x{} wg {x}: {e}", out.symbol(), c.m, c.k));
    }
    for (n, b) in &before {
        let r = region(&mem, n);
        assert!(
            &r.bytes == b && r.written.count() == 0,
            "{}: input region {n} modified",
            out.symbol()
        );
    }
    let r = region(&mem, "out");
    let end = GUARD + total * 2;
    assert!(
        r.bytes[..GUARD].iter().all(|&b| b == SENT) && r.bytes[end..].iter().all(|&b| b == SENT),
        "{}: out guards overwritten",
        out.symbol()
    );
    assert_eq!(
        r.written.count(),
        total * 2,
        "{}: written byte count",
        out.symbol()
    );
    for b in 0..r.bytes.len() {
        assert_eq!(
            r.written.contains(b),
            (GUARD..end).contains(&b),
            "{}: written set at byte {b}",
            out.symbol()
        );
    }
    r.bytes[GUARD..end].to_vec()
}

fn compare(out: DequantOut, c: &Case, extra: u32) {
    let got = run(out, c, extra);
    let want = expected(out, &c.vals);
    let bad: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        bad.is_empty(),
        "{} {}x{}: {} of {} bytes differ, first at {:?} (elem {}, got {:02x}{:02x} want {:02x}{:02x})",
        out.symbol(),
        c.m,
        c.k,
        bad.len(),
        want.len(),
        bad.first(),
        bad[0] / 2,
        got[bad[0] & !1 | 1],
        got[bad[0] & !1],
        want[bad[0] & !1 | 1],
        want[bad[0] & !1],
    );
}

#[test]
fn bf16_output_every_byte() {
    for (m, k) in SHAPES {
        compare(DequantOut::Bf16, &case(m, k), 1);
    }
}

#[test]
fn f16_output_every_byte() {
    for (m, k) in SHAPES {
        compare(DequantOut::F16, &case(m, k), 1);
    }
}

#[test]
fn oracle_covers_edges() {
    // The 16-row case holds all 254 finite codes in every row under all 16 scales.
    let c = case(16, 256);
    let w_codes: std::collections::BTreeSet<u8> = (0..16 * 256)
        .map(|i| finite_codes()[(i + 3 * (i / 256)) % 254])
        .collect();
    assert_eq!(w_codes.len(), 254);
    let h = expected(DequantOut::F16, &c.vals);
    let halves: Vec<u16> = h
        .chunks(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert!(
        halves.iter().any(|&x| x == 0x7c00 || x == 0xfc00),
        "no f16 overflow to inf"
    );
    assert!(
        halves.iter().any(|&x| x & 0x7c00 == 0 && x & 0x3ff != 0),
        "no f16 subnormal"
    );
    assert!(
        halves.iter().any(|&x| x & 0x7fff == 0),
        "no zero (+/-0 codes, total underflow)"
    );
    assert!(halves.iter().any(|&x| x == 0x8000), "no -0");
    assert!(halves.iter().any(|&x| x == 0x0000), "no +0");
}

#[test]
fn module_exports_and_resources() {
    let (emitted, text, proof) = emit_module().unwrap();
    assert_eq!(emitted.len(), 82);
    assert_eq!(proof.kernels.len(), 82);
    assert!(
        text.contains(".globl hc8_dequant_bf16\n") && text.contains(".globl hc8_dequant_f16\n")
    );
    for out in [DequantOut::Bf16, DequantOut::F16] {
        let e = emit_dequant(out).unwrap();
        let t = &e.s_text;
        let sym = out.symbol();
        assert!(t.contains(&format!(".name: {sym}\n")), "{sym}: name");
        assert!(t.contains(&format!(".kernarg_segment_size: {DEQUANT_KERNARG_BYTES}\n")));
        assert!(t.contains(".amdhsa_wavefront_size32 1"));
        assert!(t.contains(".private_segment_fixed_size: 0\n"));
        assert!(t.contains(".group_segment_fixed_size: 0\n"));
        assert!(t.contains(&format!(".max_flat_workgroup_size: {DEQUANT_BLOCK}\n")));
        let nums = |key: &str| -> Vec<u32> {
            t.lines()
                .filter_map(|l| l.trim().strip_prefix(key))
                .map(|v| v.trim().parse().unwrap())
                .collect()
        };
        assert_eq!(nums(".offset:"), [0, 8, 16, 24, 28], "{sym}: offsets");
        assert_eq!(nums(".size:"), [8, 8, 8, 4, 4], "{sym}: sizes");
        // Declared register counts cover every register the text names.
        let body = t.split(".amdgpu_metadata").next().unwrap();
        let (mut max_s, mut max_v) = (0u32, 0u32);
        for line in body.lines().map(|l| l.split(';').next().unwrap().trim()) {
            if line.starts_with('.') || line.ends_with(':') || line.is_empty() {
                continue;
            }
            let mut rest = line
                .split_whitespace()
                .skip(1)
                .collect::<Vec<_>>()
                .join(" ");
            rest = rest.replace(['[', ']', ','], " ").replace(':', " ");
            for tok in rest.split_whitespace() {
                let tok = tok.split('.').next().unwrap();
                if let Some(n) = tok.strip_prefix('s').and_then(|n| n.parse::<u32>().ok()) {
                    max_s = max_s.max(n + 1);
                } else if let Some(n) = tok.strip_prefix('v').and_then(|n| n.parse::<u32>().ok()) {
                    max_v = max_v.max(n + 1);
                }
            }
        }
        assert!(
            nums(".sgpr_count:")[0] >= max_s,
            "{sym}: sgpr_count {:?} < used {max_s}",
            nums(".sgpr_count:")
        );
        assert!(
            nums(".vgpr_count:")[0] >= max_v,
            "{sym}: vgpr_count {:?} < used {max_v}",
            nums(".vgpr_count:")
        );
        for bad in ["scratch_", "atomic", "flat_", "cmpswap"] {
            assert!(!body.contains(bad), "{sym}: forbidden {bad}");
        }
    }
}
