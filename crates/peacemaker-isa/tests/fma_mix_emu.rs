// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `FmaMix::insn` (`v_fma_mix_f32`, `v_fma_mixlo_f16`, `v_fma_mixhi_f16`) with typed per-source width / `neg`:
//! Builder-emitted kernels (per-lane loads of `a`, `b`, `c` and the destination's initial word, one `v_fma_mix*`, a store)
//! assembled natively, lifted unmodified and executed by the PeaceMaker emulator on gfx1151 and gfx1201, compared with a
//! host reference (single-rounded fused multiply-add; F16 results by an exact f64 product-sum rounded to nearest even).
//! The sources are F16 halves (and F32 where the product stays exact) in a moderate exponent range, so every
//! intermediate is exactly representable in f64.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::{
    insn::{FmaMix, Instruction, MemoryClass, MixSrc},
    ledger::Counter,
    reg::{Kind, Live, RegRef, V},
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};

const KARG: u64 = 0x1_0000;
const BUFS: [u64; 4] = [0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000]; // a, b, c, dst (initial word, then result)
const SYMBOL: &str = "fma_mix_probe";

type Srcs = (MixSrc, MixSrc, MixSrc);

fn kernel(arch: Arch, op: FmaMix, srcs: Srcs) -> Result<peacemaker_isa::Emitted, String> {
    let gfx12 = arch.gfx12();
    let mut regs = RegPlan::new(16, 24)?;
    let lane4 = regs.v::<1>("lane4", 0, Live::Whole)?;
    let a = regs.v::<1>("a", 1, Live::Whole)?;
    let b = regs.v::<1>("b", 2, Live::Whole)?;
    let c = regs.v::<1>("c", 3, Live::Whole)?;
    let d = regs.v::<1>("d", 4, Live::Whole)?;
    let ptrs = regs.s::<8>("ptrs", 8, Live::Whole)?;
    let spec = KernelSpec {
        kernel_id: SYMBOL.into(),
        variant: "probe".into(),
        arch,
        symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(32)
            .pointer("a", 0)
            .pointer("b", 8)
            .pointer("c", 16)
            .pointer("d", 24),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: 32,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let (km, vmem) = if gfx12 {
        (Counter::Km, Counter::Load)
    } else {
        (Counter::Lgkm, Counter::Vm)
    };
    let sp = |base: u8| RegRef {
        kind: Kind::S,
        base,
        len: 2,
    };
    let mut bld = Builder::new(spec, regs);
    bld.push(
        Instruction::new("s_load_b256 s[8:15], s[0:1], 0x0", vec![ptrs.reg()], vec![])
            .memory(MemoryClass::SmemLoad),
    )?;
    bld.push(Instruction::new(
        "v_lshlrev_b32_e32 v0, 2, v0",
        vec![lane4.reg()],
        vec![lane4.reg()],
    ))?;
    bld.wait(km, 0)?;
    for (dst, base) in [(a, 8u8), (b, 10), (c, 12), (d, 14)] {
        bld.push(
            Instruction::new(
                format!("global_load_b32 {}, v0, s[{base}:{}]", dst.reg(), base + 1),
                vec![dst.reg()],
                vec![lane4.reg(), sp(base)],
            )
            .memory(MemoryClass::VmemLoad),
        )?;
    }
    bld.wait(vmem, 0)?;
    bld.push(op.insn(arch, d, srcs.0, srcs.1, srcs.2)?)?;
    bld.push(
        Instruction::new(
            "global_store_b32 v0, v4, s[14:15]",
            vec![],
            vec![lane4.reg(), d.reg(), sp(14)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    bld.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    bld.finish()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
}

fn f16_to_f64(h: u16) -> f64 {
    let (s, e, m) = (h >> 15, i32::from(h >> 10 & 31), f64::from(h & 1023));
    let v = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

/// Round an f64 to the nearest f16 (ties to even); `None` when the result would be subnormal, infinite or NaN.
fn f64_to_f16_rne(x: f64) -> Option<u16> {
    if x == 0.0 {
        return Some(if x.is_sign_negative() { 0x8000 } else { 0 });
    }
    let sign = u16::from(x < 0.0) << 15;
    let a = x.abs();
    let e = a.log2().floor() as i32;
    if !(-14..=15).contains(&e) {
        return None;
    }
    let scaled = (a * 2f64.powi(10 - e)).round_ties_even();
    let (e, scaled) = if scaled >= 2048.0 {
        (e + 1, 1024.0)
    } else {
        (e, scaled)
    };
    if e > 15 {
        return None;
    }
    Some(sign | ((e + 15) as u16) << 10 | (scaled as u16 & 1023))
}

fn rand_f16(rng: &mut Rng) -> u16 {
    let r = rng.next();
    let e = 12 + (r >> 8) % 7; // 2^-3 .. 2^3
    ((r & 1) << 15 | e << 10 | (r >> 20 & 1023)) as u16
}
fn rand_f32_bits(rng: &mut Rng) -> u32 {
    let r = rng.next();
    let e = 124 + (r >> 8) % 7;
    ((r & 1) << 31 | e << 23 | (r >> 20 & 0x7f_ffff)) as u32
}

/// A source's value as f64, from the 32-bit register word.
fn read(word: u32, s: &MixSrc, zero: bool) -> f64 {
    use peacemaker_isa::insn::MixWidth::*;
    if zero {
        return if s.neg { -0.0 } else { 0.0 };
    }
    let v = match s.width {
        F32 => f64::from(f32::from_bits(word)),
        F16Lo => f16_to_f64(word as u16),
        F16Hi => f16_to_f64((word >> 16) as u16),
    };
    if s.neg {
        -v
    } else {
        v
    }
}

fn run(arch: Arch, op: FmaMix, srcs: Srcs, seed: u64) {
    use peacemaker_isa::insn::{MixValue, MixWidth};
    let elf_text = kernel(arch, op, srcs).expect("builder kernel");
    let elf = peacemaker_isa::native::assemble(&elf_text.s_text, arch)
        .unwrap_or_else(|e| panic!("native assemble: {e}\n{}", elf_text.s_text));
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))
    .unwrap()
    .program;
    let emu = Emulator::new(&program, SYMBOL).expect("emulator");
    let mut rng = Rng(seed | 1);
    let is_zero = |s: &MixSrc| s.value == MixValue::Zero;
    let mut words = [[0u32; 32]; 4];
    for l in 0..32 {
        for (w, s) in srcs_array(srcs).iter().enumerate() {
            words[w][l] = match s.width {
                MixWidth::F32 => rand_f32_bits(&mut rng),
                _ => u32::from(rand_f16(&mut rng)) | u32::from(rand_f16(&mut rng)) << 16,
            };
        }
        words[3][l] = 0xa5a5_5a5a; // destination's initial word: a half-writing op must keep the other half
    }
    let mut karg = Vec::new();
    for p in BUFS {
        karg.extend(p.to_le_bytes());
    }
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg).unwrap();
    for (i, p) in BUFS.iter().enumerate() {
        let bytes: Vec<u8> = words[i].iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.map(format!("b{i}"), *p, bytes).unwrap();
    }
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: KARG,
            workgroup: [0, 0, 0],
            block: [32, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .unwrap_or_else(|e| panic!("{arch:?} {op:?}: {e}"));
    let out = mem.read(BUFS[3], 128).unwrap();
    let ss = srcs_array(srcs);
    let (mut checked, mut bad) = (0, 0);
    for l in 0..32 {
        let v: Vec<f64> = (0..3)
            .map(|i| read(words[i][l], &ss[i], is_zero(&ss[i])))
            .collect();
        let got = u32::from_le_bytes(out[4 * l..4 * l + 4].try_into().unwrap());
        let want = match op {
            FmaMix::F32 => {
                let fused = (v[0] as f32).mul_add(v[1] as f32, v[2] as f32);
                // sources are exact in f32 (F16 widened, F32 as is); mul_add is the single-rounded fma
                fused.to_bits()
            }
            FmaMix::Lo | FmaMix::Hi => {
                let exact = v[0] * v[1] + v[2];
                let Some(h) = f64_to_f16_rne(exact) else {
                    continue;
                };
                if op == FmaMix::Lo {
                    words[3][l] & 0xffff_0000 | u32::from(h)
                } else {
                    words[3][l] & 0xffff | u32::from(h) << 16
                }
            }
        };
        checked += 1;
        if got != want {
            bad += 1;
            eprintln!(
                "{arch:?} {op:?} {srcs:?} lane {l}: emulator {got:#010x} reference {want:#010x}"
            );
        }
    }
    assert!(
        checked >= 24,
        "{arch:?} {op:?}: only {checked} lanes checked"
    );
    assert_eq!(bad, 0, "{arch:?} {op:?} {srcs:?}");
}

fn srcs_array(s: Srcs) -> [MixSrc; 3] {
    [s.0, s.1, s.2]
}

fn v(n: u8) -> V<1> {
    V::<1>(n)
}

#[test]
fn builder_fma_mix_matches_the_reference_on_gfx1151_and_gfx1201() {
    // The three spellings the Flash-Next HC-up v2 epilogue pins (`hyper_read_up_v2.gfx1151.hip`): the lo half of src0
    // times an F32 gate with the -0 addend, the hi half likewise, and plain F32 operands.
    let cases: Vec<(FmaMix, Srcs)> = vec![
        (
            FmaMix::F32,
            (
                MixSrc::f16_lo(v(1)),
                MixSrc::f32(v(2)),
                MixSrc::zero().neg(),
            ),
        ),
        (
            FmaMix::F32,
            (
                MixSrc::f16_hi(v(1)),
                MixSrc::f32(v(2)),
                MixSrc::zero().neg(),
            ),
        ),
        (
            FmaMix::F32,
            (MixSrc::f32(v(1)), MixSrc::f32(v(2)), MixSrc::f32(v(3))),
        ),
        (
            FmaMix::F32,
            (
                MixSrc::f16_lo(v(1)),
                MixSrc::f16_hi(v(2)),
                MixSrc::f32(v(3)),
            ),
        ),
        (
            FmaMix::F32,
            (
                MixSrc::f16_hi(v(1)),
                MixSrc::f16_lo(v(2)),
                MixSrc::f16_hi(v(3)),
            ),
        ),
        (
            FmaMix::F32,
            (
                MixSrc::f32(v(1)).neg(),
                MixSrc::f32(v(2)),
                MixSrc::f32(v(3)),
            ),
        ),
        (
            FmaMix::F32,
            (
                MixSrc::f32(v(1)),
                MixSrc::f32(v(2)).neg(),
                MixSrc::f32(v(3)).neg(),
            ),
        ),
        (
            FmaMix::Lo,
            (
                MixSrc::f16_lo(v(1)),
                MixSrc::f16_hi(v(2)),
                MixSrc::zero().neg(),
            ),
        ),
        (
            FmaMix::Hi,
            (
                MixSrc::f16_lo(v(1)),
                MixSrc::f16_hi(v(2)),
                MixSrc::zero().neg(),
            ),
        ),
        (
            FmaMix::Lo,
            (
                MixSrc::f16_lo(v(1)),
                MixSrc::f16_lo(v(2)),
                MixSrc::f16_hi(v(3)),
            ),
        ),
        (
            FmaMix::Hi,
            (
                MixSrc::f16_hi(v(1)).neg(),
                MixSrc::f16_hi(v(2)),
                MixSrc::f16_lo(v(3)),
            ),
        ),
    ];
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for (op, srcs) in &cases {
            for seed in 1..=3u64 {
                run(arch, *op, *srcs, seed);
            }
        }
    }
}

#[test]
fn fma_mix_spelling_and_operand_checks() {
    let d = v(4);
    // The text matches the spellings of the existing hand-written kernels (op_sel before op_sel_hi, `neg(0)` addend).
    let i = FmaMix::F32
        .insn(
            Arch::Gfx1151,
            d,
            MixSrc::f16_hi(v(1)),
            MixSrc::f32(v(2)),
            MixSrc::zero().neg(),
        )
        .unwrap();
    assert_eq!(
        i.text,
        "v_fma_mix_f32 v4, v1, v2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]"
    );
    assert_eq!(i.defs, vec![d.reg()]);
    assert_eq!(i.uses, vec![v(1).reg(), v(2).reg()]);
    let i = FmaMix::F32
        .insn(
            Arch::Gfx1201,
            d,
            MixSrc::f16_lo(v(4)),
            MixSrc::f32(v(5)),
            MixSrc::f16_lo(v(4)),
        )
        .unwrap();
    assert_eq!(i.text, "v_fma_mix_f32 v4, v4, v5, v4 op_sel_hi:[1,0,1]");
    // lo / hi variants merge into the destination: it is also a use.
    let i = FmaMix::Hi
        .insn(
            Arch::Gfx1151,
            d,
            MixSrc::f32(v(1)).neg(),
            MixSrc::f32(v(2)),
            MixSrc::zero(),
        )
        .unwrap();
    assert_eq!(i.text, "v_fma_mixhi_f16 v4, -v1, v2, 0");
    assert!(i.uses.contains(&d.reg()));
    // The inline zero is an F32 source; CDNA has no v_fma_mix.
    let mut z = MixSrc::zero();
    z.width = peacemaker_isa::insn::MixWidth::F16Lo;
    assert!(FmaMix::F32
        .insn(Arch::Gfx1151, d, z, MixSrc::zero(), MixSrc::zero())
        .is_err());
    assert!(FmaMix::F32
        .insn(
            Arch::Gfx942,
            d,
            MixSrc::f32(v(1)),
            MixSrc::f32(v(2)),
            MixSrc::f32(v(3))
        )
        .is_err());
}
