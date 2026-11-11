// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Wmma::f16` / `Wmma::bf16` (dense 16x16x16 half-precision WMMA, F32 accumulator), the Builder constructor of the
//! typed path next to `Wmma::iu4` / `Wmma::fp8`: a Builder-emitted kernel (kernarg load, per-lane fragment loads, one
//! WMMA, per-lane result store) is assembled natively, lifted unmodified and executed by the PeaceMaker emulator on
//! gfx1151 (V<8> A/B fragments) and gfx1201 (V<4>), and every D element is compared with `peacemaker_ir::emu::mma`
//! (`wmma_f32_f16` / `wmma_f32_bf16`, hardware-qualified on both architectures: `float_edges.md`, Halo `gfx1151-A3-r1`).
//! The fragment placement is the numerics probe's own register-load formula (`numerics_probe.hip`, `wmma_k`), not the
//! emulator's mapping.
use peacemaker_ir::emu::{mma, Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::{
    insn::{Instruction, MemoryClass, Wmma},
    ledger::Counter,
    reg::{Kind, Live, RegRef, V},
    Arch, Builder, KernargLayout, KernelSpec, RegPlan,
};

const KARG: u64 = 0x1_0000;
const BUF_A: u64 = 0x10_0000;
const BUF_B: u64 = 0x20_0000;
const BUF_C: u64 = 0x30_0000;
const BUF_D: u64 = 0x40_0000;
const SYMBOL: &str = "wmma_half_probe";
const SENTINEL: u32 = 0xdead_beef;

fn vr(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: Kind::V,
        base,
        len,
    }
}

/// `D[lane] = WMMA(A[lane], B[lane], C[lane])`: A/B fragments `frag` VGPRs per lane, C/D eight.
fn kernel(arch: Arch, kind: Wmma) -> Result<peacemaker_isa::Emitted, String> {
    let gfx12 = arch.gfx12();
    let mut regs = RegPlan::new(48, 24)?;
    let lane = regs.v::<1>("lane", 0, Live::Whole)?;
    let off_ab = regs.v::<1>("off_ab", 1, Live::Whole)?;
    let off_cd = regs.v::<1>("off_cd", 2, Live::Whole)?;
    let d = regs.v::<8>("d", 8, Live::Whole)?;
    let _a = regs.v::<8>("a", 16, Live::Whole)?;
    let _b = regs.v::<8>("b", 24, Live::Whole)?;
    let c = regs.v::<8>("c", 40, Live::Whole)?;
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
    let mut b = Builder::new(spec, regs);
    b.push(
        Instruction::new("s_load_b256 s[8:15], s[0:1], 0x0", vec![ptrs.reg()], vec![])
            .memory(MemoryClass::SmemLoad),
    )?;
    let ab_shift = if gfx12 { 4 } else { 5 };
    b.push(Instruction::new(
        format!("v_lshlrev_b32_e32 v1, {ab_shift}, v0"),
        vec![off_ab.reg()],
        vec![lane.reg()],
    ))?;
    b.push(Instruction::new(
        "v_lshlrev_b32_e32 v2, 5, v0",
        vec![off_cd.reg()],
        vec![lane.reg()],
    ))?;
    let (km, vmem) = if gfx12 {
        (Counter::Km, Counter::Load)
    } else {
        (Counter::Lgkm, Counter::Vm)
    };
    b.wait(km, 0)?;
    // A at v[16..], B at v[24..] (frag VGPRs each), C at v[40:47].
    let (a_base, b_base, frag) = (16u8, 24u8, if gfx12 { 4u8 } else { 8u8 });
    let load = |b: &mut Builder, dst: RegRef, off: RegRef, sbase: u8, offset: u32| {
        let suffix = if offset == 0 {
            String::new()
        } else {
            format!(" offset:{offset}")
        };
        b.push(
            Instruction::new(
                format!(
                    "global_load_b128 {dst}, {off}, s[{sbase}:{}]{suffix}",
                    sbase + 1
                ),
                vec![dst],
                vec![
                    off,
                    RegRef {
                        kind: Kind::S,
                        base: sbase,
                        len: 2,
                    },
                ],
            )
            .memory(MemoryClass::VmemLoad),
        )
    };
    for i in 0..frag / 4 {
        load(
            &mut b,
            vr(a_base + 4 * i, 4),
            off_ab.reg(),
            8,
            16 * u32::from(i),
        )?;
        load(
            &mut b,
            vr(b_base + 4 * i, 4),
            off_ab.reg(),
            10,
            16 * u32::from(i),
        )?;
    }
    load(&mut b, vr(40, 4), off_cd.reg(), 12, 0)?;
    load(&mut b, vr(44, 4), off_cd.reg(), 12, 16)?;
    b.wait(vmem, 0)?;
    let wmma = if gfx12 {
        Wmma::half(arch, kind, d, V::<4>(a_base), V::<4>(b_base), Some(c))?
    } else {
        Wmma::half(arch, kind, d, V::<8>(a_base), V::<8>(b_base), Some(c))?
    };
    b.push(wmma)?;
    for (i, offset) in [(0u8, 0u32), (1, 16)] {
        let src = vr(8 + 4 * i, 4);
        let suffix = if offset == 0 {
            String::new()
        } else {
            format!(" offset:{offset}")
        };
        b.push(
            Instruction::new(
                format!(
                    "global_store_b128 {}, {src}, s[14:15]{suffix}",
                    off_cd.reg()
                ),
                vec![],
                vec![
                    off_cd.reg(),
                    src,
                    RegRef {
                        kind: Kind::S,
                        base: 14,
                        len: 2,
                    },
                ],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.control(Instruction::new("s_endpgm", vec![], vec![]))?;
    b.finish()
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

/// Random half of the format: a normal in a moderate exponent range (some exact small integers) or zero.
fn half(rng: &mut Rng, bf16: bool) -> u16 {
    let r = rng.next();
    if r % 13 == 0 {
        return 0;
    }
    let sign = ((r >> 8) & 1) as u16;
    if bf16 {
        let e = 0x70 + ((r >> 12) % 24) as u16; // 2^-15 .. 2^8
        sign << 15 | e << 7 | (r >> 20 & 0x7f) as u16
    } else {
        let e = 6 + ((r >> 12) % 18) as u16; // 2^-9 .. 2^8
        sign << 15 | e << 10 | (r >> 20 & 0x3ff) as u16
    }
}

fn model(arch: Arch, bf16: bool, a: &[u16], b: &[u16], c: u32) -> u32 {
    let ir = if arch.gfx12() {
        peacemaker_ir::Arch::Gfx1201
    } else {
        peacemaker_ir::Arch::Gfx1151
    };
    if bf16 {
        mma::wmma_f32_bf16(ir, a, b, c)
    } else {
        mma::wmma_f32_f16(ir, a, b, c)
    }
}

fn run_case(arch: Arch, kind: Wmma, seed: u64) {
    let bf16 = kind == Wmma::Bf16;
    let gfx12 = arch.gfx12();
    let emitted = kernel(arch, kind).expect("builder kernel");
    assert!(
        emitted.s_text.contains(if bf16 {
            "v_wmma_f32_16x16x16_bf16"
        } else {
            "v_wmma_f32_16x16x16_f16"
        }),
        "{}",
        emitted.s_text
    );
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, arch).expect("native assemble");
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
    // Logical matrices in the probe's physical K order: a[r][k] is the k-th physical K slot of row r, bt[c][k] likewise.
    let a: Vec<Vec<u16>> = (0..16)
        .map(|_| (0..16).map(|_| half(&mut rng, bf16)).collect())
        .collect();
    let bt: Vec<Vec<u16>> = (0..16)
        .map(|_| (0..16).map(|_| half(&mut rng, bf16)).collect())
        .collect();
    let c: Vec<Vec<u32>> = (0..16)
        .map(|_| {
            (0..16)
                .map(|_| {
                    let v = (rng.next() % 4000) as f32 / 64.0 - 31.0;
                    v.to_bits()
                })
                .collect()
        })
        .collect();
    // Fragment load: lane l, row/col r = l & 15, half h = l >> 4; VGPR i holds physical K k0, k0 + 1 with
    // k0 = 2i (gfx11, lanes 16..31 replicate) or 8h + 2i (gfx12).
    let (frag, ab_lane_bytes) = if gfx12 { (4usize, 16usize) } else { (8, 32) };
    let mut abuf = vec![0u8; 32 * ab_lane_bytes];
    let mut bbuf = vec![0u8; 32 * ab_lane_bytes];
    let mut cbuf = vec![0u8; 32 * 32];
    for l in 0..32usize {
        let (r, h) = (l & 15, l >> 4);
        for i in 0..frag {
            let k0 = if gfx12 { 8 * h + 2 * i } else { 2 * i };
            for (buf, m) in [(&mut abuf, &a), (&mut bbuf, &bt)] {
                let w = u32::from(m[r][k0]) | u32::from(m[r][k0 + 1]) << 16;
                buf[l * ab_lane_bytes + 4 * i..][..4].copy_from_slice(&w.to_le_bytes());
            }
        }
        for j in 0..8usize {
            let row = if gfx12 { 8 * h + j } else { 2 * j + h };
            cbuf[l * 32 + 4 * j..][..4].copy_from_slice(&c[row][r].to_le_bytes());
        }
    }
    let mut karg = Vec::new();
    for p in [BUF_A, BUF_B, BUF_C, BUF_D] {
        karg.extend(p.to_le_bytes());
    }
    let mut mem = Memory::default();
    mem.map("karg".into(), KARG, karg).unwrap();
    mem.map("a".into(), BUF_A, abuf).unwrap();
    mem.map("b".into(), BUF_B, bbuf).unwrap();
    mem.map("c".into(), BUF_C, cbuf).unwrap();
    mem.map(
        "d".into(),
        BUF_D,
        SENTINEL
            .to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(32 * 32)
            .collect(),
    )
    .unwrap();
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: KARG,
            workgroup: [0, 0, 0],
            block: [32, 1, 1],
            dynamic_lds: 0,
        },
        &mut mem,
    )
    .expect("emulator run");
    let out = mem.read(BUF_D, 32 * 32).unwrap();
    // Logical K order seen by `mma.rs`: gfx12 swaps physical bits 2 and 3, gfx11 is the identity.
    let ord: Vec<usize> = (0..16)
        .map(|k| {
            if gfx12 {
                (k & 3) | ((k & 4) << 1) | ((k & 8) >> 1)
            } else {
                k
            }
        })
        .collect();
    let mut bad = 0;
    for l in 0..32usize {
        let (col, h) = (l & 15, l >> 4);
        for j in 0..8usize {
            let row = if gfx12 { 8 * h + j } else { 2 * j + h };
            let ar: Vec<u16> = ord.iter().map(|&o| a[row][o]).collect();
            let bc: Vec<u16> = ord.iter().map(|&o| bt[col][o]).collect();
            let want = model(arch, bf16, &ar, &bc, c[row][col]);
            let got = u32::from_le_bytes(out[l * 32 + 4 * j..][..4].try_into().unwrap());
            if got != want {
                bad += 1;
                eprintln!(
                    "{arch:?} {kind:?} D[{row}][{col}] emulator {got:#010x} model {want:#010x}"
                );
            }
        }
    }
    assert_eq!(bad, 0, "{arch:?} {kind:?} seed {seed}");
}

#[test]
fn builder_wmma_matches_the_model_on_gfx1151_and_gfx1201() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for kind in [Wmma::F16, Wmma::Bf16] {
            for seed in 1..=8u64 {
                run_case(arch, kind, seed);
            }
        }
    }
}

#[test]
fn fragment_width_and_kind_are_checked() {
    let (d, c) = (V::<8>(8), V::<8>(40));
    // gfx11 wants 8-VGPR fragments, gfx12 4-VGPR ones.
    assert!(Wmma::bf16(Arch::Gfx1151, d, V::<4>(16), V::<4>(24), Some(c)).is_err());
    assert!(Wmma::f16(Arch::Gfx1201, d, V::<8>(16), V::<8>(24), Some(c)).is_err());
    assert!(Wmma::half(Arch::Gfx1151, Wmma::Iu4, d, V::<8>(16), V::<8>(24), None).is_err());
    assert!(Wmma::bf16(Arch::Gfx942, d, V::<8>(16), V::<8>(24), None).is_err());
    let i = Wmma::bf16(Arch::Gfx1151, d, V::<8>(16), V::<8>(24), None).unwrap();
    assert_eq!(
        i.text,
        "v_wmma_f32_16x16x16_bf16 v[8:15], v[16:23], v[24:31], 0"
    );
    let i = Wmma::f16(Arch::Gfx1201, d, V::<4>(16), V::<4>(24), Some(c)).unwrap();
    assert_eq!(
        i.text,
        "v_wmma_f32_16x16x16_f16 v[8:15], v[16:19], v[24:27], v[40:47]"
    );
    assert_eq!(i.defs, vec![d.reg()]);
}

/// The emitted kernels pass the independent wait replay and the M7 hazard/obligation checker (toolchain builds).
#[cfg(feature = "toolchain")]
#[test]
fn builder_wmma_kernels_are_m7_clean() {
    for arch in [Arch::Gfx1151, Arch::Gfx1201] {
        for kind in [Wmma::F16, Wmma::Bf16] {
            let e = kernel(arch, kind).expect("builder kernel");
            peacemaker_isa::ledger_replay::replay_waits(&e.s_text, arch)
                .expect("independent wait replay");
            let elf = peacemaker_isa::native::assemble(&e.s_text, arch).expect("native assemble");
            let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "wmma-half-{}-{kind:?}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("k.co");
            std::fs::write(&path, &elf).unwrap();
            let summary = peacemaker_isa::pm_check::m7(&path, arch.name(), SYMBOL).expect("M7");
            assert_eq!(summary["obligations"], serde_json::json!({}));
            assert_eq!(summary["ambiguous_delays"], 0);
            std::fs::remove_dir_all(&dir).ok();
        }
    }
}
