// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 shared-expert gate/up PM builder, gfx1151 (W8 and W4 entries): the
//! fused F16 GEMM + BF16-recipe SwiGLU + selector sigmoid + F16 hidden twin of
//! the hipcc four-launch chain. Layout, mask and exchange behaviour is checked
//! on the peacemaker CPU interpreter; the hipcc-ISA op correspondence of the
//! activation is pinned by `activation_sequence_matches_the_hipcc_isa`.
use peacemaker_isa::kernels::qwen4_shared_gu_gfx11::{self, Spec};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;

/// `(wr, symbol)` of the two entries. Kernargs: `Wg Wu X Y Sel` (8-byte
/// pointers at 0, 8, 16, 24, 32) and `I K N` (4-byte values at 40, 44, 48).
const ENTRIES: [(u8, &str); 2] = [
    (8, "qwen4_shared_gu_pm_gfx1151_w8"),
    (4, "qwen4_shared_gu_pm_gfx1151_w4"),
];
const ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 4),
    (44, 4),
    (48, 4),
];

#[test]
fn spec_all_is_the_two_pm_entries() {
    let mut got: Vec<_> = Spec::ALL
        .into_iter()
        .map(|wr| Spec { arch: ARCH, wr }.symbol())
        .collect();
    let mut want: Vec<_> = ENTRIES.iter().map(|e| e.1.to_string()).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(Spec::module(ARCH), "qwen4_shared_gu_pm_gfx1151");
}

#[test]
fn entries_are_deterministic_and_target_only_gfx1151() {
    for wr in Spec::ALL {
        let spec = Spec { arch: ARCH, wr };
        let (a, b) = (
            qwen4_shared_gu_gfx11::emit(spec).unwrap(),
            qwen4_shared_gu_gfx11::emit(spec).unwrap(),
        );
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        // Two chunks per 64-K iteration: 32 WMMAs each (2 K-steps x 8 token tiles x gate/up).
        assert_eq!(a.shape.wmma, 64, "{spec:?}");
        assert_eq!(a.shape.barriers, 3, "{spec:?}");
    }
    let (_, ta, pa) = qwen4_shared_gu_gfx11::emit_module(ARCH).unwrap();
    let (_, tb, pb) = qwen4_shared_gu_gfx11::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.module), (tb, pb.module));
    for arch in [Arch::Gfx1201, Arch::Gfx1100] {
        for wr in Spec::ALL {
            assert!(
                qwen4_shared_gu_gfx11::emit(Spec { arch, wr }).is_err(),
                "{arch:?} w{wr}"
            );
        }
        assert!(
            qwen4_shared_gu_gfx11::emit_module(arch).is_err(),
            "{arch:?} module"
        );
    }
    for wr in [0, 2, 6, 16] {
        assert!(
            qwen4_shared_gu_gfx11::emit(Spec { arch: ARCH, wr }).is_err(),
            "wr {wr}"
        );
    }
}

/// The kernarg block: size and every (offset, size) of the metadata.
#[test]
fn kernarg_abi_is_the_documented_one() {
    for (wr, symbol) in ENTRIES {
        let e = qwen4_shared_gu_gfx11::emit(Spec { arch: ARCH, wr }).unwrap();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 52\n"),
            "{symbol}: kernarg size"
        );
        let field = |key: &str| -> Vec<u32> {
            e.s_text
                .lines()
                .filter_map(|l| l.trim().strip_prefix(key))
                .map(|v| v.trim().parse().unwrap())
                .collect()
        };
        assert_eq!(
            field(".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
    }
}

/// The runtime embeds one certified module: it must be exactly what the
/// builder emits today.
#[test]
fn committed_bundles_equal_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = Spec::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = qwen4_shared_gu_gfx11::emit_module(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle"
    );
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-shgu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

/// Committed per-entry shape contract path of a symbol.
fn contract_path(symbol: &str) -> String {
    let entry = symbol.strip_prefix("qwen4_shared_gu_pm_gfx1151_").unwrap();
    format!(
        "{}/kernels/qwen4_shared_gu_pm.gfx1151.{entry}.contract.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

// ------------------------------------------------------------ CPU interpreter

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 31),
        u32::from(h & 1023),
    );
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let k = 31 - m.leading_zeros(); // msb index
            (s << 31) | ((k + 103) << 23) | ((m << (23 - k)) & 0x7f_ffff)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
/// Round-to-nearest-even F32 -> F16 (finite inputs; overflow to Inf).
fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return s | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - ne) as u32;
        let (q, r, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        let q = if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        return s | q as u16;
    }
    let (q, r) = (m >> 13, m & 0x1fff);
    let mut out = ((ne as u32) << 10) | q;
    if r > 0x1000 || (r == 0x1000 && out & 1 == 1) {
        out += 1
    }
    s | out as u16
}
/// hipcc's `silu_mul_bf16_rt`.
fn bf16_rt(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let u = x.to_bits();
    f32::from_bits((u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1)) & 0xffff_0000)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A multiple of 1/16 in [-1, 1): exactly representable in F16 and F32, so the F32 dots are exact in any
    /// order (and small enough that the interpreter's unmeasured `v_div_scale_f32` ranges are never reached).
    fn quarter(&mut self) -> f32 {
        ((self.next() % 32) as i32 - 16) as f32 / 16.0
    }
}

/// Runs the whole grid of `symbol` on the interpreter; returns the F16 hidden bits and the selector.
fn run_grid(
    wr: u8,
    i: usize,
    k: usize,
    n: usize,
    seed: u64,
) -> (Vec<u16>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    let spec = Spec { arch: ARCH, wr };
    let text = qwen4_shared_gu_gfx11::emit(spec).unwrap().s_text;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let lifted = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap();
    let emu = Emulator::new(&lifted.program, &spec.symbol()).unwrap();
    let mut rng = Rng(seed | 1);
    let wg: Vec<f32> = (0..i * k).map(|_| rng.quarter()).collect();
    let wu: Vec<f32> = (0..i * k).map(|_| rng.quarter()).collect();
    let x: Vec<f32> = (0..n * k).map(|_| rng.quarter()).collect();
    let sel: Vec<f32> = (0..n)
        .map(|t| ((t as f32) * 0.37 - 3.0).sin() * 6.0 + (t % 3) as f32 * 0.0078125)
        .collect();
    let h16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&f| f32_to_f16(f).to_le_bytes())
            .collect()
    };
    let (a_kern, a_wg, a_wu, a_x, a_y, a_sel) = (
        0x1_0000u64,
        0x10_0000u64,
        0x80_0000u64,
        0x100_0000u64,
        0x180_0000u64,
        0x200_0000u64,
    );
    let mut kernarg = Vec::new();
    for p in [a_wg, a_wu, a_x, a_y, a_sel] {
        kernarg.extend(p.to_le_bytes());
    }
    for v in [i as u32, k as u32, n as u32] {
        kernarg.extend(v.to_le_bytes());
    }
    assert_eq!(kernarg.len(), 52);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), a_kern, kernarg).unwrap();
    mem.map("wg".into(), a_wg, h16(&wg)).unwrap();
    mem.map("wu".into(), a_wu, h16(&wu)).unwrap();
    mem.map("x".into(), a_x, h16(&x)).unwrap();
    mem.map("y".into(), a_y, vec![0xa5; n * i * 2]).unwrap();
    mem.map(
        "sel".into(),
        a_sel,
        sel.iter().flat_map(|f| f.to_le_bytes()).collect(),
    )
    .unwrap();
    let grid = [
        i.div_ceil(16 * usize::from(wr)) as u32,
        n.div_ceil(128) as u32,
        1,
    ];
    for gy in 0..grid[1] {
        for gx in 0..grid[0] {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: a_kern,
                    workgroup: [gx, gy, 0],
                    block: [32 * u32::from(wr), 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("w{wr} wg({gx},{gy}): {e}"));
        }
    }
    let region = |name: &str| mem.regions.iter().find(|r| r.name == name).unwrap();
    let y = region("y")
        .bytes
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let s = region("sel")
        .bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Reference (exact F32 dots: every operand is a multiple of 1/16 with bounded sums).
    let dot = |w: &[f32], t: usize, c: usize| -> f32 {
        (0..k).map(|kk| w[c * k + kk] * x[t * k + kk]).sum::<f32>()
    };
    let (mut rg, mut ru) = (vec![0f32; n * i], vec![0f32; n * i]);
    for t in 0..n {
        for c in 0..i {
            rg[t * i + c] = dot(&wg, t, c);
            ru[t * i + c] = dot(&wu, t, c);
        }
    }
    (y, s, rg, ru, sel, vec![])
}

fn check_case(wr: u8, i: usize, k: usize, n: usize, seed: u64) {
    let (y, s, rg, ru, sel, _) = run_grid(wr, i, k, n, seed);
    let mut worst = 0f64;
    let mut big = 0usize;
    for t in 0..n {
        for c in 0..i {
            let v = f64::from(bf16_rt(rg[t * i + c]));
            let u = f64::from(bf16_rt(ru[t * i + c]));
            let want = (v / (1.0 + (-v).exp()) * u) as f32;
            let got = f16_to_f32(y[t * i + c]);
            assert!(
                y[t * i + c] != 0xa5a5,
                "w{wr} I={i} K={k} N={n}: hidden[{t}][{c}] not written"
            );
            let err = f64::from((got - want).abs()) / (f64::from(want.abs()) * 0.03 + 2e-3);
            worst = worst.max(err);
            if err > 1.0 {
                big += 1;
            }
        }
    }
    assert_eq!(
        big, 0,
        "w{wr} I={i} K={k} N={n}: {big} hidden values off (worst err/tolerance {worst})"
    );
    for t in 0..n {
        let r = f64::from(bf16_rt(sel[t]));
        let want = (1.0 / (1.0 + (-r).exp())) as f32;
        assert!(
            (s[t] - want).abs() <= want * 2e-6,
            "w{wr} N={n}: selector[{t}] {} vs {want}",
            s[t]
        );
    }
}

/// Channel tail (waves past I), ragged token tail (tokens past N: no store,
/// zeros staged), several K iterations and several workgroups.
#[test]
fn interpreter_matches_the_reference_layout_w8() {
    check_case(8, 48, 128, 20, 1);
    check_case(8, 144, 192, 130, 2);
    // One 64-K iteration (the clamped next-iteration loads re-read the only one), one token.
    check_case(8, 32, 64, 1, 5);
}
#[test]
fn interpreter_matches_the_reference_layout_w4() {
    check_case(4, 80, 128, 33, 3);
    check_case(4, 64, 64, 129, 4);
    check_case(4, 16, 64, 1, 6);
}

/// The activation's op sequence is the hipcc ISA of `shared_expert_activation_bf16_f32`
/// (compiled with the product recipe: `hipcc --genco --offload-arch=gfx1151 -O3 ...`): rounding
/// (`v_bfe_u32 16,1`; `v_cmp_class_f32 0x1f8`; `v_add3_u32 0x7fff`; `v_and_b32 0xffff0000`;
/// `v_cndmask`), the expf expansion, the IEEE divide with `v_div_scale_f32` / `v_rcp_f32` / `v_div_fmas_f32` /
/// `v_div_fixup_f32` (numerator `1.0` for the selector) and the product `v_mul_f32 R(u), q`.
#[test]
fn activation_sequence_matches_the_hipcc_isa() {
    let text = qwen4_shared_gu_gfx11::emit(Spec { arch: ARCH, wr: 8 })
        .unwrap()
        .s_text;
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let count = |needle: &str| lines.iter().filter(|l| l.starts_with(needle)).count();
    // 64 hidden elements per lane: 3 rounds each (g, u, h), 26 SiLU steps; plus the selector's.
    assert_eq!(count("v_cmp_class_f32_e64"), 64 * 3 + 1);
    assert_eq!(count("v_div_scale_f32"), 64 * 2 + 2);
    assert_eq!(count("v_div_fixup_f32"), 64 + 1);
    assert_eq!(count("v_exp_f32"), 64 + 1);
    assert_eq!(count("v_cvt_f16_f32"), 64);
    // The selector's IEEE divide has numerator 1.0 in all three roles (hipcc: `v_div_scale_f32 v5, null, v4, v4, 1.0`,
    // `v_div_scale_f32 v7, vcc_lo, 1.0, v4, 1.0`, `v_div_fixup_f32 v4, v5, v4, 1.0`).
    let one = |m: &str| {
        lines
            .iter()
            .filter(|l| l.starts_with(m) && l.ends_with(", 1.0"))
            .count()
    };
    assert_eq!(
        (one("v_div_scale_f32"), one("v_div_fixup_f32")),
        (2, 1),
        "selector divide numerator 1.0"
    );
    // Product operand order of hipcc's `silu_mul_bf16_rt(up) * (v / (1 + expf(-v)))`: `v_mul_f32 R(u), q`
    // (W8: gate tile 0 element 0 is v6, up tile 0 element 0 v70).
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("v_mul_f32_e32 v6, v70, v")),
        "h = R(u) * q"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.starts_with("v_mul_f32_e32 v6, v") && l.ends_with(", v70")),
        "no q * R(u)"
    );
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// Every symbol of the module certifies: parse-back, independent wait
    /// replay, its committed shape contract (counts, registers, 20480 fixed
    /// and zero dynamic LDS), the all-lane static LDS bound, and M7's lift
    /// identity plus wait/hazard/barrier/window analyses of the linked ELF,
    /// obligation-free.
    #[test]
    fn modules_pass_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = qwen4_shared_gu_gfx11::emit_module(ARCH).unwrap();
        assert_eq!(emitted.len(), ENTRIES.len());
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-shgu-cert-{}-{}",
            ARCH.name(),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for wr in Spec::ALL {
            let symbol = Spec { arch: ARCH, wr }.symbol();
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
            let path = contract_path(&symbol);
            let contract = serde_json::from_slice(
                &std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
            )
            .unwrap();
            certify(
                &toolchain,
                &build,
                &s,
                ARCH.name(),
                &dir.join(format!("{symbol}.manifest.json")),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }

    /// Every entry uses only the static allocation: every DS access of every
    /// lane of every wave ends inside the two 10240-byte X slots, and the
    /// bound is tight (one byte less is refused).
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        for wr in Spec::ALL {
            let e = qwen4_shared_gu_gfx11::emit(Spec { arch: ARCH, wr }).unwrap();
            let symbol = Spec { arch: ARCH, wr }.symbol();
            let (end, hosted) = pm_check::lds_bounds_host(
                &e.s_text,
                &symbol,
                u32::from(wr),
                qwen4_shared_gu_gfx11::GROUP_BYTES,
                &[],
            )
            .unwrap();
            assert!(
                end > qwen4_shared_gu_gfx11::SLOT_BYTES
                    && end <= qwen4_shared_gu_gfx11::GROUP_BYTES
                    && hosted == 0,
                "{symbol}: static end {end}, {hosted} host-bounded accesses"
            );
            assert!(qwen4_shared_gu_gfx11::GROUP_BYTES <= 20480, "{symbol}");
            assert!(
                pm_check::lds_bounds_host(&e.s_text, &symbol, u32::from(wr), end - 1, &[]).is_err(),
                "{symbol}: bound is not tight"
            );
        }
    }
}
