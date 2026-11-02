// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 exact C8 IU4 producers `iu4_c8_producer_pm_gfx1201` (exports `iu4_c8_rotate_pm_gfx1201` and
//! `iu4_c8_rmsnorm_rotate_pm_gfx1201`): spec / arch / domain admission, determinism, the frozen 72-byte kernarg
//! ABI and resource descriptors, the instruction census and forbidden families, the committed bundle / contracts /
//! M7 receipt against fresh emission, and (feature `toolchain`) the ROCm-oracle certification.
//!
//! The arithmetic is proven on the CPU by running the *emitted, assembled and lifted* object on the peacemaker
//! wave32 interpreter (`peacemaker_ir::emu`, real workgroup barrier, LDS, DPP / permlane, the `v_div_*` chain and
//! `v_rsq_f32`) over full workgroups and comparing every output byte - the f32 rotated side output and all 72
//! bytes of every `block_i4_128` record - with an independent Rust transcription of the incumbent HIP graph
//! (`mq_rotate_x_i4.hip`, `fused_rmsnorm_mq_rotate.hip`, `block_i4_128_quant.hip`, `cand_mode == 8`), written
//! lane by lane from the HIP source (butterflies through per-lane arrays, remap lanes, `shfl_down` / `shfl_xor`
//! semantics). The interpreter is not the hardware: the GPU byte oracle (`iu4_c8_pm_oracle`) remains the final
//! gate.
use peacemaker_ir::emu::{trans, Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::iu4_c8_producer::{self as c8, Export};
use peacemaker_isa::{native, Arch};
use serde_json::Value;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "iu4_c8_producer_pm_gfx1201";
const ROTATE: &str = "iu4_c8_rotate_pm_gfx1201";
const NORM: &str = "iu4_c8_rmsnorm_rotate_pm_gfx1201";
const SIGMOID: &str = "iu4_c8_sigmoid_rotate_pm_gfx1201";
const NORM_OUT: &str = "iu4_c8_rmsnorm_out_rotate_pm_gfx1201";
/// The original two exports: frozen 72-byte ABI and the per-export census assertions written for them.
const LEGACY: [Export; 2] = [Export::Rotate, Export::Norm];
const KERNARG: u32 = 72;
/// `(offset, size, name, value_kind)` in kernarg order.
const ARGS: [(u32, u32, &str, &str); 10] = [
    (0, 8, "x", "global_buffer"),
    (8, 8, "rotated", "global_buffer"),
    (16, 8, "sidecar", "global_buffer"),
    (24, 8, "signs1", "global_buffer"),
    (32, 8, "signs2", "global_buffer"),
    (40, 8, "weight", "global_buffer"),
    (48, 8, "awq", "global_buffer"),
    (56, 4, "rows", "by_value"),
    (60, 4, "k", "by_value"),
    (64, 4, "eps", "by_value"),
];

// ------------------------------------------------------------------------------------------------ text census
fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
/// The instruction lines of an emitted source (directives, labels and metadata excluded).
fn insns(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| {
            let t = l.split(';').next()?.split("//").next()?.trim();
            let tok = t.split_whitespace().next()?;
            let first = tok.chars().next()?;
            (first.is_ascii_lowercase()
                && tok.contains('_')
                && !tok.ends_with(':')
                && !tok.contains('.'))
            .then_some(t)
        })
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(text: &str, name: &str) -> usize {
    insns(text).iter().filter(|l| mnemonic(l) == name).count()
}
fn metadata_args(text: &str) -> Vec<(String, u32, u32, String)> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let (Some(name), Some(off)) = (
            l.strip_prefix(".name:"),
            lines.get(i + 1).and_then(|n| n.strip_prefix(".offset:")),
        ) else {
            continue;
        };
        let size = lines[i + 2]
            .strip_prefix(".size:")
            .expect(".size after .offset");
        let kind = lines[i + 3]
            .strip_prefix(".value_kind:")
            .expect(".value_kind after .size");
        out.push((
            name.trim().to_owned(),
            off.trim().parse().unwrap(),
            size.trim().parse().unwrap(),
            kind.trim().to_owned(),
        ));
    }
    out
}
fn emitted() -> (Vec<peacemaker_isa::plan::Emitted>, String) {
    let (e, text, _) = c8::emit_module(ARCH).unwrap();
    (e, text)
}

// ------------------------------------------------------------------------------------------------ 1. spec / domain
#[test]
fn spec_and_domain_are_exact() {
    for export in c8::module_exports() {
        assert!(c8::emit(ARCH, export).is_ok());
    }
    assert!(c8::emit_module(ARCH).is_ok());
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        for export in c8::module_exports() {
            assert!(c8::emit(arch, export).is_err(), "{arch:?} {export:?}");
        }
        assert!(c8::emit_module(arch).is_err(), "{arch:?}: module");
    }
    for rows in [1usize, 2, 7, 8, 9, 63, 64, 65, 512, 4096, 8191, 8192] {
        for k in [2048usize, 4096] {
            assert!(
                c8::validate(Export::Rotate, k, rows).is_ok(),
                "rotate {k} x {rows}"
            );
        }
        assert!(
            c8::validate(Export::Norm, 2048, rows).is_ok(),
            "norm 2048 x {rows}"
        );
        assert!(
            c8::validate(Export::Norm, 4096, rows).is_err(),
            "norm 4096 x {rows}"
        );
    }
    for rows in [0usize, 8193, 1 << 20, usize::MAX] {
        assert!(
            c8::validate(Export::Rotate, 2048, rows).is_err(),
            "rows {rows}"
        );
        assert!(
            c8::validate(Export::Norm, 2048, rows).is_err(),
            "rows {rows}"
        );
    }
    for k in [0usize, 128, 256, 512, 1024, 2047, 2049, 3072, 6144, 8192] {
        assert!(c8::validate(Export::Rotate, k, 64).is_err(), "rotate k {k}");
        assert!(c8::validate(Export::Norm, k, 64).is_err(), "norm k {k}");
    }
}

// ------------------------------------------------------------------------------------------------ 2. determinism
#[test]
fn emission_is_deterministic_and_the_module_is_the_two_exports() {
    let (ea, ta, ma) = c8::emit_module(ARCH).unwrap();
    let (_, tb, mb) = c8::emit_module(ARCH).unwrap();
    assert_eq!(ta, tb);
    assert_eq!(
        serde_json::to_string(&ma).unwrap(),
        serde_json::to_string(&mb).unwrap()
    );
    assert_eq!((ma.module.as_str(), ma.arch), (MODULE, ARCH));
    assert_eq!((ma.kernels.len(), ea.len()), (4, 4));
    for (export, e) in c8::module_exports().into_iter().zip(&ea) {
        let standalone = c8::emit(ARCH, export).unwrap();
        assert_eq!(
            e.s_text, standalone.s_text,
            "{export:?}: the module entry is the standalone export"
        );
        assert_eq!(e.proof.s_text_sha256, standalone.proof.s_text_sha256);
        assert!(
            ta.contains(&format!(".amdhsa_kernel {}\n", export.symbol()))
                && ta.contains(&format!(".globl {}\n", export.symbol()))
        );
        assert!(ta.contains(&format!(".name: {}\n", export.symbol())));
    }
    assert_eq!(ta.matches(".amdhsa_kernel ").count(), 4);
    assert_eq!(
        ta.matches(".symbol: ").count(),
        4,
        "exactly the four exports"
    );
    assert!(ta.contains("amdgcn-amd-amdhsa--gfx1201"));
    let (bundle_a, bundle_b) = (
        native::bundle(
            &native::assemble(&ta, ARCH).unwrap(),
            ARCH,
            native::DEFAULT_HOST_TARGET,
        ),
        native::bundle(
            &native::assemble(&tb, ARCH).unwrap(),
            ARCH,
            native::DEFAULT_HOST_TARGET,
        ),
    );
    assert!(
        bundle_a == bundle_b,
        "native bundle bytes differ between two emissions"
    );
    assert_eq!(
        (
            c8::KERNEL_ID,
            c8::MODULE,
            c8::ROTATE_SYMBOL,
            c8::NORM_SYMBOL
        ),
        ("iu4_c8_producer_pm", MODULE, ROTATE, NORM)
    );
}

// ------------------------------------------------------------------------------------------------ 3. ABI / resources
#[test]
fn kernarg_layout_and_resources_match_contract() {
    let (entries, module_text) = emitted();
    assert_eq!(
        (
            c8::KERNARG_BYTES,
            c8::ROTATE_BLOCK,
            c8::NORM_BLOCK,
            c8::GROUP,
            c8::MAX_ROWS
        ),
        (KERNARG, 32, 256, 256, 8192)
    );
    for (export, e) in LEGACY.into_iter().zip(&entries) {
        let text = &e.s_text;
        let has = |directive: &str| text.contains(&format!("\t.amdhsa_{directive}\n"));
        let lds = if export == Export::Norm { 32 } else { 0 };
        assert_eq!(field(text, ".kernarg_segment_size:"), [KERNARG]);
        assert!(has(&format!("kernarg_size {KERNARG}")));
        assert_eq!(field(text, ".max_flat_workgroup_size:"), [export.block()]);
        let found = metadata_args(text);
        assert_eq!(found.len(), ARGS.len());
        for ((name, offset, size, kind), (eo, es, en, ek)) in found.iter().zip(ARGS) {
            assert_eq!(
                (name.as_str(), *offset, *size, kind.as_str()),
                (en, eo, es, ek),
                "{export:?} argument {en}"
            );
        }
        assert!(
            has("wavefront_size32 1") && text.contains(".wavefront_size: 32\n"),
            "wave32"
        );
        assert!(has("workgroup_processor_mode 1"), "WGP mode");
        assert!(
            has(&format!("group_segment_fixed_size {lds}"))
                && text.contains(&format!(".group_segment_fixed_size: {lds}\n")),
            "LDS"
        );
        assert!(
            has("private_segment_fixed_size 0")
                && text.contains(".private_segment_fixed_size: 0\n")
        );
        assert!(has("enable_private_segment 0") && has("uses_dynamic_stack 0"));
        assert!(
            text.contains(".sgpr_spill_count: 0\n") && text.contains(".vgpr_spill_count: 0\n"),
            "spills"
        );
        for d in [
            "float_round_mode_32 0",
            "float_round_mode_16_64 0",
            "float_denorm_mode_32 3",
            "float_denorm_mode_16_64 3",
            "fp16_overflow 0",
        ] {
            assert!(has(d), "{d}");
        }
        assert!(
            has("system_sgpr_workgroup_id_x 1")
                && has("system_sgpr_workgroup_id_y 0")
                && has("system_sgpr_workgroup_id_z 0")
        );
        let vgprs = field(text, ".amdhsa_next_free_vgpr");
        assert_eq!(vgprs, [64], "{export:?}: 64 VGPRs");
        assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0]);
        let sgprs = field(text, ".amdhsa_next_free_sgpr");
        assert!(
            sgprs.len() == 1 && sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0]
        );
        assert!(module_text.contains(&format!(".kernarg_segment_size: {KERNARG}\n")));
        // No scratch, flat, atomic, call or legacy waitcnt anywhere; DS and the barrier only in the norm.
        for l in insns(text) {
            let m = mnemonic(l);
            assert!(
                !(m.starts_with("v_wmma")
                    || m.starts_with("scratch_")
                    || m.starts_with("flat_")
                    || m.contains("atomic")
                    || m.starts_with("s_setpc")
                    || m.starts_with("s_swappc")
                    || m.starts_with("s_waitcnt")
                    || m.starts_with("ds_bpermute")
                    || m.starts_with("ds_swizzle")
                    || m.starts_with("v_s_rsq")
                    || m.starts_with("s_add_f32")
                    || m.starts_with("s_mul_f32")
                    || m.starts_with("buffer_load")),
                "forbidden instruction {l}"
            );
            if export == Export::Rotate {
                assert!(
                    !(m.starts_with("ds_") || m.starts_with("s_barrier")),
                    "rotate has no LDS or barrier: {l}"
                );
            }
        }
        assert_eq!(count(text, "s_endpgm"), 1);
        assert_eq!(count(text, "global_store_b32"), 3);
        assert_eq!(count(text, "buffer_store_b128"), 2);
        assert_eq!(
            count(text, "ds_store_b32"),
            usize::from(export == Export::Norm)
        );
        assert_eq!(
            count(text, "ds_load_b128"),
            2 * usize::from(export == Export::Norm)
        );
        assert_eq!(
            count(text, "s_barrier_signal"),
            usize::from(export == Export::Norm)
        );
        assert_eq!(
            count(text, "s_barrier_wait"),
            usize::from(export == Export::Norm)
        );
        assert_eq!(
            count(text, "v_rsq_f32_e32"),
            usize::from(export == Export::Norm),
            "vector rsq, norm only"
        );
    }
}

/// The two folded exports: 80-byte kernarg block (the frozen ten arguments plus one pointer @72), their LDS and
/// the instruction families that make them what they are.
#[test]
fn folded_exports_carry_the_extra_pointer_and_resources() {
    let (entries, _) = emitted();
    assert_eq!(c8::KERNARG_BYTES_EXT, 80);
    for (export, extra, e) in [
        (Export::SigmoidRotate, "gate", &entries[2]),
        (Export::NormOut, "norm_out", &entries[3]),
    ] {
        let text = &e.s_text;
        assert_eq!(field(text, ".kernarg_segment_size:"), [80], "{export:?}");
        let found = metadata_args(text);
        assert_eq!(found.len(), ARGS.len() + 1, "{export:?}");
        for ((name, offset, size, kind), (eo, es, en, ek)) in found.iter().zip(ARGS) {
            assert_eq!(
                (name.as_str(), *offset, *size, kind.as_str()),
                (en, eo, es, ek),
                "{export:?} argument {en}"
            );
        }
        let (name, offset, size, kind) = &found[ARGS.len()];
        assert_eq!(
            (name.as_str(), *offset, *size, kind.as_str()),
            (extra, 72, 8, "global_buffer"),
            "{export:?}"
        );
        assert_eq!(field(text, ".max_flat_workgroup_size:"), [export.block()]);
        let lds = export.lds_bytes();
        assert_eq!(
            field(text, ".group_segment_fixed_size:"),
            [lds],
            "{export:?}"
        );
        assert_eq!(field(text, ".amdhsa_next_free_vgpr"), [64], "{export:?}");
        assert_eq!(count(text, "s_load_b64"), 1);
        assert_eq!(count(text, "s_endpgm"), 1);
        assert_eq!(count(text, "v_s_rsq_f32"), 0);
        match export {
            Export::SigmoidRotate => {
                assert_eq!(count(text, "v_exp_f32_e32"), 8, "one expf per element");
                assert_eq!(count(text, "v_ldexp_f32"), 8);
                assert_eq!(
                    count(text, "global_load_b128"),
                    10,
                    "x, signs1, signs2, awq, gate"
                );
                assert_eq!(count(text, "global_store_b128"), 0);
                assert_eq!(count(text, "s_barrier_signal"), 0);
            }
            _ => {
                assert_eq!(
                    count(text, "global_store_b128"),
                    2,
                    "the normalized f32 row"
                );
                assert_eq!(count(text, "ds_store_b32"), 1);
                assert_eq!(count(text, "ds_load_b128"), 2);
                assert_eq!(count(text, "s_barrier_signal"), 1);
                assert_eq!(count(text, "v_rsq_f32_e32"), 1);
                assert_eq!(lds, 1024, "one 1 KiB partials slot");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 4. incumbent graph facts
#[test]
fn candidate_multipliers_are_the_incumbent_divide_then_add() {
    // The incumbent computes `1.0f + (float)j / 7.0f` per candidate (a full IEEE divide, then a separate add).
    let want: [u32; 8] = core::array::from_fn(|j| (1.0f32 + (j as f32) / 7.0f32).to_bits());
    let got: [u32; 8] = core::array::from_fn(|j| c8::candidate_multiplier(j as u32).to_bits());
    assert_eq!(got, want);
    assert_eq!(got[0], 1.0f32.to_bits());
    assert_eq!(got[7], 2.0f32.to_bits());
    // The emitted text carries exactly these literals, in candidate order, ahead of the base multiply.
    let (entries, _) = emitted();
    for e in &entries {
        let lits: Vec<String> = insns(&e.s_text)
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_mul_f32_e32" && l.contains(", 0x") && l.split(", ").count() == 3
            })
            .filter_map(|l| l.split(", ").nth(1).map(str::to_owned))
            .filter(|lit| want[1..7].iter().any(|w| format!("{w:#x}") == *lit))
            .collect();
        let expect: Vec<String> = want[1..7].iter().map(|w| format!("{w:#x}")).collect();
        assert_eq!(lits, expect, "candidate multiplier literals j = 1..6 in order (j = 0 and 7 are the inline 1.0 / 2.0)");
        let named: Vec<&str> = insns(&e.s_text)
            .iter()
            .filter(|l| {
                mnemonic(l) == "v_mul_f32_e32" && (l.contains(", 1.0, ") || l.contains(", 2.0, "))
            })
            .map(|l| l.split(", ").nth(1).unwrap())
            .collect();
        assert_eq!(named, ["1.0", "2.0"], "j = 0 and j = 7 multipliers");
    }
}

#[test]
fn sign_products_fuse_without_a_rounding_difference_for_unit_signs() {
    // HIP contracts `x * sign` into the butterfly FMA; the emitted code multiplies then adds. Both are one rounding of
    // the same exact value because the product with +-1 is exact (signed zeros included).
    let mut rng = Rng(0x5eed_1234);
    let signs = [1.0f32, -1.0];
    let mut cases = vec![
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        -f32::MAX,
    ];
    cases.extend((0..4000).map(|_| rng.f32(-100, 100)));
    for &a in &cases {
        for &b in &cases[..64] {
            for &sa in &signs {
                for &sb in &signs {
                    let (p, q) = (b * sb, a * sa);
                    let (add_fused, sub_fused) = (a.mul_add(sa, p), a.mul_add(sa, -p));
                    assert_eq!((q + p).to_bits(), add_fused.to_bits());
                    assert_eq!((q - p).to_bits(), sub_fused.to_bits());
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 5. interpreter identity
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Finite f32 with a random sign, mantissa and exponent in `2^lo ..= 2^hi`.
    fn f32(&mut self, lo: i32, hi: i32) -> f32 {
        let r = self.next();
        let exp = (127 + lo) as u64 + (r >> 40) % ((hi - lo + 1) as u64);
        f32::from_bits(
            ((r as u32) & 0x8000_0000) | ((exp as u32) << 23) | ((r >> 8) as u32 & 0x7f_ffff),
        )
    }
    fn pos(&mut self, lo: i32, hi: i32) -> f32 {
        self.f32(lo, hi).abs()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.max(b)
    }
}
/// `fminf(fmaxf(q, -8), 7)`.
fn clamp_q(q: f32) -> f32 {
    fmax(q, -8.0).min(7.0)
}

struct Block {
    d: f32,
    s: i32,
    qs: [u8; 64],
    /// The winning candidate (`None`: the sentinel / zero-block `best_d = 1`).
    winner: Option<u32>,
}
/// Deliberate deviations from the incumbent recipe, used only to show the test data discriminate them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Incumbent,
    /// The `1e30` sentinel replaced by infinity.
    NoSentinel,
}
/// `quantize_block_i4_128_wave<false>` over the 32 lanes (lane `l` owns `4l..4l+3`), every reduction as its
/// `__shfl_xor` butterfly.
fn quantize_ref(x: &[f32]) -> Block {
    quantize_mode(x, Mode::Incumbent)
}
fn quantize_mode(x: &[f32], mode: Mode) -> Block {
    assert_eq!(x.len(), 128);
    let amax = x.iter().fold(0.0f32, |m, v| fmax(m, v.abs()));
    let quotient = |xv: f32, d: f32| xv / d;
    let mut best_d = 1.0f32;
    let mut winner = None;
    if amax != 0.0 {
        let mut best_mse = if mode == Mode::NoSentinel {
            f32::INFINITY
        } else {
            1.0e30f32
        };
        for step in 0..8u32 {
            let j = step;
            let dj = (amax / 7.0f32) * 0.5f32 * (1.0f32 + (j as f32) / 7.0f32);
            let mut m = [0f32; 32];
            for (l, lane) in m.iter_mut().enumerate() {
                let mut mse = 0.0f32;
                for e in 0..4 {
                    let xv = x[4 * l + e];
                    let q = clamp_q(quotient(xv, dj).round_ties_even());
                    let err = (-q).mul_add(dj, xv);
                    mse = err.mul_add(err, mse);
                }
                *lane = mse;
            }
            for off in [16usize, 8, 4, 2, 1] {
                let prev = m;
                for l in 0..32 {
                    m[l] = prev[l] + prev[l ^ off];
                }
            }
            assert!(
                m.iter()
                    .all(|v| v.to_bits() == m[0].to_bits() || (v.is_nan() && m[0].is_nan())),
                "butterfly leaves lanes unequal"
            );
            if m[0] < best_mse {
                best_mse = m[0];
                best_d = dj;
                winner = Some(j);
            }
        }
    }
    let codes: Vec<i32> = x
        .iter()
        .map(|&xv| {
            clamp_q(if amax == 0.0 {
                0.0
            } else {
                quotient(xv, best_d).round_ties_even()
            }) as i32
        })
        .collect();
    let mut qs = [0u8; 64];
    for l in 0..32 {
        let c = |e: usize| (codes[4 * l + e] & 15) as u8;
        qs[2 * l] = c(0) | (c(1) << 4);
        qs[2 * l + 1] = c(2) | (c(3) << 4);
    }
    Block {
        d: best_d,
        s: codes.iter().sum(),
        qs,
        winner,
    }
}

/// The FWHT body shared by both HIP kernels: lane `p` holds `v[p][0..8]`; returns `f[8p + i]`.
fn fwht_ref(v: &mut [[f32; 8]; 32], signs2: &[f32]) -> Vec<f32> {
    for lane in v.iter_mut() {
        for stride in [1usize, 2, 4] {
            for lo in (0..8).filter(|i| i & stride == 0) {
                let (a, b) = (lane[lo], lane[lo + stride]);
                lane[lo] = a + b;
                lane[lo + stride] = a - b;
            }
        }
    }
    for stride in [1usize, 2, 4, 8, 16] {
        let prev = *v;
        for p in 0..32 {
            for i in 0..8 {
                let partner = prev[p ^ stride][i];
                v[p][i] = if p & stride != 0 {
                    partner - prev[p][i]
                } else {
                    prev[p][i] + partner
                };
            }
        }
    }
    let mut f = vec![0f32; 256];
    for p in 0..32 {
        for i in 0..8 {
            f[8 * p + i] = v[p][i] * 0.0625f32 * signs2[8 * p + i];
        }
    }
    f
}

#[derive(Clone)]
struct Inputs {
    rows: usize,
    k: usize,
    x: Vec<f32>,
    weight: Vec<f32>,
    awq: Option<Vec<f32>>,
    signs1: Vec<f32>,
    signs2: Vec<f32>,
    eps: f32,
}

/// Incumbent `mq_rotate_x_i4` (cand_mode 8): per row and group the rotated f32 and the two records.
fn reference_rotate(i: &Inputs) -> (Vec<f32>, Vec<Block>) {
    let groups = i.k / 256;
    let mut rot = vec![0f32; i.rows * i.k];
    let mut blocks: Vec<Option<Block>> = (0..2 * groups * i.rows).map(|_| None).collect();
    for row in 0..i.rows {
        for g in 0..groups {
            let mut v = [[0f32; 8]; 32];
            for (p, lane) in v.iter_mut().enumerate() {
                for (e, slot) in lane.iter_mut().enumerate() {
                    let base = g * 256 + 8 * p + e;
                    let x = i.x[row * i.k + base];
                    let x = match &i.awq {
                        Some(a) => x / a[base],
                        None => x,
                    };
                    *slot = x * i.signs1[8 * p + e];
                }
            }
            finish(i, row, g, &mut v, &mut rot, &mut blocks);
        }
    }
    (rot, blocks.into_iter().map(Option::unwrap).collect())
}

fn finish(
    i: &Inputs,
    row: usize,
    g: usize,
    v: &mut [[f32; 8]; 32],
    rot: &mut [f32],
    blocks: &mut [Option<Block>],
) {
    let f = fwht_ref(v, &i.signs2);
    for (e, &val) in f.iter().enumerate() {
        rot[row * i.k + g * 256 + e] = val;
    }
    for h in 0..2 {
        blocks[(2 * g + h) * i.rows + row] = Some(quantize_ref(&f[128 * h..128 * h + 128]));
    }
}

/// `shfl_down` of a 32-lane array by `off` (an out-of-range source returns the lane's own value).
fn shfl_down(v: &[f32; 32], off: usize) -> [f32; 32] {
    core::array::from_fn(|l| if l + off < 32 { v[l + off] } else { v[l] })
}

/// Incumbent `fused_rmsnorm_mq_rotate_i4` (cand_mode 8, block 256).
fn reference_norm(i: &Inputs) -> (Vec<f32>, Vec<Block>) {
    assert_eq!(i.k, 2048);
    let groups = 8;
    let mut rot = vec![0f32; i.rows * i.k];
    let mut blocks: Vec<Option<Block>> = (0..2 * groups * i.rows).map(|_| None).collect();
    for row in 0..i.rows {
        let xr = &i.x[row * i.k..(row + 1) * i.k];
        let local: Vec<f32> = (0..256)
            .map(|tid| {
                let mut s = 0.0f32;
                for idx in (tid..i.k).step_by(256) {
                    s = xr[idx].mul_add(xr[idx], s);
                }
                s
            })
            .collect();
        let mut reduce = [0f32; 8];
        for (w, r) in reduce.iter_mut().enumerate() {
            let mut val: [f32; 32] = core::array::from_fn(|l| local[32 * w + l]);
            for off in [16usize, 8, 4, 2, 1] {
                let down = shfl_down(&val, off);
                for l in 0..32 {
                    val[l] += down[l];
                }
            }
            *r = val[0];
        }
        let mut t: [f32; 32] = core::array::from_fn(|l| if l < 8 { reduce[l] } else { 0.0 });
        for off in [4usize, 2, 1] {
            let down = shfl_down(&t, off);
            for l in 0..32 {
                t[l] += down[l];
            }
        }
        let x = t[0] / (i.k as f32) + i.eps;
        let small = x < f32::from_bits(0x0080_0000);
        let xs = if small {
            x * f32::from_bits(0x4b80_0000)
        } else {
            x
        };
        let mut rms = f32::from_bits(trans::rsq_f32(
            peacemaker_ir::inst::Arch::Gfx1201,
            xs.to_bits(),
        ));
        if small {
            rms *= f32::from_bits(0x4580_0000);
        }
        for g in 0..groups {
            let mut v = [[0f32; 8]; 32];
            for (p, lane) in v.iter_mut().enumerate() {
                for (e, slot) in lane.iter_mut().enumerate() {
                    let base = g * 256 + 8 * p + e;
                    let y = xr[base] * i.weight[base] * rms;
                    let y = match &i.awq {
                        Some(a) => y / a[base],
                        None => y,
                    };
                    *slot = y * i.signs1[8 * p + e];
                }
            }
            finish(i, row, g, &mut v, &mut rot, &mut blocks);
        }
    }
    (rot, blocks.into_iter().map(Option::unwrap).collect())
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
}

const KARG: u64 = 0x1_0000;
const XB: u64 = 0x100_0000;
const ROTB: u64 = 0x200_0000;
const SCB: u64 = 0x300_0000;
const S1B: u64 = 0x400_0000;
const S2B: u64 = 0x401_0000;
const WB: u64 = 0x402_0000;
const AB: u64 = 0x403_0000;
const SENTINEL: u8 = 0xa5;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Launch `export` over every workgroup of `i`, returning (rotated bits, sidecar bytes) and asserting that every
/// byte of both outputs was written exactly the contracted way (no stray writes, full coverage).
fn run(rig: &Rig, export: Export, i: &Inputs) -> (Vec<u32>, Vec<u8>) {
    let emu = Emulator::new(&rig.program, export.symbol()).unwrap();
    let mut mem = Memory::default();
    let mut karg = Vec::new();
    for ptr in [
        XB,
        ROTB,
        SCB,
        S1B,
        S2B,
        if export == Export::Norm { WB } else { 0 },
        if i.awq.is_some() { AB } else { 0 },
    ] {
        karg.extend(ptr.to_le_bytes());
    }
    karg.extend((i.rows as i32).to_le_bytes());
    karg.extend((i.k as i32).to_le_bytes());
    karg.extend(i.eps.to_le_bytes());
    karg.extend([0u8; 4]);
    assert_eq!(karg.len(), 72);
    let sidecar_bytes = (i.k / 128) * i.rows * 72;
    mem.map("karg".into(), KARG, karg).unwrap();
    mem.map("x".into(), XB, f32_bytes(&i.x)).unwrap();
    mem.map("rot".into(), ROTB, vec![SENTINEL; i.rows * i.k * 4])
        .unwrap();
    mem.map("sc".into(), SCB, vec![SENTINEL; sidecar_bytes])
        .unwrap();
    mem.map("s1".into(), S1B, f32_bytes(&i.signs1)).unwrap();
    mem.map("s2".into(), S2B, f32_bytes(&i.signs2)).unwrap();
    if export == Export::Norm {
        mem.map("w".into(), WB, f32_bytes(&i.weight)).unwrap();
    }
    if let Some(a) = &i.awq {
        mem.map("awq".into(), AB, f32_bytes(a)).unwrap();
    }
    let (grid, block) = match export {
        Export::Rotate => (i.rows * i.k / 256, 32),
        Export::Norm => (i.rows, 256),
        Export::SigmoidRotate | Export::NormOut => {
            unreachable!("the folded exports are proven by the GPU byte oracle (iu4_c8_pm_oracle)")
        }
    };
    for wg in 0..grid {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: KARG,
                workgroup: [wg as u32, 0, 0],
                block: [block, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("{export:?} wg {wg}: {e}"));
    }
    let region = |n: &str| mem.regions.iter().find(|r| r.name == n).unwrap();
    let (rot, sc) = (region("rot"), region("sc"));
    assert_eq!(
        rot.written.count(),
        rot.bytes.len(),
        "rotated output not fully written"
    );
    assert_eq!(
        sc.written.count(),
        sc.bytes.len(),
        "sidecar not fully written"
    );
    for n in ["x", "s1", "s2", "karg"] {
        assert_eq!(region(n).written.count(), 0, "{n} was written");
    }
    let bits = rot
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    (bits, sc.bytes.clone())
}

fn check(tag: &str, got: &(Vec<u32>, Vec<u8>), want: &(Vec<f32>, Vec<Block>), i: &Inputs) {
    for (idx, (&g, w)) in got.0.iter().zip(&want.0).enumerate() {
        assert_eq!(
            g,
            w.to_bits(),
            "{tag}: rotated[{idx}] (row {}, col {}) = {g:#010x}, reference {:#010x}",
            idx / i.k,
            idx % i.k,
            w.to_bits()
        );
    }
    for (rec, b) in want.1.iter().enumerate() {
        let bytes = &got.1[rec * 72..rec * 72 + 72];
        let (d, s) = (
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            i32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        );
        let (half, row) = (rec / i.rows, rec % i.rows);
        assert_eq!(
            d,
            b.d.to_bits(),
            "{tag}: record {rec} (row {row}, block {half}) scale"
        );
        assert_eq!(
            s, b.s,
            "{tag}: record {rec} (row {row}, block {half}) code sum"
        );
        assert_eq!(
            &bytes[8..],
            &b.qs[..],
            "{tag}: record {rec} (row {row}, block {half}) codes"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Zero,
    Random,
    Large,
    Huge,
    /// Magnitudes whose mean square (eps 0) falls below the smallest normal: the norm's rsqrt denormal guard.
    Tiny,
}

fn signs(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn inputs(rng: &mut Rng, rows: usize, k: usize, kind: Kind, awq: bool, export: Export) -> Inputs {
    let n = rows * k;
    let x: Vec<f32> = match kind {
        Kind::Zero => vec![0.0; n],
        Kind::Random => (0..n).map(|_| rng.f32(-3, 3)).collect(),
        // Per-128-block dynamic range 2^-16 .. 2^16 with exact-zero and single-outlier blocks mixed in.
        Kind::Large => {
            let mut x = Vec::with_capacity(n);
            for _ in 0..n / 128 {
                let scale = rng.f32(-16, 16).abs().max(1.0)
                    * if rng.next() & 1 == 0 {
                        1.0
                    } else {
                        1.0 / 65536.0
                    };
                match rng.below(8) {
                    0 => x.extend(std::iter::repeat_n(0.0f32, 128)),
                    1 => {
                        let spike = rng.below(128);
                        x.extend((0..128).map(|j| if j == spike { 1.0e6 } else { rng.f32(-8, 0) }));
                    }
                    _ => x.extend((0..128).map(|_| rng.f32(-4, 4) * scale)),
                }
            }
            x
        }
        // Magnitudes large enough that the C8 mse reaches the 1e30 sentinel in some blocks.
        Kind::Huge => (0..n).map(|_| rng.f32(46, 52)).collect(),
        Kind::Tiny => (0..n).map(|_| rng.f32(-78, -72)).collect(),
    };
    // Norm weights: unit-ish for most kinds, huge for Huge so the normalised values stay at the sentinel scale.
    let weight = (0..k)
        .map(|_| {
            if matches!(kind, Kind::Huge) {
                rng.pos(44, 50)
            } else {
                rng.f32(-2, 2)
            }
        })
        .collect();
    let awq = awq.then(|| (0..k).map(|_| rng.pos(-3, 3)).collect());
    let _ = export;
    let eps = if matches!(kind, Kind::Tiny) {
        0.0
    } else {
        1.0e-6
    };
    Inputs {
        rows,
        k,
        x,
        weight,
        awq,
        signs1: signs(rng, 256),
        signs2: signs(rng, 256),
        eps,
    }
}

fn identity_cases(export: Export, ks: &[usize], rows: &[usize], seeds: u64) {
    let rig = Rig::new();
    for &k in ks {
        for &r in rows {
            for kind in [Kind::Zero, Kind::Random, Kind::Large, Kind::Huge] {
                for awq in [false, true] {
                    // Norm of the huge kind needs weights and eps that keep the mean finite: skip the AWQ twin of it.
                    for seed in 0..seeds {
                        let mut rng = Rng(0x9e37_79b9_7f4a_7c15
                            ^ (k as u64) << 20
                            ^ (r as u64) << 8
                            ^ seed
                            ^ (awq as u64) << 40
                            ^ (kind as u64) << 44);
                        let i = inputs(&mut rng, r, k, kind, awq, export);
                        let want = match export {
                            Export::Rotate => reference_rotate(&i),
                            Export::Norm => reference_norm(&i),
                            Export::SigmoidRotate | Export::NormOut => unreachable!(),
                        };
                        let got = run(&rig, export, &i);
                        check(
                            &format!("{export:?} k={k} rows={r} {kind:?} awq={awq} seed={seed}"),
                            &got,
                            &want,
                            &i,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn interpreter_rotate_k2048_is_byte_identical_to_the_incumbent_graph() {
    identity_cases(Export::Rotate, &[2048], &[1, 3], 2);
}
#[test]
fn interpreter_rotate_k4096_is_byte_identical_to_the_incumbent_graph() {
    identity_cases(Export::Rotate, &[4096], &[1, 2], 2);
}
#[test]
fn interpreter_norm_k2048_is_byte_identical_to_the_incumbent_graph() {
    identity_cases(Export::Norm, &[2048], &[1, 3], 2);
}
/// `eps = 0` and a denormal mean: `x < 2^-126` takes the incumbent's `x * 2^24` / `* 2^12` rsqrt rescaling.
#[test]
fn interpreter_norm_denormal_mean_takes_the_rsqrt_guard() {
    let rig = Rig::new();
    for awq in [false, true] {
        for seed in 0..3u64 {
            let mut rng = Rng(0x7149_0000 + seed + (awq as u64) * 100);
            let i = inputs(&mut rng, 2, 2048, Kind::Tiny, awq, Export::Norm);
            let sum: f32 = i.x[..2048].iter().fold(0.0, |s, &v| v.mul_add(v, s));
            assert!(
                sum / 2048.0 < f32::from_bits(0x0080_0000),
                "case must take the denormal guard"
            );
            let want = reference_norm(&i);
            let got = run(&rig, Export::Norm, &i);
            check(&format!("Norm tiny awq={awq} seed={seed}"), &got, &want, &i);
        }
    }
}

/// Launches outside the contract exit before touching anything: wrong K, rows 0 / 8193 and a workgroup past the
/// grid never write the sidecar or the rotated output.
#[test]
fn interpreter_unsupported_launches_write_nothing() {
    let rig = Rig::new();
    let mut rng = Rng(0xfeed);
    for (export, k, rows) in [
        (Export::Rotate, 3072usize, 2usize),
        (Export::Rotate, 2048, 0),
        (Export::Rotate, 2048, 8193),
        (Export::Norm, 4096, 2),
        (Export::Norm, 2048, 0),
        (Export::Norm, 2048, 8193),
    ] {
        let emu = Emulator::new(&rig.program, export.symbol()).unwrap();
        let base = inputs(&mut rng, 2, 2048, Kind::Random, false, export);
        let mut mem = Memory::default();
        let mut karg = Vec::new();
        for ptr in [XB, ROTB, SCB, S1B, S2B, WB, 0u64] {
            karg.extend(ptr.to_le_bytes());
        }
        karg.extend((rows as i32).to_le_bytes());
        karg.extend((k as i32).to_le_bytes());
        karg.extend(1.0e-6f32.to_le_bytes());
        karg.extend([0u8; 4]);
        mem.map("karg".into(), KARG, karg).unwrap();
        mem.map("x".into(), XB, f32_bytes(&base.x)).unwrap();
        mem.map("rot".into(), ROTB, vec![SENTINEL; 64]).unwrap();
        mem.map("sc".into(), SCB, vec![SENTINEL; 64]).unwrap();
        mem.map("s1".into(), S1B, f32_bytes(&base.signs1)).unwrap();
        mem.map("s2".into(), S2B, f32_bytes(&base.signs2)).unwrap();
        mem.map("w".into(), WB, f32_bytes(&base.weight)).unwrap();
        let block = export.block();
        for wg in [0u32, 1, 5] {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [wg, 0, 0],
                    block: [block, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("{export:?} k={k} rows={rows} wg {wg}: {e}"));
        }
        for r in &mem.regions {
            assert_eq!(
                r.written.count(),
                0,
                "{export:?} k={k} rows={rows}: {} written",
                r.name
            );
        }
    }
}

/// The identity cases above are only evidence if their data reach the search's corners and discriminate the
/// arithmetic variants a wrong emitter could silently implement: several winning candidates occur, the sentinel /
/// zero fallbacks occur, and the missing sentinel changes some record. (A reciprocal quotient or an unfused mse
/// chain only shows in near ties of `x / d` or of two candidates' mse, which random data practically never
/// reach: those are pinned by the instruction census in `contracts_are_derived_from_the_emission` - every
/// `v_rcp_f32` belongs to a full `v_div_*` expansion and the mse chains are `v_mul` + three `v_fmac`.)
#[test]
fn identity_data_reach_every_corner_and_discriminate_the_variants() {
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for export in [Export::Rotate, Export::Norm] {
        for kind in [Kind::Zero, Kind::Random, Kind::Large, Kind::Huge] {
            for awq in [false, true] {
                for seed in 0..2u64 {
                    let (k, r) = (2048usize, 3usize);
                    let mut rng = Rng(0x9e37_79b9_7f4a_7c15
                        ^ (k as u64) << 20
                        ^ (r as u64) << 8
                        ^ seed
                        ^ (awq as u64) << 40
                        ^ (kind as u64) << 44);
                    let i = inputs(&mut rng, r, k, kind, awq, export);
                    // The quantizer input is the rotated output: re-run the reference rotate to collect it.
                    let (rot, _) = if export == Export::Rotate {
                        reference_rotate(&i)
                    } else {
                        reference_norm(&i)
                    };
                    blocks.extend(rot.as_chunks::<128>().0.iter().map(|c| c.to_vec()));
                }
            }
        }
    }
    let base: Vec<Block> = blocks.iter().map(|b| quantize_ref(b)).collect();
    let mut winners = [0usize; 8];
    for b in &base {
        if let Some(j) = b.winner {
            winners[j as usize] += 1;
        }
    }
    let zero_blocks = blocks
        .iter()
        .filter(|b| b.iter().all(|v| *v == 0.0))
        .count();
    let sentinel_blocks = base
        .iter()
        .zip(&blocks)
        .filter(|(b, x)| b.winner.is_none() && x.iter().any(|v| *v != 0.0))
        .count();
    eprintln!(
        "winners {winners:?} zero_blocks {zero_blocks} sentinel_blocks {sentinel_blocks} of {}",
        blocks.len()
    );
    assert!(
        winners.iter().filter(|&&n| n > 0).count() >= 5,
        "winner histogram {winners:?}"
    );
    assert!(zero_blocks > 0 && sentinel_blocks > 0);
    let (name, mode) = ("no sentinel", Mode::NoSentinel);
    let differs = blocks.iter().zip(&base).any(|(x, b)| {
        let v = quantize_mode(x, mode);
        v.d.to_bits() != b.d.to_bits() || v.s != b.s || v.qs != b.qs
    });
    assert!(
        differs,
        "the data do not discriminate the deviation `{name}` from the incumbent recipe"
    );
}

// ------------------------------------------------------------------------------------------------ 6. contracts
#[derive(serde::Deserialize)]
struct Probe {
    forbidden: Vec<String>,
}
fn census(text: &str, pattern: &str) -> usize {
    insns(text)
        .iter()
        .filter(|l| {
            pattern
                .strip_suffix('*')
                .map_or(mnemonic(l) == pattern, |pre| mnemonic(l).starts_with(pre))
        })
        .count()
}
fn check_against_census(contract: &Value, text: &str) {
    for (name, bound) in contract["counts"].as_object().expect("counts") {
        assert_eq!(bound.as_u64(), Some(census(text, name) as u64), "{name}");
    }
    let probe: Probe = serde_json::from_value(contract.clone()).unwrap();
    for pattern in &probe.forbidden {
        assert_eq!(
            census(text, pattern),
            0,
            "forbidden instruction `{pattern}` is emitted"
        );
    }
}
#[test]
fn contracts_are_derived_from_the_emission() {
    let (entries, _) = emitted();
    for (export, e) in LEGACY.into_iter().zip(&entries) {
        let c = c8::contract(ARCH, e, export);
        assert_eq!(c["symbol"], export.symbol());
        check_against_census(&c, &e.s_text);
        let lds = if export == Export::Norm { 32 } else { 0 };
        assert_eq!(
            (
                c["launch_dynamic_lds_bytes"].as_u64(),
                c["group_segment_fixed_bytes"].as_u64()
            ),
            (Some(0), Some(lds))
        );
        assert_eq!(
            (
                c["require_wave32"].as_bool(),
                c["require_zero_spills"].as_bool(),
                c["require_zero_private"].as_bool()
            ),
            (Some(true), Some(true), Some(true))
        );
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (Some(64), Some(104))
        );
        let forbidden: Vec<String> = serde_json::from_value(c["forbidden"].clone()).unwrap();
        for m in [
            "scratch_load_b32",
            "global_atomic_add_f32",
            "s_setpc_b64",
            "ds_bpermute_b32",
            "ds_swizzle_b32",
            "v_s_rsq_f32",
            "flat_load_b32",
        ] {
            assert!(
                forbidden
                    .iter()
                    .any(|p| p.strip_suffix('*').map_or(p == m, |pre| m.starts_with(pre))),
                "`{m}` must be forbidden"
            );
        }
        let counts = c["counts"].as_object().unwrap();
        assert_eq!(
            counts["v_div_scale_f32"].as_u64(),
            counts["v_div_fixup_f32"].as_u64().map(|n| 2 * n)
        );
        assert_eq!(counts["v_rcp_f32_e32"], counts["v_div_fmas_f32"]);
        assert_eq!(
            counts["v_cmp_lt_f32_e64"], 8,
            "one strict `<` per candidate"
        );
        assert_eq!(
            counts.get("v_rsq_f32_e32").and_then(Value::as_u64),
            if export == Export::Norm {
                Some(1)
            } else {
                None
            }
        );
        assert!(
            !counts.contains_key("v_cmp_le_f32_e64") && !counts.contains_key("v_cmp_ge_f32_e64"),
            "the sentinel / tie rule is a strict compare"
        );
        // 8 candidates x 8 elements + 8 final + the base divide (+ the mean divide, + 8 AWQ divides).
        let divides = if export == Export::Norm {
            64 + 8 + 1 + 1 + 8
        } else {
            64 + 8 + 1 + 8
        };
        assert_eq!(counts["v_div_scale_f32"].as_u64(), Some(2 * divides));
        // The four-error mse chains are `err * err` then three dependent FMAs, per half and candidate (exact
        // rounding of a chain only shows in near ties, so it is pinned by the instruction count, not by data):
        // every divide holds two `v_fmac`, the norm's sum of squares seven more.
        let fmac = 8 * 2 * 3 + 2 * divides + if export == Export::Norm { 7 } else { 0 };
        assert_eq!(counts["v_fmac_f32_e32"].as_u64(), Some(fmac));
    }
}

// ------------------------------------------------------------------------------------------------ 7. committed artifacts
fn kernels_dir() -> String {
    format!("{}/kernels", env!("CARGO_MANIFEST_DIR"))
}
fn top_bundle_path() -> String {
    format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = top_bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel iu4_c8_producer_pm --arch gfx1201 ... --bundle kernels/{MODULE}.hxaco)"));
    let (_, text) = emitted();
    let fresh = native::bundle(
        &native::assemble(&text, ARCH).unwrap(),
        ARCH,
        native::DEFAULT_HOST_TARGET,
    );
    assert!(
        committed == fresh,
        "{MODULE}: fresh native emission differs from the committed bundle {path}"
    );
    let lifted = peacemaker_lift::lift_object(
        &committed,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("the committed bundle does not lift: {e}"));
    assert_eq!(format!("{:?}", lifted.program.target.arch), "Gfx1201");
    let symbols: Vec<&str> = lifted
        .program
        .kernels
        .iter()
        .map(|k| k.symbol.0.as_str())
        .collect();
    assert_eq!(
        symbols,
        [ROTATE, NORM, SIGMOID, NORM_OUT],
        "the committed bundle must hold exactly the four exports"
    );
}
#[test]
fn committed_contracts_and_m7_receipt_equal_fresh_ones() {
    let (entries, _) = emitted();
    for (export, e) in c8::module_exports().into_iter().zip(&entries) {
        let fresh = c8::contract(ARCH, e, export);
        let file = format!(
            "{}/iu4_c8_producer_pm.gfx1201.{}.contract.json",
            kernels_dir(),
            export.tag()
        );
        let committed: Value = serde_json::from_slice(
            &std::fs::read(&file).unwrap_or_else(|err| panic!("{file}: {err}")),
        )
        .unwrap();
        assert_eq!(
            committed, fresh,
            "{file}: committed contract differs from the derived contract"
        );
    }
    let file = format!("{}/iu4_c8_producer_pm.gfx1201.m7.json", kernels_dir());
    let m7: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    let mut keys: Vec<&str> = m7.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut want = [NORM, NORM_OUT, ROTATE, SIGMOID];
    want.sort_unstable();
    assert_eq!(keys, want, "{file}: receipt keys");
    for symbol in [ROTATE, NORM, SIGMOID, NORM_OUT] {
        let r = &m7[symbol];
        assert_eq!(r["lift"], "byte-exact", "{symbol}");
        assert_eq!(r["obligations"], serde_json::json!({}), "{symbol}");
        assert_eq!(r["ambiguous_delays"], 0, "{symbol}");
        assert_eq!(
            r["barrier_pairs"],
            u64::from(symbol == NORM || symbol == NORM_OUT),
            "{symbol}"
        );
        assert!(r["instructions"].as_u64().unwrap() > 0);
    }
}

// ------------------------------------------------------------------------------------------------ 8. certification
#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        audit, ledger_replay, pm_check,
        toolchain::{
            build, certify, oracle_assemble_link_bundle, read_kd, IsaShapeContract, Toolchain,
        },
    };

    const HOST: &str = "host-x86_64-unknown-linux-gnu-";

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-iu4c8-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Native code object and bundle against the ROCm oracle for each export alone and for the module.
    #[test]
    fn native_code_objects_and_bundles_equal_the_rocm_oracle() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let toolchain = Toolchain {
            host_target: HOST.into(),
            ..Toolchain::default()
        };
        let (entries, text) = emitted();
        let mut units: Vec<(String, String)> = entries
            .iter()
            .zip(c8::module_exports())
            .map(|(e, x)| (x.symbol().to_owned(), e.s_text.clone()))
            .collect();
        units.push((format!("{MODULE} (module)"), text));
        let dir = tmp("identity");
        let mut failures = Vec::new();
        for (i, (name, text)) in units.iter().enumerate() {
            let s = dir.join(format!("u{i}.s"));
            std::fs::write(&s, text).unwrap();
            let oracle = oracle_assemble_link_bundle(
                &toolchain,
                &s,
                &dir.join(format!("u{i}.hxaco")),
                "gfx1201",
            )
            .unwrap_or_else(|e| panic!("{name}: oracle: {e}"));
            let (oracle_elf, oracle_bundle) = (
                std::fs::read(&oracle.elf).unwrap(),
                std::fs::read(&oracle.hsaco).unwrap(),
            );
            match native::assemble(text, ARCH) {
                Ok(elf) => {
                    let bundle = native::bundle(&elf, ARCH, HOST);
                    if let Some(why) = native_compare::compare_native_to_oracle(
                        &elf,
                        &bundle,
                        &oracle_elf,
                        &oracle_bundle,
                    ) {
                        failures.push(format!("{name}: {why}"))
                    }
                }
                Err(e) => failures.push(format!("{name}: native: {e}")),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            failures.is_empty(),
            "{} of {} units differ from the oracle:\n{}",
            failures.len(),
            units.len(),
            failures.join("\n")
        );
    }

    /// Wait replay, descriptor FP mode and M7 on the natively linked object, without ROCm tools.
    #[test]
    fn exports_pass_wait_replay_descriptor_and_m7() {
        let (entries, text) = emitted();
        for e in &entries {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = tmp("m7");
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, ARCH).unwrap()).unwrap();
        for export in c8::module_exports() {
            let symbol = export.symbol();
            let m7 = pm_check::m7(&co, "gfx1201", symbol)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!(
                (
                    m7["lift"].as_str(),
                    &m7["obligations"],
                    &m7["ambiguous_delays"]
                ),
                (
                    Some("byte-exact"),
                    &serde_json::json!({}),
                    &serde_json::json!(0)
                )
            );
            let kd = read_kd(&co, symbol).unwrap();
            assert_eq!(
                (kd.group_segment_size, kd.private_segment_size, kd.wave32),
                (export.lds_bytes(), 0, true)
            );
            assert_eq!(
                kd.compute_pgm_rsrc1 & 0x000f_f000,
                0x000f_0000,
                "FP round/denorm mode"
            );
            assert_eq!(kd.compute_pgm_rsrc1 & (1 << 26), 0, "FP16_OVFL");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Both exports certify against the ROCm oracle build (objdump parse-back, wait replay, derived shape contract,
    /// M7), the final bundle's only LDS store is waited at its barrier, and the committed bundle / receipt are the
    /// certified ones.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (entries, text) = emitted();
        for e in &entries {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = tmp("cert");
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), "gfx1201").unwrap();
        assert!(
            build.parse_back_checked,
            "the objdump parse-back did not run"
        );
        let mut receipts = serde_json::Map::new();
        for (export, e) in c8::module_exports().into_iter().zip(&entries) {
            let symbol = export.symbol();
            let m7 = pm_check::m7(&build.elf, "gfx1201", symbol)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!(
                (
                    m7["lift"].as_str(),
                    &m7["obligations"],
                    &m7["ambiguous_delays"]
                ),
                (
                    Some("byte-exact"),
                    &serde_json::json!({}),
                    &serde_json::json!(0)
                )
            );
            let contract: IsaShapeContract = serde_json::from_value(c8::contract(ARCH, e, export))
                .unwrap_or_else(|err| panic!("contract does not parse: {err}"));
            certify(
                &toolchain,
                &build,
                &s,
                "gfx1201",
                &dir.join(format!("manifest-{}.json", export.tag())),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|err| panic!("{symbol}: {err}"));
            receipts.insert(symbol.to_owned(), m7);
        }
        let bundle = std::fs::read(&build.hsaco).unwrap();
        let scan = audit::lds_barrier_scan(&bundle).unwrap();
        assert!(
            scan.sites.is_empty(),
            "lds_store_unwaited_at_barrier: {:?}",
            scan.sites
        );
        assert_eq!(
            (scan.kernels, scan.barrier_kernels, scan.barriers),
            (4, 2, 2),
            "{scan:?}"
        );
        assert!(
            std::fs::read(top_bundle_path()).unwrap() == bundle,
            "the committed bundle is not the certified build"
        );
        let file = format!("{}/iu4_c8_producer_pm.gfx1201.m7.json", kernels_dir());
        let committed: Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            Value::Object(receipts),
            "{file}: committed M7 receipt differs from the fresh one"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
