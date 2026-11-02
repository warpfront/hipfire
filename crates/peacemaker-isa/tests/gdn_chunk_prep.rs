// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 A3B GDN chunk preparation `gdn_chunk_prep_pm_gfx1201` (one export, one module of the same name: the
//! PM twin of the incumbent HIP `gdn_chunk_prep`, HK16/HV32/HD128). Exact spec / arch / domain admission,
//! determinism, the 100-byte kernarg ABI and resource descriptor, the committed bundle / contract / M7 receipt
//! against fresh emission, and (feature `toolchain`) the ROCm-oracle certification.
//!
//! The arithmetic is proven on the CPU in two independent ways.
//!
//!  1. The *emitted* export is run, whole blocks of eight waves, through the peacemaker wave32 interpreter on
//!     exactly-sized buffers (any out-of-range access is an unmapped fault, the written-byte sets must equal the live
//!     output bytes) and every output plane is compared bit for bit with an independent Rust model of the
//!     incumbent's arithmetic (separate multiplies and adds, explicit `fma` only where the source has `fmaf`, the
//!     incumbent's `expf` / `logf` / `rsqrtf` / full-division operand graphs over the interpreter's transcendental
//!     primitives, the pair-prefix shuffle-up scan with its `(pair - g0) - g1` subtract / re-add, the XOR
//!     butterfly). The interpreter runs the emission itself, unmodified: wide global stores, the `_e32` compare / select
//!     and the negated-operand multiply all have modeled semantics. The
//!     model has teeth: the serial scan, the conventional inclusive prefix, the unfused convolution and the
//!     other in-lane norm order each change bits on the same inputs.
//!  2. The per-token value graphs of the three region loops (conv -> SiLU -> q/k norm -> q-scale -> f16) are
//!     compared, as hash-consed symbolic values with only add / multiply commutation canonicalised, against the
//!     committed hipcc regions of the incumbent (`kernels/fp8_gemm.gdn.*.region.s`); every single-operand mutation of
//!     the emission breaks the equality.
//!
//! The gate graphs (positive `expf`, `logf`, the sigmoid, the scan) were read against the hipcc disassembly of the
//! current incumbent source by hand (the receipt is the H2 handoff); the CPU interpreter does not replace the GPU
//! byte oracle (`gdn_chunk_prep_pm_oracle`), and its transcendental primitives are its own models.
use peacemaker_ir::emu::{convert::f32_to_f16, trans, Emulator, Launch, Memory};
use peacemaker_isa::kernels::gdn_chunk_prep::{self, GdnChunkPrepSpec};
use peacemaker_isa::{native, Arch};
use serde_json::Value;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const IR_ARCH: peacemaker_ir::Arch = peacemaker_ir::Arch::Gfx1201;
const MODULE: &str = "gdn_chunk_prep_pm_gfx1201";
const SYMBOL: &str = "gdn_chunk_prep_pm_gfx1201";
const KERNARG: u32 = 100;
const BLOCK: u32 = 256;
const HK: usize = 16;
const HV: usize = 32;
const HD: usize = 128;
const XW: usize = (2 * HK + HV) * HD;
/// `(offset, size, name, value_kind)` in kernarg order.
const ARGS: [(u32, u32, &str, &str); 15] = [
    (0, 8, "x", "global_buffer"),
    (8, 8, "conv_w", "global_buffer"),
    (16, 8, "conv_state", "global_buffer"),
    (24, 8, "a", "global_buffer"),
    (32, 8, "b", "global_buffer"),
    (40, 8, "dt_bias", "global_buffer"),
    (48, 8, "a_log", "global_buffer"),
    (56, 8, "q", "global_buffer"),
    (64, 8, "k", "global_buffer"),
    (72, 8, "v", "global_buffer"),
    (80, 4, "T", "by_value"),
    (84, 4, "q_scale", "by_value"),
    (88, 4, "eps", "by_value"),
    (92, 4, "hk", "by_value"),
    (96, 4, "hv", "by_value"),
];

fn spec() -> GdnChunkPrepSpec {
    GdnChunkPrepSpec::A3B
}

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
fn count_prefix(text: &str, prefix: &str) -> usize {
    insns(text)
        .iter()
        .filter(|l| mnemonic(l).starts_with(prefix))
        .count()
}
/// Kernarg `(name, offset, size, value_kind)` records of a source's metadata, in order.
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
    let (e, text, _) = gdn_chunk_prep::emit_module(ARCH, &spec()).unwrap();
    (e, text)
}
fn kernel_text() -> String {
    gdn_chunk_prep::emit(ARCH, &spec()).unwrap().s_text
}

// ------------------------------------------------------------------------------------------------ 1. spec / domain
#[test]
fn spec_and_domain_are_exact() {
    assert!(gdn_chunk_prep::emit(ARCH, &spec()).is_ok());
    assert!(gdn_chunk_prep::emit_module(ARCH, &spec()).is_ok());
    for hk in [0u32, 1, 8, 15, 16, 17, 32] {
        for hv in [0u32, 1, 16, 31, 32, 33, 48] {
            for head_dim in [0u32, 64, 127, 128, 129, 256] {
                let s = GdnChunkPrepSpec { hk, hv, head_dim };
                let exact = (hk, hv, head_dim) == (16, 32, 128);
                let case = format!("{hk}/{hv}/{head_dim}");
                assert_eq!(gdn_chunk_prep::validate(&s, 64).is_ok(), exact, "{case}");
                assert_eq!(s.validate().is_ok(), exact, "{case}: spec");
                if !exact {
                    assert!(gdn_chunk_prep::emit(ARCH, &s).is_err(), "{case}: emitted");
                    assert!(
                        gdn_chunk_prep::emit_module(ARCH, &s).is_err(),
                        "{case}: module emitted"
                    );
                }
            }
        }
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(gdn_chunk_prep::emit(arch, &spec()).is_err(), "{arch:?}");
        assert!(
            gdn_chunk_prep::emit_module(arch, &spec()).is_err(),
            "{arch:?}: module"
        );
    }
    for rows in [1usize, 2, 3, 63, 64, 65, 4096, 8191, 8192] {
        assert!(
            gdn_chunk_prep::validate(&spec(), rows).is_ok(),
            "rows {rows}"
        );
    }
    for rows in [0usize, 8193, 1 << 20, usize::MAX] {
        assert!(
            gdn_chunk_prep::validate(&spec(), rows).is_err(),
            "rows {rows}"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 2. determinism
#[test]
fn emission_is_deterministic_and_the_module_is_the_one_export() {
    let (a, b) = (
        gdn_chunk_prep::emit(ARCH, &spec()).unwrap(),
        gdn_chunk_prep::emit(ARCH, &spec()).unwrap(),
    );
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(
        serde_json::to_string(&a.proof).unwrap(),
        serde_json::to_string(&b.proof).unwrap()
    );
    let (ea, ta, ma) = gdn_chunk_prep::emit_module(ARCH, &spec()).unwrap();
    let (_, tb, mb) = gdn_chunk_prep::emit_module(ARCH, &spec()).unwrap();
    assert_eq!(ta, tb);
    assert_eq!(
        serde_json::to_string(&ma).unwrap(),
        serde_json::to_string(&mb).unwrap()
    );
    assert_eq!((ma.module.as_str(), ma.arch), (MODULE, ARCH));
    assert_eq!((ma.kernels.len(), ea.len()), (1, 1));
    assert_eq!(
        ea[0].s_text, a.s_text,
        "the module entry is the standalone export"
    );
    assert_eq!(ta.matches(".amdhsa_kernel ").count(), 1);
    assert!(
        ta.contains(&format!(".amdhsa_kernel {SYMBOL}\n"))
            && ta.contains(&format!(".globl {SYMBOL}\n"))
    );
    assert!(
        ta.contains(&format!(".name: {SYMBOL}\n")) && ta.contains("amdgcn-amd-amdhsa--gfx1201")
    );
    assert_eq!(ta.matches(".symbol: ").count(), 1, "exactly the one export");
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
            gdn_chunk_prep::KERNEL_ID,
            gdn_chunk_prep::MODULE,
            gdn_chunk_prep::SYMBOL
        ),
        ("gdn_chunk_prep_pm", MODULE, SYMBOL)
    );
}

// ------------------------------------------------------------------------------------------------ 3. ABI / resources
#[test]
fn kernarg_layout_and_resources_match_contract() {
    let (entries, module_text) = emitted();
    let e = &entries[0];
    let text = &e.s_text;
    let has = |directive: &str| text.contains(&format!("\t.amdhsa_{directive}\n"));
    assert_eq!(
        (gdn_chunk_prep::KERNARG_BYTES, gdn_chunk_prep::BLOCK),
        (KERNARG, BLOCK)
    );
    assert_eq!(field(text, ".kernarg_segment_size:"), [KERNARG]);
    assert!(has(&format!("kernarg_size {KERNARG}")));
    assert_eq!(field(text, ".max_flat_workgroup_size:"), [BLOCK]);
    let found = metadata_args(text);
    assert_eq!(found.len(), ARGS.len());
    for ((name, offset, size, kind), (eo, es, en, ek)) in found.iter().zip(ARGS) {
        assert_eq!(
            (name.as_str(), *offset, *size, kind.as_str()),
            (en, eo, es, ek),
            "argument {en}"
        );
    }
    assert!(
        has("wavefront_size32 1") && text.contains(".wavefront_size: 32\n"),
        "wave32"
    );
    assert!(
        has("workgroup_processor_mode 1") && text.contains(".workgroup_processor_mode: 1\n"),
        "WGP mode (the incumbent's)"
    );
    assert!(
        has("group_segment_fixed_size 0") && text.contains(".group_segment_fixed_size: 0\n"),
        "no LDS allocation"
    );
    assert!(
        has("private_segment_fixed_size 0") && text.contains(".private_segment_fixed_size: 0\n")
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
    // Grid x and y (ceil(T/64) x 8 blocks).
    assert!(
        has("system_sgpr_workgroup_id_x 1")
            && has("system_sgpr_workgroup_id_y 1")
            && has("system_sgpr_workgroup_id_z 0")
    );
    let vgprs = field(text, ".amdhsa_next_free_vgpr");
    assert_eq!(
        vgprs,
        [128],
        "declared VGPRs (122 used, one granule-aligned reservation)"
    );
    assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0]);
    assert!(text.contains(&format!(".vgpr_count: {}\n", vgprs[0])));
    let sgprs = field(text, ".amdhsa_next_free_sgpr");
    assert!(sgprs.len() == 1 && sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0]);
    assert!(module_text.contains(&format!(".kernarg_segment_size: {KERNARG}\n")));
    // No LDS access, barrier, WMMA, scratch, flat, atomic or dynamic call: `ds_bpermute_b32` is the only DS
    // instruction (a lane crossbar with no LDS allocation) and every one is drained by its own `s_wait_dscnt 0`
    // before any later instruction can consume the result.
    let all = insns(text);
    for l in &all {
        let m = mnemonic(l);
        assert!(
            !((m.starts_with("ds_") && m != "ds_bpermute_b32")
                || m.starts_with("s_barrier")
                || m.starts_with("v_wmma")
                || m.starts_with("scratch_")
                || m.starts_with("flat_")
                || m.starts_with("buffer_")
                || m.contains("atomic")
                || m.starts_with("s_setpc")
                || m.starts_with("s_swappc")
                || m.starts_with("s_waitcnt")),
            "forbidden instruction {l}"
        );
    }
    let perm: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, l)| mnemonic(l) == "ds_bpermute_b32")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        perm.len(),
        5,
        "one crossbar per scan level d = 1, 2, 4, 8, 16"
    );
    for i in perm {
        assert_eq!(
            all[i + 1],
            "s_wait_dscnt 0x0",
            "the crossbar result is drained before its first use: {}",
            all[i]
        );
    }
}

// ------------------------------------------------------------------------------------------------ 4. arithmetic model
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
}

fn bits(x: f32) -> u32 {
    x.to_bits()
}
fn ldexp(x: f32, n: i32) -> f32 {
    ((f64::from(x)) * 2f64.powi(n.clamp(-400, 400))) as f32
}
fn exp2_hw(x: f32) -> f32 {
    f32::from_bits(trans::exp_f32(IR_ARCH, bits(x)))
}
fn log2_hw(x: f32) -> f32 {
    f32::from_bits(trans::log_f32(IR_ARCH, bits(x)))
}
fn rcp_hw(x: f32) -> f32 {
    f32::from_bits(trans::rcp_f32(IR_ARCH, bits(x)))
}
fn rsq_hw(x: f32) -> f32 {
    f32::from_bits(trans::rsq_f32(IR_ARCH, bits(x)))
}

/// hipcc's `expf(x)` (`neg`: `expf(-x)`), as its instruction graph reads: `x * C1`, `fma(C1, x, -prod)`,
/// `rndne`, `fma(C2, x, .)`, `(prod - n) + lo`, `cvt_i32`, `v_exp_f32`, `v_ldexp_f32`, then the clamps.
fn expf(x: f32, neg: bool) -> f32 {
    let (c1, c2) = if neg {
        (f32::from_bits(0xbfb8aa3b), f32::from_bits(0xb2a5705f))
    } else {
        (f32::from_bits(0x3fb8aa3b), f32::from_bits(0x32a5705f))
    };
    let prod = x * c1;
    let lo = c1.mul_add(x, -prod);
    let n = prod.round_ties_even();
    let lo = c2.mul_add(x, lo);
    let r = (prod - n) + lo;
    let e = ldexp(exp2_hw(r), n as i32);
    // `v_cmp_ngt / nlt` are true on NaN.
    let (lo_c, hi_c) = if neg {
        (f32::from_bits(0x42ce8ed0), f32::from_bits(0xc2b17218))
    } else {
        (f32::from_bits(0xc2ce8ed0), f32::from_bits(0x42b17218))
    };
    let (keep_lo, keep_hi) = if neg {
        (!(lo_c < x), !(hi_c > x))
    } else {
        (!(lo_c > x), !(hi_c < x))
    };
    let e = if keep_lo { e } else { 0.0 };
    if keep_hi {
        e
    } else {
        f32::INFINITY
    }
}
fn expf_pos(x: f32) -> f32 {
    expf(x, false)
}
fn expf_neg(x: f32) -> f32 {
    expf(x, true)
}
/// hipcc's `logf` (OCML, f32): prescale a denormal argument by `2^32`, `v_log_f32`, the `ln2` hi/lo split, the
/// prescale correction.
fn logf(x: f32) -> f32 {
    let tiny = f32::from_bits(0x0080_0000) > x;
    let xs = if tiny { ldexp(x, 32) } else { x };
    let lg = log2_hw(xs);
    let m1 = f32::from_bits(0x3f317217) * lg;
    let m2 = f32::from_bits(0x3f317217).mul_add(lg, -m1);
    let m2 = f32::from_bits(0x3377d1cf).mul_add(lg, m2);
    let m1 = m1 + m2;
    let lg = if lg.abs() < f32::INFINITY { m1 } else { lg };
    lg - if tiny {
        f32::from_bits(0x41b17218)
    } else {
        0.0
    }
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        expf_pos(x)
    } else {
        logf(1.0 + expf_pos(x))
    }
}
/// hipcc's `rsqrtf` through its vector denormal guard.
fn rsqrtf(x: f32) -> f32 {
    let small = f32::from_bits(0x0080_0000) > x;
    let r = rsq_hw(if small {
        x * f32::from_bits(0x4b800000)
    } else {
        x
    });
    if small {
        r * f32::from_bits(0x45800000)
    } else {
        r
    }
}
/// Correctly rounded (IEEE) division: what the full division graph computes.
fn fdiv(n: f32, d: f32) -> f32 {
    n / d
}
fn silu(y: f32) -> f32 {
    fdiv(y, 1.0 + expf_neg(y))
}
fn f16_bits(x: f32) -> u16 {
    f32_to_f16(bits(x))
}

/// Model switches, one per observable order the proof pins.
#[derive(Clone, Copy, Default)]
struct Variant {
    /// The scan is a serial per-row prefix.
    serial_scan: bool,
    /// The scan is a conventional inclusive prefix (no `(pair - g0) - g1` subtract / re-add).
    plain_prefix: bool,
    /// The convolution is multiplied and added separately.
    unfused_conv: bool,
    /// The norm reduction adds the in-lane squares last.
    lane_last_norm: bool,
}

struct Case {
    t: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    state: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    dt: Vec<f32>,
    alog: Vec<f32>,
    q_scale: f32,
    eps: f32,
}

#[derive(Clone, PartialEq, Debug)]
struct Planes {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    a: Vec<u32>,
    b: Vec<u32>,
    state: Vec<u32>,
}

fn case(t: usize, seed: u64) -> Case {
    let mut r = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
    // Inputs span the input graph. Every eighth channel is an identity tap (`w = [0, 0, 0, 1]`) fed with SiLU
    // arguments past both expf clamps, deep negative and exact zeros; all other channels are dense small-weight
    // taps. The interpreter's `v_div_scale_f32` model is measured for denominator exponents outside 228..=253,
    // i.e. SiLU arguments outside about -87.3..-70 and beta arguments outside -87.3..-70, which the draws avoid;
    // gate arguments span the softplus branches (> 20, in range, < -20), log-rate parameters straddle zero.
    let identity = |ch: usize| ch % 8 == 0;
    let mut x = Vec::with_capacity(t * XW);
    for i in 0..t * XW {
        x.push(if identity(i % XW) {
            match r.below(16) {
                0 => 0.0,
                1 => r.range(100.0, 300.0),
                2 => -r.range(100.0, 300.0),
                3 => -r.range(40.0, 69.0),
                4 => r.range(-1e-3, 1e-3),
                _ => r.range(-6.0, 6.0),
            }
        } else if r.below(64) == 0 {
            0.0
        } else {
            r.range(-3.0, 3.0)
        });
    }
    let mut w = Vec::with_capacity(XW * 4);
    for i in 0..XW * 4 {
        if identity(i / 4) {
            w.extend([0.0, 0.0, 0.0, 1.0][i % 4..=i % 4].iter());
        } else {
            w.push(if r.below(97) == 0 {
                0.0
            } else {
                r.range(-1.5, 1.5)
            });
        }
    }
    let state: Vec<f32> = (0..XW * 3).map(|_| r.range(-3.0, 3.0)).collect();
    let a: Vec<f32> = (0..t * HV)
        .map(|_| match r.below(8) {
            0 => r.range(20.0, 60.0),
            1 => r.range(-60.0, -19.0),
            2 => 0.0,
            _ => r.range(-12.0, 12.0),
        })
        .collect();
    let b: Vec<f32> = (0..t * HV)
        .map(|_| match r.below(10) {
            0 => r.range(-110.0, -90.0),
            1 => r.range(85.0, 120.0),
            2 => r.range(-60.0, -45.0),
            _ => r.range(-12.0, 12.0),
        })
        .collect();
    let dt: Vec<f32> = (0..HV).map(|_| r.range(-3.0, 3.0)).collect();
    let alog: Vec<f32> = (0..HV).map(|_| r.range(-2.5, 2.5)).collect();
    Case {
        t,
        x,
        w,
        state,
        a,
        b,
        dt,
        alog,
        q_scale: 0.088_388_35,
        eps: 1e-6,
    }
}

/// The incumbent's arithmetic over the whole launch, in plain Rust.
fn model(c: &Case, var: Variant) -> Planes {
    let t = c.t;
    let (mut q, mut k, mut v) = (
        vec![0u16; t * HK * HD],
        vec![0u16; t * HK * HD],
        vec![0u16; t * HV * HD],
    );
    let mut a = c.a.iter().map(|x| bits(*x)).collect::<Vec<u32>>();
    let mut b = c.b.iter().map(|x| bits(*x)).collect::<Vec<u32>>();
    // Gates, per 64-row chunk and head.
    for chunk0 in (0..t).step_by(64) {
        let rows = (t - chunk0).min(64);
        for h in 0..HV {
            let gate = |r: usize| -> f32 {
                let off = (chunk0 + r) * HV + h;
                -expf_pos(c.alog[h]) * softplus(c.a[off] + c.dt[h])
            };
            let mut g = vec![0.0f32; 64];
            for r in 0..rows {
                g[r] = gate(r);
                let off = (chunk0 + r) * HV + h;
                b[off] = bits(fdiv(1.0, 1.0 + expf_neg(c.b[off])));
            }
            if var.serial_scan {
                let mut acc = 0.0f32;
                for r in 0..rows {
                    acc += g[r];
                    a[(chunk0 + r) * HV + h] = bits(acc);
                }
                continue;
            }
            let mut pair: Vec<f32> = (0..32).map(|l| g[2 * l] + g[2 * l + 1]).collect();
            let mut d = 1;
            while d < 32 {
                let prev = pair.clone();
                for l in d..32 {
                    pair[l] = prev[l] + prev[l - d];
                }
                d <<= 1;
            }
            for l in 0..32 {
                let (g0, g1) = (g[2 * l], g[2 * l + 1]);
                let (a0, a1) = if var.plain_prefix {
                    // Exclusive lane prefix, then sequential adds.
                    let before = if l == 0 { 0.0 } else { pair[l - 1] };
                    (before + g0, before + g0 + g1)
                } else {
                    let before = (pair[l] - g0) - g1;
                    (before + g0, (before + g0) + g1)
                };
                if 2 * l < rows {
                    a[(chunk0 + 2 * l) * HV + h] = bits(a0);
                }
                if 2 * l + 1 < rows {
                    a[(chunk0 + 2 * l + 1) * HV + h] = bits(a1);
                }
            }
        }
    }
    // Convolution / SiLU / norms.
    for unit in 0..2 * HK + HV {
        let mut lanes: Vec<Vec<[f32; 4]>> = Vec::with_capacity(32);
        for lane in 0..32 {
            let dbase = unit * HD + lane * 4;
            let mut win = [[0.0f32; 4]; 3];
            for ch in 0..4 {
                win[0][ch] = c.state[(dbase + ch) * 3 + 2];
                win[1][ch] = c.state[(dbase + ch) * 3 + 1];
                win[2][ch] = c.state[(dbase + ch) * 3];
            }
            // Per-chunk windows restart from the parent x (the chunk-0 window is the ring).
            let mut outs: Vec<[f32; 4]> = Vec::with_capacity(t);
            for r in 0..t {
                if r % 64 == 0 && r > 0 {
                    for ch in 0..4 {
                        win[0][ch] = c.x[(r - 3) * XW + dbase + ch];
                        win[1][ch] = c.x[(r - 2) * XW + dbase + ch];
                        win[2][ch] = c.x[(r - 1) * XW + dbase + ch];
                    }
                }
                let mut o = [0.0f32; 4];
                for ch in 0..4 {
                    let cur = c.x[r * XW + dbase + ch];
                    let w4 = |j: usize| c.w[(dbase + ch) * 4 + j];
                    let y = if var.unfused_conv {
                        (((w4(2) * win[2][ch]) + w4(3) * cur) + w4(1) * win[1][ch])
                            + w4(0) * win[0][ch]
                    } else {
                        let y = w4(3).mul_add(cur, w4(2) * win[2][ch]);
                        let y = w4(1).mul_add(win[1][ch], y);
                        w4(0).mul_add(win[0][ch], y)
                    };
                    o[ch] = silu(y);
                    win[0][ch] = win[1][ch];
                    win[1][ch] = win[2][ch];
                    win[2][ch] = cur;
                }
                outs.push(o);
            }
            lanes.push(outs);
        }
        // Norm across the wave per row.
        for r in 0..t {
            let mut sums: Vec<f32> = (0..32)
                .map(|l| {
                    let o = lanes[l][r];
                    let sq = [o[0] * o[0], o[1] * o[1], o[2] * o[2], o[3] * o[3]];
                    if var.lane_last_norm {
                        sq[0] + (sq[1] + (sq[2] + sq[3]))
                    } else {
                        ((sq[0] + sq[1]) + sq[2]) + sq[3]
                    }
                })
                .collect();
            if unit < 2 * HK {
                for d in [16usize, 8, 4, 2, 1] {
                    let prev = sums.clone();
                    for l in 0..32 {
                        sums[l] = prev[l] + prev[l ^ d];
                    }
                }
            }
            for l in 0..32 {
                let o = lanes[l][r];
                for ch in 0..4 {
                    let (plane, head, width) = if unit < HK {
                        (0, unit, HK)
                    } else if unit < 2 * HK {
                        (1, unit - HK, HK)
                    } else {
                        (2, unit - 2 * HK, HV)
                    };
                    let z = if plane < 2 {
                        let inv = rsqrtf(sums[l] + c.eps);
                        let mut z = o[ch] * inv;
                        if plane == 0 {
                            z *= c.q_scale;
                        }
                        z
                    } else {
                        o[ch]
                    };
                    let idx = (r * width + head) * HD + l * 4 + ch;
                    match plane {
                        0 => q[idx] = f16_bits(z),
                        1 => k[idx] = f16_bits(z),
                        _ => v[idx] = f16_bits(z),
                    }
                }
            }
        }
    }
    // Conv ring.
    let mut state = vec![0u32; XW * 3];
    for ch in 0..XW {
        let (old0, old1) = (c.state[ch * 3], c.state[ch * 3 + 1]);
        state[ch * 3] = bits(c.x[(t - 1) * XW + ch]);
        state[ch * 3 + 1] = bits(if t >= 2 { c.x[(t - 2) * XW + ch] } else { old0 });
        state[ch * 3 + 2] = bits(if t >= 3 {
            c.x[(t - 3) * XW + ch]
        } else if t == 2 {
            old0
        } else {
            old1
        });
    }
    Planes {
        q,
        k,
        v,
        a,
        b,
        state,
    }
}

// ------------------------------------------------------------------------------------------------ 5. interpreter rig
const E_KARG: u64 = 0x1_0000;
const E_X: u64 = 0x1000_0000;
const E_W: u64 = 0x2000_0000;
const E_STATE: u64 = 0x3000_0000;
const E_A: u64 = 0x4000_0000;
const E_B: u64 = 0x5000_0000;
const E_DT: u64 = 0x6000_0000;
const E_ALOG: u64 = 0x7000_0000;
const E_Q: u64 = 0x8000_0000;
const E_K: u64 = 0x9000_0000;
const E_V: u64 = 0xa000_0000;
const POISON: u16 = 0xbeef;

struct Rig {
    program: peacemaker_ir::Program,
}
impl Rig {
    fn pm() -> Self {
        let (_, text, _) = gdn_chunk_prep::emit_module(ARCH, &spec()).unwrap();
        Self::lift(&native::assemble(&text, ARCH).unwrap())
    }
    fn lift(object: &[u8]) -> Self {
        let lifted = peacemaker_lift::lift_object(
            object,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .unwrap();
        Rig {
            program: lifted.program,
        }
    }
}
fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|w| u16::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
fn region<'a>(mem: &'a Memory, name: &str) -> &'a peacemaker_ir::emu::Region {
    mem.regions.iter().find(|r| r.name == name).unwrap()
}

struct Run {
    planes: Planes,
    /// Written bytes per output plane: q, k, v, a, b, state.
    written: [usize; 6],
    inputs_untouched: bool,
}

/// One whole launch: every `(ceil(T / 64), 8)` block, on freshly mapped exactly-sized buffers (any out-of-range
/// access is an unmapped fault). `args` overrides `[T, hk, hv]` in the kernarg block.
fn run(rig: &Rig, symbol: &str, c: &Case, args: Option<[u32; 3]>) -> Result<Run, String> {
    let [t, hk, hv] = args.unwrap_or([c.t as u32, HK as u32, HV as u32]);
    let mut karg = Vec::new();
    for ptr in [E_X, E_W, E_STATE, E_A, E_B, E_DT, E_ALOG, E_Q, E_K, E_V] {
        karg.extend(ptr.to_le_bytes());
    }
    karg.extend(t.to_le_bytes());
    karg.extend(c.q_scale.to_le_bytes());
    karg.extend(c.eps.to_le_bytes());
    karg.extend(hk.to_le_bytes());
    karg.extend(hv.to_le_bytes());
    assert_eq!(karg.len(), KERNARG as usize);
    let poison = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| POISON.to_le_bytes()).collect() };
    let inputs = [
        ("x", E_X, f32s(&c.x)),
        ("conv_w", E_W, f32s(&c.w)),
        ("dt", E_DT, f32s(&c.dt)),
        ("alog", E_ALOG, f32s(&c.alog)),
    ];
    let mut mem = Memory::default();
    mem.map("kernarg".into(), E_KARG, karg.clone())?;
    for (name, base, bytes) in &inputs {
        mem.map((*name).into(), *base, bytes.clone())?;
    }
    let (state0, a0, b0) = (f32s(&c.state), f32s(&c.a), f32s(&c.b));
    mem.map("conv_state".into(), E_STATE, state0.clone())?;
    mem.map("a".into(), E_A, a0.clone())?;
    mem.map("b".into(), E_B, b0.clone())?;
    mem.map("q".into(), E_Q, poison(c.t * HK * HD))?;
    mem.map("k".into(), E_K, poison(c.t * HK * HD))?;
    mem.map("v".into(), E_V, poison(c.t * HV * HD))?;
    let emu = Emulator::new(&rig.program, symbol)?;
    for y in 0..8u32 {
        for x in 0..c.t.div_ceil(64) as u32 {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: E_KARG,
                    workgroup: [x, y, 0],
                    block: [BLOCK, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )?;
        }
    }
    let inputs_untouched = inputs.iter().all(|(name, _, bytes)| {
        let r = region(&mem, name);
        r.bytes == *bytes && r.written.count() == 0
    }) && region(&mem, "kernarg").bytes == karg
        && region(&mem, "kernarg").written.count() == 0;
    let r = |n: &str| region(&mem, n);
    Ok(Run {
        planes: Planes {
            q: halves(&r("q").bytes),
            k: halves(&r("k").bytes),
            v: halves(&r("v").bytes),
            a: words(&r("a").bytes),
            b: words(&r("b").bytes),
            state: words(&r("conv_state").bytes),
        },
        written: [
            r("q").written.count(),
            r("k").written.count(),
            r("v").written.count(),
            r("a").written.count(),
            r("b").written.count(),
            r("conv_state").written.count(),
        ],
        inputs_untouched,
    })
}

fn describe(plane: &str, i: usize, got: u32, want: u32) -> String {
    format!("{plane}[{i}]: got {got:#010x}, want {want:#010x}")
}
fn assert_planes(name: &str, got: &Planes, want: &Planes) {
    fn cmp<T: Copy + Into<u32> + PartialEq>(name: &str, plane: &str, g: &[T], w: &[T]) {
        assert_eq!(g.len(), w.len(), "{name} {plane} length");
        let bad: Vec<usize> = (0..g.len()).filter(|&i| g[i] != w[i]).collect();
        assert!(
            bad.is_empty(),
            "{name}: {} of {} {plane} words differ, first {}",
            bad.len(),
            g.len(),
            describe(plane, bad[0], g[bad[0]].into(), w[bad[0]].into())
        );
    }
    cmp(name, "q", &got.q, &want.q);
    cmp(name, "k", &got.k, &want.k);
    cmp(name, "v", &got.v, &want.v);
    cmp(name, "a", &got.a, &want.a);
    cmp(name, "b", &got.b, &want.b);
    cmp(name, "conv_state", &got.state, &want.state);
}
fn assert_written(name: &str, c: &Case, r: &Run) {
    assert!(
        r.inputs_untouched,
        "{name}: an input or the kernarg block changed"
    );
    let want = [
        c.t * HK * HD * 2,
        c.t * HK * HD * 2,
        c.t * HV * HD * 2,
        c.t * HV * 4,
        c.t * HV * 4,
        XW * 3 * 4,
    ];
    assert_eq!(
        r.written, want,
        "{name}: exactly the live output bytes are written (q, k, v, a, b, conv_state)"
    );
}

/// Row counts that cover single-row, two-row, three-row (ring) and chunk-edge shapes plus a multi-chunk tail.
const SHAPES: [usize; 8] = [1, 2, 3, 4, 63, 64, 65, 130];

#[test]
fn emitted_export_equals_the_incumbent_arithmetic_model() {
    let rig = Rig::pm();
    for (i, t) in SHAPES.into_iter().enumerate() {
        let c = case(t, 0x1234 + i as u64);
        let name = format!("T={t}");
        let r = run(&rig, SYMBOL, &c, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_written(&name, &c, &r);
        assert_planes(&name, &r.planes, &model(&c, Variant::default()));
    }
}

/// The proof has teeth: each pinned order changes bits on inputs of the same family.
#[test]
fn the_pinned_orders_are_observable() {
    let c = case(130, 77);
    let want = model(&c, Variant::default());
    let differs =
        |var: Variant, plane: &dyn Fn(&Planes) -> Vec<u32>| plane(&model(&c, var)) != plane(&want);
    let a = |p: &Planes| p.a.clone();
    let q = |p: &Planes| p.q.iter().map(|&h| u32::from(h)).collect::<Vec<u32>>();
    assert!(
        differs(
            Variant {
                serial_scan: true,
                ..Variant::default()
            },
            &a
        ),
        "a serial prefix is not the pair-prefix scan"
    );
    assert!(
        differs(
            Variant {
                plain_prefix: true,
                ..Variant::default()
            },
            &a
        ),
        "a conventional prefix is not the subtract / re-add"
    );
    assert!(
        differs(
            Variant {
                unfused_conv: true,
                ..Variant::default()
            },
            &q
        ),
        "an unfused convolution is not the fma chain"
    );
    assert!(
        differs(
            Variant {
                lane_last_norm: true,
                ..Variant::default()
            },
            &q
        ),
        "another in-lane square order is not the incumbent's"
    );
}

#[test]
fn guarded_bounds_and_domain_exits_touch_nothing() {
    let rig = Rig::pm();
    let c = case(2, 5);
    // Every launch with T outside 1..=8192, hk != 16 or hv != 32 exits before any access: the buffers are the
    // exact (T = 2) extents, so any access beyond them, or any store at all, would be caught.
    for args in [
        [0u32, 16, 32],
        [0x8000_0000, 16, 32],
        [8193, 16, 32],
        [u32::MAX, 16, 32],
        [2, 15, 32],
        [2, 16, 31],
        [2, 17, 32],
        [2, 16, 33],
        [2, 0, 0],
        [2, 32, 16],
    ] {
        let mut host = case(2, 5);
        host.t = 2;
        let r = run(&rig, SYMBOL, &host, Some(args)).unwrap_or_else(|e| panic!("{args:?}: {e}"));
        assert_eq!(r.written, [0; 6], "{args:?}: wrote");
        assert!(r.inputs_untouched, "{args:?}: inputs changed");
        assert!(
            r.planes.a == c.a.iter().map(|x| bits(*x)).collect::<Vec<u32>>(),
            "{args:?}: gate plane changed"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 6. contract
fn forbidden(contract: &Value) -> Vec<String> {
    contract["forbidden"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}
fn check_against_census(contract: &Value, text: &str) {
    let census = |name: &str| -> usize {
        match name.strip_suffix('*') {
            Some(prefix) => count_prefix(text, prefix),
            None => count(text, name),
        }
    };
    for (name, bound) in contract["counts"].as_object().expect("counts") {
        assert_eq!(bound.as_u64(), Some(census(name) as u64), "{name}");
    }
    for pattern in forbidden(contract) {
        assert_eq!(
            census(&pattern),
            0,
            "forbidden instruction `{pattern}` is emitted"
        );
    }
}

// ------------------------------------------------------------------------------------------------ 5b. operand graphs
/// Hash-consed symbolic values: one shared interner, so two instruction sequences compute the same value exactly when
/// they build the same id. Commutative IEEE operations (`add`, `mul`, the product pair of an `fma`) compare with their
/// operands sorted; everything else, including the order of subtractions, divisions and every operand of the
/// transcendental, scale, ldexp and select graphs, is operand-exact.
#[derive(Default)]
struct Interner {
    ids: std::collections::HashMap<String, u32>,
    text: Vec<String>,
}
impl Interner {
    fn node(&mut self, s: String) -> u32 {
        if let Some(&i) = self.ids.get(&s) {
            return i;
        }
        let i = self.text.len() as u32;
        self.text.push(s.clone());
        self.ids.insert(s, i);
        i
    }
    fn leaf(&mut self, name: &str) -> u32 {
        self.node(format!("leaf:{name}"))
    }
    fn op(&mut self, name: &str, args: &[u32]) -> u32 {
        let mut a = args.to_vec();
        match name {
            "add" | "mul" if a.len() == 2 => a.sort_unstable(),
            "fma" => {
                // `fma(a, b, c)`: the product pair commutes.
                if a[0] > a[1] {
                    a.swap(0, 1);
                }
            }
            _ => {}
        }
        self.node(format!(
            "{name}({})",
            a.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        ))
    }
}
struct Machine<'a> {
    i: &'a mut Interner,
    regs: std::collections::HashMap<String, u32>,
    /// Data registers of every global store, in program order.
    stores: Vec<Vec<u32>>,
    /// `ds_bpermute_b32` address values that are the lane-XOR crossbar for a distance.
    xor_addr: std::collections::HashMap<u32, u32>,
}
impl<'a> Machine<'a> {
    fn new(i: &'a mut Interner) -> Self {
        Machine {
            i,
            regs: Default::default(),
            stores: Vec::new(),
            xor_addr: Default::default(),
        }
    }
    fn set(&mut self, reg: &str, id: u32) {
        self.regs.insert(reg.to_owned(), id);
    }
    fn reg(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.regs.get(name) {
            return id;
        }
        let id = self.i.leaf(&format!("init:{name}"));
        self.regs.insert(name.to_owned(), id);
        id
    }
    /// One operand with its `-` / `|..|` modifiers.
    fn get(&mut self, tok: &str) -> u32 {
        if let Some(rest) = tok.strip_prefix('-') {
            if rest.starts_with(['v', 's', '|']) {
                let inner = self.get(rest);
                return self.i.op("neg", &[inner]);
            }
        }
        if let Some(rest) = tok.strip_prefix('|').and_then(|t| t.strip_suffix('|')) {
            let inner = self.get(rest);
            return self.i.op("abs", &[inner]);
        }
        let is_reg = tok.starts_with(['v', 's'])
            && tok[1..].starts_with(|c: char| c.is_ascii_digit())
            || matches!(tok, "vcc_lo" | "ttmp7" | "ttmp9" | "exec_lo");
        if is_reg {
            self.reg(tok)
        } else {
            self.i.leaf(&format!("lit:{tok}"))
        }
    }
    /// The `v[lo:hi]` registers of a range operand.
    fn range(tok: &str) -> Vec<String> {
        let (kind, body) = (&tok[..1], tok[2..].trim_end_matches(']'));
        let (lo, hi) = body.split_once(':').unwrap();
        (lo.parse::<u32>().unwrap()..=hi.parse().unwrap())
            .map(|n| format!("{kind}{n}"))
            .collect()
    }
    fn step(&mut self, line: &str) {
        let t = line.split(';').next().unwrap().trim();
        if t.is_empty() {
            return;
        }
        let (m, rest) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
        if m.starts_with("s_wait")
            || m == "s_delay_alu"
            || m.starts_with("s_cbranch")
            || m == "s_branch"
            || m.starts_with("s_cmp")
        {
            return;
        }
        // Only DPP carries a space-separated modifier tail on an operand.
        let (rest, dpp) = match rest.find("row_xmask:") {
            Some(p) => (
                &rest[..p],
                rest[p..]
                    .split_whitespace()
                    .next()
                    .map(|s| s.trim_start_matches("row_xmask:").to_owned()),
            ),
            None => (rest, None),
        };
        let ops: Vec<&str> = rest
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let core = m.trim_end_matches("_e32").trim_end_matches("_e64");
        if m.starts_with("global_store_") {
            let data: Vec<String> = if ops[1].contains('[') {
                Self::range(ops[1])
            } else {
                vec![ops[1].to_owned()]
            };
            let ids = data.iter().map(|r| self.reg(r)).collect();
            self.stores.push(ids);
            return;
        }
        if m.starts_with("global_load_") || m.starts_with("s_load_") {
            let dst: Vec<String> = if ops[0].contains('[') {
                Self::range(ops[0])
            } else {
                vec![ops[0].to_owned()]
            };
            for (k, r) in dst.iter().enumerate() {
                let leaf = self.i.leaf(&format!(
                    "load:{m}:{}:{k}:{}",
                    ops[1..].join(","),
                    self.i.text.len()
                ));
                self.regs.insert(r.clone(), leaf);
            }
            return;
        }
        let dst = ops[0].to_owned();
        let args: Vec<u32> = ops[1..].iter().map(|o| self.get(o)).collect();
        let id = match core {
            "v_mov_b32" | "s_mov_b32" if dpp.is_none() => args[0],
            "v_mov_b32_dpp" => {
                let k = dpp.expect("dpp control");
                let src = self.get(ops[1]);
                self.i.op(&format!("xor{k}"), &[src])
            }
            "v_permlanex16_b32" => {
                // Identity selects (0x76543210 low, 0xfedcba98 high) are the lane ^ 16 exchange.
                let (lo, hi) = (self.i.leaf("lit:0x76543210"), self.i.leaf("lit:0xfedcba98"));
                if args[1] == lo && args[2] == hi {
                    self.i.op("xor16", &[args[0]])
                } else {
                    self.i.op("permlanex16", &args)
                }
            }
            "ds_bpermute_b32" => match self.xor_addr.get(&args[0]) {
                Some(&d) => self.i.op(&format!("xor{d}"), &[args[1]]),
                None => self.i.op("bpermute", &args),
            },
            "v_fmac_f32" => {
                let old = self.reg(&dst);
                self.i.op("fma", &[args[0], args[1], old])
            }
            "v_fma_f32" => self.i.op("fma", &args),
            "v_mul_f32" => self.i.op("mul", &args),
            "v_add_f32" => self.i.op("add", &args),
            "v_sub_f32" => self.i.op("sub", &args),
            "v_div_scale_f32" => {
                // `vdst, sdst, a, b, c`
                let (a, b, c) = (self.get(ops[2]), self.get(ops[3]), self.get(ops[4]));
                let flag = self.i.op("div_scale_flag", &[a, b, c]);
                if ops[1] != "null" {
                    self.regs.insert(ops[1].to_owned(), flag);
                }
                self.i.op("div_scale", &[a, b, c])
            }
            "v_div_fmas_f32" => {
                let vcc = self.reg("vcc_lo");
                let mut a = args.clone();
                a.push(vcc);
                self.i.op("div_fmas", &a)
            }
            "v_cvt_f16_f32" => self.i.op("cvt_f16", &args),
            _ if core.starts_with("v_cmp_") => {
                let name = core.to_owned();
                self.i.op(&name, &args)
            }
            _ => {
                let name = core.to_owned();
                self.i.op(&name, &args)
            }
        };
        self.regs.insert(dst, id);
    }
    fn run(&mut self, text: &str) {
        for l in text.lines() {
            self.step(l);
        }
    }
    /// Prime the crossbar addresses: `lane ^ d` through the incumbent's `xor / cmp / cndmask / shl` chain.
    fn prime_xor_addresses(&mut self, lane_reg: &str) {
        for d in [16u32, 8, 4, 2, 1] {
            let mut scratch = Machine::new(self.i);
            let lane = scratch.i.leaf("lane");
            scratch.set(lane_reg, lane);
            scratch.run(&format!(
                "v_xor_b32_e32 v48, {d}, {lane_reg}\nv_cmp_gt_u32_e32 vcc_lo, 32, v48\nv_cndmask_b32_e32 v48, {lane_reg}, v48, vcc_lo\nv_lshlrev_b32_e32 v48, 2, v48\n"
            ));
            let addr = scratch.regs["v48"];
            self.xor_addr.insert(addr, d);
        }
    }
}

fn golden(name: &str) -> String {
    let file = format!(
        "{}/kernels/fp8_gemm.gdn.{name}.region.s",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"))
}
/// `key=reg` pairs of one `; <header>:` golden line.
fn header(text: &str, key: &str) -> Vec<(String, String)> {
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(&format!("; {key}: ")))
        .unwrap_or_else(|| panic!("golden header `{key}`"));
    line.split_whitespace()
        .map(|kv| {
            kv.split_once('=').map_or_else(
                || (String::new(), kv.to_owned()),
                |(k, v)| (k.to_owned(), v.to_owned()),
            )
        })
        .collect()
}
/// The straight-line body of one region loop of the emitted export (label to its back-edge).
fn loop_body(text: &str, label: &str) -> String {
    let start = text
        .lines()
        .position(|l| l.trim() == format!("{label}:"))
        .unwrap_or_else(|| panic!("loop {label}"));
    let mut body = Vec::new();
    for l in text.lines().skip(start + 1) {
        if l.trim() == format!("s_cbranch_scc1 {label}") {
            break;
        }
        body.push(l);
    }
    body.join("\n")
}
const CHANNEL_LEAF: [&str; 8] = ["w2", "win2", "w3", "cur", "w1", "win1", "w0", "win0"];

/// One region loop of the emitted export computes, per channel and per stored half, the very value graph of the
/// hipcc incumbent's regions (the committed goldens lifted from the production object): conv (separate `w2 * win2`,
/// then three `fma`) -> SiLU (`expf(-y)` graph, full division graph) for each of the four channels, then (q/k) the
/// per-lane squares and sums, the XOR-butterfly exchange, `rsqrtf` through the denormal guard, `z = out * inv`, the
/// separate `q_scale` multiply (q), and the f16 RNE conversions. Operands are compared exactly: only IEEE-commutative
/// add / multiply operand order is canonicalised.
#[test]
fn emitted_regions_are_the_incumbent_operand_graphs() {
    region_graphs(&kernel_text()).unwrap_or_else(|e| panic!("{e}"));
}
fn region_graphs(text: &str) -> Result<(), String> {
    // PM channel `c` register of each conv leaf (the VGPR map of the emitter).
    let leaf_reg = |name: &str, c: u32| -> String {
        let (base, off) = match name {
            "w2" => (40, 2),
            "w3" => (40, 3),
            "w1" => (40, 1),
            "w0" => (40, 0),
            "win0" => (56, 0),
            "win1" => (60, 0),
            "win2" => (64, 0),
            "cur" => (68, 0),
            _ => unreachable!(),
        };
        let reg = if base == 40 {
            base + 4 * c + off
        } else {
            base + c
        };
        format!("v{reg}")
    };
    for (label, region, golden_tail) in [
        (".Lgdcp_q", 'q', "norm_q"),
        (".Lgdcp_k", 'k', "norm_k"),
        (".Lgdcp_v", 'v', "cvt_v"),
    ] {
        let mut interner = Interner::default();
        // Incumbent: four conv/SiLU channels, then the region tail on their outputs.
        let silu = golden("conv_silu");
        let inputs = header(&silu, "inputs");
        assert_eq!(
            inputs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            ["w2", "win2", "w3", "cur", "w1", "win1", "w0", "win0"]
        );
        let mut outs = Vec::new();
        for c in 0..4u32 {
            let mut m = Machine::new(&mut interner);
            for (k, reg) in &inputs {
                let leaf = m.i.leaf(&format!("{k}_c{c}"));
                m.set(reg, leaf);
            }
            m.run(&silu);
            let out = header(&silu, "outputs")[0].1.clone();
            outs.push(m.regs[&out]);
        }
        let tail = golden(golden_tail);
        let mut m = Machine::new(&mut interner);
        for (k, reg) in header(&tail, "inputs") {
            let c: usize = k[1..].parse().unwrap();
            m.set(&reg, outs[c]);
        }
        if let Some(lane) = header_opt(&tail, "lane") {
            let lane_leaf = m.i.leaf("lane");
            m.set(&lane[0].1, lane_leaf);
            m.prime_xor_addresses(&lane[0].1);
        }
        for (k, reg) in header_opt(&tail, "sinputs").unwrap_or_default() {
            let leaf = m.i.leaf(&k);
            m.set(&reg, leaf);
        }
        m.run(&tail);
        let want: Vec<u32> = header(&tail, "outputs")
            .into_iter()
            .map(|(_, reg)| m.regs[&reg])
            .collect();
        // Emitted export: the same four channels in one interleaved loop body.
        let mut m = Machine::new(&mut interner);
        for c in 0..4u32 {
            for k in CHANNEL_LEAF {
                let leaf = m.i.leaf(&format!("{k}_c{c}"));
                m.set(&leaf_reg(k, c), leaf);
            }
        }
        let (eps, qscale) = (m.i.leaf("eps"), m.i.leaf("qscale"));
        m.set("s30", eps);
        m.set("s29", qscale);
        let psel = m.i.leaf("lit:0x76543210");
        m.set("s39", psel);
        m.run(&loop_body(&text, label));
        let store = m.stores.last().ok_or("the region stores no row")?;
        // Two packed-f16 registers: v116.l v116.h v117.l v117.h.
        let got: Vec<u32> = ["v116.l", "v116.h", "v117.l", "v117.h"]
            .iter()
            .map(|r| m.regs[*r])
            .collect();
        if store.len() != 2 {
            return Err(format!(
                "region {region}: the row store is {} registers wide",
                store.len()
            ));
        }
        for (h, (g, w)) in got.iter().zip(&want).enumerate() {
            if g != w {
                return Err(format!("region {region} half {h}: emitted graph differs from the incumbent's (got {}, want {})", m.i.text[*g as usize], m.i.text[*w as usize]));
            }
        }
    }
    Ok(())
}

/// The graph check has teeth: each single mutation of the emitted text (an `expf` constant, the order of the `expf`
/// reduction subtraction, a reduction distance, the q-scale operand, the division fixup operands, the `rsqrtf`
/// rescale constant) breaks equality with the incumbent's graph.
#[test]
fn the_operand_graph_check_has_teeth() {
    let text = kernel_text();
    for (from, to) in [
        ("0xbfb8aa3b", "0xbfb8aa3c"),
        (
            "v_sub_f32_e32 v88, v88, v108",
            "v_sub_f32_e32 v88, v108, v88",
        ),
        ("row_xmask:4 row_mask", "row_xmask:2 row_mask"),
        ("v_mul_f32_e32 v80, s29, v80", "v_mul_f32_e32 v80, s30, v80"),
        (
            "v_div_fixup_f32 v80, v96, v84, v76",
            "v_div_fixup_f32 v80, v96, v76, v84",
        ),
        ("0x45800000", "0x45800001"),
    ] {
        assert!(
            text.contains(from),
            "mutation site `{from}` is not in the emission"
        );
        assert!(
            region_graphs(&text.replace(from, to)).is_err(),
            "mutating `{from}` -> `{to}` went unnoticed"
        );
    }
}
fn header_opt(text: &str, key: &str) -> Option<Vec<(String, String)>> {
    text.lines()
        .any(|l| l.starts_with(&format!("; {key}: ")))
        .then(|| header(text, key))
}
#[test]
fn contract_is_derived_from_the_emission() {
    let (entries, _) = emitted();
    let c = gdn_chunk_prep::contract(ARCH, &entries[0]);
    assert_eq!(c["symbol"], SYMBOL);
    check_against_census(&c, &entries[0].s_text);
    assert_eq!(
        (
            c["launch_dynamic_lds_bytes"].as_u64(),
            c["group_segment_fixed_bytes"].as_u64()
        ),
        (Some(0), Some(0))
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
        (Some(128), Some(104))
    );
    for m in [
        "ds_load_b128",
        "ds_store_b32",
        "ds_swizzle_b32",
        "s_barrier_signal",
        "v_wmma_f32_16x16x16_f16",
        "scratch_load_b32",
        "global_atomic_add_f32",
        "s_setpc_b64",
    ] {
        assert!(
            forbidden(&c)
                .iter()
                .any(|p| p.strip_suffix('*').map_or(p == m, |pre| m.starts_with(pre))),
            "`{m}` must be forbidden"
        );
    }
    let counts = c["counts"].as_object().unwrap();
    assert_eq!(counts["ds_bpermute_b32"], 5);
    assert_eq!(
        (
            counts["v_exp_f32_e32"].as_u64(),
            counts["v_log_f32_e32"].as_u64(),
            counts["v_rsq_f32_e32"].as_u64()
        ),
        (Some(17), Some(2), Some(2))
    );
    assert_eq!(
        (
            counts["v_div_scale_f32"].as_u64(),
            counts["v_div_fmas_f32"].as_u64(),
            counts["v_div_fixup_f32"].as_u64()
        ),
        (Some(28), Some(14), Some(14))
    );
    assert_eq!(counts["s_endpgm"], 1);
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
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel gdn_chunk_prep_pm --arch gfx1201 ... --bundle kernels/{MODULE}.hxaco)"));
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
            frontend: peacemaker_ir::inst::Frontend::Builder,
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
        [SYMBOL],
        "the committed bundle must hold exactly the one export"
    );
}
#[test]
fn committed_contract_and_m7_receipt_equal_fresh_ones() {
    let (entries, _) = emitted();
    let fresh = gdn_chunk_prep::contract(ARCH, &entries[0]);
    let file = format!("{}/gdn_chunk_prep_pm.gfx1201.contract.json", kernels_dir());
    let committed: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    assert_eq!(
        committed, fresh,
        "{file}: committed contract differs from the derived contract"
    );
    let file = format!("{}/gdn_chunk_prep_pm.gfx1201.m7.json", kernels_dir());
    let m7: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    let keys: Vec<&str> = m7.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, [SYMBOL], "{file}: receipt keys");
    let r = &m7[SYMBOL];
    assert_eq!(r["lift"], "byte-exact");
    assert_eq!(r["obligations"], serde_json::json!({}));
    assert_eq!(r["ambiguous_delays"], 0);
    assert_eq!(r["barrier_pairs"], 0);
    assert!(r["instructions"].as_u64().unwrap() > 0);
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
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-gdnprep-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Native code object and bundle against the ROCm oracle for the export alone and for the module.
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
        let units = [
            (SYMBOL.to_owned(), entries[0].s_text.clone()),
            (format!("{MODULE} (module)"), text),
        ];
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
    fn export_passes_wait_replay_descriptor_and_m7() {
        let (entries, text) = emitted();
        ledger_replay::replay_waits(&entries[0].s_text, ARCH).unwrap();
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = tmp("m7");
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, ARCH).unwrap()).unwrap();
        let m7 =
            pm_check::m7(&co, "gfx1201", SYMBOL).unwrap_or_else(|err| panic!("{SYMBOL}: {err}"));
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
        let kd = read_kd(&co, SYMBOL).unwrap();
        assert_eq!(
            (kd.group_segment_size, kd.private_segment_size, kd.wave32),
            (0, 0, true)
        );
        assert_eq!(
            kd.compute_pgm_rsrc1 & 0x000f_f000,
            0x000f_0000,
            "FP round/denorm mode"
        );
        assert_eq!(kd.compute_pgm_rsrc1 & (1 << 26), 0, "FP16_OVFL");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The export certifies against the ROCm oracle build (objdump parse-back, wait replay, derived shape contract, M7),
    /// the final bundle has no barrier and no LDS store to drain, and the committed bundle / receipt are the certified ones.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (entries, text) = emitted();
        ledger_replay::replay_waits(&entries[0].s_text, ARCH).unwrap();
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
        let m7 = pm_check::m7(&build.elf, "gfx1201", SYMBOL)
            .unwrap_or_else(|err| panic!("{SYMBOL}: {err}"));
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
        let contract: IsaShapeContract =
            serde_json::from_value(gdn_chunk_prep::contract(ARCH, &entries[0]))
                .unwrap_or_else(|err| panic!("contract does not parse: {err}"));
        certify(
            &toolchain,
            &build,
            &s,
            "gfx1201",
            &dir.join("manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|err| panic!("{SYMBOL}: {err}"));
        let bundle = std::fs::read(&build.hsaco).unwrap();
        let scan = audit::lds_barrier_scan(&bundle).unwrap();
        assert!(
            scan.sites.is_empty(),
            "lds_store_unwaited_at_barrier: {:?}",
            scan.sites
        );
        assert_eq!(
            (scan.kernels, scan.barrier_kernels, scan.barriers),
            (1, 0, 0),
            "{scan:?}"
        );
        assert!(
            std::fs::read(top_bundle_path()).unwrap() == bundle,
            "the committed bundle is not the certified build"
        );
        let file = format!("{}/gdn_chunk_prep_pm.gfx1201.m7.json", kernels_dir());
        let committed: Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            serde_json::json!({ SYMBOL: m7 }),
            "{file}: committed M7 receipt differs from the fresh one"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
