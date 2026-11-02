// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 A3B Q prep `fa_qprep_pm_gfx1201` (one export `attention_fa2_qprep_pm_gfx1201`: one-wave Q RMSNorm,
//! gate passthrough, captured-angle half-split rotary-64, f16 RNE Q16 scatter). Exact spec / arch / domain admission,
//! determinism, the 48-byte kernarg ABI and resource descriptor, the committed bundle / contract / M7 receipt against
//! fresh emission, and (feature `toolchain`) the ROCm-oracle certification.
//!
//! The arithmetic is proven on the CPU from the emitted instruction text itself. The scalar float tail
//! (`s_add_f32`, `s_cmp_lt_f32`, `s_mul_f32`, `v_s_rsq_f32`) is outside the peacemaker CPU interpreter, so two
//! regions of the emitted code run on a minimal lane model written here instead: the reduction (`fma` leaves,
//! in-lane levels, lane XOR butterfly) against the incumbent 256-leaf LDS halving tree, bit for bit, and the output
//! region (`(q * w) * rms`, rotary, f16 RNE) against the unfused incumbent graph. The scalar tail is pinned as an
//! ordered operand-exact instruction sequence (the cached hipcc disassembly of
//! `deinterleave_q_rmsnorm_f32_batched`), the division as the same op-for-op graph the incumbent uses. The CPU model
//! does not replace the GPU byte oracle.
use peacemaker_isa::kernels::fa_prefill::{FaKvCodec, FaPrefillSpec};
use peacemaker_isa::kernels::fa_qprep;
use peacemaker_isa::{native, Arch};
use serde_json::Value;
use std::collections::HashMap;
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "fa_qprep_pm_gfx1201";
const SYMBOL: &str = "attention_fa2_qprep_pm_gfx1201";
const KERNARG: u32 = 48;
const BLOCK: u32 = 128;
/// `(offset, size, name, value_kind)` in kernarg order.
const ARGS: [(u32, u32, &str, &str); 7] = [
    (0, 8, "interleaved", "global_buffer"),
    (8, 8, "q16", "global_buffer"),
    (16, 8, "gate", "global_buffer"),
    (24, 8, "q_norm", "global_buffer"),
    (32, 8, "angles", "global_buffer"),
    (40, 4, "rows", "by_value"),
    (44, 4, "eps", "by_value"),
];

fn spec() -> FaPrefillSpec {
    FaPrefillSpec {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        kv: FaKvCodec::Q8_0,
    }
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
/// The instructions that are neither waits nor issue hints (those are inserted by the builder where needed).
fn core(text: &str) -> Vec<&str> {
    insns(text)
        .into_iter()
        .filter(|l| !mnemonic(l).starts_with("s_wait") && mnemonic(l) != "s_delay_alu")
        .collect()
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
    let (e, text, _) = fa_qprep::emit_module(ARCH, &spec()).unwrap();
    (e, text)
}
fn kernel_text() -> String {
    fa_qprep::emit(ARCH, &spec()).unwrap().s_text
}

// ------------------------------------------------------------------------------------------------ 1. spec / domain
#[test]
fn spec_and_domain_are_exact() {
    assert!(fa_qprep::emit(ARCH, &spec()).is_ok());
    assert!(fa_qprep::emit_module(ARCH, &spec()).is_ok());
    for heads in [0u32, 1, 8, 15, 16, 17, 24, 32] {
        for kv_heads in [0u32, 1, 2, 3, 4, 8] {
            for head_dim in [0u32, 64, 128, 255, 256, 257, 512] {
                let s = FaPrefillSpec {
                    heads,
                    kv_heads,
                    head_dim,
                    kv: FaKvCodec::Q8_0,
                };
                let exact = (heads, kv_heads, head_dim) == (16, 2, 256);
                let case = format!("{heads}/{kv_heads}/{head_dim}");
                assert_eq!(fa_qprep::validate(&s, 64).is_ok(), exact, "{case}");
                if !exact {
                    assert!(fa_qprep::emit(ARCH, &s).is_err(), "{case}: emitted");
                    assert!(
                        fa_qprep::emit_module(ARCH, &s).is_err(),
                        "{case}: module emitted"
                    );
                }
            }
        }
    }
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(fa_qprep::emit(arch, &spec()).is_err(), "{arch:?}");
        assert!(
            fa_qprep::emit_module(arch, &spec()).is_err(),
            "{arch:?}: module"
        );
    }
    for rows in [1usize, 2, 7, 8, 9, 63, 64, 65, 4096, 8191, 8192] {
        assert!(fa_qprep::validate(&spec(), rows).is_ok(), "rows {rows}");
    }
    for rows in [0usize, 8193, 1 << 20, usize::MAX] {
        assert!(fa_qprep::validate(&spec(), rows).is_err(), "rows {rows}");
    }
}

// ------------------------------------------------------------------------------------------------ 2. determinism
#[test]
fn emission_is_deterministic_and_the_module_is_the_one_export() {
    let (a, b) = (
        fa_qprep::emit(ARCH, &spec()).unwrap(),
        fa_qprep::emit(ARCH, &spec()).unwrap(),
    );
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(
        serde_json::to_string(&a.proof).unwrap(),
        serde_json::to_string(&b.proof).unwrap()
    );
    let (ea, ta, ma) = fa_qprep::emit_module(ARCH, &spec()).unwrap();
    let (_, tb, mb) = fa_qprep::emit_module(ARCH, &spec()).unwrap();
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
        (fa_qprep::KERNEL_ID, fa_qprep::MODULE, fa_qprep::SYMBOL),
        ("fa_qprep_pm", MODULE, SYMBOL)
    );
}

// ------------------------------------------------------------------------------------------------ 3. ABI / resources
#[test]
fn kernarg_layout_and_resources_match_contract() {
    let (entries, module_text) = emitted();
    let e = &entries[0];
    let text = &e.s_text;
    let has = |directive: &str| text.contains(&format!("\t.amdhsa_{directive}\n"));
    assert_eq!((fa_qprep::KERNARG_BYTES, fa_qprep::BLOCK), (KERNARG, BLOCK));
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
        "WGP mode"
    );
    assert!(
        has("group_segment_fixed_size 0") && text.contains(".group_segment_fixed_size: 0\n"),
        "no LDS"
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
    // Grid x only (4 * rows blocks).
    assert!(
        has("system_sgpr_workgroup_id_x 1")
            && has("system_sgpr_workgroup_id_y 0")
            && has("system_sgpr_workgroup_id_z 0")
    );
    let vgprs = field(text, ".amdhsa_next_free_vgpr");
    assert_eq!(
        vgprs,
        [56],
        "all 56 VGPRs (a whole granule) are declared and used"
    );
    assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0]);
    assert!(text.contains(&format!(".vgpr_count: {}\n", vgprs[0])));
    let sgprs = field(text, ".amdhsa_next_free_sgpr");
    assert!(sgprs.len() == 1 && sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0]);
    assert!(module_text.contains(&format!(".kernarg_segment_size: {KERNARG}\n")));
    // No LDS, barrier, WMMA, scratch, flat, atomic or dynamic call anywhere: nothing to drain.
    for l in insns(text) {
        let m = mnemonic(l);
        assert!(
            !(m.starts_with("ds_")
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
}

// ------------------------------------------------------------------------------------------------ 4. lane model
type Vregs = HashMap<u32, [u32; 32]>;

/// Independent f32 -> f16 round-to-nearest-even (finite inputs; overflow rounds to infinity).
fn f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = u64::from(m | 0x80_0000);
        let shift = (14 - exp) as u32;
        let mut q = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((exp as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn vreg(tok: &str) -> u32 {
    tok.trim_start_matches('-')
        .trim_start_matches('v')
        .split('.')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}
/// One operand of a lane op as an f32 bit pattern for `lane`.
fn operand(tok: &str, lane: usize, v: &Vregs, s: &HashMap<u32, u32>) -> u32 {
    let (neg, t) = match tok.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, tok),
    };
    let bits = if let Some(r) = t.strip_prefix('v') {
        v[&r.split('.').next().unwrap().parse().unwrap()][lane]
    } else if let Some(r) = t.strip_prefix('s') {
        s[&r.parse().unwrap()]
    } else if t == "0" {
        0
    } else if t == "1.0" {
        1.0f32.to_bits()
    } else if let Some(h) = t.strip_prefix("0x") {
        u32::from_str_radix(h, 16).unwrap()
    } else {
        panic!("unmodelled operand {tok}")
    };
    if neg {
        bits ^ 0x8000_0000
    } else {
        bits
    }
}
fn ops_of(line: &str) -> Vec<String> {
    let rest = line.split_once(char::is_whitespace).map_or("", |(_, r)| r);
    rest.split(',').map(|t| t.trim().to_owned()).collect()
}
/// Run the modelled lane subset over `lines` (waits and issue hints skipped, anything else unmodelled panics).
fn run_lanes(lines: &[&str], v: &mut Vregs, s: &HashMap<u32, u32>) {
    for line in lines {
        let m = mnemonic(line);
        if m.starts_with("s_wait") || m == "s_delay_alu" {
            continue;
        }
        let ops = ops_of(line);
        let dst = vreg(&ops[0]);
        let mut out = [0u32; 32];
        for lane in 0..32 {
            let f = |i: usize| f32::from_bits(operand(&ops[i], lane, v, s));
            out[lane] = match m {
                "v_fma_f32" => f(1).mul_add(f(2), f(3)).to_bits(),
                "v_add_f32_e32" => (f(1) + f(2)).to_bits(),
                "v_mul_f32_e32" => (f(1) * f(2)).to_bits(),
                "v_permlanex16_b32" => {
                    assert_eq!(
                        s[&ops[2].trim_start_matches('s').parse::<u32>().unwrap()],
                        0x7654_3210,
                        "identity low selects"
                    );
                    assert_eq!(ops[3], "0xfedcba98", "identity high selects");
                    v[&vreg(&ops[1])][lane ^ 16]
                }
                "v_mov_b32_dpp" => {
                    let mut it = ops[1].split_whitespace();
                    let src = vreg(it.next().unwrap());
                    let k: usize = it
                        .find_map(|t| t.strip_prefix("row_xmask:"))
                        .expect("row_xmask")
                        .parse()
                        .unwrap();
                    assert!(k < 16, "row_xmask stays inside the 16-lane row");
                    v[&src][lane ^ k]
                }
                "v_cvt_f16_f32_e32" => {
                    assert!(ops[0].ends_with(".l"), "low-half destination: {line}");
                    let old = v.get(&dst).map_or(0, |r| r[lane]);
                    (old & 0xffff_0000) | u32::from(f16_rne(f(1)))
                }
                other => panic!("unmodelled instruction {other}: {line}"),
            };
        }
        v.insert(dst, out);
    }
}

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
}

/// The incumbent `deinterleave_q_rmsnorm_f32_batched` reduction at block 256: leaf `fma(q, q, +0)` per thread, then
/// the LDS halving `sdata[t] = sdata[t + s] + sdata[t]` for `s = 128 ... 1`.
fn incumbent_sum(q: &[f32; 256]) -> f32 {
    let mut sdata: Vec<f32> = q.iter().map(|&x| x.mul_add(x, 0.0)).collect();
    let mut s = 128;
    while s > 0 {
        for t in 0..s {
            sdata[t] = sdata[t + s] + sdata[t];
        }
        s >>= 1;
    }
    sdata[0]
}

fn q_vectors() -> Vec<[f32; 256]> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut out = vec![[0.0f32; 256], [1.0; 256], [-3.5; 256]];
    let mut one = [0.0f32; 256];
    one[255] = 1.5;
    out.push(one);
    let mut tiny = [0.0f32; 256];
    for (i, t) in tiny.iter_mut().enumerate() {
        *t = f32::from_bits(1 + i as u32 * 977); // subnormal leaves: fma(q,q,0) is +0 or tiny, still exact order data
    }
    out.push(tiny);
    for case in 0..96 {
        let (lo, hi) = if case % 3 == 0 { (-30, 30) } else { (-3, 3) };
        let mut q = [0.0f32; 256];
        for x in q.iter_mut() {
            *x = rng.f32(lo, hi);
        }
        out.push(q);
    }
    out
}

/// The emitted reduction (`fma` leaves, in-lane levels, lane XOR butterfly) leaves the incumbent halving tree's root,
/// bit for bit, in every lane.
#[test]
fn emitted_reduction_is_the_incumbent_halving_tree() {
    let text = kernel_text();
    let all = core(&text);
    let leaf = all
        .iter()
        .position(|l| l.starts_with("v_fma_f32") && l.ends_with(", 0"))
        .expect("fma leaf");
    let end = all
        .iter()
        .position(|l| l.starts_with("v_div_scale_f32"))
        .expect("division");
    let red = &all[leaf..end];
    // The region is exactly: eight leaves, 4 + 2 + 1 in-lane adds, then five XOR levels of (peer, add).
    let mn: Vec<&str> = red.iter().map(|l| mnemonic(l)).collect();
    let mut want: Vec<&str> = vec!["v_fma_f32"; 8];
    want.extend(["v_add_f32_e32"; 7]);
    for k in [16u32, 8, 4, 2, 1] {
        want.push(if k == 16 {
            "v_permlanex16_b32"
        } else {
            "v_mov_b32_dpp"
        });
        want.push("v_add_f32_e32");
    }
    assert_eq!(mn, want, "reduction region");
    let xors: Vec<usize> = red
        .iter()
        .filter(|l| mnemonic(l) == "v_mov_b32_dpp")
        .map(|l| {
            l.split("row_xmask:")
                .nth(1)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect();
    assert_eq!(xors, [8, 4, 2, 1]);
    // The permlanex16 selects register holds the identity nibbles (checked by the model as it executes).
    let psel: Vec<&&str> = all
        .iter()
        .filter(|l| l.starts_with("s_mov_b32") && l.ends_with("0x76543210"))
        .collect();
    assert_eq!(psel.len(), 1);
    let psel_reg: u32 = psel[0]
        .split_whitespace()
        .nth(1)
        .unwrap()
        .trim_end_matches(',')
        .trim_start_matches('s')
        .parse()
        .unwrap();
    let s: HashMap<u32, u32> = HashMap::from([(psel_reg, 0x7654_3210)]);
    for (n, q) in q_vectors().iter().enumerate() {
        let mut v: Vregs = HashMap::new();
        for k in 0..8u32 {
            v.insert(
                8 + k,
                std::array::from_fn(|l| q[l + 32 * k as usize].to_bits()),
            );
        }
        run_lanes(red, &mut v, &s);
        let want = incumbent_sum(q).to_bits();
        let sum_reg = vreg(&ops_of(red.last().unwrap())[0]);
        for lane in 0..32 {
            assert_eq!(
                v[&sum_reg][lane], want,
                "vector {n} lane {lane}: root {:#010x} vs incumbent {want:#010x}",
                v[&sum_reg][lane]
            );
        }
    }
}

/// The proof has teeth: a serial, an in-lane-first or a pairwise-adjacent order is not the incumbent tree.
#[test]
fn the_reduction_order_is_observable() {
    let vectors = q_vectors();
    let leaves = |q: &[f32; 256]| -> Vec<f32> { q.iter().map(|&x| x.mul_add(x, 0.0)).collect() };
    let (mut serial_diff, mut adjacent_diff) = (0, 0);
    for q in &vectors {
        let want = incumbent_sum(q).to_bits();
        if leaves(q).iter().fold(0.0f32, |a, &x| a + x).to_bits() != want {
            serial_diff += 1;
        }
        let mut t = leaves(q);
        while t.len() > 1 {
            t = t.chunks(2).map(|c| c[0] + c[1]).collect();
        }
        if t[0].to_bits() != want {
            adjacent_diff += 1;
        }
    }
    assert!(
        serial_diff > 0 && adjacent_diff > 0,
        "{serial_diff} {adjacent_diff}: other orders agree on every vector"
    );
}

/// The division and the scalar rsqrt tail: operand-exact ordered sequences of the incumbent's cached disassembly.
#[test]
fn division_and_scalar_rsqrt_tail_are_the_incumbent_sequences() {
    let text = kernel_text();
    let all = core(&text);
    // mean = sum / 256.0f: the full-precision graph (two scales, rcp, Newton fmas, fmas, fixup) with den = 256.0f.
    assert_eq!(f32::from_bits(0x4380_0000), 256.0);
    assert_eq!(
        (
            count(&text, "v_div_scale_f32"),
            count(&text, "v_rcp_f32_e32"),
            count(&text, "v_div_fmas_f32"),
            count(&text, "v_div_fixup_f32")
        ),
        (2, 1, 1, 1)
    );
    let den: Vec<&&str> = all
        .iter()
        .filter(|l| l.starts_with("s_mov_b32") && l.ends_with("0x43800000"))
        .collect();
    assert_eq!(den.len(), 1);
    let den_reg = den[0]
        .split_whitespace()
        .nth(1)
        .unwrap()
        .trim_end_matches(',')
        .to_owned();
    let first = all
        .iter()
        .find(|l| l.starts_with("v_div_scale_f32"))
        .unwrap();
    assert!(
        first.contains(&format!("null, {den_reg}, {den_reg}, v")),
        "denominator operand: {first}"
    );
    // Scalar tail, in order and operand exact: eps + mean, float compare, scale by 2^24, select, rsq, 2^12, select.
    let at = all
        .iter()
        .position(|l| l.starts_with("v_div_fixup_f32"))
        .unwrap();
    let tail = &all[at + 1..];
    let rd = |l: &str| -> Vec<String> { ops_of(l) };
    let rf = rd(tail[0]);
    assert_eq!(mnemonic(tail[0]), "v_readfirstlane_b32");
    let mean = rf[0].clone();
    assert!(
        rd(tail[1]).len() == 3 && mnemonic(tail[1]) == "s_add_f32",
        "{}",
        tail[1]
    );
    let add = rd(tail[1]);
    assert_eq!(add[0], mean, "eps + mean lands in the mean register");
    assert_eq!(add[2], mean);
    let eps = add[1].clone();
    assert!(eps.starts_with('s') && eps != mean);
    assert_eq!(tail[2], format!("s_cmp_lt_f32 {mean}, 0x800000"));
    let scaled = rd(tail[3]);
    assert_eq!(
        (mnemonic(tail[3]), scaled[1].as_str(), scaled[2].as_str()),
        ("s_mul_f32", mean.as_str(), "0x4b800000")
    );
    assert_eq!(
        tail[4],
        format!("s_cselect_b32 {mean}, {}, {mean}", scaled[0])
    );
    assert_eq!(tail[5], format!("v_s_rsq_f32 {mean}, {mean}"));
    let up = rd(tail[6]);
    assert_eq!(
        (mnemonic(tail[6]), up[1].as_str(), up[2].as_str()),
        ("s_mul_f32", mean.as_str(), "0x45800000")
    );
    assert_eq!(
        up[0], scaled[0],
        "the 2^12 product reuses the scratch of the 2^24 one"
    );
    let rms = rd(tail[7]);
    assert_eq!(
        (mnemonic(tail[7]), rms[1].as_str(), rms[2].as_str()),
        ("s_cselect_b32", up[0].as_str(), mean.as_str())
    );
    assert_eq!(
        (
            f32::from_bits(0x4b80_0000),
            f32::from_bits(0x4580_0000),
            f32::from_bits(0x0080_0000)
        ),
        (16_777_216.0, 4096.0, f32::MIN_POSITIVE)
    );
    // The rms register is first read by the output multiply and by nothing else.
    assert_eq!(tail[8], format!("v_mul_f32_e32 v24, v8, v24"));
    assert_eq!(tail[9], format!("v_mul_f32_e32 v24, {}, v24", rms[0]));
    assert!(
        !all.iter().any(|l| mnemonic(l) == "v_rsq_f32_e32"),
        "the scalar rsq form is kept"
    );
    // The scale pair is value preserving: rsq(x * 2^24) * 2^12 == rsq(x) (model rsq = exact reciprocal root).
    let rsq = |x: f32| (1.0 / f64::from(x).sqrt()) as f32;
    let guarded = |x: f32| {
        if x < f32::MIN_POSITIVE {
            rsq(x * 16_777_216.0) * 4096.0
        } else {
            rsq(x)
        }
    };
    for bits in [
        1u32,
        2,
        0x1234,
        0x7f_ffff,
        0x80_0000,
        0x80_0001,
        0x3f80_0000,
        0x4000_0000,
        0x7f7f_ffff,
    ] {
        let x = f32::from_bits(bits);
        let (g, w) = (f64::from(guarded(x)), 1.0 / f64::from(x).sqrt());
        assert!(((g - w) / w).abs() < 1e-6, "x {bits:#x}: {g} vs {w}");
    }
    assert!(guarded(0.0).is_infinite());
}

/// The output region (`(q * w) * rms`, rotary on the (k = 0, k = 1) pair, f16 RNE) against the unfused incumbent graph.
#[test]
fn emitted_output_region_is_norm_rotary_and_f16_rne() {
    let text = kernel_text();
    let all = core(&text);
    let first = all
        .iter()
        .position(|l| *l == "v_mul_f32_e32 v24, v8, v24")
        .expect("first output multiply");
    let rms_line = all[first + 1];
    let rms_reg: u32 = ops_of(rms_line)[1].trim_start_matches('s').parse().unwrap();
    let last = all
        .iter()
        .rposition(|l| mnemonic(l) == "v_cvt_f16_f32_e32")
        .unwrap();
    let region = &all[first..=last];
    let mn: Vec<&str> = region.iter().map(|l| mnemonic(l)).collect();
    let mut want: Vec<&str> = ["v_mul_f32_e32"; 16].to_vec();
    want.extend(["v_mul_f32_e32", "v_mul_f32_e32", "v_fma_f32", "v_fma_f32"]);
    want.extend(["v_cvt_f16_f32_e32"; 8]);
    assert_eq!(mn, want, "output region");
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for n in 0..200 {
        let (lo, hi) = if n % 4 == 0 { (-12, 6) } else { (-3, 3) };
        let q: Vec<f32> = (0..256).map(|_| rng.f32(lo, hi)).collect();
        let w: Vec<f32> = (0..256).map(|_| rng.f32(-2, 2)).collect();
        let (cos, sin): (Vec<f32>, Vec<f32>) =
            (0..32).map(|_| (rng.f32(-6, -1), rng.f32(-6, -1))).unzip();
        let rms = rng.f32(-4, 4).abs();
        let mut v: Vregs = HashMap::new();
        for k in 0..8usize {
            v.insert(
                8 + k as u32,
                std::array::from_fn(|l| q[l + 32 * k].to_bits()),
            );
            v.insert(
                24 + k as u32,
                std::array::from_fn(|l| w[l + 32 * k].to_bits()),
            );
            v.insert(16 + k as u32, [0xdead_beef; 32]);
        }
        v.insert(32, std::array::from_fn(|l| cos[l].to_bits()));
        v.insert(33, std::array::from_fn(|l| sin[l].to_bits()));
        run_lanes(region, &mut v, &HashMap::from([(rms_reg, rms.to_bits())]));
        // Unfused incumbent: (q * w) * rms, half-split rotary-64 over (i, i + 32), then the preconvert f16 RNE.
        let mut o: Vec<f32> = (0..256).map(|d| (q[d] * w[d]) * rms).collect();
        for i in 0..32 {
            let (x0, x1) = (o[i], o[i + 32]);
            o[i] = x0.mul_add(cos[i], -(x1 * sin[i]));
            o[i + 32] = x0.mul_add(sin[i], x1 * cos[i]);
        }
        for k in 0..8usize {
            for lane in 0..32usize {
                let got = v[&(16 + k as u32)][lane];
                assert_eq!(
                    got >> 16,
                    0xdead,
                    "case {n}: the high half of the f16 register is untouched"
                );
                assert_eq!(
                    got & 0xffff,
                    u32::from(f16_rne(o[lane + 32 * k])),
                    "case {n} dim {}",
                    lane + 32 * k
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ 5. addressing / exits
struct Mem {
    mnemonic: String,
    reg: String,
    voff: String,
    base: String,
    offset: u32,
}
fn mem_lines(text: &str, prefix: &str) -> Vec<Mem> {
    insns(text)
        .into_iter()
        .filter(|l| mnemonic(l).starts_with(prefix))
        .map(|l| {
            let ops = ops_of(l);
            let tail: Vec<&str> = ops[2].split_whitespace().collect();
            let offset = tail
                .iter()
                .find_map(|t| t.strip_prefix("offset:"))
                .map_or(0, |o| o.parse().unwrap());
            let (reg, voff, base) = if prefix == "global_load" {
                (ops[0].clone(), ops[1].clone(), tail[0].to_owned())
            } else {
                (ops[1].clone(), ops[0].clone(), tail[0].to_owned())
            };
            Mem {
                mnemonic: mnemonic(l).to_owned(),
                reg,
                voff,
                base,
                offset,
            }
        })
        .collect()
}

/// Every global access (register, lane offset register, SADDR pair, immediate), against an independent layout:
/// lane `l`, dim `d = l + 32 k`; interleaved head row = 512 f32 (Q then gate), gate / q16 / angle rows contiguous.
#[test]
fn global_accesses_cover_the_layout_exactly() {
    let text = kernel_text();
    let loads = mem_lines(&text, "global_load");
    let stores = mem_lines(&text, "global_store");
    let mut want_loads: Vec<(String, String, u32)> = Vec::new(); // (reg, base, imm)
    for k in 0..8u32 {
        want_loads.push((format!("v{}", 8 + k), "s[24:25]".into(), 128 * k)); // Q
        want_loads.push((format!("v{}", 24 + k), "s[14:15]".into(), 128 * k)); // q_norm
        want_loads.push((format!("v{}", 48 + k), "s[24:25]".into(), 1024 + 128 * k));
        // gate half
    }
    want_loads.push(("v32".into(), "s[30:31]".into(), 0)); // cos[lane]
    want_loads.push(("v33".into(), "s[30:31]".into(), 128)); // sin[lane] = angles[32 + lane]
    assert!(
        loads
            .iter()
            .all(|m| m.mnemonic == "global_load_b32" && m.voff == "v2"),
        "f32 lane offsets are lane * 4"
    );
    let mut got: Vec<_> = loads
        .iter()
        .map(|m| (m.reg.clone(), m.base.clone(), m.offset))
        .collect();
    got.sort();
    want_loads.sort();
    assert_eq!(got, want_loads);
    let mut want_stores: Vec<(String, String, String, u32)> = Vec::new();
    for k in 0..8u32 {
        want_stores.push((
            "global_store_b16".into(),
            format!("v{}", 16 + k),
            "s[28:29]".into(),
            64 * k,
        )); // dim lane+32k at 2 B
        want_stores.push((
            "global_store_b32".into(),
            format!("v{}", 48 + k),
            "s[26:27]".into(),
            128 * k,
        )); // gate f32
    }
    for m in &stores {
        let lane_off = if m.mnemonic == "global_store_b16" {
            "v3"
        } else {
            "v2"
        };
        assert_eq!(m.voff, lane_off, "{}: lane offset", m.mnemonic);
    }
    let mut got: Vec<_> = stores
        .iter()
        .map(|m| (m.mnemonic.clone(), m.reg.clone(), m.base.clone(), m.offset))
        .collect();
    got.sort();
    want_stores.sort();
    assert_eq!(got, want_stores);
    // Lane `l` reads / writes element l + 32k: byte 4l + 128k (f32) and 2l + 64k (f16), all inside the 1024 / 512 B row.
    for k in 0..8u32 {
        assert_eq!((4 * 31 + 128 * k) / 4, 31 + 32 * k);
        assert_eq!((2 * 31 + 64 * k) / 2, 31 + 32 * k);
    }
}

/// The scalar address derivation: `r = 4 * block + wave` (`row = r >> 4`, `head = r & 15`); bases `r << 11`
/// (interleaved), `r << 10` (gate), `r << 9` (q16), `(r >> 4) << 8` (angles).
#[test]
fn head_row_bases_follow_the_layout() {
    let text = kernel_text();
    let all = core(&text);
    let seq = [
        "s_lshl_b32 s32, s21, 11",
        "s_add_co_u32 s24, s8, s32",
        "s_add_co_ci_u32 s25, s9, 0",
        "s_lshl_b32 s32, s21, 10",
        "s_add_co_u32 s26, s12, s32",
        "s_add_co_ci_u32 s27, s13, 0",
        "s_lshl_b32 s32, s21, 9",
        "s_add_co_u32 s28, s10, s32",
        "s_add_co_ci_u32 s29, s11, 0",
        "s_lshr_b32 s33, s21, 4",
        "s_lshl_b32 s32, s33, 8",
        "s_add_co_u32 s30, s16, s32",
        "s_add_co_ci_u32 s31, s17, 0",
    ];
    assert!(
        all.windows(seq.len()).any(|w| w == seq),
        "address derivation sequence"
    );
    // Kernarg loads: interleaved/q16/gate/q_norm pointers (s8..s15), angles pointer + rows + eps (s16..s19).
    assert!(all.contains(&"s_load_b256 s[8:15], s[0:1], 0x0"));
    assert!(all.contains(&"s_load_b128 s[16:19], s[0:1], 0x20"));
    // Independent byte layout over the whole authoring domain.
    for rows in [1u64, 2, 63, 64, 4096, 8191, 8192] {
        for r in 0..rows * 16 {
            let (row, head) = (r >> 4, r & 15);
            assert_eq!(r << 11, (row * 8192 + head * 512) * 4, "interleaved r {r}");
            assert_eq!(r << 10, (row * 4096 + head * 256) * 4, "gate r {r}");
            assert_eq!(r << 9, (row * 4096 + head * 256) * 2, "q16 r {r}");
            assert_eq!((r >> 4) << 8, row * 64 * 4, "angles r {r}");
        }
        let top = (rows * 16 - 1) << 11;
        assert!(
            top + 2048 <= rows * 8192 * 4 && top + 2048 <= u64::from(u32::MAX),
            "rows {rows}: 32-bit byte offsets"
        );
    }
}

/// The exits: before any access, `rows` outside `0..=8192` (unsigned compare) and a wave past `rows * 16`.
#[test]
fn tail_and_domain_exits() {
    let text = kernel_text();
    let all = core(&text);
    let guard = [
        "s_cmp_gt_u32 s18, 0x2000",
        "s_cbranch_scc1 .Lfaqp_end",
        "s_lshl_b32 s22, s18, 4",
        "s_lshl_b32 s21, ttmp9, 2",
        "s_add_co_i32 s21, s21, s20",
        "s_cmp_ge_i32 s21, s22",
        "s_cbranch_scc1 .Lfaqp_end",
    ];
    let at = all
        .windows(guard.len())
        .position(|w| w == guard)
        .expect("guard sequence");
    let first_access = all
        .iter()
        .position(|l| mnemonic(l).starts_with("global_"))
        .unwrap();
    assert!(
        at + guard.len() <= first_access,
        "exits precede every global access"
    );
    assert_eq!(
        all.iter().filter(|l| l.starts_with("s_cbranch")).count(),
        2,
        "two exits and nothing else branches"
    );
    assert_eq!(*all.last().unwrap(), "s_endpgm");
    // Host model of the grid (4 * rows blocks of four waves): exactly the (row, head) rows are live, each once.
    let live = |rows: i32, block: u32, wave: u32| -> Option<u32> {
        if (rows as u32) > 8192 {
            return None;
        }
        let r = (4 * block + wave) as i32;
        (r < rows << 4).then_some(r as u32)
    };
    for rows in [1i32, 2, 3, 17, 64, 65, 513, 8192] {
        let blocks = 4 * rows as u32 + 3; // overlaunch
        let mut seen = vec![0u8; (rows * 16) as usize];
        for b in 0..blocks {
            for w in 0..4 {
                if let Some(r) = live(rows, b, w) {
                    seen[r as usize] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "rows {rows}");
    }
    for rows in [0i32, -1, i32::MIN, 8193, 1 << 20, i32::MAX] {
        assert!(
            (0..40).all(|b| (0..4).all(|w| live(rows, b, w).is_none())),
            "rows {rows}"
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
#[test]
fn contract_is_derived_from_the_emission() {
    let (entries, _) = emitted();
    let c = fa_qprep::contract(ARCH, &entries[0]);
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
        (Some(56), Some(104))
    );
    for m in [
        "ds_load_b128",
        "ds_store_b32",
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
    assert_eq!(counts["global_load_b32"], 26);
    assert_eq!(
        (
            counts["global_store_b16"].as_u64(),
            counts["global_store_b32"].as_u64()
        ),
        (Some(8), Some(8))
    );
    assert_eq!(
        (counts["v_s_rsq_f32"].as_u64(), counts["s_endpgm"].as_u64()),
        (Some(1), Some(1))
    );
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
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel fa_qprep_pm --arch gfx1201 ... --bundle kernels/{MODULE}.hxaco)"));
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
    let fresh = fa_qprep::contract(ARCH, &entries[0]);
    let file = format!("{}/fa_qprep_pm.gfx1201.contract.json", kernels_dir());
    let committed: Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
            .unwrap();
    assert_eq!(
        committed, fresh,
        "{file}: committed contract differs from the derived contract"
    );
    let file = format!("{}/fa_qprep_pm.gfx1201.m7.json", kernels_dir());
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
            "peacemaker-isa-faqprep-{tag}-{}",
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
            serde_json::from_value(fa_qprep::contract(ARCH, &entries[0]))
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
        let file = format!("{}/fa_qprep_pm.gfx1201.m7.json", kernels_dir());
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
