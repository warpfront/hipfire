// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A3B path-parity PM combine v2 (gfx1201 only): the ordered BF16 grouped expert combine with eight
//! contiguous columns per lane, `moe_down_combine_grouped_bf16in_v2_pm_gfx1201` in module
//! `moe_slot_ops_v2_pm_gfx1201`. v1 (`moe_slot_ops_pm`, its contract and bundle) is frozen and tested by
//! `tests/moe_slot_ops.rs`.
//!
//! Law (per `t = workgroup_id_x`, columns `m0..m0+8` with `m0 = workgroup_id_y * 2048 + 8 * tid`; return when
//! `t >= n_tokens`, `2048 * workgroup_id_y >= dim`, `dim % 2048 != 0` or `k_top` is not in `1..=8`): `acc[i] = +0.0f32`;
//! for `k` in `0..k_top` ascending, `slot = inverse_perm[t*k_top+k]`, skipped when `slot as u32 >= grouped_rows`
//! (negative = unrouted), else `acc[i] = fma(topk_weights[t*k_top+k], f32::from_bits((y[slot*dim+m0+i] as u32) << 16), acc[i])`;
//! then `x_residual[t*dim+m0+i] = RN(x_residual[t*dim+m0+i] + acc[i])`. The per-column law is the v1 law.
//!
//! The emitted program runs on the PeaceMaker CPU interpreter (256-thread workgroups, one per
//! `(token, 2048-column block)`) against a host reference built from `f32::mul_add`. Every input buffer is
//! mapped at its exact size with a distinct poison region directly behind it, so a read or write past a slot /
//! column / token / `k_top` bound changes a compared byte. No GPU.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_isa::kernels::moe_slot_ops as ops;
use peacemaker_isa::Arch;
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "moe_slot_ops_v2_pm_gfx1201";
const SYMBOL: &str = "moe_down_combine_grouped_bf16in_v2_pm_gfx1201";
const V1_MODULE: &str = "moe_slot_ops_pm_gfx1201";
const V1_SYMBOL: &str = "moe_down_combine_grouped_bf16in_pm_gfx1201";
const BLOCK: u32 = 256;
const COLS: usize = 2048;
const KERNARG_BYTES: usize = 48;
/// Kernarg byte offsets (identical to v1): y, inverse_perm, topk_weights, x_residual, dim, k_top, grouped_rows, n_tokens.
const OFFSETS: [u32; 8] = [0, 8, 16, 24, 32, 36, 40, 44];

fn bundle_path() -> String {
    format!(
        "{}/../../kernels/{MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn count(text: &str, mnemonic: &str) -> usize {
    text.lines()
        .filter(|l| l.trim().split_whitespace().next() == Some(mnemonic))
        .count()
}

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    let (a, b) = (ops::emit_v2(ARCH).unwrap(), ops::emit_v2(ARCH).unwrap());
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(ops::symbol_v2(ARCH), SYMBOL);
    assert_eq!(ops::module_v2(ARCH), MODULE);
    assert_eq!(ops::KERNEL_ID_V2, "moe_slot_ops_v2_pm");
    assert!(a.s_text.contains(&format!("{SYMBOL}:")));
    assert_eq!(a.shape.barriers, 0);
    assert!(
        a.shape.next_free_vgpr <= 24 && a.shape.next_free_sgpr <= 44,
        "{:?}",
        a.shape
    );
    let (m1, t1, p1) = ops::emit_module_v2(ARCH).unwrap();
    let (m2, t2, p2) = ops::emit_module_v2(ARCH).unwrap();
    assert_eq!(m1.len(), 1);
    assert_eq!(m2.len(), 1);
    assert_eq!(t1, t2);
    assert_eq!(m1[0].s_text, m2[0].s_text);
    assert_eq!(m1[0].s_text, a.s_text);
    assert_eq!(
        serde_json::to_string(&p1).unwrap(),
        serde_json::to_string(&p2).unwrap()
    );
    assert!(t1.contains(&format!("{SYMBOL}:")));
    assert!(!t1.contains(&format!("{V1_SYMBOL}:")), "v2 module holds v1");
    // Every other architecture is refused, not silently emitted.
    for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942] {
        assert!(ops::emit_v2(arch).is_err(), "{arch:?}");
        assert!(ops::emit_module_v2(arch).is_err(), "{arch:?}");
    }
}

/// v2 is alongside v1: v1's module, symbol and emission are unaffected and distinct from v2's.
#[test]
fn v2_is_a_separate_module_from_v1() {
    assert_ne!(ops::module(ARCH), ops::module_v2(ARCH));
    assert_ne!(ops::symbol(ARCH), ops::symbol_v2(ARCH));
    assert_eq!(ops::module(ARCH), V1_MODULE);
    assert_eq!(ops::symbol(ARCH), V1_SYMBOL);
    let v1 = ops::emit_module(ARCH).unwrap().1;
    assert!(v1.contains(&format!("{V1_SYMBOL}:")));
    assert!(!v1.contains(&format!("{SYMBOL}:")));
    assert_ne!(
        bundle_path(),
        format!(
            "{}/../../kernels/{V1_MODULE}.hxaco",
            env!("CARGO_MANIFEST_DIR")
        )
    );
}

/// Kernarg ABI of the v2 combine, identical to v1 (48 bytes): y_grouped_bf16, inverse_perm, topk_weights,
/// x_residual (four pointers), then dim, k_top, grouped_rows, n_tokens (four u32). Block 256, wave32, and zero
/// LDS / private / spills.
#[test]
fn kernarg_layout_and_resources_match_the_launcher() {
    let text = ops::emit_module_v2(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let got: Vec<u32> = metadata
        .lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse::<u32>().unwrap())
        })
        .collect();
    assert_eq!(got, OFFSETS);
    assert!(text.contains(&format!(".amdhsa_kernarg_size {KERNARG_BYTES}\n")));
    assert!(text.contains(".amdhsa_group_segment_fixed_size 0\n"), "LDS");
    assert!(
        text.contains(".amdhsa_private_segment_fixed_size 0\n"),
        "private"
    );
    assert!(text.contains(".group_segment_fixed_size: 0\n"), "LDS md");
    assert!(
        text.contains(".private_segment_fixed_size: 0\n"),
        "private md"
    );
    assert!(text.contains(".vgpr_spill_count: 0\n"), "vgpr spills");
    assert!(text.contains(".sgpr_spill_count: 0\n"), "sgpr spills");
    assert!(text.contains(&format!(".max_flat_workgroup_size: {BLOCK}")));
    assert!(text.contains(".wavefront_size: 32"));
    assert!(text.contains(".symbol: ") && text.contains(&format!("{SYMBOL}.kd")));
}

/// Instruction census: eight fused multiply-adds (the loop body, one per column, never contracted), eight
/// residual adds, one `buffer_load_b128` of the BF16 group and two of the residual, two residual
/// `buffer_store_b128`, no scalar-column buffer op, no atomics, no LDS, no scratch, no barrier; the single
/// `s_endpgm`.
#[test]
fn census_has_eight_fma_body_and_b128_memory_only() {
    let e = ops::emit_v2(ARCH).unwrap();
    let fma = count(&e.s_text, "v_fma_f32") + count(&e.s_text, "v_fmac_f32");
    assert_eq!(fma, 8, "eight fma bodies expected");
    assert_eq!(count(&e.s_text, "v_fmac_f32"), 0);
    assert_eq!(count(&e.s_text, "v_mul_f32"), 0, "no separate multiply");
    assert_eq!(
        count(&e.s_text, "v_cndmask_b32_e64"),
        0,
        "no dummy-load select"
    );
    for (mnemonic, n) in [
        ("buffer_load_b128", 3),
        ("buffer_store_b128", 2),
        ("buffer_load_u16", 0),
        ("buffer_load_b32", 0),
        ("buffer_store_b32", 0),
        ("s_load_b32", 2),
        ("s_load_b128", 1),
        ("s_load_b256", 1),
        ("v_add_f32_e32", 8),
        ("v_and_b32_e32", 4),
        ("s_endpgm", 1),
    ] {
        assert_eq!(count(&e.s_text, mnemonic), n, "{mnemonic}");
    }
    assert_eq!(count(&e.s_text, "s_barrier_signal"), 0);
    assert_eq!(count(&e.s_text, "s_barrier_wait"), 0);
    assert!(!e.s_text.contains("atomic"));
    for l in e.s_text.lines().map(str::trim_start) {
        assert!(
            !(l.starts_with("scratch_")
                || l.starts_with("ds_")
                || l.starts_with("s_setpc")
                || l.starts_with("s_swappc")
                || l.starts_with("v_writelane")),
            "{l}"
        );
    }
    // Slot and weight come from SMEM, uniform per wave: no VMEM load other than the three b128.
    assert_eq!(
        e.s_text
            .lines()
            .filter(|l| l.trim_start().starts_with("buffer_load"))
            .count(),
        3
    );
    // The fma order is `fma(w, v, acc)` with the SGPR weight as the first source: eight of them.
    assert_eq!(
        e.s_text
            .lines()
            .filter(|l| l.trim_start().starts_with("v_fma_f32 v") && l.contains(", s"))
            .count(),
        8
    );
}

/// Host-side shape validation: `dim` a positive multiple of 2048 and `k_top` in `1..=8`; the grid is
/// `[n_tokens, dim / 2048, 1]`.
#[test]
fn host_shape_validation_matches_the_kernel_domain() {
    assert_eq!(ops::COLS_PER_LANE_V2, 8);
    assert_eq!(ops::COLS_PER_BLOCK_V2 as usize, COLS);
    assert_eq!(ops::validate_shape(2048, 8, 5), Ok([5, 1, 1]));
    assert_eq!(ops::validate_shape(4096, 1, 1), Ok([1, 2, 1]));
    assert_eq!(ops::validate_shape(6144, 4, 0), Ok([0, 3, 1]));
    for dim in [0u32, 1, 256, 1024, 2047, 2049, 4095, 4097, 3 * 2048 + 8] {
        assert!(ops::validate_shape(dim, 8, 1).is_err(), "dim {dim}");
    }
    for k_top in [0u32, 9, 16, u32::MAX] {
        assert!(
            ops::validate_shape(2048, k_top, 1).is_err(),
            "k_top {k_top}"
        );
    }
}

/// The committed `kernels/moe_slot_ops_v2_pm_gfx1201.hxaco` (the object `rdna-compute` embeds as
/// `MOE_SLOT_OPS_V2_PM_GFX1201`) must be exactly what the builder emits today (native writer, no ROCm needed;
/// `peacemaker-isa emit --kernel moe_slot_ops_v2_pm --arch gfx1201 ... --bundle`).
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = bundle_path();
    let committed = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel moe_slot_ops_v2_pm --arch gfx1201 ... --bundle)")
    });
    let (_, text, _) = ops::emit_module_v2(ARCH).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
}

/// v1 is untouched by v2: its committed bundle is still exactly the v1 emission.
#[test]
fn v1_committed_bundle_is_unchanged_by_v2() {
    let path = format!(
        "{}/../../kernels/{V1_MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (_, text, _) = ops::emit_module(ARCH).unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(committed == fresh, "{V1_MODULE}: v1 bundle changed");
}

/// The same `.text` through the ROCm oracle's `llvm-mc` + `ld.lld`.
#[test]
fn committed_bundle_text_equals_oracle_link() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let committed = std::fs::read(bundle_path()).unwrap();
    let text = ops::emit_module_v2(ARCH).unwrap().1;
    assert!(
        text_section(&link(&text, ARCH.name(), MODULE)) == text_section(&committed),
        "{MODULE}: oracle-linked emission differs from the committed bundle"
    );
}

/// The committed shape contract is exactly what `contract_v2()` derives from fresh emission
/// (`peacemaker-isa emit --kernel moe_slot_ops_v2_pm ... --shape crates/peacemaker-isa/kernels/moe_slot_ops_v2_pm.gfx1201.contract.json`).
#[test]
fn committed_contract_equals_fresh_contract() {
    let path = format!(
        "{}/kernels/moe_slot_ops_v2_pm.{}.contract.json",
        env!("CARGO_MANIFEST_DIR"),
        ARCH.name()
    );
    let committed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
            .unwrap();
    let emitted = ops::emit_v2(ARCH).unwrap();
    let fresh = ops::contract_v2(ARCH, &emitted);
    assert_eq!(committed, fresh, "{path} is stale");
    assert_eq!(committed["symbol"], SYMBOL);
    assert_eq!(committed["group_segment_fixed_bytes"], 0);
    assert_eq!(committed["launch_dynamic_lds_bytes"], 0);
    assert_eq!(committed["require_wave32"], true);
    assert_eq!(committed["require_zero_private"], true);
    assert_eq!(committed["require_zero_spills"], true);
    assert_eq!(committed["vgpr_max"], 24);
    let forbidden: Vec<&str> = committed["forbidden"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for want in ["scratch_*", "ds_*", "buffer_atomic_*"] {
        assert!(forbidden.contains(&want), "contract does not forbid {want}");
    }
    // The contract pins the census asserted above.
    for (name, n) in [
        ("s_endpgm", 1),
        ("v_fma_f32", 8),
        ("v_add_f32_e32", 8),
        ("buffer_load_b128", 3),
        ("buffer_store_b128", 2),
        ("buffer_load_u16", 0),
        ("buffer_load_b32", 0),
        ("buffer_store_b32", 0),
    ] {
        assert_eq!(committed["counts"][name], n, "contract pins {name}");
    }
    // v1's contract is a different file with different pins and is not overwritten.
    let v1: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!(
            "{}/kernels/moe_slot_ops_pm.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name()
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(v1["symbol"], V1_SYMBOL);
    assert_eq!(v1["counts"]["v_fma_f32"], 1);
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!(
        "peacemaker-isa-moe-slot-ops-v2-{}",
        std::process::id()
    ));
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

// ---- CPU interpreter vs the host reference ------------------------------------

fn lift() -> peacemaker_ir::Program {
    let text = ops::emit_module_v2(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
    .program
}
static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(lift);

/// Every instruction of the module has interpreter semantics. A failure names the exact missing opcodes.
#[test]
fn interpreter_covers_the_module() {
    let program = &*PROGRAM;
    let mut missing = std::collections::BTreeSet::new();
    for k in &program.kernels {
        for id in &k.body.layout {
            let i = k.body.insts.get(*id).unwrap();
            let row = peacemaker_ir::isa::lookup(program.target.arch, i.op, i.form).unwrap();
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                missing.insert(row.name);
            }
        }
    }
    assert!(missing.is_empty(), "no interpreter semantics: {missing:?}");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Full-mantissa finite f32 with a random sign and an exponent in `lo..=hi` (unbiased).
    fn finite(&mut self, lo: i32, hi: i32) -> f32 {
        let e = lo + self.below((hi - lo + 1) as usize) as i32;
        let sign = (self.next() & 1) as u32;
        let mant = (self.next() & 0x7f_ffff) as u32;
        f32::from_bits(sign << 31 | ((127 + e) as u32) << 23 | mant)
    }
    fn bf16(&mut self, lo: i32, hi: i32) -> u16 {
        (self.finite(lo, hi).to_bits() >> 16) as u16
    }
}

/// One launch of the combine: all data, the launch grid and the expected geometry.
struct Case {
    dim: usize,
    k_top: usize,
    grouped_rows: usize,
    n_tokens: usize,
    /// `[grouped_rows * dim]` BF16 bits.
    y: Vec<u16>,
    /// `[n_tokens * k_top]`.
    inverse_perm: Vec<i32>,
    /// `[n_tokens * k_top]`.
    weights: Vec<f32>,
    /// `[n_tokens * dim]`.
    x: Vec<f32>,
    /// Workgroups launched past `n_tokens` in x and past `ceil(dim/2048)` in y.
    extra_t: usize,
    extra_y: usize,
}

/// The host reference of the kernel's law (`fused = false` swaps in the unfused multiply + add and
/// `reverse = true` the descending slot order; both only to prove a directed case discriminates).
fn reference_with(c: &Case, fused: bool, reverse: bool) -> Vec<u32> {
    let mut out = Vec::with_capacity(c.n_tokens * c.dim);
    for t in 0..c.n_tokens {
        for m in 0..c.dim {
            let mut acc = 0.0f32;
            let ks: Vec<usize> = if reverse {
                (0..c.k_top).rev().collect()
            } else {
                (0..c.k_top).collect()
            };
            for k in ks {
                let slot = c.inverse_perm[t * c.k_top + k];
                if (slot as u32) >= c.grouped_rows as u32 {
                    continue;
                }
                let w = c.weights[t * c.k_top + k];
                let v = f32::from_bits((c.y[slot as usize * c.dim + m] as u32) << 16);
                acc = if fused {
                    w.mul_add(v, acc)
                } else {
                    w * v + acc
                };
            }
            out.push((c.x[t * c.dim + m] + acc).to_bits());
        }
    }
    out
}
fn reference(c: &Case) -> Vec<u32> {
    reference_with(c, true, false)
}

/// Bitwise equality, except that any two NaNs agree (payload and sign of a NaN are an interpreter /
/// libm detail, not a contract).
fn same(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

const TAIL: usize = 4096;
/// Tail fill of BF16 / f32 buffers: all-ones is a NaN in both.
const POISON: u8 = 0xff;
/// Tail fill of the permutation: slot 0, a valid row (all-ones would be the skipped `-1`).
const PERM_TAIL: u8 = 0x00;
const KERNARG_VA: u64 = 0x1_0000;
const PERM_VA: u64 = 0x10_0000;
const WEIGHTS_VA: u64 = 0x20_0000;
// y and x sit above 4 GiB so a 32-bit truncated base is caught.
const Y_VA: u64 = 0x7f00_0000_0000;
const X_VA: u64 = 0x7e00_0000_0000;

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// Maps `bytes` at `va` (skipped when empty) and a `TAIL`-byte region of `fill` directly behind it.
fn map_with_tail(mem: &mut Memory, name: &str, va: u64, bytes: Vec<u8>, fill: u8) {
    let len = bytes.len() as u64;
    if len > 0 {
        mem.map(name.into(), va, bytes).unwrap();
    }
    mem.map(format!("{name}_tail"), va + len, vec![fill; TAIL])
        .unwrap();
}

/// One full grid. Returns the residual after the launch; asserts that every other buffer (kernarg, inputs, the
/// poison behind every buffer) is untouched.
fn run(name: &str, c: &Case) -> Vec<u32> {
    assert_eq!(c.y.len(), c.grouped_rows * c.dim);
    assert_eq!(c.inverse_perm.len(), c.n_tokens * c.k_top);
    assert_eq!(c.weights.len(), c.n_tokens * c.k_top);
    assert_eq!(c.x.len(), c.n_tokens * c.dim);
    let emu = Emulator::new(&PROGRAM, SYMBOL).unwrap();
    let mut kern = Vec::new();
    for p in [Y_VA, PERM_VA, WEIGHTS_VA, X_VA] {
        kern.extend(p.to_le_bytes());
    }
    for v in [c.dim, c.k_top, c.grouped_rows, c.n_tokens] {
        kern.extend((v as u32).to_le_bytes());
    }
    assert_eq!(kern.len(), KERNARG_BYTES);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), KERNARG_VA, kern.clone()).unwrap();
    map_with_tail(&mut mem, "y", Y_VA, bytes_u16(&c.y), POISON);
    // The permutation tail reads as slot 0 (valid) and the weight tail as NaN: any read past the
    // `n_tokens * k_top` entries turns the output into NaN.
    map_with_tail(
        &mut mem,
        "perm",
        PERM_VA,
        bytes_i32(&c.inverse_perm),
        PERM_TAIL,
    );
    map_with_tail(
        &mut mem,
        "weights",
        WEIGHTS_VA,
        bytes_f32(&c.weights),
        POISON,
    );
    map_with_tail(&mut mem, "x", X_VA, bytes_f32(&c.x), POISON);
    let grid_x = c.n_tokens + c.extra_t;
    let grid_y = c.dim.div_ceil(COLS) + c.extra_y;
    for wy in 0..grid_y {
        for wx in 0..grid_x {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG_VA,
                    workgroup: [wx as u32, wy as u32, 0],
                    block: [BLOCK, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "{name}: dim {} k_top {} rows {} tokens {} wg ({wx},{wy}): {e}",
                    c.dim, c.k_top, c.grouped_rows, c.n_tokens
                )
            });
        }
    }
    let region = |n: &str| -> Vec<u8> {
        mem.regions
            .iter()
            .find(|r| r.name == n)
            .unwrap_or_else(|| panic!("region {n}"))
            .bytes
            .clone()
    };
    assert_eq!(region("kernarg"), kern, "{name}: kernarg modified");
    if !c.y.is_empty() {
        assert_eq!(region("y"), bytes_u16(&c.y), "{name}: y_grouped modified");
    }
    if !c.inverse_perm.is_empty() {
        assert_eq!(
            region("perm"),
            bytes_i32(&c.inverse_perm),
            "{name}: inverse_perm modified"
        );
        assert_eq!(
            region("weights"),
            bytes_f32(&c.weights),
            "{name}: topk_weights modified"
        );
    }
    for (tail, fill) in [
        ("y_tail", POISON),
        ("perm_tail", PERM_TAIL),
        ("weights_tail", POISON),
        ("x_tail", POISON),
    ] {
        assert!(
            region(tail).iter().all(|&b| b == fill),
            "{name}: {tail} written"
        );
    }
    if c.x.is_empty() {
        return Vec::new();
    }
    words(&region("x"))
}

fn check(name: &str, c: &Case) {
    let got = run(name, c);
    let want = reference(c);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            same(g, w),
            "{name}: dim {} k_top {} rows {} tokens {} extra ({},{}): token {} col {}: got {g:#010x} want {w:#010x} (perm {:?})",
            c.dim,
            c.k_top,
            c.grouped_rows,
            c.n_tokens,
            c.extra_t,
            c.extra_y,
            i / c.dim,
            i % c.dim,
            &c.inverse_perm[(i / c.dim) * c.k_top..(i / c.dim + 1) * c.k_top]
        );
    }
}

/// A random slot: mostly routed, with unrouted (-1), the first out-of-range row, far out-of-range
/// rows and negative extremes (all skipped by the unsigned compare).
fn slot(rng: &mut Rng, rows: usize) -> i32 {
    match rng.below(20) {
        0 | 1 => -1,
        2 => rows as i32,
        3 => rows as i32 + 1 + rng.below(1000) as i32,
        4 => i32::MIN,
        5 => i32::MAX,
        6 => -2 - rng.below(1 << 20) as i32,
        _ if rows == 0 => -1,
        _ => rng.below(rows) as i32,
    }
}

fn random_case(rng: &mut Rng, dim: usize, k_top: usize, n_tokens: usize, rows: usize) -> Case {
    Case {
        dim,
        k_top,
        grouped_rows: rows,
        n_tokens,
        y: (0..rows * dim).map(|_| rng.bf16(-8, 8)).collect(),
        inverse_perm: (0..n_tokens * k_top).map(|_| slot(rng, rows)).collect(),
        weights: (0..n_tokens * k_top).map(|_| rng.finite(-6, 2)).collect(),
        x: (0..n_tokens * dim).map(|_| rng.finite(-10, 10)).collect(),
        extra_t: 0,
        extra_y: 0,
    }
}

/// `k_top` 1..=8 at one, two and three 2048-column blocks, with the grid over-launched in tokens and in
/// column blocks on some cases.
#[test]
fn emulated_launches_equal_the_host_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut cases = 0usize;
    for k_top in 1..=8usize {
        for dim in [2048usize, 4096, 6144] {
            for trial in 0..2 {
                let n_tokens = 1 + rng.below(if dim == 2048 { 3 } else { 2 });
                let rows = 1 + rng.below(12);
                let mut c = random_case(&mut rng, dim, k_top, n_tokens, rows);
                c.extra_t = if trial == 0 { 0 } else { 1 + rng.below(2) };
                c.extra_y = rng.below(2);
                check(&format!("random k_top {k_top} trial {trial}"), &c);
                cases += 1;
            }
        }
    }
    println!("emulated_launches_equal_the_host_reference: {cases} launches");
}

/// Routed-only launches (every slot valid, the production shape: each token's `k_top` slots index
/// distinct grouped rows) at the A3B width.
#[test]
fn emulated_all_routed_a3b_width_equals_the_host_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let (dim, k_top, n_tokens) = (2048usize, 8usize, 2usize);
    let rows = n_tokens * k_top;
    let mut perm: Vec<i32> = (0..rows as i32).collect();
    for i in (1..perm.len()).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let mut c = random_case(&mut rng, dim, k_top, n_tokens, rows);
    c.inverse_perm = perm;
    check("a3b width", &c);
}

/// A lane's eight columns and the lane order inside a workgroup: every `y` element is its own column id, so a
/// wrong lane or element mapping (swapped halves, shifted lanes, a repeated column) changes a compared word.
#[test]
fn emulated_every_column_has_a_distinct_value() {
    let (dim, k_top, n_tokens, rows) = (4096usize, 2usize, 2usize, 2usize);
    // BF16 bits 0x3f80 + (m % 128): exact small integers-of-ulp, distinct per column within each 128 run,
    // plus the row in the sign of the second row.
    let mut y = Vec::new();
    for r in 0..rows {
        for m in 0..dim {
            let bits = 0x3f80u16 + (m % 128) as u16 + (m / 128 % 16) as u16 * 128;
            y.push(if r == 1 { bits | 0x8000 } else { bits });
        }
    }
    let c = Case {
        dim,
        k_top,
        grouped_rows: rows,
        n_tokens,
        y,
        inverse_perm: vec![0, 1, 1, 0],
        weights: vec![0.5, 0.25, 1.0, 2.0],
        x: (0..n_tokens * dim).map(|i| i as f32 * 0.0078125).collect(),
        extra_t: 0,
        extra_y: 0,
    };
    check("distinct columns", &c);
}

/// Negative, out-of-range and wrapping slots only; zero grouped rows; one grouped row. Skipped
/// slots add nothing (the residual becomes `x + 0.0`), so `-0` residuals become `+0`.
#[test]
fn emulated_unrouted_and_out_of_range_slots_are_skipped() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    for rows in [0usize, 1, 3] {
        let (dim, k_top, n_tokens) = (2048usize, 8usize, 3usize);
        let mut c = random_case(&mut rng, dim, k_top, n_tokens, rows);
        let bad = [
            -1,
            i32::MIN,
            i32::MAX,
            rows as i32,
            rows as i32 + 1,
            -(rows as i32) - 1,
            1 << 30,
            -(1 << 30),
        ];
        c.inverse_perm = (0..n_tokens * k_top).map(|i| bad[i % bad.len()]).collect();
        c.x[0] = -0.0;
        check(&format!("all skipped rows {rows}"), &c);
        let got = run("all skipped", &c);
        for (g, x) in got.iter().zip(&c.x) {
            assert!(same(*g, (x + 0.0).to_bits()), "skipped slot changed x");
        }
        assert_eq!(got[0], 0.0f32.to_bits(), "-0 residual + skipped = +0");
    }
}

/// One slot valid per token among skipped ones, at every `k` position (the guard applies per entry
/// and the loop covers all `k_top` entries, ascending).
#[test]
fn emulated_single_live_slot_at_every_k_position() {
    let mut rng = Rng(0xA5A5_5A5A_1234_8765);
    for k_top in 1..=8usize {
        for live_k in 0..k_top {
            let (dim, n_tokens, rows) = (2048usize, 2usize, 4usize);
            let mut c = random_case(&mut rng, dim, k_top, n_tokens, rows);
            for t in 0..n_tokens {
                for k in 0..k_top {
                    c.inverse_perm[t * k_top + k] = if k == live_k {
                        rng.below(rows) as i32
                    } else {
                        -1
                    };
                }
            }
            check(&format!("live k {live_k}"), &c);
        }
    }
}

/// `k_top` outside `1..=8` (the host never launches there): the whole grid returns before any access, so
/// the residual is bitwise untouched (NaN residuals included) and no buffer or tail is read into it or written.
#[test]
fn emulated_k_top_outside_domain_writes_nothing() {
    let mut rng = Rng(0x7777_1111_3333_5555);
    for k_top in [0usize, 9, 16, 33] {
        for rows in [0usize, 4] {
            let mut c = random_case(&mut rng, 2048, k_top, 2, rows);
            c.x[0] = f32::NAN;
            c.x[1] = -0.0;
            c.extra_t = 1;
            let got = run(&format!("k_top {k_top}"), &c);
            let want: Vec<u32> = c.x.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, want, "k_top {k_top} rows {rows}: residual written");
        }
    }
}

/// `dim` outside the domain (zero, or not a multiple of 2048; the host never launches there): every
/// workgroup returns before any access, including the full blocks of a ragged dim, so the residual is
/// bitwise untouched and nothing is read or written. The grid is over-launched by one column block so a
/// zero `dim` still runs a workgroup.
#[test]
fn emulated_dim_outside_domain_writes_nothing() {
    let mut rng = Rng(0x2222_4444_6666_8888);
    for dim in [0usize, 1, 7, 256, 1000, 2047, 2049, 4095, 4097, 2048 + 8] {
        let mut c = random_case(&mut rng, dim, 3, 2, 3);
        if let Some(x) = c.x.first_mut() {
            *x = f32::NAN;
        }
        c.extra_t = 1;
        c.extra_y = 1;
        let got = run(&format!("dim {dim}"), &c);
        let want: Vec<u32> = c.x.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "dim {dim}: residual written");
    }
}

/// The accumulation order is ascending `k` from +0 with a fused multiply-add: a directed case whose
/// descending-order and unfused references differ from the true one, so only the real law passes. Run on a
/// whole 2048-column block so every one of the eight columns of a lane takes the chain.
#[test]
fn emulated_order_and_fusion_are_the_ascending_fma_chain() {
    // Order: 1 + 1e8 - 1e8 = 0 ascending but 1 descending (-1e8 + 1e8 first, then + 1).
    let big = 0x4cbe_u16; // 1.17e8 as BF16: ulp 8, so adding 1.0 is absorbed
    let dim = COLS;
    let mut rows_y = vec![0u16; 3 * dim];
    for m in 0..dim {
        rows_y[m] = 0x3f80;
        rows_y[dim + m] = big;
        rows_y[2 * dim + m] = big | 0x8000;
    }
    let c = Case {
        dim,
        k_top: 3,
        grouped_rows: 3,
        n_tokens: 1,
        y: rows_y,
        inverse_perm: vec![0, 1, 2],
        weights: vec![1.0, 1.0, 1.0],
        x: vec![0.0; dim],
        extra_t: 0,
        extra_y: 0,
    };
    assert_ne!(
        reference_with(&c, true, true),
        reference_with(&c, true, false),
        "directed order case does not discriminate"
    );
    assert_eq!(reference(&c), vec![0.0f32.to_bits(); dim]);
    check("order", &c);

    // Fusion: k0 gives acc = -1.0078125 (w = 1, v = -1.0078125); k1 adds w * v with w = 1 + 2^-23,
    // v = 1.0078125: the fused result keeps the product's low bits, mul-then-add does not.
    let c = Case {
        dim,
        k_top: 2,
        grouped_rows: 2,
        n_tokens: 1,
        y: [vec![0xbf81u16; dim], vec![0x3f81u16; dim]].concat(),
        inverse_perm: vec![0, 1],
        weights: vec![1.0, f32::from_bits(0x3f80_0001)],
        x: vec![0.0; dim],
        extra_t: 0,
        extra_y: 0,
    };
    assert_ne!(
        reference_with(&c, true, false),
        reference_with(&c, false, false),
        "directed fusion case does not discriminate"
    );
    check("fusion", &c);
}

/// NaN, +-inf, +-0 and exact cancellation drawn from small palettes for every operand (BF16 rows incl.
/// quiet and signalling NaN, weights, residuals), so `inf * 0`, `inf - inf`, `-0 + +0` and NaN propagation all occur.
#[test]
fn emulated_nan_inf_and_signed_zero_equal_the_host_reference() {
    let mut rng = Rng(0xC0FF_EE00_DEAD_BEEF);
    let ybits: [u16; 12] = [
        0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81, 0x3f80, 0xbf80, 0x4040,
        0xc2c8,
    ];
    let fbits: [u32; 11] = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f00_0000,
        0x4040_0000,
        0x7149_f2ca,
    ];
    for (dim, k_top) in [(2048usize, 8usize), (2048, 3), (4096, 1)] {
        for trial in 0..4 {
            let (n_tokens, rows) = (2usize, 6usize);
            let mut c = random_case(&mut rng, dim, k_top, n_tokens, rows);
            for v in c.y.iter_mut() {
                *v = ybits[rng.below(ybits.len())];
            }
            for w in c.weights.iter_mut() {
                *w = f32::from_bits(fbits[rng.below(fbits.len())]);
            }
            for x in c.x.iter_mut() {
                *x = f32::from_bits(fbits[rng.below(fbits.len())]);
            }
            c.extra_t = trial % 2;
            check(&format!("specials dim {dim} trial {trial}"), &c);
        }
    }
}

/// All-zero data with signed-zero residuals and weights: `RN(-0 + (+0 + w*v))`, the acc starts at +0.
#[test]
fn emulated_signed_zero_chain_starts_at_positive_zero() {
    let (dim, k_top, rows) = (COLS, 2usize, 2usize);
    for (w, v, x) in [
        (-0.0f32, 0x0000u16, -0.0f32),
        (0.0, 0x8000, -0.0),
        (-0.0, 0x8000, -0.0),
        (-1.0, 0x0000, -0.0),
        (1.0, 0x8000, 0.0),
    ] {
        let c = Case {
            dim,
            k_top,
            grouped_rows: rows,
            n_tokens: 1,
            y: vec![v; rows * dim],
            inverse_perm: vec![0, 1],
            weights: vec![w; k_top],
            x: vec![x; dim],
            extra_t: 0,
            extra_y: 0,
        };
        check(&format!("zero w {w:?} v {v:#06x} x {x:?}"), &c);
        // acc = fma(w, +-0, +0) = +0 in round-to-nearest: x + 0 is -0 + 0 = +0 for x = -0.
        assert_eq!(reference(&c)[0], (x + 0.0).to_bits());
    }
}

/// Overlaunch: grids with 4 extra token workgroups and 2 extra column blocks run on exactly sized
/// buffers with poisoned tails; the extra workgroups must return before any access (a read at an
/// out-of-range `t` would pull a poisoned route, weight or residual and a write would land in a tail).
#[test]
fn overlaunched_tokens_and_column_blocks_touch_nothing() {
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    for (dim, n_tokens, k_top) in [(2048usize, 1usize, 1usize), (2048, 2, 8), (4096, 3, 4)] {
        let mut c = random_case(&mut rng, dim, k_top, n_tokens, 7);
        c.extra_t = 4;
        c.extra_y = 2;
        check("overlaunch", &c);
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The symbol certifies: parse-back, independent wait replay (DS drain, no spills), the committed
    /// shape contract, the static LDS bound, and M7's lift identity plus wait/hazard/barrier/window analyses
    /// of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = ops::emit_module_v2(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-moe-slot-ops-v2-cert-{}-{}",
            ARCH.name(),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let m7 = pm_check::m7(&build.elf, ARCH.name(), SYMBOL)
            .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
        assert_eq!(
            (m7["lift"].as_str(), &m7["obligations"]),
            (Some("byte-exact"), &serde_json::json!({})),
            "{SYMBOL}: {m7}"
        );
        let path = format!(
            "{}/kernels/moe_slot_ops_v2_pm.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            ARCH.name()
        );
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json, ops::contract_v2(ARCH, &emitted[0]), "{path} is stale");
        let contract = serde_json::from_slice(&raw).unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join("manifest.json"),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
    }
}
