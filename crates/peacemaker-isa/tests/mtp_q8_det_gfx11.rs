// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP-append DET Q8_0 -> F16 weight converter builder, gfx1151
//! (`mtp_q8_0_to_f16_pm_gfx1151`): element `i` of a contiguous Q8_0 matrix (34-byte
//! blocks: F16 scale, 32 int8) becomes `F16-RNE(f32(scale) * f32(q))`. ABI, metadata,
//! the conversion recipe and the tails are pinned on the emitted source; signs, scales
//! (subnormal, normal, overflow, underflow, zero, negative zero, Inf), every int8 value,
//! block/row layout and the bounded tails are checked on the peacemaker CPU interpreter.
use peacemaker_isa::kernels::mtp_q8_det_gfx11::{self, Spec};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;
const SYMBOL: &str = "mtp_q8_0_to_f16_pm_gfx1151";
/// Kernargs: `src dst` (8-byte pointers at 0, 8) and `elems` (4-byte value at 16).
const ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 4)];

fn text() -> String {
    mtp_q8_det_gfx11::emit(Spec { arch: ARCH }).unwrap().s_text
}

/// The instruction lines of the kernel body (comments and directives dropped).
fn insns(text: &str) -> Vec<&str> {
    let start = text.find(&format!("\n{SYMBOL}:\n")).unwrap();
    let end = text.find(&format!("\n.L{SYMBOL}_end:")).unwrap();
    text[start..end]
        .lines()
        .map(|l| l.split(';').next().unwrap().trim())
        .filter(|l| !l.is_empty() && !l.starts_with('.'))
        .collect()
}

#[test]
fn spec_is_the_one_entry() {
    assert_eq!(Spec { arch: ARCH }.symbol(), SYMBOL);
    assert_eq!(Spec::module(ARCH), "mtp_q8_det_pm_gfx1151");
    let (emitted, module, proof) = mtp_q8_det_gfx11::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!(proof.module, "mtp_q8_det_pm_gfx1151");
    assert!(module.contains(&format!(".amdhsa_kernel {SYMBOL}\n")));
}

#[test]
fn entry_is_deterministic_and_targets_only_gfx1151() {
    let spec = Spec { arch: ARCH };
    let (a, b) = (
        mtp_q8_det_gfx11::emit(spec).unwrap(),
        mtp_q8_det_gfx11::emit(spec).unwrap(),
    );
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.shape.barriers, 0);
    assert_eq!(a.shape.wmma, 0);
    for arch in [Arch::Gfx1201, Arch::Gfx1100] {
        assert!(mtp_q8_det_gfx11::emit(Spec { arch }).is_err(), "{arch:?}");
        assert!(
            mtp_q8_det_gfx11::emit_module(arch).is_err(),
            "{arch:?} module"
        );
    }
}

/// The kernarg block (20 bytes: every (offset, size)) and the launch metadata: 256 threads,
/// no LDS, 16 VGPRs, wave32, workgroup id x only.
#[test]
fn kernarg_abi_and_launch_metadata_are_the_documented_ones() {
    assert_eq!(mtp_q8_det_gfx11::KERNARG_BYTES, 20);
    assert_eq!(
        (mtp_q8_det_gfx11::BLOCK_BYTES, mtp_q8_det_gfx11::BLOCK_ELEMS),
        (34, 32)
    );
    let s = text();
    assert!(s.contains(".kernarg_segment_size: 20\n"), "kernarg size");
    let field = |key: &str| -> Vec<u32> {
        s.lines()
            .filter_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().parse().unwrap())
            .collect()
    };
    assert_eq!(
        field(".offset:"),
        ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
        "kernarg offsets"
    );
    assert_eq!(
        field(".size:"),
        ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
        "kernarg sizes"
    );
    assert!(s.contains(".max_flat_workgroup_size: 256\n"));
    assert!(s.contains(".group_segment_fixed_size: 0\n"));
    assert!(s.contains(".vgpr_count: 16\n"));
    assert!(s.contains(".wavefront_size: 32\n"));
    assert!(s.contains(".sgpr_spill_count: 0\n") && s.contains(".vgpr_spill_count: 0\n"));
    let body = insns(&s).join("\n");
    for banned in [
        "scratch_",
        "flat_",
        "global_",
        "ds_",
        "v_wmma",
        "v_fma_mix",
        "s_barrier",
        "v_readlane",
        "v_writelane",
        "s_setpc",
        "s_swappc",
    ] {
        assert!(!body.contains(banned), "{banned}");
    }
}

/// The conversion recipe, in order: bounded descriptors, one u16 scale load at `34 * block`, one
/// u16 load of the aligned word holding the quant at `34 * block + 2 + (slot & 30)` (the gfx11 table has
/// no `buffer_load_u8`), scale widened exactly, the quant byte extracted (`v_bfe_u32`), biased by 128 (`xor 0x80`), converted
/// (`v_cvt_f32_u32`) and unbiased (`+ -128.0`), ONE F32 multiply, ONE F16 RNE convert
/// (`v_cvt_f16_f32`, no rounding-mode or denorm override), a 2-byte store; the only exec write
/// is the lane mask `element < elems`, which precedes every memory access.
#[test]
fn conversion_recipe_and_tails_are_pinned() {
    let s = text();
    let lines = insns(&s);
    let pos = |needle: &str| {
        lines
            .iter()
            .position(|l| l.starts_with(needle))
            .unwrap_or_else(|| panic!("missing {needle}"))
    };
    let count = |needle: &str| lines.iter().filter(|l| l.starts_with(needle)).count();
    for (needle, n) in [
        ("buffer_load_u16", 2),
        ("buffer_store_b16", 1),
        ("v_cvt_f32_f16_e64", 1),
        ("v_cvt_f32_u32_e32", 1),
        ("v_mul_f32_e32", 1),
        ("v_cvt_f16_f32_e32", 1),
        ("v_bfe_u32 v6, v6, v9, 8", 1),
        ("v_xor_b32_e32 v6, 0x80, v6", 1),
        ("v_add_f32_e32 v6, 0xc3000000, v6", 1),
        ("s_endpgm", 1),
    ] {
        assert_eq!(count(needle), n, "{needle}");
    }
    assert!(!lines.iter().any(|l| l.starts_with("v_fma")
        || l.starts_with("v_cvt_f16_f32_e64")
        || l.contains("round")
        || l.starts_with("s_setreg")
        || l.starts_with("s_round_mode")
        || l.starts_with("s_denorm_mode")));
    // Offsets: block = element >> 5, byte = 34 * block, quant word = byte + (element & 30) + 2.
    for want in [
        "v_lshrrev_b32_e32 v2, 5, v1",
        "v_and_b32_e32 v3, 30, v1",
        "v_and_b32_e32 v9, 1, v1",
        "v_lshlrev_b32_e32 v9, 3, v9",
        "v_mul_lo_u32 v2, v2, 34",
        "v_add3_u32 v4, v2, v3, 2",
        "v_lshlrev_b32_e32 v7, 1, v1",
    ] {
        assert!(lines.contains(&want), "{want}");
    }
    // Extents: ceil(elems / 32) * 34 source bytes, elems * 2 destination bytes, raw out-of-range select.
    for want in [
        "s_add_i32 s20, s8, 31",
        "s_lshr_b32 s20, s20, 5",
        "s_mul_i32 s20, s20, 34",
        "s_lshl_b32 s21, s8, 1",
        "s_mov_b32 s14, s20",
        "s_mov_b32 s18, s21",
    ] {
        assert!(lines.contains(&want), "{want}");
    }
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("s_mov_b32 s") && l.ends_with("0x31004000"))
            .count(),
        2
    );
    // Lane mask first, then loads, then the arithmetic chain in order, then the store.
    let loads: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("buffer_load_u16"))
        .map(|(i, _)| i)
        .collect();
    let (mask, and, ld16, ld8) = (
        pos("v_cmp_gt_u32_e64 s10, s8, v1"),
        pos("s_and_b32 exec_lo, exec_lo, s10"),
        loads[0],
        loads[1],
    );
    let (cvt_in, mul, cvt_out, store) = (
        pos("v_cvt_f32_f16_e64 v5, v5.l"),
        pos("v_mul_f32_e32 v5, v5, v6"),
        pos("v_cvt_f16_f32_e32 v8.l, v5"),
        pos("buffer_store_b16"),
    );
    assert!(
        mask < and && and < ld16 && and < ld8,
        "exec mask precedes every memory access"
    );
    assert!(
        ld16 < cvt_in
            && ld8 < pos("v_cvt_f32_u32_e32")
            && pos("v_cvt_f32_u32_e32") < mul
            && cvt_in < mul
            && mul < cvt_out
            && cvt_out < store
    );
    assert_eq!(
        lines.iter().filter(|l| l.contains("exec")).count(),
        1,
        "exec is written once"
    );
    assert!(
        lines[ld16].contains("s[12:15]")
            && lines[ld8].contains("s[12:15]")
            && lines[store].contains("s[16:19]")
    );
}

/// Native assemble only: the text assembles with the native encoder (no HIP, no ROCm tool).
#[test]
fn native_assembler_accepts_the_entry() {
    let s = mtp_q8_det_gfx11::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&s, ARCH).unwrap();
    assert!(elf.starts_with(b"\x7fELF"));
}

/// The runtime embeds one certified module: it must be exactly what the builder emits today
/// (`kernels/mtp_q8_det_pm_gfx1151.hxaco`, generated by `peacemaker-isa emit --kernel
/// mtp_q8_det_gfx11 --arch gfx1151 ... --co ... --bundle`).
#[test]
fn committed_bundle_equals_native_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = Spec::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = mtp_q8_det_gfx11::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    assert!(
        text_section(&elf) == text_section(&committed),
        "{module}: native emission differs from the committed bundle"
    );
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

// ------------------------------------------------------------ CPU interpreter

/// Round-to-nearest-even F32 -> F16 (finite and infinite inputs; overflow to Inf, subnormals kept).
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
            let k = 31 - m.leading_zeros();
            (s << 31) | ((k + 103) << 23) | ((m << (23 - k)) & 0x7f_ffff)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// The recipe, element by element: `F16-RNE(f32(scale) * f32(q))`.
fn model(scale: u16, q: i8) -> u16 {
    f32_to_f16(f16_to_f32(scale) * f32::from(q))
}

/// A contiguous Q8_0 image: block `b` is `scales[b]` (little endian) then `quants[32 b..32 b + 32]`.
fn q8_image(scales: &[u16], quants: &[i8]) -> Vec<u8> {
    assert_eq!(quants.len(), scales.len() * 32);
    let mut out = Vec::new();
    for (b, d) in scales.iter().enumerate() {
        out.extend(d.to_le_bytes());
        out.extend(quants[32 * b..32 * b + 32].iter().map(|&q| q as u8));
    }
    out
}
fn expected(scales: &[u16], quants: &[i8]) -> Vec<u16> {
    quants
        .iter()
        .enumerate()
        .map(|(i, &q)| model(scales[i / 32], q))
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn quant(&mut self) -> i8 {
        (self.next() % 256) as u8 as i8
    }
}
/// Quants whose first five slots are -128, 127, 0, 1, -1 (the sign and range extremes) and the rest random.
fn quants(blocks: usize, seed: u64) -> Vec<i8> {
    let mut rng = Rng(seed | 1);
    let mut q = Vec::new();
    for _ in 0..blocks {
        q.extend([-128i8, 127, 0, 1, -1]);
        for _ in 5..32 {
            q.push(rng.quant())
        }
    }
    q
}

type Lifted = peacemaker_ir::state::Lifted<peacemaker_ir::inst::Program>;
fn lifted() -> Lifted {
    let elf =
        peacemaker_isa::native::assemble(&mtp_q8_det_gfx11::emit_module(ARCH).unwrap().1, ARCH)
            .unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
}

/// Runs the whole grid on the interpreter over the Q8_0 image `src` (mapped with exactly its own
/// length: any read past it faults) for `elems` elements; returns the F16 output (untouched
/// elements stay 0xa5a5) and checks the 4 KiB poison guard after `dst` is intact.
fn run_grid(l: &Lifted, src: &[u8], elems: usize) -> Vec<u16> {
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    let emu = Emulator::new(&l.program, SYMBOL).unwrap();
    let (a_kern, a_src, a_dst) = (0x1_0000u64, 0x10_0000u64, 0x100_0000u64);
    let mut kernarg = Vec::new();
    for p in [a_src, a_dst] {
        kernarg.extend(p.to_le_bytes());
    }
    kernarg.extend((elems as u32).to_le_bytes());
    assert_eq!(kernarg.len(), 20);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), a_kern, kernarg).unwrap();
    if !src.is_empty() {
        mem.map("src".into(), a_src, src.to_vec()).unwrap();
    }
    mem.map("dst".into(), a_dst, vec![0xa5; elems * 2 + 4096])
        .unwrap();
    let groups = elems.div_ceil(256).max(1) as u32;
    for gx in 0..groups {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: a_kern,
                workgroup: [gx, 0, 0],
                block: [256, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )
        .unwrap_or_else(|e| panic!("wg({gx}): {e}"));
    }
    let y = &mem.regions.iter().find(|r| r.name == "dst").unwrap().bytes;
    assert!(
        y[elems * 2..].iter().all(|&b| b == 0xa5),
        "elems={elems}: store past the output"
    );
    y[..elems * 2]
        .chunks(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check(l: &Lifted, scales: &[u16], q: &[i8], elems: usize) {
    let src = q8_image(scales, q);
    let got = run_grid(l, &src, elems);
    let want = expected(scales, q);
    for i in 0..elems {
        assert_eq!(got[i], want[i], "elems={elems} element {i} (block {}, slot {}): scale {:#06x}, q {} -> {:#06x} (model {:#06x})",
            i / 32, i % 32, scales[i / 32], q[i], got[i], want[i]);
    }
}

/// Every F16 exponent range (zero, negative zero, subnormals, minimum normal, 1, 0.5, normal,
/// the maximum 65504 and its negation), both signs and the int8 extremes, block by block.
#[test]
fn signs_scales_and_quants_match_the_recipe() {
    let l = lifted();
    let scales: Vec<u16> = vec![
        0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x03ff, 0x0400, 0x8400, 0x3800, 0x3c00, 0xbc00,
        0x3555, 0xb555, 0x4000, 0x5640, 0x7bff, 0xfbff, 0x7800, 0x7b00, 0x77ff, 0x2e66, 0x1400,
        0x0200,
    ];
    let q = quants(scales.len(), 7);
    check(&l, &scales, &q, scales.len() * 32);
}

/// Overflow and underflow of the F16 exponent range: 65504 * 2 overflows to Inf, 65504 * 1
/// stays, 2^-24 * 127 is the subnormal 0x007f, a negative scale or quant gives the negative
/// result, and the sign of a zero result follows `d * q` (`-0` for `q = 0` with a negative or
/// negative-zero scale). The model is a bit-level RNE reference over the exact F32 product.
#[test]
fn overflow_underflow_and_zero_edges() {
    let l = lifted();
    // 0x7bff = 65504, 0x7bfe = 65472, 0x0001 / 0x0002 / 0x0003 subnormals, 0x0400 minimum normal, 0x5000 = 32.
    let scales: Vec<u16> = vec![
        0x7bff, 0xfbff, 0x7bfe, 0x0001, 0x8001, 0x0002, 0x0003, 0x0400, 0x03ff, 0x8000, 0x0000,
        0x5000,
    ];
    let q = quants(scales.len(), 99);
    check(&l, &scales, &q, scales.len() * 32);
    // Pin the headline cases directly (independent of the model's structure).
    let scales = vec![0x7bffu16, 0x0001, 0x8000];
    let mut qs = vec![0i8; 96];
    qs[0] = 2;
    qs[1] = 1;
    qs[2] = -2; // 65504*2 -> +Inf, 65504 -> 0x7bff, -Inf
    qs[32] = 1;
    qs[33] = -1;
    qs[34] = 127; // 2^-24 -> subnormal 0x0001, 0x8001, 127 * 2^-24 -> 0x007f
    qs[64] = 5;
    qs[65] = 0;
    qs[66] = -128; // -0.0 scale: every result is a signed zero: 0x8000 (5 * -0), 0x8000 (0 * -0 = -0), 0x0000 (-0 * -128 = +0)
    let src = q8_image(&scales, &qs);
    let got = run_grid(&l, &src, 96);
    assert_eq!(&got[0..3], &[0x7c00, 0x7bff, 0xfc00]);
    assert_eq!(&got[32..35], &[0x0001, 0x8001, 0x007f]);
    assert_eq!(&got[64..67], &[0x8000, 0x8000, 0x0000]);
    let _ = scales;
}

/// A non-finite scale with a nonzero quant propagates Inf with the sign of `d * q` (`Inf * 0`
/// is NaN and is covered by the doc only: the interpreter does not model signalling-NaN inputs).
#[test]
fn infinite_scale_propagates_with_nonzero_quants() {
    let l = lifted();
    let scales = vec![0x7c00u16, 0xfc00];
    let q: Vec<i8> = (0..64)
        .map(|i| match i % 5 {
            0 => 127,
            1 => -128,
            2 => 1,
            3 => -1,
            _ => 77,
        })
        .collect();
    check(&l, &scales, &q, 64);
}

/// A scale sweep over the whole F16 range (every 64th code: all exponents, both signs) against a
/// rotating quant pattern: 1024 blocks, 128 workgroups.
#[test]
fn scale_sweep_over_the_f16_range() {
    let l = lifted();
    let scales: Vec<u16> = (0..1024u32)
        .map(|i| (i * 64) as u16)
        .filter(|s| (s & 0x7c00) != 0x7c00)
        .collect();
    let q = quants(scales.len(), 1234);
    check(&l, &scales, &q, scales.len() * 32);
}

/// Tails: element counts that leave the last workgroup partial, one block, and the empty matrix.
/// The source is mapped at exactly `ceil(elems / 32) * 34` bytes (a lane past it would fault) and the
/// destination guard after `elems * 2` bytes must stay untouched.
#[test]
fn tails_read_and_write_only_inside_the_matrix() {
    let l = lifted();
    for blocks in [1usize, 2, 3, 7, 8, 9, 17] {
        let mut scales: Vec<u16> = (0..blocks).map(|b| 0x3000 + 0x0111 * b as u16).collect();
        scales[0] = 0xbc00;
        let q = quants(blocks, 31 + blocks as u64);
        check(&l, &scales, &q, blocks * 32);
    }
    // elems == 0: nothing is read or written, the guard stays intact.
    assert!(run_grid(&l, &[], 0).is_empty());
}

/// Layout: a row-major `[M, K]` weight with `K % 64 == 0` is `M * K / 32` consecutive blocks, element
/// `row * K + col` in block `(row * K + col) / 32`; a block's value never depends on its neighbours
/// or on which workgroup converts it (a row boundary falls on a block boundary, and rows straddle workgroups).
#[test]
fn matrix_layout_is_row_major_blocks() {
    let l = lifted();
    let (m, k) = (5usize, 192usize);
    let blocks = m * k / 32;
    let scales: Vec<u16> = (0..blocks).map(|b| 0x2c00 + 0x0071 * b as u16).collect();
    let q = quants(blocks, 5);
    let src = q8_image(&scales, &q);
    let got = run_grid(&l, &src, m * k);
    for row in 0..m {
        for col in 0..k {
            let idx = row * k + col;
            assert_eq!(got[idx], model(scales[idx / 32], q[idx]), "[{row}][{col}]");
        }
    }
    // The same block converted alone (block 0 of a one-block matrix) gives the same bits as inside the matrix.
    let b7 = q8_image(&scales[7..8], &q[7 * 32..8 * 32]);
    assert_eq!(run_grid(&l, &b7, 32), got[7 * 32..8 * 32]);
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The module certifies: parse-back, independent wait replay, its committed shape contract
    /// (counts, registers, zero LDS), and M7's lift identity plus wait/hazard/barrier/window
    /// analyses of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = mtp_q8_det_gfx11::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-mtpq8-cert-{}-{}",
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
            "{SYMBOL}"
        );
        let path = format!(
            "{}/kernels/mtp_q8_det_pm.gfx1151.convert.contract.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let contract =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        certify(
            &toolchain,
            &build,
            &s,
            ARCH.name(),
            &dir.join(format!("{SYMBOL}.manifest.json")),
            Some(&contract),
            "test",
            "test",
        )
        .unwrap_or_else(|e| panic!("{SYMBOL}: {e}"));
    }
}
