// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP-append DET dense GEMM builder, gfx1151 (`mtp_dense_det_pm_gfx1151_w8`): the
//! F16 x F16 -> F32 `v_wmma_f32_16x16x16_f16` GEMM over ascending 16-K steps, 128
//! channels x 256 rows per workgroup. Determinism, ABI, metadata, LDS/register
//! mapping, the WMMA chain and the tails are pinned on the emitted source; layout,
//! masks, the exchange epilogue and batch independence are checked on the
//! peacemaker CPU interpreter.
use peacemaker_isa::kernels::mtp_dense_det_gfx11::{self, Spec};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1151;
const SYMBOL: &str = "mtp_dense_det_pm_gfx1151_w8";
/// Kernargs: `A16 X16 Y32` (8-byte pointers at 0, 8, 16) and `M K N` (4-byte values at 24, 28, 32).
const ARGS: &[(u32, u32)] = &[(0, 8), (8, 8), (16, 8), (24, 4), (28, 4), (32, 4)];

fn text() -> String {
    mtp_dense_det_gfx11::emit(Spec { arch: ARCH })
        .unwrap()
        .s_text
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
/// `v[a:b]` / `vN` operand text -> first register.
fn first_reg(op: &str) -> u32 {
    let op = op
        .trim()
        .strip_prefix("v[")
        .or_else(|| op.trim().strip_prefix('v'))
        .unwrap();
    op.split([':', ']']).next().unwrap().parse().unwrap()
}
fn imm_offset(l: &str) -> u32 {
    l.split("offset:")
        .nth(1)
        .map_or(0, |v| v.split_whitespace().next().unwrap().parse().unwrap())
}

#[test]
fn spec_is_the_one_entry() {
    assert_eq!(Spec { arch: ARCH }.symbol(), SYMBOL);
    assert_eq!(Spec::module(ARCH), "mtp_dense_det_pm_gfx1151");
    let (emitted, module, proof) = mtp_dense_det_gfx11::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert_eq!(proof.module, "mtp_dense_det_pm_gfx1151");
    assert!(module.contains(&format!(".amdhsa_kernel {SYMBOL}\n")));
}

#[test]
fn entry_is_deterministic_and_targets_only_gfx1151() {
    let spec = Spec { arch: ARCH };
    let (a, b) = (
        mtp_dense_det_gfx11::emit(spec).unwrap(),
        mtp_dense_det_gfx11::emit(spec).unwrap(),
    );
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert_eq!(a.s_text, b.s_text);
    // Two chunks per 64-K iteration: 32 WMMAs each (2 K-steps x 16 row tiles); one barrier per chunk plus the bootstrap.
    assert_eq!(a.shape.wmma, 64);
    assert_eq!(a.shape.barriers, 3);
    assert_eq!(a.shape.next_free_vgpr, 216);
    let (_, ta, pa) = mtp_dense_det_gfx11::emit_module(ARCH).unwrap();
    let (_, tb, pb) = mtp_dense_det_gfx11::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.module), (tb, pb.module));
    for arch in [Arch::Gfx1201, Arch::Gfx1100] {
        assert!(
            mtp_dense_det_gfx11::emit(Spec { arch }).is_err(),
            "{arch:?}"
        );
        assert!(
            mtp_dense_det_gfx11::emit_module(arch).is_err(),
            "{arch:?} module"
        );
    }
}

/// The kernarg block (36 bytes: every (offset, size)) and the launch metadata: 256
/// threads, 40960 static LDS bytes, 216 VGPRs, wave32, workgroup id y.
#[test]
fn kernarg_abi_and_launch_metadata_are_the_documented_ones() {
    assert_eq!(mtp_dense_det_gfx11::KERNARG_BYTES, 36);
    let s = text();
    assert!(s.contains(".kernarg_segment_size: 36\n"), "kernarg size");
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
    assert!(s.contains(".group_segment_fixed_size: 40960\n"));
    assert_eq!(
        (
            mtp_dense_det_gfx11::SLOT_BYTES,
            mtp_dense_det_gfx11::GROUP_BYTES
        ),
        (20480, 40960)
    );
    assert!(s.contains(".vgpr_count: 216\n"));
    assert!(s.contains(".wavefront_size: 32\n"));
    assert!(s.contains("system_sgpr_workgroup_id_y 1"));
    assert!(s.contains(".sgpr_spill_count: 0\n") && s.contains(".vgpr_spill_count: 0\n"));
    let body = insns(&s).join("\n");
    for banned in [
        "scratch_",
        "flat_",
        "global_",
        "ds_load_2addr",
        "ds_store_2addr",
        "v_readlane",
        "v_writelane",
        "s_setpc",
        "s_swappc",
    ] {
        assert!(!body.contains(banned), "{banned}");
    }
}

/// The WMMA chain of the 64-K loop body: 64 WMMAs; WMMA `i` accumulates tile `i % 16`
/// in place (C = D, started from +0.0 by the prologue), its A operand is the
/// K-step's register set (set `i / 32`, step `(i % 32) / 16`: K offsets 0, 16, 32, 48
/// of the iteration, ascending) and its B operand the ring register `i % 4`; each
/// tile sees the same four K steps in ascending order every iteration, so a K = 2560
/// output is exactly 160 chained WMMAs. No lane is masked in between.
#[test]
fn wmma_chain_is_ascending_k_over_one_accumulator_per_tile() {
    let s = text();
    let lines = insns(&s);
    let wmma: Vec<(usize, &str)> = lines
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, l)| l.starts_with("v_wmma_f32_16x16x16_f16"))
        .collect();
    assert_eq!(wmma.len(), 64);
    for (i, (_, l)) in wmma.iter().enumerate() {
        let ops: Vec<&str> = l.split_once(' ').unwrap().1.split("], ").collect();
        assert_eq!(ops.len(), 4, "{l}");
        let (d, a, b, c) = (
            first_reg(ops[0]),
            first_reg(ops[1]),
            first_reg(ops[2]),
            first_reg(ops[3]),
        );
        let tile = (i % 16) as u32;
        assert_eq!(d, 8 + 8 * tile, "{l}: destination tile");
        assert_eq!(c, d, "{l}: C is D");
        assert_eq!(
            a,
            168 + 16 * (i as u32 / 32) + 8 * ((i as u32 % 32) / 16),
            "{l}: A set / K step"
        );
        assert_eq!(b, 136 + 8 * (i as u32 % 4), "{l}: B ring register");
        assert!(
            ops[0].starts_with("v[")
                && ops[1].starts_with("v[")
                && ops[2].starts_with("v[")
                && ops[3].starts_with("v["),
            "{l}: eight-VGPR operands"
        );
    }
    // Zero-initialised accumulators before the first WMMA, none after the loop's last until the epilogue.
    let first = wmma[0].0;
    for n in 8..136 {
        assert!(
            lines[..first]
                .iter()
                .any(|l| *l == format!("v_mov_b32_e32 v{n}, 0")),
            "acc v{n} not zeroed"
        );
    }
    // All lanes active: nothing between the first and last WMMA touches exec.
    assert!(
        lines[first..=wmma[63].0]
            .iter()
            .all(|l| !l.contains("exec")),
        "exec modified inside the WMMA body"
    );
    // Reductions are never split: no second accumulator set, no atomics.
    assert!(!lines.iter().any(|l| l.contains("atomic")));
}

/// LDS mapping: the B fragment loads read `slot + 1280 t + 32 j (+16)` for the 2
/// slots x 16 tiles x 2 K steps (two `ds_load_b128` each) and nothing else; X is
/// staged by `ds_store_b128` at `slot + 5120 i` (four uint4 per thread, rows 64 apart).
#[test]
fn lds_pitch_slots_and_fragment_mapping() {
    let s = text();
    let lines = insns(&s);
    let mut loads: Vec<u32> = lines
        .iter()
        .filter(|l| l.starts_with("ds_load_b128"))
        .map(|l| imm_offset(l))
        .collect();
    loads.sort_unstable();
    loads.dedup();
    let mut want: Vec<u32> = Vec::new();
    for slot in 0..2u32 {
        for t in 0..16u32 {
            for j in 0..2u32 {
                for h in 0..2u32 {
                    want.push(slot * 20480 + t * 16 * 80 + j * 32 + 16 * h);
                }
            }
        }
    }
    want.sort_unstable();
    assert_eq!(loads, want);
    let mut stores: Vec<u32> = lines
        .iter()
        .filter(|l| l.starts_with("ds_store_b128"))
        .map(|l| imm_offset(l))
        .collect();
    stores.sort_unstable();
    stores.dedup();
    assert_eq!(
        stores,
        (0..2u32)
            .flat_map(|slot| (0..4u32).map(move |i| slot * 20480 + i * 64 * 80))
            .collect::<Vec<_>>()
    );
    // Fragments are read at ml * 80 (no lane-half term: lanes l and l + 16 hold the same row), staged at (tid / 4) * 80 + (tid % 4) * 16.
    assert!(
        lines
            .iter()
            .any(|l| *l == "v_mul_u32_u24_e32 v1, 0x50, v136"),
        "B read address"
    );
    assert!(
        lines
            .iter()
            .any(|l| *l == "v_mad_u32_u24 v2, v139, 0x50, v140"),
        "X stage address"
    );
    // The A fragments are direct global loads: 16 B halves at K bytes 0..112 of the iteration (SA: chunk 0 and chunk 1; SANX: next chunk 0).
    let aloads: Vec<&&str> = lines
        .iter()
        .filter(|l| l.starts_with("buffer_load_b128") && l.contains("s[4:7]"))
        .collect();
    let mut sa: Vec<u32> = aloads
        .iter()
        .filter(|l| l.ends_with(" s28 offen") || l.contains(" s28 offen "))
        .map(|l| imm_offset(l))
        .collect();
    let mut sanx: Vec<u32> = aloads
        .iter()
        .filter(|l| l.contains(" s29 offen"))
        .map(|l| imm_offset(l))
        .collect();
    sa.sort_unstable();
    sa.dedup();
    sanx.sort_unstable();
    sanx.dedup();
    assert_eq!(sa, vec![0, 16, 32, 48, 64, 80, 96, 112]);
    assert_eq!(sanx, vec![0, 16, 32, 48]);
}

/// Loop shape: one back-edge, the pair-unrolled body advances K by 64 (128 bytes of A).
#[test]
fn pair_unrolled_loop_advances_64_k() {
    let s = text();
    let lines = insns(&s);
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("s_cbranch_scc1 .Lmtpdet_loop"))
            .count(),
        1
    );
    assert!(s.contains(".Lmtpdet_loop:"));
    assert!(
        lines.iter().any(|l| *l == "s_add_i32 s28, s28, 0x80"),
        "A K offset += 128 B per iteration"
    );
    assert!(
        lines.iter().any(|l| *l == "s_add_i32 s30, s30, 2"),
        "chunk counter += 2 per iteration"
    );
    assert_eq!(lines.iter().filter(|l| **l == "s_barrier").count(), 3);
}

/// Tails: M, K, N <= 0 and K < 64 exit before any store; the weight row is clamped to
/// M - 1 (idle waves of the last workgroup column still run every WMMA and barrier);
/// A/X/Y are bounded descriptors; stores (and only stores) run under `exec = token < N
/// & channel < M`, 32 `buffer_store_b128` (16 tiles x 2 halves, 16 B at +0 and +16).
#[test]
fn tails_are_clamped_bounded_and_store_masked() {
    let s = text();
    let lines = insns(&s);
    assert!(lines.contains(&"s_cmp_lt_i32 s33, 1"));
    assert!(lines.contains(&"s_cmp_lt_i32 s23, 64"));
    // A row clamp to M - 1: row < M ? row : M - 1 (t2 = v138 holds the row, t5 = v141 the bound).
    for want in [
        "s_add_i32 s34, s22, -1",
        "v_cmp_gt_i32_e64 s42, s22, v138",
        "v_mov_b32_e32 v141, s34",
        "v_cndmask_b32_e64 v138, v141, v138, s42",
    ] {
        assert!(lines.contains(&want), "A row clamp: {want}");
    }
    assert!(
        !lines.iter().any(|l| l.starts_with("v_min_")),
        "no vector min (no emulator semantics)"
    );
    assert!(
        lines.contains(&"s_mul_i32 s33, s22, s35"),
        "A descriptor bounded at M * K * 2"
    );
    assert!(
        lines.contains(&"s_mul_i32 s34, s24, s35"),
        "X descriptor bounded at N * K * 2"
    );
    assert!(
        lines.contains(&"s_mul_i32 s38, s38, s24"),
        "Y descriptor bounded at N * M * 4"
    );
    // Descriptor word 3: raw out-of-range select.
    assert!(
        lines
            .iter()
            .filter(|l| l.starts_with("s_mov_b32 s") && l.ends_with("0x31004000"))
            .count()
            == 3
    );
    let stores: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("buffer_store_b128"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(stores.len(), 32);
    for pair in stores.chunks(2) {
        assert_eq!(imm_offset(lines[pair[0]]), 0);
        assert_eq!(imm_offset(lines[pair[1]]), 16);
        assert!(
            lines[pair[0]].contains("s[12:15], s36 offen"),
            "{}",
            lines[pair[0]]
        );
        // Immediately under the exec mask, restored right after.
        assert!(lines[..pair[0]]
            .iter()
            .rev()
            .take(3)
            .any(|l| *l == "s_and_b32 exec_lo, s40, s42"));
        assert!(lines[pair[1] + 1..]
            .iter()
            .take(3)
            .any(|l| *l == "s_mov_b32 exec_lo, -1"));
    }
    // The only exec writes are the 16 mask/restore pairs of the stores, all after the last WMMA.
    let last_wmma = lines.iter().rposition(|l| l.starts_with("v_wmma")).unwrap();
    let exec_writes: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("s_and_b32 exec_lo") || l.starts_with("s_mov_b32 exec_lo"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(exec_writes.len(), 32);
    assert!(exec_writes.iter().all(|&i| i > last_wmma));
    // M % 16 == 0: the channel mask is one compare of M against the lane's first channel; the token mask steps by 16 per tile.
    assert!(lines
        .iter()
        .any(|l| *l == "v_cmp_gt_i32_e64 s42, s22, v143"));
    assert_eq!(
        lines
            .iter()
            .filter(|l| **l == "s_add_i32 s37, s37, -16")
            .count(),
        16
    );
    // Epilogue is `0.0f + acc` over all 128 accumulator VGPRs, F32 out (no conversion).
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("v_add_f32_e32 v") && l.contains(", 0, v"))
            .count(),
        128
    );
    assert!(!lines.iter().any(|l| l.starts_with("v_cvt_")));
}

/// Native assemble only: the text assembles with the native encoder (no HIP, no ROCm tool).
#[test]
fn native_assembler_accepts_the_entry() {
    let s = mtp_dense_det_gfx11::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&s, ARCH).unwrap();
    assert!(elf.starts_with(b"\x7fELF"));
}

/// The runtime embeds one certified module: it must be exactly what the builder
/// emits today (`kernels/mtp_dense_det_pm_gfx1151.hxaco`, generated by
/// `peacemaker-isa emit --kernel mtp_dense_det_gfx11 --arch gfx1151 ... --bundle`).
#[test]
fn committed_bundle_equals_native_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = Spec::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = mtp_dense_det_gfx11::emit_module(ARCH).unwrap().1;
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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A multiple of 1/16 in [-1, 1): exactly representable in F16 and F32, so every product and partial sum is exact in F32 (the WMMA hardware model still differs from an IEEE dot on opposing signs: see `model_output`).
    fn sixteenth(&mut self) -> f32 {
        ((self.next() % 32) as i32 - 16) as f32 / 16.0
    }
    /// An arbitrary value in (-1, 1): its dots depend on the accumulation order.
    fn wild(&mut self) -> f32 {
        ((self.next() % 20001) as i32 - 10000) as f32 / 10000.0
    }
}
fn h16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&f| f32_to_f16(f)).collect()
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

type Lifted = peacemaker_ir::state::Lifted<peacemaker_ir::inst::Program>;
fn lifted() -> Lifted {
    let elf =
        peacemaker_isa::native::assemble(&mtp_dense_det_gfx11::emit_module(ARCH).unwrap().1, ARCH)
            .unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .unwrap()
}

/// Runs the whole grid on the interpreter over F16 operands `w` `[m, k]` and `x` `[n, k]`;
/// returns the F32 `[n, m]` output bits (untouched elements stay 0xa5a5a5a5) and checks the
/// 4 KiB poison guard after `y` is intact.
fn run_grid(l: &Lifted, m: usize, k: usize, n: usize, w: &[u16], x: &[u16]) -> Vec<u32> {
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    assert_eq!((w.len(), x.len()), (m * k, n * k));
    let emu = Emulator::new(&l.program, SYMBOL).unwrap();
    let (a_kern, a_w, a_x, a_y) = (0x1_0000u64, 0x10_0000u64, 0x80_0000u64, 0x100_0000u64);
    let mut kernarg = Vec::new();
    for p in [a_w, a_x, a_y] {
        kernarg.extend(p.to_le_bytes());
    }
    for v in [m as u32, k as u32, n as u32] {
        kernarg.extend(v.to_le_bytes());
    }
    assert_eq!(kernarg.len(), 36);
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let mut mem = Memory::default();
    mem.map("kernarg".into(), a_kern, kernarg).unwrap();
    mem.map("w".into(), a_w, bytes(w)).unwrap();
    mem.map("x".into(), a_x, bytes(x)).unwrap();
    mem.map("y".into(), a_y, vec![0xa5; n * m * 4 + 4096])
        .unwrap();
    let grid = [m.div_ceil(128) as u32, n.div_ceil(256) as u32, 1];
    for gy in 0..grid[1] {
        for gx in 0..grid[0] {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: a_kern,
                    workgroup: [gx, gy, 0],
                    block: [256, 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("wg({gx},{gy}): {e}"));
        }
    }
    let y = &mem.regions.iter().find(|r| r.name == "y").unwrap().bytes;
    assert!(
        y[n * m * 4..].iter().all(|&b| b == 0xa5),
        "m={m} k={k} n={n}: store past the output"
    );
    y[..n * m * 4]
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The hardware numerics of the chain, from the interpreter's own model (`peacemaker_ir::emu::mma`, measured on
/// gfx1151 silicon: eight DOT2 steps per WMMA, rounding at every step, one's-complement opposing terms): NOT an IEEE
/// F32 dot, even for operands exactly representable in F16 and F32 (published gfx1151 words in `mma.rs`'s tests, e.g.
/// 1 ULP deficits on cancellation). Output (token `t`, channel `c`) is `K / 16` such WMMAs over ascending 16-K steps
/// chained from +0.0 bits, then the epilogue's `0.0f + acc` (v_add_f32 0, acc).
fn model_output(w: &[u16], x: &[u16], k: usize, c: usize, t: usize) -> u32 {
    let mut acc = 0u32;
    for k0 in (0..k).step_by(16) {
        acc = peacemaker_ir::emu::mma::wmma_f32_f16(
            peacemaker_ir::Arch::Gfx1151,
            &w[c * k + k0..c * k + k0 + 16],
            &x[t * k + k0..t * k + k0 + 16],
            acc,
        );
    }
    (f32::from_bits(acc) + 0.0).to_bits()
}
fn check_against_model(y: &[u32], w: &[u16], x: &[u16], m: usize, k: usize, n: usize) {
    for t in 0..n {
        for c in 0..m {
            assert_eq!(
                y[t * m + c],
                model_output(w, x, k, c, t),
                "M={m} K={k} N={n}: y[{t}][{c}] = {} (kernel) vs model {}",
                f32::from_bits(y[t * m + c]),
                f32::from_bits(model_output(w, x, k, c, t))
            );
        }
    }
}

/// Operands that are multiples of 1/16 (exact in F16 and F32): every product and partial sum is exactly representable,
/// so any channel/row/K mapping error shows as a wrong value, while the expected bits come from the WMMA numerics model.
fn check_case(l: &Lifted, m: usize, k: usize, n: usize, seed: u64) {
    let mut rng = Rng(seed | 1);
    let w = h16(&(0..m * k).map(|_| rng.sixteenth()).collect::<Vec<_>>());
    let x = h16(&(0..n * k).map(|_| rng.sixteenth()).collect::<Vec<_>>());
    let y = run_grid(l, m, k, n, &w, &x);
    check_against_model(&y, &w, &x, m, k, n);
    // Sanity only (the bits are checked above): the model chain stays within per-step rounding of the scalar F32 dot (not equal: see `model_output`).
    for t in 0..n.min(8) {
        for c in 0..m {
            let dot: f32 = (0..k)
                .map(|kk| f16_to_f32(w[c * k + kk]) * f16_to_f32(x[t * k + kk]))
                .sum();
            let mag: f32 = (0..k)
                .map(|kk| (f16_to_f32(w[c * k + kk]) * f16_to_f32(x[t * k + kk])).abs())
                .sum();
            assert!(
                (f32::from_bits(y[t * m + c]) - dot).abs() <= 1e-4 * (1.0 + mag),
                "M={m} K={k} N={n}: y[{t}][{c}] far from the scalar dot {dot}"
            );
        }
    }
}

/// Exercises the channel map (every wave and lane half), K iterations and chunk parity.
#[test]
fn interpreter_matches_the_reference_layout() {
    let l = lifted();
    // One workgroup: all eight waves x two lane halves, two 64-K iterations, ragged row count.
    check_case(&l, 128, 128, 20, 1);
    // Channel tail (idle waves clamp their weight rows, store nothing) and several workgroup columns.
    check_case(&l, 144, 192, 33, 2);
    // Row tail across two 256-row blocks (the second holds 44 rows), three iterations.
    check_case(&l, 32, 192, 300, 3);
    // One 64-K iteration (the clamped next-iteration loads re-read the only one), one row, one tile.
    check_case(&l, 16, 64, 1, 4);
}

/// DET: a row's output bits depend only on that row of X and the weights. Arbitrary (non-exact) values,
/// so any change of K-chain order or tiling would show; the row computed alone (N = 1), in a 20-row batch,
/// reversed, and at rows 255/256/257 of a 257-row batch (block boundary) are identical.
#[test]
fn row_outputs_are_batch_and_position_independent() {
    let l = lifted();
    let (m, k) = (48usize, 128usize);
    let mut rng = Rng(77);
    let w = h16(&(0..m * k).map(|_| rng.wild()).collect::<Vec<_>>());
    let x20 = h16(&(0..20 * k).map(|_| rng.wild()).collect::<Vec<_>>());
    let y20 = run_grid(&l, m, k, 20, &w, &x20);
    for r in [0usize, 7, 19] {
        let alone = run_grid(&l, m, k, 1, &w, &x20[r * k..(r + 1) * k]);
        assert_eq!(
            alone,
            y20[r * m..(r + 1) * m],
            "row {r} alone vs batch of 20"
        );
    }
    let mut rev: Vec<u16> = Vec::new();
    for r in (0..20).rev() {
        rev.extend(&x20[r * k..(r + 1) * k]);
    }
    let yrev = run_grid(&l, m, k, 20, &w, &rev);
    for r in 0..20 {
        assert_eq!(
            yrev[(19 - r) * m..(20 - r) * m],
            y20[r * m..(r + 1) * m],
            "row {r} reversed"
        );
    }
    let (m2, k2) = (16usize, 64usize);
    let w2 = h16(&(0..m2 * k2).map(|_| rng.wild()).collect::<Vec<_>>());
    let x257 = h16(&(0..257 * k2).map(|_| rng.wild()).collect::<Vec<_>>());
    let y257 = run_grid(&l, m2, k2, 257, &w2, &x257);
    for r in [255usize, 256] {
        let alone = run_grid(&l, m2, k2, 1, &w2, &x257[r * k2..(r + 1) * k2]);
        assert_eq!(
            alone,
            y257[r * m2..(r + 1) * m2],
            "row {r} alone vs batch of 257"
        );
    }
    // Every output of the 257-row and 20-row batches is the model's ascending-K16 WMMA chain, bit for bit.
    check_against_model(&y257, &w2, &x257, m2, k2, 257);
    check_against_model(&y20, &w, &x20, m, k, 20);
    // Finite, written, and a sane F16-operand dot (no stale poison).
    assert!(y257
        .iter()
        .all(|&b| b != 0xa5a5_a5a5 && f32::from_bits(b).is_finite()));
    let want = (0..k2)
        .map(|kk| f16_to_f32(w2[kk]) * f16_to_f32(x257[kk]))
        .sum::<f32>();
    assert!((f32::from_bits(y257[0]) - want).abs() <= 1e-3 * (1.0 + want.abs()));
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// The module certifies: parse-back, independent wait replay, its committed shape
    /// contract (counts, registers, 40960 fixed and zero dynamic LDS), the all-lane static
    /// LDS bound, and M7's lift identity plus wait/hazard/barrier/window analyses of the
    /// linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = mtp_dense_det_gfx11::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-mtpdet-cert-{}-{}",
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
            "{}/kernels/mtp_dense_det_pm.gfx1151.w8.contract.json",
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

    /// Every DS access of every lane of every wave ends inside the two 20480-byte X
    /// slots, and the bound is tight (one byte less is refused).
    #[test]
    fn static_lds_accesses_stay_inside_the_static_allocation() {
        let e = mtp_dense_det_gfx11::emit(Spec { arch: ARCH }).unwrap();
        let (end, hosted) =
            pm_check::lds_bounds_host(&e.s_text, SYMBOL, 8, mtp_dense_det_gfx11::GROUP_BYTES, &[])
                .unwrap();
        assert!(
            end > mtp_dense_det_gfx11::SLOT_BYTES
                && end <= mtp_dense_det_gfx11::GROUP_BYTES
                && hosted == 0,
            "static end {end}, {hosted} host-bounded accesses"
        );
        assert!(
            pm_check::lds_bounds_host(&e.s_text, SYMBOL, 8, end - 1, &[]).is_err(),
            "bound is not tight"
        );
    }
}
