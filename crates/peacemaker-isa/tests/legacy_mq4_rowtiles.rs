// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G0 native-CPU numerical oracle for the `legacy_mq4_rowtiles` twins:
//! gate-up 8-output-row tiles and down indexed-expanded 8-token tiles.
//!
//! Method (no formula-only claims): the ACTUAL emitted instruction streams
//! (`gate_up::emit` / `down::emit`) are assembled by the native writer,
//! lifted, and executed workgroup by workgroup on the existing peacemaker
//! CPU interpreter (`peacemaker_ir::emu`). Every asserted vector comes out
//! of the real engine. Gate-up has two independent anchors per executed row:
//! the emitted R1 launch (every checked live position is relocated to
//! singleton token 0 with identical inputs) and a faithful Rust transcription
//! of the original AR HIP sources (not marker text):
//! `kernels/src/qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`
//! (`mq4v2_gate_up_row10` affine DAG + rows8 tiled entry) for gate-up, and
//! `kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`
//! for down. Down compares the emitted entry to the actual original native
//! C1 ELF per arch (`tests/fixtures/legacy_mq4_down/gfx1201-8ec3f2d2.elf`
//! and `gfx1151-a3fb11b0.elf`), SHA-pinned and lifted with `Frontend::Hipcc`,
//! relocating each live position to singleton token 0 with identical inputs.
//! Two separate current-source tests run the identical exercise against the
//! gfx1151 object compiled from the current singleton source and JIT flags
//! (`gfx1151-current-d9e0-05e096e0.elf`, which holds BOTH the one-row
//! singleton and the K=640 four-row twin `..._k640_r4` that rdna-compute
//! `moe.rs` launches for eager gfx1151 K=640 decode); hardware must
//! separately bind its actual loaded object. The historical fixtures remain
//! regression anchors.
//! Its independent AR transcription preserves the singleton's first-pair
//! FMA contraction and reduction order.
//!
//! Canonical geometry/codec per the frozen leaf contracts: gate-up output
//! M 1280 (M@40, unbounded by 64), fixed K 2560, ten 136 B groups per row,
//! affine qt44 dequant (one `v_fma_mix_f32` per weight), no SiLU,
//! grid `[ceil(M/8), 10, ceil(R/8)]`, block 256, 10240 B LDS reused per
//! token; down output M 2560 (M@32), K 640-class G128 (68 B groups),
//! grid `[M, 10, ceil(N/8)]`, block 32, LDS 0. Physical tile 8; logical
//! R (gate-up R@52 appended; down original N@40) runs 1..=64 with dynamic
//! tails. A live token's invalid expert stores `+0.0` per the actual AR law
//! (gate-up lane-0 `0.0` to the live row's gate/up slot, like the HIP
//! guard; down `acc = 0` kept through the skipped dot, then the shfl tree
//! and the lane-0 store run outside the expert guard, like the HIP
//! `acc = 0 ... shfl + store` tail). R-tail dead tokens store nothing.
//!
//! Sampling policy (bounded, reported): production shapes are huge, so per
//! logical R this oracle executes the first/last row block, first/last
//! rank, and first/last token tile (64-token extremes at full grid sweep
//! stay the GPU G1 consumer's job, which remains mandatory), while EVERY
//! logical R 1..=64 executes real arithmetic through the engine. Each test
//! asserts its exact executed extent and prints it. Default admission 8/4
//! is untouched; no source writers live in this file.
use peacemaker_ir::emu::{Emulator, Launch, Memory, Region};
use peacemaker_ir::inst::Frontend;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::legacy_mq4_rowtiles::{down, gate_up};
use peacemaker_isa::{native, Arch};

// ---------------------------------------------------------------------------
// Canonical geometry (frozen leaf contracts; asserted in the ABI tests).
// ---------------------------------------------------------------------------

/// Gate-up output rows (M@40). MI = M/2 rows go to y_gate, rest to y_up.
const GU_M: usize = 1280;
const GU_MI: usize = 640;
/// Gate-up fixed K (ten 136 B groups, row stride 1360 B).
const GU_K: usize = 2560;
const GU_GROUPS: usize = 10;
/// Top-k width (gate-up K_TOP@44, down k_top@48).
const KTOP: usize = 10;
/// Down output rows (M@32) and K (K@36, five 68 B G128 groups).
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
const DOWN_GROUPS: usize = 5;
/// Expert-table extent of every case.
const NEXP: i32 = 8;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;

// Emulator address map (mirrors the existing CN/wide-emu oracle layout).
const A_KERNARG: u64 = 0x1_0000;
const A_TABLE: u64 = 0x10_0000;
const A_IDX: u64 = 0x18_0000;
const A_X: u64 = 0x40_0000;
const A_YG: u64 = 0x80_0000;
const A_YU: u64 = 0xa0_0000;
const A_EXPERTS: u64 = 0x1000_0000;
/// Stride between expert bases; exceeds both tensors (GU 1_740_800 B).
const A_ESTRIDE: u64 = 0x20_0000;

// ---------------------------------------------------------------------------
// Deterministic bits (no `half` dev-dependency: local bit-exact helpers).
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Bit-exact IEEE-754 f16 -> f32 (covers normals, subnormals, zero, inf/NaN),
/// matching `half::f16::from_bits(b).to_f32()` and the emulator's converter.
fn f16_to_f32(b: u16) -> f32 {
    let (s, e, m) = (
        ((b >> 15) & 1) as u32,
        ((b >> 10) & 0x1f) as u32,
        (b & 0x3ff) as u32,
    );
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127i32 - 14;
            let mut mm = m;
            while mm & 0x400 == 0 {
                mm <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn lu32(b: &[u8], o: usize) -> u32 {
    (b[o] as u32) | ((b[o + 1] as u32) << 8) | ((b[o + 2] as u32) << 16) | ((b[o + 3] as u32) << 24)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend(x.to_le_bytes());
    }
    out
}

fn find_region<'a>(mem: &'a Memory, name: &str) -> &'a Region {
    mem.regions
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
}

fn read_f32(mem: &Memory, name: &str, word: usize) -> f32 {
    let r = find_region(mem, name);
    let o = word * 4;
    f32::from_le_bytes([r.bytes[o], r.bytes[o + 1], r.bytes[o + 2], r.bytes[o + 3]])
}

fn is_sentinel_at(mem: &Memory, name: &str, off: usize, bytes: usize) -> bool {
    let r = find_region(mem, name);
    r.bytes[off..off + bytes].iter().all(|&b| b == SENTINEL)
}

fn is_guard_intact(mem: &Memory, name: &str) -> bool {
    is_sentinel_at(mem, name, 0, GUARD)
}

fn written_bytes(mem: &Memory, names: &[&str]) -> usize {
    mem.regions
        .iter()
        .filter(|r| names.contains(&r.name.as_str()))
        .map(|r| r.written.count())
        .sum()
}

/// Normal f16 scale with exponent 8..=20 (wide exact band, as the CN oracle).
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}

/// Any normal finite f16 (affine zero planes are NOT forced symmetric here).
fn normal_f16(rng: &mut Rng) -> u16 {
    let e = 1 + rng.below(30) as u16;
    (((rng.below(2)) as u16) << 15) | (e << 10) | (rng.next() as u16 & 0x3ff)
}

/// Full-mantissa finite f32 over a wide exponent band, random signs.
fn wide_f32(rng: &mut Rng) -> f32 {
    let (s, e, m) = (
        (rng.next() & 1) as u32,
        118 + rng.below(19) as u32,
        (rng.next() & 0x7f_ffff) as u32,
    );
    f32::from_bits((s << 31) | (e << 23) | m)
}

// ---------------------------------------------------------------------------
// Genuine-format synthetic weights (real codec bytes, never payload copies).
// ---------------------------------------------------------------------------

/// Gate-up expert tensor: M rows x ten 136 B groups
/// (`[sc0 zp0 sc1 zp1]` f16 headers, 128 nibble payload bytes at +8).
fn gu_expert_tensor(rng: &mut Rng) -> Vec<u8> {
    let mut out = vec![0u8; GU_M * GU_GROUPS * 136];
    for r in 0..GU_M {
        for g in 0..GU_GROUPS {
            let base = (r * GU_GROUPS + g) * 136;
            if (r * GU_GROUPS + g) % 17 == 5 {
                // Degenerate encoder group (hi == lo): zero scale, zero nibbles.
                out[base..base + 2].copy_from_slice(&0u16.to_le_bytes());
                out[base + 2..base + 4].copy_from_slice(&normal_f16(rng).to_le_bytes());
                out[base + 4..base + 6].copy_from_slice(&0u16.to_le_bytes());
                out[base + 6..base + 8].copy_from_slice(&normal_f16(rng).to_le_bytes());
                continue;
            }
            for h in 0..2 {
                out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
                out[base + 4 * h + 2..base + 4 * h + 4]
                    .copy_from_slice(&normal_f16(rng).to_le_bytes());
            }
            for byte in &mut out[base + 8..base + 136] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

/// Down expert tensor: M rows x five 68 B groups (`[sc zp]` f16, 64 B at +4).
fn down_expert_tensor(rng: &mut Rng) -> Vec<u8> {
    let mut out = vec![0u8; DOWN_M * DOWN_GROUPS * 68];
    for r in 0..DOWN_M {
        for g in 0..DOWN_GROUPS {
            let base = (r * DOWN_GROUPS + g) * 68;
            out[base..base + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&normal_f16(rng).to_le_bytes());
            for byte in &mut out[base + 4..base + 68] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Faithful reference DAGs (original AR HIP + emitted row-tile lowering).
//
// Gate-up: `mq4v2_gate_up_row10` with the PM body's exact contraction —
// per lane `boff = tid*4`, `hoff = tid<16 ? 0 : 4`; `w = fma(sc, q, zp)`
// (single rounding; the emitted `v_fma_mix_f32` and `mul_add` agree on
// finite inputs); dot `d1*x1`, then `fma(d0,x0)`, then `fma(d_i,x_i)`
// i = 2..8; `acc[g&3] +=` groups ascending; fold `(a0+a1)+(a2+a3)`;
// the emitted swizzle/masked-bpermute lowering, which equals the snapshot
// 16/8/4/2/1 shuffle-down tree bit for bit (proven by exhaustive probe).
// Down: the indexed-expanded kernel's whole-K path — F16 scale/zero header
// pair widened exactly (`__half2float`), `w = fma(sc, q, zp)` per nibble,
// dot `d = x1*w1` then `fma(x0, w0, d)` then `fma` terms 2..3
// left-associated per group, one ascending chain, then the same lowering
// (proven identical to the snapshot tree). Store law: `acc = 0` with the dot
// skipped under the expert guard, then the shfl tree and the lane-0 store run
// OUTSIDE the guard, so a live token's invalid expert stores `+0.0`; R-tail dead tokens store nothing.
// ---------------------------------------------------------------------------

/// Wave32 snapshot 16/8/4/2/1 shuffle-down tree; lane 0 is the result.
fn lane_tree(mut lanes: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let snap = lanes;
        for l in 0..32 - off {
            lanes[l] = snap[l] + snap[l + off];
        }
    }
    lanes[0]
}

/// One lane's eight-element gate-up group dot: `d_i = fma(sc, q_i, zp)`;
/// `d1*x1`, `fma(d0,x0)`, then `fma(d_i,x_i)` i = 2..8.
fn gu_group_dot8(sc: f32, zp: f32, pk: u32, x: &[f32]) -> f32 {
    let d = |i: usize| sc.mul_add(((pk >> (4 * i)) & 15) as f32, zp);
    let mut s = d(1) * x[1];
    s = d(0).mul_add(x[0], s);
    for i in 2..8 {
        s = d(i).mul_add(x[i], s);
    }
    s
}

/// Gate-up reference row over one expert row's 1360 bytes and one x row.
fn gu_ref_row(w: &[u8], row: usize, x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let hoff = if tid < 16 { 0 } else { 4 };
        let mut acc = [0.0f32; 4];
        for g in 0..GU_GROUPS {
            let base = (row * GU_GROUPS + g) * 136;
            let h = lu32(w, base + hoff);
            let sc = f16_to_f32((h & 0xffff) as u16);
            let zp = f16_to_f32((h >> 16) as u16);
            let pk = lu32(w, base + 8 + tid * 4);
            let xb = g * 256 + tid * 8;
            acc[g & 3] += gu_group_dot8(sc, zp, pk, &x[xb..xb + 8]);
        }
        *lane = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    lane_tree(lanes)
}

/// Down reference row over one expert row's 340 bytes and one x row.
/// Per-group dot starts at pair 1 (`d = x1*w1`), folds pair 0 second
/// (`d = fma(x0, w0, d)`), then pairs 2 and 3 left-associated.
fn down_ref_row(w: &[u8], row: usize, x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..DOWN_GROUPS {
            let base = (row * DOWN_GROUPS + g) * 68;
            let sc = f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let zp = f16_to_f32(u16::from_le_bytes([w[base + 2], w[base + 3]]));
            let (b0, b1) = (w[base + 4 + tid * 2], w[base + 5 + tid * 2]);
            let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
            let xb = g * 128 + tid * 4;
            let mut s = x[xb + 1] * sc.mul_add(q[1] as f32, zp);
            s = sc.mul_add(q[0] as f32, zp).mul_add(x[xb], s);
            for i in 2..4 {
                s = sc.mul_add(q[i] as f32, zp).mul_add(x[xb + i], s);
            }
            acc += s;
        }
        *lane = acc;
    }
    lane_tree(lanes)
}

/// Test-only negative control: the old incorrect first-pair order
/// (`d = x0*w0` then `fma` terms 1..3). Never asserted equal to the engine;
/// used only to prove the encoded-domain operands distinguish the orders.
fn down_ref_row_x0_first(w: &[u8], row: usize, x: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..DOWN_GROUPS {
            let base = (row * DOWN_GROUPS + g) * 68;
            let sc = f16_to_f32(u16::from_le_bytes([w[base], w[base + 1]]));
            let zp = f16_to_f32(u16::from_le_bytes([w[base + 2], w[base + 3]]));
            let (b0, b1) = (w[base + 4 + tid * 2], w[base + 5 + tid * 2]);
            let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
            let xb = g * 128 + tid * 4;
            let mut s = x[xb] * sc.mul_add(q[0] as f32, zp);
            for i in 1..4 {
                s = sc.mul_add(q[i] as f32, zp).mul_add(x[xb + i], s);
            }
            acc += s;
        }
        *lane = acc;
    }
    lane_tree(lanes)
}

// ---------------------------------------------------------------------------
// Lift + kernarg + memory harness (existing native CPU emulator interfaces).
// ---------------------------------------------------------------------------

fn lift(emitted: peacemaker_isa::Emitted, arch: Arch, what: &str) -> Program {
    let elf = native::assemble(&emitted.s_text, arch)
        .unwrap_or_else(|e| panic!("{arch:?} {what} assemble: {e}"));
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap_or_else(|e| panic!("{arch:?} {what} lift: {e}"))
    .program
}

/// SHA-pinned `Frontend::Hipcc` lift of exact HSA ELF fixture bytes (NOT raw
/// shader text) with a byte-exact emit roundtrip. Shared by the historical
/// regression loader below and the current-source gfx1151 test. Any hash
/// mismatch, lift failure, or roundtrip drift hard refuses (panics); no
/// error-catching, no skip.
fn lift_checked_actual(arch: Arch, bytes: &[u8], expected: &str) -> Program {
    use sha2::{Digest, Sha256};
    let got = format!("{:x}", Sha256::digest(bytes));
    assert_eq!(
        got, expected,
        "{arch:?}: actual down C1 fixture SHA256 mismatch"
    );
    let lifted = peacemaker_lift::lift_object(
        bytes,
        peacemaker_lift::Options {
            frontend: Frontend::Hipcc,
        },
    )
    .unwrap_or_else(|e| panic!("{arch:?} actual down C1 lift: {e}"));
    let rebuilt = peacemaker_lift::emit::module(&lifted.program)
        .unwrap_or_else(|e| panic!("{arch:?} actual down C1 byte emission: {e}"));
    assert_eq!(
        rebuilt, bytes,
        "{arch:?}: actual down C1 byte lift changed the ELF"
    );
    lifted.program
}

/// Historical actual original native down C1 per arch, selected by the
/// identical `include_bytes!` fixtures and checked through
/// `lift_checked_actual`. SHA256 is pinned against the parent-pinned
/// statically extracted raw-code reference (no emission, no cache paths).
/// Non-moving fixture paths:
/// `tests/fixtures/legacy_mq4_down/gfx1201-8ec3f2d2.elf` —
/// raw `8ec3f2d218dc18c74031ad0c933d4ec2cb83f33211c579f1dd99ffccef29db55`
/// (N-shared HIP source `a8f8c821293c5c3bb61457a1fe3c3524709003688eea3d56f1218d430510e203`,
/// container `f6d79cd5d015ddea16d0fee5c208a8c21525a93514574d624257aaea6eab5e31`);
/// `tests/fixtures/legacy_mq4_down/gfx1151-a3fb11b0.elf` —
/// raw `a3fb11b0275c5f0b63e335436679a52814eb87a082aa091a352c9741c25a97a9`
/// (actual gfx1151 HIP source `2326440b1210f8f069ab34c0246f8bf7af30969b44eb0f3c563660039fe93bdb`,
/// NOT the N-shared `a8f8`; container
/// `6f8cc92614c0e7d8c7725ab483056d30b3fc1aa2a0327e5ce885ed4e55dd2e3b`).
/// Both arches get a real actual-C1 emulator here; there is no
/// optional/fallback path. Any absent fixture, hash mismatch, or missing
/// lifted instruction hard refuses (panics); no error-catching, no skip.
fn lift_actual_down(arch: Arch) -> Program {
    let (bytes, expected): (&[u8], &str) = match arch {
        Arch::Gfx1201 => (
            include_bytes!("fixtures/legacy_mq4_down/gfx1201-8ec3f2d2.elf"),
            "8ec3f2d218dc18c74031ad0c933d4ec2cb83f33211c579f1dd99ffccef29db55",
        ),
        Arch::Gfx1151 => (
            include_bytes!("fixtures/legacy_mq4_down/gfx1151-a3fb11b0.elf"),
            "a3fb11b0275c5f0b63e335436679a52814eb87a082aa091a352c9741c25a97a9",
        ),
        _ => panic!("{arch:?}: no actual down C1 fixture"),
    };
    lift_checked_actual(arch, bytes, expected)
}

/// Gate-up 56 B kernarg: ptrs@0/8/16/24/32, M@40, K_TOP@44, n_experts@48, R@52.
fn gu_kernarg(r: i32) -> Vec<u8> {
    let mut k = Vec::with_capacity(56);
    for p in [
        A_TABLE,
        A_IDX,
        A_X,
        A_YG + GUARD as u64,
        A_YU + GUARD as u64,
    ] {
        k.extend(p.to_le_bytes());
    }
    for v in [GU_M as i32, KTOP as i32, NEXP, r] {
        k.extend(v.to_le_bytes());
    }
    assert_eq!(k.len(), 56);
    k
}

/// Down ORIGINAL 52 B kernarg: ptrs@0/8/16/24, M@32, K@36, N@40 (= total R),
/// n_exp@44, k_top@48. There is no appended duplicate R.
fn down_kernarg(r: i32) -> Vec<u8> {
    let mut k = Vec::with_capacity(52);
    for p in [A_TABLE, A_IDX, A_X, A_YG + GUARD as u64] {
        k.extend(p.to_le_bytes());
    }
    for v in [DOWN_M as i32, DOWN_K as i32, r, NEXP, KTOP as i32] {
        k.extend(v.to_le_bytes());
    }
    assert_eq!(k.len(), 52);
    k
}

fn table_bytes() -> Vec<u8> {
    let mut t = vec![0u8; NEXP as usize * 8];
    for expert in 0..NEXP as u32 {
        let base = A_EXPERTS + u64::from(expert) * A_ESTRIDE;
        t[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
    }
    t
}

/// Gate-up memory: padded token tiles (dead-tail X zero-filled so no stray
/// read faults; dead-tail Y stays sentinel so a stray write is caught),
/// exact-size top-k indices (any over-read past R faults), sentinel Y with
/// guard words, every admitted expert mapped to the same generated tensor.
fn gu_memory(r: usize, tensor: &[u8], x: Vec<f32>, ids: Vec<i32>) -> Result<Memory, String> {
    let tiles = r.div_ceil(8);
    let pad = tiles * 8;
    let mut mem = Memory::default();
    for expert in 0..NEXP as u32 {
        mem.map(
            format!("e{expert}"),
            A_EXPERTS + u64::from(expert) * A_ESTRIDE,
            tensor.to_vec(),
        )?;
    }
    mem.map("table".into(), A_TABLE, table_bytes())?;
    let mut idb = Vec::with_capacity(ids.len() * 4);
    for id in &ids {
        idb.extend(id.to_le_bytes());
    }
    mem.map("ids".into(), A_IDX, idb)?;
    let mut xp = x;
    xp.resize(pad * GU_K, 0.0);
    mem.map("x".into(), A_X, f32_bytes(&xp))?;
    let ywords = pad * KTOP * GU_MI;
    for (name, base) in [("yg", A_YG), ("yu", A_YU)] {
        mem.map(format!("{name}_lo"), base, vec![SENTINEL; GUARD])?;
        mem.map(name.into(), base + GUARD as u64, vec![SENTINEL; ywords * 4])?;
        mem.map(
            format!("{name}_hi"),
            base + GUARD as u64 + ywords as u64 * 4,
            vec![SENTINEL; GUARD],
        )?;
    }
    Ok(mem)
}

/// Down memory: same padding/guard discipline; hidden rows are per
/// (token, rank) with K words, outputs per flat row with M words. Every
/// admitted expert maps to the same generated tensor used by the reference.
fn down_memory(r: usize, tensor: &[u8], x: Vec<f32>, ids: Vec<i32>) -> Result<Memory, String> {
    let tiles = r.div_ceil(8);
    let pad = tiles * 8;
    let mut mem = Memory::default();
    for expert in 0..NEXP as u32 {
        mem.map(
            format!("e{expert}"),
            A_EXPERTS + u64::from(expert) * A_ESTRIDE,
            tensor.to_vec(),
        )?;
    }
    mem.map("table".into(), A_TABLE, table_bytes())?;
    let mut idb = Vec::with_capacity(ids.len() * 4);
    for id in &ids {
        idb.extend(id.to_le_bytes());
    }
    mem.map("ids".into(), A_IDX, idb)?;
    let mut xp = x;
    xp.resize(pad * KTOP * DOWN_K, 0.0);
    mem.map("x".into(), A_X, f32_bytes(&xp))?;
    let ywords = pad * KTOP * DOWN_M;
    mem.map("yg_lo".into(), A_YG, vec![SENTINEL; GUARD])?;
    mem.map("yg".into(), A_YG + GUARD as u64, vec![SENTINEL; ywords * 4])?;
    mem.map(
        "yg_hi".into(),
        A_YG + GUARD as u64 + ywords as u64 * 4,
        vec![SENTINEL; GUARD],
    )?;
    Ok(mem)
}

fn kernarg_offsets(text: &str) -> Vec<u32> {
    let meta = &text[text.find("amdhsa.kernels:").expect("metadata")..];
    meta.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix(".offset:")
                .map(|v| v.trim().parse::<u32>().unwrap())
        })
        .collect()
}

fn run_wg(emu: &Emulator, mem: &mut Memory, wg: [u32; 3], block: u32, tag: &str) {
    emu.run_workgroup(
        &Launch {
            kernarg_ptr: A_KERNARG,
            workgroup: wg,
            block: [block, 1, 1],
            dynamic_lds: 0,
        },
        mem,
    )
    .unwrap_or_else(|e| panic!("{tag} wg{wg:?}: {e}"));
}

// ---------------------------------------------------------------------------
// Emission + frozen ABI contracts.
// ---------------------------------------------------------------------------

#[test]
fn gu_abi_emission_matches_frozen_contract() {
    assert_eq!(gate_up::KERNARG_BYTES, 56);
    assert_eq!(gate_up::ROW_TILE, 8);
    assert_eq!(gate_up::TOKEN_TILE, 8);
    assert_eq!(gate_up::SEALED_K_TOP, 10);
    assert_eq!(gate_up::FIXED_K, 2560);
    assert_eq!(gate_up::ROW_BYTES, 1360);
    assert_eq!(gate_up::LDS_BYTES, 10240);
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let a = gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        let b = gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert_eq!(a.s_text, b.s_text, "{arch:?}: emission deterministic");
        let sym = gate_up::symbol(arch);
        assert!(
            sym.starts_with("gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_"),
            "{arch:?}: {sym}"
        );
        assert!(a.s_text.contains(&format!("{sym}:")), "{arch:?}: label");
        assert_eq!(
            kernarg_offsets(&a.s_text),
            [0, 8, 16, 24, 32, 40, 44, 48, 52],
            "{arch:?}: kernarg offsets"
        );
        assert!(
            a.s_text.contains(".amdhsa_kernarg_size 56\n"),
            "{arch:?}: size"
        );
        assert!(
            a.s_text
                .contains(".amdhsa_group_segment_fixed_size 10240\n"),
            "{arch:?}: LDS"
        );
        let elf = native::assemble(&a.s_text, arch).unwrap();
        assert!(elf.starts_with(b"\x7fELF"));
        let prog = lift(a, arch, "gu");
        assert!(
            prog.kernels.iter().any(|k| k.symbol.0 == sym),
            "{arch:?}: lifted symbol present"
        );
    }
    assert!(
        gate_up::emit(Arch::Gfx1100).is_err(),
        "GU admits 1201/1151 only"
    );
    assert!(
        gate_up::emit(Arch::Gfx942).is_err(),
        "GU admits 1201/1151 only"
    );
}

#[test]
fn down_abi_emission_matches_frozen_contract() {
    assert_eq!(down::KERNARG_BYTES, 52);
    assert_eq!(down::TILE_ROWS, 8);
    assert_eq!(down::MAX_ROWS, 64);
    for arch in [Arch::Gfx1201, Arch::Gfx1151, Arch::Gfx1100] {
        let a = down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        let b = down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        assert_eq!(a.s_text, b.s_text, "{arch:?}: emission deterministic");
        let sym = format!(
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
            arch.name()
        );
        assert!(a.s_text.contains(&format!("{sym}:")), "{arch:?}: label");
        assert_eq!(
            kernarg_offsets(&a.s_text),
            [0, 8, 16, 24, 32, 36, 40, 44, 48],
            "{arch:?}: original 52 B offsets, N@40, no duplicate R"
        );
        assert!(
            a.s_text.contains(".amdhsa_kernarg_size 52\n"),
            "{arch:?}: size"
        );
        assert!(
            a.s_text.contains(".amdhsa_group_segment_fixed_size 0\n"),
            "{arch:?}: LDS 0"
        );
        let elf = native::assemble(&a.s_text, arch).unwrap();
        assert!(elf.starts_with(b"\x7fELF"));
        let prog = lift(a, arch, "down");
        assert!(
            prog.kernels.iter().any(|k| k.symbol.0 == sym),
            "{arch:?}: lifted symbol present"
        );
    }
    assert!(
        down::emit(Arch::Gfx942).is_err(),
        "down admits 1201/1151/1100"
    );
}

// ---------------------------------------------------------------------------
// Gate-up: every logical R 1..=64 through the real engine.
// ---------------------------------------------------------------------------

/// Per-token distinct inputs: token t's X row is its own draw and rank r of
/// token t takes expert `(t + 3*r + 1) % NEXP`, covering all admitted experts.
fn gu_inputs(r: usize, rng: &mut Rng) -> (Vec<f32>, Vec<i32>) {
    let mut x = Vec::with_capacity(r * GU_K);
    for _ in 0..r * GU_K {
        x.push(wide_f32(rng));
    }
    let mut ids = Vec::with_capacity(r * KTOP);
    for t in 0..r {
        for rank in 0..KTOP {
            ids.push(((t + 3 * rank + 1) % NEXP as usize) as i32);
        }
    }
    (x, ids)
}

fn gu_expected(tensor: &[u8], x: &[f32], token: usize, row: usize) -> f32 {
    gu_ref_row(tensor, row, &x[token * GU_K..token * GU_K + GU_K])
}

#[test]
fn gu_all_logical_r_match_r1_and_ar_reference() {
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let prog = lift(
            gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "gu",
        );
        let sym = gate_up::symbol(arch);
        let emu = Emulator::new(&prog, &sym).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ 0x6d75_31);
        let tensor = gu_expert_tensor(&mut rng);
        // Independent R1 prewarm, double-run identity (not a batch-input golden).
        let (x1, ids1) = gu_inputs(1, &mut rng);
        let mut golden = std::collections::HashMap::new();
        for pass in 0..2 {
            let mut mem = gu_memory(1, &tensor, x1.clone(), ids1.clone()).unwrap();
            mem.map("kernarg".into(), A_KERNARG, gu_kernarg(1)).unwrap();
            for bx in [0u32, 79, 80, 159] {
                for krank in [0u32, 9] {
                    run_wg(
                        &emu,
                        &mut mem,
                        [bx, krank, 0],
                        256,
                        &format!("{arch:?} R1 pass {pass}"),
                    );
                }
            }
            for bx in [0u32, 79, 80, 159] {
                for krank in [0u32, 9] {
                    for w in 0..8usize {
                        let row = bx as usize * 8 + w;
                        let (name, slot) = if row < GU_MI {
                            ("yg", row)
                        } else {
                            ("yu", row - GU_MI)
                        };
                        let got = read_f32(&mem, name, krank as usize * GU_MI + slot);
                        let key = (bx, krank, row);
                        if pass == 0 {
                            golden.insert(key, got.to_bits());
                        } else {
                            assert_eq!(
                                golden[&key],
                                got.to_bits(),
                                "{arch:?} R1 prewarm rerun differs at {key:?}"
                            );
                        }
                    }
                }
            }
            assert!(is_guard_intact(&mem, "yg_lo"));
            assert!(is_guard_intact(&mem, "yu_lo"));
        }
        let (mut total_wg, mut total_words) = (0usize, 0usize);
        for r in 1usize..=64 {
            let tiles = r.div_ceil(8);
            let (x, ids) = gu_inputs(r, &mut rng);
            let mut mem = gu_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
            mem.map("kernarg".into(), A_KERNARG, gu_kernarg(r as i32))
                .unwrap();
            let mut bzs = vec![0usize, tiles - 1];
            bzs.dedup();
            let mut wg = 0;
            // Relocate each checked token's actual X and all rank IDs to
            // singleton token 0, retaining the exact same generated weights.
            // The independent prewarm above consumed different RNG draws.
            let mut singleton_golden = std::collections::HashMap::new();
            for bz in &bzs {
                for t in 0..8usize {
                    let token = bz * 8 + t;
                    if token >= r {
                        continue;
                    }
                    let singleton_x = x[token * GU_K..(token + 1) * GU_K].to_vec();
                    let singleton_ids = ids[token * KTOP..(token + 1) * KTOP].to_vec();
                    let mut singleton = gu_memory(1, &tensor, singleton_x, singleton_ids).unwrap();
                    singleton
                        .map("kernarg".into(), A_KERNARG, gu_kernarg(1))
                        .unwrap();
                    for bx in [0u32, 159] {
                        for krank in [0u32, 9] {
                            run_wg(
                                &emu,
                                &mut singleton,
                                [bx, krank, 0],
                                256,
                                &format!("{arch:?} R{r} token {token} emitted R1"),
                            );
                            for w in 0..8usize {
                                let row = bx as usize * 8 + w;
                                let (name, slot) = if row < GU_MI {
                                    ("yg", row)
                                } else {
                                    ("yu", row - GU_MI)
                                };
                                let got = read_f32(&singleton, name, krank as usize * GU_MI + slot);
                                singleton_golden.insert((token, bx, krank, row), got.to_bits());
                            }
                        }
                    }
                    for name in ["yg_lo", "yu_lo", "yg_hi", "yu_hi"] {
                        assert!(is_guard_intact(&singleton, name));
                    }
                }
            }
            for bx in [0u32, 159] {
                for krank in [0u32, 9] {
                    for bz in &bzs {
                        run_wg(
                            &emu,
                            &mut mem,
                            [bx, krank, *bz as u32],
                            256,
                            &format!("{arch:?} R{r}"),
                        );
                        wg += 1;
                    }
                }
            }
            assert_eq!(wg, if tiles == 1 { 4 } else { 8 }, "{arch:?} R{r}: extent");
            total_wg += wg;
            let mut words = 0;
            for bx in [0u32, 159] {
                for krank in [0u32, 9] {
                    for bz in &bzs {
                        for w in 0..8usize {
                            let row = bx as usize * 8 + w;
                            for t in 0..8usize {
                                let token = bz * 8 + t;
                                if token >= r {
                                    continue;
                                }
                                let expect = gu_expected(&tensor, &x, token, row);
                                let (name, slot) = if row < GU_MI {
                                    ("yg", row)
                                } else {
                                    ("yu", row - GU_MI)
                                };
                                let got = read_f32(
                                    &mem,
                                    name,
                                    token * KTOP * GU_MI + krank as usize * GU_MI + slot,
                                );
                                assert_eq!(
                                    got.to_bits(),
                                    expect.to_bits(),
                                    "{arch:?} R{r} token {token} rank {krank} row {row}: engine vs AR reference"
                                );
                                assert_eq!(
                                    singleton_golden[&(token, bx, krank, row)],
                                    got.to_bits(),
                                    "{arch:?} R{r} token {token} differs from emitted R1"
                                );
                                words += 1;
                            }
                        }
                    }
                }
            }
            total_words += words;
            // Dead-tail tokens of the last tile stay fully sentinel.
            for token in r..tiles * 8 {
                for rank in 0..KTOP {
                    assert!(
                        is_sentinel_at(
                            &mem,
                            "yg",
                            (token * KTOP * GU_MI + rank * GU_MI) * 4,
                            GU_MI * 4
                        ),
                        "{arch:?} R{r}: dead tail token {token} rank {rank} gate written"
                    );
                    assert!(
                        is_sentinel_at(
                            &mem,
                            "yu",
                            (token * KTOP * GU_MI + rank * GU_MI) * 4,
                            GU_MI * 4
                        ),
                        "{arch:?} R{r}: dead tail token {token} rank {rank} up written"
                    );
                }
            }
            assert!(is_guard_intact(&mem, "yg_lo"));
            assert!(is_guard_intact(&mem, "yu_lo"));
            assert!(is_guard_intact(&mem, "yg_hi"));
            assert!(is_guard_intact(&mem, "yu_hi"));
            eprintln!("gu {arch:?} R={r}: workgroups={wg} words_compared={words}");
        }
        // Exact closed-form extent: R<=8 -> 4 wg each, R 9..=64 -> 8 wg each.
        assert_eq!(total_wg, 8 * 4 + 56 * 8, "{arch:?}: total workgroups");
        assert_eq!(total_words, 23552, "{arch:?}: total compared words");
        eprintln!("gu {arch:?}: total workgroups={total_wg} total words={total_words}");
    }
}

/// MI-split and neighborhood rows (gate rows 632..647 straddle row blocks,
/// up rows 640..647 open the second half), plus the full rank sweep so
/// every rank's addressing is engine-checked.
#[test]
fn gu_row_tile_boundaries_and_full_rank_sweep() {
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let prog = lift(
            gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "gu",
        );
        let sym = gate_up::symbol(arch);
        let emu = Emulator::new(&prog, &sym).unwrap();
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let tensor = gu_expert_tensor(&mut rng);
        let r = 9usize;
        let (x, ids) = gu_inputs(r, &mut rng);
        let mut mem = gu_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
        mem.map("kernarg".into(), A_KERNARG, gu_kernarg(r as i32))
            .unwrap();
        let mut wg = 0;
        for bx in [0u32, 1, 79, 80, 158, 159] {
            for krank in 0..KTOP as u32 {
                for bz in [0u32, 1] {
                    run_wg(
                        &emu,
                        &mut mem,
                        [bx, krank, bz],
                        256,
                        &format!("{arch:?} boundary"),
                    );
                    wg += 1;
                }
            }
        }
        assert_eq!(wg, 6 * 10 * 2);
        let mut words = 0;
        for bx in [0u32, 1, 79, 80, 158, 159] {
            for krank in 0..KTOP {
                for bz in [0usize, 1] {
                    for w in 0..8usize {
                        let row = bx as usize * 8 + w;
                        for t in 0..8usize {
                            let token = bz * 8 + t;
                            if token >= r {
                                continue;
                            }
                            let expect = gu_expected(&tensor, &x, token, row);
                            let (name, slot) = if row < GU_MI {
                                ("yg", row)
                            } else {
                                ("yu", row - GU_MI)
                            };
                            let got =
                                read_f32(&mem, name, token * KTOP * GU_MI + krank * GU_MI + slot);
                            assert_eq!(
                                got.to_bits(),
                                expect.to_bits(),
                                "{arch:?} boundary token {token} rank {krank} row {row}"
                            );
                            words += 1;
                        }
                    }
                }
            }
        }
        eprintln!("gu {arch:?} boundaries: workgroups={wg} words={words}");
    }
}

/// Invalid expert ids take the actual masked-zero store path per
/// (token, rank, row) — lane 0 writes `0.0` to the live row's gate/up slot,
/// exactly like the HIP guard — while every valid neighbor matches AR. Both
/// token tiles execute; dead-tail X/IDs are unmapped and Y stays sentinel.
#[test]
fn gu_invalid_expert_ids_masked_and_tail_guarded() {
    let bad = [-1, i32::MIN, NEXP, NEXP + 1];
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let prog = lift(
            gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "gu",
        );
        let sym = gate_up::symbol(arch);
        let emu = Emulator::new(&prog, &sym).unwrap();
        let mut rng = Rng(0xbad0_bad0_bad0_bad);
        let tensor = gu_expert_tensor(&mut rng);
        let r = 9usize;
        let (x, mut ids) = gu_inputs(r, &mut rng);
        for rank in 0..KTOP {
            ids[2 * KTOP + rank] = bad[rank % 4];
        }
        let mut mem = gu_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
        // Unlike the general padded harness, make any dead-token X read fault.
        mem.regions
            .iter_mut()
            .find(|region| region.name == "x")
            .unwrap()
            .bytes
            .truncate(r * GU_K * 4);
        mem.map("kernarg".into(), A_KERNARG, gu_kernarg(r as i32))
            .unwrap();
        for bx in [0u32, 159] {
            for krank in 0..KTOP as u32 {
                for bz in [0u32, 1] {
                    run_wg(
                        &emu,
                        &mut mem,
                        [bx, krank, bz],
                        256,
                        &format!("{arch:?} invalid"),
                    );
                }
            }
        }
        for bx in [0u32, 159] {
            for krank in 0..KTOP {
                for w in 0..8usize {
                    for row in [bx as usize * 8 + w] {
                        let (name, slot) = if row < GU_MI {
                            ("yg", row)
                        } else {
                            ("yu", row - GU_MI)
                        };
                        for token in 0..r {
                            let want = if token == 2 {
                                0.0f32
                            } else {
                                gu_expected(&tensor, &x, token, row)
                            };
                            let got =
                                read_f32(&mem, name, token * KTOP * GU_MI + krank * GU_MI + slot);
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "{arch:?} token {token} rank {krank} row {row}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            written_bytes(&mem, &["yg", "yu"]),
            2 * KTOP * 8 * r * 4,
            "{arch:?}: exactly eight gate + eight up rows x ten ranks x nine live tokens"
        );
        for token in r..r.div_ceil(8) * 8 {
            for name in ["yg", "yu"] {
                assert!(
                    is_sentinel_at(&mem, name, token * KTOP * GU_MI * 4, KTOP * GU_MI * 4),
                    "{arch:?}: dead token {token} {name} written"
                );
            }
        }
        for region in &mem.regions {
            if !["yg", "yu"].contains(&region.name.as_str()) {
                assert_eq!(
                    region.written.count(),
                    0,
                    "{arch:?}: immutable region {} written",
                    region.name
                );
            }
        }
        for name in ["yg_lo", "yu_lo", "yg_hi", "yu_hi"] {
            assert!(is_guard_intact(&mem, name));
        }
        eprintln!("gu {arch:?} invalid-mask: ok");
    }
}

/// Exact written-address sets for an unaligned width: only the addressed
/// words of executed workgroups are written, everything else stays sentinel.
#[test]
fn gu_store_addresses_follow_slot_coords() {
    let arch = Arch::Gfx1201;
    let prog = lift(
        gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
        arch,
        "gu",
    );
    let sym = gate_up::symbol(arch);
    let emu = Emulator::new(&prog, &sym).unwrap();
    let mut rng = Rng(0xadd4e550000001);
    let tensor = gu_expert_tensor(&mut rng);
    let r = 9usize;
    let (x, ids) = gu_inputs(r, &mut rng);
    let mut mem = gu_memory(r, &tensor, x, ids).unwrap();
    mem.map("kernarg".into(), A_KERNARG, gu_kernarg(r as i32))
        .unwrap();
    // Execute two workgroups: (bx 0, rank 0, tile 0) and (bx 159, rank 9, tile 1).
    for (bx, krank, bz) in [(0u32, 0u32, 0u32), (159, 9, 1)] {
        run_wg(&emu, &mut mem, [bx, krank, bz], 256, "coords");
    }
    // Expected written words: rows 0..8 of (tokens 0..8, rank 0) gate up to
    // the row-639/640 half split, rows 1272..1280 of (token 8, rank 9) up.
    let mut expect_gate = vec![false; 16 * KTOP * GU_MI];
    let mut expect_up = vec![false; 16 * KTOP * GU_MI];
    for token in 0..8 {
        for row in 0..8 {
            expect_gate[token * KTOP * GU_MI + row] = true;
        }
    }
    for row in 1272..1280 {
        expect_up[8 * KTOP * GU_MI + 9 * GU_MI + (row - GU_MI)] = true;
    }
    for (name, expect) in [("yg", &expect_gate), ("yu", &expect_up)] {
        let region = find_region(&mem, name);
        for (i, e) in expect.iter().enumerate() {
            let written = (0..4).any(|b| region.written.contains(i * 4 + b));
            assert_eq!(
                written, *e,
                "{name} word {i}: written={written} expected={e}"
            );
            if !e {
                assert!(
                    region.bytes[i * 4..i * 4 + 4]
                        .iter()
                        .all(|&b| b == SENTINEL),
                    "{name} word {i}: untouched word not sentinel"
                );
            }
        }
    }
}

/// Reproducible finite extremes at aligned R=8 and tailed R=7: full-K
/// signed cancellation and subnormal X, with a finite-DAG activation bound
/// retaining maximum finite f16 headers and maximum nibbles every 64th row.
#[test]
fn gu_extreme_activations() {
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let prog = lift(
            gate_up::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "gu",
        );
        let sym = gate_up::symbol(arch);
        let emu = Emulator::new(&prog, &sym).unwrap();
        let mut rng = Rng(0xe7f0_e7f0_e7f0_e7f0);
        // All generated affine weights obey |fma(sc, q, zp)| <= 16*65504:
        // finite f16 headers have magnitude <=65504 and q is in 0..=15.
        // Bound the absolute sum of ALL K terms, not a single product.
        // The longest rounded reference path is eight dot operations,
        // ceil(groups/4) stripe adds, two fold adds, five wave-tree adds.
        // With u=2^-24, gamma_d <= d*u/(1-d*u); multiplying MAX by
        // (1-d*u) reserves this exact DAG's rounding allowance. The strict
        // f32 predecessor also absorbs cap-conversion and subnormal errors.
        // The old MAX/65504/16 rounded X UP to 0x75801002: row0/g0/lane0's
        // d1=1048064 times negative X was already -inf before cancellation.
        let max_weight = 16.0 * f64::from(f16_to_f32(0x7bff));
        let rounding_depth = 8 + GU_GROUPS.div_ceil(4) + 2 + 5;
        let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
        let finite_cap = f64::from(f32::MAX) * (1.0 - rounding_depth as f64 * unit_roundoff)
            / (GU_K as f64 * max_weight);
        let max_x = f32::from_bits((finite_cap as f32).to_bits() - 1);
        let extreme_x: Vec<f32> = [
            max_x,
            -max_x,
            1e-30,
            -1e-30,
            f32::from_bits(1),
            0.0,
            -0.0,
            1.0,
        ]
        .into_iter()
        .cycle()
        .take(GU_K)
        .collect();
        let mut tensor = gu_expert_tensor(&mut rng);
        for row in (0..GU_M).step_by(64) {
            tensor[row * GU_GROUPS * 136..row * GU_GROUPS * 136 + 2]
                .copy_from_slice(&0x7bffu16.to_le_bytes());
            tensor[row * GU_GROUPS * 136 + 2..row * GU_GROUPS * 136 + 4]
                .copy_from_slice(&0x7bffu16.to_le_bytes());
            for byte in &mut tensor[row * GU_GROUPS * 136 + 8..row * GU_GROUPS * 136 + 12] {
                *byte = 0xff;
            }
        }
        for r in [8usize, 7] {
            let tiles = r.div_ceil(8);
            let x: Vec<f32> = (0..r).flat_map(|_| extreme_x.clone()).collect();
            let mut ids = vec![0i32; r * KTOP];
            for t in 0..r {
                for rank in 0..KTOP {
                    ids[t * KTOP + rank] = ((t + rank) % NEXP as usize) as i32;
                }
            }
            let mut mem = gu_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
            mem.map("kernarg".into(), A_KERNARG, gu_kernarg(r as i32))
                .unwrap();
            for bx in [0u32, 159] {
                for krank in [0u32, 9] {
                    for bz in 0..tiles as u32 {
                        run_wg(
                            &emu,
                            &mut mem,
                            [bx, krank, bz],
                            256,
                            &format!("{arch:?} extreme R{r}"),
                        );
                    }
                }
            }
            for bx in [0u32, 159] {
                for krank in [0u32, 9] {
                    for bz in 0..tiles {
                        for w in 0..8usize {
                            let row = bx as usize * 8 + w;
                            for t in 0..8usize {
                                let token = bz * 8 + t;
                                if token >= r {
                                    continue;
                                }
                                let expect = gu_expected(&tensor, &x, token, row);
                                assert!(
                                    expect.is_finite(),
                                    "{arch:?} extreme R{r}: reference not finite"
                                );
                                let (name, slot) = if row < GU_MI {
                                    ("yg", row)
                                } else {
                                    ("yu", row - GU_MI)
                                };
                                let got = read_f32(
                                    &mem,
                                    name,
                                    token * KTOP * GU_MI + krank as usize * GU_MI + slot,
                                );
                                assert_eq!(
                                    got.to_bits(),
                                    expect.to_bits(),
                                    "{arch:?} extreme R{r} token {token} rank {krank} row {row}"
                                );
                            }
                        }
                    }
                }
            }
            eprintln!("gu {arch:?} extreme R={r}: ok");
        }
    }
}

// ---------------------------------------------------------------------------
// Down: every logical R 1..=64 through the real engine.
// ---------------------------------------------------------------------------

fn down_inputs(r: usize, rng: &mut Rng) -> (Vec<f32>, Vec<i32>) {
    let mut x = Vec::with_capacity(r * KTOP * DOWN_K);
    for _ in 0..r * KTOP * DOWN_K {
        x.push(wide_f32(rng));
    }
    let mut ids = Vec::with_capacity(r * KTOP);
    for t in 0..r {
        for rank in 0..KTOP {
            ids.push(((t + 5 * rank + 2) % NEXP as usize) as i32);
        }
    }
    (x, ids)
}

fn down_expected(tensor: &[u8], x: &[f32], token: usize, rank: usize, row: usize) -> f32 {
    let flat = token * KTOP + rank;
    down_ref_row(tensor, row, &x[flat * DOWN_K..flat * DOWN_K + DOWN_K])
}

const DOWN_ROWS: [usize; 8] = [0, 1, 639, 640, 1279, 1280, 2558, 2559];

/// One actual AR singleton the tile is compared against: its symbol and the
/// number of output rows one workgroup produces (grid x = ceil(M / rows)).
#[derive(Clone, Copy)]
struct DownSingleton {
    symbol: &'static str,
    rows_per_wg: usize,
}

/// The one-row singleton (`blockIdx.x` = output row): every historical
/// fixture, and recorded/captured decode on gfx1151.
const DOWN_SINGLETON_ONE_ROW: DownSingleton = DownSingleton {
    symbol: "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
    rows_per_wg: 1,
};

/// The K=640 four-row twin (`blockIdx.x` = row / 4): what eager gfx1151 K=640
/// AR decode launches (`HIPFIRE_QWEN4_MOE_DOWN_R4`, default on).
const DOWN_SINGLETON_K640_R4: DownSingleton = DownSingleton {
    symbol: "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4",
    rows_per_wg: 4,
};

/// Full per-arch down numerical exercise shared by the historical regression
/// test and the current-source gfx1151 tests below: every logical R 1..=64
/// through the real engine, comparing the emitted tile entry against BOTH
/// the independent R1 prewarm golden, the faithful AR transcription, and the
/// given same-arch actual-C1 singleton (each live position relocated to
/// singleton token 0 with identical inputs). Every case, sample row/rank/
/// tile selection, math, assertion, and extent check is preserved verbatim;
/// only the actual-C1 `Program` and the singleton kernel/geometry differ per
/// caller.
fn down_exercise_all_r(arch: Arch, actual_prog: Program, singleton_kernel: DownSingleton) {
    // Every lane of a multi-row singleton workgroup must be among the
    // compared rows, or a per-lane divergence could hide.
    let mut lanes: Vec<usize> = DOWN_ROWS
        .iter()
        .map(|row| row % singleton_kernel.rows_per_wg)
        .collect();
    lanes.sort_unstable();
    lanes.dedup();
    assert_eq!(
        lanes.len(),
        singleton_kernel.rows_per_wg,
        "{arch:?}: DOWN_ROWS must cover every row lane of {}",
        singleton_kernel.symbol
    );
    let sym = format!(
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
        arch.name()
    );
    let prog = lift(
        down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
        arch,
        "down",
    );
    let emu = Emulator::new(&prog, &sym).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    // Same-arch actual singleton, not another launch of the tile body.
    let actual = Emulator::new(&actual_prog, singleton_kernel.symbol)
        .unwrap_or_else(|e| panic!("{arch:?} actual C1 {}: {e}", singleton_kernel.symbol));
    let mut rng = Rng(0xd09d_09d0_9d09_d09 ^ 0x54321);
    let tensor = down_expert_tensor(&mut rng);
    // Independent R1 prewarm, double-run identity (not a batch-input golden).
    let (x1, ids1) = down_inputs(1, &mut rng);
    let mut golden = std::collections::HashMap::new();
    for pass in 0..2 {
        let mut mem = down_memory(1, &tensor, x1.clone(), ids1.clone()).unwrap();
        mem.map("kernarg".into(), A_KERNARG, down_kernarg(1))
            .unwrap();
        for row in DOWN_ROWS {
            for rank in [0u32, 9] {
                run_wg(
                    &emu,
                    &mut mem,
                    [row as u32, rank, 0],
                    32,
                    &format!("{arch:?} R1 pass {pass}"),
                );
            }
        }
        for row in DOWN_ROWS {
            for rank in [0u32, 9] {
                let got = read_f32(&mem, "yg", rank as usize * DOWN_M + row);
                if pass == 0 {
                    golden.insert((row, rank), got.to_bits());
                } else {
                    assert_eq!(
                        golden[&(row, rank)],
                        got.to_bits(),
                        "{arch:?} R1 prewarm rerun differs"
                    );
                }
            }
        }
    }
    let (mut total_wg, mut total_words) = (0usize, 0usize);
    for r in 1usize..=64 {
        let tiles = r.div_ceil(8);
        let (x, ids) = down_inputs(r, &mut rng);
        let mut mem = down_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
        mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
            .unwrap();
        let mut bzs = vec![0usize, tiles - 1];
        bzs.dedup();
        // Each actual C1 singleton must see this token's actual hidden rows
        // and IDs, not the independently sampled prewarm input above.
        // Relocate all ranks together to singleton token 0; retain the
        // same expert tensor.
        let mut singleton_golden = std::collections::HashMap::new();
        for bz in &bzs {
            for t in 0..8usize {
                let token = bz * 8 + t;
                if token >= r {
                    continue;
                }
                let flat = token * KTOP;
                let singleton_x = x[flat * DOWN_K..(flat + KTOP) * DOWN_K].to_vec();
                let singleton_ids = ids[flat..flat + KTOP].to_vec();
                let mut singleton = down_memory(1, &tensor, singleton_x, singleton_ids).unwrap();
                singleton
                    .map("kernarg".into(), A_KERNARG, down_kernarg(1))
                    .unwrap();
                for row in DOWN_ROWS {
                    for rank in [0u32, 9] {
                        run_wg(
                            &actual,
                            &mut singleton,
                            [(row / singleton_kernel.rows_per_wg) as u32, rank, 0],
                            32,
                            &format!(
                                "{arch:?} R{r} token {token} actual {}",
                                singleton_kernel.symbol
                            ),
                        );
                        let got = read_f32(&singleton, "yg", rank as usize * DOWN_M + row);
                        singleton_golden.insert((token, row, rank), got.to_bits());
                    }
                }
            }
        }
        let mut wg = 0;
        for row in DOWN_ROWS {
            for rank in [0u32, 9] {
                for bz in &bzs {
                    run_wg(
                        &emu,
                        &mut mem,
                        [row as u32, rank, *bz as u32],
                        32,
                        &format!("{arch:?} R{r}"),
                    );
                    wg += 1;
                }
            }
        }
        assert_eq!(
            wg,
            if tiles == 1 { 16 } else { 32 },
            "{arch:?} R{r}: extent"
        );
        total_wg += wg;
        let mut words = 0;
        let mut order_sensitive = 0;
        for row in DOWN_ROWS {
            for rank in [0u32, 9] {
                for bz in &bzs {
                    for t in 0..8usize {
                        let token = bz * 8 + t;
                        if token >= r {
                            continue;
                        }
                        let flat = token * KTOP + rank as usize;
                        let expect = down_expected(&tensor, &x, token, rank as usize, row);
                        let wrong = down_ref_row_x0_first(
                            &tensor,
                            row,
                            &x[flat * DOWN_K..flat * DOWN_K + DOWN_K],
                        );
                        if wrong.to_bits() != expect.to_bits() {
                            order_sensitive += 1;
                        }
                        let got =
                            read_f32(&mem, "yg", (token * KTOP + rank as usize) * DOWN_M + row);
                        assert_eq!(
                            got.to_bits(),
                            expect.to_bits(),
                            "{arch:?} R{r} token {token} rank {rank} row {row}: engine vs AR reference"
                        );
                        assert_eq!(
                            singleton_golden[&(token, row, rank)],
                            got.to_bits(),
                            "{arch:?} R{r} token {token} differs from actual C1"
                        );
                        words += 1;
                    }
                }
            }
        }
        total_words += words;
        assert!(
            order_sensitive > 0,
            "{arch:?} R{r}: no compared output distinguishes x1-first from x0-first"
        );
        for token in r..tiles * 8 {
            for rank in 0..KTOP {
                assert!(
                    is_sentinel_at(&mem, "yg", ((token * KTOP + rank) * DOWN_M) * 4, DOWN_M * 4),
                    "{arch:?} R{r}: dead tail token {token} rank {rank} written"
                );
            }
        }
        assert!(is_guard_intact(&mem, "yg_lo"));
        assert!(is_guard_intact(&mem, "yg_hi"));
        eprintln!(
            "down {arch:?} R={r}: workgroups={wg} words_compared={words} order_sensitive={order_sensitive}"
        );
    }
    assert_eq!(total_wg, 8 * 16 + 56 * 32, "{arch:?}: total workgroups");
    eprintln!("down {arch:?}: total workgroups={total_wg} total words={total_words}");
}

/// Historical regression: the full all-R exercise above against the pinned
/// original C1 fixtures (`gfx1201-8ec3f2d2`, `gfx1151-a3fb11b0`).
#[test]
fn down_all_logical_r_match_r1_and_ar_reference() {
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        down_exercise_all_r(arch, lift_actual_down(arch), DOWN_SINGLETON_ONE_ROW);
    }
}

/// Current-source gfx1151 fixture: the object compiled from the current
/// singleton source and JIT flags
/// (`fixtures/legacy_mq4_down/gfx1151-current-d9e0-05e096e0.elf`, raw
/// `05e096e066a9c503e3d762584bb3a21a50d95f45785f22a31764b5500947be8b`, HIP
/// source `e50f30807a2f88c6ab20b57d9c27493f88f1809e0226be9209b0427881965745`:
/// PR #941's alternate-entry macros preserve the complete prior generic ELF
/// byte-for-byte with the recorded JIT recipe, including the K=640 twin), SHA-pinned and lifted with
/// `Frontend::Hipcc` through the same byte-exact roundtrip helper on
/// `Arch::Gfx1151`. Both singletons live in this one object.
fn lift_current_source_gfx1151_down() -> Program {
    use sha2::{Digest, Sha256};
    let source = include_bytes!(
        "../../../kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        "e50f30807a2f88c6ab20b57d9c27493f88f1809e0226be9209b0427881965745",
        "current-source gfx1151 down reference no longer matches the singleton HIP source"
    );
    lift_checked_actual(
        Arch::Gfx1151,
        include_bytes!("fixtures/legacy_mq4_down/gfx1151-current-d9e0-05e096e0.elf"),
        "05e096e066a9c503e3d762584bb3a21a50d95f45785f22a31764b5500947be8b",
    )
}

/// Current-source gfx1151, one-row singleton: recorded/captured AR decode and
/// eager decode with `HIPFIRE_QWEN4_MOE_DOWN_R4=0`.
#[test]
fn current_source_gfx1151_down_all_logical_r_match_r1_and_ar_reference() {
    down_exercise_all_r(
        Arch::Gfx1151,
        lift_current_source_gfx1151_down(),
        DOWN_SINGLETON_ONE_ROW,
    );
}

/// Current-source gfx1151, K=640 four-row twin: the singleton eager AR decode
/// launches (`moe.rs`: K == 640, `qwen4_moe_down_r4_enabled`, not recording,
/// not capturing). The tile must be bitwise this kernel's per-row result for
/// every logical R, exactly as it is for the one-row singleton.
#[test]
fn current_source_gfx1151_down_k640_r4_all_logical_r_match_r1_and_ar_reference() {
    down_exercise_all_r(
        Arch::Gfx1151,
        lift_current_source_gfx1151_down(),
        DOWN_SINGLETON_K640_R4,
    );
}

/// Down on gfx1100 (admitted arch): boundary widths plus the +0 store law for
/// live-token invalid experts.
#[test]
fn down_gfx1100_boundary_widths_and_invalid_mask() {
    let arch = Arch::Gfx1100;
    let sym = format!(
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
        arch.name()
    );
    let prog = lift(
        down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
        arch,
        "down",
    );
    let emu = Emulator::new(&prog, &sym).unwrap();
    let mut rng = Rng(0x1100_1100_1100_1100);
    let tensor = down_expert_tensor(&mut rng);
    for r in [1usize, 7, 8, 9, 63, 64] {
        let tiles = r.div_ceil(8);
        let (x, ids) = down_inputs(r, &mut rng);
        let mut mem = down_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
        mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
            .unwrap();
        for row in [0u32, 2559] {
            for rank in 0..KTOP as u32 {
                for bz in 0..tiles as u32 {
                    run_wg(&emu, &mut mem, [row, rank, bz], 32, &format!("R{r}"));
                }
            }
        }
        for row in [0usize, 2559] {
            for rank in 0..KTOP {
                for bz in 0..tiles {
                    for t in 0..8usize {
                        let token = bz * 8 + t;
                        if token >= r {
                            continue;
                        }
                        let expect = down_expected(&tensor, &x, token, rank, row);
                        let got = read_f32(&mem, "yg", (token * KTOP + rank) * DOWN_M + row);
                        assert_eq!(
                            got.to_bits(),
                            expect.to_bits(),
                            "R{r} token {token} rank {rank} row {row}"
                        );
                    }
                }
            }
        }
        eprintln!("down gfx1100 R={r}: ok");
    }
    // Invalid ids at live tokens store `+0.0` (actual AR law: the expert
    // guard skips only the dot; the shfl tree and lane-0 store run outside
    // it), while valid neighbors equal the reference.
    let bad = [-1, i32::MIN, NEXP, NEXP + 1];
    let r = 8usize;
    let (x, mut ids) = down_inputs(r, &mut rng);
    for rank in 0..KTOP {
        ids[5 * KTOP + rank] = bad[rank % 4];
    }
    let mut mem = down_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
    mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
        .unwrap();
    for row in [0u32, 1280, 2559] {
        for rank in 0..KTOP as u32 {
            run_wg(&emu, &mut mem, [row, rank, 0], 32, "invalid");
        }
    }
    for row in [0usize, 1280, 2559] {
        for rank in 0..KTOP {
            let got = read_f32(&mem, "yg", (5 * KTOP + rank) * DOWN_M + row);
            assert_eq!(
                got.to_bits(),
                0.0f32.to_bits(),
                "row {row} rank {rank}: live-token invalid slot must store +0"
            );
            let neighbor = read_f32(&mem, "yg", (4 * KTOP + rank) * DOWN_M + row);
            assert_eq!(
                neighbor.to_bits(),
                down_expected(&tensor, &x, 4, rank, row).to_bits(),
                "row {row} rank {rank}: valid neighbor untouched"
            );
        }
    }
    assert_eq!(
        written_bytes(&mem, &["yg"]),
        3 * KTOP * 8 * 4,
        "exactly sampled rows x ranks x eight live tokens written"
    );
    assert!(is_guard_intact(&mem, "yg_lo"));
    assert!(is_guard_intact(&mem, "yg_hi"));
}

/// Down invalid ids on the primary archs, all ranks: live-token invalid slots
/// store `+0.0` (AR law), dead-tail slots store nothing, with exact write
/// counts and guard words.
#[test]
fn down_invalid_expert_ids_store_plus_zero_and_tail_guarded() {
    let bad = [-1, i32::MIN, NEXP, NEXP + 1];
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let sym = format!(
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
            arch.name()
        );
        let prog = lift(
            down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "down",
        );
        let emu = Emulator::new(&prog, &sym).unwrap();
        let mut rng = Rng(0xd0ba_d0ba_d0ba_d0b);
        let tensor = down_expert_tensor(&mut rng);
        let r = 9usize;
        let (x, mut ids) = down_inputs(r, &mut rng);
        for rank in 0..KTOP {
            ids[2 * KTOP + rank] = bad[rank % 4];
        }
        let mut mem = down_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
        mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
            .unwrap();
        for row in DOWN_ROWS {
            for rank in 0..KTOP as u32 {
                for bz in 0..r.div_ceil(8) {
                    run_wg(
                        &emu,
                        &mut mem,
                        [row as u32, rank, bz as u32],
                        32,
                        &format!("{arch:?} invalid"),
                    );
                }
            }
        }
        for row in DOWN_ROWS {
            for rank in 0..KTOP {
                // Invalid token 2 is a LIVE token: the store runs, writing +0.
                let bad_got = read_f32(&mem, "yg", (2 * KTOP + rank) * DOWN_M + row);
                assert_eq!(
                    bad_got.to_bits(),
                    0.0f32.to_bits(),
                    "{arch:?} live-token invalid token 2 rank {rank} row {row}: must store +0"
                );
                let good_got = read_f32(&mem, "yg", (8 * KTOP + rank) * DOWN_M + row);
                assert_eq!(
                    good_got.to_bits(),
                    down_expected(&tensor, &x, 8, rank, row).to_bits(),
                    "{arch:?} valid token 8 rank {rank} row {row}"
                );
            }
        }
        // Dead tail tokens 9..16 of tile 1 store nothing: full sentinel.
        for token in 9..16 {
            for rank in 0..KTOP {
                assert!(
                    is_sentinel_at(&mem, "yg", ((token * KTOP + rank) * DOWN_M) * 4, DOWN_M * 4),
                    "{arch:?} dead tail token {token} rank {rank} written"
                );
            }
        }
        assert_eq!(
            written_bytes(&mem, &["yg"]),
            DOWN_ROWS.len() * KTOP * r * 4,
            "{arch:?}: exactly sampled rows x ranks x all nine live tokens written"
        );
        assert!(is_guard_intact(&mem, "yg_lo"));
        assert!(is_guard_intact(&mem, "yg_hi"));
        eprintln!("down {arch:?} invalid-mask: ok");
    }
}

/// Down exact written-address sets at unaligned R=7.
#[test]
fn down_store_addresses_follow_slot_coords() {
    let arch = Arch::Gfx1201;
    let sym = format!(
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
        arch.name()
    );
    let prog = lift(
        down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
        arch,
        "down",
    );
    let emu = Emulator::new(&prog, &sym).unwrap();
    let mut rng = Rng(0xd0ad_d0ad_d0ad_d0ad);
    let tensor = down_expert_tensor(&mut rng);
    let r = 7usize;
    let (x, ids) = down_inputs(r, &mut rng);
    let mut mem = down_memory(r, &tensor, x, ids).unwrap();
    mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
        .unwrap();
    for (row, rank) in [(0u32, 0u32), (2559, 9)] {
        run_wg(&emu, &mut mem, [row, rank, 0], 32, "coords");
    }
    let mut expect = vec![false; 8 * KTOP * DOWN_M];
    for token in 0..7 {
        expect[token * KTOP * DOWN_M + 0] = true;
        expect[(token * KTOP + 9) * DOWN_M + 2559] = true;
    }
    let region = find_region(&mem, "yg");
    for (i, e) in expect.iter().enumerate() {
        let written = (0..4).any(|b| region.written.contains(i * 4 + b));
        assert_eq!(written, *e, "word {i}: written={written} expected={e}");
        if !e {
            assert!(
                region.bytes[i * 4..i * 4 + 4]
                    .iter()
                    .all(|&b| b == SENTINEL),
                "word {i}: untouched word not sentinel"
            );
        }
    }
}

/// Down finite extremes at aligned and tailed widths.
#[test]
fn down_extreme_activations() {
    for arch in [Arch::Gfx1201, Arch::Gfx1151] {
        let sym = format!(
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{}",
            arch.name()
        );
        let prog = lift(
            down::emit(arch).unwrap_or_else(|e| panic!("{arch:?}: {e}")),
            arch,
            "down",
        );
        let emu = Emulator::new(&prog, &sym).unwrap();
        let mut rng = Rng(0xe1e1_e1e1_e1e1_e1e1);
        let extreme: Vec<f32> = [
            1e20,
            -1e20,
            1e-30,
            -1e-30,
            f32::from_bits(1),
            0.0,
            -0.0,
            1.0,
        ]
        .into_iter()
        .cycle()
        .take(DOWN_K)
        .collect();
        let tensor = down_expert_tensor(&mut rng);
        for r in [8usize, 7] {
            let tiles = r.div_ceil(8);
            let x: Vec<f32> = (0..r * KTOP).flat_map(|_| extreme.clone()).collect();
            let mut ids = vec![0i32; r * KTOP];
            for t in 0..r {
                for rank in 0..KTOP {
                    ids[t * KTOP + rank] = ((t + rank) % NEXP as usize) as i32;
                }
            }
            let mut mem = down_memory(r, &tensor, x.clone(), ids.clone()).unwrap();
            mem.map("kernarg".into(), A_KERNARG, down_kernarg(r as i32))
                .unwrap();
            for row in [0usize, 1280, 2559] {
                for rank in [0u32, 9] {
                    for bz in 0..tiles as u32 {
                        run_wg(
                            &emu,
                            &mut mem,
                            [row as u32, rank, bz],
                            32,
                            &format!("{arch:?} extreme R{r}"),
                        );
                    }
                }
            }
            for row in [0usize, 1280, 2559] {
                for rank in [0u32, 9] {
                    for bz in 0..tiles {
                        for t in 0..8usize {
                            let token = bz * 8 + t;
                            if token >= r {
                                continue;
                            }
                            let expect = down_expected(&tensor, &x, token, rank as usize, row);
                            assert!(expect.is_finite(), "{arch:?} extreme R{r}: ref not finite");
                            let got =
                                read_f32(&mem, "yg", (token * KTOP + rank as usize) * DOWN_M + row);
                            assert_eq!(
                                got.to_bits(),
                                expect.to_bits(),
                                "{arch:?} extreme R{r} token {token} rank {rank} row {row}"
                            );
                        }
                    }
                }
            }
            eprintln!("down {arch:?} extreme R={r}: ok");
        }
    }
}
