// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reusable harness: the production `iu4_gemm` kernel on the PeaceMaker CPU emulator (include with
//! `#[path = "support/iu4_emu.rs"] mod iu4_emu;`). The kernel is emitted, assembled to a native
//! code object, lifted, and every workgroup of the launch grid runs on real kernargs and mapped buffers, using the
//! launch contract of `iu4_gemm/mod.rs` (grid `[ceil(rows/128), ceil(N/128)]` for the 128-row tile, block
//! `tile.threads()`, dynamic LDS `Layout::launch`).
//!
//! The reusable harness is [`run_gemm`] with the input builders [`Problem::uniform`] / [`Problem::synth`]
//! (MQ4G256V2 weights, `block_i4_128` token-layout activations) and the exact CPU fold [`reference`]: a sparse or
//! otherwise edited kernel is checked by building its weight bytes, calling [`run_gemm`] with its `Spec`, and
//! comparing the returned `Y` against [`reference`] on the dense equivalent, or bitwise against the dense kernel.
//!
//! Reference numerics (production `mq4v2_symmetric` fold, builder emitters): per K128 block `kb` in ascending
//! order `dot = sum (code ^ 8 as signed) * q` (exact i32), `t1 = RN(sc * d)`, `acc = fma(t1, dot, acc)` from
//! `acc = +0.0`. Output `Y[col * M + row]`, f32; ADD is `RN(Y0 + sum)`.
#![allow(dead_code)]
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Epi, Spec};
use peacemaker_isa::Arch;

/// Bytes of one MQ4G256V2 group (two K128 blocks): `[sc0 zp0 sc1 zp1]` f16 pairs, then 128 code bytes.
pub const GROUP_BYTES: usize = 136;
/// Bytes of one `block_i4_128` activation record: `d` f32, `sum_q` i32, 64 code bytes.
pub const REC: usize = 72;

/// xorshift64*.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

/// f32 -> f16 bits, round to nearest even, normal range only (the weight scales are `2^-6..2^0`).
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let e = (b >> 23 & 255) as i32 - 127 + 15;
    assert!((1..=30).contains(&e), "{x} is outside the f16 normal range");
    let (mut m, rest) = (b & 0x7f_ffff, 0x1000);
    let tail = m & 0x1fff;
    m >>= 13;
    let mut out = ((b >> 16) & 0x8000) | ((e as u32) << 10) | m;
    if tail > rest || (tail == rest && m & 1 == 1) {
        out += 1; // a carry into the exponent is the correct RNE result
    }
    out as u16
}

/// f16 bits (normal or zero) -> f32.
pub fn f16_val(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from(h >> 10 & 31),
        u32::from(h & 0x3ff),
    );
    assert!(
        e != 31 && (e != 0 || m == 0),
        "{h:#x} is not a finite normal f16"
    );
    if e == 0 {
        return f32::from_bits(s << 31);
    }
    f32::from_bits(s << 31 | (e + 112) << 23 | m << 13)
}

/// One `block_i4_128` record: per-block scale `d`, `sum(q)` and 128 signed 4-bit codes (low nibble = even K).
pub fn record(d: f32, q: &[i32; 128]) -> [u8; REC] {
    let mut rec = [0u8; REC];
    rec[0..4].copy_from_slice(&d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for j in 0..64 {
        rec[8 + j] = (q[2 * j] as u8 & 0xf) | ((q[2 * j + 1] as u8 & 0xf) << 4);
    }
    rec
}

/// `d = max|x| / 7`, `q = clamp(round(x/d), -8, 7)`.
fn quantize_block(x: &[f32; 128]) -> [u8; REC] {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 7.0;
    let mut q = [0i32; 128];
    if d > 0.0 {
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = ((v / d).round() as i32).clamp(-8, 7);
        }
    }
    record(d, &q)
}

/// Host-side inputs of one SET/ADD GEMM.
pub struct Problem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// MQ4G256V2 weights `[M x K/256 x 136]`.
    pub w: Vec<u8>,
    /// `block_i4_128` records, token layout: record `kb * N + col`.
    pub recs: Vec<u8>,
    /// Initial `Y` for ADD, `[N x M]` (`Y[col * M + row]`).
    pub y0: Vec<f32>,
}

impl Problem {
    /// Every weight code `code` (signed value `code - 8`), every activation code `q`, unit scales (`sc = 1`, `d = 1`):
    /// `Y = K * (code - 8) * q` exactly.
    pub fn uniform(m: usize, n: usize, k: usize, code: u8, q: i32) -> Self {
        let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
        for blk in w.chunks_exact_mut(GROUP_BYTES) {
            for h in 0..2 {
                blk[4 * h..4 * h + 2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
                blk[4 * h + 2..4 * h + 4].copy_from_slice(&f16_bits(-8.0).to_le_bytes());
            }
            blk[8..].fill(code | code << 4);
        }
        let rec = record(1.0, &[q; 128]);
        let recs = rec
            .iter()
            .copied()
            .cycle()
            .take(k / 128 * n * REC)
            .collect();
        Self {
            m,
            n,
            k,
            w,
            recs,
            y0: vec![0.0; n * m],
        }
    }

    /// Deterministic random weights (scales `0.02..0.05`, random codes) and activations (per-column amplitude).
    pub fn synth(m: usize, n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut w = vec![0u8; m * (k / 256) * GROUP_BYTES];
        for blk in w.chunks_exact_mut(GROUP_BYTES) {
            let sc0 = 0.02 + (rng.next() % 16) as f32 * 0.001;
            let sc1 = 0.03 + (rng.next() % 16) as f32 * 0.001;
            for (i, sc) in [sc0, sc1].into_iter().enumerate() {
                let bits = f16_bits(sc);
                blk[4 * i..4 * i + 2].copy_from_slice(&bits.to_le_bytes());
                // zp is ignored by the symmetric fold; keep the asymmetric value.
                let zp = f16_bits(-8.0 * f16_val(bits));
                blk[4 * i + 2..4 * i + 4].copy_from_slice(&zp.to_le_bytes());
            }
            for q in blk[8..].chunks_exact_mut(8) {
                q.copy_from_slice(&rng.next().to_le_bytes());
            }
        }
        let mut recs = vec![0u8; k / 128 * n * REC];
        for kb in 0..k / 128 {
            for col in 0..n {
                let amp = 1.0 + (col % 4) as f32;
                let mut x = [0f32; 128];
                for v in &mut x {
                    *v = rng.unit() * amp;
                }
                let o = (kb * n + col) * REC;
                recs[o..o + REC].copy_from_slice(&quantize_block(&x));
            }
        }
        let y0 = (0..n * m).map(|_| 0.25 + rng.unit()).collect();
        Self {
            m,
            n,
            k,
            w,
            recs,
            y0,
        }
    }
}

/// Token-layout records to the slab planes `[d:4N][s:4N][qs0..32:32N][qs32..64:32N]` per K128 block.
pub fn to_slab(recs: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; recs.len()];
    for (kb, blk) in out.chunks_exact_mut(REC * n).enumerate() {
        for col in 0..n {
            let r = &recs[(kb * n + col) * REC..][..REC];
            blk[4 * col..4 * col + 4].copy_from_slice(&r[0..4]);
            blk[4 * n + 4 * col..4 * n + 4 * col + 4].copy_from_slice(&r[4..8]);
            blk[8 * n + 32 * col..8 * n + 32 * col + 32].copy_from_slice(&r[8..40]);
            blk[40 * n + 32 * col..40 * n + 32 * col + 32].copy_from_slice(&r[40..72]);
        }
    }
    out
}

/// Exact CPU fold of the dense kernel: `Y[col * M + row]` for `add` (`y0 + sum`) or SET.
pub fn reference(p: &Problem, add: bool) -> Vec<f32> {
    reference_with(p, add, true)
}

/// [`reference`] with the stage-2 accumulate chosen: `fused` is `fma(t1, dot, sum)`, otherwise the non-fused
/// `sum + RN(t1 * dot)` (`FoldOp::MulAdd`).
pub fn reference_with(p: &Problem, add: bool, fused: bool) -> Vec<f32> {
    let (m, n, k) = (p.m, p.n, p.k);
    let (gpr, nkb) = (k / 256, k / 128);
    let mut xq = Vec::with_capacity(nkb * n * 128);
    let mut xd = Vec::with_capacity(nkb * n);
    for r in p.recs.chunks_exact(REC) {
        xd.push(f32::from_le_bytes(r[0..4].try_into().unwrap()));
        for &b in &r[8..] {
            xq.push(i32::from(((b << 4) as i8) >> 4));
            xq.push(i32::from((b as i8) >> 4));
        }
    }
    let mut y = vec![0f32; n * m];
    let mut wq = vec![0i32; k];
    let mut sc = vec![0f32; nkb];
    for row in 0..m {
        for g in 0..gpr {
            let b = &p.w[(row * gpr + g) * GROUP_BYTES..][..GROUP_BYTES];
            for h in 0..2 {
                sc[2 * g + h] = f16_val(u16::from_le_bytes([b[4 * h], b[4 * h + 1]]));
            }
            for (j, &byte) in b[8..].iter().enumerate() {
                // nibble ^ 8 as signed 4-bit (low nibble = even K).
                wq[256 * g + 2 * j] = i32::from(((((byte & 0xf) ^ 8) << 4) as i8) >> 4);
                wq[256 * g + 2 * j + 1] = i32::from(((((byte >> 4) ^ 8) << 4) as i8) >> 4);
            }
        }
        for col in 0..n {
            let mut sum = 0f32;
            for kb in 0..nkb {
                let x = &xq[(kb * n + col) * 128..][..128];
                let w = &wq[kb * 128..][..128];
                let dot: i32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                let t1 = sc[kb] * xd[kb * n + col];
                sum = if fused {
                    t1.mul_add(dot as f32, sum)
                } else {
                    sum + t1 * dot as f32
                };
            }
            y[col * m + row] = if add { p.y0[col * m + row] + sum } else { sum };
        }
    }
    y
}

fn put_ptr(b: &mut Vec<u8>, p: u64) {
    b.extend_from_slice(&p.to_le_bytes());
}

/// Assemble and lift `spec`'s kernel; every opcode without emulator semantics is reported by `Emulator::new`.
/// Runs the whole launch grid (SET/ADD, 128-row or 256-row tile, token or slab activations) and returns
/// `Y[col * M + row]` read back from the output buffer. Every buffer is followed by a zeroed 4 KiB tail region,
/// so a kernel store or load past the end of a buffer is caught as a changed tail.
pub fn run_gemm(spec: Spec, p: &Problem) -> Result<Vec<f32>, String> {
    run_emitted(spec, &iu4_gemm::emit(spec)?, p)
}

/// [`run_gemm`] for an already-emitted kernel (for example one lowered from a `.rip` body): `emitted.s_text` is
/// assembled, lifted and run under `spec`'s launch contract.
pub fn run_emitted(
    spec: Spec,
    emitted: &peacemaker_isa::Emitted,
    p: &Problem,
) -> Result<Vec<f32>, String> {
    assert_eq!(spec.arch, Arch::Gfx1201);
    let add = match spec.epi {
        Epi::Set => false,
        Epi::Add => true,
        e => return Err(format!("run_gemm drives only set/add, not {}", e.name())),
    };
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, spec.arch)?;
    let lifted = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?;
    let symbol = spec.symbol();
    let emu = Emulator::new(&lifted.program, &symbol)?;
    let (m, n, k) = (p.m, p.n, p.k);
    let (kva, ava, xva, yva) = (
        0x1000_0000u64,
        0x2000_0000u64,
        0x3000_0000u64,
        0x4000_0000u64,
    );
    let xq = match spec.act {
        ALayout::Token => p.recs.clone(),
        ALayout::Slab => to_slab(&p.recs, n),
    };
    // SET/ADD explicit kernarg: `A Xq Y` at 0/8/16, `M K N add` at 24/28/32/36.
    let mut kern = Vec::new();
    for ptr in [ava, xva, yva] {
        put_ptr(&mut kern, ptr);
    }
    for v in [m, k, n, usize::from(add)] {
        kern.extend_from_slice(&(v as i32).to_le_bytes());
    }
    kern.resize(spec.kernargs().size as usize, 0);
    // ROCm fills the hidden block counts (grid size in workgroups) when it launches the kernel; the banded raster of
    // the prologue divides by them.
    let grid = [
        m.div_ceil(spec.tile.rows() as usize) as u32,
        n.div_ceil(128) as u32,
    ];
    for a in spec.kernargs().args {
        let g = match a.name.as_str() {
            "hidden_block_count_x" => grid[0],
            "hidden_block_count_y" => grid[1],
            _ => continue,
        };
        kern[a.offset as usize..a.offset as usize + 4].copy_from_slice(&g.to_le_bytes());
    }
    let y0: Vec<u8> = if add {
        p.y0.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        vec![0; n * m * 4]
    };
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kva, kern)?;
    for (name, va, bytes) in [("a", ava, p.w.clone()), ("xq", xva, xq), ("y", yva, y0)] {
        let len = bytes.len() as u64;
        mem.map(name.into(), va, bytes)?;
        mem.map(format!("{name}_tail"), va + len, vec![0; 4096])?;
    }
    let tile = spec.tile;
    for y in 0..n.div_ceil(128) {
        for x in 0..m.div_ceil(tile.rows() as usize) {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: kva,
                    workgroup: [x as u32, y as u32, 0],
                    block: [tile.threads(), 1, 1],
                    dynamic_lds: spec.lds_launch(),
                },
                &mut mem,
            )
            .map_err(|e| format!("workgroup ({x},{y}): {e}"))?;
        }
    }
    for name in ["a_tail", "xq_tail", "y_tail"] {
        let r = mem.regions.iter().find(|r| r.name == name).unwrap();
        if r.bytes.iter().any(|&b| b != 0) {
            return Err(format!("kernel wrote past the end of the {name} buffer"));
        }
    }
    Ok(mem
        .read(yva, n * m * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
