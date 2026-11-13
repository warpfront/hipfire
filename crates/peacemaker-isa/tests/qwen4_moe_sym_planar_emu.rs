// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The F1 drop-zero planar twins of the gfx1201 sealed IU4 symmetric MoE entries on the PeaceMaker emulator.
//!
//! The same logical experts (codes and f16 scales; zero fields `f16(-8 * scale)`, the loader's symmetric-header
//! check) are laid out twice: the shipped QT44 / QT53 bytes (`WeightCodec::Mq4Sym`, the oracle) and the planar bytes
//! (`[codes plane: M*K/2][scales plane: M*(K/128) f16]`, `WeightCodec::Mq4SymPlanar`, see
//! `hipfire_runtime::mq4_planar`). Every emitted entry (gate/up and down NT4/NT8, flat-slot, and the grouped-position
//! down twins) is assembled, lifted and run over its whole grid on real kernargs and exactly sized buffers, and the
//! planar entry's `Y_grouped` must equal the shipped entry's byte for byte (unwritten bytes included). The down
//! outputs are also checked against the exact CPU fold, so the oracle itself is not trusted blindly.
//!
//! Experts are routed through a non-monotone tile list (runs longer than the tile width, a partial last tile of each
//! run, padding lanes, a sentinel tile), and `K` covers a zero-trip loop, an odd tail epoch and several trips.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe, Kind, Spec, WeightCodec};
use peacemaker_isa::{native, Arch};

#[allow(dead_code)]
#[path = "support/iu4_emu.rs"]
mod iu4_emu;
use iu4_emu::{f16_bits, f16_val, record, Rng, REC};

const ARCH: Arch = Arch::Gfx1201;
const KARG: u64 = 0x10_0000;
const PTRS: u64 = 0x11_0000;
const TILES: u64 = 0x12_0000;
const SORTED: u64 = 0x13_0000;
const XQ: u64 = 0x2000_0000;
const Y: u64 = 0x3000_0000;
const WBASE: u64 = 0x4000_0000;
const WSTRIDE: u64 = 0x100_0000;
const SENTINEL: u8 = 0xa5;

/// One expert tensor `[m, k]`: 4-bit codes and f16 scale bits per 128 columns.
struct Tensor {
    m: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<u16>,
}

impl Tensor {
    fn random(m: usize, k: usize, rng: &mut Rng) -> Tensor {
        let codes = (0..m * k).map(|_| (rng.next() >> 33) as u8 & 15).collect();
        // Normal f16 scales over `2^-6 .. 2^0`, including exact powers of two.
        let scales = (0..m * (k / 128))
            .map(|i| {
                let u = (rng.unit() + 1.0) * 0.5;
                f16_bits(if i % 5 == 0 {
                    0.015625 * (1 << (i % 7)) as f32
                } else {
                    0.015625 + u * 0.98
                })
            })
            .collect();
        Tensor {
            m,
            k,
            codes,
            scales,
        }
    }

    /// Row `r`'s code bytes (`k/2`): low nibble = even column.
    fn row_bytes(&self, r: usize) -> Vec<u8> {
        (0..self.k / 2)
            .map(|b| self.codes[r * self.k + 2 * b] | self.codes[r * self.k + 2 * b + 1] << 4)
            .collect()
    }

    fn scale(&self, r: usize, h: usize) -> u16 {
        self.scales[r * (self.k / 128) + h]
    }

    /// The loader's symmetric header zero: `f16(-8 * scale)`.
    fn zero(&self, r: usize, h: usize) -> u16 {
        f16_bits(-8.0 * f16_val(self.scale(r, h)))
    }

    /// Shipped QT44 (`MQ4G256V2`, 136 B per 256 columns) or QT53 (`MQ4G128V2`, 68 B per 128 columns) bytes.
    fn shipped(&self, kind: Kind) -> Vec<u8> {
        let mut out = Vec::new();
        for r in 0..self.m {
            let bytes = self.row_bytes(r);
            match kind {
                Kind::GateUp => {
                    for g in 0..self.k / 256 {
                        for h in [2 * g, 2 * g + 1] {
                            out.extend(self.scale(r, h).to_le_bytes());
                            out.extend(self.zero(r, h).to_le_bytes());
                        }
                        out.extend(&bytes[g * 128..(g + 1) * 128]);
                    }
                }
                Kind::Down => {
                    for h in 0..self.k / 128 {
                        out.extend(self.scale(r, h).to_le_bytes());
                        out.extend(self.zero(r, h).to_le_bytes());
                        out.extend(&bytes[h * 64..(h + 1) * 64]);
                    }
                }
            }
        }
        out
    }

    /// Planar QT137 / QT138 bytes: `[codes plane][scales plane]`.
    fn planar(&self) -> Vec<u8> {
        let mut out: Vec<u8> = (0..self.m).flat_map(|r| self.row_bytes(r)).collect();
        out.extend(self.scales.iter().flat_map(|s| s.to_le_bytes()));
        out
    }
}

/// Routing and activations of one launch.
struct Case {
    kind: Kind,
    m: usize,
    k: usize,
    experts: Vec<Tensor>,
    /// Expert id per tile (`-1`: a sentinel tile past the last run).
    tiles: Vec<i32>,
    /// 16 slots per tile (`-1`: padding).
    slots: Vec<i32>,
    x_row_div: usize,
    x_src_rows: usize,
    /// Signed activation codes `[epoch][source row][128]` and per-record scales.
    xq: Vec<[i32; 128]>,
    xd: Vec<f32>,
}

impl Case {
    fn new(kind: Kind, nt: u8, k: usize, seed: u64) -> Case {
        let m = 128;
        let mut rng = Rng::new(seed);
        let experts: Vec<Tensor> = (0..3).map(|_| Tensor::random(m, k, &mut rng)).collect();
        // Runs of tiles per expert: longer than `nt` for the first (a leader at 0 and at `nt`), partial last tiles.
        let nt = nt as usize;
        let runs = [(2usize, nt + 1), (0, 2), (1, 1)];
        let mut tiles = Vec::new();
        let mut live_per_tile = Vec::new();
        for (e, n) in runs {
            for t in 0..n {
                tiles.push(e as i32);
                live_per_tile.push(if t + 1 == n { 3 + e + t % 2 } else { 16 });
            }
        }
        let total_live: usize = live_per_tile.iter().sum();
        let mut perm: Vec<i32> = (0..total_live as i32).collect();
        for i in (1..perm.len()).rev() {
            perm.swap(i, (rng.next() >> 20) as usize % (i + 1));
        }
        let mut slots = Vec::new();
        let mut next = 0;
        for &live in &live_per_tile {
            for lane in 0..16 {
                if lane < live {
                    slots.push(perm[next]);
                    next += 1;
                } else {
                    slots.push(-1);
                }
            }
        }
        // The sentinel tile: expert -1, every slot padding.
        tiles.push(-1);
        slots.extend([-1; 16]);
        let x_row_div = if kind == Kind::GateUp { 2 } else { 1 };
        let x_src_rows = total_live.div_ceil(x_row_div);
        let epochs = k / 128;
        let (mut xq, mut xd) = (Vec::new(), Vec::new());
        for _ in 0..epochs * x_src_rows {
            let mut q = [0i32; 128];
            for v in q.iter_mut() {
                *v = ((rng.next() >> 35) % 16) as i32 - 8;
            }
            xq.push(q);
            xd.push(0.01 + (rng.unit() + 1.0) * 0.1);
        }
        Case {
            kind,
            m,
            k,
            experts,
            tiles,
            slots,
            x_row_div,
            x_src_rows,
            xq,
            xd,
        }
    }

    fn m_total(&self) -> usize {
        self.tiles.len() * 16
    }

    fn row_len(&self) -> usize {
        if self.kind == Kind::GateUp {
            self.m / 2
        } else {
            self.m
        }
    }

    fn rec(&self, epoch: usize, row: usize) -> [u8; REC] {
        let i = epoch * self.x_src_rows + row;
        record(self.xd[i], &self.xq[i])
    }

    /// The activation plane: flat `[epoch][x_src_rows]` records, or the grouped-position
    /// `[epoch][m_total]` plane (record `16 * tile + lane`; padding positions stay zero).
    fn xplane(&self, gp: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for h in 0..self.k / 128 {
            if gp {
                for pos in 0..self.m_total() {
                    match self.slots[pos] {
                        s if s >= 0 => out.extend(self.rec(h, s as usize / self.x_row_div)),
                        _ => out.extend([0u8; REC]),
                    }
                }
            } else {
                for row in 0..self.x_src_rows {
                    out.extend(self.rec(h, row));
                }
            }
        }
        out
    }
}

fn lift(spec: Spec, gp: bool) -> Result<(peacemaker_ir::Program, String), String> {
    let emitted = if gp {
        moe::emit_down_gp(spec)
    } else {
        moe::emit(spec)
    }?;
    let symbol = if gp { spec.symbol_gp() } else { spec.symbol() };
    let elf = native::assemble(&emitted.s_text, ARCH)?;
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?
    .program;
    Ok((program, symbol))
}

/// Run `spec` over the whole grid and return `Y_grouped`. `planar` selects the weight byte layout (it must match the
/// codec of `spec`).
fn run(spec: Spec, gp: bool, case: &Case) -> Result<Vec<u8>, String> {
    let planar = spec.codec == WeightCodec::Mq4SymPlanar;
    let (program, symbol) = lift(spec, gp)?;
    let emu = Emulator::new(&program, &symbol)?;
    let m_total = case.m_total();
    let mut mem = Memory::default();
    let mut karg = Vec::new();
    for ptr in [PTRS, TILES, SORTED, XQ, Y] {
        karg.extend(ptr.to_le_bytes());
    }
    for v in [
        case.m,
        case.k,
        case.x_row_div,
        m_total,
        case.x_src_rows,
        case.experts.len(),
    ] {
        karg.extend((v as u32).to_le_bytes());
    }
    assert_eq!(karg.len(), moe::KERNARG_BYTES as usize);
    mem.map("karg".into(), KARG, karg)?;
    let mut ptrs = Vec::new();
    for (e, tensor) in case.experts.iter().enumerate() {
        let base = WBASE + e as u64 * WSTRIDE;
        ptrs.extend(base.to_le_bytes());
        // Exactly sized: a weight or scale load past the tensor is an unmapped access.
        let bytes = if planar {
            tensor.planar()
        } else {
            tensor.shipped(case.kind)
        };
        mem.map(format!("expert{e}"), base, bytes)?;
    }
    mem.map("ptrs".into(), PTRS, ptrs)?;
    mem.map(
        "tiles".into(),
        TILES,
        case.tiles.iter().flat_map(|t| t.to_le_bytes()).collect(),
    )?;
    mem.map(
        "sorted".into(),
        SORTED,
        case.slots.iter().flat_map(|t| t.to_le_bytes()).collect(),
    )?;
    mem.map("xq".into(), XQ, case.xplane(gp))?;
    mem.map("y".into(), Y, vec![SENTINEL; m_total * case.row_len() * 2])?;
    for tile in 0..m_total / 16 {
        for x in 0..case.m / 64 {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KARG,
                    workgroup: [x as u32, tile as u32, 0],
                    block: [spec.threads(), 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("workgroup ({x},{tile}): {e}"))?;
        }
    }
    Ok(mem.read(Y, m_total * case.row_len() * 2)?.to_vec())
}

fn bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The exact CPU fold of the down projection: per K128 epoch `h` in ascending order `dot` (exact), `t1 = RN(d * sc)`,
/// `sum = fma(t1, dot, sum)` from `+0`, stored as RNE BF16; a padding slot stores `+0`; a sentinel tile stores nothing.
fn down_reference(case: &Case) -> Vec<u8> {
    let mut y = vec![SENTINEL; case.m_total() * case.m * 2];
    for (tile, &e) in case.tiles.iter().enumerate() {
        if e < 0 {
            continue;
        }
        let t = &case.experts[e as usize];
        for lane in 0..16 {
            let pos = 16 * tile + lane;
            let slot = case.slots[pos];
            for row in 0..case.m {
                let out = if slot < 0 {
                    0
                } else {
                    let src = slot as usize / case.x_row_div;
                    let mut sum = 0f32;
                    for h in 0..case.k / 128 {
                        let i = h * case.x_src_rows + src;
                        let dot: i32 = (0..128)
                            .map(|j| {
                                (i32::from(t.codes[row * case.k + h * 128 + j]) - 8) * case.xq[i][j]
                            })
                            .sum();
                        let t1 = case.xd[i] * f16_val(t.scale(row, h));
                        sum = t1.mul_add(dot as f32, sum);
                    }
                    bf16_rne(sum)
                };
                let at = (pos * case.m + row) * 2;
                y[at..at + 2].copy_from_slice(&out.to_le_bytes());
            }
        }
    }
    y
}

fn spec(kind: Kind, nt: u8, codec: WeightCodec) -> Spec {
    Spec {
        arch: ARCH,
        kind,
        nt,
        rr: 1,
        codec,
    }
}

/// Bytes `Y_grouped` of the shipped entry and its planar twin on `case`, asserted equal and informative.
fn twin(kind: Kind, nt: u8, gp: bool, k: usize, seed: u64) -> (Case, Vec<u8>) {
    let case = Case::new(kind, nt, k, seed);
    let shipped = run(spec(kind, nt, WeightCodec::Mq4Sym), gp, &case)
        .unwrap_or_else(|e| panic!("shipped {kind:?} nt{nt} gp={gp} K={k}: {e}"));
    let planar = run(spec(kind, nt, WeightCodec::Mq4SymPlanar), gp, &case)
        .unwrap_or_else(|e| panic!("planar {kind:?} nt{nt} gp={gp} K={k}: {e}"));
    assert_eq!(shipped.len(), planar.len());
    if let Some(i) = shipped.iter().zip(&planar).position(|(a, b)| a != b) {
        panic!(
            "{kind:?} nt{nt} gp={gp} K={k}: planar Y differs from the shipped entry's at byte {i} (row {}, col {})",
            i / 2 / case.row_len(),
            i / 2 % case.row_len()
        );
    }
    // Every output of every live tile is written; only the sentinel tile stays untouched.
    let live = shipped
        .chunks_exact(2)
        .filter(|c| c[0] != SENTINEL || c[1] != SENTINEL)
        .count();
    assert_eq!(
        live,
        shipped.len() / 2 - 16 * case.row_len(),
        "written outputs"
    );
    assert!(shipped.chunks_exact(2).any(|c| c[0] != 0 || c[1] != 0));
    let tail = &shipped[(shipped.len() - 16 * case.row_len() * 2)..];
    assert!(tail.iter().all(|&b| b == SENTINEL), "sentinel tile written");
    (case, shipped)
}

#[test]
fn down_planar_equals_shipped_and_the_cpu_fold() {
    for nt in [4u8, 8] {
        // K/128 = 2 (no full trip), 3 (one trip and an odd tail), 5 (two trips and an odd tail).
        for k in [256usize, 384, 640] {
            let (case, y) = twin(Kind::Down, nt, false, k, 0x1000 + k as u64 + u64::from(nt));
            let want = down_reference(&case);
            assert!(
                y == want,
                "down nt{nt} K={k}: shipped Y differs from the CPU fold"
            );
        }
    }
}

#[test]
fn gate_up_planar_equals_shipped() {
    for nt in [4u8, 8] {
        // K/256 = 1 (no full trip), 3 (one trip and a pair), 4 (two trips and a pair).
        for k in [256usize, 768, 1024] {
            twin(
                Kind::GateUp,
                nt,
                false,
                k,
                0x2000 + k as u64 + u64::from(nt),
            );
        }
    }
}

#[test]
fn grouped_position_down_planar_equals_shipped_and_the_flat_fold() {
    for nt in [4u8, 8] {
        for k in [256usize, 640] {
            let seed = 0x3000 + k as u64 + u64::from(nt);
            let (case, y) = twin(Kind::Down, nt, true, k, seed);
            // The grouped-position planar entry reads the same activations as the flat one: its bytes are the CPU fold's.
            assert!(
                y == down_reference(&case),
                "down gp nt{nt} K={k}: Y differs from the CPU fold"
            );
        }
    }
}
