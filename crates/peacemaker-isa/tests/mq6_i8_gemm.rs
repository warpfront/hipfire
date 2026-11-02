// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `mq6_i8_gemm` on the peacemaker CPU interpreter: the linear-raster twin of both entries must equal, bit for bit,
//! a Rust transcription of the hipcc kernel's arithmetic (`gemm_mq6g256v2_mmq_i8.gfx12.hip`): per K128 block the
//! exact integer chain `C = sum (q - 32) * qx`, then `acc = fma(RN(s_h * d), fc, acc)`,
//! `acc = fma(RN(z'_h * d), (float)s, acc)` with `z' = fma(32, s_h, z_h)`; BF16 = the incumbent's `isfinite ? RNE :
//! truncate` of `0 + y`. Shapes cover M and N tails (ragged row / token tiles), a single peeled group (K = 256), one
//! loop trip plus the peeled group (K = 512) and an odd trip count; every word outside Y's `M * N` and every
//! out-of-tile word is checked against a sentinel.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq6_i8_gemm::{self, spec, Epi, RingDepth, Spec, Tile, Tune};
use peacemaker_isa::Arch;

const KERNARG: u64 = 0x1_0000;
const A_BASE: u64 = 0x100_0000;
const X_BASE: u64 = 0x1000_0000;
const Y_BASE: u64 = 0x2000_0000;
const GUARD: usize = 256;
const SENTINEL: u8 = 0xa5;

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

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 31),
        u32::from(h & 0x3ff),
    );
    assert!(e != 0 && e != 31, "test data uses normal halves only");
    f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
}

/// Normal f16 with exponent field `lo..lo+span`.
fn half(rng: &mut Rng, lo: usize, span: usize) -> u16 {
    (((rng.next() & 1) as u16) << 15)
        | (((lo + rng.below(span)) as u16) << 10)
        | (rng.next() as u16 & 0x3ff)
}

/// Activation fill: full-range random, all `-1` (a negative chain sum: the seed must stay centred), or a
/// single `+1` at qs byte `j` of every block (isolates one weight code).
#[derive(Clone, Copy)]
enum Fill {
    Random,
    NegOnes,
    OneHot(usize),
}
struct Case {
    m: usize,
    k: usize,
    n: usize,
    fill: Fill,
}

struct Data {
    /// `[M][K/256][200]`.
    a: Vec<u8>,
    /// `[K/128][N][136]`.
    x: Vec<u8>,
}

fn code(group: &[u8], half: usize, c: usize) -> u32 {
    let bits = 6 * c;
    let payload = &group[8 + 96 * half..8 + 96 * half + 96];
    let (byte, sh) = (bits / 8, bits % 8);
    let v = u32::from(payload[byte]) | u32::from(*payload.get(byte + 1).unwrap_or(&0)) << 8;
    (v >> sh) & 63
}

fn make(rng: &mut Rng, c: &Case) -> Data {
    let g = c.k / 256;
    let mut a = vec![0u8; c.m * g * 200];
    for r in 0..c.m {
        for gi in 0..g {
            let base = (r * g + gi) * 200;
            for h in 0..2 {
                // Scales / zeros over a band that keeps every fold product a normal f32.
                let s = half(rng, 6, 12);
                let z = half(rng, 6, 12);
                a[base + 4 * h..base + 4 * h + 2].copy_from_slice(&s.to_le_bytes());
                a[base + 4 * h + 2..base + 4 * h + 4].copy_from_slice(&z.to_le_bytes());
            }
            for byte in &mut a[base + 8..base + 200] {
                *byte = rng.next() as u8;
            }
        }
    }
    let mut x = vec![0u8; (c.k / 128) * c.n * 136];
    for kb in 0..c.k / 128 {
        for t in 0..c.n {
            let base = (kb * c.n + t) * 136;
            // Mix of full-range and small |q| blocks; the scale `d` over a wide positive band.
            let wide = rng.below(3) != 0;
            let mut sum = 0i32;
            for j in 0..128 {
                let q = if wide {
                    rng.below(255) as i32 - 127
                } else {
                    rng.below(9) as i32 - 4
                };
                let q = match c.fill {
                    Fill::Random => q,
                    Fill::NegOnes => -1,
                    Fill::OneHot(at) => i32::from(j == at),
                };
                sum += q;
                x[base + 8 + j] = q as i8 as u8;
            }
            let d = f32::from_bits(
                (((110 + rng.below(18)) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff),
            );
            x[base..base + 4].copy_from_slice(&d.to_le_bytes());
            x[base + 4..base + 8].copy_from_slice(&sum.to_le_bytes());
        }
    }
    Data { a, x }
}

/// Natural K index (within the K128 block) of qs byte `j` (nibble-pair order).
fn nat(j: usize) -> usize {
    let (g, r) = (j / 8, j % 8);
    8 * g + if r < 4 { 2 * r } else { 2 * (r - 4) + 1 }
}

fn reference(c: &Case, d: &Data, epi: Epi) -> Vec<u8> {
    let g = c.k / 256;
    let mut out = Vec::new();
    for t in 0..c.n {
        for r in 0..c.m {
            let mut acc = 0.0f32;
            for kb in 0..c.k / 128 {
                let (gi, h) = (kb / 2, kb % 2);
                let grp = &d.a[(r * g + gi) * 200..(r * g + gi) * 200 + 200];
                let xb = &d.x[(kb * c.n + t) * 136..(kb * c.n + t) * 136 + 136];
                let (dd, s) = (
                    f32::from_le_bytes(xb[0..4].try_into().unwrap()),
                    i32::from_le_bytes(xb[4..8].try_into().unwrap()),
                );
                let mut cq = 0i32;
                for j in 0..128 {
                    cq += code(grp, h, nat(j)) as i32 * i32::from(xb[8 + j] as i8);
                }
                let fc = (cq - 32 * s) as f32;
                let sh = f16_to_f32(u16::from_le_bytes(
                    grp[4 * h..4 * h + 2].try_into().unwrap(),
                ));
                let zh = f16_to_f32(u16::from_le_bytes(
                    grp[4 * h + 2..4 * h + 4].try_into().unwrap(),
                ));
                let zp = 32.0f32.mul_add(sh, zh);
                acc = (sh * dd).mul_add(fc, acc);
                acc = (zp * dd).mul_add(s as f32, acc);
            }
            match epi {
                Epi::Set | Epi::SetNr => out.extend(acc.to_bits().to_le_bytes()),
                Epi::SetBf16 => {
                    let u = (0.0f32 + acc).to_bits();
                    let b = if (0.0f32 + acc).is_finite() {
                        (u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16
                    } else {
                        u >> 16
                    };
                    out.extend((b as u16).to_le_bytes());
                }
            }
        }
    }
    out
}

fn run(c: &Case, epi: Epi, seed: u64) {
    run_tile(c, epi, Tile::T128x128, seed);
}

/// Run `tile`'s kernel (linear raster) on the interpreter over the whole grid, require the exact
/// CPU reference bit for bit, and return the output bytes.
fn run_tile(c: &Case, epi: Epi, tile: Tile, seed: u64) -> Vec<u8> {
    run_spec(c, Spec::emulation(epi).with_tile(tile), seed)
}

/// [`run_tile`] for any emulation spec (generator knobs included).
fn run_spec(c: &Case, spec: Spec, seed: u64) -> Vec<u8> {
    let (epi, tile) = (spec.epi, spec.tile);
    let rng = &mut Rng(seed);
    let data = make(rng, c);
    let want = reference(c, &data, epi);
    let emitted = mq6_i8_gemm::emit(spec).unwrap();
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emu = Emulator::new(&program, &spec.symbol()).unwrap();

    let mut mem = Memory::default();
    let mut karg = vec![0u8; 296];
    karg[0..8].copy_from_slice(&A_BASE.to_le_bytes());
    karg[8..16].copy_from_slice(&X_BASE.to_le_bytes());
    karg[16..24].copy_from_slice(&(Y_BASE + GUARD as u64).to_le_bytes());
    karg[24..28].copy_from_slice(&(c.m as u32).to_le_bytes());
    karg[28..32].copy_from_slice(&(c.k as u32).to_le_bytes());
    karg[32..36].copy_from_slice(&(c.n as u32).to_le_bytes());
    let (gx, gy) = (
        c.m.div_ceil(tile.rows() as usize) as u32,
        c.n.div_ceil(tile.tokens() as usize) as u32,
    );
    karg[40..44].copy_from_slice(&gx.to_le_bytes());
    karg[44..48].copy_from_slice(&gy.to_le_bytes());
    karg[48..52].copy_from_slice(&1u32.to_le_bytes());
    mem.map("karg".into(), KERNARG, karg).unwrap();
    mem.map("a".into(), A_BASE, data.a.clone()).unwrap();
    mem.map("x".into(), X_BASE, data.x.clone()).unwrap();
    mem.map("y".into(), Y_BASE, vec![SENTINEL; want.len() + 2 * GUARD])
        .unwrap();
    for gy_i in 0..gy {
        for gx_i in 0..gx {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx_i, gy_i, 0],
                    block: [tile.threads(), 1, 1],
                    dynamic_lds: spec.launch_lds(),
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("wg({gx_i},{gy_i}): {e}"));
        }
    }
    let y = &mem.regions.iter().find(|r| r.name == "y").unwrap().bytes;
    assert!(
        y[..GUARD].iter().all(|&b| b == SENTINEL),
        "guard before Y clobbered"
    );
    assert!(
        y[GUARD + want.len()..].iter().all(|&b| b == SENTINEL),
        "guard after Y clobbered"
    );
    let got = &y[GUARD..GUARD + want.len()];
    let w = if epi == Epi::SetBf16 { 2 } else { 4 };
    for (i, (g, e)) in got.chunks(w).zip(want.chunks(w)).enumerate() {
        assert_eq!(
            g,
            e,
            "M={} K={} N={} {epi:?} {tile:?}: token {} row {}",
            c.m,
            c.k,
            c.n,
            i / c.m,
            i % c.m
        );
    }
    got.to_vec()
}

#[test]
fn single_group_tails_f32() {
    run(
        &Case {
            m: 48,
            k: 256,
            n: 40,
            fill: Fill::Random,
        },
        Epi::Set,
        1,
    );
}
#[test]
fn single_group_tails_bf16() {
    run(
        &Case {
            m: 52,
            k: 256,
            n: 24,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        2,
    );
}
#[test]
fn loop_and_peel_f32() {
    run(
        &Case {
            m: 132,
            k: 512,
            n: 130,
            fill: Fill::Random,
        },
        Epi::Set,
        3,
    );
}
#[test]
fn loop_and_peel_bf16() {
    run(
        &Case {
            m: 128,
            k: 768,
            n: 128,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        4,
    );
}

#[test]
fn negative_chain_sums_f32() {
    run(
        &Case {
            m: 8,
            k: 256,
            n: 16,
            fill: Fill::NegOnes,
        },
        Epi::Set,
        5,
    );
}
#[test]
fn every_code_position_is_paired_with_its_activation_byte() {
    for j in [0, 1, 3, 4, 7, 8, 31, 32, 63, 64, 95, 127] {
        run(
            &Case {
                m: 8,
                k: 256,
                n: 16,
                fill: Fill::OneHot(j),
            },
            Epi::Set,
            6,
        );
    }
}
#[test]
fn multi_tile_grid_with_three_loop_trips() {
    run(
        &Case {
            m: 260,
            k: 1024,
            n: 200,
            fill: Fill::Random,
        },
        Epi::Set,
        8,
    );
}
#[test]
fn multi_tile_grid_bf16() {
    run(
        &Case {
            m: 136,
            k: 512,
            n: 136,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        9,
    );
}

// ---------------------------------------------------------------------------------------------
// Tiles: every tile computes the same exact math (integer dot, the same per-element fold order), so
// each is bit-equal to the CPU reference and therefore to the shipped 128x128 tile. The shapes make
// edge tiles of every kind: M and N that are not multiples of the tile in either direction, grids
// of several tiles, one-trip and looped K.

fn tile_cases() -> Vec<Case> {
    let case = |m, k, n, fill| Case { m, k, n, fill };
    vec![
        // One ragged tile per direction of every tile size.
        case(48, 256, 40, Fill::Random),
        // Ragged in M past the 256-row tile and in N past the 256-token tile; K looped.
        case(300, 512, 150, Fill::Random),
        case(132, 512, 300, Fill::Random),
        // Whole tiles, looped K with three trips, and a negative chain.
        case(256, 1024, 256, Fill::Random),
        case(260, 768, 132, Fill::NegOnes),
    ]
}

fn tiles_agree(epi: Epi) {
    for (i, c) in tile_cases().into_iter().enumerate() {
        let seed = 100 + i as u64;
        let base = run_tile(&c, epi, Tile::T128x128, seed);
        for tile in [Tile::T256x128, Tile::T128x256] {
            let got = run_tile(&c, epi, tile, seed);
            assert!(
                got == base,
                "{tile:?} differs from T128x128 at M={} K={} N={}",
                c.m,
                c.k,
                c.n
            );
        }
    }
}

#[test]
fn every_tile_is_bit_equal_to_the_reference_and_the_shipped_tile_f32() {
    tiles_agree(Epi::Set);
}

#[test]
fn every_tile_is_bit_equal_to_the_reference_and_the_shipped_tile_bf16() {
    tiles_agree(Epi::SetBf16);
}

#[test]
fn one_hot_activations_pair_every_code_position_on_every_tile() {
    // A single +1 at qs byte j of every block isolates one weight code: a wrong lane or LDS slot in
    // a tile's staging shows as a wrong code at that position.
    for tile in [Tile::T256x128, Tile::T128x256] {
        for j in [0usize, 7, 8, 63, 64, 127] {
            run_tile(
                &Case {
                    m: 260,
                    k: 256,
                    n: 132,
                    fill: Fill::OneHot(j),
                },
                Epi::Set,
                tile,
                50 + j as u64,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Offline gates: hot-loop shape contract, occupancy ceiling, determinism, assembly of the product module.

fn contract(file: &str) -> serde_json::Value {
    let path = format!("{}/kernels/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn hot_loop_meets_shape_contract_for_every_module_entry() {
    for (tile, file) in [
        (Tile::T128x128, "mq6_i8_gemm.gfx1201.hotloop.contract.json"),
        (
            Tile::T128x256,
            "mq6_i8_gemm.gfx1201.t128x256.hotloop.contract.json",
        ),
    ] {
        hot_loop_meets_contract(tile, &contract(file));
    }
}

fn hot_loop_meets_contract(tile: Tile, c: &serde_json::Value) {
    assert_eq!(
        c["symbol"].as_str().unwrap(),
        Spec::product(Epi::Set).with_tile(tile).symbol()
    );
    assert_eq!(
        c["launch_dynamic_lds_bytes"].as_u64().unwrap(),
        u64::from(tile.layout().launch)
    );
    for epi in [Epi::Set, Epi::SetBf16] {
        let e = mq6_i8_gemm::emit(Spec::product(epi).with_tile(tile)).unwrap();
        let census = mq6_i8_gemm::hot_loop_census(&e.s_text);
        for (name, bound) in c["counts"].as_object().unwrap() {
            let actual = census.get(name).copied().unwrap_or(0);
            assert_eq!(
                u64::from(actual),
                bound.as_u64().unwrap(),
                "{epi:?}: {name}"
            );
        }
        for f in c["forbidden"].as_array().unwrap() {
            let f = f.as_str().unwrap();
            assert!(
                !census
                    .keys()
                    .any(|m| m == f || f.strip_suffix('*').is_some_and(|p| m.starts_with(p))),
                "{epi:?}: forbidden {f}"
            );
        }
        assert!(u64::from(e.shape.next_free_vgpr) <= c["vgpr_max"].as_u64().unwrap());
        assert!(u64::from(e.shape.next_free_sgpr) <= c["sgpr_max"].as_u64().unwrap());
        assert!(e.s_text.contains(&format!(
            ".amdhsa_group_segment_fixed_size {}",
            c["group_segment_fixed_bytes"]
        )));
    }
}

#[test]
fn register_plan_keeps_the_occupancy_ceiling_and_lds_fits_two_workgroups() {
    for epi in [Epi::Set, Epi::SetBf16] {
        let e = mq6_i8_gemm::emit(Spec::product(epi)).unwrap();
        assert!(e.shape.next_free_vgpr <= spec::VGPR_CEILING, "{epi:?}");
        assert_eq!(
            e.shape.wmma, 256,
            "{epi:?}: 64 WMMAs x 4 blocks (2 loop + 2 peeled)"
        );
    }
    assert_eq!(spec::LAUNCH_LDS_BYTES, spec::LAYOUT.launch);
    // Two workgroups per WGP (128 KiB of LDS).
    assert!(2 * spec::LAUNCH_LDS_BYTES <= 128 * 1024);
}

/// The occupancy the tile docs state: 192 VGPRs allow 8 waves per SIMD32 (32 per WGP), and the
/// launch LDS of every tile fits at least two workgroups into the WGP's 128 KiB.
#[test]
fn every_tile_keeps_the_register_ceiling_and_the_documented_occupancy() {
    for tile in Tile::ALL {
        for epi in [Epi::Set, Epi::SetBf16] {
            let e = mq6_i8_gemm::emit(Spec::product(epi).with_tile(tile)).unwrap();
            assert!(
                e.shape.next_free_vgpr <= spec::VGPR_CEILING,
                "{tile:?} {epi:?}: {} VGPRs",
                e.shape.next_free_vgpr
            );
            assert_eq!(e.shape.wmma, 256, "{tile:?} {epi:?}");
            assert!(e
                .s_text
                .contains(&format!(".amdhsa_group_segment_fixed_size {}", 0)));
        }
        let (waves, lds) = (tile.waves(), tile.layout().launch);
        let by_vgpr = 32 / waves;
        let by_lds = 128 * 1024 / lds;
        assert!(by_vgpr >= 2 && by_lds >= 2, "{tile:?}");
        let resident = by_vgpr.min(by_lds) * waves;
        assert_eq!(
            resident,
            match tile {
                Tile::T128x128 => 24,
                _ => 32,
            },
            "{tile:?}"
        );
        // The layout stays inside its launch bytes and every plane is disjoint.
        let l = tile.layout();
        let mut planes = vec![];
        for s in 0..2 {
            planes.push((l.a[s], l.a_bytes));
            planes.push((l.w[s], l.w_bytes));
            planes.push((l.ds[s], l.ds_bytes));
            planes.push((l.sz[s], l.sz_bytes));
        }
        planes.sort();
        for w in planes.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "{tile:?}: planes overlap");
        }
        let last = planes.last().unwrap();
        assert_eq!(last.0 + last.1, l.launch, "{tile:?}");
        assert_eq!(l.a_bytes, tile.tokens() * 64);
        assert_eq!(l.w_bytes, tile.rows() * 64);
        assert_eq!(l.ds_bytes, tile.tokens() * 8);
        assert_eq!(l.sz_bytes, tile.rows() * 8);
    }
}

#[test]
fn tiles_have_their_own_variant_names_and_symbols() {
    let p = Spec::product(Epi::Set);
    assert_eq!(p.variant_name(), "128x128x8-set");
    assert_eq!(p.symbol(), "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201");
    let t = p.with_tile(Tile::T256x128);
    assert_eq!(t.variant_name(), "256x128x16-set");
    assert_eq!(t.symbol(), "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t256x128");
    let t = Spec::emulation(Epi::SetBf16).with_tile(Tile::T128x256);
    assert_eq!(t.variant_name(), "128x256x16-set_bf16-linear");
    assert_eq!(
        t.symbol(),
        "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201_t128x256_linear"
    );
}

#[test]
fn emission_is_deterministic_and_the_module_assembles() {
    let (a, text_a, proof_a) = mq6_i8_gemm::emit_module(Arch::Gfx1201).unwrap();
    let (_, text_b, proof_b) = mq6_i8_gemm::emit_module(Arch::Gfx1201).unwrap();
    assert_eq!(text_a, text_b);
    assert_eq!(proof_a.s_text_sha256, proof_b.s_text_sha256);
    assert_eq!(a.len(), 5);
    let elf = peacemaker_isa::native::assemble(&text_a, Arch::Gfx1201).unwrap();
    let mut symbols: Vec<String> = [Tile::T128x128, Tile::T128x256]
        .into_iter()
        .flat_map(|tile| {
            [Epi::Set, Epi::SetBf16].map(move |epi| Spec::product(epi).with_tile(tile).symbol())
        })
        .collect();
    symbols.push(Spec::product(Epi::SetNr).with_tile(Tile::T256x128).symbol());
    for sym in symbols {
        assert!(
            elf.windows(sym.len()).any(|w| w == sym.as_bytes()),
            "{sym} missing from the code object"
        );
    }
}

#[test]
fn gfx1201_only() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(mq6_i8_gemm::emit(Spec {
            epi: Epi::Set,
            raster: mq6_i8_gemm::Raster::Banded,
            tile: Tile::T128x128,
            arch,
            tune: Default::default(),
        })
        .is_err());
    }
}

// ---------------------------------------------------------------------------------------------
// `Epi::SetNr`: the GEMM output with the Flash-Next QSA head RMSNorm + weight + RoPE applied to the
// tile's head rows. The reference applies `qsa_norm_rope_head`'s arithmetic (serial fma sum of
// squares, `sum / 256`, `rsq`, `(1 + w) * (x * inv)`, `fma(v, cos, +-(v_partner * sin))`) to the
// exact GEMM output with the emulator's model of `v_rsq_f32`.

const NORM_BASE: u64 = 0x3000_0000;
const ROPE_BASE: u64 = 0x3100_0000;

fn nr_reference(raw: &[u8], c: &Case, head_mask: usize, norm: &[u16], table: &[f32]) -> Vec<u8> {
    let m = c.m;
    let mut out = raw.to_vec();
    let at = |t: usize, r: usize| {
        f32::from_bits(u32::from_le_bytes(
            raw[(t * m + r) * 4..(t * m + r) * 4 + 4]
                .try_into()
                .unwrap(),
        ))
    };
    for t in 0..c.n {
        for r0 in (0..m).step_by(256) {
            if (r0 & head_mask) >= 256 {
                continue;
            }
            let x: Vec<f32> = (0..256).map(|i| at(t, r0 + i)).collect();
            let mut sum = 0.0f32;
            for &v in &x {
                sum = v.mul_add(v, sum);
            }
            let mean = sum / 256.0f32;
            let inv = f32::from_bits(peacemaker_ir::emu::trans::rsq_f32(
                peacemaker_ir::inst::Arch::Gfx1201,
                (mean + 1.0e-6f32).to_bits(),
            ));
            let s: Vec<f32> = (0..256)
                .map(|i| (1.0f32 + f32::from_bits(u32::from(norm[i]) << 16)) * (x[i] * inv))
                .collect();
            let mut y = s.clone();
            for ch in 0..64 {
                let b = ch % 32;
                let (cos, sin) = (table[t * 64 + b], table[t * 64 + 32 + b]);
                y[ch] = if ch < 32 {
                    s[ch].mul_add(cos, -(s[ch + 32] * sin))
                } else {
                    s[ch].mul_add(cos, s[ch - 32] * sin)
                };
            }
            for (i, v) in y.iter().enumerate() {
                out[(t * m + r0 + i) * 4..(t * m + r0 + i) * 4 + 4]
                    .copy_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    out
}

fn run_nr(c: &Case, head_mask: usize, seed: u64) {
    let tile = Tile::T256x128;
    let rng = &mut Rng(seed);
    let data = make(rng, c);
    let raw = reference(c, &data, Epi::Set);
    let norm: Vec<u16> = (0..256)
        .map(|_| ((rng.next() as f32 / u64::MAX as f32 - 0.5) * 1.5).to_bits() as u32 >> 16)
        .map(|b| b as u16)
        .collect();
    let table: Vec<f32> = (0..c.n * 64)
        .map(|_| (rng.next() as f32 / u64::MAX as f32) * 2.0 - 1.0)
        .collect();
    let want = nr_reference(&raw, c, head_mask, &norm, &table);
    // The normed rows differ from the raw GEMM output; the plain (gate) rows are the raw output.
    let changed = want
        .chunks(4)
        .zip(raw.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    let normed_rows = (0..c.m)
        .step_by(256)
        .filter(|r0| (r0 & head_mask) < 256)
        .count()
        * 256;
    assert!(
        changed > normed_rows * c.n * 9 / 10,
        "reference leaves {changed} of {} normed values untouched",
        normed_rows * c.n
    );
    let spec = Spec::emulation(Epi::SetNr).with_tile(tile);
    let emitted = mq6_i8_gemm::emit(spec).unwrap();
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emu = Emulator::new(&program, &spec.symbol()).unwrap();

    let mut mem = Memory::default();
    let mut karg = vec![0u8; spec::NR_KERNARG_BYTES as usize];
    karg[0..8].copy_from_slice(&A_BASE.to_le_bytes());
    karg[8..16].copy_from_slice(&X_BASE.to_le_bytes());
    karg[16..24].copy_from_slice(&(Y_BASE + GUARD as u64).to_le_bytes());
    karg[24..28].copy_from_slice(&(c.m as u32).to_le_bytes());
    karg[28..32].copy_from_slice(&(c.k as u32).to_le_bytes());
    karg[32..36].copy_from_slice(&(c.n as u32).to_le_bytes());
    karg[36..40].copy_from_slice(&(head_mask as u32).to_le_bytes());
    let (gx, gy) = (
        c.m.div_ceil(tile.rows() as usize) as u32,
        c.n.div_ceil(tile.tokens() as usize) as u32,
    );
    karg[40..44].copy_from_slice(&gx.to_le_bytes());
    karg[44..48].copy_from_slice(&gy.to_le_bytes());
    karg[48..52].copy_from_slice(&1u32.to_le_bytes());
    karg[spec::NR_NORM_OFFSET as usize..spec::NR_NORM_OFFSET as usize + 8]
        .copy_from_slice(&NORM_BASE.to_le_bytes());
    karg[spec::NR_ROPE_OFFSET as usize..spec::NR_ROPE_OFFSET as usize + 8]
        .copy_from_slice(&ROPE_BASE.to_le_bytes());
    mem.map("karg".into(), KERNARG, karg).unwrap();
    mem.map("a".into(), A_BASE, data.a.clone()).unwrap();
    mem.map("x".into(), X_BASE, data.x.clone()).unwrap();
    mem.map(
        "norm".into(),
        NORM_BASE,
        norm.iter().flat_map(|b| b.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map(
        "rope".into(),
        ROPE_BASE,
        table
            .iter()
            .flat_map(|f| f.to_bits().to_le_bytes())
            .collect(),
    )
    .unwrap();
    mem.map("y".into(), Y_BASE, vec![SENTINEL; want.len() + 2 * GUARD])
        .unwrap();
    for gy_i in 0..gy {
        for gx_i in 0..gx {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx_i, gy_i, 0],
                    block: [tile.threads(), 1, 1],
                    dynamic_lds: tile.layout().launch,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("wg({gx_i},{gy_i}): {e}"));
        }
    }
    let y = &mem.regions.iter().find(|r| r.name == "y").unwrap().bytes;
    assert!(
        y[..GUARD].iter().all(|&b| b == SENTINEL),
        "guard before Y clobbered"
    );
    assert!(
        y[GUARD + want.len()..].iter().all(|&b| b == SENTINEL),
        "guard after Y clobbered"
    );
    let got = &y[GUARD..GUARD + want.len()];
    for (i, (g, e)) in got.chunks(4).zip(want.chunks(4)).enumerate() {
        assert_eq!(
            g,
            e,
            "M={} K={} N={} mask {head_mask}: token {} row {}",
            c.m,
            c.k,
            c.n,
            i / c.m,
            i % c.m
        );
    }
}

#[test]
fn set_nr_q_gate_layout_norms_the_q_half_of_every_head() {
    run_nr(
        &Case {
            m: 768,
            k: 256,
            n: 40,
            fill: Fill::Random,
        },
        511,
        200,
    );
}

#[test]
fn set_nr_k_layout_norms_every_head_with_looped_k_and_token_tails() {
    run_nr(
        &Case {
            m: 512,
            k: 768,
            n: 130,
            fill: Fill::Random,
        },
        255,
        201,
    );
}

#[test]
fn set_nr_validates_only_on_the_wide_row_tile() {
    for tile in [Tile::T128x128, Tile::T128x256] {
        assert!(mq6_i8_gemm::emit(Spec::emulation(Epi::SetNr).with_tile(tile)).is_err());
    }
}

/// The norm+RoPE epilogue changes nothing in the K loop: `set_nr`'s hot-loop census equals the plain
/// `256x128` entry's (bar the plain entry's `s_delay_alu` hints and `s_wait_dscnt` placement), the kernel stays inside the occupancy ceiling, and what it adds after the K
/// loop is the documented epilogue: four staging passes and the RoPE exchange (16 b128 stores
/// each), the inverse-RMS store, one `v_rsq_f32` and 13 barriers (the K loop's rings retire, two per
/// staging pass, two each for the inverse RMS and the exchange).
#[test]
fn set_nr_keeps_the_hot_loop_and_adds_the_documented_epilogue() {
    let plain = mq6_i8_gemm::emit(Spec::product(Epi::Set).with_tile(Tile::T256x128)).unwrap();
    let nr = mq6_i8_gemm::emit(Spec::product(Epi::SetNr).with_tile(Tile::T256x128)).unwrap();
    // The plain entries run the stacked schedule (`Spec::stacked`): `s_delay_alu` issue hints and
    // another `s_wait_dscnt` placement; `set_nr` keeps the plain schedule. Every other counted
    // instruction of the loop is the same.
    let census = |text: &str| {
        let mut c = mq6_i8_gemm::hot_loop_census(text);
        c.remove("s_delay_alu");
        c.remove("s_wait_dscnt");
        c
    };
    assert!(mq6_i8_gemm::hot_loop_census(&plain.s_text).contains_key("s_delay_alu"));
    assert!(!mq6_i8_gemm::hot_loop_census(&nr.s_text).contains_key("s_delay_alu"));
    assert_eq!(census(&plain.s_text), census(&nr.s_text));
    assert!(nr.shape.next_free_vgpr <= spec::VGPR_CEILING);
    assert_eq!(nr.shape.wmma, plain.shape.wmma);
    let count = |text: &str, name: &str| {
        text.lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count()
    };
    let added = |name: &str| count(&nr.s_text, name) - count(&plain.s_text, name);
    assert_eq!(added("v_rsq_f32_e32"), 1);
    assert_eq!(added("ds_store_b32"), 1);
    assert_eq!(added("ds_store_b128"), 5 * 16);
    assert_eq!(added("s_barrier_signal"), 13);
    assert_eq!(added("s_barrier_wait"), 13);
    assert_eq!(added("v_wmma_i32_16x16x16_iu8"), 0);
}

// ---------------------------------------------------------------------------------------------
// Ring depth 3 (`Tune { ring: Three }`): scheduling only, the result bits are depth 2's.

fn ring3(spec: Spec) -> Spec {
    spec.with_tune(Tune {
        ring: RingDepth::Three,
    })
}

/// Depth 3 on the shipped tile equals the CPU reference (hence the depth-2 kernel) bit for bit at
/// every `K` residue class of the three-group period: `n = K/256 - 1` full groups before the peeled
/// one, `r = n % 3` leading groups (K = 256: r 0, q 0; 512: r 1; 768: r 2; 1024: r 0, q 1; 1280:
/// r 1, q 1; 1536: r 2, q 1; 2560: r 0, q 3; 3328: r 0, q 4; 3840: r 2, q 4).
#[test]
fn ring3_equals_depth_2_and_the_reference_at_every_k_residue() {
    for (k, m, n, epi) in [
        (256usize, 132usize, 130usize, Epi::Set),
        (512, 132, 130, Epi::SetBf16),
        (768, 128, 128, Epi::Set),
        (1024, 132, 130, Epi::Set),
        (1280, 128, 128, Epi::SetBf16),
        (1536, 128, 128, Epi::Set),
        (2560, 128, 128, Epi::Set),
        (3328, 64, 64, Epi::Set),
        (3840, 64, 64, Epi::SetBf16),
    ] {
        let c = Case {
            m,
            k,
            n,
            fill: Fill::Random,
        };
        let seed = 700 + (k / 256) as u64;
        let d3 = run_spec(&c, ring3(Spec::emulation(epi)), seed);
        let d2 = run_spec(&c, Spec::emulation(epi), seed);
        assert_eq!(d3, d2, "ring 3 vs ring 2 at K={k} {epi:?}");
    }
}

#[test]
fn ring3_pairs_every_code_position_with_its_activation_byte() {
    for j in [0usize, 17, 64, 127] {
        let c = Case {
            m: 132,
            k: 768,
            n: 40,
            fill: Fill::OneHot(j),
        };
        let d3 = run_spec(&c, ring3(Spec::emulation(Epi::Set)), 90 + j as u64);
        let d2 = run_spec(&c, Spec::emulation(Epi::Set), 90 + j as u64);
        assert_eq!(d3, d2, "one-hot {j}");
    }
}

/// The feasibility numbers per tile (`spec::ring3_budget`): only the shipped tile fits. The
/// 256-wide tiles' third A+W slab alone takes the LDS launch past the 64 KiB workgroup limit
/// (79872 B), and their second prefetch set exceeds the 192-VGPR plan at the 2-workgroup occupancy.
#[test]
fn ring3_budget_numbers_per_tile() {
    let n = spec::ring3_budget(Tile::T128x128);
    assert_eq!(
        (n.pf_vgprs, n.vgprs, n.vgpr_budget, n.lds, n.wg_per_wgp),
        (10, 202, 256, 53248, 2)
    );
    assert!(n.lds_fits() && n.vgprs_fit());
    for (tile, pf, vgprs) in [(Tile::T256x128, 8, 200), (Tile::T128x256, 7, 199)] {
        let n = spec::ring3_budget(tile);
        assert_eq!(
            (n.pf_vgprs, n.vgprs, n.vgpr_budget, n.lds, n.wg_per_wgp),
            (pf, vgprs, 192, 79872, 2),
            "{tile:?}"
        );
        assert!(!n.lds_fits() && !n.vgprs_fit());
    }
    // The depth-3 layout keeps the depth-2 buffers where they are and appends the third slab.
    for tile in Tile::ALL {
        let l = tile.layout();
        assert_eq!(
            l.launch_for(RingDepth::Three),
            l.launch + l.a_bytes + l.w_bytes
        );
        assert_eq!(l.a_slot(2).unwrap(), l.launch);
        assert_eq!(l.w_slot(2).unwrap(), l.launch + l.a_bytes);
    }
}

#[test]
fn ring3_is_refused_where_it_does_not_fit_with_the_numbers() {
    let r3 = |tile, epi| ring3(Spec::product(epi).with_tile(tile)).validate();
    assert!(r3(Tile::T128x128, Epi::Set).is_ok());
    assert!(r3(Tile::T128x128, Epi::SetBf16).is_ok());
    let e = r3(Tile::T128x256, Epi::Set).unwrap_err();
    for want in [
        "128x256x16",
        "79872 B",
        "16384 B A",
        "8192 B W",
        "65536 B",
        "55296",
    ] {
        assert!(e.contains(want), "{want} in {e}");
    }
    let e = r3(Tile::T256x128, Epi::Set).unwrap_err();
    for want in ["256x128x16", "79872 B", "8192 B A", "16384 B W", "65536 B"] {
        assert!(e.contains(want), "{want} in {e}");
    }
    let e = r3(Tile::T256x128, Epi::SetNr).unwrap_err();
    assert!(e.contains("set_nr"), "{e}");
    // `emit` and `Gen::new` refuse too.
    assert!(mq6_i8_gemm::emit(ring3(Spec::product(Epi::Set).with_tile(Tile::T128x256))).is_err());
}

/// Depth 3 changes the spec's variant name by `-r3` and nothing else of its identity; depth 2
/// names, symbols and launch bytes are unchanged.
#[test]
fn ring3_variant_name_and_launch() {
    let d2 = Spec::product(Epi::Set);
    let d3 = ring3(d2);
    assert_eq!(d2.variant_name(), "128x128x8-set");
    assert_eq!(d3.variant_name(), "128x128x8-set-r3");
    assert_eq!(d3.symbol(), d2.symbol());
    // The launch passes the depth-2 dynamic size; the third A+W slab is the object's static LDS, so
    // fixed + dynamic covers the depth-3 layout and the depth-2 fixed size stays 0.
    assert_eq!(d2.launch_lds(), 36864);
    assert_eq!(d3.launch_lds(), 36864);
    assert_eq!((d2.static_lds(), d3.static_lds()), (0, 16384));
    assert_eq!(d3.total_lds(), 53248);
    let k2 = mq6_i8_gemm::Gen::new(d2).unwrap().kernel_spec();
    let k3 = mq6_i8_gemm::Gen::new(d3).unwrap().kernel_spec();
    assert_eq!(k2.group_segment_fixed_size, 0);
    assert_eq!(k3.group_segment_fixed_size + d3.launch_lds(), 53248);
    assert!(
        k3.group_segment_fixed_size + d3.launch_lds()
            >= d3.tile.layout().launch_for(RingDepth::Three)
    );
}

/// The depth-3 kernel keeps its register plan inside the occupancy budget `Spec::vgpr_budget`
/// derives, its WMMA count per block and the depth-2 K-loop's instruction mix per block; the
/// code is three arms + three body groups + the peeled group (14 blocks of 64 WMMAs).
#[test]
fn ring3_register_plan_and_code_shape() {
    let d2 = mq6_i8_gemm::emit(Spec::product(Epi::Set)).unwrap();
    let d3 = mq6_i8_gemm::emit(ring3(Spec::product(Epi::Set))).unwrap();
    assert!(d2.shape.next_free_vgpr <= spec::VGPR_CEILING);
    assert_eq!(d3.shape.next_free_vgpr, 202);
    assert!(d3.shape.next_free_vgpr <= ring3(Spec::product(Epi::Set)).vgpr_budget());
    let wmma = |e: &peacemaker_isa::Emitted| {
        e.s_text
            .lines()
            .filter(|l| l.trim().starts_with("v_wmma_i32_16x16x16_iu8"))
            .count()
    };
    // depth 2: 4 blocks; depth 3: 2 + 4 arm blocks, 6 body blocks, 2 peeled blocks.
    assert_eq!(wmma(&d2), 4 * 64);
    assert_eq!(wmma(&d3), 14 * 64);
}
