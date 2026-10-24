// SPDX-License-Identifier: Apache-2.0
//! G80 direct-BF16 programs in the whole-array simulator, fed with panels from OUR pack pipeline (`pack_shape`).
//!
//! * `sim_*_all_families_k128`: for each shape, EVERY panel (GdnIn 13, QsaIn 11 — each family boundary falls inside a panel
//!   or on one) is submitted as one `M = 256` (one wave), `K = 128` (two K chunks: the smallest K that still crosses a chunk
//!   boundary) G80 `Epilogue::Bf16` / `Control::Fast` program on ONE reused `Config`. The B argument is the pack_shape K = 2560
//!   panel cut down to its first two K chunks per column segment (chunk-major layout, see `truncate_panel`), A is the
//!   canonical `pack_a_bf16` of random a8 with per-row `(Dm, De)`. All 256 x 1280 BF16 outputs of every panel are compared
//!   with the `bf16_direct` CPU twin of an independent integer dot product and independently quantised grid15-up weight
//!   scales, scattered into each family's own output through `ColumnMap::locate`. Padded fused columns must come back +0.
//! * `sim_masked_programs_k2560`: the REAL program of `build_programs_for_families` (`K = 2560`, one wave) with the small
//!   masks GdnA+GdnB (GdnIn) and QsaIdx (QsaIn): the pack_shape panel is submitted as is, full-K CPU dot + twin.
//! * `pack_shape_panels_equal_canonical_b_*`: pack_shape panels (K = 2560, ALL and skip-a-family masks, family
//!   compaction across panel boundaries) equal `pack_b_bf16` of a B matrix built here from an independent MQ6 decode and
//!   grid15-up quantiser with the fused order spelled out independently; no simulator.
//!
//! The reference quantiser (`quantize_direct`) is a second, bit-stream based implementation of the packer's numerics.
//! Functional ticks are simulator ticks, never silicon cycles. Budget: a fixed number of submits of one wave,
//! `tick_limit` as in the peacemaker-npu harness (`Config` default, per wait, so a stall is an `Err`, not a hang).
use hipfire_arch_qwen4::npu_trunk::{
    build_out_programs, build_programs_for_families, pack_out, pack_shape, Family, FamilyMask,
    MatrixRef, PackedB, Shape, OUT_COLS, OUT_K, OUT_PANEL_COLS, PANEL_COLS, TRUNK_K,
};
use hipfire_runtime::coproc::EpilogueContract;
use peacemaker_npu::kernels::gemm_array::ArrayDesign;
use peacemaker_npu::kernels::gemm_core::{Control, Epilogue};
use peacemaker_npu::kernels::gemm_g80::{design_g80, pack_a_bf16, pack_b_bf16};
use peacemaker_npu::kernels::w8a8_epi::{bf16_direct, grid15_up};
use peacemaker_npu::sim::config::Config;

const WAVE_M: usize = 256;
/// K of the all-families simulator runs: two 64-wide chunks.
const K_SIM: usize = 128;
const K_CHUNK: usize = 64;
/// Bytes of one K chunk of one core column's B segment (two 80-wide N tiles x 64 K, each + Sm[96] + Se[96]).
const B_CHUNK: usize = 2 * (5120 + 384);
const B_COLS: usize = 8;
const POISON: u8 = 0xa5;
/// Same value as `Config::new()` (the peacemaker-npu harness never changes it); written out so the budget is explicit.
const TICK_LIMIT: usize = 10_000_000;
const MQ6_GROUP_BYTES: usize = 200;

// ---------------------------------------------------------------- deterministic data
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random MQ6G256V2 matrix `[m][2560]`: per 256-group `[f16 s0, f16 z0, f16 s1, f16 z1, 192 B random payload]`. Rows 3 and
/// m-1 have all-zero scales and zero points (absmax 0: ws == 0 and all codes 0); row 5 has huge-amplitude scales next to tiny
/// ones so its codes saturate at +-127.
fn random_mq6(m: usize, seed: u64) -> Vec<u8> {
    random_mq6_k(m, TRUNK_K, seed)
}

/// [`random_mq6`] for rows of width `k` (a multiple of 256).
fn random_mq6_k(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let mut rng = Rng::new(seed);
    let mut out = vec![0u8; m * groups * MQ6_GROUP_BYTES];
    for row in 0..m {
        for g in 0..groups {
            let at = (row * groups + g) * MQ6_GROUP_BYTES;
            let blank = row == 3 || row + 1 == m;
            for h in 0..2 {
                let (s, z) = if blank {
                    (0.0, 0.0)
                } else if row == 5 && h == 1 {
                    (0.001, 0.0)
                } else {
                    let r = rng.next();
                    let s = ((r & 0xff) as f32 + 1.0) / 2048.0 * if row == 5 { 64.0 } else { 1.0 };
                    let z = (((r >> 8) & 0xff) as f32 - 128.0) / 256.0;
                    (s, z)
                };
                out[at + 4 * h..at + 4 * h + 2]
                    .copy_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
                out[at + 4 * h + 2..at + 4 * h + 4]
                    .copy_from_slice(&half::f16::from_f32(z).to_bits().to_le_bytes());
            }
            for b in out[at + 8..at + MQ6_GROUP_BYTES].chunks_mut(8) {
                let r = rng.next().to_le_bytes();
                b.copy_from_slice(&r[..b.len()]);
            }
        }
    }
    out
}

struct Fam {
    family: Family,
    mq6: Vec<u8>,
    /// `[family.cols()][keep_k]` row-major (first `keep_k` of the 2560 W8 codes of each output row).
    w8: Vec<i8>,
    /// Grid15-up weight scale as f32 (what `PackedB::ws` holds).
    ws: Vec<f32>,
    /// `(Sm, Se)` per output row.
    grid: Vec<(u16, i16)>,
    keep_k: usize,
}
impl Fam {
    fn row(&self, c: usize) -> &[i8] {
        &self.w8[c * self.keep_k..(c + 1) * self.keep_k]
    }
}

/// Independent direct-BF16 weight quantiser of one MQ6G256V2 row: bit-stream 6-bit decode, `w = q*s + z` (separate multiply
/// and add), `raw = ((absmax as f64)/127) as f32`, grid15-up scale, `clamp(round_ties_even(w / ws), -127, 127)`.
fn quantize_direct(row: &[u8]) -> (Vec<i8>, f32, (u16, i16)) {
    let mut w = Vec::with_capacity(TRUNK_K);
    for group in row.chunks_exact(MQ6_GROUP_BYTES) {
        let h = |i: usize| {
            half::f16::from_bits(u16::from_le_bytes([group[2 * i], group[2 * i + 1]])).to_f32()
        };
        let payload = &group[8..];
        for j in 0..256 {
            let mut q = 0u32;
            for b in 0..6 {
                let bit = j * 6 + b;
                q |= u32::from((payload[bit / 8] >> (bit % 8)) & 1) << b;
            }
            let half = j / 128;
            w.push(q as f32 * h(2 * half) + h(2 * half + 1));
        }
    }
    let absmax = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let grid = grid15_up(((f64::from(absmax)) / 127.0) as f32);
    let ws = (f64::from(grid.0) * 2f64.powi(i32::from(grid.1))) as f32;
    let codes = w
        .iter()
        .map(|&x| {
            if ws == 0.0 {
                0
            } else {
                ((x / ws).round_ties_even() as i32).clamp(-127, 127) as i8
            }
        })
        .collect();
    (codes, ws, grid)
}

/// The selected families of `shape` (fused order), their MQ6 blobs and the reference w8 (first `keep_k` codes per row).
fn fixture(shape: Shape, which: FamilyMask, keep_k: usize) -> Vec<Fam> {
    shape
        .families()
        .iter()
        .filter(|&&f| which.contains(f))
        .map(|&family| {
            let m = family.cols();
            let mq6 = random_mq6(m, 0x5eed_0000 + family.index() as u64);
            let (mut w8, mut ws, mut grid) = (
                Vec::with_capacity(m * keep_k),
                Vec::with_capacity(m),
                Vec::with_capacity(m),
            );
            for row in mq6.chunks_exact(TRUNK_K / 256 * MQ6_GROUP_BYTES) {
                let (codes, s, g) = quantize_direct(row);
                w8.extend_from_slice(&codes[..keep_k]);
                ws.push(s);
                grid.push(g);
            }
            assert_eq!((w8.len(), ws.len()), (m * keep_k, m));
            Fam {
                family,
                mq6,
                w8,
                ws,
                grid,
                keep_k,
            }
        })
        .collect()
}

/// Per-token activation scales `(Dm, De)` (a few zero scales, the rest on the full exponent range).
fn token_scales(count: usize, seed: usize) -> Vec<(u16, i16)> {
    (0..count)
        .map(|i| {
            if (i + seed) % 17 == 0 {
                (0, 0)
            } else {
                (
                    (16384 + (i * 7919 + seed * 1237) % 16384) as u16,
                    [-163, -110, -90, -70, -35, -18, 0, 45, 110][(i + seed) % 9],
                )
            }
        })
        .collect()
}

/// The CPU twin of one output element.
fn want_bf16(i: i32, weight: (u16, i16), token: (u16, i16)) -> i32 {
    i32::from(bf16_direct(i, weight.0, weight.1, token.0, token.1))
}

fn refs(fams: &[Fam]) -> Vec<MatrixRef<'_>> {
    fams.iter()
        .map(|f| MatrixRef {
            family: f.family,
            mq6: &f.mq6,
            m: f.family.cols(),
            k: TRUNK_K,
        })
        .collect()
}

/// a8 `[WAVE_M][k]` in -127..=127 with edge rows: all +127, all -127, all 0, alternating +-127.
fn random_a8(k: usize, seed: u64) -> Vec<i8> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<i8> = (0..WAVE_M * k)
        .map(|_| (((rng.next() >> 33) % 255) as i32 - 127) as i8)
        .collect();
    for kk in 0..k {
        a[kk] = 127;
        a[k + kk] = -127;
        a[2 * k + kk] = 0;
        a[3 * k + kk] = if kk % 2 == 0 { 127 } else { -127 };
    }
    a
}

fn dot(a: &[i8], w: &[i8]) -> i32 {
    a.iter().zip(w).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Fused layout of the selected families, spelled out here: `(family, first fused col, cols)` in fused order, compacted.
fn expected_parts(shape: Shape, mask: FamilyMask) -> Vec<(Family, usize, usize)> {
    let mut at = 0;
    let mut parts = Vec::new();
    for &f in shape.families() {
        if mask.contains(f) {
            parts.push((f, at, f.cols()));
            at += f.cols();
        }
    }
    parts
}

fn assert_map(shape: Shape, mask: FamilyMask) -> hipfire_arch_qwen4::npu_trunk::ColumnMap {
    let map = shape.column_map(mask);
    let got: Vec<_> = map
        .parts
        .iter()
        .map(|&(f, c0, n)| (f, c0 as usize, n as usize))
        .collect();
    assert_eq!(got, expected_parts(shape, mask), "{shape:?} column map");
    let useful: usize = got.iter().map(|p| p.2).sum();
    assert_eq!(
        (map.useful_cols(), map.panel_count(), map.padded_cols()),
        (
            useful,
            useful.div_ceil(PANEL_COLS),
            useful.div_ceil(PANEL_COLS) * PANEL_COLS
        )
    );
    map
}

/// B `[k][1280]` of one fused panel: column `j` is the w8 row of the family column at fused col `panel*1280+j` (0 in padding).
fn fused_b(shape: Shape, mask: FamilyMask, fams: &[Fam], panel: usize, k: usize) -> Vec<i8> {
    let mut b = vec![0i8; k * PANEL_COLS];
    for (f, col0, cols) in expected_parts(shape, mask) {
        let fam = fams.iter().find(|x| x.family == f).expect("fixture family");
        for c in 0..cols {
            let fused = col0 + c;
            if fused / PANEL_COLS != panel {
                continue;
            }
            let j = fused % PANEL_COLS;
            for (kk, &w) in fam.row(c)[..k].iter().enumerate() {
                b[kk * PANEL_COLS + j] = w;
            }
        }
    }
    b
}

/// `(Sm, Se)` of the 1280 columns of one fused panel (`(0, 0)` in padding).
fn fused_scales(shape: Shape, mask: FamilyMask, fams: &[Fam], panel: usize) -> Vec<(u16, i16)> {
    let mut s = vec![(0u16, 0i16); PANEL_COLS];
    for (f, col0, cols) in expected_parts(shape, mask) {
        let fam = fams.iter().find(|x| x.family == f).expect("fixture family");
        for c in 0..cols {
            let fused = col0 + c;
            if fused / PANEL_COLS == panel {
                s[fused % PANEL_COLS] = fam.grid[c];
            }
        }
    }
    s
}

fn pack(shape: Shape, fams: &[Fam], mask: FamilyMask) -> PackedB {
    pack_shape(shape, &refs(fams), mask).unwrap_or_else(|e| panic!("{shape:?} pack_shape: {e}"))
}

/// First `kc` K chunks of every column segment of a K = 2560 panel: the G80 B stream is `[col 0..8][chunk][even, odd tile]`
/// with `B_CHUNK` bytes per (column, chunk), so a K-prefix is a per-segment prefix. Equality with the canonical `pack_in` of the
/// K-prefix matrix is asserted by every caller.
fn truncate_panel(panel: &[u8], kc: usize) -> Vec<u8> {
    let kc_full = TRUNK_K / K_CHUNK;
    assert_eq!(panel.len(), B_COLS * kc_full * B_CHUNK);
    let mut out = Vec::with_capacity(B_COLS * kc * B_CHUNK);
    for col in 0..B_COLS {
        out.extend_from_slice(&panel[col * kc_full * B_CHUNK..(col * kc_full + kc) * B_CHUNK]);
    }
    out
}

fn g80(rows: usize, k: usize) -> ArrayDesign {
    design_g80(rows, PANEL_COLS, k, Epilogue::Bf16, Control::Fast)
}

// ---------------------------------------------------------------- one checked submit
/// One submit on a (possibly reused) `Config`: returns the row-major `WAVE_M x 1280` BF16 bits (as i32). The packed inputs
/// must be unchanged by the run.
fn submit(sim: &mut Config, d: &ArrayDesign, a_host: &[u8], b_host: &[u8], what: &str) -> Vec<i32> {
    assert_eq!(
        (a_host.len(), b_host.len()),
        (d.args[0].bytes, d.args[1].bytes),
        "{what}: packed argument sizes"
    );
    let mut args = vec![
        a_host.to_vec(),
        b_host.to_vec(),
        vec![POISON; d.args[2].bytes],
    ];
    let ticks0 = sim.ticks;
    sim.submit(&d.insts, &mut args)
        .unwrap_or_else(|e| panic!("{what}: submit failed: {e}"));
    assert!(
        args[0] == a_host && args[1] == b_host,
        "{what}: packed inputs changed"
    );
    eprintln!(
        "PASS {what}: {} functional ticks (simulator ticks, not hardware cycles)",
        sim.ticks - ticks0
    );
    d.unpack_out(&args[2])
}

fn fresh(d: &ArrayDesign) -> Config {
    let mut sim = Config::from_pdi(&d.pdi).expect("PDI loads");
    sim.tick_limit = TICK_LIMIT;
    sim
}

// ---------------------------------------------------------------- all families, every panel, K = 128
fn sim_all_families(shape: Shape) {
    let mask = FamilyMask::ALL;
    let fams = fixture(shape, mask, K_SIM);
    let packed = pack(shape, &fams, mask);
    let map = assert_map(shape, mask);
    assert_eq!(packed.panels.len(), map.panel_count());
    assert_eq!(
        packed.map.parts, map.parts,
        "PackedB carries the column map"
    );

    let d = g80(WAVE_M, K_SIM);
    let a8 = random_a8(K_SIM, 0xa8a8 + shape.index() as u64);
    let ds = token_scales(WAVE_M, shape.index());
    let a_host = pack_a_bf16(&d, &a8, &ds);
    let mut sim = fresh(&d);

    // Per-family output tensors filled ONLY through ColumnMap::locate.
    let mut out: Vec<Vec<i32>> = fams
        .iter()
        .map(|f| vec![i32::MIN; WAVE_M * f.family.cols()])
        .collect();
    let mut written = vec![0usize; fams.len()];
    for p in 0..packed.panels.len() {
        let b_host = truncate_panel(&packed.panels[p], K_SIM / K_CHUNK);
        let canonical = pack_b_bf16(
            &d,
            &fused_b(shape, mask, &fams, p, K_SIM),
            &fused_scales(shape, mask, &fams, p),
        );
        assert!(b_host == canonical, "{shape:?} panel {p}: truncated pack_shape panel != canonical pack_b_bf16 of the K-prefix B");
        let c = submit(
            &mut sim,
            &d,
            &a_host,
            &b_host,
            &format!("{shape:?} panel {p}/{} K{K_SIM}", packed.panels.len()),
        );
        assert_eq!(c.len(), WAVE_M * PANEL_COLS);
        for j in 0..PANEL_COLS {
            match map.locate(p * PANEL_COLS + j) {
                Some((family, col)) => {
                    let fi = fams
                        .iter()
                        .position(|f| f.family == family)
                        .expect("family in fixture");
                    for row in 0..WAVE_M {
                        out[fi][row * family.cols() + col] = c[row * PANEL_COLS + j];
                    }
                    written[fi] += 1;
                }
                None => {
                    assert!(
                        (0..WAVE_M)
                            .all(|row| c[row * PANEL_COLS + j] == want_bf16(0, (0, 0), ds[row])),
                        "{shape:?} panel {p}: padded column {j} must be +0"
                    );
                }
            }
        }
    }
    for (fi, fam) in fams.iter().enumerate() {
        assert_eq!(
            written[fi],
            fam.family.cols(),
            "{:?}: every column scattered exactly once",
            fam.family
        );
        for c in 0..fam.family.cols() {
            let w = fam.row(c);
            for row in 0..WAVE_M {
                let want = want_bf16(
                    dot(&a8[row * K_SIM..(row + 1) * K_SIM], w),
                    fam.grid[c],
                    ds[row],
                );
                let got = out[fi][row * fam.family.cols() + c];
                assert_eq!(
                    got, want,
                    "{:?} row {row} col {c}: sim {got:#06x} != twin {want:#06x}",
                    fam.family
                );
            }
        }
        eprintln!(
            "PASS {:?}: {} x {} BF16 outputs equal the bf16_direct twin through ColumnMap",
            fam.family,
            WAVE_M,
            fam.family.cols()
        );
    }
}

#[test]
fn sim_gdn_in_all_families_k128() {
    sim_all_families(Shape::GdnIn);
}

#[test]
fn sim_qsa_in_all_families_k128() {
    sim_all_families(Shape::QsaIn);
}

// ---------------------------------------------------------------- the real K = 2560 program, small masks
#[test]
fn sim_masked_programs_k2560() {
    use Family::*;
    let mask = FamilyMask::from(GdnA) | FamilyMask::from(GdnB) | FamilyMask::from(QsaIdx);
    let progs =
        build_programs_for_families(EpilogueContract::Bf16Direct, WAVE_M, mask).expect("programs");
    assert_eq!(progs.rows_max, WAVE_M);
    for shape in Shape::ALL {
        let prog = progs.get(shape);
        assert_eq!(prog.panel_count, 1, "{shape:?}");
        let fams = fixture(shape, mask, TRUNK_K);
        let packed = pack(shape, &fams, mask);
        let map = assert_map(shape, mask);
        assert_eq!(packed.panels.len(), 1);
        assert_eq!(
            packed.panels[0].len(),
            prog.arg_bytes[1],
            "{shape:?}: panel bytes == program B argument"
        );
        assert_eq!(packed.ws.len(), prog.padded_cols);

        let a8 = random_a8(TRUNK_K, 0x2560 + shape.index() as u64);
        let ds = token_scales(WAVE_M, 3 + shape.index());
        let a_host = pack_a_bf16(&prog.design, &a8, &ds);
        let b_ref = pack_b_bf16(
            &prog.design,
            &fused_b(shape, mask, &fams, 0, TRUNK_K),
            &fused_scales(shape, mask, &fams, 0),
        );
        assert_eq!(a_host.len(), prog.arg_bytes[0]);
        assert!(
            packed.panels[0] == b_ref,
            "{shape:?}: pack_shape panel != canonical pack_b_bf16 of the independent fused B"
        );

        let mut sim = fresh(&prog.design);
        let c = submit(
            &mut sim,
            &prog.design,
            &a_host,
            &packed.panels[0],
            &format!("{shape:?} program K{TRUNK_K}"),
        );
        assert_eq!(c.len(), WAVE_M * PANEL_COLS);
        let useful = map.useful_cols();
        for j in 0..PANEL_COLS {
            match map.locate(j) {
                Some((family, col)) => {
                    let fam = fams.iter().find(|f| f.family == family).unwrap();
                    let w = fam.row(col);
                    for row in 0..WAVE_M {
                        let want = want_bf16(
                            dot(&a8[row * TRUNK_K..(row + 1) * TRUNK_K], w),
                            fam.grid[col],
                            ds[row],
                        );
                        assert_eq!(
                            c[row * PANEL_COLS + j],
                            want,
                            "{shape:?} {family:?} row {row} col {col}"
                        );
                    }
                }
                None => {
                    assert!(j >= useful);
                    assert!(
                        (0..WAVE_M)
                            .all(|row| c[row * PANEL_COLS + j] == want_bf16(0, (0, 0), ds[row])),
                        "{shape:?}: padded column {j} must be +0"
                    );
                }
            }
        }
        eprintln!("PASS {shape:?}: {} families, {useful} useful columns x {WAVE_M} rows equal the K2560 bf16_direct twin", map.parts.len());
    }
}

// ---------------------------------------------------------------- canonical B crosscheck (no simulator)
fn crosscheck(shape: Shape, mask: FamilyMask) {
    let fams = fixture(shape, mask, TRUNK_K);
    let packed = pack(shape, &fams, mask);
    let map = assert_map(shape, mask);
    assert_eq!(packed.panels.len(), map.panel_count(), "{shape:?} {mask:?}");
    assert_eq!(packed.ws.len(), map.padded_cols());
    let d = g80(1, TRUNK_K);
    for p in 0..packed.panels.len() {
        let canonical = pack_b_bf16(
            &d,
            &fused_b(shape, mask, &fams, p, TRUNK_K),
            &fused_scales(shape, mask, &fams, p),
        );
        assert_eq!(packed.panels[p].len(), canonical.len());
        assert!(
            packed.panels[p] == canonical,
            "{shape:?} {mask:?} panel {p}: pack_shape != canonical G80 BF16 B"
        );
    }
    for (family, col0, cols) in expected_parts(shape, mask) {
        let fam = fams.iter().find(|f| f.family == family).unwrap();
        for c in 0..cols {
            assert_eq!(
                packed.ws[col0 + c].to_bits(),
                fam.ws[c].to_bits(),
                "{family:?} ws {c}"
            );
        }
    }
    let useful = map.useful_cols();
    assert!(
        packed.ws[useful..].iter().all(|&w| w == 0.0),
        "padded ws is 0"
    );
    eprintln!(
        "PASS {shape:?} {mask:?}: {} panels equal the canonical pack_in B",
        packed.panels.len()
    );
}

fn mask_of(families: &[Family]) -> FamilyMask {
    families
        .iter()
        .fold(FamilyMask::NONE, |m, &f| m | FamilyMask::from(f))
}

#[test]
fn pack_shape_panels_equal_canonical_b_gdn_in() {
    use Family::*;
    crosscheck(Shape::GdnIn, FamilyMask::GDN);
    crosscheck(Shape::GdnIn, mask_of(&[GdnA, GdnB]));
    // Compaction: Z dropped, a/b land right after qkv and straddle the 8th/9th panel boundary region.
    crosscheck(Shape::GdnIn, mask_of(&[GdnQkv, GdnA, GdnB]));
    crosscheck(Shape::GdnIn, mask_of(&[GdnZ, GdnA]));
}

#[test]
fn pack_shape_panels_equal_canonical_b_qsa_in() {
    use Family::*;
    crosscheck(Shape::QsaIn, FamilyMask::QSA);
    crosscheck(Shape::QsaIn, mask_of(&[QsaIdx]));
    crosscheck(Shape::QsaIn, mask_of(&[QsaK, QsaV, QsaIdx]));
    crosscheck(Shape::QsaIn, mask_of(&[QsaQ, QsaIdx]));
}

// ---------------------------------------------------------------- the out-projection program (N512, K = 6144)
/// `pack_out` panels 0 and 4 of a random `[2560][6144]` MQ6 matrix run through the REAL `build_out_programs` design (512 rows, one
/// wave, K = 6144, row-major pitch 2560 at col0 = 0): every output equals `bf16_direct` of the independent whole-K dot product
/// and the independently quantised grid15-up scales; the all-zero rows (3 and 2559) come back +0, the panel equals
/// `pack_b_bf16` of the independent codes.
#[test]
fn sim_out_panels_k6144_equal_the_whole_k_twin() {
    let mq6 = random_mq6_k(OUT_COLS, OUT_K, 0x6144_0001);
    let packed = pack_out(Shape::GdnOut, &mq6, OUT_COLS, OUT_K).expect("pack_out");
    let progs = build_out_programs(512).expect("out programs");
    assert_eq!(progs.rows_max, 512);
    let prog = progs.get(Shape::QsaO);
    assert_eq!((prog.panel_count, prog.padded_cols), (5, OUT_COLS));
    assert_eq!(packed.panels[0].len(), prog.arg_bytes[1]);
    let rb = OUT_K / 256 * MQ6_GROUP_BYTES;
    let a8 = random_a8_rows(512, OUT_K, 0x6144_a8);
    let ds = token_scales(512, 5);
    let a_host = pack_a_bf16(&prog.design, &a8, &ds);
    assert_eq!(a_host.len(), prog.arg_bytes[0]);
    let mut sim = fresh(&prog.design);
    for panel in [0usize, 4] {
        let rows = panel * OUT_PANEL_COLS..(panel + 1) * OUT_PANEL_COLS;
        let (mut b, mut scales, mut codes) =
            (vec![0i8; OUT_K * OUT_PANEL_COLS], Vec::new(), Vec::new());
        for (c, row) in rows.clone().enumerate() {
            let (q, _, grid) = quantize_direct(&mq6[row * rb..(row + 1) * rb]);
            for (k, &v) in q.iter().enumerate() {
                b[k * OUT_PANEL_COLS + c] = v;
            }
            scales.push(grid);
            codes.push(q);
        }
        assert_eq!(
            &packed.scales[rows.clone()],
            &scales[..],
            "panel {panel}: scales"
        );
        assert!(
            packed.panels[panel] == pack_b_bf16(&prog.design, &b, &scales),
            "panel {panel}: pack_out != pack_b_bf16 of the independent codes"
        );
        let c = submit(
            &mut sim,
            &prog.design,
            &a_host,
            &packed.panels[panel],
            &format!("out panel {panel} K{OUT_K}"),
        );
        assert_eq!(c.len(), 512 * OUT_PANEL_COLS);
        for row in 0..512 {
            for col in 0..OUT_PANEL_COLS {
                let dot: i32 = a8[row * OUT_K..(row + 1) * OUT_K]
                    .iter()
                    .zip(&codes[col])
                    .map(|(&x, &w)| i32::from(x) * i32::from(w))
                    .sum();
                assert_eq!(
                    c[row * OUT_PANEL_COLS + col],
                    want_bf16(dot, scales[col], ds[row]),
                    "panel {panel} row {row} col {col}"
                );
            }
        }
    }
    assert!(
        packed.scales[3] == (0, 0) && packed.scales[OUT_COLS - 1] == (0, 0),
        "blank rows have zero scale"
    );
}

/// `m` activation rows `[m][k]` in -127..=127 (edge rows as in `random_a8`).
fn random_a8_rows(m: usize, k: usize, seed: u64) -> Vec<i8> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<i8> = (0..m * k)
        .map(|_| (((rng.next() >> 33) % 255) as i32 - 127) as i8)
        .collect();
    for kk in 0..k {
        a[kk] = 127;
        a[k + kk] = -127;
        a[2 * k + kk] = 0;
        a[3 * k + kk] = if kk % 2 == 0 { 127 } else { -127 };
    }
    a
}
