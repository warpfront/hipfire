// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU weight preparation: MQ6G256V2 trunk matrices → W8 codes + per-row scales → the
//! G80 `B` stream the NPU reads (INTERFACES.md §0 and §1).
//!
//! # Numerics (bit-exact with the lab `npu_a8emu_wprep_w8`)
//!
//! Per output row `n` of a `[N][K]` matrix:
//!
//! * `w[k] = (q as f32) * f32(s_h) + f32(z_h)`: a separate multiply then add (Rust never
//!   fuses), `h` the half of the 256-group;
//! * `absmax = max |w|` (NaN-ignoring `max`, like `fmaxf`), `ws = ((absmax as f64) / 127.0) as f32`;
//! * `w8 = clamp(round_ties_even(w / ws), -127, 127)` with an `f32` division; `ws == 0` gives
//!   all-zero codes.
//!
//! # Layout
//!
//! peacemaker-npu's canonical G80 `pack_b` takes a row-major `B[k][n]` of one 1280-column panel.
//! Each selected W8 row (one fused column) is decoded once and transposed straight into one
//! reused `K × 1280` panel scratch (zeroed per panel, so ragged final panels are zero
//! padded); the full fused W8 matrix is never materialised. Panel preparation
//! calls `ArrayDesign::pack_b` directly, without a discarded activation allocation.

use rayon::prelude::*;

use peacemaker_npu::kernels::gemm_array::ArrayDesign;
use peacemaker_npu::kernels::gemm_core::{Control, Epilogue};
use peacemaker_npu::kernels::gemm_g80::design_g80;
use peacemaker_npu::kernels::gemm_g80_n512::design_g80_n512;
use peacemaker_npu::kernels::{gemm_g80::pack_b_bf16, w8a8_epi::w8_direct_row};

use super::shapes::{
    ColumnMap, Family, FamilyMask, Shape, OUT_COLS, OUT_K, OUT_PANEL_COLS, PANEL_COLS, TRUNK_K,
};
use hipfire_runtime::coproc::EpilogueContract;

/// K covered by one MQ6G256V2 group.
pub const GROUP_K: usize = 256;
/// Bytes of one MQ6G256V2 group: four f16 header words and 192 payload bytes.
pub const GROUP_BYTES: usize = 200;

/// One MQ6G256V2 projection matrix `[m][k]` of a layer.
#[derive(Clone, Copy, Debug)]
pub struct MatrixRef<'a> {
    pub family: Family,
    pub mq6: &'a [u8],
    pub m: usize,
    pub k: usize,
}

/// The projections of one layer that share an input.
#[derive(Clone, Debug)]
pub struct LayerTrunkRefs<'a> {
    pub shape: Shape,
    pub matrices: Vec<MatrixRef<'a>>,
}

/// One layer's fused `B`: the G80-packed panels, the padded per-column scales and the column map.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedB {
    /// `map.panel_count()` panels, each in peacemaker-npu G80 `pack_b` layout for `K = 2560`, `N = 1280`.
    pub panels: Vec<Vec<u8>>,
    /// Per fused column row scale `ws_n`, zero padded to `map.padded_cols()`.
    pub ws: Vec<f32>,
    /// Bf16Direct `(Sm, Se)` per padded fused feature.
    pub scales: Vec<(u16, i16)>,
    pub map: ColumnMap,
}

/// Every layer's packed `B`.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTrunk {
    /// Per input layer: `None` when the layer's shape has no selected family (zero panels).
    pub per_layer: Vec<Option<(Shape, PackedB)>>,
    /// Per input layer: a copy of that layer's padded `PackedB::ws` (empty for `None` layers).
    pub ws: Vec<Vec<f32>>,
    /// Fused column map per shape, indexed by [`Shape::index`].
    pub map: [ColumnMap; 2],
}

/// Exact f16 → f32 widening.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: value = man * 2^-24; renormalise.
            let p = 31 - man.leading_zeros();
            sign | ((103 + p) << 23) | (((man << (10 - p)) & 0x3FF) << 13)
        }
        0x1F => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Bytes of one `k`-wide MQ6G256V2 row.
const fn row_bytes(k: usize) -> usize {
    k / GROUP_K * GROUP_BYTES
}

/// `w[k] = (q as f32) * f32(s_h) + f32(z_h)` for one row.
fn decode_row(row: &[u8], out: &mut [f32]) {
    debug_assert_eq!(row.len() / GROUP_BYTES * GROUP_K, out.len());
    for (group, dst) in row
        .chunks_exact(GROUP_BYTES)
        .zip(out.chunks_exact_mut(GROUP_K))
    {
        let hdr = |i: usize| f16_to_f32(u16::from_le_bytes([group[2 * i], group[2 * i + 1]]));
        let halves = [(hdr(0), hdr(1)), (hdr(2), hdr(3))];
        for (t, b) in group[8..].chunks_exact(3).enumerate() {
            let (b0, b1, b2) = (b[0] as u32, b[1] as u32, b[2] as u32);
            let codes = [
                b0 & 63,
                (b0 >> 6) | ((b1 & 0xF) << 2),
                (b1 >> 4) | ((b2 & 3) << 4),
                b2 >> 2,
            ];
            for (c, &q) in codes.iter().enumerate() {
                let j = t * 4 + c;
                let (s, z) = halves[j >> 7];
                // Separate multiply and add, never fused.
                dst[j] = (q as f32) * s + z;
            }
        }
    }
}

/// Quantize one row to W8 codes; `w` is a `k`-float scratch. Returns `ws`.
fn quantize_row(row: &[u8], w: &mut [f32], out: &mut [i8]) -> f32 {
    quantize_scaled_row(row, w, out, false).0
}

fn quantize_scaled_row(
    row: &[u8],
    w: &mut [f32],
    out: &mut [i8],
    direct: bool,
) -> (f32, (u16, i16)) {
    decode_row(row, w);
    if direct {
        return w8_direct_row(w, out);
    }
    let absmax = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let ws = ((absmax as f64) / 127.0) as f32;
    if ws == 0.0 {
        out.fill(0);
    } else {
        for (o, &x) in out.iter_mut().zip(w.iter()) {
            // Saturating, NaN-to-zero float→int like the device conversion, then clamp.
            *o = ((x / ws).round_ties_even() as i32).clamp(-127, 127) as i8;
        }
    }
    (ws, (0, 0))
}

/// W8 codes `[m][k]` row-major and per-row scales of an MQ6G256V2 matrix.
///
/// # Panics
/// If `k` is zero or not a multiple of 256, or `mq6.len() != m * k / 256 * 200`. `m == 0` is
/// permitted (empty results). The shape and trunk packers validate before calling this.
pub fn pack_matrix(mq6: &[u8], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(
        k > 0 && k % GROUP_K == 0,
        "k must be a positive multiple of {GROUP_K}, got {k}"
    );
    let rb = row_bytes(k);
    assert_eq!(
        Some(mq6.len()),
        m.checked_mul(rb),
        "mq6 must be m*k/{GROUP_K}*{GROUP_BYTES} = {m}*{rb} bytes"
    );
    let mut w8 = vec![0i8; m * k];
    let mut ws = Vec::with_capacity(m);
    let mut w = vec![0.0f32; k];
    for (row, out) in mq6.chunks_exact(rb).zip(w8.chunks_exact_mut(k)) {
        ws.push(quantize_row(row, &mut w, out));
    }
    (w8, ws)
}

/// Row-major direct-BF16 GPU twin of the resident panel quantizer.
pub(crate) fn pack_matrix_direct(mq6: &[u8], m: usize, k: usize, output: &mut [u8]) {
    assert_eq!(k, TRUNK_K);
    assert_eq!(mq6.len(), m * row_bytes(k));
    assert_eq!(output.len(), m * k);
    // i8/u8 have identical size/alignment and every bit pattern is valid.
    let codes =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<i8>(), output.len()) };
    let mut row = [0.0; TRUNK_K];
    for (raw, dst) in mq6
        .chunks_exact(row_bytes(k))
        .zip(codes.chunks_exact_mut(k))
    {
        quantize_scaled_row(raw, &mut row, dst, true);
    }
}

/// Validate `matrices` against `shape`/`families` and return the selected ones in fused order.
///
/// Every present entry must belong to the shape and appear strictly in fused order (so no
/// duplicates). Unselected entries may be absent and are not otherwise inspected; each
/// selected family must appear exactly once with its full row count, `K = 2560` and the exact
/// MQ6G256V2 byte length.
fn select<'m, 'a>(
    shape: Shape,
    matrices: &'m [MatrixRef<'a>],
    families: FamilyMask,
) -> Result<Vec<&'m MatrixRef<'a>>, String> {
    let order = shape.families();
    let mut prev: Option<usize> = None;
    for mat in matrices {
        let Some(pos) = order.iter().position(|&f| f == mat.family) else {
            return Err(format!("{} does not belong to shape {shape}", mat.family));
        };
        match prev {
            Some(p) if p == pos => {
                return Err(format!("duplicate {} in shape {shape}", mat.family))
            }
            Some(p) if p > pos => {
                return Err(format!(
                    "{} is out of order in shape {shape} (expected fused order {})",
                    mat.family,
                    order.iter().map(|f| f.name()).collect::<Vec<_>>().join("|")
                ))
            }
            _ => {}
        }
        prev = Some(pos);
    }
    let mut selected = Vec::new();
    for &family in order.iter().filter(|&&f| families.contains(f)) {
        let Some(mat) = matrices.iter().find(|m| m.family == family) else {
            return Err(format!("missing selected {family} in shape {shape}"));
        };
        if mat.m != family.cols() {
            return Err(format!(
                "{family}: expected {} rows, got {}",
                family.cols(),
                mat.m
            ));
        }
        if mat.k != TRUNK_K {
            return Err(format!("{family}: expected K {TRUNK_K}, got {}", mat.k));
        }
        let want = mat.m * row_bytes(mat.k);
        if mat.mq6.len() != want {
            return Err(format!(
                "{family}: expected {want} MQ6G256V2 bytes for {}x{}, got {}",
                mat.m,
                mat.k,
                mat.mq6.len()
            ));
        }
        selected.push(mat);
    }
    Ok(selected)
}

/// The design whose `B` packer is the canonical one: one panel (`N = 1280`), `K = 2560`, one A row, direct BF16.
fn packing_design_with_contract(contract: EpilogueContract) -> ArrayDesign {
    match contract {
        EpilogueContract::Bf16Direct => {
            design_g80(1, PANEL_COLS, TRUNK_K, Epilogue::Bf16, Control::Fast)
        }
    }
}

/// Pack validated selected matrices (`selected[i]` is `map.parts[i]`).
fn pack_validated(design: &ArrayDesign, map: ColumnMap, selected: &[&MatrixRef<'_>]) -> PackedB {
    let panels = map.panel_count();
    let mut ws = vec![0.0f32; map.padded_cols()];
    let mut scales = vec![(0, 0); map.padded_cols()];
    let mut b = vec![0i8; TRUNK_K * PANEL_COLS];
    let mut w = vec![0.0f32; TRUNK_K];
    let mut codes = vec![0i8; TRUNK_K];
    let rb = row_bytes(TRUNK_K);
    let mut out = Vec::with_capacity(panels);
    for p in 0..panels {
        b.fill(0);
        let (lo, hi) = (p * PANEL_COLS, (p + 1) * PANEL_COLS);
        for (&(_, col0, cols), mat) in map.parts.iter().zip(selected) {
            let c0 = col0 as usize;
            for fused in c0.max(lo)..(c0 + cols as usize).min(hi) {
                let (r, c) = (fused - c0, fused - lo);
                let (scale, grid) =
                    quantize_scaled_row(&mat.mq6[r * rb..(r + 1) * rb], &mut w, &mut codes, true);
                ws[fused] = scale;
                scales[fused] = grid;
                for (k, &v) in codes.iter().enumerate() {
                    b[k * PANEL_COLS + c] = v;
                }
            }
        }
        out.push(pack_b_bf16(design, &b, &scales[lo..hi]));
    }
    PackedB {
        panels: out,
        ws,
        scales,
        map,
    }
}

/// Pack one layer's fused `B` for the families of `families` that belong to `shape`
/// (contract [`EpilogueContract::Bf16Direct`]).
///
/// `matrices` are the layer's projections strictly in fused family order; unselected ones may
/// be absent. Zero selected columns gives zero panels.
pub fn pack_shape(
    shape: Shape,
    matrices: &[MatrixRef<'_>],
    families: FamilyMask,
) -> Result<PackedB, String> {
    pack_shape_with_contract(EpilogueContract::Bf16Direct, shape, matrices, families)
}

/// [`pack_shape`] for an explicit numerical contract.
pub fn pack_shape_with_contract(
    contract: EpilogueContract,
    shape: Shape,
    matrices: &[MatrixRef<'_>],
    families: FamilyMask,
) -> Result<PackedB, String> {
    let selected = select(shape, matrices, families)?;
    let map = shape.column_map(families);
    if map.panel_count() == 0 {
        return Ok(PackedB {
            panels: Vec::new(),
            ws: Vec::new(),
            scales: Vec::new(),
            map,
        });
    }
    Ok(pack_validated(
        &packing_design_with_contract(contract),
        map,
        &selected,
    ))
}

/// Pack every layer (contract [`EpilogueContract::Bf16Direct`]).
pub fn pack_trunk(
    layers: &[LayerTrunkRefs<'_>],
    families: FamilyMask,
) -> Result<PackedTrunk, String> {
    pack_trunk_with_contract(EpilogueContract::Bf16Direct, layers, families)
}

/// [`pack_trunk`] for an explicit numerical contract.
///
/// All layers are validated first (errors name the layer index); packing then runs on a
/// rayon `par_iter`, one layer per job, with results placed by layer index, so the output
/// is independent of thread count and scheduling. A layer whose shape has no selected family
/// is `None` with an empty `ws`.
pub fn pack_trunk_with_contract(
    contract: EpilogueContract,
    layers: &[LayerTrunkRefs<'_>],
    families: FamilyMask,
) -> Result<PackedTrunk, String> {
    let maps = Shape::ALL.map(|s| s.column_map(families));
    let mut jobs = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        let selected = select(layer.shape, &layer.matrices, families)
            .map_err(|e| format!("layer {i}: {e}"))?;
        if maps[layer.shape.index()].panel_count() > 0 {
            jobs.push((i, layer.shape, selected));
        }
    }

    let mut packed: Vec<Option<PackedB>> = (0..layers.len()).map(|_| None).collect();
    if !jobs.is_empty() {
        let design = packing_design_with_contract(contract);
        let done: Vec<(usize, PackedB)> = jobs
            .par_iter()
            .map(|(i, shape, selected)| {
                (
                    *i,
                    pack_validated(&design, maps[shape.index()].clone(), selected),
                )
            })
            .collect();
        for (i, b) in done {
            packed[i] = Some(b);
        }
    }

    let mut per_layer = Vec::with_capacity(layers.len());
    let mut ws = Vec::with_capacity(layers.len());
    for (layer, b) in layers.iter().zip(packed) {
        ws.push(b.as_ref().map(|b| b.ws.clone()).unwrap_or_default());
        per_layer.push(b.map(|b| (layer.shape, b)));
    }
    Ok(PackedTrunk {
        per_layer,
        ws,
        map: maps,
    })
}

/// One out-projection matrix packed for the N512 K6144 program (`Shape::GdnOut` / `Shape::QsaO`): the five resident
/// 512-column BF16 B panels (`pack_b_bf16` of the N512 design, each `8 * 96 * 4352` B), the grid15-up weight scale as f32 per
/// output feature and its `(Sm, Se)` decomposition (what the panels carry as sidecars). 2560 features fill the five panels
/// exactly: there is no padding.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedOut {
    pub shape: Shape,
    pub panels: Vec<Vec<u8>>,
    pub ws: Vec<f32>,
    pub scales: Vec<(u16, i16)>,
}

/// The design whose `B` packer is the canonical one of the out shapes: one panel (N512), K6144, one A row, direct BF16.
fn out_packing_design() -> ArrayDesign {
    design_g80_n512(1, OUT_PANEL_COLS, OUT_K, Epilogue::Bf16, Control::Fast)
}

fn validate_out(shape: Shape, mq6: &[u8], m: usize, k: usize) -> Result<(), String> {
    if !shape.is_out() {
        return Err(format!("{shape} is not an out-projection shape"));
    }
    if m != OUT_COLS || k != OUT_K {
        return Err(format!("{shape}: expected {OUT_COLS}x{OUT_K}, got {m}x{k}"));
    }
    let want = m * row_bytes(k);
    if mq6.len() != want {
        return Err(format!(
            "{shape}: expected {want} MQ6G256V2 bytes for {m}x{k}, got {}",
            mq6.len()
        ));
    }
    Ok(())
}

/// Quantize and pack one validated out matrix: per output row the same W8 numerics as the input shapes
/// ([`quantize_scaled_row`], direct grid15-up scale), transposed into reused `K x 512` panel scratch, then `pack_b_bf16`.
/// Panels are independent and packed in parallel; the result does not depend on thread count.
fn pack_out_validated(design: &ArrayDesign, shape: Shape, mq6: &[u8]) -> PackedOut {
    let rb = row_bytes(OUT_K);
    let panels = OUT_COLS / OUT_PANEL_COLS;
    let done: Vec<(Vec<u8>, Vec<f32>, Vec<(u16, i16)>)> = (0..panels)
        .into_par_iter()
        .map(|p| {
            let mut b = vec![0i8; OUT_K * OUT_PANEL_COLS];
            let mut w = vec![0.0f32; OUT_K];
            let mut codes = vec![0i8; OUT_K];
            let mut ws = Vec::with_capacity(OUT_PANEL_COLS);
            let mut scales = Vec::with_capacity(OUT_PANEL_COLS);
            for c in 0..OUT_PANEL_COLS {
                let row = p * OUT_PANEL_COLS + c;
                let (scale, grid) =
                    quantize_scaled_row(&mq6[row * rb..(row + 1) * rb], &mut w, &mut codes, true);
                ws.push(scale);
                scales.push(grid);
                for (k, &v) in codes.iter().enumerate() {
                    b[k * OUT_PANEL_COLS + c] = v;
                }
            }
            (pack_b_bf16(design, &b, &scales), ws, scales)
        })
        .collect();
    let mut out = PackedOut {
        shape,
        panels: Vec::with_capacity(panels),
        ws: Vec::with_capacity(OUT_COLS),
        scales: Vec::with_capacity(OUT_COLS),
    };
    for (panel, ws, scales) in done {
        out.panels.push(panel);
        out.ws.extend(ws);
        out.scales.extend(scales);
    }
    out
}

/// Pack one out-projection matrix: `mq6` is the stock `[2560][6144]` MQ6G256V2 matrix (`m = 2560`, `k = 6144`). Rows are the
/// output features; the per-row scale and the W8 codes are those of the input shapes (INTERFACES.md §0, grid15-up).
pub fn pack_out(shape: Shape, mq6: &[u8], m: usize, k: usize) -> Result<PackedOut, String> {
    validate_out(shape, mq6, m, k)?;
    Ok(pack_out_validated(&out_packing_design(), shape, mq6))
}

/// Pack every `(shape, matrix)` layer (errors name the layer index); the canonical design is built once.
pub fn pack_out_layers(layers: &[(Shape, &[u8])]) -> Result<Vec<PackedOut>, String> {
    for (i, &(shape, mq6)) in layers.iter().enumerate() {
        validate_out(shape, mq6, OUT_COLS, OUT_K).map_err(|e| format!("layer {i}: {e}"))?;
    }
    if layers.is_empty() {
        return Ok(Vec::new());
    }
    let design = out_packing_design();
    Ok(layers
        .par_iter()
        .map(|&(shape, mq6)| pack_out_validated(&design, shape, mq6))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- synthetic MQ6G256V2 data -------------------------------------------------------

    /// Finite f16 header words the independent reference below can widen exactly.
    const HDR: [u16; 6] = [0x3C00, 0x3800, 0xBC00, 0x2C00, 0x4000, 0x0000];

    fn h2f(bits: u16) -> f32 {
        match bits {
            0x3C00 => 1.0,
            0x3800 => 0.5,
            0xBC00 => -1.0,
            0x2C00 => 0.0625,
            0x4000 => 2.0,
            0xC000 => -2.0,
            0x5400 => 64.0,
            0x0000 => 0.0,
            _ => panic!("unexpected header {bits:#06x}"),
        }
    }

    /// `m` random rows of K = 2560 with random payload and headers from [`HDR`].
    fn synth(m: usize, seed: u64) -> Vec<u8> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut v = vec![0u8; m * row_bytes(TRUNK_K)];
        for g in v.chunks_exact_mut(GROUP_BYTES) {
            for i in 0..4 {
                let h = HDR[(next() % HDR.len() as u64) as usize];
                g[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
            }
            for b in &mut g[8..] {
                *b = next() as u8;
            }
        }
        v
    }

    /// Encode one group: headers `[s0, z0, s1, z1]` and 256 six-bit codes.
    fn encode_group(hdr: [u16; 4], codes: &[u8; 256]) -> [u8; GROUP_BYTES] {
        let mut g = [0u8; GROUP_BYTES];
        for (i, h) in hdr.iter().enumerate() {
            g[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
        }
        for t in 0..64 {
            let v: u32 = (0..4)
                .map(|c| (codes[4 * t + c] as u32 & 63) << (6 * c))
                .sum();
            g[8 + 3 * t..11 + 3 * t].copy_from_slice(&v.to_le_bytes()[..3]);
        }
        g
    }

    /// Independent reference for one row (bit-stream unpack, manual ties-to-even).
    fn ref_row(row: &[u8]) -> (Vec<i8>, f32) {
        let mut w = Vec::new();
        for g in row.chunks_exact(GROUP_BYTES) {
            let hd = |i: usize| u16::from_le_bytes([g[2 * i], g[2 * i + 1]]);
            for j in 0..GROUP_K {
                let (t, c, h) = (j / 4, j % 4, j / 128);
                let v =
                    g[8 + 3 * t] as u32 | (g[9 + 3 * t] as u32) << 8 | (g[10 + 3 * t] as u32) << 16;
                let q = ((v >> (6 * c)) & 63) as f32;
                let prod = q * h2f(hd(2 * h));
                w.push(prod + h2f(hd(2 * h + 1)));
            }
        }
        let am = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let ws = (am as f64 / 127.0) as f32;
        let codes = w
            .iter()
            .map(|&x| {
                if ws == 0.0 {
                    return 0i8;
                }
                let r = x / ws;
                let fl = r.floor();
                let d = r - fl;
                let n = if d < 0.5 {
                    fl
                } else if d > 0.5 {
                    fl + 1.0
                } else if fl % 2.0 == 0.0 {
                    fl
                } else {
                    fl + 1.0
                };
                n.clamp(-127.0, 127.0) as i8
            })
            .collect();
        (codes, ws)
    }

    fn mref(family: Family, mq6: &[u8]) -> MatrixRef<'_> {
        MatrixRef {
            family,
            mq6,
            m: family.cols(),
            k: TRUNK_K,
        }
    }

    // ---- scalar numerics ----------------------------------------------------------------

    #[test]
    fn f16_widening_is_exact() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative() && f16_to_f32(0x8000) == 0.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2.0f32.powi(-15));
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn pack_matrix_matches_independent_reference() {
        let m = 7;
        let mq6 = synth(m, 11);
        let (w8, ws) = pack_matrix(&mq6, m, TRUNK_K);
        assert_eq!((w8.len(), ws.len()), (m * TRUNK_K, m));
        for n in 0..m {
            let (codes, want_ws) =
                ref_row(&mq6[n * row_bytes(TRUNK_K)..(n + 1) * row_bytes(TRUNK_K)]);
            assert_eq!(ws[n].to_bits(), want_ws.to_bits(), "ws row {n}");
            assert_eq!(
                &w8[n * TRUNK_K..(n + 1) * TRUNK_K],
                &codes[..],
                "codes row {n}"
            );
        }
    }

    #[test]
    fn pack_matrix_unpacks_all_codes_exactly() {
        // Half 0: w = q (s=1, z=0). Half 1: w = 64 + q, q = 63 once, so absmax = 127 and
        // ws = 1.0 exactly: the W8 codes then equal the raw 6-bit codes (+64 in half 1).
        let mut st = 0x1234_5678u32;
        let mut codes = [0u8; 256];
        for c in codes.iter_mut() {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *c = (st >> 24) as u8 & 63;
        }
        codes[255] = 63;
        let g = encode_group([0x3C00, 0x0000, 0x3C00, 0x5400], &codes);
        let (w8, ws) = pack_matrix(&g, 1, GROUP_K);
        assert_eq!(ws, vec![1.0]);
        for j in 0..256 {
            let want = codes[j] as i8 + if j >= 128 { 64 } else { 0 };
            assert_eq!(w8[j], want, "code {j}");
        }
    }

    #[test]
    fn pack_matrix_rounds_ties_to_even_in_both_signs() {
        let mut c0 = [0u8; 256];
        let mut c1 = [0u8; 256];
        for q in 0..9 {
            c0[q] = q as u8;
            c1[q] = q as u8;
        }
        // Pin absmax to 127: half 1, s = 1, z = 64, q = 63.
        c0[255] = 63;
        c1[255] = 63;
        // Row 0: w = q/2.            Row 1: w = q/2 - 2.
        let r0 = encode_group([0x3800, 0x0000, 0x3C00, 0x5400], &c0);
        let r1 = encode_group([0x3800, 0xC000, 0x3C00, 0x5400], &c1);
        let mq6: Vec<u8> = r0.iter().chain(r1.iter()).copied().collect();
        let (w8, ws) = pack_matrix(&mq6, 2, GROUP_K);
        assert_eq!(ws, vec![1.0, 1.0]);
        assert_eq!(&w8[..9], &[0, 0, 1, 2, 2, 2, 3, 4, 4]);
        assert_eq!(&w8[256..265], &[-2, -2, -1, 0, 0, 0, 1, 2, 2]);
        assert_eq!(w8[255], 127);
        assert_eq!(w8[511], 127);
    }

    #[test]
    fn pack_matrix_scale_is_f64_division_then_f32() {
        // Half 0: w = q - 1, q = 63 -> 62; half 1: w = 0.5 q (= 0). absmax = 62.
        let mut codes = [0u8; 256];
        codes[63] = 63;
        let g = encode_group([0x3C00, 0xBC00, 0x3800, 0x0000], &codes);
        let (w8, ws) = pack_matrix(&g, 1, GROUP_K);
        let want = ((62.0f32 as f64) / 127.0) as f32;
        assert_eq!(ws[0].to_bits(), want.to_bits());
        assert_eq!(w8[63], 127);
        assert_eq!(w8[0], ((-1.0f32 / want).round_ties_even()) as i8);
    }

    #[test]
    fn pack_matrix_zero_row_has_zero_scale_and_codes() {
        let g = encode_group([0, 0, 0, 0], &[7u8; 256]);
        let (w8, ws) = pack_matrix(&g, 1, GROUP_K);
        assert_eq!(ws, vec![0.0]);
        assert!(w8.iter().all(|&c| c == 0));
    }

    #[test]
    fn pack_matrix_empty_and_invalid_arguments() {
        let (w8, ws) = pack_matrix(&[], 0, TRUNK_K);
        assert!(w8.is_empty() && ws.is_empty());
        assert!(std::panic::catch_unwind(|| pack_matrix(&[0; 200], 1, 128)).is_err());
        assert!(std::panic::catch_unwind(|| pack_matrix(&[], 0, 0)).is_err());
        assert!(std::panic::catch_unwind(|| pack_matrix(&[0; 199], 1, 256)).is_err());
        assert!(std::panic::catch_unwind(|| pack_matrix(&[0; 200], 2, 256)).is_err());
    }

    // ---- shape packing ------------------------------------------------------------------

    /// Expected panels/ws for matrices in fused order: per-row direct quantizer + canonical BF16 packer
    /// (the grid quantizer itself is checked independently by `bf16_direct_weights_match_...`).
    fn expected(mats: &[(Family, &[u8])]) -> (Vec<Vec<u8>>, Vec<f32>) {
        let rb = row_bytes(TRUNK_K);
        let mut w = vec![0.0f32; TRUNK_K];
        let mut codes = vec![0i8; TRUNK_K];
        let mut rows: Vec<(Vec<i8>, f32, (u16, i16))> = Vec::new();
        for &(f, mq6) in mats {
            for r in 0..f.cols() {
                let (ws, grid) =
                    quantize_scaled_row(&mq6[r * rb..(r + 1) * rb], &mut w, &mut codes, true);
                rows.push((codes.clone(), ws, grid));
            }
        }
        let panels = rows.len().div_ceil(PANEL_COLS);
        let mut ws = vec![0.0f32; panels * PANEL_COLS];
        let design = packing_design_with_contract(EpilogueContract::Bf16Direct);
        let mut out = Vec::new();
        for p in 0..panels {
            let mut b = vec![0i8; TRUNK_K * PANEL_COLS];
            let mut scales = vec![(0u16, 0i16); PANEL_COLS];
            for c in 0..PANEL_COLS {
                let fused = p * PANEL_COLS + c;
                if let Some((q, s, grid)) = rows.get(fused) {
                    ws[fused] = *s;
                    scales[c] = *grid;
                    for k in 0..TRUNK_K {
                        b[k * PANEL_COLS + c] = q[k];
                    }
                }
            }
            out.push(pack_b_bf16(&design, &b, &scales));
        }
        (out, ws)
    }

    #[test]
    fn pack_shape_single_panel_with_padding() {
        let (a, b) = (synth(48, 1), synth(48, 2));
        let mats = [mref(Family::GdnA, &a), mref(Family::GdnB, &b)];
        let mask = Family::GdnA | Family::GdnB;
        let packed = pack_shape(Shape::GdnIn, &mats, mask).unwrap();
        assert_eq!(packed.map, Shape::GdnIn.column_map(mask));
        assert_eq!(packed.panels.len(), 1);
        assert_eq!(
            packed.panels[0].len(),
            packing_design_with_contract(EpilogueContract::Bf16Direct).args[1].bytes
        );
        assert_eq!(packed.ws.len(), PANEL_COLS);
        assert!(packed.ws[96..].iter().all(|&s| s == 0.0));
        assert!(packed.ws[..96].iter().any(|&s| s != 0.0));
        let (panels, ws) = expected(&[(Family::GdnA, &a), (Family::GdnB, &b)]);
        assert_eq!(packed.panels, panels);
        assert_eq!(
            packed.ws.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            ws.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pack_shape_crosses_panel_boundaries_and_skips_unselected() {
        // k|v|idx = 512 + 512 + 640 = 1664 columns: panel 0 ends inside idx, panel 1 is
        // 384 live columns + zero padding. q is unselected and may be a placeholder.
        let (k, v, idx) = (synth(512, 3), synth(512, 4), synth(640, 5));
        let placeholder = MatrixRef {
            family: Family::QsaQ,
            mq6: &[],
            m: 0,
            k: 0,
        };
        let mats = [
            placeholder,
            mref(Family::QsaK, &k),
            mref(Family::QsaV, &v),
            mref(Family::QsaIdx, &idx),
        ];
        let mask = Family::QsaK | Family::QsaV | Family::QsaIdx;
        let packed = pack_shape(Shape::QsaIn, &mats, mask).unwrap();
        assert_eq!(packed.panels.len(), 2);
        assert_eq!(packed.ws.len(), 2 * PANEL_COLS);
        assert!(packed.ws[1664..].iter().all(|&s| s == 0.0));
        let (panels, ws) = expected(&[
            (Family::QsaK, &k),
            (Family::QsaV, &v),
            (Family::QsaIdx, &idx),
        ]);
        assert_eq!(packed.panels, panels);
        assert_eq!(
            packed.ws.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            ws.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
        );
        // The same selection without the placeholder is identical.
        let again = pack_shape(Shape::QsaIn, &mats[1..], mask).unwrap();
        assert_eq!(again, packed);
    }

    #[test]
    fn pack_shape_empty_selection_has_zero_panels() {
        let a = synth(48, 6);
        // Other-shape bits are ignored; present-but-unselected matrices are not packed.
        let packed = pack_shape(Shape::GdnIn, &[mref(Family::GdnA, &a)], FamilyMask::QSA).unwrap();
        assert!(packed.panels.is_empty() && packed.ws.is_empty());
        assert_eq!(packed.map, ColumnMap::default());
        let packed = pack_shape(Shape::QsaIn, &[], FamilyMask::NONE).unwrap();
        assert!(packed.panels.is_empty());
    }

    #[test]
    fn pack_shape_rejects_missing_misordered_and_invalid_inputs() {
        let (a, b) = (synth(48, 7), synth(48, 8));
        let mask = Family::GdnA | Family::GdnB;
        let ok_a = mref(Family::GdnA, &a);
        let ok_b = mref(Family::GdnB, &b);
        let err = |mats: &[MatrixRef<'_>]| pack_shape(Shape::GdnIn, mats, mask).unwrap_err();

        let e = err(&[ok_a]);
        assert!(e.contains("missing") && e.contains("gdn_b"), "{e}");
        let e = err(&[]);
        assert!(e.contains("missing") && e.contains("gdn_a"), "{e}");
        let e = err(&[ok_b, ok_a]);
        assert!(e.contains("out of order"), "{e}");
        let e = err(&[ok_a, ok_a, ok_b]);
        assert!(e.contains("duplicate"), "{e}");
        let e = err(&[ok_a, mref(Family::QsaK, &[]), ok_b]);
        assert!(e.contains("does not belong"), "{e}");
        let e = err(&[MatrixRef { m: 47, ..ok_a }, ok_b]);
        assert!(e.contains("rows"), "{e}");
        let e = err(&[MatrixRef { k: 2304, ..ok_a }, ok_b]);
        assert!(e.contains("K 2560"), "{e}");
        let e = err(&[
            MatrixRef {
                mq6: &a[..a.len() - 1],
                ..ok_a
            },
            ok_b,
        ]);
        assert!(e.contains("bytes"), "{e}");
        // Order is enforced even when the offender is unselected.
        let e = pack_shape(Shape::GdnIn, &[ok_b, ok_a], Family::GdnA.into()).unwrap_err();
        assert!(e.contains("out of order"), "{e}");
    }

    #[test]
    fn bf16_direct_validates_inputs_and_empty_masks() {
        let bad = [mref(Family::GdnB, &[])];
        let err = pack_shape_with_contract(
            EpilogueContract::Bf16Direct,
            Shape::GdnIn,
            &bad,
            FamilyMask::ALL,
        )
        .unwrap_err();
        assert!(err.contains("missing") || err.contains("expected"), "{err}");
        let empty =
            pack_trunk_with_contract(EpilogueContract::Bf16Direct, &[], FamilyMask::NONE).unwrap();
        assert!(empty.per_layer.is_empty());
        let layers = [LayerTrunkRefs {
            shape: Shape::GdnIn,
            matrices: bad.to_vec(),
        }];
        let err = pack_trunk_with_contract(EpilogueContract::Bf16Direct, &layers, FamilyMask::ALL)
            .unwrap_err();
        assert!(err.starts_with("layer 0:"), "{err}");
    }

    #[test]
    fn bf16_direct_weights_match_independent_upward_grid_quantization() {
        let rows = Family::GdnA.cols();
        let mut mq6 = synth(rows, 0x846271);
        mq6[..row_bytes(TRUNK_K)].fill(0);
        let mat = mref(Family::GdnA, &mq6);
        let packed = pack_shape_with_contract(
            EpilogueContract::Bf16Direct,
            Shape::GdnIn,
            &[mat],
            Family::GdnA.into(),
        )
        .unwrap();
        let mut expected_scales = vec![(0u16, 0i16); PANEL_COLS];
        let mut b = vec![0i8; TRUNK_K * PANEL_COLS];
        for row in 0..rows {
            let mut w = Vec::with_capacity(TRUNK_K);
            for g in mq6[row * row_bytes(TRUNK_K)..(row + 1) * row_bytes(TRUNK_K)]
                .chunks_exact(GROUP_BYTES)
            {
                let hd = |i| h2f(u16::from_le_bytes([g[2 * i], g[2 * i + 1]]));
                for j in 0..256 {
                    let bit = j * 6;
                    let mut q = 0u8;
                    for k in 0..6 {
                        q |= ((g[8 + (bit + k) / 8] >> ((bit + k) % 8)) & 1) << k;
                    }
                    let product = f32::from(q) * hd(2 * (j / 128));
                    w.push(product + hd(2 * (j / 128) + 1));
                }
            }
            let absmax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let raw = (f64::from(absmax) / 127.0) as f32;
            if raw == 0.0 {
                continue;
            }
            let mut e = f64::from(raw).log2().floor() as i16 - 14;
            let mut sm = (f64::from(raw) * 2f64.powi(-i32::from(e))).ceil() as u16;
            if sm == 32768 {
                sm = 16384;
                e += 1;
            }
            expected_scales[row] = (sm, e);
            let s = (f64::from(sm) * 2f64.powi(i32::from(e))) as f32;
            for (k, &value) in w.iter().enumerate() {
                let v = value / s;
                let floor = v.floor();
                let fraction = v - floor;
                let rounded = floor
                    + u8::from(fraction > 0.5 || (fraction == 0.5 && floor % 2.0 != 0.0)) as f32;
                b[k * PANEL_COLS + row] = rounded.clamp(-127.0, 127.0) as i8;
            }
        }
        assert_eq!(packed.scales, expected_scales);
        let design = packing_design_with_contract(EpilogueContract::Bf16Direct);
        assert_eq!(packed.panels[0], pack_b_bf16(&design, &b, &expected_scales));
        assert_eq!(packed.ws[0], 0.0);
    }

    // ---- trunk packing ------------------------------------------------------------------

    #[test]
    fn pack_trunk_matches_per_shape_and_duplicates_ws() {
        let data: Vec<Vec<u8>> = (0..5).map(|i| synth(48, 20 + i)).collect();
        let k = synth(512, 30);
        let mask = Family::GdnA | Family::GdnB | Family::QsaK;
        let layers = [
            LayerTrunkRefs {
                shape: Shape::GdnIn,
                matrices: vec![mref(Family::GdnA, &data[0]), mref(Family::GdnB, &data[1])],
            },
            LayerTrunkRefs {
                shape: Shape::QsaIn,
                matrices: vec![mref(Family::QsaK, &k)],
            },
            LayerTrunkRefs {
                shape: Shape::GdnIn,
                matrices: vec![mref(Family::GdnA, &data[2]), mref(Family::GdnB, &data[3])],
            },
        ];
        let trunk = pack_trunk(&layers, mask).unwrap();
        assert_eq!(trunk.per_layer.len(), 3);
        assert_eq!(trunk.map[0], Shape::GdnIn.column_map(mask));
        assert_eq!(trunk.map[1], Shape::QsaIn.column_map(mask));
        for (i, layer) in layers.iter().enumerate() {
            let want = pack_shape(layer.shape, &layer.matrices, mask).unwrap();
            let (shape, got) = trunk.per_layer[i].as_ref().unwrap();
            assert_eq!(*shape, layer.shape);
            assert_eq!(got, &want, "layer {i}");
            assert_eq!(trunk.ws[i], want.ws, "ws layer {i}");
        }
        // Different weights give different packed bytes (guards against layer mix-ups).
        assert_ne!(
            trunk.per_layer[0].as_ref().unwrap().1.panels,
            trunk.per_layer[2].as_ref().unwrap().1.panels
        );
    }

    #[test]
    fn pack_trunk_layers_without_selected_family_are_none() {
        let (a, b) = (synth(48, 40), synth(48, 41));
        let layers = [
            LayerTrunkRefs {
                shape: Shape::GdnIn,
                matrices: vec![mref(Family::GdnA, &a), mref(Family::GdnB, &b)],
            },
            LayerTrunkRefs {
                shape: Shape::QsaIn,
                matrices: vec![],
            },
        ];
        let mask = Family::GdnA | Family::GdnB;
        let trunk = pack_trunk(&layers, mask).unwrap();
        assert!(trunk.per_layer[0].is_some());
        assert!(trunk.per_layer[1].is_none());
        assert_eq!(trunk.ws[0].len(), PANEL_COLS);
        assert!(trunk.ws[1].is_empty());
        assert!(trunk.map[1].parts.is_empty());

        let none = pack_trunk(&layers, FamilyMask::NONE).unwrap();
        assert!(none.per_layer.iter().all(Option::is_none));
        assert!(none.ws.iter().all(Vec::is_empty));
        let empty = pack_trunk(&[], FamilyMask::ALL).unwrap();
        assert!(empty.per_layer.is_empty() && empty.ws.is_empty());
        assert_eq!(empty.map[0], Shape::GdnIn.column_map(FamilyMask::ALL));
        assert_eq!(empty.map[1], Shape::QsaIn.column_map(FamilyMask::ALL));
    }

    #[test]
    fn pack_trunk_errors_name_the_layer() {
        let (a, b) = (synth(48, 50), synth(48, 51));
        let layers = [
            LayerTrunkRefs {
                shape: Shape::GdnIn,
                matrices: vec![mref(Family::GdnA, &a), mref(Family::GdnB, &b)],
            },
            LayerTrunkRefs {
                shape: Shape::GdnIn,
                matrices: vec![mref(Family::GdnA, &a)],
            },
        ];
        let e = pack_trunk(&layers, Family::GdnA | Family::GdnB).unwrap_err();
        assert!(
            e.starts_with("layer 1:") && e.contains("missing") && e.contains("gdn_b"),
            "{e}"
        );
        let swapped = [LayerTrunkRefs {
            shape: Shape::GdnIn,
            matrices: vec![mref(Family::GdnB, &b), mref(Family::GdnA, &a)],
        }];
        let e = pack_trunk(&swapped, Family::GdnA | Family::GdnB).unwrap_err();
        assert!(
            e.starts_with("layer 0:") && e.contains("out of order"),
            "{e}"
        );
    }

    // ---- out-projection (N512 K6144) packing ---------------------------------------------

    /// `m` random rows of width `k` with random payload and headers from [`HDR`].
    fn synth_k(m: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut v = vec![0u8; m * row_bytes(k)];
        for g in v.chunks_exact_mut(GROUP_BYTES) {
            for i in 0..4 {
                let h = HDR[(next() % HDR.len() as u64) as usize];
                g[2 * i..2 * i + 2].copy_from_slice(&h.to_le_bytes());
            }
            for b in &mut g[8..] {
                *b = next() as u8;
            }
        }
        v
    }

    /// Independent direct-BF16 weight quantiser of one row (bit-stream decode, separate multiply and add, grid15-up scale,
    /// manual ties-to-even): the codes and `(Sm, Se)` the packer must produce.
    fn ref_direct_row(row: &[u8]) -> (Vec<i8>, (u16, i16)) {
        let mut w = Vec::new();
        for g in row.chunks_exact(GROUP_BYTES) {
            let hd = |i| h2f(u16::from_le_bytes([g[2 * i], g[2 * i + 1]]));
            for j in 0..256 {
                let bit = j * 6;
                let mut q = 0u8;
                for k in 0..6 {
                    q |= ((g[8 + (bit + k) / 8] >> ((bit + k) % 8)) & 1) << k;
                }
                w.push(f32::from(q) * hd(2 * (j / 128)) + hd(2 * (j / 128) + 1));
            }
        }
        let absmax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let raw = (f64::from(absmax) / 127.0) as f32;
        if raw == 0.0 {
            return (vec![0; w.len()], (0, 0));
        }
        let mut e = f64::from(raw).log2().floor() as i16 - 14;
        let mut sm = (f64::from(raw) * 2f64.powi(-i32::from(e))).ceil() as u16;
        if sm == 32768 {
            sm = 16384;
            e += 1;
        }
        let s = (f64::from(sm) * 2f64.powi(i32::from(e))) as f32;
        let codes = w
            .iter()
            .map(|&value| {
                let v = value / s;
                let floor = v.floor();
                let fraction = v - floor;
                let rounded = floor
                    + u8::from(fraction > 0.5 || (fraction == 0.5 && floor % 2.0 != 0.0)) as f32;
                rounded.clamp(-127.0, 127.0) as i8
            })
            .collect();
        (codes, (sm, e))
    }

    #[test]
    fn out_pack_matches_independent_quantizer_and_n512_b_stream() {
        let mut mq6 = synth_k(OUT_COLS, OUT_K, 0x6144);
        let rb = row_bytes(OUT_K);
        // Rows 3, 515 (panel boundary neighbours) and the last row are all-zero: zero scale, zero codes.
        for r in [3usize, 511, 512, OUT_COLS - 1] {
            mq6[r * rb..(r + 1) * rb].fill(0);
        }
        let packed = pack_out(Shape::GdnOut, &mq6, OUT_COLS, OUT_K).unwrap();
        assert_eq!(packed.shape, Shape::GdnOut);
        assert_eq!(
            (packed.panels.len(), packed.ws.len(), packed.scales.len()),
            (5, OUT_COLS, OUT_COLS)
        );
        let design = out_packing_design();
        assert_eq!(design.args[1].bytes, 8 * 96 * 4352);
        let mut scales = vec![(0u16, 0i16); OUT_COLS];
        let mut panels = Vec::new();
        for p in 0..5 {
            let mut b = vec![0i8; OUT_K * OUT_PANEL_COLS];
            for c in 0..OUT_PANEL_COLS {
                let row = p * OUT_PANEL_COLS + c;
                let (codes, grid) = ref_direct_row(&mq6[row * rb..(row + 1) * rb]);
                scales[row] = grid;
                for (k, &v) in codes.iter().enumerate() {
                    b[k * OUT_PANEL_COLS + c] = v;
                }
            }
            panels.push(pack_b_bf16(
                &design,
                &b,
                &scales[p * OUT_PANEL_COLS..(p + 1) * OUT_PANEL_COLS],
            ));
        }
        assert_eq!(packed.scales, scales);
        for (row, &(sm, se)) in scales.iter().enumerate() {
            assert_eq!(
                packed.ws[row],
                (f64::from(sm) * 2f64.powi(i32::from(se))) as f32,
                "ws row {row}"
            );
        }
        for r in [3usize, 511, 512, OUT_COLS - 1] {
            assert_eq!((packed.scales[r], packed.ws[r]), ((0, 0), 0.0));
        }
        assert!(
            packed.panels == panels,
            "pack_out panels != pack_b_bf16 of the independent W8 / grid scales"
        );
        // Same bytes for the other out shape; deterministic.
        let again = pack_out(Shape::QsaO, &mq6, OUT_COLS, OUT_K).unwrap();
        assert!(
            again.panels == packed.panels && again.ws == packed.ws && again.scales == packed.scales
        );
        let layers = pack_out_layers(&[(Shape::GdnOut, &mq6), (Shape::QsaO, &mq6)]).unwrap();
        assert_eq!(layers[0].panels, packed.panels);
        assert_eq!(
            (layers[1].shape, &layers[1].panels),
            (Shape::QsaO, &packed.panels)
        );
    }

    #[test]
    fn out_pack_validates_shape_dimensions_and_length() {
        let mq6 = synth_k(OUT_COLS, OUT_K, 7);
        assert!(pack_out(Shape::GdnIn, &mq6, OUT_COLS, OUT_K)
            .unwrap_err()
            .contains("not an out-projection"));
        assert!(pack_out(Shape::QsaO, &mq6, OUT_COLS - 1, OUT_K)
            .unwrap_err()
            .contains("2560x6144"));
        assert!(pack_out(Shape::QsaO, &mq6, OUT_COLS, 2560)
            .unwrap_err()
            .contains("2560x6144"));
        assert!(pack_out(Shape::QsaO, &mq6[1..], OUT_COLS, OUT_K)
            .unwrap_err()
            .contains("bytes"));
        let e = pack_out_layers(&[(Shape::GdnOut, &mq6), (Shape::QsaIn, &mq6)]).unwrap_err();
        assert!(e.starts_with("layer 1:"), "{e}");
        assert!(pack_out_layers(&[]).unwrap().is_empty());
    }
}
