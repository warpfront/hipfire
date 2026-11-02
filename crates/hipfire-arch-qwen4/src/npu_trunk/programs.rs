// SPDX-License-Identifier: Apache-2.0
//! G80 program construction: one deployment image + command stream per trunk shape, CPU only (no device is opened).
//!
//! Every submit of a trunk program multiplies one 256-row-wave-padded A (`rows_max` rounded up to 256 rows, `K = 2560`) by
//! ONE resident 1280-column B panel, with exact int32 accumulation and the integer direct-BF16 epilogue.
//! A shape needs `panel_count` submits per row block; the program is independent of the panel (B is an argument).
//! Row-major programs differ only in their fused C row pitch; native-tile programs share a design.

use peacemaker_npu::kernels::gemm_array::ArrayDesign;
use peacemaker_npu::kernels::gemm_core::Epilogue;
use peacemaker_npu::kernels::gemm_g80::{build_panel_program, build_panel_program_rowmajor};
use peacemaker_npu::kernels::gemm_g80_n512::build_panel_program_n512;

use super::shapes::{FamilyMask, Shape, OUT_COLS, OUT_K};
use hipfire_runtime::coproc::EpilogueContract;

/// Rows of one G80 M wave: every program is built for a whole number of these.
const WAVE_ROWS: usize = 256;
/// Rows of one G80 N512 (out-projection) M wave.
const OUT_WAVE_ROWS: usize = 512;
/// Largest row count a G80 design accepts (256 waves of 256 rows).
const ROWS_LIMIT: usize = 256 * WAVE_ROWS;

/// The program of one trunk shape.
pub struct ShapeProgram {
    pub shape: Shape,
    /// G80 PDI/TXN and argument layout: rounded M, N1280, K2560, selected epilogue, Fast control.
    pub design: ArrayDesign,
    /// 1280-column B panels (submits per row block) of the selected families of this shape; 0 when none is selected.
    pub panel_count: usize,
    /// `panel_count * 1280`: fused columns including zero padding.
    pub padded_cols: usize,
    /// Bytes of `[A, B, C]` of ONE submit (`design.args`).
    pub arg_bytes: [usize; 3],
    /// Host bytes read by ONE submit (packed A + one packed B panel).
    pub host_bytes_read: u64,
    /// Host bytes written by ONE submit (the BF16 C arena).
    pub host_bytes_written: u64,
}

/// The programs of both shapes for one epilogue contract.
pub struct NpuPrograms {
    pub contract: EpilogueContract,
    /// Rows the programs are built for: the requested maximum rounded up to a whole 256-row wave.
    pub rows_max: usize,
    /// Indexed by [`Shape::index`] in [`Shape::ALL`] order.
    pub per_shape: [ShapeProgram; 2],
}

impl NpuPrograms {
    /// The program of `shape`.
    pub fn get(&self, shape: Shape) -> &ShapeProgram {
        &self.per_shape[shape.index()]
    }
}

impl std::fmt::Debug for ShapeProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeProgram")
            .field("shape", &self.shape.index())
            .field("panel_count", &self.panel_count)
            .field("padded_cols", &self.padded_cols)
            .field("arg_bytes", &self.arg_bytes)
            .field("host_bytes_read", &self.host_bytes_read)
            .field("host_bytes_written", &self.host_bytes_written)
            .field("pdi_bytes", &self.design.pdi.len())
            .field("txn_bytes", &self.design.insts.len())
            .finish()
    }
}

impl std::fmt::Debug for NpuPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NpuPrograms")
            .field("rows_max", &self.rows_max)
            .field("per_shape", &self.per_shape)
            .finish()
    }
}

/// Programs for every family of both shapes.
pub fn build_programs(contract: EpilogueContract, rows_max: usize) -> Result<NpuPrograms, String> {
    build_programs_for_families(contract, rows_max, FamilyMask::ALL)
}

/// Programs for the families in `families`: the panel counts and padded columns follow
/// [`Shape::column_map`] of the mask, so they match the panels `pack_shape` produces for the same mask.
/// `Err` for an empty mask or `rows_max` outside `1..=65536`.
pub fn build_programs_for_families(
    contract: EpilogueContract,
    rows_max: usize,
    families: FamilyMask,
) -> Result<NpuPrograms, String> {
    let epilogue = match contract {
        EpilogueContract::Bf16Direct => Epilogue::Bf16,
    };
    if !(1..=ROWS_LIMIT).contains(&rows_max) {
        return Err(format!(
            "npu-trunk: rows_max {rows_max} outside 1..={ROWS_LIMIT}"
        ));
    }
    if families.bits() == 0 {
        return Err("npu-trunk: no families selected, nothing to program".to_string());
    }
    let rows = rows_max.div_ceil(WAVE_ROWS) * WAVE_ROWS;
    let program = |shape: Shape| {
        let map = shape.column_map(families);
        let design = build_panel_program(rows, epilogue).expect("validated G80 rows");
        let arg_bytes = [
            design.args[0].bytes,
            design.args[1].bytes,
            design.args[2].bytes,
        ];
        ShapeProgram {
            shape,
            panel_count: map.panel_count(),
            padded_cols: map.padded_cols(),
            arg_bytes,
            host_bytes_read: design.host_bytes_read(),
            host_bytes_written: design.host_bytes_written(),
            design,
        }
    };
    Ok(NpuPrograms {
        contract,
        rows_max: rows,
        per_shape: [program(Shape::ALL[0]), program(Shape::ALL[1])],
    })
}

/// Compacted row-major pitches; zero-panel shapes are not submitted.
pub fn build_rowmajor_programs_for_families(
    rows_max: usize,
    families: FamilyMask,
) -> Result<NpuPrograms, String> {
    if !(1..=ROWS_LIMIT).contains(&rows_max) {
        return Err(format!(
            "npu-trunk: rows_max {rows_max} outside 1..={ROWS_LIMIT}"
        ));
    }
    let rows = rows_max.div_ceil(WAVE_ROWS) * WAVE_ROWS;
    let program = |shape: Shape| -> Result<ShapeProgram, String> {
        let map = shape.column_map(families);
        // A zero-panel record carries a valid unused design, never an arena.
        let design = build_panel_program_rowmajor(rows, map.padded_cols().max(1280), 0)?;
        Ok(ShapeProgram {
            shape,
            panel_count: map.panel_count(),
            padded_cols: map.padded_cols(),
            arg_bytes: [
                design.args[0].bytes,
                design.args[1].bytes,
                design.args[2].bytes,
            ],
            host_bytes_read: design.host_bytes_read(),
            host_bytes_written: design.host_bytes_written(),
            design,
        })
    };
    Ok(NpuPrograms {
        contract: EpilogueContract::Bf16Direct,
        rows_max: rows,
        per_shape: [program(Shape::GdnIn)?, program(Shape::QsaIn)?],
    })
}

/// The programs of both out-projection shapes: one K6144, N512 BF16 row-major design per shape (the same program and PDI
/// for [`Shape::GdnOut`] and [`Shape::QsaO`], a PDI distinct from the input shapes').
pub struct NpuOutPrograms {
    /// Rows the programs are built for: the requested maximum rounded up to a whole 512-row wave.
    pub rows_max: usize,
    /// Indexed by `shape.index() - 2` in [`Shape::OUT`] order.
    pub per_shape: [ShapeProgram; 2],
}

impl NpuOutPrograms {
    /// The program of the out shape `shape`.
    ///
    /// # Panics
    /// For an input shape.
    pub fn get(&self, shape: Shape) -> &ShapeProgram {
        assert!(shape.is_out(), "{shape} is not an out-projection shape");
        &self.per_shape[shape.index() - Shape::OUT[0].index()]
    }
}

impl std::fmt::Debug for NpuOutPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NpuOutPrograms")
            .field("rows_max", &self.rows_max)
            .field("per_shape", &self.per_shape)
            .finish()
    }
}

/// Out-projection programs: each shape's design writes BF16 rows into a compact fused arena of pitch `OUT_COLS` (2560) at
/// `col0 = 0`; panel `p` advances arg2's base by `2 * 512 * p` bytes (the relay's per-panel command). M is rounded up to
/// 512-row waves. `Err` for `rows_max` outside `1..=65536`.
pub fn build_out_programs(rows_max: usize) -> Result<NpuOutPrograms, String> {
    if !(1..=ROWS_LIMIT).contains(&rows_max) {
        return Err(format!(
            "npu-trunk: rows_max {rows_max} outside 1..={ROWS_LIMIT}"
        ));
    }
    let rows = rows_max.div_ceil(OUT_WAVE_ROWS) * OUT_WAVE_ROWS;
    let program = |shape: Shape| -> Result<ShapeProgram, String> {
        let design = build_panel_program_n512(rows, OUT_COLS, 0, OUT_K)?;
        Ok(ShapeProgram {
            shape,
            panel_count: shape.panels(),
            padded_cols: shape.panels() * shape.panel_cols(),
            arg_bytes: [
                design.args[0].bytes,
                design.args[1].bytes,
                design.args[2].bytes,
            ],
            host_bytes_read: design.host_bytes_read(),
            host_bytes_written: design.host_bytes_written(),
            design,
        })
    };
    Ok(NpuOutPrograms {
        rows_max: rows,
        per_shape: [program(Shape::GdnOut)?, program(Shape::QsaO)?],
    })
}

/// Rows of a `rows`-row bank rounded up to the whole waves of `shape`'s program: 256-row waves (input shapes), 512-row
/// waves (out shapes). The relay runs the TXN of exactly this many rows and the producer packs A for exactly this many.
pub fn wave_padded_rows(shape: Shape, rows: usize) -> usize {
    let wave = if shape.is_out() {
        OUT_WAVE_ROWS
    } else {
        WAVE_ROWS
    };
    rows.div_ceil(wave) * wave
}

/// The PDI and command stream (TXN) of `shape`'s program at `padded_rows` (a whole number of the shape's waves), with the
/// same fused C pitch (`padded_cols` for input shapes, [`OUT_COLS`] for out shapes) as the full-row program. The 32 cores'
/// wave loop count is part of the PDI, so every row count is its own PDI (one relay hardware context) and TXN; the two input
/// shapes of one row count share their PDI. Pure CPU; the relay builds and caches one per (shape, rows).
pub fn build_program_for_rows(
    shape: Shape,
    padded_cols: usize,
    padded_rows: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if padded_rows == 0
        || padded_rows != wave_padded_rows(shape, padded_rows)
        || padded_rows > ROWS_LIMIT
    {
        return Err(format!("npu-trunk: {padded_rows} rows is not a whole number of {shape} waves within 1..={ROWS_LIMIT}"));
    }
    let design = if shape.is_out() {
        build_panel_program_n512(padded_rows, OUT_COLS, 0, OUT_K)?
    } else {
        build_panel_program_rowmajor(padded_rows, padded_cols.max(1280), 0)?
    };
    Ok((design.pdi, design.insts))
}

#[cfg(test)]
mod rowmajor_tests {
    use super::super::Family;
    use super::*;

    #[test]
    fn shape_programs_declare_fused_arena_sizes() {
        let programs = build_rowmajor_programs_for_families(8192, FamilyMask::ALL).unwrap();
        for p in &programs.per_shape {
            assert_eq!(p.arg_bytes[2], 8192 * p.padded_cols * 2);
            assert_eq!(p.panel_count * 1280, p.padded_cols);
        }
        assert_eq!(
            programs.per_shape[0].design.pdi,
            programs.per_shape[1].design.pdi
        );
    }
    #[test]
    fn qkv_only_compacts_pitch_and_omits_qsa() {
        let programs = build_rowmajor_programs_for_families(256, Family::GdnQkv.into()).unwrap();
        let gdn = programs.get(Shape::GdnIn);
        assert_eq!((gdn.panel_count, gdn.padded_cols), (8, 10240));
        assert_eq!(gdn.arg_bytes[2], 256 * 10240 * 2);
        assert_eq!(programs.get(Shape::QsaIn).panel_count, 0);
    }
    #[test]
    fn row_adaptive_programs_are_whole_wave_pdi_and_txn_pairs() {
        use std::time::Instant;
        let full = build_rowmajor_programs_for_families(8192, FamilyMask::ALL).unwrap();
        let out_full = build_out_programs(8192).unwrap();
        for (shape, rows, padded) in [
            (Shape::GdnIn, 4672, 4864),
            (Shape::QsaIn, 4647, 4864),
            (Shape::GdnIn, 1127, 1280),
            (Shape::QsaO, 4672, 5120),
            (Shape::QsaO, 8192, 8192),
            (Shape::GdnIn, 8192, 8192),
        ] {
            assert_eq!(wave_padded_rows(shape, rows), padded);
        }
        // The full-row program is the 8192-row case of the builder, byte for byte.
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            let program = full.get(shape);
            let (pdi, txn) = build_program_for_rows(shape, program.padded_cols, 8192).unwrap();
            assert_eq!(
                (pdi, txn),
                (program.design.pdi.clone(), program.design.insts.clone())
            );
        }
        let (pdi, txn) = build_program_for_rows(Shape::QsaO, OUT_COLS, 8192).unwrap();
        assert_eq!(
            (pdi, txn),
            (
                out_full.get(Shape::QsaO).design.pdi.clone(),
                out_full.get(Shape::QsaO).design.insts.clone()
            )
        );
        // Other row counts: the PDI carries the wave loop count (every count differs from the full one), the two input
        // shapes of one row count share it, the TXN shrinks with the waves. Build cost is printed (`--nocapture`).
        let (gdn_cols, qsa_cols) = (
            full.get(Shape::GdnIn).padded_cols,
            full.get(Shape::QsaIn).padded_cols,
        );
        let mut previous = 0;
        for rows in [256usize, 1280, 4352, 4864, 6144] {
            let started = Instant::now();
            let (gdn_pdi, gdn_txn) = build_program_for_rows(Shape::GdnIn, gdn_cols, rows).unwrap();
            let gdn_ms = started.elapsed().as_secs_f64() * 1e3;
            let started = Instant::now();
            let (qsa_pdi, qsa_txn) = build_program_for_rows(Shape::QsaIn, qsa_cols, rows).unwrap();
            let qsa_ms = started.elapsed().as_secs_f64() * 1e3;
            println!("program rows={rows} waves={} gdn_in txn={} B build={gdn_ms:.1} ms, qsa_in txn={} B build={qsa_ms:.1} ms, pdi={} B",
                rows / 256, gdn_txn.len(), qsa_txn.len(), gdn_pdi.len());
            assert_eq!(gdn_pdi, qsa_pdi, "{rows}");
            assert_ne!(gdn_pdi, full.get(Shape::GdnIn).design.pdi, "{rows}");
            assert!(
                gdn_txn.len() > previous
                    && gdn_txn.len() < full.get(Shape::GdnIn).design.insts.len()
            );
            previous = gdn_txn.len();
        }
        let started = Instant::now();
        let (out_pdi, out_txn) = build_program_for_rows(Shape::QsaO, OUT_COLS, 5120).unwrap();
        println!(
            "program rows=5120 waves=10 qsa_o txn={} B build={:.1} ms pdi={} B",
            out_txn.len(),
            started.elapsed().as_secs_f64() * 1e3,
            out_pdi.len()
        );
        assert_ne!(out_pdi, out_full.get(Shape::QsaO).design.pdi);
        for bad in [0usize, 100, 8000, ROWS_LIMIT + 256] {
            assert!(
                build_program_for_rows(Shape::GdnIn, 16640, bad).is_err(),
                "{bad}"
            );
        }
        assert!(
            build_program_for_rows(Shape::QsaO, OUT_COLS, 4864).is_err(),
            "out waves are 512 rows"
        );
    }
}
