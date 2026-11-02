// SPDX-License-Identifier: Apache-2.0
//! Metadata of `build_programs` / `build_programs_for_families`: G80 direct-BF16 argument sizes (computed here from the
//! documented layout, not from the design), row bounds and rounding, sidecar bytes, and masked panel counts.
//! Nothing here runs the simulator or touches a device.
use hipfire_arch_qwen4::npu_trunk::{
    build_out_programs, build_programs, build_programs_for_families, Family, FamilyMask,
    NpuPrograms, Shape, OUT_COLS, OUT_K, OUT_PANEL_COLS, PANEL_COLS, TRUNK_K,
};
use hipfire_runtime::coproc::EpilogueContract;

const WAVE_M: usize = 256;
const K_CHUNK: usize = 64;
/// One K chunk of the A half one injecting column carries (128 rows x 64 / 2 codes + Dm[128] + De[128]).
const A_CHUNK: usize = 4096 + 512;
/// One K chunk of one core column's B segment (two 80-wide N tiles x 64 codes + Sm[96] + Se[96] each).
const B_CHUNK: usize = 2 * (5120 + 384);
/// One core BF16 C tile (128 x 80 x 2 bytes).
const C_TILE: usize = 128 * 80 * 2;
const COLS: usize = 8;
const A_COLS: usize = 4;

fn waves(rows: usize) -> usize {
    rows.div_ceil(WAVE_M)
}
fn expected_args(rows: usize) -> [usize; 3] {
    let (kc, w) = (TRUNK_K / K_CHUNK, waves(rows));
    [
        A_COLS * w * kc * A_CHUNK,
        COLS * kc * B_CHUNK,
        COLS * w * 4 * C_TILE,
    ]
}
fn mask(families: &[Family]) -> FamilyMask {
    families
        .iter()
        .fold(FamilyMask::NONE, |m, &f| m | FamilyMask::from(f))
}
/// Panels of a shape for the selected families, from the family widths alone: fused order, no gaps between families.
fn expected_panels(shape: Shape, selected: FamilyMask) -> (usize, usize) {
    let useful: usize = shape
        .families()
        .iter()
        .filter(|&&f| selected.contains(f))
        .map(|f| f.cols())
        .sum();
    (
        useful.div_ceil(PANEL_COLS),
        useful.div_ceil(PANEL_COLS) * PANEL_COLS,
    )
}
fn check(progs: &NpuPrograms, rows_requested: usize, selected: FamilyMask, what: &str) {
    assert_eq!(
        progs.rows_max,
        waves(rows_requested) * WAVE_M,
        "{what}: rows_max is rounded up to whole 256-row waves"
    );
    assert!(
        progs.rows_max >= rows_requested && progs.rows_max < rows_requested + WAVE_M,
        "{what}"
    );
    let args = expected_args(rows_requested);
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let p = progs.get(shape);
        assert!(
            std::ptr::eq(p, &progs.per_shape[i]),
            "{what}: get(shape) returns per_shape[shape.index()]"
        );
        assert_eq!(shape.index(), i, "{what}: Shape::ALL order is index order");
        assert_eq!(
            p.shape.index(),
            shape.index(),
            "{what}: record carries its shape"
        );
        let (panels, padded) = expected_panels(shape, selected);
        assert_eq!(
            (p.panel_count, p.padded_cols),
            (panels, padded),
            "{what}: {i} panels / padded columns"
        );
        assert_eq!(
            p.padded_cols,
            p.panel_count * PANEL_COLS,
            "{what}: padded columns are whole panels"
        );
        assert_eq!(p.arg_bytes, args, "{what}: shape {i} [A, B, C] bytes");
        assert_eq!(
            [
                p.design.args[0].bytes,
                p.design.args[1].bytes,
                p.design.args[2].bytes
            ],
            args,
            "{what}: design args"
        );
        assert_eq!(
            p.host_bytes_read,
            (args[0] + args[1]) as u64,
            "{what}: host bytes read per submit"
        );
        assert_eq!(
            p.host_bytes_written, args[2] as u64,
            "{what}: host bytes written per submit"
        );
        assert_eq!(
            (p.host_bytes_read, p.host_bytes_written),
            (p.design.host_bytes_read(), p.design.host_bytes_written()),
            "{what}"
        );
        assert!(
            !p.design.pdi.is_empty() && !p.design.insts.is_empty(),
            "{what}: PDI and TXN are built"
        );
        assert_eq!(p.design.waves(), waves(rows_requested), "{what}: waves");
    }
}

#[test]
fn full_mask_published_panel_counts() {
    let progs = build_programs(EpilogueContract::Bf16Direct, 1).unwrap();
    // GdnIn qkv|z|a|b = 10240+6144+48+48 = 16480 cols, QsaIn q|k|v|idx = 12288+512+512+640 = 13952 cols.
    let (g, q) = (progs.get(Shape::GdnIn), progs.get(Shape::QsaIn));
    assert_eq!((g.panel_count, g.padded_cols), (13, 16640));
    assert_eq!((q.panel_count, q.padded_cols), (11, 14080));
    assert!(matches!(progs.contract, EpilogueContract::Bf16Direct));
}

#[test]
fn rows_are_rounded_to_waves_and_sizes_follow_the_g80_bf16_layout() {
    for rows in [
        1, 2, 255, 256, 257, 511, 512, 513, 4096, 16384, 65535, 65536,
    ] {
        let progs = build_programs(EpilogueContract::Bf16Direct, rows).unwrap();
        check(&progs, rows, FamilyMask::ALL, &format!("rows {rows}"));
    }
}

#[test]
fn per_submit_sizes_pin_known_values() {
    // One wave: A 4 cols * 40 chunks * 4608 = 737280, B 8 * 40 * 11008 = 3522560, C 8 * 4 * 20480 = 655360.
    let one = build_programs(EpilogueContract::Bf16Direct, 256).unwrap();
    for shape in Shape::ALL {
        assert_eq!(one.get(shape).arg_bytes, [737_280, 3_522_560, 655_360]);
    }
    // 256 waves: A and C scale with the waves, B does not.
    let max = build_programs(EpilogueContract::Bf16Direct, 65536).unwrap();
    for shape in Shape::ALL {
        let p = max.get(shape);
        assert_eq!(p.arg_bytes, [737_280 * 256, 3_522_560, 655_360 * 256]);
        assert_eq!(p.design.waves(), 256);
    }
}

#[test]
fn rows_outside_one_to_65536_are_rejected_without_building() {
    for rows in [0, 65537, 65792, usize::MAX] {
        let err = build_programs(EpilogueContract::Bf16Direct, rows).unwrap_err();
        assert!(
            err.contains(&rows.to_string()) && err.contains("65536"),
            "rows {rows}: {err}"
        );
        assert!(
            build_programs_for_families(EpilogueContract::Bf16Direct, rows, FamilyMask::ALL)
                .is_err()
        );
    }
}

#[test]
fn bf16_direct_builds_with_sidecar_byte_totals() {
    for rows in [1, 4096, 65536] {
        let progs = build_programs(EpilogueContract::Bf16Direct, rows).unwrap();
        let waves = rows.div_ceil(256);
        for shape in Shape::ALL {
            let p = progs.get(shape);
            let sizes = [
                4 * waves * 40 * 4608,
                8 * 40 * 2 * 5504,
                8 * waves * 4 * 20480,
            ];
            assert_eq!(p.arg_bytes, sizes);
            assert_eq!(
                p.design.epilogue(),
                Some(peacemaker_npu::kernels::gemm_core::Epilogue::Bf16)
            );
            assert_eq!(p.host_bytes_read, (sizes[0] + sizes[1]) as u64);
            assert_eq!(p.host_bytes_written, sizes[2] as u64);
        }
        assert!(
            build_programs_for_families(EpilogueContract::Bf16Direct, rows, FamilyMask::GDN)
                .is_ok()
        );
    }
    assert!(build_programs(EpilogueContract::Bf16Direct, 0).is_err());
}

#[test]
fn masked_programs_follow_the_column_map_panel_counts() {
    use Family::*;
    let masks: Vec<(&str, FamilyMask)> = vec![
        ("ALL", FamilyMask::ALL),
        ("GDN", FamilyMask::GDN),
        ("QSA", FamilyMask::QSA),
        ("GdnA|GdnB", mask(&[GdnA, GdnB])),
        ("QsaIdx", mask(&[QsaIdx])),
        ("GdnQkv", mask(&[GdnQkv])),
        ("GdnQkv|GdnZ", mask(&[GdnQkv, GdnZ])),
        ("QsaQ", mask(&[QsaQ])),
        ("QsaK|QsaV|QsaIdx", mask(&[QsaK, QsaV, QsaIdx])),
        ("GdnZ|QsaV", mask(&[GdnZ, QsaV])),
        ("GdnA|QsaIdx", mask(&[GdnA, QsaIdx])),
    ];
    for (name, m) in masks {
        for rows in [1, 300] {
            let progs = build_programs_for_families(EpilogueContract::Bf16Direct, rows, m).unwrap();
            check(&progs, rows, m, name);
            for shape in Shape::ALL {
                let map = shape.column_map(m);
                let p = progs.get(shape);
                assert_eq!(
                    (p.panel_count, p.padded_cols),
                    (map.panel_count(), map.padded_cols()),
                    "{name}: ColumnMap agreement"
                );
                assert!(
                    map.useful_cols() <= p.padded_cols
                        && p.padded_cols - map.useful_cols() < PANEL_COLS,
                    "{name}: padding < one panel"
                );
            }
        }
    }
    // Hand-checked values independent of Family::cols().
    let p = build_programs_for_families(EpilogueContract::Bf16Direct, 256, mask(&[GdnA, GdnB]))
        .unwrap();
    assert_eq!(
        (
            p.get(Shape::GdnIn).panel_count,
            p.get(Shape::GdnIn).padded_cols
        ),
        (1, 1280)
    );
    assert_eq!(
        (
            p.get(Shape::QsaIn).panel_count,
            p.get(Shape::QsaIn).padded_cols
        ),
        (0, 0)
    );
    let p =
        build_programs_for_families(EpilogueContract::Bf16Direct, 256, mask(&[QsaIdx])).unwrap();
    assert_eq!(
        (
            p.get(Shape::GdnIn).panel_count,
            p.get(Shape::QsaIn).panel_count
        ),
        (0, 1)
    );
    let p =
        build_programs_for_families(EpilogueContract::Bf16Direct, 256, mask(&[GdnQkv])).unwrap();
    assert_eq!(p.get(Shape::GdnIn).panel_count, 8);
    let p = build_programs_for_families(EpilogueContract::Bf16Direct, 256, mask(&[QsaQ])).unwrap();
    assert_eq!(p.get(Shape::QsaIn).panel_count, 10);
    // A zero-panel shape still carries the (mask-independent) design and sizes of its row block.
    assert_eq!(p.get(Shape::GdnIn).arg_bytes, [737_280, 3_522_560, 655_360]);
}

#[test]
fn empty_mask_is_rejected() {
    let err = build_programs_for_families(EpilogueContract::Bf16Direct, 256, FamilyMask::NONE)
        .unwrap_err();
    assert!(err.contains("no families"), "{err}");
}

#[test]
fn full_build_is_the_all_families_masked_build() {
    let a = build_programs(EpilogueContract::Bf16Direct, 1000).unwrap();
    let b =
        build_programs_for_families(EpilogueContract::Bf16Direct, 1000, FamilyMask::ALL).unwrap();
    assert_eq!(a.rows_max, b.rows_max);
    for shape in Shape::ALL {
        let (x, y) = (a.get(shape), b.get(shape));
        assert_eq!(
            (x.panel_count, x.padded_cols, x.arg_bytes),
            (y.panel_count, y.padded_cols, y.arg_bytes)
        );
        assert_eq!(
            (&x.design.pdi, &x.design.insts),
            (&y.design.pdi, &y.design.insts),
            "deterministic bytes"
        );
    }
}

/// Out-projection programs (`build_out_programs`): K = 6144, five 512-column panels, 512-row waves, one design / PDI shared by
/// `GdnOut` and `QsaO` and distinct from the input shapes'; sizes computed here from the documented layout.
#[test]
fn out_programs_are_k6144_n512_with_their_own_pdi() {
    for rows in [1usize, 511, 512, 513, 8192, 65535, 65536] {
        let progs = build_out_programs(rows).unwrap();
        let waves = rows.div_ceil(512);
        assert_eq!(progs.rows_max, waves * 512);
        let args = [
            8 * waves * 96 * 4608,
            8 * 96 * 4352,
            waves * 512 * OUT_COLS * 2,
        ];
        for shape in Shape::OUT {
            let p = progs.get(shape);
            assert_eq!(
                (p.shape, p.panel_count, p.padded_cols),
                (shape, 5, OUT_COLS)
            );
            assert_eq!(p.arg_bytes, args, "rows {rows} {shape}");
            assert_eq!(
                (p.host_bytes_read, p.host_bytes_written),
                (
                    (args[0] + args[1]) as u64,
                    (waves * 512 * OUT_PANEL_COLS * 2) as u64
                )
            ); // one panel's BF16 columns per submit
            assert_eq!(
                p.design.epilogue(),
                Some(peacemaker_npu::kernels::gemm_core::Epilogue::Bf16)
            );
            assert_eq!(p.design.waves(), waves);
        }
        assert_eq!(
            progs.get(Shape::GdnOut).design.pdi,
            progs.get(Shape::QsaO).design.pdi
        );
    }
    let out = build_out_programs(512).unwrap();
    let input = build_programs(EpilogueContract::Bf16Direct, 512).unwrap();
    assert_ne!(
        out.get(Shape::GdnOut).design.pdi,
        input.get(Shape::GdnIn).design.pdi,
        "second PDI"
    );
    assert_eq!(OUT_PANEL_COLS * 5, OUT_COLS);
    assert_eq!(OUT_K, 6144);
    assert!(std::panic::catch_unwind(|| build_out_programs(512)
        .unwrap()
        .get(Shape::GdnIn)
        .panel_count)
    .is_err());
    for rows in [0usize, 65537, usize::MAX] {
        let err = build_out_programs(rows).unwrap_err();
        assert!(
            err.contains(&rows.to_string()) && err.contains("65536"),
            "{err}"
        );
    }
    let _ = (PANEL_COLS, TRUNK_K);
}
