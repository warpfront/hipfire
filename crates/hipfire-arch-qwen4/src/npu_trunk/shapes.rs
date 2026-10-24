// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused trunk shapes: which projections share an input, how their output columns are
//! concatenated, and the panel arithmetic of the resulting `B` matrix.
//!
//! A *shape* is a set of projections that read the same activation, so one NPU GEMM
//! computes all of them: `GdnIn = qkv|z|a|b` and `QsaIn = q|k|v|idx`. A [`FamilyMask`]
//! selects the projections that run on the NPU; the selected families' columns are
//! concatenated contiguously in fused order (no per-family padding), and only the whole
//! fused width is padded up to a multiple of [`PANEL_COLS`].

use hipfire_runtime::coproc::{
    EpilogueContract, FamilyDecl, FamilySet, GroupDecl, LayerKindDecl, StageMap,
};
use std::fmt;
use std::ops::BitOr;

/// Reduction dimension of every trunk projection.
pub const TRUNK_K: usize = 2560;
/// Output columns of one NPU panel (the G80 wave width).
pub const PANEL_COLS: usize = 1280;

const FAMILY_DECLS: [FamilyDecl; 8] = {
    let mut decls = [FamilyDecl {
        name: "",
        kind: 0,
        cols: 0,
    }; 8];
    let mut i = 0;
    while i < 8 {
        let family = Family::ALL[i];
        decls[i] = FamilyDecl {
            name: family.name(),
            kind: family.shape().index(),
            cols: family.cols(),
        };
        i += 1;
    }
    decls
};

/// Flash-Next's op-level stage map: 36 GDN and 12 QSA layers (K = 2560); the NPU may own the GDN input
/// projections `qkv` and `z|a|b` and the QSA input projections `q` and `k|v|idx`, each group a contiguous
/// fused-column range of its shape.
pub static FLASH_NEXT: StageMap = StageMap {
    tag: "npu-trunk",
    kinds: [
        LayerKindDecl {
            name: "gdn",
            layers: 36,
            in_k: TRUNK_K,
            out_family: "gdn.out",
        },
        LayerKindDecl {
            name: "qsa",
            layers: 12,
            in_k: TRUNK_K,
            out_family: "qsa.o",
        },
    ],
    families: &FAMILY_DECLS,
    groups: [
        GroupDecl {
            name: "qkv",
            kind: 0,
            families: 0x01,
        },
        GroupDecl {
            name: "zab",
            kind: 0,
            families: 0x0e,
        },
        GroupDecl {
            name: "q",
            kind: 1,
            families: 0x10,
        },
        GroupDecl {
            name: "kvidx",
            kind: 1,
            families: 0xe0,
        },
    ],
    contract: EpilogueContract::Bf16Direct,
};

impl From<FamilyMask> for FamilySet {
    fn from(mask: FamilyMask) -> Self {
        FamilySet(u32::from(mask.bits()))
    }
}

impl From<FamilySet> for FamilyMask {
    fn from(set: FamilySet) -> Self {
        FamilyMask::from_bits(set.0 as u8)
    }
}

/// One trunk projection that can be offloaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    QsaQ,
    QsaK,
    QsaV,
    QsaIdx,
}

impl Family {
    /// All families in fused order (GDN `qkv|z|a|b`, then QSA `q|k|v|idx`).
    pub const ALL: [Family; 8] = [
        Family::GdnQkv,
        Family::GdnZ,
        Family::GdnA,
        Family::GdnB,
        Family::QsaQ,
        Family::QsaK,
        Family::QsaV,
        Family::QsaIdx,
    ];

    /// Output features (matrix rows `M`) of the projection.
    pub const fn cols(self) -> usize {
        match self {
            Family::GdnQkv => 10240,
            Family::GdnZ => 6144,
            Family::GdnA => 48,
            Family::GdnB => 48,
            Family::QsaQ => 12288,
            Family::QsaK => 512,
            Family::QsaV => 512,
            Family::QsaIdx => 640,
        }
    }

    /// Position in [`Family::ALL`]; also the bit of the family in a [`FamilyMask`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The shape the family belongs to.
    pub const fn shape(self) -> Shape {
        match self {
            Family::GdnQkv | Family::GdnZ | Family::GdnA | Family::GdnB => Shape::GdnIn,
            _ => Shape::QsaIn,
        }
    }

    /// Short name used in errors and logs.
    pub const fn name(self) -> &'static str {
        match self {
            Family::GdnQkv => "gdn_qkv",
            Family::GdnZ => "gdn_z",
            Family::GdnA => "gdn_a",
            Family::GdnB => "gdn_b",
            Family::QsaQ => "qsa_q",
            Family::QsaK => "qsa_k",
            Family::QsaV => "qsa_v",
            Family::QsaIdx => "qsa_idx",
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Set of families that run on the NPU (bit `Family::index()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FamilyMask(u8);

impl FamilyMask {
    pub const NONE: FamilyMask = FamilyMask(0);
    pub const ALL: FamilyMask = FamilyMask(0xFF);
    /// The four GDN families.
    pub const GDN: FamilyMask = FamilyMask(0x0F);
    /// The four QSA families.
    pub const QSA: FamilyMask = FamilyMask(0xF0);

    pub const fn from_bits(bits: u8) -> Self {
        FamilyMask(bits)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, family: Family) -> bool {
        self.0 & (1 << family.index()) != 0
    }

    /// This mask plus `family`.
    pub const fn with(self, family: Family) -> Self {
        FamilyMask(self.0 | (1 << family.index()))
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for FamilyMask {
    type Output = FamilyMask;
    fn bitor(self, rhs: FamilyMask) -> FamilyMask {
        FamilyMask(self.0 | rhs.0)
    }
}

impl From<Family> for FamilyMask {
    fn from(family: Family) -> Self {
        FamilyMask(1 << family.index())
    }
}

impl BitOr<Family> for FamilyMask {
    type Output = FamilyMask;
    fn bitor(self, rhs: Family) -> FamilyMask {
        self.with(rhs)
    }
}

impl BitOr for Family {
    type Output = FamilyMask;
    fn bitor(self, rhs: Family) -> FamilyMask {
        FamilyMask::from(self).with(rhs)
    }
}

/// Fused column map: for each selected family, `(family, first fused column, columns)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnMap {
    pub parts: Vec<(Family, u32, u32)>,
}

impl ColumnMap {
    /// Live columns (sum of the selected families' columns).
    pub fn useful_cols(&self) -> usize {
        self.parts.iter().map(|&(_, _, cols)| cols as usize).sum()
    }

    /// Live columns rounded up to whole panels; `0` when nothing is selected.
    pub fn padded_cols(&self) -> usize {
        self.panel_count() * PANEL_COLS
    }

    /// Number of [`PANEL_COLS`]-wide panels.
    pub fn panel_count(&self) -> usize {
        self.useful_cols().div_ceil(PANEL_COLS)
    }

    /// Family and family-local column of a fused column; `None` in the padding or out of range.
    pub fn locate(&self, fused_col: usize) -> Option<(Family, usize)> {
        self.parts.iter().find_map(|&(family, col0, cols)| {
            let (col0, cols) = (col0 as usize, cols as usize);
            (fused_col >= col0 && fused_col < col0 + cols).then(|| (family, fused_col - col0))
        })
    }

    /// Panels `floor(first selected col0 / PANEL_COLS)..ceil(last selected end / PANEL_COLS)` covering the parts
    /// whose family is in `families`. `None` when none of this map's parts is selected, or when the selected parts are
    /// not adjacent in the map's fused order (a skipped retained family sits between them).
    pub fn panel_range(&self, families: FamilyMask) -> Option<std::ops::Range<usize>> {
        let mut selected = self
            .parts
            .iter()
            .enumerate()
            .filter(|(_, &(family, _, _))| families.contains(family));
        let (first, &(_, col0, cols)) = selected.next()?;
        let (mut last, mut end) = (first, (col0 + cols) as usize);
        for (i, &(_, c, n)) in selected {
            if i != last + 1 {
                return None;
            }
            (last, end) = (i, (c + n) as usize);
        }
        Some(col0 as usize / PANEL_COLS..end.div_ceil(PANEL_COLS))
    }
}

/// A fused trunk shape: the two input shapes (`GdnIn`, `QsaIn`; K = 2560, 1280-column panels) and the two out-projection
/// shapes (`GdnOut`, `QsaO`; K = 6144, N = 2560 as five 512-column panels of one pass, their own program / PDI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// GDN input projections `qkv|z|a|b` (16480 columns).
    GdnIn,
    /// QSA input projections `q|k|v|idx` (13952 columns).
    QsaIn,
    /// GDN out projection (`out_proj`, `[2560][6144]`).
    GdnOut,
    /// QSA attention output projection (`o_proj`, `[2560][6144]`).
    QsaO,
}

/// Reduction dimension of the out-projections.
pub const OUT_K: usize = 6144;
/// Output features of the out-projections (the model width).
pub const OUT_COLS: usize = 2560;
/// Output columns of one out-projection panel (the G80 N512 wave width).
pub const OUT_PANEL_COLS: usize = 512;

impl Shape {
    /// The input shapes in index order: the shapes the family masks, [`ColumnMap`]s and the in-projection programs cover.
    /// Out shapes are never part of a fused family map; see [`Shape::OUT`].
    pub const ALL: [Shape; 2] = [Shape::GdnIn, Shape::QsaIn];
    /// The out-projection shapes.
    pub const OUT: [Shape; 2] = [Shape::GdnOut, Shape::QsaO];

    /// Position in the declaration order (`0..4`); for the input shapes also the index into [`Shape::ALL`] / `PackedTrunk::map`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// `GdnOut` / `QsaO`.
    pub const fn is_out(self) -> bool {
        matches!(self, Shape::GdnOut | Shape::QsaO)
    }

    /// Reduction dimension: [`TRUNK_K`] for the input shapes, [`OUT_K`] for the out shapes.
    pub const fn k(self) -> usize {
        if self.is_out() {
            OUT_K
        } else {
            TRUNK_K
        }
    }

    /// Columns of one panel: [`PANEL_COLS`] (1280) or [`OUT_PANEL_COLS`] (512).
    pub const fn panel_cols(self) -> usize {
        if self.is_out() {
            OUT_PANEL_COLS
        } else {
            PANEL_COLS
        }
    }

    /// Panels of the full-width shape: 13 / 11 for the input shapes with every family selected, 5 for the out shapes
    /// (`2560 / 512`, one pass each).
    pub const fn panels(self) -> usize {
        match self {
            Shape::GdnIn => 13,
            Shape::QsaIn => 11,
            Shape::GdnOut | Shape::QsaO => OUT_COLS / OUT_PANEL_COLS,
        }
    }

    /// Submits per row block of one panel: always one (the whole K is accumulated on the array).
    pub const fn passes(self) -> usize {
        1
    }

    /// The shape's families in fused order (none for the out shapes: one matrix, no family fusion).
    pub const fn families(self) -> &'static [Family] {
        match self {
            Shape::GdnIn => &[Family::GdnQkv, Family::GdnZ, Family::GdnA, Family::GdnB],
            Shape::QsaIn => &[Family::QsaQ, Family::QsaK, Family::QsaV, Family::QsaIdx],
            Shape::GdnOut | Shape::QsaO => &[],
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Shape::GdnIn => "gdn_in",
            Shape::QsaIn => "qsa_in",
            Shape::GdnOut => "gdn_out",
            Shape::QsaO => "qsa_o",
        }
    }

    /// Fused column map of the families of `mask` that belong to this shape, concatenated in
    /// fused order without per-family padding. Empty for the out shapes.
    pub fn column_map(self, mask: FamilyMask) -> ColumnMap {
        let mut parts = Vec::new();
        let mut col0 = 0u32;
        for &family in self.families() {
            if mask.contains(family) {
                let cols = family.cols() as u32;
                parts.push((family, col0, cols));
                col0 += cols;
            }
        }
        ColumnMap { parts }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_range_selects_contiguous_groups_only() {
        let gdn = Shape::GdnIn.column_map(FamilyMask::ALL);
        let qsa = Shape::QsaIn.column_map(FamilyMask::ALL);
        let qkv = FamilyMask::from(Family::GdnQkv);
        assert_eq!(gdn.panel_range(qkv), Some(0..8));
        assert_eq!(gdn.panel_range(FamilyMask::from_bits(0x0e)), Some(8..13));
        assert_eq!(gdn.panel_range(FamilyMask::from_bits(0x0f)), Some(0..13));
        let q = FamilyMask::from(Family::QsaQ);
        assert_eq!(qsa.panel_range(q), Some(0..10));
        assert_eq!(qsa.panel_range(FamilyMask::from_bits(0xe0)), Some(9..11));
        assert_eq!(qsa.panel_range(FamilyMask::from_bits(0xf0)), Some(0..11));
        // A skipped retained family between two selected ones, and nothing selected.
        assert_eq!(gdn.panel_range(qkv | Family::GdnA), None);
        assert_eq!(
            gdn.panel_range(FamilyMask::from(Family::GdnZ) | Family::GdnB),
            None
        );
        assert_eq!(qsa.panel_range(q | Family::QsaIdx), None);
        assert_eq!(gdn.panel_range(FamilyMask::NONE), None);
        assert_eq!(gdn.panel_range(FamilyMask::QSA), None);
    }

    #[test]
    fn family_table_and_order() {
        let names: Vec<_> = Family::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(
            names,
            ["gdn_qkv", "gdn_z", "gdn_a", "gdn_b", "qsa_q", "qsa_k", "qsa_v", "qsa_idx"]
        );
        for (i, f) in Family::ALL.iter().enumerate() {
            assert_eq!(f.index(), i);
        }
        let gdn: usize = Shape::GdnIn.families().iter().map(|f| f.cols()).sum();
        let qsa: usize = Shape::QsaIn.families().iter().map(|f| f.cols()).sum();
        assert_eq!((gdn, qsa), (16480, 13952));
        for shape in Shape::ALL {
            assert_eq!(Shape::ALL[shape.index()], shape);
            for f in shape.families() {
                assert_eq!(f.shape(), shape);
            }
        }
    }

    #[test]
    fn mask_bits_and_ops() {
        assert_eq!(FamilyMask::default(), FamilyMask::NONE);
        assert!(FamilyMask::NONE.is_empty());
        assert_eq!(FamilyMask::GDN | FamilyMask::QSA, FamilyMask::ALL);
        assert_eq!(FamilyMask::ALL.bits(), 0xFF);
        for f in Family::ALL {
            assert!(FamilyMask::ALL.contains(f));
            assert!(!FamilyMask::NONE.contains(f));
            assert_eq!(FamilyMask::GDN.contains(f), f.shape() == Shape::GdnIn);
            assert_eq!(FamilyMask::QSA.contains(f), f.shape() == Shape::QsaIn);
            let m = FamilyMask::from(f);
            assert_eq!(m.bits().count_ones(), 1);
            assert!(m.contains(f));
        }
        let m = FamilyMask::from(Family::GdnZ).with(Family::QsaIdx);
        assert_eq!(m, FamilyMask::from_bits(0b1000_0010));
        assert_eq!(m, Family::GdnZ | Family::QsaIdx);
        assert!(m.contains(Family::GdnZ) && m.contains(Family::QsaIdx));
        assert!(!m.contains(Family::GdnA));
    }

    #[test]
    fn full_shapes_match_design_arithmetic() {
        let g = Shape::GdnIn.column_map(FamilyMask::ALL);
        assert_eq!(g.useful_cols(), 16480);
        assert_eq!(g.panel_count(), 13);
        assert_eq!(g.padded_cols(), 16640);
        assert_eq!(
            g.parts,
            [
                (Family::GdnQkv, 0, 10240),
                (Family::GdnZ, 10240, 6144),
                (Family::GdnA, 16384, 48),
                (Family::GdnB, 16432, 48),
            ]
        );
        let q = Shape::QsaIn.column_map(FamilyMask::ALL);
        assert_eq!(q.useful_cols(), 13952);
        assert_eq!(q.panel_count(), 11);
        assert_eq!(q.padded_cols(), 14080);
        assert_eq!(
            q.parts,
            [
                (Family::QsaQ, 0, 12288),
                (Family::QsaK, 12288, 512),
                (Family::QsaV, 12800, 512),
                (Family::QsaIdx, 13312, 640),
            ]
        );
    }

    #[test]
    fn masked_maps_are_contiguous_and_shape_local() {
        // The other shape's bits never leak in.
        assert_eq!(
            Shape::GdnIn.column_map(FamilyMask::QSA),
            ColumnMap::default()
        );
        let m = Family::GdnZ | Family::GdnB | Family::QsaQ;
        let g = Shape::GdnIn.column_map(m);
        assert_eq!(g.parts, [(Family::GdnZ, 0, 6144), (Family::GdnB, 6144, 48)]);
        assert_eq!(g.useful_cols(), 6192);
        assert_eq!(g.panel_count(), 5);
        assert_eq!(g.padded_cols(), 6400);
        let q = Shape::QsaIn.column_map(m);
        assert_eq!(q.parts, [(Family::QsaQ, 0, 12288)]);
        assert_eq!(q.panel_count(), 10);
        assert_eq!(q.padded_cols(), 12800);
    }

    #[test]
    fn empty_selection_has_zero_panels() {
        for shape in Shape::ALL {
            let m = shape.column_map(FamilyMask::NONE);
            assert!(m.parts.is_empty());
            assert_eq!(m.useful_cols(), 0);
            assert_eq!(m.panel_count(), 0);
            assert_eq!(m.padded_cols(), 0);
            assert_eq!(m.locate(0), None);
        }
    }

    #[test]
    fn exact_panel_boundary_has_no_padding() {
        // qkv alone is eight panels exactly (no padding); a/b alone is one mostly-padded panel.
        let m = Shape::GdnIn.column_map(Family::GdnQkv.into());
        assert_eq!(m.useful_cols(), 10240);
        assert_eq!(m.panel_count(), 8);
        assert_eq!(m.padded_cols(), 10240);
        let m = Shape::GdnIn.column_map(Family::GdnA | Family::GdnB);
        assert_eq!(
            (m.useful_cols(), m.panel_count(), m.padded_cols()),
            (96, 1, 1280)
        );
    }

    #[test]
    fn locate_maps_fused_columns() {
        let q = Shape::QsaIn.column_map(FamilyMask::QSA);
        assert_eq!(q.locate(0), Some((Family::QsaQ, 0)));
        assert_eq!(q.locate(12287), Some((Family::QsaQ, 12287)));
        assert_eq!(q.locate(12288), Some((Family::QsaK, 0)));
        assert_eq!(q.locate(12800), Some((Family::QsaV, 0)));
        assert_eq!(q.locate(13311), Some((Family::QsaV, 511)));
        assert_eq!(q.locate(13312), Some((Family::QsaIdx, 0)));
        assert_eq!(q.locate(13951), Some((Family::QsaIdx, 639)));
        // Padding and beyond.
        assert_eq!(q.locate(13952), None);
        assert_eq!(q.locate(14079), None);
        assert_eq!(q.locate(usize::MAX), None);
        // Masked: columns shift down.
        let m = Shape::QsaIn.column_map(Family::QsaK | Family::QsaIdx);
        assert_eq!(m.locate(511), Some((Family::QsaK, 511)));
        assert_eq!(m.locate(512), Some((Family::QsaIdx, 0)));
        assert_eq!(m.locate(1151), Some((Family::QsaIdx, 639)));
        assert_eq!(m.locate(1152), None);
    }

    #[test]
    fn stage_map_groups_are_contiguous_ranges_of_the_fused_shapes() {
        FLASH_NEXT.validate().unwrap();
        let all = FamilyMask::from(FLASH_NEXT.all());
        assert_eq!(all, FamilyMask::ALL);
        for (bit, group) in FLASH_NEXT.groups.iter().enumerate() {
            let mask = FamilyMask::from(FLASH_NEXT.subset(1 << bit));
            let shape = Shape::ALL[group.kind];
            assert!(shape.column_map(all).panel_range(mask).is_some());
            assert!(Family::ALL
                .iter()
                .all(|f| !mask.contains(*f) || f.shape() == shape));
        }
    }

    #[test]
    fn out_shapes_are_five_512_column_panels_of_one_pass() {
        assert_eq!(Shape::OUT, [Shape::GdnOut, Shape::QsaO]);
        assert_eq!(Shape::ALL.map(Shape::index), [0, 1]);
        assert_eq!(Shape::OUT.map(Shape::index), [2, 3]);
        for shape in Shape::OUT {
            assert!(shape.is_out());
            assert_eq!(
                (
                    shape.k(),
                    shape.panel_cols(),
                    shape.panels(),
                    shape.passes()
                ),
                (6144, 512, 5, 1)
            );
            assert_eq!(shape.panels() * shape.panel_cols(), OUT_COLS);
            assert!(shape.families().is_empty());
            assert!(shape.column_map(FamilyMask::ALL).parts.is_empty());
        }
        for shape in Shape::ALL {
            assert!(!shape.is_out());
            assert_eq!((shape.k(), shape.panel_cols()), (TRUNK_K, PANEL_COLS));
        }
        assert_eq!(
            (Shape::GdnOut.name(), Shape::QsaO.name()),
            ("gdn_out", "qsa_o")
        );
    }
}
