// SPDX-License-Identifier: Apache-2.0
//! Bank-pair geometry of a co-processor group: where the session snapshot row cuts a group, and what
//! each bank launches. The executor and the numerical twin both derive their chunking from here.

/// Where in a pair group the session snapshot row falls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutSite {
    /// No interior snapshot row (the group end is published by the driver).
    None,
    /// Exactly the A/B boundary: bank A runs whole, state staged after it.
    Split,
    /// Inside `bank` at its local `row`: that bank's Core runs as two launches.
    Bank { bank: usize, row: usize },
}

/// Bank and sub-launch geometry of one pair group: two banks and the group-local
/// row of an interior session snapshot. Both the executor and the numerical twin
/// derive their chunking from this, so they cut at identical rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairShape {
    pub rows: [usize; 2],
    pub cut: Option<usize>,
}

impl PairShape {
    /// `cut` is the session boundary `boundary` made group-local at `position`;
    /// only a boundary strictly inside the group is a cut.
    pub fn new(rows: [usize; 2], position: usize, boundary: Option<usize>) -> Self {
        let total = rows[0] + rows[1];
        let cut = boundary
            .filter(|&b| b > position && b < position + total)
            .map(|b| b - position);
        Self { rows, cut }
    }

    pub fn total(&self) -> usize {
        self.rows[0] + self.rows[1]
    }

    pub fn site(&self) -> CutSite {
        match self.cut {
            None => CutSite::None,
            Some(cut) if cut == self.rows[0] => CutSite::Split,
            Some(cut) if cut < self.rows[0] => CutSite::Bank { bank: 0, row: cut },
            Some(cut) => CutSite::Bank {
                bank: 1,
                row: cut - self.rows[0],
            },
        }
    }

    /// Group-local row ranges of the sequential chunks the numerical twin runs:
    /// the banks, further split at the cut (a sub-launch is a chunk boundary).
    pub fn pieces(&self) -> Vec<std::ops::Range<usize>> {
        let mut points = vec![0, self.rows[0], self.total()];
        points.extend(self.cut);
        points.sort_unstable();
        points.dedup();
        points.windows(2).map(|w| w[0]..w[1]).collect()
    }
}

/// What one bank launches in a pair group: its place in the group, where its Core and PLE step split
/// at the session snapshot row (`cut`, bank-local), or whether its whole-bank state is staged after it
/// (`stage_after`: the cut is the A/B split). The single source of the launch geometry for every NPU mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankLaunch {
    pub offset: usize,
    pub rows: usize,
    pub cut: Option<usize>,
    pub stage_after: bool,
}

/// The two bank launches of a pair group, derived from its [`PairShape`].
pub fn launch_geometry(shape: PairShape) -> [BankLaunch; 2] {
    let site = shape.site();
    std::array::from_fn(|bank| BankLaunch {
        offset: if bank == 0 { 0 } else { shape.rows[0] },
        rows: shape.rows[bank],
        cut: match site {
            CutSite::Bank {
                bank: cut_bank,
                row,
            } if cut_bank == bank => Some(row),
            _ => None,
        },
        stage_after: bank == 0 && site == CutSite::Split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_sites_split_the_group_at_the_snapshot_row() {
        // A group of 6 + 4 rows starting at absolute row 10.
        let shape = |boundary| PairShape::new([6, 4], 10, Some(boundary));
        assert_eq!(shape(10).site(), CutSite::None);
        assert_eq!(shape(20).site(), CutSite::None);
        assert_eq!(PairShape::new([6, 4], 10, None).site(), CutSite::None);
        assert_eq!(shape(16).site(), CutSite::Split);
        assert_eq!(shape(13).site(), CutSite::Bank { bank: 0, row: 3 });
        assert_eq!(shape(18).site(), CutSite::Bank { bank: 1, row: 2 });
        assert_eq!(shape(13).pieces(), vec![0..3, 3..6, 6..10]);

        let [a, b] = launch_geometry(shape(18));
        assert_eq!(
            (a.offset, a.rows, a.cut, a.stage_after),
            (0, 6, None, false)
        );
        assert_eq!(
            (b.offset, b.rows, b.cut, b.stage_after),
            (6, 4, Some(2), false)
        );
        assert!(launch_geometry(shape(16))[0].stage_after);
    }
}
