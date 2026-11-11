//! Block-local scalar abstract domain for static CFG edge proofs.
//!
//! Architecture-neutral: one [`Word`] per SGPR dword plus the SCC bit. Facts
//! describe *every* concrete state entering the current basic block, so they are
//! reset at each block boundary and only become more precise through an
//! implemented transfer. No opcode parsing, memory state or replay state lives
//! here; typed adapters decide which instructions they model and call
//! [`Facts::kill_defs`] for everything else.

use std::ops::Range;

/// Abstract 32-bit value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Word {
    Unknown,
    Const(u32),
}

/// Abstract boolean (SCC or a derived condition).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Truth {
    Unknown,
    False,
    True,
}

/// Statically proven feasible directions of a conditional branch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EdgeChoice {
    Both,
    Taken,
    Fallthrough,
}

impl Word {
    /// Bitwise AND: constants fold, zero is absorbing, all-ones is the identity.
    pub(crate) fn and(self, rhs: Word) -> Word {
        match (self, rhs) {
            (Word::Const(0), _) | (_, Word::Const(0)) => Word::Const(0),
            (Word::Const(a), Word::Const(b)) => Word::Const(a & b),
            (Word::Const(u32::MAX), x) | (x, Word::Const(u32::MAX)) => x,
            _ => Word::Unknown,
        }
    }

    /// Select by an abstract condition; an unknown condition keeps a value only
    /// when both alternatives are the same constant.
    pub(crate) fn select(condition: Truth, if_true: Word, if_false: Word) -> Word {
        match condition {
            Truth::True => if_true,
            Truth::False => if_false,
            Truth::Unknown => match (if_true, if_false) {
                (Word::Const(a), Word::Const(b)) if a == b => Word::Const(a),
                _ => Word::Unknown,
            },
        }
    }
}

impl Truth {
    /// Whether the multi-word value is nonzero: `True` if any word is a known
    /// nonzero constant, `False` if every word is known zero, else `Unknown`.
    pub(crate) fn nonzero(words: &[Word]) -> Truth {
        let mut all_zero = true;
        for w in words {
            match w {
                Word::Const(0) => {}
                Word::Const(_) => return Truth::True,
                Word::Unknown => all_zero = false,
            }
        }
        if all_zero {
            Truth::False
        } else {
            Truth::Unknown
        }
    }
}

/// Dense per-dword SGPR facts plus SCC for one block.
#[derive(Clone, Debug)]
pub(crate) struct Facts {
    sgprs: Box<[Word]>,
    pub(crate) scc: Truth,
}

impl Facts {
    /// `sgpr_count` is the architecture's addressable SGPR dword count, supplied
    /// by the adapter.
    pub(crate) fn new(sgpr_count: usize) -> Self {
        Self {
            sgprs: vec![Word::Unknown; sgpr_count].into_boxed_slice(),
            scc: Truth::Unknown,
        }
    }

    /// Forget everything (block entry); keeps the allocation.
    pub(crate) fn reset_block(&mut self) {
        self.sgprs.fill(Word::Unknown);
        self.scc = Truth::Unknown;
    }

    /// Out-of-range reads are `Unknown`.
    pub(crate) fn read32(&self, index: usize) -> Word {
        self.sgprs.get(index).copied().unwrap_or(Word::Unknown)
    }

    /// Two architectural dwords `[index, index + 1]`, low first.
    pub(crate) fn read64(&self, index: usize) -> [Word; 2] {
        [
            self.read32(index),
            index
                .checked_add(1)
                .map_or(Word::Unknown, |i| self.read32(i)),
        ]
    }

    /// Commits a result. Untracked (out-of-range) destinations are dropped.
    pub(crate) fn write32(&mut self, index: usize, value: Word) {
        if let Some(slot) = self.sgprs.get_mut(index) {
            *slot = value;
        }
    }

    pub(crate) fn write64(&mut self, index: usize, value: [Word; 2]) {
        self.write32(index, value[0]);
        if let Some(hi) = index.checked_add(1) {
            self.write32(hi, value[1]);
        }
    }

    /// Invalidates every tracked dword overlapping any range, and SCC if
    /// `writes_scc`. Callers read their inputs before calling this, then commit
    /// supported results with `write*`.
    pub(crate) fn kill_defs(
        &mut self,
        ranges: impl IntoIterator<Item = Range<usize>>,
        writes_scc: bool,
    ) {
        let len = self.sgprs.len();
        for r in ranges {
            let end = r.end.min(len);
            if r.start < end {
                self.sgprs[r.start..end].fill(Word::Unknown);
            }
        }
        if writes_scc {
            self.scc = Truth::Unknown;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Word::{Const, Unknown};

    #[test]
    fn dword_aliasing_and_high_overwrite() {
        let mut f = Facts::new(8);
        f.write64(0, [Const(1), Const(2)]);
        assert_eq!(f.read32(0), Const(1));
        assert_eq!(f.read32(1), Const(2));
        f.write32(1, Const(9));
        assert_eq!(f.read64(0), [Const(1), Const(9)]);
        f.write32(1, Unknown);
        assert_eq!(f.read64(0), [Const(1), Unknown]);
        f.write64(1, [Const(5), Const(6)]);
        assert_eq!(f.read64(0), [Const(1), Const(5)]);
        assert_eq!(f.read64(1), [Const(5), Const(6)]);
    }

    #[test]
    fn reset_forgets_everything() {
        let mut f = Facts::new(4);
        f.write64(2, [Const(1), Const(1)]);
        f.scc = Truth::True;
        f.reset_block();
        assert_eq!(f.read64(2), [Unknown; 2]);
        assert_eq!(f.scc, Truth::Unknown);
    }

    #[test]
    fn out_of_range_is_unknown_and_never_panics() {
        let mut f = Facts::new(2);
        assert_eq!(f.read32(2), Unknown);
        assert_eq!(f.read32(usize::MAX), Unknown);
        assert_eq!(f.read64(1), [Unknown, Unknown]);
        assert_eq!(f.read64(usize::MAX), [Unknown, Unknown]);
        f.write64(1, [Const(7), Const(8)]);
        assert_eq!(f.read64(1), [Const(7), Unknown]);
        f.write32(99, Const(1));
        f.write64(usize::MAX, [Const(1), Const(1)]);
        // Deliberately malformed span from an external footprint, not an iterator.
        let reversed = core::ops::Range { start: 3, end: 1 };
        f.kill_defs([5..9, usize::MAX - 1..usize::MAX, reversed], false);
        assert_eq!(f.read32(1), Const(7));
    }

    #[test]
    fn and_zero_absorbs_and_all_ones_is_identity() {
        assert_eq!(Const(0).and(Unknown), Const(0));
        assert_eq!(Unknown.and(Const(0)), Const(0));
        assert_eq!(Const(u32::MAX).and(Unknown), Unknown);
        assert_eq!(Unknown.and(Const(u32::MAX)), Unknown);
        assert_eq!(Const(u32::MAX).and(Const(0x1234)), Const(0x1234));
        assert_eq!(Const(0xF0F0).and(Const(0xFF00)), Const(0xF000));
        assert_eq!(Const(0xF0).and(Unknown), Unknown);
        assert_eq!(Unknown.and(Unknown), Unknown);
    }

    #[test]
    fn select_known_unknown_and_equal() {
        assert_eq!(Word::select(Truth::True, Const(1), Const(2)), Const(1));
        assert_eq!(Word::select(Truth::False, Const(1), Unknown), Unknown);
        assert_eq!(Word::select(Truth::False, Unknown, Const(2)), Const(2));
        assert_eq!(Word::select(Truth::Unknown, Const(3), Const(3)), Const(3));
        assert_eq!(Word::select(Truth::Unknown, Const(3), Const(4)), Unknown);
        assert_eq!(Word::select(Truth::Unknown, Unknown, Unknown), Unknown);
        assert_eq!(Word::select(Truth::Unknown, Const(3), Unknown), Unknown);
    }

    #[test]
    fn nonzero_truth() {
        assert_eq!(Truth::nonzero(&[Const(0), Const(0)]), Truth::False);
        assert_eq!(Truth::nonzero(&[Unknown, Const(4)]), Truth::True);
        assert_eq!(Truth::nonzero(&[Const(0), Unknown]), Truth::Unknown);
        assert_eq!(Truth::nonzero(&[Unknown]), Truth::Unknown);
    }

    #[test]
    fn kill_defs_spans_overlap_and_scc() {
        let mut f = Facts::new(8);
        for i in 0..8 {
            f.write32(i, Const(i as u32 + 1));
        }
        f.scc = Truth::True;
        f.kill_defs([1..3, 6..20], false);
        assert_eq!(f.scc, Truth::True);
        let got: Vec<_> = (0..8).map(|i| f.read32(i)).collect();
        assert_eq!(
            got,
            [
                Const(1),
                Unknown,
                Unknown,
                Const(4),
                Const(5),
                Const(6),
                Unknown,
                Unknown
            ]
        );
        // A one-dword def inside a pair kills the pair's fact only for that dword.
        f.write64(4, [Const(0), Const(0)]);
        f.kill_defs(std::iter::once(5..6), true);
        assert_eq!(f.read64(4), [Const(0), Unknown]);
        assert_eq!(f.scc, Truth::Unknown);
    }

    /// The L867 certificate: mov b64 s[0:1],0; and b64 with unknown EXEC;
    /// cselect; cmp_lg 1; no predecessor or EXEC fact needed.
    #[test]
    fn l867_certificate_expressible() {
        let mut f = Facts::new(16);
        f.write64(0, [Const(0), Const(0)]);
        let exec = [Unknown, Unknown];
        let a = f.read64(0);
        let r = [a[0].and(exec[0]), a[1].and(exec[1])];
        f.kill_defs([0..2], true);
        f.write64(0, r);
        f.scc = Truth::nonzero(&r);
        assert_eq!(f.scc, Truth::False);
        let cond = f.scc;
        let v = Word::select(cond, Const(1), Const(0));
        f.kill_defs([0..1], false);
        f.write32(0, v);
        assert_eq!(f.read32(0), Const(0));
        // cmp_lg s0, 1 -> scc = (0 != 1)
        f.scc = match f.read32(0) {
            Const(x) => {
                if x != 1 {
                    Truth::True
                } else {
                    Truth::False
                }
            }
            Unknown => Truth::Unknown,
        };
        assert_eq!(f.scc, Truth::True);
        // A kill of s0 without a modeled result leaves no constant live.
        f.kill_defs([0..1], false);
        assert_eq!(f.read32(0), Unknown);
    }
}
