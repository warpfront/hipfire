//! Per-component digests of defined live bytes.
//!
//! The component layout mirrors what a family publishes through
//! `SessionState::snapshot_parts` (`hipfire-runtime/src/session_cache.rs`):
//! fixed parts (overwritten in place, digested whole) and row streams
//! (append-only rows, digested one row at a time so a difference names its
//! row). This crate holds host bytes only; the card half downloads each
//! `StatePart` / `RowStream` range into a [`HostState`].

use serde::Serialize;

/// FNV-1a 64 (the digest `spec_floor_schedule_hw.rs` uses for Qwen3.5 state).
pub fn h64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Digest of `rows` rows of `row_bytes` from a row-major buffer, row `r` at
/// `index(r)`. Lifted from `spec_floor_schedule_hw.rs` for Qwen3.5 surfaces
/// whose rows are not stored in order (ring buffers).
pub fn rows_digest(
    bytes: &[u8],
    row_bytes: usize,
    rows: impl Iterator<Item = usize>,
    index: impl Fn(usize) -> usize,
) -> u64 {
    let mut h = 0u64;
    for r in rows {
        let at = index(r) * row_bytes;
        h = h.rotate_left(7) ^ h64(&bytes[at..at + row_bytes]);
    }
    h
}

/// A fixed part: bytes overwritten in place.
#[derive(Clone, Debug)]
pub struct HostPart {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Append-only rows from row 0. `owner` names who owns the stream's rows
/// (a lane, a layer); it is reported with a first difference.
#[derive(Clone, Debug)]
pub struct HostRows {
    pub name: String,
    pub owner: String,
    pub row_bytes: usize,
    pub bytes: Vec<u8>,
}

impl HostRows {
    pub fn rows(&self) -> usize {
        if self.row_bytes == 0 {
            0
        } else {
            self.bytes.len() / self.row_bytes
        }
    }
}

/// Host copy of a family's live state, in the family's fixed order.
#[derive(Clone, Debug, Default)]
pub struct HostState {
    pub meta: Vec<u8>,
    pub fixed: Vec<HostPart>,
    pub rows: Vec<HostRows>,
}

/// Digests of one state, same shape as the state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StateDigest {
    pub meta: u64,
    pub fixed: Vec<(String, u64)>,
    pub streams: Vec<StreamDigest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StreamDigest {
    pub name: String,
    pub owner: String,
    pub rows: Vec<u64>,
}

/// The first place two states differ.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FirstDifference {
    /// Step of the input schedule (filled by the runner; 0 for a bare diff).
    pub step: usize,
    pub component: String,
    /// Row within a stream; `None` for meta, a fixed part, or a row-count change.
    pub row: Option<usize>,
    /// Owning lane or component.
    pub owner: String,
    pub detail: String,
}

impl HostState {
    pub fn digest(&self) -> StateDigest {
        StateDigest {
            meta: h64(&self.meta),
            fixed: self.fixed.iter().map(|p| (p.name.clone(), h64(&p.bytes))).collect(),
            streams: self
                .rows
                .iter()
                .map(|s| StreamDigest {
                    name: s.name.clone(),
                    owner: s.owner.clone(),
                    rows: if s.row_bytes == 0 {
                        Vec::new()
                    } else {
                        s.bytes.chunks_exact(s.row_bytes).map(h64).collect()
                    },
                })
                .collect(),
        }
    }
}

/// First difference between two digests, in layout order: meta, fixed parts,
/// then streams by row. `None` when equal.
pub fn first_difference(a: &StateDigest, b: &StateDigest) -> Option<FirstDifference> {
    let mk = |component: &str, row, owner: &str, detail: String| FirstDifference {
        step: 0,
        component: component.to_string(),
        row,
        owner: owner.to_string(),
        detail,
    };
    if a.meta != b.meta {
        return Some(mk("meta", None, "meta", "host metadata differs".into()));
    }
    if a.fixed.len() != b.fixed.len() {
        return Some(mk("layout", None, "layout", "fixed part count differs".into()));
    }
    for (x, y) in a.fixed.iter().zip(&b.fixed) {
        if x.0 != y.0 {
            return Some(mk(&x.0, None, &x.0, format!("fixed part order: {} vs {}", x.0, y.0)));
        }
        if x.1 != y.1 {
            return Some(mk(&x.0, None, &x.0, "fixed part bytes differ".into()));
        }
    }
    if a.streams.len() != b.streams.len() {
        return Some(mk("layout", None, "layout", "row stream count differs".into()));
    }
    for (x, y) in a.streams.iter().zip(&b.streams) {
        if x.name != y.name {
            return Some(mk(&x.name, None, &x.owner, format!("stream order: {} vs {}", x.name, y.name)));
        }
        let n = x.rows.len().min(y.rows.len());
        for r in 0..n {
            if x.rows[r] != y.rows[r] {
                return Some(mk(&x.name, Some(r), &x.owner, "row bytes differ".into()));
            }
        }
        if x.rows.len() != y.rows.len() {
            return Some(mk(
                &x.name,
                None,
                &x.owner,
                format!("row count {} vs {}", x.rows.len(), y.rows.len()),
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(v: u8) -> HostState {
        HostState {
            meta: vec![1],
            fixed: vec![HostPart { name: "rec".into(), bytes: vec![v; 4] }],
            rows: vec![HostRows {
                name: "kv.lane0".into(),
                owner: "lane0".into(),
                row_bytes: 2,
                bytes: vec![1, 2, 3, 4, 5, 6],
            }],
        }
    }

    #[test]
    fn equal_states_have_no_difference() {
        assert!(first_difference(&state(7).digest(), &state(7).digest()).is_none());
    }

    #[test]
    fn row_difference_names_row_and_owner() {
        let mut b = state(7);
        b.rows[0].bytes[4] ^= 1;
        let d = first_difference(&state(7).digest(), &b.digest()).unwrap();
        assert_eq!((d.component.as_str(), d.row, d.owner.as_str()), ("kv.lane0", Some(2), "lane0"));
    }

    #[test]
    fn fixed_part_difference_comes_first() {
        let mut b = state(8);
        b.rows[0].bytes[0] ^= 1;
        let d = first_difference(&state(7).digest(), &b.digest()).unwrap();
        assert_eq!(d.component, "rec");
    }

    #[test]
    fn rows_digest_follows_index() {
        let bytes = [1u8, 2, 3, 4];
        let fwd = rows_digest(&bytes, 2, 0..2, |r| r);
        let rev = rows_digest(&bytes, 2, 0..2, |r| 1 - r);
        assert_ne!(fwd, rev);
    }
}
