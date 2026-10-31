//! Multi-rank boundary contract (#863): the ledger a rank's program leaves
//! at its cross-device boundaries, and the checks every rank's ledger must
//! pass together before their buffers may be reused.
//!
//! A boundary is publish-before-wait in one node: the rank releases its
//! earlier work and bumps its own monotonic epoch, then waits until every
//! rank has reached that epoch. A rank's `k`-th boundary therefore completes
//! only after every rank has executed everything before its own `k`-th
//! boundary. Events between boundaries `k` and `k + 1` run in epoch `k`.
//!
//! Peer resources are exchange buffers named by index, each rank owning its
//! own copy. A rank writes its copy in three shapes: [`Region::All`] (the
//! producer writes every row), [`Region::Unowned`] (the gather fills the rows
//! other ranks own) and reads, as a peer, only the rows a buffer's owner owns.
//! `Owned` and `Unowned` are disjoint by construction of the gather kernel.
//!
//! [`check`] proves, for a set of ledgers:
//! - matching: every rank crossed the same boundaries in the same order (the
//!   tags agree), so no rank waits on an epoch another rank never publishes
//!   and the progress order is acyclic;
//! - peer-read completion before overwrite: no rank writes rows of a buffer
//!   in the epoch in which a peer reads them; any other epoch is ordered by a
//!   boundary in between;
//! - no read before a producer write, and no read of a retired buffer.

use std::fmt;

/// Rows of one rank's copy of an exchange buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// Every row (the producer).
    All,
    /// Rows other ranks own (the gather's destination).
    Unowned,
}

/// One ledger entry, in the rank's program order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The rank wrote `region` of its own `buffer`.
    Write { buffer: usize, region: Region },
    /// A fused publish/wait boundary. `tag` identifies the crossing (kind,
    /// buffer, layer, rows, sequence) and must be equal on every rank.
    Boundary { tag: u64 },
    /// The rank read the owned rows of every peer's `buffer`.
    PeerRead { buffer: usize },
    /// The rank's `buffer` stops being readable (freed or repurposed).
    Retire { buffer: usize },
}

/// Why a set of ledgers is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// No ledgers, or more ranks than a boundary can name.
    Ranks(usize),
    /// Rank `rank` crossed `crossed` boundaries where rank 0 crossed `expected`.
    BoundaryCount {
        rank: usize,
        crossed: usize,
        expected: usize,
    },
    /// Boundary `index` carries different tags on rank 0 and `rank`.
    TagMismatch {
        rank: usize,
        index: usize,
        expected: u64,
        found: u64,
    },
    /// `writer` writes `buffer` in epoch `epoch` while `reader` reads it.
    OverwriteWhileRead {
        writer: usize,
        reader: usize,
        buffer: usize,
        epoch: usize,
    },
    /// `reader` reads `buffer` of `owner` before `owner` produced it.
    ReadBeforeWrite {
        reader: usize,
        owner: usize,
        buffer: usize,
        epoch: usize,
    },
    /// `reader` reads `buffer` of `owner` in or after the epoch it retired.
    ReadAfterRetire {
        reader: usize,
        owner: usize,
        buffer: usize,
        epoch: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ranks(n) => write!(f, "mesh ledger: {n} ranks (need 1..=8)"),
            Self::BoundaryCount {
                rank,
                crossed,
                expected,
            } => write!(
                f,
                "mesh ledger: rank {rank} crossed {crossed} boundaries, rank 0 crossed {expected}"
            ),
            Self::TagMismatch {
                rank,
                index,
                expected,
                found,
            } => write!(
                f,
                "mesh ledger: boundary {index} is {expected:#x} on rank 0 but {found:#x} on rank {rank}"
            ),
            Self::OverwriteWhileRead {
                writer,
                reader,
                buffer,
                epoch,
            } => write!(
                f,
                "mesh ledger: rank {writer} overwrites buffer {buffer} in epoch {epoch} while rank {reader} reads it"
            ),
            Self::ReadBeforeWrite {
                reader,
                owner,
                buffer,
                epoch,
            } => write!(
                f,
                "mesh ledger: rank {reader} reads buffer {buffer} of rank {owner} in epoch {epoch} before it was produced"
            ),
            Self::ReadAfterRetire {
                reader,
                owner,
                buffer,
                epoch,
            } => write!(
                f,
                "mesh ledger: rank {reader} reads buffer {buffer} of rank {owner} in epoch {epoch} after it was retired"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// What a matching ledger set certifies: the rank count and the boundary
/// sequence every rank runs, the multi-rank program identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshIdentity {
    pub ranks: usize,
    pub boundaries: Vec<u64>,
}

/// Check one step's (or one retained body's) ledgers, rank order.
pub fn check(ledgers: &[Vec<Event>]) -> Result<MeshIdentity, MeshError> {
    if !(1..=8).contains(&ledgers.len()) {
        return Err(MeshError::Ranks(ledgers.len()));
    }
    let tags = |ledger: &[Event]| -> Vec<u64> {
        ledger
            .iter()
            .filter_map(|event| match event {
                Event::Boundary { tag } => Some(*tag),
                _ => None,
            })
            .collect()
    };
    let boundaries = tags(&ledgers[0]);
    for (rank, ledger) in ledgers.iter().enumerate().skip(1) {
        let theirs = tags(ledger);
        if theirs.len() != boundaries.len() {
            return Err(MeshError::BoundaryCount {
                rank,
                crossed: theirs.len(),
                expected: boundaries.len(),
            });
        }
        if let Some(index) = (0..theirs.len()).find(|&i| theirs[i] != boundaries[i]) {
            return Err(MeshError::TagMismatch {
                rank,
                index,
                expected: boundaries[index],
                found: theirs[index],
            });
        }
    }

    // Per rank and buffer: epochs of producer writes, of any write that
    // touches owned rows (only `All`), and of the retirement.
    let mut produced: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ledgers.len()];
    let mut retired: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ledgers.len()];
    let mut reads: Vec<(usize, usize, usize)> = Vec::new();
    for (rank, ledger) in ledgers.iter().enumerate() {
        let mut epoch = 0;
        for event in ledger {
            match *event {
                Event::Boundary { .. } => epoch += 1,
                Event::Write {
                    buffer,
                    region: Region::All,
                } => produced[rank].push((buffer, epoch)),
                Event::Write {
                    region: Region::Unowned,
                    ..
                } => {}
                Event::PeerRead { buffer } => reads.push((rank, buffer, epoch)),
                Event::Retire { buffer } => retired[rank].push((buffer, epoch)),
            }
        }
    }
    for &(reader, buffer, epoch) in &reads {
        for owner in (0..ledgers.len()).filter(|&owner| owner != reader) {
            if produced[owner].contains(&(buffer, epoch)) {
                return Err(MeshError::OverwriteWhileRead {
                    writer: owner,
                    reader,
                    buffer,
                    epoch,
                });
            }
            if !produced[owner]
                .iter()
                .any(|&(b, written)| b == buffer && written < epoch)
            {
                return Err(MeshError::ReadBeforeWrite {
                    reader,
                    owner,
                    buffer,
                    epoch,
                });
            }
            if retired[owner]
                .iter()
                .any(|&(b, gone)| b == buffer && gone <= epoch)
            {
                return Err(MeshError::ReadAfterRetire {
                    reader,
                    owner,
                    buffer,
                    epoch,
                });
            }
        }
    }
    Ok(MeshIdentity {
        ranks: ledgers.len(),
        boundaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rank's ledger for gathers on `buffers` in order: produce, cross,
    /// read the peers, fill the unowned rows.
    fn gathers(buffers: &[usize]) -> Vec<Event> {
        buffers
            .iter()
            .enumerate()
            .flat_map(|(seq, &buffer)| {
                [
                    Event::Write {
                        buffer,
                        region: Region::All,
                    },
                    Event::Boundary {
                        tag: (seq as u64) << 8 | buffer as u64,
                    },
                    Event::PeerRead { buffer },
                    Event::Write {
                        buffer,
                        region: Region::Unowned,
                    },
                ]
            })
            .collect()
    }

    #[test]
    fn alternating_buffers_pass_on_every_rank_count() {
        // Trunk parity plus the MTP head's two buffers interleaved.
        let order = [0, 1, 0, 1, 2, 0, 3, 1, 2, 3, 0];
        for ranks in 2..=8 {
            let identity = check(&vec![gathers(&order); ranks]).unwrap();
            assert_eq!(identity.ranks, ranks);
            assert_eq!(identity.boundaries.len(), order.len());
        }
    }

    #[test]
    fn a_buffer_reused_by_consecutive_gathers_is_overwritten_while_read() {
        let ledgers = vec![gathers(&[0, 1, 1]); 2];
        assert_eq!(
            check(&ledgers),
            Err(MeshError::OverwriteWhileRead {
                writer: 1,
                reader: 0,
                buffer: 1,
                epoch: 2,
            })
        );
    }

    #[test]
    fn diverged_ranks_are_refused_by_count_and_by_tag() {
        let extra = vec![gathers(&[0, 1]), gathers(&[0, 1, 0])];
        assert!(matches!(
            check(&extra),
            Err(MeshError::BoundaryCount {
                rank: 1,
                crossed: 3,
                expected: 2
            })
        ));
        let other_buffer = vec![gathers(&[0, 1]), gathers(&[0, 2])];
        assert!(matches!(
            check(&other_buffer),
            Err(MeshError::TagMismatch {
                rank: 1,
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn a_read_needs_a_producer_write_in_an_earlier_epoch() {
        // Rank 1 never produced buffer 0 before the boundary.
        let mut late = gathers(&[0]);
        late.swap(0, 1);
        let ledgers = vec![gathers(&[0]), late];
        assert_eq!(
            check(&ledgers),
            Err(MeshError::OverwriteWhileRead {
                writer: 1,
                reader: 0,
                buffer: 0,
                epoch: 1,
            })
        );
        let mut missing = gathers(&[0]);
        missing.remove(0);
        assert!(matches!(
            check(&[gathers(&[0]), missing]),
            Err(MeshError::ReadBeforeWrite {
                reader: 0,
                owner: 1,
                ..
            })
        ));
    }

    #[test]
    fn retirement_is_safe_only_after_the_last_peer_read_epoch() {
        let mut early = gathers(&[0, 1]);
        // Retire buffer 1 before its own boundary: the read in epoch 2 sees it gone.
        early.insert(5, Event::Retire { buffer: 1 });
        assert!(matches!(
            check(&[gathers(&[0, 1]), early]),
            Err(MeshError::ReadAfterRetire {
                reader: 0,
                owner: 1,
                buffer: 1,
                ..
            })
        ));
        // After one more boundary every read of buffer 1 has completed.
        let mut tail = gathers(&[0, 1, 0]);
        tail.push(Event::Retire { buffer: 1 });
        assert!(check(&[gathers(&[0, 1, 0]), tail]).is_ok());
    }
}
