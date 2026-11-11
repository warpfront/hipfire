// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The plug-in contract for the stream-carry oracles (CHUNK-CANONICAL-DESIGN
//! §3.2/§3.5/§4.1, §6 S0 kill 4).
//!
//! [`GdnStreamUnderTest`] is the typed hook later slices implement:
//!
//! * S2 (stream kernel + compute binding) wraps its edge-scan entries in an
//!   implementation to run the kills against the real device stream;
//! * S3/S4 (Qwen35 / Qwen4 owners) wrap `DeltaNetState`/`GdnGpuState` carry
//!   owners.
//!
//! The harness in [`super::matrix`] drives *any* implementation through the
//! same schedules (off-grid splits, snapshot → clobber → restore → continue,
//! final seal) and compares it, bit for bit, against the retained-input
//! reference [`super::model::RetainedReferenceStream`].
//!
//! Absolute-frame semantics (frozen by `rdna_compute::gdn_stream`):
//!
//! * Q8 / Q8Ef seams at absolute multiples of `commit_rows` (512), tiles at
//!   absolute multiples of `micro_rows`; at a genuine mode transition that
//!   begins at an off-grid `phase_start` the first tile / frame is clipped to
//!   the next absolute boundary;
//! * between seams the FP32 tile-entry state is authoritative, the Q8 plane is
//!   stale, and `pending_rows == 0` does NOT mean the Q8 plane is current;
//! * `seal` quantizes once at the actual prompt end (no-op when the last seam
//!   is exactly the end); nothing else (outer call end, snapshot, restore)
//!   quantizes.

use super::q8ef as q8;
use super::util::{bits_eq, count_diff, f32_bytes, fnv, R};
use rdna_compute::gdn_stream::{GdnStateQuant, GdnStreamMeta};
use std::any::Any;

pub const HD: usize = q8::HD;

#[derive(Clone, Copy, Debug)]
pub struct StreamGeometry {
    pub hk: usize,
    pub hv: usize,
    /// Scan microtile rows (64 for Scan64, 16 for Wy16, 1 sequential).
    pub micro_rows: usize,
    pub commit_rows: usize,
    pub conv_hist_rows: usize,
    pub state_quant: GdnStateQuant,
}

impl StreamGeometry {
    pub const fn qkv_width(&self) -> usize {
        (2 * self.hk + self.hv) * HD
    }
    pub const fn state_elems(&self) -> usize {
        self.hv * HD * HD
    }
    pub const fn out_width(&self) -> usize {
        self.hv * HD
    }
}

/// Raw post-projection, **pre-convolution** QKV and raw alpha/beta of `rows`
/// consecutive tokens (Z is not part of recurrent state).
#[derive(Clone, Debug)]
pub struct RawRows {
    pub rows: usize,
    /// `rows * qkv_width`
    pub qkv: Vec<f32>,
    /// `rows * hv`
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
}

impl RawRows {
    pub fn empty() -> Self {
        Self {
            rows: 0,
            qkv: vec![],
            alpha: vec![],
            beta: vec![],
        }
    }
    pub fn slice(&self, g: &StreamGeometry, a: usize, b: usize) -> RawRows {
        let w = g.qkv_width();
        Self {
            rows: b - a,
            qkv: self.qkv[a * w..b * w].to_vec(),
            alpha: self.alpha[a * g.hv..b * g.hv].to_vec(),
            beta: self.beta[a * g.hv..b * g.hv].to_vec(),
        }
    }
}

/// A sealed phase entry: Q8 (+EF) slot at absolute `position` plus the three
/// most recent raw QKV rows (conv history).
#[derive(Clone, Debug)]
pub struct InitialState {
    pub position: u64,
    pub live: q8::LiveState,
    /// `conv_hist_rows * qkv_width`
    pub conv: Vec<f32>,
}

/// The decoder-visible end state of a stream: Q8 codes/scales/EF + conv.
#[derive(Clone, Debug)]
pub struct SealedState {
    pub position: u64,
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
    pub ef: Vec<u16>,
    pub conv: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateDiff {
    pub position: bool,
    pub codes: usize,
    pub scales: usize,
    pub ef: usize,
    pub conv: usize,
}

impl StateDiff {
    pub fn is_empty(&self) -> bool {
        !self.position && self.codes == 0 && self.scales == 0 && self.ef == 0 && self.conv == 0
    }
}

impl SealedState {
    pub fn diff(&self, other: &SealedState) -> StateDiff {
        StateDiff {
            position: self.position != other.position,
            codes: count_diff(&self.codes, &other.codes),
            scales: if bits_eq(&self.scales, &other.scales) {
                0
            } else {
                self.scales
                    .iter()
                    .zip(&other.scales)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
                    .max(1)
            },
            ef: count_diff(&self.ef, &other.ef),
            conv: if bits_eq(&self.conv, &other.conv) {
                0
            } else {
                self.conv
                    .iter()
                    .zip(&other.conv)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
                    .max(1)
            },
        }
    }
    /// Semantic digest (padding-free by construction).
    pub fn digest(&self) -> String {
        let mut b: Vec<u8> = self.position.to_le_bytes().to_vec();
        b.extend(self.codes.iter().map(|&c| c as u8));
        b.extend(f32_bytes(&self.scales));
        for e in &self.ef {
            b.extend(e.to_le_bytes());
        }
        b.extend(f32_bytes(&self.conv));
        fnv(&b)
    }
}

/// Implementation-private snapshot payload (a deep copy of every meaningful
/// persistent field). Opaque to the harness.
pub struct OpaqueSnapshot(pub Box<dyn Any>);

/// A stream implementation under test. Single-owner, single-lane.
///
/// Contract:
/// * `append` publishes outputs only for the supplied rows, in order, as
///   `rows * hv * 128` f32 (BF16 stores widen exactly);
/// * `append` on a `Sealed` stream opens a phase at the current position
///   (`phase_start = position`);
/// * `peek_seal` returns what `seal` would produce **without** mutating the
///   stream (live state, carry and phase untouched);
/// * `snapshot` copies the complete persistent representation and must not
///   change the live numerical state (no forced boundary / split / seal);
/// * `clobber_live` destroys every live persistent buffer (recurrent, Q8/EF,
///   conv, carry) so a restore that forgets a field cannot pass by accident;
/// * `restore` is the atomic inverse of `snapshot` (an `Open` snapshot
///   restores `Open`).
pub trait GdnStreamUnderTest {
    fn label(&self) -> String;
    fn geometry(&self) -> StreamGeometry;
    /// `false` for implementations that have no stream metadata yet (current
    /// production quantizes at call end and is always "sealed"); the harness
    /// then skips the metadata comparison instead of faking it.
    fn reports_stream_meta(&self) -> bool {
        true
    }
    fn reset(&mut self, init: &InitialState) -> R<()>;
    fn meta(&self) -> GdnStreamMeta;
    fn append(&mut self, rows: &RawRows) -> R<Vec<f32>>;
    fn seal(&mut self) -> R<SealedState>;
    fn peek_seal(&mut self) -> R<SealedState>;
    fn snapshot(&mut self) -> R<OpaqueSnapshot>;
    fn restore(&mut self, snap: &OpaqueSnapshot) -> R<()>;
    fn clobber_live(&mut self) -> R<()>;
}
