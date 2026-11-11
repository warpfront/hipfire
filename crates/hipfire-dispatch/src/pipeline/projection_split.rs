// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Output-axis split of projections across ranks (#862).
//!
//! A rank thread may install a [`ProjectionSplit`] for the duration of a
//! forward ([`split_scope`]). Every projection entry point of the layer
//! operations then asks it, per weight, for this rank's band `[lo, hi)` of the
//! weight's output rows. A split call projects only those complete rows (the
//! same launch on a row view of the resident weight, so each output element
//! keeps its whole reduction) into the hook's band buffer, and the hook's
//! copy-only exchange writes every rank's band into the full `[rows, m]`
//! output before any consumer reads it. Weights the hook does not band, and
//! every thread without a hook, run unchanged.
//!
//! The hook is taken out of the thread for the duration of the split launch,
//! so nested entry points never exchange twice.

use crate::families::gemv::{weight_row_bytes, WeightRef};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::sync::Arc;

/// One gathered output of a split call: this rank's band of `rows` rows of
/// `band_bytes` sits band-compact at `offset` of the call's buffer; `dst` is
/// the full `[rows, ranks * band_bytes]` output.
pub struct SplitBand<'a> {
    pub offset: usize,
    pub band_bytes: usize,
    pub rows: usize,
    pub dst: &'a GpuTensor,
}

/// The rank-side split policy and transport.
pub trait ProjectionSplit {
    /// This rank's band `[lo, hi)` of `weight`'s output rows for a call of
    /// `rows` activation rows, or `None` to project the weight whole. Every
    /// rank must answer the same `hi - lo` for the same call.
    fn band(&self, gpu: &Gpu, weight: &WeightRef<'_>, rows: usize) -> Option<(usize, usize)>;
    /// The index of the next exchange's buffer and a non-owning view of it
    /// (the caller lays the call's bands out in it), or `None` without taking
    /// a buffer when `bytes` do not fit it.
    fn next_buffer(&self, bytes: usize) -> Option<(usize, GpuTensor)>;
    /// Publish this rank's bands of buffer `buffer`, wait for every peer's and
    /// copy all ranks' bands into the destinations.
    fn exchange(&self, gpu: &mut Gpu, buffer: usize, bands: &[SplitBand<'_>])
        -> Result<(), String>;
}

type Hook = Arc<dyn ProjectionSplit + Send + Sync>;

thread_local! {
    static SPLIT: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Restores the thread's previous hook on drop.
pub struct SplitScope(Option<Hook>);

impl Drop for SplitScope {
    fn drop(&mut self) {
        let previous = self.0.take();
        SPLIT.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Install `hook` on this thread until the returned scope drops.
pub fn split_scope(hook: Hook) -> SplitScope {
    SplitScope(SPLIT.with(|slot| slot.borrow_mut().replace(hook)))
}

/// Whether the installed hook bands `weight` for a `rows`-row call.
pub(crate) fn is_split(gpu: &Gpu, weight: &WeightRef<'_>, rows: usize) -> bool {
    SPLIT.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|hook| band_of(hook.as_ref(), gpu, weight, rows).is_some())
    })
}

/// A band the launch can take: equal bands of whole rows whose byte view
/// lies inside the weight's allocation. BF16 weights never band: their lazy
/// F16 shadows are keyed by the weight's device pointer, which the first band
/// shares with the whole matrix.
fn band_of(
    hook: &(dyn ProjectionSplit + Send + Sync),
    gpu: &Gpu,
    weight: &WeightRef<'_>,
    rows: usize,
) -> Option<(usize, usize, usize)> {
    if weight.dtype == rdna_compute::DType::BF16 {
        return None;
    }
    let (lo, hi) = hook.band(gpu, weight, rows)?;
    let row_bytes = weight_row_bytes(weight)?;
    let width = hi.checked_sub(lo)?;
    (width > 0 && hi <= weight.m && lo.is_multiple_of(width) && weight.m.is_multiple_of(width))
        .then_some((lo, hi, row_bytes))
}

/// Run `launch` over `projections` (each a weight and its full `[rows, m]`
/// output), with every weight the installed hook bands replaced by its row
/// view projecting into the hook's band buffer, then exchange the bands into
/// the full outputs. Without a hook, or with no banded weight, `launch` sees
/// `projections` unchanged.
pub(crate) fn split_projections(
    gpu: &mut Gpu,
    rows: usize,
    projections: &[(&WeightRef<'_>, &GpuTensor)],
    launch: impl FnOnce(&mut Gpu, &[(&WeightRef<'_>, &GpuTensor)]) -> Result<(), DispatchError>,
) -> Result<(), DispatchError> {
    let Some(hook) = SPLIT.with(|slot| slot.borrow().clone()) else {
        return launch(gpu, projections);
    };
    let bands: SmallVec<[Option<(usize, usize, usize)>; 4]> = projections
        .iter()
        .map(|(weight, _)| band_of(hook.as_ref(), gpu, weight, rows))
        .collect();
    if bands.iter().all(Option::is_none) {
        return launch(gpu, projections);
    }
    // Every band 256-byte aligned for the GEMM epilogues' vector stores.
    let slot = |band: &(usize, usize, usize), output: &GpuTensor| {
        (rows * (band.1 - band.0) * output.dtype.size()).next_multiple_of(256)
    };
    let need: usize = projections
        .iter()
        .zip(&bands)
        .filter_map(|((_, output), band)| band.as_ref().map(|band| slot(band, output)))
        .sum();
    // A call larger than the exchange buffer (an all-rows prefill LM head)
    // projects whole; the decision is a function of the call's shape, so
    // every rank takes it.
    let Some((buffer, staging)) = hook.next_buffer(need) else {
        return launch(gpu, projections);
    };
    let mut offset = 0usize;
    let mut views: SmallVec<[GpuTensor; 4]> = SmallVec::new();
    let mut outputs: SmallVec<[GpuTensor; 4]> = SmallVec::new();
    let mut gathered: SmallVec<[(usize, usize, usize); 4]> = SmallVec::new();
    let mut weights: SmallVec<[WeightRef<'_>; 4]> = SmallVec::new();
    for ((weight, output), band) in projections.iter().zip(&bands) {
        let Some((lo, hi, row_bytes)) = *band else {
            weights.push(**weight);
            outputs.push(output.shallow_clone());
            continue;
        };
        let m = hi - lo;
        let elem = output.dtype.size();
        let band_bytes = m * elem;
        let bytes = rows * band_bytes;
        if weight.m * elem * rows > output.buf.size() {
            return Err(DispatchError::Hip(format!(
                "projection split: the {rows} x {} output holds {} B",
                weight.m,
                output.buf.size()
            )));
        }
        // SAFETY: whole rows `[lo, hi)` of the resident weight, and `bytes` at
        // `offset` of the staging buffer (`need` fit it); non-owning views.
        let view = unsafe {
            GpuTensor {
                buf: hip_bridge::DeviceBuffer::from_raw(
                    (weight.buf.buf.as_ptr() as *mut u8).add(lo * row_bytes) as *mut _,
                    m * row_bytes,
                ),
                shape: vec![m, weight.k],
                dtype: weight.buf.dtype,
            }
        };
        let out = unsafe {
            GpuTensor {
                buf: hip_bridge::DeviceBuffer::from_raw(
                    (staging.buf.as_ptr() as *mut u8).add(offset) as *mut _,
                    bytes,
                ),
                shape: vec![rows * m],
                dtype: output.dtype,
            }
        };
        views.push(view);
        outputs.push(out);
        gathered.push((weights.len(), offset, band_bytes));
        weights.push(**weight);
        offset += slot(&(lo, hi, row_bytes), output);
    }
    // Point each banded weight at its row view (the views outlive `weights`).
    let mut view = views.iter();
    for &(index, _, _) in &gathered {
        let rows_view = view.next().expect("one view per banded weight");
        let (lo, hi, _) = bands[index].expect("banded");
        weights[index].buf = rows_view;
        weights[index].m = hi - lo;
    }
    let substituted: SmallVec<[(&WeightRef<'_>, &GpuTensor); 4]> =
        weights.iter().zip(outputs.iter()).collect();
    let previous = SPLIT.with(|slot| slot.borrow_mut().take());
    let launched = launch(gpu, &substituted);
    SPLIT.with(|slot| *slot.borrow_mut() = previous);
    launched?;
    let exchange: SmallVec<[SplitBand<'_>; 4]> = gathered
        .iter()
        .map(|&(index, offset, band_bytes)| SplitBand {
            offset,
            band_bytes,
            rows,
            dst: projections[index].1,
        })
        .collect();
    hook.exchange(gpu, buffer, &exchange)
        .map_err(DispatchError::Hip)
}
