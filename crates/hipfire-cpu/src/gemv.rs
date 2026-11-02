// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Host-side matmul core: GEMV / GEMM over quantized weight bytes that live in
//! system RAM.
//!
//! The activation contract is the GPU launcher's, not a new one: the MQ GEMV
//! kernels dot the *stored codes* against a **forward-rotated** activation
//! (`rotate_x`), because the weights were encoded post-rotation and
//! `dot(rot W, rot x) = dot(W, x)`. Callers therefore pass `x` already rotated
//! for the FWHT-rotated formats (`CpuQuant::is_fwht_g256`) and unrotated
//! otherwise — see [`crate::quant`] and [`crate::quant::decode_group_codes`].
//! Rotating it here instead would be wrong for the `Prerotated` inputs the
//! dispatch seam receives from the GPU, so the rotation belongs to whoever
//! knows which of the two it holds.
//!
//! Accumulation is plain `f32` in a fixed order (group-wise partial sums). It is
//! deliberately *not* the GPU kernel's 4-accumulator interleave: the contract for
//! the offload path is coherence, not bit-identity against a device kernel — see
//! the crate docs.

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;
#[cfg(target_arch = "x86_64")]
use std::collections::hash_map::Entry;
#[cfg(target_arch = "x86_64")]
use std::collections::HashMap;

use crate::quant::{decode_group_codes, CpuQuant, PLANAR_GROUP};
use crate::simd;

/// Largest group size over [`CpuQuant`] (256 for every G256 format). The decode
/// scratch is stack-local, so no allocation is on the inner loop.
const MAX_GROUP_ELEMS: usize = 256;

/// Bytes per weight row: `row_header_bytes + (k / group_elems) * group_bytes`.
///
/// G256 formats need `k % 256 == 0`, the 128-group formats `k % 128 == 0`
/// ([`CpuQuant::k_multiple`]); [`gemv`]/[`gemm`] assert it.
///
/// **Planar formats** ([`CpuQuant::is_planar`]) return the stride of one row in
/// the *codes plane* only (W3 `3k/8`, E8P `k/4`). That is NOT the tensor size
/// divided by `m`: the f16 scales live in a separate plane after all `m` code
/// rows. Whole-tensor sizes must use [`tensor_bytes`].
pub fn row_bytes(q: CpuQuant, k: usize) -> usize {
    q.row_header_bytes() + (k / q.group_elems()) * q.group_bytes()
}

/// Bytes of a whole `[m, k]` weight tensor: `m * row_bytes(q, k)` for the row
/// formats; for the planar formats the codes plane plus the scale plane,
/// `m * (row_bytes(q, k) + 2 * (k / 128))`.
pub fn tensor_bytes(q: CpuQuant, m: usize, k: usize) -> usize {
    if q.is_planar() {
        m * (row_bytes(q, k) + 2 * (k / PLANAR_GROUP))
    } else {
        m * row_bytes(q, k)
    }
}

/// Row `r` of an `[m, k]` tensor as `(codes_row, scales_row)`.
///
/// Row formats: the row's bytes run to the end of `packed` (kernels may read
/// the bytes right up to the next row, as before) and `scales_row` is empty.
/// Planar formats: `codes_row` is exactly the row's `row_bytes(q, k)` codes and
/// `scales_row` exactly its `2 * (k / 128)` scale bytes, so a kernel that stays
/// inside the two slices never reads a neighbouring row or the other plane.
/// `packed` must hold at least [`tensor_bytes`].
#[inline]
fn row_slices(q: CpuQuant, packed: &[u8], m: usize, k: usize, r: usize) -> (&[u8], &[u8]) {
    let rb = row_bytes(q, k);
    if q.is_planar() {
        let spr = 2 * (k / PLANAR_GROUP);
        let at = m * rb + r * spr;
        (&packed[r * rb..(r + 1) * rb], &packed[at..at + spr])
    } else {
        (&packed[r * rb..], &[])
    }
}

/// `y[0..m] = W[m,k] · x[0..k]`.
///
/// `packed` is the whole weight tensor's bytes, [`tensor_bytes`]`(q, m, k)` of them.
/// `x` carries the format's rotation (see the module docs).
pub fn gemv(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32], y: &mut [f32]) {
    gemv_with_simd(q, packed, m, k, x, y, None)
}

/// [`gemv`] with the SIMD decision forced (`Some(false)` = scalar, `None` =
/// runtime detection). The vector path is a throughput choice with a tolerance
/// contract, so the scalar path stays reachable and exact-testable, and
/// `simd::tests` compares the two.
pub fn gemv_with_simd(
    q: CpuQuant,
    packed: &[u8],
    m: usize,
    k: usize,
    x: &[f32],
    y: &mut [f32],
    requested: Option<bool>,
) {
    if m == 0 || k == 0 {
        return;
    }
    let rb = row_bytes(q, k);
    let tb = tensor_bytes(q, m, k);
    assert!(
        k % q.k_multiple() == 0,
        "gemv({q:?}): k={k} is not a multiple of {} (shape {m}x{k})",
        q.k_multiple()
    );
    assert!(
        x.len() >= k,
        "gemv({q:?}): x has {} elements, need k={k}",
        x.len()
    );
    assert!(
        y.len() >= m,
        "gemv({q:?}): y has {} elements, need m={m}",
        y.len()
    );
    assert!(
        packed.len() >= tb,
        "gemv({q:?}): packed has {} bytes, need {tb} for shape {m}x{k} (row stride {rb})",
        packed.len()
    );
    let x = &x[..k];
    // Resolve the vector/scalar decision once for the whole call rather than
    // re-detecting the CPU feature per row (`m` is thousands on every real
    // shape).
    let use_simd = simd::row_dot_enabled(q, requested);
    y[..m].par_iter_mut().enumerate().for_each(|(row, out)| {
        let (codes, scales) = row_slices(q, packed, m, k, row);
        *out = dot_row_simd(q, codes, scales, k, x, use_simd);
    });
}

/// Per-row GEMV over `n` activation rows:
/// `out[row*m .. row*m+m] = W · x[row*k .. row*k+k]`.
///
/// Parallel over every (row, output) pair. `n == 1` is [`gemv`]'s shape with the
/// same work split, so a single-token batch does not serialize.
pub fn gemm(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32], n: usize, out: &mut [f32]) {
    gemm_with_simd(q, packed, m, k, x, n, out, None)
}

/// [`gemm`] with the SIMD decision forced; see [`gemv_with_simd`].
pub fn gemm_with_simd(
    q: CpuQuant,
    packed: &[u8],
    m: usize,
    k: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
    requested: Option<bool>,
) {
    if n == 0 || m == 0 || k == 0 {
        return;
    }
    let rb = row_bytes(q, k);
    let tb = tensor_bytes(q, m, k);
    assert!(
        k % q.k_multiple() == 0,
        "gemm({q:?}): k={k} is not a multiple of {} (shape {n}x{m}x{k})",
        q.k_multiple()
    );
    assert!(
        x.len() >= n * k,
        "gemm({q:?}): x has {} elements, need n*k={}",
        x.len(),
        n * k
    );
    assert!(
        out.len() >= n * m,
        "gemm({q:?}): out has {} elements, need n*m={}",
        out.len(),
        n * m
    );
    assert!(
        packed.len() >= tb,
        "gemm({q:?}): packed has {} bytes, need {tb} for shape {m}x{k} (row stride {rb})",
        packed.len()
    );
    let x = &x[..n * k];
    let use_simd = simd::row_dot_enabled(q, requested);
    out[..n * m]
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat, o)| {
            let (token, row) = (flat / m, flat % m);
            let (codes, scales) = row_slices(q, packed, m, k, row);
            *o = dot_row_simd(q, codes, scales, k, &x[token * k..], use_simd);
        });
}

/// Random-access view of a batch of shared-weight jobs sharing one `(m, k)`.
///
/// A layer's routed-expert CPU FFN issues exactly two such batches — every
/// selected expert's gate/up projection, then every expert's down projection —
/// and [`gemv_shared_sourced`] runs one batch's jobs in a single rayon region.
/// It is a trait rather than a materialized `&[Job]` slice because a job's
/// inputs borrow the caller's buffers (`packed` from the weight owner, the
/// activation rows from its scratch) and its output is a disjoint mutable
/// sub-range of one caller buffer: building that as a slice would force a
/// per-call heap allocation, and holding it in a thread-local is not expressible
/// with safe lifetimes. The caller keeps its *index* metadata (job boundaries,
/// slot→expert and slot→activation maps) in reusable storage and answers these
/// reads instead.
///
/// Each method must be pure and cheap: `n`/`quant`/`packed`/`xs_row` are called
/// once per job (plus once per activation row for the up-front validation).
pub trait SharedJobSource {
    /// Number of jobs in the batch.
    fn jobs(&self) -> usize;
    /// Activation rows for job `j` (`m * n(j)` outputs are written for it).
    fn n(&self, j: usize) -> usize;
    /// Weight format of job `j`. Jobs may differ — a graded layer's MQ6 / MQ4 /
    /// MQ3-Lloyd experts carry different tiers.
    fn quant(&self, j: usize) -> CpuQuant;
    /// Job `j`'s `m` weight rows of `k` in `quant(j)`'s packed layout.
    fn packed(&self, j: usize) -> &[u8];
    /// Job `j`'s activation row `t` (at least `k` long, already carrying the
    /// format's rotation — see the module docs).
    fn xs_row(&self, j: usize, t: usize) -> &[f32];
}

/// Upper bound on the jobs [`gemv_shared_sourced`] splits into a stack array of
/// output chunks before falling back to a heap `Vec`. Decode (1 row) and the
/// ≤4-row verify window keep a layer's job count (one per distinct selected
/// expert) far below this; a wider batch keeps working through the fallback.
const MAX_STACK_JOBS: usize = 256;

/// Prep/exec split for one [`gemv_shared_sourced`] call, in nanoseconds.
/// `prep_ns` covers validation, output-chunk splitting, and the x86_64
/// activation-sum preparation; `exec_ns` covers the Rayon decode-and-dot
/// region (`run_shared_jobs`). `total_ns` is the whole call (`prep + exec`,
/// measured as one outer span so the parts reconcile against it).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SharedSourcedSplit {
    pub prep_ns: u64,
    pub exec_ns: u64,
    pub total_ns: u64,
}
/// Batched shared-weight projection: job `j` is `m` weight rows of `k` applied
/// to each of its `n(j)` activation rows, writing `m * n(j)` outputs
/// **row-major** (`out[r * n + t]`), the jobs concatenated in order in `out`.
///
/// The row-major layout is the point of the job. A routed expert selected by
/// several tokens is one weight row read from memory and applied to `n`
/// activations, so the row's bytes stay hot across the job's tokens: the row
/// loop is outer, the token loop inner — the same work split this had when the
/// jobs were a materialized slice. (The *decode* is still per output element,
/// exactly as [`gemv`] does it — the win is a single memory read of the weight
/// row, not a shared decode; `use_simd` is resolved once per job, not per
/// element.) Each element is the same [`dot_row_simd`] call [`gemv`] makes, so a
/// job with one activation row is bit-identical to [`gemv`], and a job with
/// several is bit-identical to calling [`gemv`] once per activation row — the
/// per-element accumulation order does not depend on `n`.
///
/// Parallel over jobs, and within a job over its weight rows; the nesting is
/// deliberate, so a layer whose selected experts are all distinct still fills
/// the pool per job, while a single expert shared by every token parallelizes
/// over that expert's rows instead.
pub fn gemv_shared_sourced<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    out: &mut [f32],
    src: &S,
    requested: Option<bool>,
) {
    let _ = gemv_shared_sourced_split(m, k, out, src, requested);
}

/// [`gemv_shared_sourced`] with its prep-vs-Rayonal-exec split returned.
/// Same validation, chunk split, sum-prep, and `run_shared_jobs` call as the
/// untraced entry point — only two extra host timestamp reads at the Rayon
/// region boundary. Bit-identical outputs.
pub fn gemv_shared_sourced_traced<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    out: &mut [f32],
    src: &S,
    requested: Option<bool>,
) -> SharedSourcedSplit {
    gemv_shared_sourced_split(m, k, out, src, requested)
}

fn gemv_shared_sourced_split<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    out: &mut [f32],
    src: &S,
    requested: Option<bool>,
) -> SharedSourcedSplit {
    let t_total = std::time::Instant::now();
    let jobs = src.jobs();
    if m == 0 || k == 0 || jobs == 0 {
        return SharedSourcedSplit::default();
    }
    // `k` granularity is per format (a 128-group format such as qt 53 takes any
    // multiple of 128), so it is asserted per job below.
    // Validate once per job, not per output element (the inner loop is m × n).
    let mut total = 0usize;
    for j in 0..jobs {
        let n = src.n(j);
        let q = src.quant(j);
        let rb = row_bytes(q, k);
        let tb = tensor_bytes(q, m, k);
        assert!(
            k % q.k_multiple() == 0,
            "gemv_shared_sourced({q:?}): k={k} is not a multiple of {}",
            q.k_multiple()
        );
        let packed = src.packed(j);
        assert!(
            packed.len() >= tb,
            "gemv_shared_sourced({q:?}): weight has {} bytes, need {tb} for shape {m}x{k} (row stride {rb})",
            packed.len()
        );
        for t in 0..n {
            let x = src.xs_row(j, t);
            assert!(
                x.len() >= k,
                "gemv_shared_sourced({q:?}): activation has {} elements, need k={k}",
                x.len()
            );
        }
        total += m * n;
    }
    assert!(
        out.len() >= total,
        "gemv_shared_sourced: out has {} elements, need {total}",
        out.len()
    );
    // Split `out` into one row-major chunk per job without a heap allocation:
    // the chunks live in a fixed-capacity stack array, with the `Vec` fallback
    // reached only by a batch wider than MAX_STACK_JOBS.
    let mut stack: [Option<&mut [f32]>; MAX_STACK_JOBS] = [const { None }; MAX_STACK_JOBS];
    let mut heap: Vec<Option<&mut [f32]>>;
    let chunks: &mut [Option<&mut [f32]>] = if jobs <= MAX_STACK_JOBS {
        let mut rest: &mut [f32] = out;
        for j in 0..jobs {
            let (chunk, tail) = rest.split_at_mut(m * src.n(j));
            stack[j] = Some(chunk);
            rest = tail;
        }
        &mut stack[..jobs]
    } else {
        heap = Vec::with_capacity(jobs);
        let mut rest: &mut [f32] = out;
        for j in 0..jobs {
            let (chunk, tail) = rest.split_at_mut(m * src.n(j));
            heap.push(Some(chunk));
            rest = tail;
        }
        &mut heap
    };
    // Prepare each unique eligible activation row's Σx once, before the rayon
    // region; the region then consumes immutable views of that scratch (never
    // the `RefCell`/map itself). On other arches the batched path is unchanged.
    #[cfg(target_arch = "x86_64")]
    {
        let split = {
            // Check the scratch out of the thread-local instead of holding a
            // `RefCell` borrow across the rayon region: `run_shared_jobs` blocks
            // this thread in a join, and rayon may steal another task onto it
            // that is itself a sourced GEMV (nested/concurrent callers on one
            // pool). That task finds an empty slot and prepares into its own
            // scratch, so the borrow is only ever held for the swap. The guard
            // returns the warm capacity on drop, including during unwind.
            // `&mut guard.0` is a plain reborrow so field-splitting (`lookup`
            // vs `sums`) stays visible to the borrow checker.
            let mut guard = CheckedOutSums::take();
            let cache = &mut guard.0;
            cache.reset();
            for j in 0..jobs {
                let n = src.n(j);
                let q = src.quant(j);
                // Match the batched region, which returns before resolving any
                // feature decision for an empty job: nothing is prepared and no
                // token window exists. The placeholder pushes keep the per-job
                // vectors positionally aligned with the job index (they are
                // never read for a job with no rows).
                if n == 0 {
                    cache.job_simd.push(false);
                    cache.token_base.push(cache.offsets.len());
                    continue;
                }
                // Resolved once per job, shared by the Σ prep and the row loop.
                let use_simd = simd::row_dot_enabled(q, requested);
                cache.job_simd.push(use_simd);
                let base = cache.offsets.len();
                cache.token_base.push(base);
                let kind = if use_simd {
                    simd::activation_sum_kind(q)
                } else {
                    None
                };
                for t in 0..n {
                    let off = match kind {
                        Some(kind) => {
                            let x = src.xs_row(j, t);
                            // Identity is meaningful only within this call — the
                            // table is cleared above — so a recycled activation
                            // buffer can never hit a value prepared for an
                            // earlier call, layer or request. GU's experts alias
                            // the same activation rows and dedup here; DN's
                            // per-expert rows are distinct addresses and do not.
                            let key = (x.as_ptr() as usize, k, kind);
                            match cache.lookup.entry(key) {
                                Entry::Occupied(e) => *e.get(),
                                Entry::Vacant(e) => {
                                    let groups = kind.len(k);
                                    let start = cache.sums.len();
                                    cache.sums.resize(start + groups, 0.0);
                                    simd::prepare_activation_sums(
                                        kind,
                                        &x[..k],
                                        &mut cache.sums[start..],
                                    );
                                    e.insert(start);
                                    start
                                }
                            }
                        }
                        None => usize::MAX,
                    };
                    cache.offsets.push(off);
                }
            }
            // No TLS borrow is held here (the scratch is checked out): a task
            // stolen onto this thread may re-enter. The workers only ever see
            // `&[…]` slices, so nothing shared-mutable is touched.
            // `prep` ends here: the `exec` span starts at the region boundary,
            // so the two are adjacent, not nested.
            let t_exec = std::time::Instant::now();
            run_shared_jobs(
                m,
                k,
                chunks,
                src,
                requested,
                &cache.sums,
                &cache.offsets,
                &cache.token_base,
                &cache.job_simd,
            );
            let exec_ns = t_exec.elapsed().as_nanos() as u64;
            let total_ns = t_total.elapsed().as_nanos() as u64;
            SharedSourcedSplit {
                prep_ns: total_ns.saturating_sub(exec_ns),
                exec_ns,
                total_ns,
            }
        };
        split
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let t_exec = std::time::Instant::now();
        run_shared_jobs(m, k, chunks, src, requested, &[], &[], &[], &[]);
        let exec_ns = t_exec.elapsed().as_nanos() as u64;
        let total_ns = t_total.elapsed().as_nanos() as u64;
        SharedSourcedSplit {
            prep_ns: total_ns.saturating_sub(exec_ns),
            exec_ns,
            total_ns,
        }
    }
}

/// Thread-local, capacity-reusing scratch for the batched GEMV's prepared
/// activation sums (x86_64 only — the vector row kernels are the only
/// consumer).
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
struct ActivationSumCache {
    /// `Σx` for every *unique* eligible activation row seen this call,
    /// `kind.len(k)` f32 each (`k / 256` for the MQ kinds, `k / 128` for the MQ4
    /// V2 kinds, `k / 8` word sums for E8P), concatenated in first-seen order.
    sums: Vec<f32>,
    /// Start index into `sums` per (job, token), flattened in job order;
    /// `usize::MAX` marks a token whose job is not prepared (scalar path or a
    /// format without an activation-sum kernel). `token_base[j]` locates job
    /// `j`'s window.
    offsets: Vec<usize>,
    /// First index into `offsets` for each job.
    token_base: Vec<usize>,
    /// The `use_simd` decision for each job, resolved once here so both the Σ
    /// prep and the row loop agree.
    job_simd: Vec<bool>,
    /// `(activation row ptr, k, kind)` → `sums` start, for intra-call
    /// deduplication. Cleared every call: pointer identity is only meaningful
    /// while the caller's buffers are unchanged, i.e. within one call.
    lookup: HashMap<(usize, usize, simd::ActivationSumKind), usize>,
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    static ACTIVATION_SUMS: RefCell<ActivationSumCache> =
        RefCell::new(ActivationSumCache::default());
}

/// Owns the thread's [`ActivationSumCache`] for the span of one sourced GEMV
/// call. The thread-local slot is left holding an empty default (no heap
/// allocation) so a reentrant call on the same thread never contends for it.
#[cfg(target_arch = "x86_64")]
struct CheckedOutSums(ActivationSumCache);

#[cfg(target_arch = "x86_64")]
impl CheckedOutSums {
    fn take() -> Self {
        Self(ACTIVATION_SUMS.with(|cell| std::mem::take(&mut *cell.borrow_mut())))
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for CheckedOutSums {
    fn drop(&mut self) {
        let owned = std::mem::take(&mut self.0);
        // `try_with`: the slot may already be destroyed at thread exit.
        let _ = ACTIVATION_SUMS.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut() {
                // A reentrant call may have parked its own scratch meanwhile;
                // keep whichever holds more capacity so warm calls stay warm.
                if owned.sums.capacity() >= slot.sums.capacity() {
                    *slot = owned;
                }
            }
        });
    }
}

#[cfg(target_arch = "x86_64")]
impl ActivationSumCache {
    /// Drop every logical entry while keeping the backing allocations, so a
    /// warm call adds no Σ data/map allocation.
    fn reset(&mut self) {
        self.sums.clear();
        self.offsets.clear();
        self.token_base.clear();
        self.job_simd.clear();
        self.lookup.clear();
    }
}

/// Balance subdivision overhead against indivisible row work, rather than
/// targeting a fixed number of blocks per worker. More outer jobs increase
/// the row budget per worker; more activation rows make each weight row
/// heavier and reduce the grain. All arguments are positive.
fn shared_rows_per_block(m: usize, n: usize, workers_per_job: usize) -> usize {
    let budget = m.div_ceil(workers_per_job).div_ceil(n).max(1);
    (budget.isqrt() * SHARED_GRAIN_SCALE).clamp(1, budget)
}

/// Multiplier on the square-root row grain. Measured on the 16-worker MQ4
/// expert arm (10 experts x 1 row): the plain square root splits a job into
/// blocks of a few microseconds, where task hand-off is a visible share of the
/// region; twice the grain trims the region time without leaving a tail longer
/// than a block.
const SHARED_GRAIN_SCALE: usize = 2;

/// The batched GEMV's per-job rayon region. On x86_64 `sums`/`offsets`/
/// `token_base`/`job_simd` are the prepared views (see [`ActivationSumCache`]);
/// on other arches they are empty and each job resolves its own `use_simd`,
/// exactly as before.
///
/// Nonempty output chunks supply the outer parallelism. The inner iterator
/// groups consecutive rows using a square-root grain of the per-worker row
/// budget; Rayon still decides how to split and steal those blocks. A
/// single-worker pool consumes each job in one block.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
fn run_shared_jobs<S: SharedJobSource + Sync>(
    m: usize,
    k: usize,
    chunks: &mut [Option<&mut [f32]>],
    src: &S,
    _requested: Option<bool>,
    sums: &[f32],
    offsets: &[usize],
    token_base: &[usize],
    job_simd: &[bool],
) {
    let active_jobs = chunks
        .iter()
        .filter(|slot| slot.as_ref().is_some_and(|out| !out.is_empty()))
        .count();
    if active_jobs == 0 {
        return;
    }
    let workers = rayon::current_num_threads();
    let workers_per_job = workers.div_ceil(active_jobs);
    chunks.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let Some(o) = slot.as_deref_mut() else {
            return;
        };
        let n = src.n(j);
        if n == 0 {
            return;
        }
        let rows_per_block = if workers == 1 {
            m
        } else {
            shared_rows_per_block(m, n, workers_per_job)
        };
        let q = src.quant(j);
        #[cfg(target_arch = "x86_64")]
        let use_simd = job_simd[j];
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = simd::row_dot_enabled(q, _requested);
        #[cfg(target_arch = "x86_64")]
        let prepared_kind = if use_simd {
            simd::activation_sum_kind(q)
        } else {
            None
        };
        #[cfg(target_arch = "x86_64")]
        let pair = prepared_kind.is_some() && simd::has_row_pair_kernel(q);
        o[..m * n]
            .par_chunks_mut(rows_per_block * n)
            .enumerate()
            .for_each(|(block, rows)| {
                let first_row = block * rows_per_block;
                #[cfg(target_arch = "x86_64")]
                if let Some(kind) = prepared_kind {
                    let base = token_base[j];
                    let sum_len = kind.len(k);
                    let sums_of = |t: usize| {
                        let off = offsets[base + t];
                        debug_assert_ne!(off, usize::MAX, "unprepared token in a prepared job");
                        &sums[off..off + sum_len]
                    };
                    let mut r = first_row;
                    let mut rest: &mut [f32] = rows;
                    if pair {
                        // Two rows at a time: each activation load is shared by
                        // both rows' FMAs; every element is bit-identical to the
                        // single-row kernel's.
                        while rest.len() >= 2 * n {
                            let (both, tail) = rest.split_at_mut(2 * n);
                            let (out0, out1) = both.split_at_mut(n);
                            let (row0, srow0) = row_slices(q, src.packed(j), m, k, r);
                            let (row1, srow1) = row_slices(q, src.packed(j), m, k, r + 1);
                            for t in 0..n {
                                let [v0, v1] = simd::row_pair_dot_with_activation_sums(
                                    q,
                                    row0,
                                    srow0,
                                    row1,
                                    srow1,
                                    k,
                                    &src.xs_row(j, t)[..k],
                                    sums_of(t),
                                );
                                out0[t] = v0;
                                out1[t] = v1;
                            }
                            rest = tail;
                            r += 2;
                        }
                    }
                    for chunk in rest.chunks_mut(n) {
                        let (row, srow) = row_slices(q, src.packed(j), m, k, r);
                        if n > 1 && simd::has_multi_token_kernel(q) {
                            multi_token_row(
                                q,
                                row,
                                k,
                                (src, j),
                                (sums, &offsets[base..base + n], sum_len),
                                chunk,
                            );
                            r += 1;
                            continue;
                        }
                        for (t, v) in chunk.iter_mut().enumerate() {
                            *v = simd::row_dot_with_activation_sums(
                                q,
                                row,
                                srow,
                                k,
                                &src.xs_row(j, t)[..k],
                                sums_of(t),
                            );
                        }
                        r += 1;
                    }
                    return;
                }
                for (within_block, chunk) in rows.chunks_mut(n).enumerate() {
                    let r = first_row + within_block;
                    let (row, srow) = row_slices(q, src.packed(j), m, k, r);
                    for (t, v) in chunk.iter_mut().enumerate() {
                        *v = dot_row_simd(q, row, srow, k, &src.xs_row(j, t)[..k], use_simd);
                    }
                }
            });
    });
}

/// One weight row against all `chunk.len()` activation rows of job `j` on the
/// prepared V2 path: each weight chunk is decoded once per group of up to
/// [`simd::MULTI_TOKEN_MAX`] rows instead of once per row
/// ([`simd::row_dot_multi_with_activation_sums`]). `out[t]` equals the
/// single-token prepared row dot bit for bit. `sum_offsets[t]` locates token
/// `t`'s prepared Σx in `sums`.
#[cfg(target_arch = "x86_64")]
fn multi_token_row<S: SharedJobSource>(
    q: CpuQuant,
    row: &[u8],
    k: usize,
    (src, j): (&S, usize),
    (sums, sum_offsets, sum_len): (&[f32], &[usize], usize),
    out: &mut [f32],
) {
    let mut t0 = 0;
    while t0 < out.len() {
        let g = (out.len() - t0).min(simd::MULTI_TOKEN_MAX);
        let mut xs: [&[f32]; simd::MULTI_TOKEN_MAX] = [&[]; simd::MULTI_TOKEN_MAX];
        let mut ss: [&[f32]; simd::MULTI_TOKEN_MAX] = [&[]; simd::MULTI_TOKEN_MAX];
        for i in 0..g {
            let off = sum_offsets[t0 + i];
            debug_assert_ne!(off, usize::MAX, "unprepared token in a prepared job");
            xs[i] = &src.xs_row(j, t0 + i)[..k];
            ss[i] = &sums[off..off + sum_len];
        }
        simd::row_dot_multi_with_activation_sums(
            q,
            row,
            k,
            &xs[..g],
            &ss[..g],
            &mut out[t0..t0 + g],
        );
        t0 += g;
    }
}

/// One output element: `Σ_j W[row][j] * x[j]`, accumulating one group at a time.
///
/// `row` / `scales_row` are [`row_slices`]'s pair: for the planar formats the
/// row's codes and its f16 scales, for the row formats the row's bytes and an
/// empty slice.
///
/// `use_simd` is resolved once per GEMV call by the caller
/// ([`simd::row_dot_enabled`]); every format has a vector kernel, so when it is
/// set the row is the kernel's and the scalar decode below is the ARM and
/// non-AVX2 fallback (and the reference `simd::tests` compares against).
fn dot_row_simd(
    q: CpuQuant,
    row: &[u8],
    scales_row: &[u8],
    k: usize,
    x: &[f32],
    use_simd: bool,
) -> f32 {
    if use_simd {
        return simd::row_dot_avx2(q, row, scales_row, k, x);
    }
    dot_row_scalar(q, row, scales_row, k, x)
}

/// The `Mq4G256` row dot on the scalar path — the SIMD fallback and the
/// reference `simd::tests` compares against.
pub(crate) fn mq4g256_row_dot_scalar(row: &[u8], k: usize, x: &[f32]) -> f32 {
    dot_row_scalar(CpuQuant::Mq4G256, row, &[], k, x)
}

fn dot_row_scalar(q: CpuQuant, row: &[u8], scales_row: &[u8], k: usize, x: &[f32]) -> f32 {
    let ge = q.group_elems();
    let gb = q.group_bytes();
    let mut scratch = [0.0f32; MAX_GROUP_ELEMS];
    let mut acc = 0.0f32;
    let planar = q.is_planar();
    let e8 = q == CpuQuant::Mfp4G32E8;
    let row_scale = if e8 {
        crate::quant::half_f16_to_f32(u16::from_le_bytes([row[0], row[1]]))
    } else {
        0.0
    };
    for g in 0..k / ge {
        let codes = &mut scratch[..ge];
        if planar {
            let d = u16::from_le_bytes([scales_row[2 * g], scales_row[2 * g + 1]]);
            crate::quant::decode_planar_group(q, &row[g * gb..(g + 1) * gb], d, codes);
        } else if e8 {
            crate::quant::decode_mfp4_e8_group(&row[16 + g * gb..], row_scale, codes);
        } else {
            decode_group_codes(q, &row[g * gb..], codes);
        }
        let xg = &x[g * ge..g * ge + ge];
        let mut partial = 0.0f32;
        for i in 0..ge {
            partial += codes[i] * xg[i];
        }
        acc += partial;
    }
    acc
}

#[cfg(test)]
mod test {
    use super::*;

    /// `[m, k]` weight bytes with distinct payloads per (row, group), so a wrong
    /// row stride or a wrong group offset cannot pass by symmetry.
    fn weights(q: CpuQuant, m: usize, k: usize) -> Vec<u8> {
        crate::testfix::weight_bytes(q, m, k)
    }

    /// Independent oracle in `f64`: decode the group with the crate's own
    /// decoder and accumulate in double precision. Exactness is not the claim —
    /// `1e-4` relative bounds the *f32 accumulation* this crate does.
    fn oracle(q: CpuQuant, packed: &[u8], m: usize, k: usize, x: &[f32]) -> Vec<f64> {
        let ge = q.group_elems();
        let gb = q.group_bytes();
        let rb = row_bytes(q, k);
        (0..m)
            .map(|row| {
                let mut acc = 0.0f64;
                for g in 0..k / ge {
                    let mut codes = vec![0.0f32; ge];
                    decode_group_codes(q, &packed[row * rb + g * gb..], &mut codes);
                    for i in 0..ge {
                        acc += codes[i] as f64 * x[g * ge + i] as f64;
                    }
                }
                acc
            })
            .collect()
    }

    fn x_of(k: usize) -> Vec<f32> {
        (0..k)
            .map(|i| {
                let v = (i as u64 * 2654435761) % 4096;
                (v as f32 - 2048.0) * 0.001_953_125
            })
            .collect()
    }

    /// `gemv_shared_sourced` must be bit-identical to calling `gemv` once per
    /// activation row, for every job — including jobs of different formats in
    /// one call (a graded layer's MQ6 / MQ4 / MQ3-Lloyd experts) and several
    /// activation rows against one weight (an expert selected by several
    /// verification rows). It is the routed-expert CPU path's replacement for a
    /// loop of per-expert GEMVs, and a rounding change there would flip greedy
    /// tokens.
    fn check_shared_gemv(m: usize, jobs: usize, requested: Option<bool>) {
        /// A batched source over plain slices — the same index/slice shape the
        /// CPU MoE splice builds from its reusable metadata.
        struct Batch<'a> {
            quants: &'a [CpuQuant],
            packs: &'a [&'a [u8]],
            xs: &'a [&'a [&'a [f32]]],
        }
        impl SharedJobSource for Batch<'_> {
            fn jobs(&self) -> usize {
                self.quants.len()
            }
            fn n(&self, j: usize) -> usize {
                self.xs[j].len()
            }
            fn quant(&self, j: usize) -> CpuQuant {
                self.quants[j]
            }
            fn packed(&self, j: usize) -> &[u8] {
                self.packs[j]
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                self.xs[j][t]
            }
        }

        let k = 512;
        let tiers = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq3G256Lloyd,
            CpuQuant::Mfp4G32E8,
            CpuQuant::Mq4G128V2,
        ];
        let quants: Vec<_> = (0..jobs).map(|j| tiers[j % tiers.len()]).collect();
        // Mixed tiers and token counts, including empty jobs between active
        // ones, exercise both prepared and unprepared row kernels.
        let ns: Vec<_> = (0..jobs).map(|j| [3usize, 1, 4, 2, 0][j % 5]).collect();
        let packed: Vec<Vec<u8>> = quants.iter().map(|&q| weights(q, m, k)).collect();
        let xs: Vec<Vec<Vec<f32>>> = ns
            .iter()
            .enumerate()
            .map(|(j, &n)| {
                (0..n)
                    .map(|t| x_of(k).iter().map(|v| v + (j * 10 + t) as f32).collect())
                    .collect()
            })
            .collect();
        let xs_refs: Vec<Vec<&[f32]>> = xs
            .iter()
            .map(|rows| rows.iter().map(|v| v.as_slice()).collect())
            .collect();
        let packs: Vec<&[u8]> = packed.iter().map(|p| p.as_slice()).collect();
        let xs_nested: Vec<&[&[f32]]> = xs_refs.iter().map(|r| r.as_slice()).collect();
        let total: usize = ns.iter().map(|n| n * m).sum();
        let src = Batch {
            quants: &quants,
            packs: &packs,
            xs: &xs_nested,
        };
        let mut out = vec![-999.0f32; total + 3];
        gemv_shared_sourced(m, k, &mut out, &src, requested);
        let mut reference = vec![0.0f32; total];
        let mut off = 0usize;
        for (j, &q) in quants.iter().enumerate() {
            let n = ns[j];
            // `gemv` produces one activation row at a time (token-major); the
            // batched output is row-major (`out[r * n + t]`), so transpose.
            let mut token_major = vec![0.0f32; n * m];
            for t in 0..n {
                gemv_with_simd(
                    q,
                    &packed[j],
                    m,
                    k,
                    &xs[j][t],
                    &mut token_major[t * m..(t + 1) * m],
                    requested,
                );
            }
            for r in 0..m {
                for t in 0..n {
                    reference[off + r * n + t] = token_major[t * m + r];
                }
            }
            off += n * m;
        }
        for (i, (got, want)) in out[..total].iter().zip(&reference).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "m={m} jobs={jobs} output {i}"
            );
        }
        assert_eq!(&out[total..], &[-999.0; 3], "output tail overwritten");
    }

    #[test]
    fn gemv_shared_matches_per_row_gemv() {
        for workers in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                for requested in [Some(false), None] {
                    for m in [1, 7, 65, 257] {
                        for jobs in [1, 4, 9] {
                            check_shared_gemv(m, jobs, requested);
                        }
                    }
                    // Exercise the output-chunk heap fallback, empty batches,
                    // and zero rows without allocating a large weight fixture.
                    check_shared_gemv(3, MAX_STACK_JOBS + 1, requested);
                    check_shared_gemv(3, 0, requested);
                    check_shared_gemv(0, 4, requested);
                }
            });
        }
    }

    /// The splice runs on a pool whose worker count is a host/knob property, and
    /// the same CPU row must produce identical bytes regardless of it: chunking
    /// only partitions whole rows (`shared_rows_per_block`), and every output
    /// element is one row dot accumulated in a fixed group order.
    #[test]
    fn gemv_shared_is_bitwise_identical_across_pool_sizes() {
        struct Batch<'a> {
            quants: &'a [CpuQuant],
            packs: &'a [Vec<u8>],
            xs: &'a [Vec<Vec<f32>>],
        }
        impl SharedJobSource for Batch<'_> {
            fn jobs(&self) -> usize {
                self.quants.len()
            }
            fn n(&self, j: usize) -> usize {
                self.xs[j].len()
            }
            fn quant(&self, j: usize) -> CpuQuant {
                self.quants[j]
            }
            fn packed(&self, j: usize) -> &[u8] {
                &self.packs[j]
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                &self.xs[j][t]
            }
        }
        let k = 512;
        let tiers = [
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq4G128V2,
            CpuQuant::Mq3G256Lloyd,
        ];
        for requested in [Some(false), None] {
            for (m, jobs) in [(1usize, 1usize), (5, 10), (257, 10), (640, 3)] {
                let quants: Vec<_> = (0..jobs).map(|j| tiers[j % tiers.len()]).collect();
                let ns: Vec<usize> = (0..jobs).map(|j| [1usize, 3, 2][j % 3]).collect();
                let packs: Vec<Vec<u8>> = quants.iter().map(|&q| weights(q, m, k)).collect();
                let xs: Vec<Vec<Vec<f32>>> = ns
                    .iter()
                    .enumerate()
                    .map(|(j, &n)| {
                        (0..n)
                            .map(|t| x_of(k).iter().map(|v| v + (j * 7 + t) as f32).collect())
                            .collect()
                    })
                    .collect();
                let src = Batch {
                    quants: &quants,
                    packs: &packs,
                    xs: &xs,
                };
                let total: usize = ns.iter().map(|n| n * m).sum();
                let mut baseline: Option<Vec<u32>> = None;
                for workers in [1usize, 3, 16] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap();
                    let bits: Vec<u32> = pool.install(|| {
                        let mut out = vec![0.0f32; total];
                        gemv_shared_sourced(m, k, &mut out, &src, requested);
                        out.iter().map(|v| v.to_bits()).collect()
                    });
                    match &baseline {
                        None => baseline = Some(bits),
                        Some(b) => assert_eq!(
                            &bits, b,
                            "m={m} jobs={jobs} requested={requested:?}: {workers} workers differ from 1"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn shared_row_grain_stays_within_worker_budget_without_overflow() {
        for m in [1, 7, 257, usize::MAX] {
            for n in [1, 4, usize::MAX] {
                for workers_per_job in [1, 2, 17, usize::MAX] {
                    let grain = shared_rows_per_block(m, n, workers_per_job);
                    assert!(
                        (1..=m.div_ceil(workers_per_job)).contains(&grain),
                        "m={m} n={n} workers_per_job={workers_per_job} grain={grain}"
                    );
                }
            }
        }
    }

    /// The batched path caches each unique activation row's Σx in a
    /// thread-local keyed by `(row pointer, k, reduction style)`, and that
    /// identity is only meaningful within one call. This drives two calls over
    /// one activation buffer whose contents are rewritten in place (same
    /// pointer, same capacity) and checks every element against an independent
    /// `gemv_with_simd` per (job, token).
    ///
    /// The fixture is built so each failure mode shows up: jobs of qt 13
    /// (`Mq4G256`, style `Mq4Quartets`) and qt 15 (`Mq6G256`, style `Mq6Pairs`)
    /// read the *same* activation row pointer (a key that ignored the style
    /// would alias them), a duplicate-format job revisits row 0 out of order
    /// (a wrong job/token offset), and one job has no rows (the empty-job
    /// placeholder must keep the per-job scratch aligned). Only the second call
    /// can catch a Σx that was not recomputed after the in-place value change.
    #[test]
    fn gemv_shared_recomputes_prepared_sums_for_a_reused_activation_buffer() {
        struct Batch<'a> {
            quants: &'a [CpuQuant],
            packs: &'a [&'a [u8]],
            xs: &'a [&'a [&'a [f32]]],
        }
        impl SharedJobSource for Batch<'_> {
            fn jobs(&self) -> usize {
                self.quants.len()
            }
            fn n(&self, j: usize) -> usize {
                self.xs[j].len()
            }
            fn quant(&self, j: usize) -> CpuQuant {
                self.quants[j]
            }
            fn packed(&self, j: usize) -> &[u8] {
                self.packs[j]
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                self.xs[j][t]
            }
        }

        let (m, k, tokens) = (5usize, 512usize, 3usize);
        // MQ4 has a different reduction style; MQ5/MQ6 share the `codes8`
        // style. The same activation buffer exercises both separation and
        // reuse, including a changed value at an unchanged address.
        let quants = [
            CpuQuant::Mq4G256,
            CpuQuant::Mq6G256,
            CpuQuant::Mq4G256,
            CpuQuant::Mq4G256,
        ];
        let packed: Vec<Vec<u8>> = quants.iter().map(|&q| weights(q, m, k)).collect();
        let packs: Vec<&[u8]> = packed.iter().map(|p| p.as_slice()).collect();
        // Token rows of the one shared buffer each (job, token) reads: job 1
        // reuses job 0's row 0 pointer under the other style, job 2 is empty,
        // job 3 (same format as job 0) revisits row 0 after row 2.
        let rows: [&[usize]; 4] = [&[0usize, 1], &[0], &[], &[2, 0]];

        let mut act: Vec<f32> = (0..tokens)
            .flat_map(|t| x_of(k).into_iter().map(move |v| v + t as f32))
            .collect();
        let cap = act.capacity();
        let ptr = act.as_ptr() as usize;

        let total: usize = rows.iter().map(|r| r.len() * m).sum();
        let mut out = vec![0.0f32; total];

        // `requested = None` is resolved identically by both entry points (the
        // shared path once per job, `gemv_with_simd` once per call), so the
        // exact comparison holds with or without the SIMD kernels.
        for pass in 0..2 {
            {
                let xs_refs: Vec<Vec<&[f32]>> = rows
                    .iter()
                    .map(|r| r.iter().map(|&tok| &act[tok * k..tok * k + k]).collect())
                    .collect();
                let xs_nested: Vec<&[&[f32]]> = xs_refs.iter().map(|r| r.as_slice()).collect();
                let src = Batch {
                    quants: &quants,
                    packs: &packs,
                    xs: &xs_nested,
                };
                gemv_shared_sourced(m, k, &mut out, &src, None);
            }
            let mut reference = vec![0.0f32; total];
            let mut base = 0usize;
            for (j, &q) in quants.iter().enumerate() {
                let n = rows[j].len();
                let mut token_major = vec![0.0f32; n * m];
                for t in 0..n {
                    let tok = rows[j][t];
                    gemv_with_simd(
                        q,
                        &packed[j],
                        m,
                        k,
                        &act[tok * k..tok * k + k],
                        &mut token_major[t * m..(t + 1) * m],
                        None,
                    );
                }
                for r in 0..m {
                    for t in 0..n {
                        reference[base + r * n + t] = token_major[t * m + r];
                    }
                }
                base += n * m;
            }
            assert_eq!(
                out, reference,
                "pass {pass}: batched Σx disagrees with independent gemv_with_simd"
            );
            // Rewrite the buffer in place: pointer and capacity must not change,
            // so only a recomputation can make the next pass correct.
            for v in act.iter_mut() {
                *v = *v * 0.5 - 0.017_578_125;
            }
            assert_eq!(act.capacity(), cap, "pass {pass}: buffer reallocated");
            assert_eq!(act.as_ptr() as usize, ptr, "pass {pass}: buffer moved");
        }
    }

    /// Serial, uncached single-row arithmetic of [`gemv_shared_sourced`]: every
    /// `(job, weight row, token)` element is one [`dot_row_simd`] call — what
    /// [`gemv`] computes for that activation row — in the batched row-major
    /// layout (`out[r * n + t]`, jobs concatenated).
    fn single_row_reference<S: SharedJobSource>(
        m: usize,
        k: usize,
        out: &mut [f32],
        src: &S,
        requested: Option<bool>,
    ) {
        let mut base = 0usize;
        for j in 0..src.jobs() {
            let n = src.n(j);
            let q = src.quant(j);
            let use_simd = simd::row_dot_enabled(q, requested);
            for r in 0..m {
                let (row, srow) = row_slices(q, src.packed(j), m, k, r);
                for t in 0..n {
                    out[base + r * n + t] =
                        dot_row_simd(q, row, srow, k, &src.xs_row(j, t)[..k], use_simd);
                }
            }
            base += m * n;
        }
    }

    /// D2 admits verify windows of 2..4 rows. A routed expert selected by `w`
    /// window rows is ONE grouped job of `n = w` activation rows; each row's
    /// bytes must equal a one-row (`n = 1`) job and independent `gemv` at every
    /// token position, with the Σx cache seeing aliased pointers (identical
    /// tokens sharing one buffer), distinct pointers with identical content, and
    /// distinct content; and the grouped path must equal the serial uncached
    /// reference.
    #[test]
    fn gemv_shared_window_rows_match_one_row_at_every_position() {
        struct Window<'a> {
            q: CpuQuant,
            packed: &'a [u8],
            rows: &'a [&'a [f32]],
        }
        impl SharedJobSource for Window<'_> {
            fn jobs(&self) -> usize {
                1
            }
            fn n(&self, _: usize) -> usize {
                self.rows.len()
            }
            fn quant(&self, _: usize) -> CpuQuant {
                self.q
            }
            fn packed(&self, _: usize) -> &[u8] {
                self.packed
            }
            fn xs_row(&self, _: usize, t: usize) -> &[f32] {
                self.rows[t]
            }
        }
        let (m, k) = (37usize, 1024usize);
        let base = x_of(k);
        let copies: Vec<Vec<f32>> = (0..4).map(|_| base.clone()).collect();
        let distinct: Vec<Vec<f32>> = (0..4)
            .map(|t| {
                base.iter()
                    .map(|v| v * (1.0 + 0.37 * t as f32) - 0.01 * t as f32)
                    .collect()
            })
            .collect();
        for requested in [Some(false), None] {
            for q in [
                CpuQuant::Mq4G256,
                CpuQuant::Mq6G256,
                CpuQuant::Mq4G256V2,
                CpuQuant::Mq4G128V2,
                CpuQuant::Mq3G256Lloyd,
                CpuQuant::Mfp4G32E8,
                CpuQuant::E8pI4G128R256,
                CpuQuant::E8pI4G128R128,
                CpuQuant::Mq3I4G128R256,
                CpuQuant::Mq3I4G128R128,
            ] {
                let packed = weights(q, m, k);
                let one_row = |x: &[f32]| {
                    let mut y = vec![0.0f32; m];
                    let src = Window {
                        q,
                        packed: &packed,
                        rows: &[x],
                    };
                    gemv_shared_sourced(m, k, &mut y, &src, requested);
                    let mut g = vec![0.0f32; m];
                    gemv_with_simd(q, &packed, m, k, x, &mut g, requested);
                    assert_eq!(
                        y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{q:?} requested={requested:?}: one-row shared job != gemv"
                    );
                    y
                };
                for width in 2..=4usize {
                    // Rows alias one buffer / copy one token / differ.
                    let aliased: Vec<&[f32]> = vec![&copies[0][..]; width];
                    let copied: Vec<&[f32]> = copies[..width].iter().map(|v| &v[..]).collect();
                    let varied: Vec<&[f32]> = distinct[..width].iter().map(|v| &v[..]).collect();
                    for (name, rows) in
                        [("aliased", aliased), ("copied", copied), ("varied", varied)]
                    {
                        let src = Window {
                            q,
                            packed: &packed,
                            rows: &rows,
                        };
                        let mut out = vec![0.0f32; m * width];
                        gemv_shared_sourced(m, k, &mut out, &src, requested);
                        let mut reference = vec![0.0f32; m * width];
                        single_row_reference(m, k, &mut reference, &src, requested);
                        assert_eq!(
                            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "{q:?} {name} width={width} requested={requested:?}: grouped != serial uncached reference"
                        );
                        for t in 0..width {
                            let y = one_row(rows[t]);
                            for r in 0..m {
                                assert_eq!(
                                    out[r * width + t].to_bits(),
                                    y[r].to_bits(),
                                    "{q:?} {name} width={width} requested={requested:?}: row {r} token {t} != one-row"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gemv_matches_the_f64_reference() {
        for (m, k) in [(3usize, 512usize), (1, 4096)] {
            for q in [
                CpuQuant::Mq4G256,
                CpuQuant::Mq4G256V2,
                CpuQuant::Mq4CG256,
                CpuQuant::Mq6G256,
                CpuQuant::Mq6G256V2,
                CpuQuant::Mq5G256,
                CpuQuant::Mq5G256V2,
                CpuQuant::Mq3G256,
                CpuQuant::Mq3G256V2,
                CpuQuant::Mq3G256Lloyd,
                CpuQuant::Mq2G256,
                CpuQuant::Mq2G256V2,
                CpuQuant::Mq2G256Lloyd,
                CpuQuant::Mq2G256LloydU,
                CpuQuant::Mq4G256Lloyd,
                CpuQuant::Hfq6G256,
                CpuQuant::Hfq4G256,
                CpuQuant::Hfq4G128,
                CpuQuant::Hfq3G256,
                CpuQuant::Hfq3G128,
                CpuQuant::Hfq2G256,
                CpuQuant::Hfq2G128,
                CpuQuant::Tq2G128,
                CpuQuant::Bq1G128,
                CpuQuant::F16,
                CpuQuant::F32,
                CpuQuant::Bf16,
                CpuQuant::Q8F16,
            ] {
                let packed = weights(q, m, k);
                let x = x_of(k);
                let mut y = vec![0.0f32; m];
                gemv(q, &packed, m, k, &x, &mut y);
                let want = oracle(q, &packed, m, k, &x);
                for (row, (got, want)) in y.iter().zip(&want).enumerate() {
                    let scale = want.abs().max(1e-6);
                    assert!(
                        ((*got as f64) - want).abs() / scale < 1e-4,
                        "{q:?} {m}x{k} row {row}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemv_matches_a_direct_scalar_dot_exactly() {
        // The whole-tensor entry point must agree with a hand-rolled loop over
        // the same decode, bit for bit — this is what pins the row stride and the
        // group offsets (a swapped stride would still "match" the f64 oracle to
        // within 1e-4 for the wrong reason only if the data were degenerate).
        let q = CpuQuant::Mq4G256;
        let (m, k) = (5usize, 512usize);
        let packed = weights(q, m, k);
        let x = x_of(k);
        let mut y = vec![0.0f32; m];
        gemv(q, &packed, m, k, &x, &mut y);
        let ge = q.group_elems();
        let gb = q.group_bytes();
        for (row, got) in y.iter().enumerate() {
            let mut acc = 0.0f32;
            for g in 0..k / ge {
                let mut codes = vec![0.0f32; ge];
                decode_group_codes(q, &packed[row * row_bytes(q, k) + g * gb..], &mut codes);
                let mut partial = 0.0f32;
                for i in 0..ge {
                    partial += codes[i] * x[g * ge + i];
                }
                acc += partial;
            }
            assert_eq!(*got, acc, "row {row}");
        }
    }

    #[test]
    fn gemm_equals_independent_gemv_calls() {
        let q = CpuQuant::Mq6G256;
        let (m, k, n) = (4usize, 512usize, 3usize);
        let packed = weights(q, m, k);
        let x: Vec<f32> = (0..n)
            .flat_map(|t| x_of(k).into_iter().map(move |v| v + t as f32))
            .collect();
        let mut batch = vec![0.0f32; n * m];
        gemm(q, &packed, m, k, &x, n, &mut batch);
        for t in 0..n {
            let mut single = vec![0.0f32; m];
            gemv(q, &packed, m, k, &x[t * k..t * k + k], &mut single);
            assert_eq!(&batch[t * m..t * m + m], &single[..], "token {t}");
        }
    }

    #[test]
    fn degenerate_shapes_are_no_ops() {
        let mut y = [7.0f32; 4];
        let x = [1.0f32; 4];
        gemv(CpuQuant::Mq4G256, &[], 0, 4, &x, &mut y);
        gemv(CpuQuant::Mq4G256, &[], 4, 0, &[], &mut y);
        assert!(y.iter().all(|v| *v == 7.0));
        let mut out = [7.0f32; 4];
        gemm(CpuQuant::Mq4G256, &[], 0, 4, &x, 2, &mut out);
        gemm(CpuQuant::Mq4G256, &[], 2, 0, &[], 2, &mut out);
        gemm(CpuQuant::Mq4G256, &[], 2, 4, &x, 0, &mut out);
        assert!(out.iter().all(|v| *v == 7.0));
    }

    #[test]
    fn row_bytes_follows_the_group_layout() {
        // 9B qwen3.5 shape: k = 4096.
        assert_eq!(row_bytes(CpuQuant::Mq4G256, 4096), 16 * 136);
        assert_eq!(row_bytes(CpuQuant::Mq4G256V2, 4096), 16 * 136);
        assert_eq!(row_bytes(CpuQuant::Mq6G256, 4096), 16 * 200);
        assert_eq!(row_bytes(CpuQuant::Mq3G256, 4096), 16 * 104);
        assert_eq!(row_bytes(CpuQuant::F16, 4096), 4096 * 2);
        assert_eq!(row_bytes(CpuQuant::F32, 4096), 4096 * 4);
        assert_eq!(row_bytes(CpuQuant::Q8F16, 4096), 128 * 34);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn unaligned_k_panics() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G256,
            &[0u8; 200],
            1,
            128,
            &[0.0f32; 128],
            &mut y,
        );
    }

    #[test]
    #[should_panic(expected = "packed has")]
    fn short_weight_buffer_panics() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G256,
            &[0u8; 135],
            1,
            256,
            &[0.0f32; 256],
            &mut y,
        );
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn g256_formats_still_reject_k_640() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G256V2,
            &[0u8; 640],
            1,
            640,
            &[0.0f32; 640],
            &mut y,
        );
    }

    #[test]
    #[should_panic(expected = "not a multiple of 128")]
    fn g128_formats_reject_k_not_a_multiple_of_128() {
        let mut y = [0.0f32; 1];
        gemv(
            CpuQuant::Mq4G128V2,
            &[0u8; 340],
            1,
            320,
            &[0.0f32; 320],
            &mut y,
        );
    }

    #[test]
    #[should_panic(expected = "not a multiple of 256")]
    fn shared_sourced_rejects_k_640_for_g256_jobs() {
        let packed = vec![0u8; 640];
        let xs = [0.0f32; 640];
        struct One<'a>(&'a [u8], &'a [f32]);
        impl SharedJobSource for One<'_> {
            fn jobs(&self) -> usize {
                1
            }
            fn n(&self, _: usize) -> usize {
                1
            }
            fn quant(&self, _: usize) -> CpuQuant {
                CpuQuant::Mq4G256V2
            }
            fn packed(&self, _: usize) -> &[u8] {
                self.0
            }
            fn xs_row(&self, _: usize, _: usize) -> &[f32] {
                self.1
            }
        }
        let mut out = [0.0f32; 1];
        gemv_shared_sourced(1, 640, &mut out, &One(&packed, &xs), None);
    }

    /// Flash-Next's down projection: qt 53, m = 2560 would be large, so a
    /// narrower `m` with the real `k = 640` (5 groups, 340 B/row). `gemv`,
    /// `gemm` and both shared-sourced entry points must accept it and agree
    /// with an `f64` oracle over the crate's own decoder, scalar and SIMD.
    #[test]
    fn qt53_k640_gemv_gemm_and_shared_match_the_oracle() {
        struct Job<'a> {
            packed: &'a [u8],
            xs: Vec<&'a [f32]>,
        }
        struct Src<'a>(Vec<Job<'a>>);
        impl SharedJobSource for Src<'_> {
            fn jobs(&self) -> usize {
                self.0.len()
            }
            fn n(&self, j: usize) -> usize {
                self.0[j].xs.len()
            }
            fn quant(&self, _: usize) -> CpuQuant {
                CpuQuant::Mq4G128V2
            }
            fn packed(&self, j: usize) -> &[u8] {
                self.0[j].packed
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                self.0[j].xs[t]
            }
        }
        let q = CpuQuant::Mq4G128V2;
        let (m, k) = (37usize, 640usize);
        assert_eq!(row_bytes(q, k), 340);
        let w_a = crate::testfix::weight_bytes(q, m, k);
        let w_b: Vec<u8> = w_a.iter().map(|b| b.rotate_left(3)).collect();
        // The rotated fixture's headers are garbage; keep the finite fixture
        // header of every group and only scramble the payload.
        let mut w_b = w_b;
        for (i, chunk) in w_b.chunks_mut(68).enumerate() {
            chunk[..4].copy_from_slice(&w_a[i * 68..i * 68 + 4]);
        }
        let xs: Vec<Vec<f32>> = (0..3)
            .map(|t| x_of(k).iter().map(|v| v + t as f32 * 0.37).collect())
            .collect();
        for requested in [Some(false), None] {
            let tol = |got: f32, want: f64, what: &str| {
                let rel = (got as f64 - want).abs() / want.abs().max(1e-3);
                assert!(rel <= 1e-4, "{what}: {got} vs {want} (rel {rel:.2e})");
            };
            // gemv.
            let mut y = vec![0.0f32; m];
            gemv_with_simd(q, &w_a, m, k, &xs[0], &mut y, requested);
            for (r, want) in oracle(q, &w_a, m, k, &xs[0]).iter().enumerate() {
                tol(y[r], *want, &format!("gemv row {r}"));
            }
            // gemm over three activation rows.
            let flat: Vec<f32> = xs.iter().flatten().copied().collect();
            let mut out = vec![0.0f32; 3 * m];
            gemm_with_simd(q, &w_a, m, k, &flat, 3, &mut out, requested);
            for t in 0..3 {
                for (r, want) in oracle(q, &w_a, m, k, &xs[t]).iter().enumerate() {
                    tol(out[t * m + r], *want, &format!("gemm t{t} row {r}"));
                }
            }
            // Shared-sourced: two distinct weights, 3 and 2 activation rows,
            // row-major outputs, bit-identical to the per-token gemv.
            let src = Src(vec![
                Job {
                    packed: &w_a,
                    xs: xs.iter().map(|v| v.as_slice()).collect(),
                },
                Job {
                    packed: &w_b,
                    xs: xs[..2].iter().map(|v| v.as_slice()).collect(),
                },
            ]);
            let mut shared = vec![0.0f32; m * 5];
            let split = gemv_shared_sourced_traced(m, k, &mut shared, &src, requested);
            assert_eq!(split.total_ns >= split.exec_ns, true);
            let mut plain = vec![0.0f32; m * 5];
            gemv_shared_sourced(m, k, &mut plain, &src, requested);
            assert_eq!(shared, plain, "traced vs untraced");
            let mut off = 0;
            for (j, w) in [&w_a, &w_b].into_iter().enumerate() {
                let n = src.0[j].xs.len();
                for t in 0..n {
                    let mut one = vec![0.0f32; m];
                    gemv_with_simd(q, w, m, k, src.0[j].xs[t], &mut one, requested);
                    for r in 0..m {
                        assert_eq!(
                            shared[off + r * n + t].to_bits(),
                            one[r].to_bits(),
                            "shared job {j} token {t} row {r}"
                        );
                        let want = oracle(q, w, m, k, src.0[j].xs[t])[r];
                        tol(shared[off + r * n + t], want, "shared vs oracle");
                    }
                }
                off += m * n;
            }
        }
    }

    /// The activation-sum scratch is checked out of the thread-local, not
    /// borrowed across the rayon region. A source callback that re-enters
    /// `gemv_shared_sourced` on the same thread (during Σ prep, and inside the
    /// region, where a one-worker pool runs every task on the caller) and
    /// concurrent callers on a small pool (a worker blocked in the region steals
    /// another call onto itself) must neither panic with `RefCell already
    /// borrowed` nor perturb the outer call's bits.
    #[test]
    fn shared_sourced_is_reentrant_from_source_callbacks_and_stolen_tasks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Plain<'a> {
            w: &'a [u8],
            xs: &'a [Vec<f32>],
        }
        impl SharedJobSource for Plain<'_> {
            fn jobs(&self) -> usize {
                1
            }
            fn n(&self, _: usize) -> usize {
                self.xs.len()
            }
            fn quant(&self, _: usize) -> CpuQuant {
                CpuQuant::Mq4G256V2
            }
            fn packed(&self, _: usize) -> &[u8] {
                self.w
            }
            fn xs_row(&self, _: usize, t: usize) -> &[f32] {
                &self.xs[t]
            }
        }
        /// Every activation read first runs a whole nested sourced GEMV over
        /// different weights/activations and checks it against its reference.
        struct Nesting<'a> {
            outer: Plain<'a>,
            inner: Plain<'a>,
            inner_want: &'a [f32],
            m: usize,
            k: usize,
            nested_calls: AtomicUsize,
        }
        impl SharedJobSource for Nesting<'_> {
            fn jobs(&self) -> usize {
                1
            }
            fn n(&self, j: usize) -> usize {
                self.outer.n(j)
            }
            fn quant(&self, j: usize) -> CpuQuant {
                self.outer.quant(j)
            }
            fn packed(&self, j: usize) -> &[u8] {
                self.outer.packed(j)
            }
            fn xs_row(&self, j: usize, t: usize) -> &[f32] {
                let mut got = vec![0.0f32; self.inner_want.len()];
                gemv_shared_sourced(self.m, self.k, &mut got, &self.inner, None);
                assert!(
                    got.iter()
                        .zip(self.inner_want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "nested call diverged"
                );
                self.nested_calls.fetch_add(1, Ordering::Relaxed);
                self.outer.xs_row(j, t)
            }
        }

        let q = CpuQuant::Mq4G256V2;
        let (m, k) = (9usize, 256usize);
        let w_outer = crate::testfix::weight_bytes(q, m, k);
        // Different weights with valid headers: the same rows, reversed.
        let w_inner: Vec<u8> = w_outer
            .chunks(row_bytes(q, k))
            .rev()
            .flatten()
            .copied()
            .collect();
        let xs_outer: Vec<Vec<f32>> = (0..3)
            .map(|t| x_of(k).iter().map(|v| v + t as f32 * 0.37).collect())
            .collect();
        let xs_inner: Vec<Vec<f32>> = (0..2)
            .map(|t| x_of(k).iter().map(|v| v * 0.5 - t as f32 * 0.11).collect())
            .collect();
        let plain_outer = Plain {
            w: &w_outer,
            xs: &xs_outer,
        };
        let plain_inner = Plain {
            w: &w_inner,
            xs: &xs_inner,
        };
        let mut want_outer = vec![0.0f32; m * xs_outer.len()];
        gemv_shared_sourced(m, k, &mut want_outer, &plain_outer, None);
        let mut want_inner = vec![0.0f32; m * xs_inner.len()];
        gemv_shared_sourced(m, k, &mut want_inner, &plain_inner, None);

        let run_once = || {
            let src = Nesting {
                outer: Plain {
                    w: &w_outer,
                    xs: &xs_outer,
                },
                inner: Plain {
                    w: &w_inner,
                    xs: &xs_inner,
                },
                inner_want: &want_inner,
                m,
                k,
                nested_calls: AtomicUsize::new(0),
            };
            let mut got = vec![0.0f32; want_outer.len()];
            gemv_shared_sourced(m, k, &mut got, &src, None);
            assert!(src.nested_calls.load(Ordering::Relaxed) > 0);
            assert!(
                got.iter()
                    .zip(&want_outer)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "outer call perturbed by reentrancy"
            );
        };

        for threads in [1usize, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            // Single call from a pool worker: nested calls land on the thread
            // that owns the outer scratch.
            pool.install(&run_once);
            // Concurrent calls: a worker blocked in the region steals another
            // whole call onto itself.
            pool.install(|| {
                (0..16usize).into_par_iter().for_each(|_| run_once());
            });
        }
    }

    /// End to end through the real format: encode weights, rotate the
    /// activation with `rotate_x_128_v2`, dot the stored codes — the result is
    /// the original-basis product `W·x` to 4-bit accuracy, and equals the dot
    /// of the *unrotated* `x` with the inverse-rotated (`dequant_group`)
    /// weights to f32 rounding. Using `x` = a weight row makes the product
    /// large (`|w|²`), so a wrong rotation (product near 0) cannot hide in the
    /// quantization tolerance.
    #[test]
    fn qt53_rotated_dot_equals_original_basis_product() {
        use crate::quant::{dequant_group, rotate_x_128_v2};
        let q = CpuQuant::Mq4G128V2;
        let (m, k) = (4usize, 640usize);
        let w: Vec<f32> = (0..m * k)
            .map(|i| {
                let v = (i as u64 * 2654435761 + 12345) % 2001;
                (v as f32 - 1000.0) * 0.001 + ((i % 7) as f32 - 3.0) * 0.05
            })
            .collect();
        let packed = crate::testfix::encode_mq4g128v2_ref(&w, m, k);
        // Activation = weight row 0 plus a little structure.
        let x: Vec<f32> = w[..k].to_vec();
        let mut xr = x.clone();
        rotate_x_128_v2(&mut xr);
        for requested in [Some(false), None] {
            let mut y = vec![0.0f32; m];
            gemv_with_simd(q, &packed, m, k, &xr, &mut y, requested);
            for row in 0..m {
                // (a) dequantized original-basis weights · unrotated x.
                let rb = row_bytes(q, k);
                let mut deq = 0.0f64;
                let mut norm = 0.0f64;
                for g in 0..k / 128 {
                    let mut orig = [0.0f32; 128];
                    dequant_group(q, &packed[row * rb + g * 68..], &mut orig);
                    for i in 0..128 {
                        deq += orig[i] as f64 * x[g * 128 + i] as f64;
                        norm += (orig[i] as f64).powi(2);
                    }
                }
                let bound = 1e-4
                    * (norm.sqrt() * (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt());
                assert!(
                    (y[row] as f64 - deq).abs() <= bound,
                    "row {row} simd={requested:?}: rotated-codes dot {} vs dequantized dot {deq}",
                    y[row]
                );
            }
            // (b) fidelity against the source weights for row 0: ~|w0|^2.
            let self_dot: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
            let rel = (y[0] as f64 - self_dot).abs() / self_dot;
            assert!(
                rel < 0.2,
                "row 0 product {} vs |w0|^2 {self_dot} (rel {rel:.3})",
                y[0]
            );
        }
    }
}

/// Property tests: the batched / partitioned entry points are bit-identical to
/// the singleton ones. These live in-file because they build weights with
/// `crate::testfix` (`pub(crate)`).
///
/// Why exact equality is the right claim: every output element is produced by
/// the same `dot_row_simd(q, row, scales_row, k, x_token, use_simd)` call regardless of how
/// the (token, row) pairs are batched or partitioned, so any difference is a
/// stride/offset bug, never rounding.
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    /// Every `CpuQuant` variant (all are valid for `gemm`: `k % 256 == 0`
    /// satisfies each format's group size).
    const ALL: [CpuQuant; 30] = [
        CpuQuant::Mq4G256,
        CpuQuant::Mq4G256V2,
        CpuQuant::Mq6G256,
        CpuQuant::Mq3G256,
        CpuQuant::Mq3G256Lloyd,
        CpuQuant::Hfq6G256,
        CpuQuant::Hfq3G256,
        CpuQuant::Hfq2G256,
        CpuQuant::Hfq4G128,
        CpuQuant::Hfq3G128,
        CpuQuant::Hfq2G128,
        CpuQuant::Mq5G256,
        CpuQuant::Mq2G256,
        CpuQuant::Mq2G256Lloyd,
        CpuQuant::Mq2G256LloydU,
        CpuQuant::Mq4G256Lloyd,
        CpuQuant::Mq4CG256,
        CpuQuant::Mq6G256V2,
        CpuQuant::Mq5G256V2,
        CpuQuant::Mq3G256V2,
        CpuQuant::Mq2G256V2,
        CpuQuant::Tq2G128,
        CpuQuant::Bq1G128,
        CpuQuant::Hfq4G256,
        CpuQuant::F16,
        CpuQuant::F32,
        CpuQuant::Bf16,
        CpuQuant::Q8F16,
        CpuQuant::Mfp4G32E8,
        CpuQuant::Mq4G128V2,
    ];

    /// Compile-time-ish guard: adding a `CpuQuant` variant without extending
    /// `ALL` fails to compile here (exhaustive match, no wildcard).
    #[allow(dead_code)]
    fn all_is_exhaustive(q: CpuQuant) {
        match q {
            CpuQuant::Mq4G256
            | CpuQuant::Mq4G256V2
            | CpuQuant::Mq6G256
            | CpuQuant::Mq3G256
            | CpuQuant::Mq3G256Lloyd
            | CpuQuant::Hfq6G256
            | CpuQuant::Hfq3G256
            | CpuQuant::Hfq2G256
            | CpuQuant::Hfq4G128
            | CpuQuant::Hfq3G128
            | CpuQuant::Hfq2G128
            | CpuQuant::Mq5G256
            | CpuQuant::Mq2G256
            | CpuQuant::Mq2G256Lloyd
            | CpuQuant::Mq2G256LloydU
            | CpuQuant::Mq4G256Lloyd
            | CpuQuant::Mq4CG256
            | CpuQuant::Mq6G256V2
            | CpuQuant::Mq5G256V2
            | CpuQuant::Mq3G256V2
            | CpuQuant::Mq2G256V2
            | CpuQuant::Tq2G128
            | CpuQuant::Bq1G128
            | CpuQuant::Hfq4G256
            | CpuQuant::F16
            | CpuQuant::F32
            | CpuQuant::Bf16
            | CpuQuant::Q8F16
            | CpuQuant::Mq4G128V2
            | CpuQuant::Mfp4G32E8 => {}
            // Planar formats (qt 131/132/135/136): excluded from `ALL` because
            // their rows are not byte-sliceable (the scale plane is separate, so
            // `prop_output_row_split_invariant` does not apply); their
            // gemv/gemm/shared-job bit-identity lives in `simd::tests`.
            CpuQuant::Mq3I4G128R256
            | CpuQuant::Mq3I4G128R128
            | CpuQuant::E8pI4G128R256
            | CpuQuant::E8pI4G128R128 => {}
        }
    }

    /// SplitMix64: deterministic byte/salt stream from the proptest-drawn seed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Bytes of the per-group scale/codebook header that must stay valid (finite
    /// decode): everything after it is free-form code payload.
    fn header_len(q: CpuQuant) -> usize {
        match q {
            CpuQuant::Mq4G128V2 => 4,
            CpuQuant::Mq4G256Lloyd => 32,
            CpuQuant::Mq3G256Lloyd => 16,
            _ => 8,
        }
    }

    /// `[m, k]` weight bytes: the crate fixture group (valid, finite header) with
    /// a random salt and random code payload. The unquantized formats keep the
    /// fixture's finite small-value tables (random bits there would be NaN/Inf).
    fn random_weights(q: CpuQuant, m: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let groups = k / q.group_elems();
        let quantized = !matches!(q, CpuQuant::F16 | CpuQuant::F32 | CpuQuant::Bf16);
        let mut out = Vec::with_capacity(m * groups * q.group_bytes());
        for _ in 0..m * groups {
            let mut g = crate::testfix::group_bytes(q, (rng.next() % 4096) as usize);
            if quantized {
                for b in &mut g[header_len(q)..] {
                    *b = rng.next() as u8;
                }
            }
            out.extend_from_slice(&g);
        }
        out
    }

    /// Finite activations in `[-512, 512)` on a 2^-6 grid.
    fn random_x(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
        (0..len)
            .map(|_| ((rng.next() % 65536) as i64 - 32768) as f32 / 64.0)
            .collect()
    }

    #[derive(Debug)]
    struct Case {
        q: CpuQuant,
        /// `Some(false)` = forced scalar, `None` = runtime SIMD detection.
        simd: Option<bool>,
        /// Output features (weight rows).
        m: usize,
        k: usize,
        /// Batch (tokens).
        n: usize,
        packed: Vec<u8>,
        x: Vec<f32>,
        /// Raw split-point entropy (reduced mod the relevant range by the property).
        cuts: Vec<usize>,
    }

    fn case() -> impl Strategy<Value = Case> {
        (
            0..ALL.len(),
            any::<bool>(),
            1usize..=24,
            1usize..=4,
            1usize..=9,
            any::<u64>(),
            prop::collection::vec(any::<usize>(), 0..5),
        )
            .prop_map(|(qi, scalar, m, groups, n, seed, cuts)| {
                let q = ALL[qi];
                // gemv/gemm assert `k % 256 == 0`, which is a multiple of every
                // format's group size (32/128/256).
                let k = 256 * groups;
                Case {
                    q,
                    simd: scalar.then_some(false),
                    m,
                    k,
                    n,
                    packed: random_weights(q, m, k, seed),
                    x: random_x(n * k, seed),
                    cuts,
                }
            })
    }

    /// Sorted, deduplicated interior split points in `1..len`, plus both ends.
    fn boundaries(raw: &[usize], len: usize) -> Vec<usize> {
        let mut b: Vec<usize> = raw
            .iter()
            .filter(|_| len > 1)
            .map(|c| 1 + c % (len - 1))
            .collect();
        b.push(0);
        b.push(len);
        b.sort_unstable();
        b.dedup();
        b
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// gemm over `n` tokens == `n` independent gemv calls, bit for bit.
        #[test]
        fn prop_gemm_equals_n_gemv(c in case()) {
            let Case { q, simd, m, k, n, packed, x, .. } = c;
            let mut batch = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut batch, simd);
            for t in 0..n {
                let mut single = vec![0.0f32; m];
                gemv_with_simd(q, &packed, m, k, &x[t * k..(t + 1) * k], &mut single, simd);
                prop_assert_eq!(
                    bits(&batch[t * m..(t + 1) * m]),
                    bits(&single),
                    "{:?} m={} k={} n={} token {}", q, m, k, n, t
                );
            }
        }

        /// gemm over the whole batch == concatenation of gemm over contiguous
        /// chunks of the batch (random split points), bit for bit.
        #[test]
        fn prop_gemm_batch_partition_invariant(c in case()) {
            let Case { q, simd, m, k, n, packed, x, cuts } = c;
            let mut full = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut full, simd);
            let b = boundaries(&cuts, n);
            let mut stitched: Vec<f32> = Vec::with_capacity(n * m);
            for w in b.windows(2) {
                let (lo, hi) = (w[0], w[1]);
                let mut part = vec![0.0f32; (hi - lo) * m];
                gemm_with_simd(q, &packed, m, k, &x[lo * k..hi * k], hi - lo, &mut part, simd);
                stitched.extend_from_slice(&part);
            }
            prop_assert_eq!(
                bits(&full),
                bits(&stitched),
                "{:?} m={} k={} n={} partition {:?}", q, m, k, n, b
            );
        }

        /// Output-row split. Weights are row-major with `row_bytes(q, k)` per
        /// output row (the layout `gemv`/`gemm` index by `row * rb`), so a
        /// contiguous row range is a contiguous byte slice. gemm/gemv over that
        /// slice == the matching columns of the full result, bit for bit.
        #[test]
        fn prop_output_row_split_invariant(c in case()) {
            let Case { q, simd, m, k, n, packed, x, cuts } = c;
            let rb = row_bytes(q, k);
            let mut full = vec![0.0f32; n * m];
            gemm_with_simd(q, &packed, m, k, &x, n, &mut full, simd);
            let b = boundaries(&cuts, m);
            for w in b.windows(2) {
                let (lo, hi) = (w[0], w[1]);
                let sub = &packed[lo * rb..hi * rb];
                let mr = hi - lo;
                let mut part = vec![0.0f32; n * mr];
                gemm_with_simd(q, sub, mr, k, &x, n, &mut part, simd);
                for t in 0..n {
                    prop_assert_eq!(
                        bits(&part[t * mr..(t + 1) * mr]),
                        bits(&full[t * m + lo..t * m + hi]),
                        "{:?} m={} k={} n={} rows {}..{} token {} (gemm)", q, m, k, n, lo, hi, t
                    );
                    let mut single = vec![0.0f32; mr];
                    gemv_with_simd(q, sub, mr, k, &x[t * k..(t + 1) * k], &mut single, simd);
                    prop_assert_eq!(
                        bits(&single),
                        bits(&full[t * m + lo..t * m + hi]),
                        "{:?} m={} k={} n={} rows {}..{} token {} (gemv)", q, m, k, n, lo, hi, t
                    );
                }
            }
        }
    }
}
