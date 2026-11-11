// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-rank handle of the Flash-Next trunk-shard route (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`):
//! the fused TP2 all-reduce + HC write ([`ArHcExchange`]) a rank's forward hands to its
//! [`HyperWriteOp`](hipfire_dispatch::pipeline::HyperWriteOp)s.
//!
//! Every EP rank runs its forward on its own thread (rank 0 on the caller's, a peer on its
//! pool worker), so the route is a thread-local installed when that rank attaches its EP
//! tables ([`install`]) and read where the forward builds its HC writes. It is leaked once
//! per load (a few bytes): the forward borrows it as `&'static dyn HyperWriteAr`.

use hipfire_dispatch::pipeline::{HyperWriteAr, QsaSelectGather, RowGather};
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::ar_hc_write::{sp_even_split, SpAgElem, StreamKind};
use rdna_compute::{Gpu, GpuTensor};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// Switch of the sequence-parallel prefill route: unset (or `1` / `on` / `true`) = ON wherever
/// the trunk-shard route runs (exact gfx1201 EP2, `HIPFIRE_QWEN4_EP_TRUNK_SHARD`); `0` / `off` /
/// `false` is the kill switch (the all-reduce route); any other value is off (fail safe).
pub(crate) const SP_ENV: &str = "HIPFIRE_QWEN4_EP_SEQ_PARALLEL";

/// Smallest chunk (rows) of the wide sequence-parallel band (any HC route; the sub-512 band is
/// [`SP_SUB512_MIN_ROWS`]..=511 on the prefill lane WMMA HC read): both halves (`n / 2` rows, floored) must
/// sit on the same side of every HC read / write kernel-route threshold as the whole chunk
/// (`wide_exact_rows` / the few-row limit up to `WIDE_VERIFY_MAX_ROWS`, the F16 WMMA and BF16
/// stream floor `QWEN4_F16_WMMA_MIN_TOKENS`, the HC up `_v2` floor, the HC down fused floors), so
/// the own-row HC read / write launches take the routes the all-rows launch takes. The binding
/// one is `HC_DOWN_FUSED_MIN_ROWS` (1536 on gfx1201): `3072 / 2`. Pinned by a unit test.
// selector-receipt: receipts/selectors/sp_min_rows.json
pub(crate) const SP_MIN_ROWS: usize = 3072;

/// Whether the raw [`SP_ENV`] value asks for the sequence-parallel prefill route.
pub(crate) fn sp_switch(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), None | Some("" | "1" | "on" | "true"))
}

/// Whether the process asks for the sequence-parallel prefill route ([`SP_ENV`], [`sp_switch`]).
pub(crate) fn sp_requested() -> bool {
    sp_switch(hipfire_config::developer_var(SP_ENV).ok().as_deref())
}

/// Switch of the sub-512 sequence-parallel band: unset (or `1` / `on` / `true`) = ON wherever the
/// sequence-parallel prefill route runs ([`SP_ENV`]); `0` / `off` / `false` is the kill switch
/// (chunks below [`SP_MIN_ROWS`] keep the all-reduce route); any other value is off (fail safe).
/// A trunk-shard prefill chunk of [`SP_SUB512_MIN_ROWS`]..=511 rows then also runs
/// sequence-parallel, provided its HC read is the pinned-split-K prefill lane WMMA entry for the
/// whole chunk AND for both halves (`Gpu::prefill_lwmma_hc_applies`, checked by the forward):
/// that entry is row-count invariant, so a row's bytes do not depend on which half carries it.
pub(crate) const SP_SUB512_ENV: &str = "HIPFIRE_QWEN4_EP_SP_SUB512";

/// Fewest rows of a sub-512 sequence-parallel chunk: both halves (the smaller is `n / 2`) above
/// the few-row window (`WIDE_VERIFY_MAX_ROWS`, 64), where the HC read leaves the lane WMMA
/// decode arm and the HC8 decode route for the prefill lane WMMA entry. Pinned by a unit test.
pub(crate) const SP_SUB512_MIN_ROWS: usize =
    2 * (rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS + 1);

static SP_SUB512_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    sp_switch(hipfire_config::developer_var(SP_SUB512_ENV).ok().as_deref())
});

/// Whether an `n`-row chunk is inside a sequence-parallel row band: at least [`SP_MIN_ROWS`], or
/// (`sub512`, [`SP_SUB512_ENV`]) [`SP_SUB512_MIN_ROWS`]..=511 rows. The forward adds the HC route
/// check of the sub-512 band.
pub(crate) fn sp_rows_in_band(n: usize, sub512: bool) -> bool {
    n >= SP_MIN_ROWS
        || (sub512
            && (SP_SUB512_MIN_ROWS..rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS).contains(&n))
}

/// [`sp_rows_in_band`] with this process's [`SP_SUB512_ENV`].
pub(crate) fn sp_chunk_rows_in_band(n: usize) -> bool {
    sp_rows_in_band(n, *SP_SUB512_ON)
}

/// Switch of sequence-parallel HC for EP prefill WAVES (`forward_lanes_scoped`) of at least
/// [`SP_SUB512_MIN_ROWS`] rows, on top of [`SP_SUB512_ENV`]: unset / `1` / `on` / `true` is on,
/// `0` / `off` / `false` is the kill switch, any other value is off.
pub(crate) const SP_SUB512_WAVE_ENV: &str = "HIPFIRE_QWEN4_EP_SP_SUB512_WAVE";

static SP_SUB512_WAVE_ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    sp_switch(
        hipfire_config::developer_var(SP_SUB512_WAVE_ENV)
            .ok()
            .as_deref(),
    )
});

/// Whether an `n`-row EP prefill wave runs its joint HC sequence-parallel: the sub-512 switch
/// (`sub512`, [`SP_SUB512_ENV`]) and the wave switch (`wave`, [`SP_SUB512_WAVE_ENV`]) are both on
/// and `n >= `[`SP_SUB512_MIN_ROWS`]. The wave's routing-row cap of 511 keeps every launch and
/// both halves on sub-512 routes (waves are capped at 2048 rows).
pub(crate) fn sp_wave_rows_in_band_from(n: usize, sub512: bool, wave: bool) -> bool {
    sub512 && wave && n >= SP_SUB512_MIN_ROWS
}

/// [`sp_wave_rows_in_band_from`] with this process's switches.
pub(crate) fn sp_wave_rows_in_band(n: usize) -> bool {
    sp_wave_rows_in_band_from(n, *SP_SUB512_ON, *SP_SUB512_WAVE_ON)
}

/// Switch of the sequence-parallel QSA indexer inside a sequence-parallel chunk (each rank selects
/// its own work-balanced rows and the block ids are all-gathered): unset (or `1` / `on` / `true`)
/// = ON wherever the sequence-parallel prefill route runs ([`SP_ENV`]); `0` / `off` / `false` is
/// the kill switch (every rank selects every row); any other value is off (fail safe).
pub(crate) const SP_INDEXER_ENV: &str = "HIPFIRE_QWEN4_EP_SP_INDEXER";

/// Whether the raw [`SP_INDEXER_ENV`] value asks for the sequence-parallel QSA indexer.
pub(crate) fn sp_indexer_switch(raw: Option<&str>) -> bool {
    matches!(raw.map(str::trim), None | Some("" | "1" | "on" | "true"))
}

/// Whether the process asks for the sequence-parallel QSA indexer ([`SP_INDEXER_ENV`],
/// [`sp_indexer_switch`]).
pub(crate) fn sp_indexer_requested() -> bool {
    sp_indexer_switch(
        hipfire_config::developer_var(SP_INDEXER_ENV)
            .ok()
            .as_deref(),
    )
}

/// The row split of one sequence-parallel chunk on one rank: `own` / `peer` are
/// `(first row, rows)` ranges of the `n`-row tile ([`sp_even_split`], mirrored on the peer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpRows {
    pub(crate) own: (usize, usize),
    pub(crate) peer: (usize, usize),
    pub(crate) n: usize,
}

impl SpRows {
    /// The split of an `n`-row chunk on `rank` (0 or 1).
    pub(crate) fn of(rank: usize, n: usize) -> Self {
        let split = sp_even_split(n);
        Self {
            own: split[rank],
            peer: split[1 - rank],
            n,
        }
    }
}

/// One rank's route.
pub(crate) struct TrunkShardRoute {
    exchange: Arc<ArHcExchange>,
    rank: usize,
    /// The mixer (GDN/QSA out-projection) writes carry the all-reduce.
    attn: bool,
    /// The FFN writes carry it (the sealed MoE then yields this rank's partial: no row
    /// gather).
    ffn: bool,
    /// [`SP_ENV`] was on when the route was installed.
    sp_on: bool,
    /// [`SP_INDEXER_ENV`] was on when the route was installed.
    sp_indexer_on: bool,
    /// The sequence-parallel split of the chunk program being executed ([`sp_begin`] /
    /// [`sp_end`]); `None` outside one. A mutex, not a cell: the route is a `Sync`
    /// [`HyperWriteAr`], though only its own rank thread ever touches the state.
    sp: Mutex<Option<SpRows>>,
}

impl TrunkShardRoute {
    pub(crate) fn new(exchange: Arc<ArHcExchange>, rank: usize, attn: bool, ffn: bool) -> Self {
        Self {
            exchange,
            rank,
            attn,
            ffn,
            sp_on: sp_requested(),
            sp_indexer_on: sp_indexer_requested(),
            sp: Mutex::new(None),
        }
    }

    fn sp_state(&self) -> Option<SpRows> {
        *self.sp.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl HyperWriteAr for TrunkShardRoute {
    fn ar_hyper_write(
        &self,
        gpu: &mut Gpu,
        partial: &GpuTensor,
        streams: &GpuTensor,
        gates: &GpuTensor,
        rows: usize,
        state_bf16: bool,
    ) -> Result<(), String> {
        // Sequence-parallel chunk: the op is built over the OWN rows (`rows == own.1`, views at
        // the own-row offset). Rebase the views to the full tile and run the reduce-scatter write,
        // which folds and writes the own rows and pushes the peer rows to the peer.
        if let Some(sp) = self.sp_state().filter(|sp| rows == sp.own.1) {
            let kind = StreamKind::from_state_bf16(state_bf16);
            let row0 = sp.own.0;
            // SAFETY: the views were cut at row `row0` of live full-tile buffers (`[n, 2560]` f32
            // partial, `[n, 4, 2560]` streams of `kind`, `[n, 4]` f32 gates), so the rebased
            // pointers are those buffers' bases; `sp_rs_write` touches only rows of `own` / `peer`
            // inside the `n`-row tile.
            return unsafe {
                self.exchange.sp_rs_write(
                    self.rank,
                    gpu,
                    partial
                        .buf
                        .as_ptr()
                        .cast::<u8>()
                        .sub(row0 * 2560 * 4)
                        .cast::<c_void>(),
                    streams
                        .buf
                        .as_ptr()
                        .cast::<u8>()
                        .sub(row0 * kind.row_bytes())
                        .cast::<c_void>(),
                    kind,
                    gates
                        .buf
                        .as_ptr()
                        .cast::<u8>()
                        .sub(row0 * 4 * 4)
                        .cast::<c_void>()
                        .cast_const(),
                    sp.own,
                    sp.peer,
                    std::ptr::null_mut(),
                )
            }
            .map_err(|error| format!("rank {}: sequence-parallel write: {error}", self.rank));
        }
        // SAFETY: the tensors are live device buffers of this rank's forward with the
        // `[rows, 2560]` f32 partial, `[rows, 4, 2560]` streams of the `state_bf16` kind and `[rows, 4]` f32
        // gate extents (`execute_hyper_write_inner` validated the op).
        unsafe {
            self.exchange.write(
                self.rank,
                gpu,
                partial.buf.as_ptr(),
                streams.buf.as_ptr(),
                StreamKind::from_state_bf16(state_bf16),
                gates.buf.as_ptr(),
                rows,
                std::ptr::null_mut(),
            )
        }
        .map_err(|error| format!("rank {}: {error}", self.rank))
    }
}

impl RowGather for TrunkShardRoute {
    fn row_gather(
        &self,
        gpu: &mut Gpu,
        tensor: &GpuTensor,
        f32_bf16: bool,
        row_elems: usize,
        rows: usize,
    ) -> Result<(), String> {
        let sp = self.sp_state().ok_or_else(|| {
            format!(
                "rank {}: row gather outside a sequence-parallel chunk",
                self.rank
            )
        })?;
        if rows != sp.n {
            return Err(format!(
                "rank {}: row gather of {rows} rows in a sequence-parallel chunk of {}",
                self.rank, sp.n
            ));
        }
        let elem = if f32_bf16 {
            SpAgElem::F32Bf16
        } else {
            SpAgElem::Bf16
        };
        // SAFETY: `tensor` is the live full tile (`rows * row_elems` elements of `elem`, the
        // dispatch op validated it); the gather touches only the rows of `own` / `peer`.
        unsafe {
            self.exchange.sp_ag_rows(
                self.rank,
                gpu,
                tensor.buf.as_ptr(),
                elem,
                row_elems,
                sp.own,
                sp.peer,
            )
        }
        .map_err(|error| format!("rank {}: sequence-parallel gather: {error}", self.rank))
    }
}

impl QsaSelectGather for TrunkShardRoute {
    fn gather_selection(
        &self,
        gpu: &mut Gpu,
        buf: &GpuTensor,
        row_elems: usize,
        own: (usize, usize),
        peer: (usize, usize),
    ) -> Result<(), String> {
        let sp = self.sp_state().ok_or_else(|| {
            format!(
                "rank {}: selection gather outside a sequence-parallel chunk",
                self.rank
            )
        })?;
        if own.0 + own.1 > sp.n || peer.0 + peer.1 > sp.n {
            return Err(format!(
                "rank {}: selection gather of {own:?} / {peer:?} in a sequence-parallel chunk of {}",
                self.rank, sp.n
            ));
        }
        // The ranges are the work-balanced split (different lengths), so the gather runs in
        // lockstep slices: every launch carries rows both ways.
        // SAFETY: `buf` is the live compact selection (`rows * row_elems` BF16-sized words, the
        // dispatch op sized it); the gather touches only the rows of `own` / `peer`, which
        // lie inside the armed chunk.
        unsafe {
            self.exchange.sp_ag_rows_lockstep(
                self.rank,
                gpu,
                buf.buf.as_ptr(),
                SpAgElem::Bf16,
                row_elems,
                own,
                peer,
            )
        }
        .map_err(|error| format!("rank {}: sequence-parallel gather: {error}", self.rank))
    }
}

thread_local! {
    static ROUTE: Cell<Option<&'static TrunkShardRoute>> = const { Cell::new(None) };
}

/// Install (or with `None` clear) this thread's route.
pub(crate) fn install(route: Option<TrunkShardRoute>) {
    let leaked = route.map(|route| &*Box::leak(Box::new(route)));
    ROUTE.with(|cell| cell.set(leaked));
}

/// The writer of the mixer HC writes, when this thread's route carries them.
pub(crate) fn attn_writer() -> Option<&'static dyn HyperWriteAr> {
    ROUTE
        .with(Cell::get)
        .filter(|r| r.attn)
        .map(|r| r as &dyn HyperWriteAr)
}

/// The writer of the FFN HC writes, when this thread's route carries them.
pub(crate) fn ffn_writer() -> Option<&'static dyn HyperWriteAr> {
    ROUTE
        .with(Cell::get)
        .filter(|r| r.ffn)
        .map(|r| r as &dyn HyperWriteAr)
}

/// Whether this thread's FFN is the partial route (no expert-row gather).
pub(crate) fn ffn_partial_active() -> bool {
    ROUTE.with(Cell::get).is_some_and(|r| r.ffn)
}

/// This thread's route when it can run sequence-parallel chunks: installed with both writers
/// ([`attn_writer`], [`ffn_writer`]), [`SP_ENV`] on at install and an exchange with the dense
/// mailbox region the SP launches need.
pub(crate) fn sp_route() -> Option<&'static TrunkShardRoute> {
    ROUTE
        .with(Cell::get)
        .filter(|r| r.attn && r.ffn && r.sp_on && r.exchange.sp_available())
}

/// The row split of an `n`-row chunk on this thread's rank, `None` without an [`sp_route`].
/// Pure: arms nothing (see [`sp_begin`]).
pub(crate) fn sp_rows(n: usize) -> Option<SpRows> {
    sp_route().map(|route| SpRows::of(route.rank, n))
}

/// Arm the SP split for the chunk program about to execute: its `HyperWrite`s with
/// `rows == rows.own.1` take the reduce-scatter route and its `RowGather` steps use these ranges.
pub(crate) fn sp_begin(rows: SpRows) {
    if let Some(route) = ROUTE.with(Cell::get) {
        *route.sp.lock().unwrap_or_else(|p| p.into_inner()) = Some(rows);
    }
}

/// Disarm the SP split (every path out of the chunk program, error paths included).
pub(crate) fn sp_end() {
    if let Some(route) = ROUTE.with(Cell::get) {
        *route.sp.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// The sequence-parallel QSA selection of this thread's rank: `(rank, route)` when this thread's
/// [`sp_route`] exists and [`SP_INDEXER_ENV`] was on at install, else `None` (every rank selects
/// every row).
pub(crate) fn sp_qsa_gather() -> Option<(usize, &'static dyn QsaSelectGather)> {
    sp_route()
        .filter(|r| r.sp_indexer_on)
        .map(|r| (r.rank, r as &'static dyn QsaSelectGather))
}

/// The work-balanced row split of the QSA indexer select of an `n`-row tile starting at
/// `position_start`, as `[(first row, rows); 2]` (rank 0 first, from row 0). Row `r` selects over
/// `position_start + r + 1` visible tokens, so its cost grows with `r` and an even row split
/// leaves the later rows' rank with more work. The cut `s` is the smallest in `[1, n - 1]` whose
/// left side (rows `0..s`) costs at least the right side, or the one before it when that makes
/// the larger side smaller (ties go to the smaller `s`). Needs `n >= 2` (both ranks own a row);
/// the SP route only runs chunks of at least [`SP_SUB512_MIN_ROWS`].
pub(crate) fn qsa_select_split(n: usize, position_start: usize) -> [(usize, usize); 2] {
    debug_assert!(n >= 2, "a QSA select split needs two rows, got {n}");
    let n = n.max(2);
    let (p, n128) = (position_start as u128, n as u128);
    // Cost of rows `0..s`: sum of `p + r + 1`.
    let left = |s: u128| s * p + s * (s + 1) / 2;
    let total = left(n128);
    let right = |s: u128| total - left(s);
    // Smallest `s` in `[1, n - 1]` with `left(s) >= right(s)`; `left` grows with `s`.
    let (mut lo, mut hi) = (1u128, n128 - 1);
    if left(hi) < right(hi) {
        lo = hi;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if left(mid) >= right(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let mut s = lo;
    // `s - 1` has `left < right`: its larger side is `right(s - 1)`.
    if s > 1 && left(s) >= right(s) && right(s - 1) <= left(s) {
        s -= 1;
    }
    let s = s as usize;
    [(0, s), (s, n - s)]
}

/// Resolve the SP kernels of `exchange` on `gpu` when [`SP_ENV`] is on and the exchange has a
/// dense mailbox region (load time: JIT is not capture-safe); a no-op otherwise.
pub(crate) fn sp_prepare(exchange: &ArHcExchange, gpu: &mut Gpu) -> Result<(), String> {
    if sp_requested() && exchange.sp_available() {
        exchange
            .sp_ensure_kernels(gpu)
            .map_err(|e| format!("resolve the sequence-parallel kernels: {e}"))?;
        if sp_indexer_requested() {
            gpu.ensure_qsa_sp_select_kernels()
                .map_err(|e| format!("resolve the sequence-parallel QSA select kernels: {e}"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_is_per_thread_and_clearable() {
        assert!(attn_writer().is_none() && ffn_writer().is_none() && !ffn_partial_active());
        let x = Arc::new(ArHcExchange::new(8).unwrap());
        install(Some(TrunkShardRoute::new(x.clone(), 0, false, true)));
        assert!(attn_writer().is_none());
        assert!(ffn_writer().is_some() && ffn_partial_active());
        std::thread::spawn(|| assert!(ffn_writer().is_none()))
            .join()
            .unwrap();
        install(None);
        assert!(ffn_writer().is_none() && !ffn_partial_active());
    }

    /// `SP_MIN_ROWS` keeps the own-row halves (`SP_MIN_ROWS / 2` rows, the smaller half of any
    /// odd chunk of at least `SP_MIN_ROWS`) at or above every HC read / write kernel-route
    /// threshold, and above the few-row routes, so the own-row launches pick the routes the
    /// all-rows launches pick.
    #[test]
    fn sp_min_rows_covers_every_hc_route_threshold() {
        use rdna_compute::gemm::{
            HC_DOWN_FUSED_GFX1151_MIN_ROWS, HC_DOWN_FUSED_MIN_ROWS, QWEN4_F16_WMMA_MIN_TOKENS,
        };
        use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
        use rdna_compute::tensor_ops::HC_UP_V2_MIN_ROWS;
        let floor = SP_MIN_ROWS / 2;
        // `rows >= T` thresholds.
        for (name, threshold) in [
            (
                "QWEN4_F16_WMMA_MIN_TOKENS (F16 WMMA read, BF16 streams)",
                QWEN4_F16_WMMA_MIN_TOKENS,
            ),
            ("HC_DOWN_FUSED_MIN_ROWS (gfx1201)", HC_DOWN_FUSED_MIN_ROWS),
            (
                "HC_DOWN_FUSED_GFX1151_MIN_ROWS",
                HC_DOWN_FUSED_GFX1151_MIN_ROWS,
            ),
            ("HC_UP_V2_MIN_ROWS", HC_UP_V2_MIN_ROWS),
        ] {
            assert!(floor >= threshold, "{name}: {floor} own rows < {threshold}");
        }
        // `rows <= limit` few-row / wide-exact routes (`wide_exact_rows`, the k4 gate GEMV, the
        // decode pairing): closed for both halves.
        for (name, limit) in [
            ("FEW_ROW_ROUTE_ROWS", FEW_ROW_ROUTE_ROWS),
            ("WIDE_VERIFY_MAX_ROWS (few_row_limit)", WIDE_VERIFY_MAX_ROWS),
        ] {
            assert!(floor > limit, "{name}: {floor} own rows <= {limit}");
        }
        // The gfx1201 row fold floor (`hc_row_fold_gfx1201_applies`: rows >= 16).
        assert!(floor >= 16);
        assert!(SP_MIN_ROWS.is_multiple_of(2));
    }

    /// The sub-512 band ([`SP_SUB512_MIN_ROWS`]..=511 rows): both halves and the whole chunk sit
    /// above every few-row route (the few-row window, the lane WMMA decode arm and the HC8 decode
    /// route all end at 64 rows) and below every >= T route of the HC read / write (the F16 WMMA
    /// read at 512, the HC down fused floors, the BF16 streams), so with the prefill lane WMMA HC
    /// entry in force (the forward's `prefill_lwmma_hc_applies` check, which also covers the HC up
    /// `_v2` floor at 128 that the halves would otherwise straddle) the own-row launches pick the
    /// whole chunk's routes. One row less and the smaller half drops into the few-row window.
    #[test]
    fn sp_sub512_band_keeps_both_halves_on_the_prefill_hc_routes() {
        use rdna_compute::gemm::{
            HC_DOWN_FUSED_GFX1151_MIN_ROWS, HC_DOWN_FUSED_MIN_ROWS, QWEN4_F16_WMMA_MIN_TOKENS,
        };
        use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
        assert_eq!(SP_SUB512_MIN_ROWS, 130);
        assert_eq!(rdna_compute::hc8::MAX_ROWS, WIDE_VERIFY_MAX_ROWS);
        assert!(rdna_compute::gemv::LANE_WMMA_MIN_ROWS <= WIDE_VERIFY_MAX_ROWS);
        for n in SP_SUB512_MIN_ROWS..QWEN4_F16_WMMA_MIN_TOKENS {
            assert!(sp_rows_in_band(n, true), "{n}");
            let [a, b] = sp_even_split(n);
            for half in [a.1, b.1] {
                assert!(half > WIDE_VERIFY_MAX_ROWS, "{n}: {half}");
                assert!(half > FEW_ROW_ROUTE_ROWS, "{n}: {half}");
                assert!(half >= 16, "{n}: row fold floor");
            }
            for rows in [a.1, b.1, n] {
                assert!(rows < QWEN4_F16_WMMA_MIN_TOKENS, "{n}");
                assert!(rows < HC_DOWN_FUSED_MIN_ROWS, "{n}");
                assert!(rows < HC_DOWN_FUSED_GFX1151_MIN_ROWS, "{n}");
            }
        }
        let below = SP_SUB512_MIN_ROWS - 1;
        assert!(sp_even_split(below)[1].1 <= WIDE_VERIFY_MAX_ROWS);
        assert!(!sp_rows_in_band(below, true));
    }

    #[test]
    fn sp_bands_follow_the_sub512_switch() {
        for (n, on, off) in [
            (64, false, false),
            (129, false, false),
            (130, true, false),
            (149, true, false),
            (511, true, false),
            (512, false, false),
            (1192, false, false),
            (3071, false, false),
            (SP_MIN_ROWS, true, true),
            (16384, true, true),
        ] {
            assert_eq!(sp_rows_in_band(n, true), on, "{n} on");
            assert_eq!(sp_rows_in_band(n, false), off, "{n} off");
        }
        assert_ne!(SP_SUB512_ENV, SP_ENV);
    }

    #[test]
    fn sp_wave_band_needs_both_switches_and_the_floor() {
        for n in [129usize, 130, 596, 1192, 2048] {
            for sub512 in [false, true] {
                for wave in [false, true] {
                    assert_eq!(
                        sp_wave_rows_in_band_from(n, sub512, wave),
                        sub512 && wave && n >= 130,
                        "{n} {sub512} {wave}"
                    );
                }
            }
        }
        assert_ne!(SP_SUB512_WAVE_ENV, SP_SUB512_ENV);
    }

    #[test]
    fn sp_switch_is_default_on_with_a_kill_switch() {
        for on in [
            None,
            Some(""),
            Some("1"),
            Some("on"),
            Some("true"),
            Some(" 1 "),
        ] {
            assert!(sp_switch(on), "{on:?}");
        }
        for off in [
            Some("0"),
            Some("off"),
            Some("false"),
            Some("lab"),
            Some("2"),
        ] {
            assert!(!sp_switch(off), "{off:?}");
        }
    }

    #[test]
    fn sp_split_mirrors_between_ranks() {
        for n in [
            SP_SUB512_MIN_ROWS,
            SP_SUB512_MIN_ROWS + 1,
            511,
            SP_MIN_ROWS,
            SP_MIN_ROWS + 1,
            4096,
            4097,
        ] {
            let (a, b) = (SpRows::of(0, n), SpRows::of(1, n));
            assert_eq!((a.own, a.peer), (b.peer, b.own));
            assert_eq!(a.own.0, 0);
            assert_eq!(a.own.0 + a.own.1, a.peer.0);
            assert_eq!(a.own.1 + a.peer.1, n);
            let floor = if n >= SP_MIN_ROWS {
                SP_MIN_ROWS / 2
            } else {
                SP_SUB512_MIN_ROWS / 2
            };
            assert!(a.own.1 >= floor && b.own.1 >= floor);
        }
    }

    #[test]
    fn sp_indexer_switch_is_default_on_with_a_kill_switch() {
        for on in [
            None,
            Some(""),
            Some("1"),
            Some("on"),
            Some("true"),
            Some(" 1 "),
        ] {
            assert!(sp_indexer_switch(on), "{on:?}");
        }
        for off in [
            Some("0"),
            Some("off"),
            Some("false"),
            Some("lab"),
            Some("2"),
        ] {
            assert!(!sp_indexer_switch(off), "{off:?}");
        }
        assert_ne!(SP_INDEXER_ENV, SP_ENV);
    }

    /// The work of a range of rows `[first, first + rows)` of a tile at `position_start`.
    fn qsa_range_cost(position_start: usize, first: usize, rows: usize) -> u128 {
        (first..first + rows)
            .map(|r| (position_start + r + 1) as u128)
            .sum()
    }

    #[test]
    fn qsa_select_split_covers_the_tile_and_balances_the_work() {
        for (p, n) in [
            (0usize, 2usize),
            (0, 3),
            (0, 5),
            (7, 2),
            (7, 9),
            (1000, 33),
            (0, 4096),
            (0, 4097),
            (1_000_000, 4096),
            (1_000_001, 4097),
        ] {
            let [a, b] = qsa_select_split(n, p);
            assert_eq!(a.0, 0, "P={p} n={n}");
            assert_eq!(a.0 + a.1, b.0, "P={p} n={n}");
            assert_eq!(a.1 + b.1, n, "P={p} n={n}");
            assert!(a.1 >= 1 && b.1 >= 1, "P={p} n={n}");
        }
    }

    /// Small tiles: the split is the smallest cut minimising the larger side (brute force).
    #[test]
    fn qsa_select_split_is_the_smallest_minimax_cut() {
        for p in [0usize, 1, 3, 64, 1000] {
            for n in 2..40usize {
                let cost = |s: usize| qsa_range_cost(p, 0, s).max(qsa_range_cost(p, s, n - s));
                let best = (1..n).min_by_key(|&s| (cost(s), s)).unwrap();
                assert_eq!(qsa_select_split(n, p)[0].1, best, "P={p} n={n}");
            }
        }
    }

    #[test]
    fn qsa_select_split_at_zero_position_favours_the_cheaper_rows() {
        let n = 4096;
        let [a, b] = qsa_select_split(n, 0);
        assert!((2890..=2900).contains(&a.1), "rank 0 rows {}", a.1);
        let (left, right) = (qsa_range_cost(0, a.0, a.1), qsa_range_cost(0, b.0, b.1));
        let total = left + right;
        let bound = total.div_ceil(2) + (n as u128);
        assert!(left.max(right) <= bound, "{left} / {right}");
    }

    #[test]
    fn qsa_select_split_deep_in_the_context_is_near_even() {
        let [a, b] = qsa_select_split(4096, 1_000_000);
        // Rank 0's rows are the cheaper ones by about `n / 2` tokens each, so it takes
        // ~`n^2 / (8 P)` (here ~2) rows more than half.
        assert!((2048..=2051).contains(&a.1), "rank 0 rows {}", a.1);
        assert_eq!(a.1 + b.1, 4096);
    }

    #[test]
    fn qsa_select_split_of_two_rows_is_one_each() {
        for p in [0usize, 5, 1_000_000] {
            assert_eq!(qsa_select_split(2, p), [(0, 1), (1, 1)]);
        }
    }

    /// Host model of the FFN partial contract (the sealed decode MoE of a trunk-shard rank):
    /// `moe_down_combine_top10_batched` accumulates the bf16 chain
    /// `acc = bf16(acc + bf16(bf16(out) * bf16(w)))` over the slots in ascending expert order
    /// from a CLEARED accumulator, non-owned slots hold exactly-zero rows (their expert is the
    /// zero dummy: the down kernel's `(scale * q + zero) * x` over all-zero bytes is `+0`), and
    /// `bf16_scaled_add` folds the rank's K-half of the shared expert as
    /// `bf16(bf16(res) + bf16(bf16(s) * bf16(dot)))`; the fused write then takes
    /// `m = bf16(p0 + p1)` of the two rank partials. This pins (a) zero rows are the identity
    /// of the chain, so a rank's partial is the chain over ITS owned slots only, (b) no slot and
    /// no shared K-half is counted twice or dropped, (c) the partial-sum grouping differs from
    /// the one-card chain only at bf16 rounding level.
    fn bf16r(x: f32) -> f32 {
        let u = x.to_bits();
        f32::from_bits(((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) << 16)
    }

    /// `slots`: (expert, weight, down row element) per slot; ascending expert order, ties by
    /// slot (the kernel's strict `<` scan).
    fn combine_chain(start: f32, slots: &[(i32, f32, f32)]) -> f32 {
        let mut order: Vec<usize> = (0..slots.len()).collect();
        order.sort_by_key(|&r| (slots[r].0, r));
        let mut acc = bf16r(start);
        for r in order {
            let (_, weight, output) = slots[r];
            let product = bf16r(bf16r(output) * bf16r(weight));
            acc = bf16r(acc + product);
        }
        acc
    }

    fn shared_add(residual: f32, dot: f32, scalar: f32) -> f32 {
        bf16r(bf16r(residual) + bf16r(bf16r(scalar) * bf16r(dot)))
    }

    #[test]
    fn ffn_partial_rows_chain_and_fold_contract() {
        // The dummy expert's down row: scale = zero = 0 (f16 zero bytes), any nibbles, any finite x.
        for nibble in 0..16u32 {
            for x in [0.0f32, 1.5, -3.25e4, 1e-30] {
                assert_eq!(((0.0f32 * nibble as f32) + 0.0) * x, 0.0);
            }
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        let mut worst = 0.0f64;
        for trial in 0..4000 {
            // 10 distinct routed experts; owners by `e % 2` (static) or dealt alternately over
            // the ascending routed set (balanced), both exactly one owner per expert.
            let mut experts = [0i32; 10];
            let mut used = std::collections::BTreeSet::new();
            for slot in experts.iter_mut() {
                loop {
                    let e = (draw() * 512.0) as i32;
                    if used.insert(e) {
                        *slot = e;
                        break;
                    }
                }
            }
            let mut sorted = experts.to_vec();
            sorted.sort_unstable();
            let owner = |e: i32| -> usize {
                if trial % 2 == 0 {
                    (e % 2) as usize
                } else {
                    sorted.iter().position(|&x| x == e).unwrap() % 2
                }
            };
            let weights: [f32; 10] = std::array::from_fn(|_| bf16r(draw() * 0.3));
            let outputs: [f32; 10] = std::array::from_fn(|_| (draw() - 0.5) * 8.0);
            let selector = bf16r(1.0 / (1.0 + (-(draw() - 0.5) * 6.0).exp()));
            let (d0, d1) = ((draw() - 0.5) * 6.0, (draw() - 0.5) * 6.0);
            // One card (EP2 / reference): chain over all slots, then the full shared down
            // (the K-halves accumulated in f32, one rounding on the way out of the GEMV).
            let all_slots: Vec<(i32, f32, f32)> = (0..10)
                .map(|r| (experts[r], weights[r], outputs[r]))
                .collect();
            let one_card = shared_add(combine_chain(0.0, &all_slots), d0 + d1, selector);
            // TP2 ranks: the kernel keeps every slot, a non-owned one with an exactly-zero row.
            let mut partial = [0.0f32; 2];
            for rank in 0..2 {
                let with_zero_rows: Vec<(i32, f32, f32)> = (0..10)
                    .map(|r| {
                        let row = if owner(experts[r]) == rank {
                            outputs[r]
                        } else {
                            0.0
                        };
                        (experts[r], weights[r], row)
                    })
                    .collect();
                let owned_only: Vec<(i32, f32, f32)> = (0..10)
                    .filter(|&r| owner(experts[r]) == rank)
                    .map(|r| (experts[r], weights[r], outputs[r]))
                    .collect();
                let zero_chain = combine_chain(0.0, &with_zero_rows);
                let skip_chain = combine_chain(0.0, &owned_only);
                assert_eq!(
                    zero_chain.to_bits(),
                    skip_chain.to_bits(),
                    "zero rows are the chain identity"
                );
                partial[rank] = shared_add(zero_chain, if rank == 0 { d0 } else { d1 }, selector);
            }
            let m = bf16r(partial[0] + partial[1]);
            // Bound: the two groupings differ by a handful of bf16 roundings of terms no
            // larger than the sum of the magnitudes involved.
            let magnitude: f64 = (0..10)
                .map(|r| (bf16r(outputs[r]) as f64 * weights[r] as f64).abs())
                .sum::<f64>()
                + (selector as f64 * (d0.abs() + d1.abs()) as f64);
            let diff = (m as f64 - one_card as f64).abs();
            worst = worst.max(diff / magnitude.max(1e-30));
            assert!(
                diff <= 32.0 * magnitude * 2f64.powi(-8),
                "trial {trial}: m {m} one-card {one_card} magnitude {magnitude}"
            );
        }
        assert!(
            worst > 0.0,
            "the two groupings must differ at rounding level somewhere"
        );
    }
}
