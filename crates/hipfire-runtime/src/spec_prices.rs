// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-load, in-memory speculation prices: the measured wall time of one
//! settled AR token and of one settled speculative window, by context
//! bucket.
//!
//! Nothing here is shipped, persisted or keyed by model, prompt or card: a
//! [`LoadPrices`] is allocated once with its speculator and dies with the
//! load. Fixed storage, no allocation after construction, pure CPU.
//!
//! The cell is the lookup controller's load-scoped price cell
//! ([`PriceCell`]: median of the last [`PRICE_WINDOW`] kept samples) and the
//! context buckets are its [`PRICE_BUCKET_EDGES`]; `spec_lookup`'s
//! `LookupPriceBook` uses the same cell.
//!
//! Wall time reaches a qwen35 DFlash request's speculate-vs-AR decision only
//! through these cells: the floor decides on token counts and a bucket's
//! [`PinnedPrices`]. Precedence, highest first: `HIPFIRE_SPEC_RETIRE` set
//! (the count path) → the pinned prices of the request's start bucket,
//! seeded or not → a seeded request without a pin: the drafter's
//! `retire_break_even()`, else `DEFAULT_BREAK_EVEN` → an unseeded request
//! without a pin: no verdict until a pin lands.

/// Samples kept per price cell (median of the last `PRICE_WINDOW`).
pub const PRICE_WINDOW: usize = 8;
/// Context buckets: `<=2K, <=8K, <=16K, <=32K, >32K` (the H0 trace buckets).
pub const PRICE_BUCKET_EDGES: [usize; 4] = [2048, 8192, 16384, 32768];
/// Number of context buckets.
pub const PRICE_BUCKETS: usize = PRICE_BUCKET_EDGES.len() + 1;
/// Widest speculative window [`LoadPrices`] prices, in verified rows (seed
/// included): the widest DFlash block.
pub const LOAD_PRICE_WINDOW_ROWS: usize = 16;
/// Kept AR samples a bucket needs before its [`PinnedPrices`] freeze.
pub const PIN_MIN_AR_SAMPLES: usize = 4;
/// Kept full-block window samples a bucket needs before its
/// [`PinnedPrices`] freeze (a full cell: the window's settled count is the
/// noisier side of a fixed-window floor's break-even).
pub const PIN_MIN_WINDOW_SAMPLES: usize = PRICE_WINDOW;

/// Context bucket of absolute position `ctx`.
pub fn price_bucket(ctx: usize) -> usize {
    PRICE_BUCKET_EDGES
        .iter()
        .position(|&edge| ctx <= edge)
        .unwrap_or(PRICE_BUCKET_EDGES.len())
}

/// One load-scoped price: the median of the last [`PRICE_WINDOW`] kept
/// samples, fixed storage.
#[derive(Clone, Copy, Debug)]
pub struct PriceCell {
    discarded_first: bool,
    len: usize,
    head: usize,
    samples: [f64; PRICE_WINDOW],
}

impl PriceCell {
    pub const EMPTY: Self = Self {
        discarded_first: false,
        len: 0,
        head: 0,
        samples: [0.0; PRICE_WINDOW],
    };

    /// Record `us`, discarding the cell's first observation (it carries
    /// first-bind / graph-capture cost).
    pub fn observe(&mut self, us: f64) {
        if !self.discarded_first {
            self.discarded_first = true;
            return;
        }
        self.keep(us);
    }

    /// Record `us` as a kept sample (no discard-first rule: the caller
    /// already excluded samples that pay one-time costs).
    pub fn keep(&mut self, us: f64) {
        self.samples[self.head] = us;
        self.head = (self.head + 1) % PRICE_WINDOW;
        self.len = (self.len + 1).min(PRICE_WINDOW);
    }

    /// Kept samples (at most [`PRICE_WINDOW`]).
    pub fn kept(&self) -> usize {
        self.len
    }

    pub fn median(&self) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let mut sorted = self.samples;
        let live = &mut sorted[..self.len];
        live.sort_by(f64::total_cmp);
        Some(if self.len % 2 == 1 {
            live[self.len / 2]
        } else {
            (live[self.len / 2 - 1] + live[self.len / 2]) / 2.0
        })
    }
}

/// Opaque equality key of the executor a load's AR token runs on. The driver
/// builds it from the replay controller's state and transport; a change
/// clears the AR prices and the pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArRouteKey(u16);

impl ArRouteKey {
    /// `state` is the driver's code of the replay state, `pm4` whether the
    /// AR token uses the PM4 transport.
    pub const fn new(state: u8, pm4: bool) -> Self {
        Self(((state as u16) << 1) | pm4 as u16)
    }
}

/// A bucket's frozen prices: the AR token and the speculative window by
/// verified rows (`0.0` = no price at that row count). A seeded request
/// decides on these alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinnedPrices {
    pub ar_us: f32,
    pub window_us: [f32; LOAD_PRICE_WINDOW_ROWS + 1],
    /// Rows of the load's full block.
    pub full_rows: usize,
}

impl PinnedPrices {
    /// Window price at `rows`; a row count with no pinned price takes the
    /// full block's.
    pub fn window(&self, rows: usize) -> f32 {
        match self.window_us.get(rows) {
            Some(&us) if us > 0.0 => us,
            _ => self.window_us[self.full_rows],
        }
    }
}

/// The per-load price cache of one speculator.
#[derive(Clone, Debug)]
pub struct LoadPrices {
    route: Option<ArRouteKey>,
    /// Wall us of one settled AR token, by bucket.
    ar: [PriceCell; PRICE_BUCKETS],
    /// Wall us of one settled speculative window, by verified rows and bucket.
    window: [[PriceCell; PRICE_BUCKETS]; LOAD_PRICE_WINDOW_ROWS + 1],
    pinned: [Option<PinnedPrices>; PRICE_BUCKETS],
    /// Rows of the load's full block (`1..=LOAD_PRICE_WINDOW_ROWS`).
    full_rows: usize,
}

impl LoadPrices {
    /// An empty cache for a load whose full block verifies `full_rows` rows.
    pub fn new(full_rows: usize) -> Self {
        Self {
            route: None,
            ar: [PriceCell::EMPTY; PRICE_BUCKETS],
            window: [[PriceCell::EMPTY; PRICE_BUCKETS]; LOAD_PRICE_WINDOW_ROWS + 1],
            pinned: [None; PRICE_BUCKETS],
            full_rows: full_rows.clamp(1, LOAD_PRICE_WINDOW_ROWS),
        }
    }

    /// Install the AR executor of the coming request. A different key than
    /// the last clears the AR prices and the pins; returns whether it did.
    pub fn set_route(&mut self, key: ArRouteKey) -> bool {
        if self.route == Some(key) {
            return false;
        }
        let cleared = self.route.is_some();
        self.route = Some(key);
        self.ar = [PriceCell::EMPTY; PRICE_BUCKETS];
        self.pinned = [None; PRICE_BUCKETS];
        cleared
    }

    pub fn route(&self) -> Option<ArRouteKey> {
        self.route
    }

    /// One settled AR token of `us` at position `ctx`. The caller records
    /// only the third and later AR tokens of a run (the first two after a
    /// speculative window still pay capture or graph re-capture), so the AR
    /// cell keeps every sample.
    pub fn record_ar(&mut self, ctx: usize, us: f32) {
        if !(us.is_finite() && us > 0.0) {
            return;
        }
        let bucket = price_bucket(ctx);
        self.ar[bucket].keep(f64::from(us));
        self.try_pin(bucket);
    }

    /// One steady speculative window of `rows` verified rows and `us` at
    /// position `ctx` (the cell discards its first observation). Rows outside
    /// `1..=LOAD_PRICE_WINDOW_ROWS` are ignored.
    pub fn record_window(&mut self, rows: usize, ctx: usize, us: f32) {
        if !(1..=LOAD_PRICE_WINDOW_ROWS).contains(&rows) || !(us.is_finite() && us > 0.0) {
            return;
        }
        let bucket = price_bucket(ctx);
        self.window[rows][bucket].observe(f64::from(us));
        self.try_pin(bucket);
    }

    fn try_pin(&mut self, bucket: usize) {
        if self.pinned[bucket].is_some()
            || self.ar[bucket].kept() < PIN_MIN_AR_SAMPLES
            || self.window[self.full_rows][bucket].kept() < PIN_MIN_WINDOW_SAMPLES
        {
            return;
        }
        let (Some(ar), Some(_)) = (
            self.ar[bucket].median(),
            self.window[self.full_rows][bucket].median(),
        ) else {
            return;
        };
        let mut window_us = [0.0f32; LOAD_PRICE_WINDOW_ROWS + 1];
        for (rows, us) in window_us.iter_mut().enumerate().skip(1) {
            *us = self.window[rows][bucket].median().unwrap_or(0.0) as f32;
        }
        self.pinned[bucket] = Some(PinnedPrices {
            ar_us: ar as f32,
            window_us,
            full_rows: self.full_rows,
        });
    }

    /// AR price at `ctx`: its own bucket, else the nearest bucket with a
    /// price (ties: the shorter context).
    pub fn ar_price(&self, ctx: usize) -> Option<f32> {
        let own = price_bucket(ctx);
        (0..PRICE_BUCKETS)
            .filter_map(|bucket| {
                self.ar[bucket]
                    .median()
                    .map(|us| (bucket.abs_diff(own), bucket, us))
            })
            .min_by_key(|&(distance, bucket, _)| (distance, bucket))
            .map(|(_, _, us)| us as f32)
    }

    /// Window price of `rows` verified rows at `ctx` (own bucket only).
    pub fn window_price(&self, rows: usize, ctx: usize) -> Option<f32> {
        if !(1..=LOAD_PRICE_WINDOW_ROWS).contains(&rows) {
            return None;
        }
        self.window[rows][price_bucket(ctx)]
            .median()
            .map(|us| us as f32)
    }

    /// The frozen prices of `ctx`'s bucket, once pinned.
    pub fn pinned(&self, ctx: usize) -> Option<PinnedPrices> {
        self.pinned[price_bucket(ctx)]
    }

    /// Rows of the load's full block.
    pub fn full_rows(&self) -> usize {
        self.full_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_edges_are_inclusive_upper_bounds() {
        assert_eq!(price_bucket(0), 0);
        assert_eq!(price_bucket(2048), 0);
        assert_eq!(price_bucket(2049), 1);
        assert_eq!(price_bucket(8192), 1);
        assert_eq!(price_bucket(8193), 2);
        assert_eq!(price_bucket(16384), 2);
        assert_eq!(price_bucket(32768), 3);
        assert_eq!(price_bucket(32769), 4);
        assert_eq!(price_bucket(usize::MAX), PRICE_BUCKETS - 1);
    }

    #[test]
    fn cell_discards_first_observation_and_keeps_every_kept_sample() {
        let mut cell = PriceCell::EMPTY;
        cell.observe(1000.0);
        assert_eq!(cell.median(), None);
        cell.observe(10.0);
        cell.observe(30.0);
        assert_eq!(cell.median(), Some(20.0));
        let mut kept = PriceCell::EMPTY;
        kept.keep(7.0);
        assert_eq!(kept.median(), Some(7.0));
        for us in 0..20 {
            kept.keep(f64::from(us));
        }
        assert_eq!(kept.kept(), PRICE_WINDOW);
        // Last 8 kept: 12..=19.
        assert_eq!(kept.median(), Some(15.5));
    }

    #[test]
    fn route_key_change_clears_ar_and_pinned_but_not_windows() {
        let mut prices = LoadPrices::new(16);
        assert!(!prices.set_route(ArRouteKey::new(1, true)));
        for _ in 0..PIN_MIN_AR_SAMPLES {
            prices.record_ar(100, 20_000.0);
        }
        for _ in 0..=PIN_MIN_WINDOW_SAMPLES {
            prices.record_window(16, 100, 70_000.0);
        }
        assert!(prices.pinned(100).is_some());
        assert!(!prices.set_route(ArRouteKey::new(1, true)));
        assert!(prices.pinned(100).is_some());
        assert!(prices.set_route(ArRouteKey::new(1, false)));
        assert_eq!(prices.ar_price(100), None);
        assert!(prices.pinned(100).is_none());
        assert_eq!(prices.window_price(16, 100), Some(70_000.0));
        assert_ne!(ArRouteKey::new(2, false), ArRouteKey::new(1, true));
    }

    #[test]
    fn pin_needs_4_ar_and_8_full_block_window_samples() {
        let mut prices = LoadPrices::new(16);
        prices.set_route(ArRouteKey::new(2, true));
        for _ in 0..PIN_MIN_AR_SAMPLES {
            prices.record_ar(10, 20_000.0);
        }
        // The window cell discards its first observation: 1 + 7 kept.
        for _ in 0..PIN_MIN_WINDOW_SAMPLES {
            prices.record_window(16, 10, 80_000.0);
        }
        assert!(prices.pinned(10).is_none(), "7 kept window samples");
        prices.record_window(16, 10, 80_000.0);
        assert_eq!(PIN_MIN_WINDOW_SAMPLES, 8);
        assert!(prices.pinned(10).is_some(), "8 kept window samples");
    }

    #[test]
    fn pin_freezes_once_and_falls_back_to_the_full_block() {
        let mut prices = LoadPrices::new(16);
        prices.set_route(ArRouteKey::new(0, false));
        for _ in 0..PIN_MIN_AR_SAMPLES - 1 {
            prices.record_ar(10, 20_000.0);
        }
        for _ in 0..=PIN_MIN_WINDOW_SAMPLES {
            prices.record_window(16, 10, 80_000.0);
        }
        assert!(prices.pinned(10).is_none(), "AR cell short of samples");
        prices.record_window(8, 10, 1.0);
        prices.record_window(8, 10, 50_000.0);
        prices.record_ar(10, 20_000.0);
        let pinned = prices.pinned(10).expect("pinned");
        assert_eq!(pinned.ar_us, 20_000.0);
        assert_eq!(pinned.window(16), 80_000.0);
        assert_eq!(pinned.window(8), 50_000.0);
        assert_eq!(pinned.window(5), 80_000.0, "no price at 5 rows: full block");
        assert_eq!(pinned.window(99), 80_000.0);
        // Later samples move the cells, never the pin.
        for _ in 0..PRICE_WINDOW {
            prices.record_ar(10, 90_000.0);
            prices.record_window(16, 10, 10_000.0);
        }
        assert_eq!(prices.pinned(10), Some(pinned));
        assert_eq!(prices.ar_price(10), Some(90_000.0));
        // Other buckets pin on their own.
        assert!(prices.pinned(9000).is_none());
    }

    #[test]
    fn ar_price_takes_the_nearest_filled_bucket() {
        let mut prices = LoadPrices::new(16);
        prices.record_ar(20_000, 30.0);
        assert_eq!(prices.ar_price(20_000), Some(30.0));
        assert_eq!(prices.ar_price(100), Some(30.0));
        prices.record_ar(100, 10.0);
        assert_eq!(prices.ar_price(5000), Some(10.0), "nearer bucket");
        assert_eq!(prices.ar_price(9000), Some(30.0));
        prices.record_ar(100, f32::NAN);
        prices.record_ar(100, 0.0);
        assert_eq!(prices.ar_price(100), Some(10.0), "invalid samples ignored");
    }

    #[test]
    fn window_rows_outside_the_table_are_ignored() {
        let mut prices = LoadPrices::new(99);
        assert_eq!(prices.full_rows(), LOAD_PRICE_WINDOW_ROWS);
        prices.record_window(0, 10, 5.0);
        prices.record_window(17, 10, 5.0);
        assert_eq!(prices.window_price(0, 10), None);
        assert_eq!(prices.window_price(17, 10), None);
    }

    #[test]
    fn lookup_price_book_keeps_its_behaviour_on_the_moved_cell() {
        use crate::spec_lookup::{LookupKind, LookupPriceBook};
        let mut book = LookupPriceBook::new();
        book.observe(LookupKind::Native, 16, 100, 9_999.0);
        assert_eq!(
            book.price(LookupKind::Native, 16, 100),
            None,
            "first discarded"
        );
        book.observe(LookupKind::Native, 16, 100, 10.0);
        book.observe(LookupKind::Native, 16, 2048, 30.0);
        book.observe(LookupKind::Native, 16, 2049, 77.0);
        assert_eq!(book.price(LookupKind::Native, 16, 0), Some(20.0));
        assert_eq!(
            book.price(LookupKind::Native, 16, 4000),
            None,
            "own cell discards"
        );
        book.observe(LookupKind::Native, 16, 4000, 50.0);
        assert_eq!(book.price(LookupKind::Native, 16, 8192), Some(50.0));
        book.observe(LookupKind::Native, 17, 100, 5.0);
        book.observe(LookupKind::Native, 0, 100, 5.0);
        assert_eq!(book.price(LookupKind::Native, 17, 100), None);
        assert_eq!(book.price(LookupKind::Native, 0, 100), None);
    }
}
