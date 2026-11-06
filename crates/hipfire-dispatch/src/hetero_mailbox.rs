//! Mailbox layout of the device-side hetero split (design v3).
//!
//! One pinned host allocation in the expert device's (Strix Halo) RAM holds one
//! slot per trunk layer. The primary (gfx1201) writes a layer's rotated
//! activation, top-k ids and weights into its slot and then the request word;
//! the Strix responder writes the cold slot rows and then the response word; the
//! primary waits on the response word and installs the rows. Device kernels on
//! both GPUs address the slot by these offsets.
//!
//! Every word (request, response, error) sits on its own 64-byte line, so the
//! word one device polls never shares a line with one the other device writes.
//! Arrays are 256-byte aligned. Sequence numbers come from [`seq_for`]; 0 never
//! occurs, so a zeroed mailbox reads as "never written".

/// Shape of one mailbox: `layers` slots of `max_rows` rows each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MailboxGeometry {
    pub layers: usize,
    pub max_rows: usize,
    pub top_k: usize,
    pub dim: usize,
}

/// Byte offsets of one layer's slot fields from the mailbox base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotOffsets {
    pub req_seq: usize,
    pub resp_seq: usize,
    pub error: usize,
    /// Rows of the current request (1..=`max_rows`), written with the route
    /// before the request word.
    pub rows: usize,
    pub x_rot: usize,
    pub ids: usize,
    pub weights: usize,
    pub cold_out: usize,
}

const ARRAY_ALIGN: usize = 256;

fn align(x: usize) -> usize {
    x.div_ceil(ARRAY_ALIGN) * ARRAY_ALIGN
}

impl MailboxGeometry {
    /// Offsets inside one slot: the three words on lines 0, 1 and 2, then the
    /// arrays from byte 256, each 256-aligned.
    fn local(&self) -> (SlotOffsets, usize) {
        let slots = self.max_rows * self.top_k;
        let x_rot = ARRAY_ALIGN;
        let ids = align(x_rot + self.max_rows * self.dim * 4);
        let weights = align(ids + slots * 4);
        let cold_out = align(weights + slots * 4);
        let end = align(cold_out + slots * self.dim * 4);
        (
            SlotOffsets { req_seq: 0, resp_seq: 64, error: 128, rows: 192, x_rot, ids, weights, cold_out },
            end,
        )
    }

    pub fn slot(&self, layer: usize) -> SlotOffsets {
        let (o, bytes) = self.local();
        let base = layer * bytes;
        SlotOffsets {
            req_seq: base + o.req_seq,
            resp_seq: base + o.resp_seq,
            error: base + o.error,
            rows: base + o.rows,
            x_rot: base + o.x_rot,
            ids: base + o.ids,
            weights: base + o.weights,
            cold_out: base + o.cold_out,
        }
    }

    pub fn slot_bytes(&self) -> usize {
        self.local().1
    }

    pub fn total_bytes(&self) -> usize {
        self.layers * self.slot_bytes()
    }
}

/// Request/response sequence of `layer` in token `token`: `token * layers +
/// layer + 1`, wrapping. The one wrapped value that would be 0 maps to 1, so 0
/// keeps meaning "never written" (a collision once every 2^32 sequences, far
/// beyond any wait window).
pub fn seq_for(token: u32, layer: u32, layers: u32) -> u32 {
    match token.wrapping_mul(layers).wrapping_add(layer).wrapping_add(1) {
        0 => 1,
        v => v,
    }
}

/// Most sub-chunks one prefill chunk may be split into.
pub const MAX_SUB_CHUNKS: usize = 8;

/// Sub-chunks a full chunk is split into. Measured (R9700 + gfx1151, USB4,
/// 2048 rows): 1 -> 21.6 ms per layer, 2 -> 16.1, 4 -> 15.4; 8 falls off the
/// grouped-row floor below.
pub const PREFILL_PIPELINE_SUB_CHUNKS: usize = 4;

/// Smallest block the expert device's grouped prefill arm admits: below 512
/// rows the QT44/QT53 route takes per-slot kernels (gfx1151: 5.6 ms at 32 rows,
/// 11.7 ms at 511, 1.5 ms at 512), so no block is split below it.
pub const MIN_SUB_CHUNK_ROWS: usize = 512;

/// The one prefill slot (layers run sequentially, so one slot serves them
/// all), placed after the decode slots in the same mailbox allocation.
///
/// A chunk is split into up to `sub_chunks` row blocks, each with its own
/// request/response pair, so the primary's out-copy of block `i + 1`, the
/// expert device's compute of block `i` and the copy back of block `i - 1`
/// overlap. Rows travel as F16 and the partial as BF16 (the BF16 round-trip
/// combine makes every partial element BF16-exact, so that leg is lossless).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillGeometry {
    pub max_rows: usize,
    pub top_k: usize,
    pub dim: usize,
    pub sub_chunks: usize,
}

/// One sub-chunk's words: the request and response on their own lines, and
/// the request's layer, first row, row count and row format (four u32 the
/// primary writes before the request word) on a third.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubChunkWords {
    pub req_seq: usize,
    pub resp_seq: usize,
    pub layer: usize,
    pub row_off: usize,
    pub rows: usize,
    /// [`ROWS_F16`] or [`ROWS_INT4`].
    pub format: usize,
}

/// Row format of a sub-chunk: F16 rows at `x_norm + row_off * dim * 2`.
pub const ROWS_F16: u32 = 0;
/// Row format of a sub-chunk: the symmetric IU4 producer's sidecar of the
/// block's rows (block-major, `dim / 128` records of 72 B per row) at
/// `x_norm + row_off * int4_row_bytes(dim)`.
pub const ROWS_INT4: u32 = 1;

/// Bytes per row of the int4 activation sidecar.
pub const fn int4_row_bytes(dim: usize) -> usize {
    dim / 128 * 72
}

/// Byte offsets of the prefill slot fields from the mailbox base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillOffsets {
    /// Shared by every sub-chunk: the first timed-out wait records its
    /// sequence here, and the primary skips the whole chunk's add.
    pub error: usize,
    /// Written once by the expert device at start: the smallest block it
    /// accepts as [`ROWS_INT4`] (0 = never; send F16).
    pub int4_min_rows: usize,
    /// `sub[..sub_chunks]` are live.
    pub sub: [SubChunkWords; MAX_SUB_CHUNKS],
    /// The chunk's rows per sub-chunk format: normed F16 rows (the expert
    /// device runs its own basis) or the int4 sidecar.
    pub x_norm: usize,
    /// `rows x top_k` i32 route ids and F32 weights.
    pub ids: usize,
    pub weights: usize,
    /// The expert device's weighted cold partial, `rows x dim` BF16.
    pub partial: usize,
}

const LINE: usize = 64;

impl PrefillGeometry {
    fn check(&self) {
        assert!(
            (1..=MAX_SUB_CHUNKS).contains(&self.sub_chunks),
            "prefill sub_chunks {} outside 1..={MAX_SUB_CHUNKS}",
            self.sub_chunks
        );
    }

    /// Local layout (from a 256-aligned start): the error word on line 0, the
    /// int4 floor on line 1, each sub-chunk's request, response and meta on
    /// three lines, then 256-aligned arrays; returns the offsets and the size.
    fn local(&self) -> (PrefillOffsets, usize) {
        self.check();
        let mut sub = [SubChunkWords::default(); MAX_SUB_CHUNKS];
        for (i, w) in sub.iter_mut().enumerate().take(self.sub_chunks) {
            let b = LINE * (2 + 3 * i);
            let m = b + 2 * LINE;
            *w = SubChunkWords { req_seq: b, resp_seq: b + LINE, layer: m, row_off: m + 4, rows: m + 8, format: m + 12 };
        }
        let slots = self.max_rows * self.top_k;
        let x_norm = align(LINE * (2 + 3 * self.sub_chunks));
        let row_area = (self.max_rows * self.dim * 2).max(self.max_rows * int4_row_bytes(self.dim));
        let ids = align(x_norm + row_area);
        let weights = align(ids + slots * 4);
        let partial = align(weights + slots * 4);
        let end = align(partial + self.max_rows * self.dim * 2);
        (PrefillOffsets { error: 0, int4_min_rows: LINE, sub, x_norm, ids, weights, partial }, end)
    }

    /// Offsets from the mailbox base when the slot starts at (or after)
    /// `base`, the end of the decode slots.
    pub fn offsets(&self, base: usize) -> PrefillOffsets {
        let b = align(base);
        let (o, _) = self.local();
        let mut sub = o.sub;
        for w in sub.iter_mut().take(self.sub_chunks) {
            *w = SubChunkWords {
                req_seq: b + w.req_seq,
                resp_seq: b + w.resp_seq,
                layer: b + w.layer,
                row_off: b + w.row_off,
                rows: b + w.rows,
                format: b + w.format,
            };
        }
        PrefillOffsets {
            error: b + o.error,
            int4_min_rows: b + o.int4_min_rows,
            sub,
            x_norm: b + o.x_norm,
            ids: b + o.ids,
            weights: b + o.weights,
            partial: b + o.partial,
        }
    }

    /// Bytes of the slot (allocate `align(decode total) + bytes()`).
    pub fn bytes(&self) -> usize {
        self.local().1
    }

    /// How many sub-chunks a `rows`-row chunk uses: at most
    /// [`PREFILL_PIPELINE_SUB_CHUNKS`] (and the slot's count), never a block
    /// below [`MIN_SUB_CHUNK_ROWS`] unless the whole chunk is.
    pub fn sub_chunks_for(&self, rows: usize) -> usize {
        self.check();
        (rows / MIN_SUB_CHUNK_ROWS).clamp(1, PREFILL_PIPELINE_SUB_CHUNKS.min(self.sub_chunks))
    }

    /// `(first row, rows)` of sub-chunk `i` of a `rows`-row chunk: near-equal
    /// blocks, the larger ones first; trailing blocks may be empty (no request).
    pub fn sub_range(&self, rows: usize, i: usize) -> (usize, usize) {
        self.check();
        let (q, r) = (rows / self.sub_chunks, rows % self.sub_chunks);
        let n = q + usize::from(i < r);
        let off = i * q + i.min(r);
        (off, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: PrefillGeometry = PrefillGeometry { max_rows: 2048, top_k: 10, dim: 2560, sub_chunks: 4 };

    fn prefill_ranges(base: usize) -> Vec<(usize, usize)> {
        let o = P.offsets(base);
        let slots = P.max_rows * P.top_k;
        let mut v = vec![(o.error, 4), (o.int4_min_rows, 4)];
        for s in &o.sub[..P.sub_chunks] {
            v.extend([(s.req_seq, 4), (s.resp_seq, 4), (s.layer, 4), (s.row_off, 4), (s.rows, 4), (s.format, 4)]);
        }
        v.extend([
            (o.x_norm, P.max_rows * P.dim * 2),
            (o.ids, slots * 4),
            (o.weights, slots * 4),
            (o.partial, P.max_rows * P.dim * 2),
        ]);
        v
    }

    #[test]
    fn prefill_slot_follows_the_decode_slots_disjoint_and_aligned() {
        let base = G.total_bytes();
        let mut all: Vec<(usize, usize)> = (0..G.layers).flat_map(|l| ranges(&G, l)).collect();
        let pre = prefill_ranges(base);
        for &(off, _) in &pre[pre.len() - 4..] {
            assert_eq!(off % 256, 0);
        }
        assert!(pre.iter().all(|&(o, _)| o >= base), "prefill slot overlaps the decode slots");
        all.extend(pre);
        all.sort();
        for w in all.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "overlap {:?} {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn prefill_words_have_their_own_cache_lines() {
        let o = P.offsets(0);
        // The error word (primary-written) and the int4 floor (expert-written) apart.
        let mut lines = vec![o.error / 64, o.int4_min_rows / 64];
        for s in &o.sub[..P.sub_chunks] {
            lines.extend([s.req_seq / 64, s.resp_seq / 64, s.layer / 64]);
            for w in [s.row_off, s.rows, s.format] {
                assert_eq!(w / 64, s.layer / 64, "the primary-written meta shares one line");
            }
        }
        let mut sorted = lines.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), lines.len(), "two words share a line: {lines:?}");
        for &(off, _) in &prefill_ranges(0)[prefill_ranges(0).len() - 4..] {
            assert!(!lines.contains(&(off / 64)), "an array shares a word's line");
        }
    }

    #[test]
    fn the_row_area_holds_either_format() {
        let o = P.offsets(0);
        let int4 = P.max_rows * int4_row_bytes(P.dim);
        assert_eq!(int4_row_bytes(2560), 1440);
        assert!(o.ids - o.x_norm >= (P.max_rows * P.dim * 2).max(int4));
    }

    #[test]
    fn prefill_bytes_cover_the_bf16_partial() {
        let o = P.offsets(0);
        assert!(P.bytes() >= o.partial + P.max_rows * P.dim * 2);
    }

    #[test]
    fn sub_chunks_tile_the_rows_in_order() {
        for rows in [1usize, 3, 4, 37, 511, 512, 2000, 2048] {
            let mut next = 0;
            for i in 0..P.sub_chunks {
                let (off, n) = P.sub_range(rows, i);
                if n == 0 {
                    continue;
                }
                assert_eq!(off, next, "rows {rows} sub {i}");
                next += n;
            }
            assert_eq!(next, rows, "rows {rows} not covered");
            // Leading sub-chunks are the larger ones (the expert device starts sooner on the first).
            let sizes: Vec<usize> = (0..P.sub_chunks).map(|i| P.sub_range(rows, i).1).collect();
            assert!(sizes.windows(2).all(|w| w[0] >= w[1]), "rows {rows}: {sizes:?}");
        }
    }

    #[test]
    fn sub_chunks_never_drop_below_the_grouped_row_floor() {
        let g = PrefillGeometry { sub_chunks: MAX_SUB_CHUNKS, ..P };
        for rows in [1usize, 37, 511, 512, 1023, 1024, 1525, 2047, 2048] {
            let s = g.sub_chunks_for(rows);
            assert!((1..=PREFILL_PIPELINE_SUB_CHUNKS).contains(&s), "rows {rows}: {s}");
            let split = PrefillGeometry { sub_chunks: s, ..g };
            let smallest = (0..s).map(|i| split.sub_range(rows, i).1).min().unwrap();
            // Every block keeps the grouped arm unless the whole chunk is below it.
            assert!(smallest >= MIN_SUB_CHUNK_ROWS || s == 1, "rows {rows}: {s} blocks, smallest {smallest}");
        }
        assert_eq!(g.sub_chunks_for(2048), PREFILL_PIPELINE_SUB_CHUNKS);
        assert_eq!(g.sub_chunks_for(1024), 2);
        assert_eq!(g.sub_chunks_for(1023), 1);
        assert_eq!(PrefillGeometry { sub_chunks: 2, ..P }.sub_chunks_for(2048), 2, "never above the slot's count");
    }

    #[test]
    fn sub_chunk_count_is_bounded() {
        assert!(PrefillGeometry { sub_chunks: MAX_SUB_CHUNKS, ..P }.bytes() > 0);
        assert!(std::panic::catch_unwind(|| PrefillGeometry { sub_chunks: MAX_SUB_CHUNKS + 1, ..P }.bytes()).is_err());
        assert!(std::panic::catch_unwind(|| PrefillGeometry { sub_chunks: 0, ..P }.bytes()).is_err());
    }

    const G: MailboxGeometry = MailboxGeometry { layers: 48, max_rows: 1, top_k: 10, dim: 2560 };

    fn ranges(g: &MailboxGeometry, l: usize) -> Vec<(usize, usize)> {
        let s = g.slot(l);
        let slots = g.max_rows * g.top_k;
        vec![
            (s.req_seq, 4),
            (s.resp_seq, 4),
            (s.error, 4),
            (s.x_rot, g.max_rows * g.dim * 4),
            (s.ids, slots * 4),
            (s.weights, slots * 4),
            (s.cold_out, slots * g.dim * 4),
        ]
    }

    #[test]
    fn slots_are_aligned_and_disjoint() {
        let mut all: Vec<(usize, usize)> = (0..G.layers).flat_map(|l| ranges(&G, l)).collect();
        for &(off, _) in &all[3..7] {
            assert_eq!(off % 256, 0);
        }
        all.sort();
        for w in all.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "overlap {:?} {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn words_have_their_own_cache_lines() {
        for l in [0, 1, 47] {
            let s = G.slot(l);
            let lines: Vec<usize> = [s.req_seq, s.resp_seq, s.error, s.rows].iter().map(|o| o / 64).collect();
            let mut distinct = lines.clone();
            distinct.dedup();
            assert_eq!(distinct.len(), 4, "the slot's words share a line: {lines:?}");
            for &(off, _) in &ranges(&G, l)[3..] {
                assert!(!lines.contains(&(off / 64)), "array shares a word's line");
            }
        }
    }

    #[test]
    fn seq_is_never_zero_and_wraps() {
        for t in [0u32, 1, 7, 1_000_000, u32::MAX - 1, u32::MAX] {
            for l in 0..48 {
                assert_ne!(seq_for(t, l, 48), 0, "t {t} l {l}");
            }
        }
        assert_ne!(seq_for(u32::MAX, 47, 48), seq_for(u32::MAX, 46, 48));
        assert_eq!(seq_for(0, 0, 48), 1);
    }

    #[test]
    fn total_bytes_covers_last_slot() {
        let last = ranges(&G, G.layers - 1);
        let end = last.iter().map(|&(o, n)| o + n).max().unwrap();
        assert!(G.total_bytes() >= end);
        assert_eq!(G.total_bytes(), G.layers * G.slot_bytes());
    }
}
