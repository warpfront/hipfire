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
            SlotOffsets { req_seq: 0, resp_seq: 64, error: 128, x_rot, ids, weights, cold_out },
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

#[cfg(test)]
mod tests {
    use super::*;

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
            let lines: Vec<usize> = [s.req_seq, s.resp_seq, s.error].iter().map(|o| o / 64).collect();
            assert_eq!(lines.len(), 3);
            assert!(lines[0] != lines[1] && lines[1] != lines[2] && lines[0] != lines[2]);
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
