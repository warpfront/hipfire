// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotSnapshot — one session's GPU state, captured so its slot can be reused.
//
// The swap unit is {KV slab, DeltaNet state, seq_len, tokens} — all four or
// none. DeltaNet state is easy to miss: it lives outside the KV arena and is
// fixed-size, so a KV-only implementation passes a short smoke test and then
// produces subtly wrong output on long conversations, with nothing raised.
//
// Only the LIVE prefix is captured: `seq_len × per_pos_bytes` per layer, not
// `cap`. A session 4K tokens into a 128K slab is ~45 MB of KV, not 1.39 GB.
// That is what makes a large session pool practical.

use crate::swap::SwapError;

/// Magic for the serialised form: "HIPF_SW1" as big-endian ASCII.
const MAGIC: u64 = 0x484950465F535731;
/// v2: added `per_pos_v_bytes` — the rotated-K tiers (asym{2,3,4}/fwht{2,3,4})
/// store K and V at different per-position strides, so one stride field can no
/// longer size both spans of the payload.
const VERSION: u32 = 2;

/// FNV-1a over the payload. Same construction `replay.rs::capture_summary`
/// uses, kept identical so there is one hash idiom in the codebase.
pub fn checksum_of(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Everything that must match for a snapshot to be restorable.
///
/// Carries more than the current scratch-only lifetime needs, on purpose: it
/// is exactly the set persistence across a daemon restart would require, so
/// enabling that later is a policy change rather than a format change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotStamp {
    pub model_hash: u64,
    pub kv_dtype_tag: u32,
    /// K-arena per-position stride in bytes.
    pub per_pos_bytes: u32,
    /// V-arena per-position stride in bytes. Equals `per_pos_bytes` on the
    /// q8/bf16 tiers; DIFFERS on the rotated-K tiers, and a payload laid out
    /// with one stride can never be restored into the other.
    pub per_pos_v_bytes: u32,
    pub n_fa_layers: u32,
    pub dn_layout_version: u32,
    pub cap: u32,
    /// Total bytes of DeltaNet state for one slot. Part of the stamp so
    /// `validate` can check the payload length without consulting the model.
    pub dn_bytes: u64,
}

/// One session's captured GPU state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotSnapshot {
    pub stamp: SnapshotStamp,
    pub seq_len: usize,
    pub tokens: Vec<u32>,
    /// Per FA layer in model order: `seq_len × per_pos_bytes` of K then the
    /// same span of V; then the DeltaNet state buffers in `DeltaNetState`'s
    /// own order (s_matrices, s_scales, conv_states, s_ef_residual).
    pub payload: Vec<u8>,
    pub checksum: u64,
}

impl SlotSnapshot {
    /// Bytes the payload must contain for this stamp and `seq_len`: per FA
    /// layer, `seq_len` positions of K at the K stride plus the same at the
    /// V stride, then the DeltaNet state.
    pub fn expected_len(&self) -> usize {
        let kv = self.stamp.n_fa_layers as usize
            * self.seq_len
            * (self.stamp.per_pos_bytes as usize + self.stamp.per_pos_v_bytes as usize);
        kv + self.stamp.dn_bytes as usize
    }

    /// Check a snapshot is safe to restore. Stamp first (cheapest and the
    /// most likely mismatch), then length, then checksum.
    pub fn validate(&self, expect: SnapshotStamp) -> Result<(), SwapError> {
        if self.stamp != expect {
            return Err(SwapError::Stamp(format!(
                "snapshot {:?} != expected {:?}",
                self.stamp, expect
            )));
        }
        let want = self.expected_len();
        if self.payload.len() != want {
            return Err(SwapError::Corrupt(format!(
                "payload is {} bytes, expected {want}",
                self.payload.len()
            )));
        }
        let got = checksum_of(&self.payload);
        if got != self.checksum {
            return Err(SwapError::Corrupt(format!(
                "checksum {got:#x} != recorded {:#x}",
                self.checksum
            )));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96 + self.tokens.len() * 4 + self.payload.len());
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.stamp.model_hash.to_le_bytes());
        out.extend_from_slice(&self.stamp.kv_dtype_tag.to_le_bytes());
        out.extend_from_slice(&self.stamp.per_pos_bytes.to_le_bytes());
        out.extend_from_slice(&self.stamp.per_pos_v_bytes.to_le_bytes());
        out.extend_from_slice(&self.stamp.n_fa_layers.to_le_bytes());
        out.extend_from_slice(&self.stamp.dn_layout_version.to_le_bytes());
        out.extend_from_slice(&self.stamp.cap.to_le_bytes());
        out.extend_from_slice(&self.stamp.dn_bytes.to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        for t in &self.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a serialised snapshot. Rejects wrong magic, wrong version and any
    /// buffer too short for what its own header claims, *before* indexing —
    /// a truncated spill file must be an error, never a panic.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SwapError> {
        const HEADER: usize = 8 + 4 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 8 + 8;
        if bytes.len() < HEADER {
            return Err(SwapError::Corrupt(format!(
                "buffer is {} bytes, shorter than the {HEADER}-byte header",
                bytes.len()
            )));
        }
        let mut o = 0usize;
        let mut u64_at = |o: &mut usize| {
            let v = u64::from_le_bytes(bytes[*o..*o + 8].try_into().expect("bounds checked"));
            *o += 8;
            v
        };
        if u64_at(&mut o) != MAGIC {
            return Err(SwapError::Corrupt("bad magic".to_string()));
        }
        let mut u32_at = |o: &mut usize| {
            let v = u32::from_le_bytes(bytes[*o..*o + 4].try_into().expect("bounds checked"));
            *o += 4;
            v
        };
        let version = u32_at(&mut o);
        if version != VERSION {
            return Err(SwapError::Stamp(format!(
                "snapshot version {version} != {VERSION}"
            )));
        }
        let model_hash = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds"));
        o += 8;
        let kv_dtype_tag = u32_at(&mut o);
        let per_pos_bytes = u32_at(&mut o);
        let per_pos_v_bytes = u32_at(&mut o);
        let n_fa_layers = u32_at(&mut o);
        let dn_layout_version = u32_at(&mut o);
        let cap = u32_at(&mut o);
        let dn_bytes = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds"));
        o += 8;
        let seq_len = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds")) as usize;
        o += 8;
        let n_tokens = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds")) as usize;
        o += 8;
        let checksum = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds"));
        o += 8;
        let payload_len = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("bounds")) as usize;
        o += 8;

        let need = o
            .checked_add(n_tokens.saturating_mul(4))
            .and_then(|x| x.checked_add(payload_len))
            .ok_or_else(|| SwapError::Corrupt("header lengths overflow".to_string()))?;
        if bytes.len() < need {
            return Err(SwapError::Corrupt(format!(
                "buffer is {} bytes, header claims {need}",
                bytes.len()
            )));
        }
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            tokens.push(u32::from_le_bytes(
                bytes[o..o + 4].try_into().expect("bounds"),
            ));
            o += 4;
        }
        let payload = bytes[o..o + payload_len].to_vec();
        Ok(SlotSnapshot {
            stamp: SnapshotStamp {
                model_hash,
                kv_dtype_tag,
                per_pos_bytes,
                per_pos_v_bytes,
                n_fa_layers,
                dn_layout_version,
                cap,
                dn_bytes,
            },
            seq_len,
            tokens,
            payload,
            checksum,
        })
    }
}

/// Capture a slot's live KV plus whatever extra per-slot state the caller
/// supplies (for Qwen3.5/3.6 that is the DeltaNet buffers).
///
/// `extra` is an ordered slice of device buffers. The order is the caller's
/// contract with itself: `restore_slot` writes them back in the same order, so
/// the two call sites must derive it the same way. Keeping it opaque here is
/// what lets this module stay model-agnostic — `hipfire-runtime` cannot depend
/// on an arch crate without a dependency cycle.
pub fn capture_slot(
    gpu: &mut rdna_compute::Gpu,
    pool: &rdna_compute::slot_pool::SlotPool,
    slot: rdna_compute::slot_pool::SlotId,
    k_arenas: &[rdna_compute::GpuTensor],
    v_arenas: &[rdna_compute::GpuTensor],
    extra: &[&rdna_compute::GpuTensor],
    tokens: &[u32],
    stamp: SnapshotStamp,
) -> Result<SlotSnapshot, SwapError> {
    let desc = pool.descriptors()[slot.0];
    let mut seq_len = desc.seq_len as usize;
    if pool.is_paged() {
        // The block table is the paged source of truth for the live length;
        // desc.seq_len is kept in sync with it, but read it from here so a
        // stale descriptor can never widen or shrink the capture. This must
        // happen before `span`/`payload` are sized so the copy loop and the
        // buffer agree by construction.
        let bt = pool
            .block_table(slot)
            .ok_or_else(|| SwapError::Gpu("capture: slot has no block table".into()))?;
        seq_len = bt.live_tokens();
    }
    let per_pos_k = stamp.per_pos_bytes as usize;
    let per_pos_v = stamp.per_pos_v_bytes as usize;
    let span_k = seq_len * per_pos_k;
    let span_v = seq_len * per_pos_v;
    let dn_total: usize = extra.iter().map(|t| t.buf.size()).sum();
    let mut payload = vec![0u8; k_arenas.len() * (span_k + span_v) + dn_total];
    let mut off = 0usize;

    if pool.is_paged() {
        // Paged mode: KV is scattered across physical pages. Copy each
        // page's worth of data from the arena into the contiguous payload.
        // The K and V pages share the block table but are sized by their
        // own arena's stride.
        let bt = pool
            .block_table(slot)
            .ok_or_else(|| SwapError::Gpu("capture: slot has no block table".into()))?;
        let page_tokens = rdna_compute::page_pool::PAGE_TOKENS;
        let n_full_pages = seq_len / page_tokens;
        let last_page_tokens = seq_len % page_tokens;
        let needed_pages = n_full_pages + usize::from(last_page_tokens > 0);
        if bt.num_pages() < needed_pages {
            return Err(SwapError::Corrupt(format!(
                "capture: block table holds {} pages but seq_len {seq_len} \
                 needs {needed_pages} — refusing to read past the table",
                bt.num_pages()
            )));
        }

        for (arenas, per_pos) in [(k_arenas, per_pos_k), (v_arenas, per_pos_v)] {
            let page_bytes = page_tokens * per_pos;
            for arena in arenas {
                // Copy full pages.
                for lp in 0..n_full_pages {
                    let phys = bt.physical(lp).expect("page count pre-validated") as usize;
                    let arena_off = phys * page_bytes;
                    let view = arena.sub_offset(arena_off, page_bytes);
                    gpu.hip
                        .memcpy_dtoh(&mut payload[off..off + page_bytes], &view.buf)
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                    off += page_bytes;
                }
                // Copy partial last page.
                if last_page_tokens > 0 {
                    let phys =
                        bt.physical(n_full_pages).expect("page count pre-validated") as usize;
                    let arena_off = phys * page_bytes;
                    let last_bytes = last_page_tokens * per_pos;
                    let view = arena.sub_offset(arena_off, last_bytes);
                    gpu.hip
                        .memcpy_dtoh(&mut payload[off..off + last_bytes], &view.buf)
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                    off += last_bytes;
                }
            }
        }
    } else {
        // Legacy mode: contiguous slab at the slot's per-arena base, each
        // arena sized by its own stride.
        for (arenas, base, span) in [
            (k_arenas, desc.legacy_k_base, span_k),
            (v_arenas, desc.legacy_v_base, span_v),
        ] {
            for arena in arenas {
                if span > 0 {
                    let view = arena.sub_offset(base as usize, span);
                    gpu.hip
                        .memcpy_dtoh(&mut payload[off..off + span], &view.buf)
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                }
                off += span;
            }
        }
    }
    for t in extra {
        let n = t.buf.size();
        gpu.hip
            .memcpy_dtoh(&mut payload[off..off + n], &t.buf)
            .map_err(|e| SwapError::Gpu(e.to_string()))?;
        off += n;
    }
    debug_assert_eq!(off, payload.len());

    let checksum = checksum_of(&payload);
    Ok(SlotSnapshot {
        stamp,
        seq_len,
        tokens: tokens.to_vec(),
        payload,
        checksum,
    })
}

/// Restore a snapshot into `slot`.
///
/// Validation happens against the HOST buffer first and the device is only
/// touched once everything checks out, so a rejected snapshot can never leave
/// a live slot half-written. The slot may be a different one than the snapshot
/// was taken from — residency is a property of the session, not the slot.
pub fn restore_slot(
    gpu: &mut rdna_compute::Gpu,
    pool: &mut rdna_compute::slot_pool::SlotPool,
    slot: rdna_compute::slot_pool::SlotId,
    k_arenas: &[rdna_compute::GpuTensor],
    v_arenas: &[rdna_compute::GpuTensor],
    extra: &[&rdna_compute::GpuTensor],
    snap: &SlotSnapshot,
    expect: SnapshotStamp,
) -> Result<(), SwapError> {
    snap.validate(expect)?;

    let desc = pool.descriptors()[slot.0];
    let per_pos_k = snap.stamp.per_pos_bytes as usize;
    let per_pos_v = snap.stamp.per_pos_v_bytes as usize;
    let span_k = snap.seq_len * per_pos_k;
    let span_v = snap.seq_len * per_pos_v;
    let dn_total: usize = extra.iter().map(|t| t.buf.size()).sum();
    if k_arenas.len() * (span_k + span_v) + dn_total != snap.payload.len() {
        return Err(SwapError::Corrupt(
            "payload does not match this rig's arena and state sizes".to_string(),
        ));
    }
    let mut off = 0usize;
    if pool.is_paged() {
        // Paged mode: ensure the slot has enough pages, then scatter the
        // contiguous payload back into the physical pages (K and V pages
        // sized by their own arena's stride).
        //
        // Ordering matters: provisioning grows the block table, but the
        // descriptor's `seq_len` (what readers and kernels see) is published
        // ONLY after every byte has landed. Publishing it here — before the
        // copies — left a window where a concurrent reader observed a length
        // over rows that had not been restored.
        pool.provision_pages_for(slot, snap.seq_len)
            .map_err(|e| SwapError::Gpu(e))?;
        let bt = pool
            .block_table(slot)
            .ok_or_else(|| SwapError::Gpu("restore: slot has no block table".into()))?;
        let page_tokens = rdna_compute::page_pool::PAGE_TOKENS;
        let n_full_pages = snap.seq_len / page_tokens;
        let last_page_tokens = snap.seq_len % page_tokens;
        let needed_pages = n_full_pages + usize::from(last_page_tokens > 0);
        if bt.num_pages() < needed_pages {
            return Err(SwapError::Corrupt(format!(
                "restore: block table holds {} pages but seq_len {} needs \
                 {needed_pages} — refusing to write past the table",
                bt.num_pages(),
                snap.seq_len
            )));
        }

        for (arenas, per_pos) in [(k_arenas, per_pos_k), (v_arenas, per_pos_v)] {
            let page_bytes = page_tokens * per_pos;
            for arena in arenas {
                // Copy full pages.
                for lp in 0..n_full_pages {
                    let phys = bt.physical(lp).expect("page count pre-validated") as usize;
                    let arena_off = phys * page_bytes;
                    let view = arena.sub_offset(arena_off, page_bytes);
                    gpu.hip
                        .memcpy_htod(&view.buf, &snap.payload[off..off + page_bytes])
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                    off += page_bytes;
                }
                // Copy partial last page.
                if last_page_tokens > 0 {
                    let phys =
                        bt.physical(n_full_pages).expect("page count pre-validated") as usize;
                    let arena_off = phys * page_bytes;
                    let last_bytes = last_page_tokens * per_pos;
                    let view = arena.sub_offset(arena_off, last_bytes);
                    gpu.hip
                        .memcpy_htod(&view.buf, &snap.payload[off..off + last_bytes])
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                    off += last_bytes;
                }
            }
        }
    } else {
        // Legacy mode: contiguous slab at the slot's per-arena base, each
        // arena sized by its own stride.
        for (arenas, base, span) in [
            (k_arenas, desc.legacy_k_base, span_k),
            (v_arenas, desc.legacy_v_base, span_v),
        ] {
            for arena in arenas {
                if span > 0 {
                    let view = arena.sub_offset(base as usize, span);
                    gpu.hip
                        .memcpy_htod(&view.buf, &snap.payload[off..off + span])
                        .map_err(|e| SwapError::Gpu(e.to_string()))?;
                }
                off += span;
            }
        }
    }
    for t in extra {
        let n = t.buf.size();
        gpu.hip
            .memcpy_htod(&t.buf, &snap.payload[off..off + n])
            .map_err(|e| SwapError::Gpu(e.to_string()))?;
        off += n;
    }
    // Both modes publish the new length LAST, after the KV payload and the
    // recurrent-state buffers (`extra`) are in place.
    pool.set_seq_len(slot, snap.seq_len)
        .map_err(|e| SwapError::Gpu(e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> SnapshotStamp {
        SnapshotStamp {
            model_hash: 0xABCD,
            kv_dtype_tag: 1,
            per_pos_bytes: 4,
            per_pos_v_bytes: 4,
            n_fa_layers: 2,
            dn_layout_version: 1,
            cap: 4096,
            dn_bytes: 64,
        }
    }

    fn snap() -> SlotSnapshot {
        // 2 layers x 2 (K,V) x 8 tokens x 4 bytes = 128, plus 64 of DN state.
        let payload = vec![7u8; 128 + 64];
        let checksum = checksum_of(&payload);
        SlotSnapshot {
            stamp: stamp(),
            seq_len: 8,
            tokens: vec![1, 2, 3],
            payload,
            checksum,
        }
    }

    #[test]
    fn a_matching_stamp_and_checksum_validates() {
        assert!(snap().validate(stamp()).is_ok());
    }

    #[test]
    fn a_mismatched_stamp_is_refused() {
        let mut other = stamp();
        other.model_hash = 0x1234;
        assert!(matches!(snap().validate(other), Err(SwapError::Stamp(_))));
    }

    #[test]
    fn a_corrupted_payload_is_refused() {
        let mut s = snap();
        s.payload[10] ^= 0xFF;
        assert!(matches!(s.validate(stamp()), Err(SwapError::Corrupt(_))));
    }

    #[test]
    fn a_truncated_payload_is_refused() {
        let mut s = snap();
        s.payload.truncate(100);
        assert!(matches!(s.validate(stamp()), Err(SwapError::Corrupt(_))));
    }

    #[test]
    fn serialise_round_trips_through_bytes() {
        let s = snap();
        let bytes = s.to_bytes();
        let back = SlotSnapshot::from_bytes(&bytes).expect("must parse");
        assert_eq!(back.seq_len, s.seq_len);
        assert_eq!(back.tokens, s.tokens);
        assert_eq!(back.payload, s.payload);
        assert!(back.validate(stamp()).is_ok());
    }

    #[test]
    fn garbage_bytes_do_not_parse() {
        assert!(SlotSnapshot::from_bytes(&[0u8; 12]).is_err());
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let bytes = snap().to_bytes();
        for cut in [0, 1, 16, 64, bytes.len() - 1] {
            assert!(
                SlotSnapshot::from_bytes(&bytes[..cut]).is_err(),
                "truncation at {cut} must be refused"
            );
        }
    }
}
