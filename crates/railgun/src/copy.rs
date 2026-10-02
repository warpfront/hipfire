//! The hipfire-owned copy kernels behind `Node::Copy` and `Node::CopyBatch`
//! (`kernels/src/railgun_copy.hip`): source, symbols, the slot-table entry
//! layout, slot counts and launch geometry.
//!
//! Receipts (railgun-cert, tier D, gfx1201/gfx1100/gfx1151): both kernels are
//! A1-`Proven` with exact modes. `railgun_copy` reads `src` and writes `dst`.
//! `railgun_copy_batch` reads its table (scalar loads only), reads the source
//! slot arguments `s0..s7` and writes the destination slot arguments
//! `d0..d7`: the table holds slot indices and 32-bit offsets, never pointers,
//! and an entry's slot select is a union of kernarg pointers, so A1
//! attributes every access to the slot arguments. Where inside a slot's
//! resource an entry reads or writes is the table's content, which `prepare`
//! writes and bounds-checks against the resource.

/// Kernel source, compiled by the runtime's JIT recipe as module [`MODULE`].
pub const SOURCE: &str = include_str!("../../../kernels/src/railgun_copy.hip");
pub const MODULE: &str = "railgun_copy";
/// `railgun_copy(const u8* src, u8* dst, u32 bytes)`.
pub const COPY_SYMBOL: &str = "railgun_copy";
/// `railgun_copy_batch(const RailgunCopyEntry* table, u32 n, const u8* s0..s7, u8* d0..d7)`.
pub const BATCH_SYMBOL: &str = "railgun_copy_batch";

/// Threads per block, both kernels.
pub const BLOCK: u32 = 256;
/// Bytes of the largest copy per block of grid.x: each thread moves about
/// two 16 B vectors of it. For the DFlash scatter's 20 KiB rows, 1 to 8
/// vectors per thread measured the same on gfx1201 (launch-bound).
pub const CHUNK_BYTES: u64 = BLOCK as u64 * 16 * 2;
/// Largest grid.x: keeps the kernel's 32-bit `stride = gridDim.x * 256`
/// at or below 2^31, so no index sum overflows.
pub const MAX_GRID_X: u32 = 1 << 23;
/// Largest single copy: the kernels count bytes in 32 bits.
pub const MAX_COPY_BYTES: u64 = u32::MAX as u64;
/// Largest `n` of one batch: the entry index is `blockIdx.y`.
pub const MAX_BATCH: u64 = 65_535;
/// Size of one table entry.
pub const ENTRY_BYTES: u64 = 24;
/// Source slot arguments of `railgun_copy_batch` (`s0..s7`), and destination
/// slot arguments (`d0..d7`); `RAILGUN_COPY_SLOTS` in the kernel.
///
/// Eight per side covers the DFlash hidden scatter (one source per extract
/// layer, 5 in the shipped drafters, into one `target_hidden`) with room for
/// drafters of up to 8 extract layers. More slots cost a longer scalar select
/// per block and 16 B of kernarg per pair; the explicit segment is 144 B here,
/// which with the hidden grid dwords the kernel loads stays well inside the
/// 512 B that A1 tracks dword by dword (beyond it pointer roots overflow and
/// the kernel is `Unknown`). A table naming more resources is refused, not
/// split: snapshot/restore (4 buffer classes × 48 layers) is not a batch.
pub const SRC_SLOTS: usize = 8;
pub const DST_SLOTS: usize = 8;
/// An entry's `offset + bytes` must not exceed this: the kernel adds 32-bit
/// offsets, so a batch reaches the first 4 GiB of each slot's resource.
pub const MAX_SLOT_END: u64 = 1 << 32;
/// Explicit kernarg bytes of `railgun_copy_batch`.
pub const BATCH_KERNARG_BYTES: usize = 16 + 8 * (SRC_SLOTS + DST_SLOTS);

/// One table entry, the mirror of `RailgunCopyEntry`: source slot and byte
/// offset, destination slot and byte offset, byte count; 24 bytes
/// little-endian with a zero `reserved` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyEntry {
    pub src_slot: u32,
    pub src_offset: u32,
    pub dst_slot: u32,
    pub dst_offset: u32,
    pub bytes: u32,
}

impl CopyEntry {
    pub fn encode(&self, out: &mut Vec<u8>) {
        for word in [self.src_slot, self.src_offset, self.dst_slot, self.dst_offset, self.bytes, 0] {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

/// One kernel launch: symbol, geometry and the explicit kernarg bytes
/// (natural alignment, padded to 8; the runtime appends the hidden suffix).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub kernarg: Vec<u8>,
}

/// grid.x for copies of at most `bytes` bytes.
pub fn grid_x(bytes: u64) -> u32 {
    bytes.div_ceil(CHUNK_BYTES).clamp(1, u64::from(MAX_GRID_X)) as u32
}

/// `railgun_copy` for one non-empty copy of `bytes ≤ MAX_COPY_BYTES`.
pub fn copy_launch(src: u64, dst: u64, bytes: u32) -> Launch {
    let mut kernarg = Vec::with_capacity(24);
    kernarg.extend_from_slice(&src.to_le_bytes());
    kernarg.extend_from_slice(&dst.to_le_bytes());
    kernarg.extend_from_slice(&bytes.to_le_bytes());
    kernarg.resize(24, 0);
    Launch { symbol: COPY_SYMBOL, grid: [grid_x(u64::from(bytes)), 1, 1], block: [BLOCK, 1, 1], kernarg }
}

/// `railgun_copy_batch` over the first `n ≥ 1` entries of the table at
/// `table`, whose largest entry is `max_bytes`, with source slot `k` bound
/// to `src[k]` and destination slot `k` to `dst[k]`. Unused slots are null,
/// so an entry naming one faults instead of reaching another allocation.
pub fn batch_launch(table: u64, n: u32, src: &[u64], dst: &[u64], max_bytes: u64) -> Launch {
    assert!(src.len() <= SRC_SLOTS && dst.len() <= DST_SLOTS, "{} source / {} destination slots", src.len(), dst.len());
    let mut kernarg = Vec::with_capacity(BATCH_KERNARG_BYTES);
    kernarg.extend_from_slice(&table.to_le_bytes());
    kernarg.extend_from_slice(&n.to_le_bytes());
    kernarg.resize(16, 0);
    for (bases, slots) in [(src, SRC_SLOTS), (dst, DST_SLOTS)] {
        for k in 0..slots {
            kernarg.extend_from_slice(&bases.get(k).copied().unwrap_or(0).to_le_bytes());
        }
    }
    Launch { symbol: BATCH_SYMBOL, grid: [grid_x(max_bytes), n, 1], block: [BLOCK, 1, 1], kernarg }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_keeps_the_kernel_stride_in_32_bits() {
        assert_eq!(grid_x(0), 1);
        assert_eq!(grid_x(CHUNK_BYTES), 1);
        assert_eq!(grid_x(CHUNK_BYTES + 1), 2);
        assert!(u64::from(grid_x(MAX_COPY_BYTES)) * u64::from(BLOCK) <= 1 << 31);
    }
}
