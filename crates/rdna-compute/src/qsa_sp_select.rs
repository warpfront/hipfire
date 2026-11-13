// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sequence-parallel QSA selection exchange (`kernels/src/qsa_sp_select.hip`).
//!
//! Inside an SP chunk each rank selects only its own rows of a prefill tile. The chosen
//! block ids of those rows are packed into a compact `[rows][budget_blocks]` i32 buffer
//! ([`Gpu::qsa_sp_pack_blocks`]), all-gathered, and the peer rows are expanded back into
//! the exact `[rows][capacity]` selection bytes of the batched selector
//! ([`Gpu::qsa_sp_expand_blocks`]). Every compact value is a block id `>= 0`, so the SP
//! wire's `0xFFFFFFFF` EMPTY marker never occurs in the payload.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

const MODULE: &str = "qsa_sp_select";
const PACK: &str = "qsa_sp_pack_blocks";
const EXPAND: &str = "qsa_sp_expand_blocks";
const BLOCK: u32 = 256;

/// Geometry and buffers of one tile's selection exchange.
pub struct QsaSpBlocks<'a> {
    /// `>= tile_rows * capacity` i32 words (bytes checked, dtype not).
    pub selected: &'a GpuTensor,
    /// `>= tile_rows * budget_blocks` i32 words (bytes checked, dtype not).
    pub compact: &'a GpuTensor,
    pub tile_rows: usize,
    /// The tile's first position `P`.
    pub position_start: usize,
    /// `(P + tile_rows) / compress`.
    pub block_count: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    /// `>= budget_blocks * compress`.
    pub capacity: usize,
}

/// Pure validation of a launch over rows `[row0, row0 + rows)` (`rows > 0`). Sizes are
/// the byte sizes of the `selected` and `compact` buffers.
#[allow(clippy::too_many_arguments)]
fn validate(
    tile_rows: usize,
    position_start: usize,
    block_count: usize,
    budget_blocks: usize,
    compress: usize,
    capacity: usize,
    selected_bytes: usize,
    compact_bytes: usize,
    row0: usize,
    rows: usize,
) -> Result<(), String> {
    let end = row0
        .checked_add(rows)
        .ok_or_else(|| "row range overflows".to_string())?;
    if end > tile_rows {
        return Err(format!(
            "rows [{row0}, {end}) exceed the tile's {tile_rows} rows"
        ));
    }
    if compress < 1 {
        return Err("compress must be >= 1".into());
    }
    if budget_blocks < 1 {
        return Err("budget_blocks must be >= 1".into());
    }
    let need_capacity = budget_blocks
        .checked_mul(compress)
        .ok_or_else(|| "budget_blocks * compress overflows".to_string())?;
    if capacity < need_capacity {
        return Err(format!(
            "capacity {capacity} is below budget_blocks * compress = {need_capacity}"
        ));
    }
    let word = std::mem::size_of::<i32>();
    let selected_need = tile_rows
        .checked_mul(capacity)
        .and_then(|n| n.checked_mul(word))
        .ok_or_else(|| "tile_rows * capacity * 4 overflows".to_string())?;
    if selected_need > selected_bytes {
        return Err(format!(
            "selected buffer holds {selected_bytes} bytes, {selected_need} required"
        ));
    }
    let compact_need = tile_rows
        .checked_mul(budget_blocks)
        .and_then(|n| n.checked_mul(word))
        .ok_or_else(|| "tile_rows * budget_blocks * 4 overflows".to_string())?;
    if compact_need > compact_bytes {
        return Err(format!(
            "compact buffer holds {compact_bytes} bytes, {compact_need} required"
        ));
    }
    let end_position = position_start
        .checked_add(tile_rows)
        .ok_or_else(|| "position_start + tile_rows overflows".to_string())?;
    for (name, v) in [
        ("row0", row0),
        ("rows", rows),
        ("tile_rows", tile_rows),
        ("position_start + tile_rows", end_position),
        ("block_count", block_count),
        (
            "block_count * compress",
            block_count.saturating_mul(compress),
        ),
        ("budget_blocks", budget_blocks),
        ("compress", compress),
        ("capacity", capacity),
    ] {
        if i32::try_from(v).is_err() {
            return Err(format!("{name} {v} exceeds i32"));
        }
    }
    Ok(())
}

/// Grid size (x) of a flat launch over `items` work items.
fn grid_x(items: usize) -> Result<u32, String> {
    u32::try_from(items.div_ceil(BLOCK as usize)).map_err(|_| "grid exceeds u32".to_string())
}

impl Gpu {
    /// Resolve both kernels (call at load: JIT is not capture-safe).
    pub fn ensure_qsa_sp_select_kernels(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(MODULE, kernels::QSA_SP_SELECT_SRC, PACK)?;
        self.ensure_kernel(MODULE, kernels::QSA_SP_SELECT_SRC, EXPAND)
    }

    fn qsa_sp_check(&self, p: &QsaSpBlocks<'_>, row0: usize, rows: usize) -> HipResult<()> {
        validate(
            p.tile_rows,
            p.position_start,
            p.block_count,
            p.budget_blocks,
            p.compress,
            p.capacity,
            p.selected.buf.size(),
            p.compact.buf.size(),
            row0,
            rows,
        )
        .map_err(|e| HipError::new(0, &format!("qsa_sp_select: {e}")))
    }

    /// `compact[R][j] = sel < 0 ? 0 : sel / compress`, `sel = selected[R][j * compress]`,
    /// for `R` in `[row0, row0 + rows)`, `j < budget_blocks`.
    pub fn qsa_sp_pack_blocks(
        &mut self,
        p: &QsaSpBlocks<'_>,
        row0: usize,
        rows: usize,
    ) -> HipResult<()> {
        if rows == 0 {
            return Ok(());
        }
        self.qsa_sp_check(p, row0, rows)?;
        let grid = grid_x(rows * p.budget_blocks)
            .map_err(|e| HipError::new(0, &format!("qsa_sp_pack_blocks: {e}")))?;
        self.ensure_qsa_sp_select_kernels()?;
        let selected = p.selected.buf.as_ptr();
        let compact = p.compact.buf.as_ptr();
        let (row0_arg, rows_arg, capacity, budget, compress) = (
            row0 as i32,
            rows as i32,
            p.capacity as i32,
            p.budget_blocks as i32,
            p.compress as i32,
        );
        let mut params: [*mut c_void; 7] = [
            &selected as *const _ as *mut c_void,
            &compact as *const _ as *mut c_void,
            &row0_arg as *const _ as *mut c_void,
            &rows_arg as *const _ as *mut c_void,
            &capacity as *const _ as *mut c_void,
            &budget as *const _ as *mut c_void,
            &compress as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(PACK, [grid, 1, 1], [BLOCK, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(selected);
            b.push_ptr(compact);
            b.push_i32(row0_arg);
            b.push_i32(rows_arg);
            b.push_i32(capacity);
            b.push_i32(budget);
            b.push_i32(compress);
            b
        })
    }

    /// Selected rows `[row0, row0 + rows)` rebuilt from `compact`, byte-identical to the
    /// batched selector's rows (`qsa_select_batched` layout).
    pub fn qsa_sp_expand_blocks(
        &mut self,
        p: &QsaSpBlocks<'_>,
        row0: usize,
        rows: usize,
    ) -> HipResult<()> {
        if rows == 0 {
            return Ok(());
        }
        self.qsa_sp_check(p, row0, rows)?;
        let grid = grid_x(rows * p.capacity)
            .map_err(|e| HipError::new(0, &format!("qsa_sp_expand_blocks: {e}")))?;
        self.ensure_qsa_sp_select_kernels()?;
        let selected = p.selected.buf.as_ptr();
        let compact = p.compact.buf.as_ptr();
        let (row0_arg, rows_arg, position, blocks, budget, compress, capacity) = (
            row0 as i32,
            rows as i32,
            p.position_start as i32,
            p.block_count as i32,
            p.budget_blocks as i32,
            p.compress as i32,
            p.capacity as i32,
        );
        let mut params: [*mut c_void; 9] = [
            &compact as *const _ as *mut c_void,
            &selected as *const _ as *mut c_void,
            &row0_arg as *const _ as *mut c_void,
            &rows_arg as *const _ as *mut c_void,
            &position as *const _ as *mut c_void,
            &blocks as *const _ as *mut c_void,
            &budget as *const _ as *mut c_void,
            &compress as *const _ as *mut c_void,
            &capacity as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(EXPAND, [grid, 1, 1], [BLOCK, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(compact);
            b.push_ptr(selected);
            b.push_i32(row0_arg);
            b.push_i32(rows_arg);
            b.push_i32(position);
            b.push_i32(blocks);
            b.push_i32(budget);
            b.push_i32(compress);
            b.push_i32(capacity);
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Qwen4: tile of 4096 rows, budget 512 blocks, compress 4, capacity 2051.
    fn ok(row0: usize, rows: usize) -> Result<(), String> {
        validate(
            4096,
            28672,
            8192,
            512,
            4,
            2051,
            4096 * 2051 * 4,
            4096 * 512 * 4,
            row0,
            rows,
        )
    }

    #[test]
    fn accepts_full_and_partial_ranges() {
        assert!(ok(0, 4096).is_ok());
        assert!(ok(2048, 2048).is_ok());
        assert!(ok(4095, 1).is_ok());
    }

    #[test]
    fn rejects_row_overrun() {
        assert!(ok(1, 4096).is_err());
        assert!(ok(4096, 1).is_err());
        assert!(ok(usize::MAX, 2).is_err());
    }

    #[test]
    fn rejects_bad_geometry() {
        let v = |compress, budget, capacity| {
            validate(8, 0, 2, budget, compress, capacity, 1 << 20, 1 << 20, 0, 8)
        };
        assert!(v(4, 512, 2051).is_ok());
        assert!(v(0, 512, 2051).is_err());
        assert!(v(4, 0, 2051).is_err());
        assert!(v(4, 512, 2047).is_err());
    }

    #[test]
    fn rejects_small_buffers() {
        let v = |sel, cmp| validate(8, 0, 2, 4, 4, 16, sel, cmp, 0, 8);
        assert!(v(8 * 16 * 4, 8 * 4 * 4).is_ok());
        assert!(v(8 * 16 * 4 - 1, 8 * 4 * 4).is_err());
        assert!(v(8 * 16 * 4, 8 * 4 * 4 - 1).is_err());
    }

    #[test]
    fn rejects_i32_overflow() {
        let big = i32::MAX as usize + 1;
        assert!(validate(8, big, 2, 4, 4, 16, 1 << 20, 1 << 20, 0, 8).is_err());
        assert!(validate(8, 0, big, 4, 4, 16, 1 << 20, 1 << 20, 0, 8).is_err());
        assert!(validate(8, 0, i32::MAX as usize, 4, 4, 16, 1 << 20, 1 << 20, 0, 8).is_err());
    }

    #[test]
    fn grid_covers_items() {
        assert_eq!(grid_x(1), Ok(1));
        assert_eq!(grid_x(256), Ok(1));
        assert_eq!(grid_x(257), Ok(2));
        assert!(grid_x(usize::MAX).is_err());
    }
}
