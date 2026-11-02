// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certified gfx1201 PeaceMaker twin of the down A4 producer `qwen4_moe_rotate128_i4`
//! (`HIPFIRE_MOE_ROTATE_I4_PM`, `PrefillRoute::MoeRotateI4Pm`).
//!
//! The kernel computes exactly what the incumbent HIP `qwen4_moe_rotate128_i4` computes: grouped
//! BF16 SwiGLU rows `h` `[P, K]` -> FWHT128 (`mq_rotate_x_128_v2` arithmetic) -> `block_i4_128`
//! under the frozen C2 `{5,7}` candidate set, one record of 72 bytes (`d` F32, `s` I32, 64 `qs`
//! bytes) per (flat slot `sorted[p]`, K128 group `g`) at `xq + 72 * (g * L + sorted[p])`. A pad
//! row (`sorted[p] < 0`) writes nothing. The module is the module key and symbol
//! `moe_rotate_i4_pm_gfx1201` of `kernels/moe_rotate_i4_pm_gfx1201.hxaco`.
//!
//! Launch: grid `[P, K/128, 1]`, block `[32, 1, 1]` (one wave32 per (row, group), no LDS).
//! Kernarg (56-byte segment): `h@0, sorted@8, xq@16, signs1@24, signs2@32` (u64 each), then u32
//! `K@40, P@44, L@48`. The incumbent's first 52 argument bytes are the same layout; only the grid
//! differs (the incumbent folds the row and group into grid x).
//!
//! Selection ([`Gpu::qwen4_moe_rotate128_i4`]): the typed flag (unset follows the exact-gfx1201
//! qualification, `0` kills) AND exact-gfx1201 raw inventory AND the requested symmetric IU4 route
//! AND the frozen C2 producer candidate set AND a supported shape. Any miss keeps the incumbent,
//! which stays reachable as [`Gpu::qwen4_moe_rotate128_i4_incumbent`] (the oracle's control).
//! The explicit lab entry [`Gpu::qwen4_moe_rotate128_i4_pm`] never consults the flag and errors on
//! an unsupported device, candidate set or shape instead of approximating.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use crate::moe::require_span;
use crate::prefill_route_capabilities::{raw_inventory, PrefillRoute};
use hip_bridge::HipResult;

/// Module key and device symbol of the PM rotate twin.
pub const MOE_ROTATE_I4_PM_GFX1201_SYMBOL: &str = "moe_rotate_i4_pm_gfx1201";
/// Device symbol of the grouped-position producer twin (module `qwen4_moe_down_gp_pm_gfx1201`).
pub const MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL: &str = "moe_rotate_i4_gp_pm_gfx1201";
/// Threads per workgroup: one wave32.
pub const MOE_ROTATE_I4_PM_BLOCK: u32 = 32;
/// Rotation width and A4 group width in columns.
pub const MOE_ROTATE_I4_PM_GROUP: usize = 128;
/// Bytes of one `block_i4_128` record.
pub const MOE_ROTATE_I4_PM_RECORD_BYTES: usize = 72;
/// HIP's grid y extent limit; grid y is the K128 group count.
const GRID_Y_MAX: usize = 65_535;
/// Most records of one sidecar (`K/128 * L`): the kernel's record index `g * L + sorted[p]` is a
/// 32-bit value (the largest index, `groups * L - 1`, must fit `u32`). Byte addresses
/// (`h + p * 2K`, `xq + 72 * index`) are formed in 64 bits.
const RECORD_INDEX_LIMIT: u64 = 1 << 32;

/// Embedded image and device symbol (also the module key) of the rotate twin for `arch`; `None`
/// where no builder object exists. The anchor of `PrefillRoute::MoeRotateI4Pm`'s raw inventory.
pub(crate) fn moe_rotate_i4_pm_kernel(arch: &str) -> Option<(&'static [u8], &'static str)> {
    match arch {
        "gfx1201" => Some((
            kernels::MOE_ROTATE_I4_PM_GFX1201,
            MOE_ROTATE_I4_PM_GFX1201_SYMBOL,
        )),
        _ => None,
    }
}

/// Validated geometry of one rotate launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRotateI4PmGeometry {
    /// `[P, K/128, 1]`.
    pub grid: [u32; 3],
    /// K128 groups per row.
    pub groups: usize,
    /// Bytes of `h` read: `P * K * 2`.
    pub h_bytes: usize,
    /// Bytes of the sidecar: `K/128 * L * 72`.
    pub xq_bytes: usize,
}

/// Host-side domain check of one launch (pure; no device, no kernel image): `K` a positive
/// multiple of 128, `P` and `L` positive, every extent within `i32` (the kernarg words are read
/// signed by the incumbent and unsigned by the twin), the group count within HIP's grid y limit,
/// the record count `K/128 * L` within the kernel's 32-bit record index (at most 2^32 records;
/// A3B K512, 8192 rows x top-8 is 262144) and the byte spans representable. The A3B down
/// profile is `K = 512`; every other K128 multiple passes through the kernel's runtime `K`
/// kernarg. Byte addresses are 64-bit.
pub fn validate_moe_rotate_i4_pm(
    k: usize,
    grouped_rows: usize,
    total_slots: usize,
) -> Result<MoeRotateI4PmGeometry, String> {
    if k == 0 || k % MOE_ROTATE_I4_PM_GROUP != 0 {
        return Err(format!("K = {k} must be a positive multiple of 128"));
    }
    if grouped_rows == 0 || total_slots == 0 {
        return Err(format!(
            "need nonzero rows (P = {grouped_rows}, L = {total_slots})"
        ));
    }
    for (name, value) in [("K", k), ("P", grouped_rows), ("L", total_slots)] {
        if i32::try_from(value).is_err() {
            return Err(format!("{name} = {value} exceeds i32"));
        }
    }
    let groups = k / MOE_ROTATE_I4_PM_GROUP;
    if groups > GRID_Y_MAX {
        return Err(format!("{groups} K128 groups exceed HIP's grid y limit"));
    }
    let h_bytes = grouped_rows
        .checked_mul(k)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| format!("h span of P = {grouped_rows} x K = {k} BF16 overflows"))?;
    if (groups as u64).saturating_mul(total_slots as u64) > RECORD_INDEX_LIMIT {
        return Err(format!(
            "{groups} groups x L = {total_slots} slots exceed the kernel's 32-bit record index (2^32 records)"
        ));
    }
    let xq_bytes = groups
        .checked_mul(total_slots)
        .and_then(|n| n.checked_mul(MOE_ROTATE_I4_PM_RECORD_BYTES))
        .ok_or_else(|| {
            format!("sidecar span of {groups} groups x L = {total_slots} records overflows")
        })?;
    Ok(MoeRotateI4PmGeometry {
        grid: [grouped_rows as u32, groups as u32, 1],
        groups,
        h_bytes,
        xq_bytes,
    })
}

/// Host-side domain check of one grouped-position launch: [`validate_moe_rotate_i4_pm`] with the
/// plane stride `P` (`grouped_rows`) in place of `L` in the 32-bit record
/// index bound, `groups * P <= 2^32`. `xq_bytes` is the grouped plane's `K/128 * P * 72`.
pub fn validate_moe_rotate_i4_gp_pm(
    k: usize,
    grouped_rows: usize,
    total_slots: usize,
) -> Result<MoeRotateI4PmGeometry, String> {
    let mut geometry = validate_moe_rotate_i4_pm(k, grouped_rows, total_slots)?;
    if (geometry.groups as u64).saturating_mul(grouped_rows as u64) > RECORD_INDEX_LIMIT {
        return Err(format!(
            "{} groups x P = {grouped_rows} rows exceed the kernel's 32-bit record index (2^32 records)",
            geometry.groups
        ));
    }
    // The down entry offsets a record through a 24-bit multiply: positions below 2^24, and the
    // plane a 32-bit descriptor extent.
    if grouped_rows >= 1 << 24 || (geometry.groups as u64) * (grouped_rows as u64) * 72 >= 1 << 32 {
        // (the per-epoch descriptor is `grouped_rows * 72` bytes; the base pointer is 64-bit)
        if grouped_rows >= 1 << 24 {
            return Err(format!("grouped rows P = {grouped_rows} exceed 2^24"));
        }
    }
    geometry.xq_bytes = geometry
        .groups
        .checked_mul(grouped_rows)
        .and_then(|n| n.checked_mul(MOE_ROTATE_I4_PM_RECORD_BYTES))
        .ok_or_else(|| "grouped plane span overflows".to_string())?;
    Ok(geometry)
}

/// Whether the rotate twin accepts this launch shape (pure).
pub fn moe_rotate_i4_pm_shape_supported(k: usize, grouped_rows: usize, total_slots: usize) -> bool {
    validate_moe_rotate_i4_pm(k, grouped_rows, total_slots).is_ok()
}

/// Pure selection of the rotate twin: the resolved typed flag (unset follows the exact-gfx1201
/// qualification, `0` kills), the
/// exact-gfx1201 raw inventory, the requested symmetric IU4 route, the exact C2 producer
/// candidate set and a supported shape. Any miss keeps the incumbent.
pub fn moe_rotate_i4_pm_selection(
    flag: bool,
    arch: &str,
    iu4_requested: bool,
    c2_candidates: bool,
    shape_supported: bool,
) -> bool {
    flag && raw_inventory(arch, PrefillRoute::MoeRotateI4Pm)
        && iu4_requested
        && c2_candidates
        && shape_supported
}

impl Gpu {
    /// The frozen C2 producer candidate set is in force: at least one `-DIU4_A4_CANDIDATES=` flag
    /// and every such flag exactly `-DIU4_A4_CANDIDATES=2` (as
    /// [`Gpu::qwen4_moe_sym_iu4_requested`] checks it).
    pub fn iu4_a4_c2_candidates_exact(&self) -> bool {
        let mut candidates = self
            .flags
            .hipcc_extra_flags
            .split_whitespace()
            .filter(|flag| flag.starts_with("-DIU4_A4_CANDIDATES="))
            .peekable();
        candidates.peek().is_some() && candidates.all(|flag| flag == "-DIU4_A4_CANDIDATES=2")
    }

    /// Whether [`Gpu::qwen4_moe_rotate128_i4`] runs the PM twin for this launch: see
    /// [`moe_rotate_i4_pm_selection`]. The requested-IU4 conjunct is either symmetric IU4 route
    /// ([`Gpu::qwen4_moe_sym_iu4_requested`] or [`Gpu::moe_sym_iu4_requested`]); the device flag
    /// is the once-resolved `flags.moe_rotate_i4_pm`.
    pub fn moe_rotate_i4_pm_selected(
        &self,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> bool {
        moe_rotate_i4_pm_selection(
            self.flags.moe_rotate_i4_pm,
            &self.arch,
            self.qwen4_moe_sym_iu4_requested() || self.moe_sym_iu4_requested(),
            self.iu4_a4_c2_candidates_exact(),
            moe_rotate_i4_pm_shape_supported(k, grouped_rows, total_slots),
        )
    }

    /// Whether [`Gpu::qwen4_moe_rotate128_i4_gp`] writes the grouped-position plane for this launch:
    /// the PM rotate twin is selected ([`Gpu::moe_rotate_i4_pm_selected`]), the device has the PM
    /// NT4/NT8 down entries ([`Gpu::qwen4_moe_sym_kernel`] on gfx1201) and the grouped shape is
    /// admitted ([`validate_moe_rotate_i4_gp_pm`]). Pure (no allocation, no launch).
    pub fn qwen4_moe_rotate128_i4_gp_selected(
        &self,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> bool {
        self.moe_rotate_i4_pm_selected(k, grouped_rows, total_slots)
            && self.qwen4_moe_down_gp_available()
            && validate_moe_rotate_i4_gp_pm(k, grouped_rows, total_slots).is_ok()
    }

    /// Down A4 producer for the Mq4Sym down GEMM [`Gpu::gemm_qwen4_moe_down_iu4_sym`]: where the
    /// PM rotate twin is selected ([`Gpu::moe_rotate_i4_pm_selected`]) and the device has the PM
    /// NT4/NT8 down entries, it writes the grouped-position plane (record `g * grouped_rows + row`,
    /// [`MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL`]) whose handle makes the down launch read each tile's
    /// 16 records contiguously (the byte-identical `..._gp` entry); anywhere else it is exactly
    /// [`Gpu::qwen4_moe_rotate128_i4`] (the flat-slot sidecar). Every other consumer of the
    /// sidecar (the slab and E8P/W3 cold codecs) keeps calling `qwen4_moe_rotate128_i4`.
    pub fn qwen4_moe_rotate128_i4_gp(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        if !self.qwen4_moe_rotate128_i4_gp_selected(k, grouped_rows, total_slots) {
            return self.qwen4_moe_rotate128_i4(h, sorted, k, grouped_rows, total_slots);
        }
        self.qwen4_moe_rotate128_i4_gp_pm(h, sorted, k, grouped_rows, total_slots)
    }

    /// Explicit lab entry of [`Gpu::qwen4_moe_rotate128_i4_gp`]: always the grouped-position PM
    /// producer; errors where it cannot run (no bundle, non-C2 candidates, unsupported shape).
    pub fn qwen4_moe_rotate128_i4_gp_pm(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        const LABEL: &str = "qwen4_moe_rotate128_i4_gp_pm";
        self.bind_thread()?;
        let refuse = |what: &str| Err(hip_bridge::HipError::new(0, &format!("{LABEL}: {what}")));
        if self.arch != "gfx1201" {
            return refuse(&format!(
                "the grouped-position producer is a gfx1201 builder module (device is {})",
                self.arch
            ));
        }
        if !self.iu4_a4_c2_candidates_exact() {
            return refuse(
                "the twin implements the frozen C2 A4 candidate set (-DIU4_A4_CANDIDATES=2)",
            );
        }
        let geometry = match validate_moe_rotate_i4_gp_pm(k, grouped_rows, total_slots) {
            Ok(geometry) => geometry,
            Err(why) => return refuse(&why),
        };
        require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i4_gp_pm grouped BF16 rows",
        )?;
        require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i4_gp_pm sorted slots",
        )?;
        self.ensure_mq_signs_128()?;
        let symbol = MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL;
        self.ensure_embedded_kernel(
            crate::gemm::QWEN4_MOE_DOWN_GP_PM_MODULE,
            kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201,
            symbol,
        )?;
        let reservation = self.reserve_qwen4_moe_down_i4_plane(k, total_slots, grouped_rows)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let (kv, pv, lv) = (k as u32, grouped_rows as u32, total_slots as u32);
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            symbol,
            geometry.h_bytes + grouped_rows * 4 + geometry.xq_bytes,
        );
        let result = self.launch_maybe_blob(
            symbol,
            geometry.grid,
            [MOE_ROTATE_I4_PM_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_u32(kv);
                b.push_u32(pv);
                b.push_u32(lv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(crate::scratch::Int4MmqPrepared::from_reservation(
            reservation,
        ))
    }

    /// Down A4 producer, certified PM twin ([`MOE_ROTATE_I4_PM_GFX1201_SYMBOL`], bundle
    /// `kernels::MOE_ROTATE_I4_PM_GFX1201`), stream-ordered on the active stream. Same operands,
    /// sidecar handle and bytes as [`Gpu::qwen4_moe_rotate128_i4_incumbent`]. Explicit lab entry:
    /// it never consults `HIPFIRE_MOE_ROTATE_I4_PM`, and errors on a device without the bundle,
    /// a non-C2 producer candidate set or an unsupported shape
    /// ([`validate_moe_rotate_i4_pm`]); there is no HIP fallback here.
    pub fn qwen4_moe_rotate128_i4_pm(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        const LABEL: &str = "qwen4_moe_rotate128_i4_pm";
        self.bind_thread()?;
        let refuse = |what: &str| Err(hip_bridge::HipError::new(0, &format!("{LABEL}: {what}")));
        let Some((image, symbol)) = moe_rotate_i4_pm_kernel(&self.arch) else {
            return refuse(&format!(
                "the rotate twin is a gfx1201 builder module (device is {}); no hipcc fallback",
                self.arch
            ));
        };
        if !self.iu4_a4_c2_candidates_exact() {
            return refuse(
                "the twin implements the frozen C2 A4 candidate set (-DIU4_A4_CANDIDATES=2)",
            );
        }
        let geometry = match validate_moe_rotate_i4_pm(k, grouped_rows, total_slots) {
            Ok(geometry) => geometry,
            Err(why) => return refuse(&why),
        };
        require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i4_pm grouped BF16 rows",
        )?;
        require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i4_pm sorted slots",
        )?;
        self.ensure_mq_signs_128()?;
        self.ensure_embedded_kernel(symbol, image, symbol)?;
        let reservation = self.reserve_qwen4_moe_down_i4_sidecar(k, total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let (kv, pv, lv) = (k as u32, grouped_rows as u32, total_slots as u32);
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            symbol,
            geometry.h_bytes + grouped_rows * 4 + geometry.xq_bytes,
        );
        let result = self.launch_maybe_blob(
            symbol,
            geometry.grid,
            [MOE_ROTATE_I4_PM_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_u32(kv);
                b.push_u32(pv);
                b.push_u32(lv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(crate::scratch::Int4MmqPrepared::from_reservation(
            reservation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a3b_down_shapes_validate_with_the_pm_grid() {
        // A3B top-8 down: K = 512, rows 512/4096/8192 tokens -> P = rows * 8 grouped rows.
        for rows in [512usize, 4096, 8192] {
            let p = rows * 8;
            let geometry = validate_moe_rotate_i4_pm(512, p, p).expect("A3B shape");
            assert_eq!(geometry.grid, [p as u32, 4, 1]);
            assert_eq!(geometry.groups, 4);
            assert_eq!(geometry.h_bytes, p * 1024);
            assert_eq!(geometry.xq_bytes, 4 * p * 72);
            assert!(moe_rotate_i4_pm_shape_supported(512, p, p));
        }
        // Padding and expert-run rounding make P larger than L: still one record per slot.
        let geometry = validate_moe_rotate_i4_pm(512, 4096 * 8 + 256, 4096 * 8).unwrap();
        assert_eq!(geometry.grid, [4096 * 8 + 256, 4, 1]);
        // The kernel takes K at run time: other K128 multiples are admitted, K640 included.
        assert_eq!(
            validate_moe_rotate_i4_pm(640, 64, 64).unwrap().grid,
            [64, 5, 1]
        );
        assert_eq!(
            validate_moe_rotate_i4_pm(128, 1, 1).unwrap().grid,
            [1, 1, 1]
        );
    }

    #[test]
    fn grouped_plane_extents_use_the_grouped_row_stride() {
        // A3B top-8 at 8192 tokens: L = 65536 slots, P up to 65536 + 256 * 15 grouped rows.
        let (l, p) = (8192 * 8, 8192 * 8 + 3840);
        let flat = validate_moe_rotate_i4_pm(512, p, l).unwrap();
        let grouped = validate_moe_rotate_i4_gp_pm(512, p, l).unwrap();
        assert_eq!(grouped.grid, flat.grid);
        assert_eq!(flat.xq_bytes, 4 * l * 72);
        assert_eq!(grouped.xq_bytes, 4 * p * 72);
        // The grouped plane may be sized below L (a launch whose rows leave slots unreferenced).
        assert_eq!(
            validate_moe_rotate_i4_gp_pm(512, l - 100, l)
                .unwrap()
                .xq_bytes,
            4 * (l - 100) * 72
        );
        // The 32-bit record index is `g * P + row`: P bounds it, not L.
        assert!(validate_moe_rotate_i4_gp_pm(512, (1 << 30) + 1, 8).is_err());
        assert!(validate_moe_rotate_i4_gp_pm(512, 1 << 24, 8).is_err());
        assert!(validate_moe_rotate_i4_gp_pm(512, (1 << 24) - 16, 8).is_ok());
        for (k, p, l) in [(0, 8, 8), (64, 8, 8), (512, 0, 8), (512, 8, 0)] {
            assert!(
                validate_moe_rotate_i4_gp_pm(k, p, l).is_err(),
                "{k} {p} {l}"
            );
        }
    }

    #[test]
    fn grouped_symbol_is_the_gp_twin_of_the_bundle_module() {
        assert_eq!(
            MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL,
            "moe_rotate_i4_gp_pm_gfx1201"
        );
        let image = kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201;
        for symbol in [
            MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL,
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
        ] {
            let descriptor = format!("{symbol}.kd");
            assert!(
                image
                    .windows(descriptor.len())
                    .any(|w| w == descriptor.as_bytes()),
                "{symbol}"
            );
        }
        assert_eq!(
            crate::gemm::QWEN4_MOE_DOWN_GP_PM_ENTRIES,
            [
                "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
                "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8"
            ]
        );
    }

    #[test]
    fn unsupported_shapes_error_instead_of_approximating() {
        for (k, p, l) in [
            (0, 8, 8),
            (64, 8, 8),
            (512 + 64, 8, 8),
            (512, 0, 8),
            (512, 8, 0),
            (128 * 65_536, 8, 8),
            (512, (i32::MAX as usize) + 1, 8),
            (512, 8, (i32::MAX as usize) + 1),
            // groups * L past 2^32 records: the u32 record index g*L + slot would wrap.
            (512, 8, (1 << 30) + 1),
            (1024, 8, 1 << 30),
            (128 * 65_535, 8, 1 << 17),
        ] {
            assert!(
                validate_moe_rotate_i4_pm(k, p, l).is_err(),
                "K={k} P={p} L={l}"
            );
            assert!(
                !moe_rotate_i4_pm_shape_supported(k, p, l),
                "K={k} P={p} L={l}"
            );
        }
    }

    #[test]
    fn record_index_limit_is_exactly_two_to_the_32_records() {
        // 4 groups x 2^30 slots = 2^32 records: the largest index 2^32 - 1 still fits u32.
        let geometry = validate_moe_rotate_i4_pm(512, 8, 1 << 30).expect("2^32 records");
        assert_eq!(geometry.xq_bytes, 4 * (1 << 30) * 72);
        // One more slot reaches 2^32 + 4 records.
        assert!(validate_moe_rotate_i4_pm(512, 8, (1 << 30) + 1).is_err());
        // K512 x 8192 rows x top-8 (the largest A3B case) is 262144 records.
        assert!(moe_rotate_i4_pm_shape_supported(
            512,
            8192 * 8 + 3840,
            8192 * 8
        ));
    }

    #[test]
    fn selection_needs_every_conjunct() {
        assert!(moe_rotate_i4_pm_selection(
            true, "gfx1201", true, true, true
        ));
        // Each conjunct alone refuses.
        assert!(!moe_rotate_i4_pm_selection(
            false, "gfx1201", true, true, true
        ));
        assert!(!moe_rotate_i4_pm_selection(
            true, "gfx1201", false, true, true
        ));
        assert!(!moe_rotate_i4_pm_selection(
            true, "gfx1201", true, false, true
        ));
        assert!(!moe_rotate_i4_pm_selection(
            true, "gfx1201", true, true, false
        ));
        // An explicit flag never opts in off the raw inventory (no card or model-name gates:
        // the inventory is the exact-architecture bundle table).
        for arch in [
            "gfx1151", "gfx1100", "gfx1200", "gfx942", "gfx906", "unknown",
        ] {
            assert!(
                !moe_rotate_i4_pm_selection(true, arch, true, true, true),
                "{arch}"
            );
        }
    }

    #[test]
    fn bundle_table_is_exact_gfx1201_and_matches_the_symbol() {
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "gfx906",
        ] {
            assert_eq!(
                moe_rotate_i4_pm_kernel(arch).is_some(),
                arch == "gfx1201",
                "{arch}"
            );
        }
        let (image, symbol) = moe_rotate_i4_pm_kernel("gfx1201").unwrap();
        assert_eq!(symbol, "moe_rotate_i4_pm_gfx1201");
        assert!(!image.is_empty());
        assert!(image
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes()));
        let descriptor = format!("{symbol}.kd");
        assert!(image
            .windows(descriptor.len())
            .any(|window| window == descriptor.as_bytes()));
    }
}
