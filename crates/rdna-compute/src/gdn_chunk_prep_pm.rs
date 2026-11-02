// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certified gfx1201 PeaceMaker twin of `gdn_chunk_prep` for the A3B GDN chunk-scan
//! preamble (`HIPFIRE_GDN_CHUNK_PREP_PM`): 16 key heads, 32 value heads, head dim 128,
//! `1..=8192` rows. The launch ABI is the incumbent's, byte for byte: ten pointers `x`,
//! `conv_w`, `conv_state`, `a` (gate, in place), `b` (beta, in place), `dt_bias`, `a_log`,
//! `q`, `k`, `v` (FP16 planes; the incumbent's "bf16" helper stores `__float2half_rn`),
//! then `T` i32, `q_scale` f32, `eps` f32, `n_key_heads` u32 and `n_value_heads` u32 (100
//! kernarg bytes); grid `(ceil(T / 64), 8)`, block 256, no LDS. It writes the same q/k/v
//! planes, the same in-place `a`/`b` and the same conv ring as the incumbent.
//!
//! The launcher is flag-independent. Production selection is [`Gpu::gdn_chunk_prep_pm_admitted`]
//! (the typed flag plus raw inventory plus this shape); its only caller sits under the MoE
//! chunk-scan admission, which already refuses hipGraph capture and replay recording.

use crate::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

pub const MODULE: &str = "gdn_chunk_prep_pm_gfx1201";
pub const SYMBOL: &str = "gdn_chunk_prep_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/gdn_chunk_prep_pm_gfx1201.hxaco");
pub const KEY_HEADS: usize = 16;
pub const VALUE_HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const MIN_ROWS: usize = 1;
pub const MAX_ROWS: usize = 8192;
pub const KERNARG_BYTES: usize = 100;
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows per C64 chunk (grid x).
const CHUNK_ROWS: usize = 64;
/// One wave owns one 128-channel Q/K/V head: `2 * 16 + 32` units, eight waves per block.
const GRID_Y: u32 = ((2 * KEY_HEADS + VALUE_HEADS) / 8) as u32;
/// Channels of one projected row: q, k and v heads.
const CHANNELS: usize = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIM;
/// Conv taps and ring depth of the width-four causal conv.
const CONV_TAPS: usize = 4;
const CONV_RING: usize = CONV_TAPS - 1;

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("gdn_chunk_prep_pm: {}", message.as_ref()))
}

/// Why a GDN chunk-scan prep keeps the incumbent `gdn_chunk_prep`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnChunkPrepPmRefusal {
    Disabled,
    InventoryUnavailable,
    Shape,
    Rows,
}

impl std::fmt::Display for GdnChunkPrepPmRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disabled => "HIPFIRE_GDN_CHUNK_PREP_PM is off",
            Self::InventoryUnavailable => {
                "no embedded image for this architecture (exact gfx1201 only)"
            }
            Self::Shape => "needs 16 key heads and 32 value heads",
            Self::Rows => "needs 1..=8192 rows",
        })
    }
}

impl GdnChunkPrepPmRefusal {
    fn log_once(self) {
        let index = match self {
            // An unset or explicit-0 flag is the incumbent, not a refusal.
            Self::Disabled => return,
            Self::InventoryUnavailable => 0,
            Self::Shape => 1,
            Self::Rows => 2,
        };
        static REFUSALS: [std::sync::Once; 3] = [
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
        ];
        REFUSALS[index].call_once(|| {
            eprintln!(
                "[gdn-chunk-prep-pm] HIPFIRE_GDN_CHUNK_PREP_PM enabled, refused: {self:?}; \
                 keeping incumbent gdn_chunk_prep"
            );
        });
    }
}

/// Pure shape admission of the twin (no flag, no architecture).
fn shape_refusal(
    rows: usize,
    key_heads: usize,
    value_heads: usize,
) -> Option<GdnChunkPrepPmRefusal> {
    if (key_heads, value_heads) != (KEY_HEADS, VALUE_HEADS) {
        Some(GdnChunkPrepPmRefusal::Shape)
    } else if !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
        Some(GdnChunkPrepPmRefusal::Rows)
    } else {
        None
    }
}

/// Pure shape check of the launcher: fixed 16/32 heads and `1..=8192` rows.
pub fn admit(rows: usize, key_heads: usize, value_heads: usize) -> Result<(), String> {
    match shape_refusal(rows, key_heads, value_heads) {
        None => Ok(()),
        Some(reason) => Err(reason.to_string()),
    }
}

/// Launch grid: one C64 chunk per block column, eight units of waves per chunk.
pub fn grid(rows: usize) -> [u32; 3] {
    // The launcher validates rows before reaching this arithmetic.
    [rows.div_ceil(CHUNK_ROWS) as u32, GRID_Y, 1]
}

fn kernargs(
    pointers: [*mut c_void; 10],
    rows: usize,
    q_scale: f32,
    eps: f32,
    key_heads: usize,
    value_heads: usize,
) -> [u8; KERNARG_BYTES] {
    let mut args = [0; KERNARG_BYTES];
    for (i, ptr) in pointers.into_iter().enumerate() {
        args[i * 8..i * 8 + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
    }
    args[80..84].copy_from_slice(&(rows as i32).to_le_bytes());
    args[84..88].copy_from_slice(&q_scale.to_le_bytes());
    args[88..92].copy_from_slice(&eps.to_le_bytes());
    args[92..96].copy_from_slice(&(key_heads as u32).to_le_bytes());
    args[96..100].copy_from_slice(&(value_heads as u32).to_le_bytes());
    args
}

fn require_span(tensor: &GpuTensor, bytes: usize, name: &str) -> HipResult<()> {
    let have = tensor.buf.size();
    if have < bytes {
        return Err(err(format!("{name} needs {bytes} bytes, has {have}")));
    }
    Ok(())
}

impl Gpu {
    /// Pure admission of the PM prep: the resolved `HIPFIRE_GDN_CHUNK_PREP_PM` flag (explicit
    /// `1` wins, explicit `0` kills, unset follows the exact-gfx1201 qualification), the raw
    /// exact-gfx1201 inventory, 16 key / 32 value heads and `1..=8192` rows. Callers add the
    /// chunk-scan admission (which refuses capture and recording).
    pub fn gdn_chunk_prep_pm_admission(
        flag: bool,
        arch: &str,
        n_tokens: usize,
        n_key_heads: usize,
        n_value_heads: usize,
    ) -> Result<(), GdnChunkPrepPmRefusal> {
        use crate::prefill_route_capabilities::{raw_inventory, PrefillRoute};
        if !flag {
            Err(GdnChunkPrepPmRefusal::Disabled)
        } else if !raw_inventory(arch, PrefillRoute::GdnChunkPrepPm) {
            Err(GdnChunkPrepPmRefusal::InventoryUnavailable)
        } else {
            shape_refusal(n_tokens, n_key_heads, n_value_heads).map_or(Ok(()), Err)
        }
    }

    /// Current-device admission of the PM prep. Enabled refusals log once per reason.
    pub fn gdn_chunk_prep_pm_admitted(
        &self,
        n_tokens: usize,
        n_key_heads: usize,
        n_value_heads: usize,
    ) -> bool {
        match Self::gdn_chunk_prep_pm_admission(
            self.flags.gdn_chunk_prep_pm,
            &self.arch,
            n_tokens,
            n_key_heads,
            n_value_heads,
        ) {
            Ok(()) => true,
            Err(reason) => {
                reason.log_once();
                false
            }
        }
    }

    /// Load the embedded image before an admitted route mutates its input scratch or the
    /// persistent conv ring, so a load failure cannot strand a half-run layer.
    pub fn gdn_chunk_prep_pm_prepare(&mut self) -> HipResult<()> {
        if self.arch != "gfx1201" {
            return Err(err("requires gfx1201; no HIP fallback"));
        }
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, SYMBOL)
    }

    /// PM twin of [`Gpu::gdn_chunk_prep`] (same argument order and ABI) for exactly 16 key
    /// heads and 32 value heads over `1..=8192` rows. `input` is the f32 projection
    /// `[rows, 8192]`, `g`/`beta` the f32 `[rows, 32]` gate planes rewritten in place,
    /// `conv_state` the f32 `[8192, 3]` ring advanced in place, and `q`/`k`/`v` the FP16
    /// planes `[rows, 16, 128]`, `[rows, 16, 128]`, `[rows, 32, 128]`. Flag-independent;
    /// gfx1201 only, no HIP fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_chunk_prep_pm(
        &mut self,
        input: &GpuTensor,
        conv_weight: &GpuTensor,
        conv_state: &GpuTensor,
        g: &GpuTensor,
        beta: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        n_tokens: usize,
        q_scale: f32,
        eps: f32,
        n_key_heads: usize,
        n_value_heads: usize,
    ) -> HipResult<()> {
        if self.arch != "gfx1201" {
            return Err(err("requires gfx1201; no HIP fallback"));
        }
        admit(n_tokens, n_key_heads, n_value_heads).map_err(err)?;
        // rows <= 8192, so none of these products can overflow usize.
        require_span(input, n_tokens * CHANNELS * 4, "input")?;
        require_span(conv_weight, CHANNELS * CONV_TAPS * 4, "conv weight")?;
        require_span(conv_state, CHANNELS * CONV_RING * 4, "conv state")?;
        require_span(g, n_tokens * VALUE_HEADS * 4, "G")?;
        require_span(beta, n_tokens * VALUE_HEADS * 4, "beta")?;
        require_span(dt_bias, VALUE_HEADS * 4, "dt bias")?;
        require_span(a_log, VALUE_HEADS * 4, "a_log")?;
        require_span(q, n_tokens * KEY_HEADS * HEAD_DIM * 2, "q")?;
        require_span(k, n_tokens * KEY_HEADS * HEAD_DIM * 2, "k")?;
        require_span(v, n_tokens * VALUE_HEADS * HEAD_DIM * 2, "v")?;
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, SYMBOL)?;
        let mut args = kernargs(
            [
                input.buf.as_ptr(),
                conv_weight.buf.as_ptr(),
                conv_state.buf.as_ptr(),
                g.buf.as_ptr(),
                beta.buf.as_ptr(),
                dt_bias.buf.as_ptr(),
                a_log.buf.as_ptr(),
                q.buf.as_ptr(),
                k.buf.as_ptr(),
                v.buf.as_ptr(),
            ],
            n_tokens,
            q_scale,
            eps,
            n_key_heads,
            n_value_heads,
        );
        self.launch_blob_recorded(
            SYMBOL,
            grid(n_tokens),
            BLOCK,
            0,
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_kernarg_abi_and_grid() {
        let args = kernargs(
            [
                0x1000usize as *mut c_void,
                0x2000usize as *mut c_void,
                0x3000usize as *mut c_void,
                0x4000usize as *mut c_void,
                0x5000usize as *mut c_void,
                0x6000usize as *mut c_void,
                0x7000usize as *mut c_void,
                0x8000usize as *mut c_void,
                0x9000usize as *mut c_void,
                0xa000usize as *mut c_void,
            ],
            8191,
            0.088_388_35,
            1e-6,
            16,
            32,
        );
        assert_eq!(args.len(), 100);
        for i in 0..10 {
            let want = 0x1000u64 * (i as u64 + 1);
            assert_eq!(
                u64::from_le_bytes(args[i * 8..i * 8 + 8].try_into().unwrap()),
                want
            );
        }
        assert_eq!(i32::from_le_bytes(args[80..84].try_into().unwrap()), 8191);
        assert_eq!(
            f32::from_le_bytes(args[84..88].try_into().unwrap()),
            0.088_388_35
        );
        assert_eq!(f32::from_le_bytes(args[88..92].try_into().unwrap()), 1e-6);
        assert_eq!(u32::from_le_bytes(args[92..96].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(args[96..100].try_into().unwrap()), 32);
        assert_eq!(BLOCK, [256, 1, 1]);
        assert_eq!(CHANNELS, 8192);
        for (rows, chunks) in [
            (1, 1),
            (63, 1),
            (64, 1),
            (65, 2),
            (4096, 64),
            (8191, 128),
            (8192, 128),
        ] {
            assert_eq!(grid(rows), [chunks, 8, 1], "rows {rows}");
        }
    }

    #[test]
    fn grid_matches_the_incumbent_launcher_for_the_fixed_geometry() {
        // `Gpu::gdn_chunk_prep`: grid [ceil(T / 64), ceil((2 * hk + hv) / 8), 1], block 256.
        for rows in [1usize, 64, 65, 513, 8192] {
            assert_eq!(
                grid(rows),
                [
                    rows.div_ceil(64) as u32,
                    (2 * KEY_HEADS + VALUE_HEADS).div_ceil(8) as u32,
                    1
                ]
            );
        }
        assert_eq!(KERNARG_BYTES, 10 * 8 + 3 * 4 + 2 * 4);
    }

    #[test]
    fn launcher_constants_match_certified_authoring_contract() {
        use peacemaker_isa::kernels::gdn_chunk_prep as author;
        assert_eq!(MODULE, author::MODULE);
        assert_eq!(SYMBOL, author::SYMBOL);
        assert_eq!(KERNARG_BYTES, author::KERNARG_BYTES as usize);
        assert_eq!(BLOCK[0] as usize, author::BLOCK as usize);
        assert_eq!(KEY_HEADS, author::HK as usize);
        assert_eq!(VALUE_HEADS, author::HV as usize);
        assert_eq!(HEAD_DIM, author::HEAD_DIM as usize);
        assert_eq!(GRID_Y as usize, author::GRID_Y as usize);
        assert_eq!(MAX_ROWS, author::MAX_ROWS as usize);
        assert_eq!(CHUNK_ROWS, author::CHUNK as usize);
    }

    #[test]
    fn the_embedded_image_carries_the_export() {
        assert!(!IMAGE.is_empty());
        assert!(
            IMAGE.windows(SYMBOL.len()).any(|w| w == SYMBOL.as_bytes()),
            "{SYMBOL} is not in the embedded image"
        );
    }

    #[test]
    fn shape_admission_is_exact() {
        for rows in [1, 2, 63, 64, 65, 4096, 8191, 8192] {
            assert_eq!(admit(rows, 16, 32), Ok(()), "rows {rows}");
        }
        for rows in [0, 8193, usize::MAX] {
            assert_eq!(
                shape_refusal(rows, 16, 32),
                Some(GdnChunkPrepPmRefusal::Rows)
            );
            assert!(admit(rows, 16, 32).is_err());
        }
        for (hk, hv) in [
            (16, 48),
            (16, 16),
            (16, 64),
            (8, 32),
            (32, 32),
            (0, 0),
            (16, 0),
            (0, 32),
        ] {
            assert_eq!(
                shape_refusal(64, hk, hv),
                Some(GdnChunkPrepPmRefusal::Shape),
                "{hk}/{hv}"
            );
            assert!(admit(64, hk, hv).is_err());
        }
        // The dense 27B geometry and out-of-range rows are refused, shape first.
        assert_eq!(shape_refusal(0, 16, 48), Some(GdnChunkPrepPmRefusal::Shape));
    }

    #[test]
    fn admission_requires_every_gate_in_order() {
        use GdnChunkPrepPmRefusal as R;
        let ok = |flag: bool, arch: &str, rows: usize, hk: usize, hv: usize| {
            Gpu::gdn_chunk_prep_pm_admission(flag, arch, rows, hk, hv)
        };
        assert_eq!(ok(true, "gfx1201", 1, 16, 32), Ok(()));
        assert_eq!(ok(true, "gfx1201", 8192, 16, 32), Ok(()));
        assert_eq!(ok(false, "gfx1201", 4096, 16, 32), Err(R::Disabled));
        for arch in [
            "gfx1200", "gfx1151", "gfx1100", "gfx942", "gfx906", "gfx1030", "unknown", "",
        ] {
            assert_eq!(
                ok(true, arch, 4096, 16, 32),
                Err(R::InventoryUnavailable),
                "{arch}"
            );
            assert_eq!(ok(false, arch, 4096, 16, 32), Err(R::Disabled), "{arch}");
        }
        assert_eq!(ok(true, "gfx1201", 4096, 16, 48), Err(R::Shape));
        assert_eq!(ok(true, "gfx1201", 0, 16, 32), Err(R::Rows));
        assert_eq!(ok(true, "gfx1201", 8193, 16, 32), Err(R::Rows));
    }

    #[test]
    fn unset_follows_the_exact_gfx1201_qualification_and_zero_kills() {
        use crate::prefill_route_capabilities::{enabled, PrefillRoute};
        let route = PrefillRoute::GdnChunkPrepPm;
        for arch in ["gfx1201", "gfx1151", "gfx1100", "gfx942", "unknown"] {
            let admit = |explicit: Option<bool>| {
                Gpu::gdn_chunk_prep_pm_admission(enabled(explicit, arch, route), arch, 4096, 16, 32)
            };
            assert_eq!(
                admit(None).is_ok(),
                arch == "gfx1201",
                "{arch}: unset follows the qualification"
            );
            assert!(admit(Some(false)).is_err(), "{arch}: 0 kills");
            assert_eq!(
                admit(Some(true)).is_ok(),
                arch == "gfx1201",
                "{arch}: 1 needs the inventory"
            );
        }
    }
}
