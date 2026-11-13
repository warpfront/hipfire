// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PeaceMaker twin of the hipcc A3B native-fp8 Q-resident FA2 prefill attention
//! (`attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` and its range form): H16/KV2/D256,
//! native fp8-E4M3 KV (516-byte rows), two-plane fp8 Q. The image exports the direct and the
//! range symbol in one module and keeps the hipcc launch ABI exactly: the 68-byte kernarg of
//! `a3b_qresident_args`, grid `[ceil(min(batch,512)*8/384), 2, ceil(batch/512)]`, block
//! `[768,1,1]` and dynamic LDS 49,936 bytes. Selection is `HIPFIRE_FA_QRESIDENT_PM`
//! (default off) inside `Gpu::qresident_a3b_launch`, which keeps every shape, span and alias
//! check shared with the hipcc arm.
use crate::Gpu;
use hip_bridge::HipResult;

pub const MODULE: &str = "fa_qresident_a3b_pm_gfx1201";
pub const SYMBOL_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_pm_gfx1201";
pub const SYMBOL_RANGE: &str = "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/fa_qresident_a3b_pm_gfx1201.hxaco");
pub const KERNARG_BYTES: usize = 68;
pub const BLOCK: [u32; 3] = [768, 1, 1];
pub const LDS: u32 = 49_936;

impl Gpu {
    /// Load the PM module's `symbol` (one of [`SYMBOL_DIRECT`] / [`SYMBOL_RANGE`]).
    pub(crate) fn fa_qresident_pm_load(&mut self, symbol: &str) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_constants() {
        assert_eq!(MODULE, "fa_qresident_a3b_pm_gfx1201");
        assert_eq!(KERNARG_BYTES, 68);
        assert_eq!(BLOCK, [768, 1, 1]);
        assert_eq!(LDS, 49_936);
        assert_eq!(LDS, crate::attention::QRESIDENT_V2_LDS_BYTES);
        assert_ne!(SYMBOL_DIRECT, SYMBOL_RANGE);
        assert!(SYMBOL_DIRECT.ends_with("_pm_gfx1201"));
        assert!(SYMBOL_RANGE.ends_with("_pm_gfx1201"));
        assert!(!IMAGE.is_empty());
    }

    #[test]
    fn matches_the_generator_contract() {
        use peacemaker_isa::kernels::fa_qresident as author;
        assert_eq!(MODULE, author::MODULE);
        assert_eq!(SYMBOL_DIRECT, author::SYMBOL_DIRECT);
        assert_eq!(SYMBOL_RANGE, author::SYMBOL_RANGE);
        assert_eq!(KERNARG_BYTES as u32, author::KERNARG_BYTES);
        assert_eq!(BLOCK[0], author::BLOCK);
        assert_eq!(LDS, author::DYNAMIC_LDS_BYTES);
    }
}
