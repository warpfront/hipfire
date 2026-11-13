// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Legality of the generated DPP / SDWA selector fields. The generated tables
//! describe `DPP_CTRL`, `SRC0_SEL`, `SRC1_SEL`, `DST_SEL` and `DST_UNUSED` as
//! bare bit ranges, so decode accepts values the ISA reserves; this module
//! refuses them.

use crate::xml::Decoded;

/// Refuse a reserved `DPP_CTRL` value, an SDWA select of 7, or a
/// `DST_UNUSED` of 3. Rows without these fields pass.
///
/// `DPP_CTRL` (CDNA3 ISA §13.3.9 Table 93; the gfx9 DPP set both CDNA parts
/// use): `quad_perm` 0x000-0x0ff, `row_shl` 0x101-0x10f, `row_shr`
/// 0x111-0x11f, `row_ror` 0x121-0x12f, `wave_shl1` 0x130, `wave_rol1` 0x134,
/// `wave_shr1` 0x138, `wave_ror1` 0x13c, `row_mirror` 0x140,
/// `row_half_mirror` 0x141, `row_bcast:15` 0x142, `row_bcast:31` 0x143 and
/// `row_newbcast` 0x150-0x15f. SDWA selects 0-6 are `BYTE_0`..`DWORD`;
/// `DST_UNUSED` 0-2 are `PAD`, `SEXT`, `PRESERVE`.
pub(crate) fn selectors(d: &Decoded) -> Result<(), String> {
    if let Some(ctrl) = d.field("DPP_CTRL") {
        if !dpp_ctrl_defined(ctrl) {
            return Err(format!("reserved DPP_CTRL {ctrl:#x}"));
        }
    }
    for name in ["DST_SEL", "SRC0_SEL", "SRC1_SEL"] {
        if let Some(sel) = d.field(name) {
            if sel > 6 {
                return Err(format!("reserved SDWA {name} {sel}"));
            }
        }
    }
    if let Some(unused) = d.field("DST_UNUSED") {
        if unused > 2 {
            return Err(format!("reserved SDWA DST_UNUSED {unused}"));
        }
    }
    Ok(())
}

fn dpp_ctrl_defined(ctrl: u64) -> bool {
    matches!(
        ctrl,
        0x000..=0x0ff
            | 0x101..=0x10f
            | 0x111..=0x11f
            | 0x121..=0x12f
            | 0x130
            | 0x134
            | 0x138
            | 0x13c
            | 0x140..=0x143
            | 0x150..=0x15f
    )
}
