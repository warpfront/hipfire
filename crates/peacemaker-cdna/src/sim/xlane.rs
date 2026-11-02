// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-lane data movement: DPP and SDWA operand semantics for the VALU,
//! and the `ds_swizzle_b32` / `ds_permute_b32` / `ds_bpermute_b32` lane
//! functions for the DS unit (CDNA3 ISA §13.3.3-13.3.4, §11 DS_SWIZZLE /
//! DS_PERMUTE / DS_BPERMUTE).
//!
//! Page numbers are the printed "N of 553" footers of the ISA guide:
//! SDWA §13.3.7 p.542-543, DPP fields §13.3.9 p.544, DPP_CTRL table 93
//! p.544-545, DS_SWIZZLE_B32 §12.12 p.427, DS_PERMUTE_B32 p.427-428,
//! DS_BPERMUTE_B32 p.429.
//!
//! These functions return raw 32-bit lane values; the VALU executor applies
//! the SRC0/SRC1 NEG/ABS modifiers (DPP and SDWA) and the EXEC write mask,
//! and performs the actual VGPR write.
use super::{fault, unsupported, SimResult, Wave};
use crate::enc::vop::{Dpp, DppCtrl, SdwaSel, SdwaUnused};
use crate::WAVE;

/// The lane a destination lane reads under `ctrl` (CDNA3 ISA §13.3.9
/// Table 93 p.544-545, `n` = destination lane, wave64).
///
/// `Ok(None)` = the control defines no source lane for `n` (out of the row /
/// wave for the shifts, or the lanes a broadcast does not feed); the caller
/// then applies BOUND_CTRL (shifts) or leaves the lane unwritten (bcast).
/// Errors: lane >= 64, a quad_perm element > 3, a row shift/rotate count
/// outside 1..=15.
///
/// | control      | rule (Table 93)                                            |
/// |--------------|------------------------------------------------------------|
/// | quad_perm    | `(n & 0x3c) + sel[n % 4]` (`sel[0]` = `cntl[1:0]`)           |
/// | row_shl:c    | `(n & 15) < 16-c` → `n + c`                                |
/// | row_shr:c    | `(n & 15) >= c` → `n - c`                                  |
/// | row_ror:c    | `(n & 15) >= c` → `n - c`, else `n + 16 - c`               |
/// | wave_shl1    | `n < 63` → `n + 1`                                         |
/// | wave_rol1    | `n < 63` → `n + 1`, else lane 0                            |
/// | wave_shr1    | `n > 0` → `n - 1`                                          |
/// | wave_ror1    | `n > 0` → `n - 1`, else lane 63                            |
/// | row_mirror   | `15 - (n & 15)` in the row of `n`                          |
/// | row_half_mirror | `7 - (n & 7)` in the 8-lane half of `n`                 |
/// | row_bcast:15 | `n > 15` → `(n & 0x30) - 1` (lane 15 → row 1, 31 → row 2, 47 → row 3) |
/// | row_bcast:31 | `n > 31` → `(n & 0x20) - 1` (lane 31 → rows 2 and 3)       |
///
/// The table prints `pix[n & 0x30 - 1]` / `pix[n & 0x20 - 1]`; the intended
/// reading (and the stated purpose, "broadcast the 15th thread of each row
/// to the next row") is `(n & 0x30) - 1`. Table 93 gives rows 1..3 as
/// written by `row_bcast:15` and rows 2..3 by `row_bcast:31`; assemblers
/// restrict the written rows with ROW_MASK (a prefix scan uses `0xa` / `0xc`).
/// The half_mirror row of Table 93 reads `7-(n&7)` with no `n & 0x8`
/// offset; the rule here keeps the lane inside its own 8-lane half
/// (`(n & 8) + 7 - (n & 7)`), the only reading under which each half mirrors
/// onto itself. [UNVERIFIED: confirm on MI300X]
pub fn dpp_source_lane(ctrl: DppCtrl, n: usize) -> SimResult<Option<usize>> {
    if n >= WAVE {
        return fault(format!("DPP lane {n} outside the wave64"));
    }
    let row_count = |c: u8| -> SimResult<usize> {
        if (1..=15).contains(&c) {
            Ok(usize::from(c))
        } else {
            unsupported(format!("DPP row shift/rotate count {c} (legal 1..=15)"))
        }
    };
    Ok(match ctrl {
        DppCtrl::QuadPerm(sel) => {
            let s = sel[n % 4];
            if s > 3 {
                return fault(format!("DPP quad_perm element {s} > 3"));
            }
            Some((n & 0x3c) + usize::from(s))
        }
        DppCtrl::RowShl(c) => {
            let c = row_count(c)?;
            (n & 15 < 16 - c).then(|| n + c)
        }
        DppCtrl::RowShr(c) => {
            let c = row_count(c)?;
            (n & 15 >= c).then(|| n - c)
        }
        DppCtrl::RowRor(c) => {
            let c = row_count(c)?;
            Some(if n & 15 >= c { n - c } else { n + 16 - c })
        }
        DppCtrl::WaveShl1 => (n < 63).then(|| n + 1),
        DppCtrl::WaveRol1 => Some(if n < 63 { n + 1 } else { 0 }),
        DppCtrl::WaveShr1 => (n > 0).then(|| n - 1),
        DppCtrl::WaveRor1 => Some(if n > 0 { n - 1 } else { 63 }),
        DppCtrl::RowMirror => Some((n & 0x30) + 15 - (n & 15)),
        DppCtrl::RowHalfMirror => Some((n & 0x38) + 7 - (n & 7)),
        DppCtrl::RowBcast15 => (n > 15).then(|| (n & 0x30) - 1),
        DppCtrl::RowBcast31 => (n > 31).then(|| (n & 0x20) - 1),
    })
}

/// The DPP source value lane `lane` reads from VGPR `vgpr` (or AGPR when
/// `agpr`): `Some(value)`, or `None` when the lane must not write (source
/// lane out of range or inactive without BOUND_CTRL, or row/bank masked off).
///
/// Rules (CDNA3 ISA §13.3.9 p.544-545):
/// * the destination lane must be enabled by EXEC and by ROW_MASK/BANK_MASK
///   ([`dpp_lane_enabled`]); the masks gate the destination write only, never
///   the source fetch;
/// * the source lane comes from [`dpp_source_lane`];
/// * no source lane (out of row/wave) or a source lane disabled in EXEC:
///   BOUND_CTRL = 1 → the lane is written with 0, BOUND_CTRL = 0 → the lane
///   is not written ("BC: 0 = do not write when source is out of range,
///   1 = write", p.544; the table's "else use bound_cntl"). `row_bcast`
///   lanes the control does not feed (rows 0 for :15, rows 0-1 for :31) are
///   left unwritten regardless of BOUND_CTRL, as the table lists no `else`.
///   [UNVERIFIED: confirm on MI300X] — whether a source lane disabled in EXEC
///   reads 0 under BOUND_CTRL (implemented, the table's "inactive source =
///   invalid" reading) or the stale VGPR contents, and whether BOUND_CTRL
///   also writes 0 on the unfed `row_bcast` rows.
///
/// NEG/ABS are not applied here.
pub fn dpp_read(w: &Wave, dpp: &Dpp, vgpr: u8, agpr: bool, lane: usize) -> SimResult<Option<u32>> {
    if lane >= WAVE {
        return fault(format!("DPP lane {lane} outside the wave64"));
    }
    let (file, limit, name) = if agpr {
        (&w.a, w.agpr_limit, 'a')
    } else {
        (&w.v, w.vgpr_limit, 'v')
    };
    if u16::from(vgpr) >= limit || usize::from(vgpr) >= file.len() {
        return fault(format!(
            "DPP source {name}{vgpr} beyond the {limit} allocated by the kernel descriptor"
        ));
    }
    if !w.lane_active(lane) || !dpp_lane_enabled(dpp, lane) {
        return Ok(None);
    }
    let src = dpp_source_lane(dpp.ctrl, lane)?;
    match src {
        Some(s) if w.lane_active(s) => Ok(Some(file[usize::from(vgpr)][s])),
        // Lanes a broadcast does not feed are never written.
        None if matches!(dpp.ctrl, DppCtrl::RowBcast15 | DppCtrl::RowBcast31) => Ok(None),
        _ => Ok(dpp.bound_ctrl.then_some(0)),
    }
}

/// Lane `lane` writes under this DPP control's ROW_MASK / BANK_MASK
/// (CDNA3 ISA §13.3.9 p.544): ROW_MASK bit `r` clear disables lanes
/// `16r..16r+15`; BANK_MASK bit `b` clear disables lanes `4b..4b+3` of every
/// row. Lanes >= 64 are never enabled.
pub fn dpp_lane_enabled(dpp: &Dpp, lane: usize) -> bool {
    lane < WAVE && dpp.row_mask >> (lane / 16) & 1 != 0 && dpp.bank_mask >> (lane % 16 / 4) & 1 != 0
}

/// An SDWA source: select `sel` of `raw`, zero- or sign-extended (`sext`)
/// to 32 bits (CDNA3 ISA §13.3.7 p.543: SRC*_SEL / SRC*_SEXT). `Dword` is the
/// whole dword (SEXT has no effect).
pub fn sdwa_src(raw: u32, sel: SdwaSel, sext: bool) -> u32 {
    let (shift, bits) = sel_field(sel);
    let v = (raw >> shift) & field_mask(bits);
    if sext && bits < 32 {
        (((v << (32 - bits)) as i32) >> (32 - bits)) as u32
    } else {
        v
    }
}

/// (lowest bit, width) of an SDWA select.
fn sel_field(sel: SdwaSel) -> (u32, u32) {
    match sel {
        SdwaSel::Byte0 => (0, 8),
        SdwaSel::Byte1 => (8, 8),
        SdwaSel::Byte2 => (16, 8),
        SdwaSel::Byte3 => (24, 8),
        SdwaSel::Word0 => (0, 16),
        SdwaSel::Word1 => (16, 16),
        SdwaSel::Dword => (0, 32),
    }
}

fn field_mask(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

/// An SDWA destination: place `result` at `sel` of the old value per
/// DST_UNUSED (CDNA3 ISA §13.3.7 p.543, DST_SEL / DST_U): the low bits of
/// `result` fill the selected field; the bits outside it are
/// * `Pad`: zero,
/// * `Sext`: bits above the field = sign bit of the field, bits below = 0
///   ("sign extend upper / zero lower"),
/// * `Preserve`: the old value.
///
/// `Dword` replaces the whole register (DST_UNUSED has nothing to apply to).
/// The ISA table lists DST_SEL 4/5/6 (word0/word1/dword) and reserves 0-3;
/// the byte selects are the assembler-accepted `dst_sel:BYTE_n` forms.
/// [UNVERIFIED: confirm on MI300X] — byte destination selects.
pub fn sdwa_dst(old: u32, result: u32, sel: SdwaSel, unused: SdwaUnused) -> u32 {
    let (shift, bits) = sel_field(sel);
    if bits == 32 {
        return result;
    }
    let mask = field_mask(bits);
    let field = (result & mask) << shift;
    match unused {
        SdwaUnused::Pad => field,
        SdwaUnused::Preserve => (old & !(mask << shift)) | field,
        SdwaUnused::Sext => {
            let top = shift + bits;
            let upper = if top >= 32 {
                0
            } else if field >> (top - 1) & 1 != 0 {
                u32::MAX << top
            } else {
                0
            };
            upper | field
        }
    }
}

/// `ds_swizzle_b32 offset:pattern`: per-lane result (lanes not in `exec`
/// unchanged by the caller; reads from inactive lanes give 0).
///
/// CDNA3 ISA §12.12 DS_SWIZZLE_B32 p.427 (`offset = offset1:offset0`, all
/// lane indices inside the pseudocode are within 32-lane halves of the
/// wave64: `j |= i & 0x20`):
/// * `offset >= 0xe000` (FFT): `j = (reverse5(i & 0x1f) >> popcount(mask)) | (i & mask) | (i & 0x20)`,
///   `mask = offset[4:0]`. The pseudocode's `reverse_bits(i & 0x1f)` is taken
///   as a 5-bit reversal (a 32-bit reversal would leave the result outside
///   the group). [UNVERIFIED: confirm on MI300X]
/// * `offset >= 0xc000` (rotate): `rotate = offset[9:5]` (negated if
///   `offset[10]`), `mask = offset[4:0]`,
///   `j = (i & mask) | ((i + rotate) & ~mask)` with the sum taken modulo 32
///   (stays in the lane group). [UNVERIFIED: confirm on MI300X] — the
///   modulo-32 wrap, which the pseudocode leaves implicit.
/// * `offset[15]` (quad-perm, `0x8000..=0xbfff`): lane `i+k` of each 4-group
///   reads group lane `offset[2k+1:2k]`.
/// * else (bitmask): `j = (((i & 0x1f) & and_mask) | or_mask) ^ xor_mask`,
///   `and_mask = offset[4:0]`, `or_mask = offset[9:5]`, `xor_mask = offset[14:10]`.
///
/// The result is `thread_in[j]` if lane `j` is enabled in `exec`, else 0.
/// The pattern is a 16-bit field, so no value is illegal.
pub fn ds_swizzle(pattern: u16, data: &[u32; WAVE], exec: u64) -> SimResult<[u32; WAVE]> {
    let offset = u32::from(pattern);
    let mut out = [0u32; WAVE];
    for (i, o) in out.iter_mut().enumerate() {
        let i = i as u32;
        let j = if offset >= 0xe000 {
            let mask = offset & 0x1f;
            let rev = (i & 0x1f).reverse_bits() >> 27;
            (rev >> mask.count_ones()) | (i & mask) | (i & 0x20)
        } else if offset >= 0xc000 {
            let mask = offset & 0x1f;
            let rot = (offset >> 5) & 0x1f;
            let rot = if offset >> 10 & 1 != 0 {
                rot.wrapping_neg()
            } else {
                rot
            };
            ((i & mask) | (i.wrapping_add(rot) & !mask & 0x1f)) | (i & 0x20)
        } else if offset >> 15 & 1 != 0 {
            let sel = (offset >> (2 * (i & 3))) & 3;
            (i & !3) + sel
        } else {
            let and_mask = offset & 0x1f;
            let or_mask = (offset >> 5) & 0x1f;
            let xor_mask = (offset >> 10) & 0x1f;
            ((((i & 0x1f) & and_mask) | or_mask) ^ xor_mask) | (i & 0x20)
        } as usize;
        *o = if exec >> j & 1 != 0 { data[j] } else { 0 };
    }
    Ok(out)
}

/// Lane index `(addr + offset) / 4 % 64` (CDNA3 ISA p.428-429: "ADDR needs to
/// be divided by 4. High-order bits are ignored").
fn perm_lane(addr: u32, offset: u16) -> usize {
    (addr.wrapping_add(u32::from(offset)) / 4 % 64) as usize
}

/// `ds_bpermute_b32`: lane i reads `data[(addr[i] + offset) / 4 % 64]`
/// (CDNA3 ISA §12.12 DS_BPERMUTE_B32 p.429). "EXEC is applied to the source
/// VGPR reads": a source lane disabled in `exec` yields 0. Lanes not in
/// `exec` are not written; their entry is returned as 0 and the caller must
/// leave the destination register unchanged for them.
pub fn ds_bpermute(addr: &[u32; WAVE], offset: u16, data: &[u32; WAVE], exec: u64) -> [u32; WAVE] {
    let mut out = [0u32; WAVE];
    for i in 0..WAVE {
        if exec >> i & 1 == 0 {
            continue;
        }
        let src = perm_lane(addr[i], offset);
        if exec >> src & 1 != 0 {
            out[i] = data[src]
        }
    }
    out
}

/// `ds_permute_b32`: lane i writes `data[i]` to lane `(addr[i] + offset) / 4 % 64`
/// (CDNA3 ISA §12.12 DS_PERMUTE_B32 p.427-428). Only lanes enabled in `exec`
/// send data ("a disabled source thread does not propagate data"); a
/// destination lane nobody selects receives 0; when several sources select
/// one lane the highest-numbered source wins. Lanes not in `exec` are not
/// written; their entry is returned as 0 and the caller must leave the
/// destination register unchanged for them.
pub fn ds_permute(addr: &[u32; WAVE], offset: u16, data: &[u32; WAVE], exec: u64) -> [u32; WAVE] {
    let mut tmp = [0u32; WAVE];
    for i in 0..WAVE {
        if exec >> i & 1 != 0 {
            tmp[perm_lane(addr[i], offset)] = data[i]
        }
    }
    for (i, t) in tmp.iter_mut().enumerate() {
        if exec >> i & 1 == 0 {
            *t = 0
        }
    }
    tmp
}
