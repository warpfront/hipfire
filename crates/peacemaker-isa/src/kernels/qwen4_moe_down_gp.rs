// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Grouped-position down activation plane for the gfx1201 A3B MoE (module `qwen4_moe_down_gp_pm_gfx1201`, bundle
//! `kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco`). Three byte-identical twins of the flat-slot route, in this order:
//!
//! * `moe_rotate_i4_gp_pm_gfx1201`: [`super::moe_rotate_i4_pm::emit_gp`], the down A4 producer writing record
//!   `g * P + row` (72 bytes: `d` F32, `s` I32, 64 `qs`) for the grouped row `row` instead of `g * L + sorted[row]`.
//! * `qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4` / `..._nt8`: [`super::qwen4_moe_sym::emit_down_gp`], the NT4 / NT8 down
//!   entries reading record `g * m_total + 16 * tile + lane` instead of gathering `sorted`.
//!
//! The record bytes, every fold, every rounding and every store are those of the incumbent entries (the originals stay in
//! `qwen4_moe_iu4_sym_pm_gfx1201` / `moe_rotate_i4_pm_gfx1201` as the oracle anchors); only where a record lives
//! changes, so a tile's 16 slots read 1152 adjacent bytes per epoch instead of 16 scattered records.
//!
//! The F1 drop-zero planar twin of the module (`qwen4_moe_down_gp_p_pm_gfx1201`, bundle
//! `kernels/qwen4_moe_down_gp_p_pm_gfx1201.hxaco`, [`super::qwen4_moe_sym::WeightCodec::Mq4SymPlanar`]) holds the same
//! three exports in the same order: the unchanged producer (same symbol, same bytes) and
//! `qwen4_moe_down_iu4_sym_p_gp_pm_gfx1201_nt4` / `..._nt8`, which read the planar qt138 down tensor
//! (`[codes plane][scales plane]`) instead of the shipped 68-byte-group layout.
use super::{moe_rotate_i4_pm, qwen4_moe_sym};
use crate::{Arch, Emitted};
use qwen4_moe_sym::WeightCodec;

/// Module key (also the bundle stem) of the shipped-layout module.
pub fn module_name(arch: Arch) -> String {
    module_name_codec(arch, WeightCodec::Mq4Sym)
}

/// Module key (also the bundle stem) for one weight codec: the shipped layout or the planar twin.
pub fn module_name_codec(arch: Arch, codec: WeightCodec) -> String {
    match codec {
        WeightCodec::Mq4SymPlanar => format!("qwen4_moe_down_gp_p_pm_{}", arch.name()),
        _ => format!("qwen4_moe_down_gp_pm_{}", arch.name()),
    }
}

fn down_spec(arch: Arch, nt: u8, codec: WeightCodec) -> qwen4_moe_sym::Spec {
    qwen4_moe_sym::Spec {
        arch,
        kind: qwen4_moe_sym::Kind::Down,
        nt,
        rr: 1,
        codec,
    }
}

/// The three exports in emission order.
pub fn symbols(arch: Arch) -> [String; 3] {
    symbols_codec(arch, WeightCodec::Mq4Sym)
}

/// The three exports of one weight codec's module in emission order (the producer is codec-independent).
pub fn symbols_codec(arch: Arch, codec: WeightCodec) -> [String; 3] {
    let down = |nt| down_spec(arch, nt, codec).symbol_gp();
    [moe_rotate_i4_pm::symbol_gp(arch), down(4), down(8)]
}

pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    emit_module_codec(arch, WeightCodec::Mq4Sym)
}

/// [`emit_module`] for one weight codec ([`WeightCodec::Mq4Sym`] or [`WeightCodec::Mq4SymPlanar`]).
pub fn emit_module_codec(
    arch: Arch,
    codec: WeightCodec,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    if arch != Arch::Gfx1201 {
        return Err("qwen4_moe_down_gp: exact gfx1201 only".into());
    }
    if !matches!(codec, WeightCodec::Mq4Sym | WeightCodec::Mq4SymPlanar) {
        return Err(format!(
            "qwen4_moe_down_gp: codec {codec:?} has no grouped-position down"
        ));
    }
    let down = |nt| qwen4_moe_sym::emit_down_gp(down_spec(arch, nt, codec));
    let emitted = vec![
        moe_rotate_i4_pm::emit_gp(moe_rotate_i4_pm::Spec { arch })?,
        down(4)?,
        down(8)?,
    ];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module_name_codec(arch, codec))?;
    Ok((emitted, text, proof))
}
