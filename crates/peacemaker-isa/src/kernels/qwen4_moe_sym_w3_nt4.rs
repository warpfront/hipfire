// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WeightCodec::Mq3I4` NT4 expert-run entries of the grouped symmetric IU4 MoE GEMM,
//! gfx1151 only: the module `qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151` holds, in this order,
//! gate/up `qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1151_nt4`, down
//! `qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4`, the down row repeats `..._nt4r2` and
//! `..._nt4r4`, then the F32 `_acc` qualification twins of gate/up and down. The bodies are
//! `qwen4_moe_sym_w3_nt4_gfx1151.rs` (mounted here); every other architecture has no entries
//! (`module_specs` is empty, `validate`/`emit` refuse).
use super::super::*;
use peacemaker_author::{Free, LdsRegion, StoreTarget, Wave};

/// The gfx1151 bodies: the MQ4 gfx1151 NT4 operand layout with the gfx1151 W3 weight side.
#[path = "qwen4_moe_sym_w3_nt4_gfx1151.rs"]
mod gfx1151;
type R = Result<(), String>;

/// The explicit subset this module builds: gfx1151, MQ3-I4, `nt == 4`; `rr == 1`, or 2 / 4 for the down entry.
fn validate(spec: Spec) -> R {
    gfx1151::validate(spec)
}

/// Every entry of the module in emission order: gate/up, down, then the down row repeats (`nt4r2`, `nt4r4`).
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    if arch == Arch::Gfx1151 {
        gfx1151::module_specs()
    } else {
        Vec::new()
    }
}

pub fn module_name(arch: Arch) -> String {
    format!("qwen4_moe_iu4_sym_w3_nt4_pm_{}", arch.name())
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, false)
}

/// Qualification entry `..._acc`: the same K loop and tile walk, but stores each live tile's F32
/// accumulators (`Y[P][M]` f32) instead of the BF16 epilogue. `rr == 1` entries only. Never dispatched.
pub fn emit_acc(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, true)
}

fn emit_impl(spec: Spec, acc: bool) -> Result<Emitted, String> {
    validate(spec)?;
    gfx1151::emit_gfx1151(spec, acc)
}

/// Every entry of the NT4 module (gate/up, down, the two down row repeats, then the `_acc` twins of
/// the two `rr == 1` entries) as one code object `qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, crate::kernels::iu4_gemm::ModuleProof), String> {
    let specs = module_specs(arch);
    if specs.is_empty() {
        return Err(format!(
            "qwen4_moe_sym w3 nt4: no entries for {}",
            arch.name()
        ));
    }
    let mut emitted = specs
        .iter()
        .map(|&s| emit(s))
        .collect::<Result<Vec<_>, _>>()?;
    for &s in specs.iter().filter(|s| s.rr == 1) {
        emitted.push(emit_acc(s)?);
    }
    let (text, proof) = crate::kernels::iu4_gemm::module(&emitted, &module_name(arch))?;
    Ok((emitted, text, proof))
}
