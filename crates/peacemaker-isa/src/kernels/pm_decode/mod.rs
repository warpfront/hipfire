//! PeaceMaker native decode twins (gfx1201). One file per HIP module; see qcal/release-0.4.2/pm-decode-twins/PLAN.md.
//!
//! Each author file exposes `build_gfx1201() -> Result<Vec<Emitted>, String>`, one
//! checked emission per selected export. [`emit_module`] consolidates them into the
//! module's single code-object source and proof with the shared
//! [`super::iu4_gemm::module`] combiner (one target header, one metadata document);
//! the native writer (`crate::native`) then links and bundles that text.
//!
//! FP-contract hazard: hipcc builds the HIP incumbents with
//! `-ffp-contract=fast`. For each 8-term group dot the backend always forms
//! `t_i = fma(sc, q_i, zp)`, but it then chooses between `fma(t1,x1, t0*x0)`
//! ("element 0 first") and `fma(t0,x0, t1*x1)` ("element 1 first"), and the
//! choice depends on the surrounding code, not on the expression. The two
//! round differently. Equal source therefore does not give equal bytes: a
//! twin, a layout change or an unrelated edit to the header loads can flip the
//! order and change about a third of the output rows. With the gfx1201
//! toolchain that shipped these twins, hipcc emits element 1 first everywhere
//! in the single-row kernels (`fused_qkvza`/`fused_qkv`/`fused_gate_up`) and,
//! in the dual-row kernels (`gemv_mq4g256v2_residual`,
//! `gemv_mq4g256v2_multirow_r2`), element 0 first on row 0 and element 1 first
//! on row 1, in the main loop and in the tails. These twins write every product
//! chain out explicitly (the per-row `order` arrays), so their DAG is pinned. A
//! new twin, or a HIP port meant to match them, has to pin the DAG in the same
//! way: an explicit `fmaf` chain under `#pragma clang fp contract(off)`. A
//! matching source expression is not an exactness argument. Pinning the HIP
//! incumbents themselves this way is byte-identical and throughput-neutral on
//! R9700 (branch `lab/gemv-dag-pin`).
pub mod fused_gate_up_hfq4g256_mq4v2;
pub mod fused_qkv_hfq4g256_mq4v2;
pub mod fused_qkvza_hfq4g256_mq4v2;
pub mod gemm_hfq4g256_rows_pm;
pub mod gemv_hfq4g256_multirow_default_mq4v2;
pub mod gemv_hfq4g256_residual_mq4v2;
pub mod gemv_hfq4g256_residual_xbatch_mq4v2;
pub mod gemv_hfq4g256_xbatch_mq4v2;

use super::iu4_gemm::ModuleProof;
use crate::Emitted;

/// One HIP module this tree twins: its name, the frozen selected exports in
/// appendix order, and the author's builder.
pub struct Module {
    pub name: &'static str,
    pub symbols: &'static [&'static str],
    pub build: fn() -> Result<Vec<Emitted>, String>,
}

/// Every PM decode module: the five W1 twins in PLAN W1 order, then new
/// batched symbols that replace no incumbent export.
pub const MODULES: [Module; 8] = [
    Module {
        name: "fused_gate_up_hfq4g256_mq4v2",
        symbols: &["fused_gate_up_mq4g256v2"],
        build: fused_gate_up_hfq4g256_mq4v2::build_gfx1201,
    },
    Module {
        name: "gemv_hfq4g256_residual_mq4v2",
        symbols: &["gemv_mq4g256v2_residual"],
        build: gemv_hfq4g256_residual_mq4v2::build_gfx1201,
    },
    Module {
        name: "fused_qkvza_hfq4g256_mq4v2",
        symbols: &["fused_qkvza_mq4g256v2"],
        build: fused_qkvza_hfq4g256_mq4v2::build_gfx1201,
    },
    Module {
        name: "gemv_hfq4g256_multirow_default_mq4v2",
        symbols: &["gemv_mq4g256v2_multirow_r2"],
        build: gemv_hfq4g256_multirow_default_mq4v2::build_gfx1201,
    },
    Module {
        name: "fused_qkv_hfq4g256_mq4v2",
        symbols: &["fused_qkv_mq4g256v2"],
        build: fused_qkv_hfq4g256_mq4v2::build_gfx1201,
    },
    Module {
        name: "gemv_hfq4g256_residual_xbatch_mq4v2",
        symbols: &["gemv_mq4g256v2_residual_xbatch"],
        build: gemv_hfq4g256_residual_xbatch_mq4v2::build_gfx1201,
    },
    Module {
        name: "gemv_hfq4g256_xbatch_mq4v2",
        symbols: &["gemv_mq4g256v2_xbatch_pm"],
        build: gemv_hfq4g256_xbatch_mq4v2::build_gfx1201,
    },
    Module {
        name: "gemm_hfq4g256_rows_pm",
        symbols: &["gemm_hfq4g256_rows_pm"],
        build: gemm_hfq4g256_rows_pm::build_gfx1201,
    },
];

pub fn module(name: &str) -> Result<&'static Module, String> {
    MODULES.iter().find(|m| m.name == name).ok_or_else(|| {
        let names: Vec<_> = MODULES.iter().map(|m| m.name).collect();
        format!("pm_decode: no module {name} ({})", names.join("|"))
    })
}

fn exported(e: &Emitted) -> Option<&str> {
    e.s_text
        .lines()
        .find_map(|l| l.trim().strip_prefix(".amdhsa_kernel "))
        .map(str::trim)
}

/// Build `name`'s exports and combine them into one module source and proof.
/// The exports must be exactly the module's frozen symbols, in order, built for
/// exact gfx1201: a diagnostic region symbol never ships under a module name.
pub fn emit_module(name: &str) -> Result<(Vec<Emitted>, String, ModuleProof), String> {
    let m = module(name)?;
    let emitted = (m.build)().map_err(|e| format!("{name}: {e}"))?;
    let got: Vec<&str> = emitted.iter().map(|e| exported(e).unwrap_or("")).collect();
    if got != m.symbols {
        return Err(format!(
            "{name}: builder exports {got:?}, the frozen selected exports are {:?}",
            m.symbols
        ));
    }
    if let Some(e) = emitted
        .iter()
        .find(|e| e.proof.arch != crate::Arch::Gfx1201)
    {
        return Err(format!(
            "{name}: {} built for {:?}, not exact gfx1201",
            exported(e).unwrap_or(""),
            e.proof.arch
        ));
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, name)?;
    Ok((emitted, text, proof))
}

/// The `peacemaker custom build --contract` shape record of a module's exports:
/// the frozen W1 ABI facts (wave32, no private segment or spills, no fixed or
/// dynamic LDS). One contract certifies one symbol.
pub fn shape_contract(name: &str) -> Result<serde_json::Value, String> {
    let m = module(name)?;
    let [symbol] = m.symbols else {
        return Err(format!(
            "{name}: a shape contract covers one symbol, module has {}",
            m.symbols.len()
        ));
    };
    Ok(serde_json::json!({
        "symbol": symbol, "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": 0,
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn module_table_is_unique_and_symbol_bound() {
        let mut names: Vec<_> = super::MODULES.iter().map(|m| m.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), super::MODULES.len());
        assert!(super::module("nope").is_err());
        for m in &super::MODULES {
            assert_eq!(
                super::shape_contract(m.name).unwrap()["symbol"],
                m.symbols[0]
            );
        }
    }
}
