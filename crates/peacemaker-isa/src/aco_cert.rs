// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! F4 for the ACO pilot: the compiler-independent static certificate.
//!
//! `certify_program` consumes a lifted program whose frontend is `Frontend::Aco`, runs the
//! shared `peacemaker_ir::edit::analyze` (wait replay, hazards, barriers, definedness,
//! LDS and resource/ABI findings) and returns a [`StaticCertificate`] only when the shared
//! analysis closed **every** obligation. Nothing is filtered, waived or downgraded here: an
//! open obligation is returned whole as `CertError::Obligations`. It is independent of the
//! Builder-specific `pm_check::m7` (which needs the ROCm toolchain and a builder `.s`), and
//! it labels its subject `Frontend::Aco`, never `Builder`.
//!
//! The same function certifies the raw program (domain [`IdentityDomain::Raw`]) and the HSA
//! program the adapter emitted (domain [`IdentityDomain::Hsa`]); the report keeps the two
//! identity domains apart. GPU equivalence is not part of this certificate: only a later
//! oracle receipt can attach it.
use peacemaker_ir::edit::analyze;
use peacemaker_ir::inst::{Abi, Frontend, KernelOrigin, Program, SymbolId};
use peacemaker_ir::state::{Analyzed, Lifted, Obligation, RoundTripProof};
use std::fmt;

/// Launch domain the certificate is claimed for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchContract {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic LDS the launch may request, bytes. Lifted kernels carry no dynamic-LDS
    /// bound, so a nonzero value cannot be certified and is rejected.
    pub dynamic_lds_max: u32,
    /// Bound inputs `(name, bytes)` the launch binds; recorded in the report, not assumed.
    pub bound_inputs: Vec<(String, u64)>,
}

/// Which identity the certified bytes belong to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityDomain {
    /// The producer's executable range (`Abi::Raw`).
    Raw,
    /// A code object with an HSA descriptor (`Abi::Hsa`).
    Hsa,
}

/// What was certified, by identity. `obligations` is empty by construction: a certificate
/// exists only when the shared analysis found none.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertificationReport {
    pub frontend: Frontend,
    pub domain: IdentityDomain,
    pub symbol: String,
    /// `RoundTripProof.input_sha256`: the whole binary / code object lifted.
    pub input_sha256: [u8; 32],
    /// `RoundTripProof.streams[].stream_sha256`.
    pub stream_sha256: Vec<[u8; 32]>,
    /// The raw sidecar hash (`Abi::Raw` only).
    pub sidecar_sha256: Option<[u8; 32]>,
    pub obligations: Vec<Obligation>,
    pub launch: LaunchContract,
    pub max_vgpr: u16,
    pub max_sgpr: u16,
}

/// The shared analyses' result for one kernel, owned (not cloned) from the lifted graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCertificate {
    pub analyzed: Analyzed<Program>,
    pub proof: RoundTripProof,
    pub report: CertificationReport,
    pub symbol: SymbolId,
}

#[derive(Debug)]
pub enum CertError {
    NoKernel(String),
    /// The subject is not an ACO-frontend kernel.
    NotAco(String),
    Invalid(String),
    Launch(String),
    Analysis(String),
    /// The shared analysis found obligations; all are listed, none is waived.
    Obligations(Vec<Obligation>),
}

impl fmt::Display for CertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKernel(s) => write!(f, "no kernel named {s}"),
            Self::NotAco(s) => write!(f, "not an ACO-frontend kernel: {s}"),
            Self::Invalid(s) => write!(f, "invalid program: {s}"),
            Self::Launch(s) => write!(f, "launch contract: {s}"),
            Self::Analysis(s) => write!(f, "shared analysis failed: {s}"),
            Self::Obligations(o) => {
                write!(f, "{} open obligation(s):", o.len())?;
                for ob in o {
                    write!(f, " [{} {:?}: {}]", ob.rule_id, ob.kind, ob.text)?
                }
                Ok(())
            }
        }
    }
}
impl std::error::Error for CertError {}

fn check_launch(launch: &LaunchContract) -> Result<(), CertError> {
    let bad = |m: String| Err(CertError::Launch(m));
    if launch.block.contains(&0) || launch.grid.contains(&0) {
        return bad(format!(
            "empty geometry grid {:?} block {:?}",
            launch.grid, launch.block
        ));
    }
    let threads: u64 = launch.block.iter().map(|&d| u64::from(d)).product();
    if threads > 1024 {
        return bad(format!(
            "block {:?} has {threads} work-items (max 1024)",
            launch.block
        ));
    }
    if launch.dynamic_lds_max != 0 {
        return bad(format!("dynamic LDS maximum {} bytes: a lifted kernel carries no dynamic-LDS bound, so it is unproved", launch.dynamic_lds_max));
    }
    Ok(())
}

/// Certify `symbol` of `lifted` for `launch` with the shared analyses.
///
/// The kernel must carry `Frontend::Aco` provenance (raw ACO executable or the HSA
/// object the ACO adapter emitted). Every obligation the analysis reports fails the call.
pub fn certify_program(
    lifted: Lifted<Program>,
    symbol: &SymbolId,
    launch: &LaunchContract,
) -> Result<StaticCertificate, CertError> {
    check_launch(launch)?;
    lifted
        .validate()
        .map_err(|e| CertError::Invalid(e.to_string()))?;
    let Lifted { program, proof, .. } = lifted;
    let kernel = program
        .kernels
        .iter()
        .find(|k| &k.symbol == symbol)
        .ok_or_else(|| CertError::NoKernel(symbol.0.clone()))?;
    match &kernel.origin {
        KernelOrigin::Frontend {
            kind: Frontend::Aco,
            ..
        } => {}
        other => return Err(CertError::NotAco(format!("{}: origin {other:?}", symbol.0))),
    }
    let (domain, sidecar_sha256) = match &kernel.abi {
        Abi::Raw(raw) => (IdentityDomain::Raw, Some(raw.sidecar_sha256)),
        Abi::Hsa { .. } => (IdentityDomain::Hsa, None),
    };
    let analyzed = analyze(program, symbol).map_err(|e| CertError::Analysis(e.to_string()))?;
    if !analyzed.obligations.is_empty() {
        return Err(CertError::Obligations(analyzed.obligations));
    }
    let summary =
        analyzed.facts.resources.first().ok_or_else(|| {
            CertError::Analysis("the analysis produced no resource summary".into())
        })?;
    if summary.uses_flat_scratch {
        return Err(CertError::Analysis(
            "flat scratch is used; scratch bounds are not certified".into(),
        ));
    }
    let report = CertificationReport {
        frontend: Frontend::Aco,
        domain,
        symbol: symbol.0.clone(),
        input_sha256: proof.input_sha256,
        stream_sha256: proof.streams.iter().map(|s| s.stream_sha256).collect(),
        sidecar_sha256,
        obligations: Vec::new(),
        launch: launch.clone(),
        max_vgpr: summary.max_vgpr,
        max_sgpr: summary.max_sgpr,
    };
    Ok(StaticCertificate {
        analyzed,
        proof,
        report,
        symbol: symbol.clone(),
    })
}
