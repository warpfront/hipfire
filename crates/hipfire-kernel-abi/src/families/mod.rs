// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contract data, one module per kernel family. Each module exports its
//! contracts as `pub static` [`KernelAbiV1`] values, its storage formats and
//! numeric contracts as `pub const`, and the list `ABIS`.

pub mod generic_elem;
pub mod generic_gemm;
pub mod generic_seq;
pub mod ggml_convert;
pub mod ggml_mmvq;
pub mod ggml_mmvq_fused;
pub mod mq6_i8;
pub mod qwen4_moe_sym;

use crate::work::WorkSpecV1;
use crate::{Arch, ExportV1, KernelAbiV1};

/// Every contract this crate defines, in a stable order (the identity
/// registry R2 census/corpus records and the drift tests iterate).
pub fn all() -> impl Iterator<Item = &'static KernelAbiV1> {
    mq6_i8::ABIS
        .iter()
        .chain(qwen4_moe_sym::ABIS.iter())
        .chain(ggml_mmvq::ABIS.iter())
        .chain(ggml_mmvq_fused::ABIS.iter())
        .chain(ggml_convert::ABIS.iter())
        .chain(generic_gemm::ABIS.iter())
        .chain(generic_seq::ABIS.iter())
        .chain(generic_elem::ABIS.iter())
        .copied()
}

/// Every work spec ([`WorkSpecV1`]), in a stable order. At most one per contract.
pub fn work_specs() -> impl Iterator<Item = &'static WorkSpecV1> {
    generic_gemm::WORK
        .iter()
        .chain(generic_seq::WORK.iter())
        .chain(generic_elem::WORK.iter())
        .copied()
}

/// The work spec attached to the contract `abi_id`, if any.
pub fn work_for(abi_id: &str) -> Option<&'static WorkSpecV1> {
    work_specs().find(|w| w.abi.id == abi_id)
}

/// The contract and export that own `symbol` on `arch` (export symbols are
/// unique per arch across all contracts). Allocation-free: launchers may call
/// it per launch.
pub fn find_export(arch: Arch, symbol: &str) -> Option<(&'static KernelAbiV1, &'static ExportV1)> {
    all().find_map(|abi| {
        abi.exports
            .iter()
            .find(|e| e.arch == arch && e.symbol == symbol)
            .map(|e| (abi, e))
    })
}

#[cfg(test)]
mod tests {
    use super::all;

    #[test]
    fn every_family_contract_is_self_consistent() {
        let mut ids = std::collections::BTreeSet::new();
        for abi in all() {
            if let Err(errors) = abi.self_check() {
                panic!("{}: {errors:#?}", abi.id);
            }
            assert!(ids.insert(abi.id), "duplicate contract id {}", abi.id);
        }
        assert_eq!(
            ids.len(),
            51 + super::generic_gemm::ABIS.len()
                + super::generic_seq::ABIS.len()
                + super::generic_elem::ABIS.len()
        );
    }

    #[test]
    fn every_work_spec_is_consistent_and_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for w in super::work_specs() {
            if let Err(e) = w.self_check() {
                panic!("{e}");
            }
            assert!(
                super::all().any(|a| std::ptr::eq(a, w.abi)),
                "{}: work spec for an unregistered contract",
                w.abi.id
            );
            assert!(seen.insert(w.abi.id), "two work specs for {}", w.abi.id);
        }
    }

    #[test]
    fn export_symbols_are_unique_per_arch_across_contracts() {
        let mut seen = std::collections::BTreeMap::new();
        for abi in all() {
            for e in abi.exports {
                if let Some(prev) = seen.insert((e.arch, e.symbol), abi.id) {
                    panic!(
                        "{} {} in both {prev} and {}",
                        e.arch.name(),
                        e.symbol,
                        abi.id
                    );
                }
            }
        }
    }
}
