// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The sealed expert-parallel plan of one Flash-Next load.
//!
//! One EP mesh epoch, every rank's physical device and one sealed
//! [`ExpertExecutionPlan`] per routed-expert group (the trunk MoE layers, then
//! the MTP head), planned once from the source index by the shared
//! [`plan_packed_expert_parallel`]. Every rank's compact expert spans
//! ([`Qwen4EpPlacement`]), static ownership and load identity
//! ([`WeightOrigin`]) derive from it, and the mesh refuses a rank bound to any
//! other seal.

use crate::bundle::Qwen4Bundle;
use crate::weights::{Qwen4EpPlacement, Qwen4Manifest, WeightError};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::model_source::SourceIdentity;
use hipfire_runtime::sealed_moe::{
    plan_packed_expert_parallel, source_generation_fingerprint, ExpertExecutionPlan,
    PackedExpertGroup,
};
use hipfire_runtime::weight_store::WeightOrigin;
use std::sync::Arc;

/// Flash-Next's EP protocol: every rank runs the replicated trunk, owned
/// experts write their slot rows, and a copy-only owner row exchange feeds the
/// unchanged slot-order combine on every rank.
pub const QWEN4_EP_EXECUTION: &str = "qwen4-owner-row-gather-replicated-combine";

const GATE_UP: &str = ".mlp.experts.gate_up_proj";
const DOWN: &str = ".mlp.experts.down_proj";
const ROUTER: &str = ".mlp.gate.weight";

/// See the module docs. Immutable once sealed; shared by every rank of the
/// load through an [`Arc`].
#[derive(Debug)]
pub struct Qwen4EpSeal {
    mesh: DeviceMesh,
    plans: Vec<ExpertExecutionPlan>,
}

impl Qwen4EpSeal {
    /// Seal the routed expert groups of `manifest` (trunk layers labelled by
    /// layer, the MTP head as `trunk_layers`) over a fresh `ranks`-way EP mesh
    /// whose rank `r` runs on `physical_devices[r]`, from `source`'s sealed
    /// tensor index.
    pub fn seal(
        manifest: &Qwen4Manifest,
        trunk_layers: usize,
        ranks: usize,
        physical_devices: &[i32],
        source: &SourceIdentity,
    ) -> Result<Self, WeightError> {
        let refuse = |reason: String| WeightError::EpPlacement(format!("EP seal: {reason}"));
        let mesh =
            DeviceMesh::rect(&[(DimKind::Ep, ranks)]).map_err(|error| refuse(error.to_string()))?;
        let mut groups = Vec::new();
        for entry in &manifest.weights {
            let Some(prefix) = entry.name.strip_suffix(GATE_UP) else {
                continue;
            };
            let router = format!("{prefix}{ROUTER}");
            let down = format!("{prefix}{DOWN}");
            for name in [&router, &down] {
                if !manifest.weights.iter().any(|other| &other.name == name) {
                    return Err(refuse(format!("{} has no {name}", entry.name)));
                }
            }
            groups.push(PackedExpertGroup {
                layer: entry.layer.unwrap_or(trunk_layers),
                router,
                gate_up: entry.name.clone(),
                down,
            });
        }
        if groups.is_empty() {
            return Err(refuse(
                "the manifest declares no routed expert stacks".into(),
            ));
        }
        let plans = plan_packed_expert_parallel(
            &groups,
            "qwen4",
            QWEN4_EP_EXECUTION,
            source,
            &mesh,
            physical_devices,
        )
        .map_err(refuse)?;
        let seal = Self { mesh, plans };
        // Every rank's compact spans must be the `e % ranks` stride the
        // exchange kernels and the compact upload hard-wire.
        for rank in 0..ranks {
            seal.placement(rank)?;
        }
        Ok(seal)
    }

    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    pub fn ranks(&self) -> usize {
        self.mesh.n_devices()
    }

    /// HIP device of every rank, rank 0 first.
    pub fn physical_devices(&self) -> &[i32] {
        self.plans[0].physical_devices()
    }

    /// One sealed plan per routed-expert group, trunk layers first.
    pub fn plans(&self) -> &[ExpertExecutionPlan] {
        &self.plans
    }

    pub fn source_fingerprint(&self) -> &str {
        self.plans[0].source_fingerprint()
    }

    /// The load identity of `rank`: this seal's mesh epoch, the logical rank
    /// and its physical device.
    pub fn origin(&self, rank: usize) -> Option<WeightOrigin> {
        let device = *self.physical_devices().get(rank)?;
        Some(WeightOrigin::from_parts(self.mesh.epoch(), rank, device))
    }

    /// Refuse a peer source that is not the generation this seal was planned
    /// from.
    pub fn check_source(&self, source: &SourceIdentity) -> Result<(), String> {
        let actual = source_generation_fingerprint(source);
        if actual != self.source_fingerprint() {
            return Err(format!(
                "qwen4 EP seal: source {actual} is not the sealed generation {}",
                self.source_fingerprint()
            ));
        }
        Ok(())
    }

    /// `rank`'s compact expert spans, read from the sealed plans: every group
    /// must give expert `e` owner `e % ranks` and, on `rank`, local slot
    /// `e / ranks` exactly when it owns `e`.
    pub fn placement(&self, rank: usize) -> Result<Qwen4EpPlacement, WeightError> {
        let ranks = self.ranks();
        if rank >= ranks {
            return Err(WeightError::EpPlacement(format!(
                "EP seal: rank {rank} is outside {ranks} ranks"
            )));
        }
        let placement = Qwen4EpPlacement { rank, ranks };
        for plan in &self.plans {
            for expert in 0..plan.n_experts() {
                let slot = placement
                    .owns(expert)
                    .then(|| placement.local_index(expert));
                if plan.owner_rank(expert) != Some(expert % ranks)
                    || plan.local_slot(rank, expert) != slot
                {
                    return Err(WeightError::EpPlacement(format!(
                        "EP seal: {} expert {expert} on rank {rank} is owned by {:?} at slot {:?}, \
                         not the stride placement",
                        plan.group(),
                        plan.owner_rank(expert),
                        plan.local_slot(rank, expert)
                    )));
                }
            }
        }
        Ok(placement)
    }
}

/// A rank's binding to its load's seal (`Qwen4Bundle::bind_ep_seal`).
#[derive(Clone, Debug)]
pub struct Qwen4EpRankSeal {
    pub seal: Arc<Qwen4EpSeal>,
    pub rank: usize,
}

impl Qwen4Bundle {
    /// Bind this loaded rank to `seal`: the weights must have been fulfilled
    /// with exactly the seal's compact spans for `rank` and under the seal's
    /// load identity for it (mesh epoch, logical rank, physical device).
    pub fn bind_ep_seal(&mut self, seal: Arc<Qwen4EpSeal>, rank: usize) -> Result<(), String> {
        let placement = seal.placement(rank).map_err(|error| error.to_string())?;
        if self.weights.ep_placement() != Some(placement) {
            return Err(format!(
                "qwen4 EP seal: rank {rank} weights carry placement {:?}, the seal gives {placement:?}",
                self.weights.ep_placement()
            ));
        }
        let origin = seal.origin(rank);
        if self.attached_origin() != origin {
            return Err(format!(
                "qwen4 EP seal: rank {rank} weights were loaded as {:?}, the seal gives {origin:?}",
                self.attached_origin()
            ));
        }
        self.ep_seal = Some(Qwen4EpRankSeal { seal, rank });
        Ok(())
    }

    /// The seal this rank is bound to, if it is an EP rank.
    pub fn ep_seal(&self) -> Option<&Qwen4EpRankSeal> {
        self.ep_seal.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::model_source::{SourceFileIdentity, SourceFormat, SourceRangeIdentity};
    use rdna_compute::DType;

    /// The Flash-Next manifest's own tensor index: every entry encoded as its
    /// declared dtype.
    fn source_of(manifest: &Qwen4Manifest, mtime_secs: i64) -> SourceIdentity {
        let ranges = manifest
            .weights
            .iter()
            .map(|entry| {
                let cols = *entry.logical_shape.last().unwrap();
                let row_bytes = match entry.dtype {
                    DType::MQ4G256V2 => cols.div_ceil(256) * 136,
                    DType::MQ4G128V2 => cols.div_ceil(128) * 68,
                    _ => cols * 4,
                };
                let rows: usize = entry.logical_shape.iter().rev().skip(1).product();
                SourceRangeIdentity {
                    name: entry.name.clone(),
                    file_index: 0,
                    offset: 0,
                    length: (rows.max(1) * row_bytes) as u64,
                    dtype: format!("{:?}", entry.dtype),
                    logical_shape: entry.logical_shape.clone(),
                }
            })
            .collect();
        SourceIdentity {
            canonical_path: "/m.mq4".into(),
            format: SourceFormat::Hfq,
            files: vec![SourceFileIdentity {
                canonical_path: "/m.mq4".into(),
                dev: 1,
                ino: 2,
                len: 3,
                mtime_secs,
                mtime_nanos: 0,
            }],
            metadata_json: String::new(),
            manifest: ranges,
        }
    }

    #[test]
    fn seal_derives_every_rank_from_one_mesh_epoch_and_refuses_another_generation() {
        let manifest = Qwen4Manifest::build(&crate::config::compact_test_config()).unwrap();
        let source = source_of(&manifest, 10);
        for (ranks, devices) in [(2, vec![0, 1]), (6, vec![5, 4, 3, 2, 1, 0])] {
            let seal = Qwen4EpSeal::seal(&manifest, 48, ranks, &devices, &source).unwrap();
            // 48 trunk MoE layers, then the MTP head labelled after them.
            let layers: Vec<_> = seal.plans().iter().map(|plan| plan.layer()).collect();
            assert_eq!(layers, (0..49).map(Some).collect::<Vec<_>>());
            assert!(seal
                .plans()
                .iter()
                .all(|plan| plan.mesh_epoch() == seal.mesh().epoch()));
            for rank in 0..ranks {
                assert_eq!(
                    seal.placement(rank).unwrap(),
                    Qwen4EpPlacement { rank, ranks }
                );
                let origin = seal.origin(rank).unwrap();
                assert_eq!(origin.mesh_epoch(), seal.mesh().epoch());
                assert_eq!(
                    (origin.logical_rank(), origin.physical_device()),
                    (rank, devices[rank])
                );
            }
            assert!(seal.placement(ranks).is_err());
            assert_eq!(seal.origin(ranks), None);
            seal.check_source(&source).unwrap();
            let replaced = seal.check_source(&source_of(&manifest, 11)).unwrap_err();
            assert!(replaced.contains("not the sealed generation"), "{replaced}");
        }
        // Two seals of the same shape are two mesh epochs.
        let a = Qwen4EpSeal::seal(&manifest, 48, 2, &[0, 1], &source).unwrap();
        let b = Qwen4EpSeal::seal(&manifest, 48, 2, &[0, 1], &source).unwrap();
        assert_ne!(a.origin(1), b.origin(1));
    }
}
