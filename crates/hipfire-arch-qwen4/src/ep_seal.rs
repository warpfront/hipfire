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
use hipfire_dispatch::pipeline::{ExpertBindingCache, ExpertTable};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::{capture_file_identity, SourceIdentity};
use hipfire_runtime::sealed_moe::{
    adapt_expert_execution_plan, plan_packed_expert_parallel, source_generation_fingerprint,
    ExpertExecutionPlan, PackedExpertGroup,
};
use hipfire_runtime::weight_store::WeightOrigin;
use std::path::Path;
use std::sync::Arc;

/// Flash-Next's EP protocol: every rank runs the replicated trunk, owned
/// experts write their slot rows, and a copy-only owner row exchange feeds the
/// unchanged slot-order combine on every rank. The canonical execution string
/// lives with the sealed planner so the runtime can recognize the owner-routed
/// exchange (rather than an all-reduce) when it plans the collective schedule.
pub const QWEN4_EP_EXECUTION: &str = hipfire_runtime::sealed_moe::OWNER_ROUTED_EP_EXECUTION;

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

/// The source generation of a Flash-Next load from `hfq` (opened from
/// `model_path`, PLE rows already resolved and attached): the base file and
/// its HFQ overlay, the attached PLE rows sidecar, and the trunk-overlay
/// sidecar beside the model when one exists. Every rank re-resolves both
/// sidecars for itself, so they are part of what the seal binds. Called
/// after the sidecars are bound, a sidecar replaced or created in between
/// reads as another generation (fail closed).
pub fn qwen4_source_generation(hfq: &HfqFile, model_path: &Path) -> Result<String, String> {
    let identity = |path: &Path| {
        capture_file_identity(path)
            .map_err(|error| format!("qwen4 EP: source generation of {}: {error}", path.display()))
    };
    let mut files = hfq
        .source_identity()
        .map_err(|error| format!("qwen4 EP: source identity: {error}"))?
        .files
        .clone();
    if let Some(rows) = hfq.ple_rows() {
        files.push(identity(rows.path())?);
    }
    let overlay = hipfire_config::memory::trunk_overlay_sidecar_path(model_path);
    if overlay.exists() {
        files.push(identity(&overlay)?);
    }
    Ok(source_generation_fingerprint(&files))
}

impl Qwen4EpSeal {
    /// Seal the routed expert groups of `manifest` (trunk layers labelled by
    /// layer, the MTP head as `trunk_layers`) over a fresh `ranks`-way EP mesh
    /// whose rank `r` runs on `physical_devices[r]`, from `source`'s sealed
    /// tensor index, binding `generation` ([`qwen4_source_generation`]).
    pub fn seal(
        manifest: &Qwen4Manifest,
        trunk_layers: usize,
        ranks: usize,
        physical_devices: &[i32],
        source: &SourceIdentity,
        generation: &str,
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
            generation,
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

    /// The source generation every rank must load from
    /// ([`qwen4_source_generation`]).
    pub fn source_generation(&self) -> &str {
        self.plans[0].source_fingerprint()
    }

    /// The load identity of `rank`: this seal's mesh epoch, the logical rank
    /// and its physical device.
    pub fn origin(&self, rank: usize) -> Option<WeightOrigin> {
        let device = *self.physical_devices().get(rank)?;
        Some(WeightOrigin::from_parts(self.mesh.epoch(), rank, device))
    }

    /// Refuse a rank whose source generation ([`qwen4_source_generation`]:
    /// base file, PLE rows and trunk-overlay sidecars) is not the one this
    /// seal was planned from.
    pub fn check_source(&self, generation: &str) -> Result<(), String> {
        if generation != self.source_generation() {
            return Err(format!(
                "qwen4 EP seal: source {generation} is not the sealed generation {}",
                self.source_generation()
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

    /// Owner rank of every global expert, read from the sealed plans (every
    /// group shares it: [`Self::placement`] proved the stride on each).
    pub fn owner_ranks(&self) -> Vec<usize> {
        let plan = &self.plans[0];
        (0..plan.n_experts())
            .map(|expert| plan.owner_rank(expert).expect("sealed owner"))
            .collect()
    }

    /// Routed experts every rank holds, read from the sealed plans. Every
    /// group must give each rank the same count: one layer's residency and
    /// stage budget covers all of them.
    pub fn expert_spans(&self) -> Result<crate::expert_residency::EpExpertSpans, String> {
        let count = |plan: &ExpertExecutionPlan, rank: usize| {
            (0..plan.n_experts())
                .filter(|&expert| plan.local_slot(rank, expert).is_some())
                .count()
        };
        let first = &self.plans[0];
        let local: Vec<usize> = (0..self.ranks()).map(|rank| count(first, rank)).collect();
        if let Some(plan) = self
            .plans
            .iter()
            .find(|plan| (0..self.ranks()).any(|rank| count(plan, rank) != local[rank]))
        {
            return Err(format!(
                "qwen4 EP seal: {} gives the ranks other expert counts than {}",
                plan.group(),
                first.group()
            ));
        }
        crate::expert_residency::EpExpertSpans::new(first.n_experts(), local)
    }

    fn plan(&self, group: usize) -> Result<&ExpertExecutionPlan, String> {
        self.plans
            .iter()
            .find(|plan| plan.layer() == Some(group))
            .ok_or_else(|| format!("qwen4 EP: the seal has no expert group {group}"))
    }
}

/// A rank's binding to its load's seal (`Qwen4Bundle::bind_ep_seal`).
#[derive(Clone, Debug)]
pub struct Qwen4EpRankSeal {
    pub seal: Arc<Qwen4EpSeal>,
    pub rank: usize,
}

impl Qwen4EpRankSeal {
    /// This rank's dispatch binding of routed group `group` (a trunk layer,
    /// or the trunk layer count for the MTP head), adapted from the sealed
    /// plan: real owner ranks and compact local slots under the owner-routed
    /// EP contract. Refuses, before any table is published, a placement of
    /// another rank or mesh, a plan of another mesh epoch, a cache sealed for
    /// another device and a rank that owns no expert.
    pub(crate) fn expert_binding(
        &self,
        group: usize,
        placement: Qwen4EpPlacement,
        device: i32,
    ) -> Result<(ExpertTable, ExpertBindingCache), String> {
        let sealed = Qwen4EpPlacement {
            rank: self.rank,
            ranks: self.seal.ranks(),
        };
        if placement != sealed {
            return Err(format!(
                "qwen4 EP binding: placement {placement:?} is not the sealed rank {sealed:?}"
            ));
        }
        let plan = self.seal.plan(group)?;
        let (table, cache) = adapt_expert_execution_plan(plan, self.rank)?;
        let contract = table
            .execution_contract()
            .ok_or("qwen4 EP binding: the adapted table carries no contract")?;
        let epoch = self.seal.mesh().epoch().as_u64();
        if contract.mesh_epoch() != epoch || !contract.is_owner_routed_ep() {
            return Err(format!(
                "qwen4 EP binding: group {group} contract (epoch {}, '{}') is not this seal's \
                 owner-routed EP route (epoch {})",
                contract.mesh_epoch(),
                contract.execution(),
                epoch
            ));
        }
        if cache.physical_device() != device {
            return Err(format!(
                "qwen4 EP binding: rank {} is sealed to device {}, executing on {device}",
                self.rank,
                cache.physical_device()
            ));
        }
        if cache.local_expert_ids().is_empty() {
            return Err(format!(
                "qwen4 EP binding: rank {} owns no expert of group {group}",
                self.rank
            ));
        }
        Ok((table, cache))
    }

    /// This rank's compact slot of every global expert of routed group
    /// `group` (`None`: another rank owns it), read from the sealed plan.
    pub(crate) fn local_slots(&self, group: usize) -> Result<Vec<Option<usize>>, String> {
        let plan = self.seal.plan(group)?;
        Ok((0..plan.n_experts())
            .map(|expert| plan.local_slot(self.rank, expert))
            .collect())
    }
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
            let seal = Qwen4EpSeal::seal(&manifest, 48, ranks, &devices, &source, "g1").unwrap();
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
            // The residency and stage spans are the seal's own compact slots.
            let spans = seal.expert_spans().unwrap();
            let seal = Arc::new(seal);
            for rank in 0..ranks {
                let placement = Qwen4EpPlacement { rank, ranks };
                assert_eq!(spans.local(rank), Some(placement.local_count(512)));
                let bound = Qwen4EpRankSeal {
                    seal: seal.clone(),
                    rank,
                };
                let slots = bound.local_slots(48).unwrap();
                assert!((0..512).all(|e| slots[e] == placement.owns(e).then(|| e / ranks)));
            }
            assert!(Qwen4EpRankSeal {
                seal: seal.clone(),
                rank: 0
            }
            .local_slots(49)
            .is_err());
            assert!(seal.placement(ranks).is_err());
            assert_eq!(seal.origin(ranks), None);
            seal.check_source("g1").unwrap();
            let replaced = seal.check_source("g2").unwrap_err();
            assert!(replaced.contains("not the sealed generation"), "{replaced}");
        }
        // Two seals of the same shape are two mesh epochs.
        let a = Qwen4EpSeal::seal(&manifest, 48, 2, &[0, 1], &source, "g1").unwrap();
        let b = Qwen4EpSeal::seal(&manifest, 48, 2, &[0, 1], &source, "g1").unwrap();
        assert_ne!(a.origin(1), b.origin(1));
    }

    /// Every rank's dispatch binding carries the sealed owners and compact
    /// slots (not a rank-0-of-1 stand-in) and is refused for another rank,
    /// device or group; two seals of one shape bind different epochs.
    #[test]
    fn rank_binding_carries_sealed_owners_and_refuses_stale_bindings() {
        let manifest = Qwen4Manifest::build(&crate::config::compact_test_config()).unwrap();
        let source = source_of(&manifest, 10);
        let ranks = 3;
        let devices = [4, 5, 6];
        let seal =
            Arc::new(Qwen4EpSeal::seal(&manifest, 48, ranks, &devices, &source, "g1").unwrap());
        let owners = seal.owner_ranks();
        for rank in 0..ranks {
            let placement = seal.placement(rank).unwrap();
            assert_eq!(
                owners.iter().filter(|&&owner| owner == rank).count(),
                placement.local_count(owners.len())
            );
            let bound = Qwen4EpRankSeal {
                seal: seal.clone(),
                rank,
            };
            for group in [0, 47, 48] {
                let (table, cache) = bound
                    .expert_binding(group, placement, devices[rank])
                    .unwrap();
                assert_eq!((cache.local_rank(), cache.rank_count()), (rank, ranks));
                for record in table.experts() {
                    let expert = record.global_id();
                    assert_eq!(record.owner_rank(), owners[expert]);
                    assert_eq!(record.owner_rank(), expert % ranks);
                    if record.owner_rank() == rank {
                        assert_eq!(cache.local_slot(expert), Some(expert / ranks));
                    }
                }
                let contract = table.execution_contract().unwrap();
                assert!(contract.is_owner_routed_ep());
                assert_eq!(contract.mesh_epoch(), seal.mesh().epoch().as_u64());
            }
        }
        let bound = Qwen4EpRankSeal {
            seal: seal.clone(),
            rank: 1,
        };
        let placement = seal.placement(1).unwrap();
        let wrong_rank = bound
            .expert_binding(0, seal.placement(2).unwrap(), devices[1])
            .unwrap_err();
        assert!(wrong_rank.contains("not the sealed rank"), "{wrong_rank}");
        let wrong_device = bound.expert_binding(0, placement, devices[0]).unwrap_err();
        assert!(wrong_device.contains("sealed to device"), "{wrong_device}");
        let no_group = bound.expert_binding(49, placement, devices[1]).unwrap_err();
        assert!(no_group.contains("no expert group"), "{no_group}");
        let other = Qwen4EpRankSeal {
            seal: Arc::new(
                Qwen4EpSeal::seal(&manifest, 48, ranks, &devices, &source, "g1").unwrap(),
            ),
            rank: 1,
        };
        let (a, _) = bound.expert_binding(0, placement, devices[1]).unwrap();
        let (b, _) = other.expert_binding(0, placement, devices[1]).unwrap();
        assert_ne!(
            a.execution_contract().unwrap().contract_id(),
            b.execution_contract().unwrap().contract_id()
        );
    }

    /// The generation covers the trunk-overlay sidecar beside the model: one
    /// that appears, is rewritten, or vanishes reads as another generation.
    #[test]
    fn source_generation_binds_the_trunk_overlay_sidecar() {
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
        let dir = std::env::temp_dir().join(format!("hipfire-qwen4-ep-gen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.hfq");
        let tensor = HfqMemTensor {
            name: "t".into(),
            quant_type: 1,
            shape: vec![2],
            group_size: 0,
            data: vec![0; 4],
        };
        write_hfqm_package_mem(&model, crate::ARCH_ID, "{}", &[tensor]).unwrap();
        let hfq = HfqFile::open(&model).unwrap();
        let base = qwen4_source_generation(&hfq, &model).unwrap();
        assert_eq!(base, qwen4_source_generation(&hfq, &model).unwrap());
        let overlay = hipfire_config::memory::trunk_overlay_sidecar_path(&model);
        std::fs::write(&overlay, b"a").unwrap();
        let with_overlay = qwen4_source_generation(&hfq, &model).unwrap();
        assert_ne!(with_overlay, base);
        std::fs::write(&overlay, b"bb").unwrap();
        assert_ne!(qwen4_source_generation(&hfq, &model).unwrap(), with_overlay);
        std::fs::remove_file(&overlay).unwrap();
        assert_eq!(qwen4_source_generation(&hfq, &model).unwrap(), base);
        drop(hfq);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
