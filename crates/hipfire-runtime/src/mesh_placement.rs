// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure admitted-mesh placement values: per-edge transport decisions and (S1)
//! the admitted mesh / band placement that weights, state and transfers are
//! planned from. No `Gpu`, no HIP, no allocation of device resources.

use std::fmt;

use crate::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use std::ops::Range;

/// One immutable layer partition, shared by weight, state and transfer planning.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BandPlacement {
    pub mesh_epoch: MeshEpoch,
    pub n_trunk_layers: usize,
    bands: Vec<Range<usize>>,
}

impl BandPlacement {
    /// Match the historical uniform split: early stages receive the remainder.
    pub fn uniform(mesh: &DeviceMesh, n_trunk_layers: usize) -> Result<Self, String> {
        let stages = mesh.size_of(DimKind::Pp);
        if n_trunk_layers < stages {
            return Err(format!(
                "{n_trunk_layers} trunk layers cannot fill {stages} nonempty PP bands"
            ));
        }
        let base = n_trunk_layers / stages;
        let remainder = n_trunk_layers % stages;
        let mut start = 0;
        let bands = (0..stages)
            .map(|stage| {
                let end = start + base + usize::from(stage < remainder);
                let band = start..end;
                start = end;
                band
            })
            .collect();
        Ok(Self {
            mesh_epoch: mesh.epoch(),
            n_trunk_layers,
            bands,
        })
    }

    /// Counts are in logical layer order, not physical device order.
    pub fn from_counts(
        mesh: &DeviceMesh,
        n_trunk_layers: usize,
        counts: &[usize],
    ) -> Result<Self, String> {
        let stages = mesh.size_of(DimKind::Pp);
        if counts.len() != stages {
            return Err(format!("PP band count {} != stages {stages}", counts.len()));
        }
        let mut start = 0usize;
        let mut bands = Vec::with_capacity(stages);
        for (stage, &count) in counts.iter().enumerate() {
            if count == 0 {
                return Err(format!("PP stage {stage} must contain at least one layer"));
            }
            let end = start
                .checked_add(count)
                .ok_or_else(|| "PP layer count overflow".to_string())?;
            bands.push(start..end);
            start = end;
        }
        if start != n_trunk_layers {
            return Err(format!(
                "PP layer counts sum {start} != trunk layers {n_trunk_layers}"
            ));
        }
        Ok(Self {
            mesh_epoch: mesh.epoch(),
            n_trunk_layers,
            bands,
        })
    }

    /// Validate input-owned operations (for example zero-based PLE layer 1).
    pub fn with_first_band_min(self, min_layers: usize) -> Result<Self, String> {
        if self.bands[0].len() < min_layers {
            return Err(format!(
                "first PP band has {} layers, requires at least {min_layers}",
                self.bands[0].len()
            ));
        }
        Ok(self)
    }

    /// Refuse a placement from another mesh or model before planning resources.
    pub fn validate_for(&self, mesh: &DeviceMesh, n_trunk_layers: usize) -> Result<(), String> {
        if self.mesh_epoch != mesh.epoch() {
            return Err("band placement belongs to a different mesh epoch".to_string());
        }
        if self.n_trunk_layers != n_trunk_layers || self.stages() != mesh.size_of(DimKind::Pp) {
            return Err("band placement does not match trunk layers or PP degree".to_string());
        }
        if self.bands.first().map(|band| band.start) != Some(0)
            || self.bands.last().map(|band| band.end) != Some(n_trunk_layers)
            || self.bands.iter().any(|band| band.is_empty())
            || self
                .bands
                .windows(2)
                .any(|pair| pair[0].end != pair[1].start)
        {
            return Err(
                "band placement must cover the trunk with contiguous nonempty bands".to_string(),
            );
        }
        Ok(())
    }

    /// Stage for a validated absolute trunk layer. Invalid layers are not clamped.
    pub fn stage_of_layer(&self, layer: usize) -> usize {
        assert!(layer < self.n_trunk_layers, "layer outside band placement");
        self.bands.partition_point(|band| band.end <= layer)
    }

    pub fn band(&self, stage: usize) -> Range<usize> {
        self.bands[stage].clone()
    }

    pub fn stages(&self) -> usize {
        self.bands.len()
    }

    /// Ordered `(after_layer, source_stage, destination_stage)` boundaries.
    pub fn boundaries(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.bands
            .iter()
            .enumerate()
            .take(self.stages() - 1)
            .map(|(stage, band)| (band.end - 1, stage, stage + 1))
    }
}

/// Physical identity retained across visibility remapping.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdmittedRank {
    pub logical: usize,
    pub device_ordinal: i32,
    pub pci_bus_id: Option<String>,
    pub arch: String,
}

/// One topology and its already-admitted physical and numerical decisions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdmittedMesh {
    pub mesh: DeviceMesh,
    pub ranks: Vec<AdmittedRank>,
    pub edges: Vec<EdgeDecision>,
    pub family: String,
    pub numerical_policy: String,
}

/// Put the first *visible* device (the primary/output owner) last in PP order.
///
/// Logical stages stay in increasing layer order: stage s < S-1 uses visible
/// device s+1, and stage S-1 uses visible device 0. Complete ordinal/PCI/arch
/// records move together; a remapped ordinal is never treated as a PCI identity.
/// A Single list is unchanged apart from normalizing its logical ID to zero.
pub fn pp_rank_order(visible: &[AdmittedRank]) -> Result<Vec<AdmittedRank>, String> {
    if visible.is_empty() {
        return Err("PP admission requires at least one visible device".to_string());
    }
    let mut ordinals = std::collections::HashSet::new();
    let mut pci_ids = std::collections::HashSet::new();
    for rank in visible {
        if rank.device_ordinal < 0 || !ordinals.insert(rank.device_ordinal) {
            return Err("visible devices contain an invalid or duplicate ordinal".to_string());
        }
        if let Some(pci) = &rank.pci_bus_id {
            if !pci_ids.insert(pci) {
                return Err("visible devices contain a duplicate PCI identity".to_string());
            }
        }
    }
    Ok(visible
        .iter()
        .cycle()
        .skip(1)
        .take(visible.len())
        .enumerate()
        .map(|(logical, rank)| AdmittedRank {
            logical,
            ..rank.clone()
        })
        .collect())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeTransportPolicy {
    Auto,
    Peer,
    Host,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EdgeDecision {
    pub src: usize,
    pub dst: usize,
    pub transport: EdgeTransport,
}

/// Resolve directed edges purely from caller-supplied byte qualification.
/// No HIP query, allocation, probe, or runtime fallback occurs here.
pub fn resolve_edge_transports(
    policy: EdgeTransportPolicy,
    qualified_peer: &dyn Fn(usize, usize) -> bool,
    edges: &[(usize, usize)],
) -> Result<Vec<EdgeDecision>, String> {
    edges
        .iter()
        .map(|&(src, dst)| {
            let transport = match policy {
                EdgeTransportPolicy::Host => EdgeTransport::HostMapped,
                EdgeTransportPolicy::Auto => {
                    if qualified_peer(src, dst) {
                        EdgeTransport::Peer
                    } else {
                        EdgeTransport::HostMapped
                    }
                }
                EdgeTransportPolicy::Peer => {
                    if !qualified_peer(src, dst) {
                        return Err(format!(
                            "peer transport edge {src}->{dst} is not byte-qualified"
                        ));
                    }
                    EdgeTransport::Peer
                }
            };
            Ok(EdgeDecision {
                src,
                dst,
                transport,
            })
        })
        .collect()
}

/// How one directed device edge moves bytes. Chosen once at admission per
/// directed edge and provisioned by `multi_gpu`'s edge transport service;
/// there is no runtime switch between the two.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum EdgeTransport {
    /// The consumer device reads the producer device's VRAM through peer
    /// access. Only for an edge whose peer reads were byte-qualified.
    Peer,
    /// Both devices map one pinned host allocation; the producer publishes
    /// into it and the consumer reads it back. Never enables peer access.
    HostMapped,
}

impl fmt::Display for EdgeTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EdgeTransport::Peer => "peer",
            EdgeTransport::HostMapped => "host-mapped",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pp(stages: usize) -> DeviceMesh {
        DeviceMesh::rect(&[(DimKind::Pp, stages)]).unwrap()
    }

    #[test]
    fn single_identity() {
        let mesh = DeviceMesh::single().unwrap();
        let placement = BandPlacement::uniform(&mesh, 48).unwrap();
        assert_eq!(placement.mesh_epoch, mesh.epoch());
        assert_eq!(placement.band(0), 0..48);
        assert_eq!(placement.stages(), 1);
        assert_eq!(placement.boundaries().count(), 0);
        for layer in 0..48 {
            assert_eq!(placement.stage_of_layer(layer), 0);
        }
        let visible = ranks(1);
        assert_eq!(pp_rank_order(&visible).unwrap(), visible);
    }

    #[test]
    fn uniform_matches_historical_layer_split() {
        for stages in 2..=8 {
            for layers in [24, 48, 49] {
                let mesh = pp(stages);
                let placement = BandPlacement::uniform(&mesh, layers).unwrap();
                let mut start = 0;
                for stage in 0..stages {
                    let count = layers / stages + usize::from(stage < layers % stages);
                    assert_eq!(placement.band(stage), start..start + count);
                    for layer in start..start + count {
                        assert_eq!(placement.stage_of_layer(layer), stage);
                    }
                    start += count;
                }
                assert_eq!(start, layers);
                let boundaries: Vec<_> = placement.boundaries().collect();
                assert_eq!(boundaries.len(), stages - 1);
                for (after, src, dst) in boundaries {
                    assert_eq!(placement.stage_of_layer(after), src);
                    assert_eq!(placement.stage_of_layer(after + 1), dst);
                }
            }
        }
    }

    #[test]
    fn custom_counts_and_input_pin() {
        for counts in [&[9, 8, 8, 8, 8, 7][..], &[13, 12, 12, 11], &[30, 18]] {
            let mesh = pp(counts.len());
            let placement = BandPlacement::from_counts(&mesh, 48, counts)
                .unwrap()
                .with_first_band_min(2)
                .unwrap();
            let mut start = 0;
            for (stage, &count) in counts.iter().enumerate() {
                assert_eq!(placement.band(stage), start..start + count);
                start += count;
            }
        }
    }

    #[test]
    fn invalid_counts_and_stale_epoch_refuse() {
        let mesh = pp(2);
        for (layers, counts) in [
            (48, &[0, 48][..]),
            (48, &[24, 23]),
            (48, &[48]),
            (48, &[usize::MAX, 1]),
        ] {
            assert!(BandPlacement::from_counts(&mesh, layers, counts).is_err());
        }
        assert!(BandPlacement::uniform(&mesh, 1).is_err());
        assert!(BandPlacement::uniform(&DeviceMesh::single().unwrap(), 0).is_err());
        assert!(BandPlacement::from_counts(&mesh, 48, &[1, 47])
            .unwrap()
            .with_first_band_min(2)
            .is_err());
        let placement = BandPlacement::uniform(&mesh, 48).unwrap();
        assert!(placement.validate_for(&pp(2), 48).is_err());
        assert!(placement.validate_for(&mesh, 49).is_err());
        assert!(placement.validate_for(&mesh, 48).is_ok());
    }

    fn ranks(count: usize) -> Vec<AdmittedRank> {
        (0..count)
            .map(|logical| AdmittedRank {
                logical,
                // Deliberately not the original PCI card number.
                device_ordinal: (logical + 10) as i32,
                pci_bus_id: Some(format!("0000:{logical:02x}:00.0")),
                arch: "gfx1100".to_string(),
            })
            .collect()
    }

    #[test]
    fn physical_output_order_preserves_identity() {
        for (count, order) in [(2, vec![1, 0]), (6, vec![1, 2, 3, 4, 5, 0])] {
            let visible = ranks(count);
            let ordered = pp_rank_order(&visible).unwrap();
            for (logical, original) in order.into_iter().enumerate() {
                assert_eq!(ordered[logical].logical, logical);
                assert_eq!(
                    ordered[logical].device_ordinal,
                    visible[original].device_ordinal
                );
                assert_eq!(ordered[logical].pci_bus_id, visible[original].pci_bus_id);
                assert_eq!(ordered[logical].arch, visible[original].arch);
            }
        }
        assert!(pp_rank_order(&[]).is_err());
        let mut duplicate = ranks(2);
        duplicate[1].device_ordinal = duplicate[0].device_ordinal;
        assert!(pp_rank_order(&duplicate).is_err());
        duplicate = ranks(2);
        duplicate[1].pci_bus_id = duplicate[0].pci_bus_id.clone();
        assert!(pp_rank_order(&duplicate).is_err());
    }

    #[test]
    fn directed_edge_policy_truth_table() {
        let edges = [(0, 1), (1, 0)];
        let qualified = |src, dst| src == 0 && dst == 1;
        let auto = resolve_edge_transports(EdgeTransportPolicy::Auto, &qualified, &edges).unwrap();
        assert_eq!(
            auto,
            vec![
                EdgeDecision {
                    src: 0,
                    dst: 1,
                    transport: EdgeTransport::Peer
                },
                EdgeDecision {
                    src: 1,
                    dst: 0,
                    transport: EdgeTransport::HostMapped
                },
            ]
        );
        assert!(resolve_edge_transports(EdgeTransportPolicy::Peer, &qualified, &edges).is_err());
        assert_eq!(
            resolve_edge_transports(EdgeTransportPolicy::Peer, &qualified, &edges[..1]).unwrap()[0]
                .transport,
            EdgeTransport::Peer
        );
        let host = resolve_edge_transports(
            EdgeTransportPolicy::Host,
            &|_, _| panic!("Host must not probe qualification"),
            &edges,
        )
        .unwrap();
        assert!(host
            .iter()
            .all(|edge| edge.transport == EdgeTransport::HostMapped));
    }
}
