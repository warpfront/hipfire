// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Which topology a loaded model actually runs on, for the daemon's synthetic
//! `bench_prefill` / `bench_decode` probes.
//!
//! `serve` admits one effective topology per load: a single card, pipeline
//! parallelism (`pp`), dense tensor parallelism (`tp`, the Qwen3.5 dense
//! `EpArch::Qwen35DenseTp` route) or expert parallelism (`ep`: Qwen3.5 MoE,
//! DeepSeek4, MiniMax, and the Qwen4 mesh). The loaded model is the authority:
//! this module reads the degrees and the device identifiers from what is
//! really resident (`Gpus.mesh` axes, `Gpus.devices`, the Qwen4 EP mesh) and
//! classifies it. The classification is a pure function of
//! [`TopologyFacts`], so every route is unit-testable without a GPU.

use crate::{carrier_for, pp_unsupported_message, LoadedModel, PpBenchRoute};
use hipfire_runtime::device_mesh::DimKind;
use hipfire_runtime::multi_gpu::Gpus;

/// `serve`'s refusal when both degrees are requested. The load path and the
/// bench admission share this one sentence.
pub const PP_EP_EXCLUSIVE_MESSAGE: &str =
    "tp (expert-parallel) and pp (pipeline-parallel) are mutually exclusive; set only one.";

/// Which expert-sharded or tensor-sharded server owns a loaded `EpState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpKind {
    Qwen35Moe,
    Qwen35DenseTp,
    Ds4,
    Minimax,
}

/// Degrees and device identifiers of one resident device mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshFacts {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
    /// HIP device ordinal of every rank, in mesh rank order (repeats under
    /// `HIPFIRE_EMULATE_GPUS`).
    pub devices: Vec<i32>,
}

impl MeshFacts {
    /// Read the degrees from the mesh axes and the ordinals from the live
    /// device handles.
    pub fn from_gpus(gpus: &Gpus) -> Self {
        Self {
            pp: gpus.mesh.size_of(DimKind::Pp),
            tp: gpus.mesh.size_of(DimKind::Tp),
            ep: gpus.mesh.size_of(DimKind::Ep),
            devices: gpus.devices.iter().map(|gpu| gpu.device_id).collect(),
        }
    }
}

/// Everything the classification reads from a `LoadedModel`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyFacts {
    pub arch_id: u32,
    /// `LoadedModel::pp`.
    pub pp: usize,
    /// `LoadedModel::pp_gpus` (Qwen3.5 legacy PP and the Qwen4 mesh PP).
    pub pp_mesh: Option<MeshFacts>,
    /// `LoadedModel::ep`: the kind of server and its mesh.
    pub ep: Option<(EpKind, MeshFacts)>,
    /// Qwen4 expert parallelism lives inside the Qwen4 bundle, not in
    /// `LoadedModel::ep`: its rank count and device ordinals.
    pub qwen4_ep: Option<MeshFacts>,
    /// The daemon's own device ordinal: the card of a single-card load.
    pub single_card: i32,
}

/// The forward route a topology executes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchRoute {
    /// One card: the architecture carrier's synthetic hooks.
    Single,
    /// Qwen4 expert parallelism: the bundle fans the carrier's single-device
    /// calls out over its ranks, so the carrier hooks are the entrypoint.
    Qwen4Ep,
    /// Qwen3.5 pipeline parallelism (`forward_prefill_batch_multi`,
    /// `forward_scratch_multi`).
    Qwen35Pp,
    /// Qwen4 mesh pipeline parallelism (`pp_prefill_final`,
    /// `pp_forward_token_or_argmax`).
    Qwen4Pp,
    /// Qwen3.5 MoE expert parallelism (`forward_prefill_batch_ep`,
    /// `forward_ep`).
    Qwen35MoeEp,
    /// Dense Qwen3.5 tensor parallelism (`forward_prefill_dense_tp`,
    /// `forward_scratch_dense_tp`).
    Qwen35DenseTp,
    /// DeepSeek4 expert parallelism.
    Ds4Ep,
    /// MiniMax expert parallelism.
    MinimaxEp,
    /// A pipeline-parallel carrier whose own bench hooks drive the bundle
    /// (it fans the single-device calls out over its bands); the mesh quiesce
    /// syncs every `pp_gpus` device.
    CarrierPp,
}

impl BenchRoute {
    /// True when the architecture carrier's `bench_prefill` /
    /// `bench_decode_prime` / `bench_decode_run` hooks (which drive the
    /// daemon's own `Gpu`) are the entrypoints.
    pub fn uses_carrier_hooks(self) -> bool {
        matches!(
            self,
            BenchRoute::Single | BenchRoute::Qwen4Ep | BenchRoute::CarrierPp
        )
    }

    /// True for every route whose forward spans model devices the daemon's
    /// single `Gpu` handle does not own. Redline capture, replay and G0
    /// observation are bound to that one handle.
    pub fn is_mesh(self) -> bool {
        !matches!(self, BenchRoute::Single)
    }
}

/// The topology report carried on every `prefill_result` / `decode_result`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchTopology {
    pub pp: usize,
    /// Dense tensor-parallel degree.
    pub tp: usize,
    /// Expert-parallel degree.
    pub ep: usize,
    /// HIP device ordinal of each loaded card, rank order, de-duplicated.
    pub cards: Vec<i32>,
}

impl BenchTopology {
    fn new(pp: usize, tp: usize, ep: usize, devices: &[i32]) -> Self {
        let mut cards: Vec<i32> = Vec::with_capacity(devices.len());
        for &device in devices {
            if !cards.contains(&device) {
                cards.push(device);
            }
        }
        Self { pp, tp, ep, cards }
    }

    /// `{"pp":N,"tp":N,"ep":N,"cards":["0","1",...]}`: positive integer
    /// degrees and the card ordinals as decimal strings.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "pp": self.pp,
            "tp": self.tp,
            "ep": self.ep,
            "cards": self.cards.iter().map(|card| card.to_string()).collect::<Vec<_>>(),
        })
    }
}

fn mesh_topology(mesh: &MeshFacts) -> Result<BenchTopology, String> {
    if mesh.devices.is_empty() {
        return Err("loaded topology has no devices".to_string());
    }
    if mesh.pp == 0 || mesh.tp == 0 || mesh.ep == 0 {
        return Err(format!(
            "loaded topology has a zero degree (pp={}, tp={}, ep={})",
            mesh.pp, mesh.tp, mesh.ep
        ));
    }
    Ok(BenchTopology::new(mesh.pp, mesh.tp, mesh.ep, &mesh.devices))
}

/// The bench route a carrier's declared pipeline bench family selects.
pub fn pp_bench_route(bench: PpBenchRoute) -> BenchRoute {
    match bench {
        PpBenchRoute::Qwen35Pp => BenchRoute::Qwen35Pp,
        PpBenchRoute::Qwen4Pp => BenchRoute::Qwen4Pp,
        PpBenchRoute::Carrier => BenchRoute::CarrierPp,
    }
}

/// Classify a loaded model's topology. Refusals reuse `serve`'s own sentences:
/// both degrees at once is [`PP_EP_EXCLUSIVE_MESSAGE`]; an architecture whose
/// carrier declares no pipeline path is the carrier's [`pp_unsupported_message`].
/// Inconsistent facts (a degree without the devices that carry it) fail closed.
pub fn classify(facts: &TopologyFacts) -> Result<(BenchRoute, BenchTopology), String> {
    let ep_loaded = facts.ep.is_some() || facts.qwen4_ep.is_some();
    if facts.ep.is_some() && facts.qwen4_ep.is_some() {
        return Err("loaded model carries two expert-parallel owners".to_string());
    }
    if facts.pp > 1 || facts.pp_mesh.is_some() {
        if ep_loaded {
            return Err(PP_EP_EXCLUSIVE_MESSAGE.to_string());
        }
        let mesh = facts
            .pp_mesh
            .as_ref()
            .ok_or_else(|| "pp>1 LoadedModel missing pp_gpus — carrier bug".to_string())?;
        let topology = mesh_topology(mesh)?;
        if topology.pp != facts.pp || topology.tp != 1 || topology.ep != 1 {
            return Err(format!(
                "pipeline mesh disagrees with the loaded degree (pp={}, mesh pp={} tp={} ep={})",
                facts.pp, topology.pp, topology.tp, topology.ep
            ));
        }
        let route = match carrier_for(facts.arch_id).and_then(|carrier| carrier.topology_caps().pp)
        {
            Some(pp_caps) => pp_bench_route(pp_caps.bench),
            None => {
                let family = carrier_for(facts.arch_id).map_or("model", |carrier| carrier.name());
                return Err(pp_unsupported_message(family, false, facts.pp));
            }
        };
        return Ok((route, topology));
    }
    if let Some((kind, mesh)) = &facts.ep {
        let topology = mesh_topology(mesh)?;
        let (route, ok) = match kind {
            EpKind::Qwen35DenseTp => (
                BenchRoute::Qwen35DenseTp,
                topology.tp == mesh.devices.len() && topology.ep == 1,
            ),
            EpKind::Qwen35Moe => (BenchRoute::Qwen35MoeEp, topology.tp == 1),
            EpKind::Ds4 => (BenchRoute::Ds4Ep, topology.tp == 1),
            EpKind::Minimax => (BenchRoute::MinimaxEp, topology.tp == 1),
        };
        if !ok || topology.pp != 1 {
            return Err(format!(
                "{kind:?} mesh axes disagree with the loaded server (pp={}, tp={}, ep={})",
                topology.pp, topology.tp, topology.ep
            ));
        }
        return Ok((route, topology));
    }
    if let Some(mesh) = &facts.qwen4_ep {
        let topology = mesh_topology(mesh)?;
        if topology.pp != 1 || topology.tp != 1 || topology.ep < 2 {
            return Err(format!(
                "qwen4 expert-parallel mesh is inconsistent (pp={}, tp={}, ep={})",
                topology.pp, topology.tp, topology.ep
            ));
        }
        return Ok((BenchRoute::Qwen4Ep, topology));
    }
    Ok((
        BenchRoute::Single,
        BenchTopology::new(1, 1, 1, &[facts.single_card]),
    ))
}

impl LoadedModel {
    /// The facts the topology classification reads, taken from what is
    /// resident: the PP/EP `Gpus` meshes (axes and device handles) and the
    /// Qwen4 bundle's EP mesh. `single_card` is the daemon `Gpu`'s ordinal.
    pub fn bench_topology_facts(&self, single_card: i32) -> TopologyFacts {
        let ep = self
            .ep
            .as_ref()
            .map(|state| (state.inner.kind(), MeshFacts::from_gpus(&state.gpus)));
        let qwen4_ep = self
            .qwen4()
            .and_then(|bundle| bundle.ep_mesh())
            .map(|mesh| MeshFacts {
                pp: 1,
                tp: 1,
                ep: mesh.ranks(),
                devices: mesh.device_ids().to_vec(),
            });
        TopologyFacts {
            arch_id: self.arch_id,
            pp: self.pp,
            pp_mesh: self.pp_gpus.as_ref().map(MeshFacts::from_gpus),
            ep,
            qwen4_ep,
            single_card,
        }
    }

    /// Route and report of this loaded model's topology
    /// ([`classify`] over [`Self::bench_topology_facts`]).
    pub fn bench_topology(&self, single_card: i32) -> Result<(BenchRoute, BenchTopology), String> {
        classify(&self.bench_topology_facts(single_card))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(pp: usize, tp: usize, ep: usize, devices: &[i32]) -> MeshFacts {
        MeshFacts {
            pp,
            tp,
            ep,
            devices: devices.to_vec(),
        }
    }

    fn facts(arch_id: u32) -> TopologyFacts {
        TopologyFacts {
            arch_id,
            pp: 1,
            pp_mesh: None,
            ep: None,
            qwen4_ep: None,
            single_card: 0,
        }
    }

    fn json_of(topology: &BenchTopology) -> serde_json::Value {
        topology.to_json()
    }

    #[test]
    fn single_card_is_one_of_everything_on_the_daemon_device() {
        let mut f = facts(5);
        f.single_card = 3;
        let (route, topology) = classify(&f).unwrap();
        assert_eq!(route, BenchRoute::Single);
        assert!(route.uses_carrier_hooks() && !route.is_mesh());
        assert_eq!(
            json_of(&topology),
            serde_json::json!({"cards":["3"],"ep":1,"pp":1,"tp":1})
        );
    }

    #[test]
    fn qwen35_and_qwen4_pipeline_report_stages_and_cards() {
        for (arch, route) in [
            (5, BenchRoute::Qwen35Pp),
            (6, BenchRoute::Qwen35Pp),
            (16, BenchRoute::Qwen4Pp),
        ] {
            let mut f = facts(arch);
            f.pp = 3;
            f.pp_mesh = Some(mesh(3, 1, 1, &[0, 1, 2]));
            let (got, topology) = classify(&f).unwrap();
            assert_eq!(got, route, "arch {arch}");
            assert!(got.is_mesh() && !got.uses_carrier_hooks());
            assert_eq!(
                json_of(&topology),
                serde_json::json!({"cards":["0","1","2"],"ep":1,"pp":3,"tp":1})
            );
        }
    }

    #[test]
    fn pipeline_without_devices_or_with_a_foreign_degree_fails_closed() {
        let mut f = facts(5);
        f.pp = 2;
        assert!(classify(&f).unwrap_err().contains("missing pp_gpus"));
        f.pp_mesh = Some(mesh(3, 1, 1, &[0, 1, 2]));
        assert!(classify(&f).unwrap_err().contains("disagrees"));
        f.pp_mesh = Some(mesh(2, 1, 1, &[]));
        assert!(classify(&f).unwrap_err().contains("no devices"));
    }

    #[test]
    fn pipeline_for_an_arch_without_a_pp_path_is_the_carrier_refusal() {
        let mut f = facts(7);
        f.pp = 2;
        f.pp_mesh = Some(mesh(2, 1, 1, &[0, 1]));
        let error = classify(&f).unwrap_err();
        assert!(
            error.contains("cannot be split across GPUs in pipeline mode")
                && error.contains("topology.pp = 2"),
            "{error}"
        );
    }

    #[test]
    fn expert_and_pipeline_together_is_serves_exclusive_refusal() {
        let mut f = facts(5);
        f.pp = 2;
        f.pp_mesh = Some(mesh(2, 1, 1, &[0, 1]));
        f.ep = Some((EpKind::Qwen35Moe, mesh(1, 1, 2, &[0, 1])));
        assert_eq!(classify(&f).unwrap_err(), PP_EP_EXCLUSIVE_MESSAGE);
        let mut f = facts(16);
        f.pp = 2;
        f.pp_mesh = Some(mesh(2, 1, 1, &[0, 1]));
        f.qwen4_ep = Some(mesh(1, 1, 2, &[0, 1]));
        assert_eq!(classify(&f).unwrap_err(), PP_EP_EXCLUSIVE_MESSAGE);
    }

    #[test]
    fn dense_tp_reports_tp_and_never_ep() {
        let mut f = facts(5);
        f.ep = Some((EpKind::Qwen35DenseTp, mesh(1, 2, 1, &[0, 1])));
        let (route, topology) = classify(&f).unwrap();
        assert_eq!(route, BenchRoute::Qwen35DenseTp);
        assert_eq!(
            json_of(&topology),
            serde_json::json!({"cards":["0","1"],"ep":1,"pp":1,"tp":2})
        );
        // A dense-TP server on an expert axis is a mislabeled mesh.
        f.ep = Some((EpKind::Qwen35DenseTp, mesh(1, 1, 2, &[0, 1])));
        assert!(classify(&f).unwrap_err().contains("disagree"));
    }

    #[test]
    fn expert_parallel_servers_report_ep_and_never_tp() {
        for (arch, kind, route) in [
            (5, EpKind::Qwen35Moe, BenchRoute::Qwen35MoeEp),
            (9, EpKind::Ds4, BenchRoute::Ds4Ep),
            (10, EpKind::Minimax, BenchRoute::MinimaxEp),
        ] {
            let mut f = facts(arch);
            f.ep = Some((kind, mesh(1, 1, 3, &[0, 1, 2])));
            let (got, topology) = classify(&f).unwrap();
            assert_eq!(got, route);
            assert_eq!(
                json_of(&topology),
                serde_json::json!({"cards":["0","1","2"],"ep":3,"pp":1,"tp":1})
            );
            f.ep = Some((kind, mesh(1, 3, 1, &[0, 1, 2])));
            assert!(classify(&f).unwrap_err().contains("disagree"), "{kind:?}");
        }
    }

    #[test]
    fn qwen4_expert_parallel_runs_the_carrier_hooks() {
        let mut f = facts(16);
        f.qwen4_ep = Some(mesh(1, 1, 3, &[0, 1, 2]));
        let (route, topology) = classify(&f).unwrap();
        assert_eq!(route, BenchRoute::Qwen4Ep);
        assert!(route.uses_carrier_hooks() && route.is_mesh());
        assert_eq!((topology.pp, topology.tp, topology.ep), (1, 1, 3));
        f.qwen4_ep = Some(mesh(1, 1, 1, &[0]));
        assert!(classify(&f).unwrap_err().contains("inconsistent"));
    }

    #[test]
    fn emulated_ranks_on_one_device_list_the_card_once() {
        let mut f = facts(16);
        f.qwen4_ep = Some(mesh(1, 1, 4, &[0, 1, 0, 1]));
        let (_, topology) = classify(&f).unwrap();
        assert_eq!(topology.ep, 4);
        assert_eq!(topology.cards, vec![0, 1]);
    }

    #[test]
    fn two_expert_owners_fail_closed() {
        let mut f = facts(5);
        f.ep = Some((EpKind::Qwen35Moe, mesh(1, 1, 2, &[0, 1])));
        f.qwen4_ep = Some(mesh(1, 1, 2, &[0, 1]));
        assert!(classify(&f).unwrap_err().contains("two expert-parallel"));
    }

    #[test]
    fn declared_pp_bench_families_map_to_their_routes() {
        assert_eq!(pp_bench_route(PpBenchRoute::Qwen35Pp), BenchRoute::Qwen35Pp);
        assert_eq!(pp_bench_route(PpBenchRoute::Qwen4Pp), BenchRoute::Qwen4Pp);
        assert_eq!(pp_bench_route(PpBenchRoute::Carrier), BenchRoute::CarrierPp);
        // The carrier route runs the carrier's hooks across devices the daemon
        // `Gpu` does not own: both a carrier-hook route and a mesh route.
        assert!(BenchRoute::CarrierPp.uses_carrier_hooks() && BenchRoute::CarrierPp.is_mesh());
    }

    #[test]
    fn every_declared_pp_carrier_classifies_to_its_declared_bench_route() {
        for carrier in crate::registry_carriers() {
            let Some(pp_caps) = carrier.topology_caps().pp else {
                continue;
            };
            let arch = (0u32..64)
                .find(|id| carrier_for(*id).is_some_and(|c| c.name() == carrier.name()))
                .unwrap_or_else(|| panic!("{} claims no arch id below 64", carrier.name()));
            let mut f = facts(arch);
            f.pp = 2;
            f.pp_mesh = Some(mesh(2, 1, 1, &[0, 1]));
            let (route, _) = classify(&f).unwrap();
            assert_eq!(route, pp_bench_route(pp_caps.bench), "{}", carrier.name());
        }
    }
}
