// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Whole16 window manifest (railgun DFlash window design A, §5).
//!
//! A manifest names the complete, ordered node sequence of one certified
//! window program: for each node the exact code-object identity (object,
//! device ELF, text/descriptor/metadata digests), the SHA-256 of the receipt
//! that carries its facts, and the toolchain pin the objects were built
//! under. `railgun-cert` generates and verifies manifests offline; the
//! runtime (which links no peacemaker crate) calls [`WindowManifest::admit`]
//! with the nodes of a requested window plan. Any node the manifest does not
//! name exactly, any node that is not a railgun node (`HipDirect`, `Graph`),
//! or any toolchain/arch mismatch refuses the whole plan.
use serde::{Deserialize, Serialize};

use crate::receipt::KNOWN_WITNESSES;
use crate::sha256_hex;

pub const WINDOW_MANIFEST_SCHEMA: &str = "railgun-window-manifest";
pub const WINDOW_MANIFEST_VERSION: u32 = 2;
/// `a1_status` values a generated manifest may carry.
pub const NODE_PROVEN: &str = "proven";
/// `conditional, witness required`: the node's facts hold only in the guarded
/// case `case_sha256` and only with a matching runtime bind witness.
pub const NODE_CONDITIONAL: &str = "conditional";
/// The only shape this manifest family describes.
pub const WHOLE16_SHAPE: &str = "whole16";

/// How a window-plan node would be executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRoute {
    /// A PM4 dispatch authored by railgun (copy/fill included).
    Railgun,
    /// A direct HIP launch outside the program: not admissible in Whole16.
    HipDirect,
    /// A HIP-graph node: not admissible in Whole16.
    Graph,
}

/// One node of the window program with its exact-artifact facts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowNodeRecord {
    pub index: u32,
    /// `draft_embed`, `draft_layer`, `draft_head`, `selector`, `verify_body`,
    /// `gdn`, `commit`, `verify_head`, `transport`.
    pub phase: String,
    /// The required node class this node satisfies (`""` when none).
    pub class: String,
    pub symbol: String,
    /// SHA-256 of the loaded object file: the receipt lookup key.
    pub artifact_sha256: String,
    pub device_elf_sha256: String,
    pub text_sha256: String,
    pub kd_sha256: String,
    pub metadata_sha256: String,
    /// SHA-256 of the canonical JSON of this kernel's receipt fact record
    /// (`receipt.railgun.kernels[symbol]`; tool versions excluded).
    pub receipt_sha256: String,
    /// A1 status of the kernel: `proven`, or `conditional` (witness required).
    pub a1_status: String,
    /// The receipt's guarded-case digest; empty unless `conditional`.
    pub case_sha256: String,
    /// Witness kinds the runtime must present for this node; empty unless
    /// `conditional`.
    pub required_witnesses: Vec<String>,
    /// Loader kernarg segment size from the descriptor/metadata.
    pub kernarg_size: u32,
    /// End of the explicit (non-hidden) arguments.
    pub explicit_end: u32,
    /// Hidden arguments the kernel loads (receipt `implicit.hidden_loaded`).
    pub hidden_loaded: Vec<String>,
}

/// One explicit argument of an ABI-table receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiArgRecord {
    pub offset: u32,
    pub size: u32,
    /// `read`, `write`, `read_write` or `scalar`.
    pub effect: String,
}

/// The FP8 attention ABI table of design §5 as observed in the actual
/// code-object metadata, for one kernel of one object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiReceipt {
    pub symbol: String,
    pub artifact_sha256: String,
    /// Whether runtime admission may use this kernel in Whole16. The GQA8
    /// sibling is receipted with `false`: the 27B shape is GQA6.
    pub admitted: bool,
    pub kernarg_size: u32,
    pub explicit_end: u32,
    pub args: Vec<AbiArgRecord>,
    pub hidden_loaded: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowManifest {
    pub schema: String,
    pub version: u32,
    pub shape: String,
    pub arch: String,
    /// SHA-256 of the toolchain id (`ToolchainPin::pin`).
    pub toolchain_pin: String,
    /// [`sequence_digest`] over `nodes`.
    pub sequence_sha256: String,
    pub nodes: Vec<WindowNodeRecord>,
    pub abi: Vec<AbiReceipt>,
}

/// The digest of an ordered node sequence: every field that identifies the
/// executed code, in order. Changing any node, its position or its object
/// changes it.
pub fn sequence_digest(nodes: &[WindowNodeRecord]) -> String {
    let mut text = String::new();
    for n in nodes {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            n.index,
            n.symbol,
            n.artifact_sha256,
            n.device_elf_sha256,
            n.text_sha256,
            n.kd_sha256,
            n.metadata_sha256,
            n.receipt_sha256,
            n.a1_status,
            n.case_sha256,
            n.required_witnesses.join(",")
        ));
    }
    sha256_hex(text.as_bytes())
}

/// A runtime-held bind witness reference for one conditional node: the
/// manifest node index, the guarded case digest it was bound for, and the
/// witness kind. Produced only by the runtime bind (not constructible from a
/// manifest or receipt alone).
#[derive(Clone, Copy, Debug)]
pub struct ConditionalWitness<'a> {
    pub index: u32,
    pub case_sha256: &'a str,
    pub kind: &'a str,
}

/// A node of a requested window plan, as the runtime sees it.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeNode<'a> {
    pub symbol: &'a str,
    pub artifact_sha256: &'a str,
    pub route: NodeRoute,
}

/// Why a window plan or manifest is refused. `index` is the plan position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowRefusal {
    pub index: Option<u32>,
    pub symbol: String,
    pub reason: String,
}

impl std::fmt::Display for WindowRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.index {
            Some(i) => write!(f, "node {i} {}: {}", self.symbol, self.reason),
            None => write!(f, "{}: {}", self.symbol, self.reason),
        }
    }
}

impl WindowRefusal {
    pub fn node(index: usize, symbol: &str, reason: impl Into<String>) -> Self {
        Self {
            index: Some(index as u32),
            symbol: symbol.to_owned(),
            reason: reason.into(),
        }
    }
    pub fn whole(reason: impl Into<String>) -> Self {
        Self {
            index: None,
            symbol: String::new(),
            reason: reason.into(),
        }
    }
}

impl WindowManifest {
    /// Internal consistency: schema, shape, recorded sequence digest and
    /// per-node facts status. Does not look at any plan.
    pub fn validate(&self) -> Vec<WindowRefusal> {
        let mut out = Vec::new();
        if self.schema != WINDOW_MANIFEST_SCHEMA || self.version != WINDOW_MANIFEST_VERSION {
            out.push(WindowRefusal::whole(format!(
                "manifest schema {} v{} is not {WINDOW_MANIFEST_SCHEMA} v{WINDOW_MANIFEST_VERSION}",
                self.schema, self.version
            )));
        }
        if self.shape != WHOLE16_SHAPE {
            out.push(WindowRefusal::whole(format!(
                "manifest shape {} is not {WHOLE16_SHAPE}",
                self.shape
            )));
        }
        if self.sequence_sha256 != sequence_digest(&self.nodes) {
            out.push(WindowRefusal::whole(
                "recorded sequence digest does not match the nodes",
            ));
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if n.index as usize != i {
                out.push(WindowRefusal::node(i, &n.symbol, "node index out of order"));
            }
            match n.a1_status.as_str() {
                NODE_PROVEN => {
                    if !n.case_sha256.is_empty() || !n.required_witnesses.is_empty() {
                        out.push(WindowRefusal::node(
                            i,
                            &n.symbol,
                            "proven node carries a guarded case",
                        ));
                    }
                }
                NODE_CONDITIONAL => {
                    if n.case_sha256.is_empty() || n.required_witnesses.is_empty() {
                        out.push(WindowRefusal::node(
                            i,
                            &n.symbol,
                            "conditional node lacks a case digest or required witnesses",
                        ));
                    }
                    for (j, kind) in n.required_witnesses.iter().enumerate() {
                        if !KNOWN_WITNESSES.contains(&kind.as_str()) {
                            out.push(WindowRefusal::node(
                                i,
                                &n.symbol,
                                format!("conditional node requires unknown witness kind {kind:?}"),
                            ));
                        }
                        if n.required_witnesses[..j].contains(kind) {
                            out.push(WindowRefusal::node(
                                i,
                                &n.symbol,
                                format!("conditional node lists witness kind {kind:?} twice"),
                            ));
                        }
                    }
                }
                other => out.push(WindowRefusal::node(
                    i,
                    &n.symbol,
                    format!("A1 facts are {other}, not proven or conditional"),
                )),
            }
        }
        out
    }

    /// Admit a requested window plan with no bind witnesses: a manifest with
    /// any `conditional` node is refused ("conditional, witness required").
    pub fn admit(
        &self,
        arch: &str,
        toolchain_pin: &str,
        plan: &[RuntimeNode],
    ) -> Result<(), Vec<WindowRefusal>> {
        self.admit_with(arch, toolchain_pin, plan, &[])
    }

    /// Admit a requested window plan: the same nodes, in the same order, each
    /// a railgun node over the manifest's exact artifact, under the manifest's
    /// arch and toolchain pin. Every `conditional` node additionally needs one
    /// witness per required kind with its index and case digest. Every
    /// violation is reported; none is ignored.
    pub fn admit_with(
        &self,
        arch: &str,
        toolchain_pin: &str,
        plan: &[RuntimeNode],
        witnesses: &[ConditionalWitness],
    ) -> Result<(), Vec<WindowRefusal>> {
        let mut out = self.validate();
        for n in self
            .nodes
            .iter()
            .filter(|n| n.a1_status == NODE_CONDITIONAL)
        {
            for kind in &n.required_witnesses {
                let held = witnesses.iter().any(|w| {
                    w.index == n.index && w.case_sha256 == n.case_sha256 && w.kind == *kind
                });
                if !held {
                    out.push(WindowRefusal::node(
                        n.index as usize,
                        &n.symbol,
                        format!("conditional, witness required: no {kind} witness for case"),
                    ));
                }
            }
        }
        if self.arch != arch {
            out.push(WindowRefusal::whole(format!(
                "manifest arch {} is not {arch}",
                self.arch
            )));
        }
        if self.toolchain_pin != toolchain_pin {
            out.push(WindowRefusal::whole(
                "manifest toolchain pin is not the loaded toolchain",
            ));
        }
        if plan.len() != self.nodes.len() {
            out.push(WindowRefusal::whole(format!(
                "plan has {} nodes, manifest {}",
                plan.len(),
                self.nodes.len()
            )));
        }
        for (i, p) in plan.iter().enumerate() {
            if p.route != NodeRoute::Railgun {
                out.push(WindowRefusal::node(
                    i,
                    p.symbol,
                    format!("route {:?} is not a railgun node", p.route),
                ));
            }
            match self.nodes.get(i) {
                None => out.push(WindowRefusal::node(
                    i,
                    p.symbol,
                    "node beyond the manifest sequence",
                )),
                Some(m) => {
                    if m.symbol != p.symbol {
                        out.push(WindowRefusal::node(
                            i,
                            p.symbol,
                            format!("manifest names {}", m.symbol),
                        ));
                    } else if m.artifact_sha256 != p.artifact_sha256 {
                        out.push(WindowRefusal::node(
                            i,
                            p.symbol,
                            "loaded object digest is not the certified object",
                        ));
                    }
                }
            }
        }
        if out.is_empty() {
            Ok(())
        } else {
            Err(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(i: u32, symbol: &str, sha: &str) -> WindowNodeRecord {
        WindowNodeRecord {
            index: i,
            phase: "verify_body".into(),
            class: String::new(),
            symbol: symbol.into(),
            artifact_sha256: sha.into(),
            device_elf_sha256: "d".into(),
            text_sha256: "t".into(),
            kd_sha256: "k".into(),
            metadata_sha256: "m".into(),
            receipt_sha256: "r".into(),
            a1_status: "proven".into(),
            case_sha256: String::new(),
            required_witnesses: Vec::new(),
            kernarg_size: 16,
            explicit_end: 16,
            hidden_loaded: vec![],
        }
    }

    fn manifest() -> WindowManifest {
        let nodes = vec![node(0, "a", "aa"), node(1, "b", "bb")];
        WindowManifest {
            schema: WINDOW_MANIFEST_SCHEMA.into(),
            version: WINDOW_MANIFEST_VERSION,
            shape: WHOLE16_SHAPE.into(),
            arch: "gfx1201".into(),
            toolchain_pin: "pin".into(),
            sequence_sha256: sequence_digest(&nodes),
            nodes,
            abi: vec![],
        }
    }

    fn plan<'a>(route1: NodeRoute, sha1: &'a str) -> Vec<RuntimeNode<'a>> {
        vec![
            RuntimeNode {
                symbol: "a",
                artifact_sha256: "aa",
                route: NodeRoute::Railgun,
            },
            RuntimeNode {
                symbol: "b",
                artifact_sha256: sha1,
                route: route1,
            },
        ]
    }

    #[test]
    fn exact_plan_is_admitted() {
        manifest()
            .admit("gfx1201", "pin", &plan(NodeRoute::Railgun, "bb"))
            .expect("admits");
    }

    #[test]
    fn wrong_digest_hipdirect_graph_toolchain_and_truncation_refuse() {
        let m = manifest();
        assert!(m
            .admit("gfx1201", "pin", &plan(NodeRoute::Railgun, "cc"))
            .is_err());
        for route in [NodeRoute::HipDirect, NodeRoute::Graph] {
            let e = m
                .admit("gfx1201", "pin", &plan(route, "bb"))
                .expect_err("refuses");
            assert!(e.iter().any(|r| r.reason.contains("not a railgun node")));
        }
        assert!(m
            .admit("gfx1201", "other", &plan(NodeRoute::Railgun, "bb"))
            .is_err());
        assert!(m
            .admit("gfx1100", "pin", &plan(NodeRoute::Railgun, "bb"))
            .is_err());
        let short = &plan(NodeRoute::Railgun, "bb")[..1];
        assert!(m.admit("gfx1201", "pin", short).is_err());
    }

    #[test]
    fn tampered_manifest_or_unproven_node_refuses() {
        let mut m = manifest();
        m.nodes[1].artifact_sha256 = "cc".into();
        assert!(m
            .admit("gfx1201", "pin", &plan(NodeRoute::Railgun, "cc"))
            .is_err());
        let mut m = manifest();
        m.nodes[0].a1_status = "unknown".into();
        m.sequence_sha256 = sequence_digest(&m.nodes);
        assert!(m
            .admit("gfx1201", "pin", &plan(NodeRoute::Railgun, "bb"))
            .is_err());
    }
}
