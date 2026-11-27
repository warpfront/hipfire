// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2-Mini (arch 24) `mq4-xts` artifact layout `qt44-padded-awq-v1`.
//!
//! One layout shared by the RTN control **B** (`--arch-id 24 --format mq4v2
//! --tier xt --mq4v2-symmetric`) and the H2 artifact **C**
//! (`--input B.hfq --mq4v2-final-codes RECORDS --output C.hfq`):
//!
//! | tensor | qt | shape | AWQ sidecar |
//! |---|---|---|---|
//! | `layers.L.attn.{wq_du,wk_dv,wv_dv,wr_du,gate_attn,wo_ud}.weight` | 44 sym | A's `[M,K]` | `[K]` |
//! | `layers.L.mlp.experts.e.gate_up_proj.weight` | 44 sym | `[2*inter, hidden]` | `[hidden]` |
//! | `layers.L.mlp.experts.e.down_proj.weight` | 44 sym | `[hidden, 1280]` (logical K 1152, zero tail **before** FWHT) | `[1280]`, tail 1.0 |
//! | `layers.L.mlp.gate.weight` (router) | 1 (F16 from source) | `[experts, hidden]` | none |
//! | `embed.weight` | 44 affine (the incumbent encoder, not symmetric) | `[vocab, hidden]` | none |
//! | `unembed.weight` | 44 symmetric RTN | `[vocab, hidden]` | none |
//!
//! The sidecars are all-ones F16 in B (a real S=1 RTN control) and H2's `S` in
//! C. Every other tensor is whatever the legacy arch-24 path emits.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use hipfire_quantize::float16::{f16_to_f32, f32_to_f16};
use hipfire_runtime::hfq::{HfqFile as RuntimeHfqFile, HfqTensorInfo};
use rayon::prelude::*;

use crate::hfq::{HfqTensor, QuantType};
use crate::quant_fwht::{
    gen_fwht_signs, quantize_mq4g256v2, quantize_mq4g256v2_symmetric, MQ4V2_GROUP_BYTES,
};

pub(crate) const ARCH_ID: u32 = 24;
pub(crate) const LAYOUT_NAME: &str = "qt44-padded-awq-v1";
pub(crate) const LAYOUT_KEY: &str = "hipfire_inkling2_mq4_layout";
pub(crate) const MANIFEST_KEY: &str = "hipfire_inkling2_quant_manifest";
pub(crate) const CALIBRATION_RTN: &str = "rtn-s1";
pub(crate) const CALIBRATION_H2: &str = "h2-v1";
/// Total tensor count of the real Inkling-2-Mini B/C artifact.
pub(crate) const MINI_TOTAL_TENSORS: usize = 13_078;
const ONE_F16: u16 = 0x3C00;
const ROWS_PER_UNIT: usize = 32;

/// Geometry of one XTS artifact. `Layout::mini()` is the real model; tests use
/// a miniature with the same structure.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) hidden: usize,
    /// Routed expert intermediate width = logical K of `down_proj`.
    pub(crate) inter: usize,
    /// Physical (padded) K of `down_proj`.
    pub(crate) down_k_phys: usize,
    pub(crate) vocab: usize,
    /// `(leaf, M, K)` of the six calibrated attention projections.
    pub(crate) attn: [(&'static str, usize, usize); 6],
}

impl Layout {
    pub(crate) fn mini() -> Self {
        Self {
            layers: 24,
            experts: 128,
            hidden: 2304,
            inter: 1152,
            down_k_phys: 1280,
            vocab: 201_024,
            attn: [
                ("wq_du", 3072, 2304),
                ("wk_dv", 1024, 2304),
                ("wv_dv", 1024, 2304),
                ("wr_du", 384, 2304),
                ("gate_attn", 3072, 2304),
                ("wo_ud", 2304, 3072),
            ],
        }
    }

    pub(crate) fn sidecars(&self) -> usize {
        self.layers * (self.attn.len() + 2 * self.experts)
    }

    fn attn_name(&self, layer: usize, leaf: &str) -> String {
        format!("model.llm.layers.{layer}.attn.{leaf}.weight")
    }
    fn gate_up_name(&self, layer: usize, e: usize) -> String {
        format!("model.llm.layers.{layer}.mlp.experts.{e}.gate_up_proj.weight")
    }
    fn down_name(&self, layer: usize, e: usize) -> String {
        format!("model.llm.layers.{layer}.mlp.experts.{e}.down_proj.weight")
    }
    fn router_name(&self, layer: usize) -> String {
        format!("model.llm.layers.{layer}.mlp.gate.weight")
    }

    /// The calibrated (qt44 + sidecar) tensors: `(name, M, K_physical)`, in a
    /// fixed order.
    pub(crate) fn calibrated(&self) -> Vec<(String, usize, usize)> {
        let mut out = Vec::with_capacity(self.sidecars());
        for l in 0..self.layers {
            for &(leaf, m, k) in &self.attn {
                out.push((self.attn_name(l, leaf), m, k));
            }
            for e in 0..self.experts {
                out.push((self.gate_up_name(l, e), 2 * self.inter, self.hidden));
                out.push((self.down_name(l, e), self.hidden, self.down_k_phys));
            }
        }
        out
    }

    fn is_down(&self, name: &str) -> bool {
        name.ends_with(".down_proj.weight")
    }
}

/// Router matrices: `model.llm.layers.N.mlp.gate.weight` (never the bias).
pub(crate) fn is_router(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.llm.layers.") else {
        return false;
    };
    let Some((layer, tail)) = rest.split_once('.') else {
        return false;
    };
    !layer.is_empty() && layer.bytes().all(|b| b.is_ascii_digit()) && tail == "mlp.gate.weight"
}

pub(crate) fn is_embed(name: &str) -> bool {
    name == "model.llm.embed.weight"
}

pub(crate) fn is_unembed(name: &str) -> bool {
    name == "model.llm.unembed.weight"
}

/// `Some(K)` when `name` is one of the six calibrated attention projections.
pub(crate) fn attn_site_k(layout: &Layout, name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.llm.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let leaf = tail.strip_prefix("attn.")?.strip_suffix(".weight")?;
    layout
        .attn
        .iter()
        .find(|(l, _, _)| *l == leaf)
        .map(|&(_, _, k)| k)
}

/// All-ones F16 sidecar payload of length `n`.
pub(crate) fn unit_sidecar_f16(n: usize) -> Vec<u8> {
    let one = ONE_F16.to_le_bytes();
    let mut v = Vec::with_capacity(2 * n);
    for _ in 0..n {
        v.extend_from_slice(&one);
    }
    v
}

fn sidecar_tensor(weight_name: &str, k: usize) -> HfqTensor {
    HfqTensor {
        name: crate::pipeline::awq_sidecar_name(weight_name),
        quant_type: QuantType::F16,
        shape: vec![k as u32],
        group_size: 0,
        data: unit_sidecar_f16(k),
        spilled_len: 0,
    }
}

/// Quantize `rows x k` (zero-padded on the right to `phys`) onto qt44. The
/// padding is applied to the F32 row **before** the FWHT, so the last 256-group
/// of a padded row mixes real and zero coordinates. `symmetric = false` is the
/// incumbent affine encoder.
pub(crate) fn quantize_padded_rows(
    data: &[f32],
    rows: usize,
    k: usize,
    phys: usize,
    symmetric: bool,
) -> Vec<u8> {
    assert!(phys >= k && phys % 256 == 0, "bad padded K {k} -> {phys}");
    assert_eq!(data.len(), rows * k);
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let encode = |unit: &[f32], unit_rows: usize| -> Vec<u8> {
        if symmetric {
            quantize_mq4g256v2_symmetric(unit, unit_rows, phys, &s1, &s2)
        } else {
            quantize_mq4g256v2(unit, unit_rows, phys, &s1, &s2)
        }
    };
    let units = rows.div_ceil(ROWS_PER_UNIT);
    let parts: Vec<Vec<u8>> = (0..units)
        .into_par_iter()
        .map(|u| {
            let r0 = u * ROWS_PER_UNIT;
            let r1 = (r0 + ROWS_PER_UNIT).min(rows);
            if phys == k {
                encode(&data[r0 * k..r1 * k], r1 - r0)
            } else {
                let mut padded = vec![0.0f32; (r1 - r0) * phys];
                for (i, r) in (r0..r1).enumerate() {
                    padded[i * phys..i * phys + k].copy_from_slice(&data[r * k..(r + 1) * k]);
                }
                encode(&padded, r1 - r0)
            }
        })
        .collect();
    parts.concat()
}

fn qt44_tensor(name: &str, rows: usize, phys: usize, data: Vec<u8>) -> HfqTensor {
    HfqTensor {
        name: name.to_string(),
        quant_type: QuantType::MQ4G256V2,
        shape: vec![rows as u32, phys as u32],
        group_size: 256,
        data,
        spilled_len: 0,
    }
}

/// Calibrated attention projection: symmetric qt44 + all-ones sidecar.
pub(crate) fn encode_attn(
    name: &str,
    data: &[f32],
    rows: usize,
    k: usize,
) -> Result<Vec<HfqTensor>, String> {
    let q = quantize_padded_rows(data, rows, k, k, true);
    crate::raon::verify_symmetric_headers(name, &q)?;
    Ok(vec![qt44_tensor(name, rows, k, q), sidecar_tensor(name, k)])
}

/// Embedding: the incumbent affine qt44 encoder (bytes equal A's), no sidecar.
/// The global symmetric request must not reach this tensor.
pub(crate) fn encode_embed(name: &str, data: &[f32], rows: usize, k: usize) -> Vec<HfqTensor> {
    vec![qt44_tensor(
        name,
        rows,
        k,
        quantize_padded_rows(data, rows, k, k, false),
    )]
}

/// Unembedding: symmetric qt44 RTN, no sidecar.
pub(crate) fn encode_unembed(
    name: &str,
    data: &[f32],
    rows: usize,
    k: usize,
) -> Result<Vec<HfqTensor>, String> {
    let q = quantize_padded_rows(data, rows, k, k, true);
    crate::raon::verify_symmetric_headers(name, &q)?;
    Ok(vec![qt44_tensor(name, rows, k, q)])
}

/// All routed experts of one stacked tensor. For `w13`, `f32_all` is the
/// converter's `[gate0, up0, gate1, up1, ...]` row order per expert and is
/// regrouped here to `[gate..., up...]`; `w2` is zero-padded to the physical K.
///
/// Emits, per expert, the symmetric qt44 weight followed by its sidecar.
pub(crate) fn encode_experts(
    layout: &Layout,
    base: &str,
    f32_all: &[f32],
    experts: usize,
    rows: usize,
    cols: usize,
    is_w13: bool,
) -> Result<Vec<HfqTensor>, String> {
    let per_expert = rows * cols;
    if f32_all.len() != experts * per_expert {
        return Err(format!(
            "{base}: dequantized length {} != {experts}*{rows}*{cols}",
            f32_all.len()
        ));
    }
    if is_w13 && (rows != 2 * layout.inter || cols != layout.hidden) {
        return Err(format!(
            "{base}: w13 is [{rows},{cols}], layout expects [{},{}]",
            2 * layout.inter,
            layout.hidden
        ));
    }
    if !is_w13 && (rows != layout.hidden || cols != layout.inter) {
        return Err(format!(
            "{base}: w2 is [{rows},{cols}], layout expects [{},{}]",
            layout.hidden, layout.inter
        ));
    }
    let leaf = if is_w13 { "gate_up_proj" } else { "down_proj" };
    let phys = if is_w13 { cols } else { layout.down_k_phys };
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let per: Vec<[HfqTensor; 2]> = (0..experts)
        .into_par_iter()
        .map(|e| {
            let src = &f32_all[e * per_expert..(e + 1) * per_expert];
            let name = format!("{base}.experts.{e}.{leaf}.weight");
            let q = if is_w13 {
                // Converter stores w13 rows as [gate0, up0, gate1, up1, ...];
                // the loader expects [gate..., up...] of shape [2*inter, hidden].
                let inter = rows / 2;
                let mut grouped = vec![0.0f32; per_expert];
                for j in 0..inter {
                    grouped[j * cols..(j + 1) * cols]
                        .copy_from_slice(&src[(2 * j) * cols..(2 * j + 1) * cols]);
                    grouped[(inter + j) * cols..(inter + j + 1) * cols]
                        .copy_from_slice(&src[(2 * j + 1) * cols..(2 * j + 2) * cols]);
                }
                quantize_mq4g256v2_symmetric(&grouped, rows, cols, &s1, &s2)
            } else {
                let mut padded = vec![0.0f32; rows * phys];
                for r in 0..rows {
                    padded[r * phys..r * phys + cols]
                        .copy_from_slice(&src[r * cols..(r + 1) * cols]);
                }
                quantize_mq4g256v2_symmetric(&padded, rows, phys, &s1, &s2)
            };
            [
                qt44_tensor(&name, rows, phys, q),
                sidecar_tensor(&name, phys),
            ]
        })
        .collect();
    for pair in &per {
        crate::raon::verify_symmetric_headers(&pair[0].name, &pair[0].data)?;
    }
    Ok(per.into_iter().flatten().collect())
}

// ── metadata ────────────────────────────────────────────────────────────────

pub(crate) fn manifest_value(layout: &Layout, calibration: &str) -> serde_json::Value {
    serde_json::json!({
        "down_k_logical": layout.inter,
        "down_k_physical": layout.down_k_phys,
        "awq_sidecars": layout.sidecars(),
        "calibration": calibration,
    })
}

/// Stamp the typed layout hint and quant manifest for a freshly written B.
pub(crate) fn stamp_metadata(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    layout: &Layout,
) {
    obj.insert(LAYOUT_KEY.to_string(), LAYOUT_NAME.into());
    obj.insert(
        MANIFEST_KEY.to_string(),
        manifest_value(layout, CALIBRATION_RTN),
    );
}

// ── template validation ─────────────────────────────────────────────────────

/// Name/qt/shape/group view of one tensor-index entry.
pub(crate) struct Entry<'a> {
    pub(crate) name: &'a str,
    pub(crate) qt: u8,
    pub(crate) shape: &'a [u32],
    pub(crate) group: u32,
}

impl<'a> From<&'a HfqTensorInfo> for Entry<'a> {
    fn from(t: &'a HfqTensorInfo) -> Self {
        Entry {
            name: &t.name,
            qt: t.quant_type,
            shape: &t.shape,
            group: t.group_size,
        }
    }
}

impl<'a> From<&'a HfqTensor> for Entry<'a> {
    fn from(t: &'a HfqTensor) -> Self {
        Entry {
            name: &t.name,
            qt: t.quant_type as u8,
            shape: &t.shape,
            group: t.group_size,
        }
    }
}

/// Certify the qt44-padded-awq-v1 census: exactly the calibrated qt44+sidecar
/// pairs, qt44 embed/unembed without sidecar, F16 routers, and nothing else
/// quantized (no qt13 / stray qt44 / orphan sidecar).
pub(crate) fn check_template(layout: &Layout, entries: &[Entry<'_>]) -> Result<(), String> {
    let mut by_name: HashMap<&str, &Entry<'_>> = HashMap::with_capacity(entries.len());
    for e in entries {
        if by_name.insert(e.name, e).is_some() {
            return Err(format!("XTS type map has duplicate tensor `{}`", e.name));
        }
    }
    let mismatch = |what: &str, e: &Entry<'_>, want: &str| -> String {
        format!(
            "XTS type map: {what} `{}` is qt={} shape={:?} group_size={}, expected {want}",
            e.name, e.qt, e.shape, e.group
        )
    };
    let mut qt44_expected = BTreeSet::new();
    let mut sidecar_expected = BTreeSet::new();
    for (name, m, k) in layout.calibrated() {
        let e = by_name
            .get(name.as_str())
            .ok_or_else(|| format!("XTS type map is missing calibrated tensor `{name}`"))?;
        if e.qt != 44 || e.group != 256 || e.shape != [m as u32, k as u32] {
            return Err(mismatch(
                "tensor",
                e,
                &format!("qt=44 [{m},{k}] group_size=256"),
            ));
        }
        let sc = crate::pipeline::awq_sidecar_name(&name);
        let s = by_name
            .get(sc.as_str())
            .ok_or_else(|| format!("XTS type map is missing AWQ sidecar `{sc}`"))?;
        if s.qt != QuantType::F16 as u8 || s.group != 0 || s.shape != [k as u32] {
            return Err(mismatch("sidecar", s, &format!("qt=1 [{k}] group_size=0")));
        }
        qt44_expected.insert(name);
        sidecar_expected.insert(sc);
    }
    let hidden = layout.hidden as u32;
    for (name, label) in [
        ("model.llm.embed.weight", "embed"),
        ("model.llm.unembed.weight", "unembed"),
    ] {
        let e = by_name
            .get(name)
            .ok_or_else(|| format!("XTS type map is missing `{name}`"))?;
        if e.qt != 44 || e.group != 256 || e.shape != [layout.vocab as u32, hidden] {
            return Err(mismatch(
                label,
                e,
                &format!("qt=44 [{},{}] group_size=256", layout.vocab, layout.hidden),
            ));
        }
        let sc = crate::pipeline::awq_sidecar_name(name);
        if by_name.contains_key(sc.as_str()) {
            return Err(format!("XTS type map: `{name}` must not carry a sidecar"));
        }
        qt44_expected.insert(name.to_string());
    }
    for l in 0..layout.layers {
        let name = layout.router_name(l);
        let e = by_name
            .get(name.as_str())
            .ok_or_else(|| format!("XTS type map is missing router `{name}`"))?;
        if e.qt != QuantType::F16 as u8 || e.shape != [layout.experts as u32, hidden] {
            return Err(mismatch(
                "router",
                e,
                &format!("qt=1 [{},{}]", layout.experts, layout.hidden),
            ));
        }
    }
    for e in entries {
        match e.qt {
            44 => {
                if !qt44_expected.contains(e.name) {
                    return Err(format!(
                        "XTS type map has unexpected qt44 tensor `{}`",
                        e.name
                    ));
                }
            }
            1 => {}
            other => {
                return Err(format!(
                    "XTS type map: `{}` has qt={other}; only qt44 and F16 are allowed",
                    e.name
                ));
            }
        }
        if e.name.ends_with(".awq_scale.weight") && !sidecar_expected.contains(e.name) {
            return Err(format!("XTS type map has orphan AWQ sidecar `{}`", e.name));
        }
    }
    Ok(())
}

fn require_str(meta: &serde_json::Value, ptr: &str, want: &str) -> Result<(), String> {
    match meta.pointer(ptr).and_then(|v| v.as_str()) {
        Some(got) if got == want => Ok(()),
        Some(got) => Err(format!(
            "input HFQ metadata `{ptr}` is `{got}`, expected `{want}`"
        )),
        None => Err(format!(
            "input HFQ metadata lacks `{ptr}` (expected `{want}`)"
        )),
    }
}

/// Arch-24 arm of the final-code template validation: the typed layout hint,
/// symmetric marker, RTN-control manifest and the full census.
pub(crate) fn validate_template(
    layout: &Layout,
    metadata_json: &str,
    tensors: &[HfqTensorInfo],
) -> Result<(), String> {
    let meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("input HFQ metadata is invalid JSON: {e}"))?;
    require_str(&meta, "/hipfire_product_tier", "xt").map_err(|_| {
        "final-code import requires an HFQ with hipfire_product_tier=`xt`".to_string()
    })?;
    require_str(&meta, "/hipfire_base_format", "mq4v2")
        .map_err(|_| "final-code import requires an MQ4V2 XT source artifact".to_string())?;
    require_str(&meta, &format!("/{LAYOUT_KEY}"), LAYOUT_NAME)?;
    if meta.get("mq4v2.symmetric") != Some(&serde_json::json!(1)) {
        return Err("input HFQ metadata lacks `mq4v2.symmetric` = 1".to_string());
    }
    let manifest = meta
        .get(MANIFEST_KEY)
        .ok_or_else(|| format!("input HFQ metadata lacks `{MANIFEST_KEY}`"))?;
    if manifest.get("down_k_logical") != Some(&serde_json::json!(layout.inter))
        || manifest.get("down_k_physical") != Some(&serde_json::json!(layout.down_k_phys))
        || manifest.get("awq_sidecars") != Some(&serde_json::json!(layout.sidecars()))
    {
        return Err(format!(
            "input HFQ `{MANIFEST_KEY}` geometry {manifest} does not match the layout \
             (down_k {}->{}, {} sidecars)",
            layout.inter,
            layout.down_k_phys,
            layout.sidecars()
        ));
    }
    if manifest.get("calibration").and_then(|v| v.as_str()) != Some(CALIBRATION_RTN) {
        return Err(format!(
            "final-code import needs the `{CALIBRATION_RTN}` control B as input; \
             `{MANIFEST_KEY}.calibration` is {}",
            manifest
                .get("calibration")
                .unwrap_or(&serde_json::Value::Null)
        ));
    }
    let entries: Vec<Entry<'_>> = tensors.iter().map(Entry::from).collect();
    check_template(layout, &entries)
}

// ── record validation ───────────────────────────────────────────────────────

/// The shared-Q producer group (`wq_du`, `wr_du`, `gate_attn` read one
/// normalized operand and therefore carry one scale vector). K and V are
/// separate.
pub(crate) fn shared_scale_group(name: &str) -> Option<String> {
    for leaf in ["wq_du", "wr_du", "gate_attn"] {
        if let Some(prefix) = name.strip_suffix(&format!("{leaf}.weight")) {
            return Some(format!("{prefix}q_producer"));
        }
    }
    None
}

/// Strict one-to-one mapping of the record names onto the calibrated tensors:
/// refuses missing, extra and (already refused by the loader) duplicate records.
pub(crate) fn check_record_set<'a>(
    layout: &Layout,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), String> {
    let expected: BTreeSet<String> = layout.calibrated().into_iter().map(|(n, _, _)| n).collect();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut extra: Vec<&str> = Vec::new();
    for n in names {
        if !seen.insert(n) {
            return Err(format!("duplicate final-code record for `{n}`"));
        }
        if !expected.contains(n) {
            extra.push(n);
        }
    }
    if !extra.is_empty() {
        return Err(format!(
            "{} final-code record(s) do not name a calibrated XTS tensor, e.g. `{}`",
            extra.len(),
            extra[0]
        ));
    }
    let missing: Vec<&String> = expected
        .iter()
        .filter(|n| !seen.contains(n.as_str()))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{} of {} calibrated XTS tensors have no final-code record, e.g. `{}`",
            missing.len(),
            expected.len(),
            missing[0]
        ));
    }
    Ok(())
}

/// Per-record arch-24 contract on top of the shared record validation:
/// symmetric `(d, z)` grid and the down S tail.
pub(crate) fn check_record(
    layout: &Layout,
    name: &str,
    m: usize,
    k: usize,
    s_f16: &[u8],
    d_z_f16: &[u8],
) -> Result<(), String> {
    let bits = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let groups = m * (k / 256);
    if d_z_f16.len() != groups * 4 * 2 {
        return Err(format!(
            "{name}: d_z_f16 has {} bytes for {groups} groups",
            d_z_f16.len()
        ));
    }
    for g in 0..groups {
        for half in 0..2 {
            let d_bits = bits(d_z_f16, g * 4 + half * 2);
            let z_bits = bits(d_z_f16, g * 4 + half * 2 + 1);
            let d = f16_to_f32(d_bits);
            if d_bits & 0x8000 != 0 || !d.is_finite() {
                return Err(format!(
                    "{name}: group {g} half {half} scale 0x{d_bits:04x} is negative or non-finite"
                ));
            }
            let want = f32_to_f16(-8.0 * d);
            if z_bits != want {
                return Err(format!(
                    "{name}: group {g} half {half} is not symmetric: d=0x{d_bits:04x} \
                     z=0x{z_bits:04x}, expected z=f16(-8d)=0x{want:04x}"
                ));
            }
        }
    }
    if layout.is_down(name) {
        if k != layout.down_k_phys {
            return Err(format!(
                "{name}: down record K={k}, expected {}",
                layout.down_k_phys
            ));
        }
        for i in layout.inter..k {
            if bits(s_f16, i) != ONE_F16 {
                return Err(format!(
                    "{name}: S_f16[{i}] = 0x{:04x} but the padded tail must be exactly 1.0 (0x3c00)",
                    bits(s_f16, i)
                ));
            }
        }
    }
    Ok(())
}

/// Pack the frozen `(d, z)` grid and codes of one XTS record.
///
/// Zero-scale halves (`d == 0`, `z == 0x8000`) reconstruct 0 whatever their
/// codes. They are canonicalized to the mid code 8, exactly what
/// `quantize_symmetric_half` writes for B's degenerate halves, whatever code
/// (0 or 8) the record carried. The shared packer only accepts code 0 under a
/// zero scale, so those halves are packed as 0 and then patched to 0x88 bytes.
pub(crate) fn pack_final_codes(
    m: usize,
    k: usize,
    d_z_f16: &[u8],
    codes: &[u8],
) -> Result<Vec<u8>, String> {
    let groups = m * (k / 256);
    let zero_half = |g: usize, half: usize| {
        let i = (g * 4 + half * 2) * 2;
        d_z_f16[i] == 0 && d_z_f16[i + 1] == 0
    };
    let mut zeroed: Option<Vec<u8>> = None;
    for g in 0..groups {
        for half in 0..2 {
            if zero_half(g, half) {
                let r = g * 256 + half * 128..g * 256 + (half + 1) * 128;
                if codes[r.clone()].iter().any(|&c| c != 0) {
                    zeroed.get_or_insert_with(|| codes.to_vec())[r].fill(0);
                }
            }
        }
    }
    let mut packed = crate::quant_fwht::pack_mq4g256v2_from_f16_grid(
        m,
        k,
        d_z_f16,
        zeroed.as_deref().unwrap_or(codes),
    )?;
    for g in 0..groups {
        for half in 0..2 {
            if zero_half(g, half) {
                let off = g * MQ4V2_GROUP_BYTES + 8 + half * 64;
                packed[off..off + 64].fill(0x88);
            }
        }
    }
    Ok(packed)
}

/// SHA-256 over `name \t sha256(record file) \n` lines in name order: one
/// hash for the whole record set, independent of directory listing order.
pub(crate) fn record_set_sha256(files: &[(String, PathBuf)]) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let sorted: BTreeMap<&str, &PathBuf> = files.iter().map(|(n, p)| (n.as_str(), p)).collect();
    let mut hasher = Sha256::new();
    for (name, path) in sorted {
        let sha = crate::calibration::sha256_file_hex(path)?;
        hasher.update(format!("{name}\t{sha}\n").as_bytes());
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// B's metadata with the manifest advanced to the H2 artifact.
pub(crate) fn import_metadata(
    source_metadata_json: &str,
    source_sha: &str,
    record_set_sha: &str,
    n_records: usize,
) -> Result<String, String> {
    let mut meta: serde_json::Value = serde_json::from_str(source_metadata_json)
        .map_err(|e| format!("input HFQ metadata is invalid JSON: {e}"))?;
    let manifest = meta
        .get_mut(MANIFEST_KEY)
        .and_then(|m| m.as_object_mut())
        .ok_or_else(|| format!("input HFQ metadata lacks object `{MANIFEST_KEY}`"))?;
    manifest.insert("calibration".to_string(), CALIBRATION_H2.into());
    manifest.insert("source_sha256".to_string(), source_sha.into());
    manifest.insert("record_set_sha256".to_string(), record_set_sha.into());
    manifest.insert("records".to_string(), n_records.into());
    serde_json::to_string(&meta).map_err(|e| format!("serialize metadata: {e}"))
}

/// Rewrite the header/metadata/index region `[0, data_offset)` of a copy of the
/// source HFQ with `new_metadata_json`, keeping the tensor index bytes and the
/// data offset (the 4096 alignment slack absorbs the longer metadata). Fails
/// when the new front matter does not fit before the data section.
pub(crate) fn rewrite_front_matter(
    file: &mut File,
    source: &RuntimeHfqFile,
    new_metadata_json: &str,
) -> Result<(), String> {
    let data_offset = source.data_offset() as usize;
    let old_meta_len = source.metadata_json.len();
    let index_len: usize = 4 + source
        .tensors()
        .iter()
        .map(|t| 2 + t.name.len() + 1 + 1 + 4 * t.shape.len() + 4 + 8)
        .sum::<usize>();
    let new_end = 32 + new_metadata_json.len() + index_len;
    if new_end > data_offset {
        return Err(format!(
            "updated metadata ({} B) does not fit before the data section at {data_offset}",
            new_metadata_json.len()
        ));
    }
    let mut front = vec![0u8; data_offset];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek: {e}"))?;
    file.read_exact(&mut front)
        .map_err(|e| format!("read front matter: {e}"))?;
    if &front[32..32 + old_meta_len] != source.metadata_json.as_bytes() {
        return Err(
            "front matter does not carry the source metadata JSON at offset 32".to_string(),
        );
    }
    let n = u32::from_le_bytes(
        front[32 + old_meta_len..36 + old_meta_len]
            .try_into()
            .unwrap(),
    );
    if n as usize != source.tensors().len() {
        return Err("tensor index does not start where the metadata JSON ends".to_string());
    }
    let index = front[32 + old_meta_len..32 + old_meta_len + index_len].to_vec();
    let mut out = Vec::with_capacity(data_offset);
    out.extend_from_slice(&front[..32]);
    out.extend_from_slice(new_metadata_json.as_bytes());
    out.extend_from_slice(&index);
    out.resize(data_offset, 0);
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.write_all(&out))
        .map_err(|e| format!("write front matter: {e}"))
}

/// Which Inkling-2 recipes this invocation selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Recipe {
    /// `--tier xt --mq4v2-symmetric --format mq4v2`: the XTS layout (B).
    pub(crate) xts: bool,
    /// Routers kept F16 from the source (implied by `xts`, or `--inkling2-f16-router`).
    pub(crate) f16_router: bool,
}

/// Resolve and validate the arch-24 recipe. Other architectures and arch 24
/// without `--tier xt --mq4v2-symmetric` keep their legacy output; every flag the
/// XTS writer would silently ignore is refused instead.
pub(crate) fn resolve_recipe(
    args: &crate::cli::QuantizeArgs,
    arch_id: u32,
    tier_is_xt: bool,
) -> Result<Recipe, String> {
    if arch_id != ARCH_ID {
        if args.inkling2_f16_router {
            return Err(
                "--inkling2-f16-router only applies to Inkling-2-Mini (--arch-id 24)".into(),
            );
        }
        return Ok(Recipe {
            xts: false,
            f16_router: false,
        });
    }
    let xts = tier_is_xt && args.mq4v2_symmetric;
    if xts {
        if args.format != "mq4v2" {
            return Err(format!(
                "the Inkling-2 XTS layout (--tier xt --mq4v2-symmetric) requires --format mq4v2, got {}",
                args.format
            ));
        }
        let refused: [(&str, bool); 15] = [
            (
                "--inkling2-f16-router (already implied by the XTS layout)",
                args.inkling2_f16_router,
            ),
            ("--awq", args.awq),
            ("--awq-alpha", args.awq_alpha.is_some()),
            ("--imatrix", args.imatrix.is_some()),
            ("--hessian-dir", args.hessian_dir.is_some()),
            ("--awq-a4-aware", args.awq_a4_aware),
            ("--awq-a4-route-c2", args.awq_a4_route_c2),
            (
                "--awq-a4-signed-capture",
                args.awq_a4_signed_capture.is_some(),
            ),
            ("--fixed-tier", args.fixed_tier.is_some()),
            (
                "--reap-overlay/--reap-bake",
                args.reap_overlay.is_some() || args.reap_bake.is_some(),
            ),
            ("--include-vision", args.include_vision),
            ("--vision-only", args.vision_only),
            ("--include-prefix", args.include_prefix.is_some()),
            ("--mq3v2-symmetric", args.mq3v2_symmetric),
            ("--moe-expert-down", args.moe_expert_down.is_some()),
        ];
        if let Some((flag, _)) = refused.iter().find(|(_, set)| *set) {
            return Err(format!(
                "{flag} cannot be combined with the Inkling-2 XTS layout (qt44-padded-awq-v1)"
            ));
        }
    }
    Ok(Recipe {
        xts,
        f16_router: xts || args.inkling2_f16_router,
    })
}

/// Bytes of one HFQ tensor-index record.
fn index_record_len(name: &str, ndims: usize) -> usize {
    2 + name.len() + 1 + 1 + 4 * ndims + 4 + 8
}

/// Largest metadata growth the import (`h2-v1` manifest: two 64-hex hashes, a
/// count and key names) may need.
const IMPORT_METADATA_GROWTH: usize = 512;

/// Guarantee the 4096-aligned slack before the data section absorbs the
/// import's longer metadata. When the writer's natural slack is smaller, a
/// `hipfire_inkling2_front_slack` filler grows the front matter into the next
/// 4096 page (cost: at most one page), so `rewrite_front_matter` can never run
/// out of room.
pub(crate) fn reserve_front_slack(
    metadata_json: &str,
    tensors: &[HfqTensor],
) -> Result<String, String> {
    let index_len: usize = 4 + tensors
        .iter()
        .map(|t| index_record_len(&t.name, t.shape.len()))
        .sum::<usize>();
    let slack = |meta_len: usize| {
        let used = 32 + meta_len + index_len;
        ((used + 4095) & !4095) - used
    };
    if slack(metadata_json.len()) >= IMPORT_METADATA_GROWTH {
        return Ok(metadata_json.to_string());
    }
    let mut meta: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("metadata is not valid JSON: {e}"))?;
    let obj = meta
        .as_object_mut()
        .ok_or("metadata must be a top-level object")?;
    obj.insert("hipfire_inkling2_front_slack".to_string(), "".into());
    let base = serde_json::to_string(&meta).map_err(|e| format!("serialize metadata: {e}"))?;
    // Grow to the next page boundary plus the required slack.
    let used = 32 + base.len() + index_len;
    let target = ((used + 4095) & !4095) + 4096 - IMPORT_METADATA_GROWTH;
    let filler = target - used;
    let obj = meta.as_object_mut().expect("object");
    obj.insert(
        "hipfire_inkling2_front_slack".to_string(),
        " ".repeat(filler).into(),
    );
    let out = serde_json::to_string(&meta).map_err(|e| format!("serialize metadata: {e}"))?;
    debug_assert!(slack(out.len()) >= IMPORT_METADATA_GROWTH);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hfq::write_hfq;
    use crate::quant_fwht::cpu_fwht_256;
    use clap::{CommandFactory, FromArgMatches};
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    use std::path::Path;

    fn tiny() -> Layout {
        Layout {
            layers: 1,
            experts: 2,
            hidden: 256,
            inter: 128,
            down_k_phys: 256,
            vocab: 256,
            attn: [
                ("wq_du", 8, 256),
                ("wk_dv", 4, 256),
                ("wv_dv", 4, 256),
                ("wr_du", 4, 256),
                ("gate_attn", 8, 256),
                ("wo_ud", 8, 512),
            ],
        }
    }

    fn rand_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn header_bits(group: &[u8]) -> [u16; 4] {
        let b = |i: usize| u16::from_le_bytes([group[2 * i], group[2 * i + 1]]);
        [b(0), b(1), b(2), b(3)]
    }

    fn unpack_codes(group: &[u8]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(256);
        for b in &group[8..MQ4V2_GROUP_BYTES] {
            codes.push(b & 0xF);
            codes.push(b >> 4);
        }
        codes
    }

    // ── padding ─────────────────────────────────────────────────────────────

    #[test]
    fn inkling2_xts_padding_zero_tail_before_fwht() {
        let (rows, k, phys) = (3usize, 128usize, 256usize);
        let data = rand_f32(rows * k, 7);
        let got = quantize_padded_rows(&data, rows, k, phys, true);
        assert_eq!(got.len(), rows * MQ4V2_GROUP_BYTES);

        // Reference: pad the F32 row with zeros, then the stock symmetric encoder.
        let mut padded = vec![0.0f32; rows * phys];
        for r in 0..rows {
            padded[r * phys..r * phys + k].copy_from_slice(&data[r * k..(r + 1) * k]);
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let want = quantize_mq4g256v2_symmetric(&padded, rows, phys, &s1, &s2);
        assert_eq!(got, want);

        // The last (only) group of a padded row mixes 128 real and 128 zero
        // coordinates: after the FWHT the second half is not zero and was
        // quantized like any other half (nonzero scale), not discarded.
        for r in 0..rows {
            let mut g = [0.0f32; 256];
            g.copy_from_slice(&padded[r * phys..(r + 1) * phys]);
            assert!(g[128..].iter().all(|&v| v == 0.0));
            cpu_fwht_256(&mut g, &s1, &s2);
            assert!(g[128..].iter().any(|&v| v != 0.0), "FWHT must mix the tail");
            let h = header_bits(&got[r * MQ4V2_GROUP_BYTES..(r + 1) * MQ4V2_GROUP_BYTES]);
            assert_ne!(h[0], 0, "real half scale");
            assert_ne!(h[2], 0, "mixed zero-tail half keeps a nonzero scale");
        }
    }

    #[test]
    fn inkling2_xts_expert_down_is_padded_and_w13_regrouped() {
        let l = tiny();
        let (rows, cols) = (l.hidden, l.inter);
        let all = rand_f32(l.experts * rows * cols, 11);
        let t = encode_experts(
            &l,
            "model.llm.layers.0.mlp",
            &all,
            l.experts,
            rows,
            cols,
            false,
        )
        .unwrap();
        assert_eq!(t.len(), 2 * l.experts);
        assert_eq!(
            t[0].name,
            "model.llm.layers.0.mlp.experts.0.down_proj.weight"
        );
        assert_eq!(t[0].shape, vec![l.hidden as u32, l.down_k_phys as u32]);
        assert_eq!(
            t[0].data.len(),
            l.hidden * (l.down_k_phys / 256) * MQ4V2_GROUP_BYTES
        );
        assert_eq!(
            t[1].name,
            "model.llm.layers.0.mlp.experts.0.down_proj.awq_scale.weight"
        );
        assert_eq!(t[1].shape, vec![l.down_k_phys as u32]);
        assert_eq!(t[1].data, unit_sidecar_f16(l.down_k_phys));

        // w13: converter order [gate0, up0, gate1, up1, ...] -> [gate..., up...].
        let (rows13, cols13) = (2 * l.inter, l.hidden);
        let all13 = rand_f32(l.experts * rows13 * cols13, 13);
        let t13 = encode_experts(
            &l,
            "model.llm.layers.0.mlp",
            &all13,
            l.experts,
            rows13,
            cols13,
            true,
        )
        .unwrap();
        let mut grouped = vec![0.0f32; rows13 * cols13];
        for j in 0..l.inter {
            grouped[j * cols13..(j + 1) * cols13]
                .copy_from_slice(&all13[(2 * j) * cols13..(2 * j + 1) * cols13]);
            grouped[(l.inter + j) * cols13..(l.inter + j + 1) * cols13]
                .copy_from_slice(&all13[(2 * j + 1) * cols13..(2 * j + 2) * cols13]);
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        assert_eq!(
            t13[0].data,
            quantize_mq4g256v2_symmetric(&grouped, rows13, cols13, &s1, &s2)
        );
        assert_eq!(t13[0].shape, vec![rows13 as u32, cols13 as u32]);
        assert_eq!(t13[1].shape, vec![cols13 as u32]);
    }

    // ── symmetric invariant, embed passthrough ──────────────────────────────

    #[test]
    fn inkling2_xts_symmetric_header_invariant() {
        let data = rand_f32(4 * 512, 3);
        let q = quantize_padded_rows(&data, 4, 512, 512, true);
        let halves = crate::raon::verify_symmetric_headers("t", &q).unwrap();
        assert_eq!(halves, 4 * 2 * 2);
        for g in q.chunks_exact(MQ4V2_GROUP_BYTES) {
            let h = header_bits(g);
            assert_eq!(h[0] & 0x8000, 0, "d >= +0");
            assert_eq!(h[2] & 0x8000, 0, "d >= +0");
            assert_eq!(h[1], f32_to_f16(-8.0 * f16_to_f32(h[0])));
            assert_eq!(h[3], f32_to_f16(-8.0 * f16_to_f32(h[2])));
        }
        // All-zero halves: d = 0, z = 0x8000 (-0.0), code 8.
        let z = quantize_padded_rows(&vec![0.0; 2 * 256], 2, 256, 256, true);
        for g in z.chunks_exact(MQ4V2_GROUP_BYTES) {
            let h = header_bits(g);
            assert_eq!([h[0], h[1], h[2], h[3]], [0, 0x8000, 0, 0x8000]);
            assert!(unpack_codes(g).iter().all(|&c| c == 8));
        }
        crate::raon::verify_symmetric_headers("z", &z).unwrap();
    }

    #[test]
    fn inkling2_xts_embed_is_the_affine_encoder_not_symmetric() {
        let l = tiny();
        let data = rand_f32(l.vocab * l.hidden, 5);
        let e = encode_embed("model.llm.embed.weight", &data, l.vocab, l.hidden);
        assert_eq!(e.len(), 1, "embed carries no sidecar");
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        assert_eq!(
            e[0].data,
            quantize_mq4g256v2(&data, l.vocab, l.hidden, &s1, &s2)
        );
        assert_ne!(
            e[0].data,
            quantize_mq4g256v2_symmetric(&data, l.vocab, l.hidden, &s1, &s2)
        );
        assert!(crate::raon::verify_symmetric_headers("embed", &e[0].data).is_err());
        assert_eq!(e[0].quant_type as u8, 44);
        // Unembed is symmetric RTN, also without a sidecar.
        let u = encode_unembed("model.llm.unembed.weight", &data, l.vocab, l.hidden).unwrap();
        assert_eq!(u.len(), 1);
        assert_eq!(
            u[0].data,
            quantize_mq4g256v2_symmetric(&data, l.vocab, l.hidden, &s1, &s2)
        );
    }

    // ── router keep and recipe resolution ───────────────────────────────────

    #[test]
    fn inkling2_xts_router_predicate_is_exact() {
        assert!(is_router("model.llm.layers.0.mlp.gate.weight"));
        assert!(is_router("model.llm.layers.23.mlp.gate.weight"));
        assert!(!is_router("model.llm.layers.0.mlp.gate.bias"));
        assert!(!is_router("model.llm.layers.x.mlp.gate.weight"));
        assert!(!is_router(
            "model.llm.layers.0.mlp.experts.1.gate_up_proj.weight"
        ));
        assert!(!is_router("model.llm.layers.0.attn.gate_attn.weight"));
        assert!(!is_router("layers.0.mlp.gate.weight"));
        let l = Layout::mini();
        assert_eq!(
            attn_site_k(&l, "model.llm.layers.5.attn.wo_ud.weight"),
            Some(3072)
        );
        assert_eq!(
            attn_site_k(&l, "model.llm.layers.5.attn.wq_du.weight"),
            Some(2304)
        );
        assert_eq!(
            attn_site_k(&l, "model.llm.layers.5.attn.attn_sink_proj.weight"),
            None
        );
        assert_eq!(
            attn_site_k(&l, "model.llm.layers.5.attn.k_sconv.weight"),
            None
        );
    }

    fn recipe(extra: &[&str]) -> Result<Recipe, String> {
        let argv = ["hipfire-quantize", "--input", "i", "--output", "o"]
            .into_iter()
            .chain(extra.iter().copied());
        let matches = crate::cli::QuantizeArgs::command()
            .try_get_matches_from(argv)
            .unwrap();
        let args = crate::cli::QuantizeArgs::from_arg_matches(&matches).unwrap();
        let tier_xt = args.tier.as_deref() == Some("xt");
        resolve_recipe(&args, args.arch_id.unwrap_or(0), tier_xt)
    }

    #[test]
    fn inkling2_xts_recipe_selection_leaves_legacy_untouched() {
        let xts = [
            "--arch-id",
            "24",
            "--format",
            "mq4v2",
            "--tier",
            "xt",
            "--mq4v2-symmetric",
        ];
        assert_eq!(
            recipe(&xts).unwrap(),
            Recipe {
                xts: true,
                f16_router: true
            }
        );
        // Legacy arch-24 invocations: nothing changes.
        for legacy in [
            vec!["--arch-id", "24", "--format", "mq4v2"],
            vec!["--arch-id", "24", "--format", "mq4v2", "--mq4v2-symmetric"],
            vec!["--arch-id", "24", "--format", "mq4v2", "--tier", "xt"],
        ] {
            assert_eq!(
                recipe(&legacy).unwrap(),
                Recipe {
                    xts: false,
                    f16_router: false
                },
                "{legacy:?}"
            );
        }
        // A_R: explicit opt-in on the legacy path.
        assert_eq!(
            recipe(&[
                "--arch-id",
                "24",
                "--format",
                "mq4v2",
                "--inkling2-f16-router"
            ])
            .unwrap(),
            Recipe {
                xts: false,
                f16_router: true
            }
        );
        // Other architectures never see the recipe, and refuse the flag.
        assert_eq!(
            recipe(&[
                "--arch-id",
                "6",
                "--format",
                "mq4v2",
                "--tier",
                "xt",
                "--mq4v2-symmetric"
            ])
            .unwrap(),
            Recipe {
                xts: false,
                f16_router: false
            }
        );
        assert!(recipe(&["--arch-id", "6", "--inkling2-f16-router"]).is_err());
        // Ignored-flag combinations are refused, not silently dropped.
        for extra in [
            ["--awq"].as_slice(),
            &["--include-vision"],
            &["--fixed-tier", "embed:q8"],
            &["--inkling2-f16-router"],
        ] {
            let mut v = xts.to_vec();
            v.extend_from_slice(extra);
            assert!(recipe(&v).is_err(), "{extra:?}");
        }
        let mut v = xts.to_vec();
        v[3] = "mq4";
        assert!(recipe(&v).unwrap_err().contains("--format mq4v2"));
    }

    // ── census ──────────────────────────────────────────────────────────────

    /// A complete tiny XTS tensor list: calibrated qt44 + unit sidecars,
    /// F16 routers, affine embed, symmetric unembed, one extra F16 tensor.
    /// `embed` goes last so the unchanged runtime reader's symmetric sampler
    /// (first four qt44 tensors) sees symmetric headers.
    fn tiny_tensors(l: &Layout) -> Vec<HfqTensor> {
        let mut t = Vec::new();
        for layer in 0..l.layers {
            let base = format!("model.llm.layers.{layer}.mlp");
            for (i, &(leaf, m, k)) in l.attn.iter().enumerate() {
                let name = l.attn_name(layer, leaf);
                t.extend(encode_attn(&name, &rand_f32(m * k, 100 + i as u32), m, k).unwrap());
            }
            let (r13, c13) = (2 * l.inter, l.hidden);
            t.extend(
                encode_experts(
                    l,
                    &base,
                    &rand_f32(l.experts * r13 * c13, 21),
                    l.experts,
                    r13,
                    c13,
                    true,
                )
                .unwrap(),
            );
            let (r2, c2) = (l.hidden, l.inter);
            t.extend(
                encode_experts(
                    l,
                    &base,
                    &rand_f32(l.experts * r2 * c2, 22),
                    l.experts,
                    r2,
                    c2,
                    false,
                )
                .unwrap(),
            );
            t.push(HfqTensor {
                name: l.router_name(layer),
                quant_type: QuantType::F16,
                shape: vec![l.experts as u32, l.hidden as u32],
                group_size: 0,
                data: vec![0x11; l.experts * l.hidden * 2],
                spilled_len: 0,
            });
        }
        t.push(HfqTensor {
            name: "model.llm.norm.weight".into(),
            quant_type: QuantType::F16,
            shape: vec![l.hidden as u32],
            group_size: 0,
            data: vec![0x22; l.hidden * 2],
            spilled_len: 0,
        });
        t.extend(
            encode_unembed(
                "model.llm.unembed.weight",
                &rand_f32(l.vocab * l.hidden, 31),
                l.vocab,
                l.hidden,
            )
            .unwrap(),
        );
        t.extend(encode_embed(
            "model.llm.embed.weight",
            &rand_f32(l.vocab * l.hidden, 32),
            l.vocab,
            l.hidden,
        ));
        t
    }

    fn check(l: &Layout, t: &[HfqTensor]) -> Result<(), String> {
        let entries: Vec<Entry<'_>> = t.iter().map(Entry::from).collect();
        check_template(l, &entries)
    }

    #[test]
    fn inkling2_xts_sidecar_census() {
        let l = tiny();
        let t = tiny_tensors(&l);
        let sidecars: Vec<&HfqTensor> = t
            .iter()
            .filter(|x| x.name.ends_with(".awq_scale.weight"))
            .collect();
        assert_eq!(sidecars.len(), l.sidecars());
        assert_eq!(l.sidecars(), 1 * (6 + 2 * 2));
        let qt44 = t.iter().filter(|x| x.quant_type as u8 == 44).count();
        assert_eq!(qt44, l.sidecars() + 2, "calibrated + embed + unembed");
        assert!(t.iter().all(|x| x.quant_type as u8 != 13), "no legacy qt13");
        check(&l, &t).unwrap();
        // Sidecars are all-ones F16 and each shared q-producer sidecar is equal.
        for s in &sidecars {
            assert!(
                s.data.chunks_exact(2).all(|c| c == [0x00, 0x3C]),
                "{}",
                s.name
            );
        }
        // The real geometry's counts (contract: 6,288 pairs, 13,078 tensors).
        let m = Layout::mini();
        assert_eq!(m.sidecars(), 6288);
        assert_eq!(m.calibrated().len(), 6288);
        assert_eq!(6144 + 144 + 2 + 6288 + 24 + 476, MINI_TOTAL_TENSORS);
    }

    #[test]
    fn inkling2_xts_census_rejects_deviations() {
        let l = tiny();
        let good = tiny_tensors(&l);
        let drop = |pat: &str| -> Vec<HfqTensor> {
            good.iter()
                .filter(|t| t.name != pat)
                .map(|t| HfqTensor {
                    name: t.name.clone(),
                    quant_type: t.quant_type,
                    shape: t.shape.clone(),
                    group_size: t.group_size,
                    data: Vec::new(),
                    spilled_len: 0,
                })
                .collect()
        };
        let e = check(&l, &drop("model.llm.layers.0.attn.wq_du.awq_scale.weight")).unwrap_err();
        assert!(e.contains("missing AWQ sidecar"), "{e}");
        let e = check(
            &l,
            &drop("model.llm.layers.0.mlp.experts.1.down_proj.weight"),
        )
        .unwrap_err();
        assert!(e.contains("missing calibrated tensor"), "{e}");
        let e = check(&l, &drop("model.llm.layers.0.mlp.gate.weight")).unwrap_err();
        assert!(e.contains("missing router"), "{e}");
        let e = check(&l, &drop("model.llm.embed.weight")).unwrap_err();
        assert!(e.contains("missing"), "{e}");

        let mut t = drop("");
        // router quantized (A's layout) instead of F16
        t.iter_mut()
            .find(|x| x.name.ends_with("mlp.gate.weight"))
            .unwrap()
            .quant_type = QuantType::MQ4G256V2;
        assert!(check(&l, &t).unwrap_err().contains("router"));
        // legacy qt13 expert
        let mut t = drop("");
        t.iter_mut()
            .find(|x| x.name.ends_with("experts.0.gate_up_proj.weight"))
            .unwrap()
            .quant_type = QuantType::MQ4G256;
        assert!(check(&l, &t).unwrap_err().contains("qt=13"));
        // down not padded
        let mut t = drop("");
        let d = t
            .iter_mut()
            .find(|x| x.name.ends_with("experts.0.down_proj.weight"))
            .unwrap();
        d.shape = vec![l.hidden as u32, l.inter as u32];
        assert!(check(&l, &t).unwrap_err().contains("down_proj"));
        // sidecar on the embed
        let mut t = drop("");
        t.push(HfqTensor {
            name: "model.llm.embed.awq_scale.weight".into(),
            quant_type: QuantType::F16,
            shape: vec![l.hidden as u32],
            group_size: 0,
            data: Vec::new(),
            spilled_len: 0,
        });
        assert!(check(&l, &t)
            .unwrap_err()
            .contains("must not carry a sidecar"));
        // stray qt44 tensor
        let mut t = drop("");
        t.push(HfqTensor {
            name: "model.llm.layers.0.attn.extra.weight".into(),
            quant_type: QuantType::MQ4G256V2,
            shape: vec![4, 256],
            group_size: 256,
            data: Vec::new(),
            spilled_len: 0,
        });
        assert!(check(&l, &t).unwrap_err().contains("unexpected qt44"));
        // orphan sidecar
        let mut t = drop("");
        t.push(HfqTensor {
            name: "model.llm.norm.awq_scale.weight".into(),
            quant_type: QuantType::F16,
            shape: vec![l.hidden as u32],
            group_size: 0,
            data: Vec::new(),
            spilled_len: 0,
        });
        assert!(check(&l, &t).unwrap_err().contains("orphan"));
    }

    #[test]
    fn inkling2_xts_metadata_stamp_and_slack() {
        let l = Layout::mini();
        let mut obj = serde_json::Map::new();
        stamp_metadata(&mut obj, &l);
        assert_eq!(obj[LAYOUT_KEY], "qt44-padded-awq-v1");
        let m = &obj[MANIFEST_KEY];
        assert_eq!(m["down_k_logical"], 1152);
        assert_eq!(m["down_k_physical"], 1280);
        assert_eq!(m["awq_sidecars"], 6288);
        assert_eq!(m["calibration"], "rtn-s1");

        // reserve_front_slack always leaves room for the import's metadata.
        let t = tiny_tensors(&tiny());
        for pad in 0..40usize {
            let meta = serde_json::json!({ "x": "a".repeat(60 * pad) }).to_string();
            let out = reserve_front_slack(&meta, &t).unwrap();
            let index_len: usize = 4 + t
                .iter()
                .map(|x| index_record_len(&x.name, x.shape.len()))
                .sum::<usize>();
            let used = 32 + out.len() + index_len;
            assert!(
                ((used + 4095) & !4095) - used >= IMPORT_METADATA_GROWTH,
                "pad {pad}"
            );
        }
    }

    // ── final-code import ───────────────────────────────────────────────────

    fn build_tiny_b(dir: &Path, l: &Layout) -> PathBuf {
        let t = tiny_tensors(l);
        let mut obj = serde_json::Map::new();
        obj.insert("architecture".into(), "inkling2_loop".into());
        obj.insert("hipfire_product_tier".into(), "xt".into());
        obj.insert("hipfire_base_format".into(), "mq4v2".into());
        obj.insert("mq4v2.symmetric".into(), 1.into());
        stamp_metadata(&mut obj, l);
        let meta = reserve_front_slack(&serde_json::Value::Object(obj).to_string(), &t).unwrap();
        let path = dir.join("b.hfq");
        write_hfq(&path, ARCH_ID, &meta, &t, None).unwrap();
        path
    }

    struct Rec {
        name: String,
        m: usize,
        k: usize,
        qt: String,
        sha: String,
        s: Vec<u8>,
        grid: Vec<u8>,
        codes: Vec<u8>,
    }

    /// One valid record per calibrated tensor, derived from B's own grid and
    /// codes (one code bumped so the import visibly changes bytes) with
    /// non-unit S (down tail 1.0; q-producer siblings share S).
    fn valid_records(b: &RuntimeHfqFile, l: &Layout, sha: &str) -> Vec<Rec> {
        let mut out = Vec::new();
        for (name, m, k) in l.calibrated() {
            let (_, data) = b.tensor_data(&name).unwrap();
            let mut grid = Vec::new();
            let mut codes = Vec::new();
            for g in data.chunks_exact(MQ4V2_GROUP_BYTES) {
                grid.extend_from_slice(&g[..8]);
                codes.extend_from_slice(&unpack_codes(g));
            }
            codes[3] = (codes[3] + 5) % 16;
            let live = if l.is_down(&name) { l.inter } else { k };
            let key = shared_scale_group(&name).unwrap_or_else(|| name.clone());
            let seed = key
                .bytes()
                .fold(0u32, |a, c| a.wrapping_mul(31).wrapping_add(c as u32));
            let s: Vec<u8> = (0..k)
                .flat_map(|i| {
                    let v = if i < live {
                        0.5 + ((i as u32 + seed) % 7) as f32 * 0.25
                    } else {
                        1.0
                    };
                    f32_to_f16(v).to_le_bytes()
                })
                .collect();
            out.push(Rec {
                name,
                m,
                k,
                qt: "44".into(),
                sha: sha.into(),
                s,
                grid,
                codes,
            });
        }
        out
    }

    fn write_records(dir: &Path, recs: &[Rec]) {
        std::fs::create_dir_all(dir).unwrap();
        for (i, r) in recs.iter().enumerate() {
            let tensors = [
                (
                    "S_f16",
                    TensorView::new(Dtype::F16, vec![r.k], &r.s).unwrap(),
                ),
                (
                    "d_z_f16",
                    TensorView::new(Dtype::F16, vec![r.m, r.k / 256, 2, 2], &r.grid).unwrap(),
                ),
                (
                    "codes_u8",
                    TensorView::new(Dtype::U8, vec![r.m, r.k], &r.codes).unwrap(),
                ),
            ];
            let metadata = std::collections::HashMap::from([
                ("name".to_string(), r.name.clone()),
                ("M".into(), r.m.to_string()),
                ("K".into(), r.k.to_string()),
                ("qt".into(), r.qt.clone()),
                ("source_sha".into(), r.sha.clone()),
            ]);
            serialize_to_file(
                tensors,
                Some(metadata),
                &dir.join(format!("r{i:03}.safetensors")),
            )
            .unwrap();
        }
    }

    fn import(b: &Path, dir: &Path, out: &Path, l: &Layout) -> Result<(usize, String), String> {
        crate::pipeline::export_mq4v2_final_codes_with(b, dir, out, l)
    }

    /// Build B, apply `mutate` to the valid records, import; returns the error text.
    fn reject(mutate: impl FnOnce(&mut Vec<Rec>)) -> String {
        let l = tiny();
        let dir = tempfile::tempdir().unwrap();
        let b = build_tiny_b(dir.path(), &l);
        let sha = crate::calibration::sha256_file_hex(&b).unwrap();
        let bf = RuntimeHfqFile::open_with_reap_plan(&b, None).unwrap();
        let mut recs = valid_records(&bf, &l, &sha);
        mutate(&mut recs);
        write_records(&dir.path().join("recs"), &recs);
        let out = dir.path().join("c.hfq");
        let e = import(&b, &dir.path().join("recs"), &out, &l).unwrap_err();
        assert!(!out.exists(), "a refused import must not leave an output");
        e
    }

    #[test]
    fn inkling2_xts_import_accepts_exact_record_set() {
        let l = tiny();
        let dir = tempfile::tempdir().unwrap();
        let b = build_tiny_b(dir.path(), &l);
        let sha = crate::calibration::sha256_file_hex(&b).unwrap();
        let bf = RuntimeHfqFile::open_with_reap_plan(&b, None).unwrap();
        let mut recs = valid_records(&bf, &l, &sha);
        // A zero-scale half carrying the mid code (what B's encoder emits) and
        // one carrying code 0 both canonicalize to d=0, z=0x8000, code 8.
        for (idx, fill) in [(0usize, 8u8), (1usize, 0u8)] {
            let r = &mut recs[idx];
            r.grid[0..4].copy_from_slice(&[0, 0, 0x00, 0x80]);
            r.codes[..128].fill(fill);
        }
        write_records(&dir.path().join("recs"), &recs);
        let c = dir.path().join("c.hfq");
        let (n, src) = import(&b, &dir.path().join("recs"), &c, &l).unwrap();
        assert_eq!(n, l.sidecars());
        assert_eq!(src, sha);

        let cf = RuntimeHfqFile::open_with_reap_plan(&c, None).unwrap();
        // Same tensor index, header and data offset; only the manifest advanced.
        assert_eq!(bf.data_offset(), cf.data_offset());
        assert_eq!(bf.tensors().len(), cf.tensors().len());
        for (x, y) in bf.tensors().iter().zip(cf.tensors()) {
            assert_eq!(
                (
                    &x.name,
                    x.quant_type,
                    &x.shape,
                    x.group_size,
                    x.data_offset,
                    x.data_size
                ),
                (
                    &y.name,
                    y.quant_type,
                    &y.shape,
                    y.group_size,
                    y.data_offset,
                    y.data_size
                )
            );
        }
        let mut mb: serde_json::Value = serde_json::from_str(&bf.metadata_json).unwrap();
        let mut mc: serde_json::Value = serde_json::from_str(&cf.metadata_json).unwrap();
        let man = mc[MANIFEST_KEY].take();
        assert_eq!(man["calibration"], "h2-v1");
        assert_eq!(man["source_sha256"], sha.as_str());
        assert_eq!(man["records"], l.sidecars());
        assert_eq!(man["record_set_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(man["awq_sidecars"], l.sidecars());
        let man_b = mb[MANIFEST_KEY].take();
        assert_eq!(man_b["calibration"], "rtn-s1");
        assert_eq!(mb, mc, "all other metadata keys are untouched");

        let calibrated: BTreeSet<String> = l.calibrated().into_iter().map(|(n, _, _)| n).collect();
        let by_rec: HashMap<&str, &Rec> = recs.iter().map(|r| (r.name.as_str(), r)).collect();
        for t in bf.tensors() {
            let (_, before) = bf.tensor_data(&t.name).unwrap();
            let (_, after) = cf.tensor_data(&t.name).unwrap();
            if let Some(stem) = t.name.strip_suffix(".awq_scale.weight") {
                let r = by_rec[format!("{stem}.weight").as_str()];
                assert_eq!(after, r.s.as_slice(), "{} sidecar", t.name);
            } else if calibrated.contains(&t.name) {
                let r = by_rec[t.name.as_str()];
                assert_eq!(
                    after,
                    pack_final_codes(r.m, r.k, &r.grid, &r.codes).unwrap(),
                    "{}",
                    t.name
                );
                assert_ne!(after, before, "{} must change", t.name);
            } else {
                assert_eq!(after, before, "{} must stay byte-identical", t.name);
            }
        }
        // Zero-scale half: header (0, 0x8000) and mid-code bytes for both records.
        for idx in [0usize, 1] {
            let (_, d) = cf.tensor_data(&recs[idx].name).unwrap();
            assert_eq!(&d[..4], &[0, 0, 0x00, 0x80]);
            assert!(
                d[8..8 + 64].iter().all(|&b| b == 0x88),
                "mid code 8 in the zero half"
            );
        }
        // The standing runtime sampler accepts the result's symmetric headers.
        assert!(cf.mq4v2_symmetric());
    }

    #[test]
    fn inkling2_xts_import_refuses_bad_record_sets() {
        // missing record
        let e = reject(|r| {
            r.pop();
        });
        assert!(e.contains("no final-code record"), "{e}");
        // extra record naming an uncalibrated tensor (embed has no sidecar)
        let e = reject(|r| {
            let extra = Rec {
                name: "model.llm.embed.weight".into(),
                ..clone_rec(&r[0])
            };
            r.push(extra);
        });
        assert!(e.contains("do not name a calibrated"), "{e}");
        // duplicate record
        let e = reject(|r| {
            let dup = clone_rec(&r[0]);
            r.push(dup);
        });
        assert!(e.contains("duplicate"), "{e}");
        // wrong source sha
        let e = reject(|r| r[2].sha = "0".repeat(64));
        assert!(e.contains("does not match input artifact"), "{e}");
        // wrong qt
        let e = reject(|r| r[2].qt = "13".into());
        assert!(e.contains("qt=13"), "{e}");
    }

    fn clone_rec(r: &Rec) -> Rec {
        Rec {
            name: r.name.clone(),
            m: r.m,
            k: r.k,
            qt: r.qt.clone(),
            sha: r.sha.clone(),
            s: r.s.clone(),
            grid: r.grid.clone(),
            codes: r.codes.clone(),
        }
    }

    #[test]
    fn inkling2_xts_import_refuses_malformed_records() {
        // wrong M (and codes sized to match so the record itself is well formed)
        let e = reject(|r| {
            let x = &mut r[0];
            x.m += 1;
            let extra_grid = x.grid[..(x.k / 256) * 8].to_vec();
            x.grid.extend_from_slice(&extra_grid);
            let extra = x.codes[..x.k].to_vec();
            x.codes.extend_from_slice(&extra);
        });
        assert!(e.contains("source type-map entry"), "{e}");
        // wrong K for a down tensor (unpadded 128): sidecar/shape mismatch
        let e = reject(|r| {
            let i = r
                .iter()
                .position(|x| x.name.ends_with("down_proj.weight"))
                .unwrap();
            let x = &mut r[i];
            x.k = 512; // 256-aligned but not the physical down K
            x.s = unit_sidecar_f16(x.k);
            x.grid = vec![0; x.m * (x.k / 256) * 8];
            x.codes = vec![0; x.m * x.k];
            for g in x.grid.chunks_exact_mut(8) {
                g[2..4].copy_from_slice(&0x8000u16.to_le_bytes());
                g[6..8].copy_from_slice(&0x8000u16.to_le_bytes());
            }
        });
        assert!(e.contains("down record K"), "{e}");
        // non-symmetric z
        let e = reject(|r| r[0].grid[2..4].copy_from_slice(&0x1234u16.to_le_bytes()));
        assert!(e.contains("not symmetric"), "{e}");
        // negative d (with z = f16(-8d) so only the sign is wrong)
        let e = reject(|r| {
            let d = f32_to_f16(-0.25);
            r[0].grid[0..2].copy_from_slice(&d.to_le_bytes());
            r[0].grid[2..4].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        });
        assert!(e.contains("negative"), "{e}");
        // d == 0 must carry z = 0x8000, not +0
        let e = reject(|r| {
            r[0].grid[0..4].copy_from_slice(&[0, 0, 0, 0]);
            r[0].codes[..128].fill(0);
        });
        assert!(e.contains("not symmetric"), "{e}");
        // non-positive and non-finite S
        let e = reject(|r| r[0].s[0..2].copy_from_slice(&0u16.to_le_bytes()));
        assert!(e.contains("finite and positive"), "{e}");
        let e = reject(|r| r[0].s[0..2].copy_from_slice(&0x7C00u16.to_le_bytes()));
        assert!(e.contains("finite and positive"), "{e}");
        // down S tail must be exactly 1.0
        let e = reject(|r| {
            let i = r
                .iter()
                .position(|x| x.name.ends_with("down_proj.weight"))
                .unwrap();
            let n = r[i].s.len();
            r[i].s[n - 2..].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
        });
        assert!(e.contains("padded tail"), "{e}");
        // wq_du / wr_du / gate_attn sidecars must be identical
        let e = reject(|r| {
            let i = r
                .iter()
                .position(|x| x.name.ends_with("attn.wr_du.weight"))
                .unwrap();
            r[i].s[0..2].copy_from_slice(&f32_to_f16(3.0).to_le_bytes());
        });
        assert!(e.contains("shared producer group"), "{e}");
    }

    #[test]
    fn inkling2_xts_import_requires_the_rtn_control_template() {
        let l = tiny();
        let dir = tempfile::tempdir().unwrap();
        let b = build_tiny_b(dir.path(), &l);
        let bf = RuntimeHfqFile::open_with_reap_plan(&b, None).unwrap();
        let tensors = bf.tensors();
        validate_template(&l, &bf.metadata_json, tensors).unwrap();
        for (from, to, needle) in [
            ("qt44-padded-awq-v1", "qt44-other", LAYOUT_KEY),
            ("\"rtn-s1\"", "\"h2-v1\"", "rtn-s1"),
            ("\"xt\"", "\"base\"", "hipfire_product_tier"),
            (
                "\"mq4v2.symmetric\":1",
                "\"mq4v2.symmetric\":0",
                "mq4v2.symmetric",
            ),
            (
                "\"down_k_physical\":256",
                "\"down_k_physical\":512",
                "geometry",
            ),
        ] {
            let meta = bf.metadata_json.replace(from, to);
            assert_ne!(meta, bf.metadata_json, "{from}");
            let e = validate_template(&l, &meta, &tensors).unwrap_err();
            assert!(e.contains(needle), "{from}: {e}");
        }
    }
}
